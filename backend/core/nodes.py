import asyncio
from typing import Any, Dict, List, Literal, Optional
import tiktoken
import numpy as np
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError
from backend.core.prompts import (
    SYNTHESIS_PROMPT,
    LEAD_RESEARCHER_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
)
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from backend.pipeline.vector import generate_embeddings
from backend.core.state import AgentState, AgentAction, SubAgentInput, Summary, TodoItem
from backend.shared.constants import RESEARCH_LLM_REASONING, RESEARCH_LLM_FAST
from backend.shared.logger import get_logger
from backend.core.tools.rag import search_specific_document_for_research

logger = get_logger("RECURSIVE_SUMMARIZER")

_ENCODER: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    """Lazily initialise and return a shared tiktoken encoder.

    Uses cl100k_base, which is compatible with GPT-4/4.1/4o/5-style models
    and already used elsewhere in this project for accurate token counting.
    """
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def _estimate_tokens(text: str) -> int:
    """Estimate token count using the cl100k_base tokenizer."""
    if not text:
        return 0
    return len(_get_encoder().encode(text))


def _group_by_tokens(
    texts: List[str], children: np.ndarray, target_tokens: int
) -> List[List[str]]:
    """Group texts into batches using the agglomerative clustering tree.

    Traverses the merge history (children_) to identify the largest possible
    clusters that satisfy the target_tokens constraint.

    Args:
        texts: List of original text chunks.
        children: (N-1, 2) array of merge operations from AgglomerativeClustering.
        target_tokens: Max token budget per batch.

    Returns:
        List of batches (each batch is a list of strings).
    """
    n_samples = len(texts)

    node_tokens = {i: _estimate_tokens(texts[i]) for i in range(n_samples)}

    node_indices = {i: [i] for i in range(n_samples)}

    valid_roots = set(range(n_samples))

    valid_nodes = set(range(n_samples))

    for idx, (c1, c2) in enumerate(children):
        new_node = n_samples + idx
        c1, c2 = int(c1), int(c2)

        is_merge_possible = (c1 in valid_nodes) and (c2 in valid_nodes)

        if is_merge_possible:
            combined_tokens = node_tokens[c1] + node_tokens[c2]

            if combined_tokens <= target_tokens:
                valid_nodes.add(new_node)
                node_tokens[new_node] = combined_tokens
                node_indices[new_node] = node_indices[c1] + node_indices[c2]

                if c1 in valid_roots:
                    valid_roots.remove(c1)
                if c2 in valid_roots:
                    valid_roots.remove(c2)
                valid_roots.add(new_node)

    batches = []
    for root in valid_roots:
        indices = node_indices[root]
        batch_texts = [texts[i] for i in indices]
        batches.append(batch_texts)

    return batches


async def _summarize_batch_findings(
    findings_batch: List[str],
    root_query: str,
) -> str:
    """Summarize a batch of findings into a single merged summary."""
    batch_text = "\n\n---\n\n".join(findings_batch)
    prompt_content = SYNTHESIS_PROMPT.format(
        findings=batch_text,
        query=root_query,
    )

    response = await RESEARCH_LLM_REASONING.ainvoke(
        [HumanMessage(content=prompt_content)]
    )
    return getattr(response, "content", str(response))


async def recursive_summarize_findings(
    raw_findings: List[str],
    root_query: str,
    target_batch_tokens: int = 1200,
) -> str:
    """Hybrid Similarity-Ordered Recursive Summarization.

    Uses sklearn to perform agglomerative clustering on embeddings, then groups
    chunks into maximally-sized batches that respect the similarity hierarchy.
    """
    current_level: List[str] = list(raw_findings)
    iteration = 0

    while len(current_level) > 1:
        iteration += 1
        n = len(current_level)
        logger.info(
            f"🔄 Recursive summarization – iteration {iteration}, "
            f"{n} chunk(s) remaining"
        )

        embeddings = np.array(
            await generate_embeddings(current_level), dtype=np.float32
        )

        sim_matrix = cosine_similarity(embeddings)
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix[dist_matrix < 0] = 0

        clustering = AgglomerativeClustering(
            n_clusters=1, linkage="average", metric="precomputed"
        ).fit(dist_matrix)

        batches = _group_by_tokens(
            current_level, clustering.children_, target_batch_tokens
        )

        # If no reduction occurred, force a single batch to guarantee convergence
        if len(batches) >= n:
            batches = [current_level]

        logger.info(f"   📦 Formed {len(batches)} batch(es) for LLM synthesis")

        tasks = [
            _summarize_batch_findings(batch, root_query=root_query)
            for batch in batches
            if batch
        ]
        next_level = await asyncio.gather(*tasks)
        current_level = list(next_level)

    return current_level[0]


class WriteTodos(BaseModel):
    """Structured output schema for the orchestrator to define tasks for sub-agents."""

    sub_agent_todos: List[TodoItem] = Field(
        description="A list of specific tasks that EVERY sub-agent must execute for their assigned document. Each item must have a 'task' and a 'status' (default 'pending')."
    )


async def orchestrator_node(
    state: AgentState, config: RunnableConfig
) -> Dict[str, Any]:
    """
    Defines tasks for sub-agents based on the user query.

    Uses structured output (no tool binding) so the LLM always returns a
    well-formed WriteTodos object deterministically.

    Args:
        state: Current agent state.
        config: Runtime configuration.

    Returns:
        State updates including messages and sub_agent_todos.
    """
    messages = state["messages"]

    todo_writer = RESEARCH_LLM_REASONING.with_structured_output(
        WriteTodos,
        method="function_calling",
        include_raw=False,
    ).with_retry(
        stop_after_attempt=3,
        retry_if_exception_type=(ValueError, ValidationError, OutputParserException),
    )

    result = await todo_writer.ainvoke(
        [{"role": "system", "content": LEAD_RESEARCHER_PROMPT}] + messages
    )

    return {
        "messages": [
            AIMessage(
                content=f"Prepared {len(result.sub_agent_todos)} research task(s) for sub-agents."
            )
        ],
        "sub_agent_todos": result.sub_agent_todos,
    }


async def document_sub_agent_node(input_data: SubAgentInput) -> Dict[str, Any]:
    """
    Processes a document using an RLM-inspired iterative retrieval loop.

    The LLM fully controls iteration: it outputs a structured AgentAction each
    turn. The external loop executes the search and feeds results back, or exits
    when the LLM signals action="finalize". No tools are ever bound to the LLM.

    A high safety ceiling (SAFETY_LIMIT) exists only as an emergency fallback to
    prevent runaway costs — the LLM is expected to finalize well before it.

    Args:
        input_data: SubAgentInput containing the document_name and assigned todos.

    Returns:
        State update with the Summary added to global_context.
    """
    SAFETY_LIMIT = 50

    doc_name = input_data["document_name"]
    todos_list = input_data.get("todos", [])

    todos_str = (
        "\n".join(f"{i + 1}. {t.task}" for i, t in enumerate(todos_list))
        if todos_list
        else "No specific sub-tasks provided."
    )

    system_prompt = RESEARCH_SYSTEM_PROMPT.format(file_name=doc_name)
    messages: List = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"**Orchestrator Assigned Tasks**:\n{todos_str}\n\n"
                f"Begin your investigation. Issue SEARCH actions to retrieve "
                f"information, then FINALIZE once all tasks are covered."
            )
        ),
    ]

    action_extractor = RESEARCH_LLM_FAST.with_structured_output(
        AgentAction,
        method="function_calling",
        include_raw=False,
    ).with_retry(
        stop_after_attempt=3,
        retry_if_exception_type=(ValueError, ValidationError, OutputParserException),
    )

    iteration = 0
    while True:
        iteration += 1

        if iteration > SAFETY_LIMIT:
            logger.error(
                f"[{doc_name}] Safety ceiling of {SAFETY_LIMIT} iterations reached — "
                f"LLM never issued 'finalize'. Forcing extraction."
            )
            messages.append(
                HumanMessage(
                    content="You have used the maximum number of searches. "
                    "You MUST finalize your findings now."
                )
            )
            try:
                action = await action_extractor.ainvoke(messages)
                findings = action.findings or "Safety limit reached; partial findings only."
            except Exception as e:
                logger.error(f"[{doc_name}] Forced finalization failed: {e}")
                findings = "Extraction failed after safety limit."
            return {
                "global_context": [Summary(document_name=doc_name, findings=findings)]
            }

        try:
            action: AgentAction = await action_extractor.ainvoke(messages)
        except Exception as e:
            logger.error(
                f"[{doc_name}] AgentAction parsing failed (iter {iteration}): {e}. "
                f"Aborting loop."
            )
            return {
                "global_context": [
                    Summary(
                        document_name=doc_name,
                        findings="Action parsing failed; no findings extracted.",
                    )
                ]
            }

        logger.info(
            f"[{doc_name}] Iter {iteration} — "
            f"action={action.action!r} | reasoning={action.reasoning!r}"
        )

        if action.action == "finalize":
            if not action.findings:
                logger.warning(
                    f"[{doc_name}] LLM finalized with empty findings (iter {iteration})."
                )
            return {
                "global_context": [
                    Summary(
                        document_name=doc_name,
                        findings=action.findings or "No findings extracted.",
                    )
                ]
            }

        if not action.query:
            logger.warning(
                f"[{doc_name}] LLM issued 'search' with no query (iter {iteration})."
            )
            messages.append(
                HumanMessage(
                    content="Your last action was 'search' but the `query` field was "
                    "empty. Please provide a specific search string, or finalize if "
                    "you have gathered enough information."
                )
            )
            continue

        search_results = await search_specific_document_for_research.ainvoke(
            {"query": action.query, "file_name": doc_name}
        )
        logger.info(f"[{doc_name}] Search query: '{action.query}'")

        messages.append(
            AIMessage(
                content=f"[SEARCH] Reasoning: {action.reasoning}\nQuery: {action.query}"
            )
        )
        messages.append(
            HumanMessage(
                content=f"Search results for '{action.query}':\n\n{search_results}"
            )
        )


async def synthesis_node(
    state: AgentState, config: RunnableConfig
) -> Command[Literal[END]]:
    """
    Aggregates all research findings and generates the final report.

    Combines summaries from global_context into a comprehensive response.

    Args:
        state: Current agent state with global_context populated.
        config: Runtime configuration.

    Returns:
        Command routing to END with final report in messages.
    """
    global_context = state.get("global_context", [])

    root_query = ""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            root_query = getattr(msg, "content", "")
            break

    if global_context:
        raw_findings_chunks: List[str] = [
            f"Document: {s.document_name}\nFindings:\n{s.findings}"
            for s in global_context
        ]

        merged_findings = await recursive_summarize_findings(
            raw_findings_chunks,
            root_query=root_query,
        )
    else:
        merged_findings = "No document findings available."

    response = AIMessage(content=merged_findings)

    return Command(goto=END, update={"messages": [response]})

