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
from backend.core.state import AgentState, SubAgentInput, Summary, TodoItem
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
            else:
                pass
        else:
            pass

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
    """Tool for defining tasks that sub-agents must execute on their assigned documents."""

    sub_agent_todos: List[TodoItem] = Field(
        description="A list of specific tasks that EVERY sub-agent must execute for their assigned document. Each item must have a 'task' and a 'status' (default 'pending')."
    )


async def orchestrator_node(
    state: AgentState, config: RunnableConfig
) -> Dict[str, Any]:
    """
    Defines tasks for sub-agents to execute on their assigned documents.

    Binds a WriteTodos tool to the LLM to generate sub_agent_todos based
    on the user query and available documents.

    Args:
        state: Current agent state.
        config: Runtime configuration.

    Returns:
        State updates including messages and sub_agent_todos.
    """
    messages = state["messages"]

    llm_with_tools = RESEARCH_LLM_REASONING.bind_tools([WriteTodos], tool_choice="auto")
    
    response = await llm_with_tools.ainvoke(
        [{"role": "system", "content": LEAD_RESEARCHER_PROMPT}] + messages
    )

    updates: Dict[str, Any] = {"messages": [response]}

    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "WriteTodos":
                sub_todos_raw = tool_call["args"].get("sub_agent_todos", [])
                if sub_todos_raw:
                    updates["sub_agent_todos"] = [TodoItem(**t) for t in sub_todos_raw]

    return updates


async def document_sub_agent_node(input_data: SubAgentInput) -> Dict[str, Any]:
    """
    Processes a document query using an agentic RAG workflow.

    This node acts as a specialized sub-agent that:
    1. Receives a topic/document ID.
    2. Uses the LLM to formulate a search query for the `search_local_documents` tool.
    3. Executes the tool to retrieve information.
    4. Synthesizes the findings into a structured Summary.

    Args:
        input_data: SubAgentInput containing the document_name (used as research topic).

    Returns:
        State update with the Summary added to global_context.
    """
    from langchain_core.messages import ToolMessage

    doc_name = input_data["document_name"]
    todos_list = input_data.get("todos", [])

    # Format TodoItems into a string list for the prompt
    # todos_list is a list of TodoItem objects (or dicts if not pushed as objects)
    todos_str = ""
    if todos_list:
        todos_str = "\n".join(
            [
                f"{i+1}. [ ] {t.task if hasattr(t, 'task') else t.get('task')}"
                for i, t in enumerate(todos_list)
            ]
        )
    else:
        todos_str = "No specific sub-tasks provided."

    model_with_tools = RESEARCH_LLM_FAST.bind_tools(
        [search_specific_document_for_research]
    )

    system_prompt = RESEARCH_SYSTEM_PROMPT.format(file_name=doc_name)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Research Topic: {doc_name}\n\n**Orchestrator Assigned Tasks**:\nYou must address the following points about this document:\n{todos_str}"
        ),
    ]

    ai_msg = await model_with_tools.ainvoke(messages)
    messages.append(ai_msg)

    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            if tool_call["name"] == "search_specific_document":
                tool_output = search_specific_document_for_research.invoke(
                    tool_call["args"]
                )

                messages.append(
                    ToolMessage(
                        content=str(tool_output),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )

    extractor = RESEARCH_LLM_FAST.with_structured_output(
        Summary,
        method="function_calling",
        include_raw=False,
    ).with_retry(
        stop_after_attempt=3,
        retry_if_exception_type=(
            ValueError,
            ValidationError,
            OutputParserException,
        ),
    )
    summary = await extractor.ainvoke(messages)

    summary.document_name = doc_name

    return {"global_context": [summary]}


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
    selected_docs = state.get("selected_documents", [])

    if len(global_context) < len(selected_docs):
        return Command(goto=END)

    root_query = ""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            root_query = getattr(msg, "content", "")
            break

    if global_context:
        raw_findings_chunks: List[str] = [
            (
                f"Document: {s.document_name}\n"
                f"Findings:\n{s.findings}"
            )
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



