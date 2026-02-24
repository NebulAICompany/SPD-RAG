import time
from typing import Any, Dict, List, Literal, Optional

import tiktoken
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

from backend.retrieval.reranker import rerank
from backend.retrieval.retriever import load_vectorstore, retrieve_top_k
from backend.shared.constants import RESEARCH_LLM_FAST, VECTORSTORE_PATH_STR
from backend.shared.logger import get_logger

logger = get_logger("AGENTIC_RAG_BASELINE")


_ENCODER: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_get_encoder().encode(text))


class AgenticRAGAction(BaseModel):
    """Structured action the agent outputs on every turn of the retrieval loop."""

    action: Literal["search", "finalize"] = Field(
        description="'search' to issue a retrieval query against the knowledge base; 'finalize' when you have enough evidence to answer the question completely."
    )
    query: Optional[str] = Field(
        default=None,
        description="Targeted search query to run (required when action='search'). Be specific — prefer concrete terms over broad paraphrases.",
    )
    reasoning: str = Field(
        description="Brief explanation of why you are taking this action."
    )
    answer: Optional[str] = Field(
        default=None,
        description="Complete, well-structured answer to the user question (required when action='finalize'). Synthesise all retrieved evidence; be precise about numbers, names, and dates. Do NOT fabricate facts.",
    )


def _make_search_tool(selected_files: List[str]):
    """Build a retrieval tool scoped to *selected_files* via closure."""

    @tool(
        "search_documents",
        parse_docstring=True,
        response_format="content",
    )
    def search_documents(query: str) -> str:
        """Search for information within the task documents.

        Args:
            query: Search query to find relevant information.
        """
        try:
            client = load_vectorstore(VECTORSTORE_PATH_STR)
        except RuntimeError as exc:
            logger.error(f"[AgenticRAG] Failed to load vectorstore: {exc}")
            return "Vectorstore unavailable — cannot retrieve documents."

        if not client.collection_exists(collection_name="documents"):
            return "No documents found in the knowledge base."

        raw_docs = retrieve_top_k(
            client=client, query=query, k=15, selected_files=selected_files
        )

        if not raw_docs:
            return f"No relevant passages found for query: '{query}'."

        doc_dicts = [
            {"content": doc["content"], "metadata": doc["metadata"]} for doc in raw_docs
        ]
        reranked = rerank(query, doc_dicts, with_score=False, top_n=5)

        if not reranked:
            return f"No relevant passages survived reranking for query: '{query}'."

        passages: List[str] = []
        for doc in reranked:
            source = doc.get("metadata", {}).get("file_name", "unknown")
            passages.append(f"[Source: {source}]\n{doc['content']}")

        logger.info(
            f"[AgenticRAG] Retrieved {len(passages)} passage(s) for query: '{query}' "
            f"across {len(selected_files)} file(s)"
        )
        return "\n\n---\n\n".join(passages)

    return search_documents


_SYSTEM_PROMPT = """You are a research agent with access to a knowledge base that spans multiple documents.
Your goal is to answer the user's question accurately and completely by iteratively searching the knowledge base.

Retrieval loop rules:
1. On each turn output exactly one structured action:
   - action="search"   → issue ONE focused query; the results will be returned to you in the next turn.
   - action="finalize" → you have gathered sufficient evidence; compose the final answer in the `answer` field and stop.

2. Search strategy:
   - Start with the most specific terms that appear in the question.
   - If initial results are sparse, try synonyms, acronyms, or closely related concepts.
   - Issue separate queries for different sub-questions; never bundle multiple unrelated topics into one query.
   - Stop searching when every part of the question is covered by concrete retrieved evidence, or when additional searches return no new information.

3. Finalisation:
   - Synthesise the retrieved evidence into a clear, concise answer.
   - Preserve exact numbers, names, dates, and technical terms.
   - If the knowledge base does not contain enough information, state that explicitly — do NOT hallucinate.
   - Respond in the same language as the question.
"""


async def run_agentic_rag(
    query: str,
    selected_files: List[str],
    max_iterations: int = 20,
) -> Dict[str, Any]:
    """Run the agentic RAG agent and return ``raw_output``, ``latency``,
    ``prompt_tokens``, ``completion_tokens``, and ``iterations``.

    Args:
        query: The user question / task prompt.
        selected_files: Oracle document filenames to restrict retrieval to.
        max_iterations: Hard ceiling on search iterations before forced finalization.
    """
    logger.info(f"[AgenticRAG] Starting — query: {query[:120]!r}")

    start_time = time.perf_counter()
    prompt_tokens = _count_tokens(query)
    search_tool = _make_search_tool(selected_files)

    messages: List = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    action_extractor = RESEARCH_LLM_FAST.with_structured_output(
        AgenticRAGAction,
        method="function_calling",
        include_raw=False,
    ).with_retry(
        stop_after_attempt=3,
        retry_if_exception_type=(ValueError, ValidationError, OutputParserException),
    )

    iteration = 0
    final_answer = ""

    while True:
        iteration += 1

        if iteration > max_iterations:
            logger.error(
                f"[AgenticRAG] Safety ceiling of {max_iterations} iterations "
                f"reached — forcing finalization."
            )
            messages.append(
                HumanMessage(
                    content=(
                        "You have reached the maximum number of search iterations. "
                        "You MUST finalize your answer now using the evidence "
                        "gathered so far."
                    )
                )
            )
            try:
                forced: AgenticRAGAction = await action_extractor.ainvoke(messages)
                final_answer = (
                    forced.answer
                    or "Safety limit reached; answer could not be fully synthesised."
                )
            except Exception as exc:
                logger.error(f"[AgenticRAG] Forced finalisation failed: {exc}")
                final_answer = "Answer extraction failed after safety limit."
            break

        try:
            action: AgenticRAGAction = await action_extractor.ainvoke(messages)
        except Exception as exc:
            logger.error(
                f"[AgenticRAG] Action parsing failed (iter {iteration}): {exc}"
            )
            final_answer = "Action parsing failed; no answer could be produced."
            break

        logger.info(
            f"[AgenticRAG] Iter {iteration} — "
            f"action={action.action!r} | reasoning={action.reasoning!r}"
        )
        if action.action == "finalize":
            if not action.answer:
                logger.warning(
                    f"[AgenticRAG] 'finalize' action with empty answer "
                    f"(iter {iteration}) — treating as empty result."
                )
            final_answer = action.answer or "No answer was produced."
            break

        if not action.query:
            logger.warning(
                f"[AgenticRAG] 'search' action with empty query (iter {iteration}). "
                f"Prompting the agent to provide a query."
            )
            messages.append(
                HumanMessage(
                    content=(
                        "Your last action was 'search' but the `query` field was empty. "
                        "Please provide a specific search string, or finalize if you "
                        "have gathered sufficient information."
                    )
                )
            )
            continue

        search_results = await search_tool.ainvoke(
            {"query": action.query}
        )

        messages.append(
            AIMessage(
                content=(
                    f"[SEARCH] Reasoning: {action.reasoning}\n"
                    f"Query: {action.query}"
                )
            )
        )
        messages.append(
            HumanMessage(
                content=(
                    f"Search results for '{action.query}':\n\n{search_results}"
                )
            )
        )

    latency = time.perf_counter() - start_time
    completion_tokens = _count_tokens(final_answer)

    logger.info(
        f"[AgenticRAG] Done — "
        f"iterations={iteration - 1}, "
        f"latency={latency:.1f}s, "
        f"completion_tokens={completion_tokens}"
    )

    return {
        "raw_output": final_answer,
        "latency": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "iterations": max(0, iteration - 1),
    }
