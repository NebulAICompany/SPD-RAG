import time
from typing import Any, Dict, List, Optional

import tiktoken
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from backend.retrieval.reranker import rerank
from backend.retrieval.retriever import load_vectorstore, retrieve_top_k
from backend.shared.constants import RESEARCH_LLM_REASONING, VECTORSTORE_PATH_STR
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
            query: Search query to find relevant information in the task documents.
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
Your goal is to answer the user's question accurately and completely.

Instructions:
- Use the search_documents tool to retrieve relevant information before answering.
- Search thoroughly for different aspects of the question to gather comprehensive evidence.
- Avoid redundant searches: if you already retrieved information on a topic, use it instead of searching again.
- Once you have gathered sufficient evidence covering all key aspects, synthesize a complete answer from the retrieved passages.
- Do NOT continue searching after you have enough information to answer confidently.
- Preserve exact numbers, names, dates, and technical terms from the retrieved passages.
- If the knowledge base lacks sufficient information, state that explicitly — do NOT hallucinate.
- Respond in the same language as the question.
"""


async def run_agentic_rag(
    query: str,
    selected_files: List[str],
    max_iterations: int = 10,
    ls_config: Optional[Dict[str, Any]] = None,
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

    # First call requires tool usage, then auto to let LLM decide when to stop
    llm_with_tools = RESEARCH_LLM_REASONING.bind_tools([search_tool], tool_choice="auto")

    messages: List[Any] = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

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
                        "Provide your final answer now based on the evidence gathered so far."
                    )
                )
            )
            try:
                forced = await RESEARCH_LLM_REASONING.ainvoke(messages, config=ls_config)
                c = forced.content
                if isinstance(c, list):
                    c = " ".join(b["text"] for b in c if isinstance(b, dict) and b.get("type") == "text")
                final_answer = c or "Safety limit reached; answer could not be fully synthesised."
            except Exception as exc:
                logger.error(f"[AgenticRAG] Forced finalisation failed: {exc}")
                final_answer = "Answer extraction failed after safety limit."
            break

        try:
            response = await llm_with_tools.ainvoke(messages, config=ls_config)
        except Exception as exc:
            logger.error(f"[AgenticRAG] LLM invocation failed (iter {iteration}): {exc}")
            final_answer = "LLM invocation failed; no answer could be produced."
            break

        messages.append(response)

        if not response.tool_calls:
            c = response.content
            if isinstance(c, list):
                c = " ".join(b["text"] for b in c if isinstance(b, dict) and b.get("type") == "text")
            final_answer = c or "No answer was produced."
            break

        logger.info(f"[AgenticRAG] Iter {iteration} — {len(response.tool_calls)} tool call(s)")

        for tool_call in response.tool_calls:
            tool_result = await search_tool.ainvoke(tool_call["args"])
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
            logger.info(f"[AgenticRAG] Searched: {tool_call['args'].get('query', '')!r}")

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
