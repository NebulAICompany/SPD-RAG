"""RAG tools used by the SPD-RAG retrieval layer.

Provides document-scoped search (for sub-agents) and normal multi-document RAG.
Uses dense retrieval (Qdrant + Cohere embed-v4.0) and Cohere rerank-v4.0-fast.
"""

from backend.retrieval.retriever import retrieve_top_k
from langchain_core.tools import tool
from typing import List, Optional
from backend.retrieval.reranker import rerank
from backend.shared.logger import get_logger
from backend.shared.constants import VECTORSTORE_PATH_STR
from backend.retrieval.retriever import load_vectorstore
import time

logger = get_logger("RAG_TOOL")


def rerank_with_retry(query, doc_contents, with_score=False, top_n=20, retries=3, delay=5):
    """Run Cohere rerank with retries on rate limit; used after retrieval."""
    for attempt in range(retries):
        try:
            return rerank(query, doc_contents, with_score=with_score, top_n=top_n)
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < retries - 1:
                logger.warning(f"Rerank rate limit hit, retrying in {delay}s... (attempt {attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                raise


@tool(
    "search_specific_document",
    parse_docstring=True,
    response_format="content",
)
async def search_specific_document_for_research(
    query: str,
    file_name: str,
) -> str:
    """Search within a single document (used by document sub-agents).

    Runs dense retrieval (Qdrant + Cohere embed-v4.0) filtered by file_name,
    then reranks with Cohere rerank-v4.0-fast (top-5). Sub-agents call this
    once per 'search' action; they do not see other documents.

    Args:
        query: Search query for relevant passages.
        file_name: Exact document name to search (no cross-document results).

    Returns:
        Concatenated passage text, or an error/empty message.
    """
    try:
        client = load_vectorstore(VECTORSTORE_PATH_STR)

        if client is None:
            return "Vectorstore not available."
        if not client.collection_exists(collection_name="documents"):
            return "No documents found in knowledge base."

        selected_files = [file_name]
        logger.info("Document-scoped search in: %s", selected_files)

        retrieved_docs = retrieve_top_k(
            client=client,
            query=query,
            k=15,
            selected_files=selected_files,
        )

        if not retrieved_docs:
            return f"No relevant information found in {file_name} for your query."

        doc_contents = [
            {"content": doc["content"], "metadata": doc["metadata"]}
            for doc in retrieved_docs
        ]
        reranked_docs = rerank_with_retry(query, doc_contents, with_score=False, top_n=5)

        if not reranked_docs:
            return f"No relevant information found in {file_name} after reranking."

        results = []

        for doc in reranked_docs:
            content = doc["content"]
            result_text = f"\n{content}\n"
            results.append(result_text)

        formatted_results = "\n---\n".join(results)

        return formatted_results

    except Exception as e:
        logger.error(f"Error in search_specific_document: {e}")
        return f"Error searching document {file_name}: {str(e)}"

def perform_normal_rag(query: str, selected_files: List[str]) -> str:
    """Run standard RAG over the given documents (dense retrieval + rerank).

    No sub-agent loop; single query over selected_files, top-30 then rerank top-20.
    Used for non-SPD-RAG or baseline flows.
    """
    try:
        client = load_vectorstore(VECTORSTORE_PATH_STR)

        if client is None:
            return "Vectorstore not available."
        if not client.collection_exists(collection_name="documents"):
            return "No documents found in knowledge base."

        logger.info("Normal RAG search in: %s", selected_files)

        retrieved_docs = retrieve_top_k(
            client=client,
            query=query,
            k=30,
            selected_files=selected_files,
        )

        if not retrieved_docs:
            return f"No relevant information found for your query."

        doc_contents = [
            {"content": doc["content"], "metadata": doc["metadata"]}
            for doc in retrieved_docs
        ]
        reranked_docs = rerank_with_retry(query, doc_contents, with_score=False, top_n=20)

        if not reranked_docs:
            return f"No relevant information found after reranking."

        results = []

        for doc in reranked_docs:
            content = doc["content"]
            result_text = f"\n{content}\n"
            results.append(result_text)

        formatted_results = "\n---\n".join(results)

        return formatted_results

    except Exception as e:
        logger.error(f"Error in perform_normal_rag: {e}")
        return f"Error performing normal RAG: {str(e)}"