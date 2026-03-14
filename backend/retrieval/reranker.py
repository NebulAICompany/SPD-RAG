"""Precision reranking of retrieved passages using Cohere rerank-v4.0-fast.

Used after dense retrieval to return only the top-n most relevant chunks (e.g. top-5
for sub-agent search, top-20 for normal RAG), reducing noise and token usage.
"""

from typing import List, Dict, Any
from backend.shared.logger import get_logger

logger = get_logger("RERANKER")


def rerank(
    query: str,
    documents: List[Dict[str, Any]],
    with_score: bool = True,
    model_name: str = "rerank-v4.0-fast",
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """Rerank documents by relevance to the query using Cohere rerank-v4.0-fast.

    On failure (e.g. config or API error), returns the first top_n documents
    in original order as fallback.

    Args:
        query: Search query.
        documents: List of dicts with 'content' and optional 'metadata'.
        with_score: If True, add 'score' to each result.
        model_name: Cohere rerank model (default rerank-v4.0-fast).
        top_n: Maximum number of results to return.

    Returns:
        Reranked list of document dicts (content, metadata, optional score).
    """
    if not documents:
        return []

    try:
        from backend.shared.constants import co

        doc_contents = [doc["content"] for doc in documents]

        response = co.rerank(
            model=model_name,
            query=query,
            documents=doc_contents,
            top_n=min(top_n, len(documents)),
        )

        reranked_results = []
        for result in response.results:
            original_doc = documents[result.index]
            reranked_doc = {
                "content": original_doc["content"],
                "metadata": original_doc.get("metadata", {}),
            }

            if with_score:
                reranked_doc["score"] = float(result.relevance_score)

            reranked_results.append(reranked_doc)

        logger.info("Reranked to %s results", len(reranked_results))
        return reranked_results

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.warning(
            "Falling back to original document order due to configuration error"
        )
        return documents[:top_n]

    except Exception as e:
        logger.error(f"Error during Cohere reranking: {e}")
        logger.warning("Falling back to original document order due to reranking error")
        return documents[:top_n]
