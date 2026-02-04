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

        logger.info(
            f"Successfully reranked documents. Returned {len(reranked_results)} results"
        )
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
