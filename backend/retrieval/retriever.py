"""Dense retrieval over Qdrant using Cohere embed-v4.0 (1536-dim).

Supports document-scoped filtering for SPD-RAG sub-agents.
"""

import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from backend.shared.constants import co
from backend.shared.logger import get_logger

logger = get_logger("RETRIEVER")

_qdrant_client: Optional[QdrantClient] = None


def embed_query(query: str) -> List[float]:
    """Embed a single query with Cohere embed-v4.0 (search_query input type, 1536 dims)."""
    query_input = [{"content": [{"type": "text", "text": query}]}]
    query_emb = co.embed(
        inputs=query_input,
        model="embed-v4.0",
        input_type="search_query",
        output_dimension=1536,
        embedding_types=["float"],
    ).embeddings.float
    return query_emb[0]


def embed_documents(documents: List[str]) -> List[List[float]]:
    """Embed multiple documents with Cohere embed-v4.0 (search_document input type, 1536 dims)."""
    embed_input = [{"content": [{"type": "text", "text": doc}]} for doc in documents]
    doc_emb = co.embed(
        inputs=embed_input,
        model="embed-v4.0",
        output_dimension=1536,
        input_type="search_document",
        embedding_types=["float"],
    ).embeddings.float
    return doc_emb


def load_vectorstore(path: str) -> QdrantClient:
    """Connect to Qdrant (server via QDRANT_URL or local path) and cache the client."""
    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    if qdrant_url:
        try:
            _qdrant_client = QdrantClient(url=qdrant_url)
            logger.info("Vectorstore connected to Qdrant at %s", qdrant_url)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Qdrant server at {qdrant_url}: {e}") from e
    else:
        try:
            _qdrant_client = QdrantClient(path=path)
            logger.info("Vectorstore loaded from path: %s", path)
        except Exception as e:
            if "already accessed" in str(e):
                raise RuntimeError(
                    f"Failed to load vectorstore from {path}: {e}\n"
                    "Hint: another process (e.g. the backend server) already holds the "
                    "Qdrant storage lock. Either stop that process before running the "
                    "evaluator, or run Qdrant as a server and set QDRANT_URL=http://localhost:6333 "
                    "in your .env so both can share it concurrently."
                ) from e
            raise RuntimeError(f"Failed to load vectorstore from {path}: {e}") from e

    if _qdrant_client.collection_exists(collection_name="documents"):
        logger.info(
            "Collection 'documents' has %s chunks",
            _qdrant_client.count(collection_name="documents"),
        )
    else:
        logger.info("No 'documents' collection found")
    return _qdrant_client


def get_vectorstore() -> Optional[QdrantClient]:
    """Return the cached Qdrant client, or None if not yet loaded."""
    return _qdrant_client


def close_vectorstore() -> None:
    global _qdrant_client
    if _qdrant_client is not None:
        try:
            _qdrant_client.close()
            logger.info("Qdrant client closed")
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {e}")
        finally:
            _qdrant_client = None


def retrieve_top_k(
    client: QdrantClient,
    query: str,
    k: int = 10,
    selected_files: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks by cosine similarity; optionally filter by file names.

    Embeds the query with Cohere embed-v4.0 and queries the 'documents' collection.
    Used by both document-scoped (sub-agent) and normal RAG search.
    """
    try:
        logger.info("Retrieving top %s for query: %s", k, query[:80] if query else "")
        if not client.collection_exists(collection_name="documents"):
            logger.info("No collection found")
            return []

        logger.info(
            f"📊 Searching through {client.count(collection_name='documents')} document chunks"
        )

        if selected_files:
            logger.info("Filtering by files: %s", selected_files)
            docs_with_scores = client.query_points(
                collection_name="documents",
                query=embed_query(query),
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.file_name",
                            match=models.MatchAny(any=selected_files),
                        )
                    ]
                ),
                limit=k,
                score_threshold=0,
            ).points
        else:
            docs_with_scores = client.query_points(
                collection_name="documents",
                query=embed_query(query),
                limit=k,
                score_threshold=0,
            ).points
        logger.info("Retrieved %s points from vectorstore", len(docs_with_scores))

        results = []
        chunks_with_images = 0
        for d in docs_with_scores:
            doc = d.payload
            score = d.score
            logger.debug(f"file_name={doc['metadata'].get('file_name')}, score={score}")
            contains_image = doc["metadata"].get("contains_image", False)

            if contains_image:
                chunks_with_images += 1

            results.append(
                {
                    "content": doc["page_content"],
                    "score": score,
                    "metadata": {
                        **doc["metadata"],
                        "match_type": "content_match",
                        "contains_image": contains_image,
                    },
                }
            )
        logger.info("Returning %s document chunks", len(results))
        return results

    except Exception as e:
        logger.error("Retrieval error: %s", e)
        return []
