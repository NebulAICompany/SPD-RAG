import json
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models
from backend.shared.constants import co
from backend.shared.logger import get_logger
from backend.retrieval.keyword_search import keyword_search

logger = get_logger("RETRIEVER")

_qdrant_client: Optional[QdrantClient] = None


def embed_query(query: str) -> List[float]:
    """Embed a single query using Cohere"""
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
    """Embed multiple documents using Cohere"""
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
    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    try:
        _qdrant_client = QdrantClient(path=path)
        logger.info(f"✅ Vectorstore loaded from {path}")
        if _qdrant_client.collection_exists(collection_name="documents"):
            logger.info(
                f"📦 Contains {_qdrant_client.count(collection_name='documents')} document chunks"
            )
        else:
            logger.info("No collection found")
        return _qdrant_client
    except Exception as e:
        raise RuntimeError(f"Failed to load vectorstore from {path}: {e}") from e


def get_vectorstore() -> Optional[QdrantClient]:
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
    chunk_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        filter_msg = f" in document '{chunk_id}'" if chunk_id else " across all documents"
        logger.info(f"🔍 Retrieving top {k} documents for query: {query}{filter_msg}")
        if not client.collection_exists(collection_name="documents"):
            logger.info("No collection found")
            return None

        logger.info(f"📊 Searching through {client.count(collection_name='documents')} document chunks")

        query_filter = None
        if chunk_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.chunk_id",
                        match=models.MatchValue(value=chunk_id)
                    )
                ]
            )
        
        docs_with_scores = client.query_points(
            collection_name="documents",
            query=embed_query(query),
            limit=k,
            score_threshold=0.2,
            query_filter=query_filter,
        ).points
        
        logger.info(f"✅ Retrieved {len(docs_with_scores)} documents from vectorstore")

        results = []
        chunks_with_images = 0
        for d in docs_with_scores:
            doc = d.payload
            score = d.score
            logger.info(f"chunk ID: {doc['metadata'].get('chunk_id')}")
            logger.info(f"score: {score}")
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
        logger.info(f"📈 Retrieved {len(results)} document chunks")
        logger.info(f"Results: {json.dumps(results, indent=4)}")
        return results

    except Exception as e:
        logger.error(f"❌ Error during retrieval: {e}")
        return []


def retrieve_with_keyword_helping(
    client: QdrantClient,
    query: str,
    query_terms: List[str],
    k: int = 10,
    chunk_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        filter_msg = f" in document '{chunk_id}'" if chunk_id else " across all documents"
        logger.info(f"🔍 Vector + keyword search helping for: '{query}'{filter_msg} (limit: {k}+3)")

        # Perform vector search
        vector_results = retrieve_top_k(
            client, query, k=k, chunk_id=chunk_id
        )

        # Perform keyword search
        keyword_results = keyword_search(
            query_terms, k=3, chunk_id=chunk_id
        )

        # Combine results (handle None case)
        if vector_results is None:
            vector_results = []

        results = vector_results + keyword_results
        logger.info(
            f"✅ Retrieved {len(results)} documents via vector + keyword search helping ({len(vector_results)} vector, {len(keyword_results)} keyword)"
        )

        return results

    except Exception as e:
        logger.error(f"❌ Error during vector + keyword search helping: {e}")
        return []
