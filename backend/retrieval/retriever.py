import os
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

    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    if qdrant_url:
        try:
            _qdrant_client = QdrantClient(url=qdrant_url)
            logger.info(f"✅ Vectorstore connected to Qdrant server at {qdrant_url}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Qdrant server at {qdrant_url}: {e}") from e
    else:
        try:
            _qdrant_client = QdrantClient(path=path)
            logger.info(f"✅ Vectorstore loaded from {path}")
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
            f"📦 Contains {_qdrant_client.count(collection_name='documents')} document chunks"
        )
    else:
        logger.info("No collection found")
    return _qdrant_client


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
    selected_files: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    try:
        logger.info(f"🔍 Retrieving top {k} documents for query: {query}")
        if not client.collection_exists(collection_name="documents"):
            logger.info("No collection found")
            return []

        logger.info(
            f"📊 Searching through {client.count(collection_name='documents')} document chunks"
        )

        if selected_files:
            # selected_files = [file.split(".")[0] for file in selected_files]
            logger.info(f"🔍 Searching through {selected_files} document chunks")
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
                score_threshold=0.05,
            ).points
        else:
            docs_with_scores = client.query_points(
                collection_name="documents",
                query=embed_query(query),
                limit=k,
                score_threshold=0.05,
            ).points
        logger.info(f"✅ Retrieved {len(docs_with_scores)} documents from vectorstore")

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
        logger.info(f"📈 Retrieved {len(results)} document chunks")
        return results

    except Exception as e:
        logger.error(f"❌ Error during retrieval: {e}")
        return []


def retrieve_with_keyword_helping(
    client: QdrantClient,
    query: str,
    query_terms: List[str],
    k: int = 10,
    selected_files: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    try:
        logger.info(f"🔍 Vector + keyword search helping for: '{query}' (limit: {k}+3)")

        # Perform vector search
        vector_results = retrieve_top_k(
            client, query, k=k, selected_files=selected_files
        )

        # Perform keyword search
        keyword_results = keyword_search(
            query_terms, k=3, selected_files=selected_files
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
