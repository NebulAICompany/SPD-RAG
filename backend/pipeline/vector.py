"""Document chunking, embedding (Cohere embed-v4.0), and Qdrant indexing for SPD-RAG.

Splits markdown-oriented text into chunks (token-counted with cl100k_base),
embeds with Cohere embed-v4.0 (1536-dim), and uploads to the 'documents' collection.
Used by the upload pipeline and by the synthesis layer for recursive summarization.
"""

from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from backend.retrieval.retriever import load_vectorstore
from backend.shared.constants import VECTORSTORE_PATH_STR, co
from backend.shared.logger import get_logger
from typing import List, Optional
import asyncio
import tiktoken
import hashlib
import time

logger = get_logger("VECTOR_PIPELINE")
enc = tiktoken.get_encoding("cl100k_base")


async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Embed texts with Cohere embed-v4.0 (search_document, 1536-dim), with batching and rate-limit retries."""
    if not texts:
        return []

    def _embed_batch(batch_texts: List[str]) -> List[List[float]]:
        if not co:
            raise RuntimeError("Cohere client not initialised")
        embed_input = [
            {"content": [{"type": "text", "text": t}]} for t in batch_texts
        ]
        return co.embed(
            inputs=embed_input,
            model="embed-v4.0",
            input_type="search_document",
            output_dimension=1536,
            embedding_types=["float"],
        ).embeddings.float

    batch_size = 8
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(13):
            try:
                batch_embs = await asyncio.to_thread(_embed_batch, batch)
                break
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < 12:
                    logger.warning(f"Embed rate limit hit, retrying in 5s... (attempt {attempt + 1}/13)")
                    await asyncio.sleep(5)
                else:
                    raise
        all_embeddings.extend(batch_embs)

    return all_embeddings


class VectorStorePipeline:
    """Chunk documents (markdown-oriented), embed with Cohere, and index in Qdrant.

    Used when uploading new documents so they become searchable by the retrieval layer.
    """

    class Embedding:
        """Generate embeddings for document chunks and upload them to Qdrant."""

        @staticmethod
        async def upload_text_embed(
            client: QdrantClient, processed_docs: List[Document]
        ):
            """Embed all chunks with Cohere embed-v4.0 and upsert into the 'documents' collection."""
            if not processed_docs:
                logger.warning("No documents to embed.")
                return

            texts = [doc.page_content for doc in processed_docs]
            logger.info(f"Embedding {len(texts)} chunks...")

            embeddings = await generate_embeddings(texts)

            if len(embeddings) != len(processed_docs):
                raise ValueError(
                    f"Embedding count ({len(embeddings)}) != doc count ({len(processed_docs)})"
                )

            all_points = []
            for idx, (doc, embedding) in enumerate(
                zip(processed_docs, embeddings)
            ):
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                point_id = int(content_hash[:15], 16) + idx

                all_points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "page_content": doc.page_content,
                            "metadata": doc.metadata,
                        },
                    )
                )

            client.upload_points(collection_name="documents", points=all_points)
            logger.info("Uploaded %s points to vectorstore", len(all_points))

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=250,
            length_function=self.length_function,
        )

    def length_function(self, text: str) -> int:
        """Token count for the splitter (cl100k_base)."""
        return len(enc.encode(text, disallowed_special=()))

    async def run(self, text_content: str, document_name: str, metadata: Optional[dict] = None):
        """Split text into chunks, embed with Cohere, and index in Qdrant under document_name."""
        try:
            if not text_content or not text_content.strip():
                logger.error("No valid text content provided")
                return

            chunk_idx = 0

            docs = self.text_splitter.create_documents([text_content])

            if not docs:
                logger.warning("No chunks created for document: %s", document_name)
                return

            logger.info("Document '%s' split into %s chunks", document_name, len(docs))

            for doc in docs:
                if not hasattr(doc, "metadata") or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["chunk_id"] = f"chunk_{chunk_idx}"
                doc.metadata["file_name"] = document_name
                if metadata:
                    doc.metadata.update(metadata)
                chunk_idx += 1

            client = load_vectorstore(VECTORSTORE_PATH_STR)

            if not client.collection_exists(collection_name="documents"):
                client.create_collection(
                    collection_name="documents",
                    vectors_config=models.VectorParams(
                        size=1536, distance=models.Distance.COSINE
                    ),
                )
            else:
                existing = client.count(
                    collection_name="documents",
                    count_filter=models.Filter(
                        must=[models.FieldCondition(
                            key="metadata.file_name",
                            match=models.MatchValue(value=document_name),
                        )]
                    ),
                )
                if existing.count > 0:
                    logger.info("Document '%s' already in vectorstore; skipping", document_name)
                    return

            logger.info("Creating embeddings with Cohere embed-v4.0")
            await self.Embedding.upload_text_embed(client, docs)
            logger.info("Vector pipeline complete")

        except Exception as e:
            logger.error("Vector pipeline error: %s", e)
            raise e
