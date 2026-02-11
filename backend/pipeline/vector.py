from qdrant_client import QdrantClient, models
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from backend.retrieval.retriever import load_vectorstore
from backend.shared.constants import VECTORSTORE_PATH_STR, co
from backend.shared.logger import get_logger
from typing import List
import asyncio
import tiktoken
import hashlib

logger = get_logger("VECTOR_PIPELINE")
enc = tiktoken.get_encoding("cl100k_base")


async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Embed texts using Cohere embed-v4.0, batching automatically (max 96/call)."""
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

    batch_size = 96
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embs = await asyncio.to_thread(_embed_batch, batch)
        all_embeddings.extend(batch_embs)

    return all_embeddings


class VectorStorePipeline:
    """Simplified Vector Store Pipeline for RRM."""

    class Embedding:
        """Handles text embedding and Qdrant upload."""

        @staticmethod
        async def upload_text_embed(
            client: QdrantClient, processed_docs: List[Document]
        ):
            """Create embeddings for all docs and upload as Qdrant points."""
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
            logger.info(f"Uploaded {len(all_points)} points to vectorstore")

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=250,
            length_function=self.length_function,
        )

    def length_function(self, text: str) -> int:
        return len(enc.encode(text))

    async def run(self, text_content: str, document_name: str, file_extension: str = None):
        """Process text content and index it."""
        try:
            if not text_content or not text_content.strip():
                logger.error("❌ No valid text content provided")
                return

            chunk_idx = 0

            if file_extension in [".xlsx", ".xls"]:
                text_content_list = text_content.split("====SHEET SEPARATOR====")
                docs = self.text_splitter.create_documents(text_content_list)
            else:
                docs = self.text_splitter.create_documents([text_content])

            if not docs:
                logger.warning(f"⚠️ No chunks created for {document_name}.")
                return

            logger.info(f"✅ Document '{document_name}' split into {len(docs)} chunks")

            for doc in docs:
                if not hasattr(doc, "metadata") or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["chunk_id"] = f"chunk_{chunk_idx}"
                doc.metadata["file_name"] = document_name
                chunk_idx += 1

            client = load_vectorstore(VECTORSTORE_PATH_STR)

            if not client.collection_exists(collection_name="documents"):
                client.create_collection(
                    collection_name="documents",
                    vectors_config=models.VectorParams(
                        size=1536, distance=models.Distance.COSINE
                    ),
                )

            logger.info("Creating embeddings using Cohere embed-v4.0...")
            await self.Embedding.upload_text_embed(client, docs)
            logger.info("Processing complete.")

        except Exception as e:
            logger.error(f"❌ Error in vector store processing: {str(e)}")
            raise e
