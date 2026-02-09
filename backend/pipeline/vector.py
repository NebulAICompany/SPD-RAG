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

class VectorStorePipeline:
    """
    Simplified Vector Store Pipeline for RRM.
    Removes PII masking and AutoContext for lightweight execution.
    """

    class Embedding:
        """Handles text embedding operations using Cohere"""

        @staticmethod
        async def upload_text_embed(
            client: QdrantClient, processed_docs: List[Document]
        ):
            """Create embeddings and upload text chunks to Qdrant"""
            batch_size = 96
            all_points = []
            
            # Helper function for Cohere embedding
            def embed_batch(texts):
                if not co:
                    logger.error("Cohere client not available. Cannot embed.")
                    return []
                
                embed_input = [
                    {"content": [{"type": "text", "text": text}]}
                    for text in texts
                ]
                return co.embed(
                    inputs=embed_input,
                    model="embed-v4.0",
                    input_type="search_document",
                    output_dimension=1536,
                    embedding_types=["float"],
                ).embeddings.float

            for i in range(0, len(processed_docs), batch_size):
                batch_docs = processed_docs[i : i + batch_size]
                logger.info(
                    f"Processing batch {i//batch_size + 1}/{(len(processed_docs) + batch_size - 1)//batch_size} ({len(batch_docs)} documents)"
                )

                batch_texts = [doc.page_content for doc in batch_docs]

                try:
                    # Run embedding in thread
                    batch_embeddings = await asyncio.to_thread(embed_batch, batch_texts)
                    
                    if not batch_embeddings:
                        logger.warning("No embeddings returned for batch.")
                        continue

                    # Create Qdrant points
                    for idx, (doc, embedding) in enumerate(
                        zip(batch_docs, batch_embeddings)
                    ):
                        # Generate ID deterministically from content + idx
                        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                        # Use a large integer ID derived from hash + index to avoid collisions
                        point_id = int(content_hash[:15], 16) + idx 
                        
                        point = models.PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "page_content": doc.page_content,
                                "metadata": doc.metadata,
                            },
                        )
                        all_points.append(point)

                except Exception as e:
                    logger.error(f"Error embedding batch: {str(e)}")
                    continue

            if all_points:
                client.upload_points(
                    collection_name="documents",
                    points=all_points,
                )
                logger.info(
                    f"Successfully uploaded {len(all_points)} points to vectorstore"
                )
            else:
                 logger.warning("No points to upload.")

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
        """
        Process text content and index it.
        """
        try:
            if not text_content or not text_content.strip():
                logger.error("❌ No valid text content provided")
                return

            chunk_idx = 0

            # Split text into chunks
            if file_extension in [".xlsx", ".xls"]:
                text_content_list = text_content.split("====SHEET SEPARATOR====")
                docs = self.text_splitter.create_documents(text_content_list)
            else:
                docs = self.text_splitter.create_documents([text_content])
            
            if not docs:
                logger.warning(
                    f"⚠️ Warning: No chunks were created for {document_name}."
                )
                return

            logger.info(
                f"   ✅ Document '{document_name}' split into {len(docs)} chunks"
            )

            # Add metadata
            for doc in docs:
                if not hasattr(doc, "metadata") or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["chunk_id"] = f"{document_name}_chunk_{chunk_idx}"
                doc.metadata["file_name"] = f"{document_name}"
                chunk_idx += 1

            # Load Vectorstore
            client = load_vectorstore(VECTORSTORE_PATH_STR)

            # Create Collection if not exists
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
