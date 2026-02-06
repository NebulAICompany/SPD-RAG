from langchain_core.tools import tool
from typing import List, Optional
from backend.retrieval.retriever import (
    get_vectorstore,
    retrieve_with_keyword_helping,
)
from backend.retrieval.reranker import rerank
from backend.shared.logger import get_logger

logger = get_logger("RAG_TOOL")


@tool(parse_docstring=True, response_format="content_and_artifact")
def search_local_documents(
    query: str,
    keywords: Optional[List[str]] = None,
    max_results: int = 5,
) -> str:
    """Search uploaded local documents in the knowledge base.

    Use this when you need information from documents that were previously uploaded.

    Args:
        query: Search query to find relevant information in local documents.
        keywords: Optional list of keywords for hybrid vector + keyword search.
        max_results: Maximum number of document chunks to return (default 5, max 10).
    """
    try:
        # Limit max_results to reasonable bounds
        max_results = min(max(1, max_results), 10)

        # Get the global vectorstore client
        client = get_vectorstore()

        # Check if client is available
        if client is None:
            return (
                "Vectorstore is not available. Please ensure documents are uploaded.",
                [],
            )

        # Check if collection exists
        if not client.collection_exists(collection_name="documents"):
            logger.info("No collection found in vectorstore")
            return "No documents have been uploaded to the knowledge base yet.", []

        # Use provided keywords or empty list if not provided
        query_terms = keywords if keywords is not None else []

        logger.info(
            f"🔍 RAG Tool - Searching through all uploaded documents"
        )

        # Retrieve documents using hybrid search (vector + keyword)
        retrieved_docs = retrieve_with_keyword_helping(
            client=client,
            query=query,
            query_terms=query_terms,
            k=15,
        )

        if not retrieved_docs:
            return "No relevant documents found for your query.", []

        # Rerank documents
        doc_contents = [
            {"content": doc["content"], "metadata": doc["metadata"]}
            for doc in retrieved_docs
        ]
        reranked_docs = rerank(query, doc_contents, with_score=False, top_n=max_results)

        if not reranked_docs:
            return "No relevant documents found after reranking.", []

        # Format results for LLM and collect sources for artifact
        results = []
        sources = []

        for doc in reranked_docs:
            content = doc["content"]
            metadata = doc.get("metadata", {})
            file_name = metadata.get("file_name", "Unknown")
            page = metadata.get("page", "")

            result_text = f"Source: {file_name}"
            result_text += f"\n{content}\n"
            if page:
                result_text += f" (Page {page})"
            results.append(result_text)

            # Collect source information for artifact
            source_name = file_name
            if page:
                source_name = f"{file_name} - Page {page}"
            sources.append(
                {"name": source_name, "file": file_name, "page": page if page else None}
            )

        formatted_results = "\n---\n".join(results)

        content = formatted_results
        artifact = sources

        return content, artifact

    except Exception as e:
        logger.error(f"Error in search_local_documents: {e}")
        return f"Error searching documents: {str(e)}", []


@tool(
    "search_specific_document",
    parse_docstring=True,
    response_format="content_and_artifact",
)
def search_specific_document_for_research(
    query: str,
    file_name: str,
    max_results: int = 3,
) -> str:
    """Search for information within a SPECIFIC local document.

    Use this tool when you are assigned to research a specific document.
    Note: Currently searches across all documents as file filtering is disabled.

    Args:
        query: Search query to find relevant information.
        file_name: The exact name of the file to search within (INFO ONLY - not used for filtering).
        max_results: Maximum number of document chunks to return (default 3, max 5).
    """
    try:
        max_results = min(max(1, max_results), 5)
        client = get_vectorstore()

        if client is None:
            return "Vectorstore not available.", []
        if not client.collection_exists(collection_name="documents"):
            return "No documents found in knowledge base.", []

        logger.info(f"🔍 Agentic Tool - Searching through all documents (file_name '{file_name}' is for reference only)")

        retrieved_docs = retrieve_with_keyword_helping(
            client=client,
            query=query,
            query_terms=[],  # Optional: could expose keywords if needed
            k=15,
        )

        if not retrieved_docs:
            return f"No relevant information found in {file_name} for your query.", []

        # Rerank
        doc_contents = [
            {"content": doc["content"], "metadata": doc["metadata"]}
            for doc in retrieved_docs
        ]
        reranked_docs = rerank(query, doc_contents, with_score=False, top_n=max_results)

        if not reranked_docs:
            return f"No relevant information found in {file_name} after reranking.", []

        # Format results
        results = []
        image_ids = []
        sources = []

        for doc in reranked_docs:
            content = doc["content"]
            metadata = doc.get("metadata", {})
            f_name = metadata.get("file_name", "Unknown")
            page = metadata.get("page", "")

            result_text = f"Source: {f_name}"
            result_text += f"\n{content}\n"
            if page:
                result_text += f" (Page {page})"


            results.append(result_text)

            # Metadata for artifact
            source_name = f"{f_name} - Page {page}" if page else f_name
            sources.append({"name": source_name, "file": f_name, "page": page or None})

        formatted_results = "\n---\n".join(results)

        return formatted_results, sources

    except Exception as e:
        logger.error(f"Error in search_specific_document: {e}")
        return f"Error searching document {file_name}: {str(e)}", []
