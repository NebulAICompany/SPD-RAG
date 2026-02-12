from langchain_core.tools import tool
from typing import List, Optional
from backend.retrieval.retriever import (
    get_vectorstore,
    retrieve_with_keyword_helping,
)
from backend.retrieval.reranker import rerank
from backend.shared.logger import get_logger

logger = get_logger("RAG_TOOL")


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

    Args:
        query: Search query to find relevant information.
        file_name: The exact name of the file to search within.
        max_results: Maximum number of document chunks to return (default 3, max 5).
    """
    try:
        max_results = min(max(1, max_results), 5)
        client = get_vectorstore()

        if client is None:
            return "Vectorstore not available.", []
        if not client.collection_exists(collection_name="documents"):
            return "No documents found in knowledge base.", []

        # Force file filter to the specific file
        selected_files = [file_name]
        logger.info(f"🔍 Agentic Tool - Searching ONLY in: {selected_files}")

        retrieved_docs = retrieve_with_keyword_helping(
            client=client,
            query=query,
            query_terms=[],  # Optional: could expose keywords if needed
            k=15,
            selected_files=selected_files,
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
