from backend.retrieval.retriever import retrieve_top_k
from langchain_core.tools import tool
from typing import List, Optional
from backend.retrieval.reranker import rerank
from backend.shared.logger import get_logger
from backend.shared.constants import VECTORSTORE_PATH_STR
from backend.retrieval.retriever import load_vectorstore

logger = get_logger("RAG_TOOL")


@tool(
    "search_specific_document",
    parse_docstring=True,
    response_format="content",
)
def search_specific_document_for_research(
    query: str,
    file_name: str,
) -> str:
    """Search for information within a SPECIFIC local document.

    Use this tool when you are assigned to research a specific document.

    Args:
        query: Search query to find relevant information.
        file_name: The exact name of the file to search within.
    """
    try:
        client = load_vectorstore(VECTORSTORE_PATH_STR)

        if client is None:
            return "Vectorstore not available."
        if not client.collection_exists(collection_name="documents"):
            return "No documents found in knowledge base."

        # Force file filter to the specific file
        selected_files = [file_name]
        logger.info(f"🔍 Agentic Tool - Searching ONLY in: {selected_files}")

        retrieved_docs = retrieve_top_k(
            client=client,
            query=query,
            k=15,
            selected_files=selected_files,
        )

        if not retrieved_docs:
            return f"No relevant information found in {file_name} for your query."

        # Rerank
        doc_contents = [
            {"content": doc["content"], "metadata": doc["metadata"]}
            for doc in retrieved_docs
        ]
        reranked_docs = rerank(query, doc_contents, with_score=False, top_n=5)

        if not reranked_docs:
            return f"No relevant information found in {file_name} after reranking."

        # Format results
        results = []

        for doc in reranked_docs:
            content = doc["content"]
            result_text += f"\n{content}\n"
            results.append(result_text)

        formatted_results = "\n---\n".join(results)

        return formatted_results

    except Exception as e:
        logger.error(f"Error in search_specific_document: {e}")
        return f"Error searching document {file_name}: {str(e)}"
