from langchain_core.tools import tool
from backend.shared.constants import tavily_client


@tool(parse_docstring=True, response_format="content_and_artifact")
def web_search_tool(query: str, search_depth: str = "basic"):
    """Perform a web search using Tavily and return the results.

    Args:
        query: The search query to perform.
        search_depth: The depth of the search (basic, advanced). Defaults to "basic".
    """
    try:
        response = tavily_client.search(
            query,
            max_results=10,
            auto_parameters=True,
            search_depth=search_depth,
            # topic="finance", Kaldırdım şu anlık, bazı sıkıntıları var gibi duruyor.
        )
        if not response:
            return "No search results available.", []

        # Build content text for LLM (without URLs)
        content_parts = []
        sources = []

        results = response.get("results", [])

        for r in results:
            title = r.get("title", "No title")
            content = r.get("content", "")
            url = r.get("url", "")

            content_parts.append(f"Title: {title}\nContent: {content}\n")

            if url:
                sources.append({"name": title, "url": url})

        # Return content for LLM and artifact for system
        content = "\n".join(content_parts)
        artifact = sources

        return content, artifact
    except Exception as e:
        return f"Error searching web: {str(e)}"