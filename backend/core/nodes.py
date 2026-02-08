from typing import Any, Dict, List, Literal
import os
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    RemoveMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel, Field
from backend.core.prompts import (
    FINAL_REPORT_GENERATION_PROMPT,
    LEAD_RESEARCHER_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
)
from backend.core.state import AgentState, SubAgentInput, Summary, TodoItem
from backend.shared.constants import RESEARCH_LLM_REASONING, RESEARCH_LLM_FAST, UPLOADS_PATH_STR
from backend.core.tools.rag import search_specific_document_for_research
from backend.shared.logger import get_logger
from backend.retrieval.retriever import get_vectorstore

logger = get_logger("NODES")



def format_todos_as_string(todos: List[TodoItem]) -> str:
    """Formats a list of TodoItems as a readable string for prompts."""
    if not todos:
        return "No tasks defined yet."
    return "\n".join([f"- {t.task} [{t.status}]" for t in todos])


def load_uploaded_documents_node(state: AgentState) -> Dict[str, Any]:
    """
    Load unique chunk identifiers from Qdrant instead of filesystem files.

    This node scans all points in the configured Qdrant collection,
    extracts the unique `chunk_id` values from payloads, and returns them
    as `selected_documents` so that downstream nodes can process each chunk
    in isolation.

    Args:
        state: Current agent state.

    Returns:
        State update with selected_documents populated from unique chunk IDs.
    """
    try:
        COLLECTION_NAME = "documents"
        CHUNK_ID_KEY = "chunk_id"
        unique_chunk_ids = set()
        scroll_offset = None

        client = get_vectorstore()
        if client is None:
            logger.error("Vectorstore not initialized. Please ensure documents are uploaded first.")
            return {"selected_documents": []}

        while True:
            points, scroll_offset= client.scroll(
                collection_name=COLLECTION_NAME,
                offset=scroll_offset,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                break

            for p in points:
                payload = p.payload or {}
                chunk_id = payload.get("metadata", {}).get("chunk_id")
                if chunk_id is not None:
                    unique_chunk_ids.add(chunk_id)

            if scroll_offset is None:
                break

        chunk_id_list = sorted(unique_chunk_ids)

        if not chunk_id_list:
            logger.warning("⚠️ No chunk IDs found in collection payloads")
            return {"selected_documents": []}

        logger.info("")
        logger.info(f"✅ Found {len(chunk_id_list)} unique chunk ID(s) to process:")

        return {"selected_documents": chunk_id_list}

    except Exception as e:
        logger.error(f"Error loading unique chunk IDs from Qdrant: {e}")
        return {"selected_documents": []}


class WriteTodos(BaseModel):
    """Tool for updating the Todo list during orchestration."""

    todos: List[TodoItem] = Field(
        description="The full updated list of todo items with their current statuses."
    )
    sub_agent_todos: List[TodoItem] = Field(
        description="A list of specific tasks that EVERY sub-agent must execute for their assigned document. Each item must have a 'task' and a 'status' (default 'pending')."
    )


async def orchestrator_node(
    state: AgentState, config: RunnableConfig
) -> Dict[str, Any]:
    """
    Manages the TODO list and decides on task delegation.

    Implements the TodoListMiddleware pattern: binds a WriteTodos tool
    to the LLM and updates state based on the tool's output.

    Args:
        state: Current agent state with todos.
        config: Runtime configuration.

    Returns:
        State updates including messages and potentially updated todo_queue.
    """
    todos = state.get("todo_queue", [])
    messages = state["messages"]

    llm_with_tools = RESEARCH_LLM_REASONING.bind_tools([WriteTodos], tool_choice="auto")

    # Build context metadata (RLM paper: root LM receives type, length, chunk info)
    selected_docs = state.get("selected_documents", [])
    if selected_docs:
        context_description = (
            f"a long document split into {len(selected_docs)} chunks: "
            f"{', '.join(selected_docs)}"
        )
    else:
        context_description = "not yet loaded into the environment"

    system_prompt = LEAD_RESEARCHER_PROMPT.format(
        context_description=context_description,
    )
    context_prompt = f"""
        Current TODO List:
        {format_todos_as_string(todos)}
        """

    full_prompt = system_prompt + context_prompt
    response = await llm_with_tools.ainvoke(
        [{"role": "system", "content": full_prompt}] + messages
    )

    updates: Dict[str, Any] = {"messages": [response]}

    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "WriteTodos":
                new_todos_raw = tool_call["args"].get("todos", [])
                sub_todos_raw = tool_call["args"].get("sub_agent_todos", [])
                updates["todo_queue"] = [TodoItem(**t) for t in new_todos_raw]
                if sub_todos_raw:
                    updates["sub_agent_todos"] = [TodoItem(**t) for t in sub_todos_raw]
                tool_outputs.append(
                    ToolMessage(
                        content="Todos updated",  # istersen burada structured içerik dönebilirsin
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                )
            elif tool_call["name"] == "web_search_tool":
                tool_output = await web_search_tool.ainvoke(tool_call["args"])
                if isinstance(tool_output, tuple):
                    content, _ = tool_output
                else:
                    content = str(tool_output)

                tool_outputs.append(
                    ToolMessage(
                        content=str(content),
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                )

        if tool_outputs:
            updates["messages"].extend(tool_outputs)

    return updates


async def document_sub_agent_node(input_data: SubAgentInput) -> Dict[str, Any]:
    """
    Processes a document query using an agentic RAG workflow.

    This node acts as a specialized sub-agent that:
    1. Receives a topic/document ID.
    2. Uses the LLM to formulate a search query for the `search_local_documents` tool.
    3. Executes the tool to retrieve information.
    4. Synthesizes the findings into a structured Summary.

    Args:
        input_data: SubAgentInput containing the chunk_id (used as research topic).

    Returns:
        State update with the Summary added to global_context.
    """
    from langchain_core.messages import ToolMessage

    chunk_id = input_data["chunk_id"]
    todos_list = input_data.get("todos", [])
    
    logger.info(f"🔎 Sub-agent analyzing document: '{chunk_id}'...")

    # Format TodoItems into a string list for the prompt
    # todos_list is a list of TodoItem objects (or dicts if not pushed as objects)
    todos_str = ""
    if todos_list:
        todos_str = "\n".join(
            [
                f"{i+1}. [ ] {t.task if hasattr(t, 'task') else t.get('task')}"
                for i, t in enumerate(todos_list)
            ]
        )
    else:
        todos_str = "No specific sub-tasks provided."

    model_with_tools = RESEARCH_LLM_FAST.bind_tools(
        [search_specific_document_for_research]
    )
    system_prompt = RESEARCH_SYSTEM_PROMPT.format(
        date=get_today_str(), file_name=chunk_id
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Research Topic: {chunk_id}\n\n**Orchestrator Assigned Tasks**:\nYou must address the following points about this document:\n{todos_str}"
        ),
    ]

    ai_msg = await model_with_tools.ainvoke(messages)
    messages.append(ai_msg)

    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            if tool_call["name"] == "search_specific_document":
                tool_output = await search_specific_document_for_research.ainvoke(
                    tool_call["args"]
                )

                if isinstance(tool_output, tuple):
                    content, _ = tool_output
                else:
                    content = str(tool_output)

                messages.append(
                    ToolMessage(
                        content=str(content),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
    else:
        pass

    # Extract structured summary with better error handling
    try:
        extractor = RESEARCH_LLM_FAST.with_structured_output(
            Summary,
            include_raw=False,
        )
        summary = await extractor.ainvoke(messages)

        summary.chunk_id = chunk_id
        logger.info(f"✅ Sub-agent completed analysis of '{chunk_id}' (Relevance: {summary.relevance_score:.2f})")

        return {"global_context": [summary]}
        
    except Exception as e:
        logger.error(f"❌ Error extracting summary for '{chunk_id}': {e}")
        # Return a fallback summary
        fallback_summary = Summary(
            chunk_id=chunk_id,
            findings="Error processing document. Please try again.",
            relevance_score=0.0
        )
        return {"global_context": [fallback_summary]}


async def synthesis_node(
    state: AgentState, config: RunnableConfig
) -> Command[Literal[END]]:
    """
    Aggregates all research findings and generates the final report.

    Combines summaries from global_context into a comprehensive response.

    Args:
        state: Current agent state with global_context populated.
        config: Runtime configuration.

    Returns:
        Command routing to END with final report in messages.
    """
    global_context = state.get("global_context", [])
    selected_chunks = state.get("selected_chunks", [])
    
    logger.info("")
    logger.info("📦 SYNTHESIS PHASE - Combining all research findings...")
    logger.info(f"   Processing {len(global_context)} chunk summaries")

    if len(global_context) < len(selected_chunks):
        return Command(goto=END)

    if global_context:
        findings_str = "\n\n".join(
            [
                f"**Chunk {s.chunk_id}** (Relevance: {s.relevance_score:.2f}):\n{s.findings}"
                for s in global_context
            ]
        )
    else:
        findings_str = "No document findings available."

    # Extract the original user query — the first HumanMessage in the conversation
    messages = state.get("messages", [])
    original_query = ""
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content:
            original_query = msg.content
            break

    prompt_content = FINAL_REPORT_GENERATION_PROMPT.format(
        query=original_query,
        findings=findings_str,
    )

    response = await RESEARCH_LLM_REASONING.ainvoke(
        [HumanMessage(content=prompt_content)]
    )
    
    logger.info("✅ Final synthesis complete - generating response...")
    return Command(goto=END, update={"messages": [response]})


async def summarize_conversation_node(
    state: AgentState, config: RunnableConfig
) -> Dict[str, Any]:
    """
    Summarizes the conversation history if it exceeds a certain length.

    Args:
        state: Current agent state.
        config: Runtime configuration.

    Returns:
        State updates with new summary and removal commands for old messages.
    """
    logger.info("📝 Summarizing conversation history...")
    messages = state.get("messages", [])

    # Check if we have enough messages to warrant summarization
    # We keep the last 4 messages to preserve immediate context for the next steps
    if len(messages) > 6:
        summary = state.get("summary", "")

        # Create summarization prompt
        if summary:
            summary_message = (
                f"This is a summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        # We summarize the messages that we are about to remove
        # e.g., if we have 10 messages, we summarize first 6, keep last 4
        messages_to_summarize = messages[:-4]

        # Invoke LLM to generate summary
        # We construct a temporary message list for the summarization task
        prompt_messages = messages_to_summarize + [
            HumanMessage(content=summary_message)
        ]
        response = await RESEARCH_LLM_FAST.ainvoke(prompt_messages)

        # Create RemoveMessage commands for the messages we summarized
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]

        return {"summary": response.content, "messages": delete_messages}

    return {}
