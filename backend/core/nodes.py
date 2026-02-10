import asyncio
from typing import Any, Dict, List, Literal, Optional
import tiktoken
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    get_buffer_string,
    RemoveMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel, Field

from backend.core.prompts import (
    FINAL_REPORT_GENERATION_PROMPT,
    INTERMEDIATE_SYNTHESIS_PROMPT,
    LEAD_RESEARCHER_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
)
from backend.core.state import AgentState, SubAgentInput, Summary, TodoItem
from backend.shared.constants import RESEARCH_LLM_REASONING, RESEARCH_LLM_FAST
from backend.core.tools.rag import search_specific_document_for_research


_ENCODER: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    """Lazily initialise and return a shared tiktoken encoder.

    Uses cl100k_base, which is compatible with GPT-4/4.1/4o/5-style models
    and already used elsewhere in this project for accurate token counting.
    """
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def _estimate_tokens(text: str) -> int:
    """Estimate token count using the cl100k_base tokenizer.

    This aligns with existing project usage (vector pipeline, benchmarks)
    and is much closer to actual model tokenisation than word-count
    heuristics.
    """
    if not text:
        return 0
    return len(_get_encoder().encode(text))


async def _summarize_batch_findings(
    findings_batch: List[str],
    root_query: str,
) -> str:
    """Summarize a batch of findings into a single merged summary."""
    batch_text = "\n\n---\n\n".join(findings_batch)
    prompt_content = INTERMEDIATE_SYNTHESIS_PROMPT.format(
        findings=batch_text,
        query=root_query,
    )

    response = await RESEARCH_LLM_REASONING.ainvoke(
        [HumanMessage(content=prompt_content)]
    )
    return getattr(response, "content", str(response))


async def recursive_summarize_findings(
    raw_findings: List[str],
    root_query: str,
    target_batch_tokens: int = 1200,
) -> str:
    """Recursively merge many findings into a single summary.
    - Groups findings into batches based on approximate token count.
    - Summarizes each batch.
    - Repeats on the new set of summaries until only one remains.
    """
    current_level = raw_findings

    while len(current_level) > 1:
        batches: List[List[str]] = []
        current_batch: List[str] = []
        current_tokens = 0

        for chunk in current_level:
            chunk_tokens = _estimate_tokens(chunk)

            if current_batch and current_tokens + chunk_tokens > target_batch_tokens:
                batches.append(current_batch)
                current_batch = [chunk]
                current_tokens = chunk_tokens
            else:
                current_batch.append(chunk)
                current_tokens += chunk_tokens

        if current_batch:
            batches.append(current_batch)

        tasks = [
            _summarize_batch_findings(batch, root_query=root_query)
            for batch in batches
            if batch
        ]
        next_level = await asyncio.gather(*tasks)

        current_level = list(next_level)

    return current_level[0]


def format_todos_as_string(todos: List[TodoItem]) -> str:
    """Formats a list of TodoItems as a readable string for prompts."""
    if not todos:
        return "No tasks defined yet."
    return "\n".join([f"- {t.task} [{t.status}]" for t in todos])


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

    system_prompt = LEAD_RESEARCHER_PROMPT
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
        input_data: SubAgentInput containing the document_name (used as research topic).

    Returns:
        State update with the Summary added to global_context.
    """
    from langchain_core.messages import ToolMessage

    doc_name = input_data["document_name"]
    todos_list = input_data.get("todos", [])

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

    system_prompt = RESEARCH_SYSTEM_PROMPT.format(file_name=doc_name)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Research Topic: {doc_name}\n\n**Orchestrator Assigned Tasks**:\nYou must address the following points about this document:\n{todos_str}"
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

    extractor = RESEARCH_LLM_FAST.with_structured_output(Summary)
    summary = await extractor.ainvoke(messages)

    summary.document_name = doc_name

    return {"global_context": [summary]}


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
    selected_docs = state.get("selected_documents", [])

    if len(global_context) < len(selected_docs):
        return Command(goto=END)

    root_query = ""
    for msg in state.get("messages", []):
        if isinstance(msg, HumanMessage):
            root_query = getattr(msg, "content", "")
            break

    if global_context:
        raw_findings_chunks: List[str] = [
            (
                f"Document: {s.document_name}\n"
                f"Relevance: {s.relevance_score:.2f}\n"
                f"Findings:\n{s.findings}"
            )
            for s in global_context
        ]

        merged_findings = await recursive_summarize_findings(
            raw_findings_chunks,
            root_query=root_query,
        )
    else:
        merged_findings = "No document findings available."

    prompt_content = FINAL_REPORT_GENERATION_PROMPT.format(
        messages=get_buffer_string(state.get("messages", [])),
        findings=merged_findings,
    )

    response = await RESEARCH_LLM_REASONING.ainvoke(
        [HumanMessage(content=prompt_content)]
    )

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
