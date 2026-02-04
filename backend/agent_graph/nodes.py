from datetime import datetime
from typing import Any, Dict, List, Literal
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
    RemoveMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel, Field
from backend.agent_graph.prompts import (
    CLARIFY_WITH_USER_INSTRUCTIONS,
    FINAL_REPORT_GENERATION_PROMPT,
    LEAD_RESEARCHER_PROMPT,
    TRANSFORM_MESSAGES_INTO_PLAN_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
)
from backend.agent_graph.state import AgentState, Plan, SubAgentInput, Summary, TodoItem
from backend.shared.constants import RESEARCH_LLM_REASONING, RESEARCH_LLM_FAST
from backend.core.tools.rag import search_specific_document_for_research
from backend.core.tools.api import web_search_tool
from backend.core.tools.skills import load_skill


def get_today_str() -> str:
    """Returns today's date as a formatted string (YYYY-MM-DD)."""
    return datetime.now().strftime("%Y-%m-%d")


def format_todos_as_string(todos: List[TodoItem]) -> str:
    """Formats a list of TodoItems as a readable string for prompts."""
    if not todos:
        return "No tasks defined yet."
    return "\n".join([f"- {t.task} [{t.status}]" for t in todos])


class AmbiguityCheck(BaseModel):
    """Model for structured output from ambiguity check."""

    is_ambiguous: bool = Field(
        description="True if the user request is vague or needs clarification"
    )
    clarifying_question: str = Field(
        description="The question to ask the user if ambiguous, else empty string"
    )


class WriteTodos(BaseModel):
    """Tool for updating the Todo list during orchestration."""

    todos: List[TodoItem] = Field(
        description="The full updated list of todo items with their current statuses."
    )
    sub_agent_todos: List[TodoItem] = Field(
        description="A list of specific tasks that EVERY sub-agent must execute for their assigned document. Each item must have a 'task' and a 'status' (default 'pending')."
    )


class PlanApprovalCheck(BaseModel):
    """Model for structured output to detect plan approval from user messages."""

    is_approved: bool = Field(
        description="True if the user has approved the plan (e.g., 'yes', 'looks good', 'proceed', 'that's great')"
    )
    wants_changes: bool = Field(description="True if the user wants to modify the plan")
    feedback: str = Field(
        description="The user's feedback or requested changes if any, else empty string"
    )


async def clarify_intent_node(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["generate_plan_node", END]]:
    """
    Analyzes the user's message to determine if clarification is needed.

    If the query is ambiguous, pauses execution (END) and sends a clarifying
    question to the user. Otherwise, proceeds to the planning phase.

    Args:
        state: Current agent state containing messages.
        config: Runtime configuration.

    Returns:
        Command routing to either END (wait for user) or generate_plan_node.
    """
    messages = state["messages"]
    selected_docs = state.get("selected_documents", [])

    # Build file context info
    file_context = ""
    if selected_docs:
        file_names = [doc.split("/")[-1].split("\\")[-1] for doc in selected_docs]
        file_context = f"\n\n**Available Files:** The user has already selected these files for analysis: {', '.join(file_names)}. Do NOT ask for files - they are already available."

    checker = RESEARCH_LLM_REASONING.with_structured_output(AmbiguityCheck)
    prompt_content = (
        CLARIFY_WITH_USER_INSTRUCTIONS.format(
            messages=get_buffer_string(messages), date=get_today_str()
        )
        + file_context
    )

    result = await checker.ainvoke([HumanMessage(content=prompt_content)])

    if result.is_ambiguous:
        return Command(
            goto=END,
            update={
                "is_ambiguous": True,
                "messages": [AIMessage(content=result.clarifying_question)],
            },
        )

    return Command(goto="generate_plan_node", update={"is_ambiguous": False})


async def generate_plan_node(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["human_approval_node"]]:
    """
    Generates a strategic plan based on the user's request.

    Creates a Plan with strategy, steps (TodoItems), and reasoning.
    Outputs a user-facing summary for approval.

    Args:
        state: Current agent state containing messages.
        config: Runtime configuration.

    Returns:
        Command routing to human_approval_node with plan in state.
    """
    messages = state["messages"]
    selected_docs = state.get("selected_documents", [])

    # Build file context info
    file_context = ""
    if selected_docs:
        file_names = [doc.split("/")[-1].split("\\")[-1] for doc in selected_docs]
        file_context = f"\n\n**Available Files for Analysis:** {', '.join(file_names)}. Include steps to analyze these documents in your plan."

    planner = RESEARCH_LLM_REASONING.with_structured_output(Plan)
    prompt_content = (
        TRANSFORM_MESSAGES_INTO_PLAN_PROMPT.format(
            messages=get_buffer_string(messages), date=get_today_str()
        )
        + file_context
    )

    plan = await planner.ainvoke([HumanMessage(content=prompt_content)])

    # Format plan summary for user approval
    steps_str = "\n".join([f"- {step.task}" for step in plan.steps])
    plan_summary_msg = (
        f"**Proposed Plan:**\n"
        f"**Strategy:** {plan.strategy}\n\n"
        f"**Steps:**\n{steps_str}\n\n"
        f"**Reasoning:** {plan.reasoning}\n\n"
        f"Do you approve this plan? (yes/no)"
    )

    return Command(
        goto="human_approval_node",
        update={
            "plan": plan,
            "todo_queue": plan.steps,
            "messages": [AIMessage(content=plan_summary_msg)],
            "human_approval_status": "pending",
        },
    )


async def human_approval_node(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["orchestrator_node", "generate_plan_node", END]]:
    """
    Uses LLM to analyze user's response to determine plan approval.

    This node uses conversation-based approval detection:
    - If user approves (says things like 'yes', 'proceed', 'looks good'), routes to orchestrator
    - If user wants changes, routes back to planning with feedback
    - If this is the first time showing the plan (no user response yet), waits for input

    Args:
        state: Current agent state.
        config: Runtime configuration.

    Returns:
        Command routing based on LLM's analysis of user's approval status.
    """
    messages = state.get("messages", [])
    approval_status = state.get("human_approval_status", "pending")

    # If plan was just generated and we haven't waited for user input yet,
    # end and wait for the user's response
    if approval_status == "pending":
        return Command(goto=END, update={"human_approval_status": "awaiting_feedback"})

    # We have user feedback - use LLM to analyze if they approved
    checker = RESEARCH_LLM_FAST.with_structured_output(PlanApprovalCheck)

    # Get the last few messages for context
    recent_messages = messages[-4:] if len(messages) > 4 else messages
    messages_str = get_buffer_string(recent_messages)

    prompt = f"""Analyze the following conversation to determine if the user has approved the proposed plan.

Conversation:
{messages_str}

Determine:
1. Has the user approved the plan? (e.g., 'yes', 'proceed', 'looks good', 'that's great', 'go ahead')
2. Does the user want to make changes to the plan?
3. What feedback or changes did they request (if any)?
"""

    result = await checker.ainvoke([HumanMessage(content=prompt)])

    if result.is_approved and not result.wants_changes:
        return Command(
            goto="orchestrator_node", update={"human_approval_status": "approved"}
        )

    if result.wants_changes:
        return Command(
            goto="generate_plan_node",
            update={
                "human_approval_status": "pending",
                "messages": [HumanMessage(content=f"User feedback: {result.feedback}")],
            },
        )

    # User hasn't clearly approved or rejected - ask for clarification
    return Command(
        goto=END,
        update={
            "human_approval_status": "awaiting_feedback",
            "messages": [
                AIMessage(
                    content="I've proposed a plan above. Would you like me to proceed with this plan, or would you like to make any changes?"
                )
            ],
        },
    )


async def orchestrator_node(
    state: AgentState, config: RunnableConfig
) -> Dict[str, Any]:
    """
    Manages the TODO list and decides on task delegation.

    Implements the TodoListMiddleware pattern: binds a WriteTodos tool
    to the LLM and updates state based on the tool's output.

    Args:
        state: Current agent state with plan and todos.
        config: Runtime configuration.

    Returns:
        State updates including messages and potentially updated todo_queue.
    """
    plan = state.get("plan")
    todos = state.get("todo_queue", [])
    messages = state["messages"]

    llm_with_tools = RESEARCH_LLM_REASONING.bind_tools([WriteTodos, web_search_tool, load_skill], tool_choice="auto")

    system_prompt = LEAD_RESEARCHER_PROMPT.format(date=get_today_str())
    context_prompt = f"""
Current Strategy: {plan.strategy if plan else 'N/A'}

Current TODO List:
{format_todos_as_string(todos)}
"""

    full_prompt = system_prompt + context_prompt
    response = await llm_with_tools.ainvoke(
        [{"role": "system", "content": full_prompt}] + messages
    )

    updates: Dict[str, Any] = {"messages": [response]}

    if response.tool_calls:
        tool_outputs = []
        for tool_call in response.tool_calls:
            if tool_call["name"] == "WriteTodos":
                new_todos_raw = tool_call["args"].get("todos", [])
                sub_todos_raw = tool_call["args"].get("sub_agent_todos", [])
                updates["todo_queue"] = [TodoItem(**t) for t in new_todos_raw]
                if sub_todos_raw:
                    updates["sub_agent_todos"] = [TodoItem(**t) for t in sub_todos_raw]
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
            elif tool_call["name"] == "load_skill":
                tool_output = await load_skill.ainvoke(tool_call["args"])
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
        todos_str = "\n".join([f"{i+1}. [ ] {t.task if hasattr(t, 'task') else t.get('task')}" for i, t in enumerate(todos_list)])
    else:
        todos_str = "No specific sub-tasks provided."

    model_with_tools = RESEARCH_LLM_FAST.bind_tools(
        [search_specific_document_for_research]
    )

    system_prompt = RESEARCH_SYSTEM_PROMPT.format(
        date=get_today_str(), file_name=doc_name
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Research Topic: {doc_name}\n\n**Orchestrator Assigned Tasks**:\nYou must address the following points about this document:\n{todos_str}"),
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

    if global_context:
        findings_str = "\n\n".join(
            [
                f"**Document {s.document_name}** (Relevance: {s.relevance_score:.2f}):\n{s.findings}"
                for s in global_context
            ]
        )
    else:
        findings_str = "No document findings available."

    prompt_content = FINAL_REPORT_GENERATION_PROMPT.format(
        messages=get_buffer_string(state.get("messages", [])),
        findings=findings_str,
        date=get_today_str(),
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