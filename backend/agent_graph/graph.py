from typing import List, Literal, Union
from langgraph.graph import START, StateGraph
from langgraph.types import Send
from langgraph.checkpoint.memory import InMemorySaver

from backend.agent_graph.nodes import (
    clarify_intent_node,
    document_sub_agent_node,
    generate_plan_node,
    human_approval_node,
    orchestrator_node,
    synthesis_node,
    summarize_conversation_node,
)
from backend.agent_graph.state import AgentState


def route_from_start(
    state: AgentState,
) -> Literal["clarify_intent_node", "human_approval_node"]:
    """
    Determines the entry point based on current state.

    If we're awaiting user feedback on a plan (user said "yes"/"no" to approve),
    route to human_approval_node to process their response.
    Otherwise, start fresh with intent clarification.
    """
    approval_status = state.get("human_approval_status", "not_started")

    # If awaiting feedback, go to approval node to process user's response
    if approval_status == "awaiting_feedback":
        return "human_approval_node"

    # Otherwise, start fresh from intent clarification
    return "clarify_intent_node"


def route_orchestrator(
    state: AgentState,
) -> Union[List[Send], Literal["synthesis_node"]]:
    """
    Determines whether to spawn parallel sub-agents or proceed to synthesis.

    This function implements the Map-Reduce fan-out logic:
    - If there are selected documents that haven't been processed (no global_context),
      spawn a sub-agent for each document.
    - Otherwise, proceed directly to synthesis.

    Args:
        state: Current agent state with selected_documents and global_context.

    Returns:
        Either a list of Send commands for parallel execution, or "synthesis_node".
    """
    selected_docs = state.get("selected_documents", [])
    context = state.get("global_context", [])
    sub_agent_todos = state.get("sub_agent_todos", [])

    # Fan-out: Process documents in parallel if we have docs but no context yet
    if selected_docs and not context:
        return [
            Send("document_sub_agent_node", {"document_name": doc_name, "todos": sub_agent_todos})
            for doc_name in selected_docs
        ]

    # Reduce: All docs processed or no docs to process
    return "synthesis_node"


def build_graph() -> StateGraph:
    """
    Constructs and returns the AIris agent graph.

    Returns:
        Compiled StateGraph with all nodes and edges configured.
    """
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("clarify_intent_node", clarify_intent_node)
    workflow.add_node("generate_plan_node", generate_plan_node)
    workflow.add_node("human_approval_node", human_approval_node)
    workflow.add_node("orchestrator_node", orchestrator_node)
    workflow.add_node("document_sub_agent_node", document_sub_agent_node)
    workflow.add_node("synthesis_node", synthesis_node)
    workflow.add_node("summarize_conversation_node", summarize_conversation_node)

    # Start with summarization, then route based on state
    workflow.add_edge(START, "summarize_conversation_node")

    workflow.add_conditional_edges(
        "summarize_conversation_node",
        route_from_start,
        ["clarify_intent_node", "human_approval_node"],
    )

    workflow.add_conditional_edges(
        "orchestrator_node",
        route_orchestrator,
        ["document_sub_agent_node", "synthesis_node"],
    )

    workflow.add_edge("document_sub_agent_node", "synthesis_node")

    return workflow


def get_compiled_graph():
    """
    Compiles and returns the graph with the global checkpointer.
    similar to how create_agent works in agents.py
    """
    workflow = build_graph()

    checkpointer = InMemorySaver()

    return workflow.compile(checkpointer=checkpointer)

graph = get_compiled_graph()
