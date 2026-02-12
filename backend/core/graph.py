from typing import List, Literal, Union
from langgraph.graph import START, StateGraph
from langgraph.types import Send
from langgraph.checkpoint.memory import InMemorySaver

from backend.core.nodes import (
    document_sub_agent_node,
    orchestrator_node,
    synthesis_node,
    summarize_conversation_node,
)
from backend.core.state import AgentState


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
            Send(
                "document_sub_agent_node",
                {"document_name": doc_name, "todos": sub_agent_todos},
            )
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
    workflow.add_node("orchestrator_node", orchestrator_node)
    workflow.add_node("document_sub_agent_node", document_sub_agent_node)
    workflow.add_node("synthesis_node", synthesis_node)
    workflow.add_node("summarize_conversation_node", summarize_conversation_node)

    # Start with summarization, then route to orchestrator
    workflow.add_edge(START, "summarize_conversation_node")
    workflow.add_edge("summarize_conversation_node", "orchestrator_node")

    workflow.add_conditional_edges(
        "orchestrator_node",
        route_orchestrator,
        ["document_sub_agent_node", "synthesis_node"],
    )

    workflow.add_edge("document_sub_agent_node", "synthesis_node")

    return workflow


_compiled_graph = None


def get_compiled_graph():
    """
    Returns (and caches) the compiled graph as a lazy singleton.
    Both langgraph.json and main.py share the same instance.
    """
    global _compiled_graph
    if _compiled_graph is None:
        workflow = build_graph()
        checkpointer = InMemorySaver()
        _compiled_graph = workflow.compile(checkpointer=checkpointer)
    return _compiled_graph


# Lazy property for langgraph.json (expects graph.py:graph)
class _LazyGraph:
    """Lazy wrapper so `from graph import graph` works without eager compilation."""
    def __getattr__(self, name):
        return getattr(get_compiled_graph(), name)
    def __call__(self, *args, **kwargs):
        return get_compiled_graph()(*args, **kwargs)

graph = _LazyGraph()
