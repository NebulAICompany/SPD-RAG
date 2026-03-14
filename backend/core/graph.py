from typing import List, Literal, Union
from langgraph.graph import START, StateGraph
from langgraph.types import Send
from langgraph.checkpoint.memory import InMemorySaver

from backend.core.nodes import (
    document_sub_agent_node,
    orchestrator_node,
    synthesis_node,
)
from backend.core.state import AgentInputState, AgentState
from backend.shared.logger import get_logger

logger = get_logger("GRAPH")

def route_orchestrator(
    state: AgentState,
) -> Union[List[Send], Literal["synthesis_node"]]:
    """Conditional edge: fan-out to sub-agents or go to synthesis.

    Map step: if selected_documents is non-empty and global_context is empty,
    returns one Send per document so each runs the retrieval loop in parallel.
    Reduce step: otherwise routes to synthesis_node to merge all findings.

    Args:
        state: Current agent state (selected_documents, global_context, sub_agent_todos).

    Returns:
        List of Send(document_sub_agent_node, SubAgentInput) or the string "synthesis_node".
    """
    selected_docs = state.get("selected_documents", [])
    context = state.get("global_context", [])
    sub_agent_todos = state.get("sub_agent_todos", [])

    # Fan-out: one sub-agent per document when we have docs and no results yet
    logger.info(
        "Routing: selected_docs=%s, context_len=%s, todos=%s",
        selected_docs, len(context) if context else 0, len(sub_agent_todos),
    )
    if selected_docs and not context:
        return [
            Send(
                "document_sub_agent_node",
                {"document_name": doc_name, "todos": sub_agent_todos},
            )
            for doc_name in selected_docs
        ]

    # Reduce: all documents processed or none selected; proceed to synthesis
    return "synthesis_node"


def build_graph() -> StateGraph:
    """Build the SPD-RAG LangGraph: coordination -> (parallel sub-agents) -> synthesis.

    Nodes: orchestrator_node (coordination), document_sub_agent_node (retrieval),
    synthesis_node (synthesis). Conditional edge after orchestrator fans out to
    sub-agents or synthesis; sub-agents then converge to synthesis.

    Returns:
        Uncompiled StateGraph with nodes and edges defined.
    """
    workflow = StateGraph(AgentState, input=AgentInputState)

    workflow.add_node("orchestrator_node", orchestrator_node)
    workflow.add_node("document_sub_agent_node", document_sub_agent_node)
    workflow.add_node("synthesis_node", synthesis_node)


    workflow.add_edge(START, "orchestrator_node")

    workflow.add_conditional_edges(
        "orchestrator_node",
        route_orchestrator,
        ["document_sub_agent_node", "synthesis_node"],
    )

    workflow.add_edge("document_sub_agent_node", "synthesis_node")

    return workflow


_compiled_graph = None


def get_compiled_graph():
    """Return the compiled SPD-RAG graph, compiling and caching on first use.

    Shared by the FastAPI app and any LangGraph tooling (e.g. langgraph.json).
    """
    global _compiled_graph
    if _compiled_graph is None:
        workflow = build_graph()
        checkpointer = InMemorySaver()
        _compiled_graph = workflow.compile(checkpointer=checkpointer)
    return _compiled_graph


# Expose a lazy graph for langgraph.json and other consumers that expect graph.py:graph
class _LazyGraph:
    """Proxy that compiles the SPD-RAG graph on first attribute access or invocation."""
    def __getattr__(self, name):
        return getattr(get_compiled_graph(), name)
    def __call__(self, *args, **kwargs):
        return get_compiled_graph()(*args, **kwargs)

graph = _LazyGraph()
