from typing import List, Literal, Union
from langgraph.graph import START, StateGraph
from langgraph.types import Send
from langgraph.checkpoint.memory import InMemorySaver

from backend.core.nodes import (
    document_sub_agent_node,
    orchestrator_node,
    synthesis_node,
    summarize_conversation_node,
    load_uploaded_documents_node,
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
    from backend.shared.logger import get_logger
    logger = get_logger("ORCHESTRATOR")
    
    selected_docs = state.get("selected_documents", [])
    context = state.get("global_context", [])
    sub_agent_todos = state.get("sub_agent_todos", [])

    # Fan-out: Process documents in parallel if we have docs but no context yet
    if selected_docs and not context:
        logger.info("")
        logger.info(f"🔀 SPAWNING {len(selected_docs)} PARALLEL SUB-AGENTS")
        logger.info(f"   Each sub-agent will research one document independently")
        for idx, doc in enumerate(selected_docs, 1):
            logger.info(f"   Sub-agent {idx}: {doc}")
        logger.info("")
        
        return [
            Send(
                "document_sub_agent_node",
                {"document_name": doc_name, "todos": sub_agent_todos},
            )
            for doc_name in selected_docs
        ]

    # Reduce: All docs processed or no docs to process
    logger.info(f"✅ All {len(context)} document(s) processed, proceeding to synthesis")
    return "synthesis_node"


def build_graph() -> StateGraph:
    """
    Constructs and returns the AIris agent graph.

    Returns:
        Compiled StateGraph with all nodes and edges configured.
    """
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("load_uploaded_documents_node", load_uploaded_documents_node)
    workflow.add_node("orchestrator_node", orchestrator_node)
    workflow.add_node("document_sub_agent_node", document_sub_agent_node)
    workflow.add_node("synthesis_node", synthesis_node)
    workflow.add_node("summarize_conversation_node", summarize_conversation_node)

    # Start with loading documents, then summarization, then route to orchestrator
    workflow.add_edge(START, "load_uploaded_documents_node")
    workflow.add_edge("load_uploaded_documents_node", "summarize_conversation_node")
    workflow.add_edge("summarize_conversation_node", "orchestrator_node")

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
