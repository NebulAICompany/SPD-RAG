from typing import Annotated, List, Literal, Optional
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


def merge_summaries(
    left: Optional[List["Summary"]], right: List["Summary"]
) -> List["Summary"]:
    """Reducer: append new sub-agent summaries to the global context.

    Used as the reducer for the global_context state key so that parallel
    sub-agent results are merged without overwriting (LangGraph Send semantics).

    Args:
        left: Existing list of Summary objects (may be None when no prior context).
        right: New Summary objects from one or more document sub-agents.

    Returns:
        Combined list of all summaries for the synthesis layer.
    """
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right


class TodoItem(BaseModel):
    """One atomic extraction task from the Shared Instruction Set.

    Every sub-agent executes the same list of TodoItems for its assigned document.
    """

    task: str = Field(description="What to extract (e.g. a specific metric, definition, or claim).")
    status: str = Field(
        default="pending",
        description="Task status: 'pending', 'in_progress', or 'completed'.",
    )


class Summary(BaseModel):
    """Output of one document sub-agent: findings from a single document.

    Produced when the sub-agent issues action='finalize'. Collected in
    global_context and consumed by the synthesis layer.
    """

    document_name: str = Field(description="Name of the document that was searched.")
    findings: str = Field(description="Extracted findings for the assigned tasks (raw facts, no synthesis).")


class AgentAction(BaseModel):
    """Structured output from the sub-agent each retrieval-loop turn.

    The runner (document_sub_agent_node) interprets this: on 'search' it runs
    the document-scoped RAG tool and appends results; on 'finalize' it pushes
    a Summary to global_context. Tools are not bound to the LLM.
    """

    action: Literal["search", "finalize"] = Field(
        description="'search' to run another retrieval query; 'finalize' when extraction is complete."
    )
    query: Optional[str] = Field(
        None,
        description="Search query string (required when action='search').",
    )
    reasoning: str = Field(
        description="Brief justification for this action.",
    )
    findings: Optional[str] = Field(
        None,
        description="Full extracted findings from the document (required when action='finalize').",
    )


class AgentInputState(MessagesState):
    """Input schema for the SPD-RAG graph invocation.

    Extends MessagesState with selected_documents (names of documents to query).
    Passed as input to graph.invoke() / graph.ainvoke().
    """
    selected_documents: List[str] = []


class AgentState(MessagesState):
    """State carried through the SPD-RAG graph (coordination, retrieval, synthesis).

    Inherits messages from MessagesState (with add_messages reducer). After the
    coordination layer: sub_agent_todos (Shared Instruction Set) and
    synthesis_directive are set. After the parallel retrieval layer: global_context
    holds one Summary per document. The synthesis layer reads global_context and
    synthesis_directive to produce the final response.
    """

    global_context: Annotated[List[Summary], merge_summaries] = []
    sub_agent_todos: List[TodoItem] = []
    selected_documents: List[str] = []
    synthesis_directive: str = ""


class SubAgentInput(TypedDict):
    """Payload for one document sub-agent, sent via LangGraph Send for parallel execution.

    Each Send carries the document name and the Shared Instruction Set (todos)
    so the sub-agent runs an isolated retrieval loop on that document only.
    """

    document_name: str
    todos: List[TodoItem]
