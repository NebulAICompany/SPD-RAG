from typing import Annotated, List, Optional
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


def merge_summaries(
    left: Optional[List["Summary"]], right: List["Summary"]
) -> List["Summary"]:
    """
    Merges new summaries into the global context.

    Args:
        left: Existing list of summaries (can be None).
        right: New summaries to append.

    Returns:
        Combined list of summaries.
    """
    if left is None:
        left = []
    if right is None:
        right = []
    return left + right


class TodoItem(BaseModel):
    """Represents a single actionable task."""

    task: str = Field(description="The specific task to be executed")
    status: str = Field(
        default="pending",
        description="Status of the task: 'pending', 'in_progress', or 'completed'",
    )


class Summary(BaseModel):
    """Research summary extracted from a single document by a sub-agent."""

    document_name: str = Field(description="The source document name")
    findings: str = Field(description="Extracted relevant information and analysis")
    relevance_score: float = Field(
        default=0.0,
        description="Relevance confidence score (0.0 - 1.0)", ge=0.0, le=1.0
    )


class AgentInputState(MessagesState):
    """
    Input state schema for the agent graph.

    Inherits the 'messages' key from MessagesState.
    Used to define the expected input structure for `graph.invoke()`.
    """

    selected_documents: List[str] = []


class AgentState(MessagesState):
    """
    Main agent state containing messages and global context.

    Inherits the 'messages' key from MessagesState, which is annotated
    with the `add_messages` reducer for automatic message handling.

    Attributes:
        selected_documents: Document IDs selected for sub-agent processing.
        global_context: Aggregated summaries from all sub-agents.
        sub_agent_todos: Tasks delegated to sub-agents for document processing.
    """

    global_context: Annotated[List[Summary], merge_summaries] = []
    sub_agent_todos: list = []
    selected_documents: List[str] = []


class SubAgentInput(TypedDict):
    """
    Input schema for the document sub-agent node.

    Passed via LangGraph's `Send` API for parallel execution.

    Attributes:
        document_name: The name of the document to process.
    """

    document_name: str
    todos: List[TodoItem]
