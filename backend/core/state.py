from typing import Annotated, List, Optional
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


def override_reducer(current_value, new_value):
    """
    Reducer function that allows overriding values in state.

    Supports three modes:
    1. Explicit override: {"type": "override", "value": <new_value>}
    2. List append: Both values are lists -> concatenate
    3. Replacement: Default behavior for non-list types

    Args:
        current_value: The existing value in state.
        new_value: The new value to merge or replace.

    Returns:
        The merged or replaced value.
    """
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    if isinstance(current_value, list) and isinstance(new_value, list):
        return current_value + new_value
    return new_value


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
        description="Relevance confidence score (0.0 - 1.0)", ge=0.0, le=1.0
    )


class AgentInputState(MessagesState):
    """
    Input state schema for the agent graph.

    Inherits only the 'messages' key from MessagesState.
    Used to define the expected input structure for `graph.invoke()`.
    """

    pass


class AgentState(MessagesState):
    """
    Main agent state containing messages and global context.

    Inherits the 'messages' key from MessagesState, which is annotated
    with the `add_messages` reducer for automatic message handling.

    Attributes:
        todo_queue: The current list of tasks being tracked.
        selected_documents: Document IDs selected for sub-agent processing.
        global_context: Aggregated summaries from all sub-agents.
    """

    todo_queue: Annotated[List[TodoItem], override_reducer]
    global_context: Annotated[List[Summary], merge_summaries]
    summary: str
    sub_agent_todos: List[TodoItem]


class SubAgentInput(TypedDict):
    """
    Input schema for the document sub-agent node.

    Passed via LangGraph's `Send` API for parallel execution.

    Attributes:
        document_name: The name of the document to process.
    """

    document_name: str
    todos: List[TodoItem]
