"""
State Definitions for Plan-Do Agent

This module defines the data structures that persist across the agent's execution.

KEY CONCEPT:
    The todo list lives in LangGraph state, NOT in the prompt.
    This is the fundamental difference from ReAct agents where everything
    stays in the LLM context.

WHY THIS MATTERS:
    - LLM context has limited tokens
    - Long task lists can overflow context
    - Persistent state survives across many steps
    - State enables checkpointing and resumption
"""

from typing import Literal, Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages


# =============================================================================
# Todo: A Single Task
# =============================================================================

class Todo(TypedDict):
    """
    Represents one task in the agent's plan.
    
    Attributes:
        content: Description of what needs to be done
        status:  Current state of the task
                 - "pending":     Not yet started
                 - "in_progress": Currently being executed
                 - "done":        Completed successfully
    
    Example:
        {
            "content": "Search for Python documentation on asyncio",
            "status": "pending"
        }
    """
    content: str
    status: Literal["pending", "in_progress", "done"]


# =============================================================================
# AgentState: The Full Agent Context  
# =============================================================================

class AgentState(TypedDict):
    """
    The complete state passed between nodes in the graph.
    
    This is LangGraph's "state" concept - a TypedDict that flows through
    the graph and gets updated by each node.
    
    Attributes:
        input:        The original user query/task
        todos:        List of planned tasks with their status
        current_step: Index of the task currently being executed (0-based)
        messages:     Conversation history (optional, for chat-based interactions)
        final_result: The final output after all tasks complete
    
    Flow:
        1. User provides input
        2. Planner fills todos
        3. Executor increments current_step and updates todo status
        4. When done, final_result contains the answer
    """
    input: str
    todos: list[Todo]
    current_step: int
    messages: Annotated[list, add_messages]  # LangGraph's message accumulator
    final_result: str


# =============================================================================
# Extended State for Production Use (Optional)
# =============================================================================

class TodoExtended(TypedDict):
    """
    Extended todo structure with dependencies and metadata.
    
    Use this when you need:
    - Task dependencies (must complete X before Y)
    - Parallel execution of independent tasks  
    - Sub-agent spawning per task
    
    Example:
        {
            "id": "task_001",
            "content": "Fetch weather data",
            "status": "pending",
            "depends_on": [],
            "result": None
        }
    """
    id: str
    content: str
    status: Literal["pending", "in_progress", "done", "failed"]
    depends_on: list[str]  # IDs of tasks that must complete first
    result: str | None     # Output from this task


class AgentStateExtended(TypedDict):
    """
    Production-grade state with execution tracking.
    
    Adds:
    - Iteration counting (for loop limits)
    - Error tracking
    - Intermediate results per step
    """
    input: str
    todos: list[TodoExtended]
    current_step: int
    messages: Annotated[list, add_messages]
    final_result: str
    iteration_count: int
    max_iterations: int
    errors: list[str]
