"""
Graph Assembly for Plan-Do Agent

This module constructs the LangGraph that connects:
    Planner → Executor → (loop) → END

LANGGRAPH CONCEPTS:
    - StateGraph: Container for nodes and edges
    - Nodes: Functions that transform state
    - Edges: Connections between nodes
    - Conditional Edges: Dynamic routing based on state
    
GRAPH STRUCTURE:

    ┌─────────┐
    │  START  │
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │ Planner │ ─── Creates todo list
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │Executor │ ◄─┐ Executes one task
    └────┬────┘   │
         │        │
         ▼        │
    ┌─────────┐   │
    │ Router  │───┘ More tasks? Loop back
    └────┬────┘
         │ (done)
         ▼
    ┌─────────┐
    │   END   │
    └─────────┘
"""

from langgraph.graph import StateGraph, END

from plan_do_agent.state import AgentState
from plan_do_agent.planner import create_planner, create_replanner
from plan_do_agent.executor import (
    create_executor,
    create_synthesizer,
    should_continue,
    should_continue_with_replan
)


# =============================================================================
# Basic Agent (No Replanning)
# =============================================================================

def create_agent(llm, with_persistence: bool = False):
    """
    Create the Plan-Do agent graph.
    
    Args:
        llm: The language model to use
        with_persistence: Enable SQLite checkpointing (for resumption)
        
    Returns:
        Compiled LangGraph agent
        
    USAGE:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-4")
        agent = create_agent(llm)
        result = agent.invoke({"input": "Research Python asyncio"})
    """
    # Create nodes
    planner = create_planner(llm)
    executor = create_executor(llm)
    
    # Build graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    
    # Set entry point
    graph.set_entry_point("planner")
    
    # Connect planner → executor
    graph.add_edge("planner", "executor")
    
    # Add conditional loop: executor → executor OR executor → END
    graph.add_conditional_edges(
        "executor",           # From node
        should_continue,      # Router function
        {
            "execute": "executor",  # Loop back
            "done": END             # Finish
        }
    )
    
    # Compile with optional persistence
    if with_persistence:
        from langgraph.checkpoint.sqlite import SqliteSaver
        # Note: SqliteSaver returns a context manager
        with SqliteSaver.from_conn_string(":memory:") as checkpointer:
            return graph.compile(checkpointer=checkpointer)
    
    return graph.compile()


# =============================================================================
# Agent with Replanning
# =============================================================================

def create_agent_with_replan(llm):
    """
    Create an agent that can adjust its plan after each task.
    
    This is closer to Claude Code / Devin behavior where the agent
    can discover new tasks or skip obsolete ones during execution.
    
    GRAPH STRUCTURE:
        
        START → Planner → Executor → Replanner ─┐
                             ▲                  │
                             └──────────────────┘
                                     │
                                     ▼ (all done)
                               Synthesizer → END
    """
    # Create nodes
    planner = create_planner(llm)
    executor = create_executor(llm)
    replanner = create_replanner(llm)
    synthesizer = create_synthesizer(llm)
    
    # Build graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("replanner", replanner)
    graph.add_node("synthesizer", synthesizer)
    
    # Set entry point
    graph.set_entry_point("planner")
    
    # Planner → Executor
    graph.add_edge("planner", "executor")
    
    # Executor → Replanner OR Executor → Synthesizer
    graph.add_conditional_edges(
        "executor",
        should_continue_with_replan,
        {
            "replan": "replanner",
            "done": "synthesizer"
        }
    )
    
    # Replanner → Executor (continue loop)
    graph.add_edge("replanner", "executor")
    
    # Synthesizer → END
    graph.add_edge("synthesizer", END)
    
    return graph.compile()


# =============================================================================
# Agent with Write Todos Tool (Article Pattern)
# =============================================================================

def create_agent_with_todo_tool(llm):
    """
    Create an agent that can modify its todo list via a tool.
    
    This matches the TodoListMiddleware pattern from the article where
    the agent has a write_todos tool to update its task list during execution.
    
    Instead of automatic replanning, the agent CHOOSES when to update todos.
    """
    from langchain_core.tools import tool
    from langchain_core.messages import AIMessage
    
    # Create a closure to hold mutable todo state
    class TodoState:
        todos: list = []
        current_step: int = 0
    
    todo_state = TodoState()
    
    @tool
    def write_todos(tasks: list[str]) -> str:
        """
        Update the todo list with new tasks. 
        Use this to add, modify, or reorder tasks.
        
        Args:
            tasks: List of task descriptions to set as the new todo list
        """
        todo_state.todos = [
            {"content": task, "status": "pending"}
            for task in tasks
        ]
        return f"Updated todo list with {len(tasks)} tasks"
    
    @tool
    def get_todos() -> str:
        """Get the current todo list with status."""
        if not todo_state.todos:
            return "No todos set"
        lines = []
        for i, todo in enumerate(todo_state.todos):
            status_icon = "✓" if todo["status"] == "done" else "○"
            lines.append(f"{status_icon} {i+1}. {todo['content']}")
        return "\n".join(lines)
    
    @tool
    def mark_done(task_number: int) -> str:
        """Mark a task as done by its number (1-indexed)."""
        idx = task_number - 1
        if 0 <= idx < len(todo_state.todos):
            todo_state.todos[idx]["status"] = "done"
            return f"Marked task {task_number} as done"
        return f"Invalid task number: {task_number}"
    
    # Bind tools to LLM
    tools = [write_todos, get_todos, mark_done]
    llm_with_tools = llm.bind_tools(tools)
    
    # This creates a ReAct-style agent with todo tools
    # It's a hybrid: ReAct loop + explicit todo management
    from langgraph.prebuilt import create_react_agent
    
    agent = create_react_agent(
        llm_with_tools,
        tools,
        state_modifier="""You are a task-oriented agent. 
Always start by creating a todo list with write_todos.
Execute tasks one by one, marking them done with mark_done.
Check progress with get_todos.
Complete all tasks before finishing."""
    )
    
    return agent


# =============================================================================
# Visualize Graph (for debugging)
# =============================================================================

def print_graph_structure(graph):
    """Print the graph structure for debugging."""
    print("\n📊 GRAPH STRUCTURE:")
    print("=" * 40)
    
    compiled = graph if hasattr(graph, 'nodes') else graph
    
    if hasattr(compiled, 'nodes'):
        print("Nodes:")
        for node in compiled.nodes:
            print(f"  - {node}")
    
    print("=" * 40)
