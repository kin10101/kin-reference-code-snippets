"""
Planner Node for Plan-Do Agent

The planner is the "brain" that breaks down a complex task into steps.
It runs ONCE at the start to create the todo list.

KEY CONCEPT:
    The planner outputs STRUCTURED data (list of tasks), not free text.
    This is achieved using structured output parsing.
    
WHY STRUCTURED OUTPUT:
    - Guarantees valid JSON
    - Enables type checking
    - Makes executor logic simple
    - Prevents parsing errors
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

from plan_do_agent.state import AgentState, Todo


# =============================================================================
# Structured Output Schema
# =============================================================================

class TodoItem(BaseModel):
    """Schema for a single task in the plan."""
    content: str = Field(description="Clear description of the task to perform")
    status: Literal["pending"] = Field(default="pending", description="Initial status")


class Plan(BaseModel):
    """Schema for the complete plan."""
    tasks: list[TodoItem] = Field(description="Ordered list of tasks to accomplish the goal")


# =============================================================================
# Planner Prompt
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are a planning agent that breaks down tasks into clear steps.

RULES:
1. Create 3-7 steps for typical tasks
2. Each step should be specific and actionable
3. Steps should be in logical order
4. Each step should be completable by a single tool or action
5. Use clear, imperative language ("Search for...", "Calculate...", "Write...")

AVAILABLE ACTIONS:
- Search the web for information
- Perform calculations
- Read files
- Write files
- Analyze information
- Summarize findings

OUTPUT:
Return a JSON list of tasks. Each task has:
- content: What needs to be done
- status: Always "pending" for new tasks
"""


# =============================================================================
# Planner Node
# =============================================================================

def create_planner(llm):
    """
    Create a planner node that generates structured task lists.
    
    Args:
        llm: The language model to use for planning
        
    Returns:
        A function that takes AgentState and returns state updates
    
    HOW IT WORKS:
        1. Takes the user's input from state
        2. Calls LLM with planning prompt
        3. Parses response into structured Plan
        4. Returns todos list for state update
    """
    # Use structured output for guaranteed JSON
    structured_llm = llm.with_structured_output(Plan)
    
    def planner(state: AgentState) -> dict:
        """
        Generate a plan from the user's input.
        
        This node runs ONCE at the start of the graph.
        It creates the todo list that the executor will work through.
        """
        print(f"\n📋 PLANNER: Creating plan for: {state['input']}")
        
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=f"Create a plan for: {state['input']}")
        ]
        
        # Get structured plan from LLM
        plan: Plan = structured_llm.invoke(messages)
        
        # Convert to Todo format for state
        todos: list[Todo] = [
            {"content": task.content, "status": task.status}
            for task in plan.tasks
        ]
        
        print(f"📋 PLANNER: Created {len(todos)} tasks:")
        for i, todo in enumerate(todos):
            print(f"   {i+1}. {todo['content']}")
        
        return {
            "todos": todos,
            "current_step": 0  # Start at first task
        }
    
    return planner


# =============================================================================
# Replanner Node (Optional)
# =============================================================================

REPLANNER_SYSTEM_PROMPT = """You are a replanning agent that adjusts plans based on progress.

Current progress:
{completed_tasks}

Remaining tasks:
{remaining_tasks}

Based on what has been learned, should the plan be adjusted?
If so, provide an updated list of remaining tasks.
If not, return the remaining tasks unchanged.
"""


def create_replanner(llm):
    """
    Create a replanner node for dynamic plan adjustment.
    
    The replanner runs after each task completes and can:
    - Add new tasks based on discoveries
    - Remove obsolete tasks
    - Reorder remaining tasks
    - Modify task descriptions
    
    This is what makes Claude Code and Devin-style agents adaptive.
    """
    structured_llm = llm.with_structured_output(Plan)
    
    def replanner(state: AgentState) -> dict:
        """
        Optionally adjust the plan based on execution progress.
        
        Called after executor completes each task.
        Can modify remaining todos based on results so far.
        """
        current_idx = state["current_step"]
        todos = state["todos"]
        
        # Separate completed and remaining tasks
        completed = [t for t in todos[:current_idx] if t["status"] == "done"]
        remaining = todos[current_idx:]
        
        # Check if replanning is needed (simple heuristic)
        if len(remaining) <= 1:
            # No point replanning with 1 or 0 tasks left
            return {}
        
        print(f"\n🔄 REPLANNER: Reviewing plan...")
        
        completed_str = "\n".join([f"✓ {t['content']}" for t in completed])
        remaining_str = "\n".join([f"- {t['content']}" for t in remaining])
        
        prompt = REPLANNER_SYSTEM_PROMPT.format(
            completed_tasks=completed_str or "None yet",
            remaining_tasks=remaining_str
        )
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Review and adjust the plan if needed.")
        ]
        
        try:
            updated_plan: Plan = structured_llm.invoke(messages)
            
            # Merge: keep completed tasks + new remaining tasks
            new_todos = completed + [
                {"content": t.content, "status": "pending"}
                for t in updated_plan.tasks
            ]
            
            print(f"🔄 REPLANNER: Plan updated with {len(updated_plan.tasks)} remaining tasks")
            
            return {"todos": new_todos}
            
        except Exception as e:
            print(f"🔄 REPLANNER: Keeping original plan (error: {e})")
            return {}
    
    return replanner
