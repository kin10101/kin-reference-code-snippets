"""
Executor Node for Plan-Do Agent

The executor runs ONE task at a time from the todo list.
It's called repeatedly until all tasks are done.

KEY CONCEPT:
    The executor doesn't know the whole plan - it just executes
    the current task, marks it done, and increments the counter.
    
WHY ONE TASK AT A TIME:
    - Easier debugging
    - Clear progress tracking
    - Can checkpoint between tasks
    - Enables replanning after each step
"""

from plan_do_agent.state import AgentState
from plan_do_agent.tools import route_to_tool, create_tool_executor


# =============================================================================
# Simple Executor
# =============================================================================

def create_executor(llm):
    """
    Create an executor node that processes one task at a time.
    
    Args:
        llm: The language model to use for task execution
        
    Returns:
        A function that takes AgentState and returns state updates
        
    EXECUTION FLOW:
        1. Get current task from todos[current_step]
        2. Mark task as "in_progress"
        3. Execute the task (call tools or LLM)
        4. Mark task as "done"
        5. Increment current_step
        6. Return updated state
    """
    tool_executor = create_tool_executor(llm)
    
    def executor(state: AgentState) -> dict:
        """
        Execute the current task in the todo list.
        
        This node runs in a loop until all tasks are complete.
        The should_continue function controls the loop.
        """
        current_idx = state["current_step"]
        todos = state["todos"].copy()  # Don't mutate original
        
        # Bounds check
        if current_idx >= len(todos):
            print("\n⚠️ EXECUTOR: No more tasks")
            return {"final_result": "All tasks completed"}
        
        # Get current task
        current_task = todos[current_idx]
        print(f"\n⚡ EXECUTOR: Running task {current_idx + 1}/{len(todos)}")
        print(f"   Task: {current_task['content']}")
        
        # Mark as in progress
        todos[current_idx] = {
            **current_task,
            "status": "in_progress"
        }
        
        # Execute the task
        try:
            result = tool_executor(current_task["content"])
            print(f"   Result: {result[:100]}..." if len(result) > 100 else f"   Result: {result}")
            
            # Mark as done
            todos[current_idx] = {
                **current_task,
                "status": "done"
            }
            
        except Exception as e:
            print(f"   Error: {e}")
            result = f"Task failed: {e}"
            # Still mark as done to prevent infinite loop
            todos[current_idx] = {
                **current_task,
                "status": "done"  # Could use "failed" with extended state
            }
        
        # Return state updates
        return {
            "todos": todos,
            "current_step": current_idx + 1,
            "final_result": result  # Last result becomes final result
        }
    
    return executor


# =============================================================================
# Router Logic
# =============================================================================

def should_continue(state: AgentState) -> str:
    """
    Decide whether to continue executing or finish.
    
    This is a "conditional edge" in LangGraph that routes to:
    - "execute": Run another task
    - "done": All tasks finished, go to END
    
    Args:
        state: Current agent state
        
    Returns:
        "execute" or "done"
    """
    current_idx = state["current_step"]
    total_tasks = len(state["todos"])
    
    if current_idx >= total_tasks:
        print(f"\n✅ ROUTER: All {total_tasks} tasks completed!")
        return "done"
    
    print(f"\n🔀 ROUTER: {total_tasks - current_idx} tasks remaining")
    return "execute"


def should_continue_with_replan(state: AgentState) -> str:
    """
    Enhanced router that supports replanning.
    
    Routes to:
    - "replan": After task completion, update the plan
    - "done": All tasks finished
    
    Use this when you want dynamic plan adjustment.
    """
    current_idx = state["current_step"]
    total_tasks = len(state["todos"])
    
    if current_idx >= total_tasks:
        print(f"\n✅ ROUTER: All {total_tasks} tasks completed!")
        return "done"
    
    # Option: Only replan every N steps, or on certain conditions
    print(f"\n🔀 ROUTER: Moving to replan ({total_tasks - current_idx} tasks remaining)")
    return "replan"


# =============================================================================
# Final Synthesizer (Optional)
# =============================================================================

def create_synthesizer(llm):
    """
    Create a node that synthesizes all results into a final answer.
    
    This runs after all tasks complete to provide a coherent response
    to the original user query.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    
    def synthesizer(state: AgentState) -> dict:
        """
        Synthesize task results into a final answer.
        """
        print("\n📝 SYNTHESIZER: Creating final response...")
        
        # Collect all task results (in production, store results per task)
        tasks_summary = "\n".join([
            f"- {todo['content']} [{todo['status']}]"
            for todo in state["todos"]
        ])
        
        messages = [
            SystemMessage(content="Summarize the completed tasks into a clear response."),
            HumanMessage(content=f"""
Original request: {state['input']}

Completed tasks:
{tasks_summary}

Latest result: {state.get('final_result', 'No result')}

Provide a concise summary answering the original request.
""")
        ]
        
        response = llm.invoke(messages)
        
        return {"final_result": response.content}
    
    return synthesizer
