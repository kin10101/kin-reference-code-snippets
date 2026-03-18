"""
Tools for Plan-Do Agent

This module defines the tools available to the executor.
Tools are functions that the agent can call to interact with the world.

DESIGN PRINCIPLE:
    Each tool should do ONE thing well.
    The planner creates tasks, the executor routes to tools.
    
HOW TOOLS WORK IN THIS ARCHITECTURE:
    1. Planner creates todo: "Search for X on the web"
    2. Executor sees this todo
    3. Tool router matches "Search" → web_search tool
    4. Tool executes and returns result
    5. Result is stored, executor moves to next todo
"""

from typing import Callable
from langchain_core.tools import tool


# =============================================================================
# Example Tools
# =============================================================================

@tool
def web_search(query: str) -> str:
    """
    Search the web for information.
    
    Args:
        query: The search query
        
    Returns:
        Search results as a string
    """
    # In production, integrate with a real search API (Tavily, Serper, etc.)
    return f"[Search Results for '{query}']: Found 3 relevant articles about {query}."


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: A math expression like "2 + 2" or "sqrt(16)"
        
    Returns:
        The result of the calculation
    """
    try:
        # WARNING: eval is dangerous in production. Use a safe math parser.
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {e}"


@tool  
def read_file(filepath: str) -> str:
    """
    Read contents of a file.
    
    Args:
        filepath: Path to the file to read
        
    Returns:
        File contents or error message
    """
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {filepath}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(filepath: str, content: str) -> str:
    """
    Write content to a file.
    
    Args:
        filepath: Path to the file to write
        content: Content to write to the file
        
    Returns:
        Success or error message
    """
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error writing file: {e}"


# =============================================================================
# Tool Registry
# =============================================================================

# All available tools - add new tools here
TOOLS = [
    web_search,
    calculator,
    read_file,
    write_file,
]

# Tool lookup by name for routing
TOOL_MAP: dict[str, Callable] = {t.name: t for t in TOOLS}


# =============================================================================
# Tool Router
# =============================================================================

def route_to_tool(task_content: str, llm) -> str:
    """
    Route a task to the appropriate tool using LLM reasoning.
    
    This is a simple keyword-based router. For production, use:
    - LLM-based tool selection
    - Semantic similarity matching
    - LangChain's tool calling with bind_tools()
    
    Args:
        task_content: The task description from the todo
        llm: The language model for tool selection
        
    Returns:
        The result of executing the selected tool
    """
    # Simple keyword matching (replace with LLM tool calling in production)
    content_lower = task_content.lower()
    
    if "search" in content_lower or "find" in content_lower:
        return web_search.invoke({"query": task_content})
    elif "calculate" in content_lower or "math" in content_lower:
        # Extract the expression from the task
        return calculator.invoke({"expression": task_content})
    elif "read" in content_lower and "file" in content_lower:
        # Would need to extract filepath from task
        return "File reading requires specific filepath"
    elif "write" in content_lower and "file" in content_lower:
        return "File writing requires specific filepath and content"
    else:
        # Default: Use LLM to handle the task directly
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=task_content)])
        return response.content


def create_tool_executor(llm):
    """
    Create a tool executor that can handle tasks using available tools.
    
    This wraps tools with the LLM to enable intelligent tool selection
    and argument extraction.
    
    Args:
        llm: The language model to use for tool calling
        
    Returns:
        An executor function that takes task content and returns results
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    
    # Bind tools to LLM for automatic tool calling
    llm_with_tools = llm.bind_tools(TOOLS)
    
    def execute_task(task_content: str) -> str:
        """Execute a single task using LLM tool calling."""
        messages = [
            SystemMessage(content="Execute the following task using available tools."),
            HumanMessage(content=task_content)
        ]
        
        response = llm_with_tools.invoke(messages)
        
        # If LLM made tool calls, execute them
        if hasattr(response, 'tool_calls') and response.tool_calls:
            results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                if tool_name in TOOL_MAP:
                    result = TOOL_MAP[tool_name].invoke(tool_args)
                    results.append(f"{tool_name}: {result}")
            return "\n".join(results)
        
        # No tools called, return LLM response
        return response.content
    
    return execute_task
