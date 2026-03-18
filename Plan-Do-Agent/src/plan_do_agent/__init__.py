"""
Plan-Do Agent: A LangGraph implementation of Plan-and-Execute with persistent todo state.

This package provides a structured approach to multi-step task execution where:
1. A planner LLM breaks down the task into steps
2. Steps are stored in persistent state (not in the LLM context)
3. An executor runs each step sequentially
4. Progress is tracked via todo status updates
"""

from plan_do_agent.state import Todo, AgentState
from plan_do_agent.graph import create_agent

__all__ = ["Todo", "AgentState", "create_agent"]
