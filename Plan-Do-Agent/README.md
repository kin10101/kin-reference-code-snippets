# Plan-Do Agent

A **Plan-and-Execute** agent with persistent todo state, built with LangGraph.

This pattern is used in advanced coding agents like Claude Code, Devin, and LangChain Deep Agents.

## Core Concept

```
User Query → Plan Tasks → Store Todos → Execute Next → Update Status → Repeat
```

**Key insight**: The todo list lives in LangGraph state, NOT in the LLM prompt. This enables:
- Long-running workflows (no context overflow)
- Progress tracking across many steps
- Checkpointing and resumption
- Dynamic replanning

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Plan-Do Agent                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   ┌─────────┐                                            │
│   │  START  │                                            │
│   └────┬────┘                                            │
│        │                                                 │
│        ▼                                                 │
│   ┌─────────┐    Creates structured                      │
│   │ Planner │ ── task list (JSON)                        │
│   └────┬────┘                                            │
│        │                                                 │
│        ▼                                                 │
│   ┌─────────┐    Executes ONE task,          ┌───────┐   │
│   │Executor │◄───updates status,────────────►│ Tools │   │
│   └────┬────┘    increments step             └───────┘   │
│        │                                                 │
│        ▼                                                 │
│   ┌─────────┐    More tasks?                             │
│   │ Router  │────────┐                                   │
│   └────┬────┘        │ yes                               │
│        │ no          │                                   │
│        ▼             │                                   │
│   ┌─────────┐        │                                   │
│   │   END   │◄───────┘                                   │
│   └─────────┘                                            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
cd Plan-Do-Agent
pip install -e .
```

Or with uv:
```bash
uv pip install -e .
```

### Set API Key

```bash
export OPENAI_API_KEY='your-key'
# Or create .env file with OPENAI_API_KEY=your-key
```

### Run

```bash
# Default example
python -m plan_do_agent.main

# Custom query
python -m plan_do_agent.main "Research Python asyncio patterns"

# With replanning
python -m plan_do_agent.main --replan

# Interactive mode
python -m plan_do_agent.main -i
```

## Project Structure

```
Plan-Do-Agent/
├── pyproject.toml           # Dependencies
├── README.md                # This file
└── src/
    └── plan_do_agent/
        ├── __init__.py      # Package exports
        ├── state.py         # 📦 Data structures (Todo, AgentState)
        ├── planner.py       # 🧠 Creates task list from user input
        ├── executor.py      # ⚡ Runs one task at a time
        ├── tools.py         # 🔧 Available tools (search, calc, etc.)
        ├── graph.py         # 🔗 Assembles the LangGraph
        └── main.py          # 🚀 Entry point
```

## How It Works

### 1. State Definition ([state.py](src/plan_do_agent/state.py))

```python
class Todo(TypedDict):
    content: str                              # Task description
    status: Literal["pending", "in_progress", "done"]

class AgentState(TypedDict):
    input: str          # User's original query
    todos: list[Todo]   # The persistent task list
    current_step: int   # Index of current task
    final_result: str   # Output after completion
```

### 2. Planner ([planner.py](src/plan_do_agent/planner.py))

Uses structured output to guarantee valid JSON:

```python
def planner(state: AgentState) -> dict:
    # LLM breaks down task into steps
    plan = llm.with_structured_output(Plan).invoke(messages)
    
    return {
        "todos": [{"content": t.content, "status": "pending"} for t in plan.tasks],
        "current_step": 0
    }
```

### 3. Executor ([executor.py](src/plan_do_agent/executor.py))

Runs ONE task at a time:

```python
def executor(state: AgentState) -> dict:
    task = state["todos"][state["current_step"]]
    
    result = tool_executor(task["content"])
    
    return {
        "current_step": state["current_step"] + 1,
        "final_result": result
    }
```

### 4. Router

Decides continue or stop:

```python
def should_continue(state: AgentState) -> str:
    if state["current_step"] >= len(state["todos"]):
        return "done"  # → END
    return "execute"   # → executor again
```

### 5. Graph Assembly ([graph.py](src/plan_do_agent/graph.py))

```python
graph = StateGraph(AgentState)

graph.add_node("planner", planner)
graph.add_node("executor", executor)

graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue, {
    "execute": "executor",
    "done": END
})

agent = graph.compile()
```

## Plan-Do vs ReAct

| Aspect | ReAct | Plan-Do |
|--------|-------|---------|
| **Pattern** | Think → Act → Observe → Think | Plan → Execute All → Done |
| **LLM Calls** | One per action | One planning + one per task |
| **State** | In LLM context | External (LangGraph state) |
| **Long Tasks** | Context overflow risk | Handles many steps |
| **Progress** | Hidden | Explicit todo tracking |

## Extension Points

### Add Custom Tools

Edit [tools.py](src/plan_do_agent/tools.py):

```python
@tool
def my_tool(arg: str) -> str:
    """Tool description."""
    return "result"

TOOLS = [..., my_tool]
```

### Enable Replanning

Use `create_agent_with_replan()` for dynamic plan adjustment:

```python
from plan_do_agent.graph import create_agent_with_replan

agent = create_agent_with_replan(llm)
```

### Add Persistence

TODO: Enable SQLite checkpointing for resumable workflows.

### Task Dependencies

Use `TodoExtended` from [state.py](src/plan_do_agent/state.py) for DAG-based execution.

## Related Patterns

- **LangChain Plan-and-Execute**: Similar, uses `PlanAndExecute` chain
- **TodoListMiddleware**: Injects `write_todos` tool for runtime updates
- **Claude Code / Devin**: Production agents using this pattern
- **LangGraph ReAct**: Alternative pattern using `create_react_agent`

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Plan-and-Execute Agents](https://blog.langchain.dev/planning-agents/)
- [LangChain Agent Types](https://python.langchain.com/docs/modules/agents/)
