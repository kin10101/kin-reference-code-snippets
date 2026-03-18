"""
Main Entry Point for Plan-Do Agent

This script demonstrates how to run the Plan-Do agent.
Run with: python -m plan_do_agent.main

ENVIRONMENT VARIABLES:
    OPENAI_API_KEY: Required for OpenAI models
    
USAGE:
    # Basic
    python -m plan_do_agent.main
    
    # With custom query
    python -m plan_do_agent.main "Research Python web frameworks"
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def run_basic_example():
    """Run a basic Plan-Do agent example."""
    from langchain_openai import ChatOpenAI
    from plan_do_agent.graph import create_agent
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Use smaller model for cost efficiency
        temperature=0        # Deterministic for planning
    )
    
    # Create agent
    agent = create_agent(llm)
    
    # Example query
    query = "Calculate the sum of 15 + 27, then search for Python math libraries"
    
    print("=" * 60)
    print("🤖 PLAN-DO AGENT")
    print("=" * 60)
    print(f"\n📥 Input: {query}\n")
    
    # Run agent
    result = agent.invoke({
        "input": query,
        "todos": [],
        "current_step": 0,
        "messages": [],
        "final_result": ""
    })
    
    print("\n" + "=" * 60)
    print("📤 FINAL RESULT:")
    print("=" * 60)
    print(result.get("final_result", "No result"))
    
    return result


def run_with_replan_example():
    """Run agent with dynamic replanning."""
    from langchain_openai import ChatOpenAI
    from plan_do_agent.graph import create_agent_with_replan
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent_with_replan(llm)
    
    query = "Research the top 3 Python web frameworks and compare them"
    
    print("=" * 60)
    print("🤖 PLAN-DO AGENT (with Replanning)")
    print("=" * 60)
    print(f"\n📥 Input: {query}\n")
    
    result = agent.invoke({
        "input": query,
        "todos": [],
        "current_step": 0,
        "messages": [],
        "final_result": ""
    })
    
    print("\n" + "=" * 60)
    print("📤 FINAL RESULT:")
    print("=" * 60)
    print(result.get("final_result", "No result"))
    
    return result


def run_interactive():
    """Run agent in interactive mode."""
    from langchain_openai import ChatOpenAI
    from plan_do_agent.graph import create_agent
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(llm)
    
    print("=" * 60)
    print("🤖 PLAN-DO AGENT - Interactive Mode")
    print("=" * 60)
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            query = input("📥 Your task: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            result = agent.invoke({
                "input": query,
                "todos": [],
                "current_step": 0,
                "messages": [],
                "final_result": ""
            })
            
            print("\n📤 Result:", result.get("final_result", "No result"))
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    """Main entry point."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        print("Or create a .env file with: OPENAI_API_KEY=your-key\n")
    
    # Parse command line args
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == "--replan":
            run_with_replan_example()
        elif arg == "--interactive" or arg == "-i":
            run_interactive()
        elif arg == "--help" or arg == "-h":
            print(__doc__)
        else:
            # Treat argument as a query
            from langchain_openai import ChatOpenAI
            from plan_do_agent.graph import create_agent
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            agent = create_agent(llm)
            
            result = agent.invoke({
                "input": arg,
                "todos": [],
                "current_step": 0,
                "messages": [],
                "final_result": ""
            })
            
            print("Result:", result.get("final_result", "No result"))
    else:
        # Run default example
        run_basic_example()


if __name__ == "__main__":
    main()
