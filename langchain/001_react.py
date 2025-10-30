"""
Pattern 001: ReAct (Reasoning + Acting)

Description:
    Agent alternates between reasoning about the task and taking actions.
    The ReAct pattern combines reasoning traces with task-specific actions,
    allowing the agent to reason about what to do and then take actions.

Components:
    - Thought: Agent reasons about current state
    - Action: Agent takes an action (e.g., tool use)
    - Observation: Agent observes the result
    - Loop continues until task is complete

Use Cases:
    - Question answering with external knowledge
    - Task completion requiring multiple steps
    - Interactive problem solving

LangChain Implementation:
    Uses LangChain's AgentExecutor with structured tools and ReAct prompt template.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


# Define example tools
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        # Safe evaluation for demo purposes
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def search_tool(query: str) -> str:
    """Simulates a search tool that retrieves information."""
    # Simulated search results
    knowledge_base = {
        "capital of france": "Paris",
        "population of tokyo": "approximately 14 million",
        "height of mount everest": "8,848.86 meters (29,031.7 feet)",
        "speed of light": "299,792,458 meters per second",
        "boiling point of water": "100°C (212°F) at sea level",
    }
    
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return f"Search result: {value}"
    
    return f"Search result: Information about '{query}' not found in knowledge base."


def get_current_date() -> str:
    """Returns the current date."""
    from datetime import datetime
    return f"Current date: {datetime.now().strftime('%Y-%m-%d')}"


# Create tools list
tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for performing mathematical calculations. Input should be a valid mathematical expression."
    ),
    Tool(
        name="Search",
        func=search_tool,
        description="Useful for searching factual information. Input should be a search query."
    ),
    Tool(
        name="GetDate",
        func=get_current_date,
        description="Returns the current date. No input required."
    )
]


# ReAct prompt template
REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""


def create_react_agent_system():
    """Creates a ReAct agent using LangChain."""
    
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create prompt template
    prompt = PromptTemplate.from_template(REACT_PROMPT)
    
    # Create agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )
    
    return agent_executor


def demonstrate_react_pattern():
    """Demonstrates the ReAct pattern with various examples."""
    
    print("=" * 80)
    print("PATTERN 001: ReAct (Reasoning + Acting)")
    print("=" * 80)
    print()
    
    # Create the ReAct agent
    agent_executor = create_react_agent_system()
    
    # Test cases demonstrating ReAct pattern
    test_questions = [
        "What is the capital of France?",
        "What is 25 * 4 + 10?",
        "What is the population of Tokyo and what is 100 divided by that number in millions?",
        "What's today's date?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'=' * 80}")
        print(f"Example {i}: {question}")
        print('=' * 80)
        
        try:
            result = agent_executor.invoke({"input": question})
            print(f"\n✓ Final Answer: {result['output']}")
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
        
        print()
    
    # Summary
    print("\n" + "=" * 80)
    print("REACT PATTERN DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("1. Thought-Action-Observation loop")
    print("2. Tool selection and usage")
    print("3. Multi-step reasoning")
    print("4. Dynamic decision making")
    print("5. Interpretable reasoning traces")
    print()
    print("LangChain Components Used:")
    print("- AgentExecutor: Manages agent execution loop")
    print("- create_react_agent: Creates ReAct-style agent")
    print("- Tool: Wraps functions as agent tools")
    print("- PromptTemplate: Structures the ReAct prompt format")
    print("- ChatOpenAI: LLM for reasoning and action selection")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    demonstrate_react_pattern()
