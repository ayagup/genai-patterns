"""
Pattern 1: ReAct (Reasoning + Acting)
Agent alternates between reasoning about the task and taking actions.
"""
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
import os

def search_tool(query: str) -> str:
    """Mock search tool"""
    return f"Search results for '{query}': Sample information about {query}"

def calculator_tool(expression: str) -> str:
    """Simple calculator"""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

class ReActPattern:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        
        # Define tools
        self.tools = [
            Tool(
                name="Search",
                func=search_tool,
                description="Useful for searching information"
            ),
            Tool(
                name="Calculator",
                func=calculator_tool,
                description="Useful for mathematical calculations"
            )
        ]
        
        # ReAct prompt template
        template = """Answer the following questions as best you can. You have access to the following tools:

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
Thought: {agent_scratchpad}"""

        self.prompt = PromptTemplate.from_template(template)
        
        # Create agent
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def run(self, query: str) -> str:
        """Execute ReAct agent"""
        return self.agent_executor.invoke({"input": query})

if __name__ == "__main__":
    react = ReActPattern()
    result = react.run("What is 25 * 4 + 10?")
    print(f"\nFinal Result: {result}")
