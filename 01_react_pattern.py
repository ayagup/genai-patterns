"""
ReAct Pattern (Reasoning + Acting)
===================================
Agent alternates between reasoning about the task and taking actions.
Components: Thought → Action → Observation loop
"""

from typing import Dict, List, Tuple
import json


class Tool:
    """Base class for tools that the agent can use"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, **kwargs) -> str:
        raise NotImplementedError


class Calculator(Tool):
    """Simple calculator tool"""
    def __init__(self):
        super().__init__("calculator", "Performs mathematical calculations")
    
    def execute(self, expression: str) -> str:
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"


class SearchTool(Tool):
    """Mock search tool"""
    def __init__(self):
        super().__init__("search", "Searches for information")
        self.knowledge_base = {
            "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
            "ai": "Artificial Intelligence is the simulation of human intelligence by machines.",
            "react": "ReAct is a pattern that combines reasoning and acting in AI agents."
        }
    
    def execute(self, query: str) -> str:
        query_lower = query.lower()
        for key, value in self.knowledge_base.items():
            if key in query_lower:
                return value
        return "No information found."


class ReActAgent:
    """ReAct Pattern Agent"""
    def __init__(self, tools: List[Tool], max_iterations: int = 5):
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.history: List[Dict] = []
    
    def think(self, task: str, context: str = "") -> Tuple[str, str]:
        """
        Reasoning step - decides what to do next
        Returns: (thought, action_plan)
        """
        # Simplified reasoning logic
        if "calculate" in task.lower() or any(op in task for op in ['+', '-', '*', '/']):
            thought = "I need to perform a calculation to answer this question."
            action_plan = "use calculator"
        elif "what" in task.lower() or "who" in task.lower() or "search" in task.lower():
            thought = "I need to search for information to answer this question."
            action_plan = "use search"
        else:
            thought = "I can answer this directly based on the context."
            action_plan = "answer directly"
        
        return thought, action_plan
    
    def act(self, action_plan: str, task: str) -> str:
        """
        Action step - executes the planned action
        Returns: observation (result of the action)
        """
        if action_plan == "use calculator":
            # Extract the mathematical expression
            expression = task.split("calculate")[-1].strip().strip("?.")
            tool = self.tools.get("calculator")
            return tool.execute(expression=expression)
        
        elif action_plan == "use search":
            tool = self.tools.get("search")
            return tool.execute(query=task)
        
        else:
            return "I've processed the information."
    
    def run(self, task: str) -> str:
        """
        Main ReAct loop
        """
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")
        
        context = ""
        
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}:")
            print("-" * 40)
            
            # Thought (Reasoning)
            thought, action_plan = self.think(task, context)
            print(f"💭 Thought: {thought}")
            print(f"📋 Action Plan: {action_plan}")
            
            # Action
            observation = self.act(action_plan, task)
            print(f"👀 Observation: {observation}")
            
            # Store in history
            self.history.append({
                "iteration": iteration + 1,
                "thought": thought,
                "action_plan": action_plan,
                "observation": observation
            })
            
            # Check if we can answer
            if action_plan == "answer directly" or "Result:" in observation or observation:
                context += f" {observation}"
                print(f"\n✅ Final Answer: {observation}\n")
                break
            
            context += f" {observation}"
            print()
        
        return observation
    
    def get_history(self) -> str:
        """Returns formatted execution history"""
        return json.dumps(self.history, indent=2)


def main():
    """Demonstrate ReAct pattern"""
    
    # Initialize tools
    tools = [
        Calculator(),
        SearchTool()
    ]
    
    # Create ReAct agent
    agent = ReActAgent(tools=tools)
    
    # Example 1: Calculation task
    print("\n" + "="*60)
    print("EXAMPLE 1: Mathematical Calculation")
    print("="*60)
    agent.run("Please calculate 25 * 4 + 10")
    
    # Example 2: Search task
    print("\n" + "="*60)
    print("EXAMPLE 2: Information Search")
    print("="*60)
    agent = ReActAgent(tools=tools)  # Reset agent
    agent.run("What is Python programming language?")
    
    # Example 3: Another search
    print("\n" + "="*60)
    print("EXAMPLE 3: ReAct Pattern Search")
    print("="*60)
    agent = ReActAgent(tools=tools)  # Reset agent
    agent.run("Tell me about the ReAct pattern")
    
    print("\n" + "="*60)
    print("Execution History (Last Task):")
    print("="*60)
    print(agent.get_history())


if __name__ == "__main__":
    main()
