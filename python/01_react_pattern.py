"""
ReAct Pattern: Reasoning + Acting
Agent alternates between reasoning and taking actions
"""
from typing import List, Dict, Any
import json
class ReActAgent:
    def __init__(self, tools: Dict[str, callable]):
        self.tools = tools
        self.max_iterations = 5
    def think(self, question: str, context: str) -> Dict[str, str]:
        """Simulate LLM reasoning about next action"""
        # In real implementation, this would call an LLM
        thought = f"I need to find information about: {question}"
        action = "search"
        action_input = question
        return {
            "thought": thought,
            "action": action,
            "action_input": action_input
        }
    def act(self, action: str, action_input: str) -> str:
        """Execute the chosen action"""
        if action in self.tools:
            return self.tools[action](action_input)
        return "Action not found"
    def run(self, question: str) -> str:
        """Main ReAct loop"""
        context = ""
        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1} ---")
            # Reasoning step
            decision = self.think(question, context)
            print(f"Thought: {decision['thought']}")
            print(f"Action: {decision['action']}({decision['action_input']})")
            # Check if we should finish
            if decision['action'] == 'finish':
                return decision['action_input']
            # Acting step
            observation = self.act(decision['action'], decision['action_input'])
            print(f"Observation: {observation}")
            context += f"\nAction: {decision['action']}\nObservation: {observation}"
        return "Max iterations reached"
# Example tools
def search_tool(query: str) -> str:
    """Simulate a search tool"""
    knowledge_base = {
        "python": "Python is a high-level programming language created by Guido van Rossum",
        "ai": "Artificial Intelligence is the simulation of human intelligence by machines"
    }
    for key in knowledge_base:
        if key in query.lower():
            return knowledge_base[key]
    return "No information found"
def calculator_tool(expression: str) -> str:
    """Simulate a calculator"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except:
        return "Invalid expression"
# Usage
if __name__ == "__main__":
    tools = {
        "search": search_tool,
        "calculator": calculator_tool
    }
    agent = ReActAgent(tools)
    result = agent.run("What is Python?")
    print(f"\nFinal Answer: {result}")
