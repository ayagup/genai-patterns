"""
Chain-of-Thought Pattern
Breaks down complex problems into intermediate reasoning steps
"""
from typing import List, Tuple
class ChainOfThoughtAgent:
    def __init__(self):
        self.steps = []
    def solve_step_by_step(self, problem: str) -> str:
        """Solve problem with explicit reasoning steps"""
        # Example: Math word problem
        if "total" in problem.lower() and "cost" in problem.lower():
            return self._solve_math_problem(problem)
        return "Unknown problem type"
    def _solve_math_problem(self, problem: str) -> str:
        """Example: Solving a math word problem step by step"""
        # Step 1: Understand the problem
        step1 = "Let me understand what we're looking for: the total cost"
        self.steps.append(step1)
        print(f"Step 1: {step1}")
        # Step 2: Identify given information
        step2 = "Given: 3 apples at $2 each, 2 oranges at $3 each"
        self.steps.append(step2)
        print(f"Step 2: {step2}")
        # Step 3: Calculate apple cost
        apple_cost = 3 * 2
        step3 = f"Cost of apples: 3 × $2 = ${apple_cost}"
        self.steps.append(step3)
        print(f"Step 3: {step3}")
        # Step 4: Calculate orange cost
        orange_cost = 2 * 3
        step4 = f"Cost of oranges: 2 × $3 = ${orange_cost}"
        self.steps.append(step4)
        print(f"Step 4: {step4}")
        # Step 5: Calculate total
        total = apple_cost + orange_cost
        step5 = f"Total cost: ${apple_cost} + ${orange_cost} = ${total}"
        self.steps.append(step5)
        print(f"Step 5: {step5}")
        return f"The total cost is ${total}"
    def get_reasoning_chain(self) -> List[str]:
        """Return all reasoning steps"""
        return self.steps
# Usage
if __name__ == "__main__":
    agent = ChainOfThoughtAgent()
    problem = "If I buy 3 apples at $2 each and 2 oranges at $3 each, what is the total cost?"
    print("Problem:", problem)
    print("\nSolving step by step:\n")
    answer = agent.solve_step_by_step(problem)
    print(f"\nFinal Answer: {answer}")
    print("\n--- Complete Reasoning Chain ---")
    for i, step in enumerate(agent.get_reasoning_chain(), 1):
        print(f"{i}. {step}")
