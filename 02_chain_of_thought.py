"""
Chain-of-Thought (CoT) Pattern
===============================
Agent breaks down complex problems into intermediate reasoning steps.
"""

from typing import List, Dict
import re


class ChainOfThoughtAgent:
    """Chain-of-Thought reasoning agent"""
    
    def __init__(self):
        self.reasoning_steps: List[str] = []
    
    def solve_math_problem(self, problem: str) -> Dict:
        """
        Solves a math problem using step-by-step reasoning
        """
        self.reasoning_steps = []
        
        print(f"\n{'='*60}")
        print(f"Problem: {problem}")
        print(f"{'='*60}\n")
        
        # Example: "If John has 5 apples and buys 3 more, then gives 2 to Mary, how many does he have?"
        
        # Step 1: Identify initial state
        step1 = "Step 1: Identify what John starts with"
        self.reasoning_steps.append(step1)
        print(f"💭 {step1}")
        
        # Extract numbers (simplified parsing)
        numbers = re.findall(r'\d+', problem)
        if len(numbers) >= 3:
            initial = int(numbers[0])
            print(f"   → John starts with {initial} apples\n")
            
            # Step 2: Add purchases
            step2 = "Step 2: Calculate after buying more"
            self.reasoning_steps.append(step2)
            print(f"💭 {step2}")
            buys = int(numbers[1])
            after_buying = initial + buys
            print(f"   → {initial} + {buys} = {after_buying} apples\n")
            
            # Step 3: Subtract what was given away
            step3 = "Step 3: Calculate after giving some away"
            self.reasoning_steps.append(step3)
            print(f"💭 {step3}")
            gives = int(numbers[2])
            final = after_buying - gives
            print(f"   → {after_buying} - {gives} = {final} apples\n")
            
            # Step 4: Final answer
            step4 = f"Step 4: Final answer is {final} apples"
            self.reasoning_steps.append(step4)
            print(f"💭 {step4}\n")
            
            return {
                "answer": final,
                "reasoning_steps": self.reasoning_steps,
                "explanation": f"John ends up with {final} apples"
            }
        
        return {"answer": None, "reasoning_steps": self.reasoning_steps, "explanation": "Could not parse problem"}
    
    def solve_logic_problem(self, problem: str) -> Dict:
        """
        Solves a logic problem using step-by-step reasoning
        """
        self.reasoning_steps = []
        
        print(f"\n{'='*60}")
        print(f"Problem: {problem}")
        print(f"{'='*60}\n")
        
        # Example: "All birds can fly. Penguins are birds. Can penguins fly?"
        
        steps = [
            ("Step 1: Identify the general rule", "All birds can fly (given premise)"),
            ("Step 2: Identify the specific case", "Penguins are birds (given premise)"),
            ("Step 3: Apply logical reasoning", "If all birds can fly, and penguins are birds..."),
            ("Step 4: Draw conclusion", "Then penguins should be able to fly"),
            ("Step 5: Reality check", "However, in reality, penguins cannot fly (exception to rule)")
        ]
        
        for step_title, step_detail in steps:
            self.reasoning_steps.append(step_title)
            print(f"💭 {step_title}")
            print(f"   → {step_detail}\n")
        
        return {
            "answer": "Logically yes, but actually no (exception)",
            "reasoning_steps": self.reasoning_steps,
            "explanation": "This demonstrates that real-world knowledge can override pure logical deduction"
        }
    
    def analyze_complex_scenario(self, scenario: str) -> Dict:
        """
        Analyzes a complex scenario step-by-step
        """
        self.reasoning_steps = []
        
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario}")
        print(f"{'='*60}\n")
        
        # Example: "A company needs to decide whether to launch a new product"
        
        analysis_steps = [
            ("Step 1: Identify the decision to be made", 
             "Should the company launch the new product?"),
            
            ("Step 2: List potential benefits",
             "• Increased revenue\n      • Market expansion\n      • Brand strengthening"),
            
            ("Step 3: List potential risks",
             "• Development costs\n      • Market uncertainty\n      • Competitor response"),
            
            ("Step 4: Consider constraints",
             "• Budget limitations\n      • Time to market\n      • Resource availability"),
            
            ("Step 5: Evaluate alternatives",
             "• Launch immediately\n      • Pilot test first\n      • Delay and gather more data"),
            
            ("Step 6: Formulate recommendation",
             "Conduct a pilot test in a limited market first")
        ]
        
        for step_title, step_detail in analysis_steps:
            self.reasoning_steps.append(step_title)
            print(f"💭 {step_title}")
            print(f"   {step_detail}\n")
        
        return {
            "recommendation": "Pilot test approach",
            "reasoning_steps": self.reasoning_steps,
            "explanation": "Balances opportunity with risk management"
        }


def demonstrate_zero_shot_cot():
    """Demonstrate Zero-Shot Chain-of-Thought"""
    print("\n" + "="*60)
    print("ZERO-SHOT CHAIN-OF-THOUGHT")
    print("Adding 'Let's think step by step' prompt")
    print("="*60)
    
    problem = "If a train travels 60 miles in 1 hour, how far will it travel in 2.5 hours?"
    print(f"\nProblem: {problem}")
    print("\nPrompt: Let's think step by step...")
    print("\nReasoning:")
    print("1. The train travels 60 miles per hour")
    print("2. We need to find distance for 2.5 hours")
    print("3. Distance = Speed × Time")
    print("4. Distance = 60 × 2.5")
    print("5. Distance = 150 miles")
    print("\n✅ Answer: 150 miles")


def demonstrate_few_shot_cot():
    """Demonstrate Few-Shot Chain-of-Thought"""
    print("\n" + "="*60)
    print("FEW-SHOT CHAIN-OF-THOUGHT")
    print("Providing examples with reasoning")
    print("="*60)
    
    print("\nExample 1:")
    print("Q: Roger has 5 tennis balls. He buys 2 more. How many does he have?")
    print("A: Let's think step by step.")
    print("   1. Roger starts with 5 balls")
    print("   2. He buys 2 more balls")
    print("   3. Total = 5 + 2 = 7 balls")
    print("   Answer: 7 balls")
    
    print("\nExample 2:")
    print("Q: The cafeteria had 23 apples. They used 20 for lunch. They bought 6 more. How many do they have?")
    print("A: Let's think step by step.")
    print("   1. Started with 23 apples")
    print("   2. Used 20, so 23 - 20 = 3 apples left")
    print("   3. Bought 6 more, so 3 + 6 = 9 apples")
    print("   Answer: 9 apples")
    
    print("\nNow solve:")
    print("Q: Sarah has 8 cookies. She eats 2 and bakes 5 more. How many does she have?")
    print("A: Let's think step by step.")
    print("   1. Sarah starts with 8 cookies")
    print("   2. She eats 2, so 8 - 2 = 6 cookies left")
    print("   3. She bakes 5 more, so 6 + 5 = 11 cookies")
    print("   Answer: 11 cookies")


def main():
    """Demonstrate Chain-of-Thought pattern"""
    
    agent = ChainOfThoughtAgent()
    
    # Example 1: Math problem
    print("\n" + "="*60)
    print("EXAMPLE 1: Mathematical Word Problem")
    print("="*60)
    result1 = agent.solve_math_problem(
        "If John has 5 apples and buys 3 more, then gives 2 to Mary, how many does he have?"
    )
    print(f"✅ Final Answer: {result1['answer']} apples")
    
    # Example 2: Logic problem
    print("\n" + "="*60)
    print("EXAMPLE 2: Logic Problem")
    print("="*60)
    result2 = agent.solve_logic_problem(
        "All birds can fly. Penguins are birds. Can penguins fly?"
    )
    print(f"✅ Conclusion: {result2['answer']}")
    
    # Example 3: Complex scenario
    print("\n" + "="*60)
    print("EXAMPLE 3: Complex Business Scenario")
    print("="*60)
    result3 = agent.analyze_complex_scenario(
        "A company needs to decide whether to launch a new product"
    )
    print(f"✅ Recommendation: {result3['recommendation']}")
    
    # Demonstrate variants
    demonstrate_zero_shot_cot()
    demonstrate_few_shot_cot()


if __name__ == "__main__":
    main()
