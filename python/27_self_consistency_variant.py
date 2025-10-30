"""
Self-Consistency Pattern
Generates multiple reasoning paths and selects most consistent answer
"""
from typing import List, Dict, Any, Tuple
from collections import Counter
import random
class SelfConsistencyAgent:
    def __init__(self, num_samples: int = 5):
        self.num_samples = num_samples
        self.reasoning_paths: List[Dict[str, Any]] = []
    def generate_reasoning_path(self, problem: str, sample_num: int) -> Dict[str, Any]:
        """Generate one reasoning path (with sampling/variation)"""
        print(f"\n--- Reasoning Path {sample_num} ---")
        # Simulate different reasoning approaches
        # In reality, this would use temperature > 0 for LLM sampling
        if "math" in problem.lower() or "calculate" in problem.lower():
            return self._solve_math_problem(problem, sample_num)
        else:
            return self._general_reasoning(problem, sample_num)
    def _solve_math_problem(self, problem: str, sample_num: int) -> Dict[str, Any]:
        """Solve math problem with variation"""
        # Example: "If John has 3 apples and buys 5 more, how many does he have?"
        # Simulate different reasoning approaches
        approaches = [
            {
                "steps": [
                    "Starting amount: 3 apples",
                    "Additional amount: 5 apples",
                    "Total = 3 + 5 = 8 apples"
                ],
                "answer": 8
            },
            {
                "steps": [
                    "Begin with 3",
                    "Add 5 more",
                    "Count: 3, 4, 5, 6, 7, 8",
                    "Final count: 8"
                ],
                "answer": 8
            },
            {
                "steps": [
                    "Total = initial + bought",
                    "Total = 3 + 5",
                    "Total = 8"
                ],
                "answer": 8
            },
            # Occasionally get wrong answer to show self-consistency filtering
            {
                "steps": [
                    "Starting with 3",
                    "Multiply by... wait, add 5",
                    "3 + 5 = 7... no 8"
                ],
                "answer": 8 if random.random() > 0.2 else 7
            }
        ]
        chosen_approach = random.choice(approaches)
        print("Steps:")
        for step in chosen_approach["steps"]:
            print(f"  - {step}")
        print(f"Answer: {chosen_approach['answer']}")
        return {
            "path_id": sample_num,
            "steps": chosen_approach["steps"],
            "answer": chosen_approach["answer"]
        }
    def _general_reasoning(self, problem: str, sample_num: int) -> Dict[str, Any]:
        """General reasoning with variation"""
        # Simulate different reasoning paths
        possible_answers = ["Answer A", "Answer B", "Answer A", "Answer A"]  # A is most consistent
        answer = possible_answers[sample_num % len(possible_answers)]
        steps = [
            f"Analyzing the problem from perspective {sample_num}",
            f"Considering various factors",
            f"Reaching conclusion: {answer}"
        ]
        print("Steps:")
        for step in steps:
            print(f"  - {step}")
        print(f"Answer: {answer}")
        return {
            "path_id": sample_num,
            "steps": steps,
            "answer": answer
        }
    def aggregate_answers(self, reasoning_paths: List[Dict[str, Any]]) -> Tuple[Any, float]:
        """Aggregate answers using majority voting"""
        answers = [path["answer"] for path in reasoning_paths]
        # Count occurrences
        answer_counts = Counter(answers)
        # Get most common answer
        most_common_answer, count = answer_counts.most_common(1)[0]
        confidence = count / len(answers)
        return most_common_answer, confidence
    def solve_with_self_consistency(self, problem: str) -> Dict[str, Any]:
        """Solve problem using self-consistency"""
        print("="*70)
        print("SELF-CONSISTENCY REASONING")
        print("="*70)
        print(f"\nProblem: {problem}")
        print(f"Generating {self.num_samples} reasoning paths...\n")
        # Generate multiple reasoning paths
        self.reasoning_paths = []
        for i in range(1, self.num_samples + 1):
            path = self.generate_reasoning_path(problem, i)
            self.reasoning_paths.append(path)
        # Aggregate answers
        print("\n" + "="*70)
        print("AGGREGATION")
        print("="*70)
        final_answer, confidence = self.aggregate_answers(self.reasoning_paths)
        # Show vote distribution
        answer_counts = Counter([path["answer"] for path in self.reasoning_paths])
        print("\nVote Distribution:")
        for answer, count in answer_counts.most_common():
            percentage = (count / self.num_samples) * 100
            print(f"  {answer}: {count}/{self.num_samples} ({percentage:.1f}%)")
        print(f"\n{'='*70}")
        print(f"FINAL ANSWER: {final_answer}")
        print(f"Confidence: {confidence:.1%}")
        print(f"{'='*70}")
        return {
            "problem": problem,
            "final_answer": final_answer,
            "confidence": confidence,
            "num_paths": self.num_samples,
            "all_answers": answer_counts
        }
# Usage
if __name__ == "__main__":
    agent = SelfConsistencyAgent(num_samples=5)
    # Example 1: Math problem
    problem1 = "If John has 3 apples and buys 5 more, how many apples does he have in total?"
    result1 = agent.solve_with_self_consistency(problem1)
    print("\n\n" + "="*80 + "\n\n")
    # Example 2: General reasoning
    problem2 = "What is the best approach to learn programming?"
    agent2 = SelfConsistencyAgent(num_samples=4)
    result2 = agent2.solve_with_self_consistency(problem2)
