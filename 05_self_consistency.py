"""
Self-Consistency Pattern
========================
Generates multiple reasoning paths and selects the most consistent answer.
Process: Sample multiple outputs → Vote/aggregate → Select best
"""

from typing import List, Dict, Tuple
from collections import Counter
import random


class SelfConsistencyAgent:
    """Agent that uses self-consistency to improve answer quality"""
    
    def __init__(self, num_samples: int = 5):
        self.num_samples = num_samples
        self.reasoning_paths: List[Dict] = []
    
    def solve_problem_variant(self, problem: str, seed: int) -> Tuple[str, str]:
        """
        Generates one reasoning path for the problem
        Returns: (answer, reasoning)
        """
        random.seed(seed)
        
        # Extract numbers from problem
        numbers = [int(word) for word in problem.split() if word.isdigit()]
        
        if len(numbers) >= 2:
            # Different reasoning approaches
            approaches = [
                self._approach_sequential,
                self._approach_grouping,
                self._approach_strategic
            ]
            
            # Randomly select an approach (simulating diverse reasoning)
            approach = random.choice(approaches)
            return approach(numbers, problem)
        
        return "Cannot solve", "Insufficient information"
    
    def _approach_sequential(self, numbers: List[int], problem: str) -> Tuple[str, str]:
        """Sequential calculation approach"""
        reasoning = "Sequential approach:\n"
        result = numbers[0]
        reasoning += f"  Start with {result}\n"
        
        for i, num in enumerate(numbers[1:], 1):
            if "add" in problem.lower() or "plus" in problem.lower():
                result += num
                reasoning += f"  Add {num}: {result}\n"
            elif "multiply" in problem.lower() or "times" in problem.lower():
                result *= num
                reasoning += f"  Multiply by {num}: {result}\n"
        
        return str(result), reasoning
    
    def _approach_grouping(self, numbers: List[int], problem: str) -> Tuple[str, str]:
        """Grouping approach"""
        reasoning = "Grouping approach:\n"
        
        if len(numbers) >= 4:
            # Group in pairs
            group1 = numbers[0] + numbers[1]
            group2 = numbers[2] + numbers[3]
            reasoning += f"  Group 1: {numbers[0]} + {numbers[1]} = {group1}\n"
            reasoning += f"  Group 2: {numbers[2]} + {numbers[3]} = {group2}\n"
            result = group1 + group2
            reasoning += f"  Combine: {group1} + {group2} = {result}\n"
        else:
            result = sum(numbers)
            reasoning += f"  Sum all: {' + '.join(map(str, numbers))} = {result}\n"
        
        return str(result), reasoning
    
    def _approach_strategic(self, numbers: List[int], problem: str) -> Tuple[str, str]:
        """Strategic approach"""
        reasoning = "Strategic approach:\n"
        
        # Sort for easier calculation
        sorted_nums = sorted(numbers)
        reasoning += f"  Sorted numbers: {sorted_nums}\n"
        
        result = sum(sorted_nums)
        reasoning += f"  Total sum: {result}\n"
        
        return str(result), reasoning
    
    def solve_with_consistency(self, problem: str) -> Dict:
        """
        Solves problem using multiple reasoning paths and selects most consistent answer
        """
        print(f"\n{'='*60}")
        print(f"Problem: {problem}")
        print(f"{'='*60}\n")
        
        self.reasoning_paths = []
        answers = []
        
        # Generate multiple reasoning paths
        print(f"Generating {self.num_samples} reasoning paths...\n")
        
        for i in range(self.num_samples):
            answer, reasoning = self.solve_problem_variant(problem, seed=i * 42)
            self.reasoning_paths.append({
                "path_id": i + 1,
                "answer": answer,
                "reasoning": reasoning
            })
            answers.append(answer)
            
            print(f"Path {i + 1}:")
            print(f"{reasoning}")
            print(f"Answer: {answer}\n")
            print("-" * 40)
        
        # Vote on answers
        answer_counts = Counter(answers)
        most_common_answer, count = answer_counts.most_common(1)[0]
        confidence = count / len(answers)
        
        print(f"\n{'='*60}")
        print("VOTING RESULTS")
        print(f"{'='*60}\n")
        
        for answer, count in answer_counts.most_common():
            percentage = (count / len(answers)) * 100
            bar = "█" * int(percentage / 5)
            print(f"{answer:>10}: {bar} {count}/{len(answers)} ({percentage:.1f}%)")
        
        print(f"\n✅ Selected Answer: {most_common_answer}")
        print(f"   Confidence: {confidence * 100:.1f}%")
        
        return {
            "answer": most_common_answer,
            "confidence": confidence,
            "vote_distribution": dict(answer_counts),
            "reasoning_paths": self.reasoning_paths
        }


class MajorityVotingAgent:
    """Demonstrates simple majority voting"""
    
    def solve_classification(self, question: str, num_votes: int = 7) -> Dict:
        """
        Simulates multiple classification attempts and uses majority vote
        """
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}\n")
        
        # Simulate multiple model predictions (in real scenario, these would be actual model calls)
        possible_answers = ["Positive", "Negative", "Neutral"]
        votes = []
        
        print(f"Collecting {num_votes} votes...\n")
        
        # Simulate voting with some randomness
        for i in range(num_votes):
            # Simulate that most votes lean towards one answer
            if i < num_votes * 0.6:
                vote = "Positive"  # Majority tends to say positive
            else:
                vote = random.choice(possible_answers)
            
            votes.append(vote)
            print(f"Vote {i + 1}: {vote}")
        
        # Count votes
        vote_counts = Counter(votes)
        winner, max_count = vote_counts.most_common(1)[0]
        confidence = max_count / len(votes)
        
        print(f"\n{'='*60}")
        print("VOTING SUMMARY")
        print(f"{'='*60}\n")
        
        for answer, count in vote_counts.most_common():
            percentage = (count / len(votes)) * 100
            print(f"{answer}: {count} votes ({percentage:.1f}%)")
        
        print(f"\n✅ Final Answer: {winner}")
        print(f"   Confidence: {confidence * 100:.1f}%")
        
        return {
            "answer": winner,
            "confidence": confidence,
            "votes": vote_counts
        }


def demonstrate_weighted_voting():
    """Demonstrates weighted voting where some models have higher weights"""
    print(f"\n{'='*60}")
    print("WEIGHTED VOTING EXAMPLE")
    print(f"{'='*60}\n")
    
    question = "Is this movie review positive or negative: 'The film was okay but nothing special'"
    print(f"Question: {question}\n")
    
    # Simulate predictions from different models with different reliabilities
    predictions = [
        {"model": "Model A (GPT-4)", "prediction": "Neutral", "weight": 3.0},
        {"model": "Model B (GPT-3.5)", "prediction": "Negative", "weight": 2.0},
        {"model": "Model C (BERT)", "prediction": "Neutral", "weight": 2.0},
        {"model": "Model D (RoBERTa)", "prediction": "Neutral", "weight": 1.5},
        {"model": "Model E (Basic)", "prediction": "Positive", "weight": 1.0},
    ]
    
    print("Model Predictions:")
    for pred in predictions:
        print(f"  {pred['model']}: {pred['prediction']} (weight: {pred['weight']})")
    
    # Calculate weighted votes
    weighted_votes = {}
    total_weight = 0
    
    for pred in predictions:
        answer = pred['prediction']
        weight = pred['weight']
        weighted_votes[answer] = weighted_votes.get(answer, 0) + weight
        total_weight += weight
    
    print(f"\n{'='*60}")
    print("WEIGHTED VOTING RESULTS")
    print(f"{'='*60}\n")
    
    sorted_votes = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)
    
    for answer, weight in sorted_votes:
        percentage = (weight / total_weight) * 100
        bar = "█" * int(percentage / 5)
        print(f"{answer:>10}: {bar} {weight:.1f}/{total_weight:.1f} ({percentage:.1f}%)")
    
    winner = sorted_votes[0][0]
    confidence = sorted_votes[0][1] / total_weight
    
    print(f"\n✅ Final Answer: {winner}")
    print(f"   Confidence: {confidence * 100:.1f}%")


def main():
    """Demonstrate Self-Consistency pattern"""
    
    # Example 1: Math problem with self-consistency
    print("\n" + "="*60)
    print("EXAMPLE 1: Math Problem with Self-Consistency")
    print("="*60)
    
    agent = SelfConsistencyAgent(num_samples=5)
    result1 = agent.solve_with_consistency(
        "What is 15 plus 27 plus 8 plus 10?"
    )
    
    # Example 2: Classification with majority voting
    print("\n\n" + "="*60)
    print("EXAMPLE 2: Sentiment Classification with Majority Voting")
    print("="*60)
    
    voting_agent = MajorityVotingAgent()
    result2 = voting_agent.solve_classification(
        "Classify sentiment: 'This product exceeded my expectations!'",
        num_votes=7
    )
    
    # Example 3: Weighted voting
    print("\n\n" + "="*60)
    print("EXAMPLE 3: Weighted Voting")
    print("="*60)
    demonstrate_weighted_voting()
    
    # Summary
    print(f"\n\n{'='*60}")
    print("KEY BENEFITS OF SELF-CONSISTENCY")
    print(f"{'='*60}")
    print("✓ Reduces hallucinations and errors")
    print("✓ Increases confidence in answers")
    print("✓ Finds consensus across diverse reasoning paths")
    print("✓ More robust than single-path reasoning")
    print("✓ Can detect and handle ambiguous cases")


if __name__ == "__main__":
    main()
