"""
Reflexion Pattern
=================
Agent reflects on past failures and successes to improve.
Components: Actor, Evaluator, Self-Reflection, Memory
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class OutcomeType(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


@dataclass
class Experience:
    """Represents a past experience"""
    task: str
    attempt: int
    action: str
    outcome: OutcomeType
    feedback: str
    reflection: Optional[str] = None


class Memory:
    """Stores past experiences and reflections"""
    
    def __init__(self):
        self.experiences: List[Experience] = []
    
    def add_experience(self, experience: Experience):
        """Add an experience to memory"""
        self.experiences.append(experience)
    
    def get_relevant_experiences(self, task: str) -> List[Experience]:
        """Retrieve experiences relevant to the current task"""
        # Simple keyword matching (in real scenario, use semantic similarity)
        relevant = []
        task_keywords = set(task.lower().split())
        
        for exp in self.experiences:
            exp_keywords = set(exp.task.lower().split())
            if task_keywords & exp_keywords:  # If there's overlap
                relevant.append(exp)
        
        return relevant
    
    def get_successful_strategies(self, task: str) -> List[str]:
        """Get strategies that led to success for similar tasks"""
        relevant = self.get_relevant_experiences(task)
        return [exp.reflection for exp in relevant 
                if exp.outcome == OutcomeType.SUCCESS and exp.reflection]


class Actor:
    """Performs actions to complete tasks"""
    
    def generate_code(self, task: str, attempt: int, memory: Memory) -> str:
        """
        Generates code to solve the task
        Uses reflections from memory to improve
        """
        # Get past learnings
        strategies = memory.get_successful_strategies(task)
        
        print(f"\n📝 Generating solution (Attempt {attempt})...")
        
        if strategies:
            print(f"💡 Using insights from past experiences:")
            for i, strategy in enumerate(strategies[-3:], 1):  # Use last 3
                print(f"   {i}. {strategy}")
        
        # Simulate code generation (with improvements based on attempt)
        if "sort list" in task.lower():
            if attempt == 1:
                # First attempt - simple but buggy
                code = """
def sort_list(arr):
    # Bug: doesn't handle edge cases
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
    return arr
"""
            elif attempt == 2:
                # Second attempt - fixes edge cases after reflection
                code = """
def sort_list(arr):
    # Fixed: handle empty list and None
    if not arr:
        return []
    result = arr.copy()  # Don't modify original
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if result[i] > result[j]:
                result[i], result[j] = result[j], result[i]
    return result
"""
            else:
                # Third attempt - optimal solution after reflection
                code = """
def sort_list(arr):
    # Optimized: use built-in sort for efficiency
    if not arr:
        return []
    return sorted(arr)
"""
        
        elif "palindrome" in task.lower():
            if attempt == 1:
                code = """
def is_palindrome(s):
    # Bug: case sensitive
    return s == s[::-1]
"""
            elif attempt == 2:
                code = """
def is_palindrome(s):
    # Fixed: case insensitive
    s = s.lower()
    return s == s[::-1]
"""
            else:
                code = """
def is_palindrome(s):
    # Optimized: handle spaces and punctuation
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]
"""
        else:
            code = f"# Generated code for: {task}\npass"
        
        return code.strip()


class Evaluator:
    """Evaluates the quality of actions"""
    
    def evaluate_code(self, code: str, task: str) -> tuple[OutcomeType, str]:
        """
        Evaluates generated code
        Returns: (outcome, feedback)
        """
        print(f"\n🔍 Evaluating solution...")
        
        feedback_items = []
        
        # Check for common issues
        if "if not arr:" not in code and "sort" in task.lower():
            feedback_items.append("- Missing edge case handling for empty input")
        
        if "copy()" not in code and "sort" in task.lower() and "arr[i], arr[j]" in code:
            feedback_items.append("- Should not modify original input")
        
        if "lower()" not in code and "palindrome" in task.lower():
            feedback_items.append("- Not handling case sensitivity")
        
        if "isalnum()" not in code and "palindrome" in task.lower():
            feedback_items.append("- Not handling special characters")
        
        if "sorted(" in code or ("isalnum()" in code and "palindrome" in task.lower()):
            outcome = OutcomeType.SUCCESS
            feedback = "✅ Solution is correct and handles edge cases well!"
        elif feedback_items:
            outcome = OutcomeType.PARTIAL
            feedback = "⚠️  Solution has issues:\n" + "\n".join(feedback_items)
        else:
            outcome = OutcomeType.SUCCESS
            feedback = "✅ Solution looks good!"
        
        print(feedback)
        return outcome, feedback


class ReflectionEngine:
    """Generates reflections from experiences"""
    
    def reflect(self, experience: Experience, memory: Memory) -> str:
        """
        Generates a reflection based on the experience
        """
        print(f"\n🤔 Reflecting on attempt {experience.attempt}...")
        
        if experience.outcome == OutcomeType.SUCCESS:
            reflection = f"Successful approach: {self._extract_success_pattern(experience)}"
            print(f"💭 {reflection}")
            
        elif experience.outcome == OutcomeType.FAILURE:
            # Analyze what went wrong
            past_failures = [exp for exp in memory.experiences 
                           if exp.task == experience.task and exp.outcome != OutcomeType.SUCCESS]
            
            reflection = f"Avoid: {self._extract_failure_pattern(experience)}. "
            reflection += self._generate_improvement_suggestion(experience, past_failures)
            print(f"💭 {reflection}")
            
        else:  # PARTIAL
            reflection = self._generate_improvement_suggestion(experience, [])
            print(f"💭 {reflection}")
        
        return reflection
    
    def _extract_success_pattern(self, experience: Experience) -> str:
        """Extract what made the solution successful"""
        if "sorted(" in experience.action:
            return "Use built-in functions when available for efficiency"
        elif "isalnum()" in experience.action:
            return "Handle special characters and normalization"
        elif "copy()" in experience.action:
            return "Don't modify input data, create a copy"
        else:
            return "Solution correctly handles the requirements"
    
    def _extract_failure_pattern(self, experience: Experience) -> str:
        """Extract what caused the failure"""
        if "edge case" in experience.feedback.lower():
            return "missing edge case handling"
        elif "case sensitive" in experience.feedback.lower():
            return "not considering case sensitivity"
        elif "modify" in experience.feedback.lower():
            return "modifying input data"
        else:
            return "incomplete implementation"
    
    def _generate_improvement_suggestion(self, experience: Experience, past_failures: List[Experience]) -> str:
        """Generate suggestions for improvement"""
        suggestions = []
        
        if "empty" in experience.feedback.lower():
            suggestions.append("Add checks for empty/None inputs")
        
        if "case" in experience.feedback.lower():
            suggestions.append("Normalize case before comparison")
        
        if "special char" in experience.feedback.lower() or "punctuation" in experience.feedback.lower():
            suggestions.append("Filter out special characters")
        
        if "modify" in experience.feedback.lower():
            suggestions.append("Create a copy to avoid side effects")
        
        if len(past_failures) > 1:
            suggestions.append("Consider using built-in functions for robustness")
        
        return "Try: " + "; ".join(suggestions) if suggestions else "Review implementation carefully"


class ReflexionAgent:
    """Main Reflexion agent that learns from experience"""
    
    def __init__(self, max_attempts: int = 3):
        self.actor = Actor()
        self.evaluator = Evaluator()
        self.reflection_engine = ReflectionEngine()
        self.memory = Memory()
        self.max_attempts = max_attempts
    
    def solve_task(self, task: str) -> Dict:
        """
        Solves a task using the Reflexion pattern
        """
        print(f"\n{'='*70}")
        print(f"🎯 Task: {task}")
        print(f"{'='*70}")
        
        for attempt in range(1, self.max_attempts + 1):
            print(f"\n{'─'*70}")
            print(f"Attempt {attempt}/{self.max_attempts}")
            print(f"{'─'*70}")
            
            # Generate solution
            code = self.actor.generate_code(task, attempt, self.memory)
            print(f"\nGenerated code:")
            print("```python")
            print(code)
            print("```")
            
            # Evaluate solution
            outcome, feedback = self.evaluator.evaluate_code(code, task)
            
            # Create experience
            experience = Experience(
                task=task,
                attempt=attempt,
                action=code,
                outcome=outcome,
                feedback=feedback
            )
            
            # Reflect on experience
            reflection = self.reflection_engine.reflect(experience, self.memory)
            experience.reflection = reflection
            
            # Store in memory
            self.memory.add_experience(experience)
            
            # Check if we succeeded
            if outcome == OutcomeType.SUCCESS:
                print(f"\n{'='*70}")
                print(f"✅ Task completed successfully in {attempt} attempt(s)!")
                print(f"{'='*70}")
                return {
                    "success": True,
                    "attempts": attempt,
                    "final_code": code,
                    "experiences": self.memory.experiences
                }
        
        print(f"\n{'='*70}")
        print(f"❌ Failed to complete task after {self.max_attempts} attempts")
        print(f"{'='*70}")
        return {
            "success": False,
            "attempts": self.max_attempts,
            "final_code": code,
            "experiences": self.memory.experiences
        }


def main():
    """Demonstrate Reflexion pattern"""
    
    # Example 1: Sorting problem
    print("\n" + "="*70)
    print("EXAMPLE 1: Learn to Sort Lists Correctly")
    print("="*70)
    
    agent1 = ReflexionAgent(max_attempts=3)
    result1 = agent1.solve_task("Write a function to sort a list of integers")
    
    # Example 2: Palindrome check (new agent, but can benefit from similar task patterns)
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Learn to Check Palindromes")
    print("="*70)
    
    agent2 = ReflexionAgent(max_attempts=3)
    result2 = agent2.solve_task("Write a function to check if a string is a palindrome")
    
    # Show learning progress
    print(f"\n\n{'='*70}")
    print("LEARNING SUMMARY")
    print(f"{'='*70}\n")
    
    print("Task 1: Sorting")
    print(f"  Attempts needed: {result1['attempts']}")
    print(f"  Success: {'Yes' if result1['success'] else 'No'}")
    
    print("\nTask 2: Palindrome")
    print(f"  Attempts needed: {result2['attempts']}")
    print(f"  Success: {'Yes' if result2['success'] else 'No'}")
    
    print(f"\n{'='*70}")
    print("KEY INSIGHTS FROM REFLEXION")
    print(f"{'='*70}")
    print("✓ Agent learns from mistakes through self-reflection")
    print("✓ Past experiences guide future attempts")
    print("✓ Iterative improvement leads to better solutions")
    print("✓ Memory of successes and failures builds expertise")


if __name__ == "__main__":
    main()
