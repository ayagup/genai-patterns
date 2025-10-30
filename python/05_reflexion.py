"""
Reflexion Pattern
Agent reflects on past failures to improve future performance
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
@dataclass
class Experience:
    task: str
    action: str
    outcome: str
    success: bool
    timestamp: datetime
    reflection: str = ""
class ReflexionAgent:
    def __init__(self):
        self.memory: List[Experience] = []
        self.max_attempts = 3
    def attempt_task(self, task: str, attempt_num: int) -> Tuple[str, bool]:
        """Attempt to complete a task"""
        print(f"\n--- Attempt {attempt_num} ---")
        print(f"Task: {task}")
        # Retrieve relevant past experiences
        relevant_memories = self.retrieve_relevant_experiences(task)
        # Use reflections to improve current attempt
        strategy = self.plan_with_reflection(task, relevant_memories)
        print(f"Strategy: {strategy}")
        # Simulate task execution
        action, success = self.execute(task, strategy, attempt_num)
        return action, success
    def retrieve_relevant_experiences(self, task: str) -> List[Experience]:
        """Retrieve similar past experiences"""
        # Simple keyword matching (in real implementation, use embeddings)
        relevant = []
        task_keywords = set(task.lower().split())
        for exp in self.memory:
            exp_keywords = set(exp.task.lower().split())
            if task_keywords & exp_keywords:  # If any overlap
                relevant.append(exp)
        return relevant
    def plan_with_reflection(self, task: str, memories: List[Experience]) -> str:
        """Plan strategy based on past reflections"""
        if not memories:
            return "Try standard approach"
        # Learn from failures
        failures = [m for m in memories if not m.success]
        if failures:
            latest_failure = failures[-1]
            return f"Avoid: {latest_failure.reflection}. Try alternative approach."
        # Learn from successes
        successes = [m for m in memories if m.success]
        if successes:
            latest_success = successes[-1]
            return f"Replicate successful approach: {latest_success.reflection}"
        return "Try standard approach"
    def execute(self, task: str, strategy: str, attempt: int) -> Tuple[str, bool]:
        """Execute the task (simulated)"""
        # Simulate task execution with increasing success probability
        import random
        # Example: Code generation task
        if "generate code" in task.lower():
            if attempt == 1:
                action = "Generated code without error handling"
                success = False
            elif attempt == 2:
                action = "Generated code with basic error handling"
                success = random.random() > 0.3
            else:
                action = "Generated code with comprehensive error handling and tests"
                success = True
        else:
            action = f"Executed strategy: {strategy}"
            success = random.random() > (0.7 - attempt * 0.2)
        print(f"Action: {action}")
        print(f"Success: {success}")
        return action, success
    def reflect(self, task: str, action: str, success: bool) -> str:
        """Reflect on the outcome and learn"""
        print("\n--- Reflection ---")
        if success:
            reflection = f"Success! The approach '{action}' worked well for '{task}'"
        else:
            # Analyze what went wrong
            if "error handling" in action.lower() and not success:
                reflection = "Failure: Need to add more comprehensive error handling"
            elif "without" in action.lower():
                reflection = "Failure: The basic approach was insufficient, need to add safety measures"
            else:
                reflection = "Failure: Current approach didn't work, need to reconsider strategy"
        print(f"Reflection: {reflection}")
        return reflection
    def learn_from_experience(self, experience: Experience):
        """Store experience in memory"""
        self.memory.append(experience)
        print(f"Learned: Stored experience in memory (total experiences: {len(self.memory)})")
    def run(self, task: str) -> bool:
        """Run task with reflexion loop"""
        print(f"\n{'='*60}")
        print(f"Starting Reflexion Loop for: {task}")
        print(f"{'='*60}")
        for attempt in range(1, self.max_attempts + 1):
            # Attempt task
            action, success = self.attempt_task(task, attempt)
            # Reflect on outcome
            reflection = self.reflect(task, action, success)
            # Store experience
            experience = Experience(
                task=task,
                action=action,
                outcome="success" if success else "failure",
                success=success,
                timestamp=datetime.now(),
                reflection=reflection
            )
            self.learn_from_experience(experience)
            # If successful, stop
            if success:
                print(f"\n✓ Task completed successfully on attempt {attempt}")
                return True
            # If not last attempt, continue loop
            if attempt < self.max_attempts:
                print(f"\n⟲ Retrying with improved approach...")
        print(f"\n✗ Task failed after {self.max_attempts} attempts")
        return False
# Usage
if __name__ == "__main__":
    agent = ReflexionAgent()
    # First task
    task1 = "Generate code for file processing"
    agent.run(task1)
    print("\n" + "="*60)
    print("Now attempting a similar task...")
    print("="*60)
    # Similar task - should benefit from previous reflection
    task2 = "Generate code for data processing"
    agent.run(task2)
    # Show memory
    print(f"\n=== Agent Memory ({len(agent.memory)} experiences) ===")
    for i, exp in enumerate(agent.memory, 1):
        print(f"\n{i}. Task: {exp.task}")
        print(f"   Action: {exp.action}")
        print(f"   Success: {exp.success}")
        print(f"   Reflection: {exp.reflection}")
