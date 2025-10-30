"""
Curriculum Learning Pattern

An agent that learns through progressively harder tasks, starting with easy
problems and gradually increasing difficulty. This mirrors human learning
and leads to better performance and faster convergence.

Use Cases:
- Training complex models with limited data
- Educational systems for personalized learning
- Game AI development with skill progression
- Robotic learning with safety requirements
- Language learning with scaffolded difficulty

Benefits:
- Faster convergence: Learn fundamentals before complex patterns
- Better generalization: Strong foundation enables transfer
- Improved stability: Avoid early catastrophic failures
- Sample efficiency: Learn more from each example
- Natural progression: Mirrors human learning processes
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import random


class DifficultyLevel(Enum):
    """Task difficulty levels"""
    TRIVIAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5


class ProgressionStrategy(Enum):
    """Strategies for curriculum progression"""
    LINEAR = "linear"  # Fixed progression through levels
    PERFORMANCE_BASED = "performance_based"  # Advance when ready
    ADAPTIVE = "adaptive"  # Dynamically adjust difficulty
    SPIRAL = "spiral"  # Revisit concepts with increasing complexity


@dataclass
class LearningTask:
    """A task in the curriculum"""
    task_id: str
    description: str
    difficulty: DifficultyLevel
    prerequisites: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    mastery_threshold: float = 0.8
    
    def __str__(self) -> str:
        return f"{self.task_id} ({self.difficulty.name})"


@dataclass
class LearningProgress:
    """Tracks learner's progress"""
    current_level: DifficultyLevel = DifficultyLevel.TRIVIAL
    completed_tasks: List[str] = field(default_factory=list)
    task_scores: Dict[str, float] = field(default_factory=dict)
    attempts_per_task: Dict[str, int] = field(default_factory=dict)
    learning_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_attempt(self, task_id: str, score: float) -> None:
        """Record learning attempt"""
        self.task_scores[task_id] = score
        self.attempts_per_task[task_id] = self.attempts_per_task.get(task_id, 0) + 1
        self.learning_history.append({
            "task_id": task_id,
            "score": score,
            "attempt": self.attempts_per_task[task_id]
        })
    
    def mark_completed(self, task_id: str) -> None:
        """Mark task as completed"""
        if task_id not in self.completed_tasks:
            self.completed_tasks.append(task_id)
    
    def get_average_score(self) -> float:
        """Get average score across all tasks"""
        if not self.task_scores:
            return 0.0
        return sum(self.task_scores.values()) / len(self.task_scores)
    
    def is_task_mastered(self, task_id: str, threshold: float = 0.8) -> bool:
        """Check if task is mastered"""
        return self.task_scores.get(task_id, 0.0) >= threshold


@dataclass
class Curriculum:
    """A structured learning curriculum"""
    name: str
    tasks: List[LearningTask]
    progression_strategy: ProgressionStrategy = ProgressionStrategy.PERFORMANCE_BASED
    
    def get_tasks_by_difficulty(self, level: DifficultyLevel) -> List[LearningTask]:
        """Get all tasks at a difficulty level"""
        return [task for task in self.tasks if task.difficulty == level]
    
    def get_next_tasks(
        self,
        progress: LearningProgress,
        max_tasks: int = 3
    ) -> List[LearningTask]:
        """Get next appropriate tasks based on progress"""
        available_tasks = []
        
        for task in self.tasks:
            # Skip completed tasks
            if task.task_id in progress.completed_tasks:
                continue
            
            # Check prerequisites
            prerequisites_met = all(
                prereq in progress.completed_tasks
                for prereq in task.prerequisites
            )
            
            if not prerequisites_met:
                continue
            
            # Check difficulty level is appropriate
            if task.difficulty.value <= progress.current_level.value + 1:
                available_tasks.append(task)
        
        # Sort by difficulty and return top tasks
        available_tasks.sort(key=lambda t: t.difficulty.value)
        return available_tasks[:max_tasks]
    
    def validate_curriculum(self) -> List[str]:
        """Validate curriculum structure"""
        issues = []
        
        # Check for circular dependencies
        task_ids = {task.task_id for task in self.tasks}
        
        for task in self.tasks:
            for prereq in task.prerequisites:
                if prereq not in task_ids:
                    issues.append(
                        f"Task {task.task_id} has unknown prerequisite: {prereq}"
                    )
        
        # Check difficulty progression
        for task in self.tasks:
            for prereq in task.prerequisites:
                prereq_task = next(
                    (t for t in self.tasks if t.task_id == prereq),
                    None
                )
                if prereq_task and prereq_task.difficulty.value > task.difficulty.value:
                    issues.append(
                        f"Task {task.task_id} easier than prerequisite {prereq}"
                    )
        
        return issues


class CurriculumLearningAgent:
    """
    Curriculum Learning Agent
    
    Learns through a structured curriculum of progressively harder tasks,
    adapting the difficulty based on performance and mastery.
    """
    
    def __init__(self, name: str = "Curriculum Learner"):
        self.name = name
        self.progress = LearningProgress()
        self.curriculum: Optional[Curriculum] = None
        self.learning_rate = 0.1
        self.patience = 3  # Attempts before adjusting difficulty
    
    def load_curriculum(self, curriculum: Curriculum) -> None:
        """Load a learning curriculum"""
        print(f"\n[Curriculum] Loading: {curriculum.name}")
        print(f"  Total tasks: {len(curriculum.tasks)}")
        print(f"  Progression: {curriculum.progression_strategy.value}")
        
        # Validate curriculum
        issues = curriculum.validate_curriculum()
        if issues:
            print("\n⚠ Curriculum validation issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("  ✓ Curriculum validated successfully")
        
        self.curriculum = curriculum
    
    def start_learning(self) -> None:
        """Begin curriculum-based learning"""
        if not self.curriculum:
            print("Error: No curriculum loaded")
            return
        
        print(f"\n{'=' * 70}")
        print(f"STARTING CURRICULUM: {self.curriculum.name}")
        print(f"{'=' * 70}")
        
        iteration = 0
        max_iterations = 20  # Prevent infinite loops in demo
        
        while iteration < max_iterations:
            # Get next appropriate tasks
            next_tasks = self.curriculum.get_next_tasks(self.progress)
            
            if not next_tasks:
                print(f"\n🎓 Curriculum completed!")
                break
            
            # Learn from next task
            task = next_tasks[0]  # Focus on one task at a time
            print(f"\n--- Iteration {iteration + 1} ---")
            self._learn_task(task)
            
            # Check if ready to progress
            if self._should_advance_level():
                self._advance_difficulty_level()
            
            iteration += 1
        
        self._print_final_report()
    
    def _learn_task(self, task: LearningTask) -> None:
        """Learn from a single task"""
        print(f"\n[Learning] Task: {task.task_id}")
        print(f"  Difficulty: {task.difficulty.name}")
        print(f"  Objectives: {', '.join(task.learning_objectives)}")
        
        # Simulate learning (in production: actual training)
        score = self._simulate_learning(task)
        
        # Record attempt
        self.progress.record_attempt(task.task_id, score)
        attempts = self.progress.attempts_per_task[task.task_id]
        
        print(f"  Attempt {attempts}: Score = {score:.3f}")
        
        # Check mastery
        if self.progress.is_task_mastered(task.task_id, task.mastery_threshold):
            self.progress.mark_completed(task.task_id)
            print(f"  ✓ Task mastered! (threshold: {task.mastery_threshold})")
        else:
            needed = task.mastery_threshold - score
            print(f"  ↻ Need {needed:.3f} more to master")
    
    def _simulate_learning(self, task: LearningTask) -> float:
        """
        Simulate learning from a task.
        In production: actual model training/fine-tuning.
        """
        # Base performance depends on difficulty and current level
        difficulty_gap = task.difficulty.value - self.progress.current_level.value
        base_score = max(0.3, 1.0 - (difficulty_gap * 0.2))
        
        # Improve with attempts
        attempts = self.progress.attempts_per_task.get(task.task_id, 0)
        improvement = min(0.3, attempts * 0.1)
        
        # Add some randomness
        noise = random.uniform(-0.1, 0.1)
        
        score = min(1.0, max(0.0, base_score + improvement + noise))
        return score
    
    def _should_advance_level(self) -> bool:
        """Check if ready to advance difficulty level"""
        if not self.curriculum:
            return False
        
        # Get recent scores at current level
        current_level_tasks = [
            task for task in self.curriculum.tasks
            if task.difficulty == self.progress.current_level
        ]
        
        if not current_level_tasks:
            return True
        
        # Check mastery of current level tasks
        mastered = sum(
            1 for task in current_level_tasks
            if self.progress.is_task_mastered(task.task_id)
        )
        
        mastery_rate = mastered / len(current_level_tasks)
        return mastery_rate >= 0.7  # Advance when 70% mastered
    
    def _advance_difficulty_level(self) -> None:
        """Advance to next difficulty level"""
        current = self.progress.current_level
        
        if current.value < DifficultyLevel.EXPERT.value:
            new_level = DifficultyLevel(current.value + 1)
            self.progress.current_level = new_level
            
            print(f"\n🎯 Level Up! {current.name} → {new_level.name}")
    
    def _print_final_report(self) -> None:
        """Print final learning report"""
        if not self.curriculum:
            return
        
        print(f"\n{'=' * 70}")
        print("LEARNING REPORT")
        print(f"{'=' * 70}")
        
        print(f"\nFinal Level: {self.progress.current_level.name}")
        print(f"Tasks Completed: {len(self.progress.completed_tasks)}")
        print(f"Average Score: {self.progress.get_average_score():.3f}")
        
        print("\n📊 Performance by Difficulty:")
        for level in DifficultyLevel:
            level_tasks = [
                task for task in self.curriculum.tasks
                if task.difficulty == level
            ]
            
            if level_tasks:
                completed = sum(
                    1 for task in level_tasks
                    if task.task_id in self.progress.completed_tasks
                )
                print(f"  {level.name:10s}: {completed}/{len(level_tasks)} completed")
    
    def get_learning_curve(self) -> List[float]:
        """Get learning curve (scores over time)"""
        return [entry["score"] for entry in self.progress.learning_history]
    
    def recommend_practice(self) -> List[str]:
        """Recommend tasks to practice"""
        if not self.curriculum:
            return []
        
        recommendations = []
        
        # Tasks that need more practice (below mastery)
        for task in self.curriculum.tasks:
            if (task.task_id in self.progress.task_scores and
                not self.progress.is_task_mastered(task.task_id)):
                recommendations.append(
                    f"Practice {task.task_id} (current: "
                    f"{self.progress.task_scores[task.task_id]:.2f})"
                )
        
        return recommendations[:5]  # Top 5 recommendations


def demonstrate_curriculum_learning():
    """
    Demonstrate Curriculum Learning pattern
    """
    print("=" * 70)
    print("CURRICULUM LEARNING PATTERN DEMONSTRATION")
    print("=" * 70)
    
    # Example 1: Math curriculum
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Mathematics Learning Curriculum")
    print("=" * 70)
    
    # Create math curriculum
    math_tasks = [
        # TRIVIAL level
        LearningTask(
            task_id="count_to_10",
            description="Count from 1 to 10",
            difficulty=DifficultyLevel.TRIVIAL,
            learning_objectives=["Number recognition", "Counting"]
        ),
        LearningTask(
            task_id="basic_addition",
            description="Add single-digit numbers",
            difficulty=DifficultyLevel.TRIVIAL,
            prerequisites=["count_to_10"],
            learning_objectives=["Addition", "Single digits"]
        ),
        
        # EASY level
        LearningTask(
            task_id="basic_subtraction",
            description="Subtract single-digit numbers",
            difficulty=DifficultyLevel.EASY,
            prerequisites=["basic_addition"],
            learning_objectives=["Subtraction", "Inverse operations"]
        ),
        LearningTask(
            task_id="two_digit_addition",
            description="Add two-digit numbers",
            difficulty=DifficultyLevel.EASY,
            prerequisites=["basic_addition"],
            learning_objectives=["Carrying", "Place value"]
        ),
        
        # MEDIUM level
        LearningTask(
            task_id="multiplication",
            description="Multiply single-digit numbers",
            difficulty=DifficultyLevel.MEDIUM,
            prerequisites=["basic_addition", "basic_subtraction"],
            learning_objectives=["Multiplication", "Times tables"],
            mastery_threshold=0.85
        ),
        LearningTask(
            task_id="division",
            description="Divide numbers with remainders",
            difficulty=DifficultyLevel.MEDIUM,
            prerequisites=["multiplication"],
            learning_objectives=["Division", "Remainders"],
            mastery_threshold=0.85
        ),
        
        # HARD level
        LearningTask(
            task_id="fractions",
            description="Operations with fractions",
            difficulty=DifficultyLevel.HARD,
            prerequisites=["multiplication", "division"],
            learning_objectives=["Fractions", "Rational numbers"],
            mastery_threshold=0.9
        ),
        
        # EXPERT level
        LearningTask(
            task_id="algebra",
            description="Solve algebraic equations",
            difficulty=DifficultyLevel.EXPERT,
            prerequisites=["fractions", "multiplication", "division"],
            learning_objectives=["Variables", "Equation solving"],
            mastery_threshold=0.9
        )
    ]
    
    math_curriculum = Curriculum(
        name="Elementary Mathematics",
        tasks=math_tasks,
        progression_strategy=ProgressionStrategy.PERFORMANCE_BASED
    )
    
    # Create agent and learn
    agent = CurriculumLearningAgent("Math Student")
    agent.load_curriculum(math_curriculum)
    agent.start_learning()
    
    # Show learning curve
    curve = agent.get_learning_curve()
    if curve:
        print(f"\n📈 Learning Curve (first 10 attempts):")
        for i, score in enumerate(curve[:10], 1):
            bar = "█" * int(score * 20)
            print(f"  {i:2d}: {bar} {score:.3f}")
    
    # Get recommendations
    recommendations = agent.recommend_practice()
    if recommendations:
        print(f"\n💡 Practice Recommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    
    # Example 2: Language learning curriculum
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Language Learning Curriculum")
    print("=" * 70)
    
    language_tasks = [
        LearningTask(
            task_id="greetings",
            description="Learn basic greetings",
            difficulty=DifficultyLevel.TRIVIAL,
            learning_objectives=["Hello", "Goodbye", "Thank you"]
        ),
        LearningTask(
            task_id="introductions",
            description="Introduce yourself",
            difficulty=DifficultyLevel.EASY,
            prerequisites=["greetings"],
            learning_objectives=["Name", "Age", "Origin"]
        ),
        LearningTask(
            task_id="simple_conversations",
            description="Have simple conversations",
            difficulty=DifficultyLevel.MEDIUM,
            prerequisites=["introductions"],
            learning_objectives=["Questions", "Answers", "Turn-taking"]
        ),
        LearningTask(
            task_id="complex_topics",
            description="Discuss complex topics",
            difficulty=DifficultyLevel.HARD,
            prerequisites=["simple_conversations"],
            learning_objectives=["Abstract concepts", "Opinions", "Debate"]
        )
    ]
    
    language_curriculum = Curriculum(
        name="Language Fundamentals",
        tasks=language_tasks,
        progression_strategy=ProgressionStrategy.ADAPTIVE
    )
    
    agent2 = CurriculumLearningAgent("Language Learner")
    agent2.load_curriculum(language_curriculum)
    agent2.start_learning()


def demonstrate_progression_strategies():
    """Show different curriculum progression strategies"""
    print("\n" + "=" * 70)
    print("CURRICULUM PROGRESSION STRATEGIES")
    print("=" * 70)
    
    print("\n1. LINEAR PROGRESSION:")
    print("   - Fixed sequence through all levels")
    print("   - Predictable, structured learning")
    print("   - Good for: Well-understood domains, certification programs")
    
    print("\n2. PERFORMANCE-BASED:")
    print("   - Advance when mastery threshold reached")
    print("   - Personalized pacing")
    print("   - Good for: Adaptive learning, diverse learners")
    
    print("\n3. ADAPTIVE:")
    print("   - Dynamically adjust difficulty based on performance")
    print("   - Optimal challenge level maintained")
    print("   - Good for: Engagement, flow state, skill development")
    
    print("\n4. SPIRAL:")
    print("   - Revisit concepts with increasing complexity")
    print("   - Reinforcement and deepening understanding")
    print("   - Good for: Complex domains, long-term retention")


def demonstrate_benefits():
    """Show benefits of curriculum learning"""
    print("\n" + "=" * 70)
    print("CURRICULUM LEARNING BENEFITS")
    print("=" * 70)
    
    print("\n✓ Key Advantages:")
    print("\n1. FASTER CONVERGENCE:")
    print("   - Learn fundamentals before complex patterns")
    print("   - Build on strong foundation")
    print("   - Avoid early catastrophic failures")
    
    print("\n2. BETTER GENERALIZATION:")
    print("   - Strong foundation enables transfer")
    print("   - Understanding principles, not just examples")
    print("   - Robust to variations")
    
    print("\n3. IMPROVED STABILITY:")
    print("   - Gradual increase in complexity")
    print("   - Less likely to break existing knowledge")
    print("   - Smoother learning curves")
    
    print("\n4. SAMPLE EFFICIENCY:")
    print("   - Learn more from each example")
    print("   - Prerequisites provide context")
    print("   - Reduced redundant learning")
    
    print("\n5. NATURAL PROGRESSION:")
    print("   - Mirrors human learning")
    print("   - Maintains motivation")
    print("   - Clear sense of progress")


if __name__ == "__main__":
    demonstrate_curriculum_learning()
    demonstrate_progression_strategies()
    demonstrate_benefits()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Curriculum learning uses progressive difficulty for better outcomes
2. Start with easy tasks to build foundation
3. Advance based on mastery, not just time
4. Structure ensures prerequisites are met
5. Adaptively adjust difficulty to maintain engagement

Best Practices:
- Design clear learning objectives for each task
- Define realistic mastery thresholds
- Validate curriculum structure for dependencies
- Track progress and adjust pacing
- Provide practice recommendations
- Balance challenge and achievability
- Consider multiple progression strategies
    """)
