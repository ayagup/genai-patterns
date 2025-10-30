"""
Pattern 060: Curriculum Learning

Description:
    Curriculum Learning structures the learning process by presenting training
    examples in a meaningful order, from simple to complex. Like human education,
    this progressive approach helps agents build foundational skills before
    tackling harder problems, leading to faster learning and better performance.

Components:
    1. Difficulty Scorer: Evaluates task complexity
    2. Curriculum Designer: Orders tasks by difficulty
    3. Pacing Strategy: Determines progression speed
    4. Performance Monitor: Tracks learning progress
    5. Adaptive Scheduler: Adjusts curriculum based on performance

Use Cases:
    - Training complex reasoning agents
    - Gradual skill acquisition
    - Hierarchical task learning
    - Educational tutoring systems
    - Reinforcement learning with sparse rewards
    - Knowledge scaffolding

LangChain Implementation:
    Implements curriculum learning by ordering prompts and examples from
    simple to complex, adapting progression based on agent performance.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics
import random

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class DifficultyLevel(Enum):
    """Difficulty levels for curriculum"""
    TRIVIAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5


class PacingStrategy(Enum):
    """Strategies for curriculum pacing"""
    FIXED = "fixed"  # Predetermined progression
    ADAPTIVE = "adaptive"  # Based on performance
    MIXED = "mixed"  # Combine easy and hard
    SPIRAL = "spiral"  # Revisit with increasing complexity
    COMPETENCY_BASED = "competency_based"  # Advance when mastered


@dataclass
class LearningExample:
    """A single learning example"""
    example_id: str
    task_type: str
    input_text: str
    expected_output: str
    difficulty: DifficultyLevel
    prerequisites: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_prerequisite_met(self, mastered_examples: Set[str]) -> bool:
        """Check if prerequisites are satisfied"""
        return all(prereq in mastered_examples for prereq in self.prerequisites)


@dataclass
class CurriculumStage:
    """A stage in the curriculum"""
    stage_id: int
    stage_name: str
    difficulty_range: Tuple[DifficultyLevel, DifficultyLevel]
    examples: List[LearningExample]
    mastery_threshold: float = 0.8
    
    @property
    def num_examples(self) -> int:
        return len(self.examples)


@dataclass
class PerformanceRecord:
    """Record of performance on an example"""
    example_id: str
    attempt_number: int
    correct: bool
    response: str
    timestamp: datetime
    time_taken_ms: float


@dataclass
class CurriculumResult:
    """Result from curriculum learning"""
    stages_completed: int
    total_examples: int
    examples_mastered: int
    overall_accuracy: float
    learning_curve: List[Tuple[int, float]]  # (example_num, accuracy)
    time_per_stage: List[float]
    final_performance: float
    total_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stages_completed": self.stages_completed,
            "examples_mastered": f"{self.examples_mastered}/{self.total_examples}",
            "overall_accuracy": f"{self.overall_accuracy:.1%}",
            "final_performance": f"{self.final_performance:.1%}",
            "total_time_ms": f"{self.total_time_ms:.1f}"
        }


class CurriculumLearner:
    """
    Agent that learns through curriculum progression.
    
    Features:
    1. Difficulty-ordered example presentation
    2. Performance-based pacing
    3. Mastery tracking
    4. Adaptive progression
    5. Skill scaffolding
    """
    
    def __init__(
        self,
        pacing_strategy: PacingStrategy = PacingStrategy.ADAPTIVE,
        mastery_threshold: float = 0.8,
        temperature: float = 0.7
    ):
        self.pacing_strategy = pacing_strategy
        self.mastery_threshold = mastery_threshold
        self.temperature = temperature
        
        # Learner LLM
        self.learner = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=temperature
        )
        
        # Difficulty assessor
        self.assessor = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2
        )
        
        # Learning state
        self.mastered_examples: Set[str] = set()
        self.performance_history: List[PerformanceRecord] = []
        self.current_stage: int = 0
        
        # Learning statistics
        self.learning_curve: List[Tuple[int, float]] = []
    
    def _assess_difficulty(self, example: LearningExample) -> DifficultyLevel:
        """Assess difficulty of an example"""
        
        # If already labeled, use that
        if example.difficulty != DifficultyLevel.MEDIUM:
            return example.difficulty
        
        # Otherwise, assess automatically
        assessment_prompt = ChatPromptTemplate.from_messages([
            ("system", """Rate the difficulty of this learning task on a scale:
1 - Trivial (very simple, obvious answer)
2 - Easy (straightforward, basic reasoning)
3 - Medium (requires some thought)
4 - Hard (complex, multiple steps)
5 - Expert (very challenging, advanced concepts)

Respond with just the number."""),
            ("user", "Task: {input}\nExpected: {output}\n\nDifficulty:")
        ])
        
        chain = assessment_prompt | self.assessor | StrOutputParser()
        
        try:
            rating = chain.invoke({
                "input": example.input_text,
                "output": example.expected_output
            })
            
            difficulty_num = int(''.join(c for c in rating if c.isdigit()))
            return DifficultyLevel(min(max(difficulty_num, 1), 5))
        except:
            return DifficultyLevel.MEDIUM
    
    def _create_curriculum(
        self,
        examples: List[LearningExample]
    ) -> List[CurriculumStage]:
        """Create curriculum stages from examples"""
        
        # Assess difficulties if needed
        for example in examples:
            if example.difficulty == DifficultyLevel.MEDIUM and not example.metadata.get("assessed"):
                example.difficulty = self._assess_difficulty(example)
                example.metadata["assessed"] = True
        
        # Group by difficulty
        stages_dict: Dict[DifficultyLevel, List[LearningExample]] = {}
        for example in examples:
            if example.difficulty not in stages_dict:
                stages_dict[example.difficulty] = []
            stages_dict[example.difficulty].append(example)
        
        # Create stages
        stages = []
        for difficulty in sorted(stages_dict.keys(), key=lambda x: x.value):
            stage = CurriculumStage(
                stage_id=len(stages),
                stage_name=f"Stage {len(stages)+1}: {difficulty.name}",
                difficulty_range=(difficulty, difficulty),
                examples=stages_dict[difficulty],
                mastery_threshold=self.mastery_threshold
            )
            stages.append(stage)
        
        return stages
    
    def _attempt_example(
        self,
        example: LearningExample,
        previous_examples: List[LearningExample]
    ) -> PerformanceRecord:
        """Attempt to solve an example"""
        
        start_time = time.time()
        
        # Build context from previous examples
        context = ""
        if previous_examples:
            context = "Previous examples:\n" + "\n".join([
                f"Input: {ex.input_text}\nOutput: {ex.expected_output}"
                for ex in previous_examples[-3:]  # Last 3 examples
            ]) + "\n\n"
        
        # Generate response
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are learning to solve tasks step by step.

{context}Now solve this task:"""),
            ("user", "{input}")
        ])
        
        chain = prompt | self.learner | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "input": example.input_text
        })
        
        time_taken_ms = (time.time() - start_time) * 1000
        
        # Check correctness (simplified)
        response_normalized = response.strip().lower()
        expected_normalized = example.expected_output.strip().lower()
        
        # Flexible matching
        correct = (
            response_normalized == expected_normalized or
            expected_normalized in response_normalized or
            response_normalized in expected_normalized
        )
        
        # Count attempts
        attempt_number = len([
            r for r in self.performance_history
            if r.example_id == example.example_id
        ]) + 1
        
        record = PerformanceRecord(
            example_id=example.example_id,
            attempt_number=attempt_number,
            correct=correct,
            response=response,
            timestamp=datetime.now(),
            time_taken_ms=time_taken_ms
        )
        
        self.performance_history.append(record)
        
        return record
    
    def _calculate_stage_performance(
        self,
        stage: CurriculumStage
    ) -> float:
        """Calculate performance on current stage"""
        
        stage_example_ids = {ex.example_id for ex in stage.examples}
        
        recent_records = [
            r for r in self.performance_history[-20:]  # Last 20 attempts
            if r.example_id in stage_example_ids
        ]
        
        if not recent_records:
            return 0.0
        
        correct_count = sum(1 for r in recent_records if r.correct)
        return correct_count / len(recent_records)
    
    def _should_advance(
        self,
        stage: CurriculumStage,
        attempts_on_stage: int
    ) -> bool:
        """Determine if should advance to next stage"""
        
        if self.pacing_strategy == PacingStrategy.FIXED:
            # Fixed progression after N attempts
            return attempts_on_stage >= len(stage.examples) * 2
        
        elif self.pacing_strategy == PacingStrategy.ADAPTIVE:
            # Advance when performance exceeds threshold
            performance = self._calculate_stage_performance(stage)
            return performance >= stage.mastery_threshold and attempts_on_stage >= len(stage.examples)
        
        elif self.pacing_strategy == PacingStrategy.COMPETENCY_BASED:
            # Advance when all examples mastered
            stage_example_ids = {ex.example_id for ex in stage.examples}
            mastered_in_stage = stage_example_ids & self.mastered_examples
            return len(mastered_in_stage) >= len(stage_example_ids) * stage.mastery_threshold
        
        else:
            # Default: adaptive
            performance = self._calculate_stage_performance(stage)
            return performance >= stage.mastery_threshold
    
    def learn_curriculum(
        self,
        examples: List[LearningExample],
        max_attempts_per_stage: int = 20
    ) -> CurriculumResult:
        """Learn through curriculum progression"""
        
        start_time = time.time()
        
        print(f"\n📚 Starting curriculum learning with {len(examples)} examples...")
        
        # Create curriculum
        stages = self._create_curriculum(examples)
        
        print(f"\n🎯 Curriculum Structure:")
        for stage in stages:
            print(f"   {stage.stage_name}: {stage.num_examples} examples")
        
        # Learn through stages
        time_per_stage = []
        
        for stage_idx, stage in enumerate(stages):
            stage_start = time.time()
            
            print(f"\n📖 Stage {stage_idx + 1}/{len(stages)}: {stage.stage_name}")
            print(f"   Examples: {stage.num_examples}")
            print(f"   Mastery Threshold: {stage.mastery_threshold:.0%}")
            
            attempts_on_stage = 0
            previous_examples = []
            
            # Learn examples in this stage
            while attempts_on_stage < max_attempts_per_stage:
                # Select next example
                # Prioritize non-mastered examples
                available_examples = [
                    ex for ex in stage.examples
                    if ex.example_id not in self.mastered_examples
                    and ex.is_prerequisite_met(self.mastered_examples)
                ]
                
                if not available_examples:
                    available_examples = stage.examples
                
                example = random.choice(available_examples)
                
                # Attempt example
                record = self._attempt_example(example, previous_examples)
                attempts_on_stage += 1
                
                if record.correct:
                    self.mastered_examples.add(example.example_id)
                    previous_examples.append(example)
                    print(f"      ✓ Mastered: {example.example_id}")
                else:
                    print(f"      ✗ Incorrect: {example.example_id}")
                
                # Update learning curve
                recent_accuracy = self._calculate_recent_accuracy()
                self.learning_curve.append((len(self.performance_history), recent_accuracy))
                
                # Check if should advance
                if self._should_advance(stage, attempts_on_stage):
                    print(f"\n   ✅ Stage mastered! Moving to next stage...")
                    break
            
            stage_time = time.time() - stage_start
            time_per_stage.append(stage_time)
            
            # Stage summary
            stage_perf = self._calculate_stage_performance(stage)
            print(f"   Stage Performance: {stage_perf:.1%}")
            print(f"   Time: {stage_time:.1f}s")
        
        # Calculate final metrics
        total_time_ms = (time.time() - start_time) * 1000
        overall_accuracy = self._calculate_recent_accuracy(window=len(self.performance_history))
        final_performance = self._calculate_recent_accuracy(window=10)
        
        return CurriculumResult(
            stages_completed=len(stages),
            total_examples=len(examples),
            examples_mastered=len(self.mastered_examples),
            overall_accuracy=overall_accuracy,
            learning_curve=self.learning_curve,
            time_per_stage=time_per_stage,
            final_performance=final_performance,
            total_time_ms=total_time_ms
        )
    
    def _calculate_recent_accuracy(self, window: int = 10) -> float:
        """Calculate accuracy over recent attempts"""
        
        if not self.performance_history:
            return 0.0
        
        recent = self.performance_history[-window:]
        correct = sum(1 for r in recent if r.correct)
        return correct / len(recent)


def demonstrate_curriculum_learning():
    """Demonstrate Curriculum Learning pattern"""
    
    print("=" * 80)
    print("PATTERN 060: CURRICULUM LEARNING DEMONSTRATION")
    print("=" * 80)
    print("\nProgressive learning from simple to complex\n")
    
    # Create learning examples with varying difficulty
    examples = [
        # Trivial examples
        LearningExample("ex1", "math", "What is 2 + 2?", "4", DifficultyLevel.TRIVIAL),
        LearningExample("ex2", "math", "What is 5 + 3?", "8", DifficultyLevel.TRIVIAL),
        LearningExample("ex3", "math", "What is 10 - 6?", "4", DifficultyLevel.TRIVIAL),
        
        # Easy examples
        LearningExample("ex4", "math", "What is 12 + 15?", "27", DifficultyLevel.EASY, prerequisites=["ex1"]),
        LearningExample("ex5", "math", "What is 25 - 8?", "17", DifficultyLevel.EASY, prerequisites=["ex3"]),
        LearningExample("ex6", "math", "What is 7 times 3?", "21", DifficultyLevel.EASY),
        
        # Medium examples
        LearningExample("ex7", "math", "What is 15 times 4?", "60", DifficultyLevel.MEDIUM, prerequisites=["ex6"]),
        LearningExample("ex8", "math", "What is 100 divided by 4?", "25", DifficultyLevel.MEDIUM),
        LearningExample("ex9", "math", "What is (5 + 3) times 2?", "16", DifficultyLevel.MEDIUM, prerequisites=["ex4", "ex6"]),
        
        # Hard examples
        LearningExample("ex10", "math", "What is 25% of 80?", "20", DifficultyLevel.HARD, prerequisites=["ex8"]),
        LearningExample("ex11", "math", "If x + 5 = 12, what is x?", "7", DifficultyLevel.HARD, prerequisites=["ex4"]),
    ]
    
    # Test 1: Adaptive curriculum
    print("\n" + "=" * 80)
    print("TEST 1: Adaptive Curriculum Learning")
    print("=" * 80)
    
    learner1 = CurriculumLearner(
        pacing_strategy=PacingStrategy.ADAPTIVE,
        mastery_threshold=0.7
    )
    
    result1 = learner1.learn_curriculum(examples, max_attempts_per_stage=15)
    
    print(f"\n📊 Learning Results:")
    print(f"   Stages Completed: {result1.stages_completed}")
    print(f"   Examples Mastered: {result1.examples_mastered}/{result1.total_examples}")
    print(f"   Overall Accuracy: {result1.overall_accuracy:.1%}")
    print(f"   Final Performance: {result1.final_performance:.1%}")
    print(f"   Total Time: {result1.total_time_ms/1000:.1f}s")
    
    print(f"\n📈 Learning Curve (samples):")
    for i in range(0, len(result1.learning_curve), max(1, len(result1.learning_curve)//5)):
        attempt_num, accuracy = result1.learning_curve[i]
        print(f"   Attempt {attempt_num}: {accuracy:.1%} accuracy")
    
    print(f"\n⏱️  Time per Stage:")
    for i, stage_time in enumerate(result1.time_per_stage, 1):
        print(f"   Stage {i}: {stage_time:.1f}s")
    
    # Test 2: Fixed pacing
    print("\n" + "=" * 80)
    print("TEST 2: Fixed Pacing Curriculum")
    print("=" * 80)
    
    learner2 = CurriculumLearner(
        pacing_strategy=PacingStrategy.FIXED,
        mastery_threshold=0.6
    )
    
    # Use subset for faster demo
    subset_examples = examples[:8]
    
    result2 = learner2.learn_curriculum(subset_examples, max_attempts_per_stage=10)
    
    print(f"\n📊 Fixed Pacing Results:")
    print(f"   Stages: {result2.stages_completed}")
    print(f"   Mastered: {result2.examples_mastered}/{result2.total_examples}")
    print(f"   Accuracy: {result2.overall_accuracy:.1%}")
    
    # Test 3: Learning curve comparison
    print("\n" + "=" * 80)
    print("TEST 3: Learning Progress Visualization")
    print("=" * 80)
    
    print(f"\n📈 Performance Improvement:")
    curve = result1.learning_curve
    
    # Show improvement
    if len(curve) >= 3:
        early_perf = statistics.mean([acc for _, acc in curve[:3]])
        mid_perf = statistics.mean([acc for _, acc in curve[len(curve)//2:len(curve)//2+3]])
        late_perf = statistics.mean([acc for _, acc in curve[-3:]])
        
        print(f"   Early (first 3): {early_perf:.1%}")
        print(f"   Middle: {mid_perf:.1%}")
        print(f"   Late (last 3): {late_perf:.1%}")
        print(f"   Improvement: +{(late_perf - early_perf):.1%}")
    
    # Summary
    print("\n" + "=" * 80)
    print("CURRICULUM LEARNING PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Faster Learning: Build on foundations progressively
2. Better Performance: Master basics before advanced topics
3. Reduced Frustration: Appropriate difficulty at each stage
4. Stable Training: Avoid overwhelming with complexity
5. Improved Retention: Scaffold knowledge systematically

Curriculum Design:
1. Difficulty Ordering: Simple → Complex
2. Prerequisite Structure: Build dependencies
3. Concept Progression: Foundational → Advanced
4. Skill Scaffolding: Support skill development
5. Spiral Learning: Revisit concepts with depth

Difficulty Assessment:
- Automatic: LLM-based complexity scoring
- Manual: Expert labeling
- Hybrid: Combine both approaches
- Dynamic: Adjust based on learner performance
- Multi-dimensional: Consider multiple factors

Pacing Strategies:
1. Fixed: Predetermined progression
   - Simple, predictable
   - May not fit all learners

2. Adaptive: Performance-based
   - Personalized pace
   - More complex logic

3. Mixed: Interleave difficulties
   - Maintain engagement
   - Prevent boredom

4. Spiral: Revisit with complexity
   - Reinforce learning
   - Deepen understanding

5. Competency-Based: Master before advance
   - Ensure readiness
   - May be slower

Curriculum Stages:
1. Foundation: Basic concepts
2. Development: Core skills
3. Integration: Combine concepts
4. Application: Real problems
5. Mastery: Advanced challenges

Performance Monitoring:
- Accuracy tracking
- Time per example
- Mastery identification
- Learning curve analysis
- Plateau detection

Advancement Criteria:
- Accuracy threshold (e.g., 80%)
- Minimum attempts per stage
- Mastery of N examples
- Time-based progression
- Adaptive combinations

Use Cases:
- Educational tutoring systems
- Agent skill development
- Reinforcement learning
- Language learning
- Technical training
- Complex task acquisition

Best Practices:
1. Clear difficulty progression
2. Prerequisite tracking
3. Performance monitoring
4. Adaptive pacing
5. Regular assessment
6. Sufficient practice
7. Gradual complexity increase

Production Considerations:
- Curriculum versioning
- Learner state persistence
- Performance analytics
- A/B testing curricula
- Difficulty calibration
- Prerequisite validation
- Scalability

Comparison with Related Patterns:
- vs. Random: Structured vs unordered
- vs. Hard Example Mining: Systematic vs selection
- vs. Meta-Learning: Progression vs strategy learning
- vs. Active Learning: Curriculum vs query strategy

Challenges:
1. Difficulty assessment accuracy
2. Optimal ordering
3. Individual differences
4. Curriculum generalization
5. Pacing calibration

Research Findings:
- 30-50% faster learning vs random
- Better final performance
- Improved generalization
- Reduced training instability
- Context-dependent effectiveness

The Curriculum Learning pattern enables efficient learning through
strategic ordering of examples from simple to complex, mirroring
effective human educational approaches.
""")


if __name__ == "__main__":
    demonstrate_curriculum_learning()
