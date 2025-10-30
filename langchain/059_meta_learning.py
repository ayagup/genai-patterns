"""
Pattern 059: Meta-Learning Agent

Description:
    Meta-learning (learning to learn) enables agents to improve their learning
    efficiency over time by learning general strategies that transfer across tasks.
    Rather than learning task-specific solutions, meta-learning agents learn how
    to adapt quickly to new tasks with minimal examples or experience.

Components:
    1. Task Distribution: Collection of related learning tasks
    2. Meta-Learner: Learns general learning strategies
    3. Base Learner: Applies strategies to specific tasks
    4. Adaptation Mechanism: Quick task-specific tuning
    5. Transfer Module: Knowledge transfer across tasks

Use Cases:
    - Few-shot learning scenarios
    - Rapid adaptation to new domains
    - Personalization with limited data
    - Multi-task learning systems
    - Continuous learning environments
    - Cross-domain knowledge transfer

LangChain Implementation:
    Implements meta-learning through prompt optimization, example selection,
    and strategy learning that transfers across tasks.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import statistics

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class MetaStrategy(Enum):
    """Meta-learning strategies"""
    MAML = "maml"  # Model-Agnostic Meta-Learning
    PROMPT_LEARNING = "prompt_learning"  # Learn optimal prompts
    EXAMPLE_SELECTION = "example_selection"  # Learn to select examples
    STRATEGY_LEARNING = "strategy_learning"  # Learn problem-solving strategies
    TRANSFER_LEARNING = "transfer_learning"  # Transfer knowledge across tasks


class TaskType(Enum):
    """Types of learning tasks"""
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    REASONING = "reasoning"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"


@dataclass
class Task:
    """A learning task"""
    task_id: str
    task_type: TaskType
    description: str
    examples: List[Tuple[str, str]]  # (input, output)
    test_cases: List[Tuple[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearnedStrategy:
    """A learned problem-solving strategy"""
    strategy_id: str
    applicable_tasks: List[TaskType]
    prompt_template: str
    example_count: int
    success_rate: float
    usage_count: int = 0
    
    def is_applicable(self, task_type: TaskType) -> bool:
        return task_type in self.applicable_tasks


@dataclass
class AdaptationResult:
    """Result from adaptation to new task"""
    task_id: str
    strategy_used: str
    examples_used: int
    performance: float
    adaptation_time_ms: float
    predictions: List[str]


@dataclass
class MetaLearningResult:
    """Result from meta-learning process"""
    tasks_trained: int
    strategies_learned: List[LearnedStrategy]
    avg_adaptation_performance: float
    total_training_time_ms: float
    transfer_effectiveness: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks_trained": self.tasks_trained,
            "strategies_learned": len(self.strategies_learned),
            "avg_performance": f"{self.avg_adaptation_performance:.2%}",
            "transfer_effectiveness": f"{self.transfer_effectiveness:.2%}",
            "training_time_ms": f"{self.total_training_time_ms:.1f}"
        }


class MetaLearningAgent:
    """
    Agent that learns to learn across tasks.
    
    Features:
    1. Strategy learning from multiple tasks
    2. Fast adaptation to new tasks
    3. Optimal example selection
    4. Prompt optimization
    5. Knowledge transfer
    """
    
    def __init__(
        self,
        meta_strategy: MetaStrategy = MetaStrategy.STRATEGY_LEARNING,
        temperature: float = 0.7
    ):
        self.meta_strategy = meta_strategy
        self.temperature = temperature
        
        # Meta-learner (learns strategies)
        self.meta_learner = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # Base learner (applies strategies)
        self.base_learner = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=temperature
        )
        
        # Learned strategies
        self.strategies: List[LearnedStrategy] = []
        
        # Task performance history
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        
        # Experience buffer
        self.experience: List[Tuple[Task, float]] = []
    
    def _learn_strategy_from_tasks(
        self,
        tasks: List[Task]
    ) -> LearnedStrategy:
        """Learn a general strategy from multiple tasks"""
        
        # Analyze patterns across tasks
        task_descriptions = "\n".join([
            f"Task {i+1} ({task.task_type.value}): {task.description}\n"
            f"Example: {task.examples[0][0]} → {task.examples[0][1]}"
            for i, task in enumerate(tasks[:3])
        ])
        
        strategy_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze these learning tasks and identify a general strategy
that would work across all of them.

Provide:
1. Strategy name
2. General approach
3. Key steps
4. When to use

Be concise but specific."""),
            ("user", "Tasks:\n{tasks}\n\nGeneral strategy:")
        ])
        
        chain = strategy_prompt | self.meta_learner | StrOutputParser()
        strategy_description = chain.invoke({"tasks": task_descriptions})
        
        # Create prompt template from strategy
        prompt_template = f"""Strategy: {strategy_description}

Examples:
{{examples}}

Now apply this strategy to: {{query}}

Response:"""
        
        # Determine applicable task types
        task_types = list(set(t.task_type for t in tasks))
        
        strategy = LearnedStrategy(
            strategy_id=f"strategy_{len(self.strategies)}",
            applicable_tasks=task_types,
            prompt_template=prompt_template,
            example_count=3,
            success_rate=0.0  # Will be updated with experience
        )
        
        return strategy
    
    def _select_examples(
        self,
        task: Task,
        k: int = 3
    ) -> List[Tuple[str, str]]:
        """Select most informative examples for task"""
        
        # Simple selection: use first k examples
        # In production: use diversity, difficulty, relevance
        return task.examples[:k]
    
    def _evaluate_performance(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> float:
        """Evaluate prediction performance"""
        
        if not predictions or not ground_truth:
            return 0.0
        
        # Simple accuracy (in production: use task-specific metrics)
        correct = sum(
            1 for pred, truth in zip(predictions, ground_truth)
            if pred.strip().lower() == truth.strip().lower()
        )
        
        return correct / len(ground_truth)
    
    def meta_train(
        self,
        tasks: List[Task],
        episodes_per_task: int = 2
    ) -> MetaLearningResult:
        """Meta-training: Learn strategies from multiple tasks"""
        
        start_time = time.time()
        
        print(f"\n🎓 Meta-Training on {len(tasks)} tasks...")
        
        # Group tasks by type
        tasks_by_type = defaultdict(list)
        for task in tasks:
            tasks_by_type[task.task_type].append(task)
        
        # Learn strategies for each task type
        for task_type, type_tasks in tasks_by_type.items():
            if len(type_tasks) >= 2:  # Need multiple tasks to learn strategy
                print(f"\n  Learning strategy for {task_type.value} tasks...")
                
                strategy = self._learn_strategy_from_tasks(type_tasks)
                
                # Evaluate strategy on held-out examples
                performances = []
                for task in type_tasks:
                    # Use strategy to predict on test cases
                    examples = self._select_examples(task, k=2)
                    examples_text = "\n".join([
                        f"Input: {inp}\nOutput: {out}"
                        for inp, out in examples
                    ])
                    
                    predictions = []
                    for test_input, _ in task.test_cases[:3]:
                        prompt = strategy.prompt_template.format(
                            examples=examples_text,
                            query=test_input
                        )
                        
                        pred = self.base_learner.invoke(prompt).content
                        predictions.append(pred)
                    
                    # Evaluate
                    ground_truth = [out for _, out in task.test_cases[:3]]
                    perf = self._evaluate_performance(predictions, ground_truth)
                    performances.append(perf)
                    
                    # Store experience
                    self.experience.append((task, perf))
                
                # Update strategy success rate
                strategy.success_rate = statistics.mean(performances)
                self.strategies.append(strategy)
                
                print(f"    Strategy learned with {strategy.success_rate:.1%} success rate")
        
        # Calculate metrics
        all_performances = [perf for _, perf in self.experience]
        avg_performance = statistics.mean(all_performances) if all_performances else 0.0
        
        # Transfer effectiveness: improvement over baseline
        # (simplified - in production, compare to zero-shot)
        transfer_effectiveness = min(avg_performance * 1.2, 1.0)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        return MetaLearningResult(
            tasks_trained=len(tasks),
            strategies_learned=self.strategies.copy(),
            avg_adaptation_performance=avg_performance,
            total_training_time_ms=total_time_ms,
            transfer_effectiveness=transfer_effectiveness
        )
    
    def adapt(
        self,
        new_task: Task,
        support_examples: Optional[List[Tuple[str, str]]] = None
    ) -> AdaptationResult:
        """Adapt to new task using learned strategies"""
        
        start_time = time.time()
        
        # Select best strategy for this task type
        applicable_strategies = [
            s for s in self.strategies
            if s.is_applicable(new_task.task_type)
        ]
        
        if applicable_strategies:
            # Use best performing strategy
            strategy = max(applicable_strategies, key=lambda s: s.success_rate)
            strategy.usage_count += 1
        else:
            # Fallback: learn from task examples only
            strategy = LearnedStrategy(
                strategy_id="fallback",
                applicable_tasks=[new_task.task_type],
                prompt_template="Examples:\n{examples}\n\nQuery: {query}\nResponse:",
                example_count=3,
                success_rate=0.5
            )
        
        # Select examples
        if support_examples:
            examples = support_examples
        else:
            examples = self._select_examples(new_task, k=strategy.example_count)
        
        examples_text = "\n".join([
            f"Input: {inp}\nOutput: {out}"
            for inp, out in examples
        ])
        
        # Make predictions on test cases
        predictions = []
        for test_input, _ in new_task.test_cases:
            prompt = strategy.prompt_template.format(
                examples=examples_text,
                query=test_input
            )
            
            pred = self.base_learner.invoke(prompt).content
            predictions.append(pred)
        
        # Evaluate performance
        ground_truth = [out for _, out in new_task.test_cases]
        performance = self._evaluate_performance(predictions, ground_truth)
        
        # Update strategy success rate
        self.performance_history[strategy.strategy_id].append(performance)
        if self.performance_history[strategy.strategy_id]:
            strategy.success_rate = statistics.mean(
                self.performance_history[strategy.strategy_id]
            )
        
        adaptation_time_ms = (time.time() - start_time) * 1000
        
        return AdaptationResult(
            task_id=new_task.task_id,
            strategy_used=strategy.strategy_id,
            examples_used=len(examples),
            performance=performance,
            adaptation_time_ms=adaptation_time_ms,
            predictions=predictions
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get meta-learning statistics"""
        
        return {
            "strategies_learned": len(self.strategies),
            "total_experience": len(self.experience),
            "strategy_details": [
                {
                    "id": s.strategy_id,
                    "task_types": [t.value for t in s.applicable_tasks],
                    "success_rate": f"{s.success_rate:.1%}",
                    "usage_count": s.usage_count
                }
                for s in self.strategies
            ]
        }


def demonstrate_meta_learning():
    """Demonstrate Meta-Learning Agent pattern"""
    
    print("=" * 80)
    print("PATTERN 059: META-LEARNING AGENT DEMONSTRATION")
    print("=" * 80)
    print("\nLearning to learn across tasks\n")
    
    # Create meta-learning agent
    agent = MetaLearningAgent(
        meta_strategy=MetaStrategy.STRATEGY_LEARNING,
        temperature=0.7
    )
    
    # Create training tasks
    training_tasks = [
        Task(
            task_id="sentiment_1",
            task_type=TaskType.CLASSIFICATION,
            description="Classify sentiment as positive or negative",
            examples=[
                ("This movie was amazing!", "positive"),
                ("I hated this book.", "negative"),
                ("Great product, highly recommend.", "positive"),
            ],
            test_cases=[
                ("Loved it!", "positive"),
                ("Terrible experience.", "negative"),
            ]
        ),
        Task(
            task_id="sentiment_2",
            task_type=TaskType.CLASSIFICATION,
            description="Determine if review is positive or negative",
            examples=[
                ("Best purchase ever!", "positive"),
                ("Waste of money.", "negative"),
                ("Exceeded expectations.", "positive"),
            ],
            test_cases=[
                ("Fantastic!", "positive"),
                ("Very disappointed.", "negative"),
            ]
        ),
        Task(
            task_id="translate_1",
            task_type=TaskType.TRANSLATION,
            description="Translate English to simple terms",
            examples=[
                ("utilize", "use"),
                ("commence", "begin"),
                ("terminate", "end"),
            ],
            test_cases=[
                ("facilitate", "help"),
                ("acquire", "get"),
            ]
        ),
        Task(
            task_id="translate_2",
            task_type=TaskType.TRANSLATION,
            description="Convert formal to informal",
            examples=[
                ("I am pleased to inform you", "I'm happy to tell you"),
                ("Please be advised", "Just so you know"),
                ("At your earliest convenience", "When you can"),
            ],
            test_cases=[
                ("It is recommended", "You should"),
                ("We regret to inform", "Sorry to say"),
            ]
        ),
    ]
    
    # Test 1: Meta-training
    print("\n" + "=" * 80)
    print("TEST 1: Meta-Training on Multiple Tasks")
    print("=" * 80)
    
    print(f"\n📚 Training Tasks:")
    for task in training_tasks:
        print(f"   - {task.task_id} ({task.task_type.value}): {task.description}")
    
    meta_result = agent.meta_train(training_tasks, episodes_per_task=1)
    
    print(f"\n📊 Meta-Training Results:")
    print(f"   Tasks Trained: {meta_result.tasks_trained}")
    print(f"   Strategies Learned: {len(meta_result.strategies_learned)}")
    print(f"   Avg Performance: {meta_result.avg_adaptation_performance:.1%}")
    print(f"   Transfer Effectiveness: {meta_result.transfer_effectiveness:.1%}")
    print(f"   Training Time: {meta_result.total_training_time_ms:.1f}ms")
    
    print(f"\n🧠 Learned Strategies:")
    for strategy in meta_result.strategies_learned:
        print(f"   {strategy.strategy_id}:")
        print(f"      Task Types: {[t.value for t in strategy.applicable_tasks]}")
        print(f"      Success Rate: {strategy.success_rate:.1%}")
    
    # Test 2: Adaptation to new task (few-shot)
    print("\n" + "=" * 80)
    print("TEST 2: Fast Adaptation to New Task (Few-Shot)")
    print("=" * 80)
    
    new_task = Task(
        task_id="sentiment_new",
        task_type=TaskType.CLASSIFICATION,
        description="Classify restaurant reviews",
        examples=[
            ("Food was delicious!", "positive"),
            ("Service was terrible.", "negative"),
        ],
        test_cases=[
            ("Best meal I've had!", "positive"),
            ("Worst restaurant ever.", "negative"),
            ("Amazing ambiance.", "positive"),
        ]
    )
    
    print(f"\n🆕 New Task: {new_task.description}")
    print(f"   Task Type: {new_task.task_type.value}")
    print(f"   Training Examples: {len(new_task.examples)}")
    print(f"   Test Cases: {len(new_task.test_cases)}")
    
    adapt_result = agent.adapt(new_task)
    
    print(f"\n⚡ Adaptation Results:")
    print(f"   Strategy Used: {adapt_result.strategy_used}")
    print(f"   Examples Used: {adapt_result.examples_used}")
    print(f"   Performance: {adapt_result.performance:.1%}")
    print(f"   Adaptation Time: {adapt_result.adaptation_time_ms:.1f}ms")
    
    print(f"\n💭 Predictions:")
    for i, (test_input, _) in enumerate(new_task.test_cases):
        print(f"   Input: {test_input}")
        print(f"   Predicted: {adapt_result.predictions[i]}")
    
    # Test 3: Transfer across task types
    print("\n" + "=" * 80)
    print("TEST 3: Transfer Learning to Different Task Type")
    print("=" * 80)
    
    different_task = Task(
        task_id="translate_new",
        task_type=TaskType.TRANSLATION,
        description="Simplify technical jargon",
        examples=[
            ("optimize", "improve"),
            ("implement", "do"),
        ],
        test_cases=[
            ("leverage", "use"),
            ("paradigm", "model"),
        ]
    )
    
    print(f"\n🔄 Transfer Task: {different_task.description}")
    print(f"   Task Type: {different_task.task_type.value}")
    
    transfer_result = agent.adapt(different_task)
    
    print(f"\n📊 Transfer Results:")
    print(f"   Strategy: {transfer_result.strategy_used}")
    print(f"   Performance: {transfer_result.performance:.1%}")
    print(f"   Adaptation Time: {transfer_result.adaptation_time_ms:.1f}ms")
    
    # Show statistics
    print(f"\n📈 Agent Statistics:")
    stats = agent.get_statistics()
    print(f"   Total Strategies: {stats['strategies_learned']}")
    print(f"   Total Experience: {stats['total_experience']}")
    
    print(f"\n   Strategy Usage:")
    for strategy_info in stats['strategy_details']:
        if strategy_info['usage_count'] > 0:
            print(f"      {strategy_info['id']}: {strategy_info['usage_count']} times "
                  f"({strategy_info['success_rate']} success)")
    
    # Summary
    print("\n" + "=" * 80)
    print("META-LEARNING AGENT PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Fast Adaptation: Learn from few examples
2. Transfer Learning: Knowledge across tasks
3. Efficiency: Less data needed per task
4. Generalization: Learn general strategies
5. Continuous Improvement: Better over time

Meta-Learning Strategies:
1. MAML: Model-agnostic meta-learning
   - Learn initialization that adapts quickly
   
2. Prompt Learning: Optimal prompt discovery
   - Learn effective prompt templates
   
3. Example Selection: Choose informative examples
   - Learn which examples help most
   
4. Strategy Learning: General approaches
   - Learn problem-solving strategies
   
5. Transfer Learning: Cross-task knowledge
   - Transfer learned representations

Meta-Training Process:
1. Sample task distribution
2. For each task:
   - Support set: few training examples
   - Query set: test examples
3. Learn strategy from patterns
4. Evaluate on held-out tasks
5. Update meta-parameters

Adaptation Process:
1. Receive new task
2. Select applicable strategy
3. Choose informative examples
4. Apply learned approach
5. Make predictions
6. Update based on feedback

Components:
- Meta-Learner: Learns strategies
- Base Learner: Applies to tasks
- Strategy Library: Learned approaches
- Experience Buffer: Historical performance
- Transfer Module: Cross-task knowledge

Use Cases:
- Few-shot learning
- Personalization (limited user data)
- Multi-domain systems
- Rapid prototyping
- Cross-lingual transfer
- Continuous learning

Key Concepts:
1. Support Set: Training examples
2. Query Set: Test examples
3. Inner Loop: Task-specific learning
4. Outer Loop: Meta-learning update
5. Task Distribution: Set of related tasks

Best Practices:
1. Diverse training tasks
2. Appropriate support set size
3. Strategy evaluation
4. Performance monitoring
5. Adaptive example selection
6. Regular meta-updates
7. Transfer validation

Production Considerations:
- Strategy library management
- Performance tracking per strategy
- Task similarity metrics
- Example caching
- Incremental meta-learning
- Cold start handling
- Computational efficiency

Comparison with Related Patterns:
- vs. Transfer Learning: Meta vs direct transfer
- vs. Few-Shot: General framework vs specific technique
- vs. Fine-tuning: Strategy learning vs parameter tuning
- vs. Ensemble: Meta-learning vs combination

Challenges:
1. Task distribution design
2. Strategy generalization
3. Computational cost
4. Overfitting to task distribution
5. Evaluation methodology

The Meta-Learning Agent pattern enables rapid adaptation to new
tasks with minimal examples by learning general strategies that
transfer effectively across related problems.
""")


if __name__ == "__main__":
    demonstrate_meta_learning()
