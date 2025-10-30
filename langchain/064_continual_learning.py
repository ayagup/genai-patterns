"""
Pattern 064: Continual Learning

Description:
    Continual Learning (also called Lifelong Learning) enables agents to learn
    continuously from new experiences without forgetting previously acquired
    knowledge. This addresses the "catastrophic forgetting" problem where neural
    networks forget old tasks when learning new ones.

Components:
    1. Experience Buffer: Stores past experiences
    2. Consolidation: Strengthens important memories
    3. Regularization: Prevents overwriting old knowledge
    4. Knowledge Integration: Merges new with existing knowledge
    5. Forgetting Prevention: Various strategies to retain knowledge
    6. Performance Monitoring: Tracks across all tasks

Use Cases:
    - Long-running personal assistants
    - Adaptive recommendation systems
    - Evolving domain knowledge
    - Multi-task learning scenarios
    - Personalization over time
    - Dynamic environment adaptation

LangChain Implementation:
    Implements continual learning using experience replay, knowledge consolidation,
    and regularization techniques to prevent catastrophic forgetting.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random
from collections import deque

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ForgettingStrategy(Enum):
    """Strategies to prevent catastrophic forgetting"""
    EXPERIENCE_REPLAY = "experience_replay"  # Replay old experiences
    REGULARIZATION = "regularization"  # Constrain changes
    PROGRESSIVE_NETWORKS = "progressive_networks"  # Add new capacity
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"  # Transfer knowledge
    CONSOLIDATION = "consolidation"  # Strengthen important knowledge


class TaskPriority(Enum):
    """Priority levels for tasks"""
    CRITICAL = "critical"  # Must never forget
    HIGH = "high"  # Important
    MEDIUM = "medium"  # Moderate importance
    LOW = "low"  # Can forget if needed


@dataclass
class Task:
    """Learning task"""
    task_id: str
    name: str
    description: str
    domain: str
    examples: List[Tuple[str, str]] = field(default_factory=list)  # (input, output) pairs
    priority: TaskPriority = TaskPriority.MEDIUM
    importance_score: float = 1.0
    
    def add_example(self, input_text: str, output_text: str):
        """Add training example"""
        self.examples.append((input_text, output_text))


@dataclass
class Experience:
    """Single learning experience"""
    experience_id: str
    task_id: str
    input_data: str
    output_data: str
    timestamp: float = field(default_factory=time.time)
    importance: float = 1.0
    rehearsal_count: int = 0


@dataclass
class PerformanceMetric:
    """Performance on a task"""
    task_id: str
    task_name: str
    accuracy: float
    timestamp: float
    num_samples: int


class ContinualLearner:
    """
    Continual learning agent.
    
    Features:
    1. Learn from new tasks continuously
    2. Prevent catastrophic forgetting
    3. Selective memory consolidation
    4. Experience replay
    5. Performance monitoring
    """
    
    def __init__(
        self,
        buffer_size: int = 100,
        forgetting_strategy: ForgettingStrategy = ForgettingStrategy.EXPERIENCE_REPLAY,
        consolidation_frequency: int = 10
    ):
        self.buffer_size = buffer_size
        self.forgetting_strategy = forgetting_strategy
        self.consolidation_frequency = consolidation_frequency
        
        # Experience buffer (ring buffer)
        self.experience_buffer: deque = deque(maxlen=buffer_size)
        
        # Task registry
        self.tasks: Dict[str, Task] = {}
        
        # Knowledge base (consolidated knowledge)
        self.knowledge_base: Dict[str, str] = {}
        
        # Performance history
        self.performance_history: List[PerformanceMetric] = []
        
        # Learner LLM
        self.learner = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # Consolidator LLM
        self.consolidator = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2
        )
        
        # Learning counter for consolidation
        self.learning_steps = 0
    
    def register_task(self, task: Task):
        """Register a new task"""
        self.tasks[task.task_id] = task
        print(f"   ✓ Registered task: {task.name} (priority: {task.priority.value})")
    
    def learn(
        self,
        task_id: str,
        input_data: str,
        output_data: str,
        importance: float = 1.0
    ):
        """Learn from a single example"""
        
        # Store experience
        experience = Experience(
            experience_id=f"exp_{len(self.experience_buffer)}",
            task_id=task_id,
            input_data=input_data,
            output_data=output_data,
            importance=importance
        )
        
        self.experience_buffer.append(experience)
        
        # Increment learning counter
        self.learning_steps += 1
        
        # Apply forgetting prevention strategy
        if self.forgetting_strategy == ForgettingStrategy.EXPERIENCE_REPLAY:
            self._experience_replay()
        
        # Consolidate periodically
        if self.learning_steps % self.consolidation_frequency == 0:
            self._consolidate_knowledge()
    
    def _experience_replay(self):
        """Replay past experiences to prevent forgetting"""
        
        if len(self.experience_buffer) < 5:
            return
        
        # Sample diverse experiences
        # Priority: high-importance, diverse tasks, recent
        
        # Get high-importance experiences
        high_importance = [
            exp for exp in self.experience_buffer
            if exp.importance > 0.7
        ]
        
        # Sample from different tasks
        task_samples = {}
        for exp in self.experience_buffer:
            if exp.task_id not in task_samples:
                task_samples[exp.task_id] = []
            task_samples[exp.task_id].append(exp)
        
        # Take one from each task
        replay_set = []
        for task_id, exps in task_samples.items():
            if exps:
                # Prefer important experiences
                exps_sorted = sorted(exps, key=lambda e: e.importance, reverse=True)
                replay_set.append(exps_sorted[0])
                exps_sorted[0].rehearsal_count += 1
        
        return replay_set
    
    def _consolidate_knowledge(self):
        """Consolidate experiences into knowledge base"""
        
        print(f"\n   🔄 Consolidating knowledge (step {self.learning_steps})...")
        
        if len(self.experience_buffer) < 3:
            return
        
        # Group experiences by task
        task_experiences = {}
        for exp in self.experience_buffer:
            if exp.task_id not in task_experiences:
                task_experiences[exp.task_id] = []
            task_experiences[exp.task_id].append(exp)
        
        # Consolidate each task
        for task_id, experiences in task_experiences.items():
            if task_id not in self.tasks:
                continue
            
            task = self.tasks[task_id]
            
            # Build summary of experiences
            exp_summary = f"Task: {task.name}\n\n"
            exp_summary += "Recent Examples:\n"
            for i, exp in enumerate(experiences[-5:], 1):  # Last 5
                exp_summary += f"{i}. Input: {exp.input_data[:100]}\n"
                exp_summary += f"   Output: {exp.output_data[:100]}\n"
            
            # Consolidate into general knowledge
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract general patterns and knowledge from these examples.

Create a concise knowledge summary that captures:
1. Key patterns and rules
2. Important exceptions
3. General principles

This will be used to remember the task without storing all examples."""),
                ("user", "{summary}")
            ])
            
            chain = prompt | self.consolidator | StrOutputParser()
            consolidated = chain.invoke({"summary": exp_summary})
            
            # Store in knowledge base
            self.knowledge_base[task_id] = consolidated
            print(f"      ✓ Consolidated {task.name}")
    
    def predict(
        self,
        task_id: str,
        input_data: str,
        use_consolidation: bool = True
    ) -> str:
        """Make prediction for a task"""
        
        if task_id not in self.tasks:
            return "Unknown task"
        
        task = self.tasks[task_id]
        
        # Build context
        context = f"Task: {task.name}\n"
        context += f"Description: {task.description}\n\n"
        
        # Add consolidated knowledge if available
        if use_consolidation and task_id in self.knowledge_base:
            context += f"Learned Knowledge:\n{self.knowledge_base[task_id]}\n\n"
        
        # Add relevant experiences
        relevant_experiences = [
            exp for exp in self.experience_buffer
            if exp.task_id == task_id
        ]
        
        if relevant_experiences:
            context += "Recent Examples:\n"
            for exp in relevant_experiences[-3:]:  # Last 3
                context += f"Input: {exp.input_data}\n"
                context += f"Output: {exp.output_data}\n\n"
        
        # Make prediction
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are making a prediction based on learned knowledge and examples.

{context}

Use the learned knowledge and examples to make an accurate prediction."""),
            ("user", "Input: {input}\n\nOutput:")
        ])
        
        chain = prompt | self.learner | StrOutputParser()
        prediction = chain.invoke({
            "context": context,
            "input": input_data
        })
        
        return prediction
    
    def evaluate(
        self,
        task_id: str,
        test_cases: List[Tuple[str, str]]
    ) -> PerformanceMetric:
        """Evaluate performance on a task"""
        
        if task_id not in self.tasks:
            return PerformanceMetric(task_id, "Unknown", 0.0, time.time(), 0)
        
        task = self.tasks[task_id]
        
        correct = 0
        total = len(test_cases)
        
        for input_data, expected_output in test_cases:
            prediction = self.predict(task_id, input_data)
            
            # Simple match (in practice, would use better comparison)
            if expected_output.lower() in prediction.lower():
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        metric = PerformanceMetric(
            task_id=task_id,
            task_name=task.name,
            accuracy=accuracy,
            timestamp=time.time(),
            num_samples=total
        )
        
        self.performance_history.append(metric)
        
        return metric
    
    def get_forgetting_metrics(self) -> Dict[str, Any]:
        """Calculate forgetting metrics"""
        
        if len(self.performance_history) < 2:
            return {"status": "insufficient data"}
        
        # Group by task
        task_performance = {}
        for metric in self.performance_history:
            if metric.task_id not in task_performance:
                task_performance[metric.task_id] = []
            task_performance[metric.task_id].append(metric)
        
        # Calculate forgetting
        forgetting_scores = {}
        for task_id, metrics in task_performance.items():
            if len(metrics) < 2:
                continue
            
            # Compare first and last
            initial_acc = metrics[0].accuracy
            latest_acc = metrics[-1].accuracy
            
            forgetting = initial_acc - latest_acc
            forgetting_scores[task_id] = {
                "task": metrics[0].task_name,
                "initial_accuracy": initial_acc,
                "latest_accuracy": latest_acc,
                "forgetting": forgetting,
                "status": "retained" if forgetting <= 0.1 else "forgotten"
            }
        
        return forgetting_scores


def demonstrate_continual_learning():
    """Demonstrate Continual Learning pattern"""
    
    print("=" * 80)
    print("PATTERN 064: CONTINUAL LEARNING DEMONSTRATION")
    print("=" * 80)
    print("\nLearning continuously without catastrophic forgetting\n")
    
    # Test 1: Sequential task learning
    print("\n" + "=" * 80)
    print("TEST 1: Learning Multiple Tasks Sequentially")
    print("=" * 80)
    
    learner = ContinualLearner(
        buffer_size=50,
        forgetting_strategy=ForgettingStrategy.EXPERIENCE_REPLAY,
        consolidation_frequency=5
    )
    
    # Task 1: Sentiment analysis
    task1 = Task(
        task_id="sentiment",
        name="Sentiment Analysis",
        description="Classify text sentiment as positive or negative",
        domain="NLP",
        priority=TaskPriority.HIGH
    )
    learner.register_task(task1)
    
    # Task 2: Language translation
    task2 = Task(
        task_id="translation",
        name="Language Translation",
        description="Translate English to Spanish",
        domain="NLP",
        priority=TaskPriority.MEDIUM
    )
    learner.register_task(task2)
    
    # Task 3: Summarization
    task3 = Task(
        task_id="summarization",
        name="Text Summarization",
        description="Create brief summaries of text",
        domain="NLP",
        priority=TaskPriority.HIGH
    )
    learner.register_task(task3)
    
    print("\n📚 Learning Task 1: Sentiment Analysis")
    sentiment_examples = [
        ("This product is amazing!", "positive"),
        ("Terrible experience, very disappointed", "negative"),
        ("Love it! Highly recommend", "positive"),
    ]
    
    for input_text, output_text in sentiment_examples:
        learner.learn("sentiment", input_text, output_text, importance=0.8)
    
    # Evaluate Task 1
    sentiment_test = [
        ("Great quality and fast shipping", "positive"),
        ("Not worth the money", "negative")
    ]
    
    perf1 = learner.evaluate("sentiment", sentiment_test)
    print(f"   Initial Performance: {perf1.accuracy:.1%}")
    
    print("\n📚 Learning Task 2: Translation")
    translation_examples = [
        ("Hello", "Hola"),
        ("Good morning", "Buenos días"),
        ("Thank you", "Gracias"),
    ]
    
    for input_text, output_text in translation_examples:
        learner.learn("translation", input_text, output_text, importance=0.6)
    
    # Re-evaluate Task 1
    perf1_after = learner.evaluate("sentiment", sentiment_test)
    print(f"   Task 1 after learning Task 2: {perf1_after.accuracy:.1%}")
    
    if perf1_after.accuracy >= perf1.accuracy - 0.1:
        print(f"   ✅ No catastrophic forgetting!")
    else:
        print(f"   ⚠️  Some forgetting detected")
    
    print("\n📚 Learning Task 3: Summarization")
    summary_examples = [
        ("AI is transforming healthcare through diagnosis and treatment.", "AI in healthcare"),
        ("Climate change requires immediate global action.", "Climate action needed"),
    ]
    
    for input_text, output_text in summary_examples:
        learner.learn("summarization", input_text, output_text, importance=0.9)
    
    # Test 2: Knowledge consolidation
    print("\n" + "=" * 80)
    print("TEST 2: Knowledge Consolidation")
    print("=" * 80)
    
    print(f"\n📊 Knowledge Base Status:")
    print(f"   Tasks Learned: {len(learner.tasks)}")
    print(f"   Experiences Stored: {len(learner.experience_buffer)}")
    print(f"   Consolidated Knowledge: {len(learner.knowledge_base)} tasks")
    
    if learner.knowledge_base:
        print(f"\n   Consolidated Knowledge Samples:")
        for task_id, knowledge in list(learner.knowledge_base.items())[:2]:
            task_name = learner.tasks[task_id].name
            print(f"\n   {task_name}:")
            lines = knowledge.split('\n')[:3]
            for line in lines:
                if line.strip():
                    print(f"      {line}")
    
    # Test 3: Long-term retention
    print("\n" + "=" * 80)
    print("TEST 3: Long-term Retention Test")
    print("=" * 80)
    
    # Continue learning more examples
    print("\n📚 Learning additional examples...")
    
    for i in range(5):
        learner.learn(
            "translation",
            f"Example {i}",
            f"Ejemplo {i}",
            importance=0.5
        )
    
    # Re-evaluate all tasks
    print(f"\n📊 Final Performance Across All Tasks:")
    
    all_tests = {
        "sentiment": sentiment_test,
        "translation": [("Hello", "Hola"), ("Thank you", "Gracias")],
        "summarization": [("Brief text", "summary")]
    }
    
    for task_id, test_cases in all_tests.items():
        if task_id in learner.tasks:
            perf = learner.evaluate(task_id, test_cases)
            print(f"   {learner.tasks[task_id].name}: {perf.accuracy:.1%}")
    
    # Test 4: Forgetting analysis
    print("\n" + "=" * 80)
    print("TEST 4: Forgetting Analysis")
    print("=" * 80)
    
    forgetting_metrics = learner.get_forgetting_metrics()
    
    if "status" not in forgetting_metrics:
        print(f"\n📈 Forgetting Metrics:")
        for task_id, metrics in forgetting_metrics.items():
            print(f"\n   {metrics['task']}:")
            print(f"      Initial: {metrics['initial_accuracy']:.1%}")
            print(f"      Latest: {metrics['latest_accuracy']:.1%}")
            print(f"      Change: {metrics['forgetting']:+.1%}")
            print(f"      Status: {metrics['status']}")
    
    # Test 5: Experience buffer analysis
    print("\n" + "=" * 80)
    print("TEST 5: Experience Buffer Analysis")
    print("=" * 80)
    
    print(f"\n🗂️  Experience Buffer Status:")
    print(f"   Capacity: {learner.buffer_size}")
    print(f"   Current Size: {len(learner.experience_buffer)}")
    print(f"   Utilization: {len(learner.experience_buffer)/learner.buffer_size:.1%}")
    
    # Task distribution
    task_distribution = {}
    for exp in learner.experience_buffer:
        task_distribution[exp.task_id] = task_distribution.get(exp.task_id, 0) + 1
    
    print(f"\n   Task Distribution:")
    for task_id, count in task_distribution.items():
        task_name = learner.tasks[task_id].name
        percentage = count / len(learner.experience_buffer) * 100
        print(f"      {task_name}: {count} ({percentage:.1f}%)")
    
    # Rehearsal statistics
    rehearsal_counts = [exp.rehearsal_count for exp in learner.experience_buffer]
    if rehearsal_counts:
        avg_rehearsal = sum(rehearsal_counts) / len(rehearsal_counts)
        max_rehearsal = max(rehearsal_counts)
        print(f"\n   Rehearsal Statistics:")
        print(f"      Average: {avg_rehearsal:.1f}")
        print(f"      Maximum: {max_rehearsal}")
    
    # Summary
    print("\n" + "=" * 80)
    print("CONTINUAL LEARNING PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Lifelong Learning: Continuously acquire new knowledge
2. No Forgetting: Retain previously learned tasks
3. Efficient Memory: Consolidate experiences
4. Adaptability: Adjust to new domains
5. Scalability: Handle many tasks over time

The Catastrophic Forgetting Problem:
- Neural networks tend to forget old tasks when learning new ones
- Old knowledge is overwritten by new knowledge
- Performance on previous tasks degrades dramatically
- Critical challenge for continual learning systems

Forgetting Prevention Strategies:

1. Experience Replay:
   - Store subset of old experiences
   - Replay while learning new tasks
   - Maintains performance on old tasks
   - Buffer management is key

2. Regularization:
   - Constrain parameter changes
   - Protect important weights
   - Elastic Weight Consolidation (EWC)
   - Synaptic Intelligence

3. Progressive Networks:
   - Add new capacity for new tasks
   - Keep old networks frozen
   - Lateral connections for transfer
   - No forgetting by design

4. Knowledge Distillation:
   - Transfer knowledge to new model
   - Old model as teacher
   - Soft targets preserve knowledge
   - Compression friendly

5. Memory Consolidation:
   - Identify important experiences
   - Strengthen critical knowledge
   - Hierarchical organization
   - Gradual integration

Experience Buffer Management:
- Fixed size (ring buffer)
- Reservoir sampling
- Priority-based retention
- Diversity sampling
- Importance weighting

Consolidation Process:
1. Group related experiences
2. Extract general patterns
3. Compress into knowledge
4. Store in knowledge base
5. Use for future predictions

Task Priority:
- Critical: Never forget (safety, core capabilities)
- High: Important to retain
- Medium: Moderate importance
- Low: Can forget under pressure

Use Cases:
- Personal Assistants: Learn user preferences over time
- Recommendation Systems: Adapt to changing tastes
- Chatbots: Expand knowledge continuously
- Robotics: Learn new skills without forgetting old
- Adaptive Systems: Evolve with environment
- Multi-task Learning: Master many tasks

Challenges:
1. Memory Limitations: Can't store everything
2. Task Interference: Tasks conflict
3. Generalization: New tasks differ from old
4. Forgetting Measurement: How to evaluate?
5. Computational Cost: Replay and consolidation
6. Task Boundaries: When is a task new?

Best Practices:
1. Use experience replay
2. Consolidate regularly
3. Priority-based retention
4. Monitor all task performance
5. Adjust buffer size appropriately
6. Combine multiple strategies
7. Test for forgetting continuously

Production Considerations:
- Storage requirements for experiences
- Replay frequency and cost
- Consolidation timing
- Performance monitoring
- Graceful degradation
- User feedback integration
- Model versioning

Metrics to Track:
- Forward Transfer: New task learning speed
- Backward Transfer: Impact on old tasks
- Forgetting: Performance degradation
- Memory Efficiency: Knowledge per byte
- Learning Speed: Time to acquire new task

Advanced Techniques:
1. Meta-Learning: Learn how to learn continually
2. Compositional Learning: Reuse components
3. Modular Networks: Task-specific modules
4. Attention Mechanisms: Focus on relevant knowledge
5. Hierarchical Memory: Multi-level organization

Comparison with Related Patterns:
- vs. Meta-Learning: Long-term vs quick adaptation
- vs. Transfer Learning: Continuous vs one-shot
- vs. Multi-Task: Sequential vs simultaneous
- vs. Few-Shot: Many tasks vs few examples

Integration with Other Patterns:
- Memory Patterns: Foundation for continual learning
- Meta-Learning: Quick adaptation to new tasks
- Active Learning: What to learn next?
- Curriculum Learning: Order of learning tasks

Biological Inspiration:
- Complementary Learning Systems
- Systems Consolidation
- Hippocampus-Neocortex interaction
- Sleep and memory replay

Research Directions:
- Scalable to thousands of tasks
- Zero-shot continual learning
- Compositional generalization
- Causal continual learning
- Fairness in continual learning

The Continual Learning pattern enables agents to grow and
adapt continuously while retaining all previously acquired
knowledge - essential for long-lived, adaptive AI systems.
""")


if __name__ == "__main__":
    demonstrate_continual_learning()
