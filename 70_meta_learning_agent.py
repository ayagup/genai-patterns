"""
Meta-Learning Agent Pattern

An agent that learns how to learn and adapt quickly to new tasks.
Implements few-shot adaptation, task transfer, and learning optimization strategies.

Use Cases:
- Personalization systems that adapt to individual users quickly
- Multi-domain agents that transfer knowledge across tasks
- Few-shot learning scenarios with limited training data
- Dynamic environments requiring rapid adaptation
- Cross-task knowledge transfer and generalization

Benefits:
- Quick adaptation: Learn new tasks from few examples
- Knowledge transfer: Apply learning from one task to another
- Sample efficiency: Requires less data for new tasks
- Generalization: Better performance on unseen tasks
- Continuous improvement: Learns better learning strategies
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import math


class TaskDomain(Enum):
    """Different task domains for meta-learning"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    REASONING = "reasoning"
    TRANSLATION = "translation"


class AdaptationStrategy(Enum):
    """Strategies for adapting to new tasks"""
    FINE_TUNING = "fine_tuning"
    PROMPT_ADAPTATION = "prompt_adaptation"
    IN_CONTEXT_LEARNING = "in_context_learning"
    PARAMETER_EFFICIENT = "parameter_efficient"


@dataclass
class Task:
    """Represents a task for meta-learning"""
    task_id: str
    domain: TaskDomain
    description: str
    support_examples: List[Dict[str, Any]]  # Training examples
    query_examples: List[Dict[str, Any]]  # Test examples
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """A meta-learning episode (task instance)"""
    task: Task
    support_set: List[Dict[str, Any]]
    query_set: List[Dict[str, Any]]
    k_shot: int  # Number of examples per class
    
    def get_support_size(self) -> int:
        return len(self.support_set)
    
    def get_query_size(self) -> int:
        return len(self.query_set)


@dataclass
class MetaKnowledge:
    """Knowledge learned across tasks"""
    task_patterns: Dict[str, Any] = field(default_factory=dict)
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)
    learned_strategies: List[str] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    
    def add_pattern(self, pattern_name: str, pattern_data: Any) -> None:
        """Store learned pattern"""
        self.task_patterns[pattern_name] = pattern_data
    
    def record_adaptation(self, task_id: str, strategy: str, performance: float) -> None:
        """Record adaptation experience"""
        self.adaptation_history.append({
            "task_id": task_id,
            "strategy": strategy,
            "performance": performance
        })
        self.performance_history.append(performance)
    
    def get_best_strategy(self, task_domain: TaskDomain) -> Optional[str]:
        """Get best performing strategy for a domain"""
        domain_adaptations = [
            a for a in self.adaptation_history
            if a.get("domain") == task_domain.value
        ]
        
        if not domain_adaptations:
            return None
        
        best = max(domain_adaptations, key=lambda x: x["performance"])
        return best["strategy"]
    
    def get_average_performance(self) -> float:
        """Get average performance across all adaptations"""
        if not self.performance_history:
            return 0.0
        return sum(self.performance_history) / len(self.performance_history)


class TaskEmbedding:
    """Learns representations of tasks for similarity comparison"""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.task_embeddings: Dict[str, List[float]] = {}
    
    def embed_task(self, task: Task) -> List[float]:
        """
        Create embedding for a task.
        In production, this would use learned embeddings.
        """
        # Simulate task embedding based on task characteristics
        embedding = [random.random() for _ in range(self.embedding_dim)]
        
        # Add domain-specific features
        domain_features = {
            TaskDomain.CLASSIFICATION: [1.0, 0.0, 0.0, 0.0, 0.0],
            TaskDomain.REGRESSION: [0.0, 1.0, 0.0, 0.0, 0.0],
            TaskDomain.GENERATION: [0.0, 0.0, 1.0, 0.0, 0.0],
            TaskDomain.REASONING: [0.0, 0.0, 0.0, 1.0, 0.0],
            TaskDomain.TRANSLATION: [0.0, 0.0, 0.0, 0.0, 1.0],
        }
        
        embedding[:5] = domain_features.get(task.domain, [0.0] * 5)
        
        self.task_embeddings[task.task_id] = embedding
        return embedding
    
    def compute_similarity(self, task1_id: str, task2_id: str) -> float:
        """Compute similarity between two tasks"""
        if task1_id not in self.task_embeddings or task2_id not in self.task_embeddings:
            return 0.0
        
        emb1 = self.task_embeddings[task1_id]
        emb2 = self.task_embeddings[task2_id]
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        magnitude1 = math.sqrt(sum(a * a for a in emb1))
        magnitude2 = math.sqrt(sum(b * b for b in emb2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def find_similar_tasks(self, task_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar tasks"""
        if task_id not in self.task_embeddings:
            return []
        
        similarities = []
        for other_id in self.task_embeddings:
            if other_id != task_id:
                sim = self.compute_similarity(task_id, other_id)
                similarities.append((other_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class MetaLearningAgent:
    """
    Meta-Learning Agent
    
    Learns how to learn and adapts quickly to new tasks using
    meta-knowledge accumulated across multiple task experiences.
    """
    
    def __init__(self, name: str = "Meta-Learning Agent"):
        self.name = name
        self.meta_knowledge = MetaKnowledge()
        self.task_embedder = TaskEmbedding()
        self.experienced_tasks: Dict[str, Task] = {}
        self.adaptation_count = 0
        
        # Meta-parameters (learned across tasks)
        self.learning_rate = 0.1
        self.adaptation_steps = 5
    
    def observe_task(self, task: Task) -> None:
        """Observe a new task and store it"""
        print(f"\n[Meta-Learning] Observing task: {task.task_id}")
        print(f"  Domain: {task.domain.value}")
        print(f"  Description: {task.description}")
        print(f"  Support examples: {len(task.support_examples)}")
        
        self.experienced_tasks[task.task_id] = task
        self.task_embedder.embed_task(task)
    
    def adapt_to_task(
        self,
        episode: Episode,
        strategy: AdaptationStrategy = AdaptationStrategy.IN_CONTEXT_LEARNING
    ) -> Dict[str, Any]:
        """
        Adapt to a new task using meta-learned knowledge
        """
        task = episode.task
        print(f"\n[Adaptation] Adapting to task: {task.task_id}")
        print(f"  Strategy: {strategy.value}")
        print(f"  Support set size: {episode.get_support_size()}")
        print(f"  Query set size: {episode.get_query_size()}")
        
        # Find similar tasks for knowledge transfer
        similar_tasks = self.task_embedder.find_similar_tasks(task.task_id, top_k=3)
        
        if similar_tasks:
            print(f"\n[Transfer Learning] Similar tasks found:")
            for task_id, similarity in similar_tasks:
                print(f"  - {task_id}: {similarity:.3f} similarity")
        
        # Adapt based on strategy
        if strategy == AdaptationStrategy.IN_CONTEXT_LEARNING:
            result = self._in_context_adaptation(episode, similar_tasks)
        elif strategy == AdaptationStrategy.FINE_TUNING:
            result = self._fine_tuning_adaptation(episode, similar_tasks)
        elif strategy == AdaptationStrategy.PROMPT_ADAPTATION:
            result = self._prompt_adaptation(episode, similar_tasks)
        else:
            result = self._parameter_efficient_adaptation(episode, similar_tasks)
        
        # Record adaptation experience
        self.meta_knowledge.record_adaptation(
            task.task_id,
            strategy.value,
            result["performance"]
        )
        
        self.adaptation_count += 1
        
        # Update meta-parameters based on experience
        self._update_meta_parameters()
        
        return result
    
    def _in_context_adaptation(
        self,
        episode: Episode,
        similar_tasks: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """Adapt using in-context learning (few-shot prompting)"""
        print(f"\n[In-Context Learning] Building few-shot context...")
        
        # Simulate building context from support examples
        context = []
        for example in episode.support_set:
            context.append(f"Example: {example}")
        
        # Simulate inference on query set
        predictions = []
        for query in episode.query_set:
            # In production: Use LLM with few-shot context
            pred = self._simulate_prediction(query, context)
            predictions.append(pred)
        
        # Simulate performance evaluation
        performance = random.uniform(0.6, 0.95)  # Higher with more experience
        
        return {
            "strategy": "in_context_learning",
            "performance": performance,
            "predictions": predictions,
            "context_size": len(context)
        }
    
    def _fine_tuning_adaptation(
        self,
        episode: Episode,
        similar_tasks: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """Adapt using fine-tuning on support set"""
        print(f"\n[Fine-Tuning] Training on support set...")
        
        # Simulate fine-tuning steps
        for step in range(self.adaptation_steps):
            loss = 1.0 / (step + 1)  # Simulated decreasing loss
            if step % 2 == 0:
                print(f"  Step {step + 1}/{self.adaptation_steps}, Loss: {loss:.4f}")
        
        # Higher performance but more computational cost
        performance = random.uniform(0.7, 0.98)
        
        return {
            "strategy": "fine_tuning",
            "performance": performance,
            "training_steps": self.adaptation_steps,
            "final_loss": loss
        }
    
    def _prompt_adaptation(
        self,
        episode: Episode,
        similar_tasks: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """Adapt by optimizing prompts"""
        print(f"\n[Prompt Adaptation] Optimizing prompt...")
        
        # Simulate prompt optimization
        initial_prompt = f"Task: {episode.task.description}"
        optimized_prompt = self._optimize_prompt(initial_prompt, episode.support_set)
        
        print(f"  Optimized prompt: {optimized_prompt[:100]}...")
        
        performance = random.uniform(0.65, 0.92)
        
        return {
            "strategy": "prompt_adaptation",
            "performance": performance,
            "prompt": optimized_prompt
        }
    
    def _parameter_efficient_adaptation(
        self,
        episode: Episode,
        similar_tasks: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """Adapt using parameter-efficient methods (e.g., adapters, LoRA)"""
        print(f"\n[Parameter-Efficient] Using efficient adaptation...")
        
        # Simulate adapter training (only small number of parameters)
        adapter_params = len(episode.support_set) * 10  # Simulated
        print(f"  Trainable parameters: {adapter_params}")
        
        performance = random.uniform(0.68, 0.95)
        
        return {
            "strategy": "parameter_efficient",
            "performance": performance,
            "adapter_params": adapter_params
        }
    
    def _simulate_prediction(self, query: Any, context: List[Any]) -> Any:
        """Simulate making a prediction"""
        # In production: Use actual LLM inference
        return {"prediction": "simulated_output", "confidence": random.uniform(0.6, 0.95)}
    
    def _optimize_prompt(self, initial_prompt: str, examples: List[Any]) -> str:
        """Optimize prompt based on examples"""
        # In production: Use prompt optimization techniques
        optimized = f"{initial_prompt}\n\nExamples:\n"
        for ex in examples[:3]:  # Use top 3 examples
            optimized += f"- {ex}\n"
        optimized += "\nNow, solve the following:"
        return optimized
    
    def _update_meta_parameters(self) -> None:
        """Update meta-parameters based on accumulated experience"""
        if self.adaptation_count > 0:
            avg_performance = self.meta_knowledge.get_average_performance()
            
            # Adjust learning rate based on performance
            if avg_performance > 0.8:
                self.learning_rate *= 0.95  # Decrease for stability
            elif avg_performance < 0.6:
                self.learning_rate *= 1.05  # Increase for exploration
            
            # Adjust adaptation steps
            if avg_performance > 0.85:
                self.adaptation_steps = max(3, self.adaptation_steps - 1)
            elif avg_performance < 0.65:
                self.adaptation_steps = min(10, self.adaptation_steps + 1)
    
    def transfer_knowledge(self, source_task_id: str, target_task: Task) -> Dict[str, Any]:
        """
        Transfer knowledge from source task to target task
        """
        print(f"\n[Knowledge Transfer]")
        print(f"  Source: {source_task_id}")
        print(f"  Target: {target_task.task_id}")
        
        if source_task_id not in self.experienced_tasks:
            return {
                "success": False,
                "message": "Source task not found"
            }
        
        source_task = self.experienced_tasks[source_task_id]
        
        # Compute task similarity
        similarity = self.task_embedder.compute_similarity(
            source_task_id,
            target_task.task_id
        )
        
        print(f"  Task similarity: {similarity:.3f}")
        
        if similarity > 0.7:
            transfer_type = "direct_transfer"
            effectiveness = "high"
        elif similarity > 0.4:
            transfer_type = "partial_transfer"
            effectiveness = "medium"
        else:
            transfer_type = "minimal_transfer"
            effectiveness = "low"
        
        print(f"  Transfer type: {transfer_type}")
        print(f"  Expected effectiveness: {effectiveness}")
        
        return {
            "success": True,
            "similarity": similarity,
            "transfer_type": transfer_type,
            "effectiveness": effectiveness
        }
    
    def get_meta_learning_stats(self) -> Dict[str, Any]:
        """Get meta-learning statistics"""
        return {
            "total_tasks_observed": len(self.experienced_tasks),
            "total_adaptations": self.adaptation_count,
            "average_performance": self.meta_knowledge.get_average_performance(),
            "current_learning_rate": self.learning_rate,
            "current_adaptation_steps": self.adaptation_steps,
            "learned_patterns": len(self.meta_knowledge.task_patterns)
        }


def demonstrate_meta_learning():
    """
    Demonstrate Meta-Learning Agent pattern
    """
    print("=" * 70)
    print("META-LEARNING AGENT PATTERN DEMONSTRATION")
    print("=" * 70)
    
    # Create meta-learning agent
    agent = MetaLearningAgent("Universal Learner")
    
    # Example 1: Text classification tasks
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Few-Shot Text Classification")
    print("=" * 70)
    
    # Create classification tasks
    sentiment_task = Task(
        task_id="sentiment_analysis",
        domain=TaskDomain.CLASSIFICATION,
        description="Classify sentiment of movie reviews",
        support_examples=[
            {"text": "Great movie!", "label": "positive"},
            {"text": "Terrible film", "label": "negative"},
            {"text": "Amazing acting", "label": "positive"}
        ],
        query_examples=[
            {"text": "Best movie ever", "label": "positive"},
            {"text": "Waste of time", "label": "negative"}
        ]
    )
    
    agent.observe_task(sentiment_task)
    
    # Create episode for adaptation
    episode = Episode(
        task=sentiment_task,
        support_set=sentiment_task.support_examples,
        query_set=sentiment_task.query_examples,
        k_shot=3
    )
    
    # Adapt to task
    result = agent.adapt_to_task(episode, AdaptationStrategy.IN_CONTEXT_LEARNING)
    print(f"\n[Results] Performance: {result['performance']:.3f}")
    
    # Example 2: Transfer learning across domains
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Transfer Learning Across Tasks")
    print("=" * 70)
    
    # Create similar task
    review_task = Task(
        task_id="product_reviews",
        domain=TaskDomain.CLASSIFICATION,
        description="Classify product review sentiment",
        support_examples=[
            {"text": "Love this product", "label": "positive"},
            {"text": "Poor quality", "label": "negative"}
        ],
        query_examples=[
            {"text": "Excellent purchase", "label": "positive"}
        ]
    )
    
    agent.observe_task(review_task)
    
    # Transfer knowledge from sentiment to reviews
    transfer_result = agent.transfer_knowledge("sentiment_analysis", review_task)
    
    if transfer_result["success"]:
        print(f"\n✓ Knowledge transfer successful!")
        print(f"  Similarity: {transfer_result['similarity']:.3f}")
        print(f"  Effectiveness: {transfer_result['effectiveness']}")
    
    # Example 3: Multiple adaptation strategies
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Comparing Adaptation Strategies")
    print("=" * 70)
    
    translation_task = Task(
        task_id="en_to_fr",
        domain=TaskDomain.TRANSLATION,
        description="Translate English to French",
        support_examples=[
            {"en": "Hello", "fr": "Bonjour"},
            {"en": "Goodbye", "fr": "Au revoir"}
        ],
        query_examples=[
            {"en": "Thank you", "fr": "Merci"}
        ]
    )
    
    agent.observe_task(translation_task)
    
    episode3 = Episode(
        task=translation_task,
        support_set=translation_task.support_examples,
        query_set=translation_task.query_examples,
        k_shot=2
    )
    
    strategies = [
        AdaptationStrategy.IN_CONTEXT_LEARNING,
        AdaptationStrategy.FINE_TUNING,
        AdaptationStrategy.PROMPT_ADAPTATION,
        AdaptationStrategy.PARAMETER_EFFICIENT
    ]
    
    results = []
    for strategy in strategies:
        result = agent.adapt_to_task(episode3, strategy)
        results.append((strategy.value, result["performance"]))
    
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    for strategy_name, performance in results:
        print(f"{strategy_name:25s}: {performance:.3f}")
    
    # Show meta-learning statistics
    print("\n" + "=" * 70)
    print("META-LEARNING STATISTICS")
    print("=" * 70)
    
    stats = agent.get_meta_learning_stats()
    for key, value in stats.items():
        print(f"{key:30s}: {value}")


def demonstrate_benefits():
    """Show benefits of meta-learning"""
    print("\n" + "=" * 70)
    print("META-LEARNING BENEFITS")
    print("=" * 70)
    
    print("\n✓ Key Advantages:")
    print("\n1. QUICK ADAPTATION:")
    print("   - Learn new tasks from just a few examples")
    print("   - No need for large training datasets")
    print("   - Rapid deployment to new domains")
    
    print("\n2. KNOWLEDGE TRANSFER:")
    print("   - Apply learning from one task to another")
    print("   - Build on accumulated experience")
    print("   - Improve with each new task")
    
    print("\n3. SAMPLE EFFICIENCY:")
    print("   - Requires minimal data for new tasks")
    print("   - Cost-effective learning")
    print("   - Faster iteration cycles")
    
    print("\n4. GENERALIZATION:")
    print("   - Better performance on unseen tasks")
    print("   - Learns general problem-solving strategies")
    print("   - More robust to distribution shift")
    
    print("\n5. CONTINUOUS IMPROVEMENT:")
    print("   - Learns better learning strategies over time")
    print("   - Self-optimizing meta-parameters")
    print("   - Accumulates meta-knowledge")


if __name__ == "__main__":
    demonstrate_meta_learning()
    demonstrate_benefits()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Meta-learning enables quick adaptation to new tasks
2. Transfers knowledge across related tasks effectively
3. Requires minimal examples for new task learning
4. Accumulates meta-knowledge over time
5. Self-improves its learning strategies

Best Practices:
- Maintain diverse task experience for better transfer
- Track task similarities for knowledge reuse
- Choose adaptation strategy based on task requirements
- Update meta-parameters based on performance
- Balance exploration vs exploitation in learning
- Monitor and analyze meta-learning statistics
    """)
