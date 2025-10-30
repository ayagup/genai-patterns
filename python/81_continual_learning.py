"""
Continual Learning Pattern

Enables agents to learn continuously from new experiences without catastrophic
forgetting of previous knowledge. Implements techniques like elastic weight
consolidation, experience replay, and progressive neural networks.

Key Concepts:
- Catastrophic forgetting prevention
- Lifelong learning
- Task-incremental learning
- Knowledge retention
- Adaptive plasticity

Use Cases:
- Long-running AI systems
- Evolving domains
- Personalized agents
- Adaptive assistants
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy
from datetime import datetime


class LearningStrategy(Enum):
    """Learning strategies for continual learning."""
    ELASTIC_WEIGHT_CONSOLIDATION = "ewc"  # Protect important weights
    EXPERIENCE_REPLAY = "replay"  # Replay old experiences
    PROGRESSIVE_NETWORKS = "progressive"  # Add new capacity
    KNOWLEDGE_DISTILLATION = "distillation"  # Transfer knowledge
    PARAMETER_ISOLATION = "isolation"  # Separate parameters per task


class TaskType(Enum):
    """Types of learning tasks."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    REASONING = "reasoning"
    CONTROL = "control"


@dataclass
class Task:
    """Represents a learning task."""
    id: str
    name: str
    task_type: TaskType
    description: str
    data: List[Dict[str, Any]] = field(default_factory=list)
    learned_at: Optional[datetime] = None
    
    def __repr__(self) -> str:
        return f"Task({self.id}: {self.name})"


@dataclass
class Experience:
    """Represents a single learning experience."""
    task_id: str
    input_data: Any
    target: Any
    timestamp: datetime
    importance: float = 1.0
    
    def __repr__(self) -> str:
        return f"Experience(task={self.task_id}, importance={self.importance:.2f})"


@dataclass
class ModelWeights:
    """Represents model parameters/weights."""
    weights: Dict[str, Any]
    task_id: str
    timestamp: datetime
    
    def copy(self) -> 'ModelWeights':
        """Create a deep copy of weights."""
        return ModelWeights(
            weights=copy.deepcopy(self.weights),
            task_id=self.task_id,
            timestamp=self.timestamp
        )


@dataclass
class FisherInformation:
    """Fisher information matrix for importance estimation."""
    task_id: str
    importance_weights: Dict[str, float]
    
    def get_importance(self, param_name: str) -> float:
        """Get importance of a parameter."""
        return self.importance_weights.get(param_name, 0.0)


class ExperienceBuffer:
    """Buffer for storing and sampling past experiences."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.experiences: List[Experience] = []
        self.task_counts: Dict[str, int] = {}
    
    def add(self, experience: Experience) -> None:
        """Add experience to buffer."""
        self.experiences.append(experience)
        self.task_counts[experience.task_id] = \
            self.task_counts.get(experience.task_id, 0) + 1
        
        # Remove oldest experiences if buffer is full
        if len(self.experiences) > self.max_size:
            removed = self.experiences.pop(0)
            self.task_counts[removed.task_id] -= 1
    
    def sample(self, n: int, task_id: Optional[str] = None) -> List[Experience]:
        """Sample experiences from buffer."""
        if task_id:
            pool = [e for e in self.experiences if e.task_id == task_id]
        else:
            pool = self.experiences
        
        if not pool:
            return []
        
        # Sample with importance weighting
        import random
        n = min(n, len(pool))
        weights = [e.importance for e in pool]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        return random.choices(pool, weights=probs, k=n)
    
    def get_task_ratio(self, task_id: str) -> float:
        """Get ratio of experiences for a task."""
        if not self.experiences:
            return 0.0
        return self.task_counts.get(task_id, 0) / len(self.experiences)


class ElasticWeightConsolidation:
    """Implements elastic weight consolidation (EWC) for continual learning."""
    
    def __init__(self, lambda_ewc: float = 0.5):
        self.lambda_ewc = lambda_ewc
        self.fisher_information: Dict[str, FisherInformation] = {}
        self.optimal_weights: Dict[str, ModelWeights] = {}
    
    def compute_fisher_information(
        self,
        task: Task,
        current_weights: ModelWeights
    ) -> FisherInformation:
        """Compute Fisher information for task."""
        # Simplified Fisher information computation
        # In practice, this would involve computing gradients
        importance_weights = {}
        
        for param_name in current_weights.weights.keys():
            # Simulate importance based on parameter magnitude and task data
            param_value = current_weights.weights[param_name]
            if isinstance(param_value, (int, float)):
                importance = abs(param_value) * len(task.data)
            else:
                importance = len(task.data)
            
            importance_weights[param_name] = importance
        
        return FisherInformation(
            task_id=task.id,
            importance_weights=importance_weights
        )
    
    def compute_ewc_loss(
        self,
        current_weights: ModelWeights,
        task_id: str
    ) -> float:
        """Compute EWC regularization loss."""
        if task_id not in self.fisher_information:
            return 0.0
        
        fisher = self.fisher_information[task_id]
        optimal = self.optimal_weights[task_id]
        
        ewc_loss = 0.0
        for param_name, current_value in current_weights.weights.items():
            if param_name in optimal.weights:
                optimal_value = optimal.weights[param_name]
                importance = fisher.get_importance(param_name)
                
                if isinstance(current_value, (int, float)):
                    diff = (current_value - optimal_value) ** 2
                    ewc_loss += importance * diff
        
        return self.lambda_ewc * ewc_loss / 2
    
    def save_task_state(self, task: Task, weights: ModelWeights) -> None:
        """Save Fisher information and optimal weights for task."""
        fisher = self.compute_fisher_information(task, weights)
        self.fisher_information[task.id] = fisher
        self.optimal_weights[task.id] = weights.copy()
    
    def get_total_ewc_loss(self, current_weights: ModelWeights) -> float:
        """Compute total EWC loss across all tasks."""
        total_loss = 0.0
        for task_id in self.fisher_information.keys():
            total_loss += self.compute_ewc_loss(current_weights, task_id)
        return total_loss


class ContinualLearningAgent:
    """Agent that learns continuously without catastrophic forgetting."""
    
    def __init__(
        self,
        name: str,
        strategy: LearningStrategy = LearningStrategy.ELASTIC_WEIGHT_CONSOLIDATION,
        buffer_size: int = 1000
    ):
        self.name = name
        self.strategy = strategy
        self.tasks: Dict[str, Task] = {}
        self.current_task: Optional[Task] = None
        
        # Model state
        self.weights = ModelWeights(
            weights={"param_1": 0.5, "param_2": 0.3, "bias": 0.1},
            task_id="init",
            timestamp=datetime.now()
        )
        
        # Continual learning components
        self.ewc = ElasticWeightConsolidation(lambda_ewc=0.5)
        self.experience_buffer = ExperienceBuffer(max_size=buffer_size)
        self.task_performance: Dict[str, List[float]] = {}
    
    def add_task(self, task: Task) -> None:
        """Add a new task to learn."""
        self.tasks[task.id] = task
        self.task_performance[task.id] = []
        print(f"[{self.name}] Added task: {task.name}")
    
    def learn_task(self, task: Task, epochs: int = 10) -> Dict[str, Any]:
        """Learn a new task while preserving previous knowledge."""
        print(f"\n[{self.name}] Learning task: {task.name}")
        print(f"Strategy: {self.strategy.value}")
        
        self.current_task = task
        task.learned_at = datetime.now()
        
        # Store experiences in buffer
        for data_point in task.data:
            experience = Experience(
                task_id=task.id,
                input_data=data_point.get("input"),
                target=data_point.get("target"),
                timestamp=datetime.now()
            )
            self.experience_buffer.add(experience)
        
        # Training simulation
        initial_loss = 1.0
        final_loss = 0.1
        
        losses = []
        for epoch in range(epochs):
            # Simulate training with strategy
            if self.strategy == LearningStrategy.ELASTIC_WEIGHT_CONSOLIDATION:
                # Compute EWC loss
                ewc_loss = self.ewc.get_total_ewc_loss(self.weights)
                task_loss = initial_loss * (1 - epoch / epochs)
                total_loss = task_loss + ewc_loss
                losses.append(total_loss)
                
            elif self.strategy == LearningStrategy.EXPERIENCE_REPLAY:
                # Sample from replay buffer
                replayed = self.experience_buffer.sample(n=10)
                task_loss = initial_loss * (1 - epoch / epochs)
                replay_loss = len(replayed) * 0.01
                total_loss = task_loss + replay_loss
                losses.append(total_loss)
            
            else:
                task_loss = initial_loss * (1 - epoch / epochs)
                losses.append(task_loss)
        
        # Update weights (simplified)
        for param_name in self.weights.weights:
            current_value = self.weights.weights[param_name]
            if isinstance(current_value, (int, float)):
                # Small random update
                import random
                self.weights.weights[param_name] = current_value + random.uniform(-0.1, 0.1)
        
        self.weights.task_id = task.id
        self.weights.timestamp = datetime.now()
        
        # Save task state for EWC
        if self.strategy == LearningStrategy.ELASTIC_WEIGHT_CONSOLIDATION:
            self.ewc.save_task_state(task, self.weights)
        
        # Record performance
        self.task_performance[task.id].append(final_loss)
        
        return {
            "task_id": task.id,
            "task_name": task.name,
            "final_loss": losses[-1],
            "epochs": epochs,
            "strategy": self.strategy.value,
            "experiences_stored": len(self.experience_buffer.experiences)
        }
    
    def evaluate_task(self, task_id: str) -> Dict[str, Any]:
        """Evaluate performance on a specific task."""
        if task_id not in self.tasks:
            return {"error": "Task not found"}
        
        task = self.tasks[task_id]
        
        # Simulate evaluation
        if task_id in self.task_performance and self.task_performance[task_id]:
            # Performance degrades slightly over time (forgetting)
            num_tasks_learned = len([t for t in self.tasks.values() if t.learned_at])
            forgetting_factor = 1.0 + (num_tasks_learned * 0.05)
            base_loss = self.task_performance[task_id][-1]
            current_loss = base_loss * forgetting_factor
        else:
            current_loss = 1.0  # Not learned yet
        
        # EWC reduces forgetting
        if self.strategy == LearningStrategy.ELASTIC_WEIGHT_CONSOLIDATION:
            current_loss *= 0.8  # 20% less forgetting with EWC
        
        return {
            "task_id": task_id,
            "task_name": task.name,
            "loss": current_loss,
            "accuracy": max(0.0, 1.0 - current_loss),
            "learned": task.learned_at is not None
        }
    
    def evaluate_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Evaluate performance on all learned tasks."""
        results = {}
        for task_id in self.tasks:
            results[task_id] = self.evaluate_task(task_id)
        return results
    
    def get_forgetting_metrics(self) -> Dict[str, Any]:
        """Calculate forgetting metrics across tasks."""
        evaluations = self.evaluate_all_tasks()
        learned_tasks = [t for t in self.tasks.values() if t.learned_at]
        
        if not learned_tasks:
            return {"average_accuracy": 0.0, "forgetting_rate": 0.0}
        
        accuracies = [e["accuracy"] for e in evaluations.values() if e.get("learned")]
        average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        # Forgetting rate: how much performance degraded from initial learning
        forgetting_rate = max(0.0, 1.0 - average_accuracy)
        
        return {
            "average_accuracy": average_accuracy,
            "forgetting_rate": forgetting_rate,
            "num_tasks_learned": len(learned_tasks),
            "strategy": self.strategy.value
        }
    
    def replay_experiences(self, n: int = 10) -> List[Experience]:
        """Replay past experiences for memory consolidation."""
        return self.experience_buffer.sample(n)


def demonstrate_continual_learning():
    """Demonstrate continual learning pattern."""
    print("=" * 60)
    print("CONTINUAL LEARNING PATTERN DEMONSTRATION")
    print("=" * 60)
    
    # Create agents with different strategies
    ewc_agent = ContinualLearningAgent(
        name="EWC Agent",
        strategy=LearningStrategy.ELASTIC_WEIGHT_CONSOLIDATION
    )
    
    replay_agent = ContinualLearningAgent(
        name="Replay Agent",
        strategy=LearningStrategy.EXPERIENCE_REPLAY
    )
    
    # Create sequential tasks
    tasks = [
        Task(
            id="task1",
            name="Classify Animals",
            task_type=TaskType.CLASSIFICATION,
            description="Classify animals into categories",
            data=[
                {"input": "dog", "target": "mammal"},
                {"input": "cat", "target": "mammal"},
                {"input": "bird", "target": "avian"}
            ]
        ),
        Task(
            id="task2",
            name="Classify Vehicles",
            task_type=TaskType.CLASSIFICATION,
            description="Classify vehicles into types",
            data=[
                {"input": "car", "target": "ground"},
                {"input": "plane", "target": "air"},
                {"input": "boat", "target": "water"}
            ]
        ),
        Task(
            id="task3",
            name="Classify Foods",
            task_type=TaskType.CLASSIFICATION,
            description="Classify foods into categories",
            data=[
                {"input": "apple", "target": "fruit"},
                {"input": "carrot", "target": "vegetable"},
                {"input": "bread", "target": "grain"}
            ]
        )
    ]
    
    # Test with EWC agent
    print("\n" + "=" * 60)
    print("Testing with Elastic Weight Consolidation")
    print("=" * 60)
    
    for task in tasks:
        ewc_agent.add_task(task)
        result = ewc_agent.learn_task(task, epochs=5)
        print(f"\nLearning result: {result['task_name']}")
        print(f"  Final loss: {result['final_loss']:.4f}")
        print(f"  Experiences stored: {result['experiences_stored']}")
    
    print("\n--- Evaluating all tasks after learning ---")
    ewc_results = ewc_agent.evaluate_all_tasks()
    for task_id, result in ewc_results.items():
        print(f"{result['task_name']}: accuracy = {result['accuracy']:.2%}")
    
    ewc_metrics = ewc_agent.get_forgetting_metrics()
    print(f"\nEWC Agent Metrics:")
    print(f"  Average accuracy: {ewc_metrics['average_accuracy']:.2%}")
    print(f"  Forgetting rate: {ewc_metrics['forgetting_rate']:.2%}")
    
    # Test with replay agent
    print("\n" + "=" * 60)
    print("Testing with Experience Replay")
    print("=" * 60)
    
    for task in tasks:
        replay_agent.add_task(task)
        result = replay_agent.learn_task(task, epochs=5)
        print(f"\nLearning result: {result['task_name']}")
        print(f"  Final loss: {result['final_loss']:.4f}")
    
    print("\n--- Evaluating all tasks after learning ---")
    replay_results = replay_agent.evaluate_all_tasks()
    for task_id, result in replay_results.items():
        print(f"{result['task_name']}: accuracy = {result['accuracy']:.2%}")
    
    replay_metrics = replay_agent.get_forgetting_metrics()
    print(f"\nReplay Agent Metrics:")
    print(f"  Average accuracy: {replay_metrics['average_accuracy']:.2%}")
    print(f"  Forgetting rate: {replay_metrics['forgetting_rate']:.2%}")
    
    # Compare strategies
    print("\n" + "=" * 60)
    print("Strategy Comparison")
    print("=" * 60)
    print(f"EWC Agent - Forgetting rate: {ewc_metrics['forgetting_rate']:.2%}")
    print(f"Replay Agent - Forgetting rate: {replay_metrics['forgetting_rate']:.2%}")
    
    better = "EWC" if ewc_metrics['forgetting_rate'] < replay_metrics['forgetting_rate'] else "Replay"
    print(f"\n{better} strategy shows less catastrophic forgetting!")


if __name__ == "__main__":
    demonstrate_continual_learning()
