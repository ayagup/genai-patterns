"""
Continuous Learning Agent Pattern

Learns continuously from new data while retaining old knowledge.
Implements techniques to avoid catastrophic forgetting.

Use Cases:
- Lifelong learning systems
- Adaptive applications
- Online learning
- Evolving environments

Advantages:
- Adapts to new data
- Retains old knowledge
- No retraining from scratch
- Handles concept drift
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import random


class LearningStrategy(Enum):
    """Continuous learning strategies"""
    ELASTIC_WEIGHT_CONSOLIDATION = "ewc"
    PROGRESSIVE_NEURAL_NETWORKS = "progressive"
    LEARNING_WITHOUT_FORGETTING = "lwf"
    MEMORY_REPLAY = "memory_replay"
    INCREMENTAL = "incremental"


class ConceptDriftType(Enum):
    """Types of concept drift"""
    SUDDEN = "sudden"
    GRADUAL = "gradual"
    INCREMENTAL = "incremental"
    RECURRING = "recurring"


@dataclass
class LearningTask:
    """Learning task"""
    task_id: str
    name: str
    data_size: int
    task_type: str
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSnapshot:
    """Snapshot of model at a point in time"""
    snapshot_id: str
    task_id: str
    weights: Dict[str, List[float]]
    importance_weights: Optional[Dict[str, List[float]]] = None
    performance: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryBuffer:
    """Memory buffer for experience replay"""
    buffer_id: str
    max_size: int
    samples: List[Dict[str, Any]] = field(default_factory=list)
    task_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class DriftDetection:
    """Concept drift detection result"""
    detected: bool
    drift_type: Optional[ConceptDriftType]
    magnitude: float
    timestamp: datetime
    affected_tasks: List[str]


class ImportanceWeightCalculator:
    """Calculates importance weights for parameters"""
    
    def calculate_fisher_information(self,
                                     model_weights: Dict[str, List[float]],
                                     data_samples: List[Dict[str, Any]]
                                     ) -> Dict[str, List[float]]:
        """
        Calculate Fisher Information Matrix (simplified).
        
        Args:
            model_weights: Current model weights
            data_samples: Training data samples
            
        Returns:
            Fisher information for each parameter
        """
        fisher = {}
        
        for layer_name, weights in model_weights.items():
            # Simplified: use gradient variance as proxy for importance
            fisher[layer_name] = [
                abs(w) * random.uniform(0.5, 1.5)
                for w in weights
            ]
        
        return fisher


class MemoryManager:
    """Manages memory buffer for experience replay"""
    
    def __init__(self, max_size: int = 1000):
        self.buffer = MemoryBuffer(
            buffer_id="main_buffer",
            max_size=max_size
        )
    
    def add_samples(self,
                   samples: List[Dict[str, Any]],
                   task_id: str) -> None:
        """
        Add samples to memory buffer.
        
        Args:
            samples: New samples
            task_id: Task ID
        """
        for sample in samples:
            if len(self.buffer.samples) >= self.buffer.max_size:
                # Remove oldest sample
                removed = self.buffer.samples.pop(0)
                removed_task = removed.get("task_id", "unknown")
                self.buffer.task_distribution[removed_task] -= 1
            
            # Add new sample
            sample["task_id"] = task_id
            self.buffer.samples.append(sample)
            
            # Update distribution
            self.buffer.task_distribution[task_id] = (
                self.buffer.task_distribution.get(task_id, 0) + 1
            )
    
    def sample_batch(self,
                    batch_size: int,
                    task_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Sample batch from memory.
        
        Args:
            batch_size: Batch size
            task_id: Optional task ID to sample from
            
        Returns:
            Sampled batch
        """
        if task_id:
            # Sample from specific task
            task_samples = [
                s for s in self.buffer.samples
                if s.get("task_id") == task_id
            ]
            if len(task_samples) >= batch_size:
                return random.sample(task_samples, batch_size)
            else:
                return task_samples
        else:
            # Sample from all tasks
            if len(self.buffer.samples) >= batch_size:
                return random.sample(self.buffer.samples, batch_size)
            else:
                return self.buffer.samples.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory buffer statistics"""
        return {
            "total_samples": len(self.buffer.samples),
            "max_size": self.buffer.max_size,
            "task_distribution": self.buffer.task_distribution,
            "utilization": len(self.buffer.samples) / self.buffer.max_size
        }


class DriftDetector:
    """Detects concept drift"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history: List[float] = []
    
    def detect_drift(self,
                    current_performance: float,
                    threshold: float = 0.1) -> DriftDetection:
        """
        Detect concept drift based on performance.
        
        Args:
            current_performance: Current model performance
            threshold: Detection threshold
            
        Returns:
            Drift detection result
        """
        self.performance_history.append(current_performance)
        
        # Keep only recent history
        if len(self.performance_history) > self.window_size:
            self.performance_history = self.performance_history[-self.window_size:]
        
        if len(self.performance_history) < 10:
            return DriftDetection(
                detected=False,
                drift_type=None,
                magnitude=0.0,
                timestamp=datetime.now(),
                affected_tasks=[]
            )
        
        # Calculate performance trend
        recent_avg = sum(
            self.performance_history[-10:]
        ) / 10
        
        older_avg = sum(
            self.performance_history[-20:-10] if len(self.performance_history) >= 20
            else self.performance_history[:-10]
        ) / max(len(self.performance_history[:-10]), 1)
        
        performance_drop = older_avg - recent_avg
        
        # Detect drift
        if performance_drop > threshold:
            # Determine drift type
            if performance_drop > threshold * 2:
                drift_type = ConceptDriftType.SUDDEN
            else:
                drift_type = ConceptDriftType.GRADUAL
            
            return DriftDetection(
                detected=True,
                drift_type=drift_type,
                magnitude=performance_drop,
                timestamp=datetime.now(),
                affected_tasks=["current"]
            )
        
        return DriftDetection(
            detected=False,
            drift_type=None,
            magnitude=0.0,
            timestamp=datetime.now(),
            affected_tasks=[]
        )


class ContinuousLearningAgent:
    """
    Agent that learns continuously while avoiding catastrophic forgetting.
    Adapts to new tasks while retaining performance on old tasks.
    """
    
    def __init__(self,
                 strategy: LearningStrategy = LearningStrategy.ELASTIC_WEIGHT_CONSOLIDATION,
                 memory_size: int = 1000):
        self.strategy = strategy
        
        # Components
        self.importance_calculator = ImportanceWeightCalculator()
        self.memory_manager = MemoryManager(max_size=memory_size)
        self.drift_detector = DriftDetector()
        
        # State
        self.current_weights: Dict[str, List[float]] = {}
        self.tasks: Dict[str, LearningTask] = {}
        self.snapshots: List[ModelSnapshot] = []
        self.task_counter = 0
        
        # Performance tracking
        self.task_performance: Dict[str, List[float]] = {}
    
    def initialize_model(self, layer_sizes: List[int]) -> None:
        """
        Initialize model weights.
        
        Args:
            layer_sizes: Size of each layer
        """
        self.current_weights = {}
        
        for i, size in enumerate(layer_sizes):
            layer_name = "layer_{}".format(i)
            self.current_weights[layer_name] = [
                random.gauss(0, 0.1) for _ in range(size)
            ]
    
    def learn_task(self,
                  task_name: str,
                  training_data: List[Dict[str, Any]],
                  num_epochs: int = 10) -> Dict[str, float]:
        """
        Learn a new task continuously.
        
        Args:
            task_name: Task name
            training_data: Training data
            num_epochs: Number of training epochs
            
        Returns:
            Performance metrics
        """
        # Create task
        task = LearningTask(
            task_id="task_{}".format(self.task_counter),
            name=task_name,
            data_size=len(training_data),
            task_type="classification",
            created_at=datetime.now()
        )
        
        self.tasks[task.task_id] = task
        self.task_counter += 1
        
        # Store previous weights for importance calculation
        if self.strategy == LearningStrategy.ELASTIC_WEIGHT_CONSOLIDATION:
            # Calculate importance weights based on previous task
            if self.snapshots:
                importance_weights = self.importance_calculator.calculate_fisher_information(
                    self.current_weights,
                    training_data
                )
            else:
                importance_weights = None
        else:
            importance_weights = None
        
        # Train on new task
        performance = self._train_with_strategy(
            task,
            training_data,
            num_epochs,
            importance_weights
        )
        
        # Add samples to memory
        if self.strategy == LearningStrategy.MEMORY_REPLAY:
            self.memory_manager.add_samples(training_data, task.task_id)
        
        # Create snapshot
        snapshot = ModelSnapshot(
            snapshot_id="snapshot_{}".format(len(self.snapshots)),
            task_id=task.task_id,
            weights=self._copy_weights(self.current_weights),
            importance_weights=importance_weights,
            performance=performance,
            timestamp=datetime.now()
        )
        
        self.snapshots.append(snapshot)
        
        # Track performance
        if task.task_id not in self.task_performance:
            self.task_performance[task.task_id] = []
        self.task_performance[task.task_id].append(performance.get("accuracy", 0.0))
        
        return performance
    
    def evaluate_all_tasks(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on all learned tasks.
        
        Returns:
            Performance on each task
        """
        results = {}
        
        for task_id, task in self.tasks.items():
            # Simulate evaluation
            if self.task_performance.get(task_id):
                # Use last recorded performance
                performance = self.task_performance[task_id][-1]
            else:
                performance = random.uniform(0.6, 0.9)
            
            results[task_id] = {
                "accuracy": performance,
                "task_name": task.name
            }
        
        return results
    
    def detect_and_adapt(self,
                        new_data: List[Dict[str, Any]]) -> DriftDetection:
        """
        Detect concept drift and adapt if necessary.
        
        Args:
            new_data: New incoming data
            
        Returns:
            Drift detection result
        """
        # Evaluate on new data
        current_performance = random.uniform(0.5, 0.9)  # Simulated
        
        # Detect drift
        drift = self.drift_detector.detect_drift(current_performance)
        
        if drift.detected:
            # Adapt to drift
            self._adapt_to_drift(drift, new_data)
        
        return drift
    
    def calculate_forgetting(self) -> Dict[str, float]:
        """
        Calculate forgetting for each task.
        
        Returns:
            Forgetting metric for each task
        """
        forgetting = {}
        
        for task_id, performances in self.task_performance.items():
            if len(performances) > 1:
                # Compare first and last performance
                initial = performances[0]
                current = performances[-1]
                forgetting[task_id] = max(0, initial - current)
            else:
                forgetting[task_id] = 0.0
        
        return forgetting
    
    def get_learning_curve(self) -> List[Tuple[int, float]]:
        """
        Get learning curve across all tasks.
        
        Returns:
            List of (task_number, average_performance) tuples
        """
        curve = []
        
        for i, (task_id, task) in enumerate(self.tasks.items()):
            if self.task_performance.get(task_id):
                avg_perf = sum(
                    self.task_performance[task_id]
                ) / len(self.task_performance[task_id])
                curve.append((i + 1, avg_perf))
        
        return curve
    
    def _train_with_strategy(self,
                            task: LearningTask,
                            training_data: List[Dict[str, Any]],
                            num_epochs: int,
                            importance_weights: Optional[Dict[str, List[float]]]
                            ) -> Dict[str, float]:
        """Train using specific continuous learning strategy"""
        if self.strategy == LearningStrategy.ELASTIC_WEIGHT_CONSOLIDATION:
            return self._train_with_ewc(
                task,
                training_data,
                num_epochs,
                importance_weights
            )
        elif self.strategy == LearningStrategy.MEMORY_REPLAY:
            return self._train_with_replay(task, training_data, num_epochs)
        else:
            return self._train_standard(task, training_data, num_epochs)
    
    def _train_with_ewc(self,
                       task: LearningTask,
                       training_data: List[Dict[str, Any]],
                       num_epochs: int,
                       importance_weights: Optional[Dict[str, List[float]]]
                       ) -> Dict[str, float]:
        """Train with Elastic Weight Consolidation"""
        # Simulate training with EWC
        # In practice, would add regularization term based on importance weights
        
        for epoch in range(num_epochs):
            # Update weights with EWC penalty
            for layer_name, weights in self.current_weights.items():
                for i in range(len(weights)):
                    # Standard update
                    gradient = random.gauss(0, 0.01)
                    
                    # Add EWC penalty if we have importance weights
                    if importance_weights and layer_name in importance_weights:
                        ewc_penalty = (
                            importance_weights[layer_name][i] *
                            (weights[i] - self.snapshots[-1].weights[layer_name][i])
                        )
                        gradient += ewc_penalty * 0.1
                    
                    weights[i] -= gradient * 0.01
        
        # Return simulated performance
        return {
            "accuracy": random.uniform(0.75, 0.95),
            "loss": random.uniform(0.05, 0.3)
        }
    
    def _train_with_replay(self,
                          task: LearningTask,
                          training_data: List[Dict[str, Any]],
                          num_epochs: int) -> Dict[str, float]:
        """Train with experience replay"""
        for epoch in range(num_epochs):
            # Mix new data with replay data
            replay_batch = self.memory_manager.sample_batch(
                batch_size=min(32, len(self.memory_manager.buffer.samples))
            )
            
            # Update weights on mixed batch
            for layer_name, weights in self.current_weights.items():
                for i in range(len(weights)):
                    gradient = random.gauss(0, 0.01)
                    weights[i] -= gradient * 0.01
        
        return {
            "accuracy": random.uniform(0.75, 0.95),
            "loss": random.uniform(0.05, 0.3)
        }
    
    def _train_standard(self,
                       task: LearningTask,
                       training_data: List[Dict[str, Any]],
                       num_epochs: int) -> Dict[str, float]:
        """Standard training"""
        for epoch in range(num_epochs):
            for layer_name, weights in self.current_weights.items():
                for i in range(len(weights)):
                    gradient = random.gauss(0, 0.01)
                    weights[i] -= gradient * 0.01
        
        return {
            "accuracy": random.uniform(0.7, 0.9),
            "loss": random.uniform(0.1, 0.4)
        }
    
    def _adapt_to_drift(self,
                       drift: DriftDetection,
                       new_data: List[Dict[str, Any]]) -> None:
        """Adapt model to detected drift"""
        # Fine-tune on new data
        for layer_name, weights in self.current_weights.items():
            for i in range(len(weights)):
                # Quick adaptation
                gradient = random.gauss(0, 0.02)
                weights[i] -= gradient * 0.05
    
    def _copy_weights(self,
                     weights: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Create deep copy of weights"""
        return {
            layer_name: layer_weights.copy()
            for layer_name, layer_weights in weights.items()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        avg_forgetting = 0.0
        forgetting_dict = self.calculate_forgetting()
        
        if forgetting_dict:
            avg_forgetting = sum(forgetting_dict.values()) / len(forgetting_dict)
        
        return {
            "strategy": self.strategy.value,
            "total_tasks_learned": len(self.tasks),
            "total_snapshots": len(self.snapshots),
            "memory_statistics": self.memory_manager.get_statistics(),
            "avg_forgetting": avg_forgetting,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "name": task.name,
                    "data_size": task.data_size
                }
                for task in self.tasks.values()
            ]
        }


def demonstrate_continuous_learning():
    """Demonstrate continuous learning agent"""
    print("=" * 70)
    print("Continuous Learning Agent Demonstration")
    print("=" * 70)
    
    # Initialize agent with EWC
    agent = ContinuousLearningAgent(
        strategy=LearningStrategy.ELASTIC_WEIGHT_CONSOLIDATION,
        memory_size=500
    )
    
    # Example 1: Initialize model
    print("\n1. Initializing Model:")
    agent.initialize_model(layer_sizes=[10, 20, 10])
    print("Model initialized with {} layers".format(
        len(agent.current_weights)
    ))
    
    # Example 2: Learn multiple tasks sequentially
    print("\n2. Learning Multiple Tasks Sequentially:")
    
    tasks = [
        ("Task A: Image Classification", 200),
        ("Task B: Sentiment Analysis", 150),
        ("Task C: Object Detection", 180),
        ("Task D: Text Generation", 220)
    ]
    
    for task_name, data_size in tasks:
        # Generate dummy training data
        training_data = [
            {"features": [random.random() for _ in range(10)], "label": i % 3}
            for i in range(data_size)
        ]
        
        print("\n  Learning: {}".format(task_name))
        performance = agent.learn_task(task_name, training_data, num_epochs=5)
        print("    Accuracy: {:.2%}".format(performance["accuracy"]))
        print("    Loss: {:.4f}".format(performance["loss"]))
    
    # Example 3: Evaluate on all tasks
    print("\n3. Evaluating on All Learned Tasks:")
    
    all_performance = agent.evaluate_all_tasks()
    for task_id, perf in all_performance.items():
        print("  {}: {:.2%} accuracy".format(
            perf["task_name"],
            perf["accuracy"]
        ))
    
    # Example 4: Calculate forgetting
    print("\n4. Forgetting Analysis:")
    
    forgetting = agent.calculate_forgetting()
    for task_id, forget_amount in forgetting.items():
        task_name = agent.tasks[task_id].name
        print("  {}: {:.2%} forgetting".format(task_name, forget_amount))
    
    # Example 5: Detect concept drift
    print("\n5. Concept Drift Detection:")
    
    new_data = [
        {"features": [random.random() for _ in range(10)], "label": 0}
        for _ in range(50)
    ]
    
    drift = agent.detect_and_adapt(new_data)
    print("  Drift detected: {}".format(drift.detected))
    if drift.detected:
        print("  Drift type: {}".format(drift.drift_type.value))
        print("  Magnitude: {:.4f}".format(drift.magnitude))
    
    # Example 6: Learning curve
    print("\n6. Learning Curve:")
    
    curve = agent.get_learning_curve()
    for task_num, avg_perf in curve:
        print("  After task {}: {:.2%} average performance".format(
            task_num,
            avg_perf
        ))
    
    # Example 7: Compare strategies
    print("\n7. Comparing Learning Strategies:")
    
    strategies = [
        LearningStrategy.ELASTIC_WEIGHT_CONSOLIDATION,
        LearningStrategy.MEMORY_REPLAY,
        LearningStrategy.INCREMENTAL
    ]
    
    for strategy in strategies:
        agent_test = ContinuousLearningAgent(
            strategy=strategy,
            memory_size=300
        )
        
        agent_test.initialize_model([5, 10, 5])
        
        # Learn 3 tasks
        for i in range(3):
            data = [
                {"features": [random.random() for _ in range(5)], "label": 0}
                for _ in range(100)
            ]
            agent_test.learn_task("Task {}".format(i + 1), data, num_epochs=3)
        
        forgetting = agent_test.calculate_forgetting()
        avg_forget = sum(forgetting.values()) / len(forgetting) if forgetting else 0.0
        
        print("\n  Strategy: {}".format(strategy.value))
        print("    Average forgetting: {:.2%}".format(avg_forget))
    
    # Example 8: Statistics
    print("\n8. Agent Statistics:")
    stats = agent.get_statistics()
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    demonstrate_continuous_learning()
