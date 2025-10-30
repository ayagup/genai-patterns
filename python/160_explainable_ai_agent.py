"""
Meta-Learning Agent Pattern

Learns how to learn - adapts learning strategies based on task performance.
Implements few-shot learning and rapid adaptation capabilities.

Use Cases:
- Few-shot learning tasks
- Rapid adaptation scenarios
- Transfer learning
- Continual learning systems

Advantages:
- Fast adaptation to new tasks
- Efficient learning from few examples
- Knowledge transfer across tasks
- Self-improving capabilities
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import random


class LearningStrategy(Enum):
    """Learning strategies"""
    GRADIENT_BASED = "gradient_based"
    METRIC_BASED = "metric_based"
    MODEL_BASED = "model_based"
    OPTIMIZATION_BASED = "optimization_based"


class TaskType(Enum):
    """Types of tasks"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    GENERATION = "generation"
    REINFORCEMENT = "reinforcement"


@dataclass
class Task:
    """Learning task"""
    task_id: str
    task_type: TaskType
    name: str
    description: str
    support_examples: List[Dict[str, Any]]  # Few-shot examples
    query_examples: List[Dict[str, Any]]  # Test examples
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningExperience:
    """Record of learning experience"""
    experience_id: str
    task_id: str
    strategy: LearningStrategy
    num_examples: int
    performance: float
    adaptation_time: float
    timestamp: datetime
    context: Dict[str, Any]


@dataclass
class MetaKnowledge:
    """Meta-level knowledge about learning"""
    strategy_performance: Dict[str, List[float]]
    task_similarities: Dict[Tuple[str, str], float]
    optimal_strategies: Dict[TaskType, LearningStrategy]
    learning_curves: Dict[str, List[Tuple[int, float]]]


class FewShotLearner:
    """Learns from few examples"""
    
    def __init__(self, strategy: LearningStrategy):
        self.strategy = strategy
        self.learned_patterns: Dict[str, Any] = {}
    
    def learn(self,
             support_examples: List[Dict[str, Any]],
             task_type: TaskType) -> Dict[str, Any]:
        """
        Learn from support examples.
        
        Args:
            support_examples: Few-shot examples
            task_type: Type of task
            
        Returns:
            Learned model/patterns
        """
        if self.strategy == LearningStrategy.METRIC_BASED:
            return self._metric_based_learning(support_examples, task_type)
        elif self.strategy == LearningStrategy.MODEL_BASED:
            return self._model_based_learning(support_examples, task_type)
        elif self.strategy == LearningStrategy.OPTIMIZATION_BASED:
            return self._optimization_based_learning(support_examples, task_type)
        else:
            return self._gradient_based_learning(support_examples, task_type)
    
    def predict(self,
                query: Dict[str, Any],
                learned_model: Dict[str, Any]) -> Any:
        """
        Make prediction using learned model.
        
        Args:
            query: Query to predict
            learned_model: Model learned from support set
            
        Returns:
            Prediction
        """
        if self.strategy == LearningStrategy.METRIC_BASED:
            return self._metric_based_prediction(query, learned_model)
        else:
            return self._default_prediction(query, learned_model)
    
    def _metric_based_learning(self,
                              support_examples: List[Dict[str, Any]],
                              task_type: TaskType) -> Dict[str, Any]:
        """Learn using metric-based approach (e.g., prototypical networks)"""
        # Compute prototypes for each class
        class_prototypes = {}
        
        for example in support_examples:
            label = example.get("label")
            features = example.get("features", [])
            
            if label not in class_prototypes:
                class_prototypes[label] = []
            class_prototypes[label].append(features)
        
        # Average features for each class
        prototypes = {}
        for label, feature_list in class_prototypes.items():
            if feature_list:
                # Simplified: average of features
                avg_features = [
                    sum(f[i] for f in feature_list) / len(feature_list)
                    for i in range(len(feature_list[0]))
                ]
                prototypes[label] = avg_features
        
        return {
            "strategy": "metric_based",
            "prototypes": prototypes,
            "num_classes": len(prototypes)
        }
    
    def _model_based_learning(self,
                             support_examples: List[Dict[str, Any]],
                             task_type: TaskType) -> Dict[str, Any]:
        """Learn using model-based approach"""
        # Build simple model from examples
        model = {
            "strategy": "model_based",
            "examples": support_examples,
            "task_type": task_type.value
        }
        return model
    
    def _optimization_based_learning(self,
                                    support_examples: List[Dict[str, Any]],
                                    task_type: TaskType) -> Dict[str, Any]:
        """Learn using optimization-based approach (e.g., MAML)"""
        # Simulate optimization steps
        model = {
            "strategy": "optimization_based",
            "parameters": {"learned": True},
            "num_steps": len(support_examples)
        }
        return model
    
    def _gradient_based_learning(self,
                                support_examples: List[Dict[str, Any]],
                                task_type: TaskType) -> Dict[str, Any]:
        """Learn using gradient-based approach"""
        model = {
            "strategy": "gradient_based",
            "weights": [random.random() for _ in range(10)]
        }
        return model
    
    def _metric_based_prediction(self,
                                query: Dict[str, Any],
                                learned_model: Dict[str, Any]) -> Any:
        """Make prediction using metric-based model"""
        prototypes = learned_model["prototypes"]
        query_features = query.get("features", [])
        
        if not query_features or not prototypes:
            return None
        
        # Find nearest prototype
        min_distance = float('inf')
        nearest_label = None
        
        for label, prototype in prototypes.items():
            # Euclidean distance
            distance = sum(
                (q - p) ** 2
                for q, p in zip(query_features, prototype)
            ) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                nearest_label = label
        
        return {
            "prediction": nearest_label,
            "confidence": 1.0 / (1.0 + min_distance)
        }
    
    def _default_prediction(self,
                           query: Dict[str, Any],
                           learned_model: Dict[str, Any]) -> Any:
        """Default prediction method"""
        return {"prediction": "default", "confidence": 0.5}


class StrategySelector:
    """Selects optimal learning strategy"""
    
    def __init__(self):
        self.strategy_scores: Dict[LearningStrategy, float] = {
            strategy: 0.5 for strategy in LearningStrategy
        }
    
    def select_strategy(self,
                       task: Task,
                       meta_knowledge: MetaKnowledge) -> LearningStrategy:
        """
        Select best strategy for task.
        
        Args:
            task: Task to learn
            meta_knowledge: Meta-level knowledge
            
        Returns:
            Selected learning strategy
        """
        # Check if we have optimal strategy for this task type
        if task.task_type in meta_knowledge.optimal_strategies:
            return meta_knowledge.optimal_strategies[task.task_type]
        
        # Select based on strategy performance
        best_strategy = max(
            self.strategy_scores.items(),
            key=lambda x: x[1]
        )[0]
        
        return best_strategy
    
    def update_strategy_scores(self,
                              strategy: LearningStrategy,
                              performance: float) -> None:
        """Update strategy scores based on performance"""
        # Update score with moving average
        current_score = self.strategy_scores[strategy]
        self.strategy_scores[strategy] = current_score * 0.8 + performance * 0.2


class MetaLearningAgent:
    """
    Agent that learns how to learn.
    Adapts learning strategies based on task characteristics and performance.
    """
    
    def __init__(self):
        self.meta_knowledge = MetaKnowledge(
            strategy_performance={},
            task_similarities={},
            optimal_strategies={},
            learning_curves={}
        )
        
        self.strategy_selector = StrategySelector()
        self.learning_experiences: List[LearningExperience] = []
        self.task_history: Dict[str, Task] = {}
    
    def learn_task(self,
                   task: Task,
                   strategy: Optional[LearningStrategy] = None) -> Tuple[Dict[str, Any], float]:
        """
        Learn a new task using meta-learning.
        
        Args:
            task: Task to learn
            strategy: Optional strategy (auto-selected if None)
            
        Returns:
            (learned_model, performance) tuple
        """
        start_time = datetime.now()
        
        # Select strategy if not provided
        if strategy is None:
            strategy = self.strategy_selector.select_strategy(
                task,
                self.meta_knowledge
            )
        
        # Create learner with selected strategy
        learner = FewShotLearner(strategy)
        
        # Learn from support examples
        learned_model = learner.learn(
            task.support_examples,
            task.task_type
        )
        
        # Evaluate on query examples
        performance = self._evaluate_model(
            learner,
            learned_model,
            task.query_examples
        )
        
        # Calculate adaptation time
        adaptation_time = (datetime.now() - start_time).total_seconds()
        
        # Record experience
        experience = LearningExperience(
            experience_id="exp_{}".format(len(self.learning_experiences)),
            task_id=task.task_id,
            strategy=strategy,
            num_examples=len(task.support_examples),
            performance=performance,
            adaptation_time=adaptation_time,
            timestamp=datetime.now(),
            context=task.metadata
        )
        
        self.learning_experiences.append(experience)
        self.task_history[task.task_id] = task
        
        # Update meta-knowledge
        self._update_meta_knowledge(task, strategy, performance)
        
        # Update strategy scores
        self.strategy_selector.update_strategy_scores(strategy, performance)
        
        return learned_model, performance
    
    def transfer_knowledge(self,
                          source_task_id: str,
                          target_task: Task) -> float:
        """
        Transfer knowledge from source task to target task.
        
        Args:
            source_task_id: Source task ID
            target_task: Target task
            
        Returns:
            Performance improvement from transfer
        """
        source_task = self.task_history.get(source_task_id)
        if not source_task:
            return 0.0
        
        # Calculate task similarity
        similarity = self._compute_task_similarity(source_task, target_task)
        
        # Store similarity
        self.meta_knowledge.task_similarities[
            (source_task_id, target_task.task_id)
        ] = similarity
        
        # If tasks are similar, use same strategy
        if similarity > 0.7:
            source_experiences = [
                exp for exp in self.learning_experiences
                if exp.task_id == source_task_id
            ]
            
            if source_experiences:
                best_experience = max(
                    source_experiences,
                    key=lambda e: e.performance
                )
                
                # Use same strategy for target task
                model, performance = self.learn_task(
                    target_task,
                    strategy=best_experience.strategy
                )
                
                # Calculate improvement (simplified)
                baseline_performance = 0.5  # Random guess
                improvement = performance - baseline_performance
                
                return improvement
        
        return 0.0
    
    def get_optimal_strategy(self, task_type: TaskType) -> LearningStrategy:
        """Get optimal strategy for task type"""
        if task_type in self.meta_knowledge.optimal_strategies:
            return self.meta_knowledge.optimal_strategies[task_type]
        
        return max(
            self.strategy_selector.strategy_scores.items(),
            key=lambda x: x[1]
        )[0]
    
    def get_learning_curve(self, task_id: str) -> List[Tuple[int, float]]:
        """Get learning curve for a task"""
        return self.meta_knowledge.learning_curves.get(task_id, [])
    
    def _evaluate_model(self,
                       learner: FewShotLearner,
                       model: Dict[str, Any],
                       query_examples: List[Dict[str, Any]]) -> float:
        """Evaluate learned model"""
        if not query_examples:
            return 0.5
        
        correct = 0
        total = len(query_examples)
        
        for example in query_examples:
            prediction = learner.predict(example, model)
            
            if prediction and isinstance(prediction, dict):
                pred_label = prediction.get("prediction")
                true_label = example.get("label")
                
                if pred_label == true_label:
                    correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def _update_meta_knowledge(self,
                              task: Task,
                              strategy: LearningStrategy,
                              performance: float) -> None:
        """Update meta-level knowledge"""
        # Update strategy performance
        if strategy.value not in self.meta_knowledge.strategy_performance:
            self.meta_knowledge.strategy_performance[strategy.value] = []
        
        self.meta_knowledge.strategy_performance[strategy.value].append(
            performance
        )
        
        # Update optimal strategies for task type
        avg_performance = sum(
            self.meta_knowledge.strategy_performance[strategy.value]
        ) / len(self.meta_knowledge.strategy_performance[strategy.value])
        
        current_best = self.meta_knowledge.optimal_strategies.get(task.task_type)
        if (not current_best or
            avg_performance > self.strategy_selector.strategy_scores.get(
                current_best, 0.0
            )):
            self.meta_knowledge.optimal_strategies[task.task_type] = strategy
        
        # Update learning curve
        if task.task_id not in self.meta_knowledge.learning_curves:
            self.meta_knowledge.learning_curves[task.task_id] = []
        
        self.meta_knowledge.learning_curves[task.task_id].append(
            (len(task.support_examples), performance)
        )
    
    def _compute_task_similarity(self, task1: Task, task2: Task) -> float:
        """Compute similarity between tasks"""
        similarity = 0.0
        
        # Task type similarity
        if task1.task_type == task2.task_type:
            similarity += 0.4
        
        # Example similarity (simplified)
        if task1.support_examples and task2.support_examples:
            # Check if examples have similar structure
            keys1 = set(task1.support_examples[0].keys())
            keys2 = set(task2.support_examples[0].keys())
            
            key_overlap = len(keys1.intersection(keys2)) / len(keys1.union(keys2))
            similarity += key_overlap * 0.3
        
        # Description similarity (simplified)
        desc1_words = set(task1.description.lower().split())
        desc2_words = set(task2.description.lower().split())
        
        if desc1_words and desc2_words:
            desc_overlap = len(desc1_words.intersection(desc2_words)) / len(
                desc1_words.union(desc2_words)
            )
            similarity += desc_overlap * 0.3
        
        return similarity
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get meta-learning statistics"""
        strategy_avg_performance = {}
        for strategy, performances in self.meta_knowledge.strategy_performance.items():
            if performances:
                strategy_avg_performance[strategy] = sum(performances) / len(performances)
        
        return {
            "total_tasks_learned": len(self.task_history),
            "total_experiences": len(self.learning_experiences),
            "strategy_performance": strategy_avg_performance,
            "optimal_strategies": {
                task_type.value: strategy.value
                for task_type, strategy in self.meta_knowledge.optimal_strategies.items()
            },
            "task_similarities_computed": len(self.meta_knowledge.task_similarities)
        }


def demonstrate_meta_learning():
    """Demonstrate meta-learning agent"""
    print("=" * 70)
    print("Meta-Learning Agent Demonstration")
    print("=" * 70)
    
    agent = MetaLearningAgent()
    
    # Example 1: Create and learn a classification task
    print("\n1. Learning Classification Task (5-way 1-shot):")
    
    task1 = Task(
        task_id="task_1",
        task_type=TaskType.CLASSIFICATION,
        name="Image Classification",
        description="Classify images into 5 categories",
        support_examples=[
            {"features": [0.1, 0.2, 0.3], "label": "cat"},
            {"features": [0.2, 0.3, 0.4], "label": "dog"},
            {"features": [0.3, 0.4, 0.5], "label": "bird"},
            {"features": [0.4, 0.5, 0.6], "label": "fish"},
            {"features": [0.5, 0.6, 0.7], "label": "rabbit"}
        ],
        query_examples=[
            {"features": [0.15, 0.25, 0.35], "label": "cat"},
            {"features": [0.25, 0.35, 0.45], "label": "dog"}
        ]
    )
    
    model1, performance1 = agent.learn_task(task1)
    print("Task: {}".format(task1.name))
    print("Strategy used: {}".format(model1.get("strategy", "unknown")))
    print("Performance: {:.2%}".format(performance1))
    
    # Example 2: Learn similar task
    print("\n2. Learning Similar Task:")
    
    task2 = Task(
        task_id="task_2",
        task_type=TaskType.CLASSIFICATION,
        name="Animal Classification",
        description="Classify animals into categories",
        support_examples=[
            {"features": [0.12, 0.22, 0.32], "label": "mammal"},
            {"features": [0.22, 0.32, 0.42], "label": "reptile"},
            {"features": [0.32, 0.42, 0.52], "label": "bird"}
        ],
        query_examples=[
            {"features": [0.14, 0.24, 0.34], "label": "mammal"}
        ]
    )
    
    model2, performance2 = agent.learn_task(task2)
    print("Task: {}".format(task2.name))
    print("Performance: {:.2%}".format(performance2))
    
    # Example 3: Transfer learning
    print("\n3. Transfer Learning:")
    improvement = agent.transfer_knowledge("task_1", task2)
    print("Performance improvement from transfer: {:.2%}".format(improvement))
    
    # Example 4: Different task type
    print("\n4. Learning Different Task Type (Regression):")
    
    task3 = Task(
        task_id="task_3",
        task_type=TaskType.REGRESSION,
        name="Value Prediction",
        description="Predict continuous values",
        support_examples=[
            {"features": [1.0, 2.0], "label": 3.0},
            {"features": [2.0, 3.0], "label": 5.0}
        ],
        query_examples=[
            {"features": [1.5, 2.5], "label": 4.0}
        ]
    )
    
    model3, performance3 = agent.learn_task(task3)
    print("Task: {}".format(task3.name))
    print("Task type: {}".format(task3.task_type.value))
    print("Performance: {:.2%}".format(performance3))
    
    # Example 5: Optimal strategies
    print("\n5. Optimal Strategies by Task Type:")
    for task_type in TaskType:
        strategy = agent.get_optimal_strategy(task_type)
        print("  {}: {}".format(task_type.value, strategy.value))
    
    # Example 6: Learning curves
    print("\n6. Learning Curves:")
    for task_id in ["task_1", "task_2"]:
        curve = agent.get_learning_curve(task_id)
        if curve:
            print("\n  Task {}:".format(task_id))
            for num_examples, perf in curve:
                print("    {} examples: {:.2%}".format(num_examples, perf))
    
    # Example 7: Statistics
    print("\n7. Meta-Learning Statistics:")
    stats = agent.get_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    demonstrate_meta_learning()
