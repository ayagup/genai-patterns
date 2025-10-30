"""
Pattern 115: Few-Shot Adaptation Agent

This pattern demonstrates few-shot learning where the agent rapidly adapts to
new tasks using only a few examples. It uses meta-learning techniques like
prototype-based learning and metric learning.

🎯 MILESTONE: Pattern 115/170 (67.6% Complete!)

Key concepts:
- Few-shot learning (learn from few examples)
- Meta-learning (learning to learn)
- Prototype-based classification
- Support set and query set
- Rapid task adaptation

Use cases:
- Quick model customization
- Personalization with limited data
- Rare event classification
- New category learning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
import time
import uuid
import math
import random


class TaskType(Enum):
    """Types of few-shot tasks"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEQUENCE = "sequence"
    GENERATION = "generation"


class AdaptationStrategy(Enum):
    """Few-shot adaptation strategies"""
    PROTOTYPICAL = "prototypical"
    MATCHING = "matching"
    RELATION = "relation"
    MAML = "maml"


@dataclass
class Example:
    """A single training example"""
    features: Dict[str, float]
    label: Any
    task_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class SupportSet:
    """Support set for few-shot learning"""
    task_id: str
    examples: List[Example]
    task_type: TaskType
    num_classes: int
    shots_per_class: int
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class Prototype:
    """Class prototype in embedding space"""
    class_label: Any
    embedding: Dict[str, float]
    support_examples: List[str]  # Example IDs
    confidence: float = 1.0
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class TaskModel:
    """Adapted model for a specific task"""
    task_id: str
    task_type: TaskType
    prototypes: List[Prototype]
    created_at: float
    num_adaptations: int = 0
    performance: Optional[float] = None
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


class EmbeddingNetwork:
    """Neural network that embeds examples into feature space"""
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.weights: Dict[str, List[float]] = {}
        self.update_count: int = 0
    
    def embed(self, features: Dict[str, float]) -> Dict[str, float]:
        """Embed features into embedding space"""
        embedding = {}
        
        # Simple transformation (in practice, would be a neural network)
        for i in range(self.embedding_dim):
            embed_key = f"e{i}"
            value = 0.0
            
            for feat_name, feat_val in features.items():
                # Hash-based pseudo-random projection
                weight = self._get_weight(feat_name, i)
                value += feat_val * weight
            
            # Apply activation (tanh)
            embedding[embed_key] = math.tanh(value)
        
        return embedding
    
    def _get_weight(self, feature_name: str, embed_idx: int) -> float:
        """Get or initialize weight"""
        key = f"{feature_name}_{embed_idx}"
        
        if key not in self.weights:
            # Initialize with small random values
            random.seed(hash(key) % (2**32))
            self.weights[key] = [random.gauss(0, 0.1) for _ in range(1)]
        
        return self.weights[key][0]
    
    def adapt(self, support_set: SupportSet):
        """Adapt embedding network to support set (simplified)"""
        # In practice, would fine-tune the network
        # Here we just increment counter
        self.update_count += 1


class PrototypicalLearner:
    """Prototypical networks for few-shot learning"""
    
    def __init__(self, embedding_network: EmbeddingNetwork):
        self.embedding_network = embedding_network
        self.prototypes: Dict[str, Prototype] = {}
    
    def compute_prototype(self, examples: List[Example],
                         class_label: Any) -> Prototype:
        """Compute prototype for a class"""
        if not examples:
            raise ValueError("Need at least one example to compute prototype")
        
        # Embed all examples
        embeddings = [self.embedding_network.embed(ex.features) for ex in examples]
        
        # Average embeddings to get prototype
        prototype_embedding = {}
        for key in embeddings[0].keys():
            values = [emb[key] for emb in embeddings]
            prototype_embedding[key] = sum(values) / len(values)
        
        # Create prototype
        prototype = Prototype(
            class_label=class_label,
            embedding=prototype_embedding,
            support_examples=[ex.id for ex in examples],
            confidence=1.0 / math.sqrt(len(examples))  # More examples = higher confidence
        )
        
        return prototype
    
    def classify(self, query_features: Dict[str, float],
                prototypes: List[Prototype]) -> Tuple[Any, float, Dict[Any, float]]:
        """Classify query using prototypes"""
        # Embed query
        query_embedding = self.embedding_network.embed(query_features)
        
        # Compute distances to all prototypes
        distances = {}
        for prototype in prototypes:
            dist = self._euclidean_distance(query_embedding, prototype.embedding)
            distances[prototype.class_label] = dist
        
        # Find nearest prototype
        nearest_class = min(distances.keys(), key=lambda k: distances[k])
        nearest_dist = distances[nearest_class]
        
        # Convert distance to confidence (closer = higher confidence)
        max_dist = max(distances.values())
        confidence = 1.0 - (nearest_dist / max_dist) if max_dist > 0 else 1.0
        
        return nearest_class, confidence, distances
    
    def _euclidean_distance(self, emb1: Dict[str, float],
                           emb2: Dict[str, float]) -> float:
        """Compute Euclidean distance between embeddings"""
        distance = 0.0
        
        for key in emb1.keys():
            if key in emb2:
                distance += (emb1[key] - emb2[key]) ** 2
        
        return math.sqrt(distance)


class MetaLearner:
    """Meta-learner that learns to learn from few examples"""
    
    def __init__(self):
        self.task_history: List[SupportSet] = []
        self.adaptation_history: List[Dict[str, Any]] = []
        self.task_performance: Dict[str, List[float]] = {}
    
    def record_adaptation(self, task_id: str, support_set: SupportSet,
                         performance: float):
        """Record task adaptation"""
        self.task_history.append(support_set)
        
        self.adaptation_history.append({
            "task_id": task_id,
            "num_classes": support_set.num_classes,
            "shots_per_class": support_set.shots_per_class,
            "performance": performance,
            "timestamp": time.time()
        })
        
        if task_id not in self.task_performance:
            self.task_performance[task_id] = []
        self.task_performance[task_id].append(performance)
    
    def get_learning_curve(self, task_id: str) -> List[float]:
        """Get learning curve for a task"""
        return self.task_performance.get(task_id, [])
    
    def predict_adaptation_quality(self, num_classes: int,
                                  shots_per_class: int) -> float:
        """Predict how well adaptation will work"""
        # Based on historical performance
        similar_tasks = [
            h for h in self.adaptation_history
            if h["num_classes"] == num_classes and h["shots_per_class"] == shots_per_class
        ]
        
        if similar_tasks:
            avg_perf = sum(h["performance"] for h in similar_tasks) / len(similar_tasks)
            return avg_perf
        else:
            # Estimate: more classes and fewer shots = harder
            difficulty = (num_classes / 10.0) / (shots_per_class / 5.0)
            return max(0.3, 1.0 - difficulty)
    
    def get_meta_statistics(self) -> Dict[str, Any]:
        """Get meta-learning statistics"""
        if not self.adaptation_history:
            return {}
        
        return {
            "total_tasks": len(set(h["task_id"] for h in self.adaptation_history)),
            "total_adaptations": len(self.adaptation_history),
            "avg_performance": sum(h["performance"] for h in self.adaptation_history) / len(self.adaptation_history),
            "best_performance": max(h["performance"] for h in self.adaptation_history),
            "tasks_learned": list(self.task_performance.keys())
        }


class FewShotAdaptationAgent:
    """
    Complete few-shot adaptation agent that rapidly learns new tasks from
    a few examples using meta-learning and prototypical networks.
    
    🎯 PATTERN 115 - MILESTONE: 115/170 patterns (67.6% complete)!
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_network = EmbeddingNetwork(embedding_dim)
        self.prototypical_learner = PrototypicalLearner(self.embedding_network)
        self.meta_learner = MetaLearner()
        self.task_models: Dict[str, TaskModel] = {}
    
    def adapt_to_task(self, task_id: str, support_set: SupportSet) -> TaskModel:
        """Adapt to a new task using support set"""
        
        # Adapt embedding network
        self.embedding_network.adapt(support_set)
        
        # Group examples by class
        examples_by_class: Dict[Any, List[Example]] = {}
        for example in support_set.examples:
            if example.label not in examples_by_class:
                examples_by_class[example.label] = []
            examples_by_class[example.label].append(example)
        
        # Compute prototype for each class
        prototypes = []
        for class_label, examples in examples_by_class.items():
            prototype = self.prototypical_learner.compute_prototype(examples, class_label)
            prototypes.append(prototype)
        
        # Create task model
        task_model = TaskModel(
            task_id=task_id,
            task_type=support_set.task_type,
            prototypes=prototypes,
            created_at=time.time(),
            num_adaptations=1
        )
        
        self.task_models[task_id] = task_model
        
        return task_model
    
    def predict(self, task_id: str, query_features: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction for a query"""
        if task_id not in self.task_models:
            raise ValueError(f"Task {task_id} not found. Need to adapt first.")
        
        task_model = self.task_models[task_id]
        
        # Classify using prototypes
        predicted_class, confidence, distances = self.prototypical_learner.classify(
            query_features,
            task_model.prototypes
        )
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_distances": distances,
            "task_id": task_id
        }
    
    def evaluate_task(self, task_id: str, query_set: List[Example]) -> Dict[str, Any]:
        """Evaluate adapted model on query set"""
        if task_id not in self.task_models:
            raise ValueError(f"Task {task_id} not found")
        
        correct = 0
        predictions = []
        
        for example in query_set:
            result = self.predict(task_id, example.features)
            is_correct = (result["predicted_class"] == example.label)
            
            predictions.append({
                "predicted": result["predicted_class"],
                "actual": example.label,
                "correct": is_correct,
                "confidence": result["confidence"]
            })
            
            if is_correct:
                correct += 1
        
        accuracy = correct / len(query_set) if query_set else 0.0
        
        # Update task model performance
        self.task_models[task_id].performance = accuracy
        
        # Record in meta-learner
        support_set = SupportSet(
            task_id=task_id,
            examples=[],  # Simplified
            task_type=self.task_models[task_id].task_type,
            num_classes=len(self.task_models[task_id].prototypes),
            shots_per_class=1  # Simplified
        )
        self.meta_learner.record_adaptation(task_id, support_set, accuracy)
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(query_set),
            "predictions": predictions
        }
    
    def quick_adapt(self, task_id: str, new_examples: List[Example]) -> Dict[str, Any]:
        """Quickly adapt existing model with new examples"""
        if task_id not in self.task_models:
            raise ValueError(f"Task {task_id} not found")
        
        task_model = self.task_models[task_id]
        
        # Group new examples by class
        examples_by_class: Dict[Any, List[Example]] = {}
        for example in new_examples:
            if example.label not in examples_by_class:
                examples_by_class[example.label] = []
            examples_by_class[example.label].append(example)
        
        # Update or create prototypes
        updated_count = 0
        for class_label, examples in examples_by_class.items():
            # Find existing prototype
            existing_prototype = None
            for proto in task_model.prototypes:
                if proto.class_label == class_label:
                    existing_prototype = proto
                    break
            
            if existing_prototype:
                # Update prototype (simple average)
                new_prototype = self.prototypical_learner.compute_prototype(examples, class_label)
                
                # Average with existing
                for key in existing_prototype.embedding.keys():
                    if key in new_prototype.embedding:
                        existing_prototype.embedding[key] = (
                            existing_prototype.embedding[key] * 0.5 +
                            new_prototype.embedding[key] * 0.5
                        )
                updated_count += 1
            else:
                # Create new prototype
                new_prototype = self.prototypical_learner.compute_prototype(examples, class_label)
                task_model.prototypes.append(new_prototype)
                updated_count += 1
        
        task_model.num_adaptations += 1
        
        return {
            "task_id": task_id,
            "prototypes_updated": updated_count,
            "total_prototypes": len(task_model.prototypes),
            "adaptations": task_model.num_adaptations
        }
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get comprehensive agent summary"""
        return {
            "tasks_learned": len(self.task_models),
            "embedding_updates": self.embedding_network.update_count,
            "meta_statistics": self.meta_learner.get_meta_statistics(),
            "task_details": {
                task_id: {
                    "num_classes": len(model.prototypes),
                    "adaptations": model.num_adaptations,
                    "performance": model.performance
                }
                for task_id, model in self.task_models.items()
            }
        }


# Demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("🎯 PATTERN 115: FEW-SHOT ADAPTATION AGENT 🎯")
    print("MILESTONE: 115/170 patterns (67.6% complete)!")
    print("Demonstration of rapid task adaptation from few examples")
    print("=" * 80)
    
    # Create agent
    agent = FewShotAdaptationAgent(embedding_dim=32)
    
    print("\n1. Task 1: Sentiment Classification (3-way, 5-shot)")
    print("-" * 40)
    
    # Create support set for Task 1
    task1_support = [
        Example({"word_positive": 0.9, "word_negative": 0.1, "length": 10}, "positive", "task1"),
        Example({"word_positive": 0.85, "word_negative": 0.15, "length": 12}, "positive", "task1"),
        Example({"word_positive": 0.88, "word_negative": 0.12, "length": 11}, "positive", "task1"),
        Example({"word_positive": 0.82, "word_negative": 0.18, "length": 9}, "positive", "task1"),
        Example({"word_positive": 0.91, "word_negative": 0.09, "length": 13}, "positive", "task1"),
        
        Example({"word_positive": 0.1, "word_negative": 0.9, "length": 8}, "negative", "task1"),
        Example({"word_positive": 0.15, "word_negative": 0.85, "length": 10}, "negative", "task1"),
        Example({"word_positive": 0.12, "word_negative": 0.88, "length": 9}, "negative", "task1"),
        Example({"word_positive": 0.18, "word_negative": 0.82, "length": 11}, "negative", "task1"),
        Example({"word_positive": 0.08, "word_negative": 0.92, "length": 7}, "negative", "task1"),
        
        Example({"word_positive": 0.5, "word_negative": 0.5, "length": 15}, "neutral", "task1"),
        Example({"word_positive": 0.48, "word_negative": 0.52, "length": 14}, "neutral", "task1"),
        Example({"word_positive": 0.52, "word_negative": 0.48, "length": 16}, "neutral", "task1"),
        Example({"word_positive": 0.49, "word_negative": 0.51, "length": 13}, "neutral", "task1"),
        Example({"word_positive": 0.51, "word_negative": 0.49, "length": 15}, "neutral", "task1"),
    ]
    
    support_set1 = SupportSet(
        task_id="sentiment",
        examples=task1_support,
        task_type=TaskType.CLASSIFICATION,
        num_classes=3,
        shots_per_class=5
    )
    
    # Adapt to Task 1
    model1 = agent.adapt_to_task("sentiment", support_set1)
    print(f"✓ Adapted to sentiment task")
    print(f"  Classes: {len(model1.prototypes)}")
    print(f"  Prototypes computed for: {', '.join(str(p.class_label) for p in model1.prototypes)}")
    
    # Test on query examples
    query1 = [
        Example({"word_positive": 0.87, "word_negative": 0.13, "length": 10}, "positive", "task1"),
        Example({"word_positive": 0.14, "word_negative": 0.86, "length": 9}, "negative", "task1"),
        Example({"word_positive": 0.50, "word_negative": 0.50, "length": 14}, "neutral", "task1"),
        Example({"word_positive": 0.90, "word_negative": 0.10, "length": 11}, "positive", "task1"),
        Example({"word_positive": 0.11, "word_negative": 0.89, "length": 8}, "negative", "task1"),
    ]
    
    eval1 = agent.evaluate_task("sentiment", query1)
    print(f"\n  Evaluation: {eval1['accuracy']:.1%} accuracy ({eval1['correct']}/{eval1['total']} correct)")
    for i, pred in enumerate(eval1['predictions'][:3], 1):
        print(f"    Query {i}: predicted={pred['predicted']}, actual={pred['actual']}, "
              f"{'✓' if pred['correct'] else '✗'} (conf={pred['confidence']:.2f})")
    
    # Task 2: Topic Classification (2-way, 3-shot)
    print("\n2. Task 2: Topic Classification (2-way, 3-shot)")
    print("-" * 40)
    
    task2_support = [
        Example({"tech_words": 0.9, "sports_words": 0.1, "length": 20}, "tech", "task2"),
        Example({"tech_words": 0.85, "sports_words": 0.15, "length": 22}, "tech", "task2"),
        Example({"tech_words": 0.88, "sports_words": 0.12, "length": 21}, "tech", "task2"),
        
        Example({"tech_words": 0.1, "sports_words": 0.9, "length": 18}, "sports", "task2"),
        Example({"tech_words": 0.15, "sports_words": 0.85, "length": 19}, "sports", "task2"),
        Example({"tech_words": 0.12, "sports_words": 0.88, "length": 17}, "sports", "task2"),
    ]
    
    support_set2 = SupportSet(
        task_id="topic",
        examples=task2_support,
        task_type=TaskType.CLASSIFICATION,
        num_classes=2,
        shots_per_class=3
    )
    
    model2 = agent.adapt_to_task("topic", support_set2)
    print(f"✓ Adapted to topic task")
    print(f"  Classes: {len(model2.prototypes)}")
    
    # Test Task 2
    query2 = [
        Example({"tech_words": 0.87, "sports_words": 0.13, "length": 21}, "tech", "task2"),
        Example({"tech_words": 0.13, "sports_words": 0.87, "length": 18}, "sports", "task2"),
        Example({"tech_words": 0.91, "sports_words": 0.09, "length": 22}, "tech", "task2"),
    ]
    
    eval2 = agent.evaluate_task("topic", query2)
    print(f"  Evaluation: {eval2['accuracy']:.1%} accuracy ({eval2['correct']}/{eval2['total']} correct)")
    
    # Quick adaptation with new examples
    print("\n3. Quick Adaptation (Task 1 with new examples)")
    print("-" * 40)
    
    new_examples = [
        Example({"word_positive": 0.93, "word_negative": 0.07, "length": 12}, "positive", "task1"),
        Example({"word_positive": 0.07, "word_negative": 0.93, "length": 8}, "negative", "task1"),
    ]
    
    adapt_result = agent.quick_adapt("sentiment", new_examples)
    print(f"✓ Quick adaptation complete")
    print(f"  Prototypes updated: {adapt_result['prototypes_updated']}")
    print(f"  Total adaptations: {adapt_result['adaptations']}")
    
    # Re-evaluate after adaptation
    eval1_new = agent.evaluate_task("sentiment", query1)
    improvement = eval1_new['accuracy'] - eval1['accuracy']
    print(f"  New accuracy: {eval1_new['accuracy']:.1%} ({'+' if improvement >= 0 else ''}{improvement:.1%})")
    
    # Agent summary
    print("\n4. Agent Summary")
    print("-" * 40)
    summary = agent.get_agent_summary()
    
    print(f"Tasks learned: {summary['tasks_learned']}")
    print(f"Embedding updates: {summary['embedding_updates']}")
    
    meta = summary["meta_statistics"]
    if meta:
        print(f"\nMeta-learning statistics:")
        print(f"  Total tasks: {meta['total_tasks']}")
        print(f"  Total adaptations: {meta['total_adaptations']}")
        print(f"  Average performance: {meta['avg_performance']:.1%}")
        print(f"  Best performance: {meta['best_performance']:.1%}")
    
    print(f"\nTask details:")
    for task_id, details in summary["task_details"].items():
        print(f"  {task_id}:")
        print(f"    Classes: {details['num_classes']}")
        print(f"    Adaptations: {details['adaptations']}")
        print(f"    Performance: {details['performance']:.1%}" if details['performance'] else "    Performance: N/A")
    
    print("\n" + "=" * 80)
    print("✓ Few-shot adaptation agent demonstration complete!")
    print("🎯 MILESTONE ACHIEVED: 115/170 patterns (67.6%)!")
    print("  Rapid task adaptation, meta-learning, and prototypical networks working.")
    print("=" * 80)
