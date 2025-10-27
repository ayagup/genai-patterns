"""
Active Learning Pattern for Agentic AI

This pattern implements active learning where the agent strategically requests
human input for cases where it is most uncertain or where labels would be most
valuable for improving the model.

Key Concepts:
1. Uncertainty Sampling - Request labels for most uncertain predictions
2. Query by Committee - Multiple models vote, request label when they disagree
3. Expected Model Change - Select samples that would most change the model
4. Diversity Sampling - Select representative samples from unlabeled pool
5. Human-in-the-Loop Integration - Efficiently use human expertise

Use Cases:
- Model training with limited labeled data
- Continuous learning from user feedback
- Quality assurance in production systems
- Ambiguous case resolution
- Cost-effective annotation
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import random
import math


class SamplingStrategy(Enum):
    """Different active learning sampling strategies"""
    UNCERTAINTY = "uncertainty"
    MARGIN = "margin"
    ENTROPY = "entropy"
    COMMITTEE = "query_by_committee"
    EXPECTED_CHANGE = "expected_model_change"
    DIVERSITY = "diversity"


@dataclass
class Sample:
    """Represents a data sample with predictions and metadata"""
    id: str
    input_data: Any
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    label: Optional[str] = None
    feature_vector: Optional[List[float]] = None


@dataclass
class QueryResult:
    """Result of querying human for label"""
    sample_id: str
    label: str
    confidence: float
    feedback: Optional[str] = None


class UncertaintySampler:
    """Samples based on prediction uncertainty"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def score(self, sample: Sample) -> float:
        """Score based on uncertainty (1 - confidence)"""
        if sample.confidence is None:
            return 1.0
        return 1.0 - sample.confidence
    
    def select(self, samples: List[Sample], n: int) -> List[Sample]:
        """Select n most uncertain samples"""
        scored = [(self.score(s), s) for s in samples]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [s for _, s in scored[:n]]


class MarginSampler:
    """Samples based on margin between top predictions"""
    
    def __init__(self, predictions_key: str = "top_predictions"):
        self.predictions_key = predictions_key
    
    def score(self, sample: Sample) -> float:
        """Score based on margin between top 2 predictions"""
        # In real implementation, would use actual probability distribution
        # For demo, use confidence as proxy
        if sample.confidence is None:
            return 1.0
        # Smaller margin = more uncertain
        return 1.0 - abs(sample.confidence - 0.5) * 2
    
    def select(self, samples: List[Sample], n: int) -> List[Sample]:
        """Select n samples with smallest margin"""
        scored = [(self.score(s), s) for s in samples]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [s for _, s in scored[:n]]


class EntropySampler:
    """Samples based on prediction entropy"""
    
    def score(self, sample: Sample) -> float:
        """Calculate entropy of predictions"""
        if sample.confidence is None:
            return 1.0
        
        # Binary entropy for demo
        p = sample.confidence
        if p == 0 or p == 1:
            return 0.0
        return -(p * math.log2(p) + (1-p) * math.log2(1-p))
    
    def select(self, samples: List[Sample], n: int) -> List[Sample]:
        """Select n samples with highest entropy"""
        scored = [(self.score(s), s) for s in samples]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [s for _, s in scored[:n]]


class CommitteeSampler:
    """Query by Committee - samples where models disagree"""
    
    def __init__(self, models: List[Any]):
        self.models = models
    
    def score(self, sample: Sample) -> float:
        """Score based on committee disagreement"""
        # In real implementation, would query all models
        # For demo, simulate with random votes
        votes = [random.choice([True, False]) for _ in self.models]
        agreement = sum(votes) / len(votes)
        # Maximum disagreement at 0.5
        return 1.0 - abs(agreement - 0.5) * 2
    
    def select(self, samples: List[Sample], n: int) -> List[Sample]:
        """Select n samples with most disagreement"""
        scored = [(self.score(s), s) for s in samples]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [s for _, s in scored[:n]]


class DiversitySampler:
    """Samples to maximize diversity in training set"""
    
    def select(self, samples: List[Sample], n: int) -> List[Sample]:
        """Select diverse samples using greedy algorithm"""
        if not samples:
            return []
        
        selected = []
        remaining = samples.copy()
        
        # Start with random sample
        first = random.choice(remaining)
        selected.append(first)
        remaining.remove(first)
        
        # Greedily add most different samples
        while len(selected) < n and remaining:
            best_sample = None
            best_distance = -1
            
            for candidate in remaining:
                # Calculate minimum distance to selected samples
                min_dist = min(
                    self._distance(candidate, s) for s in selected
                )
                if min_dist > best_distance:
                    best_distance = min_dist
                    best_sample = candidate
            
            if best_sample:
                selected.append(best_sample)
                remaining.remove(best_sample)
        
        return selected
    
    def _distance(self, s1: Sample, s2: Sample) -> float:
        """Calculate distance between samples"""
        # In real implementation, use actual feature vectors
        # For demo, use simple hash-based distance
        return abs(hash(str(s1.input_data)) - hash(str(s2.input_data)))


class ActiveLearningAgent:
    """
    Agent that uses active learning to improve with minimal labeled data.
    
    Selects the most informative samples to query humans for labels,
    maximizing learning efficiency.
    """
    
    def __init__(
        self,
        strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY,
        budget: int = 100,
        batch_size: int = 10
    ):
        self.strategy = strategy
        self.budget = budget
        self.batch_size = batch_size
        self.queries_made = 0
        self.labeled_samples: List[Sample] = []
        self.unlabeled_samples: List[Sample] = []
        
        # Initialize sampler based on strategy
        self.sampler = self._create_sampler(strategy)
    
    def _create_sampler(self, strategy: SamplingStrategy):
        """Create appropriate sampler for strategy"""
        if strategy == SamplingStrategy.UNCERTAINTY:
            return UncertaintySampler()
        elif strategy == SamplingStrategy.MARGIN:
            return MarginSampler()
        elif strategy == SamplingStrategy.ENTROPY:
            return EntropySampler()
        elif strategy == SamplingStrategy.COMMITTEE:
            return CommitteeSampler(models=[None, None, None])
        elif strategy == SamplingStrategy.DIVERSITY:
            return DiversitySampler()
        else:
            return UncertaintySampler()
    
    def add_unlabeled_samples(self, samples: List[Sample]):
        """Add samples to unlabeled pool"""
        self.unlabeled_samples.extend(samples)
        print(f"Added {len(samples)} unlabeled samples. "
              f"Total unlabeled: {len(self.unlabeled_samples)}")
    
    def predict(self, sample: Sample) -> Tuple[str, float]:
        """Make prediction on sample (placeholder)"""
        # In real implementation, use actual model
        # For demo, simulate prediction
        predictions = ["positive", "negative", "neutral"]
        prediction = random.choice(predictions)
        confidence = random.uniform(0.3, 0.95)
        
        sample.prediction = prediction
        sample.confidence = confidence
        
        return prediction, confidence
    
    def select_queries(self, n: Optional[int] = None) -> List[Sample]:
        """
        Select most informative samples to query for labels.
        
        Args:
            n: Number of samples to select (default: batch_size)
        
        Returns:
            List of selected samples for labeling
        """
        if n is None:
            n = self.batch_size
        
        # Check budget
        remaining_budget = self.budget - self.queries_made
        n = min(n, remaining_budget)
        
        if n <= 0:
            print("Budget exhausted!")
            return []
        
        # Make predictions on unlabeled samples if needed
        for sample in self.unlabeled_samples:
            if sample.prediction is None:
                self.predict(sample)
        
        # Select based on strategy
        selected = self.sampler.select(self.unlabeled_samples, n)
        
        print(f"\nSelected {len(selected)} samples for labeling:")
        for i, sample in enumerate(selected, 1):
            print(f"{i}. Sample {sample.id}: "
                  f"prediction={sample.prediction}, "
                  f"confidence={sample.confidence:.3f}")
        
        return selected
    
    def query_human(
        self,
        sample: Sample,
        human_oracle: Optional[Callable] = None
    ) -> QueryResult:
        """
        Query human for label on sample.
        
        Args:
            sample: Sample to label
            human_oracle: Optional function to simulate human labeling
        
        Returns:
            QueryResult with label and metadata
        """
        if human_oracle:
            label = human_oracle(sample)
        else:
            # Simulate human labeling
            print(f"\n[Human Query] Please label sample {sample.id}:")
            print(f"Input: {sample.input_data}")
            print(f"Model prediction: {sample.prediction} "
                  f"(confidence: {sample.confidence:.3f})")
            
            # For demo, simulate human response
            label = random.choice(["positive", "negative", "neutral"])
            print(f"Human label: {label}")
        
        self.queries_made += 1
        
        return QueryResult(
            sample_id=sample.id,
            label=label,
            confidence=1.0,
            feedback="Human annotation"
        )
    
    def update_with_label(self, sample: Sample, query_result: QueryResult):
        """Update agent with newly labeled sample"""
        sample.label = query_result.label
        
        # Move from unlabeled to labeled
        if sample in self.unlabeled_samples:
            self.unlabeled_samples.remove(sample)
        self.labeled_samples.append(sample)
        
        # In real implementation, would retrain model here
        print(f"Updated model with sample {sample.id}")
    
    def active_learning_round(
        self,
        n_queries: Optional[int] = None,
        human_oracle: Optional[Callable] = None
    ) -> List[QueryResult]:
        """
        Execute one round of active learning.
        
        Args:
            n_queries: Number of samples to query
            human_oracle: Optional function to simulate human
        
        Returns:
            List of query results
        """
        print(f"\n{'='*60}")
        print(f"Active Learning Round (Strategy: {self.strategy.value})")
        print(f"Budget: {self.queries_made}/{self.budget}")
        print(f"Labeled: {len(self.labeled_samples)}, "
              f"Unlabeled: {len(self.unlabeled_samples)}")
        print(f"{'='*60}")
        
        # Select informative samples
        selected = self.select_queries(n_queries)
        
        if not selected:
            return []
        
        # Query human for labels
        results = []
        for sample in selected:
            result = self.query_human(sample, human_oracle)
            self.update_with_label(sample, result)
            results.append(result)
        
        print(f"\nRound complete. Queried {len(results)} samples.")
        print(f"Remaining budget: {self.budget - self.queries_made}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            "strategy": self.strategy.value,
            "queries_made": self.queries_made,
            "budget_remaining": self.budget - self.queries_made,
            "labeled_samples": len(self.labeled_samples),
            "unlabeled_samples": len(self.unlabeled_samples),
            "budget_utilization": self.queries_made / self.budget
        }


def demonstrate_active_learning():
    """Demonstrate different active learning strategies"""
    
    print("Active Learning Pattern Demonstration")
    print("=" * 70)
    
    # Create unlabeled samples
    samples = [
        Sample(id=f"sample_{i}", input_data=f"text_{i}")
        for i in range(50)
    ]
    
    strategies = [
        SamplingStrategy.UNCERTAINTY,
        SamplingStrategy.ENTROPY,
        SamplingStrategy.DIVERSITY
    ]
    
    for strategy in strategies:
        print(f"\n\n{'='*70}")
        print(f"Testing Strategy: {strategy.value.upper()}")
        print(f"{'='*70}")
        
        # Create agent with strategy
        agent = ActiveLearningAgent(
            strategy=strategy,
            budget=15,
            batch_size=5
        )
        
        # Add unlabeled samples
        agent.add_unlabeled_samples(samples.copy())
        
        # Run 3 rounds of active learning
        for round_num in range(3):
            print(f"\n--- Round {round_num + 1} ---")
            agent.active_learning_round()
        
        # Print statistics
        stats = agent.get_statistics()
        print(f"\n{strategy.value.upper()} Strategy Results:")
        print(f"  Queries made: {stats['queries_made']}")
        print(f"  Labeled samples: {stats['labeled_samples']}")
        print(f"  Budget utilization: {stats['budget_utilization']:.1%}")


if __name__ == "__main__":
    demonstrate_active_learning()
    
    print("\n\n" + "="*70)
    print("Active Learning Pattern Summary")
    print("="*70)
    print("""
Key Benefits:
1. Efficient use of human expertise
2. Reduces labeling costs significantly
3. Focuses on most informative samples
4. Continuous improvement with feedback
5. Works with limited labeled data

Common Strategies:
- Uncertainty Sampling: Query most uncertain predictions
- Margin Sampling: Query when top predictions are close
- Entropy Sampling: Query highest entropy cases
- Query by Committee: Query when models disagree
- Diversity Sampling: Ensure representative coverage

Best Practices:
- Start with diverse initial set
- Balance exploration vs exploitation
- Set appropriate batch sizes
- Monitor annotation quality
- Combine multiple strategies
- Consider annotation cost

Use Cases:
- Model training with limited labels
- Production quality assurance
- Continuous learning systems
- Ambiguity resolution
- Cost-effective data annotation
""")
