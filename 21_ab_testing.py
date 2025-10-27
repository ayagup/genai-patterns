"""
A/B Testing Pattern
Compares different agent configurations
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random
import statistics
class Variant(Enum):
    CONTROL = "control"
    TREATMENT_A = "treatment_a"
    TREATMENT_B = "treatment_b"
@dataclass
class ExperimentConfig:
    name: str
    variants: Dict[str, float]  # variant_name -> traffic_percentage
    metrics: List[str]
    def __post_init__(self):
        # Validate traffic percentages sum to ~1.0
        total = sum(self.variants.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Traffic percentages must sum to 1.0, got {total}")
@dataclass
class VariantResult:
    variant_name: str
    request_count: int = 0
    success_count: int = 0
    total_latency_ms: float = 0.0
    metric_values: Dict[str, List[float]] = field(default_factory=dict)
    def add_result(self, success: bool, latency_ms: float, metrics: Dict[str, float] = None):
        """Record a result for this variant"""
        self.request_count += 1
        if success:
            self.success_count += 1
        self.total_latency_ms += latency_ms
        if metrics:
            for metric_name, value in metrics.items():
                if metric_name not in self.metric_values:
                    self.metric_values[metric_name] = []
                self.metric_values[metric_name].append(value)
    def get_stats(self) -> Dict[str, Any]:
        """Calculate statistics for this variant"""
        success_rate = (self.success_count / self.request_count 
                       if self.request_count > 0 else 0)
        avg_latency = (self.total_latency_ms / self.request_count 
                      if self.request_count > 0 else 0)
        metric_stats = {}
        for metric_name, values in self.metric_values.items():
            if values:
                metric_stats[metric_name] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values)
                }
        return {
            'variant': self.variant_name,
            'requests': self.request_count,
            'success_rate': success_rate,
            'avg_latency_ms': avg_latency,
            'metrics': metric_stats
        }
class ABTestingFramework:
    """Framework for running A/B tests on agents"""
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.results: Dict[str, VariantResult] = {
            variant: VariantResult(variant)
            for variant in experiment_config.variants.keys()
        }
        self.start_time = datetime.now()
    def assign_variant(self, user_id: str = None) -> str:
        """Assign a variant to a user/request"""
        # Use consistent hashing if user_id provided
        if user_id:
            hash_val = hash(user_id) % 100
            cumulative = 0
            for variant, percentage in self.config.variants.items():
                cumulative += percentage * 100
                if hash_val < cumulative:
                    return variant
        # Random assignment
        r = random.random()
        cumulative = 0
        for variant, percentage in self.config.variants.items():
            cumulative += percentage
            if r < cumulative:
                return variant
        # Fallback
        return list(self.config.variants.keys())[0]
    def record_result(self, variant: str, success: bool, latency_ms: float,
                     metrics: Dict[str, float] = None):
        """Record a result for a variant"""
        if variant in self.results:
            self.results[variant].add_result(success, latency_ms, metrics)
    def get_comparison(self) -> Dict[str, Any]:
        """Get comparison of all variants"""
        stats = {
            variant: result.get_stats()
            for variant, result in self.results.items()
        }
        # Determine winner based on success rate
        winner = max(stats.items(), 
                    key=lambda x: x[1]['success_rate'])
        return {
            'experiment': self.config.name,
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'variants': stats,
            'winner': {
                'variant': winner[0],
                'success_rate': winner[1]['success_rate']
            }
        }
    def print_results(self):
        """Print detailed results"""
        comparison = self.get_comparison()
        print(f"\n{'='*70}")
        print(f"A/B TEST RESULTS: {comparison['experiment']}")
        print(f"{'='*70}")
        print(f"Duration: {comparison['duration_seconds']:.1f}s")
        print(f"\nVariant Performance:")
        for variant_name, stats in comparison['variants'].items():
            print(f"\n{variant_name}:")
            print(f"  Requests: {stats['requests']}")
            print(f"  Success Rate: {stats['success_rate']:.1%}")
            print(f"  Avg Latency: {stats['avg_latency_ms']:.2f}ms")
            if stats['metrics']:
                print(f"  Metrics:")
                for metric_name, metric_stats in stats['metrics'].items():
                    print(f"    {metric_name}:")
                    print(f"      Mean: {metric_stats['mean']:.2f}")
                    print(f"      Median: {metric_stats['median']:.2f}")
                    print(f"      StdDev: {metric_stats['stdev']:.2f}")
        print(f"\n{'='*70}")
        print(f"WINNER: {comparison['winner']['variant']}")
        print(f"Success Rate: {comparison['winner']['success_rate']:.1%}")
        print(f"{'='*70}")
class ABTestedAgent:
    """Agent with A/B testing capabilities"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        # Setup A/B test
        self.experiment = ABTestingFramework(
            ExperimentConfig(
                name="Response Strategy Test",
                variants={
                    "control": 0.34,      # Original strategy
                    "treatment_a": 0.33,  # New strategy A
                    "treatment_b": 0.33   # New strategy B
                },
                metrics=["quality_score", "user_satisfaction"]
            )
        )
    def process_request(self, request: str, user_id: str = None) -> Dict[str, Any]:
        """Process request with A/B testing"""
        import time
        # Assign variant
        variant = self.experiment.assign_variant(user_id)
        print(f"\nProcessing with variant: {variant}")
        # Measure latency
        start_time = time.time()
        # Execute variant-specific logic
        if variant == "control":
            response = self._control_strategy(request)
        elif variant == "treatment_a":
            response = self._treatment_a_strategy(request)
        else:
            response = self._treatment_b_strategy(request)
        latency_ms = (time.time() - start_time) * 1000
        # Simulate metrics
        quality_score = response.get('quality', 0)
        user_satisfaction = response.get('satisfaction', 0)
        success = response.get('success', False)
        # Record result
        self.experiment.record_result(
            variant,
            success,
            latency_ms,
            metrics={
                'quality_score': quality_score,
                'user_satisfaction': user_satisfaction
            }
        )
        print(f"  Latency: {latency_ms:.2f}ms")
        print(f"  Quality: {quality_score:.2f}")
        print(f"  Satisfaction: {user_satisfaction:.2f}")
        return response
    def _control_strategy(self, request: str) -> Dict[str, Any]:
        """Original/control strategy"""
        import time
        time.sleep(0.1)  # Simulate work
        return {
            'success': random.random() > 0.15,  # 85% success rate
            'response': f"Control response for: {request}",
            'quality': random.uniform(0.7, 0.85),
            'satisfaction': random.uniform(0.65, 0.80)
        }
    def _treatment_a_strategy(self, request: str) -> Dict[str, Any]:
        """Treatment A - Optimized for speed"""
        import time
        time.sleep(0.05)  # Faster
        return {
            'success': random.random() > 0.12,  # 88% success rate
            'response': f"Treatment A response for: {request}",
            'quality': random.uniform(0.75, 0.90),
            'satisfaction': random.uniform(0.70, 0.85)
        }
    def _treatment_b_strategy(self, request: str) -> Dict[str, Any]:
        """Treatment B - Optimized for quality"""
        import time
        time.sleep(0.15)  # Slower but higher quality
        return {
            'success': random.random() > 0.10,  # 90% success rate
            'response': f"Treatment B response for: {request}",
            'quality': random.uniform(0.80, 0.95),
            'satisfaction': random.uniform(0.75, 0.90)
        }
# Usage
if __name__ == "__main__":
    print("="*80)
    print("A/B TESTING PATTERN DEMONSTRATION")
    print("="*80)
    agent = ABTestedAgent("ab-test-agent")
    # Simulate requests from different users
    requests = [
        "Analyze customer sentiment",
        "Generate summary report",
        "Recommend products",
        "Answer technical question",
        "Create visualization"
    ]
    # Run experiment
    print("\nRunning A/B test experiment...")
    print("="*80)
    num_iterations = 30
    for i in range(num_iterations):
        request = random.choice(requests)
        user_id = f"user_{i % 10}"  # 10 different users
        agent.process_request(request, user_id)
        # Small delay between requests
        import time
        time.sleep(0.05)
    # Print results
    agent.experiment.print_results()
    # Statistical significance check (simplified)
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    comparison = agent.experiment.get_comparison()
    variants = comparison['variants']
    # Compare control vs treatments
    control_sr = variants['control']['success_rate']
    for variant_name in ['treatment_a', 'treatment_b']:
        variant_sr = variants[variant_name]['success_rate']
        improvement = ((variant_sr - control_sr) / control_sr * 100) if control_sr > 0 else 0
        print(f"\n{variant_name} vs control:")
        print(f"  Success Rate Improvement: {improvement:+.1f}%")
        print(f"  Sample Size: {variants[variant_name]['requests']}")
