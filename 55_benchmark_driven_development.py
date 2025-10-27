"""
Benchmark-Driven Development Pattern Implementation

This pattern uses benchmark scores to guide agent development through:
- Baseline establishment
- Iterative improvement
- Performance tracking
- A/B testing
- Competitive analysis

Use cases:
- Research and development
- Performance optimization
- Model selection
- Feature evaluation
- Quality assurance
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import statistics


class BenchmarkType(Enum):
    """Types of benchmarks"""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    COST = "cost"
    QUALITY = "quality"
    SAFETY = "safety"
    ROBUSTNESS = "robustness"


@dataclass
class BenchmarkResult:
    """Result from a benchmark run"""
    benchmark_name: str
    version: str
    score: float
    metrics: Dict[str, float]
    timestamp: datetime
    config: Dict[str, Any]
    notes: str = ""


@dataclass
class BenchmarkSuite:
    """Collection of benchmark tests"""
    name: str
    description: str
    test_cases: List[Dict[str, Any]]
    scoring_fn: Callable
    weight: float = 1.0


@dataclass
class DevelopmentIteration:
    """Single development iteration"""
    iteration: int
    version: str
    changes: List[str]
    results: List[BenchmarkResult]
    aggregate_score: float
    improvement: float
    timestamp: datetime


class BenchmarkDrivenAgent:
    """
    Agent that uses benchmarks to guide development
    """
    
    def __init__(self, name: str = "BenchmarkAgent"):
        self.name = name
        self.benchmarks: Dict[str, BenchmarkSuite] = {}
        self.baseline: Optional[DevelopmentIteration] = None
        self.iterations: List[DevelopmentIteration] = []
        self.current_version = "v0.0"
        self.best_version = None
        self.best_score = 0.0
    
    def register_benchmark(self, suite: BenchmarkSuite):
        """Register a benchmark suite"""
        self.benchmarks[suite.name] = suite
        print(f"Registered benchmark: {suite.name}")
    
    def establish_baseline(self, version: str = "v1.0") -> DevelopmentIteration:
        """Establish baseline performance"""
        print(f"\nEstablishing baseline for version {version}...")
        
        results = []
        for name, suite in self.benchmarks.items():
            result = self._run_benchmark(suite, version, {})
            results.append(result)
        
        aggregate = self._calculate_aggregate_score(results)
        
        baseline = DevelopmentIteration(
            iteration=0,
            version=version,
            changes=["Initial baseline"],
            results=results,
            aggregate_score=aggregate,
            improvement=0.0,
            timestamp=datetime.now()
        )
        
        self.baseline = baseline
        self.iterations.append(baseline)
        self.current_version = version
        self.best_version = version
        self.best_score = aggregate
        
        print(f"Baseline established: {aggregate:.3f}")
        return baseline
    
    def run_iteration(
        self,
        version: str,
        changes: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> DevelopmentIteration:
        """Run a development iteration"""
        if not self.baseline:
            raise ValueError("Must establish baseline first")
        
        config = config or {}
        iteration_num = len(self.iterations)
        
        print(f"\nRunning iteration {iteration_num} (version {version})...")
        print(f"Changes: {', '.join(changes)}")
        
        results = []
        for name, suite in self.benchmarks.items():
            result = self._run_benchmark(suite, version, config)
            results.append(result)
        
        aggregate = self._calculate_aggregate_score(results)
        improvement = aggregate - self.baseline.aggregate_score
        
        iteration = DevelopmentIteration(
            iteration=iteration_num,
            version=version,
            changes=changes,
            results=results,
            aggregate_score=aggregate,
            improvement=improvement,
            timestamp=datetime.now()
        )
        
        self.iterations.append(iteration)
        self.current_version = version
        
        # Track best version
        if aggregate > self.best_score:
            self.best_version = version
            self.best_score = aggregate
            print(f"🎉 New best score: {aggregate:.3f} (improvement: +{improvement:.3f})")
        else:
            print(f"Score: {aggregate:.3f} (change: {improvement:+.3f})")
        
        return iteration
    
    def _run_benchmark(
        self,
        suite: BenchmarkSuite,
        version: str,
        config: Dict[str, Any]
    ) -> BenchmarkResult:
        """Run a single benchmark suite"""
        print(f"  Running {suite.name}...", end=" ")
        
        # Run test cases
        scores = []
        for test_case in suite.test_cases:
            score = self._execute_test_case(test_case, config)
            scores.append(score)
        
        # Calculate metrics
        avg_score = statistics.mean(scores)
        metrics = {
            'average': avg_score,
            'min': min(scores),
            'max': max(scores),
            'stdev': statistics.stdev(scores) if len(scores) > 1 else 0.0,
            'num_tests': len(scores)
        }
        
        result = BenchmarkResult(
            benchmark_name=suite.name,
            version=version,
            score=avg_score * suite.weight,
            metrics=metrics,
            timestamp=datetime.now(),
            config=config
        )
        
        print(f"Score: {avg_score:.3f}")
        return result
    
    def _execute_test_case(
        self,
        test_case: Dict[str, Any],
        config: Dict[str, Any]
    ) -> float:
        """Execute a single test case (simulated)"""
        # In real implementation, this would run actual tests
        # For demo, we'll simulate scores
        
        base_score = test_case.get('base_score', 0.7)
        difficulty = test_case.get('difficulty', 1.0)
        
        # Simulate config impact
        config_boost = config.get('performance_boost', 0.0)
        
        score = base_score + config_boost
        score = score / difficulty
        
        return min(1.0, max(0.0, score))
    
    def _calculate_aggregate_score(self, results: List[BenchmarkResult]) -> float:
        """Calculate weighted aggregate score"""
        if not results:
            return 0.0
        
        total_weight = sum(self.benchmarks[r.benchmark_name].weight for r in results)
        weighted_sum = sum(r.score for r in results)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions"""
        iter1 = next((i for i in self.iterations if i.version == version1), None)
        iter2 = next((i for i in self.iterations if i.version == version2), None)
        
        if not iter1 or not iter2:
            return {"error": "Version not found"}
        
        comparison = {
            "version1": version1,
            "version2": version2,
            "score_diff": iter2.aggregate_score - iter1.aggregate_score,
            "improvement": f"{((iter2.aggregate_score / iter1.aggregate_score - 1) * 100):.1f}%",
            "benchmark_comparison": []
        }
        
        for r1 in iter1.results:
            r2 = next(r for r in iter2.results if r.benchmark_name == r1.benchmark_name)
            comparison["benchmark_comparison"].append({
                "benchmark": r1.benchmark_name,
                "version1_score": f"{r1.score:.3f}",
                "version2_score": f"{r2.score:.3f}",
                "diff": f"{r2.score - r1.score:+.3f}"
            })
        
        return comparison
    
    def get_development_report(self) -> Dict[str, Any]:
        """Get comprehensive development report"""
        if not self.iterations or not self.baseline:
            return {"message": "No iterations yet"}
        
        scores = [i.aggregate_score for i in self.iterations]
        improvements = [i.improvement for i in self.iterations[1:]]
        
        return {
            "total_iterations": len(self.iterations),
            "current_version": self.current_version,
            "best_version": self.best_version,
            "best_score": f"{self.best_score:.3f}",
            "baseline_score": f"{self.baseline.aggregate_score:.3f}",
            "total_improvement": f"{self.best_score - self.baseline.aggregate_score:+.3f}",
            "improvement_rate": f"{((self.best_score / self.baseline.aggregate_score - 1) * 100):+.1f}%",
            "avg_iteration_improvement": f"{statistics.mean(improvements):.3f}" if improvements else "N/A",
            "progress_trend": "Improving" if improvements and improvements[-1] > 0 else "Declining",
            "iteration_history": [
                {
                    "iteration": i.iteration,
                    "version": i.version,
                    "score": f"{i.aggregate_score:.3f}",
                    "improvement": f"{i.improvement:+.3f}",
                    "changes": i.changes
                }
                for i in self.iterations
            ]
        }
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get suggestions for optimization"""
        suggestions = []
        
        if len(self.iterations) < 2:
            suggestions.append("Run more iterations to identify trends")
            return suggestions
        
        # Analyze trends
        recent_scores = [i.aggregate_score for i in self.iterations[-3:]]
        
        if all(recent_scores[i] <= recent_scores[i+1] for i in range(len(recent_scores)-1)):
            suggestions.append("✓ Consistent improvement trend - continue current approach")
        elif all(recent_scores[i] >= recent_scores[i+1] for i in range(len(recent_scores)-1)):
            suggestions.append("⚠ Declining performance - consider reverting to previous approach")
        else:
            suggestions.append("→ Mixed results - analyze what worked and what didn't")
        
        # Check if stuck
        if len(recent_scores) >= 3 and max(recent_scores) - min(recent_scores) < 0.01:
            suggestions.append("⚠ Performance plateau - try different optimization strategies")
        
        # Compare to best
        current_score = self.iterations[-1].aggregate_score
        if current_score < self.best_score - 0.05:
            suggestions.append(f"💡 Current version underperforming - consider reverting to {self.best_version}")
        
        # Benchmark-specific suggestions
        latest = self.iterations[-1]
        for result in latest.results:
            if result.score < 0.5:
                suggestions.append(f"⚠ Low score on {result.benchmark_name} - focus improvement here")
        
        return suggestions


def demo_benchmark_driven_development():
    """Demonstrate benchmark-driven development"""
    print("=" * 70)
    print("Benchmark-Driven Development Pattern Demo")
    print("=" * 70)
    
    agent = BenchmarkDrivenAgent()
    
    # Define benchmark suites
    accuracy_suite = BenchmarkSuite(
        name="Accuracy Benchmark",
        description="Measures correctness of responses",
        test_cases=[
            {"question": "What is 2+2?", "base_score": 0.85, "difficulty": 1.0},
            {"question": "Explain quantum physics", "base_score": 0.65, "difficulty": 1.5},
            {"question": "Translate text", "base_score": 0.75, "difficulty": 1.2}
        ],
        scoring_fn=lambda x: x,
        weight=1.0
    )
    
    latency_suite = BenchmarkSuite(
        name="Latency Benchmark",
        description="Measures response time",
        test_cases=[
            {"task": "Simple query", "base_score": 0.9, "difficulty": 1.0},
            {"task": "Complex query", "base_score": 0.7, "difficulty": 1.3}
        ],
        scoring_fn=lambda x: x,
        weight=0.8
    )
    
    quality_suite = BenchmarkSuite(
        name="Quality Benchmark",
        description="Measures output quality",
        test_cases=[
            {"aspect": "Coherence", "base_score": 0.8, "difficulty": 1.1},
            {"aspect": "Completeness", "base_score": 0.75, "difficulty": 1.0},
            {"aspect": "Relevance", "base_score": 0.85, "difficulty": 1.0}
        ],
        scoring_fn=lambda x: x,
        weight=1.2
    )
    
    print("\n1. Registering Benchmarks")
    print("-" * 70)
    agent.register_benchmark(accuracy_suite)
    agent.register_benchmark(latency_suite)
    agent.register_benchmark(quality_suite)
    
    print("\n2. Establishing Baseline")
    print("-" * 70)
    baseline = agent.establish_baseline("v1.0")
    
    print("\n3. Development Iterations")
    print("-" * 70)
    
    # Iteration 1: Improve accuracy
    agent.run_iteration(
        version="v1.1",
        changes=["Added better prompts", "Improved context handling"],
        config={"performance_boost": 0.05}
    )
    
    # Iteration 2: Optimize latency
    agent.run_iteration(
        version="v1.2",
        changes=["Implemented caching", "Reduced model size"],
        config={"performance_boost": 0.08}
    )
    
    # Iteration 3: Enhance quality
    agent.run_iteration(
        version="v1.3",
        changes=["Added verification step", "Improved output formatting"],
        config={"performance_boost": 0.12}
    )
    
    # Iteration 4: Regression (for demo)
    agent.run_iteration(
        version="v1.4",
        changes=["Experimental feature", "New approach"],
        config={"performance_boost": 0.03}
    )
    
    print("\n" + "=" * 70)
    print("4. Development Report")
    print("-" * 70)
    
    report = agent.get_development_report()
    print(json.dumps(report, indent=2))
    
    print("\n" + "=" * 70)
    print("5. Version Comparison")
    print("-" * 70)
    
    comparison = agent.compare_versions("v1.0", "v1.3")
    print(json.dumps(comparison, indent=2))
    
    print("\n" + "=" * 70)
    print("6. Optimization Suggestions")
    print("-" * 70)
    
    suggestions = agent.get_optimization_suggestions()
    for suggestion in suggestions:
        print(f"  {suggestion}")
    
    print("\n" + "=" * 70)
    print("7. Best Version Summary")
    print("-" * 70)
    
    print(f"Best Version: {agent.best_version}")
    print(f"Best Score: {agent.best_score:.3f}")
    print(f"Improvement from Baseline: {(agent.best_score - baseline.aggregate_score):.3f}")
    print(f"Improvement Rate: {((agent.best_score / baseline.aggregate_score - 1) * 100):.1f}%")
    
    print("\n" + "=" * 70)
    print("Benchmark-Driven Development Pattern Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_benchmark_driven_development()
