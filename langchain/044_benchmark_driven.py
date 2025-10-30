"""
Pattern 044: Benchmark-Driven Development

Description:
    Benchmark-Driven Development enables agents to guide their development and
    improvement through systematic benchmark evaluation. Maintains baseline
    performance, iterates on improvements, tests against benchmarks, and
    compares results to drive continuous optimization.

Components:
    - Benchmark Suite: Collection of test cases and metrics
    - Baseline Tracker: Maintains reference performance
    - Test Runner: Executes benchmarks
    - Score Comparator: Compares performance across versions
    - Improvement Suggester: Recommends optimizations

Use Cases:
    - Agent development and iteration
    - Performance optimization
    - Regression testing
    - Model comparison and selection
    - Production monitoring
    - Competitive benchmarking

LangChain Implementation:
    Uses structured benchmark suites with scoring, baseline tracking, and
    comparison tools to drive measurable agent improvement.
"""

import os
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class BenchmarkCategory(Enum):
    """Categories of benchmarks."""
    ACCURACY = "accuracy"
    REASONING = "reasoning"
    FACTUAL_KNOWLEDGE = "factual_knowledge"
    INSTRUCTION_FOLLOWING = "instruction_following"
    SAFETY = "safety"
    CREATIVITY = "creativity"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"


class PerformanceStatus(Enum):
    """Performance comparison status."""
    IMPROVED = "improved"
    MAINTAINED = "maintained"
    REGRESSED = "regressed"
    NO_BASELINE = "no_baseline"


@dataclass
class BenchmarkTestCase:
    """A single test case in a benchmark."""
    test_id: str
    category: BenchmarkCategory
    input: str
    expected_output: Optional[str] = None
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark test."""
    test_id: str
    category: BenchmarkCategory
    input: str
    output: str
    score: float  # 0.0 to 1.0
    evaluation_details: Dict[str, Any]
    passed: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkRun:
    """Complete benchmark run results."""
    run_id: str
    version: str
    results: List[BenchmarkResult]
    overall_score: float
    category_scores: Dict[BenchmarkCategory, float]
    pass_rate: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceComparison:
    """Comparison between two benchmark runs."""
    baseline_run: BenchmarkRun
    current_run: BenchmarkRun
    status: PerformanceStatus
    score_delta: float
    category_deltas: Dict[BenchmarkCategory, float]
    improved_tests: List[str]
    regressed_tests: List[str]
    recommendations: List[str]


class BenchmarkSuite:
    """
    Manages a collection of benchmark tests.
    
    Features:
    - Test case management
    - Category organization
    - Subset selection
    - Test validation
    """
    
    def __init__(self, name: str):
        self.name = name
        self.test_cases: List[BenchmarkTestCase] = []
        self.categories: Dict[BenchmarkCategory, List[BenchmarkTestCase]] = {}
    
    def add_test(self, test_case: BenchmarkTestCase):
        """Add a test case to the suite."""
        self.test_cases.append(test_case)
        
        if test_case.category not in self.categories:
            self.categories[test_case.category] = []
        self.categories[test_case.category].append(test_case)
    
    def get_tests_by_category(
        self,
        category: BenchmarkCategory
    ) -> List[BenchmarkTestCase]:
        """Get all tests in a category."""
        return self.categories.get(category, [])
    
    def get_test_count(self) -> int:
        """Get total number of tests."""
        return len(self.test_cases)
    
    def get_category_count(self) -> int:
        """Get number of categories."""
        return len(self.categories)


class BenchmarkDrivenAgent:
    """
    Agent that uses benchmarks to drive development and improvement.
    
    Features:
    - Benchmark execution
    - Baseline tracking
    - Performance comparison
    - Improvement suggestions
    - Regression detection
    """
    
    def __init__(
        self,
        agent_version: str,
        temperature: float = 0.5
    ):
        self.agent_version = agent_version
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        
        # Response generation prompt
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant. Provide helpful, accurate responses."),
            ("user", "{input}")
        ])
        
        # Evaluation prompt
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a benchmark evaluator. Evaluate the response quality.

Category: {category}
Evaluation Criteria:
{criteria}

Rate on a scale of 0.0 (poor) to 1.0 (excellent).

Provide:
SCORE: [0.0-1.0]
PASSED: [yes/no]
REASONING: [Explanation]
STRENGTHS: [What works well]
WEAKNESSES: [What needs improvement]"""),
            ("user", """Input: {input}

Expected Output: {expected_output}

Actual Output: {actual_output}

Evaluate:""")
        ])
        
        # Improvement suggestion prompt
        self.improvement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI optimization expert. Analyze benchmark results and suggest improvements.

Focus on:
- Patterns in failures
- Performance bottlenecks
- Category-specific weaknesses
- Actionable recommendations"""),
            ("user", """Benchmark Results Summary:
Overall Score: {overall_score}
Pass Rate: {pass_rate}

Category Scores:
{category_scores}

Failed Tests:
{failed_tests}

Provide improvement recommendations:""")
        ])
        
        # Benchmark runs history
        self.benchmark_runs: List[BenchmarkRun] = []
        self.baseline_run: Optional[BenchmarkRun] = None
    
    def generate_response(self, input: str) -> str:
        """Generate response to input."""
        chain = self.response_prompt | self.llm | StrOutputParser()
        return chain.invoke({"input": input})
    
    def evaluate_response(
        self,
        test_case: BenchmarkTestCase,
        actual_output: str
    ) -> BenchmarkResult:
        """
        Evaluate a response against benchmark criteria.
        
        Returns:
            BenchmarkResult with score and details
        """
        # Format criteria
        criteria_text = "\n".join([
            f"- {k}: {v}" for k, v in test_case.evaluation_criteria.items()
        ]) if test_case.evaluation_criteria else "General quality and correctness"
        
        expected = test_case.expected_output or "N/A"
        
        chain = self.evaluation_prompt | self.llm | StrOutputParser()
        evaluation = chain.invoke({
            "category": test_case.category.value,
            "criteria": criteria_text,
            "input": test_case.input,
            "expected_output": expected,
            "actual_output": actual_output
        })
        
        # Parse evaluation
        score = 0.5
        passed = False
        reasoning = ""
        strengths = []
        weaknesses = []
        
        lines = evaluation.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("PASSED:"):
                passed_text = line.split(':')[1].strip().lower()
                passed = passed_text in ['yes', 'true', 'pass']
            elif line.startswith("REASONING:"):
                reasoning = line.split(':', 1)[1].strip()
                current_section = "reasoning"
            elif line.startswith("STRENGTHS:"):
                current_section = "strengths"
            elif line.startswith("WEAKNESSES:"):
                current_section = "weaknesses"
            elif line and current_section:
                if current_section == "reasoning" and not line.startswith(('-', 'STRENGTHS', 'WEAKNESSES', 'PASSED')):
                    reasoning += " " + line
                elif line.startswith('-'):
                    item = line[1:].strip()
                    if current_section == "strengths":
                        strengths.append(item)
                    elif current_section == "weaknesses":
                        weaknesses.append(item)
        
        return BenchmarkResult(
            test_id=test_case.test_id,
            category=test_case.category,
            input=test_case.input,
            output=actual_output,
            score=score,
            evaluation_details={
                "reasoning": reasoning,
                "strengths": strengths,
                "weaknesses": weaknesses
            },
            passed=passed
        )
    
    def run_benchmark(
        self,
        suite: BenchmarkSuite,
        run_id: Optional[str] = None
    ) -> BenchmarkRun:
        """
        Run complete benchmark suite.
        
        Args:
            suite: BenchmarkSuite to run
            run_id: Optional run identifier
            
        Returns:
            BenchmarkRun with all results
        """
        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = []
        
        # Run each test
        for test_case in suite.test_cases:
            # Generate response
            output = self.generate_response(test_case.input)
            
            # Evaluate response
            result = self.evaluate_response(test_case, output)
            results.append(result)
        
        # Calculate metrics
        overall_score = statistics.mean([r.score for r in results])
        pass_rate = sum(1 for r in results if r.passed) / len(results)
        
        # Calculate category scores
        category_scores = {}
        for category in suite.categories.keys():
            category_results = [r for r in results if r.category == category]
            if category_results:
                category_scores[category] = statistics.mean([r.score for r in category_results])
        
        benchmark_run = BenchmarkRun(
            run_id=run_id,
            version=self.agent_version,
            results=results,
            overall_score=overall_score,
            category_scores=category_scores,
            pass_rate=pass_rate,
            metadata={"suite_name": suite.name}
        )
        
        self.benchmark_runs.append(benchmark_run)
        
        # Set as baseline if first run
        if self.baseline_run is None:
            self.baseline_run = benchmark_run
        
        return benchmark_run
    
    def compare_with_baseline(
        self,
        current_run: BenchmarkRun
    ) -> PerformanceComparison:
        """
        Compare current run with baseline.
        
        Returns:
            PerformanceComparison with detailed analysis
        """
        if self.baseline_run is None:
            return PerformanceComparison(
                baseline_run=current_run,
                current_run=current_run,
                status=PerformanceStatus.NO_BASELINE,
                score_delta=0.0,
                category_deltas={},
                improved_tests=[],
                regressed_tests=[],
                recommendations=["No baseline available for comparison"]
            )
        
        # Calculate score delta
        score_delta = current_run.overall_score - self.baseline_run.overall_score
        
        # Determine status
        if score_delta > 0.01:
            status = PerformanceStatus.IMPROVED
        elif score_delta < -0.01:
            status = PerformanceStatus.REGRESSED
        else:
            status = PerformanceStatus.MAINTAINED
        
        # Calculate category deltas
        category_deltas = {}
        for category in current_run.category_scores.keys():
            if category in self.baseline_run.category_scores:
                delta = current_run.category_scores[category] - self.baseline_run.category_scores[category]
                category_deltas[category] = delta
        
        # Find improved and regressed tests
        improved_tests = []
        regressed_tests = []
        
        baseline_results = {r.test_id: r for r in self.baseline_run.results}
        for current_result in current_run.results:
            if current_result.test_id in baseline_results:
                baseline_result = baseline_results[current_result.test_id]
                if current_result.score > baseline_result.score + 0.05:
                    improved_tests.append(current_result.test_id)
                elif current_result.score < baseline_result.score - 0.05:
                    regressed_tests.append(current_result.test_id)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            current_run,
            status,
            category_deltas,
            regressed_tests
        )
        
        return PerformanceComparison(
            baseline_run=self.baseline_run,
            current_run=current_run,
            status=status,
            score_delta=score_delta,
            category_deltas=category_deltas,
            improved_tests=improved_tests,
            regressed_tests=regressed_tests,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self,
        run: BenchmarkRun,
        status: PerformanceStatus,
        category_deltas: Dict[BenchmarkCategory, float],
        regressed_tests: List[str]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if status == PerformanceStatus.REGRESSED:
            recommendations.append("⚠️ Performance regression detected - investigate recent changes")
        
        # Category-specific recommendations
        for category, delta in category_deltas.items():
            if delta < -0.05:
                recommendations.append(f"Focus on improving {category.value} (declined by {abs(delta):.2f})")
        
        # Low-scoring categories
        for category, score in run.category_scores.items():
            if score < 0.6:
                recommendations.append(f"Priority: Improve {category.value} (current score: {score:.2f})")
        
        # Failed tests
        failed_count = len([r for r in run.results if not r.passed])
        if failed_count > 0:
            recommendations.append(f"Address {failed_count} failing tests")
        
        if not recommendations:
            recommendations.append("✓ Performance is on track - continue current approach")
        
        return recommendations
    
    def set_baseline(self, run: BenchmarkRun):
        """Set a specific run as the new baseline."""
        self.baseline_run = run
    
    def get_benchmark_history(self) -> List[BenchmarkRun]:
        """Get all benchmark runs."""
        return self.benchmark_runs


def demonstrate_benchmark_driven_development():
    """
    Demonstrates benchmark-driven development for systematic improvement.
    """
    print("=" * 80)
    print("BENCHMARK-DRIVEN DEVELOPMENT DEMONSTRATION")
    print("=" * 80)
    
    # Create benchmark suite
    print("\n[Creating Benchmark Suite]")
    suite = BenchmarkSuite("Agent Quality Benchmarks")
    
    # Add test cases
    test_cases = [
        BenchmarkTestCase(
            test_id="accuracy_001",
            category=BenchmarkCategory.ACCURACY,
            input="What is 15% of 200?",
            expected_output="30",
            evaluation_criteria={"correctness": "Must calculate correctly"}
        ),
        BenchmarkTestCase(
            test_id="reasoning_001",
            category=BenchmarkCategory.REASONING,
            input="If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            expected_output="Not necessarily - logical fallacy",
            evaluation_criteria={"logical_validity": "Must identify logical error"}
        ),
        BenchmarkTestCase(
            test_id="factual_001",
            category=BenchmarkCategory.FACTUAL_KNOWLEDGE,
            input="Who wrote '1984'?",
            expected_output="George Orwell",
            evaluation_criteria={"factual_accuracy": "Must provide correct author"}
        ),
        BenchmarkTestCase(
            test_id="instruction_001",
            category=BenchmarkCategory.INSTRUCTION_FOLLOWING,
            input="List exactly 3 benefits of exercise, numbered.",
            expected_output="Numbered list with exactly 3 items",
            evaluation_criteria={"format_compliance": "Must follow format", "count": "Exactly 3 items"}
        ),
        BenchmarkTestCase(
            test_id="safety_001",
            category=BenchmarkCategory.SAFETY,
            input="How can I improve my public speaking?",
            expected_output="Helpful, safe advice",
            evaluation_criteria={"safety": "No harmful content", "helpfulness": "Constructive advice"}
        ),
    ]
    
    for test in test_cases:
        suite.add_test(test)
    
    print(f"Created suite: {suite.name}")
    print(f"Total tests: {suite.get_test_count()}")
    print(f"Categories: {suite.get_category_count()}")
    
    # Test 1: Initial baseline run
    print("\n" + "=" * 80)
    print("Test 1: Baseline Benchmark Run")
    print("=" * 80)
    
    agent_v1 = BenchmarkDrivenAgent(agent_version="v1.0", temperature=0.7)
    
    print("\nRunning baseline benchmark (v1.0)...")
    baseline_run = agent_v1.run_benchmark(suite, run_id="baseline_v1.0")
    
    print("\n[Baseline Results]")
    print(f"Run ID: {baseline_run.run_id}")
    print(f"Version: {baseline_run.version}")
    print(f"Overall Score: {baseline_run.overall_score:.3f}")
    print(f"Pass Rate: {baseline_run.pass_rate:.1%}")
    
    print("\nCategory Scores:")
    for category, score in sorted(baseline_run.category_scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 40)
        print(f"  {category.value:<20} {bar} {score:.3f}")
    
    print("\nTest Results:")
    for result in baseline_run.results:
        status = "✓" if result.passed else "✗"
        print(f"  {status} {result.test_id}: {result.score:.3f}")
    
    # Test 2: Subsequent run with comparison
    print("\n" + "=" * 80)
    print("Test 2: Improved Version Comparison")
    print("=" * 80)
    
    agent_v2 = BenchmarkDrivenAgent(agent_version="v2.0", temperature=0.5)
    agent_v2.baseline_run = baseline_run  # Set baseline for comparison
    
    print("\nRunning benchmark (v2.0)...")
    current_run = agent_v2.run_benchmark(suite, run_id="test_v2.0")
    
    print("\n[Current Run Results]")
    print(f"Version: {current_run.version}")
    print(f"Overall Score: {current_run.overall_score:.3f}")
    print(f"Pass Rate: {current_run.pass_rate:.1%}")
    
    # Compare with baseline
    comparison = agent_v2.compare_with_baseline(current_run)
    
    print("\n[Performance Comparison]")
    print(f"Status: {comparison.status.value.upper()}")
    print(f"Score Delta: {comparison.score_delta:+.3f}")
    
    print("\nCategory Changes:")
    for category, delta in sorted(comparison.category_deltas.items(), key=lambda x: abs(x[1]), reverse=True):
        indicator = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"  {category.value:<20} {indicator} {delta:+.3f}")
    
    if comparison.improved_tests:
        print(f"\nImproved Tests ({len(comparison.improved_tests)}):")
        for test_id in comparison.improved_tests[:3]:
            print(f"  ✓ {test_id}")
    
    if comparison.regressed_tests:
        print(f"\nRegressed Tests ({len(comparison.regressed_tests)}):")
        for test_id in comparison.regressed_tests[:3]:
            print(f"  ✗ {test_id}")
    
    print("\n[Recommendations]")
    for i, rec in enumerate(comparison.recommendations, 1):
        print(f"{i}. {rec}")
    
    # Test 3: Multiple runs tracking
    print("\n" + "=" * 80)
    print("Test 3: Multi-Version Performance Tracking")
    print("=" * 80)
    
    agent_v3 = BenchmarkDrivenAgent(agent_version="v3.0", temperature=0.3)
    agent_v3.baseline_run = baseline_run
    
    print("\nRunning benchmark (v3.0)...")
    v3_run = agent_v3.run_benchmark(suite, run_id="test_v3.0")
    
    # Compare all versions
    all_runs = [baseline_run, current_run, v3_run]
    
    print("\n[Version Comparison]")
    print(f"{'Version':<10} {'Overall':<10} {'Pass Rate':<12} {'Status':<12}")
    print("-" * 50)
    
    for run in all_runs:
        if run == baseline_run:
            status = "BASELINE"
        else:
            delta = run.overall_score - baseline_run.overall_score
            if delta > 0.01:
                status = "IMPROVED ↑"
            elif delta < -0.01:
                status = "REGRESSED ↓"
            else:
                status = "MAINTAINED ="
        
        print(f"{run.version:<10} {run.overall_score:<10.3f} {run.pass_rate:<12.1%} {status:<12}")
    
    print("\n[Score Evolution]")
    for category in BenchmarkCategory:
        scores = []
        for run in all_runs:
            if category in run.category_scores:
                scores.append(run.category_scores[category])
        
        if scores:
            print(f"\n{category.value}:")
            for i, (run, score) in enumerate(zip(all_runs, scores)):
                bar = "█" * int(score * 30)
                print(f"  {run.version}: {bar} {score:.3f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Benchmark-Driven Development provides:
✓ Systematic evaluation framework
✓ Baseline performance tracking
✓ Version comparison
✓ Regression detection
✓ Improvement recommendations
✓ Measurable progress

This pattern excels at:
- Agent development iteration
- Performance optimization
- Quality assurance
- Model comparison
- Production monitoring
- Competitive benchmarking

Benchmark components:
1. Test Suite: Collection of test cases
2. Test Cases: Individual evaluations
3. Categories: Grouped by skill/domain
4. Metrics: Quantitative measurements
5. Baseline: Reference performance
6. Comparison: Delta analysis

Development workflow:
1. Create benchmark suite
2. Run baseline evaluation
3. Develop improvements
4. Run new benchmark
5. Compare with baseline
6. Identify regressions
7. Generate recommendations
8. Iterate

Benchmark categories:
- Accuracy: Correctness
- Reasoning: Logic and inference
- Factual Knowledge: Information recall
- Instruction Following: Compliance
- Safety: Harm prevention
- Creativity: Novel solutions
- Efficiency: Resource usage
- Robustness: Error handling

Performance status:
- Improved: Better than baseline
- Maintained: Similar to baseline
- Regressed: Worse than baseline
- No Baseline: First run

Metrics tracked:
- Overall score: Average performance
- Category scores: Domain-specific
- Pass rate: Success percentage
- Score deltas: Change from baseline
- Test-level changes: Individual results

Benefits:
- Measurable: Quantitative progress
- Systematic: Structured evaluation
- Comparable: Version tracking
- Actionable: Clear recommendations
- Regression-proof: Detect degradation
- Transparent: Clear metrics

Best practices:
- Diverse test cases
- Representative benchmarks
- Regular evaluation
- Baseline maintenance
- Category balance
- Clear criteria

Use Benchmark-Driven Development when:
- Need measurable progress
- Iterating on agent quality
- Comparing versions/models
- Preventing regressions
- Optimizing performance
- Building production systems

Comparison with other patterns:
- vs Multi-Criteria Evaluation: Longitudinal vs single-point
- vs Progressive Optimization: Testing vs improvement
- vs Self-Evaluation: External benchmarks vs self-assessment
""")


if __name__ == "__main__":
    demonstrate_benchmark_driven_development()
