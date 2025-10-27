"""
Golden Dataset Testing Pattern for Agentic AI

This pattern implements systematic testing of agents using curated test datasets
with known correct answers. Essential for regression testing, quality assurance,
and continuous evaluation of agent performance.

Key Concepts:
1. Golden Dataset - Curated test cases with verified correct answers
2. Regression Testing - Ensure changes don't break existing functionality
3. Performance Benchmarking - Track metrics over time
4. Coverage Analysis - Ensure test diversity
5. Automated Evaluation - Systematic quality assessment

Use Cases:
- Agent development and validation
- Regression testing before deployment
- Performance tracking across versions
- Quality assurance in production
- Model comparison and selection
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class TestStatus(Enum):
    """Status of test execution"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class MetricType(Enum):
    """Types of evaluation metrics"""
    EXACT_MATCH = "exact_match"
    CONTAINS = "contains"
    SIMILARITY = "similarity"
    CUSTOM = "custom"


@dataclass
class TestCase:
    """Represents a single test case"""
    id: str
    input: Any
    expected_output: Any
    category: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of executing a test case"""
    test_id: str
    status: TestStatus
    actual_output: Optional[Any] = None
    expected_output: Optional[Any] = None
    score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestSuite:
    """Collection of test cases"""
    name: str
    version: str
    test_cases: List[TestCase]
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def filter_by_category(self, category: str) -> List[TestCase]:
        """Filter test cases by category"""
        return [tc for tc in self.test_cases if tc.category == category]
    
    def filter_by_tag(self, tag: str) -> List[TestCase]:
        """Filter test cases by tag"""
        return [tc for tc in self.test_cases if tag in tc.tags]


class ExactMatchEvaluator:
    """Evaluates using exact string match"""
    
    def evaluate(self, actual: Any, expected: Any) -> float:
        """Return 1.0 if exact match, 0.0 otherwise"""
        return 1.0 if str(actual).strip() == str(expected).strip() else 0.0


class ContainsEvaluator:
    """Evaluates if expected string is contained in actual"""
    
    def evaluate(self, actual: Any, expected: Any) -> float:
        """Return 1.0 if expected is in actual, 0.0 otherwise"""
        return 1.0 if str(expected).lower() in str(actual).lower() else 0.0


class SimilarityEvaluator:
    """Evaluates semantic similarity (simplified)"""
    
    def evaluate(self, actual: Any, expected: Any) -> float:
        """Simple word overlap similarity"""
        actual_words = set(str(actual).lower().split())
        expected_words = set(str(expected).lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(actual_words & expected_words)
        return overlap / len(expected_words)


class GoldenDatasetTester:
    """
    Systematic testing framework using golden datasets.
    
    Executes test suites, evaluates results, and provides detailed reporting.
    """
    
    def __init__(
        self,
        name: str = "GoldenDatasetTester",
        default_evaluator: str = "exact_match"
    ):
        self.name = name
        self.default_evaluator = default_evaluator
        self.test_results: List[TestResult] = []
        self.test_suites: Dict[str, TestSuite] = {}
        
        # Initialize evaluators
        self.evaluators = {
            "exact_match": ExactMatchEvaluator(),
            "contains": ContainsEvaluator(),
            "similarity": SimilarityEvaluator()
        }
    
    def load_test_suite(
        self,
        suite: TestSuite
    ):
        """Load a test suite"""
        self.test_suites[suite.name] = suite
        print(f"Loaded test suite '{suite.name}' with {len(suite.test_cases)} test cases")
    
    def create_test_suite_from_dict(
        self,
        name: str,
        version: str,
        test_data: List[Dict[str, Any]]
    ) -> TestSuite:
        """Create test suite from dictionary data"""
        test_cases = []
        for item in test_data:
            test_case = TestCase(
                id=item['id'],
                input=item['input'],
                expected_output=item['expected'],
                category=item.get('category', 'general'),
                tags=item.get('tags', []),
                metadata=item.get('metadata', {})
            )
            test_cases.append(test_case)
        
        return TestSuite(
            name=name,
            version=version,
            test_cases=test_cases
        )
    
    def run_test_case(
        self,
        test_case: TestCase,
        agent_function: Callable[[Any], Any],
        evaluator_type: Optional[str] = None
    ) -> TestResult:
        """
        Run a single test case.
        
        Args:
            test_case: Test case to run
            agent_function: Function that processes input and returns output
            evaluator_type: Type of evaluator to use
        
        Returns:
            TestResult with evaluation
        """
        import time
        
        evaluator_type = evaluator_type or self.default_evaluator
        evaluator = self.evaluators.get(evaluator_type)
        
        if not evaluator:
            return TestResult(
                test_id=test_case.id,
                status=TestStatus.ERROR,
                error_message=f"Unknown evaluator: {evaluator_type}"
            )
        
        try:
            # Execute agent
            start_time = time.time()
            actual_output = agent_function(test_case.input)
            execution_time = time.time() - start_time
            
            # Evaluate result
            score = evaluator.evaluate(actual_output, test_case.expected_output)
            status = TestStatus.PASSED if score >= 0.8 else TestStatus.FAILED
            
            result = TestResult(
                test_id=test_case.id,
                status=status,
                actual_output=actual_output,
                expected_output=test_case.expected_output,
                score=score,
                metrics={evaluator_type: score},
                execution_time=execution_time
            )
            
        except Exception as e:
            result = TestResult(
                test_id=test_case.id,
                status=TestStatus.ERROR,
                error_message=str(e)
            )
        
        self.test_results.append(result)
        return result
    
    def run_test_suite(
        self,
        suite_name: str,
        agent_function: Callable[[Any], Any],
        evaluator_type: Optional[str] = None,
        filter_category: Optional[str] = None,
        filter_tags: Optional[List[str]] = None
    ) -> List[TestResult]:
        """
        Run entire test suite.
        
        Args:
            suite_name: Name of test suite to run
            agent_function: Agent function to test
            evaluator_type: Evaluation method
            filter_category: Optional category filter
            filter_tags: Optional tag filters
        
        Returns:
            List of test results
        """
        suite = self.test_suites.get(suite_name)
        if not suite:
            print(f"Test suite '{suite_name}' not found")
            return []
        
        # Filter test cases if needed
        test_cases = suite.test_cases
        if filter_category:
            test_cases = [tc for tc in test_cases if tc.category == filter_category]
        if filter_tags:
            test_cases = [tc for tc in test_cases 
                         if any(tag in tc.tags for tag in filter_tags)]
        
        print(f"\n{'='*70}")
        print(f"Running Test Suite: {suite_name} (v{suite.version})")
        print(f"Test Cases: {len(test_cases)}")
        print(f"{'='*70}\n")
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] Testing: {test_case.id}... ", end="")
            
            result = self.run_test_case(test_case, agent_function, evaluator_type)
            results.append(result)
            
            status_symbol = "✓" if result.status == TestStatus.PASSED else "✗"
            print(f"{status_symbol} {result.status.value} (score: {result.score:.2f})")
        
        return results
    
    def get_summary(
        self,
        results: Optional[List[TestResult]] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics from test results.
        
        Args:
            results: Optional list of results (uses all if None)
        
        Returns:
            Dictionary with summary statistics
        """
        results = results or self.test_results
        
        if not results:
            return {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "error": 0,
                "pass_rate": 0.0,
                "avg_score": 0.0,
                "avg_execution_time": 0.0
            }
        
        total = len(results)
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        error = sum(1 for r in results if r.status == TestStatus.ERROR)
        
        valid_results = [r for r in results if r.status != TestStatus.ERROR]
        avg_score = sum(r.score for r in valid_results) / len(valid_results) if valid_results else 0
        avg_time = sum(r.execution_time for r in results) / total
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "error": error,
            "pass_rate": passed / total if total > 0 else 0,
            "avg_score": avg_score,
            "avg_execution_time": avg_time
        }
    
    def generate_report(
        self,
        results: Optional[List[TestResult]] = None,
        detailed: bool = True
    ) -> str:
        """Generate test report"""
        results = results or self.test_results
        summary = self.get_summary(results)
        
        report = []
        report.append("=" * 70)
        report.append("TEST REPORT")
        report.append("=" * 70)
        report.append(f"Total Tests: {summary['total']}")
        report.append(f"Passed: {summary['passed']} ({summary['pass_rate']:.1%})")
        report.append(f"Failed: {summary['failed']}")
        report.append(f"Errors: {summary['error']}")
        report.append(f"Average Score: {summary['avg_score']:.3f}")
        report.append(f"Average Execution Time: {summary['avg_execution_time']:.3f}s")
        
        if detailed:
            report.append("\n" + "-" * 70)
            report.append("FAILED TESTS")
            report.append("-" * 70)
            
            failed_tests = [r for r in results if r.status == TestStatus.FAILED]
            if failed_tests:
                for result in failed_tests:
                    report.append(f"\nTest ID: {result.test_id}")
                    report.append(f"  Expected: {result.expected_output}")
                    report.append(f"  Actual: {result.actual_output}")
                    report.append(f"  Score: {result.score:.3f}")
            else:
                report.append("(No failed tests)")
        
        return "\n".join(report)
    
    def compare_runs(
        self,
        results1: List[TestResult],
        results2: List[TestResult],
        run1_name: str = "Run 1",
        run2_name: str = "Run 2"
    ) -> Dict[str, Any]:
        """Compare two test runs"""
        summary1 = self.get_summary(results1)
        summary2 = self.get_summary(results2)
        
        return {
            "run1": {
                "name": run1_name,
                **summary1
            },
            "run2": {
                "name": run2_name,
                **summary2
            },
            "improvements": {
                "pass_rate_delta": summary2['pass_rate'] - summary1['pass_rate'],
                "score_delta": summary2['avg_score'] - summary1['avg_score'],
                "time_delta": summary2['avg_execution_time'] - summary1['avg_execution_time']
            }
        }


def demonstrate_golden_dataset_testing():
    """Demonstrate golden dataset testing"""
    
    print("Golden Dataset Testing Pattern Demonstration")
    print("=" * 70)
    
    # Create tester
    tester = GoldenDatasetTester(name="QA_Tester")
    
    # Create sample test data
    test_data = [
        {
            "id": "test_001",
            "input": "What is the capital of France?",
            "expected": "Paris",
            "category": "geography",
            "tags": ["easy", "factual"]
        },
        {
            "id": "test_002",
            "input": "Calculate 15 + 27",
            "expected": "42",
            "category": "math",
            "tags": ["easy", "arithmetic"]
        },
        {
            "id": "test_003",
            "input": "Who wrote Romeo and Juliet?",
            "expected": "Shakespeare",
            "category": "literature",
            "tags": ["medium", "factual"]
        },
        {
            "id": "test_004",
            "input": "What is photosynthesis?",
            "expected": "process plants use to convert light into energy",
            "category": "science",
            "tags": ["medium", "explanatory"]
        },
        {
            "id": "test_005",
            "input": "Name three primary colors",
            "expected": "red blue yellow",
            "category": "art",
            "tags": ["easy", "factual"]
        }
    ]
    
    # Create test suite
    suite = tester.create_test_suite_from_dict(
        name="General Knowledge",
        version="1.0",
        test_data=test_data
    )
    tester.load_test_suite(suite)
    
    # Simulate agent function (simple mock)
    def mock_agent(input_text: str) -> str:
        """Mock agent for testing"""
        responses = {
            "What is the capital of France?": "Paris",
            "Calculate 15 + 27": "42",
            "Who wrote Romeo and Juliet?": "William Shakespeare",
            "What is photosynthesis?": "Photosynthesis is the process by which plants convert light into energy",
            "Name three primary colors": "red, blue, and yellow"
        }
        return responses.get(input_text, "I don't know")
    
    # Run tests with different evaluators
    print("\n\n1. Exact Match Evaluation")
    print("-" * 70)
    results_exact = tester.run_test_suite(
        "General Knowledge",
        mock_agent,
        evaluator_type="exact_match"
    )
    
    print("\n\n2. Contains Evaluation")
    print("-" * 70)
    tester.test_results = []  # Reset
    results_contains = tester.run_test_suite(
        "General Knowledge",
        mock_agent,
        evaluator_type="contains"
    )
    
    print("\n\n3. Similarity Evaluation")
    print("-" * 70)
    tester.test_results = []  # Reset
    results_similarity = tester.run_test_suite(
        "General Knowledge",
        mock_agent,
        evaluator_type="similarity"
    )
    
    # Generate reports
    print("\n\n" + "="*70)
    print("DETAILED REPORTS")
    print("="*70)
    
    print("\n--- Exact Match Results ---")
    print(tester.generate_report(results_exact, detailed=True))
    
    print("\n--- Contains Results ---")
    print(tester.generate_report(results_contains, detailed=False))
    
    print("\n--- Similarity Results ---")
    print(tester.generate_report(results_similarity, detailed=False))
    
    # Compare runs
    print("\n\n" + "="*70)
    print("COMPARISON: Exact Match vs Contains")
    print("="*70)
    comparison = tester.compare_runs(
        results_exact, results_contains,
        "Exact Match", "Contains"
    )
    print(f"Pass Rate Change: {comparison['improvements']['pass_rate_delta']:+.1%}")
    print(f"Score Change: {comparison['improvements']['score_delta']:+.3f}")


if __name__ == "__main__":
    demonstrate_golden_dataset_testing()
    
    print("\n\n" + "="*70)
    print("Golden Dataset Testing Pattern Summary")
    print("="*70)
    print("""
Key Benefits:
1. Systematic quality assurance
2. Regression detection
3. Performance tracking over time
4. Objective evaluation metrics
5. Reproducible testing

Components:
- Test Cases: Input-expected output pairs
- Test Suites: Organized collections of tests
- Evaluators: Methods to assess correctness
- Test Runner: Executes tests systematically
- Reporter: Generates insights and reports

Best Practices:
- Maintain diverse test coverage
- Version control test datasets
- Regular updates to golden set
- Multiple evaluation metrics
- Track performance trends
- Document test rationale
- Automate in CI/CD pipeline

Evaluation Methods:
- Exact Match: Precise string matching
- Contains: Check if key terms present
- Similarity: Semantic/word overlap
- Custom: Domain-specific metrics
- Multi-metric: Combine multiple methods

Use Cases:
- Agent development validation
- Pre-deployment regression testing
- Model version comparison
- Performance benchmarking
- Quality assurance gates
- Continuous evaluation
- A/B testing validation

Tips:
- Start with small curated set
- Gradually expand coverage
- Include edge cases
- Balance difficulty levels
- Update based on failures
- Use multiple evaluators
- Track false positives/negatives
""")
