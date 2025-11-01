"""
Pattern 091: Golden Dataset Testing

Description:
    Golden Dataset Testing involves using curated, high-quality test datasets with known
    expected outputs to systematically evaluate agent performance. These "golden" examples
    serve as ground truth for validating agent behavior, detecting regressions, and
    measuring improvements across different versions or configurations. This pattern is
    essential for maintaining quality in production AI systems.

    Golden datasets provide:
    - Reproducible test cases for consistent evaluation
    - Regression detection across code changes
    - Performance benchmarking and comparison
    - Quality assurance for production deployments
    - Documentation of expected behavior
    - Training data for evaluation models

Components:
    1. Golden Dataset Structure
       - Input examples (queries, prompts, contexts)
       - Expected outputs (answers, actions, decisions)
       - Metadata (difficulty, category, source)
       - Evaluation criteria (metrics, thresholds)
       - Test case descriptions

    2. Test Categories
       - Functional tests (correctness)
       - Edge cases (boundary conditions)
       - Error handling (invalid inputs)
       - Performance tests (speed, cost)
       - Regression tests (historical issues)

    3. Evaluation Metrics
       - Exact match accuracy
       - Semantic similarity
       - BLEU/ROUGE scores (text generation)
       - F1 score (classification)
       - Task-specific metrics

    4. Test Management
       - Dataset versioning
       - Test case prioritization
       - Result tracking and reporting
       - Historical comparison
       - Failure analysis

Use Cases:
    1. Regression Testing
       - Detect performance degradation
       - Validate bug fixes
       - Ensure consistent behavior
       - Track improvements over time
       - Compare model versions

    2. Quality Assurance
       - Pre-deployment validation
       - Continuous integration testing
       - Release quality gates
       - Performance monitoring
       - Compliance verification

    3. Model Evaluation
       - Compare different models
       - Evaluate prompt variations
       - Test configuration changes
       - Measure improvements
       - Benchmark performance

    4. Development Workflow
       - TDD (Test-Driven Development)
       - Rapid iteration feedback
       - Feature validation
       - Debug assistance
       - Documentation

    5. Production Monitoring
       - Ongoing quality checks
       - Canary testing
       - Shadow mode validation
       - A/B test evaluation
       - Incident investigation

LangChain Implementation:
    LangChain supports testing through:
    - Custom evaluation chains
    - Comparison evaluators
    - String distance metrics
    - LangSmith for test tracking
    - Dataset management utilities

Key Features:
    1. Comprehensive Test Coverage
       - Multiple difficulty levels
       - Diverse test categories
       - Edge case coverage
       - Error scenarios
       - Performance tests

    2. Automated Evaluation
       - Exact match checking
       - Semantic similarity (embeddings)
       - Custom scoring functions
       - Multi-metric evaluation
       - Threshold-based pass/fail

    3. Result Analysis
       - Detailed failure reports
       - Performance statistics
       - Trend analysis
       - Comparison views
       - Root cause identification

    4. Dataset Management
       - Easy addition of test cases
       - Categorization and tagging
       - Version control
       - Import/export capabilities
       - Collaborative curation

Best Practices:
    1. Dataset Quality
       - Curate carefully (quality over quantity)
       - Include diverse examples
       - Cover edge cases
       - Regular updates
       - Community validation

    2. Test Organization
       - Clear categorization
       - Descriptive naming
       - Priority levels
       - Dependency tracking
       - Logical grouping

    3. Evaluation Strategy
       - Multiple metrics
       - Appropriate thresholds
       - Context-aware scoring
       - Human validation for edge cases
       - Continuous calibration

    4. Maintenance
       - Regular review and updates
       - Remove obsolete tests
       - Add new failure cases
       - Update expected outputs
       - Document changes

Trade-offs:
    Advantages:
    - Reproducible testing
    - Early issue detection
    - Objective quality measurement
    - Regression prevention
    - Development confidence
    - Clear success criteria

    Disadvantages:
    - Dataset curation effort
    - Maintenance overhead
    - May not cover all scenarios
    - Can become outdated
    - Evaluation complexity
    - False positive/negative risks

Production Considerations:
    1. Dataset Coverage
       - Representative samples
       - Stratified by difficulty
       - Cover common patterns
       - Include rare cases
       - Balance across categories

    2. Evaluation Pipeline
       - Automated test runs
       - CI/CD integration
       - Fast feedback loops
       - Parallel execution
       - Result aggregation

    3. Metrics and Thresholds
       - Task-appropriate metrics
       - Calibrated thresholds
       - Multiple evaluation criteria
       - Weighted scoring
       - Context-specific rules

    4. Monitoring and Alerting
       - Track pass rates over time
       - Alert on regressions
       - Performance trends
       - Failure patterns
       - Quality dashboards

    5. Continuous Improvement
       - Add failing production cases
       - Update based on feedback
       - Refine evaluation criteria
       - Expand coverage
       - Community contributions
"""

import os
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class TestCategory(Enum):
    """Test case categories"""
    FUNCTIONAL = "functional"
    EDGE_CASE = "edge_case"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE = "performance"
    REGRESSION = "regression"


class Difficulty(Enum):
    """Test difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class GoldenTestCase:
    """Represents a single golden test case"""
    id: str
    input: str
    expected_output: str
    category: TestCategory
    difficulty: Difficulty
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Result of a single test case execution"""
    test_id: str
    passed: bool
    actual_output: str
    expected_output: str
    score: float
    execution_time: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResults:
    """Aggregated results for a test suite"""
    total_tests: int
    passed: int
    failed: int
    pass_rate: float
    average_score: float
    total_time: float
    results: List[TestResult]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": f"{self.pass_rate:.2f}%",
            "average_score": f"{self.average_score:.3f}",
            "total_time": f"{self.total_time:.2f}s",
            "timestamp": self.timestamp.isoformat()
        }


class GoldenDataset:
    """
    Manages a golden dataset of test cases.
    
    Provides methods for adding, retrieving, and organizing test cases.
    """
    
    def __init__(self):
        """Initialize golden dataset"""
        self.test_cases: List[GoldenTestCase] = []
    
    def add_test_case(
        self,
        id: str,
        input: str,
        expected_output: str,
        category: TestCategory,
        difficulty: Difficulty,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Add a test case to the dataset.
        
        Args:
            id: Unique test case identifier
            input: Test input
            expected_output: Expected output
            category: Test category
            difficulty: Difficulty level
            description: Test description
            metadata: Additional metadata
            tags: Test tags
        """
        test_case = GoldenTestCase(
            id=id,
            input=input,
            expected_output=expected_output,
            category=category,
            difficulty=difficulty,
            description=description,
            metadata=metadata or {},
            tags=tags or []
        )
        self.test_cases.append(test_case)
    
    def get_by_category(self, category: TestCategory) -> List[GoldenTestCase]:
        """Get test cases by category"""
        return [tc for tc in self.test_cases if tc.category == category]
    
    def get_by_difficulty(self, difficulty: Difficulty) -> List[GoldenTestCase]:
        """Get test cases by difficulty"""
        return [tc for tc in self.test_cases if tc.difficulty == difficulty]
    
    def get_by_tag(self, tag: str) -> List[GoldenTestCase]:
        """Get test cases by tag"""
        return [tc for tc in self.test_cases if tag in tc.tags]
    
    def get_all(self) -> List[GoldenTestCase]:
        """Get all test cases"""
        return self.test_cases
    
    def size(self) -> int:
        """Get dataset size"""
        return len(self.test_cases)


class Evaluator:
    """
    Evaluates agent outputs against expected outputs.
    
    Supports multiple evaluation strategies and metrics.
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.embeddings = OpenAIEmbeddings()
    
    def exact_match(self, actual: str, expected: str) -> float:
        """
        Exact match evaluation.
        
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return 1.0 if actual.strip().lower() == expected.strip().lower() else 0.0
    
    def contains_match(self, actual: str, expected: str) -> float:
        """
        Check if expected output is contained in actual output.
        
        Returns:
            1.0 if expected is in actual, 0.0 otherwise
        """
        return 1.0 if expected.lower() in actual.lower() else 0.0
    
    def semantic_similarity(self, actual: str, expected: str) -> float:
        """
        Compute semantic similarity using embeddings.
        
        Returns:
            Similarity score (0-1)
        """
        # Simple word overlap similarity (production would use actual embeddings)
        actual_words = set(actual.lower().split())
        expected_words = set(expected.lower().split())
        
        if not actual_words or not expected_words:
            return 0.0
        
        intersection = actual_words.intersection(expected_words)
        union = actual_words.union(expected_words)
        
        return len(intersection) / len(union)
    
    def fuzzy_match(self, actual: str, expected: str, threshold: float = 0.8) -> float:
        """
        Fuzzy string matching with threshold.
        
        Args:
            actual: Actual output
            expected: Expected output
            threshold: Similarity threshold
            
        Returns:
            1.0 if similarity >= threshold, else similarity score
        """
        similarity = self.semantic_similarity(actual, expected)
        return 1.0 if similarity >= threshold else similarity
    
    def evaluate(
        self,
        actual: str,
        expected: str,
        method: str = "semantic"
    ) -> float:
        """
        Evaluate using specified method.
        
        Args:
            actual: Actual output
            expected: Expected output
            method: Evaluation method (exact, contains, semantic, fuzzy)
            
        Returns:
            Score (0-1)
        """
        if method == "exact":
            return self.exact_match(actual, expected)
        elif method == "contains":
            return self.contains_match(actual, expected)
        elif method == "semantic":
            return self.semantic_similarity(actual, expected)
        elif method == "fuzzy":
            return self.fuzzy_match(actual, expected)
        else:
            return self.semantic_similarity(actual, expected)


class GoldenDatasetTester:
    """
    Tests an agent using a golden dataset.
    
    Runs test cases, evaluates results, and generates reports.
    """
    
    def __init__(
        self,
        agent_fn: Callable[[str], str],
        evaluator: Evaluator,
        pass_threshold: float = 0.8
    ):
        """
        Initialize tester.
        
        Args:
            agent_fn: Function that takes input and returns output
            evaluator: Evaluator instance
            pass_threshold: Minimum score to pass
        """
        self.agent_fn = agent_fn
        self.evaluator = evaluator
        self.pass_threshold = pass_threshold
    
    def run_test_case(
        self,
        test_case: GoldenTestCase,
        eval_method: str = "semantic"
    ) -> TestResult:
        """
        Run a single test case.
        
        Args:
            test_case: Test case to run
            eval_method: Evaluation method
            
        Returns:
            Test result
        """
        import time
        
        try:
            start_time = time.time()
            actual_output = self.agent_fn(test_case.input)
            execution_time = time.time() - start_time
            
            score = self.evaluator.evaluate(
                actual_output,
                test_case.expected_output,
                method=eval_method
            )
            
            passed = score >= self.pass_threshold
            
            return TestResult(
                test_id=test_case.id,
                passed=passed,
                actual_output=actual_output,
                expected_output=test_case.expected_output,
                score=score,
                execution_time=execution_time,
                details={
                    "category": test_case.category.value,
                    "difficulty": test_case.difficulty.value,
                    "eval_method": eval_method
                }
            )
            
        except Exception as e:
            return TestResult(
                test_id=test_case.id,
                passed=False,
                actual_output="",
                expected_output=test_case.expected_output,
                score=0.0,
                execution_time=0.0,
                error=str(e)
            )
    
    def run_test_suite(
        self,
        dataset: GoldenDataset,
        eval_method: str = "semantic"
    ) -> TestSuiteResults:
        """
        Run entire test suite.
        
        Args:
            dataset: Golden dataset
            eval_method: Evaluation method
            
        Returns:
            Test suite results
        """
        results = []
        
        for test_case in dataset.get_all():
            result = self.run_test_case(test_case, eval_method)
            results.append(result)
        
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        pass_rate = (passed / len(results) * 100) if results else 0.0
        average_score = sum(r.score for r in results) / len(results) if results else 0.0
        total_time = sum(r.execution_time for r in results)
        
        return TestSuiteResults(
            total_tests=len(results),
            passed=passed,
            failed=failed,
            pass_rate=pass_rate,
            average_score=average_score,
            total_time=total_time,
            results=results
        )
    
    def generate_report(self, results: TestSuiteResults) -> str:
        """
        Generate detailed test report.
        
        Args:
            results: Test suite results
            
        Returns:
            Formatted report
        """
        report = []
        report.append("=" * 80)
        report.append("GOLDEN DATASET TEST REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {results.timestamp}")
        report.append(f"Total Tests: {results.total_tests}")
        report.append(f"Passed: {results.passed}")
        report.append(f"Failed: {results.failed}")
        report.append(f"Pass Rate: {results.pass_rate:.2f}%")
        report.append(f"Average Score: {results.average_score:.3f}")
        report.append(f"Total Time: {results.total_time:.2f}s")
        report.append("")
        
        if results.failed > 0:
            report.append("FAILED TESTS:")
            report.append("-" * 80)
            for result in results.results:
                if not result.passed:
                    report.append(f"\nTest ID: {result.test_id}")
                    report.append(f"Score: {result.score:.3f}")
                    report.append(f"Expected: {result.expected_output[:100]}...")
                    report.append(f"Actual: {result.actual_output[:100]}...")
                    if result.error:
                        report.append(f"Error: {result.error}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def demonstrate_golden_dataset_testing():
    """Demonstrate golden dataset testing"""
    print("=" * 80)
    print("GOLDEN DATASET TESTING DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Create Golden Dataset
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Creating a Golden Dataset")
    print("=" * 80)
    
    dataset = GoldenDataset()
    
    # Add test cases
    dataset.add_test_case(
        id="test_001",
        input="What is 2 + 2?",
        expected_output="4",
        category=TestCategory.FUNCTIONAL,
        difficulty=Difficulty.EASY,
        description="Basic arithmetic",
        tags=["math", "basic"]
    )
    
    dataset.add_test_case(
        id="test_002",
        input="What is the capital of France?",
        expected_output="Paris",
        category=TestCategory.FUNCTIONAL,
        difficulty=Difficulty.EASY,
        description="Geography question",
        tags=["geography", "factual"]
    )
    
    dataset.add_test_case(
        id="test_003",
        input="Explain quantum computing in simple terms",
        expected_output="Quantum computing uses quantum mechanics principles like superposition and entanglement",
        category=TestCategory.FUNCTIONAL,
        difficulty=Difficulty.HARD,
        description="Complex explanation",
        tags=["science", "explanation"]
    )
    
    dataset.add_test_case(
        id="test_004",
        input="",
        expected_output="I need a question to answer",
        category=TestCategory.EDGE_CASE,
        difficulty=Difficulty.MEDIUM,
        description="Empty input handling",
        tags=["edge_case", "validation"]
    )
    
    print(f"\nCreated golden dataset with {dataset.size()} test cases")
    print("\nTest cases by category:")
    for category in TestCategory:
        cases = dataset.get_by_category(category)
        print(f"  {category.value}: {len(cases)} tests")
    
    print("\nTest cases by difficulty:")
    for difficulty in Difficulty:
        cases = dataset.get_by_difficulty(difficulty)
        print(f"  {difficulty.value}: {len(cases)} tests")
    
    # Example 2: Simple Agent for Testing
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Creating Test Agent")
    print("=" * 80)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    def simple_agent(input: str) -> str:
        """Simple agent for testing"""
        if not input.strip():
            return "I need a question to answer"
        
        prompt = ChatPromptTemplate.from_template(
            "Answer the following question concisely:\n\n{question}"
        )
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"question": input})
    
    print("\nAgent created: simple_agent")
    print("Agent behavior: Answers questions concisely")
    
    # Example 3: Evaluation Methods
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Testing Different Evaluation Methods")
    print("=" * 80)
    
    evaluator = Evaluator()
    
    test_actual = "The capital of France is Paris"
    test_expected = "Paris"
    
    print(f"\nActual: '{test_actual}'")
    print(f"Expected: '{test_expected}'\n")
    
    methods = ["exact", "contains", "semantic", "fuzzy"]
    for method in methods:
        score = evaluator.evaluate(test_actual, test_expected, method=method)
        print(f"{method.upper()} match: {score:.3f}")
    
    # Example 4: Running Test Suite
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Running Complete Test Suite")
    print("=" * 80)
    
    tester = GoldenDatasetTester(
        agent_fn=simple_agent,
        evaluator=evaluator,
        pass_threshold=0.7
    )
    
    print("\nRunning test suite...")
    print(f"Evaluation method: semantic similarity")
    print(f"Pass threshold: 0.7\n")
    
    results = tester.run_test_suite(dataset, eval_method="semantic")
    
    print(f"Results:")
    print(f"  Total: {results.total_tests}")
    print(f"  Passed: {results.passed}")
    print(f"  Failed: {results.failed}")
    print(f"  Pass Rate: {results.pass_rate:.2f}%")
    print(f"  Avg Score: {results.average_score:.3f}")
    print(f"  Total Time: {results.total_time:.2f}s")
    
    # Example 5: Detailed Results Analysis
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Detailed Results Analysis")
    print("=" * 80)
    
    print("\nIndividual Test Results:\n")
    for result in results.results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status} - {result.test_id}")
        print(f"  Score: {result.score:.3f}")
        print(f"  Time: {result.execution_time:.3f}s")
        print(f"  Category: {result.details.get('category', 'N/A')}")
        print(f"  Difficulty: {result.details.get('difficulty', 'N/A')}")
        if not result.passed:
            print(f"  Expected: {result.expected_output[:50]}...")
            print(f"  Actual: {result.actual_output[:50]}...")
        print()
    
    # Example 6: Category-Based Analysis
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Performance by Category")
    print("=" * 80)
    
    category_stats = {}
    for result in results.results:
        category = result.details.get('category', 'unknown')
        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0, "scores": []}
        
        category_stats[category]["total"] += 1
        if result.passed:
            category_stats[category]["passed"] += 1
        category_stats[category]["scores"].append(result.score)
    
    print("\nPerformance by Category:\n")
    for category, stats in category_stats.items():
        pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
        
        print(f"{category.upper()}:")
        print(f"  Tests: {stats['total']}")
        print(f"  Passed: {stats['passed']}/{stats['total']}")
        print(f"  Pass Rate: {pass_rate:.2f}%")
        print(f"  Avg Score: {avg_score:.3f}")
        print()
    
    # Example 7: Regression Testing
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Regression Testing Simulation")
    print("=" * 80)
    
    print("\nSimulating version comparison...")
    
    # Simulate baseline results
    baseline_pass_rate = 85.0
    current_pass_rate = results.pass_rate
    
    print(f"\nBaseline (v1.0): {baseline_pass_rate:.2f}% pass rate")
    print(f"Current (v2.0): {current_pass_rate:.2f}% pass rate")
    
    if current_pass_rate >= baseline_pass_rate:
        print("\n✓ No regression detected - Quality maintained or improved!")
    else:
        regression = baseline_pass_rate - current_pass_rate
        print(f"\n✗ REGRESSION DETECTED: {regression:.2f}% decrease in pass rate")
        print("   Action: Investigate failing tests before deployment")
    
    # Example 8: Test Report Generation
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Generating Test Report")
    print("=" * 80)
    
    report = tester.generate_report(results)
    print("\n" + report)
    
    # Summary
    print("\n" + "=" * 80)
    print("GOLDEN DATASET TESTING SUMMARY")
    print("=" * 80)
    print("""
Golden Dataset Testing Benefits:
1. Reproducibility: Consistent test results across runs
2. Regression Detection: Catch performance degradation early
3. Quality Assurance: Objective measurement of agent quality
4. Development Confidence: Safe to refactor and improve
5. Documentation: Tests document expected behavior
6. Benchmarking: Compare different configurations

Dataset Best Practices:
1. Quality Over Quantity
   - Carefully curate test cases
   - Focus on important scenarios
   - Include edge cases
   - Regular review and updates

2. Organization
   - Clear categories (functional, edge case, etc.)
   - Difficulty levels (easy, medium, hard)
   - Descriptive tags
   - Comprehensive metadata

3. Coverage
   - Common use cases
   - Edge cases and boundaries
   - Error scenarios
   - Performance critical paths
   - Regression cases from production

4. Maintenance
   - Add failing production cases
   - Update expected outputs when needed
   - Remove obsolete tests
   - Version control dataset
   - Document changes

Evaluation Methods:
1. Exact Match
   - Strict equality check
   - Best for: Factual answers, specific outputs
   - Limitation: Too strict for natural language

2. Contains Match
   - Check if expected is in actual
   - Best for: Key information presence
   - Limitation: Doesn't verify quality

3. Semantic Similarity
   - Meaning-based comparison
   - Best for: Natural language responses
   - Limitation: Computationally expensive

4. Fuzzy Match
   - Similarity with threshold
   - Best for: Near-match acceptance
   - Limitation: Threshold tuning needed

Test Suite Structure:
1. Core Functionality (60%)
   - Essential features
   - Common use cases
   - Critical paths

2. Edge Cases (20%)
   - Boundary conditions
   - Unusual inputs
   - Corner cases

3. Error Handling (10%)
   - Invalid inputs
   - System errors
   - Recovery scenarios

4. Performance (10%)
   - Speed benchmarks
   - Resource usage
   - Scalability tests

Continuous Integration:
1. Automated Test Runs
   - Run on every commit
   - Pre-deployment validation
   - Scheduled regression tests
   - Performance monitoring

2. Quality Gates
   - Minimum pass rate (e.g., 95%)
   - Maximum regression (e.g., 5%)
   - Performance thresholds
   - Coverage requirements

3. Reporting
   - Dashboard visualization
   - Trend analysis
   - Failure notifications
   - Historical comparison

When to Use Golden Dataset Testing:
✓ Production AI systems
✓ Iterative development
✓ Multiple team members
✓ Complex agent behaviors
✓ Compliance requirements
✓ Performance monitoring
✗ One-off scripts
✗ Exploratory prototypes (initially)

Production Tips:
- Start small, grow dataset organically
- Add production failures to dataset
- Regular dataset review sessions
- Version control your datasets
- Share datasets across team
- Integrate with CI/CD pipeline
- Monitor test execution time
- Keep tests fast (< 5 min for suite)
- Parallel execution for large datasets
- Alert on regressions immediately
""")
    
    print("\n" + "=" * 80)
    print("Pattern 091 (Golden Dataset Testing) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_golden_dataset_testing()
