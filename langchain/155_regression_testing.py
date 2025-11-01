"""
Pattern 155: Regression Testing

Description:
    Automated regression testing detects unintended changes in agent behavior
    when code, models, or prompts are updated. Maintains baselines, compares
    outputs, and flags regressions automatically.

Components:
    - Baseline storage and versioning
    - Test case management
    - Output comparison
    - Regression detection
    - Change analysis
    - Historical tracking

Use Cases:
    - Model version updates
    - Prompt engineering iterations
    - Code refactoring validation
    - CI/CD integration
    - Quality assurance

Benefits:
    - Early regression detection
    - Automated quality checks
    - Historical performance tracking
    - Confidence in updates

Trade-offs:
    - Baseline maintenance overhead
    - Storage requirements
    - Comparison complexity for creative tasks
    - False positive management

LangChain Implementation:
    Uses ChatOpenAI for testing, JSON storage for baselines,
    similarity metrics for comparison, and automated detection
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from difflib import SequenceMatcher
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class ChangeType(Enum):
    """Types of changes detected"""
    NO_CHANGE = "no_change"
    IMPROVEMENT = "improvement"
    REGRESSION = "regression"
    NEUTRAL = "neutral"
    ERROR = "error"


class Severity(Enum):
    """Severity levels for regressions"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class TestCase:
    """Single test case"""
    id: str
    input: str
    category: str
    expected_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of a single test execution"""
    test_id: str
    input: str
    output: str
    timestamp: str
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Comparison:
    """Comparison between baseline and current result"""
    test_id: str
    baseline_output: str
    current_output: str
    similarity_score: float
    change_type: ChangeType
    severity: Severity
    details: str
    timestamp: str


@dataclass
class RegressionReport:
    """Comprehensive regression test report"""
    version: str
    timestamp: str
    total_tests: int
    passed: int
    regressions: int
    improvements: int
    errors: int
    comparisons: List[Comparison]
    summary: str


class RegressionTester:
    """Manages regression testing for agent outputs"""
    
    def __init__(self, baseline_file: str = "baselines.json", 
                 model_name: str = "gpt-3.5-turbo"):
        """
        Initialize regression tester
        
        Args:
            baseline_file: Path to baseline storage
            model_name: LLM model to use for testing
        """
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.baseline_file = baseline_file
        self.test_cases: List[TestCase] = []
        self.baselines: Dict[str, TestResult] = {}
        self.load_baselines()
        
        # Comparison LLM for semantic analysis
        self.comparison_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
    def load_baselines(self):
        """Load baseline results from file"""
        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file, 'r') as f:
                    data = json.load(f)
                    self.baselines = {
                        k: TestResult(**v) for k, v in data.items()
                    }
                print(f"✓ Loaded {len(self.baselines)} baselines from {self.baseline_file}")
            except Exception as e:
                print(f"Error loading baselines: {e}")
                self.baselines = {}
        else:
            print(f"No baseline file found at {self.baseline_file}")
            self.baselines = {}
    
    def save_baselines(self):
        """Save baseline results to file"""
        try:
            data = {
                k: {
                    'test_id': v.test_id,
                    'input': v.input,
                    'output': v.output,
                    'timestamp': v.timestamp,
                    'execution_time': v.execution_time,
                    'error': v.error,
                    'metadata': v.metadata
                }
                for k, v in self.baselines.items()
            }
            with open(self.baseline_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved {len(self.baselines)} baselines to {self.baseline_file}")
        except Exception as e:
            print(f"Error saving baselines: {e}")
    
    def add_test_case(self, test_case: TestCase):
        """Add a test case"""
        self.test_cases.append(test_case)
    
    def run_test(self, test_case: TestCase) -> TestResult:
        """Execute a single test case"""
        import time
        
        start_time = time.time()
        error = None
        output = ""
        
        try:
            # Create prompt for the test
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Provide clear, accurate responses."),
                ("user", "{input}")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            output = chain.invoke({"input": test_case.input})
            
        except Exception as e:
            error = str(e)
            output = f"ERROR: {error}"
        
        execution_time = time.time() - start_time
        
        return TestResult(
            test_id=test_case.id,
            input=test_case.input,
            output=output,
            timestamp=datetime.now().isoformat(),
            execution_time=execution_time,
            error=error,
            metadata=test_case.metadata.copy()
        )
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Use SequenceMatcher for basic similarity
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def analyze_change(self, baseline: TestResult, current: TestResult) -> Comparison:
        """Analyze changes between baseline and current result"""
        similarity = self.calculate_similarity(baseline.output, current.output)
        
        # Determine change type
        if current.error:
            change_type = ChangeType.ERROR
            severity = Severity.CRITICAL
            details = f"Error occurred: {current.error}"
        elif similarity >= 0.95:
            change_type = ChangeType.NO_CHANGE
            severity = Severity.INFO
            details = "Output is virtually identical to baseline"
        else:
            # Use LLM to classify the change
            change_type, severity, details = self._llm_classify_change(
                baseline.output, current.output, similarity
            )
        
        return Comparison(
            test_id=current.test_id,
            baseline_output=baseline.output,
            current_output=current.output,
            similarity_score=similarity,
            change_type=change_type,
            severity=severity,
            details=details,
            timestamp=datetime.now().isoformat()
        )
    
    def _llm_classify_change(self, baseline: str, current: str, 
                           similarity: float) -> tuple[ChangeType, Severity, str]:
        """Use LLM to classify the nature of the change"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing changes in AI outputs.
Compare the baseline and current outputs and determine:
1. Change type: improvement, regression, or neutral
2. Severity: critical, high, medium, low, info
3. Brief explanation

Respond in JSON format: {{"change_type": "...", "severity": "...", "details": "..."}}"""),
            ("user", """Baseline output: {baseline}

Current output: {current}

Similarity score: {similarity:.2f}

Analyze the change:""")
        ])
        
        try:
            chain = prompt | self.comparison_llm | StrOutputParser()
            response = chain.invoke({
                "baseline": baseline,
                "current": current,
                "similarity": similarity
            })
            
            # Parse JSON response
            result = json.loads(response)
            change_type = ChangeType(result.get('change_type', 'neutral'))
            severity = Severity(result.get('severity', 'info'))
            details = result.get('details', 'No details provided')
            
        except Exception as e:
            # Fallback classification
            if similarity < 0.5:
                change_type = ChangeType.REGRESSION
                severity = Severity.HIGH
            elif similarity < 0.8:
                change_type = ChangeType.NEUTRAL
                severity = Severity.MEDIUM
            else:
                change_type = ChangeType.NEUTRAL
                severity = Severity.LOW
            details = f"Automatic classification (similarity: {similarity:.2f})"
        
        return change_type, severity, details
    
    def establish_baseline(self, version: str = "1.0.0"):
        """Run tests and establish baseline"""
        print(f"\n=== Establishing Baseline (Version {version}) ===\n")
        
        results = []
        for test_case in self.test_cases:
            print(f"Running test: {test_case.id}")
            result = self.run_test(test_case)
            results.append(result)
            self.baselines[test_case.id] = result
            print(f"  ✓ Completed in {result.execution_time:.2f}s")
        
        self.save_baselines()
        print(f"\n✓ Baseline established with {len(results)} tests")
        return results
    
    def run_regression_tests(self, version: str = "current") -> RegressionReport:
        """Run all tests and compare against baseline"""
        print(f"\n=== Running Regression Tests (Version {version}) ===\n")
        
        if not self.baselines:
            print("No baseline found. Please establish baseline first.")
            return None
        
        comparisons = []
        passed = 0
        regressions = 0
        improvements = 0
        errors = 0
        
        for test_case in self.test_cases:
            print(f"Testing: {test_case.id}")
            
            # Run current test
            current = self.run_test(test_case)
            
            # Compare with baseline if exists
            if test_case.id in self.baselines:
                baseline = self.baselines[test_case.id]
                comparison = self.analyze_change(baseline, current)
                comparisons.append(comparison)
                
                # Update counters
                if comparison.change_type == ChangeType.NO_CHANGE:
                    passed += 1
                    print(f"  ✓ PASS (no change)")
                elif comparison.change_type == ChangeType.REGRESSION:
                    regressions += 1
                    print(f"  ✗ REGRESSION ({comparison.severity.value}): {comparison.details}")
                elif comparison.change_type == ChangeType.IMPROVEMENT:
                    improvements += 1
                    print(f"  ↑ IMPROVEMENT: {comparison.details}")
                elif comparison.change_type == ChangeType.ERROR:
                    errors += 1
                    print(f"  ⚠ ERROR: {comparison.details}")
                else:
                    passed += 1
                    print(f"  ~ NEUTRAL: {comparison.details}")
            else:
                print(f"  ⚠ No baseline found for this test")
        
        # Generate summary
        total = len(self.test_cases)
        summary = self._generate_summary(passed, regressions, improvements, errors, total)
        
        report = RegressionReport(
            version=version,
            timestamp=datetime.now().isoformat(),
            total_tests=total,
            passed=passed,
            regressions=regressions,
            improvements=improvements,
            errors=errors,
            comparisons=comparisons,
            summary=summary
        )
        
        print(f"\n{summary}")
        return report
    
    def _generate_summary(self, passed: int, regressions: int, 
                         improvements: int, errors: int, total: int) -> str:
        """Generate test summary"""
        lines = [
            "=== Regression Test Summary ===",
            f"Total Tests: {total}",
            f"Passed: {passed} ({passed/total*100:.1f}%)",
            f"Regressions: {regressions} ({regressions/total*100:.1f}%)",
            f"Improvements: {improvements} ({improvements/total*100:.1f}%)",
            f"Errors: {errors} ({errors/total*100:.1f}%)"
        ]
        
        if regressions == 0 and errors == 0:
            lines.append("\n✓ All tests passed! No regressions detected.")
        elif regressions > 0:
            lines.append(f"\n⚠ WARNING: {regressions} regression(s) detected!")
        
        return "\n".join(lines)
    
    def get_regression_details(self, report: RegressionReport) -> str:
        """Get detailed regression information"""
        regressions = [c for c in report.comparisons 
                      if c.change_type == ChangeType.REGRESSION]
        
        if not regressions:
            return "No regressions found."
        
        details = ["\n=== Regression Details ===\n"]
        for i, reg in enumerate(regressions, 1):
            details.append(f"{i}. Test: {reg.test_id}")
            details.append(f"   Severity: {reg.severity.value}")
            details.append(f"   Similarity: {reg.similarity_score:.2%}")
            details.append(f"   Details: {reg.details}")
            details.append(f"   Baseline: {reg.baseline_output[:100]}...")
            details.append(f"   Current:  {reg.current_output[:100]}...")
            details.append("")
        
        return "\n".join(details)


def demonstrate_regression_testing():
    """Demonstrate regression testing pattern"""
    print("=" * 80)
    print("REGRESSION TESTING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Initialize tester
    tester = RegressionTester(baseline_file="test_baselines.json")
    
    # Define test cases
    test_cases = [
        TestCase(
            id="test_001",
            input="What is the capital of France?",
            category="factual"
        ),
        TestCase(
            id="test_002",
            input="Explain photosynthesis in simple terms.",
            category="explanation"
        ),
        TestCase(
            id="test_003",
            input="Write a haiku about coding.",
            category="creative"
        ),
        TestCase(
            id="test_004",
            input="Calculate 15% of 200.",
            category="calculation"
        ),
        TestCase(
            id="test_005",
            input="List three benefits of exercise.",
            category="informational"
        ),
    ]
    
    # Add test cases
    for tc in test_cases:
        tester.add_test_case(tc)
    
    # Example 1: Establish baseline
    print("\n" + "="*80)
    print("EXAMPLE 1: Establishing Baseline")
    print("="*80)
    tester.establish_baseline(version="1.0.0")
    
    # Example 2: Run regression tests (should pass)
    print("\n" + "="*80)
    print("EXAMPLE 2: Running Regression Tests (Same Model)")
    print("="*80)
    report = tester.run_regression_tests(version="1.0.1")
    
    # Example 3: Simulate model change and test
    print("\n" + "="*80)
    print("EXAMPLE 3: Testing with Different Model")
    print("="*80)
    tester_v2 = RegressionTester(baseline_file="test_baselines.json", 
                                  model_name="gpt-3.5-turbo")
    tester_v2.test_cases = test_cases
    tester_v2.load_baselines()
    report_v2 = tester_v2.run_regression_tests(version="2.0.0")
    
    # Example 4: Show regression details
    if report_v2 and report_v2.regressions > 0:
        print("\n" + "="*80)
        print("EXAMPLE 4: Regression Details")
        print("="*80)
        print(tester_v2.get_regression_details(report_v2))
    
    # Example 5: Compare specific test
    print("\n" + "="*80)
    print("EXAMPLE 5: Detailed Comparison for One Test")
    print("="*80)
    test_id = "test_001"
    if report and any(c.test_id == test_id for c in report.comparisons):
        comparison = next(c for c in report.comparisons if c.test_id == test_id)
        print(f"Test ID: {comparison.test_id}")
        print(f"Change Type: {comparison.change_type.value}")
        print(f"Severity: {comparison.severity.value}")
        print(f"Similarity: {comparison.similarity_score:.2%}")
        print(f"Details: {comparison.details}")
        print(f"\nBaseline Output:\n{comparison.baseline_output}")
        print(f"\nCurrent Output:\n{comparison.current_output}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Regression Testing Best Practices")
    print("="*80)
    print("""
1. BASELINE MANAGEMENT:
   - Establish baselines for stable versions
   - Version baselines appropriately
   - Update baselines when intentional changes are made
   - Store baselines in version control

2. TEST CASE DESIGN:
   - Cover diverse input types (factual, creative, calculation)
   - Include edge cases and error scenarios
   - Categorize tests for targeted analysis
   - Keep test cases stable and representative

3. COMPARISON STRATEGIES:
   - Use similarity metrics for objective comparison
   - Employ LLM-based semantic analysis
   - Define severity thresholds
   - Track both regressions and improvements

4. AUTOMATION:
   - Integrate into CI/CD pipeline
   - Run on every code/model change
   - Generate automated reports
   - Alert on critical regressions

5. CONTINUOUS IMPROVEMENT:
   - Review flagged regressions manually
   - Adjust baselines for intentional improvements
   - Expand test coverage over time
   - Track regression trends

Benefits:
✓ Early detection of unintended changes
✓ Confidence in updates and refactoring
✓ Historical performance tracking
✓ Automated quality assurance
✓ Reduced manual testing overhead
✓ Data-driven decision making

Challenges:
- Baseline maintenance for creative outputs
- False positives from legitimate variations
- Storage and comparison overhead
- Defining acceptable change thresholds
    """)


if __name__ == "__main__":
    demonstrate_regression_testing()
