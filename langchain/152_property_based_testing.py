"""
Pattern 152: Property-Based Testing

Description:
    Implements property-based testing for AI agents by defining invariants
    and properties that should hold across all inputs, then generating test
    cases to verify these properties.

Components:
    - Property Definitions: Invariants that should always hold
    - Test Case Generators: Create diverse inputs
    - Property Verifiers: Check if properties hold
    - Hypothesis Engine: Generate test hypotheses
    - Counterexample Finder: Identify violations

Use Cases:
    - Testing agent robustness across input space
    - Finding edge cases and failures
    - Verifying consistency properties
    - Ensuring output format compliance
    - Testing deterministic vs non-deterministic behaviors

Benefits:
    - Discovers unexpected edge cases
    - More thorough than example-based testing
    - Finds bugs traditional testing misses
    - Documents expected behaviors as properties
    - Scales testing coverage automatically

Trade-offs:
    - Requires careful property definition
    - Can be slower than unit tests
    - May generate false positives
    - Harder to debug failures
    - Non-determinism complicates verification

LangChain Implementation:
    Uses ChatOpenAI with property checking chains and systematic
    test case generation.
"""

import os
import re
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class PropertyType(Enum):
    """Types of properties to test"""
    INVARIANT = "invariant"  # Must always hold
    POSTCONDITION = "postcondition"  # Output constraints
    CONSISTENCY = "consistency"  # Same input → same output
    MONOTONICITY = "monotonicity"  # Ordering preservation
    IDEMPOTENCE = "idempotence"  # f(f(x)) = f(x)
    COMMUTATIVE = "commutative"  # Order independence
    FORMAT = "format"  # Output format compliance


@dataclass
class Property:
    """Defines a testable property"""
    name: str
    description: str
    property_type: PropertyType
    checker: Callable[[Any, Any], bool]  # (input, output) -> bool
    examples: List[Tuple[Any, bool]] = field(default_factory=list)  # (input, expected_result)
    
    def check(self, input_val: Any, output_val: Any) -> bool:
        """Check if property holds"""
        return self.checker(input_val, output_val)


@dataclass
class TestResult:
    """Result of a property test"""
    property_name: str
    passed: bool
    test_case: Any
    output: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PropertyTestReport:
    """Comprehensive test report"""
    property_name: str
    total_tests: int
    passed: int
    failed: int
    counterexamples: List[Dict[str, Any]]
    execution_time: float
    
    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_tests if self.total_tests > 0 else 0.0


class PropertyBasedTester:
    """
    Agent that performs property-based testing on AI systems.
    Generates test cases and verifies properties hold.
    """
    
    def __init__(self, temperature: float = 0.7):
        """Initialize the property-based tester"""
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        self.consistent_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        
        # Property registry
        self.properties: Dict[str, Property] = {}
        
        # Test case generator prompts
        self.test_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a test case generator. Generate diverse test cases for testing AI systems.
Property to test: {property_description}
Constraints: {constraints}

Generate test cases that:
1. Cover normal cases
2. Include edge cases
3. Test boundary conditions
4. Challenge the property

Return as a list, one test case per line."""),
            ("user", "Generate {count} test cases.")
        ])
        
        self.agent_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Provide clear, accurate responses."),
            ("user", "{query}")
        ])
        
        # Register default properties
        self._register_default_properties()
    
    def _register_default_properties(self):
        """Register commonly used properties"""
        
        # 1. Non-empty response
        self.register_property(Property(
            name="non_empty_response",
            description="Agent must return non-empty response",
            property_type=PropertyType.POSTCONDITION,
            checker=lambda inp, out: out is not None and len(str(out).strip()) > 0
        ))
        
        # 2. Response length reasonable
        self.register_property(Property(
            name="reasonable_length",
            description="Response length should be reasonable (not too short or long)",
            property_type=PropertyType.POSTCONDITION,
            checker=lambda inp, out: 10 < len(str(out)) < 5000
        ))
        
        # 3. No offensive content
        self.register_property(Property(
            name="no_offensive_content",
            description="Response should not contain offensive language",
            property_type=PropertyType.INVARIANT,
            checker=lambda inp, out: not self._contains_offensive(str(out))
        ))
        
        # 4. Consistency property
        self.register_property(Property(
            name="consistency",
            description="Same input should produce similar output",
            property_type=PropertyType.CONSISTENCY,
            checker=lambda inp, out: True  # Checked differently
        ))
        
        # 5. Format compliance (JSON if requested)
        self.register_property(Property(
            name="json_format",
            description="If JSON requested, output should be valid JSON",
            property_type=PropertyType.FORMAT,
            checker=lambda inp, out: self._is_valid_json_if_requested(inp, out)
        ))
        
        # 6. Question answering completeness
        self.register_property(Property(
            name="answers_question",
            description="Response should address the question asked",
            property_type=PropertyType.POSTCONDITION,
            checker=lambda inp, out: len(str(out)) > len(str(inp)) * 0.5
        ))
        
        # 7. No hallucinated facts (basic check)
        self.register_property(Property(
            name="no_obvious_hallucination",
            description="Response should not contain obviously false statements",
            property_type=PropertyType.INVARIANT,
            checker=lambda inp, out: not self._has_obvious_hallucination(str(out))
        ))
    
    def register_property(self, property_def: Property):
        """Register a new property to test"""
        self.properties[property_def.name] = property_def
        print(f"✅ Registered property: {property_def.name}")
    
    def test_property(self, property_name: str, test_count: int = 20,
                     custom_test_cases: Optional[List[Any]] = None) -> PropertyTestReport:
        """
        Test a specific property with generated or provided test cases
        
        Args:
            property_name: Name of property to test
            test_count: Number of test cases to generate
            custom_test_cases: Optional list of specific test cases
            
        Returns:
            Comprehensive test report
        """
        if property_name not in self.properties:
            raise ValueError(f"Property '{property_name}' not registered")
        
        property_def = self.properties[property_name]
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"Testing Property: {property_name}")
        print(f"Type: {property_def.property_type.value}")
        print(f"Description: {property_def.description}")
        print(f"{'='*70}")
        
        # Generate or use provided test cases
        if custom_test_cases:
            test_cases = custom_test_cases
        else:
            test_cases = self._generate_test_cases(property_def, test_count)
        
        print(f"\n🔄 Running {len(test_cases)} test cases...")
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            result = self._run_test_case(property_def, test_case, i)
            results.append(result)
        
        # Analyze results
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        
        counterexamples = []
        for r in results:
            if not r.passed:
                counterexamples.append({
                    "input": r.test_case,
                    "output": r.output,
                    "error": r.error
                })
        
        execution_time = time.time() - start_time
        
        report = PropertyTestReport(
            property_name=property_name,
            total_tests=len(results),
            passed=passed,
            failed=failed,
            counterexamples=counterexamples,
            execution_time=execution_time
        )
        
        self._print_test_results(report)
        
        return report
    
    def test_all_properties(self, test_count: int = 10) -> Dict[str, PropertyTestReport]:
        """Test all registered properties"""
        print(f"\n{'='*70}")
        print("TESTING ALL PROPERTIES")
        print(f"{'='*70}")
        print(f"Properties to test: {len(self.properties)}")
        print(f"Test cases per property: {test_count}")
        
        reports = {}
        for property_name in self.properties:
            report = self.test_property(property_name, test_count)
            reports[property_name] = report
        
        # Overall summary
        self._print_overall_summary(reports)
        
        return reports
    
    def _generate_test_cases(self, property_def: Property, count: int) -> List[str]:
        """Generate test cases for a property"""
        print(f"   🔄 Generating test cases...")
        
        # Use examples as seed
        constraints = "Include diverse scenarios, edge cases, and boundary conditions"
        
        try:
            chain = self.test_gen_prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "property_description": property_def.description,
                "constraints": constraints,
                "count": count
            })
            
            # Parse test cases (one per line or numbered)
            test_cases = []
            lines = result.strip().split('\n')
            for line in lines:
                # Remove numbering/bullets
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                cleaned = re.sub(r'^[-•]\s*', '', cleaned)
                if cleaned and len(cleaned) > 3:
                    test_cases.append(cleaned)
            
            # Fill up to count if needed
            while len(test_cases) < count:
                test_cases.append(f"Test case {len(test_cases) + 1}")
            
            return test_cases[:count]
            
        except Exception as e:
            print(f"   ⚠️  Test generation error: {str(e)}")
            # Fallback: generic test cases
            return [f"Test query {i+1}" for i in range(count)]
    
    def _run_test_case(self, property_def: Property, test_case: Any, 
                      case_number: int) -> TestResult:
        """Run a single test case"""
        start_time = time.time()
        
        try:
            # Execute agent with test case
            output = self._execute_agent(test_case)
            
            # Check property
            if property_def.property_type == PropertyType.CONSISTENCY:
                passed = self._check_consistency(test_case)
            else:
                passed = property_def.check(test_case, output)
            
            result = TestResult(
                property_name=property_def.name,
                passed=passed,
                test_case=test_case,
                output=output,
                execution_time=time.time() - start_time
            )
            
            status = "✓" if passed else "✗"
            print(f"   {status} Test {case_number}: {str(test_case)[:60]}...")
            
            return result
            
        except Exception as e:
            print(f"   ✗ Test {case_number}: ERROR - {str(e)}")
            return TestResult(
                property_name=property_def.name,
                passed=False,
                test_case=test_case,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _execute_agent(self, query: str) -> str:
        """Execute the agent being tested"""
        chain = self.agent_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
    
    def _check_consistency(self, test_case: Any) -> bool:
        """Check consistency property by running multiple times"""
        outputs = []
        
        # Run 3 times with temperature 0 for consistency
        chain = self.agent_prompt | self.consistent_llm | StrOutputParser()
        for _ in range(3):
            output = chain.invoke({"query": test_case})
            outputs.append(output)
        
        # Check if all outputs are similar (basic check)
        # For production, use more sophisticated similarity metrics
        return len(set(outputs)) <= 2  # Allow minor variation
    
    def _contains_offensive(self, text: str) -> bool:
        """Check if text contains offensive content (basic)"""
        offensive_words = ["offensive", "inappropriate", "harmful"]  # Simplified
        return any(word in text.lower() for word in offensive_words)
    
    def _is_valid_json_if_requested(self, input_text: str, output_text: str) -> bool:
        """Check if output is valid JSON when requested"""
        if "json" not in input_text.lower():
            return True  # Not requesting JSON, so property holds
        
        try:
            import json
            json.loads(output_text)
            return True
        except:
            # Check if JSON is embedded in response
            if "{" in output_text and "}" in output_text:
                start = output_text.find("{")
                end = output_text.rfind("}") + 1
                try:
                    json.loads(output_text[start:end])
                    return True
                except:
                    return False
            return False
    
    def _has_obvious_hallucination(self, text: str) -> bool:
        """Check for obvious hallucinations (basic)"""
        # Check for contradictions or obviously false statements
        false_patterns = [
            r"the sun is cold",
            r"water is dry",
            r"earth is flat",
            r"humans have \d+ legs" # where number != 2
        ]
        
        text_lower = text.lower()
        for pattern in false_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _print_test_results(self, report: PropertyTestReport):
        """Print test results"""
        print(f"\n{'='*70}")
        print(f"RESULTS: {report.property_name}")
        print(f"{'='*70}")
        print(f"Total Tests: {report.total_tests}")
        print(f"Passed: {report.passed} ({report.pass_rate*100:.1f}%)")
        print(f"Failed: {report.failed}")
        print(f"Execution Time: {report.execution_time:.2f}s")
        
        if report.counterexamples:
            print(f"\n❌ Counterexamples Found ({len(report.counterexamples)}):")
            for i, ce in enumerate(report.counterexamples[:5], 1):  # Show first 5
                print(f"\n{i}. Input: {str(ce['input'])[:80]}...")
                if ce['output']:
                    print(f"   Output: {str(ce['output'])[:80]}...")
                if ce['error']:
                    print(f"   Error: {ce['error']}")
        
        print("="*70)
    
    def _print_overall_summary(self, reports: Dict[str, PropertyTestReport]):
        """Print overall summary of all tests"""
        print(f"\n{'='*70}")
        print("OVERALL TEST SUMMARY")
        print(f"{'='*70}")
        
        total_tests = sum(r.total_tests for r in reports.values())
        total_passed = sum(r.passed for r in reports.values())
        overall_pass_rate = total_passed / total_tests if total_tests > 0 else 0
        
        print(f"\nTotal Tests Run: {total_tests}")
        print(f"Overall Pass Rate: {overall_pass_rate*100:.1f}%")
        print(f"\nProperty Results:")
        
        for name, report in reports.items():
            status = "✓" if report.pass_rate >= 0.9 else "✗" if report.pass_rate < 0.7 else "⚠"
            print(f"  {status} {name}: {report.pass_rate*100:.0f}% "
                  f"({report.passed}/{report.total_tests} passed)")
        
        print("="*70)
    
    def shrink_counterexample(self, property_name: str, counterexample: Any) -> Any:
        """
        Attempt to find minimal counterexample (shrinking)
        
        Args:
            property_name: Property that failed
            counterexample: Original counterexample
            
        Returns:
            Minimal counterexample
        """
        property_def = self.properties[property_name]
        current = str(counterexample)
        
        print(f"\n🔍 Shrinking counterexample for {property_name}...")
        print(f"Original: {current[:100]}...")
        
        # Try progressively smaller versions
        for length in range(len(current) // 2, 10, -10):
            candidate = current[:length]
            
            try:
                output = self._execute_agent(candidate)
                if not property_def.check(candidate, output):
                    current = candidate
                    print(f"   Shrunk to: {len(current)} characters")
            except:
                continue
        
        print(f"✓ Minimal counterexample: {current}")
        return current


def demonstrate_property_based_testing():
    """Demonstrate property-based testing"""
    print("="*70)
    print("Pattern 152: Property-Based Testing")
    print("="*70)
    
    tester = PropertyBasedTester()
    
    # Test 1: Non-empty response property
    print("\n" + "="*70)
    print("TEST 1: Non-Empty Response Property")
    print("="*70)
    
    test_cases = [
        "What is Python?",
        "Hello",
        "",  # Edge case
        "Explain quantum physics",
        "?"  # Edge case
    ]
    
    report1 = tester.test_property("non_empty_response", 
                                   custom_test_cases=test_cases)
    
    # Test 2: Reasonable length property
    print("\n" + "="*70)
    print("TEST 2: Reasonable Length Property")
    print("="*70)
    
    report2 = tester.test_property("reasonable_length", test_count=8)
    
    # Test 3: JSON format compliance
    print("\n" + "="*70)
    print("TEST 3: JSON Format Compliance")
    print("="*70)
    
    json_test_cases = [
        "Return the result as JSON: {name: 'test'}",
        "Give me JSON with user info",
        "What is 2+2?",  # Should pass (not requesting JSON)
        "Provide a JSON object with three fields"
    ]
    
    report3 = tester.test_property("json_format", 
                                   custom_test_cases=json_test_cases)
    
    # Test 4: Custom property
    print("\n" + "="*70)
    print("TEST 4: Custom Property - No Numbers in Text Explanation")
    print("="*70)
    
    def no_numbers_checker(inp: Any, out: Any) -> bool:
        """Custom checker: explanations should avoid raw numbers"""
        if "explain" in str(inp).lower():
            # Allow some numbers but not excessive
            return str(out).count("1") + str(out).count("2") < 10
        return True
    
    custom_prop = Property(
        name="minimal_numbers_in_explanation",
        description="Explanations should minimize raw numbers",
        property_type=PropertyType.POSTCONDITION,
        checker=no_numbers_checker
    )
    
    tester.register_property(custom_prop)
    
    explain_cases = [
        "Explain how computers work",
        "What is machine learning?",
        "Tell me about AI"
    ]
    
    report4 = tester.test_property("minimal_numbers_in_explanation",
                                   custom_test_cases=explain_cases)
    
    # Test 5: Consistency property
    print("\n" + "="*70)
    print("TEST 5: Consistency Property")
    print("="*70)
    
    consistent_cases = [
        "What is 2+2?",
        "Name the capital of France"
    ]
    
    report5 = tester.test_property("consistency", 
                                   custom_test_cases=consistent_cases)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
Property-Based Testing Pattern provides:

1. Property Definitions:
   - Invariants (must always hold)
   - Postconditions (output constraints)
   - Consistency checks
   - Format compliance
   - Custom properties

2. Test Generation:
   - Automated test case creation
   - Edge case discovery
   - Boundary condition testing
   - Coverage maximization

3. Property Verification:
   - Systematic property checking
   - Counterexample identification
   - Shrinking to minimal cases
   - Detailed reporting

4. Benefits:
   - Finds bugs traditional testing misses
   - Documents expected behaviors
   - Scales testing automatically
   - More thorough coverage

5. Results Summary:
   - Non-empty responses: {report1.pass_rate*100:.0f}% pass
   - Reasonable length: {report2.pass_rate*100:.0f}% pass
   - JSON format: {report3.pass_rate*100:.0f}% pass
   - Custom property: {report4.pass_rate*100:.0f}% pass
   - Consistency: {report5.pass_rate*100:.0f}% pass

This pattern is essential for robust AI agent testing and
quality assurance in production systems.
    """)


if __name__ == "__main__":
    demonstrate_property_based_testing()
