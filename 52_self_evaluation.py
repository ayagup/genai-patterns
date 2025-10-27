"""
Self-Evaluation Pattern Implementation

This pattern enables agents to evaluate their own outputs through:
- Confidence scoring
- Consistency checking
- Quality assessment
- Error detection
- Self-correction

Use cases:
- Quality control
- Error detection
- Autonomous validation
- Continuous improvement
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class ConfidenceLevel(Enum):
    """Confidence levels for self-evaluation"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class EvaluationResult:
    """Result of self-evaluation"""
    output: str
    confidence: float  # 0.0 to 1.0
    confidence_level: ConfidenceLevel
    quality_score: float  # 0.0 to 1.0
    consistency_score: float  # 0.0 to 1.0
    issues_found: List[str]
    suggestions: List[str]
    passed: bool


@dataclass
class EvaluationCriteria:
    """Criteria for evaluating outputs"""
    min_confidence: float = 0.7
    min_quality: float = 0.7
    min_consistency: float = 0.8
    check_factual_consistency: bool = True
    check_logical_coherence: bool = True
    check_completeness: bool = True


class SelfEvaluationAgent:
    """
    Agent that can evaluate its own outputs
    """
    
    def __init__(self, name: str = "SelfEvaluator"):
        self.name = name
        self.evaluation_history: List[EvaluationResult] = []
        self.performance_metrics = {
            'total_evaluations': 0,
            'passed': 0,
            'failed': 0,
            'avg_confidence': 0.0,
            'avg_quality': 0.0
        }
    
    def generate_with_evaluation(
        self,
        query: str,
        criteria: Optional[EvaluationCriteria] = None
    ) -> EvaluationResult:
        """Generate output and evaluate it"""
        criteria = criteria or EvaluationCriteria()
        
        # Generate output (simulated)
        output = self._generate_output(query)
        
        # Evaluate the output
        result = self.evaluate_output(output, query, criteria)
        
        # Store in history
        self.evaluation_history.append(result)
        self._update_metrics(result)
        
        return result
    
    def evaluate_output(
        self,
        output: str,
        original_query: str,
        criteria: EvaluationCriteria
    ) -> EvaluationResult:
        """Evaluate a given output"""
        
        # Calculate confidence score
        confidence = self._calculate_confidence(output, original_query)
        
        # Calculate quality score
        quality = self._calculate_quality(output)
        
        # Calculate consistency score
        consistency = self._calculate_consistency(output, original_query)
        
        # Find issues
        issues = self._find_issues(output, original_query, criteria)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(output, issues)
        
        # Determine if passed
        passed = (
            confidence >= criteria.min_confidence and
            quality >= criteria.min_quality and
            consistency >= criteria.min_consistency and
            len(issues) == 0
        )
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(confidence)
        
        result = EvaluationResult(
            output=output,
            confidence=confidence,
            confidence_level=confidence_level,
            quality_score=quality,
            consistency_score=consistency,
            issues_found=issues,
            suggestions=suggestions,
            passed=passed
        )
        
        return result
    
    def _generate_output(self, query: str) -> str:
        """Simulate output generation"""
        # In real implementation, this would call an LLM
        responses = {
            "What is 2+2?": "2 + 2 = 4. This is a basic arithmetic operation.",
            "Explain gravity": "Gravity is a fundamental force that attracts objects with mass. It keeps planets in orbit and causes objects to fall.",
            "Write a poem": "Roses are red, violets are blue, AI is learning, and so are you!",
            "Calculate 123 * 456": "123 * 456 = 56,088",
            "Who was Einstein?": "Albert Einstein was a theoretical physicist who developed the theory of relativity."
        }
        
        return responses.get(query, "I'm not certain about this answer.")
    
    def _calculate_confidence(self, output: str, query: str) -> float:
        """Calculate confidence in the output"""
        confidence = 0.5  # Base confidence
        
        # Factors that increase confidence
        if len(output) > 20:
            confidence += 0.1
        
        if any(keyword in output.lower() for keyword in ['is', 'are', 'equals', '=']):
            confidence += 0.1
        
        if not any(word in output.lower() for word in ['maybe', 'probably', 'uncertain', 'not sure']):
            confidence += 0.2
        
        # Check for specific patterns
        if '=' in output and any(char.isdigit() for char in output):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_quality(self, output: str) -> float:
        """Calculate quality of the output"""
        quality = 0.5
        
        # Length check
        if 20 <= len(output) <= 500:
            quality += 0.2
        
        # Structure check
        if output[0].isupper() and output[-1] in '.!?':
            quality += 0.1
        
        # Content check
        words = output.split()
        if len(words) >= 5:
            quality += 0.1
        
        # No repetition
        if len(set(words)) / len(words) > 0.7:
            quality += 0.1
        
        return min(1.0, quality)
    
    def _calculate_consistency(self, output: str, query: str) -> float:
        """Calculate consistency between query and output"""
        consistency = 0.5
        
        # Check if output addresses the query
        query_words = set(query.lower().split())
        output_words = set(output.lower().split())
        
        overlap = len(query_words & output_words)
        if overlap > 0:
            consistency += min(0.3, overlap * 0.1)
        
        # Check for direct answer patterns
        if any(word in output.lower() for word in ['is', 'are', 'equals']):
            consistency += 0.1
        
        # Check for explanation patterns
        if any(word in output.lower() for word in ['because', 'since', 'therefore']):
            consistency += 0.1
        
        return min(1.0, consistency)
    
    def _find_issues(
        self,
        output: str,
        query: str,
        criteria: EvaluationCriteria
    ) -> List[str]:
        """Find potential issues in the output"""
        issues = []
        
        # Check for uncertainty markers
        if any(word in output.lower() for word in ['maybe', 'probably', 'might', 'could be']):
            issues.append("Contains uncertainty markers")
        
        # Check for too short
        if len(output) < 10:
            issues.append("Output too short")
        
        # Check for too long
        if len(output) > 1000:
            issues.append("Output too verbose")
        
        # Check for contradictions
        if 'yes' in output.lower() and 'no' in output.lower():
            if output.lower().find('no') - output.lower().find('yes') < 20:
                issues.append("Potential contradiction detected")
        
        # Check for completeness
        if criteria.check_completeness:
            if output.endswith('...') or 'incomplete' in output.lower():
                issues.append("Output appears incomplete")
        
        return issues
    
    def _generate_suggestions(self, output: str, issues: List[str]) -> List[str]:
        """Generate suggestions for improvement"""
        suggestions = []
        
        if "uncertainty markers" in ' '.join(issues).lower():
            suggestions.append("Remove uncertainty language and provide definitive answer")
        
        if "too short" in ' '.join(issues).lower():
            suggestions.append("Expand answer with more details and explanation")
        
        if "too verbose" in ' '.join(issues).lower():
            suggestions.append("Condense answer to key points")
        
        if "contradiction" in ' '.join(issues).lower():
            suggestions.append("Resolve contradictory statements")
        
        if "incomplete" in ' '.join(issues).lower():
            suggestions.append("Complete the answer with remaining information")
        
        return suggestions
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to level"""
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _update_metrics(self, result: EvaluationResult):
        """Update performance metrics"""
        self.performance_metrics['total_evaluations'] += 1
        
        if result.passed:
            self.performance_metrics['passed'] += 1
        else:
            self.performance_metrics['failed'] += 1
        
        # Update running averages
        total = self.performance_metrics['total_evaluations']
        self.performance_metrics['avg_confidence'] = (
            (self.performance_metrics['avg_confidence'] * (total - 1) + result.confidence) / total
        )
        self.performance_metrics['avg_quality'] = (
            (self.performance_metrics['avg_quality'] * (total - 1) + result.quality_score) / total
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        total = self.performance_metrics['total_evaluations']
        if total == 0:
            return {"message": "No evaluations performed yet"}
        
        pass_rate = self.performance_metrics['passed'] / total
        
        return {
            "total_evaluations": total,
            "passed": self.performance_metrics['passed'],
            "failed": self.performance_metrics['failed'],
            "pass_rate": f"{pass_rate:.1%}",
            "avg_confidence": f"{self.performance_metrics['avg_confidence']:.2f}",
            "avg_quality": f"{self.performance_metrics['avg_quality']:.2f}",
            "recent_evaluations": [
                {
                    "output": e.output[:50] + "..." if len(e.output) > 50 else e.output,
                    "confidence": f"{e.confidence:.2f}",
                    "passed": e.passed
                }
                for e in self.evaluation_history[-5:]
            ]
        }


def demo_self_evaluation():
    """Demonstrate self-evaluation pattern"""
    print("=" * 60)
    print("Self-Evaluation Pattern Demo")
    print("=" * 60)
    
    agent = SelfEvaluationAgent()
    
    # Test queries
    queries = [
        "What is 2+2?",
        "Explain gravity",
        "Write a poem",
        "Calculate 123 * 456",
        "Who was Einstein?"
    ]
    
    print("\n1. Generating and Evaluating Outputs")
    print("-" * 60)
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = agent.generate_with_evaluation(query)
        
        print(f"Output: {result.output}")
        print(f"Confidence: {result.confidence:.2f} ({result.confidence_level.name})")
        print(f"Quality: {result.quality_score:.2f}")
        print(f"Consistency: {result.consistency_score:.2f}")
        print(f"Status: {'✓ PASSED' if result.passed else '✗ FAILED'}")
        
        if result.issues_found:
            print(f"Issues: {', '.join(result.issues_found)}")
        
        if result.suggestions:
            print(f"Suggestions: {', '.join(result.suggestions)}")
    
    print("\n" + "=" * 60)
    print("2. Performance Report")
    print("-" * 60)
    
    report = agent.get_performance_report()
    print(json.dumps(report, indent=2))
    
    print("\n" + "=" * 60)
    print("3. Custom Evaluation Criteria")
    print("-" * 60)
    
    strict_criteria = EvaluationCriteria(
        min_confidence=0.9,
        min_quality=0.8,
        min_consistency=0.9
    )
    
    result = agent.generate_with_evaluation("What is 2+2?", strict_criteria)
    print(f"\nWith strict criteria:")
    print(f"Query: What is 2+2?")
    print(f"Status: {'✓ PASSED' if result.passed else '✗ FAILED'}")
    print(f"Confidence: {result.confidence:.2f} (required: 0.90)")
    print(f"Quality: {result.quality_score:.2f} (required: 0.80)")
    print(f"Consistency: {result.consistency_score:.2f} (required: 0.90)")
    
    print("\n" + "=" * 60)
    print("Self-Evaluation Pattern Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_self_evaluation()
