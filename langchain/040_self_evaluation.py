"""
Pattern 040: Self-Evaluation

Description:
    Self-Evaluation enables agents to assess the quality of their own outputs
    through confidence scoring, consistency checking, and self-critique. The agent
    acts as its own judge, identifying potential errors, measuring confidence,
    and flagging outputs that may require human review or regeneration.

Components:
    - Confidence Estimator: Assigns confidence scores to outputs
    - Consistency Checker: Verifies logical consistency
    - Quality Metrics: Measures output quality
    - Self-Critique: Identifies weaknesses and errors
    - Threshold Monitor: Flags low-quality outputs

Use Cases:
    - Quality control automation
    - Error detection before delivery
    - Confidence-based routing
    - Self-improvement loops
    - Autonomous quality assurance
    - Production reliability

LangChain Implementation:
    Uses LLM-based evaluation chains and scoring mechanisms to enable
    agents to assess their own output quality and confidence.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from statistics import mean, stdev
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class QualityLevel(Enum):
    """Quality levels for outputs."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


class EvaluationDimension(Enum):
    """Dimensions for evaluation."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    SAFETY = "safety"


@dataclass
class EvaluationScore:
    """Score for a single evaluation dimension."""
    dimension: EvaluationDimension
    score: float  # 0.0 to 1.0
    reasoning: str = ""
    confidence: float = 1.0


@dataclass
class SelfEvaluation:
    """Complete self-evaluation of an output."""
    output: str
    overall_confidence: float
    quality_level: QualityLevel
    dimension_scores: List[EvaluationScore]
    identified_issues: List[str]
    suggestions: List[str]
    should_regenerate: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsistencyCheck:
    """Result of consistency checking."""
    output: str
    is_consistent: bool
    contradictions: List[str]
    logical_errors: List[str]
    consistency_score: float  # 0.0 to 1.0


class ConfidenceEstimator:
    """
    Estimates confidence in generated outputs.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        # Confidence estimation prompt
        self.confidence_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a confidence estimator. Evaluate how confident you are in this response.

Consider:
1. Factual accuracy and verifiability
2. Completeness of the answer
3. Clarity and coherence
4. Presence of uncertainties or assumptions
5. Potential for errors or misconceptions

Respond in this format:
CONFIDENCE: [0.0-1.0]
REASONING: [explanation]
UNCERTAINTIES: [list any uncertain aspects]"""),
            ("user", "Query: {query}\n\nResponse: {response}\n\nEvaluate confidence in this response.")
        ])
    
    def estimate_confidence(self, query: str, response: str) -> Tuple[float, str, List[str]]:
        """
        Estimate confidence in a response.
        
        Returns:
            Tuple of (confidence_score, reasoning, uncertainties)
        """
        chain = self.confidence_prompt | self.llm | StrOutputParser()
        evaluation = chain.invoke({"query": query, "response": response})
        
        # Parse evaluation
        confidence = 0.5
        reasoning = ""
        uncertainties = []
        
        lines = evaluation.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.split(':', 1)[1].strip()
            elif line.startswith("UNCERTAINTIES:"):
                # Collect remaining lines as uncertainties
                uncertainties = [l.strip() for l in lines[i+1:] if l.strip()]
                break
        
        return confidence, reasoning, uncertainties


class ConsistencyChecker:
    """
    Checks logical consistency of outputs.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        # Consistency checking prompt
        self.consistency_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a consistency checker. Analyze this text for logical consistency.

Check for:
1. Internal contradictions
2. Logical fallacies
3. Inconsistent statements
4. Conflicting information

Respond in this format:
CONSISTENT: [YES/NO]
CONTRADICTIONS: [list any contradictions found]
LOGICAL_ERRORS: [list any logical errors]
SCORE: [0.0-1.0]"""),
            ("user", "Text to check:\n{text}\n\nAnalyze for consistency.")
        ])
    
    def check_consistency(self, text: str) -> ConsistencyCheck:
        """
        Check consistency of text.
        
        Returns:
            ConsistencyCheck object
        """
        chain = self.consistency_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"text": text})
        
        # Parse result
        is_consistent = True
        contradictions = []
        logical_errors = []
        consistency_score = 1.0
        
        lines = result.split('\n')
        current_section = None
        
        for line in lines:
            if line.startswith("CONSISTENT:"):
                is_consistent = "YES" in line.upper()
            elif line.startswith("CONTRADICTIONS:"):
                current_section = "contradictions"
            elif line.startswith("LOGICAL_ERRORS:"):
                current_section = "logical_errors"
            elif line.startswith("SCORE:"):
                try:
                    consistency_score = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
                current_section = None
            elif line.strip() and current_section:
                if current_section == "contradictions":
                    contradictions.append(line.strip())
                elif current_section == "logical_errors":
                    logical_errors.append(line.strip())
        
        return ConsistencyCheck(
            output=text,
            is_consistent=is_consistent,
            contradictions=contradictions,
            logical_errors=logical_errors,
            consistency_score=consistency_score
        )


class QualityEvaluator:
    """
    Evaluates output quality across multiple dimensions.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        # Quality evaluation prompt
        self.quality_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quality evaluator. Evaluate this response on the given dimension.

Dimension: {dimension}

Rate from 0.0 (worst) to 1.0 (best) and explain your reasoning.

Respond in this format:
SCORE: [0.0-1.0]
REASONING: [explanation]"""),
            ("user", "Query: {query}\n\nResponse: {response}\n\nEvaluate on: {dimension}")
        ])
    
    def evaluate_dimension(
        self,
        query: str,
        response: str,
        dimension: EvaluationDimension
    ) -> EvaluationScore:
        """
        Evaluate response on a single dimension.
        
        Returns:
            EvaluationScore object
        """
        chain = self.quality_prompt | self.llm | StrOutputParser()
        evaluation = chain.invoke({
            "query": query,
            "response": response,
            "dimension": dimension.value
        })
        
        # Parse evaluation
        score = 0.5
        reasoning = ""
        
        lines = evaluation.split('\n')
        for line in lines:
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.split(':', 1)[1].strip()
        
        return EvaluationScore(
            dimension=dimension,
            score=score,
            reasoning=reasoning,
            confidence=1.0
        )


class SelfEvaluationAgent:
    """
    Agent that evaluates its own outputs for quality assurance.
    
    Features:
    - Confidence estimation
    - Consistency checking
    - Multi-dimensional quality evaluation
    - Issue identification
    - Regeneration recommendations
    """
    
    def __init__(
        self,
        response_temperature: float = 0.7,
        evaluation_temperature: float = 0.1,
        confidence_threshold: float = 0.7,
        quality_threshold: float = 0.6
    ):
        self.response_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=response_temperature)
        self.eval_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=evaluation_temperature)
        
        self.confidence_estimator = ConfidenceEstimator(self.eval_llm)
        self.consistency_checker = ConsistencyChecker(self.eval_llm)
        self.quality_evaluator = QualityEvaluator(self.eval_llm)
        
        self.confidence_threshold = confidence_threshold
        self.quality_threshold = quality_threshold
        
        # Response generation prompt
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Provide clear, accurate answers."),
            ("user", "{query}")
        ])
        
        # Evaluation history
        self.evaluations: List[SelfEvaluation] = []
    
    def generate_response(self, query: str) -> str:
        """Generate response to query."""
        chain = self.response_prompt | self.response_llm | StrOutputParser()
        return chain.invoke({"query": query})
    
    def evaluate_response(
        self,
        query: str,
        response: str,
        dimensions: Optional[List[EvaluationDimension]] = None
    ) -> SelfEvaluation:
        """
        Perform comprehensive self-evaluation of response.
        
        Args:
            query: The original query
            response: The generated response
            dimensions: Dimensions to evaluate (default: all)
            
        Returns:
            SelfEvaluation object
        """
        if dimensions is None:
            dimensions = [
                EvaluationDimension.ACCURACY,
                EvaluationDimension.COMPLETENESS,
                EvaluationDimension.CLARITY,
                EvaluationDimension.RELEVANCE
            ]
        
        # 1. Estimate confidence
        confidence, conf_reasoning, uncertainties = self.confidence_estimator.estimate_confidence(query, response)
        
        # 2. Check consistency
        consistency_check = self.consistency_checker.check_consistency(response)
        
        # 3. Evaluate on each dimension
        dimension_scores = []
        for dimension in dimensions:
            score = self.quality_evaluator.evaluate_dimension(query, response, dimension)
            dimension_scores.append(score)
        
        # 4. Calculate overall quality
        avg_score = mean([s.score for s in dimension_scores])
        
        # Adjust based on consistency
        if not consistency_check.is_consistent:
            avg_score *= 0.8  # Penalize inconsistency
        
        # Determine quality level
        if avg_score >= 0.8:
            quality_level = QualityLevel.EXCELLENT
        elif avg_score >= 0.7:
            quality_level = QualityLevel.GOOD
        elif avg_score >= 0.6:
            quality_level = QualityLevel.ACCEPTABLE
        elif avg_score >= 0.4:
            quality_level = QualityLevel.POOR
        else:
            quality_level = QualityLevel.UNACCEPTABLE
        
        # 5. Identify issues
        identified_issues = []
        
        if confidence < self.confidence_threshold:
            identified_issues.append(f"Low confidence: {confidence:.2f}")
        
        if not consistency_check.is_consistent:
            identified_issues.extend([f"Contradiction: {c}" for c in consistency_check.contradictions])
            identified_issues.extend([f"Logical error: {e}" for e in consistency_check.logical_errors])
        
        for score in dimension_scores:
            if score.score < self.quality_threshold:
                identified_issues.append(f"Low {score.dimension.value}: {score.score:.2f}")
        
        identified_issues.extend([f"Uncertainty: {u}" for u in uncertainties])
        
        # 6. Generate suggestions
        suggestions = []
        
        if confidence < self.confidence_threshold:
            suggestions.append("Consider adding caveats or acknowledging uncertainties")
        
        if not consistency_check.is_consistent:
            suggestions.append("Revise to resolve logical contradictions")
        
        for score in dimension_scores:
            if score.score < self.quality_threshold:
                suggestions.append(f"Improve {score.dimension.value}: {score.reasoning}")
        
        # 7. Decide if should regenerate
        should_regenerate = (
            confidence < self.confidence_threshold or
            not consistency_check.is_consistent or
            avg_score < self.quality_threshold
        )
        
        evaluation = SelfEvaluation(
            output=response,
            overall_confidence=confidence,
            quality_level=quality_level,
            dimension_scores=dimension_scores,
            identified_issues=identified_issues,
            suggestions=suggestions,
            should_regenerate=should_regenerate
        )
        
        self.evaluations.append(evaluation)
        
        return evaluation
    
    def process_with_evaluation(
        self,
        query: str,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Generate response with self-evaluation and automatic regeneration.
        
        Args:
            query: The query to process
            max_attempts: Maximum regeneration attempts
            
        Returns:
            Dictionary with response and evaluation
        """
        attempt = 0
        best_response = None
        best_evaluation = None
        best_score = -1
        
        while attempt < max_attempts:
            attempt += 1
            
            # Generate response
            response = self.generate_response(query)
            
            # Evaluate
            evaluation = self.evaluate_response(query, response)
            
            # Track best
            avg_score = mean([s.score for s in evaluation.dimension_scores])
            if avg_score > best_score:
                best_score = avg_score
                best_response = response
                best_evaluation = evaluation
            
            # Stop if quality is acceptable
            if not evaluation.should_regenerate:
                break
        
        return {
            "query": query,
            "response": best_response,
            "evaluation": best_evaluation,
            "attempts": attempt,
            "accepted": not best_evaluation.should_regenerate
        }
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get statistics about evaluations."""
        if not self.evaluations:
            return {"total_evaluations": 0}
        
        avg_confidence = mean([e.overall_confidence for e in self.evaluations])
        
        quality_distribution = {}
        for e in self.evaluations:
            quality_distribution[e.quality_level.value] = quality_distribution.get(e.quality_level.value, 0) + 1
        
        regeneration_rate = sum(1 for e in self.evaluations if e.should_regenerate) / len(self.evaluations)
        
        return {
            "total_evaluations": len(self.evaluations),
            "average_confidence": avg_confidence,
            "quality_distribution": quality_distribution,
            "regeneration_rate": regeneration_rate
        }


def demonstrate_self_evaluation():
    """
    Demonstrates self-evaluation with quality assessment and confidence scoring.
    """
    print("=" * 80)
    print("SELF-EVALUATION DEMONSTRATION")
    print("=" * 80)
    
    # Create self-evaluation agent
    agent = SelfEvaluationAgent(
        confidence_threshold=0.7,
        quality_threshold=0.6
    )
    
    # Test 1: Simple factual query (should have high confidence)
    print("\n" + "=" * 80)
    print("Test 1: Factual Query (Expected: High Confidence)")
    print("=" * 80)
    
    query1 = "What is the capital of France?"
    print(f"\nQuery: {query1}")
    
    result1 = agent.process_with_evaluation(query1, max_attempts=1)
    
    print(f"\nResponse: {result1['response']}")
    print(f"\nAttempts: {result1['attempts']}")
    print(f"Accepted: {result1['accepted']}")
    
    eval1 = result1['evaluation']
    print(f"\nOverall Confidence: {eval1.overall_confidence:.2f}")
    print(f"Quality Level: {eval1.quality_level.value}")
    print(f"Should Regenerate: {eval1.should_regenerate}")
    
    print("\nDimension Scores:")
    for score in eval1.dimension_scores:
        print(f"  - {score.dimension.value}: {score.score:.2f}")
    
    if eval1.identified_issues:
        print("\nIdentified Issues:")
        for issue in eval1.identified_issues[:3]:
            print(f"  - {issue}")
    
    # Test 2: Complex query (may have lower confidence)
    print("\n" + "=" * 80)
    print("Test 2: Complex Speculative Query")
    print("=" * 80)
    
    query2 = "What will be the impact of quantum computing on cryptography in 2030?"
    print(f"\nQuery: {query2}")
    
    result2 = agent.process_with_evaluation(query2, max_attempts=2)
    
    print(f"\nResponse: {result2['response'][:200]}...")
    print(f"\nAttempts: {result2['attempts']}")
    print(f"Accepted: {result2['accepted']}")
    
    eval2 = result2['evaluation']
    print(f"\nOverall Confidence: {eval2.overall_confidence:.2f}")
    print(f"Quality Level: {eval2.quality_level.value}")
    print(f"Should Regenerate: {eval2.should_regenerate}")
    
    print("\nDimension Scores:")
    for score in eval2.dimension_scores:
        print(f"  - {score.dimension.value}: {score.score:.2f} - {score.reasoning[:50]}...")
    
    if eval2.identified_issues:
        print("\nIdentified Issues:")
        for issue in eval2.identified_issues[:3]:
            print(f"  - {issue}")
    
    if eval2.suggestions:
        print("\nSuggestions:")
        for suggestion in eval2.suggestions[:3]:
            print(f"  - {suggestion}")
    
    # Test 3: Consistency checking
    print("\n" + "=" * 80)
    print("Test 3: Consistency Checking")
    print("=" * 80)
    
    query3 = "Explain machine learning"
    print(f"\nQuery: {query3}")
    
    response3 = agent.generate_response(query3)
    print(f"\nResponse: {response3[:200]}...")
    
    consistency = agent.consistency_checker.check_consistency(response3)
    print(f"\nConsistent: {consistency.is_consistent}")
    print(f"Consistency Score: {consistency.consistency_score:.2f}")
    
    if consistency.contradictions:
        print("\nContradictions Found:")
        for contradiction in consistency.contradictions:
            print(f"  - {contradiction}")
    
    # Test 4: Multi-dimensional evaluation
    print("\n" + "=" * 80)
    print("Test 4: Comprehensive Multi-Dimensional Evaluation")
    print("=" * 80)
    
    query4 = "How does photosynthesis work?"
    print(f"\nQuery: {query4}")
    
    response4 = agent.generate_response(query4)
    print(f"\nResponse: {response4[:200]}...")
    
    eval4 = agent.evaluate_response(
        query4,
        response4,
        dimensions=[
            EvaluationDimension.ACCURACY,
            EvaluationDimension.COMPLETENESS,
            EvaluationDimension.CLARITY,
            EvaluationDimension.RELEVANCE,
            EvaluationDimension.COHERENCE
        ]
    )
    
    print(f"\nComprehensive Evaluation:")
    print(f"Overall Confidence: {eval4.overall_confidence:.2f}")
    print(f"Quality Level: {eval4.quality_level.value}")
    
    print("\nAll Dimension Scores:")
    for score in eval4.dimension_scores:
        print(f"  - {score.dimension.value}: {score.score:.2f}")
    
    # Show statistics
    print("\n" + "=" * 80)
    print("Evaluation Statistics")
    print("=" * 80)
    
    stats = agent.get_evaluation_statistics()
    print(f"\nTotal Evaluations: {stats['total_evaluations']}")
    print(f"Average Confidence: {stats['average_confidence']:.2f}")
    print(f"Regeneration Rate: {stats['regeneration_rate']:.1%}")
    
    print("\nQuality Distribution:")
    for quality, count in stats['quality_distribution'].items():
        print(f"  - {quality}: {count}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Self-Evaluation provides:
✓ Confidence estimation for outputs
✓ Consistency checking for logical errors
✓ Multi-dimensional quality assessment
✓ Automatic issue identification
✓ Regeneration recommendations
✓ Autonomous quality assurance

This pattern excels at:
- Quality control automation
- Error detection before delivery
- Confidence-based routing
- Self-improvement loops
- Production reliability
- Autonomous QA

Evaluation components:
1. Confidence Estimator: Assigns confidence scores
2. Consistency Checker: Verifies logical consistency
3. Quality Evaluator: Multi-dimensional assessment
4. Issue Identifier: Flags problems
5. Suggestion Generator: Improvement recommendations

Evaluation dimensions:
- ACCURACY: Factual correctness
- COMPLETENESS: Thorough coverage
- CLARITY: Clear communication
- RELEVANCE: On-topic responses
- COHERENCE: Logical flow
- SAFETY: Safe and appropriate

Quality levels:
- EXCELLENT: 0.8-1.0
- GOOD: 0.7-0.8
- ACCEPTABLE: 0.6-0.7
- POOR: 0.4-0.6
- UNACCEPTABLE: 0.0-0.4

Evaluation process:
1. Generate response
2. Estimate confidence
3. Check consistency
4. Evaluate on dimensions
5. Identify issues
6. Generate suggestions
7. Decide if regeneration needed

Benefits:
- Autonomous QA: No human needed
- Early detection: Catch errors before delivery
- Transparency: Clear reasoning for scores
- Adaptability: Configurable thresholds
- Continuous: Always on guard
- Scalable: Handles high volume

Use Self-Evaluation when you need:
- Production quality control
- Confidence-based routing
- Error detection automation
- Self-improving systems
- High-reliability requirements
- Transparent quality metrics
""")


if __name__ == "__main__":
    demonstrate_self_evaluation()
