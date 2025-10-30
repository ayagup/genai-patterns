"""
Metacognitive Monitoring Pattern
Agent monitors its own thinking process and confidence
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random
class ConfidenceLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
class UncertaintySource(Enum):
    AMBIGUOUS_INPUT = "ambiguous_input"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    CONFLICTING_INFORMATION = "conflicting_information"
    KNOWLEDGE_GAP = "knowledge_gap"
    COMPLEX_REASONING = "complex_reasoning"
@dataclass
class ThinkingStep:
    """A step in the reasoning process"""
    step_id: int
    description: str
    confidence: float
    reasoning: str
    uncertainties: List[UncertaintySource] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
@dataclass
class MetacognitiveAssessment:
    """Assessment of the thinking process"""
    overall_confidence: float
    confidence_level: ConfidenceLevel
    reasoning_quality: float
    identified_gaps: List[str]
    recommendations: List[str]
    should_request_help: bool
class MetacognitiveMonitor:
    """Monitors and assesses the reasoning process"""
    def __init__(self):
        self.thinking_steps: List[ThinkingStep] = []
        self.assessments: List[MetacognitiveAssessment] = []
    def record_step(self, description: str, confidence: float, 
                   reasoning: str, uncertainties: List[UncertaintySource] = None):
        """Record a thinking step"""
        step = ThinkingStep(
            step_id=len(self.thinking_steps) + 1,
            description=description,
            confidence=confidence,
            reasoning=reasoning,
            uncertainties=uncertainties or []
        )
        self.thinking_steps.append(step)
        print(f"\n[Step {step.step_id}] {description}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Reasoning: {reasoning}")
        if uncertainties:
            print(f"  Uncertainties: {[u.value for u in uncertainties]}")
        return step
    def assess_confidence(self, response: str) -> float:
        """Assess confidence in a response"""
        # Multiple factors for confidence assessment
        factors = []
        # Factor 1: Average confidence of steps
        if self.thinking_steps:
            avg_step_confidence = sum(s.confidence for s in self.thinking_steps) / len(self.thinking_steps)
            factors.append(avg_step_confidence)
        # Factor 2: Number of uncertainties
        total_uncertainties = sum(len(s.uncertainties) for s in self.thinking_steps)
        uncertainty_penalty = max(0, 1.0 - (total_uncertainties * 0.1))
        factors.append(uncertainty_penalty)
        # Factor 3: Consistency of confidence across steps
        if len(self.thinking_steps) > 1:
            confidences = [s.confidence for s in self.thinking_steps]
            variance = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
            consistency = max(0, 1.0 - variance)
            factors.append(consistency)
        # Factor 4: Response completeness (simulated)
        completeness = min(1.0, len(response) / 100)  # Assume 100 chars is "complete"
        factors.append(completeness)
        # Combine factors
        overall_confidence = sum(factors) / len(factors) if factors else 0.5
        return overall_confidence
    def identify_knowledge_gaps(self) -> List[str]:
        """Identify gaps in knowledge or reasoning"""
        gaps = []
        # Check for uncertainty patterns
        uncertainty_counts = {}
        for step in self.thinking_steps:
            for uncertainty in step.uncertainties:
                uncertainty_counts[uncertainty] = uncertainty_counts.get(uncertainty, 0) + 1
        for uncertainty, count in uncertainty_counts.items():
            if count >= 2:
                gaps.append(f"Multiple instances of {uncertainty.value}")
        # Check for low confidence steps
        low_confidence_steps = [s for s in self.thinking_steps if s.confidence < 0.6]
        if low_confidence_steps:
            gaps.append(f"{len(low_confidence_steps)} steps with low confidence")
        return gaps
    def generate_recommendations(self, confidence: float, gaps: List[str]) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        if confidence < 0.5:
            recommendations.append("Consider requesting human assistance")
            recommendations.append("Gather more information before proceeding")
        elif confidence < 0.7:
            recommendations.append("Verify key assumptions")
            recommendations.append("Consider alternative approaches")
        if "knowledge_gap" in str(gaps):
            recommendations.append("Consult external knowledge sources")
        if "ambiguous_input" in str(gaps):
            recommendations.append("Request clarification from user")
        if len(self.thinking_steps) < 3:
            recommendations.append("Break down problem into more steps")
        return recommendations
    def perform_assessment(self, response: str) -> MetacognitiveAssessment:
        """Perform comprehensive metacognitive assessment"""
        print(f"\n{'='*60}")
        print("METACOGNITIVE ASSESSMENT")
        print(f"{'='*60}")
        # Assess confidence
        confidence = self.assess_confidence(response)
        # Determine confidence level
        if confidence >= 0.9:
            level = ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            level = ConfidenceLevel.HIGH
        elif confidence >= 0.6:
            level = ConfidenceLevel.MEDIUM
        elif confidence >= 0.4:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW
        # Assess reasoning quality
        reasoning_quality = self._assess_reasoning_quality()
        # Identify gaps
        gaps = self.identify_knowledge_gaps()
        # Generate recommendations
        recommendations = self.generate_recommendations(confidence, gaps)
        # Decide if help needed
        should_request_help = confidence < 0.5 or reasoning_quality < 0.6
        assessment = MetacognitiveAssessment(
            overall_confidence=confidence,
            confidence_level=level,
            reasoning_quality=reasoning_quality,
            identified_gaps=gaps,
            recommendations=recommendations,
            should_request_help=should_request_help
        )
        self.assessments.append(assessment)
        # Print assessment
        print(f"\nOverall Confidence: {confidence:.1%} ({level.value})")
        print(f"Reasoning Quality: {reasoning_quality:.1%}")
        if gaps:
            print(f"\nIdentified Gaps:")
            for gap in gaps:
                print(f"  • {gap}")
        if recommendations:
            print(f"\nRecommendations:")
            for rec in recommendations:
                print(f"  • {rec}")
        if should_request_help:
            print(f"\n⚠️  Recommendation: Request human assistance")
        return assessment
    def _assess_reasoning_quality(self) -> float:
        """Assess quality of reasoning process"""
        if not self.thinking_steps:
            return 0.0
        quality_factors = []
        # Factor 1: Depth of reasoning (number of steps)
        depth_score = min(1.0, len(self.thinking_steps) / 5)
        quality_factors.append(depth_score)
        # Factor 2: Clarity of reasoning
        clarity_score = sum(1 for s in self.thinking_steps if len(s.reasoning) > 20) / len(self.thinking_steps)
        quality_factors.append(clarity_score)
        # Factor 3: Logical flow (simulated - check if confidence improves)
        if len(self.thinking_steps) > 1:
            confidences = [s.confidence for s in self.thinking_steps]
            improving = sum(1 for i in range(1, len(confidences)) if confidences[i] >= confidences[i-1])
            flow_score = improving / (len(confidences) - 1)
            quality_factors.append(flow_score)
        return sum(quality_factors) / len(quality_factors)
class MetacognitiveAgent:
    """Agent with metacognitive monitoring capabilities"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.monitor = MetacognitiveMonitor()
    def solve_with_monitoring(self, problem: str) -> Dict[str, Any]:
        """Solve problem with metacognitive monitoring"""
        print(f"\n{'='*70}")
        print(f"SOLVING WITH METACOGNITIVE MONITORING")
        print(f"{'='*70}")
        print(f"Problem: {problem}\n")
        # Step 1: Understand the problem
        self.monitor.record_step(
            description="Understanding the problem",
            confidence=0.85,
            reasoning="Problem statement is clear and well-defined",
            uncertainties=[]
        )
        # Step 2: Identify approach
        uncertainties = []
        if "complex" in problem.lower() or "difficult" in problem.lower():
            uncertainties.append(UncertaintySource.COMPLEX_REASONING)
        self.monitor.record_step(
            description="Identifying solution approach",
            confidence=0.75,
            reasoning="Multiple approaches possible, selected most straightforward",
            uncertainties=uncertainties
        )
        # Step 3: Gather information
        confidence = 0.8
        uncertainties = []
        # Simulate knowledge gaps
        if random.random() < 0.3:
            uncertainties.append(UncertaintySource.KNOWLEDGE_GAP)
            confidence = 0.6
        self.monitor.record_step(
            description="Gathering necessary information",
            confidence=confidence,
            reasoning="Retrieved relevant facts and data",
            uncertainties=uncertainties
        )
        # Step 4: Apply reasoning
        self.monitor.record_step(
            description="Applying logical reasoning",
            confidence=0.82,
            reasoning="Step-by-step logical deduction applied",
            uncertainties=[]
        )
        # Step 5: Formulate response
        response = f"Solution to '{problem}': [Detailed answer would go here]"
        self.monitor.record_step(
            description="Formulating final response",
            confidence=0.88,
            reasoning="Response synthesized from all reasoning steps",
            uncertainties=[]
        )
        # Perform metacognitive assessment
        assessment = self.monitor.perform_assessment(response)
        return {
            'problem': problem,
            'response': response,
            'assessment': assessment,
            'thinking_steps': len(self.monitor.thinking_steps),
            'should_provide_response': not assessment.should_request_help
        }
# Usage
if __name__ == "__main__":
    print("="*80)
    print("METACOGNITIVE MONITORING PATTERN DEMONSTRATION")
    print("="*80)
    agent = MetacognitiveAgent("metacog-agent-001")
    # Example 1: Clear problem
    print("\n" + "="*80)
    print("EXAMPLE 1: Clear Problem")
    print("="*80)
    result1 = agent.solve_with_monitoring(
        "Calculate the sum of the first 10 natural numbers"
    )
    print(f"\n{'='*60}")
    print("FINAL DECISION")
    print(f"{'='*60}")
    if result1['should_provide_response']:
        print("✓ Confidence sufficient - providing response")
        print(f"Response: {result1['response']}")
    else:
        print("⚠️  Confidence insufficient - requesting human assistance")
    # Example 2: Complex problem
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Complex/Ambiguous Problem")
    print("="*80)
    agent2 = MetacognitiveAgent("metacog-agent-002")
    result2 = agent2.solve_with_monitoring(
        "Analyze the complex socioeconomic implications of emerging technologies"
    )
    print(f"\n{'='*60}")
    print("FINAL DECISION")
    print(f"{'='*60}")
    if result2['should_provide_response']:
        print("✓ Confidence sufficient - providing response")
    else:
        print("⚠️  Confidence insufficient - requesting human assistance")
        print(f"Recommendations: {result2['assessment'].recommendations}")
