"""
Metacognitive Monitoring Pattern Implementation

This module demonstrates metacognitive monitoring where an agent monitors its own 
thinking process, estimates confidence levels, quantifies uncertainty, and detects
potential errors in its reasoning.

Key Components:
- Confidence estimation for thoughts and decisions
- Uncertainty quantification and calibration
- Error detection and self-correction
- Meta-reasoning about reasoning quality
- Adaptive strategy selection based on metacognitive assessment
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple, Callable
from enum import Enum
import math
import random
import statistics
import time


class ConfidenceLevel(Enum):
    """Confidence levels for metacognitive assessment"""
    VERY_LOW = "very_low"      # 0.0-0.2
    LOW = "low"                # 0.2-0.4
    MEDIUM = "medium"          # 0.4-0.6
    HIGH = "high"              # 0.6-0.8
    VERY_HIGH = "very_high"    # 0.8-1.0


class UncertaintyType(Enum):
    """Types of uncertainty in reasoning"""
    EPISTEMIC = "epistemic"        # Knowledge uncertainty
    ALEATORIC = "aleatoric"        # Data/measurement uncertainty
    MODEL = "model"                # Model/method uncertainty
    LINGUISTIC = "linguistic"      # Language interpretation uncertainty


class ReasoningQuality(Enum):
    """Assessment of reasoning quality"""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


@dataclass
class MetacognitiveAssessment:
    """Assessment of a reasoning step or decision"""
    confidence: float  # 0.0 to 1.0
    uncertainty_types: List[UncertaintyType]
    evidence_strength: float  # How strong is supporting evidence
    coherence_score: float    # How well does this fit with other beliefs
    novelty_score: float      # How novel/surprising is this
    complexity_score: float   # How complex is the reasoning
    timestamp: float = field(default_factory=time.time)
    reasoning_quality: Optional[ReasoningQuality] = None
    potential_errors: List[str] = field(default_factory=list)
    suggested_improvements: List[str] = field(default_factory=list)


@dataclass
class ThoughtMonitor:
    """Monitors individual thoughts and reasoning steps"""
    thought_id: str
    content: str
    reasoning_type: str  # e.g., "deductive", "inductive", "abductive"
    assessment: MetacognitiveAssessment
    dependencies: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)


class ConfidenceCalibrator:
    """Calibrates confidence estimates based on past performance"""
    
    def __init__(self):
        self.prediction_history: List[Tuple[float, bool]] = []  # (confidence, was_correct)
        self.calibration_bins = 10
    
    def add_prediction(self, confidence: float, was_correct: bool):
        """Add a prediction with its actual outcome"""
        self.prediction_history.append((confidence, was_correct))
    
    def get_calibration_score(self) -> float:
        """Calculate how well-calibrated confidence estimates are"""
        if len(self.prediction_history) < 10:
            return 0.5  # Default calibration
        
        # Create bins for confidence levels
        bins = [[] for _ in range(self.calibration_bins)]
        
        for confidence, was_correct in self.prediction_history:
            bin_idx = min(int(confidence * self.calibration_bins), self.calibration_bins - 1)
            bins[bin_idx].append(was_correct)
        
        # Calculate calibration error
        total_error = 0.0
        total_predictions = 0
        
        for i, bin_predictions in enumerate(bins):
            if bin_predictions:
                expected_accuracy = (i + 0.5) / self.calibration_bins
                actual_accuracy = sum(bin_predictions) / len(bin_predictions)
                error = abs(expected_accuracy - actual_accuracy)
                total_error += error * len(bin_predictions)
                total_predictions += len(bin_predictions)
        
        if total_predictions == 0:
            return 0.5
        
        calibration_error = total_error / total_predictions
        return max(0.0, 1.0 - calibration_error)  # Higher score = better calibration
    
    def adjust_confidence(self, raw_confidence: float) -> float:
        """Adjust confidence based on calibration history"""
        calibration_score = self.get_calibration_score()
        
        if calibration_score > 0.8:
            # Well-calibrated, minimal adjustment
            return raw_confidence
        elif calibration_score > 0.6:
            # Moderately calibrated, slight adjustment toward 0.5
            return raw_confidence * 0.9 + 0.5 * 0.1
        else:
            # Poorly calibrated, stronger adjustment toward 0.5
            return raw_confidence * 0.7 + 0.5 * 0.3


class UncertaintyQuantifier:
    """Quantifies different types of uncertainty"""
    
    def __init__(self):
        self.uncertainty_factors = {
            UncertaintyType.EPISTEMIC: self._assess_epistemic_uncertainty,
            UncertaintyType.ALEATORIC: self._assess_aleatoric_uncertainty,
            UncertaintyType.MODEL: self._assess_model_uncertainty,
            UncertaintyType.LINGUISTIC: self._assess_linguistic_uncertainty
        }
    
    def _assess_epistemic_uncertainty(self, context: Dict[str, Any]) -> float:
        """Assess uncertainty due to lack of knowledge"""
        knowledge_gaps = context.get("knowledge_gaps", [])
        missing_information = context.get("missing_information", [])
        
        gap_score = len(knowledge_gaps) * 0.1
        missing_score = len(missing_information) * 0.15
        
        return min(gap_score + missing_score, 1.0)
    
    def _assess_aleatoric_uncertainty(self, context: Dict[str, Any]) -> float:
        """Assess uncertainty due to inherent randomness in data"""
        data_variance = context.get("data_variance", 0.0)
        measurement_error = context.get("measurement_error", 0.0)
        
        return min(data_variance + measurement_error, 1.0)
    
    def _assess_model_uncertainty(self, context: Dict[str, Any]) -> float:
        """Assess uncertainty due to model limitations"""
        model_complexity = context.get("model_complexity", 0.5)
        validation_score = context.get("validation_score", 0.8)
        
        complexity_penalty = model_complexity * 0.3
        validation_penalty = (1.0 - validation_score) * 0.5
        
        return min(complexity_penalty + validation_penalty, 1.0)
    
    def _assess_linguistic_uncertainty(self, context: Dict[str, Any]) -> float:
        """Assess uncertainty due to language ambiguity"""
        ambiguous_terms = context.get("ambiguous_terms", [])
        context_clarity = context.get("context_clarity", 0.8)
        
        ambiguity_score = len(ambiguous_terms) * 0.1
        clarity_penalty = (1.0 - context_clarity) * 0.4
        
        return min(ambiguity_score + clarity_penalty, 1.0)
    
    def quantify_uncertainty(self, context: Dict[str, Any]) -> Dict[UncertaintyType, float]:
        """Quantify all types of uncertainty for a given context"""
        uncertainties = {}
        
        for uncertainty_type, assessor in self.uncertainty_factors.items():
            uncertainties[uncertainty_type] = assessor(context)
        
        return uncertainties


class ErrorDetector:
    """Detects potential errors in reasoning"""
    
    def __init__(self):
        self.error_patterns = [
            self._check_logical_consistency,
            self._check_evidence_support,
            self._check_overconfidence,
            self._check_confirmation_bias,
            self._check_reasoning_gaps
        ]
    
    def _check_logical_consistency(self, thought: ThoughtMonitor, 
                                 other_thoughts: List[ThoughtMonitor]) -> List[str]:
        """Check for logical inconsistencies"""
        errors = []
        
        # Simplified consistency check
        for other in other_thoughts:
            if (thought.content and other.content and 
                "not" in thought.content.lower() and 
                thought.content.lower().replace("not ", "") in other.content.lower()):
                errors.append(f"Potential contradiction with thought {other.thought_id}")
        
        return errors
    
    def _check_evidence_support(self, thought: ThoughtMonitor, 
                              other_thoughts: List[ThoughtMonitor]) -> List[str]:
        """Check if conclusions are adequately supported by evidence"""
        errors = []
        
        evidence_ratio = len(thought.supporting_evidence) / max(len(thought.contradicting_evidence), 1)
        
        if thought.assessment.confidence > 0.7 and evidence_ratio < 2.0:
            errors.append("High confidence with insufficient supporting evidence")
        
        if len(thought.supporting_evidence) == 0 and thought.assessment.confidence > 0.5:
            errors.append("Moderate confidence with no supporting evidence")
        
        return errors
    
    def _check_overconfidence(self, thought: ThoughtMonitor, 
                            other_thoughts: List[ThoughtMonitor]) -> List[str]:
        """Check for overconfidence bias"""
        errors = []
        
        # High confidence with high uncertainty is suspicious
        total_uncertainty = sum(thought.assessment.uncertainty_types) if thought.assessment.uncertainty_types else 0
        
        if thought.assessment.confidence > 0.8 and total_uncertainty > 0.6:
            errors.append("Potential overconfidence - high confidence despite high uncertainty")
        
        # Very high confidence on complex novel topics is suspicious
        if (thought.assessment.confidence > 0.9 and 
            thought.assessment.novelty_score > 0.7 and 
            thought.assessment.complexity_score > 0.7):
            errors.append("Potential overconfidence on complex novel topic")
        
        return errors
    
    def _check_confirmation_bias(self, thought: ThoughtMonitor, 
                               other_thoughts: List[ThoughtMonitor]) -> List[str]:
        """Check for confirmation bias"""
        errors = []
        
        # Check if contradicting evidence is being ignored
        if (len(thought.contradicting_evidence) > 0 and 
            len(thought.supporting_evidence) / len(thought.contradicting_evidence) > 3):
            errors.append("Potential confirmation bias - contradicting evidence may be underweighted")
        
        return errors
    
    def _check_reasoning_gaps(self, thought: ThoughtMonitor, 
                            other_thoughts: List[ThoughtMonitor]) -> List[str]:
        """Check for gaps in reasoning chain"""
        errors = []
        
        # Check if dependencies are properly addressed
        for dep_id in thought.dependencies:
            dep_thought = next((t for t in other_thoughts if t.thought_id == dep_id), None)
            if not dep_thought:
                errors.append(f"Missing dependency: {dep_id}")
            elif dep_thought.assessment.confidence < 0.3:
                errors.append(f"Dependency {dep_id} has low confidence")
        
        return errors
    
    def detect_errors(self, thought: ThoughtMonitor, 
                     other_thoughts: List[ThoughtMonitor]) -> List[str]:
        """Detect all potential errors in a thought"""
        all_errors = []
        
        for error_checker in self.error_patterns:
            errors = error_checker(thought, other_thoughts)
            all_errors.extend(errors)
        
        return all_errors


class MetacognitiveMonitor:
    """Main metacognitive monitoring system"""
    
    def __init__(self):
        self.thoughts: List[ThoughtMonitor] = []
        self.calibrator = ConfidenceCalibrator()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.error_detector = ErrorDetector()
        self.monitoring_history: List[Dict[str, Any]] = []
    
    def _estimate_base_confidence(self, thought_content: str, context: Dict[str, Any]) -> float:
        """Estimate base confidence for a thought"""
        # Simplified confidence estimation based on content features
        confidence_factors = []
        
        # Length and complexity
        word_count = len(thought_content.split())
        if word_count > 20:  # Detailed thoughts tend to be more confident
            confidence_factors.append(0.1)
        
        # Certainty words
        certain_words = ["definitely", "certainly", "clearly", "obviously", "undoubtedly"]
        uncertain_words = ["maybe", "perhaps", "possibly", "might", "could"]
        
        certain_count = sum(1 for word in certain_words if word in thought_content.lower())
        uncertain_count = sum(1 for word in uncertain_words if word in thought_content.lower())
        
        confidence_factors.append((certain_count - uncertain_count) * 0.1)
        
        # Domain familiarity (from context)
        domain_familiarity = context.get("domain_familiarity", 0.5)
        confidence_factors.append(domain_familiarity * 0.3)
        
        # Available evidence
        evidence_count = len(context.get("supporting_evidence", []))
        confidence_factors.append(min(evidence_count * 0.1, 0.3))
        
        base_confidence = 0.5 + sum(confidence_factors)
        return max(0.0, min(base_confidence, 1.0))
    
    def _assess_reasoning_quality(self, assessment: MetacognitiveAssessment) -> ReasoningQuality:
        """Assess overall reasoning quality"""
        quality_score = (
            assessment.confidence * 0.3 +
            assessment.evidence_strength * 0.3 +
            assessment.coherence_score * 0.2 +
            (1.0 - assessment.complexity_score) * 0.1 +  # Lower complexity is better
            (1.0 - len(assessment.potential_errors) * 0.1) * 0.1
        )
        
        quality_score = max(0.0, min(quality_score, 1.0))
        
        if quality_score >= 0.8:
            return ReasoningQuality.EXCELLENT
        elif quality_score >= 0.6:
            return ReasoningQuality.GOOD
        elif quality_score >= 0.4:
            return ReasoningQuality.FAIR
        else:
            return ReasoningQuality.POOR
    
    def monitor_thought(self, thought_content: str, reasoning_type: str = "general",
                       context: Optional[Dict[str, Any]] = None) -> ThoughtMonitor:
        """Monitor a single thought and create metacognitive assessment"""
        if context is None:
            context = {}
        
        thought_id = f"thought_{len(self.thoughts)}"
        
        # Estimate base confidence
        base_confidence = self._estimate_base_confidence(thought_content, context)
        
        # Adjust confidence based on calibration
        adjusted_confidence = self.calibrator.adjust_confidence(base_confidence)
        
        # Quantify uncertainties
        uncertainties = self.uncertainty_quantifier.quantify_uncertainty(context)
        uncertainty_types = [unc_type for unc_type, value in uncertainties.items() if value > 0.3]
        
        # Create assessment
        assessment = MetacognitiveAssessment(
            confidence=adjusted_confidence,
            uncertainty_types=uncertainty_types,
            evidence_strength=context.get("evidence_strength", 0.5),
            coherence_score=context.get("coherence_score", 0.7),
            novelty_score=context.get("novelty_score", 0.3),
            complexity_score=context.get("complexity_score", 0.5)
        )
        
        # Assess reasoning quality
        assessment.reasoning_quality = self._assess_reasoning_quality(assessment)
        
        # Create thought monitor
        thought_monitor = ThoughtMonitor(
            thought_id=thought_id,
            content=thought_content,
            reasoning_type=reasoning_type,
            assessment=assessment,
            supporting_evidence=context.get("supporting_evidence", []),
            contradicting_evidence=context.get("contradicting_evidence", [])
        )
        
        # Detect potential errors
        potential_errors = self.error_detector.detect_errors(thought_monitor, self.thoughts)
        assessment.potential_errors = potential_errors
        
        # Generate improvement suggestions
        assessment.suggested_improvements = self._generate_improvement_suggestions(thought_monitor)
        
        # Store thought
        self.thoughts.append(thought_monitor)
        
        return thought_monitor
    
    def _generate_improvement_suggestions(self, thought: ThoughtMonitor) -> List[str]:
        """Generate suggestions for improving reasoning"""
        suggestions = []
        
        assessment = thought.assessment
        
        # Low confidence suggestions
        if assessment.confidence < 0.4:
            suggestions.append("Gather more supporting evidence")
            suggestions.append("Consider alternative perspectives")
        
        # High uncertainty suggestions
        if len(assessment.uncertainty_types) > 2:
            suggestions.append("Clarify ambiguous terms and concepts")
            suggestions.append("Seek additional domain expertise")
        
        # Low evidence strength suggestions
        if assessment.evidence_strength < 0.4:
            suggestions.append("Strengthen evidence base with more sources")
            suggestions.append("Validate evidence quality and reliability")
        
        # Low coherence suggestions
        if assessment.coherence_score < 0.5:
            suggestions.append("Check for logical consistency with other beliefs")
            suggestions.append("Resolve contradictions in reasoning")
        
        # High complexity suggestions
        if assessment.complexity_score > 0.8:
            suggestions.append("Break down complex reasoning into simpler steps")
            suggestions.append("Use analogies or examples to clarify")
        
        # Error-specific suggestions
        if "overconfidence" in " ".join(assessment.potential_errors):
            suggestions.append("Consider reasons why you might be wrong")
            suggestions.append("Seek disconfirming evidence")
        
        return suggestions
    
    def get_overall_confidence(self) -> float:
        """Get overall confidence across all monitored thoughts"""
        if not self.thoughts:
            return 0.5
        
        confidences = [t.assessment.confidence for t in self.thoughts]
        return statistics.mean(confidences)
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """Get a report on confidence calibration"""
        return {
            "calibration_score": self.calibrator.get_calibration_score(),
            "total_predictions": len(self.calibrator.prediction_history),
            "average_confidence": self.get_overall_confidence(),
            "confidence_distribution": self._get_confidence_distribution()
        }
    
    def _get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence levels"""
        distribution = {level.value: 0 for level in ConfidenceLevel}
        
        for thought in self.thoughts:
            conf = thought.assessment.confidence
            if conf < 0.2:
                distribution[ConfidenceLevel.VERY_LOW.value] += 1
            elif conf < 0.4:
                distribution[ConfidenceLevel.LOW.value] += 1
            elif conf < 0.6:
                distribution[ConfidenceLevel.MEDIUM.value] += 1
            elif conf < 0.8:
                distribution[ConfidenceLevel.HIGH.value] += 1
            else:
                distribution[ConfidenceLevel.VERY_HIGH.value] += 1
        
        return distribution
    
    def simulate_validation(self, thought_id: str, was_correct: bool):
        """Simulate validation of a thought for calibration training"""
        thought = next((t for t in self.thoughts if t.thought_id == thought_id), None)
        if thought:
            self.calibrator.add_prediction(thought.assessment.confidence, was_correct)
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        if not self.thoughts:
            return {"message": "No thoughts monitored yet"}
        
        quality_counts = {}
        for quality in ReasoningQuality:
            quality_counts[quality.value] = sum(
                1 for t in self.thoughts 
                if t.assessment.reasoning_quality == quality
            )
        
        error_types = {}
        for thought in self.thoughts:
            for error in thought.assessment.potential_errors:
                error_type = error.split()[0] if error else "unknown"
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_thoughts": len(self.thoughts),
            "average_confidence": self.get_overall_confidence(),
            "quality_distribution": quality_counts,
            "common_errors": error_types,
            "calibration_score": self.calibrator.get_calibration_score(),
            "improvement_opportunities": sum(
                len(t.assessment.suggested_improvements) for t in self.thoughts
            )
        }


def main():
    """Demonstration of the Metacognitive Monitoring pattern"""
    print("🧠 Metacognitive Monitoring Pattern Demonstration")
    print("=" * 80)
    print("This demonstrates an agent monitoring its own thinking:")
    print("- Confidence estimation and calibration")
    print("- Uncertainty quantification")
    print("- Error detection and self-correction")
    print("- Reasoning quality assessment")
    
    # Create monitor
    monitor = MetacognitiveMonitor()
    
    # Test thoughts with different contexts
    test_scenarios = [
        {
            "thought": "The capital of France is Paris",
            "context": {
                "domain_familiarity": 0.9,
                "supporting_evidence": ["geography textbook", "common knowledge"],
                "evidence_strength": 0.9,
                "coherence_score": 0.9,
                "novelty_score": 0.1,
                "complexity_score": 0.2
            },
            "reasoning_type": "factual_recall"
        },
        {
            "thought": "Artificial intelligence will likely achieve human-level capabilities within the next decade",
            "context": {
                "domain_familiarity": 0.6,
                "supporting_evidence": ["recent AI advances"],
                "contradicting_evidence": ["technical challenges", "timeline predictions have been wrong before"],
                "evidence_strength": 0.4,
                "coherence_score": 0.5,
                "novelty_score": 0.7,
                "complexity_score": 0.8,
                "knowledge_gaps": ["specific technical hurdles", "compute requirements"],
                "ambiguous_terms": ["human-level", "capabilities"]
            },
            "reasoning_type": "predictive"
        },
        {
            "thought": "This mathematical proof must be correct because it uses advanced techniques",
            "context": {
                "domain_familiarity": 0.3,
                "supporting_evidence": ["uses known theorems"],
                "evidence_strength": 0.3,
                "coherence_score": 0.4,
                "novelty_score": 0.8,
                "complexity_score": 0.9,
                "knowledge_gaps": ["advanced mathematical concepts", "proof verification methods"]
            },
            "reasoning_type": "logical_evaluation"
        }
    ]
    
    print(f"\n📝 Monitoring {len(test_scenarios)} thoughts...")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🧠 Thought {i}: {scenario['thought']}")
        print("-" * 60)
        
        thought_monitor = monitor.monitor_thought(
            scenario["thought"],
            scenario["reasoning_type"],
            scenario["context"]
        )
        
        assessment = thought_monitor.assessment
        
        print(f"💯 Confidence: {assessment.confidence:.2f}")
        print(f"🎯 Quality: {assessment.reasoning_quality.value}")
        print(f"🔍 Evidence Strength: {assessment.evidence_strength:.2f}")
        print(f"🧩 Coherence: {assessment.coherence_score:.2f}")
        print(f"⚠️ Uncertainty Types: {[u.value for u in assessment.uncertainty_types]}")
        
        if assessment.potential_errors:
            print(f"🚨 Potential Errors:")
            for error in assessment.potential_errors:
                print(f"   • {error}")
        
        if assessment.suggested_improvements:
            print(f"💡 Improvement Suggestions:")
            for suggestion in assessment.suggested_improvements[:3]:  # Show top 3
                print(f"   • {suggestion}")
        
        # Simulate validation for calibration
        # Higher quality thoughts are more likely to be correct
        validation_prob = 0.3 + (assessment.confidence * 0.6)
        was_correct = random.random() < validation_prob
        monitor.simulate_validation(thought_monitor.thought_id, was_correct)
        
        print(f"✅ Simulated validation: {'Correct' if was_correct else 'Incorrect'}")
    
    # Generate overall summary
    print(f"\n📊 Overall Monitoring Summary")
    print("=" * 60)
    
    summary = monitor.get_monitoring_summary()
    
    print(f"Total thoughts monitored: {summary['total_thoughts']}")
    print(f"Average confidence: {summary['average_confidence']:.2f}")
    print(f"Calibration score: {summary['calibration_score']:.2f}")
    
    print(f"\n📈 Quality Distribution:")
    for quality, count in summary['quality_distribution'].items():
        print(f"   {quality}: {count}")
    
    print(f"\n🚨 Common Error Types:")
    for error_type, count in summary['common_errors'].items():
        if count > 0:
            print(f"   {error_type}: {count}")
    
    print(f"\n💡 Total improvement opportunities: {summary['improvement_opportunities']}")
    
    # Calibration report
    calibration_report = monitor.get_calibration_report()
    print(f"\n🎯 Calibration Report:")
    print(f"   Calibration score: {calibration_report['calibration_score']:.2f}")
    print(f"   Total predictions: {calibration_report['total_predictions']}")
    
    print(f"\n📊 Confidence Distribution:")
    for level, count in calibration_report['confidence_distribution'].items():
        print(f"   {level}: {count}")
    
    print("\n\n🎯 Key Metacognitive Monitoring Features Demonstrated:")
    print("✅ Confidence estimation and calibration")
    print("✅ Multi-type uncertainty quantification")
    print("✅ Automatic error detection")
    print("✅ Reasoning quality assessment")
    print("✅ Self-improvement suggestions")
    print("✅ Evidence strength evaluation")
    print("✅ Logical consistency checking")
    print("✅ Overconfidence bias detection")


if __name__ == "__main__":
    main()