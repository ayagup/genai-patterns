"""
Pattern 123: Explainability Agent

This pattern implements explanation generation for agent decisions,
reasoning traces, counterfactual explanations, and decision justification.

Use Cases:
- AI transparency and trust
- Debugging agent behavior
- Regulatory compliance (XAI)
- User understanding and education
- Decision accountability

Category: Explainability & Transparency (1/4 = 25%) - NEW CATEGORY!
Complexity: Advanced
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime
import random


class ExplanationType(Enum):
    """Types of explanations."""
    CAUSAL = "causal"  # Why this decision
    CONTRASTIVE = "contrastive"  # Why this not that
    COUNTERFACTUAL = "counterfactual"  # What if scenarios
    TRACE_BASED = "trace_based"  # Step-by-step reasoning
    FEATURE_BASED = "feature_based"  # Which features mattered
    EXAMPLE_BASED = "example_based"  # Similar past cases


class ExplanationLevel(Enum):
    """Detail level of explanation."""
    BRIEF = "brief"  # One sentence
    STANDARD = "standard"  # Paragraph
    DETAILED = "detailed"  # Full breakdown
    TECHNICAL = "technical"  # Implementation details


@dataclass
class DecisionStep:
    """Single step in decision process."""
    step_num: int
    action: str
    reasoning: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    confidence: float
    alternatives_considered: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Feature:
    """Feature contributing to decision."""
    name: str
    value: Any
    importance: float  # 0.0 to 1.0
    contribution: str  # positive/negative/neutral


@dataclass
class Decision:
    """Agent decision to be explained."""
    decision_id: str
    action: str
    outcome: Any
    confidence: float
    context: Dict[str, Any]
    features: List[Feature]
    reasoning_trace: List[DecisionStep]
    alternatives: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Explanation:
    """Generated explanation."""
    explanation_type: ExplanationType
    level: ExplanationLevel
    text: str
    components: Dict[str, Any]  # Structured explanation parts
    confidence: float
    decision: Decision
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Counterfactual:
    """Counterfactual scenario."""
    original_decision: str
    alternative_decision: str
    changes_required: Dict[str, Tuple[Any, Any]]  # feature: (old, new)
    plausibility: float
    description: str


class CausalExplainer:
    """Generates causal explanations."""
    
    def explain(
        self,
        decision: Decision,
        level: ExplanationLevel = ExplanationLevel.STANDARD
    ) -> Explanation:
        """Generate causal explanation."""
        if level == ExplanationLevel.BRIEF:
            text = self._generate_brief_explanation(decision)
        elif level == ExplanationLevel.TECHNICAL:
            text = self._generate_technical_explanation(decision)
        else:
            text = self._generate_standard_explanation(decision)
        
        # Extract key causal factors
        causal_factors = self._identify_causal_factors(decision)
        
        return Explanation(
            explanation_type=ExplanationType.CAUSAL,
            level=level,
            text=text,
            components={
                'causal_factors': causal_factors,
                'decision_chain': [step.action for step in decision.reasoning_trace],
            },
            confidence=decision.confidence,
            decision=decision
        )
    
    def _generate_brief_explanation(self, decision: Decision) -> str:
        """Generate brief one-sentence explanation."""
        top_feature = max(decision.features, key=lambda f: f.importance)
        return (
            f"Decision: {decision.action} because {top_feature.name} "
            f"= {top_feature.value} (importance: {top_feature.importance:.2f})"
        )
    
    def _generate_standard_explanation(self, decision: Decision) -> str:
        """Generate standard paragraph explanation."""
        lines = [f"I decided to {decision.action} based on the following analysis:"]
        
        # Top 3 features
        top_features = sorted(decision.features, key=lambda f: f.importance, reverse=True)[:3]
        for i, feature in enumerate(top_features, 1):
            contribution_desc = "positively" if feature.contribution == "positive" else "negatively"
            lines.append(
                f"{i}. {feature.name} = {feature.value} "
                f"(contributed {contribution_desc}, importance: {feature.importance:.2f})"
            )
        
        # Alternatives considered
        if decision.alternatives:
            lines.append(
                f"\nI also considered: {', '.join(decision.alternatives)}, "
                f"but {decision.action} had higher confidence ({decision.confidence:.2f})."
            )
        
        return "\n".join(lines)
    
    def _generate_technical_explanation(self, decision: Decision) -> str:
        """Generate detailed technical explanation."""
        lines = [
            f"Decision ID: {decision.decision_id}",
            f"Action: {decision.action}",
            f"Confidence: {decision.confidence:.4f}",
            f"Timestamp: {decision.timestamp}",
            "\nReasoning Trace:"
        ]
        
        for step in decision.reasoning_trace:
            lines.append(f"  Step {step.step_num}: {step.action}")
            lines.append(f"    Reasoning: {step.reasoning}")
            lines.append(f"    Confidence: {step.confidence:.4f}")
            if step.alternatives_considered:
                lines.append(f"    Alternatives: {', '.join(step.alternatives_considered)}")
        
        lines.append("\nFeature Analysis:")
        for feature in sorted(decision.features, key=lambda f: f.importance, reverse=True):
            lines.append(
                f"  {feature.name}: {feature.value} "
                f"(importance={feature.importance:.4f}, contribution={feature.contribution})"
            )
        
        return "\n".join(lines)
    
    def _identify_causal_factors(self, decision: Decision) -> List[Dict[str, Any]]:
        """Identify key causal factors."""
        factors = []
        
        # High importance features are causal
        for feature in decision.features:
            if feature.importance > 0.5:
                factors.append({
                    'factor': feature.name,
                    'value': feature.value,
                    'strength': feature.importance,
                    'type': 'feature'
                })
        
        return factors


class ContrastiveExplainer:
    """Generates contrastive explanations (why this not that)."""
    
    def explain(
        self,
        decision: Decision,
        alternative: str,
        level: ExplanationLevel = ExplanationLevel.STANDARD
    ) -> Explanation:
        """Generate contrastive explanation."""
        differences = self._find_key_differences(decision, alternative)
        
        text = self._generate_contrastive_text(decision, alternative, differences, level)
        
        return Explanation(
            explanation_type=ExplanationType.CONTRASTIVE,
            level=level,
            text=text,
            components={
                'chosen': decision.action,
                'alternative': alternative,
                'key_differences': differences,
            },
            confidence=decision.confidence,
            decision=decision
        )
    
    def _find_key_differences(
        self,
        decision: Decision,
        alternative: str
    ) -> List[Dict[str, Any]]:
        """Find key differences explaining the choice."""
        differences = []
        
        # Features that favor the chosen action
        for feature in decision.features:
            if feature.importance > 0.3 and feature.contribution == "positive":
                differences.append({
                    'feature': feature.name,
                    'value': feature.value,
                    'favors': decision.action,
                    'reason': f"High positive contribution ({feature.importance:.2f})"
                })
        
        return differences
    
    def _generate_contrastive_text(
        self,
        decision: Decision,
        alternative: str,
        differences: List[Dict[str, Any]],
        level: ExplanationLevel
    ) -> str:
        """Generate contrastive explanation text."""
        if level == ExplanationLevel.BRIEF:
            if differences:
                diff = differences[0]
                return (
                    f"Chose {decision.action} over {alternative} because "
                    f"{diff['feature']} = {diff['value']}"
                )
            return f"Chose {decision.action} over {alternative} (higher confidence)"
        
        lines = [f"Why {decision.action} instead of {alternative}:"]
        
        if differences:
            for i, diff in enumerate(differences[:3], 1):
                lines.append(f"{i}. {diff['reason']}: {diff['feature']} = {diff['value']}")
        else:
            lines.append(f"Overall confidence in {decision.action}: {decision.confidence:.2f}")
        
        return "\n".join(lines)


class CounterfactualExplainer:
    """Generates counterfactual explanations."""
    
    def explain(
        self,
        decision: Decision,
        desired_outcome: Optional[str] = None,
        level: ExplanationLevel = ExplanationLevel.STANDARD
    ) -> Explanation:
        """Generate counterfactual explanation."""
        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(decision, desired_outcome)
        
        text = self._generate_counterfactual_text(decision, counterfactuals, level)
        
        return Explanation(
            explanation_type=ExplanationType.COUNTERFACTUAL,
            level=level,
            text=text,
            components={
                'original_decision': decision.action,
                'counterfactuals': counterfactuals,
            },
            confidence=decision.confidence,
            decision=decision
        )
    
    def _generate_counterfactuals(
        self,
        decision: Decision,
        desired_outcome: Optional[str] = None
    ) -> List[Counterfactual]:
        """Generate possible counterfactual scenarios."""
        counterfactuals = []
        
        # For each alternative
        alternatives = decision.alternatives if decision.alternatives else ["alternative_action"]
        
        for alt in alternatives[:2]:  # Top 2 alternatives
            # Find minimal changes needed
            changes = {}
            
            # Features that would need to change
            for feature in decision.features:
                if feature.importance > 0.4:
                    # Suggest plausible change
                    if isinstance(feature.value, (int, float)):
                        # Numeric: suggest threshold
                        if feature.contribution == "positive":
                            new_value = feature.value * 0.5  # Reduce
                        else:
                            new_value = feature.value * 1.5  # Increase
                        changes[feature.name] = (feature.value, new_value)
            
            # Calculate plausibility
            plausibility = 1.0 / (len(changes) + 1)  # Fewer changes = more plausible
            
            description = self._describe_counterfactual(decision.action, alt, changes)
            
            counterfactuals.append(Counterfactual(
                original_decision=decision.action,
                alternative_decision=alt,
                changes_required=changes,
                plausibility=plausibility,
                description=description
            ))
        
        return counterfactuals
    
    def _describe_counterfactual(
        self,
        original: str,
        alternative: str,
        changes: Dict[str, Tuple[Any, Any]]
    ) -> str:
        """Describe a counterfactual scenario."""
        if not changes:
            return f"If conditions were different, {alternative} might have been chosen."
        
        change_desc = []
        for feature, (old, new) in list(changes.items())[:2]:
            if isinstance(old, (int, float)):
                change_desc.append(f"{feature} changed from {old:.2f} to {new:.2f}")
            else:
                change_desc.append(f"{feature} changed from {old} to {new}")
        
        return (
            f"If {' and '.join(change_desc)}, "
            f"the decision would have been {alternative} instead of {original}."
        )
    
    def _generate_counterfactual_text(
        self,
        decision: Decision,
        counterfactuals: List[Counterfactual],
        level: ExplanationLevel
    ) -> str:
        """Generate counterfactual explanation text."""
        if level == ExplanationLevel.BRIEF:
            if counterfactuals:
                return counterfactuals[0].description
            return f"No simple counterfactual found for {decision.action}"
        
        lines = [f"What would change the decision from {decision.action}:"]
        
        for i, cf in enumerate(counterfactuals, 1):
            lines.append(f"\nScenario {i} (plausibility: {cf.plausibility:.2f}):")
            lines.append(f"  {cf.description}")
            if cf.changes_required:
                lines.append("  Required changes:")
                for feature, (old, new) in cf.changes_required.items():
                    lines.append(f"    - {feature}: {old} → {new}")
        
        return "\n".join(lines)


class TraceBasedExplainer:
    """Generates trace-based explanations."""
    
    def explain(
        self,
        decision: Decision,
        level: ExplanationLevel = ExplanationLevel.STANDARD
    ) -> Explanation:
        """Generate trace-based explanation."""
        text = self._generate_trace_text(decision.reasoning_trace, level)
        
        return Explanation(
            explanation_type=ExplanationType.TRACE_BASED,
            level=level,
            text=text,
            components={
                'trace': [
                    {
                        'step': step.step_num,
                        'action': step.action,
                        'reasoning': step.reasoning
                    }
                    for step in decision.reasoning_trace
                ]
            },
            confidence=decision.confidence,
            decision=decision
        )
    
    def _generate_trace_text(
        self,
        trace: List[DecisionStep],
        level: ExplanationLevel
    ) -> str:
        """Generate trace explanation text."""
        if level == ExplanationLevel.BRIEF:
            if trace:
                return f"Decision process: {' → '.join(step.action for step in trace)}"
            return "No reasoning trace available"
        
        lines = ["Step-by-step reasoning:"]
        
        for step in trace:
            lines.append(f"\nStep {step.step_num}: {step.action}")
            lines.append(f"  Reasoning: {step.reasoning}")
            
            if level == ExplanationLevel.DETAILED or level == ExplanationLevel.TECHNICAL:
                lines.append(f"  Confidence: {step.confidence:.2f}")
                if step.alternatives_considered:
                    lines.append(f"  Alternatives: {', '.join(step.alternatives_considered)}")
        
        return "\n".join(lines)


class ExplainabilityAgent:
    """Agent for generating explanations of decisions."""
    
    def __init__(self):
        self.causal_explainer = CausalExplainer()
        self.contrastive_explainer = ContrastiveExplainer()
        self.counterfactual_explainer = CounterfactualExplainer()
        self.trace_explainer = TraceBasedExplainer()
        
        # History
        self.decisions: Dict[str, Decision] = {}
        self.explanations: List[Explanation] = []
    
    def record_decision(self, decision: Decision) -> None:
        """Record a decision for later explanation."""
        self.decisions[decision.decision_id] = decision
    
    def explain(
        self,
        decision: Decision,
        explanation_type: ExplanationType = ExplanationType.CAUSAL,
        level: ExplanationLevel = ExplanationLevel.STANDARD,
        **kwargs
    ) -> Explanation:
        """Generate explanation for a decision."""
        if explanation_type == ExplanationType.CAUSAL:
            explanation = self.causal_explainer.explain(decision, level)
        elif explanation_type == ExplanationType.CONTRASTIVE:
            alternative = kwargs.get('alternative', decision.alternatives[0] if decision.alternatives else 'other')
            explanation = self.contrastive_explainer.explain(decision, alternative, level)
        elif explanation_type == ExplanationType.COUNTERFACTUAL:
            desired_outcome = kwargs.get('desired_outcome')
            explanation = self.counterfactual_explainer.explain(decision, desired_outcome, level)
        elif explanation_type == ExplanationType.TRACE_BASED:
            explanation = self.trace_explainer.explain(decision, level)
        else:
            explanation = self.causal_explainer.explain(decision, level)
        
        self.explanations.append(explanation)
        return explanation
    
    def explain_multiple_types(
        self,
        decision: Decision,
        types: List[ExplanationType],
        level: ExplanationLevel = ExplanationLevel.STANDARD
    ) -> Dict[ExplanationType, Explanation]:
        """Generate multiple types of explanations."""
        explanations = {}
        
        for exp_type in types:
            explanation = self.explain(decision, exp_type, level)
            explanations[exp_type] = explanation
        
        return explanations
    
    def compare_decisions(
        self,
        decision1: Decision,
        decision2: Decision
    ) -> str:
        """Compare two decisions."""
        lines = [
            "Decision Comparison:",
            f"\nDecision 1: {decision1.action} (confidence: {decision1.confidence:.2f})",
            f"Decision 2: {decision2.action} (confidence: {decision2.confidence:.2f})",
            "\nKey Differences:"
        ]
        
        # Compare features
        features1 = {f.name: f for f in decision1.features}
        features2 = {f.name: f for f in decision2.features}
        
        all_features = set(features1.keys()) | set(features2.keys())
        
        for fname in all_features:
            if fname in features1 and fname in features2:
                f1, f2 = features1[fname], features2[fname]
                if f1.value != f2.value:
                    lines.append(f"  {fname}: {f1.value} vs {f2.value}")
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get explanation statistics."""
        if not self.explanations:
            return {}
        
        type_counts = {}
        for exp in self.explanations:
            exp_type = exp.explanation_type.value
            type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
        
        avg_confidence = sum(e.confidence for e in self.explanations) / len(self.explanations)
        
        return {
            'total_explanations': len(self.explanations),
            'total_decisions': len(self.decisions),
            'explanation_types': type_counts,
            'average_confidence': avg_confidence,
        }


def demonstrate_explainability():
    """Demonstrate the Explainability Agent."""
    print("=" * 60)
    print("Explainability Agent Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = ExplainabilityAgent()
    
    # Create a sample decision
    print("\n1. SAMPLE DECISION SCENARIO")
    print("-" * 60)
    print("   Scenario: Loan Approval Decision")
    
    decision = Decision(
        decision_id="loan_001",
        action="approve_loan",
        outcome="approved",
        confidence=0.87,
        context={
            'applicant_id': 'A123',
            'loan_amount': 50000,
            'purpose': 'home_improvement'
        },
        features=[
            Feature("credit_score", 750, importance=0.85, contribution="positive"),
            Feature("annual_income", 85000, importance=0.70, contribution="positive"),
            Feature("debt_to_income_ratio", 0.25, importance=0.65, contribution="positive"),
            Feature("employment_years", 8, importance=0.45, contribution="positive"),
            Feature("loan_amount", 50000, importance=0.30, contribution="neutral"),
        ],
        reasoning_trace=[
            DecisionStep(
                1, "assess_creditworthiness",
                "High credit score indicates low risk",
                {'credit_score': 750}, {'risk_level': 'low'},
                0.90, ['skip_assessment']
            ),
            DecisionStep(
                2, "evaluate_income",
                "Income sufficient to support loan payments",
                {'income': 85000, 'loan': 50000}, {'income_adequate': True},
                0.85, ['request_additional_income_proof']
            ),
            DecisionStep(
                3, "check_debt_ratio",
                "Debt-to-income ratio is healthy",
                {'ratio': 0.25}, {'debt_manageable': True},
                0.88
            ),
            DecisionStep(
                4, "make_decision",
                "All factors favorable for approval",
                {'risk': 'low', 'income': 'adequate', 'debt': 'manageable'},
                {'decision': 'approve'},
                0.87, ['approve_with_conditions', 'deny', 'request_more_info']
            ),
        ],
        alternatives=["deny_loan", "approve_with_conditions", "request_more_information"]
    )
    
    agent.record_decision(decision)
    
    # Generate different types of explanations
    print("\n2. CAUSAL EXPLANATION (Why approved?)")
    print("-" * 60)
    
    causal_exp = agent.explain(decision, ExplanationType.CAUSAL, ExplanationLevel.STANDARD)
    print(causal_exp.text)
    
    print("\n3. CONTRASTIVE EXPLANATION (Why approve, not deny?)")
    print("-" * 60)
    
    contrastive_exp = agent.explain(
        decision,
        ExplanationType.CONTRASTIVE,
        ExplanationLevel.STANDARD,
        alternative="deny_loan"
    )
    print(contrastive_exp.text)
    
    print("\n4. COUNTERFACTUAL EXPLANATION (What would change outcome?)")
    print("-" * 60)
    
    counterfactual_exp = agent.explain(
        decision,
        ExplanationType.COUNTERFACTUAL,
        ExplanationLevel.STANDARD
    )
    print(counterfactual_exp.text)
    
    print("\n5. TRACE-BASED EXPLANATION (Step-by-step)")
    print("-" * 60)
    
    trace_exp = agent.explain(decision, ExplanationType.TRACE_BASED, ExplanationLevel.DETAILED)
    print(trace_exp.text)
    
    # Brief explanation
    print("\n6. BRIEF EXPLANATION (One sentence)")
    print("-" * 60)
    
    brief_exp = agent.explain(decision, ExplanationType.CAUSAL, ExplanationLevel.BRIEF)
    print(brief_exp.text)
    
    # Multiple types at once
    print("\n7. COMPREHENSIVE EXPLANATION")
    print("-" * 60)
    
    all_types = [
        ExplanationType.CAUSAL,
        ExplanationType.CONTRASTIVE,
        ExplanationType.COUNTERFACTUAL
    ]
    
    all_explanations = agent.explain_multiple_types(decision, all_types, ExplanationLevel.BRIEF)
    
    for exp_type, explanation in all_explanations.items():
        print(f"\n   {exp_type.value.upper()}:")
        print(f"   {explanation.text}")
    
    # Statistics
    print("\n8. EXPLANATION STATISTICS")
    print("-" * 60)
    
    stats = agent.get_statistics()
    print(f"   Total Explanations Generated: {stats['total_explanations']}")
    print(f"   Total Decisions Explained: {stats['total_decisions']}")
    print(f"   Average Confidence: {stats['average_confidence']:.2f}")
    print(f"   Explanation Types Used:")
    for exp_type, count in stats['explanation_types'].items():
        print(f"   - {exp_type}: {count}")
    
    print("\n" + "=" * 60)
    print("🎉 NEW CATEGORY: Explainability & Transparency (25%)")
    print("Pattern 123 starts the Explainability category!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_explainability()
