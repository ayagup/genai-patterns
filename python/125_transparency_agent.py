"""
Pattern 125: Transparency Agent

This pattern implements decision documentation, audit trails, bias detection,
and fairness metrics for AI transparency.

Use Cases:
- Regulatory compliance
- Algorithmic accountability
- Bias mitigation
- Fair AI systems
- Trust and governance

Category: Explainability & Transparency (3/4 = 75%)
Complexity: Advanced
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
import json
import hashlib


class TransparencyLevel(Enum):
    """Levels of transparency."""
    MINIMAL = "minimal"  # Basic info only
    STANDARD = "standard"  # Standard documentation
    COMPREHENSIVE = "comprehensive"  # Full details
    AUDIT_READY = "audit_ready"  # Regulatory compliance


class BiasType(Enum):
    """Types of bias to detect."""
    SELECTION_BIAS = "selection_bias"
    CONFIRMATION_BIAS = "confirmation_bias"
    DEMOGRAPHIC_BIAS = "demographic_bias"
    ALGORITHMIC_BIAS = "algorithmic_bias"
    REPRESENTATION_BIAS = "representation_bias"


class FairnessMetric(Enum):
    """Fairness metrics."""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    EQUALIZED_ODDS = "equalized_odds"
    PREDICTIVE_PARITY = "predictive_parity"


@dataclass
class DecisionRecord:
    """Complete record of a decision."""
    decision_id: str
    timestamp: datetime
    action: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    reasoning: str
    confidence: float
    alternatives_considered: List[str]
    model_version: str
    user_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'decision_id': self.decision_id,
            'timestamp': self.timestamp.isoformat(),
            'action': self.action,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'alternatives': self.alternatives_considered,
            'model_version': self.model_version,
            'user_id': self.user_id,
            'context': self.context
        }
    
    def get_hash(self) -> str:
        """Get cryptographic hash for integrity."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class AuditEntry:
    """Single audit trail entry."""
    entry_id: str
    timestamp: datetime
    event_type: str
    actor: str
    action: str
    details: Dict[str, Any]
    previous_hash: Optional[str] = None
    entry_hash: Optional[str] = None
    
    def compute_hash(self) -> str:
        """Compute entry hash."""
        content = {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'actor': self.actor,
            'action': self.action,
            'details': self.details,
            'previous_hash': self.previous_hash
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


@dataclass
class BiasReport:
    """Report on detected bias."""
    bias_type: BiasType
    severity: float  # 0.0 to 1.0
    affected_groups: List[str]
    evidence: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FairnessAnalysis:
    """Analysis of fairness metrics."""
    metric: FairnessMetric
    score: float
    protected_attribute: str
    group_scores: Dict[str, float]
    passes_threshold: bool
    threshold: float = 0.8


class DecisionDocumenter:
    """Documents decisions for transparency."""
    
    def __init__(self, transparency_level: TransparencyLevel = TransparencyLevel.STANDARD):
        self.transparency_level = transparency_level
        self.decisions: Dict[str, DecisionRecord] = {}
    
    def document_decision(
        self,
        decision_id: str,
        action: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reasoning: str,
        confidence: float,
        alternatives: List[str],
        model_version: str,
        **kwargs
    ) -> DecisionRecord:
        """Document a decision."""
        record = DecisionRecord(
            decision_id=decision_id,
            timestamp=datetime.now(),
            action=action,
            inputs=inputs,
            outputs=outputs,
            reasoning=reasoning,
            confidence=confidence,
            alternatives_considered=alternatives,
            model_version=model_version,
            user_id=kwargs.get('user_id'),
            context=kwargs.get('context', {})
        )
        
        self.decisions[decision_id] = record
        return record
    
    def get_decision_summary(self, decision_id: str) -> str:
        """Get human-readable decision summary."""
        if decision_id not in self.decisions:
            return "Decision not found"
        
        record = self.decisions[decision_id]
        
        if self.transparency_level == TransparencyLevel.MINIMAL:
            return f"Decision: {record.action} at {record.timestamp}"
        
        lines = [
            f"Decision ID: {record.decision_id}",
            f"Timestamp: {record.timestamp}",
            f"Action: {record.action}",
            f"Confidence: {record.confidence:.2f}",
        ]
        
        if self.transparency_level in [TransparencyLevel.COMPREHENSIVE, TransparencyLevel.AUDIT_READY]:
            lines.extend([
                f"Model Version: {record.model_version}",
                f"\nInputs: {json.dumps(record.inputs, indent=2)}",
                f"\nOutputs: {json.dumps(record.outputs, indent=2)}",
                f"\nReasoning: {record.reasoning}",
                f"\nAlternatives Considered: {', '.join(record.alternatives_considered)}",
            ])
        
        if self.transparency_level == TransparencyLevel.AUDIT_READY:
            lines.append(f"\nHash: {record.get_hash()}")
        
        return "\n".join(lines)
    
    def export_decisions(self, start_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Export decisions for audit."""
        decisions = []
        
        for record in self.decisions.values():
            if start_date and record.timestamp < start_date:
                continue
            
            decisions.append(record.to_dict())
        
        return decisions


class AuditTrailManager:
    """Manages tamper-evident audit trail."""
    
    def __init__(self):
        self.entries: List[AuditEntry] = []
        self.entry_index: Dict[str, AuditEntry] = {}
    
    def log_event(
        self,
        event_type: str,
        actor: str,
        action: str,
        details: Dict[str, Any]
    ) -> AuditEntry:
        """Log an auditable event."""
        entry_id = f"audit_{len(self.entries) + 1}_{datetime.now().timestamp()}"
        
        # Get previous hash for chain
        previous_hash = None
        if self.entries:
            previous_hash = self.entries[-1].entry_hash
        
        entry = AuditEntry(
            entry_id=entry_id,
            timestamp=datetime.now(),
            event_type=event_type,
            actor=actor,
            action=action,
            details=details,
            previous_hash=previous_hash
        )
        
        entry.entry_hash = entry.compute_hash()
        
        self.entries.append(entry)
        self.entry_index[entry_id] = entry
        
        return entry
    
    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """Verify audit trail integrity."""
        issues = []
        
        for i, entry in enumerate(self.entries):
            # Verify hash
            expected_hash = entry.compute_hash()
            if entry.entry_hash != expected_hash:
                issues.append(f"Entry {entry.entry_id}: Hash mismatch")
            
            # Verify chain
            if i > 0:
                prev_entry = self.entries[i-1]
                if entry.previous_hash != prev_entry.entry_hash:
                    issues.append(f"Entry {entry.entry_id}: Chain broken")
        
        return len(issues) == 0, issues
    
    def query_events(
        self,
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditEntry]:
        """Query audit trail."""
        results = []
        
        for entry in self.entries:
            if event_type and entry.event_type != event_type:
                continue
            if actor and entry.actor != actor:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            
            results.append(entry)
        
        return results
    
    def get_summary(self) -> str:
        """Get audit trail summary."""
        if not self.entries:
            return "No audit entries"
        
        event_types = {}
        actors = set()
        
        for entry in self.entries:
            event_types[entry.event_type] = event_types.get(entry.event_type, 0) + 1
            actors.add(entry.actor)
        
        lines = [
            f"Audit Trail Summary:",
            f"Total Entries: {len(self.entries)}",
            f"First Entry: {self.entries[0].timestamp}",
            f"Last Entry: {self.entries[-1].timestamp}",
            f"Unique Actors: {len(actors)}",
            f"\nEvent Types:"
        ]
        
        for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {event_type}: {count}")
        
        # Integrity check
        is_valid, issues = self.verify_integrity()
        lines.append(f"\nIntegrity: {'✓ Valid' if is_valid else '✗ Compromised'}")
        
        return "\n".join(lines)


class BiasDetector:
    """Detects bias in decisions."""
    
    def __init__(self):
        self.bias_reports: List[BiasReport] = []
    
    def detect_demographic_bias(
        self,
        decisions: List[DecisionRecord],
        protected_attribute: str
    ) -> Optional[BiasReport]:
        """Detect demographic bias."""
        if not decisions:
            return None
        
        # Group decisions by protected attribute
        groups: Dict[str, List[float]] = {}
        
        for decision in decisions:
            attr_value = decision.inputs.get(protected_attribute)
            if attr_value is not None:
                if attr_value not in groups:
                    groups[attr_value] = []
                
                # Use confidence as proxy for positive outcome
                groups[attr_value].append(decision.confidence)
        
        if len(groups) < 2:
            return None
        
        # Calculate average confidence per group
        group_averages = {
            group: sum(confs) / len(confs)
            for group, confs in groups.items()
        }
        
        # Find disparity
        max_avg = max(group_averages.values())
        min_avg = min(group_averages.values())
        disparity = max_avg - min_avg
        
        # Severity based on disparity
        severity = min(1.0, disparity * 2)  # Normalize
        
        if severity > 0.2:  # Threshold for reporting
            evidence = [
                f"Group '{group}': avg confidence = {avg:.2f}"
                for group, avg in group_averages.items()
            ]
            
            affected = [
                group for group, avg in group_averages.items()
                if avg < max_avg * 0.9
            ]
            
            report = BiasReport(
                bias_type=BiasType.DEMOGRAPHIC_BIAS,
                severity=severity,
                affected_groups=affected,
                evidence=evidence,
                recommendations=[
                    f"Review decision logic for {protected_attribute}",
                    "Ensure training data is representative",
                    "Consider fairness constraints"
                ]
            )
            
            self.bias_reports.append(report)
            return report
        
        return None
    
    def detect_selection_bias(
        self,
        decisions: List[DecisionRecord],
        expected_distribution: Dict[str, float]
    ) -> Optional[BiasReport]:
        """Detect selection bias in decision distribution."""
        if not decisions:
            return None
        
        # Count actual decisions
        actual_counts = {}
        for decision in decisions:
            action = decision.action
            actual_counts[action] = actual_counts.get(action, 0) + 1
        
        # Calculate actual distribution
        total = len(decisions)
        actual_dist = {
            action: count / total
            for action, count in actual_counts.items()
        }
        
        # Compare with expected
        deviations = []
        max_deviation = 0
        
        for action, expected_rate in expected_distribution.items():
            actual_rate = actual_dist.get(action, 0)
            deviation = abs(expected_rate - actual_rate)
            deviations.append((action, deviation))
            max_deviation = max(max_deviation, deviation)
        
        if max_deviation > 0.15:  # 15% threshold
            evidence = [
                f"{action}: expected {exp:.1%}, actual {actual_dist.get(action, 0):.1%}"
                for action, exp in expected_distribution.items()
            ]
            
            report = BiasReport(
                bias_type=BiasType.SELECTION_BIAS,
                severity=min(1.0, max_deviation * 2),
                affected_groups=[action for action, dev in deviations if dev > 0.1],
                evidence=evidence,
                recommendations=[
                    "Review selection criteria",
                    "Check for systematic patterns",
                    "Validate decision thresholds"
                ]
            )
            
            self.bias_reports.append(report)
            return report
        
        return None
    
    def get_bias_summary(self) -> str:
        """Get summary of detected biases."""
        if not self.bias_reports:
            return "No biases detected"
        
        lines = ["Bias Detection Summary:"]
        
        for i, report in enumerate(self.bias_reports, 1):
            lines.append(f"\n{i}. {report.bias_type.value} (severity: {report.severity:.2f})")
            lines.append(f"   Affected: {', '.join(report.affected_groups)}")
            lines.append("   Evidence:")
            for evidence in report.evidence:
                lines.append(f"   - {evidence}")
        
        return "\n".join(lines)


class FairnessEvaluator:
    """Evaluates fairness metrics."""
    
    def __init__(self, fairness_threshold: float = 0.8):
        self.fairness_threshold = fairness_threshold
        self.analyses: List[FairnessAnalysis] = []
    
    def evaluate_demographic_parity(
        self,
        decisions: List[DecisionRecord],
        protected_attribute: str
    ) -> FairnessAnalysis:
        """Evaluate demographic parity."""
        # Group decisions
        groups: Dict[str, List[bool]] = {}
        
        for decision in decisions:
            attr_value = str(decision.inputs.get(protected_attribute, 'unknown'))
            if attr_value not in groups:
                groups[attr_value] = []
            
            # Positive outcome = high confidence
            positive = decision.confidence > 0.5
            groups[attr_value].append(positive)
        
        # Calculate positive rate per group
        group_scores = {}
        for group, outcomes in groups.items():
            positive_rate = sum(outcomes) / len(outcomes) if outcomes else 0
            group_scores[group] = positive_rate
        
        # Overall score: ratio of min to max
        if group_scores:
            min_rate = min(group_scores.values())
            max_rate = max(group_scores.values())
            score = min_rate / max_rate if max_rate > 0 else 1.0
        else:
            score = 1.0
        
        passes = score >= self.fairness_threshold
        
        analysis = FairnessAnalysis(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            score=score,
            protected_attribute=protected_attribute,
            group_scores=group_scores,
            passes_threshold=passes,
            threshold=self.fairness_threshold
        )
        
        self.analyses.append(analysis)
        return analysis
    
    def evaluate_equal_opportunity(
        self,
        decisions: List[DecisionRecord],
        protected_attribute: str,
        favorable_label: str = "approve"
    ) -> FairnessAnalysis:
        """Evaluate equal opportunity."""
        # True positive rates per group
        groups: Dict[str, Tuple[int, int]] = {}  # (true_positives, total_positives)
        
        for decision in decisions:
            attr_value = str(decision.inputs.get(protected_attribute, 'unknown'))
            
            # Assuming ground truth in context
            actual_outcome = decision.context.get('actual_outcome', decision.action)
            
            if actual_outcome == favorable_label:
                if attr_value not in groups:
                    groups[attr_value] = (0, 0)
                
                tp, total = groups[attr_value]
                total += 1
                
                if decision.action == favorable_label:
                    tp += 1
                
                groups[attr_value] = (tp, total)
        
        # Calculate TPR per group
        group_scores = {}
        for group, (tp, total) in groups.items():
            tpr = tp / total if total > 0 else 0
            group_scores[group] = tpr
        
        # Score: ratio of min to max TPR
        if group_scores:
            min_tpr = min(group_scores.values())
            max_tpr = max(group_scores.values())
            score = min_tpr / max_tpr if max_tpr > 0 else 1.0
        else:
            score = 1.0
        
        passes = score >= self.fairness_threshold
        
        analysis = FairnessAnalysis(
            metric=FairnessMetric.EQUAL_OPPORTUNITY,
            score=score,
            protected_attribute=protected_attribute,
            group_scores=group_scores,
            passes_threshold=passes,
            threshold=self.fairness_threshold
        )
        
        self.analyses.append(analysis)
        return analysis
    
    def get_fairness_report(self) -> str:
        """Get comprehensive fairness report."""
        if not self.analyses:
            return "No fairness analyses performed"
        
        lines = ["Fairness Evaluation Report:"]
        
        for i, analysis in enumerate(self.analyses, 1):
            status = "✓ PASS" if analysis.passes_threshold else "✗ FAIL"
            lines.append(f"\n{i}. {analysis.metric.value} {status}")
            lines.append(f"   Score: {analysis.score:.2f} (threshold: {analysis.threshold})")
            lines.append(f"   Protected Attribute: {analysis.protected_attribute}")
            lines.append("   Group Scores:")
            for group, score in sorted(analysis.group_scores.items()):
                lines.append(f"   - {group}: {score:.2f}")
        
        return "\n".join(lines)


class TransparencyAgent:
    """Agent for ensuring transparency and accountability."""
    
    def __init__(
        self,
        transparency_level: TransparencyLevel = TransparencyLevel.STANDARD,
        fairness_threshold: float = 0.8
    ):
        self.documenter = DecisionDocumenter(transparency_level)
        self.audit_trail = AuditTrailManager()
        self.bias_detector = BiasDetector()
        self.fairness_evaluator = FairnessEvaluator(fairness_threshold)
        
        # Log initialization
        self.audit_trail.log_event(
            "system",
            "TransparencyAgent",
            "initialized",
            {'transparency_level': transparency_level.value}
        )
    
    def record_decision(
        self,
        decision_id: str,
        action: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        reasoning: str,
        confidence: float,
        alternatives: List[str],
        model_version: str,
        **kwargs
    ) -> DecisionRecord:
        """Record a decision with full transparency."""
        # Document decision
        record = self.documenter.document_decision(
            decision_id, action, inputs, outputs, reasoning,
            confidence, alternatives, model_version, **kwargs
        )
        
        # Log to audit trail
        self.audit_trail.log_event(
            "decision",
            kwargs.get('user_id', 'system'),
            f"made_decision: {action}",
            {
                'decision_id': decision_id,
                'confidence': confidence,
                'model_version': model_version
            }
        )
        
        return record
    
    def analyze_bias(
        self,
        protected_attribute: str,
        expected_distribution: Optional[Dict[str, float]] = None
    ) -> List[BiasReport]:
        """Analyze decisions for bias."""
        decisions = list(self.documenter.decisions.values())
        reports = []
        
        # Demographic bias
        report = self.bias_detector.detect_demographic_bias(decisions, protected_attribute)
        if report:
            reports.append(report)
        
        # Selection bias
        if expected_distribution:
            report = self.bias_detector.detect_selection_bias(decisions, expected_distribution)
            if report:
                reports.append(report)
        
        # Log analysis
        self.audit_trail.log_event(
            "bias_analysis",
            "TransparencyAgent",
            "performed_bias_check",
            {
                'protected_attribute': protected_attribute,
                'biases_found': len(reports)
            }
        )
        
        return reports
    
    def evaluate_fairness(
        self,
        protected_attribute: str,
        metrics: Optional[List[FairnessMetric]] = None
    ) -> List[FairnessAnalysis]:
        """Evaluate fairness metrics."""
        if metrics is None:
            metrics = [FairnessMetric.DEMOGRAPHIC_PARITY]
        
        decisions = list(self.documenter.decisions.values())
        analyses = []
        
        for metric in metrics:
            if metric == FairnessMetric.DEMOGRAPHIC_PARITY:
                analysis = self.fairness_evaluator.evaluate_demographic_parity(
                    decisions, protected_attribute
                )
                analyses.append(analysis)
            elif metric == FairnessMetric.EQUAL_OPPORTUNITY:
                analysis = self.fairness_evaluator.evaluate_equal_opportunity(
                    decisions, protected_attribute
                )
                analyses.append(analysis)
        
        # Log evaluation
        self.audit_trail.log_event(
            "fairness_evaluation",
            "TransparencyAgent",
            "evaluated_fairness",
            {
                'protected_attribute': protected_attribute,
                'metrics': [m.value for m in metrics]
            }
        )
        
        return analyses
    
    def generate_transparency_report(self) -> str:
        """Generate comprehensive transparency report."""
        lines = [
            "=" * 60,
            "TRANSPARENCY REPORT",
            "=" * 60,
            f"\nGenerated: {datetime.now()}",
            f"\n{self.audit_trail.get_summary()}",
            f"\n\n{self.bias_detector.get_bias_summary()}",
            f"\n\n{self.fairness_evaluator.get_fairness_report()}",
        ]
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get transparency statistics."""
        return {
            'total_decisions': len(self.documenter.decisions),
            'audit_entries': len(self.audit_trail.entries),
            'bias_reports': len(self.bias_detector.bias_reports),
            'fairness_analyses': len(self.fairness_evaluator.analyses),
            'audit_integrity': self.audit_trail.verify_integrity()[0]
        }


def demonstrate_transparency():
    """Demonstrate the Transparency Agent."""
    print("=" * 60)
    print("Transparency Agent Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = TransparencyAgent(
        transparency_level=TransparencyLevel.COMPREHENSIVE,
        fairness_threshold=0.8
    )
    
    print("\n1. RECORDING DECISIONS")
    print("-" * 60)
    
    # Simulate some loan decisions
    test_cases = [
        {'age': 25, 'gender': 'M', 'income': 50000, 'decision': 'deny', 'conf': 0.7},
        {'age': 40, 'gender': 'F', 'income': 80000, 'decision': 'approve', 'conf': 0.9},
        {'age': 30, 'gender': 'M', 'income': 70000, 'decision': 'approve', 'conf': 0.85},
        {'age': 35, 'gender': 'F', 'income': 60000, 'decision': 'deny', 'conf': 0.65},
        {'age': 45, 'gender': 'M', 'income': 90000, 'decision': 'approve', 'conf': 0.95},
        {'age': 28, 'gender': 'F', 'income': 55000, 'decision': 'approve', 'conf': 0.75},
    ]
    
    for i, case in enumerate(test_cases, 1):
        agent.record_decision(
            decision_id=f"loan_{i}",
            action=case['decision'],
            inputs={'age': case['age'], 'gender': case['gender'], 'income': case['income']},
            outputs={'decision': case['decision']},
            reasoning=f"Based on income and credit profile",
            confidence=case['conf'],
            alternatives=['approve', 'deny'],
            model_version="v1.2.0",
            user_id=f"agent_{i % 2}"
        )
    
    print(f"   Recorded {len(test_cases)} decisions")
    
    # Show one decision in detail
    print("\n2. DECISION DOCUMENTATION")
    print("-" * 60)
    summary = agent.documenter.get_decision_summary("loan_1")
    print(summary)
    
    # Bias analysis
    print("\n\n3. BIAS DETECTION")
    print("-" * 60)
    
    bias_reports = agent.analyze_bias(
        protected_attribute='gender',
        expected_distribution={'approve': 0.6, 'deny': 0.4}
    )
    
    print(agent.bias_detector.get_bias_summary())
    
    # Fairness evaluation
    print("\n\n4. FAIRNESS EVALUATION")
    print("-" * 60)
    
    fairness_analyses = agent.evaluate_fairness(
        protected_attribute='gender',
        metrics=[FairnessMetric.DEMOGRAPHIC_PARITY]
    )
    
    print(agent.fairness_evaluator.get_fairness_report())
    
    # Audit trail
    print("\n\n5. AUDIT TRAIL")
    print("-" * 60)
    print(agent.audit_trail.get_summary())
    
    # Full transparency report
    print("\n\n6. COMPREHENSIVE TRANSPARENCY REPORT")
    print("-" * 60)
    print(agent.generate_transparency_report())
    
    # Statistics
    print("\n\n7. STATISTICS")
    print("-" * 60)
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("🎉 Pattern 125 Complete - 73.5% Milestone Reached!")
    print("Explainability & Transparency Category: 75% Complete!")
    print("125/170 patterns implemented!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_transparency()
