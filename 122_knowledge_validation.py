"""
Pattern 122: Knowledge Validation Agent

This pattern implements logical consistency checking, fact verification,
source credibility assessment, and evidence weighing.

Use Cases:
- Fact-checking systems
- Knowledge base validation
- Misinformation detection
- Data quality assurance
- Scientific claim verification

Category: Knowledge Management (4/4 = 100%) - COMPLETES CATEGORY!
Complexity: Advanced
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import re


class ValidationType(Enum):
    """Types of validation checks."""
    LOGICAL_CONSISTENCY = "logical_consistency"
    FACTUAL_ACCURACY = "factual_accuracy"
    SOURCE_CREDIBILITY = "source_credibility"
    EVIDENCE_QUALITY = "evidence_quality"
    TEMPORAL_VALIDITY = "temporal_validity"
    SEMANTIC_COHERENCE = "semantic_coherence"


class ValidationStatus(Enum):
    """Validation result status."""
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    NEEDS_REVIEW = "needs_review"


class EvidenceType(Enum):
    """Types of supporting evidence."""
    EMPIRICAL = "empirical"  # Data, experiments
    TESTIMONIAL = "testimonial"  # Expert opinion
    AUTHORITATIVE = "authoritative"  # Published sources
    ANECDOTAL = "anecdotal"  # Personal accounts
    STATISTICAL = "statistical"  # Statistical analysis
    LOGICAL = "logical"  # Logical reasoning


@dataclass
class Evidence:
    """Supporting evidence for a claim."""
    evidence_type: EvidenceType
    content: str
    source: str
    strength: float = 0.5  # 0.0 to 1.0
    reliability: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_weight(self) -> float:
        """Calculate evidence weight."""
        type_weights = {
            EvidenceType.EMPIRICAL: 1.0,
            EvidenceType.STATISTICAL: 0.9,
            EvidenceType.AUTHORITATIVE: 0.8,
            EvidenceType.LOGICAL: 0.7,
            EvidenceType.TESTIMONIAL: 0.6,
            EvidenceType.ANECDOTAL: 0.3,
        }
        
        type_weight = type_weights.get(self.evidence_type, 0.5)
        return type_weight * self.strength * self.reliability


@dataclass
class Claim:
    """Knowledge claim to be validated."""
    claim_id: str
    statement: str
    subject: str
    predicate: str
    object: Any
    source: str
    confidence: float = 0.5
    evidence: List[Evidence] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validation check."""
    claim: Claim
    validation_type: ValidationType
    status: ValidationStatus
    confidence: float
    reasons: List[str]
    contradictions: List[str] = field(default_factory=list)
    supporting_evidence: List[Evidence] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LogicalRule:
    """Logical consistency rule."""
    rule_id: str
    name: str
    condition: str  # Logical expression
    consequence: str
    priority: int = 1


class LogicalConsistencyChecker:
    """Checks logical consistency of claims."""
    
    def __init__(self):
        self.rules: List[LogicalRule] = []
        self.facts: Dict[str, Any] = {}
        self.contradictions: List[Tuple[Claim, Claim]] = []
    
    def add_rule(self, rule: LogicalRule) -> None:
        """Add a logical rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def add_fact(self, key: str, value: Any) -> None:
        """Add a known fact."""
        self.facts[key] = value
    
    def check_claim(self, claim: Claim) -> ValidationResult:
        """Check logical consistency of a claim."""
        reasons = []
        contradictions = []
        
        # Check against known facts
        fact_key = f"{claim.subject}:{claim.predicate}"
        if fact_key in self.facts:
            known_value = self.facts[fact_key]
            if known_value != claim.object:
                contradictions.append(
                    f"Contradicts known fact: {claim.subject}.{claim.predicate} "
                    f"= {known_value} (claimed: {claim.object})"
                )
        
        # Check logical rules
        for rule in self.rules:
            if self._evaluate_condition(rule.condition, claim):
                if not self._evaluate_condition(rule.consequence, claim):
                    contradictions.append(
                        f"Violates rule '{rule.name}': {rule.condition} → {rule.consequence}"
                    )
        
        # Check for self-contradiction
        if self._is_self_contradictory(claim.statement):
            contradictions.append("Statement is self-contradictory")
        
        # Determine status
        if contradictions:
            status = ValidationStatus.INVALID
            confidence = 0.2
            reasons.append(f"Found {len(contradictions)} logical inconsistencies")
        else:
            status = ValidationStatus.VALID
            confidence = 0.8
            reasons.append("No logical inconsistencies detected")
        
        return ValidationResult(
            claim=claim,
            validation_type=ValidationType.LOGICAL_CONSISTENCY,
            status=status,
            confidence=confidence,
            reasons=reasons,
            contradictions=contradictions
        )
    
    def _evaluate_condition(self, condition: str, claim: Claim) -> bool:
        """Evaluate a logical condition."""
        # Simple pattern matching for demonstration
        # In production, use proper logical expression parser
        
        # Check if condition mentions the claim's predicate
        if claim.predicate in condition:
            # Simple equality check
            if f"{claim.predicate} = " in condition:
                expected = condition.split("=")[1].strip()
                return str(claim.object) == expected
            # Simple comparison
            elif f"{claim.predicate} > " in condition:
                expected = float(condition.split(">")[1].strip())
                return float(claim.object) > expected
            elif f"{claim.predicate} < " in condition:
                expected = float(condition.split("<")[1].strip())
                return float(claim.object) < expected
        
        return True  # Default to true if can't evaluate
    
    def _is_self_contradictory(self, statement: str) -> bool:
        """Check if statement is self-contradictory."""
        statement_lower = statement.lower()
        
        # Check for obvious contradictions
        patterns = [
            (r'\bnot\b.*\band\b', r'\bis\b'),
            (r'\balways\b.*\bnever\b', ''),
            (r'\ball\b.*\bnone\b', ''),
        ]
        
        for pattern1, pattern2 in patterns:
            if re.search(pattern1, statement_lower):
                if not pattern2 or re.search(pattern2, statement_lower):
                    return True
        
        return False


class FactualAccuracyChecker:
    """Checks factual accuracy of claims."""
    
    def __init__(self):
        self.knowledge_base: Dict[str, Any] = {}
        self.verified_claims: Set[str] = set()
        self.false_claims: Set[str] = set()
    
    def add_verified_fact(self, subject: str, predicate: str, value: Any) -> None:
        """Add a verified fact to knowledge base."""
        key = f"{subject}:{predicate}"
        self.knowledge_base[key] = value
        self.verified_claims.add(key)
    
    def add_false_claim(self, subject: str, predicate: str) -> None:
        """Mark a claim as false."""
        key = f"{subject}:{predicate}"
        self.false_claims.add(key)
    
    def check_claim(self, claim: Claim) -> ValidationResult:
        """Check factual accuracy of a claim."""
        reasons = []
        contradictions = []
        
        key = f"{claim.subject}:{claim.predicate}"
        
        # Check if known to be false
        if key in self.false_claims:
            status = ValidationStatus.INVALID
            confidence = 0.9
            reasons.append("Claim is known to be false")
            contradictions.append(f"Contradicts verified information")
            
        # Check against verified facts
        elif key in self.verified_claims:
            verified_value = self.knowledge_base[key]
            if self._values_match(verified_value, claim.object):
                status = ValidationStatus.VALID
                confidence = 0.95
                reasons.append("Matches verified fact in knowledge base")
            else:
                status = ValidationStatus.INVALID
                confidence = 0.85
                reasons.append(f"Contradicts verified fact (expected: {verified_value})")
                contradictions.append(
                    f"Known value is {verified_value}, not {claim.object}"
                )
        
        # Unknown claim - check evidence
        else:
            evidence_score = self._evaluate_evidence(claim.evidence)
            
            if evidence_score > 0.7:
                status = ValidationStatus.VALID
                confidence = evidence_score
                reasons.append(f"Strong supporting evidence (score: {evidence_score:.2f})")
            elif evidence_score > 0.4:
                status = ValidationStatus.UNCERTAIN
                confidence = evidence_score
                reasons.append(f"Moderate evidence (score: {evidence_score:.2f})")
            else:
                status = ValidationStatus.NEEDS_REVIEW
                confidence = evidence_score
                reasons.append(f"Insufficient evidence (score: {evidence_score:.2f})")
        
        return ValidationResult(
            claim=claim,
            validation_type=ValidationType.FACTUAL_ACCURACY,
            status=status,
            confidence=confidence,
            reasons=reasons,
            contradictions=contradictions,
            supporting_evidence=claim.evidence
        )
    
    def _values_match(self, value1: Any, value2: Any, tolerance: float = 0.01) -> bool:
        """Check if two values match with tolerance."""
        # Numeric comparison
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            return abs(float(value1) - float(value2)) <= tolerance
        
        # String comparison
        return str(value1).lower() == str(value2).lower()
    
    def _evaluate_evidence(self, evidence: List[Evidence]) -> float:
        """Evaluate quality of supporting evidence."""
        if not evidence:
            return 0.0
        
        # Weight evidence by type and quality
        total_weight = sum(e.get_weight() for e in evidence)
        max_possible = len(evidence) * 1.0  # Max weight per evidence
        
        return min(1.0, total_weight / max_possible)


class SourceCredibilityChecker:
    """Assesses source credibility."""
    
    def __init__(self):
        self.source_ratings: Dict[str, float] = {}
        self.source_history: Dict[str, List[bool]] = {}
        
        # Default ratings for source types
        self.default_ratings = {
            'academic': 0.95,
            'government': 0.90,
            'news_major': 0.75,
            'news_minor': 0.60,
            'blog': 0.50,
            'social_media': 0.30,
            'anonymous': 0.10,
        }
    
    def add_source_rating(self, source: str, rating: float) -> None:
        """Add or update source rating."""
        self.source_ratings[source] = rating
    
    def update_source_history(self, source: str, was_accurate: bool) -> None:
        """Update source accuracy history."""
        if source not in self.source_history:
            self.source_history[source] = []
        
        self.source_history[source].append(was_accurate)
        
        # Update rating based on history
        history = self.source_history[source]
        accuracy = sum(history) / len(history)
        self.source_ratings[source] = accuracy
    
    def check_claim(self, claim: Claim) -> ValidationResult:
        """Check source credibility for a claim."""
        reasons = []
        
        # Get source rating
        source_rating = self._get_source_rating(claim.source)
        
        # Check source history
        if claim.source in self.source_history:
            history = self.source_history[claim.source]
            accuracy = sum(history) / len(history)
            reasons.append(
                f"Source historical accuracy: {accuracy:.1%} "
                f"({len(history)} previous claims)"
            )
        
        # Determine status based on rating
        if source_rating > 0.8:
            status = ValidationStatus.VALID
            confidence = source_rating
            reasons.append(f"High credibility source (rating: {source_rating:.2f})")
        elif source_rating > 0.5:
            status = ValidationStatus.UNCERTAIN
            confidence = source_rating
            reasons.append(f"Medium credibility source (rating: {source_rating:.2f})")
        else:
            status = ValidationStatus.NEEDS_REVIEW
            confidence = source_rating
            reasons.append(f"Low credibility source (rating: {source_rating:.2f})")
        
        return ValidationResult(
            claim=claim,
            validation_type=ValidationType.SOURCE_CREDIBILITY,
            status=status,
            confidence=confidence,
            reasons=reasons
        )
    
    def _get_source_rating(self, source: str) -> float:
        """Get rating for a source."""
        # Check explicit rating
        if source in self.source_ratings:
            return self.source_ratings[source]
        
        # Check default ratings by type
        source_lower = source.lower()
        for source_type, rating in self.default_ratings.items():
            if source_type in source_lower:
                return rating
        
        # Default to medium-low
        return 0.4


class EvidenceQualityChecker:
    """Assesses quality of supporting evidence."""
    
    def check_claim(self, claim: Claim) -> ValidationResult:
        """Check evidence quality for a claim."""
        reasons = []
        
        if not claim.evidence:
            return ValidationResult(
                claim=claim,
                validation_type=ValidationType.EVIDENCE_QUALITY,
                status=ValidationStatus.NEEDS_REVIEW,
                confidence=0.1,
                reasons=["No supporting evidence provided"]
            )
        
        # Evaluate each piece of evidence
        evidence_scores = []
        evidence_types = set()
        
        for evidence in claim.evidence:
            score = evidence.get_weight()
            evidence_scores.append(score)
            evidence_types.add(evidence.evidence_type)
            
            reasons.append(
                f"{evidence.evidence_type.value}: strength={evidence.strength:.2f}, "
                f"reliability={evidence.reliability:.2f}, weight={score:.2f}"
            )
        
        # Calculate overall quality
        avg_score = sum(evidence_scores) / len(evidence_scores)
        
        # Bonus for diverse evidence types
        diversity_bonus = min(0.2, len(evidence_types) * 0.05)
        final_score = min(1.0, avg_score + diversity_bonus)
        
        # Determine status
        if final_score > 0.7:
            status = ValidationStatus.VALID
            reasons.append(f"High quality evidence (score: {final_score:.2f})")
        elif final_score > 0.4:
            status = ValidationStatus.UNCERTAIN
            reasons.append(f"Moderate quality evidence (score: {final_score:.2f})")
        else:
            status = ValidationStatus.NEEDS_REVIEW
            reasons.append(f"Low quality evidence (score: {final_score:.2f})")
        
        if len(evidence_types) > 1:
            reasons.append(f"Diverse evidence types: {len(evidence_types)}")
        
        return ValidationResult(
            claim=claim,
            validation_type=ValidationType.EVIDENCE_QUALITY,
            status=status,
            confidence=final_score,
            reasons=reasons,
            supporting_evidence=claim.evidence
        )


class KnowledgeValidationAgent:
    """Agent for validating knowledge claims."""
    
    def __init__(self):
        self.logical_checker = LogicalConsistencyChecker()
        self.factual_checker = FactualAccuracyChecker()
        self.source_checker = SourceCredibilityChecker()
        self.evidence_checker = EvidenceQualityChecker()
        
        # Validation history
        self.validation_history: List[ValidationResult] = []
    
    def add_logical_rule(self, rule: LogicalRule) -> None:
        """Add a logical rule for consistency checking."""
        self.logical_checker.add_rule(rule)
    
    def add_verified_fact(self, subject: str, predicate: str, value: Any) -> None:
        """Add a verified fact to knowledge base."""
        self.factual_checker.add_verified_fact(subject, predicate, value)
        self.logical_checker.add_fact(f"{subject}:{predicate}", value)
    
    def add_source_rating(self, source: str, rating: float) -> None:
        """Add source credibility rating."""
        self.source_checker.add_source_rating(source, rating)
    
    def validate_claim(
        self,
        claim: Claim,
        checks: Optional[List[ValidationType]] = None
    ) -> Dict[ValidationType, ValidationResult]:
        """Validate a claim using multiple checks."""
        if checks is None:
            checks = [
                ValidationType.LOGICAL_CONSISTENCY,
                ValidationType.FACTUAL_ACCURACY,
                ValidationType.SOURCE_CREDIBILITY,
                ValidationType.EVIDENCE_QUALITY,
            ]
        
        results = {}
        
        for check_type in checks:
            if check_type == ValidationType.LOGICAL_CONSISTENCY:
                result = self.logical_checker.check_claim(claim)
            elif check_type == ValidationType.FACTUAL_ACCURACY:
                result = self.factual_checker.check_claim(claim)
            elif check_type == ValidationType.SOURCE_CREDIBILITY:
                result = self.source_checker.check_claim(claim)
            elif check_type == ValidationType.EVIDENCE_QUALITY:
                result = self.evidence_checker.check_claim(claim)
            else:
                continue
            
            results[check_type] = result
            self.validation_history.append(result)
        
        return results
    
    def get_overall_assessment(
        self,
        results: Dict[ValidationType, ValidationResult]
    ) -> Tuple[ValidationStatus, float, List[str]]:
        """Get overall assessment from multiple validation results."""
        if not results:
            return ValidationStatus.NEEDS_REVIEW, 0.0, ["No validation performed"]
        
        # Weight different checks
        weights = {
            ValidationType.LOGICAL_CONSISTENCY: 0.3,
            ValidationType.FACTUAL_ACCURACY: 0.35,
            ValidationType.SOURCE_CREDIBILITY: 0.20,
            ValidationType.EVIDENCE_QUALITY: 0.15,
        }
        
        # Calculate weighted confidence
        total_weight = 0.0
        weighted_confidence = 0.0
        
        for check_type, result in results.items():
            weight = weights.get(check_type, 0.25)
            total_weight += weight
            weighted_confidence += result.confidence * weight
        
        overall_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
        
        # Determine overall status
        invalid_count = sum(
            1 for r in results.values()
            if r.status == ValidationStatus.INVALID
        )
        
        if invalid_count > 0:
            status = ValidationStatus.INVALID
        elif overall_confidence > 0.7:
            status = ValidationStatus.VALID
        elif overall_confidence > 0.4:
            status = ValidationStatus.UNCERTAIN
        else:
            status = ValidationStatus.NEEDS_REVIEW
        
        # Compile reasons
        reasons = []
        for check_type, result in results.items():
            reasons.append(f"{check_type.value}: {result.status.value} ({result.confidence:.2f})")
        
        return status, overall_confidence, reasons
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self.validation_history:
            return {}
        
        status_counts = {}
        type_counts = {}
        
        for result in self.validation_history:
            # Count by status
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by type
            vtype = result.validation_type.value
            type_counts[vtype] = type_counts.get(vtype, 0) + 1
        
        avg_confidence = sum(
            r.confidence for r in self.validation_history
        ) / len(self.validation_history)
        
        return {
            'total_validations': len(self.validation_history),
            'status_distribution': status_counts,
            'validation_types': type_counts,
            'average_confidence': avg_confidence,
        }


def demonstrate_knowledge_validation():
    """Demonstrate the Knowledge Validation Agent."""
    print("=" * 60)
    print("Knowledge Validation Agent Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = KnowledgeValidationAgent()
    
    # Set up knowledge base
    print("\n1. SETTING UP KNOWLEDGE BASE")
    print("-" * 60)
    
    # Add verified facts
    agent.add_verified_fact("Earth", "radius_km", 6371)
    agent.add_verified_fact("Earth", "age_billion_years", 4.5)
    agent.add_verified_fact("Speed_of_Light", "km_per_sec", 299792)
    print("   Added verified facts to knowledge base")
    
    # Add source ratings
    agent.add_source_rating("Nature Journal", 0.95)
    agent.add_source_rating("Wikipedia", 0.75)
    agent.add_source_rating("Random Blog", 0.30)
    print("   Added source credibility ratings")
    
    # Add logical rules
    agent.add_logical_rule(LogicalRule(
        "rule1",
        "Positive values",
        "radius_km > 0",
        "radius_km > 0",
        priority=1
    ))
    print("   Added logical consistency rules")
    
    # Create test claims
    print("\n2. TESTING CLAIMS")
    print("-" * 60)
    
    # Claim 1: True claim with good evidence
    claim1 = Claim(
        claim_id="c1",
        statement="Earth's radius is approximately 6371 km",
        subject="Earth",
        predicate="radius_km",
        object=6371,
        source="Nature Journal",
        confidence=0.95,
        evidence=[
            Evidence(
                EvidenceType.EMPIRICAL,
                "Satellite measurements",
                "NASA",
                strength=0.95,
                reliability=0.98
            ),
            Evidence(
                EvidenceType.AUTHORITATIVE,
                "Published in peer-reviewed journal",
                "Nature",
                strength=0.90,
                reliability=0.95
            ),
        ]
    )
    
    print("\n   Claim 1: Earth's radius = 6371 km")
    results1 = agent.validate_claim(claim1)
    
    for check_type, result in results1.items():
        print(f"\n   {check_type.value}:")
        print(f"   Status: {result.status.value}")
        print(f"   Confidence: {result.confidence:.2f}")
        for reason in result.reasons:
            print(f"   - {reason}")
    
    status, confidence, reasons = agent.get_overall_assessment(results1)
    print(f"\n   OVERALL: {status.value} (confidence: {confidence:.2f})")
    
    # Claim 2: False claim
    claim2 = Claim(
        claim_id="c2",
        statement="Earth's radius is 10000 km",
        subject="Earth",
        predicate="radius_km",
        object=10000,
        source="Random Blog",
        confidence=0.40,
        evidence=[
            Evidence(
                EvidenceType.ANECDOTAL,
                "Someone told me",
                "Anonymous",
                strength=0.3,
                reliability=0.2
            ),
        ]
    )
    
    print("\n\n   Claim 2: Earth's radius = 10000 km (INCORRECT)")
    results2 = agent.validate_claim(claim2)
    
    for check_type, result in results2.items():
        print(f"\n   {check_type.value}:")
        print(f"   Status: {result.status.value}")
        print(f"   Confidence: {result.confidence:.2f}")
        for reason in result.reasons:
            print(f"   - {reason}")
        if result.contradictions:
            print("   Contradictions:")
            for contra in result.contradictions:
                print(f"   - {contra}")
    
    status, confidence, reasons = agent.get_overall_assessment(results2)
    print(f"\n   OVERALL: {status.value} (confidence: {confidence:.2f})")
    
    # Claim 3: Unknown claim with moderate evidence
    claim3 = Claim(
        claim_id="c3",
        statement="Average ocean depth is 3688 meters",
        subject="Ocean",
        predicate="avg_depth_m",
        object=3688,
        source="Wikipedia",
        confidence=0.70,
        evidence=[
            Evidence(
                EvidenceType.STATISTICAL,
                "Calculated from bathymetric data",
                "NOAA",
                strength=0.80,
                reliability=0.85
            ),
            Evidence(
                EvidenceType.AUTHORITATIVE,
                "Referenced in oceanography textbook",
                "Academic Press",
                strength=0.75,
                reliability=0.80
            ),
        ]
    )
    
    print("\n\n   Claim 3: Ocean average depth = 3688 m (UNKNOWN)")
    results3 = agent.validate_claim(claim3)
    
    for check_type, result in results3.items():
        print(f"\n   {check_type.value}:")
        print(f"   Status: {result.status.value}")
        print(f"   Confidence: {result.confidence:.2f}")
        for reason in result.reasons:
            print(f"   - {reason}")
    
    status, confidence, reasons = agent.get_overall_assessment(results3)
    print(f"\n   OVERALL: {status.value} (confidence: {confidence:.2f})")
    
    # Statistics
    print("\n\n3. VALIDATION STATISTICS")
    print("-" * 60)
    
    stats = agent.get_statistics()
    print(f"   Total Validations: {stats['total_validations']}")
    print(f"   Average Confidence: {stats['average_confidence']:.2f}")
    print(f"\n   Status Distribution:")
    for status, count in stats['status_distribution'].items():
        print(f"   - {status}: {count}")
    print(f"\n   Validation Types:")
    for vtype, count in stats['validation_types'].items():
        print(f"   - {vtype}: {count}")
    
    print("\n" + "=" * 60)
    print("🎉 Knowledge Management Category: 100% COMPLETE!")
    print("Pattern 122 COMPLETES the Knowledge Management category!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_knowledge_validation()
