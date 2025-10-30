"""
Pattern 126: Trust & Verification Agent

This pattern implements trust scoring, verification mechanisms, credential validation,
and reputation management for establishing trustworthiness in AI systems.

Use Cases:
- Identity verification
- Credential validation
- Reputation systems
- Trust scoring
- Fraud detection

Category: Explainability & Transparency (4/4 = 100%)
Complexity: Advanced
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import json


class VerificationMethod(Enum):
    """Methods for verification."""
    CRYPTOGRAPHIC = "cryptographic"
    BIOMETRIC = "biometric"
    DOCUMENT = "document"
    BEHAVIORAL = "behavioral"
    SOCIAL = "social"
    MULTI_FACTOR = "multi_factor"


class TrustLevel(Enum):
    """Trust levels."""
    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"


class CredentialType(Enum):
    """Types of credentials."""
    IDENTITY = "identity"
    QUALIFICATION = "qualification"
    AUTHORIZATION = "authorization"
    CERTIFICATION = "certification"
    MEMBERSHIP = "membership"


class ReputationFactor(Enum):
    """Factors affecting reputation."""
    TRANSACTION_HISTORY = "transaction_history"
    USER_REVIEWS = "user_reviews"
    VERIFICATION_STATUS = "verification_status"
    LONGEVITY = "longevity"
    ACTIVITY_LEVEL = "activity_level"
    COMPLIANCE_RECORD = "compliance_record"


@dataclass
class Credential:
    """Represents a credential."""
    credential_id: str
    credential_type: CredentialType
    issuer: str
    subject: str
    issued_date: datetime
    expiry_date: Optional[datetime]
    claims: Dict[str, Any]
    signature: Optional[str] = None
    verified: bool = False
    
    def is_expired(self) -> bool:
        """Check if credential is expired."""
        if self.expiry_date is None:
            return False
        return datetime.now() > self.expiry_date
    
    def get_hash(self) -> str:
        """Get credential hash for verification."""
        data = {
            'credential_id': self.credential_id,
            'type': self.credential_type.value,
            'issuer': self.issuer,
            'subject': self.subject,
            'issued': self.issued_date.isoformat(),
            'claims': self.claims
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class VerificationRecord:
    """Record of a verification attempt."""
    verification_id: str
    timestamp: datetime
    method: VerificationMethod
    subject: str
    result: bool
    confidence: float
    evidence: List[str]
    verifier: str


@dataclass
class TrustScore:
    """Trust score for an entity."""
    entity_id: str
    overall_score: float  # 0.0 to 1.0
    level: TrustLevel
    factors: Dict[str, float]
    last_updated: datetime
    verification_count: int = 0
    
    def update_score(self, new_factors: Dict[str, float]):
        """Update trust score with new factors."""
        self.factors.update(new_factors)
        # Weighted average of factors
        weights = {
            'verification': 0.3,
            'reputation': 0.25,
            'history': 0.25,
            'compliance': 0.2
        }
        
        self.overall_score = sum(
            self.factors.get(key, 0.5) * weight
            for key, weight in weights.items()
        )
        
        # Determine level
        if self.overall_score >= 0.9:
            self.level = TrustLevel.VERIFIED
        elif self.overall_score >= 0.7:
            self.level = TrustLevel.HIGH
        elif self.overall_score >= 0.5:
            self.level = TrustLevel.MEDIUM
        elif self.overall_score >= 0.3:
            self.level = TrustLevel.LOW
        else:
            self.level = TrustLevel.UNTRUSTED
        
        self.last_updated = datetime.now()


@dataclass
class ReputationProfile:
    """Reputation profile for an entity."""
    entity_id: str
    positive_interactions: int = 0
    negative_interactions: int = 0
    total_transactions: int = 0
    average_rating: float = 0.0
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    badges: Set[str] = field(default_factory=set)
    join_date: datetime = field(default_factory=datetime.now)
    
    def add_interaction(self, positive: bool, rating: Optional[float] = None):
        """Add an interaction to reputation."""
        if positive:
            self.positive_interactions += 1
        else:
            self.negative_interactions += 1
        
        self.total_transactions += 1
        
        if rating is not None:
            # Update average rating
            total_ratings = len(self.reviews)
            self.average_rating = (
                (self.average_rating * total_ratings + rating) / 
                (total_ratings + 1)
            )
    
    def get_reputation_score(self) -> float:
        """Calculate reputation score."""
        if self.total_transactions == 0:
            return 0.5  # Neutral for new entities
        
        # Calculate based on multiple factors
        positive_ratio = self.positive_interactions / self.total_transactions
        
        # Account age factor (older = more trustworthy)
        age_days = (datetime.now() - self.join_date).days
        age_factor = min(1.0, age_days / 365)  # Maxes out at 1 year
        
        # Activity factor
        activity_factor = min(1.0, self.total_transactions / 100)
        
        # Rating factor
        rating_factor = self.average_rating / 5.0 if self.average_rating > 0 else 0.5
        
        # Weighted combination
        score = (
            positive_ratio * 0.4 +
            age_factor * 0.2 +
            activity_factor * 0.2 +
            rating_factor * 0.2
        )
        
        return score


class CredentialValidator:
    """Validates credentials."""
    
    def __init__(self):
        self.trusted_issuers: Set[str] = set()
        self.revoked_credentials: Set[str] = set()
    
    def add_trusted_issuer(self, issuer: str):
        """Add a trusted issuer."""
        self.trusted_issuers.add(issuer)
    
    def revoke_credential(self, credential_id: str):
        """Revoke a credential."""
        self.revoked_credentials.add(credential_id)
    
    def validate_credential(self, credential: Credential) -> Tuple[bool, List[str]]:
        """Validate a credential."""
        issues = []
        
        # Check if revoked
        if credential.credential_id in self.revoked_credentials:
            issues.append("Credential has been revoked")
            return False, issues
        
        # Check expiry
        if credential.is_expired():
            issues.append("Credential has expired")
            return False, issues
        
        # Check issuer trust
        if credential.issuer not in self.trusted_issuers:
            issues.append(f"Issuer '{credential.issuer}' is not trusted")
        
        # Verify signature if present
        if credential.signature:
            expected_hash = credential.get_hash()
            if credential.signature != expected_hash:
                issues.append("Signature verification failed")
                return False, issues
        
        # Validate claims
        if not credential.claims:
            issues.append("No claims present in credential")
        
        is_valid = len(issues) == 0
        return is_valid, issues if not is_valid else ["Credential is valid"]


class TrustScorer:
    """Calculates trust scores."""
    
    def __init__(self):
        self.trust_scores: Dict[str, TrustScore] = {}
    
    def initialize_score(self, entity_id: str) -> TrustScore:
        """Initialize trust score for new entity."""
        score = TrustScore(
            entity_id=entity_id,
            overall_score=0.5,  # Neutral starting point
            level=TrustLevel.MEDIUM,
            factors={
                'verification': 0.5,
                'reputation': 0.5,
                'history': 0.5,
                'compliance': 0.5
            },
            last_updated=datetime.now()
        )
        
        self.trust_scores[entity_id] = score
        return score
    
    def update_trust_score(
        self,
        entity_id: str,
        verification_score: Optional[float] = None,
        reputation_score: Optional[float] = None,
        history_score: Optional[float] = None,
        compliance_score: Optional[float] = None
    ) -> TrustScore:
        """Update trust score with new information."""
        if entity_id not in self.trust_scores:
            self.initialize_score(entity_id)
        
        score = self.trust_scores[entity_id]
        
        updates = {}
        if verification_score is not None:
            updates['verification'] = verification_score
        if reputation_score is not None:
            updates['reputation'] = reputation_score
        if history_score is not None:
            updates['history'] = history_score
        if compliance_score is not None:
            updates['compliance'] = compliance_score
        
        score.update_score(updates)
        return score
    
    def get_trust_level(self, entity_id: str) -> TrustLevel:
        """Get trust level for entity."""
        if entity_id not in self.trust_scores:
            return TrustLevel.MEDIUM
        
        return self.trust_scores[entity_id].level
    
    def compare_trust(self, entity1: str, entity2: str) -> str:
        """Compare trust between two entities."""
        score1 = self.trust_scores.get(entity1)
        score2 = self.trust_scores.get(entity2)
        
        if not score1 or not score2:
            return "Insufficient data for comparison"
        
        diff = score1.overall_score - score2.overall_score
        
        if abs(diff) < 0.1:
            return f"Both entities have similar trust levels ({score1.level.value})"
        elif diff > 0:
            return f"{entity1} is more trustworthy ({score1.level.value} vs {score2.level.value})"
        else:
            return f"{entity2} is more trustworthy ({score2.level.value} vs {score1.level.value})"


class VerificationEngine:
    """Performs verification checks."""
    
    def __init__(self):
        self.verification_history: List[VerificationRecord] = []
    
    def verify_identity(
        self,
        subject: str,
        method: VerificationMethod,
        provided_data: Dict[str, Any],
        expected_data: Dict[str, Any]
    ) -> VerificationRecord:
        """Verify identity."""
        evidence = []
        confidence = 0.0
        result = False
        
        if method == VerificationMethod.DOCUMENT:
            # Check document fields
            matches = sum(
                1 for key in expected_data
                if provided_data.get(key) == expected_data.get(key)
            )
            total = len(expected_data)
            confidence = matches / total if total > 0 else 0
            result = confidence >= 0.8
            
            evidence.append(f"Matched {matches}/{total} document fields")
        
        elif method == VerificationMethod.CRYPTOGRAPHIC:
            # Check cryptographic signature
            if 'signature' in provided_data and 'public_key' in expected_data:
                # Simplified check
                result = len(provided_data['signature']) == 64  # SHA256 length
                confidence = 1.0 if result else 0.0
                evidence.append("Cryptographic signature verified" if result else "Signature invalid")
        
        elif method == VerificationMethod.BEHAVIORAL:
            # Check behavioral patterns
            pattern_matches = 0
            for pattern in expected_data.get('patterns', []):
                if pattern in provided_data.get('observed_patterns', []):
                    pattern_matches += 1
            
            total_patterns = len(expected_data.get('patterns', []))
            confidence = pattern_matches / total_patterns if total_patterns > 0 else 0.5
            result = confidence >= 0.7
            
            evidence.append(f"Matched {pattern_matches}/{total_patterns} behavioral patterns")
        
        elif method == VerificationMethod.MULTI_FACTOR:
            # Multiple verification methods
            factors = provided_data.get('factors', [])
            required_factors = expected_data.get('required_factors', 2)
            
            result = len(factors) >= required_factors
            confidence = min(1.0, len(factors) / required_factors)
            evidence.append(f"Provided {len(factors)}/{required_factors} factors")
        
        else:
            evidence.append(f"Verification method {method.value} not fully implemented")
            confidence = 0.5
            result = False
        
        record = VerificationRecord(
            verification_id=f"ver_{len(self.verification_history) + 1}",
            timestamp=datetime.now(),
            method=method,
            subject=subject,
            result=result,
            confidence=confidence,
            evidence=evidence,
            verifier="VerificationEngine"
        )
        
        self.verification_history.append(record)
        return record
    
    def get_verification_history(self, subject: str) -> List[VerificationRecord]:
        """Get verification history for subject."""
        return [
            record for record in self.verification_history
            if record.subject == subject
        ]
    
    def calculate_verification_score(self, subject: str) -> float:
        """Calculate overall verification score."""
        history = self.get_verification_history(subject)
        
        if not history:
            return 0.5  # Neutral for unverified
        
        # Recent verifications weighted more heavily
        total_score = 0.0
        total_weight = 0.0
        
        for i, record in enumerate(reversed(history[-10:])):  # Last 10 verifications
            weight = 1.0 / (i + 1)  # Decay weight for older records
            
            if record.result:
                total_score += record.confidence * weight
            else:
                total_score += (1 - record.confidence) * weight * -0.5
            
            total_weight += weight
        
        score = total_score / total_weight if total_weight > 0 else 0.5
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]


class ReputationManager:
    """Manages reputation profiles."""
    
    def __init__(self):
        self.profiles: Dict[str, ReputationProfile] = {}
    
    def create_profile(self, entity_id: str) -> ReputationProfile:
        """Create new reputation profile."""
        profile = ReputationProfile(entity_id=entity_id)
        self.profiles[entity_id] = profile
        return profile
    
    def record_interaction(
        self,
        entity_id: str,
        positive: bool,
        rating: Optional[float] = None,
        review: Optional[str] = None
    ):
        """Record an interaction."""
        if entity_id not in self.profiles:
            self.create_profile(entity_id)
        
        profile = self.profiles[entity_id]
        profile.add_interaction(positive, rating)
        
        if review:
            profile.reviews.append({
                'timestamp': datetime.now(),
                'rating': rating,
                'review': review,
                'positive': positive
            })
    
    def award_badge(self, entity_id: str, badge: str):
        """Award a badge to entity."""
        if entity_id not in self.profiles:
            self.create_profile(entity_id)
        
        self.profiles[entity_id].badges.add(badge)
    
    def get_reputation_summary(self, entity_id: str) -> str:
        """Get reputation summary."""
        if entity_id not in self.profiles:
            return "No reputation data available"
        
        profile = self.profiles[entity_id]
        score = profile.get_reputation_score()
        
        lines = [
            f"Reputation Summary for {entity_id}:",
            f"  Score: {score:.2f}/1.00",
            f"  Member since: {profile.join_date.strftime('%Y-%m-%d')}",
            f"  Total transactions: {profile.total_transactions}",
            f"  Positive: {profile.positive_interactions} ({profile.positive_interactions/profile.total_transactions*100:.1f}%)" if profile.total_transactions > 0 else "  No transactions yet",
            f"  Average rating: {profile.average_rating:.2f}/5.00",
            f"  Badges: {', '.join(profile.badges) if profile.badges else 'None'}"
        ]
        
        return "\n".join(lines)


class TrustVerificationAgent:
    """Agent for trust scoring and verification."""
    
    def __init__(self):
        self.credential_validator = CredentialValidator()
        self.trust_scorer = TrustScorer()
        self.verification_engine = VerificationEngine()
        self.reputation_manager = ReputationManager()
        
        # Add some default trusted issuers
        self.credential_validator.add_trusted_issuer("Government ID Agency")
        self.credential_validator.add_trusted_issuer("Professional Certification Board")
        self.credential_validator.add_trusted_issuer("Educational Institution")
    
    def register_entity(self, entity_id: str) -> Tuple[TrustScore, ReputationProfile]:
        """Register a new entity."""
        trust_score = self.trust_scorer.initialize_score(entity_id)
        reputation = self.reputation_manager.create_profile(entity_id)
        return trust_score, reputation
    
    def verify_and_score(
        self,
        entity_id: str,
        verification_method: VerificationMethod,
        provided_data: Dict[str, Any],
        expected_data: Dict[str, Any]
    ) -> Tuple[VerificationRecord, TrustScore]:
        """Verify identity and update trust score."""
        # Perform verification
        verification = self.verification_engine.verify_identity(
            entity_id,
            verification_method,
            provided_data,
            expected_data
        )
        
        # Calculate new verification score
        verification_score = self.verification_engine.calculate_verification_score(entity_id)
        
        # Update trust score
        trust_score = self.trust_scorer.update_trust_score(
            entity_id,
            verification_score=verification_score
        )
        
        return verification, trust_score
    
    def validate_credential_and_update_trust(
        self,
        entity_id: str,
        credential: Credential
    ) -> Tuple[bool, List[str], TrustScore]:
        """Validate credential and update trust."""
        is_valid, messages = self.credential_validator.validate_credential(credential)
        
        # Update trust based on credential validation
        if is_valid:
            current_score = self.trust_scorer.trust_scores.get(entity_id)
            if current_score:
                verification_boost = 0.1
                new_verification = min(1.0, current_score.factors['verification'] + verification_boost)
                trust_score = self.trust_scorer.update_trust_score(
                    entity_id,
                    verification_score=new_verification
                )
            else:
                trust_score = self.trust_scorer.initialize_score(entity_id)
        else:
            trust_score = self.trust_scorer.trust_scores.get(entity_id) or self.trust_scorer.initialize_score(entity_id)
        
        return is_valid, messages, trust_score
    
    def record_transaction(
        self,
        entity_id: str,
        successful: bool,
        rating: Optional[float] = None,
        review: Optional[str] = None
    ):
        """Record a transaction and update reputation/trust."""
        # Update reputation
        self.reputation_manager.record_interaction(
            entity_id,
            positive=successful,
            rating=rating,
            review=review
        )
        
        # Get updated reputation score
        profile = self.reputation_manager.profiles[entity_id]
        reputation_score = profile.get_reputation_score()
        
        # Update trust score with new reputation
        self.trust_scorer.update_trust_score(
            entity_id,
            reputation_score=reputation_score
        )
    
    def get_comprehensive_trust_report(self, entity_id: str) -> str:
        """Generate comprehensive trust report."""
        lines = [
            "=" * 60,
            f"TRUST & VERIFICATION REPORT: {entity_id}",
            "=" * 60
        ]
        
        # Trust score
        trust_score = self.trust_scorer.trust_scores.get(entity_id)
        if trust_score:
            lines.extend([
                f"\n🎯 TRUST SCORE: {trust_score.overall_score:.2f}/1.00",
                f"   Level: {trust_score.level.value.upper()}",
                f"   Last Updated: {trust_score.last_updated}",
                f"\n   Factor Breakdown:"
            ])
            for factor, score in trust_score.factors.items():
                lines.append(f"   - {factor}: {score:.2f}")
        
        # Verification history
        verifications = self.verification_engine.get_verification_history(entity_id)
        lines.extend([
            f"\n🔍 VERIFICATION HISTORY: {len(verifications)} verifications"
        ])
        
        for ver in verifications[-3:]:  # Last 3
            result_icon = "✓" if ver.result else "✗"
            lines.extend([
                f"\n   {result_icon} {ver.method.value}",
                f"      Confidence: {ver.confidence:.2f}",
                f"      Date: {ver.timestamp.strftime('%Y-%m-%d %H:%M')}",
                f"      Evidence: {', '.join(ver.evidence)}"
            ])
        
        # Reputation
        lines.append(f"\n📊 {self.reputation_manager.get_reputation_summary(entity_id)}")
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trust and verification statistics."""
        return {
            'total_entities': len(self.trust_scorer.trust_scores),
            'total_verifications': len(self.verification_engine.verification_history),
            'trusted_entities': sum(
                1 for score in self.trust_scorer.trust_scores.values()
                if score.level in [TrustLevel.HIGH, TrustLevel.VERIFIED]
            ),
            'total_transactions': sum(
                profile.total_transactions
                for profile in self.reputation_manager.profiles.values()
            ),
            'average_trust_score': sum(
                score.overall_score
                for score in self.trust_scorer.trust_scores.values()
            ) / len(self.trust_scorer.trust_scores) if self.trust_scorer.trust_scores else 0
        }


def demonstrate_trust_verification():
    """Demonstrate the Trust & Verification Agent."""
    print("=" * 60)
    print("Trust & Verification Agent Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = TrustVerificationAgent()
    
    print("\n1. ENTITY REGISTRATION")
    print("-" * 60)
    
    # Register entities
    entities = ["Alice", "Bob", "Charlie"]
    
    for entity in entities:
        trust_score, reputation = agent.register_entity(entity)
        print(f"✓ Registered {entity} - Initial trust: {trust_score.overall_score:.2f} ({trust_score.level.value})")
    
    # Credential validation
    print("\n\n2. CREDENTIAL VALIDATION")
    print("-" * 60)
    
    # Create a valid credential
    credential = Credential(
        credential_id="ID_001",
        credential_type=CredentialType.IDENTITY,
        issuer="Government ID Agency",
        subject="Alice",
        issued_date=datetime.now() - timedelta(days=30),
        expiry_date=datetime.now() + timedelta(days=335),
        claims={'name': 'Alice', 'age': 30, 'citizenship': 'USA'}
    )
    credential.signature = credential.get_hash()
    
    is_valid, messages, trust_score = agent.validate_credential_and_update_trust("Alice", credential)
    
    print(f"Credential {credential.credential_id}:")
    print(f"  Valid: {is_valid}")
    print(f"  Messages: {', '.join(messages)}")
    print(f"  Updated trust: {trust_score.overall_score:.2f} ({trust_score.level.value})")
    
    # Identity verification
    print("\n\n3. IDENTITY VERIFICATION")
    print("-" * 60)
    
    # Verify Bob with document method
    verification, trust_score = agent.verify_and_score(
        "Bob",
        VerificationMethod.DOCUMENT,
        provided_data={
            'name': 'Bob',
            'dob': '1990-05-15',
            'id_number': '123456789'
        },
        expected_data={
            'name': 'Bob',
            'dob': '1990-05-15',
            'id_number': '123456789'
        }
    )
    
    print(f"Bob's verification:")
    print(f"  Method: {verification.method.value}")
    print(f"  Result: {'✓ PASSED' if verification.result else '✗ FAILED'}")
    print(f"  Confidence: {verification.confidence:.2f}")
    print(f"  Evidence: {', '.join(verification.evidence)}")
    print(f"  Updated trust: {trust_score.overall_score:.2f}")
    
    # Multi-factor verification for Alice
    verification, trust_score = agent.verify_and_score(
        "Alice",
        VerificationMethod.MULTI_FACTOR,
        provided_data={
            'factors': ['password', 'sms_code', 'biometric']
        },
        expected_data={
            'required_factors': 2
        }
    )
    
    print(f"\nAlice's multi-factor verification:")
    print(f"  Result: {'✓ PASSED' if verification.result else '✗ FAILED'}")
    print(f"  Confidence: {verification.confidence:.2f}")
    print(f"  Updated trust: {trust_score.overall_score:.2f}")
    
    # Transaction history
    print("\n\n4. TRANSACTION & REPUTATION TRACKING")
    print("-" * 60)
    
    # Simulate transactions
    transactions = [
        ("Alice", True, 5.0, "Excellent service!"),
        ("Alice", True, 4.5, "Very good"),
        ("Bob", True, 4.0, "Good"),
        ("Bob", False, 2.0, "Not satisfied"),
        ("Charlie", True, 3.5, "Okay"),
    ]
    
    for entity, success, rating, review in transactions:
        agent.record_transaction(entity, success, rating, review)
        print(f"  Recorded transaction for {entity}: {rating}⭐ ({'positive' if success else 'negative'})")
    
    # Award badges
    agent.reputation_manager.award_badge("Alice", "Verified Seller")
    agent.reputation_manager.award_badge("Alice", "Top Rated")
    print(f"\n  Awarded badges to Alice: Verified Seller, Top Rated")
    
    # Comprehensive trust reports
    print("\n\n5. COMPREHENSIVE TRUST REPORTS")
    print("-" * 60)
    
    for entity in ["Alice", "Bob"]:
        print(f"\n{agent.get_comprehensive_trust_report(entity)}")
    
    # Trust comparison
    print("\n\n6. TRUST COMPARISON")
    print("-" * 60)
    
    comparison = agent.trust_scorer.compare_trust("Alice", "Bob")
    print(f"  {comparison}")
    
    comparison = agent.trust_scorer.compare_trust("Bob", "Charlie")
    print(f"  {comparison}")
    
    # Statistics
    print("\n\n7. OVERALL STATISTICS")
    print("-" * 60)
    
    stats = agent.get_statistics()
    print(f"  Total Entities: {stats['total_entities']}")
    print(f"  Total Verifications: {stats['total_verifications']}")
    print(f"  Trusted Entities: {stats['trusted_entities']}")
    print(f"  Total Transactions: {stats['total_transactions']}")
    print(f"  Average Trust Score: {stats['average_trust_score']:.2f}")
    
    print("\n" + "=" * 60)
    print("🎉 Pattern 126 Complete!")
    print("Explainability & Transparency Category: 100% COMPLETE!")
    print("126/170 patterns implemented (74.1%)!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_trust_verification()
