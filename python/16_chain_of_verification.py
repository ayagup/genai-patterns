"""
Chain-of-Verification (CoVe) Pattern Implementation

This module demonstrates the Chain-of-Verification pattern where an agent
generates responses and then systematically verifies the accuracy of claims
through multiple verification steps and fact-checking procedures.

Key Components:
- Claim extraction and identification
- Multi-step verification process
- Evidence gathering and validation
- Confidence scoring and uncertainty quantification
- Revision based on verification results
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set, Callable
from enum import Enum
import random
import re
from datetime import datetime


class ClaimType(Enum):
    """Types of claims that can be verified"""
    FACTUAL = "factual"                 # Objective facts
    STATISTICAL = "statistical"         # Numbers, percentages, data
    HISTORICAL = "historical"           # Historical events and dates
    SCIENTIFIC = "scientific"           # Scientific concepts and findings
    GEOGRAPHICAL = "geographical"       # Location-based information
    BIOGRAPHICAL = "biographical"       # Information about people
    CAUSAL = "causal"                  # Cause-and-effect relationships
    COMPARATIVE = "comparative"         # Comparisons between entities
    DEFINITIONAL = "definitional"      # Definitions and explanations
    PREDICTIVE = "predictive"          # Future predictions or trends


class VerificationMethod(Enum):
    """Methods for verifying claims"""
    SOURCE_CHECK = "source_check"           # Check authoritative sources
    CROSS_REFERENCE = "cross_reference"     # Compare multiple sources
    LOGICAL_ANALYSIS = "logical_analysis"   # Check logical consistency
    QUANTITATIVE = "quantitative"          # Verify numbers and calculations
    TEMPORAL = "temporal"                  # Check dates and sequences
    EXPERT_CONSENSUS = "expert_consensus"   # Check scientific consensus
    PRIMARY_SOURCE = "primary_source"      # Verify with primary sources
    REPRODUCIBILITY = "reproducibility"    # Check if results can be reproduced


class VerificationResult(Enum):
    """Results of verification process"""
    VERIFIED = "verified"               # Claim is confirmed accurate
    REFUTED = "refuted"                # Claim is proven false
    PARTIALLY_TRUE = "partially_true"   # Claim is partially accurate
    UNCERTAIN = "uncertain"            # Cannot determine accuracy
    CONTEXT_DEPENDENT = "context_dependent"  # Truth depends on context
    OUTDATED = "outdated"              # Information is no longer current


@dataclass
class Claim:
    """Represents a claim extracted from text"""
    id: str
    text: str
    claim_type: ClaimType
    confidence: float = 0.0  # Confidence in the claim (0-1)
    source_location: str = ""  # Where in the text this claim appears
    extracted_entities: List[str] = field(default_factory=list)
    verification_priority: int = 1  # 1=low, 5=high priority for verification
    
    def __post_init__(self):
        if not self.id:
            self.id = f"claim_{random.randint(1000, 9999)}"


@dataclass
class Evidence:
    """Represents evidence for or against a claim"""
    source: str
    content: str
    reliability_score: float  # 0-1, higher is more reliable
    relevance_score: float   # 0-1, higher is more relevant
    publication_date: Optional[datetime] = None
    source_type: str = "unknown"  # academic, news, government, etc.
    supports_claim: bool = True   # Whether this evidence supports the claim
    
    def get_quality_score(self) -> float:
        """Calculate overall quality of evidence"""
        base_score = (self.reliability_score + self.relevance_score) / 2
        
        # Adjust for recency (newer is generally better for some topics)
        if self.publication_date:
            days_old = (datetime.now() - self.publication_date).days
            recency_factor = max(0.1, 1.0 - (days_old / 3650))  # 10-year decay
            base_score *= (0.8 + 0.2 * recency_factor)
        
        return base_score


@dataclass
class VerificationStep:
    """Represents a single verification step"""
    step_id: str
    method: VerificationMethod
    claim_id: str
    description: str
    evidence_gathered: List[Evidence] = field(default_factory=list)
    result: Optional[VerificationResult] = None
    confidence: float = 0.0
    reasoning: str = ""
    time_taken: float = 0.0  # Simulated time in seconds
    
    def is_complete(self) -> bool:
        """Check if verification step is complete"""
        return self.result is not None and self.confidence > 0.0


@dataclass
class VerificationPlan:
    """Plan for verifying a set of claims"""
    plan_id: str
    claims: List[str]  # Claim IDs to verify
    verification_steps: List[VerificationStep] = field(default_factory=list)
    total_estimated_time: float = 0.0
    priority_order: List[str] = field(default_factory=list)
    
    def get_completion_rate(self) -> float:
        """Get percentage of verification steps completed"""
        if not self.verification_steps:
            return 0.0
        completed = len([s for s in self.verification_steps if s.is_complete()])
        return completed / len(self.verification_steps)


class ClaimExtractor:
    """Extracts claims from text for verification"""
    
    def __init__(self):
        self.claim_patterns = self._create_claim_patterns()
        self.entity_patterns = self._create_entity_patterns()
    
    def _create_claim_patterns(self) -> Dict[ClaimType, List[str]]:
        """Create patterns for identifying different types of claims"""
        return {
            ClaimType.FACTUAL: [
                r"(is|are|was|were)\s+([^.!?]+)",
                r"(has|have|had)\s+([^.!?]+)",
                r"(will|shall|can|could|may|might)\s+([^.!?]+)"
            ],
            ClaimType.STATISTICAL: [
                r"(\d+(?:\.\d+)?)\s*(%|percent|million|billion|thousand)",
                r"(approximately|about|around|nearly)\s+(\d+)",
                r"(increased|decreased|rose|fell)\s+by\s+(\d+)"
            ],
            ClaimType.HISTORICAL: [
                r"(in|during|since|from)\s+(\d{4}|\d{1,2}th century)",
                r"(happened|occurred|took place)\s+(in|during|on)\s+([^.!?]+)",
                r"(before|after)\s+(the|World War|[A-Z][^.!?]+)"
            ],
            ClaimType.SCIENTIFIC: [
                r"(research|study|studies)\s+(show|shows|found|indicate)",
                r"(according to|based on)\s+(research|science|data)",
                r"(proven|demonstrated|established)\s+(that|to be)"
            ],
            ClaimType.GEOGRAPHICAL: [
                r"(located|situated|found)\s+(in|at|near)\s+([A-Z][^.!?]+)",
                r"(capital|largest|smallest)\s+(city|country|state)",
                r"(borders|adjacent to|next to)\s+([A-Z][^.!?]+)"
            ],
            ClaimType.CAUSAL: [
                r"(causes|caused|leads to|results in)\s+([^.!?]+)",
                r"(because|due to|as a result of)\s+([^.!?]+)",
                r"(if|when)\s+([^.!?]+),?\s+(then|will|would)"
            ],
            ClaimType.COMPARATIVE: [
                r"(more|less|better|worse|faster|slower)\s+(than)",
                r"(compared to|versus|vs\.?)\s+([^.!?]+)",
                r"(highest|lowest|best|worst|largest|smallest)"
            ]
        }
    
    def _create_entity_patterns(self) -> List[str]:
        """Create patterns for extracting entities from claims"""
        return [
            r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",  # Proper nouns
            r"\d{4}",                           # Years
            r"\d+(?:\.\d+)?%?",                 # Numbers and percentages
            r"[A-Z]{2,}",                       # Acronyms
        ]
    
    def extract_claims(self, text: str) -> List[Claim]:
        """Extract claims from text"""
        claims = []
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Determine claim type
            claim_type = self._classify_claim(sentence)
            
            # Extract entities
            entities = self._extract_entities(sentence)
            
            # Calculate verification priority
            priority = self._calculate_priority(sentence, claim_type, entities)
            
            claim = Claim(
                id=f"claim_{i+1}",
                text=sentence,
                claim_type=claim_type,
                confidence=random.uniform(0.6, 0.9),
                source_location=f"sentence_{i+1}",
                extracted_entities=entities,
                verification_priority=priority
            )
            
            claims.append(claim)
        
        return claims
    
    def _classify_claim(self, sentence: str) -> ClaimType:
        """Classify the type of claim"""
        sentence_lower = sentence.lower()
        
        # Check patterns for each claim type
        for claim_type, patterns in self.claim_patterns.items():
            for pattern in patterns:
                if re.search(pattern, sentence_lower):
                    return claim_type
        
        # Default classification based on content
        if any(word in sentence_lower for word in ['study', 'research', 'experiment']):
            return ClaimType.SCIENTIFIC
        elif any(word in sentence_lower for word in ['year', 'century', 'ago']):
            return ClaimType.HISTORICAL
        elif re.search(r'\d+', sentence):
            return ClaimType.STATISTICAL
        else:
            return ClaimType.FACTUAL
    
    def _extract_entities(self, sentence: str) -> List[str]:
        """Extract entities from a sentence"""
        entities = []
        
        for pattern in self.entity_patterns:
            matches = re.finditer(pattern, sentence)
            for match in matches:
                entity = match.group().strip()
                if entity and entity not in entities:
                    entities.append(entity)
        
        return entities[:5]  # Limit to top 5 entities
    
    def _calculate_priority(self, sentence: str, claim_type: ClaimType, entities: List[str]) -> int:
        """Calculate verification priority for a claim"""
        priority = 1
        
        # Higher priority for certain claim types
        high_priority_types = [ClaimType.SCIENTIFIC, ClaimType.STATISTICAL, ClaimType.FACTUAL]
        if claim_type in high_priority_types:
            priority += 1
        
        # Higher priority for sentences with many entities
        if len(entities) >= 3:
            priority += 1
        
        # Higher priority for definitive statements
        definitive_words = ['always', 'never', 'all', 'none', 'every', 'definitely', 'certainly']
        if any(word in sentence.lower() for word in definitive_words):
            priority += 1
        
        # Higher priority for numerical claims
        if re.search(r'\d+', sentence):
            priority += 1
        
        return min(priority, 5)  # Cap at 5


class VerificationPlanner:
    """Plans verification process for claims"""
    
    def __init__(self):
        self.method_priorities = {
            ClaimType.FACTUAL: [VerificationMethod.SOURCE_CHECK, VerificationMethod.CROSS_REFERENCE],
            ClaimType.STATISTICAL: [VerificationMethod.QUANTITATIVE, VerificationMethod.SOURCE_CHECK],
            ClaimType.HISTORICAL: [VerificationMethod.PRIMARY_SOURCE, VerificationMethod.CROSS_REFERENCE],
            ClaimType.SCIENTIFIC: [VerificationMethod.EXPERT_CONSENSUS, VerificationMethod.REPRODUCIBILITY],
            ClaimType.GEOGRAPHICAL: [VerificationMethod.SOURCE_CHECK, VerificationMethod.CROSS_REFERENCE],
            ClaimType.BIOGRAPHICAL: [VerificationMethod.PRIMARY_SOURCE, VerificationMethod.SOURCE_CHECK],
            ClaimType.CAUSAL: [VerificationMethod.LOGICAL_ANALYSIS, VerificationMethod.EXPERT_CONSENSUS],
            ClaimType.COMPARATIVE: [VerificationMethod.QUANTITATIVE, VerificationMethod.CROSS_REFERENCE],
            ClaimType.DEFINITIONAL: [VerificationMethod.SOURCE_CHECK, VerificationMethod.EXPERT_CONSENSUS],
            ClaimType.PREDICTIVE: [VerificationMethod.LOGICAL_ANALYSIS, VerificationMethod.EXPERT_CONSENSUS]
        }
    
    def create_verification_plan(self, claims: List[Claim]) -> VerificationPlan:
        """Create a verification plan for a set of claims"""
        plan_id = f"plan_{random.randint(1000, 9999)}"
        
        # Sort claims by priority
        sorted_claims = sorted(claims, key=lambda c: c.verification_priority, reverse=True)
        priority_order = [c.id for c in sorted_claims]
        
        # Create verification steps
        verification_steps = []
        total_time = 0.0
        
        for claim in sorted_claims:
            steps = self._create_steps_for_claim(claim)
            verification_steps.extend(steps)
            total_time += sum(step.time_taken for step in steps)
        
        return VerificationPlan(
            plan_id=plan_id,
            claims=[c.id for c in claims],
            verification_steps=verification_steps,
            total_estimated_time=total_time,
            priority_order=priority_order
        )
    
    def _create_steps_for_claim(self, claim: Claim) -> List[VerificationStep]:
        """Create verification steps for a specific claim"""
        steps = []
        methods = self.method_priorities.get(claim.claim_type, [VerificationMethod.SOURCE_CHECK])
        
        # Create 2-3 verification steps per claim
        for i, method in enumerate(methods[:3]):
            step = VerificationStep(
                step_id=f"{claim.id}_step_{i+1}",
                method=method,
                claim_id=claim.id,
                description=self._get_step_description(method, claim),
                time_taken=self._estimate_time(method, claim)
            )
            steps.append(step)
        
        return steps
    
    def _get_step_description(self, method: VerificationMethod, claim: Claim) -> str:
        """Get description for a verification step"""
        descriptions = {
            VerificationMethod.SOURCE_CHECK: f"Check authoritative sources for: {claim.text[:50]}...",
            VerificationMethod.CROSS_REFERENCE: f"Cross-reference multiple sources for: {claim.text[:50]}...",
            VerificationMethod.LOGICAL_ANALYSIS: f"Analyze logical consistency of: {claim.text[:50]}...",
            VerificationMethod.QUANTITATIVE: f"Verify numerical data in: {claim.text[:50]}...",
            VerificationMethod.TEMPORAL: f"Verify temporal claims in: {claim.text[:50]}...",
            VerificationMethod.EXPERT_CONSENSUS: f"Check expert consensus on: {claim.text[:50]}...",
            VerificationMethod.PRIMARY_SOURCE: f"Find primary sources for: {claim.text[:50]}...",
            VerificationMethod.REPRODUCIBILITY: f"Check reproducibility of: {claim.text[:50]}..."
        }
        return descriptions.get(method, f"Verify claim: {claim.text[:50]}...")
    
    def _estimate_time(self, method: VerificationMethod, claim: Claim) -> float:
        """Estimate time needed for verification method"""
        base_times = {
            VerificationMethod.SOURCE_CHECK: 30.0,
            VerificationMethod.CROSS_REFERENCE: 60.0,
            VerificationMethod.LOGICAL_ANALYSIS: 45.0,
            VerificationMethod.QUANTITATIVE: 90.0,
            VerificationMethod.TEMPORAL: 40.0,
            VerificationMethod.EXPERT_CONSENSUS: 120.0,
            VerificationMethod.PRIMARY_SOURCE: 150.0,
            VerificationMethod.REPRODUCIBILITY: 180.0
        }
        
        base_time = base_times.get(method, 60.0)
        
        # Adjust for claim complexity
        complexity_factor = 1.0 + (claim.verification_priority - 1) * 0.2
        
        return base_time * complexity_factor


class EvidenceGatherer:
    """Gathers evidence for verification steps"""
    
    def __init__(self):
        self.source_reliability = self._create_source_reliability()
        self.simulated_sources = self._create_simulated_sources()
    
    def _create_source_reliability(self) -> Dict[str, float]:
        """Create reliability scores for different source types"""
        return {
            "academic_journal": 0.95,
            "government_official": 0.90,
            "reputable_news": 0.80,
            "encyclopedia": 0.85,
            "expert_interview": 0.80,
            "primary_document": 0.95,
            "peer_reviewed": 0.90,
            "statistical_agency": 0.95,
            "news_wire": 0.75,
            "blog": 0.40,
            "social_media": 0.30,
            "unknown": 0.50
        }
    
    def _create_simulated_sources(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create simulated sources for different topics"""
        return {
            "scientific": [
                {"source": "Nature Journal", "type": "academic_journal", "content": "Peer-reviewed research findings"},
                {"source": "Science Magazine", "type": "academic_journal", "content": "Scientific research publication"},
                {"source": "NIH Database", "type": "government_official", "content": "Government health database"}
            ],
            "historical": [
                {"source": "Encyclopedia Britannica", "type": "encyclopedia", "content": "Historical reference"},
                {"source": "National Archives", "type": "primary_document", "content": "Primary historical document"},
                {"source": "Academic History Journal", "type": "academic_journal", "content": "Historical analysis"}
            ],
            "statistical": [
                {"source": "Census Bureau", "type": "statistical_agency", "content": "Official government statistics"},
                {"source": "World Bank Data", "type": "statistical_agency", "content": "International economic data"},
                {"source": "Pew Research", "type": "reputable_news", "content": "Survey and polling data"}
            ],
            "geographical": [
                {"source": "Geographic Survey", "type": "government_official", "content": "Official geographic data"},
                {"source": "Atlas Reference", "type": "encyclopedia", "content": "Geographic reference"},
                {"source": "Satellite Data", "type": "government_official", "content": "Satellite imagery and data"}
            ],
            "general": [
                {"source": "Reuters", "type": "news_wire", "content": "News reporting"},
                {"source": "Associated Press", "type": "news_wire", "content": "News wire service"},
                {"source": "Expert Opinion", "type": "expert_interview", "content": "Subject matter expert"}
            ]
        }
    
    def gather_evidence(self, verification_step: VerificationStep, claim: Claim) -> List[Evidence]:
        """Gather evidence for a verification step"""
        evidence_list = []
        
        # Determine source category
        source_category = self._get_source_category(claim.claim_type)
        
        # Get relevant sources
        sources = self.simulated_sources.get(source_category, self.simulated_sources["general"])
        
        # Gather 2-4 pieces of evidence
        num_sources = min(len(sources), random.randint(2, 4))
        selected_sources = random.sample(sources, num_sources)
        
        for source_info in selected_sources:
            evidence = self._create_evidence(source_info, claim, verification_step)
            evidence_list.append(evidence)
        
        return evidence_list
    
    def _get_source_category(self, claim_type: ClaimType) -> str:
        """Map claim type to source category"""
        mapping = {
            ClaimType.SCIENTIFIC: "scientific",
            ClaimType.HISTORICAL: "historical",
            ClaimType.STATISTICAL: "statistical",
            ClaimType.GEOGRAPHICAL: "geographical",
            ClaimType.BIOGRAPHICAL: "historical",
            ClaimType.FACTUAL: "general",
            ClaimType.CAUSAL: "scientific",
            ClaimType.COMPARATIVE: "statistical",
            ClaimType.DEFINITIONAL: "general",
            ClaimType.PREDICTIVE: "scientific"
        }
        return mapping.get(claim_type, "general")
    
    def _create_evidence(self, source_info: Dict[str, Any], claim: Claim, 
                        verification_step: VerificationStep) -> Evidence:
        """Create evidence object from source information"""
        source_type = source_info["type"]
        reliability = self.source_reliability.get(source_type, 0.5)
        
        # Simulate relevance based on verification method
        relevance = self._calculate_relevance(verification_step.method, claim.claim_type)
        
        # Simulate whether evidence supports or contradicts the claim
        supports_claim = random.random() > 0.3  # 70% chance of supporting evidence
        
        # Create simulated content
        content = self._generate_evidence_content(source_info, claim, supports_claim)
        
        return Evidence(
            source=source_info["source"],
            content=content,
            reliability_score=reliability + random.uniform(-0.1, 0.1),
            relevance_score=relevance + random.uniform(-0.15, 0.15),
            publication_date=datetime.now(),
            source_type=source_type,
            supports_claim=supports_claim
        )
    
    def _calculate_relevance(self, method: VerificationMethod, claim_type: ClaimType) -> float:
        """Calculate relevance score based on method and claim type"""
        base_relevance = 0.7
        
        # Higher relevance for well-matched methods
        good_matches = {
            (VerificationMethod.QUANTITATIVE, ClaimType.STATISTICAL): 0.9,
            (VerificationMethod.PRIMARY_SOURCE, ClaimType.HISTORICAL): 0.9,
            (VerificationMethod.EXPERT_CONSENSUS, ClaimType.SCIENTIFIC): 0.9,
            (VerificationMethod.SOURCE_CHECK, ClaimType.FACTUAL): 0.8,
            (VerificationMethod.LOGICAL_ANALYSIS, ClaimType.CAUSAL): 0.8
        }
        
        return good_matches.get((method, claim_type), base_relevance)
    
    def _generate_evidence_content(self, source_info: Dict[str, Any], claim: Claim, 
                                 supports_claim: bool) -> str:
        """Generate simulated evidence content"""
        base_content = source_info["content"]
        
        if supports_claim:
            prefixes = ["Confirms that", "Evidence shows", "Data indicates", "Research demonstrates"]
        else:
            prefixes = ["Contradicts the claim", "Evidence suggests otherwise", "Data shows", "Research indicates"]
        
        prefix = random.choice(prefixes)
        
        # Extract key elements from claim for content
        key_terms = claim.extracted_entities[:2] if claim.extracted_entities else ["the subject"]
        
        content = f"{prefix} regarding {', '.join(key_terms)}. {base_content} related to the claim."
        
        return content


class VerificationExecutor:
    """Executes verification steps and determines results"""
    
    def __init__(self, evidence_gatherer: EvidenceGatherer):
        self.evidence_gatherer = evidence_gatherer
        self.verification_algorithms = self._create_verification_algorithms()
    
    def _create_verification_algorithms(self) -> Dict[VerificationMethod, Callable]:
        """Create verification algorithms for different methods"""
        return {
            VerificationMethod.SOURCE_CHECK: self._verify_source_check,
            VerificationMethod.CROSS_REFERENCE: self._verify_cross_reference,
            VerificationMethod.LOGICAL_ANALYSIS: self._verify_logical_analysis,
            VerificationMethod.QUANTITATIVE: self._verify_quantitative,
            VerificationMethod.TEMPORAL: self._verify_temporal,
            VerificationMethod.EXPERT_CONSENSUS: self._verify_expert_consensus,
            VerificationMethod.PRIMARY_SOURCE: self._verify_primary_source,
            VerificationMethod.REPRODUCIBILITY: self._verify_reproducibility
        }
    
    def execute_verification_step(self, step: VerificationStep, claim: Claim) -> VerificationStep:
        """Execute a single verification step"""
        # Gather evidence
        evidence = self.evidence_gatherer.gather_evidence(step, claim)
        step.evidence_gathered = evidence
        
        # Apply verification algorithm
        if step.method in self.verification_algorithms:
            result, confidence, reasoning = self.verification_algorithms[step.method](evidence, claim)
        else:
            result, confidence, reasoning = self._default_verification(evidence, claim)
        
        step.result = result
        step.confidence = confidence
        step.reasoning = reasoning
        
        return step
    
    def _verify_source_check(self, evidence: List[Evidence], claim: Claim) -> Tuple[VerificationResult, float, str]:
        """Verify using source checking"""
        if not evidence:
            return VerificationResult.UNCERTAIN, 0.0, "No evidence found"
        
        # Analyze source reliability
        avg_reliability = sum(e.reliability_score for e in evidence) / len(evidence)
        supporting_evidence = [e for e in evidence if e.supports_claim]
        
        if len(supporting_evidence) >= len(evidence) * 0.7:  # 70% support
            if avg_reliability > 0.8:
                return VerificationResult.VERIFIED, 0.9, f"High-quality sources confirm claim (reliability: {avg_reliability:.2f})"
            else:
                return VerificationResult.VERIFIED, 0.7, f"Sources confirm claim with moderate reliability ({avg_reliability:.2f})"
        elif len(supporting_evidence) < len(evidence) * 0.3:  # <30% support
            return VerificationResult.REFUTED, 0.8, f"Most sources contradict the claim"
        else:
            return VerificationResult.PARTIALLY_TRUE, 0.6, f"Mixed evidence from sources"
    
    def _verify_cross_reference(self, evidence: List[Evidence], claim: Claim) -> Tuple[VerificationResult, float, str]:
        """Verify using cross-referencing"""
        if len(evidence) < 2:
            return VerificationResult.UNCERTAIN, 0.3, "Insufficient sources for cross-referencing"
        
        supporting_count = len([e for e in evidence if e.supports_claim])
        total_count = len(evidence)
        support_ratio = supporting_count / total_count
        
        # Check for consensus
        if support_ratio >= 0.8:
            confidence = min(0.95, 0.7 + (support_ratio - 0.8) * 1.5)
            return VerificationResult.VERIFIED, confidence, f"Strong consensus across {total_count} sources ({support_ratio:.1%} agreement)"
        elif support_ratio <= 0.2:
            confidence = min(0.9, 0.7 + (0.2 - support_ratio) * 1.5)
            return VerificationResult.REFUTED, confidence, f"Strong disagreement across sources ({support_ratio:.1%} support)"
        else:
            return VerificationResult.PARTIALLY_TRUE, 0.5, f"Mixed consensus ({support_ratio:.1%} support) across {total_count} sources"
    
    def _verify_logical_analysis(self, evidence: List[Evidence], claim: Claim) -> Tuple[VerificationResult, float, str]:
        """Verify using logical analysis"""
        # Simulate logical consistency checking
        logical_issues = []
        
        # Check for logical fallacies (simulated)
        if any(word in claim.text.lower() for word in ['always', 'never', 'all', 'none']):
            logical_issues.append("Absolute statement may be too broad")
        
        if 'because' in claim.text.lower() and 'correlation' in claim.text.lower():
            logical_issues.append("Potential correlation/causation confusion")
        
        # Analyze evidence consistency
        supporting_evidence = [e for e in evidence if e.supports_claim]
        
        if logical_issues:
            return VerificationResult.CONTEXT_DEPENDENT, 0.4, f"Logical issues found: {'; '.join(logical_issues)}"
        elif len(supporting_evidence) >= len(evidence) * 0.6:
            return VerificationResult.VERIFIED, 0.8, "Claim appears logically consistent with evidence"
        else:
            return VerificationResult.UNCERTAIN, 0.5, "Logical analysis inconclusive"
    
    def _verify_quantitative(self, evidence: List[Evidence], claim: Claim) -> Tuple[VerificationResult, float, str]:
        """Verify numerical/quantitative claims"""
        # Extract numbers from claim
        numbers = re.findall(r'\d+(?:\.\d+)?', claim.text)
        
        if not numbers:
            return VerificationResult.UNCERTAIN, 0.3, "No quantitative data found in claim"
        
        # Simulate numerical verification
        supporting_evidence = [e for e in evidence if e.supports_claim]
        
        if len(supporting_evidence) >= len(evidence) * 0.7:
            # Check if numbers are "reasonable" (simulated)
            large_numbers = [float(n) for n in numbers if float(n) > 1000]
            if large_numbers:
                return VerificationResult.VERIFIED, 0.85, f"Numerical claims supported by {len(supporting_evidence)} sources"
            else:
                return VerificationResult.VERIFIED, 0.9, f"Quantitative data verified across multiple sources"
        else:
            return VerificationResult.REFUTED, 0.7, "Numerical claims not supported by available evidence"
    
    def _verify_temporal(self, evidence: List[Evidence], claim: Claim) -> Tuple[VerificationResult, float, str]:
        """Verify temporal/chronological claims"""
        # Look for dates and temporal indicators
        years = re.findall(r'\d{4}', claim.text)
        temporal_words = ['before', 'after', 'during', 'since', 'until', 'ago']
        
        has_temporal = any(word in claim.text.lower() for word in temporal_words) or years
        
        if not has_temporal:
            return VerificationResult.UNCERTAIN, 0.4, "No clear temporal claims found"
        
        supporting_evidence = [e for e in evidence if e.supports_claim]
        
        if len(supporting_evidence) >= len(evidence) * 0.6:
            return VerificationResult.VERIFIED, 0.8, f"Temporal claims supported by historical sources"
        else:
            return VerificationResult.REFUTED, 0.7, "Temporal claims contradicted by historical evidence"
    
    def _verify_expert_consensus(self, evidence: List[Evidence], claim: Claim) -> Tuple[VerificationResult, float, str]:
        """Verify based on expert consensus"""
        expert_sources = [e for e in evidence if e.source_type in ['academic_journal', 'expert_interview', 'peer_reviewed']]
        
        if not expert_sources:
            return VerificationResult.UNCERTAIN, 0.4, "No expert sources available"
        
        expert_support = len([e for e in expert_sources if e.supports_claim])
        
        if expert_support >= len(expert_sources) * 0.8:
            return VerificationResult.VERIFIED, 0.9, f"Strong expert consensus ({expert_support}/{len(expert_sources)} experts agree)"
        elif expert_support <= len(expert_sources) * 0.2:
            return VerificationResult.REFUTED, 0.85, f"Expert consensus against claim ({expert_support}/{len(expert_sources)} experts agree)"
        else:
            return VerificationResult.PARTIALLY_TRUE, 0.6, f"Mixed expert opinion ({expert_support}/{len(expert_sources)} experts agree)"
    
    def _verify_primary_source(self, evidence: List[Evidence], claim: Claim) -> Tuple[VerificationResult, float, str]:
        """Verify using primary sources"""
        primary_sources = [e for e in evidence if e.source_type in ['primary_document', 'government_official', 'statistical_agency']]
        
        if not primary_sources:
            return VerificationResult.UNCERTAIN, 0.5, "No primary sources available"
        
        primary_support = len([e for e in primary_sources if e.supports_claim])
        
        if primary_support >= len(primary_sources) * 0.7:
            return VerificationResult.VERIFIED, 0.95, f"Primary sources confirm claim ({primary_support}/{len(primary_sources)})"
        else:
            return VerificationResult.REFUTED, 0.9, f"Primary sources contradict claim ({primary_support}/{len(primary_sources)})"
    
    def _verify_reproducibility(self, evidence: List[Evidence], claim: Claim) -> Tuple[VerificationResult, float, str]:
        """Verify reproducibility of results"""
        # Simulate reproducibility analysis
        reproducible_sources = [e for e in evidence if e.source_type in ['academic_journal', 'peer_reviewed']]
        
        if not reproducible_sources:
            return VerificationResult.UNCERTAIN, 0.4, "No reproducible sources available"
        
        # Simulate reproducibility success rate
        reproducible_count = len([e for e in reproducible_sources if e.supports_claim and random.random() > 0.3])
        
        if reproducible_count >= len(reproducible_sources) * 0.7:
            return VerificationResult.VERIFIED, 0.9, f"Results reproducible in {reproducible_count}/{len(reproducible_sources)} studies"
        else:
            return VerificationResult.UNCERTAIN, 0.5, f"Mixed reproducibility ({reproducible_count}/{len(reproducible_sources)} studies)"
    
    def _default_verification(self, evidence: List[Evidence], claim: Claim) -> Tuple[VerificationResult, float, str]:
        """Default verification when no specific method is available"""
        if not evidence:
            return VerificationResult.UNCERTAIN, 0.0, "No evidence available"
        
        supporting_evidence = [e for e in evidence if e.supports_claim]
        support_ratio = len(supporting_evidence) / len(evidence)
        
        if support_ratio >= 0.7:
            return VerificationResult.VERIFIED, 0.7, f"General evidence supports claim ({support_ratio:.1%})"
        elif support_ratio <= 0.3:
            return VerificationResult.REFUTED, 0.7, f"General evidence contradicts claim ({support_ratio:.1%})"
        else:
            return VerificationResult.PARTIALLY_TRUE, 0.5, f"Mixed evidence ({support_ratio:.1%} support)"


class ChainOfVerification:
    """Main Chain-of-Verification agent"""
    
    def __init__(self):
        self.claim_extractor = ClaimExtractor()
        self.planner = VerificationPlanner()
        self.evidence_gatherer = EvidenceGatherer()
        self.executor = VerificationExecutor(self.evidence_gatherer)
        self.verification_history: List[Dict[str, Any]] = []
    
    def verify_response(self, original_response: str, response_id: Optional[str] = None) -> Dict[str, Any]:
        """Perform chain-of-verification on a response"""
        if response_id is None:
            response_id = f"response_{random.randint(1000, 9999)}"
        
        print(f"\n🔍 Chain-of-Verification Analysis")
        print("=" * 60)
        print(f"Response ID: {response_id}")
        print(f"Original Response: {original_response}")
        
        # Step 1: Extract claims
        print(f"\n📝 Step 1: Claim Extraction")
        claims = self.claim_extractor.extract_claims(original_response)
        print(f"Extracted {len(claims)} claims for verification")
        
        for i, claim in enumerate(claims, 1):
            print(f"  {i}. [{claim.claim_type.value}] {claim.text}")
            print(f"     Priority: {claim.verification_priority}, Entities: {', '.join(claim.extracted_entities[:3])}")
        
        if not claims:
            return {
                "response_id": response_id,
                "original_response": original_response,
                "claims_found": 0,
                "verification_needed": False,
                "final_assessment": "No verifiable claims found"
            }
        
        # Step 2: Create verification plan
        print(f"\n📋 Step 2: Verification Planning")
        verification_plan = self.planner.create_verification_plan(claims)
        print(f"Created plan with {len(verification_plan.verification_steps)} verification steps")
        print(f"Estimated total time: {verification_plan.total_estimated_time:.1f} seconds")
        
        # Step 3: Execute verification
        print(f"\n⚡ Step 3: Verification Execution")
        completed_steps = []
        
        for step in verification_plan.verification_steps:
            print(f"\n   Executing: {step.description}")
            
            # Find the corresponding claim
            claim = next((c for c in claims if c.id == step.claim_id), None)
            if claim:
                completed_step = self.executor.execute_verification_step(step, claim)
                completed_steps.append(completed_step)
                
                if completed_step.result:
                    print(f"   Result: {completed_step.result.value} (confidence: {completed_step.confidence:.2f})")
                else:
                    print(f"   Result: Unknown (confidence: {completed_step.confidence:.2f})")
                print(f"   Evidence: {len(completed_step.evidence_gathered)} sources")
                print(f"   Reasoning: {completed_step.reasoning}")
        
        # Step 4: Aggregate results
        print(f"\n📊 Step 4: Result Aggregation")
        aggregated_results = self._aggregate_verification_results(claims, completed_steps)
        
        # Step 5: Generate revised response if needed
        revised_response, revision_confidence = self._generate_revised_response(
            original_response, claims, completed_steps, aggregated_results
        )
        
        result = {
            "response_id": response_id,
            "original_response": original_response,
            "revised_response": revised_response,
            "claims_found": len(claims),
            "claims_verified": len([s for s in completed_steps if s.result == VerificationResult.VERIFIED]),
            "claims_refuted": len([s for s in completed_steps if s.result == VerificationResult.REFUTED]),
            "claims_uncertain": len([s for s in completed_steps if s.result == VerificationResult.UNCERTAIN]),
            "verification_steps": len(completed_steps),
            "overall_accuracy_score": aggregated_results["overall_accuracy"],
            "confidence_score": aggregated_results["overall_confidence"],
            "revision_confidence": revision_confidence,
            "verification_needed": aggregated_results["overall_accuracy"] < 0.8,
            "detailed_results": {
                "claims": [{"id": c.id, "text": c.text, "type": c.claim_type.value} for c in claims],
                "verification_steps": [self._step_to_dict(s) for s in completed_steps],
                "aggregated_results": aggregated_results
            }
        }
        
        self.verification_history.append(result)
        
        print(f"\n✅ Verification Complete")
        print(f"Overall Accuracy: {aggregated_results['overall_accuracy']:.1%}")
        print(f"Confidence: {aggregated_results['overall_confidence']:.1%}")
        print(f"Revision Needed: {result['verification_needed']}")
        
        if revised_response != original_response:
            print(f"\n🔄 Revised Response:")
            print(revised_response)
        
        return result
    
    def _aggregate_verification_results(self, claims: List[Claim], 
                                      completed_steps: List[VerificationStep]) -> Dict[str, Any]:
        """Aggregate verification results across all claims"""
        if not completed_steps:
            return {
                "overall_accuracy": 0.0,
                "overall_confidence": 0.0,
                "verified_claims": 0,
                "refuted_claims": 0,
                "uncertain_claims": 0
            }
        
        # Count results by type
        result_counts = {
            VerificationResult.VERIFIED: 0,
            VerificationResult.REFUTED: 0,
            VerificationResult.PARTIALLY_TRUE: 0,
            VerificationResult.UNCERTAIN: 0,
            VerificationResult.CONTEXT_DEPENDENT: 0,
            VerificationResult.OUTDATED: 0
        }
        
        total_confidence = 0.0
        total_steps = len(completed_steps)
        
        for step in completed_steps:
            if step.result:
                result_counts[step.result] += 1
                total_confidence += step.confidence
        
        # Calculate overall scores
        verified_ratio = result_counts[VerificationResult.VERIFIED] / total_steps
        partially_verified_ratio = result_counts[VerificationResult.PARTIALLY_TRUE] / total_steps
        refuted_ratio = result_counts[VerificationResult.REFUTED] / total_steps
        
        # Overall accuracy considers verified and partially verified claims
        overall_accuracy = verified_ratio + (partially_verified_ratio * 0.5)
        overall_confidence = total_confidence / total_steps if total_steps > 0 else 0.0
        
        return {
            "overall_accuracy": overall_accuracy,
            "overall_confidence": overall_confidence,
            "verified_claims": result_counts[VerificationResult.VERIFIED],
            "refuted_claims": result_counts[VerificationResult.REFUTED],
            "partially_true_claims": result_counts[VerificationResult.PARTIALLY_TRUE],
            "uncertain_claims": result_counts[VerificationResult.UNCERTAIN],
            "total_steps": total_steps,
            "result_distribution": {result.value: count for result, count in result_counts.items() if count > 0}
        }
    
    def _generate_revised_response(self, original_response: str, claims: List[Claim],
                                 completed_steps: List[VerificationStep], 
                                 aggregated_results: Dict[str, Any]) -> Tuple[str, float]:
        """Generate a revised response based on verification results"""
        if aggregated_results["overall_accuracy"] >= 0.8:
            return original_response, aggregated_results["overall_confidence"]
        
        # Identify problematic claims
        refuted_claims = []
        uncertain_claims = []
        
        for step in completed_steps:
            if step.result == VerificationResult.REFUTED:
                claim = next((c for c in claims if c.id == step.claim_id), None)
                if claim:
                    refuted_claims.append((claim, step))
            elif step.result in [VerificationResult.UNCERTAIN, VerificationResult.CONTEXT_DEPENDENT]:
                claim = next((c for c in claims if c.id == step.claim_id), None)
                if claim:
                    uncertain_claims.append((claim, step))
        
        # Start with original response
        revised_response = original_response
        revision_confidence = 0.7
        
        # Add corrections for refuted claims
        if refuted_claims:
            corrections = []
            for claim, step in refuted_claims:
                correction = f"Note: The claim '{claim.text}' may not be accurate based on verification ({step.reasoning})."
                corrections.append(correction)
            
            revised_response += "\n\n⚠️ Verification Updates:\n" + "\n".join(corrections)
            revision_confidence -= len(refuted_claims) * 0.1
        
        # Add uncertainty notes for uncertain claims
        if uncertain_claims:
            uncertainties = []
            for claim, step in uncertain_claims:
                uncertainty = f"The claim '{claim.text}' requires further verification ({step.reasoning})."
                uncertainties.append(uncertainty)
            
            if not refuted_claims:  # Only add section if not already added
                revised_response += "\n\n🔍 Verification Notes:\n"
            else:
                revised_response += "\n"
            
            revised_response += "\n".join(uncertainties)
            revision_confidence -= len(uncertain_claims) * 0.05
        
        # Add overall confidence note
        accuracy_percentage = aggregated_results["overall_accuracy"] * 100
        revised_response += f"\n\n📊 Overall verification confidence: {accuracy_percentage:.0f}%"
        
        return revised_response, max(0.1, revision_confidence)
    
    def _step_to_dict(self, step: VerificationStep) -> Dict[str, Any]:
        """Convert verification step to dictionary"""
        return {
            "step_id": step.step_id,
            "method": step.method.value,
            "claim_id": step.claim_id,
            "result": step.result.value if step.result else None,
            "confidence": step.confidence,
            "reasoning": step.reasoning,
            "evidence_count": len(step.evidence_gathered),
            "evidence_quality": sum(e.get_quality_score() for e in step.evidence_gathered) / len(step.evidence_gathered) if step.evidence_gathered else 0.0
        }
    
    def get_verification_summary(self) -> Dict[str, Any]:
        """Get summary of all verifications performed"""
        if not self.verification_history:
            return {"message": "No verifications performed yet"}
        
        total_responses = len(self.verification_history)
        total_claims = sum(r["claims_found"] for r in self.verification_history)
        average_accuracy = sum(r["overall_accuracy_score"] for r in self.verification_history) / total_responses
        
        return {
            "total_responses_verified": total_responses,
            "total_claims_analyzed": total_claims,
            "average_accuracy_score": average_accuracy,
            "responses_needing_revision": len([r for r in self.verification_history if r["verification_needed"]]),
            "most_common_claim_types": self._get_most_common_claim_types(),
            "verification_performance": self._calculate_verification_performance()
        }
    
    def _get_most_common_claim_types(self) -> Dict[str, int]:
        """Get most common claim types across all verifications"""
        claim_type_counts = {}
        
        for verification in self.verification_history:
            for claim_info in verification["detailed_results"]["claims"]:
                claim_type = claim_info["type"]
                claim_type_counts[claim_type] = claim_type_counts.get(claim_type, 0) + 1
        
        return dict(sorted(claim_type_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _calculate_verification_performance(self) -> Dict[str, float]:
        """Calculate verification system performance metrics"""
        if not self.verification_history:
            return {}
        
        total_verifications = len(self.verification_history)
        
        # Calculate precision (how often we correctly identify issues)
        responses_flagged = len([r for r in self.verification_history if r["verification_needed"]])
        precision = responses_flagged / total_verifications if total_verifications > 0 else 0.0
        
        # Calculate average confidence
        average_confidence = sum(r["confidence_score"] for r in self.verification_history) / total_verifications
        
        # Calculate revision effectiveness
        revised_responses = [r for r in self.verification_history if r["revised_response"] != r["original_response"]]
        revision_rate = len(revised_responses) / total_verifications if total_verifications > 0 else 0.0
        
        return {
            "precision": precision,
            "average_confidence": average_confidence,
            "revision_rate": revision_rate,
            "average_claims_per_response": sum(r["claims_found"] for r in self.verification_history) / total_verifications
        }


def main():
    """Demonstration of the Chain-of-Verification (CoVe) pattern"""
    print("🔍 Chain-of-Verification (CoVe) Pattern Demonstration")
    print("=" * 80)
    print("This demonstrates systematic fact-checking and verification:")
    print("- Automatic claim extraction from responses")
    print("- Multi-method verification process")
    print("- Evidence gathering and quality assessment")
    print("- Confidence scoring and uncertainty quantification")
    print("- Response revision based on verification results")
    
    # Create Chain-of-Verification agent
    cove_agent = ChainOfVerification()
    
    # Test responses with different types of claims
    test_responses = [
        "The Earth's population reached 8 billion people in 2022. Climate change is caused primarily by human activities, with 97% of scientists agreeing on this fact. The Great Wall of China is visible from space, and it was built entirely during the Ming Dynasty.",
        
        "Artificial intelligence was invented by Alan Turing in 1950. Machine learning algorithms can achieve 100% accuracy on all problems. Python is the most popular programming language, used by over 80% of all programmers worldwide.",
        
        "The human brain has approximately 86 billion neurons. Memory formation involves the strengthening of synaptic connections through a process called long-term potentiation. All memories are stored permanently and never truly forgotten.",
        
        "The Pacific Ocean covers about 46% of the Earth's water surface. Mount Everest grows taller by approximately 4 millimeters each year due to tectonic activity. The deepest point in the ocean is the Challenger Deep at exactly 11,034 meters below sea level.",
        
        "Shakespeare wrote 37 plays and 154 sonnets during his lifetime. He invented over 1,700 words that are still used in English today. All of his works were written between 1590 and 1613, and he never left England."
    ]
    
    for i, response in enumerate(test_responses, 1):
        print(f"\n\n🔬 Verification Test Case {i}")
        print("=" * 80)
        
        result = cove_agent.verify_response(response, f"test_response_{i}")
        
        print(f"\n📈 Verification Summary:")
        print(f"  Claims found: {result['claims_found']}")
        print(f"  Claims verified: {result['claims_verified']}")
        print(f"  Claims refuted: {result['claims_refuted']}")
        print(f"  Claims uncertain: {result['claims_uncertain']}")
        print(f"  Overall accuracy: {result['overall_accuracy_score']:.1%}")
        print(f"  Confidence score: {result['confidence_score']:.1%}")
        print(f"  Revision needed: {result['verification_needed']}")
        
        if result['verification_needed']:
            print(f"\n💡 Key issues found:")
            for step_info in result['detailed_results']['verification_steps']:
                if step_info['result'] in ['refuted', 'uncertain']:
                    print(f"    • {step_info['reasoning']}")
    
    # Overall performance summary
    print(f"\n\n📊 Chain-of-Verification Performance Summary")
    print("=" * 80)
    
    summary = cove_agent.get_verification_summary()
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        elif isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    print("\n\n🎯 Key Chain-of-Verification Features Demonstrated:")
    print("✅ Automatic claim extraction and classification")
    print("✅ Multi-method verification strategies")
    print("✅ Evidence gathering from multiple source types")
    print("✅ Quality assessment of evidence sources")
    print("✅ Confidence scoring and uncertainty quantification")
    print("✅ Systematic fact-checking process")
    print("✅ Response revision based on verification results")
    print("✅ Performance tracking and improvement metrics")


if __name__ == "__main__":
    main()