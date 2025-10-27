"""
Pattern 133: Domain Expert Agent

This pattern implements deep specialization in a specific domain with expert
knowledge management, domain-specific reasoning, and specialized capabilities.

Use Cases:
- Medical diagnosis assistant
- Legal research and analysis
- Financial advisory systems
- Technical support specialist
- Scientific research assistant

Category: Specialization (1/6 = 16.7%)
Complexity: Advanced
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from datetime import datetime
import hashlib


class ExpertiseDomain(Enum):
    """Domains of expertise."""
    MEDICINE = "medicine"
    LAW = "law"
    FINANCE = "finance"
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    ENGINEERING = "engineering"
    EDUCATION = "education"


class ExpertiseLevel(Enum):
    """Levels of expertise."""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"


class ReasoningMode(Enum):
    """Modes of expert reasoning."""
    DEDUCTIVE = "deductive"  # General to specific
    INDUCTIVE = "inductive"  # Specific to general
    ABDUCTIVE = "abductive"  # Best explanation
    ANALOGICAL = "analogical"  # By analogy
    CASE_BASED = "case_based"  # From past cases


@dataclass
class DomainKnowledge:
    """Knowledge item in the domain."""
    knowledge_id: str
    domain: ExpertiseDomain
    topic: str
    content: str
    confidence: float
    source: str
    last_updated: datetime = field(default_factory=datetime.now)
    related_topics: List[str] = field(default_factory=list)


@dataclass
class DomainRule:
    """Rule or heuristic in the domain."""
    rule_id: str
    name: str
    condition: str
    action: str
    confidence: float
    exceptions: List[str] = field(default_factory=list)


@dataclass
class ExpertCase:
    """Case from domain experience."""
    case_id: str
    title: str
    description: str
    problem: str
    solution: str
    outcome: str
    lessons_learned: List[str]
    tags: List[str] = field(default_factory=list)


@dataclass
class ExpertiseProfile:
    """Profile of expert capabilities."""
    domain: ExpertiseDomain
    level: ExpertiseLevel
    specializations: List[str]
    years_experience: int
    knowledge_count: int
    case_count: int
    certification_level: str


@dataclass
class DomainQuery:
    """Query requiring domain expertise."""
    query_id: str
    question: str
    context: Dict[str, Any]
    required_reasoning: ReasoningMode
    complexity: str  # simple, moderate, complex


@dataclass
class ExpertResponse:
    """Response from domain expert."""
    query_id: str
    answer: str
    reasoning: str
    confidence: float
    evidence: List[str]
    recommendations: List[str]
    caveats: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class KnowledgeBase:
    """Manages domain-specific knowledge."""
    
    def __init__(self, domain: ExpertiseDomain):
        self.domain = domain
        self.knowledge_items: Dict[str, DomainKnowledge] = {}
        self.rules: Dict[str, DomainRule] = {}
        self.cases: Dict[str, ExpertCase] = {}
    
    def add_knowledge(self, knowledge: DomainKnowledge):
        """Add knowledge item."""
        self.knowledge_items[knowledge.knowledge_id] = knowledge
    
    def add_rule(self, rule: DomainRule):
        """Add domain rule."""
        self.rules[rule.rule_id] = rule
    
    def add_case(self, case: ExpertCase):
        """Add expert case."""
        self.cases[case.case_id] = case
    
    def search_knowledge(self, query: str, top_k: int = 5) -> List[DomainKnowledge]:
        """Search knowledge base."""
        # Simplified keyword search
        query_words = set(query.lower().split())
        
        scored_knowledge = []
        for knowledge in self.knowledge_items.values():
            # Calculate relevance score
            content_words = set(knowledge.content.lower().split())
            topic_words = set(knowledge.topic.lower().split())
            
            overlap = len(query_words & (content_words | topic_words))
            score = overlap * knowledge.confidence
            
            if score > 0:
                scored_knowledge.append((score, knowledge))
        
        # Sort by score and return top_k
        scored_knowledge.sort(reverse=True, key=lambda x: x[0])
        return [k for _, k in scored_knowledge[:top_k]]
    
    def find_applicable_rules(self, context: Dict[str, Any]) -> List[DomainRule]:
        """Find rules applicable to context."""
        applicable = []
        
        for rule in self.rules.values():
            # Simple keyword matching (in production, use proper rule engine)
            if any(keyword in str(context.values()) for keyword in rule.condition.lower().split()):
                applicable.append(rule)
        
        return applicable
    
    def find_similar_cases(self, problem: str, top_k: int = 3) -> List[ExpertCase]:
        """Find similar cases."""
        problem_words = set(problem.lower().split())
        
        scored_cases = []
        for case in self.cases.values():
            # Calculate similarity
            case_words = set(case.problem.lower().split())
            overlap = len(problem_words & case_words)
            
            if overlap > 0:
                scored_cases.append((overlap, case))
        
        scored_cases.sort(reverse=True, key=lambda x: x[0])
        return [c for _, c in scored_cases[:top_k]]


class ReasoningEngine:
    """Performs domain-specific reasoning."""
    
    def __init__(self, domain: ExpertiseDomain):
        self.domain = domain
        self.reasoning_history: List[Dict[str, Any]] = []
    
    def reason(
        self,
        query: DomainQuery,
        knowledge: List[DomainKnowledge],
        rules: List[DomainRule],
        cases: List[ExpertCase]
    ) -> Dict[str, Any]:
        """Perform reasoning based on mode."""
        if query.required_reasoning == ReasoningMode.DEDUCTIVE:
            return self._deductive_reasoning(query, knowledge, rules)
        elif query.required_reasoning == ReasoningMode.INDUCTIVE:
            return self._inductive_reasoning(query, knowledge)
        elif query.required_reasoning == ReasoningMode.CASE_BASED:
            return self._case_based_reasoning(query, cases)
        elif query.required_reasoning == ReasoningMode.ANALOGICAL:
            return self._analogical_reasoning(query, knowledge, cases)
        else:  # ABDUCTIVE
            return self._abductive_reasoning(query, knowledge, rules)
    
    def _deductive_reasoning(
        self,
        query: DomainQuery,
        knowledge: List[DomainKnowledge],
        rules: List[DomainRule]
    ) -> Dict[str, Any]:
        """Deductive reasoning: Apply general rules to specific case."""
        reasoning_steps = ["Starting deductive reasoning"]
        conclusions = []
        
        # Apply rules
        for rule in rules:
            reasoning_steps.append(f"Applying rule: {rule.name}")
            conclusions.append(f"If {rule.condition}, then {rule.action}")
        
        # Apply knowledge
        for k in knowledge[:2]:
            reasoning_steps.append(f"Considering: {k.topic}")
        
        return {
            'mode': ReasoningMode.DEDUCTIVE.value,
            'steps': reasoning_steps,
            'conclusions': conclusions,
            'confidence': sum(r.confidence for r in rules) / max(1, len(rules))
        }
    
    def _inductive_reasoning(
        self,
        query: DomainQuery,
        knowledge: List[DomainKnowledge]
    ) -> Dict[str, Any]:
        """Inductive reasoning: Generalize from specific instances."""
        reasoning_steps = ["Starting inductive reasoning"]
        patterns = []
        
        # Look for patterns in knowledge
        topics = [k.topic for k in knowledge]
        common_themes = set()
        
        for topic in topics:
            words = set(topic.lower().split())
            common_themes.update(words)
        
        patterns.append(f"Common themes: {', '.join(list(common_themes)[:5])}")
        
        reasoning_steps.append("Analyzing specific instances")
        reasoning_steps.append(f"Found {len(knowledge)} relevant instances")
        reasoning_steps.append("Identifying common patterns")
        
        return {
            'mode': ReasoningMode.INDUCTIVE.value,
            'steps': reasoning_steps,
            'patterns': patterns,
            'confidence': 0.7
        }
    
    def _case_based_reasoning(
        self,
        query: DomainQuery,
        cases: List[ExpertCase]
    ) -> Dict[str, Any]:
        """Case-based reasoning: Learn from past cases."""
        reasoning_steps = ["Starting case-based reasoning"]
        adaptations = []
        
        if not cases:
            return {
                'mode': ReasoningMode.CASE_BASED.value,
                'steps': ['No similar cases found'],
                'adaptations': [],
                'confidence': 0.3
            }
        
        reasoning_steps.append(f"Retrieved {len(cases)} similar cases")
        
        for case in cases:
            reasoning_steps.append(f"Analyzing case: {case.title}")
            adaptations.append({
                'case': case.title,
                'solution': case.solution,
                'outcome': case.outcome,
                'lessons': case.lessons_learned[:2]
            })
        
        return {
            'mode': ReasoningMode.CASE_BASED.value,
            'steps': reasoning_steps,
            'adaptations': adaptations,
            'confidence': 0.8
        }
    
    def _analogical_reasoning(
        self,
        query: DomainQuery,
        knowledge: List[DomainKnowledge],
        cases: List[ExpertCase]
    ) -> Dict[str, Any]:
        """Analogical reasoning: Reason by analogy."""
        reasoning_steps = ["Starting analogical reasoning"]
        analogies = []
        
        reasoning_steps.append("Searching for analogous situations")
        
        if cases:
            for case in cases[:2]:
                analogies.append({
                    'source': case.title,
                    'mapping': f"{case.problem} is analogous to {query.question}",
                    'inference': case.solution
                })
        
        reasoning_steps.append(f"Found {len(analogies)} analogies")
        
        return {
            'mode': ReasoningMode.ANALOGICAL.value,
            'steps': reasoning_steps,
            'analogies': analogies,
            'confidence': 0.65
        }
    
    def _abductive_reasoning(
        self,
        query: DomainQuery,
        knowledge: List[DomainKnowledge],
        rules: List[DomainRule]
    ) -> Dict[str, Any]:
        """Abductive reasoning: Inference to best explanation."""
        reasoning_steps = ["Starting abductive reasoning"]
        hypotheses = []
        
        reasoning_steps.append("Generating possible explanations")
        
        # Generate hypotheses from rules and knowledge
        for rule in rules[:2]:
            hypotheses.append({
                'explanation': f"Based on {rule.name}: {rule.action}",
                'support': rule.confidence
            })
        
        for k in knowledge[:2]:
            hypotheses.append({
                'explanation': f"Based on {k.topic}: {k.content[:50]}...",
                'support': k.confidence
            })
        
        # Rank hypotheses
        hypotheses.sort(key=lambda x: x['support'], reverse=True)
        
        reasoning_steps.append(f"Evaluated {len(hypotheses)} hypotheses")
        reasoning_steps.append("Selected best explanation")
        
        return {
            'mode': ReasoningMode.ABDUCTIVE.value,
            'steps': reasoning_steps,
            'hypotheses': hypotheses,
            'best_explanation': hypotheses[0] if hypotheses else None,
            'confidence': 0.75
        }


class ExpertValidator:
    """Validates expert responses and checks confidence."""
    
    def __init__(self):
        self.validation_threshold = 0.6
    
    def validate_response(
        self,
        response: ExpertResponse,
        query: DomainQuery
    ) -> Dict[str, Any]:
        """Validate expert response."""
        validation = {
            'is_valid': True,
            'confidence_check': response.confidence >= self.validation_threshold,
            'completeness_score': self._check_completeness(response),
            'evidence_strength': self._evaluate_evidence(response.evidence),
            'issues': []
        }
        
        # Check confidence
        if response.confidence < self.validation_threshold:
            validation['issues'].append("Low confidence response")
            validation['is_valid'] = False
        
        # Check for empty answer
        if not response.answer or len(response.answer) < 10:
            validation['issues'].append("Insufficient answer detail")
            validation['is_valid'] = False
        
        # Check evidence
        if len(response.evidence) < 1:
            validation['issues'].append("No supporting evidence provided")
        
        # Check for caveats on complex queries
        if query.complexity == 'complex' and len(response.caveats) < 1:
            validation['issues'].append("Complex query should include caveats")
        
        return validation
    
    def _check_completeness(self, response: ExpertResponse) -> float:
        """Check response completeness."""
        score = 0.0
        
        if response.answer:
            score += 0.3
        if response.reasoning:
            score += 0.2
        if response.evidence:
            score += 0.2
        if response.recommendations:
            score += 0.2
        if response.caveats:
            score += 0.1
        
        return score
    
    def _evaluate_evidence(self, evidence: List[str]) -> float:
        """Evaluate strength of evidence."""
        if not evidence:
            return 0.0
        
        # Simple evaluation based on count and length
        strength = min(1.0, len(evidence) / 3.0)
        avg_length = sum(len(e) for e in evidence) / len(evidence)
        
        if avg_length > 50:
            strength *= 1.2
        
        return min(1.0, strength)


class DomainExpertAgent:
    """Agent with deep domain expertise."""
    
    def __init__(
        self,
        domain: ExpertiseDomain,
        level: ExpertiseLevel = ExpertiseLevel.EXPERT
    ):
        self.domain = domain
        self.level = level
        self.knowledge_base = KnowledgeBase(domain)
        self.reasoning_engine = ReasoningEngine(domain)
        self.validator = ExpertValidator()
        self.consultation_history: List[ExpertResponse] = []
        
        # Initialize with some domain knowledge
        self._initialize_knowledge()
    
    def _initialize_knowledge(self):
        """Initialize with domain-specific knowledge."""
        if self.domain == ExpertiseDomain.MEDICINE:
            self._add_medical_knowledge()
        elif self.domain == ExpertiseDomain.LAW:
            self._add_legal_knowledge()
        elif self.domain == ExpertiseDomain.FINANCE:
            self._add_financial_knowledge()
        elif self.domain == ExpertiseDomain.TECHNOLOGY:
            self._add_technical_knowledge()
    
    def _add_medical_knowledge(self):
        """Add medical domain knowledge."""
        self.knowledge_base.add_knowledge(DomainKnowledge(
            knowledge_id="med_001",
            domain=self.domain,
            topic="Fever Management",
            content="Fever is body temperature above 38°C. Common causes include infections, inflammation. Treatment includes rest, fluids, antipyretics.",
            confidence=0.95,
            source="Medical Guidelines 2023"
        ))
        
        self.knowledge_base.add_rule(DomainRule(
            rule_id="med_rule_001",
            name="Fever Protocol",
            condition="temperature > 38°C",
            action="Recommend rest, fluids, and fever reducer if needed",
            confidence=0.9
        ))
        
        self.knowledge_base.add_case(ExpertCase(
            case_id="med_case_001",
            title="Common Cold with Fever",
            description="Patient with fever, cough, and congestion",
            problem="Viral upper respiratory infection",
            solution="Symptomatic treatment, rest, fluids",
            outcome="Recovery in 7 days",
            lessons_learned=["Antibiotics not needed for viral infection", "Hydration is key"]
        ))
    
    def _add_legal_knowledge(self):
        """Add legal domain knowledge."""
        self.knowledge_base.add_knowledge(DomainKnowledge(
            knowledge_id="law_001",
            domain=self.domain,
            topic="Contract Law Basics",
            content="A valid contract requires offer, acceptance, consideration, and mutual intent. Both parties must have legal capacity.",
            confidence=0.98,
            source="Contract Law Principles"
        ))
        
        self.knowledge_base.add_rule(DomainRule(
            rule_id="law_rule_001",
            name="Contract Formation",
            condition="offer + acceptance + consideration",
            action="Contract is legally binding",
            confidence=0.95
        ))
    
    def _add_financial_knowledge(self):
        """Add financial domain knowledge."""
        self.knowledge_base.add_knowledge(DomainKnowledge(
            knowledge_id="fin_001",
            domain=self.domain,
            topic="Risk Management",
            content="Diversification reduces portfolio risk. Don't put all eggs in one basket. Asset allocation based on risk tolerance and time horizon.",
            confidence=0.92,
            source="Investment Principles"
        ))
        
        self.knowledge_base.add_rule(DomainRule(
            rule_id="fin_rule_001",
            name="Diversification Rule",
            condition="concentrated portfolio",
            action="Recommend diversification across asset classes",
            confidence=0.9
        ))
    
    def _add_technical_knowledge(self):
        """Add technical domain knowledge."""
        self.knowledge_base.add_knowledge(DomainKnowledge(
            knowledge_id="tech_001",
            domain=self.domain,
            topic="API Design Best Practices",
            content="REST APIs should be stateless, use proper HTTP methods, provide clear error messages, and version endpoints.",
            confidence=0.9,
            source="API Design Guidelines"
        ))
        
        self.knowledge_base.add_rule(DomainRule(
            rule_id="tech_rule_001",
            name="REST Principles",
            condition="designing REST API",
            action="Apply stateless design, proper HTTP verbs, resource-based URLs",
            confidence=0.88
        ))
    
    def consult(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        reasoning_mode: ReasoningMode = ReasoningMode.DEDUCTIVE
    ) -> ExpertResponse:
        """Provide expert consultation."""
        # Create query
        query = DomainQuery(
            query_id=hashlib.md5(question.encode()).hexdigest()[:8],
            question=question,
            context=context or {},
            required_reasoning=reasoning_mode,
            complexity=self._assess_complexity(question)
        )
        
        # Search knowledge base
        relevant_knowledge = self.knowledge_base.search_knowledge(question)
        applicable_rules = self.knowledge_base.find_applicable_rules(context or {})
        similar_cases = self.knowledge_base.find_similar_cases(question)
        
        # Perform reasoning
        reasoning_result = self.reasoning_engine.reason(
            query, relevant_knowledge, applicable_rules, similar_cases
        )
        
        # Generate response
        response = self._generate_response(
            query, reasoning_result, relevant_knowledge, applicable_rules, similar_cases
        )
        
        # Validate response
        validation = self.validator.validate_response(response, query)
        
        # Add validation info to caveats if needed
        if not validation['is_valid']:
            response.caveats.extend(validation['issues'])
        
        # Record consultation
        self.consultation_history.append(response)
        
        return response
    
    def _assess_complexity(self, question: str) -> str:
        """Assess query complexity."""
        words = question.split()
        
        if len(words) < 10:
            return 'simple'
        elif len(words) < 20:
            return 'moderate'
        else:
            return 'complex'
    
    def _generate_response(
        self,
        query: DomainQuery,
        reasoning_result: Dict[str, Any],
        knowledge: List[DomainKnowledge],
        rules: List[DomainRule],
        cases: List[ExpertCase]
    ) -> ExpertResponse:
        """Generate expert response."""
        # Build answer
        answer_parts = []
        
        if knowledge:
            answer_parts.append(f"Based on {self.domain.value} expertise:")
            answer_parts.append(knowledge[0].content[:100])
        
        if reasoning_result.get('conclusions'):
            answer_parts.append("Key conclusions:")
            answer_parts.extend(reasoning_result['conclusions'][:2])
        
        if reasoning_result.get('patterns'):
            answer_parts.append("Observed patterns:")
            answer_parts.extend(reasoning_result['patterns'][:2])
        
        answer = " ".join(answer_parts) if answer_parts else "Unable to provide definitive answer."
        
        # Build reasoning explanation
        reasoning = " -> ".join(reasoning_result.get('steps', []))
        
        # Collect evidence
        evidence = [f"{k.topic}: {k.source}" for k in knowledge[:3]]
        
        # Generate recommendations
        recommendations = []
        if rules:
            for rule in rules[:2]:
                recommendations.append(f"{rule.name}: {rule.action}")
        
        if cases:
            recommendations.append(f"Consider similar case: {cases[0].title}")
        
        # Add caveats
        caveats = []
        if reasoning_result.get('confidence', 1.0) < 0.7:
            caveats.append("Moderate confidence - additional verification recommended")
        
        if query.complexity == 'complex':
            caveats.append("Complex query - consult additional specialists if needed")
        
        return ExpertResponse(
            query_id=query.query_id,
            answer=answer,
            reasoning=reasoning,
            confidence=reasoning_result.get('confidence', 0.7),
            evidence=evidence,
            recommendations=recommendations,
            caveats=caveats
        )
    
    def get_expertise_profile(self) -> ExpertiseProfile:
        """Get agent's expertise profile."""
        return ExpertiseProfile(
            domain=self.domain,
            level=self.level,
            specializations=[k.topic for k in list(self.knowledge_base.knowledge_items.values())[:5]],
            years_experience=10 if self.level == ExpertiseLevel.EXPERT else 5,
            knowledge_count=len(self.knowledge_base.knowledge_items),
            case_count=len(self.knowledge_base.cases),
            certification_level="Board Certified" if self.level == ExpertiseLevel.EXPERT else "Certified"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consultation statistics."""
        if not self.consultation_history:
            return {'total_consultations': 0}
        
        avg_confidence = sum(r.confidence for r in self.consultation_history) / len(self.consultation_history)
        
        recommendations_given = sum(len(r.recommendations) for r in self.consultation_history)
        
        return {
            'total_consultations': len(self.consultation_history),
            'average_confidence': avg_confidence,
            'total_recommendations': recommendations_given,
            'knowledge_items': len(self.knowledge_base.knowledge_items),
            'rules': len(self.knowledge_base.rules),
            'cases': len(self.knowledge_base.cases)
        }


def demonstrate_domain_expert():
    """Demonstrate the Domain Expert Agent."""
    print("=" * 60)
    print("Domain Expert Agent Demonstration")
    print("=" * 60)
    
    print("\n1. MEDICAL EXPERT")
    print("-" * 60)
    
    medical_expert = DomainExpertAgent(
        domain=ExpertiseDomain.MEDICINE,
        level=ExpertiseLevel.EXPERT
    )
    
    profile = medical_expert.get_expertise_profile()
    print(f"Domain: {profile.domain.value.upper()}")
    print(f"Level: {profile.level.value}")
    print(f"Years Experience: {profile.years_experience}")
    print(f"Knowledge Items: {profile.knowledge_count}")
    print(f"Cases: {profile.case_count}")
    
    response = medical_expert.consult(
        "Patient has fever of 38.5°C and cough. What should I do?",
        context={'temperature': 38.5, 'symptoms': ['fever', 'cough']},
        reasoning_mode=ReasoningMode.DEDUCTIVE
    )
    
    print(f"\nQuery: Patient has fever of 38.5°C and cough")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Evidence: {len(response.evidence)} sources")
    print(f"Recommendations: {len(response.recommendations)}")
    if response.recommendations:
        for rec in response.recommendations:
            print(f"  - {rec}")
    
    print("\n\n2. FINANCIAL EXPERT")
    print("-" * 60)
    
    financial_expert = DomainExpertAgent(
        domain=ExpertiseDomain.FINANCE,
        level=ExpertiseLevel.EXPERT
    )
    
    response = financial_expert.consult(
        "I have all my money in tech stocks. Is this wise?",
        context={'portfolio': 'concentrated', 'sector': 'technology'},
        reasoning_mode=ReasoningMode.CASE_BASED
    )
    
    print(f"Query: All money in tech stocks")
    print(f"Answer: {response.answer[:150]}...")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Recommendations:")
    for rec in response.recommendations:
        print(f"  - {rec}")
    if response.caveats:
        print(f"Caveats:")
        for cav in response.caveats:
            print(f"  ⚠ {cav}")
    
    print("\n\n3. TECHNOLOGY EXPERT")
    print("-" * 60)
    
    tech_expert = DomainExpertAgent(
        domain=ExpertiseDomain.TECHNOLOGY,
        level=ExpertiseLevel.EXPERT
    )
    
    response = tech_expert.consult(
        "How should I design my REST API endpoints?",
        context={'project': 'REST API', 'stage': 'design'},
        reasoning_mode=ReasoningMode.DEDUCTIVE
    )
    
    print(f"Query: REST API design")
    print(f"Answer: {response.answer[:150]}...")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"\nReasoning Mode: {ReasoningMode.DEDUCTIVE.value}")
    
    print("\n\n4. LEGAL EXPERT")
    print("-" * 60)
    
    legal_expert = DomainExpertAgent(
        domain=ExpertiseDomain.LAW,
        level=ExpertiseLevel.EXPERT
    )
    
    response = legal_expert.consult(
        "What makes a contract legally binding?",
        reasoning_mode=ReasoningMode.DEDUCTIVE
    )
    
    print(f"Query: Contract law basics")
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Evidence:")
    for ev in response.evidence:
        print(f"  - {ev}")
    
    print("\n\n5. REASONING MODES COMPARISON")
    print("-" * 60)
    
    reasoning_modes = [
        ReasoningMode.DEDUCTIVE,
        ReasoningMode.INDUCTIVE,
        ReasoningMode.CASE_BASED,
        ReasoningMode.ANALOGICAL
    ]
    
    print("Testing different reasoning modes on medical query:")
    for mode in reasoning_modes:
        response = medical_expert.consult(
            "Common cold symptoms and treatment",
            reasoning_mode=mode
        )
        print(f"\n{mode.value.upper()}:")
        print(f"  Confidence: {response.confidence:.2f}")
        print(f"  Recommendations: {len(response.recommendations)}")
    
    print("\n\n6. STATISTICS")
    print("-" * 60)
    
    print("Medical Expert:")
    stats = medical_expert.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nFinancial Expert:")
    stats = financial_expert.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("🎉 Pattern 133 Complete!")
    print("Specialization Category: 16.7%")
    print("133/170 patterns implemented (78.2%)!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_domain_expert()
