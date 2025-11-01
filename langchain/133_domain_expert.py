"""
Pattern 133: Domain Expert Agent

Description:
    The Domain Expert Agent pattern creates specialized agents with deep expertise
    in specific domains or fields. Unlike general-purpose agents, domain experts
    possess comprehensive knowledge, terminology, methodologies, and best practices
    specific to their area of specialization. They can provide expert-level advice,
    perform domain-specific reasoning, and apply specialized knowledge effectively.
    
    Domain expertise goes beyond simple knowledge retrieval - it includes understanding
    domain-specific workflows, recognizing patterns, applying heuristics, knowing
    what questions to ask, and understanding the nuances and edge cases within the
    domain. Expert agents can explain complex concepts using domain terminology,
    evaluate solutions using domain-specific criteria, and provide guidance that
    reflects years of accumulated expertise.
    
    This pattern is essential for applications requiring specialized knowledge such
    as medical diagnosis assistants, legal advisors, financial analysts, scientific
    research assistants, engineering consultants, and other professional AI systems.

Key Components:
    1. Domain Knowledge Base: Comprehensive domain-specific information
    2. Terminology Handler: Domain-specific vocabulary and jargon
    3. Best Practices Repository: Field-standard methodologies
    4. Case History: Examples and precedents from the domain
    5. Reasoning Engine: Domain-specific logic and heuristics
    6. Validation System: Domain-appropriate quality checks
    7. Reference System: Citations and sources

Domain Characteristics:
    1. Depth: Comprehensive coverage of specialized area
    2. Terminology: Field-specific vocabulary
    3. Methodology: Domain-standard approaches
    4. Context: Understanding of domain landscape
    5. Precedents: Historical cases and examples
    6. Standards: Industry regulations and best practices
    7. Boundaries: Knowing limits of expertise

Expert Behaviors:
    1. Ask clarifying questions
    2. Identify missing information
    3. Consider edge cases
    4. Reference standards and regulations
    5. Explain reasoning process
    6. Acknowledge uncertainty
    7. Provide multiple perspectives
    8. Use appropriate terminology

Use Cases:
    - Medical diagnosis assistants
    - Legal research and advisory
    - Financial analysis and planning
    - Engineering design consultation
    - Scientific research assistance
    - Tax preparation and planning
    - Architectural design
    - Software architecture advisory

Advantages:
    - Expert-level guidance
    - Domain-appropriate reasoning
    - Specialized knowledge application
    - Professional terminology usage
    - Best practice adherence
    - Quality assurance
    - Credible advice

Challenges:
    - Knowledge maintenance and updates
    - Handling edge cases
    - Avoiding overconfidence
    - Managing liability concerns
    - Balancing depth vs. accessibility
    - Keeping current with field changes
    - Regulatory compliance

LangChain Implementation:
    This implementation uses LangChain for:
    - Domain knowledge integration
    - Expert reasoning chains
    - Specialized RAG with domain documents
    - Domain-specific validation
    
Production Considerations:
    - Regularly update domain knowledge
    - Version control expert systems
    - Track expert vs. actual outcomes
    - Implement human expert review
    - Include disclaimers appropriately
    - Maintain audit trails
    - Ensure regulatory compliance
    - Support expert override mechanisms
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ExpertiseLevel(Enum):
    """Level of expertise."""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    MASTER = "master"


class ConfidenceLevel(Enum):
    """Confidence in response."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class DomainKnowledge:
    """
    Domain-specific knowledge item.
    
    Attributes:
        topic: Knowledge area
        content: Information content
        source: Where knowledge came from
        reliability: How reliable (0-1)
        last_updated: When last verified
        tags: Categorization tags
    """
    topic: str
    content: str
    source: str = "expert_knowledge"
    reliability: float = 0.9
    last_updated: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


@dataclass
class ExpertResponse:
    """
    Expert agent response.
    
    Attributes:
        answer: Main response
        confidence: Confidence level
        reasoning: Explanation of reasoning
        references: Supporting sources
        caveats: Important limitations or warnings
        follow_up_questions: Suggested clarifications
        alternatives: Alternative approaches
    """
    answer: str
    confidence: ConfidenceLevel
    reasoning: str = ""
    references: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)


@dataclass
class DomainExpertise:
    """
    Definition of domain expertise.
    
    Attributes:
        domain_name: Name of domain
        description: Domain description
        expertise_areas: Specific areas of expertise
        methodologies: Standard approaches
        key_concepts: Important domain concepts
        terminology: Domain-specific vocabulary
        standards: Relevant standards and regulations
        common_issues: Frequently encountered problems
        best_practices: Field best practices
    """
    domain_name: str
    description: str
    expertise_areas: List[str] = field(default_factory=list)
    methodologies: List[str] = field(default_factory=list)
    key_concepts: Dict[str, str] = field(default_factory=dict)
    terminology: Dict[str, str] = field(default_factory=dict)
    standards: List[str] = field(default_factory=list)
    common_issues: List[str] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)


class DomainExpertAgent:
    """
    Agent with specialized domain expertise.
    
    This agent provides expert-level guidance in a specific domain,
    using specialized knowledge, terminology, and reasoning.
    """
    
    def __init__(
        self,
        expertise: DomainExpertise,
        expertise_level: ExpertiseLevel = ExpertiseLevel.EXPERT,
        temperature: float = 0.3
    ):
        """
        Initialize domain expert agent.
        
        Args:
            expertise: Domain expertise definition
            expertise_level: Level of expertise
            temperature: LLM temperature (lower for more consistent expert advice)
        """
        self.expertise = expertise
        self.expertise_level = expertise_level
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.knowledge_base: List[DomainKnowledge] = []
        self.consultation_history: List[Dict[str, Any]] = []
    
    def _build_expert_context(self) -> str:
        """Build expert context for prompts."""
        context_parts = [
            f"You are a {self.expertise_level.value}-level expert in {self.expertise.domain_name}.",
            f"Domain: {self.expertise.description}",
        ]
        
        if self.expertise.expertise_areas:
            areas = ", ".join(self.expertise.expertise_areas)
            context_parts.append(f"Areas of expertise: {areas}")
        
        if self.expertise.methodologies:
            methods = ", ".join(self.expertise.methodologies)
            context_parts.append(f"Standard methodologies: {methods}")
        
        if self.expertise.standards:
            standards = ", ".join(self.expertise.standards)
            context_parts.append(f"Relevant standards: {standards}")
        
        if self.expertise.best_practices:
            practices = "\n".join([f"  - {p}" for p in self.expertise.best_practices])
            context_parts.append(f"Best practices:\n{practices}")
        
        if self.expertise.key_concepts:
            concepts = "\n".join([f"  - {k}: {v}" for k, v in self.expertise.key_concepts.items()])
            context_parts.append(f"Key concepts:\n{concepts}")
        
        return "\n".join(context_parts)
    
    def add_knowledge(self, knowledge: DomainKnowledge):
        """Add domain knowledge to expert's knowledge base."""
        self.knowledge_base.append(knowledge)
    
    def consult(
        self,
        question: str,
        context: Optional[str] = None,
        require_reasoning: bool = True
    ) -> ExpertResponse:
        """
        Provide expert consultation.
        
        Args:
            question: Question or problem
            context: Additional context
            require_reasoning: Include reasoning explanation
            
        Returns:
            Expert response
        """
        expert_context = self._build_expert_context()
        
        # Build prompt
        prompt_template = (
            "{expert_context}\n\n"
            "Provide expert guidance on the following:\n\n"
            "Question: {question}\n\n"
        )
        
        if context:
            prompt_template += "Additional Context: {context}\n\n"
        
        prompt_template += (
            "Provide a comprehensive expert response including:\n"
            "1. Direct answer\n"
            "2. Expert reasoning and analysis\n"
            "3. Relevant considerations and caveats\n"
            "4. Best practices to follow\n"
            "5. Potential alternatives or approaches\n\n"
            "Use appropriate domain terminology and cite relevant standards when applicable."
        )
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        input_data = {
            "expert_context": expert_context,
            "question": question
        }
        
        if context:
            input_data["context"] = context
        
        response_text = chain.invoke(input_data)
        
        # Assess confidence based on question clarity and domain coverage
        confidence = self._assess_confidence(question)
        
        # Parse response components
        answer, reasoning, caveats = self._parse_response(response_text)
        
        # Store consultation
        self.consultation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(),
            "confidence": confidence.value
        })
        
        return ExpertResponse(
            answer=answer,
            confidence=confidence,
            reasoning=reasoning,
            caveats=caveats
        )
    
    def _assess_confidence(self, question: str) -> ConfidenceLevel:
        """
        Assess confidence level in response.
        
        Args:
            question: The question asked
            
        Returns:
            Confidence level
        """
        # Check if question relates to core expertise
        question_lower = question.lower()
        
        # Count matches with expertise areas
        expertise_matches = sum(
            1 for area in self.expertise.expertise_areas
            if area.lower() in question_lower
        )
        
        # Terminology matches
        terminology_matches = sum(
            1 for term in self.expertise.terminology.keys()
            if term.lower() in question_lower
        )
        
        total_matches = expertise_matches + terminology_matches
        
        if total_matches >= 3:
            return ConfidenceLevel.VERY_HIGH
        elif total_matches >= 2:
            return ConfidenceLevel.HIGH
        elif total_matches >= 1:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _parse_response(self, response_text: str) -> Tuple[str, str, List[str]]:
        """Parse response into components."""
        # Simple parsing - in production, use more sophisticated extraction
        lines = response_text.split('\n')
        
        answer_lines = []
        reasoning_lines = []
        caveat_lines = []
        
        current_section = "answer"
        
        for line in lines:
            line_lower = line.lower()
            
            if "reasoning" in line_lower or "analysis" in line_lower:
                current_section = "reasoning"
            elif "caveat" in line_lower or "consideration" in line_lower or "warning" in line_lower:
                current_section = "caveats"
            elif line.strip():
                if current_section == "answer":
                    answer_lines.append(line)
                elif current_section == "reasoning":
                    reasoning_lines.append(line)
                elif current_section == "caveats":
                    caveat_lines.append(line)
        
        answer = "\n".join(answer_lines) if answer_lines else response_text
        reasoning = "\n".join(reasoning_lines)
        caveats = [line.strip('- ').strip() for line in caveat_lines if line.strip()]
        
        return answer, reasoning, caveats
    
    def explain_concept(self, concept: str, audience_level: str = "intermediate") -> str:
        """
        Explain a domain concept.
        
        Args:
            concept: Concept to explain
            audience_level: Target audience (novice/intermediate/advanced)
            
        Returns:
            Explanation
        """
        expert_context = self._build_expert_context()
        
        # Check if concept is in terminology
        concept_definition = self.expertise.terminology.get(concept, "")
        
        prompt = ChatPromptTemplate.from_template(
            "{expert_context}\n\n"
            "Explain the concept of '{concept}' for a {audience_level} audience.\n"
            "{definition_context}\n"
            "Provide:\n"
            "1. Clear definition\n"
            "2. Key characteristics\n"
            "3. Practical examples\n"
            "4. Common misconceptions\n"
            "5. Related concepts\n\n"
            "Use appropriate technical depth for the audience level."
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        definition_context = f"Domain definition: {concept_definition}" if concept_definition else ""
        
        explanation = chain.invoke({
            "expert_context": expert_context,
            "concept": concept,
            "audience_level": audience_level,
            "definition_context": definition_context
        })
        
        return explanation
    
    def identify_issues(self, scenario: str) -> List[str]:
        """
        Identify potential issues in a scenario.
        
        Args:
            scenario: Description of situation
            
        Returns:
            List of identified issues
        """
        expert_context = self._build_expert_context()
        
        # Include common issues from domain
        common_issues = "\n".join([f"- {issue}" for issue in self.expertise.common_issues])
        
        prompt = ChatPromptTemplate.from_template(
            "{expert_context}\n\n"
            "Common issues in this domain:\n"
            "{common_issues}\n\n"
            "Scenario:\n"
            "{scenario}\n\n"
            "As an expert, identify potential issues, problems, or concerns in this scenario.\n"
            "List each issue on a separate line starting with '- '."
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "expert_context": expert_context,
            "common_issues": common_issues,
            "scenario": scenario
        })
        
        # Parse issues
        issues = [
            line.strip('- ').strip()
            for line in response.split('\n')
            if line.strip().startswith('-')
        ]
        
        return issues
    
    def recommend_approach(self, problem: str) -> Dict[str, Any]:
        """
        Recommend approach for solving a problem.
        
        Args:
            problem: Problem description
            
        Returns:
            Recommended approach with rationale
        """
        expert_context = self._build_expert_context()
        
        methodologies = ", ".join(self.expertise.methodologies)
        practices = "\n".join([f"- {p}" for p in self.expertise.best_practices])
        
        prompt = ChatPromptTemplate.from_template(
            "{expert_context}\n\n"
            "Available methodologies: {methodologies}\n\n"
            "Best practices:\n"
            "{practices}\n\n"
            "Problem:\n"
            "{problem}\n\n"
            "Recommend the best approach to solve this problem. Include:\n"
            "APPROACH: [recommended approach]\n"
            "METHODOLOGY: [which methodology to use]\n"
            "STEPS: [key steps]\n"
            "RATIONALE: [why this approach]\n"
            "ALTERNATIVES: [other options]"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "expert_context": expert_context,
            "methodologies": methodologies,
            "practices": practices,
            "problem": problem
        })
        
        # Parse recommendation
        return {
            "raw_response": response,
            "problem": problem,
            "timestamp": datetime.now()
        }
    
    def get_expertise_summary(self) -> Dict[str, Any]:
        """Get summary of expert capabilities."""
        return {
            "domain": self.expertise.domain_name,
            "description": self.expertise.description,
            "expertise_level": self.expertise_level.value,
            "areas_of_expertise": self.expertise.expertise_areas,
            "methodologies": self.expertise.methodologies,
            "standards": self.expertise.standards,
            "total_consultations": len(self.consultation_history),
            "knowledge_base_size": len(self.knowledge_base)
        }


def demonstrate_domain_expert():
    """Demonstrate domain expert agent pattern."""
    
    print("=" * 80)
    print("DOMAIN EXPERT AGENT PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Software Architecture Expert
    print("\n" + "=" * 80)
    print("Example 1: Software Architecture Expert")
    print("=" * 80)
    
    software_expertise = DomainExpertise(
        domain_name="Software Architecture",
        description="Design and structure of software systems",
        expertise_areas=[
            "Microservices",
            "System Design",
            "Scalability",
            "Design Patterns",
            "API Design"
        ],
        methodologies=[
            "Domain-Driven Design",
            "Clean Architecture",
            "Event-Driven Architecture",
            "Hexagonal Architecture"
        ],
        key_concepts={
            "Separation of Concerns": "Dividing system into distinct features with minimal overlap",
            "SOLID Principles": "Five design principles for maintainable software",
            "Scalability": "Ability to handle growth in workload"
        },
        terminology={
            "Microservice": "Small, independent service focused on single business capability",
            "API Gateway": "Single entry point for all client requests",
            "Circuit Breaker": "Pattern preventing cascading failures"
        },
        standards=[
            "REST API Design",
            "OpenAPI Specification",
            "12-Factor App"
        ],
        common_issues=[
            "Tight coupling between components",
            "Lack of scalability planning",
            "Poor API design",
            "Missing error handling",
            "Inadequate monitoring"
        ],
        best_practices=[
            "Design for failure",
            "Implement proper logging and monitoring",
            "Use asynchronous communication where appropriate",
            "Maintain clear API contracts",
            "Document architectural decisions"
        ]
    )
    
    architect = DomainExpertAgent(
        expertise=software_expertise,
        expertise_level=ExpertiseLevel.EXPERT
    )
    
    print(f"\nExpert Domain: {software_expertise.domain_name}")
    print(f"Expertise Level: {architect.expertise_level.value}")
    print(f"Areas: {', '.join(software_expertise.expertise_areas[:3])}...")
    
    # Example 2: Expert consultation
    print("\n" + "=" * 80)
    print("Example 2: Expert Consultation")
    print("=" * 80)
    
    question = "I'm building an e-commerce platform. Should I use microservices or a monolith?"
    
    print(f"\nQuestion: {question}")
    print("\nExpert Response:")
    print("-" * 60)
    
    response = architect.consult(question)
    print(response.answer)
    print(f"\nConfidence: {response.confidence.value}")
    
    if response.caveats:
        print(f"\nCaveats:")
        for caveat in response.caveats:
            print(f"  - {caveat}")
    
    # Example 3: Concept explanation
    print("\n" + "=" * 80)
    print("Example 3: Explaining Domain Concepts")
    print("=" * 80)
    
    concept = "Microservice"
    audience = "intermediate"
    
    print(f"\nExplaining '{concept}' for {audience} audience:")
    print("-" * 60)
    
    explanation = architect.explain_concept(concept, audience_level=audience)
    print(explanation)
    
    # Example 4: Issue identification
    print("\n" + "=" * 80)
    print("Example 4: Identifying Issues in Scenario")
    print("=" * 80)
    
    scenario = """
    We have a system where the web frontend directly calls the database.
    All services share the same database. We deploy everything together as one unit.
    When traffic increases, the entire system slows down.
    """
    
    print("\nScenario:")
    print(scenario)
    print("\nIdentified Issues:")
    
    issues = architect.identify_issues(scenario)
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    
    # Example 5: Approach recommendation
    print("\n" + "=" * 80)
    print("Example 5: Recommending Approach")
    print("=" * 80)
    
    problem = "Need to scale our notification system that currently sends 10k emails/day but will need to handle 1M/day"
    
    print(f"\nProblem: {problem}")
    print("\nRecommended Approach:")
    print("-" * 60)
    
    recommendation = architect.recommend_approach(problem)
    print(recommendation["raw_response"])
    
    # Example 6: Medical Domain Expert
    print("\n" + "=" * 80)
    print("Example 6: Medical Domain Expert")
    print("=" * 80)
    
    medical_expertise = DomainExpertise(
        domain_name="General Medicine",
        description="Medical diagnosis and treatment",
        expertise_areas=[
            "Symptom Analysis",
            "Differential Diagnosis",
            "Treatment Planning",
            "Patient Care"
        ],
        methodologies=[
            "Evidence-Based Medicine",
            "Clinical Decision Making",
            "Patient-Centered Care"
        ],
        terminology={
            "Differential Diagnosis": "Process of distinguishing between similar conditions",
            "Prognosis": "Predicted course of a disease",
            "Contraindication": "Condition that makes treatment inadvisable"
        },
        standards=[
            "HIPAA Compliance",
            "Clinical Practice Guidelines",
            "Medical Ethics"
        ],
        best_practices=[
            "Always consider patient history",
            "Rule out serious conditions first",
            "Consider drug interactions",
            "Obtain informed consent",
            "Document thoroughly"
        ]
    )
    
    doctor = DomainExpertAgent(
        expertise=medical_expertise,
        expertise_level=ExpertiseLevel.EXPERT,
        temperature=0.2  # Very low for medical advice
    )
    
    print(f"\nExpert Domain: {medical_expertise.domain_name}")
    print(f"Standards: {', '.join(medical_expertise.standards)}")
    
    # Medical consultation
    medical_question = "What should I consider when a patient presents with chest pain?"
    
    print(f"\nMedical Question: {medical_question}")
    print("\nDISCLAIMER: This is a demonstration. Not real medical advice.")
    print("-" * 60)
    
    medical_response = doctor.consult(medical_question)
    print(medical_response.answer[:500] + "...")  # Truncate for demo
    print(f"\nConfidence: {medical_response.confidence.value}")
    
    # Example 7: Financial Expert
    print("\n" + "=" * 80)
    print("Example 7: Financial Planning Expert")
    print("=" * 80)
    
    financial_expertise = DomainExpertise(
        domain_name="Financial Planning",
        description="Personal and business financial planning",
        expertise_areas=[
            "Investment Strategy",
            "Retirement Planning",
            "Tax Planning",
            "Risk Management"
        ],
        methodologies=[
            "Modern Portfolio Theory",
            "Asset Allocation",
            "Risk Assessment"
        ],
        standards=[
            "SEC Regulations",
            "Fiduciary Duty",
            "CFP Standards"
        ],
        best_practices=[
            "Diversify investments",
            "Consider tax implications",
            "Match risk to time horizon",
            "Rebalance regularly",
            "Have emergency fund"
        ]
    )
    
    financial_advisor = DomainExpertAgent(
        expertise=financial_expertise,
        expertise_level=ExpertiseLevel.EXPERT
    )
    
    financial_question = "I'm 30 years old and want to start investing for retirement. Where should I start?"
    
    print(f"\nQuestion: {financial_question}")
    print("\nExpert Guidance:")
    print("-" * 60)
    
    financial_response = financial_advisor.consult(financial_question)
    print(financial_response.answer)
    
    # Example 8: Expertise summary
    print("\n" + "=" * 80)
    print("Example 8: Expert Capabilities Summary")
    print("=" * 80)
    
    summary = architect.get_expertise_summary()
    
    print("\nEXPERT PROFILE:")
    print("=" * 60)
    print(f"Domain: {summary['domain']}")
    print(f"Expertise Level: {summary['expertise_level']}")
    print(f"\nAreas of Expertise:")
    for area in summary['areas_of_expertise']:
        print(f"  • {area}")
    print(f"\nMethodologies:")
    for method in summary['methodologies']:
        print(f"  • {method}")
    print(f"\nTotal Consultations: {summary['total_consultations']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Domain Expert Agent Pattern")
    print("=" * 80)
    
    summary_text = """
    The Domain Expert Agent pattern demonstrated:
    
    1. EXPERTISE DEFINITION (Example 1):
       - Comprehensive domain specification
       - Areas of expertise
       - Standard methodologies
       - Key concepts and terminology
       - Relevant standards and regulations
       - Common issues and challenges
       - Best practices
    
    2. EXPERT CONSULTATION (Example 2):
       - Professional guidance
       - Domain-appropriate reasoning
       - Confidence assessment
       - Caveat identification
       - Best practice application
    
    3. CONCEPT EXPLANATION (Example 3):
       - Audience-appropriate depth
       - Domain terminology
       - Practical examples
       - Clear definitions
       - Related concepts
    
    4. ISSUE IDENTIFICATION (Example 4):
       - Expert problem recognition
       - Domain-specific concerns
       - Pattern matching
       - Risk assessment
       - Comprehensive analysis
    
    5. APPROACH RECOMMENDATION (Example 5):
       - Methodology selection
       - Step-by-step guidance
       - Rationale explanation
       - Alternative considerations
       - Best practice integration
    
    6. DOMAIN SPECIALIZATION (Examples 6-7):
       - Medical expertise
       - Financial planning
       - Different domain requirements
       - Standards compliance
       - Professional ethics
    
    7. MULTI-DOMAIN CAPABILITY (Examples 1, 6, 7):
       - Software architecture
       - Medical diagnosis
       - Financial planning
       - Domain-specific reasoning
       - Appropriate terminology
    
    8. TRANSPARENCY (Example 8):
       - Expertise summary
       - Capability disclosure
       - Consultation tracking
       - Knowledge base size
       - Performance metrics
    
    KEY BENEFITS:
    ✓ Expert-level guidance
    ✓ Domain-appropriate reasoning
    ✓ Specialized knowledge application
    ✓ Professional terminology
    ✓ Best practice adherence
    ✓ Quality assurance
    ✓ Credible advice
    ✓ Comprehensive coverage
    
    USE CASES:
    • Medical diagnosis assistance
    • Legal research and advisory
    • Financial analysis and planning
    • Engineering consultation
    • Scientific research support
    • Tax preparation
    • Architectural design
    • Software architecture advisory
    
    EXPERT BEHAVIORS:
    → Ask clarifying questions
    → Identify missing information
    → Consider edge cases
    → Reference standards
    → Explain reasoning
    → Acknowledge uncertainty
    → Provide multiple perspectives
    → Use appropriate terminology
    
    DOMAIN CHARACTERISTICS:
    • Depth: Comprehensive specialized coverage
    • Terminology: Field-specific vocabulary
    • Methodology: Standard approaches
    • Context: Domain landscape understanding
    • Precedents: Historical cases
    • Standards: Industry regulations
    • Boundaries: Expertise limits
    
    BEST PRACTICES:
    1. Regularly update domain knowledge
    2. Version control expert systems
    3. Track predictions vs. outcomes
    4. Include human expert review
    5. Add appropriate disclaimers
    6. Maintain audit trails
    7. Ensure regulatory compliance
    8. Support expert override
    
    TRADE-OFFS:
    • Depth vs. breadth
    • Accuracy vs. accessibility
    • Confidence vs. caution
    • Automation vs. human oversight
    
    PRODUCTION CONSIDERATIONS:
    → Regular knowledge base updates
    → Version control of expertise
    → Track expert vs. actual outcomes
    → Human expert review loops
    → Appropriate disclaimers
    → Complete audit trails
    → Regulatory compliance checks
    → Expert override mechanisms
    → Liability management
    → Certification where applicable
    
    This pattern enables agents to provide professional-grade expertise in
    specialized domains, delivering value that requires deep knowledge and
    experience while maintaining appropriate boundaries and transparency.
    """
    
    print(summary_text)


if __name__ == "__main__":
    demonstrate_domain_expert()
