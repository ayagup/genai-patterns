"""
Pattern 168: Constitutional Chain

Description:
    The Constitutional Chain pattern ensures AI outputs align with predefined principles,
    values, and rules through iterative self-critique and revision. The agent generates
    an initial response, critiques it against constitutional principles, and refines it
    until it meets all requirements.

Components:
    1. Constitution: Set of principles and rules
    2. Generator: Creates initial responses
    3. Critic: Evaluates responses against constitution
    4. Reviser: Improves responses based on critique
    5. Compliance Checker: Validates final alignment
    6. Iteration Manager: Controls refinement loops

Use Cases:
    - Ethical AI systems
    - Content moderation
    - Policy-compliant generation
    - Bias reduction
    - Safety-critical applications
    - Regulated industries

Benefits:
    - Value alignment
    - Explicit principles
    - Iterative improvement
    - Audit trail
    - Consistent behavior
    - Safety guarantees

Trade-offs:
    - Multiple LLM calls
    - Increased latency
    - May be overly cautious
    - Requires good constitution
    - Complexity overhead

LangChain Implementation:
    Implements iterative critique-revision loops with LangChain. Uses structured
    prompts for critique and revision stages with constitutional principles.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class PrincipleType(Enum):
    """Types of constitutional principles"""
    ETHICAL = "ethical"
    SAFETY = "safety"
    ACCURACY = "accuracy"
    BIAS = "bias"
    TONE = "tone"
    LEGAL = "legal"
    PRIVACY = "privacy"


class ComplianceLevel(Enum):
    """Compliance assessment levels"""
    COMPLIANT = "compliant"
    MINOR_ISSUES = "minor_issues"
    MAJOR_VIOLATIONS = "major_violations"
    CRITICAL = "critical"


@dataclass
class ConstitutionalPrinciple:
    """Represents a principle in the constitution"""
    name: str
    description: str
    principle_type: PrincipleType
    priority: int = 5  # 1-10, higher is more important
    examples_good: List[str] = field(default_factory=list)
    examples_bad: List[str] = field(default_factory=list)


@dataclass
class CritiqueResult:
    """Result of critiquing a response"""
    compliance_level: ComplianceLevel
    violations: List[str]
    suggestions: List[str]
    score: float  # 0-1 scale
    critique_text: str


@dataclass
class ConstitutionalResponse:
    """Final response with compliance information"""
    response: str
    iterations: int
    final_compliance: ComplianceLevel
    improvements_made: List[str]
    critique_history: List[CritiqueResult]


class Constitution:
    """Defines a set of principles for AI behavior"""
    
    def __init__(self, name: str):
        self.name = name
        self.principles: List[ConstitutionalPrinciple] = []
    
    def add_principle(self, principle: ConstitutionalPrinciple):
        """Add a principle to the constitution"""
        self.principles.append(principle)
    
    def get_principles_text(self) -> str:
        """Get formatted text of all principles"""
        lines = [f"Constitution: {self.name}", "=" * 60, ""]
        
        for i, principle in enumerate(sorted(self.principles, 
                                            key=lambda p: p.priority, 
                                            reverse=True), 1):
            lines.append(f"{i}. {principle.name} (Priority: {principle.priority})")
            lines.append(f"   Type: {principle.principle_type.value}")
            lines.append(f"   {principle.description}")
            if principle.examples_good:
                lines.append(f"   Good: {principle.examples_good[0]}")
            if principle.examples_bad:
                lines.append(f"   Bad: {principle.examples_bad[0]}")
            lines.append("")
        
        return "\n".join(lines)


class ConstitutionalChain:
    """Implements constitutional AI pattern with critique-revision loops"""
    
    def __init__(self, constitution: Constitution, max_iterations: int = 3):
        """
        Initialize constitutional chain
        
        Args:
            constitution: The constitution to follow
            max_iterations: Maximum number of revision iterations
        """
        self.constitution = constitution
        self.max_iterations = max_iterations
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        # Prompts
        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            ("user", "{query}")
        ])
        
        self.critique_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a constitutional AI critic. Evaluate the response 
            against the following principles:

{principles}

Identify any violations and provide specific suggestions for improvement."""),
            ("user", """Query: {query}

Response to evaluate:
{response}

Provide:
1. Compliance level: compliant, minor_issues, major_violations, or critical
2. List of violations (if any)
3. Specific suggestions for improvement
4. Overall score (0-1)

Format your response as:
COMPLIANCE: [level]
VIOLATIONS: [list violations or "none"]
SUGGESTIONS: [list suggestions]
SCORE: [0-1 score]
CRITIQUE: [detailed explanation]""")
        ])
        
        self.revision_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that revises responses to align with 
            constitutional principles. Improve the response based on the critique while 
            maintaining helpfulness and relevance."""),
            ("user", """Original query: {query}

Previous response:
{response}

Critique:
{critique}

Revise the response to address all issues while staying helpful and relevant.""")
        ])
        
        self.generation_chain = self.generation_prompt | self.llm | StrOutputParser()
        self.critique_chain = self.critique_prompt | self.llm | StrOutputParser()
        self.revision_chain = self.revision_prompt | self.llm | StrOutputParser()
    
    def generate_constitutional_response(self, query: str) -> ConstitutionalResponse:
        """Generate a response that complies with constitutional principles"""
        # Generate initial response
        response = self.generation_chain.invoke({"query": query})
        
        critique_history = []
        improvements = []
        iterations = 0
        
        principles_text = self.constitution.get_principles_text()
        
        for iteration in range(self.max_iterations):
            iterations += 1
            
            # Critique current response
            critique = self._critique_response(query, response, principles_text)
            critique_history.append(critique)
            
            # Check if compliant
            if critique.compliance_level == ComplianceLevel.COMPLIANT:
                break
            
            if critique.compliance_level == ComplianceLevel.CRITICAL:
                # Critical issues require immediate attention
                improvements.append(f"Iteration {iteration + 1}: Fixed critical issues")
            elif critique.compliance_level == ComplianceLevel.MAJOR_VIOLATIONS:
                improvements.append(f"Iteration {iteration + 1}: Fixed major violations")
            else:
                improvements.append(f"Iteration {iteration + 1}: Addressed minor issues")
            
            # Revise response
            response = self._revise_response(query, response, critique.critique_text)
        
        # Final critique
        final_critique = critique_history[-1]
        
        return ConstitutionalResponse(
            response=response,
            iterations=iterations,
            final_compliance=final_critique.compliance_level,
            improvements_made=improvements,
            critique_history=critique_history
        )
    
    def _critique_response(self, query: str, response: str, 
                          principles: str) -> CritiqueResult:
        """Critique a response against constitutional principles"""
        critique_text = self.critique_chain.invoke({
            "query": query,
            "response": response,
            "principles": principles
        })
        
        # Parse critique
        compliance_level = ComplianceLevel.COMPLIANT
        violations = []
        suggestions = []
        score = 1.0
        
        for line in critique_text.split('\n'):
            line = line.strip()
            if line.startswith('COMPLIANCE:'):
                level_str = line.split(':', 1)[1].strip().lower()
                if 'critical' in level_str:
                    compliance_level = ComplianceLevel.CRITICAL
                elif 'major' in level_str:
                    compliance_level = ComplianceLevel.MAJOR_VIOLATIONS
                elif 'minor' in level_str:
                    compliance_level = ComplianceLevel.MINOR_ISSUES
                else:
                    compliance_level = ComplianceLevel.COMPLIANT
            elif line.startswith('VIOLATIONS:'):
                viol_text = line.split(':', 1)[1].strip()
                if viol_text.lower() != 'none':
                    violations = [v.strip() for v in viol_text.split(',')]
            elif line.startswith('SUGGESTIONS:'):
                sugg_text = line.split(':', 1)[1].strip()
                suggestions = [s.strip() for s in sugg_text.split(',')]
            elif line.startswith('SCORE:'):
                try:
                    score = float(line.split(':', 1)[1].strip())
                except:
                    score = 0.8
        
        return CritiqueResult(
            compliance_level=compliance_level,
            violations=violations,
            suggestions=suggestions,
            score=score,
            critique_text=critique_text
        )
    
    def _revise_response(self, query: str, response: str, critique: str) -> str:
        """Revise response based on critique"""
        revised = self.revision_chain.invoke({
            "query": query,
            "response": response,
            "critique": critique
        })
        return revised


def demonstrate_constitutional_chain():
    """Demonstrate constitutional chain pattern"""
    print("=" * 80)
    print("CONSTITUTIONAL CHAIN PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic constitutional AI
    print("\n" + "=" * 80)
    print("Example 1: Basic Constitutional AI")
    print("=" * 80)
    
    # Create a simple constitution
    basic_constitution = Constitution("Basic Safety Constitution")
    
    basic_constitution.add_principle(ConstitutionalPrinciple(
        name="Helpful and Harmless",
        description="Responses must be helpful while avoiding harmful content",
        principle_type=PrincipleType.ETHICAL,
        priority=10,
        examples_good=["I can help you learn programming"],
        examples_bad=["I can help you hack systems"]
    ))
    
    basic_constitution.add_principle(ConstitutionalPrinciple(
        name="No Misinformation",
        description="Avoid spreading false or misleading information",
        principle_type=PrincipleType.ACCURACY,
        priority=9,
        examples_good=["According to research..."],
        examples_bad=["I think this is true but not sure..."]
    ))
    
    basic_constitution.add_principle(ConstitutionalPrinciple(
        name="Respectful Tone",
        description="Maintain respectful and professional communication",
        principle_type=PrincipleType.TONE,
        priority=7,
        examples_good=["Let me help you understand this"],
        examples_bad=["That's a stupid question"]
    ))
    
    print("\n" + basic_constitution.get_principles_text())
    
    # Create chain
    chain = ConstitutionalChain(basic_constitution, max_iterations=3)
    
    # Test query
    query = "Tell me how to get rich quick with cryptocurrency"
    
    print("\nQuery:", query)
    print("\nGenerating constitutional response...")
    print("-" * 60)
    
    result = chain.generate_constitutional_response(query)
    
    print(f"\nIterations required: {result.iterations}")
    print(f"Final compliance: {result.final_compliance.value}")
    
    if result.improvements_made:
        print("\nImprovements made:")
        for improvement in result.improvements_made:
            print(f"  - {improvement}")
    
    print(f"\nFinal response:")
    print(result.response)
    
    print("\n" + "-" * 60)
    print("Critique History:")
    for i, critique in enumerate(result.critique_history, 1):
        print(f"\nIteration {i}:")
        print(f"  Compliance: {critique.compliance_level.value}")
        print(f"  Score: {critique.score:.2f}")
        if critique.violations:
            print(f"  Violations: {', '.join(critique.violations[:2])}")
        if critique.suggestions:
            print(f"  Suggestions: {', '.join(critique.suggestions[:2])}")
    
    # Example 2: Content moderation constitution
    print("\n" + "=" * 80)
    print("Example 2: Content Moderation")
    print("=" * 80)
    
    moderation_constitution = Constitution("Content Moderation")
    
    moderation_constitution.add_principle(ConstitutionalPrinciple(
        name="No Hate Speech",
        description="Responses must not contain discriminatory or hateful content",
        principle_type=PrincipleType.ETHICAL,
        priority=10
    ))
    
    moderation_constitution.add_principle(ConstitutionalPrinciple(
        name="Age Appropriate",
        description="Content should be suitable for general audiences",
        principle_type=PrincipleType.SAFETY,
        priority=9
    ))
    
    moderation_constitution.add_principle(ConstitutionalPrinciple(
        name="No Personal Attacks",
        description="Avoid personal attacks or aggressive language",
        principle_type=PrincipleType.TONE,
        priority=8
    ))
    
    mod_chain = ConstitutionalChain(moderation_constitution, max_iterations=2)
    
    queries = [
        "What do you think about people who disagree with me?",
        "How do I respond to criticism online?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = mod_chain.generate_constitutional_response(query)
        print(f"Compliance: {result.final_compliance.value}")
        print(f"Response: {result.response[:120]}...")
    
    # Example 3: Professional communication constitution
    print("\n" + "=" * 80)
    print("Example 3: Professional Communication")
    print("=" * 80)
    
    prof_constitution = Constitution("Professional Communication")
    
    prof_constitution.add_principle(ConstitutionalPrinciple(
        name="Clarity",
        description="Responses must be clear and unambiguous",
        principle_type=PrincipleType.TONE,
        priority=8
    ))
    
    prof_constitution.add_principle(ConstitutionalPrinciple(
        name="Evidence-Based",
        description="Claims should be supported by evidence or clearly marked as opinions",
        principle_type=PrincipleType.ACCURACY,
        priority=9
    ))
    
    prof_constitution.add_principle(ConstitutionalPrinciple(
        name="Conciseness",
        description="Responses should be concise while complete",
        principle_type=PrincipleType.TONE,
        priority=6
    ))
    
    prof_chain = ConstitutionalChain(prof_constitution, max_iterations=3)
    
    query = "Explain quantum computing"
    print(f"\nQuery: {query}")
    print("\nGenerating professional response...")
    
    result = prof_chain.generate_constitutional_response(query)
    
    print(f"\nIterations: {result.iterations}")
    print(f"Final compliance: {result.final_compliance.value}")
    print(f"\nResponse:\n{result.response}")
    
    # Example 4: Comparing with and without constitutional chain
    print("\n" + "=" * 80)
    print("Example 4: Comparison - With vs Without Constitution")
    print("=" * 80)
    
    query = "Should I invest all my money in a single stock?"
    
    # Without constitution
    print(f"\nQuery: {query}")
    print("\n1. Without Constitutional Chain:")
    basic_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    basic_chain = ChatPromptTemplate.from_messages([
        ("user", "{query}")
    ]) | basic_llm | StrOutputParser()
    
    basic_response = basic_chain.invoke({"query": query})
    print(f"   {basic_response[:200]}...")
    
    # With constitution
    print("\n2. With Constitutional Chain:")
    safety_constitution = Constitution("Financial Safety")
    safety_constitution.add_principle(ConstitutionalPrinciple(
        name="Risk Disclosure",
        description="Always disclose risks in financial advice",
        principle_type=PrincipleType.ETHICAL,
        priority=10
    ))
    safety_constitution.add_principle(ConstitutionalPrinciple(
        name="No Guarantees",
        description="Avoid making guarantees about financial outcomes",
        principle_type=PrincipleType.ACCURACY,
        priority=9
    ))
    
    safety_chain = ConstitutionalChain(safety_constitution, max_iterations=2)
    result = safety_chain.generate_constitutional_response(query)
    print(f"   {result.response[:200]}...")
    print(f"\n   Iterations: {result.iterations}")
    print(f"   Compliance: {result.final_compliance.value}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Constitutional Chain pattern enables:
✓ Value-aligned AI responses
✓ Iterative self-critique and improvement
✓ Explicit principle enforcement
✓ Audit trail of improvements
✓ Consistent ethical behavior
✓ Safety and compliance guarantees
✓ Transparent reasoning process

This pattern is valuable for:
- Production AI systems requiring safety
- Content moderation platforms
- Regulated industries (finance, healthcare)
- Ethical AI development
- Bias mitigation
- Policy-compliant generation
- High-stakes decision support
    """)


if __name__ == "__main__":
    demonstrate_constitutional_chain()
