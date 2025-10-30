"""
Pattern 035: Constitutional AI

Description:
    Constitutional AI enables agents to follow explicit principles and rules in their
    behavior through self-critique and revision. The agent evaluates its responses
    against a constitution (set of principles), identifies violations, and revises
    outputs to align with values and constraints.

Components:
    - Constitution: Set of explicit principles/rules
    - Critic: Evaluates responses against constitution
    - Reviser: Modifies responses to align with principles
    - Self-Critique Loop: Iterative improvement process
    - Principle Selector: Chooses relevant principles

Use Cases:
    - Ethical AI systems
    - Content moderation and safety
    - Bias mitigation
    - Policy compliance
    - Value alignment
    - Educational content generation

LangChain Implementation:
    Uses critique-revision loops with explicit constitutional principles to
    ensure agent behavior aligns with specified values and constraints.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class PrincipleCategory(Enum):
    """Categories of constitutional principles."""
    HARMLESSNESS = "harmlessness"
    HELPFULNESS = "helpfulness"
    HONESTY = "honesty"
    FAIRNESS = "fairness"
    PRIVACY = "privacy"
    LEGAL = "legal"
    ETHICAL = "ethical"


@dataclass
class Principle:
    """A constitutional principle."""
    id: str
    category: PrincipleCategory
    description: str
    critique_request: str  # What to ask when critiquing
    revision_request: str  # What to ask when revising
    priority: int = 1  # Higher priority = more important


@dataclass
class CritiqueResult:
    """Result of critiquing a response."""
    response: str
    principle: Principle
    violation_detected: bool
    critique: str
    severity: float  # 0.0 (no violation) to 1.0 (severe)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RevisionResult:
    """Result of revising a response."""
    original_response: str
    revised_response: str
    principles_applied: List[str]
    critiques: List[CritiqueResult]
    revision_count: int
    final_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class Constitution:
    """
    Set of principles that define acceptable agent behavior.
    """
    
    def __init__(self):
        self.principles: Dict[str, Principle] = {}
        self._initialize_default_principles()
    
    def _initialize_default_principles(self):
        """Initialize with common constitutional principles."""
        default_principles = [
            Principle(
                id="harmless_1",
                category=PrincipleCategory.HARMLESSNESS,
                description="Do not provide information that could cause harm",
                critique_request="Does this response contain information that could cause physical or psychological harm?",
                revision_request="Revise to remove any harmful content while maintaining helpfulness",
                priority=10
            ),
            Principle(
                id="fair_1",
                category=PrincipleCategory.FAIRNESS,
                description="Treat all groups fairly without bias",
                critique_request="Does this response show bias or unfair treatment of any group?",
                revision_request="Revise to be fair and unbiased toward all groups",
                priority=9
            ),
            Principle(
                id="honest_1",
                category=PrincipleCategory.HONESTY,
                description="Be truthful and acknowledge uncertainty",
                critique_request="Is this response truthful? Does it acknowledge uncertainties?",
                revision_request="Revise to be more truthful and acknowledge any uncertainties",
                priority=8
            ),
            Principle(
                id="privacy_1",
                category=PrincipleCategory.PRIVACY,
                description="Respect privacy and confidentiality",
                critique_request="Does this response violate privacy or request sensitive information?",
                revision_request="Revise to respect privacy and avoid requesting sensitive information",
                priority=7
            ),
            Principle(
                id="helpful_1",
                category=PrincipleCategory.HELPFULNESS,
                description="Provide helpful and relevant information",
                critique_request="Is this response helpful and relevant to the query?",
                revision_request="Revise to be more helpful and relevant",
                priority=6
            ),
            Principle(
                id="legal_1",
                category=PrincipleCategory.LEGAL,
                description="Do not encourage illegal activities",
                critique_request="Does this response encourage or provide guidance for illegal activities?",
                revision_request="Revise to remove any encouragement of illegal activities",
                priority=10
            ),
        ]
        
        for principle in default_principles:
            self.principles[principle.id] = principle
    
    def add_principle(self, principle: Principle):
        """Add a custom principle."""
        self.principles[principle.id] = principle
    
    def get_principles_by_category(self, category: PrincipleCategory) -> List[Principle]:
        """Get all principles in a category."""
        return [p for p in self.principles.values() if p.category == category]
    
    def get_top_principles(self, n: int = 5) -> List[Principle]:
        """Get top N principles by priority."""
        return sorted(self.principles.values(), key=lambda p: p.priority, reverse=True)[:n]


class ConstitutionalAgent:
    """
    Agent that follows constitutional principles through self-critique and revision.
    
    Features:
    - Multiple critique-revision iterations
    - Priority-based principle application
    - Violation detection and scoring
    - Transparent reasoning
    """
    
    def __init__(
        self,
        constitution: Optional[Constitution] = None,
        max_revisions: int = 3,
        temperature: float = 0.7
    ):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        self.constitution = constitution or Constitution()
        self.max_revisions = max_revisions
        
        # Prompt for initial response
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Respond to the user's query."),
            ("user", "{query}")
        ])
        
        # Prompt for critique
        self.critique_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a critic evaluating responses against constitutional principles.

Principle: {principle_description}

Critique Question: {critique_request}

Response to evaluate:
{response}

Evaluate if this response violates the principle. Respond in this format:
VIOLATION: [YES/NO]
SEVERITY: [0.0-1.0]
EXPLANATION: [detailed explanation]"""),
            ("user", "Evaluate this response")
        ])
        
        # Prompt for revision
        self.revision_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are revising a response to align with constitutional principles.

Original Response:
{response}

Critiques:
{critiques}

Revision Instructions:
{revision_instructions}

Provide a revised response that addresses all critiques while maintaining helpfulness."""),
            ("user", "Provide revised response")
        ])
    
    def generate_response(self, query: str) -> str:
        """Generate initial response to query."""
        chain = self.response_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
    
    def critique_response(
        self,
        response: str,
        principles: Optional[List[Principle]] = None
    ) -> List[CritiqueResult]:
        """
        Critique response against constitutional principles.
        """
        if principles is None:
            # Use top priority principles
            principles = self.constitution.get_top_principles(n=5)
        
        critiques = []
        
        for principle in principles:
            # Generate critique
            chain = self.critique_prompt | self.llm | StrOutputParser()
            critique_text = chain.invoke({
                "principle_description": principle.description,
                "critique_request": principle.critique_request,
                "response": response
            })
            
            # Parse critique
            violation, severity = self._parse_critique(critique_text)
            
            critique_result = CritiqueResult(
                response=response,
                principle=principle,
                violation_detected=violation,
                critique=critique_text,
                severity=severity
            )
            
            critiques.append(critique_result)
        
        return critiques
    
    def revise_response(
        self,
        response: str,
        critiques: List[CritiqueResult]
    ) -> str:
        """
        Revise response based on critiques.
        """
        # Filter to only critiques with violations
        violations = [c for c in critiques if c.violation_detected]
        
        if not violations:
            return response
        
        # Prepare critique summary
        critique_summary = "\n\n".join([
            f"Principle: {c.principle.description}\n"
            f"Severity: {c.severity:.2f}\n"
            f"Critique: {c.critique}"
            for c in violations
        ])
        
        # Prepare revision instructions
        revision_instructions = "\n".join([
            c.principle.revision_request for c in violations
        ])
        
        # Generate revision
        chain = self.revision_prompt | self.llm | StrOutputParser()
        revised = chain.invoke({
            "response": response,
            "critiques": critique_summary,
            "revision_instructions": revision_instructions
        })
        
        return revised
    
    def generate_constitutional_response(
        self,
        query: str,
        principles: Optional[List[Principle]] = None
    ) -> RevisionResult:
        """
        Generate response with constitutional alignment through critique-revision loop.
        
        Process:
        1. Generate initial response
        2. Critique against principles
        3. Revise if violations found
        4. Repeat until no violations or max iterations reached
        """
        # Generate initial response
        current_response = self.generate_response(query)
        original_response = current_response
        
        all_critiques = []
        revision_count = 0
        
        for iteration in range(self.max_revisions):
            # Critique current response
            critiques = self.critique_response(current_response, principles)
            all_critiques.extend(critiques)
            
            # Check for violations
            violations = [c for c in critiques if c.violation_detected]
            
            if not violations:
                # No violations, we're done
                break
            
            # Revise response
            current_response = self.revise_response(current_response, critiques)
            revision_count += 1
        
        # Calculate final score (lower is better - based on violations)
        if all_critiques:
            final_score = 1.0 - (sum(c.severity for c in all_critiques if c.violation_detected) / len(all_critiques))
        else:
            final_score = 1.0
        
        return RevisionResult(
            original_response=original_response,
            revised_response=current_response,
            principles_applied=[p.id for p in (principles or self.constitution.get_top_principles())],
            critiques=all_critiques,
            revision_count=revision_count,
            final_score=final_score
        )
    
    def _parse_critique(self, critique_text: str) -> tuple[bool, float]:
        """Parse critique result to extract violation and severity."""
        violation = False
        severity = 0.0
        
        lines = critique_text.split('\n')
        for line in lines:
            if line.startswith("VIOLATION:"):
                violation = "YES" in line.upper()
            elif line.startswith("SEVERITY:"):
                try:
                    severity_str = line.split(':')[1].strip()
                    severity = float(severity_str)
                except (ValueError, IndexError):
                    pass
        
        return violation, severity
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about constitutional principles."""
        return {
            "total_principles": len(self.constitution.principles),
            "by_category": {
                cat.value: len(self.constitution.get_principles_by_category(cat))
                for cat in PrincipleCategory
            },
            "max_revisions": self.max_revisions
        }


def demonstrate_constitutional_ai():
    """
    Demonstrates Constitutional AI with critique-revision loops
    and principle-based alignment.
    """
    print("=" * 80)
    print("CONSTITUTIONAL AI DEMONSTRATION")
    print("=" * 80)
    
    # Create constitution
    constitution = Constitution()
    
    # Create constitutional agent
    agent = ConstitutionalAgent(constitution=constitution, max_revisions=2)
    
    # Show constitution
    print("\n" + "=" * 80)
    print("Constitutional Principles")
    print("=" * 80)
    
    top_principles = constitution.get_top_principles(n=6)
    for i, principle in enumerate(top_principles, 1):
        print(f"\n{i}. {principle.description}")
        print(f"   Category: {principle.category.value}")
        print(f"   Priority: {principle.priority}")
    
    # Test 1: Response that might violate principles
    print("\n" + "=" * 80)
    print("Test 1: Potentially Problematic Query")
    print("=" * 80)
    
    query1 = "How can I hack into someone's email account?"
    print(f"\nQuery: {query1}")
    
    result1 = agent.generate_constitutional_response(query1)
    
    print(f"\nOriginal Response:\n{result1.original_response[:200]}...")
    print(f"\nFinal Response:\n{result1.revised_response[:200]}...")
    print(f"\nRevision Count: {result1.revision_count}")
    print(f"Final Score: {result1.final_score:.2f}")
    print(f"Violations Detected: {sum(1 for c in result1.critiques if c.violation_detected)}")
    
    # Test 2: Biased content
    print("\n" + "=" * 80)
    print("Test 2: Potentially Biased Content")
    print("=" * 80)
    
    query2 = "Are people from [specific group] less intelligent?"
    print(f"\nQuery: {query2}")
    
    result2 = agent.generate_constitutional_response(query2)
    
    print(f"\nOriginal Response:\n{result2.original_response[:200]}...")
    print(f"\nFinal Response:\n{result2.revised_response[:200]}...")
    print(f"\nRevision Count: {result2.revision_count}")
    print(f"Final Score: {result2.final_score:.2f}")
    
    # Test 3: Safe, helpful query
    print("\n" + "=" * 80)
    print("Test 3: Safe, Helpful Query")
    print("=" * 80)
    
    query3 = "How can I improve my programming skills?"
    print(f"\nQuery: {query3}")
    
    result3 = agent.generate_constitutional_response(query3)
    
    print(f"\nOriginal Response:\n{result3.original_response[:200]}...")
    print(f"\nFinal Response:\n{result3.revised_response[:200]}...")
    print(f"\nRevision Count: {result3.revision_count}")
    print(f"Final Score: {result3.final_score:.2f}")
    print(f"Violations Detected: {sum(1 for c in result3.critiques if c.violation_detected)}")
    
    # Test 4: Custom principle
    print("\n" + "=" * 80)
    print("Test 4: Custom Constitutional Principle")
    print("=" * 80)
    
    # Add custom principle
    custom_principle = Principle(
        id="custom_1",
        category=PrincipleCategory.ETHICAL,
        description="Promote environmental sustainability",
        critique_request="Does this response consider environmental impact and sustainability?",
        revision_request="Revise to include environmental considerations",
        priority=5
    )
    constitution.add_principle(custom_principle)
    
    query4 = "What's the best way to travel long distances?"
    print(f"\nQuery: {query4}")
    print(f"Custom Principle: {custom_principle.description}")
    
    result4 = agent.generate_constitutional_response(query4, principles=[custom_principle])
    
    print(f"\nOriginal Response:\n{result4.original_response[:200]}...")
    print(f"\nFinal Response:\n{result4.revised_response[:200]}...")
    print(f"\nRevision Count: {result4.revision_count}")
    
    # Show statistics
    print("\n" + "=" * 80)
    print("Agent Statistics")
    print("=" * 80)
    
    stats = agent.get_statistics()
    print(f"\nTotal Principles: {stats['total_principles']}")
    print(f"Principles by Category:")
    for category, count in stats['by_category'].items():
        if count > 0:
            print(f"  - {category}: {count}")
    print(f"Max Revisions: {stats['max_revisions']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Constitutional AI provides:
✓ Explicit principles for behavior alignment
✓ Self-critique against constitutional rules
✓ Iterative revision for principle compliance
✓ Transparent reasoning about violations
✓ Priority-based principle application
✓ Flexible custom principle addition

This pattern excels at:
- Ethical AI development
- Value alignment with human values
- Content safety and moderation
- Bias detection and mitigation
- Policy compliance
- Educational content generation

Constitutional components:
1. Constitution: Set of explicit principles
2. Critic: Evaluates responses against principles
3. Reviser: Modifies responses to align
4. Self-Critique Loop: Iterative improvement
5. Principle Selector: Chooses relevant rules

Principle categories:
- HARMLESSNESS: Avoid causing harm
- HELPFULNESS: Provide useful information
- HONESTY: Be truthful and acknowledge uncertainty
- FAIRNESS: Treat all groups equitably
- PRIVACY: Respect confidentiality
- LEGAL: Comply with laws
- ETHICAL: Follow ethical guidelines

Critique-revision process:
1. Generate initial response
2. Critique against each principle
3. Identify violations and severity
4. Revise response to address violations
5. Repeat until compliant or max iterations reached

Benefits:
- Interpretable: Clear principles and reasoning
- Flexible: Easy to add custom principles
- Effective: Iterative improvement works
- Transparent: Shows original and revised
- Scalable: Principles can be prioritized

Use Constitutional AI when you need:
- Ethical AI systems with explicit values
- Content moderation with clear rules
- Bias mitigation and fairness
- Policy compliance enforcement
- Value alignment with stakeholders
- Transparent decision-making
""")


if __name__ == "__main__":
    demonstrate_constitutional_ai()
