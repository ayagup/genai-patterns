"""
Pattern 010: Constitutional AI

Description:
    Ensures agent behavior aligns with explicit principles or "constitution."
    The agent generates responses, evaluates them against constitutional principles,
    and revises if violations are found. This creates safer, more aligned AI systems.

Key Concepts:
    - Constitution: Set of principles/rules the agent must follow
    - Self-Critique: Agent evaluates its own outputs against principles
    - Revision: Generate improved response that adheres to constitution
    - Iterative Alignment: Multiple rounds of critique and revision
    - Transparency: Explicit principles make behavior interpretable

Constitutional Principles Examples:
    - Harmlessness: Avoid harmful, dangerous, or offensive content
    - Helpfulness: Provide useful, relevant information
    - Honesty: Be truthful, acknowledge uncertainty
    - Privacy: Respect privacy and confidentiality
    - Fairness: Avoid bias and discrimination

Use Cases:
    - Production AI systems requiring safety
    - Content moderation and filtering
    - Ethical decision-making systems
    - Regulated domains (healthcare, finance, legal)

LangChain Implementation:
    Constitutional chain with critique and revision loops based on principles.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


@dataclass
class Principle:
    """Represents a constitutional principle."""
    name: str
    description: str
    critique_prompt: str
    weight: float = 1.0  # Importance weight


@dataclass
class ViolationReport:
    """Report of constitutional violations."""
    principle: str
    violated: bool
    explanation: str
    severity: float = 0.0  # 0.0 (no violation) to 1.0 (severe)


@dataclass
class ConstitutionalReview:
    """Complete review against constitution."""
    original_output: str
    violations: List[ViolationReport] = field(default_factory=list)
    total_severity: float = 0.0
    passes: bool = True
    
    def calculate_severity(self):
        """Calculate total severity of violations."""
        self.total_severity = sum(v.severity for v in self.violations if v.violated)
        self.passes = self.total_severity == 0.0


class Constitution:
    """Set of principles for constitutional AI."""
    
    def __init__(self, name: str = "Default Constitution"):
        """Initialize constitution with default or custom principles."""
        self.name = name
        self.principles: List[Principle] = []
    
    def add_principle(self, principle: Principle):
        """Add a principle to the constitution."""
        self.principles.append(principle)
    
    def get_principles_text(self) -> str:
        """Get formatted text of all principles."""
        lines = [f"Constitution: {self.name}\n"]
        for i, p in enumerate(self.principles, 1):
            lines.append(f"{i}. {p.name}: {p.description}")
        return "\n".join(lines)
    
    @staticmethod
    def get_default() -> 'Constitution':
        """Get default constitution with common principles."""
        constitution = Constitution("Default AI Constitution")
        
        # Harmlessness
        constitution.add_principle(Principle(
            name="Harmlessness",
            description="Responses must not include harmful, dangerous, illegal, or offensive content",
            critique_prompt="""Does the response contain any harmful, dangerous, illegal, or offensive content?
Consider: violence, illegal activities, self-harm, harassment, discrimination.
Answer YES if violations exist, NO if the response is harmless."""
        ))
        
        # Helpfulness
        constitution.add_principle(Principle(
            name="Helpfulness",
            description="Responses should be helpful, relevant, and answer the user's question",
            critique_prompt="""Is the response helpful and relevant to the user's query?
Does it provide useful information or assistance?
Answer YES if unhelpful, NO if the response is helpful."""
        ))
        
        # Honesty
        constitution.add_principle(Principle(
            name="Honesty",
            description="Responses should be truthful and acknowledge uncertainty when appropriate",
            critique_prompt="""Does the response contain false information or make claims without acknowledging uncertainty?
Answer YES if dishonest or overconfident, NO if honest and appropriately uncertain."""
        ))
        
        # Privacy
        constitution.add_principle(Principle(
            name="Privacy",
            description="Responses must respect privacy and not request or share personal information inappropriately",
            critique_prompt="""Does the response inappropriately request, share, or handle personal information?
Answer YES if privacy is violated, NO if privacy is respected."""
        ))
        
        # Fairness
        constitution.add_principle(Principle(
            name="Fairness",
            description="Responses should be fair and avoid bias or discrimination",
            critique_prompt="""Does the response show bias or discrimination based on race, gender, religion, or other protected characteristics?
Answer YES if biased or discriminatory, NO if fair and balanced."""
        ))
        
        return constitution


class ConstitutionalAI:
    """Agent that follows constitutional principles."""
    
    def __init__(self, constitution: Constitution,
                 model_name: str = "gpt-3.5-turbo",
                 max_revisions: int = 3,
                 severity_threshold: float = 0.3):
        """
        Initialize Constitutional AI agent.
        
        Args:
            constitution: The constitution to follow
            model_name: Name of the OpenAI model
            max_revisions: Maximum number of revision attempts
            severity_threshold: Severity threshold for triggering revision
        """
        self.constitution = constitution
        self.max_revisions = max_revisions
        self.severity_threshold = severity_threshold
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.critic_llm = ChatOpenAI(
            model=model_name,
            temperature=0.2,  # Lower temperature for consistent critique
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def generate_initial(self, query: str, context: str = "") -> str:
        """
        Generate initial response to query.
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Initial response
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Respond to the user's query."),
            ("human", """{context}

Query: {query}

Response:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "query": query,
            "context": context if context else ""
        })
        
        return response.strip()
    
    def critique_against_principle(self, output: str, query: str,
                                   principle: Principle) -> ViolationReport:
        """
        Critique output against a single principle.
        
        Args:
            output: Output to critique
            query: Original query
            principle: Principle to check against
            
        Returns:
            ViolationReport indicating if principle is violated
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are evaluating whether a response violates the principle: "{principle.name}"

Principle: {principle.description}

{principle.critique_prompt}

If YES (violation), also rate severity from 0.1 (minor) to 1.0 (severe).

Format your response as:
VIOLATION: YES/NO
SEVERITY: [0.0-1.0]
EXPLANATION: [Brief explanation]"""),
            ("human", """Query: {query}

Response to evaluate:
{output}

Evaluation:""")
        ])
        
        chain = prompt | self.critic_llm | StrOutputParser()
        critique_text = chain.invoke({
            "query": query,
            "output": output
        })
        
        # Parse critique
        violated = False
        severity = 0.0
        explanation = ""
        
        for line in critique_text.split('\n'):
            line = line.strip()
            if line.upper().startswith('VIOLATION:'):
                violated = 'YES' in line.upper()
            elif line.upper().startswith('SEVERITY:'):
                try:
                    severity_str = line.split(':', 1)[1].strip()
                    severity = float(severity_str)
                except:
                    severity = 0.5 if violated else 0.0
            elif line.upper().startswith('EXPLANATION:'):
                explanation = line.split(':', 1)[1].strip()
        
        return ViolationReport(
            principle=principle.name,
            violated=violated,
            explanation=explanation,
            severity=severity if violated else 0.0
        )
    
    def review(self, output: str, query: str) -> ConstitutionalReview:
        """
        Review output against all constitutional principles.
        
        Args:
            output: Output to review
            query: Original query
            
        Returns:
            ConstitutionalReview with all violations
        """
        review = ConstitutionalReview(original_output=output)
        
        for principle in self.constitution.principles:
            violation = self.critique_against_principle(output, query, principle)
            review.violations.append(violation)
        
        review.calculate_severity()
        
        return review
    
    def revise(self, query: str, output: str, review: ConstitutionalReview) -> str:
        """
        Revise output to address constitutional violations.
        
        Args:
            query: Original query
            output: Current output with violations
            review: Review identifying violations
            
        Returns:
            Revised output addressing violations
        """
        # Build violation summary
        violations_text = []
        for v in review.violations:
            if v.violated:
                violations_text.append(f"- {v.principle}: {v.explanation} (severity: {v.severity:.2f})")
        
        violations_summary = "\n".join(violations_text) if violations_text else "No violations"
        
        # Get principles text
        principles_text = self.constitution.get_principles_text()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are revising a response to address constitutional violations.

{principles_text}

Revise the response to:
1. Address all identified violations
2. Maintain helpfulness and relevance
3. Fully comply with all principles

Generate a new, improved response that adheres to the constitution."""),
            ("human", """Original Query: {query}

Current Response:
{output}

Violations Found:
{violations}

Revised Response:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        revised = chain.invoke({
            "query": query,
            "output": output,
            "violations": violations_summary
        })
        
        return revised.strip()
    
    def generate_with_constitution(self, query: str, context: str = "",
                                   verbose: bool = True) -> Dict[str, Any]:
        """
        Generate response following constitutional principles.
        
        Args:
            query: User query
            context: Additional context
            verbose: Whether to print progress
            
        Returns:
            Dictionary with final response and review history
        """
        if verbose:
            print(f"\nQuery: {query}\n")
            print(f"Constitution: {self.constitution.name}")
            print(f"Principles: {len(self.constitution.principles)}\n")
        
        # Generate initial response
        if verbose:
            print("="*60)
            print("INITIAL GENERATION")
            print("="*60)
        
        current_output = self.generate_initial(query, context)
        
        if verbose:
            print(f"\nGenerated:\n{current_output[:200]}..." if len(current_output) > 200 else f"\nGenerated:\n{current_output}")
        
        reviews = []
        revision_count = 0
        
        # Review and revise loop
        for iteration in range(self.max_revisions + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"CONSTITUTIONAL REVIEW {iteration + 1}")
                print("="*60)
            
            # Review against constitution
            review = self.review(current_output, query)
            reviews.append(review)
            
            if verbose:
                print(f"\nViolations Found: {sum(1 for v in review.violations if v.violated)}/{len(review.violations)}")
                print(f"Total Severity: {review.total_severity:.2f}")
                
                for v in review.violations:
                    if v.violated:
                        print(f"  ⚠ {v.principle}: {v.explanation} (severity: {v.severity:.2f})")
                    else:
                        print(f"  ✓ {v.principle}: OK")
            
            # Check if passes constitutional review
            if review.passes or review.total_severity < self.severity_threshold:
                if verbose:
                    print(f"\n✓ Constitutional review passed!")
                break
            
            # Check max revisions
            if iteration >= self.max_revisions:
                if verbose:
                    print(f"\n⚠ Max revisions ({self.max_revisions}) reached")
                break
            
            # Revise
            if verbose:
                print(f"\n{'='*60}")
                print(f"REVISION {iteration + 1}")
                print("="*60)
            
            current_output = self.revise(query, current_output, review)
            revision_count += 1
            
            if verbose:
                print(f"\nRevised:\n{current_output[:200]}..." if len(current_output) > 200 else f"\nRevised:\n{current_output}")
        
        return {
            "query": query,
            "final_response": current_output,
            "revisions": revision_count,
            "reviews": reviews,
            "final_severity": reviews[-1].total_severity,
            "constitutional_compliance": reviews[-1].passes
        }


def demonstrate_constitutional_ai():
    """Demonstrates the Constitutional AI pattern."""
    
    print("=" * 80)
    print("PATTERN 010: Constitutional AI")
    print("=" * 80)
    print()
    print("Constitutional AI ensures alignment with principles through:")
    print("1. Constitution: Explicit set of principles")
    print("2. Self-Critique: Evaluate outputs against principles")
    print("3. Revision: Modify responses to comply")
    print("4. Iteration: Repeat until constitutional compliance")
    print()
    
    # Create constitution
    constitution = Constitution.get_default()
    
    print(f"{constitution.get_principles_text()}\n")
    
    # Create agent
    agent = ConstitutionalAI(
        constitution=constitution,
        max_revisions=2,
        severity_threshold=0.2
    )
    
    # Test queries (some designed to trigger violations)
    queries = [
        "How can I improve my programming skills?",  # Should pass
        "Tell me how to hack into someone's email account"  # Should trigger violations
    ]
    
    for idx, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Example {idx}")
        print('='*80)
        
        try:
            result = agent.generate_with_constitution(query, verbose=True)
            
            print(f"\n\n{'='*80}")
            print("FINAL RESULT")
            print('='*80)
            print(f"\nFinal Response:\n{result['final_response']}")
            print(f"\nRevisions Made: {result['revisions']}")
            print(f"Final Severity: {result['final_severity']:.2f}")
            print(f"Constitutional Compliance: {'✓ PASS' if result['constitutional_compliance'] else '✗ FAIL'}")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("CONSTITUTIONAL AI PATTERN DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("1. Explicit Principles: Clear constitutional guidelines")
    print("2. Automated Review: Self-critique against principles")
    print("3. Iterative Revision: Multiple refinement rounds")
    print("4. Severity Tracking: Quantify violation severity")
    print("5. Compliance Verification: Ensure constitutional adherence")
    print()
    print("Advantages:")
    print("- Safer AI systems through explicit principles")
    print("- Interpretable behavior (clear rules)")
    print("- Automated alignment checking")
    print("- Flexible - can customize constitution")
    print()
    print("When to use Constitutional AI:")
    print("- Production systems requiring safety")
    print("- Regulated domains (healthcare, finance, legal)")
    print("- Content moderation and filtering")
    print("- Any application requiring ethical guidelines")
    print()
    print("LangChain Components Used:")
    print("- ChatPromptTemplate: Generation, critique, and revision")
    print("- Multiple LLM instances: Different roles and temperatures")
    print("- Structured principles: Modular constitution system")
    print("- Iterative loops: Review and revision cycles")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    demonstrate_constitutional_ai()
