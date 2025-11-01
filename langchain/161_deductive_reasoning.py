"""
Pattern 161: Deductive Reasoning

Description:
    Deductive reasoning derives specific conclusions from general premises using
    logical rules. If the premises are true and the logic is valid, the conclusion
    must be true. This is the foundation of formal logic and mathematical proof.

Components:
    - Premise collection and validation
    - Logical rule application
    - Inference chain construction
    - Conclusion derivation
    - Validity checking

Use Cases:
    - Logical problem solving
    - Mathematical proofs
    - Rule-based systems
    - Legal reasoning
    - Formal verification

Benefits:
    - Guaranteed valid conclusions
    - Transparent reasoning
    - Systematic approach
    - Verifiable logic

Trade-offs:
    - Requires valid premises
    - Limited to formal logic
    - May be rigid
    - Doesn't discover new rules

LangChain Implementation:
    Uses ChatOpenAI for logical inference, premise validation,
    and step-by-step deduction chains
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


class LogicalOperator(Enum):
    """Logical operators"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IF_AND_ONLY_IF = "iff"


class InferenceRule(Enum):
    """Common inference rules"""
    MODUS_PONENS = "modus_ponens"  # If P→Q and P, then Q
    MODUS_TOLLENS = "modus_tollens"  # If P→Q and ¬Q, then ¬P
    HYPOTHETICAL_SYLLOGISM = "hypothetical_syllogism"  # If P→Q and Q→R, then P→R
    DISJUNCTIVE_SYLLOGISM = "disjunctive_syllogism"  # If P∨Q and ¬P, then Q
    CONTRAPOSITIVE = "contrapositive"  # P→Q equivalent to ¬Q→¬P


@dataclass
class Premise:
    """Logical premise or statement"""
    statement: str
    is_axiom: bool = False  # Assumed true
    confidence: float = 1.0
    source: Optional[str] = None


@dataclass
class InferenceStep:
    """Single step in deductive reasoning"""
    conclusion: str
    rule_applied: str
    premises_used: List[str]
    confidence: float = 1.0


@dataclass
class DeductiveProof:
    """Complete deductive proof"""
    premises: List[Premise]
    steps: List[InferenceStep]
    final_conclusion: str
    is_valid: bool
    reasoning_trace: List[str] = field(default_factory=list)


class DeductiveReasoningAgent:
    """Agent that performs deductive reasoning"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize deductive reasoning agent
        
        Args:
            model_name: LLM model to use
        """
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.validator_llm = ChatOpenAI(model=model_name, temperature=0)
    
    def prove(self, premises: List[Premise], 
              goal: str,
              max_steps: int = 10) -> DeductiveProof:
        """
        Construct deductive proof from premises to goal
        
        Args:
            premises: List of premises
            goal: Goal conclusion to prove
            max_steps: Maximum reasoning steps
            
        Returns:
            DeductiveProof with reasoning chain
        """
        reasoning_trace = []
        steps = []
        
        reasoning_trace.append("Starting deductive proof...")
        reasoning_trace.append(f"Goal: {goal}")
        
        # Format premises
        premises_text = "\n".join([
            f"- {p.statement}" for p in premises
        ])
        reasoning_trace.append(f"Given premises:\n{premises_text}")
        
        # Generate proof steps
        for i in range(max_steps):
            # Check if goal is reached
            if steps and steps[-1].conclusion.lower() in goal.lower():
                reasoning_trace.append(f"Goal reached in {i} steps!")
                break
            
            # Generate next inference step
            step = self._generate_next_step(premises, steps, goal)
            if step:
                steps.append(step)
                reasoning_trace.append(
                    f"Step {i+1}: {step.conclusion} (via {step.rule_applied})"
                )
            else:
                reasoning_trace.append("No more valid inferences possible")
                break
        
        # Validate proof
        is_valid = self._validate_proof(premises, steps, goal)
        
        final_conclusion = steps[-1].conclusion if steps else "No conclusion reached"
        
        return DeductiveProof(
            premises=premises,
            steps=steps,
            final_conclusion=final_conclusion,
            is_valid=is_valid,
            reasoning_trace=reasoning_trace
        )
    
    def _generate_next_step(self, premises: List[Premise],
                           existing_steps: List[InferenceStep],
                           goal: str) -> Optional[InferenceStep]:
        """Generate next logical inference step"""
        
        # Format context
        premises_text = "\n".join([f"- {p.statement}" for p in premises])
        
        if existing_steps:
            derived_text = "\n".join([
                f"- {s.conclusion}" for s in existing_steps
            ])
        else:
            derived_text = "None yet"
        
        # Create inference prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a logical reasoning expert. Given premises and a goal,
generate the next logical inference step using valid deductive reasoning rules.

Common rules:
- Modus Ponens: If P→Q and P, then Q
- Modus Tollens: If P→Q and ¬Q, then ¬P
- Hypothetical Syllogism: If P→Q and Q→R, then P→R
- Disjunctive Syllogism: If P∨Q and ¬P, then Q

Return only the new conclusion and the rule used."""),
            ("user", """Premises:
{premises}

Already derived:
{derived}

Goal: {goal}

What is the next logical inference? Format:
Conclusion: [your conclusion]
Rule: [rule name]
Uses: [which premises/conclusions]""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "premises": premises_text,
            "derived": derived_text,
            "goal": goal
        })
        
        # Parse response
        return self._parse_inference_step(response)
    
    def _parse_inference_step(self, response: str) -> Optional[InferenceStep]:
        """Parse inference step from LLM response"""
        lines = response.strip().split('\n')
        conclusion = None
        rule = None
        uses = []
        
        for line in lines:
            line = line.strip()
            if line.lower().startswith('conclusion:'):
                conclusion = line.split(':', 1)[1].strip()
            elif line.lower().startswith('rule:'):
                rule = line.split(':', 1)[1].strip()
            elif line.lower().startswith('uses:'):
                uses_text = line.split(':', 1)[1].strip()
                uses = [u.strip() for u in uses_text.split(',')]
        
        if conclusion and rule:
            return InferenceStep(
                conclusion=conclusion,
                rule_applied=rule,
                premises_used=uses,
                confidence=1.0
            )
        
        return None
    
    def _validate_proof(self, premises: List[Premise],
                       steps: List[InferenceStep],
                       goal: str) -> bool:
        """Validate that the proof is logically sound"""
        
        if not steps:
            return False
        
        # Check if final conclusion matches goal
        final_matches = goal.lower() in steps[-1].conclusion.lower()
        
        # Use LLM to validate logical soundness
        premises_text = "\n".join([f"- {p.statement}" for p in premises])
        steps_text = "\n".join([
            f"{i+1}. {s.conclusion} (via {s.rule_applied})"
            for i, s in enumerate(steps)
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a logic validator. Check if the reasoning steps are logically valid."),
            ("user", """Premises:
{premises}

Reasoning steps:
{steps}

Goal: {goal}

Is this proof logically valid? Respond with 'yes' or 'no' and brief explanation.""")
        ])
        
        chain = prompt | self.validator_llm | StrOutputParser()
        
        response = chain.invoke({
            "premises": premises_text,
            "steps": steps_text,
            "goal": goal
        })
        
        is_valid = 'yes' in response.lower() and final_matches
        return is_valid
    
    def apply_rule(self, rule: InferenceRule, 
                   statements: List[str]) -> Optional[str]:
        """Apply specific inference rule to statements"""
        
        rule_descriptions = {
            InferenceRule.MODUS_PONENS: "If P implies Q, and P is true, then Q is true",
            InferenceRule.MODUS_TOLLENS: "If P implies Q, and Q is false, then P is false",
            InferenceRule.HYPOTHETICAL_SYLLOGISM: "If P implies Q, and Q implies R, then P implies R",
            InferenceRule.DISJUNCTIVE_SYLLOGISM: "If P or Q, and not P, then Q",
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Apply the logical rule: {rule_descriptions.get(rule, rule.value)}"),
            ("user", """Given statements:
{statements}

What conclusion can be derived? Provide just the conclusion.""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        statements_text = "\n".join([f"- {s}" for s in statements])
        conclusion = chain.invoke({"statements": statements_text})
        
        return conclusion.strip()


def demonstrate_deductive_reasoning():
    """Demonstrate deductive reasoning pattern"""
    print("=" * 80)
    print("DEDUCTIVE REASONING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = DeductiveReasoningAgent()
    
    # Example 1: Classic syllogism
    print("\n" + "="*80)
    print("EXAMPLE 1: Classic Syllogism (Socrates)")
    print("="*80)
    premises = [
        Premise("All humans are mortal", is_axiom=True),
        Premise("Socrates is a human", is_axiom=True)
    ]
    goal = "Socrates is mortal"
    
    proof = agent.prove(premises, goal, max_steps=5)
    
    print("Premises:")
    for p in proof.premises:
        print(f"  - {p.statement}")
    
    print(f"\nGoal: {goal}")
    print(f"\nProof steps:")
    for i, step in enumerate(proof.steps, 1):
        print(f"  {i}. {step.conclusion}")
        print(f"     Rule: {step.rule_applied}")
    
    print(f"\nConclusion: {proof.final_conclusion}")
    print(f"Valid: {proof.is_valid}")
    
    # Example 2: Modus Ponens
    print("\n" + "="*80)
    print("EXAMPLE 2: Modus Ponens")
    print("="*80)
    statements = [
        "If it rains, the ground gets wet",
        "It is raining"
    ]
    conclusion = agent.apply_rule(InferenceRule.MODUS_PONENS, statements)
    print("Given:")
    for s in statements:
        print(f"  - {s}")
    print(f"\nConclusion: {conclusion}")
    
    # Example 3: Mathematical reasoning
    print("\n" + "="*80)
    print("EXAMPLE 3: Mathematical Reasoning")
    print("="*80)
    premises = [
        Premise("If x > 5, then x > 3", is_axiom=True),
        Premise("x = 7", is_axiom=True),
        Premise("7 > 5", is_axiom=True)
    ]
    goal = "x > 3"
    
    proof = agent.prove(premises, goal, max_steps=5)
    
    print("Premises:")
    for p in proof.premises:
        print(f"  - {p.statement}")
    
    print(f"\nReasoning:")
    for trace in proof.reasoning_trace:
        if "Step" in trace or "Goal" in trace:
            print(f"  {trace}")
    
    print(f"\nFinal: {proof.final_conclusion}")
    
    # Example 4: Logical chain
    print("\n" + "="*80)
    print("EXAMPLE 4: Hypothetical Syllogism")
    print("="*80)
    statements = [
        "If it rains, the roads are wet",
        "If the roads are wet, driving is dangerous"
    ]
    conclusion = agent.apply_rule(InferenceRule.HYPOTHETICAL_SYLLOGISM, statements)
    print("Given:")
    for s in statements:
        print(f"  - {s}")
    print(f"\nConclusion: {conclusion}")
    
    # Example 5: Complex proof
    print("\n" + "="*80)
    print("EXAMPLE 5: Multi-Step Proof")
    print("="*80)
    premises = [
        Premise("All birds have feathers", is_axiom=True),
        Premise("All animals with feathers can regulate body temperature", is_axiom=True),
        Premise("A penguin is a bird", is_axiom=True)
    ]
    goal = "A penguin can regulate body temperature"
    
    proof = agent.prove(premises, goal, max_steps=10)
    
    print("Premises:")
    for p in proof.premises:
        print(f"  - {p.statement}")
    
    print(f"\nGoal: {goal}")
    print(f"\nProof trace:")
    for trace in proof.reasoning_trace[-5:]:
        print(f"  {trace}")
    
    # Example 6: Modus Tollens
    print("\n" + "="*80)
    print("EXAMPLE 6: Modus Tollens (Proof by Contrapositive)")
    print("="*80)
    statements = [
        "If the alarm is working, it would have gone off",
        "The alarm did not go off"
    ]
    conclusion = agent.apply_rule(InferenceRule.MODUS_TOLLENS, statements)
    print("Given:")
    for s in statements:
        print(f"  - {s}")
    print(f"\nConclusion: {conclusion}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Deductive Reasoning Best Practices")
    print("="*80)
    print("""
1. PREMISE VALIDATION:
   - Verify premises are true
   - Identify axioms vs derived facts
   - Check premise consistency
   - Document premise sources

2. LOGICAL RULES:
   - Use established inference rules
   - Apply rules systematically
   - Maintain logical validity
   - Document rule application

3. PROOF CONSTRUCTION:
   - Build step-by-step chains
   - Connect premises to conclusion
   - Verify each step
   - Keep proofs as simple as possible

4. COMMON RULES:
   - Modus Ponens: P→Q, P ⊢ Q
   - Modus Tollens: P→Q, ¬Q ⊢ ¬P
   - Hypothetical Syllogism: P→Q, Q→R ⊢ P→R
   - Disjunctive Syllogism: P∨Q, ¬P ⊢ Q
   - Conjunction: P, Q ⊢ P∧Q

5. VALIDATION:
   - Check logical soundness
   - Verify conclusion follows
   - Test for fallacies
   - Review proof structure

6. APPLICATIONS:
   - Mathematical proofs
   - Legal reasoning
   - Rule-based systems
   - Formal verification
   - Logical problem solving

Benefits:
✓ Guarantees valid conclusions
✓ Transparent reasoning process
✓ Systematic and rigorous
✓ Verifiable logic
✓ Foundation for formal systems

Limitations:
- Requires true premises
- Limited to formal logic
- May be inflexible
- Doesn't generate new knowledge
- Can be verbose for complex proofs
    """)


if __name__ == "__main__":
    demonstrate_deductive_reasoning()
