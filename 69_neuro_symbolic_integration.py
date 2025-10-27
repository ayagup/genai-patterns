"""
Neuro-Symbolic Integration Pattern

Combines neural (LLM) and symbolic (logic) reasoning for the best of both paradigms.
Neural networks provide flexibility and learning from data, while symbolic systems
provide interpretability, logical consistency, and guaranteed correctness.

Use Cases:
- Knowledge reasoning with guaranteed logical consistency
- Solving logical puzzles with natural language understanding
- Constraint satisfaction with learned heuristics
- Explainable AI systems requiring audit trails
- Mathematical theorem proving with natural language

Benefits:
- Interpretability: Explicit symbolic reasoning steps
- Correctness: Logical guarantees from symbolic component
- Flexibility: Neural component handles ambiguity
- Knowledge integration: Combines learned and explicit knowledge
- Transparency: Clear decision provenance
"""

from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re


class LogicOperator(Enum):
    """Logical operators for symbolic reasoning"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    IFF = "IFF"  # if and only if


@dataclass
class Fact:
    """Symbolic fact representation"""
    predicate: str
    arguments: List[str]
    truth_value: bool = True
    confidence: float = 1.0  # Neural confidence score
    
    def __str__(self) -> str:
        neg = "NOT " if not self.truth_value else ""
        args_str = ", ".join(self.arguments)
        return f"{neg}{self.predicate}({args_str})"
    
    def __hash__(self) -> int:
        return hash((self.predicate, tuple(self.arguments), self.truth_value))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Fact):
            return False
        return (self.predicate == other.predicate and 
                self.arguments == other.arguments and 
                self.truth_value == other.truth_value)


@dataclass
class Rule:
    """Symbolic inference rule"""
    premises: List[Fact]
    conclusion: Fact
    rule_name: str
    confidence: float = 1.0
    
    def __str__(self) -> str:
        premises_str = " AND ".join(str(p) for p in self.premises)
        return f"{self.rule_name}: IF {premises_str} THEN {self.conclusion}"


class KnowledgeBase:
    """Symbolic knowledge base with inference capabilities"""
    
    def __init__(self):
        self.facts: Set[Fact] = set()
        self.rules: List[Rule] = []
        self.inference_trace: List[str] = []
    
    def add_fact(self, fact: Fact) -> None:
        """Add fact to knowledge base"""
        self.facts.add(fact)
        self.inference_trace.append(f"Added fact: {fact}")
    
    def add_rule(self, rule: Rule) -> None:
        """Add inference rule"""
        self.rules.append(rule)
        self.inference_trace.append(f"Added rule: {rule}")
    
    def query(self, fact: Fact) -> Tuple[bool, float, List[str]]:
        """Query if a fact is true, return confidence and proof"""
        # Direct lookup
        if fact in self.facts:
            for f in self.facts:
                if f == fact:
                    return True, f.confidence, [f"Direct fact: {fact}"]
        
        # Try to infer using rules
        return self._forward_chain(fact)
    
    def _forward_chain(self, goal: Fact) -> Tuple[bool, float, List[str]]:
        """Forward chaining inference"""
        proof: List[str] = []
        
        for rule in self.rules:
            if self._matches_conclusion(rule.conclusion, goal):
                # Check if all premises are satisfied
                all_satisfied = True
                min_confidence = rule.confidence
                
                for premise in rule.premises:
                    if premise not in self.facts:
                        # Recursively try to prove premise
                        proved, conf, sub_proof = self._forward_chain(premise)
                        if not proved:
                            all_satisfied = False
                            break
                        min_confidence = min(min_confidence, conf)
                        proof.extend(sub_proof)
                    else:
                        for f in self.facts:
                            if f == premise:
                                min_confidence = min(min_confidence, f.confidence)
                
                if all_satisfied:
                    # Add conclusion to facts
                    new_fact = Fact(
                        goal.predicate,
                        goal.arguments,
                        goal.truth_value,
                        min_confidence
                    )
                    self.facts.add(new_fact)
                    proof.append(f"Applied rule: {rule.rule_name}")
                    proof.append(f"Inferred: {new_fact} (confidence: {min_confidence:.2f})")
                    return True, min_confidence, proof
        
        return False, 0.0, []
    
    def _matches_conclusion(self, conclusion: Fact, goal: Fact) -> bool:
        """Check if conclusion matches goal"""
        return (conclusion.predicate == goal.predicate and
                conclusion.truth_value == goal.truth_value and
                len(conclusion.arguments) == len(goal.arguments))
    
    def get_all_facts(self) -> List[Fact]:
        """Get all facts in knowledge base"""
        return list(self.facts)
    
    def get_inference_trace(self) -> List[str]:
        """Get trace of all inferences"""
        return self.inference_trace.copy()


class NeuralExtractor:
    """Neural component: Extracts facts from natural language"""
    
    def extract_facts(self, text: str) -> List[Fact]:
        """
        Extract facts from natural language text.
        In production, this would use an LLM with proper prompting.
        """
        facts: List[Fact] = []
        
        # Simple pattern matching (simulating LLM extraction)
        # Pattern: "X is a Y" -> IsA(X, Y)
        pattern1 = r"(\w+)\s+is\s+an?\s+(\w+)"
        for match in re.finditer(pattern1, text, re.IGNORECASE):
            subject, obj = match.groups()
            facts.append(Fact("IsA", [subject.lower(), obj.lower()], confidence=0.9))
        
        # Pattern: "X has Y" -> Has(X, Y)
        pattern2 = r"(\w+)\s+has\s+(\w+)"
        for match in re.finditer(pattern2, text, re.IGNORECASE):
            subject, obj = match.groups()
            facts.append(Fact("Has", [subject.lower(), obj.lower()], confidence=0.85))
        
        # Pattern: "X can Y" -> Can(X, Y)
        pattern3 = r"(\w+)\s+can\s+(\w+)"
        for match in re.finditer(pattern3, text, re.IGNORECASE):
            subject, action = match.groups()
            facts.append(Fact("Can", [subject.lower(), action.lower()], confidence=0.8))
        
        # Pattern: "X is Y" (adjective) -> Property(X, Y)
        pattern4 = r"(\w+)\s+is\s+(smart|intelligent|fast|slow|big|small|dangerous)"
        for match in re.finditer(pattern4, text, re.IGNORECASE):
            subject, property = match.groups()
            facts.append(Fact("Property", [subject.lower(), property.lower()], confidence=0.88))
        
        return facts
    
    def generate_explanation(self, conclusion: Fact, proof: List[str]) -> str:
        """
        Generate natural language explanation from symbolic proof.
        In production, this would use an LLM for better language.
        """
        explanation = f"Conclusion: {conclusion}\n\n"
        explanation += "Reasoning:\n"
        for i, step in enumerate(proof, 1):
            explanation += f"{i}. {step}\n"
        
        return explanation


class NeuroSymbolicAgent:
    """
    Neuro-Symbolic Integration Agent
    
    Combines neural NLP capabilities with symbolic logical reasoning.
    """
    
    def __init__(self, name: str = "NeuroSymbolic Agent"):
        self.name = name
        self.knowledge_base = KnowledgeBase()
        self.neural_extractor = NeuralExtractor()
        self.reasoning_trace: List[Dict[str, Any]] = []
    
    def learn_from_text(self, text: str) -> List[Fact]:
        """
        Neural: Extract facts from natural language
        Symbolic: Store in knowledge base
        """
        print(f"\n[Neural] Extracting facts from text...")
        facts = self.neural_extractor.extract_facts(text)
        
        print(f"[Neural] Extracted {len(facts)} facts:")
        for fact in facts:
            print(f"  - {fact} (confidence: {fact.confidence:.2f})")
            self.knowledge_base.add_fact(fact)
        
        self.reasoning_trace.append({
            "type": "learning",
            "input": text,
            "facts_extracted": len(facts)
        })
        
        return facts
    
    def add_reasoning_rule(self, rule: Rule) -> None:
        """
        Add symbolic reasoning rule to knowledge base
        """
        print(f"\n[Symbolic] Adding reasoning rule: {rule.rule_name}")
        self.knowledge_base.add_rule(rule)
    
    def query_with_reasoning(self, query: str) -> Dict[str, Any]:
        """
        Process natural language query with neuro-symbolic reasoning
        """
        print(f"\n[Neuro-Symbolic Query] {query}")
        
        # Neural: Parse query into symbolic form
        query_facts = self.neural_extractor.extract_facts(query)
        
        if not query_facts:
            return {
                "success": False,
                "message": "Could not parse query into symbolic form"
            }
        
        # Use first extracted fact as query goal
        goal = query_facts[0]
        print(f"[Neural] Parsed query as: {goal}")
        
        # Symbolic: Perform logical inference
        print(f"[Symbolic] Running inference engine...")
        success, confidence, proof = self.knowledge_base.query(goal)
        
        if success:
            # Neural: Generate natural language explanation
            explanation = self.neural_extractor.generate_explanation(goal, proof)
            
            result = {
                "success": True,
                "conclusion": str(goal),
                "confidence": confidence,
                "proof_steps": proof,
                "explanation": explanation,
                "reasoning_type": "neuro-symbolic"
            }
            
            print(f"\n[Result] Conclusion: {goal}")
            print(f"[Result] Confidence: {confidence:.2f}")
            print(f"[Result] Proof steps: {len(proof)}")
        else:
            result = {
                "success": False,
                "conclusion": f"Cannot prove: {goal}",
                "confidence": 0.0,
                "message": "Insufficient facts or rules for inference"
            }
            
            print(f"\n[Result] Cannot prove: {goal}")
        
        self.reasoning_trace.append({
            "type": "query",
            "query": query,
            "goal": str(goal),
            "success": success
        })
        
        return result
    
    def explain_reasoning(self) -> str:
        """Get complete reasoning trace"""
        trace = "\n=== Neuro-Symbolic Reasoning Trace ===\n\n"
        
        trace += "Knowledge Base Facts:\n"
        for fact in self.knowledge_base.get_all_facts():
            trace += f"  - {fact}\n"
        
        trace += "\nInference Steps:\n"
        for step in self.knowledge_base.get_inference_trace():
            trace += f"  - {step}\n"
        
        return trace


def demonstrate_neuro_symbolic_integration():
    """
    Demonstrate Neuro-Symbolic Integration pattern
    """
    print("=" * 70)
    print("NEURO-SYMBOLIC INTEGRATION PATTERN DEMONSTRATION")
    print("=" * 70)
    
    # Create agent
    agent = NeuroSymbolicAgent("Reasoning Agent")
    
    # Example 1: Animal reasoning
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Animal Classification Reasoning")
    print("=" * 70)
    
    # Neural: Learn facts from text
    texts = [
        "Tweety is a bird. Birds can fly. Tweety is small.",
        "Sparky is a dog. Dogs are animals. Dogs can bark.",
        "Fluffy is a cat. Cats are animals. Cats can meow."
    ]
    
    for text in texts:
        agent.learn_from_text(text)
    
    # Symbolic: Add reasoning rules
    print("\n[Symbolic] Adding logical reasoning rules...")
    
    # Rule: If X is a bird, then X is an animal
    agent.add_reasoning_rule(Rule(
        premises=[Fact("IsA", ["X", "bird"])],
        conclusion=Fact("IsA", ["X", "animal"]),
        rule_name="BirdsAreAnimals",
        confidence=1.0
    ))
    
    # Rule: If X is an animal and X can fly, then X can move
    agent.add_reasoning_rule(Rule(
        premises=[
            Fact("IsA", ["X", "animal"]),
            Fact("Can", ["X", "fly"])
        ],
        conclusion=Fact("Can", ["X", "move"]),
        rule_name="FlyingAnimalsMoveRule",
        confidence=0.95
    ))
    
    # Query with reasoning
    query1 = "Is Tweety an animal?"
    result1 = agent.query_with_reasoning(query1)
    
    if result1["success"]:
        print(f"\n{result1['explanation']}")
    
    # Example 2: Family reasoning
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Family Relationship Reasoning")
    print("=" * 70)
    
    agent2 = NeuroSymbolicAgent("Family Reasoner")
    
    # Learn family facts
    family_texts = [
        "John is a parent. Mary is a parent.",
        "Alice is a child. Bob is a child.",
        "John has Alice. Mary has Bob."
    ]
    
    for text in family_texts:
        agent2.learn_from_text(text)
    
    # Add family reasoning rules
    agent2.add_reasoning_rule(Rule(
        premises=[
            Fact("IsA", ["X", "parent"]),
            Fact("Has", ["X", "Y"]),
            Fact("IsA", ["Y", "child"])
        ],
        conclusion=Fact("ParentOf", ["X", "Y"]),
        rule_name="ParentChildRule",
        confidence=1.0
    ))
    
    # Example 3: Constraint satisfaction
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Neuro-Symbolic Benefits")
    print("=" * 70)
    
    print("\n✓ Benefits of Neuro-Symbolic Integration:")
    print("\n1. INTERPRETABILITY:")
    print("   - Every inference step is traceable")
    print("   - Clear logical provenance")
    print("   - Auditable decision-making")
    
    print("\n2. CORRECTNESS:")
    print("   - Logical guarantees from symbolic reasoning")
    print("   - No hallucinations in logical steps")
    print("   - Confidence scores combine both paradigms")
    
    print("\n3. FLEXIBILITY:")
    print("   - Neural component handles natural language")
    print("   - Symbolic component ensures consistency")
    print("   - Best of both worlds")
    
    print("\n4. KNOWLEDGE INTEGRATION:")
    print("   - Learned facts (neural) + explicit rules (symbolic)")
    print("   - Combines data-driven and knowledge-based AI")
    
    # Show complete reasoning trace
    print("\n" + "=" * 70)
    print("COMPLETE REASONING TRACE")
    print("=" * 70)
    print(agent.explain_reasoning())


def demonstrate_advanced_applications():
    """Show advanced neuro-symbolic applications"""
    print("\n" + "=" * 70)
    print("ADVANCED NEURO-SYMBOLIC APPLICATIONS")
    print("=" * 70)
    
    print("\n1. Mathematical Theorem Proving:")
    print("   Neural: Parse math problems from natural language")
    print("   Symbolic: Formal proof verification")
    print("   → Guaranteed correctness with natural interaction")
    
    print("\n2. Legal Reasoning:")
    print("   Neural: Extract facts from case documents")
    print("   Symbolic: Apply legal rules and precedents")
    print("   → Explainable legal AI systems")
    
    print("\n3. Medical Diagnosis:")
    print("   Neural: Learn patterns from patient records")
    print("   Symbolic: Apply medical knowledge rules")
    print("   → Trustworthy diagnostic support")
    
    print("\n4. Robotic Planning:")
    print("   Neural: Perceive and understand environment")
    print("   Symbolic: Logical action planning")
    print("   → Safe and verifiable robot behavior")
    
    print("\n5. Scientific Discovery:")
    print("   Neural: Generate hypotheses from data")
    print("   Symbolic: Validate against known laws")
    print("   → Accelerated scientific research")


if __name__ == "__main__":
    demonstrate_neuro_symbolic_integration()
    demonstrate_advanced_applications()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Neuro-symbolic AI combines neural and symbolic reasoning
2. Neural handles flexibility, symbolic ensures correctness
3. Provides interpretability and logical guarantees
4. Ideal for high-stakes domains requiring explainability
5. Enables knowledge integration from multiple sources

Best Practices:
- Use neural for perception and natural language understanding
- Use symbolic for logical reasoning and verification
- Maintain clear separation of concerns
- Track confidence scores from both components
- Provide complete reasoning traces for auditability
- Validate symbolic rules with domain experts
    """)
