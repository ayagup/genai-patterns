"""
Pattern 058: Neuro-Symbolic Integration

Description:
    Neuro-Symbolic Integration combines neural (sub-symbolic, learned) approaches
    with symbolic (logical, rule-based) reasoning. This hybrid approach leverages
    the pattern recognition and generalization of neural methods with the
    interpretability, logical rigor, and structured reasoning of symbolic systems.

Components:
    1. Neural Module: Pattern recognition, learning from data
    2. Symbolic Module: Logical reasoning, rules, knowledge graphs
    3. Translation Layer: Converts between neural and symbolic representations
    4. Integration Strategy: Combines both approaches
    5. Verification Engine: Validates symbolic constraints

Use Cases:
    - Explainable AI requiring both learning and reasoning
    - Knowledge-grounded generation
    - Constraint-based problem solving with learning
    - Mathematical reasoning with verification
    - Legal/medical systems requiring compliance
    - Scientific reasoning with logical constraints

LangChain Implementation:
    Combines LLM-based neural reasoning with explicit symbolic rules
    and knowledge representation for hybrid problem-solving.
"""

import os
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ReasoningMode(Enum):
    """Integration modes"""
    NEURAL_FIRST = "neural_first"  # Neural then symbolic verification
    SYMBOLIC_FIRST = "symbolic_first"  # Symbolic constraints then neural
    INTERLEAVED = "interleaved"  # Alternate between both
    PARALLEL = "parallel"  # Both simultaneously, then merge
    HYBRID = "hybrid"  # Integrated throughout


class SymbolicConstraintType(Enum):
    """Types of symbolic constraints"""
    LOGICAL = "logical"  # Boolean logic rules
    MATHEMATICAL = "mathematical"  # Numerical constraints
    TEMPORAL = "temporal"  # Time-based rules
    ONTOLOGICAL = "ontological"  # Knowledge hierarchy
    PROCEDURAL = "procedural"  # Step-by-step rules


@dataclass
class SymbolicRule:
    """Symbolic rule or constraint"""
    rule_id: str
    rule_type: SymbolicConstraintType
    description: str
    condition: str  # Logical expression
    action: Optional[str] = None
    priority: int = 1
    
    def evaluate(self, facts: Dict[str, Any]) -> bool:
        """Evaluate rule against facts"""
        try:
            # Simple evaluation (in production, use proper logic engine)
            for key, value in facts.items():
                self.condition = self.condition.replace(key, str(value))
            
            # Basic evaluation
            if "AND" in self.condition:
                parts = self.condition.split("AND")
                return all(self._evaluate_simple(p.strip(), facts) for p in parts)
            elif "OR" in self.condition:
                parts = self.condition.split("OR")
                return any(self._evaluate_simple(p.strip(), facts) for p in parts)
            else:
                return self._evaluate_simple(self.condition, facts)
        except:
            return False
    
    def _evaluate_simple(self, expr: str, facts: Dict[str, Any]) -> bool:
        """Evaluate simple expression"""
        try:
            # Very simple evaluation
            if ">" in expr:
                left, right = expr.split(">")
                return float(left.strip()) > float(right.strip())
            elif "<" in expr:
                left, right = expr.split("<")
                return float(left.strip()) < float(right.strip())
            elif "==" in expr:
                left, right = expr.split("==")
                return left.strip() == right.strip()
            elif expr.strip().lower() in ["true", "1"]:
                return True
            elif expr.strip().lower() in ["false", "0"]:
                return False
            else:
                return bool(expr)
        except:
            return False


@dataclass
class KnowledgeGraph:
    """Simple knowledge graph"""
    entities: Set[str] = field(default_factory=set)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)  # (subject, predicate, object)
    facts: Dict[str, Any] = field(default_factory=dict)
    
    def add_entity(self, entity: str):
        self.entities.add(entity)
    
    def add_relation(self, subject: str, predicate: str, obj: str):
        self.relations.append((subject, predicate, obj))
        self.entities.add(subject)
        self.entities.add(obj)
    
    def query(self, pattern: Tuple[Optional[str], Optional[str], Optional[str]]) -> List[Tuple[str, str, str]]:
        """Query knowledge graph"""
        results = []
        for rel in self.relations:
            if all(
                p is None or p == r
                for p, r in zip(pattern, rel)
            ):
                results.append(rel)
        return results
    
    def to_text(self) -> str:
        """Convert to text representation"""
        text = "Knowledge Base:\n"
        for subj, pred, obj in self.relations:
            text += f"  {subj} {pred} {obj}\n"
        return text


@dataclass
class NeuroSymbolicResult:
    """Result from neuro-symbolic reasoning"""
    query: str
    neural_output: str
    symbolic_verification: Dict[str, Any]
    final_output: str
    constraints_satisfied: bool
    reasoning_trace: List[str]
    confidence: float
    execution_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:100] + "...",
            "constraints_satisfied": self.constraints_satisfied,
            "confidence": f"{self.confidence:.2f}",
            "reasoning_steps": len(self.reasoning_trace),
            "execution_time_ms": f"{self.execution_time_ms:.1f}"
        }


class NeuroSymbolicAgent:
    """
    Agent integrating neural and symbolic reasoning.
    
    Features:
    1. Neural reasoning via LLM
    2. Symbolic constraint checking
    3. Knowledge graph integration
    4. Logical verification
    5. Hybrid decision making
    """
    
    def __init__(
        self,
        reasoning_mode: ReasoningMode = ReasoningMode.NEURAL_FIRST,
        temperature: float = 0.3
    ):
        self.reasoning_mode = reasoning_mode
        self.temperature = temperature
        
        # Neural component (LLM)
        self.neural_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=temperature
        )
        
        # Symbolic components
        self.rules: List[SymbolicRule] = []
        self.knowledge_graph = KnowledgeGraph()
        
        # Reasoning trace
        self.reasoning_trace: List[str] = []
    
    def add_rule(self, rule: SymbolicRule):
        """Add symbolic rule"""
        self.rules.append(rule)
    
    def add_knowledge(
        self,
        subject: str,
        predicate: str,
        obj: str
    ):
        """Add knowledge to graph"""
        self.knowledge_graph.add_relation(subject, predicate, obj)
    
    def _neural_reasoning(self, query: str, context: Optional[str] = None) -> str:
        """Neural reasoning step"""
        
        self.reasoning_trace.append("Neural: Generating initial response")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are reasoning about a query.
Provide a thoughtful, logical response.

{context}"""),
            ("user", "{query}")
        ])
        
        context_text = context if context else "Use your knowledge to respond."
        
        chain = prompt | self.neural_model | StrOutputParser()
        response = chain.invoke({
            "query": query,
            "context": context_text
        })
        
        return response
    
    def _symbolic_verification(
        self,
        neural_output: str,
        extracted_facts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify neural output against symbolic constraints"""
        
        self.reasoning_trace.append("Symbolic: Verifying constraints")
        
        verification = {
            "all_satisfied": True,
            "violations": [],
            "satisfied_rules": [],
            "extracted_facts": extracted_facts
        }
        
        # Check each rule
        for rule in self.rules:
            satisfied = rule.evaluate(extracted_facts)
            
            if satisfied:
                verification["satisfied_rules"].append(rule.rule_id)
                self.reasoning_trace.append(f"  ✓ Rule {rule.rule_id}: {rule.description}")
            else:
                verification["all_satisfied"] = False
                verification["violations"].append({
                    "rule_id": rule.rule_id,
                    "description": rule.description,
                    "condition": rule.condition
                })
                self.reasoning_trace.append(f"  ✗ Rule {rule.rule_id}: {rule.description}")
        
        return verification
    
    def _extract_facts(self, text: str) -> Dict[str, Any]:
        """Extract facts from neural output"""
        
        self.reasoning_trace.append("Extracting facts from neural output")
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract key facts and values from the text.

Format as key-value pairs:
key1: value1
key2: value2"""),
            ("user", "Text: {text}\n\nExtracted facts:")
        ])
        
        chain = extraction_prompt | self.neural_model | StrOutputParser()
        facts_text = chain.invoke({"text": text})
        
        # Parse facts
        facts = {}
        for line in facts_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().replace(' ', '_').lower()
                value = value.strip()
                
                # Try to convert to number
                try:
                    value = float(value)
                except:
                    pass
                
                facts[key] = value
        
        return facts
    
    def _refine_with_constraints(
        self,
        query: str,
        neural_output: str,
        violations: List[Dict[str, Any]]
    ) -> str:
        """Refine neural output to satisfy constraints"""
        
        self.reasoning_trace.append("Neural: Refining to satisfy constraints")
        
        violations_text = "\n".join([
            f"- {v['description']}: {v['condition']}"
            for v in violations
        ])
        
        refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", """Your previous response violated some constraints.
Revise your response to satisfy these constraints:

{violations}

Maintain the core meaning but adjust to meet all constraints."""),
            ("user", """Original query: {query}

Previous response: {response}

Revised response:""")
        ])
        
        chain = refinement_prompt | self.neural_model | StrOutputParser()
        refined = chain.invoke({
            "query": query,
            "response": neural_output,
            "violations": violations_text
        })
        
        return refined
    
    def _integrate_knowledge(self, query: str) -> str:
        """Integrate knowledge graph into reasoning"""
        
        # Query knowledge graph for relevant facts
        # (simplified - in production, use semantic matching)
        relevant_facts = []
        
        query_lower = query.lower()
        for subject, predicate, obj in self.knowledge_graph.relations:
            if (subject.lower() in query_lower or 
                obj.lower() in query_lower):
                relevant_facts.append(f"{subject} {predicate} {obj}")
        
        if relevant_facts:
            context = "Relevant knowledge:\n" + "\n".join(relevant_facts[:5])
            self.reasoning_trace.append(f"Retrieved {len(relevant_facts)} relevant facts")
            return context
        
        return ""
    
    def reason(
        self,
        query: str,
        max_refinement_iterations: int = 3
    ) -> NeuroSymbolicResult:
        """Perform neuro-symbolic reasoning"""
        
        start_time = time.time()
        self.reasoning_trace = []
        
        if self.reasoning_mode == ReasoningMode.NEURAL_FIRST:
            # Neural reasoning first, then symbolic verification
            
            # 1. Integrate knowledge
            context = self._integrate_knowledge(query)
            
            # 2. Neural reasoning
            neural_output = self._neural_reasoning(query, context)
            
            # 3. Extract facts
            facts = self._extract_facts(neural_output)
            
            # 4. Symbolic verification
            verification = self._symbolic_verification(neural_output, facts)
            
            # 5. Refine if needed
            final_output = neural_output
            iteration = 0
            
            while not verification["all_satisfied"] and iteration < max_refinement_iterations:
                iteration += 1
                self.reasoning_trace.append(f"Refinement iteration {iteration}")
                
                final_output = self._refine_with_constraints(
                    query,
                    final_output,
                    verification["violations"]
                )
                
                # Re-verify
                facts = self._extract_facts(final_output)
                verification = self._symbolic_verification(final_output, facts)
                
                if verification["all_satisfied"]:
                    break
            
            constraints_satisfied = verification["all_satisfied"]
            confidence = 0.9 if constraints_satisfied else 0.6
            
        elif self.reasoning_mode == ReasoningMode.SYMBOLIC_FIRST:
            # Symbolic constraints first, then neural generation
            
            # 1. Gather symbolic constraints
            constraints_text = "\n".join([
                f"- {rule.description} ({rule.condition})"
                for rule in self.rules
            ])
            
            # 2. Neural generation with constraints
            context = f"You must satisfy these constraints:\n{constraints_text}\n\n"
            context += self._integrate_knowledge(query)
            
            neural_output = self._neural_reasoning(query, context)
            
            # 3. Verify
            facts = self._extract_facts(neural_output)
            verification = self._symbolic_verification(neural_output, facts)
            
            final_output = neural_output
            constraints_satisfied = verification["all_satisfied"]
            confidence = 0.85 if constraints_satisfied else 0.5
            
        else:  # HYBRID or default
            # Integrated approach
            context = self._integrate_knowledge(query)
            neural_output = self._neural_reasoning(query, context)
            facts = self._extract_facts(neural_output)
            verification = self._symbolic_verification(neural_output, facts)
            
            final_output = neural_output
            constraints_satisfied = verification["all_satisfied"]
            confidence = 0.8 if constraints_satisfied else 0.6
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return NeuroSymbolicResult(
            query=query,
            neural_output=neural_output,
            symbolic_verification=verification,
            final_output=final_output,
            constraints_satisfied=constraints_satisfied,
            reasoning_trace=self.reasoning_trace,
            confidence=confidence,
            execution_time_ms=execution_time_ms
        )


def demonstrate_neuro_symbolic():
    """Demonstrate Neuro-Symbolic Integration pattern"""
    
    print("=" * 80)
    print("PATTERN 058: NEURO-SYMBOLIC INTEGRATION DEMONSTRATION")
    print("=" * 80)
    print("\nCombining neural learning with symbolic reasoning\n")
    
    # Test 1: Neural-first with constraints
    print("\n" + "=" * 80)
    print("TEST 1: Neural-First with Constraint Verification")
    print("=" * 80)
    
    agent1 = NeuroSymbolicAgent(
        reasoning_mode=ReasoningMode.NEURAL_FIRST,
        temperature=0.3
    )
    
    # Add symbolic rules
    agent1.add_rule(SymbolicRule(
        rule_id="positive_value",
        rule_type=SymbolicConstraintType.MATHEMATICAL,
        description="Result must be positive",
        condition="result > 0",
        priority=1
    ))
    
    agent1.add_rule(SymbolicRule(
        rule_id="reasonable_range",
        rule_type=SymbolicConstraintType.MATHEMATICAL,
        description="Result must be less than 1000",
        condition="result < 1000",
        priority=1
    ))
    
    query1 = "What is 25 multiplied by 8?"
    
    print(f"\n💭 Query: {query1}")
    print(f"🔧 Constraints:")
    for rule in agent1.rules:
        print(f"   - {rule.description}")
    
    result1 = agent1.reason(query1)
    
    print(f"\n🧠 Neural Output: {result1.neural_output[:150]}...")
    print(f"\n⚙️  Symbolic Verification:")
    print(f"   Constraints Satisfied: {result1.constraints_satisfied}")
    
    if result1.symbolic_verification["violations"]:
        print(f"   Violations:")
        for v in result1.symbolic_verification["violations"]:
            print(f"      - {v['description']}")
    
    if result1.symbolic_verification["satisfied_rules"]:
        print(f"   Satisfied Rules: {', '.join(result1.symbolic_verification['satisfied_rules'])}")
    
    print(f"\n✅ Final Output: {result1.final_output[:200]}...")
    print(f"   Confidence: {result1.confidence:.2f}")
    
    # Test 2: Knowledge graph integration
    print("\n" + "=" * 80)
    print("TEST 2: Knowledge Graph Integration")
    print("=" * 80)
    
    agent2 = NeuroSymbolicAgent(
        reasoning_mode=ReasoningMode.HYBRID
    )
    
    # Build knowledge graph
    agent2.add_knowledge("Python", "is_a", "programming_language")
    agent2.add_knowledge("Python", "created_by", "Guido_van_Rossum")
    agent2.add_knowledge("Python", "released_in", "1991")
    agent2.add_knowledge("Python", "used_for", "machine_learning")
    agent2.add_knowledge("Python", "used_for", "web_development")
    
    print(f"\n📚 Knowledge Graph:")
    for subj, pred, obj in agent2.knowledge_graph.relations:
        print(f"   {subj} {pred} {obj}")
    
    query2 = "When was Python created and what is it used for?"
    
    print(f"\n💭 Query: {query2}")
    
    result2 = agent2.reason(query2)
    
    print(f"\n🔍 Reasoning Trace:")
    for i, step in enumerate(result2.reasoning_trace[:5], 1):
        print(f"   {i}. {step}")
    
    print(f"\n💬 Response: {result2.final_output[:250]}...")
    
    # Test 3: Constraint refinement
    print("\n" + "=" * 80)
    print("TEST 3: Iterative Constraint Refinement")
    print("=" * 80)
    
    agent3 = NeuroSymbolicAgent(
        reasoning_mode=ReasoningMode.NEURAL_FIRST
    )
    
    # Add logical constraint
    agent3.add_rule(SymbolicRule(
        rule_id="format_check",
        rule_type=SymbolicConstraintType.LOGICAL,
        description="Response must include steps",
        condition="steps",  # Simplified
        priority=2
    ))
    
    query3 = "How do I solve a quadratic equation?"
    
    print(f"\n💭 Query: {query3}")
    
    result3 = agent3.reason(query3, max_refinement_iterations=2)
    
    print(f"\n🔄 Refinement Process:")
    print(f"   Total Steps: {len(result3.reasoning_trace)}")
    print(f"   Constraints Satisfied: {result3.constraints_satisfied}")
    
    print(f"\n📝 Reasoning Trace:")
    for step in result3.reasoning_trace:
        print(f"   - {step}")
    
    print(f"\n✅ Final Answer: {result3.final_output[:200]}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("NEURO-SYMBOLIC INTEGRATION PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Best of Both: Neural flexibility + symbolic rigor
2. Explainability: Symbolic rules provide interpretability
3. Constraint Satisfaction: Guaranteed compliance
4. Knowledge Integration: Use structured knowledge
5. Verification: Validate neural outputs

Integration Modes:
1. Neural-First: Generate then verify
   - Fast, flexible generation
   - Symbolic post-processing

2. Symbolic-First: Constrain then generate
   - Guaranteed satisfaction
   - May limit creativity

3. Interleaved: Alternate steps
   - Balanced approach
   - More iterations

4. Parallel: Both simultaneously
   - Independent processing
   - Merge results

5. Hybrid: Fully integrated
   - Seamless combination
   - Complex implementation

Components:
1. Neural Module: LLM for pattern recognition
2. Symbolic Module: Rules, logic, knowledge graph
3. Translation Layer: Convert representations
4. Verification Engine: Check constraints
5. Refinement Loop: Iterative improvement

Symbolic Constraints:
- Logical: Boolean rules (AND, OR, NOT)
- Mathematical: Numerical constraints
- Temporal: Time-based rules
- Ontological: Knowledge hierarchies
- Procedural: Step sequences

Knowledge Representation:
- Knowledge Graphs: Entity-relation-entity
- Rules: If-then logic
- Ontologies: Hierarchical concepts
- Constraints: Mathematical inequalities
- Facts: Ground truth statements

Use Cases:
- Explainable AI systems
- Compliance-critical applications
- Mathematical reasoning
- Legal/medical systems
- Scientific hypothesis generation
- Code generation with verification

Best Practices:
1. Clear symbolic constraints
2. Efficient fact extraction
3. Iterative refinement limits
4. Knowledge graph maintenance
5. Rule priority ordering
6. Verification performance
7. Fallback strategies

Production Considerations:
- Rule engine scalability
- Knowledge graph indexing
- Refinement iteration limits
- Caching verified results
- Parallel verification
- Constraint complexity
- Performance optimization

Comparison with Related Patterns:
- vs. Pure Neural: Adds verification
- vs. Pure Symbolic: Adds flexibility
- vs. Retrieval: Structured knowledge vs search
- vs. Chain-of-Thought: Formal logic vs reasoning

Challenges:
1. Representation gap: Neural ↔ symbolic translation
2. Scalability: Large rule sets
3. Conflict resolution: Contradictory rules
4. Learning rules: Manual vs automated
5. Integration complexity

The Neuro-Symbolic Integration pattern enables robust AI systems
that combine the pattern recognition power of neural networks
with the interpretability and logical rigor of symbolic reasoning.
""")


if __name__ == "__main__":
    demonstrate_neuro_symbolic()
