"""
Pattern 063: Causal Reasoning Agent

Description:
    Causal Reasoning enables agents to understand cause-effect relationships,
    perform interventions, and reason about counterfactuals. Unlike correlation-based
    reasoning, causal reasoning distinguishes between "what happened" and "what would
    happen if we changed something", enabling better decision-making and explanation.

Components:
    1. Causal Graph: Represents causal relationships between variables
    2. Structural Equations: Model mechanisms of causation
    3. Intervention Handler: Simulates "do" operations
    4. Counterfactual Reasoner: Answers "what if" questions
    5. Causal Discovery: Learns causal structure from data
    6. Effect Estimation: Quantifies causal effects

Use Cases:
    - Root cause analysis
    - Decision-making under uncertainty
    - Policy evaluation ("what if we change X?")
    - Scientific reasoning and hypothesis testing
    - Fairness and bias analysis
    - Explanation generation

LangChain Implementation:
    Implements causal reasoning using LLM-based causal graph construction,
    intervention simulation, and counterfactual query answering.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class CausalRelationType(Enum):
    """Types of causal relationships"""
    DIRECT = "direct"  # A → B
    INDIRECT = "indirect"  # A → C → B
    COMMON_CAUSE = "common_cause"  # A ← C → B
    COLLIDER = "collider"  # A → C ← B
    BIDIRECTIONAL = "bidirectional"  # A ↔ B


@dataclass
class CausalEdge:
    """Edge in causal graph"""
    cause: str
    effect: str
    strength: float = 1.0  # 0.0 to 1.0
    mechanism: Optional[str] = None
    
    def __str__(self) -> str:
        arrow = "→" if self.strength > 0 else "⊣"
        strength_str = f"({self.strength:.2f})" if self.strength != 1.0 else ""
        return f"{self.cause} {arrow} {self.effect} {strength_str}"


@dataclass
class CausalGraph:
    """Causal graph structure"""
    variables: Set[str] = field(default_factory=set)
    edges: List[CausalEdge] = field(default_factory=list)
    
    def add_edge(self, cause: str, effect: str, strength: float = 1.0, mechanism: str = None):
        """Add causal edge"""
        self.variables.add(cause)
        self.variables.add(effect)
        self.edges.append(CausalEdge(cause, effect, strength, mechanism))
    
    def get_parents(self, variable: str) -> List[str]:
        """Get direct causes of variable"""
        return [edge.cause for edge in self.edges if edge.effect == variable]
    
    def get_children(self, variable: str) -> List[str]:
        """Get direct effects of variable"""
        return [edge.effect for edge in self.edges if edge.cause == variable]
    
    def get_ancestors(self, variable: str) -> Set[str]:
        """Get all ancestors (transitive causes)"""
        ancestors = set()
        to_visit = self.get_parents(variable)
        
        while to_visit:
            parent = to_visit.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                to_visit.extend(self.get_parents(parent))
        
        return ancestors
    
    def to_text(self) -> str:
        """Convert graph to text"""
        text = "Causal Graph:\n"
        text += f"Variables: {', '.join(sorted(self.variables))}\n\n"
        text += "Edges:\n"
        for edge in self.edges:
            text += f"  {edge}\n"
            if edge.mechanism:
                text += f"    Mechanism: {edge.mechanism}\n"
        return text


@dataclass
class Intervention:
    """Causal intervention (do-operator)"""
    variable: str
    value: Any
    description: str


@dataclass
class CounterfactualQuery:
    """Counterfactual question"""
    observed_facts: Dict[str, Any]
    intervention: Intervention
    query_variable: str
    
    def to_text(self) -> str:
        facts_str = ", ".join(f"{k}={v}" for k, v in self.observed_facts.items())
        return f"Given {facts_str}, what if we {self.intervention.description}? What would {self.query_variable} be?"


@dataclass
class CausalAnswer:
    """Answer to causal query"""
    query: str
    answer: str
    confidence: float
    reasoning: str
    affected_variables: List[str] = field(default_factory=list)


class CausalReasoningAgent:
    """
    Agent capable of causal reasoning.
    
    Features:
    1. Causal graph construction
    2. Intervention simulation
    3. Counterfactual reasoning
    4. Effect estimation
    5. Root cause analysis
    """
    
    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.causal_graph = CausalGraph()
        
        # LLM for causal discovery
        self.discovery_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # LLM for reasoning
        self.reasoning_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2
        )
    
    def build_causal_graph(self, context: str) -> CausalGraph:
        """Construct causal graph from context"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are identifying causal relationships in {self.domain}.

Analyze the context and identify:
1. Variables (factors that can change)
2. Causal relationships (A causes B)
3. Mechanisms (how A causes B)

Format your response as:
VARIABLES:
- variable1
- variable2
...

CAUSAL EDGES:
variable1 → variable2 (strength: 0.0-1.0)
Mechanism: explanation

..."""),
            ("user", "Context: {context}\n\nCausal Analysis:")
        ])
        
        chain = prompt | self.discovery_model | StrOutputParser()
        analysis = chain.invoke({"context": context})
        
        # Parse response
        graph = CausalGraph()
        lines = analysis.split('\n')
        
        section = None
        current_edge = None
        
        for line in lines:
            line = line.strip()
            
            if 'VARIABLES:' in line:
                section = 'variables'
            elif 'CAUSAL EDGES:' in line or 'EDGES:' in line:
                section = 'edges'
            elif section == 'variables' and line.startswith('-'):
                var = line.lstrip('- ').strip()
                if var:
                    graph.variables.add(var)
            elif section == 'edges' and ('→' in line or '->' in line):
                # Parse edge
                if '→' in line:
                    parts = line.split('→')
                elif '->' in line:
                    parts = line.split('->')
                else:
                    continue
                
                if len(parts) >= 2:
                    cause = parts[0].strip()
                    effect_part = parts[1].strip()
                    
                    # Extract effect and strength
                    effect = effect_part.split('(')[0].strip()
                    strength = 1.0
                    
                    if '(' in effect_part:
                        try:
                            strength_str = effect_part.split('(')[1].split(')')[0]
                            strength = float(''.join(c for c in strength_str if c.isdigit() or c == '.'))
                            if strength > 1.0:
                                strength = strength / 100.0
                        except:
                            pass
                    
                    current_edge = CausalEdge(cause, effect, strength)
                    graph.edges.append(current_edge)
                    graph.variables.add(cause)
                    graph.variables.add(effect)
            elif section == 'edges' and 'mechanism:' in line.lower() and current_edge:
                mechanism = line.split(':', 1)[1].strip()
                current_edge.mechanism = mechanism
        
        self.causal_graph = graph
        return graph
    
    def do_intervention(
        self,
        intervention: Intervention,
        context: str
    ) -> CausalAnswer:
        """Simulate causal intervention (do-operator)"""
        
        # Build/use causal graph
        if not self.causal_graph.edges:
            self.build_causal_graph(context)
        
        # Find affected variables
        affected = {intervention.variable}
        affected.update(self.causal_graph.get_children(intervention.variable))
        
        # Recursively add downstream effects
        to_check = list(self.causal_graph.get_children(intervention.variable))
        while to_check:
            var = to_check.pop()
            children = self.causal_graph.get_children(var)
            for child in children:
                if child not in affected:
                    affected.add(child)
                    to_check.append(child)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are performing causal intervention analysis in {self.domain}.

Causal Graph:
{self.causal_graph.to_text()}

The intervention sets {intervention.variable} to {intervention.value}.
This is a CAUSAL intervention (do-operator), not just observation.

Analyze:
1. What changes because of this intervention
2. Which variables are affected and how
3. What are the downstream effects

Provide clear reasoning about the causal chain."""),
            ("user", """Context: {context}

Intervention: {intervention}

What are the causal effects?""")
        ])
        
        chain = prompt | self.reasoning_model | StrOutputParser()
        analysis = chain.invoke({
            "context": context,
            "intervention": intervention.description
        })
        
        return CausalAnswer(
            query=f"do({intervention.variable}={intervention.value})",
            answer=analysis,
            confidence=0.7,
            reasoning="Based on causal graph structure and domain knowledge",
            affected_variables=list(affected)
        )
    
    def answer_counterfactual(
        self,
        query: CounterfactualQuery,
        context: str
    ) -> CausalAnswer:
        """Answer counterfactual question"""
        
        # Counterfactuals require:
        # 1. What actually happened (observed facts)
        # 2. What we change (intervention)
        # 3. What we query (counterfactual outcome)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are answering counterfactual questions in {self.domain}.

Counterfactual reasoning has three steps:
1. Abduction: Infer what must be true given observations
2. Action: Apply the intervention
3. Prediction: Determine outcome in this counterfactual world

Use the causal graph to reason about what would change."""),
            ("user", """Context: {context}

Observed Facts: {facts}

Counterfactual Intervention: {intervention}

Question: What would {query_var} be in this counterfactual scenario?

Answer:""")
        ])
        
        chain = prompt | self.reasoning_model | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "facts": str(query.observed_facts),
            "intervention": query.intervention.description,
            "query_var": query.query_variable
        })
        
        return CausalAnswer(
            query=query.to_text(),
            answer=answer,
            confidence=0.6,  # Counterfactuals are harder
            reasoning="Counterfactual reasoning via abduction-action-prediction"
        )
    
    def identify_root_cause(
        self,
        effect: str,
        context: str
    ) -> CausalAnswer:
        """Identify root cause of an effect"""
        
        if not self.causal_graph.edges:
            self.build_causal_graph(context)
        
        # Get all ancestors
        ancestors = self.causal_graph.get_ancestors(effect)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are performing root cause analysis in {self.domain}.

Causal Graph:
{self.causal_graph.to_text()}

Identify the ROOT CAUSE(S) of the observed effect.
Root causes are variables with:
1. No upstream causes (or only external causes)
2. Strong causal connection to the effect
3. Actionable or controllable

Trace back through the causal chain."""),
            ("user", """Context: {context}

Observed Effect: {effect}

Potential Causes: {ancestors}

What is the root cause?""")
        ])
        
        chain = prompt | self.reasoning_model | StrOutputParser()
        analysis = chain.invoke({
            "context": context,
            "effect": effect,
            "ancestors": ", ".join(ancestors) if ancestors else "None identified"
        })
        
        return CausalAnswer(
            query=f"Root cause of: {effect}",
            answer=analysis,
            confidence=0.75,
            reasoning="Traced causal chain to root variables"
        )


def demonstrate_causal_reasoning():
    """Demonstrate Causal Reasoning Agent pattern"""
    
    print("=" * 80)
    print("PATTERN 063: CAUSAL REASONING AGENT DEMONSTRATION")
    print("=" * 80)
    print("\nUnderstanding cause-effect relationships and counterfactuals\n")
    
    # Test 1: Building causal graph
    print("\n" + "=" * 80)
    print("TEST 1: Causal Graph Construction")
    print("=" * 80)
    
    agent = CausalReasoningAgent(domain="business")
    
    context1 = """
    A company has been experiencing declining sales. Analysis shows:
    - Marketing budget was reduced
    - Competitor launched new product
    - Customer satisfaction scores dropped
    - Website traffic decreased
    - Sales team morale is low
    The reduced marketing budget led to less advertising, which decreased website 
    traffic. The competitor's new product attracted customers, reducing market share.
    Lower sales affected team morale.
    """
    
    print(f"\n📊 Analyzing business scenario...")
    
    graph = agent.build_causal_graph(context1)
    
    print(f"\n🔗 Discovered Causal Graph:")
    print(f"   Variables: {len(graph.variables)}")
    print(f"   Edges: {len(graph.edges)}")
    print(f"\n{graph.to_text()}")
    
    # Test 2: Intervention analysis
    print("\n" + "=" * 80)
    print("TEST 2: Causal Intervention (do-operator)")
    print("=" * 80)
    
    intervention = Intervention(
        variable="marketing budget",
        value="increase by 50%",
        description="increase marketing budget by 50%"
    )
    
    print(f"\n🎯 Intervention: {intervention.description}")
    print(f"   Question: What causal effects will this have?")
    
    result = agent.do_intervention(intervention, context1)
    
    print(f"\n💡 Causal Effects:")
    print(f"   Affected Variables: {', '.join(result.affected_variables)}")
    print(f"\n   Analysis:")
    for line in result.answer.split('\n')[:10]:  # First 10 lines
        if line.strip():
            print(f"   {line}")
    
    # Test 3: Counterfactual reasoning
    print("\n" + "=" * 80)
    print("TEST 3: Counterfactual Reasoning")
    print("=" * 80)
    
    counterfactual = CounterfactualQuery(
        observed_facts={
            "marketing_budget": "reduced",
            "sales": "declined",
            "market_share": "lost"
        },
        intervention=Intervention(
            variable="marketing budget",
            value="maintained at original level",
            description="maintained marketing budget at original level"
        ),
        query_variable="sales"
    )
    
    print(f"\n❓ Counterfactual Question:")
    print(f"   {counterfactual.to_text()}")
    
    cf_result = agent.answer_counterfactual(counterfactual, context1)
    
    print(f"\n🔮 Counterfactual Answer:")
    print(f"   Confidence: {cf_result.confidence:.2%}")
    print(f"\n   Answer:")
    for line in cf_result.answer.split('\n')[:8]:
        if line.strip():
            print(f"   {line}")
    
    # Test 4: Root cause analysis
    print("\n" + "=" * 80)
    print("TEST 4: Root Cause Analysis")
    print("=" * 80)
    
    agent2 = CausalReasoningAgent(domain="software systems")
    
    context2 = """
    A web application is experiencing slow response times:
    - Database queries are taking longer
    - Server CPU usage is high
    - Memory usage increased
    - Number of active users grew
    - Cache hit rate decreased
    - Recent code deployment added complex queries
    
    The code deployment introduced inefficient queries, which increased
    database load. Growing user base put additional load on the system.
    High server load caused slower query execution. Poor cache performance
    meant more database hits.
    """
    
    print(f"\n🔍 Analyzing system performance issue...")
    
    agent2.build_causal_graph(context2)
    
    root_cause = agent2.identify_root_cause("slow response times", context2)
    
    print(f"\n🎯 Root Cause Analysis:")
    print(f"   Effect: slow response times")
    print(f"   Confidence: {root_cause.confidence:.2%}")
    print(f"\n   Root Cause(s):")
    for line in root_cause.answer.split('\n'):
        if line.strip():
            print(f"   {line}")
    
    # Test 5: Comparing correlation vs causation
    print("\n" + "=" * 80)
    print("TEST 5: Correlation vs Causation")
    print("=" * 80)
    
    agent3 = CausalReasoningAgent(domain="healthcare")
    
    context3 = """
    Study observations:
    - People who drink coffee have lower rates of heart disease
    - People who exercise regularly drink more coffee
    - People who exercise have better heart health
    - Coffee consumption and exercise are correlated
    - Income level affects both coffee drinking and exercise
    """
    
    print(f"\n📈 Analyzing correlation pattern...")
    
    graph3 = agent3.build_causal_graph(context3)
    print(f"\n   Causal Graph:")
    for edge in graph3.edges:
        print(f"   {edge}")
    
    # Compare two interventions
    print(f"\n   Comparing interventions:")
    
    int1 = Intervention("coffee consumption", "increase", "increase coffee consumption")
    int2 = Intervention("exercise", "increase", "increase exercise")
    
    result1 = agent3.do_intervention(int1, context3)
    result2 = agent3.do_intervention(int2, context3)
    
    print(f"\n   If we increase COFFEE consumption:")
    print(f"      Affected: {', '.join(result1.affected_variables)}")
    
    print(f"\n   If we increase EXERCISE:")
    print(f"      Affected: {', '.join(result2.affected_variables)}")
    
    print(f"\n   ⚠️  Causal vs Correlational insight:")
    print(f"      Correlation doesn't imply causation!")
    print(f"      The causal structure reveals true mechanisms.")
    
    # Summary
    print("\n" + "=" * 80)
    print("CAUSAL REASONING AGENT PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. True Understanding: Distinguish causation from correlation
2. Better Decisions: Predict intervention outcomes
3. Counterfactual Reasoning: Answer "what if" questions
4. Root Cause Analysis: Find true sources of problems
5. Robust Generalization: Transfer across contexts

Causal Hierarchy (Pearl's Ladder):
1. Association (Seeing): P(Y|X)
   - "What is?" (observation)
   - Correlations and patterns
   - Machine learning operates here

2. Intervention (Doing): P(Y|do(X))
   - "What if?" (action)
   - Effects of interventions
   - Experiments and A/B tests

3. Counterfactuals (Imagining): P(Y_x|X',Y')
   - "What if I had?" (retrospection)
   - Alternative histories
   - Requires full causal model

Core Concepts:
- Causal Graph: Directed graph of cause-effect
- Structural Equations: Mechanisms of causation
- do-Operator: Intervention (breaking incoming edges)
- Backdoor Criterion: Identifying confounders
- Frontdoor Criterion: Alternative identification
- Counterfactual: Alternative possible worlds

Types of Causal Questions:
1. Association: "Are X and Y related?"
2. Intervention: "What if we do X?"
3. Counterfactual: "What if we had done X?"
4. Effect Size: "How much does X affect Y?"
5. Attribution: "Did X cause this Y?"
6. Explanation: "Why did Y happen?"

Causal Graph Patterns:
- Chain: A → B → C (mediation)
- Fork: A ← B → C (common cause/confounder)
- Collider: A → B ← C (selection bias)
- Confounding: A ← C → Y (spurious correlation)

Intervention vs Observation:
- Observation: P(Y|X=x) - passive seeing
- Intervention: P(Y|do(X=x)) - active doing
- Key difference: Intervention breaks incoming edges

Use Cases:
- Medicine: Treatment effect estimation
- Policy: Evaluate policy interventions
- Business: Understand driver of metrics
- Science: Hypothesis testing
- Fairness: Identify discrimination
- Debugging: Root cause analysis
- Legal: Attribution and liability

Causal Discovery Methods:
1. Constraint-Based: Test conditional independencies
2. Score-Based: Search for best-fitting graph
3. Functional: Exploit noise structure
4. Hybrid: Combine approaches
5. LLM-Based: Extract from text/knowledge

Challenges:
1. Unobserved Confounders: Hidden common causes
2. Selection Bias: Collider conditioning
3. Measurement Error: Imperfect variables
4. Feedback Loops: Bidirectional causation
5. Time Delays: Lag between cause and effect
6. Context Dependence: Different mechanisms

Best Practices:
1. Draw causal graph explicitly
2. Identify confounders
3. Use randomization when possible
4. Combine domain knowledge + data
5. Test causal assumptions
6. Estimate uncertainty
7. Consider alternative graphs

Production Considerations:
- Causal graph versioning
- Domain expert validation
- A/B testing for verification
- Sensitivity analysis
- Handling edge cases
- Explanation generation
- Confidence calibration

Advanced Techniques:
1. Causal Effect Estimation
   - Instrumental variables
   - Difference-in-differences
   - Regression discontinuity
   - Propensity score matching

2. Mediation Analysis
   - Direct vs indirect effects
   - Path analysis

3. Time-Series Causality
   - Granger causality
   - Dynamic causal models

4. Causal Reinforcement Learning
   - Credit assignment
   - Off-policy evaluation

Comparison with Related Patterns:
- vs. Correlation: Distinguishes cause from association
- vs. Prediction: Explains mechanism, not just predicts
- vs. Counterfactual (simple): Full causal framework
- vs. Analogical: Causal structure vs similarity

Integration with Other Patterns:
- World Models: Causal = mechanism in world model
- Planning: Causal graph guides planning
- Explanation: Causal = better explanations
- Reasoning: Foundation for robust reasoning

Philosophical Note:
Causal reasoning represents understanding of "why" not just
"what". It's the difference between prediction (correlation)
and understanding (causation). Essential for AGI.

The Causal Reasoning Agent pattern enables agents to move
beyond correlation to true causal understanding, supporting
better decisions, explanations, and counterfactual reasoning.
""")


if __name__ == "__main__":
    demonstrate_causal_reasoning()
