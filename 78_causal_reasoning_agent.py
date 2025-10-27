"""
Causal Reasoning Agent Pattern

Agents that reason about cause-effect relationships using causal graphs,
interventions, and counterfactual reasoning. Enables understanding of 
why things happen and what-if scenario analysis.

Use Cases:
- Root cause analysis
- Scientific discovery
- Policy evaluation
- Explanatory AI
- What-if scenario planning

Benefits:
- Better generalization beyond correlations
- Interpretable reasoning
- Counterfactual predictions
- Intervention planning
- Transfer to new domains
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random


class CausalRelationType(Enum):
    """Types of causal relationships"""
    DIRECT_CAUSE = "direct_cause"
    INDIRECT_CAUSE = "indirect_cause"
    COMMON_CAUSE = "common_cause"
    COMMON_EFFECT = "common_effect"


@dataclass
class Variable:
    """Variable in causal model"""
    name: str
    value: Any = None
    domain: List[Any] = field(default_factory=list)
    is_observed: bool = True
    
    def set_value(self, value: Any) -> None:
        """Set variable value"""
        if self.domain and value not in self.domain:
            raise ValueError(f"Value {value} not in domain {self.domain}")
        self.value = value


@dataclass
class CausalEdge:
    """Directed edge representing causation"""
    cause: str  # Variable name
    effect: str  # Variable name
    strength: float = 1.0  # Causal strength
    mechanism: Optional[str] = None  # Description of causal mechanism


@dataclass
class CausalGraph:
    """Directed Acyclic Graph representing causal relationships"""
    variables: Dict[str, Variable] = field(default_factory=dict)
    edges: List[CausalEdge] = field(default_factory=list)
    
    def add_variable(self, variable: Variable) -> None:
        """Add variable to graph"""
        self.variables[variable.name] = variable
    
    def add_edge(self, edge: CausalEdge) -> None:
        """Add causal edge"""
        if edge.cause not in self.variables or edge.effect not in self.variables:
            raise ValueError("Both cause and effect must be in graph")
        self.edges.append(edge)
    
    def get_parents(self, variable_name: str) -> List[str]:
        """Get direct causes of variable"""
        return [e.cause for e in self.edges if e.effect == variable_name]
    
    def get_children(self, variable_name: str) -> List[str]:
        """Get direct effects of variable"""
        return [e.effect for e in self.edges if e.cause == variable_name]
    
    def get_ancestors(self, variable_name: str) -> Set[str]:
        """Get all ancestral causes (transitive closure)"""
        ancestors = set()
        to_visit = self.get_parents(variable_name)
        
        while to_visit:
            current = to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self.get_parents(current))
        
        return ancestors
    
    def get_descendants(self, variable_name: str) -> Set[str]:
        """Get all descendant effects (transitive closure)"""
        descendants = set()
        to_visit = self.get_children(variable_name)
        
        while to_visit:
            current = to_visit.pop()
            if current not in descendants:
                descendants.add(current)
                to_visit.extend(self.get_children(current))
        
        return descendants
    
    def is_dseparated(self, x: str, y: str, z: Set[str]) -> bool:
        """
        Check if X and Y are d-separated given Z
        Simplified version - full implementation would use graph algorithms
        """
        # This is a simplified heuristic
        # Full d-separation requires analyzing all paths
        if x == y:
            return False
        
        # If Z blocks all paths from X to Y
        x_ancestors = self.get_ancestors(x)
        y_ancestors = self.get_ancestors(y)
        
        # Check if conditioning set blocks paths
        return bool(z & (x_ancestors | y_ancestors))


@dataclass
class Intervention:
    """Intervention on variable (do-operator)"""
    variable: str
    value: Any
    
    def __str__(self) -> str:
        return f"do({self.variable}={self.value})"


@dataclass
class CounterfactualQuery:
    """Counterfactual what-if query"""
    actual_world: Dict[str, Any]  # What actually happened
    intervention: Intervention  # What we change
    query_variable: str  # What we want to know
    
    def __str__(self) -> str:
        return f"What if {self.intervention} (given actual: {self.actual_world})? Query: {self.query_variable}"


class CausalInference:
    """
    Performs causal inference on causal graphs
    """
    
    def __init__(self, causal_graph: CausalGraph):
        self.graph = causal_graph
    
    def observe(self, observations: Dict[str, Any]) -> None:
        """Update variable values from observations"""
        for var_name, value in observations.items():
            if var_name in self.graph.variables:
                self.graph.variables[var_name].set_value(value)
    
    def intervene(self, intervention: Intervention) -> Dict[str, Any]:
        """
        Perform intervention (do-operator)
        Removes incoming edges to intervened variable
        """
        print(f"\n[Intervention] {intervention}")
        
        # Set intervened value
        if intervention.variable in self.graph.variables:
            self.graph.variables[intervention.variable].value = intervention.value
        
        # Simulate effects (simplified - would use full structural equations)
        effects = self._propagate_effects(intervention.variable)
        
        print(f"  Effects: {effects}")
        return effects
    
    def _propagate_effects(self, start_variable: str) -> Dict[str, Any]:
        """Propagate effects through causal graph"""
        effects = {}
        descendants = self.graph.get_descendants(start_variable)
        
        for desc in descendants:
            # Simplified effect computation
            parents = self.graph.get_parents(desc)
            parent_values = [self.graph.variables[p].value for p in parents if p in self.graph.variables]
            
            # Simple aggregation (in reality would use structural equations)
            if parent_values and all(v is not None for v in parent_values):
                if all(isinstance(v, (int, float)) for v in parent_values):
                    effects[desc] = sum(parent_values) / len(parent_values)
                else:
                    effects[desc] = parent_values[0]
        
        return effects
    
    def counterfactual(self, query: CounterfactualQuery) -> Any:
        """
        Answer counterfactual query: What if we had done X instead?
        """
        print(f"\n[Counterfactual] {query}")
        
        # Step 1: Abduction - infer latent variables from actual world
        self.observe(query.actual_world)
        
        # Step 2: Action - apply intervention
        effects = self.intervene(query.intervention)
        
        # Step 3: Prediction - compute query variable
        if query.query_variable in effects:
            result = effects[query.query_variable]
        elif query.query_variable in self.graph.variables:
            result = self.graph.variables[query.query_variable].value
        else:
            result = None
        
        print(f"  Answer: {query.query_variable} = {result}")
        return result
    
    def estimate_causal_effect(
        self,
        treatment: str,
        outcome: str,
        adjustment_set: Optional[Set[str]] = None
    ) -> str:
        """
        Estimate causal effect of treatment on outcome
        """
        print(f"\n[Causal Effect] {treatment} → {outcome}")
        
        if adjustment_set:
            print(f"  Adjusting for: {adjustment_set}")
        
        # Check if backdoor criterion is satisfied
        backdoor_open = self._has_backdoor_path(treatment, outcome, adjustment_set or set())
        
        if backdoor_open:
            return "Cannot estimate - backdoor paths remain open"
        
        # Simplified effect estimation
        if outcome in self.graph.get_descendants(treatment):
            return f"{treatment} has causal effect on {outcome}"
        else:
            return f"{treatment} does not causally affect {outcome}"
    
    def _has_backdoor_path(
        self,
        treatment: str,
        outcome: str,
        adjustment_set: Set[str]
    ) -> bool:
        """Check if backdoor paths exist (simplified)"""
        # Find common ancestors
        treatment_ancestors = self.graph.get_ancestors(treatment)
        outcome_ancestors = self.graph.get_ancestors(outcome)
        common_ancestors = treatment_ancestors & outcome_ancestors
        
        # If common ancestors not in adjustment set, backdoor is open
        return bool(common_ancestors - adjustment_set)


class CausalReasoningAgent:
    """
    Agent that uses causal reasoning for decision-making
    """
    
    def __init__(self, name: str = "Causal Agent"):
        self.name = name
        self.causal_graph = CausalGraph()
        self.inference_engine = CausalInference(self.causal_graph)
        self.observations: Dict[str, Any] = {}
        
        print(f"[Agent] Initialized: {name}")
    
    def build_causal_model(
        self,
        variables: List[Variable],
        edges: List[CausalEdge]
    ) -> None:
        """Build causal model of domain"""
        print(f"\n[Building Causal Model]")
        print(f"  Variables: {len(variables)}")
        print(f"  Causal edges: {len(edges)}")
        
        for var in variables:
            self.causal_graph.add_variable(var)
        
        for edge in edges:
            self.causal_graph.add_edge(edge)
            print(f"    {edge.cause} → {edge.effect} (strength: {edge.strength})")
    
    def observe_data(self, observations: Dict[str, Any]) -> None:
        """Observe data from environment"""
        print(f"\n[Observations] {observations}")
        self.observations.update(observations)
        self.inference_engine.observe(observations)
    
    def explain_observation(self, variable: str) -> List[str]:
        """Explain why variable has its value"""
        print(f"\n[Explaining] Why {variable} = {self.causal_graph.variables[variable].value}?")
        
        causes = self.causal_graph.get_parents(variable)
        
        explanation = []
        for cause in causes:
            cause_var = self.causal_graph.variables[cause]
            explanation.append(f"{cause} = {cause_var.value}")
            print(f"  Because: {cause} = {cause_var.value}")
        
        return explanation
    
    def predict_intervention(self, intervention: Intervention) -> Dict[str, Any]:
        """Predict effects of intervention"""
        print(f"\n[Predicting Intervention]")
        return self.inference_engine.intervene(intervention)
    
    def answer_counterfactual(self, query: CounterfactualQuery) -> Any:
        """Answer what-if question"""
        return self.inference_engine.counterfactual(query)
    
    def find_root_cause(self, problem_variable: str) -> List[str]:
        """Find root causes of observed problem"""
        print(f"\n[Root Cause Analysis] for {problem_variable}")
        
        ancestors = self.causal_graph.get_ancestors(problem_variable)
        
        # Root causes are ancestors with no parents
        root_causes = []
        for ancestor in ancestors:
            if not self.causal_graph.get_parents(ancestor):
                root_causes.append(ancestor)
                print(f"  Root cause: {ancestor}")
        
        return root_causes
    
    def recommend_action(
        self,
        goal_variable: str,
        desired_value: Any
    ) -> List[Intervention]:
        """Recommend interventions to achieve goal"""
        print(f"\n[Action Recommendation] Goal: {goal_variable} = {desired_value}")
        
        # Find variables that causally affect goal
        causes = self.causal_graph.get_parents(goal_variable)
        ancestors = self.causal_graph.get_ancestors(goal_variable)
        
        recommendations = []
        
        print(f"  Direct causes: {causes}")
        print(f"  All ancestors: {ancestors}")
        
        # Recommend intervening on direct causes
        for cause in causes:
            intervention = Intervention(cause, desired_value)
            recommendations.append(intervention)
            print(f"  Recommendation: {intervention}")
        
        return recommendations


def demonstrate_causal_reasoning():
    """
    Demonstrate Causal Reasoning Agent pattern
    """
    print("=" * 70)
    print("CAUSAL REASONING AGENT DEMONSTRATION")
    print("=" * 70)
    
    # Example 1: Medical diagnosis causal model
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Medical Diagnosis")
    print("=" * 70)
    
    agent = CausalReasoningAgent("Medical Diagnosis Agent")
    
    # Build causal model
    variables = [
        Variable("smoking", domain=[True, False]),
        Variable("exercise", domain=[True, False]),
        Variable("diet", domain=["healthy", "unhealthy"]),
        Variable("heart_disease", domain=[True, False]),
        Variable("symptoms", domain=[True, False]),
    ]
    
    edges = [
        CausalEdge("smoking", "heart_disease", strength=0.8),
        CausalEdge("exercise", "heart_disease", strength=-0.5),
        CausalEdge("diet", "heart_disease", strength=0.6),
        CausalEdge("heart_disease", "symptoms", strength=0.9),
    ]
    
    agent.build_causal_model(variables, edges)
    
    # Observe patient data
    agent.observe_data({
        "smoking": True,
        "exercise": False,
        "diet": "unhealthy",
        "heart_disease": True,
        "symptoms": True
    })
    
    # Explain symptoms
    agent.explain_observation("symptoms")
    
    # Find root causes
    agent.find_root_cause("symptoms")
    
    # Example 2: Intervention prediction
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Predicting Intervention Effects")
    print("=" * 70)
    
    # What if patient starts exercising?
    intervention = Intervention("exercise", True)
    effects = agent.predict_intervention(intervention)
    
    # Recommend actions
    agent.recommend_action("heart_disease", False)
    
    # Example 3: Counterfactual reasoning
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Counterfactual Reasoning")
    print("=" * 70)
    
    # What if the patient hadn't smoked?
    counterfactual = CounterfactualQuery(
        actual_world={"smoking": True, "heart_disease": True, "symptoms": True},
        intervention=Intervention("smoking", False),
        query_variable="heart_disease"
    )
    
    result = agent.answer_counterfactual(counterfactual)
    
    # Example 4: Business scenario
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Business Decision Making")
    print("=" * 70)
    
    business_agent = CausalReasoningAgent("Business Strategy Agent")
    
    # Build business causal model
    biz_variables = [
        Variable("marketing_spend", domain=[]),
        Variable("product_quality", domain=[]),
        Variable("brand_awareness", domain=[]),
        Variable("customer_satisfaction", domain=[]),
        Variable("sales", domain=[]),
    ]
    
    biz_edges = [
        CausalEdge("marketing_spend", "brand_awareness", strength=0.7),
        CausalEdge("product_quality", "customer_satisfaction", strength=0.9),
        CausalEdge("brand_awareness", "sales", strength=0.6),
        CausalEdge("customer_satisfaction", "sales", strength=0.8),
    ]
    
    business_agent.build_causal_model(biz_variables, biz_edges)
    
    # Analyze causal effect
    effect = business_agent.inference_engine.estimate_causal_effect(
        "marketing_spend",
        "sales",
        adjustment_set={"product_quality"}
    )
    
    print(f"\nCausal effect estimate: {effect}")


def demonstrate_causal_inference():
    """Additional causal inference examples"""
    print("\n" + "=" * 70)
    print("CAUSAL INFERENCE EXAMPLES")
    print("=" * 70)
    
    # Simpson's Paradox example
    print("\nExample: Simpson's Paradox")
    print("Aggregate correlation can differ from causal effect")
    print("Adjustment for confounders is essential!")
    
    # Mediation analysis
    print("\nExample: Mediation Analysis")
    print("X → M → Y: M mediates effect of X on Y")
    print("Direct effect: X → Y")
    print("Indirect effect: X → M → Y")


if __name__ == "__main__":
    demonstrate_causal_reasoning()
    demonstrate_causal_inference()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Causal reasoning enables understanding WHY things happen
2. Interventions (do-operator) differ from observations
3. Counterfactuals answer "what if" questions
4. Causal graphs represent domain knowledge
5. Causal inference requires careful identification

Best Practices:
- Build causal models from domain knowledge
- Distinguish correlation from causation
- Use interventions for policy evaluation
- Consider confounders in analysis
- Validate causal assumptions
- Use counterfactuals for explanations
- Combine with observational data
- Be transparent about assumptions
    """)
