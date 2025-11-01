"""
Pattern 107: Multi-Objective Planning

Description:
    The Multi-Objective Planning pattern enables agents to plan and make decisions
    when multiple, often conflicting objectives must be balanced. Instead of
    optimizing for a single goal, the agent must consider tradeoffs between
    competing objectives like cost vs. quality, speed vs. accuracy, or risk vs.
    reward.
    
    This pattern is essential for realistic decision-making where perfect solutions
    don't exist and every choice involves tradeoffs. The agent must evaluate plans
    across multiple dimensions, understand Pareto-optimal solutions, and make
    informed decisions about which tradeoffs are acceptable given the context.
    
    Multi-objective planning involves objective weighting, constraint satisfaction,
    Pareto optimization, preference elicitation, and tradeoff analysis. The agent
    can present multiple plan options with different tradeoff profiles, allowing
    humans or automated systems to select based on current priorities.

Key Components:
    1. Objective Set: Multiple goals to optimize
    2. Objective Weights: Relative importance of each objective
    3. Plan Evaluator: Scores plans across all objectives
    4. Pareto Filter: Identifies non-dominated solutions
    5. Tradeoff Analyzer: Analyzes competing objectives
    6. Constraint Checker: Ensures hard constraints met
    7. Preference Engine: Incorporates user preferences

Objective Types:
    1. Maximization: Quality, performance, completeness
    2. Minimization: Cost, time, risk, resource usage
    3. Target: Hit specific value (with tolerance)
    4. Constraint: Must satisfy (hard requirement)
    
Common Objective Combinations:
    - Cost vs. Quality vs. Speed (iron triangle)
    - Risk vs. Reward
    - Precision vs. Recall
    - Exploration vs. Exploitation
    - Short-term vs. Long-term goals
    - Individual vs. Group benefit

Planning Approaches:
    1. Weighted Sum: Combine objectives with weights
    2. Lexicographic: Prioritize objectives in order
    3. Pareto Optimal: Non-dominated solutions
    4. Satisficing: Good enough on all objectives
    5. Goal Programming: Meet targets with deviations
    6. Multi-Criteria Decision Making (MCDM)

Use Cases:
    - Resource allocation with constraints
    - Project planning with multiple stakeholders
    - Route planning (time vs. cost vs. comfort)
    - Product design (features vs. complexity vs. cost)
    - Investment decisions (risk vs. return)
    - Scheduling (efficiency vs. fairness)
    - Machine learning (accuracy vs. interpretability)

Advantages:
    - Realistic modeling of complex decisions
    - Explicit tradeoff consideration
    - Multiple solution alternatives
    - Stakeholder preference incorporation
    - Better than single-objective optimization
    - Transparent decision criteria
    - Flexible priority adjustment

Challenges:
    - Objective quantification difficulty
    - Weight elicitation from users
    - Computational complexity
    - Visualizing multi-dimensional tradeoffs
    - Handling conflicting constraints
    - Determining Pareto frontier
    - Explaining tradeoff decisions

LangChain Implementation:
    This implementation uses LangChain for:
    - LLM-based objective identification
    - Plan evaluation across dimensions
    - Tradeoff analysis and explanation
    - Natural language objective specification
    
Production Considerations:
    - Define clear, measurable objectives
    - Implement efficient Pareto filtering
    - Provide tradeoff visualizations
    - Allow dynamic weight adjustment
    - Cache evaluated plans
    - Support constraint hierarchies
    - Enable sensitivity analysis
    - Log decision rationale
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ObjectiveType(Enum):
    """Type of optimization objective."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"
    CONSTRAINT = "constraint"


@dataclass
class Objective:
    """
    Represents an optimization objective.
    
    Attributes:
        name: Objective name
        objective_type: Maximization, minimization, target, or constraint
        weight: Relative importance (0-1)
        target_value: Target value (for TARGET type)
        constraint_value: Threshold (for CONSTRAINT type)
        unit: Measurement unit
    """
    name: str
    objective_type: ObjectiveType
    weight: float = 0.5
    target_value: Optional[float] = None
    constraint_value: Optional[float] = None
    unit: str = ""
    
    def evaluate(self, value: float) -> float:
        """
        Evaluate objective given a value.
        
        Args:
            value: Actual value
            
        Returns:
            Normalized score (0-1), higher is better
        """
        if self.objective_type == ObjectiveType.MAXIMIZE:
            # Normalize to 0-1 (assuming reasonable range)
            return min(1.0, max(0.0, value / 100.0))
        
        elif self.objective_type == ObjectiveType.MINIMIZE:
            # Lower is better, invert score
            return max(0.0, 1.0 - (value / 100.0))
        
        elif self.objective_type == ObjectiveType.TARGET:
            # Closer to target is better
            if self.target_value is None:
                return 0.5
            distance = abs(value - self.target_value)
            return max(0.0, 1.0 - (distance / 50.0))
        
        elif self.objective_type == ObjectiveType.CONSTRAINT:
            # Binary: constraint satisfied or not
            if self.constraint_value is None:
                return 1.0
            return 1.0 if value <= self.constraint_value else 0.0
        
        return 0.5


@dataclass
class Plan:
    """
    Represents a plan with multiple objective values.
    
    Attributes:
        plan_id: Unique identifier
        description: Plan description
        objective_values: Values for each objective
        feasible: Whether all constraints are satisfied
        dominated: Whether dominated by another plan
    """
    plan_id: str
    description: str
    objective_values: Dict[str, float] = field(default_factory=dict)
    feasible: bool = True
    dominated: bool = False
    
    def get_score(self, objective: Objective) -> float:
        """Get normalized score for an objective."""
        value = self.objective_values.get(objective.name, 0.0)
        return objective.evaluate(value)


class MultiObjectivePlanner:
    """
    Multi-objective planning system.
    
    This planner evaluates and compares plans across multiple objectives,
    identifies Pareto-optimal solutions, and analyzes tradeoffs.
    """
    
    def __init__(self, temperature: float = 0.4):
        """
        Initialize multi-objective planner.
        
        Args:
            temperature: LLM temperature
        """
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.objectives: List[Objective] = []
        self.plans: List[Plan] = []
        self.plan_counter = 0
    
    def add_objective(
        self,
        name: str,
        objective_type: ObjectiveType,
        weight: float = 0.5,
        target_value: Optional[float] = None,
        constraint_value: Optional[float] = None,
        unit: str = ""
    ) -> Objective:
        """
        Add an objective to optimize.
        
        Args:
            name: Objective name
            objective_type: Type of objective
            weight: Relative importance
            target_value: Target value (for TARGET)
            constraint_value: Constraint threshold
            unit: Measurement unit
            
        Returns:
            Created objective
        """
        objective = Objective(
            name=name,
            objective_type=objective_type,
            weight=weight,
            target_value=target_value,
            constraint_value=constraint_value,
            unit=unit
        )
        self.objectives.append(objective)
        return objective
    
    def create_plan(
        self,
        description: str,
        objective_values: Dict[str, float]
    ) -> Plan:
        """
        Create a plan with objective values.
        
        Args:
            description: Plan description
            objective_values: Values for each objective
            
        Returns:
            Created plan
        """
        self.plan_counter += 1
        
        # Check constraint satisfaction
        feasible = True
        for obj in self.objectives:
            if obj.objective_type == ObjectiveType.CONSTRAINT:
                value = objective_values.get(obj.name, float('inf'))
                if obj.constraint_value is not None and value > obj.constraint_value:
                    feasible = False
                    break
        
        plan = Plan(
            plan_id=f"plan_{self.plan_counter}",
            description=description,
            objective_values=objective_values,
            feasible=feasible
        )
        
        self.plans.append(plan)
        return plan
    
    def evaluate_plan(self, plan: Plan) -> Dict[str, Any]:
        """
        Evaluate plan across all objectives.
        
        Args:
            plan: Plan to evaluate
            
        Returns:
            Evaluation results
        """
        scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for obj in self.objectives:
            score = plan.get_score(obj)
            scores[obj.name] = score
            
            if obj.objective_type != ObjectiveType.CONSTRAINT:
                weighted_sum += score * obj.weight
                total_weight += obj.weight
        
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return {
            "plan_id": plan.plan_id,
            "feasible": plan.feasible,
            "scores": scores,
            "overall_score": overall_score,
            "objective_values": plan.objective_values
        }
    
    def find_pareto_optimal(self) -> List[Plan]:
        """
        Find Pareto-optimal (non-dominated) plans.
        
        Returns:
            List of Pareto-optimal plans
        """
        feasible_plans = [p for p in self.plans if p.feasible]
        
        if not feasible_plans:
            return []
        
        # Mark dominated plans
        for plan in feasible_plans:
            plan.dominated = False
        
        for i, plan1 in enumerate(feasible_plans):
            for plan2 in feasible_plans[i+1:]:
                # Check if plan1 dominates plan2
                plan1_better = False
                plan2_better = False
                
                for obj in self.objectives:
                    if obj.objective_type == ObjectiveType.CONSTRAINT:
                        continue
                    
                    score1 = plan1.get_score(obj)
                    score2 = plan2.get_score(obj)
                    
                    if score1 > score2:
                        plan1_better = True
                    elif score2 > score1:
                        plan2_better = True
                
                # Mark dominated plans
                if plan1_better and not plan2_better:
                    plan2.dominated = True
                elif plan2_better and not plan1_better:
                    plan1.dominated = True
        
        return [p for p in feasible_plans if not p.dominated]
    
    def analyze_tradeoffs(
        self,
        plan1: Plan,
        plan2: Plan
    ) -> Dict[str, Any]:
        """
        Analyze tradeoffs between two plans.
        
        Args:
            plan1: First plan
            plan2: Second plan
            
        Returns:
            Tradeoff analysis
        """
        tradeoffs = []
        
        for obj in self.objectives:
            if obj.objective_type == ObjectiveType.CONSTRAINT:
                continue
            
            score1 = plan1.get_score(obj)
            score2 = plan2.get_score(obj)
            value1 = plan1.objective_values.get(obj.name, 0)
            value2 = plan2.objective_values.get(obj.name, 0)
            
            difference = score2 - score1
            
            if abs(difference) > 0.05:  # Significant difference
                tradeoffs.append({
                    "objective": obj.name,
                    "plan1_value": value1,
                    "plan2_value": value2,
                    "plan1_score": score1,
                    "plan2_score": score2,
                    "difference": difference,
                    "winner": "plan2" if difference > 0 else "plan1"
                })
        
        return {
            "plan1": plan1.plan_id,
            "plan2": plan2.plan_id,
            "tradeoffs": tradeoffs,
            "num_tradeoffs": len(tradeoffs)
        }
    
    def generate_plan_with_priorities(
        self,
        task: str,
        priority_objectives: List[str]
    ) -> Plan:
        """
        Generate plan optimizing for priority objectives.
        
        Args:
            task: Task description
            priority_objectives: High-priority objective names
            
        Returns:
            Generated plan
        """
        # Create prompt emphasizing priority objectives
        objectives_text = "\n".join([
            f"- {obj.name} ({obj.objective_type.value})"
            + (f" - PRIORITY" if obj.name in priority_objectives else "")
            for obj in self.objectives
        ])
        
        prompt = ChatPromptTemplate.from_template(
            "Create a plan for this task that optimizes the priority objectives:\n\n"
            "Task: {task}\n\n"
            "Objectives:\n{objectives}\n\n"
            "Provide a brief plan description (one sentence).\n"
            "Plan:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        description = chain.invoke({
            "task": task,
            "objectives": objectives_text
        }).strip()
        
        # Generate estimated values (in practice, would come from simulation)
        objective_values = {}
        for obj in self.objectives:
            # Favor priority objectives
            if obj.name in priority_objectives:
                if obj.objective_type == ObjectiveType.MAXIMIZE:
                    objective_values[obj.name] = 80.0
                elif obj.objective_type == ObjectiveType.MINIMIZE:
                    objective_values[obj.name] = 20.0
                else:
                    objective_values[obj.name] = 50.0
            else:
                # Non-priority gets moderate values
                objective_values[obj.name] = 50.0
        
        return self.create_plan(description, objective_values)
    
    def rank_plans(self) -> List[Tuple[Plan, float]]:
        """
        Rank plans by weighted overall score.
        
        Returns:
            List of (plan, score) tuples, sorted by score
        """
        ranked = []
        
        for plan in self.plans:
            if not plan.feasible:
                continue
            
            eval_result = self.evaluate_plan(plan)
            ranked.append((plan, eval_result["overall_score"]))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def get_summary(self) -> str:
        """Get summary of plans and objectives."""
        lines = ["MULTI-OBJECTIVE PLANNING SUMMARY"]
        lines.append("=" * 60)
        
        # Objectives
        lines.append("\nObjectives:")
        for obj in self.objectives:
            lines.append(
                f"  • {obj.name} ({obj.objective_type.value})"
                f" - weight: {obj.weight:.2f}"
            )
        
        # Plans
        lines.append(f"\nTotal Plans: {len(self.plans)}")
        lines.append(f"Feasible Plans: {sum(1 for p in self.plans if p.feasible)}")
        
        # Pareto optimal
        pareto = self.find_pareto_optimal()
        lines.append(f"Pareto-Optimal Plans: {len(pareto)}")
        
        return "\n".join(lines)


def demonstrate_multi_objective_planning():
    """Demonstrate multi-objective planning pattern."""
    
    print("=" * 80)
    print("MULTI-OBJECTIVE PLANNING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic multi-objective planning
    print("\n" + "=" * 80)
    print("Example 1: Project Planning with Cost, Time, Quality")
    print("=" * 80)
    
    planner = MultiObjectivePlanner()
    
    # Define objectives
    planner.add_objective("cost", ObjectiveType.MINIMIZE, weight=0.4, unit="$")
    planner.add_objective("time", ObjectiveType.MINIMIZE, weight=0.3, unit="days")
    planner.add_objective("quality", ObjectiveType.MAXIMIZE, weight=0.3, unit="score")
    
    print("\nObjectives:")
    for obj in planner.objectives:
        print(f"  • {obj.name}: {obj.objective_type.value} (weight: {obj.weight})")
    
    # Create alternative plans
    plans_data = [
        ("Budget plan: Minimize costs, longer timeline", {"cost": 10000, "time": 60, "quality": 70}),
        ("Fast plan: Minimize time, higher costs", {"cost": 25000, "time": 30, "quality": 75}),
        ("Quality plan: Maximize quality, balanced cost/time", {"cost": 18000, "time": 45, "quality": 90}),
        ("Balanced plan: Moderate on all objectives", {"cost": 15000, "time": 45, "quality": 80}),
    ]
    
    print("\nCreating alternative plans...")
    for desc, values in plans_data:
        plan = planner.create_plan(desc, values)
        print(f"  ✓ {plan.plan_id}: {desc}")
    
    # Evaluate plans
    print("\nPlan Evaluations:")
    for plan in planner.plans:
        eval_result = planner.evaluate_plan(plan)
        print(f"\n  {plan.plan_id}: {plan.description}")
        print(f"    Overall Score: {eval_result['overall_score']:.3f}")
        for obj_name, score in eval_result['scores'].items():
            value = plan.objective_values[obj_name]
            print(f"    {obj_name}: {value} → score: {score:.3f}")
    
    # Example 2: Pareto optimal solutions
    print("\n" + "=" * 80)
    print("Example 2: Finding Pareto-Optimal Solutions")
    print("=" * 80)
    
    planner2 = MultiObjectivePlanner()
    
    planner2.add_objective("performance", ObjectiveType.MAXIMIZE, weight=0.5)
    planner2.add_objective("efficiency", ObjectiveType.MAXIMIZE, weight=0.5)
    
    # Create plans with different tradeoffs
    test_plans = [
        ("High performance, low efficiency", {"performance": 90, "efficiency": 40}),
        ("Low performance, high efficiency", {"performance": 40, "efficiency": 90}),
        ("Balanced", {"performance": 70, "efficiency": 70}),
        ("Dominated plan", {"performance": 50, "efficiency": 50}),
        ("Another balanced", {"performance": 75, "efficiency": 65}),
    ]
    
    for desc, values in test_plans:
        planner2.create_plan(desc, values)
    
    print("\nAll Plans:")
    for plan in planner2.plans:
        print(f"  {plan.plan_id}: {plan.description}")
        print(f"    Performance: {plan.objective_values['performance']}, "
              f"Efficiency: {plan.objective_values['efficiency']}")
    
    pareto_plans = planner2.find_pareto_optimal()
    
    print(f"\nPareto-Optimal Plans ({len(pareto_plans)} found):")
    for plan in pareto_plans:
        print(f"  ✓ {plan.plan_id}: {plan.description}")
        print(f"    Performance: {plan.objective_values['performance']}, "
              f"Efficiency: {plan.objective_values['efficiency']}")
    
    # Example 3: Tradeoff analysis
    print("\n" + "=" * 80)
    print("Example 3: Tradeoff Analysis Between Plans")
    print("=" * 80)
    
    planner3 = MultiObjectivePlanner()
    
    planner3.add_objective("speed", ObjectiveType.MAXIMIZE, weight=0.4)
    planner3.add_objective("accuracy", ObjectiveType.MAXIMIZE, weight=0.4)
    planner3.add_objective("cost", ObjectiveType.MINIMIZE, weight=0.2)
    
    plan_a = planner3.create_plan(
        "Fast but less accurate",
        {"speed": 90, "accuracy": 60, "cost": 20}
    )
    
    plan_b = planner3.create_plan(
        "Slow but very accurate",
        {"speed": 50, "accuracy": 95, "cost": 40}
    )
    
    print("\nComparing two plans:")
    print(f"  Plan A: {plan_a.description}")
    print(f"  Plan B: {plan_b.description}")
    
    tradeoff = planner3.analyze_tradeoffs(plan_a, plan_b)
    
    print(f"\nTradeoff Analysis ({tradeoff['num_tradeoffs']} significant differences):")
    for t in tradeoff['tradeoffs']:
        winner = "Plan A" if t['winner'] == 'plan1' else "Plan B"
        print(f"  • {t['objective']}:")
        print(f"    Plan A: {t['plan1_value']} (score: {t['plan1_score']:.3f})")
        print(f"    Plan B: {t['plan2_value']} (score: {t['plan2_score']:.3f})")
        print(f"    Winner: {winner}")
    
    # Example 4: Constraints
    print("\n" + "=" * 80)
    print("Example 4: Hard Constraints")
    print("=" * 80)
    
    planner4 = MultiObjectivePlanner()
    
    planner4.add_objective("profit", ObjectiveType.MAXIMIZE, weight=0.6)
    planner4.add_objective("customer_satisfaction", ObjectiveType.MAXIMIZE, weight=0.4)
    planner4.add_objective("budget", ObjectiveType.CONSTRAINT, constraint_value=50000)
    
    print("\nObjectives:")
    print("  • Maximize profit (weight: 0.6)")
    print("  • Maximize customer satisfaction (weight: 0.4)")
    print("  • Budget constraint: ≤ $50,000")
    
    constraint_plans = [
        ("Within budget", {"profit": 30000, "customer_satisfaction": 80, "budget": 45000}),
        ("Over budget", {"profit": 50000, "customer_satisfaction": 90, "budget": 60000}),
        ("Tight budget", {"profit": 25000, "customer_satisfaction": 75, "budget": 50000}),
    ]
    
    print("\nEvaluating plans:")
    for desc, values in constraint_plans:
        plan = planner4.create_plan(desc, values)
        status = "✓ FEASIBLE" if plan.feasible else "✗ INFEASIBLE"
        print(f"  {status}: {desc}")
        print(f"    Budget: ${values['budget']:,}")
    
    # Example 5: Ranking plans
    print("\n" + "=" * 80)
    print("Example 5: Ranking Plans by Overall Score")
    print("=" * 80)
    
    planner5 = MultiObjectivePlanner()
    
    planner5.add_objective("innovation", ObjectiveType.MAXIMIZE, weight=0.3)
    planner5.add_objective("feasibility", ObjectiveType.MAXIMIZE, weight=0.4)
    planner5.add_objective("risk", ObjectiveType.MINIMIZE, weight=0.3)
    
    ranking_plans = [
        ("Conservative approach", {"innovation": 40, "feasibility": 90, "risk": 10}),
        ("Innovative but risky", {"innovation": 90, "feasibility": 60, "risk": 70}),
        ("Moderate innovation", {"innovation": 70, "feasibility": 80, "risk": 30}),
        ("Safe and feasible", {"innovation": 50, "feasibility": 95, "risk": 15}),
    ]
    
    for desc, values in ranking_plans:
        planner5.create_plan(desc, values)
    
    ranked = planner5.rank_plans()
    
    print("\nPlans Ranked by Weighted Score:")
    for i, (plan, score) in enumerate(ranked, 1):
        print(f"  {i}. {plan.description}")
        print(f"     Overall Score: {score:.3f}")
        print(f"     Innovation: {plan.objective_values['innovation']}, "
              f"Feasibility: {plan.objective_values['feasibility']}, "
              f"Risk: {plan.objective_values['risk']}")
    
    # Example 6: Priority-based planning
    print("\n" + "=" * 80)
    print("Example 6: Priority-Based Plan Generation")
    print("=" * 80)
    
    planner6 = MultiObjectivePlanner()
    
    planner6.add_objective("reliability", ObjectiveType.MAXIMIZE, weight=0.3)
    planner6.add_objective("performance", ObjectiveType.MAXIMIZE, weight=0.3)
    planner6.add_objective("maintainability", ObjectiveType.MAXIMIZE, weight=0.2)
    planner6.add_objective("cost", ObjectiveType.MINIMIZE, weight=0.2)
    
    print("\nGenerating plans with different priorities:")
    
    # Priority on reliability
    plan1 = planner6.generate_plan_with_priorities(
        "Build production system",
        ["reliability"]
    )
    print(f"\n  Priority: Reliability")
    print(f"  Plan: {plan1.description}")
    
    # Priority on performance
    plan2 = planner6.generate_plan_with_priorities(
        "Build production system",
        ["performance"]
    )
    print(f"\n  Priority: Performance")
    print(f"  Plan: {plan2.description}")
    
    # Example 7: Weight sensitivity
    print("\n" + "=" * 80)
    print("Example 7: Objective Weight Sensitivity")
    print("=" * 80)
    
    planner7 = MultiObjectivePlanner()
    
    obj1 = planner7.add_objective("feature_count", ObjectiveType.MAXIMIZE)
    obj2 = planner7.add_objective("simplicity", ObjectiveType.MAXIMIZE)
    
    test_plan = planner7.create_plan(
        "Product design",
        {"feature_count": 70, "simplicity": 60}
    )
    
    print("\nTesting different weight combinations:")
    
    weight_combos = [
        (0.8, 0.2, "Feature-focused"),
        (0.5, 0.5, "Balanced"),
        (0.2, 0.8, "Simplicity-focused"),
    ]
    
    for w1, w2, label in weight_combos:
        obj1.weight = w1
        obj2.weight = w2
        
        eval_result = planner7.evaluate_plan(test_plan)
        print(f"\n  {label} (weights: {w1:.1f}, {w2:.1f}):")
        print(f"    Overall Score: {eval_result['overall_score']:.3f}")
    
    # Example 8: System summary
    print("\n" + "=" * 80)
    print("Example 8: Planning System Summary")
    print("=" * 80)
    
    planner8 = MultiObjectivePlanner()
    
    planner8.add_objective("throughput", ObjectiveType.MAXIMIZE, weight=0.35)
    planner8.add_objective("latency", ObjectiveType.MINIMIZE, weight=0.35)
    planner8.add_objective("resource_usage", ObjectiveType.MINIMIZE, weight=0.30)
    
    # Create several plans
    for i in range(5):
        planner8.create_plan(
            f"Design option {i+1}",
            {
                "throughput": 50 + (i * 10),
                "latency": 100 - (i * 15),
                "resource_usage": 40 + (i * 5)
            }
        )
    
    print(planner8.get_summary())
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Multi-Objective Planning Pattern")
    print("=" * 80)
    
    summary = """
    The Multi-Objective Planning pattern demonstrated:
    
    1. MULTIPLE OBJECTIVES (Example 1):
       - Cost, time, quality tradeoffs
       - Weighted objective evaluation
       - Overall score calculation
       - Alternative plan comparison
       - Objective value normalization
    
    2. PARETO OPTIMALITY (Example 2):
       - Non-dominated solutions
       - Domination checking
       - Efficient frontier identification
       - Multiple optimal alternatives
       - Tradeoff visualization
    
    3. TRADEOFF ANALYSIS (Example 3):
       - Comparing plan pairs
       - Identifying differences
       - Winner determination
       - Quantifying tradeoffs
       - Decision support
    
    4. HARD CONSTRAINTS (Example 4):
       - Constraint satisfaction
       - Feasibility checking
       - Plan filtering
       - Requirement enforcement
       - Infeasible plan rejection
    
    5. PLAN RANKING (Example 5):
       - Weighted overall scoring
       - Sorted alternatives
       - Best plan identification
       - Multi-criteria comparison
       - Decision making
    
    6. PRIORITY OBJECTIVES (Example 6):
       - LLM-guided plan generation
       - Priority emphasis
       - Context-aware planning
       - Objective focus
       - Adaptive generation
    
    7. WEIGHT SENSITIVITY (Example 7):
       - Dynamic weight adjustment
       - Impact analysis
       - Preference tuning
       - Score variation
       - Stakeholder alignment
    
    8. SYSTEM INTEGRATION (Example 8):
       - Comprehensive summary
       - Statistics tracking
       - Pareto analysis
       - System overview
       - Decision support
    
    KEY BENEFITS:
    ✓ Realistic complex decision modeling
    ✓ Explicit tradeoff consideration
    ✓ Multiple solution alternatives
    ✓ Stakeholder preference integration
    ✓ Better than single-objective
    ✓ Transparent decision criteria
    ✓ Flexible priority adjustment
    ✓ Pareto-optimal solutions
    
    USE CASES:
    • Resource allocation with constraints
    • Project planning (cost/time/quality)
    • Route planning (time/cost/comfort)
    • Product design tradeoffs
    • Investment decisions (risk/return)
    • Scheduling (efficiency/fairness)
    • ML model selection (accuracy/interpretability)
    • System design (performance/cost/reliability)
    
    OBJECTIVE TYPES:
    → Maximize: Quality, performance, revenue
    → Minimize: Cost, time, risk, resource usage
    → Target: Hit specific value
    → Constraint: Must satisfy (hard requirement)
    
    PLANNING APPROACHES:
    → Weighted Sum: Combine with weights
    → Lexicographic: Priority ordering
    → Pareto Optimal: Non-dominated set
    → Satisficing: Good enough
    → Goal Programming: Target achievement
    
    BEST PRACTICES:
    1. Define clear, measurable objectives
    2. Implement efficient Pareto filtering
    3. Provide tradeoff visualizations
    4. Allow dynamic weight adjustment
    5. Cache evaluated plans for reuse
    6. Support constraint hierarchies
    7. Enable sensitivity analysis
    8. Log decision rationale
    
    TRADE-OFFS:
    • Thoroughness vs. computational cost
    • Number of objectives vs. complexity
    • Weight precision vs. simplicity
    • Exact vs. approximate solutions
    
    PRODUCTION CONSIDERATIONS:
    → Normalize objectives to comparable scales
    → Implement efficient Pareto algorithms
    → Provide interactive weight tuning
    → Visualize tradeoff spaces (2D/3D plots)
    → Cache evaluation results
    → Support what-if analysis
    → Enable constraint relaxation
    → Log all evaluation details
    → Provide explanation of tradeoffs
    → Support group decision making
    
    This pattern enables agents to make realistic decisions when
    multiple conflicting objectives must be balanced, providing
    transparency into tradeoffs and multiple solution alternatives.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_multi_objective_planning()
