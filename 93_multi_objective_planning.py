"""
Multi-Objective Planning Pattern

Enables agents to plan while balancing multiple competing objectives
(e.g., cost, time, quality, risk) through Pareto optimization and weighted scoring.

Key Concepts:
- Multiple optimization objectives
- Pareto optimality
- Trade-off analysis
- Weighted objectives
- Multi-criteria decision making

Use Cases:
- Resource allocation with constraints
- Project planning
- Route optimization
- Investment strategies
- Design optimization
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import math


class ObjectiveType(Enum):
    """Types of objectives."""
    MINIMIZE = "minimize"  # Lower is better
    MAXIMIZE = "maximize"  # Higher is better


class PlanStatus(Enum):
    """Status of a plan."""
    CANDIDATE = "candidate"
    FEASIBLE = "feasible"
    PARETO_OPTIMAL = "pareto_optimal"
    SELECTED = "selected"
    REJECTED = "rejected"


@dataclass
class Objective:
    """An optimization objective."""
    objective_id: str
    name: str
    objective_type: ObjectiveType
    weight: float = 1.0  # Importance weight
    target: Optional[float] = None  # Target value if known
    constraint: Optional[Tuple[float, float]] = None  # (min, max) bounds
    
    def normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-1 range."""
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        
        # Flip for minimize objectives (lower is better)
        if self.objective_type == ObjectiveType.MINIMIZE:
            normalized = 1.0 - normalized
        
        return max(0.0, min(1.0, normalized))


@dataclass
class Action:
    """An action in a plan."""
    action_id: str
    name: str
    duration: float
    cost: float
    quality: float  # 0-1
    risk: float  # 0-1
    resources: Dict[str, float] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    effects: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """A plan consisting of actions."""
    plan_id: str
    name: str
    actions: List[Action]
    objective_scores: Dict[str, float] = field(default_factory=dict)
    weighted_score: float = 0.0
    status: PlanStatus = PlanStatus.CANDIDATE
    created_at: datetime = field(default_factory=datetime.now)
    
    def calculate_objective_scores(self) -> Dict[str, float]:
        """Calculate scores for common objectives."""
        if not self.actions:
            return {}
        
        scores = {
            "total_cost": sum(a.cost for a in self.actions),
            "total_duration": sum(a.duration for a in self.actions),
            "average_quality": sum(a.quality for a in self.actions) / len(self.actions),
            "average_risk": sum(a.risk for a in self.actions) / len(self.actions),
            "num_actions": len(self.actions)
        }
        
        self.objective_scores = scores
        return scores
    
    def dominates(self, other: 'Plan') -> bool:
        """Check if this plan Pareto dominates another plan."""
        if not self.objective_scores or not other.objective_scores:
            return False
        
        # This plan must be at least as good in all objectives
        # and strictly better in at least one
        better_in_some = False
        
        for obj_name in self.objective_scores:
            if obj_name not in other.objective_scores:
                continue
            
            self_score = self.objective_scores[obj_name]
            other_score = other.objective_scores[obj_name]
            
            # Assuming higher is better for normalized scores
            if self_score < other_score:
                return False
            elif self_score > other_score:
                better_in_some = True
        
        return better_in_some


class MultiObjectivePlanner:
    """Planner that optimizes for multiple objectives simultaneously."""
    
    def __init__(self, planner_id: str, name: str):
        self.planner_id = planner_id
        self.name = name
        self.objectives: Dict[str, Objective] = {}
        self.available_actions: Dict[str, Action] = {}
        self.generated_plans: List[Plan] = []
        self.pareto_frontier: List[Plan] = []
    
    def add_objective(self, objective: Objective) -> None:
        """Add an optimization objective."""
        self.objectives[objective.objective_id] = objective
        print(f"[{self.name}] Added objective: {objective.name} ({objective.objective_type.value})")
    
    def add_action(self, action: Action) -> None:
        """Add available action."""
        self.available_actions[action.action_id] = action
    
    def generate_plans(
        self,
        goal: str,
        max_plans: int = 10,
        max_actions: int = 5
    ) -> List[Plan]:
        """Generate candidate plans."""
        print(f"\n[{self.name}] Generating plans for goal: {goal}")
        
        plans = []
        
        # Strategy 1: Random combinations
        for i in range(max_plans // 2):
            actions = self._generate_random_plan(max_actions)
            if actions:
                plan = Plan(
                    plan_id=str(uuid.uuid4()),
                    name=f"Plan-Random-{i+1}",
                    actions=actions
                )
                plans.append(plan)
        
        # Strategy 2: Greedy optimization for each objective
        for obj_id, objective in self.objectives.items():
            actions = self._generate_greedy_plan(objective, max_actions)
            if actions:
                plan = Plan(
                    plan_id=str(uuid.uuid4()),
                    name=f"Plan-{objective.name}-Optimized",
                    actions=actions
                )
                plans.append(plan)
        
        # Calculate objective scores for all plans
        for plan in plans:
            plan.calculate_objective_scores()
            plan.status = PlanStatus.FEASIBLE
        
        self.generated_plans = plans
        print(f"Generated {len(plans)} candidate plans")
        
        return plans
    
    def _generate_random_plan(self, max_actions: int) -> List[Action]:
        """Generate a random plan."""
        import random
        num_actions = random.randint(1, max_actions)
        actions = random.sample(
            list(self.available_actions.values()),
            min(num_actions, len(self.available_actions))
        )
        return actions
    
    def _generate_greedy_plan(
        self,
        objective: Objective,
        max_actions: int
    ) -> List[Action]:
        """Generate plan optimized for specific objective."""
        import random
        
        actions = []
        available = list(self.available_actions.values())
        
        # Sort actions based on objective
        if objective.name == "cost":
            available.sort(key=lambda a: a.cost)
        elif objective.name == "time":
            available.sort(key=lambda a: a.duration)
        elif objective.name == "quality":
            available.sort(key=lambda a: a.quality, reverse=True)
        elif objective.name == "risk":
            available.sort(key=lambda a: a.risk)
        
        # Select top actions
        actions = available[:min(max_actions, len(available))]
        return actions
    
    def find_pareto_frontier(self) -> List[Plan]:
        """Find Pareto optimal plans."""
        print(f"\n[{self.name}] Finding Pareto frontier...")
        
        if not self.generated_plans:
            return []
        
        # Normalize objective scores
        self._normalize_scores()
        
        # Find non-dominated plans
        pareto_optimal = []
        
        for plan in self.generated_plans:
            is_dominated = False
            
            for other_plan in self.generated_plans:
                if plan.plan_id == other_plan.plan_id:
                    continue
                
                if other_plan.dominates(plan):
                    is_dominated = True
                    break
            
            if not is_dominated:
                plan.status = PlanStatus.PARETO_OPTIMAL
                pareto_optimal.append(plan)
        
        self.pareto_frontier = pareto_optimal
        
        print(f"Found {len(pareto_optimal)} Pareto optimal plans")
        return pareto_optimal
    
    def _normalize_scores(self) -> None:
        """Normalize objective scores across all plans."""
        if not self.generated_plans:
            return
        
        # Find min/max for each objective
        score_ranges = {}
        
        for obj_name in self.generated_plans[0].objective_scores.keys():
            values = [p.objective_scores[obj_name] for p in self.generated_plans]
            score_ranges[obj_name] = (min(values), max(values))
        
        # Normalize each plan's scores
        for plan in self.generated_plans:
            normalized = {}
            
            for obj_name, score in plan.objective_scores.items():
                min_val, max_val = score_ranges[obj_name]
                
                # Determine if higher or lower is better
                if obj_name in ["total_cost", "total_duration", "average_risk"]:
                    # Lower is better - flip normalization
                    if max_val != min_val:
                        normalized[obj_name] = 1.0 - (score - min_val) / (max_val - min_val)
                    else:
                        normalized[obj_name] = 0.5
                else:
                    # Higher is better
                    if max_val != min_val:
                        normalized[obj_name] = (score - min_val) / (max_val - min_val)
                    else:
                        normalized[obj_name] = 0.5
            
            plan.objective_scores = normalized
    
    def select_plan(
        self,
        preference_weights: Optional[Dict[str, float]] = None
    ) -> Optional[Plan]:
        """Select best plan based on weighted objectives."""
        if not self.pareto_frontier:
            self.find_pareto_frontier()
        
        if not self.pareto_frontier:
            return None
        
        print(f"\n[{self.name}] Selecting best plan...")
        
        # Use provided weights or equal weights
        if not preference_weights:
            preference_weights = {
                obj_name: 1.0 / len(self.objectives)
                for obj_name in self.objectives
            }
        
        # Score each plan
        for plan in self.pareto_frontier:
            weighted_score = 0.0
            
            for obj_name, weight in preference_weights.items():
                # Map objective name to score key
                score_key = self._map_objective_to_score(obj_name)
                if score_key in plan.objective_scores:
                    weighted_score += weight * plan.objective_scores[score_key]
            
            plan.weighted_score = weighted_score
        
        # Select highest scoring plan
        best_plan = max(self.pareto_frontier, key=lambda p: p.weighted_score)
        best_plan.status = PlanStatus.SELECTED
        
        print(f"Selected plan: {best_plan.name} (score: {best_plan.weighted_score:.3f})")
        
        return best_plan
    
    def _map_objective_to_score(self, obj_name: str) -> str:
        """Map objective name to score key."""
        mapping = {
            "cost": "total_cost",
            "time": "total_duration",
            "duration": "total_duration",
            "quality": "average_quality",
            "risk": "average_risk"
        }
        return mapping.get(obj_name, obj_name)
    
    def analyze_tradeoffs(self) -> Dict[str, Any]:
        """Analyze trade-offs in Pareto frontier."""
        if not self.pareto_frontier:
            return {}
        
        print(f"\n[{self.name}] Analyzing trade-offs...")
        
        analysis = {
            "num_pareto_optimal": len(self.pareto_frontier),
            "objectives": {},
            "correlations": {}
        }
        
        # Analyze each objective
        for obj_name in self.pareto_frontier[0].objective_scores.keys():
            scores = [p.objective_scores[obj_name] for p in self.pareto_frontier]
            
            analysis["objectives"][obj_name] = {
                "min": min(scores),
                "max": max(scores),
                "range": max(scores) - min(scores),
                "mean": sum(scores) / len(scores)
            }
        
        # Analyze correlations between objectives
        obj_names = list(self.pareto_frontier[0].objective_scores.keys())
        for i, obj1 in enumerate(obj_names):
            for obj2 in obj_names[i+1:]:
                correlation = self._calculate_correlation(obj1, obj2)
                analysis["correlations"][f"{obj1}_vs_{obj2}"] = correlation
        
        return analysis
    
    def _calculate_correlation(self, obj1: str, obj2: str) -> float:
        """Calculate correlation between two objectives."""
        if not self.pareto_frontier:
            return 0.0
        
        values1 = [p.objective_scores[obj1] for p in self.pareto_frontier]
        values2 = [p.objective_scores[obj2] for p in self.pareto_frontier]
        
        if len(values1) < 2:
            return 0.0
        
        # Simple correlation coefficient
        mean1 = sum(values1) / len(values1)
        mean2 = sum(values2) / len(values2)
        
        numerator = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2))
        denom1 = sum((v1 - mean1) ** 2 for v1 in values1)
        denom2 = sum((v2 - mean2) ** 2 for v2 in values2)
        
        if denom1 == 0 or denom2 == 0:
            return 0.0
        
        correlation = numerator / (math.sqrt(denom1 * denom2))
        return correlation


def demonstrate_multi_objective_planning():
    """Demonstrate multi-objective planning pattern."""
    print("=" * 60)
    print("MULTI-OBJECTIVE PLANNING DEMONSTRATION")
    print("=" * 60)
    
    # Create planner
    planner = MultiObjectivePlanner("planner1", "Project Planner")
    
    # Define objectives
    print("\n" + "=" * 60)
    print("1. Defining Objectives")
    print("=" * 60)
    
    cost_obj = Objective(
        objective_id="cost",
        name="cost",
        objective_type=ObjectiveType.MINIMIZE,
        weight=0.3
    )
    
    time_obj = Objective(
        objective_id="time",
        name="time",
        objective_type=ObjectiveType.MINIMIZE,
        weight=0.3
    )
    
    quality_obj = Objective(
        objective_id="quality",
        name="quality",
        objective_type=ObjectiveType.MAXIMIZE,
        weight=0.25
    )
    
    risk_obj = Objective(
        objective_id="risk",
        name="risk",
        objective_type=ObjectiveType.MINIMIZE,
        weight=0.15
    )
    
    planner.add_objective(cost_obj)
    planner.add_objective(time_obj)
    planner.add_objective(quality_obj)
    planner.add_objective(risk_obj)
    
    # Add available actions
    print("\n" + "=" * 60)
    print("2. Adding Available Actions")
    print("=" * 60)
    
    actions = [
        Action("a1", "Quick Implementation", duration=10, cost=5000, quality=0.6, risk=0.3),
        Action("a2", "Standard Implementation", duration=20, cost=10000, quality=0.8, risk=0.2),
        Action("a3", "Premium Implementation", duration=30, cost=20000, quality=0.95, risk=0.1),
        Action("a4", "Basic Testing", duration=5, cost=2000, quality=0.7, risk=0.4),
        Action("a5", "Thorough Testing", duration=15, cost=5000, quality=0.9, risk=0.15),
        Action("a6", "Fast Deployment", duration=2, cost=1000, quality=0.65, risk=0.35),
        Action("a7", "Careful Deployment", duration=5, cost=2500, quality=0.85, risk=0.1),
    ]
    
    for action in actions:
        planner.add_action(action)
        print(f"  Added: {action.name} (duration={action.duration}, cost=${action.cost}, quality={action.quality:.2f}, risk={action.risk:.2f})")
    
    # Generate candidate plans
    print("\n" + "=" * 60)
    print("3. Generating Candidate Plans")
    print("=" * 60)
    
    plans = planner.generate_plans(
        goal="Complete project successfully",
        max_plans=12,
        max_actions=3
    )
    
    # Find Pareto frontier
    print("\n" + "=" * 60)
    print("4. Finding Pareto Optimal Plans")
    print("=" * 60)
    
    pareto_plans = planner.find_pareto_frontier()
    
    print(f"\nPareto Frontier ({len(pareto_plans)} plans):")
    for i, plan in enumerate(pareto_plans, 1):
        print(f"\n  {i}. {plan.name}")
        print(f"     Actions: {[a.name for a in plan.actions]}")
        print(f"     Scores (normalized):")
        for obj_name, score in sorted(plan.objective_scores.items()):
            print(f"       {obj_name}: {score:.3f}")
    
    # Analyze trade-offs
    print("\n" + "=" * 60)
    print("5. Trade-off Analysis")
    print("=" * 60)
    
    analysis = planner.analyze_tradeoffs()
    
    print(f"\nObjective Ranges:")
    for obj_name, stats in analysis["objectives"].items():
        print(f"  {obj_name}:")
        print(f"    Range: {stats['min']:.3f} - {stats['max']:.3f}")
        print(f"    Spread: {stats['range']:.3f}")
    
    print(f"\nObjective Correlations:")
    for pair, corr in analysis["correlations"].items():
        print(f"  {pair}: {corr:+.3f}")
    
    # Select best plan with different preferences
    print("\n" + "=" * 60)
    print("6. Plan Selection with Different Preferences")
    print("=" * 60)
    
    # Scenario 1: Cost-focused
    print("\nScenario 1: Cost-focused (weight=0.5)")
    best_cost = planner.select_plan({"cost": 0.5, "time": 0.2, "quality": 0.2, "risk": 0.1})
    if best_cost:
        print(f"  Selected: {best_cost.name}")
        print(f"  Actions: {[a.name for a in best_cost.actions]}")
    
    # Scenario 2: Quality-focused
    print("\nScenario 2: Quality-focused (weight=0.5)")
    best_quality = planner.select_plan({"cost": 0.15, "time": 0.15, "quality": 0.5, "risk": 0.2})
    if best_quality:
        print(f"  Selected: {best_quality.name}")
        print(f"  Actions: {[a.name for a in best_quality.actions]}")
    
    # Scenario 3: Balanced
    print("\nScenario 3: Balanced (equal weights)")
    best_balanced = planner.select_plan({"cost": 0.25, "time": 0.25, "quality": 0.25, "risk": 0.25})
    if best_balanced:
        print(f"  Selected: {best_balanced.name}")
        print(f"  Actions: {[a.name for a in best_balanced.actions]}")
    
    # Summary
    print("\n" + "=" * 60)
    print("7. Summary")
    print("=" * 60)
    
    print(f"\nGenerated {len(plans)} candidate plans")
    print(f"Identified {len(pareto_plans)} Pareto optimal plans")
    print(f"Demonstrated trade-off analysis")
    print(f"Showed preference-based selection")


if __name__ == "__main__":
    demonstrate_multi_objective_planning()
