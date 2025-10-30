"""
Contingency Planning Pattern

Enables agents to create backup plans and alternative strategies for handling
unexpected events, failures, or changing conditions.

Key Concepts:
- Backup plans and alternatives
- Risk assessment and mitigation
- Trigger conditions
- What-if analysis
- Dynamic plan switching

Use Cases:
- Business continuity planning
- Mission-critical systems
- Disaster recovery
- Project risk management
- Adaptive task execution
"""

from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContingencyStatus(Enum):
    """Status of a contingency plan."""
    INACTIVE = "inactive"
    MONITORING = "monitoring"
    TRIGGERED = "triggered"
    ACTIVATED = "activated"
    COMPLETED = "completed"
    FAILED = "failed"


class EventType(Enum):
    """Types of events that can trigger contingencies."""
    FAILURE = "failure"
    DELAY = "delay"
    RESOURCE_SHORTAGE = "resource_shortage"
    QUALITY_ISSUE = "quality_issue"
    EXTERNAL_CHANGE = "external_change"
    COST_OVERRUN = "cost_overrun"
    DEPENDENCY_FAILURE = "dependency_failure"


@dataclass
class Risk:
    """A potential risk that may require contingency planning."""
    risk_id: str
    name: str
    description: str
    probability: float  # 0-1
    impact: float  # 0-1
    event_type: EventType
    risk_level: RiskLevel = RiskLevel.LOW
    
    def __post_init__(self):
        """Calculate risk level based on probability and impact."""
        risk_score = self.probability * self.impact
        
        if risk_score >= 0.7:
            self.risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.5:
            self.risk_level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            self.risk_level = RiskLevel.MEDIUM
        else:
            self.risk_level = RiskLevel.LOW
    
    def calculate_expected_loss(self, baseline_value: float) -> float:
        """Calculate expected loss if risk materializes."""
        return baseline_value * self.impact * self.probability


@dataclass
class TriggerCondition:
    """Condition that triggers a contingency plan."""
    condition_id: str
    name: str
    condition_func: Callable[[Dict[str, Any]], bool]
    description: str
    monitored_variables: List[str] = field(default_factory=list)


@dataclass
class Action:
    """An action in a plan."""
    action_id: str
    name: str
    description: str
    duration: float
    cost: float
    success_probability: float = 1.0
    resources_required: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Plan:
    """A primary or contingency plan."""
    plan_id: str
    name: str
    description: str
    actions: List[Action]
    expected_duration: float = 0.0
    expected_cost: float = 0.0
    success_probability: float = 1.0
    
    def __post_init__(self):
        """Calculate plan metrics."""
        if self.actions:
            self.expected_duration = sum(a.duration for a in self.actions)
            self.expected_cost = sum(a.cost for a in self.actions)
            self.success_probability = 1.0
            for action in self.actions:
                self.success_probability *= action.success_probability


@dataclass
class ContingencyPlan:
    """A contingency plan with triggers and activation logic."""
    contingency_id: str
    name: str
    description: str
    risk: Risk
    triggers: List[TriggerCondition]
    backup_plan: Plan
    status: ContingencyStatus = ContingencyStatus.INACTIVE
    priority: int = 1  # Higher = more important
    created_at: datetime = field(default_factory=datetime.now)
    activated_at: Optional[datetime] = None
    
    def evaluate_triggers(self, context: Dict[str, Any]) -> bool:
        """Evaluate if any trigger conditions are met."""
        for trigger in self.triggers:
            try:
                if trigger.condition_func(context):
                    return True
            except Exception as e:
                print(f"Error evaluating trigger {trigger.name}: {e}")
        return False


@dataclass
class Scenario:
    """A what-if scenario for analysis."""
    scenario_id: str
    name: str
    description: str
    probability: float
    changes: Dict[str, Any]  # Changes to apply to baseline
    risks_triggered: List[str] = field(default_factory=list)


class ContingencyPlanner:
    """Planner that creates and manages contingency plans."""
    
    def __init__(self, planner_id: str, name: str):
        self.planner_id = planner_id
        self.name = name
        self.primary_plan: Optional[Plan] = None
        self.contingencies: Dict[str, ContingencyPlan] = {}
        self.identified_risks: Dict[str, Risk] = {}
        self.scenarios: Dict[str, Scenario] = {}
        self.current_context: Dict[str, Any] = {}
        self.active_contingencies: List[str] = []
    
    def set_primary_plan(self, plan: Plan) -> None:
        """Set the primary plan."""
        self.primary_plan = plan
        print(f"[{self.name}] Set primary plan: {plan.name}")
        print(f"  Expected duration: {plan.expected_duration}")
        print(f"  Expected cost: ${plan.expected_cost}")
        print(f"  Success probability: {plan.success_probability:.2%}")
    
    def identify_risk(self, risk: Risk) -> None:
        """Identify a potential risk."""
        self.identified_risks[risk.risk_id] = risk
        print(f"[{self.name}] Identified risk: {risk.name}")
        print(f"  Level: {risk.risk_level.value}")
        print(f"  Probability: {risk.probability:.2%}, Impact: {risk.impact:.2%}")
    
    def create_contingency(
        self,
        name: str,
        description: str,
        risk: Risk,
        triggers: List[TriggerCondition],
        backup_plan: Plan,
        priority: int = 1
    ) -> ContingencyPlan:
        """Create a contingency plan."""
        contingency = ContingencyPlan(
            contingency_id=str(uuid.uuid4()),
            name=name,
            description=description,
            risk=risk,
            triggers=triggers,
            backup_plan=backup_plan,
            priority=priority
        )
        
        self.contingencies[contingency.contingency_id] = contingency
        contingency.status = ContingencyStatus.MONITORING
        
        print(f"[{self.name}] Created contingency: {name}")
        print(f"  Risk: {risk.name}")
        print(f"  Triggers: {len(triggers)}")
        print(f"  Priority: {priority}")
        
        return contingency
    
    def monitor_conditions(self, context: Dict[str, Any]) -> List[ContingencyPlan]:
        """Monitor for trigger conditions."""
        self.current_context = context
        triggered = []
        
        for contingency in self.contingencies.values():
            if contingency.status != ContingencyStatus.MONITORING:
                continue
            
            if contingency.evaluate_triggers(context):
                contingency.status = ContingencyStatus.TRIGGERED
                triggered.append(contingency)
                print(f"\n[{self.name}] TRIGGER: Contingency '{contingency.name}' triggered!")
                print(f"  Risk: {contingency.risk.name}")
        
        return triggered
    
    def activate_contingency(self, contingency_id: str) -> bool:
        """Activate a contingency plan."""
        if contingency_id not in self.contingencies:
            return False
        
        contingency = self.contingencies[contingency_id]
        
        if contingency.status != ContingencyStatus.TRIGGERED:
            print(f"[{self.name}] Cannot activate - not triggered: {contingency.name}")
            return False
        
        contingency.status = ContingencyStatus.ACTIVATED
        contingency.activated_at = datetime.now()
        self.active_contingencies.append(contingency_id)
        
        print(f"\n[{self.name}] ACTIVATED: Contingency '{contingency.name}'")
        print(f"  Backup plan: {contingency.backup_plan.name}")
        print(f"  Actions: {len(contingency.backup_plan.actions)}")
        
        return True
    
    def analyze_scenario(self, scenario: Scenario) -> Dict[str, Any]:
        """Analyze a what-if scenario."""
        print(f"\n[{self.name}] Analyzing scenario: {scenario.name}")
        print(f"  Probability: {scenario.probability:.2%}")
        
        # Determine which risks are triggered
        triggered_risks = []
        for risk_id in scenario.risks_triggered:
            if risk_id in self.identified_risks:
                triggered_risks.append(self.identified_risks[risk_id])
        
        # Find applicable contingencies
        applicable_contingencies = []
        for contingency in self.contingencies.values():
            if contingency.risk.risk_id in scenario.risks_triggered:
                applicable_contingencies.append(contingency)
        
        # Calculate total expected cost
        total_cost = self.primary_plan.expected_cost if self.primary_plan else 0.0
        
        for contingency in applicable_contingencies:
            expected_contingency_cost = (
                contingency.backup_plan.expected_cost *
                contingency.risk.probability
            )
            total_cost += expected_contingency_cost
        
        # Calculate total expected duration
        total_duration = self.primary_plan.expected_duration if self.primary_plan else 0.0
        
        for contingency in applicable_contingencies:
            expected_contingency_duration = (
                contingency.backup_plan.expected_duration *
                contingency.risk.probability
            )
            total_duration += expected_contingency_duration
        
        analysis = {
            "scenario": scenario.name,
            "probability": scenario.probability,
            "triggered_risks": len(triggered_risks),
            "applicable_contingencies": len(applicable_contingencies),
            "expected_total_cost": total_cost,
            "expected_total_duration": total_duration,
            "risks": [r.name for r in triggered_risks],
            "contingencies": [c.name for c in applicable_contingencies]
        }
        
        print(f"  Triggered risks: {len(triggered_risks)}")
        print(f"  Applicable contingencies: {len(applicable_contingencies)}")
        print(f"  Expected total cost: ${analysis['expected_total_cost']:.2f}")
        print(f"  Expected total duration: {analysis['expected_total_duration']:.1f}")
        
        return analysis
    
    def run_monte_carlo_simulation(
        self,
        scenarios: List[Scenario],
        num_iterations: int = 1000
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation across scenarios."""
        print(f"\n[{self.name}] Running Monte Carlo simulation ({num_iterations} iterations)...")
        
        import random
        
        costs = []
        durations = []
        contingency_activations = {c.contingency_id: 0 for c in self.contingencies.values()}
        
        for _ in range(num_iterations):
            # Sample a scenario based on probabilities
            rand = random.random()
            cumulative_prob = 0.0
            selected_scenario = None
            
            for scenario in scenarios:
                cumulative_prob += scenario.probability
                if rand <= cumulative_prob:
                    selected_scenario = scenario
                    break
            
            if not selected_scenario:
                selected_scenario = scenarios[-1]
            
            # Calculate cost and duration for this iteration
            cost = self.primary_plan.expected_cost if self.primary_plan else 0.0
            duration = self.primary_plan.expected_duration if self.primary_plan else 0.0
            
            # Add contingency costs
            for contingency in self.contingencies.values():
                if contingency.risk.risk_id in selected_scenario.risks_triggered:
                    if random.random() < contingency.risk.probability:
                        cost += contingency.backup_plan.expected_cost
                        duration += contingency.backup_plan.expected_duration
                        contingency_activations[contingency.contingency_id] += 1
            
            costs.append(cost)
            durations.append(duration)
        
        # Calculate statistics
        results = {
            "iterations": num_iterations,
            "cost_mean": sum(costs) / len(costs),
            "cost_min": min(costs),
            "cost_max": max(costs),
            "cost_std": self._calculate_std(costs),
            "duration_mean": sum(durations) / len(durations),
            "duration_min": min(durations),
            "duration_max": max(durations),
            "duration_std": self._calculate_std(durations),
            "contingency_activation_rates": {
                cid: count / num_iterations
                for cid, count in contingency_activations.items()
            }
        }
        
        print(f"\nSimulation Results:")
        print(f"  Cost: ${results['cost_mean']:.2f} ± ${results['cost_std']:.2f}")
        print(f"  Cost range: ${results['cost_min']:.2f} - ${results['cost_max']:.2f}")
        print(f"  Duration: {results['duration_mean']:.1f} ± {results['duration_std']:.1f}")
        print(f"  Duration range: {results['duration_min']:.1f} - {results['duration_max']:.1f}")
        
        return results
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def get_contingency_summary(self) -> Dict[str, Any]:
        """Get summary of all contingencies."""
        summary = {
            "total_contingencies": len(self.contingencies),
            "by_status": {},
            "by_risk_level": {},
            "active": len(self.active_contingencies)
        }
        
        for contingency in self.contingencies.values():
            # Count by status
            status = contingency.status.value
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1
            
            # Count by risk level
            level = contingency.risk.risk_level.value
            summary["by_risk_level"][level] = summary["by_risk_level"].get(level, 0) + 1
        
        return summary


def demonstrate_contingency_planning():
    """Demonstrate contingency planning pattern."""
    print("=" * 60)
    print("CONTINGENCY PLANNING DEMONSTRATION")
    print("=" * 60)
    
    # Create planner
    planner = ContingencyPlanner("planner1", "Project Contingency Planner")
    
    # Define primary plan
    print("\n" + "=" * 60)
    print("1. Setting Primary Plan")
    print("=" * 60)
    
    primary_actions = [
        Action("a1", "Requirements Analysis", "Gather requirements", duration=5, cost=5000, success_probability=0.95),
        Action("a2", "Design", "Create system design", duration=10, cost=10000, success_probability=0.90),
        Action("a3", "Implementation", "Build system", duration=20, cost=20000, success_probability=0.85),
        Action("a4", "Testing", "Test system", duration=10, cost=8000, success_probability=0.90),
        Action("a5", "Deployment", "Deploy to production", duration=5, cost=5000, success_probability=0.95),
    ]
    
    primary_plan = Plan(
        plan_id="primary",
        name="Standard Development Plan",
        description="Standard software development approach",
        actions=primary_actions
    )
    
    planner.set_primary_plan(primary_plan)
    
    # Identify risks
    print("\n" + "=" * 60)
    print("2. Identifying Risks")
    print("=" * 60)
    
    risk1 = Risk(
        risk_id="r1",
        name="Key Developer Unavailable",
        description="Lead developer becomes unavailable",
        probability=0.15,
        impact=0.7,
        event_type=EventType.RESOURCE_SHORTAGE
    )
    
    risk2 = Risk(
        risk_id="r2",
        name="Requirements Change",
        description="Major requirements change during development",
        probability=0.30,
        impact=0.6,
        event_type=EventType.EXTERNAL_CHANGE
    )
    
    risk3 = Risk(
        risk_id="r3",
        name="Integration Failure",
        description="Third-party integration fails",
        probability=0.20,
        impact=0.8,
        event_type=EventType.FAILURE
    )
    
    planner.identify_risk(risk1)
    planner.identify_risk(risk2)
    planner.identify_risk(risk3)
    
    # Create contingency plans
    print("\n" + "=" * 60)
    print("3. Creating Contingency Plans")
    print("=" * 60)
    
    # Contingency 1: Backup developer
    trigger1 = TriggerCondition(
        condition_id="t1",
        name="Developer availability check",
        condition_func=lambda ctx: ctx.get("developer_available", True) == False,
        description="Check if key developer is available"
    )
    
    backup_plan1 = Plan(
        plan_id="backup1",
        name="Hire Contractor",
        description="Bring in external contractor",
        actions=[
            Action("b1", "Find Contractor", "Search for qualified contractor", duration=3, cost=3000),
            Action("b2", "Onboard Contractor", "Train and onboard", duration=5, cost=5000),
            Action("b3", "Continue Development", "Resume with contractor", duration=22, cost=25000),
        ]
    )
    
    contingency1 = planner.create_contingency(
        name="Backup Developer Plan",
        description="Hire contractor if key developer unavailable",
        risk=risk1,
        triggers=[trigger1],
        backup_plan=backup_plan1,
        priority=2
    )
    
    # Contingency 2: Agile adaptation
    trigger2 = TriggerCondition(
        condition_id="t2",
        name="Requirements change detection",
        condition_func=lambda ctx: ctx.get("requirements_changed", False) == True,
        description="Check if requirements have changed"
    )
    
    backup_plan2 = Plan(
        plan_id="backup2",
        name="Agile Pivot",
        description="Adapt to new requirements",
        actions=[
            Action("b4", "Re-analyze Requirements", "Update requirements", duration=3, cost=3000),
            Action("b5", "Adjust Design", "Modify design", duration=5, cost=5000),
            Action("b6", "Incremental Development", "Develop in sprints", duration=15, cost=15000),
        ]
    )
    
    contingency2 = planner.create_contingency(
        name="Agile Adaptation Plan",
        description="Pivot to agile approach if requirements change",
        risk=risk2,
        triggers=[trigger2],
        backup_plan=backup_plan2,
        priority=3
    )
    
    # Contingency 3: Alternative integration
    trigger3 = TriggerCondition(
        condition_id="t3",
        name="Integration failure detection",
        condition_func=lambda ctx: ctx.get("integration_success", True) == False,
        description="Check if integration succeeded"
    )
    
    backup_plan3 = Plan(
        plan_id="backup3",
        name="Alternative Integration",
        description="Use alternative API or build custom integration",
        actions=[
            Action("b7", "Evaluate Alternatives", "Research other options", duration=2, cost=2000),
            Action("b8", "Build Custom Integration", "Develop custom solution", duration=8, cost=10000),
            Action("b9", "Test Integration", "Verify new integration", duration=5, cost=5000),
        ]
    )
    
    contingency3 = planner.create_contingency(
        name="Alternative Integration Plan",
        description="Switch to alternative if primary integration fails",
        risk=risk3,
        triggers=[trigger3],
        backup_plan=backup_plan3,
        priority=1
    )
    
    # Monitor conditions
    print("\n" + "=" * 60)
    print("4. Monitoring Conditions")
    print("=" * 60)
    
    # Scenario A: All going well
    context_a = {
        "developer_available": True,
        "requirements_changed": False,
        "integration_success": True
    }
    
    print("\nScenario A: Normal execution")
    triggered_a = planner.monitor_conditions(context_a)
    print(f"  Triggered contingencies: {len(triggered_a)}")
    
    # Scenario B: Integration fails
    context_b = {
        "developer_available": True,
        "requirements_changed": False,
        "integration_success": False
    }
    
    print("\nScenario B: Integration failure")
    triggered_b = planner.monitor_conditions(context_b)
    print(f"  Triggered contingencies: {len(triggered_b)}")
    
    if triggered_b:
        planner.activate_contingency(triggered_b[0].contingency_id)
    
    # What-if analysis
    print("\n" + "=" * 60)
    print("5. What-If Analysis")
    print("=" * 60)
    
    scenario1 = Scenario(
        scenario_id="s1",
        name="Best Case",
        description="Everything goes according to plan",
        probability=0.50,
        changes={},
        risks_triggered=[]
    )
    
    scenario2 = Scenario(
        scenario_id="s2",
        name="Moderate Issues",
        description="Some issues require contingency activation",
        probability=0.35,
        changes={},
        risks_triggered=["r2"]  # Requirements change
    )
    
    scenario3 = Scenario(
        scenario_id="s3",
        name="Major Issues",
        description="Multiple contingencies needed",
        probability=0.15,
        changes={},
        risks_triggered=["r1", "r3"]  # Developer + integration issues
    )
    
    analysis1 = planner.analyze_scenario(scenario1)
    analysis2 = planner.analyze_scenario(scenario2)
    analysis3 = planner.analyze_scenario(scenario3)
    
    # Monte Carlo simulation
    print("\n" + "=" * 60)
    print("6. Monte Carlo Simulation")
    print("=" * 60)
    
    simulation_results = planner.run_monte_carlo_simulation(
        [scenario1, scenario2, scenario3],
        num_iterations=1000
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("7. Contingency Summary")
    print("=" * 60)
    
    summary = planner.get_contingency_summary()
    
    print(f"\nTotal contingencies: {summary['total_contingencies']}")
    print(f"Active contingencies: {summary['active']}")
    print(f"\nBy status:")
    for status, count in summary['by_status'].items():
        print(f"  {status}: {count}")
    print(f"\nBy risk level:")
    for level, count in summary['by_risk_level'].items():
        print(f"  {level}: {count}")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_contingency_planning()
