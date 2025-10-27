"""
Probabilistic Planning Pattern

Enables agents to plan under uncertainty by reasoning about probabilistic
action outcomes and maintaining belief states over possible world states.

Key Concepts:
- Belief states (probability distributions)
- Stochastic actions
- Expected value optimization
- Uncertainty propagation
- Risk-aware planning

Use Cases:
- Robotics under uncertainty
- Financial planning
- Weather-dependent operations
- Unreliable environments
- Partially observable systems
"""

from typing import List, Dict, Any, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
import math
import random


class StateType(Enum):
    """Types of state representations."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class ActionOutcome(Enum):
    """Possible action outcomes."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"


@dataclass
class State:
    """A state in the world."""
    state_id: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, conditions: Dict[str, Any]) -> bool:
        """Check if state matches given conditions."""
        for key, value in conditions.items():
            if key not in self.properties or self.properties[key] != value:
                return False
        return True


@dataclass
class BeliefState:
    """Probability distribution over possible states."""
    belief_id: str
    name: str
    state_probabilities: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Normalize probabilities."""
        self._normalize()
    
    def _normalize(self) -> None:
        """Normalize probabilities to sum to 1."""
        total = sum(self.state_probabilities.values())
        if total > 0:
            self.state_probabilities = {
                state_id: prob / total
                for state_id, prob in self.state_probabilities.items()
            }
    
    def most_likely_state(self) -> Optional[str]:
        """Get the most likely state."""
        if not self.state_probabilities:
            return None
        return max(self.state_probabilities.items(), key=lambda x: x[1])[0]
    
    def entropy(self) -> float:
        """Calculate entropy (uncertainty) of belief state."""
        entropy = 0.0
        for prob in self.state_probabilities.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy
    
    def update(self, evidence: Dict[str, float]) -> None:
        """Update belief based on new evidence (Bayesian update)."""
        for state_id, likelihood in evidence.items():
            if state_id in self.state_probabilities:
                self.state_probabilities[state_id] *= likelihood
        
        self._normalize()


@dataclass
class ProbabilisticAction:
    """An action with probabilistic outcomes."""
    action_id: str
    name: str
    description: str
    cost: float
    duration: float
    # Outcomes: map from outcome to (probability, resulting_state_changes)
    outcomes: Dict[ActionOutcome, Tuple[float, Dict[str, Any]]] = field(default_factory=dict)
    preconditions: Dict[str, Any] = field(default_factory=dict)
    
    def expected_value(self, value_func: Callable[[Dict[str, Any]], float]) -> float:
        """Calculate expected value of action."""
        expected = 0.0
        for outcome, (prob, changes) in self.outcomes.items():
            value = value_func(changes)
            expected += prob * value
        return expected
    
    def is_applicable(self, state: State) -> bool:
        """Check if action is applicable in given state."""
        return state.matches(self.preconditions)
    
    def sample_outcome(self) -> Tuple[ActionOutcome, Dict[str, Any]]:
        """Sample a random outcome based on probabilities."""
        rand = random.random()
        cumulative = 0.0
        
        for outcome, (prob, changes) in self.outcomes.items():
            cumulative += prob
            if rand <= cumulative:
                return outcome, changes
        
        # Return last outcome if rounding error
        last_outcome, (_, last_changes) = list(self.outcomes.items())[-1]
        return last_outcome, last_changes


@dataclass
class Policy:
    """A policy mapping states to actions."""
    policy_id: str
    name: str
    state_action_map: Dict[str, str] = field(default_factory=dict)  # state_id -> action_id
    expected_value: float = 0.0


@dataclass
class PlanStep:
    """A step in a probabilistic plan."""
    step_id: str
    action: ProbabilisticAction
    belief_state: BeliefState
    expected_cost: float = 0.0
    expected_duration: float = 0.0


class ProbabilisticPlanner:
    """Planner that handles uncertainty in actions and states."""
    
    def __init__(self, planner_id: str, name: str):
        self.planner_id = planner_id
        self.name = name
        self.states: Dict[str, State] = {}
        self.actions: Dict[str, ProbabilisticAction] = {}
        self.current_belief: Optional[BeliefState] = None
        self.plans: List[List[PlanStep]] = []
        self.policies: Dict[str, Policy] = {}
    
    def add_state(self, state: State) -> None:
        """Add a possible state."""
        self.states[state.state_id] = state
    
    def add_action(self, action: ProbabilisticAction) -> None:
        """Add an action."""
        self.actions[action.action_id] = action
    
    def set_initial_belief(self, belief: BeliefState) -> None:
        """Set initial belief state."""
        self.current_belief = belief
        print(f"[{self.name}] Initial belief state: {belief.name}")
        print(f"  Entropy: {belief.entropy():.3f}")
        print(f"  Most likely state: {belief.most_likely_state()}")
    
    def predict_belief(
        self,
        belief: BeliefState,
        action: ProbabilisticAction
    ) -> BeliefState:
        """Predict belief state after taking action."""
        new_probabilities = {}
        
        # For each current state
        for current_state_id, current_prob in belief.state_probabilities.items():
            if current_state_id not in self.states:
                continue
            
            current_state = self.states[current_state_id]
            
            # Check if action is applicable
            if not action.is_applicable(current_state):
                # State unchanged if action not applicable
                new_probabilities[current_state_id] = (
                    new_probabilities.get(current_state_id, 0.0) + current_prob
                )
                continue
            
            # For each possible outcome
            for outcome, (outcome_prob, changes) in action.outcomes.items():
                # Apply changes to get new state
                new_state_id = self._apply_changes(current_state_id, changes)
                
                # Update probability
                prob_contribution = current_prob * outcome_prob
                new_probabilities[new_state_id] = (
                    new_probabilities.get(new_state_id, 0.0) + prob_contribution
                )
        
        new_belief = BeliefState(
            belief_id=str(uuid.uuid4()),
            name=f"After {action.name}",
            state_probabilities=new_probabilities
        )
        
        return new_belief
    
    def _apply_changes(self, state_id: str, changes: Dict[str, Any]) -> str:
        """Apply changes to a state (simplified - returns state_id)."""
        # In real implementation, would create/find new state with changes applied
        # For demonstration, we use a simple hashing approach
        
        if not changes:
            return state_id
        
        # Create identifier for new state based on changes
        change_str = "_".join(f"{k}={v}" for k, v in sorted(changes.items()))
        return f"{state_id}_{change_str}"
    
    def plan_with_expectation(
        self,
        goal_conditions: Dict[str, Any],
        max_steps: int = 10,
        discount_factor: float = 0.95
    ) -> List[PlanStep]:
        """Plan by maximizing expected value."""
        print(f"\n[{self.name}] Planning with expectation maximization...")
        print(f"  Goal: {goal_conditions}")
        
        if not self.current_belief:
            print("  Error: No initial belief state set")
            return []
        
        plan = []
        belief: BeliefState = self.current_belief
        
        for step_num in range(max_steps):
            # Find best action based on expected value
            best_action = None
            best_value = float('-inf')
            best_next_belief = None
            
            for action in self.actions.values():
                # Predict next belief state
                next_belief = self.predict_belief(belief, action)
                
                # Calculate expected value
                value = self._evaluate_belief(next_belief, goal_conditions)
                value -= action.cost  # Subtract cost
                value *= (discount_factor ** step_num)  # Apply discount
                
                if value > best_value:
                    best_value = value
                    best_action = action
                    best_next_belief = next_belief
            
            if not best_action or not best_next_belief:
                break
            
            # Add step to plan
            step = PlanStep(
                step_id=str(uuid.uuid4()),
                action=best_action,
                belief_state=belief,
                expected_cost=best_action.cost,
                expected_duration=best_action.duration
            )
            plan.append(step)
            
            print(f"  Step {step_num + 1}: {best_action.name} (value: {best_value:.3f})")
            
            # Check if goal reached
            if self._check_goal(best_next_belief, goal_conditions):
                print(f"  Goal reached in {len(plan)} steps!")
                break
            
            # Update belief for next iteration
            belief = best_next_belief
        
        self.plans.append(plan)
        return plan
    
    def _evaluate_belief(
        self,
        belief: BeliefState,
        goal_conditions: Dict[str, Any]
    ) -> float:
        """Evaluate how good a belief state is relative to goal."""
        # Calculate probability of being in a goal state
        goal_prob = 0.0
        
        for state_id, prob in belief.state_probabilities.items():
            if state_id in self.states:
                state = self.states[state_id]
                if state.matches(goal_conditions):
                    goal_prob += prob
        
        return goal_prob
    
    def _check_goal(
        self,
        belief: BeliefState,
        goal_conditions: Dict[str, Any]
    ) -> bool:
        """Check if goal is reached with sufficient probability."""
        goal_prob = self._evaluate_belief(belief, goal_conditions)
        return goal_prob >= 0.8  # 80% confidence threshold
    
    def simulate_plan(
        self,
        plan: List[PlanStep],
        num_simulations: int = 100
    ) -> Dict[str, Any]:
        """Simulate plan execution multiple times."""
        print(f"\n[{self.name}] Simulating plan ({num_simulations} runs)...")
        
        successes = 0
        total_costs = []
        total_durations = []
        
        for sim in range(num_simulations):
            cost = 0.0
            duration = 0.0
            success = True
            
            for step in plan:
                # Execute action
                outcome, changes = step.action.sample_outcome()
                cost += step.action.cost
                duration += step.action.duration
                
                if outcome == ActionOutcome.FAILURE:
                    success = False
                    break
            
            if success:
                successes += 1
            
            total_costs.append(cost)
            total_durations.append(duration)
        
        success_rate = successes / num_simulations
        
        results = {
            "num_simulations": num_simulations,
            "success_rate": success_rate,
            "avg_cost": sum(total_costs) / len(total_costs),
            "min_cost": min(total_costs),
            "max_cost": max(total_costs),
            "avg_duration": sum(total_durations) / len(total_durations),
            "min_duration": min(total_durations),
            "max_duration": max(total_durations)
        }
        
        print(f"\nSimulation Results:")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Average cost: ${results['avg_cost']:.2f}")
        print(f"  Cost range: ${results['min_cost']:.2f} - ${results['max_cost']:.2f}")
        print(f"  Average duration: {results['avg_duration']:.1f}")
        print(f"  Duration range: {results['min_duration']:.1f} - {results['max_duration']:.1f}")
        
        return results
    
    def compute_value_iteration(
        self,
        goal_conditions: Dict[str, Any],
        max_iterations: int = 100,
        discount_factor: float = 0.95,
        convergence_threshold: float = 0.01
    ) -> Policy:
        """Compute optimal policy using value iteration."""
        print(f"\n[{self.name}] Computing optimal policy (value iteration)...")
        
        # Initialize values
        values = {state_id: 0.0 for state_id in self.states}
        
        # Set goal states to high value
        for state_id, state in self.states.items():
            if state.matches(goal_conditions):
                values[state_id] = 100.0
        
        # Value iteration
        for iteration in range(max_iterations):
            new_values = {}
            max_change = 0.0
            
            for state_id, state in self.states.items():
                if state.matches(goal_conditions):
                    new_values[state_id] = 100.0
                    continue
                
                # Find best action
                best_value = values[state_id]
                
                for action in self.actions.values():
                    if not action.is_applicable(state):
                        continue
                    
                    # Calculate expected value
                    expected = -action.cost
                    
                    for outcome, (prob, changes) in action.outcomes.items():
                        next_state_id = self._apply_changes(state_id, changes)
                        if next_state_id in values:
                            expected += prob * discount_factor * values[next_state_id]
                    
                    best_value = max(best_value, expected)
                
                new_values[state_id] = best_value
                max_change = max(max_change, abs(new_values[state_id] - values[state_id]))
            
            values = new_values
            
            if max_change < convergence_threshold:
                print(f"  Converged after {iteration + 1} iterations")
                break
        
        # Extract policy
        policy = Policy(
            policy_id=str(uuid.uuid4()),
            name="Optimal Policy",
            expected_value=sum(values.values()) / len(values)
        )
        
        for state_id, state in self.states.items():
            if state.matches(goal_conditions):
                continue
            
            best_action = None
            best_value = float('-inf')
            
            for action in self.actions.values():
                if not action.is_applicable(state):
                    continue
                
                expected = -action.cost
                for outcome, (prob, changes) in action.outcomes.items():
                    next_state_id = self._apply_changes(state_id, changes)
                    if next_state_id in values:
                        expected += prob * discount_factor * values[next_state_id]
                
                if expected > best_value:
                    best_value = expected
                    best_action = action
            
            if best_action:
                policy.state_action_map[state_id] = best_action.action_id
        
        self.policies[policy.policy_id] = policy
        
        print(f"  Policy contains {len(policy.state_action_map)} state-action mappings")
        print(f"  Expected value: {policy.expected_value:.3f}")
        
        return policy


def demonstrate_probabilistic_planning():
    """Demonstrate probabilistic planning pattern."""
    print("=" * 60)
    print("PROBABILISTIC PLANNING DEMONSTRATION")
    print("=" * 60)
    
    # Create planner
    planner = ProbabilisticPlanner("planner1", "Delivery Robot Planner")
    
    # Define states
    print("\n" + "=" * 60)
    print("1. Defining States")
    print("=" * 60)
    
    states = [
        State("s1", "At Warehouse", {"location": "warehouse", "has_package": False}),
        State("s2", "At Warehouse with Package", {"location": "warehouse", "has_package": True}),
        State("s3", "In Transit", {"location": "transit", "has_package": True}),
        State("s4", "At Destination", {"location": "destination", "has_package": True}),
        State("s5", "Delivered", {"location": "destination", "has_package": False}),
    ]
    
    for state in states:
        planner.add_state(state)
        print(f"  Added state: {state.name}")
    
    # Define probabilistic actions
    print("\n" + "=" * 60)
    print("2. Defining Probabilistic Actions")
    print("=" * 60)
    
    # Action: Pick up package
    pickup_action = ProbabilisticAction(
        action_id="a1",
        name="Pick Up Package",
        description="Pick up package from warehouse",
        cost=5.0,
        duration=5.0,
        outcomes={
            ActionOutcome.SUCCESS: (0.95, {"has_package": True}),
            ActionOutcome.FAILURE: (0.05, {})
        },
        preconditions={"location": "warehouse", "has_package": False}
    )
    
    # Action: Drive to destination
    drive_action = ProbabilisticAction(
        action_id="a2",
        name="Drive to Destination",
        description="Drive to delivery destination",
        cost=20.0,
        duration=30.0,
        outcomes={
            ActionOutcome.SUCCESS: (0.80, {"location": "destination"}),
            ActionOutcome.PARTIAL_SUCCESS: (0.15, {"location": "transit"}),  # Traffic delay
            ActionOutcome.FAILURE: (0.05, {})  # Breakdown
        },
        preconditions={"has_package": True}
    )
    
    # Action: Deliver package
    deliver_action = ProbabilisticAction(
        action_id="a3",
        name="Deliver Package",
        description="Deliver package to customer",
        cost=5.0,
        duration=5.0,
        outcomes={
            ActionOutcome.SUCCESS: (0.90, {"has_package": False}),
            ActionOutcome.FAILURE: (0.10, {})  # Customer not home
        },
        preconditions={"location": "destination", "has_package": True}
    )
    
    # Action: Return to warehouse
    return_action = ProbabilisticAction(
        action_id="a4",
        name="Return to Warehouse",
        description="Return to warehouse",
        cost=20.0,
        duration=30.0,
        outcomes={
            ActionOutcome.SUCCESS: (0.85, {"location": "warehouse"}),
            ActionOutcome.PARTIAL_SUCCESS: (0.10, {"location": "transit"}),
            ActionOutcome.FAILURE: (0.05, {})
        },
        preconditions={}
    )
    
    planner.add_action(pickup_action)
    planner.add_action(drive_action)
    planner.add_action(deliver_action)
    planner.add_action(return_action)
    
    for action in [pickup_action, drive_action, deliver_action, return_action]:
        print(f"\n  {action.name}")
        print(f"    Cost: ${action.cost}, Duration: {action.duration}")
        print(f"    Outcomes:")
        for outcome, (prob, changes) in action.outcomes.items():
            print(f"      {outcome.value}: {prob:.0%} -> {changes}")
    
    # Set initial belief state
    print("\n" + "=" * 60)
    print("3. Setting Initial Belief State")
    print("=" * 60)
    
    initial_belief = BeliefState(
        belief_id="b1",
        name="Initial",
        state_probabilities={
            "s1": 1.0  # Certain we're at warehouse without package
        }
    )
    
    planner.set_initial_belief(initial_belief)
    
    # Plan with expectation maximization
    print("\n" + "=" * 60)
    print("4. Planning with Expectation Maximization")
    print("=" * 60)
    
    goal = {"location": "destination", "has_package": False}
    plan = planner.plan_with_expectation(goal, max_steps=5)
    
    print(f"\nGenerated plan with {len(plan)} steps:")
    total_cost = 0.0
    total_duration = 0.0
    
    for i, step in enumerate(plan, 1):
        print(f"\n  Step {i}: {step.action.name}")
        print(f"    Expected cost: ${step.expected_cost}")
        print(f"    Expected duration: {step.expected_duration}")
        total_cost += step.expected_cost
        total_duration += step.expected_duration
    
    print(f"\nTotal expected cost: ${total_cost}")
    print(f"Total expected duration: {total_duration}")
    
    # Simulate plan execution
    print("\n" + "=" * 60)
    print("5. Simulating Plan Execution")
    print("=" * 60)
    
    simulation_results = planner.simulate_plan(plan, num_simulations=100)
    
    # Compute optimal policy
    print("\n" + "=" * 60)
    print("6. Computing Optimal Policy")
    print("=" * 60)
    
    policy = planner.compute_value_iteration(goal, max_iterations=50)
    
    print(f"\nOptimal policy mappings:")
    for state_id, action_id in policy.state_action_map.items():
        if state_id in planner.states:
            state_name = planner.states[state_id].name
            action_name = planner.actions[action_id].name
            print(f"  {state_name} -> {action_name}")
    
    # Summary
    print("\n" + "=" * 60)
    print("7. Summary")
    print("=" * 60)
    
    print(f"\nStates: {len(planner.states)}")
    print(f"Actions: {len(planner.actions)}")
    print(f"Plans generated: {len(planner.plans)}")
    print(f"Policies computed: {len(planner.policies)}")
    print(f"\nDemonstrated:")
    print(f"  ✓ Belief state representation")
    print(f"  ✓ Probabilistic action outcomes")
    print(f"  ✓ Expected value planning")
    print(f"  ✓ Plan simulation")
    print(f"  ✓ Value iteration policy computation")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_probabilistic_planning()
