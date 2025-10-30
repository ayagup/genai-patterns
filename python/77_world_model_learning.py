"""
World Model Learning Pattern

Agents build internal models of their environment to predict future states
and plan actions. Enables model-based reasoning, sample-efficient learning,
and mental simulation of scenarios.

Use Cases:
- Model-based reinforcement learning
- Robotics and autonomous systems
- Game playing with planning
- Simulation-based decision making
- Predictive maintenance

Benefits:
- Sample efficiency (learn from fewer interactions)
- Planning capability without real execution
- Understanding of environment dynamics
- Risk assessment through simulation
- Transfer learning to similar environments
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import math


class ModelType(Enum):
    """Types of world models"""
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    LEARNED = "learned"
    HYBRID = "hybrid"


@dataclass
class State:
    """Environment state representation"""
    state_id: str
    features: Dict[str, Any]
    timestamp: int = 0
    
    def to_vector(self) -> List[float]:
        """Convert state to feature vector"""
        return [float(v) for v in self.features.values() if isinstance(v, (int, float))]


@dataclass
class Action:
    """Action that can be taken in environment"""
    action_id: str
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0


@dataclass
class Transition:
    """State transition record"""
    state: State
    action: Action
    next_state: State
    reward: float
    done: bool = False


@dataclass
class Prediction:
    """Model prediction"""
    predicted_state: State
    confidence: float
    reward_estimate: float
    uncertainty: float = 0.0


class WorldModel:
    """
    Base world model for environment dynamics
    """
    
    def __init__(self, model_type: ModelType = ModelType.LEARNED):
        self.model_type = model_type
        self.transitions: List[Transition] = []
        self.state_action_outcomes: Dict[Tuple[str, str], List[Transition]] = {}
        self.training_steps = 0
        
    def observe(self, transition: Transition) -> None:
        """Observe a transition from the environment"""
        self.transitions.append(transition)
        
        # Index by state-action pair
        key = (transition.state.state_id, transition.action.action_id)
        if key not in self.state_action_outcomes:
            self.state_action_outcomes[key] = []
        self.state_action_outcomes[key].append(transition)
    
    def predict(self, state: State, action: Action) -> Prediction:
        """Predict next state and reward"""
        raise NotImplementedError("Subclasses must implement predict")
    
    def train(self, num_steps: int = 1) -> Dict[str, float]:
        """Train the model on observed transitions"""
        raise NotImplementedError("Subclasses must implement train")
    
    def simulate_trajectory(
        self,
        initial_state: State,
        actions: List[Action],
        max_steps: int = 100
    ) -> List[Tuple[State, Action, Prediction]]:
        """Simulate a trajectory of actions"""
        trajectory = []
        current_state = initial_state
        
        for i, action in enumerate(actions):
            if i >= max_steps:
                break
            
            prediction = self.predict(current_state, action)
            trajectory.append((current_state, action, prediction))
            
            current_state = prediction.predicted_state
            
            if prediction.predicted_state.features.get('done', False):
                break
        
        return trajectory


class DeterministicWorldModel(WorldModel):
    """Simple deterministic world model"""
    
    def __init__(self):
        super().__init__(ModelType.DETERMINISTIC)
        self.dynamics: Dict[Tuple[str, str], Tuple[State, float]] = {}
    
    def predict(self, state: State, action: Action) -> Prediction:
        """Predict deterministically based on observed transitions"""
        key = (state.state_id, action.action_id)
        
        if key in self.dynamics:
            next_state, reward = self.dynamics[key]
            return Prediction(
                predicted_state=next_state,
                confidence=1.0,
                reward_estimate=reward,
                uncertainty=0.0
            )
        
        # No observation - return current state
        return Prediction(
            predicted_state=state,
            confidence=0.0,
            reward_estimate=0.0,
            uncertainty=1.0
        )
    
    def train(self, num_steps: int = 1) -> Dict[str, float]:
        """Learn deterministic mappings"""
        for transition in self.transitions[-num_steps:]:
            key = (transition.state.state_id, transition.action.action_id)
            self.dynamics[key] = (transition.next_state, transition.reward)
        
        self.training_steps += num_steps
        
        return {
            "training_steps": self.training_steps,
            "known_transitions": len(self.dynamics)
        }


class StochasticWorldModel(WorldModel):
    """Stochastic world model with uncertainty"""
    
    def __init__(self):
        super().__init__(ModelType.STOCHASTIC)
    
    def predict(self, state: State, action: Action) -> Prediction:
        """Predict with probability distribution over outcomes"""
        key = (state.state_id, action.action_id)
        
        if key not in self.state_action_outcomes:
            return Prediction(
                predicted_state=state,
                confidence=0.0,
                reward_estimate=0.0,
                uncertainty=1.0
            )
        
        # Get all observed outcomes
        outcomes = self.state_action_outcomes[key]
        
        if not outcomes:
            return Prediction(
                predicted_state=state,
                confidence=0.0,
                reward_estimate=0.0,
                uncertainty=1.0
            )
        
        # Sample from observed distribution
        transition = random.choice(outcomes)
        
        # Calculate statistics
        avg_reward = sum(t.reward for t in outcomes) / len(outcomes)
        confidence = min(len(outcomes) / 10.0, 1.0)  # More samples = more confidence
        
        # Estimate uncertainty
        reward_variance = sum((t.reward - avg_reward) ** 2 for t in outcomes) / len(outcomes)
        uncertainty = math.sqrt(reward_variance)
        
        return Prediction(
            predicted_state=transition.next_state,
            confidence=confidence,
            reward_estimate=avg_reward,
            uncertainty=uncertainty
        )
    
    def train(self, num_steps: int = 1) -> Dict[str, float]:
        """Update statistics from new transitions"""
        self.training_steps += num_steps
        
        unique_state_actions = len(self.state_action_outcomes)
        total_transitions = len(self.transitions)
        
        return {
            "training_steps": self.training_steps,
            "unique_state_actions": unique_state_actions,
            "total_transitions": total_transitions,
            "avg_samples_per_state_action": total_transitions / unique_state_actions if unique_state_actions > 0 else 0
        }


class ModelBasedPlanner:
    """Plans actions using world model"""
    
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.planning_depth = 5
        self.num_samples = 10
    
    def plan(
        self,
        current_state: State,
        available_actions: List[Action],
        goal_test: Callable[[State], bool]
    ) -> List[Action]:
        """Plan sequence of actions to reach goal"""
        best_plan = None
        best_value = float('-inf')
        
        # Try random shooting planning
        for _ in range(self.num_samples):
            plan = self._random_shooting(current_state, available_actions)
            value = self._evaluate_plan(current_state, plan, goal_test)
            
            if value > best_value:
                best_value = value
                best_plan = plan
        
        return best_plan if best_plan else []
    
    def _random_shooting(
        self,
        state: State,
        available_actions: List[Action]
    ) -> List[Action]:
        """Generate random action sequence"""
        return [random.choice(available_actions) for _ in range(self.planning_depth)]
    
    def _evaluate_plan(
        self,
        initial_state: State,
        actions: List[Action],
        goal_test: Callable[[State], bool]
    ) -> float:
        """Evaluate plan quality using world model"""
        trajectory = self.world_model.simulate_trajectory(initial_state, actions)
        
        total_reward = sum(pred.reward_estimate for _, _, pred in trajectory)
        
        # Bonus for reaching goal
        if trajectory:
            final_state = trajectory[-1][2].predicted_state
            if goal_test(final_state):
                total_reward += 100.0
        
        # Penalty for uncertainty
        avg_uncertainty = sum(pred.uncertainty for _, _, pred in trajectory) / len(trajectory) if trajectory else 1.0
        total_reward -= avg_uncertainty * 10.0
        
        return total_reward


class ImaginationAgent:
    """
    Agent that uses world model for mental simulation
    """
    
    def __init__(self, name: str = "Imagination Agent"):
        self.name = name
        self.world_model = StochasticWorldModel()
        self.planner = ModelBasedPlanner(self.world_model)
        self.experience_buffer: List[Transition] = []
        
        print(f"[Agent] Initialized: {name}")
        print(f"  World Model: {self.world_model.model_type.value}")
    
    def observe_transition(self, transition: Transition) -> None:
        """Observe and learn from environment transition"""
        self.experience_buffer.append(transition)
        self.world_model.observe(transition)
        
        print(f"\n[Observation] {transition.state.state_id} + {transition.action.action_type} → {transition.next_state.state_id}")
        print(f"  Reward: {transition.reward:.2f}")
    
    def train_model(self, batch_size: int = 10) -> Dict[str, Any]:
        """Train world model on recent experience"""
        if len(self.experience_buffer) < batch_size:
            return {"status": "insufficient_data", "data_count": len(self.experience_buffer)}
        
        print(f"\n[Training] World model on {batch_size} transitions")
        metrics = self.world_model.train(batch_size)
        
        print(f"  Training steps: {metrics['training_steps']}")
        print(f"  Known transitions: {metrics.get('unique_state_actions', 0)}")
        
        return metrics
    
    def imagine_scenario(
        self,
        initial_state: State,
        actions: List[Action]
    ) -> List[Tuple[State, Action, Prediction]]:
        """Mentally simulate scenario without environment interaction"""
        print(f"\n[Imagination] Simulating {len(actions)} actions")
        
        trajectory = self.world_model.simulate_trajectory(initial_state, actions)
        
        print(f"  Simulated {len(trajectory)} steps")
        for i, (state, action, prediction) in enumerate(trajectory):
            print(f"  Step {i+1}: {action.action_type} → Reward: {prediction.reward_estimate:.2f} (conf: {prediction.confidence:.2f})")
        
        return trajectory
    
    def plan_to_goal(
        self,
        current_state: State,
        available_actions: List[Action],
        goal_test: Callable[[State], bool]
    ) -> List[Action]:
        """Plan sequence of actions to reach goal"""
        print(f"\n[Planning] Finding path to goal")
        
        plan = self.planner.plan(current_state, available_actions, goal_test)
        
        print(f"  Generated plan with {len(plan)} actions")
        for i, action in enumerate(plan):
            print(f"  {i+1}. {action.action_type}")
        
        return plan
    
    def evaluate_action(
        self,
        state: State,
        action: Action
    ) -> Tuple[float, float]:
        """Evaluate action without executing it"""
        prediction = self.world_model.predict(state, action)
        
        return prediction.reward_estimate, prediction.confidence


def demonstrate_world_model_learning():
    """
    Demonstrate World Model Learning pattern
    """
    print("=" * 70)
    print("WORLD MODEL LEARNING DEMONSTRATION")
    print("=" * 70)
    
    # Example 1: Simple grid world
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Learning Grid World Dynamics")
    print("=" * 70)
    
    # Create agent
    agent = ImaginationAgent("Grid Navigator")
    
    # Define states and actions
    states = {
        'A': State('A', {'x': 0, 'y': 0}),
        'B': State('B', {'x': 1, 'y': 0}),
        'C': State('C', {'x': 2, 'y': 0}),
        'GOAL': State('GOAL', {'x': 2, 'y': 1, 'done': True})
    }
    
    actions = [
        Action('move_right', 'move', {'direction': 'right'}),
        Action('move_up', 'move', {'direction': 'up'}),
        Action('move_left', 'move', {'direction': 'left'}),
    ]
    
    # Simulate environment interactions
    transitions = [
        Transition(states['A'], actions[0], states['B'], -1.0),
        Transition(states['B'], actions[0], states['C'], -1.0),
        Transition(states['C'], actions[1], states['GOAL'], 10.0, done=True),
        Transition(states['A'], actions[0], states['B'], -1.0),  # Repeat for learning
        Transition(states['B'], actions[0], states['C'], -1.0),
    ]
    
    # Agent observes and learns
    for transition in transitions:
        agent.observe_transition(transition)
    
    # Train model
    agent.train_model(batch_size=5)
    
    # Example 2: Mental simulation
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Mental Simulation (Imagination)")
    print("=" * 70)
    
    # Imagine what would happen
    imagined_actions = [actions[0], actions[0], actions[1]]
    trajectory = agent.imagine_scenario(states['A'], imagined_actions)
    
    print(f"\nImagined total reward: {sum(p.reward_estimate for _, _, p in trajectory):.2f}")
    
    # Example 3: Model-based planning
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Planning with World Model")
    print("=" * 70)
    
    def goal_test(state: State) -> bool:
        return state.state_id == 'GOAL'
    
    # Plan to reach goal
    plan = agent.plan_to_goal(states['A'], actions, goal_test)
    
    # Simulate the plan
    if plan:
        print("\nSimulating planned actions:")
        sim_trajectory = agent.imagine_scenario(states['A'], plan)
        final_reward = sum(p.reward_estimate for _, _, p in sim_trajectory)
        print(f"Expected total reward: {final_reward:.2f}")
    
    # Example 4: Action evaluation
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Evaluating Actions Before Execution")
    print("=" * 70)
    
    print("\nEvaluating actions from state A:")
    for action in actions[:2]:
        reward, confidence = agent.evaluate_action(states['A'], action)
        print(f"  {action.action_type}: Expected reward = {reward:.2f}, Confidence = {confidence:.2f}")


def demonstrate_stochastic_model():
    """Demonstrate stochastic world model"""
    print("\n" + "=" * 70)
    print("STOCHASTIC WORLD MODEL")
    print("=" * 70)
    
    model = StochasticWorldModel()
    
    # Create states
    state_a = State('A', {'value': 1})
    state_b = State('B', {'value': 2})
    state_c = State('C', {'value': 3})
    
    action = Action('act', 'action')
    
    # Observe multiple outcomes for same state-action pair
    print("\nObserving stochastic transitions:")
    transitions = [
        Transition(state_a, action, state_b, 5.0),
        Transition(state_a, action, state_c, 3.0),
        Transition(state_a, action, state_b, 6.0),
        Transition(state_a, action, state_b, 4.0),
    ]
    
    for t in transitions:
        model.observe(t)
        print(f"  A + act → {t.next_state.state_id}, reward: {t.reward}")
    
    # Train model
    model.train(len(transitions))
    
    # Make predictions
    print("\nModel predictions:")
    for _ in range(3):
        pred = model.predict(state_a, action)
        print(f"  Predicted next state: {pred.predicted_state.state_id}")
        print(f"    Reward estimate: {pred.reward_estimate:.2f}")
        print(f"    Confidence: {pred.confidence:.2f}")
        print(f"    Uncertainty: {pred.uncertainty:.2f}")


if __name__ == "__main__":
    demonstrate_world_model_learning()
    demonstrate_stochastic_model()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. World models enable learning environment dynamics
2. Mental simulation allows planning without execution
3. Stochastic models capture uncertainty
4. Model-based planning is sample-efficient
5. Imagination reduces real-world interactions

Best Practices:
- Start with simple deterministic models
- Gradually add stochasticity as needed
- Balance model accuracy vs. complexity
- Use uncertainty estimates for safe planning
- Continuously update model with new observations
- Validate model predictions against reality
- Combine with model-free learning for robustness
- Use ensemble models for better predictions
    """)
