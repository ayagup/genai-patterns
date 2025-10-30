"""
Agentic AI Design Pattern: World Model Learning

This pattern implements world model learning where an agent builds an internal
model of its environment to predict future states, plan actions, and improve
decision-making through imagination and simulation.

Key Concepts:
1. State Prediction: Predict future environment states
2. Dynamics Modeling: Learn how actions affect the environment
3. Model-Based Planning: Use the model for lookahead planning
4. Imagination Rollouts: Simulate trajectories mentally
5. Model Uncertainty: Track model confidence and adapt

Use Cases:
- Model-based reinforcement learning
- Strategic planning with limited interaction
- Sample-efficient learning
- Simulation-based decision making
- Robotics with expensive real-world trials

Note: This is a simplified implementation demonstrating core concepts.
In production, you'd typically use neural networks for the world model.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import random
import math
import uuid


class ActionType(Enum):
    """Types of actions in the environment"""
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    INTERACT = "interact"
    WAIT = "wait"


@dataclass
class State:
    """Represents an environment state"""
    position: Tuple[int, int]
    inventory: List[str] = field(default_factory=list)
    energy: float = 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.position, tuple(sorted(self.inventory)), self.energy))
    
    def copy(self) -> 'State':
        """Create a copy of the state"""
        return State(
            position=self.position,
            inventory=self.inventory.copy(),
            energy=self.energy,
            metadata=self.metadata.copy()
        )


@dataclass
class Transition:
    """Represents a state transition"""
    state: State
    action: ActionType
    next_state: State
    reward: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_features(self) -> Tuple:
        """Extract features for learning"""
        return (
            self.state.position,
            self.action,
            len(self.state.inventory),
            self.state.energy
        )


@dataclass
class ModelPrediction:
    """Prediction from the world model"""
    predicted_state: State
    predicted_reward: float
    confidence: float  # 0-1
    variance: float = 0.0  # Uncertainty measure


class WorldModel:
    """Learns dynamics of the environment"""
    
    def __init__(self):
        # Transition model: (state, action) -> next_state
        self.transition_counts: Dict[Tuple, Dict[int, int]] = {}
        self.transition_totals: Dict[Tuple, int] = {}
        
        # Reward model: (state, action) -> reward
        self.reward_sums: Dict[Tuple, float] = {}
        self.reward_counts: Dict[Tuple, int] = {}
        
        # Track prediction accuracy
        self.predictions_made = 0
        self.predictions_correct = 0
    
    def learn(self, transition: Transition) -> None:
        """Learn from an observed transition"""
        features = transition.get_features()
        
        # Update transition model
        if features not in self.transition_counts:
            self.transition_counts[features] = {}
            self.transition_totals[features] = 0
        
        next_state_key = hash(transition.next_state)
        self.transition_counts[features][next_state_key] = \
            self.transition_counts[features].get(next_state_key, 0) + 1
        self.transition_totals[features] += 1
        
        # Update reward model
        if features not in self.reward_sums:
            self.reward_sums[features] = 0.0
            self.reward_counts[features] = 0
        
        self.reward_sums[features] += transition.reward
        self.reward_counts[features] += 1
    
    def predict(self, state: State, action: ActionType) -> ModelPrediction:
        """Predict next state and reward"""
        features = (state.position, action, len(state.inventory), state.energy)
        
        # Predict next state (use most common transition)
        if features in self.transition_counts:
            next_states = self.transition_counts[features]
            most_common = max(next_states.items(), key=lambda x: x[1])
            
            # Calculate confidence based on consistency
            total = self.transition_totals[features]
            confidence = most_common[1] / total if total > 0 else 0.0
            
            # Create predicted state (simplified - reuse state structure)
            predicted_state = state.copy()
            predicted_state.energy -= 1.0  # Basic energy decay
            
            # Apply action effects
            if action == ActionType.MOVE_UP:
                predicted_state.position = (state.position[0], state.position[1] + 1)
            elif action == ActionType.MOVE_DOWN:
                predicted_state.position = (state.position[0], state.position[1] - 1)
            elif action == ActionType.MOVE_LEFT:
                predicted_state.position = (state.position[0] - 1, state.position[1])
            elif action == ActionType.MOVE_RIGHT:
                predicted_state.position = (state.position[0] + 1, state.position[1])
        else:
            # No data - low confidence prediction
            predicted_state = state.copy()
            confidence = 0.1
        
        # Predict reward
        if features in self.reward_sums:
            predicted_reward = self.reward_sums[features] / self.reward_counts[features]
        else:
            predicted_reward = 0.0
        
        # Calculate variance (uncertainty)
        variance = 1.0 - confidence
        
        return ModelPrediction(
            predicted_state=predicted_state,
            predicted_reward=predicted_reward,
            confidence=confidence,
            variance=variance
        )
    
    def get_accuracy(self) -> float:
        """Get prediction accuracy"""
        if self.predictions_made == 0:
            return 0.0
        return self.predictions_correct / self.predictions_made


@dataclass
class ImaginaryTrajectory:
    """A simulated trajectory using the world model"""
    trajectory_id: str
    states: List[State]
    actions: List[ActionType]
    rewards: List[float]
    total_reward: float
    average_confidence: float
    
    def __post_init__(self):
        if not self.trajectory_id:
            self.trajectory_id = str(uuid.uuid4())


class ImaginationEngine:
    """Uses world model to imagine possible futures"""
    
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.trajectories_generated = 0
    
    def imagine_trajectory(self, initial_state: State, 
                          actions: List[ActionType],
                          max_steps: Optional[int] = None) -> ImaginaryTrajectory:
        """Imagine a trajectory by simulating actions"""
        if max_steps is None:
            max_steps = len(actions)
        
        states = [initial_state]
        rewards = []
        confidences = []
        current_state = initial_state.copy()
        
        for i, action in enumerate(actions[:max_steps]):
            # Predict next state
            prediction = self.world_model.predict(current_state, action)
            
            states.append(prediction.predicted_state)
            rewards.append(prediction.predicted_reward)
            confidences.append(prediction.confidence)
            
            current_state = prediction.predicted_state.copy()
            
            # Stop if energy depleted
            if current_state.energy <= 0:
                break
        
        self.trajectories_generated += 1
        
        return ImaginaryTrajectory(
            trajectory_id=str(uuid.uuid4()),
            states=states,
            actions=actions[:len(states)-1],
            rewards=rewards,
            total_reward=sum(rewards),
            average_confidence=sum(confidences) / len(confidences) if confidences else 0.0
        )
    
    def sample_trajectories(self, initial_state: State, 
                           action_candidates: List[ActionType],
                           num_samples: int = 5,
                           horizon: int = 5) -> List[ImaginaryTrajectory]:
        """Sample multiple possible trajectories"""
        trajectories = []
        
        for _ in range(num_samples):
            # Random action sequence
            actions = [random.choice(action_candidates) for _ in range(horizon)]
            trajectory = self.imagine_trajectory(initial_state, actions, horizon)
            trajectories.append(trajectory)
        
        return trajectories
    
    def best_trajectory(self, trajectories: List[ImaginaryTrajectory],
                       confidence_weight: float = 0.3) -> ImaginaryTrajectory:
        """Select best trajectory considering reward and confidence"""
        def score(traj: ImaginaryTrajectory) -> float:
            reward_score = traj.total_reward
            confidence_score = traj.average_confidence * 10  # Scale up
            return (reward_score * (1 - confidence_weight) + 
                   confidence_score * confidence_weight)
        
        return max(trajectories, key=score)


class ModelBasedPlanner:
    """Plans actions using the world model"""
    
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.imagination_engine = ImaginationEngine(world_model)
        self.plans_generated = 0
    
    def plan_actions(self, current_state: State, 
                    goal: Optional[Dict[str, Any]] = None,
                    horizon: int = 5,
                    num_samples: int = 10) -> List[ActionType]:
        """Plan best sequence of actions"""
        # Available actions
        action_candidates = [
            ActionType.MOVE_UP,
            ActionType.MOVE_DOWN,
            ActionType.MOVE_LEFT,
            ActionType.MOVE_RIGHT,
            ActionType.INTERACT,
            ActionType.WAIT
        ]
        
        # Sample multiple trajectories
        trajectories = self.imagination_engine.sample_trajectories(
            current_state,
            action_candidates,
            num_samples=num_samples,
            horizon=horizon
        )
        
        # Select best trajectory
        best = self.imagination_engine.best_trajectory(trajectories)
        
        self.plans_generated += 1
        
        return best.actions
    
    def evaluate_action(self, state: State, action: ActionType) -> float:
        """Evaluate single action value"""
        prediction = self.world_model.predict(state, action)
        
        # Value combines reward and confidence
        value = prediction.predicted_reward * prediction.confidence
        
        return value


class WorldModelAgent:
    """Agent that learns and uses a world model"""
    
    def __init__(self):
        self.world_model = WorldModel()
        self.planner = ModelBasedPlanner(self.world_model)
        self.experience_buffer: List[Transition] = []
        self.current_plan: List[ActionType] = []
        self.plan_index = 0
    
    def observe_transition(self, transition: Transition) -> None:
        """Learn from observed transition"""
        self.world_model.learn(transition)
        self.experience_buffer.append(transition)
    
    def act(self, current_state: State, replan: bool = False) -> ActionType:
        """Select action based on world model"""
        # Replan if needed or no plan exists
        if replan or not self.current_plan or self.plan_index >= len(self.current_plan):
            self.current_plan = self.planner.plan_actions(current_state)
            self.plan_index = 0
        
        # Execute next action from plan
        if self.plan_index < len(self.current_plan):
            action = self.current_plan[self.plan_index]
            self.plan_index += 1
            return action
        
        return ActionType.WAIT
    
    def get_model_quality(self) -> Dict[str, Any]:
        """Get metrics about model quality"""
        return {
            "transitions_learned": len(self.experience_buffer),
            "model_accuracy": self.world_model.get_accuracy(),
            "trajectories_imagined": self.planner.imagination_engine.trajectories_generated,
            "plans_generated": self.planner.plans_generated
        }


def demonstrate_world_model_learning():
    """Demonstrate world model learning"""
    print("=" * 70)
    print("WORLD MODEL LEARNING DEMONSTRATION")
    print("=" * 70)
    
    # Create agent
    agent = WorldModelAgent()
    
    print("\n1. LEARNING PHASE: COLLECTING EXPERIENCE")
    print("-" * 70)
    
    # Simulate collecting experience
    print("   Simulating 20 transitions in environment...")
    
    states_visited = []
    for i in range(20):
        # Create sample transitions
        state = State(
            position=(random.randint(0, 5), random.randint(0, 5)),
            inventory=[],
            energy=random.uniform(50, 100)
        )
        
        action = random.choice(list(ActionType))
        
        # Simulate next state
        next_state = state.copy()
        if action == ActionType.MOVE_UP:
            next_state.position = (state.position[0], state.position[1] + 1)
            reward = -1.0
        elif action == ActionType.MOVE_DOWN:
            next_state.position = (state.position[0], state.position[1] - 1)
            reward = -1.0
        elif action == ActionType.MOVE_LEFT:
            next_state.position = (state.position[0] - 1, state.position[1])
            reward = -1.0
        elif action == ActionType.MOVE_RIGHT:
            next_state.position = (state.position[0] + 1, state.position[1])
            reward = -1.0
        elif action == ActionType.INTERACT:
            reward = 10.0
        else:
            reward = 0.0
        
        next_state.energy -= 1.0
        
        transition = Transition(state, action, next_state, reward)
        agent.observe_transition(transition)
        
        states_visited.append(state.position)
        
        if i % 5 == 0:
            print(f"     Transition {i+1}: {state.position} --[{action.value}]--> "
                  f"{next_state.position} (reward: {reward:.1f})")
    
    print(f"\n   Total transitions learned: {len(agent.experience_buffer)}")
    
    print("\n2. WORLD MODEL PREDICTIONS")
    print("-" * 70)
    
    # Test predictions
    test_state = State(position=(2, 2), inventory=[], energy=80.0)
    
    print(f"   Current state: position={test_state.position}, energy={test_state.energy:.1f}")
    print("\n   Predictions for different actions:")
    
    for action in [ActionType.MOVE_UP, ActionType.MOVE_RIGHT, ActionType.INTERACT]:
        prediction = agent.world_model.predict(test_state, action)
        print(f"\n     Action: {action.value}")
        print(f"       Next position: {prediction.predicted_state.position}")
        print(f"       Expected reward: {prediction.predicted_reward:.2f}")
        print(f"       Confidence: {prediction.confidence:.2f}")
        print(f"       Uncertainty: {prediction.variance:.2f}")
    
    print("\n3. IMAGINATION ROLLOUTS")
    print("-" * 70)
    
    # Imagine trajectories
    imagination = ImaginationEngine(agent.world_model)
    
    action_sequence = [
        ActionType.MOVE_RIGHT,
        ActionType.MOVE_UP,
        ActionType.INTERACT,
        ActionType.MOVE_LEFT
    ]
    
    print(f"   Imagining trajectory with actions: {[a.value for a in action_sequence]}")
    
    trajectory = imagination.imagine_trajectory(test_state, action_sequence)
    
    print(f"\n   Imagined trajectory (ID: {trajectory.trajectory_id[:8]}):")
    print(f"     States visited: {len(trajectory.states)}")
    print(f"     Total reward: {trajectory.total_reward:.2f}")
    print(f"     Average confidence: {trajectory.average_confidence:.2f}")
    
    print("\n   Step-by-step imagination:")
    for i, (state, action, reward) in enumerate(zip(
        trajectory.states[:-1], trajectory.actions, trajectory.rewards
    )):
        print(f"     Step {i+1}: {state.position} --[{action.value}]--> "
              f"reward: {reward:.1f}")
    
    print("\n4. SAMPLING MULTIPLE TRAJECTORIES")
    print("-" * 70)
    
    num_samples = 5
    trajectories = imagination.sample_trajectories(
        test_state,
        list(ActionType),
        num_samples=num_samples,
        horizon=4
    )
    
    print(f"   Sampled {num_samples} random trajectories:")
    for i, traj in enumerate(trajectories, 1):
        print(f"\n     Trajectory {i}:")
        print(f"       Actions: {[a.value for a in traj.actions]}")
        print(f"       Total reward: {traj.total_reward:.2f}")
        print(f"       Confidence: {traj.average_confidence:.2f}")
    
    # Find best
    best = imagination.best_trajectory(trajectories)
    print(f"\n   Best trajectory: {[a.value for a in best.actions]}")
    print(f"     Reward: {best.total_reward:.2f}")
    
    print("\n5. MODEL-BASED PLANNING")
    print("-" * 70)
    
    # Plan using model
    print("   Planning 5-step action sequence...")
    
    planned_actions = agent.planner.plan_actions(
        test_state,
        horizon=5,
        num_samples=10
    )
    
    print(f"\n   Planned actions: {[a.value for a in planned_actions]}")
    
    # Simulate execution
    print("\n   Simulating plan execution:")
    simulated_state = test_state.copy()
    
    for i, action in enumerate(planned_actions, 1):
        prediction = agent.world_model.predict(simulated_state, action)
        print(f"     Step {i}: {action.value} -> "
              f"pos={prediction.predicted_state.position}, "
              f"reward={prediction.predicted_reward:.1f}, "
              f"conf={prediction.confidence:.2f}")
        simulated_state = prediction.predicted_state
    
    print("\n6. MODEL QUALITY METRICS")
    print("-" * 70)
    
    metrics = agent.get_model_quality()
    print(f"   Transitions learned: {metrics['transitions_learned']}")
    print(f"   Trajectories imagined: {metrics['trajectories_imagined']}")
    print(f"   Plans generated: {metrics['plans_generated']}")
    
    print("\n7. ADAPTIVE REPLANNING")
    print("-" * 70)
    
    # Show replanning with new information
    current = State(position=(3, 3), inventory=[], energy=60.0)
    
    print(f"   Initial state: {current.position}")
    print("   Generating initial plan...")
    
    action1 = agent.act(current)
    print(f"     Action 1: {action1.value}")
    
    action2 = agent.act(current)
    print(f"     Action 2: {action2.value}")
    
    # Replan
    print("\n   New information received - replanning...")
    action3 = agent.act(current, replan=True)
    print(f"     New action: {action3.value}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("1. Learning environment dynamics from experience")
    print("2. Predicting next states and rewards")
    print("3. Confidence and uncertainty estimation")
    print("4. Imagination rollouts using the model")
    print("5. Sampling multiple possible trajectories")
    print("6. Model-based planning and action selection")
    print("7. Adaptive replanning with new information")
    print("8. Model quality tracking and metrics")


if __name__ == "__main__":
    demonstrate_world_model_learning()
