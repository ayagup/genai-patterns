"""
Pattern 062: World Model Learning

Description:
    World Model Learning involves an agent building an internal model of its
    environment's dynamics. The agent learns to predict future states given
    current states and actions, enabling model-based planning, simulation,
    and what-if analysis without direct environment interaction.

Components:
    1. State Representation: How the world is encoded
    2. Dynamics Model: Predicts next state from current state and action
    3. Reward Predictor: Estimates rewards for state-action pairs
    4. Experience Collection: Gathers real-world data
    5. Model Training: Updates predictions based on experience
    6. Planning Module: Uses model for decision-making

Use Cases:
    - Robotics simulation and planning
    - Game AI with environment modeling
    - Strategic decision-making
    - Risk assessment and scenario planning
    - Sample-efficient reinforcement learning
    - Counterfactual reasoning

LangChain Implementation:
    Implements world modeling using LLM-based state prediction, transition
    dynamics, and outcome forecasting for planning and decision-making.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class StateType(Enum):
    """Types of state representations"""
    DISCRETE = "discrete"  # Finite states
    CONTINUOUS = "continuous"  # Continuous values
    HYBRID = "hybrid"  # Mix of both
    SYMBOLIC = "symbolic"  # Symbolic/linguistic


@dataclass
class State:
    """World state representation"""
    state_id: str
    description: str
    variables: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_text(self) -> str:
        """Convert state to text"""
        text = f"State: {self.description}\n"
        if self.variables:
            text += "Variables:\n"
            for key, value in self.variables.items():
                text += f"  - {key}: {value}\n"
        return text


@dataclass
class Action:
    """Action that can be taken"""
    action_id: str
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_text(self) -> str:
        """Convert action to text"""
        text = f"Action: {self.name}"
        if self.parameters:
            params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
            text += f" ({params_str})"
        return text


@dataclass
class Transition:
    """State transition experience"""
    state: State
    action: Action
    next_state: State
    reward: float
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """Model prediction"""
    predicted_state: State
    confidence: float
    predicted_reward: float
    alternative_outcomes: List[Tuple[State, float]] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Result from simulation"""
    initial_state: State
    action_sequence: List[Action]
    predicted_trajectory: List[State]
    cumulative_reward: float
    success_probability: float


class WorldModel:
    """
    World model for environment dynamics.
    
    Features:
    1. State transition prediction
    2. Reward prediction
    3. Experience-based learning
    4. Multi-step simulation
    5. Uncertainty estimation
    """
    
    def __init__(self, domain: str = "general"):
        self.domain = domain
        self.experiences: List[Transition] = []
        
        # Model for dynamics prediction
        self.dynamics_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # Model for reward prediction
        self.reward_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2
        )
        
        # Model for uncertainty estimation
        self.uncertainty_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.4
        )
    
    def add_experience(self, transition: Transition):
        """Add experience to model's memory"""
        self.experiences.append(transition)
    
    def _get_relevant_experiences(
        self,
        state: State,
        action: Action,
        k: int = 3
    ) -> List[Transition]:
        """Retrieve relevant past experiences"""
        
        if not self.experiences:
            return []
        
        # Simple relevance: recent experiences are more relevant
        # In production, would use embedding similarity
        return self.experiences[-k:] if len(self.experiences) >= k else self.experiences
    
    def predict_next_state(
        self,
        current_state: State,
        action: Action
    ) -> Prediction:
        """Predict next state given current state and action"""
        
        # Retrieve relevant experiences
        relevant = self._get_relevant_experiences(current_state, action)
        
        # Build context from experiences
        context = ""
        if relevant:
            context = "Past Similar Experiences:\n"
            for i, exp in enumerate(relevant, 1):
                context += f"{i}. State: {exp.state.description}\n"
                context += f"   Action: {exp.action.name}\n"
                context += f"   Result: {exp.next_state.description}\n"
                context += f"   Reward: {exp.reward}\n\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a world model for {self.domain}.

Predict what happens when an action is taken in a given state.

{context}

Provide:
1. Description of the resulting state
2. Key variables that changed
3. Confidence (0.0-1.0) in prediction"""),
            ("user", """Current State:
{state}

Action Taken:
{action}

Predicted Next State:""")
        ])
        
        chain = prompt | self.dynamics_model | StrOutputParser()
        prediction_text = chain.invoke({
            "state": current_state.to_text(),
            "action": action.to_text()
        })
        
        # Parse prediction
        lines = prediction_text.strip().split('\n')
        description = lines[0] if lines else "Unknown state"
        
        # Extract confidence if mentioned
        confidence = 0.7  # Default
        for line in lines:
            if 'confidence' in line.lower():
                try:
                    confidence = float(''.join(c for c in line if c.isdigit() or c == '.'))
                    if confidence > 1.0:
                        confidence = confidence / 100.0
                    confidence = max(0.0, min(1.0, confidence))
                except:
                    pass
        
        # Predict reward
        predicted_reward = self._predict_reward(current_state, action, description)
        
        predicted_state = State(
            state_id=f"predicted_{int(time.time())}",
            description=description,
            variables={}
        )
        
        return Prediction(
            predicted_state=predicted_state,
            confidence=confidence,
            predicted_reward=predicted_reward
        )
    
    def _predict_reward(
        self,
        state: State,
        action: Action,
        outcome: str
    ) -> float:
        """Predict reward for state-action-outcome"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Estimate the reward/value for an outcome in {self.domain}.

Provide a number between -1.0 (very bad) and +1.0 (very good).
Consider: goal achievement, efficiency, safety, side effects.

Respond with just a number."""),
            ("user", """State: {state}
Action: {action}
Outcome: {outcome}

Reward:""")
        ])
        
        chain = prompt | self.reward_model | StrOutputParser()
        
        try:
            reward_str = chain.invoke({
                "state": state.description,
                "action": action.name,
                "outcome": outcome
            })
            reward = float(''.join(c for c in reward_str if c.isdigit() or c == '.' or c == '-'))
            return max(-1.0, min(1.0, reward))
        except:
            return 0.0
    
    def simulate_trajectory(
        self,
        initial_state: State,
        action_sequence: List[Action],
        max_steps: int = 10
    ) -> SimulationResult:
        """Simulate a trajectory through the world"""
        
        current_state = initial_state
        trajectory = [initial_state]
        cumulative_reward = 0.0
        confidences = []
        
        for i, action in enumerate(action_sequence[:max_steps]):
            # Predict next state
            prediction = self.predict_next_state(current_state, action)
            
            # Update trajectory
            trajectory.append(prediction.predicted_state)
            cumulative_reward += prediction.predicted_reward
            confidences.append(prediction.confidence)
            
            # Move to next state
            current_state = prediction.predicted_state
        
        # Success probability = average confidence
        success_prob = sum(confidences) / len(confidences) if confidences else 0.5
        
        return SimulationResult(
            initial_state=initial_state,
            action_sequence=action_sequence,
            predicted_trajectory=trajectory,
            cumulative_reward=cumulative_reward,
            success_probability=success_prob
        )
    
    def plan_with_model(
        self,
        current_state: State,
        goal: str,
        num_alternatives: int = 3
    ) -> List[SimulationResult]:
        """Use model for planning"""
        
        # Generate alternative action sequences
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are planning in {self.domain}.

Generate {num_alternatives} different action sequences to achieve the goal.

For each plan, list 3-5 actions.

Format:
Plan 1:
1. [action]
2. [action]
...

Plan 2:
1. [action]
..."""),
            ("user", """Current State: {state}

Goal: {goal}

Action Plans:""")
        ])
        
        chain = planning_prompt | self.dynamics_model | StrOutputParser()
        plans_text = chain.invoke({
            "state": current_state.to_text(),
            "goal": goal
        })
        
        # Parse action sequences
        action_sequences = []
        current_sequence = []
        
        for line in plans_text.split('\n'):
            line = line.strip()
            
            if line.startswith('Plan'):
                if current_sequence:
                    action_sequences.append(current_sequence)
                    current_sequence = []
            elif line and (line[0].isdigit() or line.startswith('-')):
                action_name = line.lstrip('0123456789.-) ').strip()
                if action_name:
                    action = Action(
                        action_id=f"action_{len(current_sequence)}",
                        name=action_name
                    )
                    current_sequence.append(action)
        
        if current_sequence:
            action_sequences.append(current_sequence)
        
        # Simulate each sequence
        simulations = []
        for seq in action_sequences[:num_alternatives]:
            if seq:  # Only simulate non-empty sequences
                sim = self.simulate_trajectory(current_state, seq)
                simulations.append(sim)
        
        # Sort by expected reward
        simulations.sort(key=lambda s: s.cumulative_reward, reverse=True)
        
        return simulations


def demonstrate_world_model():
    """Demonstrate World Model Learning pattern"""
    
    print("=" * 80)
    print("PATTERN 062: WORLD MODEL LEARNING DEMONSTRATION")
    print("=" * 80)
    print("\nLearning environment dynamics for model-based planning\n")
    
    # Test 1: Building a world model from experience
    print("\n" + "=" * 80)
    print("TEST 1: Learning from Experience")
    print("=" * 80)
    
    world = WorldModel(domain="business strategy")
    
    # Add some experiences
    print("\n📚 Adding experiences to world model...")
    
    experiences = [
        Transition(
            state=State("s1", "Low market share, limited brand awareness"),
            action=Action("a1", "Launch aggressive marketing campaign"),
            next_state=State("s2", "Increased brand visibility, higher costs"),
            reward=0.3
        ),
        Transition(
            state=State("s2", "Increased brand visibility, higher costs"),
            action=Action("a2", "Optimize pricing strategy"),
            next_state=State("s3", "Improved market share, balanced budget"),
            reward=0.7
        ),
        Transition(
            state=State("s3", "Improved market share, balanced budget"),
            action=Action("a3", "Expand product line"),
            next_state=State("s4", "Diversified revenue, market leadership"),
            reward=0.9
        )
    ]
    
    for exp in experiences:
        world.add_experience(exp)
        print(f"   ✓ {exp.state.description} --[{exp.action.name}]--> {exp.next_state.description}")
    
    print(f"\n   Total experiences: {len(world.experiences)}")
    
    # Test 2: Predicting next state
    print("\n" + "=" * 80)
    print("TEST 2: State Prediction")
    print("=" * 80)
    
    current_state = State(
        "s_test",
        "New competitor entering market"
    )
    
    test_action = Action(
        "a_test",
        "Form strategic partnership"
    )
    
    print(f"\n🔮 Making prediction...")
    print(f"   Current: {current_state.description}")
    print(f"   Action: {test_action.name}")
    
    prediction = world.predict_next_state(current_state, test_action)
    
    print(f"\n✨ Predicted outcome:")
    print(f"   State: {prediction.predicted_state.description}")
    print(f"   Confidence: {prediction.confidence:.2f}")
    print(f"   Expected Reward: {prediction.predicted_reward:.2f}")
    
    # Test 3: Multi-step simulation
    print("\n" + "=" * 80)
    print("TEST 3: Trajectory Simulation")
    print("=" * 80)
    
    initial = State("s_start", "Startup with innovative product")
    
    actions = [
        Action("a1", "Secure seed funding"),
        Action("a2", "Build MVP"),
        Action("a3", "Beta testing with early adopters"),
        Action("a4", "Launch marketing campaign")
    ]
    
    print(f"\n🎬 Simulating trajectory...")
    print(f"   Initial State: {initial.description}")
    print(f"   Actions: {len(actions)}")
    
    simulation = world.simulate_trajectory(initial, actions)
    
    print(f"\n📊 Simulation Results:")
    print(f"   Success Probability: {simulation.success_probability:.2%}")
    print(f"   Cumulative Reward: {simulation.cumulative_reward:.2f}")
    print(f"\n   Trajectory:")
    for i, state in enumerate(simulation.predicted_trajectory):
        if i == 0:
            print(f"   0. [START] {state.description}")
        else:
            action = simulation.action_sequence[i-1]
            print(f"   {i}. --[{action.name}]--> {state.description}")
    
    # Test 4: Model-based planning
    print("\n" + "=" * 80)
    print("TEST 4: Model-Based Planning")
    print("=" * 80)
    
    planning_state = State(
        "s_plan",
        "Established company facing digital transformation"
    )
    
    goal = "Successfully transition to cloud-based operations"
    
    print(f"\n🎯 Planning Goal: {goal}")
    print(f"   Current State: {planning_state.description}")
    print(f"\n   Generating alternative plans...")
    
    plans = world.plan_with_model(planning_state, goal, num_alternatives=3)
    
    print(f"\n📋 Generated {len(plans)} alternative plans:\n")
    
    for i, plan in enumerate(plans, 1):
        print(f"   Plan {i}:")
        print(f"   Expected Reward: {plan.cumulative_reward:.2f}")
        print(f"   Success Probability: {plan.success_probability:.2%}")
        print(f"   Actions:")
        for j, action in enumerate(plan.action_sequence, 1):
            print(f"      {j}. {action.name}")
        print()
    
    if plans:
        print(f"   🏆 Recommended: Plan 1 (highest expected reward)")
    
    # Test 5: What-if analysis
    print("\n" + "=" * 80)
    print("TEST 5: What-If Analysis")
    print("=" * 80)
    
    world2 = WorldModel(domain="project management")
    
    project_state = State("proj", "Project behind schedule, team morale low")
    
    scenarios = [
        ("Hire additional developers", Action("a1", "Hire additional developers")),
        ("Reduce scope", Action("a2", "Reduce project scope")),
        ("Extend deadline", Action("a3", "Negotiate deadline extension"))
    ]
    
    print(f"\n🔍 Analyzing scenarios for: {project_state.description}\n")
    
    for scenario_name, action in scenarios:
        prediction = world2.predict_next_state(project_state, action)
        
        print(f"   Scenario: {scenario_name}")
        print(f"      Predicted outcome: {prediction.predicted_state.description}")
        print(f"      Expected reward: {prediction.predicted_reward:+.2f}")
        print(f"      Confidence: {prediction.confidence:.2%}")
        print()
    
    # Summary
    print("\n" + "=" * 80)
    print("WORLD MODEL LEARNING PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Sample Efficiency: Learn from limited experience
2. Planning Capability: Simulate before acting
3. Risk Assessment: Evaluate outcomes without execution
4. What-If Analysis: Compare alternative scenarios
5. Transfer Learning: Generalize across similar situations

World Model Components:
1. State Representation: How world is encoded
   - Discrete states (finite set)
   - Continuous states (real values)
   - Symbolic states (descriptions)
   - Hybrid (combination)

2. Dynamics Model: Predicts transitions
   - Input: Current state + Action
   - Output: Next state prediction
   - Confidence estimation

3. Reward Model: Estimates outcomes
   - Input: State + Action + Result
   - Output: Expected reward/value
   - Risk assessment

4. Experience Buffer: Stores transitions
   - Real experiences
   - Simulated experiences
   - Relevance-based retrieval

5. Planning Module: Uses model
   - Forward simulation
   - Trajectory optimization
   - Alternative comparison

Learning Process:
1. Collect Experience: Interact with environment
2. Update Model: Learn from transitions
3. Validate Predictions: Test accuracy
4. Refine Model: Improve based on errors
5. Plan Actions: Use model for decisions

Model Types:
1. Forward Model: State → Next State
   - Most common
   - Direct prediction

2. Inverse Model: Current State + Next State → Action
   - Action discovery
   - Imitation learning

3. Reward Model: State + Action → Reward
   - Value estimation
   - Policy evaluation

4. Termination Model: State → Done?
   - Episode completion
   - Goal achievement

Prediction Strategies:
- Deterministic: Single prediction
- Stochastic: Sample from distribution
- Ensemble: Multiple models
- Uncertainty-aware: Confidence bounds

Use Cases:
- Robotics: Simulate movements before execution
- Game AI: Plan strategies by modeling game
- Business: Scenario planning and forecasting
- Healthcare: Treatment outcome prediction
- Finance: Market behavior modeling
- Education: Learning path optimization

Planning with World Models:
1. Model-Predictive Control (MPC)
   - Rolling horizon planning
   - Re-plan at each step

2. Tree Search
   - Build search tree using model
   - Evaluate trajectories

3. Trajectory Optimization
   - Find optimal action sequence
   - Gradient-based or sampling

4. Policy Learning
   - Use model to improve policy
   - Dyna-style algorithms

Challenges:
1. Model Accuracy: Prediction errors compound
2. Generalization: Unseen states
3. Computational Cost: Simulation overhead
4. Model Bias: Learned from limited data
5. Non-stationarity: Environment changes

Best Practices:
1. Start with simple models
2. Validate predictions frequently
3. Use ensemble of models
4. Estimate uncertainty
5. Combine model-based and model-free
6. Update model continuously
7. Use for planning, not just prediction

Production Considerations:
- Model versioning and updates
- Prediction latency requirements
- Memory for experience buffer
- Model accuracy metrics
- Fallback to model-free
- A/B testing model changes
- Continuous validation

Advanced Techniques:
1. Latent World Models: Learn compressed representations
2. Hierarchical Models: Multi-scale predictions
3. Causal Models: Understand mechanisms
4. Meta-Learning Models: Quick adaptation
5. Uncertainty-Aware Models: Calibrated predictions

Comparison with Related Patterns:
- vs. Model-Free RL: More sample efficient
- vs. Imitation Learning: Enables planning
- vs. Heuristic Planning: Data-driven
- vs. Simulation: Learned not programmed

Integration with Other Patterns:
- Curriculum Learning: Progressive model building
- Meta-Learning: Quick model adaptation
- Active Learning: Query informative states
- Memory Patterns: Store experiences

The World Model Learning pattern enables agents to learn
environment dynamics from experience and use these models
for efficient planning, simulation, and decision-making.
""")


if __name__ == "__main__":
    demonstrate_world_model()
