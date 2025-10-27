"""
Pattern 103: Diffusion-Based Planning

This pattern implements planning using diffusion models for trajectory generation.
Diffusion models generate plans through iterative denoising, enabling flexible,
multimodal planning with gradient guidance.

Use Cases:
- Motion planning and robotics
- Game AI and strategy planning
- Sequence generation tasks
- Multi-modal planning scenarios
- Path planning with constraints

Key Features:
- Trajectory generation via diffusion process
- Iterative refinement through denoising
- Conditional generation with guidance
- Multimodal plan sampling
- Constraint satisfaction
- Gradient-based optimization

Implementation:
- Pure Python (3.8+) with comprehensive type hints
- Zero external dependencies
- Production-ready error handling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import random
import math
from datetime import datetime
import uuid


class NoiseSchedule(Enum):
    """Types of noise schedules for diffusion."""
    LINEAR = "linear"
    COSINE = "cosine"
    QUADRATIC = "quadratic"


@dataclass
class State:
    """Represents a state in the trajectory."""
    position: Tuple[float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_vector(self) -> List[float]:
        """Convert to vector representation."""
        return list(self.position) + list(self.velocity)
    
    @staticmethod
    def from_vector(vec: List[float]) -> 'State':
        """Create state from vector."""
        return State(
            position=(vec[0], vec[1]),
            velocity=(vec[2], vec[3]) if len(vec) >= 4 else (0.0, 0.0)
        )


@dataclass
class Action:
    """Represents an action."""
    delta: Tuple[float, float]
    duration: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """A sequence of states and actions."""
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    states: List[State] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    score: float = 0.0
    noise_level: float = 1.0  # Current noise level (1.0 = pure noise, 0.0 = clean)
    
    def length(self) -> int:
        """Get trajectory length."""
        return len(self.states)
    
    def to_matrix(self) -> List[List[float]]:
        """Convert trajectory to matrix form."""
        return [state.to_vector() for state in self.states]
    
    @staticmethod
    def from_matrix(matrix: List[List[float]]) -> 'Trajectory':
        """Create trajectory from matrix."""
        states = [State.from_vector(vec) for vec in matrix]
        return Trajectory(states=states)


@dataclass
class DiffusionConfig:
    """Configuration for diffusion process."""
    num_diffusion_steps: int = 100
    noise_schedule: NoiseSchedule = NoiseSchedule.LINEAR
    clip_value: float = 10.0  # Clip values during generation
    guidance_weight: float = 1.0  # Strength of guidance
    
    def get_noise_level(self, timestep: int) -> float:
        """Get noise level at timestep."""
        t = timestep / self.num_diffusion_steps
        
        if self.noise_schedule == NoiseSchedule.LINEAR:
            return 1.0 - t
        elif self.noise_schedule == NoiseSchedule.COSINE:
            return math.cos(t * math.pi / 2)
        elif self.noise_schedule == NoiseSchedule.QUADRATIC:
            return (1.0 - t) ** 2
        
        return 1.0 - t


class DiffusionModel:
    """
    Simplified diffusion model for trajectory generation.
    
    In practice, this would be a neural network trained on trajectories.
    Here we use a simplified model for demonstration.
    """
    
    def __init__(self, state_dim: int = 4, horizon: int = 10):
        self.state_dim = state_dim
        self.horizon = horizon
        
        # Simplified "learned" parameters
        self.weights: Dict[str, float] = {
            "smoothness": 0.5,
            "goal_attraction": 0.3,
            "obstacle_repulsion": 0.2
        }
    
    def predict_noise(self, trajectory: Trajectory, timestep: int,
                     condition: Optional[Dict[str, Any]] = None) -> List[List[float]]:
        """
        Predict noise to remove from trajectory.
        
        In a real diffusion model, this would be a neural network.
        Here we approximate with heuristics.
        """
        matrix = trajectory.to_matrix()
        noise = []
        
        goal = condition.get("goal") if condition else None
        obstacles = condition.get("obstacles", []) if condition else []
        
        for i, state_vec in enumerate(matrix):
            state_noise = [0.0] * len(state_vec)
            
            # Smoothness: encourage smooth trajectories
            if i > 0 and i < len(matrix) - 1:
                prev_vec = matrix[i - 1]
                next_vec = matrix[i + 1]
                
                for j in range(len(state_vec)):
                    # Second derivative (curvature)
                    curvature = state_vec[j] - (prev_vec[j] + next_vec[j]) / 2
                    state_noise[j] += self.weights["smoothness"] * curvature
            
            # Goal attraction
            if goal:
                goal_vec = goal if isinstance(goal, list) else list(goal)
                for j in range(min(len(state_vec), len(goal_vec))):
                    direction = goal_vec[j] - state_vec[j]
                    state_noise[j] += self.weights["goal_attraction"] * direction * 0.1
            
            # Obstacle repulsion
            for obstacle in obstacles:
                obs_pos = obstacle[:2] if isinstance(obstacle, (list, tuple)) else (obstacle, obstacle)
                state_pos = state_vec[:2]
                
                dx = state_pos[0] - obs_pos[0]
                dy = state_pos[1] - obs_pos[1]
                dist = math.sqrt(dx**2 + dy**2) + 1e-6
                
                if dist < 3.0:
                    repulsion = (3.0 - dist) / dist
                    state_noise[0] -= self.weights["obstacle_repulsion"] * dx * repulsion
                    state_noise[1] -= self.weights["obstacle_repulsion"] * dy * repulsion
            
            noise.append(state_noise)
        
        return noise
    
    def denoise_step(self, trajectory: Trajectory, timestep: int,
                    config: DiffusionConfig,
                    condition: Optional[Dict[str, Any]] = None) -> Trajectory:
        """
        Perform one denoising step.
        
        x_{t-1} = x_t - predicted_noise * step_size
        """
        # Predict noise
        predicted_noise = self.predict_noise(trajectory, timestep, condition)
        
        # Get step size based on noise schedule
        noise_level = config.get_noise_level(timestep)
        step_size = 1.0 / config.num_diffusion_steps
        
        # Denoise
        matrix = trajectory.to_matrix()
        denoised_matrix = []
        
        for i, (state_vec, noise_vec) in enumerate(zip(matrix, predicted_noise)):
            denoised_vec = []
            for j, (val, noise) in enumerate(zip(state_vec, noise_vec)):
                denoised_val = val - noise * step_size
                # Clip to prevent explosion
                denoised_val = max(-config.clip_value, min(config.clip_value, denoised_val))
                denoised_vec.append(denoised_val)
            denoised_matrix.append(denoised_vec)
        
        denoised_trajectory = Trajectory.from_matrix(denoised_matrix)
        denoised_trajectory.noise_level = noise_level
        denoised_trajectory.trajectory_id = trajectory.trajectory_id
        
        return denoised_trajectory


class TrajectoryScorer:
    """
    Scores trajectories based on various criteria.
    
    Used for guidance during diffusion and final selection.
    """
    
    def __init__(self):
        self.weights = {
            "goal_distance": 1.0,
            "smoothness": 0.5,
            "collision": 2.0,
            "length": 0.1
        }
    
    def score_trajectory(self, trajectory: Trajectory,
                        goal: Optional[Tuple[float, float]] = None,
                        obstacles: Optional[List[Tuple[float, float]]] = None) -> float:
        """Compute overall trajectory score (higher is better)."""
        score = 0.0
        
        if not trajectory.states:
            return -1000.0
        
        # Goal distance (negative because closer is better)
        if goal:
            final_pos = trajectory.states[-1].position
            dist = math.sqrt((final_pos[0] - goal[0])**2 + (final_pos[1] - goal[1])**2)
            score -= self.weights["goal_distance"] * dist
        
        # Smoothness (prefer smooth trajectories)
        if len(trajectory.states) >= 3:
            total_curvature = 0.0
            for i in range(1, len(trajectory.states) - 1):
                prev_pos = trajectory.states[i - 1].position
                curr_pos = trajectory.states[i].position
                next_pos = trajectory.states[i + 1].position
                
                # Measure curvature
                dx1 = curr_pos[0] - prev_pos[0]
                dy1 = curr_pos[1] - prev_pos[1]
                dx2 = next_pos[0] - curr_pos[0]
                dy2 = next_pos[1] - curr_pos[1]
                
                curvature = abs(dx2 - dx1) + abs(dy2 - dy1)
                total_curvature += curvature
            
            score -= self.weights["smoothness"] * total_curvature / len(trajectory.states)
        
        # Collision penalty
        if obstacles:
            collision_penalty = 0.0
            for state in trajectory.states:
                for obs in obstacles:
                    dist = math.sqrt((state.position[0] - obs[0])**2 + 
                                   (state.position[1] - obs[1])**2)
                    if dist < 1.0:
                        collision_penalty += (1.0 - dist) ** 2
            
            score -= self.weights["collision"] * collision_penalty
        
        # Length penalty (prefer shorter paths, all else equal)
        path_length = 0.0
        for i in range(1, len(trajectory.states)):
            prev_pos = trajectory.states[i - 1].position
            curr_pos = trajectory.states[i].position
            path_length += math.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                                    (curr_pos[1] - prev_pos[1])**2)
        
        score -= self.weights["length"] * path_length
        
        return score


class DiffusionPlanner:
    """
    Plans trajectories using diffusion models.
    
    Process:
    1. Start with pure noise trajectory
    2. Iteratively denoise using diffusion model
    3. Apply conditional guidance (goals, constraints)
    4. Score and select best trajectory
    """
    
    def __init__(self, config: Optional[DiffusionConfig] = None):
        self.config = config or DiffusionConfig()
        self.model = DiffusionModel()
        self.scorer = TrajectoryScorer()
        
        self.trajectories_generated = 0
        self.total_denoising_steps = 0
    
    def initialize_trajectory(self, start: State, horizon: int) -> Trajectory:
        """Initialize trajectory with random noise."""
        states = [start]
        
        # Generate random noisy states
        for _ in range(horizon - 1):
            # Random walk from start
            noise_x = random.gauss(0, 2.0)
            noise_y = random.gauss(0, 2.0)
            
            noisy_state = State(
                position=(start.position[0] + noise_x, start.position[1] + noise_y),
                velocity=(random.gauss(0, 0.5), random.gauss(0, 0.5))
            )
            states.append(noisy_state)
        
        return Trajectory(states=states, noise_level=1.0)
    
    def generate_trajectory(self, start: State, goal: Tuple[float, float],
                          horizon: int = 10,
                          obstacles: Optional[List[Tuple[float, float]]] = None) -> Trajectory:
        """
        Generate a single trajectory using diffusion.
        
        Args:
            start: Starting state
            goal: Goal position
            horizon: Trajectory length
            obstacles: Obstacle positions
        """
        # Initialize with noise
        trajectory = self.initialize_trajectory(start, horizon)
        
        # Conditioning information
        condition = {
            "start": start.position,
            "goal": goal,
            "obstacles": obstacles or []
        }
        
        # Iterative denoising
        for t in range(self.config.num_diffusion_steps, 0, -1):
            trajectory = self.model.denoise_step(trajectory, t, self.config, condition)
            self.total_denoising_steps += 1
            
            # Apply gradient guidance periodically
            if t % 10 == 0:
                trajectory = self._apply_guidance(trajectory, condition)
        
        # Final refinement: ensure start and goal
        if trajectory.states:
            trajectory.states[0] = start
            # Pull final state toward goal
            final = trajectory.states[-1]
            trajectory.states[-1] = State(
                position=(
                    final.position[0] * 0.3 + goal[0] * 0.7,
                    final.position[1] * 0.3 + goal[1] * 0.7
                )
            )
        
        # Score trajectory
        trajectory.score = self.scorer.score_trajectory(trajectory, goal, obstacles)
        self.trajectories_generated += 1
        
        return trajectory
    
    def _apply_guidance(self, trajectory: Trajectory, 
                       condition: Dict[str, Any]) -> Trajectory:
        """Apply gradient-based guidance to trajectory."""
        goal = condition.get("goal")
        obstacles = condition.get("obstacles", [])
        
        if not goal:
            return trajectory
        
        # Compute gradient of score with respect to trajectory
        # Simplified: adjust positions toward goal and away from obstacles
        guided_states = []
        
        for i, state in enumerate(trajectory.states):
            # Goal gradient
            dx_goal = goal[0] - state.position[0]
            dy_goal = goal[1] - state.position[1]
            
            # Obstacle gradient
            dx_obs, dy_obs = 0.0, 0.0
            for obs in obstacles:
                dx = state.position[0] - obs[0]
                dy = state.position[1] - obs[1]
                dist = math.sqrt(dx**2 + dy**2) + 1e-6
                
                if dist < 3.0:
                    strength = (3.0 - dist) / dist
                    dx_obs += dx * strength
                    dy_obs += dy * strength
            
            # Apply guidance
            guidance_scale = self.config.guidance_weight * 0.1
            new_x = state.position[0] + (dx_goal * 0.1 + dx_obs * 0.2) * guidance_scale
            new_y = state.position[1] + (dy_goal * 0.1 + dy_obs * 0.2) * guidance_scale
            
            guided_states.append(State(position=(new_x, new_y)))
        
        return Trajectory(states=guided_states, trajectory_id=trajectory.trajectory_id)
    
    def sample_trajectories(self, start: State, goal: Tuple[float, float],
                          num_samples: int = 5,
                          horizon: int = 10,
                          obstacles: Optional[List[Tuple[float, float]]] = None) -> List[Trajectory]:
        """
        Sample multiple trajectories and return them.
        
        Enables multimodal planning - finding multiple diverse solutions.
        """
        trajectories = []
        
        for _ in range(num_samples):
            trajectory = self.generate_trajectory(start, goal, horizon, obstacles)
            trajectories.append(trajectory)
        
        # Sort by score
        trajectories.sort(key=lambda t: t.score, reverse=True)
        
        return trajectories
    
    def plan(self, start: State, goal: Tuple[float, float],
            horizon: int = 10,
            num_samples: int = 5,
            obstacles: Optional[List[Tuple[float, float]]] = None) -> Trajectory:
        """
        Plan a trajectory from start to goal.
        
        Samples multiple trajectories and returns the best one.
        """
        trajectories = self.sample_trajectories(start, goal, num_samples, 
                                               horizon, obstacles)
        
        if not trajectories:
            raise ValueError("No trajectories generated")
        
        # Return best trajectory
        return trajectories[0]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get planner statistics."""
        return {
            "trajectories_generated": self.trajectories_generated,
            "total_denoising_steps": self.total_denoising_steps,
            "avg_steps_per_trajectory": (
                self.total_denoising_steps / self.trajectories_generated 
                if self.trajectories_generated > 0 else 0
            )
        }


class DiffusionPlanningAgent:
    """
    Complete agent using diffusion-based planning.
    
    Integrates:
    - Trajectory generation via diffusion
    - Multi-modal planning
    - Constraint handling
    - Plan execution and monitoring
    """
    
    def __init__(self, config: Optional[DiffusionConfig] = None):
        self.planner = DiffusionPlanner(config)
        self.current_plan: Optional[Trajectory] = None
        self.execution_step = 0
        
        self.plans_created = 0
        self.plans_executed = 0
        self.replans = 0
    
    def create_plan(self, start: State, goal: Tuple[float, float],
                   horizon: int = 10,
                   obstacles: Optional[List[Tuple[float, float]]] = None) -> Trajectory:
        """Create a plan using diffusion."""
        plan = self.planner.plan(start, goal, horizon, obstacles=obstacles)
        self.current_plan = plan
        self.execution_step = 0
        self.plans_created += 1
        
        return plan
    
    def execute_step(self) -> Optional[State]:
        """Execute one step of current plan."""
        if not self.current_plan or self.execution_step >= len(self.current_plan.states):
            return None
        
        state = self.current_plan.states[self.execution_step]
        self.execution_step += 1
        
        if self.execution_step >= len(self.current_plan.states):
            self.plans_executed += 1
        
        return state
    
    def replan(self, current_state: State, goal: Tuple[float, float],
              obstacles: Optional[List[Tuple[float, float]]] = None) -> Trajectory:
        """Replan from current state."""
        self.replans += 1
        remaining_horizon = (len(self.current_plan.states) - self.execution_step 
                           if self.current_plan else 10)
        return self.create_plan(current_state, goal, remaining_horizon, obstacles)
    
    def get_alternative_plans(self, start: State, goal: Tuple[float, float],
                            num_alternatives: int = 3,
                            obstacles: Optional[List[Tuple[float, float]]] = None) -> List[Trajectory]:
        """Get multiple alternative plans."""
        return self.planner.sample_trajectories(start, goal, num_alternatives, 
                                               obstacles=obstacles)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        planner_stats = self.planner.get_statistics()
        
        return {
            **planner_stats,
            "plans_created": self.plans_created,
            "plans_executed": self.plans_executed,
            "replans": self.replans,
            "current_plan_progress": (
                f"{self.execution_step}/{len(self.current_plan.states)}" 
                if self.current_plan else "None"
            )
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_diffusion_planning():
    """Demonstrate diffusion-based planning."""
    
    print("=" * 70)
    print("DIFFUSION-BASED PLANNING DEMONSTRATION")
    print("=" * 70)
    
    print("\n1. SETUP")
    print("-" * 70)
    
    # Configuration
    config = DiffusionConfig(
        num_diffusion_steps=50,
        noise_schedule=NoiseSchedule.COSINE,
        guidance_weight=1.5
    )
    
    print(f"   Diffusion configuration:")
    print(f"     Denoising steps: {config.num_diffusion_steps}")
    print(f"     Noise schedule: {config.noise_schedule.value}")
    print(f"     Guidance weight: {config.guidance_weight}")
    
    # Create agent
    agent = DiffusionPlanningAgent(config)
    
    print(f"\n   Agent created")
    
    print("\n2. SIMPLE TRAJECTORY GENERATION")
    print("-" * 70)
    
    start = State(position=(0.0, 0.0))
    goal = (10.0, 10.0)
    
    print(f"   Start: {start.position}")
    print(f"   Goal: {goal}")
    
    trajectory = agent.create_plan(start, goal, horizon=8)
    
    print(f"\n   Generated trajectory (ID: {trajectory.trajectory_id}):")
    print(f"     Length: {trajectory.length()} states")
    print(f"     Score: {trajectory.score:.2f}")
    print(f"\n   Trajectory states:")
    for i, state in enumerate(trajectory.states):
        print(f"     Step {i}: position=({state.position[0]:.2f}, {state.position[1]:.2f})")
    
    print("\n3. PLANNING WITH OBSTACLES")
    print("-" * 70)
    
    obstacles = [(5.0, 5.0), (7.0, 3.0), (3.0, 7.0)]
    print(f"   Obstacles at: {obstacles}")
    
    trajectory_obs = agent.create_plan(start, goal, horizon=10, obstacles=obstacles)
    
    print(f"\n   Generated trajectory avoiding obstacles:")
    print(f"     Length: {trajectory_obs.length()} states")
    print(f"     Score: {trajectory_obs.score:.2f}")
    print(f"\n   Waypoints:")
    for i in range(0, len(trajectory_obs.states), 2):
        state = trajectory_obs.states[i]
        print(f"     Step {i}: ({state.position[0]:.2f}, {state.position[1]:.2f})")
    
    print("\n4. MULTI-MODAL PLANNING")
    print("-" * 70)
    print("   Sampling multiple alternative trajectories...")
    
    alternatives = agent.get_alternative_plans(start, goal, num_alternatives=5, 
                                              obstacles=obstacles)
    
    print(f"\n   Generated {len(alternatives)} alternative plans:")
    for i, traj in enumerate(alternatives, 1):
        print(f"\n     Alternative {i}:")
        print(f"       Score: {traj.score:.2f}")
        print(f"       Start: ({traj.states[0].position[0]:.2f}, {traj.states[0].position[1]:.2f})")
        print(f"       End: ({traj.states[-1].position[0]:.2f}, {traj.states[-1].position[1]:.2f})")
        
        # Calculate path characteristics
        path_length = 0.0
        for j in range(1, len(traj.states)):
            prev = traj.states[j-1].position
            curr = traj.states[j].position
            path_length += math.sqrt((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2)
        print(f"       Path length: {path_length:.2f}")
    
    print("\n5. PLAN EXECUTION")
    print("-" * 70)
    print("   Executing plan step by step...")
    
    for step in range(4):
        state = agent.execute_step()
        if state:
            print(f"     Step {step}: Reached ({state.position[0]:.2f}, {state.position[1]:.2f})")
    
    print("\n6. ADAPTIVE REPLANNING")
    print("-" * 70)
    print("   Simulating obstacle detection - replanning required...")
    
    # Get current state
    current_state = agent.execute_step()
    if current_state:
        print(f"   Current position: ({current_state.position[0]:.2f}, {current_state.position[1]:.2f})")
        
        # New obstacle detected
        new_obstacles = obstacles + [(8.0, 8.0)]
        print(f"   New obstacle detected at: (8.0, 8.0)")
        
        new_plan = agent.replan(current_state, goal, new_obstacles)
        print(f"\n   Replanned trajectory:")
        print(f"     New plan length: {new_plan.length()} states")
        print(f"     New score: {new_plan.score:.2f}")
    
    print("\n7. STATISTICS")
    print("-" * 70)
    stats = agent.get_statistics()
    print(f"   Plans created: {stats['plans_created']}")
    print(f"   Plans executed: {stats['plans_executed']}")
    print(f"   Replans: {stats['replans']}")
    print(f"   Trajectories generated: {stats['trajectories_generated']}")
    print(f"   Total denoising steps: {stats['total_denoising_steps']}")
    print(f"   Avg steps per trajectory: {stats['avg_steps_per_trajectory']:.1f}")
    print(f"   Current plan progress: {stats['current_plan_progress']}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("1. Trajectory generation via iterative denoising")
    print("2. Conditional generation with goal guidance")
    print("3. Obstacle avoidance through gradient guidance")
    print("4. Multi-modal planning (multiple alternatives)")
    print("5. Smooth trajectory generation")
    print("6. Plan execution and monitoring")
    print("7. Adaptive replanning with new information")
    print("8. Configurable noise schedules")


if __name__ == "__main__":
    demonstrate_diffusion_planning()
