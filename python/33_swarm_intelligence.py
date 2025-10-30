"""
Swarm Intelligence Pattern
Many simple agents collaborate to solve problems
"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import random
import math
@dataclass
class Position:
    x: float
    y: float
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
@dataclass
class Particle:
    """Individual particle in the swarm"""
    id: int
    position: Position
    velocity: Position
    best_position: Position
    best_score: float
    def __post_init__(self):
        if self.best_position is None:
            self.best_position = Position(self.position.x, self.position.y)
class SwarmAgent:
    """Individual agent in swarm"""
    def __init__(self, agent_id: int, position: Position):
        self.agent_id = agent_id
        self.position = position
        self.velocity = Position(random.uniform(-1, 1), random.uniform(-1, 1))
        self.best_position = Position(position.x, position.y)
        self.best_score = float('-inf')
        self.neighbors: List['SwarmAgent'] = []
    def evaluate_position(self, objective_function) -> float:
        """Evaluate current position using objective function"""
        score = objective_function(self.position.x, self.position.y)
        # Update personal best
        if score > self.best_score:
            self.best_score = score
            self.best_position = Position(self.position.x, self.position.y)
        return score
    def update_velocity(self, global_best: Position, w: float = 0.5, c1: float = 1.5, c2: float = 1.5):
        """Update velocity based on personal and global best"""
        # Inertia component
        inertia_x = w * self.velocity.x
        inertia_y = w * self.velocity.y
        # Cognitive component (personal best)
        r1 = random.random()
        cognitive_x = c1 * r1 * (self.best_position.x - self.position.x)
        cognitive_y = c1 * r1 * (self.best_position.y - self.position.y)
        # Social component (global best)
        r2 = random.random()
        social_x = c2 * r2 * (global_best.x - self.position.x)
        social_y = c2 * r2 * (global_best.y - self.position.y)
        # Update velocity
        self.velocity.x = inertia_x + cognitive_x + social_x
        self.velocity.y = inertia_y + cognitive_y + social_y
        # Limit velocity
        max_velocity = 2.0
        velocity_magnitude = math.sqrt(self.velocity.x**2 + self.velocity.y**2)
        if velocity_magnitude > max_velocity:
            self.velocity.x = (self.velocity.x / velocity_magnitude) * max_velocity
            self.velocity.y = (self.velocity.y / velocity_magnitude) * max_velocity
    def update_position(self, bounds: Tuple[float, float, float, float]):
        """Update position based on velocity"""
        min_x, max_x, min_y, max_y = bounds
        self.position.x += self.velocity.x
        self.position.y += self.velocity.y
        # Keep within bounds
        self.position.x = max(min_x, min(max_x, self.position.x))
        self.position.y = max(min_y, min(max_y, self.position.y))
class ParticleSwarmOptimizer:
    """Particle Swarm Optimization system"""
    def __init__(self, num_agents: int, bounds: Tuple[float, float, float, float]):
        self.num_agents = num_agents
        self.bounds = bounds  # (min_x, max_x, min_y, max_y)
        self.agents: List[SwarmAgent] = []
        self.global_best_position: Position = None
        self.global_best_score: float = float('-inf')
        self.iteration_history: List[Dict[str, Any]] = []
        self._initialize_swarm()
    def _initialize_swarm(self):
        """Initialize swarm with random positions"""
        min_x, max_x, min_y, max_y = self.bounds
        for i in range(self.num_agents):
            position = Position(
                x=random.uniform(min_x, max_x),
                y=random.uniform(min_y, max_y)
            )
            agent = SwarmAgent(i, position)
            self.agents.append(agent)
        print(f"Initialized swarm with {self.num_agents} agents")
    def optimize(self, objective_function, max_iterations: int = 50, verbose: bool = True) -> Dict[str, Any]:
        """Run particle swarm optimization"""
        print(f"\n{'='*70}")
        print(f"PARTICLE SWARM OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Swarm size: {self.num_agents}")
        print(f"Max iterations: {max_iterations}")
        print(f"Search bounds: {self.bounds}\n")
        for iteration in range(max_iterations):
            # Evaluate all agents
            scores = []
            for agent in self.agents:
                score = agent.evaluate_position(objective_function)
                scores.append(score)
                # Update global best
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = Position(
                        agent.position.x,
                        agent.position.y
                    )
            # Record iteration stats
            avg_score = sum(scores) / len(scores)
            self.iteration_history.append({
                'iteration': iteration,
                'best_score': self.global_best_score,
                'avg_score': avg_score,
                'best_position': (self.global_best_position.x, self.global_best_position.y)
            })
            if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
                print(f"Iteration {iteration:3d}: Best={self.global_best_score:.4f}, "
                      f"Avg={avg_score:.4f}, "
                      f"Pos=({self.global_best_position.x:.2f}, {self.global_best_position.y:.2f})")
            # Update all agents
            for agent in self.agents:
                agent.update_velocity(self.global_best_position)
                agent.update_position(self.bounds)
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Best Score: {self.global_best_score:.4f}")
        print(f"Best Position: ({self.global_best_position.x:.4f}, {self.global_best_position.y:.4f})")
        return {
            'best_score': self.global_best_score,
            'best_position': (self.global_best_position.x, self.global_best_position.y),
            'iterations': max_iterations,
            'history': self.iteration_history
        }
    def get_swarm_state(self) -> List[Dict[str, Any]]:
        """Get current state of all agents"""
        return [
            {
                'id': agent.agent_id,
                'position': (agent.position.x, agent.position.y),
                'velocity': (agent.velocity.x, agent.velocity.y),
                'best_score': agent.best_score
            }
            for agent in self.agents
        ]
class AntColonyOptimizer:
    """Ant Colony Optimization for path finding"""
    def __init__(self, num_ants: int, num_nodes: int):
        self.num_ants = num_ants
        self.num_nodes = num_nodes
        self.pheromone: List[List[float]] = [[1.0] * num_nodes for _ in range(num_nodes)]
        self.best_path: List[int] = []
        self.best_distance: float = float('inf')
    def optimize(self, distance_matrix: List[List[float]], iterations: int = 100, 
                 alpha: float = 1.0, beta: float = 2.0, evaporation: float = 0.5) -> Dict[str, Any]:
        """Run ant colony optimization"""
        print(f"\n{'='*70}")
        print(f"ANT COLONY OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Number of ants: {self.num_ants}")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Iterations: {iterations}\n")
        for iteration in range(iterations):
            paths = []
            distances = []
            # Each ant constructs a solution
            for ant in range(self.num_ants):
                path = self._construct_path(distance_matrix, alpha, beta)
                distance = self._calculate_path_distance(path, distance_matrix)
                paths.append(path)
                distances.append(distance)
                # Update best solution
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path.copy()
            # Update pheromones
            self._update_pheromones(paths, distances, evaporation)
            if iteration % 20 == 0:
                avg_distance = sum(distances) / len(distances)
                print(f"Iteration {iteration:3d}: Best={self.best_distance:.2f}, Avg={avg_distance:.2f}")
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Best Path: {self.best_path}")
        print(f"Best Distance: {self.best_distance:.2f}")
        return {
            'best_path': self.best_path,
            'best_distance': self.best_distance,
            'iterations': iterations
        }
    def _construct_path(self, distance_matrix: List[List[float]], alpha: float, beta: float) -> List[int]:
        """Construct path for one ant"""
        path = [0]  # Start at node 0
        unvisited = set(range(1, self.num_nodes))
        while unvisited:
            current = path[-1]
            next_node = self._select_next_node(current, unvisited, distance_matrix, alpha, beta)
            path.append(next_node)
            unvisited.remove(next_node)
        return path
    def _select_next_node(self, current: int, unvisited: set, distance_matrix: List[List[float]], 
                          alpha: float, beta: float) -> int:
        """Select next node using pheromone and distance"""
        probabilities = []
        for node in unvisited:
            pheromone = self.pheromone[current][node] ** alpha
            distance = (1.0 / distance_matrix[current][node]) ** beta
            probabilities.append((node, pheromone * distance))
        # Normalize probabilities
        total = sum(p[1] for p in probabilities)
        probabilities = [(node, prob / total) for node, prob in probabilities]
        # Select node using roulette wheel selection
        r = random.random()
        cumulative = 0
        for node, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                return node
        return probabilities[-1][0]  # Fallback
    def _calculate_path_distance(self, path: List[int], distance_matrix: List[List[float]]) -> float:
        """Calculate total distance of path"""
        distance = 0
        for i in range(len(path) - 1):
            distance += distance_matrix[path[i]][path[i + 1]]
        # Return to start
        distance += distance_matrix[path[-1]][path[0]]
        return distance
    def _update_pheromones(self, paths: List[List[int]], distances: List[float], evaporation: float):
        """Update pheromone levels"""
        # Evaporation
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.pheromone[i][j] *= (1 - evaporation)
        # Add new pheromones
        for path, distance in zip(paths, distances):
            deposit = 1.0 / distance
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += deposit
                self.pheromone[path[i + 1]][path[i]] += deposit
# Usage
if __name__ == "__main__":
    print("="*80)
    print("SWARM INTELLIGENCE DEMONSTRATIONS")
    print("="*80)
    # Example 1: Particle Swarm Optimization
    print("\n" + "="*80)
    print("EXAMPLE 1: Particle Swarm Optimization")
    print("="*80)
    # Define objective function (find maximum)
    def objective_function(x: float, y: float) -> float:
        """Example: Find maximum of -(x^2 + y^2) + 10"""
        return -(x**2 + y**2) + 10
    # Create and run PSO
    pso = ParticleSwarmOptimizer(
        num_agents=30,
        bounds=(-5.0, 5.0, -5.0, 5.0)
    )
    result = pso.optimize(objective_function, max_iterations=50)
    # Example 2: Ant Colony Optimization
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Ant Colony Optimization (TSP)")
    print("="*80)
    # Create distance matrix for 5 cities
    num_cities = 5
    distance_matrix = [
        [0, 2, 3, 4, 5],
        [2, 0, 4, 5, 3],
        [3, 4, 0, 2, 4],
        [4, 5, 2, 0, 3],
        [5, 3, 4, 3, 0]
    ]
    aco = AntColonyOptimizer(num_ants=10, num_nodes=num_cities)
    result = aco.optimize(distance_matrix, iterations=100)
