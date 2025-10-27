"""
Swarm Intelligence Pattern

Multiple simple agents work together to solve complex problems.
Implements collective behavior and emergent intelligence.

Use Cases:
- Optimization problems
- Resource allocation
- Path finding
- Distributed problem solving

Advantages:
- Robust to individual failures
- Scalable solutions
- Emergent intelligence
- Parallel processing
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random
import math


class SwarmBehavior(Enum):
    """Types of swarm behaviors"""
    PARTICLE_SWARM = "particle_swarm"
    ANT_COLONY = "ant_colony"
    BEE_COLONY = "bee_colony"
    FIREFLY = "firefly"


class AgentRole(Enum):
    """Roles within swarm"""
    SCOUT = "scout"
    WORKER = "worker"
    LEADER = "leader"
    FOLLOWER = "follower"


@dataclass
class Position:
    """Position in search space"""
    coordinates: List[float]
    fitness: Optional[float] = None


@dataclass
class SwarmAgent:
    """Individual agent in swarm"""
    agent_id: str
    role: AgentRole
    position: Position
    velocity: List[float]
    personal_best: Position
    memory: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pheromone:
    """Pheromone trail for ant colony"""
    path: List[Position]
    strength: float
    age: int = 0
    evaporation_rate: float = 0.1


@dataclass
class Solution:
    """Solution found by swarm"""
    solution_id: str
    position: Position
    fitness: float
    discovered_by: str
    timestamp: datetime
    iterations: int


class FitnessFunction:
    """Defines optimization objective"""
    
    def __init__(self, function: Callable[[List[float]], float]):
        self.function = function
        self.evaluations = 0
    
    def evaluate(self, position: List[float]) -> float:
        """Evaluate fitness at position"""
        self.evaluations += 1
        return self.function(position)


class ParticleSwarmOptimizer:
    """Particle Swarm Optimization implementation"""
    
    def __init__(self,
                 num_particles: int = 30,
                 dimensions: int = 2,
                 bounds: Tuple[float, float] = (-10.0, 10.0)):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        
        self.particles: List[SwarmAgent] = []
        self.global_best: Optional[Position] = None
        
        # PSO parameters
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        
        self._initialize_particles()
    
    def optimize(self,
                fitness_func: FitnessFunction,
                max_iterations: int = 100) -> Solution:
        """
        Run particle swarm optimization.
        
        Args:
            fitness_func: Fitness function to optimize
            max_iterations: Maximum iterations
            
        Returns:
            Best solution found
        """
        for iteration in range(max_iterations):
            # Evaluate all particles
            for particle in self.particles:
                fitness = fitness_func.evaluate(particle.position.coordinates)
                particle.position.fitness = fitness
                
                # Update personal best
                if (particle.personal_best.fitness is None or
                    fitness < particle.personal_best.fitness):
                    particle.personal_best = Position(
                        coordinates=particle.position.coordinates.copy(),
                        fitness=fitness
                    )
                
                # Update global best
                if (self.global_best is None or
                    fitness < self.global_best.fitness):
                    self.global_best = Position(
                        coordinates=particle.position.coordinates.copy(),
                        fitness=fitness
                    )
            
            # Update velocities and positions
            for particle in self.particles:
                self._update_particle(particle)
        
        # Return best solution
        return Solution(
            solution_id="pso_solution",
            position=self.global_best,
            fitness=self.global_best.fitness,
            discovered_by="particle_swarm",
            timestamp=datetime.now(),
            iterations=max_iterations
        )
    
    def _initialize_particles(self) -> None:
        """Initialize particle positions and velocities"""
        for i in range(self.num_particles):
            # Random position within bounds
            position = [
                random.uniform(self.bounds[0], self.bounds[1])
                for _ in range(self.dimensions)
            ]
            
            # Random velocity
            velocity = [
                random.uniform(-1, 1)
                for _ in range(self.dimensions)
            ]
            
            pos_obj = Position(coordinates=position)
            
            particle = SwarmAgent(
                agent_id="particle_{}".format(i),
                role=AgentRole.WORKER,
                position=pos_obj,
                velocity=velocity,
                personal_best=Position(coordinates=position.copy())
            )
            
            self.particles.append(particle)
    
    def _update_particle(self, particle: SwarmAgent) -> None:
        """Update particle velocity and position"""
        for d in range(self.dimensions):
            # Random components
            r1 = random.random()
            r2 = random.random()
            
            # Velocity update
            cognitive = self.c1 * r1 * (
                particle.personal_best.coordinates[d] -
                particle.position.coordinates[d]
            )
            
            social = self.c2 * r2 * (
                self.global_best.coordinates[d] -
                particle.position.coordinates[d]
            )
            
            particle.velocity[d] = (
                self.w * particle.velocity[d] +
                cognitive +
                social
            )
            
            # Position update
            particle.position.coordinates[d] += particle.velocity[d]
            
            # Boundary check
            if particle.position.coordinates[d] < self.bounds[0]:
                particle.position.coordinates[d] = self.bounds[0]
                particle.velocity[d] *= -0.5
            elif particle.position.coordinates[d] > self.bounds[1]:
                particle.position.coordinates[d] = self.bounds[1]
                particle.velocity[d] *= -0.5


class AntColonyOptimizer:
    """Ant Colony Optimization implementation"""
    
    def __init__(self,
                 num_ants: int = 20,
                 num_nodes: int = 10):
        self.num_ants = num_ants
        self.num_nodes = num_nodes
        
        self.ants: List[SwarmAgent] = []
        self.pheromones: List[List[float]] = []
        self.distances: List[List[float]] = []
        
        # ACO parameters
        self.alpha = 1.0  # Pheromone importance
        self.beta = 2.0  # Distance importance
        self.rho = 0.5  # Evaporation rate
        self.q = 100  # Pheromone deposit factor
        
        self._initialize_colony()
    
    def optimize(self,
                distance_matrix: List[List[float]],
                max_iterations: int = 100) -> Solution:
        """
        Run ant colony optimization for TSP.
        
        Args:
            distance_matrix: Distance matrix between nodes
            max_iterations: Maximum iterations
            
        Returns:
            Best solution (tour) found
        """
        self.distances = distance_matrix
        self.num_nodes = len(distance_matrix)
        
        # Initialize pheromones
        self.pheromones = [
            [1.0 for _ in range(self.num_nodes)]
            for _ in range(self.num_nodes)
        ]
        
        best_tour = None
        best_length = float('inf')
        
        for iteration in range(max_iterations):
            # Each ant constructs a tour
            tours = []
            tour_lengths = []
            
            for ant in self.ants:
                tour = self._construct_tour(ant)
                length = self._calculate_tour_length(tour)
                
                tours.append(tour)
                tour_lengths.append(length)
                
                if length < best_length:
                    best_length = length
                    best_tour = tour
            
            # Update pheromones
            self._update_pheromones(tours, tour_lengths)
        
        return Solution(
            solution_id="aco_solution",
            position=Position(coordinates=[float(n) for n in best_tour]),
            fitness=best_length,
            discovered_by="ant_colony",
            timestamp=datetime.now(),
            iterations=max_iterations
        )
    
    def _initialize_colony(self) -> None:
        """Initialize ant colony"""
        for i in range(self.num_ants):
            ant = SwarmAgent(
                agent_id="ant_{}".format(i),
                role=AgentRole.WORKER,
                position=Position(coordinates=[0.0]),
                velocity=[0.0],
                personal_best=Position(coordinates=[0.0])
            )
            self.ants.append(ant)
    
    def _construct_tour(self, ant: SwarmAgent) -> List[int]:
        """Construct tour for ant"""
        tour = [0]  # Start at node 0
        unvisited = set(range(1, self.num_nodes))
        
        current = 0
        
        while unvisited:
            next_node = self._select_next_node(current, unvisited)
            tour.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        
        return tour
    
    def _select_next_node(self,
                         current: int,
                         unvisited: set) -> int:
        """Select next node probabilistically"""
        probabilities = []
        
        for node in unvisited:
            pheromone = self.pheromones[current][node] ** self.alpha
            distance = (1.0 / self.distances[current][node]) ** self.beta
            probabilities.append(pheromone * distance)
        
        # Normalize probabilities
        total = sum(probabilities)
        if total == 0:
            return random.choice(list(unvisited))
        
        probabilities = [p / total for p in probabilities]
        
        # Select node
        unvisited_list = list(unvisited)
        return random.choices(unvisited_list, weights=probabilities)[0]
    
    def _calculate_tour_length(self, tour: List[int]) -> float:
        """Calculate total tour length"""
        length = 0.0
        for i in range(len(tour)):
            from_node = tour[i]
            to_node = tour[(i + 1) % len(tour)]
            length += self.distances[from_node][to_node]
        return length
    
    def _update_pheromones(self,
                          tours: List[List[int]],
                          tour_lengths: List[float]) -> None:
        """Update pheromone trails"""
        # Evaporation
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.pheromones[i][j] *= (1 - self.rho)
        
        # Deposit new pheromones
        for tour, length in zip(tours, tour_lengths):
            deposit = self.q / length
            
            for i in range(len(tour)):
                from_node = tour[i]
                to_node = tour[(i + 1) % len(tour)]
                self.pheromones[from_node][to_node] += deposit
                self.pheromones[to_node][from_node] += deposit


class SwarmIntelligenceSystem:
    """
    System managing swarm intelligence algorithms.
    Coordinates multiple agents to solve optimization problems.
    """
    
    def __init__(self, behavior: SwarmBehavior = SwarmBehavior.PARTICLE_SWARM):
        self.behavior = behavior
        self.solutions: List[Solution] = []
        self.iteration_history: List[Dict[str, Any]] = []
    
    def solve(self,
             problem_type: str,
             **kwargs) -> Solution:
        """
        Solve optimization problem using swarm intelligence.
        
        Args:
            problem_type: Type of problem
            **kwargs: Problem-specific parameters
            
        Returns:
            Best solution found
        """
        if self.behavior == SwarmBehavior.PARTICLE_SWARM:
            return self._solve_with_pso(**kwargs)
        elif self.behavior == SwarmBehavior.ANT_COLONY:
            return self._solve_with_aco(**kwargs)
        else:
            raise ValueError("Unsupported behavior: {}".format(self.behavior))
    
    def _solve_with_pso(self, **kwargs) -> Solution:
        """Solve using Particle Swarm Optimization"""
        optimizer = ParticleSwarmOptimizer(
            num_particles=kwargs.get("num_particles", 30),
            dimensions=kwargs.get("dimensions", 2),
            bounds=kwargs.get("bounds", (-10.0, 10.0))
        )
        
        fitness_func = kwargs.get("fitness_function")
        if not fitness_func:
            # Default: minimize sphere function
            fitness_func = FitnessFunction(
                lambda x: sum(xi**2 for xi in x)
            )
        
        solution = optimizer.optimize(
            fitness_func,
            max_iterations=kwargs.get("max_iterations", 100)
        )
        
        self.solutions.append(solution)
        return solution
    
    def _solve_with_aco(self, **kwargs) -> Solution:
        """Solve using Ant Colony Optimization"""
        optimizer = AntColonyOptimizer(
            num_ants=kwargs.get("num_ants", 20),
            num_nodes=kwargs.get("num_nodes", 10)
        )
        
        distance_matrix = kwargs.get("distance_matrix")
        if not distance_matrix:
            # Generate random distance matrix
            num_nodes = kwargs.get("num_nodes", 10)
            distance_matrix = self._generate_random_distances(num_nodes)
        
        solution = optimizer.optimize(
            distance_matrix,
            max_iterations=kwargs.get("max_iterations", 100)
        )
        
        self.solutions.append(solution)
        return solution
    
    def _generate_random_distances(self, num_nodes: int) -> List[List[float]]:
        """Generate random symmetric distance matrix"""
        distances = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = random.uniform(1, 100)
                distances[i][j] = dist
                distances[j][i] = dist
        
        return distances
    
    def get_best_solution(self) -> Optional[Solution]:
        """Get best solution found so far"""
        if not self.solutions:
            return None
        
        return min(self.solutions, key=lambda s: s.fitness)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get swarm statistics"""
        if not self.solutions:
            return {
                "total_solutions": 0,
                "best_fitness": None
            }
        
        best = self.get_best_solution()
        
        return {
            "behavior": self.behavior.value,
            "total_solutions": len(self.solutions),
            "best_fitness": best.fitness,
            "avg_fitness": sum(s.fitness for s in self.solutions) / len(self.solutions),
            "best_discovered_by": best.discovered_by
        }


def demonstrate_swarm_intelligence():
    """Demonstrate swarm intelligence system"""
    print("=" * 70)
    print("Swarm Intelligence System Demonstration")
    print("=" * 70)
    
    # Example 1: Particle Swarm Optimization
    print("\n1. Particle Swarm Optimization (Minimize Sphere Function):")
    
    pso_system = SwarmIntelligenceSystem(SwarmBehavior.PARTICLE_SWARM)
    
    # Define fitness function (sphere function: minimize sum of squares)
    sphere_function = FitnessFunction(lambda x: sum(xi**2 for xi in x))
    
    solution = pso_system.solve(
        problem_type="continuous_optimization",
        fitness_function=sphere_function,
        num_particles=30,
        dimensions=3,
        bounds=(-5.0, 5.0),
        max_iterations=50
    )
    
    print("Best solution found:")
    print("  Position: {}".format(
        [round(x, 4) for x in solution.position.coordinates]
    ))
    print("  Fitness: {:.6f}".format(solution.fitness))
    print("  Iterations: {}".format(solution.iterations))
    print("  Evaluations: {}".format(sphere_function.evaluations))
    
    # Example 2: Ant Colony Optimization
    print("\n2. Ant Colony Optimization (Traveling Salesman Problem):")
    
    aco_system = SwarmIntelligenceSystem(SwarmBehavior.ANT_COLONY)
    
    # Create small distance matrix for demo
    distance_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]
    
    solution = aco_system.solve(
        problem_type="tsp",
        distance_matrix=distance_matrix,
        num_ants=10,
        max_iterations=50
    )
    
    print("Best tour found:")
    print("  Tour: {}".format(
        [int(x) for x in solution.position.coordinates]
    ))
    print("  Length: {:.2f}".format(solution.fitness))
    print("  Iterations: {}".format(solution.iterations))
    
    # Example 3: Multiple runs comparison
    print("\n3. Multiple PSO Runs Comparison:")
    
    results = []
    for run in range(5):
        system = SwarmIntelligenceSystem(SwarmBehavior.PARTICLE_SWARM)
        
        rastrigin = FitnessFunction(
            lambda x: 10 * len(x) + sum(xi**2 - 10 * math.cos(2 * math.pi * xi) for xi in x)
        )
        
        sol = system.solve(
            problem_type="continuous_optimization",
            fitness_function=rastrigin,
            num_particles=20,
            dimensions=2,
            bounds=(-5.12, 5.12),
            max_iterations=30
        )
        
        results.append(sol.fitness)
        print("  Run {}: Fitness = {:.6f}".format(run + 1, sol.fitness))
    
    print("\n  Average fitness: {:.6f}".format(sum(results) / len(results)))
    print("  Best fitness: {:.6f}".format(min(results)))
    
    # Example 4: Statistics
    print("\n4. System Statistics:")
    stats = pso_system.get_statistics()
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    demonstrate_swarm_intelligence()
