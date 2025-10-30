"""
Progressive Optimization Pattern
Iteratively optimizes solution through generations
"""
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import random
class OptimizationMethod(Enum):
    HILL_CLIMBING = "hill_climbing"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    GRADIENT_DESCENT = "gradient_descent"
@dataclass
class Solution:
    """A candidate solution"""
    id: int
    parameters: Dict[str, float]
    fitness: float
    generation: int
@dataclass
class OptimizationResult:
    """Result of optimization process"""
    best_solution: Solution
    all_solutions: List[Solution]
    generations: int
    convergence_history: List[float]
class ProgressiveOptimizer:
    """Optimizer using progressive refinement"""
    def __init__(self, objective_function: Callable, method: OptimizationMethod = OptimizationMethod.HILL_CLIMBING):
        self.objective_function = objective_function
        self.method = method
        self.solution_counter = 0
    def optimize(self, initial_params: Dict[str, float], 
                 max_generations: int = 50,
                 population_size: int = 10) -> OptimizationResult:
        """Run progressive optimization"""
        print(f"\n{'='*70}")
        print(f"PROGRESSIVE OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Method: {self.method.value}")
        print(f"Max Generations: {max_generations}")
        print(f"Initial Parameters: {initial_params}\n")
        if self.method == OptimizationMethod.HILL_CLIMBING:
            return self._hill_climbing(initial_params, max_generations)
        elif self.method == OptimizationMethod.SIMULATED_ANNEALING:
            return self._simulated_annealing(initial_params, max_generations)
        elif self.method == OptimizationMethod.GENETIC_ALGORITHM:
            return self._genetic_algorithm(initial_params, max_generations, population_size)
        else:
            raise ValueError(f"Method {self.method} not implemented")
    def _hill_climbing(self, initial_params: Dict[str, float], max_generations: int) -> OptimizationResult:
        """Hill climbing optimization"""
        current = self._create_solution(initial_params, 0)
        best = current
        all_solutions = [current]
        convergence_history = [current.fitness]
        print("Starting Hill Climbing Optimization\n")
        for generation in range(1, max_generations):
            # Generate neighbors
            neighbors = self._generate_neighbors(current, generation)
            # Find best neighbor
            best_neighbor = max(neighbors, key=lambda s: s.fitness)
            # Only move if neighbor is better
            if best_neighbor.fitness > current.fitness:
                current = best_neighbor
                improvement = best_neighbor.fitness - convergence_history[-1]
                print(f"Generation {generation}: Fitness={current.fitness:.4f} (↑{improvement:.4f})")
                if current.fitness > best.fitness:
                    best = current
            else:
                print(f"Generation {generation}: No improvement (stuck at {current.fitness:.4f})")
            all_solutions.extend(neighbors)
            convergence_history.append(current.fitness)
            # Early stopping if no improvement for several generations
            if len(convergence_history) > 10:
                recent = convergence_history[-10:]
                if all(abs(x - recent[0]) < 0.0001 for x in recent):
                    print(f"\nConverged at generation {generation}")
                    break
        return OptimizationResult(
            best_solution=best,
            all_solutions=all_solutions,
            generations=generation,
            convergence_history=convergence_history
        )
    def _simulated_annealing(self, initial_params: Dict[str, float], max_generations: int) -> OptimizationResult:
        """Simulated annealing optimization"""
        import math
        current = self._create_solution(initial_params, 0)
        best = current
        all_solutions = [current]
        convergence_history = [current.fitness]
        temperature = 1.0
        cooling_rate = 0.95
        print("Starting Simulated Annealing Optimization\n")
        for generation in range(1, max_generations):
            # Generate random neighbor
            neighbors = self._generate_neighbors(current, generation, count=1)
            neighbor = neighbors[0]
            # Calculate acceptance probability
            delta = neighbor.fitness - current.fitness
            if delta > 0:
                # Better solution - always accept
                current = neighbor
                accept = True
            else:
                # Worse solution - accept with probability
                probability = math.exp(delta / temperature)
                accept = random.random() < probability
                if accept:
                    current = neighbor
            if current.fitness > best.fitness:
                best = current
            status = "✓ Accepted" if accept else "✗ Rejected"
            print(f"Generation {generation}: Fitness={current.fitness:.4f}, Temp={temperature:.3f} - {status}")
            all_solutions.append(neighbor)
            convergence_history.append(current.fitness)
            # Cool down
            temperature *= cooling_rate
        return OptimizationResult(
            best_solution=best,
            all_solutions=all_solutions,
            generations=max_generations,
            convergence_history=convergence_history
        )
    def _genetic_algorithm(self, initial_params: Dict[str, float], 
                           max_generations: int, population_size: int) -> OptimizationResult:
        """Genetic algorithm optimization"""
        # Initialize population
        population = [
            self._create_solution(
                self._mutate_params(initial_params, 0.3),
                0
            )
            for _ in range(population_size)
        ]
        best = max(population, key=lambda s: s.fitness)
        all_solutions = list(population)
        convergence_history = [best.fitness]
        print("Starting Genetic Algorithm Optimization\n")
        print(f"Initial Population: {population_size} solutions")
        print(f"Best Initial Fitness: {best.fitness:.4f}\n")
        for generation in range(1, max_generations):
            # Selection
            population.sort(key=lambda s: s.fitness, reverse=True)
            survivors = population[:population_size // 2]
            # Crossover and mutation
            offspring = []
            while len(offspring) < population_size - len(survivors):
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                child_params = self._crossover(parent1.parameters, parent2.parameters)
                child_params = self._mutate_params(child_params, 0.1)
                child = self._create_solution(child_params, generation)
                offspring.append(child)
            # New population
            population = survivors + offspring
            # Track best
            gen_best = max(population, key=lambda s: s.fitness)
            if gen_best.fitness > best.fitness:
                improvement = gen_best.fitness - best.fitness
                print(f"Generation {generation}: Best={gen_best.fitness:.4f} (↑{improvement:.4f}) - NEW BEST!")
                best = gen_best
            else:
                print(f"Generation {generation}: Best={gen_best.fitness:.4f}")
            all_solutions.extend(population)
            convergence_history.append(best.fitness)
        return OptimizationResult(
            best_solution=best,
            all_solutions=all_solutions,
            generations=max_generations,
            convergence_history=convergence_history
        )
    def _create_solution(self, parameters: Dict[str, float], generation: int) -> Solution:
        """Create a solution and evaluate it"""
        fitness = self.objective_function(parameters)
        self.solution_counter += 1
        return Solution(
            id=self.solution_counter,
            parameters=parameters.copy(),
            fitness=fitness,
            generation=generation
        )
    def _generate_neighbors(self, solution: Solution, generation: int, count: int = 5) -> List[Solution]:
        """Generate neighboring solutions"""
        neighbors = []
        for _ in range(count):
            # Small random perturbation
            new_params = self._mutate_params(solution.parameters, 0.1)
            neighbor = self._create_solution(new_params, generation)
            neighbors.append(neighbor)
        return neighbors
    def _mutate_params(self, parameters: Dict[str, float], mutation_rate: float) -> Dict[str, float]:
        """Mutate parameters"""
        mutated = {}
        for key, value in parameters.items():
            if random.random() < mutation_rate:
                # Apply random change
                change = random.gauss(0, 0.1)
                mutated[key] = max(-1.0, min(1.0, value + change))  # Clamp to [-1, 1]
            else:
                mutated[key] = value
        return mutated
    def _crossover(self, params1: Dict[str, float], params2: Dict[str, float]) -> Dict[str, float]:
        """Crossover two parameter sets"""
        child = {}
        for key in params1.keys():
            # Random mix of parents
            if random.random() < 0.5:
                child[key] = params1[key]
            else:
                child[key] = params2[key]
        return child
# Example objective functions
def sphere_function(params: Dict[str, float]) -> float:
    """Simple sphere function (minimize sum of squares)"""
    # Return negative because we're maximizing
    return -sum(x**2 for x in params.values())
def rastrigin_function(params: Dict[str, float]) -> float:
    """Rastrigin function (harder to optimize)"""
    import math
    A = 10
    n = len(params)
    return -(A * n + sum(x**2 - A * math.cos(2 * math.pi * x) for x in params.values()))
# Usage
if __name__ == "__main__":
    print("="*80)
    print("PROGRESSIVE OPTIMIZATION PATTERN DEMONSTRATION")
    print("="*80)
    # Initial parameters
    initial = {'x': 0.5, 'y': 0.5, 'z': 0.5}
    # Example 1: Hill Climbing
    print("\n" + "="*80)
    print("EXAMPLE 1: Hill Climbing")
    print("="*80)
    optimizer1 = ProgressiveOptimizer(sphere_function, OptimizationMethod.HILL_CLIMBING)
    result1 = optimizer1.optimize(initial, max_generations=30)
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Generations: {result1.generations}")
    print(f"Best Fitness: {result1.best_solution.fitness:.4f}")
    print(f"Best Parameters: {result1.best_solution.parameters}")
    print(f"Total Solutions Explored: {len(result1.all_solutions)}")
    # Example 2: Simulated Annealing
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Simulated Annealing")
    print("="*80)
    optimizer2 = ProgressiveOptimizer(sphere_function, OptimizationMethod.SIMULATED_ANNEALING)
    result2 = optimizer2.optimize(initial, max_generations=30)
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Generations: {result2.generations}")
    print(f"Best Fitness: {result2.best_solution.fitness:.4f}")
    print(f"Best Parameters: {result2.best_solution.parameters}")
    # Example 3: Genetic Algorithm
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Genetic Algorithm")
    print("="*80)
    optimizer3 = ProgressiveOptimizer(rastrigin_function, OptimizationMethod.GENETIC_ALGORITHM)
    result3 = optimizer3.optimize(initial, max_generations=20, population_size=20)
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Generations: {result3.generations}")
    print(f"Best Fitness: {result3.best_solution.fitness:.4f}")
    print(f"Best Parameters: {result3.best_solution.parameters}")
    print(f"Total Solutions Explored: {len(result3.all_solutions)}")
    # Plot convergence (text-based)
    print(f"\n{'='*60}")
    print("CONVERGENCE PLOT")
    print(f"{'='*60}")
    history = result3.convergence_history
    max_fitness = max(history)
    min_fitness = min(history)
    for i, fitness in enumerate(history[::5]):  # Every 5th generation
        bar_length = int(40 * (fitness - min_fitness) / (max_fitness - min_fitness + 0.001))
        bar = "█" * bar_length
        print(f"Gen {i*5:3d}: {bar} {fitness:.4f}")
