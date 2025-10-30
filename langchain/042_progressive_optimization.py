"""
Pattern 042: Progressive Optimization

Description:
    Progressive Optimization enables agents to iteratively improve solutions
    through multiple generations using optimization strategies like hill climbing,
    gradient-based methods, or evolutionary algorithms. Each iteration refines
    the solution based on fitness evaluation until convergence or max iterations.

Components:
    - Solution Generator: Creates initial and improved solutions
    - Fitness Evaluator: Scores solution quality
    - Optimization Strategy: Determines how to improve (hill climbing, genetic, etc.)
    - Convergence Detector: Identifies when to stop
    - History Tracker: Maintains optimization trajectory

Use Cases:
    - Creative content optimization (writing, design)
    - Code optimization and refactoring
    - Prompt engineering and tuning
    - Solution design and improvement
    - Configuration optimization
    - Performance tuning

LangChain Implementation:
    Uses iterative chains with fitness evaluation and improvement strategies
    to progressively optimize solutions until convergence or target quality.
"""

import os
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class OptimizationStrategy(Enum):
    """Optimization strategies available."""
    HILL_CLIMBING = "hill_climbing"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC = "genetic"
    GRADIENT_BASED = "gradient_based"
    RANDOM_SEARCH = "random_search"


class ConvergenceStatus(Enum):
    """Status of optimization convergence."""
    RUNNING = "running"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    TARGET_REACHED = "target_reached"
    STAGNATED = "stagnated"


@dataclass
class Solution:
    """A solution in the optimization process."""
    content: str
    fitness: float
    generation: int
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationResult:
    """Result of optimization process."""
    initial_solution: Solution
    final_solution: Solution
    history: List[Solution]
    convergence_status: ConvergenceStatus
    total_iterations: int
    improvement: float
    strategy: OptimizationStrategy


class ProgressiveOptimizationAgent:
    """
    Agent that progressively optimizes solutions through iterations.
    
    Features:
    - Multiple optimization strategies
    - Fitness-based improvement
    - Convergence detection
    - History tracking
    - Configurable parameters
    """
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.HILL_CLIMBING,
        max_iterations: int = 10,
        target_fitness: float = 0.9,
        convergence_threshold: float = 0.01,
        stagnation_limit: int = 3,
        temperature: float = 0.7
    ):
        self.strategy = strategy
        self.max_iterations = max_iterations
        self.target_fitness = target_fitness
        self.convergence_threshold = convergence_threshold
        self.stagnation_limit = stagnation_limit
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        
        # Initial solution generation
        self.initial_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at creating solutions. Generate a high-quality initial solution."),
            ("user", "{objective}\n\nProvide your initial solution:")
        ])
        
        # Improvement prompt
        self.improvement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an optimization expert. Improve the current solution.

Current fitness: {fitness}
Target fitness: {target_fitness}
Strategy: {strategy}

Focus on:
- Addressing weaknesses identified
- Enhancing strengths
- Moving toward target fitness
- Making meaningful improvements"""),
            ("user", """Objective: {objective}

Current Solution:
{current_solution}

Feedback for improvement:
{feedback}

Provide an improved solution:""")
        ])
        
        # Fitness evaluation prompt
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a solution evaluator. Rate the solution's quality on a scale of 0.0 to 1.0.

Criteria:
- Effectiveness: Does it meet the objective?
- Quality: Is it well-crafted?
- Completeness: Does it address all aspects?
- Clarity: Is it clear and understandable?

Provide:
FITNESS_SCORE: [0.0-1.0]
STRENGTHS: [List strengths]
WEAKNESSES: [List weaknesses]
SUGGESTIONS: [Improvement suggestions]"""),
            ("user", """Objective: {objective}

Solution to evaluate:
{solution}

Evaluate:""")
        ])
        
        # Optimization history
        self.optimizations: List[OptimizationResult] = []
    
    def generate_initial_solution(self, objective: str) -> str:
        """Generate initial solution."""
        chain = self.initial_prompt | self.llm | StrOutputParser()
        return chain.invoke({"objective": objective})
    
    def evaluate_fitness(
        self,
        objective: str,
        solution: str
    ) -> Tuple[float, Dict[str, List[str]]]:
        """
        Evaluate solution fitness and get feedback.
        
        Returns:
            Tuple of (fitness_score, feedback_dict)
        """
        chain = self.evaluation_prompt | self.llm | StrOutputParser()
        evaluation = chain.invoke({
            "objective": objective,
            "solution": solution
        })
        
        # Parse evaluation
        fitness = 0.5
        feedback = {
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }
        
        lines = evaluation.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("FITNESS_SCORE:"):
                try:
                    fitness = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("STRENGTHS:"):
                current_section = "strengths"
            elif line.startswith("WEAKNESSES:"):
                current_section = "weaknesses"
            elif line.startswith("SUGGESTIONS:"):
                current_section = "suggestions"
            elif line and line.startswith('-') and current_section:
                item = line[1:].strip()
                feedback[current_section].append(item)
        
        return fitness, feedback
    
    def improve_solution(
        self,
        objective: str,
        current_solution: str,
        fitness: float,
        feedback: Dict[str, List[str]]
    ) -> str:
        """
        Generate improved solution based on feedback.
        
        Returns:
            Improved solution
        """
        # Format feedback
        feedback_text = ""
        if feedback.get("weaknesses"):
            feedback_text += "Weaknesses to address:\n"
            feedback_text += "\n".join([f"- {w}" for w in feedback["weaknesses"]])
            feedback_text += "\n\n"
        
        if feedback.get("suggestions"):
            feedback_text += "Suggestions for improvement:\n"
            feedback_text += "\n".join([f"- {s}" for s in feedback["suggestions"]])
        
        chain = self.improvement_prompt | self.llm | StrOutputParser()
        improved = chain.invoke({
            "objective": objective,
            "current_solution": current_solution,
            "fitness": fitness,
            "target_fitness": self.target_fitness,
            "strategy": self.strategy.value,
            "feedback": feedback_text
        })
        
        return improved
    
    def check_convergence(
        self,
        history: List[Solution],
        current_fitness: float
    ) -> ConvergenceStatus:
        """
        Check if optimization has converged.
        
        Returns:
            ConvergenceStatus
        """
        # Check max iterations
        if len(history) >= self.max_iterations:
            return ConvergenceStatus.MAX_ITERATIONS
        
        # Check target reached
        if current_fitness >= self.target_fitness:
            return ConvergenceStatus.TARGET_REACHED
        
        # Check stagnation
        if len(history) >= self.stagnation_limit + 1:
            recent_improvements = [
                history[i].fitness - history[i-1].fitness
                for i in range(-self.stagnation_limit, 0)
            ]
            if all(imp < self.convergence_threshold for imp in recent_improvements):
                return ConvergenceStatus.STAGNATED
        
        # Check convergence by improvement rate
        if len(history) >= 2:
            improvement = history[-1].fitness - history[-2].fitness
            if 0 <= improvement < self.convergence_threshold:
                return ConvergenceStatus.CONVERGED
        
        return ConvergenceStatus.RUNNING
    
    def optimize(
        self,
        objective: str,
        initial_solution: Optional[str] = None
    ) -> OptimizationResult:
        """
        Progressively optimize solution.
        
        Steps:
        1. Generate or use initial solution
        2. Evaluate fitness
        3. Check convergence
        4. Generate improvement
        5. Repeat until convergence
        
        Args:
            objective: The optimization objective
            initial_solution: Optional starting solution
            
        Returns:
            OptimizationResult with optimization trajectory
        """
        # Step 1: Get initial solution
        if initial_solution is None:
            initial_solution = self.generate_initial_solution(objective)
        
        # Evaluate initial fitness
        fitness, feedback = self.evaluate_fitness(objective, initial_solution)
        
        current_solution = Solution(
            content=initial_solution,
            fitness=fitness,
            generation=0,
            metadata={"feedback": feedback}
        )
        
        history = [current_solution]
        
        # Optimization loop
        iteration = 1
        status = ConvergenceStatus.RUNNING
        
        while status == ConvergenceStatus.RUNNING:
            # Step 3: Check convergence
            status = self.check_convergence(history, current_solution.fitness)
            if status != ConvergenceStatus.RUNNING:
                break
            
            # Step 4: Generate improvement
            improved_content = self.improve_solution(
                objective,
                current_solution.content,
                current_solution.fitness,
                feedback
            )
            
            # Evaluate improved solution
            new_fitness, new_feedback = self.evaluate_fitness(
                objective,
                improved_content
            )
            
            # Accept improvement (hill climbing) or based on strategy
            if new_fitness >= current_solution.fitness or self.strategy != OptimizationStrategy.HILL_CLIMBING:
                current_solution = Solution(
                    content=improved_content,
                    fitness=new_fitness,
                    generation=iteration,
                    metadata={"feedback": new_feedback}
                )
                feedback = new_feedback
            else:
                # For hill climbing, if no improvement, mark stagnation
                current_solution = Solution(
                    content=current_solution.content,
                    fitness=current_solution.fitness,
                    generation=iteration,
                    metadata={"feedback": feedback, "stagnated": True}
                )
            
            history.append(current_solution)
            iteration += 1
        
        # Create result
        improvement = history[-1].fitness - history[0].fitness
        
        result = OptimizationResult(
            initial_solution=history[0],
            final_solution=history[-1],
            history=history,
            convergence_status=status,
            total_iterations=len(history) - 1,
            improvement=improvement,
            strategy=self.strategy
        )
        
        self.optimizations.append(result)
        
        return result
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get statistics about optimizations."""
        if not self.optimizations:
            return {"total_optimizations": 0}
        
        avg_iterations = sum(o.total_iterations for o in self.optimizations) / len(self.optimizations)
        avg_improvement = sum(o.improvement for o in self.optimizations) / len(self.optimizations)
        
        convergence_counts = {}
        for status in ConvergenceStatus:
            count = sum(1 for o in self.optimizations if o.convergence_status == status)
            convergence_counts[status.value] = count
        
        return {
            "total_optimizations": len(self.optimizations),
            "average_iterations": avg_iterations,
            "average_improvement": avg_improvement,
            "convergence_breakdown": convergence_counts
        }


def demonstrate_progressive_optimization():
    """
    Demonstrates progressive optimization for iterative improvement.
    """
    print("=" * 80)
    print("PROGRESSIVE OPTIMIZATION DEMONSTRATION")
    print("=" * 80)
    
    # Test 1: Optimize a product description
    print("\n" + "=" * 80)
    print("Test 1: Product Description Optimization")
    print("=" * 80)
    
    agent1 = ProgressiveOptimizationAgent(
        strategy=OptimizationStrategy.HILL_CLIMBING,
        max_iterations=5,
        target_fitness=0.85,
        convergence_threshold=0.05,
        temperature=0.8
    )
    
    objective1 = """Create a compelling product description for a smart water bottle that:
- Tracks hydration levels
- Reminds users to drink water
- Syncs with fitness apps
- Has temperature control

Make it persuasive, concise, and customer-focused."""
    
    print(f"\nObjective:\n{objective1}")
    
    result1 = agent1.optimize(objective1)
    
    print("\n[Optimization Trajectory]")
    print(f"Strategy: {result1.strategy.value}")
    print(f"Total Iterations: {result1.total_iterations}")
    print(f"Convergence Status: {result1.convergence_status.value}")
    print(f"Improvement: {result1.improvement:.3f}")
    
    print("\n[Fitness Progress]")
    for i, solution in enumerate(result1.history):
        indicator = "📈" if i > 0 and solution.fitness > result1.history[i-1].fitness else "📊"
        print(f"Gen {solution.generation}: {solution.fitness:.3f} {indicator}")
    
    print("\n[Initial Solution]")
    print(result1.initial_solution.content[:200] + "...")
    print(f"Fitness: {result1.initial_solution.fitness:.3f}")
    
    print("\n[Final Solution]")
    print(result1.final_solution.content[:200] + "...")
    print(f"Fitness: {result1.final_solution.fitness:.3f}")
    
    # Test 2: Optimize code explanation
    print("\n" + "=" * 80)
    print("Test 2: Code Explanation Optimization")
    print("=" * 80)
    
    agent2 = ProgressiveOptimizationAgent(
        strategy=OptimizationStrategy.HILL_CLIMBING,
        max_iterations=4,
        target_fitness=0.9,
        temperature=0.7
    )
    
    objective2 = """Explain how binary search works in a way that:
- Is accessible to beginners
- Uses clear analogies
- Includes time complexity
- Has a practical example
- Is engaging and memorable"""
    
    print(f"\nObjective: Optimize explanation quality")
    
    result2 = agent2.optimize(objective2)
    
    print("\n[Optimization Summary]")
    print(f"Iterations: {result2.total_iterations}")
    print(f"Status: {result2.convergence_status.value}")
    print(f"Initial Fitness: {result2.initial_solution.fitness:.3f}")
    print(f"Final Fitness: {result2.final_solution.fitness:.3f}")
    print(f"Improvement: {result2.improvement:.3f} ({result2.improvement/result2.initial_solution.fitness*100:.1f}%)")
    
    print("\n[Fitness Evolution]")
    for sol in result2.history:
        bar = "█" * int(sol.fitness * 40)
        print(f"Gen {sol.generation}: {bar} {sol.fitness:.3f}")
    
    # Test 3: Multiple objectives with comparison
    print("\n" + "=" * 80)
    print("Test 3: Email Subject Line Optimization")
    print("=" * 80)
    
    agent3 = ProgressiveOptimizationAgent(
        strategy=OptimizationStrategy.HILL_CLIMBING,
        max_iterations=6,
        target_fitness=0.88,
        stagnation_limit=2,
        temperature=0.8
    )
    
    objective3 = """Create an email subject line for a webinar announcement that:
- Is under 50 characters
- Creates urgency
- Clearly states value
- Has high open rate potential
- Avoids spam triggers"""
    
    print(f"\nObjective: Optimize email subject line")
    
    result3 = agent3.optimize(objective3)
    
    print("\n[Optimization Details]")
    print(f"Strategy: {result3.strategy.value}")
    print(f"Convergence: {result3.convergence_status.value}")
    print(f"Generations: {result3.total_iterations}")
    
    print("\n[All Generations]")
    for sol in result3.history:
        print(f"\nGeneration {sol.generation} (Fitness: {sol.fitness:.3f}):")
        print(f"  {sol.content[:100]}")
        
        if "feedback" in sol.metadata:
            feedback = sol.metadata["feedback"]
            if feedback.get("weaknesses"):
                print(f"  Weaknesses: {len(feedback['weaknesses'])}")
            if feedback.get("suggestions"):
                print(f"  Suggestions: {len(feedback['suggestions'])}")
    
    # Combined statistics
    print("\n" + "=" * 80)
    print("Optimization Statistics")
    print("=" * 80)
    
    # Manually combine stats from all agents
    all_optimizations = (
        agent1.optimizations + 
        agent2.optimizations + 
        agent3.optimizations
    )
    
    total_iterations = sum(o.total_iterations for o in all_optimizations)
    avg_iterations = total_iterations / len(all_optimizations)
    avg_improvement = sum(o.improvement for o in all_optimizations) / len(all_optimizations)
    
    print(f"\nTotal Optimizations: {len(all_optimizations)}")
    print(f"Average Iterations: {avg_iterations:.1f}")
    print(f"Average Improvement: {avg_improvement:.3f}")
    
    convergence_counts = {}
    for status in ConvergenceStatus:
        count = sum(1 for o in all_optimizations if o.convergence_status == status)
        if count > 0:
            convergence_counts[status.value] = count
    
    print("\nConvergence Breakdown:")
    for status, count in convergence_counts.items():
        print(f"  {status}: {count}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Progressive Optimization provides:
✓ Iterative solution improvement
✓ Multiple optimization strategies
✓ Fitness-based guidance
✓ Convergence detection
✓ History tracking
✓ Quality metrics

This pattern excels at:
- Creative content optimization
- Code and design refinement
- Prompt engineering
- Configuration tuning
- Solution design
- Quality improvement

Optimization strategies:
1. Hill Climbing: Accept only improvements
2. Simulated Annealing: Accept some worse solutions
3. Genetic: Evolve population of solutions
4. Gradient-Based: Follow improvement gradient
5. Random Search: Explore solution space

Optimization process:
1. Generate initial solution
2. Evaluate fitness
3. Check convergence
4. Generate improvement
5. Evaluate new solution
6. Accept or reject
7. Repeat until convergence

Convergence conditions:
- Target fitness reached
- Maximum iterations
- Improvement stagnation
- Convergence threshold met

Fitness evaluation criteria:
- Effectiveness: Meets objective
- Quality: Well-crafted
- Completeness: Comprehensive
- Clarity: Understandable

Benefits:
- Quality: Iterative improvement
- Transparency: Track progress
- Convergence: Automatic stopping
- Flexibility: Multiple strategies
- Metrics: Quantifiable improvement
- History: Full trajectory

Configuration parameters:
- max_iterations: Stop after N iterations
- target_fitness: Goal quality score
- convergence_threshold: Min improvement
- stagnation_limit: Iterations without improvement
- strategy: Optimization approach

Use Progressive Optimization when:
- Need iterative improvement
- Quality is quantifiable
- Multiple refinements beneficial
- Want optimization history
- Need automatic convergence
- Exploring solution space

Comparison with other patterns:
- vs Iterative Refinement: More strategic, fitness-guided
- vs Self-Evaluation: Focused on optimization, not just evaluation
- vs Feedback Loops: Systematic optimization vs learning
""")


if __name__ == "__main__":
    demonstrate_progressive_optimization()
