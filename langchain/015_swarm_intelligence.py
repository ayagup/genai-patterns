"""
Pattern 015: Swarm Intelligence

Description:
    The Swarm Intelligence pattern uses many simple, decentralized agents that collaborate
    through local interactions to solve complex problems. Inspired by natural phenomena like
    ant colonies, bee swarms, and bird flocking, this pattern achieves emergent intelligent
    behavior without centralized control.

Components:
    - Swarm Agents: Simple agents with local decision rules
    - Environment/Pheromone System: Shared information space
    - Local Interaction Rules: How agents influence each other
    - Emergent Behavior: Complex patterns from simple rules

Use Cases:
    - Optimization problems (path finding, resource allocation)
    - Search and exploration tasks
    - Distributed problem-solving
    - Parallel hypothesis exploration

LangChain Implementation:
    Uses multiple simple LLM agents with local decision-making, a shared information
    space (pheromone trails), iterative exploration rounds, and convergence detection
    to achieve swarm-based optimization.

Key Features:
    - Decentralized decision-making
    - Pheromone-based communication
    - Iterative convergence
    - Parallel exploration
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import random
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass
class Solution:
    """Represents a solution explored by an agent."""
    agent_id: str
    content: str
    quality_score: float
    iteration: int


@dataclass
class PheromoneTrail:
    """Represents accumulated pheromone on a solution path."""
    solution_key: str
    strength: float
    contributors: List[str] = field(default_factory=list)


class SwarmBehavior(Enum):
    """Types of swarm behaviors."""
    EXPLORATION = "exploration"  # Explore diverse solutions
    EXPLOITATION = "exploitation"  # Refine best solutions
    BALANCED = "balanced"  # Mix of both


class SwarmAgent:
    """
    A simple swarm agent that makes local decisions based on pheromones.
    """
    
    def __init__(
        self,
        agent_id: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.8
    ):
        """
        Initialize a swarm agent.
        
        Args:
            agent_id: Unique identifier for this agent
            model: LLM model to use
            temperature: Creativity/exploration level
        """
        self.agent_id = agent_id
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.local_memory: List[Solution] = []
    
    def explore_solution(
        self,
        problem: str,
        pheromone_trails: List[PheromoneTrail],
        behavior: SwarmBehavior = SwarmBehavior.BALANCED
    ) -> str:
        """
        Explore a solution based on local information and pheromones.
        
        Args:
            problem: Problem to solve
            pheromone_trails: Current pheromone information
            behavior: Exploration vs exploitation preference
            
        Returns:
            Proposed solution
        """
        # Build context from pheromone trails
        if pheromone_trails and behavior != SwarmBehavior.EXPLORATION:
            # Sort by strength and take top trails
            top_trails = sorted(pheromone_trails, key=lambda x: x.strength, reverse=True)[:3]
            trail_context = "\n".join([
                f"- Strong path: {trail.solution_key} (strength: {trail.strength:.2f})"
                for trail in top_trails
            ])
        else:
            trail_context = "No strong paths found yet. Explore freely."
        
        behavior_instructions = {
            SwarmBehavior.EXPLORATION: "Focus on exploring new, diverse approaches. Be creative and unconventional.",
            SwarmBehavior.EXPLOITATION: "Focus on refining and improving the strongest existing approaches.",
            SwarmBehavior.BALANCED: "Balance between exploring new ideas and building on promising existing approaches."
        }
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a swarm agent working as part of a collective intelligence.
Your role is to propose solutions based on local information and pheromone trails left by other agents.

{behavior_instructions[behavior]}

Keep your solution concise (2-3 sentences)."""),
            ("user", """Problem: {problem}

Pheromone trails from other agents:
{trail_context}

Propose your solution:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        solution = chain.invoke({
            "problem": problem,
            "trail_context": trail_context
        })
        
        return solution.strip()
    
    def evaluate_solution(
        self,
        problem: str,
        solution: str
    ) -> float:
        """
        Evaluate the quality of a solution.
        
        Args:
            problem: Original problem
            solution: Proposed solution
            
        Returns:
            Quality score (0.0-1.0)
        """
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an evaluator. Rate solution quality from 0.0 to 1.0."),
            ("user", """Problem: {problem}

Solution: {solution}

Rate this solution's quality (0.0 = poor, 1.0 = excellent).
Consider: correctness, completeness, creativity, practicality.

Provide only a number between 0.0 and 1.0:""")
        ])
        
        chain = eval_prompt | self.llm | StrOutputParser()
        
        try:
            score_str = chain.invoke({"problem": problem, "solution": solution})
            score = float(score_str.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5  # Default moderate score


class SwarmIntelligence:
    """
    Swarm intelligence system coordinating multiple simple agents.
    
    Uses pheromone-based communication for decentralized problem-solving.
    """
    
    def __init__(
        self,
        num_agents: int = 10,
        evaporation_rate: float = 0.1,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize swarm intelligence system.
        
        Args:
            num_agents: Number of agents in the swarm
            evaporation_rate: Rate at which pheromones decay (0.0-1.0)
            model: LLM model for agents
        """
        self.num_agents = num_agents
        self.evaporation_rate = evaporation_rate
        
        # Create swarm agents with varying temperatures for diversity
        self.agents = [
            SwarmAgent(
                agent_id=f"agent_{i}",
                model=model,
                temperature=0.6 + (i / num_agents) * 0.6  # Range: 0.6-1.2
            )
            for i in range(num_agents)
        ]
        
        # Pheromone trail system
        self.pheromones: Dict[str, PheromoneTrail] = {}
        
        # Solution history
        self.all_solutions: List[Solution] = []
    
    def _compute_solution_key(self, solution: str) -> str:
        """
        Compute a key for similar solutions.
        
        Uses first 50 characters as a simple similarity metric.
        In production, would use embeddings or semantic similarity.
        """
        # Simple approach: use first significant words
        words = solution.lower().split()[:8]
        return " ".join(words)
    
    def _deposit_pheromone(
        self,
        solution: Solution
    ):
        """
        Deposit pheromone on a solution path based on quality.
        
        Args:
            solution: Solution to deposit pheromone for
        """
        key = self._compute_solution_key(solution.content)
        
        if key in self.pheromones:
            # Strengthen existing trail
            self.pheromones[key].strength += solution.quality_score
            self.pheromones[key].contributors.append(solution.agent_id)
        else:
            # Create new trail
            self.pheromones[key] = PheromoneTrail(
                solution_key=key,
                strength=solution.quality_score,
                contributors=[solution.agent_id]
            )
    
    def _evaporate_pheromones(self):
        """Evaporate pheromones to forget old solutions."""
        keys_to_remove = []
        
        for key, trail in self.pheromones.items():
            trail.strength *= (1 - self.evaporation_rate)
            
            # Remove very weak trails
            if trail.strength < 0.1:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.pheromones[key]
    
    def _determine_behavior(self, iteration: int, max_iterations: int) -> SwarmBehavior:
        """
        Determine swarm behavior based on iteration.
        
        Early iterations: More exploration
        Later iterations: More exploitation
        """
        progress = iteration / max_iterations
        
        if progress < 0.3:
            return SwarmBehavior.EXPLORATION
        elif progress > 0.7:
            return SwarmBehavior.EXPLOITATION
        else:
            return SwarmBehavior.BALANCED
    
    def solve(
        self,
        problem: str,
        iterations: int = 5,
        agents_per_iteration: Optional[int] = None
    ) -> Tuple[Solution, List[Solution]]:
        """
        Solve a problem using swarm intelligence.
        
        Args:
            problem: Problem to solve
            iterations: Number of swarm iterations
            agents_per_iteration: Agents to activate per iteration (default: all)
            
        Returns:
            (best_solution, all_solutions)
        """
        if agents_per_iteration is None:
            agents_per_iteration = self.num_agents
        
        print(f"\n[Swarm] Starting with {self.num_agents} agents, {iterations} iterations")
        
        for iteration in range(iterations):
            behavior = self._determine_behavior(iteration, iterations)
            
            print(f"\n[Swarm] Iteration {iteration + 1}/{iterations} - Behavior: {behavior.value}")
            
            # Randomly select agents to activate this iteration
            active_agents = random.sample(self.agents, min(agents_per_iteration, self.num_agents))
            
            iteration_solutions = []
            
            for agent in active_agents:
                # Get current pheromone trails
                trails = list(self.pheromones.values())
                
                # Agent explores solution
                solution_text = agent.explore_solution(problem, trails, behavior)
                
                # Agent evaluates its own solution
                quality = agent.evaluate_solution(problem, solution_text)
                
                solution = Solution(
                    agent_id=agent.agent_id,
                    content=solution_text,
                    quality_score=quality,
                    iteration=iteration
                )
                
                iteration_solutions.append(solution)
                self.all_solutions.append(solution)
                
                # Deposit pheromone
                self._deposit_pheromone(solution)
            
            # Show best solution this iteration
            best_iteration = max(iteration_solutions, key=lambda s: s.quality_score)
            print(f"[Swarm] Best this iteration (quality: {best_iteration.quality_score:.2f}): {best_iteration.content[:80]}...")
            
            # Evaporate pheromones
            self._evaporate_pheromones()
            
            # Show pheromone trail status
            print(f"[Swarm] Active pheromone trails: {len(self.pheromones)}")
        
        # Find best overall solution
        best_solution = max(self.all_solutions, key=lambda s: s.quality_score)
        
        print(f"\n[Swarm] Convergence complete!")
        print(f"[Swarm] Best solution quality: {best_solution.quality_score:.2f}")
        
        return best_solution, self.all_solutions
    
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """
        Analyze how the swarm converged on solutions.
        
        Returns:
            Analysis of convergence patterns
        """
        # Quality progression over iterations
        iterations = max(s.iteration for s in self.all_solutions) + 1
        quality_by_iteration = [
            [s.quality_score for s in self.all_solutions if s.iteration == i]
            for i in range(iterations)
        ]
        
        avg_quality = [
            sum(scores) / len(scores) if scores else 0
            for scores in quality_by_iteration
        ]
        
        max_quality = [
            max(scores) if scores else 0
            for scores in quality_by_iteration
        ]
        
        # Solution diversity (number of unique solution keys)
        unique_solutions = len(set(
            self._compute_solution_key(s.content)
            for s in self.all_solutions
        ))
        
        # Top pheromone trails
        top_trails = sorted(
            self.pheromones.values(),
            key=lambda t: t.strength,
            reverse=True
        )[:3]
        
        return {
            "iterations": iterations,
            "total_solutions": len(self.all_solutions),
            "unique_solutions": unique_solutions,
            "avg_quality_progression": avg_quality,
            "max_quality_progression": max_quality,
            "final_avg_quality": avg_quality[-1],
            "final_max_quality": max_quality[-1],
            "top_pheromone_trails": [
                {
                    "solution": trail.solution_key,
                    "strength": trail.strength,
                    "contributors": len(trail.contributors)
                }
                for trail in top_trails
            ]
        }


def demonstrate_swarm_intelligence():
    """Demonstrate the Swarm Intelligence pattern with various examples."""
    
    print("=" * 80)
    print("SWARM INTELLIGENCE PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Test 1: Problem-solving task
    print("\n" + "=" * 80)
    print("TEST 1: Creative Problem Solving")
    print("=" * 80)
    
    swarm = SwarmIntelligence(num_agents=8, evaporation_rate=0.15)
    
    problem1 = "How can a small restaurant reduce food waste while maintaining quality?"
    
    best_solution, all_solutions = swarm.solve(problem1, iterations=4)
    
    print("\n" + "-" * 80)
    print("BEST SOLUTION:")
    print("-" * 80)
    print(f"Quality: {best_solution.quality_score:.2f}")
    print(f"From: {best_solution.agent_id} (iteration {best_solution.iteration})")
    print(f"\n{best_solution.content}")
    
    # Show convergence analysis
    analysis = swarm.get_convergence_analysis()
    
    print("\n" + "-" * 80)
    print("CONVERGENCE ANALYSIS:")
    print("-" * 80)
    print(f"Total solutions explored: {analysis['total_solutions']}")
    print(f"Unique solution approaches: {analysis['unique_solutions']}")
    print(f"Quality improvement: {analysis['avg_quality_progression'][0]:.2f} -> {analysis['final_avg_quality']:.2f}")
    
    print("\nQuality progression by iteration:")
    for i, (avg, max_q) in enumerate(zip(analysis['avg_quality_progression'], analysis['max_quality_progression'])):
        print(f"  Iteration {i+1}: avg={avg:.2f}, max={max_q:.2f}")
    
    print("\nTop pheromone trails (strongest convergence):")
    for i, trail in enumerate(analysis['top_pheromone_trails'], 1):
        print(f"  {i}. {trail['solution'][:60]}...")
        print(f"     Strength: {trail['strength']:.2f}, Contributors: {trail['contributors']}")
    
    # Test 2: Optimization task
    print("\n" + "=" * 80)
    print("TEST 2: Optimization Problem")
    print("=" * 80)
    
    swarm2 = SwarmIntelligence(num_agents=6, evaporation_rate=0.2)
    
    problem2 = "Design an efficient morning routine for a busy professional with 90 minutes."
    
    best_solution2, _ = swarm2.solve(problem2, iterations=3, agents_per_iteration=4)
    
    print("\n" + "-" * 80)
    print("BEST SOLUTION:")
    print("-" * 80)
    print(f"Quality: {best_solution2.quality_score:.2f}")
    print(f"\n{best_solution2.content}")
    
    # Test 3: Compare with different swarm sizes
    print("\n" + "=" * 80)
    print("TEST 3: Swarm Size Comparison")
    print("=" * 80)
    
    problem3 = "What are three innovative ways to improve employee engagement?"
    
    swarm_sizes = [4, 8, 12]
    results = []
    
    for size in swarm_sizes:
        print(f"\nTesting swarm size: {size}")
        swarm = SwarmIntelligence(num_agents=size, evaporation_rate=0.15)
        best, all_sols = swarm.solve(problem3, iterations=3, agents_per_iteration=size)
        analysis = swarm.get_convergence_analysis()
        
        results.append({
            "size": size,
            "best_quality": best.quality_score,
            "unique_solutions": analysis['unique_solutions'],
            "solution": best.content
        })
    
    print("\n" + "-" * 80)
    print("COMPARISON RESULTS:")
    print("-" * 80)
    for result in results:
        print(f"\nSwarm size {result['size']}:")
        print(f"  Best quality: {result['best_quality']:.2f}")
        print(f"  Unique approaches: {result['unique_solutions']}")
        print(f"  Solution: {result['solution'][:100]}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Swarm Intelligence pattern demonstrates several key benefits:

1. **Decentralized Problem-Solving**: No single coordinator needed
2. **Emergent Intelligence**: Complex solutions from simple agent rules
3. **Robustness**: Failure of individual agents doesn't stop the swarm
4. **Parallel Exploration**: Multiple solutions explored simultaneously

Key Mechanisms:
- **Pheromone Trails**: Shared information about solution quality
- **Evaporation**: Forgetting of old/poor solutions over time
- **Local Decisions**: Agents decide based on nearby information
- **Convergence**: Swarm naturally converges on good solutions

Behavior Phases:
- **Exploration**: Early iterations explore diverse approaches
- **Balanced**: Middle iterations balance exploration and exploitation
- **Exploitation**: Late iterations refine best solutions

Use Cases:
- Optimization problems (routing, scheduling, design)
- Search and exploration tasks
- Distributed problem-solving
- Situations requiring diverse perspectives

The pattern is particularly effective when:
- Problem space is large and complex
- Multiple good solutions may exist
- Parallel exploration is beneficial
- Robustness to individual failures is important
- Emergent intelligence is desired over centralized control
""")


if __name__ == "__main__":
    demonstrate_swarm_intelligence()
