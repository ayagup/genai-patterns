"""
Pattern 017: Competitive Multi-Agent

Description:
    The Competitive Multi-Agent pattern uses multiple agents that compete to produce
    the best solution. Agents work independently, their solutions are evaluated against
    each other, and the winning solution is selected. This competition drives quality
    improvement and innovation.

Components:
    - Competitor Agents: Multiple agents with different strategies
    - Evaluation System: Judges solutions across multiple criteria
    - Competition Framework: Manages rounds and determines winners
    - Selection Mechanism: Chooses best solution(s)

Use Cases:
    - Creative tasks (writing, design, ideation)
    - Optimization problems with multiple local optima
    - Generating diverse solution approaches
    - Quality improvement through competition

LangChain Implementation:
    Uses multiple LLM agents with different configurations competing on tasks,
    implements multi-criteria evaluation, tournament-style competition, and
    quality-driven selection mechanisms.

Key Features:
    - Tournament-style competitions
    - Multi-criteria evaluation
    - Diverse agent strategies
    - Winner selection and ranking
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class CompetitionMode(Enum):
    """Modes of competition."""
    TOURNAMENT = "tournament"  # Bracket-style elimination
    ROUND_ROBIN = "round_robin"  # All agents compete in every round
    SURVIVAL = "survival"  # Lowest scorers eliminated each round


@dataclass
class Solution:
    """Represents a solution from a competitor."""
    agent_id: str
    content: str
    round_number: int
    scores: Dict[str, float] = None  # Criterion -> Score
    total_score: float = 0.0
    rank: Optional[int] = None


@dataclass
class CompetitionResult:
    """Results from a competition."""
    winner: Solution
    all_solutions: List[Solution]
    rounds: int
    mode: str


class CompetitorAgent:
    """
    An agent that competes to produce the best solution.
    """
    
    def __init__(
        self,
        agent_id: str,
        strategy: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        """
        Initialize a competitor agent.
        
        Args:
            agent_id: Unique identifier
            strategy: Competitive strategy description
            model: LLM model to use
            temperature: Creativity level
        """
        self.agent_id = agent_id
        self.strategy = strategy
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.wins = 0
        self.total_score = 0.0
    
    def generate_solution(
        self,
        problem: str,
        round_number: int,
        previous_best: Optional[Solution] = None
    ) -> Solution:
        """
        Generate a solution for the problem.
        
        Args:
            problem: Problem to solve
            round_number: Current round number
            previous_best: Previous best solution (for improvement)
            
        Returns:
            Generated solution
        """
        strategy_context = f"\nYour strategy: {self.strategy}"
        
        if previous_best and round_number > 1:
            improvement_context = f"\n\nPrevious best solution to beat:\n{previous_best.content}\n\nImprove upon this!"
        else:
            improvement_context = ""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a competitor in a solution competition.{strategy_context}

Generate your BEST solution. Be creative, thorough, and competitive!"""),
            ("user", """Problem: {problem}{improvement_context}

Your solution (aim to win the competition!):""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        content = chain.invoke({
            "problem": problem,
            "improvement_context": improvement_context
        })
        
        return Solution(
            agent_id=self.agent_id,
            content=content.strip(),
            round_number=round_number
        )


class CompetitionJudge:
    """
    Evaluates and ranks solutions in competitions.
    """
    
    def __init__(
        self,
        criteria: List[str],
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the judge.
        
        Args:
            criteria: List of evaluation criteria
            model: LLM model for judging
        """
        self.criteria = criteria
        self.llm = ChatOpenAI(model=model, temperature=0.2)
    
    def evaluate_solution(
        self,
        problem: str,
        solution: Solution
    ) -> Solution:
        """
        Evaluate a solution across all criteria.
        
        Args:
            problem: Original problem
            solution: Solution to evaluate
            
        Returns:
            Solution with scores populated
        """
        scores = {}
        
        for criterion in self.criteria:
            score = self._score_criterion(problem, solution.content, criterion)
            scores[criterion] = score
        
        solution.scores = scores
        solution.total_score = sum(scores.values()) / len(scores)
        
        return solution
    
    def _score_criterion(
        self,
        problem: str,
        solution: str,
        criterion: str
    ) -> float:
        """Score a solution on a specific criterion."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are an expert judge evaluating solutions based on: {criterion}"),
            ("user", """Problem: {problem}

Solution: {solution}

Rate this solution on {criterion} from 0.0 (worst) to 10.0 (best).
Consider only this specific criterion.

Provide only a number between 0.0 and 10.0:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            score_str = chain.invoke({
                "problem": problem,
                "solution": solution,
                "criterion": criterion
            })
            score = float(score_str.strip())
            return max(0.0, min(10.0, score))
        except:
            return 5.0  # Default moderate score
    
    def rank_solutions(
        self,
        solutions: List[Solution]
    ) -> List[Solution]:
        """
        Rank solutions by total score.
        
        Args:
            solutions: List of solutions to rank
            
        Returns:
            Ranked solutions (highest to lowest)
        """
        sorted_solutions = sorted(
            solutions,
            key=lambda s: s.total_score,
            reverse=True
        )
        
        for rank, solution in enumerate(sorted_solutions, 1):
            solution.rank = rank
        
        return sorted_solutions


class CompetitiveSystem:
    """
    Manages competitive multi-agent system.
    """
    
    def __init__(
        self,
        evaluation_criteria: Optional[List[str]] = None,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize competitive system.
        
        Args:
            evaluation_criteria: Criteria for evaluating solutions
            model: LLM model for agents
        """
        if evaluation_criteria is None:
            evaluation_criteria = [
                "Creativity and Innovation",
                "Practicality and Feasibility",
                "Completeness and Detail",
                "Clarity and Communication"
            ]
        
        self.criteria = evaluation_criteria
        self.judge = CompetitionJudge(evaluation_criteria, model)
        self.agents: List[CompetitorAgent] = []
        self.model = model
    
    def add_competitor(
        self,
        agent_id: str,
        strategy: str,
        temperature: float = 0.7
    ):
        """Add a competitor to the system."""
        agent = CompetitorAgent(agent_id, strategy, self.model, temperature)
        self.agents.append(agent)
    
    def run_competition(
        self,
        problem: str,
        mode: CompetitionMode = CompetitionMode.ROUND_ROBIN,
        rounds: int = 2
    ) -> CompetitionResult:
        """
        Run a competition among agents.
        
        Args:
            problem: Problem for agents to solve
            mode: Competition mode
            rounds: Number of competition rounds
            
        Returns:
            Competition results
        """
        print(f"\n[Competition] Starting {mode.value} with {len(self.agents)} competitors")
        print(f"[Competition] {rounds} rounds, criteria: {', '.join(self.criteria)}\n")
        
        all_solutions: List[Solution] = []
        current_best: Optional[Solution] = None
        
        for round_num in range(1, rounds + 1):
            print(f"{'=' * 80}")
            print(f"ROUND {round_num}/{rounds}")
            print(f"{'=' * 80}\n")
            
            round_solutions = []
            
            # Each agent generates a solution
            for agent in self.agents:
                print(f"[{agent.agent_id}] Generating solution (strategy: {agent.strategy})...")
                
                solution = agent.generate_solution(
                    problem,
                    round_num,
                    current_best
                )
                
                # Evaluate solution
                solution = self.judge.evaluate_solution(problem, solution)
                
                print(f"[{agent.agent_id}] Score: {solution.total_score:.2f}/10.0")
                print(f"[{agent.agent_id}] Solution: {solution.content[:80]}...\n")
                
                round_solutions.append(solution)
                all_solutions.append(solution)
            
            # Rank this round's solutions
            ranked = self.judge.rank_solutions(round_solutions)
            
            print(f"\n[Judge] Round {round_num} Rankings:")
            for solution in ranked[:3]:  # Top 3
                print(f"  #{solution.rank} {solution.agent_id}: {solution.total_score:.2f}/10.0")
            
            # Update current best
            if ranked[0].total_score > (current_best.total_score if current_best else 0):
                current_best = ranked[0]
                print(f"\n[Judge] New leader: {current_best.agent_id}!")
            
            # Handle elimination for survival mode
            if mode == CompetitionMode.SURVIVAL and round_num < rounds:
                eliminated = ranked[-1]
                self.agents = [a for a in self.agents if a.agent_id != eliminated.agent_id]
                print(f"\n[Competition] {eliminated.agent_id} eliminated!")
        
        # Final rankings
        print(f"\n{'=' * 80}")
        print("FINAL RESULTS")
        print(f"{'=' * 80}\n")
        
        final_ranked = self.judge.rank_solutions(all_solutions)
        winner = final_ranked[0]
        
        # Update agent stats
        for agent in self.agents:
            if agent.agent_id == winner.agent_id:
                agent.wins += 1
        
        return CompetitionResult(
            winner=winner,
            all_solutions=all_solutions,
            rounds=rounds,
            mode=mode.value
        )
    
    def get_leaderboard(self) -> List[Tuple[str, int, float]]:
        """
        Get agent leaderboard.
        
        Returns:
            List of (agent_id, wins, avg_score) tuples
        """
        return sorted(
            [(a.agent_id, a.wins, a.total_score) for a in self.agents],
            key=lambda x: (x[1], x[2]),
            reverse=True
        )


def demonstrate_competitive_multi_agent():
    """Demonstrate the Competitive Multi-Agent pattern."""
    
    print("=" * 80)
    print("COMPETITIVE MULTI-AGENT PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Test 1: Creative competition
    print("\n" + "=" * 80)
    print("TEST 1: Creative Problem-Solving Competition")
    print("=" * 80)
    
    system1 = CompetitiveSystem()
    
    # Add competitors with different strategies
    system1.add_competitor(
        "Innovator",
        "Focus on novel, cutting-edge, unconventional ideas",
        temperature=0.9
    )
    system1.add_competitor(
        "Pragmatist",
        "Focus on practical, proven, implementable solutions",
        temperature=0.5
    )
    system1.add_competitor(
        "Optimizer",
        "Focus on efficiency, cost-effectiveness, and ROI",
        temperature=0.6
    )
    system1.add_competitor(
        "User-Centric",
        "Focus on user experience, accessibility, and satisfaction",
        temperature=0.7
    )
    
    problem1 = "Design a feature to increase user engagement in a productivity app"
    
    result1 = system1.run_competition(
        problem1,
        mode=CompetitionMode.ROUND_ROBIN,
        rounds=2
    )
    
    print(f"\n{'*' * 80}")
    print("WINNER:")
    print(f"{'*' * 80}")
    print(f"Agent: {result1.winner.agent_id}")
    print(f"Score: {result1.winner.total_score:.2f}/10.0")
    print(f"Round: {result1.winner.round_number}")
    print(f"\nWinning Solution:\n{result1.winner.content}")
    
    print(f"\n{'-' * 80}")
    print("SCORE BREAKDOWN:")
    print(f"{'-' * 80}")
    for criterion, score in result1.winner.scores.items():
        print(f"  {criterion}: {score:.2f}/10.0")
    
    # Test 2: Survival mode competition
    print("\n" + "=" * 80)
    print("TEST 2: Survival Mode Competition")
    print("=" * 80)
    
    system2 = CompetitiveSystem(
        evaluation_criteria=[
            "Innovation",
            "Feasibility",
            "Impact"
        ]
    )
    
    system2.add_competitor("Alpha", "Aggressive innovation", 0.9)
    system2.add_competitor("Beta", "Balanced approach", 0.7)
    system2.add_competitor("Gamma", "Conservative and safe", 0.5)
    
    problem2 = "Propose a strategy to reduce customer churn by 30%"
    
    result2 = system2.run_competition(
        problem2,
        mode=CompetitionMode.SURVIVAL,
        rounds=2
    )
    
    print(f"\n{'*' * 80}")
    print("SURVIVOR & WINNER:")
    print(f"{'*' * 80}")
    print(f"Agent: {result2.winner.agent_id}")
    print(f"Score: {result2.winner.total_score:.2f}/10.0")
    print(f"\nWinning Solution:\n{result2.winner.content}")
    
    # Test 3: Tournament with diverse strategies
    print("\n" + "=" * 80)
    print("TEST 3: Multi-Round Tournament")
    print("=" * 80)
    
    system3 = CompetitiveSystem()
    
    system3.add_competitor("Creative", "Maximum creativity and originality", 1.0)
    system3.add_competitor("Analytical", "Data-driven and analytical", 0.4)
    system3.add_competitor("Hybrid", "Balance creativity with analysis", 0.7)
    
    problem3 = "Develop a marketing campaign for an eco-friendly product"
    
    result3 = system3.run_competition(
        problem3,
        mode=CompetitionMode.ROUND_ROBIN,
        rounds=3
    )
    
    print(f"\n{'*' * 80}")
    print("TOURNAMENT WINNER:")
    print(f"{'*' * 80}")
    print(f"Agent: {result3.winner.agent_id}")
    print(f"Average Score: {result3.winner.total_score:.2f}/10.0")
    
    # Show score progression
    print(f"\n{'-' * 80}")
    print("SCORE PROGRESSION:")
    print(f"{'-' * 80}")
    
    for agent_id in ["Creative", "Analytical", "Hybrid"]:
        agent_solutions = [s for s in result3.all_solutions if s.agent_id == agent_id]
        scores = [s.total_score for s in agent_solutions]
        print(f"{agent_id}: {' -> '.join([f'{s:.2f}' for s in scores])}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Competitive Multi-Agent pattern demonstrates several key benefits:

1. **Quality Through Competition**: Competition drives agents to produce better solutions
2. **Diverse Approaches**: Different strategies explore solution space differently
3. **Iterative Improvement**: Agents can improve based on previous best solutions
4. **Objective Selection**: Multi-criteria evaluation ensures fair judging

Competition Modes:
- **Round Robin**: All agents compete in every round (fair, comprehensive)
- **Tournament**: Bracket-style elimination (exciting, efficient)
- **Survival**: Lowest scorers eliminated each round (competitive pressure)

Evaluation Approach:
- **Multi-Criteria**: Solutions judged on multiple dimensions
- **Objective Scoring**: Numerical scores (0-10) for each criterion
- **Weighted Total**: Final score from criterion average
- **Ranking**: Clear winner determination

Use Cases:
- Creative tasks (writing, design, ideation)
- Optimization with multiple local optima
- Generating diverse solution approaches
- Quality improvement through competition
- A/B testing alternative approaches

The pattern is particularly effective when:
- Solution quality is critical
- Multiple valid approaches exist
- Diversity of perspectives adds value
- Competition motivates better outputs
- Objective evaluation is possible
""")


if __name__ == "__main__":
    demonstrate_competitive_multi_agent()
