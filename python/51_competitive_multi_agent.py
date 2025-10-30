"""
Competitive Multi-Agent Pattern
Agents compete to produce best solution
"""
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import random
import time
class CompetitionType(Enum):
    QUALITY = "quality"
    SPEED = "speed"
    EFFICIENCY = "efficiency"
    CREATIVITY = "creativity"
@dataclass
class Submission:
    """Agent's submission"""
    agent_id: str
    solution: Any
    score: float
    time_taken_ms: float
    metadata: Dict[str, Any]
@dataclass
class CompetitionResult:
    """Result of competition"""
    winner: Submission
    all_submissions: List[Submission]
    competition_type: CompetitionType
    evaluation_criteria: Dict[str, float]
class CompetitiveAgent:
    """Agent that competes with others"""
    def __init__(self, agent_id: str, name: str, strategy: str):
        self.agent_id = agent_id
        self.name = name
        self.strategy = strategy
        self.wins = 0
        self.total_competitions = 0
    def compete(self, task: str, competition_type: CompetitionType) -> Submission:
        """Generate solution for competition"""
        print(f"\n[{self.name}] Competing with {self.strategy} strategy...")
        start_time = time.time()
        # Generate solution based on strategy
        if self.strategy == "fast":
            solution = self._fast_solution(task)
        elif self.strategy == "thorough":
            solution = self._thorough_solution(task)
        elif self.strategy == "creative":
            solution = self._creative_solution(task)
        else:
            solution = self._balanced_solution(task)
        time_taken = (time.time() - start_time) * 1000
        # Calculate score
        score = self._evaluate_solution(solution, competition_type)
        print(f"  Solution: {solution}")
        print(f"  Score: {score:.2f}")
        print(f"  Time: {time_taken:.0f}ms")
        return Submission(
            agent_id=self.agent_id,
            solution=solution,
            score=score,
            time_taken_ms=time_taken,
            metadata={"strategy": self.strategy, "agent_name": self.name}
        )
    def _fast_solution(self, task: str) -> str:
        """Quick but basic solution"""
        time.sleep(0.05)
        return f"Fast solution for: {task}"
    def _thorough_solution(self, task: str) -> str:
        """Slow but high-quality solution"""
        time.sleep(0.3)
        return f"Comprehensive and detailed solution for: {task}"
    def _creative_solution(self, task: str) -> str:
        """Creative approach"""
        time.sleep(0.15)
        return f"Innovative and creative solution for: {task}"
    def _balanced_solution(self, task: str) -> str:
        """Balanced approach"""
        time.sleep(0.1)
        return f"Well-balanced solution for: {task}"
    def _evaluate_solution(self, solution: str, competition_type: CompetitionType) -> float:
        """Evaluate solution quality"""
        base_score = len(solution) / 100  # Simple metric
        # Adjust based on strategy and competition type
        if competition_type == CompetitionType.QUALITY:
            if self.strategy == "thorough":
                return base_score * random.uniform(0.9, 1.0)
            elif self.strategy == "fast":
                return base_score * random.uniform(0.6, 0.8)
        elif competition_type == CompetitionType.SPEED:
            if self.strategy == "fast":
                return base_score * random.uniform(0.9, 1.0)
            elif self.strategy == "thorough":
                return base_score * random.uniform(0.5, 0.7)
        elif competition_type == CompetitionType.CREATIVITY:
            if self.strategy == "creative":
                return base_score * random.uniform(0.9, 1.0)
            else:
                return base_score * random.uniform(0.6, 0.8)
        return base_score * random.uniform(0.7, 0.9)
class CompetitionArena:
    """Arena where agents compete"""
    def __init__(self, arena_name: str):
        self.arena_name = arena_name
        self.agents: List[CompetitiveAgent] = []
        self.competition_history: List[CompetitionResult] = []
    def register_agent(self, agent: CompetitiveAgent):
        """Register agent for competition"""
        self.agents.append(agent)
        print(f"Registered: {agent.name} ({agent.strategy} strategy)")
    def run_competition(self, task: str, competition_type: CompetitionType) -> CompetitionResult:
        """Run competition among all agents"""
        print(f"\n{'='*70}")
        print(f"COMPETITION: {self.arena_name}")
        print(f"{'='*70}")
        print(f"Task: {task}")
        print(f"Type: {competition_type.value}")
        print(f"Competitors: {len(self.agents)}")
        # Get submissions from all agents
        submissions = []
        for agent in self.agents:
            submission = agent.compete(task, competition_type)
            submissions.append(submission)
            agent.total_competitions += 1
        # Evaluate and rank
        print(f"\n{'='*70}")
        print("EVALUATION")
        print(f"{'='*70}\n")
        ranked = self._rank_submissions(submissions, competition_type)
        # Show rankings
        for i, submission in enumerate(ranked, 1):
            agent_name = submission.metadata['agent_name']
            print(f"{i}. {agent_name}: {submission.score:.2f} points ({submission.time_taken_ms:.0f}ms)")
        # Determine winner
        winner = ranked[0]
        winner_agent = next(a for a in self.agents if a.agent_id == winner.agent_id)
        winner_agent.wins += 1
        print(f"\n🏆 Winner: {winner.metadata['agent_name']}")
        # Create result
        result = CompetitionResult(
            winner=winner,
            all_submissions=submissions,
            competition_type=competition_type,
            evaluation_criteria={
                "quality_weight": 0.6 if competition_type == CompetitionType.QUALITY else 0.3,
                "speed_weight": 0.6 if competition_type == CompetitionType.SPEED else 0.2,
                "creativity_weight": 0.6 if competition_type == CompetitionType.CREATIVITY else 0.2
            }
        )
        self.competition_history.append(result)
        return result
    def _rank_submissions(self, submissions: List[Submission], 
                         competition_type: CompetitionType) -> List[Submission]:
        """Rank submissions by score"""
        # Multi-criteria ranking
        ranked = sorted(submissions, key=lambda s: (
            s.score * 0.7 +  # Quality component
            (1000 / max(s.time_taken_ms, 1)) * 0.3  # Speed component
        ), reverse=True)
        return ranked
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get overall leaderboard"""
        leaderboard = []
        for agent in self.agents:
            win_rate = agent.wins / agent.total_competitions if agent.total_competitions > 0 else 0
            leaderboard.append({
                'name': agent.name,
                'strategy': agent.strategy,
                'wins': agent.wins,
                'total_competitions': agent.total_competitions,
                'win_rate': win_rate
            })
        # Sort by win rate
        leaderboard.sort(key=lambda x: x['win_rate'], reverse=True)
        return leaderboard
    def print_leaderboard(self):
        """Print leaderboard"""
        leaderboard = self.get_leaderboard()
        print(f"\n{'='*70}")
        print("OVERALL LEADERBOARD")
        print(f"{'='*70}\n")
        print(f"{'Rank':<6} {'Agent':<25} {'Strategy':<12} {'Wins':<6} {'Total':<6} {'Win Rate':<10}")
        print("-" * 70)
        for i, entry in enumerate(leaderboard, 1):
            print(f"{i:<6} {entry['name']:<25} {entry['strategy']:<12} "
                  f"{entry['wins']:<6} {entry['total_competitions']:<6} {entry['win_rate']:.1%}")
# Usage
if __name__ == "__main__":
    print("="*80)
    print("COMPETITIVE MULTI-AGENT PATTERN DEMONSTRATION")
    print("="*80)
    # Create arena
    arena = CompetitionArena("AI Solutions Arena")
    # Create competing agents
    agents = [
        CompetitiveAgent("agent_1", "SpeedDemon", "fast"),
        CompetitiveAgent("agent_2", "DeepThinker", "thorough"),
        CompetitiveAgent("agent_3", "Innovator", "creative"),
        CompetitiveAgent("agent_4", "AllRounder", "balanced"),
    ]
    print("\nRegistering agents...")
    for agent in agents:
        arena.register_agent(agent)
    # Run multiple competitions
    competitions = [
        ("Optimize database query performance", CompetitionType.QUALITY),
        ("Generate creative marketing slogan", CompetitionType.CREATIVITY),
        ("Process data pipeline quickly", CompetitionType.SPEED),
        ("Design scalable architecture", CompetitionType.QUALITY),
        ("Create engaging user interface", CompetitionType.CREATIVITY),
    ]
    for task, comp_type in competitions:
        arena.run_competition(task, comp_type)
        print("\n" + "="*80 + "\n")
        time.sleep(0.2)
    # Show final leaderboard
    arena.print_leaderboard()
