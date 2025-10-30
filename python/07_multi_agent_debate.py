"""
Multi-Agent Debate Pattern
Multiple agents debate to reach better conclusions
"""
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum
class AgentRole(Enum):
    PROPOSER = "proposer"
    CRITIC = "critic"
    MODERATOR = "moderator"
@dataclass
class Argument:
    agent_id: str
    role: AgentRole
    content: str
    round_num: int
class DebateAgent:
    def __init__(self, agent_id: str, role: AgentRole, perspective: str):
        self.agent_id = agent_id
        self.role = role
        self.perspective = perspective
        self.arguments: List[Argument] = []
    def generate_argument(self, topic: str, round_num: int, previous_args: List[Argument]) -> Argument:
        """Generate argument based on role and previous arguments"""
        if self.role == AgentRole.PROPOSER:
            content = self._propose(topic, round_num, previous_args)
        elif self.role == AgentRole.CRITIC:
            content = self._critique(topic, round_num, previous_args)
        else:
            content = self._moderate(topic, round_num, previous_args)
        arg = Argument(
            agent_id=self.agent_id,
            role=self.role,
            content=content,
            round_num=round_num
        )
        self.arguments.append(arg)
        return arg
    def _propose(self, topic: str, round_num: int, previous_args: List[Argument]) -> str:
        """Generate a proposal"""
        if round_num == 1:
            return f"[{self.perspective}] I propose that {topic} because it offers significant benefits."
        else:
            # Respond to criticisms
            critics = [a for a in previous_args if a.role == AgentRole.CRITIC]
            if critics:
                latest_critique = critics[-1]
                return f"[{self.perspective}] Addressing the criticism: while there are concerns, the benefits outweigh them."
        return f"[{self.perspective}] Reiterating the core proposal with refinements."
    def _critique(self, topic: str, round_num: int, previous_args: List[Argument]) -> str:
        """Generate a critique"""
        proposals = [a for a in previous_args if a.role == AgentRole.PROPOSER]
        if proposals:
            latest_proposal = proposals[-1]
            return f"[{self.perspective}] I have concerns about this proposal. We need to consider potential drawbacks and risks."
        return f"[{self.perspective}] Critical analysis suggests we need more evidence."
    def _moderate(self, topic: str, round_num: int, previous_args: List[Argument]) -> str:
        """Moderate the debate"""
        return f"[Moderator] Let's synthesize both viewpoints and find common ground."
class MultiAgentDebate:
    def __init__(self, topic: str, max_rounds: int = 3):
        self.topic = topic
        self.max_rounds = max_rounds
        self.agents: List[DebateAgent] = []
        self.debate_history: List[Argument] = []
    def add_agent(self, agent: DebateAgent):
        """Add an agent to the debate"""
        self.agents.append(agent)
    def conduct_debate(self) -> str:
        """Run the debate for specified rounds"""
        print(f"\n{'='*70}")
        print(f"Multi-Agent Debate: {self.topic}")
        print(f"{'='*70}\n")
        for round_num in range(1, self.max_rounds + 1):
            print(f"\n--- Round {round_num} ---\n")
            # Each agent presents their argument
            for agent in self.agents:
                arg = agent.generate_argument(self.topic, round_num, self.debate_history)
                self.debate_history.append(arg)
                print(f"{agent.role.value.upper()} ({agent.agent_id}):")
                print(f"  {arg.content}\n")
        # Final synthesis
        conclusion = self._synthesize_conclusion()
        return conclusion
    def _synthesize_conclusion(self) -> str:
        """Synthesize final conclusion from debate"""
        print(f"\n{'='*70}")
        print("DEBATE CONCLUSION")
        print(f"{'='*70}\n")
        # Count arguments by role
        proposer_args = [a for a in self.debate_history if a.role == AgentRole.PROPOSER]
        critic_args = [a for a in self.debate_history if a.role == AgentRole.CRITIC]
        conclusion = (
            f"After {self.max_rounds} rounds of debate:\n\n"
            f"PROPOSER VIEW ({len(proposer_args)} arguments):\n"
            f"  - Emphasized benefits and positive aspects\n"
            f"  - Addressed criticisms with counter-arguments\n\n"
            f"CRITIC VIEW ({len(critic_args)} arguments):\n"
            f"  - Highlighted risks and concerns\n"
            f"  - Provided critical analysis\n\n"
            f"BALANCED CONCLUSION:\n"
            f"  The debate revealed both opportunities and challenges regarding '{self.topic}'.\n"
            f"  A nuanced approach considering both perspectives is recommended."
        )
        return conclusion
# Usage
if __name__ == "__main__":
    # Create debate
    debate = MultiAgentDebate(
        topic="adopting AI in healthcare",
        max_rounds=3
    )
    # Add agents with different perspectives
    debate.add_agent(DebateAgent(
        agent_id="agent_optimist",
        role=AgentRole.PROPOSER,
        perspective="Optimistic"
    ))
    debate.add_agent(DebateAgent(
        agent_id="agent_skeptic",
        role=AgentRole.CRITIC,
        perspective="Skeptical"
    ))
    debate.add_agent(DebateAgent(
        agent_id="agent_moderator",
        role=AgentRole.MODERATOR,
        perspective="Neutral"
    ))
    # Conduct debate
    conclusion = debate.conduct_debate()
    print(conclusion)
