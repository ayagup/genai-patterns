"""
Multi-Agent Negotiation Pattern

Implements negotiation protocols between multiple agents.
Supports various negotiation strategies and conflict resolution.

Use Cases:
- Resource allocation
- Task assignment
- Conflict resolution
- Distributed decision making

Advantages:
- Decentralized decision making
- Fair resource distribution
- Conflict resolution
- Collaborative problem solving
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import random


class NegotiationProtocol(Enum):
    """Negotiation protocols"""
    ALTERNATING_OFFERS = "alternating_offers"
    AUCTION = "auction"
    CONTRACT_NET = "contract_net"
    MONOTONIC_CONCESSION = "monotonic_concession"


class NegotiationStrategy(Enum):
    """Agent negotiation strategies"""
    CONCEDER = "conceder"  # Quickly concedes
    BOULWARE = "boulware"  # Holds firm until deadline
    LINEAR = "linear"  # Linear concession
    TIT_FOR_TAT = "tit_for_tat"  # Mirrors opponent
    HARDBALL = "hardball"  # Aggressive


class ProposalStatus(Enum):
    """Status of proposal"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTER_OFFERED = "counter_offered"
    WITHDRAWN = "withdrawn"


@dataclass
class NegotiationIssue:
    """Issue being negotiated"""
    issue_id: str
    name: str
    min_value: float
    max_value: float
    preference_direction: str  # "higher" or "lower"
    weight: float = 1.0  # Importance weight


@dataclass
class Proposal:
    """Negotiation proposal"""
    proposal_id: str
    from_agent: str
    to_agent: str
    issues: Dict[str, float]  # issue_id -> value
    utility: float
    timestamp: datetime
    status: ProposalStatus = ProposalStatus.PENDING
    response_deadline: Optional[datetime] = None


@dataclass
class NegotiationSession:
    """Negotiation session between agents"""
    session_id: str
    participants: List[str]
    protocol: NegotiationProtocol
    issues: List[NegotiationIssue]
    start_time: datetime
    end_time: Optional[datetime] = None
    proposals: List[Proposal] = field(default_factory=list)
    agreement: Optional[Proposal] = None
    status: str = "active"


@dataclass
class AgentPreferences:
    """Agent's preferences for negotiation"""
    agent_id: str
    reservation_utility: float  # Minimum acceptable utility
    target_utility: float  # Desired utility
    issue_weights: Dict[str, float]  # Importance of each issue
    strategy: NegotiationStrategy


class UtilityCalculator:
    """Calculates utility of proposals"""
    
    def calculate_utility(self,
                         proposal_values: Dict[str, float],
                         issues: List[NegotiationIssue],
                         preferences: AgentPreferences) -> float:
        """
        Calculate utility of a proposal for an agent.
        
        Args:
            proposal_values: Proposed values for each issue
            issues: List of negotiation issues
            preferences: Agent preferences
            
        Returns:
            Utility value (0-1)
        """
        total_utility = 0.0
        total_weight = 0.0
        
        for issue in issues:
            if issue.issue_id not in proposal_values:
                continue
            
            value = proposal_values[issue.issue_id]
            weight = preferences.issue_weights.get(issue.issue_id, 1.0)
            
            # Normalize value to 0-1 range
            if issue.preference_direction == "higher":
                normalized = (value - issue.min_value) / (
                    issue.max_value - issue.min_value
                )
            else:
                normalized = (issue.max_value - value) / (
                    issue.max_value - issue.min_value
                )
            
            total_utility += normalized * weight
            total_weight += weight
        
        return total_utility / total_weight if total_weight > 0 else 0.0
    
    def is_acceptable(self,
                     utility: float,
                     preferences: AgentPreferences) -> bool:
        """Check if utility meets agent's reservation utility"""
        return utility >= preferences.reservation_utility


class ProposalGenerator:
    """Generates negotiation proposals"""
    
    def __init__(self, utility_calculator: UtilityCalculator):
        self.utility_calculator = utility_calculator
    
    def generate_initial_proposal(self,
                                  agent_id: str,
                                  issues: List[NegotiationIssue],
                                  preferences: AgentPreferences) -> Dict[str, float]:
        """
        Generate initial proposal.
        
        Args:
            agent_id: Agent making proposal
            issues: Negotiation issues
            preferences: Agent preferences
            
        Returns:
            Proposal values
        """
        proposal = {}
        
        for issue in issues:
            weight = preferences.issue_weights.get(issue.issue_id, 1.0)
            
            # Start with values favoring this agent
            if issue.preference_direction == "higher":
                # High weight = ask for high value
                ratio = 0.5 + (weight * 0.4)
                value = issue.min_value + ratio * (
                    issue.max_value - issue.min_value
                )
            else:
                # High weight = ask for low value
                ratio = 0.5 - (weight * 0.4)
                value = issue.min_value + ratio * (
                    issue.max_value - issue.min_value
                )
            
            proposal[issue.issue_id] = value
        
        return proposal
    
    def generate_counter_proposal(self,
                                  agent_id: str,
                                  previous_proposal: Proposal,
                                  issues: List[NegotiationIssue],
                                  preferences: AgentPreferences,
                                  concession_rate: float) -> Dict[str, float]:
        """
        Generate counter-proposal.
        
        Args:
            agent_id: Agent making counter-proposal
            previous_proposal: Previous proposal
            issues: Negotiation issues
            preferences: Agent preferences
            concession_rate: How much to concede (0-1)
            
        Returns:
            Counter-proposal values
        """
        counter = {}
        
        for issue in issues:
            prev_value = previous_proposal.issues.get(issue.issue_id)
            if prev_value is None:
                continue
            
            # Generate ideal value for this agent
            weight = preferences.issue_weights.get(issue.issue_id, 1.0)
            
            if issue.preference_direction == "higher":
                ideal_value = issue.max_value
            else:
                ideal_value = issue.min_value
            
            # Move toward opponent's position based on concession rate
            new_value = prev_value + (ideal_value - prev_value) * (
                1 - concession_rate
            )
            
            # Ensure within bounds
            new_value = max(issue.min_value, min(issue.max_value, new_value))
            
            counter[issue.issue_id] = new_value
        
        return counter


class NegotiationAgent:
    """Agent capable of negotiation"""
    
    def __init__(self,
                 agent_id: str,
                 preferences: AgentPreferences,
                 utility_calculator: UtilityCalculator,
                 proposal_generator: ProposalGenerator):
        self.agent_id = agent_id
        self.preferences = preferences
        self.utility_calculator = utility_calculator
        self.proposal_generator = proposal_generator
        
        self.negotiation_history: List[Proposal] = []
        self.current_sessions: Dict[str, NegotiationSession] = {}
    
    def make_initial_proposal(self,
                             session: NegotiationSession,
                             to_agent: str) -> Proposal:
        """
        Make initial proposal in negotiation.
        
        Args:
            session: Negotiation session
            to_agent: Agent to propose to
            
        Returns:
            Initial proposal
        """
        proposal_values = self.proposal_generator.generate_initial_proposal(
            self.agent_id,
            session.issues,
            self.preferences
        )
        
        utility = self.utility_calculator.calculate_utility(
            proposal_values,
            session.issues,
            self.preferences
        )
        
        proposal = Proposal(
            proposal_id="prop_{}".format(len(self.negotiation_history)),
            from_agent=self.agent_id,
            to_agent=to_agent,
            issues=proposal_values,
            utility=utility,
            timestamp=datetime.now()
        )
        
        self.negotiation_history.append(proposal)
        
        return proposal
    
    def respond_to_proposal(self,
                           proposal: Proposal,
                           session: NegotiationSession,
                           round_number: int) -> Tuple[ProposalStatus, Optional[Proposal]]:
        """
        Respond to a proposal.
        
        Args:
            proposal: Received proposal
            session: Negotiation session
            round_number: Current round
            
        Returns:
            (status, counter_proposal) tuple
        """
        # Calculate utility of proposal
        utility = self.utility_calculator.calculate_utility(
            proposal.issues,
            session.issues,
            self.preferences
        )
        
        # Check if acceptable
        if self.utility_calculator.is_acceptable(utility, self.preferences):
            return ProposalStatus.ACCEPTED, None
        
        # Determine concession rate based on strategy
        concession_rate = self._calculate_concession_rate(
            round_number,
            session
        )
        
        # Generate counter-proposal
        counter_values = self.proposal_generator.generate_counter_proposal(
            self.agent_id,
            proposal,
            session.issues,
            self.preferences,
            concession_rate
        )
        
        counter_utility = self.utility_calculator.calculate_utility(
            counter_values,
            session.issues,
            self.preferences
        )
        
        counter_proposal = Proposal(
            proposal_id="prop_{}".format(len(self.negotiation_history)),
            from_agent=self.agent_id,
            to_agent=proposal.from_agent,
            issues=counter_values,
            utility=counter_utility,
            timestamp=datetime.now()
        )
        
        self.negotiation_history.append(counter_proposal)
        
        return ProposalStatus.COUNTER_OFFERED, counter_proposal
    
    def _calculate_concession_rate(self,
                                   round_number: int,
                                   session: NegotiationSession) -> float:
        """Calculate concession rate based on strategy"""
        max_rounds = 20  # Assumed maximum rounds
        progress = round_number / max_rounds
        
        strategy = self.preferences.strategy
        
        if strategy == NegotiationStrategy.CONCEDER:
            # Concede quickly early on
            return 0.2 + (progress * 0.6)
        
        elif strategy == NegotiationStrategy.BOULWARE:
            # Hold firm until late
            if progress < 0.7:
                return 0.1
            else:
                return 0.1 + ((progress - 0.7) / 0.3) * 0.7
        
        elif strategy == NegotiationStrategy.LINEAR:
            # Linear concession
            return 0.2 + (progress * 0.5)
        
        elif strategy == NegotiationStrategy.HARDBALL:
            # Minimal concession
            return 0.05 + (progress * 0.15)
        
        else:  # TIT_FOR_TAT
            # Match opponent's last concession
            if len(session.proposals) >= 2:
                # Calculate opponent's last concession
                return 0.3  # Simplified
            return 0.3


class MultiAgentNegotiationSystem:
    """
    System managing multi-agent negotiations.
    Coordinates negotiation sessions and enforces protocols.
    """
    
    def __init__(self):
        self.utility_calculator = UtilityCalculator()
        self.proposal_generator = ProposalGenerator(self.utility_calculator)
        
        self.agents: Dict[str, NegotiationAgent] = {}
        self.sessions: Dict[str, NegotiationSession] = {}
        self.session_counter = 0
    
    def register_agent(self,
                      agent_id: str,
                      preferences: AgentPreferences) -> None:
        """
        Register an agent for negotiation.
        
        Args:
            agent_id: Agent identifier
            preferences: Agent preferences
        """
        agent = NegotiationAgent(
            agent_id,
            preferences,
            self.utility_calculator,
            self.proposal_generator
        )
        
        self.agents[agent_id] = agent
    
    def create_negotiation_session(self,
                                   participant_ids: List[str],
                                   issues: List[NegotiationIssue],
                                   protocol: NegotiationProtocol = NegotiationProtocol.ALTERNATING_OFFERS
                                   ) -> str:
        """
        Create negotiation session.
        
        Args:
            participant_ids: List of participating agent IDs
            issues: Issues to negotiate
            protocol: Negotiation protocol
            
        Returns:
            Session ID
        """
        session = NegotiationSession(
            session_id="session_{}".format(self.session_counter),
            participants=participant_ids,
            protocol=protocol,
            issues=issues,
            start_time=datetime.now()
        )
        
        self.sessions[session.session_id] = session
        self.session_counter += 1
        
        # Register session with agents
        for agent_id in participant_ids:
            if agent_id in self.agents:
                self.agents[agent_id].current_sessions[session.session_id] = session
        
        return session.session_id
    
    def run_negotiation(self,
                       session_id: str,
                       max_rounds: int = 20) -> Optional[Proposal]:
        """
        Run negotiation session.
        
        Args:
            session_id: Session to run
            max_rounds: Maximum negotiation rounds
            
        Returns:
            Agreement proposal or None
        """
        session = self.sessions.get(session_id)
        if not session or len(session.participants) < 2:
            return None
        
        if session.protocol == NegotiationProtocol.ALTERNATING_OFFERS:
            return self._run_alternating_offers(session, max_rounds)
        elif session.protocol == NegotiationProtocol.AUCTION:
            return self._run_auction(session, max_rounds)
        else:
            return self._run_alternating_offers(session, max_rounds)
    
    def _run_alternating_offers(self,
                               session: NegotiationSession,
                               max_rounds: int) -> Optional[Proposal]:
        """Run alternating offers protocol"""
        agent1_id = session.participants[0]
        agent2_id = session.participants[1]
        
        agent1 = self.agents.get(agent1_id)
        agent2 = self.agents.get(agent2_id)
        
        if not agent1 or not agent2:
            return None
        
        # Agent 1 makes initial proposal
        current_proposal = agent1.make_initial_proposal(session, agent2_id)
        session.proposals.append(current_proposal)
        
        for round_num in range(max_rounds):
            # Agent 2 responds
            status, counter = agent2.respond_to_proposal(
                current_proposal,
                session,
                round_num
            )
            
            current_proposal.status = status
            
            if status == ProposalStatus.ACCEPTED:
                session.agreement = current_proposal
                session.status = "completed"
                session.end_time = datetime.now()
                return current_proposal
            
            if status == ProposalStatus.COUNTER_OFFERED and counter:
                session.proposals.append(counter)
                
                # Agent 1 responds to counter
                status, counter2 = agent1.respond_to_proposal(
                    counter,
                    session,
                    round_num
                )
                
                counter.status = status
                
                if status == ProposalStatus.ACCEPTED:
                    session.agreement = counter
                    session.status = "completed"
                    session.end_time = datetime.now()
                    return counter
                
                if status == ProposalStatus.COUNTER_OFFERED and counter2:
                    session.proposals.append(counter2)
                    current_proposal = counter2
                else:
                    break
            else:
                break
        
        # No agreement reached
        session.status = "failed"
        session.end_time = datetime.now()
        return None
    
    def _run_auction(self,
                    session: NegotiationSession,
                    max_rounds: int) -> Optional[Proposal]:
        """Run auction protocol"""
        # Simplified auction: highest utility wins
        best_proposal = None
        best_utility = -1
        
        for agent_id in session.participants:
            agent = self.agents.get(agent_id)
            if not agent:
                continue
            
            # Each agent submits a bid
            proposal = agent.make_initial_proposal(
                session,
                "auctioneer"
            )
            
            session.proposals.append(proposal)
            
            if proposal.utility > best_utility:
                best_utility = proposal.utility
                best_proposal = proposal
        
        if best_proposal:
            best_proposal.status = ProposalStatus.ACCEPTED
            session.agreement = best_proposal
            session.status = "completed"
            session.end_time = datetime.now()
        else:
            session.status = "failed"
            session.end_time = datetime.now()
        
        return best_proposal
    
    def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """Get results of negotiation session"""
        session = self.sessions.get(session_id)
        if not session:
            return {}
        
        duration = None
        if session.end_time:
            duration = (session.end_time - session.start_time).total_seconds()
        
        results = {
            "session_id": session_id,
            "protocol": session.protocol.value,
            "participants": session.participants,
            "status": session.status,
            "total_proposals": len(session.proposals),
            "duration_seconds": duration,
            "agreement_reached": session.agreement is not None
        }
        
        if session.agreement:
            results["agreement"] = {
                "from_agent": session.agreement.from_agent,
                "to_agent": session.agreement.to_agent,
                "utility": session.agreement.utility,
                "issues": session.agreement.issues
            }
        
        return results
    
    def get_agent_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics for an agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            return {}
        
        total_negotiations = len(agent.current_sessions)
        
        accepted_proposals = sum(
            1 for p in agent.negotiation_history
            if p.status == ProposalStatus.ACCEPTED
        )
        
        return {
            "agent_id": agent_id,
            "strategy": agent.preferences.strategy.value,
            "total_negotiations": total_negotiations,
            "total_proposals": len(agent.negotiation_history),
            "accepted_proposals": accepted_proposals,
            "reservation_utility": agent.preferences.reservation_utility,
            "target_utility": agent.preferences.target_utility
        }


def demonstrate_multi_agent_negotiation():
    """Demonstrate multi-agent negotiation"""
    print("=" * 70)
    print("Multi-Agent Negotiation System Demonstration")
    print("=" * 70)
    
    system = MultiAgentNegotiationSystem()
    
    # Example 1: Define negotiation issues
    print("\n1. Defining Negotiation Issues:")
    
    issues = [
        NegotiationIssue(
            issue_id="price",
            name="Price",
            min_value=100,
            max_value=200,
            preference_direction="lower",  # Buyer wants lower
            weight=1.0
        ),
        NegotiationIssue(
            issue_id="delivery_time",
            name="Delivery Time (days)",
            min_value=1,
            max_value=30,
            preference_direction="lower",  # Both want faster
            weight=0.7
        ),
        NegotiationIssue(
            issue_id="warranty",
            name="Warranty (months)",
            min_value=6,
            max_value=36,
            preference_direction="higher",  # Buyer wants more
            weight=0.5
        )
    ]
    
    for issue in issues:
        print("  {}: {} - {}".format(
            issue.name,
            issue.min_value,
            issue.max_value
        ))
    
    # Example 2: Register agents
    print("\n2. Registering Agents:")
    
    # Buyer agent (wants low price, high warranty)
    buyer_prefs = AgentPreferences(
        agent_id="buyer",
        reservation_utility=0.5,
        target_utility=0.8,
        issue_weights={
            "price": 1.0,
            "delivery_time": 0.6,
            "warranty": 0.8
        },
        strategy=NegotiationStrategy.LINEAR
    )
    
    system.register_agent("buyer", buyer_prefs)
    print("  Registered: Buyer (Linear strategy)")
    
    # Seller agent (wants high price, low warranty)
    seller_prefs = AgentPreferences(
        agent_id="seller",
        reservation_utility=0.4,
        target_utility=0.7,
        issue_weights={
            "price": 1.0,
            "delivery_time": 0.7,
            "warranty": 0.3
        },
        strategy=NegotiationStrategy.BOULWARE
    )
    
    system.register_agent("seller", seller_prefs)
    print("  Registered: Seller (Boulware strategy)")
    
    # Example 3: Create negotiation session
    print("\n3. Creating Negotiation Session:")
    
    session_id = system.create_negotiation_session(
        participant_ids=["buyer", "seller"],
        issues=issues,
        protocol=NegotiationProtocol.ALTERNATING_OFFERS
    )
    
    print("  Session ID: {}".format(session_id))
    print("  Protocol: Alternating Offers")
    
    # Example 4: Run negotiation
    print("\n4. Running Negotiation:")
    
    agreement = system.run_negotiation(session_id, max_rounds=10)
    
    if agreement:
        print("\n  Agreement Reached!")
        print("  Final offer from: {}".format(agreement.from_agent))
        print("  Agreement details:")
        for issue_id, value in agreement.issues.items():
            issue = next(i for i in issues if i.issue_id == issue_id)
            print("    {}: {:.2f}".format(issue.name, value))
        print("  Proposer utility: {:.2%}".format(agreement.utility))
    else:
        print("\n  No agreement reached")
    
    # Example 5: Session results
    print("\n5. Negotiation Results:")
    results = system.get_session_results(session_id)
    print(json.dumps(results, indent=2, default=str))
    
    # Example 6: Agent statistics
    print("\n6. Agent Statistics:")
    
    for agent_id in ["buyer", "seller"]:
        stats = system.get_agent_statistics(agent_id)
        print("\n  {}:".format(agent_id.upper()))
        print("    Strategy: {}".format(stats["strategy"]))
        print("    Total proposals: {}".format(stats["total_proposals"]))
        print("    Reservation utility: {:.2%}".format(
            stats["reservation_utility"]
        ))
    
    # Example 7: Multiple negotiations with different strategies
    print("\n7. Testing Different Strategy Combinations:")
    
    strategies = [
        (NegotiationStrategy.CONCEDER, NegotiationStrategy.HARDBALL),
        (NegotiationStrategy.LINEAR, NegotiationStrategy.LINEAR),
        (NegotiationStrategy.BOULWARE, NegotiationStrategy.CONCEDER)
    ]
    
    for buyer_strat, seller_strat in strategies:
        # Create new agents
        buyer_prefs_test = AgentPreferences(
            agent_id="buyer_test",
            reservation_utility=0.5,
            target_utility=0.8,
            issue_weights={"price": 1.0, "delivery_time": 0.6, "warranty": 0.8},
            strategy=buyer_strat
        )
        
        seller_prefs_test = AgentPreferences(
            agent_id="seller_test",
            reservation_utility=0.4,
            target_utility=0.7,
            issue_weights={"price": 1.0, "delivery_time": 0.7, "warranty": 0.3},
            strategy=seller_strat
        )
        
        system.register_agent("buyer_test", buyer_prefs_test)
        system.register_agent("seller_test", seller_prefs_test)
        
        # Run negotiation
        test_session = system.create_negotiation_session(
            ["buyer_test", "seller_test"],
            issues,
            NegotiationProtocol.ALTERNATING_OFFERS
        )
        
        test_agreement = system.run_negotiation(test_session, max_rounds=15)
        
        print("\n  {} vs {}:".format(
            buyer_strat.value,
            seller_strat.value
        ))
        
        if test_agreement:
            print("    Agreement reached")
            print("    Price: {:.2f}".format(
                test_agreement.issues.get("price", 0)
            ))
        else:
            print("    No agreement")


if __name__ == "__main__":
    demonstrate_multi_agent_negotiation()
