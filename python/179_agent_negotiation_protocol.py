"""
Agent Negotiation Protocol Pattern

Implements negotiation protocols for multi-agent coordination.
Enables agents to reach agreements through structured negotiation.

Use Cases:
- Resource allocation
- Conflict resolution
- Task assignment
- Contract negotiation

Advantages:
- Structured negotiation
- Fair outcomes
- Conflict resolution
- Automated agreements
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import random


class NegotiationStrategy(Enum):
    """Negotiation strategies"""
    COOPERATIVE = "cooperative"
    COMPETITIVE = "competitive"
    COMPROMISE = "compromise"
    ACCOMMODATING = "accommodating"


class ProposalStatus(Enum):
    """Status of proposal"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTERED = "countered"
    WITHDRAWN = "withdrawn"


class NegotiationPhase(Enum):
    """Phases of negotiation"""
    OPENING = "opening"
    BARGAINING = "bargaining"
    CLOSING = "closing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class NegotiationItem:
    """Item being negotiated"""
    item_id: str
    name: str
    value_range: Tuple[float, float]
    current_value: Optional[float] = None
    priority: int = 5
    negotiable: bool = True


@dataclass
class Proposal:
    """Negotiation proposal"""
    proposal_id: str
    proposer_id: str
    target_id: str
    items: Dict[str, float]  # item_id -> proposed value
    status: ProposalStatus
    timestamp: datetime
    expires_at: Optional[datetime] = None
    justification: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterProposal:
    """Counter-proposal to a proposal"""
    counter_id: str
    original_proposal_id: str
    proposer_id: str
    items: Dict[str, float]
    timestamp: datetime
    justification: str = ""


@dataclass
class Agreement:
    """Reached agreement"""
    agreement_id: str
    participants: List[str]
    items: Dict[str, float]
    agreed_at: datetime
    valid_until: Optional[datetime] = None
    binding: bool = True
    terms: str = ""


@dataclass
class NegotiationSession:
    """Negotiation session"""
    session_id: str
    participants: List[str]
    items: List[NegotiationItem]
    strategy: NegotiationStrategy
    phase: NegotiationPhase
    proposals: List[Proposal]
    agreement: Optional[Agreement] = None
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    max_rounds: int = 10
    current_round: int = 0


class ProposalGenerator:
    """Generates negotiation proposals"""
    
    def generate_initial_proposal(self,
                                 agent_id: str,
                                 target_id: str,
                                 items: List[NegotiationItem],
                                 strategy: NegotiationStrategy) -> Proposal:
        """
        Generate initial proposal.
        
        Args:
            agent_id: Proposing agent
            target_id: Target agent
            items: Items to negotiate
            strategy: Negotiation strategy
            
        Returns:
            Generated proposal
        """
        proposed_values = {}
        
        for item in items:
            min_val, max_val = item.value_range
            
            if strategy == NegotiationStrategy.COOPERATIVE:
                # Start with fair middle ground
                proposed_values[item.item_id] = (min_val + max_val) / 2
            
            elif strategy == NegotiationStrategy.COMPETITIVE:
                # Start with most favorable value
                proposed_values[item.item_id] = max_val
            
            elif strategy == NegotiationStrategy.COMPROMISE:
                # Start slightly above middle
                proposed_values[item.item_id] = (min_val + max_val) / 2 * 1.1
            
            else:  # ACCOMMODATING
                # Start with less favorable value
                proposed_values[item.item_id] = min_val + (max_val - min_val) * 0.3
        
        return Proposal(
            proposal_id="prop_{}".format(random.randint(1000, 9999)),
            proposer_id=agent_id,
            target_id=target_id,
            items=proposed_values,
            status=ProposalStatus.PENDING,
            timestamp=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24)
        )
    
    def generate_counter_proposal(self,
                                 agent_id: str,
                                 original_proposal: Proposal,
                                 items: List[NegotiationItem],
                                 strategy: NegotiationStrategy) -> CounterProposal:
        """
        Generate counter-proposal.
        
        Args:
            agent_id: Counter-proposing agent
            original_proposal: Original proposal
            items: Negotiation items
            strategy: Strategy to use
            
        Returns:
            Counter-proposal
        """
        counter_values = {}
        
        for item in items:
            original_value = original_proposal.items.get(item.item_id, 0)
            min_val, max_val = item.value_range
            
            if strategy == NegotiationStrategy.COOPERATIVE:
                # Move closer to middle
                middle = (min_val + max_val) / 2
                counter_values[item.item_id] = (original_value + middle) / 2
            
            elif strategy == NegotiationStrategy.COMPETITIVE:
                # Counter with opposite extreme
                counter_values[item.item_id] = min_val
            
            elif strategy == NegotiationStrategy.COMPROMISE:
                # Counter with adjusted value
                adjustment = (max_val - original_value) * 0.3
                counter_values[item.item_id] = original_value - adjustment
            
            else:  # ACCOMMODATING
                # Accept with minor adjustment
                counter_values[item.item_id] = original_value * 0.95
        
        return CounterProposal(
            counter_id="counter_{}".format(random.randint(1000, 9999)),
            original_proposal_id=original_proposal.proposal_id,
            proposer_id=agent_id,
            items=counter_values,
            timestamp=datetime.now()
        )


class ProposalEvaluator:
    """Evaluates proposals"""
    
    def evaluate_proposal(self,
                         proposal: Proposal,
                         agent_preferences: Dict[str, float],
                         items: List[NegotiationItem]) -> float:
        """
        Evaluate proposal value for agent.
        
        Args:
            proposal: Proposal to evaluate
            agent_preferences: Agent's preferred values
            items: Negotiation items
            
        Returns:
            Evaluation score (0-100)
        """
        if not proposal.items:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for item in items:
            if item.item_id not in proposal.items:
                continue
            
            proposed_value = proposal.items[item.item_id]
            preferred_value = agent_preferences.get(item.item_id, 0)
            
            # Calculate how close proposed is to preferred
            min_val, max_val = item.value_range
            value_range = max_val - min_val
            
            if value_range == 0:
                score = 100.0
            else:
                diff = abs(proposed_value - preferred_value)
                score = max(0, 100 - (diff / value_range * 100))
            
            # Weight by priority
            weight = item.priority
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def should_accept(self,
                     proposal: Proposal,
                     agent_preferences: Dict[str, float],
                     items: List[NegotiationItem],
                     threshold: float = 70.0) -> bool:
        """
        Determine if proposal should be accepted.
        
        Args:
            proposal: Proposal to evaluate
            agent_preferences: Agent preferences
            items: Negotiation items
            threshold: Acceptance threshold
            
        Returns:
            Whether to accept
        """
        score = self.evaluate_proposal(proposal, agent_preferences, items)
        return score >= threshold


class AgreementBuilder:
    """Builds agreements from proposals"""
    
    def create_agreement(self,
                        session: NegotiationSession,
                        final_proposal: Proposal) -> Agreement:
        """
        Create agreement from final proposal.
        
        Args:
            session: Negotiation session
            final_proposal: Accepted proposal
            
        Returns:
            Agreement
        """
        return Agreement(
            agreement_id="agr_{}".format(random.randint(1000, 9999)),
            participants=session.participants,
            items=final_proposal.items.copy(),
            agreed_at=datetime.now(),
            valid_until=datetime.now() + timedelta(days=30),
            binding=True,
            terms="Agreement reached through negotiation"
        )
    
    def verify_agreement(self, agreement: Agreement) -> bool:
        """Verify agreement is valid"""
        # Check all participants present
        if not agreement.participants or len(agreement.participants) < 2:
            return False
        
        # Check items present
        if not agreement.items:
            return False
        
        # Check not expired
        if agreement.valid_until and datetime.now() > agreement.valid_until:
            return False
        
        return True


class AgentNegotiationProtocol:
    """
    Implements negotiation protocol for multi-agent systems.
    Manages negotiation sessions and facilitates agreement.
    """
    
    def __init__(self):
        # Components
        self.proposal_generator = ProposalGenerator()
        self.proposal_evaluator = ProposalEvaluator()
        self.agreement_builder = AgreementBuilder()
        
        # State
        self.sessions: Dict[str, NegotiationSession] = {}
        self.agreements: List[Agreement] = []
        self.agent_preferences: Dict[str, Dict[str, float]] = {}
    
    def start_negotiation(self,
                         participants: List[str],
                         items: List[NegotiationItem],
                         strategy: NegotiationStrategy = NegotiationStrategy.COOPERATIVE,
                         max_rounds: int = 10) -> str:
        """
        Start negotiation session.
        
        Args:
            participants: Participating agents
            items: Items to negotiate
            strategy: Negotiation strategy
            max_rounds: Maximum negotiation rounds
            
        Returns:
            Session ID
        """
        session_id = "session_{}".format(random.randint(1000, 9999))
        
        session = NegotiationSession(
            session_id=session_id,
            participants=participants,
            items=items,
            strategy=strategy,
            phase=NegotiationPhase.OPENING,
            proposals=[],
            max_rounds=max_rounds
        )
        
        self.sessions[session_id] = session
        
        return session_id
    
    def set_agent_preferences(self,
                             agent_id: str,
                             preferences: Dict[str, float]) -> None:
        """
        Set agent's preferences for negotiation items.
        
        Args:
            agent_id: Agent identifier
            preferences: Preferred values for items
        """
        self.agent_preferences[agent_id] = preferences
    
    def submit_proposal(self,
                       session_id: str,
                       proposer_id: str,
                       target_id: str) -> Optional[Proposal]:
        """
        Submit proposal in negotiation.
        
        Args:
            session_id: Session identifier
            proposer_id: Proposing agent
            target_id: Target agent
            
        Returns:
            Created proposal
        """
        session = self.sessions.get(session_id)
        
        if not session:
            return None
        
        if session.phase == NegotiationPhase.COMPLETED:
            return None
        
        # Generate proposal
        proposal = self.proposal_generator.generate_initial_proposal(
            proposer_id,
            target_id,
            session.items,
            session.strategy
        )
        
        session.proposals.append(proposal)
        session.phase = NegotiationPhase.BARGAINING
        
        return proposal
    
    def respond_to_proposal(self,
                           session_id: str,
                           proposal_id: str,
                           responder_id: str,
                           accept: bool) -> Optional[Any]:
        """
        Respond to proposal.
        
        Args:
            session_id: Session identifier
            proposal_id: Proposal to respond to
            responder_id: Responding agent
            accept: Whether to accept
            
        Returns:
            Agreement if accepted, CounterProposal if countered, None otherwise
        """
        session = self.sessions.get(session_id)
        
        if not session:
            return None
        
        # Find proposal
        proposal = next(
            (p for p in session.proposals if p.proposal_id == proposal_id),
            None
        )
        
        if not proposal or proposal.status != ProposalStatus.PENDING:
            return None
        
        if accept:
            # Accept proposal
            proposal.status = ProposalStatus.ACCEPTED
            
            # Create agreement
            agreement = self.agreement_builder.create_agreement(session, proposal)
            session.agreement = agreement
            session.phase = NegotiationPhase.COMPLETED
            session.ended_at = datetime.now()
            
            self.agreements.append(agreement)
            
            return agreement
        
        else:
            # Counter proposal
            proposal.status = ProposalStatus.COUNTERED
            
            counter = self.proposal_generator.generate_counter_proposal(
                responder_id,
                proposal,
                session.items,
                session.strategy
            )
            
            session.current_round += 1
            
            # Check if max rounds reached
            if session.current_round >= session.max_rounds:
                session.phase = NegotiationPhase.FAILED
                session.ended_at = datetime.now()
            
            return counter
    
    def auto_negotiate(self,
                      session_id: str) -> Optional[Agreement]:
        """
        Automatically negotiate to reach agreement.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Agreement if reached, None otherwise
        """
        session = self.sessions.get(session_id)
        
        if not session or len(session.participants) < 2:
            return None
        
        agent1, agent2 = session.participants[0], session.participants[1]
        
        # Get preferences
        prefs1 = self.agent_preferences.get(agent1, {})
        prefs2 = self.agent_preferences.get(agent2, {})
        
        # Initial proposal from agent1
        proposal = self.submit_proposal(session_id, agent1, agent2)
        
        if not proposal:
            return None
        
        # Negotiate until agreement or max rounds
        while session.current_round < session.max_rounds:
            # Evaluate proposal for agent2
            score = self.proposal_evaluator.evaluate_proposal(
                proposal,
                prefs2,
                session.items
            )
            
            # Accept if good enough
            if score >= 70.0:
                return self.respond_to_proposal(
                    session_id,
                    proposal.proposal_id,
                    agent2,
                    accept=True
                )
            
            # Counter propose
            counter = self.respond_to_proposal(
                session_id,
                proposal.proposal_id,
                agent2,
                accept=False
            )
            
            if not counter or session.phase == NegotiationPhase.FAILED:
                break
            
            # Create new proposal from counter
            proposal = Proposal(
                proposal_id="prop_{}".format(random.randint(1000, 9999)),
                proposer_id=agent2,
                target_id=agent1,
                items=counter.items,
                status=ProposalStatus.PENDING,
                timestamp=datetime.now()
            )
            
            session.proposals.append(proposal)
            
            # Evaluate for agent1
            score = self.proposal_evaluator.evaluate_proposal(
                proposal,
                prefs1,
                session.items
            )
            
            if score >= 70.0:
                return self.respond_to_proposal(
                    session_id,
                    proposal.proposal_id,
                    agent1,
                    accept=True
                )
        
        session.phase = NegotiationPhase.FAILED
        session.ended_at = datetime.now()
        
        return None
    
    def get_session(self, session_id: str) -> Optional[NegotiationSession]:
        """Get negotiation session"""
        return self.sessions.get(session_id)
    
    def get_active_sessions(self) -> List[NegotiationSession]:
        """Get all active sessions"""
        return [
            s for s in self.sessions.values()
            if s.phase not in [NegotiationPhase.COMPLETED, NegotiationPhase.FAILED]
        ]
    
    def get_agreements(self, agent_id: Optional[str] = None) -> List[Agreement]:
        """Get agreements"""
        if agent_id:
            return [
                a for a in self.agreements
                if agent_id in a.participants
            ]
        return self.agreements
    
    def cancel_negotiation(self, session_id: str) -> bool:
        """Cancel negotiation session"""
        session = self.sessions.get(session_id)
        
        if not session:
            return False
        
        session.phase = NegotiationPhase.FAILED
        session.ended_at = datetime.now()
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get negotiation statistics"""
        total_sessions = len(self.sessions)
        completed = sum(
            1 for s in self.sessions.values()
            if s.phase == NegotiationPhase.COMPLETED
        )
        failed = sum(
            1 for s in self.sessions.values()
            if s.phase == NegotiationPhase.FAILED
        )
        
        return {
            "total_sessions": total_sessions,
            "completed": completed,
            "failed": failed,
            "success_rate": (completed / total_sessions * 100) if total_sessions > 0 else 0,
            "total_agreements": len(self.agreements),
            "active_sessions": len(self.get_active_sessions())
        }


def demonstrate_negotiation_protocol():
    """Demonstrate agent negotiation protocol"""
    print("=" * 70)
    print("Agent Negotiation Protocol Demonstration")
    print("=" * 70)
    
    protocol = AgentNegotiationProtocol()
    
    # Example 1: Setup negotiation items
    print("\n1. Setting up Negotiation Items:")
    
    items = [
        NegotiationItem(
            "price",
            "Price",
            (100.0, 200.0),
            priority=10,
            negotiable=True
        ),
        NegotiationItem(
            "delivery_time",
            "Delivery Time (days)",
            (1.0, 7.0),
            priority=7,
            negotiable=True
        ),
        NegotiationItem(
            "quality",
            "Quality Level",
            (1.0, 10.0),
            priority=8,
            negotiable=True
        )
    ]
    
    for item in items:
        print("  {}: {} - {}".format(
            item.name,
            item.value_range[0],
            item.value_range[1]
        ))
    
    # Example 2: Set agent preferences
    print("\n2. Setting Agent Preferences:")
    
    protocol.set_agent_preferences("buyer", {
        "price": 120.0,  # Low price preferred
        "delivery_time": 2.0,  # Fast delivery
        "quality": 9.0  # High quality
    })
    
    protocol.set_agent_preferences("seller", {
        "price": 180.0,  # High price preferred
        "delivery_time": 5.0,  # Flexible on time
        "quality": 7.0  # Good quality
    })
    
    print("  Buyer preferences: Low price, fast delivery, high quality")
    print("  Seller preferences: High price, flexible time, good quality")
    
    # Example 3: Start negotiation
    print("\n3. Starting Negotiation:")
    
    session_id = protocol.start_negotiation(
        participants=["buyer", "seller"],
        items=items,
        strategy=NegotiationStrategy.COOPERATIVE,
        max_rounds=10
    )
    
    print("  Session ID: {}".format(session_id))
    
    # Example 4: Manual negotiation
    print("\n4. Manual Negotiation Round:")
    
    # Buyer makes initial proposal
    proposal = protocol.submit_proposal(session_id, "buyer", "seller")
    
    if proposal:
        print("  Buyer's proposal:")
        for item_id, value in proposal.items.items():
            print("    {}: {:.1f}".format(item_id, value))
    
    # Seller evaluates
    session = protocol.get_session(session_id)
    score = protocol.proposal_evaluator.evaluate_proposal(
        proposal,
        protocol.agent_preferences["seller"],
        session.items
    )
    
    print("  Seller's evaluation: {:.1f}/100".format(score))
    
    # Example 5: Counter-proposal
    print("\n5. Counter-Proposal:")
    
    counter = protocol.respond_to_proposal(
        session_id,
        proposal.proposal_id,
        "seller",
        accept=False
    )
    
    if isinstance(counter, CounterProposal):
        print("  Seller's counter-proposal:")
        for item_id, value in counter.items.items():
            print("    {}: {:.1f}".format(item_id, value))
    
    # Example 6: Automatic negotiation
    print("\n6. Automatic Negotiation:")
    
    # Start new session for auto-negotiation
    auto_session_id = protocol.start_negotiation(
        participants=["buyer", "seller"],
        items=items,
        strategy=NegotiationStrategy.COMPROMISE,
        max_rounds=10
    )
    
    agreement = protocol.auto_negotiate(auto_session_id)
    
    if agreement:
        print("  Agreement reached!")
        print("  Agreement ID: {}".format(agreement.agreement_id))
        print("  Final terms:")
        for item_id, value in agreement.items.items():
            print("    {}: {:.1f}".format(item_id, value))
    else:
        print("  Failed to reach agreement")
    
    # Example 7: Different strategies
    print("\n7. Testing Different Strategies:")
    
    strategies = [
        NegotiationStrategy.COOPERATIVE,
        NegotiationStrategy.COMPETITIVE,
        NegotiationStrategy.COMPROMISE
    ]
    
    for strategy in strategies:
        strat_session_id = protocol.start_negotiation(
            participants=["buyer", "seller"],
            items=items,
            strategy=strategy,
            max_rounds=5
        )
        
        strat_agreement = protocol.auto_negotiate(strat_session_id)
        
        print("\n  {} strategy:".format(strategy.value.upper()))
        if strat_agreement:
            print("    Success! Agreement reached")
            print("    Price: {:.1f}".format(strat_agreement.items.get("price", 0)))
        else:
            print("    Failed to reach agreement")
    
    # Example 8: Get active sessions
    print("\n8. Active Negotiation Sessions:")
    
    active = protocol.get_active_sessions()
    print("  Active sessions: {}".format(len(active)))
    
    # Example 9: Get agreements
    print("\n9. Agreement History:")
    
    agreements = protocol.get_agreements()
    print("  Total agreements: {}".format(len(agreements)))
    
    for agreement in agreements:
        print("    {} between {}".format(
            agreement.agreement_id,
            ", ".join(agreement.participants)
        ))
    
    # Example 10: Statistics
    print("\n10. Negotiation Statistics:")
    
    stats = protocol.get_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    demonstrate_negotiation_protocol()
