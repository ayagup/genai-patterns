"""
Negotiation Protocol Pattern

Enables agents to negotiate with each other to reach mutually acceptable
agreements through bargaining, compromise, and consensus-building strategies.

Key Concepts:
- Multi-party negotiation
- Bargaining strategies
- Compromise mechanisms
- Utility maximization
- Conflict resolution

Use Cases:
- Resource allocation
- Task distribution
- Conflict resolution
- Multi-agent coordination
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class NegotiationStrategy(Enum):
    """Negotiation strategies."""
    COMPETITIVE = "competitive"  # Maximize own utility
    COOPERATIVE = "cooperative"  # Maximize joint utility
    COMPROMISING = "compromising"  # Meet in the middle
    ACCOMMODATING = "accommodating"  # Prioritize other's needs
    AVOIDING = "avoiding"  # Minimal engagement


class ProposalStatus(Enum):
    """Status of a negotiation proposal."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTERED = "countered"
    WITHDRAWN = "withdrawn"


@dataclass
class Resource:
    """Represents a resource to be negotiated."""
    name: str
    quantity: float
    unit: str
    value: float  # Base value per unit
    
    def __repr__(self) -> str:
        return f"{self.quantity} {self.unit} of {self.name}"


@dataclass
class Proposal:
    """Represents a negotiation proposal."""
    id: str
    proposer_id: str
    recipient_id: str
    resources_offered: List[Resource]
    resources_requested: List[Resource]
    terms: Dict[str, Any] = field(default_factory=dict)
    status: ProposalStatus = ProposalStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    counter_to: Optional[str] = None  # ID of proposal this counters
    
    def calculate_utility(self, agent_id: str, preferences: Dict[str, float]) -> float:
        """Calculate utility of proposal for an agent."""
        utility = 0.0
        
        if agent_id == self.proposer_id:
            # Value of received resources minus cost of given resources
            for resource in self.resources_requested:
                value = resource.quantity * preferences.get(resource.name, resource.value)
                utility += value
            
            for resource in self.resources_offered:
                cost = resource.quantity * preferences.get(resource.name, resource.value)
                utility -= cost
        else:
            # Value of received resources minus cost of given resources
            for resource in self.resources_offered:
                value = resource.quantity * preferences.get(resource.name, resource.value)
                utility += value
            
            for resource in self.resources_requested:
                cost = resource.quantity * preferences.get(resource.name, resource.value)
                utility -= cost
        
        return utility


@dataclass
class NegotiationOutcome:
    """Result of a negotiation."""
    success: bool
    final_proposal: Optional[Proposal]
    rounds: int
    participants: List[str]
    utility_distribution: Dict[str, float]
    agreement_terms: Dict[str, Any] = field(default_factory=dict)


class NegotiatingAgent:
    """Agent capable of participating in negotiations."""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        strategy: NegotiationStrategy = NegotiationStrategy.COOPERATIVE,
        resources: Optional[Dict[str, float]] = None
    ):
        self.agent_id = agent_id
        self.name = name
        self.strategy = strategy
        self.resources = resources or {}  # resource_name -> quantity
        self.preferences: Dict[str, float] = {}  # resource_name -> value
        self.reservation_utility = 0.0  # Minimum acceptable utility
        self.proposals_made: List[Proposal] = []
        self.proposals_received: List[Proposal] = []
    
    def set_preferences(self, preferences: Dict[str, float]) -> None:
        """Set agent's resource preferences/values."""
        self.preferences = preferences
    
    def set_reservation_utility(self, utility: float) -> None:
        """Set minimum acceptable utility (BATNA)."""
        self.reservation_utility = utility
    
    def has_resource(self, resource_name: str, quantity: float) -> bool:
        """Check if agent has sufficient resource."""
        return self.resources.get(resource_name, 0.0) >= quantity
    
    def make_proposal(
        self,
        recipient_id: str,
        offer: List[Resource],
        request: List[Resource],
        terms: Optional[Dict[str, Any]] = None
    ) -> Proposal:
        """Create a negotiation proposal."""
        proposal = Proposal(
            id=f"prop_{len(self.proposals_made) + 1}",
            proposer_id=self.agent_id,
            recipient_id=recipient_id,
            resources_offered=offer,
            resources_requested=request,
            terms=terms or {}
        )
        
        self.proposals_made.append(proposal)
        return proposal
    
    def evaluate_proposal(self, proposal: Proposal) -> Tuple[bool, float]:
        """Evaluate if a proposal is acceptable."""
        utility = proposal.calculate_utility(self.agent_id, self.preferences)
        
        # Check if meets reservation utility
        if utility < self.reservation_utility:
            return False, utility
        
        # Strategy-specific evaluation
        if self.strategy == NegotiationStrategy.COMPETITIVE:
            # Accept only if highly beneficial
            threshold = self.reservation_utility * 1.5
            return utility >= threshold, utility
        
        elif self.strategy == NegotiationStrategy.COOPERATIVE:
            # Accept if mutually beneficial
            return utility >= self.reservation_utility, utility
        
        elif self.strategy == NegotiationStrategy.ACCOMMODATING:
            # More willing to accept
            threshold = self.reservation_utility * 0.8
            return utility >= threshold, utility
        
        else:
            return utility >= self.reservation_utility, utility
    
    def generate_counteroffer(
        self,
        original_proposal: Proposal,
        target_utility: Optional[float] = None
    ) -> Proposal:
        """Generate a counteroffer to a proposal."""
        if target_utility is None:
            target_utility = self.reservation_utility * 1.2
        
        # Adjust quantities based on strategy
        adjusted_offer = []
        for resource in original_proposal.resources_requested:
            if self.strategy == NegotiationStrategy.COMPETITIVE:
                # Offer less
                adjusted_quantity = resource.quantity * 0.8
            elif self.strategy == NegotiationStrategy.ACCOMMODATING:
                # Offer more
                adjusted_quantity = resource.quantity * 1.1
            else:
                # Compromise - meet in middle
                adjusted_quantity = resource.quantity * 0.9
            
            adjusted_offer.append(Resource(
                name=resource.name,
                quantity=adjusted_quantity,
                unit=resource.unit,
                value=resource.value
            ))
        
        adjusted_request = []
        for resource in original_proposal.resources_offered:
            if self.strategy == NegotiationStrategy.COMPETITIVE:
                # Request more
                adjusted_quantity = resource.quantity * 1.2
            elif self.strategy == NegotiationStrategy.ACCOMMODATING:
                # Request less
                adjusted_quantity = resource.quantity * 0.9
            else:
                # Compromise
                adjusted_quantity = resource.quantity * 1.1
            
            adjusted_request.append(Resource(
                name=resource.name,
                quantity=adjusted_quantity,
                unit=resource.unit,
                value=resource.value
            ))
        
        counter = self.make_proposal(
            recipient_id=original_proposal.proposer_id,
            offer=adjusted_offer,
            request=adjusted_request,
            terms=original_proposal.terms
        )
        counter.counter_to = original_proposal.id
        
        return counter
    
    def update_resources(self, proposal: Proposal) -> None:
        """Update resource inventory after agreement."""
        if proposal.proposer_id == self.agent_id:
            # Remove offered resources
            for resource in proposal.resources_offered:
                self.resources[resource.name] = \
                    self.resources.get(resource.name, 0.0) - resource.quantity
            
            # Add requested resources
            for resource in proposal.resources_requested:
                self.resources[resource.name] = \
                    self.resources.get(resource.name, 0.0) + resource.quantity
        else:
            # Add offered resources
            for resource in proposal.resources_offered:
                self.resources[resource.name] = \
                    self.resources.get(resource.name, 0.0) + resource.quantity
            
            # Remove requested resources
            for resource in proposal.resources_requested:
                self.resources[resource.name] = \
                    self.resources.get(resource.name, 0.0) - resource.quantity


class NegotiationProtocol:
    """Manages negotiation process between agents."""
    
    def __init__(self, max_rounds: int = 10):
        self.max_rounds = max_rounds
        self.agents: Dict[str, NegotiatingAgent] = {}
        self.negotiation_history: List[Proposal] = []
    
    def register_agent(self, agent: NegotiatingAgent) -> None:
        """Register an agent for negotiation."""
        self.agents[agent.agent_id] = agent
        print(f"[Protocol] Registered agent: {agent.name}")
    
    def bilateral_negotiation(
        self,
        proposer_id: str,
        recipient_id: str,
        initial_proposal: Proposal
    ) -> NegotiationOutcome:
        """Conduct bilateral negotiation between two agents."""
        print(f"\n[Protocol] Starting negotiation between {proposer_id} and {recipient_id}")
        
        if proposer_id not in self.agents or recipient_id not in self.agents:
            return NegotiationOutcome(
                success=False,
                final_proposal=None,
                rounds=0,
                participants=[proposer_id, recipient_id],
                utility_distribution={}
            )
        
        proposer = self.agents[proposer_id]
        recipient = self.agents[recipient_id]
        
        current_proposal = initial_proposal
        self.negotiation_history.append(current_proposal)
        
        for round_num in range(1, self.max_rounds + 1):
            print(f"\nRound {round_num}:")
            print(f"  Proposal: {proposer_id} offers {current_proposal.resources_offered}")
            print(f"           requests {current_proposal.resources_requested}")
            
            # Recipient evaluates proposal
            acceptable, utility = recipient.evaluate_proposal(current_proposal)
            print(f"  {recipient_id} utility: {utility:.2f}")
            
            if acceptable:
                # Agreement reached
                current_proposal.status = ProposalStatus.ACCEPTED
                print(f"  ✓ Proposal accepted!")
                
                # Update resources
                proposer.update_resources(current_proposal)
                recipient.update_resources(current_proposal)
                
                # Calculate utilities
                proposer_utility = current_proposal.calculate_utility(
                    proposer_id, proposer.preferences
                )
                recipient_utility = utility
                
                return NegotiationOutcome(
                    success=True,
                    final_proposal=current_proposal,
                    rounds=round_num,
                    participants=[proposer_id, recipient_id],
                    utility_distribution={
                        proposer_id: proposer_utility,
                        recipient_id: recipient_utility
                    }
                )
            else:
                # Generate counteroffer
                current_proposal.status = ProposalStatus.COUNTERED
                print(f"  ✗ Proposal rejected, generating counteroffer...")
                
                counter = recipient.generate_counteroffer(current_proposal)
                self.negotiation_history.append(counter)
                
                # Check if counteroffer is acceptable to proposer
                acceptable, utility = proposer.evaluate_proposal(counter)
                print(f"  {proposer_id} evaluates counter, utility: {utility:.2f}")
                
                if acceptable:
                    counter.status = ProposalStatus.ACCEPTED
                    print(f"  ✓ Counteroffer accepted!")
                    
                    # Update resources
                    proposer.update_resources(counter)
                    recipient.update_resources(counter)
                    
                    proposer_utility = utility
                    recipient_utility = counter.calculate_utility(
                        recipient_id, recipient.preferences
                    )
                    
                    return NegotiationOutcome(
                        success=True,
                        final_proposal=counter,
                        rounds=round_num,
                        participants=[proposer_id, recipient_id],
                        utility_distribution={
                            proposer_id: proposer_utility,
                            recipient_id: recipient_utility
                        }
                    )
                else:
                    # Continue negotiation with adjusted proposal
                    current_proposal = counter
                    # Swap roles
                    proposer, recipient = recipient, proposer
                    proposer_id, recipient_id = recipient_id, proposer_id
        
        print(f"\n  ✗ Negotiation failed after {self.max_rounds} rounds")
        return NegotiationOutcome(
            success=False,
            final_proposal=current_proposal,
            rounds=self.max_rounds,
            participants=[proposer_id, recipient_id],
            utility_distribution={}
        )


def demonstrate_negotiation():
    """Demonstrate negotiation protocol."""
    print("=" * 60)
    print("NEGOTIATION PROTOCOL DEMONSTRATION")
    print("=" * 60)
    
    # Create negotiation protocol
    protocol = NegotiationProtocol(max_rounds=5)
    
    # Create agents with different strategies
    alice = NegotiatingAgent(
        "alice",
        "Alice (Cooperative)",
        strategy=NegotiationStrategy.COOPERATIVE,
        resources={"compute": 100.0, "storage": 50.0}
    )
    
    bob = NegotiatingAgent(
        "bob",
        "Bob (Competitive)",
        strategy=NegotiationStrategy.COMPETITIVE,
        resources={"bandwidth": 200.0, "memory": 80.0}
    )
    
    carol = NegotiatingAgent(
        "carol",
        "Carol (Accommodating)",
        strategy=NegotiationStrategy.ACCOMMODATING,
        resources={"compute": 75.0, "bandwidth": 150.0}
    )
    
    # Set preferences (how much each agent values each resource)
    alice.set_preferences({"compute": 1.0, "storage": 1.5, "bandwidth": 2.0, "memory": 1.2})
    bob.set_preferences({"compute": 2.5, "storage": 1.0, "bandwidth": 1.0, "memory": 1.5})
    carol.set_preferences({"compute": 1.5, "storage": 2.0, "bandwidth": 1.0, "memory": 2.0})
    
    # Set reservation utilities (BATNA - Best Alternative To Negotiated Agreement)
    alice.set_reservation_utility(10.0)
    bob.set_reservation_utility(20.0)
    carol.set_reservation_utility(5.0)
    
    # Register agents
    protocol.register_agent(alice)
    protocol.register_agent(bob)
    protocol.register_agent(carol)
    
    # Demonstration 1: Alice and Bob negotiate (Cooperative vs Competitive)
    print("\n" + "=" * 60)
    print("1. Alice (Cooperative) vs Bob (Competitive)")
    print("=" * 60)
    
    proposal1 = alice.make_proposal(
        recipient_id="bob",
        offer=[Resource("compute", 20.0, "units", 1.0)],
        request=[Resource("bandwidth", 50.0, "Mbps", 1.0)],
        terms={"duration": "1 hour", "priority": "normal"}
    )
    
    outcome1 = protocol.bilateral_negotiation("alice", "bob", proposal1)
    
    print(f"\n--- Negotiation Result ---")
    print(f"Success: {outcome1.success}")
    print(f"Rounds: {outcome1.rounds}")
    if outcome1.success:
        print(f"Utility distribution:")
        for agent_id, utility in outcome1.utility_distribution.items():
            print(f"  {agent_id}: {utility:.2f}")
    
    # Demonstration 2: Alice and Carol negotiate (Cooperative vs Accommodating)
    print("\n" + "=" * 60)
    print("2. Alice (Cooperative) vs Carol (Accommodating)")
    print("=" * 60)
    
    proposal2 = alice.make_proposal(
        recipient_id="carol",
        offer=[Resource("storage", 10.0, "GB", 1.5)],
        request=[Resource("bandwidth", 30.0, "Mbps", 1.0)],
        terms={"duration": "30 minutes", "priority": "high"}
    )
    
    outcome2 = protocol.bilateral_negotiation("alice", "carol", proposal2)
    
    print(f"\n--- Negotiation Result ---")
    print(f"Success: {outcome2.success}")
    print(f"Rounds: {outcome2.rounds}")
    if outcome2.success:
        print(f"Utility distribution:")
        for agent_id, utility in outcome2.utility_distribution.items():
            print(f"  {agent_id}: {utility:.2f}")
    
    # Show final resource allocation
    print("\n" + "=" * 60)
    print("Final Resource Allocation")
    print("=" * 60)
    
    for agent_id, agent in protocol.agents.items():
        print(f"\n{agent.name}:")
        for resource_name, quantity in agent.resources.items():
            print(f"  {resource_name}: {quantity:.1f}")


if __name__ == "__main__":
    demonstrate_negotiation()
