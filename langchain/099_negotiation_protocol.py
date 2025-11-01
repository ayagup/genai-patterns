"""
Pattern 099: Negotiation Protocol

Description:
    The Negotiation Protocol pattern enables multiple agents to reach agreements through
    structured negotiation processes. This pattern is essential for multi-agent systems
    where agents have different goals, preferences, or constraints and need to find
    mutually acceptable solutions through iterative proposal and counter-proposal exchanges.
    
    Negotiation is a fundamental aspect of multi-agent coordination, allowing agents to
    resolve conflicts, allocate resources, coordinate actions, and make collective decisions.
    This pattern provides mechanisms for proposal generation, evaluation, counter-offers,
    concession strategies, and agreement finalization.
    
    The pattern supports various negotiation strategies including competitive (zero-sum),
    cooperative (win-win), and mixed approaches. It handles single-issue and multi-issue
    negotiations, bilateral and multilateral negotiations, and includes mechanisms for
    handling deadlocks and timeouts.

Key Components:
    1. Proposal: Structured offer from one agent to others
    2. Counter-Proposal: Modified proposal in response to original
    3. Utility Function: Evaluates value of proposals for agent
    4. Concession Strategy: Determines how much to compromise
    5. Negotiation Protocol: Rules governing negotiation process
    6. Agreement: Final mutually accepted terms
    7. Mediation: Optional third-party facilitation

Negotiation Types:
    1. Distributive: Fixed resources, competitive
    2. Integrative: Value creation, cooperative
    3. Multi-issue: Multiple attributes to negotiate
    4. Sequential: Issues addressed one at a time
    5. Package: All issues negotiated together
    
Strategies:
    1. Conceding: Gradually reduce demands
    2. Hardball: Maintain firm position
    3. Tit-for-tat: Mirror opponent's behavior
    4. Time-based: Adjust based on deadline
    5. Information-based: Learn and adapt

Use Cases:
    - Resource allocation in multi-agent systems
    - Task assignment and coordination
    - Conflict resolution between agents
    - Service level agreements
    - Coalition formation
    - Price negotiation in marketplaces
    - Scheduling and planning coordination
    - Policy consensus building

Advantages:
    - Flexible conflict resolution
    - Fair resource distribution
    - Autonomous decision-making
    - Adaptable to agent preferences
    - Handles diverse objectives
    - Enables cooperation
    - Supports complex agreements

Challenges:
    - Computational complexity
    - Strategic manipulation
    - Information asymmetry
    - Deadlock situations
    - Time constraints
    - Preference elicitation
    - Trust issues

LangChain Implementation:
    This implementation uses LangChain for:
    - LLM-based proposal generation and evaluation
    - Utility calculation with reasoning
    - Counter-proposal generation
    - Agreement evaluation
    - Negotiation strategy implementation
    
Production Considerations:
    - Set negotiation timeouts and round limits
    - Implement deadlock breaking mechanisms
    - Log all negotiation steps for transparency
    - Validate proposals against constraints
    - Handle malformed proposals gracefully
    - Monitor for strategic manipulation
    - Implement fair mediation when needed
    - Consider privacy of agent preferences
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ProposalStatus(Enum):
    """Status of a negotiation proposal."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COUNTER = "counter"
    WITHDRAWN = "withdrawn"


class NegotiationOutcome(Enum):
    """Outcome of a negotiation."""
    AGREEMENT = "agreement"
    DEADLOCK = "deadlock"
    TIMEOUT = "timeout"
    WITHDRAWN = "withdrawn"


@dataclass
class Proposal:
    """
    Represents a proposal in a negotiation.
    
    Attributes:
        proposal_id: Unique identifier
        proposer: Agent making the proposal
        recipient: Agent(s) receiving proposal
        round_number: Negotiation round
        terms: Dictionary of proposed terms
        utility: Utility value for proposer
        status: Current status of proposal
        timestamp: When proposal was made
        justification: Reasoning for proposal
        metadata: Additional information
    """
    proposal_id: str
    proposer: str
    recipient: str
    round_number: int
    terms: Dict[str, Any]
    utility: float
    status: ProposalStatus
    timestamp: datetime
    justification: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NegotiationState:
    """
    Tracks the state of an ongoing negotiation.
    
    Attributes:
        negotiation_id: Unique identifier
        participants: Agents involved
        issue: What is being negotiated
        proposals: History of proposals
        current_round: Current round number
        max_rounds: Maximum allowed rounds
        start_time: When negotiation started
        end_time: When negotiation ended
        outcome: Final outcome
        agreement: Final agreed terms
        metadata: Additional information
    """
    negotiation_id: str
    participants: List[str]
    issue: str
    proposals: List[Proposal] = field(default_factory=list)
    current_round: int = 1
    max_rounds: int = 10
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    outcome: Optional[NegotiationOutcome] = None
    agreement: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class NegotiationAgent:
    """
    Agent that participates in negotiations.
    
    This agent can generate proposals, evaluate offers, make counter-proposals,
    and decide when to accept or reject proposals based on its utility function
    and negotiation strategy.
    """
    
    def __init__(
        self,
        name: str,
        preferences: Dict[str, Any],
        reservation_utility: float = 0.3,
        concession_rate: float = 0.1,
        temperature: float = 0.5
    ):
        """
        Initialize negotiation agent.
        
        Args:
            name: Agent's name
            preferences: Agent's preferences for negotiation issues
            reservation_utility: Minimum acceptable utility (walk-away point)
            concession_rate: How quickly agent makes concessions
            temperature: LLM temperature for generation
        """
        self.name = name
        self.preferences = preferences
        self.reservation_utility = reservation_utility
        self.concession_rate = concession_rate
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.negotiation_history: List[Proposal] = []
    
    def calculate_utility(self, terms: Dict[str, Any]) -> float:
        """
        Calculate utility of proposed terms.
        
        Args:
            terms: Proposed terms
            
        Returns:
            Utility value between 0 and 1
        """
        # Simple weighted sum utility calculation
        total_utility = 0.0
        total_weight = 0.0
        
        for key, value in terms.items():
            if key in self.preferences:
                pref = self.preferences[key]
                weight = pref.get("weight", 1.0)
                ideal = pref.get("ideal", value)
                minimum = pref.get("minimum", 0)
                
                # Normalize value
                if ideal != minimum:
                    normalized = (value - minimum) / (ideal - minimum)
                    normalized = max(0.0, min(1.0, normalized))
                else:
                    normalized = 1.0 if value == ideal else 0.0
                
                total_utility += weight * normalized
                total_weight += weight
        
        return total_utility / total_weight if total_weight > 0 else 0.0
    
    def generate_initial_proposal(
        self,
        issue: str,
        recipient: str,
        round_number: int = 1
    ) -> Proposal:
        """
        Generate initial proposal.
        
        Args:
            issue: Issue being negotiated
            recipient: Recipient agent
            round_number: Round number
            
        Returns:
            Initial proposal
        """
        # Start with terms close to ideal preferences
        terms = {}
        for key, pref in self.preferences.items():
            # Start at 90% of ideal for initial proposal
            ideal = pref.get("ideal", 100)
            minimum = pref.get("minimum", 0)
            initial_value = ideal * 0.9 + minimum * 0.1
            terms[key] = initial_value
        
        utility = self.calculate_utility(terms)
        
        # Generate justification using LLM
        prompt = ChatPromptTemplate.from_template(
            "You are {agent_name} negotiating about {issue}. "
            "You propose the following terms: {terms}. "
            "Write a brief 2-3 sentence justification for this proposal, "
            "emphasizing why it's fair and reasonable.\n\n"
            "Justification:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        justification = chain.invoke({
            "agent_name": self.name,
            "issue": issue,
            "terms": terms
        })
        
        proposal = Proposal(
            proposal_id=f"{self.name}_r{round_number}",
            proposer=self.name,
            recipient=recipient,
            round_number=round_number,
            terms=terms,
            utility=utility,
            status=ProposalStatus.PENDING,
            timestamp=datetime.now(),
            justification=justification.strip()
        )
        
        self.negotiation_history.append(proposal)
        return proposal
    
    def evaluate_proposal(
        self,
        proposal: Proposal,
        current_round: int
    ) -> Tuple[bool, str]:
        """
        Evaluate whether to accept a proposal.
        
        Args:
            proposal: Proposal to evaluate
            current_round: Current negotiation round
            
        Returns:
            Tuple of (accept: bool, reasoning: str)
        """
        utility = self.calculate_utility(proposal.terms)
        
        # Accept if utility meets threshold
        # Threshold decreases over rounds (concession)
        round_factor = 1.0 - (current_round * self.concession_rate)
        threshold = max(
            self.reservation_utility,
            self.reservation_utility + (1.0 - self.reservation_utility) * round_factor
        )
        
        accept = utility >= threshold
        
        # Generate reasoning using LLM
        prompt = ChatPromptTemplate.from_template(
            "You are {agent_name} evaluating a proposal about {issue}. "
            "The proposed terms are: {terms}. "
            "The utility you would gain is {utility:.2f} "
            "(your minimum acceptable is {threshold:.2f}). "
            "Decision: {'ACCEPT' if accept else 'REJECT'}. "
            "Write a brief 2-3 sentence explanation for this decision.\n\n"
            "Explanation:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        reasoning = chain.invoke({
            "agent_name": self.name,
            "issue": proposal.metadata.get("issue", "the issue"),
            "terms": proposal.terms,
            "utility": utility,
            "threshold": threshold,
            "accept": accept
        })
        
        return accept, reasoning.strip()
    
    def generate_counter_proposal(
        self,
        previous_proposal: Proposal,
        issue: str,
        round_number: int
    ) -> Proposal:
        """
        Generate counter-proposal.
        
        Args:
            previous_proposal: Proposal to counter
            issue: Issue being negotiated
            round_number: Current round number
            
        Returns:
            Counter-proposal
        """
        # Make concessions based on previous proposal and round number
        terms = {}
        for key, pref in self.preferences.items():
            ideal = pref.get("ideal", 100)
            minimum = pref.get("minimum", 0)
            previous_value = previous_proposal.terms.get(key, minimum)
            
            # Move toward middle ground with concession
            concession = self.concession_rate * round_number
            my_target = ideal * (1 - concession) + minimum * concession
            counter_value = (my_target + previous_value) / 2
            
            # Ensure within bounds
            counter_value = max(minimum, min(ideal, counter_value))
            terms[key] = counter_value
        
        utility = self.calculate_utility(terms)
        
        # Generate justification using LLM
        prompt = ChatPromptTemplate.from_template(
            "You are {agent_name} making a counter-proposal about {issue}. "
            "The previous proposal was: {previous_terms}. "
            "Your counter-proposal is: {terms}. "
            "Write a brief 2-3 sentence justification explaining "
            "how this is a reasonable compromise.\n\n"
            "Justification:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        justification = chain.invoke({
            "agent_name": self.name,
            "issue": issue,
            "previous_terms": previous_proposal.terms,
            "terms": terms
        })
        
        proposal = Proposal(
            proposal_id=f"{self.name}_r{round_number}_counter",
            proposer=self.name,
            recipient=previous_proposal.proposer,
            round_number=round_number,
            terms=terms,
            utility=utility,
            status=ProposalStatus.PENDING,
            timestamp=datetime.now(),
            justification=justification.strip(),
            metadata={"counter_to": previous_proposal.proposal_id}
        )
        
        self.negotiation_history.append(proposal)
        return proposal


class NegotiationMediator:
    """
    Mediator that facilitates negotiation between agents.
    
    The mediator manages the negotiation protocol, enforces rules,
    tracks state, and can suggest compromises when negotiations stall.
    """
    
    def __init__(self, temperature: float = 0.3):
        """
        Initialize negotiation mediator.
        
        Args:
            temperature: LLM temperature for generation
        """
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.negotiations: Dict[str, NegotiationState] = {}
    
    def start_negotiation(
        self,
        negotiation_id: str,
        participants: List[str],
        issue: str,
        max_rounds: int = 10
    ) -> NegotiationState:
        """
        Start a new negotiation.
        
        Args:
            negotiation_id: Unique identifier
            participants: List of participating agents
            issue: Issue being negotiated
            max_rounds: Maximum rounds allowed
            
        Returns:
            Initial negotiation state
        """
        state = NegotiationState(
            negotiation_id=negotiation_id,
            participants=participants,
            issue=issue,
            max_rounds=max_rounds
        )
        self.negotiations[negotiation_id] = state
        return state
    
    def process_round(
        self,
        negotiation_id: str,
        proposal: Proposal,
        response_accept: bool,
        response_reasoning: str,
        counter_proposal: Optional[Proposal] = None
    ) -> NegotiationState:
        """
        Process a negotiation round.
        
        Args:
            negotiation_id: Negotiation identifier
            proposal: Current proposal
            response_accept: Whether proposal was accepted
            response_reasoning: Reasoning for response
            counter_proposal: Counter-proposal if not accepted
            
        Returns:
            Updated negotiation state
        """
        state = self.negotiations[negotiation_id]
        
        # Add proposal to history
        proposal.metadata["response_reasoning"] = response_reasoning
        state.proposals.append(proposal)
        
        if response_accept:
            # Negotiation successful
            proposal.status = ProposalStatus.ACCEPTED
            state.outcome = NegotiationOutcome.AGREEMENT
            state.agreement = proposal.terms
            state.end_time = datetime.now()
        elif counter_proposal:
            # Counter-proposal made
            proposal.status = ProposalStatus.COUNTER
            state.proposals.append(counter_proposal)
            state.current_round += 1
        else:
            # Proposal rejected without counter
            proposal.status = ProposalStatus.REJECTED
            state.current_round += 1
        
        # Check for timeout
        if state.current_round > state.max_rounds and not state.outcome:
            state.outcome = NegotiationOutcome.TIMEOUT
            state.end_time = datetime.now()
        
        return state
    
    def suggest_compromise(
        self,
        negotiation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest compromise when negotiation stalls.
        
        Args:
            negotiation_id: Negotiation identifier
            
        Returns:
            Suggested compromise terms or None
        """
        state = self.negotiations.get(negotiation_id)
        if not state or len(state.proposals) < 2:
            return None
        
        # Get last two proposals
        last_proposals = state.proposals[-2:]
        
        # Calculate middle ground
        compromise = {}
        for key in last_proposals[0].terms.keys():
            values = [p.terms.get(key, 0) for p in last_proposals]
            compromise[key] = sum(values) / len(values)
        
        return compromise
    
    def get_negotiation_summary(
        self,
        negotiation_id: str
    ) -> str:
        """
        Generate summary of negotiation.
        
        Args:
            negotiation_id: Negotiation identifier
            
        Returns:
            Summary text
        """
        state = self.negotiations.get(negotiation_id)
        if not state:
            return "Negotiation not found."
        
        summary = f"Negotiation: {state.issue}\n"
        summary += f"Participants: {', '.join(state.participants)}\n"
        summary += f"Rounds: {state.current_round}/{state.max_rounds}\n"
        summary += f"Outcome: {state.outcome.value if state.outcome else 'ongoing'}\n"
        
        if state.agreement:
            summary += f"\nAgreed Terms:\n"
            for key, value in state.agreement.items():
                summary += f"  - {key}: {value}\n"
        
        summary += f"\nProposal History ({len(state.proposals)} proposals):\n"
        for i, prop in enumerate(state.proposals[-5:], 1):  # Last 5
            summary += f"  {i}. Round {prop.round_number}: "
            summary += f"{prop.proposer} -> {prop.recipient} "
            summary += f"[{prop.status.value}]\n"
        
        return summary


def demonstrate_negotiation_protocol():
    """Demonstrate negotiation protocol pattern."""
    
    print("=" * 80)
    print("NEGOTIATION PROTOCOL PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Simple bilateral negotiation
    print("\n" + "=" * 80)
    print("Example 1: Bilateral Price Negotiation")
    print("=" * 80)
    
    # Create buyer and seller agents
    buyer = NegotiationAgent(
        name="Buyer",
        preferences={
            "price": {"ideal": 100, "minimum": 100, "weight": 1.0},
            "delivery_time": {"ideal": 1, "minimum": 1, "weight": 0.5}
        },
        reservation_utility=0.4,
        concession_rate=0.08
    )
    
    seller = NegotiationAgent(
        name="Seller",
        preferences={
            "price": {"ideal": 150, "minimum": 120, "weight": 1.0},
            "delivery_time": {"ideal": 7, "minimum": 1, "weight": 0.3}
        },
        reservation_utility=0.3,
        concession_rate=0.1
    )
    
    mediator = NegotiationMediator()
    
    # Start negotiation
    issue = "Product Sale Agreement"
    state = mediator.start_negotiation(
        negotiation_id="negotiation_001",
        participants=["Buyer", "Seller"],
        issue=issue,
        max_rounds=8
    )
    
    print(f"\nStarting negotiation: {issue}")
    print(f"Participants: {', '.join(state.participants)}")
    print(f"Maximum rounds: {state.max_rounds}")
    
    # Negotiation rounds
    current_proposer = buyer
    current_recipient = seller
    current_proposal = buyer.generate_initial_proposal(issue, "Seller")
    
    print(f"\n{'Round':<8} {'Proposer':<12} {'Action':<15} {'Price':<10} {'Delivery':<10} {'Utility':<10}")
    print("-" * 80)
    
    while state.current_round <= state.max_rounds and not state.outcome:
        # Display proposal
        print(f"{state.current_round:<8} {current_proposal.proposer:<12} "
              f"{'Proposes':<15} "
              f"{current_proposal.terms.get('price', 0):<10.1f} "
              f"{current_proposal.terms.get('delivery_time', 0):<10.1f} "
              f"{current_proposal.utility:<10.2f}")
        
        # Recipient evaluates proposal
        accept, reasoning = current_recipient.evaluate_proposal(
            current_proposal,
            state.current_round
        )
        
        if accept:
            # Agreement reached
            state = mediator.process_round(
                "negotiation_001",
                current_proposal,
                True,
                reasoning
            )
            print(f"{'':<8} {current_recipient.name:<12} {'✓ ACCEPTS':<15}")
            break
        else:
            # Generate counter-proposal
            counter = current_recipient.generate_counter_proposal(
                current_proposal,
                issue,
                state.current_round
            )
            
            print(f"{'':<8} {current_recipient.name:<12} {'✗ Rejects':<15}")
            
            state = mediator.process_round(
                "negotiation_001",
                current_proposal,
                False,
                reasoning,
                counter
            )
            
            # Switch roles
            current_proposal = counter
            current_proposer, current_recipient = current_recipient, current_proposer
    
    # Print outcome
    print("\n" + "-" * 80)
    print(f"Outcome: {state.outcome.value.upper()}")
    if state.agreement:
        print("\nFinal Agreement:")
        for key, value in state.agreement.items():
            print(f"  - {key}: {value:.2f}")
        
        print(f"\nUtilities:")
        print(f"  Buyer utility: {buyer.calculate_utility(state.agreement):.2f}")
        print(f"  Seller utility: {seller.calculate_utility(state.agreement):.2f}")
    
    # Example 2: Multi-issue negotiation
    print("\n" + "=" * 80)
    print("Example 2: Multi-Issue Resource Allocation")
    print("=" * 80)
    
    agent_a = NegotiationAgent(
        name="Agent_A",
        preferences={
            "cpu_cores": {"ideal": 16, "minimum": 4, "weight": 1.0},
            "memory_gb": {"ideal": 64, "minimum": 16, "weight": 0.8},
            "storage_tb": {"ideal": 10, "minimum": 2, "weight": 0.6}
        },
        reservation_utility=0.35,
        concession_rate=0.12
    )
    
    agent_b = NegotiationAgent(
        name="Agent_B",
        preferences={
            "cpu_cores": {"ideal": 12, "minimum": 4, "weight": 0.9},
            "memory_gb": {"ideal": 48, "minimum": 16, "weight": 1.0},
            "storage_tb": {"ideal": 8, "minimum": 2, "weight": 0.5}
        },
        reservation_utility=0.3,
        concession_rate=0.1
    )
    
    mediator2 = NegotiationMediator()
    
    issue2 = "Server Resource Allocation"
    state2 = mediator2.start_negotiation(
        negotiation_id="negotiation_002",
        participants=["Agent_A", "Agent_B"],
        issue=issue2,
        max_rounds=6
    )
    
    print(f"\nNegotiating: {issue2}")
    print(f"Available resources to allocate between agents")
    
    # Run negotiation
    proposer = agent_a
    recipient = agent_b
    proposal = agent_a.generate_initial_proposal(issue2, "Agent_B")
    
    print(f"\n{'Round':<8} {'Proposer':<12} {'CPU':<10} {'Memory':<12} {'Storage':<12} {'Decision':<15}")
    print("-" * 80)
    
    for round_num in range(1, state2.max_rounds + 1):
        print(f"{round_num:<8} {proposal.proposer:<12} "
              f"{proposal.terms.get('cpu_cores', 0):<10.1f} "
              f"{proposal.terms.get('memory_gb', 0):<12.1f} "
              f"{proposal.terms.get('storage_tb', 0):<12.1f}", end=" ")
        
        accept, reasoning = recipient.evaluate_proposal(proposal, round_num)
        
        if accept:
            state2 = mediator2.process_round(
                "negotiation_002",
                proposal,
                True,
                reasoning
            )
            print("✓ ACCEPTED")
            break
        else:
            print("✗ Counter")
            counter = recipient.generate_counter_proposal(proposal, issue2, round_num)
            state2 = mediator2.process_round(
                "negotiation_002",
                proposal,
                False,
                reasoning,
                counter
            )
            proposal = counter
            proposer, recipient = recipient, proposer
    
    print("\n" + "-" * 80)
    if state2.agreement:
        print("Agreement Reached!")
        print(f"\nFinal Resource Allocation:")
        for key, value in state2.agreement.items():
            print(f"  {key}: {value:.1f}")
    
    # Example 3: Negotiation with mediation
    print("\n" + "=" * 80)
    print("Example 3: Mediated Negotiation with Compromise")
    print("=" * 80)
    
    # Stubborn agents with conflicting preferences
    stubborn_a = NegotiationAgent(
        name="Stubborn_A",
        preferences={
            "budget": {"ideal": 50000, "minimum": 40000, "weight": 1.0}
        },
        reservation_utility=0.7,  # Very high reservation
        concession_rate=0.02  # Very slow concessions
    )
    
    stubborn_b = NegotiationAgent(
        name="Stubborn_B",
        preferences={
            "budget": {"ideal": 80000, "minimum": 70000, "weight": 1.0}
        },
        reservation_utility=0.7,
        concession_rate=0.02
    )
    
    mediator3 = NegotiationMediator()
    
    state3 = mediator3.start_negotiation(
        negotiation_id="negotiation_003",
        participants=["Stubborn_A", "Stubborn_B"],
        issue="Project Budget Allocation",
        max_rounds=5
    )
    
    print("\nNegotiation between stubborn agents (likely to deadlock)")
    
    # Run negotiation
    proposer = stubborn_a
    recipient = stubborn_b
    proposal = stubborn_a.generate_initial_proposal(
        "Project Budget Allocation",
        "Stubborn_B"
    )
    
    for round_num in range(1, 6):
        print(f"\nRound {round_num}:")
        print(f"  Proposal from {proposal.proposer}: Budget = ${proposal.terms['budget']:,.0f}")
        
        accept, reasoning = recipient.evaluate_proposal(proposal, round_num)
        
        if accept:
            state3 = mediator3.process_round(
                "negotiation_003",
                proposal,
                True,
                reasoning
            )
            print(f"  {recipient.name} accepts!")
            break
        else:
            print(f"  {recipient.name} rejects")
            counter = recipient.generate_counter_proposal(
                proposal,
                "Project Budget Allocation",
                round_num
            )
            state3 = mediator3.process_round(
                "negotiation_003",
                proposal,
                False,
                reasoning,
                counter
            )
            proposal = counter
            proposer, recipient = recipient, proposer
    
    if not state3.outcome:
        print("\nDeadlock detected! Mediator suggests compromise...")
        compromise = mediator3.suggest_compromise("negotiation_003")
        if compromise:
            print(f"Suggested compromise: Budget = ${compromise['budget']:,.0f}")
            print(f"  Utility for Stubborn_A: {stubborn_a.calculate_utility(compromise):.2f}")
            print(f"  Utility for Stubborn_B: {stubborn_b.calculate_utility(compromise):.2f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Negotiation Protocol Pattern")
    print("=" * 80)
    
    summary = """
    The Negotiation Protocol pattern demonstrated:
    
    1. BILATERAL NEGOTIATION (Example 1):
       - Two agents negotiating price and delivery
       - Iterative proposal and counter-proposal
       - Utility-based evaluation
       - Gradual concessions leading to agreement
       - Both agents achieve acceptable utility
    
    2. MULTI-ISSUE NEGOTIATION (Example 2):
       - Resource allocation across multiple dimensions
       - Weighted preferences for different attributes
       - Trade-offs between competing interests
       - Efficient agreement through compromise
    
    3. MEDIATED NEGOTIATION (Example 3):
       - Stubborn agents with conflicting goals
       - Deadlock detection
       - Mediator-suggested compromise
       - Breaking negotiation impasses
    
    KEY BENEFITS:
    ✓ Autonomous conflict resolution
    ✓ Fair resource distribution
    ✓ Adaptable to agent preferences
    ✓ Handles complex multi-issue negotiations
    ✓ Supports cooperative and competitive scenarios
    ✓ Transparent reasoning through justifications
    ✓ Mediation for difficult negotiations
    
    USE CASES:
    • Resource allocation in distributed systems
    • Task assignment and scheduling
    • Service level agreement negotiation
    • Price negotiation in marketplaces
    • Coalition formation
    • Conflict resolution between agents
    • Policy consensus building
    
    BEST PRACTICES:
    1. Set clear reservation utilities (walk-away points)
    2. Implement gradual concession strategies
    3. Use utility functions to evaluate proposals
    4. Track negotiation history for transparency
    5. Set round limits to prevent infinite loops
    6. Provide justifications for proposals
    7. Use mediation for deadlock situations
    8. Consider fairness and efficiency
    
    TRADE-OFFS:
    • Computational cost vs. negotiation quality
    • Speed of agreement vs. optimal outcome
    • Information sharing vs. strategic advantage
    • Autonomy vs. mediation intervention
    
    PRODUCTION CONSIDERATIONS:
    → Monitor negotiation metrics (rounds, success rate)
    → Log all proposals for audit trails
    → Implement timeout and deadlock detection
    → Validate proposals against constraints
    → Handle edge cases (withdrawal, malformed proposals)
    → Consider privacy of agent preferences
    → Test with diverse agent strategies
    → Implement fair mediation mechanisms
    
    This pattern enables sophisticated multi-agent coordination through
    structured negotiation, allowing agents to autonomously reach agreements
    while respecting individual preferences and constraints.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_negotiation_protocol()
