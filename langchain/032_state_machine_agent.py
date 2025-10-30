"""
Pattern 032: State Machine Agent

Description:
    State machine agents define behavior through explicit states and transitions.
    This pattern provides deterministic, predictable agent behavior with clear
    state management, transition conditions, and event handling. Ideal for
    workflows requiring strict control and auditability.

Components:
    - State: Discrete condition of the agent
    - Transition: Rule for moving between states
    - Event: Trigger that causes transitions
    - Guard: Condition that must be met for transition
    - Action: Operation performed during transition
    - State Context: Data maintained across states

Use Cases:
    - Workflow automation
    - Conversation management
    - Game AI with distinct phases
    - Process orchestration
    - Regulatory compliance scenarios

LangChain Implementation:
    Uses state definitions, transition rules, and event handling to create
    predictable agent behavior. Can be integrated with LangGraph for
    visual workflow representation.
"""

import os
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
import json
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class EventType(Enum):
    """Types of events that can trigger transitions."""
    USER_INPUT = "user_input"
    TIMEOUT = "timeout"
    COMPLETION = "completion"
    ERROR = "error"
    EXTERNAL = "external"
    INTERNAL = "internal"


@dataclass
class Event:
    """An event that can trigger state transitions."""
    type: EventType
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class State:
    """
    Represents a discrete state in the state machine.
    
    States define what the agent can do and how it responds.
    """
    name: str
    description: str
    entry_action: Optional[Callable] = None  # Called when entering state
    exit_action: Optional[Callable] = None  # Called when leaving state
    allowed_events: Set[str] = field(default_factory=set)  # Events accepted in this state
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, State) and self.name == other.name


@dataclass
class Transition:
    """
    Defines a transition between states.
    
    Transitions specify how and when the agent moves between states.
    """
    from_state: str
    to_state: str
    event: str
    guard: Optional[Callable[[Dict[str, Any]], bool]] = None  # Condition function
    action: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None  # Transition action
    priority: int = 0  # Higher priority transitions checked first
    description: str = ""


@dataclass
class StateContext:
    """
    Maintains data across state transitions.
    
    The context is preserved and passed through the state machine.
    """
    data: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)  # State history
    event_log: List[Event] = field(default_factory=list)  # Event history
    
    def set(self, key: str, value: Any):
        """Set context value."""
        self.data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get context value."""
        return self.data.get(key, default)
    
    def log_event(self, event: Event):
        """Log an event."""
        self.event_log.append(event)
    
    def log_state(self, state_name: str):
        """Log state transition."""
        self.history.append(state_name)


class StateMachine:
    """
    Implements a finite state machine.
    
    Features:
    - State management
    - Event-driven transitions
    - Guard conditions
    - Entry/exit actions
    - Context preservation
    """
    
    def __init__(self, initial_state: str):
        self.states: Dict[str, State] = {}
        self.transitions: List[Transition] = []
        self.current_state: Optional[str] = None
        self.initial_state = initial_state
        self.context = StateContext()
        
        # Index transitions by from_state and event for efficiency
        self.transition_index: Dict[tuple, List[Transition]] = defaultdict(list)
    
    def add_state(self, state: State):
        """Add a state to the machine."""
        self.states[state.name] = state
    
    def add_transition(self, transition: Transition):
        """Add a transition to the machine."""
        self.transitions.append(transition)
        # Index by (from_state, event) for quick lookup
        key = (transition.from_state, transition.event)
        self.transition_index[key].append(transition)
        # Sort by priority
        self.transition_index[key].sort(key=lambda t: t.priority, reverse=True)
    
    def start(self):
        """Initialize the state machine."""
        if self.initial_state not in self.states:
            raise ValueError(f"Initial state '{self.initial_state}' not found")
        
        self.current_state = self.initial_state
        self.context.log_state(self.current_state)
        
        # Execute entry action
        state = self.states[self.current_state]
        if state.entry_action:
            state.entry_action(self.context)
    
    def handle_event(self, event: Event) -> bool:
        """
        Process an event and potentially transition to a new state.
        
        Returns:
            True if a transition occurred, False otherwise
        """
        if not self.current_state:
            raise RuntimeError("State machine not started")
        
        # Log event
        self.context.log_event(event)
        
        # Check if event is allowed in current state
        current_state_obj = self.states[self.current_state]
        if current_state_obj.allowed_events and event.name not in current_state_obj.allowed_events:
            return False
        
        # Find applicable transition
        key = (self.current_state, event.name)
        possible_transitions = self.transition_index.get(key, [])
        
        for transition in possible_transitions:
            # Check guard condition
            if transition.guard and not transition.guard(self.context.data):
                continue
            
            # Transition is valid - execute it
            self._execute_transition(transition, event)
            return True
        
        return False
    
    def _execute_transition(self, transition: Transition, event: Event):
        """Execute a state transition."""
        old_state = self.current_state
        new_state = transition.to_state
        
        # Exit action of current state
        if self.states[old_state].exit_action:
            self.states[old_state].exit_action(self.context)
        
        # Transition action
        if transition.action:
            result = transition.action(self.context.data)
            if result:
                self.context.data.update(result)
        
        # Update current state
        self.current_state = new_state
        self.context.log_state(new_state)
        
        # Entry action of new state
        if self.states[new_state].entry_action:
            self.states[new_state].entry_action(self.context)
    
    def get_current_state(self) -> Optional[State]:
        """Get the current state object."""
        if self.current_state:
            return self.states[self.current_state]
        return None
    
    def get_available_events(self) -> Set[str]:
        """Get events that can be handled in current state."""
        if not self.current_state:
            return set()
        
        # Get events from transitions
        events = set()
        for transition in self.transitions:
            if transition.from_state == self.current_state:
                events.add(transition.event)
        
        return events
    
    def is_in_state(self, state_name: str) -> bool:
        """Check if currently in specified state."""
        return self.current_state == state_name
    
    def get_state_history(self) -> List[str]:
        """Get history of states visited."""
        return self.context.history.copy()


class StateMachineAgent:
    """
    An agent that uses a state machine for behavior control.
    
    Combines state machine determinism with LLM flexibility for
    decision-making within states.
    """
    
    def __init__(self, state_machine: StateMachine, temperature: float = 0.3):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        self.state_machine = state_machine
        
        # Prompt for processing within a state
        self.process_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI agent operating in a state machine.

Current State: {state_name}
State Description: {state_description}
Available Events: {available_events}
Context: {context}

User Input: {user_input}

Based on the current state and user input, determine:
1. What event should be triggered (choose from available events)
2. What data should be included with the event

Respond in JSON format:
{{
  "event": "event_name",
  "data": {{...}}
}}"""),
            ("user", "{user_input}")
        ])
    
    def start(self):
        """Start the agent's state machine."""
        self.state_machine.start()
        print(f"Agent started in state: {self.state_machine.current_state}")
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input within current state context.
        
        Uses LLM to determine appropriate event to trigger.
        """
        current_state = self.state_machine.get_current_state()
        
        if not current_state:
            return {"error": "State machine not started"}
        
        available_events = self.state_machine.get_available_events()
        
        # Use LLM to determine event and data
        chain = self.process_prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "state_name": current_state.name,
            "state_description": current_state.description,
            "available_events": ", ".join(available_events),
            "context": json.dumps(self.state_machine.context.data),
            "user_input": user_input
        })
        
        # Parse response
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
            else:
                parsed = json.loads(response)
            
            event_name = parsed.get("event")
            event_data = parsed.get("data", {})
            
            # Create and handle event
            event = Event(
                type=EventType.USER_INPUT,
                name=event_name,
                data=event_data
            )
            
            transitioned = self.state_machine.handle_event(event)
            
            return {
                "success": True,
                "event": event_name,
                "transitioned": transitioned,
                "new_state": self.state_machine.current_state,
                "response": self._generate_response()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_response": response
            }
    
    def _generate_response(self) -> str:
        """Generate a response based on current state."""
        current_state = self.state_machine.get_current_state()
        
        if not current_state:
            return "I'm not in a valid state."
        
        # Generate contextual response
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are in state: {state_name}
{state_description}

Generate a brief, helpful response acknowledging the state and guiding the user."""),
            ("user", "Generate response")
        ])
        
        chain = response_prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "state_name": current_state.name,
            "state_description": current_state.description
        })
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        current_state = self.state_machine.get_current_state()
        
        return {
            "current_state": self.state_machine.current_state,
            "state_description": current_state.description if current_state else "",
            "available_events": list(self.state_machine.get_available_events()),
            "context": self.state_machine.context.data,
            "state_history": self.state_machine.get_state_history()
        }


def create_conversation_agent() -> StateMachineAgent:
    """
    Create a conversation agent with state machine.
    
    States: greeting → collecting_info → processing → providing_result → farewell
    """
    # Create state machine
    sm = StateMachine(initial_state="greeting")
    
    # Define states
    states = [
        State(
            name="greeting",
            description="Initial greeting state, welcoming the user",
            allowed_events={"start_conversation", "ask_question"}
        ),
        State(
            name="collecting_info",
            description="Collecting information from the user",
            allowed_events={"provide_info", "clarify", "cancel"}
        ),
        State(
            name="processing",
            description="Processing user request",
            allowed_events={"complete", "error"}
        ),
        State(
            name="providing_result",
            description="Providing results to user",
            allowed_events={"ask_followup", "satisfied", "retry"}
        ),
        State(
            name="farewell",
            description="Ending the conversation",
            allowed_events={"restart"}
        ),
        State(
            name="error",
            description="Error state",
            allowed_events={"retry", "cancel"}
        )
    ]
    
    for state in states:
        sm.add_state(state)
    
    # Define transitions
    transitions = [
        # From greeting
        Transition("greeting", "collecting_info", "start_conversation",
                  description="User starts conversation"),
        Transition("greeting", "providing_result", "ask_question",
                  description="User asks direct question"),
        
        # From collecting_info
        Transition("collecting_info", "processing", "provide_info",
                  description="User provides required info"),
        Transition("collecting_info", "collecting_info", "clarify",
                  description="User needs clarification"),
        Transition("collecting_info", "farewell", "cancel",
                  description="User cancels"),
        
        # From processing
        Transition("processing", "providing_result", "complete",
                  description="Processing completed successfully"),
        Transition("processing", "error", "error",
                  description="Error during processing"),
        
        # From providing_result
        Transition("providing_result", "collecting_info", "ask_followup",
                  description="User has follow-up questions"),
        Transition("providing_result", "farewell", "satisfied",
                  description="User is satisfied"),
        Transition("providing_result", "processing", "retry",
                  description="User wants to retry"),
        
        # From error
        Transition("error", "processing", "retry",
                  description="Retry after error"),
        Transition("error", "farewell", "cancel",
                  description="Cancel after error"),
        
        # From farewell
        Transition("farewell", "greeting", "restart",
                  description="Start new conversation")
    ]
    
    for transition in transitions:
        sm.add_transition(transition)
    
    # Create agent
    agent = StateMachineAgent(sm)
    
    return agent


def demonstrate_state_machine_agent():
    """
    Demonstrates state machine agent with conversation flow.
    """
    print("=" * 80)
    print("STATE MACHINE AGENT DEMONSTRATION")
    print("=" * 80)
    
    # Create conversation agent
    agent = create_conversation_agent()
    
    # Test 1: Start and follow conversation flow
    print("\n" + "=" * 80)
    print("Test 1: Conversation Flow")
    print("=" * 80)
    
    # Start agent
    agent.start()
    
    # Simulate conversation
    conversation = [
        "Hello, I need help with something",
        "I want to learn about Python programming",
        "I've provided the information",
        "That's helpful, thank you!",
        "Goodbye"
    ]
    
    for user_input in conversation:
        print(f"\nUser: {user_input}")
        
        # Get status before
        status = agent.get_status()
        print(f"Current State: {status['current_state']}")
        print(f"Available Events: {', '.join(status['available_events'])}")
        
        # Process input
        result = agent.process_input(user_input)
        
        if result.get("success"):
            print(f"Event Triggered: {result['event']}")
            print(f"Transitioned: {result['transitioned']}")
            print(f"New State: {result['new_state']}")
            print(f"Agent: {result['response']}")
        else:
            print(f"Error: {result.get('error')}")
    
    # Test 2: Show state history
    print("\n" + "=" * 80)
    print("Test 2: State History")
    print("=" * 80)
    
    history = agent.state_machine.get_state_history()
    print(f"\nState transitions: {' → '.join(history)}")
    
    # Test 3: Event log
    print("\n" + "=" * 80)
    print("Test 3: Event Log")
    print("=" * 80)
    
    print(f"\nTotal events: {len(agent.state_machine.context.event_log)}")
    for i, event in enumerate(agent.state_machine.context.event_log, 1):
        print(f"{i}. {event.name} ({event.type.value}) at {event.timestamp.strftime('%H:%M:%S')}")
    
    # Test 4: Create different state machine (order processing)
    print("\n" + "=" * 80)
    print("Test 4: Order Processing State Machine")
    print("=" * 80)
    
    # Create order processing state machine
    order_sm = StateMachine(initial_state="cart")
    
    order_states = [
        State("cart", "Shopping cart - adding items", allowed_events={"checkout", "cancel"}),
        State("payment", "Payment processing", allowed_events={"pay", "cancel"}),
        State("fulfillment", "Order fulfillment", allowed_events={"ship", "cancel"}),
        State("shipped", "Order shipped", allowed_events={"deliver", "return"}),
        State("completed", "Order completed", allowed_events={"review"}),
        State("cancelled", "Order cancelled", allowed_events={"restart"})
    ]
    
    for state in order_states:
        order_sm.add_state(state)
    
    order_transitions = [
        Transition("cart", "payment", "checkout"),
        Transition("cart", "cancelled", "cancel"),
        Transition("payment", "fulfillment", "pay"),
        Transition("payment", "cancelled", "cancel"),
        Transition("fulfillment", "shipped", "ship"),
        Transition("fulfillment", "cancelled", "cancel"),
        Transition("shipped", "completed", "deliver"),
        Transition("shipped", "cancelled", "return"),
        Transition("completed", "cart", "review"),
        Transition("cancelled", "cart", "restart")
    ]
    
    for transition in order_transitions:
        order_sm.add_transition(transition)
    
    # Start and simulate order flow
    order_sm.start()
    print(f"Starting state: {order_sm.current_state}")
    
    order_events = [
        ("checkout", "Proceed to checkout"),
        ("pay", "Payment successful"),
        ("ship", "Order shipped"),
        ("deliver", "Order delivered")
    ]
    
    for event_name, description in order_events:
        event = Event(EventType.INTERNAL, event_name)
        success = order_sm.handle_event(event)
        print(f"✓ {description} → {order_sm.current_state}")
    
    print(f"\nFinal state: {order_sm.current_state}")
    print(f"Order flow: {' → '.join(order_sm.get_state_history())}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
State Machine Agent provides:
✓ Explicit state definitions
✓ Event-driven transitions
✓ Guard conditions for validation
✓ Entry/exit actions
✓ Context preservation across states
✓ Deterministic behavior

This pattern excels at:
- Workflow automation with clear phases
- Conversation management with distinct stages
- Process orchestration
- Game AI with discrete states
- Compliance and auditing requirements

Key components:
1. States: Discrete conditions with specific behaviors
2. Transitions: Rules for moving between states
3. Events: Triggers that cause transitions
4. Guards: Conditions that must be met
5. Actions: Operations during transitions
6. Context: Data maintained across states

Advantages:
- Predictable behavior
- Easy to visualize and debug
- Clear state tracking
- Audit trail through history
- Testable transitions
- Integration with workflow tools

State machine patterns:
- Linear: greeting → processing → farewell
- Branching: processing → success | error
- Cyclic: processing → review → processing
- Hierarchical: nested sub-state machines

Use state machine agents when you need:
- Deterministic, predictable behavior
- Clear workflow phases
- Audit trails and compliance
- Visual workflow representation
- Easy testing and debugging
- Formal verification of behavior
""")


if __name__ == "__main__":
    demonstrate_state_machine_agent()
