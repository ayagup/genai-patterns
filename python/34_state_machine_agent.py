"""
State Machine Agent Pattern
Agent behavior defined by explicit states and transitions
"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
class State(Enum):
    """Possible agent states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"
@dataclass
class Transition:
    """State transition definition"""
    from_state: State
    to_state: State
    condition: Callable[[Dict[str, Any]], bool]
    action: Optional[Callable[[Dict[str, Any]], None]] = None
    name: str = ""
@dataclass
class StateEntry:
    """Log entry for state history"""
    state: State
    timestamp: datetime
    duration: float = 0.0
    metadata: Dict[str, Any] = None
class StateMachineAgent:
    """Agent that operates as a state machine"""
    def __init__(self, initial_state: State = State.IDLE):
        self.current_state = initial_state
        self.transitions: List[Transition] = []
        self.state_history: List[StateEntry] = []
        self.context: Dict[str, Any] = {}
        self.state_enter_callbacks: Dict[State, List[Callable]] = {}
        self.state_exit_callbacks: Dict[State, List[Callable]] = {}
        # Record initial state
        self._record_state_entry(initial_state)
    def add_transition(self, transition: Transition):
        """Add a state transition"""
        self.transitions.append(transition)
        print(f"Added transition: {transition.from_state.value} -> {transition.to_state.value}")
    def on_enter_state(self, state: State, callback: Callable):
        """Register callback for entering a state"""
        if state not in self.state_enter_callbacks:
            self.state_enter_callbacks[state] = []
        self.state_enter_callbacks[state].append(callback)
    def on_exit_state(self, state: State, callback: Callable):
        """Register callback for exiting a state"""
        if state not in self.state_exit_callbacks:
            self.state_exit_callbacks[state] = []
        self.state_exit_callbacks[state].append(callback)
    def transition_to(self, new_state: State, metadata: Dict[str, Any] = None):
        """Manually transition to a new state"""
        if new_state == self.current_state:
            return
        print(f"\n{'='*60}")
        print(f"STATE TRANSITION: {self.current_state.value} -> {new_state.value}")
        print(f"{'='*60}")
        # Call exit callbacks
        if self.current_state in self.state_exit_callbacks:
            for callback in self.state_exit_callbacks[self.current_state]:
                callback(self.context)
        # Update state history
        if self.state_history:
            last_entry = self.state_history[-1]
            last_entry.duration = (datetime.now() - last_entry.timestamp).total_seconds()
        old_state = self.current_state
        self.current_state = new_state
        # Record new state
        self._record_state_entry(new_state, metadata)
        # Call enter callbacks
        if new_state in self.state_enter_callbacks:
            for callback in self.state_enter_callbacks[new_state]:
                callback(self.context)
        print(f"Transition complete: {old_state.value} -> {new_state.value}")
    def _record_state_entry(self, state: State, metadata: Dict[str, Any] = None):
        """Record state entry in history"""
        entry = StateEntry(
            state=state,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.state_history.append(entry)
    def update(self, event: Dict[str, Any] = None):
        """Update state machine (check for transitions)"""
        if event:
            self.context.update(event)
        # Check all transitions from current state
        for transition in self.transitions:
            if transition.from_state != self.current_state:
                continue
            # Check if transition condition is met
            if transition.condition(self.context):
                # Execute transition action if defined
                if transition.action:
                    print(f"Executing transition action: {transition.name}")
                    transition.action(self.context)
                # Perform transition
                self.transition_to(transition.to_state, {"trigger": transition.name})
                return True
        return False
    def run_cycle(self, max_iterations: int = 10):
        """Run state machine for multiple update cycles"""
        print(f"\n{'='*70}")
        print(f"RUNNING STATE MACHINE")
        print(f"{'='*70}")
        print(f"Initial State: {self.current_state.value}")
        print(f"Max Iterations: {max_iterations}\n")
        for i in range(max_iterations):
            print(f"\n--- Cycle {i + 1} ---")
            print(f"Current State: {self.current_state.value}")
            # Update state machine
            transitioned = self.update()
            if not transitioned:
                print(f"No transition occurred")
            # Check if we've reached a terminal state
            if self.current_state in [State.COMPLETED, State.ERROR]:
                print(f"\nReached terminal state: {self.current_state.value}")
                break
            # Simulate some work in current state
            import time
            time.sleep(0.1)
        self._print_summary()
    def _print_summary(self):
        """Print execution summary"""
        print(f"\n{'='*70}")
        print(f"STATE MACHINE SUMMARY")
        print(f"{'='*70}")
        print(f"Final State: {self.current_state.value}")
        print(f"Total States Visited: {len(self.state_history)}")
        print(f"\nState History:")
        for i, entry in enumerate(self.state_history, 1):
            duration_str = f"{entry.duration:.2f}s" if entry.duration > 0 else "current"
            print(f"  {i}. {entry.state.value} - {duration_str}")
            if entry.metadata:
                print(f"     Metadata: {entry.metadata}")
        # Calculate time in each state
        state_durations: Dict[State, float] = {}
        for entry in self.state_history:
            state_durations[entry.state] = state_durations.get(entry.state, 0) + entry.duration
        if state_durations:
            print(f"\nTime in Each State:")
            for state, duration in state_durations.items():
                print(f"  {state.value}: {duration:.2f}s")
# Example: Task Processing Agent
class TaskProcessingAgent(StateMachineAgent):
    """Agent that processes tasks through defined states"""
    def __init__(self):
        super().__init__(initial_state=State.IDLE)
        self._setup_state_machine()
    def _setup_state_machine(self):
        """Setup state transitions and callbacks"""
        # Define transitions
        # IDLE -> LISTENING when task received
        self.add_transition(Transition(
            from_state=State.IDLE,
            to_state=State.LISTENING,
            condition=lambda ctx: ctx.get('task_received', False),
            action=lambda ctx: print("  Action: Preparing to listen"),
            name="task_received"
        ))
        # LISTENING -> PROCESSING when input complete
        self.add_transition(Transition(
            from_state=State.LISTENING,
            to_state=State.PROCESSING,
            condition=lambda ctx: ctx.get('input_complete', False),
            action=lambda ctx: print("  Action: Starting processing"),
            name="input_complete"
        ))
        # PROCESSING -> EXECUTING when ready to execute
        self.add_transition(Transition(
            from_state=State.PROCESSING,
            to_state=State.EXECUTING,
            condition=lambda ctx: ctx.get('ready_to_execute', False),
            action=lambda ctx: print("  Action: Beginning execution"),
            name="ready_to_execute"
        ))
        # EXECUTING -> WAITING if needs external resource
        self.add_transition(Transition(
            from_state=State.EXECUTING,
            to_state=State.WAITING,
            condition=lambda ctx: ctx.get('needs_resource', False),
            action=lambda ctx: print("  Action: Requesting external resource"),
            name="needs_resource"
        ))
        # WAITING -> EXECUTING when resource available
        self.add_transition(Transition(
            from_state=State.WAITING,
            to_state=State.EXECUTING,
            condition=lambda ctx: ctx.get('resource_available', False),
            action=lambda ctx: print("  Action: Resuming execution"),
            name="resource_available"
        ))
        # EXECUTING -> COMPLETED when done
        self.add_transition(Transition(
            from_state=State.EXECUTING,
            to_state=State.COMPLETED,
            condition=lambda ctx: ctx.get('execution_complete', False),
            action=lambda ctx: print("  Action: Finalizing results"),
            name="execution_complete"
        ))
        # Any state -> ERROR on error
        for state in State:
            if state != State.ERROR:
                self.add_transition(Transition(
                    from_state=state,
                    to_state=State.ERROR,
                    condition=lambda ctx: ctx.get('error_occurred', False),
                    action=lambda ctx: print(f"  Action: Handling error - {ctx.get('error_message')}"),
                    name="error_occurred"
                ))
        # Register state callbacks
        self.on_enter_state(State.LISTENING, lambda ctx: print("  → Entered LISTENING state"))
        self.on_enter_state(State.PROCESSING, lambda ctx: print("  → Entered PROCESSING state"))
        self.on_enter_state(State.EXECUTING, lambda ctx: print("  → Entered EXECUTING state"))
        self.on_enter_state(State.COMPLETED, lambda ctx: print("  → Entered COMPLETED state"))
        self.on_exit_state(State.IDLE, lambda ctx: print("  ← Exiting IDLE state"))
    def process_task(self, task: str):
        """Process a task through the state machine"""
        print(f"\n{'='*70}")
        print(f"PROCESSING TASK: {task}")
        print(f"{'='*70}")
        # Simulate task processing with events
        events = [
            {'task_received': True, 'task_name': task},
            {'input_complete': True},
            {'ready_to_execute': True},
            {'needs_resource': True},
            {'resource_available': True},
            {'execution_complete': True}
        ]
        for i, event in enumerate(events):
            print(f"\n--- Event {i + 1}: {list(event.keys())[0]} ---")
            self.update(event)
            # Clear event flags for next iteration
            self.context = {k: v for k, v in self.context.items() 
                           if k in ['task_name']}
            import time
            time.sleep(0.2)
# Usage
if __name__ == "__main__":
    # Create task processing agent
    agent = TaskProcessingAgent()
    # Process a task
    agent.process_task("Analyze customer data")
    # Print final summary
    print("\n" + "="*80)
    # Example with error handling
    print("\n\n" + "="*80)
    print("EXAMPLE WITH ERROR")
    print("="*80)
    agent2 = TaskProcessingAgent()
    agent2.update({'task_received': True})
    agent2.update({'input_complete': True})
    agent2.update({'error_occurred': True, 'error_message': 'Network timeout'})
