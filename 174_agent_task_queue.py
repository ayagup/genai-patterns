"""
Agent State Machine Pattern

Implements finite state machines for agent behavior.
Manages state transitions, actions, and event handling.

Use Cases:
- Workflow management
- Behavior modeling
- Game AI
- Process control

Advantages:
- Clear state management
- Predictable transitions
- Easy debugging
- Formal verification
"""

from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class TransitionType(Enum):
    """Types of state transitions"""
    AUTOMATIC = "automatic"
    CONDITIONAL = "conditional"
    EVENT_DRIVEN = "event_driven"


@dataclass
class State:
    """State in state machine"""
    state_id: str
    name: str
    is_initial: bool = False
    is_final: bool = False
    on_enter: Optional[Callable] = None
    on_exit: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    """Transition between states"""
    transition_id: str
    from_state: str
    to_state: str
    event: Optional[str] = None
    condition: Optional[Callable] = None
    action: Optional[Callable] = None
    transition_type: TransitionType = TransitionType.EVENT_DRIVEN
    priority: int = 0


@dataclass
class StateChangeEvent:
    """Event representing state change"""
    event_id: str
    from_state: str
    to_state: str
    timestamp: datetime
    trigger: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class StateMachine:
    """Basic finite state machine"""
    
    def __init__(self, machine_id: str):
        self.machine_id = machine_id
        self.states: Dict[str, State] = {}
        self.transitions: List[Transition] = []
        self.current_state: Optional[str] = None
        self.context: Dict[str, Any] = {}
        self.history: List[StateChangeEvent] = []
        self.event_counter = 0
    
    def add_state(self, state: State) -> None:
        """
        Add state to machine.
        
        Args:
            state: State to add
        """
        self.states[state.state_id] = state
        
        if state.is_initial and self.current_state is None:
            self.current_state = state.state_id
            
            # Call on_enter
            if state.on_enter:
                state.on_enter(self.context)
    
    def add_transition(self, transition: Transition) -> None:
        """
        Add transition to machine.
        
        Args:
            transition: Transition to add
        """
        self.transitions.append(transition)
        
        # Sort by priority
        self.transitions.sort(key=lambda t: t.priority, reverse=True)
    
    def trigger_event(self, event: str, **kwargs) -> bool:
        """
        Trigger event to potentially cause state transition.
        
        Args:
            event: Event name
            **kwargs: Additional context
            
        Returns:
            Whether transition occurred
        """
        if not self.current_state:
            return False
        
        # Update context
        self.context.update(kwargs)
        
        # Find matching transition
        for transition in self.transitions:
            if transition.from_state != self.current_state:
                continue
            
            # Check event
            if transition.event and transition.event != event:
                continue
            
            # Check condition
            if transition.condition:
                try:
                    if not transition.condition(self.context):
                        continue
                except Exception:
                    continue
            
            # Execute transition
            return self._execute_transition(transition, event)
        
        return False
    
    def _execute_transition(self,
                           transition: Transition,
                           trigger: Optional[str] = None) -> bool:
        """Execute state transition"""
        from_state = self.states.get(self.current_state)
        to_state = self.states.get(transition.to_state)
        
        if not from_state or not to_state:
            return False
        
        # Call on_exit
        if from_state.on_exit:
            try:
                from_state.on_exit(self.context)
            except Exception:
                pass
        
        # Execute transition action
        if transition.action:
            try:
                transition.action(self.context)
            except Exception:
                pass
        
        # Record event
        event = StateChangeEvent(
            event_id="event_{}".format(self.event_counter),
            from_state=self.current_state,
            to_state=transition.to_state,
            timestamp=datetime.now(),
            trigger=trigger,
            context=dict(self.context)
        )
        
        self.event_counter += 1
        self.history.append(event)
        
        # Change state
        self.current_state = transition.to_state
        
        # Call on_enter
        if to_state.on_enter:
            try:
                to_state.on_enter(self.context)
            except Exception:
                pass
        
        # Check automatic transitions
        self._check_automatic_transitions()
        
        return True
    
    def _check_automatic_transitions(self) -> None:
        """Check for automatic transitions"""
        for transition in self.transitions:
            if transition.transition_type != TransitionType.AUTOMATIC:
                continue
            
            if transition.from_state != self.current_state:
                continue
            
            # Check condition
            if transition.condition:
                try:
                    if transition.condition(self.context):
                        self._execute_transition(transition)
                        break
                except Exception:
                    pass
            else:
                self._execute_transition(transition)
                break
    
    def get_current_state(self) -> Optional[State]:
        """Get current state"""
        if self.current_state:
            return self.states.get(self.current_state)
        return None
    
    def is_in_final_state(self) -> bool:
        """Check if in final state"""
        state = self.get_current_state()
        return state.is_final if state else False
    
    def reset(self) -> None:
        """Reset to initial state"""
        # Find initial state
        for state in self.states.values():
            if state.is_initial:
                self.current_state = state.state_id
                self.context = {}
                self.history = []
                
                if state.on_enter:
                    state.on_enter(self.context)
                
                break


class HierarchicalStateMachine(StateMachine):
    """State machine with hierarchical states"""
    
    def __init__(self, machine_id: str):
        super().__init__(machine_id)
        self.substates: Dict[str, StateMachine] = {}
    
    def add_substate_machine(self,
                            parent_state: str,
                            submachine: StateMachine) -> None:
        """
        Add sub-state machine to a state.
        
        Args:
            parent_state: Parent state ID
            submachine: Sub-state machine
        """
        self.substates[parent_state] = submachine
    
    def trigger_event(self, event: str, **kwargs) -> bool:
        """Trigger event, checking substates first"""
        # Check if current state has substate machine
        if self.current_state in self.substates:
            submachine = self.substates[self.current_state]
            
            # Try substate first
            if submachine.trigger_event(event, **kwargs):
                return True
        
        # Try parent state machine
        return super().trigger_event(event, **kwargs)


class AgentStateMachineManager:
    """
    Manages state machines for agents.
    Supports multiple state machines per agent and coordination.
    """
    
    def __init__(self):
        self.machines: Dict[str, StateMachine] = {}
        self.agent_machines: Dict[str, List[str]] = {}
    
    def create_machine(self,
                      machine_id: str,
                      agent_id: str,
                      hierarchical: bool = False) -> StateMachine:
        """
        Create state machine for agent.
        
        Args:
            machine_id: Machine identifier
            agent_id: Agent identifier
            hierarchical: Whether to use hierarchical machine
            
        Returns:
            Created state machine
        """
        if hierarchical:
            machine = HierarchicalStateMachine(machine_id)
        else:
            machine = StateMachine(machine_id)
        
        self.machines[machine_id] = machine
        
        if agent_id not in self.agent_machines:
            self.agent_machines[agent_id] = []
        self.agent_machines[agent_id].append(machine_id)
        
        return machine
    
    def get_machine(self, machine_id: str) -> Optional[StateMachine]:
        """Get state machine by ID"""
        return self.machines.get(machine_id)
    
    def get_agent_machines(self, agent_id: str) -> List[StateMachine]:
        """Get all state machines for agent"""
        machine_ids = self.agent_machines.get(agent_id, [])
        return [
            self.machines[mid] for mid in machine_ids
            if mid in self.machines
        ]
    
    def trigger_event_all(self,
                         agent_id: str,
                         event: str,
                         **kwargs) -> int:
        """
        Trigger event on all machines for agent.
        
        Args:
            agent_id: Agent identifier
            event: Event name
            **kwargs: Event context
            
        Returns:
            Number of transitions
        """
        count = 0
        
        for machine in self.get_agent_machines(agent_id):
            if machine.trigger_event(event, **kwargs):
                count += 1
        
        return count
    
    def get_agent_states(self, agent_id: str) -> Dict[str, str]:
        """Get current states of all machines for agent"""
        states = {}
        
        for machine in self.get_agent_machines(agent_id):
            current = machine.get_current_state()
            if current:
                states[machine.machine_id] = current.name
        
        return states
    
    def export_machine_definition(self, machine_id: str) -> Dict[str, Any]:
        """Export machine definition as JSON-serializable dict"""
        machine = self.machines.get(machine_id)
        
        if not machine:
            return {}
        
        return {
            "machine_id": machine.machine_id,
            "current_state": machine.current_state,
            "states": [
                {
                    "state_id": s.state_id,
                    "name": s.name,
                    "is_initial": s.is_initial,
                    "is_final": s.is_final
                }
                for s in machine.states.values()
            ],
            "transitions": [
                {
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "event": t.event,
                    "type": t.transition_type.value
                }
                for t in machine.transitions
            ],
            "history": [
                {
                    "from_state": e.from_state,
                    "to_state": e.to_state,
                    "timestamp": e.timestamp.isoformat(),
                    "trigger": e.trigger
                }
                for e in machine.history[-10:]  # Last 10 events
            ]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        total_machines = len(self.machines)
        total_agents = len(self.agent_machines)
        
        total_states = sum(
            len(m.states) for m in self.machines.values()
        )
        
        total_transitions = sum(
            len(m.transitions) for m in self.machines.values()
        )
        
        return {
            "total_machines": total_machines,
            "total_agents": total_agents,
            "total_states": total_states,
            "total_transitions": total_transitions
        }


def demonstrate_state_machine():
    """Demonstrate agent state machine"""
    print("=" * 70)
    print("Agent State Machine Demonstration")
    print("=" * 70)
    
    manager = AgentStateMachineManager()
    
    # Example 1: Create simple state machine
    print("\n1. Creating State Machine:")
    
    machine = manager.create_machine("task_processor", "agent_1")
    
    # Define states
    idle = State("idle", "Idle", is_initial=True)
    processing = State("processing", "Processing")
    completed = State("completed", "Completed", is_final=True)
    failed = State("failed", "Failed", is_final=True)
    
    # Add callback functions
    def on_enter_processing(context):
        print("  Started processing task: {}".format(context.get("task_id")))
    
    processing.on_enter = on_enter_processing
    
    machine.add_state(idle)
    machine.add_state(processing)
    machine.add_state(completed)
    machine.add_state(failed)
    
    print("  Added 4 states")
    
    # Example 2: Add transitions
    print("\n2. Adding Transitions:")
    
    # Idle -> Processing
    machine.add_transition(Transition(
        transition_id="start_processing",
        from_state="idle",
        to_state="processing",
        event="start_task"
    ))
    
    # Processing -> Completed
    machine.add_transition(Transition(
        transition_id="complete",
        from_state="processing",
        to_state="completed",
        event="task_complete"
    ))
    
    # Processing -> Failed
    machine.add_transition(Transition(
        transition_id="fail",
        from_state="processing",
        to_state="failed",
        event="task_failed"
    ))
    
    # Completed/Failed -> Idle
    machine.add_transition(Transition(
        transition_id="reset_completed",
        from_state="completed",
        to_state="idle",
        event="reset"
    ))
    
    machine.add_transition(Transition(
        transition_id="reset_failed",
        from_state="failed",
        to_state="idle",
        event="reset"
    ))
    
    print("  Added 5 transitions")
    
    # Example 3: Trigger events
    print("\n3. Triggering State Transitions:")
    
    print("  Initial state: {}".format(machine.get_current_state().name))
    
    # Start task
    success = machine.trigger_event("start_task", task_id="task_001")
    print("  Triggered 'start_task': {}".format(success))
    print("  Current state: {}".format(machine.get_current_state().name))
    
    # Complete task
    success = machine.trigger_event("task_complete")
    print("  Triggered 'task_complete': {}".format(success))
    print("  Current state: {}".format(machine.get_current_state().name))
    print("  Is final: {}".format(machine.is_in_final_state()))
    
    # Example 4: State machine with conditions
    print("\n4. Conditional Transitions:")
    
    # Reset machine
    machine.trigger_event("reset")
    
    # Add conditional transition
    def check_priority(context):
        return context.get("priority", 0) > 5
    
    machine.add_transition(Transition(
        transition_id="priority_fast_track",
        from_state="idle",
        to_state="completed",
        event="start_task",
        condition=check_priority,
        priority=10  # Higher priority than normal transition
    ))
    
    # Trigger with high priority
    machine.trigger_event("start_task", priority=8)
    print("  High priority task -> {}".format(machine.get_current_state().name))
    
    # Example 5: Hierarchical state machine
    print("\n5. Hierarchical State Machine:")
    
    hmachine = manager.create_machine(
        "complex_workflow",
        "agent_2",
        hierarchical=True
    )
    
    # Parent states
    active = State("active", "Active", is_initial=True)
    inactive = State("inactive", "Inactive")
    
    hmachine.add_state(active)
    hmachine.add_state(inactive)
    
    # Create substates for "active"
    subm = StateMachine("active_substates")
    sub_ready = State("ready", "Ready", is_initial=True)
    sub_busy = State("busy", "Busy")
    
    subm.add_state(sub_ready)
    subm.add_state(sub_busy)
    
    subm.add_transition(Transition(
        transition_id="start_work",
        from_state="ready",
        to_state="busy",
        event="work_started"
    ))
    
    hmachine.add_substate_machine("active", subm)
    
    print("  Created hierarchical state machine")
    print("  Parent state: {}".format(hmachine.get_current_state().name))
    
    # Trigger substate event
    hmachine.trigger_event("work_started")
    print("  Substate: {}".format(subm.get_current_state().name))
    
    # Example 6: Multiple machines per agent
    print("\n6. Multiple State Machines:")
    
    # Create another machine for agent_1
    behavior_machine = manager.create_machine("behavior", "agent_1")
    
    calm = State("calm", "Calm", is_initial=True)
    alert = State("alert", "Alert")
    
    behavior_machine.add_state(calm)
    behavior_machine.add_state(alert)
    
    behavior_machine.add_transition(Transition(
        transition_id="detect_threat",
        from_state="calm",
        to_state="alert",
        event="threat_detected"
    ))
    
    # Get all machines for agent
    agent_machines = manager.get_agent_machines("agent_1")
    print("  Agent_1 has {} state machines".format(len(agent_machines)))
    
    # Example 7: Trigger event on all machines
    print("\n7. Broadcasting Events:")
    
    count = manager.trigger_event_all(
        "agent_1",
        "threat_detected"
    )
    
    print("  Event triggered {} transitions".format(count))
    
    states = manager.get_agent_states("agent_1")
    print("  Agent_1 states: {}".format(states))
    
    # Example 8: Export machine definition
    print("\n8. Exporting Machine Definition:")
    
    definition = manager.export_machine_definition("task_processor")
    print(json.dumps(definition, indent=2))
    
    # Example 9: State history
    print("\n9. State Transition History:")
    
    print("  task_processor history:")
    for event in machine.history[-5:]:
        print("    {} -> {} ({})".format(
            event.from_state,
            event.to_state,
            event.trigger or "automatic"
        ))
    
    # Example 10: Statistics
    print("\n10. State Machine Statistics:")
    stats = manager.get_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    demonstrate_state_machine()
