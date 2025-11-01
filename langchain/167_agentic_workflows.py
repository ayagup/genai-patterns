"""
Pattern 167: Agentic Workflows

Description:
    The Agentic Workflows pattern implements complex, multi-step processes where agents
    autonomously navigate through workflow states, make decisions, handle branching logic,
    and coordinate multiple tasks. It uses state machines and graph-based execution for
    sophisticated task orchestration.

Components:
    1. Workflow Definition: Graph structure with states and transitions
    2. State Manager: Tracks current workflow state
    3. Decision Engine: Makes branching decisions
    4. Task Executor: Executes tasks at each state
    5. Condition Evaluator: Evaluates transition conditions
    6. Error Handler: Manages workflow failures

Use Cases:
    - Multi-step business processes
    - Complex data pipelines
    - Decision trees and workflows
    - Approval workflows
    - Multi-stage content generation
    - Automated testing workflows

Benefits:
    - Complex process automation
    - Clear workflow visualization
    - Conditional branching support
    - State persistence
    - Error recovery
    - Audit trail

Trade-offs:
    - Increased complexity
    - State management overhead
    - Debugging challenges
    - Performance considerations
    - Requires workflow design

LangChain Implementation:
    Uses LangGraph-inspired state machines with LangChain agents for task execution.
    Implements graph-based workflows with conditional branching and state management.
"""

import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class WorkflowStatus(Enum):
    """Status of workflow execution"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class TransitionType(Enum):
    """Types of state transitions"""
    AUTOMATIC = "automatic"
    CONDITIONAL = "conditional"
    MANUAL = "manual"


@dataclass
class WorkflowState:
    """Represents a state in the workflow"""
    name: str
    description: str
    task: Optional[Callable] = None
    is_terminal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateTransition:
    """Represents a transition between states"""
    from_state: str
    to_state: str
    condition: Optional[Callable] = None
    transition_type: TransitionType = TransitionType.AUTOMATIC
    description: str = ""


@dataclass
class WorkflowContext:
    """Context passed through workflow execution"""
    data: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_result(self, state: str, result: Any):
        """Add result from a state execution"""
        self.data[f"{state}_result"] = result
        self.history.append(state)
    
    def get_result(self, state: str) -> Any:
        """Get result from previous state"""
        return self.data.get(f"{state}_result")


class AgenticWorkflow:
    """Implements agentic workflow pattern with state machines"""
    
    def __init__(self, name: str):
        """Initialize workflow"""
        self.name = name
        self.states: Dict[str, WorkflowState] = {}
        self.transitions: List[StateTransition] = []
        self.start_state: Optional[str] = None
        self.current_state: Optional[str] = None
        self.status = WorkflowStatus.NOT_STARTED
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        # Execution log
        self.execution_log: List[Dict[str, Any]] = []
    
    def add_state(self, state: WorkflowState, is_start: bool = False):
        """Add a state to the workflow"""
        self.states[state.name] = state
        if is_start:
            self.start_state = state.name
    
    def add_transition(self, transition: StateTransition):
        """Add a transition between states"""
        if transition.from_state not in self.states:
            raise ValueError(f"Unknown from_state: {transition.from_state}")
        if transition.to_state not in self.states:
            raise ValueError(f"Unknown to_state: {transition.to_state}")
        self.transitions.append(transition)
    
    def execute(self, initial_context: Optional[WorkflowContext] = None) -> WorkflowContext:
        """Execute the workflow"""
        if not self.start_state:
            raise ValueError("No start state defined")
        
        context = initial_context or WorkflowContext()
        self.current_state = self.start_state
        self.status = WorkflowStatus.IN_PROGRESS
        
        self._log_execution("workflow_started", {
            "start_state": self.start_state,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            while True:
                state = self.states[self.current_state]
                
                self._log_execution("state_entered", {
                    "state": self.current_state,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Execute state task
                if state.task:
                    try:
                        result = state.task(context, self.llm)
                        context.add_result(self.current_state, result)
                        
                        self._log_execution("state_executed", {
                            "state": self.current_state,
                            "result": str(result)[:100]
                        })
                    except Exception as e:
                        self._log_execution("state_error", {
                            "state": self.current_state,
                            "error": str(e)
                        })
                        self.status = WorkflowStatus.FAILED
                        raise
                
                # Check if terminal state
                if state.is_terminal:
                    self.status = WorkflowStatus.COMPLETED
                    self._log_execution("workflow_completed", {
                        "final_state": self.current_state,
                        "timestamp": datetime.now().isoformat()
                    })
                    break
                
                # Find next state
                next_state = self._get_next_state(context)
                
                if next_state is None:
                    self.status = WorkflowStatus.FAILED
                    raise ValueError(f"No valid transition from state: {self.current_state}")
                
                self._log_execution("transition", {
                    "from": self.current_state,
                    "to": next_state,
                    "timestamp": datetime.now().isoformat()
                })
                
                self.current_state = next_state
        
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            self._log_execution("workflow_failed", {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            raise
        
        return context
    
    def _get_next_state(self, context: WorkflowContext) -> Optional[str]:
        """Determine next state based on transitions"""
        # Find applicable transitions from current state
        applicable = [
            t for t in self.transitions 
            if t.from_state == self.current_state
        ]
        
        if not applicable:
            return None
        
        # Evaluate conditions for conditional transitions
        for transition in applicable:
            if transition.transition_type == TransitionType.AUTOMATIC:
                return transition.to_state
            elif transition.transition_type == TransitionType.CONDITIONAL:
                if transition.condition and transition.condition(context):
                    return transition.to_state
        
        # If no condition matched, try first automatic transition
        automatic = [t for t in applicable if t.transition_type == TransitionType.AUTOMATIC]
        if automatic:
            return automatic[0].to_state
        
        return None
    
    def _log_execution(self, event_type: str, data: Dict[str, Any]):
        """Log workflow execution event"""
        self.execution_log.append({
            "event": event_type,
            "data": data
        })
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of workflow execution"""
        return {
            "workflow_name": self.name,
            "status": self.status.value,
            "states_visited": len(set(e["data"].get("state", "") for e in self.execution_log)),
            "total_events": len(self.execution_log),
            "current_state": self.current_state
        }
    
    def visualize(self) -> str:
        """Create text visualization of workflow"""
        lines = [f"Workflow: {self.name}", "=" * 60, ""]
        
        lines.append("States:")
        for name, state in self.states.items():
            marker = "→" if name == self.start_state else " "
            terminal = " (terminal)" if state.is_terminal else ""
            lines.append(f"  {marker} {name}{terminal}: {state.description}")
        
        lines.append("\nTransitions:")
        for trans in self.transitions:
            cond = f" [conditional]" if trans.transition_type == TransitionType.CONDITIONAL else ""
            lines.append(f"  {trans.from_state} -> {trans.to_state}{cond}")
            if trans.description:
                lines.append(f"    {trans.description}")
        
        return "\n".join(lines)


def demonstrate_agentic_workflows():
    """Demonstrate agentic workflows"""
    print("=" * 80)
    print("AGENTIC WORKFLOWS PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Content generation workflow
    print("\n" + "=" * 80)
    print("Example 1: Content Generation Workflow")
    print("=" * 80)
    
    def research_task(context: WorkflowContext, llm: ChatOpenAI) -> str:
        """Research phase"""
        topic = context.data.get("topic", "AI")
        prompt = f"List 3 key points about {topic} for a blog post"
        chain = ChatPromptTemplate.from_messages([
            ("user", prompt)
        ]) | llm | StrOutputParser()
        return chain.invoke({})
    
    def outline_task(context: WorkflowContext, llm: ChatOpenAI) -> str:
        """Outline creation"""
        research = context.get_result("research")
        prompt = f"Create a blog post outline based on: {research}"
        chain = ChatPromptTemplate.from_messages([
            ("user", prompt)
        ]) | llm | StrOutputParser()
        return chain.invoke({})
    
    def write_task(context: WorkflowContext, llm: ChatOpenAI) -> str:
        """Writing phase"""
        outline = context.get_result("outline")
        prompt = f"Write the first paragraph based on: {outline}"
        chain = ChatPromptTemplate.from_messages([
            ("user", prompt)
        ]) | llm | StrOutputParser()
        return chain.invoke({})
    
    def review_task(context: WorkflowContext, llm: ChatOpenAI) -> str:
        """Review phase"""
        draft = context.get_result("write")
        prompt = f"Review and provide feedback on: {draft}"
        chain = ChatPromptTemplate.from_messages([
            ("user", prompt)
        ]) | llm | StrOutputParser()
        return chain.invoke({})
    
    # Create workflow
    content_workflow = AgenticWorkflow("Content Generation")
    
    # Add states
    content_workflow.add_state(WorkflowState(
        "research",
        "Research topic and gather key points",
        task=research_task
    ), is_start=True)
    
    content_workflow.add_state(WorkflowState(
        "outline",
        "Create content outline",
        task=outline_task
    ))
    
    content_workflow.add_state(WorkflowState(
        "write",
        "Write content draft",
        task=write_task
    ))
    
    content_workflow.add_state(WorkflowState(
        "review",
        "Review and provide feedback",
        task=review_task
    ))
    
    content_workflow.add_state(WorkflowState(
        "complete",
        "Content generation complete",
        is_terminal=True
    ))
    
    # Add transitions
    content_workflow.add_transition(StateTransition(
        "research", "outline", 
        transition_type=TransitionType.AUTOMATIC
    ))
    content_workflow.add_transition(StateTransition(
        "outline", "write",
        transition_type=TransitionType.AUTOMATIC
    ))
    content_workflow.add_transition(StateTransition(
        "write", "review",
        transition_type=TransitionType.AUTOMATIC
    ))
    content_workflow.add_transition(StateTransition(
        "review", "complete",
        transition_type=TransitionType.AUTOMATIC
    ))
    
    # Visualize workflow
    print("\nWorkflow Structure:")
    print(content_workflow.visualize())
    
    # Execute workflow
    print("\n" + "-" * 60)
    print("Executing Workflow...")
    print("-" * 60)
    
    initial_context = WorkflowContext(data={"topic": "Machine Learning"})
    
    try:
        result_context = content_workflow.execute(initial_context)
        
        print("\nWorkflow Execution Summary:")
        summary = content_workflow.get_execution_summary()
        print(f"  Status: {summary['status']}")
        print(f"  States visited: {summary['states_visited']}")
        print(f"  Total events: {summary['total_events']}")
        
        print("\nState Execution Path:")
        for state in result_context.history:
            print(f"  → {state}")
        
        print("\nFinal Results:")
        print(f"  Research: {result_context.get_result('research')[:80]}...")
        print(f"  Outline: {result_context.get_result('outline')[:80]}...")
        print(f"  Draft: {result_context.get_result('write')[:80]}...")
        print(f"  Review: {result_context.get_result('review')[:80]}...")
        
    except Exception as e:
        print(f"Workflow failed: {e}")
    
    # Example 2: Conditional branching workflow
    print("\n" + "=" * 80)
    print("Example 2: Conditional Branching Workflow")
    print("=" * 80)
    
    def analyze_task(context: WorkflowContext, llm: ChatOpenAI) -> Dict[str, Any]:
        """Analyze input"""
        text = context.data.get("text", "")
        prompt = f"Is this text positive or negative sentiment? Answer with just 'positive' or 'negative': {text}"
        chain = ChatPromptTemplate.from_messages([
            ("user", prompt)
        ]) | llm | StrOutputParser()
        sentiment = chain.invoke({}).strip().lower()
        return {"sentiment": sentiment}
    
    def positive_response_task(context: WorkflowContext, llm: ChatOpenAI) -> str:
        """Handle positive sentiment"""
        return "Thank you for the positive feedback!"
    
    def negative_response_task(context: WorkflowContext, llm: ChatOpenAI) -> str:
        """Handle negative sentiment"""
        return "We're sorry to hear that. How can we improve?"
    
    # Create branching workflow
    branching_workflow = AgenticWorkflow("Sentiment Analysis")
    
    # Add states
    branching_workflow.add_state(WorkflowState(
        "analyze",
        "Analyze sentiment",
        task=analyze_task
    ), is_start=True)
    
    branching_workflow.add_state(WorkflowState(
        "positive_response",
        "Generate positive response",
        task=positive_response_task
    ))
    
    branching_workflow.add_state(WorkflowState(
        "negative_response",
        "Generate negative response",
        task=negative_response_task
    ))
    
    branching_workflow.add_state(WorkflowState(
        "end",
        "End state",
        is_terminal=True
    ))
    
    # Add conditional transitions
    branching_workflow.add_transition(StateTransition(
        "analyze", "positive_response",
        condition=lambda ctx: ctx.get_result("analyze").get("sentiment") == "positive",
        transition_type=TransitionType.CONDITIONAL,
        description="If sentiment is positive"
    ))
    
    branching_workflow.add_transition(StateTransition(
        "analyze", "negative_response",
        condition=lambda ctx: ctx.get_result("analyze").get("sentiment") == "negative",
        transition_type=TransitionType.CONDITIONAL,
        description="If sentiment is negative"
    ))
    
    branching_workflow.add_transition(StateTransition(
        "positive_response", "end",
        transition_type=TransitionType.AUTOMATIC
    ))
    
    branching_workflow.add_transition(StateTransition(
        "negative_response", "end",
        transition_type=TransitionType.AUTOMATIC
    ))
    
    # Visualize
    print("\nWorkflow Structure:")
    print(branching_workflow.visualize())
    
    # Execute with positive text
    print("\n" + "-" * 60)
    print("Test 1: Positive Text")
    print("-" * 60)
    
    context1 = WorkflowContext(data={"text": "This product is amazing! I love it."})
    result1 = branching_workflow.execute(context1)
    
    print(f"Path taken: {' → '.join(result1.history)}")
    print(f"Response: {result1.get_result('positive_response')}")
    
    # Execute with negative text
    print("\n" + "-" * 60)
    print("Test 2: Negative Text")
    print("-" * 60)
    
    branching_workflow.status = WorkflowStatus.NOT_STARTED  # Reset
    context2 = WorkflowContext(data={"text": "This is terrible. Very disappointed."})
    result2 = branching_workflow.execute(context2)
    
    print(f"Path taken: {' → '.join(result2.history)}")
    print(f"Response: {result2.get_result('negative_response')}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Agentic Workflows pattern enables:
✓ Complex multi-step process automation
✓ Conditional branching based on runtime state
✓ State-based execution with context passing
✓ Clear workflow visualization
✓ Execution logging and audit trails
✓ Error handling and recovery
✓ Flexible workflow composition

This pattern is valuable for:
- Business process automation
- Multi-stage content generation
- Decision tree execution
- Approval workflows
- Complex data pipelines
- Testing and validation workflows
- Orchestrating multiple agents
    """)


if __name__ == "__main__":
    demonstrate_agentic_workflows()
