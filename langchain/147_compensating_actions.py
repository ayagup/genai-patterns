"""
Pattern 147: Compensating Actions

Description:
    Implements compensating actions for rolling back completed operations when
    a workflow fails. This pattern maintains consistency by undoing successful
    steps in reverse order, similar to database transaction rollback but for
    complex multi-step agent workflows.

Components:
    - Action Logger: Records completed actions
    - Compensation Registry: Maps actions to compensating actions
    - Rollback Coordinator: Orchestrates compensation execution
    - State Tracker: Maintains workflow state
    - Failure Handler: Detects and triggers compensation
    - Recovery Monitor: Tracks compensation success

Use Cases:
    - Multi-step agent workflows with side effects
    - API orchestration with multiple services
    - Data pipeline error recovery
    - Transaction-like behavior for agent operations
    - Distributed system coordination

Benefits:
    - Maintains system consistency
    - Automatic error recovery
    - Clear rollback semantics
    - Reduced manual cleanup
    - Better fault tolerance

Trade-offs:
    - Complexity overhead
    - Not all actions are reversible
    - Potential partial rollback states
    - Resource consumption during compensation

LangChain Implementation:
    Uses action tracking, compensation functions, and rollback orchestration
    to implement saga-like patterns for agent workflows.
"""

import os
import time
from typing import Any, Callable, Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ActionStatus(Enum):
    """Status of an action"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    COMPENSATION_FAILED = "compensation_failed"


@dataclass
class Action:
    """Represents a workflow action"""
    action_id: str
    name: str
    execute_fn: Callable
    compensate_fn: Optional[Callable] = None
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    status: ActionStatus = ActionStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    executed_at: Optional[datetime] = None
    compensated_at: Optional[datetime] = None
    idempotent: bool = False


@dataclass
class WorkflowExecution:
    """Tracks workflow execution"""
    workflow_id: str
    actions: List[Action]
    current_index: int = 0
    completed_actions: List[Action] = field(default_factory=list)
    failed_action: Optional[Action] = None
    compensation_results: List[Tuple[Action, bool]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    success: bool = False


class CompensatingActionsAgent:
    """
    Agent that implements compensating actions for workflow rollback.
    
    Executes multi-step workflows and automatically compensates completed
    actions when failures occur, maintaining system consistency.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize the compensating actions agent.
        
        Args:
            model: LLM model to use
        """
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        
        self.workflows: Dict[str, WorkflowExecution] = {}
        self.compensation_registry: Dict[str, Callable] = {}
    
    def register_compensation(
        self,
        action_name: str,
        compensate_fn: Callable
    ):
        """
        Register a compensation function for an action type.
        
        Args:
            action_name: Name of the action
            compensate_fn: Function to compensate the action
        """
        self.compensation_registry[action_name] = compensate_fn
    
    def execute_workflow(
        self,
        workflow_id: str,
        actions: List[Action]
    ) -> WorkflowExecution:
        """
        Execute a workflow with compensating action support.
        
        Args:
            workflow_id: Unique workflow identifier
            actions: List of actions to execute
            
        Returns:
            Workflow execution result
        """
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            actions=actions
        )
        
        self.workflows[workflow_id] = execution
        
        print(f"\n🚀 Starting workflow: {workflow_id}")
        print(f"   Total actions: {len(actions)}")
        
        try:
            # Execute actions sequentially
            for i, action in enumerate(actions):
                execution.current_index = i
                
                print(f"\n📍 Executing action {i+1}/{len(actions)}: {action.name}")
                
                action.status = ActionStatus.EXECUTING
                
                try:
                    # Execute the action
                    start_time = time.time()
                    result = action.execute_fn(*action.args, **action.kwargs)
                    duration = time.time() - start_time
                    
                    action.result = result
                    action.status = ActionStatus.COMPLETED
                    action.executed_at = datetime.now()
                    
                    execution.completed_actions.append(action)
                    
                    print(f"   ✅ Completed in {duration:.2f}s")
                    if result:
                        print(f"   Result: {str(result)[:100]}")
                    
                except Exception as e:
                    # Action failed - trigger compensation
                    action.status = ActionStatus.FAILED
                    action.error = str(e)
                    execution.failed_action = action
                    
                    print(f"   ❌ Failed: {e}")
                    print(f"\n🔄 Triggering compensation for {len(execution.completed_actions)} completed actions")
                    
                    self._compensate_actions(execution)
                    
                    execution.end_time = datetime.now()
                    execution.success = False
                    
                    return execution
            
            # All actions completed successfully
            execution.end_time = datetime.now()
            execution.success = True
            
            print(f"\n✅ Workflow completed successfully!")
            
            return execution
            
        except Exception as e:
            print(f"\n❌ Workflow execution error: {e}")
            execution.success = False
            execution.end_time = datetime.now()
            return execution
    
    def _compensate_actions(self, execution: WorkflowExecution):
        """
        Compensate completed actions in reverse order.
        
        Args:
            execution: Workflow execution to compensate
        """
        # Compensate in reverse order
        for action in reversed(execution.completed_actions):
            print(f"\n   🔙 Compensating: {action.name}")
            
            action.status = ActionStatus.COMPENSATING
            
            # Get compensation function
            compensate_fn = action.compensate_fn
            if not compensate_fn and action.name in self.compensation_registry:
                compensate_fn = self.compensation_registry[action.name]
            
            if not compensate_fn:
                print(f"      ⚠️  No compensation function available")
                execution.compensation_results.append((action, False))
                action.status = ActionStatus.COMPENSATION_FAILED
                continue
            
            try:
                # Execute compensation
                compensate_fn(action.result)
                action.status = ActionStatus.COMPENSATED
                action.compensated_at = datetime.now()
                execution.compensation_results.append((action, True))
                
                print(f"      ✅ Compensated successfully")
                
            except Exception as e:
                print(f"      ❌ Compensation failed: {e}")
                action.status = ActionStatus.COMPENSATION_FAILED
                execution.compensation_results.append((action, False))
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowExecution]:
        """Get status of a workflow execution."""
        return self.workflows.get(workflow_id)
    
    def generate_compensation_report(
        self,
        workflow_id: str
    ) -> str:
        """
        Generate a detailed compensation report.
        
        Args:
            workflow_id: Workflow to report on
            
        Returns:
            Formatted report
        """
        execution = self.workflows.get(workflow_id)
        if not execution:
            return f"Workflow {workflow_id} not found"
        
        report = []
        report.append("\n" + "="*60)
        report.append(f"WORKFLOW COMPENSATION REPORT: {workflow_id}")
        report.append("="*60)
        
        report.append(f"\n📊 Execution Summary:")
        report.append(f"   Total Actions: {len(execution.actions)}")
        report.append(f"   Completed: {len(execution.completed_actions)}")
        report.append(f"   Status: {'✅ SUCCESS' if execution.success else '❌ FAILED'}")
        
        if execution.failed_action:
            report.append(f"\n⚠️  Failed Action:")
            report.append(f"   Name: {execution.failed_action.name}")
            report.append(f"   Error: {execution.failed_action.error}")
            report.append(f"   Position: {execution.current_index + 1}/{len(execution.actions)}")
        
        if execution.compensation_results:
            report.append(f"\n🔄 Compensation Results:")
            successful = sum(1 for _, success in execution.compensation_results if success)
            report.append(f"   Total Compensations: {len(execution.compensation_results)}")
            report.append(f"   Successful: {successful}")
            report.append(f"   Failed: {len(execution.compensation_results) - successful}")
            
            report.append(f"\n   Details:")
            for action, success in execution.compensation_results:
                status = "✅" if success else "❌"
                report.append(f"      {status} {action.name}")
        
        if execution.end_time:
            duration = (execution.end_time - execution.start_time).total_seconds()
            report.append(f"\n⏱️  Timing:")
            report.append(f"   Duration: {duration:.2f}s")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


def demonstrate_compensating_actions():
    """Demonstrate the Compensating Actions pattern."""
    print("="*60)
    print("COMPENSATING ACTIONS PATTERN DEMONSTRATION")
    print("="*60)
    
    agent = CompensatingActionsAgent()
    
    # Example 1: Simple workflow with compensation
    print("\n" + "="*60)
    print("Example 1: Simple Workflow with Successful Compensation")
    print("="*60)
    
    # Simulated state
    state = {
        "user_created": False,
        "email_sent": False,
        "database_updated": False
    }
    
    def create_user(user_id: str) -> str:
        """Create a user account."""
        print(f"      Creating user: {user_id}")
        state["user_created"] = True
        return f"User {user_id} created"
    
    def compensate_create_user(result: str):
        """Delete the created user."""
        print(f"      Deleting user from result: {result}")
        state["user_created"] = False
    
    def send_email(email: str) -> str:
        """Send welcome email."""
        print(f"      Sending email to: {email}")
        state["email_sent"] = True
        return f"Email sent to {email}"
    
    def compensate_send_email(result: str):
        """Send cancellation email."""
        print(f"      Sending cancellation email")
        state["email_sent"] = False
    
    def update_database(data: dict) -> str:
        """Update database - THIS WILL FAIL."""
        print(f"      Updating database with: {data}")
        raise Exception("Database connection failed")
    
    # Create workflow
    actions = [
        Action(
            action_id="1",
            name="create_user",
            execute_fn=create_user,
            compensate_fn=compensate_create_user,
            args=("user123",)
        ),
        Action(
            action_id="2",
            name="send_email",
            execute_fn=send_email,
            compensate_fn=compensate_send_email,
            args=("user@example.com",)
        ),
        Action(
            action_id="3",
            name="update_database",
            execute_fn=update_database,
            args=({"user": "user123"},)
        )
    ]
    
    execution = agent.execute_workflow("workflow-001", actions)
    
    print(f"\nFinal State:")
    print(f"   User Created: {state['user_created']}")
    print(f"   Email Sent: {state['email_sent']}")
    print(f"   Database Updated: {state['database_updated']}")
    
    report = agent.generate_compensation_report("workflow-001")
    print(report)
    
    # Example 2: Multi-step API workflow
    print("\n" + "="*60)
    print("Example 2: Multi-Step API Workflow")
    print("="*60)
    
    api_state = {
        "payment_id": None,
        "inventory_reserved": False,
        "shipping_scheduled": False
    }
    
    def process_payment(amount: float) -> str:
        """Process payment."""
        payment_id = f"PAY-{int(time.time())}"
        api_state["payment_id"] = payment_id
        print(f"      Payment processed: {payment_id} for ${amount}")
        return payment_id
    
    def compensate_payment(payment_id: str):
        """Refund payment."""
        print(f"      Refunding payment: {payment_id}")
        api_state["payment_id"] = None
    
    def reserve_inventory(item_id: str, quantity: int) -> str:
        """Reserve inventory."""
        api_state["inventory_reserved"] = True
        print(f"      Reserved {quantity} units of {item_id}")
        return f"Reserved {quantity} units"
    
    def compensate_inventory(result: str):
        """Release inventory reservation."""
        print(f"      Releasing inventory reservation")
        api_state["inventory_reserved"] = False
    
    def schedule_shipping(address: str) -> str:
        """Schedule shipping - WILL FAIL."""
        print(f"      Scheduling shipping to: {address}")
        raise Exception("Shipping service unavailable")
    
    actions = [
        Action(
            action_id="1",
            name="process_payment",
            execute_fn=process_payment,
            compensate_fn=compensate_payment,
            args=(99.99,)
        ),
        Action(
            action_id="2",
            name="reserve_inventory",
            execute_fn=reserve_inventory,
            compensate_fn=compensate_inventory,
            args=("ITEM-001", 2)
        ),
        Action(
            action_id="3",
            name="schedule_shipping",
            execute_fn=schedule_shipping,
            args=("123 Main St",)
        )
    ]
    
    execution = agent.execute_workflow("workflow-002", actions)
    
    print(f"\nFinal API State:")
    print(f"   Payment ID: {api_state['payment_id']}")
    print(f"   Inventory Reserved: {api_state['inventory_reserved']}")
    print(f"   Shipping Scheduled: {api_state['shipping_scheduled']}")
    
    # Example 3: Successful workflow (no compensation needed)
    print("\n" + "="*60)
    print("Example 3: Successful Workflow (No Compensation)")
    print("="*60)
    
    success_state = {"steps": []}
    
    def step1() -> str:
        success_state["steps"].append("step1")
        return "Step 1 completed"
    
    def step2() -> str:
        success_state["steps"].append("step2")
        return "Step 2 completed"
    
    def step3() -> str:
        success_state["steps"].append("step3")
        return "Step 3 completed"
    
    actions = [
        Action("1", "step1", step1),
        Action("2", "step2", step2),
        Action("3", "step3", step3)
    ]
    
    execution = agent.execute_workflow("workflow-003", actions)
    
    print(f"\nCompleted Steps: {success_state['steps']}")
    print(f"Workflow Success: {execution.success}")
    
    # Example 4: Compensation with LLM decision
    print("\n" + "="*60)
    print("Example 4: LLM-Assisted Compensation Decision")
    print("="*60)
    
    def should_compensate_action(action: Action, error: str) -> bool:
        """Use LLM to decide if action should be compensated."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in distributed systems. "
                      "Determine if an action should be compensated (undone) "
                      "based on the error. Reply with ONLY 'yes' or 'no'."),
            ("user", "Action: {action}\nError: {error}\n\n"
                    "Should this action be compensated?")
        ])
        
        chain = prompt | agent.llm | StrOutputParser()
        result = chain.invoke({
            "action": action.name,
            "error": error
        })
        
        return "yes" in result.lower()
    
    # Simulate a workflow with LLM-guided compensation
    test_action = Action(
        action_id="test",
        name="send_notification",
        execute_fn=lambda: "Notification sent",
        compensate_fn=lambda x: print("Notification cancelled")
    )
    test_action.result = "Notification sent"
    
    test_error = "Network timeout after 30 seconds"
    
    print(f"\nAsking LLM if '{test_action.name}' should be compensated")
    print(f"Error: {test_error}")
    
    should_compensate = should_compensate_action(test_action, test_error)
    print(f"\nLLM Decision: {'Compensate' if should_compensate else 'Do not compensate'}")
    
    # Example 5: Idempotent actions
    print("\n" + "="*60)
    print("Example 5: Idempotent Actions (Safe to Retry)")
    print("="*60)
    
    idempotent_state = {"writes": 0}
    
    def idempotent_write(key: str, value: str) -> str:
        """Idempotent write operation."""
        idempotent_state["writes"] += 1
        print(f"      Write #{idempotent_state['writes']}: {key} = {value}")
        return f"Set {key} = {value}"
    
    # Idempotent actions don't need compensation
    idempotent_action = Action(
        action_id="idem",
        name="idempotent_write",
        execute_fn=idempotent_write,
        args=("config", "value123"),
        idempotent=True
    )
    
    print("Executing idempotent action multiple times:")
    for i in range(3):
        result = idempotent_action.execute_fn(*idempotent_action.args)
        print(f"   Attempt {i+1}: {result}")
    
    print(f"\n💡 Idempotent actions can be safely retried without compensation")
    print(f"   Total writes: {idempotent_state['writes']}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
The Compensating Actions pattern demonstrates:

1. Automatic Rollback: Undoes completed actions when workflow fails
2. Reverse Order: Compensates actions in reverse of execution
3. State Consistency: Maintains system consistency after failures
4. Saga Pattern: Implements saga-like transaction semantics
5. Flexible Compensation: Supports custom compensation logic

Key Benefits:
- Maintains consistency in distributed workflows
- Automatic error recovery
- Clear rollback semantics
- Reduces manual cleanup effort
- Better fault tolerance

Compensation Strategies:
- Reverse Order: Compensate from last to first (recommended)
- Parallel: Compensate all actions simultaneously (faster)
- Selective: Compensate only specific actions (intelligent)
- Best Effort: Continue compensation even if some fail

Best Practices:
- Design compensating actions alongside forward actions
- Make compensation idempotent when possible
- Log all compensation attempts
- Handle compensation failures gracefully
- Test compensation logic thoroughly
- Consider partial compensation scenarios
- Document non-compensable actions

Idempotent Actions:
- Can be safely retried without side effects
- Don't require compensation
- Examples: PUT operations, SET commands
- Mark actions as idempotent in workflow

Use Cases:
- Multi-service API orchestration
- Database transaction-like workflows
- Payment and order processing
- Resource provisioning
- Data pipeline error recovery

Common Patterns:
- Forward Progress + Compensation (Saga)
- Try-Confirm-Cancel (TCC)
- Two-Phase Commit alternative
- Eventual consistency patterns
    """)


if __name__ == "__main__":
    demonstrate_compensating_actions()
