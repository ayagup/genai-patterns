"""
Pattern 147: Compensating Actions Agent (Saga Pattern)

This pattern implements the saga pattern with compensating actions for distributed
transactions. When a step fails, compensating actions undo previous successful steps
to maintain consistency across distributed systems.

Category: Error Handling & Recovery
Use Cases:
- Distributed transactions
- Multi-step workflows
- E-commerce order processing
- Booking systems with multiple services
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, TypeVar
from enum import Enum
from datetime import datetime
import random

T = TypeVar('T')


class SagaState(Enum):
    """States of a saga execution"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPENSATING = "compensating"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATED = "compensated"


class StepStatus(Enum):
    """Status of individual saga step"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    COMPENSATION_FAILED = "compensation_failed"


class CoordinationType(Enum):
    """Types of saga coordination"""
    ORCHESTRATION = "orchestration"  # Central coordinator
    CHOREOGRAPHY = "choreography"    # Event-driven


@dataclass
class SagaStep:
    """Definition of a saga step"""
    step_id: str
    name: str
    forward_action: Callable[..., Any]
    compensating_action: Callable[..., Any]
    requires_compensation: bool = True
    timeout_seconds: float = 30.0
    retry_count: int = 3


@dataclass
class StepExecution:
    """Execution record of a step"""
    step_id: str
    status: StepStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    compensation_started_at: Optional[datetime] = None
    compensation_completed_at: Optional[datetime] = None
    compensation_result: Any = None
    compensation_error: Optional[str] = None
    retry_attempts: int = 0


@dataclass
class SagaExecution:
    """Record of entire saga execution"""
    saga_id: str
    state: SagaState
    started_at: datetime
    completed_at: Optional[datetime] = None
    step_executions: List[StepExecution] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    total_steps: int = 0
    completed_steps: int = 0
    failed_step_id: Optional[str] = None
    compensation_required: bool = False


class CompensationStrategy(Enum):
    """Strategies for compensation"""
    REVERSE_ORDER = "reverse_order"  # Compensate in reverse order
    PARALLEL = "parallel"            # Compensate all in parallel
    SELECTIVE = "selective"          # Only compensate affected steps


@dataclass
class CompensationPlan:
    """Plan for compensating failed saga"""
    steps_to_compensate: List[str]
    strategy: CompensationStrategy
    created_at: datetime = field(default_factory=datetime.now)


class SagaStepExecutor:
    """Executes individual saga steps"""
    
    def execute_forward(
        self,
        step: SagaStep,
        context: Dict[str, Any]
    ) -> StepExecution:
        """Execute forward action of step"""
        execution = StepExecution(
            step_id=step.step_id,
            status=StepStatus.EXECUTING,
            started_at=datetime.now()
        )
        
        try:
            # Execute forward action
            result = step.forward_action(context)
            
            execution.status = StepStatus.COMPLETED
            execution.result = result
            execution.completed_at = datetime.now()
            
            return execution
            
        except Exception as e:
            execution.status = StepStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()
            
            return execution
    
    def execute_compensation(
        self,
        step: SagaStep,
        forward_execution: StepExecution,
        context: Dict[str, Any]
    ) -> StepExecution:
        """Execute compensating action"""
        forward_execution.status = StepStatus.COMPENSATING
        forward_execution.compensation_started_at = datetime.now()
        
        try:
            # Execute compensating action
            result = step.compensating_action(context, forward_execution.result)
            
            forward_execution.status = StepStatus.COMPENSATED
            forward_execution.compensation_result = result
            forward_execution.compensation_completed_at = datetime.now()
            
            return forward_execution
            
        except Exception as e:
            forward_execution.status = StepStatus.COMPENSATION_FAILED
            forward_execution.compensation_error = str(e)
            forward_execution.compensation_completed_at = datetime.now()
            
            return forward_execution


class CompensationPlanner:
    """Plans compensation for failed sagas"""
    
    def create_plan(
        self,
        saga_execution: SagaExecution,
        steps: List[SagaStep],
        strategy: CompensationStrategy = CompensationStrategy.REVERSE_ORDER
    ) -> CompensationPlan:
        """Create compensation plan"""
        steps_to_compensate = []
        
        # Find all completed steps that need compensation
        for execution in saga_execution.step_executions:
            if execution.status == StepStatus.COMPLETED:
                # Find step definition
                step_def = next(
                    (s for s in steps if s.step_id == execution.step_id),
                    None
                )
                
                if step_def and step_def.requires_compensation:
                    steps_to_compensate.append(execution.step_id)
        
        # Reverse order for typical compensation
        if strategy == CompensationStrategy.REVERSE_ORDER:
            steps_to_compensate.reverse()
        
        return CompensationPlan(
            steps_to_compensate=steps_to_compensate,
            strategy=strategy
        )


class SagaOrchestrator:
    """Orchestrates saga execution with compensation"""
    
    def __init__(self):
        self.step_executor = SagaStepExecutor()
        self.compensation_planner = CompensationPlanner()
        
    def execute_saga(
        self,
        saga_id: str,
        steps: List[SagaStep],
        initial_context: Dict[str, Any]
    ) -> SagaExecution:
        """Execute saga with automatic compensation on failure"""
        saga_execution = SagaExecution(
            saga_id=saga_id,
            state=SagaState.EXECUTING,
            started_at=datetime.now(),
            context=initial_context.copy(),
            total_steps=len(steps)
        )
        
        print(f"\n{'='*60}")
        print(f"EXECUTING SAGA: {saga_id}")
        print(f"{'='*60}")
        
        # Execute steps in order
        for i, step in enumerate(steps):
            print(f"\nStep {i+1}/{len(steps)}: {step.name}")
            
            # Execute forward action
            execution = self.step_executor.execute_forward(
                step,
                saga_execution.context
            )
            saga_execution.step_executions.append(execution)
            
            if execution.status == StepStatus.COMPLETED:
                print(f"  ✓ Completed: {execution.result}")
                saga_execution.completed_steps += 1
                
                # Update context with result
                saga_execution.context[step.step_id] = execution.result
                
            else:
                print(f"  ✗ Failed: {execution.error}")
                saga_execution.state = SagaState.FAILED
                saga_execution.failed_step_id = step.step_id
                saga_execution.compensation_required = True
                
                # Start compensation
                self._compensate_saga(saga_execution, steps)
                
                return saga_execution
        
        # All steps completed successfully
        saga_execution.state = SagaState.COMPLETED
        saga_execution.completed_at = datetime.now()
        
        print(f"\n✓ SAGA COMPLETED SUCCESSFULLY")
        
        return saga_execution
    
    def _compensate_saga(
        self,
        saga_execution: SagaExecution,
        steps: List[SagaStep]
    ):
        """Compensate failed saga"""
        print(f"\n{'='*60}")
        print(f"COMPENSATING SAGA: {saga_execution.saga_id}")
        print(f"{'='*60}")
        
        saga_execution.state = SagaState.COMPENSATING
        
        # Create compensation plan
        plan = self.compensation_planner.create_plan(
            saga_execution,
            steps,
            CompensationStrategy.REVERSE_ORDER
        )
        
        print(f"Compensating {len(plan.steps_to_compensate)} step(s)")
        
        # Execute compensations
        compensation_success = True
        
        for step_id in plan.steps_to_compensate:
            # Find step definition and execution
            step = next(s for s in steps if s.step_id == step_id)
            execution = next(
                e for e in saga_execution.step_executions
                if e.step_id == step_id
            )
            
            print(f"\nCompensating: {step.name}")
            
            # Execute compensation
            self.step_executor.execute_compensation(
                step,
                execution,
                saga_execution.context
            )
            
            if execution.status == StepStatus.COMPENSATED:
                print(f"  ✓ Compensated: {execution.compensation_result}")
            else:
                print(f"  ✗ Compensation failed: {execution.compensation_error}")
                compensation_success = False
        
        # Update saga state
        if compensation_success:
            saga_execution.state = SagaState.COMPENSATED
            print(f"\n✓ SAGA FULLY COMPENSATED")
        else:
            saga_execution.state = SagaState.FAILED
            print(f"\n✗ SAGA COMPENSATION INCOMPLETE")
        
        saga_execution.completed_at = datetime.now()


class SagaMonitor:
    """Monitors saga executions"""
    
    def __init__(self):
        self.executions: Dict[str, SagaExecution] = {}
        
    def track_execution(self, execution: SagaExecution):
        """Track saga execution"""
        self.executions[execution.saga_id] = execution
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get saga statistics"""
        if not self.executions:
            return {"error": "No executions tracked"}
        
        total = len(self.executions)
        by_state = {}
        
        for execution in self.executions.values():
            state = execution.state.value
            by_state[state] = by_state.get(state, 0) + 1
        
        # Count compensations
        compensated_count = sum(
            1 for e in self.executions.values()
            if e.state == SagaState.COMPENSATED
        )
        
        # Count compensation failures
        compensation_failures = sum(
            1 for e in self.executions.values()
            for step in e.step_executions
            if step.status == StepStatus.COMPENSATION_FAILED
        )
        
        return {
            'total_executions': total,
            'by_state': by_state,
            'compensation_rate': compensated_count / total if total > 0 else 0,
            'compensation_failures': compensation_failures,
            'success_rate': by_state.get('completed', 0) / total if total > 0 else 0
        }


class CompensatingActionsAgent:
    """
    Main agent that orchestrates sagas with compensating actions.
    Ensures consistency in distributed transactions through compensation.
    """
    
    def __init__(self):
        self.orchestrator = SagaOrchestrator()
        self.monitor = SagaMonitor()
        self.saga_templates: Dict[str, List[SagaStep]] = {}
        
    def register_saga_template(
        self,
        template_name: str,
        steps: List[SagaStep]
    ):
        """Register a saga template"""
        self.saga_templates[template_name] = steps
    
    def execute_saga(
        self,
        template_name: str,
        saga_id: str,
        initial_context: Dict[str, Any]
    ) -> SagaExecution:
        """Execute saga from template"""
        if template_name not in self.saga_templates:
            raise ValueError(f"Unknown saga template: {template_name}")
        
        steps = self.saga_templates[template_name]
        execution = self.orchestrator.execute_saga(saga_id, steps, initial_context)
        
        # Track execution
        self.monitor.track_execution(execution)
        
        return execution
    
    def get_saga_status(self, saga_id: str) -> Optional[SagaExecution]:
        """Get status of saga execution"""
        return self.monitor.executions.get(saga_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get saga statistics"""
        return self.monitor.get_statistics()


def demonstrate_compensating_actions():
    """Demonstrate compensating actions pattern"""
    print("\n" + "="*60)
    print("COMPENSATING ACTIONS (SAGA) PATTERN DEMONSTRATION")
    print("="*60)
    
    agent = CompensatingActionsAgent()
    
    # Scenario 1: E-commerce order saga (success case)
    print("\n" + "-"*60)
    print("Scenario 1: Successful Order Processing")
    print("-"*60)
    
    # Define order processing steps
    def reserve_inventory(context):
        order_id = context['order_id']
        items = context['items']
        return f"Reserved {len(items)} items for order {order_id}"
    
    def compensate_inventory(context, forward_result):
        return f"Released inventory reservation"
    
    def process_payment(context):
        amount = context['amount']
        return f"Charged ${amount}"
    
    def compensate_payment(context, forward_result):
        amount = context['amount']
        return f"Refunded ${amount}"
    
    def ship_order(context):
        order_id = context['order_id']
        return f"Shipped order {order_id}"
    
    def compensate_shipment(context, forward_result):
        return f"Cancelled shipment"
    
    def send_confirmation(context):
        order_id = context['order_id']
        return f"Sent confirmation for order {order_id}"
    
    def compensate_confirmation(context, forward_result):
        return f"Sent cancellation notice"
    
    order_saga = [
        SagaStep(
            step_id="reserve_inventory",
            name="Reserve Inventory",
            forward_action=reserve_inventory,
            compensating_action=compensate_inventory
        ),
        SagaStep(
            step_id="process_payment",
            name="Process Payment",
            forward_action=process_payment,
            compensating_action=compensate_payment
        ),
        SagaStep(
            step_id="ship_order",
            name="Ship Order",
            forward_action=ship_order,
            compensating_action=compensate_shipment
        ),
        SagaStep(
            step_id="send_confirmation",
            name="Send Confirmation",
            forward_action=send_confirmation,
            compensating_action=compensate_confirmation,
            requires_compensation=False  # Email doesn't need compensation
        )
    ]
    
    agent.register_saga_template("order_processing", order_saga)
    
    # Execute successful order
    result = agent.execute_saga(
        "order_processing",
        "order-001",
        {
            'order_id': 'order-001',
            'items': ['item1', 'item2'],
            'amount': 99.99
        }
    )
    
    print(f"\nFinal State: {result.state.value}")
    print(f"Steps Completed: {result.completed_steps}/{result.total_steps}")
    
    # Scenario 2: Failed order with compensation
    print("\n" + "-"*60)
    print("Scenario 2: Failed Order with Compensation")
    print("-"*60)
    
    # Define payment failure
    def process_payment_fail(context):
        raise Exception("Payment declined - insufficient funds")
    
    failed_order_saga = [
        SagaStep(
            step_id="reserve_inventory",
            name="Reserve Inventory",
            forward_action=reserve_inventory,
            compensating_action=compensate_inventory
        ),
        SagaStep(
            step_id="process_payment",
            name="Process Payment",
            forward_action=process_payment_fail,  # This will fail
            compensating_action=compensate_payment
        ),
        SagaStep(
            step_id="ship_order",
            name="Ship Order",
            forward_action=ship_order,
            compensating_action=compensate_shipment
        )
    ]
    
    agent.register_saga_template("failed_order", failed_order_saga)
    
    result = agent.execute_saga(
        "failed_order",
        "order-002",
        {
            'order_id': 'order-002',
            'items': ['item1'],
            'amount': 49.99
        }
    )
    
    print(f"\nFinal State: {result.state.value}")
    print(f"Failed Step: {result.failed_step_id}")
    print(f"Compensation Required: {result.compensation_required}")
    
    # Show compensation details
    print("\nStep Status Summary:")
    for step in result.step_executions:
        print(f"  {step.step_id}: {step.status.value}")
    
    # Scenario 3: Booking saga with multiple services
    print("\n" + "-"*60)
    print("Scenario 3: Travel Booking Saga")
    print("-"*60)
    
    def book_flight(context):
        return f"Booked flight {context['flight_id']}"
    
    def cancel_flight(context, forward_result):
        return f"Cancelled flight booking"
    
    def book_hotel(context):
        return f"Booked hotel {context['hotel_id']}"
    
    def cancel_hotel(context, forward_result):
        return f"Cancelled hotel booking"
    
    def book_car(context):
        # Simulate failure
        raise Exception("No cars available")
    
    def cancel_car(context, forward_result):
        return f"Cancelled car rental"
    
    travel_saga = [
        SagaStep(
            step_id="book_flight",
            name="Book Flight",
            forward_action=book_flight,
            compensating_action=cancel_flight
        ),
        SagaStep(
            step_id="book_hotel",
            name="Book Hotel",
            forward_action=book_hotel,
            compensating_action=cancel_hotel
        ),
        SagaStep(
            step_id="book_car",
            name="Book Rental Car",
            forward_action=book_car,
            compensating_action=cancel_car
        )
    ]
    
    agent.register_saga_template("travel_booking", travel_saga)
    
    result = agent.execute_saga(
        "travel_booking",
        "booking-003",
        {
            'flight_id': 'FL123',
            'hotel_id': 'HT456',
            'car_id': 'CR789'
        }
    )
    
    # Scenario 4: Statistics summary
    print("\n" + "-"*60)
    print("Scenario 4: Saga Statistics")
    print("-"*60)
    
    stats = agent.get_statistics()
    
    print(f"\nTotal Saga Executions: {stats['total_executions']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Compensation Rate: {stats['compensation_rate']:.1%}")
    print(f"Compensation Failures: {stats['compensation_failures']}")
    
    print("\nBy State:")
    for state, count in stats['by_state'].items():
        print(f"  {state}: {count}")
    
    # Scenario 5: Query saga status
    print("\n" + "-"*60)
    print("Scenario 5: Query Saga Status")
    print("-"*60)
    
    saga_ids = ["order-001", "order-002", "booking-003"]
    
    for saga_id in saga_ids:
        saga = agent.get_saga_status(saga_id)
        if saga:
            duration = (saga.completed_at - saga.started_at).total_seconds() if saga.completed_at else 0
            print(f"\n{saga_id}:")
            print(f"  State: {saga.state.value}")
            print(f"  Steps: {saga.completed_steps}/{saga.total_steps}")
            print(f"  Duration: {duration:.3f}s")
            if saga.failed_step_id:
                print(f"  Failed at: {saga.failed_step_id}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Demonstrated successful saga execution")
    print(f"✓ Showed automatic compensation on failure")
    print(f"✓ Implemented reverse-order compensation")
    print(f"✓ Tracked saga executions and statistics")
    print(f"✓ Maintained consistency across distributed operations")
    print("\n✅ Error Handling & Recovery Category: Pattern 2/5 complete")
    print("Ready for distributed transaction management!")


if __name__ == "__main__":
    demonstrate_compensating_actions()
