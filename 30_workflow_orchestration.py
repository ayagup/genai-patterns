"""
Workflow Orchestration Pattern
Manages complex multi-step workflows with dependencies
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
@dataclass
class WorkflowStep:
    id: str
    name: str
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
@dataclass
class WorkflowContext:
    """Shared context across workflow steps"""
    data: Dict[str, Any] = field(default_factory=dict)
    def set(self, key: str, value: Any):
        self.data[key] = value
    def get(self, key: str, default=None) -> Any:
        return self.data.get(key, default)
class WorkflowOrchestrator:
    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.steps: Dict[str, WorkflowStep] = {}
        self.context = WorkflowContext()
        self.execution_order: List[str] = []
    def add_step(self, step: WorkflowStep):
        """Add a step to the workflow"""
        self.steps[step.id] = step
        print(f"Added step: {step.name} (ID: {step.id})")
    def can_execute_step(self, step: WorkflowStep) -> bool:
        """Check if step dependencies are satisfied"""
        for dep_id in step.dependencies:
            if dep_id not in self.steps:
                return False
            dep_step = self.steps[dep_id]
            if dep_step.status != StepStatus.COMPLETED:
                return False
        return True
    def execute_step(self, step: WorkflowStep) -> bool:
        """Execute a single workflow step"""
        print(f"\n{'='*60}")
        print(f"Executing: {step.name}")
        print(f"{'='*60}")
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now()
        try:
            # Execute the step function with context
            print(f"Running function: {step.function.__name__}")
            result = step.function(self.context)
            step.result = result
            step.status = StepStatus.COMPLETED
            step.end_time = datetime.now()
            duration = (step.end_time - step.start_time).total_seconds()
            print(f"✓ Completed in {duration:.2f}s")
            print(f"Result: {result}")
            return True
        except Exception as e:
            step.error = str(e)
            step.end_time = datetime.now()
            # Retry logic
            if step.retry_count < step.max_retries:
                step.retry_count += 1
                step.status = StepStatus.PENDING
                print(f"⚠ Failed (attempt {step.retry_count}/{step.max_retries}): {str(e)}")
                print(f"Will retry...")
                time.sleep(1)  # Brief delay before retry
                return self.execute_step(step)  # Recursive retry
            else:
                step.status = StepStatus.FAILED
                print(f"✗ Failed after {step.max_retries} retries: {str(e)}")
                return False
    def build_execution_order(self) -> List[str]:
        """Build topologically sorted execution order"""
        # Simple topological sort
        visited = set()
        order = []
        def visit(step_id: str):
            if step_id in visited:
                return
            visited.add(step_id)
            step = self.steps[step_id]
            for dep_id in step.dependencies:
                visit(dep_id)
            order.append(step_id)
        for step_id in self.steps:
            visit(step_id)
        return order
    def execute_workflow(self, parallel: bool = False) -> Dict[str, Any]:
        """Execute the entire workflow"""
        print(f"\n{'='*70}")
        print(f"WORKFLOW: {self.workflow_name}")
        print(f"{'='*70}")
        print(f"Total steps: {len(self.steps)}")
        print(f"Execution mode: {'Parallel' if parallel else 'Sequential'}")
        # Build execution order
        self.execution_order = self.build_execution_order()
        print(f"\nExecution order: {' -> '.join(self.execution_order)}")
        workflow_start = datetime.now()
        if parallel:
            # Parallel execution (simplified - real impl would use threading/async)
            self._execute_parallel()
        else:
            # Sequential execution
            self._execute_sequential()
        workflow_end = datetime.now()
        duration = (workflow_end - workflow_start).total_seconds()
        # Generate summary
        summary = self.get_summary()
        summary['total_duration'] = duration
        self._print_summary(summary)
        return summary
    def _execute_sequential(self):
        """Execute workflow sequentially"""
        for step_id in self.execution_order:
            step = self.steps[step_id]
            if not self.can_execute_step(step):
                print(f"\n⚠ Skipping {step.name}: dependencies not met")
                step.status = StepStatus.SKIPPED
                continue
            success = self.execute_step(step)
            if not success and step.status == StepStatus.FAILED:
                print(f"\n✗ Workflow failed at step: {step.name}")
                break
    def _execute_parallel(self):
        """Execute workflow with parallelization where possible"""
        # Simplified parallel execution
        executed = set()
        while len(executed) < len(self.steps):
            # Find steps that can execute now
            ready_steps = [
                step for step in self.steps.values()
                if step.id not in executed and
                step.status == StepStatus.PENDING and
                self.can_execute_step(step)
            ]
            if not ready_steps:
                break
            # Execute ready steps (in real impl, use threading)
            for step in ready_steps:
                self.execute_step(step)
                executed.add(step.id)
    def get_summary(self) -> Dict[str, Any]:
        """Get workflow execution summary"""
        total = len(self.steps)
        completed = sum(1 for s in self.steps.values() if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.steps.values() if s.status == StepStatus.FAILED)
        skipped = sum(1 for s in self.steps.values() if s.status == StepStatus.SKIPPED)
        return {
            'workflow_name': self.workflow_name,
            'total_steps': total,
            'completed': completed,
            'failed': failed,
            'skipped': skipped,
            'success_rate': completed / total if total > 0 else 0,
            'steps': {
                step_id: {
                    'name': step.name,
                    'status': step.status.value,
                    'duration': (step.end_time - step.start_time).total_seconds() 
                                if step.start_time and step.end_time else None,
                    'retries': step.retry_count
                }
                for step_id, step in self.steps.items()
            }
        }
    def _print_summary(self, summary: Dict[str, Any]):
        """Print workflow summary"""
        print(f"\n{'='*70}")
        print(f"WORKFLOW SUMMARY")
        print(f"{'='*70}")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Completed: {summary['completed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary.get('total_duration', 0):.2f}s")
        print(f"\nStep Details:")
        for step_id, step_info in summary['steps'].items():
            duration_str = f"{step_info['duration']:.2f}s" if step_info['duration'] else "N/A"
            retry_str = f" ({step_info['retries']} retries)" if step_info['retries'] > 0 else ""
            print(f"  {step_info['name']}: {step_info['status']} - {duration_str}{retry_str}")
# Example workflow functions
def load_data(context: WorkflowContext) -> Dict[str, Any]:
    """Step 1: Load data"""
    time.sleep(0.5)  # Simulate work
    data = {"records": 1000, "source": "database"}
    context.set("raw_data", data)
    return data
def validate_data(context: WorkflowContext) -> Dict[str, Any]:
    """Step 2: Validate data"""
    time.sleep(0.3)
    raw_data = context.get("raw_data")
    validated = {"valid_records": raw_data["records"] - 10, "invalid": 10}
    context.set("validated_data", validated)
    return validated
def transform_data(context: WorkflowContext) -> Dict[str, Any]:
    """Step 3: Transform data"""
    time.sleep(0.4)
    validated = context.get("validated_data")
    transformed = {"processed_records": validated["valid_records"]}
    context.set("transformed_data", transformed)
    return transformed
def analyze_data(context: WorkflowContext) -> Dict[str, Any]:
    """Step 4: Analyze data"""
    time.sleep(0.6)
    transformed = context.get("transformed_data")
    analysis = {
        "total_processed": transformed["processed_records"],
        "insights": ["Insight 1", "Insight 2"]
    }
    context.set("analysis", analysis)
    return analysis
def generate_report(context: WorkflowContext) -> str:
    """Step 5: Generate report"""
    time.sleep(0.3)
    analysis = context.get("analysis")
    report = f"Report: Processed {analysis['total_processed']} records"
    context.set("report", report)
    return report
# Usage
if __name__ == "__main__":
    # Create workflow
    workflow = WorkflowOrchestrator("Data Processing Pipeline")
    # Add steps with dependencies
    workflow.add_step(WorkflowStep(
        id="load",
        name="Load Data",
        function=load_data,
        dependencies=[]
    ))
    workflow.add_step(WorkflowStep(
        id="validate",
        name="Validate Data",
        function=validate_data,
        dependencies=["load"]
    ))
    workflow.add_step(WorkflowStep(
        id="transform",
        name="Transform Data",
        function=transform_data,
        dependencies=["validate"]
    ))
    workflow.add_step(WorkflowStep(
        id="analyze",
        name="Analyze Data",
        function=analyze_data,
        dependencies=["transform"]
    ))
    workflow.add_step(WorkflowStep(
        id="report",
        name="Generate Report",
        function=generate_report,
        dependencies=["analyze"]
    ))
    # Execute workflow
    summary = workflow.execute_workflow(parallel=False)
