"""
Agent Workflow Orchestration Pattern

Orchestrates complex workflows across multiple agents.
Manages task dependencies, parallel execution, and error handling.

Use Cases:
- Business process automation
- Multi-step data processing
- Distributed task execution
- Complex agent coordination

Advantages:
- Coordinated execution
- Dependency management
- Parallel processing
- Error recovery
"""

from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import time


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """Individual task status"""
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class WorkflowTask:
    """Task within workflow"""
    task_id: str
    name: str
    agent_id: str
    action: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.WAITING
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class Workflow:
    """Workflow definition"""
    workflow_id: str
    name: str
    tasks: Dict[str, WorkflowTask]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Runtime execution state"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_tasks: Set[str] = field(default_factory=set)
    running_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)


class DependencyResolver:
    """Resolves task dependencies"""
    
    def get_ready_tasks(self,
                       workflow: Workflow,
                       completed: Set[str]) -> List[str]:
        """
        Get tasks ready to execute.
        
        Args:
            workflow: Workflow definition
            completed: Set of completed task IDs
            
        Returns:
            List of ready task IDs
        """
        ready = []
        
        for task_id, task in workflow.tasks.items():
            # Skip if already completed or running
            if task.status in [TaskStatus.COMPLETED, TaskStatus.RUNNING]:
                continue
            
            # Check if all dependencies are met
            deps_met = all(
                dep_id in completed
                for dep_id in task.dependencies
            )
            
            if deps_met:
                ready.append(task_id)
        
        # Sort by priority
        ready.sort(
            key=lambda tid: workflow.tasks[tid].priority.value,
            reverse=True
        )
        
        return ready
    
    def validate_workflow(self, workflow: Workflow) -> Tuple[bool, List[str]]:
        """
        Validate workflow for cycles and invalid dependencies.
        
        Args:
            workflow: Workflow to validate
            
        Returns:
            (is_valid, errors) tuple
        """
        errors = []
        
        # Check for missing dependencies
        task_ids = set(workflow.tasks.keys())
        
        for task_id, task in workflow.tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    errors.append(
                        "Task {} depends on non-existent task {}".format(
                            task_id, dep_id
                        )
                    )
        
        # Check for cycles using DFS
        if self._has_cycle(workflow):
            errors.append("Workflow contains dependency cycle")
        
        return len(errors) == 0, errors
    
    def _has_cycle(self, workflow: Workflow) -> bool:
        """Check for dependency cycles"""
        visited = set()
        rec_stack = set()
        
        def dfs(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = workflow.tasks.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    if dep_id not in visited:
                        if dfs(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True
            
            rec_stack.remove(task_id)
            return False
        
        for task_id in workflow.tasks.keys():
            if task_id not in visited:
                if dfs(task_id):
                    return True
        
        return False


class TaskExecutor:
    """Executes workflow tasks"""
    
    def __init__(self):
        self.task_handlers: Dict[str, Callable] = {}
    
    def register_handler(self,
                        action: str,
                        handler: Callable) -> None:
        """
        Register task handler.
        
        Args:
            action: Action name
            handler: Handler function
        """
        self.task_handlers[action] = handler
    
    def execute_task(self,
                    task: WorkflowTask,
                    context: Dict[str, Any]) -> Tuple[bool, Any]:
        """
        Execute a single task.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            (success, result) tuple
        """
        handler = self.task_handlers.get(task.action)
        
        if not handler:
            return False, "No handler for action: {}".format(task.action)
        
        try:
            # Execute handler
            result = handler(task.parameters, context)
            return True, result
        
        except Exception as e:
            return False, str(e)


class ErrorHandler:
    """Handles workflow errors"""
    
    def __init__(self):
        self.retry_strategies: Dict[str, Callable] = {}
    
    def handle_task_error(self,
                         task: WorkflowTask,
                         error: str,
                         execution: WorkflowExecution) -> bool:
        """
        Handle task error.
        
        Args:
            task: Failed task
            error: Error message
            execution: Execution state
            
        Returns:
            Whether to retry
        """
        # Check retry count
        if task.retry_count >= task.max_retries:
            return False
        
        # Increment retry count
        task.retry_count += 1
        
        # Apply retry strategy if defined
        strategy = self.retry_strategies.get(task.action)
        
        if strategy:
            return strategy(task, error, execution)
        
        # Default: retry with exponential backoff
        backoff = 2 ** task.retry_count
        time.sleep(min(backoff, 60))  # Cap at 60 seconds
        
        return True


class WorkflowMonitor:
    """Monitors workflow execution"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def record_task_execution(self,
                             task: WorkflowTask,
                             duration: float) -> None:
        """Record task execution metrics"""
        if task.task_id not in self.metrics:
            self.metrics[task.task_id] = []
        
        self.metrics[task.task_id].append(duration)
    
    def get_task_stats(self, task_id: str) -> Dict[str, float]:
        """Get statistics for task"""
        if task_id not in self.metrics:
            return {}
        
        durations = self.metrics[task_id]
        
        return {
            "count": len(durations),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations)
        }


class AgentWorkflowOrchestrator:
    """
    Orchestrates complex workflows across multiple agents.
    Manages dependencies, parallel execution, and error handling.
    """
    
    def __init__(self):
        # Components
        self.dependency_resolver = DependencyResolver()
        self.task_executor = TaskExecutor()
        self.error_handler = ErrorHandler()
        self.monitor = WorkflowMonitor()
        
        # State
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.workflow_counter = 0
        self.execution_counter = 0
    
    def create_workflow(self,
                       name: str,
                       tasks: List[Dict[str, Any]],
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create new workflow.
        
        Args:
            name: Workflow name
            tasks: List of task definitions
            metadata: Optional metadata
            
        Returns:
            Workflow ID
        """
        if metadata is None:
            metadata = {}
        
        workflow_id = "workflow_{}".format(self.workflow_counter)
        self.workflow_counter += 1
        
        # Create workflow tasks
        workflow_tasks = {}
        
        for task_def in tasks:
            task = WorkflowTask(
                task_id=task_def["task_id"],
                name=task_def["name"],
                agent_id=task_def["agent_id"],
                action=task_def["action"],
                parameters=task_def.get("parameters", {}),
                dependencies=task_def.get("dependencies", []),
                priority=TaskPriority[task_def.get("priority", "NORMAL").upper()],
                max_retries=task_def.get("max_retries", 3)
            )
            
            workflow_tasks[task.task_id] = task
        
        # Create workflow
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            tasks=workflow_tasks,
            metadata=metadata
        )
        
        # Validate
        is_valid, errors = self.dependency_resolver.validate_workflow(workflow)
        
        if not is_valid:
            raise ValueError("Invalid workflow: {}".format(", ".join(errors)))
        
        self.workflows[workflow_id] = workflow
        
        return workflow_id
    
    def execute_workflow(self,
                        workflow_id: str,
                        context: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute workflow.
        
        Args:
            workflow_id: Workflow to execute
            context: Optional execution context
            
        Returns:
            Execution ID
        """
        if context is None:
            context = {}
        
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError("Workflow not found: {}".format(workflow_id))
        
        # Create execution
        execution = WorkflowExecution(
            execution_id="exec_{}".format(self.execution_counter),
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            started_at=datetime.now(),
            context=context
        )
        
        self.execution_counter += 1
        self.executions[execution.execution_id] = execution
        
        # Update workflow status
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.now()
        
        # Execute workflow
        self._execute_workflow_async(workflow, execution)
        
        return execution.execution_id
    
    def _execute_workflow_async(self,
                                workflow: Workflow,
                                execution: WorkflowExecution) -> None:
        """Execute workflow asynchronously"""
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get ready tasks
            ready_tasks = self.dependency_resolver.get_ready_tasks(
                workflow,
                execution.completed_tasks
            )
            
            if not ready_tasks and not execution.running_tasks:
                # No more tasks to execute
                break
            
            # Execute ready tasks
            for task_id in ready_tasks:
                task = workflow.tasks[task_id]
                
                # Skip if already running
                if task_id in execution.running_tasks:
                    continue
                
                # Execute task
                self._execute_task(task, execution, workflow)
            
            # Check if workflow is complete
            if len(execution.completed_tasks) == len(workflow.tasks):
                break
            
            # Small delay for simulation
            time.sleep(0.1)
        
        # Finalize workflow
        if len(execution.failed_tasks) > 0:
            workflow.status = WorkflowStatus.FAILED
            execution.status = WorkflowStatus.FAILED
        elif len(execution.completed_tasks) == len(workflow.tasks):
            workflow.status = WorkflowStatus.COMPLETED
            execution.status = WorkflowStatus.COMPLETED
        else:
            workflow.status = WorkflowStatus.FAILED
            execution.status = WorkflowStatus.FAILED
        
        workflow.completed_at = datetime.now()
    
    def _execute_task(self,
                     task: WorkflowTask,
                     execution: WorkflowExecution,
                     workflow: Workflow) -> None:
        """Execute single task"""
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        execution.running_tasks.add(task.task_id)
        
        # Execute
        success, result = self.task_executor.execute_task(
            task,
            execution.context
        )
        
        task.end_time = datetime.now()
        duration = (task.end_time - task.start_time).total_seconds()
        
        if success:
            # Task succeeded
            task.status = TaskStatus.COMPLETED
            task.result = result
            execution.completed_tasks.add(task.task_id)
            execution.running_tasks.remove(task.task_id)
            
            # Update context with result
            execution.context[task.task_id] = result
            
            # Record metrics
            self.monitor.record_task_execution(task, duration)
        
        else:
            # Task failed
            task.error = str(result)
            
            # Try error handling
            should_retry = self.error_handler.handle_task_error(
                task,
                task.error,
                execution
            )
            
            if should_retry:
                # Reset for retry
                task.status = TaskStatus.WAITING
                task.start_time = None
                task.end_time = None
                execution.running_tasks.remove(task.task_id)
            else:
                # Mark as failed
                task.status = TaskStatus.FAILED
                execution.failed_tasks.add(task.task_id)
                execution.running_tasks.remove(task.task_id)
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return {}
        
        # Count task statuses
        status_counts = {}
        for task in workflow.tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        duration = None
        if workflow.started_at:
            end = workflow.completed_at or datetime.now()
            duration = (end - workflow.started_at).total_seconds()
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "total_tasks": len(workflow.tasks),
            "task_status_counts": status_counts,
            "duration_seconds": duration,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None
        }
    
    def get_task_details(self,
                        workflow_id: str,
                        task_id: str) -> Dict[str, Any]:
        """Get task details"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return {}
        
        task = workflow.tasks.get(task_id)
        if not task:
            return {}
        
        duration = None
        if task.start_time and task.end_time:
            duration = (task.end_time - task.start_time).total_seconds()
        
        return {
            "task_id": task_id,
            "name": task.name,
            "agent_id": task.agent_id,
            "action": task.action,
            "status": task.status.value,
            "dependencies": task.dependencies,
            "priority": task.priority.value,
            "retry_count": task.retry_count,
            "duration_seconds": duration,
            "result": str(task.result) if task.result else None,
            "error": task.error
        }
    
    def register_task_handler(self,
                             action: str,
                             handler: Callable) -> None:
        """Register task action handler"""
        self.task_executor.register_handler(action, handler)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        total_workflows = len(self.workflows)
        
        status_counts = {}
        for workflow in self.workflows.values():
            status = workflow.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total_tasks = sum(len(w.tasks) for w in self.workflows.values())
        
        return {
            "total_workflows": total_workflows,
            "total_executions": len(self.executions),
            "total_tasks": total_tasks,
            "workflow_status_counts": status_counts,
            "registered_handlers": len(self.task_executor.task_handlers)
        }


def demonstrate_workflow_orchestration():
    """Demonstrate workflow orchestration"""
    print("=" * 70)
    print("Agent Workflow Orchestration Demonstration")
    print("=" * 70)
    
    orchestrator = AgentWorkflowOrchestrator()
    
    # Register task handlers
    def process_data_handler(params, context):
        print("  Processing data: {}".format(params.get("data")))
        return {"processed": True, "count": 100}
    
    def transform_handler(params, context):
        print("  Transforming data")
        return {"transformed": True}
    
    def validate_handler(params, context):
        print("  Validating results")
        return {"valid": True}
    
    def save_handler(params, context):
        print("  Saving results")
        return {"saved": True}
    
    orchestrator.register_task_handler("process_data", process_data_handler)
    orchestrator.register_task_handler("transform", transform_handler)
    orchestrator.register_task_handler("validate", validate_handler)
    orchestrator.register_task_handler("save", save_handler)
    
    # Example 1: Create workflow
    print("\n1. Creating Workflow:")
    
    tasks = [
        {
            "task_id": "task_1",
            "name": "Load Data",
            "agent_id": "agent_loader",
            "action": "process_data",
            "parameters": {"data": "input.csv"},
            "dependencies": [],
            "priority": "HIGH"
        },
        {
            "task_id": "task_2",
            "name": "Transform Data",
            "agent_id": "agent_transformer",
            "action": "transform",
            "parameters": {},
            "dependencies": ["task_1"],
            "priority": "NORMAL"
        },
        {
            "task_id": "task_3",
            "name": "Validate Results",
            "agent_id": "agent_validator",
            "action": "validate",
            "parameters": {},
            "dependencies": ["task_2"],
            "priority": "NORMAL"
        },
        {
            "task_id": "task_4",
            "name": "Save Results",
            "agent_id": "agent_saver",
            "action": "save",
            "parameters": {},
            "dependencies": ["task_3"],
            "priority": "CRITICAL"
        }
    ]
    
    workflow_id = orchestrator.create_workflow(
        name="Data Processing Pipeline",
        tasks=tasks,
        metadata={"project": "demo"}
    )
    
    print("  Created workflow: {}".format(workflow_id))
    print("  Total tasks: {}".format(len(tasks)))
    
    # Example 2: Execute workflow
    print("\n2. Executing Workflow:")
    
    execution_id = orchestrator.execute_workflow(
        workflow_id,
        context={"environment": "production"}
    )
    
    print("  Execution ID: {}".format(execution_id))
    
    # Wait for completion (simplified)
    time.sleep(1)
    
    # Example 3: Check workflow status
    print("\n3. Workflow Status:")
    status = orchestrator.get_workflow_status(workflow_id)
    print(json.dumps(status, indent=2))
    
    # Example 4: Task details
    print("\n4. Task Details:")
    
    for task_id in ["task_1", "task_4"]:
        details = orchestrator.get_task_details(workflow_id, task_id)
        print("\n  {}:".format(task_id))
        print("    Name: {}".format(details["name"]))
        print("    Status: {}".format(details["status"]))
        print("    Agent: {}".format(details["agent_id"]))
        if details.get("duration_seconds"):
            print("    Duration: {:.2f}s".format(details["duration_seconds"]))
    
    # Example 5: Parallel workflow
    print("\n5. Creating Parallel Workflow:")
    
    parallel_tasks = [
        {
            "task_id": "parallel_1",
            "name": "Process Batch 1",
            "agent_id": "agent_1",
            "action": "process_data",
            "parameters": {"batch": 1},
            "dependencies": []
        },
        {
            "task_id": "parallel_2",
            "name": "Process Batch 2",
            "agent_id": "agent_2",
            "action": "process_data",
            "parameters": {"batch": 2},
            "dependencies": []
        },
        {
            "task_id": "parallel_3",
            "name": "Process Batch 3",
            "agent_id": "agent_3",
            "action": "process_data",
            "parameters": {"batch": 3},
            "dependencies": []
        },
        {
            "task_id": "merge",
            "name": "Merge Results",
            "agent_id": "agent_merger",
            "action": "transform",
            "parameters": {},
            "dependencies": ["parallel_1", "parallel_2", "parallel_3"]
        }
    ]
    
    parallel_workflow = orchestrator.create_workflow(
        "Parallel Processing",
        parallel_tasks
    )
    
    print("  Created parallel workflow: {}".format(parallel_workflow))
    
    orchestrator.execute_workflow(parallel_workflow)
    time.sleep(1)
    
    parallel_status = orchestrator.get_workflow_status(parallel_workflow)
    print("  Status: {}".format(parallel_status["status"]))
    print("  Completed tasks: {}".format(
        parallel_status["task_status_counts"].get("completed", 0)
    ))
    
    # Example 6: Statistics
    print("\n6. Orchestrator Statistics:")
    stats = orchestrator.get_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    demonstrate_workflow_orchestration()
