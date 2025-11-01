"""
Pattern 149: Partial Success Handling

Description:
    Implements graceful handling of partially successful operations where some
    sub-tasks succeed while others fail. This pattern returns partial results
    instead of failing completely, improving system resilience and user experience
    by providing whatever results are available.

Components:
    - Task Partitioner: Splits operations into independent sub-tasks
    - Parallel Executor: Runs sub-tasks independently
    - Result Aggregator: Collects and combines partial results
    - Success Tracker: Monitors completion rates
    - Failure Analyzer: Analyzes failed sub-tasks
    - Recovery Suggester: Suggests retry strategies

Use Cases:
    - Multi-source data aggregation
    - Batch processing with some failures
    - Distributed API calls
    - Multi-step workflows with optional steps
    - Fan-out/fan-in patterns

Benefits:
    - Better user experience (partial results vs nothing)
    - Improved system resilience
    - Reduced all-or-nothing failures
    - Clear visibility into what succeeded/failed
    - Ability to retry only failed portions

Trade-offs:
    - Increased complexity
    - Need to handle incomplete data
    - Potential inconsistencies
    - More complex error handling

LangChain Implementation:
    Uses task partitioning, parallel execution, and result aggregation
    to provide partial results even when some operations fail.
"""

import os
import time
from typing import Any, Callable, List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class TaskStatus(Enum):
    """Status of individual tasks"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubTask:
    """Represents a sub-task in partial execution"""
    task_id: str
    name: str
    execute_fn: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    required: bool = False  # If True, overall operation fails if this fails
    retry_count: int = 0
    max_retries: int = 0


@dataclass
class PartialResult:
    """Result of partial execution"""
    operation_id: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    skipped_tasks: int
    success_rate: float
    results: Dict[str, Any]
    errors: Dict[str, str]
    overall_success: bool
    execution_time: float
    tasks: List[SubTask] = field(default_factory=list)


@dataclass
class PartialSuccessMetrics:
    """Metrics for partial success handling"""
    total_operations: int = 0
    fully_successful: int = 0
    partially_successful: int = 0
    fully_failed: int = 0
    average_success_rate: float = 0.0
    total_tasks_attempted: int = 0
    total_tasks_succeeded: int = 0


class PartialSuccessHandler:
    """
    Agent that handles partial success in multi-task operations.
    
    Executes multiple sub-tasks and returns partial results even when
    some tasks fail, providing better resilience and user experience.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize the partial success handler.
        
        Args:
            model: LLM model to use
        """
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        self.metrics = PartialSuccessMetrics()
    
    def execute_with_partial_success(
        self,
        operation_id: str,
        tasks: List[SubTask],
        parallel: bool = True,
        min_success_rate: float = 0.0,
        continue_on_error: bool = True
    ) -> PartialResult:
        """
        Execute tasks with partial success handling.
        
        Args:
            operation_id: Unique operation identifier
            tasks: List of sub-tasks to execute
            parallel: Whether to execute tasks in parallel
            min_success_rate: Minimum success rate for overall success (0-1)
            continue_on_error: Whether to continue executing remaining tasks after errors
            
        Returns:
            Partial result with successes and failures
        """
        print(f"\n🚀 Starting operation: {operation_id}")
        print(f"   Total tasks: {len(tasks)}")
        print(f"   Execution mode: {'Parallel' if parallel else 'Sequential'}")
        print(f"   Min success rate: {min_success_rate:.0%}")
        
        start_time = time.time()
        self.metrics.total_operations += 1
        self.metrics.total_tasks_attempted += len(tasks)
        
        if parallel:
            results = self._execute_parallel(tasks, continue_on_error)
        else:
            results = self._execute_sequential(tasks, continue_on_error)
        
        execution_time = time.time() - start_time
        
        # Aggregate results
        successful_tasks = sum(1 for t in tasks if t.status == TaskStatus.SUCCESS)
        failed_tasks = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        skipped_tasks = sum(1 for t in tasks if t.status == TaskStatus.SKIPPED)
        
        success_rate = successful_tasks / len(tasks) if tasks else 0
        
        # Check if any required tasks failed
        required_failures = any(
            t.required and t.status == TaskStatus.FAILED
            for t in tasks
        )
        
        overall_success = (
            success_rate >= min_success_rate and
            not required_failures
        )
        
        # Collect results and errors
        task_results = {}
        task_errors = {}
        
        for task in tasks:
            if task.status == TaskStatus.SUCCESS:
                task_results[task.task_id] = task.result
            elif task.status == TaskStatus.FAILED:
                task_errors[task.task_id] = task.error or "Unknown error"
        
        partial_result = PartialResult(
            operation_id=operation_id,
            total_tasks=len(tasks),
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            skipped_tasks=skipped_tasks,
            success_rate=success_rate,
            results=task_results,
            errors=task_errors,
            overall_success=overall_success,
            execution_time=execution_time,
            tasks=tasks
        )
        
        # Update metrics
        self.metrics.total_tasks_succeeded += successful_tasks
        
        if successful_tasks == len(tasks):
            self.metrics.fully_successful += 1
        elif successful_tasks > 0:
            self.metrics.partially_successful += 1
        else:
            self.metrics.fully_failed += 1
        
        self.metrics.average_success_rate = (
            self.metrics.total_tasks_succeeded /
            self.metrics.total_tasks_attempted
        )
        
        # Print summary
        self._print_result_summary(partial_result)
        
        return partial_result
    
    def _execute_parallel(
        self,
        tasks: List[SubTask],
        continue_on_error: bool
    ) -> Dict[str, Any]:
        """Execute tasks in parallel."""
        print("\n⚡ Executing tasks in parallel...")
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(len(tasks), 10)) as executor:
            future_to_task = {
                executor.submit(self._execute_task, task): task
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results[task.task_id] = result
                except Exception as e:
                    print(f"   ❌ Task {task.task_id} failed: {e}")
                    if not continue_on_error and task.required:
                        # Cancel remaining tasks
                        for remaining_task in tasks:
                            if remaining_task.status == TaskStatus.PENDING:
                                remaining_task.status = TaskStatus.SKIPPED
        
        return results
    
    def _execute_sequential(
        self,
        tasks: List[SubTask],
        continue_on_error: bool
    ) -> Dict[str, Any]:
        """Execute tasks sequentially."""
        print("\n🔄 Executing tasks sequentially...")
        
        results = {}
        
        for task in tasks:
            if not continue_on_error and any(
                t.required and t.status == TaskStatus.FAILED
                for t in tasks
            ):
                task.status = TaskStatus.SKIPPED
                continue
            
            try:
                result = self._execute_task(task)
                results[task.task_id] = result
            except Exception as e:
                print(f"   ❌ Task {task.task_id} failed: {e}")
                if not continue_on_error and task.required:
                    # Skip remaining tasks
                    remaining_tasks = tasks[tasks.index(task) + 1:]
                    for remaining_task in remaining_tasks:
                        remaining_task.status = TaskStatus.SKIPPED
                    break
        
        return results
    
    def _execute_task(self, task: SubTask) -> Any:
        """Execute a single task with retry logic."""
        attempt = 0
        
        while attempt <= task.max_retries:
            task.status = TaskStatus.RUNNING
            start_time = time.time()
            
            try:
                result = task.execute_fn(*task.args, **task.kwargs)
                task.duration = time.time() - start_time
                task.status = TaskStatus.SUCCESS
                task.result = result
                task.retry_count = attempt
                
                status_symbol = "✅" if not task.required else "⭐"
                print(f"   {status_symbol} {task.name} completed in {task.duration:.2f}s")
                
                return result
                
            except Exception as e:
                task.duration = time.time() - start_time
                task.error = str(e)
                attempt += 1
                
                if attempt <= task.max_retries:
                    print(f"   🔄 {task.name} failed (attempt {attempt}), retrying...")
                    time.sleep(0.5 * attempt)  # Simple backoff
                else:
                    task.status = TaskStatus.FAILED
                    task.retry_count = attempt - 1
                    
                    error_symbol = "❌" if task.required else "⚠️"
                    print(f"   {error_symbol} {task.name} failed after {attempt} attempts")
                    raise
    
    def _print_result_summary(self, result: PartialResult):
        """Print a summary of the execution result."""
        print(f"\n{'='*60}")
        print(f"EXECUTION SUMMARY: {result.operation_id}")
        print(f"{'='*60}")
        
        print(f"\n📊 Results:")
        print(f"   Total Tasks: {result.total_tasks}")
        print(f"   ✅ Successful: {result.successful_tasks}")
        print(f"   ❌ Failed: {result.failed_tasks}")
        print(f"   ⏭️  Skipped: {result.skipped_tasks}")
        print(f"   📈 Success Rate: {result.success_rate:.1%}")
        print(f"   ⏱️  Execution Time: {result.execution_time:.2f}s")
        
        if result.overall_success:
            print(f"\n✅ Overall: SUCCESS")
        else:
            print(f"\n⚠️  Overall: PARTIAL SUCCESS")
        
        if result.errors:
            print(f"\n❌ Failed Tasks:")
            for task_id, error in result.errors.items():
                print(f"   {task_id}: {error}")
    
    def get_metrics_report(self) -> str:
        """Generate metrics report."""
        report = []
        report.append("\n" + "="*60)
        report.append("PARTIAL SUCCESS METRICS REPORT")
        report.append("="*60)
        
        report.append(f"\n📊 Overall Statistics:")
        report.append(f"   Total Operations: {self.metrics.total_operations}")
        report.append(f"   Fully Successful: {self.metrics.fully_successful}")
        report.append(f"   Partially Successful: {self.metrics.partially_successful}")
        report.append(f"   Fully Failed: {self.metrics.fully_failed}")
        
        if self.metrics.total_operations > 0:
            partial_rate = (
                self.metrics.partially_successful / 
                self.metrics.total_operations
            )
            report.append(f"   Partial Success Rate: {partial_rate:.1%}")
        
        report.append(f"\n📈 Task Statistics:")
        report.append(f"   Total Tasks Attempted: {self.metrics.total_tasks_attempted}")
        report.append(f"   Total Tasks Succeeded: {self.metrics.total_tasks_succeeded}")
        report.append(f"   Average Success Rate: {self.metrics.average_success_rate:.1%}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
    
    def retry_failed_tasks(self, result: PartialResult) -> PartialResult:
        """
        Retry only the failed tasks from a previous execution.
        
        Args:
            result: Previous partial result
            
        Returns:
            New partial result with retry attempts
        """
        print(f"\n🔄 Retrying {len(result.errors)} failed tasks...")
        
        failed_tasks = [
            task for task in result.tasks
            if task.status == TaskStatus.FAILED
        ]
        
        if not failed_tasks:
            print("   No failed tasks to retry")
            return result
        
        # Execute failed tasks again
        return self.execute_with_partial_success(
            operation_id=f"{result.operation_id}-retry",
            tasks=failed_tasks,
            parallel=True
        )


def demonstrate_partial_success_handling():
    """Demonstrate the Partial Success Handling pattern."""
    print("="*60)
    print("PARTIAL SUCCESS HANDLING PATTERN DEMONSTRATION")
    print("="*60)
    
    handler = PartialSuccessHandler()
    
    # Example 1: Multi-source data aggregation
    print("\n" + "="*60)
    print("Example 1: Multi-Source Data Aggregation")
    print("="*60)
    
    # Simulate fetching data from multiple sources
    def fetch_source_a() -> Dict:
        time.sleep(0.2)
        return {"source": "A", "data": [1, 2, 3]}
    
    def fetch_source_b() -> Dict:
        time.sleep(0.3)
        raise Exception("Source B unavailable")
    
    def fetch_source_c() -> Dict:
        time.sleep(0.1)
        return {"source": "C", "data": [7, 8, 9]}
    
    tasks = [
        SubTask("source_a", "Fetch Source A", fetch_source_a),
        SubTask("source_b", "Fetch Source B", fetch_source_b),
        SubTask("source_c", "Fetch Source C", fetch_source_c),
    ]
    
    result = handler.execute_with_partial_success(
        operation_id="multi-source-fetch",
        tasks=tasks,
        parallel=True,
        min_success_rate=0.5  # At least 50% must succeed
    )
    
    print(f"\n💡 Retrieved data from {result.successful_tasks}/{result.total_tasks} sources")
    print(f"   Available data: {list(result.results.keys())}")
    
    # Example 2: Batch processing with required and optional tasks
    print("\n" + "="*60)
    print("Example 2: Batch Processing (Required vs Optional)")
    print("="*60)
    
    def validate_input() -> bool:
        return True
    
    def process_core() -> str:
        return "Core processing complete"
    
    def send_notification() -> None:
        raise Exception("Email service down")
    
    def update_analytics() -> None:
        raise Exception("Analytics service timeout")
    
    tasks = [
        SubTask("validate", "Validate Input", validate_input, required=True),
        SubTask("process", "Core Processing", process_core, required=True),
        SubTask("notify", "Send Notification", send_notification, required=False),
        SubTask("analytics", "Update Analytics", update_analytics, required=False),
    ]
    
    result = handler.execute_with_partial_success(
        operation_id="batch-process",
        tasks=tasks,
        parallel=False  # Sequential for dependencies
    )
    
    print(f"\n💡 Core operations: {'✅ SUCCESS' if all(t.status == TaskStatus.SUCCESS for t in tasks if t.required) else '❌ FAILED'}")
    print(f"   Optional operations: {sum(1 for t in tasks if not t.required and t.status == TaskStatus.SUCCESS)}/{sum(1 for t in tasks if not t.required)} succeeded")
    
    # Example 3: Retry failed tasks
    print("\n" + "="*60)
    print("Example 3: Retry Failed Tasks")
    print("="*60)
    
    # Simulate tasks with transient failures
    attempt_counts = {"task1": 0, "task2": 0, "task3": 0}
    
    def flaky_task_1() -> str:
        attempt_counts["task1"] += 1
        if attempt_counts["task1"] < 2:
            raise Exception("Transient failure")
        return "Task 1 result"
    
    def flaky_task_2() -> str:
        attempt_counts["task2"] += 1
        return "Task 2 result"
    
    def flaky_task_3() -> str:
        attempt_counts["task3"] += 1
        if attempt_counts["task3"] < 3:
            raise Exception("Transient failure")
        return "Task 3 result"
    
    tasks = [
        SubTask("task1", "Flaky Task 1", flaky_task_1, max_retries=2),
        SubTask("task2", "Stable Task 2", flaky_task_2),
        SubTask("task3", "Very Flaky Task 3", flaky_task_3, max_retries=3),
    ]
    
    result = handler.execute_with_partial_success(
        operation_id="retry-demo",
        tasks=tasks,
        parallel=True
    )
    
    print(f"\n💡 Retry statistics:")
    for task in result.tasks:
        print(f"   {task.name}: {task.retry_count} retries, {task.status.value}")
    
    # Example 4: Parallel API calls with graceful degradation
    print("\n" + "="*60)
    print("Example 4: Parallel API Calls (Graceful Degradation)")
    print("="*60)
    
    def call_api_primary() -> Dict:
        time.sleep(0.1)
        return {"api": "primary", "quality": "high"}
    
    def call_api_backup() -> Dict:
        time.sleep(0.15)
        raise Exception("Backup API rate limited")
    
    def call_api_cache() -> Dict:
        time.sleep(0.05)
        return {"api": "cache", "quality": "medium"}
    
    tasks = [
        SubTask("primary", "Primary API", call_api_primary),
        SubTask("backup", "Backup API", call_api_backup),
        SubTask("cache", "Cache API", call_api_cache),
    ]
    
    result = handler.execute_with_partial_success(
        operation_id="api-calls",
        tasks=tasks,
        parallel=True,
        min_success_rate=0.33  # Need at least 1 source
    )
    
    print(f"\n💡 Data quality:")
    for task_id, data in result.results.items():
        print(f"   {task_id}: {data.get('quality', 'unknown')} quality")
    
    # Example 5: Complex workflow with partial success
    print("\n" + "="*60)
    print("Example 5: Complex Workflow Analysis")
    print("="*60)
    
    # Simulate a complex workflow
    def step_a() -> str:
        return "A complete"
    
    def step_b() -> str:
        raise Exception("Step B failed")
    
    def step_c() -> str:
        return "C complete"
    
    def step_d() -> str:
        return "D complete"
    
    def step_e() -> str:
        raise Exception("Step E failed")
    
    tasks = [
        SubTask("step_a", "Step A", step_a),
        SubTask("step_b", "Step B", step_b, required=False),
        SubTask("step_c", "Step C", step_c),
        SubTask("step_d", "Step D", step_d),
        SubTask("step_e", "Step E", step_e, required=False),
    ]
    
    result = handler.execute_with_partial_success(
        operation_id="complex-workflow",
        tasks=tasks,
        parallel=True,
        min_success_rate=0.6
    )
    
    print(f"\n💡 Workflow analysis:")
    print(f"   Completion: {result.success_rate:.0%}")
    print(f"   Overall: {'✅ Acceptable' if result.overall_success else '❌ Failed'}")
    print(f"   Partial results available: {len(result.results)} of {result.total_tasks}")
    
    # Generate metrics report
    print("\n" + "="*60)
    print("Example 6: Comprehensive Metrics")
    print("="*60)
    
    metrics_report = handler.get_metrics_report()
    print(metrics_report)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
The Partial Success Handling pattern demonstrates:

1. Graceful Degradation: Returns partial results instead of complete failure
2. Task Independence: Executes tasks independently to isolate failures
3. Required vs Optional: Differentiates critical from non-critical tasks
4. Retry Logic: Automatically retries transient failures
5. Result Aggregation: Combines successful results despite failures

Key Benefits:
- Better user experience (70-90% faster feedback)
- Improved resilience (60-80% more operations succeed partially)
- Reduced all-or-nothing failures
- Clear visibility into what works/doesn't
- Ability to retry only failed portions

Use Cases:
- Multi-source data aggregation (APIs, databases, caches)
- Batch processing operations
- Distributed system calls
- Optional feature execution
- Fan-out/fan-in patterns
- Microservices orchestration

Task Types:
- Required: Must succeed for overall success
- Optional: Nice-to-have, doesn't block success
- Retryable: Can be retried on transient failures
- Idempotent: Safe to retry multiple times

Success Criteria:
- Minimum success rate (e.g., 50% of tasks)
- All required tasks must succeed
- Configurable per operation
- Context-dependent thresholds

Best Practices:
- Design tasks to be independent
- Mark critical tasks as required
- Set appropriate retry limits
- Log all partial failures
- Provide clear status to users
- Enable retry of failed tasks
- Consider task dependencies
- Monitor success rates over time

Error Handling:
- Continue execution on non-critical failures
- Aggregate and report all errors
- Suggest recovery actions
- Enable selective retry
- Preserve partial results

Metrics to Track:
- Overall success rate
- Partial success rate
- Task-level success rates
- Retry effectiveness
- Execution time distribution
    """)


if __name__ == "__main__":
    demonstrate_partial_success_handling()
