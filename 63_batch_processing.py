"""
Pattern 79: Batch Processing Agent
Description:
    Efficiently processes multiple tasks in batches to optimize resource
    utilization, reduce overhead, and improve throughput.
Use Cases:
    - Bulk data processing
    - Scheduled job execution
    - High-throughput systems
    - Cost optimization
Key Features:
    - Dynamic batch sizing
    - Priority-based batching
    - Error handling and retry
    - Progress tracking
Example:
    >>> agent = BatchProcessingAgent(batch_size=10)
    >>> agent.add_task(task1)
    >>> agent.add_task(task2)
    >>> results = agent.process_batch()
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import time
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
class TaskPriority(Enum):
    """Priority levels for tasks"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
class TaskStatus(Enum):
    """Status of a task"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
class BatchStrategy(Enum):
    """Batching strategies"""
    SIZE_BASED = "size_based"  # Fixed size batches
    TIME_BASED = "time_based"  # Time window batches
    PRIORITY_BASED = "priority_based"  # Group by priority
    DYNAMIC = "dynamic"  # Adaptive batching
@dataclass
class Task:
    """Individual task in the batch"""
    task_id: str
    data: Any
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
@dataclass
class Batch:
    """A batch of tasks"""
    batch_id: str
    tasks: List[Task]
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    batch_size: int = 10
    max_wait_time: float = 5.0  # seconds
    parallel_workers: int = 4
    strategy: BatchStrategy = BatchStrategy.SIZE_BASED
    enable_retries: bool = True
    enable_priority: bool = True
class BatchProcessingAgent:
    """
    Agent that processes tasks in batches
    Features:
    - Multiple batching strategies
    - Priority-based processing
    - Automatic retry logic
    - Progress tracking
    """
    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        processor: Optional[Callable] = None
    ):
        self.config = config or BatchConfig()
        self.processor = processor or self._default_processor
        self.task_queue: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        self.active_batches: Dict[str, Batch] = {}
        self.completed_batches: List[Batch] = []
        self.lock = threading.Lock()
        self.batch_counter = 0
        self.task_counter = 0
        self.running = False
        self.processing_thread: Optional[threading.Thread] = None
    def add_task(
        self,
        data: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a task to the queue
        Args:
            data: Task data
            priority: Task priority
            metadata: Optional metadata
        Returns:
            Task ID
        """
        with self.lock:
            self.task_counter += 1
            task_id = f"task_{self.task_counter}_{int(time.time())}"
            task = Task(
                task_id=task_id,
                data=data,
                priority=priority,
                metadata=metadata or {}
            )
            self.task_queue[priority].append(task)
        return task_id
    def add_tasks_bulk(
        self,
        tasks_data: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add multiple tasks at once
        Args:
            tasks_data: List of {data, priority?, metadata?}
        Returns:
            List of task IDs
        """
        task_ids = []
        for task_data in tasks_data:
            task_id = self.add_task(
                data=task_data['data'],
                priority=task_data.get('priority', TaskPriority.NORMAL),
                metadata=task_data.get('metadata')
            )
            task_ids.append(task_id)
        return task_ids
    def process_batch(self, batch_id: Optional[str] = None) -> Batch:
        """
        Process a single batch
        Args:
            batch_id: Optional specific batch to process
        Returns:
            Processed batch
        """
        # Create batch
        if batch_id and batch_id in self.active_batches:
            batch = self.active_batches[batch_id]
        else:
            batch = self._create_batch()
        if not batch.tasks:
            return batch
        batch.started_at = time.time()
        # Process based on strategy
        if self.config.parallel_workers > 1:
            self._process_batch_parallel(batch)
        else:
            self._process_batch_sequential(batch)
        batch.completed_at = time.time()
        # Move to completed
        with self.lock:
            if batch.batch_id in self.active_batches:
                del self.active_batches[batch.batch_id]
            self.completed_batches.append(batch)
        return batch
    def _create_batch(self) -> Batch:
        """Create a new batch from queued tasks"""
        with self.lock:
            self.batch_counter += 1
            batch_id = f"batch_{self.batch_counter}_{int(time.time())}"
            tasks = []
            if self.config.strategy == BatchStrategy.PRIORITY_BASED:
                # Process highest priority first
                for priority in sorted(TaskPriority, key=lambda x: x.value, reverse=True):
                    while len(tasks) < self.config.batch_size and self.task_queue[priority]:
                        tasks.append(self.task_queue[priority].popleft())
                    if len(tasks) >= self.config.batch_size:
                        break
            else:
                # Mix priorities
                total_needed = self.config.batch_size
                for priority in TaskPriority:
                    queue = self.task_queue[priority]
                    while queue and len(tasks) < total_needed:
                        tasks.append(queue.popleft())
            batch = Batch(
                batch_id=batch_id,
                tasks=tasks,
                total_tasks=len(tasks)
            )
            self.active_batches[batch_id] = batch
        return batch
    def _process_batch_sequential(self, batch: Batch):
        """Process batch tasks sequentially"""
        for task in batch.tasks:
            self._process_single_task(task, batch)
    def _process_batch_parallel(self, batch: Batch):
        """Process batch tasks in parallel"""
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = {
                executor.submit(self._process_single_task, task, batch): task
                for task in batch.tasks
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    task = futures[future]
                    task.error = str(e)
                    task.status = TaskStatus.FAILED
    def _process_single_task(self, task: Task, batch: Batch):
        """Process a single task"""
        task.status = TaskStatus.PROCESSING
        task.started_at = time.time()
        try:
            # Process the task
            result = self.processor(task.data, task.metadata)
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            with self.lock:
                batch.completed_tasks += 1
        except Exception as e:
            task.error = str(e)
            # Retry logic
            if self.config.enable_retries and task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                # Re-add to queue
                with self.lock:
                    self.task_queue[task.priority].append(task)
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                with self.lock:
                    batch.failed_tasks += 1
    def _default_processor(self, data: Any, metadata: Dict[str, Any]) -> Any:
        """Default task processor (override this)"""
        # Simulate processing
        time.sleep(0.01)
        # Simulate occasional failures
        import random
        if random.random() < 0.05:
            raise Exception("Simulated processing error")
        return f"Processed: {data}"
    def start_auto_processing(self):
        """Start automatic batch processing in background"""
        if self.running:
            return
        self.running = True
        self.processing_thread = threading.Thread(
            target=self._auto_processing_loop,
            daemon=True
        )
        self.processing_thread.start()
    def stop_auto_processing(self):
        """Stop automatic batch processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
    def _auto_processing_loop(self):
        """Background loop for automatic batch processing"""
        while self.running:
            # Check if we should process a batch
            should_process = False
            with self.lock:
                total_tasks = sum(len(q) for q in self.task_queue.values())
                if self.config.strategy == BatchStrategy.SIZE_BASED:
                    should_process = total_tasks >= self.config.batch_size
                elif self.config.strategy == BatchStrategy.TIME_BASED:
                    # Check oldest task
                    oldest_task_time = None
                    for queue in self.task_queue.values():
                        if queue:
                            oldest_task_time = queue[0].created_at
                            break
                    if oldest_task_time:
                        wait_time = time.time() - oldest_task_time
                        should_process = wait_time >= self.config.max_wait_time
                elif self.config.strategy == BatchStrategy.DYNAMIC:
                    # Adaptive: process if batch is full OR wait time exceeded
                    should_process = total_tasks >= self.config.batch_size
                    if not should_process and total_tasks > 0:
                        oldest_task_time = None
                        for queue in self.task_queue.values():
                            if queue:
                                oldest_task_time = queue[0].created_at
                                break
                        if oldest_task_time:
                            wait_time = time.time() - oldest_task_time
                            should_process = wait_time >= self.config.max_wait_time
            if should_process:
                self.process_batch()
            time.sleep(0.1)  # Small sleep to prevent tight loop
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        with self.lock:
            queue_sizes = {
                priority.name: len(queue)
                for priority, queue in self.task_queue.items()
            }
            total_queued = sum(queue_sizes.values())
            active_tasks = sum(
                len(batch.tasks) for batch in self.active_batches.values()
            )
        return {
            'total_queued': total_queued,
            'queue_by_priority': queue_sizes,
            'active_batches': len(self.active_batches),
            'active_tasks': active_tasks,
            'completed_batches': len(self.completed_batches)
        }
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        if not self.completed_batches:
            return {'message': 'No completed batches yet'}
        total_tasks = sum(b.total_tasks for b in self.completed_batches)
        total_completed = sum(b.completed_tasks for b in self.completed_batches)
        total_failed = sum(b.failed_tasks for b in self.completed_batches)
        processing_times = []
        for batch in self.completed_batches:
            if batch.started_at and batch.completed_at:
                processing_times.append(batch.completed_at - batch.started_at)
        avg_processing_time = (
            sum(processing_times) / len(processing_times)
            if processing_times else 0
        )
        return {
            'total_batches': len(self.completed_batches),
            'total_tasks_processed': total_tasks,
            'successful_tasks': total_completed,
            'failed_tasks': total_failed,
            'success_rate': total_completed / total_tasks if total_tasks > 0 else 0,
            'avg_batch_processing_time': avg_processing_time,
            'avg_tasks_per_batch': total_tasks / len(self.completed_batches),
            'throughput': total_completed / sum(processing_times) if processing_times else 0
        }
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        # Check active batches
        for batch in self.active_batches.values():
            for task in batch.tasks:
                if task.task_id == task_id:
                    return self._task_to_dict(task, batch.batch_id)
        # Check completed batches
        for batch in self.completed_batches:
            for task in batch.tasks:
                if task.task_id == task_id:
                    return self._task_to_dict(task, batch.batch_id)
        # Check queues
        for priority, queue in self.task_queue.items():
            for task in queue:
                if task.task_id == task_id:
                    return self._task_to_dict(task, None)
        return None
    def _task_to_dict(self, task: Task, batch_id: Optional[str]) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            'task_id': task.task_id,
            'batch_id': batch_id,
            'status': task.status.value,
            'priority': task.priority.name,
            'retry_count': task.retry_count,
            'created_at': task.created_at,
            'started_at': task.started_at,
            'completed_at': task.completed_at,
            'processing_time': (
                task.completed_at - task.started_at
                if task.started_at and task.completed_at else None
            ),
            'result': task.result,
            'error': task.error
        }
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """Wait for all tasks to complete"""
        start_time = time.time()
        while True:
            status = self.get_queue_status()
            if status['total_queued'] == 0 and status['active_batches'] == 0:
                return True
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)
def main():
    """Demonstrate batch processing pattern"""
    print("=" * 60)
    print("Batch Processing Agent Demonstration")
    print("=" * 60)
    print("\n1. Basic Batch Processing")
    print("-" * 60)
    # Create agent
    agent = BatchProcessingAgent(
        config=BatchConfig(batch_size=5, parallel_workers=2)
    )
    # Add tasks
    task_ids = []
    for i in range(15):
        task_id = agent.add_task(
            data=f"task_data_{i}",
            priority=TaskPriority.NORMAL
        )
        task_ids.append(task_id)
    print(f"Added {len(task_ids)} tasks")
    status = agent.get_queue_status()
    print(f"Queued tasks: {status['total_queued']}")
    # Process first batch
    print("\nProcessing first batch...")
    batch = agent.process_batch()
    print(f"Batch {batch.batch_id}:")
    print(f"  Total tasks: {batch.total_tasks}")
    print(f"  Completed: {batch.completed_tasks}")
    print(f"  Failed: {batch.failed_tasks}")
    print(f"  Processing time: {batch.completed_at - batch.started_at:.3f}s")
    print("\n" + "=" * 60)
    print("2. Priority-Based Processing")
    print("=" * 60)
    priority_agent = BatchProcessingAgent(
        config=BatchConfig(
            batch_size=10,
            strategy=BatchStrategy.PRIORITY_BASED
        )
    )
    # Add tasks with different priorities
    priorities = [
        (TaskPriority.LOW, 5),
        (TaskPriority.NORMAL, 5),
        (TaskPriority.HIGH, 3),
        (TaskPriority.CRITICAL, 2)
    ]
    for priority, count in priorities:
        for i in range(count):
            priority_agent.add_task(
                data=f"{priority.name}_task_{i}",
                priority=priority
            )
    status = priority_agent.get_queue_status()
    print("\nQueue status:")
    for priority, count in status['queue_by_priority'].items():
        print(f"  {priority}: {count} tasks")
    # Process batch (should prioritize CRITICAL and HIGH)
    batch = priority_agent.process_batch()
    print(f"\nProcessed batch with {batch.total_tasks} tasks:")
    priority_counts = defaultdict(int)
    for task in batch.tasks:
        priority_counts[task.priority.name] += 1
    for priority, count in sorted(priority_counts.items()):
        print(f"  {priority}: {count} tasks")
    print("\n" + "=" * 60)
    print("3. Automatic Background Processing")
    print("=" * 60)
    auto_agent = BatchProcessingAgent(
        config=BatchConfig(
            batch_size=5,
            strategy=BatchStrategy.DYNAMIC,
            max_wait_time=2.0
        )
    )
    # Start auto-processing
    auto_agent.start_auto_processing()
    print("Started automatic batch processing")
    # Add tasks gradually
    print("\nAdding tasks gradually...")
    for i in range(20):
        auto_agent.add_task(data=f"auto_task_{i}")
        time.sleep(0.1)  # Simulate gradual arrival
        if i % 5 == 0:
            status = auto_agent.get_queue_status()
            print(f"  After {i+1} tasks: "
                  f"{status['completed_batches']} batches completed, "
                  f"{status['total_queued']} queued")
    # Wait for completion
    print("\nWaiting for all tasks to complete...")
    auto_agent.wait_for_completion(timeout=10.0)
    auto_agent.stop_auto_processing()
    stats = auto_agent.get_batch_statistics()
    print(f"\nFinal statistics:")
    print(f"  Total batches: {stats['total_batches']}")
    print(f"  Tasks processed: {stats['total_tasks_processed']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Avg batch time: {stats['avg_batch_processing_time']:.3f}s")
    print(f"  Throughput: {stats['throughput']:.2f} tasks/sec")
    print("\n" + "=" * 60)
    print("4. Bulk Task Addition")
    print("=" * 60)
    bulk_agent = BatchProcessingAgent()
    # Prepare bulk tasks
    bulk_tasks = [
        {
            'data': f"bulk_task_{i}",
            'priority': TaskPriority.HIGH if i % 3 == 0 else TaskPriority.NORMAL,
            'metadata': {'batch_num': i // 10}
        }
        for i in range(50)
    ]
    # Add in bulk
    task_ids = bulk_agent.add_tasks_bulk(bulk_tasks)
    print(f"Added {len(task_ids)} tasks in bulk")
    # Process all batches
    batches_processed = 0
    while bulk_agent.get_queue_status()['total_queued'] > 0:
        bulk_agent.process_batch()
        batches_processed += 1
    print(f"Processed {batches_processed} batches")
    stats = bulk_agent.get_batch_statistics()
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Avg tasks per batch: {stats['avg_tasks_per_batch']:.1f}")
    print("\n" + "=" * 60)
    print("5. Task Status Tracking")
    print("=" * 60)
    # Check specific task status
    sample_task_id = task_ids[0]
    task_status = bulk_agent.get_task_status(sample_task_id)
    if task_status:
        print(f"\nTask {sample_task_id}:")
        print(f"  Status: {task_status['status']}")
        print(f"  Priority: {task_status['priority']}")
        print(f"  Batch: {task_status['batch_id']}")
        print(f"  Processing time: {task_status['processing_time']:.4f}s")
        print(f"  Result: {task_status['result']}")
    print("\n" + "=" * 60)
    print("6. Error Handling and Retries")
    print("=" * 60)
    def failing_processor(data: Any, metadata: Dict[str, Any]) -> Any:
        """Processor that fails sometimes"""
        import random
        if random.random() < 0.3:  # 30% failure rate
            raise Exception(f"Failed to process {data}")
        return f"Success: {data}"
    retry_agent = BatchProcessingAgent(
        config=BatchConfig(
            batch_size=10,
            enable_retries=True,
            parallel_workers=2
        ),
        processor=failing_processor
    )
    # Add tasks
    for i in range(20):
        retry_agent.add_task(data=f"risky_task_{i}")
    # Process batches
    while retry_agent.get_queue_status()['total_queued'] > 0:
        retry_agent.process_batch()
    stats = retry_agent.get_batch_statistics()
    print(f"\nWith retry logic:")
    print(f"  Tasks processed: {stats['total_tasks_processed']}")
    print(f"  Successful: {stats['successful_tasks']}")
    print(f"  Failed: {stats['failed_tasks']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print("\n" + "=" * 60)
    print("Batch Processing demonstration complete!")
    print("=" * 60)
if __name__ == "__main__":
    main()
