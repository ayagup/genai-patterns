"""
Asynchronous Agent Pattern for Agentic AI

This pattern implements asynchronous agents that can handle multiple tasks
concurrently, improving throughput and responsiveness in production systems.

Key Concepts:
1. Async/Await - Non-blocking I/O operations
2. Concurrent Task Execution - Handle multiple requests simultaneously
3. Task Queueing - Manage incoming requests efficiently
4. Background Processing - Long-running tasks without blocking
5. Event-Driven Architecture - React to events asynchronously

Use Cases:
- High-throughput API services
- Real-time agent systems
- Multiple concurrent conversations
- Batch processing with parallelism
- I/O-bound operations (API calls, DB queries)
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Coroutine, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time


class TaskStatus(Enum):
    """Status of async tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents an async task"""
    id: str
    name: str
    input_data: Any
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def duration(self) -> Optional[float]:
        """Calculate task duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class AsyncAgent:
    """
    Agent that processes tasks asynchronously.
    
    Handles multiple tasks concurrently using asyncio, improving
    throughput for I/O-bound operations.
    """
    
    def __init__(
        self,
        name: str = "AsyncAgent",
        max_concurrent_tasks: int = 5,
        timeout: float = 30.0
    ):
        self.name = name
        self.max_concurrent_tasks = max_concurrent_tasks
        self.timeout = timeout
        self.tasks: Dict[str, Task] = {}
        self.task_counter = 0
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def process_task(self, input_data: Any) -> Any:
        """
        Process a single task (placeholder for actual logic).
        
        In real implementation, this would call LLM APIs, databases, etc.
        """
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        # Simulate some work
        result = f"Processed: {input_data}"
        
        return result
    
    async def execute_task(self, task: Task) -> Task:
        """
        Execute a task asynchronously with error handling.
        
        Args:
            task: Task to execute
        
        Returns:
            Completed task with result or error
        """
        async with self.semaphore:  # Limit concurrent tasks
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            try:
                print(f"[{self.name}] Starting task {task.id}: {task.name}")
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    self.process_task(task.input_data),
                    timeout=self.timeout
                )
                
                task.result = result
                task.status = TaskStatus.COMPLETED
                print(f"[{self.name}] Completed task {task.id}")
                
            except asyncio.TimeoutError:
                task.error = f"Task timed out after {self.timeout}s"
                task.status = TaskStatus.FAILED
                print(f"[{self.name}] Task {task.id} timed out")
                
            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
                print(f"[{self.name}] Task {task.id} failed: {e}")
                
            finally:
                task.completed_at = datetime.now()
            
            return task
    
    async def submit_task(
        self,
        name: str,
        input_data: Any
    ) -> str:
        """
        Submit a task for async execution.
        
        Args:
            name: Task name
            input_data: Input for the task
        
        Returns:
            Task ID
        """
        self.task_counter += 1
        task_id = f"{self.name}_task_{self.task_counter}"
        
        task = Task(
            id=task_id,
            name=name,
            input_data=input_data
        )
        
        self.tasks[task_id] = task
        
        # Start task execution in background
        asyncio.create_task(self.execute_task(task))
        
        print(f"[{self.name}] Submitted task {task_id}")
        return task_id
    
    async def submit_batch(
        self,
        tasks: List[Tuple[str, Any]]
    ) -> List[str]:
        """
        Submit multiple tasks concurrently.
        
        Args:
            tasks: List of (name, input_data) tuples
        
        Returns:
            List of task IDs
        """
        task_ids = []
        for name, input_data in tasks:
            task_id = await self.submit_task(name, input_data)
            task_ids.append(task_id)
        
        return task_ids
    
    async def wait_for_task(
        self,
        task_id: str,
        poll_interval: float = 0.1
    ) -> Task:
        """
        Wait for a task to complete.
        
        Args:
            task_id: ID of task to wait for
            poll_interval: How often to check status
        
        Returns:
            Completed task
        """
        while True:
            task = self.tasks.get(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return task
            
            await asyncio.sleep(poll_interval)
    
    async def wait_for_all(
        self,
        task_ids: List[str]
    ) -> List[Task]:
        """
        Wait for multiple tasks to complete.
        
        Args:
            task_ids: List of task IDs
        
        Returns:
            List of completed tasks
        """
        tasks = await asyncio.gather(
            *[self.wait_for_task(task_id) for task_id in task_ids],
            return_exceptions=True
        )
        return tasks
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task"""
        task = self.tasks.get(task_id)
        return task.status if task else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        running = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)
        
        avg_duration = 0.0
        completed_tasks = [t for t in self.tasks.values() if t.duration()]
        if completed_tasks:
            durations: List[float] = [d for t in completed_tasks if (d := t.duration()) is not None]
            avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        return {
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": total - completed - failed - running,
            "success_rate": completed / total if total > 0 else 0,
            "avg_duration_seconds": avg_duration
        }


class ConversationAgent(AsyncAgent):
    """
    Async agent specialized for handling multiple conversations.
    
    Can maintain many concurrent user conversations efficiently.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="ConversationAgent", **kwargs)
        self.conversations: Dict[str, List[str]] = {}
    
    async def process_task(self, input_data: Any) -> Any:
        """Process a conversation message"""
        user_id = input_data.get("user_id")
        message = input_data.get("message")
        
        # Simulate LLM call with delay
        await asyncio.sleep(0.3)
        
        # Store conversation history
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        self.conversations[user_id].append(f"User: {message}")
        
        # Generate response
        response = f"Response to: {message}"
        self.conversations[user_id].append(f"Agent: {response}")
        
        return {
            "user_id": user_id,
            "response": response,
            "conversation_length": len(self.conversations[user_id])
        }
    
    async def handle_message(
        self,
        user_id: str,
        message: str
    ) -> str:
        """
        Handle a user message asynchronously.
        
        Args:
            user_id: User identifier
            message: User message
        
        Returns:
            Task ID
        """
        return await self.submit_task(
            name=f"message_from_{user_id}",
            input_data={"user_id": user_id, "message": message}
        )


class BatchProcessingAgent(AsyncAgent):
    """
    Async agent for efficient batch processing.
    
    Processes large batches of items concurrently.
    """
    
    def __init__(self, **kwargs):
        super().__init__(name="BatchAgent", max_concurrent_tasks=10, **kwargs)
    
    async def process_task(self, input_data: Any) -> Any:
        """Process a batch item"""
        item_id = input_data.get("id")
        data = input_data.get("data")
        
        # Simulate processing
        await asyncio.sleep(0.2)
        
        return {
            "id": item_id,
            "processed": True,
            "result": f"Processed {data}"
        }
    
    async def process_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> List[Any]:
        """
        Process a batch of items concurrently.
        
        Args:
            items: List of items to process
        
        Returns:
            List of results
        """
        # Submit all items
        task_ids = []
        for item in items:
            task_id = await self.submit_task(
                name=f"batch_item_{item['id']}",
                input_data=item
            )
            task_ids.append(task_id)
        
        # Wait for all to complete
        tasks = await self.wait_for_all(task_ids)
        
        # Extract results
        results = [t.result for t in tasks if t.status == TaskStatus.COMPLETED]
        
        return results


async def demonstrate_async_agent():
    """Demonstrate async agent capabilities"""
    
    print("Asynchronous Agent Pattern Demonstration")
    print("=" * 70)
    
    # 1. Basic async agent
    print("\n1. Basic Async Agent")
    print("-" * 70)
    
    agent = AsyncAgent(name="BasicAgent", max_concurrent_tasks=3)
    
    # Submit multiple tasks
    task_ids = []
    for i in range(5):
        task_id = await agent.submit_task(
            name=f"query_{i}",
            input_data=f"Question {i}"
        )
        task_ids.append(task_id)
    
    # Wait for all tasks
    tasks = await agent.wait_for_all(task_ids)
    
    print(f"\nCompleted {len(tasks)} tasks")
    stats = agent.get_statistics()
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Avg duration: {stats['avg_duration_seconds']:.3f}s")
    
    # 2. Conversation agent with multiple users
    print("\n\n2. Conversation Agent (Multiple Users)")
    print("-" * 70)
    
    conv_agent = ConversationAgent(max_concurrent_tasks=5)
    
    # Simulate multiple users sending messages
    users = ["user1", "user2", "user3"]
    message_task_ids = []
    
    for user in users:
        for msg_num in range(2):
            task_id = await conv_agent.handle_message(
                user_id=user,
                message=f"Message {msg_num} from {user}"
            )
            message_task_ids.append(task_id)
    
    # Wait for all messages to be processed
    await conv_agent.wait_for_all(message_task_ids)
    
    print(f"\nProcessed {len(message_task_ids)} messages from {len(users)} users")
    for user in users:
        print(f"{user}: {len(conv_agent.conversations[user])} turns")
    
    # 3. Batch processing agent
    print("\n\n3. Batch Processing Agent")
    print("-" * 70)
    
    batch_agent = BatchProcessingAgent()
    
    # Create batch of items
    batch_items = [
        {"id": i, "data": f"item_{i}"}
        for i in range(20)
    ]
    
    start_time = time.time()
    results = await batch_agent.process_batch(batch_items)
    duration = time.time() - start_time
    
    print(f"\nProcessed {len(results)} items in {duration:.2f}s")
    print(f"Throughput: {len(results)/duration:.1f} items/sec")
    
    stats = batch_agent.get_statistics()
    print(f"Success rate: {stats['success_rate']:.1%}")
    
    # 4. Performance comparison
    print("\n\n4. Sync vs Async Performance")
    print("-" * 70)
    
    async def sync_style(n: int):
        """Simulate synchronous processing"""
        results = []
        start = time.time()
        for i in range(n):
            await asyncio.sleep(0.1)  # Simulate work
            results.append(f"result_{i}")
        return time.time() - start, results
    
    async def async_style(n: int):
        """Asynchronous processing"""
        start = time.time()
        tasks = [asyncio.sleep(0.1) for _ in range(n)]
        await asyncio.gather(*tasks)
        duration = time.time() - start
        return duration, [f"result_{i}" for i in range(n)]
    
    n_tasks = 10
    
    sync_duration, _ = await sync_style(n_tasks)
    print(f"Synchronous: {n_tasks} tasks in {sync_duration:.2f}s")
    
    async_duration, _ = await async_style(n_tasks)
    print(f"Asynchronous: {n_tasks} tasks in {async_duration:.2f}s")
    print(f"Speedup: {sync_duration/async_duration:.1f}x")


def run_demonstration():
    """Run async demonstration"""
    asyncio.run(demonstrate_async_agent())
    
    print("\n\n" + "="*70)
    print("Asynchronous Agent Pattern Summary")
    print("="*70)
    print("""
Key Benefits:
1. Higher throughput for I/O-bound tasks
2. Better resource utilization
3. Improved responsiveness
4. Concurrent request handling
5. Scalability for multiple users

Implementation Patterns:
- async/await: Non-blocking operations
- asyncio.gather(): Run multiple tasks concurrently
- Semaphores: Limit concurrent tasks
- Task queues: Manage workload
- Event loops: Handle async execution

Best Practices:
- Use for I/O-bound operations (APIs, DBs, files)
- Limit concurrent tasks to prevent overload
- Implement timeouts for all operations
- Handle exceptions gracefully
- Monitor task queue depth
- Use connection pooling
- Profile and optimize

Common Pitfalls:
- CPU-bound tasks don't benefit (use multiprocessing)
- Blocking calls break async (use run_in_executor)
- Shared state needs locks
- Memory usage with many tasks
- Debugging can be challenging

Use Cases:
- High-traffic API services
- Multi-user chat applications
- Batch data processing
- Real-time systems
- Microservices
- Web scraping
- Parallel LLM calls

Performance Gains:
- 5-10x for I/O-bound operations
- Near-linear scaling with concurrency
- Reduced latency for users
- Better server utilization
""")


if __name__ == "__main__":
    run_demonstration()
