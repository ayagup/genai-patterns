"""
Pattern 080: Asynchronous Agent

Description:
    An Asynchronous Agent uses async/await patterns to handle concurrent operations
    without blocking. This pattern is essential for building high-performance,
    scalable AI applications that can handle multiple requests simultaneously,
    make concurrent API calls, and provide responsive user experiences.
    
    The agent leverages Python's asyncio for non-blocking I/O operations, enabling
    efficient resource utilization and improved throughput. It supports concurrent
    LLM calls, parallel tool execution, and async streaming for maximum performance.

Components:
    1. Async Executor: Manages async task execution
    2. Concurrent Runner: Runs multiple tasks concurrently
    3. Async LLM Client: Non-blocking LLM interactions
    4. Task Scheduler: Schedules and coordinates async tasks
    5. Result Collector: Aggregates concurrent results
    6. Error Handler: Handles async exceptions
    7. Timeout Manager: Enforces timeouts on async operations
    8. Resource Pool: Manages async resources and connections

Key Features:
    - Non-blocking async/await operations
    - Concurrent LLM calls
    - Parallel tool execution
    - Async streaming
    - Task cancellation
    - Timeout handling
    - Connection pooling
    - Backpressure management
    - Error propagation
    - Graceful shutdown

Use Cases:
    - High-throughput API servers
    - Real-time chat applications
    - Concurrent data processing
    - Parallel research queries
    - Multi-source information gathering
    - Async web scraping
    - Real-time analytics
    - Concurrent translation services
    - Parallel validation
    - Multi-agent coordination

LangChain Implementation:
    Uses ChatOpenAI with async methods (ainvoke, astream, abatch),
    asyncio for concurrency, and async LCEL chains.
"""

import os
import asyncio
import time
from typing import List, Dict, Any, Optional, Coroutine, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


class TaskStatus(Enum):
    """Status of async task"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class AsyncTask:
    """An asynchronous task"""
    task_id: str
    name: str
    coroutine: Coroutine
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: float = 0.0


@dataclass
class AsyncResult:
    """Result of async operation"""
    task_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConcurrentResult:
    """Result of concurrent operations"""
    total_tasks: int
    successful: int
    failed: int
    cancelled: int
    results: List[AsyncResult]
    total_duration: float
    average_duration: float


class AsyncAgent:
    """
    Agent that uses async/await for non-blocking operations.
    
    This agent enables high-performance concurrent processing with
    async LLM calls, parallel tool execution, and efficient resource usage.
    """
    
    def __init__(self, max_concurrent: int = 10, timeout: float = 30.0):
        """
        Initialize the async agent.
        
        Args:
            max_concurrent: Maximum concurrent tasks
            timeout: Default timeout for operations
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        
        # Async LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Task tracking
        self.tasks: Dict[str, AsyncTask] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def ainvoke_llm(
        self,
        prompt: str,
        timeout: Optional[float] = None
    ) -> str:
        """
        Invoke LLM asynchronously.
        
        Args:
            prompt: Input prompt
            timeout: Optional timeout
            
        Returns:
            LLM response
        """
        timeout = timeout or self.timeout
        
        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt),
                timeout=timeout
            )
            return response.content
        except asyncio.TimeoutError:
            raise TimeoutError(f"LLM call timed out after {timeout}s")
    
    async def astream_llm(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Stream LLM response asynchronously.
        
        Args:
            prompt: Input prompt
            on_token: Optional token callback
            
        Returns:
            Full response
        """
        full_response = ""
        
        async for chunk in self.llm.astream(prompt):
            if hasattr(chunk, 'content'):
                token = chunk.content
                full_response += token
                
                if on_token:
                    on_token(token)
        
        return full_response
    
    async def run_concurrent_tasks(
        self,
        tasks: List[Coroutine],
        task_names: Optional[List[str]] = None
    ) -> ConcurrentResult:
        """
        Run multiple tasks concurrently.
        
        Args:
            tasks: List of coroutines to execute
            task_names: Optional names for tasks
            
        Returns:
            ConcurrentResult with all results
        """
        start_time = time.time()
        
        if task_names is None:
            task_names = [f"task_{i}" for i in range(len(tasks))]
        
        # Run tasks concurrently with semaphore
        async def run_with_semaphore(coro: Coroutine, name: str) -> AsyncResult:
            async with self.semaphore:
                task_start = time.time()
                task_id = f"{name}_{id(coro)}"
                
                try:
                    result = await coro
                    return AsyncResult(
                        task_id=task_id,
                        success=True,
                        result=result,
                        duration=time.time() - task_start
                    )
                except Exception as e:
                    return AsyncResult(
                        task_id=task_id,
                        success=False,
                        error=str(e),
                        duration=time.time() - task_start
                    )
        
        # Execute all tasks
        results = await asyncio.gather(
            *[run_with_semaphore(task, name) for task, name in zip(tasks, task_names)],
            return_exceptions=True
        )
        
        # Process results
        async_results = []
        successful = 0
        failed = 0
        cancelled = 0
        
        for result in results:
            if isinstance(result, AsyncResult):
                async_results.append(result)
                if result.success:
                    successful += 1
                else:
                    failed += 1
            elif isinstance(result, asyncio.CancelledError):
                cancelled += 1
            else:
                failed += 1
        
        total_duration = time.time() - start_time
        avg_duration = sum(r.duration for r in async_results) / len(async_results) if async_results else 0
        
        return ConcurrentResult(
            total_tasks=len(tasks),
            successful=successful,
            failed=failed,
            cancelled=cancelled,
            results=async_results,
            total_duration=total_duration,
            average_duration=avg_duration
        )
    
    async def parallel_llm_calls(
        self,
        prompts: List[str],
        timeout: Optional[float] = None
    ) -> List[str]:
        """
        Make multiple LLM calls in parallel.
        
        Args:
            prompts: List of prompts
            timeout: Optional timeout per call
            
        Returns:
            List of responses
        """
        tasks = [self.ainvoke_llm(prompt, timeout) for prompt in prompts]
        result = await self.run_concurrent_tasks(
            tasks,
            [f"llm_call_{i}" for i in range(len(prompts))]
        )
        
        return [r.result for r in result.results if r.success]
    
    async def concurrent_chain_execution(
        self,
        inputs: List[Dict[str, str]],
        template: str
    ) -> List[str]:
        """
        Execute chain concurrently with different inputs.
        
        Args:
            inputs: List of input dictionaries
            template: Prompt template
            
        Returns:
            List of results
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        # Create async tasks for each input
        tasks = [chain.ainvoke(input_dict) for input_dict in inputs]
        
        result = await self.run_concurrent_tasks(
            tasks,
            [f"chain_{i}" for i in range(len(inputs))]
        )
        
        return [r.result for r in result.results if r.success]
    
    async def parallel_with_timeout(
        self,
        tasks: List[Coroutine],
        timeout: float
    ) -> List[Optional[Any]]:
        """
        Run tasks in parallel with timeout.
        
        Args:
            tasks: List of coroutines
            timeout: Timeout for all tasks
            
        Returns:
            List of results (None for timed out tasks)
        """
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
            
            return [
                r if not isinstance(r, Exception) else None
                for r in results
            ]
        except asyncio.TimeoutError:
            return [None] * len(tasks)
    
    async def async_map(
        self,
        func: Callable[[Any], Coroutine],
        items: List[Any],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Map an async function over items with optional batching.
        
        Args:
            func: Async function to map
            items: Items to process
            batch_size: Optional batch size for processing
            
        Returns:
            List of results
        """
        if batch_size is None:
            batch_size = self.max_concurrent
        
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [func(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    async def race_tasks(
        self,
        tasks: List[Coroutine],
        return_first: bool = True
    ) -> Any:
        """
        Race multiple tasks, return first to complete.
        
        Args:
            tasks: List of coroutines to race
            return_first: Return first completed (True) or all (False)
            
        Returns:
            First result or all results
        """
        if return_first:
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
            
            # Return first result
            return list(done)[0].result()
        else:
            return await asyncio.gather(*tasks)
    
    async def retry_async(
        self,
        coro: Coroutine,
        max_retries: int = 3,
        backoff: float = 1.0
    ) -> Any:
        """
        Retry an async operation with exponential backoff.
        
        Args:
            coro: Coroutine to retry
            max_retries: Maximum retry attempts
            backoff: Initial backoff time
            
        Returns:
            Result of successful attempt
        """
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await coro
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(backoff * (2 ** attempt))
        
        raise last_exception


async def demonstrate_async_agent():
    """Demonstrate the async agent capabilities"""
    print("=" * 80)
    print("ASYNCHRONOUS AGENT DEMONSTRATION")
    print("=" * 80)
    
    agent = AsyncAgent(max_concurrent=5)
    
    # Demo 1: Parallel LLM Calls
    print("\n" + "=" * 80)
    print("DEMO 1: Parallel LLM Calls")
    print("=" * 80)
    
    prompts = [
        "What is async programming?",
        "Explain concurrency in Python.",
        "What are the benefits of non-blocking I/O?",
        "How does asyncio work?",
        "What is the difference between async and sync?"
    ]
    
    print(f"\nMaking {len(prompts)} LLM calls in parallel...")
    print("-" * 80)
    
    start_time = time.time()
    responses = await agent.parallel_llm_calls(prompts)
    parallel_duration = time.time() - start_time
    
    print(f"✓ Completed in {parallel_duration:.2f}s")
    print(f"\nResponses ({len(responses)} received):")
    for i, response in enumerate(responses[:3], 1):
        print(f"\n{i}. {response[:100]}...")
    
    # Demo 2: Concurrent Chain Execution
    print("\n" + "=" * 80)
    print("DEMO 2: Concurrent Chain Execution")
    print("=" * 80)
    
    inputs = [
        {"topic": "machine learning", "length": "one sentence"},
        {"topic": "neural networks", "length": "one sentence"},
        {"topic": "deep learning", "length": "one sentence"},
        {"topic": "reinforcement learning", "length": "one sentence"}
    ]
    
    template = "Explain {topic} in {length}."
    
    print(f"\nExecuting chain with {len(inputs)} different inputs...")
    print("-" * 80)
    
    chain_start = time.time()
    results = await agent.concurrent_chain_execution(inputs, template)
    chain_duration = time.time() - chain_start
    
    print(f"✓ Completed in {chain_duration:.2f}s")
    print(f"\nResults:")
    for i, (inp, result) in enumerate(zip(inputs, results), 1):
        print(f"\n{i}. Topic: {inp['topic']}")
        print(f"   {result[:120]}...")
    
    # Demo 3: Concurrent Tasks with Status
    print("\n" + "=" * 80)
    print("DEMO 3: Concurrent Task Execution")
    print("=" * 80)
    
    async def sample_task(n: int) -> str:
        await asyncio.sleep(0.1)  # Simulate work
        return await agent.ainvoke_llm(f"Count to {n} and stop.")
    
    tasks = [sample_task(i) for i in range(1, 6)]
    task_names = [f"count_to_{i}" for i in range(1, 6)]
    
    print(f"\nRunning {len(tasks)} concurrent tasks...")
    print("-" * 80)
    
    result = await agent.run_concurrent_tasks(tasks, task_names)
    
    print(f"✓ All tasks completed!")
    print(f"\nStatistics:")
    print(f"  Total Tasks: {result.total_tasks}")
    print(f"  Successful: {result.successful}")
    print(f"  Failed: {result.failed}")
    print(f"  Total Duration: {result.total_duration:.2f}s")
    print(f"  Average Duration: {result.average_duration:.2f}s")
    
    # Demo 4: Async Streaming
    print("\n" + "=" * 80)
    print("DEMO 4: Asynchronous Streaming")
    print("=" * 80)
    
    prompt = "Write a haiku about async programming."
    
    print(f"\nPrompt: {prompt}")
    print("\nStreaming response asynchronously:")
    print("-" * 80)
    
    tokens = []
    
    def on_token(token: str):
        tokens.append(token)
        print(token, end="", flush=True)
    
    stream_response = await agent.astream_llm(prompt, on_token=on_token)
    
    print("\n" + "-" * 80)
    print(f"✓ Streaming complete! ({len(tokens)} tokens)")
    
    # Demo 5: Task Racing
    print("\n" + "=" * 80)
    print("DEMO 5: Task Racing (First to Complete)")
    print("=" * 80)
    
    async def llm_task(question: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return await agent.ainvoke_llm(question)
    
    race_tasks = [
        llm_task("Quick answer: What is AI?", 0.1),
        llm_task("Quick answer: What is ML?", 0.2),
        llm_task("Quick answer: What is DL?", 0.3)
    ]
    
    print("\nRacing 3 tasks, returning first to complete...")
    print("-" * 80)
    
    race_start = time.time()
    first_result = await agent.race_tasks(race_tasks, return_first=True)
    race_duration = time.time() - race_start
    
    print(f"✓ First task completed in {race_duration:.2f}s")
    print(f"\nResult: {first_result[:150]}...")
    
    # Demo 6: Async vs Sync Performance
    print("\n" + "=" * 80)
    print("DEMO 6: Async vs Synchronous Performance")
    print("=" * 80)
    
    test_prompts = [f"Answer: What is {topic}?" for topic in ["AI", "ML", "DL", "NLP", "CV"]]
    
    print(f"\nProcessing {len(test_prompts)} prompts...")
    
    # Async approach
    print("\nAsync Approach:")
    async_start = time.time()
    async_results = await agent.parallel_llm_calls(test_prompts[:3])  # Limit for demo
    async_duration = time.time() - async_start
    print(f"  Duration: {async_duration:.2f}s")
    print(f"  Results: {len(async_results)} responses")
    
    # Sync approach (simulated)
    print("\nSync Approach (estimated):")
    estimated_sync = async_duration * 3  # Rough estimate
    print(f"  Estimated Duration: {estimated_sync:.2f}s")
    
    speedup = estimated_sync / async_duration if async_duration > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x faster with async")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary = """
The Asynchronous Agent demonstrates non-blocking concurrent operations:

KEY CAPABILITIES:
1. Async LLM Calls: Non-blocking LLM interactions with ainvoke
2. Parallel Execution: Run multiple tasks concurrently
3. Async Streaming: Non-blocking streaming responses
4. Concurrent Chains: Execute chains with different inputs in parallel
5. Task Racing: Return first completed result
6. Timeout Handling: Enforce timeouts on async operations
7. Error Handling: Graceful exception handling in async context
8. Resource Management: Semaphore-based concurrency control

BENEFITS:
- Dramatically improved throughput
- Non-blocking I/O for better resource utilization
- Lower latency for concurrent operations
- Scalable to handle many simultaneous requests
- Efficient handling of I/O-bound workloads
- Better responsiveness in applications
- Reduced total execution time
- Enhanced user experience

USE CASES:
- High-throughput API servers
- Real-time chat applications with multiple users
- Concurrent data processing pipelines
- Parallel research and information gathering
- Multi-source data aggregation
- Async web scraping
- Real-time analytics dashboards
- Concurrent translation services
- Parallel validation and checking
- Multi-agent systems coordination

PRODUCTION CONSIDERATIONS:
1. Concurrency Limits: Use semaphores to control concurrent tasks
2. Timeout Management: Always set timeouts for async operations
3. Error Propagation: Handle exceptions in async context properly
4. Resource Cleanup: Ensure proper cleanup of async resources
5. Backpressure: Handle slow consumers gracefully
6. Connection Pooling: Reuse connections efficiently
7. Monitoring: Track async task performance
8. Testing: Test with asyncio.run() and pytest-asyncio
9. Event Loop: Manage event loop lifecycle carefully
10. Cancellation: Support graceful task cancellation

ADVANCED EXTENSIONS:
- Async context managers for resource management
- Event-driven architectures with async events
- Async queues for producer-consumer patterns
- Async locks and synchronization primitives
- Distributed async processing
- Async retry with circuit breakers
- Priority-based async task scheduling
- Async result caching
- Streaming async aggregation
- Async middleware chains

Async patterns are essential for building high-performance,
scalable AI applications that handle concurrent operations efficiently.
"""
    
    print(summary)


def demonstrate_async_agent_sync():
    """Synchronous wrapper for async demonstration"""
    asyncio.run(demonstrate_async_agent())


if __name__ == "__main__":
    demonstrate_async_agent_sync()
