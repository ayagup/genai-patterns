"""
Pattern 142: Speculative Execution

Description:
    The Speculative Execution pattern optimizes performance by predicting and executing
    likely operations before they're actually needed, enabling parallel processing and
    reduced latency. The agent anticipates probable next steps and begins computation
    proactively, discarding results if predictions prove incorrect. This pattern is
    particularly valuable for operations with high latency or when multiple potential
    paths exist.

    Speculative execution leverages parallelism to compute multiple branches
    simultaneously, betting that at least one will be correct. When predictions succeed,
    the agent gains significant performance improvements by having results ready
    immediately. When wrong, the cost is the wasted computation, which must be balanced
    against the benefits.

    This implementation provides speculative execution capabilities including predictive
    computation, parallel branch execution, result management, confidence-based
    speculation, and waste tracking. It supports both simple lookahead and complex
    multi-path speculation with cancellation mechanisms.

Components:
    - Predictive Computation: Anticipate likely operations
    - Parallel Execution: Run multiple possibilities simultaneously
    - Result Management: Handle multiple speculative results
    - Confidence-Based: Speculate based on prediction confidence
    - Waste Tracking: Monitor efficiency of speculation
    - Cancellation: Stop incorrect speculations early

Use Cases:
    - High-latency operations (API calls, database queries)
    - Branching workflows with predictable paths
    - User interaction prediction
    - Data prefetching
    - Multi-modal processing
    - Recommendation systems
    - Search optimization
    - Decision trees

LangChain Implementation:
    This implementation uses:
    - Parallel chain execution with asyncio
    - Predictive path selection with LLM
    - Speculative result caching
    - Confidence-based speculation triggers
    - Waste tracking and efficiency metrics
    - Result selection and fallback

Benefits:
    - Reduces perceived latency
    - Improves responsiveness
    - Enables parallel processing
    - Optimizes resource utilization
    - Handles uncertainty gracefully
    - Provides fallback options
    - Accelerates common paths

Trade-offs:
    - Wastes resources on incorrect predictions
    - Increases complexity
    - May overload systems
    - Requires careful tuning
    - Can mask performance issues
    - Needs good prediction models
    - Risk of cascade failures

Production Considerations:
    - Monitor speculation accuracy
    - Set resource limits
    - Implement cancellation
    - Track waste metrics
    - Use confidence thresholds
    - Limit speculation depth
    - Handle failures gracefully
    - Provide fallback paths
    - Consider cost implications
    - Test prediction models
    - Monitor system load
    - Balance parallelism
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class SpeculationStatus(Enum):
    """Status of speculative execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    USED = "used"
    WASTED = "wasted"


@dataclass
class SpeculativeTask:
    """Represents a speculative task."""
    task_id: str
    path_id: str
    computation: Callable[[], Any]
    confidence: float
    status: SpeculationStatus = SpeculationStatus.PENDING
    result: Optional[Any] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    
    def execute(self) -> Any:
        """Execute the speculative task."""
        self.status = SpeculationStatus.RUNNING
        self.start_time = datetime.now()
        try:
            self.result = self.computation()
            self.status = SpeculationStatus.COMPLETED
        except Exception as e:
            self.status = SpeculationStatus.FAILED
            self.result = None
        finally:
            self.end_time = datetime.now()
            if self.start_time and self.end_time:
                self.execution_time = (self.end_time - self.start_time).total_seconds()
        return self.result


@dataclass
class SpeculationMetrics:
    """Metrics for speculative execution."""
    total_speculations: int = 0
    successful: int = 0
    wasted: int = 0
    cancelled: int = 0
    failed: int = 0
    total_time_saved: float = 0.0
    total_time_wasted: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate speculation success rate."""
        total = self.successful + self.wasted
        return self.successful / total if total > 0 else 0.0
    
    @property
    def efficiency(self) -> float:
        """Calculate efficiency (time saved vs wasted)."""
        total = self.total_time_saved + self.total_time_wasted
        return self.total_time_saved / total if total > 0 else 0.0


class SpeculativeExecutionAgent:
    """
    Agent that uses speculative execution for performance optimization.
    
    This agent anticipates likely operations and executes them proactively,
    reducing latency when predictions are correct.
    """
    
    def __init__(self, temperature: float = 0.3, max_workers: int = 4):
        """Initialize the speculative execution agent."""
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, SpeculativeTask] = {}
        self.metrics = SpeculationMetrics()
        
        # Speculation thresholds
        self.min_confidence = 0.3  # Minimum confidence to speculate
        self.max_parallel = 3  # Maximum parallel speculations
        
        # Create path prediction chain
        prediction_prompt = ChatPromptTemplate.from_template(
            """You are an execution path predictor. Predict the most likely next steps.

Current Context: {context}
Current State: {state}
Available Paths: {available_paths}

Based on the context and patterns, predict which paths are most likely to be taken.

Respond in this format:
PATH_1: path_name (confidence: 0-1)
PATH_2: path_name (confidence: 0-1)
PATH_3: path_name (confidence: 0-1)
REASONING: explanation"""
        )
        self.prediction_chain = prediction_prompt | self.llm | StrOutputParser()
    
    def predict_paths(
        self,
        context: Dict[str, Any],
        available_paths: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Predict likely execution paths with confidence scores.
        
        Returns:
            List of (path_name, confidence) tuples
        """
        try:
            result = self.prediction_chain.invoke({
                "context": str(context),
                "state": context.get("current_state", "unknown"),
                "available_paths": ", ".join(available_paths)
            })
            
            # Parse predictions
            predictions = []
            for line in result.split('\n'):
                if line.startswith("PATH_"):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        path_part = parts[1].strip()
                        # Extract path name and confidence
                        if "(confidence:" in path_part:
                            path_name = path_part.split("(confidence:")[0].strip()
                            confidence_str = path_part.split("(confidence:")[1].split(")")[0].strip()
                            try:
                                confidence = float(confidence_str)
                                predictions.append((path_name, confidence))
                            except ValueError:
                                pass
            
            return predictions[:self.max_parallel]
            
        except Exception as e:
            print(f"Error predicting paths: {e}")
            # Default: equal confidence for all paths
            return [(path, 1.0 / len(available_paths)) for path in available_paths[:self.max_parallel]]
    
    def speculate(
        self,
        paths: Dict[str, Callable[[], Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Future]:
        """
        Execute multiple paths speculatively in parallel.
        
        Args:
            paths: Dictionary of path_name -> computation function
            context: Context for path prediction
            
        Returns:
            Dictionary of path_name -> Future
        """
        # Predict likely paths
        predictions = self.predict_paths(context, list(paths.keys()))
        
        # Filter by confidence threshold
        valid_predictions = [
            (path, conf) for path, conf in predictions
            if conf >= self.min_confidence and path in paths
        ]
        
        # Submit speculative tasks
        futures = {}
        for path_name, confidence in valid_predictions:
            task = SpeculativeTask(
                task_id=f"spec_{len(self.active_tasks)}_{time.time()}",
                path_id=path_name,
                computation=paths[path_name],
                confidence=confidence
            )
            self.active_tasks[task.task_id] = task
            futures[path_name] = self.executor.submit(task.execute)
            self.metrics.total_speculations += 1
        
        return futures
    
    def select_result(
        self,
        futures: Dict[str, Future],
        actual_path: str,
        timeout: float = 10.0
    ) -> Any:
        """
        Select the result from the actual path taken.
        
        Args:
            futures: Dictionary of path_name -> Future
            actual_path: The path that was actually taken
            timeout: Maximum time to wait for result
            
        Returns:
            Result from the actual path
        """
        if actual_path not in futures:
            raise ValueError(f"Actual path {actual_path} not in futures")
        
        # Wait for the actual path result
        actual_future = futures[actual_path]
        try:
            result = actual_future.result(timeout=timeout)
            
            # Update metrics for actual path
            actual_task = self._find_task_by_path(actual_path)
            if actual_task:
                actual_task.status = SpeculationStatus.USED
                self.metrics.successful += 1
                if actual_task.execution_time:
                    # Time saved is the time it would have taken if we waited
                    self.metrics.total_time_saved += actual_task.execution_time
            
            # Cancel and mark other paths as wasted
            for path_name, future in futures.items():
                if path_name != actual_path:
                    future.cancel()
                    task = self._find_task_by_path(path_name)
                    if task:
                        if task.status == SpeculationStatus.COMPLETED:
                            task.status = SpeculationStatus.WASTED
                            self.metrics.wasted += 1
                            if task.execution_time:
                                self.metrics.total_time_wasted += task.execution_time
                        else:
                            task.status = SpeculationStatus.CANCELLED
                            self.metrics.cancelled += 1
            
            return result
            
        except Exception as e:
            self.metrics.failed += 1
            raise e
    
    def _find_task_by_path(self, path_name: str) -> Optional[SpeculativeTask]:
        """Find task by path name."""
        for task in self.active_tasks.values():
            if task.path_id == path_name:
                return task
        return None
    
    def speculative_search(
        self,
        query: str,
        search_strategies: Dict[str, Callable[[str], Any]]
    ) -> Any:
        """
        Speculatively execute multiple search strategies.
        
        Args:
            query: Search query
            search_strategies: Dictionary of strategy_name -> search function
            
        Returns:
            Best search result
        """
        context = {"query": query, "query_type": "search"}
        
        # Create speculation tasks
        tasks = {
            name: lambda name=name: strategy(query)
            for name, strategy in search_strategies.items()
        }
        
        # Execute speculatively
        futures = self.speculate(tasks, context)
        
        # For search, we take the first one that completes
        completed_results = {}
        for name, future in futures.items():
            try:
                result = future.result(timeout=5.0)
                completed_results[name] = result
            except Exception:
                pass
        
        # Return first completed result
        if completed_results:
            first_strategy = list(completed_results.keys())[0]
            return self.select_result(futures, first_strategy, timeout=0.1)
        
        return None
    
    def prefetch_data(
        self,
        likely_requests: List[str],
        fetch_function: Callable[[str], Any],
        confidence_scores: Optional[List[float]] = None
    ) -> Dict[str, Future]:
        """
        Prefetch data for likely requests.
        
        Args:
            likely_requests: List of likely data requests
            fetch_function: Function to fetch data
            confidence_scores: Optional confidence for each request
            
        Returns:
            Dictionary of request -> Future
        """
        if confidence_scores is None:
            confidence_scores = [1.0 / len(likely_requests)] * len(likely_requests)
        
        futures = {}
        for request, confidence in zip(likely_requests, confidence_scores):
            if confidence >= self.min_confidence:
                task = SpeculativeTask(
                    task_id=f"prefetch_{len(self.active_tasks)}_{time.time()}",
                    path_id=request,
                    computation=lambda req=request: fetch_function(req),
                    confidence=confidence
                )
                self.active_tasks[task.task_id] = task
                futures[request] = self.executor.submit(task.execute)
                self.metrics.total_speculations += 1
        
        return futures
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get speculation metrics."""
        return {
            "total_speculations": self.metrics.total_speculations,
            "successful": self.metrics.successful,
            "wasted": self.metrics.wasted,
            "cancelled": self.metrics.cancelled,
            "failed": self.metrics.failed,
            "success_rate": f"{self.metrics.success_rate:.2%}",
            "efficiency": f"{self.metrics.efficiency:.2%}",
            "time_saved": f"{self.metrics.total_time_saved:.2f}s",
            "time_wasted": f"{self.metrics.total_time_wasted:.2f}s",
            "net_benefit": f"{self.metrics.total_time_saved - self.metrics.total_time_wasted:.2f}s"
        }
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


def demonstrate_speculative_execution():
    """Demonstrate the speculative execution pattern."""
    print("=" * 80)
    print("Speculative Execution Pattern Demonstration")
    print("=" * 80)
    
    agent = SpeculativeExecutionAgent(max_workers=4)
    
    try:
        # Demonstration 1: Basic Speculative Execution
        print("\n" + "=" * 80)
        print("Demonstration 1: Basic Speculative Execution")
        print("=" * 80)
        
        def path_a():
            time.sleep(0.5)
            return "Result from Path A"
        
        def path_b():
            time.sleep(0.5)
            return "Result from Path B"
        
        def path_c():
            time.sleep(0.5)
            return "Result from Path C"
        
        paths = {
            "path_a": path_a,
            "path_b": path_b,
            "path_c": path_c
        }
        
        context = {"current_state": "initial", "user_preference": "fast"}
        
        print("\nStarting speculative execution of 3 paths...")
        start = time.time()
        futures = agent.speculate(paths, context)
        print(f"Speculated paths: {list(futures.keys())}")
        
        # Simulate that path_b was the actual path taken
        actual_path = "path_b"
        print(f"\nActual path taken: {actual_path}")
        
        result = agent.select_result(futures, actual_path)
        elapsed = time.time() - start
        
        print(f"Result: {result}")
        print(f"Total time: {elapsed:.2f}s (with speculation)")
        print("Without speculation, would have taken 0.5s after decision")
        
        # Demonstration 2: Confidence-Based Speculation
        print("\n" + "=" * 80)
        print("Demonstration 2: Confidence-Based Speculation")
        print("=" * 80)
        
        def operation_high_conf():
            time.sleep(0.3)
            return "High confidence result"
        
        def operation_low_conf():
            time.sleep(0.3)
            return "Low confidence result"
        
        # Manually set predictions with different confidences
        print("\nPredictions:")
        print("  Path 1 (high confidence): 0.8")
        print("  Path 2 (low confidence): 0.2")
        
        paths = {
            "high_conf": operation_high_conf,
            "low_conf": operation_low_conf
        }
        
        # Only high confidence should be speculated
        agent.min_confidence = 0.5
        futures = agent.speculate(paths, {"confidence_test": True})
        
        print(f"\nSpeculated paths (min confidence 0.5): {list(futures.keys())}")
        
        # Demonstration 3: Prefetching
        print("\n" + "=" * 80)
        print("Demonstration 3: Data Prefetching")
        print("=" * 80)
        
        def fetch_data(item_id: str) -> str:
            """Simulate data fetching."""
            time.sleep(0.2)
            return f"Data for {item_id}"
        
        likely_requests = ["item_1", "item_2", "item_3"]
        confidence_scores = [0.7, 0.5, 0.3]
        
        print(f"Likely requests: {likely_requests}")
        print(f"Confidence scores: {confidence_scores}")
        print(f"Prefetching data speculatively...")
        
        prefetch_futures = agent.prefetch_data(
            likely_requests,
            fetch_data,
            confidence_scores
        )
        
        print(f"Prefetched: {list(prefetch_futures.keys())}")
        
        # User actually requests item_2
        actual_request = "item_2"
        print(f"\nUser requests: {actual_request}")
        
        if actual_request in prefetch_futures:
            result = agent.select_result(prefetch_futures, actual_request, timeout=1.0)
            print(f"Result (from prefetch): {result}")
        
        # Demonstration 4: Multi-Path Processing
        print("\n" + "=" * 80)
        print("Demonstration 4: Multi-Path Processing")
        print("=" * 80)
        
        def process_with_method_a(data: str) -> str:
            time.sleep(0.4)
            return f"Processed {data} with method A"
        
        def process_with_method_b(data: str) -> str:
            time.sleep(0.4)
            return f"Processed {data} with method B"
        
        def process_with_method_c(data: str) -> str:
            time.sleep(0.4)
            return f"Processed {data} with method C"
        
        data = "sample_data"
        methods = {
            "method_a": lambda: process_with_method_a(data),
            "method_b": lambda: process_with_method_b(data),
            "method_c": lambda: process_with_method_c(data)
        }
        
        context = {"data_type": "text", "processing_mode": "fast"}
        
        print(f"Processing {data} with multiple methods speculatively...")
        futures = agent.speculate(methods, context)
        print(f"Speculated methods: {list(futures.keys())}")
        
        # Decision: use method_a
        chosen_method = list(futures.keys())[0] if futures else "method_a"
        print(f"\nChosen method: {chosen_method}")
        
        result = agent.select_result(futures, chosen_method)
        print(f"Result: {result}")
        
        # Demonstration 5: Early Cancellation
        print("\n" + "=" * 80)
        print("Demonstration 5: Early Cancellation")
        print("=" * 80)
        
        def slow_operation_1():
            time.sleep(2.0)
            return "Slow result 1"
        
        def slow_operation_2():
            time.sleep(2.0)
            return "Slow result 2"
        
        def fast_operation():
            time.sleep(0.1)
            return "Fast result"
        
        paths = {
            "slow_1": slow_operation_1,
            "slow_2": slow_operation_2,
            "fast": fast_operation
        }
        
        print("Starting speculation with slow and fast paths...")
        futures = agent.speculate(paths, {"cancel_test": True})
        
        # Quickly decide on fast path
        time.sleep(0.2)
        actual_path = "fast"
        print(f"\nQuick decision: use {actual_path}")
        print("Cancelling slow speculations...")
        
        result = agent.select_result(futures, actual_path, timeout=0.5)
        print(f"Result: {result}")
        print("Slow operations were cancelled early")
        
        # Demonstration 6: Speculation Metrics
        print("\n" + "=" * 80)
        print("Demonstration 6: Speculation Metrics")
        print("=" * 80)
        
        metrics = agent.get_metrics()
        print("\nSpeculation Performance:")
        print(f"  Total Speculations: {metrics['total_speculations']}")
        print(f"  Successful: {metrics['successful']}")
        print(f"  Wasted: {metrics['wasted']}")
        print(f"  Cancelled: {metrics['cancelled']}")
        print(f"  Failed: {metrics['failed']}")
        print(f"  Success Rate: {metrics['success_rate']}")
        print(f"  Efficiency: {metrics['efficiency']}")
        print(f"  Time Saved: {metrics['time_saved']}")
        print(f"  Time Wasted: {metrics['time_wasted']}")
        print(f"  Net Benefit: {metrics['net_benefit']}")
        
        # Summary
        print("\n" + "=" * 80)
        print("Summary: Speculative Execution Pattern")
        print("=" * 80)
        print("""
The Speculative Execution pattern optimizes performance through prediction:

Key Features Demonstrated:
1. Basic Speculation - Execute multiple paths in parallel
2. Confidence-Based - Only speculate on likely paths
3. Data Prefetching - Preload likely-needed data
4. Multi-Path Processing - Try multiple approaches simultaneously
5. Early Cancellation - Stop unused speculations quickly
6. Metrics Tracking - Monitor speculation effectiveness

Benefits:
- Reduces perceived latency
- Improves responsiveness
- Enables parallel processing
- Optimizes resource utilization
- Provides fallback options
- Accelerates common paths

Best Practices:
- Monitor speculation accuracy
- Set resource limits
- Implement cancellation
- Track waste metrics
- Use confidence thresholds
- Limit speculation depth
- Handle failures gracefully
- Provide fallback paths
- Consider cost implications
- Test prediction models
- Monitor system load
- Balance parallelism

Common Use Cases:
- High-latency operations
- Branching workflows
- User interaction prediction
- Data prefetching
- Multi-modal processing
- Recommendation systems
- Search optimization
- Decision trees

Performance Considerations:
- Speculation wastes resources on wrong predictions
- Best for high-latency operations
- Requires good prediction models
- Monitor success rate and efficiency
- Balance speculation vs. waste
- Consider system capacity

This pattern is essential for responsive AI agents that need to minimize
latency by anticipating likely operations.
""")
    
    finally:
        # Cleanup
        agent.shutdown()


if __name__ == "__main__":
    demonstrate_speculative_execution()
