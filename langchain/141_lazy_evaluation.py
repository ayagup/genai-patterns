"""
Pattern 141: Lazy Evaluation

Description:
    The Lazy Evaluation pattern defers computation until results are actually needed,
    optimizing resource usage by avoiding unnecessary work. Instead of eagerly computing
    all results upfront, lazy evaluation creates computation plans that execute only
    when values are requested. This pattern is particularly valuable for expensive
    operations, large datasets, and scenarios where not all computed values may be used.

    Lazy evaluation improves performance by reducing wasted computation, lowering
    memory usage, enabling infinite sequences, and allowing for optimization opportunities
    through computation graph analysis. It's especially useful in AI agents that might
    explore multiple paths but only need results from successful branches, or when
    processing large datasets where early termination is possible.

    This implementation provides lazy evaluation capabilities for LangChain agents,
    including deferred chain execution, lazy data loading, conditional computation,
    result streaming, and computation graph optimization. It supports both simple
    delayed execution and complex multi-stage lazy pipelines.

Components:
    - Deferred Execution: Delay computation until needed
    - Lazy Data Loading: Load data on-demand
    - Conditional Computation: Skip unnecessary branches
    - Result Streaming: Process results incrementally
    - Graph Optimization: Optimize computation plans
    - Caching Integration: Cache lazy results

Use Cases:
    - Large dataset processing
    - Multi-path exploration
    - Conditional workflows
    - Resource-constrained environments
    - Expensive API calls
    - Database query optimization
    - Streaming data processing
    - Early termination scenarios

LangChain Implementation:
    This implementation uses:
    - Python generators for lazy iteration
    - Deferred chain execution with LCEL
    - Lazy data loading with document loaders
    - Conditional branching in chains
    - Streaming for incremental results
    - Custom lazy wrappers for expensive operations

Benefits:
    - Reduces unnecessary computation
    - Lowers memory usage
    - Enables early termination
    - Improves responsiveness
    - Supports infinite sequences
    - Allows optimization opportunities
    - Reduces API costs

Trade-offs:
    - Can make debugging harder
    - May cause unexpected delays
    - Complexity in error handling
    - Potential memory leaks if not careful
    - Harder to reason about execution flow
    - May hide performance issues

Production Considerations:
    - Profile to identify expensive operations
    - Use for truly expensive computations
    - Combine with caching
    - Handle errors in lazy contexts
    - Set timeouts for lazy operations
    - Monitor actual execution patterns
    - Document lazy behavior clearly
    - Test edge cases thoroughly
    - Consider memory management
    - Balance with eager evaluation
    - Use generators appropriately
    - Profile actual vs potential savings
"""

import os
import time
from typing import List, Dict, Any, Optional, Iterator, Callable, Generator
from datetime import datetime
from dataclasses import dataclass
from functools import wraps
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass
class LazyResult:
    """Wrapper for lazy computation result."""
    computation: Callable[[], Any]
    _cached_value: Optional[Any] = None
    _computed: bool = False
    _computation_time: Optional[float] = None
    
    def compute(self) -> Any:
        """Execute the lazy computation."""
        if not self._computed:
            start = time.time()
            self._cached_value = self.computation()
            self._computation_time = time.time() - start
            self._computed = True
        return self._cached_value
    
    @property
    def value(self) -> Any:
        """Get the computed value."""
        return self.compute()
    
    @property
    def is_computed(self) -> bool:
        """Check if value has been computed."""
        return self._computed
    
    @property
    def computation_time(self) -> Optional[float]:
        """Get computation time in seconds."""
        return self._computation_time


def lazy(func: Callable) -> Callable:
    """Decorator to make a function lazy."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        def computation():
            return func(*args, **kwargs)
        return LazyResult(computation=computation)
    return wrapper


class LazyChain:
    """Lazy evaluation wrapper for LangChain chains."""
    
    def __init__(self, chain):
        """Initialize lazy chain."""
        self.chain = chain
        self._result = None
        self._computed = False
    
    def compute(self, input_data: Dict[str, Any]) -> Any:
        """Execute the chain."""
        if not self._computed:
            self._result = self.chain.invoke(input_data)
            self._computed = True
        return self._result
    
    def stream(self, input_data: Dict[str, Any]) -> Iterator[Any]:
        """Stream results lazily."""
        for chunk in self.chain.stream(input_data):
            yield chunk


class LazyDataLoader:
    """Lazy data loader that loads data on demand."""
    
    def __init__(self, data_source: str):
        """Initialize lazy data loader."""
        self.data_source = data_source
        self._data = None
        self._loaded = False
        self._load_time = None
    
    def load(self) -> List[Any]:
        """Load data (simulated)."""
        if not self._loaded:
            start = time.time()
            # Simulate expensive data loading
            time.sleep(0.1)
            self._data = [f"Item {i} from {self.data_source}" for i in range(100)]
            self._load_time = time.time() - start
            self._loaded = True
        return self._data
    
    def iter_lazy(self, chunk_size: int = 10) -> Generator[List[Any], None, None]:
        """Iterate over data in chunks."""
        data = self.load()
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
    
    @property
    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return self._loaded


class LazyEvaluationAgent:
    """
    Agent that uses lazy evaluation for efficient processing.
    
    This agent defers expensive computations until results are needed,
    optimizing resource usage and enabling early termination.
    """
    
    def __init__(self, temperature: float = 0.3):
        """Initialize the lazy evaluation agent."""
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.computation_stats = {
            "deferred": 0,
            "executed": 0,
            "saved": 0
        }
        
        # Create lazy chains
        self.analysis_chain = LazyChain(
            ChatPromptTemplate.from_template(
                "Analyze this text: {text}\n\nProvide a detailed analysis."
            ) | self.llm | StrOutputParser()
        )
    
    @lazy
    def expensive_computation(self, value: int) -> int:
        """Simulate expensive computation."""
        time.sleep(0.5)  # Simulate work
        return value ** 2
    
    def lazy_filter(
        self,
        items: List[Any],
        condition: Callable[[Any], bool]
    ) -> Generator[Any, None, None]:
        """Lazy filter that yields items matching condition."""
        for item in items:
            if condition(item):
                yield item
    
    def lazy_map(
        self,
        items: List[Any],
        transform: Callable[[Any], Any]
    ) -> Generator[Any, None, None]:
        """Lazy map that transforms items on demand."""
        for item in items:
            yield transform(item)
    
    def conditional_computation(
        self,
        condition: bool,
        true_computation: Callable[[], Any],
        false_computation: Callable[[], Any]
    ) -> LazyResult:
        """Conditionally defer computation."""
        if condition:
            return LazyResult(computation=true_computation)
        else:
            return LazyResult(computation=false_computation)
    
    def lazy_pipeline(
        self,
        data: List[Any],
        operations: List[Callable[[Any], Any]],
        early_exit: Optional[Callable[[Any], bool]] = None
    ) -> Generator[Any, None, None]:
        """
        Execute a lazy pipeline with early exit capability.
        
        Args:
            data: Input data
            operations: List of operations to apply
            early_exit: Optional function to check if we should stop early
        """
        for item in data:
            result = item
            
            # Apply operations lazily
            for op in operations:
                result = op(result)
                
                # Check early exit condition
                if early_exit and early_exit(result):
                    yield result
                    return  # Early termination
            
            yield result
    
    def parallel_lazy_evaluation(
        self,
        tasks: List[Callable[[], Any]],
        max_results: Optional[int] = None
    ) -> Generator[Any, None, None]:
        """
        Evaluate multiple lazy tasks, stopping when enough results obtained.
        
        Args:
            tasks: List of lazy computations
            max_results: Maximum number of results to compute
        """
        count = 0
        for task in tasks:
            if max_results and count >= max_results:
                break
            
            result = LazyResult(computation=task)
            yield result
            count += 1
    
    def lazy_llm_batch(
        self,
        prompts: List[str],
        process_until: Optional[Callable[[str], bool]] = None
    ) -> Generator[str, None, None]:
        """
        Process prompts lazily, with optional early termination.
        
        Args:
            prompts: List of prompts to process
            process_until: Optional function to check if we should stop
        """
        for prompt in prompts:
            result = self.llm.invoke(prompt).content
            yield result
            
            if process_until and process_until(result):
                break  # Early termination
    
    def get_stats(self) -> Dict[str, Any]:
        """Get computation statistics."""
        total = self.computation_stats["executed"] + self.computation_stats["saved"]
        return {
            "deferred": self.computation_stats["deferred"],
            "executed": self.computation_stats["executed"],
            "saved": self.computation_stats["saved"],
            "efficiency": (
                self.computation_stats["saved"] / total * 100 
                if total > 0 else 0
            )
        }


def demonstrate_lazy_evaluation():
    """Demonstrate the lazy evaluation pattern."""
    print("=" * 80)
    print("Lazy Evaluation Pattern Demonstration")
    print("=" * 80)
    
    agent = LazyEvaluationAgent()
    
    # Demonstration 1: Basic Lazy Computation
    print("\n" + "=" * 80)
    print("Demonstration 1: Basic Lazy Computation")
    print("=" * 80)
    
    print("\nCreating lazy computation...")
    lazy_result = agent.expensive_computation(10)
    print(f"Computation created (not executed yet)")
    print(f"Is computed: {lazy_result.is_computed}")
    
    print("\nAccessing value (triggers computation)...")
    start = time.time()
    value = lazy_result.value
    elapsed = time.time() - start
    print(f"Result: {value}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Is computed: {lazy_result.is_computed}")
    
    print("\nAccessing value again (uses cached)...")
    start = time.time()
    value2 = lazy_result.value
    elapsed = time.time() - start
    print(f"Result: {value2}")
    print(f"Time: {elapsed:.3f}s (cached)")
    
    # Demonstration 2: Lazy Filter
    print("\n" + "=" * 80)
    print("Demonstration 2: Lazy Filter (Early Termination)")
    print("=" * 80)
    
    numbers = list(range(1, 101))
    print(f"Processing {len(numbers)} numbers...")
    
    # Eager evaluation
    start = time.time()
    eager_result = [x for x in numbers if x % 7 == 0][:5]  # Get first 5
    eager_time = time.time() - start
    print(f"\nEager evaluation: {eager_result}")
    print(f"Time: {eager_time:.4f}s (processed all {len(numbers)} items)")
    
    # Lazy evaluation with early termination
    start = time.time()
    lazy_filtered = agent.lazy_filter(numbers, lambda x: x % 7 == 0)
    lazy_result = []
    for item in lazy_filtered:
        lazy_result.append(item)
        if len(lazy_result) >= 5:
            break  # Early termination
    lazy_time = time.time() - start
    print(f"\nLazy evaluation: {lazy_result}")
    print(f"Time: {lazy_time:.4f}s (stopped after finding 5)")
    
    # Demonstration 3: Lazy Data Loading
    print("\n" + "=" * 80)
    print("Demonstration 3: Lazy Data Loading")
    print("=" * 80)
    
    loader = LazyDataLoader("database")
    print(f"Data loader created")
    print(f"Is loaded: {loader.is_loaded}")
    
    print("\nProcessing first chunk only...")
    for chunk in loader.iter_lazy(chunk_size=10):
        print(f"Processing chunk of {len(chunk)} items")
        # Only process first chunk
        break
    
    print(f"Is loaded: {loader.is_loaded} (only loaded what was needed)")
    
    # Demonstration 4: Conditional Lazy Computation
    print("\n" + "=" * 80)
    print("Demonstration 4: Conditional Lazy Computation")
    print("=" * 80)
    
    def expensive_path_a():
        time.sleep(0.3)
        return "Result from expensive path A"
    
    def expensive_path_b():
        time.sleep(0.3)
        return "Result from expensive path B"
    
    condition = True
    print(f"Condition: {condition}")
    print("Creating conditional lazy computation...")
    
    result = agent.conditional_computation(
        condition,
        expensive_path_a,
        expensive_path_b
    )
    
    print(f"Computation deferred, is computed: {result.is_computed}")
    print("\nNow computing...")
    value = result.value
    print(f"Result: {value}")
    print(f"Computation time: {result.computation_time:.3f}s")
    
    # Demonstration 5: Lazy Pipeline with Early Exit
    print("\n" + "=" * 80)
    print("Demonstration 5: Lazy Pipeline with Early Exit")
    print("=" * 80)
    
    data = list(range(1, 21))
    
    operations = [
        lambda x: x * 2,
        lambda x: x + 10,
        lambda x: x ** 2
    ]
    
    early_exit = lambda x: x > 1000
    
    print(f"Processing {len(data)} items through pipeline...")
    print("Early exit when result > 1000")
    
    results = list(agent.lazy_pipeline(data, operations, early_exit))
    print(f"\nProcessed {len(results)} items before early exit")
    print(f"Results: {results}")
    
    # Demonstration 6: Lazy Map Transformation
    print("\n" + "=" * 80)
    print("Demonstration 6: Lazy Map (Process Only What's Needed)")
    print("=" * 80)
    
    items = ["apple", "banana", "cherry", "date", "elderberry"]
    
    def expensive_transform(item: str) -> str:
        """Simulate expensive transformation."""
        time.sleep(0.1)
        return item.upper()
    
    print(f"Items to transform: {items}")
    print("Transforming lazily, taking only first 3...")
    
    start = time.time()
    transformed = agent.lazy_map(items, expensive_transform)
    result = []
    for item in transformed:
        result.append(item)
        if len(result) >= 3:
            break
    elapsed = time.time() - start
    
    print(f"Transformed: {result}")
    print(f"Time: {elapsed:.2f}s (only transformed 3 items, not all {len(items)})")
    
    # Demonstration 7: Parallel Lazy Tasks with Limit
    print("\n" + "=" * 80)
    print("Demonstration 7: Parallel Lazy Tasks (Limited Execution)")
    print("=" * 80)
    
    tasks = [
        lambda: f"Task 1 result",
        lambda: f"Task 2 result",
        lambda: f"Task 3 result",
        lambda: f"Task 4 result",
        lambda: f"Task 5 result",
    ]
    
    print(f"Created {len(tasks)} tasks")
    print("Executing only first 2 tasks...")
    
    lazy_results = agent.parallel_lazy_evaluation(tasks, max_results=2)
    executed = []
    for lazy_result in lazy_results:
        executed.append(lazy_result.value)
    
    print(f"Executed: {len(executed)} tasks")
    print(f"Results: {executed}")
    print(f"Saved: {len(tasks) - len(executed)} task executions")
    
    # Demonstration 8: Lazy LLM Batch Processing
    print("\n" + "=" * 80)
    print("Demonstration 8: Lazy LLM Batch (Early Termination)")
    print("=" * 80)
    
    prompts = [
        "Count to 3",
        "Count to 5",
        "Count to 10",
        "Count to 20"
    ]
    
    print(f"Created {len(prompts)} prompts")
    print("Processing until we get a result with '10' in it...")
    
    processed = 0
    found_result = None
    
    for result in agent.lazy_llm_batch(
        prompts,
        process_until=lambda r: "10" in r
    ):
        processed += 1
        if "10" in result:
            found_result = result
            break
    
    print(f"\nProcessed: {processed}/{len(prompts)} prompts")
    print(f"Found result: {found_result[:50]}...")
    print(f"Saved: {len(prompts) - processed} LLM calls")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary: Lazy Evaluation Pattern")
    print("=" * 80)
    print("""
The Lazy Evaluation pattern optimizes resource usage by deferring computation:

Key Features Demonstrated:
1. Basic Lazy Computation - Defer expensive operations until needed
2. Lazy Filter - Filter with early termination
3. Lazy Data Loading - Load data on-demand
4. Conditional Computation - Only compute chosen branch
5. Lazy Pipeline - Process with early exit
6. Lazy Map - Transform only what's needed
7. Limited Execution - Execute only required tasks
8. LLM Batch Processing - Stop when condition met

Benefits:
- Reduces unnecessary computation
- Lowers memory usage
- Enables early termination
- Improves responsiveness
- Reduces API costs
- Supports infinite sequences
- Allows optimization

Best Practices:
- Use for expensive operations
- Combine with caching
- Handle errors properly
- Set timeouts
- Monitor execution patterns
- Document lazy behavior
- Test edge cases
- Balance with eager evaluation
- Use generators appropriately
- Profile actual savings

Common Use Cases:
- Large dataset processing
- Multi-path exploration
- Conditional workflows
- Resource-constrained environments
- Expensive API calls
- Database queries
- Streaming data
- Early termination scenarios

This pattern is essential for efficient AI agents that need to optimize
resource usage and avoid unnecessary computation.
""")


if __name__ == "__main__":
    demonstrate_lazy_evaluation()
