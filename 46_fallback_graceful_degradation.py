"""
Fallback/Graceful Degradation Pattern
Alternative strategies when primary approach fails
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
class FallbackLevel(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    EMERGENCY = "emergency"
    HUMAN_ESCALATION = "human_escalation"
@dataclass
class FallbackStrategy:
    """A fallback strategy"""
    level: FallbackLevel
    name: str
    handler: Callable
    timeout_seconds: float = 5.0
    max_retries: int = 1
@dataclass
class ExecutionResult:
    """Result of execution attempt"""
    success: bool
    result: Any
    level_used: FallbackLevel
    error: Optional[str] = None
    latency_ms: float = 0.0
class FallbackAgent:
    """Agent with graceful degradation"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.strategies: List[FallbackStrategy] = []
        self.execution_history: List[ExecutionResult] = []
    def register_strategy(self, strategy: FallbackStrategy):
        """Register a fallback strategy"""
        self.strategies.append(strategy)
        # Sort by level priority
        self.strategies.sort(key=lambda s: list(FallbackLevel).index(s.level))
        print(f"Registered {strategy.level.value} strategy: {strategy.name}")
    def execute_with_fallback(self, task: str, context: Dict[str, Any] = None) -> ExecutionResult:
        """Execute task with fallback strategies"""
        print(f"\n{'='*70}")
        print(f"EXECUTING WITH FALLBACK")
        print(f"{'='*70}")
        print(f"Task: {task}\n")
        context = context or {}
        for strategy in self.strategies:
            print(f"\n--- Trying {strategy.level.value}: {strategy.name} ---")
            for attempt in range(strategy.max_retries):
                if attempt > 0:
                    print(f"  Retry {attempt + 1}/{strategy.max_retries}")
                start_time = time.time()
                try:
                    # Execute strategy with timeout
                    result = self._execute_with_timeout(
                        strategy.handler,
                        task,
                        context,
                        strategy.timeout_seconds
                    )
                    latency_ms = (time.time() - start_time) * 1000
                    print(f"  ✓ Success in {latency_ms:.0f}ms")
                    execution_result = ExecutionResult(
                        success=True,
                        result=result,
                        level_used=strategy.level,
                        latency_ms=latency_ms
                    )
                    self.execution_history.append(execution_result)
                    return execution_result
                except TimeoutError as e:
                    print(f"  ⏱ Timeout after {strategy.timeout_seconds}s")
                    if attempt < strategy.max_retries - 1:
                        continue
                except Exception as e:
                    print(f"  ✗ Failed: {str(e)}")
                    if attempt < strategy.max_retries - 1:
                        continue
            print(f"  ⚠ {strategy.name} exhausted all retries")
        # All strategies failed
        print(f"\n✗ All fallback strategies failed")
        result = ExecutionResult(
            success=False,
            result=None,
            level_used=FallbackLevel.HUMAN_ESCALATION,
            error="All fallback strategies exhausted"
        )
        self.execution_history.append(result)
        return result
    def _execute_with_timeout(self, handler: Callable, task: str, 
                              context: Dict[str, Any], timeout: float) -> Any:
        """Execute handler with timeout"""
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution exceeded {timeout}s")
        # Set timeout alarm (Unix only - for demo)
        # In production, use threading.Timer or asyncio.wait_for
        try:
            # Simple timeout simulation
            start = time.time()
            result = handler(task, context)
            elapsed = time.time() - start
            if elapsed > timeout:
                raise TimeoutError(f"Execution exceeded {timeout}s")
            return result
        except Exception as e:
            raise e
    def get_statistics(self) -> Dict[str, Any]:
        """Get fallback statistics"""
        if not self.execution_history:
            return {"total_executions": 0}
        total = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        level_usage = {}
        for result in self.execution_history:
            level = result.level_used.value
            level_usage[level] = level_usage.get(level, 0) + 1
        avg_latency = sum(r.latency_ms for r in self.execution_history if r.success) / max(successful, 1)
        return {
            'total_executions': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total,
            'level_usage': level_usage,
            'avg_latency_ms': avg_latency
        }
# Example strategy handlers
def primary_strategy(task: str, context: Dict[str, Any]) -> str:
    """Primary high-quality but slow strategy"""
    import random
    time.sleep(0.3)
    # 70% success rate
    if random.random() < 0.7:
        return f"High-quality result for: {task}"
    else:
        raise Exception("Primary strategy failed")
def secondary_strategy(task: str, context: Dict[str, Any]) -> str:
    """Secondary faster but lower quality"""
    import random
    time.sleep(0.1)
    # 80% success rate
    if random.random() < 0.8:
        return f"Good result for: {task}"
    else:
        raise Exception("Secondary strategy failed")
def tertiary_strategy(task: str, context: Dict[str, Any]) -> str:
    """Tertiary very fast, basic quality"""
    time.sleep(0.05)
    # 95% success rate
    if random.random() < 0.95:
        return f"Basic result for: {task}"
    else:
        raise Exception("Tertiary strategy failed")
def emergency_strategy(task: str, context: Dict[str, Any]) -> str:
    """Emergency always-works fallback"""
    return f"Emergency fallback result for: {task}"
# Usage
if __name__ == "__main__":
    print("="*80)
    print("FALLBACK/GRACEFUL DEGRADATION PATTERN DEMONSTRATION")
    print("="*80)
    agent = FallbackAgent("fallback-agent-001")
    # Register strategies in priority order
    agent.register_strategy(FallbackStrategy(
        level=FallbackLevel.PRIMARY,
        name="Advanced AI Model",
        handler=primary_strategy,
        timeout_seconds=1.0,
        max_retries=2
    ))
    agent.register_strategy(FallbackStrategy(
        level=FallbackLevel.SECONDARY,
        name="Standard AI Model",
        handler=secondary_strategy,
        timeout_seconds=0.5,
        max_retries=2
    ))
    agent.register_strategy(FallbackStrategy(
        level=FallbackLevel.TERTIARY,
        name="Fast Basic Model",
        handler=tertiary_strategy,
        timeout_seconds=0.2,
        max_retries=1
    ))
    agent.register_strategy(FallbackStrategy(
        level=FallbackLevel.EMERGENCY,
        name="Template Response",
        handler=emergency_strategy,
        timeout_seconds=0.1,
        max_retries=1
    ))
    # Test with multiple tasks
    tasks = [
        "Analyze customer sentiment",
        "Generate product description",
        "Summarize document",
        "Translate text",
        "Answer question"
    ]
    for task in tasks:
        result = agent.execute_with_fallback(task)
        print(f"\n{'='*60}")
        print(f"RESULT")
        print(f"{'='*60}")
        print(f"Success: {result.success}")
        print(f"Level Used: {result.level_used.value}")
        if result.success:
            print(f"Result: {result.result}")
            print(f"Latency: {result.latency_ms:.0f}ms")
        print()
    # Statistics
    stats = agent.get_statistics()
    print(f"\n{'='*70}")
    print("FALLBACK STATISTICS")
    print(f"{'='*70}")
    print(f"Total Executions: {stats['total_executions']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Average Latency: {stats['avg_latency_ms']:.0f}ms")
    print(f"\nLevel Usage:")
    for level, count in stats['level_usage'].items():
        percentage = (count / stats['total_executions']) * 100
        print(f"  {level}: {count} ({percentage:.1f}%)")
