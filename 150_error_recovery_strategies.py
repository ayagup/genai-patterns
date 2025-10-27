"""
Pattern 150: Error Recovery Strategies Agent

This pattern implements comprehensive error recovery with multiple strategies,
fallback chains, and automatic recovery orchestration. Provides intelligent
recovery selection and execution for resilient systems.

Category: Error Handling & Recovery
Use Cases:
- Resilient service architectures
- Automatic error recovery
- Multi-level fallback systems
- Production error handling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, TypeVar
from enum import Enum
from datetime import datetime
import random
import time

T = TypeVar('T')


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY = "retry"                   # Retry the operation
    FALLBACK = "fallback"             # Use fallback value/method
    CACHE = "cache"                   # Use cached value
    DEFAULT = "default"               # Use default value
    ALTERNATIVE_SERVICE = "alternative_service"  # Use backup service
    DEGRADED_MODE = "degraded_mode"   # Operate with reduced functionality
    COMPENSATE = "compensate"         # Execute compensating action
    ESCALATE = "escalate"             # Escalate to human
    SKIP = "skip"                     # Skip and continue
    ABORT = "abort"                   # Abort operation


class RecoveryPriority(Enum):
    """Priority of recovery strategies"""
    HIGH = "high"       # Try first
    MEDIUM = "medium"   # Try if high priority fails
    LOW = "low"         # Last resort
    FALLBACK = "fallback"  # Use only as fallback


class RecoveryOutcome(Enum):
    """Outcome of recovery attempt"""
    SUCCESS = "success"           # Recovery succeeded
    FAILED = "failed"             # Recovery failed
    PARTIAL = "partial"           # Partial recovery
    SKIPPED = "skipped"           # Strategy skipped
    NOT_APPLICABLE = "not_applicable"  # Strategy not applicable


@dataclass
class RecoveryContext:
    """Context information for recovery"""
    error_type: str
    error_message: str
    operation_name: str
    attempt_count: int = 0
    max_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RecoveryResult:
    """Result of a recovery attempt"""
    strategy: RecoveryStrategy
    outcome: RecoveryOutcome
    recovered_value: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    attempts_used: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryChain:
    """Chain of recovery strategies to try in sequence"""
    name: str
    strategies: List[RecoveryStrategy]
    priorities: Dict[RecoveryStrategy, RecoveryPriority] = field(default_factory=dict)
    max_attempts_per_strategy: int = 3
    stop_on_first_success: bool = True


class RetryRecoverer:
    """Implements retry recovery strategy"""
    
    def __init__(self, max_retries: int = 3, backoff_ms: int = 100):
        self.max_retries = max_retries
        self.backoff_ms = backoff_ms
    
    def recover(
        self,
        operation: Callable,
        context: RecoveryContext
    ) -> RecoveryResult:
        """Attempt recovery by retrying"""
        start_time = time.time()
        attempts = 0
        
        for attempt in range(self.max_retries):
            attempts += 1
            
            try:
                result = operation()
                execution_time = (time.time() - start_time) * 1000
                
                return RecoveryResult(
                    strategy=RecoveryStrategy.RETRY,
                    outcome=RecoveryOutcome.SUCCESS,
                    recovered_value=result,
                    execution_time_ms=execution_time,
                    attempts_used=attempts,
                    metadata={'final_attempt': attempt + 1}
                )
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_ms * (attempt + 1) / 1000)
                continue
        
        execution_time = (time.time() - start_time) * 1000
        
        return RecoveryResult(
            strategy=RecoveryStrategy.RETRY,
            outcome=RecoveryOutcome.FAILED,
            error="Max retries exceeded",
            execution_time_ms=execution_time,
            attempts_used=attempts
        )


class FallbackRecoverer:
    """Implements fallback recovery strategy"""
    
    def __init__(self, fallback_value: Any = None, fallback_func: Optional[Callable] = None):
        self.fallback_value = fallback_value
        self.fallback_func = fallback_func
    
    def recover(
        self,
        operation: Callable,
        context: RecoveryContext
    ) -> RecoveryResult:
        """Attempt recovery using fallback"""
        start_time = time.time()
        
        try:
            if self.fallback_func:
                result = self.fallback_func()
            else:
                result = self.fallback_value
            
            execution_time = (time.time() - start_time) * 1000
            
            return RecoveryResult(
                strategy=RecoveryStrategy.FALLBACK,
                outcome=RecoveryOutcome.SUCCESS,
                recovered_value=result,
                execution_time_ms=execution_time,
                metadata={'fallback_used': True}
            )
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return RecoveryResult(
                strategy=RecoveryStrategy.FALLBACK,
                outcome=RecoveryOutcome.FAILED,
                error=str(e),
                execution_time_ms=execution_time
            )


class CacheRecoverer:
    """Implements cache-based recovery strategy"""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
    
    def recover(
        self,
        operation: Callable,
        context: RecoveryContext
    ) -> RecoveryResult:
        """Attempt recovery using cached value"""
        start_time = time.time()
        
        cache_key = context.operation_name
        
        if cache_key in self.cache:
            execution_time = (time.time() - start_time) * 1000
            
            return RecoveryResult(
                strategy=RecoveryStrategy.CACHE,
                outcome=RecoveryOutcome.SUCCESS,
                recovered_value=self.cache[cache_key],
                execution_time_ms=execution_time,
                metadata={'cache_hit': True, 'stale': True}
            )
        
        execution_time = (time.time() - start_time) * 1000
        
        return RecoveryResult(
            strategy=RecoveryStrategy.CACHE,
            outcome=RecoveryOutcome.NOT_APPLICABLE,
            error="No cached value available",
            execution_time_ms=execution_time,
            metadata={'cache_hit': False}
        )
    
    def store(self, key: str, value: Any):
        """Store value in cache"""
        self.cache[key] = value


class AlternativeServiceRecoverer:
    """Implements alternative service recovery strategy"""
    
    def __init__(self, alternative_services: List[Callable]):
        self.alternative_services = alternative_services
        self.current_service_index = 0
    
    def recover(
        self,
        operation: Callable,
        context: RecoveryContext
    ) -> RecoveryResult:
        """Attempt recovery using alternative service"""
        start_time = time.time()
        
        for i, service in enumerate(self.alternative_services):
            try:
                result = service()
                execution_time = (time.time() - start_time) * 1000
                
                return RecoveryResult(
                    strategy=RecoveryStrategy.ALTERNATIVE_SERVICE,
                    outcome=RecoveryOutcome.SUCCESS,
                    recovered_value=result,
                    execution_time_ms=execution_time,
                    metadata={'service_index': i, 'service_name': f'alternative_{i+1}'}
                )
            
            except Exception:
                continue
        
        execution_time = (time.time() - start_time) * 1000
        
        return RecoveryResult(
            strategy=RecoveryStrategy.ALTERNATIVE_SERVICE,
            outcome=RecoveryOutcome.FAILED,
            error="All alternative services failed",
            execution_time_ms=execution_time,
            metadata={'services_tried': len(self.alternative_services)}
        )


class DegradedModeRecoverer:
    """Implements degraded mode recovery strategy"""
    
    def __init__(self, degraded_operation: Callable):
        self.degraded_operation = degraded_operation
    
    def recover(
        self,
        operation: Callable,
        context: RecoveryContext
    ) -> RecoveryResult:
        """Attempt recovery with degraded functionality"""
        start_time = time.time()
        
        try:
            result = self.degraded_operation()
            execution_time = (time.time() - start_time) * 1000
            
            return RecoveryResult(
                strategy=RecoveryStrategy.DEGRADED_MODE,
                outcome=RecoveryOutcome.PARTIAL,
                recovered_value=result,
                execution_time_ms=execution_time,
                metadata={'degraded': True, 'full_functionality': False}
            )
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return RecoveryResult(
                strategy=RecoveryStrategy.DEGRADED_MODE,
                outcome=RecoveryOutcome.FAILED,
                error=str(e),
                execution_time_ms=execution_time
            )


class RecoveryOrchestrator:
    """Orchestrates recovery strategies in sequence"""
    
    def __init__(self):
        self.recoverers: Dict[RecoveryStrategy, Any] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        
    def register_recoverer(self, strategy: RecoveryStrategy, recoverer: Any):
        """Register a recovery strategy handler"""
        self.recoverers[strategy] = recoverer
    
    def execute_recovery(
        self,
        operation: Callable,
        context: RecoveryContext,
        chain: RecoveryChain
    ) -> RecoveryResult:
        """Execute recovery chain"""
        results: List[RecoveryResult] = []
        
        # Sort strategies by priority
        sorted_strategies = sorted(
            chain.strategies,
            key=lambda s: self._get_priority_value(chain.priorities.get(s, RecoveryPriority.MEDIUM))
        )
        
        for strategy in sorted_strategies:
            recoverer = self.recoverers.get(strategy)
            
            if not recoverer:
                results.append(RecoveryResult(
                    strategy=strategy,
                    outcome=RecoveryOutcome.NOT_APPLICABLE,
                    error="No recoverer registered for strategy"
                ))
                continue
            
            # Execute recovery
            result = recoverer.recover(operation, context)
            results.append(result)
            
            # Record in history
            self.recovery_history.append({
                'timestamp': datetime.now(),
                'operation': context.operation_name,
                'strategy': strategy.value,
                'outcome': result.outcome.value,
                'execution_time_ms': result.execution_time_ms
            })
            
            # Stop on first success if configured
            if chain.stop_on_first_success and result.outcome == RecoveryOutcome.SUCCESS:
                return result
        
        # No strategy succeeded, return last result
        return results[-1] if results else RecoveryResult(
            strategy=RecoveryStrategy.ABORT,
            outcome=RecoveryOutcome.FAILED,
            error="No recovery strategies available"
        )
    
    def _get_priority_value(self, priority: RecoveryPriority) -> int:
        """Convert priority to numeric value for sorting"""
        priority_map = {
            RecoveryPriority.HIGH: 1,
            RecoveryPriority.MEDIUM: 2,
            RecoveryPriority.LOW: 3,
            RecoveryPriority.FALLBACK: 4
        }
        return priority_map.get(priority, 2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        if not self.recovery_history:
            return {"error": "No recovery history"}
        
        total = len(self.recovery_history)
        
        # By strategy
        by_strategy = {}
        for record in self.recovery_history:
            strategy = record['strategy']
            by_strategy[strategy] = by_strategy.get(strategy, 0) + 1
        
        # By outcome
        by_outcome = {}
        for record in self.recovery_history:
            outcome = record['outcome']
            by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
        
        # Success rate
        successes = by_outcome.get('success', 0)
        success_rate = successes / total if total > 0 else 0.0
        
        # Average execution time
        avg_time = sum(r['execution_time_ms'] for r in self.recovery_history) / total
        
        return {
            'total_recoveries': total,
            'by_strategy': by_strategy,
            'by_outcome': by_outcome,
            'success_rate': f"{success_rate:.1%}",
            'average_execution_time_ms': f"{avg_time:.2f}"
        }


class ErrorRecoveryStrategiesAgent:
    """
    Main agent that manages error recovery with multiple strategies and chains.
    Provides intelligent recovery orchestration for resilient systems.
    """
    
    def __init__(self):
        self.orchestrator = RecoveryOrchestrator()
        self.cache_recoverer = CacheRecoverer()
        self._register_default_recoverers()
        self.recovery_chains: Dict[str, RecoveryChain] = {}
        
    def _register_default_recoverers(self):
        """Register default recovery strategies"""
        self.orchestrator.register_recoverer(
            RecoveryStrategy.RETRY,
            RetryRecoverer(max_retries=3, backoff_ms=100)
        )
        self.orchestrator.register_recoverer(
            RecoveryStrategy.CACHE,
            self.cache_recoverer
        )
        self.orchestrator.register_recoverer(
            RecoveryStrategy.FALLBACK,
            FallbackRecoverer(fallback_value="default_value")
        )
    
    def register_recovery_chain(self, chain: RecoveryChain):
        """Register a recovery chain"""
        self.recovery_chains[chain.name] = chain
    
    def register_recoverer(self, strategy: RecoveryStrategy, recoverer: Any):
        """Register a custom recovery strategy"""
        self.orchestrator.register_recoverer(strategy, recoverer)
    
    def recover(
        self,
        operation: Callable,
        operation_name: str,
        chain_name: str = "default",
        error_type: str = "unknown",
        error_message: str = ""
    ) -> RecoveryResult:
        """Execute recovery for an operation"""
        # Get recovery chain
        chain = self.recovery_chains.get(chain_name)
        
        if not chain:
            # Use default chain
            chain = RecoveryChain(
                name="default",
                strategies=[
                    RecoveryStrategy.RETRY,
                    RecoveryStrategy.CACHE,
                    RecoveryStrategy.FALLBACK
                ],
                priorities={
                    RecoveryStrategy.RETRY: RecoveryPriority.HIGH,
                    RecoveryStrategy.CACHE: RecoveryPriority.MEDIUM,
                    RecoveryStrategy.FALLBACK: RecoveryPriority.LOW
                }
            )
        
        # Create recovery context
        context = RecoveryContext(
            error_type=error_type,
            error_message=error_message,
            operation_name=operation_name
        )
        
        # Execute recovery
        return self.orchestrator.execute_recovery(operation, context, chain)
    
    def cache_result(self, operation_name: str, value: Any):
        """Cache a result for future recovery"""
        self.cache_recoverer.store(operation_name, value)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics"""
        return self.orchestrator.get_statistics()


def demonstrate_error_recovery_strategies():
    """Demonstrate error recovery strategies pattern"""
    print("\n" + "="*60)
    print("ERROR RECOVERY STRATEGIES PATTERN DEMONSTRATION")
    print("="*60)
    
    agent = ErrorRecoveryStrategiesAgent()
    
    # Helper functions
    failure_count = {'count': 0}
    
    def flaky_operation():
        """Operation that fails first few times"""
        failure_count['count'] += 1
        if failure_count['count'] < 3:
            raise Exception("Temporary failure")
        return "Success after retries"
    
    def always_fails():
        """Operation that always fails"""
        raise Exception("Permanent failure")
    
    def alternative_service_1():
        """First alternative service (fails)"""
        raise Exception("Alternative 1 failed")
    
    def alternative_service_2():
        """Second alternative service (succeeds)"""
        return "Result from alternative service 2"
    
    def degraded_service():
        """Degraded but working service"""
        return "Degraded result (limited functionality)"
    
    # Scenario 1: Retry recovery
    print("\n" + "-"*60)
    print("Scenario 1: Retry Recovery Strategy")
    print("-"*60)
    
    failure_count['count'] = 0  # Reset counter
    
    result = agent.recover(
        operation=flaky_operation,
        operation_name="flaky_api_call",
        error_type="TransientError"
    )
    
    print(f"Operation: flaky_api_call")
    print(f"  Strategy: {result.strategy.value}")
    print(f"  Outcome: {result.outcome.value}")
    print(f"  Result: {result.recovered_value}")
    print(f"  Attempts: {result.attempts_used}")
    print(f"  Time: {result.execution_time_ms:.2f}ms")
    
    # Scenario 2: Cache recovery
    print("\n" + "-"*60)
    print("Scenario 2: Cache Recovery Strategy")
    print("-"*60)
    
    # Cache a value first
    agent.cache_result("cached_operation", "Cached data from previous success")
    
    result = agent.recover(
        operation=always_fails,
        operation_name="cached_operation",
        error_type="ServiceUnavailable"
    )
    
    print(f"Operation: cached_operation (failed, using cache)")
    print(f"  Strategy: {result.strategy.value}")
    print(f"  Outcome: {result.outcome.value}")
    print(f"  Result: {result.recovered_value}")
    print(f"  Cache Hit: {result.metadata.get('cache_hit')}")
    print(f"  Stale: {result.metadata.get('stale')}")
    
    # Scenario 3: Alternative service recovery
    print("\n" + "-"*60)
    print("Scenario 3: Alternative Service Recovery")
    print("-"*60)
    
    # Register alternative service recoverer
    alt_recoverer = AlternativeServiceRecoverer([
        alternative_service_1,
        alternative_service_2
    ])
    agent.register_recoverer(RecoveryStrategy.ALTERNATIVE_SERVICE, alt_recoverer)
    
    # Register chain with alternative service
    chain = RecoveryChain(
        name="with_alternatives",
        strategies=[
            RecoveryStrategy.RETRY,
            RecoveryStrategy.ALTERNATIVE_SERVICE,
            RecoveryStrategy.FALLBACK
        ],
        priorities={
            RecoveryStrategy.RETRY: RecoveryPriority.HIGH,
            RecoveryStrategy.ALTERNATIVE_SERVICE: RecoveryPriority.MEDIUM,
            RecoveryStrategy.FALLBACK: RecoveryPriority.LOW
        }
    )
    agent.register_recovery_chain(chain)
    
    result = agent.recover(
        operation=always_fails,
        operation_name="multi_service_call",
        chain_name="with_alternatives",
        error_type="ServiceDown"
    )
    
    print(f"Operation: multi_service_call")
    print(f"  Strategy: {result.strategy.value}")
    print(f"  Outcome: {result.outcome.value}")
    print(f"  Result: {result.recovered_value}")
    print(f"  Service Used: {result.metadata.get('service_name')}")
    
    # Scenario 4: Degraded mode recovery
    print("\n" + "-"*60)
    print("Scenario 4: Degraded Mode Recovery")
    print("-"*60)
    
    # Register degraded mode recoverer
    degraded_recoverer = DegradedModeRecoverer(degraded_service)
    agent.register_recoverer(RecoveryStrategy.DEGRADED_MODE, degraded_recoverer)
    
    # Register chain with degraded mode
    chain = RecoveryChain(
        name="with_degraded",
        strategies=[
            RecoveryStrategy.RETRY,
            RecoveryStrategy.DEGRADED_MODE,
            RecoveryStrategy.FALLBACK
        ],
        priorities={
            RecoveryStrategy.RETRY: RecoveryPriority.HIGH,
            RecoveryStrategy.DEGRADED_MODE: RecoveryPriority.MEDIUM,
            RecoveryStrategy.FALLBACK: RecoveryPriority.LOW
        }
    )
    agent.register_recovery_chain(chain)
    
    result = agent.recover(
        operation=always_fails,
        operation_name="feature_request",
        chain_name="with_degraded",
        error_type="FeatureUnavailable"
    )
    
    print(f"Operation: feature_request")
    print(f"  Strategy: {result.strategy.value}")
    print(f"  Outcome: {result.outcome.value}")
    print(f"  Result: {result.recovered_value}")
    print(f"  Degraded: {result.metadata.get('degraded')}")
    print(f"  Full Functionality: {result.metadata.get('full_functionality')}")
    
    # Scenario 5: Complete recovery chain
    print("\n" + "-"*60)
    print("Scenario 5: Complete Recovery Chain")
    print("-"*60)
    
    chain = RecoveryChain(
        name="complete_chain",
        strategies=[
            RecoveryStrategy.RETRY,
            RecoveryStrategy.CACHE,
            RecoveryStrategy.ALTERNATIVE_SERVICE,
            RecoveryStrategy.DEGRADED_MODE,
            RecoveryStrategy.FALLBACK
        ],
        priorities={
            RecoveryStrategy.RETRY: RecoveryPriority.HIGH,
            RecoveryStrategy.CACHE: RecoveryPriority.HIGH,
            RecoveryStrategy.ALTERNATIVE_SERVICE: RecoveryPriority.MEDIUM,
            RecoveryStrategy.DEGRADED_MODE: RecoveryPriority.MEDIUM,
            RecoveryStrategy.FALLBACK: RecoveryPriority.LOW
        }
    )
    agent.register_recovery_chain(chain)
    
    print("Recovery chain: retry → cache → alternative → degraded → fallback")
    
    result = agent.recover(
        operation=always_fails,
        operation_name="critical_operation",
        chain_name="complete_chain",
        error_type="CompleteFailure"
    )
    
    print(f"\nOperation: critical_operation")
    print(f"  Final Strategy: {result.strategy.value}")
    print(f"  Outcome: {result.outcome.value}")
    print(f"  Result: {result.recovered_value}")
    
    # Scenario 6: Statistics
    print("\n" + "-"*60)
    print("Scenario 6: Recovery Statistics")
    print("-"*60)
    
    stats = agent.get_statistics()
    
    print(f"\nTotal Recoveries: {stats['total_recoveries']}")
    print(f"Success Rate: {stats['success_rate']}")
    print(f"Average Execution Time: {stats['average_execution_time_ms']}")
    
    print("\nBy Strategy:")
    for strategy, count in sorted(stats['by_strategy'].items()):
        print(f"  {strategy}: {count}")
    
    print("\nBy Outcome:")
    for outcome, count in sorted(stats['by_outcome'].items()):
        print(f"  {outcome}: {count}")
    
    # Scenario 7: Multiple operations with different recovery needs
    print("\n" + "-"*60)
    print("Scenario 7: Multiple Operations")
    print("-"*60)
    
    operations = [
        ("user_profile", flaky_operation, "default"),
        ("cached_data", always_fails, "default"),
        ("external_api", always_fails, "with_alternatives"),
        ("feature_toggle", always_fails, "with_degraded"),
    ]
    
    print("Recovering multiple failed operations:\n")
    
    for op_name, op_func, chain_name in operations:
        failure_count['count'] = 0  # Reset for flaky operation
        result = agent.recover(op_func, op_name, chain_name)
        print(f"  {op_name:20} → {result.strategy.value:20} ({result.outcome.value})")
    
    # Final statistics
    print("\n" + "-"*60)
    print("Final Statistics")
    print("-"*60)
    
    stats = agent.get_statistics()
    
    print(f"\nTotal Recovery Attempts: {stats['total_recoveries']}")
    print(f"Overall Success Rate: {stats['success_rate']}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Executed {stats['total_recoveries']} recovery attempts")
    print(f"✓ {stats['success_rate']} success rate across all strategies")
    print(f"✓ Multiple recovery strategies: retry, cache, alternative, degraded, fallback")
    print(f"✓ Prioritized recovery chains with automatic orchestration")
    print(f"✓ Stop-on-first-success optimization")
    print(f"✓ Comprehensive recovery history and statistics")
    print("\n✅ Error Handling & Recovery Category: COMPLETE (5/5 patterns)")
    print("🎯 Milestone: 150/170 patterns = 88.2%")
    print("Error recovery ready for production resilience!")


if __name__ == "__main__":
    demonstrate_error_recovery_strategies()
