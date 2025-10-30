"""
Pattern 146: Retry with Backoff Agent

This pattern implements intelligent retry mechanisms with exponential backoff,
jitter, and circuit breaker integration to handle transient failures gracefully.
Prevents cascading failures and system overload during outages.

Category: Error Handling & Recovery
Use Cases:
- API calls with transient failures
- Database connection retries
- Network request handling
- Distributed system communication
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, TypeVar
from enum import Enum
from datetime import datetime, timedelta
import random
import time
import math

T = TypeVar('T')


class BackoffStrategy(Enum):
    """Backoff strategies"""
    CONSTANT = "constant"         # Fixed delay
    LINEAR = "linear"             # Linear increase
    EXPONENTIAL = "exponential"   # Exponential increase
    FIBONACCI = "fibonacci"       # Fibonacci sequence
    POLYNOMIAL = "polynomial"     # Polynomial growth


class JitterType(Enum):
    """Jitter types to prevent thundering herd"""
    NONE = "none"           # No jitter
    FULL = "full"           # Full jitter (0 to delay)
    EQUAL = "equal"         # Equal jitter (delay/2 to delay)
    DECORRELATED = "decorrelated"  # Decorrelated jitter


class RetryDecision(Enum):
    """Retry decision outcomes"""
    RETRY = "retry"         # Retry the operation
    FAIL = "fail"           # Give up
    FALLBACK = "fallback"   # Use fallback
    CIRCUIT_OPEN = "circuit_open"  # Circuit breaker open


class FailureType(Enum):
    """Types of failures"""
    TRANSIENT = "transient"     # Temporary failure
    PERMANENT = "permanent"     # Permanent failure
    TIMEOUT = "timeout"         # Operation timeout
    RATE_LIMIT = "rate_limit"   # Rate limit exceeded
    NETWORK = "network"         # Network error
    RESOURCE = "resource"       # Resource exhaustion


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    base_delay_ms: float = 100.0
    max_delay_ms: float = 30000.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter_type: JitterType = JitterType.EQUAL
    timeout_ms: Optional[float] = None
    retryable_errors: List[FailureType] = field(default_factory=lambda: [
        FailureType.TRANSIENT,
        FailureType.TIMEOUT,
        FailureType.NETWORK
    ])


@dataclass
class RetryAttempt:
    """Record of a retry attempt"""
    attempt_number: int
    timestamp: datetime
    delay_ms: float
    error: Optional[str] = None
    success: bool = False
    duration_ms: float = 0.0


@dataclass
class RetryResult:
    """Result of retry operation"""
    success: bool
    value: Any = None
    error: Optional[str] = None
    attempts: List[RetryAttempt] = field(default_factory=list)
    total_time_ms: float = 0.0
    gave_up: bool = False
    circuit_broken: bool = False


@dataclass
class CircuitBreakerState:
    """State of circuit breaker"""
    state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.now)


class BackoffCalculator:
    """Calculates retry delays with backoff strategies"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.fibonacci_cache = [0, 1]
        
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        # Calculate base delay
        if self.config.backoff_strategy == BackoffStrategy.CONSTANT:
            delay = self.config.base_delay_ms
            
        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.config.base_delay_ms * attempt
            
        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.config.base_delay_ms * (
                self.config.backoff_multiplier ** (attempt - 1)
            )
            
        elif self.config.backoff_strategy == BackoffStrategy.FIBONACCI:
            delay = self.config.base_delay_ms * self._fibonacci(attempt)
            
        elif self.config.backoff_strategy == BackoffStrategy.POLYNOMIAL:
            delay = self.config.base_delay_ms * (attempt ** 2)
            
        else:
            delay = self.config.base_delay_ms
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay_ms)
        
        # Apply jitter
        delay = self._apply_jitter(delay, attempt)
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number"""
        while len(self.fibonacci_cache) <= n:
            self.fibonacci_cache.append(
                self.fibonacci_cache[-1] + self.fibonacci_cache[-2]
            )
        return self.fibonacci_cache[n]
    
    def _apply_jitter(self, delay: float, attempt: int) -> float:
        """Apply jitter to delay"""
        if self.config.jitter_type == JitterType.NONE:
            return delay
            
        elif self.config.jitter_type == JitterType.FULL:
            # Random value between 0 and delay
            return random.uniform(0, delay)
            
        elif self.config.jitter_type == JitterType.EQUAL:
            # Random value between delay/2 and delay
            return random.uniform(delay / 2, delay)
            
        elif self.config.jitter_type == JitterType.DECORRELATED:
            # Decorrelated jitter (previous delay influences current)
            return random.uniform(
                self.config.base_delay_ms,
                delay * 3
            )
        
        return delay


class ErrorClassifier:
    """Classifies errors to determine if retry is appropriate"""
    
    def classify_error(self, error: Exception) -> FailureType:
        """Classify error type"""
        error_str = str(error).lower()
        
        if any(keyword in error_str for keyword in ['timeout', 'timed out']):
            return FailureType.TIMEOUT
            
        elif any(keyword in error_str for keyword in ['rate limit', 'too many requests']):
            return FailureType.RATE_LIMIT
            
        elif any(keyword in error_str for keyword in ['connection', 'network', 'unavailable']):
            return FailureType.NETWORK
            
        elif any(keyword in error_str for keyword in ['resource', 'capacity', 'overload']):
            return FailureType.RESOURCE
            
        elif any(keyword in error_str for keyword in ['not found', 'invalid', 'forbidden']):
            return FailureType.PERMANENT
            
        else:
            return FailureType.TRANSIENT
    
    def is_retryable(
        self,
        error: Exception,
        config: RetryConfig
    ) -> bool:
        """Determine if error is retryable"""
        error_type = self.classify_error(error)
        return error_type in config.retryable_errors


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 60.0
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.state = CircuitBreakerState()
        
    def record_success(self):
        """Record successful operation"""
        if self.state.state == "half_open":
            self.state.success_count += 1
            
            if self.state.success_count >= self.success_threshold:
                self._transition_to_closed()
                
        elif self.state.state == "closed":
            # Reset failure count on success
            self.state.failure_count = 0
    
    def record_failure(self):
        """Record failed operation"""
        self.state.last_failure_time = datetime.now()
        
        if self.state.state == "half_open":
            # Failure in half-open state reopens circuit
            self._transition_to_open()
            
        elif self.state.state == "closed":
            self.state.failure_count += 1
            
            if self.state.failure_count >= self.failure_threshold:
                self._transition_to_open()
    
    def can_attempt(self) -> bool:
        """Check if operation can be attempted"""
        if self.state.state == "closed":
            return True
            
        elif self.state.state == "open":
            # Check if timeout has elapsed
            if self.state.last_failure_time:
                elapsed = (datetime.now() - self.state.last_failure_time).total_seconds()
                if elapsed >= self.timeout_seconds:
                    self._transition_to_half_open()
                    return True
            return False
            
        elif self.state.state == "half_open":
            return True
        
        return False
    
    def _transition_to_open(self):
        """Transition to open state"""
        self.state.state = "open"
        self.state.failure_count = 0
        self.state.success_count = 0
        self.state.last_state_change = datetime.now()
    
    def _transition_to_half_open(self):
        """Transition to half-open state"""
        self.state.state = "half_open"
        self.state.success_count = 0
        self.state.last_state_change = datetime.now()
    
    def _transition_to_closed(self):
        """Transition to closed state"""
        self.state.state = "closed"
        self.state.failure_count = 0
        self.state.success_count = 0
        self.state.last_state_change = datetime.now()
    
    def get_state(self) -> str:
        """Get current state"""
        return self.state.state


class RetryExecutor:
    """Executes operations with retry logic"""
    
    def __init__(
        self,
        config: RetryConfig,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        self.config = config
        self.backoff_calculator = BackoffCalculator(config)
        self.error_classifier = ErrorClassifier()
        self.circuit_breaker = circuit_breaker
        
    def execute(
        self,
        operation: Callable[[], T],
        operation_name: str = "operation"
    ) -> RetryResult:
        """Execute operation with retry logic"""
        attempts = []
        start_time = time.time()
        
        for attempt_num in range(1, self.config.max_attempts + 1):
            # Check circuit breaker
            if self.circuit_breaker and not self.circuit_breaker.can_attempt():
                return RetryResult(
                    success=False,
                    error="Circuit breaker is open",
                    attempts=attempts,
                    total_time_ms=(time.time() - start_time) * 1000,
                    circuit_broken=True
                )
            
            # Calculate delay for this attempt (except first)
            delay_ms = 0.0
            if attempt_num > 1:
                delay_ms = self.backoff_calculator.calculate_delay(attempt_num - 1)
                time.sleep(delay_ms / 1000.0)
            
            # Execute operation
            attempt_start = time.time()
            attempt = RetryAttempt(
                attempt_number=attempt_num,
                timestamp=datetime.now(),
                delay_ms=delay_ms
            )
            
            try:
                result = operation()
                attempt.success = True
                attempt.duration_ms = (time.time() - attempt_start) * 1000
                attempts.append(attempt)
                
                # Record success with circuit breaker
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                return RetryResult(
                    success=True,
                    value=result,
                    attempts=attempts,
                    total_time_ms=(time.time() - start_time) * 1000
                )
                
            except Exception as e:
                attempt.error = str(e)
                attempt.duration_ms = (time.time() - attempt_start) * 1000
                attempts.append(attempt)
                
                # Record failure with circuit breaker
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                
                # Check if error is retryable
                if not self.error_classifier.is_retryable(e, self.config):
                    return RetryResult(
                        success=False,
                        error=f"Non-retryable error: {str(e)}",
                        attempts=attempts,
                        total_time_ms=(time.time() - start_time) * 1000,
                        gave_up=True
                    )
                
                # Check if we've exhausted attempts
                if attempt_num >= self.config.max_attempts:
                    return RetryResult(
                        success=False,
                        error=f"Max attempts ({self.config.max_attempts}) exceeded: {str(e)}",
                        attempts=attempts,
                        total_time_ms=(time.time() - start_time) * 1000,
                        gave_up=True
                    )
        
        # Should not reach here
        return RetryResult(
            success=False,
            error="Unexpected retry termination",
            attempts=attempts,
            total_time_ms=(time.time() - start_time) * 1000,
            gave_up=True
        )


class RetryPolicy:
    """Manages retry policies for different operations"""
    
    def __init__(self):
        self.policies: Dict[str, RetryConfig] = {}
        self.default_policy = RetryConfig()
        
    def register_policy(self, operation_type: str, config: RetryConfig):
        """Register a retry policy for operation type"""
        self.policies[operation_type] = config
    
    def get_policy(self, operation_type: str) -> RetryConfig:
        """Get retry policy for operation type"""
        return self.policies.get(operation_type, self.default_policy)


class RetryMetrics:
    """Tracks retry metrics"""
    
    def __init__(self):
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_attempts = 0
        self.total_retry_time_ms = 0.0
        self.operations_by_attempts: Dict[int, int] = {}
        self.circuit_breaker_trips = 0
        
    def record_result(self, result: RetryResult):
        """Record operation result"""
        self.total_operations += 1
        self.total_attempts += len(result.attempts)
        self.total_retry_time_ms += result.total_time_ms
        
        if result.success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        
        # Track attempts distribution
        num_attempts = len(result.attempts)
        self.operations_by_attempts[num_attempts] = \
            self.operations_by_attempts.get(num_attempts, 0) + 1
        
        if result.circuit_broken:
            self.circuit_breaker_trips += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if self.total_operations == 0:
            return {"error": "No operations recorded"}
        
        return {
            'total_operations': self.total_operations,
            'successful': self.successful_operations,
            'failed': self.failed_operations,
            'success_rate': self.successful_operations / self.total_operations,
            'avg_attempts': self.total_attempts / self.total_operations,
            'total_retry_time_ms': self.total_retry_time_ms,
            'avg_retry_time_ms': self.total_retry_time_ms / self.total_operations,
            'attempts_distribution': self.operations_by_attempts,
            'circuit_breaker_trips': self.circuit_breaker_trips
        }


class RetryWithBackoffAgent:
    """
    Main agent that orchestrates retry logic with backoff and circuit breaking.
    Handles transient failures gracefully and prevents cascading failures.
    """
    
    def __init__(self):
        self.retry_policy = RetryPolicy()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.metrics = RetryMetrics()
        
    def register_policy(
        self,
        operation_type: str,
        config: RetryConfig
    ):
        """Register retry policy for operation type"""
        self.retry_policy.register_policy(operation_type, config)
    
    def register_circuit_breaker(
        self,
        service_name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 60.0
    ):
        """Register circuit breaker for service"""
        self.circuit_breakers[service_name] = CircuitBreaker(
            failure_threshold,
            success_threshold,
            timeout_seconds
        )
    
    def execute_with_retry(
        self,
        operation: Callable[[], T],
        operation_type: str = "default",
        service_name: Optional[str] = None
    ) -> RetryResult:
        """Execute operation with retry logic"""
        # Get policy and circuit breaker
        config = self.retry_policy.get_policy(operation_type)
        circuit_breaker = self.circuit_breakers.get(service_name) if service_name else None
        
        # Create executor and run
        executor = RetryExecutor(config, circuit_breaker)
        result = executor.execute(operation, operation_type)
        
        # Record metrics
        self.metrics.record_result(result)
        
        return result
    
    def get_circuit_breaker_state(self, service_name: str) -> Optional[str]:
        """Get circuit breaker state for service"""
        cb = self.circuit_breakers.get(service_name)
        return cb.get_state() if cb else None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retry metrics"""
        return self.metrics.get_summary()
    
    def reset_metrics(self):
        """Reset metrics"""
        self.metrics = RetryMetrics()


def demonstrate_retry_with_backoff():
    """Demonstrate retry with backoff pattern"""
    print("\n" + "="*60)
    print("RETRY WITH BACKOFF PATTERN DEMONSTRATION")
    print("="*60)
    
    agent = RetryWithBackoffAgent()
    
    # Scenario 1: Exponential backoff with transient failures
    print("\n" + "-"*60)
    print("Scenario 1: Exponential Backoff with Transient Failures")
    print("-"*60)
    
    # Simulate operation that fails twice then succeeds
    attempt_counter = {'count': 0}
    
    def flaky_operation():
        attempt_counter['count'] += 1
        if attempt_counter['count'] < 3:
            raise Exception("Transient network error")
        return "Success!"
    
    config_exponential = RetryConfig(
        max_attempts=5,
        base_delay_ms=100,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        backoff_multiplier=2.0,
        jitter_type=JitterType.EQUAL
    )
    
    agent.register_policy("api_call", config_exponential)
    
    print("Executing flaky operation with exponential backoff...")
    result = agent.execute_with_retry(flaky_operation, "api_call")
    
    print(f"\nResult: {'✓ Success' if result.success else '✗ Failed'}")
    print(f"Total attempts: {len(result.attempts)}")
    print(f"Total time: {result.total_time_ms:.1f}ms")
    
    for attempt in result.attempts:
        status = "✓" if attempt.success else "✗"
        print(f"  Attempt {attempt.attempt_number}: {status} "
              f"(delay: {attempt.delay_ms:.1f}ms, duration: {attempt.duration_ms:.1f}ms)")
        if attempt.error:
            print(f"    Error: {attempt.error}")
    
    # Scenario 2: Different backoff strategies
    print("\n" + "-"*60)
    print("Scenario 2: Comparing Backoff Strategies")
    print("-"*60)
    
    strategies = [
        (BackoffStrategy.CONSTANT, "Constant"),
        (BackoffStrategy.LINEAR, "Linear"),
        (BackoffStrategy.EXPONENTIAL, "Exponential"),
        (BackoffStrategy.FIBONACCI, "Fibonacci")
    ]
    
    for strategy, name in strategies:
        config = RetryConfig(
            max_attempts=5,
            base_delay_ms=50,
            backoff_strategy=strategy,
            jitter_type=JitterType.NONE
        )
        
        calculator = BackoffCalculator(config)
        delays = [calculator.calculate_delay(i) for i in range(1, 6)]
        
        print(f"\n{name} Backoff:")
        print(f"  Delays: {[f'{d:.0f}ms' for d in delays]}")
    
    # Scenario 3: Circuit breaker integration
    print("\n" + "-"*60)
    print("Scenario 3: Circuit Breaker Integration")
    print("-"*60)
    
    agent.register_circuit_breaker(
        "external_api",
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=5.0
    )
    
    # Simulate multiple failures to trip circuit breaker
    failure_counter = {'count': 0}
    
    def failing_service():
        failure_counter['count'] += 1
        raise Exception("Service unavailable")
    
    print("Attempting operations that will fail and trip circuit breaker...")
    
    for i in range(5):
        result = agent.execute_with_retry(
            failing_service,
            "api_call",
            service_name="external_api"
        )
        
        cb_state = agent.get_circuit_breaker_state("external_api")
        print(f"  Attempt {i+1}: Circuit breaker state = {cb_state}")
        
        if result.circuit_broken:
            print(f"    ⚡ Circuit breaker OPEN - request blocked")
            break
    
    # Scenario 4: Jitter comparison
    print("\n" + "-"*60)
    print("Scenario 4: Jitter Types Comparison")
    print("-"*60)
    
    jitter_types = [
        (JitterType.NONE, "No Jitter"),
        (JitterType.FULL, "Full Jitter"),
        (JitterType.EQUAL, "Equal Jitter")
    ]
    
    for jitter, name in jitter_types:
        config = RetryConfig(
            max_attempts=5,
            base_delay_ms=1000,
            backoff_strategy=BackoffStrategy.EXPONENTIAL,
            jitter_type=jitter
        )
        
        calculator = BackoffCalculator(config)
        # Sample multiple times to show variation
        samples = [
            [calculator.calculate_delay(3) for _ in range(3)]
        ]
        
        print(f"\n{name} (attempt 3, base delay 1000ms):")
        print(f"  Samples: {[f'{d:.0f}ms' for d in samples[0]]}")
    
    # Scenario 5: Non-retryable errors
    print("\n" + "-"*60)
    print("Scenario 5: Non-Retryable Errors")
    print("-"*60)
    
    def permanent_failure():
        raise Exception("Resource not found - permanent error")
    
    config_selective = RetryConfig(
        max_attempts=5,
        retryable_errors=[FailureType.TRANSIENT, FailureType.TIMEOUT]
    )
    
    agent.register_policy("selective_retry", config_selective)
    
    print("Executing operation with permanent error...")
    result = agent.execute_with_retry(permanent_failure, "selective_retry")
    
    print(f"\nResult: {'✓ Success' if result.success else '✗ Failed'}")
    print(f"Attempts made: {len(result.attempts)}")
    print(f"Gave up: {result.gave_up}")
    print(f"Reason: {result.error}")
    
    # Scenario 6: Metrics summary
    print("\n" + "-"*60)
    print("Scenario 6: Retry Metrics Summary")
    print("-"*60)
    
    metrics = agent.get_metrics()
    
    print(f"\nTotal Operations: {metrics['total_operations']}")
    print(f"Successful: {metrics['successful']}")
    print(f"Failed: {metrics['failed']}")
    print(f"Success Rate: {metrics['success_rate']:.1%}")
    print(f"Average Attempts: {metrics['avg_attempts']:.2f}")
    print(f"Total Retry Time: {metrics['total_retry_time_ms']:.1f}ms")
    print(f"Average Retry Time: {metrics['avg_retry_time_ms']:.1f}ms")
    print(f"Circuit Breaker Trips: {metrics['circuit_breaker_trips']}")
    
    print("\nAttempts Distribution:")
    for attempts, count in sorted(metrics['attempts_distribution'].items()):
        print(f"  {attempts} attempt(s): {count} operation(s)")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Demonstrated exponential backoff with jitter")
    print(f"✓ Compared multiple backoff strategies")
    print(f"✓ Integrated circuit breaker to prevent cascading failures")
    print(f"✓ Showed selective retries based on error types")
    print(f"✓ Tracked comprehensive retry metrics")
    print("\n✅ Error Handling & Recovery Category: Pattern 1/5 complete")
    print("Ready for production resilience!")


if __name__ == "__main__":
    demonstrate_retry_with_backoff()
