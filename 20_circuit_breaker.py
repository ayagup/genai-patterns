"""
Circuit Breaker Pattern
Stops agent when error rate exceeds threshold
"""
from typing import Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import time
import random
class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered
@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout_seconds: int = 60   # Time before attempting recovery
@dataclass
class CircuitStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures"""
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self.state_changed_at = datetime.now()
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                print(f"  [{self.name}] Circuit transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.state_changed_at = datetime.now()
            else:
                self.stats.rejected_requests += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Rejecting request."
                )
        # Attempt to execute
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.stats.last_failure_time is None:
            return True
        elapsed = datetime.now() - self.stats.last_failure_time
        return elapsed.total_seconds() >= self.config.timeout_seconds
    def _on_success(self):
        """Handle successful request"""
        self.stats.total_requests += 1
        self.stats.successful_requests += 1
        self.stats.consecutive_failures = 0
        self.stats.consecutive_successes += 1
        print(f"  [{self.name}] ✓ Request succeeded (consecutive: {self.stats.consecutive_successes})")
        if self.state == CircuitState.HALF_OPEN:
            if self.stats.consecutive_successes >= self.config.success_threshold:
                print(f"  [{self.name}] Circuit closing (service recovered)")
                self.state = CircuitState.CLOSED
                self.state_changed_at = datetime.now()
                self.stats.consecutive_successes = 0
    def _on_failure(self):
        """Handle failed request"""
        self.stats.total_requests += 1
        self.stats.failed_requests += 1
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.last_failure_time = datetime.now()
        print(f"  [{self.name}] ✗ Request failed (consecutive: {self.stats.consecutive_failures})")
        # Check if we should open the circuit
        if self.stats.consecutive_failures >= self.config.failure_threshold:
            if self.state != CircuitState.OPEN:
                print(f"  [{self.name}] !!! Circuit opening (failure threshold reached)")
                self.state = CircuitState.OPEN
                self.state_changed_at = datetime.now()
    def get_stats(self) -> dict:
        """Get circuit breaker statistics"""
        total = self.stats.total_requests
        success_rate = (self.stats.successful_requests / total * 100) if total > 0 else 0
        return {
            'name': self.name,
            'state': self.state.value,
            'total_requests': total,
            'successful': self.stats.successful_requests,
            'failed': self.stats.failed_requests,
            'rejected': self.stats.rejected_requests,
            'success_rate': success_rate,
            'consecutive_failures': self.stats.consecutive_failures,
            'time_in_state': (datetime.now() - self.state_changed_at).total_seconds()
        }
    def reset(self):
        """Manually reset circuit breaker"""
        print(f"  [{self.name}] Circuit manually reset")
        self.state = CircuitState.CLOSED
        self.stats.consecutive_failures = 0
        self.stats.consecutive_successes = 0
        self.state_changed_at = datetime.now()
class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass
class ProtectedAgent:
    """Agent with circuit breaker protection"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        # Create circuit breakers for different operations
        self.api_circuit = CircuitBreaker(
            "API_Circuit",
            CircuitBreakerConfig(failure_threshold=3, timeout_seconds=5)
        )
        self.db_circuit = CircuitBreaker(
            "DB_Circuit",
            CircuitBreakerConfig(failure_threshold=5, timeout_seconds=10)
        )
    def call_external_api(self, endpoint: str) -> dict:
        """Call external API with circuit breaker protection"""
        print(f"\nCalling API: {endpoint}")
        def api_call():
            # Simulate API call that might fail
            time.sleep(0.1)
            # Simulate occasional failures
            if random.random() < 0.3:  # 30% failure rate
                raise Exception("API connection timeout")
            return {"status": "success", "data": "API response"}
        try:
            result = self.api_circuit.call(api_call)
            return result
        except CircuitBreakerOpenError as e:
            print(f"  Circuit breaker prevented call: {e}")
            return {"status": "error", "message": "Circuit breaker open"}
        except Exception as e:
            print(f"  API call failed: {e}")
            return {"status": "error", "message": str(e)}
    def query_database(self, query: str) -> dict:
        """Query database with circuit breaker protection"""
        print(f"\nQuerying DB: {query}")
        def db_query():
            time.sleep(0.05)
            # Simulate occasional failures
            if random.random() < 0.2:  # 20% failure rate
                raise Exception("Database connection lost")
            return {"status": "success", "rows": 10}
        try:
            result = self.db_circuit.call(db_query)
            return result
        except CircuitBreakerOpenError as e:
            print(f"  Circuit breaker prevented call: {e}")
            return {"status": "error", "message": "Circuit breaker open"}
        except Exception as e:
            print(f"  DB query failed: {e}")
            return {"status": "error", "message": str(e)}
    def print_circuit_status(self):
        """Print status of all circuit breakers"""
        print(f"\n{'='*70}")
        print(f"CIRCUIT BREAKER STATUS")
        print(f"{'='*70}")
        for circuit in [self.api_circuit, self.db_circuit]:
            stats = circuit.get_stats()
            print(f"\n{stats['name']}:")
            print(f"  State: {stats['state'].upper()}")
            print(f"  Total Requests: {stats['total_requests']}")
            print(f"  Successful: {stats['successful']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Rejected: {stats['rejected']}")
            print(f"  Success Rate: {stats['success_rate']:.1f}%")
            print(f"  Consecutive Failures: {stats['consecutive_failures']}")
            print(f"  Time in Current State: {stats['time_in_state']:.1f}s")
# Usage
if __name__ == "__main__":
    print("="*80)
    print("CIRCUIT BREAKER PATTERN DEMONSTRATION")
    print("="*80)
    agent = ProtectedAgent("agent-001")
    # Simulate multiple API calls
    print("\n" + "="*80)
    print("Testing API Circuit Breaker")
    print("="*80)
    for i in range(15):
        print(f"\n--- Request {i+1} ---")
        agent.call_external_api(f"/api/endpoint/{i}")
        time.sleep(0.2)
        # Show status after some requests
        if i == 7:
            agent.print_circuit_status()
    # Wait for circuit to potentially reset
    print("\n\nWaiting for circuit breaker timeout...")
    time.sleep(6)
    print("\n" + "="*80)
    print("Retrying after timeout")
    print("="*80)
    for i in range(5):
        print(f"\n--- Retry Request {i+1} ---")
        agent.call_external_api(f"/api/endpoint/retry-{i}")
        time.sleep(0.2)
    # Final status
    agent.print_circuit_status()
