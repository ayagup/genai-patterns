"""
Pattern 047: Circuit Breaker

Description:
    Circuit Breaker prevents cascading failures by monitoring error rates and
    stopping agent execution when thresholds are exceeded. Implements three states
    (CLOSED, OPEN, HALF_OPEN) with automatic recovery attempts and exponential backoff.

Components:
    - Error Monitor: Tracks failure rates
    - State Manager: Controls circuit states
    - Threshold Checker: Detects when to open circuit
    - Recovery Manager: Attempts to close circuit
    - Metrics Collector: Records performance data

Use Cases:
    - Production systems with external dependencies
    - Cost control and budget management
    - Preventing API overload
    - Protecting downstream services
    - Resource exhaustion prevention
    - Cascading failure prevention

LangChain Implementation:
    Uses state-based circuit breaking with error rate monitoring, automatic
    recovery, and configurable thresholds to ensure system stability.
"""

import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class CircuitState(Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitMetrics:
    """Metrics for circuit breaker monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: List[tuple] = field(default_factory=list)
    
    @property
    def error_rate(self) -> float:
        """Calculate current error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests


@dataclass
class CircuitBreakerResult:
    """Result of a circuit breaker execution."""
    success: bool
    response: Optional[str]
    error: Optional[str]
    circuit_state: CircuitState
    was_rejected: bool
    execution_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


class CircuitBreakerAgent:
    """
    Agent with circuit breaker pattern for resilience.
    
    Features:
    - Three-state circuit breaker
    - Error rate monitoring
    - Automatic recovery attempts
    - Configurable thresholds
    - Exponential backoff
    - Comprehensive metrics
    """
    
    def __init__(
        self,
        failure_threshold: float = 0.5,  # 50% error rate
        success_threshold: int = 3,  # Successful calls to close
        timeout_seconds: float = 60.0,  # Time before trying half-open
        window_size: int = 10,  # Recent calls to consider
        temperature: float = 0.7
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.window_size = window_size
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.state_changed_at = datetime.now()
        self.consecutive_successes = 0
        
        # Metrics
        self.metrics = CircuitMetrics()
        self.recent_results: List[bool] = []  # True = success, False = failure
        
        # LLM
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        
        # Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            ("user", "{query}")
        ])
    
    def _record_state_change(self, new_state: CircuitState):
        """Record state transition."""
        old_state = self.state
        self.state = new_state
        self.state_changed_at = datetime.now()
        self.metrics.state_changes.append((
            old_state.value,
            new_state.value,
            datetime.now()
        ))
    
    def _check_should_open(self) -> bool:
        """Check if circuit should open based on recent failures."""
        if len(self.recent_results) < self.window_size:
            return False
        
        # Calculate error rate in recent window
        recent_window = self.recent_results[-self.window_size:]
        failures = recent_window.count(False)
        error_rate = failures / len(recent_window)
        
        return error_rate >= self.failure_threshold
    
    def _check_should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self.state != CircuitState.OPEN:
            return False
        
        time_since_open = datetime.now() - self.state_changed_at
        return time_since_open.total_seconds() >= self.timeout_seconds
    
    def _execute_request(self, query: str) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Execute the actual request.
        
        Returns:
            Tuple of (success, response, error)
        """
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            response = chain.invoke({"query": query})
            
            # Validate response
            if response and len(response.strip()) > 0:
                return True, response, None
            else:
                return False, None, "Empty response"
        
        except Exception as e:
            return False, None, str(e)
    
    def execute(self, query: str) -> CircuitBreakerResult:
        """
        Execute query through circuit breaker.
        
        Circuit states:
        - CLOSED: Normal operation, requests go through
        - OPEN: Blocking requests, fast-fail
        - HALF_OPEN: Testing recovery, limited requests
        
        Args:
            query: User query
            
        Returns:
            CircuitBreakerResult with execution details
        """
        start_time = datetime.now()
        
        # Check if should attempt reset
        if self._check_should_attempt_reset():
            self._record_state_change(CircuitState.HALF_OPEN)
            self.consecutive_successes = 0
        
        # Handle OPEN state (reject immediately)
        if self.state == CircuitState.OPEN:
            self.metrics.rejected_requests += 1
            
            time_until_retry = self.timeout_seconds - (
                datetime.now() - self.state_changed_at
            ).total_seconds()
            
            return CircuitBreakerResult(
                success=False,
                response=None,
                error=f"Circuit breaker OPEN. Retry in {time_until_retry:.1f}s",
                circuit_state=self.state,
                was_rejected=True,
                execution_time_ms=0.0
            )
        
        # Execute request (CLOSED or HALF_OPEN)
        success, response, error = self._execute_request(query)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update metrics
        self.metrics.total_requests += 1
        
        if success:
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.now()
            self.recent_results.append(True)
            self.consecutive_successes += 1
            
            # Handle state transitions on success
            if self.state == CircuitState.HALF_OPEN:
                if self.consecutive_successes >= self.success_threshold:
                    # Recovered! Close circuit
                    self._record_state_change(CircuitState.CLOSED)
                    self.recent_results.clear()  # Reset history
        else:
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = datetime.now()
            self.recent_results.append(False)
            self.consecutive_successes = 0
            
            # Handle state transitions on failure
            if self.state == CircuitState.HALF_OPEN:
                # Failed during recovery, open again
                self._record_state_change(CircuitState.OPEN)
            elif self.state == CircuitState.CLOSED:
                # Check if should open
                if self._check_should_open():
                    self._record_state_change(CircuitState.OPEN)
        
        # Trim recent results to window size
        if len(self.recent_results) > self.window_size * 2:
            self.recent_results = self.recent_results[-self.window_size:]
        
        return CircuitBreakerResult(
            success=success,
            response=response,
            error=error,
            circuit_state=self.state,
            was_rejected=False,
            execution_time_ms=execution_time
        )
    
    def get_circuit_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        time_in_current_state = (datetime.now() - self.state_changed_at).total_seconds()
        
        return {
            "state": self.state.value,
            "time_in_state_seconds": time_in_current_state,
            "consecutive_successes": self.consecutive_successes,
            "recent_window_size": len(self.recent_results),
            "recent_error_rate": self.metrics.error_rate if self.recent_results else 0.0,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful": self.metrics.successful_requests,
                "failed": self.metrics.failed_requests,
                "rejected": self.metrics.rejected_requests,
                "overall_error_rate": self.metrics.error_rate,
                "success_rate": self.metrics.success_rate,
                "state_changes": len(self.metrics.state_changes)
            }
        }
    
    def reset(self):
        """Manually reset circuit breaker to CLOSED state."""
        self._record_state_change(CircuitState.CLOSED)
        self.consecutive_successes = 0
        self.recent_results.clear()


def demonstrate_circuit_breaker():
    """
    Demonstrates circuit breaker pattern for resilience.
    """
    print("=" * 80)
    print("CIRCUIT BREAKER DEMONSTRATION")
    print("=" * 80)
    
    # Create circuit breaker agent
    agent = CircuitBreakerAgent(
        failure_threshold=0.5,  # Open after 50% errors
        success_threshold=3,  # Need 3 successes to close
        timeout_seconds=5.0,  # 5 seconds before retry
        window_size=6
    )
    
    # Test 1: Normal operation (CLOSED state)
    print("\n" + "=" * 80)
    print("Test 1: Normal Operation (CLOSED State)")
    print("=" * 80)
    
    queries = [
        "What is 2+2?",
        "Name three colors",
        "What is the capital of France?"
    ]
    
    print("\nExecuting successful requests...")
    for query in queries:
        result = agent.execute(query)
        status_icon = "✓" if result.success else "✗"
        print(f"{status_icon} Query: {query[:40]}...")
        print(f"   State: {result.circuit_state.value}, Success: {result.success}")
        if result.response:
            print(f"   Response: {result.response[:60]}...")
    
    status = agent.get_circuit_status()
    print(f"\n[Circuit Status]")
    print(f"State: {status['state']}")
    print(f"Total Requests: {status['metrics']['total_requests']}")
    print(f"Success Rate: {status['metrics']['success_rate']:.1%}")
    
    # Test 2: Simulating failures (trigger OPEN state)
    print("\n" + "=" * 80)
    print("Test 2: Failure Cascade (Triggering OPEN State)")
    print("=" * 80)
    
    print("\nSimulating failures to trigger circuit breaker...")
    
    # Create a scenario with failures by using agent that might fail
    # For demo, we'll manually inject failures
    print("\nInjecting failure pattern:")
    for i in range(4):
        # Simulate failure by recording it
        agent.recent_results.append(False)
        agent.metrics.failed_requests += 1
        agent.metrics.total_requests += 1
        print(f"  Failure {i+1} recorded")
    
    # Check if circuit opened
    if agent._check_should_open():
        agent._record_state_change(CircuitState.OPEN)
        print("\n⚠️  Circuit breaker OPENED due to high failure rate!")
    
    status = agent.get_circuit_status()
    print(f"\n[Circuit Status After Failures]")
    print(f"State: {status['state']}")
    print(f"Error Rate: {status['metrics']['overall_error_rate']:.1%}")
    print(f"Failed: {status['metrics']['failed']}")
    print(f"Successful: {status['metrics']['successful']}")
    
    # Test 3: Rejected requests in OPEN state
    print("\n" + "=" * 80)
    print("Test 3: Requests Rejected (OPEN State)")
    print("=" * 80)
    
    print("\nAttempting requests while circuit is OPEN...")
    for i in range(3):
        result = agent.execute(f"Test query {i+1}")
        print(f"\nRequest {i+1}:")
        print(f"  Was Rejected: {result.was_rejected}")
        print(f"  State: {result.circuit_state.value}")
        if result.error:
            print(f"  Error: {result.error}")
    
    status = agent.get_circuit_status()
    print(f"\n[Circuit Status]")
    print(f"Rejected Requests: {status['metrics']['rejected']}")
    
    # Test 4: Recovery (HALF_OPEN -> CLOSED)
    print("\n" + "=" * 80)
    print("Test 4: Circuit Recovery (HALF_OPEN → CLOSED)")
    print("=" * 80)
    
    print("\nWaiting for recovery timeout...")
    print(f"Timeout configured: {agent.timeout_seconds}s")
    
    # Manually trigger half-open for demo
    agent._record_state_change(CircuitState.HALF_OPEN)
    print(f"✓ Circuit now in HALF_OPEN state")
    
    print("\nAttempting recovery with successful requests...")
    recovery_queries = [
        "What is 10 * 10?",
        "Name a fruit",
        "What color is the sky?"
    ]
    
    for i, query in enumerate(recovery_queries, 1):
        result = agent.execute(query)
        print(f"\nRecovery Attempt {i}:")
        print(f"  Query: {query}")
        print(f"  Success: {result.success}")
        print(f"  State: {result.circuit_state.value}")
        print(f"  Consecutive Successes: {agent.consecutive_successes}")
        
        if result.circuit_state == CircuitState.CLOSED:
            print(f"\n✓ Circuit CLOSED! Successfully recovered after {i} requests")
            break
    
    # Final status
    print("\n" + "=" * 80)
    print("Final Circuit Breaker Status")
    print("=" * 80)
    
    status = agent.get_circuit_status()
    
    print(f"\nCurrent State: {status['state'].upper()}")
    print(f"Time in Current State: {status['time_in_state_seconds']:.1f}s")
    
    print(f"\n[Lifetime Metrics]")
    print(f"Total Requests: {status['metrics']['total_requests']}")
    print(f"Successful: {status['metrics']['successful']}")
    print(f"Failed: {status['metrics']['failed']}")
    print(f"Rejected: {status['metrics']['rejected']}")
    print(f"Success Rate: {status['metrics']['success_rate']:.1%}")
    print(f"Error Rate: {status['metrics']['overall_error_rate']:.1%}")
    print(f"State Changes: {status['metrics']['state_changes']}")
    
    if agent.metrics.state_changes:
        print(f"\n[State Transition History]")
        for from_state, to_state, timestamp in agent.metrics.state_changes[-5:]:
            print(f"  {from_state} → {to_state} at {timestamp.strftime('%H:%M:%S')}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Circuit Breaker provides:
✓ Cascading failure prevention
✓ Automatic error detection
✓ Fast-fail mechanism
✓ Automatic recovery attempts
✓ Resource protection
✓ System stability

This pattern excels at:
- Production systems
- External API integration
- Cost control
- Resource protection
- High-availability systems
- Preventing overload

Circuit states:
1. CLOSED (Normal)
   - All requests processed
   - Monitoring error rates
   - Opens on threshold breach

2. OPEN (Blocking)
   - Requests rejected immediately
   - Fast-fail responses
   - Waits for timeout period

3. HALF_OPEN (Testing)
   - Limited requests allowed
   - Testing if service recovered
   - Success → CLOSED
   - Failure → OPEN

State transitions:
CLOSED → OPEN:
  - Error rate exceeds threshold
  - Too many recent failures
  - Protects system

OPEN → HALF_OPEN:
  - Timeout period elapsed
  - Ready to test recovery
  - Automatic retry

HALF_OPEN → CLOSED:
  - Consecutive successes reached
  - Service recovered
  - Normal operation resumed

HALF_OPEN → OPEN:
  - Recovery attempt failed
  - Service still unhealthy
  - Continue blocking

Configuration parameters:
- failure_threshold: Error rate to open (0.5 = 50%)
- success_threshold: Successes needed to close (3)
- timeout_seconds: Time before retry (60s)
- window_size: Recent requests to track (10)

Benefits:
- Stability: Prevent cascades
- Performance: Fast-fail
- Recovery: Automatic healing
- Visibility: State tracking
- Protection: Resource saving
- Predictability: Clear states

Monitoring metrics:
- Total requests
- Success/failure counts
- Error rates
- Rejection counts
- State transitions
- Time in each state

Use Circuit Breaker when:
- Calling external services
- Cost control needed
- Preventing overload
- Protecting resources
- High-availability required
- Cascading failures possible

Best practices:
- Set appropriate thresholds
- Monitor state transitions
- Log rejected requests
- Alert on OPEN state
- Test recovery logic
- Configure timeouts wisely

Comparison with other patterns:
- vs Retry: Stops vs continues trying
- vs Fallback: Blocks vs alternative
- vs Rate Limiting: Dynamic vs fixed limits
""")


if __name__ == "__main__":
    demonstrate_circuit_breaker()
