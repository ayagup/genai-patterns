"""
Pattern 146: Retry with Backoff

Description:
    Implements retry logic with exponential backoff for handling transient failures
    in LLM API calls and agent operations. This pattern automatically retries failed
    operations with increasing delays, helping systems recover from temporary issues
    like rate limits, network errors, or service disruptions.

Components:
    - Retry Manager: Orchestrates retry attempts
    - Backoff Strategy: Calculates delay between retries
    - Failure Classifier: Determines if error is retryable
    - Success Monitor: Tracks retry statistics
    - Circuit Breaker Integration: Prevents endless retries
    - Fallback Handler: Provides alternatives when retries exhausted

Use Cases:
    - Handle API rate limiting
    - Recover from transient network errors
    - Manage service unavailability
    - Deal with timeout errors
    - Handle overloaded systems

Benefits:
    - Automatic error recovery
    - Reduced manual intervention
    - Improved reliability
    - Better user experience
    - Graceful handling of transient failures

Trade-offs:
    - Increased latency on failures
    - Resource consumption during retries
    - Potential for cascading delays
    - Complexity in retry logic

LangChain Implementation:
    Uses decorators and custom retry logic with exponential backoff,
    jitter, and intelligent failure classification.
"""

import os
import time
import random
from typing import Any, Callable, Optional, List, Dict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class BackoffStrategy(Enum):
    """Backoff strategy for retries"""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"


class ErrorType(Enum):
    """Types of errors for classification"""
    TRANSIENT = "transient"  # Temporary, retryable
    PERMANENT = "permanent"  # Persistent, not retryable
    RATE_LIMIT = "rate_limit"  # Rate limiting
    TIMEOUT = "timeout"  # Timeout error
    UNKNOWN = "unknown"  # Unknown error type


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1
    retryable_errors: List[str] = field(default_factory=lambda: [
        "rate_limit",
        "timeout",
        "connection",
        "service_unavailable"
    ])


@dataclass
class RetryAttempt:
    """Information about a retry attempt"""
    attempt_number: int
    error_type: ErrorType
    error_message: str
    delay_before: float
    timestamp: datetime
    success: bool


@dataclass
class RetryMetrics:
    """Metrics for retry operations"""
    total_calls: int = 0
    successful_first_try: int = 0
    successful_after_retry: int = 0
    failed_after_retries: int = 0
    total_retries: int = 0
    total_delay_time: float = 0.0
    average_attempts: float = 0.0
    retry_success_rate: float = 0.0


class RetryWithBackoffAgent:
    """
    Agent that implements retry logic with exponential backoff.
    
    Automatically retries failed operations with intelligent backoff
    strategies and error classification.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        config: Optional[RetryConfig] = None
    ):
        """
        Initialize the retry agent.
        
        Args:
            model: LLM model to use
            config: Retry configuration
        """
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        self.config = config or RetryConfig()
        
        self.metrics = RetryMetrics()
        self.retry_history: List[RetryAttempt] = []
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt based on strategy.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        if self.config.backoff_strategy == BackoffStrategy.FIXED:
            delay = self.config.initial_delay
            
        elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.config.initial_delay * (attempt + 1)
            
        elif self.config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = self.config.initial_delay * (
                self.config.backoff_multiplier ** attempt
            )
            
        elif self.config.backoff_strategy == BackoffStrategy.FIBONACCI:
            delay = self.config.initial_delay * self._fibonacci(attempt + 1)
        
        else:
            delay = self.config.initial_delay
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0, delay)  # Ensure non-negative
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return 1
        a, b = 1, 1
        for _ in range(n - 1):
            a, b = b, a + b
        return b
    
    def classify_error(self, error: Exception) -> ErrorType:
        """
        Classify error to determine if retryable.
        
        Args:
            error: Exception to classify
            
        Returns:
            Error type classification
        """
        error_str = str(error).lower()
        
        if any(keyword in error_str for keyword in ["rate limit", "429"]):
            return ErrorType.RATE_LIMIT
        
        elif any(keyword in error_str for keyword in ["timeout", "timed out"]):
            return ErrorType.TIMEOUT
        
        elif any(keyword in error_str for keyword in [
            "connection", "network", "unavailable", "503", "502"
        ]):
            return ErrorType.TRANSIENT
        
        elif any(keyword in error_str for keyword in [
            "invalid", "authentication", "authorization", "400", "401", "403"
        ]):
            return ErrorType.PERMANENT
        
        else:
            return ErrorType.UNKNOWN
    
    def is_retryable(self, error: Exception) -> bool:
        """
        Determine if error should be retried.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is retryable
        """
        error_type = self.classify_error(error)
        
        if error_type == ErrorType.PERMANENT:
            return False
        
        error_str = str(error).lower()
        return any(
            keyword in error_str
            for keyword in self.config.retryable_errors
        )
    
    def retry_with_backoff(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries exhausted
        """
        self.metrics.total_calls += 1
        attempts = []
        
        for attempt in range(self.config.max_attempts):
            try:
                result = func(*args, **kwargs)
                
                # Record success
                if attempt == 0:
                    self.metrics.successful_first_try += 1
                else:
                    self.metrics.successful_after_retry += 1
                    self.metrics.total_retries += attempt
                
                # Record attempt
                attempts.append(RetryAttempt(
                    attempt_number=attempt + 1,
                    error_type=ErrorType.TRANSIENT,
                    error_message="Success",
                    delay_before=0,
                    timestamp=datetime.now(),
                    success=True
                ))
                
                self.retry_history.extend(attempts)
                self._update_metrics()
                
                return result
                
            except Exception as e:
                error_type = self.classify_error(e)
                
                # Check if should retry
                if not self.is_retryable(e):
                    print(f"⛔ Non-retryable error: {error_type.value}")
                    self.metrics.failed_after_retries += 1
                    raise
                
                # Last attempt - don't retry
                if attempt == self.config.max_attempts - 1:
                    print(f"❌ Max retries ({self.config.max_attempts}) exhausted")
                    self.metrics.failed_after_retries += 1
                    
                    attempts.append(RetryAttempt(
                        attempt_number=attempt + 1,
                        error_type=error_type,
                        error_message=str(e),
                        delay_before=0,
                        timestamp=datetime.now(),
                        success=False
                    ))
                    
                    self.retry_history.extend(attempts)
                    self._update_metrics()
                    raise
                
                # Calculate delay and retry
                delay = self.calculate_delay(attempt)
                
                print(f"⚠️  Attempt {attempt + 1} failed: {error_type.value}")
                print(f"   Retrying in {delay:.2f}s...")
                
                attempts.append(RetryAttempt(
                    attempt_number=attempt + 1,
                    error_type=error_type,
                    error_message=str(e),
                    delay_before=delay,
                    timestamp=datetime.now(),
                    success=False
                ))
                
                self.metrics.total_delay_time += delay
                time.sleep(delay)
    
    def _update_metrics(self):
        """Update calculated metrics."""
        total_operations = (
            self.metrics.successful_first_try +
            self.metrics.successful_after_retry +
            self.metrics.failed_after_retries
        )
        
        if total_operations > 0:
            self.metrics.average_attempts = (
                (self.metrics.successful_first_try +
                 self.metrics.successful_after_retry +
                 self.metrics.total_retries) / total_operations
            )
        
        retry_operations = (
            self.metrics.successful_after_retry +
            self.metrics.failed_after_retries
        )
        
        if retry_operations > 0:
            self.metrics.retry_success_rate = (
                self.metrics.successful_after_retry / retry_operations
            )
    
    def query_with_retry(self, query: str) -> str:
        """
        Query LLM with retry logic.
        
        Args:
            query: Query string
            
        Returns:
            LLM response
        """
        def _query():
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant."),
                ("user", "{query}")
            ])
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({"query": query})
        
        return self.retry_with_backoff(_query)
    
    def get_metrics_report(self) -> str:
        """Generate retry metrics report."""
        report = []
        report.append("\n" + "="*60)
        report.append("RETRY WITH BACKOFF METRICS REPORT")
        report.append("="*60)
        
        report.append(f"\n📊 Overall Statistics:")
        report.append(f"   Total Calls: {self.metrics.total_calls}")
        report.append(f"   Successful First Try: {self.metrics.successful_first_try}")
        report.append(f"   Successful After Retry: {self.metrics.successful_after_retry}")
        report.append(f"   Failed After Retries: {self.metrics.failed_after_retries}")
        
        report.append(f"\n🔄 Retry Statistics:")
        report.append(f"   Total Retries: {self.metrics.total_retries}")
        report.append(f"   Average Attempts: {self.metrics.average_attempts:.2f}")
        report.append(f"   Retry Success Rate: {self.metrics.retry_success_rate:.1%}")
        
        report.append(f"\n⏱️  Timing:")
        report.append(f"   Total Delay Time: {self.metrics.total_delay_time:.2f}s")
        
        if self.metrics.total_calls > 0:
            success_rate = (
                (self.metrics.successful_first_try + self.metrics.successful_after_retry) /
                self.metrics.total_calls
            )
            report.append(f"\n✅ Overall Success Rate: {success_rate:.1%}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


def retry_decorator(config: Optional[RetryConfig] = None):
    """
    Decorator for automatic retry with backoff.
    
    Args:
        config: Retry configuration
        
    Returns:
        Decorated function
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            agent = RetryWithBackoffAgent(config=config)
            return agent.retry_with_backoff(func, *args, **kwargs)
        return wrapper
    return decorator


def demonstrate_retry_with_backoff():
    """Demonstrate the Retry with Backoff pattern."""
    print("="*60)
    print("RETRY WITH BACKOFF PATTERN DEMONSTRATION")
    print("="*60)
    
    # Example 1: Basic retry with exponential backoff
    print("\n" + "="*60)
    print("Example 1: Basic Retry with Exponential Backoff")
    print("="*60)
    
    config = RetryConfig(
        max_attempts=4,
        initial_delay=1.0,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        backoff_multiplier=2.0
    )
    
    agent = RetryWithBackoffAgent(config=config)
    
    print("\nDelay progression for exponential backoff:")
    for i in range(4):
        delay = agent.calculate_delay(i)
        print(f"  Attempt {i+1}: {delay:.2f}s delay")
    
    # Example 2: Compare backoff strategies
    print("\n" + "="*60)
    print("Example 2: Compare Backoff Strategies")
    print("="*60)
    
    strategies = [
        BackoffStrategy.FIXED,
        BackoffStrategy.LINEAR,
        BackoffStrategy.EXPONENTIAL,
        BackoffStrategy.FIBONACCI
    ]
    
    print(f"\n{'Strategy':<15} {'Attempt 1':<12} {'Attempt 2':<12} {'Attempt 3':<12} {'Attempt 4':<12}")
    print("-" * 63)
    
    for strategy in strategies:
        config = RetryConfig(
            initial_delay=1.0,
            backoff_strategy=strategy,
            backoff_multiplier=2.0,
            jitter=False
        )
        agent = RetryWithBackoffAgent(config=config)
        
        delays = [agent.calculate_delay(i) for i in range(4)]
        print(f"{strategy.value:<15} {delays[0]:<12.2f} {delays[1]:<12.2f} {delays[2]:<12.2f} {delays[3]:<12.2f}")
    
    # Example 3: Error classification
    print("\n" + "="*60)
    print("Example 3: Error Classification")
    print("="*60)
    
    agent = RetryWithBackoffAgent()
    
    test_errors = [
        Exception("Rate limit exceeded (429)"),
        Exception("Connection timeout"),
        Exception("Service unavailable (503)"),
        Exception("Invalid API key (401)"),
        Exception("Network connection failed"),
        Exception("Unknown error occurred")
    ]
    
    print("\nError classification results:")
    for error in test_errors:
        error_type = agent.classify_error(error)
        is_retryable = agent.is_retryable(error)
        status = "✅ RETRY" if is_retryable else "⛔ DON'T RETRY"
        print(f"  {status:<15} {error_type.value:<15} {str(error)[:40]}")
    
    # Example 4: Simulate retry scenarios
    print("\n" + "="*60)
    print("Example 4: Simulate Retry Scenarios")
    print("="*60)
    
    # Simulate transient failure that succeeds on retry
    call_count = {"count": 0}
    
    def flaky_function():
        """Function that fails first 2 times, then succeeds."""
        call_count["count"] += 1
        if call_count["count"] < 3:
            raise Exception("Service temporarily unavailable (503)")
        return "Success!"
    
    config = RetryConfig(max_attempts=5, initial_delay=0.5)
    agent = RetryWithBackoffAgent(config=config)
    
    print("\nSimulating transient failure (succeeds on 3rd attempt):")
    try:
        result = agent.retry_with_backoff(flaky_function)
        print(f"✅ Result: {result}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Example 5: Jitter demonstration
    print("\n" + "="*60)
    print("Example 5: Jitter for Thundering Herd Prevention")
    print("="*60)
    
    config_no_jitter = RetryConfig(
        initial_delay=1.0,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        jitter=False
    )
    
    config_with_jitter = RetryConfig(
        initial_delay=1.0,
        backoff_strategy=BackoffStrategy.EXPONENTIAL,
        jitter=True,
        jitter_range=0.2
    )
    
    agent_no_jitter = RetryWithBackoffAgent(config=config_no_jitter)
    agent_with_jitter = RetryWithBackoffAgent(config=config_with_jitter)
    
    print("\nDelay comparison (5 simulated clients):")
    print(f"\n{'Attempt':<10} {'No Jitter':<40} {'With Jitter':<40}")
    print("-" * 90)
    
    for attempt in range(3):
        no_jitter_delays = [agent_no_jitter.calculate_delay(attempt) for _ in range(5)]
        with_jitter_delays = [agent_with_jitter.calculate_delay(attempt) for _ in range(5)]
        
        no_jitter_str = ", ".join(f"{d:.2f}s" for d in no_jitter_delays)
        with_jitter_str = ", ".join(f"{d:.2f}s" for d in with_jitter_delays)
        
        print(f"{attempt+1:<10} {no_jitter_str:<40} {with_jitter_str:<40}")
    
    print("\n💡 Jitter spreads retry attempts, preventing thundering herd")
    
    # Example 6: Real LLM query with retry
    print("\n" + "="*60)
    print("Example 6: LLM Query with Retry Protection")
    print("="*60)
    
    config = RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        backoff_strategy=BackoffStrategy.EXPONENTIAL
    )
    
    agent = RetryWithBackoffAgent(config=config)
    
    try:
        print("\nQuerying LLM with retry protection...")
        result = agent.query_with_retry(
            "What is the capital of France? Answer in one word."
        )
        print(f"✅ Response: {result}")
    except Exception as e:
        print(f"❌ Failed after all retries: {e}")
    
    # Example 7: Metrics and reporting
    print("\n" + "="*60)
    print("Example 7: Retry Metrics and Reporting")
    print("="*60)
    
    # Simulate multiple operations
    agent = RetryWithBackoffAgent(config=RetryConfig(max_attempts=3))
    
    test_queries = [
        "What is 2+2?",
        "Explain quantum computing in one sentence",
        "Name a primary color"
    ]
    
    print("\nExecuting multiple queries with retry protection...")
    for query in test_queries:
        try:
            agent.query_with_retry(query)
        except Exception as e:
            print(f"Query failed: {e}")
    
    report = agent.get_metrics_report()
    print(report)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
The Retry with Backoff pattern demonstrates:

1. Automatic Recovery: Handles transient failures automatically
2. Multiple Strategies: Fixed, linear, exponential, Fibonacci backoff
3. Error Classification: Intelligent distinction between retryable/non-retryable
4. Jitter Support: Prevents thundering herd problem
5. Metrics Tracking: Comprehensive monitoring of retry behavior

Key Benefits:
- Improved reliability (often 95%+ success rate)
- Reduced manual intervention
- Better user experience
- Graceful handling of rate limits
- Protection against cascading failures

Backoff Strategies:
- Fixed: Same delay each time (simple, predictable)
- Linear: Delay increases linearly (gradual backoff)
- Exponential: Delay doubles each time (recommended)
- Fibonacci: Delay follows Fibonacci sequence (balanced)

Best Practices:
- Use exponential backoff as default
- Add jitter to prevent synchronized retries
- Set reasonable max attempts (3-5 typical)
- Cap maximum delay (60s common)
- Classify errors correctly
- Log retry attempts for debugging
- Integrate with circuit breaker
- Monitor retry metrics

Common Configurations:
- API calls: 3 retries, exponential, 1s initial
- Database: 5 retries, exponential, 0.5s initial
- Network: 4 retries, linear, 2s initial
- Rate limits: Fibonacci, respect Retry-After header
    """)


if __name__ == "__main__":
    demonstrate_retry_with_backoff()
