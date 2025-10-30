"""
Pattern 049: Rate Limiting & Throttling

Description:
    The Rate Limiting & Throttling pattern controls the frequency and volume of
    agent operations to prevent abuse, manage costs, respect API limits, and ensure
    fair resource allocation. This pattern implements various rate limiting algorithms
    and cost tracking mechanisms.

Components:
    1. Rate Limiter: Enforces request frequency limits
    2. Cost Tracker: Monitors and limits operational costs
    3. Quota Manager: Manages usage quotas per user/tenant
    4. Token Bucket: Allows burst traffic while limiting average rate
    5. Backpressure Handler: Manages queue when limits exceeded

Use Cases:
    - API cost management (OpenAI, Anthropic, etc.)
    - Multi-tenant systems with fair usage
    - Public-facing agents preventing abuse
    - Resource allocation in shared environments
    - Preventing rate limit violations from upstream APIs
    - Budget control for AI operations

LangChain Implementation:
    Uses custom callbacks and wrappers around LLM invocations to track
    and enforce rate limits, cost limits, and usage quotas.
"""

import os
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    TOKEN_BUCKET = "token_bucket"  # Allows bursts, refills over time
    FIXED_WINDOW = "fixed_window"  # Fixed time windows
    SLIDING_WINDOW = "sliding_window"  # Sliding time window
    LEAKY_BUCKET = "leaky_bucket"  # Constant output rate
    CONCURRENCY = "concurrency"  # Limits concurrent requests


class LimitStatus(Enum):
    """Status of rate limit check"""
    ALLOWED = "allowed"
    RATE_LIMITED = "rate_limited"
    COST_LIMITED = "cost_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    CONCURRENT_LIMITED = "concurrent_limited"


class LimitAction(Enum):
    """Action to take when limit exceeded"""
    REJECT = "reject"  # Reject request immediately
    QUEUE = "queue"  # Queue for later processing
    THROTTLE = "throttle"  # Slow down processing
    UPGRADE_PROMPT = "upgrade_prompt"  # Suggest upgrade


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_second: float = 1.0
    requests_per_minute: float = 20.0
    requests_per_hour: float = 100.0
    requests_per_day: float = 1000.0
    max_concurrent_requests: int = 5
    
    # Cost limits
    max_cost_per_request: float = 0.10  # $0.10
    max_cost_per_hour: float = 1.00  # $1.00
    max_cost_per_day: float = 10.00  # $10.00
    
    # Token limits (for token bucket)
    bucket_capacity: int = 10  # Max burst size
    refill_rate: float = 1.0  # Tokens per second
    
    # Action when limited
    limit_action: LimitAction = LimitAction.QUEUE
    max_queue_size: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "requests_per_second": self.requests_per_second,
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "requests_per_day": self.requests_per_day,
            "max_concurrent": self.max_concurrent_requests,
            "cost_limits": {
                "per_request": f"${self.max_cost_per_request:.2f}",
                "per_hour": f"${self.max_cost_per_hour:.2f}",
                "per_day": f"${self.max_cost_per_day:.2f}"
            },
            "bucket_capacity": self.bucket_capacity,
            "refill_rate": f"{self.refill_rate}/s",
            "limit_action": self.limit_action.value,
            "max_queue_size": self.max_queue_size
        }


@dataclass
class UsageMetrics:
    """Usage metrics tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    limited_requests: int = 0
    queued_requests: int = 0
    
    total_cost: float = 0.0
    total_tokens: int = 0
    
    # Time-windowed metrics
    requests_last_second: int = 0
    requests_last_minute: int = 0
    requests_last_hour: int = 0
    requests_last_day: int = 0
    
    cost_last_hour: float = 0.0
    cost_last_day: float = 0.0
    
    # Timestamps for tracking
    request_timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    cost_timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "limited_requests": self.limited_requests,
            "queued_requests": self.queued_requests,
            "success_rate": f"{(self.successful_requests / max(1, self.total_requests)) * 100:.1f}%",
            "limit_rate": f"{(self.limited_requests / max(1, self.total_requests)) * 100:.1f}%",
            "total_cost": f"${self.total_cost:.4f}",
            "total_tokens": self.total_tokens,
            "current_rates": {
                "per_second": self.requests_last_second,
                "per_minute": self.requests_last_minute,
                "per_hour": self.requests_last_hour,
                "per_day": self.requests_last_day
            },
            "recent_costs": {
                "last_hour": f"${self.cost_last_hour:.4f}",
                "last_day": f"${self.cost_last_day:.4f}"
            }
        }


@dataclass
class LimitDecision:
    """Decision about whether to allow request"""
    status: LimitStatus
    allowed: bool
    reason: Optional[str]
    retry_after_seconds: Optional[float]
    current_rate: Dict[str, float]
    estimated_cost: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "allowed": self.allowed,
            "reason": self.reason,
            "retry_after_seconds": self.retry_after_seconds,
            "current_rate": self.current_rate,
            "estimated_cost": f"${self.estimated_cost:.4f}"
        }


@dataclass
class RateLimitedResult:
    """Result from rate-limited operation"""
    success: bool
    response: Optional[str]
    limit_decision: LimitDecision
    actual_cost: float
    actual_tokens: int
    wait_time_seconds: float
    was_queued: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "response": response[:100] + "..." if (response := self.response) and len(response) > 100 else self.response,
            "limit_decision": self.limit_decision.to_dict(),
            "actual_cost": f"${self.actual_cost:.4f}",
            "actual_tokens": self.actual_tokens,
            "wait_time": f"{self.wait_time_seconds:.2f}s",
            "was_queued": self.was_queued
        }


class TokenBucket:
    """Token bucket algorithm for rate limiting with burst support"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = capacity
        self.last_refill = time.time()
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def consume(self, tokens: int = 1) -> Tuple[bool, float]:
        """
        Try to consume tokens. Returns (success, retry_after_seconds).
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, 0.0
        else:
            # Calculate how long until we have enough tokens
            tokens_needed = tokens - self.tokens
            retry_after = tokens_needed / self.refill_rate
            return False, retry_after
    
    def available_tokens(self) -> float:
        """Get current available tokens"""
        self._refill()
        return self.tokens


class RateLimiter:
    """
    Rate limiter with multiple algorithms and cost tracking.
    
    This implementation provides:
    1. Token bucket algorithm for burst support
    2. Sliding window rate tracking
    3. Cost tracking and limits
    4. Concurrent request limiting
    5. Request queuing when limits exceeded
    """
    
    # Estimated token costs per model (tokens per request, cost per 1K tokens)
    MODEL_COSTS = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    }
    
    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET
    ):
        self.config = config or RateLimitConfig()
        self.algorithm = algorithm
        self.metrics = UsageMetrics()
        
        # Token bucket for rate limiting
        self.token_bucket = TokenBucket(
            capacity=self.config.bucket_capacity,
            refill_rate=self.config.refill_rate
        )
        
        # Concurrent request tracking
        self.active_requests = 0
        
        # Request queue
        self.queue: deque = deque()
    
    def _update_windowed_metrics(self):
        """Update time-windowed metrics"""
        now = time.time()
        
        # Clean old timestamps and count recent requests
        self.metrics.request_timestamps = deque(
            [ts for ts in self.metrics.request_timestamps if now - ts < 86400],  # Keep 24h
            maxlen=1000
        )
        
        self.metrics.requests_last_second = sum(
            1 for ts in self.metrics.request_timestamps if now - ts < 1
        )
        self.metrics.requests_last_minute = sum(
            1 for ts in self.metrics.request_timestamps if now - ts < 60
        )
        self.metrics.requests_last_hour = sum(
            1 for ts in self.metrics.request_timestamps if now - ts < 3600
        )
        self.metrics.requests_last_day = len(self.metrics.request_timestamps)
        
        # Update cost metrics
        self.metrics.cost_timestamps = deque(
            [(ts, cost) for ts, cost in self.metrics.cost_timestamps if now - ts < 86400],
            maxlen=1000
        )
        
        self.metrics.cost_last_hour = sum(
            cost for ts, cost in self.metrics.cost_timestamps if now - ts < 3600
        )
        self.metrics.cost_last_day = sum(
            cost for ts, cost in self.metrics.cost_timestamps
        )
    
    def _estimate_cost(self, model: str, prompt_tokens: int = 100, completion_tokens: int = 100) -> float:
        """Estimate cost for a request"""
        costs = self.MODEL_COSTS.get(model, self.MODEL_COSTS["gpt-3.5-turbo"])
        
        input_cost = (prompt_tokens / 1000) * costs["input"]
        output_cost = (completion_tokens / 1000) * costs["output"]
        
        return input_cost + output_cost
    
    def check_limits(self, estimated_tokens: int = 100, model: str = "gpt-3.5-turbo") -> LimitDecision:
        """Check if request should be allowed based on current limits"""
        self._update_windowed_metrics()
        
        estimated_cost = self._estimate_cost(model, estimated_tokens, estimated_tokens)
        
        # Check concurrent requests
        if self.active_requests >= self.config.max_concurrent_requests:
            return LimitDecision(
                status=LimitStatus.CONCURRENT_LIMITED,
                allowed=False,
                reason=f"Concurrent request limit reached ({self.config.max_concurrent_requests})",
                retry_after_seconds=1.0,
                current_rate=self._get_current_rates(),
                estimated_cost=estimated_cost
            )
        
        # Check rate limits using token bucket
        can_proceed, retry_after = self.token_bucket.consume(1)
        if not can_proceed:
            return LimitDecision(
                status=LimitStatus.RATE_LIMITED,
                allowed=False,
                reason=f"Rate limit exceeded. Available tokens: {self.token_bucket.available_tokens():.1f}",
                retry_after_seconds=retry_after,
                current_rate=self._get_current_rates(),
                estimated_cost=estimated_cost
            )
        
        # Check requests per time window
        if self.metrics.requests_last_second >= self.config.requests_per_second:
            return LimitDecision(
                status=LimitStatus.RATE_LIMITED,
                allowed=False,
                reason=f"Requests per second limit exceeded ({self.config.requests_per_second})",
                retry_after_seconds=1.0,
                current_rate=self._get_current_rates(),
                estimated_cost=estimated_cost
            )
        
        if self.metrics.requests_last_minute >= self.config.requests_per_minute:
            return LimitDecision(
                status=LimitStatus.RATE_LIMITED,
                allowed=False,
                reason=f"Requests per minute limit exceeded ({self.config.requests_per_minute})",
                retry_after_seconds=60.0 - (time.time() - min(self.metrics.request_timestamps)),
                current_rate=self._get_current_rates(),
                estimated_cost=estimated_cost
            )
        
        # Check cost limits
        if estimated_cost > self.config.max_cost_per_request:
            return LimitDecision(
                status=LimitStatus.COST_LIMITED,
                allowed=False,
                reason=f"Request cost ${estimated_cost:.4f} exceeds limit ${self.config.max_cost_per_request:.2f}",
                retry_after_seconds=None,
                current_rate=self._get_current_rates(),
                estimated_cost=estimated_cost
            )
        
        if self.metrics.cost_last_hour + estimated_cost > self.config.max_cost_per_hour:
            return LimitDecision(
                status=LimitStatus.COST_LIMITED,
                allowed=False,
                reason=f"Hourly cost limit would be exceeded (${self.metrics.cost_last_hour:.2f} + ${estimated_cost:.4f} > ${self.config.max_cost_per_hour:.2f})",
                retry_after_seconds=3600.0,
                current_rate=self._get_current_rates(),
                estimated_cost=estimated_cost
            )
        
        # All checks passed
        return LimitDecision(
            status=LimitStatus.ALLOWED,
            allowed=True,
            reason=None,
            retry_after_seconds=None,
            current_rate=self._get_current_rates(),
            estimated_cost=estimated_cost
        )
    
    def _get_current_rates(self) -> Dict[str, float]:
        """Get current request rates"""
        return {
            "per_second": self.metrics.requests_last_second,
            "per_minute": self.metrics.requests_last_minute,
            "per_hour": self.metrics.requests_last_hour,
            "per_day": self.metrics.requests_last_day
        }
    
    def _record_request(self, cost: float, tokens: int):
        """Record a request in metrics"""
        now = time.time()
        
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.metrics.total_cost += cost
        self.metrics.total_tokens += tokens
        
        self.metrics.request_timestamps.append(now)
        self.metrics.cost_timestamps.append((now, cost))
        
        self._update_windowed_metrics()
    
    def execute_with_limit(
        self,
        llm: ChatOpenAI,
        prompt: str,
        estimated_tokens: int = 100
    ) -> RateLimitedResult:
        """Execute LLM request with rate limiting"""
        start_time = time.time()
        was_queued = False
        
        # Check limits
        decision = self.check_limits(estimated_tokens, llm.model_name)
        
        if not decision.allowed:
            self.metrics.limited_requests += 1
            
            # Handle based on configured action
            if self.config.limit_action == LimitAction.REJECT:
                return RateLimitedResult(
                    success=False,
                    response=None,
                    limit_decision=decision,
                    actual_cost=0.0,
                    actual_tokens=0,
                    wait_time_seconds=0.0,
                    was_queued=False
                )
            
            elif self.config.limit_action == LimitAction.QUEUE:
                # Wait and retry
                if decision.retry_after_seconds:
                    was_queued = True
                    self.metrics.queued_requests += 1
                    time.sleep(min(decision.retry_after_seconds, 5.0))  # Max 5s wait
                    # Retry
                    decision = self.check_limits(estimated_tokens, llm.model_name)
                    if not decision.allowed:
                        return RateLimitedResult(
                            success=False,
                            response=None,
                            limit_decision=decision,
                            actual_cost=0.0,
                            actual_tokens=0,
                            wait_time_seconds=time.time() - start_time,
                            was_queued=was_queued
                        )
        
        # Execute request
        try:
            self.active_requests += 1
            response = llm.invoke(prompt)
            
            # Estimate actual cost (simplified - production would use actual token counts)
            actual_cost = decision.estimated_cost
            actual_tokens = estimated_tokens * 2  # Rough estimate
            
            # Record metrics
            self._record_request(actual_cost, actual_tokens)
            
            return RateLimitedResult(
                success=True,
                response=response.content,
                limit_decision=decision,
                actual_cost=actual_cost,
                actual_tokens=actual_tokens,
                wait_time_seconds=time.time() - start_time,
                was_queued=was_queued
            )
            
        except Exception as e:
            return RateLimitedResult(
                success=False,
                response=None,
                limit_decision=decision,
                actual_cost=0.0,
                actual_tokens=0,
                wait_time_seconds=time.time() - start_time,
                was_queued=was_queued
            )
        
        finally:
            self.active_requests -= 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current usage metrics"""
        self._update_windowed_metrics()
        return self.metrics.to_dict()
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = UsageMetrics()
        self.token_bucket = TokenBucket(
            capacity=self.config.bucket_capacity,
            refill_rate=self.config.refill_rate
        )


def demonstrate_rate_limiting():
    """Demonstrate rate limiting and throttling patterns"""
    
    print("=" * 80)
    print("PATTERN 049: RATE LIMITING & THROTTLING DEMONSTRATION")
    print("=" * 80)
    print("\nDemonstrating cost control and request rate management\n")
    
    # Create rate-limited LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    
    # Test 1: Basic rate limiting with token bucket
    print("\n" + "=" * 80)
    print("TEST 1: Token Bucket Rate Limiting")
    print("=" * 80)
    
    config1 = RateLimitConfig(
        requests_per_second=2.0,
        requests_per_minute=10.0,
        bucket_capacity=5,  # Allow 5 burst requests
        refill_rate=1.0,  # 1 token per second
        limit_action=LimitAction.REJECT
    )
    
    limiter1 = RateLimiter(config=config1)
    
    print(f"\nConfiguration:")
    print(f"  Bucket capacity: {config1.bucket_capacity} tokens")
    print(f"  Refill rate: {config1.refill_rate} tokens/second")
    print(f"  Action on limit: {config1.limit_action.value}")
    
    print(f"\nSending burst of 7 requests...")
    
    for i in range(7):
        result = limiter1.execute_with_limit(
            llm,
            f"Say 'Request {i+1}' in a creative way",
            estimated_tokens=50
        )
        
        status_symbol = "✓" if result.success else "✗"
        print(f"\n  {status_symbol} Request {i+1}: {result.limit_decision.status.value}")
        
        if result.success:
            print(f"     Response: {result.response[:60]}...")
            print(f"     Cost: {result.actual_cost:.6f}")
        else:
            print(f"     Reason: {result.limit_decision.reason}")
            if result.limit_decision.retry_after_seconds:
                print(f"     Retry after: {result.limit_decision.retry_after_seconds:.1f}s")
        
        time.sleep(0.1)  # Small delay between requests
    
    print(f"\nMetrics after burst:")
    metrics1 = limiter1.get_metrics()
    print(f"  Total requests: {metrics1['total_requests']}")
    print(f"  Successful: {metrics1['successful_requests']}")
    print(f"  Limited: {metrics1['limited_requests']}")
    print(f"  Success rate: {metrics1['success_rate']}")
    
    # Test 2: Cost-based limiting
    print("\n" + "=" * 80)
    print("TEST 2: Cost-Based Limiting")
    print("=" * 80)
    
    config2 = RateLimitConfig(
        max_cost_per_request=0.01,  # $0.01 per request
        max_cost_per_hour=0.05,  # $0.05 per hour
        requests_per_second=10.0,  # High rate limit
        limit_action=LimitAction.REJECT
    )
    
    limiter2 = RateLimiter(config=config2)
    
    print(f"\nConfiguration:")
    print(f"  Max cost per request: ${config2.max_cost_per_request:.2f}")
    print(f"  Max cost per hour: ${config2.max_cost_per_hour:.2f}")
    
    print(f"\nSending requests until cost limit reached...")
    
    for i in range(10):
        result = limiter2.execute_with_limit(
            llm,
            "Tell me a very short fact",
            estimated_tokens=100
        )
        
        status_symbol = "✓" if result.success else "✗"
        print(f"\n  {status_symbol} Request {i+1}:")
        print(f"     Status: {result.limit_decision.status.value}")
        print(f"     Estimated cost: ${result.limit_decision.estimated_cost:.6f}")
        
        if not result.success and result.limit_decision.status == LimitStatus.COST_LIMITED:
            print(f"     Cost limit reached!")
            print(f"     Reason: {result.limit_decision.reason}")
            break
        
        time.sleep(0.2)
    
    metrics2 = limiter2.get_metrics()
    print(f"\nCost metrics:")
    print(f"  Total cost: {metrics2['total_cost']}")
    print(f"  Cost last hour: {metrics2['recent_costs']['last_hour']}")
    print(f"  Requests limited by cost: {metrics2['limited_requests']}")
    
    # Test 3: Request queuing
    print("\n" + "=" * 80)
    print("TEST 3: Request Queuing on Rate Limit")
    print("=" * 80)
    
    config3 = RateLimitConfig(
        requests_per_second=1.0,  # 1 per second
        bucket_capacity=2,  # Small burst
        refill_rate=1.0,
        limit_action=LimitAction.QUEUE,  # Queue instead of reject
        max_queue_size=10
    )
    
    limiter3 = RateLimiter(config=config3)
    
    print(f"\nConfiguration:")
    print(f"  Rate: {config3.requests_per_second} req/s")
    print(f"  Action on limit: {config3.limit_action.value}")
    
    print(f"\nSending 4 rapid requests (will queue)...")
    
    start = time.time()
    for i in range(4):
        result = limiter3.execute_with_limit(
            llm,
            f"Count to {i+1}",
            estimated_tokens=30
        )
        
        elapsed = time.time() - start
        status_symbol = "✓" if result.success else "✗"
        queue_marker = " [QUEUED]" if result.was_queued else ""
        
        print(f"\n  {status_symbol} Request {i+1} (t={elapsed:.1f}s){queue_marker}:")
        print(f"     Wait time: {result.wait_time_seconds:.2f}s")
        print(f"     Status: {result.limit_decision.status.value}")
    
    # Test 4: Concurrent request limiting
    print("\n" + "=" * 80)
    print("TEST 4: Concurrent Request Limiting")
    print("=" * 80)
    
    config4 = RateLimitConfig(
        max_concurrent_requests=2,  # Only 2 at a time
        requests_per_second=10.0,  # High rate
        limit_action=LimitAction.REJECT
    )
    
    limiter4 = RateLimiter(config=config4)
    
    print(f"\nConfiguration:")
    print(f"  Max concurrent: {config4.max_concurrent_requests}")
    
    print(f"\nSimulating concurrent request check...")
    
    # Simulate checking multiple requests
    for i in range(5):
        decision = limiter4.check_limits(estimated_tokens=50)
        print(f"\n  Request {i+1}:")
        print(f"     Status: {decision.status.value}")
        print(f"     Allowed: {decision.allowed}")
        print(f"     Active requests: {limiter4.active_requests}")
        
        if decision.allowed:
            limiter4.active_requests += 1  # Simulate active
    
    # Release requests
    limiter4.active_requests = 0
    
    # Test 5: Multi-tenant rate limiting
    print("\n" + "=" * 80)
    print("TEST 5: Multi-Tenant Rate Limiting Simulation")
    print("=" * 80)
    
    # Create separate limiters per tenant
    tenant_configs = {
        "free_tier": RateLimitConfig(
            requests_per_minute=5.0,
            max_cost_per_hour=0.10,
            limit_action=LimitAction.REJECT
        ),
        "premium_tier": RateLimitConfig(
            requests_per_minute=50.0,
            max_cost_per_hour=5.00,
            limit_action=LimitAction.QUEUE
        )
    }
    
    tenant_limiters = {
        tier: RateLimiter(config=config)
        for tier, config in tenant_configs.items()
    }
    
    print(f"\nTenant configurations:")
    for tier, config in tenant_configs.items():
        print(f"\n  {tier.upper()}:")
        print(f"    Requests/min: {config.requests_per_minute}")
        print(f"    Cost/hour: ${config.max_cost_per_hour:.2f}")
        print(f"    Action: {config.limit_action.value}")
    
    print(f"\nSimulating usage from both tiers...")
    
    for tier, limiter in tenant_limiters.items():
        print(f"\n  Testing {tier}:")
        
        for i in range(3):
            result = limiter.execute_with_limit(
                llm,
                f"Quick response {i+1}",
                estimated_tokens=40
            )
            
            status_symbol = "✓" if result.success else "✗"
            print(f"    {status_symbol} Request {i+1}: {result.limit_decision.status.value}")
        
        metrics = limiter.get_metrics()
        print(f"    Metrics: {metrics['successful_requests']}/{metrics['total_requests']} successful")
    
    # Summary
    print("\n" + "=" * 80)
    print("RATE LIMITING PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Cost Control: Prevents runaway API costs
2. Fair Usage: Ensures fair resource allocation across users
3. API Protection: Respects upstream rate limits
4. Burst Handling: Allows temporary traffic spikes
5. Multi-Tenant Support: Different limits per user tier

Implementation Features:
1. Token bucket algorithm for burst traffic
2. Sliding window rate tracking
3. Cost tracking and budget enforcement
4. Concurrent request limiting
5. Flexible actions (reject, queue, throttle)
6. Per-tenant quota management
7. Comprehensive metrics and monitoring

Rate Limiting Algorithms:
1. Token Bucket: Allows bursts, refills over time (implemented)
2. Fixed Window: Fixed time period limits (basic)
3. Sliding Window: More accurate rate tracking
4. Leaky Bucket: Constant output rate
5. Concurrency Limiting: Max simultaneous requests

Configuration Options:
- requests_per_second/minute/hour/day: Time-based limits
- max_cost_per_request/hour/day: Cost-based limits
- max_concurrent_requests: Simultaneous execution limit
- bucket_capacity: Burst size (token bucket)
- refill_rate: Token replenishment rate
- limit_action: reject, queue, throttle, upgrade_prompt
- max_queue_size: Maximum queued requests

Use Cases:
- Public APIs with free/paid tiers
- Multi-tenant SaaS applications
- Budget-constrained environments
- High-traffic agent systems
- Preventing API abuse
- Fair resource allocation

Best Practices:
1. Set appropriate limits based on actual usage patterns
2. Monitor metrics to detect abuse or issues
3. Provide clear feedback when limits exceeded
4. Implement graceful degradation strategies
5. Use different tiers for different user types
6. Track costs accurately including all API calls
7. Alert on approaching limits
8. Test limit enforcement thoroughly

Production Considerations:
- Distributed rate limiting (Redis, etc.)
- Per-user vs per-tenant vs global limits
- Dynamic limit adjustment based on load
- Retry-after headers in responses
- Webhook notifications on limit exceeded
- Integration with billing systems
- Fallback to cheaper models when limited
- Caching to reduce API calls

Comparison with Related Patterns:
- vs. Circuit Breaker: Prevents abuse vs prevents cascading failures
- vs. Throttling: Limits rate vs slows execution
- vs. Quota Management: Per-request limits vs total allowance
- vs. Backpressure: Push-based limits vs pull-based flow control

The Rate Limiting pattern is essential for production AI systems to manage
costs, ensure fair usage, and prevent abuse while maintaining good user
experience through appropriate handling of limit exceeded scenarios.
""")


if __name__ == "__main__":
    demonstrate_rate_limiting()
