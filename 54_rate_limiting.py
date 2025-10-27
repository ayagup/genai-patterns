"""
Rate Limiting & Throttling Pattern
Controls frequency of agent actions
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import time
class RateLimitStrategy(Enum):
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    max_requests: int
    window_seconds: int
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_size: Optional[int] = None  # For token bucket
@dataclass
class RequestRecord:
    """Record of a request"""
    timestamp: datetime
    user_id: str
    resource: str
    allowed: bool
class RateLimiter:
    """Rate limiter implementation"""
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests: Dict[str, deque] = {}  # user_id -> request times
        self.tokens: Dict[str, float] = {}  # For token bucket
        self.last_refill: Dict[str, datetime] = {}
        self.request_history: List[RequestRecord] = []
    def is_allowed(self, user_id: str, resource: str = "default") -> bool:
        """Check if request is allowed"""
        key = f"{user_id}:{resource}"
        current_time = datetime.now()
        if self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            allowed = self._sliding_window_check(key, current_time)
        elif self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            allowed = self._token_bucket_check(key, current_time)
        elif self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
            allowed = self._fixed_window_check(key, current_time)
        else:
            allowed = True
        # Record request
        self.request_history.append(RequestRecord(
            timestamp=current_time,
            user_id=user_id,
            resource=resource,
            allowed=allowed
        ))
        return allowed
    def _sliding_window_check(self, key: str, current_time: datetime) -> bool:
        """Sliding window rate limiting"""
        if key not in self.requests:
            self.requests[key] = deque()
        # Remove old requests outside window
        window_start = current_time - timedelta(seconds=self.config.window_seconds)
        while self.requests[key] and self.requests[key][0] < window_start:
            self.requests[key].popleft()
        # Check if under limit
        if len(self.requests[key]) < self.config.max_requests:
            self.requests[key].append(current_time)
            return True
        return False
    def _token_bucket_check(self, key: str, current_time: datetime) -> bool:
        """Token bucket rate limiting"""
        if key not in self.tokens:
            self.tokens[key] = self.config.burst_size or self.config.max_requests
            self.last_refill[key] = current_time
        # Refill tokens
        time_passed = (current_time - self.last_refill[key]).total_seconds()
        refill_rate = self.config.max_requests / self.config.window_seconds
        tokens_to_add = time_passed * refill_rate
        max_tokens = self.config.burst_size or self.config.max_requests
        self.tokens[key] = min(max_tokens, self.tokens[key] + tokens_to_add)
        self.last_refill[key] = current_time
        # Try to consume token
        if self.tokens[key] >= 1.0:
            self.tokens[key] -= 1.0
            return True
        return False
    def _fixed_window_check(self, key: str, current_time: datetime) -> bool:
        """Fixed window rate limiting"""
        if key not in self.requests:
            self.requests[key] = deque()
        # Calculate current window
        window_start = datetime(
            current_time.year,
            current_time.month,
            current_time.day,
            current_time.hour,
            current_time.minute // (self.config.window_seconds // 60),
            0
        )
        # Remove requests from previous windows
        self.requests[key] = deque([
            t for t in self.requests[key] if t >= window_start
        ])
        # Check limit
        if len(self.requests[key]) < self.config.max_requests:
            self.requests[key].append(current_time)
            return True
        return False
    def get_remaining(self, user_id: str, resource: str = "default") -> int:
        """Get remaining requests for user"""
        key = f"{user_id}:{resource}"
        if key not in self.requests:
            return self.config.max_requests
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return int(self.tokens.get(key, 0))
        else:
            return self.config.max_requests - len(self.requests[key])
    def reset(self, user_id: str, resource: str = "default"):
        """Reset rate limit for user"""
        key = f"{user_id}:{resource}"
        if key in self.requests:
            self.requests[key].clear()
        if key in self.tokens:
            self.tokens[key] = self.config.max_requests
class RateLimitedAgent:
    """Agent with rate limiting"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        # Different rate limits for different operations
        self.limiters = {
            'query': RateLimiter(RateLimitConfig(
                max_requests=10,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW
            )),
            'expensive_operation': RateLimiter(RateLimitConfig(
                max_requests=3,
                window_seconds=60,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                burst_size=5
            ))
        }
    def process_request(self, user_id: str, operation: str, request: str) -> Dict[str, Any]:
        """Process request with rate limiting"""
        print(f"\n[{self.agent_id}] Processing {operation} for user {user_id}")
        limiter = self.limiters.get(operation, self.limiters['query'])
        # Check rate limit
        if not limiter.is_allowed(user_id, operation):
            remaining = limiter.get_remaining(user_id, operation)
            print(f"  ✗ Rate limit exceeded")
            print(f"  Remaining: {remaining}")
            return {
                'success': False,
                'error': 'Rate limit exceeded',
                'retry_after_seconds': limiter.config.window_seconds,
                'remaining': remaining
            }
        # Process request
        remaining = limiter.get_remaining(user_id, operation)
        print(f"  ✓ Request allowed")
        print(f"  Remaining: {remaining}")
        result = self._execute_operation(operation, request)
        return {
            'success': True,
            'result': result,
            'remaining': remaining
        }
    def _execute_operation(self, operation: str, request: str) -> Any:
        """Execute the operation"""
        import time
        time.sleep(0.1)  # Simulate work
        return f"Result for {operation}: {request}"
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        stats = {}
        for name, limiter in self.limiters.items():
            total = len(limiter.request_history)
            allowed = sum(1 for r in limiter.request_history if r.allowed)
            rejected = total - allowed
            stats[name] = {
                'total_requests': total,
                'allowed': allowed,
                'rejected': rejected,
                'rejection_rate': rejected / total if total > 0 else 0
            }
        return stats
# Usage
if __name__ == "__main__":
    print("="*80)
    print("RATE LIMITING & THROTTLING PATTERN DEMONSTRATION")
    print("="*80)
    agent = RateLimitedAgent("rate-limited-agent")
    # Simulate requests from user
    user_id = "user_123"
    print("\nSimulating burst of requests...")
    # Test query rate limit (10 per minute)
    for i in range(15):
        result = agent.process_request(user_id, 'query', f"Query {i+1}")
        if not result['success']:
            print(f"  Request {i+1}: BLOCKED - {result['error']}")
        time.sleep(0.1)
    print("\n" + "="*80)
    print("\nTesting expensive operation limit (3 per minute with burst)...")
    # Test expensive operation limit
    for i in range(8):
        result = agent.process_request(user_id, 'expensive_operation', f"Expensive {i+1}")
        time.sleep(0.2)
    # Statistics
    print("\n" + "="*80)
    print("RATE LIMITING STATISTICS")
    print("="*80)
    stats = agent.get_statistics()
    for operation, stat in stats.items():
        print(f"\n{operation}:")
        print(f"  Total Requests: {stat['total_requests']}")
        print(f"  Allowed: {stat['allowed']}")
        print(f"  Rejected: {stat['rejected']}")
        print(f"  Rejection Rate: {stat['rejection_rate']:.1%}")
