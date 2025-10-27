"""
Agentic Design Pattern: Result Memoization

This pattern implements intelligent result caching (memoization) where expensive
computation results are stored and reused. The agent manages multi-level caches,
TTL (time-to-live), cache invalidation strategies, and smart eviction policies.

Category: Performance Optimization
Use Cases:
- Repeated queries with same parameters
- Expensive computational results
- API response caching
- Database query results
- Complex calculations
- Resource-intensive operations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import time
from collections import OrderedDict


class CacheLevel(Enum):
    """Cache hierarchy levels"""
    L1 = 1  # Fast, small, in-memory
    L2 = 2  # Medium speed, larger
    L3 = 3  # Slower, very large


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based


class InvalidationStrategy(Enum):
    """Cache invalidation strategies"""
    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    DEPENDENCY_BASED = "dependency_based"
    MANUAL = "manual"


@dataclass
class CacheEntry:
    """Represents a cached value"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl_seconds is None:
            return False
        
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def update_access(self) -> None:
        """Update access tracking"""
        self.accessed_at = datetime.now()
        self.access_count += 1


@dataclass
class CacheStatistics:
    """Statistics for cache performance"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheTier:
    """Single tier of cache with specific policy"""
    
    def __init__(self, 
                 level: CacheLevel,
                 max_size: int = 1000,
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
                 default_ttl: Optional[int] = None):
        self.level = level
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl
        
        if eviction_policy == EvictionPolicy.LRU:
            self.storage = OrderedDict()
        else:
            self.storage: Dict[str, CacheEntry] = {}
        
        self.stats = CacheStatistics()
    
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from cache"""
        if key not in self.storage:
            self.stats.misses += 1
            return None
        
        entry = self.storage[key]
        
        # Check expiration
        if entry.is_expired():
            self.invalidate(key)
            self.stats.misses += 1
            return None
        
        # Update access
        entry.update_access()
        
        # Move to end for LRU
        if self.eviction_policy == EvictionPolicy.LRU:
            self.storage.move_to_end(key)
        
        self.stats.hits += 1
        return entry
    
    def put(self, key: str, entry: CacheEntry) -> None:
        """Put value in cache"""
        
        # Apply default TTL if not set
        if entry.ttl_seconds is None and self.default_ttl is not None:
            entry.ttl_seconds = self.default_ttl
        
        # Check if eviction needed
        if len(self.storage) >= self.max_size and key not in self.storage:
            self._evict_one()
        
        self.storage[key] = entry
        self.stats.total_entries = len(self.storage)
        self.stats.total_size_bytes += entry.size_bytes
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry"""
        if key in self.storage:
            entry = self.storage.pop(key)
            self.stats.invalidations += 1
            self.stats.total_entries = len(self.storage)
            self.stats.total_size_bytes -= entry.size_bytes
            return True
        return False
    
    def _evict_one(self) -> None:
        """Evict one entry based on policy"""
        if not self.storage:
            return
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove first (oldest)
            key, entry = self.storage.popitem(last=False)
        
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            key = min(self.storage.keys(), 
                     key=lambda k: self.storage[k].access_count)
            entry = self.storage.pop(key)
        
        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Remove oldest by creation time
            key = min(self.storage.keys(),
                     key=lambda k: self.storage[k].created_at)
            entry = self.storage.pop(key)
        
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove expired entries, then oldest
            expired = [k for k, e in self.storage.items() if e.is_expired()]
            if expired:
                key = expired[0]
            else:
                key = min(self.storage.keys(),
                         key=lambda k: self.storage[k].created_at)
            entry = self.storage.pop(key)
        
        self.stats.evictions += 1
        self.stats.total_entries = len(self.storage)
        self.stats.total_size_bytes -= entry.size_bytes
    
    def clear(self) -> None:
        """Clear all entries"""
        self.storage.clear()
        self.stats.total_entries = 0
        self.stats.total_size_bytes = 0


class MultiLevelCache:
    """Multi-level cache hierarchy"""
    
    def __init__(self):
        self.tiers: Dict[CacheLevel, CacheTier] = {}
        self.promotion_threshold = 3  # Accesses before promoting
    
    def add_tier(self, tier: CacheTier) -> None:
        """Add a cache tier"""
        self.tiers[tier.level] = tier
    
    def get(self, key: str) -> Optional[Any]:
        """Get from multi-level cache"""
        
        # Try each level from L1 to L3
        for level in [CacheLevel.L1, CacheLevel.L2, CacheLevel.L3]:
            if level not in self.tiers:
                continue
            
            tier = self.tiers[level]
            entry = tier.get(key)
            
            if entry:
                # Promote to higher tier if accessed frequently
                if entry.access_count >= self.promotion_threshold:
                    self._promote(key, entry, level)
                
                return entry.value
        
        return None
    
    def put(self, key: str, value: Any, 
           ttl_seconds: Optional[int] = None,
           tags: Optional[List[str]] = None,
           level: CacheLevel = CacheLevel.L1) -> None:
        """Put in specific cache level"""
        
        if level not in self.tiers:
            return
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            ttl_seconds=ttl_seconds,
            size_bytes=len(str(value)),  # Simplified size calculation
            tags=tags or []
        )
        
        self.tiers[level].put(key, entry)
    
    def _promote(self, key: str, entry: CacheEntry, from_level: CacheLevel) -> None:
        """Promote entry to higher cache level"""
        if from_level == CacheLevel.L2 and CacheLevel.L1 in self.tiers:
            self.tiers[CacheLevel.L1].put(key, entry)
        elif from_level == CacheLevel.L3 and CacheLevel.L2 in self.tiers:
            self.tiers[CacheLevel.L2].put(key, entry)
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with specific tag"""
        invalidated = 0
        
        for tier in self.tiers.values():
            keys_to_invalidate = [
                k for k, e in tier.storage.items()
                if tag in e.tags
            ]
            
            for key in keys_to_invalidate:
                if tier.invalidate(key):
                    invalidated += 1
        
        return invalidated
    
    def get_statistics(self) -> Dict[CacheLevel, CacheStatistics]:
        """Get statistics for all tiers"""
        return {level: tier.stats for level, tier in self.tiers.items()}


class DependencyTracker:
    """Tracks dependencies between cached values"""
    
    def __init__(self):
        self.dependencies: Dict[str, List[str]] = {}  # key -> dependent keys
        self.dependents: Dict[str, List[str]] = {}    # key -> keys it depends on
    
    def add_dependency(self, key: str, depends_on: str) -> None:
        """Add a dependency relationship"""
        if depends_on not in self.dependencies:
            self.dependencies[depends_on] = []
        self.dependencies[depends_on].append(key)
        
        if key not in self.dependents:
            self.dependents[key] = []
        self.dependents[key].append(depends_on)
    
    def get_affected_keys(self, key: str) -> List[str]:
        """Get all keys that depend on this key"""
        affected = []
        
        def collect_dependents(k: str):
            if k in self.dependencies:
                for dependent in self.dependencies[k]:
                    if dependent not in affected:
                        affected.append(dependent)
                        collect_dependents(dependent)
        
        collect_dependents(key)
        return affected


class MemoizationDecorator:
    """Decorator for automatic function memoization"""
    
    def __init__(self, cache: MultiLevelCache, ttl_seconds: Optional[int] = None):
        self.cache = cache
        self.ttl_seconds = ttl_seconds
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator implementation"""
        
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = self._create_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = self.cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            self.cache.put(key, result, self.ttl_seconds)
            
            return result
        
        return wrapper
    
    def _create_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create cache key from function signature"""
        key_data = f"{func_name}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()


class ResultMemoizationAgent:
    """
    Main agent for result memoization pattern
    
    Responsibilities:
    - Manage multi-level cache hierarchy
    - Handle cache hits and misses
    - Implement eviction policies
    - Track dependencies
    - Invalidate stale entries
    - Optimize cache performance
    """
    
    def __init__(self):
        self.cache = MultiLevelCache()
        self.dependency_tracker = DependencyTracker()
        self.memoization_stats = {
            "computations_saved": 0,
            "time_saved": 0.0
        }
        
        # Setup default tiers
        self._setup_default_tiers()
    
    def _setup_default_tiers(self) -> None:
        """Setup default cache tiers"""
        # L1: Fast, small, LRU
        l1 = CacheTier(
            level=CacheLevel.L1,
            max_size=100,
            eviction_policy=EvictionPolicy.LRU,
            default_ttl=300  # 5 minutes
        )
        self.cache.add_tier(l1)
        print("✓ L1 Cache: 100 entries, LRU, 5min TTL")
        
        # L2: Medium, LFU
        l2 = CacheTier(
            level=CacheLevel.L2,
            max_size=500,
            eviction_policy=EvictionPolicy.LFU,
            default_ttl=1800  # 30 minutes
        )
        self.cache.add_tier(l2)
        print("✓ L2 Cache: 500 entries, LFU, 30min TTL")
        
        # L3: Large, FIFO
        l3 = CacheTier(
            level=CacheLevel.L3,
            max_size=2000,
            eviction_policy=EvictionPolicy.FIFO,
            default_ttl=3600  # 1 hour
        )
        self.cache.add_tier(l3)
        print("✓ L3 Cache: 2000 entries, FIFO, 1hr TTL")
    
    def memoize(self, 
               key: str,
               computation: Callable[[], Any],
               ttl_seconds: Optional[int] = None,
               tags: Optional[List[str]] = None,
               dependencies: Optional[List[str]] = None) -> Tuple[Any, bool]:
        """Memoize a computation result"""
        
        # Check cache
        cached_result = self.cache.get(key)
        if cached_result is not None:
            self.memoization_stats["computations_saved"] += 1
            return cached_result, True  # Cache hit
        
        # Compute result
        start_time = time.time()
        result = computation()
        computation_time = time.time() - start_time
        
        # Store in cache
        self.cache.put(key, result, ttl_seconds, tags)
        
        # Track dependencies
        if dependencies:
            for dep in dependencies:
                self.dependency_tracker.add_dependency(key, dep)
        
        return result, False  # Cache miss
    
    def invalidate_by_key(self, key: str, cascade: bool = True) -> int:
        """Invalidate cache entry and optionally cascade to dependents"""
        invalidated = 0
        
        # Invalidate in all tiers
        for tier in self.cache.tiers.values():
            if tier.invalidate(key):
                invalidated += 1
        
        # Cascade to dependents
        if cascade:
            affected = self.dependency_tracker.get_affected_keys(key)
            for affected_key in affected:
                invalidated += self.invalidate_by_key(affected_key, cascade=False)
        
        return invalidated
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with specific tag"""
        return self.cache.invalidate_by_tag(tag)
    
    def create_decorator(self, ttl_seconds: Optional[int] = None) -> MemoizationDecorator:
        """Create a memoization decorator"""
        return MemoizationDecorator(self.cache, ttl_seconds)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        tier_stats = self.cache.get_statistics()
        
        total_hits = sum(s.hits for s in tier_stats.values())
        total_misses = sum(s.misses for s in tier_stats.values())
        total_requests = total_hits + total_misses
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            "overall_hit_rate": round(overall_hit_rate, 3),
            "total_requests": total_requests,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "computations_saved": self.memoization_stats["computations_saved"],
            "time_saved_seconds": round(self.memoization_stats["time_saved"], 3),
            "l1_hit_rate": round(tier_stats[CacheLevel.L1].hit_rate(), 3),
            "l2_hit_rate": round(tier_stats[CacheLevel.L2].hit_rate(), 3),
            "l3_hit_rate": round(tier_stats[CacheLevel.L3].hit_rate(), 3),
            "total_evictions": sum(s.evictions for s in tier_stats.values()),
            "total_invalidations": sum(s.invalidations for s in tier_stats.values())
        }


def demonstrate_result_memoization():
    """Demonstrate the result memoization pattern"""
    
    print("=" * 60)
    print("Result Memoization Pattern Demonstration")
    print("=" * 60)
    
    # Create agent
    print("\n1. Initializing Multi-Level Cache")
    print("-" * 60)
    agent = ResultMemoizationAgent()
    
    # Scenario 1: Basic memoization
    print("\n2. Basic Memoization")
    print("-" * 60)
    
    def expensive_computation():
        time.sleep(0.1)
        return "Expensive result"
    
    print("First call (cache miss):")
    start = time.time()
    result1, cached1 = agent.memoize("key1", expensive_computation)
    time1 = time.time() - start
    print(f"  Result: {result1}")
    print(f"  Cached: {cached1}")
    print(f"  Time: {time1:.3f}s")
    
    print("\nSecond call (cache hit):")
    start = time.time()
    result2, cached2 = agent.memoize("key1", expensive_computation)
    time2 = time.time() - start
    print(f"  Result: {result2}")
    print(f"  Cached: {cached2}")
    print(f"  Time: {time2:.3f}s")
    print(f"  Speedup: {time1/time2:.1f}x")
    
    # Scenario 2: Tag-based invalidation
    print("\n3. Tag-Based Invalidation")
    print("-" * 60)
    
    for i in range(5):
        agent.memoize(
            f"user_data_{i}",
            lambda: f"User {i} data",
            tags=["user_data"]
        )
    
    print("Cached 5 user data entries with tag 'user_data'")
    
    invalidated = agent.invalidate_by_tag("user_data")
    print(f"Invalidated {invalidated} entries by tag")
    
    # Scenario 3: Dependency-based invalidation
    print("\n4. Dependency-Based Invalidation")
    print("-" * 60)
    
    # Create dependent computations
    agent.memoize("base_data", lambda: "Base", dependencies=[])
    agent.memoize("derived_1", lambda: "Derived1", dependencies=["base_data"])
    agent.memoize("derived_2", lambda: "Derived2", dependencies=["base_data"])
    agent.memoize("derived_2_1", lambda: "Derived2.1", dependencies=["derived_2"])
    
    print("Created dependency chain:")
    print("  base_data")
    print("  ├── derived_1")
    print("  └── derived_2")
    print("      └── derived_2_1")
    
    print("\nInvalidating base_data (with cascade):")
    invalidated = agent.invalidate_by_key("base_data", cascade=True)
    print(f"Invalidated {invalidated} entries (base + dependents)")
    
    # Scenario 4: Decorator usage
    print("\n5. Decorator Usage")
    print("-" * 60)
    
    memoize_decorator = agent.create_decorator(ttl_seconds=60)
    
    @memoize_decorator
    def fibonacci(n: int) -> int:
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)
    
    print("Calculating fibonacci(10) with memoization:")
    start = time.time()
    result = fibonacci(10)
    time_memo = time.time() - start
    print(f"  Result: {result}")
    print(f"  Time: {time_memo:.4f}s")
    
    # Scenario 5: Performance comparison
    print("\n6. Performance Comparison")
    print("-" * 60)
    
    def slow_function(x: int) -> int:
        time.sleep(0.01)
        return x * x
    
    # Without memoization
    print("Without memoization (10 repeated calls):")
    start = time.time()
    for _ in range(10):
        slow_function(5)
    time_no_memo = time.time() - start
    print(f"  Time: {time_no_memo:.3f}s")
    
    # With memoization
    print("\nWith memoization (10 repeated calls):")
    start = time.time()
    for i in range(10):
        agent.memoize(f"perf_test_{i%3}", lambda: slow_function(5))
    time_with_memo = time.time() - start
    print(f"  Time: {time_with_memo:.3f}s")
    print(f"  Speedup: {time_no_memo/time_with_memo:.1f}x")
    
    # Statistics
    print("\n7. Cache Statistics")
    print("-" * 60)
    
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_result_memoization()
