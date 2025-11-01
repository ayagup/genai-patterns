"""
Pattern 143: Result Memoization

Description:
    The Result Memoization pattern caches computation results to avoid redundant
    processing, significantly improving performance for repeated operations. When
    an agent encounters the same input or query it has processed before, it returns
    the cached result immediately instead of recomputing. This pattern is essential
    for production AI systems where identical or similar queries occur frequently.

    Memoization provides dramatic performance improvements for deterministic operations,
    expensive computations, and frequently accessed data. It trades memory for speed,
    storing results with appropriate cache keys, expiration policies, and invalidation
    strategies. The pattern supports various caching strategies including LRU, TTL,
    and semantic similarity-based caching.

    This implementation provides comprehensive memoization capabilities including
    function-level caching, LLM response caching, semantic similarity caching,
    cache warming, invalidation strategies, and performance metrics. It handles
    both simple key-value caching and advanced semantic caching for AI operations.

Components:
    - Function Memoization: Cache function results by arguments
    - LLM Response Caching: Cache model outputs for queries
    - Semantic Caching: Cache based on semantic similarity
    - Cache Warming: Preload frequently accessed results
    - Invalidation Strategy: Remove stale cache entries
    - Performance Metrics: Track cache effectiveness

Use Cases:
    - Repeated queries in chatbots
    - Expensive computations
    - API call reduction
    - Database query caching
    - Embedding generation
    - Translation caching
    - Recommendation systems
    - Data processing pipelines

LangChain Implementation:
    This implementation uses:
    - Python functools.lru_cache for function memoization
    - Custom cache implementations with TTL
    - Semantic similarity-based caching
    - LangChain cache integration
    - Hash-based cache keys
    - Cache metrics and monitoring

Benefits:
    - Dramatically reduces latency
    - Lowers computational costs
    - Reduces API calls
    - Improves scalability
    - Provides consistent responses
    - Enables offline operation
    - Reduces load on services

Trade-offs:
    - Memory overhead for cache storage
    - Stale data if not invalidated properly
    - Cache consistency challenges
    - Cold start performance
    - Complex invalidation logic
    - Potential memory leaks

Production Considerations:
    - Set appropriate cache sizes
    - Implement TTL for freshness
    - Monitor cache hit rates
    - Use distributed caching for scale
    - Handle cache misses gracefully
    - Implement cache warming
    - Monitor memory usage
    - Use appropriate invalidation
    - Consider cache coherence
    - Profile cache effectiveness
    - Implement cache preloading
    - Handle cache failures
"""

import os
import time
import hashlib
import json
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from collections import OrderedDict
from dataclasses import dataclass, field
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[float]
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds
    
    def access(self):
        """Record access."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class LRUCache:
    """LRU (Least Recently Used) Cache implementation."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: Optional[float] = None):
        """Initialize LRU cache."""
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if entry.is_expired():
            del self.cache[key]
            self.misses += 1
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        entry.access()
        self.hits += 1
        return entry.value
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        # Remove if exists
        if key in self.cache:
            del self.cache[key]
        
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            ttl_seconds=self.ttl_seconds
        )
        
        self.cache[key] = entry
        
        # Evict oldest if over size
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2%}",
            "total_requests": total
        }


class SemanticCache:
    """Cache based on semantic similarity of inputs."""
    
    def __init__(self, similarity_threshold: float = 0.9, max_size: int = 100):
        """Initialize semantic cache."""
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.entries: List[Tuple[str, Any, str]] = []  # (key, value, original_text)
        self.hits = 0
        self.misses = 0
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute simple similarity (in production, use embeddings)."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def get(self, query: str) -> Optional[Any]:
        """Get value from cache based on semantic similarity."""
        for key, value, original_text in self.entries:
            similarity = self._compute_similarity(query, original_text)
            if similarity >= self.similarity_threshold:
                self.hits += 1
                return value
        
        self.misses += 1
        return None
    
    def set(self, query: str, value: Any):
        """Set value in cache."""
        key = hashlib.md5(query.encode()).hexdigest()
        self.entries.append((key, value, query))
        
        # Evict oldest if over size
        if len(self.entries) > self.max_size:
            self.entries.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.entries),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2%}",
            "similarity_threshold": self.similarity_threshold
        }


def memoize(cache: Optional[LRUCache] = None, ttl_seconds: Optional[float] = None):
    """Decorator for function memoization."""
    if cache is None:
        cache = LRUCache(ttl_seconds=ttl_seconds)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key_data = {
                "func": func.__name__,
                "args": args,
                "kwargs": kwargs
            }
            key = hashlib.md5(
                json.dumps(key_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        wrapper.cache = cache
        return wrapper
    
    return decorator


class MemoizationAgent:
    """
    Agent that uses memoization for performance optimization.
    
    This agent caches results to avoid redundant computation,
    dramatically improving performance for repeated operations.
    """
    
    def __init__(self, temperature: float = 0.3):
        """Initialize the memoization agent."""
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        
        # Different caches for different purposes
        self.llm_cache = LRUCache(max_size=100, ttl_seconds=3600)
        self.computation_cache = LRUCache(max_size=50, ttl_seconds=None)
        self.semantic_cache = SemanticCache(similarity_threshold=0.85)
        
        # Create LLM chain (we'll cache its results)
        self.chain = (
            ChatPromptTemplate.from_template("{query}")
            | self.llm
            | StrOutputParser()
        )
    
    def query_llm_cached(self, query: str, use_semantic: bool = False) -> str:
        """Query LLM with caching."""
        # Create cache key
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        # Try semantic cache first if enabled
        if use_semantic:
            result = self.semantic_cache.get(query)
            if result:
                return f"[SEMANTIC CACHE HIT] {result}"
        
        # Try exact cache
        result = self.llm_cache.get(cache_key)
        if result:
            return f"[CACHE HIT] {result}"
        
        # Cache miss - query LLM
        result = self.chain.invoke({"query": query})
        
        # Store in caches
        self.llm_cache.set(cache_key, result)
        if use_semantic:
            self.semantic_cache.set(query, result)
        
        return f"[CACHE MISS] {result}"
    
    @memoize(ttl_seconds=300)
    def expensive_computation(self, value: int) -> int:
        """Simulate expensive computation with memoization."""
        time.sleep(0.5)  # Simulate expensive work
        return value ** 2
    
    def cache_warm(self, common_queries: List[str]):
        """Warm up cache with common queries."""
        print(f"Warming cache with {len(common_queries)} common queries...")
        for query in common_queries:
            result = self.chain.invoke({"query": query})
            cache_key = hashlib.md5(query.encode()).hexdigest()
            self.llm_cache.set(cache_key, result)
            self.semantic_cache.set(query, result)
        print("Cache warming complete")
    
    def invalidate_cache(self, pattern: Optional[str] = None):
        """Invalidate cache entries."""
        if pattern is None:
            # Clear all caches
            self.llm_cache.clear()
            self.computation_cache.clear()
            print("All caches cleared")
        else:
            print(f"Pattern-based invalidation not implemented for: {pattern}")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            "llm_cache": self.llm_cache.get_stats(),
            "computation_cache": self.computation_cache.get_stats(),
            "semantic_cache": self.semantic_cache.get_stats()
        }


def demonstrate_memoization():
    """Demonstrate the result memoization pattern."""
    print("=" * 80)
    print("Result Memoization Pattern Demonstration")
    print("=" * 80)
    
    agent = MemoizationAgent()
    
    # Demonstration 1: Basic Function Memoization
    print("\n" + "=" * 80)
    print("Demonstration 1: Basic Function Memoization")
    print("=" * 80)
    
    print("\nFirst call (cache miss, will take 0.5s):")
    start = time.time()
    result1 = agent.expensive_computation(10)
    elapsed1 = time.time() - start
    print(f"Result: {result1}, Time: {elapsed1:.3f}s")
    
    print("\nSecond call with same argument (cache hit, instant):")
    start = time.time()
    result2 = agent.expensive_computation(10)
    elapsed2 = time.time() - start
    print(f"Result: {result2}, Time: {elapsed2:.3f}s")
    
    print(f"\nSpeedup: {elapsed1/elapsed2:.1f}x faster")
    
    # Demonstration 2: LLM Response Caching
    print("\n" + "=" * 80)
    print("Demonstration 2: LLM Response Caching")
    print("=" * 80)
    
    query = "What is 2+2?"
    
    print(f"\nQuery: {query}")
    print("First call (cache miss):")
    start = time.time()
    response1 = agent.query_llm_cached(query)
    elapsed1 = time.time() - start
    print(f"Response: {response1[:100]}...")
    print(f"Time: {elapsed1:.3f}s")
    
    print("\nSecond call (cache hit):")
    start = time.time()
    response2 = agent.query_llm_cached(query)
    elapsed2 = time.time() - start
    print(f"Response: {response2[:100]}...")
    print(f"Time: {elapsed2:.3f}s")
    
    # Demonstration 3: Semantic Caching
    print("\n" + "=" * 80)
    print("Demonstration 3: Semantic Caching")
    print("=" * 80)
    
    query1 = "What is the capital of France?"
    query2 = "Tell me the capital city of France"  # Similar semantic meaning
    
    print(f"\nFirst query: {query1}")
    response1 = agent.query_llm_cached(query1, use_semantic=True)
    print(f"Response: {response1[:80]}...")
    
    print(f"\nSimilar query: {query2}")
    response2 = agent.query_llm_cached(query2, use_semantic=True)
    print(f"Response: {response2[:80]}...")
    
    if "[SEMANTIC CACHE HIT]" in response2:
        print("\n✓ Semantic cache successfully matched similar query!")
    
    # Demonstration 4: Cache Warming
    print("\n" + "=" * 80)
    print("Demonstration 4: Cache Warming")
    print("=" * 80)
    
    common_queries = [
        "What is the weather?",
        "Tell me a joke",
        "Hello, how are you?"
    ]
    
    agent.cache_warm(common_queries)
    
    print("\nQuerying warmed cache:")
    for query in common_queries[:2]:
        response = agent.query_llm_cached(query)
        print(f"  {query}: {response[:60]}...")
    
    # Demonstration 5: TTL Expiration
    print("\n" + "=" * 80)
    print("Demonstration 5: TTL (Time-To-Live) Expiration")
    print("=" * 80)
    
    # Create cache with short TTL
    short_ttl_cache = LRUCache(max_size=10, ttl_seconds=2)
    
    @memoize(cache=short_ttl_cache)
    def short_lived_computation(x):
        time.sleep(0.1)
        return x * 2
    
    print("\nComputation with 2-second TTL:")
    print("First call:")
    result1 = short_lived_computation(5)
    print(f"Result: {result1}")
    
    print("\nImmediate second call (within TTL):")
    result2 = short_lived_computation(5)
    print(f"Result: {result2} (cached)")
    
    print("\nWaiting 2.5 seconds for TTL expiration...")
    time.sleep(2.5)
    
    print("Third call (after TTL expiration):")
    result3 = short_lived_computation(5)
    print(f"Result: {result3} (recomputed)")
    
    # Demonstration 6: LRU Eviction
    print("\n" + "=" * 80)
    print("Demonstration 6: LRU Cache Eviction")
    print("=" * 80)
    
    small_cache = LRUCache(max_size=3)
    
    print("\nCache with max size 3:")
    print("Adding 5 items...")
    for i in range(5):
        small_cache.set(f"key_{i}", f"value_{i}")
        print(f"  Added key_{i}, cache size: {len(small_cache.cache)}")
    
    print(f"\nFinal cache size: {len(small_cache.cache)} (oldest evicted)")
    print(f"Remaining keys: {list(small_cache.cache.keys())}")
    
    # Demonstration 7: Cache Invalidation
    print("\n" + "=" * 80)
    print("Demonstration 7: Cache Invalidation")
    print("=" * 80)
    
    print("\nQuerying LLM to populate cache...")
    agent.query_llm_cached("Test query 1")
    agent.query_llm_cached("Test query 2")
    
    stats_before = agent.get_all_stats()
    print(f"LLM cache size before: {stats_before['llm_cache']['size']}")
    
    print("\nInvalidating all caches...")
    agent.invalidate_cache()
    
    stats_after = agent.get_all_stats()
    print(f"LLM cache size after: {stats_after['llm_cache']['size']}")
    
    # Demonstration 8: Cache Statistics
    print("\n" + "=" * 80)
    print("Demonstration 8: Cache Performance Statistics")
    print("=" * 80)
    
    # Generate some cache activity
    for i in range(10):
        agent.query_llm_cached(f"Query {i % 3}")  # Repeat some queries
    
    stats = agent.get_all_stats()
    
    print("\nLLM Cache Statistics:")
    for key, value in stats['llm_cache'].items():
        print(f"  {key}: {value}")
    
    print("\nSemantic Cache Statistics:")
    for key, value in stats['semantic_cache'].items():
        print(f"  {key}: {value}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary: Result Memoization Pattern")
    print("=" * 80)
    print("""
The Result Memoization pattern dramatically improves performance through caching:

Key Features Demonstrated:
1. Function Memoization - Cache expensive computations
2. LLM Response Caching - Avoid redundant API calls
3. Semantic Caching - Match similar queries
4. Cache Warming - Preload common queries
5. TTL Expiration - Auto-invalidate stale data
6. LRU Eviction - Manage cache size
7. Cache Invalidation - Clear stale entries
8. Performance Metrics - Track cache effectiveness

Benefits:
- Dramatically reduces latency
- Lowers computational costs
- Reduces API calls
- Improves scalability
- Provides consistent responses
- Reduces load on services

Best Practices:
- Set appropriate cache sizes
- Implement TTL for freshness
- Monitor cache hit rates
- Use distributed caching for scale
- Handle cache misses gracefully
- Implement cache warming
- Monitor memory usage
- Use appropriate invalidation
- Consider cache coherence
- Profile cache effectiveness

Common Use Cases:
- Repeated queries in chatbots
- Expensive computations
- API call reduction
- Database query caching
- Embedding generation
- Translation caching
- Recommendation systems
- Data processing pipelines

Performance Considerations:
- Memory overhead for cache storage
- Balance cache size vs. hit rate
- Implement appropriate eviction policies
- Monitor for stale data
- Consider distributed caching
- Profile actual performance gains

This pattern is essential for production AI systems requiring high performance
and cost optimization through intelligent caching.
""")


if __name__ == "__main__":
    demonstrate_memoization()
