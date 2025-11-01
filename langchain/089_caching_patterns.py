"""
Pattern 089: Caching Patterns

Description:
    The Caching Patterns involve storing and reusing previously computed results to
    improve performance, reduce latency, and minimize costs. In the context of LLM
    applications, caching is critical for optimizing response times and reducing API
    calls for repeated or similar queries. Effective caching strategies can
    dramatically improve user experience while significantly cutting operational costs.

    Caching in LLM applications targets:
    - Complete responses (exact match caching)
    - Embeddings (vector representations)
    - Prompt templates (reusable structures)
    - Intermediate results (multi-step workflows)
    - API responses (rate limit optimization)

Components:
    1. Cache Types
       - In-memory cache (fast, volatile)
       - Disk cache (persistent, slower)
       - Distributed cache (Redis, Memcached)
       - Database cache (structured, queryable)
       - CDN cache (geographic distribution)

    2. Cache Keys
       - Exact match (full prompt hash)
       - Semantic match (embedding similarity)
       - Parametric keys (template + parameters)
       - Hierarchical keys (nested cache structures)
       - Time-based keys (temporal validity)

    3. Cache Policies
       - TTL (Time To Live) - expire after time
       - LRU (Least Recently Used) - evict old entries
       - LFU (Least Frequently Used) - evict unpopular
       - Size-based - limit total cache size
       - Custom policies - application-specific

    4. Invalidation Strategies
       - Time-based expiration
       - Event-triggered invalidation
       - Manual invalidation
       - Cascade invalidation (dependencies)
       - Version-based invalidation

Use Cases:
    1. Repeated Queries
       - FAQ responses
       - Common questions
       - Popular content
       - Standard procedures
       - Frequently accessed data

    2. Expensive Operations
       - Complex reasoning chains
       - Multi-step workflows
       - Large context processing
       - Batch operations
       - Resource-intensive computations

    3. Rate Limit Optimization
       - Reduce API calls
       - Stay within quotas
       - Handle traffic spikes
       - Cost optimization
       - Performance consistency

    4. Embeddings and Vectors
       - Document embeddings
       - Query embeddings
       - Semantic search optimization
       - Similarity computations
       - Vector database queries

    5. Multi-User Applications
       - Shared responses
       - Common workflows
       - Popular queries
       - Template responses
       - Collaborative features

LangChain Implementation:
    LangChain supports caching through:
    - InMemoryCache for simple caching
    - SQLiteCache for persistent caching
    - Custom cache implementations
    - Semantic similarity caching
    - Embedding caching

Key Features:
    1. Multiple Cache Layers
       - L1: In-memory (fastest)
       - L2: Local disk (persistent)
       - L3: Distributed cache (shared)
       - L4: Database (long-term)

    2. Intelligent Matching
       - Exact match for identical queries
       - Semantic match for similar queries
       - Fuzzy match with thresholds
       - Parametric templates

    3. Performance Optimization
       - Fast lookup (< 1ms for in-memory)
       - Efficient serialization
       - Compression for large entries
       - Lazy loading

    4. Observability
       - Hit rate monitoring
       - Miss analysis
       - Performance metrics
       - Cost savings tracking

Best Practices:
    1. Cache Appropriately
       - Cache stable, reusable results
       - Don't cache rapidly changing data
       - Consider data sensitivity
       - Balance freshness vs performance

    2. Set Proper TTLs
       - Shorter for dynamic content
       - Longer for stable content
       - Consider update frequency
       - Account for user expectations

    3. Monitor Performance
       - Track hit rates (target > 70%)
       - Measure latency improvements
       - Calculate cost savings
       - Identify cache misses patterns

    4. Handle Cache Misses
       - Graceful fallback to computation
       - Transparent to users
       - Background cache warming
       - Predictive pre-caching

Trade-offs:
    Advantages:
    - Dramatically improved response times
    - Significant cost reduction
    - Better user experience
    - Reduced API rate limiting
    - Lower resource consumption
    - Improved scalability

    Disadvantages:
    - Stale data risk
    - Memory/storage overhead
    - Cache consistency challenges
    - Complexity in invalidation
    - Initial cache warming time
    - Debugging complexity

Production Considerations:
    1. Cache Size Management
       - Monitor memory usage
       - Implement eviction policies
       - Set size limits
       - Regular cleanup
       - Compression strategies

    2. Cache Consistency
       - Version tracking
       - Update propagation
       - Invalidation timing
       - Distributed consistency
       - Race condition handling

    3. Security
       - Access control for cached data
       - Encryption for sensitive content
       - Cache poisoning prevention
       - User isolation in multi-tenant
       - Audit trails

    4. Monitoring
       - Hit/miss rates
       - Cache size trends
       - Eviction frequency
       - Cost savings
       - Performance impact

    5. Disaster Recovery
       - Cache rebuild procedures
       - Backup strategies
       - Failover mechanisms
       - Data loss handling
       - Recovery time objectives
"""

import os
import time
import hashlib
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import OrderedDict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.caches import InMemoryCache
from langchain.globals import set_llm_cache

load_dotenv()


class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


@dataclass
class CacheEntry:
    """Represents a cached entry"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[int] = None  # TTL in seconds
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl
    
    def touch(self):
        """Update access information"""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return 100.0 - self.hit_rate


class SimpleCache:
    """
    Simple in-memory cache with multiple eviction policies.
    
    This cache supports:
    1. Multiple eviction policies (LRU, LFU, FIFO, TTL)
    2. Size limits
    3. TTL per entry
    4. Performance statistics
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        policy: CachePolicy = CachePolicy.LRU,
        default_ttl: Optional[int] = None
    ):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries
            policy: Eviction policy
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.policy = policy
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
    
    def _make_key(self, key: Any) -> str:
        """Create cache key from input"""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        return hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()
    
    def _evict(self):
        """Evict entries based on policy"""
        if len(self.cache) < self.max_size:
            return
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used
            self.cache.popitem(last=False)
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            lfu_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            del self.cache[lfu_key]
        elif self.policy == CachePolicy.FIFO:
            # Remove oldest
            self.cache.popitem(last=False)
        
        self.stats.evictions += 1
    
    def _remove_expired(self):
        """Remove expired entries"""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        self.stats.total_requests += 1
        cache_key = self._make_key(key)
        
        # Remove expired entries
        self._remove_expired()
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            
            if entry.is_expired():
                del self.cache[cache_key]
                self.stats.misses += 1
                return None
            
            entry.touch()
            
            # Move to end for LRU
            if self.policy == CachePolicy.LRU:
                self.cache.move_to_end(cache_key)
            
            self.stats.hits += 1
            return entry.value
        
        self.stats.misses += 1
        return None
    
    def set(self, key: Any, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
        """
        cache_key = self._make_key(key)
        
        # Evict if necessary
        if cache_key not in self.cache:
            self._evict()
        
        entry = CacheEntry(
            key=cache_key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl=ttl or self.default_ttl
        )
        
        self.cache[cache_key] = entry
        
        # Move to end for LRU
        if self.policy == CachePolicy.LRU:
            self.cache.move_to_end(cache_key)
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.stats = CacheStats()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
            "total_requests": self.stats.total_requests,
            "hit_rate": f"{self.stats.hit_rate:.2f}%",
            "miss_rate": f"{self.stats.miss_rate:.2f}%",
            "current_size": len(self.cache),
            "max_size": self.max_size
        }


class SemanticCache:
    """
    Semantic cache using embeddings for similarity matching.
    
    This cache finds similar queries using vector similarity,
    allowing cache hits for semantically similar but not identical queries.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        max_size: int = 500
    ):
        """
        Initialize semantic cache.
        
        Args:
            similarity_threshold: Minimum similarity for cache hit
            max_size: Maximum cache entries
        """
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.embeddings = OpenAIEmbeddings()
        self.cache: List[Dict[str, Any]] = []
        self.stats = CacheStats()
    
    def _compute_similarity(self, query1: str, query2: str) -> float:
        """
        Compute semantic similarity between queries.
        
        Args:
            query1: First query
            query2: Second query
            
        Returns:
            Similarity score (0-1)
        """
        # Simple implementation - in production use precomputed embeddings
        # This is for demonstration only
        query1_lower = query1.lower()
        query2_lower = query2.lower()
        
        # Simple word overlap similarity
        words1 = set(query1_lower.split())
        words2 = set(query2_lower.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def get(self, query: str) -> Optional[Any]:
        """
        Get value for semantically similar query.
        
        Args:
            query: Query string
            
        Returns:
            Cached value or None
        """
        self.stats.total_requests += 1
        
        # Find most similar cached query
        best_match = None
        best_similarity = 0.0
        
        for entry in self.cache:
            similarity = self._compute_similarity(query, entry["query"])
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
        if best_match and best_similarity >= self.similarity_threshold:
            self.stats.hits += 1
            # Update access time
            best_match["last_accessed"] = datetime.now()
            return best_match["value"]
        
        self.stats.misses += 1
        return None
    
    def set(self, query: str, value: Any):
        """
        Cache value for query.
        
        Args:
            query: Query string
            value: Value to cache
        """
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.pop(0)
            self.stats.evictions += 1
        
        self.cache.append({
            "query": query,
            "value": value,
            "created_at": datetime.now(),
            "last_accessed": datetime.now()
        })
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.stats = CacheStats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
            "total_requests": self.stats.total_requests,
            "hit_rate": f"{self.stats.hit_rate:.2f}%",
            "current_size": len(self.cache),
            "max_size": self.max_size,
            "similarity_threshold": self.similarity_threshold
        }


class CachedLLMAgent:
    """
    LLM agent with caching support.
    
    This agent automatically caches responses and retrieves them
    for repeated or similar queries.
    """
    
    def __init__(
        self,
        cache: SimpleCache,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize cached agent.
        
        Args:
            cache: Cache instance to use
            model: LLM model name
        """
        self.cache = cache
        self.llm = ChatOpenAI(model=model, temperature=0.7)
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Generate response with caching.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Dictionary with response and cache info
        """
        # Check cache first
        cached_response = self.cache.get(prompt)
        
        if cached_response is not None:
            return {
                "response": cached_response,
                "from_cache": True,
                "cache_stats": self.cache.get_stats()
            }
        
        # Generate new response
        start_time = time.time()
        response = self.llm.invoke(prompt)
        generation_time = time.time() - start_time
        
        # Cache the response
        self.cache.set(prompt, response.content)
        
        return {
            "response": response.content,
            "from_cache": False,
            "generation_time": generation_time,
            "cache_stats": self.cache.get_stats()
        }


def demonstrate_caching_patterns():
    """Demonstrate caching patterns"""
    print("=" * 80)
    print("CACHING PATTERNS DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic LRU Cache
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic LRU Cache")
    print("=" * 80)
    
    cache = SimpleCache(max_size=5, policy=CachePolicy.LRU)
    agent = CachedLLMAgent(cache)
    
    print("\nMaking requests (first time - cache misses):\n")
    
    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Python?",  # Repeat
        "What is Java?",
        "What is Python?",  # Repeat again
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"Request {i}: {query}")
        result = agent.generate(query)
        
        cache_status = "HIT ✓" if result["from_cache"] else "MISS ✗"
        print(f"  Status: {cache_status}")
        
        if not result["from_cache"]:
            print(f"  Generation time: {result['generation_time']:.2f}s")
        print(f"  Response: {result['response'][:80]}...")
        print()
    
    stats = cache.get_stats()
    print("Cache Statistics:")
    print(f"  Hit Rate: {stats['hit_rate']}")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    
    # Example 2: TTL-Based Caching
    print("\n" + "=" * 80)
    print("EXAMPLE 2: TTL-Based Caching (Time Expiration)")
    print("=" * 80)
    
    ttl_cache = SimpleCache(max_size=100, default_ttl=5)  # 5 second TTL
    
    print("\nCaching value with 5-second TTL...")
    ttl_cache.set("test_key", "test_value", ttl=5)
    
    print("Immediate retrieval:")
    value = ttl_cache.get("test_key")
    print(f"  Result: {'HIT' if value else 'MISS'} - {value}")
    
    print("\nWaiting 6 seconds for expiration...")
    time.sleep(6)
    
    print("Retrieval after TTL:")
    value = ttl_cache.get("test_key")
    print(f"  Result: {'HIT' if value else 'MISS'} - {value}")
    
    # Example 3: Cache Eviction Policies
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Cache Eviction Policies Comparison")
    print("=" * 80)
    
    policies = [CachePolicy.LRU, CachePolicy.LFU, CachePolicy.FIFO]
    
    for policy in policies:
        print(f"\n{policy.value.upper()} Policy:")
        test_cache = SimpleCache(max_size=3, policy=policy)
        
        # Add 5 items to cache of size 3
        for i in range(5):
            key = f"item_{i}"
            test_cache.set(key, f"value_{i}")
            print(f"  Added {key}, cache size: {test_cache.size()}")
        
        stats = test_cache.get_stats()
        print(f"  Evictions: {stats['evictions']}")
    
    # Example 4: Semantic Cache
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Semantic Cache (Similar Query Matching)")
    print("=" * 80)
    
    semantic_cache = SemanticCache(similarity_threshold=0.7, max_size=100)
    
    # Cache a response
    original_query = "What is machine learning?"
    original_response = "Machine learning is a subset of AI that enables systems to learn from data."
    semantic_cache.set(original_query, original_response)
    
    print(f"\nCached query: '{original_query}'")
    print(f"Response: {original_response}\n")
    
    # Try similar queries
    similar_queries = [
        "What is machine learning?",  # Exact match
        "What's machine learning?",   # Slight variation
        "Tell me about machine learning",  # Similar
        "Explain artificial intelligence",  # Different topic
    ]
    
    for query in similar_queries:
        result = semantic_cache.get(query)
        status = "HIT ✓" if result else "MISS ✗"
        print(f"Query: '{query}'")
        print(f"  Status: {status}")
        if result:
            print(f"  Retrieved: {result}")
        print()
    
    # Example 5: Multi-Layer Cache
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Multi-Layer Cache Strategy")
    print("=" * 80)
    
    # L1: Small, fast, in-memory cache
    l1_cache = SimpleCache(max_size=10, policy=CachePolicy.LRU)
    
    # L2: Larger, still in-memory
    l2_cache = SimpleCache(max_size=100, policy=CachePolicy.LRU)
    
    print("\nMulti-layer cache simulation:")
    print("  L1: Small (10 items), fast")
    print("  L2: Large (100 items), moderate\n")
    
    def multi_layer_get(key: str) -> Optional[str]:
        """Get from multi-layer cache"""
        # Try L1 first
        value = l1_cache.get(key)
        if value:
            print(f"  L1 HIT for '{key}'")
            return value
        
        # Try L2
        value = l2_cache.get(key)
        if value:
            print(f"  L2 HIT for '{key}'")
            # Promote to L1
            l1_cache.set(key, value)
            return value
        
        print(f"  MISS for '{key}'")
        return None
    
    # Simulate access pattern
    l1_cache.set("frequently_accessed", "hot_data")
    l2_cache.set("occasionally_accessed", "warm_data")
    
    print("Access patterns:")
    multi_layer_get("frequently_accessed")
    multi_layer_get("occasionally_accessed")
    multi_layer_get("never_cached")
    
    # Example 6: Cache Performance Metrics
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Cache Performance Analysis")
    print("=" * 80)
    
    perf_cache = SimpleCache(max_size=50, policy=CachePolicy.LRU)
    perf_agent = CachedLLMAgent(perf_cache)
    
    print("\nSimulating workload with repeated queries...\n")
    
    workload = [
        "What is AI?",
        "What is ML?",
        "What is AI?",  # Repeat
        "What is DL?",
        "What is AI?",  # Repeat
        "What is ML?",  # Repeat
        "What is NLP?",
        "What is AI?",  # Repeat
    ]
    
    total_time_with_cache = 0
    total_time_without_cache = 0
    
    for query in workload:
        result = perf_agent.generate(query)
        
        if result["from_cache"]:
            total_time_with_cache += 0.001  # Assume 1ms for cache hit
        else:
            gen_time = result.get("generation_time", 1.0)
            total_time_with_cache += gen_time
            total_time_without_cache += gen_time
    
    # Estimate time without cache (all queries would need generation)
    total_time_without_cache = len(workload) * 1.0  # Assume 1s per generation
    
    print(f"Performance Analysis:")
    print(f"  Total Queries: {len(workload)}")
    print(f"  Time WITH cache: {total_time_with_cache:.2f}s")
    print(f"  Time WITHOUT cache (estimated): {total_time_without_cache:.2f}s")
    print(f"  Speedup: {total_time_without_cache/total_time_with_cache:.1f}x")
    print(f"  Time Saved: {total_time_without_cache - total_time_with_cache:.2f}s")
    
    final_stats = perf_cache.get_stats()
    print(f"\nFinal Cache Statistics:")
    print(f"  Hit Rate: {final_stats['hit_rate']}")
    print(f"  Hits: {final_stats['hits']}")
    print(f"  Misses: {final_stats['misses']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("CACHING PATTERNS SUMMARY")
    print("=" * 80)
    print("""
Caching Pattern Benefits:
1. Performance: Dramatically faster response times (10-1000x)
2. Cost Reduction: Fewer API calls = lower costs
3. Scalability: Handle more requests with same resources
4. Reliability: Reduce dependency on external services
5. User Experience: Instant responses for cached queries

Cache Types Demonstrated:
1. Simple Cache: Basic in-memory caching with eviction policies
2. TTL Cache: Time-based expiration for freshness
3. Semantic Cache: Similarity-based matching for related queries
4. Multi-Layer Cache: Hierarchical caching strategy
5. Performance-Optimized: Focus on hit rates and speedup

Eviction Policies:
1. LRU (Least Recently Used): Remove oldest accessed items
2. LFU (Least Frequently Used): Remove least popular items
3. FIFO (First In First Out): Remove oldest items
4. TTL (Time To Live): Expire after time period

Best Practices:
1. Cache Strategy Selection:
   - Exact match: Simple, fast, for identical queries
   - Semantic: Better coverage, for similar queries
   - TTL: For time-sensitive data
   - Multi-layer: For diverse access patterns

2. TTL Configuration:
   - Short (minutes): Dynamic, frequently changing data
   - Medium (hours): Semi-static content
   - Long (days): Stable, rarely changing data
   - No TTL: Static reference data

3. Size Management:
   - Monitor memory usage
   - Set appropriate limits
   - Implement eviction policies
   - Regular cleanup

4. Performance Monitoring:
   - Target >70% hit rate for good performance
   - Track cache size trends
   - Monitor eviction frequency
   - Measure latency improvements

When to Use Caching:
✓ Repeated identical queries
✓ Similar semantic queries
✓ Expensive computations
✓ Rate-limited APIs
✓ Stable, unchanging data
✗ Real-time data requirements
✗ Highly personalized responses
✗ Security-sensitive information
✗ Rapidly changing data

Production Considerations:
- Distributed caching (Redis) for multi-instance deployments
- Cache warming strategies for cold starts
- Monitoring and alerting on hit rates
- Regular cache analysis and optimization
- Security: encrypt sensitive cached data
- Compliance: respect data retention policies
- Backup and recovery procedures
- Cache versioning for updates

Cost-Benefit Analysis:
- Typical API call: $0.001-0.01 per request
- Cache hit cost: ~$0.000001 per request
- Potential savings: 90-99% cost reduction
- Performance gain: 10-1000x faster responses
- ROI: Immediate for high-volume applications
""")
    
    print("\n" + "=" * 80)
    print("Pattern 089 (Caching Patterns) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_caching_patterns()
