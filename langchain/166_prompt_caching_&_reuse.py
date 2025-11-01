"""
Pattern 166: Prompt Caching & Reuse

Description:
    The Prompt Caching & Reuse pattern stores and retrieves prompts and their results
    to avoid redundant LLM calls, reduce latency, and save costs. It uses semantic
    similarity to match queries with cached results and manages cache invalidation
    and updates.

Components:
    1. Cache Storage: Stores prompts and results
    2. Similarity Matcher: Finds semantically similar cached prompts
    3. Cache Manager: Handles cache lifecycle
    4. Invalidation Strategy: Manages cache freshness
    5. Hit/Miss Tracker: Monitors cache performance
    6. Compression: Optimizes cache storage

Use Cases:
    - Repeated query handling
    - Cost optimization
    - Latency reduction
    - Common question answering
    - Template-based generation
    - Batch processing optimization

Benefits:
    - Reduced API costs
    - Lower latency
    - Better resource utilization
    - Consistent responses
    - Offline capability

Trade-offs:
    - Storage overhead
    - Cache invalidation complexity
    - Stale data risk
    - Memory usage
    - Similarity matching overhead

LangChain Implementation:
    Combines semantic similarity search with LangChain caching mechanisms.
    Uses embeddings for matching similar queries efficiently.
"""

import os
import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


@dataclass
class CacheEntry:
    """Represents a cached prompt and response"""
    key: str
    prompt: str
    response: str
    timestamp: datetime
    hit_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ttl: Optional[int] = None  # Time-to-live in seconds
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl is None:
            return False
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return CacheEntry(**data)


class PromptCacheManager:
    """Manages prompt caching and reuse with semantic similarity matching"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 max_cache_size: int = 1000,
                 default_ttl: Optional[int] = None):
        """
        Initialize prompt cache manager
        
        Args:
            similarity_threshold: Minimum similarity score for cache hit (0-1)
            max_cache_size: Maximum number of entries to cache
            default_ttl: Default time-to-live in seconds
        """
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.default_ttl = default_ttl
        
        self.cache: Dict[str, CacheEntry] = {}
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_cost_saved": 0.0,
            "total_time_saved": 0.0
        }
    
    def get_or_generate(self, prompt: str, 
                       force_refresh: bool = False,
                       ttl: Optional[int] = None) -> Tuple[str, bool]:
        """
        Get cached response or generate new one
        
        Args:
            prompt: The prompt to process
            force_refresh: Skip cache and generate new response
            ttl: Time-to-live for this entry
            
        Returns:
            Tuple of (response, was_cached)
        """
        self.stats["total_requests"] += 1
        
        if not force_refresh:
            # Try to find in cache
            cached_response = self._find_similar_cached(prompt)
            if cached_response:
                self.stats["cache_hits"] += 1
                self.stats["total_cost_saved"] += 0.002  # Estimated savings
                self.stats["total_time_saved"] += 0.5  # Estimated time saved
                return cached_response, True
        
        # Cache miss - generate new response
        self.stats["cache_misses"] += 1
        response = self._generate_response(prompt)
        
        # Cache the result
        self._add_to_cache(prompt, response, ttl or self.default_ttl)
        
        return response, False
    
    def batch_get_or_generate(self, prompts: List[str]) -> List[Tuple[str, bool]]:
        """Process multiple prompts efficiently"""
        results = []
        for prompt in prompts:
            result, cached = self.get_or_generate(prompt)
            results.append((result, cached))
        return results
    
    def invalidate(self, pattern: Optional[str] = None):
        """Invalidate cache entries matching pattern"""
        if pattern is None:
            # Clear all
            self.cache.clear()
        else:
            # Remove matching entries
            keys_to_remove = [
                key for key, entry in self.cache.items()
                if pattern.lower() in entry.prompt.lower()
            ]
            for key in keys_to_remove:
                del self.cache[key]
    
    def _find_similar_cached(self, prompt: str) -> Optional[str]:
        """Find semantically similar cached prompt"""
        if not self.cache:
            return None
        
        # Clean expired entries
        self._clean_expired()
        
        # Get embedding for query prompt
        query_embedding = self.embeddings.embed_query(prompt)
        
        # Find most similar cached entry
        best_match = None
        best_similarity = 0.0
        
        for entry in self.cache.values():
            if entry.embedding is None:
                continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = entry
        
        if best_match:
            best_match.hit_count += 1
            return best_match.response
        
        return None
    
    def _generate_response(self, prompt: str) -> str:
        """Generate new response using LLM"""
        chain = ChatPromptTemplate.from_messages([
            ("user", "{prompt}")
        ]) | self.llm | StrOutputParser()
        
        response = chain.invoke({"prompt": prompt})
        return response
    
    def _add_to_cache(self, prompt: str, response: str, ttl: Optional[int] = None):
        """Add entry to cache"""
        # Generate cache key
        key = self._generate_key(prompt)
        
        # Get embedding
        embedding = self.embeddings.embed_query(prompt)
        
        # Create entry
        entry = CacheEntry(
            key=key,
            prompt=prompt,
            response=response,
            timestamp=datetime.now(),
            embedding=embedding,
            ttl=ttl
        )
        
        # Check cache size
        if len(self.cache) >= self.max_cache_size:
            self._evict_entry()
        
        # Add to cache
        self.cache[key] = entry
    
    def _generate_key(self, prompt: str) -> str:
        """Generate cache key from prompt"""
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _clean_expired(self):
        """Remove expired entries"""
        keys_to_remove = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        for key in keys_to_remove:
            del self.cache[key]
    
    def _evict_entry(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        # Find entry with lowest hit count and oldest timestamp
        lru_key = min(
            self.cache.keys(),
            key=lambda k: (self.cache[k].hit_count, self.cache[k].timestamp)
        )
        del self.cache[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.stats["total_requests"]
        hit_rate = self.stats["cache_hits"] / total if total > 0 else 0
        
        return {
            **self.stats,
            "cache_hit_rate": hit_rate,
            "cache_size": len(self.cache),
            "avg_hit_count": sum(e.hit_count for e in self.cache.values()) / len(self.cache)
                            if self.cache else 0
        }
    
    def get_popular_queries(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently accessed cached queries"""
        sorted_entries = sorted(
            self.cache.values(),
            key=lambda e: e.hit_count,
            reverse=True
        )[:top_n]
        
        return [
            {
                "prompt": entry.prompt,
                "hit_count": entry.hit_count,
                "age_seconds": (datetime.now() - entry.timestamp).total_seconds()
            }
            for entry in sorted_entries
        ]
    
    def save_cache(self, filepath: str):
        """Save cache to file"""
        cache_data = {
            key: entry.to_dict()
            for key, entry in self.cache.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def load_cache(self, filepath: str):
        """Load cache from file"""
        try:
            with open(filepath, 'r') as f:
                cache_data = json.load(f)
            
            self.cache = {
                key: CacheEntry.from_dict(data)
                for key, data in cache_data.items()
            }
        except FileNotFoundError:
            print(f"Cache file {filepath} not found")
        except Exception as e:
            print(f"Error loading cache: {e}")


def demonstrate_prompt_caching():
    """Demonstrate prompt caching and reuse"""
    print("=" * 80)
    print("PROMPT CACHING & REUSE PATTERN DEMONSTRATION")
    print("=" * 80)
    
    cache_manager = PromptCacheManager(
        similarity_threshold=0.85,
        max_cache_size=100,
        default_ttl=3600  # 1 hour
    )
    
    # Example 1: Basic caching
    print("\n" + "=" * 80)
    print("Example 1: Basic Caching")
    print("=" * 80)
    
    prompts = [
        "What is the capital of France?",
        "Tell me about the capital of France",  # Similar query
        "What is the capital of Germany?",
    ]
    
    print("\nProcessing prompts with caching:")
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Query: {prompt}")
        response, was_cached = cache_manager.get_or_generate(prompt)
        status = "CACHE HIT" if was_cached else "CACHE MISS"
        print(f"   Status: {status}")
        print(f"   Response: {response[:100]}...")
    
    # Example 2: Similar query matching
    print("\n" + "=" * 80)
    print("Example 2: Semantic Similarity Matching")
    print("=" * 80)
    
    original = "Explain quantum computing"
    similar = "Can you describe quantum computing?"
    
    print(f"\nOriginal query: {original}")
    response1, cached1 = cache_manager.get_or_generate(original)
    print(f"Status: {'CACHED' if cached1 else 'GENERATED'}")
    print(f"Response: {response1[:100]}...")
    
    print(f"\nSimilar query: {similar}")
    response2, cached2 = cache_manager.get_or_generate(similar)
    print(f"Status: {'CACHED' if cached2 else 'GENERATED'}")
    print(f"Match found: {cached2}")
    if cached2:
        print("✓ Similar query matched with cached result!")
    
    # Example 3: Batch processing
    print("\n" + "=" * 80)
    print("Example 3: Batch Processing with Cache")
    print("=" * 80)
    
    batch_prompts = [
        "What is machine learning?",
        "Define machine learning",  # Similar
        "What is deep learning?",
        "Explain deep learning",  # Similar
        "What is machine learning?",  # Exact duplicate
    ]
    
    print(f"\nProcessing batch of {len(batch_prompts)} prompts:")
    start_time = time.time()
    results = cache_manager.batch_get_or_generate(batch_prompts)
    elapsed = time.time() - start_time
    
    hits = sum(1 for _, cached in results if cached)
    misses = len(results) - hits
    
    print(f"\nResults:")
    print(f"  Total queries: {len(results)}")
    print(f"  Cache hits: {hits}")
    print(f"  Cache misses: {misses}")
    print(f"  Hit rate: {hits/len(results):.1%}")
    print(f"  Processing time: {elapsed:.2f}s")
    
    # Example 4: Cache statistics
    print("\n" + "=" * 80)
    print("Example 4: Cache Statistics")
    print("=" * 80)
    
    stats = cache_manager.get_stats()
    
    print("\nCache Performance:")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Cache size: {stats['cache_size']} entries")
    print(f"  Estimated cost saved: ${stats['total_cost_saved']:.4f}")
    print(f"  Estimated time saved: {stats['total_time_saved']:.2f}s")
    
    # Example 5: Popular queries
    print("\n" + "=" * 80)
    print("Example 5: Popular Queries Analysis")
    print("=" * 80)
    
    popular = cache_manager.get_popular_queries(top_n=5)
    
    print("\nMost frequently accessed queries:")
    for i, query_info in enumerate(popular, 1):
        age_min = query_info['age_seconds'] / 60
        print(f"\n{i}. Hit count: {query_info['hit_count']}")
        print(f"   Age: {age_min:.1f} minutes")
        print(f"   Query: {query_info['prompt'][:60]}...")
    
    # Example 6: TTL and expiration
    print("\n" + "=" * 80)
    print("Example 6: Time-to-Live (TTL) Management")
    print("=" * 80)
    
    # Add entry with short TTL
    short_ttl_prompt = "What's the current time?"
    response, cached = cache_manager.get_or_generate(short_ttl_prompt, ttl=2)
    print(f"\nAdded query with 2 second TTL: {short_ttl_prompt}")
    print(f"Response: {response[:80]}...")
    
    print("\nImmediate retry (should hit cache):")
    response, cached = cache_manager.get_or_generate(short_ttl_prompt)
    print(f"Cached: {cached}")
    
    print("\nWaiting 3 seconds for expiration...")
    time.sleep(3)
    
    print("Retry after expiration (should miss cache):")
    response, cached = cache_manager.get_or_generate(short_ttl_prompt)
    print(f"Cached: {cached}")
    
    # Example 7: Cache invalidation
    print("\n" + "=" * 80)
    print("Example 7: Cache Invalidation")
    print("=" * 80)
    
    print(f"\nCache size before invalidation: {len(cache_manager.cache)}")
    
    # Invalidate specific pattern
    cache_manager.invalidate(pattern="capital")
    print(f"Cache size after invalidating 'capital' queries: {len(cache_manager.cache)}")
    
    # Test that invalidated query is regenerated
    test_prompt = "What is the capital of France?"
    response, cached = cache_manager.get_or_generate(test_prompt)
    print(f"\nRetrying invalidated query: {test_prompt}")
    print(f"Cached: {cached} (should be False)")
    
    # Final statistics
    print("\n" + "=" * 80)
    print("Final Statistics")
    print("=" * 80)
    
    final_stats = cache_manager.get_stats()
    print(f"\nOverall Performance:")
    print(f"  Total requests processed: {final_stats['total_requests']}")
    print(f"  Overall hit rate: {final_stats['cache_hit_rate']:.1%}")
    print(f"  Final cache size: {final_stats['cache_size']} entries")
    print(f"  Total cost saved: ${final_stats['total_cost_saved']:.4f}")
    print(f"  Total time saved: {final_stats['total_time_saved']:.2f}s")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Prompt Caching & Reuse pattern enables:
✓ Semantic similarity matching for cache hits
✓ Significant cost reduction through reuse
✓ Lower latency for repeated queries
✓ TTL-based cache expiration
✓ Cache invalidation strategies
✓ Performance monitoring and analytics
✓ Batch processing optimization

This pattern is valuable for:
- High-traffic applications with repeated queries
- Cost-sensitive deployments
- Latency-critical systems
- Common question answering systems
- Template-based generation
- Offline or low-connectivity scenarios
    """)


if __name__ == "__main__":
    demonstrate_prompt_caching()
