"""
Caching Patterns
Different levels of caching for performance optimization
"""
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import time
@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int
    ttl_seconds: Optional[int]
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds
class LRUCache:
    """Least Recently Used cache"""
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: list = []  # Track access order
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
        entry = self.cache[key]
        # Check expiration
        if entry.is_expired():
            del self.cache[key]
            self.access_order.remove(key)
            return None
        # Update access info
        entry.accessed_at = datetime.now()
        entry.access_count += 1
        # Move to end (most recently used)
        self.access_order.remove(key)
        self.access_order.append(key)
        return entry.value
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Put value in cache"""
        # Remove if already exists
        if key in self.cache:
            self.access_order.remove(key)
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        # Add new entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            ttl_seconds=ttl_seconds
        )
        self.cache[key] = entry
        self.access_order.append(key)
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_order.clear()
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_accesses": total_accesses,
            "utilization": len(self.cache) / self.max_size
        }
class SemanticCache:
    """Cache based on semantic similarity of queries"""
    def __init__(self, similarity_threshold: float = 0.9):
        self.cache: Dict[str, CacheEntry] = {}
        self.similarity_threshold = similarity_threshold
    def _compute_embedding(self, text: str) -> list:
        """Compute simple embedding (in reality, use proper embedding model)"""
        # Simple word-based embedding for demonstration
        words = text.lower().split()
        embedding = [0] * 100
        for word in words:
            idx = hash(word) % 100
            embedding[idx] += 1
        # Normalize
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        return embedding
    def _cosine_similarity(self, emb1: list, emb2: list) -> float:
        """Calculate cosine similarity"""
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        return dot_product
    def get(self, query: str) -> Optional[Any]:
        """Get cached result for semantically similar query"""
        query_embedding = self._compute_embedding(query)
        best_match = None
        best_similarity = 0
        for key, entry in self.cache.items():
            if entry.is_expired():
                continue
            cached_embedding = self._compute_embedding(key)
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        if best_similarity >= self.similarity_threshold:
            print(f"  Cache hit (similarity: {best_similarity:.2f})")
            best_match.access_count += 1
            return best_match.value
        print(f"  Cache miss (best similarity: {best_similarity:.2f})")
        return None
    def put(self, query: str, value: Any, ttl_seconds: Optional[int] = 3600):
        """Cache result for query"""
        entry = CacheEntry(
            key=query,
            value=value,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            ttl_seconds=ttl_seconds
        )
        self.cache[query] = entry
class PromptCache:
    """Cache for LLM prompts and responses"""
    def __init__(self, max_size: int = 50):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
    def _hash_prompt(self, prompt: str, model: str, temperature: float) -> str:
        """Create hash of prompt parameters"""
        key_data = f"{prompt}:{model}:{temperature}"
        return hashlib.md5(key_data.encode()).hexdigest()
    def get(self, prompt: str, model: str = "default", temperature: float = 0.0) -> Optional[str]:
        """Get cached response"""
        key = self._hash_prompt(prompt, model, temperature)
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                print(f"  Prompt cache hit")
                entry.access_count += 1
                return entry.value
            else:
                del self.cache[key]
        print(f"  Prompt cache miss")
        return None
    def put(self, prompt: str, response: str, model: str = "default", 
            temperature: float = 0.0, ttl_seconds: int = 3600):
        """Cache prompt response"""
        key = self._hash_prompt(prompt, model, temperature)
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].created_at)
            del self.cache[oldest_key]
        entry = CacheEntry(
            key=key,
            value=response,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            ttl_seconds=ttl_seconds
        )
        self.cache[key] = entry
class CachedAgent:
    """Agent with multi-level caching"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.lru_cache = LRUCache(max_size=100)
        self.semantic_cache = SemanticCache(similarity_threshold=0.85)
        self.prompt_cache = PromptCache(max_size=50)
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_latency_saved_ms = 0
    def query(self, question: str) -> str:
        """Query with caching"""
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print(f"{'='*60}")
        start_time = time.time()
        # Try semantic cache first
        print("\nChecking semantic cache...")
        cached_response = self.semantic_cache.get(question)
        if cached_response:
            self.cache_hits += 1
            latency_saved = 2000  # Assume 2s saved
            self.total_latency_saved_ms += latency_saved
            print(f"✓ Returning cached response")
            print(f"  Latency saved: {latency_saved}ms")
            return cached_response
        # Cache miss - generate response
        self.cache_misses += 1
        print("✗ No cache hit, generating response...")
        # Simulate LLM call
        time.sleep(0.5)  # Simulate latency
        response = f"Generated response for: {question}"
        # Cache the response
        self.semantic_cache.put(question, response)
        latency = (time.time() - start_time) * 1000
        print(f"  Generated in {latency:.0f}ms")
        return response
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        return {
            "total_requests": total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_latency_saved_ms": self.total_latency_saved_ms,
            "lru_cache_stats": self.lru_cache.get_stats()
        }
    def print_cache_report(self):
        """Print cache performance report"""
        stats = self.get_cache_stats()
        print(f"\n{'='*70}")
        print(f"CACHE PERFORMANCE REPORT")
        print(f"{'='*70}")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Cache Hits: {stats['cache_hits']}")
        print(f"Cache Misses: {stats['cache_misses']}")
        print(f"Hit Rate: {stats['hit_rate']:.1%}")
        print(f"Total Latency Saved: {stats['total_latency_saved_ms']:.0f}ms")
        if stats['total_requests'] > 0:
            avg_saved = stats['total_latency_saved_ms'] / stats['total_requests']
            print(f"Avg Latency Saved per Request: {avg_saved:.0f}ms")
# Usage
if __name__ == "__main__":
    print("="*80)
    print("CACHING PATTERNS DEMONSTRATION")
    print("="*80)
    agent = CachedAgent("cached-agent-001")
    # Test queries (some similar, some different)
    queries = [
        "What is machine learning?",
        "Explain machine learning",  # Similar to first
        "What are neural networks?",
        "Tell me about deep learning",
        "What is machine learning?",  # Exact repeat
        "How does machine learning work?",  # Similar to first
    ]
    for query in queries:
        response = agent.query(query)
        time.sleep(0.1)
    # Print cache report
    agent.print_cache_report()
