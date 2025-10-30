"""
Semantic Caching Pattern

Caches responses based on semantic similarity rather than exact matching.
Reduces redundant LLM calls for similar queries.

Use Cases:
- Cost optimization
- Latency reduction
- FAQ systems
- Repetitive query handling

Advantages:
- Significant cost savings
- Faster response times
- Handles query variations
- Automatic cache invalidation
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib


class CacheStrategy(Enum):
    """Cache management strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SEMANTIC_TTL = "semantic_ttl"  # TTL with semantic similarity


@dataclass
class CacheEntry:
    """Entry in semantic cache"""
    entry_id: str
    query: str
    query_embedding: List[float]
    response: str
    metadata: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class CacheStats:
    """Cache statistics"""
    total_entries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_similarity_score: float
    total_cost_saved: float
    total_time_saved: float


class SemanticEmbedder:
    """Generates semantic embeddings for queries"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Simplified embedding using hash-based approach
        # In production, use actual embedding model like sentence-transformers
        text_normalized = text.lower().strip()
        
        # Generate multiple hash values
        embedding = []
        for i in range(self.embedding_dim):
            hash_obj = hashlib.sha256((text_normalized + str(i)).encode())
            hash_val = int(hash_obj.hexdigest(), 16) % 10000
            embedding.append(hash_val / 10000.0)
        
        return embedding
    
    def compute_similarity(self,
                          embedding1: List[float],
                          embedding2: List[float]) -> float:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        if len(embedding1) != len(embedding2):
            return 0.0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class CacheManager:
    """Manages cache entries and eviction"""
    
    def __init__(self,
                 max_entries: int = 1000,
                 strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_entries = max_entries
        self.strategy = strategy
        self.entries: Dict[str, CacheEntry] = {}
    
    def add_entry(self, entry: CacheEntry) -> None:
        """Add entry to cache"""
        # Check if cache is full
        if len(self.entries) >= self.max_entries:
            self._evict_entry()
        
        self.entries[entry.entry_id] = entry
    
    def get_entry(self, entry_id: str) -> Optional[CacheEntry]:
        """Get entry from cache"""
        entry = self.entries.get(entry_id)
        
        if entry:
            # Update access info
            entry.last_accessed = datetime.now()
            entry.access_count += 1
        
        return entry
    
    def remove_entry(self, entry_id: str) -> bool:
        """Remove entry from cache"""
        if entry_id in self.entries:
            del self.entries[entry_id]
            return True
        return False
    
    def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate entries with specific tags"""
        removed = 0
        to_remove = []
        
        for entry_id, entry in self.entries.items():
            if any(tag in entry.tags for tag in tags):
                to_remove.append(entry_id)
        
        for entry_id in to_remove:
            self.remove_entry(entry_id)
            removed += 1
        
        return removed
    
    def clean_expired(self) -> int:
        """Remove expired entries"""
        removed = 0
        to_remove = []
        current_time = datetime.now()
        
        for entry_id, entry in self.entries.items():
            if entry.ttl_seconds:
                age = (current_time - entry.created_at).total_seconds()
                if age > entry.ttl_seconds:
                    to_remove.append(entry_id)
        
        for entry_id in to_remove:
            self.remove_entry(entry_id)
            removed += 1
        
        return removed
    
    def _evict_entry(self) -> None:
        """Evict entry based on strategy"""
        if not self.entries:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently accessed
            oldest = min(self.entries.values(),
                        key=lambda e: e.last_accessed)
            self.remove_entry(oldest.entry_id)
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently accessed
            least_used = min(self.entries.values(),
                           key=lambda e: e.access_count)
            self.remove_entry(least_used.entry_id)
        
        elif self.strategy == CacheStrategy.TTL:
            # Remove oldest by creation time
            oldest = min(self.entries.values(),
                        key=lambda e: e.created_at)
            self.remove_entry(oldest.entry_id)
    
    def get_all_entries(self) -> List[CacheEntry]:
        """Get all cache entries"""
        return list(self.entries.values())


class SemanticCachingAgent:
    """
    Agent that implements semantic caching for LLM queries.
    Matches queries based on semantic similarity instead of exact match.
    """
    
    def __init__(self,
                 similarity_threshold: float = 0.85,
                 max_cache_entries: int = 1000,
                 default_ttl: int = 3600,
                 cache_strategy: CacheStrategy = CacheStrategy.LRU):
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        
        self.embedder = SemanticEmbedder()
        self.cache_manager = CacheManager(max_cache_entries, cache_strategy)
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.similarity_scores: List[float] = []
        self.cost_saved = 0.0  # In dollars
        self.time_saved = 0.0  # In seconds
    
    def query(self,
             query: str,
             force_refresh: bool = False,
             tags: Optional[List[str]] = None,
             metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, bool]:
        """
        Query with semantic caching.
        
        Args:
            query: Query string
            force_refresh: Force cache miss
            tags: Optional tags for cache entry
            metadata: Optional metadata
            
        Returns:
            (response, from_cache) tuple
        """
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}
        
        # Clean expired entries
        self.cache_manager.clean_expired()
        
        if not force_refresh:
            # Try to find similar cached query
            cache_hit = self._find_similar_query(query)
            
            if cache_hit:
                self.cache_hits += 1
                self.cost_saved += metadata.get("estimated_cost", 0.01)
                self.time_saved += metadata.get("estimated_latency", 1.0)
                return cache_hit.response, True
        
        # Cache miss - generate new response
        self.cache_misses += 1
        response = self._generate_response(query, metadata)
        
        # Cache the response
        self._cache_response(query, response, tags, metadata)
        
        return response, False
    
    def invalidate_cache(self,
                        query: Optional[str] = None,
                        tags: Optional[List[str]] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            query: Specific query to invalidate
            tags: Tags to invalidate
            
        Returns:
            Number of entries invalidated
        """
        if tags:
            return self.cache_manager.invalidate_by_tags(tags)
        
        if query:
            query_embedding = self.embedder.embed_text(query)
            cache_hit = self._find_similar_query(query)
            if cache_hit:
                self.cache_manager.remove_entry(cache_hit.entry_id)
                return 1
        
        return 0
    
    def get_cache_stats(self) -> CacheStats:
        """Get cache statistics"""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_queries) if total_queries > 0 else 0.0
        
        avg_similarity = (
            sum(self.similarity_scores) / len(self.similarity_scores)
            if self.similarity_scores else 0.0
        )
        
        return CacheStats(
            total_entries=len(self.cache_manager.entries),
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            hit_rate=hit_rate,
            avg_similarity_score=avg_similarity,
            total_cost_saved=self.cost_saved,
            total_time_saved=self.time_saved
        )
    
    def _find_similar_query(self, query: str) -> Optional[CacheEntry]:
        """Find semantically similar cached query"""
        query_embedding = self.embedder.embed_text(query)
        
        best_match = None
        best_similarity = 0.0
        
        for entry in self.cache_manager.get_all_entries():
            similarity = self.embedder.compute_similarity(
                query_embedding,
                entry.query_embedding
            )
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = entry
        
        if best_match:
            self.similarity_scores.append(best_similarity)
        
        return best_match
    
    def _generate_response(self,
                          query: str,
                          metadata: Dict[str, Any]) -> str:
        """
        Generate response (simulate LLM call).
        
        Args:
            query: Query string
            metadata: Optional metadata
            
        Returns:
            Generated response
        """
        # Simulate LLM response generation
        # In production, this would call actual LLM
        return "Response to: {}".format(query)
    
    def _cache_response(self,
                       query: str,
                       response: str,
                       tags: List[str],
                       metadata: Dict[str, Any]) -> None:
        """Cache a query-response pair"""
        query_embedding = self.embedder.embed_text(query)
        
        entry = CacheEntry(
            entry_id="cache_{}".format(len(self.cache_manager.entries)),
            query=query,
            query_embedding=query_embedding,
            response=response,
            metadata=metadata,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            ttl_seconds=metadata.get("ttl", self.default_ttl),
            tags=tags
        )
        
        self.cache_manager.add_entry(entry)
    
    def export_cache(self) -> List[Dict[str, Any]]:
        """Export cache entries"""
        return [
            {
                "entry_id": entry.entry_id,
                "query": entry.query,
                "response": entry.response,
                "created_at": entry.created_at.isoformat(),
                "access_count": entry.access_count,
                "tags": entry.tags
            }
            for entry in self.cache_manager.get_all_entries()
        ]


def demonstrate_semantic_caching():
    """Demonstrate semantic caching agent"""
    print("=" * 70)
    print("Semantic Caching Agent Demonstration")
    print("=" * 70)
    
    agent = SemanticCachingAgent(
        similarity_threshold=0.85,
        max_cache_entries=100,
        default_ttl=3600
    )
    
    # Example 1: Cache miss (first query)
    print("\n1. First Query (Cache Miss Expected):")
    query1 = "What is machine learning?"
    response1, from_cache1 = agent.query(
        query1,
        metadata={"estimated_cost": 0.01, "estimated_latency": 1.5}
    )
    print("Query: {}".format(query1))
    print("Response: {}".format(response1))
    print("From cache: {}".format(from_cache1))
    
    # Example 2: Similar query (cache hit expected)
    print("\n2. Similar Query (Cache Hit Expected):")
    query2 = "What is ML?"  # Similar to "What is machine learning?"
    response2, from_cache2 = agent.query(
        query2,
        metadata={"estimated_cost": 0.01, "estimated_latency": 1.5}
    )
    print("Query: {}".format(query2))
    print("Response: {}".format(response2))
    print("From cache: {}".format(from_cache2))
    
    # Example 3: Different query (cache miss)
    print("\n3. Different Query (Cache Miss Expected):")
    query3 = "How do neural networks work?"
    response3, from_cache3 = agent.query(
        query3,
        tags=["neural-networks", "deep-learning"],
        metadata={"estimated_cost": 0.01, "estimated_latency": 1.5}
    )
    print("Query: {}".format(query3))
    print("From cache: {}".format(from_cache3))
    
    # Example 4: Variations of queries
    print("\n4. Testing Query Variations:")
    variations = [
        "Explain machine learning",
        "Tell me about machine learning",
        "What's machine learning?",
        "Define machine learning"
    ]
    
    for i, var_query in enumerate(variations, 1):
        response, from_cache = agent.query(
            var_query,
            metadata={"estimated_cost": 0.01, "estimated_latency": 1.5}
        )
        print("\n  {}) Query: {}".format(i, var_query))
        print("     From cache: {}".format(from_cache))
    
    # Example 5: Cache statistics
    print("\n5. Cache Statistics:")
    stats = agent.get_cache_stats()
    print("Total entries: {}".format(stats.total_entries))
    print("Cache hits: {}".format(stats.cache_hits))
    print("Cache misses: {}".format(stats.cache_misses))
    print("Hit rate: {:.1%}".format(stats.hit_rate))
    print("Avg similarity score: {:.3f}".format(stats.avg_similarity_score))
    print("Cost saved: ${:.4f}".format(stats.total_cost_saved))
    print("Time saved: {:.2f}s".format(stats.total_time_saved))
    
    # Example 6: Cache invalidation
    print("\n6. Cache Invalidation by Tags:")
    invalidated = agent.invalidate_cache(tags=["neural-networks"])
    print("Invalidated {} entries with 'neural-networks' tag".format(invalidated))
    
    # Example 7: Export cache
    print("\n7. Cache Export (First 3 Entries):")
    exported = agent.export_cache()
    for entry in exported[:3]:
        print("\nEntry ID: {}".format(entry["entry_id"]))
        print("  Query: {}".format(entry["query"]))
        print("  Access count: {}".format(entry["access_count"]))
        print("  Tags: {}".format(entry["tags"]))


if __name__ == "__main__":
    demonstrate_semantic_caching()
