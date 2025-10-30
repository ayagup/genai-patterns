"""
Agentic AI Design Pattern: Memory Augmentation

This pattern implements memory augmentation techniques that extend an agent's
memory capacity and capabilities through external memory systems, hierarchical
storage, and intelligent memory management.

Key Concepts:
1. External Memory: Store information in external systems (databases, files)
2. Memory Expansion: Expand beyond working memory limits
3. Hierarchical Storage: Multi-tier memory architecture
4. Intelligent Retrieval: Smart access to augmented memory
5. Memory Compression: Efficient storage of large information

Use Cases:
- Long-running agents requiring persistent memory
- Knowledge-intensive tasks
- Learning systems with growing knowledge bases
- Agents handling large documents or datasets
- Personal assistants with user history
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import hashlib


class MemoryTier(Enum):
    """Tiers in memory hierarchy"""
    WORKING = "working"  # Active, immediate access
    SHORT_TERM = "short_term"  # Recent, fast access
    LONG_TERM = "long_term"  # Persistent, slower access
    ARCHIVE = "archive"  # Cold storage, rare access


class MemoryImportance(Enum):
    """Importance levels for memory prioritization"""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


@dataclass
class MemoryEntry:
    """Single memory entry with metadata"""
    entry_id: str
    content: Any
    timestamp: datetime
    importance: MemoryImportance
    access_count: int = 0
    last_access: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed: bool = False
    
    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = str(uuid.uuid4())
        if self.last_access is None:
            self.last_access = self.timestamp
    
    def access(self) -> None:
        """Record access to this memory"""
        self.access_count += 1
        self.last_access = datetime.now()
    
    def get_hash(self) -> str:
        """Generate hash of content for deduplication"""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def recency_score(self) -> float:
        """Calculate recency score (0-1)"""
        if self.last_access is None:
            return 0.0
        hours_since_access = (datetime.now() - self.last_access).total_seconds() / 3600
        # Exponential decay: score decreases over time
        return 1.0 / (1.0 + hours_since_access / 24.0)
    
    def frequency_score(self) -> float:
        """Calculate frequency score based on access count"""
        # Logarithmic scaling
        import math
        return math.log(1 + self.access_count) / math.log(100)
    
    def combined_score(self) -> float:
        """Combined score for memory importance"""
        recency = self.recency_score()
        frequency = self.frequency_score()
        importance = self.importance.value / 4.0
        
        # Weighted combination
        return (recency * 0.3 + frequency * 0.3 + importance * 0.4)


@dataclass
class MemoryStats:
    """Statistics about memory usage"""
    total_entries: int = 0
    entries_by_tier: Dict[MemoryTier, int] = field(default_factory=dict)
    total_size_bytes: int = 0
    compression_ratio: float = 1.0
    cache_hit_rate: float = 0.0
    
    def __str__(self) -> str:
        return (f"MemoryStats(entries={self.total_entries}, "
                f"size={self.total_size_bytes / 1024:.2f}KB, "
                f"compression={self.compression_ratio:.2f}x, "
                f"cache_hit_rate={self.cache_hit_rate:.1%})")


class ExternalMemoryStore:
    """External memory storage system"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.storage: Dict[str, MemoryEntry] = {}
        self.indices: Dict[str, Set[str]] = {}  # Tag -> entry_ids
        
    def store(self, entry: MemoryEntry) -> bool:
        """Store entry in external memory"""
        if len(self.storage) >= self.capacity:
            # Evict least important entries
            self._evict_entries(1)
        
        self.storage[entry.entry_id] = entry
        
        # Update indices
        for tag in entry.tags:
            if tag not in self.indices:
                self.indices[tag] = set()
            self.indices[tag].add(entry.entry_id)
        
        return True
    
    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve entry by ID"""
        entry = self.storage.get(entry_id)
        if entry:
            entry.access()
        return entry
    
    def search_by_tags(self, tags: Set[str]) -> List[MemoryEntry]:
        """Search entries by tags"""
        if not tags:
            return list(self.storage.values())
        
        # Find entries matching all tags
        matching_ids = None
        for tag in tags:
            tag_ids = self.indices.get(tag, set())
            if matching_ids is None:
                matching_ids = tag_ids.copy()
            else:
                matching_ids &= tag_ids
        
        if matching_ids is None:
            return []
        
        results = []
        for entry_id in matching_ids:
            entry = self.storage.get(entry_id)
            if entry:
                entry.access()
                results.append(entry)
        
        return results
    
    def _evict_entries(self, count: int) -> None:
        """Evict least important entries"""
        if not self.storage:
            return
        
        # Sort by combined score (ascending)
        entries = sorted(self.storage.values(), key=lambda e: e.combined_score())
        
        for entry in entries[:count]:
            self._remove_entry(entry.entry_id)
    
    def _remove_entry(self, entry_id: str) -> None:
        """Remove entry from storage and indices"""
        entry = self.storage.get(entry_id)
        if not entry:
            return
        
        # Remove from storage
        del self.storage[entry_id]
        
        # Remove from indices
        for tag in entry.tags:
            if tag in self.indices:
                self.indices[tag].discard(entry_id)
                if not self.indices[tag]:
                    del self.indices[tag]
    
    def get_size(self) -> int:
        """Get current size of storage"""
        return len(self.storage)


class MemoryCache:
    """Fast cache for frequently accessed memories"""
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache: Dict[str, MemoryEntry] = {}
        self.access_order: List[str] = []  # LRU tracking
        self.hits = 0
        self.misses = 0
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get entry from cache"""
        if entry_id in self.cache:
            self.hits += 1
            # Update LRU order
            self.access_order.remove(entry_id)
            self.access_order.append(entry_id)
            return self.cache[entry_id]
        
        self.misses += 1
        return None
    
    def put(self, entry: MemoryEntry) -> None:
        """Put entry in cache"""
        if entry.entry_id in self.cache:
            # Already in cache, update LRU order
            self.access_order.remove(entry.entry_id)
        elif len(self.cache) >= self.capacity:
            # Evict least recently used
            lru_id = self.access_order.pop(0)
            del self.cache[lru_id]
        
        self.cache[entry.entry_id] = entry
        self.access_order.append(entry.entry_id)
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


class HierarchicalMemory:
    """Multi-tier hierarchical memory system"""
    
    def __init__(self, working_capacity: int = 10, 
                 short_term_capacity: int = 100,
                 long_term_capacity: int = 10000):
        self.tiers: Dict[MemoryTier, List[MemoryEntry]] = {
            MemoryTier.WORKING: [],
            MemoryTier.SHORT_TERM: [],
            MemoryTier.LONG_TERM: [],
            MemoryTier.ARCHIVE: []
        }
        self.capacities = {
            MemoryTier.WORKING: working_capacity,
            MemoryTier.SHORT_TERM: short_term_capacity,
            MemoryTier.LONG_TERM: long_term_capacity,
            MemoryTier.ARCHIVE: float('inf')
        }
        self.tier_by_id: Dict[str, MemoryTier] = {}
    
    def store(self, entry: MemoryEntry, tier: MemoryTier = MemoryTier.WORKING) -> None:
        """Store entry in specified tier"""
        # Check capacity
        if len(self.tiers[tier]) >= self.capacities[tier]:
            self._promote_demote(tier)
        
        self.tiers[tier].append(entry)
        self.tier_by_id[entry.entry_id] = tier
    
    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve entry from any tier"""
        tier = self.tier_by_id.get(entry_id)
        if tier is None:
            return None
        
        for entry in self.tiers[tier]:
            if entry.entry_id == entry_id:
                entry.access()
                # Promote frequently accessed items
                if entry.access_count > 5 and tier != MemoryTier.WORKING:
                    self._promote_entry(entry, tier)
                return entry
        
        return None
    
    def _promote_demote(self, tier: MemoryTier) -> None:
        """Promote important items, demote less important ones"""
        if tier == MemoryTier.ARCHIVE:
            return  # Archive has no demotion
        
        entries = self.tiers[tier]
        if not entries:
            return
        
        # Sort by combined score
        sorted_entries = sorted(entries, key=lambda e: e.combined_score(), reverse=True)
        
        # Keep top entries, demote bottom entries
        keep_count = int(self.capacities[tier] * 0.8)
        to_keep = sorted_entries[:keep_count]
        to_demote = sorted_entries[keep_count:]
        
        # Update tier
        self.tiers[tier] = to_keep
        
        # Demote entries to next tier
        next_tier = self._get_next_tier(tier)
        if next_tier:
            for entry in to_demote:
                self.tiers[next_tier].append(entry)
                self.tier_by_id[entry.entry_id] = next_tier
    
    def _promote_entry(self, entry: MemoryEntry, from_tier: MemoryTier) -> None:
        """Promote entry to higher tier"""
        prev_tier = self._get_prev_tier(from_tier)
        if prev_tier is None:
            return
        
        # Remove from current tier
        self.tiers[from_tier].remove(entry)
        
        # Add to higher tier
        self.store(entry, prev_tier)
    
    def _get_next_tier(self, tier: MemoryTier) -> Optional[MemoryTier]:
        """Get next (lower) tier"""
        tier_order = [MemoryTier.WORKING, MemoryTier.SHORT_TERM, 
                     MemoryTier.LONG_TERM, MemoryTier.ARCHIVE]
        try:
            idx = tier_order.index(tier)
            if idx < len(tier_order) - 1:
                return tier_order[idx + 1]
        except ValueError:
            pass
        return None
    
    def _get_prev_tier(self, tier: MemoryTier) -> Optional[MemoryTier]:
        """Get previous (higher) tier"""
        tier_order = [MemoryTier.WORKING, MemoryTier.SHORT_TERM, 
                     MemoryTier.LONG_TERM, MemoryTier.ARCHIVE]
        try:
            idx = tier_order.index(tier)
            if idx > 0:
                return tier_order[idx - 1]
        except ValueError:
            pass
        return None
    
    def get_stats(self) -> MemoryStats:
        """Get memory statistics"""
        stats = MemoryStats()
        stats.total_entries = sum(len(entries) for entries in self.tiers.values())
        stats.entries_by_tier = {
            tier: len(entries) for tier, entries in self.tiers.items()
        }
        return stats


class AugmentedMemorySystem:
    """Complete augmented memory system with all features"""
    
    def __init__(self, working_capacity: int = 10, cache_capacity: int = 50):
        self.hierarchical_memory = HierarchicalMemory(
            working_capacity=working_capacity,
            short_term_capacity=working_capacity * 10,
            long_term_capacity=working_capacity * 100
        )
        self.external_store = ExternalMemoryStore(capacity=working_capacity * 1000)
        self.cache = MemoryCache(capacity=cache_capacity)
        self.deduplication_hashes: Dict[str, str] = {}  # hash -> entry_id
    
    def store(self, content: Any, importance: MemoryImportance = MemoryImportance.MEDIUM,
             tags: Optional[Set[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store memory with automatic tier selection"""
        entry = MemoryEntry(
            entry_id=str(uuid.uuid4()),
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            tags=tags or set(),
            metadata=metadata or {}
        )
        
        # Check for duplicates
        content_hash = entry.get_hash()
        if content_hash in self.deduplication_hashes:
            # Return existing entry ID
            return self.deduplication_hashes[content_hash]
        
        self.deduplication_hashes[content_hash] = entry.entry_id
        
        # Store in appropriate tier based on importance
        if importance == MemoryImportance.CRITICAL:
            tier = MemoryTier.WORKING
        elif importance == MemoryImportance.HIGH:
            tier = MemoryTier.SHORT_TERM
        else:
            tier = MemoryTier.LONG_TERM
        
        self.hierarchical_memory.store(entry, tier)
        self.external_store.store(entry)
        
        return entry.entry_id
    
    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve memory with caching"""
        # Check cache first
        cached = self.cache.get(entry_id)
        if cached:
            return cached
        
        # Check hierarchical memory
        entry = self.hierarchical_memory.retrieve(entry_id)
        if entry:
            self.cache.put(entry)
            return entry
        
        # Check external store
        entry = self.external_store.retrieve(entry_id)
        if entry:
            self.cache.put(entry)
            return entry
        
        return None
    
    def search(self, tags: Optional[Set[str]] = None, 
              min_importance: Optional[MemoryImportance] = None,
              max_age_hours: Optional[float] = None) -> List[MemoryEntry]:
        """Search memories with filters"""
        # Search external store
        results = self.external_store.search_by_tags(tags or set())
        
        # Apply filters
        filtered = []
        now = datetime.now()
        
        for entry in results:
            # Importance filter
            if min_importance and entry.importance.value < min_importance.value:
                continue
            
            # Age filter
            if max_age_hours:
                age_hours = (now - entry.timestamp).total_seconds() / 3600
                if age_hours > max_age_hours:
                    continue
            
            filtered.append(entry)
        
        # Sort by combined score
        filtered.sort(key=lambda e: e.combined_score(), reverse=True)
        
        return filtered
    
    def get_statistics(self) -> MemoryStats:
        """Get comprehensive memory statistics"""
        stats = self.hierarchical_memory.get_stats()
        stats.cache_hit_rate = self.cache.hit_rate()
        stats.total_size_bytes = len(self.deduplication_hashes) * 1024  # Estimate
        return stats
    
    def consolidate_memories(self) -> int:
        """Consolidate and organize memories"""
        # This would implement memory consolidation strategies
        # For now, just trigger tier promotion/demotion
        consolidated = 0
        
        for tier in [MemoryTier.WORKING, MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM]:
            self.hierarchical_memory._promote_demote(tier)
            consolidated += 1
        
        return consolidated


def demonstrate_memory_augmentation():
    """Demonstrate memory augmentation system"""
    print("=" * 70)
    print("MEMORY AUGMENTATION DEMONSTRATION")
    print("=" * 70)
    
    # Create augmented memory system
    memory = AugmentedMemorySystem(working_capacity=5, cache_capacity=10)
    
    print("\n1. STORING MEMORIES WITH DIFFERENT IMPORTANCE LEVELS")
    print("-" * 70)
    
    # Store various memories
    memories = [
        ("Critical system configuration", MemoryImportance.CRITICAL, {"system", "config"}),
        ("User preferences", MemoryImportance.HIGH, {"user", "settings"}),
        ("Recent conversation turn 1", MemoryImportance.MEDIUM, {"conversation"}),
        ("Recent conversation turn 2", MemoryImportance.MEDIUM, {"conversation"}),
        ("Historical data point", MemoryImportance.LOW, {"data", "historical"}),
        ("Important fact", MemoryImportance.HIGH, {"fact", "important"}),
        ("Temporary calculation", MemoryImportance.LOW, {"temp", "calculation"}),
        ("User name: John", MemoryImportance.CRITICAL, {"user", "identity"}),
    ]
    
    stored_ids = []
    for content, importance, tags in memories:
        entry_id = memory.store(content, importance, tags)
        stored_ids.append(entry_id)
        print(f"   Stored: {content[:40]:<40} | {importance.name:<10} | {entry_id[:8]}")
    
    # Show statistics
    stats = memory.get_statistics()
    print(f"\n   {stats}")
    
    print("\n2. RETRIEVING MEMORIES (WITH CACHING)")
    print("-" * 70)
    
    # Retrieve some memories multiple times
    test_ids = stored_ids[:3]
    for _ in range(3):
        for entry_id in test_ids:
            entry = memory.retrieve(entry_id)
            if entry:
                print(f"   Retrieved: {str(entry.content)[:40]:<40} | Access count: {entry.access_count}")
    
    print(f"\n   Cache hit rate: {memory.cache.hit_rate():.1%}")
    
    print("\n3. SEARCHING MEMORIES BY TAGS")
    print("-" * 70)
    
    # Search by different tag combinations
    searches = [
        {"user"},
        {"conversation"},
        {"important"},
        {"system", "config"}
    ]
    
    for tags in searches:
        results = memory.search(tags=tags)
        print(f"\n   Search for tags {tags}:")
        print(f"   Found {len(results)} memories:")
        for entry in results[:3]:  # Show top 3
            print(f"     - {str(entry.content)[:50]} (score: {entry.combined_score():.3f})")
    
    print("\n4. FILTERING BY IMPORTANCE AND AGE")
    print("-" * 70)
    
    # Search with filters
    high_priority = memory.search(min_importance=MemoryImportance.HIGH)
    print(f"   High+ importance memories: {len(high_priority)}")
    for entry in high_priority:
        print(f"     - {str(entry.content)[:50]} ({entry.importance.name})")
    
    print("\n5. MEMORY CONSOLIDATION")
    print("-" * 70)
    
    print("   Before consolidation:")
    stats_before = memory.get_statistics()
    print(f"   {stats_before}")
    
    tiers_consolidated = memory.consolidate_memories()
    
    print(f"\n   Consolidated {tiers_consolidated} memory tiers")
    print("   After consolidation:")
    stats_after = memory.get_statistics()
    print(f"   {stats_after}")
    
    print("\n6. DEDUPLICATION TEST")
    print("-" * 70)
    
    # Try to store duplicate
    duplicate_content = "User name: John"
    id1 = memory.store(duplicate_content, MemoryImportance.CRITICAL, {"user"})
    id2 = memory.store(duplicate_content, MemoryImportance.CRITICAL, {"user"})
    
    print(f"   First storage ID:  {id1[:8]}")
    print(f"   Second storage ID: {id2[:8]}")
    print(f"   Deduplicated: {id1 == id2}")
    
    print("\n7. HIERARCHICAL MEMORY TIER DISTRIBUTION")
    print("-" * 70)
    
    stats = memory.get_statistics()
    print(f"   Memory distribution across tiers:")
    for tier, count in stats.entries_by_tier.items():
        print(f"     {tier.value:>15}: {count:>3} entries")
    
    print("\n8. MEMORY SCORING EXAMPLE")
    print("-" * 70)
    
    # Show scores for different memories
    print("   Memory scores (recency + frequency + importance):")
    for entry_id in stored_ids[:5]:
        entry = memory.retrieve(entry_id)
        if entry:
            print(f"     {str(entry.content)[:40]:<40}")
            print(f"       Recency: {entry.recency_score():.3f}, "
                  f"Frequency: {entry.frequency_score():.3f}, "
                  f"Combined: {entry.combined_score():.3f}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("1. Multi-tier hierarchical memory (working/short-term/long-term/archive)")
    print("2. Intelligent caching with LRU eviction")
    print("3. Tag-based search and retrieval")
    print("4. Importance-based storage and filtering")
    print("5. Automatic deduplication")
    print("6. Memory consolidation and organization")
    print("7. Access tracking and scoring")
    print("8. External memory store with large capacity")


if __name__ == "__main__":
    demonstrate_memory_augmentation()
