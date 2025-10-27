"""
Pattern 31: Memory Consolidation
Description:
    Consolidates short-term memories into long-term storage through
    processes of rehearsal, compression, and selective retention.
Use Cases:
    - Knowledge distillation
    - Memory optimization
    - Important information retention
    - Forgetting unimportant details
Key Features:
    - Importance-based consolidation
    - Memory compression
    - Selective retention
    - Periodic consolidation cycles
Example:
    >>> consolidator = MemoryConsolidator()
    >>> consolidator.add_short_term_memory(memory)
    >>> consolidator.consolidate()
    >>> long_term = consolidator.get_long_term_memories()
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import time
import math
from collections import defaultdict, deque
import heapq
class MemoryImportance(Enum):
    """Importance levels for memories"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    TRIVIAL = 1
class ConsolidationStrategy(Enum):
    """Strategies for memory consolidation"""
    IMPORTANCE_BASED = "importance_based"
    FREQUENCY_BASED = "frequency_based"
    RECENCY_BASED = "recency_based"
    SEMANTIC_CLUSTERING = "semantic_clustering"
    HYBRID = "hybrid"
@dataclass
class Memory:
    """A memory item"""
    memory_id: str
    content: Any
    importance: MemoryImportance
    created_at: float
    last_accessed: float = 0.0
    access_count: int = 0
    consolidation_score: float = 0.0
    is_consolidated: bool = False
    related_memories: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
@dataclass
class ConsolidatedMemory:
    """A consolidated long-term memory"""
    consolidated_id: str
    source_memory_ids: List[str]
    consolidated_content: Any
    importance: MemoryImportance
    consolidation_time: float
    access_count: int = 0
    last_accessed: float = 0.0
    compression_ratio: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
@dataclass
class ConsolidationMetrics:
    """Metrics for consolidation process"""
    memories_processed: int
    memories_consolidated: int
    memories_discarded: int
    compression_ratio: float
    consolidation_duration: float
    average_importance: float
class MemoryConsolidator:
    """
    System for consolidating short-term memories into long-term storage
    Features:
    - Importance-based retention
    - Memory compression
    - Periodic consolidation
    - Forgetting curve simulation
    """
    def __init__(
        self,
        strategy: ConsolidationStrategy = ConsolidationStrategy.HYBRID,
        consolidation_threshold: float = 0.5,
        max_short_term_capacity: int = 1000,
        forgetting_rate: float = 0.1
    ):
        self.strategy = strategy
        self.consolidation_threshold = consolidation_threshold
        self.max_short_term_capacity = max_short_term_capacity
        self.forgetting_rate = forgetting_rate
        self.short_term_memories: Dict[str, Memory] = {}
        self.long_term_memories: Dict[str, ConsolidatedMemory] = {}
        self.discarded_memories: List[str] = []
        self.consolidation_history: List[ConsolidationMetrics] = []
        self.memory_counter = 0
        self.consolidated_counter = 0
    def add_short_term_memory(
        self,
        content: Any,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a memory to short-term storage
        Args:
            content: Memory content
            importance: Importance level
            tags: Categorization tags
            metadata: Additional metadata
        Returns:
            Memory ID
        """
        self.memory_counter += 1
        memory_id = f"mem_{self.memory_counter}_{int(time.time())}"
        memory = Memory(
            memory_id=memory_id,
            content=content,
            importance=importance,
            created_at=time.time(),
            tags=tags or set(),
            metadata=metadata or {}
        )
        self.short_term_memories[memory_id] = memory
        # Trigger consolidation if capacity exceeded
        if len(self.short_term_memories) > self.max_short_term_capacity:
            self.consolidate()
        return memory_id
    def consolidate(
        self,
        force: bool = False
    ) -> ConsolidationMetrics:
        """
        Consolidate short-term memories into long-term storage
        Args:
            force: Force consolidation regardless of threshold
        Returns:
            Consolidation metrics
        """
        start_time = time.time()
        # Calculate consolidation scores
        self._calculate_consolidation_scores()
        # Apply forgetting curve
        self._apply_forgetting()
        # Select memories for consolidation
        to_consolidate, to_discard = self._select_memories_for_consolidation(force)
        # Consolidate selected memories
        consolidated_ids = self._perform_consolidation(to_consolidate)
        # Discard low-value memories
        self._discard_memories(to_discard)
        # Calculate metrics
        total_processed = len(to_consolidate) + len(to_discard)
        compression_ratio = (
            len(consolidated_ids) / len(to_consolidate)
            if to_consolidate else 1.0
        )
        avg_importance = (
            sum(mem.importance.value for mem in to_consolidate) / len(to_consolidate)
            if to_consolidate else 0.0
        )
        metrics = ConsolidationMetrics(
            memories_processed=total_processed,
            memories_consolidated=len(consolidated_ids),
            memories_discarded=len(to_discard),
            compression_ratio=compression_ratio,
            consolidation_duration=time.time() - start_time,
            average_importance=avg_importance
        )
        self.consolidation_history.append(metrics)
        return metrics
    def _calculate_consolidation_scores(self):
        """Calculate consolidation scores for all short-term memories"""
        current_time = time.time()
        for memory in self.short_term_memories.values():
            score = 0.0
            # Importance component
            importance_score = memory.importance.value / 5.0
            score += importance_score * 0.4
            # Access frequency component
            if memory.access_count > 0:
                frequency_score = min(memory.access_count / 10.0, 1.0)
                score += frequency_score * 0.3
            # Recency component
            age = current_time - memory.created_at
            recency_score = math.exp(-age / 86400)  # Decay over days
            score += recency_score * 0.2
            # Last access component
            if memory.last_accessed > 0:
                time_since_access = current_time - memory.last_accessed
                access_recency = math.exp(-time_since_access / 3600)  # Decay over hours
                score += access_recency * 0.1
            memory.consolidation_score = score
    def _apply_forgetting(self):
        """Apply forgetting curve to reduce scores over time"""
        current_time = time.time()
        for memory in self.short_term_memories.values():
            age = current_time - memory.created_at
            # Ebbinghaus forgetting curve
            retention = math.exp(-self.forgetting_rate * age / 86400)
            # Adjust consolidation score by retention
            memory.consolidation_score *= retention
    def _select_memories_for_consolidation(
        self,
        force: bool
    ) -> tuple[List[Memory], List[Memory]]:
        """Select which memories to consolidate vs discard"""
        memories_list = list(self.short_term_memories.values())
        to_consolidate = []
        to_discard = []
        for memory in memories_list:
            if force or memory.consolidation_score >= self.consolidation_threshold:
                to_consolidate.append(memory)
            elif memory.consolidation_score < self.consolidation_threshold * 0.3:
                # Very low score - candidate for discarding
                to_discard.append(memory)
        return to_consolidate, to_discard
    def _perform_consolidation(
        self,
        memories: List[Memory]
    ) -> List[str]:
        """Perform actual consolidation of memories"""
        consolidated_ids = []
        if self.strategy == ConsolidationStrategy.SEMANTIC_CLUSTERING:
            # Group by semantic similarity (tags)
            clusters = self._cluster_memories(memories)
            for cluster in clusters:
                if len(cluster) > 1:
                    # Consolidate cluster into single memory
                    consolidated_id = self._consolidate_cluster(cluster)
                    consolidated_ids.append(consolidated_id)
                else:
                    # Single memory - store as-is
                    consolidated_id = self._store_individual_memory(cluster[0])
                    consolidated_ids.append(consolidated_id)
        else:
            # Store each memory individually
            for memory in memories:
                consolidated_id = self._store_individual_memory(memory)
                consolidated_ids.append(consolidated_id)
        # Remove from short-term
        for memory in memories:
            if memory.memory_id in self.short_term_memories:
                del self.short_term_memories[memory.memory_id]
        return consolidated_ids
    def _cluster_memories(
        self,
        memories: List[Memory]
    ) -> List[List[Memory]]:
        """Cluster memories by semantic similarity"""
        # Group by tag overlap
        tag_groups: Dict[frozenset, List[Memory]] = defaultdict(list)
        for memory in memories:
            tag_key = frozenset(memory.tags)
            tag_groups[tag_key].append(memory)
        # Convert to list of clusters
        clusters = []
        for memory_group in tag_groups.values():
            if len(memory_group) >= 3:
                # Large enough to be a cluster
                clusters.append(memory_group)
            else:
                # Add as individual clusters
                for memory in memory_group:
                    clusters.append([memory])
        return clusters
    def _consolidate_cluster(
        self,
        cluster: List[Memory]
    ) -> str:
        """Consolidate a cluster of memories into one"""
        self.consolidated_counter += 1
        consolidated_id = f"consolidated_{self.consolidated_counter}_{int(time.time())}"
        # Determine consolidated content
        if len(cluster) > 5:
            # Summarize for large clusters
            consolidated_content = {
                'type': 'summary',
                'count': len(cluster),
                'sample_content': [mem.content for mem in cluster[:3]],
                'common_tags': set.intersection(*[mem.tags for mem in cluster])
            }
        else:
            # Keep all content for small clusters
            consolidated_content = {
                'type': 'aggregated',
                'memories': [mem.content for mem in cluster]
            }
        # Calculate average importance
        avg_importance_value = sum(mem.importance.value for mem in cluster) / len(cluster)
        importance = MemoryImportance(round(avg_importance_value))
        # Create consolidated memory
        consolidated = ConsolidatedMemory(
            consolidated_id=consolidated_id,
            source_memory_ids=[mem.memory_id for mem in cluster],
            consolidated_content=consolidated_content,
            importance=importance,
            consolidation_time=time.time(),
            compression_ratio=1.0 / len(cluster),
            metadata={
                'cluster_size': len(cluster),
                'consolidation_strategy': self.strategy.value
            }
        )
        self.long_term_memories[consolidated_id] = consolidated
        return consolidated_id
    def _store_individual_memory(
        self,
        memory: Memory
    ) -> str:
        """Store individual memory in long-term storage"""
        self.consolidated_counter += 1
        consolidated_id = f"consolidated_{self.consolidated_counter}_{int(time.time())}"
        consolidated = ConsolidatedMemory(
            consolidated_id=consolidated_id,
            source_memory_ids=[memory.memory_id],
            consolidated_content=memory.content,
            importance=memory.importance,
            consolidation_time=time.time(),
            compression_ratio=1.0,
            metadata={
                'original_tags': memory.tags,
                'original_metadata': memory.metadata
            }
        )
        self.long_term_memories[consolidated_id] = consolidated
        return consolidated_id
    def _discard_memories(self, memories: List[Memory]):
        """Discard low-value memories"""
        for memory in memories:
            self.discarded_memories.append(memory.memory_id)
            if memory.memory_id in self.short_term_memories:
                del self.short_term_memories[memory.memory_id]
    def access_memory(self, memory_id: str):
        """Record access to a memory"""
        current_time = time.time()
        # Check short-term
        if memory_id in self.short_term_memories:
            memory = self.short_term_memories[memory_id]
            memory.access_count += 1
            memory.last_accessed = current_time
            return memory.content
        # Check long-term
        if memory_id in self.long_term_memories:
            memory = self.long_term_memories[memory_id]
            memory.access_count += 1
            memory.last_accessed = current_time
            return memory.consolidated_content
        return None
    def get_long_term_memories(
        self,
        importance_filter: Optional[MemoryImportance] = None,
        limit: Optional[int] = None
    ) -> List[ConsolidatedMemory]:
        """Get long-term memories"""
        memories = list(self.long_term_memories.values())
        # Filter by importance
        if importance_filter:
            memories = [
                mem for mem in memories
                if mem.importance.value >= importance_filter.value
            ]
        # Sort by access count (most accessed first)
        memories.sort(key=lambda x: x.access_count, reverse=True)
        if limit:
            memories = memories[:limit]
        return memories
    def get_short_term_status(self) -> Dict[str, Any]:
        """Get status of short-term memory"""
        if not self.short_term_memories:
            return {
                'count': 0,
                'capacity_used': 0.0
            }
        importance_distribution = defaultdict(int)
        total_score = 0.0
        for memory in self.short_term_memories.values():
            importance_distribution[memory.importance.value] += 1
            total_score += memory.consolidation_score
        return {
            'count': len(self.short_term_memories),
            'capacity_used': len(self.short_term_memories) / self.max_short_term_capacity,
            'avg_consolidation_score': total_score / len(self.short_term_memories),
            'importance_distribution': dict(importance_distribution),
            'ready_for_consolidation': sum(
                1 for mem in self.short_term_memories.values()
                if mem.consolidation_score >= self.consolidation_threshold
            )
        }
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        total_consolidations = len(self.consolidation_history)
        if total_consolidations > 0:
            avg_compression = sum(
                m.compression_ratio for m in self.consolidation_history
            ) / total_consolidations
            total_processed = sum(m.memories_processed for m in self.consolidation_history)
            total_consolidated = sum(m.memories_consolidated for m in self.consolidation_history)
            total_discarded = sum(m.memories_discarded for m in self.consolidation_history)
        else:
            avg_compression = 0.0
            total_processed = 0
            total_consolidated = 0
            total_discarded = 0
        return {
            'short_term_count': len(self.short_term_memories),
            'long_term_count': len(self.long_term_memories),
            'discarded_count': len(self.discarded_memories),
            'total_consolidations': total_consolidations,
            'total_memories_processed': total_processed,
            'total_memories_consolidated': total_consolidated,
            'total_memories_discarded': total_discarded,
            'average_compression_ratio': avg_compression,
            'consolidation_strategy': self.strategy.value
        }
def main():
    """Demonstrate memory consolidation"""
    print("=" * 60)
    print("Memory Consolidation Demonstration")
    print("=" * 60)
    consolidator = MemoryConsolidator(
        strategy=ConsolidationStrategy.SEMANTIC_CLUSTERING,
        max_short_term_capacity=20,
        consolidation_threshold=0.5
    )
    print("\n1. Adding Short-Term Memories")
    print("-" * 60)
    # Add various memories with different importance levels
    memories_data = [
        ("User asked about Python", MemoryImportance.MEDIUM, {'python', 'question'}),
        ("Explained list comprehensions", MemoryImportance.HIGH, {'python', 'explanation'}),
        ("User said thank you", MemoryImportance.LOW, {'interaction'}),
        ("Discussed async programming", MemoryImportance.HIGH, {'python', 'async'}),
        ("User reported bug in code", MemoryImportance.CRITICAL, {'error', 'bug'}),
        ("Fixed the bug", MemoryImportance.HIGH, {'error', 'fix'}),
        ("User asked about JavaScript", MemoryImportance.MEDIUM, {'javascript', 'question'}),
        ("Explained closures", MemoryImportance.HIGH, {'javascript', 'explanation'}),
        ("Small talk about weather", MemoryImportance.TRIVIAL, {'chat'}),
        ("User completed tutorial", MemoryImportance.HIGH, {'achievement', 'python'}),
    ]
    memory_ids = []
    for content, importance, tags in memories_data:
        mem_id = consolidator.add_short_term_memory(
            content=content,
            importance=importance,
            tags=tags
        )
        memory_ids.append(mem_id)
        print(f"Added ({importance.name}): {content}")
    # Simulate some accesses
    for _ in range(3):
        consolidator.access_memory(memory_ids[1])  # Frequently access important memory
    consolidator.access_memory(memory_ids[4])  # Access critical memory
    print(f"\nTotal short-term memories: {len(consolidator.short_term_memories)}")
    print("\n" + "=" * 60)
    print("2. Short-Term Memory Status")
    print("=" * 60)
    status = consolidator.get_short_term_status()
    print(f"\nMemory Count: {status['count']}")
    print(f"Capacity Used: {status['capacity_used']:.1%}")
    print(f"Avg Consolidation Score: {status['avg_consolidation_score']:.3f}")
    print(f"Ready for Consolidation: {status['ready_for_consolidation']}")
    print("\nImportance Distribution:")
    for importance_val, count in sorted(status['importance_distribution'].items(), reverse=True):
        importance_name = MemoryImportance(importance_val).name
        print(f"  {importance_name}: {count}")
    print("\n" + "=" * 60)
    print("3. Performing Consolidation")
    print("=" * 60)
    print("\nConsolidating memories...")
    metrics = consolidator.consolidate()
    print(f"\nConsolidation Complete:")
    print(f"  Processed: {metrics.memories_processed} memories")
    print(f"  Consolidated: {metrics.memories_consolidated} memories")
    print(f"  Discarded: {metrics.memories_discarded} memories")
    print(f"  Compression Ratio: {metrics.compression_ratio:.2f}")
    print(f"  Duration: {metrics.consolidation_duration:.4f}s")
    print(f"  Avg Importance: {metrics.average_importance:.2f}")
    print("\n" + "=" * 60)
    print("4. Long-Term Memory Contents")
    print("=" * 60)
    long_term = consolidator.get_long_term_memories()
    print(f"\nLong-term memories: {len(long_term)}")
    for i, memory in enumerate(long_term[:5], 1):
        print(f"\n{i}. Consolidated Memory ID: {memory.consolidated_id}")
        print(f"   Importance: {memory.importance.name}")
        print(f"   Source Memories: {len(memory.source_memory_ids)}")
        print(f"   Compression: {memory.compression_ratio:.2f}")
        print(f"   Content Type: {memory.consolidated_content.get('type', 'individual')}")
        if isinstance(memory.consolidated_content, dict):
            if memory.consolidated_content.get('type') == 'summary':
                print(f"   Cluster Size: {memory.consolidated_content['count']}")
                print(f"   Common Tags: {memory.consolidated_content.get('common_tags', set())}")
            elif memory.consolidated_content.get('type') == 'aggregated':
                print(f"   Memories: {len(memory.consolidated_content['memories'])}")
    print("\n" + "=" * 60)
    print("5. Adding More Memories and Auto-Consolidation")
    print("=" * 60)
    print("\nAdding 15 more memories to trigger auto-consolidation...")
    for i in range(15):
        importance = MemoryImportance.MEDIUM if i % 2 == 0 else MemoryImportance.LOW
        consolidator.add_short_term_memory(
            content=f"Additional memory {i}",
            importance=importance,
            tags={'auto', f'batch_{i // 5}'}
        )
    print(f"Short-term count: {len(consolidator.short_term_memories)}")
    print(f"Long-term count: {len(consolidator.long_term_memories)}")
    print(f"Discarded count: {len(consolidator.discarded_memories)}")
    print("\n" + "=" * 60)
    print("6. Filtering Long-Term Memories")
    print("=" * 60)
    print("\nHigh importance memories only:")
    high_importance = consolidator.get_long_term_memories(
        importance_filter=MemoryImportance.HIGH,
        limit=5
    )
    for memory in high_importance:
        print(f"  - {memory.consolidated_id}: {memory.importance.name}")
        print(f"    Access count: {memory.access_count}")
    print("\n" + "=" * 60)
    print("7. Consolidation History")
    print("=" * 60)
    print(f"\nTotal consolidation cycles: {len(consolidator.consolidation_history)}")
    for i, metrics in enumerate(consolidator.consolidation_history, 1):
        print(f"\nCycle {i}:")
        print(f"  Processed: {metrics.memories_processed}")
        print(f"  Consolidated: {metrics.memories_consolidated}")
        print(f"  Discarded: {metrics.memories_discarded}")
        print(f"  Compression: {metrics.compression_ratio:.2f}")
    print("\n" + "=" * 60)
    print("8. Overall Statistics")
    print("=" * 60)
    stats = consolidator.get_statistics()
    print(f"\nCurrent State:")
    print(f"  Short-term: {stats['short_term_count']} memories")
    print(f"  Long-term: {stats['long_term_count']} memories")
    print(f"  Discarded: {stats['discarded_count']} memories")
    print(f"\nConsolidation Summary:")
    print(f"  Total Cycles: {stats['total_consolidations']}")
    print(f"  Total Processed: {stats['total_memories_processed']}")
    print(f"  Total Consolidated: {stats['total_memories_consolidated']}")
    print(f"  Total Discarded: {stats['total_memories_discarded']}")
    print(f"  Avg Compression: {stats['average_compression_ratio']:.2f}")
    print(f"  Strategy: {stats['consolidation_strategy']}")
    print("\n" + "=" * 60)
    print("Memory Consolidation demonstration complete!")
    print("=" * 60)
if __name__ == "__main__":
    main()
