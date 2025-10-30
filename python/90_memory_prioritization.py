"""
Memory Prioritization Pattern

Enables agents to manage memory efficiently by prioritizing important information,
implementing selective retention, and managing memory decay.

Key Concepts:
- Importance scoring
- Memory decay
- Selective retention
- Working vs long-term memory
- Memory consolidation

Use Cases:
- Long-running agents
- Resource-constrained environments
- Information overload management
- Adaptive memory systems
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import uuid
import math


class MemoryType(Enum):
    """Types of memory."""
    WORKING = "working"  # Short-term, high capacity
    SHORT_TERM = "short_term"  # Temporary storage
    LONG_TERM = "long_term"  # Persistent storage
    EPISODIC = "episodic"  # Event memories
    SEMANTIC = "semantic"  # Fact memories


class ImportanceLevel(Enum):
    """Importance levels for memories."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    TRIVIAL = 1


@dataclass
class MemoryItem:
    """A single memory item."""
    memory_id: str
    content: Any
    memory_type: MemoryType
    importance: float = 0.5  # 0-1 scale
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    emotional_valence: float = 0.0  # -1 to 1 (negative to positive)
    context_tags: List[str] = field(default_factory=list)
    related_memories: List[str] = field(default_factory=list)
    
    # Decay parameters
    decay_rate: float = 0.1  # How quickly importance decays
    consolidation_strength: float = 0.0  # How well consolidated (0-1)
    
    def calculate_current_importance(self) -> float:
        """Calculate current importance with decay."""
        time_since_access = datetime.now() - self.last_accessed
        hours_passed = time_since_access.total_seconds() / 3600
        
        # Apply exponential decay
        decay_factor = math.exp(-self.decay_rate * hours_passed)
        
        # Factor in access frequency (more access = higher importance)
        access_boost = min(0.3, self.access_count * 0.05)
        
        # Factor in consolidation (consolidated memories decay slower)
        consolidation_factor = 1.0 + (self.consolidation_strength * 0.5)
        
        current_importance = (
            self.importance * decay_factor * consolidation_factor + access_boost
        )
        
        return min(1.0, max(0.0, current_importance))
    
    def access(self) -> None:
        """Record an access to this memory."""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class MemoryStats:
    """Statistics about memory usage."""
    total_memories: int
    working_memory_count: int
    short_term_count: int
    long_term_count: int
    average_importance: float
    memory_capacity_used: float  # 0-1
    oldest_memory_age_hours: float
    most_accessed_memory_id: Optional[str]


class PrioritizedMemorySystem:
    """Memory system with importance-based prioritization."""
    
    def __init__(
        self,
        working_memory_capacity: int = 7,  # Miller's Law: 7±2 items
        short_term_capacity: int = 50,
        long_term_capacity: int = 1000
    ):
        self.working_memory_capacity = working_memory_capacity
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity
        
        self.memories: Dict[str, MemoryItem] = {}
        self.working_memory: List[str] = []  # IDs of items in working memory
        
        # Importance thresholds
        self.consolidation_threshold = 0.7  # Move to long-term
        self.retention_threshold = 0.3  # Below this, consider forgetting
    
    def store(
        self,
        content: Any,
        memory_type: MemoryType = MemoryType.WORKING,
        importance: float = 0.5,
        context_tags: Optional[List[str]] = None
    ) -> str:
        """Store a new memory."""
        memory_id = str(uuid.uuid4())
        
        memory = MemoryItem(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            context_tags=context_tags or []
        )
        
        self.memories[memory_id] = memory
        
        # Add to working memory if appropriate
        if memory_type == MemoryType.WORKING:
            self._add_to_working_memory(memory_id)
        
        # Check capacity and perform cleanup if needed
        self._manage_capacity()
        
        return memory_id
    
    def retrieve(
        self,
        memory_id: str
    ) -> Optional[MemoryItem]:
        """Retrieve a memory by ID."""
        memory = self.memories.get(memory_id)
        
        if memory:
            memory.access()
            
            # Move to working memory if accessed
            if memory_id not in self.working_memory:
                self._add_to_working_memory(memory_id)
        
        return memory
    
    def search(
        self,
        query: str,
        context_tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
        limit: int = 10
    ) -> List[MemoryItem]:
        """Search memories by content and context."""
        results = []
        
        for memory in self.memories.values():
            # Calculate relevance score
            relevance = 0.0
            
            # Content matching (simplified)
            if isinstance(memory.content, str) and query.lower() in memory.content.lower():
                relevance += 0.5
            
            # Context tag matching
            if context_tags:
                tag_overlap = len(set(memory.context_tags) & set(context_tags))
                relevance += tag_overlap * 0.2
            
            # Current importance
            current_importance = memory.calculate_current_importance()
            
            if current_importance >= min_importance and relevance > 0:
                results.append((memory, relevance + current_importance))
        
        # Sort by combined score
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Access the retrieved memories
        for memory, _ in results[:limit]:
            memory.access()
        
        return [memory for memory, _ in results[:limit]]
    
    def consolidate_memories(self) -> int:
        """Consolidate important short-term memories to long-term."""
        consolidated_count = 0
        
        for memory in list(self.memories.values()):
            if memory.memory_type == MemoryType.SHORT_TERM:
                current_importance = memory.calculate_current_importance()
                
                if current_importance >= self.consolidation_threshold:
                    # Promote to long-term memory
                    memory.memory_type = MemoryType.LONG_TERM
                    memory.consolidation_strength = current_importance
                    consolidated_count += 1
        
        return consolidated_count
    
    def forget_unimportant(self) -> int:
        """Remove memories below retention threshold."""
        to_forget = []
        
        for memory_id, memory in self.memories.items():
            current_importance = memory.calculate_current_importance()
            
            # Don't forget recently created memories
            age = datetime.now() - memory.created_at
            if age < timedelta(hours=1):
                continue
            
            # Don't forget long-term memories easily
            if memory.memory_type == MemoryType.LONG_TERM:
                threshold = self.retention_threshold * 0.5
            else:
                threshold = self.retention_threshold
            
            if current_importance < threshold:
                to_forget.append(memory_id)
        
        # Remove forgotten memories
        for memory_id in to_forget:
            del self.memories[memory_id]
            if memory_id in self.working_memory:
                self.working_memory.remove(memory_id)
        
        return len(to_forget)
    
    def get_most_important(
        self,
        n: int = 10,
        memory_type: Optional[MemoryType] = None
    ) -> List[MemoryItem]:
        """Get the N most important memories."""
        memories = list(self.memories.values())
        
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        
        # Sort by current importance
        memories.sort(
            key=lambda m: m.calculate_current_importance(),
            reverse=True
        )
        
        return memories[:n]
    
    def get_recent(
        self,
        hours: float = 24.0,
        min_importance: float = 0.0
    ) -> List[MemoryItem]:
        """Get recent memories above importance threshold."""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        recent = [
            m for m in self.memories.values()
            if m.created_at >= cutoff and 
            m.calculate_current_importance() >= min_importance
        ]
        
        recent.sort(key=lambda m: m.created_at, reverse=True)
        return recent
    
    def link_memories(
        self,
        memory_id1: str,
        memory_id2: str
    ) -> bool:
        """Create a link between related memories."""
        if memory_id1 not in self.memories or memory_id2 not in self.memories:
            return False
        
        memory1 = self.memories[memory_id1]
        memory2 = self.memories[memory_id2]
        
        if memory_id2 not in memory1.related_memories:
            memory1.related_memories.append(memory_id2)
        
        if memory_id1 not in memory2.related_memories:
            memory2.related_memories.append(memory_id1)
        
        # Boost importance of linked memories
        memory1.importance = min(1.0, memory1.importance + 0.05)
        memory2.importance = min(1.0, memory2.importance + 0.05)
        
        return True
    
    def get_related_memories(
        self,
        memory_id: str,
        depth: int = 1
    ) -> List[MemoryItem]:
        """Get memories related to a given memory."""
        if memory_id not in self.memories:
            return []
        
        related = []
        visited = set()
        
        def _traverse(mid: str, current_depth: int):
            if current_depth > depth or mid in visited:
                return
            
            visited.add(mid)
            memory = self.memories.get(mid)
            
            if memory:
                if mid != memory_id:
                    related.append(memory)
                
                for related_id in memory.related_memories:
                    _traverse(related_id, current_depth + 1)
        
        _traverse(memory_id, 0)
        return related
    
    def get_statistics(self) -> MemoryStats:
        """Get memory system statistics."""
        if not self.memories:
            return MemoryStats(
                total_memories=0,
                working_memory_count=0,
                short_term_count=0,
                long_term_count=0,
                average_importance=0.0,
                memory_capacity_used=0.0,
                oldest_memory_age_hours=0.0,
                most_accessed_memory_id=None
            )
        
        type_counts = defaultdict(int)
        total_importance = 0.0
        most_accessed = max(
            self.memories.values(),
            key=lambda m: m.access_count
        )
        oldest = min(
            self.memories.values(),
            key=lambda m: m.created_at
        )
        
        for memory in self.memories.values():
            type_counts[memory.memory_type] += 1
            total_importance += memory.calculate_current_importance()
        
        oldest_age = (datetime.now() - oldest.created_at).total_seconds() / 3600
        
        # Calculate capacity usage
        total_capacity = (
            self.working_memory_capacity +
            self.short_term_capacity +
            self.long_term_capacity
        )
        capacity_used = len(self.memories) / total_capacity
        
        return MemoryStats(
            total_memories=len(self.memories),
            working_memory_count=type_counts[MemoryType.WORKING],
            short_term_count=type_counts[MemoryType.SHORT_TERM],
            long_term_count=type_counts[MemoryType.LONG_TERM],
            average_importance=total_importance / len(self.memories),
            memory_capacity_used=capacity_used,
            oldest_memory_age_hours=oldest_age,
            most_accessed_memory_id=most_accessed.memory_id
        )
    
    def _add_to_working_memory(self, memory_id: str) -> None:
        """Add memory to working memory, evicting if necessary."""
        if memory_id in self.working_memory:
            return
        
        # If at capacity, remove least important item
        if len(self.working_memory) >= self.working_memory_capacity:
            least_important = min(
                self.working_memory,
                key=lambda mid: self.memories[mid].calculate_current_importance()
            )
            self.working_memory.remove(least_important)
            
            # Demote to short-term memory
            if least_important in self.memories:
                self.memories[least_important].memory_type = MemoryType.SHORT_TERM
        
        self.working_memory.append(memory_id)
    
    def _manage_capacity(self) -> None:
        """Manage memory capacity by type."""
        # Count memories by type
        type_counts = defaultdict(int)
        for memory in self.memories.values():
            type_counts[memory.memory_type] += 1
        
        # Check short-term capacity
        if type_counts[MemoryType.SHORT_TERM] > self.short_term_capacity:
            self.consolidate_memories()
            self.forget_unimportant()
        
        # Check long-term capacity
        if type_counts[MemoryType.LONG_TERM] > self.long_term_capacity:
            self.forget_unimportant()


def demonstrate_memory_prioritization():
    """Demonstrate memory prioritization pattern."""
    print("=" * 60)
    print("MEMORY PRIORITIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Create memory system
    memory_system = PrioritizedMemorySystem(
        working_memory_capacity=5,
        short_term_capacity=20,
        long_term_capacity=100
    )
    
    # Store various memories with different importance
    print("\n" + "=" * 60)
    print("1. Storing Memories with Different Importance")
    print("=" * 60)
    
    memories = []
    
    # Critical information
    mid1 = memory_system.store(
        "System password: secret123",
        MemoryType.WORKING,
        importance=0.95,
        context_tags=["security", "critical"]
    )
    memories.append(mid1)
    print("Stored CRITICAL: System password")
    
    # Important task
    mid2 = memory_system.store(
        "Complete project report by Friday",
        MemoryType.WORKING,
        importance=0.80,
        context_tags=["task", "deadline"]
    )
    memories.append(mid2)
    print("Stored HIGH: Project deadline")
    
    # Medium importance
    mid3 = memory_system.store(
        "Meeting notes from team sync",
        MemoryType.SHORT_TERM,
        importance=0.50,
        context_tags=["meeting", "notes"]
    )
    memories.append(mid3)
    print("Stored MEDIUM: Meeting notes")
    
    # Low importance
    for i in range(5):
        mid = memory_system.store(
            f"Random fact #{i}",
            MemoryType.SHORT_TERM,
            importance=0.20,
            context_tags=["trivia"]
        )
        memories.append(mid)
    print("Stored LOW: 5 random facts")
    
    # Link related memories
    print("\n" + "=" * 60)
    print("2. Linking Related Memories")
    print("=" * 60)
    
    memory_system.link_memories(mid1, mid2)
    print(f"Linked password memory with deadline memory")
    
    # Access some memories multiple times
    print("\n" + "=" * 60)
    print("3. Accessing Memories (Importance Boost)")
    print("=" * 60)
    
    for _ in range(3):
        memory_system.retrieve(mid1)
    print("Accessed password memory 3 times")
    
    for _ in range(2):
        memory_system.retrieve(mid2)
    print("Accessed deadline memory 2 times")
    
    # Show current statistics
    print("\n" + "=" * 60)
    print("4. Memory System Statistics")
    print("=" * 60)
    
    stats = memory_system.get_statistics()
    print(f"\nTotal memories: {stats.total_memories}")
    print(f"Working memory: {stats.working_memory_count}/{memory_system.working_memory_capacity}")
    print(f"Short-term: {stats.short_term_count}/{memory_system.short_term_capacity}")
    print(f"Long-term: {stats.long_term_count}/{memory_system.long_term_capacity}")
    print(f"Average importance: {stats.average_importance:.2f}")
    print(f"Capacity used: {stats.memory_capacity_used * 100:.1f}%")
    
    # Show most important memories
    print("\n" + "=" * 60)
    print("5. Most Important Memories")
    print("=" * 60)
    
    top_memories = memory_system.get_most_important(n=3)
    for i, memory in enumerate(top_memories, 1):
        importance = memory.calculate_current_importance()
        print(f"\n{i}. Importance: {importance:.2f}")
        print(f"   Content: {memory.content}")
        print(f"   Access count: {memory.access_count}")
        print(f"   Tags: {memory.context_tags}")
    
    # Search memories
    print("\n" + "=" * 60)
    print("6. Memory Search")
    print("=" * 60)
    
    results = memory_system.search(
        "project",
        context_tags=["task"],
        min_importance=0.5
    )
    
    print(f"\nFound {len(results)} memories matching 'project' with task tag:")
    for memory in results:
        print(f"  - {memory.content} (importance: {memory.calculate_current_importance():.2f})")
    
    # Consolidation
    print("\n" + "=" * 60)
    print("7. Memory Consolidation")
    print("=" * 60)
    
    consolidated = memory_system.consolidate_memories()
    print(f"\nConsolidated {consolidated} memories to long-term storage")
    
    # Forgetting
    print("\n" + "=" * 60)
    print("8. Forgetting Unimportant Memories")
    print("=" * 60)
    
    forgotten = memory_system.forget_unimportant()
    print(f"\nForgot {forgotten} low-importance memories")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("9. Final Statistics")
    print("=" * 60)
    
    final_stats = memory_system.get_statistics()
    print(f"\nRemaining memories: {final_stats.total_memories}")
    print(f"Average importance: {final_stats.average_importance:.2f}")
    print(f"Memory preserved: {(1 - forgotten / stats.total_memories) * 100:.1f}%")


if __name__ == "__main__":
    demonstrate_memory_prioritization()
