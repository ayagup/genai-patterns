"""
Pattern 105: Hierarchical Memory

This pattern implements hierarchical memory organization with multi-level
abstraction, enabling efficient storage and retrieval across different
levels of detail.

Use Cases:
- Multi-scale information organization
- Efficient hierarchical retrieval
- Abstraction-based reasoning
- Long-term memory management
- Context-aware access

Key Features:
- Multi-level memory hierarchy
- Automatic abstraction generation
- Level-specific retrieval
- Bottom-up and top-down access
- Memory consolidation
- Adaptive organization

Implementation:
- Pure Python (3.8+) with comprehensive type hints
- Zero external dependencies
- Production-ready error handling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import uuid
import math


class MemoryLevel(Enum):
    """Levels in memory hierarchy."""
    SENSORY = 0      # Raw, detailed memories
    SHORT_TERM = 1   # Working memory
    EPISODIC = 2     # Event memories
    SEMANTIC = 3     # Abstract concepts
    CONSOLIDATED = 4 # Long-term stable


@dataclass
class HierarchicalMemory:
    """A memory node in the hierarchy."""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: MemoryLevel = MemoryLevel.SENSORY
    content: str = ""
    abstraction: str = ""  # Higher-level summary
    
    # Hierarchy relationships
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    importance: float = 0.5  # 0-1
    stability: float = 0.0   # How consolidated (0-1)
    
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def access(self) -> None:
        """Record access to this memory."""
        self.access_count += 1
        self.last_accessed = datetime.now()
        self.importance = min(1.0, self.importance + 0.05)
    
    def age_hours(self) -> float:
        """Get age of memory in hours."""
        return (datetime.now() - self.timestamp).total_seconds() / 3600
    
    def decay_importance(self, decay_rate: float = 0.1) -> None:
        """Decay importance over time."""
        self.importance *= (1 - decay_rate)
        self.importance = max(0.0, self.importance)


@dataclass
class MemoryQuery:
    """Query for hierarchical memory retrieval."""
    query_text: str
    preferred_level: Optional[MemoryLevel] = None
    max_results: int = 5
    min_importance: float = 0.0
    time_range: Optional[Tuple[datetime, datetime]] = None
    include_children: bool = False
    include_parents: bool = False


@dataclass
class MemoryRetrievalResult:
    """Result of memory retrieval."""
    memories: List[HierarchicalMemory]
    relevance_scores: Dict[str, float]
    retrieval_path: List[MemoryLevel]  # Levels searched
    query: MemoryQuery


class AbstractionGenerator:
    """
    Generates abstractions from lower-level memories.
    
    Creates higher-level summaries and semantic representations.
    """
    
    def __init__(self):
        pass
    
    def generate_abstraction(self, memories: List[HierarchicalMemory]) -> str:
        """Generate abstraction from multiple memories."""
        if not memories:
            return ""
        
        if len(memories) == 1:
            return self._summarize_single(memories[0])
        
        # Extract key themes
        all_content = " ".join(m.content for m in memories)
        
        # Extract common words (simplified)
        words = all_content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Skip short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords = [word for word, _ in top_words]
        
        # Create abstract summary
        abstraction = f"Cluster of {len(memories)} memories about: {', '.join(keywords)}"
        
        # Add time context
        if memories:
            earliest = min(m.timestamp for m in memories)
            latest = max(m.timestamp for m in memories)
            duration = (latest - earliest).total_seconds() / 3600
            
            if duration < 1:
                abstraction += " (recent activity)"
            elif duration < 24:
                abstraction += " (same day)"
            else:
                abstraction += f" (span: {duration/24:.1f} days)"
        
        return abstraction
    
    def _summarize_single(self, memory: HierarchicalMemory) -> str:
        """Summarize a single memory."""
        # Take first sentence or first N words
        sentences = memory.content.split(". ")
        if sentences:
            return sentences[0]
        
        words = memory.content.split()
        if len(words) > 20:
            return " ".join(words[:20]) + "..."
        
        return memory.content


class MemoryConsolidator:
    """
    Consolidates memories into higher levels.
    
    Process:
    1. Group related lower-level memories
    2. Generate abstraction
    3. Create higher-level memory node
    4. Link relationships
    """
    
    def __init__(self, abstraction_generator: AbstractionGenerator):
        self.abstraction_generator = abstraction_generator
    
    def consolidate_memories(self, memories: List[HierarchicalMemory],
                            target_level: MemoryLevel) -> HierarchicalMemory:
        """Consolidate multiple memories into higher level."""
        if not memories:
            raise ValueError("Cannot consolidate empty memory list")
        
        # Generate abstraction
        abstraction = self.abstraction_generator.generate_abstraction(memories)
        
        # Calculate consolidated importance
        avg_importance = sum(m.importance for m in memories) / len(memories)
        
        # Create consolidated memory
        consolidated = HierarchicalMemory(
            level=target_level,
            content=abstraction,
            abstraction=abstraction,
            children_ids=[m.memory_id for m in memories],
            importance=avg_importance,
            stability=0.7,  # Start with high stability
            tags=set().union(*(m.tags for m in memories))
        )
        
        # Link children to parent
        for mem in memories:
            mem.parent_id = consolidated.memory_id
        
        return consolidated
    
    def should_consolidate(self, memories: List[HierarchicalMemory],
                          min_cluster_size: int = 3,
                          min_age_hours: float = 24.0) -> bool:
        """Determine if memories should be consolidated."""
        if len(memories) < min_cluster_size:
            return False
        
        # Check age
        avg_age = sum(m.age_hours() for m in memories) / len(memories)
        if avg_age < min_age_hours:
            return False
        
        # Check similarity (simplified - check tag overlap)
        if not memories:
            return False
        
        common_tags = set(memories[0].tags)
        for mem in memories[1:]:
            common_tags &= mem.tags
        
        # Require some common tags
        return len(common_tags) > 0


class HierarchicalMemoryIndex:
    """
    Index for efficient hierarchical retrieval.
    
    Maintains:
    - Level-specific indices
    - Tag indices
    - Temporal indices
    - Importance rankings
    """
    
    def __init__(self):
        # Level-based index
        self.level_index: Dict[MemoryLevel, Set[str]] = {
            level: set() for level in MemoryLevel
        }
        
        # Tag index
        self.tag_index: Dict[str, Set[str]] = {}
        
        # Importance index (sorted lists per level)
        self.importance_index: Dict[MemoryLevel, List[str]] = {
            level: [] for level in MemoryLevel
        }
    
    def add_memory(self, memory: HierarchicalMemory) -> None:
        """Add memory to indices."""
        # Add to level index
        self.level_index[memory.level].add(memory.memory_id)
        
        # Add to tag index
        for tag in memory.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(memory.memory_id)
        
        # Add to importance index
        self.importance_index[memory.level].append(memory.memory_id)
        # Keep sorted by importance (would need memory ref for accurate sort)
    
    def remove_memory(self, memory: HierarchicalMemory) -> None:
        """Remove memory from indices."""
        self.level_index[memory.level].discard(memory.memory_id)
        
        for tag in memory.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(memory.memory_id)
        
        if memory.memory_id in self.importance_index[memory.level]:
            self.importance_index[memory.level].remove(memory.memory_id)
    
    def find_by_tag(self, tag: str, level: Optional[MemoryLevel] = None) -> Set[str]:
        """Find memories by tag, optionally filtered by level."""
        memory_ids = self.tag_index.get(tag, set())
        
        if level is not None:
            memory_ids &= self.level_index[level]
        
        return memory_ids
    
    def find_by_level(self, level: MemoryLevel) -> Set[str]:
        """Find all memories at a specific level."""
        return self.level_index[level].copy()


class HierarchicalMemoryAgent:
    """
    Agent managing hierarchical memory organization.
    
    Capabilities:
    - Multi-level memory storage
    - Automatic consolidation
    - Hierarchical retrieval
    - Memory promotion/demotion
    - Adaptive organization
    """
    
    def __init__(self):
        self.memories: Dict[str, HierarchicalMemory] = {}
        self.index = HierarchicalMemoryIndex()
        self.abstraction_generator = AbstractionGenerator()
        self.consolidator = MemoryConsolidator(self.abstraction_generator)
        
        # Statistics
        self.memories_added = 0
        self.consolidations = 0
        self.retrievals = 0
    
    def add_memory(self, content: str, level: MemoryLevel = MemoryLevel.SENSORY,
                  importance: float = 0.5, tags: Optional[Set[str]] = None) -> HierarchicalMemory:
        """Add a new memory at specified level."""
        memory = HierarchicalMemory(
            level=level,
            content=content,
            importance=importance,
            tags=tags or set()
        )
        
        self.memories[memory.memory_id] = memory
        self.index.add_memory(memory)
        self.memories_added += 1
        
        # Try automatic consolidation
        self._try_auto_consolidate(memory)
        
        return memory
    
    def retrieve(self, query: MemoryQuery) -> MemoryRetrievalResult:
        """Retrieve memories based on query."""
        self.retrievals += 1
        
        # Determine search strategy
        if query.preferred_level:
            levels_to_search = [query.preferred_level]
        else:
            # Search from abstract to detailed
            levels_to_search = [
                MemoryLevel.CONSOLIDATED,
                MemoryLevel.SEMANTIC,
                MemoryLevel.EPISODIC,
                MemoryLevel.SHORT_TERM,
                MemoryLevel.SENSORY
            ]
        
        results = []
        relevance_scores = {}
        
        for level in levels_to_search:
            level_memories = self._search_level(query, level)
            results.extend(level_memories)
            
            # Score relevance
            for mem in level_memories:
                score = self._compute_relevance(mem, query)
                relevance_scores[mem.memory_id] = score
            
            # Stop if we have enough results
            if len(results) >= query.max_results:
                break
        
        # Sort by relevance
        results.sort(key=lambda m: relevance_scores.get(m.memory_id, 0), reverse=True)
        results = results[:query.max_results]
        
        # Update access counts
        for mem in results:
            mem.access()
        
        return MemoryRetrievalResult(
            memories=results,
            relevance_scores=relevance_scores,
            retrieval_path=levels_to_search,
            query=query
        )
    
    def _search_level(self, query: MemoryQuery, level: MemoryLevel) -> List[HierarchicalMemory]:
        """Search memories at a specific level."""
        candidate_ids = self.index.find_by_level(level)
        
        # Filter by query
        matches = []
        query_lower = query.query_text.lower()
        
        for mem_id in candidate_ids:
            mem = self.memories.get(mem_id)
            if not mem:
                continue
            
            # Check importance filter
            if mem.importance < query.min_importance:
                continue
            
            # Check time range
            if query.time_range:
                start, end = query.time_range
                if not (start <= mem.timestamp <= end):
                    continue
            
            # Check text match
            if query_lower in mem.content.lower() or query_lower in mem.abstraction.lower():
                matches.append(mem)
            else:
                # Check tag match
                query_words = set(query_lower.split())
                if query_words & mem.tags:
                    matches.append(mem)
        
        return matches
    
    def _compute_relevance(self, memory: HierarchicalMemory, query: MemoryQuery) -> float:
        """Compute relevance score for memory."""
        score = 0.0
        query_lower = query.query_text.lower()
        
        # Text match
        if query_lower in memory.content.lower():
            score += 1.0
        if query_lower in memory.abstraction.lower():
            score += 0.5
        
        # Tag match
        query_words = set(query_lower.split())
        tag_overlap = query_words & memory.tags
        score += len(tag_overlap) * 0.3
        
        # Importance factor
        score *= (0.5 + memory.importance)
        
        # Recency factor
        age_days = memory.age_hours() / 24
        recency_factor = 1.0 / (1.0 + age_days * 0.1)
        score *= recency_factor
        
        return score
    
    def _try_auto_consolidate(self, new_memory: HierarchicalMemory) -> None:
        """Try to automatically consolidate related memories."""
        # Find related memories at same level
        related = []
        
        for mem_id in self.index.find_by_level(new_memory.level):
            mem = self.memories.get(mem_id)
            if mem and mem.memory_id != new_memory.memory_id:
                # Check if related (common tags)
                if mem.tags & new_memory.tags:
                    related.append(mem)
        
        related.append(new_memory)
        
        # Check if should consolidate
        if self.consolidator.should_consolidate(related, min_cluster_size=3, min_age_hours=0.1):
            self.consolidate(related)
    
    def consolidate(self, memories: List[HierarchicalMemory]) -> HierarchicalMemory:
        """Manually consolidate memories to higher level."""
        if not memories:
            raise ValueError("No memories to consolidate")
        
        # Determine target level
        current_level = memories[0].level
        target_level_value = min(current_level.value + 1, MemoryLevel.CONSOLIDATED.value)
        target_level = MemoryLevel(target_level_value)
        
        # Consolidate
        consolidated = self.consolidator.consolidate_memories(memories, target_level)
        
        # Add to storage
        self.memories[consolidated.memory_id] = consolidated
        self.index.add_memory(consolidated)
        
        self.consolidations += 1
        
        return consolidated
    
    def get_memory_tree(self, root_id: str, max_depth: int = 3) -> Dict[str, Any]:
        """Get memory and its children as a tree structure."""
        root = self.memories.get(root_id)
        if not root:
            return {}
        
        def build_tree(mem: HierarchicalMemory, depth: int) -> Dict[str, Any]:
            if depth >= max_depth:
                return {
                    "id": mem.memory_id,
                    "level": mem.level.name,
                    "content": mem.content[:50] + "...",
                    "children": []
                }
            
            children = []
            for child_id in mem.children_ids:
                child = self.memories.get(child_id)
                if child:
                    children.append(build_tree(child, depth + 1))
            
            return {
                "id": mem.memory_id,
                "level": mem.level.name,
                "content": mem.content[:100],
                "abstraction": mem.abstraction[:100],
                "importance": mem.importance,
                "children": children
            }
        
        return build_tree(root, 0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        level_counts = {level.name: 0 for level in MemoryLevel}
        total_importance = 0.0
        
        for mem in self.memories.values():
            level_counts[mem.level.name] += 1
            total_importance += mem.importance
        
        return {
            "total_memories": len(self.memories),
            "memories_by_level": level_counts,
            "average_importance": (
                total_importance / len(self.memories) 
                if self.memories else 0.0
            ),
            "memories_added": self.memories_added,
            "consolidations_performed": self.consolidations,
            "total_retrievals": self.retrievals
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_hierarchical_memory():
    """Demonstrate hierarchical memory organization."""
    
    print("=" * 70)
    print("HIERARCHICAL MEMORY DEMONSTRATION")
    print("=" * 70)
    
    print("\n1. SETUP")
    print("-" * 70)
    
    agent = HierarchicalMemoryAgent()
    print("   Hierarchical memory agent created")
    
    print("\n2. ADDING SENSORY MEMORIES")
    print("-" * 70)
    print("   Adding detailed, low-level memories...")
    
    sensory_memories = [
        ("Saw red car pass by on Main Street", {"car", "red", "street"}),
        ("Heard loud noise from construction site", {"noise", "construction"}),
        ("Red traffic light at intersection", {"traffic", "red", "intersection"}),
        ("Smelled fresh coffee from cafe", {"coffee", "smell", "cafe"}),
        ("Red stop sign at corner", {"red", "sign", "corner"}),
    ]
    
    for content, tags in sensory_memories:
        mem = agent.add_memory(content, MemoryLevel.SENSORY, 0.3, tags)
        print(f"     Added: {mem.memory_id} - {content[:40]}")
    
    print("\n3. ADDING SHORT-TERM MEMORIES")
    print("-" * 70)
    print("   Adding working memory items...")
    
    short_term = [
        ("Need to remember to buy groceries after work", {"task", "groceries"}),
        ("Meeting scheduled for 3pm today", {"meeting", "schedule"}),
        ("Phone number: 555-0123", {"contact", "phone"}),
    ]
    
    for content, tags in short_term:
        mem = agent.add_memory(content, MemoryLevel.SHORT_TERM, 0.6, tags)
        print(f"     Added: {mem.memory_id} - {content}")
    
    print("\n4. ADDING EPISODIC MEMORIES")
    print("-" * 70)
    print("   Adding event memories...")
    
    episodic = [
        ("Yesterday's team meeting discussed project timeline and deliverables", 
         {"meeting", "project", "team"}),
        ("Lunch at Italian restaurant with colleagues last week", 
         {"lunch", "restaurant", "colleagues"}),
        ("Completed code review for feature implementation", 
         {"code", "review", "work"}),
    ]
    
    for content, tags in episodic:
        mem = agent.add_memory(content, MemoryLevel.EPISODIC, 0.7, tags)
        print(f"     Added: {mem.memory_id} - {content[:50]}...")
    
    print("\n5. MANUAL CONSOLIDATION")
    print("-" * 70)
    print("   Consolidating related memories...")
    
    # Find memories about "red" things
    red_tag_ids = agent.index.find_by_tag("red", MemoryLevel.SENSORY)
    red_memories = [agent.memories[mid] for mid in red_tag_ids]
    
    if len(red_memories) >= 2:
        print(f"   Found {len(red_memories)} memories with 'red' tag")
        consolidated = agent.consolidate(red_memories)
        print(f"\n   Created consolidated memory:")
        print(f"     ID: {consolidated.memory_id}")
        print(f"     Level: {consolidated.level.name}")
        print(f"     Content: {consolidated.content}")
        print(f"     Children: {len(consolidated.children_ids)}")
    
    print("\n6. HIERARCHICAL RETRIEVAL")
    print("-" * 70)
    print("   Querying memory system...")
    
    # Query at different levels
    queries = [
        ("meeting", MemoryLevel.EPISODIC, "Episodic level"),
        ("red", None, "Any level"),
        ("project", None, "Any level"),
    ]
    
    for query_text, level, description in queries:
        print(f"\n   Query: '{query_text}' ({description})")
        
        query = MemoryQuery(
            query_text=query_text,
            preferred_level=level,
            max_results=3
        )
        
        result = agent.retrieve(query)
        
        print(f"   Found {len(result.memories)} memories:")
        for mem in result.memories:
            score = result.relevance_scores.get(mem.memory_id, 0)
            print(f"     - [{mem.level.name}] {mem.content[:60]}...")
            print(f"       Relevance: {score:.2f}, Importance: {mem.importance:.2f}")
    
    print("\n7. MEMORY TREE VISUALIZATION")
    print("-" * 70)
    
    # Find a consolidated memory
    consolidated_mems = [m for m in agent.memories.values() 
                        if m.level.value > MemoryLevel.SENSORY.value and m.children_ids]
    
    if consolidated_mems:
        tree_mem = consolidated_mems[0]
        print(f"   Memory tree for: {tree_mem.memory_id}")
        
        tree = agent.get_memory_tree(tree_mem.memory_id)
        
        def print_tree(node: Dict[str, Any], indent: int = 0):
            prefix = "  " * indent
            print(f"{prefix}• [{node['level']}] {node['content']}")
            for child in node.get('children', []):
                print_tree(child, indent + 1)
        
        print_tree(tree)
    
    print("\n8. STATISTICS")
    print("-" * 70)
    
    stats = agent.get_statistics()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Memories by level:")
    for level, count in stats['memories_by_level'].items():
        if count > 0:
            print(f"     {level}: {count}")
    print(f"   Average importance: {stats['average_importance']:.2f}")
    print(f"   Memories added: {stats['memories_added']}")
    print(f"   Consolidations: {stats['consolidations_performed']}")
    print(f"   Total retrievals: {stats['total_retrievals']}")
    
    print("\n9. TIME-BASED DECAY")
    print("-" * 70)
    print("   Simulating importance decay...")
    
    before_importance = sum(m.importance for m in agent.memories.values())
    
    for mem in agent.memories.values():
        mem.decay_importance(decay_rate=0.05)
    
    after_importance = sum(m.importance for m in agent.memories.values())
    
    print(f"   Total importance before decay: {before_importance:.2f}")
    print(f"   Total importance after decay: {after_importance:.2f}")
    print(f"   Importance lost: {before_importance - after_importance:.2f}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("1. Multi-level memory hierarchy (Sensory → Consolidated)")
    print("2. Automatic memory organization by level")
    print("3. Memory consolidation (bottom-up)")
    print("4. Hierarchical retrieval with level preferences")
    print("5. Relevance scoring across levels")
    print("6. Memory tree visualization")
    print("7. Importance-based filtering")
    print("8. Time-based memory decay")
    print("9. Tag-based indexing and search")


if __name__ == "__main__":
    demonstrate_hierarchical_memory()
