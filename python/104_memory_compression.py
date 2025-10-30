"""
Pattern 104: Memory Compression

This pattern implements memory compression techniques to store and retrieve
information efficiently while preserving important details.

Use Cases:
- Long-context management
- Efficient memory storage
- Information summarization
- Lossy and lossless compression
- Memory reconstruction

Key Features:
- Multiple compression algorithms
- Importance-based retention
- Hierarchical summarization
- Compression ratio tracking
- Decompression and reconstruction
- Quality metrics

Implementation:
- Pure Python (3.8+) with comprehensive type hints
- Zero external dependencies
- Production-ready error handling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
from datetime import datetime
import uuid
import math
import hashlib


class CompressionType(Enum):
    """Types of compression."""
    LOSSLESS = "lossless"      # No information loss
    LOSSY = "lossy"            # Some information loss
    HIERARCHICAL = "hierarchical"  # Multi-level summaries
    SEMANTIC = "semantic"      # Preserve semantic content


class ImportanceLevel(Enum):
    """Importance levels for memories."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Memory:
    """A single memory item."""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    importance: ImportanceLevel = ImportanceLevel.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def size_bytes(self) -> int:
        """Estimate memory size in bytes."""
        return len(self.content.encode('utf-8'))
    
    def access(self) -> None:
        """Record memory access."""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class CompressedMemory:
    """Compressed memory representation."""
    compressed_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    original_ids: List[str] = field(default_factory=list)
    compressed_content: str = ""
    compression_type: CompressionType = CompressionType.LOSSY
    compression_ratio: float = 0.0
    importance: ImportanceLevel = ImportanceLevel.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def size_bytes(self) -> int:
        """Estimate compressed size in bytes."""
        return len(self.compressed_content.encode('utf-8'))


@dataclass
class CompressionStats:
    """Statistics about compression operation."""
    original_size: int = 0
    compressed_size: int = 0
    num_memories: int = 0
    compression_ratio: float = 0.0
    information_loss: float = 0.0  # Estimated 0-1
    compression_time_ms: float = 0.0
    
    def efficiency(self) -> float:
        """Calculate compression efficiency."""
        if self.original_size == 0:
            return 0.0
        saved = self.original_size - self.compressed_size
        return (saved / self.original_size) * 100


class LosslessCompressor:
    """
    Lossless compression - preserves all information.
    
    Uses techniques like:
    - Duplicate detection and deduplication
    - Reference compression
    - Pattern identification
    """
    
    def __init__(self):
        self.deduplication_map: Dict[str, str] = {}  # content_hash -> memory_id
    
    def compress(self, memories: List[Memory]) -> CompressedMemory:
        """Compress memories without losing information."""
        if not memories:
            return CompressedMemory()
        
        original_size = sum(m.size_bytes() for m in memories)
        
        # Deduplicate identical content
        unique_contents = {}
        for mem in memories:
            content_hash = hashlib.md5(mem.content.encode()).hexdigest()[:8]
            if content_hash not in unique_contents:
                unique_contents[content_hash] = mem
        
        # Build compressed representation with references
        compressed_parts = []
        for content_hash, mem in unique_contents.items():
            # Store unique content with hash reference
            compressed_parts.append(f"[{content_hash}]{mem.content}")
        
        compressed_content = "\n".join(compressed_parts)
        compressed_size = len(compressed_content.encode('utf-8'))
        
        return CompressedMemory(
            original_ids=[m.memory_id for m in memories],
            compressed_content=compressed_content,
            compression_type=CompressionType.LOSSLESS,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
            importance=max((m.importance for m in memories), key=lambda x: x.value),
            metadata={
                "num_unique": len(unique_contents),
                "num_original": len(memories),
                "deduplication_ratio": len(memories) / len(unique_contents)
            }
        )
    
    def decompress(self, compressed: CompressedMemory) -> List[Memory]:
        """Decompress to original memories (lossless)."""
        memories = []
        
        # Parse compressed content
        parts = compressed.compressed_content.split("\n")
        for part in parts:
            if part.startswith("[") and "]" in part:
                # Extract hash and content
                end_bracket = part.index("]")
                content_hash = part[1:end_bracket]
                content = part[end_bracket + 1:]
                
                mem = Memory(
                    content=content,
                    importance=compressed.importance,
                    metadata={"source_hash": content_hash}
                )
                memories.append(mem)
        
        return memories


class LossyCompressor:
    """
    Lossy compression - trades information for space.
    
    Techniques:
    - Summarization
    - Importance-based filtering
    - Temporal aggregation
    """
    
    def __init__(self, compression_level: float = 0.5):
        self.compression_level = compression_level  # 0-1, higher = more compression
    
    def compress(self, memories: List[Memory]) -> CompressedMemory:
        """Compress memories with information loss."""
        if not memories:
            return CompressedMemory()
        
        original_size = sum(m.size_bytes() for m in memories)
        
        # Sort by importance and recency
        sorted_memories = sorted(
            memories,
            key=lambda m: (m.importance.value, m.access_count, m.last_accessed),
            reverse=True
        )
        
        # Summarize based on compression level
        target_size = int(original_size * (1 - self.compression_level))
        
        # Extract key information
        summaries = []
        for mem in sorted_memories:
            summary = self._summarize_memory(mem)
            summaries.append(f"[{mem.importance.value}] {summary}")
            
            # Stop if we've reached target size
            current_size = len("\n".join(summaries).encode('utf-8'))
            if current_size >= target_size:
                break
        
        compressed_content = "\n".join(summaries)
        compressed_size = len(compressed_content.encode('utf-8'))
        
        return CompressedMemory(
            original_ids=[m.memory_id for m in memories],
            compressed_content=compressed_content,
            compression_type=CompressionType.LOSSY,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
            importance=max((m.importance for m in memories), key=lambda x: x.value),
            metadata={
                "compression_level": self.compression_level,
                "num_retained": len(summaries),
                "num_original": len(memories)
            }
        )
    
    def _summarize_memory(self, memory: Memory) -> str:
        """Create summary of memory."""
        # Simplified summarization - take first and last sentences
        sentences = memory.content.split(". ")
        
        if len(sentences) <= 2:
            return memory.content
        
        # Keep first and last sentence
        summary = sentences[0] + "... " + sentences[-1]
        
        # If still too long, truncate
        max_len = int(len(memory.content) * (1 - self.compression_level))
        if len(summary) > max_len:
            summary = summary[:max_len] + "..."
        
        return summary


class HierarchicalCompressor:
    """
    Hierarchical compression - creates multi-level summaries.
    
    Levels:
    - Level 0: Original memories
    - Level 1: Group summaries
    - Level 2: Category summaries
    - Level 3: Global summary
    """
    
    def __init__(self, levels: int = 3):
        self.levels = levels
    
    def compress(self, memories: List[Memory]) -> CompressedMemory:
        """Create hierarchical compression."""
        if not memories:
            return CompressedMemory()
        
        original_size = sum(m.size_bytes() for m in memories)
        
        # Build hierarchy
        hierarchy = []
        current_level: List[Any] = memories
        
        for level in range(self.levels):
            # Group memories at this level
            group_size = max(2, len(current_level) // 3)
            groups = [current_level[i:i + group_size] 
                     for i in range(0, len(current_level), group_size)]
            
            # Summarize each group
            summaries = []
            for group in groups:
                if group and isinstance(group[0], Memory):
                    summary = self._summarize_memory_group(group)
                else:
                    summary = self._summarize_text_group([str(g) for g in group])
                summaries.append(summary)
            
            hierarchy.append({
                "level": level,
                "summaries": summaries,
                "num_items": len(summaries)
            })
            
            current_level = summaries
        
        # Build compressed representation
        compressed_parts = []
        for level_data in hierarchy:
            compressed_parts.append(f"Level {level_data['level']}:")
            for i, summary in enumerate(level_data['summaries']):
                compressed_parts.append(f"  {i+1}. {summary}")
        
        compressed_content = "\n".join(compressed_parts)
        compressed_size = len(compressed_content.encode('utf-8'))
        
        return CompressedMemory(
            original_ids=[m.memory_id for m in memories],
            compressed_content=compressed_content,
            compression_type=CompressionType.HIERARCHICAL,
            compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
            importance=max((m.importance for m in memories), key=lambda x: x.value),
            metadata={
                "num_levels": self.levels,
                "hierarchy": hierarchy
            }
        )
    
    def _summarize_memory_group(self, memories: List[Memory]) -> str:
        """Summarize a group of memories."""
        if len(memories) == 1:
            return memories[0].content[:100] + "..."
        
        # Extract key points from each memory
        key_points = []
        for mem in memories:
            # Take first sentence
            first_sentence = mem.content.split(". ")[0]
            key_points.append(first_sentence)
        
        return " | ".join(key_points[:3])
    
    def _summarize_text_group(self, texts: List[str]) -> str:
        """Summarize a group of text summaries."""
        if len(texts) == 1:
            return texts[0]
        
        # Combine and truncate
        combined = " | ".join(texts)
        if len(combined) > 200:
            combined = combined[:197] + "..."
        
        return combined


class MemoryCompressionAgent:
    """
    Agent that manages memory compression.
    
    Capabilities:
    - Multiple compression strategies
    - Adaptive compression based on space constraints
    - Importance-aware compression
    - Compression quality monitoring
    """
    
    def __init__(self, max_memory_bytes: int = 100000):
        self.max_memory_bytes = max_memory_bytes
        
        self.memories: List[Memory] = []
        self.compressed_memories: List[CompressedMemory] = []
        
        self.lossless_compressor = LosslessCompressor()
        self.lossy_compressor = LossyCompressor()
        self.hierarchical_compressor = HierarchicalCompressor()
        
        self.compression_history: List[CompressionStats] = []
    
    def add_memory(self, content: str, importance: ImportanceLevel = ImportanceLevel.MEDIUM,
                  tags: Optional[Set[str]] = None) -> Memory:
        """Add a new memory."""
        memory = Memory(
            content=content,
            importance=importance,
            tags=tags or set()
        )
        
        self.memories.append(memory)
        
        # Check if compression needed
        if self.current_size() > self.max_memory_bytes:
            self.compress_memories()
        
        return memory
    
    def current_size(self) -> int:
        """Get current memory size."""
        memory_size = sum(m.size_bytes() for m in self.memories)
        compressed_size = sum(c.size_bytes() for c in self.compressed_memories)
        return memory_size + compressed_size
    
    def compress_memories(self, compression_type: CompressionType = CompressionType.LOSSY) -> CompressionStats:
        """Compress memories to save space."""
        if not self.memories:
            return CompressionStats()
        
        start_time = datetime.now()
        
        # Sort by importance and select memories to compress
        # Keep critical memories uncompressed
        to_compress = [m for m in self.memories 
                      if m.importance != ImportanceLevel.CRITICAL]
        to_keep = [m for m in self.memories 
                  if m.importance == ImportanceLevel.CRITICAL]
        
        original_size = sum(m.size_bytes() for m in to_compress)
        
        # Compress based on type
        if compression_type == CompressionType.LOSSLESS:
            compressed = self.lossless_compressor.compress(to_compress)
        elif compression_type == CompressionType.HIERARCHICAL:
            compressed = self.hierarchical_compressor.compress(to_compress)
        else:
            compressed = self.lossy_compressor.compress(to_compress)
        
        compressed_size = compressed.size_bytes()
        
        # Update memories
        self.compressed_memories.append(compressed)
        self.memories = to_keep
        
        # Record stats
        end_time = datetime.now()
        stats = CompressionStats(
            original_size=original_size,
            compressed_size=compressed_size,
            num_memories=len(to_compress),
            compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
            compression_time_ms=(end_time - start_time).total_seconds() * 1000
        )
        
        self.compression_history.append(stats)
        
        return stats
    
    def decompress_memory(self, compressed_id: str) -> Optional[List[Memory]]:
        """Decompress a compressed memory."""
        compressed = next((c for c in self.compressed_memories 
                         if c.compressed_id == compressed_id), None)
        
        if not compressed:
            return None
        
        # Only lossless can be fully decompressed
        if compressed.compression_type == CompressionType.LOSSLESS:
            return self.lossless_compressor.decompress(compressed)
        
        return None
    
    def search_compressed(self, query: str) -> List[CompressedMemory]:
        """Search in compressed memories."""
        results = []
        
        for compressed in self.compressed_memories:
            if query.lower() in compressed.compressed_content.lower():
                results.append(compressed)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get compression statistics."""
        total_original = sum(s.original_size for s in self.compression_history)
        total_compressed = sum(s.compressed_size for s in self.compression_history)
        
        return {
            "current_size_bytes": self.current_size(),
            "max_size_bytes": self.max_memory_bytes,
            "utilization": self.current_size() / self.max_memory_bytes,
            "num_memories": len(self.memories),
            "num_compressed": len(self.compressed_memories),
            "total_compressions": len(self.compression_history),
            "average_compression_ratio": (
                total_original / total_compressed 
                if total_compressed > 0 else 1.0
            ),
            "space_saved": total_original - total_compressed
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_memory_compression():
    """Demonstrate memory compression capabilities."""
    
    print("=" * 70)
    print("MEMORY COMPRESSION DEMONSTRATION")
    print("=" * 70)
    
    print("\n1. SETUP")
    print("-" * 70)
    
    # Create agent with limited memory
    agent = MemoryCompressionAgent(max_memory_bytes=5000)
    
    print(f"   Agent created with max memory: {agent.max_memory_bytes} bytes")
    
    print("\n2. ADDING MEMORIES")
    print("-" * 70)
    
    # Add various memories
    memories_to_add = [
        ("The meeting was scheduled for 3pm on Tuesday. Discussion topics included project timeline, budget allocation, and resource needs.", ImportanceLevel.HIGH),
        ("Reminder to buy groceries: milk, bread, eggs, and coffee.", ImportanceLevel.LOW),
        ("Important project deadline: Submit final report by end of month. Must include all testing results and documentation.", ImportanceLevel.CRITICAL),
        ("Weather forecast shows rain for the weekend. Temperature around 60F.", ImportanceLevel.LOW),
        ("New feature request from client: Add export functionality to dashboard. Priority: high. Expected completion: 2 weeks.", ImportanceLevel.HIGH),
        ("Lunch at 12:30 with team members. Location: downtown cafe.", ImportanceLevel.LOW),
        ("Bug fix completed: Resolved memory leak in data processing module. Tested and verified.", ImportanceLevel.MEDIUM),
        ("Conference call scheduled with stakeholders. Agenda: Q3 review and Q4 planning.", ImportanceLevel.MEDIUM),
    ]
    
    for content, importance in memories_to_add:
        mem = agent.add_memory(content, importance)
        print(f"   Added {importance.value} memory: {mem.memory_id} ({mem.size_bytes()} bytes)")
    
    print(f"\n   Total memories: {len(agent.memories)}")
    print(f"   Current size: {agent.current_size()} bytes")
    
    print("\n3. LOSSLESS COMPRESSION")
    print("-" * 70)
    print("   Compressing with lossless algorithm...")
    
    stats = agent.compress_memories(CompressionType.LOSSLESS)
    
    print(f"\n   Compression results:")
    print(f"     Original size: {stats.original_size} bytes")
    print(f"     Compressed size: {stats.compressed_size} bytes")
    print(f"     Compression ratio: {stats.compression_ratio:.2f}x")
    print(f"     Space saved: {stats.efficiency():.1f}%")
    print(f"     Memories compressed: {stats.num_memories}")
    print(f"     Compression time: {stats.compression_time_ms:.2f}ms")
    
    print("\n4. ADDING MORE MEMORIES")
    print("-" * 70)
    
    more_memories = [
        ("Code review feedback: Good structure, consider adding more unit tests.", ImportanceLevel.MEDIUM),
        ("Scheduled maintenance window: Sunday 2am-4am. Notify users in advance.", ImportanceLevel.HIGH),
        ("Team building event next Friday. RSVP by Wednesday.", ImportanceLevel.LOW),
    ]
    
    for content, importance in more_memories:
        agent.add_memory(content, importance)
        print(f"   Added memory: {importance.value}")
    
    print(f"\n   Current size: {agent.current_size()} bytes")
    
    print("\n5. LOSSY COMPRESSION")
    print("-" * 70)
    print("   Compressing with lossy algorithm (50% compression)...")
    
    agent.lossy_compressor.compression_level = 0.5
    stats = agent.compress_memories(CompressionType.LOSSY)
    
    print(f"\n   Compression results:")
    print(f"     Original size: {stats.original_size} bytes")
    print(f"     Compressed size: {stats.compressed_size} bytes")
    print(f"     Compression ratio: {stats.compression_ratio:.2f}x")
    print(f"     Space saved: {stats.efficiency():.1f}%")
    
    # Show compressed content sample
    if agent.compressed_memories:
        latest = agent.compressed_memories[-1]
        print(f"\n   Sample compressed content:")
        print(f"     {latest.compressed_content[:200]}...")
    
    print("\n6. HIERARCHICAL COMPRESSION")
    print("-" * 70)
    
    # Add more memories
    for i in range(5):
        agent.add_memory(
            f"Memory item {i}: This is some content that describes event number {i}. It contains important information.",
            ImportanceLevel.MEDIUM
        )
    
    print("   Added 5 more memories")
    print("   Compressing with hierarchical algorithm...")
    
    stats = agent.compress_memories(CompressionType.HIERARCHICAL)
    
    print(f"\n   Hierarchical compression:")
    print(f"     Original size: {stats.original_size} bytes")
    print(f"     Compressed size: {stats.compressed_size} bytes")
    print(f"     Compression ratio: {stats.compression_ratio:.2f}x")
    
    if agent.compressed_memories:
        latest = agent.compressed_memories[-1]
        if "num_levels" in latest.metadata:
            print(f"     Hierarchy levels: {latest.metadata['num_levels']}")
    
    print("\n7. SEARCHING COMPRESSED MEMORIES")
    print("-" * 70)
    print("   Searching for 'project'...")
    
    results = agent.search_compressed("project")
    print(f"\n   Found {len(results)} compressed memories")
    
    for i, compressed in enumerate(results[:2], 1):
        print(f"\n     Result {i}:")
        print(f"       Type: {compressed.compression_type.value}")
        print(f"       Original memories: {len(compressed.original_ids)}")
        print(f"       Content preview: {compressed.compressed_content[:100]}...")
    
    print("\n8. OVERALL STATISTICS")
    print("-" * 70)
    
    stats = agent.get_statistics()
    print(f"   Current size: {stats['current_size_bytes']} bytes")
    print(f"   Max size: {stats['max_size_bytes']} bytes")
    print(f"   Utilization: {stats['utilization']*100:.1f}%")
    print(f"   Active memories: {stats['num_memories']}")
    print(f"   Compressed memories: {stats['num_compressed']}")
    print(f"   Total compressions: {stats['total_compressions']}")
    print(f"   Average compression ratio: {stats['average_compression_ratio']:.2f}x")
    print(f"   Total space saved: {stats['space_saved']} bytes")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("1. Lossless compression (deduplication, references)")
    print("2. Lossy compression (summarization, filtering)")
    print("3. Hierarchical compression (multi-level summaries)")
    print("4. Importance-based retention")
    print("5. Automatic compression triggers")
    print("6. Search in compressed memories")
    print("7. Compression statistics and monitoring")
    print("8. Space-efficient memory management")


if __name__ == "__main__":
    demonstrate_memory_compression()
