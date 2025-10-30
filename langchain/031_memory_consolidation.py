"""
Pattern 031: Memory Consolidation

Description:
    Memory consolidation processes and organizes memories over time through
    summarization, compression, and hierarchical structuring. This pattern prevents
    memory overflow, improves retrieval efficiency, and maintains the most important
    information while discarding or compressing less relevant details.

Components:
    - Consolidation Scheduler: Triggers consolidation processes
    - Summarizer: Creates compressed representations of memories
    - Importance Evaluator: Scores memory significance
    - Memory Pruner: Removes or archives low-value memories
    - Hierarchical Organizer: Creates multi-level memory structures
    - Index Builder: Creates efficient retrieval structures

Use Cases:
    - Long-running agents with accumulating memory
    - Managing limited memory resources
    - Improving retrieval speed
    - Creating knowledge hierarchies
    - Intelligent forgetting

LangChain Implementation:
    Uses LLM-based summarization, importance scoring, and hierarchical
    organization to maintain an efficient and useful memory structure.
"""

import os
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class MemoryLevel(Enum):
    """Hierarchical levels of memory consolidation."""
    RAW = "raw"  # Original, unconsolidated memories
    CONSOLIDATED = "consolidated"  # Grouped and summarized
    ABSTRACTED = "abstracted"  # High-level abstractions
    ARCHIVED = "archived"  # Long-term storage


class ImportanceCriterion(Enum):
    """Criteria for evaluating memory importance."""
    FREQUENCY = "frequency"  # How often accessed
    RECENCY = "recency"  # How recent
    EMOTIONAL = "emotional"  # Emotional significance
    RELEVANCE = "relevance"  # Topic relevance
    UNIQUENESS = "uniqueness"  # How unique/rare
    OUTCOME = "outcome"  # Success/failure impact


@dataclass
class Memory:
    """Individual memory unit."""
    id: str
    content: str
    timestamp: datetime
    level: MemoryLevel
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    consolidated_from: List[str] = field(default_factory=list)  # Source memory IDs
    
    def mark_accessed(self):
        """Mark memory as accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class ConsolidatedMemory(Memory):
    """Consolidated memory containing summarized information."""
    summary: str = ""
    source_count: int = 0
    time_span: Optional[timedelta] = None
    
    def __post_init__(self):
        self.level = MemoryLevel.CONSOLIDATED


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""
    memories_processed: int
    memories_consolidated: int
    memories_pruned: int
    memories_archived: int
    new_memories_created: int
    space_saved: float  # Percentage


class ImportanceEvaluator:
    """
    Evaluates memory importance using multiple criteria.
    
    Combines various factors to produce a single importance score.
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Evaluate the importance of this memory on a scale of 0.0 to 1.0.

Memory: {content}
Context: {context}

Consider:
- Relevance to long-term goals
- Uniqueness of information
- Emotional or practical significance
- Potential future utility

Respond with just a number between 0.0 and 1.0."""),
            ("user", "Evaluate importance")
        ])
    
    def evaluate(
        self,
        memory: Memory,
        current_time: datetime,
        recency_weight: float = 0.3,
        access_weight: float = 0.3,
        llm_weight: float = 0.4
    ) -> float:
        """
        Calculate importance score for a memory.
        
        Combines multiple factors with configurable weights.
        """
        # Recency score (exponential decay)
        hours_ago = (current_time - memory.timestamp).total_seconds() / 3600
        recency_score = max(0.0, 1.0 - (hours_ago / (24 * 7)))  # Decay over 1 week
        
        # Access frequency score (normalized)
        access_score = min(1.0, memory.access_count / 10.0)
        
        # LLM-based semantic importance
        llm_score = self._evaluate_semantic_importance(memory)
        
        # Weighted combination
        importance = (
            recency_weight * recency_score +
            access_weight * access_score +
            llm_weight * llm_score
        )
        
        return importance
    
    def _evaluate_semantic_importance(self, memory: Memory) -> float:
        """Use LLM to evaluate semantic importance."""
        try:
            chain = self.evaluation_prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "content": memory.content,
                "context": json.dumps(memory.metadata)
            })
            
            # Extract number from response
            score_str = result.strip()
            score = float(score_str)
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except Exception:
            # Fallback to default
            return 0.5


class MemoryConsolidator:
    """
    Performs memory consolidation operations.
    
    Capabilities:
    - Group related memories
    - Create summaries
    - Build hierarchies
    - Prune low-value memories
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        # Prompt for summarizing multiple memories
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """Summarize these related memories into a concise consolidated memory.

Memories to consolidate:
{memories}

Create a summary that captures the essential information while being more concise.
Focus on patterns, key facts, and important outcomes."""),
            ("user", "Create consolidated summary")
        ])
        
        # Prompt for finding relationships
        self.grouping_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze these memories and identify which ones are related and should be grouped together.

Memories:
{memories}

Return groups as JSON array of arrays containing memory indices.
Example: [[0, 1, 3], [2, 4], [5]]"""),
            ("user", "Group related memories")
        ])
    
    def consolidate_memories(
        self,
        memories: List[Memory],
        importance_threshold: float = 0.3,
        max_age_days: int = 30
    ) -> ConsolidationResult:
        """
        Perform full consolidation cycle on memories.
        
        Steps:
        1. Evaluate importance
        2. Group related memories
        3. Summarize groups
        4. Prune low-value memories
        5. Update memory levels
        """
        current_time = datetime.now()
        evaluator = ImportanceEvaluator(self.llm)
        
        result = ConsolidationResult(
            memories_processed=len(memories),
            memories_consolidated=0,
            memories_pruned=0,
            memories_archived=0,
            new_memories_created=0,
            space_saved=0.0
        )
        
        # Step 1: Update importance scores
        for memory in memories:
            memory.importance_score = evaluator.evaluate(memory, current_time)
        
        # Step 2: Identify memories to prune
        prunable = [
            m for m in memories
            if m.importance_score < importance_threshold
            and m.level == MemoryLevel.RAW
        ]
        result.memories_pruned = len(prunable)
        
        # Step 3: Group related memories for consolidation
        consolidatable = [
            m for m in memories
            if m.level == MemoryLevel.RAW
            and m not in prunable
            and (current_time - m.timestamp).days <= max_age_days
        ]
        
        if len(consolidatable) >= 2:
            groups = self._group_memories(consolidatable)
            
            # Step 4: Create consolidated memories
            for group in groups:
                if len(group) >= 2:  # Only consolidate groups of 2+
                    consolidated = self._create_consolidated_memory(group)
                    result.new_memories_created += 1
                    result.memories_consolidated += len(group)
        
        # Step 5: Archive old memories
        archivable = [
            m for m in memories
            if (current_time - m.timestamp).days > max_age_days
            and m.level == MemoryLevel.CONSOLIDATED
        ]
        result.memories_archived = len(archivable)
        
        # Calculate space saved
        if memories:
            original_size = sum(len(m.content) for m in memories)
            pruned_size = sum(len(m.content) for m in prunable)
            if original_size > 0:
                result.space_saved = (pruned_size / original_size) * 100
        
        return result
    
    def _group_memories(self, memories: List[Memory]) -> List[List[Memory]]:
        """
        Group related memories together.
        
        Uses tag similarity and temporal proximity.
        """
        if len(memories) <= 1:
            return [[m] for m in memories]
        
        groups = []
        processed = set()
        
        for i, memory in enumerate(memories):
            if i in processed:
                continue
            
            group = [memory]
            processed.add(i)
            
            # Find similar memories
            for j, other in enumerate(memories[i+1:], start=i+1):
                if j in processed:
                    continue
                
                # Check tag overlap
                if memory.tags and other.tags:
                    overlap = memory.tags & other.tags
                    if len(overlap) >= 1:  # At least one common tag
                        group.append(other)
                        processed.add(j)
                        continue
                
                # Check temporal proximity (within 1 day)
                time_diff = abs((memory.timestamp - other.timestamp).total_seconds())
                if time_diff < 86400:  # 24 hours
                    group.append(other)
                    processed.add(j)
            
            groups.append(group)
        
        return groups
    
    def _create_consolidated_memory(self, group: List[Memory]) -> ConsolidatedMemory:
        """
        Create a consolidated memory from a group.
        """
        # Prepare memory texts
        memory_texts = "\n\n".join([
            f"Memory {i+1} ({m.timestamp.strftime('%Y-%m-%d %H:%M')}):\n{m.content}"
            for i, m in enumerate(group)
        ])
        
        # Generate summary
        chain = self.summary_prompt | self.llm | StrOutputParser()
        summary = chain.invoke({"memories": memory_texts})
        
        # Combine metadata
        combined_tags = set()
        for memory in group:
            combined_tags.update(memory.tags)
        
        # Calculate time span
        timestamps = [m.timestamp for m in group]
        time_span = max(timestamps) - min(timestamps)
        
        # Calculate average importance
        avg_importance = sum(m.importance_score for m in group) / len(group)
        
        # Create consolidated memory
        consolidated = ConsolidatedMemory(
            id=f"consolidated_{group[0].id}",
            content=summary,
            summary=summary,
            timestamp=min(timestamps),  # Use earliest timestamp
            level=MemoryLevel.CONSOLIDATED,
            importance_score=avg_importance,
            tags=combined_tags,
            consolidated_from=[m.id for m in group],
            source_count=len(group),
            time_span=time_span
        )
        
        return consolidated


class ConsolidationScheduler:
    """
    Manages when and how consolidation is triggered.
    
    Supports multiple triggering strategies:
    - Time-based (periodic)
    - Size-based (memory threshold)
    - Demand-based (on query)
    """
    
    def __init__(
        self,
        consolidation_interval_hours: int = 24,
        size_threshold: int = 100,
        importance_threshold: float = 0.3
    ):
        self.consolidation_interval_hours = consolidation_interval_hours
        self.size_threshold = size_threshold
        self.importance_threshold = importance_threshold
        self.last_consolidation = datetime.now()
    
    def should_consolidate(
        self,
        memory_count: int,
        time_since_last: Optional[timedelta] = None
    ) -> bool:
        """
        Determine if consolidation should be triggered.
        """
        # Size-based trigger
        if memory_count >= self.size_threshold:
            return True
        
        # Time-based trigger
        if time_since_last is None:
            time_since_last = datetime.now() - self.last_consolidation
        
        hours_since = time_since_last.total_seconds() / 3600
        if hours_since >= self.consolidation_interval_hours:
            return True
        
        return False
    
    def mark_consolidated(self):
        """Mark that consolidation has occurred."""
        self.last_consolidation = datetime.now()


class MemoryConsolidationAgent:
    """
    Agent that manages memory with automatic consolidation.
    
    Features:
    - Automatic memory consolidation
    - Importance-based retention
    - Hierarchical memory organization
    - Efficient retrieval
    """
    
    def __init__(self, temperature: float = 0.3):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        self.consolidator = MemoryConsolidator(self.llm)
        self.scheduler = ConsolidationScheduler()
        
        self.raw_memories: List[Memory] = []
        self.consolidated_memories: List[ConsolidatedMemory] = []
        self.archived_memories: List[Memory] = []
        
        self._next_id = 1
    
    def add_memory(
        self,
        content: str,
        tags: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new memory and check if consolidation is needed."""
        memory = Memory(
            id=f"mem_{self._next_id:04d}",
            content=content,
            timestamp=datetime.now(),
            level=MemoryLevel.RAW,
            tags=tags or set(),
            metadata=metadata or {}
        )
        
        self._next_id += 1
        self.raw_memories.append(memory)
        
        # Check if consolidation should be triggered
        if self.scheduler.should_consolidate(len(self.raw_memories)):
            self.consolidate()
        
        return memory.id
    
    def consolidate(self) -> ConsolidationResult:
        """
        Perform memory consolidation.
        
        This processes raw memories, creates consolidated versions,
        and prunes low-value memories.
        """
        print("\n🔄 Starting memory consolidation...")
        
        result = self.consolidator.consolidate_memories(
            self.raw_memories,
            importance_threshold=self.scheduler.importance_threshold
        )
        
        # Apply consolidation results
        # Remove pruned memories
        self.raw_memories = [
            m for m in self.raw_memories
            if m.importance_score >= self.scheduler.importance_threshold
        ]
        
        # Mark scheduler
        self.scheduler.mark_consolidated()
        
        print(f"✓ Consolidation complete: {result.memories_consolidated} memories consolidated, "
              f"{result.memories_pruned} pruned, {result.space_saved:.1f}% space saved")
        
        return result
    
    def retrieve_memories(
        self,
        query: str,
        include_consolidated: bool = True,
        max_results: int = 5
    ) -> List[Memory]:
        """
        Retrieve relevant memories.
        
        Searches across raw and consolidated memories.
        """
        query_lower = query.lower()
        all_memories = self.raw_memories[:]
        
        if include_consolidated:
            all_memories.extend(self.consolidated_memories)
        
        # Simple keyword matching
        relevant = []
        for memory in all_memories:
            if query_lower in memory.content.lower():
                memory.mark_accessed()
                relevant.append(memory)
        
        # Sort by importance * recency
        current_time = datetime.now()
        relevant.sort(
            key=lambda m: m.importance_score * (
                1.0 / (1.0 + (current_time - m.timestamp).days)
            ),
            reverse=True
        )
        
        return relevant[:max_results]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total_memories = (
            len(self.raw_memories) +
            len(self.consolidated_memories) +
            len(self.archived_memories)
        )
        
        consolidated_count = sum(
            len(m.consolidated_from)
            for m in self.consolidated_memories
        )
        
        return {
            "total_memories": total_memories,
            "raw_memories": len(self.raw_memories),
            "consolidated_memories": len(self.consolidated_memories),
            "archived_memories": len(self.archived_memories),
            "consolidation_ratio": (
                consolidated_count / total_memories if total_memories > 0 else 0
            ),
            "avg_importance": (
                sum(m.importance_score for m in self.raw_memories) / len(self.raw_memories)
                if self.raw_memories else 0
            )
        }


def demonstrate_memory_consolidation():
    """
    Demonstrates memory consolidation with automatic processing,
    importance evaluation, and hierarchical organization.
    """
    print("=" * 80)
    print("MEMORY CONSOLIDATION DEMONSTRATION")
    print("=" * 80)
    
    # Create agent
    agent = MemoryConsolidationAgent()
    
    # Test 1: Add many memories to trigger consolidation
    print("\n" + "=" * 80)
    print("Test 1: Adding Memories (Will Trigger Consolidation)")
    print("=" * 80)
    
    # Add memories in groups
    python_memories = [
        ("Learned about Python list comprehensions", {"python", "programming"}),
        ("Practiced Python decorators", {"python", "programming"}),
        ("Studied Python generators", {"python", "programming"}),
    ]
    
    database_memories = [
        ("Learned SQL JOIN operations", {"sql", "database"}),
        ("Practiced database indexing", {"sql", "database"}),
        ("Studied database normalization", {"sql", "database"}),
    ]
    
    misc_memories = [
        ("Read about machine learning basics", {"ml", "learning"}),
        ("Explored neural networks", {"ml", "learning"}),
        ("Had coffee at 3pm", {"personal"}),
        ("Answered user questions about APIs", {"work", "api"}),
    ]
    
    all_memory_groups = [python_memories, database_memories, misc_memories]
    
    for memory_group in all_memory_groups:
        for content, tags in memory_group:
            agent.add_memory(content, tags=tags)
            print(f"✓ Added: {content}")
    
    # Show initial statistics
    stats = agent.get_statistics()
    print(f"\nInitial statistics:")
    print(f"  Raw memories: {stats['raw_memories']}")
    print(f"  Consolidated memories: {stats['consolidated_memories']}")
    
    # Test 2: Manual consolidation
    print("\n" + "=" * 80)
    print("Test 2: Manual Consolidation Trigger")
    print("=" * 80)
    
    result = agent.consolidate()
    print(f"\nConsolidation results:")
    print(f"  Memories processed: {result.memories_processed}")
    print(f"  Memories consolidated: {result.memories_consolidated}")
    print(f"  Memories pruned: {result.memories_pruned}")
    print(f"  New consolidated memories: {result.new_memories_created}")
    print(f"  Space saved: {result.space_saved:.1f}%")
    
    # Show updated statistics
    stats = agent.get_statistics()
    print(f"\nUpdated statistics:")
    print(f"  Total memories: {stats['total_memories']}")
    print(f"  Raw memories: {stats['raw_memories']}")
    print(f"  Consolidated memories: {stats['consolidated_memories']}")
    print(f"  Consolidation ratio: {stats['consolidation_ratio']:.2f}")
    print(f"  Average importance: {stats['avg_importance']:.2f}")
    
    # Test 3: Retrieve memories
    print("\n" + "=" * 80)
    print("Test 3: Memory Retrieval After Consolidation")
    print("=" * 80)
    
    queries = ["Python", "database", "machine learning"]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        memories = agent.retrieve_memories(query, max_results=3)
        print(f"Found {len(memories)} relevant memories:")
        for memory in memories:
            level_icon = "📝" if memory.level == MemoryLevel.RAW else "📚"
            print(f"  {level_icon} [{memory.importance_score:.2f}] {memory.content[:60]}...")
            if isinstance(memory, ConsolidatedMemory) and memory.consolidated_from:
                print(f"      (Consolidated from {len(memory.consolidated_from)} memories)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Memory Consolidation provides:
✓ Automatic memory organization and compression
✓ Importance-based retention
✓ Hierarchical memory structures
✓ Intelligent forgetting (pruning)
✓ Improved retrieval efficiency

This pattern excels at:
- Managing long-running agents with accumulating memory
- Preventing memory overflow
- Maintaining important information while discarding noise
- Creating knowledge hierarchies
- Balancing detail with efficiency

Key consolidation operations:
1. Importance Evaluation: Score memories based on multiple factors
2. Grouping: Identify related memories
3. Summarization: Create compressed representations
4. Pruning: Remove low-value memories
5. Archiving: Move old memories to long-term storage

Triggering strategies:
- Time-based: Periodic consolidation (every 24 hours)
- Size-based: When memory count exceeds threshold (100 items)
- Demand-based: Before retrieval operations

Benefits:
- Reduced memory footprint
- Faster retrieval
- Better organization
- Preserved important information
- Natural forgetting of irrelevant details

Use memory consolidation when you need to:
- Run agents for extended periods
- Manage limited memory resources
- Maintain retrieval performance
- Build knowledge hierarchies
- Implement realistic forgetting
""")


if __name__ == "__main__":
    demonstrate_memory_consolidation()
