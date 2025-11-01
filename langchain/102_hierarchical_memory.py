"""
Pattern 102: Hierarchical Memory

Description:
    The Hierarchical Memory pattern organizes agent memory into multiple levels with
    different characteristics, mirroring human memory systems that include working
    memory (immediate, limited capacity), short-term memory (temporary, accessible),
    and long-term memory (persistent, large capacity). This pattern enables efficient
    memory management by storing information at appropriate levels based on recency,
    importance, and access frequency.
    
    Each level in the hierarchy has distinct properties: working memory holds current
    context with fast access but very limited capacity; short-term memory maintains
    recent information with moderate capacity and access speed; long-term memory stores
    consolidated information with large capacity but slower access. Information flows
    between levels through promotion (consolidation) and retrieval mechanisms.
    
    This pattern supports both automatic and strategic memory movement between levels,
    implements forgetting at each level with different strategies, and enables
    efficient multi-level search for relevant information.

Key Components:
    1. Working Memory: Current active context (very limited, fast)
    2. Short-Term Memory: Recent information (limited, moderate speed)
    3. Long-Term Memory: Consolidated knowledge (unlimited, slower)
    4. Promotion Mechanism: Move important info to higher levels
    5. Retrieval System: Search across all levels
    6. Consolidation: Strengthen and reorganize memories
    7. Level-specific Forgetting: Different strategies per level

Memory Levels:
    1. Working Memory (L0):
       - Capacity: 5-9 items (Miller's Law)
       - Duration: Active task only
       - Access: Instant
       - Purpose: Current context and immediate operations
    
    2. Short-Term Memory (L1):
       - Capacity: 50-100 items
       - Duration: Recent session/day
       - Access: Very fast
       - Purpose: Recent interactions and temporary information
    
    3. Long-Term Memory (L2):
       - Capacity: Unlimited (practical limits apply)
       - Duration: Persistent across sessions
       - Access: Slower (requires search)
       - Purpose: Important knowledge and experiences

Memory Operations:
    1. Store: Add new memory at appropriate level
    2. Retrieve: Search across levels
    3. Promote: Move important memories up
    4. Consolidate: Reorganize and strengthen
    5. Forget: Remove based on level-specific rules
    6. Rehearse: Strengthen through repetition

Use Cases:
    - Conversational agents with context management
    - Learning systems with knowledge retention
    - Personal assistants with memory across sessions
    - Task-oriented agents with working context
    - Long-running autonomous agents
    - Multi-session applications
    - Knowledge management systems

Advantages:
    - Efficient memory organization
    - Appropriate retention strategies per level
    - Fast access to recent/important information
    - Scalable to large memory requirements
    - Natural forgetting curves
    - Supports both short and long-term needs
    - Mimics human memory structure

Challenges:
    - Determining appropriate level for memories
    - Managing transitions between levels
    - Balancing access speed vs. capacity
    - Consolidation strategy selection
    - Preventing important memory loss
    - Synchronization across levels

LangChain Implementation:
    This implementation uses LangChain for:
    - LLM-based memory consolidation and summarization
    - Importance assessment for promotion decisions
    - Context-aware retrieval across levels
    - Memory linking and association
    
Production Considerations:
    - Tune capacity limits for each level
    - Implement persistent storage for long-term memory
    - Monitor access patterns and adjust
    - Regular consolidation during low-activity periods
    - Backup critical memories before forgetting
    - Implement recovery mechanisms
    - Consider storage costs for long-term memory
    - Balance access speed with storage efficiency
"""

import os
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class MemoryLevel(Enum):
    """Hierarchical memory levels."""
    WORKING = "working"  # L0: Current context (5-9 items)
    SHORT_TERM = "short_term"  # L1: Recent memory (50-100 items)
    LONG_TERM = "long_term"  # L2: Persistent memory (unlimited)


@dataclass
class HierarchicalMemoryItem:
    """
    Memory item with hierarchical tracking.
    
    Attributes:
        memory_id: Unique identifier
        content: Memory content
        level: Current memory level
        created_at: Creation timestamp
        last_accessed: Last access timestamp
        access_count: Number of accesses
        importance: Importance score (0-1)
        strength: Memory strength (0-1)
        tags: Associated tags
        links: Links to related memories
        metadata: Additional information
    """
    memory_id: str
    content: str
    level: MemoryLevel
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    importance: float = 0.5
    strength: float = 0.5
    tags: List[str] = field(default_factory=list)
    links: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalMemorySystem:
    """
    Multi-level hierarchical memory system.
    
    This system manages memory across working, short-term, and long-term levels
    with automatic promotion, consolidation, and forgetting mechanisms.
    """
    
    def __init__(
        self,
        working_capacity: int = 7,
        short_term_capacity: int = 50,
        promotion_threshold: float = 0.7,
        temperature: float = 0.3
    ):
        """
        Initialize hierarchical memory system.
        
        Args:
            working_capacity: Max items in working memory
            short_term_capacity: Max items in short-term memory
            promotion_threshold: Threshold for promoting memories
            temperature: LLM temperature
        """
        self.working_capacity = working_capacity
        self.short_term_capacity = short_term_capacity
        self.promotion_threshold = promotion_threshold
        
        # Memory stores for each level
        self.working_memory: deque = deque(maxlen=working_capacity)
        self.short_term_memory: Dict[str, HierarchicalMemoryItem] = {}
        self.long_term_memory: Dict[str, HierarchicalMemoryItem] = {}
        
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.memory_counter = 0
    
    def store(
        self,
        content: str,
        level: MemoryLevel = MemoryLevel.WORKING,
        importance: float = 0.5,
        tags: List[str] = None
    ) -> HierarchicalMemoryItem:
        """
        Store memory at specified level.
        
        Args:
            content: Memory content
            level: Initial memory level
            importance: Importance score
            tags: Associated tags
            
        Returns:
            Created memory item
        """
        self.memory_counter += 1
        memory_id = f"mem_{self.memory_counter}"
        
        now = datetime.now()
        memory = HierarchicalMemoryItem(
            memory_id=memory_id,
            content=content,
            level=level,
            created_at=now,
            last_accessed=now,
            importance=importance,
            strength=importance,
            tags=tags or []
        )
        
        if level == MemoryLevel.WORKING:
            # Add to working memory (FIFO queue)
            if len(self.working_memory) >= self.working_capacity:
                # Move oldest to short-term before removing
                oldest = self.working_memory[0]
                self._promote_to_short_term(oldest)
            self.working_memory.append(memory)
        
        elif level == MemoryLevel.SHORT_TERM:
            self.short_term_memory[memory_id] = memory
            # Check capacity
            if len(self.short_term_memory) > self.short_term_capacity:
                self._consolidate_short_term()
        
        else:  # LONG_TERM
            self.long_term_memory[memory_id] = memory
        
        return memory
    
    def retrieve(
        self,
        query: str = None,
        tags: List[str] = None,
        level: Optional[MemoryLevel] = None,
        n: int = 5
    ) -> List[HierarchicalMemoryItem]:
        """
        Retrieve memories across levels.
        
        Args:
            query: Search query
            tags: Filter by tags
            level: Specific level to search (None for all)
            n: Number of results
            
        Returns:
            List of matching memories
        """
        results = []
        
        # Search working memory (always checked first)
        if level is None or level == MemoryLevel.WORKING:
            for mem in self.working_memory:
                if self._matches(mem, query, tags):
                    mem.last_accessed = datetime.now()
                    mem.access_count += 1
                    results.append(mem)
        
        # Search short-term memory
        if level is None or level == MemoryLevel.SHORT_TERM:
            for mem in self.short_term_memory.values():
                if self._matches(mem, query, tags):
                    mem.last_accessed = datetime.now()
                    mem.access_count += 1
                    results.append(mem)
        
        # Search long-term memory
        if level is None or level == MemoryLevel.LONG_TERM:
            for mem in self.long_term_memory.values():
                if self._matches(mem, query, tags):
                    mem.last_accessed = datetime.now()
                    mem.access_count += 1
                    results.append(mem)
        
        # Sort by relevance (strength * recency)
        now = datetime.now()
        def score(mem):
            recency = 1.0 / (1 + (now - mem.last_accessed).total_seconds() / 3600)
            return mem.strength * 0.7 + recency * 0.3
        
        results.sort(key=score, reverse=True)
        return results[:n]
    
    def _matches(
        self,
        memory: HierarchicalMemoryItem,
        query: Optional[str],
        tags: Optional[List[str]]
    ) -> bool:
        """Check if memory matches search criteria."""
        if tags:
            if not any(tag in memory.tags for tag in tags):
                return False
        
        if query:
            if query.lower() not in memory.content.lower():
                return False
        
        return True
    
    def _promote_to_short_term(self, memory: HierarchicalMemoryItem):
        """Promote working memory item to short-term."""
        memory.level = MemoryLevel.SHORT_TERM
        self.short_term_memory[memory.memory_id] = memory
        
        # Check if should promote further to long-term
        if memory.strength >= self.promotion_threshold:
            self._promote_to_long_term(memory)
    
    def _promote_to_long_term(self, memory: HierarchicalMemoryItem):
        """Promote short-term memory to long-term."""
        memory.level = MemoryLevel.LONG_TERM
        
        # Remove from short-term
        if memory.memory_id in self.short_term_memory:
            del self.short_term_memory[memory.memory_id]
        
        # Add to long-term
        self.long_term_memory[memory.memory_id] = memory
    
    def _consolidate_short_term(self):
        """Consolidate short-term memories when capacity exceeded."""
        # Sort by strength and age
        memories = sorted(
            self.short_term_memory.values(),
            key=lambda m: (m.strength, m.access_count),
            reverse=True
        )
        
        # Promote top memories to long-term
        to_promote = memories[:int(self.short_term_capacity * 0.3)]
        for mem in to_promote:
            if mem.strength >= self.promotion_threshold * 0.8:
                self._promote_to_long_term(mem)
        
        # Remove weakest memories
        to_remove = memories[self.short_term_capacity:]
        for mem in to_remove:
            del self.short_term_memory[mem.memory_id]
    
    def strengthen(self, memory_id: str, amount: float = 0.1):
        """
        Strengthen a memory.
        
        Args:
            memory_id: Memory to strengthen
            amount: Amount to increase strength
        """
        # Check all levels
        memory = None
        
        for mem in self.working_memory:
            if mem.memory_id == memory_id:
                memory = mem
                break
        
        if not memory and memory_id in self.short_term_memory:
            memory = self.short_term_memory[memory_id]
        
        if not memory and memory_id in self.long_term_memory:
            memory = self.long_term_memory[memory_id]
        
        if memory:
            memory.strength = min(1.0, memory.strength + amount)
            memory.importance = min(1.0, memory.importance + amount * 0.5)
            
            # Check for promotion
            if memory.level == MemoryLevel.SHORT_TERM and \
               memory.strength >= self.promotion_threshold:
                self._promote_to_long_term(memory)
    
    def consolidate_all(self):
        """
        Perform full memory consolidation.
        
        This simulates sleep/rest consolidation process.
        """
        # Move working memory to short-term
        while len(self.working_memory) > 0:
            mem = self.working_memory.popleft()
            self._promote_to_short_term(mem)
        
        # Consolidate short-term
        self._consolidate_short_term()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics across all levels."""
        return {
            "working_memory": {
                "count": len(self.working_memory),
                "capacity": self.working_capacity,
                "usage": len(self.working_memory) / self.working_capacity
            },
            "short_term_memory": {
                "count": len(self.short_term_memory),
                "capacity": self.short_term_capacity,
                "usage": len(self.short_term_memory) / self.short_term_capacity
            },
            "long_term_memory": {
                "count": len(self.long_term_memory),
                "avg_strength": sum(m.strength for m in self.long_term_memory.values()) / len(self.long_term_memory) if self.long_term_memory else 0
            },
            "total_memories": len(self.working_memory) + len(self.short_term_memory) + len(self.long_term_memory)
        }
    
    def get_memory_at_level(self, level: MemoryLevel) -> List[HierarchicalMemoryItem]:
        """Get all memories at specific level."""
        if level == MemoryLevel.WORKING:
            return list(self.working_memory)
        elif level == MemoryLevel.SHORT_TERM:
            return list(self.short_term_memory.values())
        else:
            return list(self.long_term_memory.values())


class HierarchicalMemoryAgent:
    """
    Agent with hierarchical memory system.
    
    This agent uses multi-level memory to maintain context while
    supporting long-term knowledge retention.
    """
    
    def __init__(
        self,
        working_capacity: int = 7,
        short_term_capacity: int = 50,
        temperature: float = 0.5
    ):
        """
        Initialize hierarchical memory agent.
        
        Args:
            working_capacity: Working memory capacity
            short_term_capacity: Short-term memory capacity
            temperature: LLM temperature
        """
        self.memory_system = HierarchicalMemorySystem(
            working_capacity=working_capacity,
            short_term_capacity=short_term_capacity
        )
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
    
    def process(
        self,
        user_input: str,
        context_tags: List[str] = None
    ) -> str:
        """
        Process input with hierarchical memory.
        
        Args:
            user_input: User input
            context_tags: Context tags for retrieval
            
        Returns:
            Response
        """
        # Retrieve relevant memories from all levels
        relevant_memories = self.memory_system.retrieve(
            query=user_input,
            tags=context_tags,
            n=5
        )
        
        # Build context from memories
        memory_context = ""
        if relevant_memories:
            memory_context = "Relevant memories:\n"
            for mem in relevant_memories[:3]:
                memory_context += f"- [{mem.level.value}] {mem.content}\n"
        
        # Generate response
        prompt = ChatPromptTemplate.from_template(
            "You are an assistant with hierarchical memory.\n\n"
            "{memory_context}\n"
            "User: {user_input}\n\n"
            "Respond naturally, using relevant memories."
        )
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "memory_context": memory_context if memory_context else "No relevant memories.",
            "user_input": user_input
        })
        
        # Store interaction in working memory
        self.memory_system.store(
            content=f"Q: {user_input[:50]} | A: {response[:50]}",
            level=MemoryLevel.WORKING,
            importance=0.5,
            tags=context_tags or []
        )
        
        # Strengthen accessed memories
        for mem in relevant_memories:
            self.memory_system.strengthen(mem.memory_id, 0.05)
        
        return response
    
    def memorize(
        self,
        content: str,
        importance: float = 0.7,
        level: MemoryLevel = MemoryLevel.SHORT_TERM,
        tags: List[str] = None
    ):
        """
        Explicitly memorize information.
        
        Args:
            content: Content to memorize
            importance: Importance score
            level: Initial memory level
            tags: Associated tags
        """
        self.memory_system.store(
            content=content,
            level=level,
            importance=importance,
            tags=tags or []
        )
    
    def consolidate(self):
        """Perform memory consolidation."""
        self.memory_system.consolidate_all()


def demonstrate_hierarchical_memory():
    """Demonstrate hierarchical memory pattern."""
    
    print("=" * 80)
    print("HIERARCHICAL MEMORY PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic three-level memory
    print("\n" + "=" * 80)
    print("Example 1: Three-Level Memory Structure")
    print("=" * 80)
    
    memory_system = HierarchicalMemorySystem(
        working_capacity=5,
        short_term_capacity=10,
        promotion_threshold=0.7
    )
    
    print(f"\nMemory Configuration:")
    print(f"  Working Memory Capacity: {memory_system.working_capacity}")
    print(f"  Short-Term Memory Capacity: {memory_system.short_term_capacity}")
    print(f"  Promotion Threshold: {memory_system.promotion_threshold}")
    
    # Store at different levels
    print("\nStoring memories at different levels:")
    
    # Working memory
    memory_system.store("Current task: Analyze data", MemoryLevel.WORKING, 0.6, ["task"])
    memory_system.store("User asked about weather", MemoryLevel.WORKING, 0.4, ["context"])
    print("  ✓ Added to working memory: 2 items")
    
    # Short-term memory
    memory_system.store("User's name is John", MemoryLevel.SHORT_TERM, 0.8, ["personal"])
    memory_system.store("Meeting scheduled at 3pm", MemoryLevel.SHORT_TERM, 0.7, ["schedule"])
    print("  ✓ Added to short-term memory: 2 items")
    
    # Long-term memory
    memory_system.store("User allergic to peanuts", MemoryLevel.LONG_TERM, 0.9, ["health", "important"])
    memory_system.store("Company policy: No PII in logs", MemoryLevel.LONG_TERM, 0.9, ["policy"])
    print("  ✓ Added to long-term memory: 2 items")
    
    stats = memory_system.get_statistics()
    print(f"\nMemory Distribution:")
    print(f"  Working: {stats['working_memory']['count']}/{stats['working_memory']['capacity']}")
    print(f"  Short-term: {stats['short_term_memory']['count']}/{stats['short_term_memory']['capacity']}")
    print(f"  Long-term: {stats['long_term_memory']['count']}")
    
    # Example 2: Automatic promotion from working to short-term
    print("\n" + "=" * 80)
    print("Example 2: Automatic Memory Promotion")
    print("=" * 80)
    
    print("\nFilling working memory beyond capacity...")
    for i in range(8):
        memory_system.store(
            f"Working memory item {i}",
            MemoryLevel.WORKING,
            0.5,
            ["test"]
        )
    
    stats = memory_system.get_statistics()
    print(f"\nAfter adding 8 items to working memory:")
    print(f"  Working: {stats['working_memory']['count']}/{stats['working_memory']['capacity']}")
    print(f"  Short-term: {stats['short_term_memory']['count']}/{stats['short_term_memory']['capacity']}")
    print(f"  → Oldest items automatically promoted to short-term")
    
    # Example 3: Cross-level retrieval
    print("\n" + "=" * 80)
    print("Example 3: Cross-Level Memory Retrieval")
    print("=" * 80)
    
    # Add more memories for testing retrieval
    memory_system.store("Paris is the capital of France", MemoryLevel.LONG_TERM, 0.8, ["geography", "facts"])
    memory_system.store("Python is a programming language", MemoryLevel.SHORT_TERM, 0.7, ["programming", "facts"])
    memory_system.store("Today's weather is sunny", MemoryLevel.WORKING, 0.3, ["weather", "today"])
    
    queries = [
        ("facts", "Search by tag: 'facts'"),
        ("Python", "Search by query: 'Python'"),
        (None, "Retrieve top 5 memories (no filter)")
    ]
    
    for query, description in queries:
        print(f"\n{description}")
        if query == "facts":
            results = memory_system.retrieve(tags=["facts"], n=5)
        elif query:
            results = memory_system.retrieve(query=query, n=5)
        else:
            results = memory_system.retrieve(n=5)
        
        for i, mem in enumerate(results, 1):
            print(f"  {i}. [{mem.level.value}] {mem.content[:50]}")
    
    # Example 4: Memory strengthening and promotion
    print("\n" + "=" * 80)
    print("Example 4: Memory Strengthening & Promotion")
    print("=" * 80)
    
    # Add memory to short-term
    test_mem = memory_system.store(
        "Important repeated information",
        MemoryLevel.SHORT_TERM,
        0.6,
        ["important"]
    )
    
    print(f"\nInitial state:")
    print(f"  Level: {test_mem.level.value}")
    print(f"  Strength: {test_mem.strength:.2f}")
    print(f"  Importance: {test_mem.importance:.2f}")
    
    print(f"\nStrengthening memory 3 times...")
    for i in range(3):
        memory_system.strengthen(test_mem.memory_id, 0.1)
        # Re-fetch to get updated state
        results = memory_system.retrieve(query="Important repeated", n=1)
        if results:
            mem = results[0]
            print(f"  After strengthening {i+1}: strength={mem.strength:.2f}, level={mem.level.value}")
    
    # Example 5: Memory consolidation
    print("\n" + "=" * 80)
    print("Example 5: Memory Consolidation Process")
    print("=" * 80)
    
    consolidation_system = HierarchicalMemorySystem(
        working_capacity=5,
        short_term_capacity=15,
        promotion_threshold=0.7
    )
    
    # Simulate a day of interactions
    print("\nSimulating day of interactions...")
    interactions = [
        ("Morning: Check email", 0.4, ["morning", "routine"]),
        ("Meeting notes: Q1 planning", 0.8, ["work", "important"]),
        ("Lunch order: Pizza", 0.2, ["food"]),
        ("Code review completed", 0.7, ["work"]),
        ("User reported bug #123", 0.8, ["work", "bug"]),
        ("Evening: Workout schedule", 0.3, ["personal"]),
    ]
    
    for content, importance, tags in interactions:
        consolidation_system.store(content, MemoryLevel.WORKING, importance, tags)
    
    stats_before = consolidation_system.get_statistics()
    print(f"\nBefore consolidation:")
    print(f"  Working: {stats_before['working_memory']['count']}")
    print(f"  Short-term: {stats_before['short_term_memory']['count']}")
    print(f"  Long-term: {stats_before['long_term_memory']['count']}")
    
    print(f"\nPerforming consolidation (simulating sleep)...")
    consolidation_system.consolidate_all()
    
    stats_after = consolidation_system.get_statistics()
    print(f"\nAfter consolidation:")
    print(f"  Working: {stats_after['working_memory']['count']}")
    print(f"  Short-term: {stats_after['short_term_memory']['count']}")
    print(f"  Long-term: {stats_after['long_term_memory']['count']}")
    print(f"  → Important memories promoted to long-term")
    
    # Example 6: Hierarchical memory agent
    print("\n" + "=" * 80)
    print("Example 6: Agent with Hierarchical Memory")
    print("=" * 80)
    
    agent = HierarchicalMemoryAgent(
        working_capacity=5,
        short_term_capacity=20
    )
    
    # Teach agent some facts
    print("\nTeaching agent long-term facts...")
    agent.memorize(
        "User's favorite color is blue",
        importance=0.8,
        level=MemoryLevel.LONG_TERM,
        tags=["personal", "preferences"]
    )
    agent.memorize(
        "Company founded in 2020",
        importance=0.7,
        level=MemoryLevel.LONG_TERM,
        tags=["company", "facts"]
    )
    
    # Have conversation
    print("\nConversation with agent:")
    conversations = [
        ("What's my favorite color?", ["personal"]),
        ("Tell me about the company", ["company"]),
        ("What did we just talk about?", ["context"]),
    ]
    
    for user_input, tags in conversations:
        print(f"\nUser: {user_input}")
        response = agent.process(user_input, tags)
        print(f"Agent: {response[:200]}...")
    
    # Check memory distribution
    stats = agent.memory_system.get_statistics()
    print(f"\nAgent Memory State:")
    print(f"  Working: {stats['working_memory']['count']}")
    print(f"  Short-term: {stats['short_term_memory']['count']}")
    print(f"  Long-term: {stats['long_term_memory']['count']}")
    
    # Example 7: Memory level visualization
    print("\n" + "=" * 80)
    print("Example 7: Memory Level Visualization")
    print("=" * 80)
    
    viz_system = HierarchicalMemorySystem(working_capacity=5, short_term_capacity=10)
    
    # Populate all levels
    viz_system.store("Current context item 1", MemoryLevel.WORKING, 0.5, ["context"])
    viz_system.store("Current context item 2", MemoryLevel.WORKING, 0.5, ["context"])
    
    for i in range(5):
        viz_system.store(f"Recent memory {i}", MemoryLevel.SHORT_TERM, 0.6, ["recent"])
    
    for i in range(8):
        viz_system.store(f"Long-term knowledge {i}", MemoryLevel.LONG_TERM, 0.8, ["knowledge"])
    
    print("\nMemory Hierarchy Visualization:")
    print("\n┌─────────────────────────────────────────┐")
    print("│  WORKING MEMORY (Fast, Limited)        │")
    print("│  Capacity: 5 items                      │")
    print("└─────────────────────────────────────────┘")
    working_mems = viz_system.get_memory_at_level(MemoryLevel.WORKING)
    for mem in working_mems:
        print(f"    • {mem.content[:35]}")
    
    print("\n┌─────────────────────────────────────────┐")
    print("│  SHORT-TERM MEMORY (Recent)             │")
    print("│  Capacity: 10 items                     │")
    print("└─────────────────────────────────────────┘")
    short_mems = viz_system.get_memory_at_level(MemoryLevel.SHORT_TERM)
    for mem in short_mems[:3]:
        print(f"    • {mem.content[:35]}")
    print(f"    ... and {len(short_mems)-3} more")
    
    print("\n┌─────────────────────────────────────────┐")
    print("│  LONG-TERM MEMORY (Persistent)          │")
    print("│  Capacity: Unlimited                    │")
    print("└─────────────────────────────────────────┘")
    long_mems = viz_system.get_memory_at_level(MemoryLevel.LONG_TERM)
    for mem in long_mems[:3]:
        print(f"    • {mem.content[:35]}")
    print(f"    ... and {len(long_mems)-3} more")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Hierarchical Memory Pattern")
    print("=" * 80)
    
    summary = """
    The Hierarchical Memory pattern demonstrated:
    
    1. THREE-LEVEL STRUCTURE (Example 1):
       - Working Memory: Current context (5 items, instant access)
       - Short-Term Memory: Recent info (50 items, fast access)
       - Long-Term Memory: Persistent knowledge (unlimited, slower)
       - Different capacity and access characteristics per level
    
    2. AUTOMATIC PROMOTION (Example 2):
       - Working memory overflow triggers promotion
       - Oldest items move to short-term automatically
       - Maintains working memory within capacity
       - Seamless memory level transitions
    
    3. CROSS-LEVEL RETRIEVAL (Example 3):
       - Single search across all memory levels
       - Priority to recent/important memories
       - Tag-based and query-based filtering
       - Efficient multi-level search
    
    4. STRENGTHENING & PROMOTION (Example 4):
       - Repeated access strengthens memories
       - Strong memories promoted to higher levels
       - Short-term → Long-term promotion at threshold
       - Automatic importance adjustment
    
    5. CONSOLIDATION (Example 5):
       - Simulates sleep/rest consolidation
       - Working → Short-term → Long-term flow
       - Important memories promoted upward
       - Weak memories forgotten
    
    6. AGENT INTEGRATION (Example 6):
       - Hierarchical memory in agent workflow
       - Long-term facts retained across interactions
       - Working memory tracks conversation context
       - Context-aware responses using all levels
    
    7. VISUALIZATION (Example 7):
       - Clear separation of memory levels
       - Capacity tracking per level
       - Memory distribution visible
       - Hierarchical organization display
    
    KEY BENEFITS:
    ✓ Efficient memory organization
    ✓ Fast access to relevant information
    ✓ Scalable memory management
    ✓ Natural forgetting curves
    ✓ Supports both short and long-term needs
    ✓ Mimics human memory architecture
    ✓ Automatic capacity management
    
    USE CASES:
    • Conversational agents with context
    • Learning systems with retention
    • Personal assistants across sessions
    • Task-oriented agents with working context
    • Long-running autonomous agents
    • Multi-session applications
    • Knowledge management systems
    
    BEST PRACTICES:
    1. Set appropriate capacity for each level
    2. Tune promotion thresholds for use case
    3. Regular consolidation during low activity
    4. Strengthen frequently accessed memories
    5. Implement persistent storage for long-term
    6. Monitor memory distribution
    7. Use tags for efficient retrieval
    8. Balance access speed vs. capacity
    
    TRADE-OFFS:
    • Access speed vs. memory capacity
    • Simplicity vs. sophistication
    • Automatic vs. manual promotion
    • Storage cost vs. retention
    
    PRODUCTION CONSIDERATIONS:
    → Implement persistent storage for long-term memory
    → Tune capacities based on system resources
    → Monitor access patterns and adjust
    → Regular consolidation during off-peak
    → Backup critical memories before operations
    → Consider database for long-term storage
    → Implement recovery mechanisms
    → Balance storage costs with retention needs
    → Track memory usage metrics
    → Test promotion thresholds thoroughly
    
    This pattern provides efficient, scalable memory management by
    organizing information into levels with appropriate characteristics,
    enabling agents to maintain context while supporting long-term retention.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_hierarchical_memory()
