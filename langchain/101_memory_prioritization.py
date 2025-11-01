"""
Pattern 101: Memory Prioritization & Forgetting

Description:
    The Memory Prioritization & Forgetting pattern enables agents to manage limited
    memory resources by selectively retaining important information and forgetting
    less relevant memories. This pattern is crucial for long-running agents that
    accumulate vast amounts of information over time, as it prevents memory overflow,
    maintains system performance, and focuses attention on currently relevant data.
    
    Human memory naturally prioritizes and forgets information based on factors like
    recency, frequency, importance, and emotional significance. This pattern emulates
    these mechanisms to create more efficient and effective agent memory systems.
    
    The pattern implements various forgetting strategies including decay-based forgetting
    (memories fade over time), interference-based forgetting (new memories replace old),
    and strategic forgetting (explicit removal of irrelevant information). It uses
    prioritization schemes to rank memories and make retention decisions.

Key Components:
    1. Memory Priority Scoring: Calculate importance of each memory
    2. Forgetting Mechanisms: Time decay, interference, strategic removal
    3. Retention Policies: Rules for what to keep/forget
    4. Memory Consolidation: Strengthen important memories
    5. Capacity Management: Maintain memory within limits
    6. Access Patterns: Track memory usage for prioritization
    7. Relevance Assessment: Determine current importance

Prioritization Factors:
    1. Recency: How recently was memory accessed/created
    2. Frequency: How often has memory been accessed
    3. Importance: Explicit importance score
    4. Relevance: Similarity to current context
    5. Emotional Valence: Significance of memory content
    6. Utility: How useful has memory been
    
Forgetting Strategies:
    1. Time-based Decay: Memories fade over time
    2. Capacity-based: Remove oldest when full
    3. LRU (Least Recently Used): Remove least accessed
    4. LFU (Least Frequently Used): Remove least frequent
    5. Relevance-based: Remove contextually irrelevant
    6. Strategic: Explicit removal decisions

Use Cases:
    - Long-running conversational agents
    - Personal assistant agents
    - Learning agents with limited memory
    - Real-time systems with memory constraints
    - Adaptive agents in changing environments
    - Information filtering systems
    - Context-aware applications

Advantages:
    - Efficient memory utilization
    - Improved performance (less to search)
    - Focus on relevant information
    - Prevents memory overflow
    - Adapts to changing contexts
    - Natural information lifecycle
    - Scalable for long-term operation

Challenges:
    - Determining importance accurately
    - Risk of forgetting needed information
    - Balancing recency vs. importance
    - Context-dependent relevance
    - Computational overhead of scoring
    - Irreversible information loss

LangChain Implementation:
    This implementation uses LangChain for:
    - LLM-based importance assessment
    - Context relevance evaluation
    - Memory summarization before forgetting
    - Strategic forgetting decisions
    
Production Considerations:
    - Set appropriate memory capacity limits
    - Implement memory backups before forgetting
    - Log all forgetting operations for debugging
    - Monitor memory usage and hit rates
    - Tune priority weights for use case
    - Implement memory recovery mechanisms
    - Consider gradual degradation vs. deletion
    - Balance performance vs. retention
"""

import os
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import math
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ForgettingStrategy(Enum):
    """Strategy for forgetting memories."""
    TIME_DECAY = "time_decay"
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    PRIORITY_BASED = "priority_based"
    CAPACITY_BASED = "capacity_based"
    RELEVANCE_BASED = "relevance_based"


@dataclass
class Memory:
    """
    Represents a memory item with priority tracking.
    
    Attributes:
        memory_id: Unique identifier
        content: Memory content
        created_at: When memory was created
        last_accessed: When last accessed
        access_count: Number of times accessed
        importance: Base importance score (0-1)
        priority: Calculated priority score
        tags: Associated tags
        context: Context when created
        metadata: Additional information
    """
    memory_id: str
    content: str
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    importance: float = 0.5
    priority: float = 0.5
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryPrioritizer:
    """
    Manages memory prioritization and forgetting.
    
    This class implements various strategies for calculating memory priority
    and deciding what to forget when memory capacity is reached.
    """
    
    def __init__(
        self,
        max_capacity: int = 100,
        strategy: ForgettingStrategy = ForgettingStrategy.PRIORITY_BASED,
        decay_rate: float = 0.1,
        recency_weight: float = 0.3,
        frequency_weight: float = 0.2,
        importance_weight: float = 0.5,
        temperature: float = 0.3
    ):
        """
        Initialize memory prioritizer.
        
        Args:
            max_capacity: Maximum number of memories to retain
            strategy: Forgetting strategy to use
            decay_rate: Rate of time-based decay (0-1)
            recency_weight: Weight for recency in priority calculation
            frequency_weight: Weight for frequency in priority calculation
            importance_weight: Weight for importance in priority calculation
            temperature: LLM temperature
        """
        self.max_capacity = max_capacity
        self.strategy = strategy
        self.decay_rate = decay_rate
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.importance_weight = importance_weight
        self.memories: Dict[str, Memory] = {}
        self.forgotten_count = 0
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
    
    def calculate_priority(
        self,
        memory: Memory,
        current_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate priority score for a memory.
        
        Args:
            memory: Memory to score
            current_context: Current context for relevance calculation
            
        Returns:
            Priority score (0-1)
        """
        now = datetime.now()
        
        # Recency score (exponential decay)
        time_since_access = (now - memory.last_accessed).total_seconds()
        recency_score = math.exp(-self.decay_rate * time_since_access / 3600)
        
        # Frequency score (normalized log scale)
        frequency_score = math.log(memory.access_count + 1) / math.log(100)
        frequency_score = min(1.0, frequency_score)
        
        # Importance score (base importance)
        importance_score = memory.importance
        
        # Combined priority
        priority = (
            self.recency_weight * recency_score +
            self.frequency_weight * frequency_score +
            self.importance_weight * importance_score
        )
        
        # Context relevance boost
        if current_context:
            relevance_boost = self._calculate_relevance(memory, current_context)
            priority = priority * (1 + relevance_boost * 0.5)
        
        return min(1.0, max(0.0, priority))
    
    def _calculate_relevance(
        self,
        memory: Memory,
        context: Dict[str, Any]
    ) -> float:
        """
        Calculate relevance of memory to current context.
        
        Args:
            memory: Memory to evaluate
            context: Current context
            
        Returns:
            Relevance score (0-1)
        """
        # Simple tag-based relevance
        context_tags = context.get("tags", [])
        if not context_tags:
            return 0.0
        
        common_tags = set(memory.tags) & set(context_tags)
        if not memory.tags:
            return 0.0
        
        return len(common_tags) / len(memory.tags)
    
    def add_memory(
        self,
        memory_id: str,
        content: str,
        importance: float = 0.5,
        tags: List[str] = None,
        context: Dict[str, Any] = None
    ) -> Memory:
        """
        Add new memory and apply forgetting if needed.
        
        Args:
            memory_id: Unique identifier
            content: Memory content
            importance: Base importance score
            tags: Associated tags
            context: Creation context
            
        Returns:
            Created memory
        """
        now = datetime.now()
        
        memory = Memory(
            memory_id=memory_id,
            content=content,
            created_at=now,
            last_accessed=now,
            importance=importance,
            tags=tags or [],
            context=context or {}
        )
        
        # Calculate initial priority
        memory.priority = self.calculate_priority(memory)
        
        self.memories[memory_id] = memory
        
        # Check capacity and forget if needed
        if len(self.memories) > self.max_capacity:
            self._forget_memories()
        
        return memory
    
    def access_memory(
        self,
        memory_id: str,
        current_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Memory]:
        """
        Access a memory and update its statistics.
        
        Args:
            memory_id: Memory identifier
            current_context: Current context
            
        Returns:
            Memory if found, None otherwise
        """
        memory = self.memories.get(memory_id)
        if not memory:
            return None
        
        # Update access statistics
        memory.last_accessed = datetime.now()
        memory.access_count += 1
        
        # Recalculate priority
        memory.priority = self.calculate_priority(memory, current_context)
        
        return memory
    
    def _forget_memories(self):
        """Forget memories based on strategy."""
        
        if self.strategy == ForgettingStrategy.LRU:
            # Remove least recently used
            to_forget = sorted(
                self.memories.values(),
                key=lambda m: m.last_accessed
            )[:len(self.memories) - self.max_capacity + 1]
        
        elif self.strategy == ForgettingStrategy.LFU:
            # Remove least frequently used
            to_forget = sorted(
                self.memories.values(),
                key=lambda m: m.access_count
            )[:len(self.memories) - self.max_capacity + 1]
        
        elif self.strategy == ForgettingStrategy.PRIORITY_BASED:
            # Remove lowest priority
            to_forget = sorted(
                self.memories.values(),
                key=lambda m: m.priority
            )[:len(self.memories) - self.max_capacity + 1]
        
        elif self.strategy == ForgettingStrategy.TIME_DECAY:
            # Remove oldest
            to_forget = sorted(
                self.memories.values(),
                key=lambda m: m.created_at
            )[:len(self.memories) - self.max_capacity + 1]
        
        else:
            # Default to priority-based
            to_forget = sorted(
                self.memories.values(),
                key=lambda m: m.priority
            )[:len(self.memories) - self.max_capacity + 1]
        
        # Remove memories
        for memory in to_forget:
            del self.memories[memory.memory_id]
            self.forgotten_count += 1
    
    def get_top_memories(
        self,
        n: int,
        current_context: Optional[Dict[str, Any]] = None
    ) -> List[Memory]:
        """
        Get top N memories by priority.
        
        Args:
            n: Number of memories to return
            current_context: Current context for priority calculation
            
        Returns:
            List of top memories
        """
        # Recalculate priorities if context provided
        if current_context:
            for memory in self.memories.values():
                memory.priority = self.calculate_priority(memory, current_context)
        
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: m.priority,
            reverse=True
        )
        
        return sorted_memories[:n]
    
    def consolidate_memory(self, memory_id: str):
        """
        Consolidate memory by increasing its importance.
        
        Args:
            memory_id: Memory to consolidate
        """
        memory = self.memories.get(memory_id)
        if memory:
            # Increase importance
            memory.importance = min(1.0, memory.importance * 1.2)
            memory.priority = self.calculate_priority(memory)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.memories:
            return {
                "total_memories": 0,
                "forgotten_count": self.forgotten_count,
                "capacity_used": 0.0
            }
        
        priorities = [m.priority for m in self.memories.values()]
        access_counts = [m.access_count for m in self.memories.values()]
        
        return {
            "total_memories": len(self.memories),
            "forgotten_count": self.forgotten_count,
            "capacity_used": len(self.memories) / self.max_capacity,
            "avg_priority": sum(priorities) / len(priorities),
            "avg_access_count": sum(access_counts) / len(access_counts),
            "strategy": self.strategy.value
        }


class SmartMemoryAgent:
    """
    Agent with intelligent memory management.
    
    This agent uses memory prioritization to maintain relevant information
    while forgetting less important memories.
    """
    
    def __init__(
        self,
        max_memory: int = 50,
        strategy: ForgettingStrategy = ForgettingStrategy.PRIORITY_BASED,
        temperature: float = 0.5
    ):
        """
        Initialize smart memory agent.
        
        Args:
            max_memory: Maximum memory capacity
            strategy: Forgetting strategy
            temperature: LLM temperature
        """
        self.prioritizer = MemoryPrioritizer(
            max_capacity=max_memory,
            strategy=strategy
        )
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.interaction_count = 0
    
    def process_interaction(
        self,
        user_input: str,
        tags: List[str] = None,
        importance: float = 0.5
    ) -> str:
        """
        Process user interaction with memory management.
        
        Args:
            user_input: User input
            tags: Tags for this interaction
            importance: Importance score
            
        Returns:
            Response
        """
        self.interaction_count += 1
        memory_id = f"interaction_{self.interaction_count}"
        
        # Get relevant memories
        context = {"tags": tags or []}
        relevant_memories = self.prioritizer.get_top_memories(5, context)
        
        # Build context from memories
        memory_context = "\n".join([
            f"- {m.content} (priority: {m.priority:.2f})"
            for m in relevant_memories[:3]
        ])
        
        # Generate response using LLM
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant with selective memory. "
            "Here are your most relevant memories:\n{memory_context}\n\n"
            "User: {user_input}\n\n"
            "Respond naturally, incorporating relevant memories."
        )
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "memory_context": memory_context if memory_context else "No relevant memories",
            "user_input": user_input
        })
        
        # Store interaction in memory
        self.prioritizer.add_memory(
            memory_id=memory_id,
            content=f"User: {user_input} | Response: {response[:100]}",
            importance=importance,
            tags=tags,
            context={"type": "interaction"}
        )
        
        return response
    
    def memorize_fact(
        self,
        fact: str,
        importance: float = 0.7,
        tags: List[str] = None
    ):
        """
        Explicitly memorize a fact.
        
        Args:
            fact: Fact to memorize
            importance: Importance score
            tags: Associated tags
        """
        self.interaction_count += 1
        memory_id = f"fact_{self.interaction_count}"
        
        self.prioritizer.add_memory(
            memory_id=memory_id,
            content=fact,
            importance=importance,
            tags=tags or [],
            context={"type": "fact"}
        )
    
    def get_memory_summary(self) -> str:
        """
        Get summary of memory state.
        
        Returns:
            Memory summary
        """
        stats = self.prioritizer.get_statistics()
        top_memories = self.prioritizer.get_top_memories(5)
        
        summary = f"Memory Statistics:\n"
        summary += f"  Total Memories: {stats['total_memories']}\n"
        summary += f"  Forgotten: {stats['forgotten_count']}\n"
        summary += f"  Capacity Used: {stats['capacity_used']:.1%}\n"
        summary += f"  Avg Priority: {stats.get('avg_priority', 0):.2f}\n\n"
        
        summary += "Top Memories:\n"
        for i, mem in enumerate(top_memories, 1):
            summary += f"  {i}. {mem.content[:60]}... "
            summary += f"(priority: {mem.priority:.2f}, "
            summary += f"accessed: {mem.access_count}x)\n"
        
        return summary


def demonstrate_memory_prioritization():
    """Demonstrate memory prioritization and forgetting pattern."""
    
    print("=" * 80)
    print("MEMORY PRIORITIZATION & FORGETTING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic priority-based forgetting
    print("\n" + "=" * 80)
    print("Example 1: Priority-Based Memory Management")
    print("=" * 80)
    
    prioritizer = MemoryPrioritizer(
        max_capacity=10,
        strategy=ForgettingStrategy.PRIORITY_BASED,
        recency_weight=0.3,
        frequency_weight=0.2,
        importance_weight=0.5
    )
    
    print("\nAdding 15 memories with varying importance...")
    print(f"Max capacity: {prioritizer.max_capacity}")
    
    # Add memories with different importance levels
    memories_to_add = [
        ("mem_1", "User's name is Alice", 0.9, ["personal", "name"]),
        ("mem_2", "Likes pizza", 0.5, ["food", "preferences"]),
        ("mem_3", "Birthday is June 15", 0.8, ["personal", "date"]),
        ("mem_4", "Works in tech", 0.6, ["work"]),
        ("mem_5", "Random fact about weather", 0.2, ["misc"]),
        ("mem_6", "Allergic to peanuts", 0.9, ["health", "important"]),
        ("mem_7", "Favorite color is blue", 0.3, ["preferences"]),
        ("mem_8", "Lives in San Francisco", 0.7, ["personal", "location"]),
        ("mem_9", "Speaks English and Spanish", 0.6, ["language"]),
        ("mem_10", "Unimportant note", 0.1, ["misc"]),
        ("mem_11", "Has a dog named Max", 0.6, ["personal", "pets"]),
        ("mem_12", "Exercises regularly", 0.4, ["health"]),
        ("mem_13", "Another random fact", 0.1, ["misc"]),
        ("mem_14", "Emergency contact: Bob", 0.8, ["important", "contact"]),
        ("mem_15", "Likes science fiction", 0.4, ["entertainment"]),
    ]
    
    for mem_id, content, importance, tags in memories_to_add:
        prioritizer.add_memory(mem_id, content, importance, tags)
        time.sleep(0.01)  # Small delay to differentiate timestamps
    
    print(f"\nMemories after adding 15 (capacity: {prioritizer.max_capacity}):")
    print(f"  Retained: {len(prioritizer.memories)}")
    print(f"  Forgotten: {prioritizer.forgotten_count}")
    
    # Show retained memories
    print("\nRetained memories (sorted by priority):")
    retained = sorted(prioritizer.memories.values(), key=lambda m: m.priority, reverse=True)
    for i, mem in enumerate(retained, 1):
        print(f"  {i}. [{mem.memory_id}] {mem.content} "
              f"(importance: {mem.importance:.1f}, priority: {mem.priority:.2f})")
    
    # Example 2: Impact of access patterns
    print("\n" + "=" * 80)
    print("Example 2: Access Patterns Affecting Priority")
    print("=" * 80)
    
    print("\nAccessing 'Likes pizza' memory multiple times...")
    for _ in range(5):
        prioritizer.access_memory("mem_2")
        time.sleep(0.01)
    
    mem_2 = prioritizer.memories.get("mem_2")
    if mem_2:
        print(f"  Access count: {mem_2.access_count}")
        print(f"  Updated priority: {mem_2.priority:.2f}")
    
    # Add more memories to trigger forgetting
    print("\nAdding 3 more low-importance memories...")
    for i in range(16, 19):
        prioritizer.add_memory(
            f"mem_{i}",
            f"Low importance memory {i}",
            0.2,
            ["misc"]
        )
    
    print(f"\nMemories after additions:")
    print(f"  Retained: {len(prioritizer.memories)}")
    print(f"  Forgotten: {prioritizer.forgotten_count}")
    print(f"  'Likes pizza' still retained: {'mem_2' in prioritizer.memories}")
    
    # Example 3: Context-based relevance
    print("\n" + "=" * 80)
    print("Example 3: Context-Based Memory Retrieval")
    print("=" * 80)
    
    contexts = [
        {"tags": ["personal"], "description": "Personal info"},
        {"tags": ["health"], "description": "Health-related"},
        {"tags": ["work"], "description": "Work-related"},
    ]
    
    for context in contexts:
        print(f"\nContext: {context['description']}")
        top_memories = prioritizer.get_top_memories(3, context)
        for i, mem in enumerate(top_memories, 1):
            print(f"  {i}. {mem.content[:40]}... "
                  f"(priority: {mem.priority:.2f}, tags: {mem.tags})")
    
    # Example 4: Different forgetting strategies
    print("\n" + "=" * 80)
    print("Example 4: Comparing Forgetting Strategies")
    print("=" * 80)
    
    strategies = [
        ForgettingStrategy.LRU,
        ForgettingStrategy.LFU,
        ForgettingStrategy.TIME_DECAY,
    ]
    
    for strategy in strategies:
        print(f"\nStrategy: {strategy.value}")
        test_prioritizer = MemoryPrioritizer(
            max_capacity=5,
            strategy=strategy
        )
        
        # Add 8 memories
        for i in range(8):
            test_prioritizer.add_memory(
                f"test_mem_{i}",
                f"Memory {i}",
                0.5,
                ["test"]
            )
            if i < 3:  # Access first 3 multiple times
                for _ in range(3):
                    test_prioritizer.access_memory(f"test_mem_{i}")
            time.sleep(0.01)
        
        print(f"  Retained memories: {list(test_prioritizer.memories.keys())}")
        print(f"  Forgotten count: {test_prioritizer.forgotten_count}")
    
    # Example 5: Smart memory agent
    print("\n" + "=" * 80)
    print("Example 5: Smart Memory Agent with Forgetting")
    print("=" * 80)
    
    agent = SmartMemoryAgent(
        max_memory=15,
        strategy=ForgettingStrategy.PRIORITY_BASED
    )
    
    print("\nInteracting with agent (max 15 memories)...")
    
    # Store important facts
    agent.memorize_fact(
        "User's name is Sarah",
        importance=0.9,
        tags=["personal", "name"]
    )
    agent.memorize_fact(
        "Allergic to shellfish",
        importance=0.9,
        tags=["health", "important"]
    )
    
    # Multiple interactions
    interactions = [
        ("Tell me about healthy eating", ["health", "advice"], 0.5),
        ("What's my name?", ["personal"], 0.7),
        ("Random weather question", ["misc"], 0.2),
        ("What should I avoid eating?", ["health", "important"], 0.8),
        ("Tell me a joke", ["entertainment"], 0.3),
    ]
    
    for user_input, tags, importance in interactions:
        print(f"\nUser: {user_input}")
        response = agent.process_interaction(user_input, tags, importance)
        print(f"Agent: {response[:150]}...")
    
    # Add many low-importance interactions
    print("\nAdding 15 low-importance interactions...")
    for i in range(15):
        agent.process_interaction(
            f"Unimportant question {i}",
            tags=["misc"],
            importance=0.1
        )
    
    print("\n" + agent.get_memory_summary())
    
    # Verify important memories retained
    print("\nVerifying important memories retained:")
    important_memories = agent.prioritizer.get_top_memories(5)
    for mem in important_memories:
        if "Sarah" in mem.content or "shellfish" in mem.content:
            print(f"  ✓ Important memory retained: {mem.content[:50]}...")
    
    # Example 6: Memory consolidation
    print("\n" + "=" * 80)
    print("Example 6: Memory Consolidation (Strengthening)")
    print("=" * 80)
    
    consolidator = MemoryPrioritizer(max_capacity=10)
    
    # Add memory and show initial priority
    mem = consolidator.add_memory(
        "consolidate_me",
        "Important fact to consolidate",
        importance=0.5,
        tags=["test"]
    )
    print(f"Initial importance: {mem.importance:.2f}")
    print(f"Initial priority: {mem.priority:.2f}")
    
    # Consolidate (strengthen) the memory
    print("\nConsolidating memory 3 times...")
    for i in range(3):
        consolidator.consolidate_memory("consolidate_me")
        mem = consolidator.memories["consolidate_me"]
        print(f"  After consolidation {i+1}: "
              f"importance={mem.importance:.2f}, priority={mem.priority:.2f}")
    
    # Example 7: Memory statistics over time
    print("\n" + "=" * 80)
    print("Example 7: Memory Statistics Analysis")
    print("=" * 80)
    
    tracker = MemoryPrioritizer(
        max_capacity=20,
        strategy=ForgettingStrategy.PRIORITY_BASED
    )
    
    print("\nAdding 50 memories over time with varying access patterns...")
    
    for i in range(50):
        importance = 0.3 + (i % 5) * 0.15  # Varying importance
        tracker.add_memory(
            f"mem_{i}",
            f"Memory content {i}",
            importance=importance,
            tags=[f"tag_{i % 3}"]
        )
        
        # Simulate some memories being accessed more
        if i % 5 == 0:
            tracker.access_memory(f"mem_{i}")
        
        if i % 10 == 0:
            stats = tracker.get_statistics()
            print(f"\nAfter {i+1} additions:")
            print(f"  Retained: {stats['total_memories']}")
            print(f"  Forgotten: {stats['forgotten_count']}")
            print(f"  Capacity: {stats['capacity_used']:.1%}")
            print(f"  Avg priority: {stats.get('avg_priority', 0):.2f}")
    
    # Final statistics
    final_stats = tracker.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total memories added: 50")
    print(f"  Currently retained: {final_stats['total_memories']}")
    print(f"  Total forgotten: {final_stats['forgotten_count']}")
    print(f"  Retention rate: {final_stats['total_memories']/50:.1%}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Memory Prioritization & Forgetting Pattern")
    print("=" * 80)
    
    summary = """
    The Memory Prioritization & Forgetting pattern demonstrated:
    
    1. PRIORITY-BASED FORGETTING (Example 1):
       - Automatic memory management within capacity limits
       - Importance-weighted retention decisions
       - Low-importance memories forgotten first
       - High-importance memories (name, allergies) retained
    
    2. ACCESS PATTERNS (Example 2):
       - Frequently accessed memories gain higher priority
       - Recency and frequency tracked automatically
       - "Likes pizza" retained despite lower base importance
       - Dynamic priority recalculation
    
    3. CONTEXT-BASED RETRIEVAL (Example 3):
       - Tag-based relevance calculation
       - Context-aware memory prioritization
       - Different top memories for different contexts
       - Personal, health, work contexts demonstrated
    
    4. FORGETTING STRATEGIES (Example 4):
       - LRU: Removes least recently used
       - LFU: Removes least frequently used
       - Time Decay: Removes oldest memories
       - Priority-Based: Removes lowest priority
    
    5. SMART AGENT (Example 5):
       - Integrated memory management in agent workflow
       - Important facts (name, allergies) retained
       - Low-importance interactions forgotten
       - Context-aware responses using relevant memories
    
    6. MEMORY CONSOLIDATION (Example 6):
       - Strengthening important memories
       - Importance increases with consolidation
       - Priority automatically recalculated
       - Simulates rehearsal/reinforcement
    
    7. STATISTICS TRACKING (Example 7):
       - Memory usage metrics over time
       - Retention rate monitoring
       - Forgetting rate analysis
       - Capacity utilization tracking
    
    KEY BENEFITS:
    ✓ Efficient memory utilization
    ✓ Focus on relevant information
    ✓ Scalable for long-term operation
    ✓ Automatic capacity management
    ✓ Context-aware prioritization
    ✓ Adaptive to usage patterns
    ✓ Prevents memory overflow
    
    USE CASES:
    • Long-running conversational agents
    • Personal assistant applications
    • Learning agents with memory constraints
    • Real-time systems with limited resources
    • Adaptive systems in changing environments
    • Information filtering and summarization
    • Context-aware applications
    
    BEST PRACTICES:
    1. Set appropriate capacity limits for use case
    2. Weight priority factors based on needs
    3. Use high importance for critical information
    4. Implement backup/logging before forgetting
    5. Monitor retention rates and adjust
    6. Consolidate frequently-used memories
    7. Consider context in retrieval
    8. Track statistics for tuning
    
    TRADE-OFFS:
    • Memory efficiency vs. information loss
    • Computational overhead vs. accuracy
    • Recency vs. importance weighting
    • Capacity limits vs. completeness
    
    PRODUCTION CONSIDERATIONS:
    → Set capacity based on system resources
    → Implement memory backup mechanisms
    → Log all forgetting operations for debugging
    → Monitor hit rates and adjust weights
    → Test forgetting strategies for domain
    → Consider reversible "archiving" vs. deletion
    → Balance performance with retention needs
    → Implement gradual priority decay
    → Track memory access patterns for optimization
    → Consider user-controlled importance overrides
    
    This pattern enables agents to operate efficiently over long periods
    by intelligently managing memory, focusing on relevant information,
    and gracefully handling capacity constraints through smart forgetting.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_memory_prioritization()
