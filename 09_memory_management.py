"""
Memory Management Patterns
Short-term, long-term, and working memory
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
@dataclass
class Memory:
    content: str
    timestamp: datetime
    memory_type: str  # 'episodic', 'semantic', 'procedural'
    importance: float = 0.5
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
class ShortTermMemory:
    """Buffer for recent context - limited capacity"""
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
    def add(self, memory: Memory):
        """Add to short-term memory"""
        self.buffer.append(memory)
    def get_recent(self, n: int = None) -> List[Memory]:
        """Get n most recent memories"""
        if n is None:
            return list(self.buffer)
        return list(self.buffer)[-n:]
    def clear(self):
        """Clear short-term memory"""
        self.buffer.clear()
class LongTermMemory:
    """Persistent storage for important information"""
    def __init__(self):
        self.memories: List[Memory] = []
    def add(self, memory: Memory):
        """Add to long-term memory"""
        self.memories.append(memory)
    def search(self, query: str, top_k: int = 5) -> List[Memory]:
        """Search long-term memory"""
        # Simple keyword search (in reality, use embeddings)
        query_words = set(query.lower().split())
        results = []
        for memory in self.memories:
            memory_words = set(memory.content.lower().split())
            overlap = len(query_words & memory_words)
            if overlap > 0:
                # Update access stats
                memory.access_count += 1
                memory.last_access = datetime.now()
                results.append((memory, overlap))
        # Sort by relevance and importance
        results.sort(key=lambda x: (x[1], x[0].importance), reverse=True)
        return [mem for mem, _ in results[:top_k]]
    def consolidate(self, short_term: ShortTermMemory, importance_threshold: float = 0.6):
        """Move important memories from short-term to long-term"""
        for memory in short_term.get_recent():
            if memory.importance >= importance_threshold:
                self.add(memory)
                print(f"Consolidated to long-term: {memory.content[:50]}...")
    def forget(self, max_age_days: int = 30, min_importance: float = 0.3):
        """Remove old, unimportant memories"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        before_count = len(self.memories)
        self.memories = [
            m for m in self.memories
            if m.last_access > cutoff_date or m.importance > min_importance
        ]
        after_count = len(self.memories)
        forgotten = before_count - after_count
        if forgotten > 0:
            print(f"Forgot {forgotten} old/unimportant memories")
class WorkingMemory:
    """Active workspace for current task"""
    def __init__(self):
        self.current_goal: Optional[str] = None
        self.active_context: Dict[str, Any] = {}
        self.intermediate_results: List[Any] = []
    def set_goal(self, goal: str):
        """Set current goal"""
        self.current_goal = goal
        self.active_context = {"goal": goal}
    def update_context(self, key: str, value: Any):
        """Update active context"""
        self.active_context[key] = value
    def add_result(self, result: Any):
        """Add intermediate result"""
        self.intermediate_results.append(result)
    def clear(self):
        """Clear working memory"""
        self.current_goal = None
        self.active_context.clear()
        self.intermediate_results.clear()
    def get_state(self) -> Dict[str, Any]:
        """Get current working memory state"""
        return {
            "goal": self.current_goal,
            "context": self.active_context,
            "results_count": len(self.intermediate_results)
        }
class MemoryAgent:
    """Agent with comprehensive memory system"""
    def __init__(self):
        self.short_term = ShortTermMemory(max_size=10)
        self.long_term = LongTermMemory()
        self.working = WorkingMemory()
    def perceive(self, information: str, importance: float = 0.5, memory_type: str = "episodic"):
        """Process new information"""
        memory = Memory(
            content=information,
            timestamp=datetime.now(),
            memory_type=memory_type,
            importance=importance
        )
        # Add to short-term memory
        self.short_term.add(memory)
        # If important, also add to long-term
        if importance > 0.7:
            self.long_term.add(memory)
        print(f"Perceived: {information[:50]}... (importance: {importance})")
    def recall(self, query: str) -> List[Memory]:
        """Recall relevant memories"""
        print(f"\nRecalling memories for: {query}")
        # Search long-term memory
        lt_memories = self.long_term.search(query, top_k=3)
        # Get recent short-term memories
        st_memories = self.short_term.get_recent(5)
        # Combine and deduplicate
        all_memories = lt_memories + [m for m in st_memories if m not in lt_memories]
        print(f"Found {len(all_memories)} relevant memories")
        return all_memories
    def think(self, task: str):
        """Process a task using working memory"""
        print(f"\n--- Thinking about task ---")
        print(f"Task: {task}")
        # Set up working memory
        self.working.set_goal(task)
        # Recall relevant information
        relevant_memories = self.recall(task)
        # Use memories in working memory
        self.working.update_context("relevant_memories", len(relevant_memories))
        for i, memory in enumerate(relevant_memories):
            print(f"\nUsing memory {i+1}: {memory.content[:60]}...")
            self.working.add_result(f"Processed memory {i+1}")
        # Simulate thinking process
        conclusion = f"Completed analysis of '{task}' using {len(relevant_memories)} memories"
        self.working.add_result(conclusion)
        print(f"\nWorking Memory State: {self.working.get_state()}")
        return conclusion
    def consolidate_memories(self):
        """Consolidate memories from short-term to long-term"""
        print("\n--- Consolidating Memories ---")
        self.long_term.consolidate(self.short_term, importance_threshold=0.6)
    def sleep(self):
        """Simulate sleep - consolidate and clean memories"""
        print("\n=== Sleep Cycle ===")
        self.consolidate_memories()
        self.long_term.forget(max_age_days=30, min_importance=0.3)
        self.working.clear()
        print("Sleep cycle complete\n")
# Usage
if __name__ == "__main__":
    agent = MemoryAgent()
    print("="*70)
    print("MEMORY MANAGEMENT DEMONSTRATION")
    print("="*70)
    # Day 1: Learning
    print("\n--- Day 1: Learning Phase ---")
    agent.perceive("Python is a programming language", importance=0.8, memory_type="semantic")
    agent.perceive("I wrote a hello world program today", importance=0.5, memory_type="episodic")
    agent.perceive("Functions in Python use the 'def' keyword", importance=0.9, memory_type="semantic")
    agent.perceive("Had lunch at 12pm", importance=0.2, memory_type="episodic")
    agent.perceive("Classes in Python use the 'class' keyword", importance=0.9, memory_type="semantic")
    # Task execution
    agent.think("How do I define a function in Python?")
    # End of day
    agent.sleep()
    # Day 2: More learning
    print("\n--- Day 2: Continued Learning ---")
    agent.perceive("Python supports object-oriented programming", importance=0.8, memory_type="semantic")
    agent.perceive("Debugged a tricky issue today", importance=0.6, memory_type="episodic")
    # Another task
    agent.think("What do I know about Python programming?")
    # Memory stats
    print("\n" + "="*70)
    print("MEMORY STATISTICS")
    print("="*70)
    print(f"Short-term memories: {len(agent.short_term.buffer)}")
    print(f"Long-term memories: {len(agent.long_term.memories)}")
    print("\n--- Long-term Memory Contents ---")
    for i, mem in enumerate(agent.long_term.memories, 1):
        print(f"{i}. [{mem.memory_type}] {mem.content}")
        print(f"   Importance: {mem.importance}, Access count: {mem.access_count}")
