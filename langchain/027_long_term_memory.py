"""
Pattern 027: Long-Term Memory

Description:
    The Long-Term Memory pattern enables agents to store and retrieve information
    across multiple sessions and conversations. Unlike short-term memory that exists
    only within a session, long-term memory persists indefinitely, allowing agents
    to remember user preferences, past interactions, learned facts, and accumulated
    knowledge over time.

Components:
    - Persistent Storage: Vector databases, traditional databases, file systems
    - Memory Indexing: Efficient retrieval mechanisms
    - Memory Types: Episodic (events), Semantic (facts), Procedural (skills)
    - Retrieval System: Similarity search, keyword matching
    - Memory Consolidation: Organizing and optimizing stored memories

Use Cases:
    - Personalized user experiences
    - Learning from past interactions
    - Knowledge accumulation
    - User preference tracking
    - Long-term relationship building

LangChain Implementation:
    Uses vector stores (Chroma, FAISS) for semantic memory storage,
    implements retrieval strategies, and combines with conversation
    context for personalized interactions.

Key Features:
    - Cross-session persistence
    - Semantic similarity search
    - Memory categorization
    - Efficient retrieval
    - Memory aging and importance
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


class MemoryType(Enum):
    """Types of long-term memories."""
    EPISODIC = "episodic"  # Specific events and experiences
    SEMANTIC = "semantic"  # Facts and knowledge
    PROCEDURAL = "procedural"  # Skills and procedures
    PREFERENCE = "preference"  # User preferences
    INTERACTION = "interaction"  # Past interactions


@dataclass
class Memory:
    """Represents a long-term memory."""
    id: str
    content: str
    memory_type: MemoryType
    timestamp: datetime
    importance: float = 1.0  # 0-1 scale
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            importance=data.get("importance", 1.0),
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None
        )


class SimpleVectorStore:
    """
    Simple vector store for semantic memory storage (mock implementation).
    In production, would use Chroma, FAISS, or Pinecone.
    """
    
    def __init__(self):
        """Initialize vector store."""
        self.memories: List[Memory] = []
        self.embeddings_model = None  # Would use OpenAIEmbeddings() in production
    
    def add_memory(self, memory: Memory):
        """Add a memory to the store."""
        self.memories.append(memory)
    
    def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        k: int = 5
    ) -> List[Memory]:
        """
        Search for relevant memories.
        
        Args:
            query: Search query
            memory_type: Filter by memory type
            k: Number of results
            
        Returns:
            List of relevant memories
        """
        # Filter by type if specified
        candidates = self.memories
        if memory_type:
            candidates = [m for m in candidates if m.memory_type == memory_type]
        
        # Simple keyword matching (in production, would use vector similarity)
        query_lower = query.lower()
        scored_memories = []
        
        for memory in candidates:
            # Simple scoring based on keyword matches
            content_lower = memory.content.lower()
            score = 0.0
            
            for word in query_lower.split():
                if word in content_lower:
                    score += 1.0
            
            # Boost by importance and recency
            score *= memory.importance
            
            days_old = (datetime.now() - memory.timestamp).days
            recency_factor = 1.0 / (1.0 + days_old * 0.01)
            score *= recency_factor
            
            if score > 0:
                scored_memories.append((score, memory))
        
        # Sort by score and return top k
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        results = [memory for _, memory in scored_memories[:k]]
        
        # Update access statistics
        for memory in results:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
        
        return results
    
    def get_all_memories(self) -> List[Memory]:
        """Get all memories."""
        return self.memories.copy()
    
    def delete_memory(self, memory_id: str):
        """Delete a memory by ID."""
        self.memories = [m for m in self.memories if m.id != memory_id]


class PersistentMemoryStore:
    """
    Persistent storage for long-term memory (file-based mock).
    In production, would use proper database.
    """
    
    def __init__(self, storage_path: str = "memory_store.json"):
        """
        Initialize persistent store.
        
        Args:
            storage_path: Path to storage file
        """
        self.storage_path = Path(storage_path)
        self.vector_store = SimpleVectorStore()
        self._load()
    
    def _load(self):
        """Load memories from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for memory_data in data:
                        memory = Memory.from_dict(memory_data)
                        self.vector_store.add_memory(memory)
            except Exception as e:
                print(f"Warning: Could not load memories: {e}")
    
    def _save(self):
        """Save memories to disk."""
        try:
            data = [m.to_dict() for m in self.vector_store.get_all_memories()]
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save memories: {e}")
    
    def add_memory(self, memory: Memory):
        """Add and persist a memory."""
        self.vector_store.add_memory(memory)
        self._save()
    
    def search(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        k: int = 5
    ) -> List[Memory]:
        """Search for relevant memories."""
        return self.vector_store.search(query, memory_type, k)
    
    def get_all_memories(self) -> List[Memory]:
        """Get all stored memories."""
        return self.vector_store.get_all_memories()
    
    def clear(self):
        """Clear all memories."""
        self.vector_store.memories.clear()
        self._save()


class LongTermMemoryAgent:
    """
    Agent with long-term memory capabilities.
    """
    
    def __init__(
        self,
        user_id: str,
        storage_path: Optional[str] = None,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize agent with long-term memory.
        
        Args:
            user_id: Unique user identifier
            storage_path: Path to memory storage
            model: LLM model to use
        """
        self.user_id = user_id
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        
        # Initialize memory store
        if storage_path is None:
            storage_path = f"memory_{user_id}.json"
        self.memory_store = PersistentMemoryStore(storage_path)
        
        self.interaction_count = 0
    
    def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float = 1.0,
        metadata: Optional[Dict] = None
    ):
        """
        Store a new memory.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            importance: Importance score (0-1)
            metadata: Additional metadata
        """
        memory = Memory(
            id=f"{self.user_id}_{datetime.now().timestamp()}",
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {}
        )
        self.memory_store.add_memory(memory)
    
    def recall_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        k: int = 5
    ) -> List[Memory]:
        """
        Recall relevant memories.
        
        Args:
            query: Query for memory retrieval
            memory_type: Filter by type
            k: Number of memories to retrieve
            
        Returns:
            List of relevant memories
        """
        return self.memory_store.search(query, memory_type, k)
    
    def chat(self, user_message: str) -> str:
        """
        Chat with memory-augmented responses.
        
        Args:
            user_message: User's message
            
        Returns:
            Agent's response
        """
        self.interaction_count += 1
        
        # Recall relevant memories
        relevant_memories = self.recall_memories(user_message, k=3)
        
        # Format memories for context
        memory_context = ""
        if relevant_memories:
            memory_lines = []
            for mem in relevant_memories:
                memory_lines.append(
                    f"[{mem.memory_type.value}] {mem.content} "
                    f"(from {mem.timestamp.strftime('%Y-%m-%d')})"
                )
            memory_context = "\n".join(memory_lines)
        
        # Create prompt with memory context
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with long-term memory.
Use the relevant memories to personalize your responses and maintain continuity
across conversations."""),
            ("user", """Relevant Memories:
{memory_context}

Current Message: {message}

Provide a helpful, personalized response:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "memory_context": memory_context if memory_context else "No relevant memories found.",
            "message": user_message
        })
        
        # Store this interaction
        self.store_memory(
            content=f"User: {user_message} | AI: {response}",
            memory_type=MemoryType.INTERACTION,
            importance=0.5
        )
        
        # Extract and store important information
        self._extract_and_store_info(user_message)
        
        return response.strip()
    
    def _extract_and_store_info(self, message: str):
        """Extract and store important information from message."""
        # Simple extraction logic (in production, would use NER or LLM)
        message_lower = message.lower()
        
        # Detect preferences
        if any(word in message_lower for word in ["like", "prefer", "favorite", "love"]):
            self.store_memory(
                content=message,
                memory_type=MemoryType.PREFERENCE,
                importance=0.8
            )
        
        # Detect facts
        if any(word in message_lower for word in ["my name is", "i am", "i work", "i live"]):
            self.store_memory(
                content=message,
                memory_type=MemoryType.SEMANTIC,
                importance=0.9
            )
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of stored memories."""
        all_memories = self.memory_store.get_all_memories()
        
        memory_by_type = {}
        for mem_type in MemoryType:
            memory_by_type[mem_type.value] = sum(
                1 for m in all_memories if m.memory_type == mem_type
            )
        
        total_accesses = sum(m.access_count for m in all_memories)
        
        return {
            "user_id": self.user_id,
            "total_memories": len(all_memories),
            "by_type": memory_by_type,
            "total_accesses": total_accesses,
            "interactions": self.interaction_count
        }


def demonstrate_long_term_memory():
    """Demonstrate the Long-Term Memory pattern."""
    
    print("=" * 80)
    print("LONG-TERM MEMORY PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Create agent for user
    user_id = "alice_demo"
    agent = LongTermMemoryAgent(user_id=user_id, storage_path=f"memory_{user_id}_demo.json")
    
    # Clear any existing memories
    agent.memory_store.clear()
    
    # Session 1: Initial interaction
    print("\n" + "=" * 80)
    print("SESSION 1: Initial Interaction")
    print("=" * 80)
    
    messages_session1 = [
        "Hi, my name is Alice and I'm a software engineer.",
        "I really love Python programming and machine learning.",
        "My favorite food is sushi."
    ]
    
    for msg in messages_session1:
        print(f"\n👤 User: {msg}")
        response = agent.chat(msg)
        print(f"🤖 Agent: {response}")
    
    print("\n📊 Memory Summary after Session 1:")
    summary = agent.get_memory_summary()
    print(json.dumps(summary, indent=2))
    
    # Simulate new session (create new agent instance)
    print("\n\n" + "=" * 80)
    print("SESSION 2: Later Conversation (Testing Persistence)")
    print("=" * 80)
    print("(Simulating new session - creating fresh agent instance)")
    
    agent2 = LongTermMemoryAgent(user_id=user_id, storage_path=f"memory_{user_id}_demo.json")
    
    messages_session2 = [
        "What's my name?",
        "What do I do for work?",
        "What are my interests?",
        "What food do I like?"
    ]
    
    for msg in messages_session2:
        print(f"\n👤 User: {msg}")
        response = agent2.chat(msg)
        print(f"🤖 Agent: {response}")
    
    print("\n📊 Memory Summary after Session 2:")
    summary2 = agent2.get_memory_summary()
    print(json.dumps(summary2, indent=2))
    
    # Session 3: Memory accumulation
    print("\n\n" + "=" * 80)
    print("SESSION 3: Adding More Memories")
    print("=" * 80)
    
    agent3 = LongTermMemoryAgent(user_id=user_id, storage_path=f"memory_{user_id}_demo.json")
    
    messages_session3 = [
        "I recently started learning about transformers.",
        "I'm working on a project about recommendation systems.",
        "Can you suggest something based on my interests?"
    ]
    
    for msg in messages_session3:
        print(f"\n👤 User: {msg}")
        response = agent3.chat(msg)
        print(f"🤖 Agent: {response}")
    
    # Show memory details
    print("\n\n" + "=" * 80)
    print("STORED MEMORIES ANALYSIS")
    print("=" * 80)
    
    all_memories = agent3.memory_store.get_all_memories()
    
    print(f"\nTotal memories: {len(all_memories)}")
    
    for mem_type in MemoryType:
        type_memories = [m for m in all_memories if m.memory_type == mem_type]
        if type_memories:
            print(f"\n{mem_type.value.upper()} Memories ({len(type_memories)}):")
            for mem in type_memories[:3]:  # Show first 3
                print(f"  - {mem.content[:80]}...")
                print(f"    Importance: {mem.importance:.1f}, Accessed: {mem.access_count} times")
    
    # Clean up demo file
    try:
        os.remove(f"memory_{user_id}_demo.json")
    except:
        pass
    
    # Summary
    print("\n\n" + "=" * 80)
    print("PATTERN SUMMARY")
    print("=" * 80)
    print("""
The Long-Term Memory pattern demonstrates:

1. **Cross-Session Persistence**: Memories survive across sessions
2. **Memory Categorization**: Different types (episodic, semantic, preference)
3. **Semantic Retrieval**: Find relevant memories via similarity
4. **Memory Statistics**: Track access patterns and importance
5. **Personalization**: Use past memories for context-aware responses

Memory Types:

Episodic Memory:
- Stores specific events and experiences
- "When I visited Paris..."
- Time and context-stamped

Semantic Memory:
- Stores facts and knowledge
- "Paris is the capital of France"
- General information

Procedural Memory:
- Stores skills and procedures
- "How to make coffee"
- Step-by-step knowledge

Preference Memory:
- Stores user preferences
- "I prefer dark mode"
- Likes and dislikes

Interaction Memory:
- Stores past conversations
- Complete interaction history
- Enables continuity

Key Benefits:
- **Personalization**: Tailored responses based on history
- **Continuity**: Maintains context across sessions
- **Learning**: Accumulates knowledge over time
- **Relationship**: Builds long-term user relationships
- **Efficiency**: Doesn't repeat questions

Implementation Considerations:
- Vector databases for semantic search (Chroma, Pinecone, FAISS)
- Memory importance and decay
- Privacy and data retention
- Storage optimization
- Cross-device synchronization

Use Cases:
- Personal assistants with memory of user
- Educational tutors tracking student progress
- Healthcare agents remembering patient history
- Customer service with account context
- Long-term collaborative projects

This pattern transforms agents from stateless responders to
persistent entities with accumulated knowledge and relationships.
""")


if __name__ == "__main__":
    demonstrate_long_term_memory()
