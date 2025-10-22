"""
RAG (Retrieval-Augmented Generation) and Memory Patterns
========================================================
Demonstrates information retrieval and memory management patterns.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import math


@dataclass
class Document:
    """Represents a document in the knowledge base"""
    id: str
    content: str
    metadata: Dict
    embedding: Optional[List[float]] = None


# ============================================================================
# 1. RAG PATTERN
# ============================================================================

class VectorStore:
    """Simple vector store for document retrieval"""
    
    def __init__(self):
        self.documents: List[Document] = []
    
    def add_document(self, doc: Document):
        """Add document to store"""
        # Generate simple embedding (in reality, use proper embedding model)
        doc.embedding = self._generate_embedding(doc.content)
        self.documents.append(doc)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate simple embedding based on word frequencies"""
        words = text.lower().split()
        # Simple bag-of-words representation (normalized)
        vocab = set(word for doc in self.documents for word in doc.content.lower().split())
        vocab.update(words)
        
        embedding = []
        for word in sorted(vocab):
            count = words.count(word)
            embedding.append(count / len(words) if words else 0)
        
        return embedding[:10]  # Truncate for simplicity
    
    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """Search for relevant documents"""
        query_embedding = self._generate_embedding(query)
        
        # Calculate similarity scores
        scored_docs = []
        for doc in self.documents:
            similarity = self._cosine_similarity(query_embedding, doc.embedding or [])
            scored_docs.append((similarity, doc))
        
        # Sort by similarity and return top_k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_docs[:top_k]]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        # Pad shorter vector
        max_len = max(len(vec1), len(vec2))
        vec1 = vec1 + [0] * (max_len - len(vec1))
        vec2 = vec2 + [0] * (max_len - len(vec2))
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class RAGAgent:
    """Agent that uses Retrieval-Augmented Generation"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def answer_question(self, question: str) -> Dict:
        """Answer question using RAG pattern"""
        print(f"\n{'='*70}")
        print(f"Question: {question}")
        print(f"{'='*70}\n")
        
        # Step 1: Retrieve relevant documents
        print("🔍 Retrieving relevant documents...")
        relevant_docs = self.vector_store.search(question, top_k=3)
        
        print(f"Found {len(relevant_docs)} relevant documents:\n")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"{i}. {doc.metadata.get('title', 'Untitled')}")
            print(f"   {doc.content[:100]}...")
            print()
        
        # Step 2: Generate answer using retrieved context
        print("💭 Generating answer from retrieved context...\n")
        answer = self._generate_answer(question, relevant_docs)
        
        print(f"{'─'*70}")
        print(f"Answer: {answer}")
        print(f"{'─'*70}")
        
        # Step 3: Cite sources
        print(f"\nSources:")
        for doc in relevant_docs:
            print(f"  • {doc.metadata.get('title', 'Untitled')} ({doc.metadata.get('source', 'Unknown')})")
        
        return {
            "question": question,
            "answer": answer,
            "sources": relevant_docs,
            "num_sources_used": len(relevant_docs)
        }
    
    def _generate_answer(self, question: str, documents: List[Document]) -> str:
        """Generate answer based on retrieved documents"""
        # Combine document contents
        context = " ".join(doc.content for doc in documents)
        
        # Simple answer generation (in reality, use LLM)
        if "python" in question.lower():
            return (
                "Python is a high-level programming language created by Guido van Rossum. "
                "It emphasizes code readability and supports multiple programming paradigms. "
                "Python is widely used in web development, data science, artificial intelligence, "
                "and automation due to its simplicity and extensive library ecosystem."
            )
        elif "ai" in question.lower() or "artificial intelligence" in question.lower():
            return (
                "Artificial Intelligence (AI) refers to the simulation of human intelligence by machines. "
                "It encompasses various techniques including machine learning, natural language processing, "
                "and computer vision. AI systems can perform tasks that typically require human intelligence, "
                "such as visual perception, speech recognition, decision-making, and language translation."
            )
        elif "react" in question.lower() and "pattern" in question.lower():
            return (
                "The ReAct pattern combines reasoning and acting in AI agents. "
                "It follows a Thought → Action → Observation loop where the agent alternates between "
                "reasoning about the task and taking actions. This approach enables interpretable "
                "decision-making and dynamic adjustment based on observations."
            )
        else:
            return f"Based on the retrieved documents: {context[:200]}..."


# ============================================================================
# 2. MEMORY MANAGEMENT PATTERNS
# ============================================================================

class ShortTermMemory:
    """Maintains context within current conversation"""
    
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages: List[Dict] = []
    
    def add_message(self, role: str, content: str):
        """Add message to memory"""
        self.messages.append({"role": role, "content": content})
        
        # Keep only recent messages (sliding window)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self) -> str:
        """Get current conversation context"""
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.messages)
    
    def clear(self):
        """Clear memory"""
        self.messages = []


class LongTermMemory:
    """Stores information across sessions"""
    
    def __init__(self):
        self.episodic_memory: List[Dict] = []  # Events/experiences
        self.semantic_memory: Dict[str, any] = {}  # Facts/knowledge
        self.procedural_memory: Dict[str, List[str]] = {}  # Skills/procedures
    
    def store_episode(self, episode: Dict):
        """Store an episodic memory"""
        self.episodic_memory.append(episode)
    
    def store_fact(self, key: str, value: any):
        """Store semantic knowledge"""
        self.semantic_memory[key] = value
    
    def store_procedure(self, name: str, steps: List[str]):
        """Store procedural knowledge"""
        self.procedural_memory[name] = steps
    
    def recall_episodes(self, query: str, limit: int = 5) -> List[Dict]:
        """Recall relevant episodes"""
        # Simple keyword matching
        relevant = []
        for episode in self.episodic_memory[-limit:]:
            if query.lower() in str(episode).lower():
                relevant.append(episode)
        return relevant
    
    def recall_fact(self, key: str) -> Optional[any]:
        """Recall a fact"""
        return self.semantic_memory.get(key)
    
    def recall_procedure(self, name: str) -> Optional[List[str]]:
        """Recall a procedure"""
        return self.procedural_memory.get(name)


class MemoryAgentWithMemory:
    """Agent with both short-term and long-term memory"""
    
    def __init__(self):
        self.short_term = ShortTermMemory(max_messages=5)
        self.long_term = LongTermMemory()
    
    def process_interaction(self, user_input: str) -> str:
        """Process user interaction using memory"""
        print(f"\nUser: {user_input}")
        
        # Add to short-term memory
        self.short_term.add_message("user", user_input)
        
        # Check long-term memory for relevant information
        relevant_facts = self._check_longterm_memory(user_input)
        
        # Generate response considering both memories
        response = self._generate_response(user_input, relevant_facts)
        
        # Add response to short-term memory
        self.short_term.add_message("assistant", response)
        
        # Store important information in long-term memory
        self._update_longterm_memory(user_input, response)
        
        print(f"Assistant: {response}")
        
        return response
    
    def _check_longterm_memory(self, input_text: str) -> List:
        """Check long-term memory for relevant information"""
        relevant = []
        
        # Check for user preferences
        if "name" in input_text.lower():
            name = self.long_term.recall_fact("user_name")
            if name:
                relevant.append(f"User's name is {name}")
        
        # Check for past episodes
        episodes = self.long_term.recall_episodes(input_text, limit=3)
        relevant.extend([f"Recalled: {ep}" for ep in episodes])
        
        return relevant
    
    def _generate_response(self, input_text: str, relevant_facts: List) -> str:
        """Generate response using memory"""
        if "my name is" in input_text.lower():
            name = input_text.lower().split("my name is")[-1].strip().rstrip(".")
            return f"Nice to meet you, {name}! I'll remember that."
        
        elif "what is my name" in input_text.lower() or "do you remember my name" in input_text.lower():
            name = self.long_term.recall_fact("user_name")
            if name:
                return f"Yes, your name is {name}."
            else:
                return "I don't believe you've told me your name yet."
        
        elif "what did we discuss" in input_text.lower():
            context = self.short_term.get_context()
            if context:
                return f"We discussed: {context[:100]}..."
            return "We just started our conversation."
        
        else:
            # Use short-term context
            recent_context = self.short_term.messages[-3:] if len(self.short_term.messages) > 0 else []
            if recent_context:
                return f"Based on our conversation, I can help you with that."
            return "I'm here to help! What would you like to know?"
    
    def _update_longterm_memory(self, user_input: str, response: str):
        """Update long-term memory with important information"""
        # Store user name
        if "my name is" in user_input.lower():
            name = user_input.lower().split("my name is")[-1].strip().rstrip(".")
            self.long_term.store_fact("user_name", name)
        
        # Store episode
        self.long_term.store_episode({
            "timestamp": "2025-10-22",  # Would use actual timestamp
            "user_input": user_input,
            "response": response
        })
    
    def show_memory_state(self):
        """Display current memory state"""
        print(f"\n{'='*70}")
        print("MEMORY STATE")
        print(f"{'='*70}")
        
        print(f"\nShort-Term Memory ({len(self.short_term.messages)} messages):")
        for msg in self.short_term.messages:
            print(f"  {msg['role']}: {msg['content'][:50]}...")
        
        print(f"\nLong-Term Memory:")
        print(f"  Facts: {len(self.long_term.semantic_memory)} stored")
        for key, value in list(self.long_term.semantic_memory.items())[:5]:
            print(f"    • {key}: {value}")
        
        print(f"  Episodes: {len(self.long_term.episodic_memory)} stored")
        for ep in self.long_term.episodic_memory[-3:]:
            print(f"    • {ep.get('timestamp', 'N/A')}: {ep.get('user_input', 'N/A')[:40]}...")


def main():
    """Demonstrate RAG and Memory patterns"""
    
    # ========================================================================
    # EXAMPLE 1: RAG PATTERN
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 1: RAG (RETRIEVAL-AUGMENTED GENERATION) PATTERN")
    print("="*70)
    
    # Create knowledge base
    vector_store = VectorStore()
    
    documents = [
        Document(
            id="1",
            content="Python is a high-level, interpreted programming language created by Guido van Rossum in 1991. It emphasizes code readability with significant whitespace.",
            metadata={"title": "Python Programming", "source": "Tech Encyclopedia"}
        ),
        Document(
            id="2",
            content="Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.",
            metadata={"title": "Introduction to AI", "source": "AI Textbook"}
        ),
        Document(
            id="3",
            content="The ReAct pattern in AI agents combines reasoning and acting. The agent alternates between thinking about the task and taking actions, creating a Thought-Action-Observation loop.",
            metadata={"title": "ReAct Pattern", "source": "AI Design Patterns"}
        ),
        Document(
            id="4",
            content="Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn.",
            metadata={"title": "Machine Learning Basics", "source": "ML Course"}
        ),
        Document(
            id="5",
            content="Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics.",
            metadata={"title": "NLP Overview", "source": "NLP Guide"}
        )
    ]
    
    for doc in documents:
        vector_store.add_document(doc)
    
    # Create RAG agent
    rag_agent = RAGAgent(vector_store)
    
    # Ask questions
    rag_agent.answer_question("What is Python and who created it?")
    print("\n")
    rag_agent.answer_question("Explain the ReAct pattern in AI agents")
    
    # ========================================================================
    # EXAMPLE 2: MEMORY PATTERNS
    # ========================================================================
    print("\n\n" + "="*70)
    print("EXAMPLE 2: MEMORY MANAGEMENT PATTERNS")
    print("="*70)
    
    agent = MemoryAgentWithMemory()
    
    # Simulate conversation
    print(f"\n{'─'*70}")
    print("Conversation with Memory Agent")
    print(f"{'─'*70}")
    
    agent.process_interaction("Hello! My name is Alice.")
    agent.process_interaction("I'm interested in learning about AI.")
    agent.process_interaction("Can you recommend some resources?")
    agent.process_interaction("What is my name?")
    agent.process_interaction("What did we discuss earlier?")
    
    # Show memory state
    agent.show_memory_state()
    
    # Summary
    print(f"\n\n{'='*70}")
    print("KEY PATTERNS SUMMARY")
    print(f"{'='*70}")
    
    print("\n1. RAG (Retrieval-Augmented Generation):")
    print("   ✓ Retrieves relevant information before generating")
    print("   ✓ Grounds responses in factual sources")
    print("   ✓ Provides citations and transparency")
    print("   ✓ Reduces hallucinations")
    
    print("\n2. Memory Management:")
    print("   ✓ Short-term: Maintains conversation context")
    print("   ✓ Long-term: Stores facts, experiences, and procedures")
    print("   ✓ Enables personalization and learning")
    print("   ✓ Supports coherent multi-turn interactions")


if __name__ == "__main__":
    main()
