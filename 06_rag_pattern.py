"""
Retrieval-Augmented Generation (RAG) Pattern
Retrieves relevant information before generating response
"""
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
@dataclass
class Document:
    id: str
    content: str
    metadata: Dict
    embedding: np.ndarray = None
class SimpleVectorStore:
    """Simple in-memory vector store"""
    def __init__(self):
        self.documents: List[Document] = []
    def add_document(self, doc: Document):
        """Add document with embedding"""
        if doc.embedding is None:
            doc.embedding = self._create_embedding(doc.content)
        self.documents.append(doc)
    def _create_embedding(self, text: str) -> np.ndarray:
        """Create simple embedding (in reality, use a proper embedding model)"""
        # Simple word-based embedding for demonstration
        words = text.lower().split()
        # Create a simple hash-based embedding
        embedding = np.zeros(128)
        for word in words:
            idx = hash(word) % 128
            embedding[idx] += 1
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        query_embedding = self._create_embedding(query)
        # Calculate cosine similarity
        results = []
        for doc in self.documents:
            similarity = np.dot(query_embedding, doc.embedding)
            results.append((doc, similarity))
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
class RAGAgent:
    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store
    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """Retrieve relevant documents"""
        print(f"\n=== Retrieval Phase ===")
        print(f"Query: {query}")
        results = self.vector_store.search(query, top_k)
        print(f"\nRetrieved {len(results)} relevant documents:")
        retrieved_docs = []
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{i}. [Score: {score:.3f}] {doc.metadata.get('title', 'Untitled')}")
            print(f"   {doc.content[:200]}...")
            retrieved_docs.append(doc)
        return retrieved_docs
    def generate(self, query: str, context_docs: List[Document]) -> str:
        """Generate answer using retrieved context"""
        print(f"\n=== Generation Phase ===")
        # Combine context
        context = "\n\n".join([
            f"Source {i+1}: {doc.content}"
            for i, doc in enumerate(context_docs)
        ])
        # Simulated LLM generation (in real implementation, call actual LLM)
        print("Generating answer based on retrieved context...")
        # Simple rule-based generation for demonstration
        answer = self._simple_generate(query, context_docs)
        return answer
    def _simple_generate(self, query: str, docs: List[Document]) -> str:
        """Simple answer generation (simulation)"""
        # Extract relevant sentences
        query_words = set(query.lower().split())
        relevant_parts = []
        for doc in docs:
            sentences = doc.content.split('.')
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                if query_words & sentence_words:
                    relevant_parts.append(sentence.strip())
        if relevant_parts:
            answer = f"Based on the retrieved information: {' '.join(relevant_parts[:2])}."
        else:
            answer = "I couldn't find specific information to answer your question."
        # Add source attribution
        sources = [doc.metadata.get('title', 'Unknown') for doc in docs]
        answer += f"\n\nSources: {', '.join(sources)}"
        return answer
    def query(self, question: str, top_k: int = 3) -> str:
        """Complete RAG pipeline: retrieve + generate"""
        print(f"\n{'='*60}")
        print(f"RAG Query Pipeline")
        print(f"{'='*60}")
        # Step 1: Retrieve relevant documents
        relevant_docs = self.retrieve(question, top_k)
        # Step 2: Generate answer using retrieved context
        answer = self.generate(question, relevant_docs)
        print(f"\n=== Final Answer ===")
        print(answer)
        return answer
# Usage
if __name__ == "__main__":
    # Create vector store and add documents
    vector_store = SimpleVectorStore()
    # Add sample documents
    documents = [
        Document(
            id="doc1",
            content="Python is a high-level, interpreted programming language. "
                   "It was created by Guido van Rossum and first released in 1991. "
                   "Python emphasizes code readability and supports multiple programming paradigms.",
            metadata={"title": "Python Programming Language", "source": "tech_wiki"}
        ),
        Document(
            id="doc2",
            content="Machine learning is a subset of artificial intelligence. "
                   "It focuses on building systems that can learn from data. "
                   "Python is widely used for machine learning due to libraries like scikit-learn and TensorFlow.",
            metadata={"title": "Machine Learning Overview", "source": "ai_guide"}
        ),
        Document(
            id="doc3",
            content="Web development with Python can be done using frameworks like Django and Flask. "
                   "Django is a high-level framework that encourages rapid development. "
                   "Flask is a lightweight micro-framework for building web applications.",
            metadata={"title": "Python Web Frameworks", "source": "web_dev_guide"}
        ),
        Document(
            id="doc4",
            content="Data science involves extracting insights from data. "
                   "Python is the most popular language for data science. "
                   "Libraries like Pandas, NumPy, and Matplotlib are essential tools.",
            metadata={"title": "Data Science with Python", "source": "data_guide"}
        ),
    ]
    for doc in documents:
        vector_store.add_document(doc)
    print(f"Loaded {len(documents)} documents into vector store\n")
    # Create RAG agent
    agent = RAGAgent(vector_store)
    # Query 1
    agent.query("What is Python and who created it?")
    # Query 2
    print("\n" + "="*80 + "\n")
    agent.query("How is Python used in machine learning?")
    # Query 3
    print("\n" + "="*80 + "\n")
    agent.query("What frameworks are available for web development?")
