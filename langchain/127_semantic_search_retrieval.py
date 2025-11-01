"""
Pattern 127: Semantic Search & Retrieval

Description:
    Retrieves information based on semantic similarity rather than keyword matching,
    using embeddings and vector databases for intelligent information retrieval.

Components:
    - Text embeddings
    - Vector similarity search
    - Semantic ranking
    - Hybrid retrieval (keyword + semantic)

Use Cases:
    - Intelligent document search
    - Question answering systems
    - Recommendation systems

LangChain Implementation:
    Uses LangChain embeddings, vector stores, and retrievers for semantic search.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.retrievers import BM25Retriever

load_dotenv()


class SemanticSearchAgent:
    """Agent for semantic search and retrieval"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore: Optional[Chroma] = None
        self.documents: List[Document] = []
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        
        # QA prompt
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer. If you cannot answer from the context, say so.
Always cite which part of the context you used."""),
            ("user", """Context:
{context}

Question: {question}

Answer:""")
        ])
    
    def load_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """Load documents into the system"""
        print(f"\n📚 Loading {len(documents)} documents...")
        
        # Create Document objects
        if metadatas:
            self.documents = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(documents, metadatas)
            ]
        else:
            self.documents = [
                Document(page_content=doc, metadata={"id": i})
                for i, doc in enumerate(documents)
            ]
        
        # Split documents
        split_docs = self.text_splitter.split_documents(self.documents)
        print(f"   Split into {len(split_docs)} chunks")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            collection_name="semantic_search"
        )
        
        print(f"   ✓ Vectorstore created with {len(split_docs)} embeddings")
    
    def semantic_search(self, query: str, k: int = 3) -> List[Document]:
        """Pure semantic search using vector similarity"""
        print(f"\n🔍 Semantic search for: '{query}'")
        
        if not self.vectorstore:
            print("   ❌ No documents loaded!")
            return []
        
        # Retrieve similar documents
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        print(f"   Found {len(results)} results:\n")
        for i, (doc, score) in enumerate(results, 1):
            print(f"   {i}. Score: {score:.4f}")
            print(f"      {doc.page_content[:100]}...\n")
        
        return [doc for doc, _ in results]
    
    def hybrid_search(self, query: str, k: int = 3) -> List[Document]:
        """Hybrid search combining semantic and keyword matching"""
        print(f"\n🔍 Hybrid search for: '{query}'")
        
        if not self.vectorstore or not self.documents:
            print("   ❌ No documents loaded!")
            return []
        
        # Create semantic retriever
        semantic_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Create keyword retriever (BM25)
        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = k
        
        # Combine retrievers
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.7, 0.3]  # Favor semantic
        )
        
        results = ensemble_retriever.get_relevant_documents(query)
        
        print(f"   Found {len(results)} results (semantic + keyword)\n")
        for i, doc in enumerate(results[:k], 1):
            print(f"   {i}. {doc.page_content[:100]}...\n")
        
        return results[:k]
    
    def search_with_reranking(self, query: str, k: int = 3) -> List[Document]:
        """Search with LLM-based reranking/compression"""
        print(f"\n🔍 Search with reranking for: '{query}'")
        
        if not self.vectorstore:
            print("   ❌ No documents loaded!")
            return []
        
        # Base retriever
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": k * 2})
        
        # Compressor for reranking
        compressor = LLMChainExtractor.from_llm(self.llm)
        
        # Compression retriever
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        results = compression_retriever.get_relevant_documents(query)
        
        print(f"   Found and reranked {len(results)} results\n")
        for i, doc in enumerate(results[:k], 1):
            print(f"   {i}. {doc.page_content[:150]}...\n")
        
        return results[:k]
    
    def answer_question(self, question: str, search_method: str = "semantic") -> str:
        """Answer question using retrieved context"""
        print(f"\n💬 Answering question: '{question}'")
        print(f"   Using {search_method} search")
        
        # Retrieve context
        if search_method == "semantic":
            context_docs = self.semantic_search(question, k=3)
        elif search_method == "hybrid":
            context_docs = self.hybrid_search(question, k=3)
        else:  # reranking
            context_docs = self.search_with_reranking(question, k=3)
        
        if not context_docs:
            return "No relevant context found."
        
        # Combine context
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Generate answer
        chain = self.qa_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        
        print(f"\n📝 Answer:\n{answer}")
        
        return answer
    
    def find_similar_documents(self, reference_text: str, k: int = 3) -> List[Document]:
        """Find documents similar to reference text"""
        print(f"\n🔍 Finding similar documents to reference text...")
        
        if not self.vectorstore:
            print("   ❌ No documents loaded!")
            return []
        
        results = self.vectorstore.similarity_search(reference_text, k=k)
        
        print(f"   Found {len(results)} similar documents:\n")
        for i, doc in enumerate(results, 1):
            print(f"   {i}. {doc.page_content[:100]}...")
            if 'id' in doc.metadata:
                print(f"      (Document ID: {doc.metadata['id']})\n")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search system statistics"""
        return {
            "total_documents": len(self.documents),
            "vectorstore_loaded": self.vectorstore is not None,
            "embedding_model": "text-embedding-ada-002"
        }


def demonstrate_semantic_search():
    """Demonstrate semantic search and retrieval"""
    print("=" * 80)
    print("Pattern 127: Semantic Search & Retrieval")
    print("=" * 80)
    
    agent = SemanticSearchAgent()
    
    # Example 1: Load documents
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Loading Documents")
    print("=" * 80)
    
    documents = [
        """Python is a high-level, interpreted programming language known for its simplicity 
        and readability. Created by Guido van Rossum in 1991, Python emphasizes code 
        readability and allows programmers to express concepts in fewer lines of code.""",
        
        """Machine learning is a subset of artificial intelligence that enables systems to 
        learn and improve from experience without being explicitly programmed. It focuses 
        on developing computer programs that can access data and learn from it.""",
        
        """Deep learning is a subset of machine learning that uses neural networks with 
        multiple layers. These deep neural networks attempt to simulate the behavior of 
        the human brain, allowing it to learn from large amounts of data.""",
        
        """Natural Language Processing (NLP) is a branch of AI that helps computers understand, 
        interpret, and manipulate human language. NLP draws from many disciplines, including 
        computer science and computational linguistics.""",
        
        """Computer vision is a field of artificial intelligence that trains computers to 
        interpret and understand the visual world. Using digital images and deep learning 
        models, machines can accurately identify and classify objects.""",
        
        """The Transformer architecture, introduced in the paper 'Attention is All You Need', 
        revolutionized NLP. It relies entirely on attention mechanisms, dispensing with 
        recurrence and convolutions entirely."""
    ]
    
    metadatas = [
        {"topic": "programming", "language": "python"},
        {"topic": "AI", "subtopic": "machine_learning"},
        {"topic": "AI", "subtopic": "deep_learning"},
        {"topic": "AI", "subtopic": "NLP"},
        {"topic": "AI", "subtopic": "computer_vision"},
        {"topic": "AI", "subtopic": "transformers"}
    ]
    
    agent.load_documents(documents, metadatas)
    
    # Example 2: Semantic search
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Pure Semantic Search")
    print("=" * 80)
    
    queries = [
        "neural networks and learning from data",
        "understanding human language with computers",
        "programming language for beginners"
    ]
    
    for query in queries:
        agent.semantic_search(query, k=2)
    
    # Example 3: Hybrid search
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Hybrid Search (Semantic + Keyword)")
    print("=" * 80)
    
    agent.hybrid_search("deep neural networks", k=2)
    
    # Example 4: Search with reranking
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Search with LLM Reranking")
    print("=" * 80)
    
    agent.search_with_reranking("AI techniques for processing text", k=2)
    
    # Example 5: Question answering with different search methods
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Question Answering")
    print("=" * 80)
    
    questions = [
        "What is the Transformer architecture?",
        "Who created Python and when?",
        "What is the relationship between deep learning and machine learning?"
    ]
    
    methods = ["semantic", "hybrid", "reranking"]
    
    for question in questions[:1]:  # Just first question with all methods
        print(f"\nQuestion: {question}\n")
        for method in methods:
            print(f"\n--- Method: {method} ---")
            agent.answer_question(question, search_method=method)
    
    # Example 6: Find similar documents
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Finding Similar Documents")
    print("=" * 80)
    
    reference = "Neural networks that can process and generate human language"
    agent.find_similar_documents(reference, k=3)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Semantic Search & Retrieval Pattern:
- Finds information based on meaning, not just keywords
- Uses embeddings to capture semantic similarity
- Multiple search strategies available
- Enables intelligent question answering

Search Methods:
1. Pure Semantic: Vector similarity only
2. Hybrid: Combines semantic + keyword (BM25)
3. With Reranking: LLM compresses and reranks results

Key Benefits:
✓ Understanding user intent
✓ Finding conceptually similar content
✓ Better than keyword matching
✓ Handles synonyms and paraphrases
✓ Improved retrieval accuracy

Use Cases:
• Document search systems
• Question answering
• Recommendation engines
• Knowledge bases
• Research assistants
    """)


if __name__ == "__main__":
    demonstrate_semantic_search()
