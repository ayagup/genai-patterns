"""
Pattern 023: Retrieval-Augmented Generation (RAG)

Description:
    Agent retrieves relevant information from a knowledge base before generating responses.
    RAG combines the power of retrieval systems with generative models to provide
    grounded, factual responses based on external knowledge.

Components:
    - Document Loader: Loads documents into the system
    - Text Splitter: Chunks documents into manageable pieces
    - Embedding Model: Converts text to vector representations
    - Vector Store: Stores and retrieves embeddings efficiently
    - Retriever: Finds relevant documents
    - LLM: Generates response based on retrieved context

Use Cases:
    - Question answering over documents
    - Knowledge-intensive tasks
    - Chat with your data applications
    - Customer support with documentation

LangChain Implementation:
    Uses LangChain's built-in RAG components with vector stores and retrievers.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()


# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    {
        "content": """LangChain is a framework for developing applications powered by language models. 
It enables applications that are context-aware and can reason. LangChain provides modular 
components for working with language models, as well as pre-built chains and agents.""",
        "metadata": {"source": "langchain_intro", "topic": "framework"}
    },
    {
        "content": """Vector databases are specialized databases designed to store and query 
high-dimensional vectors efficiently. They use techniques like approximate nearest neighbor 
search to find similar vectors quickly. Popular vector databases include Pinecone, Weaviate, 
and Chroma.""",
        "metadata": {"source": "vector_db_guide", "topic": "databases"}
    },
    {
        "content": """Embeddings are numerical representations of text that capture semantic meaning. 
Words or sentences with similar meanings have similar embeddings. Common embedding models include 
OpenAI's text-embedding-ada-002 and open-source alternatives like Sentence Transformers.""",
        "metadata": {"source": "embeddings_explained", "topic": "embeddings"}
    },
    {
        "content": """Retrieval-Augmented Generation (RAG) combines retrieval systems with language models. 
First, relevant documents are retrieved from a knowledge base. Then, these documents are provided 
as context to the language model, which generates a response grounded in the retrieved information.""",
        "metadata": {"source": "rag_overview", "topic": "rag"}
    },
    {
        "content": """Semantic search finds results based on meaning rather than exact keyword matches. 
It uses embeddings to represent queries and documents in a vector space, then finds the most 
semantically similar documents. This enables finding relevant information even when different 
words are used.""",
        "metadata": {"source": "semantic_search_guide", "topic": "search"}
    }
]


class RAGAgent:
    """Agent that uses Retrieval-Augmented Generation."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        k_documents: int = 3
    ):
        """
        Initialize the RAG agent.
        
        Args:
            model_name: LLM model to use for generation
            embedding_model: Embedding model for vector representations
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            k_documents: Number of documents to retrieve
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        self.k_documents = k_documents
        self.vectorstore = None
        self.retriever = None
    
    def load_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Load documents into the vector store.
        
        Args:
            documents: List of dictionaries with 'content' and 'metadata' keys
        """
        # Convert to Document objects
        docs = [
            Document(page_content=doc["content"], metadata=doc["metadata"])
            for doc in documents
        ]
        
        # Split documents into chunks
        splits = self.text_splitter.split_documents(docs)
        
        print(f"Loaded {len(docs)} documents, split into {len(splits)} chunks")
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name="rag_demo"
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k_documents}
        )
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        if not self.retriever:
            raise ValueError("No documents loaded. Call load_documents first.")
        
        return self.retriever.get_relevant_documents(query)
    
    def query_with_retrieval_qa(self, query: str) -> Dict[str, Any]:
        """
        Query using RetrievalQA chain (simple approach).
        
        Args:
            query: Question to answer
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.retriever:
            raise ValueError("No documents loaded. Call load_documents first.")
        
        # Create RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff" puts all docs into prompt
            retriever=self.retriever,
            return_source_documents=True
        )
        
        result = qa_chain.invoke({"query": query})
        
        return {
            "query": query,
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    
    def query_with_custom_prompt(self, query: str) -> Dict[str, Any]:
        """
        Query using custom prompt template (more control).
        
        Args:
            query: Question to answer
            
        Returns:
            Dictionary with answer and retrieved documents
        """
        if not self.retriever:
            raise ValueError("No documents loaded. Call load_documents first.")
        
        # Custom prompt template
        template = """Answer the question based on the following context. 
If you cannot answer the question based on the context, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG chain using LCEL (LangChain Expression Language)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Get retrieved documents for transparency
        retrieved_docs = self.retrieve_documents(query)
        
        # Generate answer
        answer = rag_chain.invoke(query)
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "num_retrieved": len(retrieved_docs)
        }
    
    def query_with_sources(self, query: str) -> Dict[str, Any]:
        """
        Query with explicit source citations.
        
        Args:
            query: Question to answer
            
        Returns:
            Dictionary with answer and cited sources
        """
        if not self.retriever:
            raise ValueError("No documents loaded. Call load_documents first.")
        
        # Retrieve documents
        docs = self.retrieve_documents(query)
        
        # Create context with source labels
        context_with_sources = ""
        sources = []
        
        for i, doc in enumerate(docs, 1):
            source_label = f"[Source {i}: {doc.metadata.get('source', 'unknown')}]"
            context_with_sources += f"{source_label}\n{doc.page_content}\n\n"
            sources.append({
                "id": i,
                "source": doc.metadata.get('source', 'unknown'),
                "topic": doc.metadata.get('topic', 'unknown'),
                "content": doc.page_content
            })
        
        # Generate answer with citations
        template = """Answer the question based on the following sources. 
Cite your sources using [Source N] notation where applicable.

Sources:
{context}

Question: {question}

Answer (with citations):"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context_with_sources,
            "question": query
        })
        
        return {
            "query": query,
            "answer": answer,
            "sources": sources
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.vectorstore:
            try:
                self.vectorstore.delete_collection()
            except:
                pass


def demonstrate_rag_pattern():
    """Demonstrates the RAG pattern with various approaches."""
    
    print("=" * 80)
    print("PATTERN 023: Retrieval-Augmented Generation (RAG)")
    print("=" * 80)
    print()
    
    # Create RAG agent
    agent = RAGAgent(k_documents=3)
    
    # Load sample documents
    print("Loading documents into vector store...")
    agent.load_documents(SAMPLE_DOCUMENTS)
    print()
    
    # Test queries
    queries = [
        "What is LangChain?",
        "How do vector databases work?",
        "What is RAG and how does it combine retrieval with generation?",
    ]
    
    # Approach 1: RetrievalQA chain
    print("\n" + "=" * 80)
    print("APPROACH 1: RetrievalQA Chain (Simple)")
    print("=" * 80)
    
    for i, query in enumerate(queries[:2], 1):
        print(f"\n{'- ' * 40}")
        print(f"Query {i}: {query}")
        print('- ' * 40)
        
        try:
            result = agent.query_with_retrieval_qa(query)
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nSources used: {len(result['source_documents'])} documents")
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
    
    # Approach 2: Custom prompt with LCEL
    print("\n\n" + "=" * 80)
    print("APPROACH 2: Custom Prompt with LCEL")
    print("=" * 80)
    
    query = queries[1]
    print(f"\n{'- ' * 40}")
    print(f"Query: {query}")
    print('- ' * 40)
    
    try:
        result = agent.query_with_custom_prompt(query)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nRetrieved {result['num_retrieved']} documents")
        print("\nRetrieved document topics:")
        for doc in result['retrieved_documents']:
            print(f"  - {doc.metadata.get('topic', 'unknown')}")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
    
    # Approach 3: With source citations
    print("\n\n" + "=" * 80)
    print("APPROACH 3: Answer with Source Citations")
    print("=" * 80)
    
    query = queries[2]
    print(f"\n{'- ' * 40}")
    print(f"Query: {query}")
    print('- ' * 40)
    
    try:
        result = agent.query_with_sources(query)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources:")
        for source in result['sources']:
            print(f"  [{source['id']}] {source['source']} (Topic: {source['topic']})")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
    
    # Cleanup
    agent.cleanup()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("RAG PATTERN DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("RAG Components Demonstrated:")
    print("1. Document loading and chunking")
    print("2. Vector embeddings and storage")
    print("3. Semantic retrieval")
    print("4. Context-aware generation")
    print("5. Source attribution")
    print()
    print("Key Benefits:")
    print("- Grounded responses based on actual data")
    print("- Reduced hallucinations")
    print("- Up-to-date information")
    print("- Source traceability")
    print("- Scalable knowledge base")
    print()
    print("LangChain Components Used:")
    print("- RecursiveCharacterTextSplitter: Chunks documents")
    print("- OpenAIEmbeddings: Creates vector embeddings")
    print("- Chroma: Vector store for efficient retrieval")
    print("- RetrievalQA: Pre-built RAG chain")
    print("- LCEL: Custom chain composition")
    print("- Document retriever: Finds relevant documents")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    demonstrate_rag_pattern()
