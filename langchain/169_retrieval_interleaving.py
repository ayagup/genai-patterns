"""
Pattern 169: Retrieval Interleaving

Description:
    The Retrieval Interleaving pattern dynamically retrieves information during
    generation, interleaving retrieval steps with reasoning and generation steps.
    Unlike standard RAG which retrieves once upfront, this pattern retrieves
    multiple times as needed based on the generation process.

Components:
    1. Generation Monitor: Tracks generation progress
    2. Retrieval Trigger: Decides when to retrieve
    3. Query Generator: Creates retrieval queries
    4. Retriever: Fetches relevant information
    5. Context Integrator: Merges retrieved content
    6. Iteration Controller: Manages interleaving loops

Use Cases:
    - Complex multi-hop reasoning
    - Long-form content generation
    - Research-intensive tasks
    - Dynamic information needs
    - Iterative question answering
    - Knowledge synthesis

Benefits:
    - Adaptive information gathering
    - Better context relevance
    - Handles complex queries
    - Reduces initial retrieval overhead
    - More accurate results

Trade-offs:
    - Multiple retrieval calls
    - Increased latency
    - Complex orchestration
    - Higher costs
    - Requires good triggering logic

LangChain Implementation:
    Implements iterative retrieval with LangChain's retrieval components.
    Uses LLM to decide when and what to retrieve during generation.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()


class RetrievalDecision(Enum):
    """Decision on whether to retrieve"""
    RETRIEVE = "retrieve"
    CONTINUE_GENERATION = "continue"
    COMPLETE = "complete"


@dataclass
class RetrievalStep:
    """Represents a retrieval step"""
    query: str
    documents: List[Document]
    timestamp: int
    reasoning: str


@dataclass
class GenerationStep:
    """Represents a generation step"""
    content: str
    timestamp: int
    used_retrieval: bool


@dataclass
class InterleavedResult:
    """Result of interleaved retrieval-generation"""
    final_output: str
    retrieval_steps: List[RetrievalStep]
    generation_steps: List[GenerationStep]
    total_retrievals: int
    total_tokens_estimated: int


class MockVectorStore:
    """Mock vector store for demonstration"""
    
    def __init__(self):
        """Initialize with sample documents"""
        self.documents = [
            Document(page_content="Machine learning is a subset of AI that enables computers to learn from data.", 
                    metadata={"source": "ml_basics"}),
            Document(page_content="Deep learning uses neural networks with multiple layers to process data.",
                    metadata={"source": "dl_intro"}),
            Document(page_content="Natural language processing (NLP) enables computers to understand human language.",
                    metadata={"source": "nlp_overview"}),
            Document(page_content="Transformers revolutionized NLP with attention mechanisms.",
                    metadata={"source": "transformers"}),
            Document(page_content="GPT models are large language models trained on vast amounts of text data.",
                    metadata={"source": "gpt_info"}),
            Document(page_content="RAG combines retrieval with generation for better accuracy.",
                    metadata={"source": "rag_concept"}),
            Document(page_content="Computer vision enables machines to interpret visual information.",
                    metadata={"source": "cv_basics"}),
            Document(page_content="Reinforcement learning trains agents through rewards and penalties.",
                    metadata={"source": "rl_intro"}),
        ]
        self.embeddings = OpenAIEmbeddings()
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Mock similarity search"""
        # Simple keyword matching for demonstration
        query_lower = query.lower()
        scored_docs = []
        
        for doc in self.documents:
            content_lower = doc.page_content.lower()
            score = sum(1 for word in query_lower.split() if word in content_lower)
            scored_docs.append((score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:k]]


class RetrievalInterleavingAgent:
    """Agent that interleaves retrieval with generation"""
    
    def __init__(self, vector_store: Optional[MockVectorStore] = None,
                 max_retrievals: int = 5):
        """
        Initialize interleaving agent
        
        Args:
            vector_store: Vector store for retrieval
            max_retrievals: Maximum number of retrieval steps
        """
        self.vector_store = vector_store or MockVectorStore()
        self.max_retrievals = max_retrievals
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        # Prompts
        self.decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI that decides when to retrieve more information.
            Analyze the current generation state and decide if you need more information.
            
            Respond with one of: RETRIEVE, CONTINUE, COMPLETE
            If RETRIEVE, also provide a search query."""),
            ("user", """Query: {query}

Current generation:
{current_generation}

Context so far:
{context}

Decision (RETRIEVE/CONTINUE/COMPLETE) and query if retrieving:""")
        ])
        
        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Generate content based on 
            the query and available context. Be factual and cite information when possible."""),
            ("user", """Query: {query}

Available context:
{context}

Previous generation:
{previous_generation}

Continue or complete the response:""")
        ])
        
        self.decision_chain = self.decision_prompt | self.llm | StrOutputParser()
        self.generation_chain = self.generation_prompt | self.llm | StrOutputParser()
    
    def generate_with_interleaved_retrieval(self, query: str) -> InterleavedResult:
        """Generate response with interleaved retrieval"""
        retrieval_steps: List[RetrievalStep] = []
        generation_steps: List[GenerationStep] = []
        
        current_generation = ""
        retrieved_context = ""
        step_counter = 0
        
        # Initial generation attempt
        initial_gen = self._generate(query, "", "")
        current_generation = initial_gen
        generation_steps.append(GenerationStep(
            content=initial_gen,
            timestamp=step_counter,
            used_retrieval=False
        ))
        step_counter += 1
        
        # Interleaving loop
        for retrieval_count in range(self.max_retrievals):
            # Decide if we need more information
            decision, retrieval_query = self._decide_next_step(
                query, current_generation, retrieved_context
            )
            
            if decision == RetrievalDecision.COMPLETE:
                break
            
            if decision == RetrievalDecision.RETRIEVE and retrieval_query:
                # Retrieve documents
                docs = self.vector_store.similarity_search(retrieval_query, k=2)
                
                retrieval_steps.append(RetrievalStep(
                    query=retrieval_query,
                    documents=docs,
                    timestamp=step_counter,
                    reasoning=f"Needed information about: {retrieval_query}"
                ))
                step_counter += 1
                
                # Update context
                new_context = "\n\n".join([doc.page_content for doc in docs])
                retrieved_context += "\n\n" + new_context
                
                # Generate with new context
                new_generation = self._generate(query, retrieved_context, current_generation)
                current_generation = new_generation
                
                generation_steps.append(GenerationStep(
                    content=new_generation,
                    timestamp=step_counter,
                    used_retrieval=True
                ))
                step_counter += 1
            
            elif decision == RetrievalDecision.CONTINUE_GENERATION:
                # Continue generation without retrieval
                new_generation = self._generate(query, retrieved_context, current_generation)
                current_generation = new_generation
                
                generation_steps.append(GenerationStep(
                    content=new_generation,
                    timestamp=step_counter,
                    used_retrieval=False
                ))
                step_counter += 1
        
        # Estimate tokens
        total_tokens = sum(len(step.content.split()) for step in generation_steps) * 1.3
        total_tokens += sum(len(step.query.split()) for step in retrieval_steps) * 1.3
        
        return InterleavedResult(
            final_output=current_generation,
            retrieval_steps=retrieval_steps,
            generation_steps=generation_steps,
            total_retrievals=len(retrieval_steps),
            total_tokens_estimated=int(total_tokens)
        )
    
    def _decide_next_step(self, query: str, current_generation: str,
                         context: str) -> Tuple[RetrievalDecision, Optional[str]]:
        """Decide whether to retrieve, continue, or complete"""
        decision_text = self.decision_chain.invoke({
            "query": query,
            "current_generation": current_generation[:200],
            "context": context[:300] if context else "No context yet"
        })
        
        # Parse decision
        decision_text_upper = decision_text.upper()
        
        if "COMPLETE" in decision_text_upper:
            return RetrievalDecision.COMPLETE, None
        elif "RETRIEVE" in decision_text_upper:
            # Extract query from response
            lines = decision_text.split('\n')
            query_line = next((line for line in lines if "query" in line.lower()), None)
            if query_line and ':' in query_line:
                retrieval_query = query_line.split(':', 1)[1].strip()
            else:
                retrieval_query = query  # Fallback to original query
            return RetrievalDecision.RETRIEVE, retrieval_query
        else:
            return RetrievalDecision.CONTINUE_GENERATION, None
    
    def _generate(self, query: str, context: str, previous: str) -> str:
        """Generate content with available context"""
        response = self.generation_chain.invoke({
            "query": query,
            "context": context if context else "No additional context available",
            "previous_generation": previous if previous else "Starting response"
        })
        return response


def demonstrate_retrieval_interleaving():
    """Demonstrate retrieval interleaving pattern"""
    print("=" * 80)
    print("RETRIEVAL INTERLEAVING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Create agent with mock vector store
    agent = RetrievalInterleavingAgent(max_retrievals=3)
    
    # Example 1: Complex query requiring multiple retrievals
    print("\n" + "=" * 80)
    print("Example 1: Multi-Hop Question")
    print("=" * 80)
    
    query1 = "Explain how transformers relate to GPT models and why they're important for NLP"
    
    print(f"\nQuery: {query1}")
    print("\nExecuting with interleaved retrieval...")
    print("-" * 60)
    
    result1 = agent.generate_with_interleaved_retrieval(query1)
    
    print(f"\nFinal Output:")
    print(result1.final_output)
    
    print(f"\n" + "-" * 60)
    print(f"Execution Summary:")
    print(f"  Total retrievals: {result1.total_retrievals}")
    print(f"  Total generation steps: {len(result1.generation_steps)}")
    print(f"  Estimated tokens: {result1.total_tokens_estimated}")
    
    print(f"\nRetrieval Steps:")
    for i, step in enumerate(result1.retrieval_steps, 1):
        print(f"\n  Step {i} (timestamp {step.timestamp}):")
        print(f"    Query: {step.query}")
        print(f"    Retrieved {len(step.documents)} documents")
        for j, doc in enumerate(step.documents, 1):
            print(f"      Doc {j}: {doc.page_content[:60]}...")
    
    print(f"\nGeneration Steps:")
    for i, step in enumerate(result1.generation_steps, 1):
        used = "with retrieval" if step.used_retrieval else "without retrieval"
        print(f"  Step {i} (timestamp {step.timestamp}): Generated {used}")
        print(f"    Content: {step.content[:80]}...")
    
    # Example 2: Simple query (fewer retrievals)
    print("\n" + "=" * 80)
    print("Example 2: Simple Query")
    print("=" * 80)
    
    query2 = "What is machine learning?"
    
    print(f"\nQuery: {query2}")
    result2 = agent.generate_with_interleaved_retrieval(query2)
    
    print(f"\nFinal Output:")
    print(result2.final_output)
    
    print(f"\nExecution Summary:")
    print(f"  Total retrievals: {result2.total_retrievals}")
    print(f"  (Simpler query required fewer retrievals)")
    
    # Example 3: Comparing with standard RAG
    print("\n" + "=" * 80)
    print("Example 3: Comparison - Interleaved vs Standard RAG")
    print("=" * 80)
    
    query3 = "Explain deep learning and how it relates to computer vision and NLP"
    
    print(f"\nQuery: {query3}")
    
    # Interleaved approach
    print("\n1. Interleaved Retrieval Approach:")
    result_interleaved = agent.generate_with_interleaved_retrieval(query3)
    print(f"   Retrievals: {result_interleaved.total_retrievals}")
    print(f"   Output: {result_interleaved.final_output[:150]}...")
    
    # Standard RAG (single retrieval)
    print("\n2. Standard RAG (single retrieval):")
    docs = agent.vector_store.similarity_search(query3, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    standard_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the provided context to answer the query."),
        ("user", "Context:\n{context}\n\nQuery: {query}\n\nAnswer:")
    ])
    standard_chain = standard_prompt | agent.llm | StrOutputParser()
    standard_result = standard_chain.invoke({"context": context, "query": query3})
    
    print(f"   Retrievals: 1")
    print(f"   Output: {standard_result[:150]}...")
    
    print("\n   Analysis:")
    print(f"   - Interleaved made {result_interleaved.total_retrievals} targeted retrievals")
    print(f"   - Standard RAG made 1 broad retrieval")
    print(f"   - Interleaved can gather more specific information as needed")
    
    # Example 4: Retrieval pattern analysis
    print("\n" + "=" * 80)
    print("Example 4: Retrieval Pattern Analysis")
    print("=" * 80)
    
    queries = [
        "What is NLP?",
        "Explain the relationship between machine learning, deep learning, and transformers",
        "How does reinforcement learning work?"
    ]
    
    print("\nAnalyzing retrieval patterns for different query types:\n")
    
    for query in queries:
        result = agent.generate_with_interleaved_retrieval(query)
        print(f"Query: {query}")
        print(f"  Retrievals: {result.total_retrievals}")
        print(f"  Generation steps: {len(result.generation_steps)}")
        
        if result.retrieval_steps:
            print(f"  Retrieval queries:")
            for step in result.retrieval_steps:
                print(f"    - {step.query}")
        print()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Retrieval Interleaving pattern enables:
✓ Dynamic, adaptive information gathering
✓ Multiple retrieval steps during generation
✓ More targeted and relevant retrievals
✓ Better handling of complex multi-hop queries
✓ Reduced initial retrieval overhead
✓ Context-aware retrieval decisions
✓ Improved accuracy for knowledge-intensive tasks

This pattern is valuable for:
- Complex research questions
- Multi-hop reasoning tasks
- Long-form content generation
- Dynamic information needs
- Iterative question answering
- Knowledge synthesis tasks
- Exploratory queries

Compared to standard RAG:
- Standard RAG: One retrieval upfront, simpler, faster
- Interleaved: Multiple targeted retrievals, more complex, better for complex queries
    """)


if __name__ == "__main__":
    demonstrate_retrieval_interleaving()
