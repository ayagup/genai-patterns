"""
Advanced RAG Pattern Implementation

This module demonstrates advanced Retrieval-Augmented Generation with:
- Query planning and decomposition
- Multi-hop reasoning across documents
- Document ranking and reranking
- Context synthesis and fusion
- Citation tracking

Key Components:
- Query planner for breaking down complex queries
- Multi-stage retrieval with ranking
- Cross-document reasoning
- Answer synthesis with citations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Set
from enum import Enum
import random
import re
from datetime import datetime


class QueryType(Enum):
    """Types of queries that can be processed"""
    FACTUAL = "factual"           # Simple factual lookup
    ANALYTICAL = "analytical"     # Requires analysis across sources
    COMPARATIVE = "comparative"   # Comparing multiple entities
    AGGREGATIVE = "aggregative"   # Aggregating information
    MULTI_HOP = "multi_hop"      # Requires multiple retrieval steps
    TEMPORAL = "temporal"         # Time-based queries
    CAUSAL = "causal"            # Cause-effect relationships


class RetrievalStrategy(Enum):
    """Strategies for retrieving documents"""
    DENSE = "dense"              # Dense vector similarity
    SPARSE = "sparse"            # Keyword/BM25 search
    HYBRID = "hybrid"            # Combination of dense and sparse
    SEMANTIC = "semantic"        # Semantic similarity
    GRAPH = "graph"              # Graph-based retrieval


@dataclass
class Document:
    """Represents a document in the knowledge base"""
    id: str
    content: str
    title: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def get_snippet(self, max_length: int = 200) -> str:
        """Get a snippet of the document"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."


@dataclass
class RetrievalResult:
    """Result from document retrieval"""
    document: Document
    score: float
    relevance_score: float = 0.0
    rank: int = 0
    retrieval_method: str = ""
    matched_query: str = ""
    
    def get_combined_score(self) -> float:
        """Combine different scoring factors"""
        return (self.score * 0.6) + (self.relevance_score * 0.4)


@dataclass
class QueryPlan:
    """Plan for executing a complex query"""
    original_query: str
    query_type: QueryType
    sub_queries: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    retrieval_strategies: Dict[str, RetrievalStrategy] = field(default_factory=dict)
    expected_hops: int = 1
    
    def is_multi_hop(self) -> bool:
        """Check if this is a multi-hop query"""
        return self.expected_hops > 1 or len(self.sub_queries) > 1


@dataclass
class Citation:
    """Citation information for generated content"""
    document_id: str
    document_title: str
    snippet: str
    relevance: float
    position: int  # Position in the answer where this is cited


@dataclass
class RAGResponse:
    """Response from RAG system"""
    query: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    retrieved_documents: List[RetrievalResult] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_trace: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class QueryPlanner:
    """Plans how to execute complex queries"""
    
    def __init__(self):
        self.query_patterns = self._create_query_patterns()
    
    def _create_query_patterns(self) -> Dict[QueryType, List[str]]:
        """Create patterns for identifying query types"""
        return {
            QueryType.FACTUAL: [
                r"what is",
                r"who is",
                r"when did",
                r"where is",
                r"define"
            ],
            QueryType.ANALYTICAL: [
                r"analyze",
                r"explain why",
                r"how does",
                r"what causes"
            ],
            QueryType.COMPARATIVE: [
                r"compare",
                r"difference between",
                r"versus",
                r"better than",
                r"vs"
            ],
            QueryType.AGGREGATIVE: [
                r"list all",
                r"summarize",
                r"what are the",
                r"give me all"
            ],
            QueryType.MULTI_HOP: [
                r"who.*and.*where",
                r"what.*and.*why",
                r"find.*then.*get"
            ],
            QueryType.TEMPORAL: [
                r"history of",
                r"timeline",
                r"evolution of",
                r"over time"
            ],
            QueryType.CAUSAL: [
                r"why did",
                r"what led to",
                r"cause of",
                r"reason for"
            ]
        }
    
    def plan_query(self, query: str) -> QueryPlan:
        """Create an execution plan for a query"""
        print(f"\n📋 Planning query: {query}")
        
        # Classify query type
        query_type = self._classify_query(query)
        print(f"   Query type: {query_type.value}")
        
        # Decompose into sub-queries if needed
        sub_queries = self._decompose_query(query, query_type)
        print(f"   Sub-queries: {len(sub_queries)}")
        
        # Identify dependencies
        dependencies = self._identify_dependencies(sub_queries)
        
        # Determine retrieval strategies
        retrieval_strategies = self._select_retrieval_strategies(sub_queries, query_type)
        
        # Estimate number of hops needed
        expected_hops = self._estimate_hops(query, query_type, sub_queries)
        print(f"   Expected hops: {expected_hops}")
        
        return QueryPlan(
            original_query=query,
            query_type=query_type,
            sub_queries=sub_queries,
            dependencies=dependencies,
            retrieval_strategies=retrieval_strategies,
            expected_hops=expected_hops
        )
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify the type of query"""
        query_lower = query.lower()
        
        # Check patterns for each query type
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        
        # Default to factual
        return QueryType.FACTUAL
    
    def _decompose_query(self, query: str, query_type: QueryType) -> List[str]:
        """Decompose complex query into sub-queries"""
        sub_queries = [query]  # Always include original
        
        if query_type == QueryType.COMPARATIVE:
            # Extract entities being compared
            entities = self._extract_entities(query)
            if len(entities) >= 2:
                for entity in entities:
                    sub_queries.append(f"What are the characteristics of {entity}?")
        
        elif query_type == QueryType.MULTI_HOP:
            # Split on conjunctions
            parts = re.split(r'\band\b|\bthen\b', query, flags=re.IGNORECASE)
            sub_queries.extend([p.strip() for p in parts if p.strip() and p.strip() != query])
        
        elif query_type == QueryType.CAUSAL:
            # Add background context query
            sub_queries.append(f"What is the context of {self._extract_main_topic(query)}?")
        
        elif query_type == QueryType.AGGREGATIVE:
            # Keep as single query but mark for comprehensive retrieval
            pass
        
        return sub_queries
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query"""
        # Simple extraction - in practice, use NER
        words = query.split()
        entities = []
        
        # Look for capitalized words (simple heuristic)
        for word in words:
            if word[0].isupper() and len(word) > 1:
                entities.append(word)
        
        return entities[:3]  # Limit to 3
    
    def _extract_main_topic(self, query: str) -> str:
        """Extract main topic from query"""
        # Remove question words
        topic = re.sub(r'^(what|who|when|where|why|how)\s+', '', query, flags=re.IGNORECASE)
        # Take first few words
        words = topic.split()[:5]
        return ' '.join(words)
    
    def _identify_dependencies(self, sub_queries: List[str]) -> Dict[str, List[str]]:
        """Identify dependencies between sub-queries"""
        dependencies = {}
        
        # First query has no dependencies
        if len(sub_queries) > 1:
            for i, query in enumerate(sub_queries[1:], 1):
                # Each subsequent query depends on the previous one
                dependencies[query] = [sub_queries[i-1]]
        
        return dependencies
    
    def _select_retrieval_strategies(self, sub_queries: List[str], 
                                   query_type: QueryType) -> Dict[str, RetrievalStrategy]:
        """Select retrieval strategy for each sub-query"""
        strategies = {}
        
        # Default strategy based on query type
        default_strategy = {
            QueryType.FACTUAL: RetrievalStrategy.HYBRID,
            QueryType.ANALYTICAL: RetrievalStrategy.SEMANTIC,
            QueryType.COMPARATIVE: RetrievalStrategy.HYBRID,
            QueryType.AGGREGATIVE: RetrievalStrategy.DENSE,
            QueryType.MULTI_HOP: RetrievalStrategy.SEMANTIC,
            QueryType.TEMPORAL: RetrievalStrategy.SPARSE,
            QueryType.CAUSAL: RetrievalStrategy.SEMANTIC
        }.get(query_type, RetrievalStrategy.HYBRID)
        
        for query in sub_queries:
            strategies[query] = default_strategy
        
        return strategies
    
    def _estimate_hops(self, query: str, query_type: QueryType, 
                      sub_queries: List[str]) -> int:
        """Estimate number of retrieval hops needed"""
        base_hops = 1
        
        # Add hops based on query type
        if query_type in [QueryType.MULTI_HOP, QueryType.CAUSAL]:
            base_hops += 1
        
        # Add hops based on sub-queries
        if len(sub_queries) > 2:
            base_hops += 1
        
        # Check for complex patterns
        if re.search(r'\band\b.*\band\b', query, re.IGNORECASE):
            base_hops += 1
        
        return min(base_hops, 4)  # Cap at 4 hops


class DocumentRetriever:
    """Retrieves documents using multiple strategies"""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self._build_indices()
    
    def _build_indices(self):
        """Build search indices"""
        print(f"📚 Building indices for {len(self.documents)} documents...")
        
        # Build keyword index (simulated)
        self.keyword_index = {}
        for doc in self.documents:
            words = set(doc.content.lower().split())
            for word in words:
                if word not in self.keyword_index:
                    self.keyword_index[word] = []
                self.keyword_index[word].append(doc.id)
        
        # Simulate embeddings for documents
        for doc in self.documents:
            if doc.embedding is None:
                doc.embedding = [random.random() for _ in range(128)]
    
    def retrieve(self, query: str, strategy: RetrievalStrategy, 
                top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve documents using specified strategy"""
        if strategy == RetrievalStrategy.DENSE:
            return self._dense_retrieval(query, top_k)
        elif strategy == RetrievalStrategy.SPARSE:
            return self._sparse_retrieval(query, top_k)
        elif strategy == RetrievalStrategy.HYBRID:
            return self._hybrid_retrieval(query, top_k)
        elif strategy == RetrievalStrategy.SEMANTIC:
            return self._semantic_retrieval(query, top_k)
        else:
            return self._hybrid_retrieval(query, top_k)
    
    def _dense_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Dense vector similarity retrieval"""
        query_embedding = [random.random() for _ in range(128)]
        
        results = []
        for doc in self.documents:
            # Simulate cosine similarity
            if doc.embedding is None:
                continue
            similarity = self._cosine_similarity(query_embedding, doc.embedding)
            
            results.append(RetrievalResult(
                document=doc,
                score=similarity,
                relevance_score=similarity,
                retrieval_method="dense",
                matched_query=query
            ))
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1
        
        return results[:top_k]
    
    def _sparse_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Sparse keyword-based retrieval (BM25-like)"""
        query_words = set(query.lower().split())
        
        scores = {}
        for word in query_words:
            if word in self.keyword_index:
                for doc_id in self.keyword_index[word]:
                    scores[doc_id] = scores.get(doc_id, 0) + 1
        
        results = []
        for doc in self.documents:
            score = scores.get(doc.id, 0)
            if score > 0:
                # Normalize by document length
                normalized_score = score / (len(doc.content.split()) + 1)
                
                results.append(RetrievalResult(
                    document=doc,
                    score=normalized_score,
                    relevance_score=normalized_score,
                    retrieval_method="sparse",
                    matched_query=query
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1
        
        return results[:top_k]
    
    def _hybrid_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Hybrid retrieval combining dense and sparse"""
        dense_results = self._dense_retrieval(query, top_k * 2)
        sparse_results = self._sparse_retrieval(query, top_k * 2)
        
        # Combine and deduplicate
        combined = {}
        
        for result in dense_results:
            combined[result.document.id] = result
            combined[result.document.id].retrieval_method = "hybrid"
        
        for result in sparse_results:
            if result.document.id in combined:
                # Average the scores
                existing = combined[result.document.id]
                existing.score = (existing.score + result.score) / 2
                existing.relevance_score = (existing.relevance_score + result.relevance_score) / 2
            else:
                combined[result.document.id] = result
                combined[result.document.id].retrieval_method = "hybrid"
        
        results = list(combined.values())
        results.sort(key=lambda x: x.get_combined_score(), reverse=True)
        
        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1
        
        return results[:top_k]
    
    def _semantic_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Semantic similarity retrieval"""
        # Similar to dense but with semantic understanding
        return self._dense_retrieval(query, top_k)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


class DocumentReranker:
    """Reranks retrieved documents based on relevance"""
    
    def __init__(self):
        self.reranking_factors = {
            'content_relevance': 0.4,
            'title_match': 0.2,
            'recency': 0.15,
            'authority': 0.15,
            'diversity': 0.1
        }
    
    def rerank(self, query: str, results: List[RetrievalResult], 
               top_k: int = 5) -> List[RetrievalResult]:
        """Rerank results based on multiple factors"""
        print(f"   🔄 Reranking {len(results)} results...")
        
        for result in results:
            rerank_score = self._calculate_rerank_score(query, result)
            result.relevance_score = rerank_score
        
        # Sort by new relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results[:top_k]):
            result.rank = i + 1
        
        return results[:top_k]
    
    def _calculate_rerank_score(self, query: str, result: RetrievalResult) -> float:
        """Calculate reranking score"""
        scores = {}
        
        # Content relevance (based on original score)
        scores['content_relevance'] = result.score
        
        # Title match
        scores['title_match'] = self._title_match_score(query, result.document.title)
        
        # Recency
        scores['recency'] = self._recency_score(result.document)
        
        # Authority (simulated)
        scores['authority'] = random.uniform(0.5, 1.0)
        
        # Diversity (simulated)
        scores['diversity'] = random.uniform(0.3, 0.9)
        
        # Weighted sum
        final_score = sum(
            scores[factor] * weight 
            for factor, weight in self.reranking_factors.items()
        )
        
        return final_score
    
    def _title_match_score(self, query: str, title: str) -> float:
        """Calculate how well title matches query"""
        query_words = set(query.lower().split())
        title_words = set(title.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & title_words)
        return overlap / len(query_words)
    
    def _recency_score(self, document: Document) -> float:
        """Calculate recency score"""
        # Simulate recency based on document metadata
        if 'date' in document.metadata:
            # More recent = higher score
            return random.uniform(0.6, 1.0)
        return 0.5


class AnswerSynthesizer:
    """Synthesizes final answer from retrieved documents"""
    
    def __init__(self):
        self.max_context_length = 2000
    
    def synthesize(self, query: str, results: List[RetrievalResult], 
                  query_type: QueryType) -> Tuple[str, List[Citation]]:
        """Synthesize answer from retrieved documents"""
        print(f"   ✍️  Synthesizing answer from {len(results)} documents...")
        
        # Extract relevant context
        context = self._extract_context(results)
        
        # Generate answer based on query type
        answer = self._generate_answer(query, context, query_type, results)
        
        # Generate citations
        citations = self._generate_citations(answer, results)
        
        return answer, citations
    
    def _extract_context(self, results: List[RetrievalResult]) -> str:
        """Extract and combine context from documents"""
        context_parts = []
        total_length = 0
        
        for result in results:
            doc_context = f"[{result.document.title}]: {result.document.content}"
            
            if total_length + len(doc_context) > self.max_context_length:
                # Truncate to fit
                remaining = self.max_context_length - total_length
                doc_context = doc_context[:remaining] + "..."
                context_parts.append(doc_context)
                break
            
            context_parts.append(doc_context)
            total_length += len(doc_context)
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str, 
                        query_type: QueryType, 
                        results: List[RetrievalResult]) -> str:
        """Generate answer based on context and query type"""
        # Simulate answer generation (in practice, use LLM)
        
        if query_type == QueryType.FACTUAL:
            answer = f"Based on the retrieved information, {query.lower()} can be answered as follows: "
            answer += f"According to {results[0].document.title}, "
            answer += results[0].document.get_snippet(150)
        
        elif query_type == QueryType.COMPARATIVE:
            answer = f"Comparing the entities in your query: "
            for i, result in enumerate(results[:3], 1):
                answer += f"\n{i}. {result.document.title}: {result.document.get_snippet(100)}"
        
        elif query_type == QueryType.ANALYTICAL:
            answer = f"Analyzing {query}: "
            answer += "Based on multiple sources, we can understand that "
            answer += results[0].document.get_snippet(200)
            if len(results) > 1:
                answer += f" Additionally, {results[1].document.title} suggests that "
                answer += results[1].document.get_snippet(100)
        
        elif query_type == QueryType.AGGREGATIVE:
            answer = f"Aggregating information about {query}: "
            for i, result in enumerate(results, 1):
                answer += f"\n{i}. From {result.document.title}: {result.document.get_snippet(80)}"
        
        elif query_type == QueryType.MULTI_HOP:
            answer = "Following multiple information sources: "
            for i, result in enumerate(results, 1):
                answer += f"\nStep {i} ({result.document.title}): {result.document.get_snippet(100)}"
        
        else:
            answer = f"Regarding {query}: "
            answer += results[0].document.get_snippet(200)
        
        return answer
    
    def _generate_citations(self, answer: str, 
                          results: List[RetrievalResult]) -> List[Citation]:
        """Generate citations for the answer"""
        citations = []
        
        for i, result in enumerate(results, 1):
            citation = Citation(
                document_id=result.document.id,
                document_title=result.document.title,
                snippet=result.document.get_snippet(100),
                relevance=result.relevance_score,
                position=i
            )
            citations.append(citation)
        
        return citations


class AdvancedRAGAgent:
    """Main Advanced RAG agent orchestrating the entire pipeline"""
    
    def __init__(self, documents: List[Document]):
        self.query_planner = QueryPlanner()
        self.retriever = DocumentRetriever(documents)
        self.reranker = DocumentReranker()
        self.synthesizer = AnswerSynthesizer()
        self.query_history: List[RAGResponse] = []
    
    def query(self, query: str, top_k: int = 5) -> RAGResponse:
        """Process a query through the advanced RAG pipeline"""
        start_time = datetime.now()
        
        print(f"\n🔍 Advanced RAG Query Processing")
        print("=" * 60)
        print(f"Query: {query}")
        
        reasoning_trace = []
        
        # Step 1: Plan the query
        reasoning_trace.append("Step 1: Query Planning")
        plan = self.query_planner.plan_query(query)
        reasoning_trace.append(f"  - Query type: {plan.query_type.value}")
        reasoning_trace.append(f"  - Sub-queries: {len(plan.sub_queries)}")
        reasoning_trace.append(f"  - Expected hops: {plan.expected_hops}")
        
        # Step 2: Multi-hop retrieval
        all_results = []
        
        if plan.is_multi_hop():
            print(f"\n🔗 Multi-hop retrieval ({plan.expected_hops} hops)")
            reasoning_trace.append(f"Step 2: Multi-hop Retrieval ({plan.expected_hops} hops)")
            
            for hop, sub_query in enumerate(plan.sub_queries, 1):
                print(f"\n   Hop {hop}: {sub_query}")
                reasoning_trace.append(f"  Hop {hop}: {sub_query}")
                
                strategy = plan.retrieval_strategies.get(sub_query, RetrievalStrategy.HYBRID)
                results = self.retriever.retrieve(sub_query, strategy, top_k)
                
                print(f"   Retrieved {len(results)} documents")
                reasoning_trace.append(f"    Retrieved {len(results)} documents using {strategy.value}")
                
                all_results.extend(results)
        else:
            print(f"\n📥 Single-hop retrieval")
            reasoning_trace.append("Step 2: Single-hop Retrieval")
            
            strategy = plan.retrieval_strategies.get(query, RetrievalStrategy.HYBRID)
            all_results = self.retriever.retrieve(query, strategy, top_k * 2)
            
            print(f"   Retrieved {len(all_results)} documents")
            reasoning_trace.append(f"  Retrieved {len(all_results)} documents using {strategy.value}")
        
        # Step 3: Rerank results
        print(f"\n📊 Reranking and deduplication")
        reasoning_trace.append("Step 3: Reranking")
        
        # Deduplicate
        unique_results = {}
        for result in all_results:
            if result.document.id not in unique_results:
                unique_results[result.document.id] = result
            else:
                # Keep the one with higher score
                if result.score > unique_results[result.document.id].score:
                    unique_results[result.document.id] = result
        
        deduplicated_results = list(unique_results.values())
        print(f"   Deduplicated to {len(deduplicated_results)} unique documents")
        reasoning_trace.append(f"  Deduplicated to {len(deduplicated_results)} documents")
        
        # Rerank
        final_results = self.reranker.rerank(query, deduplicated_results, top_k)
        reasoning_trace.append(f"  Final top-{top_k} documents selected")
        
        # Step 4: Synthesize answer
        print(f"\n📝 Answer synthesis")
        reasoning_trace.append("Step 4: Answer Synthesis")
        
        answer, citations = self.synthesizer.synthesize(query, final_results, plan.query_type)
        reasoning_trace.append(f"  Generated answer with {len(citations)} citations")
        
        # Calculate confidence
        confidence = self._calculate_confidence(final_results, plan)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        response = RAGResponse(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_documents=final_results,
            confidence=confidence,
            reasoning_trace=reasoning_trace,
            execution_time=execution_time
        )
        
        self.query_history.append(response)
        
        print(f"\n✅ Query processed in {execution_time:.2f}s")
        print(f"   Confidence: {confidence:.1%}")
        
        return response
    
    def _calculate_confidence(self, results: List[RetrievalResult], 
                            plan: QueryPlan) -> float:
        """Calculate confidence in the answer"""
        if not results:
            return 0.0
        
        # Base confidence on retrieval scores
        avg_score = sum(r.relevance_score for r in results) / len(results)
        
        # Adjust for query complexity
        complexity_penalty = 0.1 * (plan.expected_hops - 1)
        
        # Adjust for number of results
        if len(results) < 3:
            coverage_penalty = 0.1
        else:
            coverage_penalty = 0.0
        
        confidence = avg_score - complexity_penalty - coverage_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about RAG usage"""
        if not self.query_history:
            return {"message": "No queries processed yet"}
        
        total_queries = len(self.query_history)
        avg_execution_time = sum(r.execution_time for r in self.query_history) / total_queries
        avg_confidence = sum(r.confidence for r in self.query_history) / total_queries
        avg_citations = sum(len(r.citations) for r in self.query_history) / total_queries
        
        return {
            "total_queries": total_queries,
            "average_execution_time": avg_execution_time,
            "average_confidence": avg_confidence,
            "average_citations_per_answer": avg_citations,
            "total_documents_retrieved": sum(len(r.retrieved_documents) for r in self.query_history)
        }


def main():
    """Demonstration of the Advanced RAG pattern"""
    print("🚀 Advanced RAG Pattern Demonstration")
    print("=" * 80)
    print("This demonstrates advanced retrieval-augmented generation:")
    print("- Query planning and decomposition")
    print("- Multi-hop reasoning across documents")
    print("- Document ranking and reranking")
    print("- Answer synthesis with citations")
    
    # Create sample knowledge base
    documents = [
        Document(
            id="doc1",
            title="Introduction to Machine Learning",
            content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience. It focuses on developing computer programs that can access data and use it to learn for themselves. The process involves feeding data to algorithms and allowing them to learn patterns.",
            metadata={"category": "AI", "date": "2023"}
        ),
        Document(
            id="doc2",
            title="Deep Learning Fundamentals",
            content="Deep learning is a specialized subset of machine learning that uses neural networks with multiple layers. These networks can learn hierarchical representations of data. Deep learning has revolutionized fields like computer vision and natural language processing.",
            metadata={"category": "AI", "date": "2023"}
        ),
        Document(
            id="doc3",
            title="Natural Language Processing",
            content="Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. NLP combines computational linguistics with machine learning and deep learning. Applications include chatbots, translation, and sentiment analysis.",
            metadata={"category": "NLP", "date": "2024"}
        ),
        Document(
            id="doc4",
            title="Computer Vision Applications",
            content="Computer vision enables machines to interpret and understand visual information from the world. It uses deep learning models to analyze images and videos. Applications include facial recognition, object detection, autonomous vehicles, and medical image analysis.",
            metadata={"category": "CV", "date": "2024"}
        ),
        Document(
            id="doc5",
            title="Reinforcement Learning",
            content="Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions. This approach has been successful in game playing, robotics, and autonomous systems.",
            metadata={"category": "AI", "date": "2023"}
        ),
        Document(
            id="doc6",
            title="Neural Network Architectures",
            content="Neural networks are computing systems inspired by biological neural networks. Common architectures include feedforward networks, convolutional neural networks (CNNs), and recurrent neural networks (RNNs). Transformers have become the dominant architecture for NLP tasks.",
            metadata={"category": "AI", "date": "2024"}
        ),
        Document(
            id="doc7",
            title="AI Ethics and Safety",
            content="AI ethics addresses the moral implications of artificial intelligence systems. Key concerns include bias in AI systems, privacy, transparency, and accountability. Ensuring AI safety involves developing systems that are reliable, robust, and aligned with human values.",
            metadata={"category": "Ethics", "date": "2024"}
        ),
        Document(
            id="doc8",
            title="Transfer Learning",
            content="Transfer learning is a machine learning technique where a model trained on one task is adapted for another related task. This approach is particularly useful when labeled data is limited. Pre-trained models can be fine-tuned for specific applications, saving time and computational resources.",
            metadata={"category": "AI", "date": "2023"}
        )
    ]
    
    # Create Advanced RAG agent
    rag_agent = AdvancedRAGAgent(documents)
    
    # Test queries
    test_queries = [
        "What is machine learning and how does it work?",
        "Compare deep learning and reinforcement learning",
        "Explain natural language processing and its applications",
        "What are the ethical concerns in AI and how can they be addressed?",
        "How does transfer learning work and why is it useful?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'='*80}")
        print(f"Test Query {i}/{len(test_queries)}")
        print(f"{'='*80}")
        
        response = rag_agent.query(query, top_k=3)
        
        print(f"\n📄 Answer:")
        print(response.answer)
        
        print(f"\n📚 Citations ({len(response.citations)}):")
        for citation in response.citations:
            print(f"  [{citation.position}] {citation.document_title}")
            print(f"      Relevance: {citation.relevance:.2%}")
            print(f"      Snippet: {citation.snippet[:100]}...")
        
        print(f"\n🔍 Reasoning Trace:")
        for step in response.reasoning_trace:
            print(f"  {step}")
        
        input("\nPress Enter to continue to next query...")
    
    # Show statistics
    print(f"\n\n{'='*80}")
    print("📊 RAG System Statistics")
    print(f"{'='*80}")
    
    stats = rag_agent.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    print("\n\n🎯 Key Advanced RAG Features Demonstrated:")
    print("✅ Query planning and decomposition")
    print("✅ Multi-hop reasoning across documents")
    print("✅ Hybrid retrieval strategies (dense + sparse)")
    print("✅ Document reranking for relevance")
    print("✅ Answer synthesis with proper citations")
    print("✅ Confidence scoring")
    print("✅ Query type classification")
    print("✅ Comprehensive reasoning traces")


if __name__ == "__main__":
    main()
