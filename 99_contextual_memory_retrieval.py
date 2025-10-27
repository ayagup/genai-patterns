"""
Agentic AI Design Pattern: Contextual Memory Retrieval

This pattern implements sophisticated memory retrieval that adapts to current
context, uses semantic similarity, and applies intelligent ranking to find
the most relevant memories for the current situation.

Key Concepts:
1. Context Awareness: Understanding current situation/task
2. Semantic Similarity: Finding conceptually related memories
3. Relevance Scoring: Ranking memories by relevance to context
4. Adaptive Retrieval: Adjusting retrieval based on task requirements
5. Multi-Modal Context: Combining different types of contextual information

Use Cases:
- Conversational agents maintaining context
- Recommendation systems
- Personal assistants with contextual awareness
- Knowledge-based systems
- Learning agents adapting to situations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import uuid
import re
import math


class ContextType(Enum):
    """Types of context for memory retrieval"""
    TASK = "task"  # Current task being performed
    TEMPORAL = "temporal"  # Time-based context
    SEMANTIC = "semantic"  # Meaning-based context
    SITUATIONAL = "situational"  # Current situation
    USER = "user"  # User-specific context
    DOMAIN = "domain"  # Domain/topic context


@dataclass
class Context:
    """Represents current context for memory retrieval"""
    context_type: ContextType
    keywords: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    importance_weight: float = 1.0
    
    def matches_keyword(self, keyword: str) -> bool:
        """Check if context contains keyword"""
        keyword_lower = keyword.lower()
        return any(kw.lower() == keyword_lower or keyword_lower in kw.lower() 
                  for kw in self.keywords)
    
    def keyword_overlap(self, other_keywords: Set[str]) -> float:
        """Calculate keyword overlap with other set"""
        if not self.keywords or not other_keywords:
            return 0.0
        
        # Normalize keywords
        self_lower = {kw.lower() for kw in self.keywords}
        other_lower = {kw.lower() for kw in other_keywords}
        
        intersection = self_lower & other_lower
        union = self_lower | other_lower
        
        # Jaccard similarity
        return len(intersection) / len(union) if union else 0.0


@dataclass
class ContextualMemory:
    """Memory entry with contextual information"""
    memory_id: str
    content: Any
    context: List[Context]
    timestamp: datetime
    access_count: int = 0
    last_access: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    embedding: Optional[List[float]] = None  # For semantic similarity
    
    def __post_init__(self):
        if not self.memory_id:
            self.memory_id = str(uuid.uuid4())
        if self.last_access is None:
            self.last_access = self.timestamp
    
    def access(self) -> None:
        """Record access to memory"""
        self.access_count += 1
        self.last_access = datetime.now()
    
    def get_all_keywords(self) -> Set[str]:
        """Get all keywords from all contexts"""
        keywords = set(self.tags)
        for ctx in self.context:
            keywords.update(ctx.keywords)
        return keywords
    
    def context_relevance(self, query_context: List[Context]) -> float:
        """Calculate relevance to query context"""
        if not query_context or not self.context:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for q_ctx in query_context:
            best_match = 0.0
            for m_ctx in self.context:
                # Same context type is more relevant
                type_bonus = 1.5 if q_ctx.context_type == m_ctx.context_type else 1.0
                
                # Calculate keyword overlap
                overlap = q_ctx.keyword_overlap(m_ctx.keywords)
                match_score = overlap * type_bonus
                
                best_match = max(best_match, match_score)
            
            total_score += best_match * q_ctx.importance_weight
            total_weight += q_ctx.importance_weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def recency_score(self) -> float:
        """Calculate recency score"""
        if self.last_access is None:
            return 0.0
        
        hours_since = (datetime.now() - self.last_access).total_seconds() / 3600
        # Exponential decay
        return math.exp(-hours_since / 24.0)
    
    def frequency_score(self) -> float:
        """Calculate frequency score"""
        # Logarithmic scaling
        return math.log(1 + self.access_count) / math.log(100)


@dataclass
class RetrievalQuery:
    """Query for contextual memory retrieval"""
    query_text: str
    context: List[Context]
    max_results: int = 10
    min_relevance: float = 0.3
    include_recent: bool = True
    include_frequent: bool = True
    recency_weight: float = 0.3
    relevance_weight: float = 0.5
    frequency_weight: float = 0.2
    
    def extract_keywords(self) -> Set[str]:
        """Extract keywords from query text"""
        # Simple keyword extraction (in practice, use NLP)
        words = re.findall(r'\b\w+\b', self.query_text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        return {w for w in words if w not in stop_words and len(w) > 2}


@dataclass
class RetrievalResult:
    """Result of contextual memory retrieval"""
    memory: ContextualMemory
    relevance_score: float
    recency_score: float
    frequency_score: float
    combined_score: float
    explanation: str = ""
    
    def __str__(self) -> str:
        return (f"Result(memory={str(self.memory.content)[:50]}, "
                f"score={self.combined_score:.3f})")


class SemanticSimilarityCalculator:
    """Calculates semantic similarity between texts"""
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity (simplified)"""
        # In practice, use embeddings and cosine similarity
        # Here we use simple word overlap as approximation
        
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        # Jaccard similarity
        return len(intersection) / len(union)
    
    def calculate_embedding_similarity(self, embedding1: List[float], 
                                      embedding2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        if not embedding1 or not embedding2:
            return 0.0
        
        if len(embedding1) != len(embedding2):
            return 0.0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        mag1 = math.sqrt(sum(a * a for a in embedding1))
        mag2 = math.sqrt(sum(b * b for b in embedding2))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)


class ContextualMemoryRetriever:
    """Retrieves memories based on context"""
    
    def __init__(self):
        self.memories: Dict[str, ContextualMemory] = {}
        self.semantic_calculator = SemanticSimilarityCalculator()
        self.context_history: List[List[Context]] = []
    
    def store(self, content: Any, context: List[Context], 
             tags: Optional[Set[str]] = None) -> str:
        """Store memory with context"""
        memory = ContextualMemory(
            memory_id=str(uuid.uuid4()),
            content=content,
            context=context,
            timestamp=datetime.now(),
            tags=tags or set()
        )
        
        self.memories[memory.memory_id] = memory
        return memory.memory_id
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Retrieve memories based on contextual query"""
        results = []
        
        # Extract query keywords
        query_keywords = query.extract_keywords()
        
        for memory in self.memories.values():
            # Calculate different score components
            
            # 1. Context relevance
            context_score = memory.context_relevance(query.context)
            
            # 2. Keyword overlap
            memory_keywords = memory.get_all_keywords()
            keyword_score = 0.0
            if query_keywords and memory_keywords:
                intersection = query_keywords & {k.lower() for k in memory_keywords}
                union = query_keywords | {k.lower() for k in memory_keywords}
                keyword_score = len(intersection) / len(union) if union else 0.0
            
            # 3. Semantic similarity
            semantic_score = self.semantic_calculator.calculate_similarity(
                query.query_text, 
                str(memory.content)
            )
            
            # Combined relevance score
            relevance_score = (context_score * 0.4 + 
                             keyword_score * 0.3 + 
                             semantic_score * 0.3)
            
            # Skip if below minimum relevance
            if relevance_score < query.min_relevance:
                continue
            
            # Recency and frequency scores
            recency_score = memory.recency_score() if query.include_recent else 0.0
            frequency_score = memory.frequency_score() if query.include_frequent else 0.0
            
            # Combined final score
            combined_score = (
                relevance_score * query.relevance_weight +
                recency_score * query.recency_weight +
                frequency_score * query.frequency_weight
            )
            
            # Create explanation
            explanation = self._generate_explanation(
                context_score, keyword_score, semantic_score, 
                recency_score, frequency_score
            )
            
            result = RetrievalResult(
                memory=memory,
                relevance_score=relevance_score,
                recency_score=recency_score,
                frequency_score=frequency_score,
                combined_score=combined_score,
                explanation=explanation
            )
            
            results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda r: r.combined_score, reverse=True)
        
        # Update memory access
        for result in results[:query.max_results]:
            result.memory.access()
        
        # Store context history
        self.context_history.append(query.context)
        
        return results[:query.max_results]
    
    def _generate_explanation(self, context_score: float, keyword_score: float,
                            semantic_score: float, recency_score: float,
                            frequency_score: float) -> str:
        """Generate explanation for retrieval"""
        components = []
        
        if context_score > 0.5:
            components.append(f"strong context match ({context_score:.2f})")
        if keyword_score > 0.5:
            components.append(f"high keyword overlap ({keyword_score:.2f})")
        if semantic_score > 0.5:
            components.append(f"semantic similarity ({semantic_score:.2f})")
        if recency_score > 0.7:
            components.append("recently accessed")
        if frequency_score > 0.5:
            components.append("frequently accessed")
        
        if not components:
            return "weak match"
        
        return ", ".join(components)
    
    def retrieve_with_adaptive_threshold(self, query: RetrievalQuery,
                                        target_count: int = 5) -> List[RetrievalResult]:
        """Adaptively adjust threshold to get desired result count"""
        thresholds = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
        
        for threshold in thresholds:
            query.min_relevance = threshold
            results = self.retrieve(query)
            
            if len(results) >= target_count:
                return results[:target_count]
        
        return self.retrieve(query)
    
    def get_context_patterns(self) -> Dict[ContextType, int]:
        """Analyze context usage patterns"""
        patterns: Dict[ContextType, int] = {}
        
        for contexts in self.context_history:
            for ctx in contexts:
                patterns[ctx.context_type] = patterns.get(ctx.context_type, 0) + 1
        
        return patterns


class AdaptiveContextBuilder:
    """Builds context adaptively based on situation"""
    
    def __init__(self):
        self.context_templates: Dict[str, List[ContextType]] = {
            "conversation": [ContextType.TASK, ContextType.TEMPORAL, ContextType.USER],
            "search": [ContextType.SEMANTIC, ContextType.DOMAIN],
            "recommendation": [ContextType.USER, ContextType.SITUATIONAL, ContextType.TEMPORAL],
            "analysis": [ContextType.DOMAIN, ContextType.SEMANTIC, ContextType.TASK],
        }
    
    def build_context(self, situation: str, keywords: Set[str],
                     metadata: Optional[Dict[str, Any]] = None) -> List[Context]:
        """Build context for given situation"""
        context_types = self.context_templates.get(situation, [ContextType.TASK])
        
        contexts = []
        for ctx_type in context_types:
            # Adjust keywords and importance based on context type
            ctx_keywords = keywords.copy()
            importance = 1.0
            
            if ctx_type == ContextType.TEMPORAL:
                # Add time-based keywords
                now = datetime.now()
                ctx_keywords.add(f"{now.hour}:00")
                ctx_keywords.add(now.strftime("%A"))
            elif ctx_type == ContextType.USER:
                importance = 1.5  # User context is more important
            
            contexts.append(Context(
                context_type=ctx_type,
                keywords=ctx_keywords,
                metadata=metadata or {},
                importance_weight=importance
            ))
        
        return contexts


def demonstrate_contextual_memory_retrieval():
    """Demonstrate contextual memory retrieval"""
    print("=" * 70)
    print("CONTEXTUAL MEMORY RETRIEVAL DEMONSTRATION")
    print("=" * 70)
    
    # Create retriever and context builder
    retriever = ContextualMemoryRetriever()
    context_builder = AdaptiveContextBuilder()
    
    print("\n1. STORING MEMORIES WITH DIFFERENT CONTEXTS")
    print("-" * 70)
    
    # Store memories with various contexts
    memories_data = [
        ("User asked about Python programming", "conversation", 
         {"python", "programming", "question"}),
        ("Discussed machine learning algorithms", "conversation",
         {"machine learning", "algorithms", "AI"}),
        ("User preference: prefers concise answers", "conversation",
         {"user", "preference", "concise"}),
        ("Python is a high-level programming language", "search",
         {"python", "programming", "language"}),
        ("Neural networks are used in deep learning", "search",
         {"neural", "networks", "deep learning"}),
        ("User completed Python tutorial yesterday", "recommendation",
         {"python", "tutorial", "learning"}),
        ("Code review best practices", "analysis",
         {"code", "review", "best practices"}),
        ("Debugging techniques for Python", "analysis",
         {"debugging", "python", "techniques"}),
    ]
    
    stored_ids = []
    for content, situation, keywords in memories_data:
        context = context_builder.build_context(situation, keywords)
        memory_id = retriever.store(content, context, keywords)
        stored_ids.append(memory_id)
        print(f"   Stored: {content[:50]:<50} | {situation}")
    
    print(f"\n   Total memories stored: {len(stored_ids)}")
    
    print("\n2. CONTEXT-BASED RETRIEVAL")
    print("-" * 70)
    
    # Query 1: User asking about Python
    print("\n   Query: 'Tell me about Python programming'")
    query1_context = context_builder.build_context(
        "conversation",
        {"python", "programming", "tell"}
    )
    query1 = RetrievalQuery(
        query_text="Tell me about Python programming",
        context=query1_context,
        max_results=3
    )
    
    results1 = retriever.retrieve(query1)
    print(f"   Retrieved {len(results1)} memories:")
    for i, result in enumerate(results1, 1):
        print(f"     {i}. {str(result.memory.content)[:55]}")
        print(f"        Score: {result.combined_score:.3f} "
              f"(rel={result.relevance_score:.2f}, "
              f"rec={result.recency_score:.2f}, "
              f"freq={result.frequency_score:.2f})")
        print(f"        Reason: {result.explanation}")
    
    # Query 2: Looking for debugging info
    print("\n   Query: 'How do I debug Python code?'")
    query2_context = context_builder.build_context(
        "search",
        {"debug", "python", "code"}
    )
    query2 = RetrievalQuery(
        query_text="How do I debug Python code?",
        context=query2_context,
        max_results=3
    )
    
    results2 = retriever.retrieve(query2)
    print(f"   Retrieved {len(results2)} memories:")
    for i, result in enumerate(results2, 1):
        print(f"     {i}. {str(result.memory.content)[:55]}")
        print(f"        Score: {result.combined_score:.3f}")
    
    print("\n3. ADAPTIVE THRESHOLD RETRIEVAL")
    print("-" * 70)
    
    # Query with adaptive threshold
    print("\n   Query: 'Machine learning recommendations'")
    query3_context = context_builder.build_context(
        "recommendation",
        {"machine", "learning"}
    )
    query3 = RetrievalQuery(
        query_text="Machine learning recommendations",
        context=query3_context,
        max_results=10
    )
    
    adaptive_results = retriever.retrieve_with_adaptive_threshold(query3, target_count=3)
    print(f"   Retrieved {len(adaptive_results)} memories (target: 3):")
    for i, result in enumerate(adaptive_results, 1):
        print(f"     {i}. {str(result.memory.content)[:55]}")
        print(f"        Score: {result.combined_score:.3f}, "
              f"Threshold used: {query3.min_relevance:.2f}")
    
    print("\n4. WEIGHTED RETRIEVAL (EMPHASIZE RECENCY)")
    print("-" * 70)
    
    # Access some memories to change recency
    retriever.memories[stored_ids[0]].access()
    retriever.memories[stored_ids[0]].access()
    retriever.memories[stored_ids[1]].access()
    
    print("\n   Query: 'Python discussion' (recency-weighted)")
    query4_context = context_builder.build_context(
        "conversation",
        {"python", "discussion"}
    )
    query4 = RetrievalQuery(
        query_text="Python discussion",
        context=query4_context,
        max_results=3,
        recency_weight=0.6,  # Emphasize recent
        relevance_weight=0.3,
        frequency_weight=0.1
    )
    
    results4 = retriever.retrieve(query4)
    print(f"   Retrieved {len(results4)} memories:")
    for i, result in enumerate(results4, 1):
        print(f"     {i}. {str(result.memory.content)[:55]}")
        print(f"        Combined: {result.combined_score:.3f} "
              f"(recency weighted higher)")
        print(f"        Access count: {result.memory.access_count}")
    
    print("\n5. CONTEXT USAGE PATTERNS")
    print("-" * 70)
    
    patterns = retriever.get_context_patterns()
    print("   Context type usage:")
    for ctx_type, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"     {ctx_type.value:>15}: {count:>2} times")
    
    print("\n6. MULTI-CONTEXT QUERY")
    print("-" * 70)
    
    # Create query with multiple explicit contexts
    multi_context = [
        Context(ContextType.TASK, {"python", "learning"}, importance_weight=1.5),
        Context(ContextType.USER, {"tutorial", "beginner"}, importance_weight=1.2),
        Context(ContextType.DOMAIN, {"programming"}, importance_weight=1.0)
    ]
    
    query5 = RetrievalQuery(
        query_text="Python learning resources",
        context=multi_context,
        max_results=3
    )
    
    print("   Query with 3 explicit contexts:")
    print("     - TASK: python, learning (weight=1.5)")
    print("     - USER: tutorial, beginner (weight=1.2)")
    print("     - DOMAIN: programming (weight=1.0)")
    
    results5 = retriever.retrieve(query5)
    print(f"\n   Retrieved {len(results5)} memories:")
    for i, result in enumerate(results5, 1):
        print(f"     {i}. {str(result.memory.content)[:55]}")
        print(f"        Score: {result.combined_score:.3f}")
        print(f"        Explanation: {result.explanation}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("1. Context-aware memory storage and retrieval")
    print("2. Multi-dimensional scoring (relevance, recency, frequency)")
    print("3. Adaptive threshold adjustment")
    print("4. Weighted retrieval with configurable priorities")
    print("5. Context usage pattern analysis")
    print("6. Multi-context queries with importance weighting")
    print("7. Automatic keyword extraction and semantic matching")
    print("8. Explainable retrieval with reasoning")


if __name__ == "__main__":
    demonstrate_contextual_memory_retrieval()
