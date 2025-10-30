"""
Pattern 057: Attention Mechanism Patterns

Description:
    Attention mechanisms enable agents to selectively focus on the most relevant
    parts of large inputs, contexts, or action spaces. Inspired by human attention
    and transformer architectures, these patterns help agents efficiently process
    information by allocating computational resources where they matter most.

Components:
    1. Attention Scorer: Calculates relevance scores
    2. Context Selector: Chooses relevant information
    3. Focus Manager: Manages attention allocation
    4. Multi-Head Attention: Multiple attention perspectives
    5. Self-Attention: Internal context relationships

Use Cases:
    - Long document processing
    - Multi-source information synthesis
    - Large context windows
    - Memory retrieval from extensive history
    - Action selection in large action spaces
    - Feature selection in complex data

LangChain Implementation:
    Implements attention scoring and selection mechanisms using LLMs
    to identify and focus on relevant portions of context.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class AttentionType(Enum):
    """Types of attention mechanisms"""
    SELF_ATTENTION = "self_attention"  # Internal relationships
    CROSS_ATTENTION = "cross_attention"  # Query-context relationships
    MULTI_HEAD = "multi_head"  # Multiple attention perspectives
    SPARSE = "sparse"  # Attend to subset
    HIERARCHICAL = "hierarchical"  # Multi-level attention


class FocusMode(Enum):
    """Modes of attention focus"""
    NARROW = "narrow"  # Focus on very specific parts
    BROAD = "broad"  # Consider wider context
    ADAPTIVE = "adaptive"  # Adjust based on task
    DISTRIBUTED = "distributed"  # Spread across multiple items


@dataclass
class AttentionScore:
    """Attention score for an item"""
    item_id: str
    content: str
    score: float  # 0.0-1.0
    reasoning: Optional[str] = None
    
    def __repr__(self) -> str:
        return f"AttentionScore(id={self.item_id}, score={self.score:.3f})"


@dataclass
class AttentionHead:
    """Single attention head with specific perspective"""
    head_id: str
    perspective: str
    scores: List[AttentionScore]
    
    @property
    def top_items(self, k: int = 3) -> List[AttentionScore]:
        return sorted(self.scores, key=lambda x: x.score, reverse=True)[:k]


@dataclass
class AttentionResult:
    """Result from attention mechanism"""
    query: str
    attended_items: List[AttentionScore]
    attention_type: AttentionType
    focus_mode: FocusMode
    total_items: int
    items_attended: int
    attention_distribution: Dict[str, float]
    execution_time_ms: float
    
    @property
    def coverage(self) -> float:
        """Percentage of items attended"""
        return self.items_attended / self.total_items if self.total_items > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:80] + "...",
            "attention_type": self.attention_type.value,
            "focus_mode": self.focus_mode.value,
            "coverage": f"{self.coverage:.1%}",
            "top_attended": [
                f"{item.item_id} ({item.score:.2f})"
                for item in self.attended_items[:3]
            ],
            "execution_time_ms": f"{self.execution_time_ms:.1f}"
        }


class AttentionMechanism:
    """
    Implements attention mechanisms for selective focus.
    
    Features:
    1. Query-based attention scoring
    2. Multi-head attention for diverse perspectives
    3. Sparse attention for efficiency
    4. Hierarchical attention for structured data
    5. Adaptive focus modes
    """
    
    def __init__(
        self,
        attention_type: AttentionType = AttentionType.CROSS_ATTENTION,
        focus_mode: FocusMode = FocusMode.ADAPTIVE,
        num_heads: int = 3,
        temperature: float = 0.3
    ):
        self.attention_type = attention_type
        self.focus_mode = focus_mode
        self.num_heads = num_heads
        self.temperature = temperature
        
        # Attention scorer LLM
        self.scorer = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=temperature
        )
        
        # Synthesizer for multi-head combination
        self.synthesizer = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2
        )
    
    def _score_relevance(
        self,
        query: str,
        item: str,
        item_id: str,
        perspective: str = "general"
    ) -> AttentionScore:
        """Score relevance of item to query"""
        
        scoring_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are scoring relevance from a {perspective} perspective.

Rate how relevant the item is to the query on a scale of 0.0 to 1.0.

Respond with just the number (e.g., 0.85)"""),
            ("user", "Query: {query}\n\nItem: {item}\n\nRelevance score:")
        ])
        
        chain = scoring_prompt | self.scorer | StrOutputParser()
        
        try:
            score_str = chain.invoke({"query": query, "item": item}).strip()
            # Extract number from response
            score = float(''.join(c for c in score_str if c.isdigit() or c == '.'))
            score = max(0.0, min(1.0, score))
        except:
            # Fallback: simple keyword matching
            query_words = set(query.lower().split())
            item_words = set(item.lower().split())
            overlap = len(query_words & item_words)
            score = min(1.0, overlap / max(len(query_words), 1) * 2)
        
        return AttentionScore(
            item_id=item_id,
            content=item,
            score=score,
            reasoning=f"{perspective} relevance"
        )
    
    def _cross_attention(
        self,
        query: str,
        context_items: List[Tuple[str, str]],  # (id, content)
        top_k: int = 5
    ) -> List[AttentionScore]:
        """Cross-attention: query attending to context"""
        
        scores = []
        
        for item_id, content in context_items:
            score = self._score_relevance(query, content, item_id)
            scores.append(score)
        
        # Apply softmax-like normalization
        total = sum(s.score for s in scores)
        if total > 0:
            for score in scores:
                score.score = score.score / total
        
        # Return top-k
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:top_k]
    
    def _self_attention(
        self,
        items: List[Tuple[str, str]],  # (id, content)
        top_k: int = 5
    ) -> List[AttentionScore]:
        """Self-attention: items attending to each other"""
        
        # For each item, calculate its relationship to all others
        attention_matrix = []
        
        for i, (id1, content1) in enumerate(items):
            row_scores = []
            for j, (id2, content2) in enumerate(items):
                if i != j:
                    # Simple similarity (could use embeddings)
                    words1 = set(content1.lower().split())
                    words2 = set(content2.lower().split())
                    similarity = len(words1 & words2) / max(len(words1 | words2), 1)
                    row_scores.append((j, similarity))
            
            # Sum of attention from this item to others
            total_attention = sum(s for _, s in row_scores)
            attention_matrix.append((i, total_attention))
        
        # Sort by total attention received
        attention_matrix.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k most attended items
        result = []
        for idx, attention in attention_matrix[:top_k]:
            item_id, content = items[idx]
            result.append(AttentionScore(
                item_id=item_id,
                content=content,
                score=attention,
                reasoning="Self-attention importance"
            ))
        
        # Normalize scores
        total = sum(s.score for s in result)
        if total > 0:
            for score in result:
                score.score = score.score / total
        
        return result
    
    def _multi_head_attention(
        self,
        query: str,
        context_items: List[Tuple[str, str]],
        perspectives: List[str],
        top_k: int = 5
    ) -> Tuple[List[AttentionScore], List[AttentionHead]]:
        """Multi-head attention: multiple perspectives"""
        
        heads = []
        
        # Each head has different perspective
        for i, perspective in enumerate(perspectives[:self.num_heads]):
            head_scores = []
            
            for item_id, content in context_items:
                score = self._score_relevance(
                    query, content, item_id, perspective=perspective
                )
                head_scores.append(score)
            
            head = AttentionHead(
                head_id=f"head_{i}",
                perspective=perspective,
                scores=head_scores
            )
            heads.append(head)
        
        # Combine heads (average scores)
        combined_scores = {}
        for head in heads:
            for score in head.scores:
                if score.item_id not in combined_scores:
                    combined_scores[score.item_id] = {
                        "content": score.content,
                        "scores": []
                    }
                combined_scores[score.item_id]["scores"].append(score.score)
        
        # Average across heads
        final_scores = []
        for item_id, data in combined_scores.items():
            avg_score = sum(data["scores"]) / len(data["scores"])
            final_scores.append(AttentionScore(
                item_id=item_id,
                content=data["content"],
                score=avg_score,
                reasoning="Multi-head average"
            ))
        
        # Normalize and return top-k
        total = sum(s.score for s in final_scores)
        if total > 0:
            for score in final_scores:
                score.score = score.score / total
        
        final_scores.sort(key=lambda x: x.score, reverse=True)
        
        return final_scores[:top_k], heads
    
    def _sparse_attention(
        self,
        query: str,
        context_items: List[Tuple[str, str]],
        top_k: int = 5
    ) -> List[AttentionScore]:
        """Sparse attention: only attend to subset"""
        
        # Quick filtering using keywords
        query_keywords = set(query.lower().split())
        
        # Score only items with keyword overlap
        candidate_items = []
        for item_id, content in context_items:
            content_words = set(content.lower().split())
            if query_keywords & content_words:  # Has overlap
                candidate_items.append((item_id, content))
        
        # If too few candidates, expand
        if len(candidate_items) < top_k:
            candidate_items = context_items[:top_k * 2]
        
        # Full attention on candidates only
        return self._cross_attention(query, candidate_items, top_k)
    
    def attend(
        self,
        query: str,
        context_items: List[Tuple[str, str]],
        top_k: int = 5
    ) -> AttentionResult:
        """Apply attention mechanism"""
        
        start_time = time.time()
        
        total_items = len(context_items)
        
        # Apply appropriate attention type
        if self.attention_type == AttentionType.CROSS_ATTENTION:
            attended = self._cross_attention(query, context_items, top_k)
            
        elif self.attention_type == AttentionType.SELF_ATTENTION:
            attended = self._self_attention(context_items, top_k)
            
        elif self.attention_type == AttentionType.MULTI_HEAD:
            perspectives = ["factual", "emotional", "practical"]
            attended, heads = self._multi_head_attention(
                query, context_items, perspectives, top_k
            )
            
        elif self.attention_type == AttentionType.SPARSE:
            attended = self._sparse_attention(query, context_items, top_k)
            
        else:  # Default to cross-attention
            attended = self._cross_attention(query, context_items, top_k)
        
        # Calculate attention distribution
        distribution = {
            item.item_id: item.score
            for item in attended
        }
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return AttentionResult(
            query=query,
            attended_items=attended,
            attention_type=self.attention_type,
            focus_mode=self.focus_mode,
            total_items=total_items,
            items_attended=len(attended),
            attention_distribution=distribution,
            execution_time_ms=execution_time_ms
        )
    
    def focused_response(
        self,
        query: str,
        context_items: List[Tuple[str, str]],
        top_k: int = 3
    ) -> Tuple[str, AttentionResult]:
        """Generate response using attended context"""
        
        # Apply attention
        attention = self.attend(query, context_items, top_k)
        
        # Build focused context
        focused_context = "\n\n".join([
            f"[Relevance: {item.score:.2f}] {item.content}"
            for item in attention.attended_items
        ])
        
        # Generate response with focused context
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You have access to focused, relevant context.
Use this context to provide an accurate, helpful response.

Relevant Context:
{context}"""),
            ("user", "{query}")
        ])
        
        chain = response_prompt | self.synthesizer | StrOutputParser()
        response = chain.invoke({
            "query": query,
            "context": focused_context
        })
        
        return response, attention


def demonstrate_attention_mechanisms():
    """Demonstrate Attention Mechanism patterns"""
    
    print("=" * 80)
    print("PATTERN 057: ATTENTION MECHANISM PATTERNS DEMONSTRATION")
    print("=" * 80)
    print("\nSelective focus on relevant information\n")
    
    # Create sample context
    documents = [
        ("doc1", "Python is a high-level programming language known for its simplicity and readability."),
        ("doc2", "Machine learning models require large amounts of training data to perform well."),
        ("doc3", "The transformer architecture uses self-attention mechanisms for sequence processing."),
        ("doc4", "Python's simplicity makes it popular for machine learning and data science applications."),
        ("doc5", "Deep learning frameworks like TensorFlow and PyTorch are commonly used in Python."),
        ("doc6", "Attention mechanisms allow models to focus on relevant parts of the input."),
        ("doc7", "Natural language processing has benefited greatly from attention-based models."),
        ("doc8", "Python's extensive libraries make it suitable for various AI tasks."),
    ]
    
    # Test 1: Cross-attention
    print("\n" + "=" * 80)
    print("TEST 1: Cross-Attention (Query attending to Context)")
    print("=" * 80)
    
    attn1 = AttentionMechanism(
        attention_type=AttentionType.CROSS_ATTENTION,
        focus_mode=FocusMode.NARROW
    )
    
    query1 = "What is Python used for in machine learning?"
    
    print(f"\n💭 Query: {query1}")
    print(f"📚 Total Documents: {len(documents)}")
    
    result1 = attn1.attend(query1, documents, top_k=3)
    
    print(f"\n🎯 Attention Results:")
    print(f"   Coverage: {result1.coverage:.1%} ({result1.items_attended}/{result1.total_items})")
    print(f"   Execution Time: {result1.execution_time_ms:.1f}ms")
    
    print(f"\n📊 Top Attended Items:")
    for i, item in enumerate(result1.attended_items, 1):
        print(f"   {i}. {item.item_id} (score: {item.score:.3f})")
        print(f"      {item.content[:80]}...")
    
    # Test 2: Self-attention
    print("\n" + "=" * 80)
    print("TEST 2: Self-Attention (Items attending to each other)")
    print("=" * 80)
    
    attn2 = AttentionMechanism(
        attention_type=AttentionType.SELF_ATTENTION,
        focus_mode=FocusMode.ADAPTIVE
    )
    
    result2 = attn2.attend("", documents, top_k=4)  # No query for self-attention
    
    print(f"\n🔗 Most Interconnected Documents:")
    for i, item in enumerate(result2.attended_items, 1):
        print(f"   {i}. {item.item_id} (attention: {item.score:.3f})")
        print(f"      {item.content[:80]}...")
    
    # Test 3: Multi-head attention
    print("\n" + "=" * 80)
    print("TEST 3: Multi-Head Attention (Multiple Perspectives)")
    print("=" * 80)
    
    attn3 = AttentionMechanism(
        attention_type=AttentionType.MULTI_HEAD,
        num_heads=3
    )
    
    query3 = "Explain attention mechanisms"
    
    print(f"\n💭 Query: {query3}")
    print(f"👁️  Perspectives: factual, emotional, practical")
    
    result3 = attn3.attend(query3, documents, top_k=3)
    
    print(f"\n📊 Combined Multi-Head Results:")
    for i, item in enumerate(result3.attended_items, 1):
        print(f"   {i}. {item.item_id} (combined score: {item.score:.3f})")
        print(f"      {item.content[:80]}...")
    
    # Test 4: Focused response generation
    print("\n" + "=" * 80)
    print("TEST 4: Focused Response Generation")
    print("=" * 80)
    
    attn4 = AttentionMechanism(
        attention_type=AttentionType.CROSS_ATTENTION,
        focus_mode=FocusMode.NARROW
    )
    
    query4 = "Why is Python popular for AI development?"
    
    print(f"\n💭 Query: {query4}")
    
    response, attention = attn4.focused_response(query4, documents, top_k=3)
    
    print(f"\n🎯 Attention Focus:")
    for item in attention.attended_items:
        print(f"   - {item.item_id}: {item.score:.3f}")
    
    print(f"\n💬 Focused Response:")
    print(f"   {response}")
    
    # Test 5: Sparse attention
    print("\n" + "=" * 80)
    print("TEST 5: Sparse Attention (Efficient Processing)")
    print("=" * 80)
    
    # Create larger context
    large_docs = documents + [
        (f"doc{i}", f"This is an irrelevant document number {i}.")
        for i in range(9, 20)
    ]
    
    attn5 = AttentionMechanism(
        attention_type=AttentionType.SPARSE,
        focus_mode=FocusMode.ADAPTIVE
    )
    
    query5 = "transformer architecture attention"
    
    print(f"\n💭 Query: {query5}")
    print(f"📚 Total Documents: {len(large_docs)}")
    
    result5 = attn5.attend(query5, large_docs, top_k=3)
    
    print(f"\n⚡ Sparse Attention Results:")
    print(f"   Only scored relevant subset")
    print(f"   Execution Time: {result5.execution_time_ms:.1f}ms")
    
    print(f"\n🎯 Top Results:")
    for i, item in enumerate(result5.attended_items, 1):
        print(f"   {i}. {item.item_id}: {item.score:.3f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("ATTENTION MECHANISM PATTERNS SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Selective Focus: Process only relevant information
2. Scalability: Handle large contexts efficiently
3. Interpretability: Attention scores show reasoning
4. Quality: Focus improves response accuracy
5. Efficiency: Reduce computational cost

Attention Types:
1. Cross-Attention: Query attending to context
   - Use: Question answering, retrieval
   - Pattern: Query → Context scoring

2. Self-Attention: Internal relationships
   - Use: Document clustering, summarization
   - Pattern: Items → Mutual relevance

3. Multi-Head Attention: Multiple perspectives
   - Use: Complex analysis, diverse views
   - Pattern: Parallel attention heads → Combine

4. Sparse Attention: Subset focus
   - Use: Large-scale processing
   - Pattern: Filter → Attend to subset

5. Hierarchical Attention: Multi-level focus
   - Use: Structured documents
   - Pattern: Coarse → Fine attention

Focus Modes:
- Narrow: Highly selective (top-1 or top-3)
- Broad: Consider wider context (top-10+)
- Adaptive: Adjust based on task
- Distributed: Spread attention evenly

Scoring Methods:
1. LLM-based: Relevance scoring by LLM
2. Embedding-based: Cosine similarity
3. Keyword: Simple word overlap
4. Learned: Trained attention weights

Applications:
- Long document Q&A
- Multi-document synthesis
- Memory retrieval
- Context selection
- Feature importance
- Action space filtering

Implementation Patterns:
1. Score → Normalize → Select top-k
2. Multiple heads → Average/Weighted combine
3. Hierarchical: Sentence → Word attention
4. Causal: Attend only to past
5. Bidirectional: Full context attention

Best Practices:
1. Choose appropriate top-k
2. Normalize attention scores (softmax)
3. Multi-head for complex tasks
4. Sparse for large contexts
5. Cache attention scores
6. Visualize attention patterns
7. Combine with retrieval

Production Considerations:
- Caching scored items
- Parallel scoring
- Incremental attention updates
- Attention visualization
- Performance profiling
- Memory management
- Score normalization methods

Comparison with Related Patterns:
- vs. Retrieval: Attention vs search
- vs. Filtering: Soft vs hard selection
- vs. Ranking: Continuous scores vs discrete order
- vs. Memory: Attention for retrieval

Attention mechanisms are fundamental for efficient processing
of large contexts, enabling agents to focus computational
resources where they matter most for the task at hand.
""")


if __name__ == "__main__":
    demonstrate_attention_mechanisms()
