"""
Attention Mechanism Patterns for Agentic AI

This pattern implements various attention mechanisms that allow agents to
focus on relevant information when processing long contexts, multiple inputs,
or complex information streams.

Key Concepts:
1. Self-Attention - Attend to different parts of the same sequence
2. Cross-Attention - Attend between two different sequences
3. Multi-Head Attention - Parallel attention with different learned patterns
4. Soft Attention - Weighted combination of all inputs
5. Hard Attention - Select specific inputs (discrete)
6. Contextual Attention - Focus based on task context

Use Cases:
- Long document processing
- Multi-modal information integration
- Selective information retrieval
- Context-aware reasoning
- Dynamic focus in conversations
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class AttentionScore:
    """Represents an attention score for a key-value pair"""
    key: str
    value: Any
    score: float
    weight: float  # Normalized score


@dataclass
class AttentionOutput:
    """Output from attention mechanism"""
    context: Any
    attention_weights: List[AttentionScore]
    focused_items: List[str]


class SoftAttention:
    """
    Soft attention mechanism - computes weighted sum over all inputs.
    
    Uses similarity scores to weight the importance of different inputs.
    """
    
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def compute_similarity(self, query: str, key: str) -> float:
        """
        Compute similarity between query and key.
        In real implementation, would use embeddings/vectors.
        """
        # Simple word overlap for demo
        query_words = set(query.lower().split())
        key_words = set(key.lower().split())
        
        if not query_words or not key_words:
            return 0.0
        
        overlap = len(query_words & key_words)
        union = len(query_words | key_words)
        
        return overlap / union if union > 0 else 0.0
    
    def softmax(self, scores: List[float]) -> List[float]:
        """Apply softmax with temperature"""
        scaled_scores = [s / self.temperature for s in scores]
        max_score = max(scaled_scores) if scaled_scores else 0
        exp_scores = [math.exp(s - max_score) for s in scaled_scores]
        total = sum(exp_scores)
        return [s / total for s in exp_scores] if total > 0 else [0.0] * len(scores)
    
    def attend(
        self,
        query: str,
        keys: List[str],
        values: List[Any]
    ) -> AttentionOutput:
        """
        Apply soft attention mechanism.
        
        Args:
            query: Query to focus attention
            keys: List of keys to attend over
            values: Corresponding values
        
        Returns:
            AttentionOutput with weighted context
        """
        if len(keys) != len(values):
            raise ValueError("Keys and values must have same length")
        
        # Compute similarity scores
        scores = [self.compute_similarity(query, key) for key in keys]
        
        # Apply softmax to get weights
        weights = self.softmax(scores)
        
        # Create attention scores
        attention_scores = [
            AttentionScore(key=k, value=v, score=s, weight=w)
            for k, v, s, w in zip(keys, values, scores, weights)
        ]
        
        # Compute weighted context
        if all(isinstance(v, str) for v in values):
            # For text, concatenate with weights as guide
            context_parts = []
            for score in attention_scores:
                if score.weight > 0.1:  # Threshold for inclusion
                    context_parts.append(score.value)
            context = " ".join(context_parts)
        else:
            context = values  # Return all values for non-text
        
        # Identify focused items (top weights)
        sorted_scores = sorted(attention_scores, key=lambda x: x.weight, reverse=True)
        focused_items = [s.key for s in sorted_scores[:3]]
        
        return AttentionOutput(
            context=context,
            attention_weights=attention_scores,
            focused_items=focused_items
        )


class HardAttention:
    """
    Hard attention mechanism - selects discrete subset of inputs.
    
    Uses threshold-based selection rather than weighted combination.
    """
    
    def __init__(self, threshold: float = 0.5, top_k: int = 3):
        self.threshold = threshold
        self.top_k = top_k
    
    def compute_relevance(self, query: str, key: str) -> float:
        """Compute relevance score"""
        query_words = set(query.lower().split())
        key_words = set(key.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & key_words)
        return overlap / len(query_words)
    
    def attend(
        self,
        query: str,
        keys: List[str],
        values: List[Any]
    ) -> AttentionOutput:
        """
        Apply hard attention - select top-k or above threshold.
        
        Args:
            query: Query to focus attention
            keys: List of keys
            values: Corresponding values
        
        Returns:
            AttentionOutput with selected items
        """
        # Compute relevance scores
        scores = [self.compute_relevance(query, key) for key in keys]
        
        # Create attention scores
        attention_scores = [
            AttentionScore(key=k, value=v, score=s, weight=float(s > self.threshold))
            for k, v, s in zip(keys, values, scores)
        ]
        
        # Select top-k items
        sorted_scores = sorted(attention_scores, key=lambda x: x.score, reverse=True)
        selected = sorted_scores[:self.top_k]
        
        # Extract context from selected items
        context = [s.value for s in selected if s.score > 0]
        focused_items = [s.key for s in selected if s.score > 0]
        
        return AttentionOutput(
            context=" ".join(context) if all(isinstance(v, str) for v in context) else context,
            attention_weights=attention_scores,
            focused_items=focused_items
        )


class MultiHeadAttention:
    """
    Multi-head attention - multiple parallel attention mechanisms.
    
    Each head can learn different patterns of relevance.
    """
    
    def __init__(self, num_heads: int = 4, temperature: float = 1.0):
        self.num_heads = num_heads
        self.heads = [SoftAttention(temperature=temperature) for _ in range(num_heads)]
        self.head_aspects = [
            "semantic",
            "temporal",
            "structural",
            "factual"
        ][:num_heads]
    
    def attend(
        self,
        query: str,
        keys: List[str],
        values: List[Any]
    ) -> AttentionOutput:
        """
        Apply multi-head attention and combine results.
        
        Each head focuses on different aspects.
        """
        print(f"\n[Multi-Head Attention] Processing with {self.num_heads} heads")
        
        head_outputs = []
        for i, (head, aspect) in enumerate(zip(self.heads, self.head_aspects)):
            # Modify query to focus on different aspects
            aspect_query = f"{query} {aspect}"
            output = head.attend(aspect_query, keys, values)
            head_outputs.append(output)
            print(f"  Head {i+1} ({aspect}): {output.focused_items[:2]}")
        
        # Aggregate attention from all heads
        aggregated_weights = {}
        for output in head_outputs:
            for score in output.attention_weights:
                if score.key not in aggregated_weights:
                    aggregated_weights[score.key] = []
                aggregated_weights[score.key].append(score.weight)
        
        # Average weights across heads
        final_scores = []
        for key in keys:
            avg_weight = sum(aggregated_weights.get(key, [0])) / self.num_heads
            idx = keys.index(key)
            final_scores.append(
                AttentionScore(
                    key=key,
                    value=values[idx],
                    score=avg_weight,
                    weight=avg_weight
                )
            )
        
        # Select top items
        sorted_scores = sorted(final_scores, key=lambda x: x.weight, reverse=True)
        focused_items = [s.key for s in sorted_scores[:3]]
        context = [s.value for s in sorted_scores if s.weight > 0.1]
        
        return AttentionOutput(
            context=" ".join(context) if all(isinstance(v, str) for v in context) else context,
            attention_weights=final_scores,
            focused_items=focused_items
        )


class CrossAttention:
    """
    Cross-attention between two different sequences/modalities.
    
    Allows attending from one sequence to another (e.g., image to text).
    """
    
    def __init__(self):
        self.soft_attention = SoftAttention()
    
    def attend(
        self,
        queries: List[str],
        keys: List[str],
        values: List[Any]
    ) -> List[AttentionOutput]:
        """
        Apply cross-attention from queries to keys/values.
        
        Args:
            queries: List of queries (e.g., from modality A)
            keys: List of keys (e.g., from modality B)
            values: Corresponding values
        
        Returns:
            List of AttentionOutputs, one per query
        """
        outputs = []
        
        for query in queries:
            output = self.soft_attention.attend(query, keys, values)
            outputs.append(output)
        
        return outputs


class ContextualAttention:
    """
    Contextual attention that adapts based on task context.
    
    Uses task description to modulate attention patterns.
    """
    
    def __init__(self):
        self.soft_attention = SoftAttention()
        self.context_boost = 1.5  # Boost for context-relevant items
    
    def attend(
        self,
        query: str,
        keys: List[str],
        values: List[Any],
        context: str
    ) -> AttentionOutput:
        """
        Apply contextual attention.
        
        Args:
            query: Query string
            keys: Keys to attend over
            values: Corresponding values
            context: Task context for modulation
        
        Returns:
            AttentionOutput with context-aware weighting
        """
        # Incorporate context into query
        contextual_query = f"{context} {query}"
        
        # Get base attention
        output = self.soft_attention.attend(contextual_query, keys, values)
        
        # Boost weights for context-relevant items
        context_words = set(context.lower().split())
        for score in output.attention_weights:
            key_words = set(score.key.lower().split())
            if key_words & context_words:
                score.weight *= self.context_boost
        
        # Renormalize weights
        total_weight = sum(s.weight for s in output.attention_weights)
        if total_weight > 0:
            for score in output.attention_weights:
                score.weight /= total_weight
        
        return output


class AttentionAgent:
    """
    Agent that uses attention mechanisms to process information.
    
    Can apply different attention strategies based on task requirements.
    """
    
    def __init__(self):
        self.soft_attention = SoftAttention(temperature=0.5)
        self.hard_attention = HardAttention(threshold=0.4, top_k=3)
        self.multi_head = MultiHeadAttention(num_heads=4)
        self.contextual = ContextualAttention()
    
    def process_with_attention(
        self,
        query: str,
        documents: List[Dict[str, str]],
        strategy: str = "soft",
        context: Optional[str] = None
    ) -> AttentionOutput:
        """
        Process documents with specified attention strategy.
        
        Args:
            query: User query
            documents: List of documents with 'title' and 'content'
            strategy: Attention strategy ('soft', 'hard', 'multi_head', 'contextual')
            context: Optional task context
        
        Returns:
            AttentionOutput with focused information
        """
        print(f"\n{'='*70}")
        print(f"Attention Agent Processing")
        print(f"Query: {query}")
        print(f"Strategy: {strategy}")
        print(f"Documents: {len(documents)}")
        print(f"{'='*70}")
        
        # Extract keys and values from documents
        keys = [doc['title'] for doc in documents]
        values = [doc['content'] for doc in documents]
        
        # Apply appropriate attention mechanism
        if strategy == "soft":
            output = self.soft_attention.attend(query, keys, values)
        elif strategy == "hard":
            output = self.hard_attention.attend(query, keys, values)
        elif strategy == "multi_head":
            output = self.multi_head.attend(query, keys, values)
        elif strategy == "contextual" and context:
            output = self.contextual.attend(query, keys, values, context)
        else:
            output = self.soft_attention.attend(query, keys, values)
        
        # Display results
        print(f"\nFocused Items:")
        for item in output.focused_items:
            print(f"  - {item}")
        
        print(f"\nTop Attention Weights:")
        sorted_weights = sorted(output.attention_weights, 
                              key=lambda x: x.weight, reverse=True)
        for score in sorted_weights[:5]:
            print(f"  {score.key}: {score.weight:.3f}")
        
        return output
    
    def answer_with_attention(
        self,
        question: str,
        documents: List[Dict[str, str]],
        strategy: str = "multi_head"
    ) -> str:
        """
        Answer question using attention to focus on relevant documents.
        
        Args:
            question: Question to answer
            documents: Knowledge base documents
            strategy: Attention strategy to use
        
        Returns:
            Generated answer
        """
        # Apply attention to focus on relevant documents
        attention_output = self.process_with_attention(
            query=question,
            documents=documents,
            strategy=strategy
        )
        
        # Generate answer from focused context
        answer = f"Based on the documents, particularly focusing on " \
                f"{', '.join(attention_output.focused_items[:2])}, "
        
        # In real implementation, would use LLM to generate answer
        answer += f"the answer involves: {attention_output.context[:200]}..."
        
        return answer


def demonstrate_attention_mechanisms():
    """Demonstrate different attention mechanisms"""
    
    print("Attention Mechanism Patterns Demonstration")
    print("=" * 70)
    
    # Sample documents
    documents = [
        {
            "title": "Python Programming Basics",
            "content": "Python is a high-level programming language known for readability."
        },
        {
            "title": "Machine Learning with Python",
            "content": "Python provides excellent libraries for machine learning like scikit-learn."
        },
        {
            "title": "Web Development with Flask",
            "content": "Flask is a lightweight web framework for Python applications."
        },
        {
            "title": "Data Science Tools",
            "content": "Python offers pandas and numpy for data science and analysis."
        },
        {
            "title": "Neural Networks Introduction",
            "content": "Neural networks are the foundation of deep learning and AI."
        },
        {
            "title": "Natural Language Processing",
            "content": "NLP enables machines to understand and generate human language."
        }
    ]
    
    # Create agent
    agent = AttentionAgent()
    
    # Test different queries and strategies
    test_cases = [
        ("How do I learn machine learning?", "soft"),
        ("What tools for data science?", "hard"),
        ("Explain web development", "multi_head"),
    ]
    
    for query, strategy in test_cases:
        print(f"\n\n{'='*70}")
        output = agent.process_with_attention(query, documents, strategy)
        print(f"\nContext Summary: {output.context[:150]}...")
    
    # Demonstrate question answering
    print(f"\n\n{'='*70}")
    print("Question Answering with Attention")
    print("=" * 70)
    
    question = "What Python libraries are useful for AI?"
    answer = agent.answer_with_attention(question, documents, strategy="multi_head")
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    demonstrate_attention_mechanisms()
    
    print("\n\n" + "="*70)
    print("Attention Mechanism Patterns Summary")
    print("="*70)
    print("""
Key Benefits:
1. Efficient processing of long contexts
2. Dynamic focus on relevant information
3. Interpretable attention weights
4. Handles variable-length inputs
5. Multi-modal information integration

Attention Types:
- Soft Attention: Weighted combination of all inputs
- Hard Attention: Discrete selection of inputs
- Multi-Head: Multiple parallel attention patterns
- Cross-Attention: Between different sequences/modalities
- Contextual: Task-aware attention modulation

Best Practices:
- Choose appropriate attention type for task
- Use multi-head for complex patterns
- Monitor attention distributions
- Combine with other mechanisms
- Regularize attention weights
- Visualize attention for debugging

Use Cases:
- Long document question answering
- Multi-document summarization
- Information retrieval and ranking
- Multi-modal data fusion
- Context-aware dialogue systems
- Selective memory retrieval

Common Patterns:
1. Query-Key-Value: Standard attention computation
2. Self-Attention: Attend within same sequence
3. Cross-Attention: Attend between sequences
4. Scaled Dot-Product: Attention with scaling
5. Additive Attention: Using neural network
""")
