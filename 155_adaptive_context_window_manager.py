"""
Adaptive Context Window Manager Pattern

Intelligently manages context window to fit within token limits.
Dynamically prioritizes and compresses information based on relevance.

Use Cases:
- Long document processing
- Multi-turn conversations
- Large codebase analysis
- Research paper summarization

Advantages:
- Efficient token usage
- Maintains important context
- Handles token limits gracefully
- Adaptive prioritization
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class ContentType(Enum):
    """Types of content in context"""
    SYSTEM_PROMPT = "system_prompt"
    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    BACKGROUND_INFO = "background_info"
    EXAMPLE = "example"
    INSTRUCTION = "instruction"
    REFERENCE = "reference"


class PriorityLevel(Enum):
    """Priority levels for content"""
    CRITICAL = "critical"  # Must keep
    HIGH = "high"  # Keep if possible
    MEDIUM = "medium"  # Can compress
    LOW = "low"  # Can drop


@dataclass
class ContextChunk:
    """Chunk of context with metadata"""
    chunk_id: str
    content: str
    content_type: ContentType
    priority: PriorityLevel
    token_count: int
    timestamp: datetime
    relevance_score: float = 0.5
    compressed: bool = False
    original_length: Optional[int] = None


@dataclass
class CompressionStrategy:
    """Strategy for compressing content"""
    name: str
    compression_ratio: float  # Target compression (0-1)
    applicable_types: List[ContentType]
    min_length: int  # Minimum length to compress


@dataclass
class WindowState:
    """Current state of context window"""
    total_tokens: int
    max_tokens: int
    chunks: List[ContextChunk]
    utilization: float
    compression_applied: bool


class TokenEstimator:
    """Estimates token count for text"""
    
    def __init__(self):
        # Approximate: 1 token ≈ 4 characters
        self.chars_per_token = 4
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation based on characters
        return len(text) // self.chars_per_token
    
    def estimate_tokens_batch(self, texts: List[str]) -> List[int]:
        """Estimate tokens for multiple texts"""
        return [self.estimate_tokens(text) for text in texts]


class ContentCompressor:
    """Compresses content while preserving meaning"""
    
    def __init__(self):
        self.strategies = {
            "extractive": self._extractive_compression,
            "abbreviation": self._abbreviation_compression,
            "summarization": self._summarization_compression
        }
    
    def compress(self,
                content: str,
                target_ratio: float,
                content_type: ContentType) -> str:
        """
        Compress content to target ratio.
        
        Args:
            content: Content to compress
            target_ratio: Target compression ratio (0-1)
            content_type: Type of content
            
        Returns:
            Compressed content
        """
        if target_ratio >= 1.0:
            return content
        
        # Choose strategy based on content type
        if content_type in [ContentType.BACKGROUND_INFO, ContentType.REFERENCE]:
            return self._extractive_compression(content, target_ratio)
        elif content_type == ContentType.EXAMPLE:
            return self._abbreviation_compression(content, target_ratio)
        else:
            return self._summarization_compression(content, target_ratio)
    
    def _extractive_compression(self, content: str, target_ratio: float) -> str:
        """Extract most important sentences"""
        sentences = content.split('. ')
        target_count = max(1, int(len(sentences) * target_ratio))
        
        # Simple importance: length and position
        scored_sentences = []
        for i, sent in enumerate(sentences):
            # Prefer longer sentences and earlier sentences
            score = len(sent) * (1.0 - i / len(sentences) * 0.3)
            scored_sentences.append((sent, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s for s, _ in scored_sentences[:target_count]]
        
        return '. '.join(top_sentences)
    
    def _abbreviation_compression(self, content: str, target_ratio: float) -> str:
        """Compress using abbreviations"""
        # Simple abbreviation of common words
        abbreviations = {
            'information': 'info',
            'development': 'dev',
            'application': 'app',
            'configuration': 'config',
            'documentation': 'docs',
            'repository': 'repo',
            'implementation': 'impl'
        }
        
        compressed = content
        for full, abbr in abbreviations.items():
            compressed = compressed.replace(full, abbr)
        
        # If still too long, truncate
        target_len = int(len(content) * target_ratio)
        if len(compressed) > target_len:
            compressed = compressed[:target_len] + "..."
        
        return compressed
    
    def _summarization_compression(self, content: str, target_ratio: float) -> str:
        """Summarize content"""
        # Simple summarization: first and last parts
        target_len = int(len(content) * target_ratio)
        
        if len(content) <= target_len:
            return content
        
        # Keep first 70% and last 30% of target
        first_part_len = int(target_len * 0.7)
        last_part_len = int(target_len * 0.3)
        
        first_part = content[:first_part_len]
        last_part = content[-last_part_len:]
        
        return "{} [...] {}".format(first_part, last_part)


class RelevanceScorer:
    """Scores content relevance to current query"""
    
    def score_relevance(self,
                       chunk: ContextChunk,
                       query: str,
                       recent_context: List[str]) -> float:
        """
        Score relevance of chunk to query.
        
        Args:
            chunk: Context chunk to score
            query: Current query
            recent_context: Recent conversation context
            
        Returns:
            Relevance score (0-1)
        """
        score = 0.0
        
        # Content type importance
        type_scores = {
            ContentType.SYSTEM_PROMPT: 0.9,
            ContentType.INSTRUCTION: 0.8,
            ContentType.USER_MESSAGE: 0.7,
            ContentType.ASSISTANT_MESSAGE: 0.6,
            ContentType.EXAMPLE: 0.5,
            ContentType.BACKGROUND_INFO: 0.4,
            ContentType.REFERENCE: 0.3
        }
        score += type_scores.get(chunk.content_type, 0.5) * 0.3
        
        # Keyword overlap with query
        query_words = set(query.lower().split())
        chunk_words = set(chunk.content.lower().split())
        if query_words:
            overlap = len(query_words.intersection(chunk_words))
            score += (overlap / len(query_words)) * 0.4
        
        # Recency (more recent = more relevant)
        # Simplified: assume more recent chunks are more relevant
        score += 0.3  # Base recency score
        
        return min(score, 1.0)


class AdaptiveContextWindowManager:
    """
    Manages context window adaptively within token limits.
    Prioritizes, compresses, and drops content as needed.
    """
    
    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.token_estimator = TokenEstimator()
        self.compressor = ContentCompressor()
        self.relevance_scorer = RelevanceScorer()
        
        self.chunks: List[ContextChunk] = []
        self.chunk_counter = 0
        self.compression_history: List[Dict[str, Any]] = []
        
        # Reserve tokens for response
        self.reserved_tokens = 512
        self.available_tokens = max_tokens - self.reserved_tokens
    
    def add_content(self,
                   content: str,
                   content_type: ContentType,
                   priority: PriorityLevel = PriorityLevel.MEDIUM) -> str:
        """
        Add content to context window.
        
        Args:
            content: Content to add
            content_type: Type of content
            priority: Priority level
            
        Returns:
            Chunk ID
        """
        # Estimate tokens
        token_count = self.token_estimator.estimate_tokens(content)
        
        # Create chunk
        chunk = ContextChunk(
            chunk_id="chunk_{}".format(self.chunk_counter),
            content=content,
            content_type=content_type,
            priority=priority,
            token_count=token_count,
            timestamp=datetime.now()
        )
        
        self.chunks.append(chunk)
        self.chunk_counter += 1
        
        # Check if we need to manage window
        total_tokens = sum(c.token_count for c in self.chunks)
        if total_tokens > self.available_tokens:
            self._manage_window()
        
        return chunk.chunk_id
    
    def build_context(self, query: Optional[str] = None) -> str:
        """
        Build context string from managed chunks.
        
        Args:
            query: Optional current query for relevance scoring
            
        Returns:
            Context string ready for LLM
        """
        # Update relevance scores if query provided
        if query:
            recent_context = [c.content for c in self.chunks[-3:]]
            for chunk in self.chunks:
                chunk.relevance_score = self.relevance_scorer.score_relevance(
                    chunk, query, recent_context
                )
        
        # Sort chunks by priority and relevance
        sorted_chunks = sorted(
            self.chunks,
            key=lambda c: (
                self._priority_to_score(c.priority),
                c.relevance_score,
                c.timestamp
            ),
            reverse=True
        )
        
        # Build context within token limit
        context_parts = []
        total_tokens = 0
        
        for chunk in sorted_chunks:
            if total_tokens + chunk.token_count <= self.available_tokens:
                context_parts.append(chunk.content)
                total_tokens += chunk.token_count
            elif chunk.priority == PriorityLevel.CRITICAL:
                # Must include critical content, compress if needed
                compressed = self.compressor.compress(
                    chunk.content,
                    target_ratio=0.5,
                    content_type=chunk.content_type
                )
                compressed_tokens = self.token_estimator.estimate_tokens(compressed)
                
                if total_tokens + compressed_tokens <= self.available_tokens:
                    context_parts.append(compressed)
                    total_tokens += compressed_tokens
        
        return "\n\n".join(context_parts)
    
    def optimize_for_query(self, query: str) -> WindowState:
        """
        Optimize context window for specific query.
        
        Args:
            query: Query to optimize for
            
        Returns:
            Window state after optimization
        """
        # Score all chunks for relevance
        recent_context = [c.content for c in self.chunks[-3:]]
        for chunk in self.chunks:
            chunk.relevance_score = self.relevance_scorer.score_relevance(
                chunk, query, recent_context
            )
        
        # Manage window with relevance scores
        self._manage_window()
        
        # Get current state
        return self.get_window_state()
    
    def compress_chunk(self,
                      chunk_id: str,
                      target_ratio: float = 0.5) -> bool:
        """
        Compress a specific chunk.
        
        Args:
            chunk_id: Chunk to compress
            target_ratio: Target compression ratio
            
        Returns:
            Success status
        """
        chunk = self._find_chunk(chunk_id)
        if not chunk or chunk.compressed:
            return False
        
        # Compress content
        compressed_content = self.compressor.compress(
            chunk.content,
            target_ratio,
            chunk.content_type
        )
        
        # Update chunk
        chunk.original_length = len(chunk.content)
        chunk.content = compressed_content
        chunk.token_count = self.token_estimator.estimate_tokens(compressed_content)
        chunk.compressed = True
        
        # Record compression
        self.compression_history.append({
            "chunk_id": chunk_id,
            "original_tokens": chunk.original_length // 4,
            "compressed_tokens": chunk.token_count,
            "ratio": target_ratio,
            "timestamp": datetime.now()
        })
        
        return True
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk from context"""
        chunk = self._find_chunk(chunk_id)
        if chunk:
            self.chunks.remove(chunk)
            return True
        return False
    
    def get_window_state(self) -> WindowState:
        """Get current window state"""
        total_tokens = sum(c.token_count for c in self.chunks)
        
        return WindowState(
            total_tokens=total_tokens,
            max_tokens=self.available_tokens,
            chunks=list(self.chunks),
            utilization=total_tokens / self.available_tokens if self.available_tokens > 0 else 0,
            compression_applied=any(c.compressed for c in self.chunks)
        )
    
    def clear_low_priority(self) -> int:
        """Clear low priority chunks"""
        removed = 0
        self.chunks = [
            c for c in self.chunks
            if c.priority != PriorityLevel.LOW
        ]
        return removed
    
    def _manage_window(self) -> None:
        """Manage window to fit within token limit"""
        total_tokens = sum(c.token_count for c in self.chunks)
        
        if total_tokens <= self.available_tokens:
            return
        
        # Strategy 1: Remove low priority chunks
        low_priority = [c for c in self.chunks if c.priority == PriorityLevel.LOW]
        for chunk in low_priority:
            self.chunks.remove(chunk)
            total_tokens -= chunk.token_count
            if total_tokens <= self.available_tokens:
                return
        
        # Strategy 2: Compress medium priority chunks
        medium_priority = [
            c for c in self.chunks
            if c.priority == PriorityLevel.MEDIUM and not c.compressed
        ]
        
        for chunk in medium_priority:
            if self.compress_chunk(chunk.chunk_id, target_ratio=0.6):
                total_tokens = sum(c.token_count for c in self.chunks)
                if total_tokens <= self.available_tokens:
                    return
        
        # Strategy 3: Compress high priority chunks
        high_priority = [
            c for c in self.chunks
            if c.priority == PriorityLevel.HIGH and not c.compressed
        ]
        
        for chunk in high_priority:
            if self.compress_chunk(chunk.chunk_id, target_ratio=0.7):
                total_tokens = sum(c.token_count for c in self.chunks)
                if total_tokens <= self.available_tokens:
                    return
        
        # Strategy 4: Remove oldest medium priority chunks
        medium_chunks = [c for c in self.chunks if c.priority == PriorityLevel.MEDIUM]
        medium_chunks.sort(key=lambda c: c.timestamp)
        
        for chunk in medium_chunks:
            self.chunks.remove(chunk)
            total_tokens -= chunk.token_count
            if total_tokens <= self.available_tokens:
                return
    
    def _find_chunk(self, chunk_id: str) -> Optional[ContextChunk]:
        """Find chunk by ID"""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None
    
    def _priority_to_score(self, priority: PriorityLevel) -> float:
        """Convert priority to numeric score"""
        priority_scores = {
            PriorityLevel.CRITICAL: 1.0,
            PriorityLevel.HIGH: 0.7,
            PriorityLevel.MEDIUM: 0.4,
            PriorityLevel.LOW: 0.1
        }
        return priority_scores.get(priority, 0.5)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get context window statistics"""
        total_tokens = sum(c.token_count for c in self.chunks)
        
        by_type = {}
        by_priority = {}
        
        for chunk in self.chunks:
            content_type = chunk.content_type.value
            priority = chunk.priority.value
            
            by_type[content_type] = by_type.get(content_type, 0) + chunk.token_count
            by_priority[priority] = by_priority.get(priority, 0) + chunk.token_count
        
        return {
            "total_chunks": len(self.chunks),
            "total_tokens": total_tokens,
            "available_tokens": self.available_tokens,
            "utilization": round(total_tokens / self.available_tokens, 2),
            "compressed_chunks": sum(1 for c in self.chunks if c.compressed),
            "tokens_by_type": by_type,
            "tokens_by_priority": by_priority,
            "compression_operations": len(self.compression_history)
        }


def demonstrate_adaptive_context_window():
    """Demonstrate adaptive context window manager"""
    print("=" * 70)
    print("Adaptive Context Window Manager Demonstration")
    print("=" * 70)
    
    # Create manager with 2000 token limit
    manager = AdaptiveContextWindowManager(max_tokens=2000)
    
    # Example 1: Add various content types
    print("\n1. Adding Content to Context Window:")
    
    system_prompt = "You are a helpful AI assistant specializing in software development."
    manager.add_content(system_prompt, ContentType.SYSTEM_PROMPT, PriorityLevel.CRITICAL)
    print("Added system prompt")
    
    instruction = "Help the user with their coding questions. Provide clear, concise explanations with examples."
    manager.add_content(instruction, ContentType.INSTRUCTION, PriorityLevel.HIGH)
    print("Added instruction")
    
    background = """
    The user is working on a Python project using FastAPI for building REST APIs.
    They have experience with Django but are new to FastAPI. The project involves
    creating microservices for a large-scale application with authentication,
    database integration, and real-time features using WebSockets.
    """
    manager.add_content(background, ContentType.BACKGROUND_INFO, PriorityLevel.MEDIUM)
    print("Added background info")
    
    # Add conversation history
    for i in range(5):
        user_msg = "User message {} about FastAPI and microservices architecture".format(i + 1)
        assistant_msg = "Assistant response {} explaining FastAPI concepts and best practices".format(i + 1)
        
        manager.add_content(user_msg, ContentType.USER_MESSAGE, PriorityLevel.MEDIUM)
        manager.add_content(assistant_msg, ContentType.ASSISTANT_MESSAGE, PriorityLevel.MEDIUM)
    
    print("\nAdded 5 conversation turns")
    
    # Check window state
    state = manager.get_window_state()
    print("\nWindow State:")
    print("  Total chunks: {}".format(len(state.chunks)))
    print("  Total tokens: {}".format(state.total_tokens))
    print("  Utilization: {:.1%}".format(state.utilization))
    print("  Compression applied: {}".format(state.compression_applied))
    
    # Example 2: Optimize for specific query
    print("\n2. Optimizing for Specific Query:")
    query = "How do I implement WebSocket authentication in FastAPI?"
    
    optimized_state = manager.optimize_for_query(query)
    print("Query: {}".format(query))
    print("Optimized state:")
    print("  Total tokens: {}".format(optimized_state.total_tokens))
    print("  Utilization: {:.1%}".format(optimized_state.utilization))
    print("  Chunks with high relevance: {}".format(
        sum(1 for c in optimized_state.chunks if c.relevance_score > 0.7)
    ))
    
    # Example 3: Build context
    print("\n3. Building Context String:")
    context = manager.build_context(query)
    print("Context length: {} characters".format(len(context)))
    print("Estimated tokens: {}".format(
        manager.token_estimator.estimate_tokens(context)
    ))
    print("\nContext preview:")
    print("-" * 60)
    print(context[:300])
    print("...")
    
    # Example 4: Add large content that exceeds limit
    print("\n4. Adding Large Content (Triggers Management):")
    large_content = """
    Detailed technical documentation about FastAPI WebSockets...
    """ * 100  # Large content
    
    manager.add_content(large_content, ContentType.REFERENCE, PriorityLevel.LOW)
    
    state = manager.get_window_state()
    print("After adding large content:")
    print("  Total tokens: {}".format(state.total_tokens))
    print("  Utilization: {:.1%}".format(state.utilization))
    print("  Compression applied: {}".format(state.compression_applied))
    
    # Example 5: Statistics
    print("\n5. Context Window Statistics:")
    stats = manager.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Example 6: Manual compression
    print("\n6. Manual Chunk Compression:")
    if manager.chunks:
        chunk_to_compress = manager.chunks[0]
        print("Original tokens: {}".format(chunk_to_compress.token_count))
        
        success = manager.compress_chunk(chunk_to_compress.chunk_id, target_ratio=0.5)
        print("Compression success: {}".format(success))
        print("Compressed tokens: {}".format(chunk_to_compress.token_count))
        print("Compression ratio: {:.1%}".format(
            chunk_to_compress.token_count / (chunk_to_compress.original_length // 4)
            if chunk_to_compress.original_length else 0
        ))


if __name__ == "__main__":
    demonstrate_adaptive_context_window()
