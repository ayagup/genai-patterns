"""
Token Budget Management Pattern

Manages token usage within limits through summarization, truncation, and compression
strategies. Essential for cost control and respecting API constraints in LLM applications.

Use Cases:
- Long conversations requiring context management
- Cost-sensitive production applications
- API rate limit compliance
- Document processing with size constraints
- Multi-turn dialogues with history

Benefits:
- Cost control: Stay within budget limits
- Performance: Faster processing with smaller contexts
- Reliability: Avoid token limit errors
- Scalability: Handle long conversations gracefully
- Flexibility: Adaptive strategies based on importance
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re


class CompressionStrategy(Enum):
    """Strategies for compressing content"""
    TRUNCATE_OLD = "truncate_old"  # Remove oldest messages
    TRUNCATE_MIDDLE = "truncate_middle"  # Keep start and end
    SUMMARIZE = "summarize"  # Summarize old content
    SELECTIVE = "selective"  # Keep important, drop unimportant
    HIERARCHICAL = "hierarchical"  # Multi-level summaries


class MessagePriority(Enum):
    """Priority levels for messages"""
    CRITICAL = 3  # System prompts, key instructions
    HIGH = 2  # Recent messages, important context
    MEDIUM = 1  # Regular conversation
    LOW = 0  # Optional, can be dropped first


@dataclass
class Message:
    """A message in the conversation"""
    role: str  # system, user, assistant
    content: str
    priority: MessagePriority = MessagePriority.MEDIUM
    token_count: int = 0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def estimate_tokens(self) -> int:
        """
        Estimate token count for message.
        In production: Use tiktoken or similar tokenizer.
        """
        # Simple approximation: ~4 chars per token
        if self.token_count == 0:
            self.token_count = len(self.content) // 4 + 1
        return self.token_count
    
    def __str__(self) -> str:
        return f"[{self.role}] {self.content[:50]}... ({self.token_count} tokens)"


@dataclass
class TokenBudget:
    """Token budget configuration"""
    max_total_tokens: int = 4096  # Model's max context
    max_output_tokens: int = 1000  # Reserved for response
    buffer_tokens: int = 100  # Safety buffer
    
    def get_available_tokens(self) -> int:
        """Get tokens available for input"""
        return self.max_total_tokens - self.max_output_tokens - self.buffer_tokens
    
    def get_usage_percentage(self, used_tokens: int) -> float:
        """Get percentage of budget used"""
        available = self.get_available_tokens()
        return (used_tokens / available) * 100 if available > 0 else 100.0


@dataclass
class ConversationSnapshot:
    """Snapshot of conversation state"""
    messages: List[Message]
    total_tokens: int
    timestamp: float
    compression_applied: bool = False
    

class TokenCounter:
    """Utilities for counting tokens"""
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Count tokens in text.
        In production: Use proper tokenizer (tiktoken for OpenAI).
        """
        # Simple approximation
        return len(text) // 4 + 1
    
    @staticmethod
    def count_message_tokens(message: Message) -> int:
        """Count tokens in a message including role"""
        # Add overhead for message formatting
        content_tokens = TokenCounter.count_tokens(message.content)
        role_overhead = 4  # Approximate overhead for role formatting
        return content_tokens + role_overhead
    
    @staticmethod
    def count_conversation_tokens(messages: List[Message]) -> int:
        """Count total tokens in conversation"""
        return sum(TokenCounter.count_message_tokens(msg) for msg in messages)


class TextCompressor:
    """Compresses text to reduce token usage"""
    
    @staticmethod
    def summarize(text: str, max_tokens: int) -> str:
        """
        Summarize text to fit within token budget.
        In production: Use LLM to generate proper summary.
        """
        current_tokens = TokenCounter.count_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
        
        # Simple compression: take first and last portions
        ratio = max_tokens / current_tokens
        chars_to_keep = int(len(text) * ratio)
        
        first_half = chars_to_keep // 2
        second_half = chars_to_keep - first_half
        
        if first_half > 0 and second_half > 0:
            compressed = (
                text[:first_half] +
                f"\n[...{current_tokens - max_tokens} tokens summarized...]\n" +
                text[-second_half:]
            )
            return compressed
        
        return text[:chars_to_keep]
    
    @staticmethod
    def extract_key_points(text: str, max_points: int = 5) -> str:
        """
        Extract key points from text.
        In production: Use LLM for better extraction.
        """
        # Simple extraction: find sentences with key indicators
        sentences = re.split(r'[.!?]+', text)
        
        key_indicators = [
            'important', 'key', 'must', 'critical', 'essential',
            'remember', 'note', 'significant', 'main', 'primary'
        ]
        
        key_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in key_indicators):
                key_sentences.append(sentence)
                if len(key_sentences) >= max_points:
                    break
        
        if not key_sentences:
            # Fall back to first few sentences
            key_sentences = [s.strip() for s in sentences[:max_points] if s.strip()]
        
        return ". ".join(key_sentences) + "."
    
    @staticmethod
    def remove_redundancy(text: str) -> str:
        """Remove redundant content"""
        # Remove duplicate lines
        lines = text.split('\n')
        unique_lines = []
        seen = set()
        
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower and line_lower not in seen:
                unique_lines.append(line)
                seen.add(line_lower)
        
        return '\n'.join(unique_lines)


class TokenBudgetManager:
    """
    Token Budget Management Agent
    
    Manages token usage within budget constraints using various
    compression and optimization strategies.
    """
    
    def __init__(
        self,
        budget: TokenBudget,
        strategy: CompressionStrategy = CompressionStrategy.SELECTIVE
    ):
        self.budget = budget
        self.strategy = strategy
        self.conversation_history: List[Message] = []
        self.snapshots: List[ConversationSnapshot] = []
        self.total_tokens_used = 0
        self.compressions_applied = 0
    
    def add_message(self, message: Message) -> bool:
        """
        Add message to conversation, managing budget.
        Returns True if added successfully.
        """
        message.estimate_tokens()
        
        print(f"\n[Token Budget] Adding message: {message.role}")
        print(f"  Tokens: {message.token_count}")
        
        # Add to history
        self.conversation_history.append(message)
        self.total_tokens_used += message.token_count
        
        # Check if budget exceeded
        available = self.budget.get_available_tokens()
        current_usage = TokenCounter.count_conversation_tokens(self.conversation_history)
        
        print(f"  Current usage: {current_usage}/{available} tokens")
        print(f"  Budget used: {self.budget.get_usage_percentage(current_usage):.1f}%")
        
        if current_usage > available:
            print(f"  ⚠ Budget exceeded! Applying compression...")
            self._compress_history()
            return True
        
        return True
    
    def _compress_history(self) -> None:
        """Compress conversation history based on strategy"""
        self.compressions_applied += 1
        
        print(f"\n[Compression] Strategy: {self.strategy.value}")
        
        if self.strategy == CompressionStrategy.TRUNCATE_OLD:
            self._truncate_old()
        elif self.strategy == CompressionStrategy.TRUNCATE_MIDDLE:
            self._truncate_middle()
        elif self.strategy == CompressionStrategy.SUMMARIZE:
            self._summarize_old()
        elif self.strategy == CompressionStrategy.SELECTIVE:
            self._selective_compression()
        elif self.strategy == CompressionStrategy.HIERARCHICAL:
            self._hierarchical_compression()
    
    def _truncate_old(self) -> None:
        """Remove oldest messages until within budget"""
        available = self.budget.get_available_tokens()
        
        # Keep critical messages
        critical_messages = [
            msg for msg in self.conversation_history
            if msg.priority == MessagePriority.CRITICAL
        ]
        
        # Try to fit recent messages
        other_messages = [
            msg for msg in self.conversation_history
            if msg.priority != MessagePriority.CRITICAL
        ]
        
        # Start from most recent
        other_messages.reverse()
        
        kept_messages = critical_messages.copy()
        current_tokens = sum(msg.token_count for msg in kept_messages)
        
        for msg in other_messages:
            if current_tokens + msg.token_count <= available:
                kept_messages.append(msg)
                current_tokens += msg.token_count
            else:
                break
        
        # Restore chronological order
        kept_messages.sort(key=lambda m: m.timestamp)
        
        removed = len(self.conversation_history) - len(kept_messages)
        self.conversation_history = kept_messages
        
        print(f"  Removed {removed} oldest messages")
        print(f"  Remaining: {len(self.conversation_history)} messages")
    
    def _truncate_middle(self) -> None:
        """Keep first and last messages, remove middle"""
        if len(self.conversation_history) <= 4:
            return
        
        available = self.budget.get_available_tokens()
        
        # Always keep first (system) and last few messages
        keep_first = 2
        keep_last = 3
        
        first_messages = self.conversation_history[:keep_first]
        last_messages = self.conversation_history[-keep_last:]
        
        # Check if this fits
        kept_tokens = sum(
            msg.token_count
            for msg in first_messages + last_messages
        )
        
        if kept_tokens <= available:
            removed = len(self.conversation_history) - (keep_first + keep_last)
            self.conversation_history = first_messages + last_messages
            print(f"  Removed {removed} middle messages")
        else:
            # Fall back to truncate old
            self._truncate_old()
    
    def _summarize_old(self) -> None:
        """Summarize old messages"""
        if len(self.conversation_history) <= 3:
            return
        
        # Keep last 3 messages as-is
        recent_messages = self.conversation_history[-3:]
        old_messages = self.conversation_history[:-3]
        
        # Create summary of old messages
        old_content = "\n".join(
            f"{msg.role}: {msg.content}"
            for msg in old_messages
        )
        
        summary_target_tokens = self.budget.get_available_tokens() // 4
        summary = TextCompressor.summarize(old_content, summary_target_tokens)
        
        summary_message = Message(
            role="system",
            content=f"Previous conversation summary:\n{summary}",
            priority=MessagePriority.HIGH
        )
        summary_message.estimate_tokens()
        
        self.conversation_history = [summary_message] + recent_messages
        
        print(f"  Summarized {len(old_messages)} messages")
        print(f"  Summary tokens: {summary_message.token_count}")
    
    def _selective_compression(self) -> None:
        """Keep important messages, compress or drop others"""
        available = self.budget.get_available_tokens()
        
        # Sort by priority (descending) then timestamp (descending for recency)
        sorted_messages = sorted(
            self.conversation_history,
            key=lambda m: (m.priority.value, m.timestamp),
            reverse=True
        )
        
        kept_messages = []
        current_tokens = 0
        
        for msg in sorted_messages:
            if current_tokens + msg.token_count <= available:
                kept_messages.append(msg)
                current_tokens += msg.token_count
            elif msg.priority.value >= MessagePriority.HIGH.value:
                # Compress important messages rather than dropping
                compressed_content = TextCompressor.summarize(
                    msg.content,
                    msg.token_count // 2
                )
                compressed_msg = Message(
                    role=msg.role,
                    content=compressed_content,
                    priority=msg.priority,
                    timestamp=msg.timestamp
                )
                compressed_msg.estimate_tokens()
                
                if current_tokens + compressed_msg.token_count <= available:
                    kept_messages.append(compressed_msg)
                    current_tokens += compressed_msg.token_count
        
        # Restore chronological order
        kept_messages.sort(key=lambda m: m.timestamp)
        
        removed = len(self.conversation_history) - len(kept_messages)
        self.conversation_history = kept_messages
        
        print(f"  Kept {len(kept_messages)} messages (removed/compressed {removed})")
    
    def _hierarchical_compression(self) -> None:
        """Create multi-level summaries"""
        if len(self.conversation_history) <= 5:
            return
        
        # Group messages into chunks
        chunk_size = 5
        chunks = [
            self.conversation_history[i:i + chunk_size]
            for i in range(0, len(self.conversation_history), chunk_size)
        ]
        
        summaries = []
        
        for i, chunk in enumerate(chunks[:-1]):  # Don't summarize last chunk
            chunk_content = "\n".join(
                f"{msg.role}: {msg.content}"
                for msg in chunk
            )
            
            summary = TextCompressor.extract_key_points(chunk_content)
            
            summary_msg = Message(
                role="system",
                content=f"Summary (messages {i*chunk_size+1}-{(i+1)*chunk_size}): {summary}",
                priority=MessagePriority.HIGH
            )
            summary_msg.estimate_tokens()
            summaries.append(summary_msg)
        
        # Keep summaries + last chunk
        self.conversation_history = summaries + chunks[-1]
        
        print(f"  Created {len(summaries)} hierarchical summaries")
    
    def get_current_context(self) -> List[Message]:
        """Get current conversation context"""
        return self.conversation_history.copy()
    
    def get_token_stats(self) -> Dict[str, Any]:
        """Get token usage statistics"""
        current_usage = TokenCounter.count_conversation_tokens(self.conversation_history)
        available = self.budget.get_available_tokens()
        
        return {
            "current_tokens": current_usage,
            "available_tokens": available,
            "max_tokens": self.budget.max_total_tokens,
            "usage_percentage": self.budget.get_usage_percentage(current_usage),
            "messages_count": len(self.conversation_history),
            "compressions_applied": self.compressions_applied,
            "total_tokens_processed": self.total_tokens_used
        }
    
    def create_snapshot(self) -> ConversationSnapshot:
        """Create snapshot of current state"""
        snapshot = ConversationSnapshot(
            messages=self.get_current_context(),
            total_tokens=TokenCounter.count_conversation_tokens(self.conversation_history),
            timestamp=len(self.snapshots),
            compression_applied=self.compressions_applied > 0
        )
        self.snapshots.append(snapshot)
        return snapshot


def demonstrate_token_budget_management():
    """
    Demonstrate Token Budget Management pattern
    """
    print("=" * 70)
    print("TOKEN BUDGET MANAGEMENT PATTERN DEMONSTRATION")
    print("=" * 70)
    
    # Example 1: Basic budget management
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Token Budget Management")
    print("=" * 70)
    
    # Create budget and manager
    budget = TokenBudget(
        max_total_tokens=1000,
        max_output_tokens=200,
        buffer_tokens=50
    )
    
    manager = TokenBudgetManager(budget, CompressionStrategy.TRUNCATE_OLD)
    
    print(f"\nBudget Configuration:")
    print(f"  Max total tokens: {budget.max_total_tokens}")
    print(f"  Available for input: {budget.get_available_tokens()}")
    
    # Add system message
    manager.add_message(Message(
        role="system",
        content="You are a helpful assistant.",
        priority=MessagePriority.CRITICAL
    ))
    
    # Simulate conversation
    user_messages = [
        "Hello, how are you?",
        "Can you explain quantum computing?",
        "What are the main applications?",
        "How does it differ from classical computing?",
        "What companies are leading in this field?",
        "What about quantum cryptography?",
        "When will quantum computers be mainstream?",
        "What are the current limitations?",
    ]
    
    for i, content in enumerate(user_messages):
        # User message
        manager.add_message(Message(
            role="user",
            content=content,
            timestamp=i * 2
        ))
        
        # Simulate assistant response
        response = f"Response to: {content}. " * 20  # Long response
        manager.add_message(Message(
            role="assistant",
            content=response,
            timestamp=i * 2 + 1
        ))
    
    # Show final stats
    stats = manager.get_token_stats()
    print(f"\n{'=' * 70}")
    print("FINAL STATISTICS")
    print(f"{'=' * 70}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.2f}")
        else:
            print(f"{key:25s}: {value}")
    
    # Example 2: Compare compression strategies
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Comparing Compression Strategies")
    print("=" * 70)
    
    strategies = [
        CompressionStrategy.TRUNCATE_OLD,
        CompressionStrategy.SUMMARIZE,
        CompressionStrategy.SELECTIVE,
    ]
    
    results = []
    
    for strategy in strategies:
        manager = TokenBudgetManager(
            TokenBudget(max_total_tokens=500, max_output_tokens=100),
            strategy
        )
        
        # Add many messages to trigger compression
        for i in range(15):
            manager.add_message(Message(
                role="user",
                content=f"Message {i}: " + "test " * 30,
                timestamp=i
            ))
        
        stats = manager.get_token_stats()
        results.append((strategy.value, stats))
    
    print(f"\n{'Strategy':<20s} {'Messages':<10s} {'Tokens':<10s} {'Compressions':<15s}")
    print("-" * 70)
    for strategy_name, stats in results:
        print(
            f"{strategy_name:<20s} "
            f"{stats['messages_count']:<10d} "
            f"{stats['current_tokens']:<10d} "
            f"{stats['compressions_applied']:<15d}"
        )


def demonstrate_best_practices():
    """Show best practices for token budget management"""
    print("\n" + "=" * 70)
    print("TOKEN BUDGET MANAGEMENT BEST PRACTICES")
    print("=" * 70)
    
    print("\n1. MONITOR USAGE:")
    print("   - Track token counts continuously")
    print("   - Set up alerts for high usage")
    print("   - Log compression events")
    
    print("\n2. PRIORITIZE CONTENT:")
    print("   - Mark critical messages (system prompts)")
    print("   - Protect recent context")
    print("   - Drop low-priority content first")
    
    print("\n3. CHOOSE RIGHT STRATEGY:")
    print("   - Truncate: Simple, fast, predictable")
    print("   - Summarize: Better context preservation")
    print("   - Selective: Balance importance and recency")
    print("   - Hierarchical: Best for very long conversations")
    
    print("\n4. COST OPTIMIZATION:")
    print("   - Use smaller models when possible")
    print("   - Cache frequently used prompts")
    print("   - Compress aggressively for low-priority tasks")
    
    print("\n5. USER EXPERIENCE:")
    print("   - Maintain conversation coherence")
    print("   - Avoid abrupt context loss")
    print("   - Provide feedback on compression")


if __name__ == "__main__":
    demonstrate_token_budget_management()
    demonstrate_best_practices()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Token budget management is essential for cost control
2. Multiple compression strategies available for different needs
3. Prioritize important messages to preserve critical context
4. Monitor usage continuously to avoid budget exhaustion
5. Balance cost, performance, and context quality

Best Practices:
- Set realistic budgets with safety buffers
- Use appropriate compression strategy for use case
- Prioritize messages by importance
- Track and analyze token usage patterns
- Implement graceful degradation
- Test compression impact on quality
- Consider model-specific token limits
    """)
