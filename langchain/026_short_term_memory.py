"""
Pattern 026: Short-Term Memory

Description:
    The Short-Term Memory pattern enables agents to maintain context within the current
    conversation or task session. It provides immediate access to recent interactions,
    allowing the agent to reference previous messages, maintain conversation flow, and
    build upon earlier context without requiring persistent storage.

Components:
    - Conversation Buffer: Stores recent message history
    - Context Window Manager: Manages token limits
    - Message Formatter: Structures messages for LLM
    - Relevance Filter: Keeps most relevant context
    - Memory Summarizer: Condenses long conversations

Use Cases:
    - Multi-turn conversations
    - Task continuity within session
    - Context-aware responses
    - Follow-up questions
    - Anaphora resolution

LangChain Implementation:
    Uses LangChain's memory abstractions including ConversationBufferMemory,
    ConversationBufferWindowMemory, and ConversationSummaryMemory for
    different short-term memory strategies.

Key Features:
    - Automatic message history tracking
    - Context window management
    - Memory clearing and reset
    - Selective memory retention
    - Token-aware buffering
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()


class MemoryStrategy(Enum):
    """Strategy for managing short-term memory."""
    BUFFER = "buffer"  # Keep all messages
    WINDOW = "window"  # Keep last N messages
    TOKEN_LIMIT = "token_limit"  # Keep within token limit
    SUMMARY = "summary"  # Summarize older messages


@dataclass
class Message:
    """Represents a conversation message."""
    role: str  # "human", "ai", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_langchain_message(self):
        """Convert to LangChain message format."""
        if self.role == "human":
            return HumanMessage(content=self.content)
        elif self.role == "ai":
            return AIMessage(content=self.content)
        else:
            return SystemMessage(content=self.content)


@dataclass
class ConversationContext:
    """Context for current conversation."""
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
    
    def get_recent_messages(self, n: int) -> List[Message]:
        """Get the N most recent messages."""
        return self.messages[-n:] if n > 0 else []
    
    def get_messages_by_role(self, role: str) -> List[Message]:
        """Get all messages from a specific role."""
        return [msg for msg in self.messages if msg.role == role]
    
    def clear(self):
        """Clear all messages."""
        self.messages.clear()
        self.started_at = datetime.now()


class BufferMemory:
    """
    Simple buffer that stores all conversation messages.
    """
    
    def __init__(self, max_messages: Optional[int] = None):
        """
        Initialize buffer memory.
        
        Args:
            max_messages: Maximum number of messages to store (None = unlimited)
        """
        self.max_messages = max_messages
        self.context = ConversationContext()
    
    def add_interaction(self, human_message: str, ai_message: str):
        """Add a human-AI interaction pair."""
        self.context.add_message("human", human_message)
        self.context.add_message("ai", ai_message)
        
        # Trim if exceeding max
        if self.max_messages and len(self.context.messages) > self.max_messages:
            excess = len(self.context.messages) - self.max_messages
            self.context.messages = self.context.messages[excess:]
    
    def get_context_string(self) -> str:
        """Get conversation history as formatted string."""
        lines = []
        for msg in self.context.messages:
            role = msg.role.capitalize()
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
    
    def get_langchain_messages(self) -> List:
        """Get messages in LangChain format."""
        return [msg.to_langchain_message() for msg in self.context.messages]
    
    def clear(self):
        """Clear memory."""
        self.context.clear()


class WindowMemory:
    """
    Sliding window memory that keeps only the last N messages.
    """
    
    def __init__(self, window_size: int = 10):
        """
        Initialize window memory.
        
        Args:
            window_size: Number of recent messages to keep
        """
        self.window_size = window_size
        self.context = ConversationContext()
    
    def add_interaction(self, human_message: str, ai_message: str):
        """Add interaction and maintain window."""
        self.context.add_message("human", human_message)
        self.context.add_message("ai", ai_message)
        
        # Keep only window_size messages
        if len(self.context.messages) > self.window_size:
            self.context.messages = self.context.messages[-self.window_size:]
    
    def get_context_string(self) -> str:
        """Get recent conversation history."""
        lines = []
        for msg in self.context.messages:
            role = msg.role.capitalize()
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
    
    def get_langchain_messages(self) -> List:
        """Get messages in LangChain format."""
        return [msg.to_langchain_message() for msg in self.context.messages]


class TokenLimitMemory:
    """
    Memory that keeps messages within a token budget.
    """
    
    def __init__(self, max_tokens: int = 2000):
        """
        Initialize token-limit memory.
        
        Args:
            max_tokens: Maximum tokens to keep in memory
        """
        self.max_tokens = max_tokens
        self.context = ConversationContext()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def add_interaction(self, human_message: str, ai_message: str):
        """Add interaction and maintain token limit."""
        self.context.add_message("human", human_message)
        self.context.add_message("ai", ai_message)
        
        # Remove oldest messages until under token limit
        while self._get_total_tokens() > self.max_tokens and len(self.context.messages) > 2:
            self.context.messages.pop(0)
    
    def _get_total_tokens(self) -> int:
        """Calculate total tokens in memory."""
        total = 0
        for msg in self.context.messages:
            total += self._estimate_tokens(msg.content)
        return total
    
    def get_context_string(self) -> str:
        """Get conversation history."""
        lines = []
        for msg in self.context.messages:
            role = msg.role.capitalize()
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
    
    def get_token_count(self) -> int:
        """Get current token count."""
        return self._get_total_tokens()


class SummaryMemory:
    """
    Memory that summarizes older messages to save space.
    """
    
    def __init__(
        self,
        recent_messages: int = 4,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize summary memory.
        
        Args:
            recent_messages: Number of recent messages to keep verbatim
            model: LLM model for summarization
        """
        self.recent_messages = recent_messages
        self.llm = ChatOpenAI(model=model, temperature=0.3)
        self.context = ConversationContext()
        self.summary = ""
    
    def add_interaction(self, human_message: str, ai_message: str):
        """Add interaction and update summary if needed."""
        self.context.add_message("human", human_message)
        self.context.add_message("ai", ai_message)
        
        # If we have more than recent_messages, summarize older ones
        if len(self.context.messages) > self.recent_messages:
            messages_to_summarize = self.context.messages[:-self.recent_messages]
            self.summary = self._create_summary(messages_to_summarize)
            self.context.messages = self.context.messages[-self.recent_messages:]
    
    def _create_summary(self, messages: List[Message]) -> str:
        """Create summary of messages."""
        if not messages:
            return self.summary
        
        # Format messages
        formatted = "\n".join([
            f"{msg.role.capitalize()}: {msg.content}"
            for msg in messages
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Summarize the following conversation, preserving key information
and context. Be concise but comprehensive."""),
            ("user", "Previous summary: {summary}\n\nNew messages:\n{messages}\n\nProvide updated summary:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        new_summary = chain.invoke({
            "summary": self.summary if self.summary else "None",
            "messages": formatted
        })
        
        return new_summary.strip()
    
    def get_context_string(self) -> str:
        """Get conversation context with summary."""
        parts = []
        
        if self.summary:
            parts.append(f"[Summary of earlier conversation: {self.summary}]\n")
        
        for msg in self.context.messages:
            role = msg.role.capitalize()
            parts.append(f"{role}: {msg.content}")
        
        return "\n".join(parts)


class ShortTermMemoryAgent:
    """
    Agent with short-term memory for conversations.
    """
    
    def __init__(
        self,
        memory_strategy: MemoryStrategy = MemoryStrategy.WINDOW,
        model: str = "gpt-3.5-turbo",
        **memory_kwargs
    ):
        """
        Initialize agent with short-term memory.
        
        Args:
            memory_strategy: Strategy for memory management
            model: LLM model to use
            **memory_kwargs: Arguments for memory initialization
        """
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        self.memory_strategy = memory_strategy
        
        # Initialize appropriate memory type
        if memory_strategy == MemoryStrategy.BUFFER:
            self.memory = BufferMemory(**memory_kwargs)
        elif memory_strategy == MemoryStrategy.WINDOW:
            self.memory = WindowMemory(**memory_kwargs)
        elif memory_strategy == MemoryStrategy.TOKEN_LIMIT:
            self.memory = TokenLimitMemory(**memory_kwargs)
        elif memory_strategy == MemoryStrategy.SUMMARY:
            self.memory = SummaryMemory(**memory_kwargs)
    
    def chat(self, user_message: str) -> str:
        """
        Chat with the agent.
        
        Args:
            user_message: User's message
            
        Returns:
            Agent's response
        """
        # Get conversation context
        context = self.memory.get_context_string()
        
        # Create prompt with context
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Use the conversation history
to provide context-aware responses. Reference previous messages when relevant."""),
            ("user", "{context}\n\nHuman: {message}\n\nProvide a helpful response:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        # Generate response
        response = chain.invoke({
            "context": context if context else "No previous conversation.",
            "message": user_message
        })
        
        # Store interaction
        self.memory.add_interaction(user_message, response)
        
        return response.strip()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        message_count = len(self.memory.context.messages)
        
        summary = {
            "strategy": self.memory_strategy.value,
            "message_count": message_count,
            "human_messages": len(self.memory.context.get_messages_by_role("human")),
            "ai_messages": len(self.memory.context.get_messages_by_role("ai")),
            "started_at": self.memory.context.started_at.isoformat()
        }
        
        # Add strategy-specific info
        if hasattr(self.memory, 'get_token_count'):
            summary["token_count"] = self.memory.get_token_count()
        
        if hasattr(self.memory, 'summary') and self.memory.summary:
            summary["has_summary"] = True
            summary["summary_length"] = len(self.memory.summary)
        
        return summary
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.context.clear()


def demonstrate_short_term_memory():
    """Demonstrate the Short-Term Memory pattern."""
    
    print("=" * 80)
    print("SHORT-TERM MEMORY PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Test 1: Buffer Memory
    print("\n" + "=" * 80)
    print("TEST 1: Buffer Memory (Unlimited)")
    print("=" * 80)
    
    agent1 = ShortTermMemoryAgent(
        memory_strategy=MemoryStrategy.BUFFER,
        max_messages=None
    )
    
    conversation1 = [
        "Hi, my name is Alice.",
        "What's my name?",
        "I'm interested in learning about AI.",
        "What did I say I was interested in?"
    ]
    
    for msg in conversation1:
        print(f"\n👤 Human: {msg}")
        response = agent1.chat(msg)
        print(f"🤖 AI: {response}")
    
    summary1 = agent1.get_conversation_summary()
    print(f"\n📊 Conversation Summary:")
    print(f"   Messages: {summary1['message_count']}")
    print(f"   Strategy: {summary1['strategy']}")
    
    # Test 2: Window Memory
    print("\n\n" + "=" * 80)
    print("TEST 2: Window Memory (Last 4 Messages)")
    print("=" * 80)
    
    agent2 = ShortTermMemoryAgent(
        memory_strategy=MemoryStrategy.WINDOW,
        window_size=4
    )
    
    conversation2 = [
        "My favorite color is blue.",
        "I live in New York.",
        "I work as a teacher.",
        "I like pizza.",
        "What's my favorite color?",  # This should be forgotten
        "Where do I live?",  # This should be remembered
    ]
    
    for msg in conversation2:
        print(f"\n👤 Human: {msg}")
        response = agent2.chat(msg)
        print(f"🤖 AI: {response}")
    
    summary2 = agent2.get_conversation_summary()
    print(f"\n📊 Conversation Summary:")
    print(f"   Messages in window: {summary2['message_count']}")
    print(f"   Strategy: {summary2['strategy']}")
    
    # Test 3: Token Limit Memory
    print("\n\n" + "=" * 80)
    print("TEST 3: Token Limit Memory (2000 tokens)")
    print("=" * 80)
    
    agent3 = ShortTermMemoryAgent(
        memory_strategy=MemoryStrategy.TOKEN_LIMIT,
        max_tokens=2000
    )
    
    conversation3 = [
        "Let me tell you about my day. " * 20,  # Long message
        "What did I just tell you about?",
        "Here's another long story. " * 20,  # Another long message
        "Can you remember both stories?"
    ]
    
    for msg in conversation3:
        msg_preview = msg[:60] + "..." if len(msg) > 60 else msg
        print(f"\n👤 Human: {msg_preview}")
        response = agent3.chat(msg)
        response_preview = response[:100] + "..." if len(response) > 100 else response
        print(f"🤖 AI: {response_preview}")
    
    summary3 = agent3.get_conversation_summary()
    print(f"\n📊 Conversation Summary:")
    print(f"   Messages: {summary3['message_count']}")
    print(f"   Token count: {summary3['token_count']}")
    print(f"   Strategy: {summary3['strategy']}")
    
    # Summary
    print("\n\n" + "=" * 80)
    print("PATTERN SUMMARY")
    print("=" * 80)
    print("""
The Short-Term Memory pattern demonstrates:

1. **Context Continuity**: Maintains conversation flow
2. **Multiple Strategies**: Different approaches for different needs
3. **Resource Management**: Controls memory usage
4. **Context Awareness**: References previous messages
5. **Session Scoped**: Memory persists within session

Memory Strategies Compared:

Buffer Memory:
- ✓ Complete conversation history
- ✓ Perfect recall
- ✗ Unbounded growth
- Use: Short conversations, debugging

Window Memory:
- ✓ Fixed memory size
- ✓ Recent context preserved
- ✗ Forgets older messages
- Use: Long conversations, resource constraints

Token Limit Memory:
- ✓ Token budget control
- ✓ Adapts to message length
- ✗ May lose context unpredictably
- Use: API token limits, cost control

Summary Memory:
- ✓ Compact representation
- ✓ Preserves key information
- ✗ Lossy compression
- Use: Very long conversations

Key Benefits:
- **Natural Conversations**: Agents remember context
- **Anaphora Resolution**: Handle "it", "that", "they" references
- **Follow-up Questions**: Build on previous topics
- **Resource Control**: Manage memory/token usage
- **Session Management**: Clear boundaries

Use Cases:
- Customer service chatbots
- Personal assistants
- Tutorial/teaching agents
- Technical support
- Conversational interfaces

This pattern is fundamental for any conversational AI system,
enabling natural multi-turn interactions with context awareness.
""")


if __name__ == "__main__":
    demonstrate_short_term_memory()
