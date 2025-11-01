"""
Pattern 157: Adapter/Wrapper Pattern

Description:
    Provides unified interfaces to different LLM providers and models, abstracting
    away provider-specific details. Enables easy switching between models without
    changing application code.

Components:
    - Base interface definition
    - Provider-specific adapters
    - Response normalization
    - Error handling translation
    - Configuration management

Use Cases:
    - Multi-provider support
    - Model switching
    - Vendor independence
    - Testing with different models
    - Cost optimization

Benefits:
    - Provider independence
    - Simplified integration
    - Easy model switching
    - Consistent interface
    - Testability

Trade-offs:
    - Abstraction overhead
    - May limit provider-specific features
    - Mapping complexity
    - Performance overhead

LangChain Implementation:
    Uses abstract base classes, provider-specific implementations,
    and normalization layers for consistent responses
"""

import os
from typing import Dict, Any, List, Optional, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

load_dotenv()


class ModelProvider(Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


@dataclass
class ModelConfig:
    """Model configuration"""
    provider: ModelProvider
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None


@dataclass
class Message:
    """Standardized message format"""
    role: str  # system, user, assistant
    content: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CompletionResponse:
    """Standardized completion response"""
    content: str
    model: str
    provider: ModelProvider
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters"""
    
    @abstractmethod
    def complete(self, messages: List[Message], **kwargs) -> CompletionResponse:
        """Generate completion from messages"""
        pass
    
    @abstractmethod
    def stream(self, messages: List[Message], **kwargs):
        """Stream completion"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI models"""
    
    def __init__(self, config: ModelConfig):
        """Initialize OpenAI adapter"""
        self.config = config
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key or os.getenv("OPENAI_API_KEY")
        )
    
    def complete(self, messages: List[Message], **kwargs) -> CompletionResponse:
        """Generate completion"""
        # Convert messages to LangChain format
        lc_messages = self._convert_messages(messages)
        
        try:
            # Generate response
            response = self.llm.invoke(lc_messages)
            
            return CompletionResponse(
                content=response.content,
                model=self.config.model_name,
                provider=ModelProvider.OPENAI,
                finish_reason="stop",
                metadata={
                    "response_type": type(response).__name__
                }
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI completion failed: {str(e)}")
    
    def stream(self, messages: List[Message], **kwargs):
        """Stream completion"""
        lc_messages = self._convert_messages(messages)
        
        try:
            for chunk in self.llm.stream(lc_messages):
                yield chunk.content
        except Exception as e:
            raise RuntimeError(f"OpenAI streaming failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": ModelProvider.OPENAI.value,
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
    
    def _convert_messages(self, messages: List[Message]) -> List[BaseMessage]:
        """Convert standard messages to LangChain messages"""
        lc_messages = []
        for msg in messages:
            if msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
        return lc_messages


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic models (simulated)"""
    
    def __init__(self, config: ModelConfig):
        """Initialize Anthropic adapter"""
        self.config = config
        # In real implementation, use actual Anthropic client
        # For demo, we'll use OpenAI as fallback
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=config.temperature
        )
    
    def complete(self, messages: List[Message], **kwargs) -> CompletionResponse:
        """Generate completion"""
        # Convert to format expected by Anthropic (simulated)
        prompt = self._format_anthropic_prompt(messages)
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            return CompletionResponse(
                content=response.content,
                model=self.config.model_name,
                provider=ModelProvider.ANTHROPIC,
                finish_reason="end_turn",
                metadata={
                    "simulated": True
                }
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic completion failed: {str(e)}")
    
    def stream(self, messages: List[Message], **kwargs):
        """Stream completion"""
        prompt = self._format_anthropic_prompt(messages)
        
        try:
            for chunk in self.llm.stream([HumanMessage(content=prompt)]):
                yield chunk.content
        except Exception as e:
            raise RuntimeError(f"Anthropic streaming failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": ModelProvider.ANTHROPIC.value,
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "simulated": True
        }
    
    def _format_anthropic_prompt(self, messages: List[Message]) -> str:
        """Format messages for Anthropic API"""
        # Anthropic uses a different format
        formatted = []
        for msg in messages:
            if msg.role == "system":
                formatted.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted.append(f"Human: {msg.content}")
            elif msg.role == "assistant":
                formatted.append(f"Assistant: {msg.content}")
        formatted.append("Assistant:")
        return "\n\n".join(formatted)


class GoogleAdapter(LLMAdapter):
    """Adapter for Google models (simulated)"""
    
    def __init__(self, config: ModelConfig):
        """Initialize Google adapter"""
        self.config = config
        # In real implementation, use actual Google client
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=config.temperature
        )
    
    def complete(self, messages: List[Message], **kwargs) -> CompletionResponse:
        """Generate completion"""
        # Convert to Google format (simulated)
        lc_messages = []
        for msg in messages:
            if msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
        
        try:
            response = self.llm.invoke(lc_messages)
            
            return CompletionResponse(
                content=response.content,
                model=self.config.model_name,
                provider=ModelProvider.GOOGLE,
                finish_reason="STOP",
                metadata={
                    "simulated": True
                }
            )
        except Exception as e:
            raise RuntimeError(f"Google completion failed: {str(e)}")
    
    def stream(self, messages: List[Message], **kwargs):
        """Stream completion"""
        lc_messages = []
        for msg in messages:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
        
        try:
            for chunk in self.llm.stream(lc_messages):
                yield chunk.content
        except Exception as e:
            raise RuntimeError(f"Google streaming failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "provider": ModelProvider.GOOGLE.value,
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "simulated": True
        }


class UnifiedLLMInterface:
    """Unified interface for multiple LLM providers"""
    
    def __init__(self):
        """Initialize unified interface"""
        self.adapters: Dict[ModelProvider, LLMAdapter] = {}
        self.default_provider: Optional[ModelProvider] = None
    
    def register_adapter(self, provider: ModelProvider, adapter: LLMAdapter):
        """Register a provider adapter"""
        self.adapters[provider] = adapter
        if self.default_provider is None:
            self.default_provider = provider
        print(f"✓ Registered adapter for {provider.value}")
    
    def set_default_provider(self, provider: ModelProvider):
        """Set default provider"""
        if provider not in self.adapters:
            raise ValueError(f"Provider {provider.value} not registered")
        self.default_provider = provider
        print(f"✓ Set default provider to {provider.value}")
    
    def complete(self, messages: List[Message], 
                provider: Optional[ModelProvider] = None, 
                **kwargs) -> CompletionResponse:
        """Generate completion using specified or default provider"""
        provider = provider or self.default_provider
        
        if provider not in self.adapters:
            raise ValueError(f"Provider {provider.value} not registered")
        
        adapter = self.adapters[provider]
        return adapter.complete(messages, **kwargs)
    
    def stream(self, messages: List[Message], 
              provider: Optional[ModelProvider] = None, 
              **kwargs):
        """Stream completion using specified or default provider"""
        provider = provider or self.default_provider
        
        if provider not in self.adapters:
            raise ValueError(f"Provider {provider.value} not registered")
        
        adapter = self.adapters[provider]
        return adapter.stream(messages, **kwargs)
    
    def complete_with_fallback(self, messages: List[Message], 
                              providers: Optional[List[ModelProvider]] = None,
                              **kwargs) -> CompletionResponse:
        """Try completion with fallback providers"""
        if providers is None:
            providers = [self.default_provider] + [
                p for p in self.adapters.keys() if p != self.default_provider
            ]
        
        last_error = None
        for provider in providers:
            try:
                print(f"Trying provider: {provider.value}")
                return self.complete(messages, provider=provider, **kwargs)
            except Exception as e:
                print(f"  Failed: {str(e)}")
                last_error = e
                continue
        
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    def get_all_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered models"""
        return {
            provider.value: adapter.get_model_info()
            for provider, adapter in self.adapters.items()
        }


def demonstrate_adapter_pattern():
    """Demonstrate adapter/wrapper pattern"""
    print("=" * 80)
    print("ADAPTER/WRAPPER PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Initialize unified interface
    interface = UnifiedLLMInterface()
    
    # Register adapters for different providers
    print("\n" + "="*80)
    print("SETUP: Registering Provider Adapters")
    print("="*80)
    
    openai_config = ModelConfig(
        provider=ModelProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    interface.register_adapter(
        ModelProvider.OPENAI,
        OpenAIAdapter(openai_config)
    )
    
    anthropic_config = ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-sonnet",
        temperature=0.7
    )
    interface.register_adapter(
        ModelProvider.ANTHROPIC,
        AnthropicAdapter(anthropic_config)
    )
    
    google_config = ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-pro",
        temperature=0.7
    )
    interface.register_adapter(
        ModelProvider.GOOGLE,
        GoogleAdapter(google_config)
    )
    
    # Example 1: Complete with default provider
    print("\n" + "="*80)
    print("EXAMPLE 1: Completion with Default Provider")
    print("="*80)
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What is the capital of France?")
    ]
    response = interface.complete(messages)
    print(f"Provider: {response.provider.value}")
    print(f"Model: {response.model}")
    print(f"Response: {response.content[:200]}...")
    
    # Example 2: Complete with specific provider
    print("\n" + "="*80)
    print("EXAMPLE 2: Completion with Specific Provider (Anthropic)")
    print("="*80)
    response = interface.complete(messages, provider=ModelProvider.ANTHROPIC)
    print(f"Provider: {response.provider.value}")
    print(f"Model: {response.model}")
    print(f"Response: {response.content[:200]}...")
    
    # Example 3: Switch between providers
    print("\n" + "="*80)
    print("EXAMPLE 3: Switching Between Providers")
    print("="*80)
    question = Message(role="user", content="Explain quantum computing briefly.")
    
    for provider in [ModelProvider.OPENAI, ModelProvider.GOOGLE, ModelProvider.ANTHROPIC]:
        response = interface.complete([question], provider=provider)
        print(f"\n{provider.value}:")
        print(f"  {response.content[:150]}...")
    
    # Example 4: Streaming response
    print("\n" + "="*80)
    print("EXAMPLE 4: Streaming Response")
    print("="*80)
    stream_messages = [
        Message(role="user", content="Count from 1 to 5.")
    ]
    print("Streaming: ", end="", flush=True)
    for chunk in interface.stream(stream_messages):
        print(chunk, end="", flush=True)
    print()
    
    # Example 5: Fallback mechanism
    print("\n" + "="*80)
    print("EXAMPLE 5: Fallback Mechanism")
    print("="*80)
    messages = [
        Message(role="user", content="What is machine learning?")
    ]
    response = interface.complete_with_fallback(messages)
    print(f"Succeeded with provider: {response.provider.value}")
    print(f"Response: {response.content[:200]}...")
    
    # Example 6: Model information
    print("\n" + "="*80)
    print("EXAMPLE 6: Model Information")
    print("="*80)
    all_info = interface.get_all_model_info()
    for provider, info in all_info.items():
        print(f"\n{provider}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Example 7: Conversation across providers
    print("\n" + "="*80)
    print("EXAMPLE 7: Multi-Turn Conversation")
    print("="*80)
    conversation = [
        Message(role="user", content="What is Python?")
    ]
    
    # First turn with OpenAI
    response1 = interface.complete(conversation, provider=ModelProvider.OPENAI)
    print(f"Turn 1 ({response1.provider.value}): {response1.content[:100]}...")
    
    # Add response and continue
    conversation.append(Message(role="assistant", content=response1.content))
    conversation.append(Message(role="user", content="What are its main features?"))
    
    # Second turn with different provider
    response2 = interface.complete(conversation, provider=ModelProvider.GOOGLE)
    print(f"Turn 2 ({response2.provider.value}): {response2.content[:100]}...")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Adapter/Wrapper Pattern Best Practices")
    print("="*80)
    print("""
1. INTERFACE DESIGN:
   - Define clear, provider-agnostic interface
   - Standardize message formats
   - Normalize response structures
   - Handle provider-specific features gracefully

2. ADAPTER IMPLEMENTATION:
   - Encapsulate provider-specific logic
   - Handle format conversions
   - Map error codes consistently
   - Maintain provider-specific optimizations

3. RESPONSE NORMALIZATION:
   - Standardize response format
   - Map finish reasons consistently
   - Include provider metadata
   - Handle streaming uniformly

4. ERROR HANDLING:
   - Translate provider-specific errors
   - Provide meaningful error messages
   - Implement fallback mechanisms
   - Log provider-specific details

5. CONFIGURATION:
   - Centralize configuration management
   - Support provider-specific parameters
   - Enable easy provider switching
   - Validate configurations

6. TESTING:
   - Test each adapter independently
   - Verify format conversions
   - Test fallback mechanisms
   - Mock external API calls

Benefits:
✓ Provider independence and flexibility
✓ Easy model switching
✓ Simplified application code
✓ Consistent interface across providers
✓ Facilitates testing and mocking
✓ Vendor lock-in prevention

Challenges:
- Abstraction may limit provider features
- Mapping complexity for edge cases
- Performance overhead
- Maintaining feature parity
- Handling provider-specific capabilities
    """)


if __name__ == "__main__":
    demonstrate_adapter_pattern()
