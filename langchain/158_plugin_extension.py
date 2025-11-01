"""
Pattern 158: Plugin/Extension Architecture

Description:
    Enables extensibility through a plugin system where new capabilities can be
    added dynamically without modifying core code. Plugins follow a standard
    interface and can be loaded, configured, and executed at runtime.

Components:
    - Plugin interface/protocol
    - Plugin registry
    - Plugin loader
    - Lifecycle management
    - Dependency injection
    - Configuration system

Use Cases:
    - Custom tool integration
    - Domain-specific capabilities
    - Third-party extensions
    - Modular agent systems
    - Dynamic feature addition

Benefits:
    - Extensibility without core changes
    - Modular architecture
    - Third-party contributions
    - Dynamic capability loading
    - Separation of concerns

Trade-offs:
    - Complexity overhead
    - Plugin management burden
    - Potential security risks
    - Version compatibility issues

LangChain Implementation:
    Uses protocol classes for interfaces, dynamic loading,
    registry pattern, and lifecycle hooks
"""

import os
import importlib
import inspect
from typing import Dict, Any, List, Optional, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class PluginStatus(Enum):
    """Plugin status"""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    DISABLED = "disabled"
    ERROR = "error"


class PluginType(Enum):
    """Plugin types"""
    TOOL = "tool"
    PROCESSOR = "processor"
    FILTER = "filter"
    ANALYZER = "analyzer"
    GENERATOR = "generator"


@dataclass
class PluginMetadata:
    """Plugin metadata"""
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginConfig:
    """Plugin configuration"""
    enabled: bool = True
    priority: int = 0
    settings: Dict[str, Any] = field(default_factory=dict)


class Plugin(Protocol):
    """Plugin interface using Protocol (structural typing)"""
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        ...
    
    def initialize(self, config: PluginConfig) -> bool:
        """Initialize plugin"""
        ...
    
    def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Execute plugin functionality"""
        ...
    
    def shutdown(self) -> bool:
        """Cleanup and shutdown"""
        ...


class BasePlugin(ABC):
    """Base class for plugins"""
    
    def __init__(self):
        """Initialize base plugin"""
        self.config: Optional[PluginConfig] = None
        self.status = PluginStatus.UNLOADED
        self._metadata: Optional[PluginMetadata] = None
    
    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata"""
        if self._metadata is None:
            self._metadata = self._create_metadata()
        return self._metadata
    
    @abstractmethod
    def _create_metadata(self) -> PluginMetadata:
        """Create plugin metadata"""
        pass
    
    def initialize(self, config: PluginConfig) -> bool:
        """Initialize plugin"""
        try:
            self.config = config
            self._on_initialize()
            self.status = PluginStatus.INITIALIZED
            if config.enabled:
                self.status = PluginStatus.ACTIVE
            return True
        except Exception as e:
            self.status = PluginStatus.ERROR
            print(f"Plugin initialization failed: {e}")
            return False
    
    def _on_initialize(self):
        """Override for custom initialization"""
        pass
    
    @abstractmethod
    def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Execute plugin functionality"""
        pass
    
    def shutdown(self) -> bool:
        """Cleanup and shutdown"""
        try:
            self._on_shutdown()
            self.status = PluginStatus.UNLOADED
            return True
        except Exception as e:
            print(f"Plugin shutdown failed: {e}")
            return False
    
    def _on_shutdown(self):
        """Override for custom shutdown"""
        pass


# Example Plugin Implementations

class SentimentAnalysisPlugin(BasePlugin):
    """Plugin for sentiment analysis"""
    
    def _create_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="sentiment_analyzer",
            version="1.0.0",
            author="Example Corp",
            description="Analyzes sentiment of text using LLM",
            plugin_type=PluginType.ANALYZER
        )
    
    def _on_initialize(self):
        """Initialize LLM"""
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Analyze sentiment"""
        text = input_data if isinstance(input_data, str) else str(input_data)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the sentiment of the given text. Respond with: positive, negative, or neutral."),
            ("user", "{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        sentiment = chain.invoke({"text": text})
        
        return {
            "sentiment": sentiment.strip().lower(),
            "text": text,
            "plugin": self.metadata.name
        }


class TextSummarizerPlugin(BasePlugin):
    """Plugin for text summarization"""
    
    def _create_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="text_summarizer",
            version="1.0.0",
            author="Example Corp",
            description="Summarizes text content",
            plugin_type=PluginType.PROCESSOR
        )
    
    def _on_initialize(self):
        """Initialize LLM"""
        max_words = self.config.settings.get("max_words", 50) if self.config else 50
        self.max_words = max_words
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Summarize text"""
        text = input_data if isinstance(input_data, str) else str(input_data)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following text in {max_words} words or less."),
            ("user", "{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        summary = chain.invoke({"text": text, "max_words": self.max_words})
        
        return {
            "summary": summary,
            "original_length": len(text.split()),
            "summary_length": len(summary.split()),
            "plugin": self.metadata.name
        }


class KeywordExtractorPlugin(BasePlugin):
    """Plugin for keyword extraction"""
    
    def _create_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="keyword_extractor",
            version="1.0.0",
            author="Example Corp",
            description="Extracts keywords from text",
            plugin_type=PluginType.ANALYZER
        )
    
    def _on_initialize(self):
        """Initialize LLM"""
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Extract keywords"""
        text = input_data if isinstance(input_data, str) else str(input_data)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract 5-7 key terms or phrases from the text. Return as comma-separated list."),
            ("user", "{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        keywords_str = chain.invoke({"text": text})
        
        keywords = [k.strip() for k in keywords_str.split(",")]
        
        return {
            "keywords": keywords,
            "count": len(keywords),
            "plugin": self.metadata.name
        }


class ProfanityFilterPlugin(BasePlugin):
    """Plugin for filtering profanity"""
    
    def _create_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="profanity_filter",
            version="1.0.0",
            author="Example Corp",
            description="Filters inappropriate content",
            plugin_type=PluginType.FILTER
        )
    
    def _on_initialize(self):
        """Initialize filter"""
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    def execute(self, input_data: Any, context: Dict[str, Any]) -> Any:
        """Filter content"""
        text = input_data if isinstance(input_data, str) else str(input_data)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze if the text contains profanity or inappropriate content. Respond with 'yes' or 'no'."),
            ("user", "{text}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        contains_profanity = chain.invoke({"text": text})
        
        is_inappropriate = "yes" in contains_profanity.lower()
        
        return {
            "is_appropriate": not is_inappropriate,
            "filtered": is_inappropriate,
            "text": text if not is_inappropriate else "[CONTENT FILTERED]",
            "plugin": self.metadata.name
        }


class PluginRegistry:
    """Manages plugin registration and lifecycle"""
    
    def __init__(self):
        """Initialize plugin registry"""
        self.plugins: Dict[str, BasePlugin] = {}
        self.configs: Dict[str, PluginConfig] = {}
        self.execution_order: List[str] = []
    
    def register_plugin(self, plugin: BasePlugin, config: Optional[PluginConfig] = None):
        """Register a plugin"""
        if config is None:
            config = PluginConfig()
        
        plugin_name = plugin.metadata.name
        
        # Check dependencies
        for dep in plugin.metadata.dependencies:
            if dep not in self.plugins:
                print(f"Warning: Missing dependency '{dep}' for plugin '{plugin_name}'")
        
        # Register plugin
        self.plugins[plugin_name] = plugin
        self.configs[plugin_name] = config
        
        # Initialize plugin
        success = plugin.initialize(config)
        
        if success:
            # Add to execution order based on priority
            self._update_execution_order(plugin_name, config.priority)
            print(f"✓ Registered and initialized plugin: {plugin_name}")
        else:
            print(f"✗ Failed to initialize plugin: {plugin_name}")
        
        return success
    
    def _update_execution_order(self, plugin_name: str, priority: int):
        """Update execution order based on priority"""
        if plugin_name in self.execution_order:
            self.execution_order.remove(plugin_name)
        
        # Insert based on priority (higher priority first)
        inserted = False
        for i, name in enumerate(self.execution_order):
            if self.configs[name].priority < priority:
                self.execution_order.insert(i, plugin_name)
                inserted = True
                break
        
        if not inserted:
            self.execution_order.append(plugin_name)
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin"""
        if plugin_name not in self.plugins:
            return False
        
        plugin = self.plugins[plugin_name]
        plugin.shutdown()
        
        del self.plugins[plugin_name]
        del self.configs[plugin_name]
        self.execution_order.remove(plugin_name)
        
        print(f"✓ Unregistered plugin: {plugin_name}")
        return True
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get plugin by name"""
        return self.plugins.get(plugin_name)
    
    def execute_plugin(self, plugin_name: str, input_data: Any, 
                      context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute specific plugin"""
        if context is None:
            context = {}
        
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        if plugin.status != PluginStatus.ACTIVE:
            raise RuntimeError(f"Plugin not active: {plugin_name}")
        
        return plugin.execute(input_data, context)
    
    def execute_pipeline(self, input_data: Any, 
                        plugin_types: Optional[List[PluginType]] = None,
                        context: Optional[Dict[str, Any]] = None) -> Any:
        """Execute plugins in pipeline"""
        if context is None:
            context = {}
        
        result = input_data
        
        for plugin_name in self.execution_order:
            plugin = self.plugins[plugin_name]
            config = self.configs[plugin_name]
            
            # Skip if not enabled
            if not config.enabled or plugin.status != PluginStatus.ACTIVE:
                continue
            
            # Filter by type if specified
            if plugin_types and plugin.metadata.plugin_type not in plugin_types:
                continue
            
            # Execute plugin
            result = plugin.execute(result, context)
        
        return result
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins"""
        return [
            {
                "name": plugin.metadata.name,
                "version": plugin.metadata.version,
                "type": plugin.metadata.plugin_type.value,
                "status": plugin.status.value,
                "enabled": self.configs[name].enabled,
                "priority": self.configs[name].priority
            }
            for name, plugin in self.plugins.items()
        ]
    
    def shutdown_all(self):
        """Shutdown all plugins"""
        for plugin_name in list(self.plugins.keys()):
            self.unregister_plugin(plugin_name)


def demonstrate_plugin_architecture():
    """Demonstrate plugin/extension architecture"""
    print("=" * 80)
    print("PLUGIN/EXTENSION ARCHITECTURE DEMONSTRATION")
    print("=" * 80)
    
    # Initialize registry
    registry = PluginRegistry()
    
    # Example 1: Register plugins
    print("\n" + "="*80)
    print("EXAMPLE 1: Registering Plugins")
    print("="*80)
    
    # Register sentiment analyzer
    sentiment_plugin = SentimentAnalysisPlugin()
    registry.register_plugin(sentiment_plugin, PluginConfig(
        enabled=True,
        priority=10
    ))
    
    # Register summarizer
    summarizer_plugin = TextSummarizerPlugin()
    registry.register_plugin(summarizer_plugin, PluginConfig(
        enabled=True,
        priority=5,
        settings={"max_words": 30}
    ))
    
    # Register keyword extractor
    keyword_plugin = KeywordExtractorPlugin()
    registry.register_plugin(keyword_plugin, PluginConfig(
        enabled=True,
        priority=3
    ))
    
    # Register profanity filter
    filter_plugin = ProfanityFilterPlugin()
    registry.register_plugin(filter_plugin, PluginConfig(
        enabled=True,
        priority=20  # High priority - run first
    ))
    
    # Example 2: List plugins
    print("\n" + "="*80)
    print("EXAMPLE 2: List Registered Plugins")
    print("="*80)
    plugins = registry.list_plugins()
    for plugin in plugins:
        print(f"- {plugin['name']} v{plugin['version']}")
        print(f"  Type: {plugin['type']}, Status: {plugin['status']}")
        print(f"  Priority: {plugin['priority']}, Enabled: {plugin['enabled']}")
    
    # Example 3: Execute individual plugin
    print("\n" + "="*80)
    print("EXAMPLE 3: Execute Individual Plugin")
    print("="*80)
    text = "This product is amazing! I love it so much!"
    result = registry.execute_plugin("sentiment_analyzer", text)
    print(f"Input: {text}")
    print(f"Result: {result}")
    
    # Example 4: Execute multiple plugins
    print("\n" + "="*80)
    print("EXAMPLE 4: Execute Multiple Plugins")
    print("="*80)
    text = "Artificial intelligence is transforming how we work and live. Machine learning models can now process vast amounts of data and make predictions with high accuracy."
    
    print(f"Input: {text[:100]}...\n")
    
    # Sentiment
    sentiment_result = registry.execute_plugin("sentiment_analyzer", text)
    print(f"Sentiment: {sentiment_result['sentiment']}")
    
    # Summary
    summary_result = registry.execute_plugin("text_summarizer", text)
    print(f"Summary: {summary_result['summary']}")
    
    # Keywords
    keyword_result = registry.execute_plugin("keyword_extractor", text)
    print(f"Keywords: {', '.join(keyword_result['keywords'])}")
    
    # Example 5: Filter by plugin type
    print("\n" + "="*80)
    print("EXAMPLE 5: Execute by Plugin Type")
    print("="*80)
    text = "The company announced record profits this quarter."
    
    # Execute only analyzers
    print("Executing ANALYZER plugins:")
    result = registry.execute_pipeline(text, plugin_types=[PluginType.ANALYZER])
    print(f"Result: {result}")
    
    # Example 6: Content filtering
    print("\n" + "="*80)
    print("EXAMPLE 6: Content Filtering")
    print("="*80)
    clean_text = "This is a nice and appropriate message."
    result = registry.execute_plugin("profanity_filter", clean_text)
    print(f"Text: {clean_text}")
    print(f"Is Appropriate: {result['is_appropriate']}")
    print(f"Output: {result['text']}")
    
    # Example 7: Execution order (priority)
    print("\n" + "="*80)
    print("EXAMPLE 7: Plugin Execution Order")
    print("="*80)
    print("Execution order (by priority):")
    for i, plugin_name in enumerate(registry.execution_order, 1):
        plugin = registry.get_plugin(plugin_name)
        priority = registry.configs[plugin_name].priority
        print(f"{i}. {plugin_name} (priority: {priority})")
    
    # Cleanup
    print("\n" + "="*80)
    print("CLEANUP: Shutting Down Plugins")
    print("="*80)
    registry.shutdown_all()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Plugin Architecture Best Practices")
    print("="*80)
    print("""
1. PLUGIN INTERFACE:
   - Define clear plugin interface/protocol
   - Standardize lifecycle methods (initialize, execute, shutdown)
   - Use metadata for plugin description
   - Support configuration schemas

2. REGISTRY MANAGEMENT:
   - Centralized plugin registration
   - Dependency checking
   - Priority-based execution
   - Enable/disable plugins dynamically

3. LIFECYCLE MANAGEMENT:
   - Proper initialization
   - Resource cleanup on shutdown
   - Error handling and status tracking
   - Graceful failure handling

4. EXECUTION PATTERNS:
   - Individual plugin execution
   - Pipeline execution
   - Type-based filtering
   - Context passing between plugins

5. CONFIGURATION:
   - Plugin-specific settings
   - Priority control
   - Enable/disable flags
   - Runtime configuration updates

6. EXTENSIBILITY:
   - Easy to add new plugins
   - No core code modification
   - Third-party plugin support
   - Hot-loading capabilities (advanced)

Benefits:
✓ Modular and extensible architecture
✓ Separation of concerns
✓ Easy feature addition
✓ Third-party contributions
✓ Dynamic capability enhancement
✓ Clean core codebase

Challenges:
- Plugin management complexity
- Version compatibility
- Security considerations for third-party plugins
- Performance overhead
- Dependency management
- Testing across plugin combinations
    """)


if __name__ == "__main__":
    demonstrate_plugin_architecture()
