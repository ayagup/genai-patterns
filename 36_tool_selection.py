"""
Dynamic Tool Selection Pattern Implementation

This module demonstrates dynamic tool selection and orchestration including:
- Tool discovery and registration
- Capability matching with queries
- Dynamic tool routing
- Multi-tool orchestration
- Tool execution with error handling

Key Components:
- Tool registry with capability descriptions
- Semantic matching between queries and tools
- Tool execution orchestrator
- Fallback and error handling
- Tool performance tracking
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Set, Tuple
from enum import Enum
import random
import re
from datetime import datetime


class ToolCategory(Enum):
    """Categories of tools"""
    COMPUTATION = "computation"
    INFORMATION = "information"
    COMMUNICATION = "communication"
    FILE_OPS = "file_ops"
    API = "api"
    DATA_PROCESSING = "data_processing"
    ANALYSIS = "analysis"
    GENERATION = "generation"


class ToolStatus(Enum):
    """Status of tool execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    INVALID_INPUT = "invalid_input"


@dataclass
class ToolCapability:
    """Describes what a tool can do"""
    name: str
    description: str
    keywords: Set[str]
    input_types: List[str]
    output_type: str
    examples: List[str] = field(default_factory=list)
    
    def matches_query(self, query: str) -> float:
        """Calculate how well this capability matches a query"""
        query_lower = query.lower()
        score = 0.0
        
        # Check if any keywords are in the query
        for keyword in self.keywords:
            if keyword.lower() in query_lower:
                score += 0.3
        
        # Check description similarity (simple word overlap)
        query_words = set(query_lower.split())
        desc_words = set(self.description.lower().split())
        overlap = len(query_words & desc_words)
        if query_words:
            score += (overlap / len(query_words)) * 0.7
        
        return min(1.0, score)


@dataclass
class Tool:
    """Represents a callable tool"""
    id: str
    name: str
    category: ToolCategory
    capabilities: List[ToolCapability]
    function: Callable
    cost: float = 1.0  # Relative execution cost
    reliability: float = 1.0  # Historical reliability (0-1)
    avg_execution_time: float = 0.0  # Average execution time in seconds
    usage_count: int = 0
    success_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool"""
        return self.function(*args, **kwargs)
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.usage_count == 0:
            return 1.0
        return self.success_count / self.usage_count
    
    def update_stats(self, success: bool, execution_time: float):
        """Update tool statistics"""
        self.usage_count += 1
        if success:
            self.success_count += 1
        
        # Update average execution time (exponential moving average)
        if self.avg_execution_time == 0:
            self.avg_execution_time = execution_time
        else:
            alpha = 0.3  # Smoothing factor
            self.avg_execution_time = (alpha * execution_time + 
                                      (1 - alpha) * self.avg_execution_time)


@dataclass
class ToolExecutionResult:
    """Result from tool execution"""
    tool_id: str
    tool_name: str
    status: ToolStatus
    output: Any = None
    error_message: str = ""
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    input_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSelectionResult:
    """Result from tool selection process"""
    query: str
    selected_tools: List[Tool]
    selection_scores: Dict[str, float]
    selection_reasoning: str
    execution_plan: List[str] = field(default_factory=list)


class ToolRegistry:
    """Registry for managing available tools"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.category_index: Dict[ToolCategory, List[str]] = {}
        self.capability_index: Dict[str, List[str]] = {}  # keyword -> tool_ids
    
    def register_tool(self, tool: Tool):
        """Register a tool in the registry"""
        self.tools[tool.id] = tool
        
        # Update category index
        if tool.category not in self.category_index:
            self.category_index[tool.category] = []
        self.category_index[tool.category].append(tool.id)
        
        # Update capability index
        for capability in tool.capabilities:
            for keyword in capability.keywords:
                if keyword not in self.capability_index:
                    self.capability_index[keyword] = []
                if tool.id not in self.capability_index[keyword]:
                    self.capability_index[keyword].append(tool.id)
        
        print(f"✅ Registered tool: {tool.name} ({tool.id})")
    
    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get a tool by ID"""
        return self.tools.get(tool_id)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get all tools in a category"""
        tool_ids = self.category_index.get(category, [])
        return [self.tools[tid] for tid in tool_ids]
    
    def search_tools(self, query: str, top_k: int = 5) -> List[Tuple[Tool, float]]:
        """Search for tools matching a query"""
        scores = {}
        
        # Score all tools based on capability matching
        for tool_id, tool in self.tools.items():
            max_score = 0.0
            for capability in tool.capabilities:
                score = capability.matches_query(query)
                max_score = max(max_score, score)
            
            if max_score > 0:
                # Adjust score based on tool reliability and success rate
                adjusted_score = max_score * tool.get_success_rate() * tool.reliability
                scores[tool_id] = adjusted_score
        
        # Sort by score
        sorted_tools = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k tools with scores
        results = [(self.tools[tool_id], score) for tool_id, score in sorted_tools[:top_k]]
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_tools": len(self.tools),
            "tools_by_category": {cat.value: len(tools) for cat, tools in self.category_index.items()},
            "total_executions": sum(t.usage_count for t in self.tools.values()),
            "average_success_rate": sum(t.get_success_rate() for t in self.tools.values()) / len(self.tools) if self.tools else 0
        }


class ToolSelector:
    """Selects appropriate tools for queries"""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.selection_history: List[ToolSelectionResult] = []
    
    def select_tools(self, query: str, max_tools: int = 3) -> ToolSelectionResult:
        """Select appropriate tools for a query"""
        print(f"\n🔧 Tool Selection for: {query}")
        
        # Search for matching tools
        candidates = self.registry.search_tools(query, top_k=max_tools * 2)
        
        if not candidates:
            print("   ⚠️  No matching tools found")
            return ToolSelectionResult(
                query=query,
                selected_tools=[],
                selection_scores={},
                selection_reasoning="No matching tools found"
            )
        
        # Apply selection strategy
        selected_tools, scores, reasoning = self._apply_selection_strategy(
            query, candidates, max_tools
        )
        
        # Create execution plan
        execution_plan = self._create_execution_plan(query, selected_tools)
        
        result = ToolSelectionResult(
            query=query,
            selected_tools=selected_tools,
            selection_scores=scores,
            selection_reasoning=reasoning,
            execution_plan=execution_plan
        )
        
        self.selection_history.append(result)
        
        print(f"   ✅ Selected {len(selected_tools)} tool(s):")
        for tool in selected_tools:
            score = scores.get(tool.id, 0.0)
            print(f"      • {tool.name} (score: {score:.2f}, category: {tool.category.value})")
        
        return result
    
    def _apply_selection_strategy(self, query: str, 
                                 candidates: List[Tuple[Tool, float]],
                                 max_tools: int) -> Tuple[List[Tool], Dict[str, float], str]:
        """Apply tool selection strategy"""
        selected_tools = []
        scores = {}
        reasoning_parts = []
        
        # Strategy 1: Select highest scoring tool
        if candidates:
            best_tool, best_score = candidates[0]
            selected_tools.append(best_tool)
            scores[best_tool.id] = best_score
            reasoning_parts.append(f"Primary tool: {best_tool.name} (score: {best_score:.2f})")
        
        # Strategy 2: Add complementary tools from different categories
        if len(selected_tools) < max_tools:
            used_categories = {t.category for t in selected_tools}
            
            for tool, score in candidates[1:]:
                if len(selected_tools) >= max_tools:
                    break
                
                # Prefer tools from different categories
                if tool.category not in used_categories:
                    selected_tools.append(tool)
                    scores[tool.id] = score
                    used_categories.add(tool.category)
                    reasoning_parts.append(f"Complementary tool: {tool.name} (category: {tool.category.value})")
        
        # Strategy 3: Add high-reliability backup if available
        if len(selected_tools) < max_tools:
            for tool, score in candidates:
                if tool not in selected_tools and tool.reliability > 0.9:
                    selected_tools.append(tool)
                    scores[tool.id] = score * 0.8  # Slightly lower score for backup
                    reasoning_parts.append(f"Backup tool: {tool.name} (reliability: {tool.reliability:.2f})")
                    break
        
        reasoning = "; ".join(reasoning_parts)
        
        return selected_tools, scores, reasoning
    
    def _create_execution_plan(self, query: str, tools: List[Tool]) -> List[str]:
        """Create an execution plan for selected tools"""
        plan = []
        
        if len(tools) == 1:
            plan.append(f"Execute {tools[0].name} with query")
        else:
            # Determine execution order based on tool dependencies and categories
            computation_tools = [t for t in tools if t.category == ToolCategory.COMPUTATION]
            info_tools = [t for t in tools if t.category == ToolCategory.INFORMATION]
            other_tools = [t for t in tools if t not in computation_tools and t not in info_tools]
            
            # Information gathering first
            for tool in info_tools:
                plan.append(f"1. Use {tool.name} to gather information")
            
            # Then computation
            for tool in computation_tools:
                plan.append(f"2. Use {tool.name} to process results")
            
            # Finally other tools
            for tool in other_tools:
                plan.append(f"3. Use {tool.name} as needed")
        
        return plan


class ToolExecutor:
    """Executes tools and handles errors"""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.execution_history: List[ToolExecutionResult] = []
        self.timeout_seconds = 30.0
    
    def execute_tool(self, tool_id: str, *args, **kwargs) -> ToolExecutionResult:
        """Execute a single tool"""
        tool = self.registry.get_tool(tool_id)
        
        if tool is None:
            return ToolExecutionResult(
                tool_id=tool_id,
                tool_name="Unknown",
                status=ToolStatus.NOT_FOUND,
                error_message=f"Tool {tool_id} not found in registry"
            )
        
        print(f"   ⚙️  Executing: {tool.name}")
        
        start_time = datetime.now()
        
        try:
            # Validate inputs (simplified)
            if not self._validate_inputs(tool, args, kwargs):
                result = ToolExecutionResult(
                    tool_id=tool_id,
                    tool_name=tool.name,
                    status=ToolStatus.INVALID_INPUT,
                    error_message="Invalid input parameters",
                    input_args=kwargs
                )
            else:
                # Execute tool
                output = tool.execute(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Check for timeout
                if execution_time > self.timeout_seconds:
                    result = ToolExecutionResult(
                        tool_id=tool_id,
                        tool_name=tool.name,
                        status=ToolStatus.TIMEOUT,
                        error_message=f"Execution exceeded timeout of {self.timeout_seconds}s",
                        execution_time=execution_time,
                        input_args=kwargs
                    )
                else:
                    result = ToolExecutionResult(
                        tool_id=tool_id,
                        tool_name=tool.name,
                        status=ToolStatus.SUCCESS,
                        output=output,
                        execution_time=execution_time,
                        input_args=kwargs
                    )
                    
                    print(f"      ✅ Success ({execution_time:.2f}s)")
                
                # Update tool statistics
                tool.update_stats(
                    success=(result.status == ToolStatus.SUCCESS),
                    execution_time=execution_time
                )
        
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            result = ToolExecutionResult(
                tool_id=tool_id,
                tool_name=tool.name,
                status=ToolStatus.FAILURE,
                error_message=str(e),
                execution_time=execution_time,
                input_args=kwargs
            )
            
            print(f"      ❌ Failed: {str(e)}")
            
            # Update tool statistics
            tool.update_stats(success=False, execution_time=execution_time)
        
        self.execution_history.append(result)
        
        return result
    
    def execute_plan(self, selection_result: ToolSelectionResult, 
                    *args, **kwargs) -> List[ToolExecutionResult]:
        """Execute a tool selection plan"""
        print(f"\n⚡ Executing Tool Plan")
        print(f"   Query: {selection_result.query}")
        print(f"   Tools: {len(selection_result.selected_tools)}")
        
        results = []
        
        for tool in selection_result.selected_tools:
            result = self.execute_tool(tool.id, *args, **kwargs)
            results.append(result)
            
            # If critical tool fails, might want to try fallback
            if result.status == ToolStatus.FAILURE and tool.category == ToolCategory.COMPUTATION:
                print(f"   ⚠️  Critical tool failed, attempting fallback...")
                # Could implement fallback logic here
        
        return results
    
    def _validate_inputs(self, tool: Tool, args: tuple, kwargs: dict) -> bool:
        """Validate tool inputs (simplified)"""
        # In practice, would check against tool's input_types
        return True
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"message": "No executions yet"}
        
        total = len(self.execution_history)
        successful = len([r for r in self.execution_history if r.status == ToolStatus.SUCCESS])
        failed = len([r for r in self.execution_history if r.status == ToolStatus.FAILURE])
        timeouts = len([r for r in self.execution_history if r.status == ToolStatus.TIMEOUT])
        
        avg_time = sum(r.execution_time for r in self.execution_history) / total
        
        return {
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "timeouts": timeouts,
            "success_rate": successful / total,
            "average_execution_time": avg_time
        }


class DynamicToolOrchestrator:
    """Orchestrates the complete tool selection and execution pipeline"""
    
    def __init__(self):
        self.registry = ToolRegistry()
        self.selector = ToolSelector(self.registry)
        self.executor = ToolExecutor(self.registry)
    
    def register_tool(self, tool: Tool):
        """Register a tool"""
        self.registry.register_tool(tool)
    
    def process_query(self, query: str, max_tools: int = 3, 
                     **execution_kwargs) -> Dict[str, Any]:
        """Complete pipeline: select and execute tools for a query"""
        print(f"\n{'='*80}")
        print(f"Processing Query: {query}")
        print(f"{'='*80}")
        
        # Step 1: Select tools
        selection_result = self.selector.select_tools(query, max_tools)
        
        if not selection_result.selected_tools:
            return {
                "query": query,
                "status": "no_tools_found",
                "message": "No suitable tools found for this query"
            }
        
        # Step 2: Execute selected tools
        execution_results = self.executor.execute_plan(selection_result, **execution_kwargs)
        
        # Step 3: Aggregate results
        final_output = self._aggregate_results(execution_results)
        
        return {
            "query": query,
            "status": "completed",
            "selection": {
                "tools_selected": [t.name for t in selection_result.selected_tools],
                "selection_reasoning": selection_result.selection_reasoning
            },
            "execution": {
                "results": [
                    {
                        "tool": r.tool_name,
                        "status": r.status.value,
                        "output": r.output,
                        "execution_time": r.execution_time
                    }
                    for r in execution_results
                ]
            },
            "final_output": final_output
        }
    
    def _aggregate_results(self, results: List[ToolExecutionResult]) -> Any:
        """Aggregate results from multiple tools"""
        successful_results = [r for r in results if r.status == ToolStatus.SUCCESS]
        
        if not successful_results:
            return "No successful tool executions"
        
        # Simple aggregation - in practice, would be more sophisticated
        if len(successful_results) == 1:
            return successful_results[0].output
        
        # Combine multiple results
        combined = {
            "primary_result": successful_results[0].output,
            "additional_results": [r.output for r in successful_results[1:]]
        }
        
        return combined
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            "registry": self.registry.get_statistics(),
            "execution": self.executor.get_execution_statistics(),
            "selection_history": len(self.selector.selection_history)
        }


# Example tool implementations

def calculator_tool(expression: str) -> float:
    """Simple calculator"""
    try:
        # Simplified - in practice would use safe eval
        result = eval(expression)
        return float(result)
    except:
        return 0.0


def search_tool(query: str) -> List[str]:
    """Simulated search"""
    results = [
        f"Search result 1 for: {query}",
        f"Search result 2 for: {query}",
        f"Search result 3 for: {query}"
    ]
    return results


def weather_tool(location: str) -> Dict[str, Any]:
    """Simulated weather lookup"""
    return {
        "location": location,
        "temperature": random.randint(50, 90),
        "condition": random.choice(["Sunny", "Cloudy", "Rainy"]),
        "humidity": random.randint(30, 90)
    }


def translator_tool(text: str, target_language: str = "es") -> str:
    """Simulated translation"""
    return f"[Translated to {target_language}]: {text}"


def data_analyzer_tool(data: List[float]) -> Dict[str, float]:
    """Analyze numerical data"""
    if not data:
        return {}
    
    return {
        "mean": sum(data) / len(data),
        "min": min(data),
        "max": max(data),
        "count": len(data)
    }


def code_generator_tool(description: str) -> str:
    """Generate code from description"""
    return f"# Generated code for: {description}\ndef generated_function():\n    pass"


def main():
    """Demonstration of dynamic tool selection pattern"""
    print("🔧 Dynamic Tool Selection Pattern Demonstration")
    print("=" * 80)
    print("This demonstrates dynamic tool selection and orchestration:")
    print("- Tool discovery and registration")
    print("- Capability-based matching")
    print("- Dynamic tool selection")
    print("- Multi-tool orchestration")
    print("- Error handling and fallbacks")
    
    # Create orchestrator
    orchestrator = DynamicToolOrchestrator()
    
    # Register tools
    print(f"\n{'='*80}")
    print("1. Registering Tools")
    print(f"{'='*80}")
    
    # Calculator tool
    orchestrator.register_tool(Tool(
        id="calc_001",
        name="Calculator",
        category=ToolCategory.COMPUTATION,
        capabilities=[
            ToolCapability(
                name="arithmetic",
                description="Perform arithmetic calculations",
                keywords={"calculate", "compute", "math", "add", "subtract", "multiply", "divide"},
                input_types=["string"],
                output_type="float",
                examples=["Calculate 5 + 3", "Compute 10 * 20"]
            )
        ],
        function=calculator_tool,
        cost=0.5,
        reliability=0.98
    ))
    
    # Search tool
    orchestrator.register_tool(Tool(
        id="search_001",
        name="Web Search",
        category=ToolCategory.INFORMATION,
        capabilities=[
            ToolCapability(
                name="search",
                description="Search for information on the web",
                keywords={"search", "find", "lookup", "query", "information"},
                input_types=["string"],
                output_type="list",
                examples=["Search for Python tutorials", "Find information about AI"]
            )
        ],
        function=search_tool,
        cost=1.0,
        reliability=0.95
    ))
    
    # Weather tool
    orchestrator.register_tool(Tool(
        id="weather_001",
        name="Weather Service",
        category=ToolCategory.INFORMATION,
        capabilities=[
            ToolCapability(
                name="weather",
                description="Get current weather for a location",
                keywords={"weather", "temperature", "forecast", "climate"},
                input_types=["string"],
                output_type="dict",
                examples=["Get weather for New York", "Check temperature in London"]
            )
        ],
        function=weather_tool,
        cost=1.5,
        reliability=0.92
    ))
    
    # Translator tool
    orchestrator.register_tool(Tool(
        id="translate_001",
        name="Translator",
        category=ToolCategory.COMMUNICATION,
        capabilities=[
            ToolCapability(
                name="translation",
                description="Translate text between languages",
                keywords={"translate", "language", "convert"},
                input_types=["string"],
                output_type="string",
                examples=["Translate hello to Spanish", "Convert text to French"]
            )
        ],
        function=translator_tool,
        cost=2.0,
        reliability=0.90
    ))
    
    # Data analyzer tool
    orchestrator.register_tool(Tool(
        id="analyzer_001",
        name="Data Analyzer",
        category=ToolCategory.ANALYSIS,
        capabilities=[
            ToolCapability(
                name="analysis",
                description="Analyze numerical data and compute statistics",
                keywords={"analyze", "statistics", "data", "compute", "mean", "average"},
                input_types=["list"],
                output_type="dict",
                examples=["Analyze data [1,2,3,4,5]", "Compute statistics"]
            )
        ],
        function=data_analyzer_tool,
        cost=1.0,
        reliability=0.96
    ))
    
    # Code generator tool
    orchestrator.register_tool(Tool(
        id="codegen_001",
        name="Code Generator",
        category=ToolCategory.GENERATION,
        capabilities=[
            ToolCapability(
                name="code_generation",
                description="Generate code from natural language descriptions",
                keywords={"generate", "code", "program", "function", "create"},
                input_types=["string"],
                output_type="string",
                examples=["Generate a function to sort a list", "Create code for fibonacci"]
            )
        ],
        function=code_generator_tool,
        cost=3.0,
        reliability=0.88
    ))
    
    # Test queries
    test_queries = [
        ("Calculate 15 * 23 + 100", {"expression": "15 * 23 + 100"}),
        ("Search for machine learning tutorials", {"query": "machine learning tutorials"}),
        ("What's the weather in San Francisco?", {"location": "San Francisco"}),
        ("Translate hello world to Spanish", {"text": "hello world", "target_language": "es"}),
        ("Analyze the data", {"data": [10.5, 20.3, 15.7, 30.2, 25.1]})
    ]
    
    for i, (query, kwargs) in enumerate(test_queries, 1):
        print(f"\n\n{'='*80}")
        print(f"Test Query {i}/{len(test_queries)}")
        print(f"{'='*80}")
        
        result = orchestrator.process_query(query, max_tools=2, **kwargs)
        
        print(f"\n📊 Result Summary:")
        print(f"   Status: {result['status']}")
        if result['status'] == 'completed':
            print(f"   Tools used: {', '.join(result['selection']['tools_selected'])}")
            print(f"   Reasoning: {result['selection']['selection_reasoning']}")
            print(f"   Final output: {result['final_output']}")
        
        input("\nPress Enter to continue to next query...")
    
    # Show comprehensive statistics
    print(f"\n\n{'='*80}")
    print("📈 System Statistics")
    print(f"{'='*80}")
    
    stats = orchestrator.get_comprehensive_statistics()
    
    print(f"\nRegistry Statistics:")
    for key, value in stats['registry'].items():
        print(f"  {key}: {value}")
    
    print(f"\nExecution Statistics:")
    for key, value in stats['execution'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n\n🎯 Key Tool Selection Features Demonstrated:")
    print("✅ Dynamic tool registration and discovery")
    print("✅ Capability-based matching")
    print("✅ Semantic query-to-tool matching")
    print("✅ Multi-tool selection strategies")
    print("✅ Execution orchestration")
    print("✅ Error handling and status tracking")
    print("✅ Tool performance monitoring")
    print("✅ Cost and reliability considerations")


if __name__ == "__main__":
    main()
