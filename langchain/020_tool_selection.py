"""
Pattern 020: Tool Selection & Use

Description:
    The Tool Selection & Use pattern enables agents to discover, select, and use external
    tools and APIs to extend their capabilities beyond the LLM's inherent knowledge. The
    agent analyzes the task, identifies appropriate tools, invokes them correctly, and
    processes the results.

Components:
    - Tool Registry: Catalog of available tools with descriptions
    - Selection Logic: Determines which tool(s) to use for a task
    - Parameter Extraction: Extracts required parameters from context
    - Tool Invocation: Executes tools with proper error handling
    - Result Integration: Incorporates tool outputs into responses

Use Cases:
    - Web search and information retrieval
    - Mathematical calculations
    - API integrations (weather, news, databases)
    - File operations and system commands
    - Data transformations

LangChain Implementation:
    Uses LangChain's Tool abstraction with structured tool definitions,
    implements tool selection logic, and integrates results into agent workflow.

Key Features:
    - Dynamic tool discovery
    - Context-aware tool selection
    - Robust error handling
    - Result formatting and integration
"""

import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ToolCategory(Enum):
    """Categories of tools available."""
    SEARCH = "search"
    COMPUTATION = "computation"
    DATA = "data"
    COMMUNICATION = "communication"
    UTILITY = "utility"


@dataclass
class ToolDefinition:
    """Definition of a tool available to the agent."""
    name: str
    category: ToolCategory
    description: str
    parameters: List[str]
    function: Callable
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM context."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "parameters": self.parameters
        }


@dataclass
class ToolInvocation:
    """Record of a tool invocation."""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    success: bool
    error: Optional[str] = None


class ToolRegistry:
    """
    Registry of available tools with discovery and selection capabilities.
    """
    
    def __init__(self):
        """Initialize tool registry with built-in tools."""
        self.tools: Dict[str, ToolDefinition] = {}
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register built-in tools."""
        # Calculator tool
        self.register_tool(ToolDefinition(
            name="calculator",
            category=ToolCategory.COMPUTATION,
            description="Performs mathematical calculations. Supports +, -, *, /, **, sqrt, etc.",
            parameters=["expression"],
            function=self._calculator
        ))
        
        # Date/Time tool
        self.register_tool(ToolDefinition(
            name="datetime",
            category=ToolCategory.UTILITY,
            description="Gets current date and time, or formats dates",
            parameters=["format"],
            function=self._datetime
        ))
        
        # String manipulation tool
        self.register_tool(ToolDefinition(
            name="string_transform",
            category=ToolCategory.DATA,
            description="Transforms strings: uppercase, lowercase, reverse, length, etc.",
            parameters=["text", "operation"],
            function=self._string_transform
        ))
        
        # Search simulator (mock)
        self.register_tool(ToolDefinition(
            name="web_search",
            category=ToolCategory.SEARCH,
            description="Searches the web for information (simulated)",
            parameters=["query"],
            function=self._web_search_mock
        ))
        
        # Data formatter
        self.register_tool(ToolDefinition(
            name="format_data",
            category=ToolCategory.DATA,
            description="Formats data as JSON, CSV, or table",
            parameters=["data", "format"],
            function=self._format_data
        ))
    
    def register_tool(self, tool: ToolDefinition):
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[ToolDefinition]:
        """Get all tools in a category."""
        return [tool for tool in self.tools.values() if tool.category == category]
    
    def search_tools(self, query: str) -> List[ToolDefinition]:
        """Search tools by query in name or description."""
        query_lower = query.lower()
        return [
            tool for tool in self.tools.values()
            if query_lower in tool.name.lower() or query_lower in tool.description.lower()
        ]
    
    def get_all_tools_description(self) -> str:
        """Get formatted description of all tools."""
        descriptions = []
        for tool in self.tools.values():
            params = ", ".join(tool.parameters)
            descriptions.append(
                f"- {tool.name} ({tool.category.value}): {tool.description}\n"
                f"  Parameters: {params}"
            )
        return "\n\n".join(descriptions)
    
    # Built-in tool implementations
    
    def _calculator(self, expression: str) -> str:
        """Calculate mathematical expression."""
        try:
            # Safe eval with limited scope
            allowed_names = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow, "sqrt": lambda x: x ** 0.5
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _datetime(self, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Get current date/time."""
        try:
            return datetime.now().strftime(format)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _string_transform(self, text: str, operation: str) -> str:
        """Transform string."""
        operations = {
            "uppercase": lambda s: s.upper(),
            "lowercase": lambda s: s.lower(),
            "reverse": lambda s: s[::-1],
            "length": lambda s: str(len(s)),
            "capitalize": lambda s: s.capitalize(),
            "title": lambda s: s.title()
        }
        
        if operation in operations:
            return operations[operation](text)
        else:
            return f"Unknown operation: {operation}"
    
    def _web_search_mock(self, query: str) -> str:
        """Mock web search (simulated)."""
        # In real implementation, this would use actual search API
        mock_results = {
            "weather": "Current weather: Sunny, 72°F",
            "news": "Latest news: Tech stocks rise, new AI breakthrough announced",
            "python": "Python is a high-level programming language known for readability",
            "default": f"Search results for '{query}': Multiple relevant articles found"
        }
        
        for key, result in mock_results.items():
            if key in query.lower():
                return result
        return mock_results["default"]
    
    def _format_data(self, data: str, format: str) -> str:
        """Format data in specified format."""
        try:
            if format == "json":
                # Try to parse and pretty-print
                parsed = json.loads(data)
                return json.dumps(parsed, indent=2)
            elif format == "uppercase":
                return data.upper()
            else:
                return data
        except Exception as e:
            return f"Error formatting: {str(e)}"


class ToolUsingAgent:
    """
    Agent that can discover, select, and use tools to accomplish tasks.
    """
    
    def __init__(
        self,
        tool_registry: ToolRegistry,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize tool-using agent.
        
        Args:
            tool_registry: Registry of available tools
            model: LLM model to use
        """
        self.tool_registry = tool_registry
        self.llm = ChatOpenAI(model=model, temperature=0.3)
        self.invocation_history: List[ToolInvocation] = []
    
    def select_tools(self, task: str) -> List[str]:
        """
        Select appropriate tools for a task.
        
        Args:
            task: Task description
            
        Returns:
            List of selected tool names
        """
        tools_description = self.tool_registry.get_all_tools_description()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a tool selection expert. Given a task and available tools,
select the most appropriate tool(s) to accomplish the task."""),
            ("user", """Task: {task}

Available Tools:
{tools_description}

Select the best tool(s) for this task. Respond with ONLY the tool name(s), one per line.
If no tool is needed, respond with "NONE".""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "task": task,
            "tools_description": tools_description
        })
        
        # Parse tool names from response
        lines = response.strip().split("\n")
        tool_names = [
            line.strip()
            for line in lines
            if line.strip() and line.strip() != "NONE"
        ]
        
        # Validate tool names
        valid_tools = [
            name for name in tool_names
            if self.tool_registry.get_tool(name) is not None
        ]
        
        return valid_tools
    
    def extract_parameters(
        self,
        task: str,
        tool: ToolDefinition
    ) -> Dict[str, Any]:
        """
        Extract tool parameters from task description.
        
        Args:
            task: Task description
            tool: Tool definition
            
        Returns:
            Dictionary of parameter values
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a parameter extraction expert. Extract the required
parameters for a tool from the task description."""),
            ("user", """Task: {task}

Tool: {tool_name}
Description: {tool_description}
Required Parameters: {parameters}

Extract the parameter values from the task. Respond ONLY with a JSON object
mapping parameter names to their values. If a parameter is not mentioned,
use a reasonable default or omit it.

Example: {{"expression": "2+2", "format": "json"}}""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "task": task,
            "tool_name": tool.name,
            "tool_description": tool.description,
            "parameters": ", ".join(tool.parameters)
        })
        
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parameters = json.loads(json_str)
                return parameters
            else:
                return {}
        except Exception as e:
            print(f"Error parsing parameters: {e}")
            return {}
    
    def invoke_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> ToolInvocation:
        """
        Invoke a tool with given parameters.
        
        Args:
            tool_name: Name of tool to invoke
            parameters: Parameters for the tool
            
        Returns:
            ToolInvocation record
        """
        tool = self.tool_registry.get_tool(tool_name)
        
        if tool is None:
            invocation = ToolInvocation(
                tool_name=tool_name,
                parameters=parameters,
                result=None,
                success=False,
                error=f"Tool '{tool_name}' not found"
            )
            self.invocation_history.append(invocation)
            return invocation
        
        try:
            # Invoke tool function with parameters
            result = tool.function(**parameters)
            
            invocation = ToolInvocation(
                tool_name=tool_name,
                parameters=parameters,
                result=result,
                success=True
            )
        except Exception as e:
            invocation = ToolInvocation(
                tool_name=tool_name,
                parameters=parameters,
                result=None,
                success=False,
                error=str(e)
            )
        
        self.invocation_history.append(invocation)
        return invocation
    
    def process_task(self, task: str) -> Dict[str, Any]:
        """
        Process a task using tool selection and invocation.
        
        Args:
            task: Task to process
            
        Returns:
            Results including tool invocations and final response
        """
        print(f"\n[Agent] Processing task: {task}\n")
        
        # Step 1: Select tools
        print("[Agent] Step 1: Selecting tools...")
        selected_tools = self.select_tools(task)
        
        if not selected_tools:
            print("[Agent] No tools needed for this task\n")
            return {
                "task": task,
                "tools_used": [],
                "invocations": [],
                "final_response": "Task does not require tool usage"
            }
        
        print(f"[Agent] Selected tools: {', '.join(selected_tools)}\n")
        
        # Step 2: Extract parameters and invoke tools
        print("[Agent] Step 2: Invoking tools...")
        invocations = []
        
        for tool_name in selected_tools:
            tool = self.tool_registry.get_tool(tool_name)
            
            # Extract parameters
            parameters = self.extract_parameters(task, tool)
            print(f"[Agent] {tool_name} parameters: {parameters}")
            
            # Invoke tool
            invocation = self.invoke_tool(tool_name, parameters)
            invocations.append(invocation)
            
            if invocation.success:
                print(f"[Agent] {tool_name} result: {invocation.result}\n")
            else:
                print(f"[Agent] {tool_name} error: {invocation.error}\n")
        
        # Step 3: Integrate results
        print("[Agent] Step 3: Integrating results...")
        final_response = self._integrate_results(task, invocations)
        
        return {
            "task": task,
            "tools_used": selected_tools,
            "invocations": invocations,
            "final_response": final_response
        }
    
    def _integrate_results(
        self,
        task: str,
        invocations: List[ToolInvocation]
    ) -> str:
        """Integrate tool results into final response."""
        results_text = "\n".join([
            f"- {inv.tool_name}: {inv.result if inv.success else f'Error: {inv.error}'}"
            for inv in invocations
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You integrate tool results into coherent responses."),
            ("user", """Task: {task}

Tool Results:
{results_text}

Provide a complete answer to the task incorporating these results naturally.""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "task": task,
            "results_text": results_text
        })
        
        return response.strip()


def demonstrate_tool_selection():
    """Demonstrate the Tool Selection & Use pattern."""
    
    print("=" * 80)
    print("TOOL SELECTION & USE PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Initialize registry and agent
    registry = ToolRegistry()
    agent = ToolUsingAgent(registry)
    
    # Show available tools
    print("\n" + "=" * 80)
    print("AVAILABLE TOOLS")
    print("=" * 80)
    print(registry.get_all_tools_description())
    
    # Test 1: Calculation task
    print("\n" + "=" * 80)
    print("TEST 1: Mathematical Calculation")
    print("=" * 80)
    
    task1 = "Calculate the result of 15 squared plus 25"
    result1 = agent.process_task(task1)
    
    print("=" * 80)
    print("FINAL RESPONSE:")
    print("=" * 80)
    print(result1["final_response"])
    
    # Test 2: Multiple tools
    print("\n" + "=" * 80)
    print("TEST 2: Multiple Tools")
    print("=" * 80)
    
    task2 = "What is the current date and time, and also calculate 100 divided by 4?"
    result2 = agent.process_task(task2)
    
    print("=" * 80)
    print("FINAL RESPONSE:")
    print("=" * 80)
    print(result2["final_response"])
    
    print("\n" + "-" * 80)
    print("TOOLS USED:")
    print("-" * 80)
    for inv in result2["invocations"]:
        print(f"  - {inv.tool_name}: {'Success' if inv.success else 'Failed'}")
    
    # Test 3: String manipulation
    print("\n" + "=" * 80)
    print("TEST 3: String Transformation")
    print("=" * 80)
    
    task3 = "Convert 'hello world' to uppercase and tell me its length"
    result3 = agent.process_task(task3)
    
    print("=" * 80)
    print("FINAL RESPONSE:")
    print("=" * 80)
    print(result3["final_response"])
    
    # Test 4: Search task
    print("\n" + "=" * 80)
    print("TEST 4: Web Search (Simulated)")
    print("=" * 80)
    
    task4 = "Search for information about Python programming"
    result4 = agent.process_task(task4)
    
    print("=" * 80)
    print("FINAL RESPONSE:")
    print("=" * 80)
    print(result4["final_response"])
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
Total tasks processed: 4
Total tool invocations: {len(agent.invocation_history)}
Success rate: {sum(1 for inv in agent.invocation_history if inv.success) / len(agent.invocation_history) * 100:.0f}%

The Tool Selection & Use pattern demonstrates:

1. **Tool Discovery**: Agent can identify available tools
2. **Intelligent Selection**: Chooses appropriate tools for tasks
3. **Parameter Extraction**: Extracts parameters from natural language
4. **Robust Invocation**: Handles tool execution with error handling
5. **Result Integration**: Combines tool outputs into coherent responses

Key Benefits:
- **Extended Capabilities**: Goes beyond LLM knowledge
- **Task-Appropriate**: Selects right tool for each job
- **Error Resilient**: Graceful handling of failures
- **Composable**: Can use multiple tools together

Use Cases:
- API integrations (weather, news, databases)
- Mathematical computations
- Web searches and data retrieval
- File operations
- Data transformations

This pattern is the foundation for agentic AI systems that can interact
with the real world through tools and APIs.
""")


if __name__ == "__main__":
    demonstrate_tool_selection()
