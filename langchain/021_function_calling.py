"""
Pattern 021: Function Calling

Description:
    The Function Calling pattern enables LLMs to invoke structured functions with
    type-safe parameters. Unlike general tool use, function calling emphasizes strong
    typing, structured schemas, and direct integration with code functions. The LLM
    analyzes user intent, selects appropriate functions, extracts typed parameters,
    and invokes functions with validated inputs.

Components:
    - Function Schema: Structured definitions with parameter types
    - Parameter Validation: Type checking and constraint enforcement
    - Function Binding: Direct mapping to callable functions
    - Response Formatting: Structured outputs from function calls
    - Multi-turn Handling: Multiple function calls in sequence

Use Cases:
    - Database queries with typed parameters
    - API calls with strict schemas
    - System operations requiring validation
    - Structured data processing
    - Multi-step function orchestration

LangChain Implementation:
    Uses LangChain's function calling with OpenAI's function calling API,
    implements type-safe parameter extraction, and handles multi-turn conversations.

Key Features:
    - Type-safe parameter extraction
    - JSON schema validation
    - Multiple function selection
    - Error handling and retries
    - Conversation context preservation
"""

import os
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ParameterType(Enum):
    """Types of function parameters."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class FunctionParameter:
    """Definition of a function parameter."""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format."""
        schema = {
            "type": self.type.value,
            "description": self.description
        }
        if self.enum:
            schema["enum"] = self.enum
        return schema


@dataclass
class FunctionDefinition:
    """Complete function definition with schema."""
    name: str
    description: str
    parameters: List[FunctionParameter]
    function: Callable
    returns: str = "object"
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling schema."""
        required_params = [p.name for p in self.parameters if p.required]
        
        properties = {
            param.name: param.to_schema()
            for param in self.parameters
        }
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_params
            }
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate parameters against schema."""
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                return False, f"Missing required parameter: {param.name}"
        
        # Check types (basic validation)
        for param in self.parameters:
            if param.name in params:
                value = params[param.name]
                
                # Type checking
                if param.type == ParameterType.STRING and not isinstance(value, str):
                    return False, f"Parameter {param.name} must be string"
                elif param.type == ParameterType.INTEGER and not isinstance(value, int):
                    return False, f"Parameter {param.name} must be integer"
                elif param.type == ParameterType.NUMBER and not isinstance(value, (int, float)):
                    return False, f"Parameter {param.name} must be number"
                elif param.type == ParameterType.BOOLEAN and not isinstance(value, bool):
                    return False, f"Parameter {param.name} must be boolean"
                elif param.type == ParameterType.ARRAY and not isinstance(value, list):
                    return False, f"Parameter {param.name} must be array"
                
                # Enum validation
                if param.enum and value not in param.enum:
                    return False, f"Parameter {param.name} must be one of {param.enum}"
        
        return True, None


@dataclass
class FunctionCall:
    """Record of a function call."""
    function_name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


class FunctionRegistry:
    """
    Registry of callable functions with schemas.
    """
    
    def __init__(self):
        """Initialize function registry."""
        self.functions: Dict[str, FunctionDefinition] = {}
        self._register_builtin_functions()
    
    def _register_builtin_functions(self):
        """Register built-in functions."""
        
        # Get user info function
        self.register_function(FunctionDefinition(
            name="get_user_info",
            description="Retrieves information about a user by their ID",
            parameters=[
                FunctionParameter("user_id", ParameterType.INTEGER, "The user's ID"),
                FunctionParameter("fields", ParameterType.ARRAY, "Fields to retrieve", required=False)
            ],
            function=self._get_user_info
        ))
        
        # Create order function
        self.register_function(FunctionDefinition(
            name="create_order",
            description="Creates a new order in the system",
            parameters=[
                FunctionParameter("product_id", ParameterType.STRING, "Product identifier"),
                FunctionParameter("quantity", ParameterType.INTEGER, "Number of items"),
                FunctionParameter("priority", ParameterType.STRING, "Order priority", 
                                enum=["low", "normal", "high", "urgent"])
            ],
            function=self._create_order
        ))
        
        # Search database function
        self.register_function(FunctionDefinition(
            name="search_database",
            description="Searches the database with filters",
            parameters=[
                FunctionParameter("table", ParameterType.STRING, "Table name"),
                FunctionParameter("query", ParameterType.STRING, "Search query"),
                FunctionParameter("limit", ParameterType.INTEGER, "Max results", required=False, default=10)
            ],
            function=self._search_database
        ))
        
        # Send notification function
        self.register_function(FunctionDefinition(
            name="send_notification",
            description="Sends a notification to a user",
            parameters=[
                FunctionParameter("user_id", ParameterType.INTEGER, "Recipient user ID"),
                FunctionParameter("message", ParameterType.STRING, "Notification message"),
                FunctionParameter("urgent", ParameterType.BOOLEAN, "Is urgent", required=False, default=False)
            ],
            function=self._send_notification
        ))
        
        # Calculate price function
        self.register_function(FunctionDefinition(
            name="calculate_price",
            description="Calculates total price with tax and discount",
            parameters=[
                FunctionParameter("base_price", ParameterType.NUMBER, "Base price"),
                FunctionParameter("tax_rate", ParameterType.NUMBER, "Tax rate (0-1)", required=False, default=0.1),
                FunctionParameter("discount", ParameterType.NUMBER, "Discount amount", required=False, default=0)
            ],
            function=self._calculate_price
        ))
    
    def register_function(self, function_def: FunctionDefinition):
        """Register a new function."""
        self.functions[function_def.name] = function_def
    
    def get_function(self, name: str) -> Optional[FunctionDefinition]:
        """Get function by name."""
        return self.functions.get(name)
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """Get all function schemas in OpenAI format."""
        return [func.to_openai_schema() for func in self.functions.values()]
    
    # Built-in function implementations (mock)
    
    def _get_user_info(self, user_id: int, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """Mock user info retrieval."""
        users = {
            1: {"name": "Alice", "email": "alice@example.com", "age": 30, "status": "active"},
            2: {"name": "Bob", "email": "bob@example.com", "age": 25, "status": "active"},
            3: {"name": "Charlie", "email": "charlie@example.com", "age": 35, "status": "inactive"}
        }
        
        user = users.get(user_id, {"error": "User not found"})
        
        if fields and "error" not in user:
            user = {k: v for k, v in user.items() if k in fields}
        
        return user
    
    def _create_order(self, product_id: str, quantity: int, priority: str) -> Dict[str, Any]:
        """Mock order creation."""
        import random
        return {
            "order_id": random.randint(1000, 9999),
            "product_id": product_id,
            "quantity": quantity,
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
    
    def _search_database(self, table: str, query: str, limit: int = 10) -> Dict[str, Any]:
        """Mock database search."""
        mock_results = {
            "products": [
                {"id": "P001", "name": "Laptop", "price": 999.99},
                {"id": "P002", "name": "Mouse", "price": 29.99}
            ],
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"}
            ]
        }
        
        results = mock_results.get(table, [])
        return {
            "table": table,
            "query": query,
            "results": results[:limit],
            "total": len(results)
        }
    
    def _send_notification(self, user_id: int, message: str, urgent: bool = False) -> Dict[str, Any]:
        """Mock notification sending."""
        return {
            "notification_id": f"N{user_id}{datetime.now().microsecond}",
            "user_id": user_id,
            "message": message,
            "urgent": urgent,
            "sent": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_price(self, base_price: float, tax_rate: float = 0.1, discount: float = 0) -> Dict[str, Any]:
        """Calculate total price."""
        tax = base_price * tax_rate
        total = base_price + tax - discount
        
        return {
            "base_price": base_price,
            "tax": tax,
            "discount": discount,
            "total": total
        }


class FunctionCallingAgent:
    """
    Agent that uses function calling to accomplish tasks.
    """
    
    def __init__(
        self,
        function_registry: FunctionRegistry,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize function calling agent.
        
        Args:
            function_registry: Registry of available functions
            model: LLM model to use
        """
        self.function_registry = function_registry
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        self.call_history: List[FunctionCall] = []
    
    def select_and_extract(
        self,
        user_request: str,
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Select functions and extract parameters from user request.
        
        Args:
            user_request: User's request
            context: Optional context from previous turns
            
        Returns:
            List of function calls to make
        """
        # Get all function schemas
        schemas = self.function_registry.get_all_schemas()
        schemas_text = json.dumps(schemas, indent=2)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a function calling expert. Analyze user requests and
determine which functions to call with what parameters.

Available functions:
{schemas}

Respond with a JSON array of function calls. Each call should have:
- "function": function name
- "arguments": object with parameter values

If multiple functions are needed, include them all in the array.
If no function is needed, respond with an empty array []."""),
            ("user", "{request}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        full_request = user_request
        if context:
            full_request = f"Context: {context}\n\nRequest: {user_request}"
        
        response = chain.invoke({
            "schemas": schemas_text,
            "request": full_request
        })
        
        # Parse function calls
        try:
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                function_calls = json.loads(json_str)
                return function_calls if isinstance(function_calls, list) else []
            else:
                return []
        except Exception as e:
            print(f"Error parsing function calls: {e}")
            return []
    
    def execute_function(
        self,
        function_name: str,
        arguments: Dict[str, Any]
    ) -> FunctionCall:
        """
        Execute a function with given arguments.
        
        Args:
            function_name: Name of function to call
            arguments: Arguments for the function
            
        Returns:
            FunctionCall record
        """
        func_def = self.function_registry.get_function(function_name)
        
        if func_def is None:
            call = FunctionCall(
                function_name=function_name,
                arguments=arguments,
                result=None,
                success=False,
                error=f"Function '{function_name}' not found"
            )
            self.call_history.append(call)
            return call
        
        # Validate parameters
        valid, error = func_def.validate_parameters(arguments)
        if not valid:
            call = FunctionCall(
                function_name=function_name,
                arguments=arguments,
                result=None,
                success=False,
                error=error
            )
            self.call_history.append(call)
            return call
        
        # Execute function
        try:
            result = func_def.function(**arguments)
            call = FunctionCall(
                function_name=function_name,
                arguments=arguments,
                result=result,
                success=True
            )
        except Exception as e:
            call = FunctionCall(
                function_name=function_name,
                arguments=arguments,
                result=None,
                success=False,
                error=str(e)
            )
        
        self.call_history.append(call)
        return call
    
    def process_request(
        self,
        user_request: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user request with function calling.
        
        Args:
            user_request: User's request
            context: Optional context
            
        Returns:
            Results including function calls and response
        """
        print(f"\n[Agent] Processing request: {user_request}\n")
        
        # Select and extract function calls
        print("[Agent] Analyzing request and selecting functions...")
        function_calls = self.select_and_extract(user_request, context)
        
        if not function_calls:
            print("[Agent] No function calls needed\n")
            return {
                "request": user_request,
                "function_calls": [],
                "response": "No functions needed for this request"
            }
        
        print(f"[Agent] Planning {len(function_calls)} function call(s)\n")
        
        # Execute function calls
        results = []
        for fc in function_calls:
            function_name = fc.get("function", "")
            arguments = fc.get("arguments", {})
            
            print(f"[Agent] Calling: {function_name}")
            print(f"[Agent] Arguments: {json.dumps(arguments, indent=2)}")
            
            call = self.execute_function(function_name, arguments)
            results.append(call)
            
            if call.success:
                print(f"[Agent] Success: {json.dumps(call.result, indent=2)}\n")
            else:
                print(f"[Agent] Error: {call.error}\n")
        
        # Generate final response
        response = self._generate_response(user_request, results)
        
        return {
            "request": user_request,
            "function_calls": results,
            "response": response
        }
    
    def _generate_response(
        self,
        user_request: str,
        function_calls: List[FunctionCall]
    ) -> str:
        """Generate natural language response from function results."""
        results_text = "\n".join([
            f"Function: {call.function_name}\n"
            f"Result: {json.dumps(call.result, indent=2) if call.success else f'Error: {call.error}'}"
            for call in function_calls
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You generate natural responses from function call results."),
            ("user", """User Request: {request}

Function Results:
{results}

Provide a natural, helpful response incorporating these results.""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "request": user_request,
            "results": results_text
        })
        
        return response.strip()


def demonstrate_function_calling():
    """Demonstrate the Function Calling pattern."""
    
    print("=" * 80)
    print("FUNCTION CALLING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Initialize registry and agent
    registry = FunctionRegistry()
    agent = FunctionCallingAgent(registry)
    
    # Show available functions
    print("\n" + "=" * 80)
    print("AVAILABLE FUNCTIONS")
    print("=" * 80)
    schemas = registry.get_all_schemas()
    for schema in schemas:
        print(f"\n{schema['name']}:")
        print(f"  Description: {schema['description']}")
        print(f"  Parameters: {list(schema['parameters']['properties'].keys())}")
    
    # Test 1: Single function call
    print("\n" + "=" * 80)
    print("TEST 1: Single Function Call")
    print("=" * 80)
    
    request1 = "Get information about user ID 1"
    result1 = agent.process_request(request1)
    
    print("=" * 80)
    print("FINAL RESPONSE:")
    print("=" * 80)
    print(result1["response"])
    
    # Test 2: Multiple function calls
    print("\n" + "=" * 80)
    print("TEST 2: Multiple Function Calls")
    print("=" * 80)
    
    request2 = "Create an order for product P001 with quantity 5 and high priority, then send an urgent notification to user 1 about the order"
    result2 = agent.process_request(request2)
    
    print("=" * 80)
    print("FINAL RESPONSE:")
    print("=" * 80)
    print(result2["response"])
    
    # Test 3: Function with optional parameters
    print("\n" + "=" * 80)
    print("TEST 3: Optional Parameters")
    print("=" * 80)
    
    request3 = "Calculate the price for a $100 item with 15% tax and a $10 discount"
    result3 = agent.process_request(request3)
    
    print("=" * 80)
    print("FINAL RESPONSE:")
    print("=" * 80)
    print(result3["response"])
    
    # Test 4: Database search
    print("\n" + "=" * 80)
    print("TEST 4: Database Query")
    print("=" * 80)
    
    request4 = "Search the products table for laptops, limit to 5 results"
    result4 = agent.process_request(request4)
    
    print("=" * 80)
    print("FINAL RESPONSE:")
    print("=" * 80)
    print(result4["response"])
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
Total requests processed: 4
Total function calls made: {len(agent.call_history)}
Success rate: {sum(1 for call in agent.call_history if call.success) / len(agent.call_history) * 100:.0f}%

Function Call Breakdown:
""")
    
    for call in agent.call_history:
        status = "✓" if call.success else "✗"
        print(f"  {status} {call.function_name}({list(call.arguments.keys())})")
    
    print("""
The Function Calling pattern demonstrates:

1. **Type-Safe Invocation**: Strong parameter typing and validation
2. **Schema-Driven**: JSON schemas define function contracts
3. **Multi-Function**: Can orchestrate multiple function calls
4. **Parameter Extraction**: LLM extracts typed parameters from text
5. **Error Handling**: Validates inputs before execution

Key Benefits:
- **Type Safety**: Prevents invalid function calls
- **Clear Contracts**: Well-defined function interfaces
- **Composable**: Multiple functions work together
- **Reliable**: Validation before execution
- **Structured**: Predictable input/output formats

Use Cases:
- Database operations with typed queries
- API integrations with strict schemas
- System commands requiring validation
- Multi-step workflows with dependencies
- Enterprise integrations

This pattern is essential for production agentic systems that need
reliable, type-safe interactions with external systems and APIs.
""")


if __name__ == "__main__":
    demonstrate_function_calling()
