"""
Function Calling Pattern
Structured way for LLM to call predefined functions
"""
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import json
import inspect
@dataclass
class FunctionDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
class FunctionRegistry:
    """Registry of available functions"""
    def __init__(self):
        self.functions: Dict[str, FunctionDefinition] = {}
    def register(self, func: Callable, description: str, parameters: Dict[str, Any]):
        """Register a function"""
        func_def = FunctionDefinition(
            name=func.__name__,
            description=description,
            parameters=parameters,
            function=func
        )
        self.functions[func.__name__] = func_def
        print(f"Registered function: {func.__name__}")
    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-style function schemas"""
        schemas = []
        for func_def in self.functions.values():
            schema = {
                "name": func_def.name,
                "description": func_def.description,
                "parameters": func_def.parameters
            }
            schemas.append(schema)
        return schemas
    def call_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a function by name"""
        if name not in self.functions:
            raise ValueError(f"Function {name} not found")
        func_def = self.functions[name]
        return func_def.function(**arguments)
class FunctionCallingAgent:
    def __init__(self, registry: FunctionRegistry):
        self.registry = registry
        self.call_history: List[Dict[str, Any]] = []
    def decide_function_call(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Decide which function to call (simulates LLM decision)"""
        # In reality, this would be done by an LLM
        # Here we use simple keyword matching for demonstration
        user_lower = user_input.lower()
        if "weather" in user_lower:
            # Extract city
            words = user_input.split()
            city = "New York"  # Default
            for i, word in enumerate(words):
                if word.lower() in ["in", "at", "for"] and i + 1 < len(words):
                    city = words[i + 1].strip("?.,!")
                    break
            return {
                "name": "get_weather",
                "arguments": {"city": city}
            }
        elif "calculate" in user_lower or "+" in user_input or "*" in user_input:
            # Extract expression
            import re
            match = re.search(r'(\d+\s*[\+\-\*\/]\s*\d+)', user_input)
            if match:
                return {
                    "name": "calculate",
                    "arguments": {"expression": match.group(1)}
                }
        elif "search" in user_lower or "find" in user_lower:
            # Extract query
            query_start = user_input.lower().find("search for ")
            if query_start != -1:
                query = user_input[query_start + 11:].strip("?.,!")
            else:
                query = user_input
            return {
                "name": "web_search",
                "arguments": {"query": query}
            }
        elif "email" in user_lower or "send" in user_lower:
            return {
                "name": "send_email",
                "arguments": {
                    "to": "user@example.com",
                    "subject": "Message",
                    "body": user_input
                }
            }
        return None
    def process_request(self, user_input: str) -> str:
        """Process user request with function calling"""
        print(f"\n{'='*70}")
        print(f"User Request: {user_input}")
        print(f"{'='*70}\n")
        # Decide if function call is needed
        function_call = self.decide_function_call(user_input)
        if function_call is None:
            response = "I don't have a specific function for that request."
            print(f"Response: {response}")
            return response
        # Show function call decision
        print(f"Function Call Decision:")
        print(f"  Function: {function_call['name']}")
        print(f"  Arguments: {json.dumps(function_call['arguments'], indent=4)}\n")
        # Execute function
        try:
            print(f"Executing function...")
            result = self.registry.call_function(
                function_call['name'],
                function_call['arguments']
            )
            # Log the call
            self.call_history.append({
                "user_input": user_input,
                "function": function_call['name'],
                "arguments": function_call['arguments'],
                "result": result
            })
            print(f"Function Result: {result}\n")
            # Generate natural language response
            response = self.generate_response(user_input, function_call, result)
            print(f"Final Response: {response}")
            return response
        except Exception as e:
            error_msg = f"Error executing function: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    def generate_response(self, user_input: str, function_call: Dict, result: Any) -> str:
        """Generate natural language response from function result"""
        func_name = function_call['name']
        if func_name == "get_weather":
            city = function_call['arguments']['city']
            return f"The weather in {city} is {result['condition']} with a temperature of {result['temperature']}°F."
        elif func_name == "calculate":
            expr = function_call['arguments']['expression']
            return f"The result of {expr} is {result}."
        elif func_name == "web_search":
            return f"I found {len(result)} results for your search: {result[0] if result else 'No results'}"
        elif func_name == "send_email":
            return f"Email sent successfully to {function_call['arguments']['to']}."
        return f"Function {func_name} completed with result: {result}"
# Define actual functions to be called
def get_weather(city: str) -> Dict[str, Any]:
    """Get weather for a city"""
    # Simulated weather data
    weather_db = {
        "New York": {"temperature": 72, "condition": "Sunny"},
        "London": {"temperature": 65, "condition": "Cloudy"},
        "Tokyo": {"temperature": 78, "condition": "Rainy"},
    }
    return weather_db.get(city, {"temperature": 70, "condition": "Unknown"})
def calculate(expression: str) -> float:
    """Calculate a mathematical expression"""
    try:
        # Safe evaluation (in production, use a proper math parser)
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except:
        return "Error in calculation"
def web_search(query: str) -> List[str]:
    """Perform web search"""
    # Simulated search results
    return [
        f"Result 1 for '{query}'",
        f"Result 2 for '{query}'",
        f"Result 3 for '{query}'"
    ]
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email"""
    # Simulated email sending
    return f"Email sent to {to}"
# Usage
if __name__ == "__main__":
    # Create registry and register functions
    registry = FunctionRegistry()
    registry.register(
        func=get_weather,
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                }
            },
            "required": ["city"]
        }
    )
    registry.register(
        func=calculate,
        description="Calculate a mathematical expression",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    )
    registry.register(
        func=web_search,
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    )
    registry.register(
        func=send_email,
        description="Send an email",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"}
            },
            "required": ["to", "subject", "body"]
        }
    )
    # Create agent
    agent = FunctionCallingAgent(registry)
    # Test various requests
    requests = [
        "What's the weather in London?",
        "Calculate 15 + 27",
        "Search for artificial intelligence",
        "Send an email reminder"
    ]
    for request in requests:
        agent.process_request(request)
        print("\n" + "="*80 + "\n")
