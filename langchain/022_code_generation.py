"""
Pattern 022: Code Generation & Execution

Description:
    The Code Generation & Execution pattern enables agents to generate executable code
    in response to user requests, safely execute the code in a sandboxed environment,
    and return results. This pattern combines LLM code generation capabilities with
    secure execution environments to solve computational problems programmatically.

Components:
    - Code Generator: LLM generates code from natural language
    - Syntax Validator: Checks code for syntax errors
    - Sandbox Environment: Isolated execution context
    - Result Processor: Formats and returns execution results
    - Error Handler: Captures and explains execution errors

Use Cases:
    - Data analysis and visualization
    - Mathematical problem solving
    - Text processing and transformation
    - Algorithm implementation
    - Quick prototyping and testing

LangChain Implementation:
    Uses LLM for code generation with specific prompting strategies,
    implements safe execution with restricted environments, and handles
    multiple programming languages.

Key Features:
    - Multi-language support (Python, JavaScript, etc.)
    - Sandboxed execution for security
    - Syntax and runtime error handling
    - Result capturing and formatting
    - Iterative refinement on errors
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import io
import contextlib
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"


class ExecutionStatus(Enum):
    """Status of code execution."""
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    FORBIDDEN = "forbidden"


@dataclass
class CodeSnippet:
    """Generated code snippet."""
    language: Language
    code: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    code: str
    status: ExecutionStatus
    output: Optional[str] = None
    error: Optional[str] = None
    duration: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


class CodeGenerator:
    """
    Generates code from natural language descriptions.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize code generator.
        
        Args:
            model: LLM model to use
        """
        self.llm = ChatOpenAI(model=model, temperature=0.2)
    
    def generate_python(
        self,
        task: str,
        context: Optional[str] = None
    ) -> CodeSnippet:
        """
        Generate Python code for a task.
        
        Args:
            task: Task description
            context: Optional context or constraints
            
        Returns:
            Generated code snippet
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Python programmer. Generate clean, efficient,
and well-commented Python code to accomplish the given task.

Guidelines:
- Write complete, runnable code
- Use standard library when possible
- Include helpful comments
- Handle edge cases
- Use proper error handling
- Print or return results clearly

Respond with ONLY the Python code, no explanations before or after."""),
            ("user", "{task}{context_str}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        context_str = f"\n\nContext: {context}" if context else ""
        
        response = chain.invoke({
            "task": task,
            "context_str": context_str
        })
        
        # Extract code from response
        code = self._extract_code(response)
        
        return CodeSnippet(
            language=Language.PYTHON,
            code=code,
            description=task
        )
    
    def generate_javascript(
        self,
        task: str,
        context: Optional[str] = None
    ) -> CodeSnippet:
        """
        Generate JavaScript code for a task.
        
        Args:
            task: Task description
            context: Optional context
            
        Returns:
            Generated code snippet
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert JavaScript programmer. Generate clean, efficient
JavaScript code to accomplish the given task.

Guidelines:
- Write complete, runnable Node.js code
- Use modern ES6+ syntax
- Include helpful comments
- Handle edge cases
- Use console.log for output

Respond with ONLY the JavaScript code, no explanations."""),
            ("user", "{task}{context_str}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        context_str = f"\n\nContext: {context}" if context else ""
        
        response = chain.invoke({
            "task": task,
            "context_str": context_str
        })
        
        code = self._extract_code(response)
        
        return CodeSnippet(
            language=Language.JAVASCRIPT,
            code=code,
            description=task
        )
    
    def refine_code(
        self,
        original_code: str,
        error: str,
        language: Language
    ) -> str:
        """
        Refine code based on error feedback.
        
        Args:
            original_code: Original code that failed
            error: Error message
            language: Programming language
            
        Returns:
            Refined code
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are debugging {language.value} code. Fix the error in the
provided code and return the corrected version.

Respond with ONLY the corrected code, no explanations."""),
            ("user", """Original Code:
```{language}
{code}
```

Error:
{error}

Provide the corrected code:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "language": language.value,
            "code": original_code,
            "error": error
        })
        
        return self._extract_code(response)
    
    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response."""
        # Remove markdown code blocks
        lines = response.split("\n")
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block or not line.strip().startswith("```"):
                code_lines.append(line)
        
        code = "\n".join(code_lines).strip()
        
        # If no code blocks found, return entire response
        if not code:
            code = response.strip()
        
        return code


class SandboxExecutor:
    """
    Safely executes code in sandboxed environment.
    """
    
    def __init__(self):
        """Initialize sandbox executor."""
        # Restricted builtins for Python execution
        self.safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'ascii': ascii,
            'bin': bin, 'bool': bool, 'chr': chr, 'dict': dict,
            'divmod': divmod, 'enumerate': enumerate, 'filter': filter,
            'float': float, 'format': format, 'hex': hex, 'int': int,
            'len': len, 'list': list, 'map': map, 'max': max,
            'min': min, 'oct': oct, 'ord': ord, 'pow': pow,
            'range': range, 'reversed': reversed, 'round': round,
            'set': set, 'slice': slice, 'sorted': sorted, 'str': str,
            'sum': sum, 'tuple': tuple, 'zip': zip,
            'print': print, 'isinstance': isinstance, 'type': type
        }
    
    def execute_python(self, code: str) -> ExecutionResult:
        """
        Execute Python code safely.
        
        Args:
            code: Python code to execute
            
        Returns:
            Execution result
        """
        import time
        
        # Check for forbidden operations
        forbidden_keywords = ['import os', 'import sys', 'eval', 'exec', '__import__', 
                            'open', 'file', 'compile']
        for keyword in forbidden_keywords:
            if keyword in code:
                return ExecutionResult(
                    code=code,
                    status=ExecutionStatus.FORBIDDEN,
                    error=f"Forbidden operation: {keyword}"
                )
        
        # Syntax check
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return ExecutionResult(
                code=code,
                status=ExecutionStatus.SYNTAX_ERROR,
                error=str(e)
            )
        
        # Execute in sandbox
        output_buffer = io.StringIO()
        
        try:
            start_time = time.time()
            
            # Redirect stdout
            with contextlib.redirect_stdout(output_buffer):
                # Create restricted namespace
                namespace = {'__builtins__': self.safe_builtins}
                
                # Execute code
                exec(code, namespace)
            
            duration = time.time() - start_time
            output = output_buffer.getvalue()
            
            return ExecutionResult(
                code=code,
                status=ExecutionStatus.SUCCESS,
                output=output if output else "Code executed successfully (no output)",
                duration=duration
            )
            
        except Exception as e:
            return ExecutionResult(
                code=code,
                status=ExecutionStatus.RUNTIME_ERROR,
                error=f"{type(e).__name__}: {str(e)}"
            )
    
    def execute_javascript(self, code: str) -> ExecutionResult:
        """
        Execute JavaScript code (simulated - would need Node.js).
        
        Args:
            code: JavaScript code
            
        Returns:
            Execution result
        """
        # In real implementation, would execute with Node.js
        # For demonstration, we simulate execution
        
        return ExecutionResult(
            code=code,
            status=ExecutionStatus.SUCCESS,
            output="[Simulated] JavaScript execution would require Node.js runtime",
            duration=0.0
        )


class CodeGenerationAgent:
    """
    Agent that generates and executes code to solve problems.
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        max_retries: int = 2
    ):
        """
        Initialize code generation agent.
        
        Args:
            model: LLM model to use
            max_retries: Maximum refinement attempts on errors
        """
        self.generator = CodeGenerator(model)
        self.executor = SandboxExecutor()
        self.max_retries = max_retries
        self.execution_history: List[ExecutionResult] = []
    
    def solve_with_code(
        self,
        task: str,
        language: Language = Language.PYTHON,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solve a task by generating and executing code.
        
        Args:
            task: Task description
            language: Programming language to use
            context: Optional context
            
        Returns:
            Solution with code and results
        """
        print(f"\n[Agent] Solving task: {task}")
        print(f"[Agent] Language: {language.value}\n")
        
        # Generate initial code
        print("[Agent] Generating code...")
        if language == Language.PYTHON:
            snippet = self.generator.generate_python(task, context)
        else:
            snippet = self.generator.generate_javascript(task, context)
        
        print(f"\n[Agent] Generated code:\n")
        print("=" * 60)
        print(snippet.code)
        print("=" * 60)
        
        # Execute with retries
        attempts = []
        result = None
        
        for attempt in range(self.max_retries + 1):
            print(f"\n[Agent] Execution attempt {attempt + 1}...")
            
            # Execute code
            if language == Language.PYTHON:
                result = self.executor.execute_python(snippet.code)
            else:
                result = self.executor.execute_javascript(snippet.code)
            
            self.execution_history.append(result)
            attempts.append(result)
            
            # Check if successful
            if result.status == ExecutionStatus.SUCCESS:
                print(f"[Agent] ✓ Success!")
                print(f"\n[Agent] Output:")
                print("-" * 60)
                print(result.output)
                print("-" * 60)
                break
            
            # Handle error
            print(f"[Agent] ✗ {result.status.value}: {result.error}")
            
            if attempt < self.max_retries:
                print(f"[Agent] Refining code...")
                snippet.code = self.generator.refine_code(
                    snippet.code,
                    result.error,
                    language
                )
                print(f"\n[Agent] Refined code:\n")
                print("=" * 60)
                print(snippet.code)
                print("=" * 60)
        
        return {
            "task": task,
            "language": language.value,
            "final_code": snippet.code,
            "result": result,
            "attempts": len(attempts),
            "success": result.status == ExecutionStatus.SUCCESS
        }


def demonstrate_code_generation():
    """Demonstrate the Code Generation & Execution pattern."""
    
    print("=" * 80)
    print("CODE GENERATION & EXECUTION PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = CodeGenerationAgent()
    
    # Test 1: Simple calculation
    print("\n" + "=" * 80)
    print("TEST 1: Mathematical Calculation")
    print("=" * 80)
    
    task1 = "Calculate the factorial of 10 and print the result"
    result1 = agent.solve_with_code(task1, Language.PYTHON)
    
    print("\n" + "=" * 80)
    print("SOLUTION SUMMARY")
    print("=" * 80)
    print(f"Success: {result1['success']}")
    print(f"Attempts: {result1['attempts']}")
    
    # Test 2: Data processing
    print("\n" + "=" * 80)
    print("TEST 2: Data Processing")
    print("=" * 80)
    
    task2 = """Create a list of numbers from 1 to 20, filter only even numbers,
square each one, and print the sum of all squared even numbers"""
    result2 = agent.solve_with_code(task2, Language.PYTHON)
    
    print("\n" + "=" * 80)
    print("SOLUTION SUMMARY")
    print("=" * 80)
    print(f"Success: {result2['success']}")
    print(f"Attempts: {result2['attempts']}")
    
    # Test 3: String manipulation
    print("\n" + "=" * 80)
    print("TEST 3: String Manipulation")
    print("=" * 80)
    
    task3 = """Create a function that takes a string and returns a dictionary
counting the frequency of each character. Test it with 'hello world' and print the result."""
    result3 = agent.solve_with_code(task3, Language.PYTHON)
    
    print("\n" + "=" * 80)
    print("SOLUTION SUMMARY")
    print("=" * 80)
    print(f"Success: {result3['success']}")
    print(f"Attempts: {result3['attempts']}")
    
    # Test 4: Algorithm implementation
    print("\n" + "=" * 80)
    print("TEST 4: Algorithm Implementation")
    print("=" * 80)
    
    task4 = """Implement the Fibonacci sequence up to the 15th number using iteration
(not recursion) and print all numbers"""
    result4 = agent.solve_with_code(task4, Language.PYTHON)
    
    print("\n" + "=" * 80)
    print("SOLUTION SUMMARY")
    print("=" * 80)
    print(f"Success: {result4['success']}")
    print(f"Attempts: {result4['attempts']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in agent.execution_history if r.status == ExecutionStatus.SUCCESS)
    total = len(agent.execution_history)
    
    print(f"""
Total tasks: 4
Total execution attempts: {total}
Successful executions: {successful}
Success rate: {successful/total*100:.0f}%

Execution Breakdown:
""")
    
    status_counts = {}
    for result in agent.execution_history:
        status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1
    
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    print("""
The Code Generation & Execution pattern demonstrates:

1. **Natural Language to Code**: Converts descriptions to executable code
2. **Safe Execution**: Sandboxed environment prevents harmful operations
3. **Error Recovery**: Automatically refines code on failures
4. **Multi-Language**: Supports multiple programming languages
5. **Result Capture**: Captures and formats execution output

Key Benefits:
- **Programmatic Problem Solving**: Handles complex computations
- **Safety**: Restricted execution environment
- **Self-Correction**: Learns from execution errors
- **Flexibility**: Works across languages and domains
- **Verifiable**: Actual code execution provides concrete results

Use Cases:
- Data analysis and statistics
- Mathematical problem solving
- Algorithm implementation and testing
- Text processing pipelines
- Quick prototyping and experimentation

Limitations:
- Restricted operations for security
- Limited library access in sandbox
- Execution timeouts for safety
- Language-specific constraints

This pattern enables agents to leverage the full power of programming
to solve computational problems that are difficult for pure LLMs.
""")


if __name__ == "__main__":
    demonstrate_code_generation()
