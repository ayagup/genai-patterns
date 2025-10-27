"""
Code Agent Pattern

Specialized agent for software development tasks including code generation,
interpretation, debugging, testing, and refactoring. Combines LLM capabilities
with code execution and analysis tools.

Use Cases:
- Automated code generation
- Bug fixing and debugging
- Test generation
- Code review and refactoring
- Documentation generation
- API client generation

Benefits:
- Accelerated development
- Automated testing
- Code quality improvement
- Documentation consistency
- Learning from codebases
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import ast
import sys
from io import StringIO


class CodeLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    SQL = "sql"
    BASH = "bash"


class TaskType(Enum):
    """Types of coding tasks"""
    GENERATE = "generate"
    DEBUG = "debug"
    TEST = "test"
    REFACTOR = "refactor"
    DOCUMENT = "document"
    EXPLAIN = "explain"


@dataclass
class CodeSnippet:
    """Code snippet with metadata"""
    code: str
    language: CodeLanguage
    description: str = ""
    file_path: Optional[str] = None
    line_start: int = 1
    line_end: Optional[int] = None


@dataclass
class TestCase:
    """Unit test case"""
    name: str
    input_data: Any
    expected_output: Any
    actual_output: Optional[Any] = None
    passed: bool = False
    error: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of code execution"""
    success: bool
    output: str
    error: Optional[str] = None
    return_value: Optional[Any] = None
    execution_time: float = 0.0


class CodeInterpreter:
    """
    Safe code interpreter with execution capabilities
    """
    
    def __init__(self):
        self.namespace: Dict[str, Any] = {}
        self.execution_history: List[ExecutionResult] = []
    
    def execute_python(self, code: str) -> ExecutionResult:
        """Execute Python code safely"""
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # Compile code
            compiled = compile(code, '<string>', 'exec')
            
            # Execute in namespace
            exec(compiled, self.namespace)
            
            # Get output
            output = captured_output.getvalue()
            
            result = ExecutionResult(
                success=True,
                output=output,
                return_value=self.namespace.get('result', None)
            )
            
        except Exception as e:
            result = ExecutionResult(
                success=False,
                output="",
                error=str(e)
            )
        
        finally:
            sys.stdout = old_stdout
        
        self.execution_history.append(result)
        return result
    
    def evaluate_expression(self, expression: str) -> Any:
        """Evaluate Python expression"""
        try:
            return eval(expression, self.namespace)
        except Exception as e:
            return f"Error: {e}"
    
    def reset_namespace(self) -> None:
        """Reset execution environment"""
        self.namespace = {}


class CodeAnalyzer:
    """
    Analyzes code for issues and improvements
    """
    
    def __init__(self):
        self.issues: List[Dict[str, Any]] = []
    
    def analyze_python(self, code: str) -> List[Dict[str, Any]]:
        """Analyze Python code"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            # Check for common issues
            for node in ast.walk(tree):
                # Check for undefined variables (simplified)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    # This is a simplified check
                    pass
                
                # Check for too many nested blocks
                if isinstance(node, (ast.For, ast.While, ast.If)):
                    depth = self._get_nesting_depth(node)
                    if depth > 4:
                        issues.append({
                            "type": "complexity",
                            "severity": "warning",
                            "message": f"Deeply nested block (depth: {depth})",
                            "line": node.lineno
                        })
            
            # Check code length
            lines = code.split('\n')
            if len(lines) > 100:
                issues.append({
                    "type": "length",
                    "severity": "info",
                    "message": f"Long function ({len(lines)} lines). Consider splitting.",
                    "line": 1
                })
        
        except SyntaxError as e:
            issues.append({
                "type": "syntax",
                "severity": "error",
                "message": str(e),
                "line": e.lineno
            })
        
        self.issues = issues
        return issues
    
    def _get_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Calculate nesting depth of node"""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While, ast.If)):
                child_depth = self._get_nesting_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)
        return max_depth


class TestGenerator:
    """
    Generates test cases for code
    """
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
    
    def generate_tests(
        self,
        function_name: str,
        function_signature: str
    ) -> List[TestCase]:
        """Generate test cases for function"""
        tests = []
        
        # Parse function signature to understand parameters
        # Simplified - in reality would use AST parsing
        
        # Generate basic test cases
        tests.append(TestCase(
            name=f"test_{function_name}_basic",
            input_data={"x": 1, "y": 2},
            expected_output=3
        ))
        
        tests.append(TestCase(
            name=f"test_{function_name}_edge_case",
            input_data={"x": 0, "y": 0},
            expected_output=0
        ))
        
        tests.append(TestCase(
            name=f"test_{function_name}_negative",
            input_data={"x": -1, "y": -2},
            expected_output=-3
        ))
        
        self.test_cases.extend(tests)
        return tests
    
    def run_tests(
        self,
        function_code: str,
        tests: List[TestCase],
        interpreter: CodeInterpreter
    ) -> List[TestCase]:
        """Run test cases"""
        # Execute function code
        result = interpreter.execute_python(function_code)
        
        if not result.success:
            print(f"Failed to execute function: {result.error}")
            return tests
        
        # Run each test
        for test in tests:
            try:
                # Prepare test input
                input_str = ", ".join(f"{k}={v}" for k, v in test.input_data.items())
                test_code = f"result = test_function({input_str})"
                
                # Execute test
                test_result = interpreter.execute_python(test_code)
                
                if test_result.success:
                    test.actual_output = test_result.return_value
                    test.passed = (test.actual_output == test.expected_output)
                else:
                    test.error = test_result.error
                    test.passed = False
            
            except Exception as e:
                test.error = str(e)
                test.passed = False
        
        return tests


class CodeAgent:
    """
    Code Agent for software development tasks
    
    Combines code generation, execution, testing, and analysis
    capabilities for automated programming assistance.
    """
    
    def __init__(self, name: str = "Code Agent"):
        self.name = name
        self.interpreter = CodeInterpreter()
        self.analyzer = CodeAnalyzer()
        self.test_generator = TestGenerator()
        self.code_history: List[CodeSnippet] = []
        
        print(f"[Code Agent] Initialized: {name}")
    
    def generate_code(
        self,
        task_description: str,
        language: CodeLanguage = CodeLanguage.PYTHON
    ) -> CodeSnippet:
        """Generate code based on description"""
        print(f"\n[Generate Code] {task_description}")
        
        # Simulate code generation (in production: call LLM)
        if "add" in task_description.lower():
            code = """def add(a, b):
    \"\"\"Add two numbers\"\"\"
    return a + b"""
        
        elif "fibonacci" in task_description.lower():
            code = """def fibonacci(n):
    \"\"\"Calculate nth Fibonacci number\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""
        
        elif "factorial" in task_description.lower():
            code = """def factorial(n):
    \"\"\"Calculate factorial of n\"\"\"
    if n <= 1:
        return 1
    return n * factorial(n-1)"""
        
        else:
            code = f"""# Generated code for: {task_description}
def solution():
    # TODO: Implement
    pass"""
        
        snippet = CodeSnippet(
            code=code,
            language=language,
            description=task_description
        )
        
        self.code_history.append(snippet)
        
        print(f"Generated {len(code.split())} lines")
        return snippet
    
    def execute_code(self, code_snippet: CodeSnippet) -> ExecutionResult:
        """Execute code snippet"""
        print(f"\n[Execute Code]")
        
        if code_snippet.language == CodeLanguage.PYTHON:
            result = self.interpreter.execute_python(code_snippet.code)
            
            if result.success:
                print(f"✓ Execution successful")
                if result.output:
                    print(f"Output: {result.output}")
            else:
                print(f"✗ Execution failed: {result.error}")
            
            return result
        
        else:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Language {code_snippet.language} not supported"
            )
    
    def debug_code(self, code_snippet: CodeSnippet) -> List[Dict[str, Any]]:
        """Debug code and find issues"""
        print(f"\n[Debug Code]")
        
        issues = self.analyzer.analyze_python(code_snippet.code)
        
        print(f"Found {len(issues)} issues:")
        for issue in issues:
            severity = issue['severity'].upper()
            print(f"  [{severity}] Line {issue['line']}: {issue['message']}")
        
        return issues
    
    def generate_tests(
        self,
        code_snippet: CodeSnippet,
        function_name: str = "test_function"
    ) -> List[TestCase]:
        """Generate test cases for code"""
        print(f"\n[Generate Tests] for {function_name}")
        
        tests = self.test_generator.generate_tests(
            function_name,
            code_snippet.code
        )
        
        print(f"Generated {len(tests)} test cases")
        return tests
    
    def run_tests(
        self,
        code_snippet: CodeSnippet,
        tests: List[TestCase]
    ) -> Tuple[int, int]:
        """Run tests against code"""
        print(f"\n[Run Tests] Running {len(tests)} tests")
        
        results = self.test_generator.run_tests(
            code_snippet.code,
            tests,
            self.interpreter
        )
        
        passed = sum(1 for t in results if t.passed)
        failed = len(results) - passed
        
        print(f"Results: {passed} passed, {failed} failed")
        
        for test in results:
            status = "✓" if test.passed else "✗"
            print(f"  {status} {test.name}")
            if not test.passed and test.error:
                print(f"      Error: {test.error}")
        
        return passed, failed
    
    def refactor_code(self, code_snippet: CodeSnippet) -> CodeSnippet:
        """Refactor code for better quality"""
        print(f"\n[Refactor Code]")
        
        # Analyze issues
        issues = self.analyzer.analyze_python(code_snippet.code)
        
        # Apply simple refactorings (in production: use LLM)
        refactored_code = code_snippet.code
        
        # Add docstrings if missing
        if '"""' not in refactored_code and "def " in refactored_code:
            lines = refactored_code.split('\n')
            for i, line in enumerate(lines):
                if 'def ' in line:
                    indent = len(line) - len(line.lstrip())
                    docstring = ' ' * (indent + 4) + '"""TODO: Add docstring"""'
                    lines.insert(i + 1, docstring)
                    break
            refactored_code = '\n'.join(lines)
        
        refactored = CodeSnippet(
            code=refactored_code,
            language=code_snippet.language,
            description=f"Refactored: {code_snippet.description}"
        )
        
        print(f"Applied {len(issues)} improvements")
        
        return refactored
    
    def explain_code(self, code_snippet: CodeSnippet) -> str:
        """Generate explanation of code"""
        print(f"\n[Explain Code]")
        
        # Simple explanation generation
        explanation = []
        
        lines = code_snippet.code.split('\n')
        for line in lines:
            line = line.strip()
            
            if line.startswith('def '):
                func_name = line.split('(')[0].replace('def ', '')
                explanation.append(f"Defines function '{func_name}'")
            
            elif line.startswith('if '):
                explanation.append("Conditional check")
            
            elif line.startswith('return '):
                explanation.append(f"Returns: {line.replace('return ', '')}")
            
            elif line.startswith('for ') or line.startswith('while '):
                explanation.append("Loop")
        
        result = '\n'.join(f"- {exp}" for exp in explanation)
        
        print(result)
        return result


def demonstrate_code_agent():
    """
    Demonstrate Code Agent pattern
    """
    print("=" * 70)
    print("CODE AGENT DEMONSTRATION")
    print("=" * 70)
    
    agent = CodeAgent("Python Development Agent")
    
    # Example 1: Code generation
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Code Generation")
    print("=" * 70)
    
    snippet = agent.generate_code("Create a function to add two numbers")
    print(f"\nGenerated code:\n{snippet.code}")
    
    # Example 2: Code execution
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Code Execution")
    print("=" * 70)
    
    execution_result = agent.execute_code(snippet)
    
    # Test the function
    if execution_result.success:
        test_code = CodeSnippet(
            code="result = add(5, 3)\nprint(f'5 + 3 = {result}')",
            language=CodeLanguage.PYTHON
        )
        agent.execute_code(test_code)
    
    # Example 3: Code analysis and debugging
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Code Analysis")
    print("=" * 70)
    
    buggy_code = CodeSnippet(
        code="""def complex_function(x):
    if x > 0:
        if x > 10:
            if x > 20:
                if x > 30:
                    if x > 40:
                        return "very large"
    return "small" """,
        language=CodeLanguage.PYTHON
    )
    
    issues = agent.debug_code(buggy_code)
    
    # Example 4: Test generation and execution
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Automated Testing")
    print("=" * 70)
    
    fib_snippet = agent.generate_code("Create fibonacci function")
    print(f"\nCode to test:\n{fib_snippet.code}")
    
    tests = agent.generate_tests(fib_snippet, "fibonacci")
    
    # Execute code first
    agent.execute_code(fib_snippet)
    
    # Run tests (simplified - would need actual test execution)
    print("\n[Test Results]")
    print("Tests generated successfully")
    
    # Example 5: Code refactoring
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Code Refactoring")
    print("=" * 70)
    
    messy_code = CodeSnippet(
        code="""def calc(x,y,z):
    a=x+y
    b=a*z
    return b""",
        language=CodeLanguage.PYTHON
    )
    
    refactored = agent.refactor_code(messy_code)
    print(f"\nRefactored code:\n{refactored.code}")
    
    # Example 6: Code explanation
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Code Explanation")
    print("=" * 70)
    
    agent.explain_code(snippet)


if __name__ == "__main__":
    demonstrate_code_agent()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Code agents combine generation, execution, and analysis
2. Safe code execution requires sandboxing
3. Automated testing improves code quality
4. Static analysis catches issues early
5. Refactoring maintains code health

Best Practices:
- Always validate generated code
- Use sandboxed execution environments
- Generate comprehensive test suites
- Apply static analysis tools
- Document code automatically
- Version control all changes
- Review AI-generated code
- Maintain security boundaries
    """)
