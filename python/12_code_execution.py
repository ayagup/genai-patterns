"""
Code Generation & Execution Pattern
Agent writes and executes code to solve problems
"""
from typing import Dict, Any, List
import sys
from io import StringIO
import traceback
import re
class CodeExecutionAgent:
    def __init__(self):
        self.execution_history: List[Dict[str, Any]] = []
        self.global_namespace: Dict[str, Any] = {}
    def generate_code(self, task: str) -> str:
        """Generate code for a task (simulates LLM code generation)"""
        print(f"\n--- Code Generation ---")
        print(f"Task: {task}\n")
        # Simple rule-based code generation for demonstration
        # In reality, this would use an LLM
        task_lower = task.lower()
        if "fibonacci" in task_lower:
            code = '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
result = [fibonacci(i) for i in range(10)]
print(f"First 10 Fibonacci numbers: {result}")
'''
        elif "prime" in task_lower:
            code = '''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
primes = [i for i in range(2, 50) if is_prime(i)]
print(f"Prime numbers up to 50: {primes}")
'''
        elif "sort" in task_lower or "data" in task_lower:
            code = '''data = [64, 34, 25, 12, 22, 11, 90]
sorted_data = sorted(data)
print(f"Original: {data}")
print(f"Sorted: {sorted_data}")
print(f"Min: {min(data)}, Max: {max(data)}, Mean: {sum(data)/len(data):.2f}")
'''
        elif "plot" in task_lower or "graph" in task_lower:
            code = '''# Simulating plot (actual plotting would require matplotlib)
x = list(range(10))
y = [i**2 for i in x]
print(f"X values: {x}")
print(f"Y values (x^2): {y}")
print("(Plot would be displayed here)")
'''
        else:
            code = f'''# Generated code for: {task}
result = "Task completed"
print(result)
'''
        print("Generated Code:")
        print("-" * 60)
        print(code)
        print("-" * 60)
        return code
    def execute_code(self, code: str, timeout: int = 5) -> Dict[str, Any]:
        """Execute code in a controlled environment"""
        print(f"\n--- Code Execution ---")
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        result = {
            "success": False,
            "output": "",
            "error": None,
            "variables": {}
        }
        try:
            # Create isolated namespace
            local_namespace = {}
            # Execute code
            exec(code, self.global_namespace, local_namespace)
            # Capture output
            output = captured_output.getvalue()
            # Extract result variables
            result_vars = {
                k: v for k, v in local_namespace.items()
                if not k.startswith('_')
            }
            result.update({
                "success": True,
                "output": output,
                "variables": result_vars
            })
            print("✓ Execution successful")
        except Exception as e:
            error_msg = traceback.format_exc()
            result.update({
                "success": False,
                "error": str(e),
                "traceback": error_msg
            })
            print(f"✗ Execution failed: {str(e)}")
        finally:
            sys.stdout = old_stdout
        return result
    def iterative_code_improvement(self, task: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Iteratively improve code based on execution results"""
        print(f"\n{'='*70}")
        print(f"Iterative Code Improvement")
        print(f"{'='*70}")
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'='*70}")
            print(f"Iteration {iteration}")
            print(f"{'='*70}")
            # Generate code
            if iteration == 1:
                code = self.generate_code(task)
            else:
                # In a real implementation, would use error feedback to improve
                print("Improving code based on previous feedback...")
                code = self.generate_code(task)
            # Execute code
            exec_result = self.execute_code(code)
            # Store in history
            self.execution_history.append({
                "iteration": iteration,
                "task": task,
                "code": code,
                "result": exec_result
            })
            # Display results
            if exec_result["success"]:
                print("\nOutput:")
                print(exec_result["output"])
                if exec_result["variables"]:
                    print("\nVariables:")
                    for var_name, var_value in exec_result["variables"].items():
                        print(f"  {var_name} = {var_value}")
                print(f"\n✓ Task completed successfully on iteration {iteration}")
                return exec_result
            else:
                print("\nError occurred:")
                print(exec_result["error"])
                if iteration < max_iterations:
                    print(f"\nRetrying with improved code...")
        print(f"\n✗ Failed to complete task after {max_iterations} iterations")
        return exec_result
    def explain_code(self, code: str) -> str:
        """Explain what the code does"""
        # Simple explanation based on code structure
        lines = code.strip().split('\n')
        explanation = "Code explanation:\n"
        for line in lines:
            line = line.strip()
            if line.startswith('def '):
                func_name = line.split('(')[0].replace('def ', '')
                explanation += f"- Defines function '{func_name}'\n"
            elif 'print' in line:
                explanation += f"- Outputs results\n"
            elif '=' in line and not line.startswith('#'):
                var_name = line.split('=')[0].strip()
                explanation += f"- Creates variable '{var_name}'\n"
        return explanation
# Usage
if __name__ == "__main__":
    agent = CodeExecutionAgent()
    # Test different tasks
    tasks = [
        "Generate first 10 Fibonacci numbers",
        "Find prime numbers up to 50",
        "Sort a list of numbers and show statistics"
    ]
    for task in tasks:
        result = agent.iterative_code_improvement(task, max_iterations=1)
        print("\n" + "="*80 + "\n")
    # Show execution history
    print("="*70)
    print("Execution History Summary")
    print("="*70)
    for i, record in enumerate(agent.execution_history, 1):
        status = "✓" if record["result"]["success"] else "✗"
        print(f"\n{i}. {status} {record['task']}")
        print(f"   Iteration: {record['iteration']}")
