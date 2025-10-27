"""
Sandboxing Pattern
Executes agent actions in isolated environment
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import subprocess
import tempfile
import os
import json
class SandboxType(Enum):
    PROCESS = "process"
    DOCKER = "docker"
    VM = "virtual_machine"
    RESTRICTED_PYTHON = "restricted_python"
@dataclass
class SandboxConfig:
    """Configuration for sandbox"""
    sandbox_type: SandboxType
    memory_limit_mb: int = 512
    cpu_limit_percent: int = 50
    timeout_seconds: int = 5
    network_enabled: bool = False
    allowed_modules: List[str] = None
    def __post_init__(self):
        if self.allowed_modules is None:
            self.allowed_modules = ['math', 'json', 'datetime']
@dataclass
class SandboxResult:
    """Result from sandbox execution"""
    success: bool
    output: Any
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0
class Sandbox:
    """Isolated execution environment"""
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.execution_count = 0
    def execute(self, code: str, context: Dict[str, Any] = None) -> SandboxResult:
        """Execute code in sandbox"""
        self.execution_count += 1
        print(f"\n{'='*60}")
        print(f"SANDBOX EXECUTION #{self.execution_count}")
        print(f"{'='*60}")
        print(f"Type: {self.config.sandbox_type.value}")
        print(f"Memory Limit: {self.config.memory_limit_mb}MB")
        print(f"Timeout: {self.config.timeout_seconds}s")
        print(f"\nCode to execute:")
        print("-" * 60)
        print(code)
        print("-" * 60)
        if self.config.sandbox_type == SandboxType.RESTRICTED_PYTHON:
            return self._execute_restricted_python(code, context)
        elif self.config.sandbox_type == SandboxType.PROCESS:
            return self._execute_in_process(code, context)
        else:
            return SandboxResult(
                success=False,
                output=None,
                error=f"Sandbox type {self.config.sandbox_type.value} not implemented"
            )
    def _execute_restricted_python(self, code: str, context: Dict[str, Any]) -> SandboxResult:
        """Execute Python code with restrictions"""
        import time
        start_time = time.time()
        # Create restricted globals
        restricted_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'sum': sum,
                'min': min,
                'max': max,
            }
        }
        # Add allowed modules
        for module_name in self.config.allowed_modules:
            try:
                restricted_globals[module_name] = __import__(module_name)
            except ImportError:
                pass
        # Add context
        if context:
            restricted_globals.update(context)
        # Capture output
        import io
        import sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            # Execute with timeout (simplified)
            local_vars = {}
            exec(code, restricted_globals, local_vars)
            execution_time = (time.time() - start_time) * 1000
            # Get output
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            # Get result
            result = local_vars.get('result', stdout or "Execution completed")
            print(f"\n✓ Execution successful ({execution_time:.0f}ms)")
            if stdout:
                print(f"Output: {stdout}")
            return SandboxResult(
                success=True,
                output=result,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=execution_time
            )
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            print(f"\n✗ Execution failed: {str(e)}")
            return SandboxResult(
                success=False,
                output=None,
                error=str(e),
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time_ms=execution_time
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    def _execute_in_process(self, code: str, context: Dict[str, Any]) -> SandboxResult:
        """Execute in separate process"""
        import time
        start_time = time.time()
        # Create temporary file with code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        try:
            # Execute in subprocess with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            execution_time = (time.time() - start_time) * 1000
            success = result.returncode == 0
            if success:
                print(f"\n✓ Process execution successful ({execution_time:.0f}ms)")
            else:
                print(f"\n✗ Process execution failed (exit code: {result.returncode})")
            return SandboxResult(
                success=success,
                output=result.stdout,
                error=result.stderr if not success else None,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time_ms=execution_time
            )
        except subprocess.TimeoutExpired:
            print(f"\n✗ Execution timeout after {self.config.timeout_seconds}s")
            return SandboxResult(
                success=False,
                output=None,
                error=f"Timeout after {self.config.timeout_seconds}s"
            )
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
class SandboxedAgent:
    """Agent that executes code in sandbox"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.sandbox = Sandbox(SandboxConfig(
            sandbox_type=SandboxType.RESTRICTED_PYTHON,
            memory_limit_mb=256,
            timeout_seconds=3,
            allowed_modules=['math', 'json', 'datetime', 'random']
        ))
        self.execution_history: List[SandboxResult] = []
    def execute_user_code(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute user-provided code safely"""
        print(f"\n{'='*70}")
        print(f"SANDBOXED CODE EXECUTION")
        print(f"{'='*70}")
        # Validate code first
        if not self._validate_code(code):
            return {
                'success': False,
                'error': 'Code validation failed: contains prohibited operations'
            }
        # Execute in sandbox
        result = self.sandbox.execute(code, context)
        self.execution_history.append(result)
        return {
            'success': result.success,
            'output': result.output,
            'error': result.error,
            'execution_time_ms': result.execution_time_ms
        }
    def _validate_code(self, code: str) -> bool:
        """Validate code for dangerous operations"""
        prohibited = [
            'import os',
            'import sys',
            'import subprocess',
            '__import__',
            'eval',
            'exec',
            'open',
            'file',
            'input',
            'raw_input',
        ]
        code_lower = code.lower()
        for prohibited_item in prohibited:
            if prohibited_item in code_lower:
                print(f"⚠️  Validation failed: contains '{prohibited_item}'")
                return False
        return True
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {'total_executions': 0}
        total = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        avg_time = sum(r.execution_time_ms for r in self.execution_history) / total
        return {
            'total_executions': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total,
            'avg_execution_time_ms': avg_time
        }
# Usage
if __name__ == "__main__":
    print("="*80)
    print("SANDBOXING PATTERN DEMONSTRATION")
    print("="*80)
    agent = SandboxedAgent("sandbox-agent-001")
    # Test Case 1: Safe code
    print("\n" + "="*80)
    print("TEST 1: Safe Mathematical Calculation")
    print("="*80)
    safe_code = """
import math
def calculate():
    result = math.sqrt(16) + math.pi
    print(f"Result: {result}")
    return result
result = calculate()
"""
    result1 = agent.execute_user_code(safe_code)
    print(f"\nResult: {result1}")
    # Test Case 2: Prohibited code
    print("\n\n" + "="*80)
    print("TEST 2: Dangerous Code (should be blocked)")
    print("="*80)
    dangerous_code = """
import os
os.system('rm -rf /')  # Dangerous!
"""
    result2 = agent.execute_user_code(dangerous_code)
    print(f"\nResult: {result2}")
    # Test Case 3: Code with context
    print("\n\n" + "="*80)
    print("TEST 3: Code with Context Variables")
    print("="*80)
    context_code = """
# Use provided context
total = sum(numbers)
average = total / len(numbers)
result = {'total': total, 'average': average}
print(f"Numbers: {numbers}")
print(f"Total: {total}, Average: {average}")
"""
    result3 = agent.execute_user_code(
        context_code,
        context={'numbers': [1, 2, 3, 4, 5]}
    )
    print(f"\nResult: {result3}")
    # Test Case 4: Timeout
    print("\n\n" + "="*80)
    print("TEST 4: Infinite Loop (should timeout)")
    print("="*80)
    timeout_code = """
while True:
    pass
"""
    result4 = agent.execute_user_code(timeout_code)
    print(f"\nResult: {result4}")
    # Statistics
    stats = agent.get_statistics()
    print(f"\n{'='*70}")
    print("EXECUTION STATISTICS")
    print(f"{'='*70}")
    print(f"Total Executions: {stats['total_executions']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Avg Execution Time: {stats['avg_execution_time_ms']:.0f}ms")
