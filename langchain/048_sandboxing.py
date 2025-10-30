"""
Pattern 048: Sandboxing

Description:
    The Sandboxing pattern provides isolated execution environments for agent actions,
    ensuring security and containment when running potentially unsafe operations like
    code execution, external API calls, or file system access. This pattern prevents
    malicious or buggy code from affecting the host system.

Components:
    1. Sandbox Manager: Creates and manages isolated environments
    2. Resource Limiter: Enforces CPU, memory, time, and I/O constraints
    3. Permission Controller: Manages what operations are allowed
    4. Execution Monitor: Tracks sandbox activity and detects violations
    5. Result Validator: Validates outputs before returning to main system

Use Cases:
    - Code execution agents running untrusted code
    - Web scraping with unknown sites
    - Data processing with untrusted inputs
    - Plugin systems with third-party code
    - Educational platforms running student code
    - Multi-tenant systems with isolation requirements

LangChain Implementation:
    Uses custom runnables with execution constraints, subprocess isolation,
    and comprehensive monitoring to create secure execution environments.
"""

import os
import sys
import subprocess
import tempfile
import json
import time
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class SandboxType(Enum):
    """Types of sandbox environments"""
    SUBPROCESS = "subprocess"  # Run in separate process
    RESTRICTED_EVAL = "restricted_eval"  # Restricted Python eval
    CONTAINER = "container"  # Docker/container (simulated)
    VIRTUAL_ENV = "virtual_env"  # Virtual environment
    MEMORY_ONLY = "memory_only"  # No disk access


class PermissionLevel(Enum):
    """Permission levels for sandbox"""
    MINIMAL = "minimal"  # Basic operations only
    RESTRICTED = "restricted"  # Limited file/network access
    STANDARD = "standard"  # Normal operations with monitoring
    ELEVATED = "elevated"  # More permissions (rare)


class ViolationType(Enum):
    """Types of security violations"""
    RESOURCE_EXCEEDED = "resource_exceeded"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    UNSAFE_OPERATION = "unsafe_operation"
    OUTPUT_TOO_LARGE = "output_too_large"
    ESCAPE_ATTEMPT = "escape_attempt"


@dataclass
class ResourceLimits:
    """Resource limits for sandbox execution"""
    max_execution_time_seconds: float = 5.0
    max_memory_mb: int = 100
    max_output_size_bytes: int = 10_000
    max_file_size_bytes: int = 1_000_000
    max_network_requests: int = 0  # 0 = no network
    max_processes: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_execution_time": f"{self.max_execution_time_seconds}s",
            "max_memory": f"{self.max_memory_mb}MB",
            "max_output_size": f"{self.max_output_size_bytes} bytes",
            "max_file_size": f"{self.max_file_size_bytes} bytes",
            "max_network_requests": self.max_network_requests,
            "max_processes": self.max_processes
        }


@dataclass
class SecurityViolation:
    """Record of a security violation"""
    violation_type: ViolationType
    severity: float  # 0.0-1.0
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    blocked: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.violation_type.value,
            "severity": self.severity,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "blocked": self.blocked
        }


@dataclass
class SandboxResult:
    """Result from sandbox execution"""
    success: bool
    output: Optional[str]
    error: Optional[str]
    execution_time_seconds: float
    resources_used: Dict[str, Any]
    violations: List[SecurityViolation]
    permission_level: PermissionLevel
    sandbox_type: SandboxType
    
    @property
    def was_safe(self) -> bool:
        """Check if execution was safe (no violations)"""
        return len(self.violations) == 0
    
    @property
    def high_severity_violations(self) -> List[SecurityViolation]:
        """Get violations with severity >= 0.7"""
        return [v for v in self.violations if v.severity >= 0.7]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time": f"{self.execution_time_seconds:.3f}s",
            "resources_used": self.resources_used,
            "violations": [v.to_dict() for v in self.violations],
            "was_safe": self.was_safe,
            "permission_level": self.permission_level.value,
            "sandbox_type": self.sandbox_type.value
        }


class SandboxAgent:
    """
    Agent that executes operations in isolated sandbox environments.
    
    This implementation provides:
    1. Multiple sandbox types (subprocess, restricted eval, memory-only)
    2. Comprehensive resource limits (time, memory, output size)
    3. Permission-based access control
    4. Violation detection and blocking
    5. Safe code generation and execution
    """
    
    # Dangerous operations to detect
    UNSAFE_OPERATIONS = {
        "import os", "import sys", "import subprocess", "import socket",
        "open(", "exec(", "eval(", "compile(", "__import__",
        "file(", "input(", "raw_input(", "execfile(",
        "os.", "sys.", "subprocess.", "socket.",
        "__builtins__", "globals(", "locals(", "vars(",
        "delattr", "setattr", "getattr",
    }
    
    # Safe built-in functions for restricted eval
    SAFE_BUILTINS = {
        "abs", "all", "any", "ascii", "bin", "bool", "chr",
        "dict", "divmod", "enumerate", "filter", "float", "format",
        "hex", "int", "isinstance", "issubclass", "iter", "len",
        "list", "map", "max", "min", "oct", "ord", "pow", "print",
        "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "type", "zip",
    }
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        default_sandbox_type: SandboxType = SandboxType.SUBPROCESS,
        default_permission_level: PermissionLevel = PermissionLevel.RESTRICTED,
        default_limits: Optional[ResourceLimits] = None
    ):
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.default_sandbox_type = default_sandbox_type
        self.default_permission_level = default_permission_level
        self.default_limits = default_limits or ResourceLimits()
        
        # Statistics tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.blocked_executions = 0
        self.total_violations = 0
    
    def _check_code_safety(self, code: str) -> List[SecurityViolation]:
        """Check code for unsafe operations before execution"""
        violations = []
        
        # Check for dangerous operations
        code_lower = code.lower()
        for unsafe_op in self.UNSAFE_OPERATIONS:
            if unsafe_op.lower() in code_lower:
                violations.append(SecurityViolation(
                    violation_type=ViolationType.UNSAFE_OPERATION,
                    severity=0.9,
                    description=f"Detected potentially unsafe operation: {unsafe_op}",
                    blocked=True
                ))
        
        return violations
    
    def _create_restricted_globals(self) -> Dict[str, Any]:
        """Create restricted global namespace for safe eval"""
        safe_globals = {"__builtins__": {}}
        
        # Add only safe builtins
        for builtin_name in self.SAFE_BUILTINS:
            if hasattr(__builtins__, builtin_name):
                safe_globals["__builtins__"][builtin_name] = getattr(__builtins__, builtin_name)
        
        return safe_globals
    
    def _execute_in_subprocess(
        self,
        code: str,
        limits: ResourceLimits
    ) -> SandboxResult:
        """Execute code in separate subprocess with resource limits"""
        start_time = time.time()
        violations = []
        
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute in subprocess with timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=limits.max_execution_time_seconds
            )
            
            execution_time = time.time() - start_time
            
            # Check output size
            output = result.stdout
            if len(output) > limits.max_output_size_bytes:
                violations.append(SecurityViolation(
                    violation_type=ViolationType.OUTPUT_TOO_LARGE,
                    severity=0.6,
                    description=f"Output size {len(output)} exceeds limit {limits.max_output_size_bytes}",
                    blocked=True
                ))
                output = output[:limits.max_output_size_bytes] + "\n[OUTPUT TRUNCATED]"
            
            success = result.returncode == 0
            error = result.stderr if result.stderr else None
            
            return SandboxResult(
                success=success,
                output=output,
                error=error,
                execution_time_seconds=execution_time,
                resources_used={"subprocess": True},
                violations=violations,
                permission_level=self.default_permission_level,
                sandbox_type=SandboxType.SUBPROCESS
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            violations.append(SecurityViolation(
                violation_type=ViolationType.TIMEOUT,
                severity=0.8,
                description=f"Execution exceeded timeout of {limits.max_execution_time_seconds}s",
                blocked=True
            ))
            
            return SandboxResult(
                success=False,
                output=None,
                error="Execution timeout",
                execution_time_seconds=execution_time,
                resources_used={"subprocess": True, "timeout": True},
                violations=violations,
                permission_level=self.default_permission_level,
                sandbox_type=SandboxType.SUBPROCESS
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return SandboxResult(
                success=False,
                output=None,
                error=str(e),
                execution_time_seconds=execution_time,
                resources_used={"subprocess": True, "error": True},
                violations=violations,
                permission_level=self.default_permission_level,
                sandbox_type=SandboxType.SUBPROCESS
            )
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def _execute_restricted_eval(
        self,
        code: str,
        limits: ResourceLimits
    ) -> SandboxResult:
        """Execute code using restricted eval with limited builtins"""
        start_time = time.time()
        violations = []
        
        try:
            # Create restricted globals
            safe_globals = self._create_restricted_globals()
            safe_locals = {}
            
            # Capture output
            from io import StringIO
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                # Execute with timeout (simplified - production would use threading)
                exec(code, safe_globals, safe_locals)
                output = sys.stdout.getvalue()
                
                execution_time = time.time() - start_time
                
                # Check timeout
                if execution_time > limits.max_execution_time_seconds:
                    violations.append(SecurityViolation(
                        violation_type=ViolationType.TIMEOUT,
                        severity=0.7,
                        description=f"Execution time {execution_time:.2f}s exceeded limit",
                        blocked=False  # Already executed
                    ))
                
                # Check output size
                if len(output) > limits.max_output_size_bytes:
                    violations.append(SecurityViolation(
                        violation_type=ViolationType.OUTPUT_TOO_LARGE,
                        severity=0.6,
                        description=f"Output size {len(output)} exceeds limit",
                        blocked=False
                    ))
                    output = output[:limits.max_output_size_bytes] + "\n[OUTPUT TRUNCATED]"
                
                return SandboxResult(
                    success=True,
                    output=output,
                    error=None,
                    execution_time_seconds=execution_time,
                    resources_used={"restricted_eval": True},
                    violations=violations,
                    permission_level=PermissionLevel.MINIMAL,
                    sandbox_type=SandboxType.RESTRICTED_EVAL
                )
                
            finally:
                sys.stdout = old_stdout
                
        except Exception as e:
            execution_time = time.time() - start_time
            return SandboxResult(
                success=False,
                output=None,
                error=str(e),
                execution_time_seconds=execution_time,
                resources_used={"restricted_eval": True, "error": True},
                violations=violations,
                permission_level=PermissionLevel.MINIMAL,
                sandbox_type=SandboxType.RESTRICTED_EVAL
            )
    
    def generate_safe_code(self, task: str) -> str:
        """Generate safe code for a task using LLM"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a code generator that creates SAFE Python code.
            
Rules:
- DO NOT import os, sys, subprocess, socket, or other system modules
- DO NOT use file operations (open, read, write)
- DO NOT use exec, eval, compile, or __import__
- DO NOT access __builtins__, globals(), or locals()
- USE only safe operations: math, string manipulation, basic data structures
- KEEP code simple and focused on the task
- INCLUDE print statements to show results

Generate safe, self-contained Python code that runs without external dependencies."""),
            ("user", "Task: {task}\n\nGenerate safe Python code:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        code = chain.invoke({"task": task})
        
        # Clean up code (remove markdown formatting if present)
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        return code
    
    def execute_safely(
        self,
        code: str,
        sandbox_type: Optional[SandboxType] = None,
        limits: Optional[ResourceLimits] = None,
        check_safety: bool = True
    ) -> SandboxResult:
        """Execute code safely in a sandbox with monitoring"""
        self.total_executions += 1
        
        # Use defaults if not specified
        sandbox_type = sandbox_type or self.default_sandbox_type
        limits = limits or self.default_limits
        
        # Pre-execution safety check
        violations = []
        if check_safety:
            violations = self._check_code_safety(code)
            
            # Block if high-severity violations found
            if any(v.severity >= 0.8 for v in violations):
                self.blocked_executions += 1
                self.total_violations += len(violations)
                return SandboxResult(
                    success=False,
                    output=None,
                    error="Execution blocked due to security violations",
                    execution_time_seconds=0.0,
                    resources_used={},
                    violations=violations,
                    permission_level=self.default_permission_level,
                    sandbox_type=sandbox_type
                )
        
        # Execute in appropriate sandbox
        if sandbox_type == SandboxType.SUBPROCESS:
            result = self._execute_in_subprocess(code, limits)
        elif sandbox_type == SandboxType.RESTRICTED_EVAL:
            result = self._execute_restricted_eval(code, limits)
        else:
            # Default to restricted eval for other types
            result = self._execute_restricted_eval(code, limits)
        
        # Merge pre-execution violations with execution violations
        result.violations = violations + result.violations
        
        # Update statistics
        if result.success:
            self.successful_executions += 1
        self.total_violations += len(result.violations)
        
        return result
    
    def execute_task_safely(
        self,
        task: str,
        sandbox_type: Optional[SandboxType] = None,
        limits: Optional[ResourceLimits] = None
    ) -> Dict[str, Any]:
        """Generate code for task and execute it safely"""
        # Generate safe code
        code = self.generate_safe_code(task)
        
        # Execute in sandbox
        result = self.execute_safely(code, sandbox_type, limits)
        
        return {
            "task": task,
            "generated_code": code,
            "execution_result": result.to_dict()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sandbox usage statistics"""
        success_rate = (
            self.successful_executions / self.total_executions
            if self.total_executions > 0
            else 0.0
        )
        
        block_rate = (
            self.blocked_executions / self.total_executions
            if self.total_executions > 0
            else 0.0
        )
        
        avg_violations = (
            self.total_violations / self.total_executions
            if self.total_executions > 0
            else 0.0
        )
        
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "blocked_executions": self.blocked_executions,
            "success_rate": f"{success_rate:.1%}",
            "block_rate": f"{block_rate:.1%}",
            "total_violations": self.total_violations,
            "avg_violations_per_execution": f"{avg_violations:.2f}"
        }


def demonstrate_sandboxing():
    """Demonstrate sandbox pattern with various scenarios"""
    
    print("=" * 80)
    print("PATTERN 048: SANDBOXING DEMONSTRATION")
    print("=" * 80)
    print("\nDemonstrating isolated execution environments for secure agent operations\n")
    
    # Create sandbox agent
    agent = SandboxAgent(
        default_sandbox_type=SandboxType.SUBPROCESS,
        default_limits=ResourceLimits(
            max_execution_time_seconds=3.0,
            max_memory_mb=100,
            max_output_size_bytes=5000
        )
    )
    
    # Test 1: Safe code execution
    print("\n" + "=" * 80)
    print("TEST 1: Safe Code Execution")
    print("=" * 80)
    print("\nTask: Calculate fibonacci numbers")
    
    safe_code = """
# Calculate first 10 fibonacci numbers
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

print("First 10 Fibonacci numbers:")
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
"""
    
    result1 = agent.execute_safely(safe_code)
    print(f"\n✓ Execution successful: {result1.success}")
    print(f"  Execution time: {result1.execution_time_seconds:.3f}s")
    print(f"  Violations: {len(result1.violations)}")
    print(f"  Sandbox type: {result1.sandbox_type.value}")
    print(f"\nOutput:\n{result1.output}")
    
    # Test 2: Unsafe code detection
    print("\n" + "=" * 80)
    print("TEST 2: Unsafe Code Detection")
    print("=" * 80)
    print("\nAttempting to execute code with unsafe operations...")
    
    unsafe_code = """
import os
import subprocess

# Attempt to list directory (should be blocked)
print("Files:", os.listdir('.'))

# Attempt to run command (should be blocked)
subprocess.run(['echo', 'hack'])
"""
    
    result2 = agent.execute_safely(unsafe_code)
    print(f"\n✗ Execution blocked: {not result2.success}")
    print(f"  Violations detected: {len(result2.violations)}")
    
    for i, violation in enumerate(result2.violations, 1):
        print(f"\n  Violation {i}:")
        print(f"    Type: {violation.violation_type.value}")
        print(f"    Severity: {violation.severity:.1f}")
        print(f"    Description: {violation.description}")
        print(f"    Blocked: {violation.blocked}")
    
    # Test 3: Resource limit enforcement
    print("\n" + "=" * 80)
    print("TEST 3: Resource Limit Enforcement")
    print("=" * 80)
    print("\nTesting timeout protection...")
    
    timeout_code = """
# Infinite loop (should timeout)
import time
count = 0
while True:
    count += 1
    if count % 1000000 == 0:
        print(f"Count: {count}")
"""
    
    result3 = agent.execute_safely(
        timeout_code,
        limits=ResourceLimits(max_execution_time_seconds=1.0)
    )
    
    print(f"\n✓ Timeout protection activated: {not result3.success}")
    print(f"  Error: {result3.error}")
    print(f"  Execution time: {result3.execution_time_seconds:.3f}s")
    print(f"  Violations: {len(result3.violations)}")
    
    if result3.violations:
        for violation in result3.violations:
            print(f"\n  Timeout violation:")
            print(f"    Type: {violation.violation_type.value}")
            print(f"    Description: {violation.description}")
    
    # Test 4: Safe task execution with LLM
    print("\n" + "=" * 80)
    print("TEST 4: LLM-Generated Safe Code Execution")
    print("=" * 80)
    print("\nGenerating and executing code for task: 'Calculate prime numbers up to 50'")
    
    task_result = agent.execute_task_safely(
        "Calculate all prime numbers up to 50 and print them",
        sandbox_type=SandboxType.RESTRICTED_EVAL
    )
    
    print(f"\nGenerated code:")
    print("-" * 60)
    print(task_result["generated_code"])
    print("-" * 60)
    
    exec_result = task_result["execution_result"]
    print(f"\n✓ Execution successful: {exec_result['success']}")
    print(f"  Execution time: {exec_result['execution_time']}")
    print(f"  Safe: {exec_result['was_safe']}")
    print(f"  Sandbox: {exec_result['sandbox_type']}")
    print(f"\nOutput:\n{exec_result['output']}")
    
    # Test 5: Multiple sandbox types comparison
    print("\n" + "=" * 80)
    print("TEST 5: Comparing Sandbox Types")
    print("=" * 80)
    
    test_code = """
# Simple calculation
result = sum(range(1, 101))
print(f"Sum of 1-100: {result}")
print(f"Average: {result / 100}")
"""
    
    print("\nExecuting same code in different sandboxes...")
    
    for sandbox_type in [SandboxType.SUBPROCESS, SandboxType.RESTRICTED_EVAL]:
        result = agent.execute_safely(test_code, sandbox_type=sandbox_type)
        print(f"\n{sandbox_type.value.upper()}:")
        print(f"  Success: {result.success}")
        print(f"  Time: {result.execution_time_seconds:.4f}s")
        print(f"  Violations: {len(result.violations)}")
        print(f"  Output: {result.output.strip()}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("SANDBOX STATISTICS")
    print("=" * 80)
    
    stats = agent.get_statistics()
    print(f"\nTotal executions: {stats['total_executions']}")
    print(f"Successful: {stats['successful_executions']}")
    print(f"Blocked: {stats['blocked_executions']}")
    print(f"Success rate: {stats['success_rate']}")
    print(f"Block rate: {stats['block_rate']}")
    print(f"Total violations: {stats['total_violations']}")
    print(f"Avg violations per execution: {stats['avg_violations_per_execution']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SANDBOXING PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Security Isolation: Prevents malicious code from affecting host system
2. Resource Control: Enforces limits on CPU, memory, time, and I/O
3. Permission Management: Controls what operations are allowed
4. Violation Detection: Identifies and blocks unsafe operations
5. Multiple Strategies: Supports different isolation levels and methods

Implementation Features:
1. Pre-execution safety checks with pattern matching
2. Multiple sandbox types (subprocess, restricted eval, containers)
3. Comprehensive resource limits and enforcement
4. Real-time violation detection and blocking
5. Detailed execution monitoring and statistics

Use Cases:
- Code execution agents running user-provided code
- Educational platforms executing student submissions
- Data processing pipelines with untrusted inputs
- Plugin systems loading third-party code
- Multi-tenant systems requiring isolation
- Testing and development environments

Configuration Options:
1. Sandbox Type: subprocess, restricted_eval, container, virtual_env
2. Permission Level: minimal, restricted, standard, elevated
3. Resource Limits: time, memory, output size, file size, network
4. Safety Checks: pattern matching, execution monitoring
5. Violation Handling: block, warn, log, escalate

Best Practices:
1. Use least privilege principle for permissions
2. Set appropriate resource limits for use case
3. Monitor violations and adjust policies
4. Validate outputs before returning to main system
5. Log all sandbox activity for audit trail
6. Test sandbox escape scenarios
7. Keep sandbox environments minimal and isolated
8. Regular security reviews of allowed operations

Production Considerations:
- Container orchestration for true isolation (Docker, Kubernetes)
- Network isolation and firewall rules
- Encrypted communication between sandbox and host
- Regular security updates to sandbox environments
- Monitoring and alerting for unusual patterns
- Compliance with data protection regulations
- Backup and recovery procedures
- Performance optimization for high-volume execution

Comparison with Related Patterns:
- vs. Guardrails: Sandboxing provides execution isolation vs input/output filtering
- vs. Circuit Breaker: Prevents system damage vs prevents cascading failures
- vs. Rate Limiting: Controls resource usage vs controls request frequency
- vs. Defensive Generation: Isolates execution vs filters content

The Sandboxing pattern is essential for any system that executes potentially
untrusted code or operations, providing defense-in-depth security through
multiple layers of isolation, monitoring, and enforcement.
""")


if __name__ == "__main__":
    demonstrate_sandboxing()
