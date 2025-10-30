"""
Self-Healing Agent System Pattern

Automatically detects, diagnoses, and recovers from failures.
Implements monitoring, fault detection, and recovery strategies.

Use Cases:
- Production systems
- Autonomous operations
- Fault-tolerant applications
- High-availability services

Advantages:
- Automatic error recovery
- Reduced downtime
- Self-diagnosis capabilities
- Learning from failures
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import traceback


class HealthStatus(Enum):
    """Health status of system components"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class FailureType(Enum):
    """Types of failures"""
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    INVALID_INPUT = "invalid_input"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Recovery actions"""
    RETRY = "retry"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    RESTART = "restart"
    SCALE_UP = "scale_up"
    DEGRADE_GRACEFULLY = "degrade_gracefully"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class HealthMetrics:
    """Health metrics for a component"""
    component_id: str
    status: HealthStatus
    uptime_seconds: float
    request_count: int
    error_count: int
    error_rate: float
    avg_latency: float
    last_error: Optional[str] = None
    last_check: datetime = field(default_factory=datetime.now)


@dataclass
class Failure:
    """Represents a system failure"""
    failure_id: str
    component_id: str
    failure_type: FailureType
    error_message: str
    stack_trace: Optional[str]
    timestamp: datetime
    context: Dict[str, Any]
    severity: int = 5  # 1-10, higher is more severe


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from failures"""
    strategy_id: str
    applicable_failures: List[FailureType]
    actions: List[RecoveryAction]
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    timeout_seconds: float = 30.0


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""
    attempt_id: str
    failure_id: str
    action: RecoveryAction
    timestamp: datetime
    success: bool
    duration_seconds: float
    details: Dict[str, Any]


class HealthMonitor:
    """Monitors system health"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.metrics: Dict[str, HealthMetrics] = {}
        self.thresholds = {
            "error_rate": 0.05,  # 5%
            "latency_ms": 1000,  # 1 second
            "uptime_hours": 1  # Minimum uptime
        }
    
    def check_health(self, component_id: str) -> HealthStatus:
        """
        Check health of a component.
        
        Args:
            component_id: Component to check
            
        Returns:
            Health status
        """
        metrics = self.metrics.get(component_id)
        if not metrics:
            return HealthStatus.UNKNOWN
        
        # Check error rate
        if metrics.error_rate > self.thresholds["error_rate"]:
            return HealthStatus.UNHEALTHY
        
        # Check latency
        if metrics.avg_latency > self.thresholds["latency_ms"]:
            return HealthStatus.DEGRADED
        
        # Check recent errors
        if metrics.last_error:
            time_since_error = (datetime.now() - metrics.last_check).total_seconds()
            if time_since_error < 60:  # Error in last minute
                return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def update_metrics(self,
                      component_id: str,
                      request_count: int = 0,
                      error_count: int = 0,
                      latency: float = 0.0,
                      error_message: Optional[str] = None) -> None:
        """Update metrics for a component"""
        if component_id not in self.metrics:
            self.metrics[component_id] = HealthMetrics(
                component_id=component_id,
                status=HealthStatus.HEALTHY,
                uptime_seconds=0.0,
                request_count=0,
                error_count=0,
                error_rate=0.0,
                avg_latency=0.0
            )
        
        metrics = self.metrics[component_id]
        
        # Update counts
        metrics.request_count += request_count
        metrics.error_count += error_count
        
        # Calculate error rate
        if metrics.request_count > 0:
            metrics.error_rate = metrics.error_count / metrics.request_count
        
        # Update latency (moving average)
        if latency > 0:
            if metrics.avg_latency == 0:
                metrics.avg_latency = latency
            else:
                metrics.avg_latency = (metrics.avg_latency * 0.9 + latency * 0.1)
        
        # Update error info
        if error_message:
            metrics.last_error = error_message
        
        metrics.last_check = datetime.now()
        metrics.status = self.check_health(component_id)
    
    def get_unhealthy_components(self) -> List[str]:
        """Get list of unhealthy components"""
        unhealthy = []
        for component_id, metrics in self.metrics.items():
            if metrics.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]:
                unhealthy.append(component_id)
        return unhealthy


class FailureDiagnostic:
    """Diagnoses failures"""
    
    def __init__(self):
        self.diagnostic_rules = self._initialize_rules()
    
    def diagnose(self, failure: Failure) -> Dict[str, Any]:
        """
        Diagnose a failure.
        
        Args:
            failure: Failure to diagnose
            
        Returns:
            Diagnostic information
        """
        diagnosis = {
            "failure_id": failure.failure_id,
            "failure_type": failure.failure_type.value,
            "root_cause": None,
            "contributing_factors": [],
            "recommended_actions": []
        }
        
        # Apply diagnostic rules
        for rule in self.diagnostic_rules:
            if rule["failure_type"] == failure.failure_type:
                diagnosis["root_cause"] = rule["root_cause"]
                diagnosis["recommended_actions"] = rule["actions"]
                break
        
        # Analyze context
        if failure.context:
            diagnosis["contributing_factors"] = self._analyze_context(
                failure.context
            )
        
        return diagnosis
    
    def _initialize_rules(self) -> List[Dict[str, Any]]:
        """Initialize diagnostic rules"""
        return [
            {
                "failure_type": FailureType.TIMEOUT,
                "root_cause": "Operation exceeded time limit",
                "actions": [RecoveryAction.RETRY_WITH_BACKOFF, RecoveryAction.SCALE_UP]
            },
            {
                "failure_type": FailureType.RATE_LIMIT,
                "root_cause": "API rate limit exceeded",
                "actions": [RecoveryAction.RETRY_WITH_BACKOFF, RecoveryAction.CIRCUIT_BREAK]
            },
            {
                "failure_type": FailureType.AUTHENTICATION,
                "root_cause": "Authentication failed",
                "actions": [RecoveryAction.MANUAL_INTERVENTION]
            },
            {
                "failure_type": FailureType.RESOURCE_EXHAUSTION,
                "root_cause": "System resources exhausted",
                "actions": [RecoveryAction.SCALE_UP, RecoveryAction.RESTART]
            },
            {
                "failure_type": FailureType.EXTERNAL_SERVICE,
                "root_cause": "External service unavailable",
                "actions": [RecoveryAction.FALLBACK, RecoveryAction.CIRCUIT_BREAK]
            }
        ]
    
    def _analyze_context(self, context: Dict[str, Any]) -> List[str]:
        """Analyze failure context"""
        factors = []
        
        if context.get("high_load"):
            factors.append("System under high load")
        
        if context.get("low_memory"):
            factors.append("Low memory available")
        
        if context.get("network_issues"):
            factors.append("Network connectivity problems")
        
        return factors


class RecoveryEngine:
    """Executes recovery strategies"""
    
    def __init__(self):
        self.strategies: Dict[str, RecoveryStrategy] = {}
        self.recovery_history: List[RecoveryAttempt] = []
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._initialize_strategies()
    
    def recover(self,
                failure: Failure,
                component_id: str,
                operation: Callable) -> Tuple[bool, Optional[Any]]:
        """
        Attempt to recover from failure.
        
        Args:
            failure: Failure to recover from
            component_id: Component that failed
            operation: Operation to retry
            
        Returns:
            (success, result) tuple
        """
        # Find applicable strategy
        strategy = self._find_strategy(failure.failure_type)
        
        if not strategy:
            return False, None
        
        # Check circuit breaker
        if self._is_circuit_open(component_id):
            return False, None
        
        # Execute recovery actions
        for action in strategy.actions:
            attempt = self._execute_action(
                failure,
                action,
                component_id,
                operation,
                strategy
            )
            
            self.recovery_history.append(attempt)
            
            if attempt.success:
                # Reset circuit breaker on success
                self._reset_circuit_breaker(component_id)
                return True, attempt.details.get("result")
        
        # All actions failed - open circuit breaker
        self._open_circuit_breaker(component_id)
        
        return False, None
    
    def add_strategy(self, strategy: RecoveryStrategy) -> None:
        """Add recovery strategy"""
        self.strategies[strategy.strategy_id] = strategy
    
    def _find_strategy(self, failure_type: FailureType) -> Optional[RecoveryStrategy]:
        """Find strategy for failure type"""
        for strategy in self.strategies.values():
            if failure_type in strategy.applicable_failures:
                return strategy
        return None
    
    def _execute_action(self,
                       failure: Failure,
                       action: RecoveryAction,
                       component_id: str,
                       operation: Callable,
                       strategy: RecoveryStrategy) -> RecoveryAttempt:
        """Execute a recovery action"""
        start_time = datetime.now()
        attempt = RecoveryAttempt(
            attempt_id="recovery_{}_{}".format(
                failure.failure_id,
                len(self.recovery_history)
            ),
            failure_id=failure.failure_id,
            action=action,
            timestamp=start_time,
            success=False,
            duration_seconds=0.0,
            details={}
        )
        
        try:
            if action == RecoveryAction.RETRY:
                result = self._retry(operation, max_retries=strategy.max_retries)
                attempt.success = True
                attempt.details["result"] = result
            
            elif action == RecoveryAction.RETRY_WITH_BACKOFF:
                result = self._retry_with_backoff(
                    operation,
                    max_retries=strategy.max_retries,
                    backoff_multiplier=strategy.backoff_multiplier
                )
                attempt.success = True
                attempt.details["result"] = result
            
            elif action == RecoveryAction.FALLBACK:
                result = self._fallback(operation)
                attempt.success = True
                attempt.details["result"] = result
                attempt.details["fallback_used"] = True
            
            elif action == RecoveryAction.CIRCUIT_BREAK:
                self._open_circuit_breaker(component_id)
                attempt.success = True
                attempt.details["circuit_broken"] = True
            
            elif action == RecoveryAction.DEGRADE_GRACEFULLY:
                result = self._degrade_gracefully(operation)
                attempt.success = True
                attempt.details["result"] = result
                attempt.details["degraded"] = True
        
        except Exception as e:
            attempt.success = False
            attempt.details["error"] = str(e)
        
        attempt.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        return attempt
    
    def _retry(self, operation: Callable, max_retries: int = 3) -> Any:
        """Simple retry"""
        last_error = None
        
        for i in range(max_retries):
            try:
                return operation()
            except Exception as e:
                last_error = e
                if i == max_retries - 1:
                    raise last_error
        
        raise last_error
    
    def _retry_with_backoff(self,
                           operation: Callable,
                           max_retries: int = 3,
                           backoff_multiplier: float = 2.0) -> Any:
        """Retry with exponential backoff"""
        import time
        last_error = None
        
        for i in range(max_retries):
            try:
                return operation()
            except Exception as e:
                last_error = e
                if i < max_retries - 1:
                    # Exponential backoff
                    delay = backoff_multiplier ** i
                    time.sleep(delay)
                else:
                    raise last_error
        
        raise last_error
    
    def _fallback(self, operation: Callable) -> Any:
        """Use fallback mechanism"""
        # Return cached or default result
        return {"fallback": True, "data": None}
    
    def _degrade_gracefully(self, operation: Callable) -> Any:
        """Degrade service gracefully"""
        # Return partial results or simplified response
        return {"degraded": True, "partial_data": True}
    
    def _is_circuit_open(self, component_id: str) -> bool:
        """Check if circuit breaker is open"""
        if component_id not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[component_id]
        
        # Check if enough time has passed to retry
        if breaker["state"] == "open":
            time_elapsed = (datetime.now() - breaker["opened_at"]).total_seconds()
            if time_elapsed > breaker["timeout"]:
                # Move to half-open state
                breaker["state"] = "half-open"
                return False
            return True
        
        return False
    
    def _open_circuit_breaker(self, component_id: str) -> None:
        """Open circuit breaker"""
        self.circuit_breakers[component_id] = {
            "state": "open",
            "opened_at": datetime.now(),
            "timeout": 60,  # Try again after 60 seconds
            "failure_count": self.circuit_breakers.get(component_id, {}).get(
                "failure_count", 0
            ) + 1
        }
    
    def _reset_circuit_breaker(self, component_id: str) -> None:
        """Reset circuit breaker"""
        if component_id in self.circuit_breakers:
            self.circuit_breakers[component_id]["state"] = "closed"
            self.circuit_breakers[component_id]["failure_count"] = 0
    
    def _initialize_strategies(self) -> None:
        """Initialize default recovery strategies"""
        self.add_strategy(RecoveryStrategy(
            strategy_id="timeout_strategy",
            applicable_failures=[FailureType.TIMEOUT],
            actions=[
                RecoveryAction.RETRY_WITH_BACKOFF,
                RecoveryAction.DEGRADE_GRACEFULLY
            ],
            max_retries=3,
            backoff_multiplier=2.0
        ))
        
        self.add_strategy(RecoveryStrategy(
            strategy_id="rate_limit_strategy",
            applicable_failures=[FailureType.RATE_LIMIT],
            actions=[
                RecoveryAction.RETRY_WITH_BACKOFF,
                RecoveryAction.CIRCUIT_BREAK
            ],
            max_retries=5,
            backoff_multiplier=3.0
        ))
        
        self.add_strategy(RecoveryStrategy(
            strategy_id="external_service_strategy",
            applicable_failures=[FailureType.EXTERNAL_SERVICE],
            actions=[
                RecoveryAction.RETRY_WITH_BACKOFF,
                RecoveryAction.FALLBACK,
                RecoveryAction.CIRCUIT_BREAK
            ],
            max_retries=3,
            backoff_multiplier=2.0
        ))


class SelfHealingAgent:
    """
    Self-healing agent system that monitors, diagnoses, and recovers from failures.
    """
    
    def __init__(self, check_interval: int = 30):
        self.health_monitor = HealthMonitor(check_interval)
        self.diagnostic = FailureDiagnostic()
        self.recovery_engine = RecoveryEngine()
        
        self.failures: Dict[str, Failure] = {}
        self.failure_counter = 0
        
        # Learning from failures
        self.failure_patterns: Dict[str, List[Failure]] = {}
    
    def execute_with_healing(self,
                           component_id: str,
                           operation: Callable,
                           context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute operation with automatic healing on failure.
        
        Args:
            component_id: Component identifier
            operation: Operation to execute
            context: Optional context
            
        Returns:
            Operation result
        """
        if context is None:
            context = {}
        
        start_time = datetime.now()
        
        try:
            # Execute operation
            result = operation()
            
            # Update metrics on success
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.health_monitor.update_metrics(
                component_id,
                request_count=1,
                latency=latency
            )
            
            return result
        
        except Exception as e:
            # Record failure
            failure = self._create_failure(component_id, e, context)
            self.failures[failure.failure_id] = failure
            
            # Update metrics
            self.health_monitor.update_metrics(
                component_id,
                request_count=1,
                error_count=1,
                error_message=str(e)
            )
            
            # Diagnose failure
            diagnosis = self.diagnostic.diagnose(failure)
            
            # Attempt recovery
            recovered, result = self.recovery_engine.recover(
                failure,
                component_id,
                operation
            )
            
            if recovered:
                # Update metrics on recovery
                self.health_monitor.update_metrics(
                    component_id,
                    request_count=1
                )
                return result
            else:
                # Recovery failed
                raise Exception("Recovery failed for {}: {}".format(
                    component_id, str(e)
                ))
    
    def get_health_status(self, component_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get health status.
        
        Args:
            component_id: Optional specific component
            
        Returns:
            Health status information
        """
        if component_id:
            metrics = self.health_monitor.metrics.get(component_id)
            if metrics:
                return {
                    "component_id": component_id,
                    "status": metrics.status.value,
                    "error_rate": metrics.error_rate,
                    "avg_latency_ms": metrics.avg_latency,
                    "request_count": metrics.request_count,
                    "error_count": metrics.error_count
                }
        
        # Return overall health
        all_components = {}
        for comp_id, metrics in self.health_monitor.metrics.items():
            all_components[comp_id] = {
                "status": metrics.status.value,
                "error_rate": metrics.error_rate
            }
        
        return {
            "components": all_components,
            "unhealthy_count": len(self.health_monitor.get_unhealthy_components())
        }
    
    def get_failure_report(self) -> Dict[str, Any]:
        """Get failure report"""
        failure_by_type = {}
        for failure in self.failures.values():
            ftype = failure.failure_type.value
            failure_by_type[ftype] = failure_by_type.get(ftype, 0) + 1
        
        recovery_success_rate = 0.0
        if self.recovery_engine.recovery_history:
            successful = sum(
                1 for attempt in self.recovery_engine.recovery_history
                if attempt.success
            )
            recovery_success_rate = successful / len(
                self.recovery_engine.recovery_history
            )
        
        return {
            "total_failures": len(self.failures),
            "failures_by_type": failure_by_type,
            "recovery_attempts": len(self.recovery_engine.recovery_history),
            "recovery_success_rate": recovery_success_rate,
            "circuit_breakers": len(self.recovery_engine.circuit_breakers)
        }
    
    def _create_failure(self,
                       component_id: str,
                       error: Exception,
                       context: Dict[str, Any]) -> Failure:
        """Create failure record"""
        failure_type = self._classify_failure(error)
        
        failure = Failure(
            failure_id="failure_{}".format(self.failure_counter),
            component_id=component_id,
            failure_type=failure_type,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            timestamp=datetime.now(),
            context=context,
            severity=self._calculate_severity(failure_type)
        )
        
        self.failure_counter += 1
        
        # Learn from failure pattern
        if failure_type.value not in self.failure_patterns:
            self.failure_patterns[failure_type.value] = []
        self.failure_patterns[failure_type.value].append(failure)
        
        return failure
    
    def _classify_failure(self, error: Exception) -> FailureType:
        """Classify failure type"""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return FailureType.TIMEOUT
        elif "rate limit" in error_str:
            return FailureType.RATE_LIMIT
        elif "auth" in error_str:
            return FailureType.AUTHENTICATION
        elif "memory" in error_str or "resource" in error_str:
            return FailureType.RESOURCE_EXHAUSTION
        elif "external" in error_str or "service" in error_str:
            return FailureType.EXTERNAL_SERVICE
        else:
            return FailureType.UNKNOWN
    
    def _calculate_severity(self, failure_type: FailureType) -> int:
        """Calculate failure severity (1-10)"""
        severity_map = {
            FailureType.TIMEOUT: 4,
            FailureType.RATE_LIMIT: 3,
            FailureType.AUTHENTICATION: 8,
            FailureType.RESOURCE_EXHAUSTION: 7,
            FailureType.EXTERNAL_SERVICE: 5,
            FailureType.INVALID_INPUT: 2,
            FailureType.CONFIGURATION: 6,
            FailureType.UNKNOWN: 5
        }
        return severity_map.get(failure_type, 5)


def demonstrate_self_healing():
    """Demonstrate self-healing agent system"""
    print("=" * 70)
    print("Self-Healing Agent System Demonstration")
    print("=" * 70)
    
    agent = SelfHealingAgent()
    
    # Example 1: Successful operation
    print("\n1. Successful Operation:")
    
    def successful_operation():
        return "Success!"
    
    result = agent.execute_with_healing("api_service", successful_operation)
    print("Result: {}".format(result))
    
    health = agent.get_health_status("api_service")
    print("Health: {}".format(json.dumps(health, indent=2)))
    
    # Example 2: Operation with recoverable failure
    print("\n2. Operation with Recoverable Failure:")
    
    call_count = [0]
    
    def flaky_operation():
        call_count[0] += 1
        if call_count[0] < 3:
            raise Exception("Temporary timeout error")
        return "Success after retries"
    
    try:
        result = agent.execute_with_healing(
            "flaky_service",
            flaky_operation,
            context={"retry_enabled": True}
        )
        print("Result: {}".format(result))
        print("Recovered after {} attempts".format(call_count[0]))
    except Exception as e:
        print("Failed: {}".format(e))
    
    # Example 3: Rate limit failure
    print("\n3. Rate Limit Failure:")
    
    def rate_limited_operation():
        raise Exception("Rate limit exceeded")
    
    try:
        result = agent.execute_with_healing(
            "rate_limited_service",
            rate_limited_operation
        )
    except Exception as e:
        print("Failed after recovery attempts: {}".format(str(e)[:80]))
    
    # Example 4: Health status
    print("\n4. Overall Health Status:")
    overall_health = agent.get_health_status()
    print(json.dumps(overall_health, indent=2, default=str))
    
    # Example 5: Failure report
    print("\n5. Failure Report:")
    report = agent.get_failure_report()
    print(json.dumps(report, indent=2))
    
    # Example 6: Circuit breaker status
    print("\n6. Circuit Breaker Status:")
    for comp_id, breaker in agent.recovery_engine.circuit_breakers.items():
        print("\nComponent: {}".format(comp_id))
        print("  State: {}".format(breaker["state"]))
        print("  Failure count: {}".format(breaker["failure_count"]))
    
    # Example 7: Recovery history
    print("\n7. Recent Recovery Attempts:")
    for attempt in agent.recovery_engine.recovery_history[-5:]:
        print("\nAttempt: {}".format(attempt.attempt_id))
        print("  Action: {}".format(attempt.action.value))
        print("  Success: {}".format(attempt.success))
        print("  Duration: {:.3f}s".format(attempt.duration_seconds))


if __name__ == "__main__":
    demonstrate_self_healing()
