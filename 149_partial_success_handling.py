"""
Pattern 149: Partial Success Handling Agent

This pattern handles scenarios where operations partially succeed - some tasks
complete successfully while others fail. Aggregates results, tracks failures,
and provides comprehensive reporting with degraded service handling.

Category: Error Handling & Recovery
Use Cases:
- Batch processing with failures
- Distributed operations
- Multi-service requests
- Fault-tolerant systems
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, TypeVar, Generic
from enum import Enum
from datetime import datetime
import random
import time

T = TypeVar('T')


class OperationStatus(Enum):
    """Status of individual operations"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class AggregationStrategy(Enum):
    """Strategies for aggregating partial results"""
    ALL_OR_NOTHING = "all_or_nothing"    # Fail if any fails
    BEST_EFFORT = "best_effort"          # Accept any successes
    THRESHOLD = "threshold"              # Require minimum success rate
    CRITICAL_PATH = "critical_path"      # Require critical operations
    MAJORITY = "majority"                # Require majority success


class FailureHandling(Enum):
    """How to handle failures"""
    STOP_ON_FIRST = "stop_on_first"      # Stop on first failure
    CONTINUE_ALL = "continue_all"        # Continue despite failures
    STOP_ON_CRITICAL = "stop_on_critical"  # Stop only on critical failures


@dataclass
class OperationResult(Generic[T]):
    """Result of a single operation"""
    operation_id: str
    status: OperationStatus
    result: Optional[T] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    is_critical: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedResult(Generic[T]):
    """Aggregated result from multiple operations"""
    overall_status: OperationStatus
    success_count: int
    failure_count: int
    total_count: int
    success_rate: float
    successful_results: List[T]
    failed_operations: List[str]
    individual_results: List[OperationResult[T]]
    is_acceptable: bool
    degradation_level: float  # 0.0 = no degradation, 1.0 = total failure
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchOperation(Generic[T]):
    """Definition of a batch operation"""
    operation_id: str
    operation_func: Callable[..., T]
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    is_critical: bool = False
    timeout_ms: Optional[float] = None
    retry_count: int = 0


class ResultAggregator:
    """Aggregates results from multiple operations"""
    
    def __init__(self, strategy: AggregationStrategy = AggregationStrategy.BEST_EFFORT):
        self.strategy = strategy
        self.threshold = 0.5  # For threshold strategy
        
    def aggregate(
        self,
        results: List[OperationResult],
        threshold: Optional[float] = None
    ) -> AggregatedResult:
        """Aggregate operation results"""
        if threshold:
            self.threshold = threshold
        
        total = len(results)
        successes = [r for r in results if r.status == OperationStatus.SUCCESS]
        failures = [r for r in results if r.status == OperationStatus.FAILED]
        
        success_count = len(successes)
        failure_count = len(failures)
        success_rate = success_count / total if total > 0 else 0.0
        
        # Determine overall status
        if success_count == total:
            overall_status = OperationStatus.SUCCESS
        elif success_count == 0:
            overall_status = OperationStatus.FAILED
        else:
            overall_status = OperationStatus.PARTIAL
        
        # Check if result is acceptable
        is_acceptable = self._check_acceptability(results, success_rate)
        
        # Calculate degradation
        degradation = 1.0 - success_rate
        
        # Add critical failures to degradation
        critical_failures = [r for r in failures if r.is_critical]
        if critical_failures:
            degradation = min(1.0, degradation + 0.5)
        
        # Aggregate successful results
        successful_results = [r.result for r in successes if r.result is not None]
        failed_operations = [r.operation_id for r in failures]
        
        # Calculate total execution time
        total_time = sum(r.execution_time_ms for r in results)
        
        return AggregatedResult(
            overall_status=overall_status,
            success_count=success_count,
            failure_count=failure_count,
            total_count=total,
            success_rate=success_rate,
            successful_results=successful_results,
            failed_operations=failed_operations,
            individual_results=results,
            is_acceptable=is_acceptable,
            degradation_level=degradation,
            execution_time_ms=total_time,
            metadata={
                'strategy': self.strategy.value,
                'threshold': self.threshold,
                'critical_failures': len(critical_failures)
            }
        )
    
    def _check_acceptability(
        self,
        results: List[OperationResult],
        success_rate: float
    ) -> bool:
        """Check if result is acceptable based on strategy"""
        if self.strategy == AggregationStrategy.ALL_OR_NOTHING:
            return success_rate == 1.0
        
        elif self.strategy == AggregationStrategy.BEST_EFFORT:
            return success_rate > 0.0
        
        elif self.strategy == AggregationStrategy.THRESHOLD:
            return success_rate >= self.threshold
        
        elif self.strategy == AggregationStrategy.CRITICAL_PATH:
            critical_results = [r for r in results if r.is_critical]
            if not critical_results:
                return success_rate > 0.0
            critical_success = all(r.status == OperationStatus.SUCCESS for r in critical_results)
            return critical_success
        
        elif self.strategy == AggregationStrategy.MAJORITY:
            return success_rate > 0.5
        
        return False


class PartialSuccessHandler:
    """Handles partial success scenarios"""
    
    def __init__(self, failure_handling: FailureHandling = FailureHandling.CONTINUE_ALL):
        self.failure_handling = failure_handling
        
    def execute_batch(
        self,
        operations: List[BatchOperation],
        aggregator: ResultAggregator
    ) -> AggregatedResult:
        """Execute a batch of operations"""
        results: List[OperationResult] = []
        
        for op in operations:
            # Check if should continue
            if not self._should_continue(results):
                results.append(OperationResult(
                    operation_id=op.operation_id,
                    status=OperationStatus.SKIPPED,
                    error="Skipped due to failure handling policy",
                    is_critical=op.is_critical
                ))
                continue
            
            # Execute operation
            result = self._execute_operation(op)
            results.append(result)
        
        # Aggregate results
        return aggregator.aggregate(results)
    
    def _should_continue(self, results: List[OperationResult]) -> bool:
        """Check if should continue executing operations"""
        if not results:
            return True
        
        if self.failure_handling == FailureHandling.CONTINUE_ALL:
            return True
        
        if self.failure_handling == FailureHandling.STOP_ON_FIRST:
            return not any(r.status == OperationStatus.FAILED for r in results)
        
        if self.failure_handling == FailureHandling.STOP_ON_CRITICAL:
            return not any(
                r.status == OperationStatus.FAILED and r.is_critical
                for r in results
            )
        
        return True
    
    def _execute_operation(self, operation: BatchOperation) -> OperationResult:
        """Execute a single operation"""
        start_time = time.time()
        
        try:
            result = operation.operation_func(*operation.args, **operation.kwargs)
            execution_time = (time.time() - start_time) * 1000
            
            return OperationResult(
                operation_id=operation.operation_id,
                status=OperationStatus.SUCCESS,
                result=result,
                execution_time_ms=execution_time,
                is_critical=operation.is_critical
            )
        
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return OperationResult(
                operation_id=operation.operation_id,
                status=OperationStatus.FAILED,
                error=str(e),
                execution_time_ms=execution_time,
                is_critical=operation.is_critical
            )


class DegradedServiceManager:
    """Manages degraded service scenarios"""
    
    def __init__(self):
        self.degradation_thresholds = {
            'minimal': 0.1,      # < 10% degradation
            'moderate': 0.3,     # 10-30% degradation
            'significant': 0.5,  # 30-50% degradation
            'severe': 0.75,      # 50-75% degradation
            'critical': 1.0      # > 75% degradation
        }
    
    def get_service_level(self, degradation: float) -> str:
        """Get service level based on degradation"""
        if degradation < self.degradation_thresholds['minimal']:
            return "full_service"
        elif degradation < self.degradation_thresholds['moderate']:
            return "minimal_degradation"
        elif degradation < self.degradation_thresholds['significant']:
            return "moderate_degradation"
        elif degradation < self.degradation_thresholds['severe']:
            return "significant_degradation"
        elif degradation < self.degradation_thresholds['critical']:
            return "severe_degradation"
        else:
            return "critical_failure"
    
    def get_fallback_actions(self, service_level: str) -> List[str]:
        """Get recommended fallback actions"""
        actions = {
            'full_service': [],
            'minimal_degradation': ['log_warning', 'monitor_closely'],
            'moderate_degradation': ['enable_caching', 'reduce_features', 'notify_team'],
            'significant_degradation': ['enable_fallbacks', 'disable_non_critical', 'alert_oncall'],
            'severe_degradation': ['emergency_mode', 'disable_most_features', 'page_oncall'],
            'critical_failure': ['complete_shutdown', 'maintenance_mode', 'emergency_response']
        }
        return actions.get(service_level, [])
    
    def create_degraded_response(
        self,
        aggregated_result: AggregatedResult,
        service_name: str = "service"
    ) -> Dict[str, Any]:
        """Create a response for degraded service"""
        service_level = self.get_service_level(aggregated_result.degradation_level)
        fallback_actions = self.get_fallback_actions(service_level)
        
        return {
            'service': service_name,
            'status': aggregated_result.overall_status.value,
            'service_level': service_level,
            'success_rate': f"{aggregated_result.success_rate:.1%}",
            'available_results': len(aggregated_result.successful_results),
            'failed_operations': aggregated_result.failed_operations,
            'degradation': f"{aggregated_result.degradation_level:.1%}",
            'is_acceptable': aggregated_result.is_acceptable,
            'fallback_actions': fallback_actions,
            'message': self._get_user_message(service_level, aggregated_result)
        }
    
    def _get_user_message(
        self,
        service_level: str,
        result: AggregatedResult
    ) -> str:
        """Get user-facing message"""
        if service_level == 'full_service':
            return "All services operating normally"
        elif service_level == 'minimal_degradation':
            return f"Service operating with minor issues ({result.failure_count} operations failed)"
        elif service_level == 'moderate_degradation':
            return f"Service partially degraded ({result.success_rate:.0%} success rate)"
        elif service_level == 'significant_degradation':
            return f"Service significantly degraded ({result.failure_count} failures)"
        elif service_level == 'severe_degradation':
            return f"Service severely impacted ({result.success_rate:.0%} success rate)"
        else:
            return "Service unavailable - critical failures detected"


class PartialSuccessHandlingAgent:
    """
    Main agent that handles partial success scenarios.
    Aggregates results, manages failures, and provides degraded service handling.
    """
    
    def __init__(
        self,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.BEST_EFFORT,
        failure_handling: FailureHandling = FailureHandling.CONTINUE_ALL
    ):
        self.aggregator = ResultAggregator(aggregation_strategy)
        self.handler = PartialSuccessHandler(failure_handling)
        self.degraded_service_manager = DegradedServiceManager()
        self.execution_history: List[Dict[str, Any]] = []
    
    def execute_operations(
        self,
        operations: List[BatchOperation],
        service_name: str = "service",
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute operations and handle partial success"""
        # Execute batch
        result = self.handler.execute_batch(operations, self.aggregator)
        
        # If threshold provided, update it
        if threshold:
            self.aggregator.threshold = threshold
            # Re-check acceptability
            result.is_acceptable = self.aggregator._check_acceptability(
                result.individual_results,
                result.success_rate
            )
        
        # Create degraded response
        response = self.degraded_service_manager.create_degraded_response(
            result,
            service_name
        )
        
        # Record execution
        self.execution_history.append({
            'timestamp': datetime.now(),
            'service': service_name,
            'total_operations': result.total_count,
            'success_count': result.success_count,
            'success_rate': result.success_rate,
            'degradation': result.degradation_level,
            'is_acceptable': result.is_acceptable
        })
        
        return {
            'aggregated_result': result,
            'service_response': response
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"error": "No execution history"}
        
        total_executions = len(self.execution_history)
        total_operations = sum(h['total_operations'] for h in self.execution_history)
        total_successes = sum(h['success_count'] for h in self.execution_history)
        
        avg_success_rate = sum(h['success_rate'] for h in self.execution_history) / total_executions
        avg_degradation = sum(h['degradation'] for h in self.execution_history) / total_executions
        acceptable_rate = sum(1 for h in self.execution_history if h['is_acceptable']) / total_executions
        
        return {
            'total_executions': total_executions,
            'total_operations': total_operations,
            'total_successes': total_successes,
            'average_success_rate': f"{avg_success_rate:.1%}",
            'average_degradation': f"{avg_degradation:.1%}",
            'acceptable_rate': f"{acceptable_rate:.1%}",
            'aggregation_strategy': self.aggregator.strategy.value,
            'failure_handling': self.handler.failure_handling.value
        }


def demonstrate_partial_success_handling():
    """Demonstrate partial success handling pattern"""
    print("\n" + "="*60)
    print("PARTIAL SUCCESS HANDLING PATTERN DEMONSTRATION")
    print("="*60)
    
    # Helper functions that simulate operations
    def process_item(item_id: int, should_fail: bool = False) -> str:
        """Simulate processing an item"""
        if should_fail:
            raise Exception(f"Failed to process item {item_id}")
        return f"Processed item {item_id}"
    
    def fetch_data(source: str, should_fail: bool = False) -> Dict[str, Any]:
        """Simulate fetching data"""
        if should_fail:
            raise Exception(f"Failed to fetch from {source}")
        return {'source': source, 'data': f'Data from {source}'}
    
    # Scenario 1: Best effort with multiple failures
    print("\n" + "-"*60)
    print("Scenario 1: Best Effort Strategy (Accept Any Success)")
    print("-"*60)
    
    agent = PartialSuccessHandlingAgent(
        aggregation_strategy=AggregationStrategy.BEST_EFFORT,
        failure_handling=FailureHandling.CONTINUE_ALL
    )
    
    operations = [
        BatchOperation("op1", process_item, args=(1, False)),
        BatchOperation("op2", process_item, args=(2, True)),   # Will fail
        BatchOperation("op3", process_item, args=(3, False)),
        BatchOperation("op4", process_item, args=(4, True)),   # Will fail
        BatchOperation("op5", process_item, args=(5, False)),
    ]
    
    result = agent.execute_operations(operations, "batch_processor")
    
    agg_result = result['aggregated_result']
    response = result['service_response']
    
    print(f"Executed {agg_result.total_count} operations:")
    print(f"  Success: {agg_result.success_count}")
    print(f"  Failed: {agg_result.failure_count}")
    print(f"  Success Rate: {agg_result.success_rate:.0%}")
    print(f"  Status: {agg_result.overall_status.value}")
    print(f"  Acceptable: {agg_result.is_acceptable}")
    print(f"\nService Level: {response['service_level']}")
    print(f"Message: {response['message']}")
    print(f"Failed Operations: {', '.join(response['failed_operations'])}")
    
    # Scenario 2: Threshold strategy
    print("\n" + "-"*60)
    print("Scenario 2: Threshold Strategy (80% Required)")
    print("-"*60)
    
    agent = PartialSuccessHandlingAgent(
        aggregation_strategy=AggregationStrategy.THRESHOLD,
        failure_handling=FailureHandling.CONTINUE_ALL
    )
    
    operations = [
        BatchOperation("fetch1", fetch_data, args=("database", False)),
        BatchOperation("fetch2", fetch_data, args=("cache", False)),
        BatchOperation("fetch3", fetch_data, args=("api", True)),  # Will fail
        BatchOperation("fetch4", fetch_data, args=("file", False)),
        BatchOperation("fetch5", fetch_data, args=("remote", False)),
    ]
    
    result = agent.execute_operations(operations, "data_fetcher", threshold=0.8)
    
    agg_result = result['aggregated_result']
    response = result['service_response']
    
    print(f"Threshold: 80% success required")
    print(f"Actual Success Rate: {agg_result.success_rate:.0%}")
    print(f"Acceptable: {agg_result.is_acceptable}")
    print(f"Degradation: {response['degradation']}")
    print(f"Service Level: {response['service_level']}")
    
    # Scenario 3: Critical path strategy
    print("\n" + "-"*60)
    print("Scenario 3: Critical Path Strategy")
    print("-"*60)
    
    agent = PartialSuccessHandlingAgent(
        aggregation_strategy=AggregationStrategy.CRITICAL_PATH,
        failure_handling=FailureHandling.STOP_ON_CRITICAL
    )
    
    operations = [
        BatchOperation("auth", process_item, args=(1, False), is_critical=True),
        BatchOperation("load_profile", process_item, args=(2, False)),
        BatchOperation("load_preferences", process_item, args=(3, True)),  # Non-critical fail
        BatchOperation("load_history", process_item, args=(4, True)),      # Non-critical fail
    ]
    
    result = agent.execute_operations(operations, "user_session")
    
    agg_result = result['aggregated_result']
    response = result['service_response']
    
    print(f"Critical operations: {sum(1 for op in operations if op.is_critical)}")
    print(f"Total success rate: {agg_result.success_rate:.0%}")
    print(f"Acceptable: {agg_result.is_acceptable} (critical operations succeeded)")
    print(f"Service Level: {response['service_level']}")
    print(f"Available Results: {response['available_results']}/{agg_result.total_count}")
    
    # Scenario 4: Stop on first failure
    print("\n" + "-"*60)
    print("Scenario 4: Stop on First Failure")
    print("-"*60)
    
    agent = PartialSuccessHandlingAgent(
        aggregation_strategy=AggregationStrategy.ALL_OR_NOTHING,
        failure_handling=FailureHandling.STOP_ON_FIRST
    )
    
    operations = [
        BatchOperation("step1", process_item, args=(1, False)),
        BatchOperation("step2", process_item, args=(2, True)),   # Will fail
        BatchOperation("step3", process_item, args=(3, False)),  # Will be skipped
        BatchOperation("step4", process_item, args=(4, False)),  # Will be skipped
    ]
    
    result = agent.execute_operations(operations, "transaction")
    
    agg_result = result['aggregated_result']
    
    print(f"Executed: {agg_result.success_count + agg_result.failure_count}/{agg_result.total_count}")
    print(f"Skipped: {sum(1 for r in agg_result.individual_results if r.status == OperationStatus.SKIPPED)}")
    print(f"Status breakdown:")
    for r in agg_result.individual_results:
        print(f"  {r.operation_id}: {r.status.value}")
    
    # Scenario 5: Majority strategy
    print("\n" + "-"*60)
    print("Scenario 5: Majority Strategy (>50% Required)")
    print("-"*60)
    
    agent = PartialSuccessHandlingAgent(
        aggregation_strategy=AggregationStrategy.MAJORITY
    )
    
    # Test with majority success
    operations = [
        BatchOperation("node1", fetch_data, args=("node1", False)),
        BatchOperation("node2", fetch_data, args=("node2", False)),
        BatchOperation("node3", fetch_data, args=("node3", False)),
        BatchOperation("node4", fetch_data, args=("node4", True)),  # Fail
        BatchOperation("node5", fetch_data, args=("node5", True)),  # Fail
    ]
    
    result = agent.execute_operations(operations, "distributed_query")
    
    agg_result = result['aggregated_result']
    response = result['service_response']
    
    print(f"Success Rate: {agg_result.success_rate:.0%}")
    print(f"Majority achieved: {agg_result.is_acceptable}")
    print(f"Degradation: {response['degradation']}")
    print(f"Fallback Actions: {', '.join(response['fallback_actions'])}")
    
    # Scenario 6: Severe degradation
    print("\n" + "-"*60)
    print("Scenario 6: Severe Degradation (Most Operations Fail)")
    print("-"*60)
    
    agent = PartialSuccessHandlingAgent(
        aggregation_strategy=AggregationStrategy.BEST_EFFORT
    )
    
    operations = [
        BatchOperation("op1", process_item, args=(1, True)),   # Fail
        BatchOperation("op2", process_item, args=(2, True)),   # Fail
        BatchOperation("op3", process_item, args=(3, False)),  # Success
        BatchOperation("op4", process_item, args=(4, True)),   # Fail
        BatchOperation("op5", process_item, args=(5, True)),   # Fail
    ]
    
    result = agent.execute_operations(operations, "critical_service")
    
    agg_result = result['aggregated_result']
    response = result['service_response']
    
    print(f"Success Rate: {agg_result.success_rate:.0%}")
    print(f"Degradation Level: {response['degradation']}")
    print(f"Service Level: {response['service_level']}")
    print(f"Message: {response['message']}")
    print(f"Recommended Actions:")
    for action in response['fallback_actions']:
        print(f"  - {action}")
    
    # Scenario 7: Statistics across all executions
    print("\n" + "-"*60)
    print("Scenario 7: Aggregate Statistics")
    print("-"*60)
    
    stats = agent.get_statistics()
    
    print(f"\nTotal Executions: {stats['total_executions']}")
    print(f"Total Operations: {stats['total_operations']}")
    print(f"Total Successes: {stats['total_successes']}")
    print(f"Average Success Rate: {stats['average_success_rate']}")
    print(f"Average Degradation: {stats['average_degradation']}")
    print(f"Acceptable Rate: {stats['acceptable_rate']}")
    print(f"Strategy: {stats['aggregation_strategy']}")
    print(f"Failure Handling: {stats['failure_handling']}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Handled {stats['total_operations']} operations across {stats['total_executions']} batches")
    print(f"✓ {stats['average_success_rate']} average success rate with {stats['average_degradation']} degradation")
    print(f"✓ Multiple aggregation strategies: best effort, threshold, critical path, majority")
    print(f"✓ Intelligent failure handling: continue all, stop on first, stop on critical")
    print(f"✓ Degraded service management with 6 service levels")
    print(f"✓ Automatic fallback action recommendations")
    print("\n✅ Error Handling & Recovery Category: Pattern 4/5 complete")
    print("Partial success handling ready for production!")


if __name__ == "__main__":
    demonstrate_partial_success_handling()
