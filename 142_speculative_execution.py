"""
Agentic Design Pattern: Speculative Execution

This pattern implements speculative execution where the agent predicts likely execution
paths and pre-computes results in parallel. When the actual decision is made, results
are ready, reducing latency.

Category: Performance Optimization
Use Cases:
- Low-latency applications
- Predictable workflows
- Multi-path decision making
- User interaction optimization
- Preemptive computation
- Response time reduction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime
import hashlib
import time
import threading
from collections import defaultdict

class ExecutionState(Enum):
    """State of speculative execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    USED = "used"
    WASTED = "wasted"


class PredictionStrategy(Enum):
    """Strategy for predicting execution paths"""
    FREQUENCY_BASED = "frequency_based"
    PATTERN_BASED = "pattern_based"
    ML_BASED = "ml_based"
    HEURISTIC = "heuristic"
    ALL_PATHS = "all_paths"


@dataclass
class ExecutionPath:
    """Represents a potential execution path"""
    path_id: str
    name: str
    computation: Callable[[], Any]
    probability: float  # 0-1
    priority: int
    cost_estimate: float
    state: ExecutionState = ExecutionState.PENDING
    result: Optional[Any] = None
    execution_time: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class SpeculativeResult:
    """Result of speculative execution"""
    path_id: str
    result: Any
    execution_time: float
    was_speculative: bool
    prediction_correct: bool


@dataclass
class PathPrediction:
    """Prediction of which path will be taken"""
    predicted_paths: List[str]
    confidence: float
    strategy: PredictionStrategy
    reasoning: str


class PathPredictor:
    """Predicts which execution paths are likely"""
    
    def __init__(self, strategy: PredictionStrategy = PredictionStrategy.FREQUENCY_BASED):
        self.strategy = strategy
        self.execution_history: List[Dict[str, Any]] = []
        self.path_frequencies: Dict[str, int] = defaultdict(int)
        self.transition_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.last_path: Optional[str] = None
    
    def record_execution(self, path_id: str, context: Dict[str, Any]) -> None:
        """Record an execution for learning"""
        self.execution_history.append({
            "path_id": path_id,
            "context": context,
            "timestamp": datetime.now()
        })
        
        self.path_frequencies[path_id] += 1
        
        # Update transition matrix
        if self.last_path:
            self.transition_matrix[self.last_path][path_id] += 1
        
        self.last_path = path_id
    
    def predict_paths(self, 
                     available_paths: List[str],
                     context: Dict[str, Any],
                     num_predictions: int = 3) -> PathPrediction:
        """Predict which paths are likely to be executed"""
        
        if self.strategy == PredictionStrategy.ALL_PATHS:
            return PathPrediction(
                predicted_paths=available_paths,
                confidence=1.0,
                strategy=self.strategy,
                reasoning="Execute all paths speculatively"
            )
        
        elif self.strategy == PredictionStrategy.FREQUENCY_BASED:
            # Predict based on historical frequency
            sorted_paths = sorted(
                available_paths,
                key=lambda p: self.path_frequencies.get(p, 0),
                reverse=True
            )
            
            top_paths = sorted_paths[:num_predictions]
            total_freq = sum(self.path_frequencies.values())
            confidence = (sum(self.path_frequencies.get(p, 0) for p in top_paths) / 
                         total_freq if total_freq > 0 else 0.5)
            
            return PathPrediction(
                predicted_paths=top_paths,
                confidence=confidence,
                strategy=self.strategy,
                reasoning=f"Based on {total_freq} historical executions"
            )
        
        elif self.strategy == PredictionStrategy.PATTERN_BASED:
            # Predict based on previous path
            if self.last_path and self.last_path in self.transition_matrix:
                transitions = self.transition_matrix[self.last_path]
                sorted_next = sorted(
                    available_paths,
                    key=lambda p: transitions.get(p, 0),
                    reverse=True
                )
                
                top_paths = sorted_next[:num_predictions]
                total_trans = sum(transitions.values())
                confidence = (sum(transitions.get(p, 0) for p in top_paths) /
                            total_trans if total_trans > 0 else 0.3)
                
                return PathPrediction(
                    predicted_paths=top_paths,
                    confidence=confidence,
                    strategy=self.strategy,
                    reasoning=f"Based on transitions from {self.last_path}"
                )
            else:
                # Fall back to frequency
                return self.predict_paths(available_paths, context, num_predictions)
        
        elif self.strategy == PredictionStrategy.HEURISTIC:
            # Use context-based heuristics
            predicted = []
            
            # Check for hints in context
            if "priority" in context:
                priority = context["priority"]
                if priority == "high" and "fast_path" in available_paths:
                    predicted.append("fast_path")
            
            if "user_type" in context:
                user_type = context["user_type"]
                if user_type == "premium" and "premium_path" in available_paths:
                    predicted.append("premium_path")
            
            # Fill remaining with most frequent
            remaining = [p for p in available_paths if p not in predicted]
            sorted_remaining = sorted(
                remaining,
                key=lambda p: self.path_frequencies.get(p, 0),
                reverse=True
            )
            predicted.extend(sorted_remaining[:num_predictions - len(predicted)])
            
            return PathPrediction(
                predicted_paths=predicted[:num_predictions],
                confidence=0.7,
                strategy=self.strategy,
                reasoning="Context-based heuristics"
            )
        
        # Default: return most frequent
        return self.predict_paths(available_paths, context, num_predictions)


class SpeculativeExecutor:
    """Manages speculative execution of multiple paths"""
    
    def __init__(self, max_parallel: int = 3):
        self.max_parallel = max_parallel
        self.active_executions: Dict[str, ExecutionPath] = {}
        self.execution_threads: Dict[str, threading.Thread] = {}
        self.execution_lock = threading.Lock()
        self.stats = {
            "total_speculative": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "wasted_computations": 0,
            "time_saved": 0.0,
            "time_wasted": 0.0
        }
    
    def execute_speculatively(self, paths: List[ExecutionPath]) -> None:
        """Start speculative execution of multiple paths"""
        
        # Sort by probability and priority
        sorted_paths = sorted(
            paths,
            key=lambda p: (p.probability, p.priority),
            reverse=True
        )
        
        # Execute top paths up to max_parallel
        for path in sorted_paths[:self.max_parallel]:
            self._start_execution(path)
            self.stats["total_speculative"] += 1
    
    def _start_execution(self, path: ExecutionPath) -> None:
        """Start execution of a single path"""
        
        def execute():
            with self.execution_lock:
                if path.state == ExecutionState.CANCELLED:
                    return
                
                path.state = ExecutionState.RUNNING
                path.started_at = datetime.now()
            
            start_time = time.time()
            
            try:
                result = path.computation()
                
                with self.execution_lock:
                    if path.state != ExecutionState.CANCELLED:
                        path.result = result
                        path.execution_time = time.time() - start_time
                        path.completed_at = datetime.now()
                        path.state = ExecutionState.COMPLETED
            except Exception as e:
                with self.execution_lock:
                    path.state = ExecutionState.CANCELLED
                    path.result = None
        
        thread = threading.Thread(target=execute, daemon=True)
        
        with self.execution_lock:
            self.active_executions[path.path_id] = path
            self.execution_threads[path.path_id] = thread
        
        thread.start()
    
    def get_result(self, path_id: str, timeout: float = 5.0) -> Optional[SpeculativeResult]:
        """Get result from a specific path"""
        
        if path_id not in self.active_executions:
            return None
        
        path = self.active_executions[path_id]
        thread = self.execution_threads.get(path_id)
        
        if thread and thread.is_alive():
            # Wait for completion
            thread.join(timeout=timeout)
        
        with self.execution_lock:
            if path.state == ExecutionState.COMPLETED:
                path.state = ExecutionState.USED
                
                # Mark other paths as wasted
                for other_id, other_path in self.active_executions.items():
                    if other_id != path_id and other_path.state == ExecutionState.COMPLETED:
                        other_path.state = ExecutionState.WASTED
                        self.stats["wasted_computations"] += 1
                        if other_path.execution_time:
                            self.stats["time_wasted"] += other_path.execution_time
                
                self.stats["correct_predictions"] += 1
                
                return SpeculativeResult(
                    path_id=path_id,
                    result=path.result,
                    execution_time=path.execution_time or 0.0,
                    was_speculative=True,
                    prediction_correct=True
                )
            else:
                self.stats["incorrect_predictions"] += 1
                return None
    
    def cancel_all(self) -> None:
        """Cancel all active executions"""
        with self.execution_lock:
            for path in self.active_executions.values():
                if path.state in [ExecutionState.PENDING, ExecutionState.RUNNING]:
                    path.state = ExecutionState.CANCELLED
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total_predictions = self.stats["correct_predictions"] + self.stats["incorrect_predictions"]
        accuracy = (self.stats["correct_predictions"] / total_predictions 
                   if total_predictions > 0 else 0)
        
        waste_ratio = (self.stats["wasted_computations"] / 
                      max(self.stats["total_speculative"], 1))
        
        return {
            "total_speculative_executions": self.stats["total_speculative"],
            "correct_predictions": self.stats["correct_predictions"],
            "incorrect_predictions": self.stats["incorrect_predictions"],
            "prediction_accuracy": round(accuracy, 3),
            "wasted_computations": self.stats["wasted_computations"],
            "waste_ratio": round(waste_ratio, 3),
            "time_saved": round(self.stats["time_saved"], 3),
            "time_wasted": round(self.stats["time_wasted"], 3)
        }


class CostBenefitAnalyzer:
    """Analyzes cost-benefit of speculative execution"""
    
    def __init__(self):
        self.analyses: List[Dict[str, Any]] = []
    
    def analyze(self, paths: List[ExecutionPath], prediction: PathPrediction) -> Dict[str, Any]:
        """Analyze if speculative execution is beneficial"""
        
        # Calculate expected cost
        total_cost = sum(p.cost_estimate * p.probability for p in paths)
        
        # Calculate potential benefit (reduced latency)
        max_latency = max(p.cost_estimate for p in paths)
        expected_latency_savings = max_latency * prediction.confidence
        
        # Decision
        should_speculate = (prediction.confidence > 0.5 and 
                          expected_latency_savings > total_cost * 0.3)
        
        analysis = {
            "total_cost": round(total_cost, 3),
            "max_latency": round(max_latency, 3),
            "expected_savings": round(expected_latency_savings, 3),
            "prediction_confidence": round(prediction.confidence, 3),
            "should_speculate": should_speculate,
            "reasoning": self._generate_reasoning(should_speculate, prediction.confidence, total_cost)
        }
        
        self.analyses.append(analysis)
        return analysis
    
    def _generate_reasoning(self, should_speculate: bool, confidence: float, cost: float) -> str:
        """Generate reasoning for decision"""
        if should_speculate:
            if confidence > 0.8:
                return "High confidence prediction justifies speculative execution"
            elif cost < 5.0:
                return "Low cost makes speculative execution worthwhile"
            else:
                return "Expected latency savings outweigh costs"
        else:
            if confidence < 0.5:
                return "Low prediction confidence makes speculation risky"
            else:
                return "High cost relative to expected benefit"


class SpeculativeExecutionAgent:
    """
    Main agent for speculative execution pattern
    
    Responsibilities:
    - Predict likely execution paths
    - Execute paths speculatively in parallel
    - Select correct result when decision is made
    - Analyze cost-benefit
    - Learn from execution patterns
    """
    
    def __init__(self, 
                 prediction_strategy: PredictionStrategy = PredictionStrategy.FREQUENCY_BASED,
                 max_parallel: int = 3):
        self.predictor = PathPredictor(prediction_strategy)
        self.executor = SpeculativeExecutor(max_parallel)
        self.analyzer = CostBenefitAnalyzer()
        self.paths: Dict[str, ExecutionPath] = {}
    
    def register_path(self, 
                     name: str,
                     computation: Callable[[], Any],
                     probability: float = 0.5,
                     priority: int = 1,
                     cost_estimate: float = 1.0) -> str:
        """Register an execution path"""
        
        path_id = self._generate_id()
        
        path = ExecutionPath(
            path_id=path_id,
            name=name,
            computation=computation,
            probability=probability,
            priority=priority,
            cost_estimate=cost_estimate
        )
        
        self.paths[path_id] = path
        print(f"✓ Registered path: {name} (prob: {probability:.2f}, cost: {cost_estimate:.1f})")
        
        return path_id
    
    def execute_with_speculation(self,
                                context: Dict[str, Any],
                                actual_path_selector: Callable[[Dict[str, Any]], str],
                                num_speculative: int = 3) -> SpeculativeResult:
        """Execute with speculative parallelization"""
        
        # Get available paths
        available_paths = list(self.paths.keys())
        
        # Predict likely paths
        prediction = self.predictor.predict_paths(
            available_paths,
            context,
            num_speculative
        )
        
        print(f"\n📊 Prediction: {len(prediction.predicted_paths)} paths")
        print(f"   Confidence: {prediction.confidence:.2f}")
        print(f"   Strategy: {prediction.strategy.value}")
        print(f"   Reasoning: {prediction.reasoning}")
        
        # Analyze cost-benefit
        predicted_path_objs = [self.paths[pid] for pid in prediction.predicted_paths 
                               if pid in self.paths]
        analysis = self.analyzer.analyze(predicted_path_objs, prediction)
        
        print(f"\n💰 Cost-Benefit Analysis:")
        print(f"   Should speculate: {analysis['should_speculate']}")
        print(f"   Expected cost: {analysis['total_cost']}")
        print(f"   Expected savings: {analysis['expected_savings']}")
        print(f"   Reasoning: {analysis['reasoning']}")
        
        if analysis['should_speculate']:
            # Start speculative execution
            print(f"\n⚡ Starting speculative execution...")
            self.executor.execute_speculatively(predicted_path_objs)
            
            # Small delay to let speculation get ahead
            time.sleep(0.01)
        
        # Make actual decision
        print(f"\n🎯 Making actual decision...")
        start_decision_time = time.time()
        actual_path_id = actual_path_selector(context)
        
        # Get result
        if actual_path_id in self.paths:
            result = self.executor.get_result(actual_path_id, timeout=5.0)
            
            if result:
                # Speculative execution was successful
                print(f"✓ Used speculative result")
                
                # Record for learning
                self.predictor.record_execution(actual_path_id, context)
                
                return result
            else:
                # Need to execute now (speculation missed or not used)
                print(f"⚠ Speculation missed, executing now...")
                actual_path = self.paths[actual_path_id]
                actual_start = time.time()
                actual_result = actual_path.computation()
                actual_time = time.time() - actual_start
                
                # Record for learning
                self.predictor.record_execution(actual_path_id, context)
                
                return SpeculativeResult(
                    path_id=actual_path_id,
                    result=actual_result,
                    execution_time=actual_time,
                    was_speculative=False,
                    prediction_correct=False
                )
        
        raise ValueError(f"Path {actual_path_id} not found")
    
    def execute_without_speculation(self,
                                   context: Dict[str, Any],
                                   path_selector: Callable[[Dict[str, Any]], str]) -> SpeculativeResult:
        """Execute without speculation (baseline)"""
        
        path_id = path_selector(context)
        
        if path_id not in self.paths:
            raise ValueError(f"Path {path_id} not found")
        
        path = self.paths[path_id]
        start_time = time.time()
        result = path.computation()
        execution_time = time.time() - start_time
        
        return SpeculativeResult(
            path_id=path_id,
            result=result,
            execution_time=execution_time,
            was_speculative=False,
            prediction_correct=False
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        exec_stats = self.executor.get_statistics()
        
        return {
            "registered_paths": len(self.paths),
            "execution_history": len(self.predictor.execution_history),
            **exec_stats,
            "analyses_performed": len(self.analyzer.analyses)
        }
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        import random
        return hashlib.md5(f"{datetime.now()}{random.random()}".encode()).hexdigest()[:12]


def demonstrate_speculative_execution():
    """Demonstrate the speculative execution pattern"""
    
    print("=" * 60)
    print("Speculative Execution Pattern Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = SpeculativeExecutionAgent(
        prediction_strategy=PredictionStrategy.FREQUENCY_BASED,
        max_parallel=3
    )
    
    # Register execution paths
    print("\n1. Registering Execution Paths")
    print("-" * 60)
    
    def fast_path():
        time.sleep(0.05)
        return "Fast result"
    
    def standard_path():
        time.sleep(0.1)
        return "Standard result"
    
    def slow_path():
        time.sleep(0.2)
        return "Slow result"
    
    def premium_path():
        time.sleep(0.03)
        return "Premium result"
    
    fast_id = agent.register_path("fast_path", fast_path, 0.6, 3, 2.0)
    standard_id = agent.register_path("standard_path", standard_path, 0.3, 2, 3.0)
    slow_id = agent.register_path("slow_path", slow_path, 0.1, 1, 5.0)
    premium_id = agent.register_path("premium_path", premium_path, 0.5, 4, 1.5)
    
    # Scenario 1: Execute with speculation (frequently used path)
    print("\n2. Execution with Speculation - Frequent Path")
    print("-" * 60)
    
    context1 = {"user_type": "standard", "priority": "medium"}
    
    result1 = agent.execute_with_speculation(
        context1,
        lambda ctx: fast_id,
        num_speculative=2
    )
    
    print(f"\n✅ Result: {result1.result}")
    print(f"   Execution time: {result1.execution_time:.3f}s")
    print(f"   Was speculative: {result1.was_speculative}")
    print(f"   Prediction correct: {result1.prediction_correct}")
    
    # Build history
    print("\n3. Building Execution History")
    print("-" * 60)
    
    for i in range(5):
        context = {"user_type": "standard", "iteration": i}
        result = agent.execute_with_speculation(
            context,
            lambda ctx: fast_id if i < 4 else standard_id,
            num_speculative=2
        )
        print(f"  Iteration {i+1}: {result.path_id[:8]}... ({result.execution_time:.3f}s)")
    
    # Scenario 2: Pattern-based prediction
    print("\n4. Pattern-Based Prediction")
    print("-" * 60)
    
    agent_pattern = SpeculativeExecutionAgent(
        prediction_strategy=PredictionStrategy.PATTERN_BASED,
        max_parallel=3
    )
    
    # Register same paths
    p1 = agent_pattern.register_path("path1", lambda: "P1", 0.5, 1, 1.0)
    p2 = agent_pattern.register_path("path2", lambda: "P2", 0.5, 1, 1.0)
    
    # Build pattern: path1 -> path2 -> path1 -> path2
    for i in range(4):
        path_id = p1 if i % 2 == 0 else p2
        result = agent_pattern.execute_with_speculation(
            {"step": i},
            lambda ctx: path_id,
            num_speculative=1
        )
    
    # Scenario 3: Compare with and without speculation
    print("\n5. Performance Comparison")
    print("-" * 60)
    
    print("\nWith speculation:")
    start_spec = time.time()
    result_spec = agent.execute_with_speculation(
        {"user_type": "standard"},
        lambda ctx: fast_id,
        num_speculative=2
    )
    time_spec = time.time() - start_spec
    print(f"Total time: {time_spec:.3f}s")
    
    print("\nWithout speculation:")
    start_no_spec = time.time()
    result_no_spec = agent.execute_without_speculation(
        {"user_type": "standard"},
        lambda ctx: fast_id
    )
    time_no_spec = time.time() - start_no_spec
    print(f"Total time: {time_no_spec:.3f}s")
    
    speedup = time_no_spec / time_spec if time_spec > 0 else 1
    print(f"\n📈 Speedup: {speedup:.2f}x")
    
    # Statistics
    print("\n6. Overall Statistics")
    print("-" * 60)
    
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_speculative_execution()
