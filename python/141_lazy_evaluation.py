"""
Agentic Design Pattern: Lazy Evaluation

This pattern implements lazy evaluation where computations are delayed until their
results are actually needed. The agent defers expensive operations, chains computations,
and only executes when values are required.

Category: Performance Optimization
Use Cases:
- Expensive computation pipelines
- Conditional logic optimization
- Resource-constrained environments
- Query optimization
- Data processing pipelines
- Just-in-time computation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Generic, TypeVar
from enum import Enum
from datetime import datetime
import hashlib
import time

T = TypeVar('T')


class EvaluationState(Enum):
    """State of lazy evaluation"""
    PENDING = "pending"
    EVALUATING = "evaluating"
    COMPUTED = "computed"
    CACHED = "cached"
    ERROR = "error"


class ComputationType(Enum):
    """Types of computations"""
    SIMPLE = "simple"
    EXPENSIVE = "expensive"
    IO_BOUND = "io_bound"
    CPU_BOUND = "cpu_bound"
    MEMORY_INTENSIVE = "memory_intensive"


@dataclass
class LazyValue(Generic[T]):
    """Represents a lazily evaluated value"""
    computation_id: str
    computation: Callable[[], T]
    state: EvaluationState = EvaluationState.PENDING
    result: Optional[T] = None
    error: Optional[str] = None
    computation_time: Optional[float] = None
    evaluated_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self) -> T:
        """Force evaluation of the lazy value"""
        if self.state == EvaluationState.COMPUTED or self.state == EvaluationState.CACHED:
            return self.result
        
        self.state = EvaluationState.EVALUATING
        start_time = time.time()
        
        try:
            self.result = self.computation()
            self.state = EvaluationState.COMPUTED
            self.computation_time = time.time() - start_time
            self.evaluated_at = datetime.now()
            return self.result
        except Exception as e:
            self.state = EvaluationState.ERROR
            self.error = str(e)
            raise
    
    def is_evaluated(self) -> bool:
        """Check if value has been evaluated"""
        return self.state in [EvaluationState.COMPUTED, EvaluationState.CACHED]


@dataclass
class ComputationNode:
    """Node in computation graph"""
    node_id: str
    name: str
    computation_type: ComputationType
    lazy_value: LazyValue
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)


@dataclass
class EvaluationPlan:
    """Plan for evaluating lazy computations"""
    plan_id: str
    required_nodes: List[str]
    evaluation_order: List[str]
    estimated_cost: float
    parallel_groups: List[List[str]] = field(default_factory=list)


class ComputationGraph:
    """Manages dependencies between lazy computations"""
    
    def __init__(self):
        self.nodes: Dict[str, ComputationNode] = {}
        self.evaluation_history: List[Dict[str, Any]] = []
    
    def add_node(self, node: ComputationNode) -> None:
        """Add a computation node"""
        self.nodes[node.node_id] = node
        
        # Update dependencies
        for dep_id in node.dependencies:
            if dep_id in self.nodes:
                self.nodes[dep_id].dependents.append(node.node_id)
    
    def get_evaluation_order(self, target_node: str) -> List[str]:
        """Get topological order for evaluating a node"""
        if target_node not in self.nodes:
            return []
        
        visited = set()
        order = []
        
        def dfs(node_id: str):
            if node_id in visited:
                return
            
            visited.add(node_id)
            node = self.nodes[node_id]
            
            # Visit dependencies first
            for dep_id in node.dependencies:
                if dep_id in self.nodes:
                    dfs(dep_id)
            
            order.append(node_id)
        
        dfs(target_node)
        return order
    
    def find_parallel_groups(self, order: List[str]) -> List[List[str]]:
        """Identify nodes that can be evaluated in parallel"""
        groups = []
        remaining = set(order)
        
        while remaining:
            # Find nodes with no dependencies in remaining set
            group = []
            for node_id in remaining:
                node = self.nodes[node_id]
                if not any(dep in remaining for dep in node.dependencies):
                    group.append(node_id)
            
            if group:
                groups.append(group)
                remaining -= set(group)
            else:
                break  # Circular dependency or error
        
        return groups
    
    def estimate_cost(self, nodes: List[str]) -> float:
        """Estimate computational cost"""
        cost = 0.0
        cost_map = {
            ComputationType.SIMPLE: 1.0,
            ComputationType.EXPENSIVE: 10.0,
            ComputationType.IO_BOUND: 5.0,
            ComputationType.CPU_BOUND: 8.0,
            ComputationType.MEMORY_INTENSIVE: 7.0
        }
        
        for node_id in nodes:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                cost += cost_map.get(node.computation_type, 5.0)
        
        return cost


class LazyEvaluator:
    """Manages lazy evaluation of computations"""
    
    def __init__(self, computation_graph: ComputationGraph):
        self.graph = computation_graph
        self.cache: Dict[str, Any] = {}
        self.stats = {
            "total_evaluations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_computation_time": 0.0
        }
    
    def evaluate(self, node_id: str, use_cache: bool = True) -> Any:
        """Evaluate a lazy computation"""
        
        # Check cache
        if use_cache and node_id in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[node_id]
        
        self.stats["cache_misses"] += 1
        
        if node_id not in self.graph.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        node = self.graph.nodes[node_id]
        
        # Get evaluation order
        order = self.graph.get_evaluation_order(node_id)
        
        # Evaluate dependencies first
        for dep_id in order[:-1]:  # All except target
            if dep_id in self.graph.nodes:
                dep_node = self.graph.nodes[dep_id]
                if not dep_node.lazy_value.is_evaluated():
                    result = dep_node.lazy_value.evaluate()
                    if use_cache:
                        self.cache[dep_id] = result
        
        # Evaluate target
        result = node.lazy_value.evaluate()
        
        if use_cache:
            self.cache[node_id] = result
        
        self.stats["total_evaluations"] += 1
        if node.lazy_value.computation_time:
            self.stats["total_computation_time"] += node.lazy_value.computation_time
        
        # Record evaluation
        self.graph.evaluation_history.append({
            "node_id": node_id,
            "timestamp": datetime.now(),
            "computation_time": node.lazy_value.computation_time,
            "dependencies_evaluated": len(order) - 1
        })
        
        return result
    
    def create_evaluation_plan(self, node_id: str) -> EvaluationPlan:
        """Create an optimized evaluation plan"""
        order = self.graph.get_evaluation_order(node_id)
        parallel_groups = self.graph.find_parallel_groups(order)
        cost = self.graph.estimate_cost(order)
        
        return EvaluationPlan(
            plan_id=self._generate_id(),
            required_nodes=order,
            evaluation_order=order,
            estimated_cost=cost,
            parallel_groups=parallel_groups
        )
    
    def clear_cache(self) -> None:
        """Clear evaluation cache"""
        self.cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        cache_total = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = self.stats["cache_hits"] / cache_total if cache_total > 0 else 0
        
        return {
            "total_evaluations": self.stats["total_evaluations"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_hit_rate": round(cache_hit_rate, 3),
            "total_computation_time": round(self.stats["total_computation_time"], 3),
            "cached_values": len(self.cache)
        }
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        import random
        return hashlib.md5(f"{datetime.now()}{random.random()}".encode()).hexdigest()[:12]


class ConditionalEvaluator:
    """Handles conditional lazy evaluation"""
    
    def __init__(self):
        self.branches: Dict[str, LazyValue] = {}
        self.conditions: Dict[str, Callable[[], bool]] = {}
    
    def add_branch(self, branch_id: str, condition: Callable[[], bool], 
                   computation: Callable[[], Any]) -> None:
        """Add a conditional branch"""
        self.conditions[branch_id] = condition
        self.branches[branch_id] = LazyValue(
            computation_id=branch_id,
            computation=computation
        )
    
    def evaluate_conditional(self) -> tuple[str, Any]:
        """Evaluate only the branch that matches condition"""
        for branch_id, condition in self.conditions.items():
            if condition():
                result = self.branches[branch_id].evaluate()
                return branch_id, result
        
        return "none", None


class PipelineOptimizer:
    """Optimizes lazy evaluation pipelines"""
    
    def __init__(self, graph: ComputationGraph):
        self.graph = graph
    
    def identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities to optimize evaluation"""
        opportunities = []
        
        # Find nodes that are never evaluated
        never_evaluated = []
        for node_id, node in self.graph.nodes.items():
            if not node.lazy_value.is_evaluated():
                never_evaluated.append(node_id)
        
        if never_evaluated:
            opportunities.append({
                "type": "unused_computations",
                "nodes": never_evaluated,
                "potential_savings": "These computations are never used"
            })
        
        # Find expensive computations that could be cached
        expensive_uncached = []
        for node_id, node in self.graph.nodes.items():
            if (node.computation_type in [ComputationType.EXPENSIVE, 
                                          ComputationType.CPU_BOUND] and
                len(node.dependents) > 1):
                expensive_uncached.append(node_id)
        
        if expensive_uncached:
            opportunities.append({
                "type": "caching_opportunity",
                "nodes": expensive_uncached,
                "potential_savings": "Expensive computations with multiple dependents"
            })
        
        # Find parallel execution opportunities
        for node_id in self.graph.nodes:
            order = self.graph.get_evaluation_order(node_id)
            parallel_groups = self.graph.find_parallel_groups(order)
            parallel_count = sum(len(g) for g in parallel_groups if len(g) > 1)
            
            if parallel_count > 0:
                opportunities.append({
                    "type": "parallelization",
                    "node": node_id,
                    "parallel_opportunities": parallel_count,
                    "potential_savings": f"{parallel_count} computations can run in parallel"
                })
                break  # Only report once
        
        return opportunities
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        opportunities = self.identify_optimization_opportunities()
        
        total_nodes = len(self.graph.nodes)
        evaluated_nodes = sum(1 for n in self.graph.nodes.values() 
                             if n.lazy_value.is_evaluated())
        
        return {
            "total_nodes": total_nodes,
            "evaluated_nodes": evaluated_nodes,
            "unevaluated_nodes": total_nodes - evaluated_nodes,
            "optimization_opportunities": len(opportunities),
            "opportunities": opportunities
        }


class LazyEvaluationAgent:
    """
    Main agent for lazy evaluation pattern
    
    Responsibilities:
    - Defer computations until needed
    - Manage computation dependencies
    - Optimize evaluation order
    - Cache results when beneficial
    - Provide conditional evaluation
    """
    
    def __init__(self):
        self.graph = ComputationGraph()
        self.evaluator = LazyEvaluator(self.graph)
        self.conditional_evaluator = ConditionalEvaluator()
        self.optimizer = PipelineOptimizer(self.graph)
    
    def create_lazy_computation(self, 
                               name: str,
                               computation: Callable[[], Any],
                               computation_type: ComputationType = ComputationType.SIMPLE,
                               dependencies: Optional[List[str]] = None) -> str:
        """Create a lazy computation"""
        
        node_id = self._generate_id()
        
        lazy_value = LazyValue(
            computation_id=node_id,
            computation=computation,
            dependencies=dependencies or []
        )
        
        node = ComputationNode(
            node_id=node_id,
            name=name,
            computation_type=computation_type,
            lazy_value=lazy_value,
            dependencies=dependencies or []
        )
        
        self.graph.add_node(node)
        print(f"✓ Created lazy computation: {name} (ID: {node_id})")
        
        return node_id
    
    def create_lazy_chain(self, 
                         steps: List[tuple[str, Callable, ComputationType]]) -> str:
        """Create a chain of lazy computations"""
        
        node_ids = []
        
        for i, (name, computation, comp_type) in enumerate(steps):
            deps = [node_ids[-1]] if node_ids else []
            node_id = self.create_lazy_computation(name, computation, comp_type, deps)
            node_ids.append(node_id)
        
        print(f"✓ Created lazy chain with {len(steps)} steps")
        return node_ids[-1]  # Return final node
    
    def evaluate(self, node_id: str, use_cache: bool = True) -> Any:
        """Evaluate a lazy computation"""
        return self.evaluator.evaluate(node_id, use_cache)
    
    def create_evaluation_plan(self, node_id: str) -> EvaluationPlan:
        """Create evaluation plan for a computation"""
        return self.evaluator.create_evaluation_plan(node_id)
    
    def add_conditional_branch(self, branch_id: str, 
                              condition: Callable[[], bool],
                              computation: Callable[[], Any]) -> None:
        """Add a conditional lazy branch"""
        self.conditional_evaluator.add_branch(branch_id, condition, computation)
        print(f"✓ Added conditional branch: {branch_id}")
    
    def evaluate_conditional(self) -> tuple[str, Any]:
        """Evaluate conditional branches"""
        return self.conditional_evaluator.evaluate_conditional()
    
    def optimize(self) -> Dict[str, Any]:
        """Get optimization recommendations"""
        return self.optimizer.generate_optimization_report()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get lazy evaluation statistics"""
        eval_stats = self.evaluator.get_statistics()
        
        return {
            "total_computations": len(self.graph.nodes),
            "evaluated_computations": sum(1 for n in self.graph.nodes.values() 
                                         if n.lazy_value.is_evaluated()),
            **eval_stats,
            "evaluation_history_entries": len(self.graph.evaluation_history)
        }
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        import random
        return hashlib.md5(f"{datetime.now()}{random.random()}".encode()).hexdigest()[:12]


def demonstrate_lazy_evaluation():
    """Demonstrate the lazy evaluation pattern"""
    
    print("=" * 60)
    print("Lazy Evaluation Pattern Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = LazyEvaluationAgent()
    
    # Scenario 1: Simple lazy computation
    print("\n1. Creating Lazy Computations")
    print("-" * 60)
    
    # Simulate expensive computation
    def expensive_computation():
        time.sleep(0.1)  # Simulate work
        return "Expensive result"
    
    def cheap_computation():
        return "Cheap result"
    
    lazy1 = agent.create_lazy_computation(
        name="Expensive Operation",
        computation=expensive_computation,
        computation_type=ComputationType.EXPENSIVE
    )
    
    lazy2 = agent.create_lazy_computation(
        name="Cheap Operation",
        computation=cheap_computation,
        computation_type=ComputationType.SIMPLE
    )
    
    print("✓ Computations created but not yet evaluated")
    
    # Scenario 2: Create computation chain
    print("\n2. Creating Lazy Computation Chain")
    print("-" * 60)
    
    counter = {"value": 0}
    
    def step1():
        counter["value"] += 1
        return f"Step 1 complete (count: {counter['value']})"
    
    def step2():
        counter["value"] += 10
        return f"Step 2 complete (count: {counter['value']})"
    
    def step3():
        counter["value"] += 100
        return f"Step 3 complete (count: {counter['value']})"
    
    chain_end = agent.create_lazy_chain([
        ("Step 1", step1, ComputationType.SIMPLE),
        ("Step 2", step2, ComputationType.SIMPLE),
        ("Step 3", step3, ComputationType.SIMPLE)
    ])
    
    print(f"Counter value before evaluation: {counter['value']}")
    
    # Scenario 3: Evaluation plan
    print("\n3. Creating Evaluation Plan")
    print("-" * 60)
    
    plan = agent.create_evaluation_plan(chain_end)
    print(f"Plan ID: {plan.plan_id}")
    print(f"Required nodes: {len(plan.required_nodes)}")
    print(f"Evaluation order: {len(plan.evaluation_order)} steps")
    print(f"Estimated cost: {plan.estimated_cost}")
    print(f"Parallel groups: {len(plan.parallel_groups)}")
    for i, group in enumerate(plan.parallel_groups):
        print(f"  Group {i+1}: {len(group)} nodes can run in parallel")
    
    # Scenario 4: Force evaluation
    print("\n4. Forcing Evaluation")
    print("-" * 60)
    
    print("\nEvaluating chain...")
    start_time = time.time()
    result = agent.evaluate(chain_end)
    eval_time = time.time() - start_time
    
    print(f"Result: {result}")
    print(f"Evaluation time: {eval_time:.3f}s")
    print(f"Counter value after evaluation: {counter['value']}")
    
    # Scenario 5: Cached evaluation
    print("\n5. Cached Re-evaluation")
    print("-" * 60)
    
    print("\nRe-evaluating chain (should use cache)...")
    start_time = time.time()
    result2 = agent.evaluate(chain_end, use_cache=True)
    cached_time = time.time() - start_time
    
    print(f"Result: {result2}")
    print(f"Evaluation time: {cached_time:.3f}s")
    print(f"Speedup: {eval_time/cached_time:.1f}x faster")
    
    # Scenario 6: Conditional evaluation
    print("\n6. Conditional Lazy Evaluation")
    print("-" * 60)
    
    condition_flag = {"use_expensive": False}
    
    agent.add_conditional_branch(
        "cheap_path",
        lambda: not condition_flag["use_expensive"],
        lambda: "Cheap path result"
    )
    
    agent.add_conditional_branch(
        "expensive_path",
        lambda: condition_flag["use_expensive"],
        expensive_computation
    )
    
    print("\nEvaluating with cheap path condition:")
    branch_id, branch_result = agent.evaluate_conditional()
    print(f"Selected branch: {branch_id}")
    print(f"Result: {branch_result}")
    
    condition_flag["use_expensive"] = True
    print("\nEvaluating with expensive path condition:")
    branch_id, branch_result = agent.evaluate_conditional()
    print(f"Selected branch: {branch_id}")
    print(f"Result: {branch_result}")
    
    # Scenario 7: Optimization analysis
    print("\n7. Optimization Analysis")
    print("-" * 60)
    
    # Create some unevaluated nodes
    unused1 = agent.create_lazy_computation(
        "Unused Computation 1",
        lambda: "Never used",
        ComputationType.EXPENSIVE
    )
    
    unused2 = agent.create_lazy_computation(
        "Unused Computation 2",
        lambda: "Also never used",
        ComputationType.CPU_BOUND
    )
    
    optimization = agent.optimize()
    print(f"Total computations: {optimization['total_nodes']}")
    print(f"Evaluated: {optimization['evaluated_nodes']}")
    print(f"Unevaluated: {optimization['unevaluated_nodes']}")
    print(f"\nOptimization opportunities found: {optimization['optimization_opportunities']}")
    
    for opp in optimization['opportunities']:
        print(f"\n  Type: {opp['type']}")
        if 'nodes' in opp:
            print(f"  Affected nodes: {len(opp['nodes'])}")
        print(f"  Potential savings: {opp['potential_savings']}")
    
    # Statistics
    print("\n8. Overall Statistics")
    print("-" * 60)
    
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_lazy_evaluation()
