"""
Agentic Design Pattern: Workflow Coordination Agent

This pattern implements an agent that orchestrates multi-agent workflows with
dependency resolution, progress monitoring, failure recovery, and parallel execution.

Key Components:
1. WorkflowNode - Represents a step in the workflow
2. WorkflowGraph - DAG of workflow steps with dependencies
3. ExecutionMonitor - Tracks execution progress and status
4. FailureRecovery - Handles failures with retry and recovery
5. WorkflowCoordinationAgent - Main orchestrator

Features:
- DAG-based workflow definition
- Dependency resolution and ordering
- Parallel execution of independent tasks
- Progress monitoring and status tracking
- Failure handling with retries
- Dynamic workflow modification
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from enum import Enum
from collections import defaultdict, deque
import random
import time


class NodeStatus(Enum):
    """Status of workflow nodes."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(Enum):
    """Overall workflow status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class FailureStrategy(Enum):
    """Strategies for handling failures."""
    RETRY = "retry"
    SKIP = "skip"
    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"
    ROLLBACK = "rollback"


class ExecutionMode(Enum):
    """Workflow execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"  # Mix of sequential and parallel


@dataclass
class NodeResult:
    """Result of node execution."""
    node_id: str
    status: NodeStatus
    output: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowNode:
    """Represents a step in the workflow."""
    node_id: str
    name: str
    agent_id: str  # Agent responsible for this node
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    failure_strategy: FailureStrategy = FailureStrategy.RETRY
    max_retries: int = 3
    timeout: float = 60.0
    
    # Execution state
    status: NodeStatus = NodeStatus.PENDING
    result: Optional[NodeResult] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    # For simulation
    execution_func: Optional[Callable] = None
    failure_probability: float = 0.0  # For testing


@dataclass
class WorkflowEdge:
    """Represents a dependency between nodes."""
    from_node: str
    to_node: str
    condition: Optional[Callable] = None  # Optional conditional execution
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    workflow_id: str
    name: str
    nodes: Dict[str, WorkflowNode]
    edges: List[WorkflowEdge]
    execution_mode: ExecutionMode = ExecutionMode.HYBRID
    max_parallel: int = 4
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    """Context passed between workflow nodes."""
    workflow_id: str
    shared_data: Dict[str, Any] = field(default_factory=dict)
    node_outputs: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)


class WorkflowGraph:
    """Manages workflow DAG structure and dependencies."""
    
    def __init__(self, definition: WorkflowDefinition):
        self.definition = definition
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)
        self._build_graph()
        
    def _build_graph(self):
        """Build adjacency lists from edges."""
        for edge in self.definition.edges:
            self.adjacency[edge.from_node].append(edge.to_node)
            self.reverse_adjacency[edge.to_node].append(edge.from_node)
    
    def get_dependencies(self, node_id: str) -> List[str]:
        """Get all dependencies of a node."""
        return self.reverse_adjacency.get(node_id, [])
    
    def get_dependents(self, node_id: str) -> List[str]:
        """Get all nodes that depend on this node."""
        return self.adjacency.get(node_id, [])
    
    def topological_sort(self) -> List[str]:
        """
        Perform topological sort to get execution order.
        
        Returns:
            List of node IDs in topological order
        """
        in_degree = {node_id: 0 for node_id in self.definition.nodes}
        
        for node_id in self.definition.nodes:
            for dependent in self.adjacency[node_id]:
                in_degree[dependent] += 1
        
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            for dependent in self.adjacency[node_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) != len(self.definition.nodes):
            raise ValueError("Workflow contains cycles!")
        
        return result
    
    def get_ready_nodes(self, completed_nodes: Set[str]) -> List[str]:
        """
        Get nodes that are ready to execute.
        
        A node is ready if:
        1. It's in PENDING status
        2. All its dependencies are completed
        """
        ready = []
        
        for node_id, node in self.definition.nodes.items():
            if node.status != NodeStatus.PENDING:
                continue
            
            dependencies = self.get_dependencies(node_id)
            
            if all(dep in completed_nodes for dep in dependencies):
                ready.append(node_id)
        
        return ready
    
    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the workflow graph."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: str, path: List[str]):
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            for dependent in self.adjacency[node_id]:
                if dependent not in visited:
                    dfs(dependent, path.copy())
                elif dependent in rec_stack:
                    # Found cycle
                    cycle_start = path.index(dependent)
                    cycles.append(path[cycle_start:] + [dependent])
            
            rec_stack.remove(node_id)
        
        for node_id in self.definition.nodes:
            if node_id not in visited:
                dfs(node_id, [])
        
        return cycles
    
    def get_execution_levels(self) -> List[List[str]]:
        """
        Get nodes grouped by execution level (for parallel execution).
        
        Returns:
            List of levels, where each level is a list of node IDs
            that can be executed in parallel
        """
        levels = []
        processed = set()
        
        while len(processed) < len(self.definition.nodes):
            # Find nodes with all dependencies processed
            current_level = []
            
            for node_id in self.definition.nodes:
                if node_id in processed:
                    continue
                
                dependencies = self.get_dependencies(node_id)
                if all(dep in processed for dep in dependencies):
                    current_level.append(node_id)
            
            if not current_level:
                # Shouldn't happen if no cycles
                break
            
            levels.append(current_level)
            processed.update(current_level)
        
        return levels


class ExecutionMonitor:
    """Monitors workflow execution progress."""
    
    def __init__(self):
        self.execution_history: List[NodeResult] = []
        self.metrics: Dict[str, Any] = {
            'total_nodes': 0,
            'completed_nodes': 0,
            'failed_nodes': 0,
            'skipped_nodes': 0,
            'total_execution_time': 0.0,
            'parallel_efficiency': 0.0
        }
        
    def record_result(self, result: NodeResult):
        """Record node execution result."""
        self.execution_history.append(result)
        
        # Update metrics
        if result.status == NodeStatus.COMPLETED:
            self.metrics['completed_nodes'] += 1
        elif result.status == NodeStatus.FAILED:
            self.metrics['failed_nodes'] += 1
        elif result.status == NodeStatus.SKIPPED:
            self.metrics['skipped_nodes'] += 1
        
        self.metrics['total_execution_time'] += result.execution_time
    
    def get_progress(self, total_nodes: int) -> Dict[str, Any]:
        """Get current execution progress."""
        completed = self.metrics['completed_nodes']
        failed = self.metrics['failed_nodes']
        skipped = self.metrics['skipped_nodes']
        processed = completed + failed + skipped
        
        return {
            'total': total_nodes,
            'completed': completed,
            'failed': failed,
            'skipped': skipped,
            'processed': processed,
            'remaining': total_nodes - processed,
            'progress_percent': (processed / total_nodes * 100) if total_nodes > 0 else 0,
            'success_rate': (completed / processed * 100) if processed > 0 else 0
        }
    
    def get_node_metrics(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific node."""
        results = [r for r in self.execution_history if r.node_id == node_id]
        
        if not results:
            return None
        
        latest = results[-1]
        return {
            'status': latest.status.value,
            'execution_time': latest.execution_time,
            'retry_count': latest.retry_count,
            'total_attempts': len(results),
            'output_size': len(str(latest.output)) if latest.output else 0
        }


class FailureRecovery:
    """Handles failure recovery strategies."""
    
    def __init__(self):
        self.retry_counts: Dict[str, int] = defaultdict(int)
        self.recovery_actions: List[Dict[str, Any]] = []
        
    def should_retry(self, node: WorkflowNode, error: str) -> bool:
        """Determine if node should be retried."""
        if node.failure_strategy != FailureStrategy.RETRY:
            return False
        
        current_retries = self.retry_counts[node.node_id]
        return current_retries < node.max_retries
    
    def handle_failure(
        self,
        node: WorkflowNode,
        error: str,
        context: ExecutionContext
    ) -> Tuple[bool, str]:
        """
        Handle node failure.
        
        Returns:
            Tuple of (should_continue, action_taken)
        """
        action = ""
        
        if node.failure_strategy == FailureStrategy.RETRY:
            if self.should_retry(node, error):
                self.retry_counts[node.node_id] += 1
                action = f"Retrying (attempt {self.retry_counts[node.node_id] + 1}/{node.max_retries + 1})"
                return True, action
            else:
                action = "Max retries exceeded, failing node"
                return False, action
        
        elif node.failure_strategy == FailureStrategy.SKIP:
            action = "Skipping node and continuing workflow"
            return True, action
        
        elif node.failure_strategy == FailureStrategy.FAIL_FAST:
            action = "Failing workflow immediately"
            return False, action
        
        elif node.failure_strategy == FailureStrategy.CONTINUE:
            action = "Marking as failed but continuing workflow"
            return True, action
        
        elif node.failure_strategy == FailureStrategy.ROLLBACK:
            action = "Rolling back previous changes"
            # Would implement rollback logic here
            return False, action
        
        return False, "Unknown failure strategy"
    
    def record_recovery(self, node_id: str, action: str, success: bool):
        """Record recovery action."""
        self.recovery_actions.append({
            'node_id': node_id,
            'action': action,
            'success': success,
            'timestamp': time.time()
        })


class WorkflowCoordinationAgent:
    """
    Main agent for workflow coordination.
    
    Orchestrates:
    - Workflow execution
    - Dependency resolution
    - Parallel execution
    - Progress monitoring
    - Failure handling
    """
    
    def __init__(self, execution_mode: ExecutionMode = ExecutionMode.HYBRID):
        self.execution_mode = execution_mode
        
        # Components
        self.graph: Optional[WorkflowGraph] = None
        self.monitor = ExecutionMonitor()
        self.recovery = FailureRecovery()
        
        # State
        self.workflow: Optional[WorkflowDefinition] = None
        self.context: Optional[ExecutionContext] = None
        self.status = WorkflowStatus.NOT_STARTED
        self.completed_nodes: Set[str] = set()
        self.running_nodes: Set[str] = set()
        
        # Statistics
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
    def load_workflow(self, workflow: WorkflowDefinition):
        """Load a workflow definition."""
        self.workflow = workflow
        self.graph = WorkflowGraph(workflow)
        self.context = ExecutionContext(workflow_id=workflow.workflow_id)
        
        # Validate workflow
        cycles = self.graph.detect_cycles()
        if cycles:
            raise ValueError(f"Workflow contains cycles: {cycles}")
        
        print(f"\n✓ Loaded workflow: {workflow.name}")
        print(f"  Nodes: {len(workflow.nodes)}")
        print(f"  Edges: {len(workflow.edges)}")
        print(f"  Execution mode: {workflow.execution_mode.value}")
    
    def execute_workflow(self) -> Dict[str, Any]:
        """
        Execute the entire workflow.
        
        Returns:
            Execution summary
        """
        if not self.workflow or not self.graph:
            raise ValueError("No workflow loaded")
        
        print(f"\n{'='*80}")
        print(f"🚀 Executing Workflow: {self.workflow.name}")
        print(f"{'='*80}\n")
        
        self.status = WorkflowStatus.RUNNING
        self.start_time = time.time()
        self.completed_nodes.clear()
        self.running_nodes.clear()
        
        try:
            if self.execution_mode == ExecutionMode.SEQUENTIAL:
                self._execute_sequential()
            elif self.execution_mode == ExecutionMode.PARALLEL:
                self._execute_parallel()
            else:  # HYBRID
                self._execute_hybrid()
            
            self.status = WorkflowStatus.COMPLETED
            print(f"\n✅ Workflow completed successfully!")
            
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            print(f"\n❌ Workflow failed: {e}")
        
        finally:
            self.end_time = time.time()
        
        return self.get_execution_summary()
    
    def _execute_sequential(self):
        """Execute workflow sequentially."""
        order = self.graph.topological_sort()
        
        for node_id in order:
            node = self.workflow.nodes[node_id]
            self._execute_node(node)
            
            if node.status == NodeStatus.FAILED:
                if node.failure_strategy == FailureStrategy.FAIL_FAST:
                    raise RuntimeError(f"Node {node_id} failed with fail-fast strategy")
    
    def _execute_parallel(self):
        """Execute workflow with maximum parallelism."""
        levels = self.graph.get_execution_levels()
        
        for level_idx, level_nodes in enumerate(levels):
            print(f"\n📊 Executing Level {level_idx + 1}/{len(levels)} ({len(level_nodes)} nodes)")
            
            # Execute all nodes in this level (simulated parallel)
            for node_id in level_nodes:
                node = self.workflow.nodes[node_id]
                self._execute_node(node)
    
    def _execute_hybrid(self):
        """Execute workflow with controlled parallelism."""
        max_parallel = self.workflow.max_parallel
        
        while len(self.completed_nodes) < len(self.workflow.nodes):
            # Get ready nodes
            ready_nodes = self.graph.get_ready_nodes(self.completed_nodes)
            
            if not ready_nodes and not self.running_nodes:
                # No nodes ready and none running - check for failures
                failed_nodes = [
                    n for n in self.workflow.nodes.values()
                    if n.status == NodeStatus.FAILED
                ]
                if failed_nodes:
                    break
                # Might be stuck
                print("⚠️  Warning: No nodes ready but workflow incomplete")
                break
            
            # Execute nodes up to max_parallel
            for node_id in ready_nodes:
                if len(self.running_nodes) >= max_parallel:
                    break
                
                node = self.workflow.nodes[node_id]
                self.running_nodes.add(node_id)
                self._execute_node(node)
                self.running_nodes.remove(node_id)
            
            # Small delay to simulate parallel execution
            time.sleep(0.1)
    
    def _execute_node(self, node: WorkflowNode) -> bool:
        """
        Execute a single node.
        
        Returns:
            True if successful, False otherwise
        """
        node.status = NodeStatus.RUNNING
        node.start_time = time.time()
        
        print(f"▶️  Executing: {node.name} (Agent: {node.agent_id})")
        
        try:
            # Simulate execution
            if node.execution_func:
                output = node.execution_func(self.context)
            else:
                output = self._simulate_execution(node)
            
            # Check for simulated failure
            if random.random() < node.failure_probability:
                raise RuntimeError(f"Simulated failure in {node.node_id}")
            
            node.end_time = time.time()
            execution_time = node.end_time - node.start_time
            
            # Success
            node.status = NodeStatus.COMPLETED
            self.completed_nodes.add(node.node_id)
            
            result = NodeResult(
                node_id=node.node_id,
                status=NodeStatus.COMPLETED,
                output=output,
                execution_time=execution_time,
                retry_count=self.recovery.retry_counts[node.node_id]
            )
            
            node.result = result
            self.monitor.record_result(result)
            self.context.node_outputs[node.node_id] = output
            
            print(f"   ✓ Completed in {execution_time:.2f}s")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ✗ Failed: {error_msg}")
            
            # Handle failure
            should_continue, action = self.recovery.handle_failure(
                node, error_msg, self.context
            )
            
            print(f"   → {action}")
            
            if action.startswith("Retrying"):
                # Reset and retry
                node.status = NodeStatus.PENDING
                return self._execute_node(node)
            
            # Mark as failed
            node.status = NodeStatus.FAILED if not should_continue else NodeStatus.SKIPPED
            node.end_time = time.time()
            
            result = NodeResult(
                node_id=node.node_id,
                status=node.status,
                error=error_msg,
                execution_time=node.end_time - node.start_time,
                retry_count=self.recovery.retry_counts[node.node_id]
            )
            
            node.result = result
            self.monitor.record_result(result)
            
            if node.status == NodeStatus.SKIPPED:
                self.completed_nodes.add(node.node_id)
            
            return should_continue
    
    def _simulate_execution(self, node: WorkflowNode) -> Any:
        """Simulate node execution."""
        # Simulate some work
        work_time = random.uniform(0.1, 0.5)
        time.sleep(work_time)
        
        # Generate output based on node type
        return {
            'node_id': node.node_id,
            'agent': node.agent_id,
            'result': f"Output from {node.name}",
            'processed_items': random.randint(10, 100)
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of workflow execution."""
        if not self.workflow:
            return {"error": "No workflow loaded"}
        
        total_time = self.end_time - self.start_time if self.start_time and self.end_time else 0
        progress = self.monitor.get_progress(len(self.workflow.nodes))
        
        # Node status breakdown
        status_counts = defaultdict(int)
        for node in self.workflow.nodes.values():
            status_counts[node.status.value] += 1
        
        # Calculate critical path
        max_path_time = 0.0
        for node in self.workflow.nodes.values():
            if node.result:
                max_path_time = max(max_path_time, node.result.execution_time)
        
        parallel_efficiency = (max_path_time / total_time * 100) if total_time > 0 else 0
        
        return {
            'workflow_id': self.workflow.workflow_id,
            'workflow_name': self.workflow.name,
            'status': self.status.value,
            'total_time': total_time,
            'progress': progress,
            'node_status': dict(status_counts),
            'completed_nodes': len(self.completed_nodes),
            'failed_nodes': self.monitor.metrics['failed_nodes'],
            'skipped_nodes': self.monitor.metrics['skipped_nodes'],
            'total_retries': sum(self.recovery.retry_counts.values()),
            'parallel_efficiency': parallel_efficiency,
            'recovery_actions': len(self.recovery.recovery_actions)
        }
    
    def visualize_workflow(self) -> str:
        """Generate text visualization of workflow."""
        if not self.workflow or not self.graph:
            return "No workflow loaded"
        
        lines = []
        lines.append("\n" + "="*80)
        lines.append(f"WORKFLOW VISUALIZATION: {self.workflow.name}")
        lines.append("="*80 + "\n")
        
        levels = self.graph.get_execution_levels()
        
        for level_idx, level_nodes in enumerate(levels):
            lines.append(f"Level {level_idx + 1}:")
            for node_id in level_nodes:
                node = self.workflow.nodes[node_id]
                status_emoji = {
                    NodeStatus.PENDING: "⏸️ ",
                    NodeStatus.READY: "⏳",
                    NodeStatus.RUNNING: "▶️ ",
                    NodeStatus.COMPLETED: "✅",
                    NodeStatus.FAILED: "❌",
                    NodeStatus.SKIPPED: "⏭️ ",
                    NodeStatus.CANCELLED: "🚫"
                }.get(node.status, "❓")
                
                deps = self.graph.get_dependencies(node_id)
                deps_str = f" (deps: {', '.join(deps)})" if deps else ""
                
                lines.append(f"  {status_emoji} {node.name}{deps_str}")
            lines.append("")
        
        return "\n".join(lines)


# Demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("WORKFLOW COORDINATION AGENT DEMONSTRATION")
    print("=" * 80)
    
    # Create workflow definition
    print("\n📋 Creating Workflow: Data Processing Pipeline")
    
    # Define nodes
    nodes = {
        'ingest': WorkflowNode(
            node_id='ingest',
            name='Data Ingestion',
            agent_id='ingestion_agent',
            dependencies=[],
            failure_strategy=FailureStrategy.RETRY,
            max_retries=2,
            failure_probability=0.0
        ),
        'validate': WorkflowNode(
            node_id='validate',
            name='Data Validation',
            agent_id='validation_agent',
            dependencies=['ingest'],
            failure_strategy=FailureStrategy.RETRY,
            max_retries=2,
            failure_probability=0.1  # 10% chance of failure for demo
        ),
        'transform': WorkflowNode(
            node_id='transform',
            name='Data Transformation',
            agent_id='transform_agent',
            dependencies=['validate'],
            failure_strategy=FailureStrategy.RETRY,
            max_retries=1,
            failure_probability=0.0
        ),
        'enrich': WorkflowNode(
            node_id='enrich',
            name='Data Enrichment',
            agent_id='enrichment_agent',
            dependencies=['validate'],  # Can run parallel with transform
            failure_strategy=FailureStrategy.SKIP,
            failure_probability=0.0
        ),
        'aggregate': WorkflowNode(
            node_id='aggregate',
            name='Data Aggregation',
            agent_id='aggregation_agent',
            dependencies=['transform', 'enrich'],
            failure_strategy=FailureStrategy.RETRY,
            max_retries=2,
            failure_probability=0.0
        ),
        'analyze': WorkflowNode(
            node_id='analyze',
            name='Data Analysis',
            agent_id='analysis_agent',
            dependencies=['aggregate'],
            failure_strategy=FailureStrategy.RETRY,
            max_retries=1,
            failure_probability=0.0
        ),
        'report': WorkflowNode(
            node_id='report',
            name='Report Generation',
            agent_id='reporting_agent',
            dependencies=['analyze'],
            failure_strategy=FailureStrategy.RETRY,
            max_retries=2,
            failure_probability=0.0
        ),
        'notify': WorkflowNode(
            node_id='notify',
            name='Notification',
            agent_id='notification_agent',
            dependencies=['report'],
            failure_strategy=FailureStrategy.CONTINUE,
            failure_probability=0.0
        )
    }
    
    # Define edges
    edges = [
        WorkflowEdge('ingest', 'validate'),
        WorkflowEdge('validate', 'transform'),
        WorkflowEdge('validate', 'enrich'),
        WorkflowEdge('transform', 'aggregate'),
        WorkflowEdge('enrich', 'aggregate'),
        WorkflowEdge('aggregate', 'analyze'),
        WorkflowEdge('analyze', 'report'),
        WorkflowEdge('report', 'notify'),
    ]
    
    workflow = WorkflowDefinition(
        workflow_id='data_pipeline_001',
        name='Data Processing Pipeline',
        nodes=nodes,
        edges=edges,
        execution_mode=ExecutionMode.HYBRID,
        max_parallel=3
    )
    
    # Create agent
    agent = WorkflowCoordinationAgent(execution_mode=ExecutionMode.HYBRID)
    
    # Load workflow
    agent.load_workflow(workflow)
    
    # Visualize workflow structure
    print(agent.visualize_workflow())
    
    # Execute workflow
    summary = agent.execute_workflow()
    
    # Display results
    print(f"\n{'='*80}")
    print(f"📊 EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Workflow: {summary['workflow_name']}")
    print(f"Status: {summary['status']}")
    print(f"Total Time: {summary['total_time']:.2f}s")
    print(f"Parallel Efficiency: {summary['parallel_efficiency']:.1f}%")
    
    print(f"\n📈 Progress:")
    progress = summary['progress']
    print(f"  Total Nodes: {progress['total']}")
    print(f"  Completed: {progress['completed']}")
    print(f"  Failed: {progress['failed']}")
    print(f"  Skipped: {progress['skipped']}")
    print(f"  Success Rate: {progress['success_rate']:.1f}%")
    
    print(f"\n📋 Node Status:")
    for status, count in summary['node_status'].items():
        print(f"  {status}: {count}")
    
    print(f"\n🔄 Recovery:")
    print(f"  Total Retries: {summary['total_retries']}")
    print(f"  Recovery Actions: {summary['recovery_actions']}")
    
    # Show final workflow state
    print(agent.visualize_workflow())
    
    print("\n" + "="*80)
    print("✅ Workflow Coordination Agent demonstration complete!")
    print("="*80)
    print("\nKey Achievements:")
    print("• DAG-based workflow orchestration")
    print("• Dependency resolution and parallel execution")
    print("• Failure handling with retry strategies")
    print("• Progress monitoring and status tracking")
    print("• Dynamic workflow coordination")
    print(f"• Successfully processed {summary['completed_nodes']}/{len(nodes)} nodes")
