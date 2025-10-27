"""
Hierarchical Task Decomposition Pattern

Breaks down complex tasks into hierarchical subtasks recursively.
Manages dependencies and execution order of subtasks.

Use Cases:
- Project planning
- Complex workflow automation
- Goal-oriented systems
- Multi-step problem solving

Advantages:
- Handles complexity systematically
- Clear task hierarchy
- Parallel execution opportunities
- Progress tracking
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class TaskStatus(Enum):
    """Status of a task"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Types of tasks"""
    ATOMIC = "atomic"  # Cannot be decomposed further
    COMPOSITE = "composite"  # Can be broken down
    SEQUENTIAL = "sequential"  # Subtasks must run in order
    PARALLEL = "parallel"  # Subtasks can run in parallel


@dataclass
class Task:
    """Represents a task in the hierarchy"""
    task_id: str
    name: str
    description: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    parent_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    estimated_effort: int = 1  # Story points or time units
    actual_effort: Optional[int] = None
    priority: int = 5  # 1-10, higher is more important
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionRule:
    """Rule for decomposing tasks"""
    rule_id: str
    pattern: str  # Task pattern to match
    subtask_templates: List[Dict[str, Any]]
    conditions: Dict[str, Any]


@dataclass
class ExecutionPlan:
    """Plan for executing tasks"""
    plan_id: str
    root_task_id: str
    execution_order: List[List[str]]  # Groups of tasks that can run in parallel
    total_tasks: int
    estimated_total_effort: int
    critical_path: List[str]


class TaskDecomposer:
    """Decomposes tasks into subtasks"""
    
    def __init__(self):
        self.decomposition_rules: Dict[str, DecompositionRule] = {}
        self._initialize_default_rules()
    
    def decompose_task(self, task: Task, context: Dict[str, Any]) -> List[Task]:
        """
        Decompose a task into subtasks.
        
        Args:
            task: Task to decompose
            context: Context information
            
        Returns:
            List of subtasks
        """
        if task.task_type == TaskType.ATOMIC:
            return []
        
        # Find matching decomposition rule
        rule = self._find_matching_rule(task, context)
        
        if not rule:
            # Default decomposition
            return self._default_decomposition(task)
        
        # Apply rule
        subtasks = []
        for i, template in enumerate(rule.subtask_templates):
            subtask = Task(
                task_id="{}_sub_{}".format(task.task_id, i),
                name=template.get("name", "Subtask {}".format(i + 1)),
                description=template.get("description", ""),
                task_type=TaskType(template.get("type", "atomic")),
                parent_id=task.task_id,
                priority=template.get("priority", task.priority),
                estimated_effort=template.get("effort", 1),
                metadata=template.get("metadata", {})
            )
            
            # Set dependencies
            if "depends_on" in template:
                dep_indices = template["depends_on"]
                subtask.dependencies = [
                    "{}_sub_{}".format(task.task_id, idx)
                    for idx in dep_indices
                ]
            
            subtasks.append(subtask)
        
        return subtasks
    
    def add_rule(self, rule: DecompositionRule) -> None:
        """Add a decomposition rule"""
        self.decomposition_rules[rule.rule_id] = rule
    
    def _find_matching_rule(self,
                           task: Task,
                           context: Dict[str, Any]) -> Optional[DecompositionRule]:
        """Find matching decomposition rule"""
        for rule in self.decomposition_rules.values():
            if self._matches_pattern(task, rule.pattern, context):
                if self._check_conditions(rule.conditions, context):
                    return rule
        return None
    
    def _matches_pattern(self,
                        task: Task,
                        pattern: str,
                        context: Dict[str, Any]) -> bool:
        """Check if task matches pattern"""
        # Simple pattern matching
        pattern_lower = pattern.lower()
        task_name_lower = task.name.lower()
        task_desc_lower = task.description.lower()
        
        return (pattern_lower in task_name_lower or 
                pattern_lower in task_desc_lower)
    
    def _check_conditions(self,
                         conditions: Dict[str, Any],
                         context: Dict[str, Any]) -> bool:
        """Check if conditions are met"""
        for key, expected_value in conditions.items():
            if key not in context or context[key] != expected_value:
                return False
        return True
    
    def _default_decomposition(self, task: Task) -> List[Task]:
        """Default decomposition strategy"""
        # Break into planning, execution, and verification
        subtasks = [
            Task(
                task_id="{}_plan".format(task.task_id),
                name="Plan: {}".format(task.name),
                description="Plan how to accomplish {}".format(task.name),
                task_type=TaskType.ATOMIC,
                parent_id=task.task_id,
                priority=task.priority
            ),
            Task(
                task_id="{}_execute".format(task.task_id),
                name="Execute: {}".format(task.name),
                description="Execute {}".format(task.name),
                task_type=TaskType.ATOMIC,
                parent_id=task.task_id,
                dependencies=["{}_plan".format(task.task_id)],
                priority=task.priority
            ),
            Task(
                task_id="{}_verify".format(task.task_id),
                name="Verify: {}".format(task.name),
                description="Verify completion of {}".format(task.name),
                task_type=TaskType.ATOMIC,
                parent_id=task.task_id,
                dependencies=["{}_execute".format(task.task_id)],
                priority=task.priority
            )
        ]
        
        return subtasks
    
    def _initialize_default_rules(self) -> None:
        """Initialize default decomposition rules"""
        # Rule for software development tasks
        self.add_rule(DecompositionRule(
            rule_id="software_dev",
            pattern="develop",
            subtask_templates=[
                {
                    "name": "Requirements Analysis",
                    "type": "atomic",
                    "effort": 2
                },
                {
                    "name": "Design",
                    "type": "atomic",
                    "effort": 3,
                    "depends_on": [0]
                },
                {
                    "name": "Implementation",
                    "type": "atomic",
                    "effort": 5,
                    "depends_on": [1]
                },
                {
                    "name": "Testing",
                    "type": "atomic",
                    "effort": 3,
                    "depends_on": [2]
                },
                {
                    "name": "Deployment",
                    "type": "atomic",
                    "effort": 2,
                    "depends_on": [3]
                }
            ],
            conditions={}
        ))
        
        # Rule for research tasks
        self.add_rule(DecompositionRule(
            rule_id="research",
            pattern="research",
            subtask_templates=[
                {
                    "name": "Literature Review",
                    "type": "atomic",
                    "effort": 3
                },
                {
                    "name": "Data Collection",
                    "type": "atomic",
                    "effort": 4,
                    "depends_on": [0]
                },
                {
                    "name": "Analysis",
                    "type": "atomic",
                    "effort": 5,
                    "depends_on": [1]
                },
                {
                    "name": "Report Writing",
                    "type": "atomic",
                    "effort": 3,
                    "depends_on": [2]
                }
            ],
            conditions={}
        ))


class DependencyAnalyzer:
    """Analyzes task dependencies"""
    
    def find_dependencies(self, tasks: List[Task]) -> Dict[str, Set[str]]:
        """
        Find all dependencies for each task.
        
        Args:
            tasks: List of tasks
            
        Returns:
            Dictionary mapping task_id to set of dependency task_ids
        """
        dep_graph = {}
        
        for task in tasks:
            dep_graph[task.task_id] = set(task.dependencies)
        
        return dep_graph
    
    def topological_sort(self, tasks: List[Task]) -> List[List[str]]:
        """
        Perform topological sort to determine execution order.
        Returns list of lists where each sublist contains tasks that can run in parallel.
        
        Args:
            tasks: List of tasks
            
        Returns:
            Execution order (levels of parallel tasks)
        """
        # Build dependency graph
        dep_graph = self.find_dependencies(tasks)
        in_degree = {task.task_id: len(deps) for task, deps in 
                     zip(tasks, dep_graph.values())}
        
        result = []
        remaining = set(task.task_id for task in tasks)
        
        while remaining:
            # Find tasks with no dependencies
            ready = [tid for tid in remaining if in_degree[tid] == 0]
            
            if not ready:
                # Circular dependency detected
                break
            
            result.append(ready)
            
            # Remove ready tasks and update in_degrees
            for tid in ready:
                remaining.remove(tid)
                # Update dependents
                for other_tid in remaining:
                    if tid in dep_graph.get(other_tid, set()):
                        in_degree[other_tid] -= 1
        
        return result
    
    def find_critical_path(self,
                          tasks: List[Task],
                          execution_order: List[List[str]]) -> List[str]:
        """
        Find critical path (longest path through dependencies).
        
        Args:
            tasks: List of tasks
            execution_order: Execution order from topological sort
            
        Returns:
            List of task IDs in critical path
        """
        task_map = {task.task_id: task for task in tasks}
        
        # Calculate earliest start time for each task
        earliest_start = {}
        for level in execution_order:
            for tid in level:
                task = task_map[tid]
                if not task.dependencies:
                    earliest_start[tid] = 0
                else:
                    max_dep_time = max(
                        earliest_start.get(dep, 0) + task_map[dep].estimated_effort
                        for dep in task.dependencies
                    )
                    earliest_start[tid] = max_dep_time
        
        # Find task with maximum completion time
        if not earliest_start:
            return []
        
        end_task = max(earliest_start.items(),
                      key=lambda x: x[1] + task_map[x[0]].estimated_effort)
        
        # Backtrack to find critical path
        critical_path = []
        current = end_task[0]
        
        while current:
            critical_path.insert(0, current)
            task = task_map[current]
            
            if not task.dependencies:
                break
            
            # Find dependency on critical path
            current = max(
                task.dependencies,
                key=lambda dep: earliest_start.get(dep, 0) + task_map[dep].estimated_effort
            )
        
        return critical_path
    
    def detect_circular_dependencies(self, tasks: List[Task]) -> List[List[str]]:
        """
        Detect circular dependencies.
        
        Args:
            tasks: List of tasks
            
        Returns:
            List of circular dependency chains
        """
        dep_graph = self.find_dependencies(tasks)
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in dep_graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
            
            rec_stack.remove(node)
        
        for task in tasks:
            if task.task_id not in visited:
                dfs(task.task_id, [])
        
        return cycles


class HierarchicalTaskDecompositionAgent:
    """
    Agent that decomposes complex tasks hierarchically.
    Manages task hierarchy, dependencies, and execution planning.
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.decomposer = TaskDecomposer()
        self.dependency_analyzer = DependencyAnalyzer()
        self.execution_plans: Dict[str, ExecutionPlan] = {}
        self.task_counter = 0
    
    def create_task(self,
                   name: str,
                   description: str,
                   task_type: TaskType = TaskType.COMPOSITE,
                   priority: int = 5,
                   estimated_effort: int = 1,
                   metadata: Optional[Dict[str, Any]] = None) -> Task:
        """
        Create a new task.
        
        Args:
            name: Task name
            description: Task description
            task_type: Type of task
            priority: Priority (1-10)
            estimated_effort: Estimated effort
            metadata: Optional metadata
            
        Returns:
            Created task
        """
        task = Task(
            task_id="task_{}".format(self.task_counter),
            name=name,
            description=description,
            task_type=task_type,
            priority=priority,
            estimated_effort=estimated_effort,
            metadata=metadata or {}
        )
        
        self.tasks[task.task_id] = task
        self.task_counter += 1
        
        return task
    
    def decompose_recursively(self,
                             task_id: str,
                             max_depth: int = 3,
                             current_depth: int = 0,
                             context: Optional[Dict[str, Any]] = None) -> List[Task]:
        """
        Recursively decompose a task.
        
        Args:
            task_id: Task to decompose
            max_depth: Maximum decomposition depth
            current_depth: Current depth
            context: Optional context
            
        Returns:
            List of all tasks (original + subtasks)
        """
        if context is None:
            context = {}
        
        task = self.tasks.get(task_id)
        if not task or current_depth >= max_depth:
            return [task] if task else []
        
        if task.task_type == TaskType.ATOMIC:
            return [task]
        
        # Decompose task
        subtasks = self.decomposer.decompose_task(task, context)
        
        # Store subtasks
        task.subtasks = [st.task_id for st in subtasks]
        for subtask in subtasks:
            self.tasks[subtask.task_id] = subtask
        
        # Recursively decompose subtasks
        all_tasks = [task]
        for subtask in subtasks:
            decomposed = self.decompose_recursively(
                subtask.task_id,
                max_depth,
                current_depth + 1,
                context
            )
            all_tasks.extend(decomposed[1:])  # Skip the subtask itself (already added)
        
        return all_tasks
    
    def create_execution_plan(self, root_task_id: str) -> ExecutionPlan:
        """
        Create execution plan for a task and its subtasks.
        
        Args:
            root_task_id: Root task ID
            
        Returns:
            Execution plan
        """
        # Get all tasks in hierarchy
        all_tasks = self._get_task_hierarchy(root_task_id)
        
        # Filter to atomic tasks (only these need execution)
        atomic_tasks = [t for t in all_tasks if t.task_type == TaskType.ATOMIC]
        
        # Determine execution order
        execution_order = self.dependency_analyzer.topological_sort(atomic_tasks)
        
        # Find critical path
        critical_path = self.dependency_analyzer.find_critical_path(
            atomic_tasks,
            execution_order
        )
        
        # Calculate total effort
        total_effort = sum(t.estimated_effort for t in atomic_tasks)
        
        # Create plan
        plan = ExecutionPlan(
            plan_id="plan_{}_{}".format(root_task_id, len(self.execution_plans)),
            root_task_id=root_task_id,
            execution_order=execution_order,
            total_tasks=len(atomic_tasks),
            estimated_total_effort=total_effort,
            critical_path=critical_path
        )
        
        self.execution_plans[plan.plan_id] = plan
        
        return plan
    
    def execute_task(self, task_id: str) -> bool:
        """
        Mark task as in progress (simulate execution).
        
        Args:
            task_id: Task to execute
            
        Returns:
            Success status
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        # Check dependencies
        if not self._are_dependencies_met(task):
            task.status = TaskStatus.BLOCKED
            return False
        
        # Start execution
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        return True
    
    def complete_task(self, task_id: str, actual_effort: Optional[int] = None) -> bool:
        """
        Mark task as completed.
        
        Args:
            task_id: Task to complete
            actual_effort: Actual effort spent
            
        Returns:
            Success status
        """
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.actual_effort = actual_effort or task.estimated_effort
        
        # Update parent task status
        if task.parent_id:
            self._update_parent_status(task.parent_id)
        
        return True
    
    def get_task_hierarchy(self, root_task_id: str) -> Dict[str, Any]:
        """
        Get task hierarchy as nested dictionary.
        
        Args:
            root_task_id: Root task ID
            
        Returns:
            Nested task hierarchy
        """
        task = self.tasks.get(root_task_id)
        if not task:
            return {}
        
        result = {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status.value,
            "type": task.task_type.value,
            "priority": task.priority,
            "estimated_effort": task.estimated_effort,
            "subtasks": []
        }
        
        for subtask_id in task.subtasks:
            result["subtasks"].append(self.get_task_hierarchy(subtask_id))
        
        return result
    
    def get_progress(self, root_task_id: str) -> Dict[str, Any]:
        """
        Get progress for a task and its subtasks.
        
        Args:
            root_task_id: Root task ID
            
        Returns:
            Progress information
        """
        all_tasks = self._get_task_hierarchy(root_task_id)
        
        total_tasks = len(all_tasks)
        completed = sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED)
        in_progress = sum(1 for t in all_tasks if t.status == TaskStatus.IN_PROGRESS)
        blocked = sum(1 for t in all_tasks if t.status == TaskStatus.BLOCKED)
        
        total_effort = sum(t.estimated_effort for t in all_tasks)
        completed_effort = sum(
            t.actual_effort or t.estimated_effort
            for t in all_tasks if t.status == TaskStatus.COMPLETED
        )
        
        return {
            "total_tasks": total_tasks,
            "completed": completed,
            "in_progress": in_progress,
            "blocked": blocked,
            "pending": total_tasks - completed - in_progress - blocked,
            "completion_percentage": (completed / total_tasks * 100) if total_tasks > 0 else 0,
            "effort_percentage": (completed_effort / total_effort * 100) if total_effort > 0 else 0
        }
    
    def _get_task_hierarchy(self, task_id: str) -> List[Task]:
        """Get all tasks in hierarchy"""
        task = self.tasks.get(task_id)
        if not task:
            return []
        
        result = [task]
        for subtask_id in task.subtasks:
            result.extend(self._get_task_hierarchy(subtask_id))
        
        return result
    
    def _are_dependencies_met(self, task: Task) -> bool:
        """Check if all dependencies are completed"""
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def _update_parent_status(self, parent_id: str) -> None:
        """Update parent task status based on subtasks"""
        parent = self.tasks.get(parent_id)
        if not parent:
            return
        
        # Check if all subtasks are completed
        all_completed = all(
            self.tasks[sid].status == TaskStatus.COMPLETED
            for sid in parent.subtasks
        )
        
        if all_completed:
            parent.status = TaskStatus.COMPLETED
            parent.completed_at = datetime.now()
            
            # Recursively update parent's parent
            if parent.parent_id:
                self._update_parent_status(parent.parent_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        status_counts = {}
        type_counts = {}
        
        for task in self.tasks.values():
            status = task.status.value
            task_type = task.task_type.value
            
            status_counts[status] = status_counts.get(status, 0) + 1
            type_counts[task_type] = type_counts.get(task_type, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "tasks_by_status": status_counts,
            "tasks_by_type": type_counts,
            "execution_plans": len(self.execution_plans)
        }


def demonstrate_hierarchical_task_decomposition():
    """Demonstrate hierarchical task decomposition agent"""
    print("=" * 70)
    print("Hierarchical Task Decomposition Agent Demonstration")
    print("=" * 70)
    
    agent = HierarchicalTaskDecompositionAgent()
    
    # Example 1: Create and decompose a complex task
    print("\n1. Creating Complex Task:")
    root_task = agent.create_task(
        name="Develop Web Application",
        description="Build a full-stack web application with user authentication",
        task_type=TaskType.COMPOSITE,
        priority=8,
        estimated_effort=20
    )
    print("Created task: {} ({})".format(root_task.name, root_task.task_id))
    
    # Decompose recursively
    print("\n2. Decomposing Task Recursively:")
    all_tasks = agent.decompose_recursively(root_task.task_id, max_depth=2)
    print("Total tasks after decomposition: {}".format(len(all_tasks)))
    
    # Show hierarchy
    print("\n3. Task Hierarchy:")
    hierarchy = agent.get_task_hierarchy(root_task.task_id)
    print(json.dumps(hierarchy, indent=2))
    
    # Create execution plan
    print("\n4. Creating Execution Plan:")
    plan = agent.create_execution_plan(root_task.task_id)
    print("Plan ID: {}".format(plan.plan_id))
    print("Total atomic tasks: {}".format(plan.total_tasks))
    print("Estimated total effort: {}".format(plan.estimated_total_effort))
    print("Execution levels: {}".format(len(plan.execution_order)))
    
    print("\nExecution order (parallel groups):")
    for i, level in enumerate(plan.execution_order, 1):
        print("  Level {}: {} tasks can run in parallel".format(i, len(level)))
        for task_id in level[:2]:  # Show first 2
            task = agent.tasks[task_id]
            print("    - {}".format(task.name))
    
    print("\nCritical path: {} tasks".format(len(plan.critical_path)))
    for task_id in plan.critical_path:
        task = agent.tasks[task_id]
        print("  -> {} (effort: {})".format(task.name, task.estimated_effort))
    
    # Simulate execution
    print("\n5. Simulating Task Execution:")
    executed_count = 0
    for level in plan.execution_order[:2]:  # Execute first 2 levels
        print("\nExecuting level tasks:")
        for task_id in level:
            if agent.execute_task(task_id):
                task = agent.tasks[task_id]
                print("  Started: {}".format(task.name))
                # Complete immediately for demo
                agent.complete_task(task_id)
                print("  Completed: {}".format(task.name))
                executed_count += 1
    
    # Check progress
    print("\n6. Task Progress:")
    progress = agent.get_progress(root_task.task_id)
    print(json.dumps(progress, indent=2))
    
    # Example 2: Research task decomposition
    print("\n7. Creating Research Task:")
    research_task = agent.create_task(
        name="Research Machine Learning for Healthcare",
        description="Comprehensive research on ML applications in healthcare",
        task_type=TaskType.COMPOSITE,
        priority=7,
        estimated_effort=15
    )
    
    research_tasks = agent.decompose_recursively(research_task.task_id, max_depth=1)
    print("Research task decomposed into {} subtasks".format(
        len(research_tasks) - 1
    ))
    
    # Show research subtasks
    for subtask_id in agent.tasks[research_task.task_id].subtasks:
        subtask = agent.tasks[subtask_id]
        print("  - {}".format(subtask.name))
    
    # Statistics
    print("\n8. Agent Statistics:")
    stats = agent.get_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    demonstrate_hierarchical_task_decomposition()
