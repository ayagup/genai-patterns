"""
Pattern 109: Dynamic Task Allocation Agent

This pattern implements intelligent task distribution and scheduling across
agents or resources with dynamic load balancing and optimization.

Use Cases:
- Multi-agent task coordination
- Resource optimization
- Dynamic scheduling
- Load balancing across workers
- Priority-based task management
- Deadline-aware scheduling

Key Features:
- Dynamic task allocation algorithms
- Agent capability matching
- Load balancing strategies
- Priority-based scheduling
- Deadline awareness
- Performance monitoring
- Adaptive reallocation

Implementation:
- Pure Python (3.8+) with comprehensive type hints
- Zero external dependencies
- Production-ready error handling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import uuid
import heapq


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AllocationStrategy(Enum):
    """Task allocation strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPABILITY_MATCH = "capability_match"
    PRIORITY_BASED = "priority_based"
    DEADLINE_AWARE = "deadline_aware"


@dataclass
class WorkerAgent:
    """Worker agent that executes tasks."""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    
    # Capabilities
    capabilities: Set[str] = field(default_factory=set)
    skill_levels: Dict[str, float] = field(default_factory=dict)  # skill -> proficiency
    
    # Capacity
    max_concurrent_tasks: int = 3
    current_load: int = 0
    
    # Performance
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_completion_time: float = 0.0
    
    # State
    is_available: bool = True
    current_tasks: List[str] = field(default_factory=list)  # task_ids
    
    # Statistics
    created_at: datetime = field(default_factory=datetime.now)
    last_task_at: Optional[datetime] = None
    
    def get_load_percentage(self) -> float:
        """Get current load as percentage of capacity."""
        return self.current_load / self.max_concurrent_tasks if self.max_concurrent_tasks > 0 else 0.0
    
    def can_handle_task(self, required_capabilities: Set[str]) -> bool:
        """Check if agent can handle task requirements."""
        return required_capabilities.issubset(self.capabilities)
    
    def get_capability_score(self, required_capabilities: Set[str]) -> float:
        """Calculate how well agent matches task requirements."""
        if not required_capabilities:
            return 1.0
        
        scores = []
        for cap in required_capabilities:
            if cap in self.skill_levels:
                scores.append(self.skill_levels[cap])
            elif cap in self.capabilities:
                scores.append(0.5)  # Has capability but no proficiency info
            else:
                return 0.0  # Missing required capability
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def assign_task(self, task_id: str) -> bool:
        """Assign task to agent."""
        if self.current_load >= self.max_concurrent_tasks:
            return False
        
        self.current_tasks.append(task_id)
        self.current_load += 1
        self.last_task_at = datetime.now()
        return True
    
    def complete_task(self, task_id: str, success: bool = True) -> None:
        """Mark task as completed."""
        if task_id in self.current_tasks:
            self.current_tasks.remove(task_id)
            self.current_load = max(0, self.current_load - 1)
            
            self.total_tasks += 1
            if success:
                self.completed_tasks += 1
            else:
                self.failed_tasks += 1


@dataclass
class TaskItem:
    """Task to be allocated and executed."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    
    # Requirements
    required_capabilities: Set[str] = field(default_factory=set)
    estimated_duration: int = 60  # seconds
    
    # Priority and deadline
    priority: TaskPriority = TaskPriority.NORMAL
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    # Assignment
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None  # agent_id
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # task_ids
    dependents: List[str] = field(default_factory=list)  # task_ids
    
    def __lt__(self, other: 'TaskItem') -> bool:
        """For priority queue ordering."""
        # Higher priority first
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        
        # Then by deadline (earlier first)
        if self.deadline and other.deadline:
            return self.deadline < other.deadline
        elif self.deadline:
            return True
        elif other.deadline:
            return False
        
        # Finally by creation time (older first)
        return self.created_at < other.created_at
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if task is ready (dependencies met)."""
        return all(dep_id in completed_tasks for dep_id in self.dependencies)
    
    def get_urgency_score(self) -> float:
        """Calculate urgency based on deadline and priority."""
        score = self.priority.value / 5.0  # Normalize to 0-1
        
        if self.deadline:
            time_until_deadline = (self.deadline - datetime.now()).total_seconds()
            if time_until_deadline < 0:
                score += 2.0  # Overdue!
            elif time_until_deadline < self.estimated_duration:
                score += 1.0  # Tight deadline
            elif time_until_deadline < self.estimated_duration * 2:
                score += 0.5  # Approaching deadline
        
        return score


class TaskQueue:
    """
    Priority queue for tasks.
    
    Supports multiple ordering strategies and dynamic reordering.
    """
    
    def __init__(self):
        self.tasks: Dict[str, TaskItem] = {}
        self.pending_queue: List[TaskItem] = []  # Priority heap
        
        # Indices
        self.status_index: Dict[TaskStatus, Set[str]] = {
            status: set() for status in TaskStatus
        }
        self.agent_index: Dict[str, Set[str]] = {}  # agent_id -> task_ids
    
    def add_task(self, task: TaskItem) -> None:
        """Add task to queue."""
        self.tasks[task.task_id] = task
        
        # Update status index
        self.status_index[task.status].add(task.task_id)
        
        # Add to pending queue if pending
        if task.status == TaskStatus.PENDING:
            heapq.heappush(self.pending_queue, task)
    
    def get_next_task(self, agent: WorkerAgent,
                     completed_tasks: Set[str]) -> Optional[TaskItem]:
        """Get next task for agent considering capabilities and dependencies."""
        # Rebuild queue to get fresh ordering
        self.pending_queue = [
            t for t in self.pending_queue
            if t.task_id in self.tasks and self.tasks[t.task_id].status == TaskStatus.PENDING
        ]
        heapq.heapify(self.pending_queue)
        
        # Find suitable task
        temp_removed = []
        task_found = None
        
        while self.pending_queue:
            task = heapq.heappop(self.pending_queue)
            
            # Check if task still exists and is pending
            if task.task_id not in self.tasks:
                continue
            
            current_task = self.tasks[task.task_id]
            if current_task.status != TaskStatus.PENDING:
                continue
            
            # Check if agent can handle it
            if not agent.can_handle_task(task.required_capabilities):
                temp_removed.append(task)
                continue
            
            # Check if dependencies are met
            if not task.is_ready(completed_tasks):
                temp_removed.append(task)
                continue
            
            # Found suitable task
            task_found = task
            break
        
        # Restore removed tasks
        for task in temp_removed:
            heapq.heappush(self.pending_queue, task)
        
        return task_found
    
    def update_task_status(self, task_id: str, status: TaskStatus,
                          agent_id: Optional[str] = None) -> None:
        """Update task status."""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        old_status = task.status
        
        # Update indices
        self.status_index[old_status].discard(task_id)
        self.status_index[status].add(task_id)
        
        # Update task
        task.status = status
        
        if status == TaskStatus.ASSIGNED and agent_id:
            task.assigned_to = agent_id
            task.assigned_at = datetime.now()
            
            # Update agent index
            if agent_id not in self.agent_index:
                self.agent_index[agent_id] = set()
            self.agent_index[agent_id].add(task_id)
        
        elif status == TaskStatus.IN_PROGRESS:
            task.started_at = datetime.now()
        
        elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            task.completed_at = datetime.now()
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[TaskItem]:
        """Get all tasks with given status."""
        task_ids = self.status_index.get(status, set())
        return [self.tasks[tid] for tid in task_ids if tid in self.tasks]


class TaskAllocator:
    """
    Allocates tasks to agents using various strategies.
    
    Supports multiple allocation algorithms and dynamic rebalancing.
    """
    
    def __init__(self, strategy: AllocationStrategy = AllocationStrategy.CAPABILITY_MATCH):
        self.strategy = strategy
        self.round_robin_index = 0
    
    def allocate(self, task: TaskItem, agents: List[WorkerAgent],
                completed_tasks: Set[str]) -> Optional[WorkerAgent]:
        """
        Allocate task to best agent based on strategy.
        
        Returns selected agent or None if no suitable agent.
        """
        # Filter available agents
        available = [
            agent for agent in agents
            if agent.is_available and agent.current_load < agent.max_concurrent_tasks
        ]
        
        if not available:
            return None
        
        # Filter by capabilities
        capable = [
            agent for agent in available
            if agent.can_handle_task(task.required_capabilities)
        ]
        
        if not capable:
            return None
        
        # Apply strategy
        if self.strategy == AllocationStrategy.ROUND_ROBIN:
            return self._round_robin(capable)
        
        elif self.strategy == AllocationStrategy.LEAST_LOADED:
            return self._least_loaded(capable)
        
        elif self.strategy == AllocationStrategy.CAPABILITY_MATCH:
            return self._capability_match(capable, task)
        
        elif self.strategy == AllocationStrategy.PRIORITY_BASED:
            return self._priority_based(capable, task)
        
        elif self.strategy == AllocationStrategy.DEADLINE_AWARE:
            return self._deadline_aware(capable, task)
        
        return capable[0] if capable else None
    
    def _round_robin(self, agents: List[WorkerAgent]) -> WorkerAgent:
        """Simple round-robin allocation."""
        agent = agents[self.round_robin_index % len(agents)]
        self.round_robin_index += 1
        return agent
    
    def _least_loaded(self, agents: List[WorkerAgent]) -> WorkerAgent:
        """Allocate to agent with lowest load."""
        return min(agents, key=lambda a: a.get_load_percentage())
    
    def _capability_match(self, agents: List[WorkerAgent], task: TaskItem) -> WorkerAgent:
        """Allocate to agent with best capability match."""
        def score(agent: WorkerAgent) -> float:
            capability_score = agent.get_capability_score(task.required_capabilities)
            load_penalty = agent.get_load_percentage() * 0.3
            return capability_score - load_penalty
        
        return max(agents, key=score)
    
    def _priority_based(self, agents: List[WorkerAgent], task: TaskItem) -> WorkerAgent:
        """Allocate considering task priority."""
        # For high priority tasks, prefer agents with better track record
        if task.priority.value >= TaskPriority.HIGH.value:
            def score(agent: WorkerAgent) -> float:
                success_rate = (
                    agent.completed_tasks / agent.total_tasks
                    if agent.total_tasks > 0 else 0.5
                )
                return success_rate - agent.get_load_percentage() * 0.2
            
            return max(agents, key=score)
        else:
            return self._least_loaded(agents)
    
    def _deadline_aware(self, agents: List[WorkerAgent], task: TaskItem) -> WorkerAgent:
        """Allocate considering task deadline."""
        urgency = task.get_urgency_score()
        
        # For urgent tasks, find fastest agent
        if urgency > 1.5:
            def score(agent: WorkerAgent) -> float:
                speed = 1.0 / (agent.average_completion_time + 1.0)
                return speed - agent.get_load_percentage() * 0.3
            
            return max(agents, key=score)
        else:
            return self._least_loaded(agents)


class DynamicTaskAllocationAgent:
    """
    Agent that manages dynamic task allocation and scheduling.
    
    Features:
    - Multiple allocation strategies
    - Load balancing
    - Priority-based scheduling
    - Deadline awareness
    - Performance monitoring
    """
    
    def __init__(self, strategy: AllocationStrategy = AllocationStrategy.CAPABILITY_MATCH):
        self.task_queue = TaskQueue()
        self.allocator = TaskAllocator(strategy)
        
        # Worker registry
        self.workers: Dict[str, WorkerAgent] = {}
        
        # Completed tasks
        self.completed_tasks: Set[str] = set()
        
        # Statistics
        self.total_allocated = 0
        self.total_completed = 0
        self.total_failed = 0
        self.allocation_time = []
    
    def register_worker(self, worker: WorkerAgent) -> None:
        """Register worker agent."""
        self.workers[worker.agent_id] = worker
    
    def submit_task(self, task: TaskItem) -> None:
        """Submit task for allocation."""
        self.task_queue.add_task(task)
    
    def allocate_tasks(self) -> List[Tuple[TaskItem, WorkerAgent]]:
        """
        Allocate pending tasks to available workers.
        
        Returns list of (task, agent) allocations made.
        """
        allocations = []
        
        # Get available workers
        available_workers = [
            w for w in self.workers.values()
            if w.is_available and w.current_load < w.max_concurrent_tasks
        ]
        
        if not available_workers:
            return allocations
        
        # Try to allocate tasks
        while available_workers:
            # Get next task
            task = self.task_queue.get_next_task(
                available_workers[0],  # Any agent for capability check
                self.completed_tasks
            )
            
            if not task:
                break  # No more suitable tasks
            
            # Find best agent
            agent = self.allocator.allocate(
                task,
                available_workers,
                self.completed_tasks
            )
            
            if not agent:
                break  # No suitable agent
            
            # Allocate task
            if agent.assign_task(task.task_id):
                self.task_queue.update_task_status(
                    task.task_id,
                    TaskStatus.ASSIGNED,
                    agent.agent_id
                )
                
                allocations.append((task, agent))
                self.total_allocated += 1
                
                # Remove agent if now fully loaded
                if agent.current_load >= agent.max_concurrent_tasks:
                    available_workers.remove(agent)
        
        return allocations
    
    def complete_task(self, task_id: str, agent_id: str,
                     success: bool = True, result: Any = None,
                     error: Optional[str] = None) -> None:
        """Mark task as completed."""
        # Update task
        status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        self.task_queue.update_task_status(task_id, status)
        
        if task_id in self.task_queue.tasks:
            task = self.task_queue.tasks[task_id]
            task.result = result
            task.error = error
        
        # Update worker
        if agent_id in self.workers:
            self.workers[agent_id].complete_task(task_id, success)
        
        # Update statistics
        if success:
            self.completed_tasks.add(task_id)
            self.total_completed += 1
        else:
            self.total_failed += 1
    
    def rebalance_load(self) -> List[Tuple[str, str, str]]:
        """
        Rebalance load across workers.
        
        Returns list of (task_id, from_agent, to_agent) transfers.
        """
        transfers = []
        
        # Find overloaded and underloaded workers
        workers_by_load = sorted(
            self.workers.values(),
            key=lambda w: w.get_load_percentage(),
            reverse=True
        )
        
        if len(workers_by_load) < 2:
            return transfers
        
        overloaded = [w for w in workers_by_load if w.get_load_percentage() > 0.8]
        underloaded = [w for w in workers_by_load if w.get_load_percentage() < 0.5]
        
        # Transfer tasks from overloaded to underloaded
        for source in overloaded:
            for target in underloaded:
                if source.current_load <= target.current_load:
                    break
                
                # Find transferable task
                for task_id in source.current_tasks:
                    if task_id not in self.task_queue.tasks:
                        continue
                    
                    task = self.task_queue.tasks[task_id]
                    
                    # Check if target can handle it
                    if target.can_handle_task(task.required_capabilities):
                        # Transfer
                        source.complete_task(task_id, success=False)
                        target.assign_task(task_id)
                        
                        self.task_queue.update_task_status(
                            task_id,
                            TaskStatus.ASSIGNED,
                            target.agent_id
                        )
                        
                        transfers.append((task_id, source.agent_id, target.agent_id))
                        break
        
        return transfers
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get allocation statistics."""
        pending = len(self.task_queue.get_tasks_by_status(TaskStatus.PENDING))
        in_progress = len(self.task_queue.get_tasks_by_status(TaskStatus.IN_PROGRESS))
        
        return {
            "total_workers": len(self.workers),
            "available_workers": sum(1 for w in self.workers.values() if w.is_available),
            "total_tasks": len(self.task_queue.tasks),
            "pending_tasks": pending,
            "in_progress_tasks": in_progress,
            "completed_tasks": self.total_completed,
            "failed_tasks": self.total_failed,
            "total_allocated": self.total_allocated,
            "completion_rate": (
                self.total_completed / self.total_allocated
                if self.total_allocated > 0 else 0.0
            )
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_task_allocation():
    """Demonstrate dynamic task allocation."""
    
    print("=" * 70)
    print("DYNAMIC TASK ALLOCATION DEMONSTRATION")
    print("=" * 70)
    
    print("\n1. INITIALIZING SYSTEM")
    print("-" * 70)
    
    allocator = DynamicTaskAllocationAgent(AllocationStrategy.CAPABILITY_MATCH)
    print("   System initialized with CAPABILITY_MATCH strategy")
    
    print("\n2. REGISTERING WORKERS")
    print("-" * 70)
    print("   Creating worker agents...")
    
    # Create workers with different capabilities
    worker1 = WorkerAgent(
        name="Worker-1",
        capabilities={"data_processing", "analysis", "reporting"},
        skill_levels={"data_processing": 0.9, "analysis": 0.8},
        max_concurrent_tasks=2
    )
    allocator.register_worker(worker1)
    print(f"     {worker1.name}: {worker1.capabilities}")
    
    worker2 = WorkerAgent(
        name="Worker-2",
        capabilities={"data_processing", "machine_learning"},
        skill_levels={"data_processing": 0.7, "machine_learning": 0.9},
        max_concurrent_tasks=3
    )
    allocator.register_worker(worker2)
    print(f"     {worker2.name}: {worker2.capabilities}")
    
    worker3 = WorkerAgent(
        name="Worker-3",
        capabilities={"reporting", "visualization"},
        skill_levels={"reporting": 0.85, "visualization": 0.9},
        max_concurrent_tasks=2
    )
    allocator.register_worker(worker3)
    print(f"     {worker3.name}: {worker3.capabilities}")
    
    print("\n3. SUBMITTING TASKS")
    print("-" * 70)
    print("   Creating diverse task set...")
    
    # Create tasks with different requirements
    tasks = [
        TaskItem(
            name="Data Analysis",
            required_capabilities={"data_processing", "analysis"},
            priority=TaskPriority.HIGH,
            estimated_duration=300
        ),
        TaskItem(
            name="ML Model Training",
            required_capabilities={"machine_learning", "data_processing"},
            priority=TaskPriority.NORMAL,
            estimated_duration=600
        ),
        TaskItem(
            name="Report Generation",
            required_capabilities={"reporting"},
            priority=TaskPriority.NORMAL,
            estimated_duration=180
        ),
        TaskItem(
            name="Data Visualization",
            required_capabilities={"visualization", "reporting"},
            priority=TaskPriority.LOW,
            estimated_duration=240
        ),
        TaskItem(
            name="Critical Analysis",
            required_capabilities={"analysis", "data_processing"},
            priority=TaskPriority.CRITICAL,
            deadline=datetime.now() + timedelta(minutes=10),
            estimated_duration=420
        )
    ]
    
    for task in tasks:
        allocator.submit_task(task)
        print(f"     {task.name}: priority={task.priority.name}, "
              f"caps={task.required_capabilities}")
    
    print("\n4. INITIAL ALLOCATION")
    print("-" * 70)
    print("   Allocating tasks to workers...")
    
    allocations = allocator.allocate_tasks()
    print(f"\n     Allocated {len(allocations)} tasks:")
    for task, agent in allocations:
        print(f"       {task.name} → {agent.name}")
        print(f"         Match score: {agent.get_capability_score(task.required_capabilities):.2f}")
        print(f"         Worker load: {agent.get_load_percentage():.1%}")
    
    print("\n5. WORKER LOAD STATUS")
    print("-" * 70)
    
    for worker in allocator.workers.values():
        print(f"   {worker.name}:")
        print(f"     Load: {worker.current_load}/{worker.max_concurrent_tasks} "
              f"({worker.get_load_percentage():.1%})")
        print(f"     Current tasks: {len(worker.current_tasks)}")
    
    print("\n6. COMPLETING TASKS")
    print("-" * 70)
    print("   Simulating task completion...")
    
    # Complete some tasks
    if allocations:
        task1, agent1 = allocations[0]
        allocator.complete_task(task1.task_id, agent1.agent_id, success=True)
        print(f"     ✓ {task1.name} completed by {agent1.name}")
        
        if len(allocations) > 1:
            task2, agent2 = allocations[1]
            allocator.complete_task(task2.task_id, agent2.agent_id, success=True)
            print(f"     ✓ {task2.name} completed by {agent2.name}")
    
    print("\n7. SECOND ALLOCATION ROUND")
    print("-" * 70)
    print("   Allocating remaining tasks...")
    
    allocations2 = allocator.allocate_tasks()
    if allocations2:
        print(f"\n     Allocated {len(allocations2)} more tasks:")
        for task, agent in allocations2:
            print(f"       {task.name} → {agent.name}")
    else:
        print("     No more tasks to allocate")
    
    print("\n8. LOAD REBALANCING")
    print("-" * 70)
    print("   Checking for load imbalance...")
    
    transfers = allocator.rebalance_load()
    if transfers:
        print(f"     Rebalanced {len(transfers)} tasks:")
        for task_id, from_agent, to_agent in transfers:
            print(f"       Task {task_id}: {from_agent} → {to_agent}")
    else:
        print("     Load is balanced")
    
    print("\n9. SYSTEM STATISTICS")
    print("-" * 70)
    
    stats = allocator.get_statistics()
    print(f"   Total workers: {stats['total_workers']}")
    print(f"   Available workers: {stats['available_workers']}")
    print(f"   Total tasks: {stats['total_tasks']}")
    print(f"   Pending: {stats['pending_tasks']}")
    print(f"   In progress: {stats['in_progress_tasks']}")
    print(f"   Completed: {stats['completed_tasks']}")
    print(f"   Failed: {stats['failed_tasks']}")
    print(f"   Completion rate: {stats['completion_rate']:.1%}")
    
    print("\n10. WORKER PERFORMANCE")
    print("-" * 70)
    
    for worker in allocator.workers.values():
        success_rate = (
            worker.completed_tasks / worker.total_tasks
            if worker.total_tasks > 0 else 0.0
        )
        print(f"   {worker.name}:")
        print(f"     Total tasks: {worker.total_tasks}")
        print(f"     Completed: {worker.completed_tasks}")
        print(f"     Success rate: {success_rate:.1%}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("1. Worker registration with capabilities and capacities")
    print("2. Task submission with priorities and requirements")
    print("3. Capability-based task allocation")
    print("4. Load-aware worker selection")
    print("5. Priority-based task ordering")
    print("6. Dynamic task completion tracking")
    print("7. Multi-round allocation")
    print("8. Load rebalancing across workers")
    print("9. System statistics and monitoring")
    print("10. Worker performance tracking")


if __name__ == "__main__":
    demonstrate_task_allocation()
