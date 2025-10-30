"""
Task Allocation & Scheduling Pattern

Intelligent distribution of tasks across agents with priority-based scheduling,
load balancing, and resource optimization. Ensures efficient utilization and
timely completion of work.

Use Cases:
- Multi-agent task distribution
- Resource-constrained environments
- Priority-based workflows
- Load balancing across agents
- Deadline-driven task management

Benefits:
- Optimal resource utilization
- Fair task distribution
- Priority enforcement
- Deadline awareness
- Adaptive scheduling
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import heapq
import time


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(Enum):
    """Agent availability status"""
    IDLE = "idle"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"


@dataclass
class Task:
    """A task to be executed"""
    task_id: str
    name: str
    priority: TaskPriority
    estimated_duration: float  # in seconds
    required_capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    data: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    
    def __lt__(self, other):
        """Compare for priority queue (lower priority value = higher priority)"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        
        # If same priority, use deadline
        if self.deadline and other.deadline:
            return self.deadline < other.deadline
        elif self.deadline:
            return True
        
        return False
    
    def is_ready(self, completed_tasks: set) -> bool:
        """Check if all dependencies are completed"""
        return all(dep in completed_tasks for dep in self.dependencies)


@dataclass
class Agent:
    """An agent that can execute tasks"""
    agent_id: str
    name: str
    capabilities: List[str]
    max_concurrent_tasks: int = 1
    status: AgentStatus = AgentStatus.IDLE
    current_tasks: List[str] = field(default_factory=list)
    completed_tasks: int = 0
    total_processing_time: float = 0.0
    
    def can_handle(self, task: Task) -> bool:
        """Check if agent can handle task"""
        if self.status == AgentStatus.UNAVAILABLE:
            return False
        
        if len(self.current_tasks) >= self.max_concurrent_tasks:
            return False
        
        # Check if agent has required capabilities
        return all(cap in self.capabilities for cap in task.required_capabilities)
    
    def get_load(self) -> float:
        """Get current load (0.0 to 1.0)"""
        return len(self.current_tasks) / self.max_concurrent_tasks
    
    def assign_task(self, task_id: str) -> None:
        """Assign task to agent"""
        self.current_tasks.append(task_id)
        if len(self.current_tasks) >= self.max_concurrent_tasks:
            self.status = AgentStatus.BUSY
    
    def complete_task(self, task_id: str, duration: float) -> None:
        """Mark task as completed"""
        if task_id in self.current_tasks:
            self.current_tasks.remove(task_id)
            self.completed_tasks += 1
            self.total_processing_time += duration
            
            if len(self.current_tasks) < self.max_concurrent_tasks:
                self.status = AgentStatus.IDLE


class AllocationStrategy(Enum):
    """Task allocation strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPABILITY_MATCH = "capability_match"
    DEADLINE_AWARE = "deadline_aware"


class TaskQueue:
    """Priority-based task queue"""
    
    def __init__(self):
        self.tasks: List[Task] = []
        self.task_map: Dict[str, Task] = {}
    
    def enqueue(self, task: Task) -> None:
        """Add task to queue"""
        heapq.heappush(self.tasks, task)
        self.task_map[task.task_id] = task
    
    def dequeue(self) -> Optional[Task]:
        """Remove and return highest priority task"""
        if not self.tasks:
            return None
        
        task = heapq.heappop(self.tasks)
        return task
    
    def peek(self) -> Optional[Task]:
        """View highest priority task without removing"""
        return self.tasks[0] if self.tasks else None
    
    def get_ready_tasks(self, completed_tasks: set) -> List[Task]:
        """Get tasks ready for execution"""
        return [t for t in self.tasks if t.is_ready(completed_tasks)]
    
    def remove(self, task_id: str) -> bool:
        """Remove task from queue"""
        if task_id not in self.task_map:
            return False
        
        task = self.task_map[task_id]
        self.tasks.remove(task)
        heapq.heapify(self.tasks)
        del self.task_map[task_id]
        return True
    
    def size(self) -> int:
        """Get queue size"""
        return len(self.tasks)


class TaskAllocator:
    """Allocates tasks to agents"""
    
    def __init__(self, strategy: AllocationStrategy = AllocationStrategy.LEAST_LOADED):
        self.strategy = strategy
        self.round_robin_index = 0
    
    def allocate(self, task: Task, agents: List[Agent]) -> Optional[Agent]:
        """Allocate task to an agent"""
        # Filter agents that can handle the task
        capable_agents = [a for a in agents if a.can_handle(task)]
        
        if not capable_agents:
            return None
        
        if self.strategy == AllocationStrategy.ROUND_ROBIN:
            return self._round_robin(capable_agents)
        elif self.strategy == AllocationStrategy.LEAST_LOADED:
            return self._least_loaded(capable_agents)
        elif self.strategy == AllocationStrategy.CAPABILITY_MATCH:
            return self._capability_match(task, capable_agents)
        elif self.strategy == AllocationStrategy.DEADLINE_AWARE:
            return self._deadline_aware(task, capable_agents)
        
        return capable_agents[0]
    
    def _round_robin(self, agents: List[Agent]) -> Agent:
        """Round-robin allocation"""
        agent = agents[self.round_robin_index % len(agents)]
        self.round_robin_index += 1
        return agent
    
    def _least_loaded(self, agents: List[Agent]) -> Agent:
        """Allocate to least loaded agent"""
        return min(agents, key=lambda a: a.get_load())
    
    def _capability_match(self, task: Task, agents: List[Agent]) -> Agent:
        """Allocate to agent with best capability match"""
        def match_score(agent: Agent) -> int:
            # Prefer agents with exact capability match
            extra_capabilities = len(agent.capabilities) - len(task.required_capabilities)
            return extra_capabilities
        
        return min(agents, key=match_score)
    
    def _deadline_aware(self, task: Task, agents: List[Agent]) -> Agent:
        """Allocate considering deadlines and agent load"""
        if not task.deadline:
            return self._least_loaded(agents)
        
        # Calculate urgency score for each agent
        def urgency_score(agent: Agent) -> float:
            load_factor = agent.get_load()
            if task.deadline:
                time_to_deadline = (task.deadline - datetime.now()).total_seconds()
            else:
                time_to_deadline = 3600  # Default 1 hour
            
            # Higher score = more urgent
            urgency = 1.0 / max(time_to_deadline, 1.0)
            
            # Prefer less loaded agents for urgent tasks
            return load_factor + (urgency * 10.0)
        
        return min(agents, key=urgency_score)


class TaskScheduler:
    """
    Task Allocation & Scheduling System
    
    Manages task queues, agent allocation, and intelligent scheduling
    with priority handling and dependency resolution.
    """
    
    def __init__(
        self,
        name: str = "Task Scheduler",
        allocation_strategy: AllocationStrategy = AllocationStrategy.LEAST_LOADED
    ):
        self.name = name
        self.queue = TaskQueue()
        self.agents: Dict[str, Agent] = {}
        self.allocator = TaskAllocator(allocation_strategy)
        self.completed_tasks: set = set()
        self.failed_tasks: set = set()
        
        print(f"[Scheduler] Initialized: {name}")
        print(f"  Strategy: {allocation_strategy.value}")
    
    def register_agent(self, agent: Agent) -> None:
        """Register an agent"""
        self.agents[agent.agent_id] = agent
        print(f"[Scheduler] Registered agent: {agent.name}")
        print(f"  Capabilities: {', '.join(agent.capabilities)}")
        print(f"  Max concurrent: {agent.max_concurrent_tasks}")
    
    def submit_task(self, task: Task) -> None:
        """Submit a task for scheduling"""
        self.queue.enqueue(task)
        print(f"\n[Task Submitted] {task.name}")
        print(f"  Priority: {task.priority.name}")
        print(f"  Duration: {task.estimated_duration}s")
        if task.deadline:
            print(f"  Deadline: {task.deadline}")
        if task.dependencies:
            print(f"  Dependencies: {', '.join(task.dependencies)}")
    
    def schedule(self) -> List[str]:
        """Schedule ready tasks to available agents"""
        scheduled = []
        
        # Get tasks ready for execution
        ready_tasks = self.queue.get_ready_tasks(self.completed_tasks)
        
        for task in ready_tasks:
            if task.status != TaskStatus.PENDING:
                continue
            
            # Try to allocate agent
            agent = self.allocator.allocate(task, list(self.agents.values()))
            
            if agent:
                self._assign_task(task, agent)
                scheduled.append(task.task_id)
        
        return scheduled
    
    def _assign_task(self, task: Task, agent: Agent) -> None:
        """Assign task to agent"""
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent = agent.agent_id
        agent.assign_task(task.task_id)
        
        print(f"\n[Assignment] {task.name} → {agent.name}")
        print(f"  Agent load: {agent.get_load():.1%}")
    
    def execute_task(self, task_id: str, executor: Callable) -> bool:
        """Execute a task"""
        task = self.queue.task_map.get(task_id)
        if not task or task.status != TaskStatus.ASSIGNED:
            return False
        
        if not task.assigned_agent:
            return False
        
        agent = self.agents.get(task.assigned_agent)
        if not agent:
            return False
        
        print(f"\n[Executing] {task.name} on {agent.name}")
        
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            # Execute task
            start_time = time.time()
            result = executor(task)
            duration = time.time() - start_time
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            self.completed_tasks.add(task_id)
            agent.complete_task(task_id, duration)
            self.queue.remove(task_id)
            
            print(f"  ✓ Completed in {duration:.2f}s")
            return True
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            self.failed_tasks.add(task_id)
            agent.complete_task(task_id, 0)
            
            print(f"  ✗ Failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks) + self.queue.size()
        
        agent_stats = []
        for agent in self.agents.values():
            agent_stats.append({
                "name": agent.name,
                "status": agent.status.value,
                "load": agent.get_load(),
                "completed": agent.completed_tasks,
                "avg_time": agent.total_processing_time / agent.completed_tasks if agent.completed_tasks > 0 else 0
            })
        
        return {
            "total_tasks": total_tasks,
            "completed": len(self.completed_tasks),
            "failed": len(self.failed_tasks),
            "pending": self.queue.size(),
            "completion_rate": len(self.completed_tasks) / total_tasks if total_tasks > 0 else 0,
            "agents": agent_stats
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        task = self.queue.task_map.get(task_id)
        if not task or task.status not in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
            return False
        
        if task.assigned_agent:
            agent = self.agents.get(task.assigned_agent)
            if agent:
                agent.complete_task(task_id, 0)
        
        task.status = TaskStatus.CANCELLED
        self.queue.remove(task_id)
        
        print(f"[Cancelled] {task.name}")
        return True


def demonstrate_task_allocation():
    """
    Demonstrate Task Allocation & Scheduling pattern
    """
    print("=" * 70)
    print("TASK ALLOCATION & SCHEDULING DEMONSTRATION")
    print("=" * 70)
    
    # Create scheduler
    scheduler = TaskScheduler("Production Scheduler", AllocationStrategy.LEAST_LOADED)
    
    # Example 1: Register agents
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Agent Registration")
    print("=" * 70)
    
    agents = [
        Agent("agent-1", "Data Agent", ["data_processing", "analysis"], max_concurrent_tasks=2),
        Agent("agent-2", "Code Agent", ["code_generation", "debugging"], max_concurrent_tasks=3),
        Agent("agent-3", "General Agent", ["data_processing", "code_generation", "analysis"], max_concurrent_tasks=1),
    ]
    
    for agent in agents:
        scheduler.register_agent(agent)
    
    # Example 2: Submit tasks with priorities
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Task Submission with Priorities")
    print("=" * 70)
    
    tasks = [
        Task("task-1", "Analyze Data", TaskPriority.HIGH, 5.0, ["analysis"]),
        Task("task-2", "Generate Code", TaskPriority.MEDIUM, 3.0, ["code_generation"]),
        Task("task-3", "Process Dataset", TaskPriority.CRITICAL, 8.0, ["data_processing"]),
        Task("task-4", "Debug Issue", TaskPriority.LOW, 2.0, ["debugging"]),
        Task("task-5", "Complex Analysis", TaskPriority.HIGH, 6.0, ["analysis", "data_processing"]),
    ]
    
    for task in tasks:
        scheduler.submit_task(task)
    
    # Example 3: Schedule and execute tasks
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Task Scheduling and Execution")
    print("=" * 70)
    
    # Simple task executor
    def execute_task_logic(task: Task) -> str:
        time.sleep(0.1)  # Simulate work
        return f"Result of {task.name}"
    
    # Schedule tasks
    scheduled = scheduler.schedule()
    print(f"\n{len(scheduled)} tasks scheduled")
    
    # Execute scheduled tasks
    for task_id in scheduled:
        scheduler.execute_task(task_id, execute_task_logic)
    
    # Example 4: Task dependencies
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Dependent Tasks")
    print("=" * 70)
    
    dependent_tasks = [
        Task("dep-1", "Fetch Data", TaskPriority.HIGH, 2.0, ["data_processing"]),
        Task("dep-2", "Transform Data", TaskPriority.HIGH, 3.0, ["data_processing"], dependencies=["dep-1"]),
        Task("dep-3", "Analyze Results", TaskPriority.HIGH, 4.0, ["analysis"], dependencies=["dep-2"]),
    ]
    
    for task in dependent_tasks:
        scheduler.submit_task(task)
    
    # Schedule and execute in order
    for _ in range(3):
        scheduled = scheduler.schedule()
        if not scheduled:
            break
        
        for task_id in scheduled:
            scheduler.execute_task(task_id, execute_task_logic)
    
    # Example 5: Statistics
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Scheduler Statistics")
    print("=" * 70)
    
    stats = scheduler.get_statistics()
    
    print(f"\n[Overall Statistics]")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Pending: {stats['pending']}")
    print(f"  Completion rate: {stats['completion_rate']:.1%}")
    
    print(f"\n[Agent Performance]")
    for agent_stat in stats['agents']:
        print(f"  {agent_stat['name']}:")
        print(f"    Status: {agent_stat['status']}")
        print(f"    Load: {agent_stat['load']:.1%}")
        print(f"    Completed: {agent_stat['completed']}")
        print(f"    Avg time: {agent_stat['avg_time']:.2f}s")


def demonstrate_deadline_scheduling():
    """Demonstrate deadline-aware scheduling"""
    print("\n" + "=" * 70)
    print("DEADLINE-AWARE SCHEDULING")
    print("=" * 70)
    
    scheduler = TaskScheduler("Deadline Scheduler", AllocationStrategy.DEADLINE_AWARE)
    
    # Register agents
    agents = [
        Agent("urgent-1", "Fast Agent", ["processing"], max_concurrent_tasks=3),
        Agent("urgent-2", "Backup Agent", ["processing"], max_concurrent_tasks=2),
    ]
    
    for agent in agents:
        scheduler.register_agent(agent)
    
    # Submit tasks with deadlines
    now = datetime.now()
    tasks = [
        Task("urgent-1", "Critical Task", TaskPriority.CRITICAL, 2.0, 
             ["processing"], deadline=now + timedelta(seconds=5)),
        Task("urgent-2", "Normal Task", TaskPriority.MEDIUM, 3.0, 
             ["processing"], deadline=now + timedelta(seconds=30)),
        Task("urgent-3", "Low Priority", TaskPriority.LOW, 1.0, 
             ["processing"], deadline=now + timedelta(seconds=60)),
    ]
    
    for task in tasks:
        scheduler.submit_task(task)
    
    # Schedule based on deadlines
    scheduled = scheduler.schedule()
    print(f"\n{len(scheduled)} tasks scheduled by deadline urgency")


if __name__ == "__main__":
    demonstrate_task_allocation()
    demonstrate_deadline_scheduling()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Priority queues ensure important tasks execute first
2. Dependency tracking prevents premature execution
3. Load balancing optimizes resource utilization
4. Deadline awareness improves on-time completion
5. Multiple allocation strategies for different needs

Best Practices:
- Define clear task priorities and dependencies
- Monitor agent load and redistribute if needed
- Set realistic deadlines and estimated durations
- Handle task failures with retries or fallbacks
- Track metrics for optimization
- Consider capability matching for efficiency
- Implement fair scheduling policies
- Plan for agent failures and task reassignment
    """)
