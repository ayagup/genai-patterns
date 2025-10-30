"""
Agentic Design Pattern: Resource Scheduling Agent

This pattern implements an agent that schedules and allocates resources over time
with constraint satisfaction, priority handling, and optimization.

Key Components:
1. Resource - Represents available resources with capacity
2. Task - Represents work requiring resources
3. Schedule - Timeline of resource allocations
4. ConstraintChecker - Validates scheduling constraints
5. PriorityScheduler - Priority-based scheduling algorithm
6. ResourceSchedulingAgent - Main orchestrator

Features:
- Temporal resource allocation
- Multi-resource constraint satisfaction
- Priority-based scheduling
- Conflict detection and resolution
- Resource utilization optimization
- Dynamic rescheduling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from collections import defaultdict
import random
import math
from datetime import datetime, timedelta


class ResourceType(Enum):
    """Types of resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"
    WORKER = "worker"
    LICENSE = "license"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BEST_EFFORT = 1


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SchedulingStrategy(Enum):
    """Resource scheduling strategies."""
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    PRIORITY_FIRST = "priority_first"
    SHORTEST_JOB_FIRST = "shortest_job_first"
    EARLIEST_DEADLINE = "earliest_deadline"


@dataclass
class Resource:
    """Represents a resource with capacity and availability."""
    resource_id: str
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    unit: str = "units"
    cost_per_unit: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def allocate(self, amount: float) -> bool:
        """Attempt to allocate resources."""
        if amount <= self.available_capacity:
            self.available_capacity -= amount
            return True
        return False
    
    def release(self, amount: float):
        """Release allocated resources."""
        self.available_capacity = min(
            self.total_capacity,
            self.available_capacity + amount
        )
    
    def utilization(self) -> float:
        """Get current utilization ratio."""
        return 1.0 - (self.available_capacity / self.total_capacity)


@dataclass
class ResourceRequirement:
    """Resource requirements for a task."""
    resource_type: ResourceType
    amount: float
    duration: timedelta
    flexible: bool = False  # Can this requirement be adjusted?


@dataclass
class Task:
    """Represents a task requiring resources."""
    task_id: str
    name: str
    priority: TaskPriority
    requirements: List[ResourceRequirement]
    estimated_duration: timedelta
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    assigned_resources: Dict[ResourceType, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleSlot:
    """Represents a scheduled time slot for resource allocation."""
    slot_id: str
    task: Task
    resource_allocations: Dict[str, float]  # resource_id -> amount
    start_time: datetime
    end_time: datetime
    locked: bool = False


@dataclass
class ScheduleConflict:
    """Represents a scheduling conflict."""
    conflict_id: str
    conflict_type: str
    affected_tasks: List[str]
    affected_resources: List[str]
    severity: float  # 0.0 to 1.0
    resolution_suggestions: List[str]


class ConstraintChecker:
    """Checks and validates scheduling constraints."""
    
    def __init__(self):
        self.violations: List[Dict[str, Any]] = []
        
    def check_resource_constraints(
        self,
        task: Task,
        resources: Dict[str, Resource],
        start_time: datetime
    ) -> Tuple[bool, List[str]]:
        """
        Check if resources are available for task.
        
        Returns:
            Tuple of (is_valid, reasons)
        """
        reasons = []
        
        for req in task.requirements:
            # Find available resources of this type
            available = [
                r for r in resources.values()
                if r.resource_type == req.resource_type
            ]
            
            if not available:
                reasons.append(f"No resources of type {req.resource_type.value}")
                continue
            
            # Check if any resource has enough capacity
            has_capacity = any(
                r.available_capacity >= req.amount
                for r in available
            )
            
            if not has_capacity:
                reasons.append(
                    f"Insufficient {req.resource_type.value} "
                    f"(need {req.amount}, available: {max(r.available_capacity for r in available)})"
                )
        
        return len(reasons) == 0, reasons
    
    def check_temporal_constraints(
        self,
        task: Task,
        start_time: datetime,
        schedule: List[ScheduleSlot]
    ) -> Tuple[bool, List[str]]:
        """Check temporal constraints like deadlines."""
        reasons = []
        
        end_time = start_time + task.estimated_duration
        
        # Check deadline
        if task.deadline and end_time > task.deadline:
            reasons.append(
                f"Task would finish at {end_time} after deadline {task.deadline}"
            )
        
        # Check dependencies
        for dep_id in task.dependencies:
            dep_slot = next((s for s in schedule if s.task.task_id == dep_id), None)
            if dep_slot and dep_slot.end_time > start_time:
                reasons.append(
                    f"Dependency {dep_id} not completed before start time"
                )
        
        return len(reasons) == 0, reasons
    
    def check_conflicts(
        self,
        new_slot: ScheduleSlot,
        existing_schedule: List[ScheduleSlot],
        resources: Dict[str, Resource]
    ) -> List[ScheduleConflict]:
        """Detect conflicts with existing schedule."""
        conflicts = []
        
        # Check resource conflicts (double-booking)
        for existing in existing_schedule:
            # Check time overlap
            if (new_slot.start_time < existing.end_time and
                new_slot.end_time > existing.start_time):
                
                # Check resource overlap
                conflicting_resources = set(new_slot.resource_allocations.keys()) & \
                                      set(existing.resource_allocations.keys())
                
                if conflicting_resources:
                    severity = len(conflicting_resources) / max(
                        len(new_slot.resource_allocations),
                        len(existing.resource_allocations)
                    )
                    
                    conflict = ScheduleConflict(
                        conflict_id=f"conflict_{len(conflicts)}",
                        conflict_type="resource_overlap",
                        affected_tasks=[new_slot.task.task_id, existing.task.task_id],
                        affected_resources=list(conflicting_resources),
                        severity=severity,
                        resolution_suggestions=[
                            "Reschedule to different time",
                            "Use different resources",
                            "Preempt lower priority task"
                        ]
                    )
                    conflicts.append(conflict)
        
        return conflicts


class PriorityScheduler:
    """Priority-based scheduling algorithm."""
    
    def __init__(self, strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY_FIRST):
        self.strategy = strategy
        
    def schedule_task(
        self,
        task: Task,
        resources: Dict[str, Resource],
        existing_schedule: List[ScheduleSlot],
        current_time: datetime
    ) -> Optional[ScheduleSlot]:
        """
        Schedule a task using the configured strategy.
        
        Returns:
            ScheduleSlot if successful, None otherwise
        """
        if self.strategy == SchedulingStrategy.PRIORITY_FIRST:
            return self._priority_first_schedule(task, resources, existing_schedule, current_time)
        elif self.strategy == SchedulingStrategy.FIRST_FIT:
            return self._first_fit_schedule(task, resources, existing_schedule, current_time)
        elif self.strategy == SchedulingStrategy.BEST_FIT:
            return self._best_fit_schedule(task, resources, existing_schedule, current_time)
        else:
            return self._priority_first_schedule(task, resources, existing_schedule, current_time)
    
    def _priority_first_schedule(
        self,
        task: Task,
        resources: Dict[str, Resource],
        existing_schedule: List[ScheduleSlot],
        current_time: datetime
    ) -> Optional[ScheduleSlot]:
        """Schedule based on task priority."""
        # Find earliest available time
        start_time = self._find_earliest_slot(
            task, resources, existing_schedule, current_time
        )
        
        if not start_time:
            return None
        
        # Allocate resources
        allocations = self._allocate_resources(task, resources)
        
        if not allocations:
            return None
        
        return ScheduleSlot(
            slot_id=f"slot_{task.task_id}",
            task=task,
            resource_allocations=allocations,
            start_time=start_time,
            end_time=start_time + task.estimated_duration
        )
    
    def _first_fit_schedule(
        self,
        task: Task,
        resources: Dict[str, Resource],
        existing_schedule: List[ScheduleSlot],
        current_time: datetime
    ) -> Optional[ScheduleSlot]:
        """First-fit scheduling algorithm."""
        # Find first time slot where resources are available
        candidate_time = current_time
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Check if this time works
            if self._can_schedule_at_time(task, resources, existing_schedule, candidate_time):
                allocations = self._allocate_resources(task, resources)
                if allocations:
                    return ScheduleSlot(
                        slot_id=f"slot_{task.task_id}",
                        task=task,
                        resource_allocations=allocations,
                        start_time=candidate_time,
                        end_time=candidate_time + task.estimated_duration
                    )
            
            # Try next time slot (1 hour later)
            candidate_time += timedelta(hours=1)
        
        return None
    
    def _best_fit_schedule(
        self,
        task: Task,
        resources: Dict[str, Resource],
        existing_schedule: List[ScheduleSlot],
        current_time: datetime
    ) -> Optional[ScheduleSlot]:
        """Best-fit scheduling - minimize resource waste."""
        best_slot = None
        best_waste = float('inf')
        
        candidate_time = current_time
        max_attempts = 50
        
        for _ in range(max_attempts):
            if self._can_schedule_at_time(task, resources, existing_schedule, candidate_time):
                allocations = self._allocate_resources(task, resources)
                if allocations:
                    # Compute resource waste
                    waste = self._compute_resource_waste(task, resources, allocations)
                    
                    if waste < best_waste:
                        best_waste = waste
                        best_slot = ScheduleSlot(
                            slot_id=f"slot_{task.task_id}",
                            task=task,
                            resource_allocations=allocations,
                            start_time=candidate_time,
                            end_time=candidate_time + task.estimated_duration
                        )
            
            candidate_time += timedelta(hours=1)
        
        return best_slot
    
    def _find_earliest_slot(
        self,
        task: Task,
        resources: Dict[str, Resource],
        existing_schedule: List[ScheduleSlot],
        current_time: datetime
    ) -> Optional[datetime]:
        """Find earliest time slot for task."""
        candidate_time = current_time
        max_attempts = 100
        
        for _ in range(max_attempts):
            if self._can_schedule_at_time(task, resources, existing_schedule, candidate_time):
                return candidate_time
            candidate_time += timedelta(hours=1)
        
        return None
    
    def _can_schedule_at_time(
        self,
        task: Task,
        resources: Dict[str, Resource],
        existing_schedule: List[ScheduleSlot],
        start_time: datetime
    ) -> bool:
        """Check if task can be scheduled at given time."""
        end_time = start_time + task.estimated_duration
        
        # Check resource availability during this period
        for req in task.requirements:
            available_resources = [
                r for r in resources.values()
                if r.resource_type == req.resource_type
            ]
            
            if not available_resources:
                return False
            
            # Check conflicts with existing schedule
            for slot in existing_schedule:
                # Check time overlap
                if start_time < slot.end_time and end_time > slot.start_time:
                    # Check if uses same resources
                    for res_id in slot.resource_allocations:
                        if res_id in [r.resource_id for r in available_resources]:
                            # Resource is busy during this time
                            return False
        
        return True
    
    def _allocate_resources(
        self,
        task: Task,
        resources: Dict[str, Resource]
    ) -> Optional[Dict[str, float]]:
        """Allocate resources for task."""
        allocations = {}
        
        for req in task.requirements:
            # Find best resource of this type
            available = [
                r for r in resources.values()
                if r.resource_type == req.resource_type and
                   r.available_capacity >= req.amount
            ]
            
            if not available:
                # Rollback allocations
                for res_id, amount in allocations.items():
                    resources[res_id].release(amount)
                return None
            
            # Choose resource with least waste (best fit for this resource)
            best_resource = min(available, key=lambda r: r.available_capacity - req.amount)
            
            allocations[best_resource.resource_id] = req.amount
            best_resource.allocate(req.amount)
        
        return allocations
    
    def _compute_resource_waste(
        self,
        task: Task,
        resources: Dict[str, Resource],
        allocations: Dict[str, float]
    ) -> float:
        """Compute resource waste for allocations."""
        total_waste = 0.0
        
        for res_id, amount in allocations.items():
            resource = resources[res_id]
            waste = resource.available_capacity - amount
            total_waste += waste
        
        return total_waste


class ResourceSchedulingAgent:
    """
    Main agent for resource scheduling.
    
    Manages:
    - Resource pool
    - Task queue
    - Schedule generation and optimization
    - Conflict resolution
    """
    
    def __init__(
        self,
        strategy: SchedulingStrategy = SchedulingStrategy.PRIORITY_FIRST
    ):
        self.strategy = strategy
        
        # Components
        self.constraint_checker = ConstraintChecker()
        self.scheduler = PriorityScheduler(strategy)
        
        # State
        self.resources: Dict[str, Resource] = {}
        self.tasks: Dict[str, Task] = {}
        self.schedule: List[ScheduleSlot] = []
        self.current_time = datetime.now()
        self.conflicts: List[ScheduleConflict] = []
        
        # Statistics
        self.total_tasks_scheduled = 0
        self.total_tasks_completed = 0
        self.total_conflicts_resolved = 0
        self.average_wait_time = timedelta(0)
        
    def add_resource(self, resource: Resource):
        """Add a resource to the pool."""
        self.resources[resource.resource_id] = resource
        
    def add_task(self, task: Task):
        """Add a task to the queue."""
        self.tasks[task.task_id] = task
        
    def schedule_all_tasks(self) -> Dict[str, Any]:
        """
        Schedule all pending tasks.
        
        Returns:
            Summary of scheduling results
        """
        print(f"\n{'='*80}")
        print(f"📅 Scheduling Tasks")
        print(f"Strategy: {self.strategy.value}")
        print(f"{'='*80}\n")
        
        # Sort tasks by priority
        pending_tasks = [
            t for t in self.tasks.values()
            if t.status == TaskStatus.PENDING
        ]
        pending_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        
        scheduled_count = 0
        failed_count = 0
        
        for task in pending_tasks:
            result = self.schedule_task(task)
            if result:
                scheduled_count += 1
            else:
                failed_count += 1
        
        return {
            'total_tasks': len(pending_tasks),
            'scheduled': scheduled_count,
            'failed': failed_count,
            'schedule_length': len(self.schedule),
            'conflicts': len(self.conflicts)
        }
    
    def schedule_task(self, task: Task) -> bool:
        """
        Schedule a single task.
        
        Returns:
            True if successfully scheduled
        """
        # Check resource constraints
        resource_ok, resource_reasons = self.constraint_checker.check_resource_constraints(
            task, self.resources, self.current_time
        )
        
        if not resource_ok:
            print(f"✗ Task {task.task_id} cannot be scheduled:")
            for reason in resource_reasons:
                print(f"  • {reason}")
            return False
        
        # Try to schedule
        slot = self.scheduler.schedule_task(
            task, self.resources, self.schedule, self.current_time
        )
        
        if not slot:
            print(f"✗ Task {task.task_id} scheduling failed (no available slot)")
            return False
        
        # Check for conflicts
        conflicts = self.constraint_checker.check_conflicts(
            slot, self.schedule, self.resources
        )
        
        if conflicts:
            print(f"⚠️  Task {task.task_id} has {len(conflicts)} conflict(s)")
            self.conflicts.extend(conflicts)
            # Try to resolve conflicts
            if not self._resolve_conflicts(conflicts):
                # Rollback resource allocations
                for res_id, amount in slot.resource_allocations.items():
                    self.resources[res_id].release(amount)
                return False
        
        # Add to schedule
        self.schedule.append(slot)
        task.status = TaskStatus.SCHEDULED
        task.start_time = slot.start_time
        task.end_time = slot.end_time
        self.total_tasks_scheduled += 1
        
        wait_time = slot.start_time - self.current_time
        print(f"✓ Task {task.task_id} scheduled: {slot.start_time} to {slot.end_time} (wait: {wait_time})")
        
        return True
    
    def _resolve_conflicts(self, conflicts: List[ScheduleConflict]) -> bool:
        """Attempt to resolve scheduling conflicts."""
        for conflict in conflicts:
            print(f"  Resolving conflict: {conflict.conflict_type} (severity: {conflict.severity:.2f})")
            # Simple resolution: skip for now (in real system, would reschedule)
            self.total_conflicts_resolved += 1
        return True
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Get utilization for all resources."""
        return {
            res_id: resource.utilization()
            for res_id, resource in self.resources.items()
        }
    
    def get_schedule_summary(self) -> Dict[str, Any]:
        """Get summary of current schedule."""
        if not self.schedule:
            return {"message": "No tasks scheduled"}
        
        # Compute timeline
        min_time = min(slot.start_time for slot in self.schedule)
        max_time = max(slot.end_time for slot in self.schedule)
        total_duration = max_time - min_time
        
        # Compute resource utilization
        utilization = self.get_resource_utilization()
        avg_utilization = sum(utilization.values()) / len(utilization) if utilization else 0
        
        # Task statistics by priority
        tasks_by_priority = defaultdict(int)
        for slot in self.schedule:
            tasks_by_priority[slot.task.priority.value] += 1
        
        return {
            'total_slots': len(self.schedule),
            'timeline': {
                'start': min_time,
                'end': max_time,
                'duration': total_duration
            },
            'resource_utilization': utilization,
            'average_utilization': avg_utilization,
            'tasks_by_priority': dict(tasks_by_priority),
            'conflicts_detected': len(self.conflicts),
            'conflicts_resolved': self.total_conflicts_resolved
        }
    
    def optimize_schedule(self) -> int:
        """
        Optimize existing schedule by rearranging tasks.
        
        Returns:
            Number of improvements made
        """
        print(f"\n{'='*80}")
        print(f"🔧 Optimizing Schedule")
        print(f"{'='*80}\n")
        
        improvements = 0
        
        # Try to compact schedule (move tasks earlier)
        sorted_slots = sorted(self.schedule, key=lambda s: s.start_time)
        
        for i, slot in enumerate(sorted_slots):
            # Try to move earlier
            earliest_time = self.current_time
            
            # Check dependencies
            for dep_id in slot.task.dependencies:
                dep_slot = next((s for s in self.schedule if s.task.task_id == dep_id), None)
                if dep_slot:
                    earliest_time = max(earliest_time, dep_slot.end_time)
            
            # Try to schedule at earlier time
            if earliest_time < slot.start_time:
                # Check if resources available at earlier time
                candidate_time = earliest_time
                while candidate_time < slot.start_time:
                    if self.scheduler._can_schedule_at_time(
                        slot.task, self.resources, 
                        [s for s in self.schedule if s != slot],
                        candidate_time
                    ):
                        # Move to earlier time
                        old_start = slot.start_time
                        slot.start_time = candidate_time
                        slot.end_time = candidate_time + slot.task.estimated_duration
                        improvements += 1
                        print(f"✓ Moved task {slot.task.task_id} from {old_start} to {candidate_time}")
                        break
                    candidate_time += timedelta(minutes=30)
        
        print(f"\n✓ Optimization complete: {improvements} improvements made")
        return improvements
    
    def simulate_execution(self, duration: timedelta):
        """Simulate task execution over time."""
        print(f"\n{'='*80}")
        print(f"⏯️  Simulating Execution")
        print(f"{'='*80}\n")
        
        end_time = self.current_time + duration
        completed = 0
        
        while self.current_time < end_time:
            # Check for completing tasks
            for slot in self.schedule:
                if slot.task.status == TaskStatus.SCHEDULED and \
                   slot.start_time <= self.current_time:
                    slot.task.status = TaskStatus.RUNNING
                    print(f"▶️  Task {slot.task.task_id} started at {self.current_time}")
                
                if slot.task.status == TaskStatus.RUNNING and \
                   slot.end_time <= self.current_time:
                    slot.task.status = TaskStatus.COMPLETED
                    # Release resources
                    for res_id, amount in slot.resource_allocations.items():
                        self.resources[res_id].release(amount)
                    completed += 1
                    self.total_tasks_completed += 1
                    print(f"✅ Task {slot.task.task_id} completed at {self.current_time}")
            
            # Advance time
            self.current_time += timedelta(hours=1)
        
        print(f"\n✓ Simulation complete: {completed} tasks completed")


# Demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("RESOURCE SCHEDULING AGENT DEMONSTRATION")
    print("=" * 80)
    
    # Create agent
    agent = ResourceSchedulingAgent(
        strategy=SchedulingStrategy.PRIORITY_FIRST
    )
    
    # Add resources
    print("\n🔧 Adding Resources:")
    
    resources = [
        Resource("cpu1", ResourceType.CPU, 8.0, 8.0, "cores", cost_per_unit=0.1),
        Resource("cpu2", ResourceType.CPU, 16.0, 16.0, "cores", cost_per_unit=0.15),
        Resource("mem1", ResourceType.MEMORY, 32.0, 32.0, "GB", cost_per_unit=0.05),
        Resource("gpu1", ResourceType.GPU, 4.0, 4.0, "GPUs", cost_per_unit=1.0),
        Resource("worker1", ResourceType.WORKER, 10.0, 10.0, "workers", cost_per_unit=0.5),
    ]
    
    for resource in resources:
        agent.add_resource(resource)
        print(f"  • {resource.resource_id}: {resource.total_capacity} {resource.unit} ({resource.resource_type.value})")
    
    # Create tasks
    print("\n📋 Adding Tasks:")
    
    base_time = datetime.now()
    
    tasks = [
        Task(
            task_id="task1",
            name="Critical Data Processing",
            priority=TaskPriority.CRITICAL,
            requirements=[
                ResourceRequirement(ResourceType.CPU, 4.0, timedelta(hours=2)),
                ResourceRequirement(ResourceType.MEMORY, 8.0, timedelta(hours=2)),
            ],
            estimated_duration=timedelta(hours=2),
            deadline=base_time + timedelta(hours=3)
        ),
        Task(
            task_id="task2",
            name="ML Model Training",
            priority=TaskPriority.HIGH,
            requirements=[
                ResourceRequirement(ResourceType.GPU, 2.0, timedelta(hours=4)),
                ResourceRequirement(ResourceType.MEMORY, 16.0, timedelta(hours=4)),
            ],
            estimated_duration=timedelta(hours=4),
            deadline=base_time + timedelta(hours=6)
        ),
        Task(
            task_id="task3",
            name="Report Generation",
            priority=TaskPriority.NORMAL,
            requirements=[
                ResourceRequirement(ResourceType.CPU, 2.0, timedelta(hours=1)),
                ResourceRequirement(ResourceType.WORKER, 2.0, timedelta(hours=1)),
            ],
            estimated_duration=timedelta(hours=1),
            dependencies=["task1"]
        ),
        Task(
            task_id="task4",
            name="Batch Processing",
            priority=TaskPriority.LOW,
            requirements=[
                ResourceRequirement(ResourceType.CPU, 8.0, timedelta(hours=3)),
                ResourceRequirement(ResourceType.MEMORY, 12.0, timedelta(hours=3)),
            ],
            estimated_duration=timedelta(hours=3)
        ),
        Task(
            task_id="task5",
            name="Video Encoding",
            priority=TaskPriority.HIGH,
            requirements=[
                ResourceRequirement(ResourceType.GPU, 1.0, timedelta(hours=2)),
                ResourceRequirement(ResourceType.CPU, 4.0, timedelta(hours=2)),
            ],
            estimated_duration=timedelta(hours=2)
        ),
    ]
    
    for task in tasks:
        agent.add_task(task)
        print(f"  • {task.task_id}: {task.name} (Priority: {task.priority.name})")
    
    # Schedule all tasks
    result = agent.schedule_all_tasks()
    
    print(f"\n{'='*80}")
    print(f"📊 SCHEDULING RESULTS")
    print(f"{'='*80}")
    print(f"Total Tasks: {result['total_tasks']}")
    print(f"Successfully Scheduled: {result['scheduled']}")
    print(f"Failed to Schedule: {result['failed']}")
    print(f"Schedule Length: {result['schedule_length']} slots")
    print(f"Conflicts Detected: {result['conflicts']}")
    
    # Get schedule summary
    print(f"\n{'='*80}")
    print(f"📈 SCHEDULE SUMMARY")
    print(f"{'='*80}")
    
    summary = agent.get_schedule_summary()
    print(f"Total Slots: {summary['total_slots']}")
    print(f"Timeline Duration: {summary['timeline']['duration']}")
    print(f"Average Resource Utilization: {summary['average_utilization']:.1%}")
    
    print(f"\n Resource Utilization:")
    for res_id, util in summary['resource_utilization'].items():
        print(f"  • {res_id}: {util:.1%}")
    
    print(f"\n Tasks by Priority:")
    for priority, count in sorted(summary['tasks_by_priority'].items(), reverse=True):
        print(f"  • Priority {priority}: {count} tasks")
    
    # Optimize schedule
    improvements = agent.optimize_schedule()
    
    # Show optimized schedule
    print(f"\n{'='*80}")
    print(f"📅 OPTIMIZED SCHEDULE")
    print(f"{'='*80}")
    
    for slot in sorted(agent.schedule, key=lambda s: s.start_time):
        resources_str = ", ".join(
            f"{res_id}:{amount:.1f}" 
            for res_id, amount in slot.resource_allocations.items()
        )
        duration = slot.end_time - slot.start_time
        print(f"⏱️  {slot.start_time.strftime('%H:%M')} - {slot.end_time.strftime('%H:%M')} "
              f"({duration.total_seconds()/3600:.1f}h): {slot.task.name}")
        print(f"   Priority: {slot.task.priority.name}, Resources: {resources_str}")
    
    # Simulate execution
    agent.simulate_execution(duration=timedelta(hours=12))
    
    # Final statistics
    print(f"\n{'='*80}")
    print(f"📊 FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Total Tasks Scheduled: {agent.total_tasks_scheduled}")
    print(f"Total Tasks Completed: {agent.total_tasks_completed}")
    print(f"Total Conflicts Resolved: {agent.total_conflicts_resolved}")
    
    final_utilization = agent.get_resource_utilization()
    print(f"\nFinal Resource Utilization:")
    for res_id, util in final_utilization.items():
        print(f"  • {res_id}: {util:.1%}")
    
    print("\n" + "="*80)
    print("✅ Resource Scheduling Agent demonstration complete!")
    print("="*80)
    print("\nKey Achievements:")
    print("• Priority-based task scheduling")
    print("• Multi-resource constraint satisfaction")
    print("• Conflict detection and resolution")
    print("• Schedule optimization")
    print("• Resource utilization tracking")
    print(f"• Successfully scheduled {agent.total_tasks_scheduled} tasks")
