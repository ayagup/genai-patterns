"""
Temporal Planning Pattern

Enables agents to plan with explicit time constraints, deadlines, and temporal
relationships between actions. Handles scheduling, duration estimation, and
time-dependent resources.

Key Concepts:
- Time constraints and deadlines
- Temporal relationships (before, after, during)
- Resource availability over time
- Schedule optimization
- Temporal reasoning

Use Cases:
- Project scheduling
- Resource booking systems
- Manufacturing planning
- Meeting coordination
- Task prioritization with deadlines
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import uuid


class TemporalRelation(Enum):
    """Temporal relationships between actions."""
    BEFORE = "before"  # A must finish before B starts
    AFTER = "after"  # A starts after B finishes
    DURING = "during"  # A happens during B
    MEETS = "meets"  # A finishes exactly when B starts
    OVERLAPS = "overlaps"  # A and B overlap in time
    STARTS = "starts"  # A and B start at same time
    FINISHES = "finishes"  # A and B finish at same time
    EQUALS = "equals"  # A and B have same start and end


class PriorityLevel(Enum):
    """Priority levels for actions."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ActionStatus(Enum):
    """Status of an action in schedule."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TimeWindow:
    """Time window for action execution."""
    start: datetime
    end: datetime
    
    def duration(self) -> timedelta:
        """Get duration of time window."""
        return self.end - self.start
    
    def overlaps(self, other: 'TimeWindow') -> bool:
        """Check if this window overlaps with another."""
        return (self.start < other.end) and (other.start < self.end)
    
    def contains(self, time: datetime) -> bool:
        """Check if time falls within window."""
        return self.start <= time <= self.end


@dataclass
class Resource:
    """A resource that can be allocated over time."""
    resource_id: str
    name: str
    capacity: float
    availability: Dict[datetime, float] = field(default_factory=dict)
    
    def is_available(self, time_window: TimeWindow, required: float) -> bool:
        """Check if resource is available for required amount during window."""
        # Simplified: check if average availability meets requirement
        return required <= self.capacity


@dataclass
class TemporalAction:
    """An action with temporal properties."""
    action_id: str
    name: str
    description: str
    duration: timedelta
    deadline: Optional[datetime] = None
    earliest_start: Optional[datetime] = None
    priority: PriorityLevel = PriorityLevel.MEDIUM
    resources_required: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Action IDs
    temporal_constraints: Dict[str, Tuple[TemporalRelation, str]] = field(default_factory=dict)
    status: ActionStatus = ActionStatus.PENDING
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    
    def get_time_window(self) -> Optional[TimeWindow]:
        """Get scheduled time window."""
        if self.scheduled_start and self.scheduled_end:
            return TimeWindow(self.scheduled_start, self.scheduled_end)
        return None


@dataclass
class Schedule:
    """A schedule of actions over time."""
    schedule_id: str
    name: str
    actions: List[TemporalAction] = field(default_factory=list)
    makespan: Optional[timedelta] = None
    total_delay: timedelta = timedelta(0)
    feasible: bool = True
    conflicts: List[str] = field(default_factory=list)
    
    def calculate_makespan(self) -> timedelta:
        """Calculate total time from start to finish."""
        if not self.actions:
            return timedelta(0)
        
        scheduled_actions = [a for a in self.actions if a.scheduled_start and a.scheduled_end]
        if not scheduled_actions:
            return timedelta(0)
        
        starts = [a.scheduled_start for a in scheduled_actions if a.scheduled_start]
        ends = [a.scheduled_end for a in scheduled_actions if a.scheduled_end]
        
        if not starts or not ends:
            return timedelta(0)
        
        earliest = min(starts)
        latest = max(ends)
        self.makespan = latest - earliest
        return self.makespan
    
    def calculate_delays(self) -> timedelta:
        """Calculate total delays beyond deadlines."""
        total = timedelta(0)
        for action in self.actions:
            if action.deadline and action.scheduled_end:
                if action.scheduled_end > action.deadline:
                    total += action.scheduled_end - action.deadline
        self.total_delay = total
        return total


class TemporalPlanner:
    """Planner that handles temporal constraints and scheduling."""
    
    def __init__(self, planner_id: str, name: str):
        self.planner_id = planner_id
        self.name = name
        self.actions: Dict[str, TemporalAction] = {}
        self.resources: Dict[str, Resource] = {}
        self.schedules: List[Schedule] = []
        self.current_time: datetime = datetime.now()
    
    def add_action(self, action: TemporalAction) -> None:
        """Add an action to plan."""
        self.actions[action.action_id] = action
        print(f"[{self.name}] Added action: {action.name}")
        print(f"  Duration: {action.duration}")
        print(f"  Priority: {action.priority.name}")
        if action.deadline:
            print(f"  Deadline: {action.deadline}")
    
    def add_resource(self, resource: Resource) -> None:
        """Add a resource."""
        self.resources[resource.resource_id] = resource
    
    def add_temporal_constraint(
        self,
        action1_id: str,
        relation: TemporalRelation,
        action2_id: str
    ) -> None:
        """Add temporal constraint between two actions."""
        if action1_id in self.actions:
            self.actions[action1_id].temporal_constraints[action2_id] = (relation, action2_id)
            print(f"[{self.name}] Constraint: {action1_id} {relation.value} {action2_id}")
    
    def schedule_earliest_start(self) -> Schedule:
        """Schedule actions as early as possible."""
        print(f"\n[{self.name}] Scheduling with earliest-start strategy...")
        
        schedule = Schedule(
            schedule_id=str(uuid.uuid4()),
            name="Earliest-Start Schedule"
        )
        
        # Sort by priority and earliest start time
        sorted_actions = sorted(
            self.actions.values(),
            key=lambda a: (
                -a.priority.value,
                a.earliest_start or self.current_time,
                a.deadline or datetime.max
            )
        )
        
        scheduled = set()
        
        for action in sorted_actions:
            # Find earliest feasible start time
            start_time = self._find_earliest_start(action, scheduled)
            
            if start_time:
                action.scheduled_start = start_time
                action.scheduled_end = start_time + action.duration
                action.status = ActionStatus.SCHEDULED
                scheduled.add(action.action_id)
                schedule.actions.append(action)
                
                print(f"  Scheduled: {action.name}")
                print(f"    Start: {action.scheduled_start}")
                print(f"    End: {action.scheduled_end}")
            else:
                schedule.feasible = False
                schedule.conflicts.append(f"Cannot schedule: {action.name}")
        
        schedule.calculate_makespan()
        schedule.calculate_delays()
        self.schedules.append(schedule)
        
        return schedule
    
    def _find_earliest_start(
        self,
        action: TemporalAction,
        scheduled: Set[str]
    ) -> Optional[datetime]:
        """Find earliest feasible start time for action."""
        # Start with earliest allowed time
        earliest = action.earliest_start or self.current_time
        
        # Consider dependencies
        for dep_id in action.dependencies:
            if dep_id in scheduled and dep_id in self.actions:
                dep_action = self.actions[dep_id]
                if dep_action.scheduled_end:
                    earliest = max(earliest, dep_action.scheduled_end)
        
        # Consider temporal constraints
        for other_id, (relation, _) in action.temporal_constraints.items():
            if other_id in scheduled and other_id in self.actions:
                other = self.actions[other_id]
                if other.scheduled_start and other.scheduled_end:
                    if relation == TemporalRelation.AFTER:
                        earliest = max(earliest, other.scheduled_end)
                    elif relation == TemporalRelation.BEFORE:
                        # This action must finish before other starts
                        # So start time must allow finishing before other.scheduled_start
                        latest_start = other.scheduled_start - action.duration
                        if latest_start < earliest:
                            return None  # Infeasible
        
        # Check deadline feasibility
        if action.deadline:
            if earliest + action.duration > action.deadline:
                return None  # Cannot meet deadline
        
        return earliest
    
    def schedule_deadline_driven(self) -> Schedule:
        """Schedule actions working backward from deadlines."""
        print(f"\n[{self.name}] Scheduling with deadline-driven strategy...")
        
        schedule = Schedule(
            schedule_id=str(uuid.uuid4()),
            name="Deadline-Driven Schedule"
        )
        
        # Sort by deadline (actions with deadlines first)
        sorted_actions = sorted(
            self.actions.values(),
            key=lambda a: (
                a.deadline is None,  # Deadlines first
                a.deadline or datetime.max,
                -a.priority.value
            )
        )
        
        scheduled = set()
        
        for action in sorted_actions:
            # Try to schedule as late as possible before deadline
            start_time = self._find_latest_start(action, scheduled)
            
            if start_time:
                action.scheduled_start = start_time
                action.scheduled_end = start_time + action.duration
                action.status = ActionStatus.SCHEDULED
                scheduled.add(action.action_id)
                schedule.actions.append(action)
                
                print(f"  Scheduled: {action.name}")
                print(f"    Start: {action.scheduled_start}")
                print(f"    End: {action.scheduled_end}")
            else:
                schedule.feasible = False
                schedule.conflicts.append(f"Cannot schedule: {action.name}")
        
        schedule.calculate_makespan()
        schedule.calculate_delays()
        self.schedules.append(schedule)
        
        return schedule
    
    def _find_latest_start(
        self,
        action: TemporalAction,
        scheduled: Set[str]
    ) -> Optional[datetime]:
        """Find latest feasible start time for action."""
        # Start with deadline or far future
        if action.deadline:
            latest = action.deadline - action.duration
        else:
            latest = self.current_time + timedelta(days=365)  # Default: 1 year
        
        # Consider earliest start constraint
        if action.earliest_start:
            latest = max(latest, action.earliest_start)
        
        # Check dependencies and constraints
        for dep_id in action.dependencies:
            if dep_id in scheduled and dep_id in self.actions:
                dep_action = self.actions[dep_id]
                if dep_action.scheduled_end:
                    latest = max(latest, dep_action.scheduled_end)
        
        return latest
    
    def optimize_schedule(
        self,
        schedule: Schedule,
        iterations: int = 10
    ) -> Schedule:
        """Optimize schedule to minimize makespan and delays."""
        print(f"\n[{self.name}] Optimizing schedule...")
        
        best_schedule = schedule
        best_score = self._evaluate_schedule(schedule)
        
        for i in range(iterations):
            # Try swapping action order
            new_schedule = self._perturb_schedule(schedule)
            score = self._evaluate_schedule(new_schedule)
            
            if score < best_score:
                best_schedule = new_schedule
                best_score = score
                print(f"  Iteration {i+1}: Improved score to {best_score:.2f}")
        
        return best_schedule
    
    def _evaluate_schedule(self, schedule: Schedule) -> float:
        """Evaluate schedule quality (lower is better)."""
        makespan = schedule.calculate_makespan().total_seconds()
        delays = schedule.calculate_delays().total_seconds()
        
        # Weighted sum: prioritize minimizing delays
        return makespan + 10 * delays
    
    def _perturb_schedule(self, schedule: Schedule) -> Schedule:
        """Create perturbed version of schedule."""
        import random
        
        new_schedule = Schedule(
            schedule_id=str(uuid.uuid4()),
            name=f"Perturbed {schedule.name}"
        )
        
        # Copy actions and randomly adjust start times slightly
        for action in schedule.actions:
            new_action = TemporalAction(
                action_id=action.action_id,
                name=action.name,
                description=action.description,
                duration=action.duration,
                deadline=action.deadline,
                earliest_start=action.earliest_start,
                priority=action.priority,
                resources_required=action.resources_required,
                dependencies=action.dependencies,
                temporal_constraints=action.temporal_constraints
            )
            
            if action.scheduled_start:
                # Small random adjustment
                adjustment = timedelta(minutes=random.randint(-30, 30))
                new_action.scheduled_start = action.scheduled_start + adjustment
                new_action.scheduled_end = new_action.scheduled_start + new_action.duration
            
            new_schedule.actions.append(new_action)
        
        return new_schedule
    
    def detect_conflicts(self, schedule: Schedule) -> List[str]:
        """Detect scheduling conflicts."""
        conflicts = []
        
        # Check resource conflicts
        for i, action1 in enumerate(schedule.actions):
            window1 = action1.get_time_window()
            if not window1:
                continue
            
            for action2 in schedule.actions[i+1:]:
                window2 = action2.get_time_window()
                if not window2:
                    continue
                
                # Check if actions overlap and use same resources
                if window1.overlaps(window2):
                    shared_resources = set(action1.resources_required.keys()) & set(action2.resources_required.keys())
                    if shared_resources:
                        conflicts.append(
                            f"Resource conflict: {action1.name} and {action2.name} "
                            f"overlap and both need {shared_resources}"
                        )
        
        # Check deadline violations
        for action in schedule.actions:
            if action.deadline and action.scheduled_end:
                if action.scheduled_end > action.deadline:
                    delay = action.scheduled_end - action.deadline
                    conflicts.append(
                        f"Deadline violation: {action.name} finishes {delay} after deadline"
                    )
        
        return conflicts


def demonstrate_temporal_planning():
    """Demonstrate temporal planning pattern."""
    print("=" * 60)
    print("TEMPORAL PLANNING DEMONSTRATION")
    print("=" * 60)
    
    # Create planner
    planner = TemporalPlanner("planner1", "Project Scheduler")
    planner.current_time = datetime(2024, 1, 1, 9, 0)  # Start of day
    
    # Define actions
    print("\n" + "=" * 60)
    print("1. Defining Actions with Temporal Properties")
    print("=" * 60)
    
    actions = [
        TemporalAction(
            action_id="a1",
            name="Requirements Analysis",
            description="Gather and analyze requirements",
            duration=timedelta(hours=8),
            deadline=datetime(2024, 1, 2, 17, 0),
            earliest_start=datetime(2024, 1, 1, 9, 0),
            priority=PriorityLevel.HIGH
        ),
        TemporalAction(
            action_id="a2",
            name="Design Phase",
            description="Create system design",
            duration=timedelta(hours=16),
            deadline=datetime(2024, 1, 5, 17, 0),
            priority=PriorityLevel.HIGH,
            dependencies=["a1"]
        ),
        TemporalAction(
            action_id="a3",
            name="Implementation",
            description="Code the system",
            duration=timedelta(hours=40),
            deadline=datetime(2024, 1, 12, 17, 0),
            priority=PriorityLevel.CRITICAL,
            dependencies=["a2"]
        ),
        TemporalAction(
            action_id="a4",
            name="Testing",
            description="Test the system",
            duration=timedelta(hours=16),
            deadline=datetime(2024, 1, 15, 17, 0),
            priority=PriorityLevel.CRITICAL,
            dependencies=["a3"]
        ),
        TemporalAction(
            action_id="a5",
            name="Documentation",
            description="Write documentation",
            duration=timedelta(hours=12),
            deadline=datetime(2024, 1, 15, 17, 0),
            priority=PriorityLevel.MEDIUM,
            dependencies=["a2"]
        ),
    ]
    
    for action in actions:
        planner.add_action(action)
    
    # Add temporal constraints
    print("\n" + "=" * 60)
    print("2. Adding Temporal Constraints")
    print("=" * 60)
    
    planner.add_temporal_constraint("a1", TemporalRelation.BEFORE, "a2")
    planner.add_temporal_constraint("a2", TemporalRelation.BEFORE, "a3")
    planner.add_temporal_constraint("a3", TemporalRelation.BEFORE, "a4")
    planner.add_temporal_constraint("a5", TemporalRelation.AFTER, "a2")
    
    # Strategy 1: Earliest-start scheduling
    print("\n" + "=" * 60)
    print("3. Earliest-Start Scheduling")
    print("=" * 60)
    
    schedule1 = planner.schedule_earliest_start()
    
    print(f"\nSchedule Summary:")
    print(f"  Feasible: {schedule1.feasible}")
    print(f"  Makespan: {schedule1.makespan}")
    print(f"  Total Delays: {schedule1.total_delay}")
    print(f"  Actions Scheduled: {len(schedule1.actions)}")
    
    # Strategy 2: Deadline-driven scheduling
    print("\n" + "=" * 60)
    print("4. Deadline-Driven Scheduling")
    print("=" * 60)
    
    schedule2 = planner.schedule_deadline_driven()
    
    print(f"\nSchedule Summary:")
    print(f"  Feasible: {schedule2.feasible}")
    print(f"  Makespan: {schedule2.makespan}")
    print(f"  Total Delays: {schedule2.total_delay}")
    print(f"  Actions Scheduled: {len(schedule2.actions)}")
    
    # Detect conflicts
    print("\n" + "=" * 60)
    print("5. Conflict Detection")
    print("=" * 60)
    
    conflicts1 = planner.detect_conflicts(schedule1)
    print(f"\nEarliest-Start Schedule Conflicts: {len(conflicts1)}")
    for conflict in conflicts1:
        print(f"  - {conflict}")
    
    conflicts2 = planner.detect_conflicts(schedule2)
    print(f"\nDeadline-Driven Schedule Conflicts: {len(conflicts2)}")
    for conflict in conflicts2:
        print(f"  - {conflict}")
    
    # Comparison
    print("\n" + "=" * 60)
    print("6. Schedule Comparison")
    print("=" * 60)
    
    print("\n  Earliest-Start vs Deadline-Driven:")
    print(f"    Makespan: {schedule1.makespan} vs {schedule2.makespan}")
    print(f"    Delays: {schedule1.total_delay} vs {schedule2.total_delay}")
    print(f"    Conflicts: {len(conflicts1)} vs {len(conflicts2)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("7. Summary")
    print("=" * 60)
    
    print(f"\nActions: {len(planner.actions)}")
    print(f"Schedules Generated: {len(planner.schedules)}")
    print(f"\nDemonstrated:")
    print(f"  ✓ Temporal action definition")
    print(f"  ✓ Temporal constraints")
    print(f"  ✓ Earliest-start scheduling")
    print(f"  ✓ Deadline-driven scheduling")
    print(f"  ✓ Conflict detection")
    print(f"  ✓ Schedule comparison")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_temporal_planning()
