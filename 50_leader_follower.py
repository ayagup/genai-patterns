"""
Leader-Follower Multi-Agent Pattern
One agent leads while others assist or follow
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
class AgentRole(Enum):
    LEADER = "leader"
    FOLLOWER = "follower"
    SPECIALIST = "specialist"
class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
@dataclass
class Task:
    """A task to be completed"""
    id: str
    description: str
    required_skill: str
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    result: Any = None
@dataclass
class Agent:
    """An agent in the leader-follower system"""
    id: str
    name: str
    role: AgentRole
    skills: List[str]
    capacity: int = 3
    current_tasks: List[Task] = field(default_factory=list)
    completed_tasks: int = 0
    def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle task"""
        return (len(self.current_tasks) < self.capacity and 
                task.required_skill in self.skills)
    def assign_task(self, task: Task):
        """Assign task to agent"""
        task.assigned_to = self.id
        task.status = TaskStatus.ASSIGNED
        self.current_tasks.append(task)
    def complete_task(self, task: Task, result: Any):
        """Mark task as completed"""
        task.status = TaskStatus.COMPLETED
        task.result = result
        if task in self.current_tasks:
            self.current_tasks.remove(task)
        self.completed_tasks += 1
class LeaderAgent(Agent):
    """Leader agent that coordinates others"""
    def __init__(self, id: str, name: str, skills: List[str]):
        super().__init__(id, name, AgentRole.LEADER, skills, capacity=5)
        self.followers: List[Agent] = []
    def add_follower(self, follower: Agent):
        """Add a follower agent"""
        self.followers.append(follower)
        print(f"[{self.name}] Added follower: {follower.name} ({follower.role.value})")
    def delegate_task(self, task: Task) -> bool:
        """Delegate task to best available follower"""
        print(f"\n[{self.name}] Delegating task: {task.description}")
        print(f"  Required skill: {task.required_skill}")
        # Find capable followers
        capable = [f for f in self.followers if f.can_handle_task(task)]
        if not capable:
            print(f"  ⚠ No capable followers available")
            # Leader handles it if possible
            if self.can_handle_task(task):
                print(f"  → Leader will handle it")
                self.assign_task(task)
                return True
            else:
                print(f"  ✗ Cannot delegate")
                return False
        # Select best follower (least loaded)
        best_follower = min(capable, key=lambda f: len(f.current_tasks))
        print(f"  → Assigned to: {best_follower.name}")
        best_follower.assign_task(task)
        return True
    def coordinate(self, tasks: List[Task]):
        """Coordinate task execution"""
        print(f"\n{'='*70}")
        print(f"LEADER COORDINATION: {self.name}")
        print(f"{'='*70}")
        print(f"Total tasks: {len(tasks)}")
        print(f"Followers: {len(self.followers)}\n")
        # Prioritize tasks
        tasks.sort(key=lambda t: t.priority, reverse=True)
        # Delegate tasks
        for task in tasks:
            self.delegate_task(task)
        # Execute tasks
        print(f"\n{'='*70}")
        print("EXECUTION PHASE")
        print(f"{'='*70}\n")
        self._execute_all_tasks()
        # Report results
        self._report_results(tasks)
    def _execute_all_tasks(self):
        """Execute all assigned tasks"""
        # Execute leader's tasks
        for task in self.current_tasks[:]:
            result = self._execute_task(task)
            self.complete_task(task, result)
        # Execute followers' tasks
        for follower in self.followers:
            for task in follower.current_tasks[:]:
                result = self._execute_task(task)
                follower.complete_task(task, result)
    def _execute_task(self, task: Task) -> Any:
        """Execute a task"""
        print(f"[{task.assigned_to}] Executing: {task.description}")
        # Simulate task execution
        import time
        import random
        time.sleep(0.1)
        # Simulate occasional failure
        if random.random() < 0.1:
            task.status = TaskStatus.FAILED
            print(f"  ✗ Failed")
            return None
        result = f"Result for {task.description}"
        print(f"  ✓ Completed")
        return result
    def _report_results(self, tasks: List[Task]):
        """Report coordination results"""
        print(f"\n{'='*70}")
        print("COORDINATION RESULTS")
        print(f"{'='*70}\n")
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        print(f"Total Tasks: {len(tasks)}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {completed/len(tasks):.1%}")
        print(f"\nAgent Performance:")
        print(f"  {self.name} (Leader): {self.completed_tasks} tasks")
        for follower in self.followers:
            print(f"  {follower.name} ({follower.role.value}): {follower.completed_tasks} tasks")
        # Task breakdown by skill
        print(f"\nTasks by Skill:")
        skills = {}
        for task in tasks:
            skill = task.required_skill
            skills[skill] = skills.get(skill, 0) + 1
        for skill, count in skills.items():
            print(f"  {skill}: {count}")
class LeaderFollowerSystem:
    """Leader-follower multi-agent system"""
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.leaders: List[LeaderAgent] = []
        self.tasks: List[Task] = []
    def add_leader(self, leader: LeaderAgent):
        """Add leader to system"""
        self.leaders.append(leader)
        print(f"Added leader: {leader.name}")
    def create_team(self, leader: LeaderAgent, followers: List[Agent]):
        """Create team with leader and followers"""
        print(f"\n{'='*60}")
        print(f"CREATING TEAM: {leader.name}'s Team")
        print(f"{'='*60}\n")
        for follower in followers:
            leader.add_follower(follower)
        if leader not in self.leaders:
            self.add_leader(leader)
    def add_tasks(self, tasks: List[Task]):
        """Add tasks to system"""
        self.tasks.extend(tasks)
        print(f"Added {len(tasks)} tasks to system")
    def execute(self):
        """Execute all tasks"""
        print(f"\n{'='*70}")
        print(f"LEADER-FOLLOWER SYSTEM: {self.system_name}")
        print(f"{'='*70}\n")
        for leader in self.leaders:
            # Assign tasks to this leader's team
            leader.coordinate(self.tasks)
# Usage
if __name__ == "__main__":
    print("="*80)
    print("LEADER-FOLLOWER PATTERN DEMONSTRATION")
    print("="*80)
    # Create leader
    leader = LeaderAgent(
        id="leader_1",
        name="Project Manager",
        skills=["planning", "coordination", "coding"]
    )
    # Create follower agents
    followers = [
        Agent(
            id="dev_1",
            name="Backend Developer",
            role=AgentRole.FOLLOWER,
            skills=["coding", "database", "api"],
            capacity=3
        ),
        Agent(
            id="dev_2",
            name="Frontend Developer",
            role=AgentRole.FOLLOWER,
            skills=["coding", "ui", "design"],
            capacity=3
        ),
        Agent(
            id="specialist_1",
            name="DevOps Engineer",
            role=AgentRole.SPECIALIST,
            skills=["deployment", "infrastructure", "monitoring"],
            capacity=2
        ),
        Agent(
            id="tester_1",
            name="QA Tester",
            role=AgentRole.SPECIALIST,
            skills=["testing", "quality_assurance"],
            capacity=4
        )
    ]
    # Create system
    system = LeaderFollowerSystem("Software Development Team")
    # Build team
    system.create_team(leader, followers)
    # Create tasks
    tasks = [
        Task("T1", "Implement user authentication", "coding", priority=5),
        Task("T2", "Design user interface", "ui", priority=4),
        Task("T3", "Set up database", "database", priority=5),
        Task("T4", "Create API endpoints", "api", priority=4),
        Task("T5", "Deploy to production", "deployment", priority=3),
        Task("T6", "Write unit tests", "testing", priority=3),
        Task("T7", "Set up monitoring", "monitoring", priority=2),
        Task("T8", "Quality assurance", "quality_assurance", priority=3),
        Task("T9", "Implement dashboard", "ui", priority=2),
        Task("T10", "Optimize database queries", "database", priority=2),
    ]
    system.add_tasks(tasks)
    # Execute
    system.execute()
