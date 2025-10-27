"""
Hierarchical Planning Pattern
Breaks down goals into hierarchical sub-goals
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
class GoalStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
class GoalLevel(Enum):
    STRATEGIC = "strategic"      # High-level, long-term
    TACTICAL = "tactical"        # Mid-level, medium-term
    OPERATIONAL = "operational"  # Low-level, immediate
@dataclass
class Goal:
    """Hierarchical goal node"""
    id: str
    description: str
    level: GoalLevel
    status: GoalStatus = GoalStatus.PENDING
    parent_id: Optional[str] = None
    children: List['Goal'] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    def add_child(self, child: 'Goal'):
        """Add sub-goal"""
        child.parent_id = self.id
        self.children.append(child)
    def is_ready(self, completed_goals: set) -> bool:
        """Check if all dependencies are met"""
        return all(dep_id in completed_goals for dep_id in self.dependencies)
    def get_progress(self) -> float:
        """Calculate completion progress"""
        if not self.children:
            return 1.0 if self.status == GoalStatus.COMPLETED else 0.0
        return sum(child.get_progress() for child in self.children) / len(self.children)
class HierarchicalPlanner:
    """Planner that creates hierarchical goal structures"""
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.root_goals: List[Goal] = []
        self.completed_goals: set = set()
        self.goal_counter = 0
    def create_goal(self, description: str, level: GoalLevel, 
                    parent: Optional[Goal] = None, 
                    dependencies: List[str] = None) -> Goal:
        """Create a new goal"""
        self.goal_counter += 1
        goal_id = f"G{self.goal_counter:03d}"
        goal = Goal(
            id=goal_id,
            description=description,
            level=level,
            dependencies=dependencies or []
        )
        self.goals[goal_id] = goal
        if parent:
            parent.add_child(goal)
        else:
            self.root_goals.append(goal)
        return goal
    def decompose_goal(self, goal: Goal, decomposition: List[Dict[str, Any]]) -> List[Goal]:
        """Decompose a goal into sub-goals"""
        sub_goals = []
        # Determine sub-goal level
        if goal.level == GoalLevel.STRATEGIC:
            sub_level = GoalLevel.TACTICAL
        elif goal.level == GoalLevel.TACTICAL:
            sub_level = GoalLevel.OPERATIONAL
        else:
            sub_level = GoalLevel.OPERATIONAL
        for sub_goal_info in decomposition:
            sub_goal = self.create_goal(
                description=sub_goal_info['description'],
                level=sub_level,
                parent=goal,
                dependencies=sub_goal_info.get('dependencies', [])
            )
            sub_goals.append(sub_goal)
        print(f"Decomposed '{goal.description}' into {len(sub_goals)} sub-goals")
        return sub_goals
    def get_executable_goals(self) -> List[Goal]:
        """Get goals that are ready to execute"""
        executable = []
        for goal in self.goals.values():
            if (goal.status == GoalStatus.PENDING and
                not goal.children and  # Leaf node
                goal.is_ready(self.completed_goals)):
                executable.append(goal)
        return executable
    def execute_goal(self, goal: Goal) -> bool:
        """Execute a goal"""
        print(f"\nExecuting: [{goal.level.value}] {goal.description}")
        goal.status = GoalStatus.IN_PROGRESS
        # Simulate execution
        import time
        import random
        time.sleep(0.1)
        # Simulate success/failure
        success = random.random() > 0.1  # 90% success rate
        if success:
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = datetime.now()
            self.completed_goals.add(goal.id)
            print(f"  ✓ Completed")
            # Check if parent can be completed
            if goal.parent_id:
                self._check_parent_completion(goal.parent_id)
            return True
        else:
            goal.status = GoalStatus.FAILED
            print(f"  ✗ Failed")
            return False
    def _check_parent_completion(self, parent_id: str):
        """Check if parent goal can be marked complete"""
        parent = self.goals[parent_id]
        if all(child.status == GoalStatus.COMPLETED for child in parent.children):
            parent.status = GoalStatus.COMPLETED
            parent.completed_at = datetime.now()
            self.completed_goals.add(parent_id)
            print(f"  ✓ Parent goal completed: {parent.description}")
            # Recursively check parent's parent
            if parent.parent_id:
                self._check_parent_completion(parent.parent_id)
    def visualize_hierarchy(self, goal: Goal = None, indent: int = 0):
        """Print goal hierarchy"""
        if goal is None:
            print(f"\n{'='*70}")
            print("GOAL HIERARCHY")
            print(f"{'='*70}")
            for root in self.root_goals:
                self.visualize_hierarchy(root)
            return
        prefix = "  " * indent
        status_symbol = {
            GoalStatus.PENDING: "○",
            GoalStatus.IN_PROGRESS: "◐",
            GoalStatus.COMPLETED: "●",
            GoalStatus.FAILED: "✗",
            GoalStatus.BLOCKED: "⊗"
        }[goal.status]
        progress = goal.get_progress() * 100
        print(f"{prefix}{status_symbol} [{goal.level.value[0].upper()}] {goal.description} ({progress:.0f}%)")
        for child in goal.children:
            self.visualize_hierarchy(child, indent + 1)
    def get_statistics(self) -> Dict[str, Any]:
        """Get planning statistics"""
        total = len(self.goals)
        by_status = {status: 0 for status in GoalStatus}
        by_level = {level: 0 for level in GoalLevel}
        for goal in self.goals.values():
            by_status[goal.status] += 1
            by_level[goal.level] += 1
        return {
            'total_goals': total,
            'completed': by_status[GoalStatus.COMPLETED],
            'in_progress': by_status[GoalStatus.IN_PROGRESS],
            'pending': by_status[GoalStatus.PENDING],
            'failed': by_status[GoalStatus.FAILED],
            'by_level': {level.value: count for level, count in by_level.items()},
            'overall_progress': len(self.completed_goals) / total if total > 0 else 0
        }
class HierarchicalPlanningAgent:
    """Agent that uses hierarchical planning"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.planner = HierarchicalPlanner()
    def plan_and_execute(self, strategic_goal: str) -> Dict[str, Any]:
        """Create hierarchical plan and execute"""
        print(f"\n{'='*70}")
        print(f"HIERARCHICAL PLANNING: {strategic_goal}")
        print(f"{'='*70}")
        # Create strategic goal
        root = self.planner.create_goal(
            description=strategic_goal,
            level=GoalLevel.STRATEGIC
        )
        # Decompose into tactical goals
        tactical_decomposition = self._decompose_strategic(strategic_goal)
        tactical_goals = self.planner.decompose_goal(root, tactical_decomposition)
        # Decompose tactical into operational
        for tactical in tactical_goals:
            operational_decomposition = self._decompose_tactical(tactical.description)
            self.planner.decompose_goal(tactical, operational_decomposition)
        # Visualize initial plan
        self.planner.visualize_hierarchy()
        # Execute plan
        print(f"\n{'='*70}")
        print("EXECUTION")
        print(f"{'='*70}")
        max_iterations = 50
        iteration = 0
        while iteration < max_iterations:
            executable = self.planner.get_executable_goals()
            if not executable:
                # Check if done
                if root.status == GoalStatus.COMPLETED:
                    print(f"\n✓ Strategic goal achieved!")
                    break
                else:
                    print(f"\n⚠ No executable goals remaining")
                    break
            # Execute one goal
            goal = executable[0]
            self.planner.execute_goal(goal)
            iteration += 1
        # Final visualization
        self.planner.visualize_hierarchy()
        # Statistics
        stats = self.planner.get_statistics()
        print(f"\n{'='*70}")
        print("STATISTICS")
        print(f"{'='*70}")
        print(f"Total Goals: {stats['total_goals']}")
        print(f"Completed: {stats['completed']}")
        print(f"Failed: {stats['failed']}")
        print(f"Overall Progress: {stats['overall_progress']:.1%}")
        print(f"\nGoals by Level:")
        for level, count in stats['by_level'].items():
            print(f"  {level}: {count}")
        return stats
    def _decompose_strategic(self, strategic_goal: str) -> List[Dict[str, Any]]:
        """Decompose strategic goal into tactical goals"""
        # Simulated decomposition based on goal
        if "launch product" in strategic_goal.lower():
            return [
                {'description': "Develop product features"},
                {'description': "Create marketing campaign", 'dependencies': []},
                {'description': "Set up distribution", 'dependencies': []},
                {'description': "Train support team", 'dependencies': ['G002']},  # Depends on development
            ]
        else:
            return [
                {'description': f"Phase 1 of {strategic_goal}"},
                {'description': f"Phase 2 of {strategic_goal}", 'dependencies': []},
                {'description': f"Phase 3 of {strategic_goal}", 'dependencies': []},
            ]
    def _decompose_tactical(self, tactical_goal: str) -> List[Dict[str, Any]]:
        """Decompose tactical goal into operational goals"""
        # Simulated decomposition
        if "develop" in tactical_goal.lower():
            return [
                {'description': "Design architecture"},
                {'description': "Implement core features", 'dependencies': []},
                {'description': "Write tests", 'dependencies': []},
                {'description': "Conduct code review", 'dependencies': []},
            ]
        elif "marketing" in tactical_goal.lower():
            return [
                {'description': "Define target audience"},
                {'description': "Create content", 'dependencies': []},
                {'description': "Launch campaigns", 'dependencies': []},
            ]
        else:
            return [
                {'description': f"Step 1: {tactical_goal}"},
                {'description': f"Step 2: {tactical_goal}", 'dependencies': []},
                {'description': f"Step 3: {tactical_goal}", 'dependencies': []},
            ]
# Usage
if __name__ == "__main__":
    print("="*80)
    print("HIERARCHICAL PLANNING PATTERN DEMONSTRATION")
    print("="*80)
    agent = HierarchicalPlanningAgent("planner-001")
    # Plan and execute strategic goal
    result = agent.plan_and_execute(
        "Successfully launch new AI product to market"
    )
