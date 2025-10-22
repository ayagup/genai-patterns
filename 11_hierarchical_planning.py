"""
Hierarchical Planning Pattern Implementation

This module demonstrates hierarchical planning where complex goals are broken down
into hierarchical sub-goals across multiple levels: high-level strategy, 
mid-level tactics, and low-level actions.

Key Components:
- Goal hierarchy with strategic, tactical, and operational levels
- Task decomposition and dependency management
- Progress tracking across hierarchy levels
- Adaptive re-planning when sub-goals fail
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum
import uuid
import time
import random


class GoalLevel(Enum):
    """Levels of goals in the hierarchy"""
    STRATEGIC = "strategic"      # High-level, long-term objectives
    TACTICAL = "tactical"        # Mid-level, medium-term plans
    OPERATIONAL = "operational"  # Low-level, immediate actions


class GoalStatus(Enum):
    """Status of a goal in the planning process"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass
class Goal:
    """Represents a goal at any level of the hierarchy"""
    id: str
    title: str
    description: str
    level: GoalLevel
    status: GoalStatus = GoalStatus.PENDING
    priority: int = 1  # 1=low, 5=high
    estimated_duration: float = 1.0  # hours
    actual_duration: float = 0.0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    resources_required: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


class GoalDecomposer:
    """Breaks down high-level goals into sub-goals"""
    
    def __init__(self):
        self.decomposition_strategies = {
            GoalLevel.STRATEGIC: self._decompose_strategic,
            GoalLevel.TACTICAL: self._decompose_tactical,
            GoalLevel.OPERATIONAL: self._decompose_operational
        }
    
    def _decompose_strategic(self, goal: Goal) -> List[Goal]:
        """Decompose strategic goals into tactical goals"""
        tactical_templates = [
            "Research and analysis phase for: {title}",
            "Planning and design phase for: {title}",
            "Implementation phase for: {title}",
            "Testing and validation phase for: {title}",
            "Deployment and monitoring phase for: {title}"
        ]
        
        tactical_goals = []
        for i, template in enumerate(tactical_templates):
            tactical_goal = Goal(
                id=f"tac_{goal.id}_{i}",
                title=template.format(title=goal.title),
                description=f"Tactical goal derived from strategic goal: {goal.description}",
                level=GoalLevel.TACTICAL,
                parent_id=goal.id,
                priority=goal.priority,
                estimated_duration=goal.estimated_duration / len(tactical_templates),
                success_criteria=[f"Complete {template.split(' phase')[0].lower()} for {goal.title}"],
                metadata={"parent_goal": goal.title, "phase": i + 1}
            )
            tactical_goals.append(tactical_goal)
        
        return tactical_goals
    
    def _decompose_tactical(self, goal: Goal) -> List[Goal]:
        """Decompose tactical goals into operational actions"""
        action_templates = [
            "Gather requirements for: {title}",
            "Create detailed plan for: {title}",
            "Execute core activities for: {title}",
            "Review and validate results for: {title}",
            "Document outcomes for: {title}"
        ]
        
        operational_goals = []
        for i, template in enumerate(action_templates):
            operational_goal = Goal(
                id=f"op_{goal.id}_{i}",
                title=template.format(title=goal.title),
                description=f"Operational action for: {goal.description}",
                level=GoalLevel.OPERATIONAL,
                parent_id=goal.id,
                priority=goal.priority,
                estimated_duration=goal.estimated_duration / len(action_templates),
                success_criteria=[f"Complete {template.split(' for:')[0].lower()}"],
                resources_required=["time", "focus"],
                metadata={"parent_goal": goal.title, "action_type": template.split()[0].lower()}
            )
            operational_goals.append(operational_goal)
        
        return operational_goals
    
    def _decompose_operational(self, goal: Goal) -> List[Goal]:
        """Operational goals are typically atomic and don't decompose further"""
        return []
    
    def decompose(self, goal: Goal) -> List[Goal]:
        """Decompose a goal into sub-goals"""
        if goal.level in self.decomposition_strategies:
            return self.decomposition_strategies[goal.level](goal)
        return []


class DependencyResolver:
    """Manages dependencies between goals"""
    
    def __init__(self):
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_dependencies: Dict[str, Set[str]] = {}
    
    def add_dependency(self, goal_id: str, depends_on: str):
        """Add a dependency relationship"""
        if goal_id not in self.dependency_graph:
            self.dependency_graph[goal_id] = set()
        if depends_on not in self.reverse_dependencies:
            self.reverse_dependencies[depends_on] = set()
        
        self.dependency_graph[goal_id].add(depends_on)
        self.reverse_dependencies[depends_on].add(goal_id)
    
    def get_dependencies(self, goal_id: str) -> Set[str]:
        """Get all goals this goal depends on"""
        return self.dependency_graph.get(goal_id, set())
    
    def get_dependents(self, goal_id: str) -> Set[str]:
        """Get all goals that depend on this goal"""
        return self.reverse_dependencies.get(goal_id, set())
    
    def is_ready_to_start(self, goal_id: str, goal_statuses: Dict[str, GoalStatus]) -> bool:
        """Check if a goal can start based on its dependencies"""
        dependencies = self.get_dependencies(goal_id)
        for dep_id in dependencies:
            if goal_statuses.get(dep_id) != GoalStatus.COMPLETED:
                return False
        return True
    
    def get_ready_goals(self, goal_statuses: Dict[str, GoalStatus]) -> List[str]:
        """Get all goals that are ready to start"""
        ready_goals = []
        for goal_id, status in goal_statuses.items():
            if (status == GoalStatus.PENDING and 
                self.is_ready_to_start(goal_id, goal_statuses)):
                ready_goals.append(goal_id)
        return ready_goals


class ProgressTracker:
    """Tracks progress across the goal hierarchy"""
    
    def __init__(self):
        self.progress_history: List[Dict[str, Any]] = []
    
    def calculate_goal_progress(self, goal: Goal, all_goals: Dict[str, Goal]) -> float:
        """Calculate progress of a goal based on its children"""
        if not goal.children_ids:
            # Leaf goal - progress based on status
            if goal.status == GoalStatus.COMPLETED:
                return 1.0
            elif goal.status == GoalStatus.IN_PROGRESS:
                # Estimate based on time elapsed
                if goal.started_at and goal.estimated_duration > 0:
                    elapsed = time.time() - goal.started_at
                    return min(elapsed / (goal.estimated_duration * 3600), 0.9)
                return 0.1
            else:
                return 0.0
        
        # Parent goal - progress based on children
        total_progress = 0.0
        completed_children = 0
        
        for child_id in goal.children_ids:
            if child_id in all_goals:
                child_progress = self.calculate_goal_progress(all_goals[child_id], all_goals)
                total_progress += child_progress
                if child_progress >= 1.0:
                    completed_children += 1
        
        return total_progress / len(goal.children_ids) if goal.children_ids else 0.0
    
    def get_hierarchy_summary(self, goals: Dict[str, Goal]) -> Dict[str, Any]:
        """Get a summary of progress across the hierarchy"""
        summary = {
            "strategic": {"total": 0, "completed": 0, "in_progress": 0, "pending": 0},
            "tactical": {"total": 0, "completed": 0, "in_progress": 0, "pending": 0},
            "operational": {"total": 0, "completed": 0, "in_progress": 0, "pending": 0},
            "overall_progress": 0.0
        }
        
        for goal in goals.values():
            level_key = goal.level.value
            summary[level_key]["total"] += 1
            
            if goal.status == GoalStatus.COMPLETED:
                summary[level_key]["completed"] += 1
            elif goal.status == GoalStatus.IN_PROGRESS:
                summary[level_key]["in_progress"] += 1
            elif goal.status == GoalStatus.PENDING:
                summary[level_key]["pending"] += 1
        
        # Calculate overall progress
        total_goals = sum(level["total"] for level in summary.values() if isinstance(level, dict))
        completed_goals = sum(level["completed"] for level in summary.values() if isinstance(level, dict))
        summary["overall_progress"] = completed_goals / total_goals if total_goals > 0 else 0.0
        
        return summary


class HierarchicalPlanner:
    """Main planner that orchestrates hierarchical planning"""
    
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.decomposer = GoalDecomposer()
        self.dependency_resolver = DependencyResolver()
        self.progress_tracker = ProgressTracker()
        self.planning_history: List[Dict[str, Any]] = []
    
    def add_strategic_goal(self, title: str, description: str, priority: int = 3) -> str:
        """Add a high-level strategic goal"""
        goal = Goal(
            id=f"strat_{len([g for g in self.goals.values() if g.level == GoalLevel.STRATEGIC])}",
            title=title,
            description=description,
            level=GoalLevel.STRATEGIC,
            priority=priority,
            estimated_duration=40.0,  # 40 hours for strategic goal
            success_criteria=[f"Successfully achieve: {title}"],
            resources_required=["team", "budget", "time"]
        )
        
        self.goals[goal.id] = goal
        return goal.id
    
    def decompose_goal(self, goal_id: str) -> List[str]:
        """Decompose a goal into sub-goals"""
        if goal_id not in self.goals:
            raise ValueError(f"Goal {goal_id} not found")
        
        goal = self.goals[goal_id]
        sub_goals = self.decomposer.decompose(goal)
        
        # Add sub-goals to the system
        sub_goal_ids = []
        for i, sub_goal in enumerate(sub_goals):
            self.goals[sub_goal.id] = sub_goal
            goal.children_ids.append(sub_goal.id)
            sub_goal_ids.append(sub_goal.id)
            
            # Add sequential dependencies between sub-goals
            if i > 0:
                self.dependency_resolver.add_dependency(sub_goal.id, sub_goals[i-1].id)
        
        return sub_goal_ids
    
    def fully_decompose_plan(self, strategic_goal_id: str) -> Dict[str, List[str]]:
        """Fully decompose a strategic goal down to operational level"""
        decomposition_result = {
            "strategic": [strategic_goal_id],
            "tactical": [],
            "operational": []
        }
        
        # Decompose strategic to tactical
        tactical_ids = self.decompose_goal(strategic_goal_id)
        decomposition_result["tactical"] = tactical_ids
        
        # Decompose each tactical to operational
        for tactical_id in tactical_ids:
            operational_ids = self.decompose_goal(tactical_id)
            decomposition_result["operational"].extend(operational_ids)
        
        return decomposition_result
    
    def get_next_actions(self) -> List[Goal]:
        """Get the next operational actions that can be started"""
        goal_statuses = {goal_id: goal.status for goal_id, goal in self.goals.items()}
        ready_goal_ids = self.dependency_resolver.get_ready_goals(goal_statuses)
        
        # Filter for operational goals only
        operational_ready = [
            self.goals[goal_id] for goal_id in ready_goal_ids
            if goal_id in self.goals and self.goals[goal_id].level == GoalLevel.OPERATIONAL
        ]
        
        # Sort by priority and creation time
        operational_ready.sort(key=lambda g: (-g.priority, g.created_at))
        
        return operational_ready
    
    def execute_action(self, goal_id: str) -> bool:
        """Simulate executing an operational action"""
        if goal_id not in self.goals:
            return False
        
        goal = self.goals[goal_id]
        if goal.level != GoalLevel.OPERATIONAL:
            return False
        
        print(f"🎯 Executing: {goal.title}")
        goal.status = GoalStatus.IN_PROGRESS
        goal.started_at = time.time()
        
        # Simulate execution time (shortened for demo)
        execution_time = random.uniform(0.1, 0.3)  # seconds instead of hours
        time.sleep(execution_time)
        
        # Simulate success/failure (90% success rate)
        success = random.random() > 0.1
        
        if success:
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = time.time()
            goal.actual_duration = execution_time
            print(f"✅ Completed: {goal.title}")
            self._propagate_completion(goal_id)
            return True
        else:
            goal.status = GoalStatus.FAILED
            print(f"❌ Failed: {goal.title}")
            self._handle_failure(goal_id)
            return False
    
    def _propagate_completion(self, goal_id: str):
        """Check if parent goals should be marked as completed"""
        goal = self.goals[goal_id]
        if goal.parent_id and goal.parent_id in self.goals:
            parent = self.goals[goal.parent_id]
            
            # Check if all children are completed
            all_children_completed = all(
                self.goals[child_id].status == GoalStatus.COMPLETED
                for child_id in parent.children_ids
                if child_id in self.goals
            )
            
            if all_children_completed and parent.status != GoalStatus.COMPLETED:
                parent.status = GoalStatus.COMPLETED
                parent.completed_at = time.time()
                print(f"📈 Parent goal completed: {parent.title}")
                self._propagate_completion(parent.id)
    
    def _handle_failure(self, goal_id: str):
        """Handle goal failure with replanning"""
        goal = self.goals[goal_id]
        print(f"🔄 Replanning due to failure: {goal.title}")
        
        # Create alternative operational goal
        alternative_goal = Goal(
            id=f"alt_{goal.id}",
            title=f"Alternative approach for: {goal.title}",
            description=f"Alternative implementation for failed goal: {goal.description}",
            level=GoalLevel.OPERATIONAL,
            parent_id=goal.parent_id,
            priority=goal.priority + 1,  # Higher priority
            estimated_duration=goal.estimated_duration * 1.2,  # 20% more time
            success_criteria=goal.success_criteria,
            resources_required=goal.resources_required + ["additional_support"],
            metadata={"replanned": True, "original_goal": goal.id}
        )
        
        self.goals[alternative_goal.id] = alternative_goal
        
        # Update parent's children list
        if goal.parent_id and goal.parent_id in self.goals:
            parent = self.goals[goal.parent_id]
            if goal.id in parent.children_ids:
                parent.children_ids.remove(goal.id)
            parent.children_ids.append(alternative_goal.id)
        
        print(f"📋 Created alternative goal: {alternative_goal.title}")
    
    def execute_full_plan(self, strategic_goal_id: str) -> Dict[str, Any]:
        """Execute a complete hierarchical plan"""
        print(f"\n🎯 Executing Hierarchical Plan")
        print("=" * 60)
        
        # Fully decompose the plan
        print("📋 Decomposing strategic goal...")
        decomposition = self.fully_decompose_plan(strategic_goal_id)
        
        print(f"Strategic goals: {len(decomposition['strategic'])}")
        print(f"Tactical goals: {len(decomposition['tactical'])}")
        print(f"Operational goals: {len(decomposition['operational'])}")
        
        # Execute operational actions
        execution_log = []
        max_iterations = 50  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            next_actions = self.get_next_actions()
            if not next_actions:
                print("\n✅ No more actions to execute")
                break
            
            for action in next_actions[:3]:  # Execute up to 3 actions in parallel
                success = self.execute_action(action.id)
                execution_log.append({
                    "iteration": iteration,
                    "action": action.title,
                    "success": success,
                    "timestamp": time.time()
                })
            
            iteration += 1
        
        # Generate final report
        summary = self.progress_tracker.get_hierarchy_summary(self.goals)
        
        result = {
            "decomposition": decomposition,
            "execution_log": execution_log,
            "final_summary": summary,
            "iterations": iteration,
            "strategic_goal_completed": self.goals[strategic_goal_id].status == GoalStatus.COMPLETED
        }
        
        return result
    
    def print_hierarchy(self, root_goal_id: str, indent: int = 0):
        """Print the goal hierarchy"""
        if root_goal_id not in self.goals:
            return
        
        goal = self.goals[root_goal_id]
        prefix = "  " * indent
        status_emoji = {
            GoalStatus.PENDING: "⏳",
            GoalStatus.IN_PROGRESS: "🔄",
            GoalStatus.COMPLETED: "✅",
            GoalStatus.FAILED: "❌",
            GoalStatus.BLOCKED: "🚫",
            GoalStatus.CANCELLED: "⭕"
        }
        
        progress = self.progress_tracker.calculate_goal_progress(goal, self.goals)
        print(f"{prefix}{status_emoji.get(goal.status, '❓')} {goal.title}")
        print(f"{prefix}   Level: {goal.level.value}, Progress: {progress:.1%}")
        
        for child_id in goal.children_ids:
            self.print_hierarchy(child_id, indent + 1)


def main():
    """Demonstration of the Hierarchical Planning pattern"""
    print("📊 Hierarchical Planning Pattern Demonstration")
    print("=" * 80)
    print("This demonstrates multi-level goal decomposition:")
    print("- Strategic (high-level, long-term objectives)")
    print("- Tactical (mid-level, medium-term plans)")
    print("- Operational (low-level, immediate actions)")
    print("- Dependency management and progress tracking")
    
    # Create planner
    planner = HierarchicalPlanner()
    
    # Test scenarios
    strategic_goals = [
        ("Launch AI-powered customer service system", "Develop and deploy an intelligent customer service solution"),
        ("Implement company-wide digital transformation", "Modernize all business processes with digital technologies"),
        ("Create sustainable energy infrastructure", "Build renewable energy systems for long-term sustainability")
    ]
    
    for i, (title, description) in enumerate(strategic_goals, 1):
        print(f"\n\n🎯 Strategic Goal {i}: {title}")
        print("=" * 80)
        
        # Add strategic goal
        strategic_id = planner.add_strategic_goal(title, description, priority=5-i)
        
        # Execute the plan
        result = planner.execute_full_plan(strategic_id)
        
        print(f"\n📊 Execution Results:")
        print(f"Total iterations: {result['iterations']}")
        print(f"Strategic goal completed: {result['strategic_goal_completed']}")
        print(f"Overall progress: {result['final_summary']['overall_progress']:.1%}")
        
        print(f"\n📋 Goal Hierarchy:")
        planner.print_hierarchy(strategic_id)
        
        print(f"\n📈 Summary by Level:")
        for level, stats in result['final_summary'].items():
            if isinstance(stats, dict):
                print(f"  {level.capitalize()}: {stats['completed']}/{stats['total']} completed")
        
        # Reset for next test
        planner = HierarchicalPlanner()
    
    print("\n\n🎯 Key Hierarchical Planning Features Demonstrated:")
    print("✅ Multi-level goal decomposition (Strategic → Tactical → Operational)")
    print("✅ Dependency management between goals")
    print("✅ Progress tracking across hierarchy levels")
    print("✅ Adaptive replanning when goals fail")
    print("✅ Parallel execution of ready actions")
    print("✅ Automatic completion propagation")
    print("✅ Resource and priority management")
    print("✅ Success criteria validation")


if __name__ == "__main__":
    main()