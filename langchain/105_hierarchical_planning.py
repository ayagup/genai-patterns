"""
Pattern 105: Hierarchical Planning

Description:
    The Hierarchical Planning pattern breaks down complex goals into a hierarchy
    of sub-goals and actions at multiple levels of abstraction. This pattern
    enables agents to handle long-horizon tasks by decomposing them into manageable
    pieces, planning at different levels of granularity, and progressively refining
    high-level strategies into concrete actions.
    
    Hierarchical planning is inspired by hierarchical task networks (HTN) and
    goal decomposition techniques. It allows agents to reason about problems at
    multiple scales: strategic (what to achieve), tactical (how to achieve it),
    and operational (specific actions). Each level focuses on appropriate
    abstractions, making planning more tractable and interpretable.
    
    This pattern is particularly effective for complex, multi-step tasks where
    flat planning would result in intractably large search spaces. By organizing
    goals hierarchically, the agent can plan efficiently, adapt at different
    levels, and explain its reasoning at appropriate levels of detail.

Key Components:
    1. Goal Hierarchy: Tree structure of goals and sub-goals
    2. Abstraction Levels: Strategic, tactical, operational layers
    3. Decomposition Rules: How to break goals into sub-goals
    4. Level Planner: Plans at each abstraction level
    5. Refinement Engine: Converts abstract plans to concrete actions
    6. Execution Monitor: Tracks progress at all levels
    7. Re-planning Trigger: Identifies when to replan at each level

Abstraction Levels:
    1. Strategic Level: High-level goals and objectives
       - What needs to be achieved
       - Overall strategy and approach
       - Long-term planning horizon
    
    2. Tactical Level: Sub-goals and procedures
       - How to achieve strategic goals
       - Intermediate milestones
       - Medium-term planning
    
    3. Operational Level: Concrete actions
       - Specific actions to execute
       - Immediate next steps
       - Short-term execution

Planning Phases:
    1. Goal Decomposition: Break high-level goal into sub-goals
    2. Level Planning: Plan at each abstraction level
    3. Plan Refinement: Refine abstract plans into detailed plans
    4. Execution: Execute lowest-level actions
    5. Monitoring: Track progress at all levels
    6. Re-planning: Adapt when needed at appropriate level

Use Cases:
    - Complex project planning
    - Multi-step task execution
    - Robotic mission planning
    - Software development workflows
    - Travel itinerary planning
    - Research project organization
    - Business process automation

Advantages:
    - Manages complexity through abstraction
    - More efficient than flat planning
    - Allows reasoning at appropriate granularity
    - Easier to explain and understand
    - Facilitates partial replanning
    - Modular and reusable sub-plans
    - Scales to long-horizon tasks

Challenges:
    - Defining appropriate abstraction levels
    - Goal decomposition complexity
    - Ensuring consistency across levels
    - Handling dependencies between sub-goals
    - Determining when to replan at each level
    - Balancing planning depth vs. efficiency

LangChain Implementation:
    This implementation uses LangChain for:
    - LLM-based goal decomposition
    - Multi-level plan generation
    - Plan refinement and detailing
    - Natural language planning
    
Production Considerations:
    - Define clear abstraction levels for domain
    - Implement efficient goal decomposition
    - Track dependencies between sub-goals
    - Monitor execution at all levels
    - Enable replanning at appropriate level
    - Cache and reuse common sub-plans
    - Provide visualization of plan hierarchy
    - Handle failures at correct abstraction level
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class PlanLevel(Enum):
    """Planning abstraction levels."""
    STRATEGIC = "strategic"  # High-level goals
    TACTICAL = "tactical"    # Sub-goals and procedures
    OPERATIONAL = "operational"  # Concrete actions


class GoalStatus(Enum):
    """Status of a goal."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Goal:
    """
    Represents a goal at any level of the hierarchy.
    
    Attributes:
        goal_id: Unique identifier
        description: Goal description
        level: Abstraction level
        parent_id: Parent goal ID (None for root)
        children_ids: List of child goal IDs
        status: Current status
        priority: Priority (1-10)
        dependencies: IDs of goals that must complete first
        actions: Concrete actions (for operational level)
        metadata: Additional information
    """
    goal_id: str
    description: str
    level: PlanLevel
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    status: GoalStatus = GoalStatus.PENDING
    priority: int = 5
    dependencies: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class HierarchicalPlanner:
    """
    Hierarchical planning system that plans at multiple abstraction levels.
    
    This planner decomposes high-level goals into hierarchies of sub-goals,
    plans at each level, and progressively refines abstract plans into
    concrete actions.
    """
    
    def __init__(self, temperature: float = 0.3):
        """
        Initialize hierarchical planner.
        
        Args:
            temperature: LLM temperature for planning
        """
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.goals: Dict[str, Goal] = {}
        self.goal_counter = 0
        self.execution_log: List[Dict[str, Any]] = []
    
    def create_goal(
        self,
        description: str,
        level: PlanLevel,
        parent_id: Optional[str] = None,
        priority: int = 5,
        dependencies: Optional[List[str]] = None
    ) -> Goal:
        """
        Create a new goal.
        
        Args:
            description: Goal description
            level: Abstraction level
            parent_id: Parent goal ID
            priority: Priority (1-10)
            dependencies: Goal dependencies
            
        Returns:
            Created goal
        """
        self.goal_counter += 1
        goal = Goal(
            goal_id=f"goal_{self.goal_counter}",
            description=description,
            level=level,
            parent_id=parent_id,
            priority=priority,
            dependencies=dependencies or []
        )
        
        self.goals[goal.goal_id] = goal
        
        # Update parent's children
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].children_ids.append(goal.goal_id)
        
        return goal
    
    def decompose_goal(
        self,
        goal: Goal,
        target_level: PlanLevel
    ) -> List[Goal]:
        """
        Decompose a goal into sub-goals at lower abstraction level.
        
        Args:
            goal: Goal to decompose
            target_level: Target abstraction level for sub-goals
            
        Returns:
            List of sub-goals
        """
        # Use LLM to decompose goal
        prompt = ChatPromptTemplate.from_template(
            "You are a hierarchical planning assistant.\n\n"
            "Goal: {goal_description}\n"
            "Current Level: {current_level}\n"
            "Target Level: {target_level}\n\n"
            "Decompose this goal into 3-5 sub-goals at the {target_level} level.\n"
            "List each sub-goal on a new line, numbered.\n"
            "Sub-goals should be concrete, achievable, and collectively accomplish the parent goal.\n\n"
            "Sub-goals:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "goal_description": goal.description,
            "current_level": goal.level.value,
            "target_level": target_level.value
        })
        
        # Parse sub-goals from LLM output
        lines = [line.strip() for line in result.strip().split('\n') if line.strip()]
        sub_goals = []
        
        for line in lines:
            # Remove numbering (e.g., "1. ", "1) ", etc.)
            description = line
            for prefix in ["1.", "2.", "3.", "4.", "5.", "1)", "2)", "3)", "4)", "5)", "-", "*"]:
                if description.startswith(prefix):
                    description = description[len(prefix):].strip()
                    break
            
            if description:
                sub_goal = self.create_goal(
                    description=description,
                    level=target_level,
                    parent_id=goal.goal_id,
                    priority=goal.priority
                )
                sub_goals.append(sub_goal)
        
        return sub_goals
    
    def plan_strategic(self, objective: str) -> Goal:
        """
        Create strategic-level plan.
        
        Args:
            objective: High-level objective
            
        Returns:
            Strategic goal
        """
        # Create strategic goal
        strategic_goal = self.create_goal(
            description=objective,
            level=PlanLevel.STRATEGIC,
            priority=10
        )
        
        return strategic_goal
    
    def plan_tactical(self, strategic_goal: Goal) -> List[Goal]:
        """
        Create tactical-level plans from strategic goal.
        
        Args:
            strategic_goal: Strategic goal to decompose
            
        Returns:
            List of tactical goals
        """
        tactical_goals = self.decompose_goal(strategic_goal, PlanLevel.TACTICAL)
        return tactical_goals
    
    def plan_operational(self, tactical_goal: Goal) -> List[Goal]:
        """
        Create operational-level plans from tactical goal.
        
        Args:
            tactical_goal: Tactical goal to decompose
            
        Returns:
            List of operational goals
        """
        operational_goals = self.decompose_goal(tactical_goal, PlanLevel.OPERATIONAL)
        
        # Generate concrete actions for operational goals
        for op_goal in operational_goals:
            op_goal.actions = self.generate_actions(op_goal)
        
        return operational_goals
    
    def generate_actions(self, operational_goal: Goal) -> List[str]:
        """
        Generate concrete actions for an operational goal.
        
        Args:
            operational_goal: Operational goal
            
        Returns:
            List of concrete actions
        """
        prompt = ChatPromptTemplate.from_template(
            "Generate 2-4 specific, concrete actions to accomplish this goal:\n\n"
            "Goal: {goal}\n\n"
            "List each action on a new line, numbered.\n"
            "Actions should be specific and executable.\n\n"
            "Actions:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"goal": operational_goal.description})
        
        # Parse actions
        lines = [line.strip() for line in result.strip().split('\n') if line.strip()]
        actions = []
        
        for line in lines:
            # Remove numbering
            action = line
            for prefix in ["1.", "2.", "3.", "4.", "1)", "2)", "3)", "4)", "-", "*"]:
                if action.startswith(prefix):
                    action = action[len(prefix):].strip()
                    break
            if action:
                actions.append(action)
        
        return actions
    
    def create_full_hierarchy(self, objective: str) -> Goal:
        """
        Create complete hierarchical plan from objective.
        
        Args:
            objective: High-level objective
            
        Returns:
            Root strategic goal with full hierarchy
        """
        # Strategic level
        strategic_goal = self.plan_strategic(objective)
        
        # Tactical level
        tactical_goals = self.plan_tactical(strategic_goal)
        
        # Operational level
        for tactical_goal in tactical_goals:
            self.plan_operational(tactical_goal)
        
        return strategic_goal
    
    def get_next_action(self) -> Optional[Tuple[Goal, str]]:
        """
        Get next executable action from the plan.
        
        Returns:
            Tuple of (goal, action) or None if no actions available
        """
        # Find operational goals that are ready to execute
        operational_goals = [
            g for g in self.goals.values()
            if g.level == PlanLevel.OPERATIONAL and g.status == GoalStatus.PENDING
        ]
        
        # Sort by priority
        operational_goals.sort(key=lambda g: g.priority, reverse=True)
        
        for goal in operational_goals:
            # Check dependencies
            deps_satisfied = all(
                self.goals[dep_id].status == GoalStatus.COMPLETED
                for dep_id in goal.dependencies
                if dep_id in self.goals
            )
            
            if deps_satisfied and goal.actions:
                return (goal, goal.actions[0])
        
        return None
    
    def execute_action(self, goal: Goal, action: str) -> bool:
        """
        Execute an action (simulated).
        
        Args:
            goal: Goal being executed
            action: Action to execute
            
        Returns:
            Success status
        """
        # Mark goal as in progress
        if goal.status == GoalStatus.PENDING:
            goal.status = GoalStatus.IN_PROGRESS
        
        # Log execution
        self.execution_log.append({
            "timestamp": datetime.now(),
            "goal_id": goal.goal_id,
            "action": action,
            "level": goal.level.value
        })
        
        # Remove completed action
        if action in goal.actions:
            goal.actions.remove(action)
        
        # Mark goal complete if no more actions
        if not goal.actions:
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = datetime.now()
            self.propagate_completion(goal)
        
        return True
    
    def propagate_completion(self, goal: Goal):
        """
        Propagate goal completion up the hierarchy.
        
        Args:
            goal: Completed goal
        """
        if not goal.parent_id or goal.parent_id not in self.goals:
            return
        
        parent = self.goals[goal.parent_id]
        
        # Check if all children are complete
        all_children_complete = all(
            self.goals[child_id].status == GoalStatus.COMPLETED
            for child_id in parent.children_ids
            if child_id in self.goals
        )
        
        if all_children_complete:
            parent.status = GoalStatus.COMPLETED
            parent.completed_at = datetime.now()
            self.propagate_completion(parent)
    
    def get_hierarchy_summary(self, goal_id: Optional[str] = None) -> str:
        """
        Get textual summary of goal hierarchy.
        
        Args:
            goal_id: Root goal ID (None for all roots)
            
        Returns:
            Formatted hierarchy string
        """
        def format_goal(goal: Goal, indent: int = 0) -> str:
            status_symbol = {
                GoalStatus.PENDING: "⏳",
                GoalStatus.IN_PROGRESS: "🔄",
                GoalStatus.COMPLETED: "✅",
                GoalStatus.FAILED: "❌",
                GoalStatus.BLOCKED: "🚫"
            }
            
            symbol = status_symbol.get(goal.status, "❓")
            prefix = "  " * indent
            
            result = f"{prefix}{symbol} [{goal.level.value}] {goal.description}\n"
            
            # Add actions for operational level
            if goal.level == PlanLevel.OPERATIONAL and goal.actions:
                for action in goal.actions:
                    result += f"{prefix}  → {action}\n"
            
            # Recursively add children
            for child_id in goal.children_ids:
                if child_id in self.goals:
                    result += format_goal(self.goals[child_id], indent + 1)
            
            return result
        
        if goal_id:
            if goal_id in self.goals:
                return format_goal(self.goals[goal_id])
            return "Goal not found"
        
        # Format all root goals
        root_goals = [g for g in self.goals.values() if g.parent_id is None]
        return "\n".join(format_goal(g) for g in root_goals)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get planning statistics."""
        goals_by_level = {level: 0 for level in PlanLevel}
        goals_by_status = {status: 0 for status in GoalStatus}
        
        for goal in self.goals.values():
            goals_by_level[goal.level] += 1
            goals_by_status[goal.status] += 1
        
        return {
            "total_goals": len(self.goals),
            "by_level": {level.value: count for level, count in goals_by_level.items()},
            "by_status": {status.value: count for status, count in goals_by_status.items()},
            "total_actions_executed": len(self.execution_log)
        }


def demonstrate_hierarchical_planning():
    """Demonstrate hierarchical planning pattern."""
    
    print("=" * 80)
    print("HIERARCHICAL PLANNING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic three-level hierarchy
    print("\n" + "=" * 80)
    print("Example 1: Three-Level Planning Hierarchy")
    print("=" * 80)
    
    planner = HierarchicalPlanner()
    
    print("\nCreating hierarchical plan for: 'Build a web application'")
    
    # Strategic level
    strategic = planner.plan_strategic("Build a web application")
    print(f"\n[STRATEGIC] {strategic.description}")
    
    # Tactical level
    tactical = planner.plan_tactical(strategic)
    print(f"\n[TACTICAL] Decomposed into {len(tactical)} sub-goals:")
    for i, tac_goal in enumerate(tactical, 1):
        print(f"  {i}. {tac_goal.description}")
    
    # Operational level (for first tactical goal)
    if tactical:
        operational = planner.plan_operational(tactical[0])
        print(f"\n[OPERATIONAL] '{tactical[0].description}' decomposed into {len(operational)} goals:")
        for i, op_goal in enumerate(operational, 1):
            print(f"  {i}. {op_goal.description}")
            if op_goal.actions:
                for action in op_goal.actions:
                    print(f"     → {action}")
    
    # Example 2: Complete hierarchy creation
    print("\n" + "=" * 80)
    print("Example 2: Complete Hierarchical Plan")
    print("=" * 80)
    
    planner2 = HierarchicalPlanner()
    
    objective = "Organize a team offsite event"
    print(f"\nObjective: {objective}\n")
    
    root = planner2.create_full_hierarchy(objective)
    
    print("Complete Hierarchy:")
    print(planner2.get_hierarchy_summary())
    
    # Example 3: Plan execution
    print("\n" + "=" * 80)
    print("Example 3: Executing Hierarchical Plan")
    print("=" * 80)
    
    planner3 = HierarchicalPlanner()
    
    # Create simpler plan for execution demo
    strategic = planner3.create_goal(
        "Complete project report",
        PlanLevel.STRATEGIC
    )
    
    tactical1 = planner3.create_goal(
        "Gather data and research",
        PlanLevel.TACTICAL,
        parent_id=strategic.goal_id
    )
    
    tactical2 = planner3.create_goal(
        "Write and format report",
        PlanLevel.TACTICAL,
        parent_id=strategic.goal_id,
        dependencies=[tactical1.goal_id]
    )
    
    # Operational goals for tactical1
    op1 = planner3.create_goal(
        "Collect data from databases",
        PlanLevel.OPERATIONAL,
        parent_id=tactical1.goal_id
    )
    op1.actions = ["Connect to database", "Run queries", "Export results"]
    
    op2 = planner3.create_goal(
        "Review literature",
        PlanLevel.OPERATIONAL,
        parent_id=tactical1.goal_id
    )
    op2.actions = ["Search papers", "Read abstracts"]
    
    print("Initial Plan:")
    print(planner3.get_hierarchy_summary())
    
    print("\nExecuting actions...")
    
    # Execute actions
    for i in range(5):
        next_item = planner3.get_next_action()
        if next_item:
            goal, action = next_item
            print(f"  {i+1}. Executing: {action} (for '{goal.description}')")
            planner3.execute_action(goal, action)
        else:
            break
    
    print("\nUpdated Plan:")
    print(planner3.get_hierarchy_summary())
    
    # Example 4: Priority-based planning
    print("\n" + "=" * 80)
    print("Example 4: Priority-Based Goal Selection")
    print("=" * 80)
    
    planner4 = HierarchicalPlanner()
    
    # Create goals with different priorities
    strategic = planner4.create_goal(
        "Launch product",
        PlanLevel.STRATEGIC,
        priority=10
    )
    
    high_priority = planner4.create_goal(
        "Fix critical bug",
        PlanLevel.OPERATIONAL,
        parent_id=strategic.goal_id,
        priority=9
    )
    high_priority.actions = ["Identify bug", "Write fix", "Test fix"]
    
    low_priority = planner4.create_goal(
        "Update documentation",
        PlanLevel.OPERATIONAL,
        parent_id=strategic.goal_id,
        priority=3
    )
    low_priority.actions = ["Review docs", "Make updates"]
    
    print("Goals by priority:")
    for goal in sorted(planner4.goals.values(), key=lambda g: g.priority, reverse=True):
        if goal.level == PlanLevel.OPERATIONAL:
            print(f"  Priority {goal.priority}: {goal.description}")
    
    print("\nExecution order (priority-based):")
    for i in range(5):
        next_item = planner4.get_next_action()
        if next_item:
            goal, action = next_item
            print(f"  {i+1}. [P{goal.priority}] {action}")
            planner4.execute_action(goal, action)
        else:
            break
    
    # Example 5: Dependency handling
    print("\n" + "=" * 80)
    print("Example 5: Goal Dependencies")
    print("=" * 80)
    
    planner5 = HierarchicalPlanner()
    
    strategic = planner5.create_goal("Deploy application", PlanLevel.STRATEGIC)
    
    goal_a = planner5.create_goal(
        "Run tests",
        PlanLevel.OPERATIONAL,
        parent_id=strategic.goal_id
    )
    goal_a.actions = ["Execute test suite"]
    
    goal_b = planner5.create_goal(
        "Build for production",
        PlanLevel.OPERATIONAL,
        parent_id=strategic.goal_id,
        dependencies=[goal_a.goal_id]
    )
    goal_b.actions = ["Compile code"]
    
    goal_c = planner5.create_goal(
        "Deploy to server",
        PlanLevel.OPERATIONAL,
        parent_id=strategic.goal_id,
        dependencies=[goal_b.goal_id]
    )
    goal_c.actions = ["Upload to server"]
    
    print("Dependency chain: Tests → Build → Deploy\n")
    print("Initial state:")
    for goal in [goal_a, goal_b, goal_c]:
        deps = ", ".join(goal.dependencies) if goal.dependencies else "none"
        print(f"  {goal.description}: dependencies=[{deps}], status={goal.status.value}")
    
    print("\nExecution respects dependencies:")
    for i in range(3):
        next_item = planner5.get_next_action()
        if next_item:
            goal, action = next_item
            print(f"  {i+1}. {action} (from '{goal.description}')")
            planner5.execute_action(goal, action)
    
    # Example 6: Progress monitoring
    print("\n" + "=" * 80)
    print("Example 6: Multi-Level Progress Monitoring")
    print("=" * 80)
    
    planner6 = HierarchicalPlanner()
    
    root = planner6.create_full_hierarchy("Write research paper")
    
    print("Initial Statistics:")
    stats = planner6.get_statistics()
    print(f"  Total Goals: {stats['total_goals']}")
    print(f"  By Level: {stats['by_level']}")
    print(f"  By Status: {stats['by_status']}")
    
    # Execute some actions
    print("\nExecuting 10 actions...")
    for _ in range(10):
        next_item = planner6.get_next_action()
        if next_item:
            goal, action = next_item
            planner6.execute_action(goal, action)
    
    print("\nUpdated Statistics:")
    stats = planner6.get_statistics()
    print(f"  Total Goals: {stats['total_goals']}")
    print(f"  By Status: {stats['by_status']}")
    print(f"  Actions Executed: {stats['total_actions_executed']}")
    
    # Example 7: Plan visualization
    print("\n" + "=" * 80)
    print("Example 7: Hierarchical Plan Visualization")
    print("=" * 80)
    
    planner7 = HierarchicalPlanner()
    
    # Create multi-level plan
    strategic = planner7.create_goal("Learn Python", PlanLevel.STRATEGIC)
    
    tac1 = planner7.create_goal("Master basics", PlanLevel.TACTICAL, parent_id=strategic.goal_id)
    tac2 = planner7.create_goal("Build projects", PlanLevel.TACTICAL, parent_id=strategic.goal_id)
    
    op1 = planner7.create_goal("Learn syntax", PlanLevel.OPERATIONAL, parent_id=tac1.goal_id)
    op1.actions = ["Study variables", "Practice loops"]
    op1.status = GoalStatus.COMPLETED  # Mark as done
    
    op2 = planner7.create_goal("Learn data structures", PlanLevel.OPERATIONAL, parent_id=tac1.goal_id)
    op2.actions = ["Study lists", "Study dictionaries"]
    op2.status = GoalStatus.IN_PROGRESS
    
    op3 = planner7.create_goal("Build web app", PlanLevel.OPERATIONAL, parent_id=tac2.goal_id)
    op3.actions = ["Choose framework", "Create app"]
    
    print("Hierarchical Plan with Status:")
    print(planner7.get_hierarchy_summary())
    
    # Example 8: Replanning scenario
    print("\n" + "=" * 80)
    print("Example 8: Adaptive Replanning")
    print("=" * 80)
    
    planner8 = HierarchicalPlanner()
    
    print("Original Plan:")
    strategic = planner8.create_goal("Travel to conference", PlanLevel.STRATEGIC)
    
    tac1 = planner8.create_goal("Book transportation", PlanLevel.TACTICAL, parent_id=strategic.goal_id)
    op1 = planner8.create_goal("Book flight", PlanLevel.OPERATIONAL, parent_id=tac1.goal_id)
    op1.actions = ["Search flights", "Purchase ticket"]
    
    print(planner8.get_hierarchy_summary())
    
    print("\nSimulating failure (flight cancelled)...")
    op1.status = GoalStatus.FAILED
    
    # Create alternative plan
    print("\nReplanning at tactical level:")
    op2 = planner8.create_goal("Book train", PlanLevel.OPERATIONAL, parent_id=tac1.goal_id)
    op2.actions = ["Search trains", "Purchase ticket"]
    
    print(planner8.get_hierarchy_summary())
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Hierarchical Planning Pattern")
    print("=" * 80)
    
    summary = """
    The Hierarchical Planning pattern demonstrated:
    
    1. THREE-LEVEL HIERARCHY (Example 1):
       - Strategic: High-level objectives
       - Tactical: Sub-goals and procedures
       - Operational: Concrete actions
       - Progressive decomposition
       - Abstraction management
    
    2. COMPLETE HIERARCHY (Example 2):
       - Full plan generation
       - Multi-level decomposition
       - Parent-child relationships
       - Structured goal tree
       - Hierarchical visualization
    
    3. PLAN EXECUTION (Example 3):
       - Sequential action execution
       - Status tracking at all levels
       - Progress propagation
       - Completion detection
       - Hierarchical updates
    
    4. PRIORITY HANDLING (Example 4):
       - Priority-based selection
       - Urgent task handling
       - Importance ordering
       - Resource allocation
       - Efficient scheduling
    
    5. DEPENDENCY MANAGEMENT (Example 5):
       - Goal dependencies
       - Execution ordering
       - Prerequisite checking
       - Constraint satisfaction
       - Sequential execution
    
    6. PROGRESS MONITORING (Example 6):
       - Multi-level tracking
       - Status aggregation
       - Statistics collection
       - Performance metrics
       - Completion monitoring
    
    7. PLAN VISUALIZATION (Example 7):
       - Hierarchical display
       - Status indicators
       - Tree structure
       - Action details
       - Clear representation
    
    8. ADAPTIVE REPLANNING (Example 8):
       - Failure handling
       - Alternative generation
       - Level-appropriate replanning
       - Plan modification
       - Resilience
    
    KEY BENEFITS:
    ✓ Manages complexity through abstraction
    ✓ Efficient planning for long horizons
    ✓ Reasoning at appropriate granularity
    ✓ Easy to understand and explain
    ✓ Facilitates partial replanning
    ✓ Modular and reusable
    ✓ Scales to complex tasks
    ✓ Clear progress tracking
    
    USE CASES:
    • Complex project planning
    • Multi-step task execution
    • Robotic mission planning
    • Software development workflows
    • Travel itinerary planning
    • Research project organization
    • Business process automation
    • Strategic decision-making
    
    ABSTRACTION LEVELS:
    → Strategic: What to achieve (objectives)
    → Tactical: How to achieve (procedures)
    → Operational: Specific actions (execution)
    
    BEST PRACTICES:
    1. Define clear abstraction levels for domain
    2. Decompose goals systematically
    3. Track dependencies between goals
    4. Monitor execution at all levels
    5. Enable replanning at appropriate level
    6. Cache and reuse common sub-plans
    7. Visualize plan hierarchy
    8. Handle failures at correct level
    
    TRADE-OFFS:
    • Planning overhead vs. execution efficiency
    • Abstraction depth vs. simplicity
    • Flexibility vs. structure
    • Decomposition detail vs. overview
    
    PRODUCTION CONSIDERATIONS:
    → Implement domain-specific decomposition rules
    → Use caching for common sub-plans
    → Monitor performance at each level
    → Enable interactive plan adjustment
    → Provide clear visualization
    → Track dependencies explicitly
    → Implement efficient replanning
    → Handle partial plan execution
    → Support concurrent goal execution
    → Log decisions at all levels
    
    This pattern enables agents to tackle complex, long-horizon tasks
    by organizing planning at multiple levels of abstraction, from
    high-level strategies to concrete actions.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_hierarchical_planning()
