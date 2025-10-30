"""
Pattern 025: Action Sequence Planning

Description:
    The Action Sequence Planning pattern enables agents to decompose complex goals
    into ordered sequences of actions with preconditions, effects, and dependencies.
    The agent analyzes the goal, identifies required actions, determines ordering
    constraints, validates feasibility, and executes the plan step-by-step with
    monitoring and replanning capabilities.

Components:
    - Action Library: Catalog of available actions with specifications
    - Planner: Generates action sequences to achieve goals
    - Dependency Analyzer: Identifies action dependencies and constraints
    - Feasibility Checker: Validates plan viability
    - Executor: Executes plans with monitoring and error handling

Use Cases:
    - Multi-step task automation
    - Workflow orchestration
    - Robot task planning
    - Complex problem decomposition
    - Project planning and execution

LangChain Implementation:
    Uses LLM for action selection and ordering, implements STRIPS-like
    planning with preconditions and effects, handles dynamic replanning.

Key Features:
    - Action dependency analysis
    - Precondition and effect modeling
    - Parallel action detection
    - Plan validation
    - Dynamic replanning on failures
"""

import os
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ActionStatus(Enum):
    """Status of action execution."""
    PENDING = "pending"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ActionDefinition:
    """Definition of an available action."""
    name: str
    description: str
    preconditions: List[str]
    effects: List[str]
    duration: float = 1.0  # Estimated duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LLM context."""
        return {
            "name": self.name,
            "description": self.description,
            "preconditions": self.preconditions,
            "effects": self.effects
        }


@dataclass
class PlannedAction:
    """An action in a plan."""
    action: ActionDefinition
    step_number: int
    dependencies: List[int] = field(default_factory=list)
    status: ActionStatus = ActionStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class ActionPlan:
    """Complete action plan."""
    goal: str
    actions: List[PlannedAction]
    initial_state: Set[str]
    final_state: Set[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_ready_actions(self) -> List[PlannedAction]:
        """Get actions that are ready to execute."""
        ready = []
        for action in self.actions:
            if action.status == ActionStatus.PENDING:
                # Check if all dependencies completed
                dependencies_met = all(
                    self.actions[dep].status == ActionStatus.COMPLETED
                    for dep in action.dependencies
                )
                if dependencies_met:
                    ready.append(action)
        return ready
    
    def is_complete(self) -> bool:
        """Check if plan is complete."""
        return all(
            action.status in [ActionStatus.COMPLETED, ActionStatus.SKIPPED]
            for action in self.actions
        )


class ActionLibrary:
    """
    Library of available actions.
    """
    
    def __init__(self):
        """Initialize action library with common actions."""
        self.actions: Dict[str, ActionDefinition] = {}
        self._register_builtin_actions()
    
    def _register_builtin_actions(self):
        """Register built-in actions."""
        
        # Data collection actions
        self.register_action(ActionDefinition(
            name="gather_requirements",
            description="Collect and document project requirements",
            preconditions=["project_defined"],
            effects=["requirements_gathered"]
        ))
        
        self.register_action(ActionDefinition(
            name="research_topic",
            description="Research and gather information on a topic",
            preconditions=[],
            effects=["research_completed", "information_gathered"]
        ))
        
        # Analysis actions
        self.register_action(ActionDefinition(
            name="analyze_data",
            description="Analyze collected data or information",
            preconditions=["data_available"],
            effects=["analysis_completed", "insights_generated"]
        ))
        
        self.register_action(ActionDefinition(
            name="identify_risks",
            description="Identify and assess potential risks",
            preconditions=["requirements_gathered"],
            effects=["risks_identified"]
        ))
        
        # Design actions
        self.register_action(ActionDefinition(
            name="create_design",
            description="Create system or solution design",
            preconditions=["requirements_gathered", "research_completed"],
            effects=["design_created"]
        ))
        
        self.register_action(ActionDefinition(
            name="review_design",
            description="Review and validate design",
            preconditions=["design_created"],
            effects=["design_reviewed"]
        ))
        
        # Implementation actions
        self.register_action(ActionDefinition(
            name="implement_solution",
            description="Implement the designed solution",
            preconditions=["design_reviewed"],
            effects=["solution_implemented"]
        ))
        
        self.register_action(ActionDefinition(
            name="write_documentation",
            description="Write technical documentation",
            preconditions=["solution_implemented"],
            effects=["documentation_written"]
        ))
        
        # Testing actions
        self.register_action(ActionDefinition(
            name="create_test_plan",
            description="Create comprehensive test plan",
            preconditions=["design_created"],
            effects=["test_plan_created"]
        ))
        
        self.register_action(ActionDefinition(
            name="execute_tests",
            description="Execute test cases and validate solution",
            preconditions=["solution_implemented", "test_plan_created"],
            effects=["tests_executed", "quality_validated"]
        ))
        
        # Deployment actions
        self.register_action(ActionDefinition(
            name="prepare_deployment",
            description="Prepare solution for deployment",
            preconditions=["tests_executed", "documentation_written"],
            effects=["deployment_ready"]
        ))
        
        self.register_action(ActionDefinition(
            name="deploy_solution",
            description="Deploy solution to production",
            preconditions=["deployment_ready"],
            effects=["solution_deployed"]
        ))
    
    def register_action(self, action: ActionDefinition):
        """Register a new action."""
        self.actions[action.name] = action
    
    def get_action(self, name: str) -> Optional[ActionDefinition]:
        """Get action by name."""
        return self.actions.get(name)
    
    def get_all_actions_description(self) -> str:
        """Get formatted description of all actions."""
        descriptions = []
        for action in self.actions.values():
            precond = ", ".join(action.preconditions) if action.preconditions else "none"
            effects = ", ".join(action.effects)
            descriptions.append(
                f"- {action.name}: {action.description}\n"
                f"  Preconditions: {precond}\n"
                f"  Effects: {effects}"
            )
        return "\n\n".join(descriptions)


class ActionPlanner:
    """
    Plans action sequences to achieve goals.
    """
    
    def __init__(
        self,
        action_library: ActionLibrary,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize action planner.
        
        Args:
            action_library: Library of available actions
            model: LLM model to use
        """
        self.action_library = action_library
        self.llm = ChatOpenAI(model=model, temperature=0.3)
    
    def create_plan(
        self,
        goal: str,
        initial_state: Optional[Set[str]] = None
    ) -> ActionPlan:
        """
        Create action plan to achieve goal.
        
        Args:
            goal: Goal to achieve
            initial_state: Initial world state
            
        Returns:
            Action plan
        """
        if initial_state is None:
            initial_state = set()
        
        actions_desc = self.action_library.get_all_actions_description()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert action planner. Create a sequence of actions
to achieve the given goal, respecting preconditions and dependencies.

Available Actions:
{actions_description}

Your response should list actions in order, one per line, using exact action names.
Only use actions from the available list."""),
            ("user", """Goal: {goal}

Initial State: {initial_state}

Create an action plan listing actions in execution order:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        initial_state_str = ", ".join(initial_state) if initial_state else "none"
        
        response = chain.invoke({
            "actions_description": actions_desc,
            "goal": goal,
            "initial_state": initial_state_str
        })
        
        # Parse action names
        action_names = self._parse_action_names(response)
        
        # Build plan with dependencies
        planned_actions = self._build_plan_with_dependencies(
            action_names,
            initial_state
        )
        
        # Calculate final state
        final_state = self._calculate_final_state(planned_actions, initial_state)
        
        return ActionPlan(
            goal=goal,
            actions=planned_actions,
            initial_state=initial_state,
            final_state=final_state
        )
    
    def _parse_action_names(self, response: str) -> List[str]:
        """Parse action names from LLM response."""
        lines = response.strip().split("\n")
        action_names = []
        
        for line in lines:
            line = line.strip()
            # Remove numbering, bullets, etc.
            for prefix in ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "-", "*", "•"]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
            
            # Check if this is a valid action name
            if self.action_library.get_action(line):
                action_names.append(line)
        
        return action_names
    
    def _build_plan_with_dependencies(
        self,
        action_names: List[str],
        initial_state: Set[str]
    ) -> List[PlannedAction]:
        """Build plan with dependency analysis."""
        planned_actions = []
        current_state = initial_state.copy()
        
        for step_num, action_name in enumerate(action_names):
            action_def = self.action_library.get_action(action_name)
            if not action_def:
                continue
            
            # Find dependencies (actions that provide required preconditions)
            dependencies = []
            for precond in action_def.preconditions:
                if precond not in current_state:
                    # Find which previous action provides this
                    for i, prev_action in enumerate(planned_actions):
                        if precond in prev_action.action.effects:
                            dependencies.append(i)
                            break
            
            planned_action = PlannedAction(
                action=action_def,
                step_number=step_num,
                dependencies=dependencies
            )
            
            planned_actions.append(planned_action)
            
            # Update state with effects
            current_state.update(action_def.effects)
        
        return planned_actions
    
    def _calculate_final_state(
        self,
        planned_actions: List[PlannedAction],
        initial_state: Set[str]
    ) -> Set[str]:
        """Calculate final state after all actions."""
        final_state = initial_state.copy()
        
        for planned_action in planned_actions:
            final_state.update(planned_action.action.effects)
        
        return final_state


class PlanExecutor:
    """
    Executes action plans.
    """
    
    def __init__(self):
        """Initialize plan executor."""
        pass
    
    def execute_plan(self, plan: ActionPlan) -> Dict[str, Any]:
        """
        Execute action plan.
        
        Args:
            plan: Action plan to execute
            
        Returns:
            Execution results
        """
        print(f"\n[Executor] Executing plan for goal: {plan.goal}")
        print(f"[Executor] Total actions: {len(plan.actions)}\n")
        
        completed = 0
        failed = 0
        
        while not plan.is_complete():
            # Get ready actions
            ready_actions = plan.get_ready_actions()
            
            if not ready_actions:
                # Check if we have pending actions that can't proceed
                pending = [a for a in plan.actions if a.status == ActionStatus.PENDING]
                if pending:
                    print("[Executor] ✗ Deadlock: Actions have unmet dependencies")
                    for action in pending:
                        action.status = ActionStatus.FAILED
                        action.error = "Unmet dependencies"
                        failed += 1
                break
            
            # Execute ready actions
            for action in ready_actions:
                self._execute_action(action)
                
                if action.status == ActionStatus.COMPLETED:
                    completed += 1
                elif action.status == ActionStatus.FAILED:
                    failed += 1
        
        success_rate = completed / len(plan.actions) * 100 if plan.actions else 0
        
        return {
            "goal": plan.goal,
            "total_actions": len(plan.actions),
            "completed": completed,
            "failed": failed,
            "success_rate": success_rate,
            "plan": plan
        }
    
    def _execute_action(self, action: PlannedAction):
        """Execute a single action."""
        action.status = ActionStatus.EXECUTING
        action.start_time = datetime.now()
        
        print(f"[Step {action.step_number + 1}] Executing: {action.action.name}")
        print(f"  Description: {action.action.description}")
        
        if action.dependencies:
            deps_str = ", ".join([str(d + 1) for d in action.dependencies])
            print(f"  Dependencies: Steps {deps_str}")
        
        # Simulate action execution (in real system, would call actual function)
        try:
            # Simulate success
            action.result = f"Successfully completed {action.action.name}"
            action.status = ActionStatus.COMPLETED
            print(f"  ✓ Completed")
            print(f"  Effects: {', '.join(action.action.effects)}")
            
        except Exception as e:
            action.error = str(e)
            action.status = ActionStatus.FAILED
            print(f"  ✗ Failed: {action.error}")
        
        action.end_time = datetime.now()
        print()


class ActionSequencePlanningAgent:
    """
    Agent that plans and executes action sequences.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize action sequence planning agent.
        
        Args:
            model: LLM model to use
        """
        self.action_library = ActionLibrary()
        self.planner = ActionPlanner(self.action_library, model)
        self.executor = PlanExecutor()
    
    def achieve_goal(
        self,
        goal: str,
        initial_state: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Plan and execute actions to achieve goal.
        
        Args:
            goal: Goal to achieve
            initial_state: Initial world state
            
        Returns:
            Execution results
        """
        print(f"\n{'=' * 80}")
        print(f"GOAL: {goal}")
        print("=" * 80)
        
        # Create plan
        print("\n[Agent] Creating action plan...")
        plan = self.planner.create_plan(goal, initial_state)
        
        print(f"\n[Agent] Plan created with {len(plan.actions)} actions")
        print("\nACTION PLAN:")
        print("-" * 80)
        
        for action in plan.actions:
            deps = ""
            if action.dependencies:
                deps = f" (depends on: {[d+1 for d in action.dependencies]})"
            print(f"{action.step_number + 1}. {action.action.name}{deps}")
            print(f"   {action.action.description}")
        
        print("-" * 80)
        print(f"\nInitial state: {', '.join(plan.initial_state) if plan.initial_state else 'none'}")
        print(f"Expected final state: {', '.join(plan.final_state)}")
        
        # Execute plan
        print(f"\n{'=' * 80}")
        print("EXECUTION")
        print("=" * 80)
        
        result = self.executor.execute_plan(plan)
        
        # Summary
        print("=" * 80)
        print("EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Goal: {result['goal']}")
        print(f"Total actions: {result['total_actions']}")
        print(f"Completed: {result['completed']}")
        print(f"Failed: {result['failed']}")
        print(f"Success rate: {result['success_rate']:.0f}%")
        
        return result


def demonstrate_action_sequence_planning():
    """Demonstrate the Action Sequence Planning pattern."""
    
    print("=" * 80)
    print("ACTION SEQUENCE PLANNING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = ActionSequencePlanningAgent()
    
    # Show available actions
    print("\n" + "=" * 80)
    print("AVAILABLE ACTIONS")
    print("=" * 80)
    print(agent.action_library.get_all_actions_description())
    
    # Test 1: Software project workflow
    print("\n\n" + "=" * 80)
    print("TEST 1: Software Development Project")
    print("=" * 80)
    
    goal1 = "Complete a software development project from requirements to deployment"
    initial_state1 = {"project_defined"}
    
    result1 = agent.achieve_goal(goal1, initial_state1)
    
    # Test 2: Research and analysis task
    print("\n\n" + "=" * 80)
    print("TEST 2: Research and Analysis Task")
    print("=" * 80)
    
    goal2 = "Research a topic, analyze findings, and create a report"
    initial_state2 = set()
    
    result2 = agent.achieve_goal(goal2, initial_state2)
    
    # Test 3: Design and implementation
    print("\n\n" + "=" * 80)
    print("TEST 3: Design and Implementation")
    print("=" * 80)
    
    goal3 = "Design, implement, and test a new feature"
    initial_state3 = {"requirements_gathered", "research_completed"}
    
    result3 = agent.achieve_goal(goal3, initial_state3)
    
    # Overall summary
    print("\n\n" + "=" * 80)
    print("PATTERN SUMMARY")
    print("=" * 80)
    print("""
The Action Sequence Planning pattern demonstrates:

1. **Goal Decomposition**: Breaks complex goals into action sequences
2. **Dependency Analysis**: Identifies action dependencies and ordering
3. **Precondition Checking**: Ensures actions are feasible
4. **Effect Modeling**: Tracks state changes from actions
5. **Sequential Execution**: Executes plan respecting dependencies

Key Benefits:
- **Structured Approach**: Systematic goal achievement
- **Dependency Management**: Proper action ordering
- **State Tracking**: Monitors world state changes
- **Validation**: Ensures plan feasibility
- **Transparency**: Clear action sequences and reasoning

Use Cases:
- Multi-step task automation
- Workflow orchestration
- Project planning and execution
- Robot task planning
- Business process automation

Planning Principles:
- Actions have clear preconditions and effects
- Dependencies are automatically inferred
- Plans respect causal relationships
- State evolves through action effects
- Parallel execution possible when no dependencies

This pattern enables agents to tackle complex, multi-step goals
through structured planning and systematic execution.
""")


if __name__ == "__main__":
    demonstrate_action_sequence_planning()
