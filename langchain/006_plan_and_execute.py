"""
Pattern 006: Plan-and-Execute

Description:
    Separates planning and execution phases for complex multi-step tasks.
    First creates a high-level plan, then executes each step sequentially,
    with optional re-planning based on intermediate results.

Key Concepts:
    - Planning Phase: Create structured plan with steps
    - Execution Phase: Execute each step in sequence
    - Monitoring: Track progress and outcomes
    - Re-planning: Adjust plan based on execution results
    - State Management: Maintain context across steps

Use Cases:
    - Complex multi-step tasks requiring coordination
    - Research and information gathering workflows
    - Content creation with multiple stages
    - Problem-solving with dependencies between steps

LangChain Implementation:
    Uses LangGraph for state management and sequential execution with
    separate planner and executor chains.
"""

import os
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


class StepStatus(Enum):
    """Status of an execution step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """Represents a single step in the execution plan."""
    id: int
    description: str
    dependencies: List[int] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Represents the complete execution plan."""
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    current_step: int = 0
    completed_steps: int = 0
    
    def get_next_step(self) -> Optional[PlanStep]:
        """Get the next pending step."""
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                # Check if dependencies are met
                if all(
                    self.steps[dep_id - 1].status == StepStatus.COMPLETED 
                    for dep_id in step.dependencies
                ):
                    return step
        return None
    
    def is_complete(self) -> bool:
        """Check if all steps are completed or skipped."""
        return all(
            step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]
            for step in self.steps
        )
    
    def get_progress(self) -> float:
        """Get completion progress as percentage."""
        if not self.steps:
            return 0.0
        completed = sum(
            1 for step in self.steps 
            if step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]
        )
        return completed / len(self.steps)


class PlanAndExecuteAgent:
    """Agent that uses plan-and-execute pattern."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", 
                 enable_replanning: bool = True):
        """
        Initialize the Plan-and-Execute agent.
        
        Args:
            model_name: Name of the OpenAI model
            enable_replanning: Whether to enable dynamic re-planning
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.enable_replanning = enable_replanning
        self.execution_history: List[Dict[str, Any]] = []
    
    def create_plan(self, goal: str, context: str = "") -> ExecutionPlan:
        """
        Create an execution plan for the given goal.
        
        Args:
            goal: The goal to achieve
            context: Additional context information
            
        Returns:
            ExecutionPlan with structured steps
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Create a detailed step-by-step plan to achieve the goal.
Break down the task into clear, actionable steps.
Format your response as:
1. [First step]
2. [Second step]
3. [Third step]
etc.

Each step should be specific and achievable."""),
            ("human", """Goal: {goal}

Context: {context}

Plan:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "goal": goal,
            "context": context if context else "No additional context."
        })
        
        # Parse steps
        plan = ExecutionPlan(goal=goal)
        step_id = 1
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Extract step description
                description = line.split('.', 1)[-1].strip().lstrip('- ').strip()
                if description:
                    step = PlanStep(
                        id=step_id,
                        description=description
                    )
                    plan.steps.append(step)
                    step_id += 1
        
        return plan
    
    def execute_step(self, step: PlanStep, plan: ExecutionPlan) -> str:
        """
        Execute a single step of the plan.
        
        Args:
            step: The step to execute
            plan: The complete plan for context
            
        Returns:
            Result of executing the step
        """
        # Gather context from previous steps
        context = self._build_context(plan)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Execute the given step and provide a detailed result.
Be specific and thorough in your execution.
If the step requires information from previous steps, use the provided context."""),
            ("human", """Goal: {goal}

Previous steps context:
{context}

Current step to execute: {step_description}

Execution result:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({
                "goal": plan.goal,
                "context": context,
                "step_description": step.description
            })
            
            return result.strip()
            
        except Exception as e:
            raise Exception(f"Step execution failed: {str(e)}")
    
    def _build_context(self, plan: ExecutionPlan) -> str:
        """Build context string from completed steps."""
        context_parts = []
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED and step.result:
                context_parts.append(f"Step {step.id}: {step.description}\nResult: {step.result}")
        
        return "\n\n".join(context_parts) if context_parts else "No previous steps completed."
    
    def should_replan(self, plan: ExecutionPlan, last_step: PlanStep) -> bool:
        """
        Determine if re-planning is needed based on execution results.
        
        Args:
            plan: Current execution plan
            last_step: Most recently executed step
            
        Returns:
            True if re-planning is recommended
        """
        if not self.enable_replanning:
            return False
        
        # Simple heuristic: replan if step failed or result suggests issues
        if last_step.status == StepStatus.FAILED:
            return True
        
        # Use LLM to evaluate if re-planning is needed
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Evaluate if the plan should be adjusted based on the execution result.
Consider if:
- The result reveals new information that changes the approach
- There are unexpected obstacles
- The remaining steps need to be modified

Respond with only 'YES' or 'NO'."""),
            ("human", """Goal: {goal}

Completed step: {step_description}
Result: {result}

Should we replan? (YES/NO):""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "goal": plan.goal,
                "step_description": last_step.description,
                "result": last_step.result
            })
            
            return "yes" in response.lower()
            
        except:
            return False
    
    def replan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        Create a new plan based on current progress.
        
        Args:
            plan: Current execution plan
            
        Returns:
            Updated execution plan
        """
        context = self._build_context(plan)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Given the progress so far, create an updated plan for the remaining work.
Consider what has been accomplished and what still needs to be done.

Format as:
1. [First remaining step]
2. [Second remaining step]
etc."""),
            ("human", """Goal: {goal}

Progress so far:
{context}

Updated plan for remaining work:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "goal": plan.goal,
            "context": context
        })
        
        # Create new plan with completed steps + new steps
        new_plan = ExecutionPlan(goal=plan.goal)
        
        # Keep completed steps
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED:
                new_plan.steps.append(step)
        
        # Add new steps
        step_id = len(new_plan.steps) + 1
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                description = line.split('.', 1)[-1].strip().lstrip('- ').strip()
                if description:
                    new_step = PlanStep(
                        id=step_id,
                        description=description
                    )
                    new_plan.steps.append(new_step)
                    step_id += 1
        
        return new_plan
    
    def execute_plan(self, plan: ExecutionPlan, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute the complete plan.
        
        Args:
            plan: The execution plan
            max_steps: Maximum number of steps to execute (None for all)
            
        Returns:
            Dictionary with execution results
        """
        print(f"\nGoal: {plan.goal}")
        print(f"\nPlan ({len(plan.steps)} steps):")
        for step in plan.steps:
            print(f"  {step.id}. {step.description}")
        print()
        
        executed_steps = 0
        
        while not plan.is_complete():
            if max_steps and executed_steps >= max_steps:
                print(f"\nReached maximum steps limit ({max_steps})")
                break
            
            # Get next step
            next_step = plan.get_next_step()
            if not next_step:
                print("\nNo more executable steps (dependencies not met)")
                break
            
            # Execute step
            print(f"\n{'='*60}")
            print(f"Executing Step {next_step.id}: {next_step.description}")
            print('='*60)
            
            next_step.status = StepStatus.IN_PROGRESS
            
            try:
                result = self.execute_step(next_step, plan)
                next_step.result = result
                next_step.status = StepStatus.COMPLETED
                plan.completed_steps += 1
                
                print(f"\n✓ Step {next_step.id} completed")
                print(f"Result: {result[:200]}..." if len(result) > 200 else f"Result: {result}")
                
                # Check if re-planning is needed
                if self.should_replan(plan, next_step):
                    print(f"\n⟳ Re-planning needed based on results...")
                    plan = self.replan(plan)
                    print(f"Updated plan now has {len(plan.steps)} steps")
                
            except Exception as e:
                next_step.status = StepStatus.FAILED
                next_step.error = str(e)
                print(f"\n✗ Step {next_step.id} failed: {str(e)}")
            
            executed_steps += 1
        
        # Generate final summary
        summary = self._generate_summary(plan)
        
        return {
            "goal": plan.goal,
            "total_steps": len(plan.steps),
            "completed_steps": plan.completed_steps,
            "progress": plan.get_progress(),
            "status": "completed" if plan.is_complete() else "incomplete",
            "summary": summary
        }
    
    def _generate_summary(self, plan: ExecutionPlan) -> str:
        """Generate a summary of the execution."""
        completed_results = []
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED and step.result:
                completed_results.append(f"Step {step.id}: {step.result}")
        
        if not completed_results:
            return "No steps completed successfully."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide a concise summary of what was accomplished."),
            ("human", """Goal: {goal}

Results:
{results}

Summary:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            summary = chain.invoke({
                "goal": plan.goal,
                "results": "\n\n".join(completed_results)
            })
            return summary.strip()
        except:
            return "Summary generation failed."
    
    def solve_with_plan_execute(self, goal: str, context: str = "") -> Dict[str, Any]:
        """
        Solve a task using plan-and-execute pattern.
        
        Args:
            goal: The goal to achieve
            context: Additional context
            
        Returns:
            Execution results
        """
        print(f"\n{'='*80}")
        print("PLAN-AND-EXECUTE: Creating plan...")
        print('='*80)
        
        # Create initial plan
        plan = self.create_plan(goal, context)
        
        # Execute the plan
        result = self.execute_plan(plan)
        
        return result


def demonstrate_plan_execute():
    """Demonstrates the Plan-and-Execute pattern."""
    
    print("=" * 80)
    print("PATTERN 006: Plan-and-Execute")
    print("=" * 80)
    print()
    print("Plan-and-Execute separates planning from execution:")
    print("1. Planning Phase: Create structured multi-step plan")
    print("2. Execution Phase: Execute each step sequentially")
    print("3. Monitoring: Track progress and results")
    print("4. Re-planning: Adjust plan based on intermediate results")
    print()
    
    # Create agent
    agent = PlanAndExecuteAgent(enable_replanning=True)
    
    # Test goals
    goals = [
        {
            "goal": "Research and write a brief report on renewable energy trends",
            "context": "Focus on solar and wind energy developments in the last 5 years"
        },
        {
            "goal": "Plan a team building event for 20 people",
            "context": "Budget: $2000, Duration: Half day, Location: Urban area"
        }
    ]
    
    for i, test_case in enumerate(goals, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}")
        print('='*80)
        
        try:
            result = agent.solve_with_plan_execute(
                goal=test_case['goal'],
                context=test_case['context']
            )
            
            print(f"\n\n{'='*80}")
            print("EXECUTION SUMMARY")
            print('='*80)
            print(f"\nGoal: {result['goal']}")
            print(f"Status: {result['status'].upper()}")
            print(f"Progress: {result['progress']:.0%}")
            print(f"Steps Completed: {result['completed_steps']}/{result['total_steps']}")
            print(f"\nSummary:\n{result['summary']}")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("PLAN-AND-EXECUTE PATTERN DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("1. Structured Planning: Breaking goals into actionable steps")
    print("2. Sequential Execution: Executing steps with dependency tracking")
    print("3. Context Management: Passing results between steps")
    print("4. Progress Monitoring: Tracking completion status")
    print("5. Dynamic Re-planning: Adjusting plan based on results")
    print()
    print("Advantages:")
    print("- Clear separation of planning and execution")
    print("- Better handling of complex multi-step tasks")
    print("- Adaptability through re-planning")
    print("- Progress tracking and monitoring")
    print()
    print("LangChain Components Used:")
    print("- ChatPromptTemplate: Structures planning and execution prompts")
    print("- StrOutputParser: Parses LLM outputs")
    print("- State Management: Tracks plan and execution progress")
    print("- Sequential Chains: Executes steps in order")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    demonstrate_plan_execute()
