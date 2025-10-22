"""
Plan-and-Execute Pattern
========================
Separates planning from execution phases.
Components: Planner, Executor, Re-planner
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """Represents a single task in the plan"""
    id: int
    description: str
    dependencies: List[int]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    
    def __repr__(self):
        return f"Task({self.id}: {self.description[:30]}... - {self.status.value})"


class Planner:
    """Creates execution plans"""
    
    def create_plan(self, goal: str) -> List[Task]:
        """
        Creates a plan to achieve the goal
        Returns a list of tasks with dependencies
        """
        print(f"\n{'='*60}")
        print(f"🎯 Planning for goal: {goal}")
        print(f"{'='*60}\n")
        
        tasks = []
        
        if "research paper" in goal.lower():
            tasks = [
                Task(1, "Define research topic and scope", []),
                Task(2, "Conduct literature review", [1]),
                Task(3, "Identify research gap", [2]),
                Task(4, "Design methodology", [3]),
                Task(5, "Collect and analyze data", [4]),
                Task(6, "Write initial draft", [5]),
                Task(7, "Review and revise", [6]),
                Task(8, "Format and finalize", [7])
            ]
        
        elif "web application" in goal.lower():
            tasks = [
                Task(1, "Define requirements and features", []),
                Task(2, "Design database schema", [1]),
                Task(3, "Create API endpoints", [2]),
                Task(4, "Implement frontend UI", [1]),
                Task(5, "Integrate frontend with API", [3, 4]),
                Task(6, "Write tests", [5]),
                Task(7, "Deploy to production", [6])
            ]
        
        elif "dinner party" in goal.lower():
            tasks = [
                Task(1, "Create guest list", []),
                Task(2, "Send invitations", [1]),
                Task(3, "Plan menu", [1]),
                Task(4, "Shop for ingredients", [3]),
                Task(5, "Prepare venue", [1]),
                Task(6, "Cook food", [4]),
                Task(7, "Set table", [5]),
                Task(8, "Welcome guests", [2, 7])
            ]
        
        else:
            # Generic task breakdown
            tasks = [
                Task(1, "Analyze the goal and requirements", []),
                Task(2, "Break down into subtasks", [1]),
                Task(3, "Execute each subtask", [2]),
                Task(4, "Review and validate results", [3])
            ]
        
        print("📋 Plan created with the following tasks:\n")
        for task in tasks:
            deps = f" (depends on: {task.dependencies})" if task.dependencies else ""
            print(f"  {task.id}. {task.description}{deps}")
        
        return tasks


class Executor:
    """Executes tasks according to the plan"""
    
    def can_execute(self, task: Task, completed_tasks: List[int]) -> bool:
        """Check if all dependencies are met"""
        return all(dep_id in completed_tasks for dep_id in task.dependencies)
    
    def execute_task(self, task: Task) -> bool:
        """
        Execute a single task
        Returns True if successful, False otherwise
        """
        print(f"\n⚙️  Executing Task {task.id}: {task.description}")
        task.status = TaskStatus.IN_PROGRESS
        
        # Simulate task execution (in real scenario, this would do actual work)
        try:
            # Simulate some tasks that might fail
            if "analyze data" in task.description.lower() and task.id == 5:
                # Simulate a failure
                raise Exception("Insufficient data quality")
            
            # Simulate success
            task.result = f"Completed: {task.description}"
            task.status = TaskStatus.COMPLETED
            print(f"   ✅ Success: {task.description}")
            return True
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            print(f"   ❌ Failed: {str(e)}")
            return False
    
    def execute_plan(self, tasks: List[Task]) -> Dict:
        """
        Execute all tasks in the plan, respecting dependencies
        """
        print(f"\n{'='*60}")
        print("🚀 Executing Plan")
        print(f"{'='*60}")
        
        completed_tasks = []
        failed_tasks = []
        blocked_tasks = []
        
        max_iterations = len(tasks) * 2  # Prevent infinite loops
        iteration = 0
        
        while len(completed_tasks) + len(failed_tasks) + len(blocked_tasks) < len(tasks):
            if iteration >= max_iterations:
                print("\n⚠️  Maximum iterations reached")
                break
            
            iteration += 1
            made_progress = False
            
            for task in tasks:
                if task.status == TaskStatus.COMPLETED:
                    if task.id not in completed_tasks:
                        completed_tasks.append(task.id)
                    continue
                
                if task.status == TaskStatus.FAILED:
                    if task.id not in failed_tasks:
                        failed_tasks.append(task.id)
                    continue
                
                if task.status == TaskStatus.BLOCKED:
                    continue
                
                # Check if we can execute this task
                if self.can_execute(task, completed_tasks):
                    success = self.execute_task(task)
                    made_progress = True
                    
                    if not success:
                        # Mark dependent tasks as blocked
                        for other_task in tasks:
                            if task.id in other_task.dependencies:
                                other_task.status = TaskStatus.BLOCKED
                                blocked_tasks.append(other_task.id)
                                print(f"   🚫 Task {other_task.id} blocked due to dependency failure")
            
            if not made_progress:
                break
        
        return {
            "completed": completed_tasks,
            "failed": failed_tasks,
            "blocked": blocked_tasks,
            "total": len(tasks)
        }


class Replanner:
    """Adjusts plans when execution fails"""
    
    def replan(self, original_tasks: List[Task], execution_results: Dict) -> List[Task]:
        """
        Create a new plan based on execution results
        """
        print(f"\n{'='*60}")
        print("🔄 Replanning after failures")
        print(f"{'='*60}\n")
        
        if not execution_results["failed"]:
            print("No replanning needed - all tasks succeeded")
            return original_tasks
        
        new_tasks = []
        task_id_counter = max(t.id for t in original_tasks) + 1
        
        for task in original_tasks:
            if task.status == TaskStatus.FAILED:
                print(f"❌ Task {task.id} failed: {task.error}")
                print(f"   Creating alternative approach...")
                
                # Create alternative tasks
                alt_task = Task(
                    id=task_id_counter,
                    description=f"Alternative approach: {task.description}",
                    dependencies=task.dependencies,
                    status=TaskStatus.PENDING
                )
                new_tasks.append(alt_task)
                task_id_counter += 1
                
                print(f"   ✨ Created Task {alt_task.id}: {alt_task.description}")
                
            elif task.status == TaskStatus.BLOCKED:
                # Reset blocked tasks to pending
                task.status = TaskStatus.PENDING
                new_tasks.append(task)
                print(f"🔓 Task {task.id} unblocked and reset to pending")
                
            else:
                new_tasks.append(task)
        
        return new_tasks


class PlanAndExecuteAgent:
    """Main agent that coordinates planning and execution"""
    
    def __init__(self):
        self.planner = Planner()
        self.executor = Executor()
        self.replanner = Replanner()
    
    def achieve_goal(self, goal: str, max_replans: int = 2):
        """
        Main method to achieve a goal using plan-and-execute pattern
        """
        print(f"\n{'='*70}")
        print(f"🎯 GOAL: {goal}")
        print(f"{'='*70}")
        
        # Planning phase
        tasks = self.planner.create_plan(goal)
        
        replan_count = 0
        while replan_count <= max_replans:
            # Execution phase
            results = self.executor.execute_plan(tasks)
            
            # Report results
            print(f"\n{'='*60}")
            print("📊 Execution Results")
            print(f"{'='*60}")
            print(f"✅ Completed: {len(results['completed'])}/{results['total']}")
            print(f"❌ Failed: {len(results['failed'])}/{results['total']}")
            print(f"🚫 Blocked: {len(results['blocked'])}/{results['total']}")
            
            # Check if we need to replan
            if results['failed'] and replan_count < max_replans:
                replan_count += 1
                print(f"\n🔄 Replanning (attempt {replan_count}/{max_replans})")
                tasks = self.replanner.replan(tasks, results)
            else:
                break
        
        # Final summary
        print(f"\n{'='*60}")
        print("🏁 Final Summary")
        print(f"{'='*60}")
        success_rate = len(results['completed']) / results['total'] * 100
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Replanning attempts: {replan_count}")
        
        if success_rate == 100:
            print("🎉 Goal achieved successfully!")
        elif success_rate >= 70:
            print("⚠️  Goal partially achieved")
        else:
            print("❌ Goal not achieved")


def main():
    """Demonstrate Plan-and-Execute pattern"""
    
    agent = PlanAndExecuteAgent()
    
    # Example 1: Research paper
    print("\n" + "="*70)
    print("EXAMPLE 1: Write a Research Paper")
    print("="*70)
    agent.achieve_goal("Write a research paper on AI agents")
    
    # Example 2: Web application
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Build a Web Application")
    print("="*70)
    agent = PlanAndExecuteAgent()
    agent.achieve_goal("Build a web application for task management")
    
    # Example 3: Event planning
    print("\n\n" + "="*70)
    print("EXAMPLE 3: Organize a Dinner Party")
    print("="*70)
    agent = PlanAndExecuteAgent()
    agent.achieve_goal("Organize a dinner party for 10 guests")


if __name__ == "__main__":
    main()
