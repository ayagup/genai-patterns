"""
Plan-and-Execute Pattern
Separates planning from execution phases
"""
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
@dataclass
class Task:
    id: int
    description: str
    dependencies: List[int]
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
class PlanAndExecuteAgent:
    def __init__(self):
        self.tasks: List[Task] = []
        self.task_counter = 0
    def plan(self, goal: str) -> List[Task]:
        """Create a plan to achieve the goal"""
        print(f"\n=== Planning Phase ===")
        print(f"Goal: {goal}\n")
        # Example: Planning a data analysis workflow
        if "analyze data" in goal.lower():
            self.tasks = [
                Task(0, "Load data from source", []),
                Task(1, "Clean and preprocess data", [0]),
                Task(2, "Perform statistical analysis", [1]),
                Task(3, "Create visualizations", [1]),
                Task(4, "Generate report", [2, 3])
            ]
        else:
            # Generic planning
            self.tasks = [
                Task(0, f"Step 1: Understand {goal}", []),
                Task(1, f"Step 2: Gather resources", [0]),
                Task(2, f"Step 3: Execute main task", [1]),
                Task(3, f"Step 4: Verify results", [2])
            ]
        print("Generated Plan:")
        for task in self.tasks:
            deps = f" (depends on: {task.dependencies})" if task.dependencies else ""
            print(f"  Task {task.id}: {task.description}{deps}")
        return self.tasks
    def can_execute(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            dep_task = self.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    def execute_task(self, task: Task) -> bool:
        """Execute a single task"""
        print(f"\nExecuting Task {task.id}: {task.description}")
        task.status = TaskStatus.IN_PROGRESS
        # Simulate task execution
        try:
            # In real implementation, this would call actual functions/tools
            if "load data" in task.description.lower():
                task.result = {"rows": 1000, "columns": 10}
            elif "clean" in task.description.lower():
                task.result = {"cleaned_rows": 950}
            elif "analysis" in task.description.lower():
                task.result = {"mean": 42.5, "std": 12.3}
            elif "visualization" in task.description.lower():
                task.result = {"charts_created": 3}
            elif "report" in task.description.lower():
                task.result = {"report_path": "/tmp/report.pdf"}
            else:
                task.result = "Task completed"
            task.status = TaskStatus.COMPLETED
            print(f"  ✓ Completed: {task.result}")
            return True
        except Exception as e:
            task.status = TaskStatus.FAILED
            print(f"  ✗ Failed: {str(e)}")
            return False
    def replan(self, failed_task: Task) -> List[Task]:
        """Replan when a task fails"""
        print(f"\n=== Replanning ===")
        print(f"Task {failed_task.id} failed. Creating alternative approach...")
        # Insert a new task before the failed one
        new_task = Task(
            id=len(self.tasks),
            description=f"Alternative approach for: {failed_task.description}",
            dependencies=failed_task.dependencies,
            status=TaskStatus.PENDING
        )
        self.tasks.append(new_task)
        # Update failed task to depend on new task
        failed_task.dependencies.append(new_task.id)
        failed_task.status = TaskStatus.PENDING
        return self.tasks
    def execute(self) -> Dict[str, Any]:
        """Execute all tasks in dependency order"""
        print(f"\n=== Execution Phase ===")
        max_iterations = 20
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            # Find executable tasks
            executable = [t for t in self.tasks if 
                         t.status == TaskStatus.PENDING and 
                         self.can_execute(t)]
            if not executable:
                # Check if all done
                if all(t.status == TaskStatus.COMPLETED for t in self.tasks):
                    print("\n✓ All tasks completed successfully!")
                    break
                # Check for failures
                failed = [t for t in self.tasks if t.status == TaskStatus.FAILED]
                if failed:
                    print(f"\n✗ {len(failed)} task(s) failed")
                    break
                # No tasks ready - might be stuck
                print("\n⚠ No tasks ready to execute")
                break
            # Execute ready tasks
            for task in executable:
                success = self.execute_task(task)
                if not success:
                    self.replan(task)
        # Gather results
        results = {
            "total_tasks": len(self.tasks),
            "completed": sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self.tasks if t.status == TaskStatus.FAILED),
            "task_results": {t.id: t.result for t in self.tasks if t.result}
        }
        return results
# Usage
if __name__ == "__main__":
    agent = PlanAndExecuteAgent()
    goal = "Analyze data and create a comprehensive report"
    # Planning phase
    plan = agent.plan(goal)
    # Execution phase
    results = agent.execute()
    # Summary
    print(f"\n=== Execution Summary ===")
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Completed: {results['completed']}")
    print(f"Failed: {results['failed']}")
