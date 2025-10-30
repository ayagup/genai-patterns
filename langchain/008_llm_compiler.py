"""
Pattern 008: LLM Compiler

Description:
    Optimizes LLM execution by analyzing task dependencies, parallelizing
    independent operations, and optimizing the execution plan like a compiler
    optimizes code. Combines planning with execution optimization.

Key Concepts:
    - Task Decomposition: Break complex tasks into subtasks
    - Dependency Analysis: Build dependency graph (DAG)
    - Optimization: Identify parallelizable operations
    - Scheduling: Optimal execution order
    - Resource Management: Efficient LLM call allocation

Compiler Phases:
    1. Parsing: Understand and decompose the task
    2. Analysis: Build dependency graph
    3. Optimization: Find parallelization opportunities
    4. Code Generation: Create execution plan
    5. Execution: Run optimized plan

Use Cases:
    - Complex multi-step workflows
    - Research tasks with multiple branches
    - Data processing pipelines
    - Tasks with many independent subtasks

LangChain Implementation:
    Uses dependency graphs and parallel execution with LangChain tools.
"""

import os
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


class TaskStatus(Enum):
    """Status of a task in the execution plan."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Represents a single task in the execution plan."""
    id: str
    description: str
    dependencies: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    parallel_group: Optional[int] = None  # For grouping parallel tasks
    
    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if this task can be executed."""
        return (self.status == TaskStatus.PENDING and 
                self.dependencies.issubset(completed_tasks))


@dataclass
class ExecutionPlan:
    """Represents the optimized execution plan."""
    goal: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    dependency_graph: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    parallel_groups: List[List[str]] = field(default_factory=list)
    
    def add_task(self, task: Task):
        """Add a task to the plan."""
        self.tasks[task.id] = task
        for dep in task.dependencies:
            self.dependency_graph[dep].add(task.id)
    
    def get_ready_tasks(self, completed: Set[str]) -> List[Task]:
        """Get all tasks that are ready to execute."""
        ready = []
        for task in self.tasks.values():
            if task.can_execute(completed):
                ready.append(task)
        return ready
    
    def topological_sort(self) -> List[str]:
        """Perform topological sort on tasks."""
        in_degree = {task_id: len(task.dependencies) 
                     for task_id, task in self.tasks.items()}
        
        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])
        sorted_tasks = []
        
        while queue:
            task_id = queue.popleft()
            sorted_tasks.append(task_id)
            
            for dependent in self.dependency_graph[task_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        return sorted_tasks


class LLMCompiler:
    """Compiler that optimizes LLM task execution."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", max_parallel: int = 3):
        """
        Initialize the LLM Compiler.
        
        Args:
            model_name: Name of the OpenAI model
            max_parallel: Maximum number of parallel tasks
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.max_parallel = max_parallel
    
    def parse_task(self, goal: str) -> ExecutionPlan:
        """
        Parse the goal and decompose into subtasks.
        
        Args:
            goal: The high-level goal
            
        Returns:
            Initial execution plan with tasks
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Decompose the goal into specific, actionable subtasks.
For each subtask:
1. Give it a unique ID (T1, T2, T3, etc.)
2. Describe what needs to be done
3. List dependencies (which tasks must complete first)

Format:
T1: [description] | Dependencies: none
T2: [description] | Dependencies: T1
T3: [description] | Dependencies: T1
T4: [description] | Dependencies: T2, T3
...

Be specific and ensure dependencies are correct."""),
            ("human", "Goal: {goal}\n\nTask Decomposition:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"goal": goal})
        
        # Parse tasks
        plan = ExecutionPlan(goal=goal)
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line or ':' not in line:
                continue
            
            # Parse: T1: description | Dependencies: T2, T3
            parts = line.split('|')
            if len(parts) < 2:
                continue
            
            task_part = parts[0].strip()
            dep_part = parts[1].strip()
            
            # Extract task ID and description
            if ':' in task_part:
                task_id, description = task_part.split(':', 1)
                task_id = task_id.strip()
                description = description.strip()
            else:
                continue
            
            # Extract dependencies
            dependencies = set()
            if 'dependencies:' in dep_part.lower():
                dep_text = dep_part.split(':', 1)[1].strip()
                if dep_text.lower() != 'none':
                    # Split by comma and extract task IDs
                    for dep in dep_text.split(','):
                        dep = dep.strip()
                        if dep and dep[0] == 'T':
                            dependencies.add(dep)
            
            # Create task
            task = Task(
                id=task_id,
                description=description,
                dependencies=dependencies
            )
            plan.add_task(task)
        
        return plan
    
    def analyze_dependencies(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        Analyze dependencies and optimize execution order.
        
        Args:
            plan: Initial execution plan
            
        Returns:
            Plan with optimized execution order
        """
        # Topological sort to get valid execution order
        sorted_tasks = plan.topological_sort()
        
        # Identify parallel groups (tasks at same level)
        completed = set()
        parallel_groups = []
        
        while completed != set(plan.tasks.keys()):
            # Get all tasks that can execute now
            ready_tasks = []
            for task_id in sorted_tasks:
                if task_id not in completed:
                    task = plan.tasks[task_id]
                    if task.can_execute(completed):
                        ready_tasks.append(task_id)
            
            if not ready_tasks:
                break  # Circular dependency or error
            
            parallel_groups.append(ready_tasks)
            completed.update(ready_tasks)
            
            # Assign parallel group numbers
            for task_id in ready_tasks:
                plan.tasks[task_id].parallel_group = len(parallel_groups) - 1
        
        plan.parallel_groups = parallel_groups
        
        return plan
    
    def optimize_plan(self, plan: ExecutionPlan) -> ExecutionPlan:
        """
        Apply optimizations to the execution plan.
        
        Args:
            plan: Analyzed execution plan
            
        Returns:
            Optimized plan
        """
        # Optimization strategies:
        # 1. Limit parallel groups to max_parallel
        # 2. Try to balance load across groups
        # 3. Merge small groups if beneficial
        
        optimized_groups = []
        for group in plan.parallel_groups:
            if len(group) <= self.max_parallel:
                optimized_groups.append(group)
            else:
                # Split large groups
                for i in range(0, len(group), self.max_parallel):
                    optimized_groups.append(group[i:i + self.max_parallel])
        
        plan.parallel_groups = optimized_groups
        
        return plan
    
    def execute_task(self, task: Task, plan: ExecutionPlan) -> str:
        """
        Execute a single task.
        
        Args:
            task: The task to execute
            plan: The complete plan for context
            
        Returns:
            Task result
        """
        # Build context from completed dependencies
        context_parts = []
        for dep_id in task.dependencies:
            dep_task = plan.tasks.get(dep_id)
            if dep_task and dep_task.result:
                context_parts.append(f"{dep_id}: {dep_task.result}")
        
        context = "\n".join(context_parts) if context_parts else "No prior context."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Execute the task using the provided context from previous steps."),
            ("human", """Goal: {goal}

Prior Results:
{context}

Task: {task_description}

Result:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "goal": plan.goal,
            "context": context,
            "task_description": task.description
        })
        
        return result.strip()
    
    def execute_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Execute the optimized plan.
        
        Args:
            plan: The execution plan
            
        Returns:
            Execution results
        """
        print(f"\nGoal: {plan.goal}\n")
        print(f"Total Tasks: {len(plan.tasks)}")
        print(f"Parallel Groups: {len(plan.parallel_groups)}\n")
        
        # Show execution plan
        print("Execution Plan:")
        for i, group in enumerate(plan.parallel_groups):
            group_tasks = [plan.tasks[tid].description[:50] for tid in group]
            print(f"  Group {i+1} ({len(group)} tasks in parallel):")
            for task_id in group:
                deps = list(plan.tasks[task_id].dependencies)
                dep_str = f" [depends on: {', '.join(deps)}]" if deps else ""
                print(f"    - {task_id}: {plan.tasks[task_id].description[:60]}{dep_str}")
        print()
        
        completed = set()
        total_llm_calls = 0
        
        # Execute groups in sequence
        for group_idx, group in enumerate(plan.parallel_groups):
            print(f"{'='*60}")
            print(f"Executing Group {group_idx + 1} ({len(group)} tasks)")
            print('='*60)
            
            # Execute tasks in group in parallel
            with ThreadPoolExecutor(max_workers=min(len(group), self.max_parallel)) as executor:
                futures = {}
                for task_id in group:
                    task = plan.tasks[task_id]
                    task.status = TaskStatus.RUNNING
                    future = executor.submit(self.execute_task, task, plan)
                    futures[future] = task
                
                # Collect results
                for future in as_completed(futures):
                    task = futures[future]
                    total_llm_calls += 1
                    
                    try:
                        result = future.result()
                        task.result = result
                        task.status = TaskStatus.COMPLETED
                        completed.add(task.id)
                        print(f"\n✓ {task.id} completed")
                        print(f"  Result: {result[:80]}...")
                    except Exception as e:
                        task.status = TaskStatus.FAILED
                        task.error = str(e)
                        print(f"\n✗ {task.id} failed: {str(e)}")
        
        # Generate final summary
        summary = self._synthesize_results(plan)
        
        return {
            "goal": plan.goal,
            "total_tasks": len(plan.tasks),
            "completed_tasks": len(completed),
            "parallel_groups": len(plan.parallel_groups),
            "total_llm_calls": total_llm_calls,
            "summary": summary
        }
    
    def _synthesize_results(self, plan: ExecutionPlan) -> str:
        """Synthesize final summary from all task results."""
        results = []
        for task in plan.tasks.values():
            if task.status == TaskStatus.COMPLETED and task.result:
                results.append(f"{task.id}: {task.result}")
        
        if not results:
            return "No tasks completed successfully."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Synthesize the results into a comprehensive summary."),
            ("human", """Goal: {goal}

Task Results:
{results}

Summary:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            summary = chain.invoke({
                "goal": plan.goal,
                "results": "\n\n".join(results)
            })
            return summary.strip()
        except:
            return "Summary generation failed."
    
    def compile_and_execute(self, goal: str) -> Dict[str, Any]:
        """
        Compile and execute a goal (full pipeline).
        
        Args:
            goal: The goal to achieve
            
        Returns:
            Execution results
        """
        print(f"\n{'='*80}")
        print("LLM COMPILER: PARSING")
        print('='*80)
        
        # Phase 1: Parse and decompose
        plan = self.parse_task(goal)
        print(f"\nParsed {len(plan.tasks)} tasks")
        
        print(f"\n{'='*80}")
        print("LLM COMPILER: ANALYSIS")
        print('='*80)
        
        # Phase 2: Analyze dependencies
        plan = self.analyze_dependencies(plan)
        print(f"\nIdentified {len(plan.parallel_groups)} execution groups")
        
        print(f"\n{'='*80}")
        print("LLM COMPILER: OPTIMIZATION")
        print('='*80)
        
        # Phase 3: Optimize
        plan = self.optimize_plan(plan)
        print(f"\nOptimized for {self.max_parallel} max parallel tasks")
        
        print(f"\n{'='*80}")
        print("LLM COMPILER: EXECUTION")
        print('='*80)
        
        # Phase 4: Execute
        result = self.execute_plan(plan)
        
        return result


def demonstrate_llm_compiler():
    """Demonstrates the LLM Compiler pattern."""
    
    print("=" * 80)
    print("PATTERN 008: LLM Compiler")
    print("=" * 80)
    print()
    print("LLM Compiler optimizes execution like a code compiler:")
    print("1. Parsing: Decompose task into subtasks")
    print("2. Analysis: Build dependency graph")
    print("3. Optimization: Identify parallelization opportunities")
    print("4. Execution: Run optimized plan with parallel execution")
    print()
    
    # Create compiler
    compiler = LLMCompiler(max_parallel=3)
    
    # Test goals
    goals = [
        "Research the history of artificial intelligence and create a timeline of major milestones",
        "Analyze the advantages and disadvantages of renewable energy sources and compare them"
    ]
    
    for i, goal in enumerate(goals, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}")
        print('='*80)
        
        try:
            result = compiler.compile_and_execute(goal)
            
            print(f"\n\n{'='*80}")
            print("COMPILATION SUMMARY")
            print('='*80)
            print(f"\nGoal: {result['goal']}")
            print(f"Total Tasks: {result['total_tasks']}")
            print(f"Completed: {result['completed_tasks']}/{result['total_tasks']}")
            print(f"Parallel Groups: {result['parallel_groups']}")
            print(f"Total LLM Calls: {result['total_llm_calls']}")
            print(f"\nFinal Summary:\n{result['summary']}")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("LLM COMPILER PATTERN DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("1. Task Decomposition: Breaking complex goals into subtasks")
    print("2. Dependency Analysis: Building execution dependency graph")
    print("3. Parallelization: Executing independent tasks concurrently")
    print("4. Optimization: Grouping and scheduling for efficiency")
    print("5. Resource Management: Limiting parallel execution")
    print()
    print("Advantages:")
    print("- Reduced total execution time through parallelization")
    print("- Efficient resource utilization")
    print("- Clear dependency management")
    print("- Systematic optimization approach")
    print()
    print("When to use LLM Compiler:")
    print("- Complex multi-step workflows")
    print("- Tasks with many independent subtasks")
    print("- Time-sensitive applications")
    print("- Research and analysis tasks")
    print()
    print("LangChain Components Used:")
    print("- ChatPromptTemplate for task decomposition")
    print("- Dependency graph (DAG) management")
    print("- ThreadPoolExecutor for parallel execution")
    print("- Result synthesis chain")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    demonstrate_llm_compiler()
