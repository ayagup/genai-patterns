"""
Pattern 014: Leader-Follower

Description:
    The Leader-Follower pattern implements hierarchical agent coordination where one
    leader/coordinator agent manages and delegates tasks to multiple follower/specialist
    agents. The leader is responsible for task decomposition, delegation, coordination,
    and result synthesis.

Components:
    - Leader Agent: Coordinates, delegates, and synthesizes results
    - Follower Agents: Specialized agents that execute specific tasks
    - Communication Protocol: Request-response between leader and followers
    - Task Assignment Strategy: How tasks are allocated to followers

Use Cases:
    - Complex workflows requiring coordination
    - Tasks benefiting from specialization
    - Multi-step processes with dependencies
    - Projects requiring project management

LangChain Implementation:
    Uses a coordinator LLM for task planning and delegation, specialized LLMs
    with different system prompts for specific domains, and implements task
    assignment, execution tracking, and result synthesis.

Key Features:
    - Dynamic task decomposition
    - Specialist agent assignment
    - Progress tracking and monitoring
    - Hierarchical result synthesis
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class TaskStatus(Enum):
    """Status of a task in the workflow."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SpecialistType(Enum):
    """Types of specialist agents available."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    WRITING = "writing"
    CODING = "coding"
    MATH = "math"
    CREATIVE = "creative"


@dataclass
class Task:
    """Represents a task to be completed."""
    task_id: str
    description: str
    specialist_type: SpecialistType
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class WorkflowResult:
    """Result from complete workflow execution."""
    final_output: str
    tasks: List[Task]
    execution_summary: str


class SpecialistAgent:
    """
    A specialist agent with expertise in a specific domain.
    """
    
    def __init__(
        self,
        specialist_type: SpecialistType,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        """
        Initialize a specialist agent.
        
        Args:
            specialist_type: Type of specialist
            model: LLM model to use
            temperature: Creativity level
        """
        self.specialist_type = specialist_type
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        
        # Define specialist system prompts
        self.system_prompts = {
            SpecialistType.RESEARCH: """You are a research specialist. Your expertise is in:
- Finding and summarizing information
- Conducting literature reviews
- Fact-checking and verification
- Identifying credible sources
Provide thorough, well-researched responses with citations when possible.""",
            
            SpecialistType.ANALYSIS: """You are an analysis specialist. Your expertise is in:
- Data analysis and interpretation
- Pattern recognition
- Critical thinking and evaluation
- Drawing insights from information
Provide detailed analytical insights and evidence-based conclusions.""",
            
            SpecialistType.WRITING: """You are a writing specialist. Your expertise is in:
- Clear and engaging writing
- Content structuring and organization
- Grammar and style
- Adapting tone for different audiences
Provide well-written, polished content.""",
            
            SpecialistType.CODING: """You are a coding specialist. Your expertise is in:
- Writing clean, efficient code
- Debugging and problem-solving
- Best practices and design patterns
- Multiple programming languages
Provide production-ready code with explanations.""",
            
            SpecialistType.MATH: """You are a mathematics specialist. Your expertise is in:
- Mathematical problem-solving
- Statistical analysis
- Logical reasoning
- Step-by-step calculations
Provide accurate solutions with clear explanations.""",
            
            SpecialistType.CREATIVE: """You are a creative specialist. Your expertise is in:
- Brainstorming and ideation
- Creative problem-solving
- Innovative thinking
- Generating unique perspectives
Provide creative, out-of-the-box ideas and solutions."""
        }
    
    def execute_task(self, task: Task, context: Optional[str] = None) -> str:
        """
        Execute a task using specialist expertise.
        
        Args:
            task: Task to execute
            context: Optional context from previous tasks
            
        Returns:
            Task result as string
        """
        system_prompt = self.system_prompts[self.specialist_type]
        
        user_message = f"Task: {task.description}"
        if context:
            user_message += f"\n\nContext from previous tasks:\n{context}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_message)
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({})
        
        return result


class LeaderAgent:
    """
    Leader agent that coordinates and manages specialist agents.
    
    Responsibilities:
    - Decompose complex tasks into subtasks
    - Assign tasks to appropriate specialists
    - Monitor progress and handle dependencies
    - Synthesize results into final output
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize the leader agent."""
        self.llm = ChatOpenAI(model=model, temperature=0.3)
        
        # Initialize specialist agents
        self.specialists: Dict[SpecialistType, SpecialistAgent] = {
            specialist_type: SpecialistAgent(specialist_type)
            for specialist_type in SpecialistType
        }
    
    def decompose_task(self, overall_task: str) -> List[Task]:
        """
        Decompose an overall task into subtasks for specialists.
        
        Args:
            overall_task: The main task to decompose
            
        Returns:
            List of subtasks with assigned specialists
        """
        specialist_descriptions = "\n".join([
            f"- {s.value}: {self.specialists[s].system_prompts[s].split('Your expertise is in:')[1].split('Provide')[0].strip()}"
            for s in SpecialistType
        ])
        
        decomposition_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a project coordinator. Decompose tasks into subtasks and assign
to appropriate specialists. Be specific and ensure tasks are actionable."""),
            ("user", """Decompose this task into 3-5 subtasks:

Task: {overall_task}

Available specialists:
{specialist_descriptions}

Provide your decomposition in this exact format:
TASK 1: [specialist_type] - [clear task description]
TASK 2: [specialist_type] - [clear task description]
...

Use these specialist types: {specialist_types}""")
        ])
        
        chain = decomposition_prompt | self.llm | StrOutputParser()
        
        decomposition = chain.invoke({
            "overall_task": overall_task,
            "specialist_descriptions": specialist_descriptions,
            "specialist_types": ", ".join([s.value for s in SpecialistType])
        })
        
        # Parse the decomposition
        tasks = []
        for i, line in enumerate(decomposition.strip().split("\n")):
            if line.strip().startswith("TASK"):
                try:
                    # Extract specialist type and description
                    parts = line.split(":", 1)[1].strip()
                    specialist_str, description = parts.split("-", 1)
                    specialist_str = specialist_str.strip().lower()
                    
                    # Map to specialist type
                    specialist_type = None
                    for s in SpecialistType:
                        if s.value in specialist_str:
                            specialist_type = s
                            break
                    
                    if specialist_type:
                        task = Task(
                            task_id=f"task_{i+1}",
                            description=description.strip(),
                            specialist_type=specialist_type
                        )
                        tasks.append(task)
                except Exception as e:
                    print(f"Error parsing task line: {line} - {e}")
                    continue
        
        return tasks
    
    def assign_dependencies(self, tasks: List[Task]) -> List[Task]:
        """
        Determine task dependencies for proper execution order.
        
        Args:
            tasks: List of tasks to analyze
            
        Returns:
            Tasks with dependencies populated
        """
        # Simple heuristic: research -> analysis -> others
        for i, task in enumerate(tasks):
            if task.specialist_type == SpecialistType.ANALYSIS:
                # Analysis depends on research
                for prev_task in tasks[:i]:
                    if prev_task.specialist_type == SpecialistType.RESEARCH:
                        task.dependencies.append(prev_task.task_id)
            
            elif task.specialist_type in [SpecialistType.WRITING, SpecialistType.CREATIVE]:
                # Writing/creative depends on research and analysis
                for prev_task in tasks[:i]:
                    if prev_task.specialist_type in [SpecialistType.RESEARCH, SpecialistType.ANALYSIS]:
                        task.dependencies.append(prev_task.task_id)
        
        return tasks
    
    def execute_workflow(self, tasks: List[Task]) -> List[Task]:
        """
        Execute tasks in proper order respecting dependencies.
        
        Args:
            tasks: List of tasks to execute
            
        Returns:
            Completed tasks with results
        """
        completed_tasks: Dict[str, Task] = {}
        
        # Build context from completed tasks
        def get_context_for_task(task: Task) -> str:
            context_parts = []
            for dep_id in task.dependencies:
                if dep_id in completed_tasks:
                    dep_task = completed_tasks[dep_id]
                    context_parts.append(
                        f"From {dep_id} ({dep_task.specialist_type.value}):\n{dep_task.result}\n"
                    )
            return "\n".join(context_parts)
        
        # Execute tasks
        for task in tasks:
            print(f"\n[Leader] Assigning {task.task_id} to {task.specialist_type.value} specialist...")
            
            task.status = TaskStatus.IN_PROGRESS
            
            try:
                # Get specialist agent
                specialist = self.specialists[task.specialist_type]
                
                # Get context from dependencies
                context = get_context_for_task(task)
                
                # Execute task
                result = specialist.execute_task(task, context)
                
                task.result = result
                task.status = TaskStatus.COMPLETED
                completed_tasks[task.task_id] = task
                
                print(f"[Leader] ✓ {task.task_id} completed by {task.specialist_type.value}")
                
            except Exception as e:
                print(f"[Leader] ✗ {task.task_id} failed: {e}")
                task.status = TaskStatus.FAILED
                task.result = f"Error: {str(e)}"
        
        return tasks
    
    def synthesize_results(
        self,
        overall_task: str,
        tasks: List[Task]
    ) -> str:
        """
        Synthesize results from all tasks into final output.
        
        Args:
            overall_task: Original task description
            tasks: Completed tasks with results
            
        Returns:
            Final synthesized result
        """
        # Compile all task results
        task_results = "\n\n".join([
            f"**{task.task_id}** ({task.specialist_type.value}):\n{task.result}"
            for task in tasks
            if task.status == TaskStatus.COMPLETED
        ])
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a project coordinator synthesizing results from multiple specialists.
Create a cohesive, comprehensive final output that integrates all specialist contributions."""),
            ("user", """Original Task: {overall_task}

Specialist Results:
{task_results}

Synthesize these results into a coherent final output that addresses the original task.
Integrate insights from all specialists into a unified response.""")
        ])
        
        chain = synthesis_prompt | self.llm | StrOutputParser()
        
        final_output = chain.invoke({
            "overall_task": overall_task,
            "task_results": task_results
        })
        
        return final_output
    
    def coordinate(self, overall_task: str) -> WorkflowResult:
        """
        Coordinate the complete workflow from task to final output.
        
        Args:
            overall_task: The main task to complete
            
        Returns:
            WorkflowResult with final output and execution details
        """
        print(f"\n[Leader] Coordinating task: {overall_task}\n")
        
        # Step 1: Decompose task
        print("[Leader] Step 1: Decomposing task...")
        tasks = self.decompose_task(overall_task)
        print(f"[Leader] Created {len(tasks)} subtasks")
        
        for task in tasks:
            print(f"  - {task.task_id}: {task.description[:60]}... [{task.specialist_type.value}]")
        
        # Step 2: Assign dependencies
        print("\n[Leader] Step 2: Determining task dependencies...")
        tasks = self.assign_dependencies(tasks)
        
        # Step 3: Execute workflow
        print("\n[Leader] Step 3: Executing workflow...")
        tasks = self.execute_workflow(tasks)
        
        # Step 4: Synthesize results
        print("\n[Leader] Step 4: Synthesizing results...")
        final_output = self.synthesize_results(overall_task, tasks)
        
        # Create execution summary
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        execution_summary = f"Completed {completed}/{len(tasks)} tasks using {len(set(t.specialist_type for t in tasks))} specialist types"
        
        print(f"\n[Leader] Workflow complete! {execution_summary}")
        
        return WorkflowResult(
            final_output=final_output,
            tasks=tasks,
            execution_summary=execution_summary
        )


def demonstrate_leader_follower():
    """Demonstrate the Leader-Follower pattern with various examples."""
    
    print("=" * 80)
    print("LEADER-FOLLOWER PATTERN DEMONSTRATION")
    print("=" * 80)
    
    leader = LeaderAgent()
    
    # Test 1: Research and analysis task
    print("\n" + "=" * 80)
    print("TEST 1: Complex Research and Analysis Task")
    print("=" * 80)
    
    task1 = "Analyze the impact of artificial intelligence on the job market and write a summary report"
    
    result = leader.coordinate(task1)
    
    print("\n" + "-" * 80)
    print("FINAL OUTPUT:")
    print("-" * 80)
    print(result.final_output)
    
    print("\n" + "-" * 80)
    print("TASK BREAKDOWN:")
    print("-" * 80)
    for task in result.tasks:
        status_symbol = "✓" if task.status == TaskStatus.COMPLETED else "✗"
        print(f"{status_symbol} {task.task_id} [{task.specialist_type.value}]: {task.description[:60]}...")
    
    # Test 2: Creative and technical task
    print("\n" + "=" * 80)
    print("TEST 2: Creative + Technical Task")
    print("=" * 80)
    
    task2 = "Design a simple Python game concept and create a basic implementation plan"
    
    result = leader.coordinate(task2)
    
    print("\n" + "-" * 80)
    print("FINAL OUTPUT:")
    print("-" * 80)
    print(result.final_output[:500] + "...")
    
    print("\n" + "-" * 80)
    print("SPECIALIST UTILIZATION:")
    print("-" * 80)
    specialist_usage = {}
    for task in result.tasks:
        specialist_type = task.specialist_type.value
        specialist_usage[specialist_type] = specialist_usage.get(specialist_type, 0) + 1
    
    for specialist, count in specialist_usage.items():
        print(f"  {specialist}: {count} task(s)")
    
    # Test 3: Multi-domain problem
    print("\n" + "=" * 80)
    print("TEST 3: Multi-Domain Problem Solving")
    print("=" * 80)
    
    task3 = "Calculate the ROI of implementing renewable energy in a small business and provide recommendations"
    
    result = leader.coordinate(task3)
    
    print("\n" + "-" * 80)
    print("FINAL OUTPUT:")
    print("-" * 80)
    print(result.final_output)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Leader-Follower pattern demonstrates several key benefits:

1. **Specialization**: Each agent focuses on what it does best
2. **Coordination**: Leader manages workflow and dependencies
3. **Scalability**: Easy to add new specialist types
4. **Quality**: Specialist expertise improves output quality

Architecture:
- **Leader Agent**: Plans, delegates, coordinates, synthesizes
- **Follower Agents**: Execute specialized tasks with expertise
- **Task Management**: Handles dependencies and execution order
- **Result Synthesis**: Combines specialist outputs into coherent result

Use Cases:
- Complex workflows requiring multiple skills
- Projects benefiting from domain specialization
- Multi-step processes with dependencies
- Tasks requiring project management

The pattern is particularly effective when:
- Tasks can be decomposed into specialized subtasks
- Different subtasks require different expertise
- Coordination overhead is justified by quality improvement
- Clear task dependencies exist
""")


if __name__ == "__main__":
    demonstrate_leader_follower()
