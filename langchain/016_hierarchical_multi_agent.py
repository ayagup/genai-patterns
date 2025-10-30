"""
Pattern 016: Hierarchical Multi-Agent

Description:
    The Hierarchical Multi-Agent pattern organizes agents in a hierarchical structure with
    multiple levels of management and specialization. Managers coordinate workers, workers
    execute tasks, and specialists provide domain expertise. This creates a scalable
    organizational structure for complex problems.

Components:
    - Manager Agents: High-level coordination and decision-making
    - Worker Agents: Execute assigned tasks
    - Specialist Agents: Provide domain expertise
    - Communication Protocol: Top-down delegation and bottom-up reporting
    - Task Decomposition: Breaking complex tasks into hierarchical subtasks

Use Cases:
    - Large-scale project management
    - Complex multi-domain problems
    - Organizational modeling and simulation
    - Scalable distributed systems

LangChain Implementation:
    Uses multiple LLM instances representing different hierarchy levels,
    implements delegation chains, task assignment strategies, and result
    aggregation across organizational levels.

Key Features:
    - Multi-level hierarchy (Manager -> Worker -> Specialist)
    - Top-down task delegation
    - Bottom-up result reporting
    - Cross-functional coordination
"""

import os
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class AgentLevel(Enum):
    """Hierarchy levels in the organization."""
    EXECUTIVE = "executive"
    MANAGER = "manager"
    WORKER = "worker"
    SPECIALIST = "specialist"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class HierarchicalTask:
    """Represents a task in the hierarchy."""
    task_id: str
    description: str
    priority: TaskPriority
    assigned_to: Optional[str] = None
    assigned_level: Optional[AgentLevel] = None
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    result: Optional[str] = None
    completed: bool = False


@dataclass
class AgentReport:
    """Report from an agent to their manager."""
    agent_id: str
    agent_level: AgentLevel
    task_id: str
    status: str
    result: str
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class HierarchicalAgent:
    """
    Base class for agents in the hierarchy.
    """
    
    def __init__(
        self,
        agent_id: str,
        level: AgentLevel,
        specialty: Optional[str] = None,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize a hierarchical agent.
        
        Args:
            agent_id: Unique identifier
            level: Hierarchy level
            specialty: Optional domain specialty
            model: LLM model to use
        """
        self.agent_id = agent_id
        self.level = level
        self.specialty = specialty
        self.llm = ChatOpenAI(model=model, temperature=0.5)
        
        # Reporting relationships
        self.manager: Optional[str] = None
        self.direct_reports: List[str] = []
        
        # Task management
        self.assigned_tasks: List[HierarchicalTask] = []
        self.completed_tasks: List[HierarchicalTask] = []
    
    def get_role_description(self) -> str:
        """Get the role description for this agent."""
        descriptions = {
            AgentLevel.EXECUTIVE: "You are an executive leader responsible for strategic direction and high-level coordination.",
            AgentLevel.MANAGER: f"You are a manager responsible for coordinating teams and ensuring task completion. Specialty: {self.specialty or 'General'}",
            AgentLevel.WORKER: f"You are a worker who executes assigned tasks efficiently. Specialty: {self.specialty or 'General'}",
            AgentLevel.SPECIALIST: f"You are a specialist with deep expertise in {self.specialty}. You provide expert guidance and solutions."
        }
        return descriptions[self.level]
    
    def delegate_task(
        self,
        task: HierarchicalTask,
        subordinate_id: str
    ) -> str:
        """
        Delegate a task to a subordinate with clear instructions.
        
        Args:
            task: Task to delegate
            subordinate_id: ID of subordinate to delegate to
            
        Returns:
            Delegation message
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{self.get_role_description()}\n\nYou are delegating a task to {subordinate_id}."),
            ("user", """Task: {task_description}
Priority: {priority}

Provide clear delegation instructions including:
1. What needs to be done
2. Expected outcomes
3. Any constraints or guidelines
4. Reporting requirements

Keep it concise (3-4 sentences).""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        delegation = chain.invoke({
            "task_description": task.description,
            "priority": task.priority.value
        })
        
        return delegation.strip()
    
    def execute_task(
        self,
        task: HierarchicalTask,
        context: Optional[str] = None
    ) -> str:
        """
        Execute an assigned task.
        
        Args:
            task: Task to execute
            context: Optional context from manager
            
        Returns:
            Task result
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_role_description()),
            ("user", """Execute this task:

Task: {task_description}
Priority: {priority}
{context_section}

Provide your result:""")
        ])
        
        context_section = f"\nManager's Instructions:\n{context}" if context else ""
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({
            "task_description": task.description,
            "priority": task.priority.value,
            "context_section": context_section
        })
        
        return result.strip()
    
    def create_report(
        self,
        task: HierarchicalTask,
        result: str
    ) -> AgentReport:
        """
        Create a report for manager.
        
        Args:
            task: Completed task
            result: Task result
            
        Returns:
            Agent report
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{self.get_role_description()}\n\nCreate a concise status report for your manager."),
            ("user", """Task: {task_description}
Result: {result}

Provide:
1. Status (one word: Completed/Blocked/In-Progress)
2. Any issues encountered (or "None")
3. Brief recommendations (or "None")

Format:
STATUS: [status]
ISSUES: [issues]
RECOMMENDATIONS: [recommendations]""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        report_text = chain.invoke({
            "task_description": task.description,
            "result": result
        })
        
        # Parse report
        lines = report_text.strip().split("\n")
        status = "Completed"
        issues = []
        recommendations = []
        
        for line in lines:
            if line.startswith("STATUS:"):
                status = line.split(":", 1)[1].strip()
            elif line.startswith("ISSUES:"):
                issue_text = line.split(":", 1)[1].strip()
                if issue_text.lower() != "none":
                    issues = [issue_text]
            elif line.startswith("RECOMMENDATIONS:"):
                rec_text = line.split(":", 1)[1].strip()
                if rec_text.lower() != "none":
                    recommendations = [rec_text]
        
        return AgentReport(
            agent_id=self.agent_id,
            agent_level=self.level,
            task_id=task.task_id,
            status=status,
            result=result,
            issues=issues,
            recommendations=recommendations
        )


class HierarchicalOrganization:
    """
    Manages a hierarchical multi-agent organization.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize the hierarchical organization."""
        self.model = model
        self.agents: Dict[str, HierarchicalAgent] = {}
        self.tasks: Dict[str, HierarchicalTask] = {}
        self.reports: List[AgentReport] = []
        
        # Build default organization structure
        self._build_organization()
    
    def _build_organization(self):
        """Build a default organizational structure."""
        # Executive level
        self.add_agent("CEO", AgentLevel.EXECUTIVE, "Strategic Leadership")
        
        # Manager level
        self.add_agent("PM_Tech", AgentLevel.MANAGER, "Technology")
        self.add_agent("PM_Ops", AgentLevel.MANAGER, "Operations")
        self.add_agent("PM_Research", AgentLevel.MANAGER, "Research")
        
        # Worker level
        self.add_agent("Worker_Dev", AgentLevel.WORKER, "Development")
        self.add_agent("Worker_QA", AgentLevel.WORKER, "Quality Assurance")
        self.add_agent("Worker_Data", AgentLevel.WORKER, "Data Processing")
        
        # Specialist level
        self.add_agent("Spec_ML", AgentLevel.SPECIALIST, "Machine Learning")
        self.add_agent("Spec_Security", AgentLevel.SPECIALIST, "Security")
        
        # Set up reporting structure
        self.set_manager("PM_Tech", "CEO")
        self.set_manager("PM_Ops", "CEO")
        self.set_manager("PM_Research", "CEO")
        
        self.set_manager("Worker_Dev", "PM_Tech")
        self.set_manager("Worker_QA", "PM_Tech")
        self.set_manager("Worker_Data", "PM_Ops")
        
        self.set_manager("Spec_ML", "PM_Research")
        self.set_manager("Spec_Security", "PM_Tech")
    
    def add_agent(
        self,
        agent_id: str,
        level: AgentLevel,
        specialty: Optional[str] = None
    ):
        """Add an agent to the organization."""
        agent = HierarchicalAgent(agent_id, level, specialty, self.model)
        self.agents[agent_id] = agent
    
    def set_manager(self, agent_id: str, manager_id: str):
        """Set manager-subordinate relationship."""
        if agent_id in self.agents and manager_id in self.agents:
            self.agents[agent_id].manager = manager_id
            self.agents[manager_id].direct_reports.append(agent_id)
    
    def decompose_task(
        self,
        task_description: str,
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> List[HierarchicalTask]:
        """
        Decompose a high-level task into hierarchical subtasks.
        
        Args:
            task_description: Main task description
            priority: Task priority
            
        Returns:
            List of hierarchical tasks
        """
        # CEO decomposes into manager-level tasks
        ceo = self.agents["CEO"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", ceo.get_role_description()),
            ("user", """Decompose this high-level task into 3-4 manager-level subtasks:

Task: {task_description}

For each subtask, specify:
1. Which manager should handle it (PM_Tech, PM_Ops, or PM_Research)
2. Brief description

Format each as:
[MANAGER_ID]: [task description]""")
        ])
        
        chain = prompt | ceo.llm | StrOutputParser()
        
        decomposition = chain.invoke({"task_description": task_description})
        
        # Parse decomposition
        tasks = []
        parent_task = HierarchicalTask(
            task_id="task_0",
            description=task_description,
            priority=priority,
            assigned_to="CEO",
            assigned_level=AgentLevel.EXECUTIVE
        )
        self.tasks[parent_task.task_id] = parent_task
        tasks.append(parent_task)
        
        for i, line in enumerate(decomposition.strip().split("\n"), 1):
            if ":" in line:
                try:
                    manager_id, desc = line.split(":", 1)
                    manager_id = manager_id.strip().replace("[", "").replace("]", "")
                    
                    if manager_id in self.agents:
                        subtask = HierarchicalTask(
                            task_id=f"task_{i}",
                            description=desc.strip(),
                            priority=priority,
                            assigned_to=manager_id,
                            assigned_level=AgentLevel.MANAGER,
                            parent_task_id=parent_task.task_id
                        )
                        self.tasks[subtask.task_id] = subtask
                        parent_task.subtasks.append(subtask.task_id)
                        tasks.append(subtask)
                except:
                    continue
        
        return tasks
    
    def execute_hierarchical_workflow(
        self,
        main_task: str,
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> Dict[str, Any]:
        """
        Execute a complete hierarchical workflow.
        
        Args:
            main_task: Main task description
            priority: Task priority
            
        Returns:
            Workflow results and summary
        """
        print(f"\n[Organization] Executive assigning task: {main_task}")
        print(f"[Organization] Priority: {priority.value}\n")
        
        # Step 1: Decompose at executive level
        tasks = self.decompose_task(main_task, priority)
        
        print(f"[CEO] Decomposed into {len(tasks)-1} manager-level tasks")
        
        # Step 2: Managers delegate to workers
        for task in tasks[1:]:  # Skip parent task
            if task.assigned_level == AgentLevel.MANAGER:
                manager = self.agents[task.assigned_to]
                
                print(f"\n[{manager.agent_id}] Received task: {task.description[:60]}...")
                
                # Manager selects appropriate subordinate
                subordinates = [self.agents[sid] for sid in manager.direct_reports]
                if subordinates:
                    # Select first available subordinate (simplified)
                    selected = subordinates[0]
                    
                    # Create worker task
                    worker_task = HierarchicalTask(
                        task_id=f"{task.task_id}_w",
                        description=task.description,
                        priority=priority,
                        assigned_to=selected.agent_id,
                        assigned_level=selected.level,
                        parent_task_id=task.task_id
                    )
                    self.tasks[worker_task.task_id] = worker_task
                    task.subtasks.append(worker_task.task_id)
                    
                    # Delegate with instructions
                    delegation = manager.delegate_task(worker_task, selected.agent_id)
                    print(f"[{manager.agent_id}] Delegating to {selected.agent_id}")
                    print(f"   Instructions: {delegation[:80]}...")
                    
                    # Step 3: Workers execute
                    print(f"\n[{selected.agent_id}] Executing task...")
                    result = selected.execute_task(worker_task, delegation)
                    worker_task.result = result
                    worker_task.completed = True
                    
                    print(f"[{selected.agent_id}] Completed: {result[:80]}...")
                    
                    # Step 4: Workers report to manager
                    report = selected.create_report(worker_task, result)
                    self.reports.append(report)
                    
                    print(f"[{selected.agent_id}] Report status: {report.status}")
                    
                    # Step 5: Manager synthesizes results
                    task.result = result
                    task.completed = True
        
        # Step 6: CEO synthesizes executive summary
        print(f"\n[CEO] Synthesizing final results...")
        
        all_results = "\n\n".join([
            f"Manager {t.assigned_to}: {t.result}"
            for t in tasks[1:]
            if t.result
        ])
        
        exec_summary = self._create_executive_summary(main_task, all_results)
        tasks[0].result = exec_summary
        tasks[0].completed = True
        
        return {
            "main_task": main_task,
            "executive_summary": exec_summary,
            "manager_results": {t.assigned_to: t.result for t in tasks[1:] if t.result},
            "reports": self.reports,
            "total_tasks": len(self.tasks),
            "completed_tasks": sum(1 for t in self.tasks.values() if t.completed)
        }
    
    def _create_executive_summary(
        self,
        main_task: str,
        all_results: str
    ) -> str:
        """Create executive summary from all results."""
        ceo = self.agents["CEO"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", ceo.get_role_description()),
            ("user", """Synthesize an executive summary from the results below:

Main Task: {main_task}

Team Results:
{all_results}

Provide a concise executive summary (3-4 sentences) highlighting:
1. Overall completion status
2. Key findings
3. Strategic recommendations""")
        ])
        
        chain = prompt | ceo.llm | StrOutputParser()
        
        summary = chain.invoke({
            "main_task": main_task,
            "all_results": all_results
        })
        
        return summary.strip()
    
    def get_org_chart(self) -> str:
        """Generate a text representation of the org chart."""
        lines = []
        
        def print_agent(agent_id: str, indent: int = 0):
            agent = self.agents[agent_id]
            prefix = "  " * indent
            lines.append(f"{prefix}- {agent_id} ({agent.level.value}) [{agent.specialty or 'General'}]")
            
            for report_id in agent.direct_reports:
                print_agent(report_id, indent + 1)
        
        print_agent("CEO")
        return "\n".join(lines)


def demonstrate_hierarchical_multi_agent():
    """Demonstrate the Hierarchical Multi-Agent pattern."""
    
    print("=" * 80)
    print("HIERARCHICAL MULTI-AGENT PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Create organization
    org = HierarchicalOrganization()
    
    # Show org chart
    print("\n" + "=" * 80)
    print("ORGANIZATIONAL STRUCTURE")
    print("=" * 80)
    print(org.get_org_chart())
    
    # Test 1: Complex project
    print("\n" + "=" * 80)
    print("TEST 1: Complex Project Execution")
    print("=" * 80)
    
    task1 = "Launch a new mobile app with AI-powered features"
    
    result1 = org.execute_hierarchical_workflow(task1, TaskPriority.HIGH)
    
    print("\n" + "-" * 80)
    print("EXECUTIVE SUMMARY:")
    print("-" * 80)
    print(result1["executive_summary"])
    
    print("\n" + "-" * 80)
    print("DEPARTMENTAL RESULTS:")
    print("-" * 80)
    for manager, result in result1["manager_results"].items():
        print(f"\n{manager}:")
        print(f"  {result[:150]}...")
    
    print("\n" + "-" * 80)
    print("WORKFLOW METRICS:")
    print("-" * 80)
    print(f"  Total tasks created: {result1['total_tasks']}")
    print(f"  Completed tasks: {result1['completed_tasks']}")
    print(f"  Reports submitted: {len(result1['reports'])}")
    
    # Test 2: Strategic initiative
    print("\n" + "=" * 80)
    print("TEST 2: Strategic Initiative")
    print("=" * 80)
    
    task2 = "Improve system security and implement new compliance measures"
    
    result2 = org.execute_hierarchical_workflow(task2, TaskPriority.CRITICAL)
    
    print("\n" + "-" * 80)
    print("EXECUTIVE SUMMARY:")
    print("-" * 80)
    print(result2["executive_summary"])
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Hierarchical Multi-Agent pattern demonstrates several key benefits:

1. **Scalability**: Can handle large, complex tasks through decomposition
2. **Specialization**: Agents focus on their area of expertise
3. **Clear Accountability**: Well-defined reporting relationships
4. **Efficient Coordination**: Structured communication channels

Organizational Structure:
- **Executive Level**: Strategic direction and overall coordination
- **Manager Level**: Team coordination and task delegation
- **Worker Level**: Task execution and implementation
- **Specialist Level**: Domain expertise and specialized guidance

Communication Flow:
- **Top-Down**: Task delegation with clear instructions
- **Bottom-Up**: Status reports and results
- **Cross-Functional**: Coordination between departments

Use Cases:
- Large-scale project management
- Complex multi-domain problems
- Organizational simulation
- Scalable distributed systems

The pattern is particularly effective when:
- Tasks require multiple areas of expertise
- Clear accountability is important
- Scalability is needed for large projects
- Organizational modeling is beneficial
- Structured communication is preferred
""")


if __name__ == "__main__":
    demonstrate_hierarchical_multi_agent()
