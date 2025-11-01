"""
Pattern 108: Collaborative Planning

Description:
    The Collaborative Planning pattern enables multiple agents to work together
    in creating and executing plans toward shared goals. Rather than planning in
    isolation, agents coordinate their planning efforts, share information, divide
    tasks, resolve conflicts, and synchronize their actions to achieve objectives
    that would be difficult or impossible for individual agents.
    
    Collaborative planning is essential for multi-agent systems where agents have
    complementary capabilities, shared resources, or interdependent goals. The
    pattern addresses challenges like distributed knowledge, conflicting preferences,
    resource contention, and coordination overhead.
    
    This pattern includes mechanisms for goal sharing, task decomposition and
    allocation, plan merging, conflict resolution, resource negotiation, and
    synchronized execution. Agents may use various collaboration strategies from
    centralized coordination to fully distributed consensus-based planning.

Key Components:
    1. Shared Goal Repository: Common objectives
    2. Task Allocator: Assigns tasks to agents
    3. Plan Merger: Combines individual plans
    4. Conflict Resolver: Handles plan conflicts
    5. Resource Negotiator: Allocates shared resources
    6. Synchronization Manager: Coordinates timing
    7. Communication Protocol: Agent interaction

Collaboration Strategies:
    1. Centralized: Single coordinator assigns tasks
    2. Hierarchical: Tree of coordinators
    3. Distributed: Peer-to-peer negotiation
    4. Auction-based: Agents bid for tasks
    5. Consensus: Democratic decision making
    6. Blackboard: Shared workspace

Planning Phases:
    1. Goal Identification: Define shared objectives
    2. Capability Sharing: Agents advertise skills
    3. Task Decomposition: Break into subtasks
    4. Task Allocation: Assign to appropriate agents
    5. Plan Integration: Merge individual plans
    6. Conflict Resolution: Resolve incompatibilities
    7. Synchronized Execution: Coordinate timing

Use Cases:
    - Multi-robot coordination
    - Distributed project management
    - Supply chain planning
    - Military operations planning
    - Smart city management
    - Collaborative problem solving
    - Team-based game AI

Advantages:
    - Leverages diverse capabilities
    - Handles complex tasks beyond single agent
    - Load balancing across agents
    - Fault tolerance through redundancy
    - Parallel execution speedup
    - Shared knowledge utilization
    - Flexible task distribution

Challenges:
    - Communication overhead
    - Conflict resolution complexity
    - Synchronization requirements
    - Trust and verification
    - Resource contention
    - Plan consistency maintenance
    - Scalability with agent count

LangChain Implementation:
    This implementation uses LangChain for:
    - LLM-based task decomposition
    - Agent capability matching
    - Plan integration strategies
    - Natural language coordination
    
Production Considerations:
    - Implement efficient communication protocols
    - Use asynchronous messaging
    - Handle agent failures gracefully
    - Monitor collaboration overhead
    - Implement timeout mechanisms
    - Log all coordination decisions
    - Support dynamic agent addition/removal
    - Enable human oversight
"""

import os
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class TaskStatus(Enum):
    """Status of a task."""
    UNASSIGNED = "unassigned"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageType(Enum):
    """Types of coordination messages."""
    TASK_OFFER = "task_offer"
    TASK_BID = "task_bid"
    TASK_ASSIGNMENT = "task_assignment"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_GRANT = "resource_grant"
    STATUS_UPDATE = "status_update"
    PLAN_PROPOSAL = "plan_proposal"


@dataclass
class Task:
    """
    Represents a task in collaborative plan.
    
    Attributes:
        task_id: Unique identifier
        description: Task description
        required_capabilities: Capabilities needed
        estimated_effort: Effort estimate
        dependencies: Tasks that must complete first
        assigned_to: Agent ID assigned to
        status: Current status
        priority: Task priority
    """
    task_id: str
    description: str
    required_capabilities: List[str] = field(default_factory=list)
    estimated_effort: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.UNASSIGNED
    priority: int = 5
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class CollaborativeAgent:
    """
    Agent participating in collaborative planning.
    
    Attributes:
        agent_id: Unique identifier
        name: Agent name
        capabilities: List of capabilities
        capacity: Available work capacity
        current_load: Current workload
        assigned_tasks: Tasks assigned to this agent
    """
    agent_id: str
    name: str
    capabilities: List[str] = field(default_factory=list)
    capacity: float = 10.0
    current_load: float = 0.0
    assigned_tasks: List[str] = field(default_factory=list)
    
    def can_handle(self, task: Task) -> bool:
        """Check if agent can handle a task."""
        # Check capabilities
        has_capabilities = all(
            cap in self.capabilities for cap in task.required_capabilities
        )
        # Check capacity
        has_capacity = (self.current_load + task.estimated_effort) <= self.capacity
        return has_capabilities and has_capacity
    
    def get_available_capacity(self) -> float:
        """Get remaining capacity."""
        return self.capacity - self.current_load


@dataclass
class Message:
    """
    Communication message between agents.
    
    Attributes:
        sender_id: Sending agent ID
        receiver_id: Receiving agent ID (None for broadcast)
        message_type: Type of message
        content: Message payload
        timestamp: When sent
    """
    sender_id: str
    receiver_id: Optional[str]
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class CollaborativePlanner:
    """
    Collaborative planning system for multi-agent coordination.
    
    This planner enables agents to work together in planning and
    executing tasks toward shared goals.
    """
    
    def __init__(self, temperature: float = 0.4):
        """
        Initialize collaborative planner.
        
        Args:
            temperature: LLM temperature
        """
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.agents: Dict[str, CollaborativeAgent] = {}
        self.tasks: Dict[str, Task] = {}
        self.messages: List[Message] = []
        self.task_counter = 0
    
    def add_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: List[str],
        capacity: float = 10.0
    ) -> CollaborativeAgent:
        """
        Add agent to the planning system.
        
        Args:
            agent_id: Agent identifier
            name: Agent name
            capabilities: Agent capabilities
            capacity: Work capacity
            
        Returns:
            Created agent
        """
        agent = CollaborativeAgent(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities,
            capacity=capacity
        )
        self.agents[agent_id] = agent
        return agent
    
    def decompose_goal(self, goal: str, num_tasks: int = 5) -> List[Task]:
        """
        Decompose goal into tasks using LLM.
        
        Args:
            goal: Goal description
            num_tasks: Number of tasks to create
            
        Returns:
            List of tasks
        """
        prompt = ChatPromptTemplate.from_template(
            "Decompose this goal into {num_tasks} concrete tasks:\n\n"
            "Goal: {goal}\n\n"
            "For each task, provide:\n"
            "- Description\n"
            "- Required capabilities (e.g., analysis, coding, communication)\n"
            "- Estimated effort (1-5)\n\n"
            "Format as:\n"
            "Task N: [description]\n"
            "Capabilities: [cap1, cap2]\n"
            "Effort: [1-5]\n"
            "---\n"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"goal": goal, "num_tasks": num_tasks})
        
        # Parse tasks
        tasks = []
        task_blocks = result.split("---")
        
        for block in task_blocks:
            if not block.strip():
                continue
            
            lines = [line.strip() for line in block.strip().split('\n')]
            description = ""
            capabilities = []
            effort = 1.0
            
            for line in lines:
                if line.startswith("Task"):
                    description = line.split(":", 1)[1].strip() if ":" in line else line
                elif line.startswith("Capabilities:"):
                    caps_text = line.split(":", 1)[1].strip()
                    capabilities = [c.strip() for c in caps_text.split(",")]
                elif line.startswith("Effort:"):
                    try:
                        effort = float(line.split(":")[1].strip())
                    except:
                        effort = 1.0
            
            if description:
                self.task_counter += 1
                task = Task(
                    task_id=f"task_{self.task_counter}",
                    description=description,
                    required_capabilities=capabilities,
                    estimated_effort=effort
                )
                self.tasks[task.task_id] = task
                tasks.append(task)
        
        return tasks
    
    def allocate_tasks_greedy(self) -> Dict[str, List[Task]]:
        """
        Allocate tasks to agents using greedy algorithm.
        
        Returns:
            Allocation map: agent_id -> tasks
        """
        allocation: Dict[str, List[Task]] = {aid: [] for aid in self.agents.keys()}
        
        # Sort tasks by priority (descending)
        unassigned = [t for t in self.tasks.values() if t.status == TaskStatus.UNASSIGNED]
        unassigned.sort(key=lambda t: t.priority, reverse=True)
        
        for task in unassigned:
            # Find capable agents
            capable_agents = [
                agent for agent in self.agents.values()
                if agent.can_handle(task)
            ]
            
            if not capable_agents:
                continue
            
            # Assign to agent with most available capacity
            capable_agents.sort(key=lambda a: a.get_available_capacity(), reverse=True)
            best_agent = capable_agents[0]
            
            # Assign task
            task.assigned_to = best_agent.agent_id
            task.status = TaskStatus.ASSIGNED
            best_agent.assigned_tasks.append(task.task_id)
            best_agent.current_load += task.estimated_effort
            allocation[best_agent.agent_id].append(task)
        
        return allocation
    
    def allocate_tasks_auction(self) -> Dict[str, List[Task]]:
        """
        Allocate tasks using auction-based mechanism.
        
        Returns:
            Allocation map: agent_id -> tasks
        """
        allocation: Dict[str, List[Task]] = {aid: [] for aid in self.agents.keys()}
        
        unassigned = [t for t in self.tasks.values() if t.status == TaskStatus.UNASSIGNED]
        
        for task in unassigned:
            # Agents bid for task
            bids: List[tuple] = []
            
            for agent in self.agents.values():
                if agent.can_handle(task):
                    # Bid value: higher if more capacity, lower if already loaded
                    bid_value = agent.get_available_capacity() / agent.capacity
                    bids.append((agent, bid_value))
            
            if bids:
                # Award to highest bidder
                bids.sort(key=lambda x: x[1], reverse=True)
                winner = bids[0][0]
                
                task.assigned_to = winner.agent_id
                task.status = TaskStatus.ASSIGNED
                winner.assigned_tasks.append(task.task_id)
                winner.current_load += task.estimated_effort
                allocation[winner.agent_id].append(task)
                
                # Send message
                self.messages.append(Message(
                    sender_id="coordinator",
                    receiver_id=winner.agent_id,
                    message_type=MessageType.TASK_ASSIGNMENT,
                    content={"task_id": task.task_id}
                ))
        
        return allocation
    
    def identify_conflicts(self) -> List[Dict[str, Any]]:
        """
        Identify conflicts in task assignments.
        
        Returns:
            List of conflicts
        """
        conflicts = []
        
        # Resource conflicts: agents over capacity
        for agent in self.agents.values():
            if agent.current_load > agent.capacity:
                conflicts.append({
                    "type": "capacity_exceeded",
                    "agent_id": agent.agent_id,
                    "load": agent.current_load,
                    "capacity": agent.capacity,
                    "excess": agent.current_load - agent.capacity
                })
        
        # Dependency conflicts: tasks assigned before dependencies
        for task in self.tasks.values():
            if task.status == TaskStatus.ASSIGNED:
                for dep_id in task.dependencies:
                    if dep_id in self.tasks:
                        dep_task = self.tasks[dep_id]
                        if dep_task.status == TaskStatus.UNASSIGNED:
                            conflicts.append({
                                "type": "dependency_unassigned",
                                "task_id": task.task_id,
                                "dependency_id": dep_id
                            })
        
        return conflicts
    
    def resolve_capacity_conflict(self, agent_id: str):
        """
        Resolve capacity conflict by reassigning tasks.
        
        Args:
            agent_id: Agent with capacity issue
        """
        agent = self.agents[agent_id]
        
        # Sort agent's tasks by priority (keep high priority)
        agent_tasks = [
            self.tasks[tid] for tid in agent.assigned_tasks
            if tid in self.tasks
        ]
        agent_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # Remove lowest priority tasks until under capacity
        while agent.current_load > agent.capacity and agent_tasks:
            removed_task = agent_tasks.pop()
            removed_task.status = TaskStatus.UNASSIGNED
            removed_task.assigned_to = None
            agent.assigned_tasks.remove(removed_task.task_id)
            agent.current_load -= removed_task.estimated_effort
    
    def execute_tasks(self, agent_id: str, num_tasks: int = 1) -> List[Task]:
        """
        Simulate executing tasks for an agent.
        
        Args:
            agent_id: Agent executing tasks
            num_tasks: Number of tasks to execute
            
        Returns:
            Completed tasks
        """
        agent = self.agents[agent_id]
        completed = []
        
        # Get assigned tasks that are ready (dependencies met)
        ready_tasks = []
        for task_id in agent.assigned_tasks:
            if task_id not in self.tasks:
                continue
            
            task = self.tasks[task_id]
            if task.status != TaskStatus.ASSIGNED:
                continue
            
            # Check dependencies
            deps_met = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
                if dep_id in self.tasks
            )
            
            if deps_met:
                ready_tasks.append(task)
        
        # Execute up to num_tasks
        for task in ready_tasks[:num_tasks]:
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            agent.current_load -= task.estimated_effort
            completed.append(task)
            
            # Send status update
            self.messages.append(Message(
                sender_id=agent_id,
                receiver_id=None,  # Broadcast
                message_type=MessageType.STATUS_UPDATE,
                content={
                    "task_id": task.task_id,
                    "status": "completed"
                }
            ))
        
        return completed
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get collaborative planning progress."""
        total_tasks = len(self.tasks)
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        in_progress = sum(1 for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS)
        assigned = sum(1 for t in self.tasks.values() if t.status == TaskStatus.ASSIGNED)
        unassigned = sum(1 for t in self.tasks.values() if t.status == TaskStatus.UNASSIGNED)
        
        return {
            "total_tasks": total_tasks,
            "completed": completed,
            "in_progress": in_progress,
            "assigned": assigned,
            "unassigned": unassigned,
            "completion_rate": completed / total_tasks if total_tasks > 0 else 0,
            "total_agents": len(self.agents),
            "messages_sent": len(self.messages)
        }
    
    def get_agent_summary(self) -> str:
        """Get summary of agent workloads."""
        lines = ["AGENT WORKLOADS:"]
        
        for agent in self.agents.values():
            utilization = (agent.current_load / agent.capacity * 100) if agent.capacity > 0 else 0
            lines.append(
                f"  {agent.name}: {len(agent.assigned_tasks)} tasks, "
                f"{agent.current_load:.1f}/{agent.capacity:.1f} capacity ({utilization:.0f}%)"
            )
        
        return "\n".join(lines)


def demonstrate_collaborative_planning():
    """Demonstrate collaborative planning pattern."""
    
    print("=" * 80)
    print("COLLABORATIVE PLANNING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic collaborative planning
    print("\n" + "=" * 80)
    print("Example 1: Multi-Agent Task Allocation")
    print("=" * 80)
    
    planner = CollaborativePlanner()
    
    # Add agents with different capabilities
    planner.add_agent("agent_1", "Alice", ["analysis", "coding"], capacity=10.0)
    planner.add_agent("agent_2", "Bob", ["coding", "testing"], capacity=10.0)
    planner.add_agent("agent_3", "Charlie", ["design", "communication"], capacity=10.0)
    
    print("\nAgents:")
    for agent in planner.agents.values():
        print(f"  • {agent.name}: {', '.join(agent.capabilities)}")
    
    # Decompose goal into tasks
    print("\nDecomposing goal...")
    tasks = planner.decompose_goal("Build a web application", num_tasks=5)
    
    print(f"\nCreated {len(tasks)} tasks:")
    for task in tasks:
        print(f"  {task.task_id}: {task.description}")
        print(f"    Capabilities: {', '.join(task.required_capabilities)}")
        print(f"    Effort: {task.estimated_effort}")
    
    # Allocate tasks
    print("\nAllocating tasks (greedy)...")
    allocation = planner.allocate_tasks_greedy()
    
    print("\nTask Allocation:")
    for agent_id, tasks in allocation.items():
        agent = planner.agents[agent_id]
        if tasks:
            print(f"  {agent.name}:")
            for task in tasks:
                print(f"    - {task.description}")
    
    print("\n" + planner.get_agent_summary())
    
    # Example 2: Auction-based allocation
    print("\n" + "=" * 80)
    print("Example 2: Auction-Based Task Allocation")
    print("=" * 80)
    
    planner2 = CollaborativePlanner()
    
    planner2.add_agent("agent_a", "Agent A", ["coding"], capacity=8.0)
    planner2.add_agent("agent_b", "Agent B", ["coding"], capacity=12.0)
    planner2.add_agent("agent_c", "Agent C", ["coding"], capacity=6.0)
    
    # Create tasks manually
    for i in range(4):
        planner2.task_counter += 1
        task = Task(
            task_id=f"task_{planner2.task_counter}",
            description=f"Coding task {i+1}",
            required_capabilities=["coding"],
            estimated_effort=3.0
        )
        planner2.tasks[task.task_id] = task
    
    print("\nRunning auction-based allocation...")
    allocation2 = planner2.allocate_tasks_auction()
    
    print("\nAuction Results:")
    for agent_id, tasks in allocation2.items():
        agent = planner2.agents[agent_id]
        print(f"  {agent.name}: {len(tasks)} tasks "
              f"(load: {agent.current_load}/{agent.capacity})")
    
    # Example 3: Conflict detection and resolution
    print("\n" + "=" * 80)
    print("Example 3: Conflict Detection and Resolution")
    print("=" * 80)
    
    planner3 = CollaborativePlanner()
    
    agent_d = planner3.add_agent("agent_d", "Agent D", ["task_skill"], capacity=5.0)
    
    # Create tasks that exceed capacity
    for i in range(4):
        planner3.task_counter += 1
        task = Task(
            task_id=f"task_{planner3.task_counter}",
            description=f"Task {i+1}",
            required_capabilities=["task_skill"],
            estimated_effort=2.0,
            status=TaskStatus.ASSIGNED,
            assigned_to="agent_d"
        )
        planner3.tasks[task.task_id] = task
        agent_d.assigned_tasks.append(task.task_id)
        agent_d.current_load += task.estimated_effort
    
    print(f"\nAgent D Load: {agent_d.current_load}/{agent_d.capacity}")
    print("Status: OVERLOADED")
    
    print("\nDetecting conflicts...")
    conflicts = planner3.identify_conflicts()
    
    for conflict in conflicts:
        print(f"  Conflict: {conflict['type']}")
        print(f"    Agent: {conflict['agent_id']}")
        print(f"    Excess load: {conflict['excess']:.1f}")
    
    print("\nResolving capacity conflict...")
    planner3.resolve_capacity_conflict("agent_d")
    
    print(f"\nAfter Resolution:")
    print(f"  Agent D Load: {agent_d.current_load}/{agent_d.capacity}")
    print(f"  Assigned tasks: {len(agent_d.assigned_tasks)}")
    unassigned = sum(1 for t in planner3.tasks.values() if t.status == TaskStatus.UNASSIGNED)
    print(f"  Unassigned tasks: {unassigned}")
    
    # Example 4: Dependency handling
    print("\n" + "=" * 80)
    print("Example 4: Task Dependencies")
    print("=" * 80)
    
    planner4 = CollaborativePlanner()
    
    planner4.add_agent("agent_e", "Agent E", ["any"], capacity=20.0)
    
    # Create tasks with dependencies
    task1 = Task(
        task_id="task_1",
        description="Design system",
        required_capabilities=["any"],
        estimated_effort=2.0,
        status=TaskStatus.ASSIGNED,
        assigned_to="agent_e"
    )
    
    task2 = Task(
        task_id="task_2",
        description="Implement system",
        required_capabilities=["any"],
        estimated_effort=3.0,
        dependencies=["task_1"],
        status=TaskStatus.ASSIGNED,
        assigned_to="agent_e"
    )
    
    task3 = Task(
        task_id="task_3",
        description="Test system",
        required_capabilities=["any"],
        estimated_effort=2.0,
        dependencies=["task_2"],
        status=TaskStatus.ASSIGNED,
        assigned_to="agent_e"
    )
    
    planner4.tasks.update({"task_1": task1, "task_2": task2, "task_3": task3})
    planner4.agents["agent_e"].assigned_tasks = ["task_1", "task_2", "task_3"]
    planner4.agents["agent_e"].current_load = 7.0
    
    print("\nTask Dependencies:")
    print("  task_1 (Design) → task_2 (Implement) → task_3 (Test)")
    
    print("\nExecuting tasks in order:")
    
    # Execute task 1
    completed = planner4.execute_tasks("agent_e", num_tasks=1)
    print(f"  ✓ Completed: {completed[0].description}")
    
    # Execute task 2
    completed = planner4.execute_tasks("agent_e", num_tasks=1)
    print(f"  ✓ Completed: {completed[0].description}")
    
    # Execute task 3
    completed = planner4.execute_tasks("agent_e", num_tasks=1)
    print(f"  ✓ Completed: {completed[0].description}")
    
    # Example 5: Progress tracking
    print("\n" + "=" * 80)
    print("Example 5: Collaborative Progress Tracking")
    print("=" * 80)
    
    planner5 = CollaborativePlanner()
    
    planner5.add_agent("agent_f", "Agent F", ["skill_a"], capacity=10.0)
    planner5.add_agent("agent_g", "Agent G", ["skill_b"], capacity=10.0)
    
    # Create and allocate tasks
    planner5.decompose_goal("Complete project", num_tasks=6)
    planner5.allocate_tasks_greedy()
    
    print("\nInitial Progress:")
    progress = planner5.get_progress_summary()
    print(f"  Total Tasks: {progress['total_tasks']}")
    print(f"  Assigned: {progress['assigned']}")
    print(f"  Unassigned: {progress['unassigned']}")
    print(f"  Completion Rate: {progress['completion_rate']:.0%}")
    
    # Simulate some work
    print("\nSimulating task execution...")
    for agent_id in planner5.agents.keys():
        planner5.execute_tasks(agent_id, num_tasks=2)
    
    print("\nUpdated Progress:")
    progress = planner5.get_progress_summary()
    print(f"  Completed: {progress['completed']}")
    print(f"  Completion Rate: {progress['completion_rate']:.0%}")
    print(f"  Messages Sent: {progress['messages_sent']}")
    
    # Example 6: Load balancing
    print("\n" + "=" * 80)
    print("Example 6: Load Balancing Across Agents")
    print("=" * 80)
    
    planner6 = CollaborativePlanner()
    
    planner6.add_agent("agent_h", "Agent H", ["general"], capacity=10.0)
    planner6.add_agent("agent_i", "Agent I", ["general"], capacity=15.0)
    planner6.add_agent("agent_j", "Agent J", ["general"], capacity=8.0)
    
    # Create uniform tasks
    for i in range(10):
        planner6.task_counter += 1
        task = Task(
            task_id=f"task_{planner6.task_counter}",
            description=f"Task {i+1}",
            required_capabilities=["general"],
            estimated_effort=2.0
        )
        planner6.tasks[task.task_id] = task
    
    print("\nBefore Allocation:")
    print(planner6.get_agent_summary())
    
    planner6.allocate_tasks_greedy()
    
    print("\nAfter Allocation:")
    print(planner6.get_agent_summary())
    
    # Example 7: Message-based coordination
    print("\n" + "=" * 80)
    print("Example 7: Message-Based Coordination")
    print("=" * 80)
    
    planner7 = CollaborativePlanner()
    
    planner7.add_agent("agent_k", "Agent K", ["skill"], capacity=10.0)
    planner7.add_agent("agent_l", "Agent L", ["skill"], capacity=10.0)
    
    # Create tasks and allocate
    for i in range(4):
        planner7.task_counter += 1
        task = Task(
            task_id=f"task_{planner7.task_counter}",
            description=f"Task {i+1}",
            required_capabilities=["skill"],
            estimated_effort=2.0
        )
        planner7.tasks[task.task_id] = task
    
    planner7.allocate_tasks_auction()
    
    print("\nCoordination Messages:")
    for msg in planner7.messages:
        receiver = msg.receiver_id or "ALL"
        print(f"  {msg.sender_id} → {receiver}: {msg.message_type.value}")
        if msg.message_type == MessageType.TASK_ASSIGNMENT:
            print(f"    Task: {msg.content['task_id']}")
    
    # Execute and send status updates
    for agent_id in planner7.agents.keys():
        planner7.execute_tasks(agent_id, num_tasks=1)
    
    print(f"\nTotal Messages: {len(planner7.messages)}")
    
    # Example 8: System summary
    print("\n" + "=" * 80)
    print("Example 8: Collaborative Planning Summary")
    print("=" * 80)
    
    planner8 = CollaborativePlanner()
    
    # Setup system
    planner8.add_agent("agent_m", "Specialist 1", ["analysis"], capacity=10.0)
    planner8.add_agent("agent_n", "Specialist 2", ["coding"], capacity=10.0)
    planner8.add_agent("agent_o", "Generalist", ["analysis", "coding"], capacity=12.0)
    
    planner8.decompose_goal("Build and deploy application", num_tasks=8)
    planner8.allocate_tasks_greedy()
    
    # Execute some tasks
    for agent_id in planner8.agents.keys():
        planner8.execute_tasks(agent_id, num_tasks=1)
    
    print("\nSYSTEM SUMMARY")
    print("=" * 60)
    print(planner8.get_agent_summary())
    
    print("\nPROGRESS:")
    progress = planner8.get_progress_summary()
    for key, value in progress.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Collaborative Planning Pattern")
    print("=" * 80)
    
    summary = """
    The Collaborative Planning pattern demonstrated:
    
    1. MULTI-AGENT ALLOCATION (Example 1):
       - Goal decomposition
       - Agent capabilities matching
       - Greedy task allocation
       - Workload distribution
       - LLM-based task generation
    
    2. AUCTION MECHANISM (Example 2):
       - Bidding-based allocation
       - Capacity-based bidding
       - Winner selection
       - Fair task distribution
       - Market-based coordination
    
    3. CONFLICT RESOLUTION (Example 3):
       - Capacity conflict detection
       - Over-allocation handling
       - Task reassignment
       - Constraint satisfaction
       - Automatic resolution
    
    4. DEPENDENCY MANAGEMENT (Example 4):
       - Task dependencies
       - Sequential execution
       - Prerequisite checking
       - Ordered completion
       - Dependency-aware scheduling
    
    5. PROGRESS TRACKING (Example 5):
       - Status monitoring
       - Completion rates
       - Message counting
       - System metrics
       - Real-time updates
    
    6. LOAD BALANCING (Example 6):
       - Capacity-aware allocation
       - Even workload distribution
       - Utilization optimization
       - Resource efficiency
       - Balanced assignment
    
    7. MESSAGE COORDINATION (Example 7):
       - Assignment messages
       - Status updates
       - Broadcast communication
       - Coordination protocol
       - Event notification
    
    8. SYSTEM INTEGRATION (Example 8):
       - Comprehensive summary
       - Agent workloads
       - Progress metrics
       - System statistics
       - Complete view
    
    KEY BENEFITS:
    ✓ Leverages diverse agent capabilities
    ✓ Handles complex distributed tasks
    ✓ Load balancing across agents
    ✓ Fault tolerance through redundancy
    ✓ Parallel execution speedup
    ✓ Shared knowledge utilization
    ✓ Flexible task distribution
    ✓ Scalable coordination
    
    USE CASES:
    • Multi-robot coordination
    • Distributed project management
    • Supply chain planning
    • Military operations planning
    • Smart city management
    • Collaborative problem solving
    • Team-based game AI
    • Cloud resource orchestration
    
    COLLABORATION STRATEGIES:
    → Centralized: Coordinator assigns
    → Hierarchical: Tree of coordinators
    → Distributed: Peer negotiation
    → Auction-based: Competitive bidding
    → Consensus: Democratic voting
    
    BEST PRACTICES:
    1. Implement efficient communication protocols
    2. Use asynchronous messaging
    3. Handle agent failures gracefully
    4. Monitor collaboration overhead
    5. Implement timeout mechanisms
    6. Log all coordination decisions
    7. Support dynamic agent join/leave
    8. Enable human oversight
    
    TRADE-OFFS:
    • Communication overhead vs. coordination quality
    • Centralized vs. distributed control
    • Planning time vs. execution efficiency
    • Autonomy vs. coordination
    
    PRODUCTION CONSIDERATIONS:
    → Use message queues for async communication
    → Implement heartbeat monitoring
    → Handle network partitions
    → Cache agent capabilities
    → Support graceful degradation
    → Monitor communication latency
    → Log all agent interactions
    → Implement conflict resolution strategies
    → Support dynamic reconfiguration
    → Enable human intervention points
    
    This pattern enables multiple agents to work together effectively
    by coordinating their planning, sharing tasks based on capabilities,
    and synchronizing execution toward shared goals.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_collaborative_planning()
