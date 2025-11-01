"""
Pattern 119: Task Allocation & Scheduling

Description:
    Assigns tasks to agents based on capabilities, load, and constraints
    using auction-based or optimal assignment algorithms.

Components:
    - Task queue
    - Agent capabilities
    - Allocation algorithm
    - Load balancing

Use Cases:
    - Multi-agent systems
    - Distributed computing
    - Resource optimization

LangChain Implementation:
    Uses LLM-based task analysis and agent matching with LangGraph for orchestration.
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass
class Task:
    """Represents a task to be allocated."""
    id: str
    description: str
    required_skills: List[str]
    priority: int
    estimated_duration: int
    deadline: str = None


@dataclass
class Agent:
    """Represents an agent with capabilities."""
    id: str
    name: str
    skills: List[str]
    current_load: int = 0
    max_capacity: int = 100
    performance_history: List[float] = None
    
    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []


class TaskAllocationScheduler:
    """Schedules and allocates tasks to agents optimally."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.agents: List[Agent] = []
        self.task_queue: List[Task] = []
        self.assignments: Dict[str, str] = {}  # task_id -> agent_id
        
    def register_agent(self, agent: Agent):
        """Register an agent in the system."""
        self.agents.append(agent)
        print(f"✓ Registered agent: {agent.name} with skills {agent.skills}")
    
    def add_task(self, task: Task):
        """Add task to queue."""
        self.task_queue.append(task)
        print(f"✓ Added task: {task.description} (Priority: {task.priority})")
    
    def analyze_task_requirements(self, task: Task) -> Dict[str, Any]:
        """Analyze task requirements using LLM."""
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze this task and provide detailed requirements."),
            ("user", """Task: {description}
Required Skills: {skills}
Priority: {priority}

Provide:
1. Complexity level (1-10)
2. Required expertise level
3. Any special considerations""")
        ])
        
        chain = analysis_prompt | self.llm | StrOutputParser()
        analysis = chain.invoke({
            "description": task.description,
            "skills": ", ".join(task.required_skills),
            "priority": task.priority
        })
        
        return {"analysis": analysis}
    
    def calculate_agent_score(self, agent: Agent, task: Task) -> float:
        """Calculate how well an agent matches a task."""
        scoring_prompt = ChatPromptTemplate.from_messages([
            ("system", "Score how well this agent matches the task requirements (0-100)."),
            ("user", """Agent: {agent_name}
Agent Skills: {agent_skills}
Current Load: {current_load}/{max_capacity}
Performance History: {performance}

Task: {task_description}
Required Skills: {required_skills}
Priority: {priority}

Return only a numerical score (0-100):""")
        ])
        
        chain = scoring_prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "agent_name": agent.name,
            "agent_skills": ", ".join(agent.skills),
            "current_load": agent.current_load,
            "max_capacity": agent.max_capacity,
            "performance": sum(agent.performance_history) / len(agent.performance_history) if agent.performance_history else "No history",
            "task_description": task.description,
            "required_skills": ", ".join(task.required_skills),
            "priority": task.priority
        })
        
        try:
            score = float(response.strip().split()[0])
            
            # Adjust for current load
            load_factor = (agent.max_capacity - agent.current_load) / agent.max_capacity
            adjusted_score = score * load_factor
            
            return max(0.0, min(100.0, adjusted_score))
        except:
            return 0.0
    
    def allocate_tasks(self):
        """Allocate tasks to agents using optimal matching."""
        print("\n=== Task Allocation Process ===\n")
        
        # Sort tasks by priority
        sorted_tasks = sorted(self.task_queue, key=lambda t: t.priority, reverse=True)
        
        for task in sorted_tasks:
            print(f"Allocating: {task.description}")
            
            # Analyze task
            analysis = self.analyze_task_requirements(task)
            print(f"  Task Analysis: {analysis['analysis'][:100]}...")
            
            # Score all agents
            agent_scores = []
            for agent in self.agents:
                score = self.calculate_agent_score(agent, task)
                agent_scores.append((agent, score))
                print(f"  {agent.name}: {score:.1f}")
            
            # Select best agent
            if agent_scores:
                best_agent, best_score = max(agent_scores, key=lambda x: x[1])
                
                if best_score > 0:
                    self.assignments[task.id] = best_agent.id
                    best_agent.current_load += task.estimated_duration
                    print(f"  ✓ Assigned to: {best_agent.name} (Score: {best_score:.1f})")
                else:
                    print(f"  ✗ No suitable agent found")
            print()
        
        self.task_queue.clear()
    
    def get_schedule_summary(self) -> str:
        """Generate schedule summary."""
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize this task allocation and schedule."),
            ("user", """Assignments:
{assignments}

Agent Loads:
{agent_loads}

Provide:
1. Overall load distribution
2. Potential bottlenecks
3. Optimization suggestions""")
        ])
        
        assignments_str = "\n".join([
            f"Task {task_id} -> Agent {agent_id}"
            for task_id, agent_id in self.assignments.items()
        ])
        
        agent_loads_str = "\n".join([
            f"{agent.name}: {agent.current_load}/{agent.max_capacity} ({agent.current_load/agent.max_capacity*100:.1f}%)"
            for agent in self.agents
        ])
        
        chain = summary_prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "assignments": assignments_str,
            "agent_loads": agent_loads_str
        })
    
    def rebalance_load(self):
        """Rebalance load across agents if needed."""
        avg_load = sum(a.current_load for a in self.agents) / len(self.agents)
        overloaded = [a for a in self.agents if a.current_load > avg_load * 1.5]
        underloaded = [a for a in self.agents if a.current_load < avg_load * 0.5]
        
        if overloaded and underloaded:
            print("\n=== Load Rebalancing ===")
            print(f"Overloaded agents: {[a.name for a in overloaded]}")
            print(f"Underloaded agents: {[a.name for a in underloaded]}")
            print("Recommending task redistribution...")


def demonstrate_task_allocation():
    """Demonstrate task allocation and scheduling pattern."""
    print("=== Task Allocation & Scheduling Pattern ===\n")
    
    scheduler = TaskAllocationScheduler()
    
    # Register agents
    print("1. Registering Agents")
    print("-" * 50)
    
    scheduler.register_agent(Agent(
        id="agent1",
        name="Alice",
        skills=["python", "data_analysis", "machine_learning"],
        max_capacity=100,
        performance_history=[0.85, 0.90, 0.88]
    ))
    
    scheduler.register_agent(Agent(
        id="agent2",
        name="Bob",
        skills=["javascript", "frontend", "ui_design"],
        max_capacity=100,
        performance_history=[0.92, 0.89, 0.91]
    ))
    
    scheduler.register_agent(Agent(
        id="agent3",
        name="Charlie",
        skills=["devops", "cloud", "kubernetes"],
        max_capacity=100,
        performance_history=[0.88, 0.87, 0.90]
    ))
    
    scheduler.register_agent(Agent(
        id="agent4",
        name="Diana",
        skills=["python", "backend", "database"],
        max_capacity=100,
        performance_history=[0.90, 0.93, 0.91]
    ))
    
    print()
    
    # Add tasks
    print("2. Adding Tasks to Queue")
    print("-" * 50)
    
    scheduler.add_task(Task(
        id="task1",
        description="Build machine learning model for customer churn prediction",
        required_skills=["python", "machine_learning", "data_analysis"],
        priority=9,
        estimated_duration=40
    ))
    
    scheduler.add_task(Task(
        id="task2",
        description="Design and implement responsive dashboard UI",
        required_skills=["javascript", "frontend", "ui_design"],
        priority=8,
        estimated_duration=30
    ))
    
    scheduler.add_task(Task(
        id="task3",
        description="Set up CI/CD pipeline with Kubernetes deployment",
        required_skills=["devops", "cloud", "kubernetes"],
        priority=7,
        estimated_duration=35
    ))
    
    scheduler.add_task(Task(
        id="task4",
        description="Optimize database queries and add indexes",
        required_skills=["database", "backend", "python"],
        priority=8,
        estimated_duration=25
    ))
    
    scheduler.add_task(Task(
        id="task5",
        description="Implement user authentication system",
        required_skills=["backend", "python", "security"],
        priority=9,
        estimated_duration=45
    ))
    
    print()
    
    # Allocate tasks
    print("3. Allocating Tasks to Agents")
    print("-" * 50)
    scheduler.allocate_tasks()
    
    # Schedule summary
    print("4. Schedule Summary")
    print("-" * 50)
    summary = scheduler.get_schedule_summary()
    print(summary)
    print()
    
    # Check for rebalancing needs
    print("5. Load Balancing Check")
    print("-" * 50)
    scheduler.rebalance_load()
    
    print("\n=== Summary ===")
    print(f"Total assignments: {len(scheduler.assignments)}")
    print("Task allocation demonstrated with:")
    print("- Skill-based matching")
    print("- Load-aware assignment")
    print("- Priority-based scheduling")
    print("- Performance history consideration")
    print("- Load balancing recommendations")


if __name__ == "__main__":
    demonstrate_task_allocation()
