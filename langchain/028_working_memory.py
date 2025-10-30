"""
Pattern 028: Working Memory

Description:
    The Working Memory pattern provides an active workspace for the agent's current
    task processing. Unlike short-term memory (conversation history) or long-term
    memory (persistent storage), working memory holds intermediate results, active
    goals, and context needed for the current cognitive task. It's dynamically
    updated as the agent progresses through multi-step reasoning.

Components:
    - Active Goals: Current objectives being pursued
    - Intermediate Results: Partial solutions and computations
    - Task Context: Relevant information for current task
    - Attention Focus: What the agent is currently processing
    - Scratch Space: Temporary storage for calculations

Use Cases:
    - Multi-step problem solving
    - Complex reasoning tasks
    - Task decomposition and execution
    - Mathematical computations
    - Planning and execution monitoring

LangChain Implementation:
    Uses structured state management to track active goals, intermediate
    results, and task progress. Implements focus mechanisms to maintain
    relevant context during complex operations.

Key Features:
    - Task-scoped memory
    - Goal tracking
    - Intermediate result storage
    - Dynamic updates
    - Context relevance filtering
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


class GoalStatus(Enum):
    """Status of a goal in working memory."""
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"


@dataclass
class Goal:
    """Represents a goal in working memory."""
    id: str
    description: str
    status: GoalStatus = GoalStatus.ACTIVE
    parent_goal_id: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "parent_goal_id": self.parent_goal_id,
            "sub_goals": self.sub_goals,
            "result": str(self.result) if self.result is not None else None,
            "metadata": self.metadata
        }


@dataclass
class IntermediateResult:
    """Stores intermediate computation results."""
    id: str
    description: str
    value: Any
    goal_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "value": str(self.value),
            "goal_id": self.goal_id,
            "timestamp": self.timestamp.isoformat()
        }


class WorkingMemorySpace:
    """
    Working memory workspace for active task processing.
    """
    
    def __init__(self, capacity: int = 7):
        """
        Initialize working memory.
        
        Args:
            capacity: Maximum number of active items (Miller's Law: 7±2)
        """
        self.capacity = capacity
        self.goals: Dict[str, Goal] = {}
        self.intermediate_results: Dict[str, IntermediateResult] = {}
        self.task_context: Dict[str, Any] = {}
        self.attention_focus: Optional[str] = None  # Current goal ID
        self.scratch_space: Dict[str, Any] = {}
        
    def add_goal(
        self,
        goal_id: str,
        description: str,
        parent_goal_id: Optional[str] = None
    ) -> Goal:
        """
        Add a new goal to working memory.
        
        Args:
            goal_id: Unique goal identifier
            description: Goal description
            parent_goal_id: Parent goal if this is a sub-goal
            
        Returns:
            Created goal
        """
        goal = Goal(
            id=goal_id,
            description=description,
            parent_goal_id=parent_goal_id
        )
        
        self.goals[goal_id] = goal
        
        # Link to parent
        if parent_goal_id and parent_goal_id in self.goals:
            self.goals[parent_goal_id].sub_goals.append(goal_id)
        
        # Set as focus if first goal
        if self.attention_focus is None:
            self.attention_focus = goal_id
        
        return goal
    
    def update_goal_status(self, goal_id: str, status: GoalStatus, result: Any = None):
        """Update goal status and optionally store result."""
        if goal_id in self.goals:
            self.goals[goal_id].status = status
            if result is not None:
                self.goals[goal_id].result = result
    
    def store_intermediate(
        self,
        result_id: str,
        description: str,
        value: Any,
        goal_id: str
    ):
        """Store an intermediate result."""
        result = IntermediateResult(
            id=result_id,
            description=description,
            value=value,
            goal_id=goal_id
        )
        self.intermediate_results[result_id] = result
    
    def get_active_goals(self) -> List[Goal]:
        """Get all active goals."""
        return [
            goal for goal in self.goals.values()
            if goal.status in [GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS]
        ]
    
    def get_current_focus(self) -> Optional[Goal]:
        """Get currently focused goal."""
        if self.attention_focus and self.attention_focus in self.goals:
            return self.goals[self.attention_focus]
        return None
    
    def shift_focus(self, goal_id: str):
        """Shift attention to a different goal."""
        if goal_id in self.goals:
            self.attention_focus = goal_id
    
    def get_goal_context(self, goal_id: str) -> Dict[str, Any]:
        """Get all relevant context for a goal."""
        if goal_id not in self.goals:
            return {}
        
        goal = self.goals[goal_id]
        context = {
            "goal": goal.to_dict(),
            "intermediate_results": [
                ir.to_dict() for ir in self.intermediate_results.values()
                if ir.goal_id == goal_id
            ],
            "sub_goals": [
                self.goals[sg_id].to_dict()
                for sg_id in goal.sub_goals
                if sg_id in self.goals
            ]
        }
        
        return context
    
    def get_capacity_usage(self) -> float:
        """Get percentage of working memory capacity used."""
        active_items = len(self.get_active_goals()) + len(self.intermediate_results)
        return min(100.0, (active_items / self.capacity) * 100)
    
    def clear(self):
        """Clear working memory."""
        self.goals.clear()
        self.intermediate_results.clear()
        self.task_context.clear()
        self.attention_focus = None
        self.scratch_space.clear()
    
    def get_memory_state(self) -> Dict[str, Any]:
        """Get complete state of working memory."""
        return {
            "capacity": self.capacity,
            "capacity_usage": self.get_capacity_usage(),
            "total_goals": len(self.goals),
            "active_goals": len(self.get_active_goals()),
            "intermediate_results": len(self.intermediate_results),
            "current_focus": self.attention_focus,
            "goals": [g.to_dict() for g in self.goals.values()],
            "results": [ir.to_dict() for ir in self.intermediate_results.values()]
        }


class WorkingMemoryAgent:
    """
    Agent with working memory for complex task processing.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize agent with working memory.
        
        Args:
            model: LLM model to use
        """
        self.llm = ChatOpenAI(model=model, temperature=0.3)
        self.working_memory = WorkingMemorySpace()
        self.task_counter = 0
    
    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """
        Solve a problem using working memory.
        
        Args:
            problem: Problem description
            
        Returns:
            Solution with working memory trace
        """
        print(f"\n[Agent] Solving problem: {problem}")
        print(f"[Agent] Initializing working memory...\n")
        
        # Clear working memory for new task
        self.working_memory.clear()
        
        # Create main goal
        main_goal_id = "goal_0"
        self.working_memory.add_goal(main_goal_id, problem)
        
        # Decompose problem into sub-goals
        print("[Step 1] Decomposing problem into sub-goals...")
        sub_goals = self._decompose_problem(problem)
        
        for i, sub_goal in enumerate(sub_goals):
            sub_goal_id = f"goal_{i+1}"
            self.working_memory.add_goal(sub_goal_id, sub_goal, main_goal_id)
            print(f"  Sub-goal {i+1}: {sub_goal}")
        
        print(f"\n[Working Memory] Capacity: {self.working_memory.get_capacity_usage():.1f}%\n")
        
        # Process each sub-goal
        for i, sub_goal in enumerate(sub_goals):
            sub_goal_id = f"goal_{i+1}"
            
            # Shift focus to this sub-goal
            self.working_memory.shift_focus(sub_goal_id)
            current_focus = self.working_memory.get_current_focus()
            
            print(f"[Step {i+2}] Processing: {current_focus.description}")
            
            # Mark as in progress
            self.working_memory.update_goal_status(sub_goal_id, GoalStatus.IN_PROGRESS)
            
            # Solve sub-goal
            result = self._solve_sub_goal(sub_goal)
            
            # Store result
            self.working_memory.update_goal_status(
                sub_goal_id,
                GoalStatus.COMPLETED,
                result
            )
            
            # Store as intermediate result
            self.working_memory.store_intermediate(
                f"result_{i}",
                f"Solution to: {sub_goal}",
                result,
                sub_goal_id
            )
            
            print(f"  ✓ Result: {result}")
            print(f"  [Working Memory] Capacity: {self.working_memory.get_capacity_usage():.1f}%\n")
        
        # Synthesize final solution
        print("[Final Step] Synthesizing solution from sub-goals...")
        final_solution = self._synthesize_solution(problem)
        
        # Mark main goal as completed
        self.working_memory.update_goal_status(
            main_goal_id,
            GoalStatus.COMPLETED,
            final_solution
        )
        
        print(f"\n✓ Solution: {final_solution}\n")
        
        # Get memory state
        memory_state = self.working_memory.get_memory_state()
        
        return {
            "problem": problem,
            "solution": final_solution,
            "sub_goals": sub_goals,
            "working_memory_trace": memory_state
        }
    
    def _decompose_problem(self, problem: str) -> List[str]:
        """Decompose problem into sub-goals."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at problem decomposition. Break down
the given problem into 3-5 clear, actionable sub-goals."""),
            ("user", """Problem: {problem}

List 3-5 sub-goals needed to solve this problem. Be specific and actionable.
Respond with one sub-goal per line.""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"problem": problem})
        
        # Parse sub-goals
        lines = response.strip().split("\n")
        sub_goals = []
        for line in lines:
            line = line.strip()
            # Remove numbering
            for prefix in ["1.", "2.", "3.", "4.", "5.", "-", "*", "•"]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
            if line:
                sub_goals.append(line)
        
        return sub_goals[:5]  # Limit to 5 sub-goals
    
    def _solve_sub_goal(self, sub_goal: str) -> str:
        """Solve a single sub-goal."""
        # Get context from working memory
        focus = self.working_memory.get_current_focus()
        context = self.working_memory.get_goal_context(focus.id) if focus else {}
        
        # Get previously completed results
        previous_results = [
            ir for ir in self.working_memory.intermediate_results.values()
        ]
        
        previous_context = ""
        if previous_results:
            previous_context = "\n".join([
                f"- {ir.description}: {ir.value}"
                for ir in previous_results
            ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You solve sub-goals clearly and concisely."),
            ("user", """Sub-goal: {sub_goal}

Previous Results:
{previous_context}

Provide a clear, concise solution:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({
            "sub_goal": sub_goal,
            "previous_context": previous_context if previous_context else "None yet"
        })
        
        return result.strip()
    
    def _synthesize_solution(self, problem: str) -> str:
        """Synthesize final solution from sub-goal results."""
        # Get all intermediate results
        results_text = "\n".join([
            f"{i+1}. {ir.description}\n   {ir.value}"
            for i, ir in enumerate(self.working_memory.intermediate_results.values())
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You synthesize complete solutions from sub-goal results."),
            ("user", """Original Problem: {problem}

Sub-goal Results:
{results_text}

Provide a complete, coherent solution to the original problem:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        solution = chain.invoke({
            "problem": problem,
            "results_text": results_text
        })
        
        return solution.strip()


def demonstrate_working_memory():
    """Demonstrate the Working Memory pattern."""
    
    print("=" * 80)
    print("WORKING MEMORY PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = WorkingMemoryAgent()
    
    # Test 1: Multi-step math problem
    print("\n" + "=" * 80)
    print("TEST 1: Multi-Step Mathematical Problem")
    print("=" * 80)
    
    problem1 = """A store sells apples for $2 each and oranges for $3 each. 
If you buy 5 apples and 4 oranges, get a 10% discount, and pay 8% tax, 
what is the final price?"""
    
    result1 = agent.solve_problem(problem1)
    
    print("=" * 80)
    print("WORKING MEMORY TRACE")
    print("=" * 80)
    print(f"Goals tracked: {result1['working_memory_trace']['total_goals']}")
    print(f"Active computations: {result1['working_memory_trace']['intermediate_results']}")
    print(f"Capacity used: {result1['working_memory_trace']['capacity_usage']:.1f}%")
    
    # Test 2: Planning task
    print("\n\n" + "=" * 80)
    print("TEST 2: Planning Task")
    print("=" * 80)
    
    problem2 = """Plan a weekend trip to a nearby city including:
transportation, accommodation, activities, and budget estimation."""
    
    result2 = agent.solve_problem(problem2)
    
    print("=" * 80)
    print("WORKING MEMORY TRACE")
    print("=" * 80)
    print(f"Goals tracked: {result2['working_memory_trace']['total_goals']}")
    print(f"Sub-goals completed: {len(result2['sub_goals'])}")
    
    # Test 3: Analytical task
    print("\n\n" + "=" * 80)
    print("TEST 3: Analytical Reasoning")
    print("=" * 80)
    
    problem3 = """Analyze the pros and cons of remote work and provide
a recommendation for a tech company."""
    
    result3 = agent.solve_problem(problem3)
    
    print("=" * 80)
    print("WORKING MEMORY TRACE")
    print("=" * 80)
    for goal in result3['working_memory_trace']['goals']:
        print(f"\nGoal: {goal['description']}")
        print(f"  Status: {goal['status']}")
        if goal['result']:
            preview = goal['result'][:100] + "..." if len(goal['result']) > 100 else goal['result']
            print(f"  Result: {preview}")
    
    # Summary
    print("\n\n" + "=" * 80)
    print("PATTERN SUMMARY")
    print("=" * 80)
    print("""
The Working Memory pattern demonstrates:

1. **Active Workspace**: Maintains current task context
2. **Goal Tracking**: Monitors progress on objectives
3. **Intermediate Storage**: Stores partial results
4. **Attention Focus**: Tracks current processing target
5. **Capacity Management**: Limited capacity (Miller's Law)

Key Components:

Goals:
- Hierarchical goal structure (main goal → sub-goals)
- Status tracking (active, in-progress, completed, blocked)
- Result storage
- Parent-child relationships

Intermediate Results:
- Temporary computation storage
- Linked to specific goals
- Time-stamped for ordering
- Available for later steps

Attention Focus:
- Single goal actively processed
- Context switching between goals
- Maintains processing coherence

Working Memory Characteristics:

Capacity Limitation (Miller's Law):
- 7±2 items simultaneously
- Prevents cognitive overload
- Forces prioritization
- Matches human working memory

Dynamic Updates:
- Content changes as task progresses
- Old items removed when complete
- New items added as needed
- Maintains relevance

Task-Scoped:
- Cleared between tasks
- Fresh workspace for each problem
- No cross-task contamination

Differences from Other Memory Types:

vs Short-Term Memory:
- Working: Active processing workspace
- Short-term: Conversation history buffer
- Working is goal-oriented, short-term is sequential

vs Long-Term Memory:
- Working: Temporary, task-scoped
- Long-term: Permanent, cross-session
- Working is volatile, long-term persists

Use Cases:
- Multi-step problem solving
- Complex calculations
- Task planning and execution
- Reasoning chains
- Resource allocation

Benefits:
- **Organized Processing**: Structured approach to complexity
- **Progress Tracking**: Monitor goal completion
- **Context Maintenance**: Relevant information available
- **Intermediate Access**: Use partial results
- **Cognitive Modeling**: Mimics human working memory

This pattern is essential for agents tackling complex, multi-step
tasks that require maintaining and manipulating intermediate state.
""")


if __name__ == "__main__":
    demonstrate_working_memory()
