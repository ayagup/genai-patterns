"""
Cognitive Architecture Pattern Implementation

This pattern implements a comprehensive system modeling human-like cognition:
- Perception (input processing)
- Attention (focus management)
- Memory (short-term, long-term, working)
- Reasoning (problem solving)
- Action (decision execution)

Use cases:
- General intelligence research
- Complex autonomous agents
- Human-like AI systems
- Cognitive modeling
- Advanced agent architectures
"""

from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import time


class CognitiveState(Enum):
    """States of cognitive processing"""
    PERCEIVING = "perceiving"
    ATTENDING = "attending"
    THINKING = "thinking"
    DECIDING = "deciding"
    ACTING = "acting"
    REFLECTING = "reflecting"


class MemoryType(Enum):
    """Types of memory"""
    SENSORY = "sensory"
    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"
    PROCEDURAL = "procedural"


@dataclass
class Percept:
    """Represents a perceived input"""
    stimulus: str
    modality: str  # "text", "visual", "audio", etc.
    timestamp: float
    salience: float  # 0.0 to 1.0
    processed: bool = False


@dataclass
class MemoryItem:
    """Item stored in memory"""
    content: Any
    memory_type: MemoryType
    timestamp: float
    access_count: int = 0
    strength: float = 1.0  # Decays over time
    tags: Set[str] = field(default_factory=set)


@dataclass
class Goal:
    """Represents a goal"""
    description: str
    priority: float  # 0.0 to 1.0
    achieved: bool = False
    subgoals: List['Goal'] = field(default_factory=list)


@dataclass
class Action:
    """Represents an action"""
    name: str
    parameters: Dict[str, Any]
    expected_outcome: str
    executed: bool = False
    outcome: Optional[str] = None


class PerceptionModule:
    """Handles perception of inputs"""
    
    def __init__(self):
        self.percepts: List[Percept] = []
        self.filters: Dict[str, float] = {}  # Modality -> threshold
    
    def perceive(self, stimulus: str, modality: str = "text") -> Percept:
        """Process incoming stimulus"""
        salience = self._calculate_salience(stimulus, modality)
        
        percept = Percept(
            stimulus=stimulus,
            modality=modality,
            timestamp=time.time(),
            salience=salience
        )
        
        self.percepts.append(percept)
        return percept
    
    def _calculate_salience(self, stimulus: str, modality: str) -> float:
        """Calculate how salient/important a stimulus is"""
        salience = 0.5  # Base salience
        
        # Increase for urgent keywords
        urgent_keywords = ["urgent", "important", "critical", "error", "warning"]
        if any(keyword in stimulus.lower() for keyword in urgent_keywords):
            salience += 0.3
        
        # Increase for questions
        if "?" in stimulus:
            salience += 0.1
        
        # Increase for novel content
        if len(self.percepts) > 0:
            if stimulus not in [p.stimulus for p in self.percepts[-5:]]:
                salience += 0.1
        
        return min(1.0, salience)


class AttentionModule:
    """Manages attentional focus"""
    
    def __init__(self, capacity: int = 3):
        self.capacity = capacity
        self.focus: List[Any] = []
        self.attention_weights: Dict[str, float] = {}
    
    def focus_on(self, items: List[Percept]) -> List[Percept]:
        """Select items to focus on based on salience"""
        # Sort by salience
        sorted_items = sorted(items, key=lambda x: x.salience, reverse=True)
        
        # Focus on top items within capacity
        focused = sorted_items[:self.capacity]
        self.focus = focused
        
        return focused
    
    def shift_attention(self, new_focus: Any):
        """Shift attention to new item"""
        if len(self.focus) >= self.capacity:
            self.focus.pop(0)  # Remove oldest
        self.focus.append(new_focus)


class MemoryModule:
    """Manages different types of memory"""
    
    def __init__(self):
        self.memories: Dict[MemoryType, List[MemoryItem]] = {
            memory_type: [] for memory_type in MemoryType
        }
        self.working_memory_capacity = 7  # Miller's Law
    
    def store(self, content: Any, memory_type: MemoryType, tags: Optional[Set[str]] = None):
        """Store item in memory"""
        item = MemoryItem(
            content=content,
            memory_type=memory_type,
            timestamp=time.time(),
            tags=tags or set()
        )
        
        self.memories[memory_type].append(item)
        
        # Manage capacity for working memory
        if memory_type == MemoryType.WORKING:
            if len(self.memories[memory_type]) > self.working_memory_capacity:
                # Remove weakest item
                self.memories[memory_type].sort(key=lambda x: x.strength)
                removed = self.memories[memory_type].pop(0)
                # Move to long-term if strong enough
                if removed.strength > 0.5:
                    removed.memory_type = MemoryType.LONG_TERM
                    self.memories[MemoryType.LONG_TERM].append(removed)
    
    def retrieve(self, query: str, memory_type: Optional[MemoryType] = None) -> List[MemoryItem]:
        """Retrieve items from memory"""
        results = []
        
        # Search in specified memory type or all
        search_types = [memory_type] if memory_type else list(MemoryType)
        
        for mem_type in search_types:
            for item in self.memories[mem_type]:
                if self._matches_query(item, query):
                    item.access_count += 1
                    item.strength = min(1.0, item.strength + 0.1)  # Strengthen on access
                    results.append(item)
        
        return results
    
    def _matches_query(self, item: MemoryItem, query: str) -> bool:
        """Check if memory item matches query"""
        query_lower = query.lower()
        
        # Check content
        if isinstance(item.content, str):
            if query_lower in item.content.lower():
                return True
        
        # Check tags
        if any(query_lower in tag.lower() for tag in item.tags):
            return True
        
        return False
    
    def consolidate(self):
        """Consolidate memories (e.g., move from short-term to long-term)"""
        # Move strong short-term memories to long-term
        for item in self.memories[MemoryType.SHORT_TERM][:]:
            if item.strength > 0.7 and item.access_count > 2:
                self.memories[MemoryType.SHORT_TERM].remove(item)
                item.memory_type = MemoryType.LONG_TERM
                self.memories[MemoryType.LONG_TERM].append(item)


class ReasoningModule:
    """Handles reasoning and problem-solving"""
    
    def __init__(self):
        self.reasoning_history: List[Dict[str, Any]] = []
    
    def reason(self, problem: str, context: List[MemoryItem]) -> Dict[str, Any]:
        """Perform reasoning on a problem"""
        # Extract relevant information from context
        relevant_info = [item.content for item in context]
        
        # Simple reasoning simulation
        reasoning_steps = []
        
        # Step 1: Understand the problem
        reasoning_steps.append(f"Understanding: {problem}")
        
        # Step 2: Identify what we know
        if relevant_info:
            reasoning_steps.append(f"Known facts: {len(relevant_info)} items")
        
        # Step 3: Generate solution
        solution = self._generate_solution(problem, relevant_info)
        reasoning_steps.append(f"Solution: {solution}")
        
        result = {
            "problem": problem,
            "steps": reasoning_steps,
            "solution": solution,
            "confidence": self._estimate_confidence(problem, relevant_info)
        }
        
        self.reasoning_history.append(result)
        return result
    
    def _generate_solution(self, problem: str, context: List[Any]) -> str:
        """Generate solution (simulated)"""
        # In real implementation, this would use actual reasoning
        if "?" in problem:
            return f"Based on context, here's what I found: {len(context)} relevant items"
        else:
            return "Let me help you with that task"
    
    def _estimate_confidence(self, problem: str, context: List[Any]) -> float:
        """Estimate confidence in reasoning"""
        confidence = 0.5
        
        # More context = higher confidence
        confidence += min(0.3, len(context) * 0.1)
        
        # Shorter problems = higher confidence
        if len(problem) < 50:
            confidence += 0.1
        
        return min(1.0, confidence)


class ActionModule:
    """Handles action selection and execution"""
    
    def __init__(self):
        self.available_actions: Dict[str, Callable] = {}
        self.action_history: List[Action] = []
    
    def register_action(self, name: str, action_fn: Callable):
        """Register an available action"""
        self.available_actions[name] = action_fn
    
    def select_action(self, goal: Goal, reasoning_result: Dict[str, Any]) -> Action:
        """Select appropriate action for goal"""
        # Simple action selection
        action_name = self._determine_action(goal, reasoning_result)
        
        action = Action(
            name=action_name,
            parameters={"goal": goal.description},
            expected_outcome=f"Achieve: {goal.description}"
        )
        
        return action
    
    def execute_action(self, action: Action) -> str:
        """Execute an action"""
        if action.name in self.available_actions:
            result = self.available_actions[action.name](action.parameters)
            action.executed = True
            action.outcome = result
        else:
            result = f"Simulated execution of {action.name}"
            action.executed = True
            action.outcome = result
        
        self.action_history.append(action)
        return result
    
    def _determine_action(self, goal: Goal, reasoning_result: Dict[str, Any]) -> str:
        """Determine which action to take"""
        # Simple heuristic for action selection
        if "answer" in goal.description.lower():
            return "answer_question"
        elif "find" in goal.description.lower():
            return "search"
        elif "create" in goal.description.lower():
            return "generate"
        else:
            return "respond"


class CognitiveArchitecture:
    """
    Complete cognitive architecture integrating all modules
    """
    
    def __init__(self, name: str = "CognitiveAgent"):
        self.name = name
        self.state = CognitiveState.PERCEIVING
        
        # Initialize modules
        self.perception = PerceptionModule()
        self.attention = AttentionModule()
        self.memory = MemoryModule()
        self.reasoning = ReasoningModule()
        self.action = ActionModule()
        
        # Goals and monitoring
        self.goals: List[Goal] = []
        self.current_goal: Optional[Goal] = None
        self.processing_history: List[Dict[str, Any]] = []
        
        # Register default actions
        self._register_default_actions()
    
    def _register_default_actions(self):
        """Register default actions"""
        self.action.register_action(
            "answer_question",
            lambda params: f"Answering: {params.get('goal', 'unknown')}"
        )
        self.action.register_action(
            "search",
            lambda params: f"Searching for: {params.get('goal', 'unknown')}"
        )
        self.action.register_action(
            "generate",
            lambda params: f"Generating: {params.get('goal', 'unknown')}"
        )
        self.action.register_action(
            "respond",
            lambda params: f"Responding to: {params.get('goal', 'unknown')}"
        )
    
    def process(self, input_stimulus: str, goal_description: str) -> Dict[str, Any]:
        """Complete cognitive processing cycle"""
        cycle_start = time.time()
        
        # Create goal
        goal = Goal(description=goal_description, priority=1.0)
        self.goals.append(goal)
        self.current_goal = goal
        
        # Phase 1: PERCEPTION
        self.state = CognitiveState.PERCEIVING
        percept = self.perception.perceive(input_stimulus)
        self.memory.store(percept, MemoryType.SENSORY, tags={"input", "recent"})
        
        # Phase 2: ATTENTION
        self.state = CognitiveState.ATTENDING
        focused_percepts = self.attention.focus_on([percept])
        for p in focused_percepts:
            self.memory.store(p, MemoryType.SHORT_TERM, tags={"attended"})
        
        # Phase 3: REASONING
        self.state = CognitiveState.THINKING
        # Retrieve relevant memories
        relevant_memories = self.memory.retrieve(input_stimulus, MemoryType.LONG_TERM)
        relevant_memories.extend(self.memory.retrieve(input_stimulus, MemoryType.SHORT_TERM))
        
        # Perform reasoning
        reasoning_result = self.reasoning.reason(goal_description, relevant_memories)
        self.memory.store(
            reasoning_result,
            MemoryType.WORKING,
            tags={"reasoning", "recent"}
        )
        
        # Phase 4: DECISION
        self.state = CognitiveState.DECIDING
        selected_action = self.action.select_action(goal, reasoning_result)
        
        # Phase 5: ACTION
        self.state = CognitiveState.ACTING
        outcome = self.action.execute_action(selected_action)
        
        # Phase 6: REFLECTION
        self.state = CognitiveState.REFLECTING
        self.memory.consolidate()
        
        # Mark goal as achieved
        goal.achieved = True
        
        # Record processing
        processing_time = time.time() - cycle_start
        result = {
            "input": input_stimulus,
            "goal": goal_description,
            "percept_salience": percept.salience,
            "reasoning_steps": len(reasoning_result["steps"]),
            "reasoning_confidence": reasoning_result["confidence"],
            "action_taken": selected_action.name,
            "outcome": outcome,
            "processing_time_ms": int(processing_time * 1000),
            "memories_accessed": len(relevant_memories)
        }
        
        self.processing_history.append(result)
        return result
    
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get current cognitive state"""
        return {
            "name": self.name,
            "current_state": self.state.value,
            "current_goal": self.current_goal.description if self.current_goal else None,
            "goals_total": len(self.goals),
            "goals_achieved": sum(1 for g in self.goals if g.achieved),
            "attention_focus": len(self.attention.focus),
            "memory_stats": {
                mem_type.value: len(self.memory.memories[mem_type])
                for mem_type in MemoryType
            },
            "reasoning_history_size": len(self.reasoning.reasoning_history),
            "actions_taken": len(self.action.action_history)
        }


def demo_cognitive_architecture():
    """Demonstrate cognitive architecture"""
    print("="*70)
    print("Cognitive Architecture Pattern Demo")
    print("="*70)
    
    # Create cognitive agent
    agent = CognitiveArchitecture("CognitiveAgent-1")
    
    print("\n1. Initial Cognitive State")
    print("-"*70)
    import json
    print(json.dumps(agent.get_cognitive_state(), indent=2))
    
    print("\n" + "="*70)
    print("2. Processing Cycle 1: Simple Query")
    print("-"*70)
    
    result1 = agent.process(
        input_stimulus="What is artificial intelligence?",
        goal_description="Answer question about AI"
    )
    print(json.dumps(result1, indent=2))
    
    print("\n" + "="*70)
    print("3. Processing Cycle 2: Complex Task")
    print("-"*70)
    
    result2 = agent.process(
        input_stimulus="Find information about machine learning algorithms",
        goal_description="Search for ML algorithms"
    )
    print(json.dumps(result2, indent=2))
    
    print("\n" + "="*70)
    print("4. Processing Cycle 3: Creation Task")
    print("-"*70)
    
    result3 = agent.process(
        input_stimulus="Create a summary of recent interactions",
        goal_description="Generate summary"
    )
    print(json.dumps(result3, indent=2))
    
    print("\n" + "="*70)
    print("5. Final Cognitive State")
    print("-"*70)
    print(json.dumps(agent.get_cognitive_state(), indent=2))
    
    print("\n" + "="*70)
    print("6. Processing History")
    print("-"*70)
    
    for i, record in enumerate(agent.processing_history, 1):
        print(f"\nCycle {i}:")
        print(f"  Input: {record['input']}")
        print(f"  Goal: {record['goal']}")
        print(f"  Action: {record['action_taken']}")
        print(f"  Outcome: {record['outcome']}")
        print(f"  Confidence: {record['reasoning_confidence']:.2f}")
        print(f"  Processing Time: {record['processing_time_ms']}ms")
    
    print("\n" + "="*70)
    print("Cognitive Architecture Pattern Complete!")
    print("="*70)


if __name__ == "__main__":
    demo_cognitive_architecture()
