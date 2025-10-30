"""
Blackboard System Pattern Implementation

This pattern implements a shared knowledge space where multiple agents contribute:
- Centralized knowledge board
- Multiple knowledge sources (agents)
- Control mechanism
- Collaborative problem-solving
- Opportunistic reasoning

Use cases:
- Complex problem solving
- Multi-expert systems
- Collaborative AI
- Distributed reasoning
- Knowledge integration
"""

from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time


class KnowledgeLevel(Enum):
    """Hierarchical levels of knowledge"""
    RAW_DATA = 1
    INFORMATION = 2
    INSIGHT = 3
    HYPOTHESIS = 4
    SOLUTION = 5


class KnowledgeState(Enum):
    """State of knowledge items"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATED = "validated"
    REJECTED = "rejected"
    FINAL = "final"


@dataclass
class KnowledgeItem:
    """Item on the blackboard"""
    id: str
    content: Any
    level: KnowledgeLevel
    state: KnowledgeState
    source: str
    confidence: float
    timestamp: float
    dependencies: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)


@dataclass
class KnowledgeSource:
    """Agent that contributes to blackboard"""
    name: str
    expertise: Set[str]
    activation_fn: Callable[[List[KnowledgeItem]], float]
    contribution_fn: Callable[[List[KnowledgeItem]], Optional[KnowledgeItem]]
    priority: int = 5


class Blackboard:
    """
    Centralized knowledge repository
    """
    
    def __init__(self, name: str = "MainBlackboard"):
        self.name = name
        self.items: Dict[str, KnowledgeItem] = {}
        self.item_counter = 0
        self.history: List[Dict[str, Any]] = []
    
    def add_item(self, item: KnowledgeItem) -> str:
        """Add item to blackboard"""
        self.items[item.id] = item
        self.history.append({
            "action": "add",
            "item_id": item.id,
            "source": item.source,
            "level": item.level.name,
            "timestamp": time.time()
        })
        return item.id
    
    def update_item(self, item_id: str, updates: Dict[str, Any]):
        """Update existing item"""
        if item_id in self.items:
            item = self.items[item_id]
            for key, value in updates.items():
                if hasattr(item, key):
                    setattr(item, key, value)
            
            self.history.append({
                "action": "update",
                "item_id": item_id,
                "updates": list(updates.keys()),
                "timestamp": time.time()
            })
    
    def get_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """Retrieve item by ID"""
        return self.items.get(item_id)
    
    def get_items_by_level(self, level: KnowledgeLevel) -> List[KnowledgeItem]:
        """Get all items at specific level"""
        return [item for item in self.items.values() if item.level == level]
    
    def get_items_by_state(self, state: KnowledgeState) -> List[KnowledgeItem]:
        """Get all items in specific state"""
        return [item for item in self.items.values() if item.state == state]
    
    def get_items_by_tags(self, tags: Set[str]) -> List[KnowledgeItem]:
        """Get items matching tags"""
        return [item for item in self.items.values() if item.tags & tags]
    
    def generate_id(self) -> str:
        """Generate unique ID for item"""
        self.item_counter += 1
        return f"KB{self.item_counter:04d}"
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of blackboard state"""
        return {
            "total_items": len(self.items),
            "by_level": {
                level.name: len(self.get_items_by_level(level))
                for level in KnowledgeLevel
            },
            "by_state": {
                state.name: len(self.get_items_by_state(state))
                for state in KnowledgeState
            },
            "history_size": len(self.history)
        }


class ControlMechanism:
    """
    Controls execution of knowledge sources
    """
    
    def __init__(self, blackboard: Blackboard):
        self.blackboard = blackboard
        self.knowledge_sources: List[KnowledgeSource] = []
        self.execution_history: List[Dict[str, Any]] = []
    
    def register_source(self, source: KnowledgeSource):
        """Register a knowledge source"""
        self.knowledge_sources.append(source)
        print(f"Registered knowledge source: {source.name} (expertise: {', '.join(source.expertise)})")
    
    def run_cycle(self) -> Dict[str, Any]:
        """Run one reasoning cycle"""
        current_items = list(self.blackboard.items.values())
        
        # Calculate activation levels for each source
        activations = []
        for source in self.knowledge_sources:
            activation = source.activation_fn(current_items)
            if activation > 0:
                activations.append((source, activation))
        
        if not activations:
            return {
                "cycle_complete": True,
                "contributions": 0,
                "reason": "No sources activated"
            }
        
        # Sort by activation level and priority
        activations.sort(key=lambda x: (x[1], x[0].priority), reverse=True)
        
        # Execute highest activated source
        source, activation_level = activations[0]
        
        contribution = source.contribution_fn(current_items)
        
        result = {
            "cycle_complete": True,
            "source": source.name,
            "activation": activation_level,
            "contribution": contribution is not None
        }
        
        if contribution:
            item_id = self.blackboard.add_item(contribution)
            result["item_id"] = item_id
            result["level"] = contribution.level.name
        
        self.execution_history.append(result)
        return result
    
    def solve_problem(self, max_cycles: int = 20) -> Dict[str, Any]:
        """Run multiple cycles to solve problem"""
        print(f"\n{'='*70}")
        print("Starting Problem Solving on Blackboard")
        print(f"{'='*70}")
        
        for cycle in range(max_cycles):
            print(f"\n--- Cycle {cycle + 1} ---")
            
            result = self.run_cycle()
            
            if result.get("contribution"):
                print(f"✓ {result['source']} contributed (activation: {result['activation']:.2f})")
                print(f"  Added: {result['level']} item ({result['item_id']})")
            else:
                print(f"✗ {result.get('reason', 'No contribution')}")
            
            # Check if problem is solved
            solutions = self.blackboard.get_items_by_level(KnowledgeLevel.SOLUTION)
            final_solutions = [s for s in solutions if s.state == KnowledgeState.FINAL]
            
            if final_solutions:
                print(f"\n🎉 Solution found after {cycle + 1} cycles!")
                return {
                    "solved": True,
                    "cycles": cycle + 1,
                    "solutions": len(final_solutions),
                    "total_items": len(self.blackboard.items)
                }
        
        return {
            "solved": False,
            "cycles": max_cycles,
            "reason": "Max cycles reached",
            "total_items": len(self.blackboard.items)
        }


class BlackboardSystem:
    """
    Complete blackboard system integrating all components
    """
    
    def __init__(self, name: str = "BlackboardSystem"):
        self.name = name
        self.blackboard = Blackboard()
        self.control = ControlMechanism(self.blackboard)
        self._register_default_sources()
    
    def _register_default_sources(self):
        """Register default knowledge sources"""
        
        # Data Collector - processes raw data
        def data_activation(items: List[KnowledgeItem]) -> float:
            raw_pending = [i for i in items 
                          if i.level == KnowledgeLevel.RAW_DATA 
                          and i.state == KnowledgeState.PENDING]
            return min(1.0, len(raw_pending) * 0.3)
        
        def data_contribution(items: List[KnowledgeItem]) -> Optional[KnowledgeItem]:
            raw_pending = [i for i in items 
                          if i.level == KnowledgeLevel.RAW_DATA 
                          and i.state == KnowledgeState.PENDING]
            
            if raw_pending:
                # Process first pending item
                raw_item = raw_pending[0]
                self.blackboard.update_item(raw_item.id, {"state": KnowledgeState.IN_PROGRESS})
                
                # Create information item
                return KnowledgeItem(
                    id=self.blackboard.generate_id(),
                    content=f"Processed: {raw_item.content}",
                    level=KnowledgeLevel.INFORMATION,
                    state=KnowledgeState.VALIDATED,
                    source="DataCollector",
                    confidence=0.8,
                    timestamp=time.time(),
                    dependencies={raw_item.id},
                    tags={"processed", "information"}
                )
            return None
        
        self.control.register_source(KnowledgeSource(
            name="DataCollector",
            expertise={"data_processing", "information_extraction"},
            activation_fn=data_activation,
            contribution_fn=data_contribution,
            priority=10
        ))
        
        # Analyzer - generates insights
        def analyzer_activation(items: List[KnowledgeItem]) -> float:
            info_validated = [i for i in items 
                             if i.level == KnowledgeLevel.INFORMATION 
                             and i.state == KnowledgeState.VALIDATED]
            return min(1.0, len(info_validated) * 0.4)
        
        def analyzer_contribution(items: List[KnowledgeItem]) -> Optional[KnowledgeItem]:
            info_items = [i for i in items 
                         if i.level == KnowledgeLevel.INFORMATION 
                         and i.state == KnowledgeState.VALIDATED]
            
            if len(info_items) >= 2:
                # Combine information into insight
                return KnowledgeItem(
                    id=self.blackboard.generate_id(),
                    content=f"Insight from {len(info_items)} information items",
                    level=KnowledgeLevel.INSIGHT,
                    state=KnowledgeState.VALIDATED,
                    source="Analyzer",
                    confidence=0.75,
                    timestamp=time.time(),
                    dependencies={i.id for i in info_items[:2]},
                    tags={"insight", "analyzed"}
                )
            return None
        
        self.control.register_source(KnowledgeSource(
            name="Analyzer",
            expertise={"pattern_recognition", "insight_generation"},
            activation_fn=analyzer_activation,
            contribution_fn=analyzer_contribution,
            priority=8
        ))
        
        # Hypothesis Generator
        def hypothesis_activation(items: List[KnowledgeItem]) -> float:
            insights = [i for i in items 
                       if i.level == KnowledgeLevel.INSIGHT 
                       and i.state == KnowledgeState.VALIDATED]
            return min(1.0, len(insights) * 0.5)
        
        def hypothesis_contribution(items: List[KnowledgeItem]) -> Optional[KnowledgeItem]:
            insights = [i for i in items 
                       if i.level == KnowledgeLevel.INSIGHT 
                       and i.state == KnowledgeState.VALIDATED]
            
            if insights:
                return KnowledgeItem(
                    id=self.blackboard.generate_id(),
                    content=f"Hypothesis based on {len(insights)} insights",
                    level=KnowledgeLevel.HYPOTHESIS,
                    state=KnowledgeState.VALIDATED,
                    source="HypothesisGenerator",
                    confidence=0.7,
                    timestamp=time.time(),
                    dependencies={i.id for i in insights},
                    tags={"hypothesis", "theory"}
                )
            return None
        
        self.control.register_source(KnowledgeSource(
            name="HypothesisGenerator",
            expertise={"hypothesis_formation", "theory_building"},
            activation_fn=hypothesis_activation,
            contribution_fn=hypothesis_contribution,
            priority=7
        ))
        
        # Solution Synthesizer
        def solution_activation(items: List[KnowledgeItem]) -> float:
            hypotheses = [i for i in items 
                         if i.level == KnowledgeLevel.HYPOTHESIS 
                         and i.state == KnowledgeState.VALIDATED]
            return min(1.0, len(hypotheses) * 0.6)
        
        def solution_contribution(items: List[KnowledgeItem]) -> Optional[KnowledgeItem]:
            hypotheses = [i for i in items 
                         if i.level == KnowledgeLevel.HYPOTHESIS 
                         and i.state == KnowledgeState.VALIDATED]
            
            if hypotheses:
                return KnowledgeItem(
                    id=self.blackboard.generate_id(),
                    content=f"Solution synthesized from {len(hypotheses)} hypotheses",
                    level=KnowledgeLevel.SOLUTION,
                    state=KnowledgeState.FINAL,
                    source="SolutionSynthesizer",
                    confidence=0.85,
                    timestamp=time.time(),
                    dependencies={i.id for i in hypotheses},
                    tags={"solution", "final"}
                )
            return None
        
        self.control.register_source(KnowledgeSource(
            name="SolutionSynthesizer",
            expertise={"solution_synthesis", "integration"},
            activation_fn=solution_activation,
            contribution_fn=solution_contribution,
            priority=6
        ))
    
    def add_problem_data(self, data: str, tags: Optional[Set[str]] = None):
        """Add raw data to blackboard"""
        item = KnowledgeItem(
            id=self.blackboard.generate_id(),
            content=data,
            level=KnowledgeLevel.RAW_DATA,
            state=KnowledgeState.PENDING,
            source="User",
            confidence=1.0,
            timestamp=time.time(),
            tags=tags or {"input"}
        )
        self.blackboard.add_item(item)
        print(f"Added raw data: {item.id}")
    
    def solve(self, problem_description: str, data_items: List[str]) -> Dict[str, Any]:
        """Solve a problem using blackboard"""
        print(f"\n{'='*70}")
        print(f"Problem: {problem_description}")
        print(f"{'='*70}")
        
        # Add all data items
        print("\nAdding problem data to blackboard...")
        for data in data_items:
            self.add_problem_data(data, {"problem_data"})
        
        # Run problem solving
        result = self.control.solve_problem()
        
        return {
            "problem": problem_description,
            "solved": result["solved"],
            "cycles": result["cycles"],
            "blackboard_state": self.blackboard.get_state_summary(),
            "solutions": [
                {
                    "id": s.id,
                    "content": s.content,
                    "confidence": s.confidence,
                    "dependencies": len(s.dependencies)
                }
                for s in self.blackboard.get_items_by_level(KnowledgeLevel.SOLUTION)
            ]
        }


def demo_blackboard_system():
    """Demonstrate blackboard system"""
    print("="*70)
    print("Blackboard System Pattern Demo")
    print("="*70)
    
    system = BlackboardSystem("MedicalDiagnosis")
    
    print("\n1. Problem Setup")
    print("-"*70)
    
    # Simulate medical diagnosis problem
    problem = "Diagnose patient condition based on symptoms"
    symptoms = [
        "Patient reports fever",
        "Patient has persistent cough",
        "Blood test shows elevated white blood cells",
        "X-ray indicates lung inflammation"
    ]
    
    result = system.solve(problem, symptoms)
    
    print("\n" + "="*70)
    print("2. Final Results")
    print("-"*70)
    
    import json
    print(json.dumps(result, indent=2))
    
    print("\n" + "="*70)
    print("3. Blackboard State Visualization")
    print("-"*70)
    
    for level in KnowledgeLevel:
        items = system.blackboard.get_items_by_level(level)
        print(f"\n{level.name} ({len(items)} items):")
        for item in items:
            print(f"  [{item.id}] {item.content[:60]}... ({item.state.name})")
    
    print("\n" + "="*70)
    print("4. Knowledge Flow")
    print("-"*70)
    
    print("\nProcessing Pipeline:")
    print("RAW_DATA → INFORMATION → INSIGHT → HYPOTHESIS → SOLUTION")
    
    print("\nDependency Chain:")
    solutions = system.blackboard.get_items_by_level(KnowledgeLevel.SOLUTION)
    if solutions:
        solution = solutions[0]
        print(f"Solution {solution.id} depends on:")
        for dep_id in solution.dependencies:
            dep = system.blackboard.get_item(dep_id)
            if dep:
                print(f"  {dep.id} ({dep.level.name}): {dep.content[:50]}...")
    
    print("\n" + "="*70)
    print("Blackboard System Pattern Complete!")
    print("="*70)


if __name__ == "__main__":
    demo_blackboard_system()
