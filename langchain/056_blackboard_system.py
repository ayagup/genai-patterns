"""
Pattern 056: Blackboard System

Description:
    The Blackboard System pattern implements a shared workspace (blackboard) where
    multiple independent knowledge sources (agents) collaboratively solve complex
    problems. Each knowledge source monitors the blackboard and contributes when
    it has relevant expertise, enabling opportunistic problem-solving through
    emergent collaboration.

Components:
    1. Blackboard: Shared data structure storing problem state
    2. Knowledge Sources (KS): Independent expert agents
    3. Control Module: Coordinates knowledge source activation
    4. Problem Space: Hierarchical representation of problem
    5. Solution Assembly: Combines partial solutions

Use Cases:
    - Complex multi-disciplinary problems
    - Speech and image recognition systems
    - Medical diagnosis with multiple specialists
    - Software architecture design
    - Scientific research synthesis
    - Strategic planning and decision-making

LangChain Implementation:
    Uses a shared state dictionary as blackboard, with multiple LLM-based
    knowledge sources that opportunistically contribute based on current state.
"""

import os
import time
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class KSPriority(Enum):
    """Priority levels for knowledge sources"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1


class BlackboardState(Enum):
    """States of blackboard processing"""
    INITIALIZED = "initialized"
    GATHERING_INFO = "gathering_info"
    ANALYZING = "analyzing"
    GENERATING_SOLUTIONS = "generating_solutions"
    REFINING = "refining"
    COMPLETE = "complete"


@dataclass
class BlackboardEntry:
    """Entry on the blackboard"""
    entry_id: str
    layer: str  # hypothesis, evidence, solution, etc.
    content: str
    contributor: str  # which KS added this
    confidence: float  # 0.0-1.0
    timestamp: datetime
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entry_id,
            "layer": self.layer,
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "contributor": self.contributor,
            "confidence": f"{self.confidence:.2f}",
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class KSActivation:
    """Knowledge source activation record"""
    ks_name: str
    trigger_condition: str
    contribution: str
    confidence: float
    timestamp: datetime


@dataclass
class Blackboard:
    """Shared workspace for collaborative problem-solving"""
    problem: str
    state: BlackboardState
    entries: List[BlackboardEntry] = field(default_factory=list)
    activations: List[KSActivation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_entry(self, entry: BlackboardEntry):
        """Add entry to blackboard"""
        self.entries.append(entry)
    
    def get_entries_by_layer(self, layer: str) -> List[BlackboardEntry]:
        """Get all entries in a specific layer"""
        return [e for e in self.entries if e.layer == layer]
    
    def get_latest_entries(self, n: int = 5) -> List[BlackboardEntry]:
        """Get most recent entries"""
        return sorted(self.entries, key=lambda x: x.timestamp, reverse=True)[:n]
    
    def record_activation(self, activation: KSActivation):
        """Record knowledge source activation"""
        self.activations.append(activation)
    
    def get_state_summary(self) -> str:
        """Get summary of current blackboard state"""
        layer_counts = {}
        for entry in self.entries:
            layer_counts[entry.layer] = layer_counts.get(entry.layer, 0) + 1
        
        summary = f"Problem: {self.problem}\n"
        summary += f"State: {self.state.value}\n"
        summary += f"Total Entries: {len(self.entries)}\n"
        summary += "Entries by Layer:\n"
        for layer, count in layer_counts.items():
            summary += f"  - {layer}: {count}\n"
        
        return summary


class KnowledgeSource:
    """Independent expert agent that contributes to blackboard"""
    
    def __init__(
        self,
        name: str,
        expertise: str,
        trigger_conditions: List[str],
        priority: KSPriority,
        llm: ChatOpenAI,
        system_prompt: str
    ):
        self.name = name
        self.expertise = expertise
        self.trigger_conditions = trigger_conditions
        self.priority = priority
        self.llm = llm
        self.system_prompt = system_prompt
        self.activation_count = 0
    
    def should_activate(self, blackboard: Blackboard) -> tuple[bool, str]:
        """Determine if this KS should contribute"""
        
        # Check if any trigger conditions are met
        bb_summary = blackboard.get_state_summary()
        latest_entries = blackboard.get_latest_entries(3)
        
        for condition in self.trigger_conditions:
            # Simple keyword matching (could be more sophisticated)
            if condition.lower() in bb_summary.lower():
                return True, f"Triggered by: {condition}"
            
            # Check recent entries
            for entry in latest_entries:
                if condition.lower() in entry.content.lower():
                    return True, f"Triggered by entry: {entry.entry_id}"
        
        return False, ""
    
    def contribute(self, blackboard: Blackboard) -> Optional[BlackboardEntry]:
        """Generate contribution to blackboard"""
        
        # Get context from blackboard
        recent_entries = blackboard.get_latest_entries(5)
        context = "\n".join([
            f"[{e.layer}] {e.content[:100]}"
            for e in recent_entries
        ])
        
        # Generate contribution
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", """Problem: {problem}

Current Blackboard State:
{context}

Based on your expertise in {expertise}, provide your contribution.
Be specific and build on existing information.

Your contribution:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            contribution = chain.invoke({
                "problem": blackboard.problem,
                "context": context or "Empty blackboard",
                "expertise": self.expertise
            })
            
            # Determine layer for contribution
            layer = "analysis"  # Default
            if "hypothesis" in contribution.lower() or "theory" in contribution.lower():
                layer = "hypothesis"
            elif "evidence" in contribution.lower() or "data" in contribution.lower():
                layer = "evidence"
            elif "solution" in contribution.lower() or "approach" in contribution.lower():
                layer = "solution"
            elif "recommend" in contribution.lower():
                layer = "recommendation"
            
            # Create entry
            entry = BlackboardEntry(
                entry_id=f"{self.name}_{len(blackboard.entries)}",
                layer=layer,
                content=contribution,
                contributor=self.name,
                confidence=0.7 + (self.priority.value * 0.05),
                timestamp=datetime.now()
            )
            
            self.activation_count += 1
            
            return entry
            
        except Exception as e:
            print(f"Error in {self.name} contribution: {str(e)}")
            return None


class BlackboardControl:
    """Controls knowledge source activation and coordination"""
    
    def __init__(self, problem: str):
        self.blackboard = Blackboard(
            problem=problem,
            state=BlackboardState.INITIALIZED
        )
        self.knowledge_sources: List[KnowledgeSource] = []
        self.max_iterations = 10
        self.current_iteration = 0
    
    def register_knowledge_source(self, ks: KnowledgeSource):
        """Register a knowledge source"""
        self.knowledge_sources.append(ks)
    
    def _select_next_ks(self) -> Optional[KnowledgeSource]:
        """Select next knowledge source to activate"""
        
        # Find all KS that can contribute
        candidates = []
        
        for ks in self.knowledge_sources:
            should_activate, trigger = ks.should_activate(self.blackboard)
            if should_activate:
                candidates.append((ks, trigger))
        
        if not candidates:
            return None
        
        # Sort by priority and activation count (prefer high priority, low activation)
        candidates.sort(
            key=lambda x: (x[0].priority.value, -x[0].activation_count),
            reverse=True
        )
        
        return candidates[0][0]
    
    def _update_state(self):
        """Update blackboard state based on progress"""
        
        entry_count = len(self.blackboard.entries)
        
        if entry_count == 0:
            self.blackboard.state = BlackboardState.INITIALIZED
        elif entry_count < 3:
            self.blackboard.state = BlackboardState.GATHERING_INFO
        elif entry_count < 6:
            self.blackboard.state = BlackboardState.ANALYZING
        elif entry_count < 9:
            self.blackboard.state = BlackboardState.GENERATING_SOLUTIONS
        else:
            self.blackboard.state = BlackboardState.REFINING
    
    def solve(self, max_iterations: Optional[int] = None) -> Blackboard:
        """Execute blackboard problem-solving cycle"""
        
        if max_iterations:
            self.max_iterations = max_iterations
        
        print(f"\n🎯 Starting blackboard problem-solving for: {self.blackboard.problem}")
        
        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            
            print(f"\n--- Iteration {self.current_iteration} ---")
            print(f"State: {self.blackboard.state.value}")
            
            # Select next KS to activate
            next_ks = self._select_next_ks()
            
            if not next_ks:
                print("No knowledge sources can contribute further")
                break
            
            print(f"Activating: {next_ks.name} (Priority: {next_ks.priority.name})")
            
            # Get contribution
            entry = next_ks.contribute(self.blackboard)
            
            if entry:
                self.blackboard.add_entry(entry)
                print(f"Contribution: [{entry.layer}] {entry.content[:80]}...")
                
                # Record activation
                activation = KSActivation(
                    ks_name=next_ks.name,
                    trigger_condition=f"Iteration {self.current_iteration}",
                    contribution=entry.content,
                    confidence=entry.confidence,
                    timestamp=datetime.now()
                )
                self.blackboard.record_activation(activation)
            
            # Update state
            self._update_state()
            
            # Check if we have enough solutions
            solutions = self.blackboard.get_entries_by_layer("solution")
            recommendations = self.blackboard.get_entries_by_layer("recommendation")
            
            if len(solutions) >= 2 and len(recommendations) >= 1:
                print("\nSufficient solutions generated")
                self.blackboard.state = BlackboardState.COMPLETE
                break
        
        return self.blackboard


def create_default_knowledge_sources() -> List[KnowledgeSource]:
    """Create default set of knowledge sources"""
    
    knowledge_sources = []
    
    # Problem Analyzer
    ks1 = KnowledgeSource(
        name="ProblemAnalyzer",
        expertise="problem decomposition and analysis",
        trigger_conditions=["initialized", "problem"],
        priority=KSPriority.CRITICAL,
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3),
        system_prompt="""You are a problem analysis expert. Break down complex problems
into components, identify key challenges, and frame the problem clearly."""
    )
    knowledge_sources.append(ks1)
    
    # Hypothesis Generator
    ks2 = KnowledgeSource(
        name="HypothesisGenerator",
        expertise="generating hypotheses and theories",
        trigger_conditions=["analysis", "decomp", "component"],
        priority=KSPriority.HIGH,
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7),
        system_prompt="""You are a creative thinker who generates hypotheses and theories.
Propose possible explanations and approaches based on the analysis."""
    )
    knowledge_sources.append(ks2)
    
    # Evidence Collector
    ks3 = KnowledgeSource(
        name="EvidenceCollector",
        expertise="gathering and evaluating evidence",
        trigger_conditions=["hypothesis", "theory", "claim"],
        priority=KSPriority.MEDIUM,
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2),
        system_prompt="""You are an evidence expert who evaluates claims and hypotheses.
Identify what evidence would support or refute the hypotheses."""
    )
    knowledge_sources.append(ks3)
    
    # Solution Designer
    ks4 = KnowledgeSource(
        name="SolutionDesigner",
        expertise="designing practical solutions",
        trigger_conditions=["evidence", "hypothesis", "analyzing"],
        priority=KSPriority.HIGH,
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5),
        system_prompt="""You are a solution architect who designs practical approaches.
Create concrete, actionable solutions based on the analysis and evidence."""
    )
    knowledge_sources.append(ks4)
    
    # Critic
    ks5 = KnowledgeSource(
        name="Critic",
        expertise="critical evaluation and refinement",
        trigger_conditions=["solution", "approach", "recommend"],
        priority=KSPriority.MEDIUM,
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4),
        system_prompt="""You are a critical evaluator who identifies weaknesses and risks.
Critique proposed solutions and suggest improvements."""
    )
    knowledge_sources.append(ks5)
    
    # Synthesizer
    ks6 = KnowledgeSource(
        name="Synthesizer",
        expertise="synthesis and final recommendations",
        trigger_conditions=["solution", "refining"],
        priority=KSPriority.CRITICAL,
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3),
        system_prompt="""You are a synthesizer who combines multiple perspectives.
Provide final recommendations that integrate all contributions."""
    )
    knowledge_sources.append(ks6)
    
    return knowledge_sources


def demonstrate_blackboard_system():
    """Demonstrate Blackboard System pattern"""
    
    print("=" * 80)
    print("PATTERN 056: BLACKBOARD SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("\nCollaborative problem-solving through shared workspace\n")
    
    # Test 1: Technical problem
    print("\n" + "=" * 80)
    print("TEST 1: Technical Problem Solving")
    print("=" * 80)
    
    problem1 = "How can we improve the performance of a slow web application?"
    
    control1 = BlackboardControl(problem1)
    
    # Register knowledge sources
    for ks in create_default_knowledge_sources():
        control1.register_knowledge_source(ks)
    
    print(f"\nRegistered {len(control1.knowledge_sources)} knowledge sources:")
    for ks in control1.knowledge_sources:
        print(f"  - {ks.name}: {ks.expertise}")
    
    # Solve problem
    result1 = control1.solve(max_iterations=8)
    
    print(f"\n📊 Final Blackboard State:")
    print(f"   Total Entries: {len(result1.entries)}")
    print(f"   Total Activations: {len(result1.activations)}")
    print(f"   Final State: {result1.state.value}")
    
    # Show contributions by layer
    print(f"\n📚 Entries by Layer:")
    layers = {}
    for entry in result1.entries:
        if entry.layer not in layers:
            layers[entry.layer] = []
        layers[entry.layer].append(entry)
    
    for layer, entries in layers.items():
        print(f"\n   {layer.upper()} ({len(entries)} entries):")
        for entry in entries[:2]:  # Show first 2 per layer
            print(f"      [{entry.contributor}] {entry.content[:100]}...")
    
    # Test 2: Strategic planning problem
    print("\n" + "=" * 80)
    print("TEST 2: Strategic Planning Problem")
    print("=" * 80)
    
    problem2 = "What strategy should a startup use to compete with established companies?"
    
    control2 = BlackboardControl(problem2)
    for ks in create_default_knowledge_sources():
        control2.register_knowledge_source(ks)
    
    result2 = control2.solve(max_iterations=6)
    
    print(f"\n🎯 Problem Solving Summary:")
    print(f"   Problem: {problem2}")
    print(f"   Iterations: {control2.current_iteration}")
    print(f"   Knowledge Sources Activated:")
    
    # Count activations per KS
    ks_counts = {}
    for activation in result2.activations:
        ks_counts[activation.ks_name] = ks_counts.get(activation.ks_name, 0) + 1
    
    for ks_name, count in sorted(ks_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"      {ks_name}: {count} times")
    
    # Show final recommendation
    recommendations = result2.get_entries_by_layer("recommendation")
    if recommendations:
        print(f"\n💡 Final Recommendation:")
        print(f"   {recommendations[-1].content[:300]}...")
    
    # Test 3: Show collaboration pattern
    print("\n" + "=" * 80)
    print("TEST 3: Collaboration Pattern Analysis")
    print("=" * 80)
    
    print(f"\n🔄 Knowledge Source Activation Sequence:")
    for i, activation in enumerate(result2.activations, 1):
        print(f"   {i}. {activation.ks_name} (confidence: {activation.confidence:.2f})")
    
    print(f"\n🏗️  Blackboard Evolution:")
    for i in range(0, len(result2.entries), 2):
        entry = result2.entries[i]
        print(f"   Entry {i+1}: [{entry.layer}] by {entry.contributor}")
    
    # Summary
    print("\n" + "=" * 80)
    print("BLACKBOARD SYSTEM PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Opportunistic Problem-Solving: KS contribute when relevant
2. Emergent Collaboration: No predefined workflow
3. Modularity: Independent knowledge sources
4. Incremental Progress: Builds solution step-by-step
5. Transparency: All contributions visible on blackboard

System Components:
1. Blackboard: Shared workspace with layered structure
2. Knowledge Sources: Independent expert agents
3. Control Module: Coordinates KS activation
4. Activation Conditions: Triggers for KS contributions
5. Priority System: Manages contribution order

Blackboard Layers:
- Hypothesis: Theories and possible explanations
- Evidence: Supporting or refuting data
- Analysis: Problem decomposition and insights
- Solution: Concrete approaches and methods
- Recommendation: Final synthesized advice

Knowledge Source Types:
1. Problem Analyzer: Breaks down problems
2. Hypothesis Generator: Creates theories
3. Evidence Collector: Gathers supporting data
4. Solution Designer: Designs approaches
5. Critic: Evaluates and refines
6. Synthesizer: Combines perspectives

Control Strategies:
1. Priority-Based: High-priority KS activate first
2. Trigger-Based: KS activate when conditions met
3. Load-Balanced: Distribute activations evenly
4. State-Driven: Blackboard state guides activation

Activation Patterns:
- Initial Analysis: Problem decomposition
- Hypothesis Generation: Multiple theories
- Evidence Collection: Supporting data
- Solution Design: Practical approaches
- Critical Evaluation: Refinement
- Synthesis: Final recommendations

Use Cases:
- Complex multi-disciplinary problems
- Medical diagnosis systems
- Scientific research synthesis
- Software architecture design
- Strategic planning
- Signal processing and recognition

Best Practices:
1. Clear KS expertise boundaries
2. Well-defined trigger conditions
3. Appropriate priority levels
4. Layered blackboard structure
5. Limit iteration count
6. Record all contributions
7. Final synthesis step

Production Considerations:
- Blackboard persistence
- Concurrent KS activation
- Conflict resolution
- Termination criteria
- Performance optimization
- Scalability with many KS
- Debugging and tracing

Comparison with Related Patterns:
- vs. Multi-Agent: Shared state vs communication
- vs. Pipeline: Opportunistic vs sequential
- vs. Mixture of Agents: Collaboration vs aggregation
- vs. Ensemble: Emergent vs predetermined combination

The Blackboard System pattern excels at complex problems requiring
multiple perspectives and emergent, opportunistic collaboration where
the solution path is not predetermined.
""")


if __name__ == "__main__":
    demonstrate_blackboard_system()
