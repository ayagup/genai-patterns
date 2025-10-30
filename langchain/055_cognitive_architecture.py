"""
Pattern 055: Cognitive Architecture

Description:
    The Cognitive Architecture pattern models human-like cognitive processes in AI agents,
    including perception, attention, working memory, reasoning, planning, and action.
    This creates agents that process information in stages similar to human cognition,
    enabling more sophisticated and contextual decision-making.

Components:
    1. Perception Module: Processes and interprets inputs
    2. Attention Mechanism: Focuses on relevant information
    3. Working Memory: Maintains current context
    4. Long-Term Memory: Stores knowledge and experiences
    5. Reasoning Engine: Logical inference and problem-solving
    6. Planning Module: Goal-directed action planning
    7. Action Selection: Chooses and executes actions

Use Cases:
    - Complex decision-making systems
    - Autonomous agents with human-like reasoning
    - Educational tutoring systems
    - Game AI with realistic behavior
    - Personal assistants with context awareness
    - Research and analysis agents

LangChain Implementation:
    Implements a multi-stage cognitive pipeline using LangChain components
    to simulate perception, memory, reasoning, and action selection.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class CognitiveState(Enum):
    """States in cognitive processing cycle"""
    PERCEIVING = "perceiving"  # Processing input
    ATTENDING = "attending"  # Focusing attention
    ENCODING = "encoding"  # Storing in memory
    REASONING = "reasoning"  # Logical processing
    PLANNING = "planning"  # Creating action plan
    ACTING = "acting"  # Executing action
    REFLECTING = "reflecting"  # Evaluating outcome


class AttentionFocus(Enum):
    """What the agent is focusing on"""
    GOAL = "goal"  # Primary objective
    PROBLEM = "problem"  # Current problem
    CONTEXT = "context"  # Surrounding context
    MEMORY = "memory"  # Past experiences
    ACTION = "action"  # Potential actions


@dataclass
class PerceptualInput:
    """Input perceived by the agent"""
    raw_input: str
    modality: str  # "text", "visual", "audio", etc.
    timestamp: datetime
    features: Dict[str, Any] = field(default_factory=dict)
    
    def to_text(self) -> str:
        return f"[{self.modality}] {self.raw_input}"


@dataclass
class WorkingMemoryItem:
    """Item in working memory"""
    content: str
    relevance: float  # 0.0-1.0
    timestamp: datetime
    source: str  # perception, reasoning, memory
    
    def __str__(self) -> str:
        return f"{self.content} (rel: {self.relevance:.2f})"


@dataclass
class LongTermMemory:
    """Long-term memory store"""
    facts: List[str] = field(default_factory=list)
    experiences: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    
    def add_fact(self, fact: str):
        if fact not in self.facts:
            self.facts.append(fact)
    
    def add_experience(self, experience: str):
        self.experiences.append(experience)
        # Keep recent experiences
        if len(self.experiences) > 50:
            self.experiences = self.experiences[-50:]
    
    def recall(self, query: str, k: int = 3) -> List[str]:
        """Recall relevant memories"""
        # Simple keyword matching (could use embeddings in production)
        query_lower = query.lower()
        
        relevant = []
        for fact in self.facts:
            if any(word in fact.lower() for word in query_lower.split()):
                relevant.append(("fact", fact))
        
        for exp in self.experiences[-10:]:  # Recent experiences
            if any(word in exp.lower() for word in query_lower.split()):
                relevant.append(("experience", exp))
        
        return relevant[:k]


@dataclass
class ReasoningResult:
    """Result from reasoning process"""
    conclusion: str
    reasoning_steps: List[str]
    confidence: float
    used_memories: List[str]


@dataclass
class ActionPlan:
    """Plan for achieving goal"""
    goal: str
    steps: List[str]
    expected_outcome: str
    confidence: float


@dataclass
class CognitiveResult:
    """Complete result from cognitive processing"""
    input: str
    perception: str
    attention_focus: AttentionFocus
    reasoning: ReasoningResult
    plan: ActionPlan
    action: str
    reflection: str
    cognitive_trace: List[Tuple[CognitiveState, str]]
    execution_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input[:100] + "..." if len(self.input) > 100 else self.input,
            "attention": self.attention_focus.value,
            "conclusion": self.reasoning.conclusion[:100] + "...",
            "plan_steps": len(self.plan.steps),
            "action": self.action[:100] + "..." if len(self.action) > 100 else self.action,
            "execution_time_ms": f"{self.execution_time_ms:.1f}"
        }


class CognitiveAgent:
    """
    Agent with human-like cognitive architecture.
    
    Implements:
    1. Perception: Input processing and feature extraction
    2. Attention: Selective focus on relevant information
    3. Working Memory: Temporary context storage (limited capacity)
    4. Long-Term Memory: Persistent knowledge storage
    5. Reasoning: Logical inference and problem-solving
    6. Planning: Goal-directed action planning
    7. Action: Selection and execution
    8. Reflection: Learning from outcomes
    """
    
    def __init__(
        self,
        working_memory_capacity: int = 7,  # Miller's 7±2
        temperature: float = 0.7
    ):
        self.working_memory_capacity = working_memory_capacity
        self.temperature = temperature
        
        # Cognitive components
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
        
        # Memory systems
        self.working_memory: deque = deque(maxlen=working_memory_capacity)
        self.long_term_memory = LongTermMemory()
        
        # Current state
        self.current_goal: Optional[str] = None
        self.attention_focus: AttentionFocus = AttentionFocus.CONTEXT
        
        # Cognitive trace for transparency
        self.cognitive_trace: List[Tuple[CognitiveState, str]] = []
    
    def _perceive(self, input_text: str) -> Tuple[str, PerceptualInput]:
        """Perception: Process and interpret input"""
        
        self.cognitive_trace.append((CognitiveState.PERCEIVING, "Processing input"))
        
        # Create perceptual input
        perceptual_input = PerceptualInput(
            raw_input=input_text,
            modality="text",
            timestamp=datetime.now()
        )
        
        # Extract features and intent
        perception_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are processing sensory input. Extract:
1. Main topic/subject
2. Intent (question, request, statement, etc.)
3. Key entities mentioned
4. Emotional tone

Format: Topic: X | Intent: Y | Entities: Z | Tone: W"""),
            ("user", "{input}")
        ])
        
        chain = perception_prompt | self.llm | StrOutputParser()
        perception_result = chain.invoke({"input": input_text})
        
        self.cognitive_trace.append((CognitiveState.PERCEIVING, perception_result))
        
        return perception_result, perceptual_input
    
    def _attend(self, perception: str, input_text: str) -> AttentionFocus:
        """Attention: Focus on most relevant aspect"""
        
        self.cognitive_trace.append((CognitiveState.ATTENDING, "Directing attention"))
        
        # Determine what to focus on
        attention_prompt = ChatPromptTemplate.from_messages([
            ("system", """Determine attentional focus for this input.
Choose: goal, problem, context, memory, or action

Respond with just the focus word."""),
            ("user", "Input: {input}\nPerception: {perception}\nFocus:")
        ])
        
        chain = attention_prompt | self.llm | StrOutputParser()
        focus_str = chain.invoke({
            "input": input_text,
            "perception": perception
        }).strip().lower()
        
        # Map to enum
        focus_map = {
            "goal": AttentionFocus.GOAL,
            "problem": AttentionFocus.PROBLEM,
            "context": AttentionFocus.CONTEXT,
            "memory": AttentionFocus.MEMORY,
            "action": AttentionFocus.ACTION
        }
        
        self.attention_focus = focus_map.get(focus_str, AttentionFocus.CONTEXT)
        self.cognitive_trace.append((CognitiveState.ATTENDING, f"Focus: {self.attention_focus.value}"))
        
        return self.attention_focus
    
    def _encode_working_memory(self, content: str, relevance: float, source: str):
        """Encode information into working memory"""
        
        item = WorkingMemoryItem(
            content=content,
            relevance=relevance,
            timestamp=datetime.now(),
            source=source
        )
        
        self.working_memory.append(item)
        self.cognitive_trace.append((CognitiveState.ENCODING, f"Stored: {content[:50]}"))
    
    def _reason(self, input_text: str, perception: str) -> ReasoningResult:
        """Reasoning: Logical inference and problem-solving"""
        
        self.cognitive_trace.append((CognitiveState.REASONING, "Applying reasoning"))
        
        # Recall relevant long-term memories
        recalled_memories = self.long_term_memory.recall(input_text)
        
        # Get working memory context
        wm_context = "\n".join([
            f"- {item.content}"
            for item in sorted(self.working_memory, key=lambda x: x.relevance, reverse=True)
        ])
        
        # Build reasoning prompt
        memory_context = "\n".join([
            f"[{mem_type}] {mem_content}"
            for mem_type, mem_content in recalled_memories
        ]) if recalled_memories else "No relevant memories"
        
        reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are reasoning about a situation step-by-step.

Working Memory Context:
{working_memory}

Long-Term Memory:
{long_term_memory}

Provide:
1. Step-by-step reasoning
2. Logical conclusion
3. Confidence level (0.0-1.0)

Format:
Steps: [numbered list]
Conclusion: [statement]
Confidence: [0.0-1.0]"""),
            ("user", "Situation: {input}\nPerception: {perception}\n\nReason:")
        ])
        
        chain = reasoning_prompt | self.llm | StrOutputParser()
        reasoning_output = chain.invoke({
            "input": input_text,
            "perception": perception,
            "working_memory": wm_context or "Empty",
            "long_term_memory": memory_context
        })
        
        # Parse reasoning output (simplified)
        lines = reasoning_output.split('\n')
        steps = []
        conclusion = ""
        confidence = 0.7
        
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '-')):
                steps.append(line.strip())
            elif 'conclusion:' in line.lower():
                conclusion = line.split(':', 1)[1].strip()
            elif 'confidence:' in line.lower():
                try:
                    conf_str = line.split(':', 1)[1].strip()
                    confidence = float(conf_str)
                except:
                    pass
        
        if not conclusion:
            conclusion = reasoning_output.split('\n')[0]
        
        result = ReasoningResult(
            conclusion=conclusion,
            reasoning_steps=steps,
            confidence=confidence,
            used_memories=[m[1] for m in recalled_memories]
        )
        
        self.cognitive_trace.append((CognitiveState.REASONING, f"Conclusion: {conclusion[:50]}"))
        
        # Store conclusion in working memory
        self._encode_working_memory(conclusion, relevance=0.9, source="reasoning")
        
        return result
    
    def _plan(self, reasoning: ReasoningResult, input_text: str) -> ActionPlan:
        """Planning: Create action plan to achieve goal"""
        
        self.cognitive_trace.append((CognitiveState.PLANNING, "Creating action plan"))
        
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """Create an action plan based on the reasoning.

Provide:
1. Clear goal
2. Ordered steps to achieve it
3. Expected outcome

Format:
Goal: [goal]
Steps:
1. [step]
2. [step]
...
Expected: [outcome]"""),
            ("user", "Situation: {input}\nConclusion: {conclusion}\n\nPlan:")
        ])
        
        chain = planning_prompt | self.llm | StrOutputParser()
        plan_output = chain.invoke({
            "input": input_text,
            "conclusion": reasoning.conclusion
        })
        
        # Parse plan
        lines = plan_output.split('\n')
        goal = ""
        steps = []
        expected = ""
        
        for line in lines:
            if 'goal:' in line.lower():
                goal = line.split(':', 1)[1].strip()
            elif line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '-')):
                steps.append(line.strip())
            elif 'expected:' in line.lower():
                expected = line.split(':', 1)[1].strip()
        
        if not goal:
            goal = input_text
        if not expected:
            expected = reasoning.conclusion
        
        plan = ActionPlan(
            goal=goal,
            steps=steps if steps else ["Execute reasoning conclusion"],
            expected_outcome=expected,
            confidence=reasoning.confidence
        )
        
        self.current_goal = goal
        self.cognitive_trace.append((CognitiveState.PLANNING, f"Goal: {goal[:50]}"))
        
        return plan
    
    def _act(self, plan: ActionPlan, reasoning: ReasoningResult) -> str:
        """Action: Select and execute action"""
        
        self.cognitive_trace.append((CognitiveState.ACTING, "Executing action"))
        
        # Generate response based on plan and reasoning
        action_prompt = ChatPromptTemplate.from_messages([
            ("system", """Based on the plan and reasoning, provide a helpful response.

Be clear, actionable, and directly address the user's need."""),
            ("user", """Goal: {goal}
Conclusion: {conclusion}
Plan: {plan}

Provide response:""")
        ])
        
        chain = action_prompt | self.llm | StrOutputParser()
        action = chain.invoke({
            "goal": plan.goal,
            "conclusion": reasoning.conclusion,
            "plan": "\n".join(plan.steps)
        })
        
        self.cognitive_trace.append((CognitiveState.ACTING, f"Action: {action[:50]}"))
        
        # Store action in long-term memory as experience
        self.long_term_memory.add_experience(
            f"Goal: {plan.goal} | Action: {action[:100]}"
        )
        
        return action
    
    def _reflect(self, action: str, reasoning: ReasoningResult) -> str:
        """Reflection: Evaluate outcome and learn"""
        
        self.cognitive_trace.append((CognitiveState.REFLECTING, "Reflecting on action"))
        
        reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """Reflect on the action taken.

Consider:
1. Was it appropriate?
2. What was learned?
3. How to improve?

Be brief (2-3 sentences)."""),
            ("user", "Action: {action}\nConfidence: {confidence}\n\nReflection:")
        ])
        
        chain = reflection_prompt | self.llm | StrOutputParser()
        reflection = chain.invoke({
            "action": action,
            "confidence": reasoning.confidence
        })
        
        self.cognitive_trace.append((CognitiveState.REFLECTING, reflection))
        
        # Store reflection as learning
        self.long_term_memory.add_experience(f"Learned: {reflection}")
        
        return reflection
    
    def process(self, input_text: str) -> CognitiveResult:
        """Process input through complete cognitive cycle"""
        
        start_time = time.time()
        self.cognitive_trace = []
        
        # 1. Perception
        perception, perceptual_input = self._perceive(input_text)
        
        # 2. Attention
        attention_focus = self._attend(perception, input_text)
        
        # 3. Encoding (store perception in working memory)
        self._encode_working_memory(perception, relevance=0.8, source="perception")
        
        # 4. Reasoning
        reasoning = self._reason(input_text, perception)
        
        # 5. Planning
        plan = self._plan(reasoning, input_text)
        
        # 6. Action
        action = self._act(plan, reasoning)
        
        # 7. Reflection
        reflection = self._reflect(action, reasoning)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return CognitiveResult(
            input=input_text,
            perception=perception,
            attention_focus=attention_focus,
            reasoning=reasoning,
            plan=plan,
            action=action,
            reflection=reflection,
            cognitive_trace=self.cognitive_trace,
            execution_time_ms=execution_time_ms
        )


def demonstrate_cognitive_architecture():
    """Demonstrate Cognitive Architecture pattern"""
    
    print("=" * 80)
    print("PATTERN 055: COGNITIVE ARCHITECTURE DEMONSTRATION")
    print("=" * 80)
    print("\nHuman-like cognitive processing with perception, reasoning, and action\n")
    
    # Create cognitive agent
    agent = CognitiveAgent(working_memory_capacity=7, temperature=0.7)
    
    # Add some initial knowledge
    agent.long_term_memory.add_fact("Python is a programming language")
    agent.long_term_memory.add_fact("Lists in Python are mutable")
    agent.long_term_memory.add_skill("Able to explain programming concepts")
    agent.long_term_memory.add_goal("Help users learn effectively")
    
    # Test 1: Simple query
    print("\n" + "=" * 80)
    print("TEST 1: Simple Query Processing")
    print("=" * 80)
    
    query1 = "How do I reverse a list in Python?"
    
    print(f"\n💭 Input: {query1}")
    result1 = agent.process(query1)
    
    print(f"\n🔍 Cognitive Processing Trace:")
    for state, description in result1.cognitive_trace[:5]:  # Show first 5
        print(f"   [{state.value}] {description[:80]}")
    
    print(f"\n👁️  Perception: {result1.perception}")
    print(f"🎯 Attention Focus: {result1.attention_focus.value}")
    print(f"🧠 Reasoning Conclusion: {result1.reasoning.conclusion}")
    print(f"📋 Plan ({len(result1.plan.steps)} steps): {result1.plan.goal}")
    print(f"⚡ Action: {result1.action[:200]}...")
    print(f"🤔 Reflection: {result1.reflection}")
    print(f"⏱️  Execution Time: {result1.execution_time_ms:.1f}ms")
    
    # Test 2: Complex reasoning task
    print("\n" + "=" * 80)
    print("TEST 2: Complex Reasoning Task")
    print("=" * 80)
    
    query2 = "I need to build a web app but don't know where to start. What should I do?"
    
    print(f"\n💭 Input: {query2}")
    result2 = agent.process(query2)
    
    print(f"\n🔍 Full Cognitive Cycle:")
    print(f"   Perception: {result2.perception[:80]}...")
    print(f"   Attention: {result2.attention_focus.value}")
    print(f"   Reasoning Steps:")
    for i, step in enumerate(result2.reasoning.steps[:3], 1):
        print(f"      {i}. {step}")
    print(f"   Conclusion: {result2.reasoning.conclusion}")
    print(f"   Plan Goal: {result2.plan.goal}")
    print(f"   Action Plan:")
    for i, step in enumerate(result2.plan.steps[:4], 1):
        print(f"      {i}. {step}")
    
    print(f"\n💬 Final Response: {result2.action[:250]}...")
    
    # Test 3: Memory integration
    print("\n" + "=" * 80)
    print("TEST 3: Memory Integration")
    print("=" * 80)
    
    # First query establishes context
    query3a = "What are decorators in Python?"
    print(f"\n💭 Query 1: {query3a}")
    result3a = agent.process(query3a)
    print(f"   Response: {result3a.action[:150]}...")
    
    # Second query builds on previous
    query3b = "Can you give me a practical example?"
    print(f"\n💭 Query 2: {query3b}")
    result3b = agent.process(query3b)
    
    print(f"\n📚 Working Memory State ({len(agent.working_memory)} items):")
    for item in list(agent.working_memory)[-3:]:
        print(f"   - {str(item)[:60]}...")
    
    print(f"\n💬 Contextual Response: {result3b.action[:200]}...")
    
    # Show memory usage
    if result3b.reasoning.used_memories:
        print(f"\n🧠 Long-Term Memories Recalled:")
        for mem in result3b.reasoning.used_memories[:2]:
            print(f"   - {mem[:70]}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("COGNITIVE ARCHITECTURE PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Human-Like Processing: Mirrors human cognitive stages
2. Transparency: Clear cognitive trace for debugging
3. Context Awareness: Working and long-term memory integration
4. Adaptive Reasoning: Adjusts to different input types
5. Learning: Reflection enables continuous improvement

Cognitive Components:
1. Perception: Input processing and interpretation
2. Attention: Selective focus on relevant information
3. Working Memory: Limited-capacity temporary storage (7±2 items)
4. Long-Term Memory: Persistent knowledge storage
5. Reasoning: Logical inference and problem-solving
6. Planning: Goal-directed action sequencing
7. Action: Response generation and execution
8. Reflection: Outcome evaluation and learning

Memory Systems:
- Working Memory: Temporary, limited capacity, high relevance
- Long-Term Memory: Permanent, unlimited, organized by type
  * Facts: Declarative knowledge
  * Experiences: Episodic memories
  * Skills: Procedural knowledge
  * Goals: Long-term objectives

Attention Mechanism:
- Goal-focused: Primary objective attention
- Problem-focused: Current challenge attention
- Context-focused: Environmental awareness
- Memory-focused: Past experience recall
- Action-focused: Next step planning

Processing Stages:
1. Perceive → Extract features and intent
2. Attend → Focus on relevant aspects
3. Encode → Store in working memory
4. Reason → Logical inference
5. Plan → Create action sequence
6. Act → Execute plan
7. Reflect → Learn from outcome

Use Cases:
- Intelligent tutoring systems
- Personal assistant agents
- Complex decision support
- Game AI characters
- Research and analysis agents
- Autonomous problem-solving

Best Practices:
1. Limit working memory capacity (realistic)
2. Prioritize by relevance scores
3. Regular reflection for learning
4. Clear cognitive traces for transparency
5. Balance depth vs speed
6. Maintain long-term memory
7. Adaptive attention mechanisms

Production Considerations:
- Memory persistence (database)
- Working memory cleanup
- Long-term memory search (embeddings)
- Cognitive trace logging
- Performance optimization
- State management
- Error recovery

Comparison with Related Patterns:
- vs. ReAct: Full cognitive cycle vs action-observation
- vs. Chain-of-Thought: Multi-stage vs single reasoning
- vs. Memory Pattern: Integrated architecture vs storage
- vs. Agent: Cognitive model vs tool use

The Cognitive Architecture pattern provides sophisticated, human-like
information processing that enables contextual understanding, adaptive
reasoning, and continuous learning through reflection.
""")


if __name__ == "__main__":
    demonstrate_cognitive_architecture()
