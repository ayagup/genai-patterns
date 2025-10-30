"""
Pattern 019: Society of Mind

Description:
    The Society of Mind pattern implements a cognitive architecture where multiple
    specialized sub-agents (each handling a specific aspect of cognition) work together
    to form a coherent intelligent system. Inspired by Marvin Minsky's theory, this
    pattern views intelligence as emerging from the interaction of many simple processes.

Components:
    - Perception Agents: Process and interpret inputs
    - Memory Agents: Store and retrieve information
    - Reasoning Agents: Logical thinking and analysis
    - Planning Agents: Goal setting and strategy
    - Execution Agents: Action taking and output generation
    - Meta-Cognitive Agent: Coordinates other agents

Use Cases:
    - Complex cognitive modeling
    - General AI systems
    - Multi-faceted problem-solving
    - Comprehensive decision-making systems

LangChain Implementation:
    Uses specialized LLM agents for different cognitive functions, implements
    inter-agent communication protocols, maintains shared mental state, and
    coordinates through a meta-cognitive controller.

Key Features:
    - Specialized cognitive sub-agents
    - Integrated cognitive architecture
    - Emergent intelligent behavior
    - Meta-cognitive coordination
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


class CognitiveFunction(Enum):
    """Types of cognitive functions."""
    PERCEPTION = "perception"
    MEMORY = "memory"
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTION = "execution"
    EMOTION = "emotion"


@dataclass
class MentalState:
    """Shared mental state of the society."""
    current_input: Optional[str] = None
    perceived_situation: Optional[str] = None
    relevant_memories: List[str] = field(default_factory=list)
    reasoning_output: Optional[str] = None
    plan: Optional[str] = None
    emotional_state: Optional[str] = None
    final_output: Optional[str] = None


@dataclass
class CognitiveOutput:
    """Output from a cognitive agent."""
    function: CognitiveFunction
    agent_id: str
    output: str
    confidence: float


class CognitiveAgent:
    """
    A specialized cognitive agent handling one aspect of cognition.
    """
    
    def __init__(
        self,
        agent_id: str,
        function: CognitiveFunction,
        description: str,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize a cognitive agent.
        
        Args:
            agent_id: Unique identifier
            function: Cognitive function this agent handles
            description: Description of agent's role
            model: LLM model to use
        """
        self.agent_id = agent_id
        self.function = function
        self.description = description
        self.llm = ChatOpenAI(model=model, temperature=0.6)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this cognitive agent."""
        prompts = {
            CognitiveFunction.PERCEPTION: f"""You are the Perception Agent.
Role: {self.description}
You interpret inputs, identify key elements, and understand the situation.
Provide clear, objective perception of what you observe.""",
            
            CognitiveFunction.MEMORY: f"""You are the Memory Agent.
Role: {self.description}
You recall relevant past experiences, facts, and patterns.
Provide relevant memories that can inform current processing.""",
            
            CognitiveFunction.REASONING: f"""You are the Reasoning Agent.
Role: {self.description}
You analyze information logically, identify patterns, and draw conclusions.
Provide sound logical reasoning and analysis.""",
            
            CognitiveFunction.PLANNING: f"""You are the Planning Agent.
Role: {self.description}
You create strategies, set goals, and plan actions.
Provide clear, actionable plans.""",
            
            CognitiveFunction.EXECUTION: f"""You are the Execution Agent.
Role: {self.description}
You take plans and create concrete outputs and actions.
Provide clear, actionable outputs.""",
            
            CognitiveFunction.EMOTION: f"""You are the Emotion Agent.
Role: {self.description}
You assess emotional context and implications.
Provide emotional intelligence and empathy."""
        }
        return prompts[self.function]
    
    def process(
        self,
        mental_state: MentalState,
        context: Optional[str] = None
    ) -> CognitiveOutput:
        """
        Process information according to cognitive function.
        
        Args:
            mental_state: Current mental state
            context: Optional additional context
            
        Returns:
            Cognitive output
        """
        # Build context based on what's available in mental state
        state_context = self._build_context(mental_state)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            ("user", """Current Mental State:
{state_context}

{additional_context}

Perform your cognitive function and provide output (2-3 sentences):""")
        ])
        
        additional = f"Additional Context: {context}" if context else ""
        
        chain = prompt | self.llm | StrOutputParser()
        
        output = chain.invoke({
            "state_context": state_context,
            "additional_context": additional
        })
        
        return CognitiveOutput(
            function=self.function,
            agent_id=self.agent_id,
            output=output.strip(),
            confidence=0.8  # Simplified confidence
        )
    
    def _build_context(self, mental_state: MentalState) -> str:
        """Build context string from mental state."""
        parts = []
        
        if mental_state.current_input:
            parts.append(f"Input: {mental_state.current_input}")
        if mental_state.perceived_situation:
            parts.append(f"Perception: {mental_state.perceived_situation}")
        if mental_state.relevant_memories:
            parts.append(f"Memories: {'; '.join(mental_state.relevant_memories[:3])}")
        if mental_state.reasoning_output:
            parts.append(f"Reasoning: {mental_state.reasoning_output}")
        if mental_state.plan:
            parts.append(f"Plan: {mental_state.plan}")
        if mental_state.emotional_state:
            parts.append(f"Emotional Context: {mental_state.emotional_state}")
        
        return "\n".join(parts) if parts else "No information yet"


class MetaCognitiveController:
    """
    Meta-cognitive controller that coordinates cognitive agents.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize meta-cognitive controller."""
        self.llm = ChatOpenAI(model=model, temperature=0.5)
    
    def decide_next_step(
        self,
        mental_state: MentalState,
        completed_functions: List[CognitiveFunction]
    ) -> Optional[CognitiveFunction]:
        """
        Decide which cognitive function to invoke next.
        
        Args:
            mental_state: Current mental state
            completed_functions: Functions already completed
            
        Returns:
            Next cognitive function to invoke, or None if done
        """
        # Standard cognitive pipeline
        pipeline = [
            CognitiveFunction.PERCEPTION,
            CognitiveFunction.MEMORY,
            CognitiveFunction.EMOTION,
            CognitiveFunction.REASONING,
            CognitiveFunction.PLANNING,
            CognitiveFunction.EXECUTION
        ]
        
        for function in pipeline:
            if function not in completed_functions:
                return function
        
        return None  # All done
    
    def integrate_outputs(
        self,
        mental_state: MentalState,
        outputs: List[CognitiveOutput]
    ) -> str:
        """
        Integrate outputs from all cognitive agents.
        
        Args:
            mental_state: Final mental state
            outputs: All cognitive outputs
            
        Returns:
            Integrated final response
        """
        outputs_text = "\n\n".join([
            f"{out.function.value.upper()}:\n{out.output}"
            for out in outputs
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are the Meta-Cognitive Integrator.
You synthesize outputs from specialized cognitive agents into a coherent response."""),
            ("user", """Original Input: {original_input}

Cognitive Processing Results:
{outputs_text}

Synthesize these cognitive outputs into a unified, coherent final response that:
1. Addresses the original input completely
2. Integrates insights from all cognitive functions
3. Is clear and actionable
4. Shows balanced consideration of all aspects""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        integration = chain.invoke({
            "original_input": mental_state.current_input,
            "outputs_text": outputs_text
        })
        
        return integration.strip()


class SocietyOfMind:
    """
    Implements the Society of Mind cognitive architecture.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize the society of mind."""
        self.model = model
        self.cognitive_agents: Dict[CognitiveFunction, CognitiveAgent] = {}
        self.meta_controller = MetaCognitiveController(model)
        self.mental_state = MentalState()
        
        # Initialize cognitive agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all cognitive agents."""
        agents_config = {
            CognitiveFunction.PERCEPTION: ("Perceptor", "Interprets and understands inputs"),
            CognitiveFunction.MEMORY: ("Memorist", "Recalls relevant past information"),
            CognitiveFunction.REASONING: ("Reasoner", "Analyzes and draws conclusions"),
            CognitiveFunction.PLANNING: ("Planner", "Creates strategies and plans"),
            CognitiveFunction.EXECUTION: ("Executor", "Generates concrete outputs"),
            CognitiveFunction.EMOTION: ("Empath", "Assesses emotional context")
        }
        
        for function, (agent_id, description) in agents_config.items():
            self.cognitive_agents[function] = CognitiveAgent(
                agent_id,
                function,
                description,
                self.model
            )
    
    def process(self, input_text: str) -> Dict[str, Any]:
        """
        Process input through the entire society of mind.
        
        Args:
            input_text: Input to process
            
        Returns:
            Processing results with all cognitive outputs
        """
        print(f"\n[Society of Mind] Processing: {input_text}\n")
        
        # Initialize mental state
        self.mental_state = MentalState(current_input=input_text)
        
        completed_functions: List[CognitiveFunction] = []
        cognitive_outputs: List[CognitiveOutput] = []
        
        # Cognitive processing pipeline
        while True:
            # Meta-controller decides next step
            next_function = self.meta_controller.decide_next_step(
                self.mental_state,
                completed_functions
            )
            
            if next_function is None:
                break  # Processing complete
            
            # Invoke cognitive agent
            agent = self.cognitive_agents[next_function]
            
            print(f"[{agent.agent_id}] Processing ({next_function.value})...")
            
            output = agent.process(self.mental_state)
            cognitive_outputs.append(output)
            completed_functions.append(next_function)
            
            print(f"[{agent.agent_id}] Output: {output.output}\n")
            
            # Update mental state
            self._update_mental_state(output)
        
        # Integrate all outputs
        print("[Meta-Controller] Integrating cognitive outputs...\n")
        
        final_output = self.meta_controller.integrate_outputs(
            self.mental_state,
            cognitive_outputs
        )
        
        self.mental_state.final_output = final_output
        
        return {
            "input": input_text,
            "cognitive_outputs": cognitive_outputs,
            "mental_state": self.mental_state,
            "final_output": final_output
        }
    
    def _update_mental_state(self, output: CognitiveOutput):
        """Update mental state based on cognitive output."""
        if output.function == CognitiveFunction.PERCEPTION:
            self.mental_state.perceived_situation = output.output
        elif output.function == CognitiveFunction.MEMORY:
            self.mental_state.relevant_memories = [output.output]
        elif output.function == CognitiveFunction.REASONING:
            self.mental_state.reasoning_output = output.output
        elif output.function == CognitiveFunction.PLANNING:
            self.mental_state.plan = output.output
        elif output.function == CognitiveFunction.EMOTION:
            self.mental_state.emotional_state = output.output
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get summary of the cognitive architecture."""
        return {
            "cognitive_agents": [
                {
                    "id": agent.agent_id,
                    "function": agent.function.value,
                    "description": agent.description
                }
                for agent in self.cognitive_agents.values()
            ],
            "processing_pipeline": [f.value for f in CognitiveFunction]
        }


def demonstrate_society_of_mind():
    """Demonstrate the Society of Mind pattern."""
    
    print("=" * 80)
    print("SOCIETY OF MIND PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Initialize the society
    society = SocietyOfMind()
    
    # Show architecture
    print("\n" + "=" * 80)
    print("COGNITIVE ARCHITECTURE")
    print("=" * 80)
    
    arch = society.get_architecture_summary()
    print("\nCognitive Agents:")
    for agent in arch["cognitive_agents"]:
        print(f"  - {agent['id']} ({agent['function']}): {agent['description']}")
    
    # Test 1: Decision-making scenario
    print("\n" + "=" * 80)
    print("TEST 1: Complex Decision-Making")
    print("=" * 80)
    
    input1 = "I'm considering changing careers from engineering to product management. What should I consider?"
    
    result1 = society.process(input1)
    
    print("=" * 80)
    print("COGNITIVE PROCESSING BREAKDOWN:")
    print("=" * 80)
    for output in result1["cognitive_outputs"]:
        print(f"\n{output.function.value.upper()}:")
        print(f"  {output.output}")
    
    print("\n" + "=" * 80)
    print("FINAL INTEGRATED RESPONSE:")
    print("=" * 80)
    print(result1["final_output"])
    
    # Test 2: Creative problem-solving
    print("\n" + "=" * 80)
    print("TEST 2: Creative Problem-Solving")
    print("=" * 80)
    
    input2 = "How can I make studying more engaging and effective for students who find it boring?"
    
    result2 = society.process(input2)
    
    print("=" * 80)
    print("FINAL INTEGRATED RESPONSE:")
    print("=" * 80)
    print(result2["final_output"])
    
    print("\n" + "-" * 80)
    print("MENTAL STATE SNAPSHOT:")
    print("-" * 80)
    state = result2["mental_state"]
    print(f"Perception: {state.perceived_situation[:80]}...")
    print(f"Emotion: {state.emotional_state[:80]}...")
    print(f"Plan: {state.plan[:80]}...")
    
    # Test 3: Interpersonal situation
    print("\n" + "=" * 80)
    print("TEST 3: Interpersonal Situation")
    print("=" * 80)
    
    input3 = "My teammate is often late to meetings and it's affecting team morale. How should I address this?"
    
    result3 = society.process(input3)
    
    print("=" * 80)
    print("COGNITIVE PROCESSING BREAKDOWN:")
    print("=" * 80)
    for output in result3["cognitive_outputs"]:
        print(f"\n{output.function.value.upper()}:")
        print(f"  {output.output}")
    
    print("\n" + "=" * 80)
    print("FINAL INTEGRATED RESPONSE:")
    print("=" * 80)
    print(result3["final_output"])
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Society of Mind pattern demonstrates several key benefits:

1. **Cognitive Specialization**: Each agent handles one aspect of cognition
2. **Emergent Intelligence**: Complex behavior from simple component interactions
3. **Comprehensive Processing**: All aspects of problem considered
4. **Integrated Output**: Coherent response from diverse cognitive functions

Cognitive Architecture:
- **Perception**: Interprets and understands inputs
- **Memory**: Recalls relevant past information
- **Emotion**: Assesses emotional context and empathy
- **Reasoning**: Analyzes logically and draws conclusions
- **Planning**: Creates strategies and action plans
- **Execution**: Generates concrete outputs

Processing Flow:
1. Input received and stored in mental state
2. Each cognitive agent processes in sequence
3. Mental state updated after each agent
4. Meta-controller integrates all outputs
5. Final coherent response generated

Use Cases:
- Complex decision-making systems
- General AI architectures
- Multi-faceted problem-solving
- Cognitive modeling and simulation
- Comprehensive analysis tasks

The pattern is particularly effective when:
- Problems require multiple perspectives
- Cognitive diversity adds value
- Comprehensive analysis is needed
- Emotional intelligence matters
- Integration of multiple factors is important

This architecture models how human intelligence emerges from the interaction
of many specialized mental processes, creating a more human-like AI system.
""")


if __name__ == "__main__":
    demonstrate_society_of_mind()
