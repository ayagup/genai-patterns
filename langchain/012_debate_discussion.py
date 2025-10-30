"""
Pattern 012: Debate/Discussion (Multi-Agent)

Description:
    Multiple agents engage in debate or discussion to reach better conclusions.
    Each agent represents different perspectives, argues for their position,
    and the system synthesizes the best answer from the debate. This reduces
    bias and improves reasoning quality through adversarial collaboration.

Key Concepts:
    - Multiple Perspectives: Each agent takes different viewpoint
    - Argument Generation: Agents present reasoned arguments
    - Counter-Arguments: Agents respond to each other's points
    - Synthesis: Final answer combines best insights from debate
    - Moderation: Optional moderator guides discussion

Benefits:
    - Reduced individual bias
    - More thorough exploration of problem space
    - Error correction through peer review
    - Higher quality conclusions

Use Cases:
    - Complex decision-making
    - Ethical dilemmas
    - Strategy formulation
    - Content review and validation

LangChain Implementation:
    Multiple LLM instances with different personas engaging in structured debate.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


class AgentRole(Enum):
    """Roles agents can take in debate."""
    PROPONENT = "proponent"
    OPPONENT = "opponent"
    SKEPTIC = "skeptic"
    PRAGMATIST = "pragmatist"
    MODERATOR = "moderator"


@dataclass
class DebateMessage:
    """Represents a message in the debate."""
    round: int
    agent_id: str
    role: AgentRole
    content: str
    responding_to: Optional[str] = None


@dataclass
class DebateTranscript:
    """Complete transcript of a debate."""
    question: str
    messages: List[DebateMessage] = field(default_factory=list)
    final_synthesis: str = ""
    
    def add_message(self, message: DebateMessage):
        """Add a message to the transcript."""
        self.messages.append(message)
    
    def get_round_messages(self, round_num: int) -> List[DebateMessage]:
        """Get all messages from a specific round."""
        return [msg for msg in self.messages if msg.round == round_num]


class DebateAgent:
    """Individual agent participating in debate."""
    
    def __init__(self, agent_id: str, role: AgentRole, 
                 model_name: str = "gpt-3.5-turbo"):
        """
        Initialize a debate agent.
        
        Args:
            agent_id: Unique identifier for the agent
            role: Role the agent plays in debate
            model_name: Name of the OpenAI model
        """
        self.agent_id = agent_id
        self.role = role
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,  # Some creativity for diverse perspectives
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Role-specific system prompts
        self.system_prompts = {
            AgentRole.PROPONENT: "You are advocating FOR the position. Present strong arguments supporting it.",
            AgentRole.OPPONENT: "You are arguing AGAINST the position. Present strong counterarguments.",
            AgentRole.SKEPTIC: "You are a critical skeptic. Question assumptions and identify weaknesses.",
            AgentRole.PRAGMATIST: "You focus on practical implications and real-world feasibility."
        }
    
    def generate_opening_statement(self, question: str) -> str:
        """Generate opening statement for the debate."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{self.system_prompts.get(self.role, 'You are a thoughtful debater.')}
            
This is your opening statement. Present your main position clearly and concisely."""),
            ("human", "Question: {question}\n\nYour opening statement:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question})
    
    def respond_to_arguments(self, question: str, previous_messages: List[DebateMessage]) -> str:
        """Generate response to previous arguments."""
        # Build context from previous messages
        context_parts = []
        for msg in previous_messages:
            context_parts.append(f"{msg.agent_id} ({msg.role.value}): {msg.content}")
        
        context = "\n\n".join(context_parts)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""{self.system_prompts.get(self.role, 'You are a thoughtful debater.')}

Respond to the previous arguments. Address key points, present counter-arguments where appropriate,
and strengthen your position."""),
            ("human", """Question: {question}

Previous Arguments:
{context}

Your response:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "question": question,
            "context": context
        })


class DebateSystem:
    """System for managing multi-agent debate."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", num_rounds: int = 2):
        """
        Initialize the debate system.
        
        Args:
            model_name: Name of the OpenAI model
            num_rounds: Number of debate rounds
        """
        self.model_name = model_name
        self.num_rounds = num_rounds
        self.moderator_llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def conduct_debate(self, question: str, 
                      agent_roles: Optional[List[AgentRole]] = None,
                      verbose: bool = True) -> DebateTranscript:
        """
        Conduct a structured debate on the question.
        
        Args:
            question: Question to debate
            agent_roles: Roles for agents (default: proponent and opponent)
            verbose: Whether to print debate progress
            
        Returns:
            Complete debate transcript with synthesis
        """
        if agent_roles is None:
            agent_roles = [AgentRole.PROPONENT, AgentRole.OPPONENT]
        
        transcript = DebateTranscript(question=question)
        
        if verbose:
            print(f"\nQuestion: {question}\n")
            print(f"{'='*60}")
            print(f"MULTI-AGENT DEBATE")
            print(f"{'='*60}")
            print(f"Participants: {len(agent_roles)}")
            print(f"Rounds: {self.num_rounds}\n")
        
        # Create agents
        agents = []
        for i, role in enumerate(agent_roles):
            agent = DebateAgent(
                agent_id=f"Agent_{i+1}",
                role=role,
                model_name=self.model_name
            )
            agents.append(agent)
            if verbose:
                print(f"  {agent.agent_id}: {role.value}")
        
        print()
        
        # Round 1: Opening statements
        if verbose:
            print(f"\n{'='*60}")
            print(f"ROUND 1: Opening Statements")
            print('='*60)
        
        for agent in agents:
            statement = agent.generate_opening_statement(question)
            
            message = DebateMessage(
                round=1,
                agent_id=agent.agent_id,
                role=agent.role,
                content=statement
            )
            transcript.add_message(message)
            
            if verbose:
                print(f"\n{agent.agent_id} ({agent.role.value}):")
                print(f"{statement}")
        
        # Subsequent rounds: Responses and rebuttals
        for round_num in range(2, self.num_rounds + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"ROUND {round_num}: Discussion")
                print('='*60)
            
            # Get previous round messages for context
            prev_messages = transcript.get_round_messages(round_num - 1)
            
            for agent in agents:
                # Agent responds to previous arguments
                response = agent.respond_to_arguments(question, prev_messages)
                
                message = DebateMessage(
                    round=round_num,
                    agent_id=agent.agent_id,
                    role=agent.role,
                    content=response
                )
                transcript.add_message(message)
                
                if verbose:
                    print(f"\n{agent.agent_id} ({agent.role.value}):")
                    print(f"{response}")
        
        # Synthesis: Combine insights from debate
        if verbose:
            print(f"\n{'='*60}")
            print("SYNTHESIS")
            print('='*60)
        
        synthesis = self._synthesize_debate(question, transcript)
        transcript.final_synthesis = synthesis
        
        if verbose:
            print(f"\n{synthesis}")
        
        return transcript
    
    def _synthesize_debate(self, question: str, transcript: DebateTranscript) -> str:
        """
        Synthesize the debate into a final answer.
        
        Args:
            question: Original question
            transcript: Complete debate transcript
            
        Returns:
            Synthesized answer
        """
        # Build debate summary
        debate_text = []
        for msg in transcript.messages:
            debate_text.append(f"[Round {msg.round}] {msg.agent_id} ({msg.role.value}):\n{msg.content}")
        
        debate_summary = "\n\n".join(debate_text)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a moderator synthesizing a debate. 
            
Review all arguments presented and create a balanced, thoughtful conclusion that:
1. Acknowledges strong points from all sides
2. Identifies areas of consensus
3. Notes remaining disagreements
4. Provides a nuanced final answer

Be objective and fair to all perspectives."""),
            ("human", """Question: {question}

Debate Transcript:
{debate}

Synthesis:""")
        ])
        
        chain = prompt | self.moderator_llm | StrOutputParser()
        synthesis = chain.invoke({
            "question": question,
            "debate": debate_summary
        })
        
        return synthesis


def demonstrate_debate_pattern():
    """Demonstrates the Debate/Discussion pattern."""
    
    print("=" * 80)
    print("PATTERN 012: Debate/Discussion (Multi-Agent)")
    print("=" * 80)
    print()
    print("Multi-Agent Debate improves reasoning through:")
    print("1. Multiple Perspectives: Different viewpoints represented")
    print("2. Adversarial Collaboration: Agents challenge each other")
    print("3. Iterative Refinement: Arguments evolve through rounds")
    print("4. Synthesis: Best insights combined into final answer")
    print()
    
    # Create debate system
    system = DebateSystem(num_rounds=2)
    
    # Test questions
    questions = [
        "Should artificial intelligence development be regulated by governments?",
        "Is remote work more productive than office work for software development?"
    ]
    
    for idx, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Debate {idx}")
        print('='*80)
        
        try:
            # Conduct debate with different roles
            roles = [AgentRole.PROPONENT, AgentRole.OPPONENT, AgentRole.SKEPTIC]
            
            transcript = system.conduct_debate(
                question=question,
                agent_roles=roles,
                verbose=True
            )
            
            print(f"\n\n{'='*80}")
            print("DEBATE SUMMARY")
            print('='*80)
            print(f"\nQuestion: {transcript.question}")
            print(f"Total Messages: {len(transcript.messages)}")
            print(f"Rounds: {system.num_rounds}")
            print(f"\nFinal Synthesis:\n{transcript.final_synthesis}")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("DEBATE/DISCUSSION PATTERN DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("1. Multiple Agent Roles: Proponent, Opponent, Skeptic")
    print("2. Structured Rounds: Opening statements followed by discussion")
    print("3. Contextual Responses: Agents respond to previous arguments")
    print("4. Moderator Synthesis: Balanced conclusion from all perspectives")
    print("5. Transcript Management: Complete debate history maintained")
    print()
    print("Advantages:")
    print("- Reduces individual bias")
    print("- More thorough problem exploration")
    print("- Error correction through peer review")
    print("- Higher quality conclusions")
    print()
    print("When to use Debate Pattern:")
    print("- Complex decision-making")
    print("- Ethical or controversial questions")
    print("- Strategy formulation")
    print("- Content validation and review")
    print()
    print("LangChain Components Used:")
    print("- Multiple ChatOpenAI instances: Different agents")
    print("- Role-based system prompts: Agent personas")
    print("- Structured debate management: Rounds and synthesis")
    print("- Context building: Previous arguments inform responses")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    demonstrate_debate_pattern()
