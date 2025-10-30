"""
Pattern 018: Cooperative Multi-Agent

Description:
    The Cooperative Multi-Agent pattern features agents working together toward shared
    goals through collaboration, communication, and coordination. Agents share information,
    coordinate actions, and help each other to achieve collective objectives more
    effectively than they could individually.

Components:
    - Cooperative Agents: Agents designed to collaborate
    - Shared Goal System: Common objectives all agents work toward
    - Communication Protocol: Message passing and information sharing
    - Coordination Mechanism: Synchronizing agent actions
    - Resource Sharing: Pooling knowledge and capabilities

Use Cases:
    - Complex collaborative tasks
    - Distributed problem-solving
    - Team-based projects
    - Multi-perspective analysis

LangChain Implementation:
    Uses multiple LLM agents with shared context, implements message passing
    between agents, maintains shared knowledge base, and coordinates actions
    through explicit communication protocols.

Key Features:
    - Shared goal alignment
    - Inter-agent communication
    - Knowledge sharing
    - Coordinated action planning
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


class MessageType(Enum):
    """Types of messages agents can exchange."""
    INFORMATION = "information"
    REQUEST = "request"
    OFFER = "offer"
    COORDINATION = "coordination"
    RESULT = "result"


@dataclass
class Message:
    """Message exchanged between agents."""
    from_agent: str
    to_agent: Optional[str]  # None means broadcast to all
    message_type: MessageType
    content: str
    context: Optional[Dict[str, Any]] = None


@dataclass
class SharedKnowledge:
    """Knowledge base shared among cooperative agents."""
    facts: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)


@dataclass
class CooperativeTask:
    """Task for cooperative agents to work on."""
    task_id: str
    description: str
    assigned_agents: List[str]
    result: Optional[str] = None
    completed: bool = False


class CooperativeAgent:
    """
    An agent designed for cooperation and collaboration.
    """
    
    def __init__(
        self,
        agent_id: str,
        role: str,
        expertise: str,
        model: str = "gpt-3.5-turbo"
    ):
        """
        Initialize a cooperative agent.
        
        Args:
            agent_id: Unique identifier
            role: Agent's role in the team
            expertise: Agent's area of expertise
            model: LLM model to use
        """
        self.agent_id = agent_id
        self.role = role
        self.expertise = expertise
        self.llm = ChatOpenAI(model=model, temperature=0.6)
        
        # Collaboration state
        self.inbox: List[Message] = []
        self.outbox: List[Message] = []
        self.knowledge: List[str] = []
    
    def get_persona(self) -> str:
        """Get agent's cooperative persona."""
        return f"""You are {self.agent_id}, a cooperative team member.
Role: {self.role}
Expertise: {self.expertise}

You are collaborative, supportive, and focused on team success.
You actively share information, offer help, and coordinate with teammates."""
    
    def analyze_task(
        self,
        task: CooperativeTask,
        shared_knowledge: SharedKnowledge
    ) -> Dict[str, Any]:
        """
        Analyze a task from the agent's perspective.
        
        Args:
            task: Task to analyze
            shared_knowledge: Shared team knowledge
            
        Returns:
            Analysis with contributions and needs
        """
        knowledge_context = "\n".join([
            f"- {fact}" for fact in shared_knowledge.facts[:5]
        ]) if shared_knowledge.facts else "No shared knowledge yet"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_persona()),
            ("user", """Task: {task_description}

Team's shared knowledge:
{knowledge_context}

Analyze this task and provide:
1. What you can contribute (based on your expertise)
2. What help you need from teammates
3. Key considerations

Be specific and collaborative.""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        analysis = chain.invoke({
            "task_description": task.description,
            "knowledge_context": knowledge_context
        })
        
        return {"analysis": analysis.strip()}
    
    def contribute_to_task(
        self,
        task: CooperativeTask,
        team_messages: List[Message],
        shared_knowledge: SharedKnowledge
    ) -> str:
        """
        Contribute to solving the task.
        
        Args:
            task: Task to contribute to
            team_messages: Recent messages from team
            shared_knowledge: Shared knowledge base
            
        Returns:
            Agent's contribution
        """
        # Build context from team communication
        recent_messages = [
            f"{msg.from_agent} ({msg.message_type.value}): {msg.content}"
            for msg in team_messages[-5:]  # Last 5 messages
        ]
        message_context = "\n".join(recent_messages) if recent_messages else "No recent messages"
        
        knowledge_context = "\n".join([
            f"- {item}" for item in shared_knowledge.facts[:5]
        ]) if shared_knowledge.facts else "None yet"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_persona()),
            ("user", """Task: {task_description}

Recent team communication:
{message_context}

Shared knowledge:
{knowledge_context}

Provide your contribution to solving this task.
Build upon teammates' input and share your expertise.
Be specific and actionable (2-3 sentences).""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        contribution = chain.invoke({
            "task_description": task.description,
            "message_context": message_context,
            "knowledge_context": knowledge_context
        })
        
        return contribution.strip()
    
    def send_message(
        self,
        to_agent: Optional[str],
        message_type: MessageType,
        content: str
    ) -> Message:
        """
        Send a message to another agent or broadcast.
        
        Args:
            to_agent: Recipient agent ID (None for broadcast)
            message_type: Type of message
            content: Message content
            
        Returns:
            Created message
        """
        message = Message(
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type,
            content=content
        )
        self.outbox.append(message)
        return message
    
    def receive_message(self, message: Message):
        """Receive a message from another agent."""
        self.inbox.append(message)


class CooperativeTeam:
    """
    Manages a team of cooperative agents.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize cooperative team."""
        self.model = model
        self.agents: Dict[str, CooperativeAgent] = {}
        self.shared_knowledge = SharedKnowledge()
        self.message_log: List[Message] = []
        self.completed_tasks: List[CooperativeTask] = []
    
    def add_agent(
        self,
        agent_id: str,
        role: str,
        expertise: str
    ):
        """Add an agent to the team."""
        agent = CooperativeAgent(agent_id, role, expertise, self.model)
        self.agents[agent_id] = agent
    
    def broadcast_message(self, message: Message):
        """Broadcast a message to all agents."""
        for agent in self.agents.values():
            if agent.agent_id != message.from_agent:
                agent.receive_message(message)
        self.message_log.append(message)
    
    def direct_message(self, message: Message):
        """Send a message to a specific agent."""
        if message.to_agent and message.to_agent in self.agents:
            self.agents[message.to_agent].receive_message(message)
        self.message_log.append(message)
    
    def share_knowledge(self, knowledge_item: str, category: str = "facts"):
        """Add knowledge to shared knowledge base."""
        if category == "facts":
            self.shared_knowledge.facts.append(knowledge_item)
        elif category == "insights":
            self.shared_knowledge.insights.append(knowledge_item)
        elif category == "decisions":
            self.shared_knowledge.decisions.append(knowledge_item)
    
    def collaborate_on_task(
        self,
        task_description: str,
        assigned_agents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Agents collaborate to complete a task.
        
        Args:
            task_description: Description of the task
            assigned_agents: Agents to assign (None = all agents)
            
        Returns:
            Collaboration results
        """
        if assigned_agents is None:
            assigned_agents = list(self.agents.keys())
        
        task = CooperativeTask(
            task_id=f"task_{len(self.completed_tasks) + 1}",
            description=task_description,
            assigned_agents=assigned_agents
        )
        
        print(f"\n[Team] Starting collaborative task: {task_description}")
        print(f"[Team] Assigned: {', '.join(assigned_agents)}\n")
        
        # Phase 1: Individual analysis
        print("=" * 80)
        print("PHASE 1: Individual Analysis")
        print("=" * 80 + "\n")
        
        analyses = {}
        for agent_id in assigned_agents:
            agent = self.agents[agent_id]
            analysis = agent.analyze_task(task, self.shared_knowledge)
            analyses[agent_id] = analysis
            
            print(f"[{agent_id}] Analysis:")
            print(f"  {analysis['analysis'][:150]}...\n")
            
            # Share analysis with team
            msg = agent.send_message(
                None,  # Broadcast
                MessageType.INFORMATION,
                f"My analysis: {analysis['analysis']}"
            )
            self.broadcast_message(msg)
        
        # Phase 2: Coordination and planning
        print("=" * 80)
        print("PHASE 2: Coordination & Planning")
        print("=" * 80 + "\n")
        
        # First agent initiates coordination
        coordinator_id = assigned_agents[0]
        coordinator = self.agents[coordinator_id]
        
        coord_msg = coordinator.send_message(
            None,
            MessageType.COORDINATION,
            f"Let's coordinate our efforts. I'll focus on {coordinator.expertise}. Who can handle other aspects?"
        )
        self.broadcast_message(coord_msg)
        print(f"[{coordinator_id}] Coordination: {coord_msg.content}\n")
        
        # Other agents respond with offers
        for agent_id in assigned_agents[1:]:
            agent = self.agents[agent_id]
            offer_msg = agent.send_message(
                coordinator_id,
                MessageType.OFFER,
                f"I can contribute my {agent.expertise} expertise"
            )
            self.direct_message(offer_msg)
            print(f"[{agent_id}] Offer: {offer_msg.content}\n")
        
        # Phase 3: Collaborative execution
        print("=" * 80)
        print("PHASE 3: Collaborative Execution")
        print("=" * 80 + "\n")
        
        contributions = {}
        for agent_id in assigned_agents:
            agent = self.agents[agent_id]
            
            contribution = agent.contribute_to_task(
                task,
                self.message_log,
                self.shared_knowledge
            )
            contributions[agent_id] = contribution
            
            print(f"[{agent_id}] Contribution:")
            print(f"  {contribution}\n")
            
            # Share contribution
            result_msg = agent.send_message(
                None,
                MessageType.RESULT,
                contribution
            )
            self.broadcast_message(result_msg)
            
            # Add to shared knowledge
            self.share_knowledge(
                f"{agent_id}: {contribution}",
                "insights"
            )
        
        # Phase 4: Synthesis
        print("=" * 80)
        print("PHASE 4: Synthesis")
        print("=" * 80 + "\n")
        
        final_result = self._synthesize_contributions(
            task_description,
            contributions
        )
        
        task.result = final_result
        task.completed = True
        self.completed_tasks.append(task)
        
        print(f"[Team] Task completed successfully!\n")
        
        return {
            "task": task,
            "individual_analyses": analyses,
            "contributions": contributions,
            "final_result": final_result,
            "messages_exchanged": len(self.message_log),
            "shared_knowledge_items": len(self.shared_knowledge.facts) + len(self.shared_knowledge.insights)
        }
    
    def _synthesize_contributions(
        self,
        task_description: str,
        contributions: Dict[str, str]
    ) -> str:
        """Synthesize individual contributions into final result."""
        # Use a synthesis agent
        synthesizer = ChatOpenAI(model=self.model, temperature=0.5)
        
        contributions_text = "\n\n".join([
            f"{agent_id} ({self.agents[agent_id].expertise}):\n{contrib}"
            for agent_id, contrib in contributions.items()
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are synthesizing collaborative team outputs into a cohesive result."),
            ("user", """Task: {task_description}

Team Contributions:
{contributions_text}

Synthesize these contributions into a unified, coherent solution that:
1. Integrates all team members' expertise
2. Addresses the task completely
3. Highlights collaborative synergies
4. Provides actionable results""")
        ])
        
        chain = prompt | synthesizer | StrOutputParser()
        
        synthesis = chain.invoke({
            "task_description": task_description,
            "contributions_text": contributions_text
        })
        
        return synthesis.strip()
    
    def get_team_summary(self) -> Dict[str, Any]:
        """Get summary of team composition and performance."""
        return {
            "team_size": len(self.agents),
            "agents": [
                {
                    "id": agent.agent_id,
                    "role": agent.role,
                    "expertise": agent.expertise
                }
                for agent in self.agents.values()
            ],
            "tasks_completed": len(self.completed_tasks),
            "messages_exchanged": len(self.message_log),
            "shared_knowledge_items": len(self.shared_knowledge.facts) + len(self.shared_knowledge.insights)
        }


def demonstrate_cooperative_multi_agent():
    """Demonstrate the Cooperative Multi-Agent pattern."""
    
    print("=" * 80)
    print("COOPERATIVE MULTI-AGENT PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Test 1: Product development team
    print("\n" + "=" * 80)
    print("TEST 1: Product Development Team")
    print("=" * 80)
    
    team1 = CooperativeTeam()
    
    # Build diverse team
    team1.add_agent("Alice", "Product Manager", "Product Strategy")
    team1.add_agent("Bob", "Engineer", "Technical Implementation")
    team1.add_agent("Carol", "Designer", "User Experience")
    team1.add_agent("Dave", "Analyst", "Data Analysis")
    
    print("\nTeam Composition:")
    for agent_id, agent in team1.agents.items():
        print(f"  - {agent_id}: {agent.role} ({agent.expertise})")
    
    task1 = "Design and plan a new feature that helps users track their productivity goals"
    
    result1 = team1.collaborate_on_task(task1)
    
    print("=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(result1["final_result"])
    
    print("\n" + "-" * 80)
    print("COLLABORATION METRICS:")
    print("-" * 80)
    print(f"  Messages exchanged: {result1['messages_exchanged']}")
    print(f"  Knowledge items shared: {result1['shared_knowledge_items']}")
    print(f"  Agents participated: {len(result1['contributions'])}")
    
    # Test 2: Research team
    print("\n" + "=" * 80)
    print("TEST 2: Research Team Collaboration")
    print("=" * 80)
    
    team2 = CooperativeTeam()
    
    team2.add_agent("ResearcherA", "Data Scientist", "Machine Learning")
    team2.add_agent("ResearcherB", "Domain Expert", "Healthcare")
    team2.add_agent("ResearcherC", "Statistician", "Statistical Analysis")
    
    print("\nTeam Composition:")
    for agent_id, agent in team2.agents.items():
        print(f"  - {agent_id}: {agent.role} ({agent.expertise})")
    
    task2 = "Develop a methodology to predict patient readmission risk using ML"
    
    result2 = team2.collaborate_on_task(task2)
    
    print("=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(result2["final_result"])
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Cooperative Multi-Agent pattern demonstrates several key benefits:

1. **Synergy**: Combined expertise exceeds individual capabilities
2. **Knowledge Sharing**: Agents build on each other's contributions
3. **Coordinated Action**: Agents work together toward shared goals
4. **Communication**: Rich information exchange improves outcomes

Collaboration Phases:
1. **Individual Analysis**: Each agent analyzes from their perspective
2. **Coordination**: Agents plan how to work together
3. **Execution**: Agents contribute their expertise
4. **Synthesis**: Contributions integrated into final result

Communication Types:
- **Information**: Sharing facts and knowledge
- **Request**: Asking for help or information
- **Offer**: Offering assistance or expertise
- **Coordination**: Planning joint action
- **Result**: Sharing completed work

Use Cases:
- Complex multi-disciplinary projects
- Team-based problem-solving
- Distributed collaborative work
- Multi-perspective analysis
- Cross-functional initiatives

The pattern is particularly effective when:
- Tasks require diverse expertise
- Collaboration adds value over individual work
- Communication and coordination are feasible
- Shared goals align team members
- Synergies between specialties exist
""")


if __name__ == "__main__":
    demonstrate_cooperative_multi_agent()
