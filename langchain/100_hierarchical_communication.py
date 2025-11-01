"""
Pattern 100: Hierarchical Communication

Description:
    The Hierarchical Communication pattern establishes structured communication channels
    between agents organized in a hierarchical structure. This pattern is essential for
    large-scale multi-agent systems where flat communication would be inefficient or
    chaotic. It defines clear lines of authority, responsibility, and communication flow.
    
    Hierarchical structures enable scalable coordination by organizing agents into levels:
    executives/managers at higher levels provide strategic direction and oversight, while
    workers at lower levels handle operational tasks. Communication flows both top-down
    (commands, directives, goals) and bottom-up (reports, escalations, feedback).
    
    This pattern supports various organizational structures including strict hierarchies,
    matrix organizations, and hybrid models. It handles delegation, escalation, reporting,
    and cross-level communication while maintaining clear accountability and reducing
    communication overhead through proper routing and filtering.

Key Components:
    1. Hierarchical Levels: Organization into layers (executive, manager, worker)
    2. Authority Structure: Who can command whom
    3. Communication Channels: Allowed communication paths
    4. Delegation: Task assignment down the hierarchy
    5. Escalation: Problem raising up the hierarchy
    6. Reporting: Status updates to superiors
    7. Coordination: Cross-branch communication
    8. Information Filtering: Aggregation at each level

Organizational Structures:
    1. Strict Hierarchy: Tree structure, single reporting line
    2. Matrix Organization: Multiple reporting lines
    3. Flat Hierarchy: Few levels, wide spans
    4. Deep Hierarchy: Many levels, narrow spans
    5. Hybrid: Mix of hierarchical and flat structures
    
Communication Types:
    1. Top-Down: Commands, goals, policies from above
    2. Bottom-Up: Reports, feedback, escalations from below
    3. Lateral: Peer-to-peer at same level
    4. Cross-Level: Skip-level communication (exceptional)
    5. Broadcast: Announcements to entire subtree

Use Cases:
    - Large-scale multi-agent systems
    - Enterprise workflow automation
    - Military/command structures
    - Corporate organizational modeling
    - Distributed task management
    - Crisis management systems
    - Project management hierarchies
    - Robotic swarm coordination

Advantages:
    - Scalable communication structure
    - Clear lines of authority
    - Reduced communication overhead
    - Organized task delegation
    - Efficient escalation paths
    - Information aggregation
    - Accountability tracking
    - Span of control management

Challenges:
    - Communication delays through levels
    - Information loss in aggregation
    - Rigidity in dynamic environments
    - Single points of failure
    - Bureaucratic overhead
    - Potential bottlenecks at higher levels
    - Difficulty with cross-functional needs

LangChain Implementation:
    This implementation uses LangChain for:
    - LLM-based delegation and task breakdown
    - Status report generation and summarization
    - Escalation decision-making
    - Command interpretation
    - Cross-level communication reasoning
    
Production Considerations:
    - Design appropriate span of control (3-10 subordinates typical)
    - Implement timeout handling for non-responsive nodes
    - Log all communications for audit trails
    - Balance hierarchy depth vs. communication overhead
    - Support emergency bypass channels
    - Monitor for bottlenecks at key levels
    - Implement redundancy for critical nodes
    - Handle dynamic reorganization gracefully
"""

import os
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class AgentLevel(Enum):
    """Hierarchical level of an agent."""
    EXECUTIVE = "executive"
    MANAGER = "manager"
    WORKER = "worker"


class MessageDirection(Enum):
    """Direction of communication in hierarchy."""
    DOWN = "down"  # Superior to subordinate
    UP = "up"      # Subordinate to superior
    LATERAL = "lateral"  # Peer to peer
    BROADCAST = "broadcast"  # To entire subtree


class MessageType(Enum):
    """Type of hierarchical message."""
    COMMAND = "command"
    DELEGATION = "delegation"
    REPORT = "report"
    ESCALATION = "escalation"
    QUERY = "query"
    RESPONSE = "response"
    ANNOUNCEMENT = "announcement"


@dataclass
class HierarchicalMessage:
    """
    Message in hierarchical communication system.
    
    Attributes:
        message_id: Unique identifier
        sender: Sending agent
        recipients: List of recipient agents
        message_type: Type of message
        direction: Communication direction
        content: Message content
        priority: Priority level (1-5, 5 highest)
        timestamp: When message was sent
        requires_response: Whether response is expected
        parent_message_id: Reference to parent message
        metadata: Additional information
    """
    message_id: str
    sender: str
    recipients: List[str]
    message_type: MessageType
    direction: MessageDirection
    content: Dict[str, Any]
    priority: int = 3
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False
    parent_message_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentNode:
    """
    Node in hierarchical agent structure.
    
    Attributes:
        agent_id: Unique identifier
        level: Hierarchical level
        superior: Agent's supervisor (None for root)
        subordinates: List of direct reports
        capabilities: Agent's capabilities
        current_tasks: Active tasks
        metadata: Additional information
    """
    agent_id: str
    level: AgentLevel
    superior: Optional[str] = None
    subordinates: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    current_tasks: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalAgent:
    """
    Agent that operates within hierarchical communication structure.
    
    This agent can send commands to subordinates, report to superiors,
    escalate issues, delegate tasks, and communicate with peers.
    """
    
    def __init__(
        self,
        agent_id: str,
        level: AgentLevel,
        capabilities: List[str],
        temperature: float = 0.4
    ):
        """
        Initialize hierarchical agent.
        
        Args:
            agent_id: Agent's unique identifier
            level: Hierarchical level
            capabilities: List of capabilities
            temperature: LLM temperature
        """
        self.agent_id = agent_id
        self.level = level
        self.capabilities = capabilities
        self.superior: Optional[str] = None
        self.subordinates: List[str] = []
        self.peers: Set[str] = set()
        self.tasks: List[Dict[str, Any]] = []
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.message_history: List[HierarchicalMessage] = []
    
    def delegate_task(
        self,
        task: Dict[str, Any],
        subordinate_id: str
    ) -> HierarchicalMessage:
        """
        Delegate task to subordinate.
        
        Args:
            task: Task to delegate
            subordinate_id: ID of subordinate to delegate to
            
        Returns:
            Delegation message
        """
        # Generate delegation message using LLM
        prompt = ChatPromptTemplate.from_template(
            "You are {agent_id}, a {level} agent delegating a task. "
            "Task: {task_description}. "
            "Write clear, concise delegation instructions (2-3 sentences) "
            "for your subordinate, including objectives and expectations.\n\n"
            "Instructions:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        instructions = chain.invoke({
            "agent_id": self.agent_id,
            "level": self.level.value,
            "task_description": task.get("description", "Unknown task")
        })
        
        message = HierarchicalMessage(
            message_id=f"{self.agent_id}_del_{len(self.message_history)}",
            sender=self.agent_id,
            recipients=[subordinate_id],
            message_type=MessageType.DELEGATION,
            direction=MessageDirection.DOWN,
            content={
                "task": task,
                "instructions": instructions.strip(),
                "deadline": task.get("deadline"),
                "priority": task.get("priority", 3)
            },
            priority=task.get("priority", 3),
            requires_response=True
        )
        
        self.message_history.append(message)
        return message
    
    def report_status(
        self,
        status: str,
        details: Dict[str, Any]
    ) -> HierarchicalMessage:
        """
        Report status to superior.
        
        Args:
            status: Status summary
            details: Detailed information
            
        Returns:
            Status report message
        """
        # Generate summary using LLM
        prompt = ChatPromptTemplate.from_template(
            "You are {agent_id}, a {level} agent reporting to your superior. "
            "Status: {status}. "
            "Details: {details}. "
            "Write a professional 2-3 sentence status report.\n\n"
            "Report:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        report = chain.invoke({
            "agent_id": self.agent_id,
            "level": self.level.value,
            "status": status,
            "details": details
        })
        
        message = HierarchicalMessage(
            message_id=f"{self.agent_id}_rep_{len(self.message_history)}",
            sender=self.agent_id,
            recipients=[self.superior] if self.superior else [],
            message_type=MessageType.REPORT,
            direction=MessageDirection.UP,
            content={
                "status": status,
                "report": report.strip(),
                "details": details,
                "tasks_completed": len([t for t in self.tasks if t.get("completed")]),
                "tasks_active": len([t for t in self.tasks if not t.get("completed")])
            },
            priority=2
        )
        
        self.message_history.append(message)
        return message
    
    def escalate_issue(
        self,
        issue: str,
        severity: int,
        context: Dict[str, Any]
    ) -> HierarchicalMessage:
        """
        Escalate issue to superior.
        
        Args:
            issue: Issue description
            severity: Severity level (1-5)
            context: Additional context
            
        Returns:
            Escalation message
        """
        # Generate escalation using LLM
        prompt = ChatPromptTemplate.from_template(
            "You are {agent_id} escalating an issue to your superior. "
            "Issue: {issue}. "
            "Severity: {severity}/5. "
            "Write a clear 2-3 sentence escalation explaining the problem "
            "and why it needs higher-level attention.\n\n"
            "Escalation:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        escalation_text = chain.invoke({
            "agent_id": self.agent_id,
            "issue": issue,
            "severity": severity
        })
        
        message = HierarchicalMessage(
            message_id=f"{self.agent_id}_esc_{len(self.message_history)}",
            sender=self.agent_id,
            recipients=[self.superior] if self.superior else [],
            message_type=MessageType.ESCALATION,
            direction=MessageDirection.UP,
            content={
                "issue": issue,
                "severity": severity,
                "escalation": escalation_text.strip(),
                "context": context
            },
            priority=min(5, severity + 1),
            requires_response=True
        )
        
        self.message_history.append(message)
        return message
    
    def broadcast_announcement(
        self,
        announcement: str
    ) -> HierarchicalMessage:
        """
        Broadcast announcement to all subordinates.
        
        Args:
            announcement: Announcement text
            
        Returns:
            Broadcast message
        """
        message = HierarchicalMessage(
            message_id=f"{self.agent_id}_bcast_{len(self.message_history)}",
            sender=self.agent_id,
            recipients=self.subordinates.copy(),
            message_type=MessageType.ANNOUNCEMENT,
            direction=MessageDirection.BROADCAST,
            content={
                "announcement": announcement,
                "scope": "all_subordinates"
            },
            priority=3
        )
        
        self.message_history.append(message)
        return message


class HierarchicalOrganization:
    """
    Manages hierarchical organization of agents.
    
    This class maintains the organizational structure, routes messages,
    handles escalations, and provides organizational queries.
    """
    
    def __init__(self):
        """Initialize hierarchical organization."""
        self.agents: Dict[str, AgentNode] = {}
        self.root: Optional[str] = None
        self.llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
        self.message_log: List[HierarchicalMessage] = []
    
    def add_agent(
        self,
        agent_id: str,
        level: AgentLevel,
        superior: Optional[str] = None,
        capabilities: List[str] = None
    ):
        """
        Add agent to organization.
        
        Args:
            agent_id: Agent identifier
            level: Hierarchical level
            superior: Superior agent (None for root)
            capabilities: Agent capabilities
        """
        node = AgentNode(
            agent_id=agent_id,
            level=level,
            superior=superior,
            capabilities=capabilities or []
        )
        
        self.agents[agent_id] = node
        
        if superior is None:
            self.root = agent_id
        else:
            if superior in self.agents:
                self.agents[superior].subordinates.append(agent_id)
    
    def get_reporting_chain(self, agent_id: str) -> List[str]:
        """
        Get chain of command from agent to root.
        
        Args:
            agent_id: Starting agent
            
        Returns:
            List of agent IDs in reporting chain
        """
        chain = []
        current = agent_id
        
        while current:
            chain.append(current)
            node = self.agents.get(current)
            if not node or not node.superior:
                break
            current = node.superior
        
        return chain
    
    def get_subordinate_tree(self, agent_id: str) -> List[str]:
        """
        Get all subordinates in tree.
        
        Args:
            agent_id: Root agent
            
        Returns:
            List of all subordinate agent IDs
        """
        subordinates = []
        node = self.agents.get(agent_id)
        
        if not node:
            return subordinates
        
        for sub_id in node.subordinates:
            subordinates.append(sub_id)
            subordinates.extend(self.get_subordinate_tree(sub_id))
        
        return subordinates
    
    def route_message(self, message: HierarchicalMessage) -> bool:
        """
        Route message through hierarchy.
        
        Args:
            message: Message to route
            
        Returns:
            True if routing successful
        """
        self.message_log.append(message)
        
        # Validate sender exists
        if message.sender not in self.agents:
            return False
        
        # Validate recipients exist
        for recipient in message.recipients:
            if recipient not in self.agents:
                return False
        
        # Validate direction
        sender_node = self.agents[message.sender]
        
        if message.direction == MessageDirection.DOWN:
            # Must send to subordinates
            for recipient in message.recipients:
                if recipient not in sender_node.subordinates:
                    return False
        
        elif message.direction == MessageDirection.UP:
            # Must send to superior
            if message.recipients and message.recipients[0] != sender_node.superior:
                return False
        
        return True
    
    def aggregate_reports(
        self,
        manager_id: str
    ) -> Dict[str, Any]:
        """
        Aggregate reports from subordinates.
        
        Args:
            manager_id: Manager agent ID
            
        Returns:
            Aggregated report data
        """
        manager_node = self.agents.get(manager_id)
        if not manager_node:
            return {}
        
        # Collect reports from subordinates
        reports = []
        for msg in self.message_log[-50:]:  # Last 50 messages
            if (msg.message_type == MessageType.REPORT and
                msg.sender in manager_node.subordinates and
                manager_id in msg.recipients):
                reports.append(msg.content)
        
        # Aggregate statistics
        total_completed = sum(r.get("tasks_completed", 0) for r in reports)
        total_active = sum(r.get("tasks_active", 0) for r in reports)
        
        return {
            "subordinate_count": len(manager_node.subordinates),
            "reports_received": len(reports),
            "total_tasks_completed": total_completed,
            "total_tasks_active": total_active,
            "subordinate_status": {
                r.get("status", "unknown"): r.get("status", "unknown")
                for r in reports
            }
        }
    
    def visualize_hierarchy(self) -> str:
        """
        Create text visualization of hierarchy.
        
        Returns:
            Tree structure as text
        """
        if not self.root:
            return "Empty hierarchy"
        
        def build_tree(agent_id: str, level: int = 0) -> str:
            node = self.agents.get(agent_id)
            if not node:
                return ""
            
            indent = "  " * level
            tree = f"{indent}├─ {agent_id} ({node.level.value})\n"
            
            for sub_id in node.subordinates:
                tree += build_tree(sub_id, level + 1)
            
            return tree
        
        root_node = self.agents[self.root]
        tree = f"{self.root} ({root_node.level.value})\n"
        
        for sub_id in root_node.subordinates:
            tree += build_tree(sub_id, 1)
        
        return tree


def demonstrate_hierarchical_communication():
    """Demonstrate hierarchical communication pattern."""
    
    print("=" * 80)
    print("HIERARCHICAL COMMUNICATION PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic hierarchical structure
    print("\n" + "=" * 80)
    print("Example 1: Three-Level Organizational Hierarchy")
    print("=" * 80)
    
    # Create organization
    org = HierarchicalOrganization()
    
    # Add executive
    org.add_agent(
        "CEO",
        AgentLevel.EXECUTIVE,
        capabilities=["strategy", "resource_allocation"]
    )
    
    # Add managers
    org.add_agent(
        "Engineering_Manager",
        AgentLevel.MANAGER,
        superior="CEO",
        capabilities=["technical_oversight", "team_management"]
    )
    
    org.add_agent(
        "Operations_Manager",
        AgentLevel.MANAGER,
        superior="CEO",
        capabilities=["operations", "logistics"]
    )
    
    # Add workers
    org.add_agent(
        "Engineer_1",
        AgentLevel.WORKER,
        superior="Engineering_Manager",
        capabilities=["coding", "testing"]
    )
    
    org.add_agent(
        "Engineer_2",
        AgentLevel.WORKER,
        superior="Engineering_Manager",
        capabilities=["coding", "deployment"]
    )
    
    org.add_agent(
        "Operations_1",
        AgentLevel.WORKER,
        superior="Operations_Manager",
        capabilities=["monitoring", "maintenance"]
    )
    
    print("\nOrganizational Structure:")
    print(org.visualize_hierarchy())
    
    # Create agent instances
    ceo = HierarchicalAgent("CEO", AgentLevel.EXECUTIVE, ["strategy"])
    eng_mgr = HierarchicalAgent("Engineering_Manager", AgentLevel.MANAGER, ["technical"])
    eng_mgr.superior = "CEO"
    eng_mgr.subordinates = ["Engineer_1", "Engineer_2"]
    
    engineer1 = HierarchicalAgent("Engineer_1", AgentLevel.WORKER, ["coding"])
    engineer1.superior = "Engineering_Manager"
    
    # Example 2: Top-down communication (delegation)
    print("\n" + "=" * 80)
    print("Example 2: Top-Down Communication (Task Delegation)")
    print("=" * 80)
    
    print("\nCEO delegates to Engineering Manager:")
    task = {
        "description": "Implement new authentication system",
        "priority": 4,
        "deadline": "2 weeks"
    }
    
    delegation_msg = ceo.delegate_task(task, "Engineering_Manager")
    org.route_message(delegation_msg)
    
    print(f"  From: {delegation_msg.sender}")
    print(f"  To: {delegation_msg.recipients}")
    print(f"  Type: {delegation_msg.message_type.value}")
    print(f"  Priority: {delegation_msg.priority}")
    print(f"  Instructions: {delegation_msg.content['instructions']}")
    
    print("\nEngineering Manager delegates to Engineer:")
    sub_task = {
        "description": "Design authentication API",
        "priority": 4,
        "deadline": "1 week"
    }
    
    sub_delegation = eng_mgr.delegate_task(sub_task, "Engineer_1")
    org.route_message(sub_delegation)
    
    print(f"  From: {sub_delegation.sender}")
    print(f"  To: {sub_delegation.recipients}")
    print(f"  Instructions: {sub_delegation.content['instructions']}")
    
    # Example 3: Bottom-up communication (reporting)
    print("\n" + "=" * 80)
    print("Example 3: Bottom-Up Communication (Status Reporting)")
    print("=" * 80)
    
    # Worker reports to manager
    engineer1.tasks = [
        {"task": "API design", "completed": True},
        {"task": "Unit tests", "completed": False}
    ]
    
    print("\nEngineer reports to Engineering Manager:")
    status_msg = engineer1.report_status(
        status="In Progress",
        details={
            "completed_work": "API design completed",
            "current_work": "Writing unit tests",
            "blockers": "None"
        }
    )
    org.route_message(status_msg)
    
    print(f"  From: {status_msg.sender}")
    print(f"  To: {status_msg.recipients}")
    print(f"  Status: {status_msg.content['status']}")
    print(f"  Report: {status_msg.content['report']}")
    print(f"  Tasks Completed: {status_msg.content['tasks_completed']}")
    print(f"  Tasks Active: {status_msg.content['tasks_active']}")
    
    # Manager aggregates and reports up
    eng_mgr.tasks = [
        {"task": "Auth system", "completed": False},
        {"task": "Code review", "completed": True}
    ]
    
    print("\nEngineering Manager reports to CEO:")
    mgr_status = eng_mgr.report_status(
        status="On Track",
        details={
            "team_progress": "Authentication system 40% complete",
            "resources": "Team fully allocated",
            "risks": "No major risks identified"
        }
    )
    org.route_message(mgr_status)
    
    print(f"  From: {mgr_status.sender}")
    print(f"  To: {mgr_status.recipients}")
    print(f"  Report: {mgr_status.content['report']}")
    
    # Example 4: Escalation
    print("\n" + "=" * 80)
    print("Example 4: Issue Escalation")
    print("=" * 80)
    
    print("\nEngineer escalates blocking issue:")
    escalation = engineer1.escalate_issue(
        issue="Critical security vulnerability discovered in dependency",
        severity=4,
        context={
            "dependency": "auth-library v2.1",
            "cve": "CVE-2024-1234",
            "impact": "All users affected"
        }
    )
    org.route_message(escalation)
    
    print(f"  From: {escalation.sender}")
    print(f"  To: {escalation.recipients}")
    print(f"  Severity: {escalation.content['severity']}/5")
    print(f"  Issue: {escalation.content['issue']}")
    print(f"  Escalation: {escalation.content['escalation']}")
    print(f"  Priority: {escalation.priority}")
    
    # Manager can escalate further
    print("\nEngineering Manager escalates to CEO:")
    further_escalation = eng_mgr.escalate_issue(
        issue="Critical security vulnerability requires immediate action",
        severity=5,
        context={
            "original_reporter": "Engineer_1",
            "required_action": "Emergency patch deployment",
            "estimated_downtime": "2 hours"
        }
    )
    org.route_message(further_escalation)
    
    print(f"  From: {further_escalation.sender}")
    print(f"  To: {further_escalation.recipients}")
    print(f"  Severity: {further_escalation.content['severity']}/5")
    print(f"  Escalation: {further_escalation.content['escalation']}")
    
    # Example 5: Broadcast communication
    print("\n" + "=" * 80)
    print("Example 5: Broadcast Communication")
    print("=" * 80)
    
    print("\nCEO broadcasts announcement to all managers:")
    announcement = ceo.broadcast_announcement(
        "Company-wide security review initiated. All projects on hold for assessment."
    )
    # Manually set recipients for demo
    announcement.recipients = ["Engineering_Manager", "Operations_Manager"]
    org.route_message(announcement)
    
    print(f"  From: {announcement.sender}")
    print(f"  To: {', '.join(announcement.recipients)}")
    print(f"  Announcement: {announcement.content['announcement']}")
    
    print("\nEngineering Manager broadcasts to team:")
    team_announcement = eng_mgr.broadcast_announcement(
        "Security review starting today. Please document all dependencies."
    )
    team_announcement.recipients = ["Engineer_1", "Engineer_2"]
    org.route_message(team_announcement)
    
    print(f"  From: {team_announcement.sender}")
    print(f"  To: {', '.join(team_announcement.recipients)}")
    print(f"  Announcement: {team_announcement.content['announcement']}")
    
    # Example 6: Report aggregation
    print("\n" + "=" * 80)
    print("Example 6: Report Aggregation")
    print("=" * 80)
    
    print("\nEngineering Manager aggregates team reports:")
    aggregated = org.aggregate_reports("Engineering_Manager")
    
    print(f"  Subordinates: {aggregated['subordinate_count']}")
    print(f"  Reports Received: {aggregated['reports_received']}")
    print(f"  Total Tasks Completed: {aggregated['total_tasks_completed']}")
    print(f"  Total Tasks Active: {aggregated['total_tasks_active']}")
    
    # Example 7: Organizational queries
    print("\n" + "=" * 80)
    print("Example 7: Organizational Queries")
    print("=" * 80)
    
    print("\nReporting chain for Engineer_1:")
    chain = org.get_reporting_chain("Engineer_1")
    print(f"  {' → '.join(chain)}")
    
    print("\nAll subordinates under CEO:")
    subordinates = org.get_subordinate_tree("CEO")
    print(f"  Total: {len(subordinates)} agents")
    print(f"  {', '.join(subordinates)}")
    
    print("\nAll subordinates under Engineering_Manager:")
    eng_subs = org.get_subordinate_tree("Engineering_Manager")
    print(f"  Total: {len(eng_subs)} agents")
    print(f"  {', '.join(eng_subs)}")
    
    # Communication statistics
    print("\n" + "=" * 80)
    print("Communication Statistics")
    print("=" * 80)
    
    msg_types = defaultdict(int)
    msg_directions = defaultdict(int)
    
    for msg in org.message_log:
        msg_types[msg.message_type.value] += 1
        msg_directions[msg.direction.value] += 1
    
    print("\nMessage Types:")
    for msg_type, count in sorted(msg_types.items()):
        print(f"  {msg_type}: {count}")
    
    print("\nMessage Directions:")
    for direction, count in sorted(msg_directions.items()):
        print(f"  {direction}: {count}")
    
    print(f"\nTotal Messages: {len(org.message_log)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Hierarchical Communication Pattern")
    print("=" * 80)
    
    summary = """
    The Hierarchical Communication pattern demonstrated:
    
    1. ORGANIZATIONAL STRUCTURE (Example 1):
       - Three-level hierarchy: Executive → Managers → Workers
       - Clear reporting lines and spans of control
       - Tree-based organizational model
       - Visualization of hierarchical relationships
    
    2. TOP-DOWN COMMUNICATION (Example 2):
       - Task delegation from superiors to subordinates
       - Multi-level delegation cascade
       - Clear instructions and expectations
       - Priority and deadline tracking
    
    3. BOTTOM-UP COMMUNICATION (Example 3):
       - Status reporting from workers to managers
       - Report aggregation at manager level
       - Consolidated reporting to executives
       - Progress and metrics tracking
    
    4. ESCALATION MECHANISM (Example 4):
       - Issue escalation through reporting chain
       - Severity-based prioritization
       - Multi-level escalation for critical issues
       - Context preservation through escalation
    
    5. BROADCAST COMMUNICATION (Example 5):
       - Top-down announcements to entire subtrees
       - Manager-to-team broadcasts
       - Efficient one-to-many communication
       - Cascading announcements through levels
    
    6. REPORT AGGREGATION (Example 6):
       - Automated aggregation of subordinate reports
       - Statistical summaries for managers
       - Workload and progress visibility
       - Data-driven decision support
    
    7. ORGANIZATIONAL QUERIES (Example 7):
       - Reporting chain traversal
       - Subordinate tree enumeration
       - Span of control analysis
       - Organizational structure navigation
    
    KEY BENEFITS:
    ✓ Scalable communication structure
    ✓ Clear authority and accountability
    ✓ Efficient information flow
    ✓ Reduced communication overhead
    ✓ Organized task delegation
    ✓ Systematic escalation paths
    ✓ Information aggregation at each level
    ✓ Traceable decision-making
    
    USE CASES:
    • Large-scale multi-agent systems
    • Enterprise workflow automation
    • Military/command structures
    • Corporate organizational modeling
    • Distributed task management
    • Crisis management systems
    • Project management hierarchies
    • Robotic swarm coordination
    
    BEST PRACTICES:
    1. Keep span of control reasonable (3-10 subordinates)
    2. Define clear roles and responsibilities
    3. Establish escalation thresholds
    4. Aggregate reports at each level
    5. Allow emergency bypass when needed
    6. Log all communications for audit
    7. Monitor for bottlenecks at key levels
    8. Support dynamic reorganization
    
    TRADE-OFFS:
    • Communication speed vs. organizational control
    • Hierarchy depth vs. span of control
    • Autonomy vs. oversight
    • Flexibility vs. structure
    
    PRODUCTION CONSIDERATIONS:
    → Design appropriate organizational depth
    → Implement timeout handling for non-responsive nodes
    → Support redundancy for critical positions
    → Monitor communication bottlenecks
    → Log all messages for compliance and audit
    → Handle node failures gracefully
    → Support emergency communication channels
    → Balance delegation with oversight
    → Implement report aggregation efficiently
    → Consider matrix structures for complex needs
    
    This pattern enables scalable, organized communication in large
    multi-agent systems through clear hierarchical structure, defined
    communication channels, and systematic information flow.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_hierarchical_communication()
