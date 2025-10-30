"""
Agent Communication Protocol Pattern

Defines standardized protocols for agent-to-agent communication.
Implements message formats, routing, and delivery guarantees.

Use Cases:
- Multi-agent systems
- Distributed agents
- Message passing
- Protocol standardization

Advantages:
- Standardized communication
- Reliable message delivery
- Protocol flexibility
- Interoperability
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid
from collections import defaultdict, deque


class MessageType(Enum):
    """Types of messages"""
    REQUEST = "request"
    RESPONSE = "response"
    INFORM = "inform"
    QUERY = "query"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"
    SUBSCRIBE = "subscribe"
    NOTIFY = "notify"


class PerformativeType(Enum):
    """Speech act performatives (based on FIPA)"""
    INFORM = "inform"
    REQUEST = "request"
    QUERY_IF = "query_if"
    QUERY_REF = "query_ref"
    CFP = "cfp"  # Call for proposal
    PROPOSE = "propose"
    ACCEPT_PROPOSAL = "accept_proposal"
    REJECT_PROPOSAL = "reject_proposal"
    AGREE = "agree"
    REFUSE = "refuse"
    CONFIRM = "confirm"
    DISCONFIRM = "disconfirm"


class Priority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Message:
    """Agent communication message"""
    message_id: str
    performative: PerformativeType
    sender: str
    receiver: str
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to: Optional[str] = None
    conversation_id: Optional[str] = None
    priority: Priority = Priority.NORMAL
    protocol: str = "default"
    language: str = "json"
    ontology: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """Conversation between agents"""
    conversation_id: str
    participants: List[str]
    protocol: str
    started_at: datetime
    messages: List[Message] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False


@dataclass
class Subscription:
    """Topic subscription"""
    subscriber_id: str
    topic: str
    filter_criteria: Optional[Dict[str, Any]] = None
    callback: Optional[Callable] = None


class MessageRouter:
    """Routes messages between agents"""
    
    def __init__(self):
        self.routes: Dict[str, str] = {}  # agent_id -> address
        self.message_queue: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.delivered_messages: Set[str] = set()
    
    def register_agent(self, agent_id: str, address: str = "local") -> None:
        """
        Register agent with router.
        
        Args:
            agent_id: Agent identifier
            address: Agent address
        """
        self.routes[agent_id] = address
    
    def route_message(self, message: Message) -> bool:
        """
        Route message to recipient.
        
        Args:
            message: Message to route
            
        Returns:
            Whether routing succeeded
        """
        if message.receiver not in self.routes:
            return False
        
        # Add to recipient's queue
        self.message_queue[message.receiver].append(message)
        
        return True
    
    def get_messages(self,
                    agent_id: str,
                    max_count: Optional[int] = None) -> List[Message]:
        """
        Get messages for agent.
        
        Args:
            agent_id: Agent identifier
            max_count: Maximum messages to retrieve
            
        Returns:
            List of messages
        """
        if agent_id not in self.message_queue:
            return []
        
        queue = self.message_queue[agent_id]
        
        if max_count:
            messages = []
            for _ in range(min(max_count, len(queue))):
                if queue:
                    messages.append(queue.popleft())
            return messages
        else:
            messages = list(queue)
            queue.clear()
            return messages
    
    def broadcast(self,
                 message: Message,
                 exclude: Optional[List[str]] = None) -> int:
        """
        Broadcast message to all agents.
        
        Args:
            message: Message to broadcast
            exclude: Agents to exclude
            
        Returns:
            Number of recipients
        """
        if exclude is None:
            exclude = []
        
        count = 0
        
        for agent_id in self.routes.keys():
            if agent_id not in exclude and agent_id != message.sender:
                message_copy = Message(
                    message_id=str(uuid.uuid4()),
                    performative=message.performative,
                    sender=message.sender,
                    receiver=agent_id,
                    content=message.content,
                    conversation_id=message.conversation_id,
                    priority=message.priority,
                    protocol=message.protocol
                )
                
                self.route_message(message_copy)
                count += 1
        
        return count


class ConversationManager:
    """Manages agent conversations"""
    
    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}
    
    def start_conversation(self,
                          participants: List[str],
                          protocol: str = "default") -> str:
        """
        Start new conversation.
        
        Args:
            participants: List of participant agent IDs
            protocol: Conversation protocol
            
        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())
        
        conversation = Conversation(
            conversation_id=conversation_id,
            participants=participants,
            protocol=protocol,
            started_at=datetime.now()
        )
        
        self.conversations[conversation_id] = conversation
        
        return conversation_id
    
    def add_message(self,
                   conversation_id: str,
                   message: Message) -> bool:
        """
        Add message to conversation.
        
        Args:
            conversation_id: Conversation ID
            message: Message to add
            
        Returns:
            Whether message was added
        """
        conversation = self.conversations.get(conversation_id)
        
        if not conversation:
            return False
        
        conversation.messages.append(message)
        
        return True
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        return self.conversations.get(conversation_id)
    
    def get_agent_conversations(self, agent_id: str) -> List[Conversation]:
        """Get all conversations involving agent"""
        return [
            conv for conv in self.conversations.values()
            if agent_id in conv.participants
        ]


class TopicManager:
    """Manages pub/sub topics"""
    
    def __init__(self):
        self.subscriptions: Dict[str, List[Subscription]] = defaultdict(list)
        self.message_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
    
    def subscribe(self,
                 subscriber_id: str,
                 topic: str,
                 filter_criteria: Optional[Dict[str, Any]] = None,
                 callback: Optional[Callable] = None) -> None:
        """
        Subscribe to topic.
        
        Args:
            subscriber_id: Subscriber agent ID
            topic: Topic name
            filter_criteria: Optional message filter
            callback: Optional callback function
        """
        subscription = Subscription(
            subscriber_id=subscriber_id,
            topic=topic,
            filter_criteria=filter_criteria,
            callback=callback
        )
        
        self.subscriptions[topic].append(subscription)
    
    def unsubscribe(self, subscriber_id: str, topic: str) -> bool:
        """
        Unsubscribe from topic.
        
        Args:
            subscriber_id: Subscriber agent ID
            topic: Topic name
            
        Returns:
            Whether unsubscription succeeded
        """
        if topic not in self.subscriptions:
            return False
        
        self.subscriptions[topic] = [
            sub for sub in self.subscriptions[topic]
            if sub.subscriber_id != subscriber_id
        ]
        
        return True
    
    def publish(self, topic: str, message: Message) -> int:
        """
        Publish message to topic.
        
        Args:
            topic: Topic name
            message: Message to publish
            
        Returns:
            Number of subscribers notified
        """
        if topic not in self.subscriptions:
            return 0
        
        # Store in history
        self.message_history[topic].append(message)
        
        count = 0
        
        for subscription in self.subscriptions[topic]:
            # Check filter
            if subscription.filter_criteria:
                if not self._matches_filter(message, subscription.filter_criteria):
                    continue
            
            # Call callback if provided
            if subscription.callback:
                try:
                    subscription.callback(message)
                    count += 1
                except Exception:
                    pass
            else:
                count += 1
        
        return count
    
    def _matches_filter(self,
                       message: Message,
                       criteria: Dict[str, Any]) -> bool:
        """Check if message matches filter criteria"""
        for key, value in criteria.items():
            if hasattr(message, key):
                if getattr(message, key) != value:
                    return False
            elif key in message.metadata:
                if message.metadata[key] != value:
                    return False
            else:
                return False
        
        return True
    
    def get_topic_messages(self,
                          topic: str,
                          limit: int = 10) -> List[Message]:
        """Get recent messages from topic"""
        if topic not in self.message_history:
            return []
        
        history = list(self.message_history[topic])
        return history[-limit:]


class ProtocolHandler:
    """Handles communication protocols"""
    
    def __init__(self):
        self.protocols: Dict[str, Callable] = {}
    
    def register_protocol(self,
                         protocol_name: str,
                         handler: Callable) -> None:
        """
        Register protocol handler.
        
        Args:
            protocol_name: Protocol name
            handler: Handler function
        """
        self.protocols[protocol_name] = handler
    
    def handle_message(self,
                      message: Message,
                      conversation: Optional[Conversation] = None) -> Optional[Message]:
        """
        Handle message according to protocol.
        
        Args:
            message: Incoming message
            conversation: Associated conversation
            
        Returns:
            Response message if any
        """
        handler = self.protocols.get(message.protocol)
        
        if not handler:
            return None
        
        try:
            return handler(message, conversation)
        except Exception:
            return None


class AgentCommunicationSystem:
    """
    Comprehensive agent communication system.
    Handles messaging, conversations, pub/sub, and protocols.
    """
    
    def __init__(self):
        # Components
        self.router = MessageRouter()
        self.conversation_manager = ConversationManager()
        self.topic_manager = TopicManager()
        self.protocol_handler = ProtocolHandler()
        
        # Setup default protocols
        self._setup_default_protocols()
    
    def _setup_default_protocols(self) -> None:
        """Setup default protocol handlers"""
        def request_response_protocol(message: Message,
                                     conversation: Optional[Conversation]) -> Optional[Message]:
            """Handle request-response protocol"""
            if message.performative == PerformativeType.REQUEST:
                # Generate response
                response = Message(
                    message_id=str(uuid.uuid4()),
                    performative=PerformativeType.INFORM,
                    sender=message.receiver,
                    receiver=message.sender,
                    content={"status": "processed"},
                    reply_to=message.message_id,
                    conversation_id=message.conversation_id,
                    protocol=message.protocol
                )
                return response
            
            return None
        
        self.protocol_handler.register_protocol(
            "request-response",
            request_response_protocol
        )
    
    def register_agent(self, agent_id: str, address: str = "local") -> None:
        """Register agent in system"""
        self.router.register_agent(agent_id, address)
    
    def send_message(self,
                    sender: str,
                    receiver: str,
                    performative: PerformativeType,
                    content: Any,
                    conversation_id: Optional[str] = None,
                    reply_to: Optional[str] = None,
                    priority: Priority = Priority.NORMAL,
                    protocol: str = "default") -> str:
        """
        Send message from one agent to another.
        
        Args:
            sender: Sender agent ID
            receiver: Receiver agent ID
            performative: Message performative
            content: Message content
            conversation_id: Optional conversation ID
            reply_to: Optional message ID being replied to
            priority: Message priority
            protocol: Communication protocol
            
        Returns:
            Message ID
        """
        message = Message(
            message_id=str(uuid.uuid4()),
            performative=performative,
            sender=sender,
            receiver=receiver,
            content=content,
            conversation_id=conversation_id,
            reply_to=reply_to,
            priority=priority,
            protocol=protocol
        )
        
        # Route message
        self.router.route_message(message)
        
        # Add to conversation if applicable
        if conversation_id:
            self.conversation_manager.add_message(conversation_id, message)
        
        # Handle protocol
        response = self.protocol_handler.handle_message(message)
        
        if response:
            self.router.route_message(response)
        
        return message.message_id
    
    def receive_messages(self,
                        agent_id: str,
                        max_count: Optional[int] = None) -> List[Message]:
        """Receive messages for agent"""
        return self.router.get_messages(agent_id, max_count)
    
    def start_conversation(self,
                          initiator: str,
                          participants: List[str],
                          protocol: str = "default") -> str:
        """
        Start conversation between agents.
        
        Args:
            initiator: Initiating agent
            participants: Other participants
            protocol: Conversation protocol
            
        Returns:
            Conversation ID
        """
        all_participants = [initiator] + participants
        
        return self.conversation_manager.start_conversation(
            all_participants,
            protocol
        )
    
    def subscribe(self,
                 subscriber_id: str,
                 topic: str,
                 filter_criteria: Optional[Dict[str, Any]] = None,
                 callback: Optional[Callable] = None) -> None:
        """Subscribe agent to topic"""
        self.topic_manager.subscribe(
            subscriber_id,
            topic,
            filter_criteria,
            callback
        )
    
    def publish(self,
               publisher_id: str,
               topic: str,
               content: Any) -> int:
        """
        Publish message to topic.
        
        Args:
            publisher_id: Publishing agent ID
            topic: Topic name
            content: Message content
            
        Returns:
            Number of subscribers notified
        """
        message = Message(
            message_id=str(uuid.uuid4()),
            performative=PerformativeType.INFORM,
            sender=publisher_id,
            receiver="*",  # Broadcast
            content=content
        )
        
        return self.topic_manager.publish(topic, message)
    
    def broadcast(self,
                 sender: str,
                 performative: PerformativeType,
                 content: Any,
                 exclude: Optional[List[str]] = None) -> int:
        """
        Broadcast message to all agents.
        
        Args:
            sender: Sender agent ID
            performative: Message performative
            content: Message content
            exclude: Agents to exclude
            
        Returns:
            Number of recipients
        """
        message = Message(
            message_id=str(uuid.uuid4()),
            performative=performative,
            sender=sender,
            receiver="*",
            content=content
        )
        
        return self.router.broadcast(message, exclude)
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get conversation details"""
        return self.conversation_manager.get_conversation(conversation_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_messages = sum(
            len(queue) for queue in self.router.message_queue.values()
        )
        
        return {
            "registered_agents": len(self.router.routes),
            "active_conversations": len(self.conversation_manager.conversations),
            "total_topics": len(self.topic_manager.subscriptions),
            "queued_messages": total_messages,
            "registered_protocols": len(self.protocol_handler.protocols)
        }


def demonstrate_agent_communication():
    """Demonstrate agent communication system"""
    print("=" * 70)
    print("Agent Communication Protocol Demonstration")
    print("=" * 70)
    
    system = AgentCommunicationSystem()
    
    # Example 1: Register agents
    print("\n1. Registering Agents:")
    
    agents = ["agent_1", "agent_2", "agent_3", "agent_4"]
    
    for agent_id in agents:
        system.register_agent(agent_id)
        print("  Registered: {}".format(agent_id))
    
    # Example 2: Send messages
    print("\n2. Sending Messages:")
    
    msg_id = system.send_message(
        sender="agent_1",
        receiver="agent_2",
        performative=PerformativeType.REQUEST,
        content={"action": "process_data", "data": [1, 2, 3]},
        priority=Priority.HIGH
    )
    
    print("  Sent message: {}".format(msg_id))
    print("  From: agent_1")
    print("  To: agent_2")
    print("  Type: REQUEST")
    
    # Example 3: Receive messages
    print("\n3. Receiving Messages:")
    
    messages = system.receive_messages("agent_2")
    
    print("  agent_2 received {} message(s)".format(len(messages)))
    
    for msg in messages:
        print("    From: {}".format(msg.sender))
        print("    Performative: {}".format(msg.performative.value))
        print("    Content: {}".format(msg.content))
    
    # Example 4: Conversation
    print("\n4. Starting Conversation:")
    
    conv_id = system.start_conversation(
        initiator="agent_1",
        participants=["agent_2", "agent_3"],
        protocol="negotiation"
    )
    
    print("  Conversation ID: {}".format(conv_id))
    
    # Send messages in conversation
    for i in range(3):
        system.send_message(
            sender="agent_1",
            receiver="agent_2",
            performative=PerformativeType.PROPOSE,
            content={"proposal": i},
            conversation_id=conv_id
        )
    
    conversation = system.get_conversation(conv_id)
    print("  Messages in conversation: {}".format(len(conversation.messages)))
    
    # Example 5: Pub/Sub
    print("\n5. Publish/Subscribe:")
    
    # Subscribe to topic
    def on_alert(message: Message):
        print("  Alert received: {}".format(message.content))
    
    system.subscribe("agent_3", "alerts", callback=on_alert)
    system.subscribe("agent_4", "alerts", callback=on_alert)
    
    print("  Subscribed agent_3 and agent_4 to 'alerts'")
    
    # Publish to topic
    count = system.publish(
        publisher_id="agent_1",
        topic="alerts",
        content={"level": "warning", "message": "High CPU usage"}
    )
    
    print("  Published alert to {} subscribers".format(count))
    
    # Example 6: Broadcasting
    print("\n6. Broadcasting Messages:")
    
    count = system.broadcast(
        sender="agent_1",
        performative=PerformativeType.INFORM,
        content={"announcement": "System maintenance at 10 PM"},
        exclude=["agent_1"]
    )
    
    print("  Broadcast message to {} agents".format(count))
    
    # Example 7: Priority messages
    print("\n7. Priority Message Handling:")
    
    # Send messages with different priorities
    priorities = [Priority.LOW, Priority.NORMAL, Priority.HIGH, Priority.URGENT]
    
    for priority in priorities:
        system.send_message(
            sender="agent_1",
            receiver="agent_2",
            performative=PerformativeType.INFORM,
            content={"priority_level": priority.value},
            priority=priority
        )
    
    messages = system.receive_messages("agent_2")
    print("  agent_2 received {} prioritized messages".format(len(messages)))
    
    # Example 8: Filtered subscriptions
    print("\n8. Filtered Subscriptions:")
    
    system.subscribe(
        "agent_4",
        "events",
        filter_criteria={"priority": Priority.HIGH}
    )
    
    # Publish with different priorities
    system.publish("agent_1", "events", {"priority": Priority.LOW, "data": "low"})
    system.publish("agent_1", "events", {"priority": Priority.HIGH, "data": "high"})
    
    print("  Published 2 events with filters")
    
    # Example 9: Statistics
    print("\n9. Communication Statistics:")
    stats = system.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Example 10: Protocol handling
    print("\n10. Custom Protocol:")
    
    def custom_protocol_handler(message: Message,
                                conversation: Optional[Conversation]) -> Optional[Message]:
        print("  Custom protocol handling message: {}".format(message.message_id))
        return None
    
    system.protocol_handler.register_protocol("custom", custom_protocol_handler)
    
    system.send_message(
        sender="agent_1",
        receiver="agent_2",
        performative=PerformativeType.REQUEST,
        content={"custom": "data"},
        protocol="custom"
    )
    
    print("  Registered and used custom protocol")


if __name__ == "__main__":
    demonstrate_agent_communication()
