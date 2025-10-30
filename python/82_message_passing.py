"""
Message Passing Pattern

Enables agents to communicate asynchronously through messages using various
communication protocols like direct messaging, publish-subscribe, and message queues.

Key Concepts:
- Asynchronous communication
- Message queues
- Publish-subscribe pattern
- Point-to-point messaging
- Broadcasting

Use Cases:
- Multi-agent systems
- Distributed agents
- Event-driven architectures
- Loosely coupled systems
"""

from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict, deque
import uuid


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class MessageType(Enum):
    """Types of messages."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    COMMAND = "command"
    EVENT = "event"
    QUERY = "query"


@dataclass
class Message:
    """Represents a message between agents."""
    id: str
    sender_id: str
    content: Any
    message_type: MessageType
    timestamp: datetime = field(default_factory=datetime.now)
    priority: MessagePriority = MessagePriority.NORMAL
    topic: Optional[str] = None
    recipient_id: Optional[str] = None  # For point-to-point
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"Message({self.id[:8]}... from {self.sender_id} type={self.message_type.value})"


class MessageQueue:
    """FIFO message queue with priority support."""
    
    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size
        self.messages: deque = deque()
        self.priority_messages: Dict[MessagePriority, deque] = {
            priority: deque() for priority in MessagePriority
        }
    
    def enqueue(self, message: Message) -> bool:
        """Add message to queue."""
        if self.max_size and self.size() >= self.max_size:
            return False
        
        if message.priority == MessagePriority.NORMAL:
            self.messages.append(message)
        else:
            self.priority_messages[message.priority].append(message)
        
        return True
    
    def dequeue(self) -> Optional[Message]:
        """Remove and return highest priority message."""
        # Check priority queues first (highest to lowest)
        for priority in sorted(MessagePriority, key=lambda p: p.value, reverse=True):
            if priority != MessagePriority.NORMAL and self.priority_messages[priority]:
                return self.priority_messages[priority].popleft()
        
        # Check normal queue
        if self.messages:
            return self.messages.popleft()
        
        return None
    
    def peek(self) -> Optional[Message]:
        """View next message without removing."""
        for priority in sorted(MessagePriority, key=lambda p: p.value, reverse=True):
            if priority != MessagePriority.NORMAL and self.priority_messages[priority]:
                return self.priority_messages[priority][0]
        
        if self.messages:
            return self.messages[0]
        
        return None
    
    def size(self) -> int:
        """Get total number of messages."""
        total = len(self.messages)
        for queue in self.priority_messages.values():
            total += len(queue)
        return total
    
    def clear(self) -> None:
        """Remove all messages."""
        self.messages.clear()
        for queue in self.priority_messages.values():
            queue.clear()


class MessageBroker:
    """Central message broker for publish-subscribe pattern."""
    
    def __init__(self):
        self.topics: Dict[str, Set[str]] = defaultdict(set)  # topic -> subscriber_ids
        self.message_history: Dict[str, List[Message]] = defaultdict(list)
    
    def subscribe(self, agent_id: str, topic: str) -> None:
        """Subscribe agent to a topic."""
        self.topics[topic].add(agent_id)
        print(f"[Broker] {agent_id} subscribed to '{topic}'")
    
    def unsubscribe(self, agent_id: str, topic: str) -> None:
        """Unsubscribe agent from a topic."""
        if topic in self.topics:
            self.topics[topic].discard(agent_id)
            print(f"[Broker] {agent_id} unsubscribed from '{topic}'")
    
    def publish(self, message: Message) -> List[str]:
        """Publish message to topic subscribers."""
        if not message.topic:
            return []
        
        subscribers = self.topics.get(message.topic, set())
        self.message_history[message.topic].append(message)
        
        return list(subscribers)
    
    def get_subscribers(self, topic: str) -> List[str]:
        """Get all subscribers for a topic."""
        return list(self.topics.get(topic, set()))
    
    def get_topic_history(self, topic: str, limit: int = 10) -> List[Message]:
        """Get recent messages for a topic."""
        return self.message_history[topic][-limit:]


class MessagingAgent:
    """Agent capable of sending and receiving messages."""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.inbox = MessageQueue(max_size=100)
        self.sent_messages: List[Message] = []
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.subscriptions: Set[str] = set()
    
    def send_message(
        self,
        content: Any,
        message_type: MessageType,
        recipient_id: Optional[str] = None,
        topic: Optional[str] = None,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> Message:
        """Send a message."""
        message = Message(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            content=content,
            message_type=message_type,
            recipient_id=recipient_id,
            topic=topic,
            priority=priority
        )
        
        self.sent_messages.append(message)
        return message
    
    def receive_message(self, message: Message) -> bool:
        """Receive a message into inbox."""
        return self.inbox.enqueue(message)
    
    def process_messages(self) -> List[Dict[str, Any]]:
        """Process all messages in inbox."""
        results = []
        
        while self.inbox.size() > 0:
            message = self.inbox.dequeue()
            if message:
                result = self.handle_message(message)
                results.append(result)
        
        return results
    
    def handle_message(self, message: Message) -> Dict[str, Any]:
        """Handle a received message."""
        handler = self.message_handlers.get(message.message_type)
        
        if handler:
            response = handler(message)
        else:
            response = self.default_handler(message)
        
        return {
            "message_id": message.id,
            "handled_by": self.agent_id,
            "response": response
        }
    
    def default_handler(self, message: Message) -> str:
        """Default message handler."""
        return f"{self.name} received: {message.content}"
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[Message], Any]
    ) -> None:
        """Register a message handler."""
        self.message_handlers[message_type] = handler
    
    def subscribe_to(self, topic: str) -> None:
        """Subscribe to a topic."""
        self.subscriptions.add(topic)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get messaging statistics."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "inbox_size": self.inbox.size(),
            "sent_count": len(self.sent_messages),
            "subscriptions": len(self.subscriptions)
        }


class MessagePassingSystem:
    """Coordinates message passing between agents."""
    
    def __init__(self):
        self.agents: Dict[str, MessagingAgent] = {}
        self.broker = MessageBroker()
        self.message_log: List[Message] = []
    
    def register_agent(self, agent: MessagingAgent) -> None:
        """Register an agent with the system."""
        self.agents[agent.agent_id] = agent
        print(f"[System] Registered agent: {agent.name} ({agent.agent_id})")
    
    def send_direct(
        self,
        sender_id: str,
        recipient_id: str,
        content: Any,
        message_type: MessageType = MessageType.REQUEST,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """Send direct point-to-point message."""
        if sender_id not in self.agents or recipient_id not in self.agents:
            return False
        
        sender = self.agents[sender_id]
        recipient = self.agents[recipient_id]
        
        message = sender.send_message(
            content=content,
            message_type=message_type,
            recipient_id=recipient_id,
            priority=priority
        )
        
        self.message_log.append(message)
        return recipient.receive_message(message)
    
    def broadcast(
        self,
        sender_id: str,
        content: Any,
        message_type: MessageType = MessageType.NOTIFICATION,
        exclude_sender: bool = True
    ) -> int:
        """Broadcast message to all agents."""
        if sender_id not in self.agents:
            return 0
        
        sender = self.agents[sender_id]
        delivered = 0
        
        for agent_id, agent in self.agents.items():
            if exclude_sender and agent_id == sender_id:
                continue
            
            message = Message(
                id=str(uuid.uuid4()),
                sender_id=sender_id,
                content=content,
                message_type=message_type
            )
            
            if agent.receive_message(message):
                delivered += 1
            
            self.message_log.append(message)
        
        return delivered
    
    def publish_to_topic(
        self,
        sender_id: str,
        topic: str,
        content: Any,
        message_type: MessageType = MessageType.EVENT
    ) -> int:
        """Publish message to topic."""
        if sender_id not in self.agents:
            return 0
        
        sender = self.agents[sender_id]
        
        message = sender.send_message(
            content=content,
            message_type=message_type,
            topic=topic
        )
        
        subscribers = self.broker.publish(message)
        delivered = 0
        
        for subscriber_id in subscribers:
            if subscriber_id in self.agents:
                agent = self.agents[subscriber_id]
                if agent.receive_message(message):
                    delivered += 1
        
        self.message_log.append(message)
        return delivered
    
    def subscribe_agent(self, agent_id: str, topic: str) -> bool:
        """Subscribe agent to topic."""
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        agent.subscribe_to(topic)
        self.broker.subscribe(agent_id, topic)
        return True
    
    def process_all_messages(self) -> Dict[str, List[Dict[str, Any]]]:
        """Process messages for all agents."""
        results = {}
        for agent_id, agent in self.agents.items():
            results[agent_id] = agent.process_messages()
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics."""
        return {
            "total_agents": len(self.agents),
            "total_messages": len(self.message_log),
            "active_topics": len(self.broker.topics),
            "agent_stats": {aid: agent.get_stats() for aid, agent in self.agents.items()}
        }


def demonstrate_message_passing():
    """Demonstrate message passing patterns."""
    print("=" * 60)
    print("MESSAGE PASSING PATTERN DEMONSTRATION")
    print("=" * 60)
    
    # Create messaging system
    system = MessagePassingSystem()
    
    # Create agents
    coordinator = MessagingAgent("coordinator", "Coordinator")
    worker1 = MessagingAgent("worker1", "Worker 1")
    worker2 = MessagingAgent("worker2", "Worker 2")
    monitor = MessagingAgent("monitor", "Monitor")
    
    # Register custom handlers
    def handle_command(msg: Message) -> str:
        return f"Executing command: {msg.content}"
    
    worker1.register_handler(MessageType.COMMAND, handle_command)
    worker2.register_handler(MessageType.COMMAND, handle_command)
    
    # Register agents
    system.register_agent(coordinator)
    system.register_agent(worker1)
    system.register_agent(worker2)
    system.register_agent(monitor)
    
    # Demonstration 1: Point-to-Point Messaging
    print("\n" + "=" * 60)
    print("1. Point-to-Point Messaging")
    print("=" * 60)
    
    system.send_direct(
        "coordinator",
        "worker1",
        "Process dataset A",
        MessageType.COMMAND
    )
    
    system.send_direct(
        "coordinator",
        "worker2",
        "Process dataset B",
        MessageType.COMMAND,
        priority=MessagePriority.HIGH
    )
    
    print("\nProcessing messages...")
    results = system.process_all_messages()
    
    for agent_id, agent_results in results.items():
        if agent_results:
            print(f"\n{system.agents[agent_id].name}:")
            for result in agent_results:
                print(f"  - {result['response']}")
    
    # Demonstration 2: Publish-Subscribe
    print("\n" + "=" * 60)
    print("2. Publish-Subscribe Pattern")
    print("=" * 60)
    
    # Subscribe agents to topics
    system.subscribe_agent("worker1", "tasks")
    system.subscribe_agent("worker2", "tasks")
    system.subscribe_agent("monitor", "tasks")
    system.subscribe_agent("monitor", "status")
    
    # Publish to topics
    print("\nPublishing to 'tasks' topic...")
    delivered = system.publish_to_topic(
        "coordinator",
        "tasks",
        "New batch job available",
        MessageType.NOTIFICATION
    )
    print(f"Message delivered to {delivered} subscribers")
    
    print("\nPublishing to 'status' topic...")
    delivered = system.publish_to_topic(
        "worker1",
        "status",
        "Task completed successfully",
        MessageType.EVENT
    )
    print(f"Message delivered to {delivered} subscribers")
    
    # Process published messages
    print("\nProcessing published messages...")
    results = system.process_all_messages()
    
    for agent_id, agent_results in results.items():
        if agent_results:
            print(f"\n{system.agents[agent_id].name}:")
            for result in agent_results:
                print(f"  - {result['response']}")
    
    # Demonstration 3: Broadcasting
    print("\n" + "=" * 60)
    print("3. Broadcasting")
    print("=" * 60)
    
    delivered = system.broadcast(
        "coordinator",
        "System maintenance in 10 minutes",
        MessageType.NOTIFICATION
    )
    print(f"\nBroadcast delivered to {delivered} agents")
    
    print("\nProcessing broadcast messages...")
    results = system.process_all_messages()
    
    for agent_id, agent_results in results.items():
        if agent_results:
            print(f"\n{system.agents[agent_id].name}:")
            for result in agent_results:
                print(f"  - {result['response']}")
    
    # Demonstration 4: Priority Messages
    print("\n" + "=" * 60)
    print("4. Priority Message Handling")
    print("=" * 60)
    
    # Send messages with different priorities
    system.send_direct("coordinator", "worker1", "Low priority task", 
                      MessageType.REQUEST, MessagePriority.LOW)
    system.send_direct("coordinator", "worker1", "Normal task", 
                      MessageType.REQUEST, MessagePriority.NORMAL)
    system.send_direct("coordinator", "worker1", "High priority task", 
                      MessageType.REQUEST, MessagePriority.HIGH)
    system.send_direct("coordinator", "worker1", "Urgent task", 
                      MessageType.REQUEST, MessagePriority.URGENT)
    
    print("\nProcessing messages (should handle by priority)...")
    results = system.process_all_messages()
    
    for agent_id, agent_results in results.items():
        if agent_results and agent_id == "worker1":
            print(f"\n{system.agents[agent_id].name} processed in order:")
            for i, result in enumerate(agent_results, 1):
                print(f"  {i}. {result['response']}")
    
    # System Statistics
    print("\n" + "=" * 60)
    print("System Statistics")
    print("=" * 60)
    
    stats = system.get_system_stats()
    print(f"Total agents: {stats['total_agents']}")
    print(f"Total messages: {stats['total_messages']}")
    print(f"Active topics: {stats['active_topics']}")
    
    print("\nPer-agent statistics:")
    for agent_id, agent_stats in stats['agent_stats'].items():
        print(f"\n{agent_stats['name']}:")
        print(f"  Messages sent: {agent_stats['sent_count']}")
        print(f"  Inbox size: {agent_stats['inbox_size']}")
        print(f"  Subscriptions: {agent_stats['subscriptions']}")


if __name__ == "__main__":
    demonstrate_message_passing()
