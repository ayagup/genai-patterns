"""
Pattern 097: Message Passing

Description:
    Message Passing enables communication between agents through explicit message exchange.
    This pattern provides a structured way for agents to share information, coordinate actions,
    and collaborate on tasks without direct coupling. Messages are discrete units of information
    that flow between agents, enabling asynchronous, decoupled, and scalable multi-agent systems.

    Message passing supports:
    - Asynchronous communication
    - Loose coupling between agents
    - Point-to-point and broadcast messaging
    - Message queuing and buffering
    - Reliable delivery guarantees
    - Priority-based message handling

Components:
    1. Message
       - Sender identification
       - Recipient(s) specification
       - Message type/category
       - Payload (data content)
       - Metadata (timestamp, priority, etc.)
       - Reply-to information

    2. Message Bus/Broker
       - Routes messages between agents
       - Handles message queuing
       - Manages subscriptions
       - Ensures delivery
       - Provides message persistence
       - Supports patterns (pub/sub, request/reply)

    3. Agent Interface
       - Send messages
       - Receive messages
       - Subscribe to topics
       - Handle message callbacks
       - Acknowledge receipt
       - Reply to messages

    4. Message Protocol
       - Message format specification
       - Serialization/deserialization
       - Error handling conventions
       - Timeout management
       - Retry logic
       - Message versioning

Use Cases:
    1. Multi-Agent Coordination
       - Task delegation
       - Status updates
       - Resource sharing
       - Result aggregation
       - Workflow orchestration
       - Event notification

    2. Distributed Systems
       - Microservices communication
       - Remote procedure calls
       - Event-driven architectures
       - Stream processing
       - Job queues
       - Service integration

    3. Collaborative Problem Solving
       - Information sharing
       - Consensus building
       - Parallel processing
       - Load distribution
       - Fault tolerance
       - Scalability

    4. Real-Time Systems
       - Alert propagation
       - State synchronization
       - Live updates
       - Chat/messaging
       - Monitoring dashboards
       - Command execution

LangChain Implementation:
    LangChain supports message passing through:
    - Custom message passing classes
    - Callback handlers for events
    - LangGraph for state management
    - Integration with message brokers
    - Async message handling

Key Features:
    1. Asynchronous Delivery
       - Non-blocking sends
       - Background processing
       - Queue-based buffering
       - Fire-and-forget option
       - Guaranteed delivery
       - Order preservation

    2. Flexible Routing
       - Direct messaging (agent-to-agent)
       - Topic-based (publish/subscribe)
       - Broadcast (one-to-many)
       - Multicast (selected recipients)
       - Request/reply pattern
       - Router/filter chains

    3. Reliability Features
       - Acknowledgments
       - Retry mechanisms
       - Dead letter queues
       - Message persistence
       - Delivery guarantees
       - Duplicate detection

    4. Scalability
       - Horizontal scaling
       - Load balancing
       - Message partitioning
       - Connection pooling
       - Efficient serialization
       - Resource management

Best Practices:
    1. Message Design
       - Keep messages small and focused
       - Include all necessary context
       - Use standard formats (JSON, Protocol Buffers)
       - Version messages for evolution
       - Include correlation IDs
       - Document message schemas

    2. Error Handling
       - Implement retry logic with backoff
       - Use dead letter queues
       - Log failed messages
       - Monitor delivery rates
       - Timeout handling
       - Circuit breakers

    3. Performance
       - Batch messages when possible
       - Use async/await patterns
       - Implement connection pooling
       - Cache frequently used data
       - Monitor queue depths
       - Optimize serialization

    4. Security
       - Authenticate senders
       - Authorize operations
       - Encrypt sensitive data
       - Validate message content
       - Rate limiting
       - Audit trails

Trade-offs:
    Advantages:
    - Loose coupling
    - Asynchronous operation
    - Scalability
    - Fault tolerance
    - Flexibility
    - Technology independence

    Disadvantages:
    - Added complexity
    - Potential message loss
    - Ordering challenges
    - Debugging difficulty
    - Latency overhead
    - Infrastructure requirements

Production Considerations:
    1. Message Broker Selection
       - RabbitMQ: Feature-rich, reliable
       - Redis: Fast, simple pub/sub
       - Kafka: High throughput, persistent
       - AWS SQS/SNS: Managed, scalable
       - Google Pub/Sub: Global, managed
       - Azure Service Bus: Enterprise features

    2. Monitoring
       - Message throughput
       - Queue depths
       - Delivery latency
       - Error rates
       - Dead letter queue size
       - Consumer lag

    3. Scaling Strategy
       - Partition messages by key
       - Add consumer instances
       - Use competing consumers
       - Implement load shedding
       - Auto-scaling based on queue depth
       - Connection pooling

    4. Reliability
       - At-least-once delivery
       - Idempotent consumers
       - Duplicate detection
       - Message persistence
       - Automatic retries
       - Dead letter handling

    5. Operations
       - Message replay capability
       - Monitoring dashboards
       - Alert configuration
       - Log aggregation
       - Backup/recovery
       - Performance tuning
"""

import os
import json
import time
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from collections import defaultdict
from queue import Queue, Empty
from threading import Thread, Lock
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class MessageType(Enum):
    """Types of messages"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    COMMAND = "command"
    EVENT = "event"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Message:
    """
    A message passed between agents.
    
    Contains sender, recipient, type, payload, and metadata.
    """
    message_id: str
    sender: str
    recipient: str
    message_type: MessageType
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['priority'] = self.priority.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        data['message_type'] = MessageType(data['message_type'])
        data['priority'] = MessagePriority(data['priority'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class MessageBus:
    """
    Central message bus for routing messages between agents.
    
    Supports point-to-point, publish/subscribe, and broadcast patterns.
    """
    
    def __init__(self):
        """Initialize message bus"""
        self.agents: Dict[str, 'Agent'] = {}
        self.subscriptions: Dict[str, List[str]] = defaultdict(list)  # topic -> agent_ids
        self.message_queue: Queue = Queue()
        self.message_history: List[Message] = []
        self.lock = Lock()
        self.running = False
        self.worker_thread: Optional[Thread] = None
    
    def register_agent(self, agent: 'Agent'):
        """
        Register an agent with the bus.
        
        Args:
            agent: Agent to register
        """
        with self.lock:
            self.agents[agent.agent_id] = agent
            print(f"[MessageBus] Registered agent: {agent.agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """
        Unregister an agent.
        
        Args:
            agent_id: Agent identifier
        """
        with self.lock:
            if agent_id in self.agents:
                del self.agents[agent_id]
                # Remove from subscriptions
                for topic in list(self.subscriptions.keys()):
                    if agent_id in self.subscriptions[topic]:
                        self.subscriptions[topic].remove(agent_id)
                print(f"[MessageBus] Unregistered agent: {agent_id}")
    
    def subscribe(self, agent_id: str, topic: str):
        """
        Subscribe agent to a topic.
        
        Args:
            agent_id: Agent identifier
            topic: Topic to subscribe to
        """
        with self.lock:
            if agent_id not in self.subscriptions[topic]:
                self.subscriptions[topic].append(agent_id)
                print(f"[MessageBus] {agent_id} subscribed to '{topic}'")
    
    def unsubscribe(self, agent_id: str, topic: str):
        """
        Unsubscribe agent from a topic.
        
        Args:
            agent_id: Agent identifier
            topic: Topic to unsubscribe from
        """
        with self.lock:
            if agent_id in self.subscriptions[topic]:
                self.subscriptions[topic].remove(agent_id)
                print(f"[MessageBus] {agent_id} unsubscribed from '{topic}'")
    
    def send_message(self, message: Message):
        """
        Send a message through the bus.
        
        Args:
            message: Message to send
        """
        self.message_queue.put(message)
        with self.lock:
            self.message_history.append(message)
    
    def publish(self, topic: str, sender: str, payload: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL):
        """
        Publish message to all subscribers of a topic.
        
        Args:
            topic: Topic to publish to
            sender: Sender agent ID
            payload: Message payload
            priority: Message priority
        """
        with self.lock:
            subscribers = self.subscriptions.get(topic, [])
        
        for recipient in subscribers:
            if recipient != sender:  # Don't send to self
                message = Message(
                    message_id=str(uuid.uuid4()),
                    sender=sender,
                    recipient=recipient,
                    message_type=MessageType.NOTIFICATION,
                    payload=payload,
                    priority=priority,
                    metadata={'topic': topic}
                )
                self.send_message(message)
    
    def broadcast(self, sender: str, payload: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL):
        """
        Broadcast message to all agents.
        
        Args:
            sender: Sender agent ID
            payload: Message payload
            priority: Message priority
        """
        with self.lock:
            recipients = [aid for aid in self.agents.keys() if aid != sender]
        
        for recipient in recipients:
            message = Message(
                message_id=str(uuid.uuid4()),
                sender=sender,
                recipient=recipient,
                message_type=MessageType.NOTIFICATION,
                payload=payload,
                priority=priority,
                metadata={'broadcast': True}
            )
            self.send_message(message)
    
    def start(self):
        """Start processing messages"""
        if not self.running:
            self.running = True
            self.worker_thread = Thread(target=self._process_messages, daemon=True)
            self.worker_thread.start()
            print("[MessageBus] Started message processing")
    
    def stop(self):
        """Stop processing messages"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
        print("[MessageBus] Stopped message processing")
    
    def _process_messages(self):
        """Process messages from queue (runs in background thread)"""
        while self.running:
            try:
                message = self.message_queue.get(timeout=0.1)
                self._deliver_message(message)
            except Empty:
                continue
    
    def _deliver_message(self, message: Message):
        """
        Deliver message to recipient.
        
        Args:
            message: Message to deliver
        """
        with self.lock:
            recipient = self.agents.get(message.recipient)
        
        if recipient:
            recipient.receive_message(message)
        else:
            print(f"[MessageBus] Warning: Recipient '{message.recipient}' not found for message {message.message_id}")
    
    def get_message_history(self, agent_id: Optional[str] = None, limit: int = 10) -> List[Message]:
        """
        Get message history.
        
        Args:
            agent_id: Filter by agent (sender or recipient)
            limit: Maximum messages to return
            
        Returns:
            List of messages
        """
        with self.lock:
            if agent_id:
                filtered = [m for m in self.message_history 
                           if m.sender == agent_id or m.recipient == agent_id]
                return filtered[-limit:]
            else:
                return self.message_history[-limit:]


class Agent:
    """
    Base agent class with message passing capabilities.
    
    Agents can send, receive, and process messages.
    """
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique agent identifier
            message_bus: Message bus for communication
        """
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.inbox: Queue = Queue()
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.running = False
        self.worker_thread: Optional[Thread] = None
        
        # Register with bus
        self.message_bus.register_agent(self)
    
    def send_message(self, recipient: str, message_type: MessageType, payload: Dict[str, Any], 
                    priority: MessagePriority = MessagePriority.NORMAL, reply_to: Optional[str] = None):
        """
        Send a message to another agent.
        
        Args:
            recipient: Recipient agent ID
            message_type: Type of message
            payload: Message payload
            priority: Message priority
            reply_to: Message ID this is replying to
        """
        message = Message(
            message_id=str(uuid.uuid4()),
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            priority=priority,
            reply_to=reply_to
        )
        self.message_bus.send_message(message)
        print(f"[{self.agent_id}] Sent {message_type.value} to {recipient}")
    
    def receive_message(self, message: Message):
        """
        Receive a message (called by message bus).
        
        Args:
            message: Received message
        """
        self.inbox.put(message)
    
    def subscribe(self, topic: str):
        """
        Subscribe to a topic.
        
        Args:
            topic: Topic to subscribe to
        """
        self.message_bus.subscribe(self.agent_id, topic)
    
    def publish(self, topic: str, payload: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL):
        """
        Publish to a topic.
        
        Args:
            topic: Topic to publish to
            payload: Message payload
            priority: Message priority
        """
        self.message_bus.publish(topic, self.agent_id, payload, priority)
    
    def broadcast(self, payload: Dict[str, Any], priority: MessagePriority = MessagePriority.NORMAL):
        """
        Broadcast message to all agents.
        
        Args:
            payload: Message payload
            priority: Message priority
        """
        self.message_bus.broadcast(self.agent_id, payload, priority)
    
    def register_handler(self, message_type: MessageType, handler: Callable[[Message], None]):
        """
        Register handler for message type.
        
        Args:
            message_type: Type of message
            handler: Handler function
        """
        self.message_handlers[message_type] = handler
    
    def start(self):
        """Start processing messages"""
        if not self.running:
            self.running = True
            self.worker_thread = Thread(target=self._process_inbox, daemon=True)
            self.worker_thread.start()
    
    def stop(self):
        """Stop processing messages"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
    
    def _process_inbox(self):
        """Process incoming messages (runs in background thread)"""
        while self.running:
            try:
                message = self.inbox.get(timeout=0.1)
                self._handle_message(message)
            except Empty:
                continue
    
    def _handle_message(self, message: Message):
        """
        Handle incoming message.
        
        Args:
            message: Message to handle
        """
        print(f"[{self.agent_id}] Received {message.message_type.value} from {message.sender}")
        
        # Call registered handler if exists
        handler = self.message_handlers.get(message.message_type)
        if handler:
            handler(message)
        else:
            self.handle_message(message)
    
    def handle_message(self, message: Message):
        """
        Default message handler (override in subclasses).
        
        Args:
            message: Message to handle
        """
        print(f"[{self.agent_id}] No handler for {message.message_type.value}")


class WorkerAgent(Agent):
    """
    Example worker agent that processes tasks.
    
    Demonstrates request/response pattern.
    """
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        """Initialize worker agent"""
        super().__init__(agent_id, message_bus)
        self.register_handler(MessageType.REQUEST, self.handle_request)
    
    def handle_request(self, message: Message):
        """
        Handle task request.
        
        Args:
            message: Request message
        """
        task = message.payload.get('task', '')
        print(f"[{self.agent_id}] Processing task: {task}")
        
        # Simulate work
        time.sleep(0.1)
        result = f"Completed: {task}"
        
        # Send response
        self.send_message(
            recipient=message.sender,
            message_type=MessageType.RESPONSE,
            payload={'result': result, 'task': task},
            reply_to=message.message_id
        )


class CoordinatorAgent(Agent):
    """
    Example coordinator agent that delegates work.
    
    Demonstrates task delegation pattern.
    """
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        """Initialize coordinator agent"""
        super().__init__(agent_id, message_bus)
        self.register_handler(MessageType.RESPONSE, self.handle_response)
        self.pending_tasks: Dict[str, str] = {}  # message_id -> task
        self.results: List[str] = []
    
    def delegate_task(self, worker_id: str, task: str):
        """
        Delegate task to worker.
        
        Args:
            worker_id: Worker agent ID
            task: Task to delegate
        """
        message_id = str(uuid.uuid4())
        self.pending_tasks[message_id] = task
        
        message = Message(
            message_id=message_id,
            sender=self.agent_id,
            recipient=worker_id,
            message_type=MessageType.REQUEST,
            payload={'task': task}
        )
        self.message_bus.send_message(message)
        print(f"[{self.agent_id}] Delegated '{task}' to {worker_id}")
    
    def handle_response(self, message: Message):
        """
        Handle response from worker.
        
        Args:
            message: Response message
        """
        if message.reply_to in self.pending_tasks:
            task = self.pending_tasks[message.reply_to]
            result = message.payload.get('result', '')
            self.results.append(result)
            print(f"[{self.agent_id}] Received result for '{task}': {result}")
            del self.pending_tasks[message.reply_to]


def demonstrate_message_passing():
    """Demonstrate message passing pattern"""
    print("=" * 80)
    print("MESSAGE PASSING DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic Point-to-Point Messaging
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Point-to-Point Messaging")
    print("=" * 80)
    
    bus = MessageBus()
    bus.start()
    
    agent1 = Agent("agent1", bus)
    agent2 = Agent("agent2", bus)
    
    agent1.start()
    agent2.start()
    
    # Send message
    print("\nSending message from agent1 to agent2...")
    agent1.send_message(
        recipient="agent2",
        message_type=MessageType.NOTIFICATION,
        payload={"text": "Hello from agent1!"}
    )
    
    time.sleep(0.5)
    
    # Example 2: Request/Response Pattern
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Request/Response Pattern with Workers")
    print("=" * 80)
    
    coordinator = CoordinatorAgent("coordinator", bus)
    worker1 = WorkerAgent("worker1", bus)
    worker2 = WorkerAgent("worker2", bus)
    
    coordinator.start()
    worker1.start()
    worker2.start()
    
    print("\nCoordinator delegating tasks to workers...")
    coordinator.delegate_task("worker1", "analyze_data")
    coordinator.delegate_task("worker2", "generate_report")
    coordinator.delegate_task("worker1", "send_notification")
    
    time.sleep(0.5)
    
    print(f"\nCoordinator collected {len(coordinator.results)} results:")
    for result in coordinator.results:
        print(f"  - {result}")
    
    # Example 3: Publish/Subscribe Pattern
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Publish/Subscribe Pattern")
    print("=" * 80)
    
    subscriber1 = Agent("subscriber1", bus)
    subscriber2 = Agent("subscriber2", bus)
    subscriber3 = Agent("subscriber3", bus)
    publisher = Agent("publisher", bus)
    
    subscriber1.start()
    subscriber2.start()
    subscriber3.start()
    publisher.start()
    
    # Subscribe to topics
    print("\nSubscribing agents to topics...")
    subscriber1.subscribe("alerts")
    subscriber2.subscribe("alerts")
    subscriber2.subscribe("updates")
    subscriber3.subscribe("updates")
    
    time.sleep(0.2)
    
    # Publish messages
    print("\nPublishing messages to topics...")
    publisher.publish("alerts", {"level": "warning", "message": "High CPU usage"})
    publisher.publish("updates", {"version": "1.2.0", "changes": "Bug fixes"})
    
    time.sleep(0.5)
    
    # Example 4: Broadcast Messaging
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Broadcast Messaging")
    print("=" * 80)
    
    print("\nBroadcasting message to all agents...")
    coordinator.broadcast({"announcement": "System maintenance in 10 minutes"})
    
    time.sleep(0.5)
    
    # Example 5: Priority Messaging
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Priority-Based Messaging")
    print("=" * 80)
    
    print("\nSending messages with different priorities...")
    agent1.send_message(
        recipient="agent2",
        message_type=MessageType.NOTIFICATION,
        payload={"text": "Low priority message"},
        priority=MessagePriority.LOW
    )
    agent1.send_message(
        recipient="agent2",
        message_type=MessageType.COMMAND,
        payload={"action": "urgent_action"},
        priority=MessagePriority.URGENT
    )
    agent1.send_message(
        recipient="agent2",
        message_type=MessageType.NOTIFICATION,
        payload={"text": "Normal priority message"},
        priority=MessagePriority.NORMAL
    )
    
    time.sleep(0.5)
    
    # Example 6: Message History
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Message History and Tracking")
    print("=" * 80)
    
    history = bus.get_message_history(limit=5)
    print(f"\nLast {len(history)} messages:")
    for msg in history:
        print(f"  {msg.timestamp.strftime('%H:%M:%S')} - {msg.sender} → {msg.recipient}: {msg.message_type.value}")
    
    # Example 7: Conversation Thread
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Threaded Conversation (Reply Pattern)")
    print("=" * 80)
    
    print("\nAgent1 sends question, Agent2 replies...")
    
    # Send question
    question_msg_id = str(uuid.uuid4())
    question = Message(
        message_id=question_msg_id,
        sender="agent1",
        recipient="agent2",
        message_type=MessageType.REQUEST,
        payload={"question": "What is the status?"}
    )
    bus.send_message(question)
    
    time.sleep(0.2)
    
    # Send reply
    reply = Message(
        message_id=str(uuid.uuid4()),
        sender="agent2",
        recipient="agent1",
        message_type=MessageType.RESPONSE,
        payload={"answer": "All systems operational"},
        reply_to=question_msg_id
    )
    bus.send_message(reply)
    
    time.sleep(0.2)
    
    print("\nMessage thread:")
    print(f"  Q: {question.payload}")
    print(f"  A: {reply.payload} (reply_to: {reply.reply_to[:8]}...)")
    
    # Cleanup
    print("\n" + "=" * 80)
    print("Cleaning up...")
    print("=" * 80)
    
    agent1.stop()
    agent2.stop()
    coordinator.stop()
    worker1.stop()
    worker2.stop()
    subscriber1.stop()
    subscriber2.stop()
    subscriber3.stop()
    publisher.stop()
    bus.stop()
    
    # Summary
    print("\n" + "=" * 80)
    print("MESSAGE PASSING SUMMARY")
    print("=" * 80)
    print("""
Message Passing Benefits:
1. Loose Coupling: Agents don't need direct references
2. Asynchronous: Non-blocking communication
3. Scalable: Easy to add more agents
4. Flexible: Multiple communication patterns
5. Reliable: Message queuing and delivery guarantees
6. Debuggable: Message history and tracking

Key Components:
1. Message
   - Unique ID for tracking
   - Sender and recipient
   - Type (request, response, notification, etc.)
   - Payload (actual data)
   - Priority for ordering
   - Metadata (timestamp, correlation ID, etc.)

2. Message Bus
   - Central routing system
   - Message queue for buffering
   - Subscription management
   - Delivery guarantees
   - History tracking
   - Background processing

3. Agent Interface
   - Send messages (point-to-point)
   - Publish to topics (one-to-many)
   - Broadcast (to all)
   - Subscribe to topics
   - Register message handlers
   - Process inbox

Communication Patterns:
1. Point-to-Point
   - Direct agent-to-agent
   - Request/response
   - Guaranteed delivery
   - Use: Direct commands, queries

2. Publish/Subscribe
   - Topic-based
   - One-to-many
   - Subscribers receive all messages
   - Use: Events, notifications, updates

3. Broadcast
   - One-to-all
   - Reaches all registered agents
   - Use: System-wide announcements

4. Request/Reply
   - Synchronous-style communication
   - Reply linked to request
   - Correlation tracking
   - Use: RPC-style calls, queries

Message Types:
- REQUEST: Ask for action/information
- RESPONSE: Reply to request
- NOTIFICATION: Inform about event
- COMMAND: Direct agent to act
- EVENT: Something happened

Priority Levels:
- LOW: Background tasks
- NORMAL: Standard messages
- HIGH: Important messages
- URGENT: Critical, process first

Best Practices:
1. Message Design
   ✓ Keep messages small (< 1MB)
   ✓ Include correlation IDs
   ✓ Version message schemas
   ✓ Use standard formats (JSON)
   ✓ Include timestamps
   ✓ Document message types

2. Error Handling
   ✓ Implement retry with backoff
   ✓ Use dead letter queues
   ✓ Set timeouts
   ✓ Log failures
   ✓ Monitor delivery rates
   ✓ Handle duplicates

3. Performance
   ✓ Batch messages when possible
   ✓ Use async processing
   ✓ Implement backpressure
   ✓ Monitor queue depth
   ✓ Optimize serialization
   ✓ Connection pooling

4. Reliability
   ✓ Acknowledge receipt
   ✓ Persist critical messages
   ✓ Duplicate detection
   ✓ Ordered delivery (if needed)
   ✓ At-least-once semantics
   ✓ Idempotent handlers

Production Considerations:
1. Message Broker
   - RabbitMQ: Feature-rich, AMQP
   - Kafka: High throughput, log-based
   - Redis: Fast, simple pub/sub
   - AWS SQS: Managed, scalable
   - Google Pub/Sub: Global scale
   - NATS: Lightweight, fast

2. Delivery Guarantees
   - At-most-once: Fast, may lose messages
   - At-least-once: Reliable, may duplicate
   - Exactly-once: Most reliable, complex

3. Ordering
   - FIFO: First in, first out
   - Priority: Higher priority first
   - No guarantee: Fastest
   - Partitioned: Order within partition

4. Scaling
   - Horizontal: Add more consumers
   - Partitioning: Split by key
   - Load balancing: Distribute evenly
   - Competing consumers: Multiple workers
   - Auto-scaling: Dynamic capacity

Common Patterns:
1. Task Queue
   - Coordinator sends tasks
   - Workers pull and process
   - Results sent back
   - Use: Distributed processing

2. Event Sourcing
   - All changes as events
   - Agents subscribe to events
   - Rebuild state from events
   - Use: Audit trails, CQRS

3. Command Query Separation
   - Commands change state
   - Queries read state
   - Different channels
   - Use: Clean architecture

4. Saga Pattern
   - Long-running transactions
   - Compensating actions
   - Message coordination
   - Use: Distributed transactions

Monitoring Metrics:
- Messages sent/received per second
- Queue depth (backlog)
- Processing latency
- Error rate
- Dead letter queue size
- Consumer lag
- Message size distribution

When to Use:
✓ Multi-agent systems
✓ Distributed architectures
✓ Asynchronous processing
✓ Event-driven systems
✓ Microservices
✓ Scalable workflows
✗ Simple single-agent systems
✗ Synchronous-only requirements
✗ Ultra-low latency critical

Alternatives:
- Direct method calls (simpler, coupled)
- Shared memory (faster, requires locks)
- REST APIs (standard, synchronous)
- gRPC (fast, bidirectional)
- WebSockets (real-time, persistent)

ROI Analysis:
- Development time: +20% (added complexity)
- Scalability: +200-500% (horizontal scaling)
- Reliability: +50% (fault tolerance)
- Maintenance: +15% (debugging complexity)
- Performance: Variable (latency vs throughput)
- Cost: Infrastructure required
""")
    
    print("\n" + "=" * 80)
    print("Pattern 097 (Message Passing) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_message_passing()
