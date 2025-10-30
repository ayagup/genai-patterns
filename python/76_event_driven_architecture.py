"""
Event-Driven Architecture Pattern

Reactive system design where agents respond to events through publishers,
subscribers, and message brokers. Enables loose coupling, scalability,
and real-time responsiveness.

Use Cases:
- Real-time agent systems
- Asynchronous workflows
- Event streaming platforms
- Microservices communication
- Reactive agent behaviors

Benefits:
- Loose coupling between components
- Scalability through async processing
- Real-time event handling
- Audit trails and event sourcing
- Flexible system evolution
"""

from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time
from collections import defaultdict


class EventPriority(Enum):
    """Event priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class Event:
    """An event in the system"""
    event_id: str
    event_type: str
    source: str
    timestamp: datetime
    data: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now()


@dataclass
class EventFilter:
    """Filter for event subscriptions"""
    event_types: Optional[Set[str]] = None
    sources: Optional[Set[str]] = None
    priority: Optional[EventPriority] = None
    custom_filter: Optional[Callable[[Event], bool]] = None
    
    def matches(self, event: Event) -> bool:
        """Check if event matches filter"""
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        if self.sources and event.source not in self.sources:
            return False
        
        if self.priority and event.priority.value > self.priority.value:
            return False
        
        if self.custom_filter and not self.custom_filter(event):
            return False
        
        return True


class EventHandler:
    """Handles events for a subscriber"""
    
    def __init__(
        self,
        handler_id: str,
        callback: Callable[[Event], None],
        event_filter: Optional[EventFilter] = None
    ):
        self.handler_id = handler_id
        self.callback = callback
        self.filter = event_filter or EventFilter()
        self.events_processed = 0
        self.last_event_time: Optional[datetime] = None
    
    def can_handle(self, event: Event) -> bool:
        """Check if handler can process event"""
        return self.filter.matches(event)
    
    def handle(self, event: Event) -> None:
        """Process event"""
        self.callback(event)
        self.events_processed += 1
        self.last_event_time = datetime.now()


class EventBus:
    """In-memory event bus for publish-subscribe"""
    
    def __init__(self, name: str = "Event Bus"):
        self.name = name
        self.handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.total_events = 0
        
        print(f"[Event Bus] Initialized: {name}")
    
    def subscribe(
        self,
        subscriber_id: str,
        handler: EventHandler,
        event_types: Optional[List[str]] = None
    ) -> None:
        """Subscribe to events"""
        if event_types:
            for event_type in event_types:
                self.handlers[event_type].append(handler)
                print(f"[Subscribe] {subscriber_id} → {event_type}")
        else:
            # Subscribe to all events
            self.handlers["*"].append(handler)
            print(f"[Subscribe] {subscriber_id} → ALL events")
    
    def unsubscribe(
        self,
        subscriber_id: str,
        event_types: Optional[List[str]] = None
    ) -> None:
        """Unsubscribe from events"""
        if event_types:
            for event_type in event_types:
                self.handlers[event_type] = [
                    h for h in self.handlers[event_type]
                    if h.handler_id != subscriber_id
                ]
        else:
            # Unsubscribe from all
            for event_type in self.handlers:
                self.handlers[event_type] = [
                    h for h in self.handlers[event_type]
                    if h.handler_id != subscriber_id
                ]
    
    def publish(self, event: Event) -> int:
        """Publish event to subscribers"""
        self.event_history.append(event)
        self.total_events += 1
        
        print(f"\n[Event Published] {event.event_type}")
        print(f"  Source: {event.source}")
        print(f"  Priority: {event.priority.name}")
        print(f"  Timestamp: {event.timestamp}")
        
        # Get handlers for this event type
        handlers = self.handlers.get(event.event_type, []) + self.handlers.get("*", [])
        
        # Filter and invoke handlers
        invoked = 0
        for handler in handlers:
            if handler.can_handle(event):
                try:
                    handler.handle(event)
                    invoked += 1
                except Exception as e:
                    print(f"  Error in handler {handler.handler_id}: {e}")
        
        print(f"  Handlers invoked: {invoked}")
        return invoked
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            "total_events": self.total_events,
            "event_types": len(self.handlers),
            "total_handlers": sum(len(h) for h in self.handlers.values()),
            "history_size": len(self.event_history)
        }


class MessageQueue:
    """Message queue for reliable event delivery"""
    
    def __init__(self, name: str):
        self.name = name
        self.queue: List[Event] = []
        self.processing: Dict[str, Event] = {}
        self.dead_letter: List[Event] = []
        self.max_retries = 3
        self.retry_counts: Dict[str, int] = defaultdict(int)
    
    def enqueue(self, event: Event) -> None:
        """Add event to queue"""
        self.queue.append(event)
        print(f"[Queue] Enqueued: {event.event_type} (queue size: {len(self.queue)})")
    
    def dequeue(self) -> Optional[Event]:
        """Get next event from queue"""
        if not self.queue:
            return None
        
        event = self.queue.pop(0)
        self.processing[event.event_id] = event
        return event
    
    def acknowledge(self, event_id: str) -> bool:
        """Acknowledge successful processing"""
        if event_id in self.processing:
            del self.processing[event_id]
            if event_id in self.retry_counts:
                del self.retry_counts[event_id]
            return True
        return False
    
    def reject(self, event_id: str, requeue: bool = True) -> bool:
        """Reject event (optionally requeue or send to dead letter)"""
        if event_id not in self.processing:
            return False
        
        event = self.processing[event_id]
        del self.processing[event_id]
        
        self.retry_counts[event_id] += 1
        
        if self.retry_counts[event_id] >= self.max_retries:
            # Send to dead letter queue
            self.dead_letter.append(event)
            print(f"[Queue] Moved to dead letter: {event.event_type}")
            return True
        
        if requeue:
            self.queue.insert(0, event)  # Add to front for retry
            print(f"[Queue] Requeued: {event.event_type} (retry {self.retry_counts[event_id]})")
        
        return True
    
    def size(self) -> int:
        """Get queue size"""
        return len(self.queue)


class EventBroker:
    """Message broker with topics and queues"""
    
    def __init__(self, name: str = "Event Broker"):
        self.name = name
        self.topics: Dict[str, EventBus] = {}
        self.queues: Dict[str, MessageQueue] = {}
        
        print(f"[Broker] Initialized: {name}")
    
    def create_topic(self, topic_name: str) -> EventBus:
        """Create a new topic"""
        if topic_name not in self.topics:
            self.topics[topic_name] = EventBus(topic_name)
            print(f"[Broker] Created topic: {topic_name}")
        return self.topics[topic_name]
    
    def create_queue(self, queue_name: str) -> MessageQueue:
        """Create a new queue"""
        if queue_name not in self.queues:
            self.queues[queue_name] = MessageQueue(queue_name)
            print(f"[Broker] Created queue: {queue_name}")
        return self.queues[queue_name]
    
    def publish_to_topic(self, topic_name: str, event: Event) -> int:
        """Publish event to topic"""
        if topic_name not in self.topics:
            self.create_topic(topic_name)
        
        return self.topics[topic_name].publish(event)
    
    def send_to_queue(self, queue_name: str, event: Event) -> None:
        """Send event to queue"""
        if queue_name not in self.queues:
            self.create_queue(queue_name)
        
        self.queues[queue_name].enqueue(event)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get broker statistics"""
        return {
            "topics": len(self.topics),
            "queues": len(self.queues),
            "topic_stats": {
                name: bus.get_statistics()
                for name, bus in self.topics.items()
            },
            "queue_stats": {
                name: {
                    "size": queue.size(),
                    "processing": len(queue.processing),
                    "dead_letter": len(queue.dead_letter)
                }
                for name, queue in self.queues.items()
            }
        }


class EventSourcingStore:
    """Event store for event sourcing pattern"""
    
    def __init__(self):
        self.events: List[Event] = []
        self.snapshots: Dict[str, Any] = {}
    
    def append(self, event: Event) -> None:
        """Append event to store"""
        self.events.append(event)
    
    def get_events(
        self,
        stream_id: Optional[str] = None,
        event_type: Optional[str] = None,
        from_timestamp: Optional[datetime] = None
    ) -> List[Event]:
        """Query events from store"""
        filtered = self.events
        
        if stream_id:
            filtered = [e for e in filtered if e.correlation_id == stream_id]
        
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        
        if from_timestamp:
            filtered = [e for e in filtered if e.timestamp >= from_timestamp]
        
        return filtered
    
    def rebuild_state(self, stream_id: str) -> Dict[str, Any]:
        """Rebuild state from events"""
        events = self.get_events(stream_id=stream_id)
        
        # Simple state reconstruction
        state: Dict[str, Any] = {}
        
        for event in events:
            # Apply event to state
            state.update(event.data)
        
        return state
    
    def create_snapshot(self, stream_id: str, state: Any) -> None:
        """Create state snapshot"""
        self.snapshots[stream_id] = {
            "state": state,
            "timestamp": datetime.now(),
            "event_count": len(self.get_events(stream_id=stream_id))
        }


class EventDrivenAgent:
    """Agent that operates in event-driven manner"""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.event_handlers: Dict[str, Callable] = {}
        
    def on(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """Register event handler"""
        self.event_handlers[event_type] = handler
        print(f"[Agent {self.name}] Registered handler for: {event_type}")
    
    def emit(self, broker: EventBroker, topic: str, event: Event) -> None:
        """Emit event"""
        event.source = self.agent_id
        broker.publish_to_topic(topic, event)
    
    def handle_event(self, event: Event) -> None:
        """Handle incoming event"""
        handler = self.event_handlers.get(event.event_type)
        if handler:
            print(f"[Agent {self.name}] Handling: {event.event_type}")
            handler(event)


def demonstrate_event_driven_architecture():
    """
    Demonstrate Event-Driven Architecture pattern
    """
    print("=" * 70)
    print("EVENT-DRIVEN ARCHITECTURE DEMONSTRATION")
    print("=" * 70)
    
    # Create broker
    broker = EventBroker("AI Agent Broker")
    
    # Example 1: Topic-based pub-sub
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Publish-Subscribe with Topics")
    print("=" * 70)
    
    # Create topic
    user_events = broker.create_topic("user.events")
    
    # Create subscribers
    def log_handler(event: Event):
        print(f"  [Logger] Received: {event.event_type}")
    
    def analytics_handler(event: Event):
        print(f"  [Analytics] Processing: {event.event_type}")
    
    logger = EventHandler("logger", log_handler)
    analytics = EventHandler("analytics", analytics_handler)
    
    user_events.subscribe("logger", logger)
    user_events.subscribe("analytics", analytics)
    
    # Publish events
    events = [
        Event("evt-1", "user.login", "auth-service", datetime.now(), 
              {"user_id": "user123"}),
        Event("evt-2", "user.action", "app-service", datetime.now(), 
              {"action": "click", "user_id": "user123"}),
    ]
    
    for event in events:
        broker.publish_to_topic("user.events", event)
    
    # Example 2: Message queues
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Message Queue for Reliable Delivery")
    print("=" * 70)
    
    task_queue = broker.create_queue("tasks")
    
    # Send events to queue
    for i in range(3):
        event = Event(
            f"task-{i}",
            "task.process",
            "scheduler",
            datetime.now(),
            {"task_id": i}
        )
        broker.send_to_queue("tasks", event)
    
    # Process queue
    print("\n[Processing Queue]")
    while task_queue.size() > 0:
        event = task_queue.dequeue()
        if event:
            print(f"  Processing: {event.event_type}")
            # Simulate processing
            time.sleep(0.01)
            task_queue.acknowledge(event.event_id)
    
    # Example 3: Event-driven agents
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Event-Driven Agent Communication")
    print("=" * 70)
    
    # Create agents
    reasoning_agent = EventDrivenAgent("reasoning-1", "Reasoning Agent")
    memory_agent = EventDrivenAgent("memory-1", "Memory Agent")
    
    # Setup event handlers
    def on_query(event: Event):
        query = event.data.get("query")
        print(f"    Processing query: {query}")
        
        # Emit response event
        response_event = Event(
            f"resp-{event.event_id}",
            "query.response",
            reasoning_agent.agent_id,
            datetime.now(),
            {"response": f"Answer to: {query}"},
            correlation_id=event.correlation_id
        )
        reasoning_agent.emit(broker, "agent.events", response_event)
    
    def on_response(event: Event):
        response = event.data.get("response")
        print(f"    Storing response: {response}")
    
    reasoning_agent.on("query.submitted", on_query)
    memory_agent.on("query.response", on_response)
    
    # Create topic for agent events
    agent_events = broker.create_topic("agent.events")
    
    # Subscribe agents
    reasoning_handler = EventHandler(
        "reasoning",
        reasoning_agent.handle_event,
        EventFilter(event_types={"query.submitted"})
    )
    
    memory_handler = EventHandler(
        "memory",
        memory_agent.handle_event,
        EventFilter(event_types={"query.response"})
    )
    
    agent_events.subscribe("reasoning", reasoning_handler, ["query.submitted"])
    agent_events.subscribe("memory", memory_handler, ["query.response"])
    
    # Emit query
    query_event = Event(
        "query-1",
        "query.submitted",
        "user",
        datetime.now(),
        {"query": "What is the weather?"},
        correlation_id="conv-123"
    )
    
    broker.publish_to_topic("agent.events", query_event)
    
    # Example 4: Event sourcing
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Event Sourcing")
    print("=" * 70)
    
    event_store = EventSourcingStore()
    
    # Record events
    events = [
        Event("es-1", "conversation.started", "system", datetime.now(),
              {"conversation_id": "conv-456", "user": "Alice"}),
        Event("es-2", "message.sent", "user", datetime.now(),
              {"text": "Hello", "conversation_id": "conv-456"},
              correlation_id="conv-456"),
        Event("es-3", "message.sent", "agent", datetime.now(),
              {"text": "Hi! How can I help?", "conversation_id": "conv-456"},
              correlation_id="conv-456"),
    ]
    
    for event in events:
        event_store.append(event)
        print(f"  Stored: {event.event_type}")
    
    # Rebuild state
    print("\n[Rebuilding Conversation State]")
    state = event_store.rebuild_state("conv-456")
    print(f"  Rebuilt state: {state}")
    
    # Example 5: Statistics
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Broker Statistics")
    print("=" * 70)
    
    stats = broker.get_statistics()
    print(f"\n[Broker Statistics]")
    print(f"  Topics: {stats['topics']}")
    print(f"  Queues: {stats['queues']}")
    
    for topic_name, topic_stats in stats['topic_stats'].items():
        print(f"\n  Topic: {topic_name}")
        print(f"    Total events: {topic_stats['total_events']}")
        print(f"    Handlers: {topic_stats['total_handlers']}")


if __name__ == "__main__":
    demonstrate_event_driven_architecture()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Events enable loose coupling between components
2. Pub-sub allows multiple subscribers per event
3. Message queues provide reliable delivery
4. Event sourcing creates audit trails
5. Async processing improves scalability

Best Practices:
- Design clear event schemas
- Use correlation IDs for tracking
- Implement idempotent handlers
- Handle failures with dead letter queues
- Monitor event flow and latency
- Version events for backward compatibility
- Consider eventual consistency
- Implement proper error handling
    """)
