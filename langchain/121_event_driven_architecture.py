"""
Pattern 121: Event-Driven Architecture

Description:
    Agents react to events in the system using pub-sub patterns and
    event buses for loose coupling and scalability.

Components:
    - Event bus
    - Publishers
    - Subscribers
    - Event handlers

Use Cases:
    - Real-time systems
    - Microservices
    - Reactive applications

LangChain Implementation:
    Implements event-driven pattern with event handlers and async processing.
"""

import os
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class EventType(Enum):
    """Types of events in the system."""
    USER_ACTION = "user_action"
    SYSTEM_STATE = "system_state"
    DATA_UPDATE = "data_update"
    ERROR = "error"
    NOTIFICATION = "notification"


@dataclass
class Event:
    """Represents an event in the system."""
    id: str
    type: EventType
    source: str
    data: Dict[str, Any]
    timestamp: str
    priority: int = 5


class EventBus:
    """Central event bus for publish-subscribe pattern."""
    
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_history: List[Event] = []
        
    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe a handler to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
        print(f"✓ Subscribed handler to {event_type.value} events")
    
    def publish(self, event: Event):
        """Publish an event to all subscribers."""
        print(f"\n📤 Publishing event: {event.type.value} from {event.source}")
        print(f"   Data: {event.data}")
        
        self.event_history.append(event)
        
        # Notify subscribers
        if event.type in self.subscribers:
            for handler in self.subscribers[event.type]:
                try:
                    handler(event)
                except Exception as e:
                    print(f"   ⚠ Handler error: {e}")
    
    def get_event_count(self, event_type: EventType = None) -> int:
        """Get count of events."""
        if event_type:
            return sum(1 for e in self.event_history if e.type == event_type)
        return len(self.event_history)


class EventDrivenAgent:
    """Agent that reacts to events."""
    
    def __init__(self, name: str, event_bus: EventBus, model_name: str = "gpt-4"):
        self.name = name
        self.event_bus = event_bus
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.handled_events = []
        
    def handle_event(self, event: Event):
        """Handle an incoming event."""
        print(f"   📥 {self.name} received event")
        
        # Process event with LLM
        response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an event-driven agent named {agent_name}.
React appropriately to the event based on its type and data."""),
            ("user", """Event Type: {event_type}
Source: {source}
Data: {data}

Provide your response action:""")
        ])
        
        chain = response_prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "agent_name": self.name,
            "event_type": event.type.value,
            "source": event.source,
            "data": str(event.data)
        })
        
        print(f"   🔧 {self.name} action: {response[:100]}...")
        
        self.handled_events.append({
            "event_id": event.id,
            "response": response
        })
        
        # Potentially publish new events
        if "notify" in response.lower():
            self.publish_notification(event)
    
    def publish_notification(self, original_event: Event):
        """Publish a notification event."""
        notification_event = Event(
            id=f"notif_{len(self.event_bus.event_history)}",
            type=EventType.NOTIFICATION,
            source=self.name,
            data={"message": f"Processed event {original_event.id}"},
            timestamp=datetime.now().isoformat()
        )
        self.event_bus.publish(notification_event)


class EventAnalyzer:
    """Analyzes event patterns."""
    
    def __init__(self, event_bus: EventBus, model_name: str = "gpt-4"):
        self.event_bus = event_bus
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
    
    def analyze_patterns(self) -> str:
        """Analyze event patterns using LLM."""
        events_summary = "\n".join([
            f"{e.type.value}: {e.source} -> {e.data}"
            for e in self.event_bus.event_history[-20:]
        ])
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze these event patterns and identify trends."),
            ("user", """Recent Events:
{events}

Provide:
1. Event frequency patterns
2. Common event chains
3. Potential issues
4. Optimization suggestions""")
        ])
        
        chain = analysis_prompt | self.llm | StrOutputParser()
        return chain.invoke({"events": events_summary})


def demonstrate_event_driven_architecture():
    """Demonstrate event-driven architecture pattern."""
    print("=== Event-Driven Architecture Pattern ===\n")
    
    # Create event bus
    event_bus = EventBus()
    
    # Create agents
    print("1. Creating Event-Driven Agents")
    print("-" * 50)
    
    monitoring_agent = EventDrivenAgent("MonitoringAgent", event_bus)
    analytics_agent = EventDrivenAgent("AnalyticsAgent", event_bus)
    notification_agent = EventDrivenAgent("NotificationAgent", event_bus)
    
    # Subscribe agents to events
    print("\n2. Subscribing to Events")
    print("-" * 50)
    
    event_bus.subscribe(EventType.USER_ACTION, monitoring_agent.handle_event)
    event_bus.subscribe(EventType.USER_ACTION, analytics_agent.handle_event)
    
    event_bus.subscribe(EventType.SYSTEM_STATE, monitoring_agent.handle_event)
    
    event_bus.subscribe(EventType.ERROR, monitoring_agent.handle_event)
    event_bus.subscribe(EventType.ERROR, notification_agent.handle_event)
    
    event_bus.subscribe(EventType.NOTIFICATION, notification_agent.handle_event)
    
    print()
    
    # Publish events
    print("3. Publishing Events")
    print("-" * 50)
    
    # User action event
    event_bus.publish(Event(
        id="evt_001",
        type=EventType.USER_ACTION,
        source="WebApp",
        data={"action": "login", "user_id": "user123"},
        timestamp=datetime.now().isoformat(),
        priority=5
    ))
    
    # System state event
    event_bus.publish(Event(
        id="evt_002",
        type=EventType.SYSTEM_STATE,
        source="SystemMonitor",
        data={"cpu": 75, "memory": 60, "status": "healthy"},
        timestamp=datetime.now().isoformat(),
        priority=3
    ))
    
    # Data update event
    event_bus.publish(Event(
        id="evt_003",
        type=EventType.DATA_UPDATE,
        source="Database",
        data={"table": "users", "operation": "insert", "count": 5},
        timestamp=datetime.now().isoformat(),
        priority=4
    ))
    
    # Error event
    event_bus.publish(Event(
        id="evt_004",
        type=EventType.ERROR,
        source="APIGateway",
        data={"error": "RateLimitExceeded", "endpoint": "/api/data"},
        timestamp=datetime.now().isoformat(),
        priority=9
    ))
    
    # Another user action
    event_bus.publish(Event(
        id="evt_005",
        type=EventType.USER_ACTION,
        source="MobileApp",
        data={"action": "purchase", "amount": 99.99},
        timestamp=datetime.now().isoformat(),
        priority=7
    ))
    
    # Analyze event patterns
    print("\n4. Analyzing Event Patterns")
    print("-" * 50)
    
    analyzer = EventAnalyzer(event_bus)
    analysis = analyzer.analyze_patterns()
    print(analysis)
    
    # Statistics
    print("\n5. Event Statistics")
    print("-" * 50)
    print(f"Total events: {event_bus.get_event_count()}")
    print(f"User actions: {event_bus.get_event_count(EventType.USER_ACTION)}")
    print(f"System states: {event_bus.get_event_count(EventType.SYSTEM_STATE)}")
    print(f"Errors: {event_bus.get_event_count(EventType.ERROR)}")
    print(f"Notifications: {event_bus.get_event_count(EventType.NOTIFICATION)}")
    
    print(f"\nMonitoring agent handled: {len(monitoring_agent.handled_events)} events")
    print(f"Analytics agent handled: {len(analytics_agent.handled_events)} events")
    print(f"Notification agent handled: {len(notification_agent.handled_events)} events")
    
    print("\n=== Summary ===")
    print("Event-driven architecture demonstrated with:")
    print("- Event bus (pub-sub pattern)")
    print("- Multiple event types")
    print("- Event subscribers and handlers")
    print("- Event-driven agents")
    print("- Event chain reactions")
    print("- Pattern analysis")
    print("- Loose coupling between components")


if __name__ == "__main__":
    demonstrate_event_driven_architecture()
