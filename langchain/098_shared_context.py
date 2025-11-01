"""
Pattern 098: Shared Context/Workspace

Description:
    Shared Context/Workspace provides a common space where multiple agents can read and write
    shared information, coordinate activities, and collaborate on tasks. This pattern enables
    agents to maintain shared state, exchange data without direct messaging, and build upon
    each other's work. The shared workspace acts as a collaborative knowledge base and
    coordination medium for multi-agent systems.

    Shared workspace features:
    - Common data structures accessible to all agents
    - Read/write operations with concurrency control
    - Version tracking and conflict resolution
    - Notifications on changes
    - Access control and permissions
    - Transaction support

Components:
    1. Workspace
       - Shared data store
       - Key-value storage
       - Document/object storage
       - Access control lists
       - Change notifications
       - Version history

    2. Context Manager
       - Manages workspace state
       - Handles concurrent access
       - Resolves conflicts
       - Maintains consistency
       - Provides transactions
       - Enforces policies

    3. Agent Interface
       - Read operations
       - Write operations
       - Subscribe to changes
       - Lock/unlock resources
       - Query workspace
       - Batch operations

    4. Synchronization
       - Optimistic locking
       - Pessimistic locking
       - Version control
       - Merge strategies
       - Conflict detection
       - Resolution policies

Use Cases:
    1. Collaborative Problem Solving
       - Shared scratchpad
       - Intermediate results
       - Partial solutions
       - Knowledge accumulation
       - Iterative refinement
       - Collective intelligence

    2. Multi-Agent Coordination
       - Task assignments
       - Status tracking
       - Resource allocation
       - Progress monitoring
       - Goal management
       - Constraint sharing

    3. Knowledge Sharing
       - Common knowledge base
       - Learned patterns
       - Best practices
       - Historical data
       - Shared context
       - Collective memory

    4. Workflow Management
       - Process state
       - Task dependencies
       - Input/output data
       - Checkpoint saving
       - Error recovery
       - Audit trails

LangChain Implementation:
    LangChain supports shared context through:
    - Memory objects shared across agents
    - Vector stores for shared knowledge
    - Custom state management
    - LangGraph for workflow state
    - Integration with databases

Key Features:
    1. Concurrent Access
       - Multiple readers
       - Controlled writers
       - Lock management
       - Deadlock prevention
       - Fair scheduling
       - Priority access

    2. Consistency Models
       - Strong consistency
       - Eventual consistency
       - Causal consistency
       - Read-your-writes
       - Monotonic reads
       - Application-specific

    3. Change Management
       - Version tracking
       - Change logs
       - Rollback capability
       - Diff computation
       - Merge operations
       - Conflict resolution

    4. Scalability
       - Partitioning
       - Replication
       - Caching
       - Lazy loading
       - Efficient indexing
       - Distributed storage

Best Practices:
    1. Data Organization
       - Clear naming conventions
       - Logical structure
       - Minimize shared state
       - Immutable where possible
       - Document schemas
       - Version data formats

    2. Concurrency Control
       - Use appropriate locking
       - Keep critical sections small
       - Avoid long-held locks
       - Handle deadlocks
       - Implement timeouts
       - Test concurrent scenarios

    3. Performance
       - Cache frequently read data
       - Batch operations
       - Async notifications
       - Index for queries
       - Minimize contention
       - Monitor bottlenecks

    4. Reliability
       - Persist important data
       - Backup regularly
       - Handle failures gracefully
       - Validate writes
       - Log operations
       - Support recovery

Trade-offs:
    Advantages:
    - Simple coordination
    - Shared visibility
    - Flexible collaboration
    - Natural data sharing
    - Easy debugging
    - Common patterns

    Disadvantages:
    - Potential bottleneck
    - Concurrency complexity
    - Consistency challenges
    - Scalability limits
    - Testing difficulty
    - State management overhead

Production Considerations:
    1. Storage Backend
       - Redis: Fast, in-memory
       - MongoDB: Document store
       - PostgreSQL: Relational, ACID
       - DynamoDB: Managed, scalable
       - Etcd: Distributed coordination
       - Memcached: Simple caching

    2. Consistency Strategy
       - ACID transactions for critical data
       - Eventual consistency for non-critical
       - CRDTs for conflict-free updates
       - Last-write-wins for simple cases
       - Application-level resolution
       - Version vectors for causality

    3. Performance Optimization
       - Read-through caching
       - Write-behind buffering
       - Optimistic locking
       - Sharding by key
       - Denormalization
       - Materialized views

    4. Monitoring
       - Access patterns
       - Lock contention
       - Query performance
       - Storage usage
       - Conflict rate
       - Error frequency

    5. Security
       - Authentication
       - Authorization (ACLs)
       - Encryption at rest
       - Audit logging
       - Input validation
       - Rate limiting
"""

import os
import json
import time
import threading
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
from copy import deepcopy
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class AccessMode(Enum):
    """Access modes for workspace resources"""
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


@dataclass
class WorkspaceEntry:
    """Entry in the shared workspace"""
    key: str
    value: Any
    version: int
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkspaceEvent:
    """Event notification for workspace changes"""
    event_type: str  # created, updated, deleted
    key: str
    value: Any
    agent_id: str
    timestamp: datetime


class SharedWorkspace:
    """
    Shared workspace for multi-agent collaboration.
    
    Provides concurrent access with locking, versioning, and change notifications.
    """
    
    def __init__(self):
        """Initialize shared workspace"""
        self.data: Dict[str, WorkspaceEntry] = {}
        self.locks: Dict[str, str] = {}  # key -> agent_id holding lock
        self.subscribers: Dict[str, Set[Callable]] = defaultdict(set)  # key -> callbacks
        self.history: List[WorkspaceEvent] = []
        self.lock = threading.RLock()  # Reentrant lock for thread safety
    
    def read(self, key: str, default: Any = None) -> Any:
        """
        Read value from workspace.
        
        Args:
            key: Key to read
            default: Default value if key doesn't exist
            
        Returns:
            Value or default
        """
        with self.lock:
            entry = self.data.get(key)
            if entry:
                return deepcopy(entry.value)
            return default
    
    def write(self, key: str, value: Any, agent_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Write value to workspace.
        
        Args:
            key: Key to write
            value: Value to write
            agent_id: Agent performing write
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        with self.lock:
            now = datetime.now()
            
            if key in self.data:
                # Update existing
                entry = self.data[key]
                entry.value = value
                entry.version += 1
                entry.updated_at = now
                entry.updated_by = agent_id
                if metadata:
                    entry.metadata.update(metadata)
                event_type = "updated"
            else:
                # Create new
                entry = WorkspaceEntry(
                    key=key,
                    value=value,
                    version=1,
                    created_at=now,
                    updated_at=now,
                    created_by=agent_id,
                    updated_by=agent_id,
                    metadata=metadata or {}
                )
                self.data[key] = entry
                event_type = "created"
            
            # Record event
            event = WorkspaceEvent(
                event_type=event_type,
                key=key,
                value=value,
                agent_id=agent_id,
                timestamp=now
            )
            self.history.append(event)
            
            # Notify subscribers
            self._notify_subscribers(key, event)
            
            return True
    
    def delete(self, key: str, agent_id: str) -> bool:
        """
        Delete value from workspace.
        
        Args:
            key: Key to delete
            agent_id: Agent performing deletion
            
        Returns:
            True if deleted, False if not found
        """
        with self.lock:
            if key in self.data:
                value = self.data[key].value
                del self.data[key]
                
                # Record event
                event = WorkspaceEvent(
                    event_type="deleted",
                    key=key,
                    value=value,
                    agent_id=agent_id,
                    timestamp=datetime.now()
                )
                self.history.append(event)
                
                # Notify subscribers
                self._notify_subscribers(key, event)
                
                return True
            return False
    
    def update(self, key: str, updater: Callable[[Any], Any], agent_id: str) -> bool:
        """
        Atomically update value using updater function.
        
        Args:
            key: Key to update
            updater: Function that takes current value and returns new value
            agent_id: Agent performing update
            
        Returns:
            True if successful
        """
        with self.lock:
            current = self.read(key)
            new_value = updater(current)
            return self.write(key, new_value, agent_id)
    
    def lock_key(self, key: str, agent_id: str, timeout: float = 5.0) -> bool:
        """
        Acquire exclusive lock on key.
        
        Args:
            key: Key to lock
            agent_id: Agent requesting lock
            timeout: Maximum time to wait for lock
            
        Returns:
            True if lock acquired
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                if key not in self.locks or self.locks[key] == agent_id:
                    self.locks[key] = agent_id
                    return True
            time.sleep(0.01)
        
        return False
    
    def unlock_key(self, key: str, agent_id: str) -> bool:
        """
        Release lock on key.
        
        Args:
            key: Key to unlock
            agent_id: Agent releasing lock
            
        Returns:
            True if unlocked
        """
        with self.lock:
            if key in self.locks and self.locks[key] == agent_id:
                del self.locks[key]
                return True
            return False
    
    def subscribe(self, key: str, callback: Callable[[WorkspaceEvent], None]):
        """
        Subscribe to changes on a key.
        
        Args:
            key: Key to watch
            callback: Function to call on changes
        """
        with self.lock:
            self.subscribers[key].add(callback)
    
    def unsubscribe(self, key: str, callback: Callable[[WorkspaceEvent], None]):
        """
        Unsubscribe from changes on a key.
        
        Args:
            key: Key to stop watching
            callback: Callback to remove
        """
        with self.lock:
            if key in self.subscribers:
                self.subscribers[key].discard(callback)
    
    def _notify_subscribers(self, key: str, event: WorkspaceEvent):
        """
        Notify subscribers of change.
        
        Args:
            key: Key that changed
            event: Event details
        """
        # Notify exact key subscribers
        for callback in self.subscribers.get(key, []):
            try:
                callback(event)
            except Exception as e:
                print(f"Error in subscriber callback: {e}")
        
        # Notify wildcard subscribers
        for callback in self.subscribers.get("*", []):
            try:
                callback(event)
            except Exception as e:
                print(f"Error in wildcard subscriber callback: {e}")
    
    def get_all_keys(self) -> List[str]:
        """Get all keys in workspace"""
        with self.lock:
            return list(self.data.keys())
    
    def get_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a key.
        
        Args:
            key: Key to get info for
            
        Returns:
            Info dictionary or None
        """
        with self.lock:
            if key in self.data:
                entry = self.data[key]
                return {
                    'key': entry.key,
                    'version': entry.version,
                    'created_at': entry.created_at.isoformat(),
                    'updated_at': entry.updated_at.isoformat(),
                    'created_by': entry.created_by,
                    'updated_by': entry.updated_by,
                    'metadata': entry.metadata,
                    'is_locked': key in self.locks,
                    'locked_by': self.locks.get(key)
                }
            return None
    
    def get_history(self, key: Optional[str] = None, limit: int = 10) -> List[WorkspaceEvent]:
        """
        Get history of changes.
        
        Args:
            key: Filter by key (None for all)
            limit: Maximum events to return
            
        Returns:
            List of events
        """
        with self.lock:
            if key:
                events = [e for e in self.history if e.key == key]
            else:
                events = self.history
            return events[-limit:]
    
    def clear(self):
        """Clear all data from workspace"""
        with self.lock:
            self.data.clear()
            self.locks.clear()
            self.history.clear()


class CollaborativeAgent:
    """
    Agent that works with shared workspace.
    
    Demonstrates collaborative problem solving.
    """
    
    def __init__(self, agent_id: str, workspace: SharedWorkspace):
        """
        Initialize collaborative agent.
        
        Args:
            agent_id: Unique agent identifier
            workspace: Shared workspace
        """
        self.agent_id = agent_id
        self.workspace = workspace
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    def contribute(self, key: str, contribution: Any):
        """
        Add contribution to workspace.
        
        Args:
            key: Key to write to
            contribution: Data to contribute
        """
        print(f"[{self.agent_id}] Contributing to '{key}'")
        self.workspace.write(key, contribution, self.agent_id)
    
    def read_contribution(self, key: str) -> Any:
        """
        Read contribution from workspace.
        
        Args:
            key: Key to read
            
        Returns:
            Value or None
        """
        value = self.workspace.read(key)
        if value is not None:
            print(f"[{self.agent_id}] Read '{key}': {value}")
        return value
    
    def append_to_list(self, key: str, item: Any):
        """
        Append item to a list in workspace.
        
        Args:
            key: Key of list
            item: Item to append
        """
        print(f"[{self.agent_id}] Appending to '{key}'")
        
        def updater(current):
            if current is None:
                return [item]
            elif isinstance(current, list):
                return current + [item]
            else:
                return [current, item]
        
        self.workspace.update(key, updater, self.agent_id)
    
    def increment_counter(self, key: str, amount: int = 1):
        """
        Atomically increment a counter.
        
        Args:
            key: Key of counter
            amount: Amount to increment
        """
        print(f"[{self.agent_id}] Incrementing '{key}' by {amount}")
        
        def updater(current):
            if current is None:
                return amount
            return current + amount
        
        self.workspace.update(key, updater, self.agent_id)
    
    def locked_operation(self, key: str, operation: Callable):
        """
        Perform operation with exclusive lock.
        
        Args:
            key: Key to lock
            operation: Operation to perform
        """
        if self.workspace.lock_key(key, self.agent_id):
            try:
                print(f"[{self.agent_id}] Acquired lock on '{key}'")
                operation()
            finally:
                self.workspace.unlock_key(key, self.agent_id)
                print(f"[{self.agent_id}] Released lock on '{key}'")
        else:
            print(f"[{self.agent_id}] Failed to acquire lock on '{key}'")


def demonstrate_shared_workspace():
    """Demonstrate shared context/workspace pattern"""
    print("=" * 80)
    print("SHARED CONTEXT/WORKSPACE DEMONSTRATION")
    print("=" * 80)
    
    workspace = SharedWorkspace()
    
    # Example 1: Basic Read/Write
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Read/Write Operations")
    print("=" * 80)
    
    agent1 = CollaborativeAgent("agent1", workspace)
    agent2 = CollaborativeAgent("agent2", workspace)
    
    print("\nAgent1 writes, Agent2 reads:")
    agent1.contribute("shared_data", {"status": "initialized", "count": 0})
    data = agent2.read_contribution("shared_data")
    
    # Example 2: Collaborative List Building
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Collaborative List Building")
    print("=" * 80)
    
    agent3 = CollaborativeAgent("agent3", workspace)
    
    print("\nMultiple agents appending to shared list:")
    agent1.append_to_list("tasks", "Analyze data")
    agent2.append_to_list("tasks", "Generate report")
    agent3.append_to_list("tasks", "Send notifications")
    
    tasks = workspace.read("tasks")
    print(f"\nFinal task list: {tasks}")
    
    # Example 3: Atomic Counter Updates
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Atomic Counter Updates")
    print("=" * 80)
    
    print("\nMultiple agents incrementing shared counter:")
    workspace.write("counter", 0, "system")
    
    agent1.increment_counter("counter", 5)
    agent2.increment_counter("counter", 3)
    agent3.increment_counter("counter", 7)
    
    final_count = workspace.read("counter")
    print(f"\nFinal counter value: {final_count}")
    
    # Example 4: Locking for Critical Sections
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Exclusive Locking")
    print("=" * 80)
    
    workspace.write("resource", {"balance": 100}, "system")
    
    def withdraw_funds():
        resource = workspace.read("resource")
        if resource["balance"] >= 30:
            resource["balance"] -= 30
            workspace.write("resource", resource, agent1.agent_id)
            print(f"[{agent1.agent_id}] Withdrew 30, new balance: {resource['balance']}")
    
    print("\nAgent1 performing locked operation:")
    agent1.locked_operation("resource", withdraw_funds)
    
    # Example 5: Change Notifications
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Change Notifications (Subscriptions)")
    print("=" * 80)
    
    def on_status_change(event: WorkspaceEvent):
        print(f"[Subscriber] '{event.key}' {event.event_type} by {event.agent_id}: {event.value}")
    
    print("\nSubscribing to 'system_status' changes:")
    workspace.subscribe("system_status", on_status_change)
    
    agent1.contribute("system_status", "starting")
    time.sleep(0.1)
    agent2.contribute("system_status", "running")
    time.sleep(0.1)
    agent3.contribute("system_status", "completed")
    time.sleep(0.1)
    
    # Example 6: Versioning and Metadata
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Versioning and Metadata")
    print("=" * 80)
    
    print("\nTracking versions of shared document:")
    workspace.write("document", "Version 1 content", "agent1")
    print(f"Info: {workspace.get_info('document')}")
    
    time.sleep(0.1)
    workspace.write("document", "Version 2 content", "agent2", {"reviewed": True})
    print(f"Info: {workspace.get_info('document')}")
    
    time.sleep(0.1)
    workspace.write("document", "Version 3 content", "agent3")
    print(f"Info: {workspace.get_info('document')}")
    
    # Example 7: History Tracking
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Change History")
    print("=" * 80)
    
    history = workspace.get_history("document")
    print(f"\nHistory for 'document' ({len(history)} events):")
    for event in history:
        print(f"  {event.timestamp.strftime('%H:%M:%S')} - {event.event_type} by {event.agent_id}")
    
    # Example 8: Workspace Overview
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Workspace Overview")
    print("=" * 80)
    
    all_keys = workspace.get_all_keys()
    print(f"\nWorkspace contains {len(all_keys)} keys:")
    for key in all_keys:
        info = workspace.get_info(key)
        print(f"  - {key}: v{info['version']}, updated by {info['updated_by']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SHARED CONTEXT/WORKSPACE SUMMARY")
    print("=" * 80)
    print("""
Shared Workspace Benefits:
1. Centralized State: Single source of truth
2. Easy Coordination: Natural collaboration point
3. Flexible Access: Read/write as needed
4. Transparency: All agents see same data
5. Simple Model: Familiar data structure
6. Change Tracking: Version history

Key Components:
1. Workspace
   - Key-value storage
   - Thread-safe operations
   - Version tracking
   - Change history
   - Notification system
   - Lock management

2. Operations
   - read(key): Get value
   - write(key, value): Set value
   - update(key, updater): Atomic update
   - delete(key): Remove value
   - lock/unlock: Exclusive access
   - subscribe: Watch for changes

3. Concurrency Control
   - Thread-safe operations
   - Optimistic updates (default)
   - Pessimistic locking (when needed)
   - Atomic operations
   - Version tracking
   - Conflict detection

Use Cases:
1. Collaborative Problem Solving
   - Shared scratchpad
   - Incremental solutions
   - Knowledge accumulation
   - Team coordination

2. Workflow State
   - Task tracking
   - Progress monitoring
   - Status updates
   - Result sharing

3. Knowledge Base
   - Shared facts
   - Learned patterns
   - Best practices
   - Historical data

4. Resource Management
   - Resource allocation
   - Capacity tracking
   - Scheduling
   - Conflict resolution

Design Patterns:
1. Shared Scratchpad
   - Agents contribute ideas
   - Build on each other's work
   - Iterative refinement
   - Collective intelligence

2. Blackboard Pattern
   - Problem-solving workspace
   - Multiple experts contribute
   - Control component coordinates
   - Solution emerges

3. Task Board
   - Todo/In Progress/Done
   - Agents claim tasks
   - Update status
   - Track completion

4. Data Pipeline
   - Input → Transform → Output
   - Each agent processes
   - Results in workspace
   - Next agent consumes

Concurrency Strategies:
1. No Locking (Simple)
   - For non-critical data
   - Last write wins
   - Fast, simple
   - Risk of conflicts

2. Optimistic Locking
   - Read freely
   - Check version on write
   - Retry on conflict
   - Good for low contention

3. Pessimistic Locking
   - Lock before read/write
   - Hold during operation
   - Unlock after
   - Good for critical sections

4. Atomic Operations
   - Single operation
   - All or nothing
   - No intermediate state
   - Best for counters, lists

Best Practices:
1. Data Organization
   ✓ Logical key structure
   ✓ Minimize shared state
   ✓ Clear ownership
   ✓ Document schemas
   ✓ Version formats
   ✓ Clean up old data

2. Concurrency
   ✓ Keep locks short
   ✓ Use atomic operations
   ✓ Handle conflicts
   ✓ Avoid deadlocks
   ✓ Test concurrent access
   ✓ Monitor contention

3. Performance
   ✓ Cache frequently read
   ✓ Batch updates
   ✓ Index for queries
   ✓ Lazy loading
   ✓ Minimize writes
   ✓ Async notifications

4. Reliability
   ✓ Persist critical data
   ✓ Backup regularly
   ✓ Validate writes
   ✓ Handle failures
   ✓ Log operations
   ✓ Support rollback

Common Pitfalls:
❌ Too much shared state (tight coupling)
✓ Share only necessary data

❌ Long-held locks (deadlocks)
✓ Minimize critical sections

❌ No conflict handling
✓ Implement retry logic

❌ Missing validation
✓ Validate all writes

❌ No access control
✓ Implement permissions

Production Considerations:
1. Storage Backend
   - Redis: Fast, in-memory
   - MongoDB: Document store
   - PostgreSQL: ACID compliance
   - Memcached: Simple cache
   - DynamoDB: Scalable cloud
   - Etcd: Distributed config

2. Consistency Model
   - Strong: All see same value
   - Eventual: Converges over time
   - Causal: Cause before effect
   - Read-your-writes: See own changes
   - Choose based on needs

3. Scalability
   - Partition by key
   - Replicate for reads
   - Cache hot data
   - Shard large datasets
   - Horizontal scaling
   - Load balancing

4. Monitoring
   - Access patterns
   - Lock contention
   - Read/write ratios
   - Storage usage
   - Conflict rate
   - Performance metrics

When to Use:
✓ Multi-agent collaboration
✓ Shared state needed
✓ Flexible data sharing
✓ Simple coordination
✓ Transparent operations
✓ Natural fit for problem
✗ No shared state needed
✗ Message passing sufficient
✗ Ultra-high performance critical
✗ Complex distributed system

Alternatives:
- Message Passing: Decoupled, async
- Event Sourcing: Immutable events
- Tuple Spaces: Linda-style coordination
- Distributed Cache: Geographically distributed
- Blockchain: Immutable, consensus

ROI Analysis:
- Simplicity: High (natural pattern)
- Performance: Medium (can be bottleneck)
- Scalability: Medium (single point)
- Reliability: High (with proper backend)
- Maintainability: High (centralized state)
- Cost: Low (simple infrastructure)
""")
    
    print("\n" + "=" * 80)
    print("Pattern 098 (Shared Context/Workspace) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_shared_workspace()
