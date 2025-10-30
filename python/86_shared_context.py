"""
Shared Context/Workspace Pattern

Enables multiple agents to collaborate by sharing a common workspace where they
can read and write information, coordinate activities, and build on each other's work.

Key Concepts:
- Shared memory space
- Collaborative editing
- Conflict resolution
- Access control
- State synchronization

Use Cases:
- Collaborative problem solving
- Multi-agent workflows
- Distributed knowledge building
- Team coordination
"""

from typing import List, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
import uuid


class AccessLevel(Enum):
    """Access levels for workspace resources."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class ResourceType(Enum):
    """Types of shared resources."""
    DOCUMENT = "document"
    DATA = "data"
    TASK = "task"
    KNOWLEDGE = "knowledge"
    ARTIFACT = "artifact"


class LockType(Enum):
    """Types of resource locks."""
    NONE = "none"
    READ_LOCK = "read"
    WRITE_LOCK = "write"
    EXCLUSIVE = "exclusive"


@dataclass
class SharedResource:
    """Represents a resource in the shared workspace."""
    resource_id: str
    name: str
    resource_type: ResourceType
    content: Any
    owner_id: str
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    lock_type: LockType = LockType.NONE
    locked_by: Optional[str] = None
    
    def update_content(self, new_content: Any, agent_id: str) -> bool:
        """Update resource content."""
        if self.lock_type == LockType.WRITE_LOCK and self.locked_by != agent_id:
            return False
        
        self.content = new_content
        self.modified_at = datetime.now()
        self.version += 1
        return True


@dataclass
class WorkspaceEvent:
    """Event in the workspace."""
    event_id: str
    event_type: str
    agent_id: str
    resource_id: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPermission:
    """Agent permissions for workspace."""
    agent_id: str
    access_level: AccessLevel
    allowed_resources: Set[str] = field(default_factory=set)  # Empty = all
    
    def can_read(self, resource_id: str) -> bool:
        """Check if agent can read resource."""
        if not self.allowed_resources:
            return True
        return resource_id in self.allowed_resources
    
    def can_write(self, resource_id: str) -> bool:
        """Check if agent can write to resource."""
        if self.access_level == AccessLevel.READ:
            return False
        return self.can_read(resource_id)


class SharedWorkspace:
    """Shared workspace for multi-agent collaboration."""
    
    def __init__(self, workspace_id: str, name: str):
        self.workspace_id = workspace_id
        self.name = name
        self.resources: Dict[str, SharedResource] = {}
        self.permissions: Dict[str, AgentPermission] = {}
        self.events: List[WorkspaceEvent] = []
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
    
    def register_agent(
        self,
        agent_id: str,
        access_level: AccessLevel = AccessLevel.WRITE
    ) -> None:
        """Register agent with workspace."""
        self.permissions[agent_id] = AgentPermission(
            agent_id=agent_id,
            access_level=access_level
        )
        self._log_event("agent_registered", agent_id, None)
    
    def create_resource(
        self,
        agent_id: str,
        name: str,
        resource_type: ResourceType,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[SharedResource]:
        """Create a new shared resource."""
        if agent_id not in self.permissions:
            return None
        
        if self.permissions[agent_id].access_level == AccessLevel.READ:
            return None
        
        resource = SharedResource(
            resource_id=str(uuid.uuid4()),
            name=name,
            resource_type=resource_type,
            content=content,
            owner_id=agent_id,
            metadata=metadata or {}
        )
        
        self.resources[resource.resource_id] = resource
        self._log_event("resource_created", agent_id, resource.resource_id)
        self._notify_subscribers("resource_created", resource)
        
        return resource
    
    def read_resource(
        self,
        agent_id: str,
        resource_id: str
    ) -> Optional[SharedResource]:
        """Read a shared resource."""
        if agent_id not in self.permissions:
            return None
        
        if resource_id not in self.resources:
            return None
        
        perm = self.permissions[agent_id]
        if not perm.can_read(resource_id):
            return None
        
        resource = self.resources[resource_id]
        self._log_event("resource_read", agent_id, resource_id)
        
        return resource
    
    def update_resource(
        self,
        agent_id: str,
        resource_id: str,
        new_content: Any
    ) -> bool:
        """Update a shared resource."""
        if agent_id not in self.permissions:
            return False
        
        if resource_id not in self.resources:
            return False
        
        perm = self.permissions[agent_id]
        if not perm.can_write(resource_id):
            return False
        
        resource = self.resources[resource_id]
        if resource.update_content(new_content, agent_id):
            self._log_event("resource_updated", agent_id, resource_id)
            self._notify_subscribers("resource_updated", resource)
            return True
        
        return False
    
    def acquire_lock(
        self,
        agent_id: str,
        resource_id: str,
        lock_type: LockType = LockType.WRITE_LOCK
    ) -> bool:
        """Acquire lock on resource."""
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        
        if resource.lock_type != LockType.NONE:
            return False  # Already locked
        
        resource.lock_type = lock_type
        resource.locked_by = agent_id
        self._log_event("lock_acquired", agent_id, resource_id)
        
        return True
    
    def release_lock(
        self,
        agent_id: str,
        resource_id: str
    ) -> bool:
        """Release lock on resource."""
        if resource_id not in self.resources:
            return False
        
        resource = self.resources[resource_id]
        
        if resource.locked_by != agent_id:
            return False
        
        resource.lock_type = LockType.NONE
        resource.locked_by = None
        self._log_event("lock_released", agent_id, resource_id)
        
        return True
    
    def query_resources(
        self,
        agent_id: str,
        resource_type: Optional[ResourceType] = None,
        owner_id: Optional[str] = None
    ) -> List[SharedResource]:
        """Query resources in workspace."""
        if agent_id not in self.permissions:
            return []
        
        perm = self.permissions[agent_id]
        results = []
        
        for resource in self.resources.values():
            if not perm.can_read(resource.resource_id):
                continue
            
            if resource_type and resource.resource_type != resource_type:
                continue
            
            if owner_id and resource.owner_id != owner_id:
                continue
            
            results.append(resource)
        
        return results
    
    def subscribe(
        self,
        event_type: str,
        callback: Callable[[SharedResource], None]
    ) -> None:
        """Subscribe to workspace events."""
        self.subscribers[event_type].append(callback)
    
    def get_workspace_state(self) -> Dict[str, Any]:
        """Get current workspace state."""
        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "total_resources": len(self.resources),
            "registered_agents": len(self.permissions),
            "total_events": len(self.events),
            "resources_by_type": self._count_by_type(),
            "locked_resources": sum(
                1 for r in self.resources.values() 
                if r.lock_type != LockType.NONE
            )
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count resources by type."""
        counts = defaultdict(int)
        for resource in self.resources.values():
            counts[resource.resource_type.value] += 1
        return dict(counts)
    
    def _log_event(
        self,
        event_type: str,
        agent_id: str,
        resource_id: Optional[str]
    ) -> None:
        """Log workspace event."""
        event = WorkspaceEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            agent_id=agent_id,
            resource_id=resource_id
        )
        self.events.append(event)
    
    def _notify_subscribers(self, event_type: str, resource: SharedResource) -> None:
        """Notify event subscribers."""
        for callback in self.subscribers.get(event_type, []):
            try:
                callback(resource)
            except Exception:
                pass  # Don't let subscriber errors break workspace


class CollaborativeAgent:
    """Agent that works in a shared workspace."""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.workspace: Optional[SharedWorkspace] = None
        self.local_cache: Dict[str, Any] = {}
    
    def join_workspace(self, workspace: SharedWorkspace) -> None:
        """Join a shared workspace."""
        self.workspace = workspace
        workspace.register_agent(self.agent_id)
        print(f"[{self.name}] Joined workspace: {workspace.name}")
    
    def create_document(
        self,
        title: str,
        content: str
    ) -> Optional[str]:
        """Create a document in workspace."""
        if not self.workspace:
            return None
        
        resource = self.workspace.create_resource(
            agent_id=self.agent_id,
            name=title,
            resource_type=ResourceType.DOCUMENT,
            content=content
        )
        
        if resource:
            print(f"[{self.name}] Created document: {title}")
            return resource.resource_id
        
        return None
    
    def read_document(self, resource_id: str) -> Optional[str]:
        """Read a document from workspace."""
        if not self.workspace:
            return None
        
        resource = self.workspace.read_resource(self.agent_id, resource_id)
        
        if resource:
            print(f"[{self.name}] Read document: {resource.name} (v{resource.version})")
            return resource.content
        
        return None
    
    def update_document(
        self,
        resource_id: str,
        new_content: str,
        use_lock: bool = True
    ) -> bool:
        """Update a document in workspace."""
        if not self.workspace:
            return False
        
        # Acquire lock if requested
        if use_lock:
            if not self.workspace.acquire_lock(
                self.agent_id,
                resource_id,
                LockType.WRITE_LOCK
            ):
                print(f"[{self.name}] Could not acquire lock")
                return False
        
        # Update content
        success = self.workspace.update_resource(
            self.agent_id,
            resource_id,
            new_content
        )
        
        # Release lock
        if use_lock:
            self.workspace.release_lock(self.agent_id, resource_id)
        
        if success:
            print(f"[{self.name}] Updated document")
        
        return success
    
    def collaborate_on_document(
        self,
        resource_id: str,
        contribution: str
    ) -> bool:
        """Add contribution to existing document."""
        if not self.workspace:
            return False
        
        # Read current content
        current_content = self.read_document(resource_id)
        if current_content is None:
            return False
        
        # Add contribution
        new_content = f"{current_content}\n\n--- Contribution by {self.name} ---\n{contribution}"
        
        # Update with lock
        return self.update_document(resource_id, new_content, use_lock=True)
    
    def list_all_documents(self) -> List[SharedResource]:
        """List all accessible documents."""
        if not self.workspace:
            return []
        
        return self.workspace.query_resources(
            self.agent_id,
            resource_type=ResourceType.DOCUMENT
        )


def demonstrate_shared_context():
    """Demonstrate shared context/workspace pattern."""
    print("=" * 60)
    print("SHARED CONTEXT/WORKSPACE DEMONSTRATION")
    print("=" * 60)
    
    # Create shared workspace
    workspace = SharedWorkspace("ws1", "Research Collaboration")
    
    # Create collaborative agents
    alice = CollaborativeAgent("alice", "Alice")
    bob = CollaborativeAgent("bob", "Bob")
    carol = CollaborativeAgent("carol", "Carol")
    
    # Subscribe to workspace events
    def on_resource_created(resource: SharedResource):
        print(f"  📢 Event: Resource '{resource.name}' created by {resource.owner_id}")
    
    def on_resource_updated(resource: SharedResource):
        print(f"  📢 Event: Resource '{resource.name}' updated (v{resource.version})")
    
    workspace.subscribe("resource_created", on_resource_created)
    workspace.subscribe("resource_updated", on_resource_updated)
    
    # Agents join workspace
    print("\n" + "=" * 60)
    print("1. Agents Joining Workspace")
    print("=" * 60)
    
    alice.join_workspace(workspace)
    bob.join_workspace(workspace)
    carol.join_workspace(workspace)
    
    # Alice creates initial document
    print("\n" + "=" * 60)
    print("2. Creating Shared Document")
    print("=" * 60)
    
    doc_id = alice.create_document(
        "Research Proposal",
        "# AI Research Proposal\n\nTopic: Multi-Agent Collaboration Systems"
    )
    
    if not doc_id:
        print("Failed to create document")
        return
    
    # Bob reads and contributes
    print("\n" + "=" * 60)
    print("3. Collaborative Editing")
    print("=" * 60)
    
    content = bob.read_document(doc_id)
    if content:
        print(f"\nBob reads content:\n{content[:50]}...")
    
    bob.collaborate_on_document(
        doc_id,
        "Methodology: We will implement shared workspace patterns\nwith conflict resolution mechanisms."
    )
    
    # Carol adds her contribution
    carol.collaborate_on_document(
        doc_id,
        "Expected Outcomes:\n- Improved agent coordination\n- Reduced communication overhead\n- Better knowledge sharing"
    )
    
    # Alice reviews final document
    print("\n" + "=" * 60)
    print("4. Final Document Review")
    print("=" * 60)
    
    final_content = alice.read_document(doc_id)
    if final_content:
        print(f"\nFinal document:\n{final_content}")

    
    # List all documents
    print("\n" + "=" * 60)
    print("5. Workspace Summary")
    print("=" * 60)
    
    docs = alice.list_all_documents()
    print(f"\nTotal documents: {len(docs)}")
    for doc in docs:
        print(f"  - {doc.name} (v{doc.version}, by {doc.owner_id})")
    
    # Workspace state
    state = workspace.get_workspace_state()
    print(f"\nWorkspace State:")
    print(f"  Total resources: {state['total_resources']}")
    print(f"  Registered agents: {state['registered_agents']}")
    print(f"  Total events: {state['total_events']}")
    print(f"  Resources by type: {state['resources_by_type']}")


if __name__ == "__main__":
    demonstrate_shared_context()
