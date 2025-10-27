"""
Agentic Design Pattern: Permission & Authorization

This pattern implements role-based access control (RBAC) and fine-grained permission
management for agent actions. The agent checks permissions before executing actions,
manages roles and privileges, and enforces authorization policies.

Category: Control & Governance
Use Cases:
- Multi-user agent systems
- Enterprise access control
- Resource protection
- Hierarchical permission models
- Delegation and temporary access
- Principle of least privilege enforcement
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import random


class PermissionLevel(Enum):
    """Permission access levels"""
    NONE = 0
    READ = 1
    WRITE = 2
    EXECUTE = 3
    ADMIN = 4
    OWNER = 5


class AccessDecision(Enum):
    """Authorization decision outcomes"""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"  # Requires additional verification


class ResourceType(Enum):
    """Types of resources that can be protected"""
    DATA = "data"
    SERVICE = "service"
    FUNCTION = "function"
    CONFIGURATION = "configuration"
    USER = "user"
    SYSTEM = "system"


@dataclass
class Permission:
    """Represents a specific permission"""
    permission_id: str
    name: str
    description: str
    resource_type: ResourceType
    level: PermissionLevel
    actions: List[str]  # Specific actions allowed (read, write, delete, etc.)
    conditions: Optional[Dict[str, Any]] = None  # Additional conditions


@dataclass
class Role:
    """Represents a role with associated permissions"""
    role_id: str
    name: str
    description: str
    permissions: Set[str]  # Set of permission IDs
    inherits_from: Optional[List[str]] = None  # Role inheritance
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class User:
    """Represents a user with roles and permissions"""
    user_id: str
    username: str
    roles: Set[str]  # Set of role IDs
    direct_permissions: Set[str] = field(default_factory=set)  # Direct permission grants
    attributes: Dict[str, Any] = field(default_factory=dict)
    active: bool = True


@dataclass
class AccessRequest:
    """Represents an access request"""
    request_id: str
    user_id: str
    action: str
    resource: str
    resource_type: ResourceType
    context: Dict[str, Any]
    timestamp: datetime


@dataclass
class AccessDecisionRecord:
    """Records an authorization decision"""
    decision_id: str
    request: AccessRequest
    decision: AccessDecision
    reason: str
    evaluated_permissions: List[str]
    timestamp: datetime
    conditions_met: bool = True


@dataclass
class TemporaryAccess:
    """Represents temporary access grant"""
    grant_id: str
    user_id: str
    permission_id: str
    granted_by: str
    granted_at: datetime
    expires_at: datetime
    reason: str
    active: bool = True


class PermissionRegistry:
    """Manages permission definitions"""
    
    def __init__(self):
        self.permissions: Dict[str, Permission] = {}
    
    def register_permission(self, permission: Permission) -> None:
        """Register a new permission"""
        self.permissions[permission.permission_id] = permission
    
    def get_permission(self, permission_id: str) -> Optional[Permission]:
        """Get a permission by ID"""
        return self.permissions.get(permission_id)
    
    def find_permissions_for_action(self, action: str, resource_type: ResourceType) -> List[Permission]:
        """Find permissions that allow a specific action"""
        matching = []
        for perm in self.permissions.values():
            if perm.resource_type == resource_type and action in perm.actions:
                matching.append(perm)
        return matching


class RoleManager:
    """Manages roles and role-based access control"""
    
    def __init__(self, permission_registry: PermissionRegistry):
        self.roles: Dict[str, Role] = {}
        self.permission_registry = permission_registry
    
    def create_role(self, role: Role) -> None:
        """Create a new role"""
        self.roles[role.role_id] = role
    
    def add_permission_to_role(self, role_id: str, permission_id: str) -> bool:
        """Add a permission to a role"""
        if role_id not in self.roles:
            return False
        
        if permission_id not in self.permission_registry.permissions:
            return False
        
        self.roles[role_id].permissions.add(permission_id)
        return True
    
    def get_role_permissions(self, role_id: str, include_inherited: bool = True) -> Set[str]:
        """Get all permissions for a role"""
        if role_id not in self.roles:
            return set()
        
        role = self.roles[role_id]
        permissions = role.permissions.copy()
        
        # Add inherited permissions
        if include_inherited and role.inherits_from:
            for parent_role_id in role.inherits_from:
                permissions.update(self.get_role_permissions(parent_role_id, True))
        
        return permissions
    
    def check_role_hierarchy(self, role_id: str, target_role_id: str) -> bool:
        """Check if role_id inherits from target_role_id"""
        if role_id not in self.roles:
            return False
        
        role = self.roles[role_id]
        
        if not role.inherits_from:
            return False
        
        if target_role_id in role.inherits_from:
            return True
        
        # Check recursively
        for parent_role_id in role.inherits_from:
            if self.check_role_hierarchy(parent_role_id, target_role_id):
                return True
        
        return False


class UserManager:
    """Manages users and their role assignments"""
    
    def __init__(self, role_manager: RoleManager):
        self.users: Dict[str, User] = {}
        self.role_manager = role_manager
    
    def register_user(self, user: User) -> None:
        """Register a new user"""
        self.users[user.user_id] = user
    
    def assign_role(self, user_id: str, role_id: str) -> bool:
        """Assign a role to a user"""
        if user_id not in self.users:
            return False
        
        if role_id not in self.role_manager.roles:
            return False
        
        self.users[user_id].roles.add(role_id)
        return True
    
    def revoke_role(self, user_id: str, role_id: str) -> bool:
        """Revoke a role from a user"""
        if user_id not in self.users:
            return False
        
        if role_id in self.users[user_id].roles:
            self.users[user_id].roles.remove(role_id)
            return True
        
        return False
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user (from roles and direct grants)"""
        if user_id not in self.users:
            return set()
        
        user = self.users[user_id]
        permissions = user.direct_permissions.copy()
        
        # Add permissions from all roles
        for role_id in user.roles:
            permissions.update(self.role_manager.get_role_permissions(role_id))
        
        return permissions
    
    def grant_direct_permission(self, user_id: str, permission_id: str) -> bool:
        """Grant a permission directly to a user"""
        if user_id not in self.users:
            return False
        
        self.users[user_id].direct_permissions.add(permission_id)
        return True


class AuthorizationEngine:
    """Core authorization decision engine"""
    
    def __init__(self, 
                 permission_registry: PermissionRegistry,
                 role_manager: RoleManager,
                 user_manager: UserManager):
        self.permission_registry = permission_registry
        self.role_manager = role_manager
        self.user_manager = user_manager
        self.decision_log: List[AccessDecisionRecord] = []
    
    def authorize(self, request: AccessRequest) -> AccessDecisionRecord:
        """Make an authorization decision"""
        
        # Check if user exists and is active
        user = self.user_manager.users.get(request.user_id)
        if not user or not user.active:
            return self._create_decision(request, AccessDecision.DENY, "User not found or inactive", [])
        
        # Get user's permissions
        user_permissions = self.user_manager.get_user_permissions(request.user_id)
        
        # Find permissions that could allow this action
        matching_permissions = self.permission_registry.find_permissions_for_action(
            request.action, 
            request.resource_type
        )
        
        # Check if user has any of the required permissions
        allowed_permissions = []
        for perm in matching_permissions:
            if perm.permission_id in user_permissions:
                # Check conditions if present
                if self._check_conditions(perm, request, user):
                    allowed_permissions.append(perm.permission_id)
        
        # Make decision
        if allowed_permissions:
            decision = AccessDecision.ALLOW
            reason = f"User has required permissions: {', '.join(allowed_permissions)}"
            conditions_met = True
        else:
            decision = AccessDecision.DENY
            reason = "User lacks required permissions for this action"
            conditions_met = False
        
        return self._create_decision(request, decision, reason, allowed_permissions, conditions_met)
    
    def _check_conditions(self, permission: Permission, request: AccessRequest, user: User) -> bool:
        """Check if permission conditions are met"""
        if not permission.conditions:
            return True
        
        conditions = permission.conditions
        
        # Check time-based conditions
        if "allowed_hours" in conditions:
            current_hour = datetime.now().hour
            if current_hour not in conditions["allowed_hours"]:
                return False
        
        # Check attribute-based conditions
        if "required_attributes" in conditions:
            for attr, value in conditions["required_attributes"].items():
                if user.attributes.get(attr) != value:
                    return False
        
        # Check resource-based conditions
        if "resource_pattern" in conditions:
            pattern = conditions["resource_pattern"]
            if not request.resource.startswith(pattern):
                return False
        
        return True
    
    def _create_decision(self, 
                        request: AccessRequest,
                        decision: AccessDecision,
                        reason: str,
                        evaluated_permissions: List[str],
                        conditions_met: bool = True) -> AccessDecisionRecord:
        """Create and log a decision record"""
        record = AccessDecisionRecord(
            decision_id=self._generate_id(),
            request=request,
            decision=decision,
            reason=reason,
            evaluated_permissions=evaluated_permissions,
            timestamp=datetime.now(),
            conditions_met=conditions_met
        )
        
        self.decision_log.append(record)
        return record
    
    def get_decision_history(self, user_id: Optional[str] = None) -> List[AccessDecisionRecord]:
        """Get authorization decision history"""
        if user_id:
            return [d for d in self.decision_log if d.request.user_id == user_id]
        return self.decision_log
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        return hashlib.md5(f"{datetime.now()}{random.random()}".encode()).hexdigest()[:12]


class TemporaryAccessManager:
    """Manages temporary access grants"""
    
    def __init__(self, user_manager: UserManager):
        self.user_manager = user_manager
        self.temporary_grants: Dict[str, TemporaryAccess] = {}
    
    def grant_temporary_access(self,
                              user_id: str,
                              permission_id: str,
                              granted_by: str,
                              duration_hours: int,
                              reason: str) -> Optional[TemporaryAccess]:
        """Grant temporary access to a user"""
        
        if user_id not in self.user_manager.users:
            return None
        
        grant = TemporaryAccess(
            grant_id=self._generate_id(),
            user_id=user_id,
            permission_id=permission_id,
            granted_by=granted_by,
            granted_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=duration_hours),
            reason=reason,
            active=True
        )
        
        self.temporary_grants[grant.grant_id] = grant
        
        # Actually grant the permission
        self.user_manager.grant_direct_permission(user_id, permission_id)
        
        return grant
    
    def revoke_temporary_access(self, grant_id: str) -> bool:
        """Revoke a temporary access grant"""
        if grant_id not in self.temporary_grants:
            return False
        
        grant = self.temporary_grants[grant_id]
        grant.active = False
        
        # Remove the permission from user
        user = self.user_manager.users.get(grant.user_id)
        if user and grant.permission_id in user.direct_permissions:
            user.direct_permissions.remove(grant.permission_id)
        
        return True
    
    def cleanup_expired_grants(self) -> int:
        """Remove expired temporary access grants"""
        now = datetime.now()
        expired = []
        
        for grant_id, grant in self.temporary_grants.items():
            if grant.active and grant.expires_at < now:
                expired.append(grant_id)
        
        for grant_id in expired:
            self.revoke_temporary_access(grant_id)
        
        return len(expired)
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        return hashlib.md5(f"{datetime.now()}{random.random()}".encode()).hexdigest()[:12]


class PermissionAuthorizationAgent:
    """
    Main agent for permission and authorization management
    
    Responsibilities:
    - Manage roles and permissions
    - Check authorization for actions
    - Grant and revoke access
    - Handle temporary access
    - Audit authorization decisions
    """
    
    def __init__(self):
        self.permission_registry = PermissionRegistry()
        self.role_manager = RoleManager(self.permission_registry)
        self.user_manager = UserManager(self.role_manager)
        self.auth_engine = AuthorizationEngine(
            self.permission_registry,
            self.role_manager,
            self.user_manager
        )
        self.temp_access_manager = TemporaryAccessManager(self.user_manager)
    
    def register_permission(self, permission: Permission) -> None:
        """Register a new permission"""
        self.permission_registry.register_permission(permission)
        print(f"✓ Registered permission: {permission.name}")
    
    def create_role(self, role: Role) -> None:
        """Create a new role"""
        self.role_manager.create_role(role)
        print(f"✓ Created role: {role.name}")
    
    def register_user(self, user: User) -> None:
        """Register a new user"""
        self.user_manager.register_user(user)
        print(f"✓ Registered user: {user.username}")
    
    def assign_role(self, user_id: str, role_id: str) -> bool:
        """Assign a role to a user"""
        success = self.user_manager.assign_role(user_id, role_id)
        if success:
            print(f"✓ Assigned role {role_id} to user {user_id}")
        return success
    
    def check_access(self, user_id: str, action: str, resource: str, 
                    resource_type: ResourceType, context: Optional[Dict[str, Any]] = None) -> AccessDecisionRecord:
        """Check if user has access to perform action on resource"""
        
        request = AccessRequest(
            request_id=self._generate_id(),
            user_id=user_id,
            action=action,
            resource=resource,
            resource_type=resource_type,
            context=context or {},
            timestamp=datetime.now()
        )
        
        return self.auth_engine.authorize(request)
    
    def grant_temporary_access(self, user_id: str, permission_id: str,
                              granted_by: str, duration_hours: int, reason: str) -> Optional[TemporaryAccess]:
        """Grant temporary access to a user"""
        return self.temp_access_manager.grant_temporary_access(
            user_id, permission_id, granted_by, duration_hours, reason
        )
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user"""
        return self.user_manager.get_user_permissions(user_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get authorization statistics"""
        decisions = self.auth_engine.decision_log
        
        total_decisions = len(decisions)
        allowed = sum(1 for d in decisions if d.decision == AccessDecision.ALLOW)
        denied = sum(1 for d in decisions if d.decision == AccessDecision.DENY)
        
        active_temp_grants = sum(1 for g in self.temp_access_manager.temporary_grants.values() if g.active)
        
        return {
            "total_permissions": len(self.permission_registry.permissions),
            "total_roles": len(self.role_manager.roles),
            "total_users": len(self.user_manager.users),
            "active_users": sum(1 for u in self.user_manager.users.values() if u.active),
            "total_authorization_decisions": total_decisions,
            "decisions_allowed": allowed,
            "decisions_denied": denied,
            "allow_rate": round(allowed / total_decisions, 3) if total_decisions > 0 else 0,
            "active_temporary_grants": active_temp_grants
        }
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        return hashlib.md5(f"{datetime.now()}{random.random()}".encode()).hexdigest()[:12]


def demonstrate_permission_authorization():
    """Demonstrate the permission and authorization pattern"""
    
    print("=" * 60)
    print("Permission & Authorization Agent Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = PermissionAuthorizationAgent()
    
    # Setup permissions
    print("\n1. Setting Up Permissions")
    print("-" * 60)
    
    read_perm = Permission(
        permission_id="perm_read_data",
        name="Read Data",
        description="Allows reading data",
        resource_type=ResourceType.DATA,
        level=PermissionLevel.READ,
        actions=["read", "view"]
    )
    agent.register_permission(read_perm)
    
    write_perm = Permission(
        permission_id="perm_write_data",
        name="Write Data",
        description="Allows writing data",
        resource_type=ResourceType.DATA,
        level=PermissionLevel.WRITE,
        actions=["write", "update", "create"]
    )
    agent.register_permission(write_perm)
    
    delete_perm = Permission(
        permission_id="perm_delete_data",
        name="Delete Data",
        description="Allows deleting data",
        resource_type=ResourceType.DATA,
        level=PermissionLevel.ADMIN,
        actions=["delete"]
    )
    agent.register_permission(delete_perm)
    
    admin_perm = Permission(
        permission_id="perm_admin",
        name="Admin Access",
        description="Full system access",
        resource_type=ResourceType.SYSTEM,
        level=PermissionLevel.ADMIN,
        actions=["read", "write", "delete", "execute", "configure"]
    )
    agent.register_permission(admin_perm)
    
    # Setup roles
    print("\n2. Creating Roles")
    print("-" * 60)
    
    viewer_role = Role(
        role_id="role_viewer",
        name="Viewer",
        description="Can only view data",
        permissions={"perm_read_data"}
    )
    agent.create_role(viewer_role)
    
    editor_role = Role(
        role_id="role_editor",
        name="Editor",
        description="Can view and edit data",
        permissions={"perm_read_data", "perm_write_data"},
        inherits_from=["role_viewer"]
    )
    agent.create_role(editor_role)
    
    admin_role = Role(
        role_id="role_admin",
        name="Administrator",
        description="Full access",
        permissions={"perm_read_data", "perm_write_data", "perm_delete_data", "perm_admin"},
        inherits_from=["role_editor"]
    )
    agent.create_role(admin_role)
    
    # Register users
    print("\n3. Registering Users")
    print("-" * 60)
    
    user1 = User(
        user_id="user_001",
        username="alice_viewer",
        roles=set()
    )
    agent.register_user(user1)
    agent.assign_role("user_001", "role_viewer")
    
    user2 = User(
        user_id="user_002",
        username="bob_editor",
        roles=set()
    )
    agent.register_user(user2)
    agent.assign_role("user_002", "role_editor")
    
    user3 = User(
        user_id="user_003",
        username="charlie_admin",
        roles=set()
    )
    agent.register_user(user3)
    agent.assign_role("user_003", "role_admin")
    
    # Test access control
    print("\n4. Testing Access Control")
    print("-" * 60)
    
    # Alice tries to read (should be allowed)
    print("\nAlice (Viewer) tries to read data:")
    decision1 = agent.check_access("user_001", "read", "/data/file1.txt", ResourceType.DATA)
    print(f"Decision: {decision1.decision.value}")
    print(f"Reason: {decision1.reason}")
    
    # Alice tries to write (should be denied)
    print("\nAlice (Viewer) tries to write data:")
    decision2 = agent.check_access("user_001", "write", "/data/file1.txt", ResourceType.DATA)
    print(f"Decision: {decision2.decision.value}")
    print(f"Reason: {decision2.reason}")
    
    # Bob tries to write (should be allowed)
    print("\nBob (Editor) tries to write data:")
    decision3 = agent.check_access("user_002", "write", "/data/file2.txt", ResourceType.DATA)
    print(f"Decision: {decision3.decision.value}")
    print(f"Reason: {decision3.reason}")
    
    # Bob tries to delete (should be denied)
    print("\nBob (Editor) tries to delete data:")
    decision4 = agent.check_access("user_002", "delete", "/data/file2.txt", ResourceType.DATA)
    print(f"Decision: {decision4.decision.value}")
    print(f"Reason: {decision4.reason}")
    
    # Charlie tries to delete (should be allowed)
    print("\nCharlie (Admin) tries to delete data:")
    decision5 = agent.check_access("user_003", "delete", "/data/file3.txt", ResourceType.DATA)
    print(f"Decision: {decision5.decision.value}")
    print(f"Reason: {decision5.reason}")
    
    # Temporary access
    print("\n5. Granting Temporary Access")
    print("-" * 60)
    
    temp_grant = agent.grant_temporary_access(
        user_id="user_001",
        permission_id="perm_write_data",
        granted_by="user_003",
        duration_hours=2,
        reason="Temporary editing privileges for project update"
    )
    
    if temp_grant:
        print(f"✓ Granted temporary access to Alice")
        print(f"  Grant ID: {temp_grant.grant_id}")
        print(f"  Expires: {temp_grant.expires_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Reason: {temp_grant.reason}")
    
    # Alice tries to write again (should now be allowed)
    print("\nAlice tries to write data (with temporary access):")
    decision6 = agent.check_access("user_001", "write", "/data/file4.txt", ResourceType.DATA)
    print(f"Decision: {decision6.decision.value}")
    print(f"Reason: {decision6.reason}")
    
    # View user permissions
    print("\n6. User Permissions Summary")
    print("-" * 60)
    
    for user_id in ["user_001", "user_002", "user_003"]:
        permissions = agent.get_user_permissions(user_id)
        user = agent.user_manager.users[user_id]
        print(f"\n{user.username}:")
        print(f"  Roles: {', '.join(user.roles)}")
        print(f"  Total Permissions: {len(permissions)}")
        print(f"  Permissions: {', '.join(permissions)}")
    
    # Statistics
    print("\n7. Authorization Statistics")
    print("-" * 60)
    
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_permission_authorization()
