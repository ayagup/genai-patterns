"""
Pattern 139: Permission & Authorization

Description:
    The Permission & Authorization pattern implements fine-grained access control
    for agent systems, determining who can perform what actions on which resources.
    This pattern provides role-based access control (RBAC), attribute-based access
    control (ABAC), and resource-level permissions to ensure that agents and users
    only access what they're authorized to access.

    Authorization is fundamental to secure AI systems, especially in multi-user
    environments or when handling sensitive data. It enforces the principle of
    least privilege, prevents unauthorized access, and ensures compliance with
    security policies. The pattern supports hierarchical roles, permission
    inheritance, dynamic permissions, and context-aware authorization.

    This implementation provides a comprehensive authorization system with role
    management, permission assignment, access control checks, resource-level
    permissions, and permission delegation. It supports both static and dynamic
    authorization decisions based on user attributes, resource properties, and
    environmental context.

Components:
    - Role Management: Define and manage user roles
    - Permission Assignment: Assign permissions to roles and users
    - Access Control: Check if action is authorized
    - Resource Protection: Control access to specific resources
    - Permission Delegation: Temporary permission grants
    - Context-Aware Authorization: Consider context in decisions

Use Cases:
    - Multi-user AI systems
    - Enterprise agent platforms
    - Data access control
    - API authorization
    - Healthcare systems (HIPAA compliance)
    - Financial systems
    - Government systems
    - Multi-tenant platforms

LangChain Implementation:
    This implementation uses:
    - Custom RBAC and ABAC systems
    - Permission hierarchy and inheritance
    - Resource-level access control
    - LLM for complex authorization decisions
    - Context-aware permission evaluation
    - Audit logging integration

Benefits:
    - Enforces least privilege principle
    - Prevents unauthorized access
    - Supports compliance requirements
    - Enables fine-grained control
    - Facilitates security auditing
    - Supports multi-tenancy
    - Enables delegation

Trade-offs:
    - Adds complexity to system
    - Requires permission management
    - May impact performance
    - Needs careful role design
    - Can be difficult to debug
    - Requires ongoing maintenance

Production Considerations:
    - Design clear role hierarchy
    - Implement efficient permission checks
    - Cache authorization decisions
    - Log all authorization events
    - Support emergency access
    - Implement permission review process
    - Handle permission conflicts
    - Support temporal permissions
    - Provide audit capabilities
    - Test permission combinations
    - Document permission model
    - Monitor unauthorized attempts
"""

import os
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ActionType(Enum):
    """Types of actions."""
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    EXPORT = "export"
    SHARE = "share"


class ResourceType(Enum):
    """Types of resources."""
    DATA = "data"
    MODEL = "model"
    AGENT = "agent"
    CONFIG = "config"
    USER = "user"
    API = "api"


@dataclass
class Permission:
    """Permission definition."""
    permission_id: str
    name: str
    description: str
    resource_type: ResourceType
    action: ActionType
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.permission_id)


@dataclass
class Role:
    """Role definition."""
    role_id: str
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_permission(self, permission: Permission):
        """Add permission to role."""
        self.permissions.add(permission)
    
    def remove_permission(self, permission: Permission):
        """Remove permission from role."""
        self.permissions.discard(permission)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if role has permission."""
        return permission in self.permissions


@dataclass
class User:
    """User definition."""
    user_id: str
    username: str
    roles: Set[str] = field(default_factory=set)
    direct_permissions: Set[Permission] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_role(self, role_id: str):
        """Add role to user."""
        self.roles.add(role_id)
    
    def remove_role(self, role_id: str):
        """Remove role from user."""
        self.roles.discard(role_id)
    
    def add_permission(self, permission: Permission):
        """Add direct permission to user."""
        self.direct_permissions.add(permission)


@dataclass
class Resource:
    """Resource definition."""
    resource_id: str
    resource_type: ResourceType
    owner_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AuthorizationRequest:
    """Authorization request."""
    user_id: str
    action: ActionType
    resource: Resource
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthorizationResult:
    """Authorization result."""
    authorized: bool
    reason: str
    matched_permissions: List[Permission]
    timestamp: datetime = field(default_factory=datetime.now)


class PermissionAuthorizationAgent:
    """
    Agent that provides permission and authorization management.
    
    This agent manages roles, permissions, and authorization decisions
    using RBAC and ABAC approaches.
    """
    
    def __init__(self, temperature: float = 0.3):
        """Initialize the authorization agent."""
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.resources: Dict[str, Resource] = {}
        self.permissions: Dict[str, Permission] = {}
        
        # Create authorization chain for complex decisions
        auth_prompt = ChatPromptTemplate.from_template(
            """You are an authorization expert. Determine if the user should be authorized for this action.

User: {user_id}
User Attributes: {user_attributes}
Action: {action}
Resource Type: {resource_type}
Resource Attributes: {resource_attributes}
Context: {context}

Consider:
1. User's roles and permissions
2. Resource sensitivity
3. Context (time, location, etc.)
4. Business rules and policies

Respond in this format:
AUTHORIZED: yes/no
REASON: explanation
RISK_LEVEL: low/medium/high
CONDITIONS: any conditions for authorization"""
        )
        self.auth_chain = auth_prompt | self.llm | StrOutputParser()
    
    def create_permission(
        self,
        permission_id: str,
        name: str,
        description: str,
        resource_type: ResourceType,
        action: ActionType,
        conditions: Optional[Dict[str, Any]] = None
    ) -> Permission:
        """Create a new permission."""
        permission = Permission(
            permission_id=permission_id,
            name=name,
            description=description,
            resource_type=resource_type,
            action=action,
            conditions=conditions or {}
        )
        self.permissions[permission_id] = permission
        return permission
    
    def create_role(
        self,
        role_id: str,
        name: str,
        description: str,
        parent_roles: Optional[Set[str]] = None
    ) -> Role:
        """Create a new role."""
        role = Role(
            role_id=role_id,
            name=name,
            description=description,
            parent_roles=parent_roles or set()
        )
        self.roles[role_id] = role
        return role
    
    def create_user(
        self,
        user_id: str,
        username: str,
        roles: Optional[Set[str]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> User:
        """Create a new user."""
        user = User(
            user_id=user_id,
            username=username,
            roles=roles or set(),
            attributes=attributes or {}
        )
        self.users[user_id] = user
        return user
    
    def create_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
        owner_id: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Resource:
        """Create a new resource."""
        resource = Resource(
            resource_id=resource_id,
            resource_type=resource_type,
            owner_id=owner_id,
            attributes=attributes or {}
        )
        self.resources[resource_id] = resource
        return resource
    
    def assign_permission_to_role(
        self,
        role_id: str,
        permission_id: str
    ) -> bool:
        """Assign permission to role."""
        if role_id not in self.roles or permission_id not in self.permissions:
            return False
        
        role = self.roles[role_id]
        permission = self.permissions[permission_id]
        role.add_permission(permission)
        return True
    
    def assign_role_to_user(
        self,
        user_id: str,
        role_id: str
    ) -> bool:
        """Assign role to user."""
        if user_id not in self.users or role_id not in self.roles:
            return False
        
        user = self.users[user_id]
        user.add_role(role_id)
        return True
    
    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user (from roles and direct)."""
        if user_id not in self.users:
            return set()
        
        user = self.users[user_id]
        permissions = user.direct_permissions.copy()
        
        # Add permissions from roles
        for role_id in user.roles:
            if role_id in self.roles:
                role = self.roles[role_id]
                permissions.update(role.permissions)
                
                # Add permissions from parent roles (inheritance)
                for parent_id in role.parent_roles:
                    if parent_id in self.roles:
                        permissions.update(self.roles[parent_id].permissions)
        
        return permissions
    
    def check_authorization(
        self,
        request: AuthorizationRequest
    ) -> AuthorizationResult:
        """
        Check if user is authorized for action on resource.
        
        This performs both static (RBAC) and dynamic (ABAC) checks.
        """
        if request.user_id not in self.users:
            return AuthorizationResult(
                authorized=False,
                reason="User not found",
                matched_permissions=[]
            )
        
        user = self.users[request.user_id]
        user_permissions = self.get_user_permissions(request.user_id)
        
        # Check if user is owner
        if request.resource.owner_id == request.user_id:
            return AuthorizationResult(
                authorized=True,
                reason="User is resource owner",
                matched_permissions=[]
            )
        
        # Check static permissions (RBAC)
        matched = []
        for permission in user_permissions:
            if (permission.resource_type == request.resource.resource_type and
                permission.action == request.action):
                
                # Check conditions
                if self._check_conditions(permission.conditions, request.context):
                    matched.append(permission)
        
        if matched:
            return AuthorizationResult(
                authorized=True,
                reason=f"Authorized via permissions: {', '.join(p.name for p in matched)}",
                matched_permissions=matched
            )
        
        # If no static match, use LLM for dynamic decision (ABAC)
        return self._dynamic_authorization(user, request)
    
    def _check_conditions(
        self,
        conditions: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """Check if conditions are met."""
        if not conditions:
            return True
        
        for key, value in conditions.items():
            if key not in context or context[key] != value:
                return False
        
        return True
    
    def _dynamic_authorization(
        self,
        user: User,
        request: AuthorizationRequest
    ) -> AuthorizationResult:
        """Use LLM for dynamic authorization decision."""
        try:
            result = self.auth_chain.invoke({
                "user_id": user.user_id,
                "user_attributes": str(user.attributes),
                "action": request.action.value,
                "resource_type": request.resource.resource_type.value,
                "resource_attributes": str(request.resource.attributes),
                "context": str(request.context)
            })
            
            # Parse result
            authorized = "yes" in result.lower().split("AUTHORIZED:")[1].split("\n")[0]
            reason_line = [l for l in result.split("\n") if l.startswith("REASON:")]
            reason = reason_line[0].replace("REASON:", "").strip() if reason_line else "Dynamic authorization"
            
            return AuthorizationResult(
                authorized=authorized,
                reason=reason,
                matched_permissions=[]
            )
            
        except Exception as e:
            return AuthorizationResult(
                authorized=False,
                reason=f"Authorization check failed: {e}",
                matched_permissions=[]
            )
    
    def get_authorization_summary(self) -> Dict[str, Any]:
        """Get summary of authorization system."""
        return {
            "total_users": len(self.users),
            "total_roles": len(self.roles),
            "total_permissions": len(self.permissions),
            "total_resources": len(self.resources),
            "users_by_role": {
                role_id: len([u for u in self.users.values() if role_id in u.roles])
                for role_id in self.roles.keys()
            }
        }


def demonstrate_permission_authorization():
    """Demonstrate the permission and authorization pattern."""
    print("=" * 80)
    print("Permission & Authorization Pattern Demonstration")
    print("=" * 80)
    
    agent = PermissionAuthorizationAgent()
    
    # Demonstration 1: Create Permissions
    print("\n" + "=" * 80)
    print("Demonstration 1: Create Permissions")
    print("=" * 80)
    
    read_data = agent.create_permission(
        permission_id="perm_read_data",
        name="Read Data",
        description="Permission to read data resources",
        resource_type=ResourceType.DATA,
        action=ActionType.READ
    )
    
    write_data = agent.create_permission(
        permission_id="perm_write_data",
        name="Write Data",
        description="Permission to write data resources",
        resource_type=ResourceType.DATA,
        action=ActionType.WRITE
    )
    
    admin_all = agent.create_permission(
        permission_id="perm_admin",
        name="Admin All",
        description="Full administrative access",
        resource_type=ResourceType.DATA,
        action=ActionType.ADMIN
    )
    
    print(f"Created permissions:")
    print(f"  - {read_data.name}: {read_data.description}")
    print(f"  - {write_data.name}: {write_data.description}")
    print(f"  - {admin_all.name}: {admin_all.description}")
    
    # Demonstration 2: Create Roles
    print("\n" + "=" * 80)
    print("Demonstration 2: Create Roles with Hierarchy")
    print("=" * 80)
    
    viewer_role = agent.create_role(
        role_id="role_viewer",
        name="Viewer",
        description="Can view data"
    )
    agent.assign_permission_to_role("role_viewer", "perm_read_data")
    
    editor_role = agent.create_role(
        role_id="role_editor",
        name="Editor",
        description="Can view and edit data",
        parent_roles={"role_viewer"}
    )
    agent.assign_permission_to_role("role_editor", "perm_write_data")
    
    admin_role = agent.create_role(
        role_id="role_admin",
        name="Admin",
        description="Full system access",
        parent_roles={"role_editor"}
    )
    agent.assign_permission_to_role("role_admin", "perm_admin")
    
    print(f"Created role hierarchy:")
    print(f"  Admin -> Editor -> Viewer")
    print(f"  - Viewer: {len(viewer_role.permissions)} permissions")
    print(f"  - Editor: {len(editor_role.permissions)} permissions (+ inherited)")
    print(f"  - Admin: {len(admin_role.permissions)} permissions (+ inherited)")
    
    # Demonstration 3: Create Users
    print("\n" + "=" * 80)
    print("Demonstration 3: Create Users with Roles")
    print("=" * 80)
    
    user1 = agent.create_user(
        user_id="user_001",
        username="alice",
        roles={"role_viewer"},
        attributes={"department": "sales", "level": "junior"}
    )
    
    user2 = agent.create_user(
        user_id="user_002",
        username="bob",
        roles={"role_editor"},
        attributes={"department": "engineering", "level": "senior"}
    )
    
    user3 = agent.create_user(
        user_id="user_003",
        username="charlie",
        roles={"role_admin"},
        attributes={"department": "it", "level": "manager"}
    )
    
    print(f"Created users:")
    print(f"  - {user1.username}: {user1.roles}")
    print(f"  - {user2.username}: {user2.roles}")
    print(f"  - {user3.username}: {user3.roles}")
    
    # Demonstration 4: Create Resources
    print("\n" + "=" * 80)
    print("Demonstration 4: Create Protected Resources")
    print("=" * 80)
    
    resource1 = agent.create_resource(
        resource_id="res_001",
        resource_type=ResourceType.DATA,
        owner_id="user_003",
        attributes={"classification": "public", "department": "sales"}
    )
    
    resource2 = agent.create_resource(
        resource_id="res_002",
        resource_type=ResourceType.DATA,
        owner_id="user_003",
        attributes={"classification": "confidential", "department": "finance"}
    )
    
    print(f"Created resources:")
    print(f"  - {resource1.resource_id}: {resource1.attributes['classification']}")
    print(f"  - {resource2.resource_id}: {resource2.attributes['classification']}")
    
    # Demonstration 5: Authorization Checks
    print("\n" + "=" * 80)
    print("Demonstration 5: Authorization Checks")
    print("=" * 80)
    
    # Alice (viewer) tries to read
    request1 = AuthorizationRequest(
        user_id="user_001",
        action=ActionType.READ,
        resource=resource1,
        context={}
    )
    result1 = agent.check_authorization(request1)
    print(f"\nAlice (viewer) reads public data:")
    print(f"  Authorized: {result1.authorized}")
    print(f"  Reason: {result1.reason}")
    
    # Alice tries to write
    request2 = AuthorizationRequest(
        user_id="user_001",
        action=ActionType.WRITE,
        resource=resource1,
        context={}
    )
    result2 = agent.check_authorization(request2)
    print(f"\nAlice (viewer) writes data:")
    print(f"  Authorized: {result2.authorized}")
    print(f"  Reason: {result2.reason}")
    
    # Bob (editor) tries to write
    request3 = AuthorizationRequest(
        user_id="user_002",
        action=ActionType.WRITE,
        resource=resource1,
        context={}
    )
    result3 = agent.check_authorization(request3)
    print(f"\nBob (editor) writes data:")
    print(f"  Authorized: {result3.authorized}")
    print(f"  Reason: {result3.reason}")
    
    # Demonstration 6: Owner Access
    print("\n" + "=" * 80)
    print("Demonstration 6: Resource Owner Access")
    print("=" * 80)
    
    # Charlie is owner
    request4 = AuthorizationRequest(
        user_id="user_003",
        action=ActionType.DELETE,
        resource=resource1,
        context={}
    )
    result4 = agent.check_authorization(request4)
    print(f"\nCharlie (owner) deletes resource:")
    print(f"  Authorized: {result4.authorized}")
    print(f"  Reason: {result4.reason}")
    
    # Demonstration 7: User Permissions Summary
    print("\n" + "=" * 80)
    print("Demonstration 7: User Permissions Summary")
    print("=" * 80)
    
    for user_id in ["user_001", "user_002", "user_003"]:
        user = agent.users[user_id]
        permissions = agent.get_user_permissions(user_id)
        print(f"\n{user.username} permissions:")
        print(f"  Roles: {', '.join(user.roles)}")
        print(f"  Total Permissions: {len(permissions)}")
        for perm in permissions:
            print(f"    - {perm.name}: {perm.action.value} {perm.resource_type.value}")
    
    # Demonstration 8: System Summary
    print("\n" + "=" * 80)
    print("Demonstration 8: Authorization System Summary")
    print("=" * 80)
    
    summary = agent.get_authorization_summary()
    print(f"\nAuthorization System:")
    print(f"  Users: {summary['total_users']}")
    print(f"  Roles: {summary['total_roles']}")
    print(f"  Permissions: {summary['total_permissions']}")
    print(f"  Resources: {summary['total_resources']}")
    print(f"\n  Users by Role:")
    for role, count in summary['users_by_role'].items():
        if count > 0:
            print(f"    {role}: {count}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary: Permission & Authorization Pattern")
    print("=" * 80)
    print("""
The Permission & Authorization pattern provides fine-grained access control:

Key Features Demonstrated:
1. Permission Management - Create and manage permissions
2. Role-Based Access Control - RBAC with role hierarchy
3. User Management - Assign roles and permissions to users
4. Resource Protection - Control access to specific resources
5. Authorization Checks - Verify user access rights
6. Owner Access - Automatic authorization for resource owners
7. Permission Inheritance - Inherit permissions from parent roles
8. System Management - Track and manage authorization state

Benefits:
- Enforces least privilege
- Prevents unauthorized access
- Supports compliance
- Enables fine-grained control
- Facilitates auditing
- Supports multi-tenancy
- Enables role delegation

Best Practices:
- Design clear role hierarchy
- Use principle of least privilege
- Cache authorization decisions
- Log authorization events
- Support emergency access
- Review permissions regularly
- Handle permission conflicts
- Test permission combinations
- Document permission model
- Monitor unauthorized attempts
- Support temporal permissions
- Provide audit capabilities

This pattern is essential for secure multi-user AI systems requiring
access control, compliance, and security enforcement.
""")


if __name__ == "__main__":
    demonstrate_permission_authorization()
