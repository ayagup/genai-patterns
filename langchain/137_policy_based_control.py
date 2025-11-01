"""
Pattern 137: Policy-Based Control

Description:
    The Policy-Based Control pattern implements governance mechanisms that enforce
    organizational policies, regulatory requirements, and operational constraints on
    agent behavior. This pattern provides a framework for defining, validating, and
    enforcing policies that control what actions agents can take, what data they can
    access, and how they should behave in various contexts.

    Policy-based control is essential for production deployments where agents must
    operate within well-defined boundaries. It enables organizations to maintain
    compliance with regulations, enforce security policies, implement business rules,
    and ensure ethical AI behavior. The pattern supports both hard constraints (must
    be followed) and soft constraints (should be followed with exceptions).

    This implementation provides a comprehensive policy management system with rule
    definition, policy validation, violation detection, enforcement mechanisms, and
    policy composition. It supports multiple policy types including security policies,
    data policies, behavioral policies, and resource policies, with the ability to
    combine policies and handle conflicts.

Components:
    - Policy Definition: Structured rules with conditions and actions
    - Policy Validation: Checks if actions comply with policies
    - Violation Detection: Identifies policy breaches
    - Enforcement Mechanisms: Blocks, modifies, or logs violations
    - Policy Composition: Combines multiple policies
    - Conflict Resolution: Handles policy conflicts

Use Cases:
    - Regulatory compliance (GDPR, HIPAA, SOC2)
    - Security policy enforcement
    - Business rule implementation
    - Ethical AI constraints
    - Resource usage governance
    - Data access control
    - Content moderation
    - Operational safety

LangChain Implementation:
    This implementation uses:
    - ChatOpenAI for policy validation and interpretation
    - Custom policy engine for rule enforcement
    - Structured policy definitions with conditions and actions
    - LLM-based policy reasoning for complex scenarios
    - Policy composition and inheritance mechanisms
    - Violation tracking and audit logging

Benefits:
    - Ensures compliance with regulations and standards
    - Enforces organizational policies consistently
    - Provides audit trail for policy enforcement
    - Reduces risk of policy violations
    - Enables fine-grained control over agent behavior
    - Supports policy versioning and evolution
    - Facilitates policy management at scale

Trade-offs:
    - May restrict agent flexibility and autonomy
    - Can increase latency due to policy checks
    - Requires careful policy design to avoid conflicts
    - May need ongoing policy maintenance
    - Can create complexity in policy management
    - Requires balance between control and capability

Production Considerations:
    - Design clear, unambiguous policies
    - Implement efficient policy evaluation
    - Cache policy decisions when possible
    - Log all policy violations with context
    - Support policy testing and simulation
    - Implement policy versioning
    - Provide policy override mechanisms
    - Monitor policy effectiveness
    - Support policy composition
    - Handle policy conflicts gracefully
    - Implement emergency policy updates
    - Provide policy documentation
"""

import os
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class PolicyType(Enum):
    """Types of policies."""
    SECURITY = "security"
    DATA = "data"
    BEHAVIORAL = "behavioral"
    RESOURCE = "resource"
    COMPLIANCE = "compliance"
    ETHICAL = "ethical"
    OPERATIONAL = "operational"
    CONTENT = "content"


class PolicySeverity(Enum):
    """Severity levels for policy violations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class EnforcementAction(Enum):
    """Actions to take when policy is violated."""
    BLOCK = "block"
    MODIFY = "modify"
    LOG = "log"
    WARN = "warn"
    ESCALATE = "escalate"
    ALLOW = "allow"


@dataclass
class PolicyCondition:
    """Condition for policy rule."""
    field: str
    operator: str  # eq, ne, gt, lt, contains, matches, etc.
    value: Any
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        if self.field not in context:
            return False
        
        field_value = context[self.field]
        
        if self.operator == "eq":
            return field_value == self.value
        elif self.operator == "ne":
            return field_value != self.value
        elif self.operator == "gt":
            return field_value > self.value
        elif self.operator == "lt":
            return field_value < self.value
        elif self.operator == "gte":
            return field_value >= self.value
        elif self.operator == "lte":
            return field_value <= self.value
        elif self.operator == "contains":
            return self.value in field_value
        elif self.operator == "not_contains":
            return self.value not in field_value
        elif self.operator == "in":
            return field_value in self.value
        elif self.operator == "not_in":
            return field_value not in self.value
        else:
            return False


@dataclass
class Policy:
    """Policy definition."""
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    severity: PolicySeverity
    conditions: List[PolicyCondition]
    enforcement_action: EnforcementAction
    enabled: bool = True
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyViolation:
    """Record of policy violation."""
    violation_id: str
    policy_id: str
    policy_name: str
    severity: PolicySeverity
    action: str
    context: Dict[str, Any]
    enforcement_action: EnforcementAction
    timestamp: datetime = field(default_factory=datetime.now)
    resolution: Optional[str] = None


class PolicyBasedControlAgent:
    """
    Agent that enforces policy-based control.
    
    This agent manages policies, validates actions against policies,
    detects violations, and enforces policy constraints.
    """
    
    def __init__(self, temperature: float = 0.3):
        """Initialize the policy control agent."""
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.policies: Dict[str, Policy] = {}
        self.violations: List[PolicyViolation] = []
        
        # Create policy validation chain
        validation_prompt = ChatPromptTemplate.from_template(
            """You are a policy compliance expert. Analyze if the proposed action complies with the given policy.

Policy: {policy_name}
Description: {policy_description}
Type: {policy_type}
Severity: {severity}

Proposed Action: {action}
Context: {context}

Evaluate if the action violates the policy. Consider:
1. Does the action match the conditions specified in the policy?
2. Are there any regulatory or compliance concerns?
3. What are the potential risks?

Respond in this format:
VIOLATES: yes/no
REASON: explanation
RISK_LEVEL: critical/high/medium/low
RECOMMENDATION: what should be done"""
        )
        self.validation_chain = validation_prompt | self.llm | StrOutputParser()
    
    def add_policy(self, policy: Policy) -> None:
        """Add a policy to the system."""
        self.policies[policy.policy_id] = policy
        print(f"Added policy: {policy.name} (ID: {policy.policy_id})")
    
    def remove_policy(self, policy_id: str) -> None:
        """Remove a policy from the system."""
        if policy_id in self.policies:
            policy = self.policies.pop(policy_id)
            print(f"Removed policy: {policy.name}")
        else:
            print(f"Policy not found: {policy_id}")
    
    def enable_policy(self, policy_id: str) -> None:
        """Enable a policy."""
        if policy_id in self.policies:
            self.policies[policy_id].enabled = True
            print(f"Enabled policy: {self.policies[policy_id].name}")
    
    def disable_policy(self, policy_id: str) -> None:
        """Disable a policy."""
        if policy_id in self.policies:
            self.policies[policy_id].enabled = False
            print(f"Disabled policy: {self.policies[policy_id].name}")
    
    def validate_action(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> Tuple[bool, List[PolicyViolation]]:
        """
        Validate an action against all applicable policies.
        
        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = []
        
        # Get applicable policies (enabled, sorted by priority)
        applicable_policies = sorted(
            [p for p in self.policies.values() if p.enabled],
            key=lambda p: p.priority,
            reverse=True
        )
        
        for policy in applicable_policies:
            # Check if policy conditions match
            if self._conditions_match(policy.conditions, context):
                # Use LLM for detailed validation
                violation = self._check_policy_violation(policy, action, context)
                if violation:
                    violations.append(violation)
                    self.violations.append(violation)
        
        is_valid = len(violations) == 0
        return is_valid, violations
    
    def _conditions_match(
        self,
        conditions: List[PolicyCondition],
        context: Dict[str, Any]
    ) -> bool:
        """Check if all conditions match the context."""
        if not conditions:
            return True  # No conditions means always applicable
        
        return all(condition.evaluate(context) for condition in conditions)
    
    def _check_policy_violation(
        self,
        policy: Policy,
        action: str,
        context: Dict[str, Any]
    ) -> Optional[PolicyViolation]:
        """Check if action violates policy using LLM."""
        try:
            result = self.validation_chain.invoke({
                "policy_name": policy.name,
                "policy_description": policy.description,
                "policy_type": policy.policy_type.value,
                "severity": policy.severity.value,
                "action": action,
                "context": str(context)
            })
            
            # Parse result
            lines = result.strip().split('\n')
            violates = False
            reason = ""
            risk_level = "medium"
            
            for line in lines:
                if line.startswith("VIOLATES:"):
                    violates = "yes" in line.lower()
                elif line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()
                elif line.startswith("RISK_LEVEL:"):
                    risk_level = line.replace("RISK_LEVEL:", "").strip()
            
            if violates:
                violation = PolicyViolation(
                    violation_id=f"violation_{datetime.now().timestamp()}",
                    policy_id=policy.policy_id,
                    policy_name=policy.name,
                    severity=policy.severity,
                    action=action,
                    context=context,
                    enforcement_action=policy.enforcement_action
                )
                return violation
            
            return None
            
        except Exception as e:
            print(f"Error checking policy violation: {e}")
            return None
    
    def enforce_policies(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enforce policies on an action.
        
        Returns:
            Dictionary with enforcement result
        """
        is_valid, violations = self.validate_action(action, context)
        
        if is_valid:
            return {
                "allowed": True,
                "action": action,
                "violations": [],
                "message": "Action complies with all policies"
            }
        
        # Determine enforcement action based on violations
        critical_violations = [
            v for v in violations 
            if v.severity in [PolicySeverity.CRITICAL, PolicySeverity.HIGH]
        ]
        
        if critical_violations:
            # Block critical violations
            return {
                "allowed": False,
                "action": None,
                "violations": violations,
                "message": f"Action blocked due to {len(critical_violations)} critical policy violations"
            }
        else:
            # Log and warn for lower severity
            return {
                "allowed": True,
                "action": action,
                "violations": violations,
                "message": f"Action allowed with {len(violations)} policy warnings"
            }
    
    def compose_policies(
        self,
        policy_ids: List[str],
        composition_strategy: str = "all"
    ) -> Policy:
        """
        Compose multiple policies into a single policy.
        
        Args:
            policy_ids: List of policy IDs to compose
            composition_strategy: "all" (AND), "any" (OR)
        """
        policies = [self.policies[pid] for pid in policy_ids if pid in self.policies]
        
        if not policies:
            raise ValueError("No valid policies to compose")
        
        # Create composite policy
        composite = Policy(
            policy_id=f"composite_{datetime.now().timestamp()}",
            name=f"Composite: {', '.join(p.name for p in policies)}",
            description=f"Composite policy combining: {', '.join(p.name for p in policies)}",
            policy_type=PolicyType.COMPLIANCE,
            severity=max(policies, key=lambda p: p.severity.value).severity,
            conditions=[],  # Conditions evaluated separately
            enforcement_action=EnforcementAction.BLOCK,
            priority=max(p.priority for p in policies)
        )
        
        return composite
    
    def get_violations_by_severity(
        self,
        severity: PolicySeverity
    ) -> List[PolicyViolation]:
        """Get violations by severity level."""
        return [v for v in self.violations if v.severity == severity]
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of policy system."""
        return {
            "total_policies": len(self.policies),
            "enabled_policies": len([p for p in self.policies.values() if p.enabled]),
            "disabled_policies": len([p for p in self.policies.values() if not p.enabled]),
            "total_violations": len(self.violations),
            "critical_violations": len(self.get_violations_by_severity(PolicySeverity.CRITICAL)),
            "high_violations": len(self.get_violations_by_severity(PolicySeverity.HIGH)),
            "policy_types": {
                ptype.value: len([p for p in self.policies.values() if p.policy_type == ptype])
                for ptype in PolicyType
            }
        }


def demonstrate_policy_based_control():
    """Demonstrate the policy-based control pattern."""
    print("=" * 80)
    print("Policy-Based Control Pattern Demonstration")
    print("=" * 80)
    
    agent = PolicyBasedControlAgent()
    
    # Demonstration 1: Security Policy
    print("\n" + "=" * 80)
    print("Demonstration 1: Security Policy Enforcement")
    print("=" * 80)
    
    security_policy = Policy(
        policy_id="sec_001",
        name="Data Access Policy",
        description="Restricts access to sensitive data based on user role",
        policy_type=PolicyType.SECURITY,
        severity=PolicySeverity.CRITICAL,
        conditions=[
            PolicyCondition(field="data_type", operator="eq", value="sensitive")
        ],
        enforcement_action=EnforcementAction.BLOCK,
        priority=100
    )
    
    agent.add_policy(security_policy)
    
    # Test action that violates policy
    action = "Access customer payment information"
    context = {
        "data_type": "sensitive",
        "user_role": "analyst",
        "required_role": "admin"
    }
    
    result = agent.enforce_policies(action, context)
    print(f"\nAction: {action}")
    print(f"Allowed: {result['allowed']}")
    print(f"Violations: {len(result['violations'])}")
    print(f"Message: {result['message']}")
    
    # Demonstration 2: Resource Policy
    print("\n" + "=" * 80)
    print("Demonstration 2: Resource Usage Policy")
    print("=" * 80)
    
    resource_policy = Policy(
        policy_id="res_001",
        name="API Rate Limit Policy",
        description="Limits API calls to prevent abuse",
        policy_type=PolicyType.RESOURCE,
        severity=PolicySeverity.HIGH,
        conditions=[
            PolicyCondition(field="api_calls", operator="gt", value=100)
        ],
        enforcement_action=EnforcementAction.BLOCK,
        priority=80
    )
    
    agent.add_policy(resource_policy)
    
    action = "Make API call to external service"
    context = {
        "api_calls": 150,
        "time_window": "1 hour"
    }
    
    result = agent.enforce_policies(action, context)
    print(f"\nAction: {action}")
    print(f"Allowed: {result['allowed']}")
    print(f"Message: {result['message']}")
    
    # Demonstration 3: Content Policy
    print("\n" + "=" * 80)
    print("Demonstration 3: Content Moderation Policy")
    print("=" * 80)
    
    content_policy = Policy(
        policy_id="cont_001",
        name="Content Safety Policy",
        description="Prevents generation of harmful content",
        policy_type=PolicyType.CONTENT,
        severity=PolicySeverity.CRITICAL,
        conditions=[
            PolicyCondition(field="content_category", operator="in", 
                          value=["harmful", "illegal", "offensive"])
        ],
        enforcement_action=EnforcementAction.BLOCK,
        priority=100
    )
    
    agent.add_policy(content_policy)
    
    action = "Generate content about violence"
    context = {
        "content_category": "harmful",
        "user_age": 15
    }
    
    result = agent.enforce_policies(action, context)
    print(f"\nAction: {action}")
    print(f"Allowed: {result['allowed']}")
    print(f"Message: {result['message']}")
    
    # Demonstration 4: Compliance Policy
    print("\n" + "=" * 80)
    print("Demonstration 4: Regulatory Compliance Policy")
    print("=" * 80)
    
    compliance_policy = Policy(
        policy_id="comp_001",
        name="GDPR Compliance Policy",
        description="Ensures GDPR compliance for EU users",
        policy_type=PolicyType.COMPLIANCE,
        severity=PolicySeverity.CRITICAL,
        conditions=[
            PolicyCondition(field="user_location", operator="eq", value="EU"),
            PolicyCondition(field="data_processing", operator="eq", value="personal")
        ],
        enforcement_action=EnforcementAction.BLOCK,
        priority=100
    )
    
    agent.add_policy(compliance_policy)
    
    action = "Process user data for marketing"
    context = {
        "user_location": "EU",
        "data_processing": "personal",
        "consent_given": False
    }
    
    result = agent.enforce_policies(action, context)
    print(f"\nAction: {action}")
    print(f"Allowed: {result['allowed']}")
    print(f"Message: {result['message']}")
    
    # Demonstration 5: Policy Enable/Disable
    print("\n" + "=" * 80)
    print("Demonstration 5: Dynamic Policy Management")
    print("=" * 80)
    
    print("\nDisabling resource policy...")
    agent.disable_policy("res_001")
    
    action = "Make API call to external service"
    context = {
        "api_calls": 150,
        "time_window": "1 hour"
    }
    
    result = agent.enforce_policies(action, context)
    print(f"\nAction: {action}")
    print(f"Allowed: {result['allowed']}")
    print(f"Message: {result['message']}")
    
    print("\nRe-enabling resource policy...")
    agent.enable_policy("res_001")
    
    # Demonstration 6: Multiple Policy Violations
    print("\n" + "=" * 80)
    print("Demonstration 6: Multiple Policy Violations")
    print("=" * 80)
    
    action = "Access and export customer data"
    context = {
        "data_type": "sensitive",
        "user_role": "analyst",
        "user_location": "EU",
        "data_processing": "personal",
        "consent_given": False
    }
    
    is_valid, violations = agent.validate_action(action, context)
    print(f"\nAction: {action}")
    print(f"Valid: {is_valid}")
    print(f"Violations: {len(violations)}")
    for i, violation in enumerate(violations, 1):
        print(f"\n  Violation {i}:")
        print(f"    Policy: {violation.policy_name}")
        print(f"    Severity: {violation.severity.value}")
        print(f"    Action: {violation.enforcement_action.value}")
    
    # Demonstration 7: Policy Priority
    print("\n" + "=" * 80)
    print("Demonstration 7: Policy Priority Handling")
    print("=" * 80)
    
    low_priority_policy = Policy(
        policy_id="low_001",
        name="Low Priority Warning",
        description="Low priority warning policy",
        policy_type=PolicyType.OPERATIONAL,
        severity=PolicySeverity.LOW,
        conditions=[
            PolicyCondition(field="action_type", operator="eq", value="read")
        ],
        enforcement_action=EnforcementAction.LOG,
        priority=10
    )
    
    agent.add_policy(low_priority_policy)
    
    action = "Read configuration file"
    context = {
        "action_type": "read",
        "file_type": "config"
    }
    
    result = agent.enforce_policies(action, context)
    print(f"\nAction: {action}")
    print(f"Allowed: {result['allowed']}")
    print(f"Violations: {len(result['violations'])}")
    print(f"Message: {result['message']}")
    
    # Demonstration 8: Policy Summary
    print("\n" + "=" * 80)
    print("Demonstration 8: Policy System Summary")
    print("=" * 80)
    
    summary = agent.get_policy_summary()
    print("\nPolicy System Summary:")
    print(f"  Total Policies: {summary['total_policies']}")
    print(f"  Enabled: {summary['enabled_policies']}")
    print(f"  Disabled: {summary['disabled_policies']}")
    print(f"  Total Violations: {summary['total_violations']}")
    print(f"  Critical Violations: {summary['critical_violations']}")
    print(f"  High Violations: {summary['high_violations']}")
    print(f"\n  Policies by Type:")
    for ptype, count in summary['policy_types'].items():
        if count > 0:
            print(f"    {ptype}: {count}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary: Policy-Based Control Pattern")
    print("=" * 80)
    print("""
The Policy-Based Control pattern provides comprehensive governance for agent behavior:

Key Features Demonstrated:
1. Policy Definition - Structured rules with conditions and enforcement actions
2. Security Policies - Data access control and protection
3. Resource Policies - API rate limiting and resource governance
4. Content Policies - Content moderation and safety
5. Compliance Policies - Regulatory compliance (GDPR, etc.)
6. Dynamic Management - Enable/disable policies at runtime
7. Multiple Violations - Handle multiple concurrent policy violations
8. Priority System - Policy evaluation based on priority

Benefits:
- Ensures regulatory compliance
- Enforces organizational policies
- Provides audit trail
- Enables fine-grained control
- Supports policy composition
- Reduces compliance risk

Best Practices:
- Design clear, unambiguous policies
- Use appropriate severity levels
- Implement efficient policy evaluation
- Cache policy decisions when possible
- Log all violations with context
- Support policy testing
- Provide policy override mechanisms
- Monitor policy effectiveness
- Handle policy conflicts gracefully
- Document all policies thoroughly

This pattern is essential for production AI systems requiring governance,
compliance, and controlled agent behavior.
""")


if __name__ == "__main__":
    demonstrate_policy_based_control()
