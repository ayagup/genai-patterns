"""
Agentic Design Pattern: Policy-Based Control

This pattern implements policy enforcement and compliance management for agent behavior.
The agent enforces rules, checks compliance, detects violations, and manages policy hierarchies.

Category: Control & Governance
Use Cases:
- Regulatory compliance enforcement
- Corporate policy management
- Security policy enforcement
- Quality assurance rules
- Operational guidelines
- Risk management controls
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Callable
from enum import Enum
from datetime import datetime
import hashlib
import random


class PolicyType(Enum):
    """Types of policies that can be enforced"""
    SECURITY = "security"
    COMPLIANCE = "compliance"
    QUALITY = "quality"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    ETHICAL = "ethical"


class PolicySeverity(Enum):
    """Severity levels for policy violations"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PolicyAction(Enum):
    """Actions to take on policy violations"""
    LOG = "log"
    WARN = "warn"
    BLOCK = "block"
    ESCALATE = "escalate"
    QUARANTINE = "quarantine"


class EnforcementMode(Enum):
    """Policy enforcement modes"""
    PERMISSIVE = "permissive"  # Log violations only
    ADVISORY = "advisory"       # Warn but allow
    ENFORCING = "enforcing"     # Block violations
    STRICT = "strict"           # Block and escalate


class ComplianceStatus(Enum):
    """Compliance check status"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNKNOWN = "unknown"


@dataclass
class Policy:
    """Represents a policy rule"""
    policy_id: str
    name: str
    description: str
    policy_type: PolicyType
    severity: PolicySeverity
    condition: Callable[[Dict[str, Any]], bool]  # Function to check condition
    action: PolicyAction
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyViolation:
    """Represents a policy violation"""
    violation_id: str
    policy_id: str
    policy_name: str
    severity: PolicySeverity
    timestamp: datetime
    context: Dict[str, Any]
    details: str
    action_taken: PolicyAction
    resolved: bool = False


@dataclass
class ComplianceCheck:
    """Represents a compliance check result"""
    check_id: str
    timestamp: datetime
    status: ComplianceStatus
    policies_checked: int
    violations: List[PolicyViolation]
    score: float  # 0-1, compliance percentage
    recommendations: List[str]


@dataclass
class PolicyContext:
    """Context for policy evaluation"""
    action: str
    actor: str
    resource: Optional[str]
    attributes: Dict[str, Any]
    timestamp: datetime


class PolicyEngine:
    """Core policy evaluation and enforcement engine"""
    
    def __init__(self):
        self.policies: Dict[str, Policy] = {}
        self.policy_hierarchy: Dict[str, List[str]] = {}  # parent -> children
        
    def add_policy(self, policy: Policy, parent_id: Optional[str] = None) -> None:
        """Add a policy to the engine"""
        self.policies[policy.policy_id] = policy
        
        if parent_id:
            if parent_id not in self.policy_hierarchy:
                self.policy_hierarchy[parent_id] = []
            self.policy_hierarchy[parent_id].append(policy.policy_id)
    
    def evaluate_policy(self, policy: Policy, context: Dict[str, Any]) -> bool:
        """Evaluate if a policy is satisfied"""
        if not policy.enabled:
            return True
        
        try:
            return policy.condition(context)
        except Exception as e:
            print(f"Error evaluating policy {policy.policy_id}: {e}")
            return False
    
    def evaluate_all(self, context: Dict[str, Any]) -> List[PolicyViolation]:
        """Evaluate all enabled policies"""
        violations = []
        
        for policy in self.policies.values():
            if not policy.enabled:
                continue
            
            if not self.evaluate_policy(policy, context):
                violation = PolicyViolation(
                    violation_id=self._generate_id(),
                    policy_id=policy.policy_id,
                    policy_name=policy.name,
                    severity=policy.severity,
                    timestamp=datetime.now(),
                    context=context,
                    details=f"Policy '{policy.name}' violated",
                    action_taken=policy.action
                )
                violations.append(violation)
        
        return violations
    
    def get_policy_chain(self, policy_id: str) -> List[Policy]:
        """Get policy and all its parent policies"""
        chain = []
        
        if policy_id in self.policies:
            chain.append(self.policies[policy_id])
        
        # Find parent policies (simplified - assumes single parent)
        for parent_id, children in self.policy_hierarchy.items():
            if policy_id in children and parent_id in self.policies:
                chain.extend(self.get_policy_chain(parent_id))
        
        return chain
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        return hashlib.md5(f"{datetime.now()}{random.random()}".encode()).hexdigest()[:12]


class ViolationDetector:
    """Detects and categorizes policy violations"""
    
    def __init__(self):
        self.violation_patterns: Dict[str, List[str]] = {
            "data_leak": ["sensitive", "confidential", "private"],
            "unauthorized_access": ["permission denied", "forbidden", "unauthorized"],
            "resource_abuse": ["quota exceeded", "rate limit", "overuse"],
            "security_risk": ["vulnerability", "exploit", "injection"]
        }
    
    def detect_violations(self, action: str, context: Dict[str, Any]) -> List[str]:
        """Detect potential violation categories"""
        detected = []
        action_lower = action.lower()
        
        for category, patterns in self.violation_patterns.items():
            if any(pattern in action_lower for pattern in patterns):
                detected.append(category)
        
        # Check context for additional violations
        if context.get("risk_score", 0) > 0.7:
            detected.append("high_risk_operation")
        
        if context.get("requires_approval") and not context.get("approved"):
            detected.append("unapproved_action")
        
        return detected
    
    def calculate_risk_score(self, violations: List[PolicyViolation]) -> float:
        """Calculate overall risk score from violations"""
        if not violations:
            return 0.0
        
        severity_weights = {
            PolicySeverity.INFO: 0.1,
            PolicySeverity.WARNING: 0.3,
            PolicySeverity.ERROR: 0.6,
            PolicySeverity.CRITICAL: 1.0
        }
        
        total_score = sum(severity_weights.get(v.severity, 0.5) for v in violations)
        return min(total_score / len(violations), 1.0)


class ComplianceManager:
    """Manages compliance checking and reporting"""
    
    def __init__(self, policy_engine: PolicyEngine):
        self.policy_engine = policy_engine
        self.compliance_history: List[ComplianceCheck] = []
    
    def check_compliance(self, context: Dict[str, Any]) -> ComplianceCheck:
        """Perform comprehensive compliance check"""
        violations = self.policy_engine.evaluate_all(context)
        
        total_policies = len([p for p in self.policy_engine.policies.values() if p.enabled])
        compliant_policies = total_policies - len(violations)
        
        score = compliant_policies / total_policies if total_policies > 0 else 1.0
        
        status = self._determine_status(score, violations)
        recommendations = self._generate_recommendations(violations)
        
        check = ComplianceCheck(
            check_id=self._generate_id(),
            timestamp=datetime.now(),
            status=status,
            policies_checked=total_policies,
            violations=violations,
            score=score,
            recommendations=recommendations
        )
        
        self.compliance_history.append(check)
        return check
    
    def _determine_status(self, score: float, violations: List[PolicyViolation]) -> ComplianceStatus:
        """Determine overall compliance status"""
        if score == 1.0:
            return ComplianceStatus.COMPLIANT
        elif score >= 0.8:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        elif any(v.severity == PolicySeverity.CRITICAL for v in violations):
            return ComplianceStatus.NON_COMPLIANT
        else:
            return ComplianceStatus.PARTIALLY_COMPLIANT
    
    def _generate_recommendations(self, violations: List[PolicyViolation]) -> List[str]:
        """Generate recommendations based on violations"""
        recommendations = []
        
        critical_violations = [v for v in violations if v.severity == PolicySeverity.CRITICAL]
        if critical_violations:
            recommendations.append("Address critical violations immediately")
        
        security_violations = [v for v in violations if "security" in v.policy_name.lower()]
        if security_violations:
            recommendations.append("Review and enhance security controls")
        
        if len(violations) > 5:
            recommendations.append("Consider policy training for relevant teams")
        
        return recommendations
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report"""
        if not self.compliance_history:
            return {"status": "No compliance checks performed"}
        
        recent_checks = self.compliance_history[-10:]
        avg_score = sum(c.score for c in recent_checks) / len(recent_checks)
        
        all_violations = []
        for check in recent_checks:
            all_violations.extend(check.violations)
        
        violation_by_type = {}
        for violation in all_violations:
            policy_type = violation.policy_name
            violation_by_type[policy_type] = violation_by_type.get(policy_type, 0) + 1
        
        return {
            "total_checks": len(self.compliance_history),
            "recent_avg_score": round(avg_score, 3),
            "total_violations": len(all_violations),
            "violations_by_type": violation_by_type,
            "latest_status": recent_checks[-1].status.value
        }
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        return hashlib.md5(f"{datetime.now()}{random.random()}".encode()).hexdigest()[:12]


class PolicyEnforcementAgent:
    """
    Main agent that enforces policies and manages compliance
    
    Responsibilities:
    - Evaluate actions against policies
    - Enforce policy rules
    - Detect and handle violations
    - Manage compliance status
    - Generate compliance reports
    """
    
    def __init__(self, enforcement_mode: EnforcementMode = EnforcementMode.ENFORCING):
        self.policy_engine = PolicyEngine()
        self.violation_detector = ViolationDetector()
        self.compliance_manager = ComplianceManager(self.policy_engine)
        self.enforcement_mode = enforcement_mode
        self.action_log: List[Dict[str, Any]] = []
    
    def register_policy(self, policy: Policy, parent_id: Optional[str] = None) -> None:
        """Register a new policy"""
        self.policy_engine.add_policy(policy, parent_id)
        print(f"✓ Registered policy: {policy.name} ({policy.policy_type.value})")
    
    def evaluate_action(self, action: str, actor: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate if an action should be allowed based on policies
        
        Returns:
            Decision with allowed status, violations, and enforcement actions
        """
        policy_context = {
            "action": action,
            "actor": actor,
            **context
        }
        
        # Detect potential violations
        violation_categories = self.violation_detector.detect_violations(action, context)
        
        # Evaluate against all policies
        violations = self.policy_engine.evaluate_all(policy_context)
        
        # Calculate risk
        risk_score = self.violation_detector.calculate_risk_score(violations)
        
        # Determine if action is allowed
        allowed = self._determine_allowance(violations)
        
        # Take enforcement actions
        enforcement_actions = self._enforce(violations, allowed)
        
        result = {
            "action": action,
            "actor": actor,
            "allowed": allowed,
            "violations": violations,
            "violation_categories": violation_categories,
            "risk_score": round(risk_score, 3),
            "enforcement_actions": enforcement_actions,
            "timestamp": datetime.now()
        }
        
        self.action_log.append(result)
        return result
    
    def _determine_allowance(self, violations: List[PolicyViolation]) -> bool:
        """Determine if action should be allowed based on mode and violations"""
        if self.enforcement_mode == EnforcementMode.PERMISSIVE:
            return True
        
        if self.enforcement_mode == EnforcementMode.ADVISORY:
            # Allow unless critical violations
            return not any(v.severity == PolicySeverity.CRITICAL for v in violations)
        
        if self.enforcement_mode == EnforcementMode.ENFORCING:
            # Block if any error or critical violations
            return not any(v.severity in [PolicySeverity.ERROR, PolicySeverity.CRITICAL] 
                          for v in violations)
        
        # STRICT mode - block any violation
        return len(violations) == 0
    
    def _enforce(self, violations: List[PolicyViolation], allowed: bool) -> List[str]:
        """Take enforcement actions based on violations"""
        actions = []
        
        for violation in violations:
            if violation.action_taken == PolicyAction.LOG:
                actions.append(f"Logged violation: {violation.policy_name}")
            
            elif violation.action_taken == PolicyAction.WARN:
                actions.append(f"Warning issued for: {violation.policy_name}")
            
            elif violation.action_taken == PolicyAction.BLOCK:
                actions.append(f"Action blocked due to: {violation.policy_name}")
            
            elif violation.action_taken == PolicyAction.ESCALATE:
                actions.append(f"Escalated violation: {violation.policy_name}")
            
            elif violation.action_taken == PolicyAction.QUARANTINE:
                actions.append(f"Resource quarantined: {violation.policy_name}")
        
        return actions
    
    def check_compliance(self, context: Dict[str, Any]) -> ComplianceCheck:
        """Perform compliance check"""
        return self.compliance_manager.check_compliance(context)
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Get compliance report"""
        return self.compliance_manager.get_compliance_report()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        total_actions = len(self.action_log)
        allowed_actions = sum(1 for a in self.action_log if a["allowed"])
        blocked_actions = total_actions - allowed_actions
        
        all_violations = []
        for action in self.action_log:
            all_violations.extend(action["violations"])
        
        avg_risk = sum(a["risk_score"] for a in self.action_log) / total_actions if total_actions > 0 else 0
        
        return {
            "total_actions_evaluated": total_actions,
            "allowed_actions": allowed_actions,
            "blocked_actions": blocked_actions,
            "total_violations_detected": len(all_violations),
            "average_risk_score": round(avg_risk, 3),
            "enforcement_mode": self.enforcement_mode.value,
            "total_policies": len(self.policy_engine.policies),
            "enabled_policies": len([p for p in self.policy_engine.policies.values() if p.enabled])
        }


def demonstrate_policy_based_control():
    """Demonstrate the policy-based control pattern"""
    
    print("=" * 60)
    print("Policy-Based Control Agent Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = PolicyEnforcementAgent(enforcement_mode=EnforcementMode.ENFORCING)
    
    # Define example policies
    print("\n1. Registering Security Policies")
    print("-" * 60)
    
    # Security policy: Prevent access to sensitive data
    security_policy = Policy(
        policy_id="sec_001",
        name="Sensitive Data Access Control",
        description="Restrict access to sensitive data",
        policy_type=PolicyType.SECURITY,
        severity=PolicySeverity.CRITICAL,
        condition=lambda ctx: not ("sensitive" in ctx.get("action", "").lower() and 
                                   ctx.get("clearance_level", 0) < 3),
        action=PolicyAction.BLOCK,
        tags=["security", "data-protection"]
    )
    agent.register_policy(security_policy)
    
    # Compliance policy: Require approval for high-value transactions
    compliance_policy = Policy(
        policy_id="comp_001",
        name="High-Value Transaction Approval",
        description="Require approval for transactions over $10,000",
        policy_type=PolicyType.COMPLIANCE,
        severity=PolicySeverity.ERROR,
        condition=lambda ctx: not (ctx.get("transaction_amount", 0) > 10000 and 
                                   not ctx.get("approved", False)),
        action=PolicyAction.ESCALATE,
        tags=["compliance", "financial"]
    )
    agent.register_policy(compliance_policy)
    
    # Quality policy: Ensure minimum quality score
    quality_policy = Policy(
        policy_id="qual_001",
        name="Minimum Quality Standard",
        description="Ensure outputs meet quality threshold",
        policy_type=PolicyType.QUALITY,
        severity=PolicySeverity.WARNING,
        condition=lambda ctx: ctx.get("quality_score", 1.0) >= 0.7,
        action=PolicyAction.WARN,
        tags=["quality"]
    )
    agent.register_policy(quality_policy)
    
    # Operational policy: Rate limiting
    operational_policy = Policy(
        policy_id="ops_001",
        name="API Rate Limiting",
        description="Prevent excessive API calls",
        policy_type=PolicyType.OPERATIONAL,
        severity=PolicySeverity.WARNING,
        condition=lambda ctx: ctx.get("requests_per_minute", 0) <= 100,
        action=PolicyAction.WARN,
        tags=["operations", "rate-limiting"]
    )
    agent.register_policy(operational_policy)
    
    # Test scenarios
    print("\n2. Evaluating Actions Against Policies")
    print("-" * 60)
    
    # Scenario 1: Allowed action
    print("\nScenario 1: Standard data access (should be allowed)")
    result1 = agent.evaluate_action(
        action="read_user_profile",
        actor="user_123",
        context={"clearance_level": 2, "resource": "user_profiles"}
    )
    print(f"Action: {result1['action']}")
    print(f"Allowed: {result1['allowed']}")
    print(f"Violations: {len(result1['violations'])}")
    print(f"Risk Score: {result1['risk_score']}")
    
    # Scenario 2: Blocked - sensitive data without clearance
    print("\nScenario 2: Access sensitive data without clearance (should be blocked)")
    result2 = agent.evaluate_action(
        action="read_sensitive_medical_records",
        actor="user_456",
        context={"clearance_level": 1, "resource": "medical_records"}
    )
    print(f"Action: {result2['action']}")
    print(f"Allowed: {result2['allowed']}")
    print(f"Violations: {len(result2['violations'])}")
    if result2['violations']:
        print(f"Violation: {result2['violations'][0].policy_name}")
    print(f"Enforcement: {', '.join(result2['enforcement_actions'])}")
    
    # Scenario 3: High-value transaction without approval
    print("\nScenario 3: High-value transaction without approval (should escalate)")
    result3 = agent.evaluate_action(
        action="process_payment",
        actor="user_789",
        context={"transaction_amount": 15000, "approved": False}
    )
    print(f"Action: {result3['action']}")
    print(f"Allowed: {result3['allowed']}")
    print(f"Violations: {len(result3['violations'])}")
    if result3['violations']:
        print(f"Violation: {result3['violations'][0].policy_name}")
    print(f"Enforcement: {', '.join(result3['enforcement_actions'])}")
    
    # Scenario 4: Low quality output
    print("\nScenario 4: Low quality output (warning)")
    result4 = agent.evaluate_action(
        action="generate_report",
        actor="system",
        context={"quality_score": 0.6}
    )
    print(f"Action: {result4['action']}")
    print(f"Allowed: {result4['allowed']}")
    print(f"Violations: {len(result4['violations'])}")
    if result4['violations']:
        print(f"Violation: {result4['violations'][0].policy_name}")
    
    # Scenario 5: Rate limit exceeded
    print("\nScenario 5: Excessive API calls (rate limit)")
    result5 = agent.evaluate_action(
        action="api_call",
        actor="service_client",
        context={"requests_per_minute": 150}
    )
    print(f"Action: {result5['action']}")
    print(f"Allowed: {result5['allowed']}")
    print(f"Violations: {len(result5['violations'])}")
    
    # Compliance check
    print("\n3. Compliance Check")
    print("-" * 60)
    
    compliance_check = agent.check_compliance({
        "clearance_level": 2,
        "transaction_amount": 5000,
        "quality_score": 0.8,
        "requests_per_minute": 50
    })
    
    print(f"Compliance Status: {compliance_check.status.value}")
    print(f"Compliance Score: {compliance_check.score:.1%}")
    print(f"Policies Checked: {compliance_check.policies_checked}")
    print(f"Violations Found: {len(compliance_check.violations)}")
    if compliance_check.recommendations:
        print(f"Recommendations: {', '.join(compliance_check.recommendations)}")
    
    # Statistics
    print("\n4. Agent Statistics")
    print("-" * 60)
    
    stats = agent.get_statistics()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Compliance report
    print("\n5. Compliance Report")
    print("-" * 60)
    
    report = agent.get_compliance_report()
    for key, value in report.items():
        if isinstance(value, dict):
            print(f"{key.replace('_', ' ').title()}:")
            for k, v in value.items():
                print(f"  - {k}: {v}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_policy_based_control()
