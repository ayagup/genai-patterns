"""
Pattern 140: Escalation Pattern

Description:
    The Escalation Pattern implements mechanisms for identifying situations that
    require human intervention or higher-level decision-making and routing them
    appropriately through an escalation hierarchy. This pattern recognizes when an
    agent encounters scenarios beyond its capabilities, authority, or confidence
    level and escalates to appropriate handlers based on severity and context.

    Escalation is critical for AI safety and reliability, ensuring that complex,
    high-stakes, or ambiguous situations receive appropriate human oversight. The
    pattern supports multiple escalation levels, severity assessment, automatic
    triggering based on rules or confidence thresholds, and routing to appropriate
    stakeholders. It balances agent autonomy with human control.

    This implementation provides a comprehensive escalation system with severity
    assessment, trigger conditions, escalation routing, stakeholder management,
    escalation tracking, and resolution workflows. It supports both rule-based
    and confidence-based escalation with customizable escalation paths.

Components:
    - Severity Assessment: Evaluate situation severity
    - Trigger Detection: Identify when to escalate
    - Escalation Routing: Route to appropriate handler
    - Stakeholder Management: Manage escalation recipients
    - Escalation Tracking: Monitor escalation status
    - Resolution Workflow: Handle escalation lifecycle

Use Cases:
    - High-stakes decisions (medical, financial, legal)
    - Low-confidence predictions
    - Policy violations
    - Security incidents
    - Customer complaints
    - System errors
    - Ethical dilemmas
    - Edge cases

LangChain Implementation:
    This implementation uses:
    - LLM for severity assessment and decision complexity analysis
    - Rule-based escalation triggers
    - Confidence-threshold escalation
    - Hierarchical escalation routing
    - Escalation event tracking
    - Integration with human-in-the-loop systems

Benefits:
    - Ensures human oversight for critical decisions
    - Prevents autonomous errors in high-stakes scenarios
    - Provides safety net for uncertain situations
    - Enables graduated response based on severity
    - Supports compliance requirements
    - Facilitates learning from escalations
    - Maintains audit trail

Trade-offs:
    - Reduces full autonomy
    - Introduces latency for escalated cases
    - Requires human availability
    - May create bottlenecks
    - Needs clear escalation criteria
    - Can interrupt workflow

Production Considerations:
    - Define clear escalation criteria
    - Set appropriate confidence thresholds
    - Implement escalation routing rules
    - Ensure stakeholder availability
    - Track escalation metrics
    - Provide escalation status updates
    - Support escalation history
    - Enable escalation feedback
    - Monitor escalation patterns
    - Reduce false escalations
    - Support emergency escalations
    - Document escalation procedures
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class EscalationLevel(Enum):
    """Escalation severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EscalationStatus(Enum):
    """Status of escalation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


class EscalationTrigger(Enum):
    """Reasons for escalation."""
    LOW_CONFIDENCE = "low_confidence"
    POLICY_VIOLATION = "policy_violation"
    HIGH_RISK = "high_risk"
    SECURITY_INCIDENT = "security_incident"
    ERROR = "error"
    AMBIGUOUS = "ambiguous"
    SENSITIVE_DATA = "sensitive_data"
    MANUAL_REQUEST = "manual_request"
    THRESHOLD_EXCEEDED = "threshold_exceeded"


@dataclass
class Stakeholder:
    """Escalation stakeholder."""
    stakeholder_id: str
    name: str
    role: str
    email: str
    escalation_levels: List[EscalationLevel]
    availability: bool = True
    priority: int = 0


@dataclass
class EscalationCase:
    """Escalation case record."""
    case_id: str
    level: EscalationLevel
    trigger: EscalationTrigger
    title: str
    description: str
    context: Dict[str, Any]
    agent_id: str
    confidence_score: Optional[float]
    created_at: datetime
    assigned_to: Optional[str] = None
    status: EscalationStatus = EscalationStatus.PENDING
    resolution: Optional[str] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationRule:
    """Rule for triggering escalation."""
    rule_id: str
    name: str
    condition: str  # e.g., "confidence < 0.5", "contains sensitive_data"
    level: EscalationLevel
    trigger: EscalationTrigger
    enabled: bool = True


class EscalationAgent:
    """
    Agent that manages escalation of decisions and situations.
    
    This agent identifies when to escalate, assesses severity,
    routes to appropriate stakeholders, and tracks resolution.
    """
    
    def __init__(self, agent_id: str, temperature: float = 0.3):
        """Initialize the escalation agent."""
        self.agent_id = agent_id
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.stakeholders: Dict[str, Stakeholder] = {}
        self.cases: List[EscalationCase] = []
        self.rules: Dict[str, EscalationRule] = {}
        
        # Escalation thresholds
        self.confidence_threshold = 0.6
        self.risk_threshold = 0.7
        
        # Create severity assessment chain
        severity_prompt = ChatPromptTemplate.from_template(
            """You are an escalation triage specialist. Assess the severity of this situation.

Situation: {title}
Description: {description}
Context: {context}
Trigger: {trigger}

Evaluate the severity considering:
1. Potential impact (safety, financial, reputational)
2. Urgency (time-sensitive?)
3. Complexity (requires expert judgment?)
4. Risk level (high/medium/low)

Respond in this format:
SEVERITY: critical/high/medium/low
URGENCY: immediate/high/medium/low
REQUIRES_EXPERT: yes/no
RECOMMENDED_STAKEHOLDER: role description
REASONING: explanation"""
        )
        self.severity_chain = severity_prompt | self.llm | StrOutputParser()
    
    def add_stakeholder(self, stakeholder: Stakeholder) -> None:
        """Add escalation stakeholder."""
        self.stakeholders[stakeholder.stakeholder_id] = stakeholder
        print(f"Added stakeholder: {stakeholder.name} ({stakeholder.role})")
    
    def add_rule(self, rule: EscalationRule) -> None:
        """Add escalation rule."""
        self.rules[rule.rule_id] = rule
        print(f"Added escalation rule: {rule.name}")
    
    def should_escalate(
        self,
        context: Dict[str, Any],
        confidence: Optional[float] = None
    ) -> tuple[bool, Optional[EscalationTrigger], Optional[EscalationLevel]]:
        """
        Determine if situation should be escalated.
        
        Returns:
            Tuple of (should_escalate, trigger, level)
        """
        # Check confidence threshold
        if confidence is not None and confidence < self.confidence_threshold:
            return True, EscalationTrigger.LOW_CONFIDENCE, EscalationLevel.MEDIUM
        
        # Check rules
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if self._evaluate_rule(rule, context):
                return True, rule.trigger, rule.level
        
        # Check for explicit escalation flags
        if context.get("escalate", False):
            return True, EscalationTrigger.MANUAL_REQUEST, EscalationLevel.HIGH
        
        if context.get("policy_violation", False):
            return True, EscalationTrigger.POLICY_VIOLATION, EscalationLevel.HIGH
        
        if context.get("security_incident", False):
            return True, EscalationTrigger.SECURITY_INCIDENT, EscalationLevel.CRITICAL
        
        return False, None, None
    
    def _evaluate_rule(
        self,
        rule: EscalationRule,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate if rule condition is met."""
        condition = rule.condition.lower()
        
        # Simple condition evaluation
        if "sensitive_data" in condition:
            return context.get("has_sensitive_data", False)
        
        if "high_risk" in condition:
            risk_score = context.get("risk_score", 0)
            return risk_score > self.risk_threshold
        
        if "error" in condition:
            return context.get("error_occurred", False)
        
        return False
    
    def create_escalation(
        self,
        title: str,
        description: str,
        context: Dict[str, Any],
        trigger: EscalationTrigger,
        level: Optional[EscalationLevel] = None,
        confidence: Optional[float] = None
    ) -> EscalationCase:
        """Create an escalation case."""
        # Assess severity if not provided
        if level is None:
            level = self._assess_severity(title, description, context, trigger)
        
        case = EscalationCase(
            case_id=f"esc_{len(self.cases) + 1}_{datetime.now().timestamp()}",
            level=level,
            trigger=trigger,
            title=title,
            description=description,
            context=context,
            agent_id=self.agent_id,
            confidence_score=confidence,
            created_at=datetime.now()
        )
        
        # Route to stakeholder
        stakeholder = self._route_escalation(case)
        if stakeholder:
            case.assigned_to = stakeholder.stakeholder_id
        
        self.cases.append(case)
        return case
    
    def _assess_severity(
        self,
        title: str,
        description: str,
        context: Dict[str, Any],
        trigger: EscalationTrigger
    ) -> EscalationLevel:
        """Assess severity using LLM."""
        try:
            result = self.severity_chain.invoke({
                "title": title,
                "description": description,
                "context": str(context),
                "trigger": trigger.value
            })
            
            # Parse severity
            for line in result.split('\n'):
                if line.startswith("SEVERITY:"):
                    severity = line.replace("SEVERITY:", "").strip().lower()
                    if severity == "critical":
                        return EscalationLevel.CRITICAL
                    elif severity == "high":
                        return EscalationLevel.HIGH
                    elif severity == "medium":
                        return EscalationLevel.MEDIUM
                    else:
                        return EscalationLevel.LOW
            
            return EscalationLevel.MEDIUM
            
        except Exception as e:
            print(f"Error assessing severity: {e}")
            return EscalationLevel.MEDIUM
    
    def _route_escalation(
        self,
        case: EscalationCase
    ) -> Optional[Stakeholder]:
        """Route escalation to appropriate stakeholder."""
        # Find available stakeholders for this level
        candidates = [
            s for s in self.stakeholders.values()
            if case.level in s.escalation_levels and s.availability
        ]
        
        if not candidates:
            return None
        
        # Sort by priority and return highest
        candidates.sort(key=lambda s: s.priority, reverse=True)
        return candidates[0]
    
    def update_case_status(
        self,
        case_id: str,
        status: EscalationStatus,
        resolution: Optional[str] = None
    ) -> bool:
        """Update escalation case status."""
        case = next((c for c in self.cases if c.case_id == case_id), None)
        if not case:
            return False
        
        case.status = status
        if resolution:
            case.resolution = resolution
        
        if status == EscalationStatus.RESOLVED:
            case.resolved_at = datetime.now()
        
        return True
    
    def get_pending_escalations(
        self,
        level: Optional[EscalationLevel] = None,
        stakeholder_id: Optional[str] = None
    ) -> List[EscalationCase]:
        """Get pending escalation cases."""
        pending = [c for c in self.cases if c.status == EscalationStatus.PENDING]
        
        if level:
            pending = [c for c in pending if c.level == level]
        
        if stakeholder_id:
            pending = [c for c in pending if c.assigned_to == stakeholder_id]
        
        return pending
    
    def get_escalation_metrics(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get escalation metrics."""
        cutoff = datetime.now() - timedelta(days=days)
        recent_cases = [c for c in self.cases if c.created_at >= cutoff]
        
        # Resolution time for resolved cases
        resolved = [c for c in recent_cases if c.status == EscalationStatus.RESOLVED]
        avg_resolution_time = None
        if resolved:
            times = [
                (c.resolved_at - c.created_at).total_seconds() / 3600
                for c in resolved if c.resolved_at
            ]
            avg_resolution_time = sum(times) / len(times) if times else None
        
        return {
            "period_days": days,
            "total_escalations": len(recent_cases),
            "pending": len([c for c in recent_cases if c.status == EscalationStatus.PENDING]),
            "in_progress": len([c for c in recent_cases if c.status == EscalationStatus.IN_PROGRESS]),
            "resolved": len(resolved),
            "by_level": {
                level.value: len([c for c in recent_cases if c.level == level])
                for level in EscalationLevel
            },
            "by_trigger": {
                trigger.value: len([c for c in recent_cases if c.trigger == trigger])
                for trigger in EscalationTrigger
            },
            "avg_resolution_hours": avg_resolution_time
        }


def demonstrate_escalation():
    """Demonstrate the escalation pattern."""
    print("=" * 80)
    print("Escalation Pattern Demonstration")
    print("=" * 80)
    
    agent = EscalationAgent(agent_id="agent_001")
    
    # Demonstration 1: Add Stakeholders
    print("\n" + "=" * 80)
    print("Demonstration 1: Configure Escalation Stakeholders")
    print("=" * 80)
    
    support = Stakeholder(
        stakeholder_id="stake_001",
        name="Support Team",
        role="support",
        email="support@company.com",
        escalation_levels=[EscalationLevel.LOW, EscalationLevel.MEDIUM],
        priority=10
    )
    
    manager = Stakeholder(
        stakeholder_id="stake_002",
        name="Engineering Manager",
        role="manager",
        email="manager@company.com",
        escalation_levels=[EscalationLevel.MEDIUM, EscalationLevel.HIGH],
        priority=20
    )
    
    director = Stakeholder(
        stakeholder_id="stake_003",
        name="Director",
        role="director",
        email="director@company.com",
        escalation_levels=[EscalationLevel.HIGH, EscalationLevel.CRITICAL],
        priority=30
    )
    
    agent.add_stakeholder(support)
    agent.add_stakeholder(manager)
    agent.add_stakeholder(director)
    
    # Demonstration 2: Add Escalation Rules
    print("\n" + "=" * 80)
    print("Demonstration 2: Define Escalation Rules")
    print("=" * 80)
    
    sensitive_rule = EscalationRule(
        rule_id="rule_001",
        name="Sensitive Data Rule",
        condition="contains sensitive_data",
        level=EscalationLevel.HIGH,
        trigger=EscalationTrigger.SENSITIVE_DATA
    )
    
    risk_rule = EscalationRule(
        rule_id="rule_002",
        name="High Risk Rule",
        condition="high_risk",
        level=EscalationLevel.HIGH,
        trigger=EscalationTrigger.HIGH_RISK
    )
    
    agent.add_rule(sensitive_rule)
    agent.add_rule(risk_rule)
    
    # Demonstration 3: Low Confidence Escalation
    print("\n" + "=" * 80)
    print("Demonstration 3: Low Confidence Escalation")
    print("=" * 80)
    
    should_escalate, trigger, level = agent.should_escalate(
        context={"task": "medical_diagnosis"},
        confidence=0.45
    )
    
    print(f"Should escalate: {should_escalate}")
    print(f"Trigger: {trigger.value if trigger else None}")
    print(f"Level: {level.value if level else None}")
    
    if should_escalate:
        case1 = agent.create_escalation(
            title="Low Confidence Medical Diagnosis",
            description="Agent confidence below threshold for medical diagnosis",
            context={"task": "medical_diagnosis", "symptoms": "fever, cough"},
            trigger=trigger,
            level=level,
            confidence=0.45
        )
        print(f"\nCreated escalation case: {case1.case_id}")
        print(f"Assigned to: {case1.assigned_to}")
        print(f"Status: {case1.status.value}")
    
    # Demonstration 4: Security Incident Escalation
    print("\n" + "=" * 80)
    print("Demonstration 4: Security Incident Escalation")
    print("=" * 80)
    
    case2 = agent.create_escalation(
        title="Unauthorized Access Attempt",
        description="Multiple failed login attempts detected",
        context={
            "security_incident": True,
            "source_ip": "192.168.1.100",
            "attempts": 10
        },
        trigger=EscalationTrigger.SECURITY_INCIDENT,
        level=EscalationLevel.CRITICAL
    )
    
    print(f"Created critical escalation: {case2.case_id}")
    print(f"Level: {case2.level.value}")
    print(f"Trigger: {case2.trigger.value}")
    print(f"Assigned to: {case2.assigned_to}")
    
    # Demonstration 5: Rule-Based Escalation
    print("\n" + "=" * 80)
    print("Demonstration 5: Rule-Based Escalation")
    print("=" * 80)
    
    should_escalate, trigger, level = agent.should_escalate(
        context={
            "has_sensitive_data": True,
            "data_type": "PII"
        }
    )
    
    if should_escalate:
        case3 = agent.create_escalation(
            title="Sensitive Data Access",
            description="Request to access personally identifiable information",
            context={"has_sensitive_data": True, "data_type": "PII"},
            trigger=trigger,
            level=level
        )
        print(f"Created escalation: {case3.case_id}")
        print(f"Trigger: {trigger.value}")
        print(f"Level: {level.value}")
    
    # Demonstration 6: Manual Escalation Request
    print("\n" + "=" * 80)
    print("Demonstration 6: Manual Escalation Request")
    print("=" * 80)
    
    case4 = agent.create_escalation(
        title="Complex Customer Complaint",
        description="Customer requesting refund for premium service",
        context={
            "escalate": True,
            "customer_id": "cust_12345",
            "amount": 5000
        },
        trigger=EscalationTrigger.MANUAL_REQUEST
    )
    
    print(f"Manual escalation: {case4.case_id}")
    print(f"Level: {case4.level.value}")
    print(f"Assigned to: {case4.assigned_to}")
    
    # Demonstration 7: Case Resolution
    print("\n" + "=" * 80)
    print("Demonstration 7: Escalation Resolution")
    print("=" * 80)
    
    print(f"\nResolving case: {case3.case_id}")
    agent.update_case_status(
        case_id=case3.case_id,
        status=EscalationStatus.RESOLVED,
        resolution="Access granted with additional monitoring"
    )
    
    resolved_case = next(c for c in agent.cases if c.case_id == case3.case_id)
    print(f"Status: {resolved_case.status.value}")
    print(f"Resolution: {resolved_case.resolution}")
    
    # Demonstration 8: Pending Escalations
    print("\n" + "=" * 80)
    print("Demonstration 8: Pending Escalations")
    print("=" * 80)
    
    pending = agent.get_pending_escalations()
    print(f"\nTotal pending escalations: {len(pending)}")
    
    for case in pending:
        print(f"\n  Case: {case.case_id}")
        print(f"    Title: {case.title}")
        print(f"    Level: {case.level.value}")
        print(f"    Assigned to: {case.assigned_to}")
        print(f"    Age: {(datetime.now() - case.created_at).seconds // 60} minutes")
    
    # Demonstration 9: Critical Escalations Only
    print("\n" + "=" * 80)
    print("Demonstration 9: Critical Escalations")
    print("=" * 80)
    
    critical = agent.get_pending_escalations(level=EscalationLevel.CRITICAL)
    print(f"\nCritical pending escalations: {len(critical)}")
    for case in critical:
        print(f"  - {case.title}")
    
    # Demonstration 10: Escalation Metrics
    print("\n" + "=" * 80)
    print("Demonstration 10: Escalation Metrics")
    print("=" * 80)
    
    metrics = agent.get_escalation_metrics(days=7)
    print(f"\nEscalation Metrics (last {metrics['period_days']} days):")
    print(f"  Total Escalations: {metrics['total_escalations']}")
    print(f"  Pending: {metrics['pending']}")
    print(f"  In Progress: {metrics['in_progress']}")
    print(f"  Resolved: {metrics['resolved']}")
    print(f"\n  By Level:")
    for level, count in metrics['by_level'].items():
        if count > 0:
            print(f"    {level}: {count}")
    print(f"\n  By Trigger:")
    for trigger, count in metrics['by_trigger'].items():
        if count > 0:
            print(f"    {trigger}: {count}")
    
    if metrics['avg_resolution_hours']:
        print(f"\n  Avg Resolution Time: {metrics['avg_resolution_hours']:.1f} hours")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary: Escalation Pattern")
    print("=" * 80)
    print("""
The Escalation Pattern ensures appropriate human oversight for critical situations:

Key Features Demonstrated:
1. Stakeholder Management - Configure escalation recipients by level
2. Escalation Rules - Define automatic escalation triggers
3. Confidence-Based Escalation - Escalate low-confidence decisions
4. Security Escalation - Handle security incidents
5. Rule-Based Escalation - Trigger based on context rules
6. Manual Escalation - Support explicit escalation requests
7. Case Resolution - Track escalation lifecycle
8. Pending Cases - Monitor unresolved escalations
9. Level Filtering - Focus on critical cases
10. Metrics Tracking - Monitor escalation patterns

Benefits:
- Ensures human oversight for critical decisions
- Prevents autonomous errors
- Provides safety net for uncertainty
- Graduated response by severity
- Supports compliance
- Facilitates learning
- Maintains audit trail

Best Practices:
- Define clear escalation criteria
- Set appropriate thresholds
- Implement routing rules
- Ensure stakeholder availability
- Track metrics
- Provide status updates
- Support escalation history
- Enable feedback loops
- Monitor patterns
- Reduce false escalations
- Support emergency escalations
- Document procedures

This pattern is essential for safe AI systems requiring human oversight,
especially in high-stakes domains like healthcare, finance, and security.
""")


if __name__ == "__main__":
    demonstrate_escalation()
