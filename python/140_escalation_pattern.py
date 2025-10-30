"""
Agentic Design Pattern: Escalation Pattern

This pattern implements escalation handling for issues, requests, and incidents that
require higher-level attention. The agent routes problems to appropriate handlers based
on severity, priority, and escalation rules.

Category: Control & Governance
Use Cases:
- Support ticket escalation
- Incident management
- Exception handling
- Quality issues
- Security alerts
- Service level agreement (SLA) management
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import random


class Priority(Enum):
    """Priority levels for issues"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class Severity(Enum):
    """Severity levels for issues"""
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"
    CATASTROPHIC = "catastrophic"


class IssueStatus(Enum):
    """Status of an issue"""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class EscalationTrigger(Enum):
    """What triggers an escalation"""
    TIME_THRESHOLD = "time_threshold"
    SEVERITY = "severity"
    IMPACT = "impact"
    MANUAL = "manual"
    SLA_BREACH = "sla_breach"
    REPEATED_OCCURRENCE = "repeated_occurrence"


class HandlerLevel(Enum):
    """Handler tier levels"""
    TIER_1 = 1  # Basic support
    TIER_2 = 2  # Advanced support
    TIER_3 = 3  # Expert/Engineering
    MANAGEMENT = 4  # Management
    EXECUTIVE = 5  # Executive


@dataclass
class Issue:
    """Represents an issue that may need escalation"""
    issue_id: str
    title: str
    description: str
    priority: Priority
    severity: Severity
    category: str
    reported_by: str
    reported_at: datetime
    status: IssueStatus
    assigned_to: Optional[str] = None
    assigned_level: Optional[HandlerLevel] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class EscalationRule:
    """Defines when and how to escalate"""
    rule_id: str
    name: str
    description: str
    trigger: EscalationTrigger
    conditions: Dict[str, Any]
    target_level: HandlerLevel
    notify: List[str]  # Who to notify
    priority_boost: int = 0  # How much to increase priority
    enabled: bool = True


@dataclass
class EscalationEvent:
    """Records an escalation occurrence"""
    event_id: str
    issue_id: str
    timestamp: datetime
    from_level: Optional[HandlerLevel]
    to_level: HandlerLevel
    trigger: EscalationTrigger
    reason: str
    notified_parties: List[str]
    automated: bool = True


@dataclass
class Handler:
    """Represents a handler who can work on issues"""
    handler_id: str
    name: str
    level: HandlerLevel
    specialties: List[str]
    current_load: int = 0
    max_capacity: int = 10
    available: bool = True
    contact_info: Dict[str, str] = field(default_factory=dict)


@dataclass
class SLA:
    """Service Level Agreement definition"""
    sla_id: str
    name: str
    priority_thresholds: Dict[Priority, int]  # Priority -> response time in minutes
    resolution_thresholds: Dict[Priority, int]  # Priority -> resolution time in hours


class EscalationEngine:
    """Core engine for evaluating escalation rules"""
    
    def __init__(self):
        self.rules: Dict[str, EscalationRule] = {}
        self.escalation_history: List[EscalationEvent] = []
    
    def add_rule(self, rule: EscalationRule) -> None:
        """Add an escalation rule"""
        self.rules[rule.rule_id] = rule
    
    def evaluate_issue(self, issue: Issue, context: Dict[str, Any]) -> List[EscalationRule]:
        """Evaluate if issue should be escalated"""
        triggered_rules = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if self._check_rule(rule, issue, context):
                triggered_rules.append(rule)
        
        return triggered_rules
    
    def _check_rule(self, rule: EscalationRule, issue: Issue, context: Dict[str, Any]) -> bool:
        """Check if a rule is triggered"""
        
        if rule.trigger == EscalationTrigger.TIME_THRESHOLD:
            # Check if issue has been open too long
            time_open = (datetime.now() - issue.reported_at).total_seconds() / 60
            threshold = rule.conditions.get("minutes", 60)
            return time_open >= threshold
        
        elif rule.trigger == EscalationTrigger.SEVERITY:
            # Check if severity meets threshold
            severity_levels = ["minor", "moderate", "major", "critical", "catastrophic"]
            threshold = rule.conditions.get("min_severity", "major")
            return severity_levels.index(issue.severity.value) >= severity_levels.index(threshold)
        
        elif rule.trigger == EscalationTrigger.IMPACT:
            # Check impact score
            impact_score = context.get("impact_score", 0)
            threshold = rule.conditions.get("min_impact", 5)
            return impact_score >= threshold
        
        elif rule.trigger == EscalationTrigger.SLA_BREACH:
            # Check if SLA is breached
            return context.get("sla_breached", False)
        
        elif rule.trigger == EscalationTrigger.REPEATED_OCCURRENCE:
            # Check for repeated similar issues
            occurrences = context.get("similar_issues_count", 0)
            threshold = rule.conditions.get("min_occurrences", 3)
            return occurrences >= threshold
        
        return False
    
    def record_escalation(self, event: EscalationEvent) -> None:
        """Record an escalation event"""
        self.escalation_history.append(event)
    
    def get_escalation_stats(self) -> Dict[str, Any]:
        """Get escalation statistics"""
        total = len(self.escalation_history)
        
        if total == 0:
            return {"total_escalations": 0}
        
        by_trigger = {}
        by_level = {}
        automated = sum(1 for e in self.escalation_history if e.automated)
        
        for event in self.escalation_history:
            trigger = event.trigger.value
            by_trigger[trigger] = by_trigger.get(trigger, 0) + 1
            
            level = event.to_level.value
            by_level[level] = by_level.get(level, 0) + 1
        
        return {
            "total_escalations": total,
            "automated_escalations": automated,
            "manual_escalations": total - automated,
            "by_trigger": by_trigger,
            "by_target_level": by_level
        }


class HandlerPool:
    """Manages available handlers"""
    
    def __init__(self):
        self.handlers: Dict[str, Handler] = {}
    
    def add_handler(self, handler: Handler) -> None:
        """Add a handler to the pool"""
        self.handlers[handler.handler_id] = handler
    
    def find_handler(self, level: HandlerLevel, specialty: Optional[str] = None) -> Optional[Handler]:
        """Find an available handler at the specified level"""
        candidates = [
            h for h in self.handlers.values()
            if h.level == level and h.available and h.current_load < h.max_capacity
        ]
        
        # Filter by specialty if specified
        if specialty:
            candidates = [h for h in candidates if specialty in h.specialties]
        
        # Return handler with lowest load
        if candidates:
            return min(candidates, key=lambda h: h.current_load)
        
        return None
    
    def assign_issue(self, handler_id: str) -> bool:
        """Assign an issue to a handler"""
        if handler_id in self.handlers:
            handler = self.handlers[handler_id]
            if handler.current_load < handler.max_capacity:
                handler.current_load += 1
                return True
        return False
    
    def release_issue(self, handler_id: str) -> bool:
        """Release an issue from a handler"""
        if handler_id in self.handlers:
            handler = self.handlers[handler_id]
            if handler.current_load > 0:
                handler.current_load -= 1
                return True
        return False


class NotificationManager:
    """Manages escalation notifications"""
    
    def __init__(self):
        self.notifications: List[Dict[str, Any]] = []
    
    def send_notification(self, 
                         recipients: List[str],
                         subject: str,
                         message: str,
                         priority: Priority,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """Send an escalation notification"""
        
        notification = {
            "notification_id": self._generate_id(),
            "timestamp": datetime.now(),
            "recipients": recipients,
            "subject": subject,
            "message": message,
            "priority": priority.value,
            "metadata": metadata or {},
            "delivered": True  # Simulated
        }
        
        self.notifications.append(notification)
        
        print(f"📧 NOTIFICATION: {subject}")
        print(f"   To: {', '.join(recipients)}")
        print(f"   Priority: {priority.name}")
    
    def get_notification_history(self, recipient: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get notification history"""
        if recipient:
            return [n for n in self.notifications if recipient in n["recipients"]]
        return self.notifications
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        return hashlib.md5(f"{datetime.now()}{random.random()}".encode()).hexdigest()[:12]


class SLAManager:
    """Manages SLA compliance"""
    
    def __init__(self):
        self.slas: Dict[str, SLA] = {}
    
    def add_sla(self, sla: SLA) -> None:
        """Add an SLA definition"""
        self.slas[sla.sla_id] = sla
    
    def check_sla(self, issue: Issue, sla_id: str) -> Dict[str, Any]:
        """Check if issue is in SLA compliance"""
        
        if sla_id not in self.slas:
            return {"compliant": None, "reason": "SLA not found"}
        
        sla = self.slas[sla_id]
        
        # Check response time
        time_open_minutes = (datetime.now() - issue.reported_at).total_seconds() / 60
        response_threshold = sla.priority_thresholds.get(issue.priority, 60)
        
        response_compliant = time_open_minutes <= response_threshold
        
        return {
            "compliant": response_compliant,
            "time_open_minutes": round(time_open_minutes, 2),
            "response_threshold_minutes": response_threshold,
            "time_remaining_minutes": max(0, response_threshold - time_open_minutes)
        }


class EscalationAgent:
    """
    Main agent for escalation management
    
    Responsibilities:
    - Monitor issues for escalation triggers
    - Route escalated issues to appropriate handlers
    - Send notifications
    - Track SLA compliance
    - Manage escalation workflows
    """
    
    def __init__(self):
        self.escalation_engine = EscalationEngine()
        self.handler_pool = HandlerPool()
        self.notification_manager = NotificationManager()
        self.sla_manager = SLAManager()
        self.issues: Dict[str, Issue] = {}
    
    def register_issue(self, issue: Issue) -> None:
        """Register a new issue"""
        self.issues[issue.issue_id] = issue
        
        # Auto-assign to Tier 1 if not assigned
        if not issue.assigned_to:
            handler = self.handler_pool.find_handler(HandlerLevel.TIER_1)
            if handler:
                self.assign_issue(issue.issue_id, handler.handler_id)
        
        print(f"✓ Registered issue: {issue.title} (Priority: {issue.priority.name})")
    
    def add_escalation_rule(self, rule: EscalationRule) -> None:
        """Add an escalation rule"""
        self.escalation_engine.add_rule(rule)
        print(f"✓ Added escalation rule: {rule.name}")
    
    def add_handler(self, handler: Handler) -> None:
        """Add a handler"""
        self.handler_pool.add_handler(handler)
        print(f"✓ Added handler: {handler.name} (Level: {handler.level.name})")
    
    def check_escalation(self, issue_id: str, context: Optional[Dict[str, Any]] = None) -> List[EscalationEvent]:
        """Check if issue should be escalated"""
        
        if issue_id not in self.issues:
            return []
        
        issue = self.issues[issue_id]
        context = context or {}
        
        # Evaluate escalation rules
        triggered_rules = self.escalation_engine.evaluate_issue(issue, context)
        
        events = []
        for rule in triggered_rules:
            event = self._escalate_issue(issue, rule)
            events.append(event)
        
        return events
    
    def _escalate_issue(self, issue: Issue, rule: EscalationRule) -> EscalationEvent:
        """Escalate an issue"""
        
        from_level = issue.assigned_level
        to_level = rule.target_level
        
        # Find new handler
        handler = self.handler_pool.find_handler(to_level)
        
        # Reassign issue
        if issue.assigned_to:
            self.handler_pool.release_issue(issue.assigned_to)
        
        if handler:
            issue.assigned_to = handler.handler_id
            issue.assigned_level = to_level
            self.handler_pool.assign_issue(handler.handler_id)
        
        # Boost priority if specified
        if rule.priority_boost > 0:
            new_priority_value = min(issue.priority.value + rule.priority_boost, Priority.EMERGENCY.value)
            issue.priority = Priority(new_priority_value)
        
        # Update status
        issue.status = IssueStatus.ESCALATED
        
        # Create event
        event = EscalationEvent(
            event_id=self._generate_id(),
            issue_id=issue.issue_id,
            timestamp=datetime.now(),
            from_level=from_level,
            to_level=to_level,
            trigger=rule.trigger,
            reason=rule.description,
            notified_parties=rule.notify,
            automated=True
        )
        
        self.escalation_engine.record_escalation(event)
        
        # Send notifications
        self.notification_manager.send_notification(
            recipients=rule.notify,
            subject=f"ESCALATION: {issue.title}",
            message=f"Issue {issue.issue_id} has been escalated to {to_level.name}. Reason: {rule.description}",
            priority=issue.priority,
            metadata={"issue_id": issue.issue_id, "rule_id": rule.rule_id}
        )
        
        print(f"\n🔺 ESCALATED: {issue.title}")
        print(f"   From: {from_level.name if from_level else 'None'} → To: {to_level.name}")
        print(f"   Trigger: {rule.trigger.value}")
        print(f"   Assigned to: {handler.name if handler else 'Unassigned'}")
        
        return event
    
    def manual_escalate(self, issue_id: str, target_level: HandlerLevel, reason: str, notify: List[str]) -> Optional[EscalationEvent]:
        """Manually escalate an issue"""
        
        if issue_id not in self.issues:
            return None
        
        issue = self.issues[issue_id]
        from_level = issue.assigned_level
        
        # Find handler
        handler = self.handler_pool.find_handler(target_level)
        
        # Reassign
        if issue.assigned_to:
            self.handler_pool.release_issue(issue.assigned_to)
        
        if handler:
            issue.assigned_to = handler.handler_id
            issue.assigned_level = target_level
            self.handler_pool.assign_issue(handler.handler_id)
        
        issue.status = IssueStatus.ESCALATED
        
        # Create event
        event = EscalationEvent(
            event_id=self._generate_id(),
            issue_id=issue.issue_id,
            timestamp=datetime.now(),
            from_level=from_level,
            to_level=target_level,
            trigger=EscalationTrigger.MANUAL,
            reason=reason,
            notified_parties=notify,
            automated=False
        )
        
        self.escalation_engine.record_escalation(event)
        
        # Send notifications
        self.notification_manager.send_notification(
            recipients=notify,
            subject=f"MANUAL ESCALATION: {issue.title}",
            message=f"Issue {issue.issue_id} has been manually escalated. Reason: {reason}",
            priority=issue.priority,
            metadata={"issue_id": issue.issue_id}
        )
        
        return event
    
    def assign_issue(self, issue_id: str, handler_id: str) -> bool:
        """Assign an issue to a handler"""
        if issue_id not in self.issues:
            return False
        
        issue = self.issues[issue_id]
        
        if self.handler_pool.assign_issue(handler_id):
            issue.assigned_to = handler_id
            handler = self.handler_pool.handlers[handler_id]
            issue.assigned_level = handler.level
            issue.status = IssueStatus.IN_PROGRESS
            return True
        
        return False
    
    def resolve_issue(self, issue_id: str) -> bool:
        """Mark an issue as resolved"""
        if issue_id not in self.issues:
            return False
        
        issue = self.issues[issue_id]
        issue.status = IssueStatus.RESOLVED
        
        if issue.assigned_to:
            self.handler_pool.release_issue(issue.assigned_to)
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get escalation statistics"""
        total_issues = len(self.issues)
        
        status_counts = {}
        priority_counts = {}
        
        for issue in self.issues.values():
            status_counts[issue.status.value] = status_counts.get(issue.status.value, 0) + 1
            priority_counts[issue.priority.name] = priority_counts.get(issue.priority.name, 0) + 1
        
        escalation_stats = self.escalation_engine.get_escalation_stats()
        
        return {
            "total_issues": total_issues,
            "issues_by_status": status_counts,
            "issues_by_priority": priority_counts,
            "total_handlers": len(self.handler_pool.handlers),
            "escalation_stats": escalation_stats,
            "total_notifications": len(self.notification_manager.notifications)
        }
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        return hashlib.md5(f"{datetime.now()}{random.random()}".encode()).hexdigest()[:12]


def demonstrate_escalation():
    """Demonstrate the escalation pattern"""
    
    print("=" * 60)
    print("Escalation Pattern Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = EscalationAgent()
    
    # Setup handlers
    print("\n1. Setting Up Handler Pool")
    print("-" * 60)
    
    agent.add_handler(Handler("h1", "Alice (T1)", HandlerLevel.TIER_1, ["general", "basic"]))
    agent.add_handler(Handler("h2", "Bob (T2)", HandlerLevel.TIER_2, ["technical", "advanced"]))
    agent.add_handler(Handler("h3", "Charlie (T3)", HandlerLevel.TIER_3, ["expert", "engineering"]))
    agent.add_handler(Handler("h4", "Diana (Mgr)", HandlerLevel.MANAGEMENT, ["escalation", "management"]))
    
    # Setup escalation rules
    print("\n2. Configuring Escalation Rules")
    print("-" * 60)
    
    time_rule = EscalationRule(
        rule_id="rule_time",
        name="Time Threshold Escalation",
        description="Escalate if open for more than 30 minutes",
        trigger=EscalationTrigger.TIME_THRESHOLD,
        conditions={"minutes": 30},
        target_level=HandlerLevel.TIER_2,
        notify=["bob@example.com", "manager@example.com"]
    )
    agent.add_escalation_rule(time_rule)
    
    severity_rule = EscalationRule(
        rule_id="rule_severity",
        name="High Severity Escalation",
        description="Escalate critical issues immediately",
        trigger=EscalationTrigger.SEVERITY,
        conditions={"min_severity": "critical"},
        target_level=HandlerLevel.TIER_3,
        notify=["charlie@example.com", "diana@example.com"],
        priority_boost=1
    )
    agent.add_escalation_rule(severity_rule)
    
    # Register issues
    print("\n3. Registering Issues")
    print("-" * 60)
    
    issue1 = Issue(
        issue_id="ISS-001",
        title="Login page slow",
        description="Users reporting slow login page",
        priority=Priority.MEDIUM,
        severity=Severity.MODERATE,
        category="performance",
        reported_by="user123",
        reported_at=datetime.now() - timedelta(minutes=35),  # 35 minutes ago
        status=IssueStatus.NEW
    )
    agent.register_issue(issue1)
    
    issue2 = Issue(
        issue_id="ISS-002",
        title="Database connection failure",
        description="Production database unreachable",
        priority=Priority.CRITICAL,
        severity=Severity.CRITICAL,
        category="infrastructure",
        reported_by="monitoring",
        reported_at=datetime.now(),
        status=IssueStatus.NEW
    )
    agent.register_issue(issue2)
    
    issue3 = Issue(
        issue_id="ISS-003",
        title="Typo in documentation",
        description="Minor typo in user guide",
        priority=Priority.LOW,
        severity=Severity.MINOR,
        category="documentation",
        reported_by="user456",
        reported_at=datetime.now(),
        status=IssueStatus.NEW
    )
    agent.register_issue(issue3)
    
    # Check for escalations
    print("\n4. Checking for Automatic Escalations")
    print("-" * 60)
    
    # Check issue 1 (should escalate due to time)
    print("\nChecking ISS-001 (open for 35 minutes):")
    events1 = agent.check_escalation("ISS-001")
    
    # Check issue 2 (should escalate due to severity)
    print("\nChecking ISS-002 (critical severity):")
    events2 = agent.check_escalation("ISS-002")
    
    # Check issue 3 (should not escalate)
    print("\nChecking ISS-003 (low priority):")
    events3 = agent.check_escalation("ISS-003")
    if not events3:
        print("No escalation needed for ISS-003")
    
    # Manual escalation
    print("\n5. Manual Escalation")
    print("-" * 60)
    
    print("\nManually escalating ISS-001 to Management:")
    manual_event = agent.manual_escalate(
        issue_id="ISS-001",
        target_level=HandlerLevel.MANAGEMENT,
        reason="Customer VIP status requires management attention",
        notify=["diana@example.com", "executive@example.com"]
    )
    
    # Resolve some issues
    print("\n6. Resolving Issues")
    print("-" * 60)
    
    agent.resolve_issue("ISS-003")
    print("✓ Resolved ISS-003")
    
    # View issue statuses
    print("\n7. Current Issue Status")
    print("-" * 60)
    
    for issue_id, issue in agent.issues.items():
        print(f"\n{issue_id}: {issue.title}")
        print(f"  Status: {issue.status.value}")
        print(f"  Priority: {issue.priority.name}")
        print(f"  Severity: {issue.severity.value}")
        if issue.assigned_to:
            handler = agent.handler_pool.handlers[issue.assigned_to]
            print(f"  Assigned to: {handler.name} ({handler.level.name})")
    
    # Statistics
    print("\n8. Escalation Statistics")
    print("-" * 60)
    
    stats = agent.get_statistics()
    print(f"Total Issues: {stats['total_issues']}")
    print(f"\nIssues by Status:")
    for status, count in stats['issues_by_status'].items():
        print(f"  - {status}: {count}")
    print(f"\nIssues by Priority:")
    for priority, count in stats['issues_by_priority'].items():
        print(f"  - {priority}: {count}")
    print(f"\nEscalation Stats:")
    esc_stats = stats['escalation_stats']
    print(f"  Total Escalations: {esc_stats.get('total_escalations', 0)}")
    print(f"  Automated: {esc_stats.get('automated_escalations', 0)}")
    print(f"  Manual: {esc_stats.get('manual_escalations', 0)}")
    if 'by_trigger' in esc_stats:
        print(f"  By Trigger:")
        for trigger, count in esc_stats['by_trigger'].items():
            print(f"    - {trigger}: {count}")
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)
    print("\n✅ Control & Governance Category: COMPLETE (4/4 patterns)")


if __name__ == "__main__":
    demonstrate_escalation()
