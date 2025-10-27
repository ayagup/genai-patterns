"""
Agentic Design Pattern: Audit Trail & Logging

This pattern implements comprehensive logging and audit trail generation for agent actions.
The agent tracks all activities, maintains audit logs, generates compliance reports, and
enables forensic analysis of agent behavior.

Category: Control & Governance
Use Cases:
- Compliance auditing and reporting
- Security incident investigation
- Performance monitoring and analysis
- Debugging and troubleshooting
- Regulatory compliance (SOX, GDPR, HIPAA)
- Change tracking and accountability
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import json


class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventType(Enum):
    """Types of events to log"""
    ACTION = "action"
    DECISION = "decision"
    ERROR = "error"
    ACCESS = "access"
    CHANGE = "change"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    PERFORMANCE = "performance"


class AuditCategory(Enum):
    """Categories for audit events"""
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "configuration_change"
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_EVENT = "compliance_event"


@dataclass
class LogEntry:
    """Represents a single log entry"""
    log_id: str
    timestamp: datetime
    level: LogLevel
    event_type: EventType
    actor: str
    action: str
    resource: Optional[str]
    status: str  # success, failure, pending
    details: Dict[str, Any]
    duration_ms: Optional[int] = None
    error_message: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class AuditEntry:
    """Represents an audit trail entry"""
    audit_id: str
    timestamp: datetime
    category: AuditCategory
    actor: str
    action: str
    resource: str
    before_state: Optional[Dict[str, Any]]
    after_state: Optional[Dict[str, Any]]
    outcome: str
    risk_level: int  # 0-10
    compliance_relevant: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Compliance report generated from audit logs"""
    report_id: str
    period_start: datetime
    period_end: datetime
    total_events: int
    security_events: int
    compliance_events: int
    violations: List[str]
    high_risk_activities: List[AuditEntry]
    recommendations: List[str]
    generated_at: datetime


class LogStorage:
    """Manages log storage and retrieval"""
    
    def __init__(self, max_entries: int = 10000):
        self.logs: List[LogEntry] = []
        self.max_entries = max_entries
        self.indices: Dict[str, List[int]] = {
            "actor": {},
            "action": {},
            "level": {},
            "event_type": {}
        }
    
    def add_log(self, log: LogEntry) -> None:
        """Add a log entry"""
        self.logs.append(log)
        
        # Update indices
        self._update_index("actor", log.actor, len(self.logs) - 1)
        self._update_index("action", log.action, len(self.logs) - 1)
        self._update_index("level", log.level.value, len(self.logs) - 1)
        self._update_index("event_type", log.event_type.value, len(self.logs) - 1)
        
        # Rotate if needed
        if len(self.logs) > self.max_entries:
            self.logs = self.logs[-self.max_entries:]
            self._rebuild_indices()
    
    def _update_index(self, index_name: str, key: str, position: int) -> None:
        """Update an index"""
        if key not in self.indices[index_name]:
            self.indices[index_name][key] = []
        self.indices[index_name][key].append(position)
    
    def _rebuild_indices(self) -> None:
        """Rebuild all indices after rotation"""
        self.indices = {
            "actor": {},
            "action": {},
            "level": {},
            "event_type": {}
        }
        for i, log in enumerate(self.logs):
            self._update_index("actor", log.actor, i)
            self._update_index("action", log.action, i)
            self._update_index("level", log.level.value, i)
            self._update_index("event_type", log.event_type.value, i)
    
    def query(self, filters: Dict[str, Any]) -> List[LogEntry]:
        """Query logs with filters"""
        results = []
        
        for log in self.logs:
            match = True
            
            if "actor" in filters and log.actor != filters["actor"]:
                match = False
            if "action" in filters and log.action != filters["action"]:
                match = False
            if "level" in filters and log.level.value != filters["level"]:
                match = False
            if "event_type" in filters and log.event_type.value != filters["event_type"]:
                match = False
            if "start_time" in filters and log.timestamp < filters["start_time"]:
                match = False
            if "end_time" in filters and log.timestamp > filters["end_time"]:
                match = False
            
            if match:
                results.append(log)
        
        return results
    
    def get_recent(self, count: int = 10) -> List[LogEntry]:
        """Get most recent log entries"""
        return self.logs[-count:] if len(self.logs) >= count else self.logs


class AuditTrailManager:
    """Manages audit trail generation and storage"""
    
    def __init__(self):
        self.audit_entries: List[AuditEntry] = []
        self.compliance_logs: List[AuditEntry] = []
    
    def record_audit(self, entry: AuditEntry) -> None:
        """Record an audit entry"""
        self.audit_entries.append(entry)
        
        if entry.compliance_relevant:
            self.compliance_logs.append(entry)
    
    def get_audit_trail(self, 
                       resource: Optional[str] = None,
                       actor: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[AuditEntry]:
        """Get audit trail with filters"""
        results = []
        
        for entry in self.audit_entries:
            match = True
            
            if resource and entry.resource != resource:
                match = False
            if actor and entry.actor != actor:
                match = False
            if start_time and entry.timestamp < start_time:
                match = False
            if end_time and entry.timestamp > end_time:
                match = False
            
            if match:
                results.append(entry)
        
        return results
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalous patterns in audit trail"""
        anomalies = []
        
        # Check for unusual access patterns
        actor_actions = {}
        for entry in self.audit_entries[-100:]:  # Last 100 entries
            if entry.actor not in actor_actions:
                actor_actions[entry.actor] = []
            actor_actions[entry.actor].append(entry)
        
        for actor, actions in actor_actions.items():
            # Detect rapid succession of actions
            if len(actions) > 10:
                time_span = (actions[-1].timestamp - actions[0].timestamp).total_seconds()
                if time_span < 60:  # More than 10 actions in a minute
                    anomalies.append({
                        "type": "rapid_actions",
                        "actor": actor,
                        "count": len(actions),
                        "time_span_seconds": time_span
                    })
            
            # Detect high-risk activity clusters
            high_risk = [a for a in actions if a.risk_level >= 7]
            if len(high_risk) >= 3:
                anomalies.append({
                    "type": "high_risk_cluster",
                    "actor": actor,
                    "high_risk_count": len(high_risk)
                })
        
        return anomalies
    
    def generate_compliance_report(self, 
                                   start_date: datetime,
                                   end_date: datetime) -> ComplianceReport:
        """Generate a compliance report for a time period"""
        period_entries = [
            e for e in self.audit_entries
            if start_date <= e.timestamp <= end_date
        ]
        
        security_events = len([e for e in period_entries 
                              if e.category == AuditCategory.SECURITY_EVENT])
        compliance_events = len([e for e in period_entries 
                                if e.category == AuditCategory.COMPLIANCE_EVENT])
        
        # Identify violations (failed actions with high risk)
        violations = [
            f"{e.action} by {e.actor} on {e.resource}"
            for e in period_entries
            if e.outcome == "failure" and e.risk_level >= 7
        ]
        
        # High risk activities
        high_risk = [e for e in period_entries if e.risk_level >= 8]
        
        # Generate recommendations
        recommendations = []
        if len(violations) > 5:
            recommendations.append("Review access controls and permissions")
        if security_events > len(period_entries) * 0.1:
            recommendations.append("Investigate elevated security event rate")
        if len(high_risk) > 0:
            recommendations.append("Conduct security review of high-risk activities")
        
        return ComplianceReport(
            report_id=self._generate_id(),
            period_start=start_date,
            period_end=end_date,
            total_events=len(period_entries),
            security_events=security_events,
            compliance_events=compliance_events,
            violations=violations,
            high_risk_activities=high_risk[:5],  # Top 5
            recommendations=recommendations,
            generated_at=datetime.now()
        )
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        import random
        return hashlib.md5(f"{datetime.now()}{random.random()}".encode()).hexdigest()[:12]


class EventLogger:
    """Handles event logging with structured formatting"""
    
    def __init__(self, storage: LogStorage):
        self.storage = storage
    
    def log(self, 
            level: LogLevel,
            event_type: EventType,
            actor: str,
            action: str,
            status: str,
            details: Dict[str, Any],
            resource: Optional[str] = None,
            duration_ms: Optional[int] = None,
            error_message: Optional[str] = None,
            tags: Optional[List[str]] = None) -> LogEntry:
        """Create and store a log entry"""
        
        log_entry = LogEntry(
            log_id=self._generate_id(),
            timestamp=datetime.now(),
            level=level,
            event_type=event_type,
            actor=actor,
            action=action,
            resource=resource,
            status=status,
            details=details,
            duration_ms=duration_ms,
            error_message=error_message,
            tags=tags or []
        )
        
        self.storage.add_log(log_entry)
        return log_entry
    
    def log_action(self, actor: str, action: str, resource: str, 
                   status: str = "success", **kwargs) -> LogEntry:
        """Convenience method for logging actions"""
        return self.log(
            level=LogLevel.INFO,
            event_type=EventType.ACTION,
            actor=actor,
            action=action,
            resource=resource,
            status=status,
            details=kwargs
        )
    
    def log_error(self, actor: str, action: str, error: str, **kwargs) -> LogEntry:
        """Convenience method for logging errors"""
        return self.log(
            level=LogLevel.ERROR,
            event_type=EventType.ERROR,
            actor=actor,
            action=action,
            status="failure",
            error_message=error,
            details=kwargs
        )
    
    def log_security_event(self, actor: str, action: str, 
                          risk_level: int, **kwargs) -> LogEntry:
        """Convenience method for logging security events"""
        return self.log(
            level=LogLevel.WARNING if risk_level < 7 else LogLevel.CRITICAL,
            event_type=EventType.SECURITY,
            actor=actor,
            action=action,
            status="detected",
            details={"risk_level": risk_level, **kwargs},
            tags=["security"]
        )
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        import random
        return hashlib.md5(f"{datetime.now()}{random.random()}".encode()).hexdigest()[:12]


class AuditLoggingAgent:
    """
    Main agent for audit trail and logging management
    
    Responsibilities:
    - Log all agent actions and events
    - Maintain audit trails
    - Generate compliance reports
    - Detect anomalies
    - Provide forensic analysis capabilities
    """
    
    def __init__(self):
        self.log_storage = LogStorage()
        self.event_logger = EventLogger(self.log_storage)
        self.audit_trail = AuditTrailManager()
    
    def log_action(self, actor: str, action: str, resource: str,
                  before_state: Optional[Dict[str, Any]] = None,
                  after_state: Optional[Dict[str, Any]] = None,
                  risk_level: int = 3,
                  compliance_relevant: bool = False,
                  **kwargs) -> tuple[LogEntry, AuditEntry]:
        """Log an action with both event log and audit trail"""
        
        # Create event log
        status = kwargs.get("status", "success")
        log_entry = self.event_logger.log_action(
            actor=actor,
            action=action,
            resource=resource,
            status=status,
            **kwargs
        )
        
        # Create audit entry
        audit_entry = AuditEntry(
            audit_id=self._generate_id(),
            timestamp=datetime.now(),
            category=self._categorize_action(action),
            actor=actor,
            action=action,
            resource=resource,
            before_state=before_state,
            after_state=after_state,
            outcome=status,
            risk_level=risk_level,
            compliance_relevant=compliance_relevant,
            metadata=kwargs
        )
        
        self.audit_trail.record_audit(audit_entry)
        
        return log_entry, audit_entry
    
    def log_error(self, actor: str, action: str, error: str, **kwargs) -> LogEntry:
        """Log an error event"""
        return self.event_logger.log_error(actor, action, error, **kwargs)
    
    def log_security_event(self, actor: str, action: str, risk_level: int, **kwargs) -> LogEntry:
        """Log a security event"""
        return self.event_logger.log_security_event(actor, action, risk_level, **kwargs)
    
    def query_logs(self, filters: Dict[str, Any]) -> List[LogEntry]:
        """Query event logs"""
        return self.log_storage.query(filters)
    
    def get_audit_trail(self, **filters) -> List[AuditEntry]:
        """Get audit trail with filters"""
        return self.audit_trail.get_audit_trail(**filters)
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalous patterns"""
        return self.audit_trail.detect_anomalies()
    
    def generate_compliance_report(self, days_back: int = 30) -> ComplianceReport:
        """Generate compliance report for recent period"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        return self.audit_trail.generate_compliance_report(start_date, end_date)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        logs = self.log_storage.logs
        
        if not logs:
            return {"message": "No logs available"}
        
        level_counts = {}
        event_type_counts = {}
        
        for log in logs:
            level_counts[log.level.value] = level_counts.get(log.level.value, 0) + 1
            event_type_counts[log.event_type.value] = event_type_counts.get(log.event_type.value, 0) + 1
        
        return {
            "total_logs": len(logs),
            "total_audit_entries": len(self.audit_trail.audit_entries),
            "compliance_relevant_entries": len(self.audit_trail.compliance_logs),
            "logs_by_level": level_counts,
            "logs_by_event_type": event_type_counts,
            "oldest_log": logs[0].timestamp.isoformat() if logs else None,
            "newest_log": logs[-1].timestamp.isoformat() if logs else None
        }
    
    def _categorize_action(self, action: str) -> AuditCategory:
        """Categorize an action for audit purposes"""
        action_lower = action.lower()
        
        if any(word in action_lower for word in ["read", "access", "view"]):
            return AuditCategory.DATA_ACCESS
        elif any(word in action_lower for word in ["config", "setting", "update"]):
            return AuditCategory.CONFIGURATION_CHANGE
        elif any(word in action_lower for word in ["security", "auth", "permission"]):
            return AuditCategory.SECURITY_EVENT
        elif any(word in action_lower for word in ["compliance", "audit", "policy"]):
            return AuditCategory.COMPLIANCE_EVENT
        elif any(word in action_lower for word in ["user", "login", "logout"]):
            return AuditCategory.USER_ACTION
        else:
            return AuditCategory.SYSTEM_EVENT
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        import random
        return hashlib.md5(f"{datetime.now()}{random.random()}".encode()).hexdigest()[:12]


def demonstrate_audit_logging():
    """Demonstrate the audit trail and logging pattern"""
    
    print("=" * 60)
    print("Audit Trail & Logging Agent Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = AuditLoggingAgent()
    
    # Scenario 1: Normal user actions
    print("\n1. Logging Normal User Actions")
    print("-" * 60)
    
    log1, audit1 = agent.log_action(
        actor="user_001",
        action="read_document",
        resource="/docs/report_2024.pdf",
        before_state=None,
        after_state={"access_count": 1},
        risk_level=2,
        compliance_relevant=False
    )
    print(f"✓ Logged: {log1.action} by {log1.actor}")
    print(f"  Log ID: {log1.log_id}, Audit ID: {audit1.audit_id}")
    
    log2, audit2 = agent.log_action(
        actor="user_001",
        action="update_profile",
        resource="/users/user_001",
        before_state={"email": "old@example.com"},
        after_state={"email": "new@example.com"},
        risk_level=3,
        compliance_relevant=True
    )
    print(f"✓ Logged: {log2.action} by {log2.actor}")
    
    # Scenario 2: Security events
    print("\n2. Logging Security Events")
    print("-" * 60)
    
    sec_log1 = agent.log_security_event(
        actor="user_002",
        action="failed_login_attempt",
        risk_level=6,
        reason="invalid_password",
        ip_address="192.168.1.100"
    )
    print(f"⚠ Security Event: {sec_log1.action}")
    print(f"  Risk Level: {sec_log1.details['risk_level']}")
    
    sec_log2 = agent.log_security_event(
        actor="user_003",
        action="unauthorized_access_attempt",
        risk_level=9,
        resource="/admin/sensitive_data",
        blocked=True
    )
    print(f"🚨 Critical Security Event: {sec_log2.action}")
    print(f"  Level: {sec_log2.level.value}, Risk: {sec_log2.details['risk_level']}")
    
    # Scenario 3: Error logging
    print("\n3. Logging Errors")
    print("-" * 60)
    
    error_log = agent.log_error(
        actor="system",
        action="process_payment",
        error="Payment gateway timeout",
        transaction_id="TXN123456",
        amount=100.00
    )
    print(f"❌ Error: {error_log.error_message}")
    print(f"  Action: {error_log.action}, Status: {error_log.status}")
    
    # Scenario 4: Compliance-relevant actions
    print("\n4. Logging Compliance-Relevant Actions")
    print("-" * 60)
    
    comp_log, comp_audit = agent.log_action(
        actor="admin_001",
        action="delete_user_data",
        resource="/users/user_deleted/data",
        before_state={"records": 150},
        after_state={"records": 0},
        risk_level=8,
        compliance_relevant=True,
        reason="GDPR data deletion request",
        request_id="GDPR-2024-001"
    )
    print(f"📋 Compliance Action: {comp_log.action}")
    print(f"  Compliance Relevant: {comp_audit.compliance_relevant}")
    print(f"  Risk Level: {comp_audit.risk_level}")
    
    # Scenario 5: Query logs
    print("\n5. Querying Logs")
    print("-" * 60)
    
    # Query by actor
    user_logs = agent.query_logs({"actor": "user_001"})
    print(f"Logs for user_001: {len(user_logs)} entries")
    for log in user_logs:
        print(f"  - {log.action} at {log.timestamp.strftime('%H:%M:%S')}")
    
    # Query security events
    security_logs = agent.query_logs({"event_type": "security"})
    print(f"\nSecurity events: {len(security_logs)} entries")
    for log in security_logs:
        print(f"  - {log.action} (Risk: {log.details.get('risk_level', 'N/A')})")
    
    # Scenario 6: Audit trail
    print("\n6. Audit Trail Analysis")
    print("-" * 60)
    
    # Get audit trail for specific resource
    audit_trail = agent.get_audit_trail(actor="user_001")
    print(f"Audit entries for user_001: {len(audit_trail)}")
    for entry in audit_trail:
        print(f"  - {entry.action} on {entry.resource}")
        if entry.before_state and entry.after_state:
            print(f"    Changed from {entry.before_state} to {entry.after_state}")
    
    # Scenario 7: Anomaly detection
    print("\n7. Anomaly Detection")
    print("-" * 60)
    
    # Simulate rapid actions
    for i in range(15):
        agent.log_action(
            actor="suspicious_user",
            action=f"rapid_action_{i}",
            resource="/api/endpoint",
            risk_level=7 if i > 10 else 3
        )
    
    anomalies = agent.detect_anomalies()
    print(f"Detected {len(anomalies)} anomalies:")
    for anomaly in anomalies:
        print(f"  - Type: {anomaly['type']}")
        print(f"    Actor: {anomaly['actor']}")
        if 'count' in anomaly:
            print(f"    Actions: {anomaly['count']} in {anomaly.get('time_span_seconds', 0):.1f}s")
    
    # Scenario 8: Compliance report
    print("\n8. Compliance Report Generation")
    print("-" * 60)
    
    report = agent.generate_compliance_report(days_back=1)
    print(f"Report ID: {report.report_id}")
    print(f"Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}")
    print(f"Total Events: {report.total_events}")
    print(f"Security Events: {report.security_events}")
    print(f"Compliance Events: {report.compliance_events}")
    print(f"Violations: {len(report.violations)}")
    if report.violations:
        print("  Sample violations:")
        for violation in report.violations[:3]:
            print(f"    - {violation}")
    print(f"High Risk Activities: {len(report.high_risk_activities)}")
    if report.recommendations:
        print("Recommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")
    
    # Statistics
    print("\n9. Overall Statistics")
    print("-" * 60)
    
    stats = agent.get_statistics()
    for key, value in stats.items():
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
    demonstrate_audit_logging()
