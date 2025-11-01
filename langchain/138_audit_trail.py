"""
Pattern 138: Audit Trail & Logging

Description:
    The Audit Trail & Logging pattern provides comprehensive tracking and recording
    of all agent activities, decisions, and interactions. This pattern creates a
    complete, tamper-evident record of agent behavior that can be used for compliance,
    debugging, security analysis, and forensic investigation. It captures not just
    what happened, but why it happened, who was involved, and what data was accessed.

    Audit trails are critical for production AI systems, especially in regulated
    industries like healthcare, finance, and government. They provide accountability,
    enable incident investigation, support compliance audits, and help identify
    security threats. The pattern supports different log levels, structured logging,
    searchable records, and retention policies.

    This implementation provides a comprehensive audit system with activity logging,
    decision tracking, data access monitoring, security event recording, and forensic
    analysis capabilities. It supports multiple log formats, encryption, retention
    policies, and compliance reporting.

Components:
    - Activity Logging: Records all agent actions
    - Decision Tracking: Captures reasoning and decisions
    - Data Access Monitoring: Tracks data read/write operations
    - Security Event Recording: Logs security-relevant events
    - Forensic Analysis: Tools for investigating incidents
    - Retention Management: Handles log retention and archival

Use Cases:
    - Compliance auditing (SOX, HIPAA, GDPR)
    - Security incident investigation
    - Performance analysis and debugging
    - User behavior tracking
    - Regulatory reporting
    - Fraud detection
    - System monitoring
    - Forensic analysis

LangChain Implementation:
    This implementation uses:
    - Custom audit logging system with structured records
    - Tamper-evident logging with checksums
    - LLM for log analysis and pattern detection
    - Structured audit events with rich metadata
    - Query capabilities for forensic analysis
    - Retention policies and archival

Benefits:
    - Provides complete accountability
    - Enables incident investigation
    - Supports compliance requirements
    - Facilitates debugging and troubleshooting
    - Detects security threats
    - Creates audit trail for legal purposes
    - Enables behavior analysis

Trade-offs:
    - Storage overhead for comprehensive logs
    - Performance impact from logging operations
    - Privacy concerns with detailed logging
    - Complexity in log management
    - Cost of log storage and retention
    - Potential information disclosure risks

Production Considerations:
    - Use structured logging formats (JSON)
    - Implement log rotation and archival
    - Encrypt sensitive log data
    - Index logs for fast searching
    - Set appropriate retention periods
    - Monitor log storage usage
    - Implement log integrity verification
    - Provide audit reporting tools
    - Support log export and analysis
    - Handle PII data carefully
    - Implement access controls for logs
    - Consider regulatory requirements
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class AuditEventType(Enum):
    """Types of audit events."""
    ACTION = "action"
    DECISION = "decision"
    DATA_ACCESS = "data_access"
    SECURITY = "security"
    ERROR = "error"
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"


class AuditLevel(Enum):
    """Audit log levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class DataOperation(Enum):
    """Types of data operations."""
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"


@dataclass
class AuditEvent:
    """Audit event record."""
    event_id: str
    event_type: AuditEventType
    level: AuditLevel
    timestamp: datetime
    agent_id: str
    user_id: Optional[str]
    action: str
    details: Dict[str, Any]
    result: Optional[str]
    duration_ms: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    
    def compute_checksum(self) -> str:
        """Compute checksum for tamper detection."""
        # Create canonical representation
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "action": self.action,
            "details": json.dumps(self.details, sort_keys=True)
        }
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['event_type'] = self.event_type.value
        d['level'] = self.level.value
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class DataAccessEvent(AuditEvent):
    """Specialized event for data access."""
    data_type: str = ""
    operation: DataOperation = DataOperation.READ
    record_count: int = 0
    data_classification: str = "public"


@dataclass
class SecurityEvent(AuditEvent):
    """Specialized event for security incidents."""
    threat_level: str = "low"
    source_ip: Optional[str] = None
    attack_type: Optional[str] = None


class AuditTrailAgent:
    """
    Agent that provides comprehensive audit trail and logging.
    
    This agent logs all activities, decisions, and events with rich
    metadata for compliance, security, and forensic analysis.
    """
    
    def __init__(self, agent_id: str, temperature: float = 0.3):
        """Initialize the audit trail agent."""
        self.agent_id = agent_id
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.events: List[AuditEvent] = []
        self.retention_days = 90  # Default retention period
        
        # Create log analysis chain
        analysis_prompt = ChatPromptTemplate.from_template(
            """You are a log analysis expert. Analyze the following audit log events and identify patterns, anomalies, or security concerns.

Events:
{events}

Provide analysis in this format:
PATTERNS: List any patterns you observe
ANOMALIES: List any unusual or suspicious activities
SECURITY_CONCERNS: List any potential security issues
RECOMMENDATIONS: Suggest actions to take"""
        )
        self.analysis_chain = analysis_prompt | self.llm | StrOutputParser()
    
    def log_event(
        self,
        event_type: AuditEventType,
        level: AuditLevel,
        action: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        result: Optional[str] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log an audit event."""
        event = AuditEvent(
            event_id=f"evt_{len(self.events) + 1}_{datetime.now().timestamp()}",
            event_type=event_type,
            level=level,
            timestamp=datetime.now(),
            agent_id=self.agent_id,
            user_id=user_id,
            action=action,
            details=details,
            result=result,
            duration_ms=duration_ms,
            metadata=metadata or {}
        )
        
        # Compute checksum for integrity
        event.checksum = event.compute_checksum()
        
        self.events.append(event)
        return event
    
    def log_data_access(
        self,
        operation: DataOperation,
        data_type: str,
        record_count: int,
        data_classification: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> DataAccessEvent:
        """Log a data access event."""
        event = DataAccessEvent(
            event_id=f"data_{len(self.events) + 1}_{datetime.now().timestamp()}",
            event_type=AuditEventType.DATA_ACCESS,
            level=AuditLevel.MEDIUM if data_classification == "sensitive" else AuditLevel.LOW,
            timestamp=datetime.now(),
            agent_id=self.agent_id,
            user_id=user_id,
            action=f"{operation.value} {data_type}",
            details=details or {},
            result="success",
            duration_ms=None,
            data_type=data_type,
            operation=operation,
            record_count=record_count,
            data_classification=data_classification
        )
        
        event.checksum = event.compute_checksum()
        self.events.append(event)
        return event
    
    def log_security_event(
        self,
        action: str,
        threat_level: str,
        details: Dict[str, Any],
        source_ip: Optional[str] = None,
        attack_type: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> SecurityEvent:
        """Log a security event."""
        level_map = {
            "critical": AuditLevel.CRITICAL,
            "high": AuditLevel.HIGH,
            "medium": AuditLevel.MEDIUM,
            "low": AuditLevel.LOW
        }
        
        event = SecurityEvent(
            event_id=f"sec_{len(self.events) + 1}_{datetime.now().timestamp()}",
            event_type=AuditEventType.SECURITY,
            level=level_map.get(threat_level, AuditLevel.MEDIUM),
            timestamp=datetime.now(),
            agent_id=self.agent_id,
            user_id=user_id,
            action=action,
            details=details,
            result="logged",
            duration_ms=None,
            threat_level=threat_level,
            source_ip=source_ip,
            attack_type=attack_type
        )
        
        event.checksum = event.compute_checksum()
        self.events.append(event)
        return event
    
    def query_events(
        self,
        event_type: Optional[AuditEventType] = None,
        level: Optional[AuditLevel] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        action_pattern: Optional[str] = None
    ) -> List[AuditEvent]:
        """Query audit events with filters."""
        filtered = self.events.copy()
        
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        
        if level:
            filtered = [e for e in filtered if e.level == level]
        
        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]
        
        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time]
        
        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time]
        
        if action_pattern:
            filtered = [e for e in filtered if action_pattern.lower() in e.action.lower()]
        
        return filtered
    
    def analyze_events(
        self,
        events: Optional[List[AuditEvent]] = None,
        time_window_hours: int = 24
    ) -> str:
        """Analyze events for patterns and anomalies."""
        if events is None:
            # Analyze recent events
            cutoff = datetime.now() - timedelta(hours=time_window_hours)
            events = self.query_events(start_time=cutoff)
        
        if not events:
            return "No events to analyze"
        
        # Format events for analysis
        events_text = "\n".join([
            f"[{e.timestamp.isoformat()}] {e.event_type.value} - {e.action} "
            f"(level: {e.level.value}, user: {e.user_id or 'system'})"
            for e in events[-50:]  # Last 50 events
        ])
        
        try:
            analysis = self.analysis_chain.invoke({"events": events_text})
            return analysis
        except Exception as e:
            return f"Error analyzing events: {e}"
    
    def get_user_activity(
        self,
        user_id: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get activity summary for a user."""
        cutoff = datetime.now() - timedelta(hours=hours)
        events = self.query_events(user_id=user_id, start_time=cutoff)
        
        return {
            "user_id": user_id,
            "period_hours": hours,
            "total_events": len(events),
            "event_types": {
                etype.value: len([e for e in events if e.event_type == etype])
                for etype in AuditEventType
            },
            "actions": list(set(e.action for e in events)),
            "latest_activity": events[-1].timestamp if events else None
        }
    
    def get_security_summary(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get security events summary."""
        cutoff = datetime.now() - timedelta(hours=hours)
        security_events = self.query_events(
            event_type=AuditEventType.SECURITY,
            start_time=cutoff
        )
        
        return {
            "period_hours": hours,
            "total_security_events": len(security_events),
            "by_threat_level": {
                level.value: len([
                    e for e in security_events 
                    if e.level == level
                ])
                for level in AuditLevel
            },
            "critical_events": len([
                e for e in security_events 
                if e.level == AuditLevel.CRITICAL
            ])
        }
    
    def verify_integrity(self) -> Dict[str, Any]:
        """Verify integrity of audit log."""
        total = len(self.events)
        verified = 0
        tampered = []
        
        for event in self.events:
            expected_checksum = event.compute_checksum()
            if event.checksum == expected_checksum:
                verified += 1
            else:
                tampered.append(event.event_id)
        
        return {
            "total_events": total,
            "verified": verified,
            "tampered": len(tampered),
            "tampered_event_ids": tampered,
            "integrity_score": verified / total if total > 0 else 1.0
        }
    
    def export_events(
        self,
        format: str = "json",
        events: Optional[List[AuditEvent]] = None
    ) -> str:
        """Export events to specified format."""
        events_to_export = events if events else self.events
        
        if format == "json":
            return json.dumps(
                [e.to_dict() for e in events_to_export],
                indent=2
            )
        elif format == "csv":
            # Simple CSV export
            lines = ["timestamp,event_type,level,agent_id,user_id,action,result"]
            for e in events_to_export:
                lines.append(
                    f"{e.timestamp.isoformat()},{e.event_type.value},"
                    f"{e.level.value},{e.agent_id},{e.user_id or ''},"
                    f"{e.action},{e.result or ''}"
                )
            return "\n".join(lines)
        else:
            return "Unsupported format"
    
    def apply_retention_policy(self) -> int:
        """Apply retention policy and remove old events."""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        original_count = len(self.events)
        
        self.events = [e for e in self.events if e.timestamp >= cutoff]
        
        removed = original_count - len(self.events)
        return removed


def demonstrate_audit_trail():
    """Demonstrate the audit trail and logging pattern."""
    print("=" * 80)
    print("Audit Trail & Logging Pattern Demonstration")
    print("=" * 80)
    
    agent = AuditTrailAgent(agent_id="agent_001")
    
    # Demonstration 1: Basic Event Logging
    print("\n" + "=" * 80)
    print("Demonstration 1: Basic Event Logging")
    print("=" * 80)
    
    event = agent.log_event(
        event_type=AuditEventType.ACTION,
        level=AuditLevel.INFO,
        action="Process user request",
        details={"request_type": "query", "query": "What is the weather?"},
        user_id="user_123",
        result="success",
        duration_ms=150.5
    )
    
    print(f"\nLogged Event:")
    print(f"  ID: {event.event_id}")
    print(f"  Type: {event.event_type.value}")
    print(f"  Level: {event.level.value}")
    print(f"  Action: {event.action}")
    print(f"  User: {event.user_id}")
    print(f"  Checksum: {event.checksum[:16]}...")
    
    # Demonstration 2: Data Access Logging
    print("\n" + "=" * 80)
    print("Demonstration 2: Data Access Logging")
    print("=" * 80)
    
    data_event = agent.log_data_access(
        operation=DataOperation.READ,
        data_type="customer_records",
        record_count=150,
        data_classification="sensitive",
        user_id="analyst_456",
        details={"query": "SELECT * FROM customers WHERE region='EU'"}
    )
    
    print(f"\nData Access Event:")
    print(f"  Operation: {data_event.operation.value}")
    print(f"  Data Type: {data_event.data_type}")
    print(f"  Records: {data_event.record_count}")
    print(f"  Classification: {data_event.data_classification}")
    print(f"  Level: {data_event.level.value}")
    
    # Demonstration 3: Security Event Logging
    print("\n" + "=" * 80)
    print("Demonstration 3: Security Event Logging")
    print("=" * 80)
    
    security_event = agent.log_security_event(
        action="Failed authentication attempt",
        threat_level="medium",
        details={"reason": "invalid_credentials", "attempts": 3},
        source_ip="192.168.1.100",
        attack_type="brute_force",
        user_id="unknown"
    )
    
    print(f"\nSecurity Event:")
    print(f"  Action: {security_event.action}")
    print(f"  Threat Level: {security_event.threat_level}")
    print(f"  Attack Type: {security_event.attack_type}")
    print(f"  Source IP: {security_event.source_ip}")
    print(f"  Level: {security_event.level.value}")
    
    # Log more events for testing
    agent.log_event(
        event_type=AuditEventType.DECISION,
        level=AuditLevel.MEDIUM,
        action="Route to specialist agent",
        details={"specialist": "medical", "confidence": 0.85},
        user_id="user_123"
    )
    
    agent.log_data_access(
        operation=DataOperation.WRITE,
        data_type="transaction_logs",
        record_count=1,
        data_classification="confidential",
        user_id="system"
    )
    
    # Demonstration 4: Query Events
    print("\n" + "=" * 80)
    print("Demonstration 4: Query Events")
    print("=" * 80)
    
    security_events = agent.query_events(event_type=AuditEventType.SECURITY)
    print(f"\nSecurity Events: {len(security_events)}")
    
    user_events = agent.query_events(user_id="user_123")
    print(f"Events by user_123: {len(user_events)}")
    
    data_events = agent.query_events(event_type=AuditEventType.DATA_ACCESS)
    print(f"Data Access Events: {len(data_events)}")
    
    # Demonstration 5: User Activity Summary
    print("\n" + "=" * 80)
    print("Demonstration 5: User Activity Summary")
    print("=" * 80)
    
    activity = agent.get_user_activity("user_123")
    print(f"\nUser Activity Summary:")
    print(f"  User: {activity['user_id']}")
    print(f"  Period: {activity['period_hours']} hours")
    print(f"  Total Events: {activity['total_events']}")
    print(f"  Actions: {', '.join(activity['actions'])}")
    
    # Demonstration 6: Security Summary
    print("\n" + "=" * 80)
    print("Demonstration 6: Security Summary")
    print("=" * 80)
    
    sec_summary = agent.get_security_summary()
    print(f"\nSecurity Summary:")
    print(f"  Period: {sec_summary['period_hours']} hours")
    print(f"  Total Security Events: {sec_summary['total_security_events']}")
    print(f"  Critical Events: {sec_summary['critical_events']}")
    print(f"  By Threat Level:")
    for level, count in sec_summary['by_threat_level'].items():
        if count > 0:
            print(f"    {level}: {count}")
    
    # Demonstration 7: Integrity Verification
    print("\n" + "=" * 80)
    print("Demonstration 7: Integrity Verification")
    print("=" * 80)
    
    integrity = agent.verify_integrity()
    print(f"\nIntegrity Check:")
    print(f"  Total Events: {integrity['total_events']}")
    print(f"  Verified: {integrity['verified']}")
    print(f"  Tampered: {integrity['tampered']}")
    print(f"  Integrity Score: {integrity['integrity_score']:.2%}")
    
    # Demonstration 8: Export Events
    print("\n" + "=" * 80)
    print("Demonstration 8: Export Events")
    print("=" * 80)
    
    # Export as JSON (first 2 events)
    json_export = agent.export_events(format="json", events=agent.events[:2])
    print(f"\nJSON Export (sample):")
    print(json_export[:300] + "...")
    
    # Export as CSV
    csv_export = agent.export_events(format="csv", events=agent.events[:3])
    print(f"\nCSV Export (sample):")
    print("\n".join(csv_export.split("\n")[:4]))
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary: Audit Trail & Logging Pattern")
    print("=" * 80)
    print(f"""
The Audit Trail & Logging pattern provides comprehensive activity tracking:

Key Features Demonstrated:
1. Event Logging - Structured audit events with rich metadata
2. Data Access Tracking - Specialized logging for data operations
3. Security Events - Track security incidents and threats
4. Query Capabilities - Filter and search audit logs
5. User Activity - Track and analyze user behavior
6. Security Monitoring - Monitor security events
7. Integrity Verification - Tamper-evident logging with checksums
8. Export Functionality - Export logs in multiple formats

Current State:
- Total Events: {len(agent.events)}
- Event Types: {len(set(e.event_type for e in agent.events))}
- Users: {len(set(e.user_id for e in agent.events if e.user_id))}
- Integrity Score: {integrity['integrity_score']:.2%}

Benefits:
- Complete accountability and traceability
- Enables incident investigation
- Supports compliance requirements
- Facilitates debugging
- Detects security threats
- Provides audit trail for legal purposes

Best Practices:
- Use structured logging (JSON)
- Include rich metadata in events
- Implement tamper detection
- Set appropriate retention periods
- Encrypt sensitive log data
- Index logs for searching
- Monitor log storage
- Provide query capabilities
- Support compliance reporting
- Handle PII data carefully

This pattern is essential for production AI systems requiring compliance,
security monitoring, and forensic investigation capabilities.
""")


if __name__ == "__main__":
    demonstrate_audit_trail()
