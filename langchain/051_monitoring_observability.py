"""
Pattern 051: Monitoring & Observability

Description:
    The Monitoring & Observability pattern provides comprehensive tracking of agent
    behavior, performance, costs, and errors to enable debugging, optimization, and
    operational visibility. This pattern implements metrics collection, logging,
    tracing, alerting, and dashboard-ready data aggregation.

Components:
    1. Metrics Collector: Tracks quantitative measurements
    2. Logger: Records events and contextual information
    3. Tracer: Follows execution paths through system
    4. Alert Manager: Triggers notifications on thresholds
    5. Dashboard Aggregator: Prepares data for visualization

Use Cases:
    - Production system monitoring
    - Performance optimization
    - Cost tracking and budgeting
    - Debugging and troubleshooting
    - Compliance and audit trails
    - SLA monitoring

LangChain Implementation:
    Uses custom callbacks to intercept and track all LLM interactions,
    with structured logging, metrics aggregation, and trace correlation.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class MetricType(Enum):
    """Types of metrics to track"""
    COUNTER = "counter"  # Incremental count
    GAUGE = "gauge"  # Current value
    HISTOGRAM = "histogram"  # Distribution
    TIMER = "timer"  # Duration measurement


class LogLevel(Enum):
    """Log severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    message: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "message": self.message,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "metadata": self.metadata
        }


@dataclass
class Metric:
    """Metric measurement"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags
        }


@dataclass
class Span:
    """Trace span for a single operation"""
    span_id: str
    trace_id: str
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "running"
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_span_id: Optional[str] = None
    
    def finish(self):
        """Mark span as finished"""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = "completed"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "operation": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "metadata": self.metadata,
            "parent_span_id": self.parent_span_id
        }


@dataclass
class Alert:
    """Alert notification"""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    threshold_value: float
    actual_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "metric": self.metric_name,
            "threshold": threshold_value,
            "actual": self.actual_value,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged
        }


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot"""
    timestamp: datetime
    requests_per_second: float
    avg_latency_ms: float
    error_rate: float
    total_cost: float
    active_requests: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "requests_per_second": f"{self.requests_per_second:.2f}",
            "avg_latency_ms": f"{self.avg_latency_ms:.1f}",
            "error_rate": f"{self.error_rate:.2%}",
            "total_cost": f"${self.total_cost:.4f}",
            "active_requests": self.active_requests
        }


class ObservabilityMonitor:
    """
    Comprehensive monitoring and observability system.
    
    This implementation provides:
    1. Structured logging with levels and context
    2. Metrics collection (counters, gauges, timers, histograms)
    3. Distributed tracing with span correlation
    4. Alert management with threshold-based triggers
    5. Performance aggregation for dashboards
    """
    
    def __init__(
        self,
        service_name: str = "agent-service",
        enable_logging: bool = True,
        enable_metrics: bool = True,
        enable_tracing: bool = True,
        enable_alerting: bool = True
    ):
        self.service_name = service_name
        self.enable_logging = enable_logging
        self.enable_metrics = enable_metrics
        self.enable_tracing = enable_tracing
        self.enable_alerting = enable_alerting
        
        # Storage
        self.logs: deque = deque(maxlen=1000)
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.spans: Dict[str, Span] = {}
        self.traces: Dict[str, List[Span]] = defaultdict(list)
        self.alerts: List[Alert] = []
        
        # Performance tracking
        self.request_times: deque = deque(maxlen=100)
        self.request_timestamps: deque = deque(maxlen=100)
        self.error_count = 0
        self.total_requests = 0
        self.total_cost = 0.0
        self.active_requests = 0
        
        # Alert thresholds
        self.alert_thresholds = {
            "error_rate": {"threshold": 0.1, "severity": AlertSeverity.HIGH},
            "avg_latency_ms": {"threshold": 5000, "severity": AlertSeverity.MEDIUM},
            "cost_per_hour": {"threshold": 10.0, "severity": AlertSeverity.HIGH},
            "requests_per_second": {"threshold": 100, "severity": AlertSeverity.LOW}
        }
    
    # ===== LOGGING =====
    
    def log(
        self,
        level: LogLevel,
        message: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **metadata
    ):
        """Log a message with structured context"""
        if not self.enable_logging:
            return
        
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            trace_id=trace_id,
            span_id=span_id,
            metadata=metadata
        )
        
        self.logs.append(entry)
        
        # Print to console (in production, would send to logging service)
        level_symbol = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌",
            LogLevel.CRITICAL: "🚨"
        }.get(level, "📝")
        
        if level.value in ["error", "critical", "warning"]:
            print(f"{level_symbol} [{level.value.upper()}] {message}")
    
    def debug(self, message: str, **metadata):
        """Log debug message"""
        self.log(LogLevel.DEBUG, message, **metadata)
    
    def info(self, message: str, **metadata):
        """Log info message"""
        self.log(LogLevel.INFO, message, **metadata)
    
    def warning(self, message: str, **metadata):
        """Log warning message"""
        self.log(LogLevel.WARNING, message, **metadata)
    
    def error(self, message: str, **metadata):
        """Log error message"""
        self.log(LogLevel.ERROR, message, **metadata)
        self.error_count += 1
    
    def critical(self, message: str, **metadata):
        """Log critical message"""
        self.log(LogLevel.CRITICAL, message, **metadata)
    
    # ===== METRICS =====
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        **tags
    ):
        """Record a metric measurement"""
        if not self.enable_metrics:
            return
        
        metric = Metric(
            name=name,
            type=metric_type,
            value=value,
            timestamp=datetime.now(),
            tags=tags
        )
        
        self.metrics[name].append(metric)
        
        # Check alert thresholds
        self._check_alert_threshold(name, value)
    
    def increment_counter(self, name: str, amount: float = 1.0, **tags):
        """Increment a counter metric"""
        self.record_metric(name, amount, MetricType.COUNTER, **tags)
    
    def set_gauge(self, name: str, value: float, **tags):
        """Set a gauge metric"""
        self.record_metric(name, value, MetricType.GAUGE, **tags)
    
    def record_timer(self, name: str, duration_ms: float, **tags):
        """Record a timer metric"""
        self.record_metric(name, duration_ms, MetricType.TIMER, **tags)
    
    # ===== TRACING =====
    
    def start_span(
        self,
        operation_name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        **metadata
    ) -> Span:
        """Start a new trace span"""
        if not self.enable_tracing:
            return None
        
        import uuid
        span_id = str(uuid.uuid4())[:8]
        trace_id = trace_id or str(uuid.uuid4())[:8]
        
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            metadata=metadata,
            parent_span_id=parent_span_id
        )
        
        self.spans[span_id] = span
        self.traces[trace_id].append(span)
        
        self.debug(
            f"Started span: {operation_name}",
            trace_id=trace_id,
            span_id=span_id
        )
        
        return span
    
    def finish_span(self, span: Span, **metadata):
        """Finish a trace span"""
        if not span:
            return
        
        span.finish()
        span.metadata.update(metadata)
        
        self.debug(
            f"Finished span: {span.operation_name} ({span.duration_ms:.1f}ms)",
            trace_id=span.trace_id,
            span_id=span.span_id
        )
        
        # Record latency metric
        self.record_timer(
            f"span.{span.operation_name}.duration",
            span.duration_ms,
            trace_id=span.trace_id
        )
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace"""
        return self.traces.get(trace_id, [])
    
    # ===== ALERTING =====
    
    def _check_alert_threshold(self, metric_name: str, value: float):
        """Check if metric exceeds alert threshold"""
        if not self.enable_alerting:
            return
        
        config = self.alert_thresholds.get(metric_name)
        if not config:
            return
        
        threshold = config["threshold"]
        severity = config["severity"]
        
        if value > threshold:
            self.trigger_alert(
                severity=severity,
                title=f"{metric_name} exceeded threshold",
                description=f"{metric_name} = {value:.2f}, threshold = {threshold:.2f}",
                metric_name=metric_name,
                threshold_value=threshold,
                actual_value=value
            )
    
    def trigger_alert(
        self,
        severity: AlertSeverity,
        title: str,
        description: str,
        metric_name: str,
        threshold_value: float,
        actual_value: float
    ):
        """Trigger an alert"""
        import uuid
        
        alert = Alert(
            alert_id=str(uuid.uuid4())[:8],
            severity=severity,
            title=title,
            description=description,
            metric_name=metric_name,
            threshold_value=threshold_value,
            actual_value=actual_value
        )
        
        self.alerts.append(alert)
        
        severity_symbol = {
            AlertSeverity.LOW: "🔵",
            AlertSeverity.MEDIUM: "🟡",
            AlertSeverity.HIGH: "🟠",
            AlertSeverity.CRITICAL: "🔴"
        }.get(severity, "⚠️")
        
        print(f"\n{severity_symbol} ALERT [{severity.value.upper()}]: {title}")
        print(f"   {description}")
    
    # ===== PERFORMANCE TRACKING =====
    
    def track_request(self, duration_ms: float, cost: float, error: bool = False):
        """Track request performance"""
        self.total_requests += 1
        self.request_times.append(duration_ms)
        self.request_timestamps.append(time.time())
        self.total_cost += cost
        
        if error:
            self.error_count += 1
        
        # Record metrics
        self.record_timer("request.duration", duration_ms)
        self.record_metric("request.cost", cost, MetricType.COUNTER)
        
        if error:
            self.increment_counter("request.errors")
    
    def get_performance_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot"""
        now = time.time()
        
        # Calculate requests per second (last 10 seconds)
        recent_requests = [
            ts for ts in self.request_timestamps
            if now - ts < 10
        ]
        rps = len(recent_requests) / 10 if recent_requests else 0.0
        
        # Calculate average latency
        avg_latency = (
            sum(self.request_times) / len(self.request_times)
            if self.request_times else 0.0
        )
        
        # Calculate error rate
        error_rate = (
            self.error_count / max(1, self.total_requests)
        )
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            requests_per_second=rps,
            avg_latency_ms=avg_latency,
            error_rate=error_rate,
            total_cost=self.total_cost,
            active_requests=self.active_requests
        )
        
        # Check thresholds
        self._check_alert_threshold("error_rate", error_rate)
        self._check_alert_threshold("avg_latency_ms", avg_latency)
        
        return snapshot
    
    # ===== DASHBOARD DATA =====
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get aggregated data for dashboard"""
        snapshot = self.get_performance_snapshot()
        
        # Get recent logs by level
        log_counts = defaultdict(int)
        for log in self.logs:
            log_counts[log.level.value] += 1
        
        # Get metric summaries
        metric_summaries = {}
        for name, metric_list in self.metrics.items():
            if metric_list:
                values = [m.value for m in metric_list[-10:]]  # Last 10
                metric_summaries[name] = {
                    "count": len(metric_list),
                    "latest": values[-1] if values else 0,
                    "avg": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0
                }
        
        # Get active alerts
        active_alerts = [a for a in self.alerts if not a.acknowledged]
        
        return {
            "service": self.service_name,
            "timestamp": datetime.now().isoformat(),
            "performance": snapshot.to_dict(),
            "logs": {
                "total": len(self.logs),
                "by_level": dict(log_counts)
            },
            "metrics": metric_summaries,
            "alerts": {
                "total": len(self.alerts),
                "active": len(active_alerts),
                "by_severity": self._count_by_severity(active_alerts)
            },
            "traces": {
                "total_traces": len(self.traces),
                "total_spans": len(self.spans)
            }
        }
    
    def _count_by_severity(self, alerts: List[Alert]) -> Dict[str, int]:
        """Count alerts by severity"""
        counts = defaultdict(int)
        for alert in alerts:
            counts[alert.severity.value] += 1
        return dict(counts)


class MonitoredAgent:
    """Agent with comprehensive monitoring"""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        monitor: Optional[ObservabilityMonitor] = None
    ):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        self.monitor = monitor or ObservabilityMonitor()
        self.model_name = model_name
    
    def process_query(self, query: str, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """Process query with full monitoring"""
        import uuid
        trace_id = trace_id or str(uuid.uuid4())[:8]
        
        # Start overall span
        overall_span = self.monitor.start_span(
            "process_query",
            trace_id=trace_id,
            query=query[:100]
        )
        
        self.monitor.info(
            f"Processing query",
            trace_id=trace_id,
            query_length=len(query)
        )
        
        self.monitor.active_requests += 1
        start_time = time.time()
        
        try:
            # Execute LLM call
            llm_span = self.monitor.start_span(
                "llm_call",
                trace_id=trace_id,
                parent_span_id=overall_span.span_id if overall_span else None,
                model=self.model_name
            )
            
            response = self.llm.invoke(query)
            
            self.monitor.finish_span(
                llm_span,
                tokens=len(response.content.split())
            )
            
            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            cost = 0.0001  # Simplified cost
            
            self.monitor.track_request(duration_ms, cost, error=False)
            
            self.monitor.finish_span(
                overall_span,
                success=True,
                response_length=len(response.content)
            )
            
            self.monitor.info(
                f"Query processed successfully",
                trace_id=trace_id,
                duration_ms=f"{duration_ms:.1f}"
            )
            
            return {
                "success": True,
                "response": response.content,
                "trace_id": trace_id,
                "duration_ms": duration_ms,
                "cost": cost
            }
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            self.monitor.track_request(duration_ms, 0.0, error=True)
            
            self.monitor.error(
                f"Query processing failed: {str(e)}",
                trace_id=trace_id,
                error_type=type(e).__name__
            )
            
            if overall_span:
                self.monitor.finish_span(
                    overall_span,
                    success=False,
                    error=str(e)
                )
            
            return {
                "success": False,
                "error": str(e),
                "trace_id": trace_id,
                "duration_ms": duration_ms
            }
        
        finally:
            self.monitor.active_requests -= 1


def demonstrate_monitoring():
    """Demonstrate monitoring and observability pattern"""
    
    print("=" * 80)
    print("PATTERN 051: MONITORING & OBSERVABILITY DEMONSTRATION")
    print("=" * 80)
    print("\nDemonstrating comprehensive system monitoring and observability\n")
    
    # Create monitor and agent
    monitor = ObservabilityMonitor(service_name="demo-agent")
    agent = MonitoredAgent(monitor=monitor)
    
    # Test 1: Basic monitoring
    print("\n" + "=" * 80)
    print("TEST 1: Basic Request Monitoring")
    print("=" * 80)
    
    monitor.info("Starting monitoring demonstration")
    
    result1 = agent.process_query("What is machine learning?")
    
    print(f"\n✓ Request completed:")
    print(f"  Success: {result1['success']}")
    print(f"  Trace ID: {result1['trace_id']}")
    print(f"  Duration: {result1['duration_ms']:.1f}ms")
    print(f"  Cost: ${result1.get('cost', 0):.6f}")
    
    # Test 2: Multiple requests with metrics
    print("\n" + "=" * 80)
    print("TEST 2: Multiple Requests - Metrics Collection")
    print("=" * 80)
    
    queries = [
        "Explain neural networks briefly",
        "What is deep learning?",
        "Define artificial intelligence"
    ]
    
    print(f"\nProcessing {len(queries)} queries...")
    
    for i, query in enumerate(queries, 1):
        result = agent.process_query(query)
        print(f"  Request {i}: {result['duration_ms']:.1f}ms")
        time.sleep(0.1)
    
    # Show performance snapshot
    snapshot = monitor.get_performance_snapshot()
    print(f"\n📊 Performance Snapshot:")
    print(f"  Requests/sec: {snapshot.requests_per_second:.2f}")
    print(f"  Avg Latency: {snapshot.avg_latency_ms:.1f}ms")
    print(f"  Error Rate: {snapshot.error_rate:.2%}")
    print(f"  Total Cost: ${snapshot.total_cost:.6f}")
    
    # Test 3: Error tracking
    print("\n" + "=" * 80)
    print("TEST 3: Error Tracking and Logging")
    print("=" * 80)
    
    # Simulate error by using invalid input
    monitor.warning("Testing error handling")
    
    try:
        # This will succeed but we'll log it as problematic
        result3 = agent.process_query("")
        monitor.warning("Empty query processed", trace_id=result3.get('trace_id'))
    except Exception as e:
        monitor.error(f"Error occurred: {str(e)}")
    
    # Show log summary
    print(f"\n📝 Log Summary:")
    log_counts = defaultdict(int)
    for log in monitor.logs:
        log_counts[log.level.value] += 1
    
    for level, count in sorted(log_counts.items()):
        print(f"  {level.upper()}: {count}")
    
    # Test 4: Distributed tracing
    print("\n" + "=" * 80)
    print("TEST 4: Distributed Tracing")
    print("=" * 80)
    
    import uuid
    trace_id = str(uuid.uuid4())[:8]
    
    print(f"\nTrace ID: {trace_id}")
    print("Processing multi-step operation...")
    
    # Simulate multi-step process
    main_span = monitor.start_span("multi_step_process", trace_id=trace_id)
    
    step1_span = monitor.start_span(
        "step1_validate",
        trace_id=trace_id,
        parent_span_id=main_span.span_id
    )
    time.sleep(0.05)
    monitor.finish_span(step1_span, validation="passed")
    
    step2_span = monitor.start_span(
        "step2_process",
        trace_id=trace_id,
        parent_span_id=main_span.span_id
    )
    result4 = agent.process_query("Quick test", trace_id=trace_id)
    monitor.finish_span(step2_span)
    
    monitor.finish_span(main_span)
    
    # Show trace
    trace_spans = monitor.get_trace(trace_id)
    print(f"\n🔍 Trace contains {len(trace_spans)} spans:")
    
    for span in trace_spans:
        indent = "  " if span.parent_span_id else ""
        print(f"{indent}• {span.operation_name}: {span.duration_ms:.1f}ms [{span.status}]")
    
    # Test 5: Alert triggering
    print("\n" + "=" * 80)
    print("TEST 5: Alert Management")
    print("=" * 80)
    
    print("\nSimulating high error rate...")
    
    # Simulate high error rate
    for i in range(5):
        monitor.track_request(100.0, 0.001, error=True)
    
    # Trigger custom alert
    monitor.trigger_alert(
        severity=AlertSeverity.MEDIUM,
        title="Unusual pattern detected",
        description="Query complexity increasing",
        metric_name="query_complexity",
        threshold_value=0.8,
        actual_value=0.95
    )
    
    print(f"\n⚠️  Active Alerts: {len([a for a in monitor.alerts if not a.acknowledged])}")
    
    for alert in monitor.alerts[-2:]:  # Show last 2
        print(f"\n  Alert: {alert.title}")
        print(f"    Severity: {alert.severity.value}")
        print(f"    {alert.description}")
    
    # Test 6: Dashboard data
    print("\n" + "=" * 80)
    print("TEST 6: Dashboard Data Aggregation")
    print("=" * 80)
    
    dashboard = monitor.get_dashboard_data()
    
    print(f"\n📊 Dashboard Summary:")
    print(f"\nService: {dashboard['service']}")
    print(f"\nPerformance:")
    for key, value in dashboard['performance'].items():
        if key != 'timestamp':
            print(f"  {key}: {value}")
    
    print(f"\nLogs:")
    print(f"  Total: {dashboard['logs']['total']}")
    print(f"  By Level: {dashboard['logs']['by_level']}")
    
    print(f"\nMetrics:")
    print(f"  Tracked metrics: {len(dashboard['metrics'])}")
    
    print(f"\nAlerts:")
    print(f"  Total: {dashboard['alerts']['total']}")
    print(f"  Active: {dashboard['alerts']['active']}")
    print(f"  By Severity: {dashboard['alerts']['by_severity']}")
    
    print(f"\nTraces:")
    print(f"  Total traces: {dashboard['traces']['total_traces']}")
    print(f"  Total spans: {dashboard['traces']['total_spans']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("MONITORING & OBSERVABILITY PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Visibility: Complete insight into system behavior
2. Debugging: Trace requests through distributed systems
3. Performance: Identify bottlenecks and optimization opportunities
4. Reliability: Early detection of issues via alerts
5. Compliance: Audit trails and logs for regulations

Implementation Features:
1. Structured logging with levels and context
2. Metrics collection (counters, gauges, timers, histograms)
3. Distributed tracing with span correlation
4. Alert management with severity-based triggers
5. Dashboard-ready data aggregation
6. Performance snapshots and trends

Monitoring Components:
1. Logs: Contextual event records with severity levels
2. Metrics: Quantitative measurements over time
3. Traces: Request flow through system components
4. Alerts: Notifications when thresholds exceeded
5. Dashboards: Visualized aggregated data

Metric Types:
- Counter: Cumulative count (requests, errors)
- Gauge: Point-in-time value (active connections, memory)
- Timer: Duration measurements (latency, processing time)
- Histogram: Distribution of values

Trace Components:
- Trace ID: Unique identifier for request journey
- Span: Single operation within trace
- Parent-Child: Relationships between spans
- Metadata: Additional context per span

Use Cases:
- Production system monitoring
- Performance optimization
- Cost tracking and budgeting
- Debugging distributed systems
- SLA monitoring and compliance
- Capacity planning
- Incident response

Best Practices:
1. Log at appropriate levels (don't over-log)
2. Include trace/span IDs for correlation
3. Set meaningful alert thresholds
4. Aggregate metrics for dashboard efficiency
5. Sample high-volume traces
6. Retain logs based on importance
7. Monitor the monitoring system itself
8. Document metric meanings and units

Production Considerations:
- Centralized logging (ELK, Splunk, CloudWatch)
- Time-series databases for metrics (Prometheus, InfluxDB)
- Distributed tracing systems (Jaeger, Zipkin)
- Alert routing (PagerDuty, Opsgenie)
- Dashboard platforms (Grafana, Datadog)
- Log retention policies
- Sampling strategies for high volume
- Cost optimization for storage

Comparison with Related Patterns:
- vs. Circuit Breaker: Observes vs prevents failures
- vs. Rate Limiting: Tracks usage vs enforces limits
- vs. Audit Trail: Operational data vs compliance records
- vs. Profiling: Real-time monitoring vs performance analysis

The Monitoring & Observability pattern is essential for production AI systems,
providing the visibility needed to operate, debug, optimize, and ensure
reliability of complex distributed agent systems.
""")


if __name__ == "__main__":
    demonstrate_monitoring()
