"""
Monitoring & Observability Pattern
Comprehensive tracking of agent behavior
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time
import json
class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
@dataclass
class LogEntry:
    timestamp: datetime
    level: LogLevel
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
@dataclass
class Metric:
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
@dataclass
class Span:
    """Distributed tracing span"""
    span_id: str
    trace_id: str
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[LogEntry] = field(default_factory=list)
    parent_span_id: Optional[str] = None
class MetricsCollector:
    """Collects and aggregates metrics"""
    def __init__(self):
        self.metrics: List[Metric] = []
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        self.counters[name] = self.counters.get(name, 0) + value
        metric = Metric(
            name=name,
            type=MetricType.COUNTER,
            value=self.counters[name],
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.metrics.append(metric)
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric"""
        self.gauges[name] = value
        metric = Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.metrics.append(metric)
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value"""
        metric = Metric(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.metrics.append(metric)
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            "total_metrics": len(self.metrics),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "recent_metrics": self.metrics[-10:]
        }
class Logger:
    """Structured logging system"""
    def __init__(self, name: str):
        self.name = name
        self.logs: List[LogEntry] = []
        self.current_trace_id: Optional[str] = None
    def set_trace_id(self, trace_id: str):
        """Set current trace ID for correlation"""
        self.current_trace_id = trace_id
    def log(self, level: LogLevel, message: str, context: Dict[str, Any] = None):
        """Log a message"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            context=context or {},
            trace_id=self.current_trace_id
        )
        self.logs.append(entry)
        # Print to console
        timestamp_str = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        trace_str = f"[{entry.trace_id}]" if entry.trace_id else ""
        context_str = f" {json.dumps(entry.context)}" if entry.context else ""
        print(f"{timestamp_str} [{self.name}] {trace_str} {level.value.upper()}: {message}{context_str}")
    def debug(self, message: str, context: Dict[str, Any] = None):
        self.log(LogLevel.DEBUG, message, context)
    def info(self, message: str, context: Dict[str, Any] = None):
        self.log(LogLevel.INFO, message, context)
    def warning(self, message: str, context: Dict[str, Any] = None):
        self.log(LogLevel.WARNING, message, context)
    def error(self, message: str, context: Dict[str, Any] = None):
        self.log(LogLevel.ERROR, message, context)
    def critical(self, message: str, context: Dict[str, Any] = None):
        self.log(LogLevel.CRITICAL, message, context)
    def get_logs(self, level: Optional[LogLevel] = None, limit: int = None) -> List[LogEntry]:
        """Get filtered logs"""
        logs = self.logs
        if level:
            logs = [log for log in logs if log.level == level]
        if limit:
            logs = logs[-limit:]
        return logs
class Tracer:
    """Distributed tracing system"""
    def __init__(self):
        self.spans: List[Span] = []
        self.active_spans: Dict[str, Span] = {}
    def start_span(self, operation_name: str, trace_id: str = None, 
                   parent_span_id: str = None, tags: Dict[str, Any] = None) -> Span:
        """Start a new span"""
        import uuid
        span_id = str(uuid.uuid4())
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            tags=tags or {},
            parent_span_id=parent_span_id
        )
        self.active_spans[span_id] = span
        return span
    def finish_span(self, span: Span, tags: Dict[str, Any] = None):
        """Finish a span"""
        span.end_time = datetime.now()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        if tags:
            span.tags.update(tags)
        self.spans.append(span)
        if span.span_id in self.active_spans:
            del self.active_spans[span.span_id]
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace"""
        return [span for span in self.spans if span.trace_id == trace_id]
    def print_trace(self, trace_id: str):
        """Print trace tree"""
        spans = self.get_trace(trace_id)
        print(f"\n{'='*70}")
        print(f"TRACE: {trace_id}")
        print(f"{'='*70}")
        # Build tree
        root_spans = [s for s in spans if s.parent_span_id is None]
        def print_span(span: Span, indent: int = 0):
            prefix = "  " * indent
            duration = f"{span.duration_ms:.2f}ms" if span.duration_ms else "active"
            print(f"{prefix}├─ {span.operation_name} ({duration})")
            if span.tags:
                for key, value in span.tags.items():
                    print(f"{prefix}   {key}: {value}")
            # Print children
            children = [s for s in spans if s.parent_span_id == span.span_id]
            for child in children:
                print_span(child, indent + 1)
        for root in root_spans:
            print_span(root)
class ObservableAgent:
    """Agent with comprehensive observability"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = Logger(agent_id)
        self.metrics = MetricsCollector()
        self.tracer = Tracer()
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
    def process_request(self, request: str) -> Dict[str, Any]:
        """Process a request with full observability"""
        # Start trace
        span = self.tracer.start_span(
            operation_name="process_request",
            tags={"agent_id": self.agent_id, "request": request}
        )
        self.logger.set_trace_id(span.trace_id)
        self.logger.info("Processing request", {"request": request})
        # Increment request counter
        self.metrics.increment_counter("requests_total", tags={"agent": self.agent_id})
        start_time = time.time()
        try:
            # Simulate request processing
            result = self._execute_request(request, span)
            # Record success
            self.metrics.increment_counter("requests_success", tags={"agent": self.agent_id})
            self.logger.info("Request completed successfully", {"result": result})
            success = True
        except Exception as e:
            # Record error
            self.error_count += 1
            self.metrics.increment_counter("requests_error", tags={"agent": self.agent_id})
            self.logger.error(f"Request failed: {str(e)}", {"error": str(e)})
            result = {"error": str(e)}
            success = False
            span.tags["error"] = True
            span.tags["error_message"] = str(e)
        finally:
            # Record latency
            latency = (time.time() - start_time) * 1000  # ms
            self.metrics.record_histogram("request_latency_ms", latency, 
                                         tags={"agent": self.agent_id})
            # Update gauges
            self.request_count += 1
            self.total_latency += latency
            avg_latency = self.total_latency / self.request_count
            self.metrics.set_gauge("requests_in_flight", 0, tags={"agent": self.agent_id})
            self.metrics.set_gauge("avg_latency_ms", avg_latency, tags={"agent": self.agent_id})
            self.metrics.set_gauge("error_rate", self.error_count / self.request_count,
                                 tags={"agent": self.agent_id})
            # Finish span
            self.tracer.finish_span(span, tags={"success": success, "latency_ms": latency})
        return result
    def _execute_request(self, request: str, parent_span: Span) -> Dict[str, Any]:
        """Execute request with nested spans"""
        # Step 1: Parse request
        parse_span = self.tracer.start_span(
            operation_name="parse_request",
            trace_id=parent_span.trace_id,
            parent_span_id=parent_span.span_id
        )
        self.logger.debug("Parsing request")
        time.sleep(0.1)  # Simulate work
        parsed = {"task": request, "priority": "normal"}
        self.tracer.finish_span(parse_span, tags={"parsed_items": len(parsed)})
        # Step 2: Validate
        validate_span = self.tracer.start_span(
            operation_name="validate_request",
            trace_id=parent_span.trace_id,
            parent_span_id=parent_span.span_id
        )
        self.logger.debug("Validating request")
        time.sleep(0.05)
        self.tracer.finish_span(validate_span, tags={"valid": True})
        # Step 3: Execute
        execute_span = self.tracer.start_span(
            operation_name="execute_task",
            trace_id=parent_span.trace_id,
            parent_span_id=parent_span.span_id
        )
        self.logger.info("Executing task")
        time.sleep(0.15)
        result = {"status": "completed", "output": f"Processed: {request}"}
        self.tracer.finish_span(execute_span, tags={"output_size": len(str(result))})
        return result
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        health = "healthy"
        if error_rate > 0.1:
            health = "degraded"
        if error_rate > 0.5:
            health = "unhealthy"
        return {
            "status": health,
            "requests_total": self.request_count,
            "errors_total": self.error_count,
            "error_rate": error_rate,
            "avg_latency_ms": avg_latency
        }
    def print_observability_report(self):
        """Print comprehensive observability report"""
        print(f"\n{'='*70}")
        print(f"OBSERVABILITY REPORT: {self.agent_id}")
        print(f"{'='*70}")
        # Health status
        health = self.get_health_status()
        print(f"\nHealth Status: {health['status'].upper()}")
        print(f"  Requests: {health['requests_total']}")
        print(f"  Errors: {health['errors_total']}")
        print(f"  Error Rate: {health['error_rate']:.1%}")
        print(f"  Avg Latency: {health['avg_latency_ms']:.2f}ms")
        # Metrics summary
        metrics_summary = self.metrics.get_summary()
        print(f"\nMetrics Summary:")
        print(f"  Total Metrics Collected: {metrics_summary['total_metrics']}")
        if metrics_summary['counters']:
            print(f"\n  Counters:")
            for name, value in metrics_summary['counters'].items():
                print(f"    {name}: {value}")
        if metrics_summary['gauges']:
            print(f"\n  Gauges:")
            for name, value in metrics_summary['gauges'].items():
                print(f"    {name}: {value:.2f}")
        # Log summary
        error_logs = self.logger.get_logs(LogLevel.ERROR)
        warning_logs = self.logger.get_logs(LogLevel.WARNING)
        print(f"\nLog Summary:")
        print(f"  Total Logs: {len(self.logger.logs)}")
        print(f"  Errors: {len(error_logs)}")
        print(f"  Warnings: {len(warning_logs)}")
        if error_logs:
            print(f"\n  Recent Errors:")
            for log in error_logs[-3:]:
                print(f"    {log.timestamp}: {log.message}")
        # Trace summary
        print(f"\nTrace Summary:")
        print(f"  Total Spans: {len(self.tracer.spans)}")
        print(f"  Active Spans: {len(self.tracer.active_spans)}")
# Usage
if __name__ == "__main__":
    print("="*80)
    print("MONITORING & OBSERVABILITY DEMONSTRATION")
    print("="*80)
    # Create observable agent
    agent = ObservableAgent("agent-001")
    # Process multiple requests
    requests = [
        "Analyze customer data",
        "Generate report",
        "Send notifications",
        "Update database"
    ]
    print("\nProcessing requests...\n")
    for request in requests:
        result = agent.process_request(request)
        time.sleep(0.2)
    # Get a trace
    if agent.tracer.spans:
        first_trace_id = agent.tracer.spans[0].trace_id
        agent.tracer.print_trace(first_trace_id)
    # Print comprehensive report
    agent.print_observability_report()
