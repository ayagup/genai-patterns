"""
Agent Performance Monitor Pattern

Monitors and analyzes agent performance metrics in real-time.
Provides insights, alerts, and optimization recommendations.

Use Cases:
- Performance tracking
- Bottleneck detection
- Resource optimization
- Quality assurance

Advantages:
- Real-time monitoring
- Performance insights
- Proactive alerts
- Optimization guidance
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import statistics
import json


class MetricType(Enum):
    """Types of performance metrics"""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    RESOURCE_USAGE = "resource_usage"
    QUEUE_LENGTH = "queue_length"
    LATENCY = "latency"
    ACCURACY = "accuracy"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ThresholdType(Enum):
    """Threshold comparison types"""
    ABOVE = "above"
    BELOW = "below"
    EQUAL = "equal"
    RANGE = "range"


@dataclass
class MetricValue:
    """Single metric measurement"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    agent_id: str
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricStats:
    """Statistical summary of metrics"""
    metric_type: MetricType
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentile_95: float
    percentile_99: float


@dataclass
class Threshold:
    """Performance threshold"""
    threshold_id: str
    metric_type: MetricType
    threshold_type: ThresholdType
    value: float
    upper_bound: Optional[float] = None  # For range type
    severity: AlertSeverity = AlertSeverity.WARNING
    description: str = ""


@dataclass
class Alert:
    """Performance alert"""
    alert_id: str
    agent_id: str
    metric_type: MetricType
    severity: AlertSeverity
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class PerformanceReport:
    """Performance analysis report"""
    report_id: str
    agent_id: str
    start_time: datetime
    end_time: datetime
    metrics: Dict[MetricType, MetricStats]
    alerts: List[Alert]
    recommendations: List[str]
    overall_score: float


class MetricCollector:
    """Collects and stores metrics"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self.metric_counter = 0
    
    def record_metric(self,
                     agent_id: str,
                     metric_type: MetricType,
                     value: float,
                     tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric value.
        
        Args:
            agent_id: Agent identifier
            metric_type: Type of metric
            value: Metric value
            tags: Optional tags
        """
        if tags is None:
            tags = {}
        
        metric = MetricValue(
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            agent_id=agent_id,
            tags=tags
        )
        
        key = "{}:{}".format(agent_id, metric_type.value)
        self.metrics[key].append(metric)
    
    def get_metrics(self,
                   agent_id: str,
                   metric_type: MetricType,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[MetricValue]:
        """
        Get metrics for agent.
        
        Args:
            agent_id: Agent identifier
            metric_type: Type of metric
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of metrics
        """
        key = "{}:{}".format(agent_id, metric_type.value)
        
        if key not in self.metrics:
            return []
        
        metrics = list(self.metrics[key])
        
        # Apply time filters
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]
        
        return metrics
    
    def get_latest_metric(self,
                         agent_id: str,
                         metric_type: MetricType) -> Optional[MetricValue]:
        """Get most recent metric"""
        key = "{}:{}".format(agent_id, metric_type.value)
        
        if key in self.metrics and self.metrics[key]:
            return self.metrics[key][-1]
        
        return None


class MetricAnalyzer:
    """Analyzes metric data"""
    
    def calculate_stats(self, metrics: List[MetricValue]) -> Optional[MetricStats]:
        """
        Calculate statistics for metrics.
        
        Args:
            metrics: List of metrics
            
        Returns:
            Statistical summary
        """
        if not metrics:
            return None
        
        values = [m.value for m in metrics]
        
        stats = MetricStats(
            metric_type=metrics[0].metric_type,
            count=len(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
            min_value=min(values),
            max_value=max(values),
            percentile_95=self._percentile(values, 95),
            percentile_99=self._percentile(values, 99)
        )
        
        return stats
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        index = min(index, len(sorted_values) - 1)
        
        return sorted_values[index]
    
    def detect_trend(self,
                    metrics: List[MetricValue],
                    window_size: int = 10) -> str:
        """
        Detect trend in metrics.
        
        Args:
            metrics: List of metrics
            window_size: Window for trend analysis
            
        Returns:
            Trend direction: "increasing", "decreasing", or "stable"
        """
        if len(metrics) < window_size:
            return "stable"
        
        recent = [m.value for m in metrics[-window_size:]]
        
        # Simple linear regression slope
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(recent)
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        # Threshold for trend detection
        threshold = statistics.stdev(recent) * 0.1 if len(recent) > 1 else 0
        
        if slope > threshold:
            return "increasing"
        elif slope < -threshold:
            return "decreasing"
        else:
            return "stable"
    
    def detect_anomalies(self,
                        metrics: List[MetricValue],
                        std_threshold: float = 3.0) -> List[MetricValue]:
        """
        Detect anomalous metrics using standard deviation.
        
        Args:
            metrics: List of metrics
            std_threshold: Number of standard deviations
            
        Returns:
            List of anomalous metrics
        """
        if len(metrics) < 3:
            return []
        
        values = [m.value for m in metrics]
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)
        
        anomalies = []
        
        for metric in metrics:
            z_score = abs(metric.value - mean) / std_dev if std_dev > 0 else 0
            
            if z_score > std_threshold:
                anomalies.append(metric)
        
        return anomalies


class AlertManager:
    """Manages performance alerts"""
    
    def __init__(self):
        self.thresholds: Dict[str, List[Threshold]] = defaultdict(list)
        self.alerts: List[Alert] = []
        self.alert_counter = 0
        self.alert_handlers: List[Callable] = []
    
    def add_threshold(self, threshold: Threshold) -> None:
        """Add performance threshold"""
        self.thresholds[threshold.metric_type.value].append(threshold)
    
    def check_thresholds(self,
                        agent_id: str,
                        metric: MetricValue) -> List[Alert]:
        """
        Check metric against thresholds.
        
        Args:
            agent_id: Agent identifier
            metric: Metric to check
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        thresholds = self.thresholds.get(metric.metric_type.value, [])
        
        for threshold in thresholds:
            violated = False
            
            if threshold.threshold_type == ThresholdType.ABOVE:
                violated = metric.value > threshold.value
            elif threshold.threshold_type == ThresholdType.BELOW:
                violated = metric.value < threshold.value
            elif threshold.threshold_type == ThresholdType.EQUAL:
                violated = metric.value == threshold.value
            elif threshold.threshold_type == ThresholdType.RANGE:
                if threshold.upper_bound is not None:
                    violated = not (threshold.value <= metric.value <= threshold.upper_bound)
            
            if violated:
                alert = Alert(
                    alert_id="alert_{}".format(self.alert_counter),
                    agent_id=agent_id,
                    metric_type=metric.metric_type,
                    severity=threshold.severity,
                    message=threshold.description or "Threshold violated",
                    current_value=metric.value,
                    threshold_value=threshold.value,
                    timestamp=datetime.now()
                )
                
                self.alert_counter += 1
                self.alerts.append(alert)
                triggered_alerts.append(alert)
                
                # Trigger handlers
                for handler in self.alert_handlers:
                    try:
                        handler(alert)
                    except Exception:
                        pass
        
        return triggered_alerts
    
    def add_alert_handler(self, handler: Callable) -> None:
        """Add alert handler callback"""
        self.alert_handlers.append(handler)
    
    def get_active_alerts(self, agent_id: Optional[str] = None) -> List[Alert]:
        """Get active (unresolved) alerts"""
        active = [a for a in self.alerts if not a.resolved]
        
        if agent_id:
            active = [a for a in active if a.agent_id == agent_id]
        
        return active
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
        return False


class PerformanceOptimizer:
    """Provides optimization recommendations"""
    
    def analyze_performance(self,
                          agent_id: str,
                          metrics_stats: Dict[MetricType, MetricStats]) -> List[str]:
        """
        Analyze performance and provide recommendations.
        
        Args:
            agent_id: Agent identifier
            metrics_stats: Metric statistics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check response time
        if MetricType.RESPONSE_TIME in metrics_stats:
            rt_stats = metrics_stats[MetricType.RESPONSE_TIME]
            
            if rt_stats.mean > 1000:  # Over 1 second
                recommendations.append(
                    "High response time detected ({:.0f}ms avg). "
                    "Consider optimizing processing logic or adding caching.".format(
                        rt_stats.mean
                    )
                )
            
            if rt_stats.percentile_99 > rt_stats.mean * 3:
                recommendations.append(
                    "High response time variability. "
                    "Investigate outliers and optimize slow paths."
                )
        
        # Check error rate
        if MetricType.ERROR_RATE in metrics_stats:
            err_stats = metrics_stats[MetricType.ERROR_RATE]
            
            if err_stats.mean > 5:  # Over 5% error rate
                recommendations.append(
                    "Elevated error rate ({:.1f}%). "
                    "Review error logs and implement error handling improvements.".format(
                        err_stats.mean
                    )
                )
        
        # Check throughput
        if MetricType.THROUGHPUT in metrics_stats:
            tp_stats = metrics_stats[MetricType.THROUGHPUT]
            
            if tp_stats.std_dev > tp_stats.mean * 0.5:
                recommendations.append(
                    "Inconsistent throughput. "
                    "Consider implementing load balancing or rate limiting."
                )
        
        # Check resource usage
        if MetricType.RESOURCE_USAGE in metrics_stats:
            ru_stats = metrics_stats[MetricType.RESOURCE_USAGE]
            
            if ru_stats.mean > 80:  # Over 80% usage
                recommendations.append(
                    "High resource utilization ({:.0f}%). "
                    "Scale resources or optimize resource consumption.".format(
                        ru_stats.mean
                    )
                )
        
        # Check queue length
        if MetricType.QUEUE_LENGTH in metrics_stats:
            ql_stats = metrics_stats[MetricType.QUEUE_LENGTH]
            
            if ql_stats.mean > 100:
                recommendations.append(
                    "Large queue backlog ({:.0f} items avg). "
                    "Increase processing capacity or implement queue management.".format(
                        ql_stats.mean
                    )
                )
        
        return recommendations
    
    def calculate_performance_score(self,
                                   metrics_stats: Dict[MetricType, MetricStats]) -> float:
        """
        Calculate overall performance score (0-100).
        
        Args:
            metrics_stats: Metric statistics
            
        Returns:
            Performance score
        """
        scores = []
        
        # Response time score (inverse)
        if MetricType.RESPONSE_TIME in metrics_stats:
            rt = metrics_stats[MetricType.RESPONSE_TIME].mean
            rt_score = max(0, 100 - (rt / 10))  # 0ms = 100, 1000ms = 0
            scores.append(rt_score)
        
        # Error rate score (inverse)
        if MetricType.ERROR_RATE in metrics_stats:
            err_rate = metrics_stats[MetricType.ERROR_RATE].mean
            err_score = max(0, 100 - err_rate)
            scores.append(err_score)
        
        # Success rate score (direct)
        if MetricType.SUCCESS_RATE in metrics_stats:
            success_rate = metrics_stats[MetricType.SUCCESS_RATE].mean
            scores.append(success_rate)
        
        # Throughput score (normalized to 100)
        if MetricType.THROUGHPUT in metrics_stats:
            tp = metrics_stats[MetricType.THROUGHPUT].mean
            tp_score = min(100, tp)  # Assume 100 req/s is perfect
            scores.append(tp_score)
        
        if not scores:
            return 50.0  # Default neutral score
        
        return statistics.mean(scores)


class AgentPerformanceMonitor:
    """
    Comprehensive performance monitoring system for agents.
    Tracks metrics, generates alerts, and provides optimization insights.
    """
    
    def __init__(self):
        # Components
        self.collector = MetricCollector()
        self.analyzer = MetricAnalyzer()
        self.alert_manager = AlertManager()
        self.optimizer = PerformanceOptimizer()
        
        # Setup default thresholds
        self._setup_default_thresholds()
        
        self.report_counter = 0
    
    def _setup_default_thresholds(self) -> None:
        """Setup default performance thresholds"""
        # Response time threshold
        self.alert_manager.add_threshold(Threshold(
            threshold_id="rt_warning",
            metric_type=MetricType.RESPONSE_TIME,
            threshold_type=ThresholdType.ABOVE,
            value=2000,  # 2 seconds
            severity=AlertSeverity.WARNING,
            description="Response time exceeds 2 seconds"
        ))
        
        self.alert_manager.add_threshold(Threshold(
            threshold_id="rt_critical",
            metric_type=MetricType.RESPONSE_TIME,
            threshold_type=ThresholdType.ABOVE,
            value=5000,  # 5 seconds
            severity=AlertSeverity.CRITICAL,
            description="Response time exceeds 5 seconds"
        ))
        
        # Error rate threshold
        self.alert_manager.add_threshold(Threshold(
            threshold_id="error_warning",
            metric_type=MetricType.ERROR_RATE,
            threshold_type=ThresholdType.ABOVE,
            value=5.0,  # 5%
            severity=AlertSeverity.WARNING,
            description="Error rate exceeds 5%"
        ))
        
        # Resource usage threshold
        self.alert_manager.add_threshold(Threshold(
            threshold_id="resource_warning",
            metric_type=MetricType.RESOURCE_USAGE,
            threshold_type=ThresholdType.ABOVE,
            value=90.0,  # 90%
            severity=AlertSeverity.WARNING,
            description="Resource usage exceeds 90%"
        ))
    
    def record(self,
              agent_id: str,
              metric_type: MetricType,
              value: float,
              tags: Optional[Dict[str, str]] = None) -> List[Alert]:
        """
        Record metric and check thresholds.
        
        Args:
            agent_id: Agent identifier
            metric_type: Type of metric
            value: Metric value
            tags: Optional tags
            
        Returns:
            List of triggered alerts
        """
        # Record metric
        self.collector.record_metric(agent_id, metric_type, value, tags)
        
        # Get latest metric
        metric = self.collector.get_latest_metric(agent_id, metric_type)
        
        if not metric:
            return []
        
        # Check thresholds
        return self.alert_manager.check_thresholds(agent_id, metric)
    
    def get_current_metrics(self,
                           agent_id: str) -> Dict[MetricType, float]:
        """Get current metric values for agent"""
        current = {}
        
        for metric_type in MetricType:
            metric = self.collector.get_latest_metric(agent_id, metric_type)
            if metric:
                current[metric_type] = metric.value
        
        return current
    
    def get_metric_stats(self,
                        agent_id: str,
                        metric_type: MetricType,
                        time_window: Optional[timedelta] = None) -> Optional[MetricStats]:
        """
        Get statistics for a metric.
        
        Args:
            agent_id: Agent identifier
            metric_type: Type of metric
            time_window: Optional time window
            
        Returns:
            Metric statistics
        """
        start_time = None
        if time_window:
            start_time = datetime.now() - time_window
        
        metrics = self.collector.get_metrics(
            agent_id,
            metric_type,
            start_time=start_time
        )
        
        return self.analyzer.calculate_stats(metrics)
    
    def detect_anomalies(self,
                        agent_id: str,
                        metric_type: MetricType) -> List[MetricValue]:
        """Detect anomalies in metrics"""
        metrics = self.collector.get_metrics(agent_id, metric_type)
        return self.analyzer.detect_anomalies(metrics)
    
    def get_trend(self,
                 agent_id: str,
                 metric_type: MetricType) -> str:
        """Get trend for metric"""
        metrics = self.collector.get_metrics(agent_id, metric_type)
        return self.analyzer.detect_trend(metrics)
    
    def generate_report(self,
                       agent_id: str,
                       time_window: timedelta = timedelta(hours=1)) -> PerformanceReport:
        """
        Generate performance report for agent.
        
        Args:
            agent_id: Agent identifier
            time_window: Time window for report
            
        Returns:
            Performance report
        """
        end_time = datetime.now()
        start_time = end_time - time_window
        
        # Collect statistics for all metrics
        metrics_stats = {}
        
        for metric_type in MetricType:
            stats = self.get_metric_stats(
                agent_id,
                metric_type,
                time_window
            )
            
            if stats:
                metrics_stats[metric_type] = stats
        
        # Get active alerts
        alerts = self.alert_manager.get_active_alerts(agent_id)
        
        # Get recommendations
        recommendations = self.optimizer.analyze_performance(
            agent_id,
            metrics_stats
        )
        
        # Calculate performance score
        score = self.optimizer.calculate_performance_score(metrics_stats)
        
        report = PerformanceReport(
            report_id="report_{}".format(self.report_counter),
            agent_id=agent_id,
            start_time=start_time,
            end_time=end_time,
            metrics=metrics_stats,
            alerts=alerts,
            recommendations=recommendations,
            overall_score=score
        )
        
        self.report_counter += 1
        
        return report
    
    def add_custom_threshold(self,
                            metric_type: MetricType,
                            threshold_type: ThresholdType,
                            value: float,
                            severity: AlertSeverity = AlertSeverity.WARNING,
                            description: str = "") -> str:
        """Add custom threshold"""
        threshold_id = "custom_{}".format(len(self.alert_manager.thresholds))
        
        threshold = Threshold(
            threshold_id=threshold_id,
            metric_type=metric_type,
            threshold_type=threshold_type,
            value=value,
            severity=severity,
            description=description
        )
        
        self.alert_manager.add_threshold(threshold)
        
        return threshold_id
    
    def subscribe_to_alerts(self, handler: Callable) -> None:
        """Subscribe to alert notifications"""
        self.alert_manager.add_alert_handler(handler)
    
    def get_dashboard_data(self, agent_id: str) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        current_metrics = self.get_current_metrics(agent_id)
        active_alerts = self.alert_manager.get_active_alerts(agent_id)
        
        # Get recent trends
        trends = {}
        for metric_type in MetricType:
            trends[metric_type.value] = self.get_trend(agent_id, metric_type)
        
        return {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "current_metrics": {
                mt.value: v for mt, v in current_metrics.items()
            },
            "active_alerts": [
                {
                    "severity": a.severity.value,
                    "message": a.message,
                    "value": a.current_value
                }
                for a in active_alerts
            ],
            "trends": trends
        }


def demonstrate_performance_monitor():
    """Demonstrate agent performance monitor"""
    print("=" * 70)
    print("Agent Performance Monitor Demonstration")
    print("=" * 70)
    
    monitor = AgentPerformanceMonitor()
    
    # Example 1: Record metrics
    print("\n1. Recording Performance Metrics:")
    
    import random
    import time
    
    agent_id = "agent_1"
    
    # Simulate metrics over time
    for i in range(20):
        # Response time with some variance
        response_time = 500 + random.gauss(0, 200)
        alerts = monitor.record(agent_id, MetricType.RESPONSE_TIME, response_time)
        
        # Throughput
        throughput = 50 + random.gauss(0, 10)
        monitor.record(agent_id, MetricType.THROUGHPUT, throughput)
        
        # Error rate
        error_rate = random.uniform(0, 3)
        monitor.record(agent_id, MetricType.ERROR_RATE, error_rate)
        
        # Success rate
        monitor.record(agent_id, MetricType.SUCCESS_RATE, 100 - error_rate)
        
        if alerts:
            print("  Alert triggered at step {}: {}".format(i, alerts[0].message))
        
        time.sleep(0.01)  # Small delay
    
    print("  Recorded 20 metric samples")
    
    # Example 2: Get current metrics
    print("\n2. Current Metrics:")
    
    current = monitor.get_current_metrics(agent_id)
    
    for metric_type, value in current.items():
        print("  {}: {:.2f}".format(metric_type.value, value))
    
    # Example 3: Get statistics
    print("\n3. Metric Statistics (Last Hour):")
    
    stats = monitor.get_metric_stats(
        agent_id,
        MetricType.RESPONSE_TIME,
        timedelta(hours=1)
    )
    
    if stats:
        print("  Response Time Statistics:")
        print("    Mean: {:.2f}ms".format(stats.mean))
        print("    Median: {:.2f}ms".format(stats.median))
        print("    Std Dev: {:.2f}ms".format(stats.std_dev))
        print("    Min: {:.2f}ms".format(stats.min_value))
        print("    Max: {:.2f}ms".format(stats.max_value))
        print("    95th percentile: {:.2f}ms".format(stats.percentile_95))
        print("    99th percentile: {:.2f}ms".format(stats.percentile_99))
    
    # Example 4: Detect trends
    print("\n4. Performance Trends:")
    
    for metric_type in [MetricType.RESPONSE_TIME, MetricType.THROUGHPUT, MetricType.ERROR_RATE]:
        trend = monitor.get_trend(agent_id, metric_type)
        print("  {}: {}".format(metric_type.value, trend))
    
    # Example 5: Add custom threshold
    print("\n5. Adding Custom Threshold:")
    
    threshold_id = monitor.add_custom_threshold(
        metric_type=MetricType.THROUGHPUT,
        threshold_type=ThresholdType.BELOW,
        value=30.0,
        severity=AlertSeverity.WARNING,
        description="Throughput below 30 req/s"
    )
    
    print("  Added threshold: {}".format(threshold_id))
    
    # Trigger threshold
    monitor.record(agent_id, MetricType.THROUGHPUT, 25.0)
    
    # Example 6: Alert handling
    print("\n6. Alert Management:")
    
    active_alerts = monitor.alert_manager.get_active_alerts(agent_id)
    
    print("  Active alerts: {}".format(len(active_alerts)))
    
    for alert in active_alerts[:3]:
        print("    [{}] {}: {:.2f}".format(
            alert.severity.value.upper(),
            alert.message,
            alert.current_value
        ))
    
    # Example 7: Subscribe to alerts
    print("\n7. Alert Subscription:")
    
    def alert_handler(alert: Alert):
        print("  ⚠️  Alert received: {} - {}".format(
            alert.severity.value.upper(),
            alert.message
        ))
    
    monitor.subscribe_to_alerts(alert_handler)
    
    # Trigger alert
    monitor.record(agent_id, MetricType.RESPONSE_TIME, 6000)  # Over threshold
    
    # Example 8: Detect anomalies
    print("\n8. Anomaly Detection:")
    
    # Add some anomalous data
    monitor.record(agent_id, MetricType.RESPONSE_TIME, 5000)  # Anomaly
    
    anomalies = monitor.detect_anomalies(agent_id, MetricType.RESPONSE_TIME)
    
    print("  Detected {} anomalies in response time".format(len(anomalies)))
    
    for anomaly in anomalies[:3]:
        print("    Value: {:.2f}ms at {}".format(
            anomaly.value,
            anomaly.timestamp.strftime("%H:%M:%S")
        ))
    
    # Example 9: Generate performance report
    print("\n9. Performance Report:")
    
    report = monitor.generate_report(agent_id, timedelta(hours=1))
    
    print("  Report ID: {}".format(report.report_id))
    print("  Time Range: {} to {}".format(
        report.start_time.strftime("%H:%M:%S"),
        report.end_time.strftime("%H:%M:%S")
    ))
    print("  Overall Score: {:.1f}/100".format(report.overall_score))
    print("  Active Alerts: {}".format(len(report.alerts)))
    
    if report.recommendations:
        print("  Recommendations:")
        for rec in report.recommendations:
            print("    - {}".format(rec))
    
    # Example 10: Dashboard data
    print("\n10. Dashboard Data:")
    
    dashboard = monitor.get_dashboard_data(agent_id)
    print(json.dumps(dashboard, indent=2))


if __name__ == "__main__":
    demonstrate_performance_monitor()
