"""
Pattern 096: Performance Profiling

Description:
    Performance Profiling systematically measures and analyzes agent execution to identify
    bottlenecks, optimize resource usage, and improve response times. This pattern provides
    detailed insights into where time and resources are spent, enabling data-driven
    optimization decisions. Profiling is essential for production systems where performance,
    cost, and user experience are critical.

    Key metrics profiled:
    - Execution time (total, per step, per component)
    - LLM token usage (input, output, total)
    - API call counts and latency
    - Memory usage and allocation patterns
    - Cache hit/miss rates
    - Concurrent operation efficiency
    - Cost per operation

Components:
    1. Profiler
       - Time measurement (wall clock, CPU)
       - Memory tracking
       - Token counting
       - API call monitoring
       - Resource utilization
       - Custom metric collection

    2. Metrics Collector
       - Aggregates measurements
       - Calculates statistics
       - Tracks trends over time
       - Identifies outliers
       - Correlates metrics
       - Exports data

    3. Analyzer
       - Identifies bottlenecks
       - Compares components
       - Detects regressions
       - Suggests optimizations
       - Benchmarks performance
       - Generates reports

    4. Visualizer
       - Flame graphs
       - Timeline charts
       - Breakdown pie charts
       - Trend graphs
       - Comparison tables
       - Heatmaps

Use Cases:
    1. Performance Optimization
       - Identify slow components
       - Optimize hot paths
       - Reduce latency
       - Improve throughput
       - Minimize resource usage
       - Cost reduction

    2. Capacity Planning
       - Understand resource needs
       - Plan scaling requirements
       - Budget forecasting
       - Load handling capacity
       - Infrastructure sizing
       - Cost projections

    3. Regression Detection
       - Catch performance degradation
       - Validate optimizations
       - Monitor after deployments
       - Track trends over time
       - Alert on anomalies
       - Automated testing

    4. Debugging & Root Cause
       - Understand slow requests
       - Identify failure points
       - Analyze edge cases
       - Reproduce issues
       - Validate fixes
       - Production troubleshooting

    5. SLA Compliance
       - Monitor latency targets
       - Track availability
       - Measure throughput
       - Ensure quality of service
       - Report on SLAs
       - Alert on violations

LangChain Implementation:
    LangChain supports profiling through:
    - Callback handlers for timing
    - Token usage tracking
    - LangSmith for comprehensive monitoring
    - Custom profiling wrappers
    - Integration with monitoring tools

Key Features:
    1. Low Overhead
       - Minimal performance impact (< 2%)
       - Efficient data collection
       - Sampling strategies
       - Async operations
       - Batch processing

    2. Comprehensive Coverage
       - End-to-end timing
       - Per-component breakdown
       - LLM call analysis
       - Tool execution timing
       - Memory profiling
       - Network latency

    3. Actionable Insights
       - Clear bottleneck identification
       - Optimization suggestions
       - Cost breakdowns
       - Performance comparisons
       - Trend analysis
       - Automated alerts

    4. Production Ready
       - Always-on capability
       - Sampling support
       - Secure data handling
       - Scalable architecture
       - Integration friendly
       - Export options

Best Practices:
    1. What to Profile
       - Total execution time
       - LLM calls (time, tokens, cost)
       - Tool executions
       - Retrieval operations
       - Parsing/processing time
       - Network calls
       - Database queries
       - Cache operations

    2. Metric Collection
       - Start time, end time, duration
       - Token counts (input, output, total)
       - API call counts
       - Memory usage (peak, average)
       - Error rates
       - Retry counts
       - Cache hit rates

    3. Analysis Techniques
       - Identify top time consumers
       - Calculate percentiles (p50, p95, p99)
       - Detect outliers
       - Compare before/after
       - Trend analysis
       - Cost attribution

    4. Optimization Priorities
       - Focus on hot paths (80/20 rule)
       - Optimize high-frequency calls
       - Reduce expensive operations
       - Improve cache usage
       - Batch when possible
       - Parallel execution

Trade-offs:
    Advantages:
    - Identifies bottlenecks
    - Data-driven optimization
    - Cost reduction
    - Improved user experience
    - Regression detection
    - Capacity planning

    Disadvantages:
    - Small overhead (1-2%)
    - Implementation complexity
    - Storage requirements
    - Analysis time needed
    - May expose limitations
    - Maintenance burden

Production Considerations:
    1. Sampling Strategy
       - Development: 100% profiling
       - Staging: 50-100%
       - Production: 1-10% baseline
       - Slow requests: Always profile
       - Adaptive sampling

    2. Storage & Retention
       - Hot data: 7 days (full detail)
       - Warm data: 30 days (aggregated)
       - Cold data: 90+ days (summary)
       - Compression for old data
       - Archival policies

    3. Performance Impact
       - Target: < 2% overhead
       - Measure actual impact
       - Use efficient collectors
       - Async writes
       - Batch operations

    4. Privacy & Security
       - Redact sensitive data
       - Secure storage
       - Access controls
       - Retention limits
       - Compliance adherence

    5. Integration
       - Export to monitoring systems
       - Alerting integration
       - Dashboard visualization
       - API for analysis
       - CI/CD integration
"""

import os
import time
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class MetricType(Enum):
    """Types of metrics to track"""
    TIME = "time"  # Execution time
    TOKENS = "tokens"  # Token usage
    COST = "cost"  # API cost
    MEMORY = "memory"  # Memory usage
    COUNT = "count"  # Call counts


@dataclass
class ProfileMetric:
    """A single profiling metric"""
    name: str
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentProfile:
    """Profile data for a component"""
    component_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    metrics: List[ProfileMetric] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_metric(self, metric: ProfileMetric):
        """Add a metric to profile"""
        self.metrics.append(metric)
    
    def finalize(self):
        """Finalize profile with end time"""
        if not self.end_time:
            self.end_time = datetime.now()
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000


@dataclass
class ExecutionProfile:
    """Complete execution profile"""
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_ms: Optional[float] = None
    components: List[ComponentProfile] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_component(self, component: ComponentProfile):
        """Add component profile"""
        self.components.append(component)
    
    def finalize(self):
        """Finalize execution profile"""
        if not self.end_time:
            self.end_time = datetime.now()
            self.total_duration_ms = (self.end_time - self.start_time).total_seconds() * 1000


class Profiler:
    """
    Performance profiler for agent execution.
    
    Tracks timing, token usage, costs, and other metrics.
    """
    
    def __init__(self):
        """Initialize profiler"""
        self.profiles: List[ExecutionProfile] = []
        self.current_profile: Optional[ExecutionProfile] = None
        self.component_stack: List[ComponentProfile] = []
    
    def start_execution(self, execution_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start profiling an execution.
        
        Args:
            execution_id: Unique execution identifier
            metadata: Optional metadata
            
        Returns:
            Execution ID
        """
        profile = ExecutionProfile(
            execution_id=execution_id,
            start_time=datetime.now(),
            metadata=metadata or {}
        )
        self.current_profile = profile
        self.profiles.append(profile)
        return execution_id
    
    def end_execution(self):
        """End profiling current execution"""
        if self.current_profile:
            self.current_profile.finalize()
            self.current_profile = None
    
    def start_component(self, component_name: str, metadata: Optional[Dict[str, Any]] = None) -> ComponentProfile:
        """
        Start profiling a component.
        
        Args:
            component_name: Component name
            metadata: Optional metadata
            
        Returns:
            Component profile
        """
        component = ComponentProfile(
            component_name=component_name,
            start_time=datetime.now(),
            metadata=metadata or {}
        )
        
        if self.current_profile:
            self.current_profile.add_component(component)
        
        self.component_stack.append(component)
        return component
    
    def end_component(self):
        """End profiling current component"""
        if self.component_stack:
            component = self.component_stack.pop()
            component.finalize()
    
    def add_metric(self, metric: ProfileMetric):
        """Add metric to current component"""
        if self.component_stack:
            self.component_stack[-1].add_metric(metric)
    
    def get_profile(self, execution_id: str) -> Optional[ExecutionProfile]:
        """Get profile by execution ID"""
        for profile in self.profiles:
            if profile.execution_id == execution_id:
                return profile
        return None
    
    def clear_profiles(self):
        """Clear all profiles"""
        self.profiles.clear()
        self.current_profile = None
        self.component_stack.clear()


class ProfileAnalyzer:
    """
    Analyzes profiling data to identify bottlenecks and trends.
    
    Provides insights and optimization recommendations.
    """
    
    def __init__(self):
        """Initialize analyzer"""
        pass
    
    def analyze_execution(self, profile: ExecutionProfile) -> Dict[str, Any]:
        """
        Analyze execution profile.
        
        Args:
            profile: Execution profile
            
        Returns:
            Analysis results
        """
        if not profile.components:
            return {
                "total_duration_ms": profile.total_duration_ms or 0,
                "component_count": 0,
                "components": []
            }
        
        # Component timing breakdown
        components_analysis = []
        total_component_time = 0.0
        
        for component in profile.components:
            if component.duration_ms:
                total_component_time += component.duration_ms
                
                components_analysis.append({
                    "name": component.component_name,
                    "duration_ms": component.duration_ms,
                    "percentage": 0.0,  # Will calculate after
                    "metric_count": len(component.metrics)
                })
        
        # Calculate percentages
        if total_component_time > 0:
            for comp in components_analysis:
                comp["percentage"] = (comp["duration_ms"] / total_component_time) * 100
        
        # Sort by duration
        components_analysis.sort(key=lambda c: c["duration_ms"], reverse=True)
        
        return {
            "execution_id": profile.execution_id,
            "total_duration_ms": profile.total_duration_ms or 0,
            "component_count": len(profile.components),
            "total_component_time_ms": total_component_time,
            "components": components_analysis,
            "overhead_ms": (profile.total_duration_ms or 0) - total_component_time
        }
    
    def identify_bottlenecks(self, profile: ExecutionProfile, threshold_pct: float = 20.0) -> List[Dict[str, Any]]:
        """
        Identify bottleneck components.
        
        Args:
            profile: Execution profile
            threshold_pct: Threshold percentage for bottleneck
            
        Returns:
            List of bottlenecks
        """
        analysis = self.analyze_execution(profile)
        
        bottlenecks = []
        for component in analysis["components"]:
            if component["percentage"] >= threshold_pct:
                bottlenecks.append({
                    "component": component["name"],
                    "duration_ms": component["duration_ms"],
                    "percentage": component["percentage"],
                    "severity": "high" if component["percentage"] >= 40 else "medium"
                })
        
        return bottlenecks
    
    def calculate_statistics(self, profiles: List[ExecutionProfile]) -> Dict[str, Any]:
        """
        Calculate statistics across multiple profiles.
        
        Args:
            profiles: List of execution profiles
            
        Returns:
            Statistical summary
        """
        if not profiles:
            return {}
        
        durations = [p.total_duration_ms for p in profiles if p.total_duration_ms]
        
        if not durations:
            return {}
        
        durations.sort()
        count = len(durations)
        
        return {
            "count": count,
            "min_ms": durations[0],
            "max_ms": durations[-1],
            "mean_ms": sum(durations) / count,
            "median_ms": durations[count // 2],
            "p95_ms": durations[int(count * 0.95)] if count > 1 else durations[0],
            "p99_ms": durations[int(count * 0.99)] if count > 1 else durations[0]
        }
    
    def compare_profiles(self, profile1: ExecutionProfile, profile2: ExecutionProfile) -> Dict[str, Any]:
        """
        Compare two profiles.
        
        Args:
            profile1: First profile (baseline)
            profile2: Second profile (current)
            
        Returns:
            Comparison results
        """
        analysis1 = self.analyze_execution(profile1)
        analysis2 = self.analyze_execution(profile2)
        
        duration_change = analysis2["total_duration_ms"] - analysis1["total_duration_ms"]
        duration_change_pct = (duration_change / analysis1["total_duration_ms"]) * 100 if analysis1["total_duration_ms"] > 0 else 0
        
        return {
            "baseline_duration_ms": analysis1["total_duration_ms"],
            "current_duration_ms": analysis2["total_duration_ms"],
            "change_ms": duration_change,
            "change_percentage": duration_change_pct,
            "is_regression": duration_change_pct > 10,  # > 10% slower
            "is_improvement": duration_change_pct < -10  # > 10% faster
        }
    
    def generate_recommendations(self, profile: ExecutionProfile) -> List[str]:
        """
        Generate optimization recommendations.
        
        Args:
            profile: Execution profile
            
        Returns:
            List of recommendations
        """
        recommendations = []
        bottlenecks = self.identify_bottlenecks(profile)
        
        for bottleneck in bottlenecks:
            component = bottleneck["component"]
            
            if "llm" in component.lower():
                recommendations.append(
                    f"Consider caching LLM responses for {component} (takes {bottleneck['percentage']:.1f}% of time)"
                )
                recommendations.append(
                    f"Try using a faster model for {component} if high accuracy isn't critical"
                )
            elif "retrieval" in component.lower() or "search" in component.lower():
                recommendations.append(
                    f"Optimize retrieval in {component} - consider better indexing or caching"
                )
            elif "tool" in component.lower():
                recommendations.append(
                    f"Tool execution in {component} is slow - consider async execution or caching"
                )
            else:
                recommendations.append(
                    f"Investigate {component} - it accounts for {bottleneck['percentage']:.1f}% of execution time"
                )
        
        if not bottlenecks:
            recommendations.append("No significant bottlenecks detected - performance is well-balanced")
        
        return recommendations


class ProfileVisualizer:
    """
    Visualizes profiling data.
    
    Creates text-based visualizations for analysis.
    """
    
    def __init__(self):
        """Initialize visualizer"""
        pass
    
    def visualize_breakdown(self, analysis: Dict[str, Any]) -> str:
        """
        Visualize time breakdown by component.
        
        Args:
            analysis: Analysis results
            
        Returns:
            Text visualization
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"EXECUTION BREAKDOWN: {analysis['execution_id']}")
        lines.append("=" * 80)
        lines.append(f"Total Duration: {analysis['total_duration_ms']:.2f}ms")
        lines.append(f"Components: {analysis['component_count']}")
        lines.append("")
        
        if analysis["components"]:
            lines.append("Time by Component:")
            lines.append("")
            
            for component in analysis["components"]:
                # Create bar
                bar_length = int((component["percentage"] / 100) * 40)
                bar = "█" * bar_length
                
                lines.append(f"{component['name']:30s} {bar} {component['percentage']:5.1f}% ({component['duration_ms']:7.2f}ms)")
        
        return "\n".join(lines)
    
    def visualize_statistics(self, stats: Dict[str, Any]) -> str:
        """
        Visualize statistics.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            Text visualization
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE STATISTICS")
        lines.append("=" * 80)
        lines.append(f"Sample Size: {stats.get('count', 0)} executions")
        lines.append("")
        lines.append(f"Min:     {stats.get('min_ms', 0):8.2f}ms")
        lines.append(f"Median:  {stats.get('median_ms', 0):8.2f}ms")
        lines.append(f"Mean:    {stats.get('mean_ms', 0):8.2f}ms")
        lines.append(f"P95:     {stats.get('p95_ms', 0):8.2f}ms")
        lines.append(f"P99:     {stats.get('p99_ms', 0):8.2f}ms")
        lines.append(f"Max:     {stats.get('max_ms', 0):8.2f}ms")
        
        return "\n".join(lines)


class ProfiledAgent:
    """
    Example agent with comprehensive profiling.
    
    Demonstrates performance profiling in practice.
    """
    
    def __init__(self, profiler: Profiler):
        """
        Initialize profiled agent.
        
        Args:
            profiler: Profiler instance
        """
        self.profiler = profiler
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    def process_query(self, query: str, execution_id: str) -> str:
        """
        Process query with profiling.
        
        Args:
            query: User query
            execution_id: Execution identifier
            
        Returns:
            Response
        """
        # Start execution profiling
        self.profiler.start_execution(execution_id, {"query": query})
        
        try:
            # Step 1: Query analysis
            self.profiler.start_component("query_analysis")
            time.sleep(0.05)  # Simulate processing
            query_type = "question" if "?" in query else "statement"
            self.profiler.end_component()
            
            # Step 2: LLM call
            self.profiler.start_component("llm_generation")
            start_time = time.time()
            response = self._call_llm(query)
            llm_duration = (time.time() - start_time) * 1000
            
            # Add metrics
            self.profiler.add_metric(ProfileMetric(
                name="llm_tokens_input",
                metric_type=MetricType.TOKENS,
                value=len(query.split()) * 1.3,  # Approximate
                unit="tokens",
                timestamp=datetime.now()
            ))
            self.profiler.add_metric(ProfileMetric(
                name="llm_tokens_output",
                metric_type=MetricType.TOKENS,
                value=len(response.split()) * 1.3,  # Approximate
                unit="tokens",
                timestamp=datetime.now()
            ))
            
            self.profiler.end_component()
            
            # Step 3: Post-processing
            self.profiler.start_component("post_processing")
            time.sleep(0.02)  # Simulate processing
            processed_response = response.strip()
            self.profiler.end_component()
            
            # End execution profiling
            self.profiler.end_execution()
            
            return processed_response
            
        except Exception as e:
            self.profiler.end_execution()
            raise
    
    def _call_llm(self, query: str) -> str:
        """Call LLM"""
        prompt = ChatPromptTemplate.from_template("Answer concisely: {query}")
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})


def demonstrate_performance_profiling():
    """Demonstrate performance profiling"""
    print("=" * 80)
    print("PERFORMANCE PROFILING DEMONSTRATION")
    print("=" * 80)
    
    profiler = Profiler()
    analyzer = ProfileAnalyzer()
    visualizer = ProfileVisualizer()
    
    # Example 1: Basic Profiling
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Component Profiling")
    print("=" * 80)
    
    execution_id = "exec_001"
    profiler.start_execution(execution_id, {"user": "demo"})
    
    # Simulate components
    profiler.start_component("initialization")
    time.sleep(0.05)
    profiler.end_component()
    
    profiler.start_component("data_loading")
    time.sleep(0.1)
    profiler.add_metric(ProfileMetric(
        name="records_loaded",
        metric_type=MetricType.COUNT,
        value=100,
        unit="records",
        timestamp=datetime.now()
    ))
    profiler.end_component()
    
    profiler.start_component("processing")
    time.sleep(0.15)
    profiler.end_component()
    
    profiler.start_component("output_formatting")
    time.sleep(0.03)
    profiler.end_component()
    
    profiler.end_execution()
    
    # Analyze
    profile = profiler.get_profile(execution_id)
    if profile:
        analysis = analyzer.analyze_execution(profile)
        print("\n" + visualizer.visualize_breakdown(analysis))
    
    # Example 2: Bottleneck Identification
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Bottleneck Identification")
    print("=" * 80)
    
    if profile:
        bottlenecks = analyzer.identify_bottlenecks(profile, threshold_pct=15.0)
        
        print(f"\nIdentified {len(bottlenecks)} bottleneck(s):\n")
        for bottleneck in bottlenecks:
            print(f"Component: {bottleneck['component']}")
            print(f"  Duration: {bottleneck['duration_ms']:.2f}ms")
            print(f"  Percentage: {bottleneck['percentage']:.1f}%")
            print(f"  Severity: {bottleneck['severity']}")
            print()
    
    # Example 3: Recommendations
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Optimization Recommendations")
    print("=" * 80)
    
    if profile:
        recommendations = analyzer.generate_recommendations(profile)
        
        print("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    # Example 4: Profiled Agent Execution
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Full Agent Execution Profiling")
    print("=" * 80)
    
    profiler.clear_profiles()
    agent = ProfiledAgent(profiler)
    
    print("\nExecuting agent with profiling...")
    response = agent.process_query("What is artificial intelligence?", "exec_002")
    print(f"\nResponse: {response[:100]}...")
    
    # Analyze agent execution
    agent_profile = profiler.get_profile("exec_002")
    if agent_profile:
        agent_analysis = analyzer.analyze_execution(agent_profile)
        print("\n" + visualizer.visualize_breakdown(agent_analysis))
    
    # Example 5: Statistical Analysis
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Statistical Analysis Across Multiple Runs")
    print("=" * 80)
    
    # Simulate multiple executions
    print("\nRunning 10 executions...")
    for i in range(10):
        agent.process_query(f"Query {i}", f"exec_batch_{i}")
    
    # Calculate statistics
    all_profiles = profiler.profiles
    stats = analyzer.calculate_statistics(all_profiles)
    
    print("\n" + visualizer.visualize_statistics(stats))
    
    # Example 6: Performance Comparison
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Before/After Comparison")
    print("=" * 80)
    
    # Get two profiles to compare
    if len(all_profiles) >= 2:
        baseline = all_profiles[0]
        current = all_profiles[-1]
        
        comparison = analyzer.compare_profiles(baseline, current)
        
        print("\nPerformance Comparison:")
        print(f"Baseline: {comparison['baseline_duration_ms']:.2f}ms")
        print(f"Current:  {comparison['current_duration_ms']:.2f}ms")
        print(f"Change:   {comparison['change_ms']:+.2f}ms ({comparison['change_percentage']:+.1f}%)")
        
        if comparison['is_regression']:
            print("⚠️  REGRESSION DETECTED - Performance degraded")
        elif comparison['is_improvement']:
            print("✓ IMPROVEMENT - Performance improved")
        else:
            print("→ STABLE - No significant change")
    
    # Example 7: Token Usage Analysis
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Token Usage and Cost Analysis")
    print("=" * 80)
    
    if agent_profile:
        total_input_tokens = 0
        total_output_tokens = 0
        
        for component in agent_profile.components:
            for metric in component.metrics:
                if "input" in metric.name:
                    total_input_tokens += metric.value
                elif "output" in metric.name:
                    total_output_tokens += metric.value
        
        total_tokens = total_input_tokens + total_output_tokens
        
        # Approximate costs (GPT-3.5-turbo pricing)
        input_cost = (total_input_tokens / 1000) * 0.0015
        output_cost = (total_output_tokens / 1000) * 0.002
        total_cost = input_cost + output_cost
        
        print("\nToken Usage:")
        print(f"Input tokens:  {total_input_tokens:.0f}")
        print(f"Output tokens: {total_output_tokens:.0f}")
        print(f"Total tokens:  {total_tokens:.0f}")
        print(f"\nEstimated Cost:")
        print(f"Input cost:    ${input_cost:.6f}")
        print(f"Output cost:   ${output_cost:.6f}")
        print(f"Total cost:    ${total_cost:.6f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("PERFORMANCE PROFILING SUMMARY")
    print("=" * 80)
    print("""
Performance Profiling Benefits:
1. Identify Bottlenecks: Find slow components
2. Data-Driven Optimization: Focus efforts where they matter
3. Cost Control: Track and optimize API usage
4. Regression Detection: Catch performance degradation
5. Capacity Planning: Understand resource needs
6. SLA Compliance: Monitor latency targets

Key Metrics to Track:
1. Execution Time
   - Total duration (wall clock)
   - Per-component breakdown
   - CPU time vs wall time
   - Percentiles (p50, p95, p99)
   - Min/max/mean/median

2. Token Usage
   - Input tokens per call
   - Output tokens per call
   - Total tokens per execution
   - Cost per execution
   - Cost trends over time

3. API Calls
   - Call counts by endpoint
   - Latency per call
   - Error rates
   - Retry counts
   - Rate limit hits

4. Memory Usage
   - Peak memory
   - Average memory
   - Memory leaks detection
   - Allocation patterns
   - Cache memory

5. Cache Performance
   - Hit rate
   - Miss rate
   - Cache size
   - Eviction rate
   - Time saved

6. Throughput
   - Requests per second
   - Concurrent operations
   - Queue lengths
   - Success rate
   - Error rate

Profiling Workflow:
```
1. Instrument Code
   ├── Add profiling wrappers
   ├── Track entry/exit points
   ├── Collect metrics
   └── Handle errors

2. Execute & Collect
   ├── Run profiled code
   ├── Gather measurements
   ├── Store profile data
   └── Tag with metadata

3. Analyze Data
   ├── Calculate statistics
   ├── Identify bottlenecks
   ├── Compare versions
   └── Detect trends

4. Generate Insights
   ├── Bottleneck ranking
   ├── Optimization suggestions
   ├── Cost breakdown
   └── Performance report

5. Optimize & Repeat
   ├── Implement improvements
   ├── Profile again
   ├── Validate changes
   └── Monitor trends
```

Analysis Techniques:
1. Time Breakdown
   - Which component takes most time?
   - What percentage per component?
   - Sequential vs parallel?
   - Overhead calculation

2. Statistical Analysis
   - Mean, median, mode
   - Standard deviation
   - Percentiles (p50, p95, p99)
   - Outlier detection
   - Trend analysis

3. Comparison Analysis
   - Before vs after optimization
   - Version A vs version B
   - Expected vs actual
   - Production vs development
   - Baseline tracking

4. Cost Analysis
   - Cost per execution
   - Cost by component
   - Cost trends
   - Budget projection
   - ROI of optimizations

Optimization Strategies:
1. Caching (Highest Impact)
   - Cache LLM responses
   - Cache retrieval results
   - Cache computations
   - Set appropriate TTL
   - Invalidation strategy

2. Model Selection
   - Use faster models when possible
   - GPT-3.5 vs GPT-4 trade-off
   - Smaller models for simple tasks
   - Local models for common patterns
   - Model routing by complexity

3. Parallel Execution
   - Independent operations in parallel
   - Batch API calls
   - Async processing
   - Resource pooling
   - Load balancing

4. Prompt Optimization
   - Shorter prompts (less tokens)
   - Clear instructions (less retries)
   - Better examples (higher success)
   - Output format optimization
   - Token budget management

5. Retrieval Optimization
   - Better indexing
   - Smaller chunks
   - Top-k optimization
   - Embedding cache
   - Pre-computed similarities

6. Algorithm Improvements
   - More efficient logic
   - Early termination
   - Lazy evaluation
   - Result reuse
   - Approximations when appropriate

Production Profiling:
1. Sampling Strategy
   - Baseline: 1-5% always on
   - Errors: 100% profiling
   - Slow requests: 100% profiling
   - Adaptive: Increase on issues
   - Feature flags for control

2. Overhead Management
   - Target: < 2% overhead
   - Async data collection
   - Batch writes
   - Efficient storage
   - Sampling when needed

3. Storage & Retention
   - Hot: 7 days (full detail)
   - Warm: 30 days (aggregated)
   - Cold: 90+ days (summary)
   - Automatic cleanup
   - Compression

4. Alerting
   - Latency > threshold
   - Error rate spike
   - Cost > budget
   - Regression detected
   - Resource exhaustion

5. Integration
   - Export to monitoring (DataDog, New Relic)
   - Dashboard visualization (Grafana)
   - CI/CD integration
   - Automated regression tests
   - Alert systems

Common Bottlenecks:
1. LLM Calls (Most Common)
   - Solution: Caching, faster models, parallel calls
   - Impact: 50-80% of total time

2. Retrieval/Search
   - Solution: Better indexing, caching, smaller top-k
   - Impact: 10-30% of total time

3. Tool Execution
   - Solution: Async execution, caching, timeout optimization
   - Impact: 5-20% of total time

4. Parsing/Processing
   - Solution: Efficient algorithms, caching, lazy evaluation
   - Impact: 5-15% of total time

5. Network Latency
   - Solution: Regional deployment, CDN, connection pooling
   - Impact: Variable

Profiling Tools:
- LangSmith: Native LangChain profiling
- cProfile: Python profiling
- memory_profiler: Memory tracking
- py-spy: Sampling profiler
- OpenTelemetry: Distributed tracing
- Custom: Roll your own

Best Practices:
1. ✓ Profile early and often
2. ✓ Focus on hot paths (80/20 rule)
3. ✓ Measure before optimizing
4. ✓ Track trends over time
5. ✓ Set performance budgets
6. ✓ Automate profiling in CI/CD
7. ✓ Profile production (with sampling)
8. ✓ Document optimizations
9. ✗ Don't optimize prematurely
10. ✗ Don't profile everything (overhead)

Metrics Dashboard Example:
```
=== Agent Performance Dashboard ===

Latency:
  P50: 1,250ms  ████████████████████ (Good)
  P95: 2,800ms  ████████████████████████████ (Warning)
  P99: 5,200ms  ████████████████████████████████████ (Alert)

Cost (last 24h):
  Total: $12.45
  Per request: $0.0025
  Trend: ↓ 15% (Improving)

Bottlenecks:
  1. LLM calls:    65% of time
  2. Retrieval:    20% of time
  3. Processing:   10% of time
  4. Other:         5% of time

Cache Performance:
  Hit rate:  78%  (Good)
  Saved:     $3.20 today
  Avg speedup: 850ms per hit
```

When to Profile:
✓ Development (optimize hot paths)
✓ Pre-production (validate performance)
✓ Production (monitor continuously)
✓ After changes (detect regressions)
✓ On errors (understand failures)
✓ Performance issues (debug slowness)
✗ Every single request (too much overhead)

ROI Analysis:
- Optimization time: 1-2 days
- Performance gain: 30-50% faster
- Cost reduction: 20-40%
- User satisfaction: +15-25%
- Engineering efficiency: +30%
- Total ROI: Very high
""")
    
    print("\n" + "=" * 80)
    print("Pattern 096 (Performance Profiling) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_performance_profiling()
