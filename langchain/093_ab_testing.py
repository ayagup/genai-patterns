"""
Pattern 093: A/B Testing Pattern

Description:
    A/B Testing (also called split testing) compares two or more variants of an agent,
    prompt, model, or configuration to determine which performs better. This pattern is
    essential for data-driven optimization of AI systems, enabling evidence-based decisions
    about changes rather than relying on intuition. A/B testing helps identify improvements,
    validate hypotheses, and continuously optimize agent performance in production.

    A/B testing enables:
    - Objective comparison of variants
    - Statistical significance validation
    - Risk mitigation through gradual rollout
    - Continuous improvement culture
    - Data-driven decision making
    - ROI measurement for changes

Components:
    1. Variant Definition
       - Control (baseline/current version)
       - Treatment (new version/s)
       - Configuration differences
       - Hypothesis statement
       - Success metrics

    2. Traffic Splitting
       - Random assignment
       - Consistent user experience
       - Traffic allocation ratios
       - Segment-based splitting
       - Gradual rollout

    3. Metrics Collection
       - Primary metrics (key objectives)
       - Secondary metrics (side effects)
       - Guardrail metrics (safety)
       - Performance metrics (cost, latency)
       - User satisfaction metrics

    4. Statistical Analysis
       - Sample size calculation
       - Significance testing
       - Confidence intervals
       - Effect size measurement
       - Winner determination

Use Cases:
    1. Prompt Optimization
       - Compare prompt variations
       - Test different instructions
       - Optimize few-shot examples
       - Refine system messages
       - Improve output formatting

    2. Model Selection
       - Compare different models (GPT-4 vs GPT-3.5)
       - Test model parameters (temperature, etc.)
       - Evaluate fine-tuned models
       - Assess cost/performance trade-offs
       - Validate model upgrades

    3. Feature Testing
       - New capabilities evaluation
       - Tool addition impact
       - Memory strategy comparison
       - Retrieval method testing
       - Chain architecture optimization

    4. Configuration Tuning
       - Response length optimization
       - Timeout adjustment
       - Retry logic tuning
       - Cache strategy comparison
       - Rate limit optimization

    5. User Experience
       - Response style testing
       - Persona variations
       - Output format preferences
       - Interaction flow optimization
       - Error message improvements

LangChain Implementation:
    LangChain supports A/B testing through:
    - Multiple chain configurations
    - Custom callbacks for tracking
    - LangSmith for experiment tracking
    - Flexible chain composition
    - Variant switching logic

Key Features:
    1. Experiment Management
       - Multiple variants support
       - Traffic allocation control
       - Consistent assignment
       - Experiment lifecycle
       - Result tracking

    2. Metrics Framework
       - Flexible metric definitions
       - Automatic collection
       - Real-time monitoring
       - Historical comparison
       - Statistical analysis

    3. Safety Mechanisms
       - Guardrail metrics
       - Auto-stop on degradation
       - Gradual rollout
       - Easy rollback
       - Error rate monitoring

    4. Analysis Tools
       - Statistical significance
       - Confidence intervals
       - Visualization support
       - Winner determination
       - ROI calculation

Best Practices:
    1. Experiment Design
       - Clear hypothesis
       - Single variable change
       - Appropriate metrics
       - Sufficient sample size
       - Realistic duration

    2. Traffic Allocation
       - Start small (5-10% treatment)
       - Increase gradually
       - Monitor continuously
       - Keep control stable
       - Random assignment

    3. Metric Selection
       - Primary metric (main goal)
       - Secondary metrics (side effects)
       - Guardrails (safety limits)
       - Leading indicators
       - Lagging indicators

    4. Analysis Rigor
       - Wait for significance
       - Check for novelty effects
       - Consider practical significance
       - Review all metrics
       - Document decisions

Trade-offs:
    Advantages:
    - Objective decision making
    - Risk reduction
    - Continuous optimization
    - Evidence-based improvements
    - Stakeholder confidence
    - Measurable impact

    Disadvantages:
    - Requires traffic volume
    - Time to statistical significance
    - Complexity overhead
    - May slow innovation
    - Requires discipline
    - Can be misinterpreted

Production Considerations:
    1. Sample Size
       - Calculate minimum required
       - Consider baseline metrics
       - Factor in expected lift
       - Account for variance
       - Plan for duration

    2. Experiment Duration
       - Minimum: 1-2 weeks
       - Cover business cycles
       - Avoid holidays/events
       - Monitor stability
       - Check for time effects

    3. Monitoring
       - Real-time dashboards
       - Alert on anomalies
       - Track guardrails
       - Monitor costs
       - Review regularly

    4. Decision Criteria
       - Statistical significance (p < 0.05)
       - Practical significance (effect size)
       - Cost implications
       - All metrics green
       - Business alignment

    5. Rollout Strategy
       - Start with 5-10%
       - Monitor for issues
       - Increase to 50%
       - Full rollout if successful
       - Easy rollback plan
"""

import os
import random
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import math
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class VariantStatus(Enum):
    """Variant status in experiment"""
    ACTIVE = "active"
    PAUSED = "paused"
    WINNER = "winner"
    STOPPED = "stopped"


@dataclass
class Variant:
    """Represents an experiment variant"""
    id: str
    name: str
    description: str
    traffic_allocation: float  # 0.0 to 1.0
    config: Dict[str, Any]
    is_control: bool = False
    status: VariantStatus = VariantStatus.ACTIVE


@dataclass
class Metric:
    """Represents a metric to track"""
    name: str
    description: str
    is_primary: bool = False
    is_guardrail: bool = False
    higher_is_better: bool = True
    threshold: Optional[float] = None  # For guardrail metrics


@dataclass
class VariantMetrics:
    """Metrics for a variant"""
    variant_id: str
    sample_size: int
    metrics: Dict[str, List[float]]  # metric_name -> values
    
    def get_mean(self, metric_name: str) -> float:
        """Calculate mean for a metric"""
        values = self.metrics.get(metric_name, [])
        return sum(values) / len(values) if values else 0.0
    
    def get_std(self, metric_name: str) -> float:
        """Calculate standard deviation for a metric"""
        values = self.metrics.get(metric_name, [])
        if len(values) < 2:
            return 0.0
        
        mean = self.get_mean(metric_name)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)


@dataclass
class ABTestResult:
    """Result of A/B test analysis"""
    experiment_id: str
    control_variant: str
    treatment_variant: str
    metric_name: str
    control_mean: float
    treatment_mean: float
    lift: float  # Percentage improvement
    p_value: float
    is_significant: bool
    confidence: float
    winner: Optional[str] = None


class ABExperiment:
    """
    A/B test experiment manager.
    
    Manages variants, traffic allocation, and metrics collection.
    """
    
    def __init__(
        self,
        experiment_id: str,
        name: str,
        description: str,
        metrics: List[Metric]
    ):
        """
        Initialize A/B experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            name: Experiment name
            description: Experiment description
            metrics: Metrics to track
        """
        self.experiment_id = experiment_id
        self.name = name
        self.description = description
        self.metrics = {m.name: m for m in metrics}
        self.variants: List[Variant] = []
        self.variant_metrics: Dict[str, VariantMetrics] = {}
        self.started_at: Optional[datetime] = None
    
    def add_variant(self, variant: Variant):
        """
        Add a variant to the experiment.
        
        Args:
            variant: Variant to add
        """
        self.variants.append(variant)
        self.variant_metrics[variant.id] = VariantMetrics(
            variant_id=variant.id,
            sample_size=0,
            metrics={m: [] for m in self.metrics.keys()}
        )
    
    def select_variant(self, user_id: Optional[str] = None) -> Variant:
        """
        Select variant for a request.
        
        Args:
            user_id: Optional user ID for consistent assignment
            
        Returns:
            Selected variant
        """
        # Use user_id for consistent assignment if provided
        if user_id:
            # Simple hash-based assignment for consistency
            hash_value = hash(user_id) % 1000 / 1000.0
        else:
            hash_value = random.random()
        
        # Select variant based on traffic allocation
        cumulative = 0.0
        for variant in self.variants:
            if variant.status != VariantStatus.ACTIVE:
                continue
            
            cumulative += variant.traffic_allocation
            if hash_value <= cumulative:
                return variant
        
        # Default to first active variant
        return next(v for v in self.variants if v.status == VariantStatus.ACTIVE)
    
    def record_metric(
        self,
        variant_id: str,
        metric_name: str,
        value: float
    ):
        """
        Record a metric value for a variant.
        
        Args:
            variant_id: Variant identifier
            metric_name: Metric name
            value: Metric value
        """
        if variant_id in self.variant_metrics:
            metrics = self.variant_metrics[variant_id]
            if metric_name in metrics.metrics:
                metrics.metrics[metric_name].append(value)
                metrics.sample_size += 1
    
    def get_variant_metrics(self, variant_id: str) -> VariantMetrics:
        """Get metrics for a variant"""
        return self.variant_metrics.get(variant_id)
    
    def check_guardrails(self) -> Dict[str, bool]:
        """
        Check if guardrail metrics are within thresholds.
        
        Returns:
            Dictionary of guardrail checks
        """
        results = {}
        
        for metric_name, metric in self.metrics.items():
            if not metric.is_guardrail:
                continue
            
            for variant in self.variants:
                if variant.is_control:
                    continue
                
                variant_metrics = self.variant_metrics[variant.id]
                mean_value = variant_metrics.get_mean(metric_name)
                
                if metric.threshold is not None:
                    if metric.higher_is_better:
                        passed = mean_value >= metric.threshold
                    else:
                        passed = mean_value <= metric.threshold
                    
                    results[f"{variant.id}_{metric_name}"] = passed
        
        return results


class ABTestAnalyzer:
    """
    Analyzes A/B test results and determines winners.
    
    Performs statistical tests and calculates significance.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize analyzer.
        
        Args:
            confidence_level: Confidence level for significance (default 95%)
        """
        self.confidence_level = confidence_level
        self.z_score = 1.96  # For 95% confidence
    
    def calculate_t_statistic(
        self,
        mean1: float,
        std1: float,
        n1: int,
        mean2: float,
        std2: float,
        n2: int
    ) -> float:
        """
        Calculate t-statistic for two samples.
        
        Returns:
            T-statistic
        """
        if n1 < 2 or n2 < 2:
            return 0.0
        
        pooled_std = math.sqrt(
            ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        )
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / (pooled_std * math.sqrt(1/n1 + 1/n2))
    
    def calculate_p_value(self, t_stat: float, df: int) -> float:
        """
        Calculate p-value from t-statistic (simplified).
        
        This is a simplified approximation for demonstration.
        Production code should use scipy.stats.
        
        Returns:
            P-value
        """
        # Simplified: convert to z-score approximation
        # For large samples, t-distribution ≈ normal distribution
        if df < 30:
            # Conservative estimate for small samples
            return 0.05 if abs(t_stat) > 2.0 else 0.5
        else:
            # Normal approximation for large samples
            return 0.01 if abs(t_stat) > 2.58 else 0.05 if abs(t_stat) > 1.96 else 0.5
    
    def compare_variants(
        self,
        experiment: ABExperiment,
        control_id: str,
        treatment_id: str,
        metric_name: str
    ) -> ABTestResult:
        """
        Compare two variants on a metric.
        
        Args:
            experiment: A/B experiment
            control_id: Control variant ID
            treatment_id: Treatment variant ID
            metric_name: Metric to compare
            
        Returns:
            Test result
        """
        control_metrics = experiment.get_variant_metrics(control_id)
        treatment_metrics = experiment.get_variant_metrics(treatment_id)
        
        control_mean = control_metrics.get_mean(metric_name)
        control_std = control_metrics.get_std(metric_name)
        control_n = control_metrics.sample_size
        
        treatment_mean = treatment_metrics.get_mean(metric_name)
        treatment_std = treatment_metrics.get_std(metric_name)
        treatment_n = treatment_metrics.sample_size
        
        # Calculate lift
        if control_mean != 0:
            lift = ((treatment_mean - control_mean) / control_mean) * 100
        else:
            lift = 0.0
        
        # Calculate t-statistic
        t_stat = self.calculate_t_statistic(
            treatment_mean, treatment_std, treatment_n,
            control_mean, control_std, control_n
        )
        
        # Calculate p-value
        df = control_n + treatment_n - 2
        p_value = self.calculate_p_value(t_stat, df)
        
        # Determine significance
        is_significant = p_value < (1 - self.confidence_level)
        
        # Determine winner
        metric = experiment.metrics[metric_name]
        if is_significant:
            if metric.higher_is_better:
                winner = treatment_id if treatment_mean > control_mean else control_id
            else:
                winner = treatment_id if treatment_mean < control_mean else control_id
        else:
            winner = None
        
        return ABTestResult(
            experiment_id=experiment.experiment_id,
            control_variant=control_id,
            treatment_variant=treatment_id,
            metric_name=metric_name,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            lift=lift,
            p_value=p_value,
            is_significant=is_significant,
            confidence=self.confidence_level,
            winner=winner
        )


def demonstrate_ab_testing():
    """Demonstrate A/B testing pattern"""
    print("=" * 80)
    print("A/B TESTING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Create A/B Experiment
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Setting Up A/B Experiment")
    print("=" * 80)
    
    # Define metrics
    metrics = [
        Metric(
            name="response_quality",
            description="Quality of response (1-5 scale)",
            is_primary=True,
            higher_is_better=True
        ),
        Metric(
            name="response_length",
            description="Response length in words",
            is_primary=False,
            higher_is_better=False
        ),
        Metric(
            name="error_rate",
            description="Error rate percentage",
            is_primary=False,
            is_guardrail=True,
            higher_is_better=False,
            threshold=5.0  # Max 5% error rate
        ),
        Metric(
            name="latency",
            description="Response time in seconds",
            is_primary=False,
            is_guardrail=True,
            higher_is_better=False,
            threshold=3.0  # Max 3 seconds
        )
    ]
    
    experiment = ABExperiment(
        experiment_id="exp_001",
        name="Prompt Optimization Test",
        description="Testing new prompt format vs current",
        metrics=metrics
    )
    
    # Add control variant
    control = Variant(
        id="control",
        name="Current Prompt",
        description="Existing prompt format",
        traffic_allocation=0.5,
        config={"prompt_version": "v1"},
        is_control=True
    )
    experiment.add_variant(control)
    
    # Add treatment variant
    treatment = Variant(
        id="treatment",
        name="New Prompt",
        description="Optimized prompt format",
        traffic_allocation=0.5,
        config={"prompt_version": "v2"}
    )
    experiment.add_variant(treatment)
    
    print(f"\nExperiment Created: {experiment.name}")
    print(f"ID: {experiment.experiment_id}")
    print(f"\nVariants:")
    for variant in experiment.variants:
        control_label = " (CONTROL)" if variant.is_control else ""
        print(f"  - {variant.name}{control_label}")
        print(f"    ID: {variant.id}")
        print(f"    Traffic: {variant.traffic_allocation * 100}%")
        print(f"    Config: {variant.config}")
    
    print(f"\nMetrics:")
    for metric in metrics:
        metric_type = []
        if metric.is_primary:
            metric_type.append("PRIMARY")
        if metric.is_guardrail:
            metric_type.append("GUARDRAIL")
        if not metric_type:
            metric_type.append("SECONDARY")
        
        print(f"  - {metric.name} ({', '.join(metric_type)})")
        print(f"    {metric.description}")
    
    # Example 2: Simulate Traffic and Collect Metrics
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Simulating Traffic and Collecting Metrics")
    print("=" * 80)
    
    print("\nSimulating 100 requests...\n")
    
    random.seed(42)  # For reproducibility
    
    for i in range(100):
        # Select variant
        user_id = f"user_{i % 20}"  # Simulate 20 users
        variant = experiment.select_variant(user_id)
        
        # Simulate metrics (treatment has 10% improvement)
        if variant.id == "control":
            quality = random.gauss(3.5, 0.5)  # Mean 3.5, std 0.5
            length = random.gauss(100, 20)
            error_rate = random.uniform(0, 3)
            latency = random.gauss(2.0, 0.3)
        else:  # treatment
            quality = random.gauss(3.85, 0.5)  # 10% higher
            length = random.gauss(95, 20)  # 5% shorter
            error_rate = random.uniform(0, 2.5)  # Lower error rate
            latency = random.gauss(1.9, 0.3)  # Slightly faster
        
        # Clamp values to reasonable ranges
        quality = max(1, min(5, quality))
        error_rate = max(0, min(10, error_rate))
        
        # Record metrics
        experiment.record_metric(variant.id, "response_quality", quality)
        experiment.record_metric(variant.id, "response_length", length)
        experiment.record_metric(variant.id, "error_rate", error_rate)
        experiment.record_metric(variant.id, "latency", latency)
    
    print("Traffic simulation complete.\n")
    
    # Show sample sizes
    for variant in experiment.variants:
        metrics = experiment.get_variant_metrics(variant.id)
        print(f"{variant.name}: {metrics.sample_size} samples")
    
    # Example 3: View Variant Performance
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Variant Performance Summary")
    print("=" * 80)
    
    for variant in experiment.variants:
        variant_metrics = experiment.get_variant_metrics(variant.id)
        
        print(f"\n{variant.name} ({variant.id}):")
        print(f"  Sample Size: {variant_metrics.sample_size}")
        print(f"  Metrics:")
        
        for metric_name in experiment.metrics.keys():
            mean = variant_metrics.get_mean(metric_name)
            std = variant_metrics.get_std(metric_name)
            print(f"    {metric_name}: {mean:.2f} (±{std:.2f})")
    
    # Example 4: Statistical Analysis
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Statistical Analysis")
    print("=" * 80)
    
    analyzer = ABTestAnalyzer(confidence_level=0.95)
    
    print("\nComparing Control vs Treatment:\n")
    
    results = []
    for metric_name in experiment.metrics.keys():
        result = analyzer.compare_variants(
            experiment,
            control_id="control",
            treatment_id="treatment",
            metric_name=metric_name
        )
        results.append(result)
        
        significance = "✓ SIGNIFICANT" if result.is_significant else "✗ Not Significant"
        print(f"{metric_name.upper()}:")
        print(f"  Control: {result.control_mean:.2f}")
        print(f"  Treatment: {result.treatment_mean:.2f}")
        print(f"  Lift: {result.lift:+.2f}%")
        print(f"  P-value: {result.p_value:.4f}")
        print(f"  {significance}")
        if result.winner:
            print(f"  Winner: {result.winner}")
        print()
    
    # Example 5: Guardrail Checks
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Guardrail Metrics Check")
    print("=" * 80)
    
    guardrail_results = experiment.check_guardrails()
    
    print("\nGuardrail Checks:")
    all_passed = all(guardrail_results.values())
    
    for check, passed in guardrail_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check}: {status}")
    
    if all_passed:
        print("\n✓ All guardrails passed - Safe to proceed")
    else:
        print("\n✗ Some guardrails failed - Review before rollout")
    
    # Example 6: Decision Making
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Making the Decision")
    print("=" * 80)
    
    # Check primary metric
    primary_result = next(r for r in results if experiment.metrics[r.metric_name].is_primary)
    
    print("\nDecision Criteria:")
    print(f"1. Primary Metric ({primary_result.metric_name}):")
    print(f"   Lift: {primary_result.lift:+.2f}%")
    print(f"   Significant: {primary_result.is_significant}")
    
    print(f"\n2. Guardrail Metrics:")
    print(f"   All Passed: {all_passed}")
    
    print(f"\n3. Secondary Metrics:")
    for result in results:
        if not experiment.metrics[result.metric_name].is_primary:
            direction = "↑" if result.lift > 0 else "↓"
            print(f"   {result.metric_name}: {direction} {abs(result.lift):.1f}%")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("=" * 80)
    
    if primary_result.is_significant and all_passed and primary_result.lift > 0:
        print("""
✓ DEPLOY TREATMENT

Justification:
- Primary metric shows significant improvement
- All guardrails passed
- No significant degradation in secondary metrics
- Safe to rollout to 100% of traffic

Next Steps:
1. Increase treatment traffic to 100%
2. Monitor for 1 week
3. Validate sustained improvement
4. Document learnings
""")
    else:
        print("""
✗ KEEP CONTROL

Justification:
- Treatment did not show sufficient improvement
- OR guardrail metrics were violated
- OR improvement not statistically significant

Next Steps:
1. Analyze why treatment underperformed
2. Iterate on design
3. Run new experiment
4. Consider alternative approaches
""")
    
    # Example 7: Gradual Rollout Simulation
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Gradual Rollout Strategy")
    print("=" * 80)
    
    print("\nRecommended rollout plan:\n")
    
    rollout_stages = [
        {"traffic": 5, "duration": "2 days", "action": "Initial validation"},
        {"traffic": 10, "duration": "3 days", "action": "Monitor stability"},
        {"traffic": 25, "duration": "1 week", "action": "Expand cautiously"},
        {"traffic": 50, "duration": "1 week", "action": "Half traffic"},
        {"traffic": 100, "duration": "Ongoing", "action": "Full rollout"}
    ]
    
    for stage in rollout_stages:
        print(f"Stage {stage['traffic']}%:")
        print(f"  Duration: {stage['duration']}")
        print(f"  Action: {stage['action']}")
        print(f"  If issues detected: Rollback to previous stage")
        print()
    
    # Summary
    print("\n" + "=" * 80)
    print("A/B TESTING SUMMARY")
    print("=" * 80)
    print("""
A/B Testing Benefits:
1. Objective Decision Making: Data over intuition
2. Risk Mitigation: Gradual rollout limits exposure
3. Continuous Improvement: Systematic optimization
4. Evidence-Based: Statistical confidence
5. Stakeholder Alignment: Clear success criteria
6. ROI Measurement: Quantifiable impact

Key Components:
1. Variants
   - Control: Current/baseline version
   - Treatment: New version to test
   - Clear differences between variants
   - Consistent user experience

2. Metrics
   - Primary: Main success metric
   - Secondary: Side effect monitoring
   - Guardrails: Safety boundaries
   - Leading & lagging indicators

3. Traffic Splitting
   - Random assignment for unbiased results
   - Consistent per-user (same user = same variant)
   - Flexible allocation (50/50, 90/10, etc.)
   - Gradual rollout capability

4. Statistical Analysis
   - Significance testing (p-value < 0.05)
   - Effect size (practical significance)
   - Confidence intervals
   - Sample size validation

Best Practices:
1. Experiment Design
   - Clear hypothesis statement
   - Single variable change
   - Measurable success criteria
   - Appropriate sample size
   - Realistic timeline

2. Metric Selection
   - One primary metric (main goal)
   - Multiple secondary metrics
   - Guardrail metrics (boundaries)
   - Balance short/long term
   - Track user sentiment

3. Running Experiments
   - Minimum 1-2 weeks duration
   - Wait for statistical significance
   - Monitor guardrails continuously
   - Check for novelty effects
   - Consider seasonality

4. Making Decisions
   - Require statistical significance (p < 0.05)
   - Check practical significance (meaningful lift)
   - Validate all guardrails passed
   - Review all secondary metrics
   - Document reasoning

5. Rollout Strategy
   - Start small (5-10% traffic)
   - Increase gradually
   - Monitor at each stage
   - Easy rollback mechanism
   - Full rollout only after validation

Common Pitfalls to Avoid:
1. Stopping too early (not enough samples)
2. Multiple testing without correction
3. Ignoring guardrail metrics
4. Changing experiment mid-flight
5. Cherry-picking metrics
6. Insufficient traffic volume
7. Not considering novelty effects
8. Forgetting about opportunity cost

When to Use A/B Testing:
✓ Production systems with traffic
✓ Uncertain about impact
✓ Multiple competing approaches
✓ Optimization opportunities
✓ Risk-averse changes
✓ Stakeholder buy-in needed
✗ Low traffic applications
✗ Obviously correct changes
✗ Emergency fixes
✗ Insufficient time

Sample Size Guidelines:
- Minimum 100 samples per variant
- More for smaller expected differences
- Calculator: n = (Z * σ / E)²
  where Z = z-score (1.96 for 95% confidence)
        σ = standard deviation
        E = margin of error

Statistical Significance:
- Standard: p-value < 0.05 (95% confidence)
- Conservative: p-value < 0.01 (99% confidence)
- Always report effect size, not just p-value
- Consider practical significance

Production Tips:
- Integrate with feature flags
- Real-time monitoring dashboards
- Automated guardrail alerts
- Easy rollback capability
- Experiment registry/documentation
- Post-experiment analysis
- Share learnings across team
- Iterate based on results
""")
    
    print("\n" + "=" * 80)
    print("Pattern 093 (A/B Testing) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_ab_testing()
