"""
Pattern 154: Canary Deployment

Description:
    Implements gradual rollout of new agent versions by routing a small
    percentage of traffic to the canary version, monitoring metrics, and
    automatically rolling back if issues are detected.

Components:
    - Traffic Splitter: Routes requests between versions
    - Health Monitor: Tracks canary metrics
    - Rollback Controller: Automatic rollback on issues
    - Metrics Collector: Performance and error tracking
    - Gradual Ramp: Increases traffic percentage over time

Use Cases:
    - Safe production deployments
    - Gradual feature rollout
    - Risk mitigation for changes
    - A/B testing with safety
    - Automated deployment validation

Benefits:
    - Minimal blast radius
    - Automatic rollback
    - Real user validation
    - Gradual confidence building
    - Production safety

Trade-offs:
    - Deployment complexity
    - Monitoring overhead
    - Slower full rollout
    - Dual version maintenance
    - Metrics interpretation

LangChain Implementation:
    Uses traffic routing with health monitoring and automatic rollback.
"""

import os
import time
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class DeploymentStatus(Enum):
    """Status of canary deployment"""
    STARTING = "starting"
    HEALTHY = "healthy"
    WARNING = "warning"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    PROMOTED = "promoted"


@dataclass
class CanaryMetrics:
    """Metrics for canary and production versions"""
    # Request counts
    production_requests: int = 0
    canary_requests: int = 0
    
    # Success rates
    production_successes: int = 0
    canary_successes: int = 0
    production_errors: int = 0
    canary_errors: int = 0
    
    # Latency
    production_latency_sum: float = 0.0
    canary_latency_sum: float = 0.0
    
    def get_production_error_rate(self) -> float:
        """Calculate production error rate"""
        if self.production_requests == 0:
            return 0.0
        return self.production_errors / self.production_requests
    
    def get_canary_error_rate(self) -> float:
        """Calculate canary error rate"""
        if self.canary_requests == 0:
            return 0.0
        return self.canary_errors / self.canary_requests
    
    def get_production_avg_latency(self) -> float:
        """Calculate average production latency"""
        if self.production_requests == 0:
            return 0.0
        return self.production_latency_sum / self.production_requests
    
    def get_canary_avg_latency(self) -> float:
        """Calculate average canary latency"""
        if self.canary_requests == 0:
            return 0.0
        return self.canary_latency_sum / self.canary_requests


@dataclass
class CanaryConfig:
    """Configuration for canary deployment"""
    initial_traffic_percent: float = 5.0  # Start with 5%
    max_traffic_percent: float = 50.0  # Max before promotion
    ramp_step: float = 5.0  # Increase by 5% each step
    ramp_interval_seconds: float = 60.0  # Wait 60s between steps
    
    # Health thresholds
    max_error_rate_increase: float = 0.05  # Max 5% error rate increase
    max_latency_increase: float = 0.20  # Max 20% latency increase
    min_requests_for_decision: int = 10  # Min requests before decisions
    
    # Rollback settings
    auto_rollback: bool = True
    rollback_on_error_spike: bool = True


class CanaryDeployment:
    """
    Manages canary deployment with gradual traffic routing and
    automatic rollback on issues.
    """
    
    def __init__(self, config: Optional[CanaryConfig] = None):
        """Initialize canary deployment"""
        self.config = config or CanaryConfig()
        
        # Production version (stable)
        self.production_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Canary version (new version being tested)
        self.canary_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.8  # Different config for canary
        )
        
        # State
        self.current_traffic_percent = self.config.initial_traffic_percent
        self.status = DeploymentStatus.STARTING
        self.metrics = CanaryMetrics()
        self.deployment_start_time = datetime.now()
        self.last_ramp_time = time.time()
        
        # Prompts
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            ("user", "{query}")
        ])
    
    def route_request(self, query: str) -> Dict[str, Any]:
        """
        Route request to either production or canary based on traffic split
        
        Args:
            query: User query
            
        Returns:
            Response with version and metrics
        """
        # Decide routing
        use_canary = random.random() * 100 < self.current_traffic_percent
        
        start_time = time.time()
        
        try:
            if use_canary:
                # Route to canary
                chain = self.prompt | self.canary_llm | StrOutputParser()
                result = chain.invoke({"query": query})
                version = "canary"
                
                # Update canary metrics
                self.metrics.canary_requests += 1
                self.metrics.canary_successes += 1
                latency = time.time() - start_time
                self.metrics.canary_latency_sum += latency
                
            else:
                # Route to production
                chain = self.prompt | self.production_llm | StrOutputParser()
                result = chain.invoke({"query": query})
                version = "production"
                
                # Update production metrics
                self.metrics.production_requests += 1
                self.metrics.production_successes += 1
                latency = time.time() - start_time
                self.metrics.production_latency_sum += latency
            
            # Check health after each request
            self._check_health()
            
            # Consider ramping up traffic
            self._consider_ramp_up()
            
            return {
                "success": True,
                "result": result,
                "version": version,
                "latency": latency,
                "canary_traffic_percent": self.current_traffic_percent,
                "status": self.status.value
            }
            
        except Exception as e:
            # Track error
            latency = time.time() - start_time
            
            if use_canary:
                self.metrics.canary_requests += 1
                self.metrics.canary_errors += 1
                self.metrics.canary_latency_sum += latency
                version = "canary"
            else:
                self.metrics.production_requests += 1
                self.metrics.production_errors += 1
                self.metrics.production_latency_sum += latency
                version = "production"
            
            # Check if we should rollback
            self._check_health()
            
            return {
                "success": False,
                "error": str(e),
                "version": version,
                "latency": latency,
                "canary_traffic_percent": self.current_traffic_percent,
                "status": self.status.value
            }
    
    def _check_health(self):
        """Check canary health and trigger rollback if needed"""
        if self.status == DeploymentStatus.ROLLED_BACK:
            return  # Already rolled back
        
        if self.metrics.canary_requests < self.config.min_requests_for_decision:
            return  # Not enough data yet
        
        # Calculate metrics
        canary_error_rate = self.metrics.get_canary_error_rate()
        prod_error_rate = self.metrics.get_production_error_rate()
        
        canary_latency = self.metrics.get_canary_avg_latency()
        prod_latency = self.metrics.get_production_avg_latency()
        
        # Check error rate
        if prod_error_rate > 0:
            error_rate_increase = (canary_error_rate - prod_error_rate) / prod_error_rate
        else:
            error_rate_increase = canary_error_rate
        
        # Check latency
        if prod_latency > 0:
            latency_increase = (canary_latency - prod_latency) / prod_latency
        else:
            latency_increase = 0.0
        
        # Determine health
        issues = []
        
        if error_rate_increase > self.config.max_error_rate_increase:
            issues.append(f"Error rate {error_rate_increase*100:.1f}% higher")
        
        if latency_increase > self.config.max_latency_increase:
            issues.append(f"Latency {latency_increase*100:.1f}% higher")
        
        if issues:
            if self.config.auto_rollback:
                print(f"\n🚨 ROLLBACK TRIGGERED: {', '.join(issues)}")
                self._rollback()
            else:
                self.status = DeploymentStatus.WARNING
                print(f"\n⚠️  WARNING: {', '.join(issues)}")
        elif self.status != DeploymentStatus.PROMOTED:
            self.status = DeploymentStatus.HEALTHY
    
    def _consider_ramp_up(self):
        """Consider increasing traffic to canary"""
        if self.status != DeploymentStatus.HEALTHY:
            return  # Only ramp if healthy
        
        if self.current_traffic_percent >= self.config.max_traffic_percent:
            return  # Already at max
        
        # Check if enough time has passed
        time_since_last_ramp = time.time() - self.last_ramp_time
        if time_since_last_ramp < self.config.ramp_interval_seconds:
            return
        
        # Ramp up
        old_percent = self.current_traffic_percent
        self.current_traffic_percent = min(
            self.current_traffic_percent + self.config.ramp_step,
            self.config.max_traffic_percent
        )
        self.last_ramp_time = time.time()
        
        print(f"\n📈 RAMP UP: {old_percent:.0f}% → {self.current_traffic_percent:.0f}% traffic to canary")
        
        # Check if we should promote
        if self.current_traffic_percent >= self.config.max_traffic_percent:
            print(f"\n✅ CANARY READY FOR PROMOTION (reached {self.config.max_traffic_percent}% traffic)")
    
    def _rollback(self):
        """Rollback to production version"""
        self.status = DeploymentStatus.ROLLING_BACK
        self.current_traffic_percent = 0.0
        print(f"\n⏪ Rolling back to production version...")
        self.status = DeploymentStatus.ROLLED_BACK
        print(f"✅ Rollback complete - 100% traffic to production")
    
    def promote_canary(self):
        """Promote canary to production"""
        if self.status == DeploymentStatus.ROLLED_BACK:
            print("❌ Cannot promote - deployment was rolled back")
            return False
        
        print(f"\n🎉 PROMOTING CANARY TO PRODUCTION")
        self.current_traffic_percent = 100.0
        self.status = DeploymentStatus.PROMOTED
        return True
    
    def get_status_report(self) -> str:
        """Generate status report"""
        m = self.metrics
        
        report = [
            "\n" + "="*70,
            "CANARY DEPLOYMENT STATUS",
            "="*70,
            f"\nStatus: {self.status.value.upper()}",
            f"Canary Traffic: {self.current_traffic_percent:.1f}%",
            f"Deployment Duration: {(datetime.now() - self.deployment_start_time).total_seconds():.0f}s",
            "\nProduction Metrics:",
            f"  Requests: {m.production_requests}",
            f"  Success Rate: {(m.production_successes/m.production_requests*100 if m.production_requests > 0 else 0):.1f}%",
            f"  Error Rate: {m.get_production_error_rate()*100:.1f}%",
            f"  Avg Latency: {m.get_production_avg_latency():.3f}s",
            "\nCanary Metrics:",
            f"  Requests: {m.canary_requests}",
            f"  Success Rate: {(m.canary_successes/m.canary_requests*100 if m.canary_requests > 0 else 0):.1f}%",
            f"  Error Rate: {m.get_canary_error_rate()*100:.1f}%",
            f"  Avg Latency: {m.get_canary_avg_latency():.3f}s",
        ]
        
        # Comparison
        if m.canary_requests >= self.config.min_requests_for_decision:
            error_diff = (m.get_canary_error_rate() - m.get_production_error_rate()) * 100
            latency_diff_pct = ((m.get_canary_avg_latency() - m.get_production_avg_latency()) 
                               / m.get_production_avg_latency() * 100) if m.get_production_avg_latency() > 0 else 0
            
            report.extend([
                "\nComparison (Canary vs Production):",
                f"  Error Rate Diff: {error_diff:+.1f}%",
                f"  Latency Diff: {latency_diff_pct:+.1f}%"
            ])
        
        report.append("="*70)
        return "\n".join(report)


def demonstrate_canary_deployment():
    """Demonstrate canary deployment pattern"""
    print("="*70)
    print("Pattern 154: Canary Deployment")
    print("="*70)
    
    # Configure canary deployment
    config = CanaryConfig(
        initial_traffic_percent=10.0,
        max_traffic_percent=50.0,
        ramp_step=10.0,
        ramp_interval_seconds=5.0,  # Short for demo
        max_error_rate_increase=0.10,
        max_latency_increase=0.30,
        min_requests_for_decision=5,
        auto_rollback=True
    )
    
    deployment = CanaryDeployment(config)
    
    print(f"\n🚀 Starting canary deployment")
    print(f"   Initial traffic: {config.initial_traffic_percent}%")
    print(f"   Ramp step: {config.ramp_step}%")
    print(f"   Target: {config.max_traffic_percent}%")
    
    # Simulate requests
    test_queries = [
        "What is Python?",
        "Explain machine learning",
        "How does AI work?",
        "What is cloud computing?",
        "Tell me about databases",
    ] * 10  # 50 total requests
    
    print(f"\n{'='*70}")
    print("SIMULATING PRODUCTION TRAFFIC")
    print(f"{'='*70}")
    
    results = []
    for i, query in enumerate(test_queries, 1):
        result = deployment.route_request(query)
        results.append(result)
        
        if i % 10 == 0:
            print(f"\n📊 After {i} requests:")
            print(deployment.get_status_report())
            
            # Small delay to allow ramping
            time.sleep(1)
    
    # Final status
    print(f"\n{'='*70}")
    print("FINAL STATUS")
    print(f"{'='*70}")
    print(deployment.get_status_report())
    
    # Show routing distribution
    production_count = sum(1 for r in results if r.get('version') == 'production')
    canary_count = sum(1 for r in results if r.get('version') == 'canary')
    
    print(f"\n📊 Traffic Distribution:")
    print(f"   Production: {production_count} requests ({production_count/len(results)*100:.1f}%)")
    print(f"   Canary: {canary_count} requests ({canary_count/len(results)*100:.1f}%)")
    
    # Promotion decision
    if deployment.status == DeploymentStatus.HEALTHY:
        print(f"\n✅ Canary is healthy and ready for promotion!")
        deployment.promote_canary()
    elif deployment.status == DeploymentStatus.ROLLED_BACK:
        print(f"\n❌ Canary was rolled back due to issues")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Canary Deployment Pattern provides:

1. Gradual Rollout:
   - Start with small traffic percentage
   - Incrementally increase over time
   - Monitor at each step
   - Automatic promotion when ready

2. Safety Features:
   - Continuous health monitoring
   - Automatic rollback on issues
   - Configurable thresholds
   - Minimal blast radius

3. Metrics Tracking:
   - Error rates comparison
   - Latency comparison
   - Success rates
   - Traffic distribution

4. Deployment Control:
   - Traffic percentage control
   - Ramp-up automation
   - Manual promotion option
   - Emergency rollback

5. Production Safety:
   - Real user validation
   - Low-risk testing
   - Quick issue detection
   - Automatic recovery

This pattern is essential for safely deploying changes to
production AI systems with minimal risk and automatic safeguards.
    """)


if __name__ == "__main__":
    demonstrate_canary_deployment()
