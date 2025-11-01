"""
Pattern 088: Token Budget Management

Description:
    The Token Budget Management pattern monitors and controls token usage across LLM
    interactions to optimize costs, prevent budget overruns, and ensure efficient
    resource utilization. This pattern is critical for production deployments where
    token costs can escalate quickly, especially with high-volume applications or
    complex multi-step workflows.

    Token budget management involves:
    - Real-time token usage tracking
    - Cost estimation and forecasting
    - Budget enforcement and alerts
    - Optimization strategies for token efficiency
    - Usage analytics and reporting

Components:
    1. Token Counting
       - Input token counting (prompt tokens)
       - Output token counting (completion tokens)
       - Total token tracking per request
       - Cumulative usage tracking
       - Model-specific token limits

    2. Cost Calculation
       - Model pricing tables (per-token costs)
       - Request cost estimation
       - Cumulative cost tracking
       - Budget vs actual comparison
       - Cost forecasting

    3. Budget Enforcement
       - Hard limits (stop when exceeded)
       - Soft limits (warnings when approaching)
       - Rate-based budgets (tokens per hour/day)
       - User-specific quotas
       - Priority-based allocation

    4. Optimization Strategies
       - Prompt compression
       - Response length limiting
       - Model selection (cost vs quality)
       - Caching frequently used responses
       - Batching requests

Use Cases:
    1. Cost Control
       - Prevent unexpected charges
       - Stay within allocated budgets
       - Cost attribution per user/project
       - Predictable monthly spending
       - Resource optimization

    2. Multi-Tenant Applications
       - Per-user token quotas
       - Fair resource allocation
       - Usage-based billing
       - Tier-based limits (free/premium)
       - Abuse prevention

    3. Development and Testing
       - Sandbox environments with limits
       - Cost-effective experimentation
       - Testing budget allocation
       - Development vs production budgets
       - CI/CD pipeline limits

    4. High-Volume Applications
       - Aggregate usage tracking
       - Peak usage management
       - Resource planning
       - Scaling decisions
       - Cost optimization at scale

    5. Enterprise Deployments
       - Department-level budgets
       - Project-based allocation
       - Chargeback mechanisms
       - Compliance and auditing
       - Executive reporting

LangChain Implementation:
    LangChain supports token budget management through:
    - Callback handlers for token tracking
    - get_openai_callback for cost monitoring
    - Custom token counters
    - Usage analytics integration
    - Budget enforcement wrappers

Key Features:
    1. Real-Time Tracking
       - Immediate token counting
       - Running total updates
       - Cost calculation on-the-fly
       - Budget threshold monitoring
       - Alert triggering

    2. Multi-Model Support
       - Different pricing for different models
       - Model-specific token limits
       - Automatic model selection
       - Cost-aware routing
       - Fallback to cheaper models

    3. Granular Control
       - Request-level limits
       - Session-level budgets
       - User-level quotas
       - Project-level allocations
       - Global organization limits

    4. Analytics and Reporting
       - Usage trends over time
       - Cost breakdown by model
       - Per-user consumption
       - Peak usage identification
       - Cost forecasting

Best Practices:
    1. Set Appropriate Budgets
       - Based on historical data
       - Include buffer for peaks
       - Align with business value
       - Regular review and adjustment
       - Clear escalation procedures

    2. Implement Progressive Enforcement
       - Soft warnings at 70-80%
       - Stricter controls at 90%
       - Hard stops at 100%
       - Grace period for critical operations
       - Override mechanisms for emergencies

    3. Optimize Token Usage
       - Compress verbose prompts
       - Limit response lengths
       - Use cheaper models when sufficient
       - Cache common queries
       - Batch similar requests

    4. Monitor and Alert
       - Real-time dashboards
       - Automated alerts
       - Anomaly detection
       - Trend analysis
       - Regular reporting

Trade-offs:
    Advantages:
    - Predictable costs
    - Prevents budget overruns
    - Enables cost optimization
    - Fair resource allocation
    - Usage visibility and control
    - Informed scaling decisions

    Disadvantages:
    - Implementation complexity
    - Tracking overhead (minimal)
    - May limit functionality at budget
    - Requires ongoing management
    - False positives in alerting
    - User friction from limits

Production Considerations:
    1. Accuracy
       - Precise token counting
       - Up-to-date pricing information
       - Correct model identification
       - Cache hit accounting
       - Network overhead handling

    2. Performance
       - Minimal tracking overhead
       - Efficient storage of usage data
       - Fast budget checks
       - Optimized queries for analytics
       - Scalable architecture

    3. User Experience
       - Clear budget status visibility
       - Helpful error messages
       - Grace period handling
       - Budget increase requests
       - Usage optimization tips

    4. Maintenance
       - Regular pricing updates
       - Budget policy reviews
       - Alert threshold tuning
       - Anomaly investigation
       - Capacity planning

    5. Security
       - Access control for budget data
       - Secure token counting
       - Audit trail for usage
       - Fraud detection
       - Quota manipulation prevention
"""

import os
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()


class BudgetStatus(Enum):
    """Budget status levels"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"


class ModelTier(Enum):
    """Model cost tiers"""
    CHEAP = "cheap"
    STANDARD = "standard"
    PREMIUM = "premium"


@dataclass
class ModelPricing:
    """Pricing information for a model"""
    model_name: str
    input_cost_per_1k: float  # Cost per 1000 input tokens
    output_cost_per_1k: float  # Cost per 1000 output tokens
    tier: ModelTier


@dataclass
class TokenUsage:
    """Token usage for a single request"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    model: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BudgetConfig:
    """Configuration for token budget management"""
    total_budget: float  # Total budget in USD
    warning_threshold: float = 0.8  # Warning at 80%
    critical_threshold: float = 0.9  # Critical at 90%
    enforce_hard_limit: bool = True
    default_model: str = "gpt-3.5-turbo"
    allow_model_downgrade: bool = True


class TokenBudgetManager:
    """
    Manages token budgets and costs across LLM interactions.
    
    This manager:
    1. Tracks token usage in real-time
    2. Calculates costs based on model pricing
    3. Enforces budget limits
    4. Provides usage analytics
    """
    
    # Model pricing (as of 2025)
    MODEL_PRICING = {
        "gpt-3.5-turbo": ModelPricing("gpt-3.5-turbo", 0.0005, 0.0015, ModelTier.CHEAP),
        "gpt-4": ModelPricing("gpt-4", 0.03, 0.06, ModelTier.PREMIUM),
        "gpt-4-turbo": ModelPricing("gpt-4-turbo", 0.01, 0.03, ModelTier.STANDARD),
    }
    
    def __init__(self, config: BudgetConfig):
        """
        Initialize token budget manager.
        
        Args:
            config: Budget configuration
        """
        self.config = config
        self.total_spent = 0.0
        self.usage_history: List[TokenUsage] = []
        self.requests_count = 0
        self.model_usage: Dict[str, int] = defaultdict(int)
        self.alert_callbacks: List[Callable] = []
    
    def add_alert_callback(self, callback: Callable[[BudgetStatus, Dict], None]):
        """Add callback for budget alerts"""
        self.alert_callbacks.append(callback)
    
    def _trigger_alerts(self, status: BudgetStatus, details: Dict[str, Any]):
        """Trigger all registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(status, details)
            except Exception as e:
                print(f"Alert callback error: {e}")
    
    def get_status(self) -> BudgetStatus:
        """Get current budget status"""
        usage_ratio = self.total_spent / self.config.total_budget
        
        if usage_ratio >= 1.0:
            return BudgetStatus.EXCEEDED
        elif usage_ratio >= self.config.critical_threshold:
            return BudgetStatus.CRITICAL
        elif usage_ratio >= self.config.warning_threshold:
            return BudgetStatus.WARNING
        else:
            return BudgetStatus.NORMAL
    
    def can_proceed(self, estimated_cost: float = 0.0) -> tuple[bool, str]:
        """
        Check if request can proceed within budget.
        
        Args:
            estimated_cost: Estimated cost of request
            
        Returns:
            Tuple of (can_proceed, reason)
        """
        status = self.get_status()
        
        if status == BudgetStatus.EXCEEDED:
            if self.config.enforce_hard_limit:
                return False, "Budget exceeded. Request denied."
            else:
                return True, "Budget exceeded but enforcement disabled."
        
        projected_cost = self.total_spent + estimated_cost
        if projected_cost > self.config.total_budget:
            if self.config.enforce_hard_limit:
                return False, f"Request would exceed budget. Projected: ${projected_cost:.4f}"
            
        if status == BudgetStatus.CRITICAL:
            return True, "Budget critical. Proceed with caution."
        
        return True, "OK"
    
    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str
    ) -> float:
        """
        Calculate cost for token usage.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name
            
        Returns:
            Cost in USD
        """
        pricing = self.MODEL_PRICING.get(model)
        if not pricing:
            # Default to gpt-3.5-turbo pricing
            pricing = self.MODEL_PRICING["gpt-3.5-turbo"]
        
        input_cost = (prompt_tokens / 1000) * pricing.input_cost_per_1k
        output_cost = (completion_tokens / 1000) * pricing.output_cost_per_1k
        
        return input_cost + output_cost
    
    def record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str
    ) -> TokenUsage:
        """
        Record token usage and update budget.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name
            
        Returns:
            TokenUsage object
        """
        total_tokens = prompt_tokens + completion_tokens
        cost = self.calculate_cost(prompt_tokens, completion_tokens, model)
        
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            model=model
        )
        
        # Update tracking
        self.total_spent += cost
        self.usage_history.append(usage)
        self.requests_count += 1
        self.model_usage[model] += total_tokens
        
        # Check status and trigger alerts if needed
        status = self.get_status()
        if status in [BudgetStatus.WARNING, BudgetStatus.CRITICAL, BudgetStatus.EXCEEDED]:
            self._trigger_alerts(status, {
                "total_spent": self.total_spent,
                "budget": self.config.total_budget,
                "usage_ratio": self.total_spent / self.config.total_budget,
                "latest_usage": usage
            })
        
        return usage
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget"""
        return max(0, self.config.total_budget - self.total_spent)
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary"""
        status = self.get_status()
        
        return {
            "total_budget": self.config.total_budget,
            "total_spent": self.total_spent,
            "remaining": self.get_remaining_budget(),
            "usage_percentage": (self.total_spent / self.config.total_budget) * 100,
            "status": status.value,
            "total_requests": self.requests_count,
            "total_tokens": sum(u.total_tokens for u in self.usage_history),
            "model_breakdown": dict(self.model_usage),
            "average_cost_per_request": self.total_spent / max(1, self.requests_count)
        }
    
    def suggest_cheaper_model(self, current_model: str) -> Optional[str]:
        """Suggest a cheaper model alternative"""
        if not self.config.allow_model_downgrade:
            return None
        
        current_pricing = self.MODEL_PRICING.get(current_model)
        if not current_pricing:
            return None
        
        # Find cheaper alternatives
        cheaper_models = [
            model for model, pricing in self.MODEL_PRICING.items()
            if pricing.tier.value < current_pricing.tier.value
        ]
        
        return cheaper_models[0] if cheaper_models else None
    
    def reset_budget(self, new_budget: Optional[float] = None):
        """Reset budget and usage tracking"""
        if new_budget:
            self.config.total_budget = new_budget
        
        self.total_spent = 0.0
        self.usage_history = []
        self.requests_count = 0
        self.model_usage = defaultdict(int)


class BudgetAwareLLMAgent:
    """
    LLM agent with token budget awareness.
    
    This agent:
    1. Tracks token usage automatically
    2. Enforces budget limits
    3. Optimizes token usage
    4. Provides cost visibility
    """
    
    def __init__(
        self,
        budget_manager: TokenBudgetManager,
        model: Optional[str] = None
    ):
        """
        Initialize budget-aware agent.
        
        Args:
            budget_manager: Token budget manager
            model: Model to use (defaults to config default)
        """
        self.budget_manager = budget_manager
        self.model = model or budget_manager.config.default_model
        self.llm = ChatOpenAI(model=self.model, temperature=0.7)
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate response with budget tracking.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens for response
            
        Returns:
            Dictionary with response and usage info
        """
        # Check budget before proceeding
        can_proceed, reason = self.budget_manager.can_proceed()
        if not can_proceed:
            return {
                "success": False,
                "error": reason,
                "response": None,
                "usage": None
            }
        
        # Generate with token tracking
        with get_openai_callback() as cb:
            try:
                response = self.llm.invoke(prompt)
                
                # Record usage
                usage = self.budget_manager.record_usage(
                    prompt_tokens=cb.prompt_tokens,
                    completion_tokens=cb.completion_tokens,
                    model=self.model
                )
                
                return {
                    "success": True,
                    "response": response.content,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                        "cost": usage.cost
                    },
                    "budget_status": self.budget_manager.get_status().value
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "response": None,
                    "usage": None
                }
    
    def generate_with_limit(
        self,
        prompt: str,
        max_cost: float
    ) -> Dict[str, Any]:
        """
        Generate response with per-request cost limit.
        
        Args:
            prompt: Input prompt
            max_cost: Maximum cost for this request
            
        Returns:
            Dictionary with response and usage info
        """
        # Estimate and check
        can_proceed, reason = self.budget_manager.can_proceed(max_cost)
        if not can_proceed:
            return {
                "success": False,
                "error": reason,
                "response": None
            }
        
        return self.generate(prompt)
    
    def batch_generate(
        self,
        prompts: List[str],
        max_total_cost: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple prompts with budget tracking.
        
        Args:
            prompts: List of prompts
            max_total_cost: Optional cost limit for batch
            
        Returns:
            List of results
        """
        results = []
        batch_cost = 0.0
        
        for prompt in prompts:
            # Check batch cost limit
            if max_total_cost and batch_cost >= max_total_cost:
                results.append({
                    "success": False,
                    "error": "Batch cost limit reached",
                    "response": None
                })
                continue
            
            result = self.generate(prompt)
            results.append(result)
            
            if result["success"] and result["usage"]:
                batch_cost += result["usage"]["cost"]
        
        return results


def demonstrate_token_budget_management():
    """Demonstrate token budget management patterns"""
    print("=" * 80)
    print("TOKEN BUDGET MANAGEMENT PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic Budget Tracking
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Budget Tracking")
    print("=" * 80)
    
    config = BudgetConfig(
        total_budget=1.00,  # $1.00 budget
        warning_threshold=0.7,
        critical_threshold=0.9,
        enforce_hard_limit=False  # Allow going over for demo
    )
    
    budget_manager = TokenBudgetManager(config)
    agent = BudgetAwareLLMAgent(budget_manager)
    
    print(f"\nInitial Budget: ${config.total_budget}")
    print("\nGenerating responses...\n")
    
    prompts = [
        "What is machine learning?",
        "Explain neural networks briefly.",
        "What is deep learning?"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Request {i}: {prompt}")
        result = agent.generate(prompt)
        
        if result["success"]:
            print(f"Response: {result['response'][:100]}...")
            print(f"Tokens: {result['usage']['total_tokens']}")
            print(f"Cost: ${result['usage']['cost']:.6f}")
            print(f"Status: {result['budget_status']}")
        else:
            print(f"Error: {result['error']}")
        print()
    
    # Show summary
    summary = budget_manager.get_usage_summary()
    print("\nBudget Summary:")
    print(f"  Total Spent: ${summary['total_spent']:.6f}")
    print(f"  Remaining: ${summary['remaining']:.6f}")
    print(f"  Usage: {summary['usage_percentage']:.1f}%")
    print(f"  Status: {summary['status']}")
    print(f"  Total Requests: {summary['total_requests']}")
    print(f"  Total Tokens: {summary['total_tokens']}")
    print(f"  Avg Cost/Request: ${summary['average_cost_per_request']:.6f}")
    
    # Example 2: Budget Alerts
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Budget Alerts and Thresholds")
    print("=" * 80)
    
    def budget_alert_handler(status: BudgetStatus, details: Dict):
        """Custom alert handler"""
        print(f"\n🚨 BUDGET ALERT: {status.value.upper()}")
        print(f"   Spent: ${details['total_spent']:.6f}")
        print(f"   Budget: ${details['budget']:.6f}")
        print(f"   Usage: {details['usage_ratio']*100:.1f}%")
    
    config2 = BudgetConfig(
        total_budget=0.10,  # Small budget to trigger alerts
        warning_threshold=0.5,
        critical_threshold=0.8,
        enforce_hard_limit=False
    )
    
    budget_manager2 = TokenBudgetManager(config2)
    budget_manager2.add_alert_callback(budget_alert_handler)
    agent2 = BudgetAwareLLMAgent(budget_manager2)
    
    print(f"\nBudget: ${config2.total_budget}")
    print("Generating responses to trigger alerts...\n")
    
    test_prompts = [
        "Explain quantum computing in detail.",
        "What are the applications of AI?",
        "Describe blockchain technology."
    ]
    
    for prompt in test_prompts:
        result = agent2.generate(prompt)
        if result["success"]:
            print(f"✓ Generated response (${result['usage']['cost']:.6f})")
    
    # Example 3: Cost Estimation and Limits
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Per-Request Cost Limits")
    print("=" * 80)
    
    config3 = BudgetConfig(total_budget=0.50, enforce_hard_limit=True)
    budget_manager3 = TokenBudgetManager(config3)
    agent3 = BudgetAwareLLMAgent(budget_manager3)
    
    print("\nTrying request with cost limit...\n")
    
    result = agent3.generate_with_limit(
        "Write a short poem about AI",
        max_cost=0.01
    )
    
    if result["success"]:
        print(f"Response: {result['response']}")
        print(f"Cost: ${result['usage']['cost']:.6f}")
    else:
        print(f"Request blocked: {result['error']}")
    
    # Example 4: Model Cost Comparison
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Model Cost Comparison")
    print("=" * 80)
    
    print("\nModel Pricing:")
    for model_name, pricing in TokenBudgetManager.MODEL_PRICING.items():
        print(f"\n{model_name} ({pricing.tier.value}):")
        print(f"  Input: ${pricing.input_cost_per_1k}/1K tokens")
        print(f"  Output: ${pricing.output_cost_per_1k}/1K tokens")
    
    # Calculate example costs
    print("\n\nExample cost for 1000 input + 500 output tokens:")
    for model_name in TokenBudgetManager.MODEL_PRICING.keys():
        cost = budget_manager3.calculate_cost(1000, 500, model_name)
        print(f"  {model_name}: ${cost:.6f}")
    
    # Example 5: Batch Processing with Budget
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Batch Processing with Budget Control")
    print("=" * 80)
    
    config5 = BudgetConfig(total_budget=0.50)
    budget_manager5 = TokenBudgetManager(config5)
    agent5 = BudgetAwareLLMAgent(budget_manager5, model="gpt-3.5-turbo")
    
    batch_prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Java?",
        "What is C++?",
        "What is Ruby?"
    ]
    
    print(f"\nProcessing {len(batch_prompts)} prompts with batch cost limit...")
    print(f"Batch cost limit: $0.05\n")
    
    results = agent5.batch_generate(batch_prompts, max_total_cost=0.05)
    
    successful = sum(1 for r in results if r["success"])
    total_cost = sum(r["usage"]["cost"] for r in results if r["success"] and r["usage"])
    
    print(f"\nBatch Results:")
    print(f"  Successful: {successful}/{len(batch_prompts)}")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Average Cost: ${total_cost/max(1, successful):.6f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TOKEN BUDGET MANAGEMENT SUMMARY")
    print("=" * 80)
    print("""
Token Budget Management Benefits:
1. Cost Control: Prevent unexpected charges and budget overruns
2. Visibility: Real-time tracking of token usage and costs
3. Optimization: Identify opportunities to reduce token consumption
4. Accountability: Per-user or per-project cost attribution
5. Planning: Forecast future costs based on usage trends
6. Alerts: Proactive notifications when approaching limits

Key Components Demonstrated:
1. Real-Time Tracking: Automatic token and cost monitoring
2. Budget Thresholds: Warning, critical, and exceeded states
3. Cost Calculation: Model-specific pricing and estimates
4. Enforcement: Hard and soft budget limits
5. Alerts: Custom callback handlers for notifications
6. Analytics: Usage summaries and breakdowns

Budget Management Strategies:
1. Progressive Thresholds:
   - Normal: < 80% of budget
   - Warning: 80-90% of budget
   - Critical: 90-100% of budget
   - Exceeded: > 100% of budget

2. Cost Optimization:
   - Use cheaper models when quality allows
   - Compress verbose prompts
   - Limit response lengths
   - Cache common queries
   - Batch similar requests

3. Monitoring:
   - Real-time dashboards
   - Usage trend analysis
   - Anomaly detection
   - Regular reporting
   - Forecasting

Best Practices:
1. Set realistic budgets based on historical data
2. Implement progressive enforcement (soft → hard limits)
3. Monitor usage patterns and trends
4. Optimize prompts to reduce token consumption
5. Use alerts to catch issues early
6. Regular budget reviews and adjustments
7. Cost attribution by user/project/feature
8. Document budget policies clearly

Model Selection for Cost:
- gpt-3.5-turbo: Cheapest, good for simple tasks
- gpt-4-turbo: Balanced cost/quality
- gpt-4: Premium, best quality but expensive

Production Considerations:
- Accurate token counting (model-specific)
- Up-to-date pricing information
- Efficient usage data storage
- Scalable tracking architecture
- User-friendly budget notifications
- Grace periods for critical operations
- Emergency override procedures
- Regular auditing and reconciliation

Common Use Cases:
- SaaS applications with freemium tiers
- Multi-tenant platforms
- Development/testing environments
- High-volume production systems
- Enterprise cost centers
- Research project budgets
- Educational platforms with quotas
""")
    
    print("\n" + "=" * 80)
    print("Pattern 088 (Token Budget Management) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_token_budget_management()
