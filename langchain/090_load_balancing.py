"""
Pattern 090: Load Balancing Pattern

Description:
    The Load Balancing Pattern distributes requests across multiple LLM endpoints or
    instances to achieve high availability, improved performance, and fault tolerance.
    In production LLM applications, load balancing is critical for handling high traffic
    volumes, preventing single points of failure, and ensuring consistent service
    quality even when individual endpoints fail or become overloaded.

    Load balancing strategies enable:
    - High availability through redundancy
    - Improved performance via parallel processing
    - Cost optimization through provider selection
    - Rate limit management across multiple accounts
    - Geographic distribution for lower latency
    - Failover and disaster recovery

Components:
    1. Load Balancing Strategies
       - Round Robin: Distribute equally across endpoints
       - Weighted Round Robin: Priority-based distribution
       - Least Loaded: Send to endpoint with lowest load
       - Random: Random endpoint selection
       - Geographic: Route based on location
       - Cost-Based: Choose cheapest available endpoint

    2. Health Checking
       - Endpoint availability monitoring
       - Response time tracking
       - Error rate monitoring
       - Periodic health probes
       - Automatic endpoint removal/recovery

    3. Request Management
       - Request queuing and buffering
       - Retry logic with backoff
       - Circuit breaker pattern
       - Request timeout handling
       - Priority-based routing

    4. Failover Mechanisms
       - Automatic endpoint switching
       - Degraded mode operation
       - Fallback to alternative providers
       - Request replay on failure
       - State preservation

Use Cases:
    1. High-Traffic Applications
       - Chatbots serving thousands of users
       - API services with SLA requirements
       - Real-time processing pipelines
       - Multi-tenant platforms
       - Peak load handling

    2. Multi-Provider Deployments
       - OpenAI + Anthropic + Google
       - Primary and backup providers
       - Cost optimization strategies
       - Feature-specific routing
       - Geographic compliance

    3. Rate Limit Management
       - Distribute across multiple API keys
       - Prevent rate limit exhaustion
       - Smooth request distribution
       - Burst handling
       - Cost control

    4. Disaster Recovery
       - Automatic failover
       - Multi-region deployment
       - Zero-downtime updates
       - Service degradation handling
       - Business continuity

    5. Performance Optimization
       - Parallel request processing
       - Load-based routing
       - Latency optimization
       - Resource utilization
       - Capacity planning

LangChain Implementation:
    LangChain doesn't have built-in load balancing, but you can implement
    it using:
    - Custom LLM wrappers
    - Callback handlers for monitoring
    - Retry logic with different models
    - Custom routing logic
    - Health check mechanisms

Key Features:
    1. Endpoint Management
       - Dynamic endpoint registration
       - Health status tracking
       - Automatic recovery
       - Configuration updates
       - Version management

    2. Intelligent Routing
       - Strategy selection
       - Real-time decision making
       - Performance-based routing
       - Cost optimization
       - SLA enforcement

    3. Fault Tolerance
       - Automatic retry with exponential backoff
       - Circuit breaker to prevent cascading failures
       - Graceful degradation
       - Error handling and recovery
       - Request preservation

    4. Monitoring
       - Request distribution metrics
       - Endpoint health status
       - Response time tracking
       - Error rate monitoring
       - Cost tracking per endpoint

Best Practices:
    1. Health Monitoring
       - Regular health checks (every 30-60s)
       - Multiple health metrics (latency, errors, availability)
       - Automatic endpoint recovery
       - Alert on degraded performance
       - Log all health transitions

    2. Load Distribution
       - Consider endpoint capacity
       - Account for different model speeds
       - Balance cost vs performance
       - Avoid overloading any single endpoint
       - Reserve capacity for spikes

    3. Failover Strategy
       - Fast failure detection (< 5s)
       - Quick failover (< 1s)
       - Preserve request context
       - Transparent to users
       - Log all failovers

    4. Request Management
       - Set appropriate timeouts
       - Implement retry with backoff
       - Queue management for spikes
       - Priority handling
       - Request deduplication

Trade-offs:
    Advantages:
    - High availability (99.9%+ uptime)
    - Better performance through parallelization
    - Fault tolerance and resilience
    - Flexible scaling
    - Cost optimization opportunities
    - Geographic distribution

    Disadvantages:
    - Increased complexity
    - Additional monitoring requirements
    - Potential consistency issues
    - Higher operational costs
    - Configuration management overhead
    - Testing complexity

Production Considerations:
    1. Endpoint Configuration
       - Multiple providers (OpenAI, Anthropic, etc.)
       - Multiple API keys per provider
       - Geographic distribution
       - Capacity planning
       - Cost monitoring

    2. Monitoring and Alerting
       - Real-time health dashboards
       - Performance metrics
       - Error tracking
       - Cost tracking
       - SLA compliance monitoring

    3. Security
       - Secure credential management
       - Rate limiting per client
       - Access control
       - Audit logging
       - Encryption in transit

    4. Scaling
       - Horizontal scaling capability
       - Auto-scaling based on load
       - Capacity planning
       - Performance testing
       - Load testing

    5. Disaster Recovery
       - Multi-region deployment
       - Backup endpoints
       - Failover procedures
       - Data consistency
       - Recovery time objectives (RTO)
"""

import os
import time
import random
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    FASTEST_RESPONSE = "fastest_response"


class EndpointStatus(Enum):
    """Endpoint health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class EndpointConfig:
    """Configuration for an LLM endpoint"""
    id: str
    model: str
    api_key: Optional[str] = None
    weight: int = 1  # For weighted load balancing
    max_requests_per_minute: int = 60
    timeout: int = 30
    provider: str = "openai"


@dataclass
class EndpointMetrics:
    """Metrics for an endpoint"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    active_requests: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    status: EndpointStatus = EndpointStatus.UNKNOWN
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests


@dataclass
class LoadBalancerStats:
    """Overall load balancer statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    failovers: int = 0
    total_response_time: float = 0.0
    endpoint_metrics: Dict[str, EndpointMetrics] = field(default_factory=dict)


class LLMEndpoint:
    """
    Represents a single LLM endpoint with health tracking.
    """
    
    def __init__(self, config: EndpointConfig):
        """
        Initialize endpoint.
        
        Args:
            config: Endpoint configuration
        """
        self.config = config
        self.metrics = EndpointMetrics()
        self.llm: Optional[ChatOpenAI] = None
        self.last_health_check: Optional[datetime] = None
        self.request_times: deque = deque(maxlen=100)  # Track recent request times
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM client"""
        try:
            # Use provided API key or default from environment
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            
            self.llm = ChatOpenAI(
                model=self.config.model,
                openai_api_key=api_key,
                timeout=self.config.timeout,
                temperature=0.7
            )
            self.metrics.status = EndpointStatus.HEALTHY
        except Exception as e:
            print(f"Error initializing endpoint {self.config.id}: {e}")
            self.metrics.status = EndpointStatus.UNHEALTHY
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.now()
    
    def is_available(self) -> bool:
        """Check if endpoint is available for requests"""
        return (
            self.metrics.status in [EndpointStatus.HEALTHY, EndpointStatus.DEGRADED]
            and self.llm is not None
            and self.metrics.active_requests < 10  # Max concurrent requests
        )
    
    def check_health(self) -> EndpointStatus:
        """
        Perform health check on endpoint.
        
        Returns:
            Current health status
        """
        try:
            if self.llm is None:
                self._initialize_llm()
            
            # Simple health check - try a minimal request
            start_time = time.time()
            self.llm.invoke("test")
            response_time = time.time() - start_time
            
            # Update status based on response time and error rate
            if response_time < 2.0 and self.metrics.success_rate > 95:
                self.metrics.status = EndpointStatus.HEALTHY
            elif response_time < 5.0 and self.metrics.success_rate > 80:
                self.metrics.status = EndpointStatus.DEGRADED
            else:
                self.metrics.status = EndpointStatus.UNHEALTHY
            
            self.last_health_check = datetime.now()
            return self.metrics.status
            
        except Exception as e:
            self.metrics.status = EndpointStatus.UNHEALTHY
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.now()
            return EndpointStatus.UNHEALTHY
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Generate response using this endpoint.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Response dictionary
        """
        self.metrics.total_requests += 1
        self.metrics.active_requests += 1
        
        try:
            start_time = time.time()
            response = self.llm.invoke(prompt)
            response_time = time.time() - start_time
            
            self.metrics.successful_requests += 1
            self.metrics.total_response_time += response_time
            self.metrics.active_requests -= 1
            self.request_times.append(response_time)
            
            return {
                "response": response.content,
                "endpoint_id": self.config.id,
                "response_time": response_time,
                "success": True
            }
            
        except Exception as e:
            self.metrics.failed_requests += 1
            self.metrics.active_requests -= 1
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.now()
            
            # Mark as unhealthy after repeated failures
            if self.metrics.success_rate < 50:
                self.metrics.status = EndpointStatus.UNHEALTHY
            
            return {
                "response": None,
                "endpoint_id": self.config.id,
                "error": str(e),
                "success": False
            }


class LoadBalancer:
    """
    Load balancer for distributing requests across multiple LLM endpoints.
    
    Supports multiple load balancing strategies and automatic failover.
    """
    
    def __init__(
        self,
        endpoints: List[EndpointConfig],
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        health_check_interval: int = 60
    ):
        """
        Initialize load balancer.
        
        Args:
            endpoints: List of endpoint configurations
            strategy: Load balancing strategy
            health_check_interval: Seconds between health checks
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.endpoints: List[LLMEndpoint] = []
        self.stats = LoadBalancerStats()
        self.current_endpoint_index = 0
        self.last_health_check = datetime.now()
        
        # Initialize endpoints
        for config in endpoints:
            endpoint = LLMEndpoint(config)
            self.endpoints.append(endpoint)
            self.stats.endpoint_metrics[config.id] = endpoint.metrics
    
    def _select_endpoint_round_robin(self) -> Optional[LLMEndpoint]:
        """Select endpoint using round-robin strategy"""
        attempts = 0
        while attempts < len(self.endpoints):
            endpoint = self.endpoints[self.current_endpoint_index]
            self.current_endpoint_index = (self.current_endpoint_index + 1) % len(self.endpoints)
            
            if endpoint.is_available():
                return endpoint
            
            attempts += 1
        
        return None
    
    def _select_endpoint_weighted(self) -> Optional[LLMEndpoint]:
        """Select endpoint using weighted round-robin strategy"""
        # Create weighted list
        weighted_endpoints = []
        for endpoint in self.endpoints:
            if endpoint.is_available():
                weighted_endpoints.extend([endpoint] * endpoint.config.weight)
        
        if not weighted_endpoints:
            return None
        
        # Round-robin through weighted list
        endpoint = weighted_endpoints[self.current_endpoint_index % len(weighted_endpoints)]
        self.current_endpoint_index += 1
        return endpoint
    
    def _select_endpoint_least_loaded(self) -> Optional[LLMEndpoint]:
        """Select endpoint with least active requests"""
        available = [e for e in self.endpoints if e.is_available()]
        if not available:
            return None
        
        return min(available, key=lambda e: e.metrics.active_requests)
    
    def _select_endpoint_random(self) -> Optional[LLMEndpoint]:
        """Select random available endpoint"""
        available = [e for e in self.endpoints if e.is_available()]
        if not available:
            return None
        
        return random.choice(available)
    
    def _select_endpoint_fastest(self) -> Optional[LLMEndpoint]:
        """Select endpoint with fastest average response time"""
        available = [e for e in self.endpoints if e.is_available()]
        if not available:
            return None
        
        # Sort by average response time (endpoints with no history get high priority)
        return min(
            available,
            key=lambda e: e.metrics.average_response_time if e.metrics.successful_requests > 0 else 0.0
        )
    
    def _select_endpoint(self) -> Optional[LLMEndpoint]:
        """
        Select endpoint based on strategy.
        
        Returns:
            Selected endpoint or None if none available
        """
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._select_endpoint_round_robin()
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._select_endpoint_weighted()
        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._select_endpoint_least_loaded()
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._select_endpoint_random()
        elif self.strategy == LoadBalancingStrategy.FASTEST_RESPONSE:
            return self._select_endpoint_fastest()
        else:
            return self._select_endpoint_round_robin()
    
    def _perform_health_checks(self):
        """Perform health checks on all endpoints if needed"""
        now = datetime.now()
        if (now - self.last_health_check).seconds < self.health_check_interval:
            return
        
        for endpoint in self.endpoints:
            endpoint.check_health()
        
        self.last_health_check = now
    
    def generate(
        self,
        prompt: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate response with load balancing and failover.
        
        Args:
            prompt: Input prompt
            max_retries: Maximum retry attempts
            
        Returns:
            Response dictionary
        """
        self.stats.total_requests += 1
        
        # Perform health checks if needed
        self._perform_health_checks()
        
        attempts = 0
        last_error = None
        
        while attempts < max_retries:
            # Select endpoint
            endpoint = self._select_endpoint()
            
            if endpoint is None:
                last_error = "No available endpoints"
                break
            
            # Try to generate response
            result = endpoint.generate(prompt)
            
            if result["success"]:
                self.stats.successful_requests += 1
                self.stats.total_response_time += result["response_time"]
                return result
            
            # Request failed, try another endpoint
            self.stats.failovers += 1
            last_error = result.get("error", "Unknown error")
            attempts += 1
            
            # Brief delay before retry
            time.sleep(0.1 * attempts)  # Exponential backoff
        
        # All retries failed
        self.stats.failed_requests += 1
        return {
            "response": None,
            "error": f"All endpoints failed after {attempts} attempts. Last error: {last_error}",
            "success": False
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        return {
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "failovers": self.stats.failovers,
            "success_rate": (
                (self.stats.successful_requests / self.stats.total_requests * 100)
                if self.stats.total_requests > 0 else 0.0
            ),
            "average_response_time": (
                self.stats.total_response_time / self.stats.successful_requests
                if self.stats.successful_requests > 0 else 0.0
            ),
            "endpoints": [
                {
                    "id": e.config.id,
                    "model": e.config.model,
                    "status": e.metrics.status.value,
                    "total_requests": e.metrics.total_requests,
                    "success_rate": f"{e.metrics.success_rate:.2f}%",
                    "avg_response_time": f"{e.metrics.average_response_time:.3f}s",
                    "active_requests": e.metrics.active_requests
                }
                for e in self.endpoints
            ]
        }


def demonstrate_load_balancing():
    """Demonstrate load balancing patterns"""
    print("=" * 80)
    print("LOAD BALANCING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic Round-Robin Load Balancing
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Round-Robin Load Balancing")
    print("=" * 80)
    
    # Configure multiple endpoints (simulating multiple API keys or models)
    endpoints = [
        EndpointConfig(id="endpoint-1", model="gpt-3.5-turbo", weight=1),
        EndpointConfig(id="endpoint-2", model="gpt-3.5-turbo", weight=1),
        EndpointConfig(id="endpoint-3", model="gpt-3.5-turbo", weight=1),
    ]
    
    lb = LoadBalancer(endpoints, strategy=LoadBalancingStrategy.ROUND_ROBIN)
    
    print("\nConfigured 3 endpoints with round-robin strategy\n")
    print("Sending 6 requests to observe distribution:\n")
    
    queries = [
        "What is 2+2?",
        "What is Python?",
        "What is AI?",
        "What is ML?",
        "What is DL?",
        "What is NLP?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"Request {i}: {query}")
        result = lb.generate(query)
        
        if result["success"]:
            print(f"  ✓ Endpoint: {result['endpoint_id']}")
            print(f"  Response time: {result['response_time']:.3f}s")
            print(f"  Response: {result['response'][:60]}...")
        else:
            print(f"  ✗ Error: {result['error']}")
        print()
    
    stats = lb.get_stats()
    print("Load Distribution:")
    for endpoint in stats["endpoints"]:
        print(f"  {endpoint['id']}: {endpoint['total_requests']} requests")
    
    # Example 2: Weighted Load Balancing
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Weighted Load Balancing")
    print("=" * 80)
    
    weighted_endpoints = [
        EndpointConfig(id="premium-endpoint", model="gpt-4", weight=3),  # Gets more traffic
        EndpointConfig(id="standard-endpoint", model="gpt-3.5-turbo", weight=1),
    ]
    
    weighted_lb = LoadBalancer(
        weighted_endpoints,
        strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN
    )
    
    print("\nConfigured 2 endpoints:")
    print("  - Premium (GPT-4): weight=3")
    print("  - Standard (GPT-3.5): weight=1")
    print("\nSending 8 requests:\n")
    
    for i in range(8):
        result = weighted_lb.generate(f"Request {i+1}")
        if result["success"]:
            print(f"Request {i+1} → {result['endpoint_id']}")
    
    print("\nExpected distribution: ~6 to premium, ~2 to standard")
    stats = weighted_lb.get_stats()
    for endpoint in stats["endpoints"]:
        print(f"  {endpoint['id']}: {endpoint['total_requests']} requests")
    
    # Example 3: Least-Loaded Strategy
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Least-Loaded Strategy")
    print("=" * 80)
    
    ll_endpoints = [
        EndpointConfig(id="endpoint-A", model="gpt-3.5-turbo"),
        EndpointConfig(id="endpoint-B", model="gpt-3.5-turbo"),
    ]
    
    ll_lb = LoadBalancer(ll_endpoints, strategy=LoadBalancingStrategy.LEAST_LOADED)
    
    print("\nLeast-loaded strategy routes to endpoint with fewest active requests")
    print("This optimizes for load distribution in real-time\n")
    
    for i in range(4):
        result = ll_lb.generate(f"Query {i+1}")
        if result["success"]:
            print(f"Query {i+1} → {result['endpoint_id']}")
    
    # Example 4: Failover Handling
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Automatic Failover")
    print("=" * 80)
    
    print("\nSimulating endpoint failure scenario:")
    print("  - 3 endpoints configured")
    print("  - Mark one as unhealthy")
    print("  - Load balancer automatically routes around it\n")
    
    failover_endpoints = [
        EndpointConfig(id="healthy-1", model="gpt-3.5-turbo"),
        EndpointConfig(id="unhealthy", model="gpt-3.5-turbo"),
        EndpointConfig(id="healthy-2", model="gpt-3.5-turbo"),
    ]
    
    fo_lb = LoadBalancer(failover_endpoints, strategy=LoadBalancingStrategy.ROUND_ROBIN)
    
    # Simulate endpoint failure
    fo_lb.endpoints[1].metrics.status = EndpointStatus.UNHEALTHY
    print("Marked 'unhealthy' endpoint as UNHEALTHY\n")
    
    print("Sending 4 requests:\n")
    for i in range(4):
        result = fo_lb.generate(f"Query {i+1}")
        if result["success"]:
            print(f"Query {i+1} → {result['endpoint_id']} ✓")
    
    print("\nOnly healthy endpoints received requests!")
    
    # Example 5: Performance Comparison
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Strategy Performance Comparison")
    print("=" * 80)
    
    strategies = [
        LoadBalancingStrategy.ROUND_ROBIN,
        LoadBalancingStrategy.RANDOM,
        LoadBalancingStrategy.LEAST_LOADED,
    ]
    
    print("\nComparing different load balancing strategies:\n")
    
    for strategy in strategies:
        test_endpoints = [
            EndpointConfig(id=f"ep-{i}", model="gpt-3.5-turbo")
            for i in range(3)
        ]
        
        test_lb = LoadBalancer(test_endpoints, strategy=strategy)
        
        # Send some requests
        for i in range(6):
            test_lb.generate(f"Test query {i}")
        
        stats = test_lb.get_stats()
        print(f"{strategy.value.upper()}:")
        print(f"  Success Rate: {stats['success_rate']:.2f}%")
        print(f"  Total Requests: {stats['total_requests']}")
        for ep in stats["endpoints"]:
            print(f"    {ep['id']}: {ep['total_requests']} requests")
        print()
    
    # Example 6: Monitoring and Statistics
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Comprehensive Statistics")
    print("=" * 80)
    
    monitor_endpoints = [
        EndpointConfig(id="prod-1", model="gpt-3.5-turbo"),
        EndpointConfig(id="prod-2", model="gpt-3.5-turbo"),
    ]
    
    monitor_lb = LoadBalancer(monitor_endpoints, strategy=LoadBalancingStrategy.ROUND_ROBIN)
    
    print("\nSimulating production workload...\n")
    
    # Simulate various scenarios
    for i in range(10):
        result = monitor_lb.generate(f"Production query {i+1}")
    
    stats = monitor_lb.get_stats()
    
    print("Load Balancer Statistics:")
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Successful: {stats['successful_requests']}")
    print(f"  Failed: {stats['failed_requests']}")
    print(f"  Success Rate: {stats['success_rate']:.2f}%")
    print(f"  Failovers: {stats['failovers']}")
    print(f"  Avg Response Time: {stats['average_response_time']:.3f}s")
    
    print("\nPer-Endpoint Metrics:")
    for ep in stats["endpoints"]:
        print(f"\n  {ep['id']} ({ep['model']}):")
        print(f"    Status: {ep['status']}")
        print(f"    Requests: {ep['total_requests']}")
        print(f"    Success Rate: {ep['success_rate']}")
        print(f"    Avg Response Time: {ep['avg_response_time']}")
        print(f"    Active Requests: {ep['active_requests']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("LOAD BALANCING PATTERN SUMMARY")
    print("=" * 80)
    print("""
Load Balancing Benefits:
1. High Availability: 99.9%+ uptime through redundancy
2. Performance: Better throughput via parallel processing
3. Fault Tolerance: Automatic failover on endpoint failure
4. Scalability: Easy to add more endpoints
5. Cost Optimization: Route based on cost/performance
6. Rate Limit Management: Distribute across API keys

Load Balancing Strategies:
1. Round Robin
   - Equal distribution across endpoints
   - Simple and predictable
   - Good for homogeneous endpoints
   - Use when: All endpoints have similar capacity

2. Weighted Round Robin
   - Priority-based distribution
   - Higher weight = more requests
   - Good for heterogeneous endpoints
   - Use when: Different endpoint capacities

3. Least Loaded
   - Route to least busy endpoint
   - Optimizes for current load
   - Dynamic adjustment
   - Use when: Variable request durations

4. Random
   - Random endpoint selection
   - Stateless and simple
   - Good load distribution over time
   - Use when: Simplicity preferred

5. Fastest Response
   - Route to historically fastest endpoint
   - Performance-optimized
   - Adaptive to endpoint characteristics
   - Use when: Latency is critical

Health Checking:
1. Active Health Checks
   - Regular test requests
   - Verify endpoint availability
   - Monitor response times
   - Interval: 30-60 seconds

2. Passive Health Monitoring
   - Track request success/failure
   - Monitor error rates
   - Calculate response times
   - Continuous monitoring

3. Status Levels
   - HEALTHY: < 2s response, > 95% success
   - DEGRADED: < 5s response, > 80% success
   - UNHEALTHY: Slow or < 80% success
   - UNKNOWN: Not yet tested

Failover Strategies:
1. Immediate Retry
   - Try another endpoint immediately
   - Best for transient failures
   - No user-visible delay

2. Exponential Backoff
   - Increasing delays between retries
   - Prevents overwhelming failed endpoints
   - Format: 100ms, 200ms, 400ms...

3. Circuit Breaker
   - Stop trying failed endpoints temporarily
   - Give time to recover
   - Prevent cascading failures

4. Graceful Degradation
   - Return cached/simpler response
   - Maintain partial functionality
   - Better than complete failure

Best Practices:
1. Endpoint Configuration
   - At least 2 endpoints minimum
   - 3+ recommended for production
   - Mix providers for true redundancy
   - Consider geographic distribution

2. Monitoring
   - Track per-endpoint metrics
   - Monitor overall success rate
   - Alert on degraded performance
   - Dashboard for real-time visibility

3. Timeout Configuration
   - Set appropriate timeouts (15-30s)
   - Fail fast on unresponsive endpoints
   - Allow time for LLM generation
   - Balance speed vs success

4. Retry Logic
   - Max 3 retry attempts
   - Use exponential backoff
   - Track retry reasons
   - Avoid retry storms

5. Testing
   - Test failover scenarios
   - Simulate endpoint failures
   - Load testing with multiple endpoints
   - Verify health check accuracy

Production Deployment:
1. Multi-Provider Setup
   - Primary: OpenAI GPT-4
   - Secondary: OpenAI GPT-3.5 (fallback)
   - Tertiary: Anthropic Claude (backup)

2. Geographic Distribution
   - US-East endpoint
   - US-West endpoint
   - EU endpoint
   - Route based on user location

3. Cost Optimization
   - Use cheaper models when possible
   - Route simple queries to GPT-3.5
   - Complex queries to GPT-4
   - Monitor cost per endpoint

4. Rate Limit Management
   - Multiple API keys per provider
   - Distribute requests evenly
   - Track usage per key
   - Automatic key rotation

When to Use Load Balancing:
✓ High-traffic applications (>1000 req/hr)
✓ SLA requirements (>99% uptime)
✓ Rate limit concerns
✓ Multi-provider deployments
✓ Geographic distribution needed
✗ Low-traffic applications
✗ Single endpoint sufficient
✗ Cost constraints (added complexity)

ROI Analysis:
- Uptime improvement: 99% → 99.9% (10x fewer outages)
- Performance: 2-3x throughput capacity
- User satisfaction: Reduced error rates
- Cost: +20% complexity, but worth it for scale
- Break-even: ~500 requests/hour
""")
    
    print("\n" + "=" * 80)
    print("Pattern 090 (Load Balancing) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_load_balancing()
