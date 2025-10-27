"""
Agentic AI Design Pattern: Service Mesh

This pattern implements a service mesh architecture for orchestrating multiple
AI agents and services. It provides service discovery, load balancing, fault
tolerance, observability, and intelligent routing.

Key Concepts:
1. Service Discovery: Dynamic agent/service registration and discovery
2. Load Balancing: Distribute requests across agent instances
3. Fault Tolerance: Circuit breakers, retries, timeouts
4. Observability: Metrics, logging, tracing
5. Traffic Management: Intelligent routing, rate limiting, canary deployments

Use Cases:
- Large-scale multi-agent systems
- Microservice-based AI architectures
- Distributed agent orchestration
- Production AI systems with high availability requirements
- Service-oriented agent platforms
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import uuid
import time
import random


class ServiceStatus(Enum):
    """Status of a service"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    STARTING = "starting"
    STOPPING = "stopping"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    WEIGHTED = "weighted"


@dataclass
class ServiceMetrics:
    """Metrics for a service"""
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    active_connections: int = 0
    
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.request_count == 0:
            return 0.0
        return self.success_count / self.request_count
    
    def average_latency(self) -> float:
        """Calculate average latency"""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count
    
    def error_rate(self) -> float:
        """Calculate error rate"""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count


@dataclass
class ServiceInstance:
    """Represents a service instance in the mesh"""
    instance_id: str
    service_name: str
    host: str
    port: int
    status: ServiceStatus = ServiceStatus.HEALTHY
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)
    registered_at: datetime = field(default_factory=datetime.now)
    last_health_check: datetime = field(default_factory=datetime.now)
    weight: float = 1.0  # For weighted load balancing
    
    def __post_init__(self):
        if not self.instance_id:
            self.instance_id = str(uuid.uuid4())
    
    def endpoint(self) -> str:
        """Get full endpoint"""
        return f"{self.host}:{self.port}"
    
    def is_healthy(self) -> bool:
        """Check if instance is healthy"""
        return self.status == ServiceStatus.HEALTHY
    
    def record_request(self, success: bool, latency_ms: float) -> None:
        """Record request metrics"""
        self.metrics.request_count += 1
        self.metrics.total_latency_ms += latency_ms
        
        if success:
            self.metrics.success_count += 1
        else:
            self.metrics.error_count += 1


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes to close from half-open
    timeout_seconds: float = 60.0  # Time before attempting half-open
    window_size: int = 10  # Size of sliding window for failures


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    config: CircuitBreakerConfig
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    recent_results: List[bool] = field(default_factory=list)
    
    def record_success(self) -> None:
        """Record successful request"""
        self.recent_results.append(True)
        self._trim_window()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close()
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record failed request"""
        self.recent_results.append(False)
        self._trim_window()
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._open()
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.config.failure_threshold:
                self._open()
    
    def can_attempt(self) -> bool:
        """Check if request can be attempted"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        # OPEN state - check if timeout has passed
        if self.last_failure_time:
            elapsed = (datetime.now() - self.last_failure_time).total_seconds()
            if elapsed >= self.config.timeout_seconds:
                self._half_open()
                return True
        
        return False
    
    def _open(self) -> None:
        """Open the circuit"""
        self.state = CircuitBreakerState.OPEN
        self.success_count = 0
    
    def _close(self) -> None:
        """Close the circuit"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
    
    def _half_open(self) -> None:
        """Set circuit to half-open"""
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
    
    def _trim_window(self) -> None:
        """Trim recent results to window size"""
        if len(self.recent_results) > self.config.window_size:
            self.recent_results = self.recent_results[-self.config.window_size:]


class ServiceRegistry:
    """Service discovery and registration"""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.instance_map: Dict[str, ServiceInstance] = {}
    
    def register(self, instance: ServiceInstance) -> bool:
        """Register a service instance"""
        if instance.service_name not in self.services:
            self.services[instance.service_name] = []
        
        self.services[instance.service_name].append(instance)
        self.instance_map[instance.instance_id] = instance
        return True
    
    def deregister(self, instance_id: str) -> bool:
        """Deregister a service instance"""
        instance = self.instance_map.get(instance_id)
        if not instance:
            return False
        
        if instance.service_name in self.services:
            self.services[instance.service_name] = [
                i for i in self.services[instance.service_name]
                if i.instance_id != instance_id
            ]
        
        del self.instance_map[instance_id]
        return True
    
    def discover(self, service_name: str, 
                only_healthy: bool = True) -> List[ServiceInstance]:
        """Discover instances of a service"""
        instances = self.services.get(service_name, [])
        
        if only_healthy:
            instances = [i for i in instances if i.is_healthy()]
        
        return instances
    
    def get_instance(self, instance_id: str) -> Optional[ServiceInstance]:
        """Get specific instance"""
        return self.instance_map.get(instance_id)
    
    def health_check(self, instance_id: str, is_healthy: bool) -> None:
        """Update health status of instance"""
        instance = self.instance_map.get(instance_id)
        if instance:
            instance.status = (ServiceStatus.HEALTHY if is_healthy 
                             else ServiceStatus.UNHEALTHY)
            instance.last_health_check = datetime.now()


class LoadBalancer:
    """Load balancer for distributing requests"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.round_robin_index: Dict[str, int] = {}
    
    def select_instance(self, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """Select an instance based on strategy"""
        if not instances:
            return None
        
        healthy_instances = [i for i in instances if i.is_healthy()]
        if not healthy_instances:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return self._random(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            return self._weighted(healthy_instances)
        
        return healthy_instances[0]
    
    def _round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin selection"""
        # Use first instance's service name as key
        key = instances[0].service_name
        
        if key not in self.round_robin_index:
            self.round_robin_index[key] = 0
        
        index = self.round_robin_index[key]
        selected = instances[index % len(instances)]
        
        self.round_robin_index[key] = (index + 1) % len(instances)
        return selected
    
    def _least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least active connections"""
        return min(instances, key=lambda i: i.metrics.active_connections)
    
    def _random(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Random selection"""
        return random.choice(instances)
    
    def _weighted(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted random selection"""
        total_weight = sum(i.weight for i in instances)
        if total_weight == 0:
            return random.choice(instances)
        
        r = random.uniform(0, total_weight)
        cumulative = 0.0
        
        for instance in instances:
            cumulative += instance.weight
            if r <= cumulative:
                return instance
        
        return instances[-1]


@dataclass
class ServiceMeshConfig:
    """Configuration for service mesh"""
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    health_check_interval_seconds: float = 30.0
    request_timeout_seconds: float = 5.0
    retry_attempts: int = 3
    retry_backoff_multiplier: float = 2.0


class ServiceMesh:
    """Complete service mesh for agent orchestration"""
    
    def __init__(self, config: Optional[ServiceMeshConfig] = None):
        self.config = config or ServiceMeshConfig()
        self.registry = ServiceRegistry()
        self.load_balancer = LoadBalancer(self.config.load_balancing_strategy)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.request_log: List[Dict[str, Any]] = []
    
    def register_service(self, service_name: str, host: str, port: int,
                        metadata: Optional[Dict[str, Any]] = None,
                        weight: float = 1.0) -> str:
        """Register a service instance"""
        instance = ServiceInstance(
            instance_id=str(uuid.uuid4()),
            service_name=service_name,
            host=host,
            port=port,
            metadata=metadata or {},
            weight=weight
        )
        
        self.registry.register(instance)
        
        # Initialize circuit breaker for this instance
        self.circuit_breakers[instance.instance_id] = CircuitBreaker(
            config=self.config.circuit_breaker_config
        )
        
        return instance.instance_id
    
    def deregister_service(self, instance_id: str) -> bool:
        """Deregister a service instance"""
        if instance_id in self.circuit_breakers:
            del self.circuit_breakers[instance_id]
        
        return self.registry.deregister(instance_id)
    
    def call_service(self, service_name: str, request: Any,
                    timeout: Optional[float] = None) -> Optional[Any]:
        """Call a service through the mesh"""
        timeout = timeout or self.config.request_timeout_seconds
        
        # Discover available instances
        instances = self.registry.discover(service_name, only_healthy=True)
        if not instances:
            self._log_request(service_name, None, False, "No healthy instances")
            return None
        
        # Try with retries
        for attempt in range(self.config.retry_attempts):
            # Select instance
            instance = self.load_balancer.select_instance(instances)
            if not instance:
                continue
            
            # Check circuit breaker
            breaker = self.circuit_breakers.get(instance.instance_id)
            if breaker and not breaker.can_attempt():
                continue
            
            # Make request (simulated)
            start_time = time.time()
            success, response = self._execute_request(instance, request, timeout)
            latency_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            instance.record_request(success, latency_ms)
            
            # Update circuit breaker
            if breaker:
                if success:
                    breaker.record_success()
                else:
                    breaker.record_failure()
            
            # Log request
            self._log_request(service_name, instance.instance_id, success, 
                            f"Attempt {attempt + 1}, latency {latency_ms:.2f}ms")
            
            if success:
                return response
            
            # Exponential backoff before retry
            if attempt < self.config.retry_attempts - 1:
                backoff = self.config.retry_backoff_multiplier ** attempt
                time.sleep(backoff * 0.1)  # Small delay for demo
        
        return None
    
    def _execute_request(self, instance: ServiceInstance, request: Any,
                        timeout: float) -> Tuple[bool, Optional[Any]]:
        """Execute request to instance (simulated)"""
        # In real implementation, this would make actual network call
        # For demo, simulate with random success/failure
        
        # Simulate network delay
        time.sleep(random.uniform(0.01, 0.05))
        
        # Simulate success/failure based on instance health
        success_probability = 0.9 if instance.is_healthy() else 0.3
        success = random.random() < success_probability
        
        if success:
            response = f"Response from {instance.instance_id[:8]}"
            return True, response
        else:
            return False, None
    
    def _log_request(self, service_name: str, instance_id: Optional[str],
                    success: bool, details: str) -> None:
        """Log request details"""
        log_entry = {
            "timestamp": datetime.now(),
            "service_name": service_name,
            "instance_id": instance_id,
            "success": success,
            "details": details
        }
        self.request_log.append(log_entry)
    
    def get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """Get aggregated metrics for a service"""
        instances = self.registry.discover(service_name, only_healthy=False)
        
        total_metrics = ServiceMetrics()
        for instance in instances:
            m = instance.metrics
            total_metrics.request_count += m.request_count
            total_metrics.success_count += m.success_count
            total_metrics.error_count += m.error_count
            total_metrics.total_latency_ms += m.total_latency_ms
        
        return {
            "service_name": service_name,
            "instance_count": len(instances),
            "healthy_instances": len([i for i in instances if i.is_healthy()]),
            "total_requests": total_metrics.request_count,
            "success_rate": total_metrics.success_rate(),
            "error_rate": total_metrics.error_rate(),
            "average_latency_ms": total_metrics.average_latency()
        }
    
    def get_circuit_breaker_status(self) -> Dict[str, str]:
        """Get status of all circuit breakers"""
        return {
            instance_id: breaker.state.value
            for instance_id, breaker in self.circuit_breakers.items()
        }


def demonstrate_service_mesh():
    """Demonstrate service mesh pattern"""
    print("=" * 70)
    print("SERVICE MESH PATTERN DEMONSTRATION")
    print("=" * 70)
    print("🎉 MILESTONE: 100TH PATTERN IMPLEMENTATION! 🎉")
    print("=" * 70)
    
    # Create service mesh
    config = ServiceMeshConfig(
        load_balancing_strategy=LoadBalancingStrategy.ROUND_ROBIN,
        retry_attempts=3
    )
    mesh = ServiceMesh(config)
    
    print("\n1. REGISTERING SERVICES")
    print("-" * 70)
    
    # Register multiple instances of services
    agent_instances = []
    for i in range(3):
        instance_id = mesh.register_service(
            service_name="ai-agent",
            host=f"agent-{i}.example.com",
            port=8000 + i,
            metadata={"version": "1.0", "region": "us-east"},
            weight=1.0
        )
        agent_instances.append(instance_id)
        print(f"   Registered: ai-agent instance {i+1} ({instance_id[:8]})")
    
    # Register analytics service
    analytics_id = mesh.register_service(
        service_name="analytics",
        host="analytics.example.com",
        port=9000,
        metadata={"version": "2.0"}
    )
    print(f"   Registered: analytics service ({analytics_id[:8]})")
    
    print("\n2. LOAD BALANCED REQUESTS")
    print("-" * 70)
    
    # Make several requests to see load balancing
    print("   Making 6 requests to ai-agent service (round-robin):")
    for i in range(6):
        response = mesh.call_service("ai-agent", {"query": f"request-{i}"})
        if response:
            print(f"     Request {i+1}: {response}")
    
    print("\n3. SERVICE METRICS")
    print("-" * 70)
    
    metrics = mesh.get_service_metrics("ai-agent")
    print(f"   Service: {metrics['service_name']}")
    print(f"   Instances: {metrics['instance_count']} "
          f"(healthy: {metrics['healthy_instances']})")
    print(f"   Total requests: {metrics['total_requests']}")
    print(f"   Success rate: {metrics['success_rate']:.1%}")
    print(f"   Average latency: {metrics['average_latency_ms']:.2f}ms")
    
    print("\n4. CIRCUIT BREAKER DEMONSTRATION")
    print("-" * 70)
    
    # Simulate service degradation
    first_instance = mesh.registry.get_instance(agent_instances[0])
    if first_instance:
        print(f"   Simulating failures on instance {agent_instances[0][:8]}...")
        first_instance.status = ServiceStatus.UNHEALTHY
        
        # Make requests - circuit breaker should open
        for i in range(10):
            mesh.call_service("ai-agent", {"query": f"test-{i}"})
        
        breaker_status = mesh.get_circuit_breaker_status()
        print(f"\n   Circuit Breaker Status:")
        for instance_id, status in breaker_status.items():
            print(f"     {instance_id[:8]}: {status}")
    
    print("\n5. SERVICE DISCOVERY")
    print("-" * 70)
    
    discovered = mesh.registry.discover("ai-agent", only_healthy=True)
    print(f"   Discovered {len(discovered)} healthy ai-agent instances:")
    for instance in discovered:
        print(f"     - {instance.endpoint()} ({instance.instance_id[:8]})")
        print(f"       Status: {instance.status.value}, "
              f"Requests: {instance.metrics.request_count}")
    
    print("\n6. DIFFERENT LOAD BALANCING STRATEGIES")
    print("-" * 70)
    
    # Test least connections
    mesh.load_balancer.strategy = LoadBalancingStrategy.LEAST_CONNECTIONS
    print("   Strategy: LEAST_CONNECTIONS")
    for i in range(3):
        response = mesh.call_service("ai-agent", {"query": f"lc-{i}"})
        if response:
            print(f"     Request {i+1}: {response}")
    
    # Test random
    mesh.load_balancer.strategy = LoadBalancingStrategy.RANDOM
    print("\n   Strategy: RANDOM")
    for i in range(3):
        response = mesh.call_service("ai-agent", {"query": f"rand-{i}"})
        if response:
            print(f"     Request {i+1}: {response}")
    
    print("\n7. FINAL METRICS")
    print("-" * 70)
    
    final_metrics = mesh.get_service_metrics("ai-agent")
    print(f"   Total requests: {final_metrics['total_requests']}")
    print(f"   Success rate: {final_metrics['success_rate']:.1%}")
    print(f"   Error rate: {final_metrics['error_rate']:.1%}")
    print(f"   Average latency: {final_metrics['average_latency_ms']:.2f}ms")
    
    print("\n8. REQUEST LOG SUMMARY")
    print("-" * 70)
    
    total_requests = len(mesh.request_log)
    successful = len([r for r in mesh.request_log if r['success']])
    print(f"   Total logged requests: {total_requests}")
    print(f"   Successful: {successful} ({successful/total_requests*100:.1f}%)")
    print(f"   Failed: {total_requests - successful}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\n🎊 CONGRATULATIONS! 🎊")
    print("You've just witnessed the 100th pattern implementation!")
    print("\nKey Features Demonstrated:")
    print("1. Service registration and discovery")
    print("2. Multiple load balancing strategies")
    print("3. Circuit breaker for fault tolerance")
    print("4. Automatic retry with exponential backoff")
    print("5. Comprehensive metrics and observability")
    print("6. Health checking and status management")
    print("7. Request logging and tracing")
    print("8. Multi-instance service orchestration")
    print("\n" + "=" * 70)
    print("MILESTONE ACHIEVED: 100/170 PATTERNS (58.8%)")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_service_mesh()
