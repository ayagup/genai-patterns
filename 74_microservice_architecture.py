"""
Microservice Agent Architecture Pattern

Decomposes agents into microservices with API gateway, service mesh, and
distributed processing. Enables scalability, independent deployment, and
fault isolation in large-scale agent systems.

Use Cases:
- Large-scale production agent systems
- Multi-tenant agent platforms
- Cloud-native agent deployments
- High-availability agent services
- Distributed AI systems

Benefits:
- Scalability: Independent scaling of services
- Fault isolation: Failures contained to services
- Independent deployment: Update services separately
- Technology diversity: Different stacks per service
- Team autonomy: Distributed development
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import random


class ServiceStatus(Enum):
    """Status of a microservice"""
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    RANDOM = "random"


@dataclass
class ServiceInstance:
    """A microservice instance"""
    instance_id: str
    service_name: str
    host: str
    port: int
    status: ServiceStatus = ServiceStatus.STARTING
    health_check_endpoint: str = "/health"
    current_load: int = 0
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_healthy(self) -> bool:
        """Check if instance is healthy"""
        return self.status == ServiceStatus.HEALTHY
    
    def get_address(self) -> str:
        """Get full address"""
        return f"{self.host}:{self.port}"


@dataclass
class ServiceRequest:
    """Request to a microservice"""
    request_id: str
    service_name: str
    endpoint: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    body: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 30.0
    

@dataclass
class ServiceResponse:
    """Response from a microservice"""
    request_id: str
    status_code: int
    body: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    latency_ms: float = 0.0
    instance_id: Optional[str] = None


class ServiceRegistry:
    """Service registry for microservices"""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.registration_count = 0
    
    def register(self, instance: ServiceInstance) -> None:
        """Register a service instance"""
        if instance.service_name not in self.services:
            self.services[instance.service_name] = []
        
        self.services[instance.service_name].append(instance)
        self.registration_count += 1
        
        print(f"[Registry] Registered: {instance.service_name} "
              f"({instance.instance_id}) at {instance.get_address()}")
    
    def deregister(self, instance_id: str) -> bool:
        """Deregister a service instance"""
        for service_name, instances in self.services.items():
            for i, instance in enumerate(instances):
                if instance.instance_id == instance_id:
                    instances.pop(i)
                    print(f"[Registry] Deregistered: {instance_id}")
                    return True
        return False
    
    def discover(self, service_name: str) -> List[ServiceInstance]:
        """Discover instances of a service"""
        return self.services.get(service_name, [])
    
    def get_healthy_instances(self, service_name: str) -> List[ServiceInstance]:
        """Get healthy instances of a service"""
        instances = self.discover(service_name)
        return [i for i in instances if i.is_healthy()]
    
    def health_check_all(self) -> Dict[str, int]:
        """Run health check on all services"""
        results = {}
        
        for service_name, instances in self.services.items():
            healthy = sum(1 for i in instances if i.is_healthy())
            results[service_name] = healthy
        
        return results


class APIGateway:
    """API Gateway for routing requests"""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.load_balancer = LoadBalancer(registry)
        self.request_count = 0
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
    
    def route(self, request: ServiceRequest) -> ServiceResponse:
        """Route request to appropriate service"""
        self.request_count += 1
        
        print(f"\n[Gateway] Routing request {request.request_id}")
        print(f"  Service: {request.service_name}")
        print(f"  Endpoint: {request.endpoint}")
        
        # Check rate limiting
        if not self._check_rate_limit(request):
            return ServiceResponse(
                request_id=request.request_id,
                status_code=429,
                body={"error": "Rate limit exceeded"}
            )
        
        # Get service instance through load balancer
        instance = self.load_balancer.select_instance(request.service_name)
        
        if not instance:
            return ServiceResponse(
                request_id=request.request_id,
                status_code=503,
                body={"error": "Service unavailable"}
            )
        
        # Forward request to service
        start_time = time.time()
        response = self._forward_request(instance, request)
        latency = (time.time() - start_time) * 1000
        
        response.latency_ms = latency
        response.instance_id = instance.instance_id
        
        print(f"  Routed to: {instance.instance_id}")
        print(f"  Status: {response.status_code}")
        print(f"  Latency: {latency:.2f}ms")
        
        return response
    
    def _check_rate_limit(self, request: ServiceRequest) -> bool:
        """Check if request exceeds rate limit"""
        # Simple rate limiting (in production: use Redis or similar)
        client_id = request.headers.get("X-Client-ID", "anonymous")
        
        if client_id not in self.rate_limiters:
            self.rate_limiters[client_id] = {"count": 0, "window_start": time.time()}
        
        limiter = self.rate_limiters[client_id]
        
        # Reset if window expired (1 second window)
        if time.time() - limiter["window_start"] > 1.0:
            limiter["count"] = 0
            limiter["window_start"] = time.time()
        
        # Check limit (100 requests per second)
        if limiter["count"] >= 100:
            return False
        
        limiter["count"] += 1
        return True
    
    def _forward_request(
        self,
        instance: ServiceInstance,
        request: ServiceRequest
    ) -> ServiceResponse:
        """Forward request to service instance"""
        # Simulate service call
        time.sleep(random.uniform(0.01, 0.05))
        
        # Simulate occasional failures
        if random.random() < 0.05:
            return ServiceResponse(
                request_id=request.request_id,
                status_code=500,
                body={"error": "Internal server error"}
            )
        
        return ServiceResponse(
            request_id=request.request_id,
            status_code=200,
            body={"result": f"Processed by {instance.service_name}"}
        )


class LoadBalancer:
    """Load balancer for distributing requests"""
    
    def __init__(
        self,
        registry: ServiceRegistry,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    ):
        self.registry = registry
        self.strategy = strategy
        self.round_robin_indices: Dict[str, int] = {}
    
    def select_instance(self, service_name: str) -> Optional[ServiceInstance]:
        """Select instance using load balancing strategy"""
        instances = self.registry.get_healthy_instances(service_name)
        
        if not instances:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(service_name, instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            return self._weighted(instances)
        else:  # RANDOM
            return random.choice(instances)
    
    def _round_robin(
        self,
        service_name: str,
        instances: List[ServiceInstance]
    ) -> ServiceInstance:
        """Round-robin selection"""
        if service_name not in self.round_robin_indices:
            self.round_robin_indices[service_name] = 0
        
        index = self.round_robin_indices[service_name]
        instance = instances[index % len(instances)]
        
        self.round_robin_indices[service_name] = (index + 1) % len(instances)
        
        return instance
    
    def _least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least connections"""
        return min(instances, key=lambda i: i.current_load)
    
    def _weighted(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted random selection"""
        total_weight = sum(i.weight for i in instances)
        r = random.uniform(0, total_weight)
        
        cumsum = 0
        for instance in instances:
            cumsum += instance.weight
            if r <= cumsum:
                return instance
        
        return instances[-1]


class ServiceMesh:
    """Service mesh for inter-service communication"""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.retry_policies: Dict[str, Dict[str, Any]] = {}
    
    def call_service(
        self,
        from_service: str,
        to_service: str,
        request: ServiceRequest
    ) -> ServiceResponse:
        """Call service through mesh with resilience patterns"""
        print(f"\n[Service Mesh] {from_service} → {to_service}")
        
        # Check circuit breaker
        if self._is_circuit_open(to_service):
            return ServiceResponse(
                request_id=request.request_id,
                status_code=503,
                body={"error": "Circuit breaker open"}
            )
        
        # Try with retries
        max_retries = self._get_retry_count(to_service)
        
        for attempt in range(max_retries + 1):
            response = self._attempt_call(to_service, request)
            
            if response.status_code == 200:
                self._record_success(to_service)
                return response
            
            if attempt < max_retries:
                delay = 2 ** attempt * 0.1  # Exponential backoff
                print(f"  Retry {attempt + 1}/{max_retries} after {delay:.2f}s")
                time.sleep(delay)
        
        self._record_failure(to_service)
        return response
    
    def _attempt_call(
        self,
        service_name: str,
        request: ServiceRequest
    ) -> ServiceResponse:
        """Attempt to call service"""
        instances = self.registry.get_healthy_instances(service_name)
        
        if not instances:
            return ServiceResponse(
                request_id=request.request_id,
                status_code=503,
                body={"error": "No healthy instances"}
            )
        
        instance = random.choice(instances)
        
        # Simulate call
        time.sleep(random.uniform(0.01, 0.05))
        
        if random.random() < 0.1:  # 10% failure rate
            return ServiceResponse(
                request_id=request.request_id,
                status_code=500,
                body={"error": "Service error"}
            )
        
        return ServiceResponse(
            request_id=request.request_id,
            status_code=200,
            body={"result": f"Response from {service_name}"}
        )
    
    def _is_circuit_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = {
                "failures": 0,
                "threshold": 5,
                "open": False
            }
        
        return self.circuit_breakers[service_name]["open"]
    
    def _record_success(self, service_name: str) -> None:
        """Record successful call"""
        if service_name in self.circuit_breakers:
            self.circuit_breakers[service_name]["failures"] = 0
            self.circuit_breakers[service_name]["open"] = False
    
    def _record_failure(self, service_name: str) -> None:
        """Record failed call"""
        breaker = self.circuit_breakers[service_name]
        breaker["failures"] += 1
        
        if breaker["failures"] >= breaker["threshold"]:
            breaker["open"] = True
            print(f"  [Circuit Breaker] Opened for {service_name}")
    
    def _get_retry_count(self, service_name: str) -> int:
        """Get retry count for service"""
        if service_name not in self.retry_policies:
            self.retry_policies[service_name] = {"max_retries": 3}
        
        return self.retry_policies[service_name]["max_retries"]


class MicroserviceAgentSystem:
    """
    Microservice Agent Architecture
    
    Complete microservice-based agent system with service registry,
    API gateway, load balancing, and service mesh.
    """
    
    def __init__(self, system_name: str = "Agent Microservices"):
        self.system_name = system_name
        self.registry = ServiceRegistry()
        self.gateway = APIGateway(self.registry)
        self.service_mesh = ServiceMesh(self.registry)
        
        print(f"[System] Initialized: {system_name}")
    
    def deploy_service(
        self,
        service_name: str,
        instance_count: int = 3
    ) -> List[ServiceInstance]:
        """Deploy a microservice with multiple instances"""
        print(f"\n[Deploy] Deploying {service_name} ({instance_count} instances)")
        
        instances = []
        
        for i in range(instance_count):
            instance = ServiceInstance(
                instance_id=f"{service_name}-{i}",
                service_name=service_name,
                host=f"192.168.1.{100 + i}",
                port=8080 + i,
                status=ServiceStatus.HEALTHY,
                weight=1.0 if i < instance_count - 1 else 0.5  # Last instance half weight
            )
            
            self.registry.register(instance)
            instances.append(instance)
        
        return instances
    
    def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process request through gateway"""
        return self.gateway.route(request)
    
    def service_to_service_call(
        self,
        from_service: str,
        to_service: str,
        request: ServiceRequest
    ) -> ServiceResponse:
        """Inter-service communication through mesh"""
        return self.service_mesh.call_service(from_service, to_service, request)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        health = self.registry.health_check_all()
        
        total_instances = sum(len(instances) for instances in self.registry.services.values())
        healthy_instances = sum(health.values())
        
        return {
            "total_services": len(self.registry.services),
            "total_instances": total_instances,
            "healthy_instances": healthy_instances,
            "health_percentage": (healthy_instances / total_instances * 100) if total_instances > 0 else 0,
            "services": health
        }
    
    def scale_service(self, service_name: str, target_instances: int) -> None:
        """Scale service to target instance count"""
        current_instances = self.registry.discover(service_name)
        current_count = len(current_instances)
        
        print(f"\n[Scale] {service_name}: {current_count} → {target_instances}")
        
        if target_instances > current_count:
            # Scale up
            for i in range(current_count, target_instances):
                instance = ServiceInstance(
                    instance_id=f"{service_name}-{i}",
                    service_name=service_name,
                    host=f"192.168.1.{100 + i}",
                    port=8080 + i,
                    status=ServiceStatus.HEALTHY
                )
                self.registry.register(instance)
        
        elif target_instances < current_count:
            # Scale down
            for instance in current_instances[target_instances:]:
                self.registry.deregister(instance.instance_id)


def demonstrate_microservice_architecture():
    """
    Demonstrate Microservice Agent Architecture pattern
    """
    print("=" * 70)
    print("MICROSERVICE AGENT ARCHITECTURE DEMONSTRATION")
    print("=" * 70)
    
    # Create microservice system
    system = MicroserviceAgentSystem("AI Agent Platform")
    
    # Example 1: Deploy services
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Service Deployment")
    print("=" * 70)
    
    # Deploy multiple services
    system.deploy_service("reasoning-service", instance_count=3)
    system.deploy_service("memory-service", instance_count=2)
    system.deploy_service("tool-service", instance_count=2)
    
    # Check system health
    health = system.get_system_health()
    print(f"\n[System Health]")
    print(f"  Total services: {health['total_services']}")
    print(f"  Total instances: {health['total_instances']}")
    print(f"  Health: {health['health_percentage']:.1f}%")
    
    # Example 2: Request routing
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Request Routing through API Gateway")
    print("=" * 70)
    
    # Process multiple requests
    for i in range(5):
        request = ServiceRequest(
            request_id=f"req-{i}",
            service_name="reasoning-service",
            endpoint="/analyze",
            method="POST",
            headers={"X-Client-ID": "client-1"},
            body={"query": f"Question {i}"}
        )
        
        response = system.process_request(request)
    
    # Example 3: Service-to-service communication
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Inter-Service Communication")
    print("=" * 70)
    
    # Reasoning service calls memory service
    request = ServiceRequest(
        request_id="inter-service-1",
        service_name="memory-service",
        endpoint="/retrieve",
        body={"context": "previous conversation"}
    )
    
    response = system.service_to_service_call(
        "reasoning-service",
        "memory-service",
        request
    )
    
    # Example 4: Auto-scaling
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Service Scaling")
    print("=" * 70)
    
    # Scale up reasoning service
    system.scale_service("reasoning-service", target_instances=5)
    
    # Check health after scaling
    health = system.get_system_health()
    print(f"\nAfter scaling:")
    print(f"  Total instances: {health['total_instances']}")
    
    # Scale down
    system.scale_service("reasoning-service", target_instances=2)


def demonstrate_architecture_benefits():
    """Show benefits of microservice architecture"""
    print("\n" + "=" * 70)
    print("MICROSERVICE ARCHITECTURE BENEFITS")
    print("=" * 70)
    
    print("\n✓ Key Advantages:")
    print("\n1. INDEPENDENT SCALABILITY:")
    print("   - Scale services independently based on load")
    print("   - Cost-effective resource utilization")
    print("   - Horizontal scaling for high demand services")
    
    print("\n2. FAULT ISOLATION:")
    print("   - Failures contained to individual services")
    print("   - System remains partially operational")
    print("   - Easier debugging and recovery")
    
    print("\n3. INDEPENDENT DEPLOYMENT:")
    print("   - Deploy services without system downtime")
    print("   - Faster release cycles")
    print("   - Canary deployments and A/B testing")
    
    print("\n4. TECHNOLOGY DIVERSITY:")
    print("   - Different languages/frameworks per service")
    print("   - Best tool for each job")
    print("   - Easier technology upgrades")
    
    print("\n5. TEAM AUTONOMY:")
    print("   - Teams own services end-to-end")
    print("   - Parallel development")
    print("   - Clear boundaries and APIs")


if __name__ == "__main__":
    demonstrate_microservice_architecture()
    demonstrate_architecture_benefits()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Microservices enable independent scaling and deployment
2. Service registry provides dynamic service discovery
3. API gateway handles routing and cross-cutting concerns
4. Service mesh adds resilience patterns
5. Load balancing distributes traffic efficiently

Best Practices:
- Design services around business capabilities
- Implement health checks for all services
- Use circuit breakers for fault tolerance
- Implement proper logging and monitoring
- Version APIs carefully
- Use asynchronous communication where possible
- Implement distributed tracing
- Plan for eventual consistency
    """)
