"""
Pattern 122: Service Mesh Pattern

Description:
    Infrastructure layer for agent-to-agent communication with service
    discovery, load balancing, and observability.

Components:
    - Service registry
    - Load balancer
    - Circuit breaker
    - Observability layer

Use Cases:
    - Large-scale distributed agents
    - Microservices architecture
    - Reliable communication

LangChain Implementation:
    Implements service mesh concepts for agent communication management.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import random
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ServiceStatus(Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ServiceInstance:
    """Represents a service instance."""
    id: str
    name: str
    host: str
    port: int
    status: ServiceStatus = ServiceStatus.HEALTHY
    load: int = 0
    max_capacity: int = 100
    response_times: List[float] = None
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []


class ServiceRegistry:
    """Service discovery and registration."""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceInstance]] = {}
        
    def register(self, service_name: str, instance: ServiceInstance):
        """Register a service instance."""
        if service_name not in self.services:
            self.services[service_name] = []
        self.services[service_name].append(instance)
        print(f"✓ Registered {service_name} instance: {instance.id}")
    
    def deregister(self, service_name: str, instance_id: str):
        """Deregister a service instance."""
        if service_name in self.services:
            self.services[service_name] = [
                i for i in self.services[service_name] if i.id != instance_id
            ]
            print(f"✓ Deregistered {service_name} instance: {instance_id}")
    
    def discover(self, service_name: str) -> List[ServiceInstance]:
        """Discover healthy instances of a service."""
        instances = self.services.get(service_name, [])
        return [i for i in instances if i.status == ServiceStatus.HEALTHY]
    
    def get_all_services(self) -> List[str]:
        """Get list of all registered services."""
        return list(self.services.keys())


class LoadBalancer:
    """Load balancing for service requests."""
    
    def __init__(self, strategy: str = "least_loaded"):
        self.strategy = strategy
        
    def select_instance(self, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """Select an instance based on load balancing strategy."""
        if not instances:
            return None
        
        if self.strategy == "round_robin":
            return random.choice(instances)
        
        elif self.strategy == "least_loaded":
            return min(instances, key=lambda i: i.load)
        
        elif self.strategy == "best_performance":
            # Choose instance with best average response time
            instances_with_times = [
                i for i in instances if i.response_times
            ]
            if instances_with_times:
                return min(
                    instances_with_times,
                    key=lambda i: sum(i.response_times) / len(i.response_times)
                )
            return instances[0]
        
        return instances[0]


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 3, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures: Dict[str, int] = {}
        self.open_circuits: Dict[str, datetime] = {}
        
    def record_failure(self, service_id: str):
        """Record a service failure."""
        self.failures[service_id] = self.failures.get(service_id, 0) + 1
        
        if self.failures[service_id] >= self.failure_threshold:
            self.open_circuits[service_id] = datetime.now()
            print(f"⚠ Circuit opened for {service_id}")
    
    def record_success(self, service_id: str):
        """Record a successful call."""
        self.failures[service_id] = 0
        if service_id in self.open_circuits:
            del self.open_circuits[service_id]
            print(f"✓ Circuit closed for {service_id}")
    
    def is_open(self, service_id: str) -> bool:
        """Check if circuit is open."""
        if service_id in self.open_circuits:
            # Check if timeout has passed
            elapsed = (datetime.now() - self.open_circuits[service_id]).seconds
            if elapsed > self.timeout:
                del self.open_circuits[service_id]
                return False
            return True
        return False


class ServiceMesh:
    """Service mesh for agent communication."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.registry = ServiceRegistry()
        self.load_balancer = LoadBalancer(strategy="least_loaded")
        self.circuit_breaker = CircuitBreaker()
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.request_log: List[Dict[str, Any]] = []
        
    def call_service(self, service_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a service call through the mesh."""
        print(f"\n🔄 Calling service: {service_name}")
        
        # Discover instances
        instances = self.registry.discover(service_name)
        
        if not instances:
            print(f"  ✗ No healthy instances found")
            return {"error": "ServiceUnavailable"}
        
        # Select instance via load balancer
        instance = self.load_balancer.select_instance(instances)
        
        if not instance:
            return {"error": "NoInstanceAvailable"}
        
        # Check circuit breaker
        if self.circuit_breaker.is_open(instance.id):
            print(f"  ⚠ Circuit open for {instance.id}, trying another instance")
            # Try another instance
            instances.remove(instance)
            if instances:
                instance = self.load_balancer.select_instance(instances)
            else:
                return {"error": "AllCircuitsOpen"}
        
        print(f"  → Routing to instance: {instance.id} (load: {instance.load})")
        
        # Simulate service call
        try:
            start_time = datetime.now()
            
            # Simulate processing
            instance.load += 10
            response_time = random.uniform(0.1, 0.5)
            instance.response_times.append(response_time)
            
            # Simulate occasional failures
            if random.random() < 0.1:  # 10% failure rate
                raise Exception("ServiceError")
            
            # Log request
            self.request_log.append({
                "service": service_name,
                "instance": instance.id,
                "timestamp": start_time.isoformat(),
                "response_time": response_time,
                "status": "success"
            })
            
            # Record success
            self.circuit_breaker.record_success(instance.id)
            instance.load -= 10
            
            print(f"  ✓ Success (response time: {response_time:.3f}s)")
            
            return {
                "success": True,
                "data": f"Processed by {instance.id}",
                "response_time": response_time
            }
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            
            # Record failure
            self.circuit_breaker.record_failure(instance.id)
            instance.status = ServiceStatus.DEGRADED
            instance.load -= 10
            
            self.request_log.append({
                "service": service_name,
                "instance": instance.id,
                "timestamp": datetime.now().isoformat(),
                "status": "failed",
                "error": str(e)
            })
            
            return {"error": str(e)}
    
    def get_observability_report(self) -> str:
        """Generate observability report."""
        report_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze service mesh metrics and provide insights."),
            ("user", """Request Log:
{log}

Service Registry:
{registry}

Provide:
1. Service health summary
2. Performance metrics
3. Bottlenecks identified
4. Recommendations""")
        ])
        
        log_summary = "\n".join([
            f"{r['service']}: {r['status']} ({r.get('response_time', 'N/A')}s)"
            for r in self.request_log[-20:]
        ])
        
        registry_summary = "\n".join([
            f"{name}: {len(instances)} instances"
            for name, instances in self.registry.services.items()
        ])
        
        chain = report_prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "log": log_summary,
            "registry": registry_summary
        })


def demonstrate_service_mesh():
    """Demonstrate service mesh pattern."""
    print("=== Service Mesh Pattern ===\n")
    
    mesh = ServiceMesh()
    
    # Register services
    print("1. Registering Service Instances")
    print("-" * 50)
    
    # Data processing service
    mesh.registry.register("data-processor", ServiceInstance(
        id="dp-1", name="data-processor", host="10.0.1.1", port=8001
    ))
    mesh.registry.register("data-processor", ServiceInstance(
        id="dp-2", name="data-processor", host="10.0.1.2", port=8001
    ))
    mesh.registry.register("data-processor", ServiceInstance(
        id="dp-3", name="data-processor", host="10.0.1.3", port=8001
    ))
    
    # Analytics service
    mesh.registry.register("analytics", ServiceInstance(
        id="an-1", name="analytics", host="10.0.2.1", port=8002
    ))
    mesh.registry.register("analytics", ServiceInstance(
        id="an-2", name="analytics", host="10.0.2.2", port=8002
    ))
    
    print()
    
    # Make service calls
    print("2. Making Service Calls Through Mesh")
    print("-" * 50)
    
    for i in range(10):
        result = mesh.call_service("data-processor", {"data": f"request_{i}"})
    
    for i in range(5):
        result = mesh.call_service("analytics", {"query": f"query_{i}"})
    
    # Observability
    print("\n3. Service Mesh Observability")
    print("-" * 50)
    
    print(f"Total requests: {len(mesh.request_log)}")
    successful = sum(1 for r in mesh.request_log if r['status'] == 'success')
    print(f"Successful: {successful}")
    print(f"Failed: {len(mesh.request_log) - successful}")
    print(f"Success rate: {successful/len(mesh.request_log)*100:.1f}%")
    
    # Generate report
    print("\n4. Detailed Observability Report")
    print("-" * 50)
    report = mesh.get_observability_report()
    print(report)
    
    print("\n=== Summary ===")
    print("Service mesh demonstrated with:")
    print("- Service discovery and registration")
    print("- Load balancing (least loaded strategy)")
    print("- Circuit breaker pattern")
    print("- Request routing")
    print("- Health monitoring")
    print("- Observability and metrics")


if __name__ == "__main__":
    demonstrate_service_mesh()
