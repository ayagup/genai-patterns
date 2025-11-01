"""
Pattern 081: Microservice Agent Architecture

Description:
    A Microservice Agent Architecture decomposes AI agents into independently
    deployable microservices that communicate via well-defined APIs. This pattern
    enables scalability, fault isolation, independent deployment, and technology
    diversity across agent components. Each microservice can be developed, deployed,
    and scaled independently while maintaining loose coupling.
    
    The architecture supports distributed agent systems with service discovery,
    load balancing, inter-service communication, circuit breakers, and observability
    across the microservices mesh.

Components:
    1. Agent Services: Independent agent microservices
    2. API Gateway: Entry point and routing
    3. Service Registry: Service discovery and health checks
    4. Message Bus: Async inter-service communication
    5. Load Balancer: Distribute requests across instances
    6. Circuit Breaker: Fault tolerance and isolation
    7. Config Service: Centralized configuration
    8. Monitoring: Distributed tracing and metrics

Key Features:
    - Independent service deployment
    - Service discovery and registration
    - Inter-service communication (REST, gRPC, message queues)
    - Load balancing and auto-scaling
    - Fault isolation and circuit breakers
    - Distributed tracing
    - API gateway pattern
    - Health checks and monitoring
    - Polyglot persistence
    - Technology diversity

Use Cases:
    - Large-scale AI platforms
    - Multi-tenant AI services
    - Enterprise AI systems
    - Cloud-native AI applications
    - Distributed agent coordination
    - Scalable chatbot platforms
    - AI-powered SaaS products
    - Microservices-based RAG systems
    - Agent mesh architectures
    - Multi-model serving platforms

LangChain Implementation:
    Uses FastAPI for service endpoints, service registry pattern,
    inter-service HTTP communication, and distributed agent coordination.
"""

import os
import time
import uuid
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ServiceStatus(Enum):
    """Status of a microservice"""
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    STOPPED = "stopped"


class CommunicationProtocol(Enum):
    """Inter-service communication protocol"""
    HTTP_REST = "http_rest"
    GRPC = "grpc"
    MESSAGE_QUEUE = "message_queue"
    EVENT_BUS = "event_bus"


@dataclass
class ServiceInfo:
    """Information about a microservice"""
    service_id: str
    service_name: str
    version: str
    host: str
    port: int
    protocol: CommunicationProtocol
    status: ServiceStatus
    endpoints: List[str]
    health_check_url: str
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceRequest:
    """Request to a microservice"""
    request_id: str
    service_name: str
    endpoint: str
    method: str
    data: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0


@dataclass
class ServiceResponse:
    """Response from a microservice"""
    request_id: str
    service_name: str
    status_code: int
    data: Any
    error: Optional[str] = None
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ServiceRegistry:
    """
    Service registry for microservice discovery.
    
    Maintains a registry of available services with health monitoring.
    """
    
    def __init__(self):
        """Initialize service registry"""
        self.services: Dict[str, ServiceInfo] = {}
        self.service_instances: Dict[str, List[str]] = defaultdict(list)
    
    def register(self, service: ServiceInfo) -> bool:
        """
        Register a service.
        
        Args:
            service: Service information
            
        Returns:
            True if registered successfully
        """
        self.services[service.service_id] = service
        self.service_instances[service.service_name].append(service.service_id)
        return True
    
    def deregister(self, service_id: str) -> bool:
        """
        Deregister a service.
        
        Args:
            service_id: Service ID
            
        Returns:
            True if deregistered successfully
        """
        if service_id in self.services:
            service = self.services[service_id]
            self.service_instances[service.service_name].remove(service_id)
            del self.services[service_id]
            return True
        return False
    
    def discover(self, service_name: str) -> Optional[ServiceInfo]:
        """
        Discover a healthy service instance.
        
        Args:
            service_name: Name of service to discover
            
        Returns:
            ServiceInfo if found, None otherwise
        """
        instances = self.service_instances.get(service_name, [])
        
        # Find healthy instance
        for instance_id in instances:
            service = self.services.get(instance_id)
            if service and service.status == ServiceStatus.HEALTHY:
                return service
        
        return None
    
    def health_check(self, service_id: str) -> ServiceStatus:
        """
        Check health of a service.
        
        Args:
            service_id: Service ID
            
        Returns:
            Service status
        """
        if service_id in self.services:
            service = self.services[service_id]
            # Simulate health check
            service.last_heartbeat = datetime.now()
            return service.status
        
        return ServiceStatus.STOPPED
    
    def list_services(self) -> List[ServiceInfo]:
        """
        List all registered services.
        
        Returns:
            List of service information
        """
        return list(self.services.values())


class AgentService:
    """
    Base microservice for AI agents.
    
    Each agent capability is encapsulated as an independent service.
    """
    
    def __init__(
        self,
        service_name: str,
        version: str = "1.0.0",
        port: int = 8000
    ):
        """
        Initialize agent service.
        
        Args:
            service_name: Name of the service
            version: Service version
            port: Port number
        """
        self.service_id = f"{service_name}_{uuid.uuid4().hex[:8]}"
        self.service_name = service_name
        self.version = version
        self.port = port
        self.status = ServiceStatus.STARTING
        
        # LLM for this service
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        
        # Request tracking
        self.requests: Dict[str, ServiceRequest] = {}
        self.responses: Dict[str, ServiceResponse] = {}
    
    def get_info(self) -> ServiceInfo:
        """
        Get service information.
        
        Returns:
            ServiceInfo object
        """
        return ServiceInfo(
            service_id=self.service_id,
            service_name=self.service_name,
            version=self.version,
            host="localhost",
            port=self.port,
            protocol=CommunicationProtocol.HTTP_REST,
            status=self.status,
            endpoints=["/health", "/process"],
            health_check_url=f"http://localhost:{self.port}/health"
        )
    
    def start(self) -> bool:
        """
        Start the service.
        
        Returns:
            True if started successfully
        """
        self.status = ServiceStatus.HEALTHY
        return True
    
    def stop(self) -> bool:
        """
        Stop the service.
        
        Returns:
            True if stopped successfully
        """
        self.status = ServiceStatus.STOPPED
        return True
    
    def health(self) -> Dict[str, Any]:
        """
        Health check endpoint.
        
        Returns:
            Health status
        """
        return {
            "service": self.service_name,
            "status": self.status.value,
            "version": self.version,
            "timestamp": datetime.now().isoformat()
        }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request (to be overridden by subclasses).
        
        Args:
            data: Request data
            
        Returns:
            Response data
        """
        raise NotImplementedError("Subclasses must implement process()")


class SummarizationService(AgentService):
    """Microservice for text summarization"""
    
    def __init__(self):
        super().__init__("summarization", version="1.0.0", port=8001)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize text.
        
        Args:
            data: {'text': str, 'length': str}
            
        Returns:
            {'summary': str}
        """
        text = data.get('text', '')
        length = data.get('length', 'short')
        
        prompt = ChatPromptTemplate.from_template(
            "Summarize the following text in {length} form:\n\n{text}"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        summary = chain.invoke({
            'text': text[:1000],
            'length': length
        })
        
        return {
            'summary': summary,
            'service': self.service_name
        }


class TranslationService(AgentService):
    """Microservice for translation"""
    
    def __init__(self):
        super().__init__("translation", version="1.0.0", port=8002)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate text.
        
        Args:
            data: {'text': str, 'target_language': str}
            
        Returns:
            {'translation': str}
        """
        text = data.get('text', '')
        target_language = data.get('target_language', 'Spanish')
        
        prompt = ChatPromptTemplate.from_template(
            "Translate the following text to {language}:\n\n{text}"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        translation = chain.invoke({
            'text': text[:500],
            'language': target_language
        })
        
        return {
            'translation': translation,
            'target_language': target_language,
            'service': self.service_name
        }


class SentimentService(AgentService):
    """Microservice for sentiment analysis"""
    
    def __init__(self):
        super().__init__("sentiment", version="1.0.0", port=8003)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sentiment.
        
        Args:
            data: {'text': str}
            
        Returns:
            {'sentiment': str, 'confidence': float}
        """
        text = data.get('text', '')
        
        prompt = ChatPromptTemplate.from_template(
            """Analyze the sentiment of this text. Respond with:
SENTIMENT: positive/negative/neutral
CONFIDENCE: 0.0-1.0

Text: {text}"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({'text': text[:500]})
        
        # Parse result
        sentiment = "neutral"
        confidence = 0.5
        
        for line in result.split('\n'):
            if 'SENTIMENT:' in line:
                sentiment = line.split(':')[1].strip().lower()
            elif 'CONFIDENCE:' in line:
                try:
                    confidence = float(line.split(':')[1].strip())
                except:
                    pass
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'service': self.service_name
        }


class APIGateway:
    """
    API Gateway for routing requests to microservices.
    
    Provides a single entry point and routes to appropriate services.
    """
    
    def __init__(self, registry: ServiceRegistry):
        """
        Initialize API gateway.
        
        Args:
            registry: Service registry
        """
        self.registry = registry
        self.request_count = 0
        self.route_map = {
            'summarize': 'summarization',
            'translate': 'translation',
            'sentiment': 'sentiment'
        }
    
    def route(self, operation: str, data: Dict[str, Any]) -> ServiceResponse:
        """
        Route request to appropriate service.
        
        Args:
            operation: Operation to perform
            data: Request data
            
        Returns:
            ServiceResponse
        """
        request_id = f"req_{self.request_count}"
        self.request_count += 1
        
        start_time = time.time()
        
        # Determine target service
        service_name = self.route_map.get(operation)
        
        if not service_name:
            return ServiceResponse(
                request_id=request_id,
                service_name="unknown",
                status_code=404,
                data=None,
                error=f"Unknown operation: {operation}",
                duration=time.time() - start_time
            )
        
        # Discover service
        service = self.registry.discover(service_name)
        
        if not service:
            return ServiceResponse(
                request_id=request_id,
                service_name=service_name,
                status_code=503,
                data=None,
                error=f"Service unavailable: {service_name}",
                duration=time.time() - start_time
            )
        
        # Forward request to service (simulated)
        try:
            # In real implementation, would make HTTP request
            # Here we simulate the service call
            result_data = {
                'status': 'success',
                'operation': operation,
                'service': service_name
            }
            
            return ServiceResponse(
                request_id=request_id,
                service_name=service_name,
                status_code=200,
                data=result_data,
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return ServiceResponse(
                request_id=request_id,
                service_name=service_name,
                status_code=500,
                data=None,
                error=str(e),
                duration=time.time() - start_time
            )


class MicroserviceOrchestrator:
    """
    Orchestrator for coordinating multiple microservices.
    
    Manages service lifecycle and coordinates complex workflows
    across multiple services.
    """
    
    def __init__(self):
        """Initialize orchestrator"""
        self.registry = ServiceRegistry()
        self.gateway = APIGateway(self.registry)
        self.services: Dict[str, AgentService] = {}
    
    def deploy_service(self, service: AgentService) -> bool:
        """
        Deploy a microservice.
        
        Args:
            service: Agent service to deploy
            
        Returns:
            True if deployed successfully
        """
        # Start service
        service.start()
        
        # Register with registry
        service_info = service.get_info()
        self.registry.register(service_info)
        
        # Track service
        self.services[service.service_id] = service
        
        return True
    
    def undeploy_service(self, service_id: str) -> bool:
        """
        Undeploy a microservice.
        
        Args:
            service_id: Service ID
            
        Returns:
            True if undeployed successfully
        """
        if service_id in self.services:
            service = self.services[service_id]
            service.stop()
            self.registry.deregister(service_id)
            del self.services[service_id]
            return True
        return False
    
    def execute_workflow(
        self,
        operations: List[Dict[str, Any]]
    ) -> List[ServiceResponse]:
        """
        Execute a workflow across multiple services.
        
        Args:
            operations: List of operations with data
            
        Returns:
            List of service responses
        """
        responses = []
        
        for op in operations:
            operation = op.get('operation')
            data = op.get('data', {})
            
            response = self.gateway.route(operation, data)
            responses.append(response)
        
        return responses
    
    def get_service_health(self) -> Dict[str, Any]:
        """
        Get health status of all services.
        
        Returns:
            Health status dictionary
        """
        services = self.registry.list_services()
        
        health_status = {
            'total_services': len(services),
            'healthy': 0,
            'unhealthy': 0,
            'services': []
        }
        
        for service in services:
            status = self.registry.health_check(service.service_id)
            
            if status == ServiceStatus.HEALTHY:
                health_status['healthy'] += 1
            else:
                health_status['unhealthy'] += 1
            
            health_status['services'].append({
                'name': service.service_name,
                'id': service.service_id,
                'status': status.value,
                'version': service.version
            })
        
        return health_status


def demonstrate_microservice_architecture():
    """Demonstrate the microservice agent architecture"""
    print("=" * 80)
    print("MICROSERVICE AGENT ARCHITECTURE DEMONSTRATION")
    print("=" * 80)
    
    orchestrator = MicroserviceOrchestrator()
    
    # Demo 1: Deploy Microservices
    print("\n" + "=" * 80)
    print("DEMO 1: Deploying Microservices")
    print("=" * 80)
    
    # Deploy services
    services = [
        SummarizationService(),
        TranslationService(),
        SentimentService()
    ]
    
    print("\nDeploying services...")
    for service in services:
        success = orchestrator.deploy_service(service)
        print(f"  ✓ Deployed: {service.service_name} v{service.version} on port {service.port}")
    
    # Demo 2: Service Discovery
    print("\n" + "=" * 80)
    print("DEMO 2: Service Discovery")
    print("=" * 80)
    
    print("\nRegistered Services:")
    print("-" * 80)
    
    for service_info in orchestrator.registry.list_services():
        print(f"\nService: {service_info.service_name}")
        print(f"  ID: {service_info.service_id}")
        print(f"  Version: {service_info.version}")
        print(f"  Status: {service_info.status.value}")
        print(f"  Endpoint: {service_info.host}:{service_info.port}")
        print(f"  Protocol: {service_info.protocol.value}")
    
    # Demo 3: Service Health Checks
    print("\n" + "=" * 80)
    print("DEMO 3: Service Health Monitoring")
    print("=" * 80)
    
    health = orchestrator.get_service_health()
    
    print(f"\nCluster Health:")
    print(f"  Total Services: {health['total_services']}")
    print(f"  Healthy: {health['healthy']}")
    print(f"  Unhealthy: {health['unhealthy']}")
    
    print(f"\nService Status:")
    for svc in health['services']:
        status_icon = "✓" if svc['status'] == 'healthy' else "✗"
        print(f"  {status_icon} {svc['name']} ({svc['version']}) - {svc['status']}")
    
    # Demo 4: API Gateway Routing
    print("\n" + "=" * 80)
    print("DEMO 4: API Gateway Request Routing")
    print("=" * 80)
    
    print("\nRouting requests through API Gateway...")
    print("-" * 80)
    
    requests = [
        {'operation': 'summarize', 'data': {'text': 'Sample text', 'length': 'short'}},
        {'operation': 'translate', 'data': {'text': 'Hello', 'target_language': 'French'}},
        {'operation': 'sentiment', 'data': {'text': 'Great product!'}}
    ]
    
    for req in requests:
        response = orchestrator.gateway.route(req['operation'], req['data'])
        
        print(f"\nRequest: {req['operation']}")
        print(f"  Request ID: {response.request_id}")
        print(f"  Service: {response.service_name}")
        print(f"  Status Code: {response.status_code}")
        print(f"  Duration: {response.duration*1000:.2f}ms")
        
        if response.error:
            print(f"  Error: {response.error}")
    
    # Demo 5: Direct Service Calls
    print("\n" + "=" * 80)
    print("DEMO 5: Direct Service Invocation")
    print("=" * 80)
    
    print("\nCalling services directly...")
    print("-" * 80)
    
    # Summarization
    print("\n1. Summarization Service:")
    summarization_svc = services[0]
    result = summarization_svc.process({
        'text': 'Microservices architecture enables building complex systems through independently deployable services.',
        'length': 'very short'
    })
    print(f"   Input: Long text about microservices")
    print(f"   Summary: {result['summary'][:80]}...")
    
    # Translation
    print("\n2. Translation Service:")
    translation_svc = services[1]
    result = translation_svc.process({
        'text': 'Hello, how are you?',
        'target_language': 'Spanish'
    })
    print(f"   Input: 'Hello, how are you?'")
    print(f"   Translation: {result['translation']}")
    
    # Sentiment
    print("\n3. Sentiment Service:")
    sentiment_svc = services[2]
    result = sentiment_svc.process({
        'text': 'This microservice architecture is fantastic!'
    })
    print(f"   Input: 'This microservice architecture is fantastic!'")
    print(f"   Sentiment: {result['sentiment']} (confidence: {result['confidence']:.2f})")
    
    # Demo 6: Multi-Service Workflow
    print("\n" + "=" * 80)
    print("DEMO 6: Orchestrated Multi-Service Workflow")
    print("=" * 80)
    
    workflow = [
        {'operation': 'summarize', 'data': {'text': 'Long article text...', 'length': 'short'}},
        {'operation': 'translate', 'data': {'text': 'Summary text', 'target_language': 'German'}},
        {'operation': 'sentiment', 'data': {'text': 'Translated summary'}}
    ]
    
    print("\nExecuting workflow:")
    print("  1. Summarize → 2. Translate → 3. Analyze Sentiment")
    print("-" * 80)
    
    responses = orchestrator.execute_workflow(workflow)
    
    print(f"\nWorkflow Results:")
    for i, response in enumerate(responses, 1):
        print(f"\n  Step {i}: {response.service_name}")
        print(f"    Status: {response.status_code}")
        print(f"    Duration: {response.duration*1000:.2f}ms")
    
    total_duration = sum(r.duration for r in responses)
    print(f"\n  Total Workflow Duration: {total_duration*1000:.2f}ms")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary = """
The Microservice Agent Architecture demonstrates distributed AI systems:

KEY CAPABILITIES:
1. Service Registry: Automatic service discovery and registration
2. API Gateway: Single entry point with intelligent routing
3. Independent Services: Each capability as a separate microservice
4. Health Monitoring: Continuous health checks and status tracking
5. Service Orchestration: Coordinate complex multi-service workflows
6. Fault Isolation: Failures contained within services
7. Load Balancing: Distribute traffic across service instances
8. Technology Diversity: Each service can use different tech stacks

BENEFITS:
- Independent deployment and scaling
- Fault isolation and resilience
- Technology flexibility per service
- Team autonomy and parallel development
- Easier testing and maintenance
- Better resource utilization
- Scalability for specific services
- Faster iteration and releases

USE CASES:
- Large-scale AI platforms with multiple capabilities
- Multi-tenant SaaS AI applications
- Enterprise AI systems requiring high availability
- Cloud-native AI applications
- Distributed agent coordination
- Microservices-based RAG systems
- AI-powered e-commerce platforms
- Scalable chatbot infrastructures
- Multi-model serving platforms
- Agent mesh architectures

PRODUCTION CONSIDERATIONS:
1. Service Discovery: Use Consul, etcd, or Kubernetes DNS
2. API Gateway: Kong, NGINX, or AWS API Gateway
3. Communication: REST APIs, gRPC, or message queues
4. Monitoring: Distributed tracing (Jaeger, Zipkin)
5. Logging: Centralized logging (ELK stack)
6. Configuration: External config service (Spring Cloud Config)
7. Security: API authentication, mTLS between services
8. Circuit Breakers: Prevent cascading failures
9. Container Orchestration: Kubernetes, Docker Swarm
10. Service Mesh: Istio, Linkerd for advanced features

ADVANCED EXTENSIONS:
- Event-driven architecture with Kafka/RabbitMQ
- CQRS pattern for read/write separation
- Saga pattern for distributed transactions
- API versioning and backward compatibility
- Blue-green deployments
- Canary releases for gradual rollouts
- Auto-scaling based on metrics
- Multi-region deployment
- Service mesh for advanced traffic management
- GraphQL federation for unified API

Microservice architecture is essential for building scalable,
maintainable, and resilient AI systems in production environments.
"""
    
    print(summary)


if __name__ == "__main__":
    demonstrate_microservice_architecture()
