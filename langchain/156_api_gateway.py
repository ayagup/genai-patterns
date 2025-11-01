"""
Pattern 156: API Gateway Pattern

Description:
    Centralized gateway for managing API requests to multiple agents/services.
    Provides routing, authentication, rate limiting, monitoring, and unified
    interface for diverse backend services.

Components:
    - Request router
    - Authentication & authorization
    - Rate limiting
    - Load balancing
    - Monitoring & logging
    - Response transformation
    - Error handling

Use Cases:
    - Multi-agent systems
    - Microservices architecture
    - API versioning
    - Security enforcement
    - Traffic management

Benefits:
    - Centralized control
    - Security enforcement
    - Traffic management
    - Simplified client interface
    - Monitoring visibility

Trade-offs:
    - Single point of failure
    - Added latency
    - Complexity overhead
    - Gateway bottleneck potential

LangChain Implementation:
    Uses routing logic, rate limiting, authentication checks,
    and multiple LLM backends with unified interface
"""

import os
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class ServiceType(Enum):
    """Types of backend services"""
    CHAT = "chat"
    COMPLETION = "completion"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"


class AuthLevel(Enum):
    """Authentication levels"""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    PREMIUM = "premium"
    ADMIN = "admin"


@dataclass
class Route:
    """API route configuration"""
    path: str
    service_type: ServiceType
    handler: Callable
    auth_level: AuthLevel = AuthLevel.PUBLIC
    rate_limit: int = 60  # requests per minute
    timeout: float = 30.0  # seconds


@dataclass
class ServiceConfig:
    """Backend service configuration"""
    name: str
    service_type: ServiceType
    model_name: str
    enabled: bool = True
    weight: int = 1  # For load balancing
    max_concurrent: int = 10


@dataclass
class Request:
    """Gateway request"""
    path: str
    method: str
    payload: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    user_id: Optional[str] = None
    api_key: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Response:
    """Gateway response"""
    status_code: int
    data: Any
    headers: Dict[str, str] = field(default_factory=dict)
    execution_time: float = 0.0
    service_name: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RateLimitInfo:
    """Rate limit tracking"""
    requests: List[datetime] = field(default_factory=list)
    limit: int = 60
    window: int = 60  # seconds


class APIGateway:
    """Centralized API gateway for agent services"""
    
    def __init__(self):
        """Initialize API gateway"""
        self.routes: Dict[str, Route] = {}
        self.services: Dict[str, ServiceConfig] = {}
        self.rate_limits: Dict[str, RateLimitInfo] = defaultdict(
            lambda: RateLimitInfo()
        )
        self.api_keys: Dict[str, AuthLevel] = {}
        self.request_log: List[Dict[str, Any]] = []
        
        # Initialize services
        self._initialize_services()
        
        # Set up routes
        self._setup_routes()
    
    def _initialize_services(self):
        """Initialize backend services"""
        self.services = {
            "chat_service": ServiceConfig(
                name="chat_service",
                service_type=ServiceType.CHAT,
                model_name="gpt-3.5-turbo"
            ),
            "analysis_service": ServiceConfig(
                name="analysis_service",
                service_type=ServiceType.ANALYSIS,
                model_name="gpt-3.5-turbo"
            ),
            "translation_service": ServiceConfig(
                name="translation_service",
                service_type=ServiceType.TRANSLATION,
                model_name="gpt-3.5-turbo"
            ),
        }
        
        # Initialize LLMs for services
        self.llms = {
            name: ChatOpenAI(model=config.model_name, temperature=0.7)
            for name, config in self.services.items()
        }
    
    def _setup_routes(self):
        """Set up API routes"""
        self.routes = {
            "/chat": Route(
                path="/chat",
                service_type=ServiceType.CHAT,
                handler=self._handle_chat,
                auth_level=AuthLevel.AUTHENTICATED,
                rate_limit=30
            ),
            "/analyze": Route(
                path="/analyze",
                service_type=ServiceType.ANALYSIS,
                handler=self._handle_analysis,
                auth_level=AuthLevel.PREMIUM,
                rate_limit=10
            ),
            "/translate": Route(
                path="/translate",
                service_type=ServiceType.TRANSLATION,
                handler=self._handle_translation,
                auth_level=AuthLevel.AUTHENTICATED,
                rate_limit=20
            ),
            "/summarize": Route(
                path="/summarize",
                service_type=ServiceType.SUMMARIZATION,
                handler=self._handle_summarization,
                auth_level=AuthLevel.AUTHENTICATED,
                rate_limit=15
            ),
        }
    
    def register_api_key(self, api_key: str, auth_level: AuthLevel):
        """Register an API key with authentication level"""
        self.api_keys[api_key] = auth_level
    
    def authenticate(self, request: Request) -> tuple[bool, Optional[str]]:
        """Authenticate request"""
        # Get route
        route = self.routes.get(request.path)
        if not route:
            return False, "Route not found"
        
        # Check authentication level
        if route.auth_level == AuthLevel.PUBLIC:
            return True, None
        
        # Check API key
        if not request.api_key:
            return False, "API key required"
        
        user_level = self.api_keys.get(request.api_key)
        if not user_level:
            return False, "Invalid API key"
        
        # Check authorization level
        required_levels = {
            AuthLevel.AUTHENTICATED: [AuthLevel.AUTHENTICATED, AuthLevel.PREMIUM, AuthLevel.ADMIN],
            AuthLevel.PREMIUM: [AuthLevel.PREMIUM, AuthLevel.ADMIN],
            AuthLevel.ADMIN: [AuthLevel.ADMIN]
        }
        
        if user_level not in required_levels.get(route.auth_level, []):
            return False, f"Insufficient permissions. Requires {route.auth_level.value}"
        
        return True, None
    
    def check_rate_limit(self, request: Request) -> tuple[bool, Optional[str]]:
        """Check if request exceeds rate limit"""
        route = self.routes.get(request.path)
        if not route:
            return True, None
        
        # Get rate limit info for user
        key = request.api_key or request.user_id or "anonymous"
        rate_info = self.rate_limits[key]
        
        # Update limit if different from current
        if rate_info.limit != route.rate_limit:
            rate_info.limit = route.rate_limit
        
        # Clean old requests outside window
        now = datetime.now()
        cutoff = now - timedelta(seconds=rate_info.window)
        rate_info.requests = [
            req_time for req_time in rate_info.requests
            if req_time > cutoff
        ]
        
        # Check limit
        if len(rate_info.requests) >= rate_info.limit:
            return False, f"Rate limit exceeded. Max {rate_info.limit} requests per minute"
        
        # Add current request
        rate_info.requests.append(now)
        return True, None
    
    def route_request(self, request: Request) -> Response:
        """Route and process request through gateway"""
        start_time = time.time()
        
        # Log request
        self._log_request(request)
        
        # Authenticate
        auth_ok, auth_error = self.authenticate(request)
        if not auth_ok:
            return Response(
                status_code=401,
                data=None,
                error=auth_error,
                execution_time=time.time() - start_time
            )
        
        # Check rate limit
        rate_ok, rate_error = self.check_rate_limit(request)
        if not rate_ok:
            return Response(
                status_code=429,
                data=None,
                error=rate_error,
                execution_time=time.time() - start_time
            )
        
        # Get route
        route = self.routes.get(request.path)
        if not route:
            return Response(
                status_code=404,
                data=None,
                error=f"Route not found: {request.path}",
                execution_time=time.time() - start_time
            )
        
        # Select service
        service = self._select_service(route.service_type)
        if not service:
            return Response(
                status_code=503,
                data=None,
                error="No available service",
                execution_time=time.time() - start_time
            )
        
        # Execute request
        try:
            result = route.handler(request, service)
            execution_time = time.time() - start_time
            
            return Response(
                status_code=200,
                data=result,
                execution_time=execution_time,
                service_name=service.name
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return Response(
                status_code=500,
                data=None,
                error=str(e),
                execution_time=execution_time,
                service_name=service.name
            )
    
    def _select_service(self, service_type: ServiceType) -> Optional[ServiceConfig]:
        """Select service based on type and load balancing"""
        # Find services matching type
        matching = [
            s for s in self.services.values()
            if s.service_type == service_type and s.enabled
        ]
        
        if not matching:
            return None
        
        # Simple weighted random selection
        import random
        weights = [s.weight for s in matching]
        return random.choices(matching, weights=weights)[0]
    
    def _handle_chat(self, request: Request, service: ServiceConfig) -> str:
        """Handle chat request"""
        llm = self.llms[service.name]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            ("user", "{message}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"message": request.payload.get("message", "")})
    
    def _handle_analysis(self, request: Request, service: ServiceConfig) -> str:
        """Handle analysis request"""
        llm = self.llms[service.name]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert analyst. Provide detailed analysis."),
            ("user", "Analyze the following:\n\n{text}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"text": request.payload.get("text", "")})
    
    def _handle_translation(self, request: Request, service: ServiceConfig) -> str:
        """Handle translation request"""
        llm = self.llms[service.name]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional translator."),
            ("user", "Translate the following from {source} to {target}:\n\n{text}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "source": request.payload.get("source_lang", "auto"),
            "target": request.payload.get("target_lang", "English"),
            "text": request.payload.get("text", "")
        })
    
    def _handle_summarization(self, request: Request, service: ServiceConfig) -> str:
        """Handle summarization request"""
        llm = self.llms[service.name]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at creating concise summaries."),
            ("user", "Summarize the following in {max_words} words:\n\n{text}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({
            "max_words": request.payload.get("max_words", 100),
            "text": request.payload.get("text", "")
        })
    
    def _log_request(self, request: Request):
        """Log request for monitoring"""
        self.request_log.append({
            "timestamp": request.timestamp.isoformat(),
            "path": request.path,
            "method": request.method,
            "user_id": request.user_id,
            "api_key_hash": hash(request.api_key) if request.api_key else None
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get gateway metrics"""
        now = datetime.now()
        recent = now - timedelta(minutes=5)
        
        recent_requests = [
            log for log in self.request_log
            if datetime.fromisoformat(log["timestamp"]) > recent
        ]
        
        # Count by path
        by_path = defaultdict(int)
        for log in recent_requests:
            by_path[log["path"]] += 1
        
        # Rate limit status
        rate_status = {}
        for key, info in self.rate_limits.items():
            rate_status[key] = {
                "current": len(info.requests),
                "limit": info.limit,
                "usage_pct": len(info.requests) / info.limit * 100
            }
        
        return {
            "total_requests_5min": len(recent_requests),
            "requests_by_path": dict(by_path),
            "rate_limit_status": rate_status,
            "active_services": len([s for s in self.services.values() if s.enabled]),
            "total_services": len(self.services)
        }


def demonstrate_api_gateway():
    """Demonstrate API gateway pattern"""
    print("=" * 80)
    print("API GATEWAY PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Initialize gateway
    gateway = APIGateway()
    
    # Register API keys
    gateway.register_api_key("free_user_key", AuthLevel.AUTHENTICATED)
    gateway.register_api_key("premium_user_key", AuthLevel.PREMIUM)
    gateway.register_api_key("admin_key", AuthLevel.ADMIN)
    
    # Example 1: Successful authenticated request
    print("\n" + "="*80)
    print("EXAMPLE 1: Successful Chat Request (Authenticated)")
    print("="*80)
    request = Request(
        path="/chat",
        method="POST",
        payload={"message": "What is machine learning?"},
        api_key="free_user_key",
        user_id="user123"
    )
    response = gateway.route_request(request)
    print(f"Status: {response.status_code}")
    print(f"Service: {response.service_name}")
    print(f"Time: {response.execution_time:.2f}s")
    print(f"Response: {response.data[:200] if response.data else response.error}...")
    
    # Example 2: Premium service request
    print("\n" + "="*80)
    print("EXAMPLE 2: Analysis Request (Premium)")
    print("="*80)
    request = Request(
        path="/analyze",
        method="POST",
        payload={"text": "The market showed strong growth in Q1 with a 15% increase in sales."},
        api_key="premium_user_key"
    )
    response = gateway.route_request(request)
    print(f"Status: {response.status_code}")
    print(f"Service: {response.service_name}")
    print(f"Response: {response.data[:200] if response.data else response.error}...")
    
    # Example 3: Authentication failure
    print("\n" + "="*80)
    print("EXAMPLE 3: Authentication Failure")
    print("="*80)
    request = Request(
        path="/analyze",
        method="POST",
        payload={"text": "Test"},
        api_key="free_user_key"  # Not premium
    )
    response = gateway.route_request(request)
    print(f"Status: {response.status_code}")
    print(f"Error: {response.error}")
    
    # Example 4: Rate limiting
    print("\n" + "="*80)
    print("EXAMPLE 4: Rate Limiting")
    print("="*80)
    # Make multiple requests quickly
    for i in range(3):
        request = Request(
            path="/chat",
            method="POST",
            payload={"message": f"Request {i+1}"},
            api_key="free_user_key"
        )
        response = gateway.route_request(request)
        print(f"Request {i+1}: Status {response.status_code} - {response.error if response.error else 'Success'}")
    
    # Example 5: Translation service
    print("\n" + "="*80)
    print("EXAMPLE 5: Translation Service")
    print("="*80)
    request = Request(
        path="/translate",
        method="POST",
        payload={
            "text": "Hello, how are you?",
            "source_lang": "English",
            "target_lang": "Spanish"
        },
        api_key="free_user_key"
    )
    response = gateway.route_request(request)
    print(f"Status: {response.status_code}")
    print(f"Translation: {response.data if response.data else response.error}")
    
    # Example 6: Gateway metrics
    print("\n" + "="*80)
    print("EXAMPLE 6: Gateway Metrics")
    print("="*80)
    metrics = gateway.get_metrics()
    print("Gateway Metrics:")
    print(f"  Total requests (5 min): {metrics['total_requests_5min']}")
    print(f"  Requests by path: {metrics['requests_by_path']}")
    print(f"  Active services: {metrics['active_services']}/{metrics['total_services']}")
    
    # Example 7: Route not found
    print("\n" + "="*80)
    print("EXAMPLE 7: Route Not Found")
    print("="*80)
    request = Request(
        path="/nonexistent",
        method="POST",
        payload={},
        api_key="free_user_key"
    )
    response = gateway.route_request(request)
    print(f"Status: {response.status_code}")
    print(f"Error: {response.error}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: API Gateway Pattern Best Practices")
    print("="*80)
    print("""
1. ROUTING:
   - Centralized request routing
   - Service discovery and selection
   - Load balancing across backends
   - Flexible routing rules

2. AUTHENTICATION & AUTHORIZATION:
   - API key management
   - Multiple authorization levels
   - Fine-grained access control
   - Secure credential handling

3. RATE LIMITING:
   - Per-user rate limits
   - Per-route limits
   - Sliding window implementation
   - Graceful limit enforcement

4. MONITORING:
   - Request logging
   - Performance metrics
   - Service health tracking
   - Real-time visibility

5. ERROR HANDLING:
   - Standardized error responses
   - Proper HTTP status codes
   - Detailed error messages
   - Graceful degradation

6. SERVICE MANAGEMENT:
   - Service registration
   - Health checks
   - Dynamic service discovery
   - Version management

Benefits:
✓ Centralized control and management
✓ Security enforcement at gateway level
✓ Simplified client integration
✓ Traffic management and optimization
✓ Monitoring and observability
✓ Service abstraction

Challenges:
- Single point of failure risk
- Added network latency
- Gateway complexity
- Potential bottleneck
- Requires high availability
    """)


if __name__ == "__main__":
    demonstrate_api_gateway()
