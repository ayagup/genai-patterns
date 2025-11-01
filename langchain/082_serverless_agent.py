"""
Pattern 082: Serverless Agent

Description:
    A Serverless Agent runs on serverless computing platforms (AWS Lambda,
    Azure Functions, Google Cloud Functions) where infrastructure is fully
    managed by the cloud provider. This pattern enables event-driven, auto-scaling
    AI agents that only consume resources when invoked, optimizing costs and
    operational overhead.
    
    The agent is designed for stateless execution, cold start optimization,
    event-driven triggers, and seamless scaling from zero to thousands of
    concurrent executions without manual intervention.

Components:
    1. Function Handler: Entry point for serverless execution
    2. Event Processor: Processes various event types
    3. State Manager: External state storage (S3, DynamoDB)
    4. Cold Start Optimizer: Minimizes initialization time
    5. Environment Manager: Configuration and secrets
    6. Response Builder: Formats function responses
    7. Error Handler: Graceful error handling
    8. Metrics Collector: CloudWatch/monitoring integration

Key Features:
    - Stateless execution model
    - Event-driven triggers
    - Auto-scaling from zero
    - Pay-per-execution pricing
    - No server management
    - Built-in fault tolerance
    - Managed infrastructure
    - Cold start optimization
    - Multiple trigger sources
    - Seamless integration with cloud services

Use Cases:
    - API endpoints for AI services
    - Event-driven AI processing
    - Webhook handlers
    - Scheduled AI tasks
    - Image/document processing
    - Real-time data transformation
    - Chatbot backends
    - ETL pipelines
    - IoT data processing
    - Microservices backends

LangChain Implementation:
    Simulates AWS Lambda-style function handlers with event processing,
    stateless execution, and external state management patterns.
"""

import os
import time
import json
import uuid
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


class TriggerType(Enum):
    """Types of serverless triggers"""
    HTTP_API = "http_api"
    S3_EVENT = "s3_event"
    DYNAMODB_STREAM = "dynamodb_stream"
    SNS_MESSAGE = "sns_message"
    SQS_QUEUE = "sqs_queue"
    SCHEDULED = "scheduled"
    WEBHOOK = "webhook"
    IOT_EVENT = "iot_event"


class ExecutionStatus(Enum):
    """Status of function execution"""
    COLD_START = "cold_start"
    WARM_START = "warm_start"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ServerlessEvent:
    """Event that triggers serverless function"""
    event_id: str
    trigger_type: TriggerType
    timestamp: datetime
    data: Dict[str, Any]
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerlessContext:
    """Execution context for serverless function"""
    request_id: str
    function_name: str
    function_version: str
    memory_limit_mb: int
    timeout_seconds: int
    remaining_time_ms: int
    cold_start: bool = True
    invocation_count: int = 0


@dataclass
class ServerlessResponse:
    """Response from serverless function"""
    status_code: int
    body: Any
    headers: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    execution_duration_ms: float = 0.0
    cold_start: bool = False
    memory_used_mb: float = 0.0


@dataclass
class FunctionMetrics:
    """Metrics for serverless function"""
    function_name: str
    invocations: int = 0
    errors: int = 0
    cold_starts: int = 0
    warm_starts: int = 0
    total_duration_ms: float = 0.0
    average_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    min_duration_ms: float = float('inf')


class ServerlessAgent:
    """
    Base serverless agent with stateless execution.
    
    Designed for serverless platforms with event-driven triggers,
    cold start optimization, and external state management.
    """
    
    # Class-level cache (simulates container reuse)
    _warm_instances: Dict[str, 'ServerlessAgent'] = {}
    _llm_cache: Optional[ChatOpenAI] = None
    
    def __init__(
        self,
        function_name: str,
        memory_limit_mb: int = 512,
        timeout_seconds: int = 30
    ):
        """
        Initialize serverless agent.
        
        Args:
            function_name: Name of the function
            memory_limit_mb: Memory limit
            timeout_seconds: Execution timeout
        """
        self.function_name = function_name
        self.memory_limit_mb = memory_limit_mb
        self.timeout_seconds = timeout_seconds
        
        self.invocation_count = 0
        self.metrics = FunctionMetrics(function_name=function_name)
        
        # Initialize LLM (reuse cached if available)
        if ServerlessAgent._llm_cache is None:
            ServerlessAgent._llm_cache = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7
            )
        
        self.llm = ServerlessAgent._llm_cache
    
    @classmethod
    def get_or_create(cls, function_name: str) -> 'ServerlessAgent':
        """
        Get warm instance or create new (simulates Lambda container reuse).
        
        Args:
            function_name: Function name
            
        Returns:
            ServerlessAgent instance
        """
        if function_name in cls._warm_instances:
            return cls._warm_instances[function_name]
        
        instance = cls(function_name)
        cls._warm_instances[function_name] = instance
        return instance
    
    def handler(
        self,
        event: ServerlessEvent,
        context: ServerlessContext
    ) -> ServerlessResponse:
        """
        Main handler function (entry point).
        
        Args:
            event: Triggering event
            context: Execution context
            
        Returns:
            ServerlessResponse
        """
        start_time = time.time()
        
        self.invocation_count += 1
        context.invocation_count = self.invocation_count
        
        # Track metrics
        self.metrics.invocations += 1
        if context.cold_start:
            self.metrics.cold_starts += 1
        else:
            self.metrics.warm_starts += 1
        
        try:
            # Process event based on trigger type
            result = self._process_event(event, context)
            
            execution_duration = (time.time() - start_time) * 1000
            
            # Update metrics
            self.metrics.total_duration_ms += execution_duration
            self.metrics.average_duration_ms = (
                self.metrics.total_duration_ms / self.metrics.invocations
            )
            self.metrics.max_duration_ms = max(
                self.metrics.max_duration_ms,
                execution_duration
            )
            self.metrics.min_duration_ms = min(
                self.metrics.min_duration_ms,
                execution_duration
            )
            
            return ServerlessResponse(
                status_code=200,
                body=result,
                headers={"Content-Type": "application/json"},
                execution_duration_ms=execution_duration,
                cold_start=context.cold_start,
                memory_used_mb=self.memory_limit_mb * 0.6  # Simulated
            )
            
        except Exception as e:
            self.metrics.errors += 1
            
            return ServerlessResponse(
                status_code=500,
                body=None,
                error=str(e),
                execution_duration_ms=(time.time() - start_time) * 1000,
                cold_start=context.cold_start
            )
    
    def _process_event(
        self,
        event: ServerlessEvent,
        context: ServerlessContext
    ) -> Dict[str, Any]:
        """
        Process event (to be overridden by subclasses).
        
        Args:
            event: Event to process
            context: Execution context
            
        Returns:
            Processing result
        """
        return {
            "message": "Event processed",
            "event_id": event.event_id,
            "function": self.function_name
        }


class TextProcessingFunction(ServerlessAgent):
    """Serverless function for text processing"""
    
    def __init__(self):
        super().__init__("text-processor", memory_limit_mb=512, timeout_seconds=30)
    
    def _process_event(
        self,
        event: ServerlessEvent,
        context: ServerlessContext
    ) -> Dict[str, Any]:
        """Process text from event"""
        data = event.data
        operation = data.get('operation', 'summarize')
        text = data.get('text', '')
        
        if operation == 'summarize':
            result = self._summarize(text)
        elif operation == 'sentiment':
            result = self._analyze_sentiment(text)
        elif operation == 'extract':
            result = self._extract_keywords(text)
        else:
            result = {'error': f'Unknown operation: {operation}'}
        
        return {
            'operation': operation,
            'result': result,
            'cold_start': context.cold_start,
            'invocation': context.invocation_count
        }
    
    def _summarize(self, text: str) -> str:
        """Summarize text"""
        prompt = ChatPromptTemplate.from_template(
            "Summarize this text in one sentence: {text}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({'text': text[:500]})
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment"""
        prompt = ChatPromptTemplate.from_template(
            "What is the sentiment of this text? Answer in one word (positive/negative/neutral): {text}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({'text': text[:500]})
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords"""
        prompt = ChatPromptTemplate.from_template(
            "Extract 5 key words from this text (comma-separated): {text}"
        )
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({'text': text[:500]})
        return [k.strip() for k in result.split(',')[:5]]


class WebhookFunction(ServerlessAgent):
    """Serverless function for webhook processing"""
    
    def __init__(self):
        super().__init__("webhook-handler", memory_limit_mb=256, timeout_seconds=15)
    
    def _process_event(
        self,
        event: ServerlessEvent,
        context: ServerlessContext
    ) -> Dict[str, Any]:
        """Process webhook event"""
        data = event.data
        webhook_type = data.get('type', 'generic')
        payload = data.get('payload', {})
        
        # Process webhook
        response = {
            'webhook_type': webhook_type,
            'processed_at': datetime.now().isoformat(),
            'status': 'received',
            'payload_size': len(str(payload))
        }
        
        return response


class ScheduledFunction(ServerlessAgent):
    """Serverless function for scheduled tasks"""
    
    def __init__(self):
        super().__init__("scheduled-task", memory_limit_mb=1024, timeout_seconds=60)
    
    def _process_event(
        self,
        event: ServerlessEvent,
        context: ServerlessContext
    ) -> Dict[str, Any]:
        """Process scheduled event"""
        task_type = event.data.get('task', 'report')
        
        if task_type == 'report':
            result = self._generate_report()
        elif task_type == 'cleanup':
            result = self._cleanup_data()
        elif task_type == 'sync':
            result = self._sync_data()
        else:
            result = {'error': 'Unknown task type'}
        
        return {
            'task': task_type,
            'result': result,
            'scheduled_time': event.timestamp.isoformat()
        }
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate report"""
        return {
            'report_type': 'daily_summary',
            'generated_at': datetime.now().isoformat(),
            'items_processed': 100
        }
    
    def _cleanup_data(self) -> Dict[str, Any]:
        """Cleanup old data"""
        return {
            'items_deleted': 50,
            'space_freed_mb': 250
        }
    
    def _sync_data(self) -> Dict[str, Any]:
        """Sync data"""
        return {
            'items_synced': 75,
            'sync_duration_ms': 1500
        }


class ServerlessPlatform:
    """
    Simulates serverless platform (like AWS Lambda).
    
    Manages function lifecycle, invocations, and cold/warm starts.
    """
    
    def __init__(self):
        """Initialize platform"""
        self.functions: Dict[str, ServerlessAgent] = {}
        self.function_metrics: Dict[str, FunctionMetrics] = {}
    
    def deploy_function(self, function: ServerlessAgent) -> bool:
        """
        Deploy a serverless function.
        
        Args:
            function: Function to deploy
            
        Returns:
            True if deployed successfully
        """
        self.functions[function.function_name] = function
        self.function_metrics[function.function_name] = function.metrics
        return True
    
    def invoke(
        self,
        function_name: str,
        event: ServerlessEvent
    ) -> ServerlessResponse:
        """
        Invoke a serverless function.
        
        Args:
            function_name: Name of function to invoke
            event: Event to process
            
        Returns:
            ServerlessResponse
        """
        if function_name not in self.functions:
            return ServerlessResponse(
                status_code=404,
                body=None,
                error=f"Function not found: {function_name}"
            )
        
        function = self.functions[function_name]
        
        # Determine if cold start (simplified)
        is_cold_start = function.invocation_count == 0
        
        # Create context
        context = ServerlessContext(
            request_id=f"req_{uuid.uuid4().hex[:8]}",
            function_name=function_name,
            function_version="1.0",
            memory_limit_mb=function.memory_limit_mb,
            timeout_seconds=function.timeout_seconds,
            remaining_time_ms=function.timeout_seconds * 1000,
            cold_start=is_cold_start
        )
        
        # Invoke function
        return function.handler(event, context)
    
    def get_metrics(self, function_name: str) -> Optional[FunctionMetrics]:
        """
        Get function metrics.
        
        Args:
            function_name: Function name
            
        Returns:
            FunctionMetrics if found
        """
        return self.function_metrics.get(function_name)


def demonstrate_serverless_agent():
    """Demonstrate the serverless agent pattern"""
    print("=" * 80)
    print("SERVERLESS AGENT DEMONSTRATION")
    print("=" * 80)
    
    platform = ServerlessPlatform()
    
    # Demo 1: Deploy Serverless Functions
    print("\n" + "=" * 80)
    print("DEMO 1: Deploying Serverless Functions")
    print("=" * 80)
    
    functions = [
        TextProcessingFunction(),
        WebhookFunction(),
        ScheduledFunction()
    ]
    
    print("\nDeploying functions to serverless platform...")
    for func in functions:
        platform.deploy_function(func)
        print(f"  ✓ Deployed: {func.function_name}")
        print(f"    Memory: {func.memory_limit_mb}MB")
        print(f"    Timeout: {func.timeout_seconds}s")
    
    # Demo 2: Cold Start vs Warm Start
    print("\n" + "=" * 80)
    print("DEMO 2: Cold Start vs Warm Start")
    print("=" * 80)
    
    event = ServerlessEvent(
        event_id="evt_001",
        trigger_type=TriggerType.HTTP_API,
        timestamp=datetime.now(),
        data={
            'operation': 'summarize',
            'text': 'Serverless computing allows running code without managing servers.'
        }
    )
    
    print("\nFirst invocation (Cold Start):")
    response1 = platform.invoke("text-processor", event)
    print(f"  Status: {response1.status_code}")
    print(f"  Cold Start: {response1.cold_start}")
    print(f"  Duration: {response1.execution_duration_ms:.2f}ms")
    print(f"  Memory Used: {response1.memory_used_mb:.1f}MB")
    
    print("\nSecond invocation (Warm Start):")
    response2 = platform.invoke("text-processor", event)
    print(f"  Status: {response2.status_code}")
    print(f"  Cold Start: {response2.cold_start}")
    print(f"  Duration: {response2.execution_duration_ms:.2f}ms")
    print(f"  Memory Used: {response2.memory_used_mb:.1f}MB")
    
    speedup = response1.execution_duration_ms / response2.execution_duration_ms if response2.execution_duration_ms > 0 else 1
    print(f"\n  Warm start {speedup:.2f}x faster than cold start")
    
    # Demo 3: Multiple Operations
    print("\n" + "=" * 80)
    print("DEMO 3: Multiple Text Processing Operations")
    print("=" * 80)
    
    operations = [
        ('summarize', 'AI agents are autonomous systems that can reason and act.'),
        ('sentiment', 'This serverless architecture is amazing!'),
        ('extract', 'Machine learning models process data to make predictions.')
    ]
    
    print("\nProcessing multiple operations...")
    print("-" * 80)
    
    for operation, text in operations:
        event = ServerlessEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            trigger_type=TriggerType.HTTP_API,
            timestamp=datetime.now(),
            data={'operation': operation, 'text': text}
        )
        
        response = platform.invoke("text-processor", event)
        
        print(f"\nOperation: {operation}")
        print(f"  Input: {text[:50]}...")
        print(f"  Duration: {response.execution_duration_ms:.2f}ms")
        
        if response.body and 'result' in response.body:
            result = response.body['result']
            if isinstance(result, str):
                print(f"  Result: {result[:80]}...")
            else:
                print(f"  Result: {result}")
    
    # Demo 4: Webhook Handler
    print("\n" + "=" * 80)
    print("DEMO 4: Webhook Event Processing")
    print("=" * 80)
    
    webhook_event = ServerlessEvent(
        event_id="webhook_001",
        trigger_type=TriggerType.WEBHOOK,
        timestamp=datetime.now(),
        data={
            'type': 'user_signup',
            'payload': {
                'user_id': '12345',
                'email': 'user@example.com',
                'timestamp': datetime.now().isoformat()
            }
        }
    )
    
    print("\nProcessing webhook event...")
    response = platform.invoke("webhook-handler", webhook_event)
    
    print(f"  Status: {response.status_code}")
    print(f"  Duration: {response.execution_duration_ms:.2f}ms")
    print(f"  Response:")
    for key, value in response.body.items():
        print(f"    {key}: {value}")
    
    # Demo 5: Scheduled Task
    print("\n" + "=" * 80)
    print("DEMO 5: Scheduled Task Execution")
    print("=" * 80)
    
    scheduled_events = [
        {'task': 'report'},
        {'task': 'cleanup'},
        {'task': 'sync'}
    ]
    
    print("\nExecuting scheduled tasks...")
    print("-" * 80)
    
    for task_data in scheduled_events:
        event = ServerlessEvent(
            event_id=f"sched_{uuid.uuid4().hex[:8]}",
            trigger_type=TriggerType.SCHEDULED,
            timestamp=datetime.now(),
            data=task_data
        )
        
        response = platform.invoke("scheduled-task", event)
        
        print(f"\nTask: {task_data['task']}")
        print(f"  Duration: {response.execution_duration_ms:.2f}ms")
        print(f"  Result: {response.body.get('result', {})}")
    
    # Demo 6: Function Metrics
    print("\n" + "=" * 80)
    print("DEMO 6: Function Metrics & Monitoring")
    print("=" * 80)
    
    print("\nFunction Metrics:")
    print("-" * 80)
    
    for func_name in ["text-processor", "webhook-handler", "scheduled-task"]:
        metrics = platform.get_metrics(func_name)
        
        if metrics:
            print(f"\n{func_name}:")
            print(f"  Total Invocations: {metrics.invocations}")
            print(f"  Cold Starts: {metrics.cold_starts}")
            print(f"  Warm Starts: {metrics.warm_starts}")
            print(f"  Errors: {metrics.errors}")
            print(f"  Avg Duration: {metrics.average_duration_ms:.2f}ms")
            print(f"  Min Duration: {metrics.min_duration_ms:.2f}ms")
            print(f"  Max Duration: {metrics.max_duration_ms:.2f}ms")
            
            cold_start_pct = (metrics.cold_starts / metrics.invocations * 100) if metrics.invocations > 0 else 0
            print(f"  Cold Start Rate: {cold_start_pct:.1f}%")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary = """
The Serverless Agent demonstrates cloud-native, event-driven AI:

KEY CAPABILITIES:
1. Stateless Execution: No persistent state between invocations
2. Event-Driven: Triggered by various event sources
3. Auto-Scaling: Scales automatically from zero to thousands
4. Cold/Warm Starts: Container reuse optimization
5. Pay-Per-Execution: Only pay for actual compute time
6. Managed Infrastructure: No server management required
7. Multiple Triggers: HTTP, webhooks, schedules, queues, streams
8. Built-in Monitoring: Integrated metrics and logging

BENEFITS:
- Zero infrastructure management
- Automatic scaling with demand
- Cost optimization (pay per use)
- High availability by default
- Fast deployment and iteration
- Built-in fault tolerance
- Easy integration with cloud services
- Reduced operational overhead

USE CASES:
- REST API endpoints for AI services
- Webhook processing for integrations
- Scheduled AI tasks (reports, cleanup, sync)
- Event-driven data processing
- Real-time stream processing
- Image/document processing pipelines
- Chatbot backends
- ETL and data transformation
- IoT data processing
- Microservices backends

PRODUCTION CONSIDERATIONS:
1. Cold Starts: Minimize with provisioned concurrency
2. Statelessness: Use external storage (S3, DynamoDB, RDS)
3. Timeout Limits: Functions have max execution time
4. Memory Allocation: Balance cost vs performance
5. Concurrency Limits: Understand and configure limits
6. VPC Networking: Consider cold start impact
7. Error Handling: Implement retries and DLQs
8. Monitoring: Use CloudWatch, X-Ray for observability
9. Security: IAM roles, secrets management
10. Cost Management: Monitor and optimize usage

ADVANCED EXTENSIONS:
- Lambda layers for shared dependencies
- Step Functions for complex workflows
- EventBridge for event routing
- API Gateway for HTTP endpoints
- SQS/SNS for async messaging
- Lambda@Edge for CDN integration
- Provisioned concurrency for predictable latency
- Container image deployment
- Multi-region deployment
- Custom runtimes and extensions

Serverless architecture is ideal for event-driven AI workloads
with variable demand, optimizing both cost and scalability.
"""
    
    print(summary)


if __name__ == "__main__":
    demonstrate_serverless_agent()
