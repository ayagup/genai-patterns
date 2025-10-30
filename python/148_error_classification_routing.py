"""
Pattern 148: Error Classification & Routing Agent

This pattern implements intelligent error classification and routing to appropriate
handlers based on error type, severity, and context. Routes errors to specialized
handlers for optimal recovery strategies.

Category: Error Handling & Recovery
Use Cases:
- Production error handling
- Multi-service error management
- Intelligent error recovery
- Error triage systems
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from enum import Enum
from datetime import datetime
import re
import random


class ErrorCategory(Enum):
    """High-level error categories"""
    TRANSIENT = "transient"           # Temporary, retry likely to succeed
    CONFIGURATION = "configuration"   # Configuration or setup issues
    AUTHENTICATION = "authentication" # Auth/authz failures
    VALIDATION = "validation"         # Input validation errors
    RESOURCE = "resource"             # Resource exhaustion
    NETWORK = "network"               # Network connectivity
    TIMEOUT = "timeout"               # Operation timeouts
    RATE_LIMIT = "rate_limit"        # Rate limiting
    DATA = "data"                     # Data integrity/format issues
    BUSINESS_LOGIC = "business_logic" # Business rule violations
    SYSTEM = "system"                 # System/infrastructure errors
    UNKNOWN = "unknown"               # Unclassified errors


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"   # System-wide impact
    HIGH = "high"          # Major feature impact
    MEDIUM = "medium"      # Minor feature impact
    LOW = "low"            # Minimal impact
    INFO = "info"          # Informational


class HandlerStrategy(Enum):
    """Error handling strategies"""
    RETRY = "retry"               # Retry operation
    FALLBACK = "fallback"         # Use fallback/default
    ESCALATE = "escalate"         # Escalate to human
    IGNORE = "ignore"             # Log and continue
    COMPENSATE = "compensate"     # Execute compensation
    CIRCUIT_BREAK = "circuit_break"  # Open circuit breaker
    THROTTLE = "throttle"         # Apply throttling


@dataclass
class ErrorSignature:
    """Signature for identifying error patterns"""
    keywords: List[str]
    regex_patterns: List[str]
    error_codes: Set[int]
    exception_types: Set[str]


@dataclass
class ErrorClassification:
    """Classification result for an error"""
    category: ErrorCategory
    severity: ErrorSeverity
    confidence: float
    recommended_strategy: HandlerStrategy
    context: Dict[str, Any] = field(default_factory=dict)
    classification_time: datetime = field(default_factory=datetime.now)


@dataclass
class ErrorContext:
    """Context information about an error"""
    error_message: str
    exception_type: str
    stack_trace: Optional[str] = None
    error_code: Optional[int] = None
    service_name: Optional[str] = None
    operation_name: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandlerResult:
    """Result of error handling"""
    success: bool
    strategy_used: HandlerStrategy
    handler_name: str
    resolution: Optional[str] = None
    should_retry: bool = False
    escalated: bool = False
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorClassifier:
    """Classifies errors into categories"""
    
    def __init__(self):
        self.signatures = self._build_signatures()
        
    def _build_signatures(self) -> Dict[ErrorCategory, ErrorSignature]:
        """Build error signatures for classification"""
        return {
            ErrorCategory.TRANSIENT: ErrorSignature(
                keywords=['temporary', 'transient', 'unavailable', 'busy'],
                regex_patterns=[r'temporarily unavailable', r'try again'],
                error_codes={503, 504, 429},
                exception_types={'TransientError', 'TemporaryFailure'}
            ),
            ErrorCategory.NETWORK: ErrorSignature(
                keywords=['connection', 'network', 'socket', 'dns', 'unreachable'],
                regex_patterns=[r'connection.*failed', r'network.*error'],
                error_codes={502, 503, 504},
                exception_types={'NetworkError', 'ConnectionError', 'TimeoutError'}
            ),
            ErrorCategory.AUTHENTICATION: ErrorSignature(
                keywords=['unauthorized', 'forbidden', 'authentication', 'credentials'],
                regex_patterns=[r'auth.*failed', r'invalid.*token'],
                error_codes={401, 403},
                exception_types={'AuthenticationError', 'PermissionError'}
            ),
            ErrorCategory.VALIDATION: ErrorSignature(
                keywords=['invalid', 'validation', 'malformed', 'bad request'],
                regex_patterns=[r'invalid.*input', r'validation.*failed'],
                error_codes={400, 422},
                exception_types={'ValueError', 'ValidationError'}
            ),
            ErrorCategory.RATE_LIMIT: ErrorSignature(
                keywords=['rate limit', 'too many requests', 'quota exceeded'],
                regex_patterns=[r'rate.*limit', r'quota.*exceeded'],
                error_codes={429},
                exception_types={'RateLimitError', 'QuotaExceededError'}
            ),
            ErrorCategory.TIMEOUT: ErrorSignature(
                keywords=['timeout', 'timed out', 'deadline exceeded'],
                regex_patterns=[r'timeout', r'deadline.*exceeded'],
                error_codes={408, 504},
                exception_types={'TimeoutError', 'DeadlineExceeded'}
            ),
            ErrorCategory.RESOURCE: ErrorSignature(
                keywords=['memory', 'disk', 'capacity', 'resource', 'exhausted'],
                regex_patterns=[r'out of.*memory', r'disk.*full'],
                error_codes={507},
                exception_types={'MemoryError', 'ResourceExhausted'}
            ),
            ErrorCategory.DATA: ErrorSignature(
                keywords=['corrupt', 'integrity', 'format', 'parse'],
                regex_patterns=[r'data.*corrupt', r'parse.*error'],
                error_codes=set(),
                exception_types={'DataError', 'IntegrityError', 'ParseError'}
            ),
            ErrorCategory.CONFIGURATION: ErrorSignature(
                keywords=['configuration', 'config', 'missing', 'not found'],
                regex_patterns=[r'config.*missing', r'not.*configured'],
                error_codes={500},
                exception_types={'ConfigurationError', 'FileNotFoundError'}
            )
        }
    
    def classify(self, error_context: ErrorContext) -> ErrorClassification:
        """Classify an error"""
        # Score each category
        scores: Dict[ErrorCategory, float] = {}
        
        for category, signature in self.signatures.items():
            score = self._calculate_match_score(error_context, signature)
            scores[category] = score
        
        # Find best match
        if not scores or max(scores.values()) == 0:
            category = ErrorCategory.UNKNOWN
            confidence = 0.0
        else:
            category = max(scores.items(), key=lambda x: x[1])[0]
            confidence = scores[category]
        
        # Determine severity
        severity = self._determine_severity(error_context, category)
        
        # Recommend strategy
        strategy = self._recommend_strategy(category, severity)
        
        return ErrorClassification(
            category=category,
            severity=severity,
            confidence=confidence,
            recommended_strategy=strategy,
            context={
                'scores': {k.value: v for k, v in scores.items()},
                'error_code': error_context.error_code,
                'service': error_context.service_name
            }
        )
    
    def _calculate_match_score(
        self,
        context: ErrorContext,
        signature: ErrorSignature
    ) -> float:
        """Calculate how well context matches signature"""
        score = 0.0
        matches = 0
        total_checks = 0
        
        error_text = (context.error_message + ' ' + context.exception_type).lower()
        
        # Check keywords
        for keyword in signature.keywords:
            total_checks += 1
            if keyword in error_text:
                matches += 1
                score += 0.3
        
        # Check regex patterns
        for pattern in signature.regex_patterns:
            total_checks += 1
            if re.search(pattern, error_text, re.IGNORECASE):
                matches += 1
                score += 0.4
        
        # Check error codes
        if context.error_code:
            total_checks += 1
            if context.error_code in signature.error_codes:
                matches += 1
                score += 0.5
        
        # Check exception types
        total_checks += 1
        if context.exception_type in signature.exception_types:
            matches += 1
            score += 0.5
        
        # Normalize score
        if total_checks > 0:
            confidence = matches / total_checks
            score = min(1.0, score * confidence)
        
        return score
    
    def _determine_severity(
        self,
        context: ErrorContext,
        category: ErrorCategory
    ) -> ErrorSeverity:
        """Determine error severity"""
        # Critical categories
        if category in [ErrorCategory.SYSTEM, ErrorCategory.RESOURCE]:
            return ErrorSeverity.CRITICAL
        
        # High severity categories
        if category in [ErrorCategory.AUTHENTICATION, ErrorCategory.DATA]:
            return ErrorSeverity.HIGH
        
        # Medium severity
        if category in [ErrorCategory.NETWORK, ErrorCategory.TIMEOUT, ErrorCategory.CONFIGURATION]:
            return ErrorSeverity.MEDIUM
        
        # Low severity
        if category in [ErrorCategory.TRANSIENT, ErrorCategory.RATE_LIMIT]:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _recommend_strategy(
        self,
        category: ErrorCategory,
        severity: ErrorSeverity
    ) -> HandlerStrategy:
        """Recommend handling strategy"""
        strategy_map = {
            ErrorCategory.TRANSIENT: HandlerStrategy.RETRY,
            ErrorCategory.NETWORK: HandlerStrategy.RETRY,
            ErrorCategory.TIMEOUT: HandlerStrategy.RETRY,
            ErrorCategory.RATE_LIMIT: HandlerStrategy.THROTTLE,
            ErrorCategory.AUTHENTICATION: HandlerStrategy.ESCALATE,
            ErrorCategory.VALIDATION: HandlerStrategy.FALLBACK,
            ErrorCategory.RESOURCE: HandlerStrategy.CIRCUIT_BREAK,
            ErrorCategory.DATA: HandlerStrategy.COMPENSATE,
            ErrorCategory.CONFIGURATION: HandlerStrategy.ESCALATE,
            ErrorCategory.BUSINESS_LOGIC: HandlerStrategy.FALLBACK,
            ErrorCategory.SYSTEM: HandlerStrategy.ESCALATE,
            ErrorCategory.UNKNOWN: HandlerStrategy.ESCALATE
        }
        
        return strategy_map.get(category, HandlerStrategy.ESCALATE)


class ErrorHandler:
    """Base class for error handlers"""
    
    def __init__(self, name: str, categories: List[ErrorCategory]):
        self.name = name
        self.categories = categories
        self.handled_count = 0
        
    def can_handle(self, classification: ErrorClassification) -> bool:
        """Check if handler can handle this error"""
        return classification.category in self.categories
    
    def handle(
        self,
        error_context: ErrorContext,
        classification: ErrorClassification
    ) -> HandlerResult:
        """Handle the error"""
        raise NotImplementedError


class RetryHandler(ErrorHandler):
    """Handles errors with retry strategy"""
    
    def __init__(self):
        super().__init__(
            "RetryHandler",
            [ErrorCategory.TRANSIENT, ErrorCategory.NETWORK, ErrorCategory.TIMEOUT]
        )
    
    def handle(
        self,
        error_context: ErrorContext,
        classification: ErrorClassification
    ) -> HandlerResult:
        self.handled_count += 1
        
        return HandlerResult(
            success=True,
            strategy_used=HandlerStrategy.RETRY,
            handler_name=self.name,
            resolution=f"Will retry operation for {classification.category.value} error",
            should_retry=True,
            metadata={
                'max_retries': 3,
                'backoff_ms': 100
            }
        )


class ThrottleHandler(ErrorHandler):
    """Handles rate limit errors with throttling"""
    
    def __init__(self):
        super().__init__(
            "ThrottleHandler",
            [ErrorCategory.RATE_LIMIT]
        )
    
    def handle(
        self,
        error_context: ErrorContext,
        classification: ErrorClassification
    ) -> HandlerResult:
        self.handled_count += 1
        
        return HandlerResult(
            success=True,
            strategy_used=HandlerStrategy.THROTTLE,
            handler_name=self.name,
            resolution="Applied throttling to reduce request rate",
            should_retry=True,
            metadata={
                'throttle_seconds': 60,
                'reduced_rate': 0.5
            }
        )


class FallbackHandler(ErrorHandler):
    """Handles errors with fallback values"""
    
    def __init__(self):
        super().__init__(
            "FallbackHandler",
            [ErrorCategory.VALIDATION, ErrorCategory.BUSINESS_LOGIC, ErrorCategory.DATA]
        )
    
    def handle(
        self,
        error_context: ErrorContext,
        classification: ErrorClassification
    ) -> HandlerResult:
        self.handled_count += 1
        
        return HandlerResult(
            success=True,
            strategy_used=HandlerStrategy.FALLBACK,
            handler_name=self.name,
            resolution="Using fallback/default value",
            should_retry=False,
            metadata={
                'fallback_value': 'default',
                'degraded_mode': True
            }
        )


class EscalationHandler(ErrorHandler):
    """Handles errors requiring human intervention"""
    
    def __init__(self):
        super().__init__(
            "EscalationHandler",
            [ErrorCategory.AUTHENTICATION, ErrorCategory.CONFIGURATION, 
             ErrorCategory.SYSTEM, ErrorCategory.UNKNOWN]
        )
    
    def handle(
        self,
        error_context: ErrorContext,
        classification: ErrorClassification
    ) -> HandlerResult:
        self.handled_count += 1
        
        return HandlerResult(
            success=True,
            strategy_used=HandlerStrategy.ESCALATE,
            handler_name=self.name,
            resolution=f"Escalated {classification.severity.value} severity error to operations team",
            escalated=True,
            metadata={
                'escalation_level': 'L2',
                'notification_sent': True
            }
        )


class CircuitBreakerHandler(ErrorHandler):
    """Handles resource exhaustion with circuit breaking"""
    
    def __init__(self):
        super().__init__(
            "CircuitBreakerHandler",
            [ErrorCategory.RESOURCE]
        )
    
    def handle(
        self,
        error_context: ErrorContext,
        classification: ErrorClassification
    ) -> HandlerResult:
        self.handled_count += 1
        
        return HandlerResult(
            success=True,
            strategy_used=HandlerStrategy.CIRCUIT_BREAK,
            handler_name=self.name,
            resolution="Opened circuit breaker to prevent resource exhaustion",
            should_retry=False,
            metadata={
                'circuit_state': 'open',
                'cooldown_seconds': 60
            }
        )


class ErrorRouter:
    """Routes errors to appropriate handlers"""
    
    def __init__(self):
        self.handlers: List[ErrorHandler] = []
        self.default_handler = EscalationHandler()
        self.routing_history: List[Dict[str, Any]] = []
        
    def register_handler(self, handler: ErrorHandler):
        """Register an error handler"""
        self.handlers.append(handler)
    
    def route(
        self,
        error_context: ErrorContext,
        classification: ErrorClassification
    ) -> HandlerResult:
        """Route error to appropriate handler"""
        # Find matching handler
        selected_handler = None
        
        for handler in self.handlers:
            if handler.can_handle(classification):
                selected_handler = handler
                break
        
        # Use default if no match
        if not selected_handler:
            selected_handler = self.default_handler
        
        # Handle error
        result = selected_handler.handle(error_context, classification)
        
        # Record routing decision
        self.routing_history.append({
            'timestamp': datetime.now(),
            'category': classification.category.value,
            'severity': classification.severity.value,
            'handler': selected_handler.name,
            'strategy': result.strategy_used.value,
            'success': result.success
        })
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        if not self.routing_history:
            return {"error": "No routing history"}
        
        total = len(self.routing_history)
        
        # By category
        by_category = {}
        for record in self.routing_history:
            cat = record['category']
            by_category[cat] = by_category.get(cat, 0) + 1
        
        # By handler
        by_handler = {}
        for record in self.routing_history:
            handler = record['handler']
            by_handler[handler] = by_handler.get(handler, 0) + 1
        
        # By strategy
        by_strategy = {}
        for record in self.routing_history:
            strategy = record['strategy']
            by_strategy[strategy] = by_strategy.get(strategy, 0) + 1
        
        return {
            'total_errors': total,
            'by_category': by_category,
            'by_handler': by_handler,
            'by_strategy': by_strategy,
            'handler_utilization': {
                handler.name: {
                    'handled': handler.handled_count,
                    'categories': [c.value for c in handler.categories]
                }
                for handler in self.handlers + [self.default_handler]
            }
        }


class ErrorClassificationRoutingAgent:
    """
    Main agent that classifies errors and routes them to appropriate handlers.
    Provides intelligent error management with specialized recovery strategies.
    """
    
    def __init__(self):
        self.classifier = ErrorClassifier()
        self.router = ErrorRouter()
        self._register_default_handlers()
        
    def _register_default_handlers(self):
        """Register default error handlers"""
        self.router.register_handler(RetryHandler())
        self.router.register_handler(ThrottleHandler())
        self.router.register_handler(FallbackHandler())
        self.router.register_handler(CircuitBreakerHandler())
        self.router.register_handler(EscalationHandler())
    
    def handle_error(
        self,
        error_message: str,
        exception_type: str = "Exception",
        error_code: Optional[int] = None,
        service_name: Optional[str] = None,
        **kwargs
    ) -> Tuple[ErrorClassification, HandlerResult]:
        """Handle an error with classification and routing"""
        # Create error context
        context = ErrorContext(
            error_message=error_message,
            exception_type=exception_type,
            error_code=error_code,
            service_name=service_name,
            **kwargs
        )
        
        # Classify error
        classification = self.classifier.classify(context)
        
        # Route to handler
        result = self.router.route(context, classification)
        
        return classification, result
    
    def register_handler(self, handler: ErrorHandler):
        """Register custom error handler"""
        self.router.register_handler(handler)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return self.router.get_statistics()


def demonstrate_error_classification_routing():
    """Demonstrate error classification and routing pattern"""
    print("\n" + "="*60)
    print("ERROR CLASSIFICATION & ROUTING PATTERN DEMONSTRATION")
    print("="*60)
    
    agent = ErrorClassificationRoutingAgent()
    
    # Scenario 1: Network error
    print("\n" + "-"*60)
    print("Scenario 1: Network Connection Error")
    print("-"*60)
    
    classification, result = agent.handle_error(
        error_message="Connection to database failed: network unreachable",
        exception_type="ConnectionError",
        error_code=503,
        service_name="database"
    )
    
    print(f"Error: Connection to database failed")
    print(f"  Category: {classification.category.value}")
    print(f"  Severity: {classification.severity.value}")
    print(f"  Confidence: {classification.confidence:.2%}")
    print(f"  Recommended Strategy: {classification.recommended_strategy.value}")
    print(f"\nHandling:")
    print(f"  Handler: {result.handler_name}")
    print(f"  Strategy: {result.strategy_used.value}")
    print(f"  Resolution: {result.resolution}")
    print(f"  Should Retry: {result.should_retry}")
    
    # Scenario 2: Rate limit error
    print("\n" + "-"*60)
    print("Scenario 2: Rate Limit Exceeded")
    print("-"*60)
    
    classification, result = agent.handle_error(
        error_message="Rate limit exceeded: too many requests",
        exception_type="RateLimitError",
        error_code=429,
        service_name="api"
    )
    
    print(f"Error: Rate limit exceeded")
    print(f"  Category: {classification.category.value}")
    print(f"  Handler: {result.handler_name}")
    print(f"  Strategy: {result.strategy_used.value}")
    print(f"  Throttle for: {result.metadata.get('throttle_seconds')}s")
    
    # Scenario 3: Validation error
    print("\n" + "-"*60)
    print("Scenario 3: Input Validation Error")
    print("-"*60)
    
    classification, result = agent.handle_error(
        error_message="Invalid input: email format is malformed",
        exception_type="ValueError",
        error_code=400,
        service_name="user_service"
    )
    
    print(f"Error: Invalid input format")
    print(f"  Category: {classification.category.value}")
    print(f"  Handler: {result.handler_name}")
    print(f"  Strategy: {result.strategy_used.value}")
    print(f"  Using fallback: {result.metadata.get('fallback_value')}")
    
    # Scenario 4: Authentication error
    print("\n" + "-"*60)
    print("Scenario 4: Authentication Failure")
    print("-"*60)
    
    classification, result = agent.handle_error(
        error_message="Authentication failed: invalid credentials",
        exception_type="AuthenticationError",
        error_code=401,
        service_name="auth_service"
    )
    
    print(f"Error: Authentication failed")
    print(f"  Category: {classification.category.value}")
    print(f"  Severity: {classification.severity.value}")
    print(f"  Handler: {result.handler_name}")
    print(f"  Escalated: {result.escalated}")
    print(f"  Escalation Level: {result.metadata.get('escalation_level')}")
    
    # Scenario 5: Resource exhaustion
    print("\n" + "-"*60)
    print("Scenario 5: Resource Exhaustion")
    print("-"*60)
    
    classification, result = agent.handle_error(
        error_message="Out of memory: heap exhausted",
        exception_type="MemoryError",
        error_code=507,
        service_name="compute_service"
    )
    
    print(f"Error: Out of memory")
    print(f"  Category: {classification.category.value}")
    print(f"  Severity: {classification.severity.value}")
    print(f"  Handler: {result.handler_name}")
    print(f"  Circuit State: {result.metadata.get('circuit_state')}")
    print(f"  Cooldown: {result.metadata.get('cooldown_seconds')}s")
    
    # Scenario 6: Multiple errors - batch processing
    print("\n" + "-"*60)
    print("Scenario 6: Batch Error Processing")
    print("-"*60)
    
    errors = [
        ("Connection timeout", "TimeoutError", 504),
        ("Data corruption detected", "DataError", None),
        ("Configuration file missing", "FileNotFoundError", 500),
        ("Transient failure occurred", "TransientError", 503)
    ]
    
    print("Processing batch of errors:\n")
    
    for error_msg, exc_type, code in errors:
        classification, result = agent.handle_error(
            error_message=error_msg,
            exception_type=exc_type,
            error_code=code
        )
        print(f"  {error_msg[:40]:40} → {result.strategy_used.value:15} ({result.handler_name})")
    
    # Scenario 7: Statistics
    print("\n" + "-"*60)
    print("Scenario 7: Error Handling Statistics")
    print("-"*60)
    
    stats = agent.get_statistics()
    
    print(f"\nTotal Errors Processed: {stats['total_errors']}")
    
    print("\nBy Category:")
    for category, count in sorted(stats['by_category'].items()):
        print(f"  {category}: {count}")
    
    print("\nBy Handler:")
    for handler, count in sorted(stats['by_handler'].items()):
        print(f"  {handler}: {count}")
    
    print("\nBy Strategy:")
    for strategy, count in sorted(stats['by_strategy'].items()):
        print(f"  {strategy}: {count}")
    
    print("\nHandler Utilization:")
    for handler, info in stats['handler_utilization'].items():
        print(f"  {handler}: {info['handled']} errors")
        print(f"    Categories: {', '.join(info['categories'])}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Classified {stats['total_errors']} errors across {len(stats['by_category'])} categories")
    print(f"✓ Routed to {len(stats['by_handler'])} specialized handlers")
    print(f"✓ Applied {len(stats['by_strategy'])} different recovery strategies")
    print(f"✓ Automatic error classification with confidence scoring")
    print(f"✓ Intelligent routing based on error type and severity")
    print("\n✅ Error Handling & Recovery Category: Pattern 3/5 complete")
    print("Ready for production error management!")


if __name__ == "__main__":
    demonstrate_error_classification_routing()
