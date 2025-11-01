"""
Pattern 148: Error Classification & Routing

Description:
    Implements intelligent error classification and routing to appropriate
    handlers based on error type, severity, and context. This pattern categorizes
    errors and routes them to specialized recovery strategies, enabling more
    sophisticated error handling than simple try-catch blocks.

Components:
    - Error Classifier: Categorizes errors by type and severity
    - Error Router: Routes errors to appropriate handlers
    - Handler Registry: Maps error types to handlers
    - Context Analyzer: Extracts error context
    - Recovery Strategies: Specialized recovery for each error type
    - Escalation Manager: Escalates unhandled errors

Use Cases:
    - Complex agent error handling
    - Multi-service application error management
    - Production system fault tolerance
    - Intelligent error recovery
    - Error analytics and monitoring

Benefits:
    - Targeted error recovery
    - Better error handling granularity
    - Reduced error resolution time
    - Improved system reliability
    - Better debugging and monitoring

Trade-offs:
    - Increased complexity
    - Requires comprehensive error taxonomy
    - Handler maintenance overhead
    - Potential routing errors

LangChain Implementation:
    Uses error classification, pattern matching, and routing to specialized
    handlers for intelligent error management.
"""

import os
import re
import traceback
from typing import Any, Callable, Optional, List, Dict, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ErrorCategory(Enum):
    """High-level error categories"""
    TRANSIENT = "transient"  # Temporary, retryable
    PERMANENT = "permanent"  # Persistent, not retryable
    RESOURCE = "resource"  # Resource-related (memory, disk, etc.)
    RATE_LIMIT = "rate_limit"  # API rate limiting
    AUTHENTICATION = "authentication"  # Auth/permission errors
    VALIDATION = "validation"  # Input validation errors
    TIMEOUT = "timeout"  # Timeout errors
    NETWORK = "network"  # Network connectivity
    BUSINESS_LOGIC = "business_logic"  # Business rule violations
    UNKNOWN = "unknown"  # Unclassified


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"  # Minor, can be ignored
    MEDIUM = "medium"  # Important, should be handled
    HIGH = "high"  # Critical, requires immediate attention
    CRITICAL = "critical"  # System-level, may need escalation


@dataclass
class ErrorContext:
    """Context information about an error"""
    exception: Exception
    exception_type: str
    message: str
    traceback_str: str
    timestamp: datetime
    operation: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassifiedError:
    """Classified error with category and severity"""
    context: ErrorContext
    category: ErrorCategory
    severity: ErrorSeverity
    retryable: bool
    suggested_action: str
    confidence: float = 1.0


@dataclass
class ErrorStats:
    """Statistics about error handling"""
    total_errors: int = 0
    by_category: Dict[ErrorCategory, int] = field(default_factory=lambda: defaultdict(int))
    by_severity: Dict[ErrorSeverity, int] = field(default_factory=lambda: defaultdict(int))
    by_handler: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    escalations: int = 0


class ErrorClassificationRouter:
    """
    Agent that classifies and routes errors to appropriate handlers.
    
    Analyzes errors, categorizes them, and routes to specialized recovery
    strategies for intelligent error management.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize the error classification router.
        
        Args:
            model: LLM model to use
        """
        self.llm = ChatOpenAI(model=model, temperature=0)
        
        self.handlers: Dict[ErrorCategory, Callable] = {}
        self.classification_rules: List[Dict[str, Any]] = []
        self.stats = ErrorStats()
        
        # Set up default classification rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Set up default error classification rules."""
        self.classification_rules = [
            {
                "patterns": ["rate limit", "429", "too many requests"],
                "category": ErrorCategory.RATE_LIMIT,
                "severity": ErrorSeverity.MEDIUM,
                "retryable": True,
                "action": "Retry with exponential backoff"
            },
            {
                "patterns": ["timeout", "timed out", "deadline exceeded"],
                "category": ErrorCategory.TIMEOUT,
                "severity": ErrorSeverity.MEDIUM,
                "retryable": True,
                "action": "Retry with increased timeout"
            },
            {
                "patterns": ["connection", "network", "unreachable", "503"],
                "category": ErrorCategory.NETWORK,
                "severity": ErrorSeverity.MEDIUM,
                "retryable": True,
                "action": "Check network and retry"
            },
            {
                "patterns": ["authentication", "unauthorized", "401", "403"],
                "category": ErrorCategory.AUTHENTICATION,
                "severity": ErrorSeverity.HIGH,
                "retryable": False,
                "action": "Verify credentials"
            },
            {
                "patterns": ["validation", "invalid", "bad request", "400"],
                "category": ErrorCategory.VALIDATION,
                "severity": ErrorSeverity.MEDIUM,
                "retryable": False,
                "action": "Fix input and retry"
            },
            {
                "patterns": ["out of memory", "memory error", "allocation failed"],
                "category": ErrorCategory.RESOURCE,
                "severity": ErrorSeverity.HIGH,
                "retryable": False,
                "action": "Reduce resource usage or scale up"
            },
            {
                "patterns": ["disk full", "no space left", "quota exceeded"],
                "category": ErrorCategory.RESOURCE,
                "severity": ErrorSeverity.HIGH,
                "retryable": False,
                "action": "Free up space or increase quota"
            }
        ]
    
    def classify_error(self, error: Exception, **context) -> ClassifiedError:
        """
        Classify an error based on type and message.
        
        Args:
            error: Exception to classify
            **context: Additional context
            
        Returns:
            Classified error with category and severity
        """
        self.stats.total_errors += 1
        
        # Create error context
        error_context = ErrorContext(
            exception=error,
            exception_type=type(error).__name__,
            message=str(error),
            traceback_str=traceback.format_exc(),
            timestamp=datetime.now(),
            operation=context.get("operation"),
            user_id=context.get("user_id"),
            request_id=context.get("request_id"),
            additional_data=context
        )
        
        # Try rule-based classification first
        classified = self._classify_by_rules(error_context)
        
        if classified.category == ErrorCategory.UNKNOWN:
            # Fall back to LLM-based classification
            classified = self._classify_by_llm(error_context)
        
        # Update stats
        self.stats.by_category[classified.category] += 1
        self.stats.by_severity[classified.severity] += 1
        
        return classified
    
    def _classify_by_rules(self, context: ErrorContext) -> ClassifiedError:
        """Classify error using rule-based matching."""
        error_str = f"{context.exception_type} {context.message}".lower()
        
        for rule in self.classification_rules:
            if any(pattern in error_str for pattern in rule["patterns"]):
                return ClassifiedError(
                    context=context,
                    category=rule["category"],
                    severity=rule["severity"],
                    retryable=rule["retryable"],
                    suggested_action=rule["action"],
                    confidence=1.0
                )
        
        # No rule matched
        return ClassifiedError(
            context=context,
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            retryable=False,
            suggested_action="Investigate error",
            confidence=0.0
        )
    
    def _classify_by_llm(self, context: ErrorContext) -> ClassifiedError:
        """Classify error using LLM."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert in error classification. "
                      "Classify the error into one of these categories:\n"
                      "- transient: Temporary, can be retried\n"
                      "- permanent: Persistent, cannot be retried\n"
                      "- resource: Resource-related (memory, disk)\n"
                      "- rate_limit: API rate limiting\n"
                      "- authentication: Auth/permission errors\n"
                      "- validation: Input validation errors\n"
                      "- timeout: Timeout errors\n"
                      "- network: Network connectivity\n"
                      "- business_logic: Business rule violations\n\n"
                      "Also determine severity (low/medium/high/critical) "
                      "and if it's retryable (yes/no).\n\n"
                      "Respond in format: category|severity|retryable|action"),
            ("user", "Error Type: {error_type}\n"
                    "Message: {message}\n"
                    "Context: {operation}\n\n"
                    "Classify this error:")
        ])
        
        try:
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "error_type": context.exception_type,
                "message": context.message,
                "operation": context.operation or "unknown"
            })
            
            # Parse result
            parts = result.split("|")
            if len(parts) >= 4:
                category = ErrorCategory(parts[0].strip().lower())
                severity = ErrorSeverity(parts[1].strip().lower())
                retryable = "yes" in parts[2].lower()
                action = parts[3].strip()
                
                return ClassifiedError(
                    context=context,
                    category=category,
                    severity=severity,
                    retryable=retryable,
                    suggested_action=action,
                    confidence=0.8
                )
                
        except Exception as e:
            print(f"LLM classification failed: {e}")
        
        # Fallback
        return ClassifiedError(
            context=context,
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            retryable=False,
            suggested_action="Manual investigation required",
            confidence=0.5
        )
    
    def register_handler(
        self,
        category: ErrorCategory,
        handler: Callable[[ClassifiedError], Any]
    ):
        """
        Register a handler for an error category.
        
        Args:
            category: Error category
            handler: Handler function
        """
        self.handlers[category] = handler
    
    def route_error(self, classified_error: ClassifiedError) -> Any:
        """
        Route error to appropriate handler.
        
        Args:
            classified_error: Classified error
            
        Returns:
            Handler result
        """
        category = classified_error.category
        
        if category in self.handlers:
            handler = self.handlers[category]
            handler_name = handler.__name__
            
            print(f"\n🔀 Routing to handler: {handler_name}")
            self.stats.by_handler[handler_name] += 1
            
            try:
                result = handler(classified_error)
                self.stats.successful_recoveries += 1
                return result
            except Exception as e:
                print(f"❌ Handler failed: {e}")
                self.stats.failed_recoveries += 1
                return self._default_handler(classified_error)
        else:
            print(f"⚠️  No handler for category: {category.value}")
            return self._default_handler(classified_error)
    
    def _default_handler(self, classified_error: ClassifiedError) -> Dict[str, Any]:
        """Default handler for unhandled errors."""
        if classified_error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            print(f"🚨 Escalating {classified_error.severity.value} severity error")
            self.stats.escalations += 1
        
        return {
            "handled": False,
            "category": classified_error.category.value,
            "severity": classified_error.severity.value,
            "action": classified_error.suggested_action,
            "escalated": classified_error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
        }
    
    def handle_error(self, error: Exception, **context) -> Any:
        """
        Complete error handling: classify and route.
        
        Args:
            error: Exception to handle
            **context: Additional context
            
        Returns:
            Handler result
        """
        # Classify error
        classified = self.classify_error(error, **context)
        
        print(f"\n📋 Error Classification:")
        print(f"   Category: {classified.category.value}")
        print(f"   Severity: {classified.severity.value}")
        print(f"   Retryable: {classified.retryable}")
        print(f"   Confidence: {classified.confidence:.1%}")
        print(f"   Suggested Action: {classified.suggested_action}")
        
        # Route to handler
        result = self.route_error(classified)
        
        return result
    
    def get_stats_report(self) -> str:
        """Generate error statistics report."""
        report = []
        report.append("\n" + "="*60)
        report.append("ERROR CLASSIFICATION & ROUTING STATISTICS")
        report.append("="*60)
        
        report.append(f"\n📊 Overall Statistics:")
        report.append(f"   Total Errors: {self.stats.total_errors}")
        report.append(f"   Successful Recoveries: {self.stats.successful_recoveries}")
        report.append(f"   Failed Recoveries: {self.stats.failed_recoveries}")
        report.append(f"   Escalations: {self.stats.escalations}")
        
        if self.stats.total_errors > 0:
            recovery_rate = (
                self.stats.successful_recoveries / self.stats.total_errors
            )
            report.append(f"   Recovery Rate: {recovery_rate:.1%}")
        
        report.append(f"\n📂 By Category:")
        for category, count in sorted(
            self.stats.by_category.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            report.append(f"   {category.value}: {count}")
        
        report.append(f"\n⚠️  By Severity:")
        for severity, count in sorted(
            self.stats.by_severity.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            report.append(f"   {severity.value}: {count}")
        
        if self.stats.by_handler:
            report.append(f"\n🔧 By Handler:")
            for handler, count in sorted(
                self.stats.by_handler.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                report.append(f"   {handler}: {count}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


def demonstrate_error_classification_routing():
    """Demonstrate the Error Classification & Routing pattern."""
    print("="*60)
    print("ERROR CLASSIFICATION & ROUTING PATTERN DEMONSTRATION")
    print("="*60)
    
    router = ErrorClassificationRouter()
    
    # Example 1: Register handlers for different error types
    print("\n" + "="*60)
    print("Example 1: Register Specialized Error Handlers")
    print("="*60)
    
    def handle_rate_limit(error: ClassifiedError) -> Dict:
        """Handler for rate limit errors."""
        print("   💤 Handling rate limit: waiting before retry...")
        return {"action": "retry_after_delay", "delay": 60}
    
    def handle_authentication(error: ClassifiedError) -> Dict:
        """Handler for authentication errors."""
        print("   🔐 Handling auth error: refreshing credentials...")
        return {"action": "refresh_credentials"}
    
    def handle_validation(error: ClassifiedError) -> Dict:
        """Handler for validation errors."""
        print("   ✏️  Handling validation error: correcting input...")
        return {"action": "fix_input", "message": error.context.message}
    
    def handle_network(error: ClassifiedError) -> Dict:
        """Handler for network errors."""
        print("   🌐 Handling network error: checking connection...")
        return {"action": "check_network_and_retry"}
    
    # Register handlers
    router.register_handler(ErrorCategory.RATE_LIMIT, handle_rate_limit)
    router.register_handler(ErrorCategory.AUTHENTICATION, handle_authentication)
    router.register_handler(ErrorCategory.VALIDATION, handle_validation)
    router.register_handler(ErrorCategory.NETWORK, handle_network)
    
    print("✅ Registered 4 specialized handlers")
    
    # Example 2: Classify and route various errors
    print("\n" + "="*60)
    print("Example 2: Classify and Route Various Errors")
    print("="*60)
    
    test_errors = [
        (Exception("Rate limit exceeded (429)"), "api_call"),
        (Exception("Connection timeout after 30s"), "database_query"),
        (Exception("Invalid API key provided"), "authentication"),
        (Exception("Invalid input: email format incorrect"), "user_registration"),
        (Exception("Network unreachable"), "external_service"),
    ]
    
    for error, operation in test_errors:
        print(f"\n{'='*60}")
        print(f"Error: {error}")
        print(f"Operation: {operation}")
        
        result = router.handle_error(error, operation=operation)
        print(f"Result: {result}")
    
    # Example 3: LLM-based classification for unknown errors
    print("\n" + "="*60)
    print("Example 3: LLM-Based Classification")
    print("="*60)
    
    unknown_error = Exception(
        "Failed to process payment: insufficient funds in account"
    )
    
    print(f"\nClassifying unknown error: {unknown_error}")
    classified = router.classify_error(
        unknown_error,
        operation="payment_processing"
    )
    
    print(f"\n📋 LLM Classification:")
    print(f"   Category: {classified.category.value}")
    print(f"   Severity: {classified.severity.value}")
    print(f"   Confidence: {classified.confidence:.1%}")
    print(f"   Suggested Action: {classified.suggested_action}")
    
    # Example 4: Error severity-based routing
    print("\n" + "="*60)
    print("Example 4: Severity-Based Escalation")
    print("="*60)
    
    severity_errors = [
        (Exception("Minor warning: cache miss"), ErrorSeverity.LOW),
        (Exception("API timeout on retry"), ErrorSeverity.MEDIUM),
        (Exception("Database connection pool exhausted"), ErrorSeverity.HIGH),
        (Exception("Critical system failure: all services down"), ErrorSeverity.CRITICAL),
    ]
    
    print("\nProcessing errors by severity:")
    for error, expected_severity in severity_errors:
        classified = router.classify_error(error)
        severity = classified.severity
        
        status = "🔴" if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else "🟡"
        print(f"\n{status} {severity.value.upper()}: {error}")
        
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            print(f"   ⚠️  Escalating to on-call team")
    
    # Example 5: Context-aware classification
    print("\n" + "="*60)
    print("Example 5: Context-Aware Error Classification")
    print("="*60)
    
    context_error = Exception("Operation failed")
    
    contexts = [
        {"operation": "payment_processing", "user_id": "user123", "amount": 1000},
        {"operation": "cache_refresh", "service": "recommendations"},
        {"operation": "user_login", "user_id": "admin", "ip": "192.168.1.1"}
    ]
    
    print("\nSame error in different contexts:")
    for ctx in contexts:
        print(f"\nContext: {ctx['operation']}")
        classified = router.classify_error(context_error, **ctx)
        print(f"   Severity: {classified.severity.value}")
        print(f"   Action: {classified.suggested_action}")
    
    # Example 6: Error pattern analysis
    print("\n" + "="*60)
    print("Example 6: Error Pattern Analysis")
    print("="*60)
    
    # Simulate multiple errors
    simulated_errors = [
        Exception("Rate limit exceeded"),
        Exception("Rate limit exceeded"),
        Exception("Rate limit exceeded"),
        Exception("Connection timeout"),
        Exception("Connection timeout"),
        Exception("Invalid credentials"),
    ]
    
    print("\nProcessing batch of errors:")
    for i, error in enumerate(simulated_errors, 1):
        router.handle_error(error, request_id=f"req-{i}")
    
    # Generate statistics
    stats_report = router.get_stats_report()
    print(stats_report)
    
    # Identify patterns
    print("\n💡 Pattern Analysis:")
    if router.stats.by_category[ErrorCategory.RATE_LIMIT] >= 3:
        print("   ⚠️  High rate of rate limit errors detected")
        print("   💡 Recommendation: Implement request throttling")
    
    if router.stats.by_category[ErrorCategory.NETWORK] >= 2:
        print("   ⚠️  Multiple network errors detected")
        print("   💡 Recommendation: Check service health")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
The Error Classification & Routing pattern demonstrates:

1. Intelligent Classification: Categorizes errors by type and severity
2. Specialized Handlers: Routes errors to appropriate recovery strategies
3. Rule-Based + LLM: Combines rule matching with AI classification
4. Context Awareness: Considers operation context in classification
5. Escalation Logic: Escalates severe errors automatically

Key Benefits:
- Targeted error recovery (80-95% automation typical)
- Reduced mean time to recovery (MTTR)
- Better error visibility and monitoring
- Intelligent escalation of critical issues
- Comprehensive error analytics

Error Categories:
- Transient: Temporary, retry with backoff
- Permanent: Persistent, manual intervention
- Rate Limit: Backoff and respect limits
- Authentication: Refresh credentials
- Validation: Fix input data
- Resource: Scale or optimize
- Network: Check connectivity
- Timeout: Increase limits or optimize

Handler Strategies:
- Retry: For transient errors
- Refresh: For authentication issues
- Validate: For input errors
- Scale: For resource errors
- Alert: For critical errors
- Log: For analysis and debugging

Best Practices:
- Maintain comprehensive error taxonomy
- Use rule-based classification first (fast)
- Fall back to LLM for unknown errors
- Log all errors with full context
- Monitor error patterns over time
- Update handlers based on learnings
- Test handlers thoroughly
- Document recovery procedures

Severity Escalation:
- Low: Log only
- Medium: Automated recovery
- High: Alert and automated recovery
- Critical: Immediate escalation to on-call

Use Cases:
- Production system error handling
- Multi-service applications
- API gateway error management
- Microservices fault tolerance
- Distributed system coordination
    """)


if __name__ == "__main__":
    demonstrate_error_classification_routing()
