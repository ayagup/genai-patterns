"""
Pattern 150: Error Recovery Strategies

Description:
    Implements comprehensive error recovery mechanisms with multiple fallback strategies.
    Goes beyond simple retry to include alternative approaches, degraded service modes,
    and intelligent recovery path selection.

Components:
    - Recovery Strategy Registry: Maps error types to recovery strategies
    - Strategy Selector: Chooses appropriate recovery approach
    - Fallback Chain: Ordered list of alternative strategies
    - Recovery Context: Tracks recovery attempts and outcomes
    - Strategy Executor: Executes selected recovery strategy

Use Cases:
    - Production systems requiring high availability
    - Multi-step workflows with diverse failure modes
    - Systems with multiple service dependencies
    - Critical operations requiring guaranteed completion
    - Graceful degradation scenarios

Benefits:
    - Improved system resilience
    - Better user experience during failures
    - Reduced manual intervention
    - Comprehensive failure handling
    - Intelligent recovery path selection

Trade-offs:
    - Increased system complexity
    - More code paths to test
    - Potential for unexpected behaviors
    - Recovery strategy maintenance overhead
    - May delay error reporting

LangChain Implementation:
    Uses custom error handling with LangChain chains, fallback chains,
    and strategy pattern for recovery mechanisms.
"""

import os
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


class RecoveryStrategyType(Enum):
    """Types of recovery strategies"""
    RETRY = "retry"
    FALLBACK_MODEL = "fallback_model"
    SIMPLIFIED_REQUEST = "simplified_request"
    CACHED_RESPONSE = "cached_response"
    DEFAULT_RESPONSE = "default_response"
    ALTERNATIVE_APPROACH = "alternative_approach"
    DEGRADED_SERVICE = "degraded_service"
    HUMAN_ESCALATION = "human_escalation"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RecoveryContext:
    """Context for tracking recovery attempts"""
    original_request: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    attempts: int = 0
    max_attempts: int = 3
    attempted_strategies: List[RecoveryStrategyType] = field(default_factory=list)
    recovery_history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def record_attempt(self, strategy: RecoveryStrategyType, success: bool, 
                      result: Any = None, error: str = None):
        """Record a recovery attempt"""
        self.attempts += 1
        self.attempted_strategies.append(strategy)
        self.recovery_history.append({
            "strategy": strategy.value,
            "success": success,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })


@dataclass
class RecoveryStrategy:
    """Defines a recovery strategy"""
    name: RecoveryStrategyType
    priority: int  # Lower number = higher priority
    applicable_errors: List[str]  # Error patterns this strategy can handle
    max_retries: int = 3
    timeout: float = 30.0
    executor: Optional[Callable] = None


class ErrorRecoveryAgent:
    """
    Agent that implements comprehensive error recovery strategies.
    Automatically selects and executes appropriate recovery mechanisms.
    """
    
    def __init__(self, temperature: float = 0.7):
        """Initialize the error recovery agent"""
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        self.fallback_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        
        # Response cache for fallback
        self.response_cache: Dict[str, Any] = {}
        
        # Registry of recovery strategies
        self.strategies: List[RecoveryStrategy] = self._initialize_strategies()
        
        # Prompts
        self.main_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Provide accurate and comprehensive responses."),
            ("user", "{query}")
        ])
        
        self.simplified_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Provide a brief, simple response."),
            ("user", "Provide a short answer to: {query}")
        ])
        
        self.degraded_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant operating in degraded mode. Provide basic, limited responses."),
            ("user", "In one sentence, respond to: {query}")
        ])
        
    def _initialize_strategies(self) -> List[RecoveryStrategy]:
        """Initialize recovery strategy registry"""
        return [
            RecoveryStrategy(
                name=RecoveryStrategyType.RETRY,
                priority=1,
                applicable_errors=["timeout", "rate_limit", "connection"],
                max_retries=3,
                executor=self._retry_strategy
            ),
            RecoveryStrategy(
                name=RecoveryStrategyType.FALLBACK_MODEL,
                priority=2,
                applicable_errors=["model_error", "api_error"],
                executor=self._fallback_model_strategy
            ),
            RecoveryStrategy(
                name=RecoveryStrategyType.SIMPLIFIED_REQUEST,
                priority=3,
                applicable_errors=["context_length", "complexity"],
                executor=self._simplified_request_strategy
            ),
            RecoveryStrategy(
                name=RecoveryStrategyType.CACHED_RESPONSE,
                priority=4,
                applicable_errors=["all"],
                executor=self._cached_response_strategy
            ),
            RecoveryStrategy(
                name=RecoveryStrategyType.ALTERNATIVE_APPROACH,
                priority=5,
                applicable_errors=["all"],
                executor=self._alternative_approach_strategy
            ),
            RecoveryStrategy(
                name=RecoveryStrategyType.DEGRADED_SERVICE,
                priority=6,
                applicable_errors=["all"],
                executor=self._degraded_service_strategy
            ),
            RecoveryStrategy(
                name=RecoveryStrategyType.DEFAULT_RESPONSE,
                priority=7,
                applicable_errors=["all"],
                executor=self._default_response_strategy
            ),
            RecoveryStrategy(
                name=RecoveryStrategyType.HUMAN_ESCALATION,
                priority=8,
                applicable_errors=["all"],
                executor=self._human_escalation_strategy
            )
        ]
    
    def execute_with_recovery(self, query: str, max_recovery_attempts: int = 5) -> Dict[str, Any]:
        """
        Execute query with comprehensive error recovery
        
        Args:
            query: Query to execute
            max_recovery_attempts: Maximum recovery attempts
            
        Returns:
            Dictionary with result and recovery information
        """
        start_time = time.time()
        
        # Try primary execution
        try:
            result = self._primary_execution(query)
            
            return {
                "success": True,
                "result": result,
                "recovery_used": False,
                "execution_time": time.time() - start_time
            }
            
        except Exception as e:
            # Primary execution failed, initiate recovery
            context = self._create_recovery_context(query, e)
            
            recovery_result = self._execute_recovery(context, max_recovery_attempts)
            
            return {
                "success": recovery_result["success"],
                "result": recovery_result.get("result"),
                "recovery_used": True,
                "recovery_attempts": context.attempts,
                "strategies_attempted": [s.value for s in context.attempted_strategies],
                "recovery_history": context.recovery_history,
                "execution_time": time.time() - start_time,
                "error": recovery_result.get("error")
            }
    
    def _primary_execution(self, query: str) -> str:
        """Execute primary request"""
        chain = self.main_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"query": query})
        
        # Cache successful response
        self.response_cache[query] = result
        
        return result
    
    def _create_recovery_context(self, query: str, error: Exception) -> RecoveryContext:
        """Create recovery context from error"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Determine severity
        severity = self._determine_severity(error_type, error_message)
        
        return RecoveryContext(
            original_request=query,
            error_type=error_type,
            error_message=error_message,
            severity=severity
        )
    
    def _determine_severity(self, error_type: str, error_message: str) -> ErrorSeverity:
        """Determine error severity"""
        critical_patterns = ["critical", "fatal", "security"]
        high_patterns = ["failed", "error", "exception"]
        medium_patterns = ["timeout", "rate_limit"]
        
        error_lower = f"{error_type} {error_message}".lower()
        
        if any(p in error_lower for p in critical_patterns):
            return ErrorSeverity.CRITICAL
        elif any(p in error_lower for p in high_patterns):
            return ErrorSeverity.HIGH
        elif any(p in error_lower for p in medium_patterns):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _execute_recovery(self, context: RecoveryContext, 
                         max_attempts: int) -> Dict[str, Any]:
        """Execute recovery strategies"""
        
        # Select applicable strategies
        strategies = self._select_strategies(context)
        
        for strategy in strategies:
            if context.attempts >= max_attempts:
                break
            
            if strategy.name in context.attempted_strategies:
                continue
            
            print(f"\n🔄 Attempting recovery with strategy: {strategy.name.value}")
            
            try:
                result = strategy.executor(context)
                
                if result is not None:
                    context.record_attempt(strategy.name, True, result)
                    return {
                        "success": True,
                        "result": result,
                        "recovery_strategy": strategy.name.value
                    }
                else:
                    context.record_attempt(strategy.name, False, 
                                         error="Strategy returned None")
                    
            except Exception as e:
                context.record_attempt(strategy.name, False, error=str(e))
                print(f"   ❌ Strategy failed: {str(e)}")
        
        # All strategies failed
        return {
            "success": False,
            "error": f"All recovery strategies exhausted. Original error: {context.error_message}"
        }
    
    def _select_strategies(self, context: RecoveryContext) -> List[RecoveryStrategy]:
        """Select applicable recovery strategies"""
        applicable = []
        
        for strategy in self.strategies:
            # Check if strategy applies to this error
            if "all" in strategy.applicable_errors:
                applicable.append(strategy)
            elif any(pattern in context.error_type.lower() or 
                    pattern in context.error_message.lower() 
                    for pattern in strategy.applicable_errors):
                applicable.append(strategy)
        
        # Sort by priority
        return sorted(applicable, key=lambda s: s.priority)
    
    # Recovery Strategy Implementations
    
    def _retry_strategy(self, context: RecoveryContext) -> Optional[str]:
        """Simple retry with exponential backoff"""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                
                print(f"   🔄 Retry attempt {attempt + 1}/{max_retries}")
                return self._primary_execution(context.original_request)
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"   ❌ All retries failed")
                    return None
        
        return None
    
    def _fallback_model_strategy(self, context: RecoveryContext) -> Optional[str]:
        """Use fallback model with different settings"""
        try:
            print(f"   🔄 Using fallback model")
            chain = self.main_prompt | self.fallback_llm | StrOutputParser()
            return chain.invoke({"query": context.original_request})
        except Exception as e:
            print(f"   ❌ Fallback model failed: {str(e)}")
            return None
    
    def _simplified_request_strategy(self, context: RecoveryContext) -> Optional[str]:
        """Simplify request to reduce complexity"""
        try:
            print(f"   🔄 Using simplified request")
            chain = self.simplified_prompt | self.llm | StrOutputParser()
            return chain.invoke({"query": context.original_request})
        except Exception as e:
            print(f"   ❌ Simplified request failed: {str(e)}")
            return None
    
    def _cached_response_strategy(self, context: RecoveryContext) -> Optional[str]:
        """Return cached response if available"""
        if context.original_request in self.response_cache:
            print(f"   ✅ Using cached response")
            return self.response_cache[context.original_request]
        
        # Look for similar cached queries
        for cached_query, cached_response in self.response_cache.items():
            if self._queries_similar(context.original_request, cached_query):
                print(f"   ✅ Using similar cached response")
                return f"[Based on similar query] {cached_response}"
        
        return None
    
    def _alternative_approach_strategy(self, context: RecoveryContext) -> Optional[str]:
        """Try alternative approach using LLM suggestion"""
        try:
            print(f"   🔄 Generating alternative approach")
            
            alt_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. The user asked: '{original_query}' but we encountered an error. Provide a helpful response that addresses their underlying need."),
                ("user", "Provide an alternative response to: {query}")
            ])
            
            chain = alt_prompt | self.llm | StrOutputParser()
            return chain.invoke({
                "query": context.original_request,
                "original_query": context.original_request
            })
        except Exception as e:
            print(f"   ❌ Alternative approach failed: {str(e)}")
            return None
    
    def _degraded_service_strategy(self, context: RecoveryContext) -> Optional[str]:
        """Provide degraded service with limited functionality"""
        try:
            print(f"   🔄 Using degraded service mode")
            chain = self.degraded_prompt | self.fallback_llm | StrOutputParser()
            result = chain.invoke({"query": context.original_request})
            return f"[Degraded Mode] {result}"
        except Exception as e:
            print(f"   ❌ Degraded service failed: {str(e)}")
            return None
    
    def _default_response_strategy(self, context: RecoveryContext) -> Optional[str]:
        """Return safe default response"""
        print(f"   ✅ Using default response")
        return (f"I apologize, but I'm currently unable to fully process your request "
                f"due to technical difficulties. Your request was: '{context.original_request}'. "
                f"Please try again later or rephrase your question.")
    
    def _human_escalation_strategy(self, context: RecoveryContext) -> Optional[str]:
        """Escalate to human operator"""
        print(f"   🚨 Escalating to human operator")
        return (f"[ESCALATED TO HUMAN OPERATOR]\n"
                f"Request: {context.original_request}\n"
                f"Error: {context.error_type} - {context.error_message}\n"
                f"Severity: {context.severity.value}\n"
                f"Attempted strategies: {[s.value for s in context.attempted_strategies]}\n"
                f"A human operator will review this request.")
    
    def _queries_similar(self, query1: str, query2: str, threshold: float = 0.7) -> bool:
        """Check if two queries are similar (simple implementation)"""
        # Simple word overlap similarity
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        similarity = overlap / max(len(words1), len(words2))
        
        return similarity >= threshold
    
    def generate_recovery_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive recovery report"""
        total = len(results)
        successful = sum(1 for r in results if r["success"])
        with_recovery = sum(1 for r in results if r.get("recovery_used", False))
        
        report = [
            "\n" + "="*70,
            "ERROR RECOVERY REPORT",
            "="*70,
            f"\nTotal Executions: {total}",
            f"Successful: {successful} ({successful/total*100:.1f}%)",
            f"Required Recovery: {with_recovery} ({with_recovery/total*100:.1f}%)",
            f"Failed: {total - successful} ({(total-successful)/total*100:.1f}%)"
        ]
        
        # Strategy usage statistics
        strategy_usage = {}
        for result in results:
            if result.get("strategies_attempted"):
                for strategy in result["strategies_attempted"]:
                    strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        if strategy_usage:
            report.append("\nRecovery Strategy Usage:")
            for strategy, count in sorted(strategy_usage.items(), 
                                         key=lambda x: x[1], reverse=True):
                report.append(f"  - {strategy}: {count} times")
        
        # Performance metrics
        avg_time = sum(r["execution_time"] for r in results) / total
        recovery_times = [r["execution_time"] for r in results if r.get("recovery_used")]
        avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0
        
        report.extend([
            f"\nPerformance Metrics:",
            f"  - Average execution time: {avg_time:.2f}s",
            f"  - Average recovery time: {avg_recovery_time:.2f}s"
        ])
        
        report.append("="*70)
        return "\n".join(report)


def demonstrate_error_recovery():
    """Demonstrate comprehensive error recovery strategies"""
    print("="*70)
    print("Pattern 150: Error Recovery Strategies")
    print("="*70)
    
    agent = ErrorRecoveryAgent()
    results = []
    
    # Test cases
    test_cases = [
        {
            "query": "What is the capital of France?",
            "description": "Normal query (should succeed)"
        },
        {
            "query": "Explain quantum computing in detail",
            "description": "Complex query (may need simplified approach)"
        },
        {
            "query": "What is 2 + 2?",
            "description": "Simple query (should use cache on retry)"
        },
        {
            "query": "Translate this ancient Sumerian text: ...",
            "description": "Difficult query (may need alternative approach)"
        },
        {
            "query": "Generate a 50-page research paper on...",
            "description": "Oversized request (should use degraded mode)"
        }
    ]
    
    print("\n" + "="*70)
    print("TESTING ERROR RECOVERY MECHANISMS")
    print("="*70)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}: {test_case['description']}")
        print(f"Query: {test_case['query'][:60]}...")
        print(f"{'='*70}")
        
        result = agent.execute_with_recovery(test_case["query"])
        results.append(result)
        
        print(f"\n✅ Success: {result['success']}")
        if result['recovery_used']:
            print(f"🔄 Recovery Used: Yes")
            print(f"   Attempts: {result['recovery_attempts']}")
            print(f"   Strategies: {', '.join(result['strategies_attempted'])}")
        else:
            print(f"🔄 Recovery Used: No (primary execution succeeded)")
        
        if result['success']:
            response = result['result'][:100] + "..." if len(result['result']) > 100 else result['result']
            print(f"📝 Response: {response}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        print(f"⏱️  Execution time: {result['execution_time']:.2f}s")
    
    # Generate comprehensive report
    print("\n" + agent.generate_recovery_report(results))
    
    # Demonstrate recovery history
    print("\n" + "="*70)
    print("RECOVERY HISTORY EXAMPLE")
    print("="*70)
    
    complex_result = [r for r in results if r.get('recovery_used', False)]
    if complex_result:
        example = complex_result[0]
        if 'recovery_history' in example:
            print("\nDetailed Recovery History:")
            for i, attempt in enumerate(example['recovery_history'], 1):
                print(f"\nAttempt {i}:")
                print(f"  Strategy: {attempt['strategy']}")
                print(f"  Success: {attempt['success']}")
                if attempt['success']:
                    result_preview = str(attempt['result'])[:60]
                    print(f"  Result: {result_preview}...")
                else:
                    print(f"  Error: {attempt.get('error', 'Unknown')}")
                print(f"  Timestamp: {attempt['timestamp']}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Error Recovery Strategies Pattern provides:

1. Multiple Recovery Mechanisms:
   - Retry with backoff
   - Fallback models
   - Simplified requests
   - Cached responses
   - Alternative approaches
   - Degraded service modes
   - Default responses
   - Human escalation

2. Intelligent Strategy Selection:
   - Error type analysis
   - Severity assessment
   - Priority-based ordering
   - Context-aware selection

3. Comprehensive Tracking:
   - Recovery attempts
   - Strategy effectiveness
   - Performance metrics
   - Detailed history

4. Production Ready:
   - Graceful degradation
   - High availability
   - User experience focus
   - Monitoring and reporting

This pattern is essential for building resilient, production-grade
AI systems that maintain functionality even under adverse conditions.
    """)


if __name__ == "__main__":
    demonstrate_error_recovery()
