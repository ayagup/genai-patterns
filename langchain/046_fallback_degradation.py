"""
Pattern 046: Fallback/Graceful Degradation

Description:
    Fallback/Graceful Degradation enables agents to maintain functionality when
    primary approaches fail by using alternative strategies at multiple levels.
    Implements cascading fallback mechanisms from optimal to minimal functionality,
    with human escalation as the final safety net.

Components:
    - Primary Handler: Optimal approach
    - Secondary Handlers: Fallback alternatives
    - Degradation Manager: Coordinates fallback levels
    - Error Classifier: Determines fallback strategy
    - Escalation Manager: Human intervention coordination

Use Cases:
    - Production systems requiring high availability
    - Critical applications with zero downtime needs
    - Multi-model deployments
    - API reliability improvements
    - Fault-tolerant systems
    - Customer-facing applications

LangChain Implementation:
    Uses tiered fallback chains with automatic degradation, error handling,
    and escalation to ensure continuous service availability.
"""

import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class FallbackLevel(Enum):
    """Levels of fallback degradation."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    MINIMAL = "minimal"
    HUMAN_ESCALATION = "human_escalation"


class FailureType(Enum):
    """Types of failures that trigger fallbacks."""
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    QUALITY_FAILURE = "quality_failure"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class FallbackAttempt:
    """Record of a fallback attempt."""
    level: FallbackLevel
    strategy: str
    success: bool
    error: Optional[str] = None
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FallbackResult:
    """Result of fallback execution."""
    query: str
    final_response: str
    final_level: FallbackLevel
    attempts: List[FallbackAttempt]
    total_fallbacks: int
    succeeded: bool
    total_latency_ms: float


class FallbackAgent:
    """
    Agent with cascading fallback mechanisms for reliability.
    
    Features:
    - Multi-level fallback strategies
    - Automatic degradation
    - Error classification
    - Performance tracking
    - Human escalation
    """
    
    def __init__(
        self,
        max_fallback_attempts: int = 4,
        timeout_seconds: float = 10.0
    ):
        self.max_fallback_attempts = max_fallback_attempts
        self.timeout_seconds = timeout_seconds
        
        # Different LLM configurations for fallback levels
        self.primary_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            request_timeout=timeout_seconds
        )
        
        self.secondary_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.5,  # More conservative
            request_timeout=timeout_seconds / 2
        )
        
        # Prompts for different levels
        self.primary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI assistant. Provide detailed, comprehensive responses."),
            ("user", "{query}")
        ])
        
        self.secondary_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Provide clear, concise responses."),
            ("user", "{query}")
        ])
        
        self.minimal_prompt = ChatPromptTemplate.from_messages([
            ("system", "Provide a brief, factual response."),
            ("user", "{query}")
        ])
        
        # Execution history
        self.executions: List[FallbackResult] = []
    
    def _primary_strategy(self, query: str) -> str:
        """
        Primary strategy: Full-featured response.
        
        Returns:
            Detailed response
        """
        chain = self.primary_prompt | self.primary_llm | StrOutputParser()
        return chain.invoke({"query": query})
    
    def _secondary_strategy(self, query: str) -> str:
        """
        Secondary strategy: Conservative response.
        
        Returns:
            Concise response
        """
        chain = self.secondary_prompt | self.secondary_llm | StrOutputParser()
        return chain.invoke({"query": query})
    
    def _tertiary_strategy(self, query: str) -> str:
        """
        Tertiary strategy: Minimal response.
        
        Returns:
            Brief response
        """
        chain = self.minimal_prompt | self.secondary_llm | StrOutputParser()
        return chain.invoke({"query": query})
    
    def _minimal_strategy(self, query: str) -> str:
        """
        Minimal strategy: Template-based response.
        
        Returns:
            Template response
        """
        return f"I understand you're asking about: '{query}'. Due to high system load, I can only provide a basic acknowledgment. Please try again shortly or contact support for detailed assistance."
    
    def _human_escalation(self, query: str) -> str:
        """
        Human escalation: Request human intervention.
        
        Returns:
            Escalation message
        """
        return f"This query requires human assistance: '{query}'. A human operator has been notified and will respond shortly. Reference ID: {datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def execute_with_fallback(
        self,
        query: str
    ) -> FallbackResult:
        """
        Execute query with automatic fallback on failures.
        
        Fallback cascade:
        1. PRIMARY: Full-featured response (GPT-3.5, detailed)
        2. SECONDARY: Conservative response (GPT-3.5, concise)
        3. TERTIARY: Minimal LLM response (brief)
        4. MINIMAL: Template response (no LLM)
        5. HUMAN_ESCALATION: Request human help
        
        Args:
            query: User query
            
        Returns:
            FallbackResult with execution details
        """
        attempts = []
        start_time = datetime.now()
        
        # Define fallback strategies
        strategies = [
            (FallbackLevel.PRIMARY, "Full-featured GPT-3.5", self._primary_strategy),
            (FallbackLevel.SECONDARY, "Conservative GPT-3.5", self._secondary_strategy),
            (FallbackLevel.TERTIARY, "Minimal LLM", self._tertiary_strategy),
            (FallbackLevel.MINIMAL, "Template response", self._minimal_strategy),
            (FallbackLevel.HUMAN_ESCALATION, "Human escalation", self._human_escalation),
        ]
        
        final_response = None
        final_level = None
        
        for level, strategy_name, strategy_func in strategies:
            attempt_start = datetime.now()
            
            try:
                response = strategy_func(query)
                attempt_latency = (datetime.now() - attempt_start).total_seconds() * 1000
                
                # Validate response
                if response and len(response.strip()) > 0:
                    attempts.append(FallbackAttempt(
                        level=level,
                        strategy=strategy_name,
                        success=True,
                        latency_ms=attempt_latency
                    ))
                    
                    final_response = response
                    final_level = level
                    break  # Success!
                else:
                    raise ValueError("Empty response")
            
            except Exception as e:
                attempt_latency = (datetime.now() - attempt_start).total_seconds() * 1000
                attempts.append(FallbackAttempt(
                    level=level,
                    strategy=strategy_name,
                    success=False,
                    error=str(e),
                    latency_ms=attempt_latency
                ))
                
                # Continue to next fallback
                continue
        
        # If all fallbacks failed (shouldn't happen with human escalation)
        if final_response is None:
            final_response = "System temporarily unavailable. Please try again later."
            final_level = FallbackLevel.HUMAN_ESCALATION
        
        total_latency = (datetime.now() - start_time).total_seconds() * 1000
        
        result = FallbackResult(
            query=query,
            final_response=final_response,
            final_level=final_level,
            attempts=attempts,
            total_fallbacks=len(attempts) - 1,  # -1 because first attempt isn't a fallback
            succeeded=final_response is not None,
            total_latency_ms=total_latency
        )
        
        self.executions.append(result)
        
        return result
    
    def get_reliability_statistics(self) -> Dict[str, Any]:
        """Get statistics about fallback usage and reliability."""
        if not self.executions:
            return {"total_executions": 0}
        
        total = len(self.executions)
        primary_success = sum(1 for e in self.executions if e.final_level == FallbackLevel.PRIMARY)
        required_fallback = sum(1 for e in self.executions if e.total_fallbacks > 0)
        
        level_usage = {}
        for level in FallbackLevel:
            count = sum(1 for e in self.executions if e.final_level == level)
            if count > 0:
                level_usage[level.value] = count
        
        avg_latency = sum(e.total_latency_ms for e in self.executions) / total
        avg_fallbacks = sum(e.total_fallbacks for e in self.executions) / total
        
        return {
            "total_executions": total,
            "primary_success_count": primary_success,
            "primary_success_rate": primary_success / total,
            "required_fallback_count": required_fallback,
            "fallback_rate": required_fallback / total,
            "average_fallbacks_per_execution": avg_fallbacks,
            "average_latency_ms": avg_latency,
            "level_usage": level_usage
        }


def demonstrate_fallback_graceful_degradation():
    """
    Demonstrates fallback and graceful degradation mechanisms.
    """
    print("=" * 80)
    print("FALLBACK/GRACEFUL DEGRADATION DEMONSTRATION")
    print("=" * 80)
    
    # Create fallback agent
    agent = FallbackAgent(
        max_fallback_attempts=4,
        timeout_seconds=30.0
    )
    
    # Test 1: Normal execution (should use primary)
    print("\n" + "=" * 80)
    print("Test 1: Normal Execution")
    print("=" * 80)
    
    query1 = "What are the benefits of regular exercise?"
    print(f"\nQuery: {query1}")
    
    result1 = agent.execute_with_fallback(query1)
    
    print("\n[Execution Summary]")
    print(f"Final Level: {result1.final_level.value}")
    print(f"Total Fallbacks: {result1.total_fallbacks}")
    print(f"Succeeded: {result1.succeeded}")
    print(f"Total Latency: {result1.total_latency_ms:.2f}ms")
    
    print("\n[Attempts]")
    for i, attempt in enumerate(result1.attempts, 1):
        status = "✓" if attempt.success else "✗"
        print(f"{i}. {status} {attempt.level.value} - {attempt.strategy}")
        print(f"   Latency: {attempt.latency_ms:.2f}ms")
        if attempt.error:
            print(f"   Error: {attempt.error[:100]}...")
    
    print("\n[Response Preview]")
    print(result1.final_response[:250] + "..." if len(result1.final_response) > 250 else result1.final_response)
    
    # Test 2: Multiple queries to show fallback patterns
    print("\n" + "=" * 80)
    print("Test 2: Multiple Executions")
    print("=" * 80)
    
    test_queries = [
        "How does photosynthesis work?",
        "What is machine learning?",
        "Explain the water cycle",
    ]
    
    print("\nExecuting multiple queries...")
    for query in test_queries:
        result = agent.execute_with_fallback(query)
        print(f"\nQuery: {query[:50]}...")
        print(f"  Level: {result.final_level.value}, Fallbacks: {result.total_fallbacks}, Latency: {result.total_latency_ms:.0f}ms")
    
    # Test 3: Simulated fallback scenario
    print("\n" + "=" * 80)
    print("Test 3: Fallback Cascade Visualization")
    print("=" * 80)
    
    query3 = "Explain quantum computing"
    print(f"\nQuery: {query3}")
    
    result3 = agent.execute_with_fallback(query3)
    
    print("\n[Fallback Cascade]")
    print("Level Hierarchy:")
    print("  1. PRIMARY      → Full-featured response")
    print("  2. SECONDARY    → Conservative response")
    print("  3. TERTIARY     → Minimal LLM response")
    print("  4. MINIMAL      → Template response")
    print("  5. ESCALATION   → Human intervention")
    
    print(f"\n Executed Level: {result3.final_level.value}")
    print(f"  Result: {'SUCCESS ✓' if result3.succeeded else 'FAILED ✗'}")
    
    # Show reliability statistics
    print("\n" + "=" * 80)
    print("Reliability Statistics")
    print("=" * 80)
    
    stats = agent.get_reliability_statistics()
    
    print(f"\nTotal Executions: {stats['total_executions']}")
    print(f"Primary Success Count: {stats['primary_success_count']}")
    print(f"Primary Success Rate: {stats['primary_success_rate']:.1%}")
    print(f"Required Fallback Count: {stats['required_fallback_count']}")
    print(f"Fallback Rate: {stats['fallback_rate']:.1%}")
    print(f"Average Fallbacks per Execution: {stats['average_fallbacks_per_execution']:.2f}")
    print(f"Average Latency: {stats['average_latency_ms']:.2f}ms")
    
    print("\n[Level Usage Distribution]")
    for level, count in sorted(stats['level_usage'].items(), key=lambda x: list(FallbackLevel).index(FallbackLevel(x[0]))):
        percentage = (count / stats['total_executions']) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {level:<20} {bar} {count} ({percentage:.1f}%)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Fallback/Graceful Degradation provides:
✓ Multi-level fallback strategies
✓ Automatic degradation
✓ Continuous availability
✓ Error resilience
✓ Performance tracking
✓ Human escalation

This pattern excels at:
- Production systems
- High-availability applications
- Critical services
- API reliability
- Fault tolerance
- Customer-facing systems

Fallback levels:
1. PRIMARY: Optimal performance
   - Full-featured LLM
   - Detailed responses
   - Best quality

2. SECONDARY: Conservative
   - Same LLM, adjusted parameters
   - Concise responses
   - Good quality

3. TERTIARY: Minimal LLM
   - Brief responses
   - Basic quality
   - Fast execution

4. MINIMAL: Template-based
   - No LLM required
   - Acknowledgment only
   - Immediate response

5. HUMAN_ESCALATION: Last resort
   - Request human help
   - Provide reference ID
   - Guaranteed response

Fallback triggers:
- Timeouts: Slow responses
- API Errors: Service failures
- Rate Limits: Quota exceeded
- Quality Failures: Poor outputs
- Unavailability: Service down

Benefits:
- Reliability: Always available
- Resilience: Handles failures
- Performance: Optimizes resources
- User Experience: No downtime
- Flexibility: Multiple strategies
- Graceful: Degraded but functional

Degradation strategy:
- Quality: High → Medium → Low
- Latency: Accept longer waits at higher levels
- Cost: Expensive → Cheap → Free
- Complexity: Complex → Simple → Basic

Use Fallback/Graceful Degradation when:
- High availability required
- Cannot afford downtime
- Multiple failure modes possible
- User experience critical
- Production deployments
- SLA requirements exist

Configuration:
- max_fallback_attempts: How many levels
- timeout_seconds: Per-level timeout
- fallback_strategies: Custom handlers
- escalation_policy: Human notification

Comparison with other patterns:
- vs Circuit Breaker: Fallback vs prevention
- vs Retry: Different strategies vs same attempt
- vs Redundancy: Degradation vs duplication
""")


if __name__ == "__main__":
    demonstrate_fallback_graceful_degradation()
