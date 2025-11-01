"""
Pattern 153: Shadow Mode Testing

Description:
    Runs new agent versions alongside production without affecting users.
    Captures both production and shadow responses, compares results, and
    identifies differences for validation before full deployment.

Components:
    - Shadow Executor: Runs shadow version in parallel
    - Response Comparator: Compares production vs shadow outputs
    - Difference Analyzer: Identifies and categorizes differences
    - Metrics Collector: Tracks shadow performance
    - Rollback Monitor: Detects degradation signals

Use Cases:
    - Testing new model versions safely
    - Validating prompt changes
    - A/B testing without user impact
    - Gradual feature rollout
    - Performance regression detection

Benefits:
    - Zero user impact during testing
    - Real production traffic testing
    - Early problem detection
    - Confidence before full deployment
    - Comprehensive comparison data

Trade-offs:
    - Doubled computational cost
    - Infrastructure complexity
    - Storage for dual responses
    - Latency considerations
    - Analysis overhead

LangChain Implementation:
    Uses parallel execution with ChatOpenAI, response comparison,
    and comprehensive metrics tracking.
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class DifferenceType(Enum):
    """Types of differences between responses"""
    IDENTICAL = "identical"
    MINOR = "minor"  # Small wording differences
    MODERATE = "moderate"  # Different approach, similar meaning
    MAJOR = "major"  # Significantly different
    ERROR = "error"  # One succeeded, one failed


class HealthStatus(Enum):
    """Health status of shadow version"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    FAILING = "failing"


@dataclass
class ShadowResponse:
    """Response from shadow execution"""
    production_response: str
    shadow_response: str
    production_time: float
    shadow_time: float
    production_error: Optional[str] = None
    shadow_error: Optional[str] = None
    difference_type: Optional[DifferenceType] = None
    similarity_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ShadowMetrics:
    """Metrics for shadow testing"""
    total_requests: int = 0
    identical_responses: int = 0
    minor_differences: int = 0
    moderate_differences: int = 0
    major_differences: int = 0
    production_errors: int = 0
    shadow_errors: int = 0
    avg_production_latency: float = 0.0
    avg_shadow_latency: float = 0.0
    shadow_slower_by: float = 0.0  # Percentage
    
    def get_difference_rate(self, diff_type: DifferenceType) -> float:
        """Get rate of specific difference type"""
        if self.total_requests == 0:
            return 0.0
        
        count_map = {
            DifferenceType.IDENTICAL: self.identical_responses,
            DifferenceType.MINOR: self.minor_differences,
            DifferenceType.MODERATE: self.moderate_differences,
            DifferenceType.MAJOR: self.major_differences
        }
        
        return count_map.get(diff_type, 0) / self.total_requests
    
    def get_health_status(self) -> HealthStatus:
        """Determine overall health status"""
        if self.total_requests < 10:
            return HealthStatus.HEALTHY  # Not enough data
        
        major_rate = self.get_difference_rate(DifferenceType.MAJOR)
        error_rate = self.shadow_errors / self.total_requests
        
        if error_rate > 0.3 or major_rate > 0.5:
            return HealthStatus.FAILING
        elif error_rate > 0.1 or major_rate > 0.3:
            return HealthStatus.DEGRADED
        elif error_rate > 0.05 or major_rate > 0.15:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


class ShadowModeTester:
    """
    Agent that runs shadow testing by executing production and shadow
    versions in parallel, comparing results without affecting users.
    """
    
    def __init__(self, production_model: str = "gpt-3.5-turbo",
                 shadow_model: str = "gpt-3.5-turbo",
                 production_temp: float = 0.7,
                 shadow_temp: float = 0.7):
        """Initialize shadow mode tester"""
        # Production agent
        self.production_llm = ChatOpenAI(
            model=production_model,
            temperature=production_temp
        )
        
        # Shadow agent (different version or configuration)
        self.shadow_llm = ChatOpenAI(
            model=shadow_model,
            temperature=shadow_temp
        )
        
        # Metrics tracking
        self.metrics = ShadowMetrics()
        self.responses: List[ShadowResponse] = []
        
        # Difference categorization LLM
        self.analyzer_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        
        # Prompts
        self.production_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Provide accurate, clear responses."),
            ("user", "{query}")
        ])
        
        self.shadow_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Provide accurate, clear responses."),
            ("user", "{query}")
        ])
        
        self.comparison_prompt = ChatPromptTemplate.from_messages([
            ("system", """Compare two AI responses and categorize the difference.
            
Categories:
- identical: Same or nearly identical responses
- minor: Small wording differences, same meaning
- moderate: Different approach but similar quality
- major: Significantly different responses or quality

Response 1 (Production):
{response1}

Response 2 (Shadow):
{response2}

Return only the category name."""),
            ("user", "Categorize the difference between these responses.")
        ])
    
    def execute_shadow_request(self, query: str) -> ShadowResponse:
        """
        Execute query on both production and shadow versions
        
        Args:
            query: User query to execute
            
        Returns:
            ShadowResponse with both results and comparison
        """
        # Execute in parallel using threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both requests
            prod_future = executor.submit(self._execute_production, query)
            shadow_future = executor.submit(self._execute_shadow, query)
            
            # Wait for both to complete
            prod_result, prod_time, prod_error = prod_future.result()
            shadow_result, shadow_time, shadow_error = shadow_future.result()
        
        # Compare responses
        if prod_error is None and shadow_error is None:
            diff_type, similarity = self._compare_responses(prod_result, shadow_result)
        else:
            diff_type = DifferenceType.ERROR
            similarity = 0.0
        
        # Create response object
        response = ShadowResponse(
            production_response=prod_result,
            shadow_response=shadow_result,
            production_time=prod_time,
            shadow_time=shadow_time,
            production_error=prod_error,
            shadow_error=shadow_error,
            difference_type=diff_type,
            similarity_score=similarity
        )
        
        # Update metrics
        self._update_metrics(response)
        
        # Store response
        self.responses.append(response)
        
        # Return production result to user (shadow is hidden)
        return response
    
    def _execute_production(self, query: str) -> Tuple[str, float, Optional[str]]:
        """Execute on production system"""
        start_time = time.time()
        
        try:
            chain = self.production_prompt | self.production_llm | StrOutputParser()
            result = chain.invoke({"query": query})
            execution_time = time.time() - start_time
            return result, execution_time, None
            
        except Exception as e:
            execution_time = time.time() - start_time
            return "", execution_time, str(e)
    
    def _execute_shadow(self, query: str) -> Tuple[str, float, Optional[str]]:
        """Execute on shadow system"""
        start_time = time.time()
        
        try:
            chain = self.shadow_prompt | self.shadow_llm | StrOutputParser()
            result = chain.invoke({"query": query})
            execution_time = time.time() - start_time
            return result, execution_time, None
            
        except Exception as e:
            execution_time = time.time() - start_time
            return "", execution_time, str(e)
    
    def _compare_responses(self, prod_response: str, 
                          shadow_response: str) -> Tuple[DifferenceType, float]:
        """Compare two responses and categorize difference"""
        # Quick check for identical
        if prod_response.strip() == shadow_response.strip():
            return DifferenceType.IDENTICAL, 1.0
        
        # Calculate basic similarity
        similarity = self._calculate_similarity(prod_response, shadow_response)
        
        # Use LLM for categorization
        try:
            chain = self.comparison_prompt | self.analyzer_llm | StrOutputParser()
            category_str = chain.invoke({
                "response1": prod_response[:500],  # Truncate for efficiency
                "response2": shadow_response[:500]
            }).strip().lower()
            
            # Map to enum
            category_map = {
                "identical": DifferenceType.IDENTICAL,
                "minor": DifferenceType.MINOR,
                "moderate": DifferenceType.MODERATE,
                "major": DifferenceType.MAJOR
            }
            
            diff_type = category_map.get(category_str, DifferenceType.MODERATE)
            
        except Exception as e:
            # Fallback to similarity-based categorization
            if similarity > 0.9:
                diff_type = DifferenceType.MINOR
            elif similarity > 0.7:
                diff_type = DifferenceType.MODERATE
            else:
                diff_type = DifferenceType.MAJOR
        
        return diff_type, similarity
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-overlap similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _update_metrics(self, response: ShadowResponse):
        """Update metrics based on response"""
        self.metrics.total_requests += 1
        
        # Count difference types
        if response.difference_type == DifferenceType.IDENTICAL:
            self.metrics.identical_responses += 1
        elif response.difference_type == DifferenceType.MINOR:
            self.metrics.minor_differences += 1
        elif response.difference_type == DifferenceType.MODERATE:
            self.metrics.moderate_differences += 1
        elif response.difference_type == DifferenceType.MAJOR:
            self.metrics.major_differences += 1
        
        # Count errors
        if response.production_error:
            self.metrics.production_errors += 1
        if response.shadow_error:
            self.metrics.shadow_errors += 1
        
        # Update latency metrics (running average)
        n = self.metrics.total_requests
        self.metrics.avg_production_latency = (
            (self.metrics.avg_production_latency * (n - 1) + response.production_time) / n
        )
        self.metrics.avg_shadow_latency = (
            (self.metrics.avg_shadow_latency * (n - 1) + response.shadow_time) / n
        )
        
        # Calculate relative performance
        if self.metrics.avg_production_latency > 0:
            self.metrics.shadow_slower_by = (
                (self.metrics.avg_shadow_latency - self.metrics.avg_production_latency) 
                / self.metrics.avg_production_latency * 100
            )
    
    def batch_test(self, queries: List[str], 
                   show_progress: bool = True) -> List[ShadowResponse]:
        """
        Test multiple queries in shadow mode
        
        Args:
            queries: List of queries to test
            show_progress: Whether to show progress
            
        Returns:
            List of shadow responses
        """
        print(f"\n🔄 Starting shadow mode testing for {len(queries)} queries...")
        
        results = []
        for i, query in enumerate(queries, 1):
            if show_progress:
                print(f"   Processing {i}/{len(queries)}: {query[:50]}...")
            
            response = self.execute_shadow_request(query)
            results.append(response)
            
            # Show inline status
            if show_progress:
                status = "✓" if response.difference_type in [DifferenceType.IDENTICAL, DifferenceType.MINOR] else "⚠"
                print(f"   {status} Difference: {response.difference_type.value}, Similarity: {response.similarity_score:.2f}")
        
        return results
    
    def get_health_status(self) -> HealthStatus:
        """Get current health status of shadow version"""
        return self.metrics.get_health_status()
    
    def get_divergent_examples(self, min_difference: DifferenceType = DifferenceType.MODERATE,
                               limit: int = 10) -> List[ShadowResponse]:
        """Get examples where responses diverged significantly"""
        divergent = [
            r for r in self.responses
            if r.difference_type in [DifferenceType.MODERATE, DifferenceType.MAJOR]
        ]
        
        # Sort by similarity (lowest first = most divergent)
        divergent.sort(key=lambda r: r.similarity_score)
        
        return divergent[:limit]
    
    def generate_report(self) -> str:
        """Generate comprehensive shadow testing report"""
        metrics = self.metrics
        health = self.get_health_status()
        
        report = [
            "\n" + "="*70,
            "SHADOW MODE TESTING REPORT",
            "="*70,
            f"\nTotal Requests: {metrics.total_requests}",
            f"Health Status: {health.value.upper()} {'🟢' if health == HealthStatus.HEALTHY else '🟡' if health == HealthStatus.WARNING else '🟠' if health == HealthStatus.DEGRADED else '🔴'}",
            "\nResponse Comparison:",
            f"  Identical: {metrics.identical_responses} ({metrics.get_difference_rate(DifferenceType.IDENTICAL)*100:.1f}%)",
            f"  Minor Differences: {metrics.minor_differences} ({metrics.get_difference_rate(DifferenceType.MINOR)*100:.1f}%)",
            f"  Moderate Differences: {metrics.moderate_differences} ({metrics.get_difference_rate(DifferenceType.MODERATE)*100:.1f}%)",
            f"  Major Differences: {metrics.major_differences} ({metrics.get_difference_rate(DifferenceType.MAJOR)*100:.1f}%)",
            "\nError Rates:",
            f"  Production Errors: {metrics.production_errors} ({metrics.production_errors/metrics.total_requests*100:.1f}%)",
            f"  Shadow Errors: {metrics.shadow_errors} ({metrics.shadow_errors/metrics.total_requests*100:.1f}%)",
            "\nPerformance:",
            f"  Avg Production Latency: {metrics.avg_production_latency:.3f}s",
            f"  Avg Shadow Latency: {metrics.avg_shadow_latency:.3f}s",
            f"  Shadow Performance: {metrics.shadow_slower_by:+.1f}% {'(slower)' if metrics.shadow_slower_by > 0 else '(faster)'}",
        ]
        
        # Add recommendation
        report.append("\nRecommendation:")
        if health == HealthStatus.HEALTHY:
            report.append("  ✅ Shadow version looks good for promotion to production")
        elif health == HealthStatus.WARNING:
            report.append("  ⚠️  Monitor shadow version closely before promotion")
        elif health == HealthStatus.DEGRADED:
            report.append("  ⚠️  Shadow version shows degradation - investigate before promotion")
        else:
            report.append("  ❌ Shadow version failing - DO NOT promote to production")
        
        report.append("="*70)
        return "\n".join(report)
    
    def show_divergent_examples(self, limit: int = 5):
        """Show examples where responses diverged"""
        divergent = self.get_divergent_examples(limit=limit)
        
        if not divergent:
            print("\n✅ No significant divergent examples found")
            return
        
        print(f"\n{'='*70}")
        print(f"DIVERGENT EXAMPLES (Top {len(divergent)})")
        print(f"{'='*70}")
        
        for i, response in enumerate(divergent, 1):
            print(f"\nExample {i}:")
            print(f"Difference Type: {response.difference_type.value}")
            print(f"Similarity: {response.similarity_score:.2f}")
            print(f"\nProduction Response:")
            print(f"  {response.production_response[:200]}...")
            print(f"\nShadow Response:")
            print(f"  {response.shadow_response[:200]}...")
            print("-" * 70)


def demonstrate_shadow_mode():
    """Demonstrate shadow mode testing"""
    print("="*70)
    print("Pattern 153: Shadow Mode Testing")
    print("="*70)
    
    # Create tester with different configurations
    # In real scenario, these would be different model versions or prompts
    tester = ShadowModeTester(
        production_model="gpt-3.5-turbo",
        shadow_model="gpt-3.5-turbo",
        production_temp=0.7,
        shadow_temp=0.9  # Different temperature for shadow
    )
    
    # Test queries
    test_queries = [
        "What is Python?",
        "Explain machine learning in simple terms",
        "How do I learn programming?",
        "What's the difference between AI and ML?",
        "Write a haiku about coding",
        "What are best practices for API design?",
        "Explain recursion with an example",
        "How does garbage collection work?",
        "What is the difference between REST and GraphQL?",
        "Explain async/await in JavaScript"
    ]
    
    print("\n" + "="*70)
    print("BATCH SHADOW TESTING")
    print("="*70)
    print(f"Production: GPT-3.5-Turbo (temp=0.7)")
    print(f"Shadow: GPT-3.5-Turbo (temp=0.9)")
    
    # Run batch test
    results = tester.batch_test(test_queries, show_progress=True)
    
    # Generate report
    print(tester.generate_report())
    
    # Show divergent examples
    tester.show_divergent_examples(limit=3)
    
    # Test individual request
    print("\n" + "="*70)
    print("INDIVIDUAL REQUEST EXAMPLE")
    print("="*70)
    
    query = "Explain quantum computing"
    print(f"\nQuery: {query}")
    
    response = tester.execute_shadow_request(query)
    
    print(f"\n📊 Results:")
    print(f"  Difference Type: {response.difference_type.value}")
    print(f"  Similarity Score: {response.similarity_score:.2f}")
    print(f"  Production Time: {response.production_time:.3f}s")
    print(f"  Shadow Time: {response.shadow_time:.3f}s")
    
    print(f"\n🏭 Production Response:")
    print(f"  {response.production_response[:150]}...")
    
    print(f"\n🔬 Shadow Response:")
    print(f"  {response.shadow_response[:150]}...")
    
    # Health check
    health = tester.get_health_status()
    print(f"\n🏥 Shadow Health Status: {health.value.upper()}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Shadow Mode Testing Pattern provides:

1. Zero-Impact Testing:
   - Tests run parallel to production
   - Users only see production responses
   - Shadow results collected for analysis
   - No service disruption

2. Comprehensive Comparison:
   - Response similarity analysis
   - Difference categorization
   - Performance comparison
   - Error rate tracking

3. Health Monitoring:
   - Automatic health status assessment
   - Divergence detection
   - Performance regression identification
   - Error rate monitoring

4. Deployment Confidence:
   - Real production traffic testing
   - Early problem detection
   - Data-driven promotion decisions
   - Risk mitigation

5. Use Cases:
   - New model version validation
   - Prompt engineering testing
   - Configuration changes
   - Performance optimization
   - Feature rollout validation

This pattern is essential for safely testing changes to production
AI systems before full deployment.
    """)


if __name__ == "__main__":
    demonstrate_shadow_mode()
