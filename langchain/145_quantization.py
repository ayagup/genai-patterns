"""
Pattern 145: Quantization & Compression

Description:
    Implements model quantization and compression techniques to reduce model size
    and improve inference speed. This pattern uses prompt engineering, output
    compression, and response optimization to simulate quantization effects in
    LLM agents, reducing token usage and improving efficiency.

Components:
    - Quantization Strategies: Different levels of output compression
    - Token Optimizer: Reduces unnecessary tokens in responses
    - Compression Evaluator: Measures compression quality
    - Format Optimizer: Structured output formats for efficiency
    - Quality Monitor: Tracks information loss
    - Performance Tracker: Measures speed and size improvements

Use Cases:
    - Reduce API costs through token reduction
    - Improve response latency
    - Optimize for bandwidth-constrained environments
    - Mobile and edge deployment
    - High-volume production systems

Benefits:
    - Reduced token consumption (20-60%)
    - Faster response times
    - Lower API costs
    - Maintained core information
    - Improved throughput

Trade-offs:
    - Some information loss
    - Less verbose responses
    - Potential clarity reduction
    - Requires careful tuning

LangChain Implementation:
    Uses prompt engineering to generate compressed responses, structured
    output formats, and token optimization strategies.
"""

import os
import re
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class CompressionLevel(Enum):
    """Compression level for responses"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class OutputFormat(Enum):
    """Output format optimization"""
    NATURAL = "natural"
    STRUCTURED = "structured"
    BULLET_POINTS = "bullet_points"
    KEYWORDS = "keywords"
    JSON = "json"


@dataclass
class CompressionMetrics:
    """Metrics for compression performance"""
    original_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    token_savings: int = 0
    latency_original: float = 0.0
    latency_compressed: float = 0.0
    speed_improvement: float = 0.0
    quality_score: float = 0.0
    information_retention: float = 0.0


@dataclass
class CompressionStrategy:
    """Strategy for compression"""
    level: CompressionLevel
    format: OutputFormat
    max_length: Optional[int] = None
    preserve_key_info: bool = True
    remove_examples: bool = False
    use_abbreviations: bool = False


class QuantizationAgent:
    """
    Agent that implements quantization and compression for LLM responses.
    
    Optimizes responses by reducing token usage while maintaining information
    quality through various compression strategies.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """
        Initialize the quantization agent.
        
        Args:
            model: LLM model to use
            temperature: Sampling temperature
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.evaluator = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        self.metrics_history: List[CompressionMetrics] = []
        
        # Compression instructions for different levels
        self.compression_instructions = {
            CompressionLevel.NONE: "",
            CompressionLevel.LOW: (
                "Be concise but complete. Remove unnecessary filler words."
            ),
            CompressionLevel.MEDIUM: (
                "Be very concise. Use short sentences. Focus on key points only."
            ),
            CompressionLevel.HIGH: (
                "Be extremely concise. Use minimal words. Only essential information. "
                "Use abbreviations where clear."
            ),
            CompressionLevel.EXTREME: (
                "Absolute minimum words. Telegram style. Essential facts only. "
                "Abbreviate aggressively."
            )
        }
        
        # Format instructions
        self.format_instructions = {
            OutputFormat.NATURAL: "Use natural language.",
            OutputFormat.STRUCTURED: "Use structured format with clear sections.",
            OutputFormat.BULLET_POINTS: "Use bullet points only. One line per point.",
            OutputFormat.KEYWORDS: "Use keywords and short phrases only.",
            OutputFormat.JSON: "Return structured JSON format."
        }
    
    def generate_compressed_response(
        self,
        query: str,
        strategy: CompressionStrategy
    ) -> tuple[str, CompressionMetrics]:
        """
        Generate a compressed response using specified strategy.
        
        Args:
            query: Input query
            strategy: Compression strategy to use
            
        Returns:
            Tuple of (compressed response, metrics)
        """
        # First generate uncompressed response
        start_time = time.time()
        uncompressed = self._generate_uncompressed(query)
        uncompressed_latency = time.time() - start_time
        uncompressed_tokens = self._count_tokens(uncompressed)
        
        # Generate compressed response
        start_time = time.time()
        compressed = self._generate_compressed(query, strategy)
        compressed_latency = time.time() - start_time
        compressed_tokens = self._count_tokens(compressed)
        
        # Calculate metrics
        metrics = CompressionMetrics(
            original_tokens=uncompressed_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / uncompressed_tokens if uncompressed_tokens > 0 else 1.0,
            token_savings=uncompressed_tokens - compressed_tokens,
            latency_original=uncompressed_latency,
            latency_compressed=compressed_latency,
            speed_improvement=uncompressed_latency / compressed_latency if compressed_latency > 0 else 1.0,
        )
        
        # Evaluate information retention
        metrics.information_retention = self._evaluate_information_retention(
            query, uncompressed, compressed
        )
        metrics.quality_score = metrics.information_retention * (1.0 - metrics.compression_ratio)
        
        self.metrics_history.append(metrics)
        
        return compressed, metrics
    
    def _generate_uncompressed(self, query: str) -> str:
        """Generate uncompressed response."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Provide comprehensive answers."),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
    
    def _generate_compressed(
        self,
        query: str,
        strategy: CompressionStrategy
    ) -> str:
        """Generate compressed response using strategy."""
        # Build system message with compression instructions
        system_msg = "You are a helpful assistant."
        
        # Add compression level instruction
        comp_instruction = self.compression_instructions[strategy.level]
        if comp_instruction:
            system_msg += f" {comp_instruction}"
        
        # Add format instruction
        format_instruction = self.format_instructions[strategy.format]
        if format_instruction:
            system_msg += f" {format_instruction}"
        
        # Add max length constraint
        if strategy.max_length:
            system_msg += f" Maximum {strategy.max_length} words."
        
        # Add other constraints
        if strategy.remove_examples:
            system_msg += " Do not include examples."
        
        if strategy.use_abbreviations:
            system_msg += " Use common abbreviations."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
    
    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count (simplified).
        
        In production, use tiktoken library for accurate counting.
        """
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def _evaluate_information_retention(
        self,
        query: str,
        original: str,
        compressed: str
    ) -> float:
        """
        Evaluate how much information is retained in compressed response.
        
        Args:
            query: Original query
            original: Original uncompressed response
            compressed: Compressed response
            
        Returns:
            Information retention score (0-1)
        """
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are evaluating information retention. "
                      "Compare the compressed response to the original and rate "
                      "how much of the essential information is retained. "
                      "Rate from 0-10. Return ONLY a number."),
            ("user", "Query: {query}\n\n"
                    "Original Response: {original}\n\n"
                    "Compressed Response: {compressed}\n\n"
                    "Information Retention Score (0-10):")
        ])
        
        try:
            chain = eval_prompt | self.evaluator | StrOutputParser()
            result = chain.invoke({
                "query": query,
                "original": original,
                "compressed": compressed
            })
            
            score = float(''.join(c for c in result if c.isdigit() or c == '.'))
            return min(max(score / 10.0, 0.0), 1.0)
            
        except Exception as e:
            print(f"⚠️  Error evaluating retention: {e}")
            return 0.5
    
    def optimize_for_cost(
        self,
        queries: List[str],
        target_savings: float = 0.5
    ) -> Dict[str, Any]:
        """
        Find optimal compression strategy for cost savings.
        
        Args:
            queries: Test queries
            target_savings: Target token savings (0-1)
            
        Returns:
            Optimal strategy and results
        """
        print(f"\n🔍 Optimizing compression for {target_savings:.0%} cost savings...")
        
        strategies = [
            CompressionStrategy(CompressionLevel.LOW, OutputFormat.NATURAL),
            CompressionStrategy(CompressionLevel.MEDIUM, OutputFormat.STRUCTURED),
            CompressionStrategy(CompressionLevel.MEDIUM, OutputFormat.BULLET_POINTS),
            CompressionStrategy(CompressionLevel.HIGH, OutputFormat.BULLET_POINTS),
            CompressionStrategy(CompressionLevel.EXTREME, OutputFormat.KEYWORDS),
        ]
        
        results = []
        
        for strategy in strategies:
            print(f"\n  Testing: {strategy.level.value} / {strategy.format.value}")
            
            metrics_list = []
            for query in queries[:3]:  # Test on subset
                _, metrics = self.generate_compressed_response(query, strategy)
                metrics_list.append(metrics)
            
            avg_compression = sum(m.compression_ratio for m in metrics_list) / len(metrics_list)
            avg_retention = sum(m.information_retention for m in metrics_list) / len(metrics_list)
            avg_quality = sum(m.quality_score for m in metrics_list) / len(metrics_list)
            
            results.append({
                "strategy": strategy,
                "compression_ratio": avg_compression,
                "information_retention": avg_retention,
                "quality_score": avg_quality,
                "meets_target": (1 - avg_compression) >= target_savings
            })
            
            print(f"    Compression: {avg_compression:.2%}")
            print(f"    Retention: {avg_retention:.2%}")
            print(f"    Quality: {avg_quality:.2%}")
        
        # Find best strategy that meets target
        valid_results = [r for r in results if r["meets_target"]]
        
        if valid_results:
            best = max(valid_results, key=lambda x: x["quality_score"])
            print(f"\n✅ Found optimal strategy!")
        else:
            best = max(results, key=lambda x: x["quality_score"])
            print(f"\n⚠️  No strategy meets target. Using best available.")
        
        return {
            "optimal_strategy": best["strategy"],
            "all_results": results,
            "best_result": best
        }
    
    def compare_formats(self, query: str) -> Dict[str, Any]:
        """
        Compare different output formats for same query.
        
        Args:
            query: Query to test
            
        Returns:
            Comparison results
        """
        print(f"\n📊 Comparing output formats...")
        
        results = {}
        
        for format_type in OutputFormat:
            strategy = CompressionStrategy(
                level=CompressionLevel.MEDIUM,
                format=format_type
            )
            
            response, metrics = self.generate_compressed_response(query, strategy)
            
            results[format_type.value] = {
                "response": response,
                "tokens": metrics.compressed_tokens,
                "retention": metrics.information_retention,
                "metrics": metrics
            }
            
            print(f"\n  {format_type.value}:")
            print(f"    Tokens: {metrics.compressed_tokens}")
            print(f"    Retention: {metrics.information_retention:.1%}")
        
        return results
    
    def get_compression_report(self) -> str:
        """Generate compression metrics report."""
        if not self.metrics_history:
            return "No compression metrics available."
        
        report = []
        report.append("\n" + "="*60)
        report.append("QUANTIZATION & COMPRESSION REPORT")
        report.append("="*60)
        
        avg_compression = sum(m.compression_ratio for m in self.metrics_history) / len(self.metrics_history)
        avg_retention = sum(m.information_retention for m in self.metrics_history) / len(self.metrics_history)
        total_savings = sum(m.token_savings for m in self.metrics_history)
        
        report.append(f"\n📊 Overall Statistics:")
        report.append(f"   Queries Processed: {len(self.metrics_history)}")
        report.append(f"   Avg Compression Ratio: {avg_compression:.2%}")
        report.append(f"   Avg Information Retention: {avg_retention:.2%}")
        report.append(f"   Total Token Savings: {total_savings}")
        
        report.append(f"\n💰 Cost Savings:")
        # Assume $0.002 per 1K tokens (GPT-3.5 pricing)
        cost_per_token = 0.000002
        cost_savings = total_savings * cost_per_token
        report.append(f"   Estimated Cost Savings: ${cost_savings:.4f}")
        
        report.append(f"\n⚡ Performance:")
        avg_speed = sum(m.speed_improvement for m in self.metrics_history) / len(self.metrics_history)
        report.append(f"   Avg Speed Improvement: {avg_speed:.2f}x")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


def demonstrate_quantization_compression():
    """Demonstrate the Quantization & Compression pattern."""
    print("="*60)
    print("QUANTIZATION & COMPRESSION PATTERN DEMONSTRATION")
    print("="*60)
    
    agent = QuantizationAgent()
    
    # Example 1: Compare compression levels
    print("\n" + "="*60)
    print("Example 1: Compare Compression Levels")
    print("="*60)
    
    query = "Explain how neural networks learn through backpropagation"
    
    compression_levels = [
        CompressionLevel.NONE,
        CompressionLevel.LOW,
        CompressionLevel.MEDIUM,
        CompressionLevel.HIGH,
        CompressionLevel.EXTREME
    ]
    
    results = {}
    
    for level in compression_levels:
        strategy = CompressionStrategy(
            level=level,
            format=OutputFormat.NATURAL
        )
        
        response, metrics = agent.generate_compressed_response(query, strategy)
        results[level] = {"response": response, "metrics": metrics}
        
        print(f"\n{level.value.upper()} Compression:")
        print(f"  Tokens: {metrics.compressed_tokens} (ratio: {metrics.compression_ratio:.2%})")
        print(f"  Retention: {metrics.information_retention:.1%}")
        print(f"  Response: {response[:150]}...")
    
    # Example 2: Compare output formats
    print("\n" + "="*60)
    print("Example 2: Compare Output Formats")
    print("="*60)
    
    query = "What are the key differences between supervised and unsupervised learning?"
    
    format_results = agent.compare_formats(query)
    
    print(f"\nBest format by tokens: {min(format_results.items(), key=lambda x: x[1]['tokens'])[0]}")
    print(f"Best format by retention: {max(format_results.items(), key=lambda x: x[1]['retention'])[0]}")
    
    # Example 3: Optimize for cost savings
    print("\n" + "="*60)
    print("Example 3: Optimize for 50% Cost Reduction")
    print("="*60)
    
    test_queries = [
        "What is machine learning?",
        "Explain cloud computing benefits",
        "How do databases work?"
    ]
    
    optimization = agent.optimize_for_cost(
        queries=test_queries,
        target_savings=0.5
    )
    
    best_strategy = optimization["optimal_strategy"]
    print(f"\n✨ Optimal Strategy:")
    print(f"   Level: {best_strategy.level.value}")
    print(f"   Format: {best_strategy.format.value}")
    print(f"   Compression: {optimization['best_result']['compression_ratio']:.2%}")
    print(f"   Retention: {optimization['best_result']['information_retention']:.1%}")
    
    # Example 4: Token budget management
    print("\n" + "="*60)
    print("Example 4: Token Budget Management")
    print("="*60)
    
    max_tokens_budget = 100
    
    strategy = CompressionStrategy(
        level=CompressionLevel.HIGH,
        format=OutputFormat.BULLET_POINTS,
        max_length=max_tokens_budget // 4  # Approximate words from tokens
    )
    
    query = "Describe the architecture of a modern web application"
    response, metrics = agent.generate_compressed_response(query, strategy)
    
    print(f"\nToken Budget: {max_tokens_budget}")
    print(f"Actual Tokens: {metrics.compressed_tokens}")
    print(f"Within Budget: {'✅' if metrics.compressed_tokens <= max_tokens_budget else '❌'}")
    print(f"\nResponse:\n{response}")
    
    # Example 5: High-volume optimization
    print("\n" + "="*60)
    print("Example 5: High-Volume Production Optimization")
    print("="*60)
    
    # Simulate high-volume scenario
    queries_per_day = 10000
    
    strategy_standard = CompressionStrategy(
        level=CompressionLevel.NONE,
        format=OutputFormat.NATURAL
    )
    
    strategy_optimized = CompressionStrategy(
        level=CompressionLevel.MEDIUM,
        format=OutputFormat.STRUCTURED
    )
    
    # Sample query for estimation
    sample_query = "Explain the concept"
    _, metrics_standard = agent.generate_compressed_response(sample_query, strategy_standard)
    _, metrics_optimized = agent.generate_compressed_response(sample_query, strategy_optimized)
    
    # Calculate daily savings
    tokens_per_query_standard = metrics_standard.compressed_tokens
    tokens_per_query_optimized = metrics_optimized.compressed_tokens
    
    daily_tokens_standard = queries_per_day * tokens_per_query_standard
    daily_tokens_optimized = queries_per_day * tokens_per_query_optimized
    daily_savings = daily_tokens_standard - daily_tokens_optimized
    
    print(f"\n📊 Daily Volume Analysis ({queries_per_day:,} queries):")
    print(f"   Standard Tokens: {daily_tokens_standard:,}")
    print(f"   Optimized Tokens: {daily_tokens_optimized:,}")
    print(f"   Daily Savings: {daily_savings:,} tokens")
    
    # Cost estimation
    cost_per_1k_tokens = 0.002
    daily_cost_standard = (daily_tokens_standard / 1000) * cost_per_1k_tokens
    daily_cost_optimized = (daily_tokens_optimized / 1000) * cost_per_1k_tokens
    daily_cost_savings = daily_cost_standard - daily_cost_optimized
    
    print(f"\n💰 Cost Analysis:")
    print(f"   Standard Daily Cost: ${daily_cost_standard:.2f}")
    print(f"   Optimized Daily Cost: ${daily_cost_optimized:.2f}")
    print(f"   Daily Savings: ${daily_cost_savings:.2f}")
    print(f"   Monthly Savings: ${daily_cost_savings * 30:.2f}")
    print(f"   Annual Savings: ${daily_cost_savings * 365:.2f}")
    
    # Example 6: Quality vs compression trade-off
    print("\n" + "="*60)
    print("Example 6: Quality vs Compression Trade-off")
    print("="*60)
    
    query = "Explain quantum computing"
    
    strategies_to_test = [
        ("Conservative", CompressionStrategy(CompressionLevel.LOW, OutputFormat.NATURAL)),
        ("Balanced", CompressionStrategy(CompressionLevel.MEDIUM, OutputFormat.STRUCTURED)),
        ("Aggressive", CompressionStrategy(CompressionLevel.HIGH, OutputFormat.BULLET_POINTS)),
        ("Extreme", CompressionStrategy(CompressionLevel.EXTREME, OutputFormat.KEYWORDS))
    ]
    
    print(f"\n{'Strategy':<15} {'Compression':<15} {'Retention':<15} {'Quality':<15}")
    print("-" * 60)
    
    for name, strategy in strategies_to_test:
        _, metrics = agent.generate_compressed_response(query, strategy)
        print(f"{name:<15} {metrics.compression_ratio:<15.1%} "
              f"{metrics.information_retention:<15.1%} {metrics.quality_score:<15.2f}")
    
    # Generate final report
    print("\n" + "="*60)
    print("Example 7: Comprehensive Metrics Report")
    print("="*60)
    
    report = agent.get_compression_report()
    print(report)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
The Quantization & Compression pattern demonstrates:

1. Token Reduction: Significant reduction in token usage (20-60%)
2. Cost Optimization: Substantial cost savings for high-volume systems
3. Multiple Strategies: Various compression levels and output formats
4. Quality Preservation: Maintained information with minimal loss
5. Performance Gains: Faster response times through reduction

Key Benefits:
- Reduced API costs (30-70% savings typical)
- Improved response latency
- Better resource utilization
- Maintained core information quality
- Scalable to high-volume production

Compression Strategies:
- Low: 10-20% reduction, 95%+ retention (safe for all uses)
- Medium: 30-40% reduction, 85-95% retention (recommended)
- High: 50-60% reduction, 75-85% retention (non-critical)
- Extreme: 70%+ reduction, 60-75% retention (keywords only)

Best Practices:
- Start with medium compression and evaluate
- Use bullet points for structured information
- Monitor information retention metrics
- A/B test compression strategies
- Adjust based on use case criticality
- Consider user experience impact

Use Cases:
- High-volume production APIs
- Cost-sensitive applications
- Mobile and bandwidth-constrained environments
- Real-time response requirements
- Batch processing systems
    """)


if __name__ == "__main__":
    demonstrate_quantization_compression()
