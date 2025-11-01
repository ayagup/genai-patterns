"""
Pattern 170: Model Routing & Selection

Description:
    The Model Routing & Selection pattern intelligently routes queries to the most
    appropriate model based on query characteristics, model capabilities, performance
    requirements, and cost constraints. It analyzes queries, classifies them, and
    selects optimal models dynamically.

Components:
    1. Query Analyzer: Analyzes query characteristics
    2. Classifier: Categorizes query type and complexity
    3. Model Registry: Maintains available models and capabilities
    4. Router: Selects optimal model
    5. Performance Monitor: Tracks model performance
    6. Cost Optimizer: Balances quality and cost

Use Cases:
    - Cost-optimized AI systems
    - Performance-critical applications
    - Multi-model architectures
    - Adaptive system design
    - Resource-constrained environments
    - Quality-cost trade-offs

Benefits:
    - Optimal resource utilization
    - Cost efficiency
    - Performance optimization
    - Scalability
    - Flexible model upgrades
    - Quality-aware routing

Trade-offs:
    - Routing overhead
    - Classification accuracy dependency
    - Complexity
    - Multiple model management
    - Potential routing errors

LangChain Implementation:
    Implements intelligent query classification and model selection using LangChain's
    model abstraction. Routes queries based on complexity, domain, and requirements.

🎉 THIS IS PATTERN 170 - THE FINAL PATTERN! 🎉
Completing all 170 Agentic AI Design Patterns!
"""

import os
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class QueryDomain(Enum):
    """Query domain categories"""
    GENERAL = "general"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    CODE = "code"


class ModelSize(Enum):
    """Model size categories"""
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    XLARGE = "xlarge"


@dataclass
class QueryProfile:
    """Profile of a query"""
    text: str
    complexity: QueryComplexity
    domain: QueryDomain
    estimated_tokens: int
    requires_reasoning: bool = False
    requires_creativity: bool = False
    latency_sensitive: bool = False


@dataclass
class ModelSpec:
    """Specification of a model"""
    name: str
    model_id: str
    size: ModelSize
    capabilities: List[str]
    cost_per_1k_tokens: float
    avg_latency_ms: float
    quality_score: float  # 0-1
    max_tokens: int
    best_for: List[QueryDomain]
    available: bool = True


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    selected_model: str
    reasoning: str
    confidence: float
    alternatives: List[str]
    estimated_cost: float
    estimated_latency: float


class ModelRouter:
    """Intelligent model routing and selection"""
    
    def __init__(self):
        """Initialize model router"""
        self.models: Dict[str, ModelSpec] = {}
        self.llm_instances: Dict[str, ChatOpenAI] = {}
        
        # Performance tracking
        self.routing_history: List[Dict[str, Any]] = []
        self.model_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "queries_routed": 0,
            "total_cost": 0.0,
            "total_latency": 0.0,
            "success_count": 0
        })
        
        # Register default models
        self._register_default_models()
        
        # Classification prompt
        self.classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier. Analyze the query and determine:
            1. Complexity: simple, moderate, complex, or expert
            2. Domain: general, technical, creative, analytical, conversational, or code
            3. Special requirements: reasoning, creativity, latency-sensitive"""),
            ("user", """Query: {query}

Classify this query. Respond in format:
COMPLEXITY: [level]
DOMAIN: [domain]
REASONING_REQUIRED: [yes/no]
CREATIVITY_REQUIRED: [yes/no]
LATENCY_SENSITIVE: [yes/no]""")
        ])
    
    def _register_default_models(self):
        """Register default model specifications"""
        # Tiny model - ultra fast, cheap
        self.register_model(ModelSpec(
            name="GPT-3.5 Turbo Mini",
            model_id="gpt-3.5-turbo",
            size=ModelSize.SMALL,
            capabilities=["conversation", "simple_qa", "summarization"],
            cost_per_1k_tokens=0.0015,
            avg_latency_ms=500,
            quality_score=0.7,
            max_tokens=4096,
            best_for=[QueryDomain.GENERAL, QueryDomain.CONVERSATIONAL]
        ))
        
        # Medium model - balanced
        self.register_model(ModelSpec(
            name="GPT-3.5 Turbo",
            model_id="gpt-3.5-turbo",
            size=ModelSize.MEDIUM,
            capabilities=["reasoning", "analysis", "code", "creativity"],
            cost_per_1k_tokens=0.002,
            avg_latency_ms=800,
            quality_score=0.8,
            max_tokens=4096,
            best_for=[QueryDomain.TECHNICAL, QueryDomain.CODE, QueryDomain.ANALYTICAL]
        ))
        
        # Large model - high quality
        self.register_model(ModelSpec(
            name="GPT-4",
            model_id="gpt-4",
            size=ModelSize.LARGE,
            capabilities=["advanced_reasoning", "complex_analysis", "expert_code", "creative_writing"],
            cost_per_1k_tokens=0.03,
            avg_latency_ms=2000,
            quality_score=0.95,
            max_tokens=8192,
            best_for=[QueryDomain.TECHNICAL, QueryDomain.ANALYTICAL, QueryDomain.CREATIVE]
        ))
        
        # Specialized code model (simulated)
        self.register_model(ModelSpec(
            name="Code Specialist",
            model_id="gpt-3.5-turbo",
            size=ModelSize.MEDIUM,
            capabilities=["code_generation", "code_review", "debugging"],
            cost_per_1k_tokens=0.0025,
            avg_latency_ms=900,
            quality_score=0.85,
            max_tokens=4096,
            best_for=[QueryDomain.CODE]
        ))
    
    def register_model(self, spec: ModelSpec):
        """Register a model"""
        self.models[spec.name] = spec
        try:
            self.llm_instances[spec.name] = ChatOpenAI(
                model=spec.model_id,
                temperature=0.7
            )
        except Exception as e:
            print(f"Warning: Could not create LLM for {spec.name}: {e}")
            spec.available = False
    
    def route_query(self, query: str, 
                   max_cost: Optional[float] = None,
                   max_latency: Optional[float] = None,
                   min_quality: Optional[float] = None) -> RoutingDecision:
        """
        Route query to optimal model
        
        Args:
            query: The query to route
            max_cost: Maximum acceptable cost
            max_latency: Maximum acceptable latency (ms)
            min_quality: Minimum acceptable quality score
            
        Returns:
            RoutingDecision with selected model and reasoning
        """
        # Classify query
        profile = self._classify_query(query)
        
        # Score and rank models
        scored_models = self._score_models(profile, max_cost, max_latency, min_quality)
        
        if not scored_models:
            raise ValueError("No suitable model found for query")
        
        # Select best model
        best_score, best_model = scored_models[0]
        alternatives = [name for _, name in scored_models[1:3]]
        
        # Estimate costs
        estimated_tokens = profile.estimated_tokens
        estimated_cost = (estimated_tokens / 1000) * best_model.cost_per_1k_tokens
        estimated_latency = best_model.avg_latency_ms
        
        # Generate reasoning
        reasoning = self._generate_routing_reasoning(profile, best_model, best_score)
        
        # Record decision
        self._record_routing(query, best_model.name, profile, estimated_cost)
        
        return RoutingDecision(
            selected_model=best_model.name,
            reasoning=reasoning,
            confidence=best_score / 100,  # Normalize to 0-1
            alternatives=alternatives,
            estimated_cost=estimated_cost,
            estimated_latency=estimated_latency
        )
    
    def execute_with_routing(self, query: str, **routing_kwargs) -> Dict[str, Any]:
        """Route and execute query"""
        # Route query
        decision = self.route_query(query, **routing_kwargs)
        
        # Execute on selected model
        start_time = time.time()
        
        try:
            llm = self.llm_instances[decision.selected_model]
            prompt = ChatPromptTemplate.from_messages([
                ("user", "{query}")
            ])
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"query": query})
            
            execution_time = time.time() - start_time
            
            # Update performance stats
            self._update_performance(decision.selected_model, 
                                   decision.estimated_cost,
                                   execution_time * 1000,
                                   success=True)
            
            return {
                "result": result,
                "routing_decision": decision,
                "actual_latency_ms": execution_time * 1000,
                "success": True
            }
            
        except Exception as e:
            self._update_performance(decision.selected_model, 0, 0, success=False)
            return {
                "result": None,
                "routing_decision": decision,
                "error": str(e),
                "success": False
            }
    
    def _classify_query(self, query: str) -> QueryProfile:
        """Classify query characteristics"""
        # Use LLM to classify
        classifier = self.classifier_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0) | StrOutputParser()
        
        try:
            classification = classifier.invoke({"query": query})
            
            # Parse classification
            complexity = QueryComplexity.MODERATE
            domain = QueryDomain.GENERAL
            requires_reasoning = False
            requires_creativity = False
            latency_sensitive = False
            
            for line in classification.split('\n'):
                line = line.strip().upper()
                if line.startswith('COMPLEXITY:'):
                    comp_str = line.split(':', 1)[1].strip().lower()
                    if 'simple' in comp_str:
                        complexity = QueryComplexity.SIMPLE
                    elif 'expert' in comp_str:
                        complexity = QueryComplexity.EXPERT
                    elif 'complex' in comp_str:
                        complexity = QueryComplexity.COMPLEX
                elif line.startswith('DOMAIN:'):
                    domain_str = line.split(':', 1)[1].strip().lower()
                    if 'code' in domain_str:
                        domain = QueryDomain.CODE
                    elif 'technical' in domain_str:
                        domain = QueryDomain.TECHNICAL
                    elif 'creative' in domain_str:
                        domain = QueryDomain.CREATIVE
                    elif 'analytical' in domain_str:
                        domain = QueryDomain.ANALYTICAL
                    elif 'conversational' in domain_str:
                        domain = QueryDomain.CONVERSATIONAL
                elif 'REASONING' in line:
                    requires_reasoning = 'YES' in line
                elif 'CREATIVITY' in line:
                    requires_creativity = 'YES' in line
                elif 'LATENCY' in line:
                    latency_sensitive = 'YES' in line
            
            estimated_tokens = len(query.split()) * 1.5
            
            return QueryProfile(
                text=query,
                complexity=complexity,
                domain=domain,
                estimated_tokens=int(estimated_tokens),
                requires_reasoning=requires_reasoning,
                requires_creativity=requires_creativity,
                latency_sensitive=latency_sensitive
            )
            
        except Exception as e:
            # Fallback to simple heuristics
            return QueryProfile(
                text=query,
                complexity=QueryComplexity.MODERATE,
                domain=QueryDomain.GENERAL,
                estimated_tokens=len(query.split()) * 1.5,
                requires_reasoning=len(query.split()) > 20,
                requires_creativity=any(word in query.lower() for word in ['creative', 'imagine', 'story']),
                latency_sensitive=any(word in query.lower() for word in ['quick', 'fast', 'urgent'])
            )
    
    def _score_models(self, profile: QueryProfile,
                     max_cost: Optional[float],
                     max_latency: Optional[float],
                     min_quality: Optional[float]) -> List[Tuple[float, ModelSpec]]:
        """Score and rank models for query"""
        scores = []
        
        for model in self.models.values():
            if not model.available:
                continue
            
            # Check hard constraints
            if max_cost:
                est_cost = (profile.estimated_tokens / 1000) * model.cost_per_1k_tokens
                if est_cost > max_cost:
                    continue
            
            if max_latency and model.avg_latency_ms > max_latency:
                continue
            
            if min_quality and model.quality_score < min_quality:
                continue
            
            # Calculate score
            score = 0.0
            
            # Domain match (30%)
            if profile.domain in model.best_for:
                score += 30
            
            # Quality score (25%)
            score += model.quality_score * 25
            
            # Complexity match (20%)
            if profile.complexity == QueryComplexity.SIMPLE and model.size in [ModelSize.TINY, ModelSize.SMALL]:
                score += 20
            elif profile.complexity == QueryComplexity.EXPERT and model.size in [ModelSize.LARGE, ModelSize.XLARGE]:
                score += 20
            elif profile.complexity == QueryComplexity.MODERATE:
                score += 15
            
            # Cost efficiency (15%)
            cost_score = 1.0 - min(model.cost_per_1k_tokens / 0.05, 1.0)
            score += cost_score * 15
            
            # Latency (10%)
            if profile.latency_sensitive:
                latency_score = 1.0 - min(model.avg_latency_ms / 3000, 1.0)
                score += latency_score * 10
            
            scores.append((score, model))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores
    
    def _generate_routing_reasoning(self, profile: QueryProfile, 
                                    model: ModelSpec, score: float) -> str:
        """Generate human-readable routing reasoning"""
        reasons = []
        
        if profile.domain in model.best_for:
            reasons.append(f"specialized for {profile.domain.value} domain")
        
        if profile.complexity == QueryComplexity.SIMPLE:
            reasons.append("query is simple, using efficient model")
        elif profile.complexity == QueryComplexity.EXPERT:
            reasons.append("complex query requires high-capability model")
        
        if model.quality_score >= 0.9:
            reasons.append("high quality model selected")
        
        if model.cost_per_1k_tokens < 0.005:
            reasons.append("cost-effective choice")
        
        return f"Selected {model.name}: " + ", ".join(reasons)
    
    def _record_routing(self, query: str, model_name: str, 
                       profile: QueryProfile, estimated_cost: float):
        """Record routing decision"""
        self.routing_history.append({
            "query": query[:100],
            "model": model_name,
            "complexity": profile.complexity.value,
            "domain": profile.domain.value,
            "estimated_cost": estimated_cost,
            "timestamp": time.time()
        })
        
        self.model_performance[model_name]["queries_routed"] += 1
    
    def _update_performance(self, model_name: str, cost: float, 
                          latency: float, success: bool):
        """Update model performance statistics"""
        stats = self.model_performance[model_name]
        stats["total_cost"] += cost
        stats["total_latency"] += latency
        if success:
            stats["success_count"] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get routing performance report"""
        report = {
            "total_queries": len(self.routing_history),
            "models": {}
        }
        
        for model_name, stats in self.model_performance.items():
            if stats["queries_routed"] > 0:
                report["models"][model_name] = {
                    "queries_routed": stats["queries_routed"],
                    "success_rate": stats["success_count"] / stats["queries_routed"],
                    "avg_cost": stats["total_cost"] / stats["queries_routed"],
                    "avg_latency_ms": stats["total_latency"] / stats["queries_routed"]
                }
        
        return report


def demonstrate_model_routing():
    """Demonstrate model routing and selection"""
    print("=" * 80)
    print("🎉 PATTERN 170: MODEL ROUTING & SELECTION 🎉")
    print("THE FINAL PATTERN - COMPLETING ALL 170 AGENTIC AI DESIGN PATTERNS!")
    print("=" * 80)
    
    router = ModelRouter()
    
    # Example 1: Different query complexities
    print("\n" + "=" * 80)
    print("Example 1: Routing Based on Query Complexity")
    print("=" * 80)
    
    queries = [
        "What is 2+2?",  # Simple
        "Explain the difference between machine learning and deep learning",  # Moderate
        "Design a distributed system architecture for a real-time analytics platform with fault tolerance"  # Complex
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")
        decision = router.route_query(query)
        print(f"   Selected: {decision.selected_model}")
        print(f"   Reasoning: {decision.reasoning}")
        print(f"   Confidence: {decision.confidence:.2f}")
        print(f"   Estimated cost: ${decision.estimated_cost:.6f}")
        print(f"   Estimated latency: {decision.estimated_latency:.0f}ms")
    
    # Example 2: Domain-specific routing
    print("\n" + "=" * 80)
    print("Example 2: Domain-Specific Routing")
    print("=" * 80)
    
    domain_queries = [
        ("Write a Python function to reverse a string", QueryDomain.CODE),
        ("Write a creative short story about a robot", QueryDomain.CREATIVE),
        ("Analyze the pros and cons of remote work", QueryDomain.ANALYTICAL)
    ]
    
    for query, expected_domain in domain_queries:
        print(f"\nQuery ({expected_domain.value}): {query}")
        decision = router.route_query(query)
        print(f"  → Routed to: {decision.selected_model}")
        print(f"  → Reasoning: {decision.reasoning}")
    
    # Example 3: Cost-constrained routing
    print("\n" + "=" * 80)
    print("Example 3: Cost-Constrained Routing")
    print("=" * 80)
    
    query = "Explain quantum computing"
    
    print(f"\nQuery: {query}")
    
    # No cost constraint
    decision1 = router.route_query(query)
    print(f"\n1. No cost constraint:")
    print(f"   Model: {decision1.selected_model}")
    print(f"   Cost: ${decision1.estimated_cost:.6f}")
    
    # Tight cost constraint
    decision2 = router.route_query(query, max_cost=0.001)
    print(f"\n2. Max cost $0.001:")
    print(f"   Model: {decision2.selected_model}")
    print(f"   Cost: ${decision2.estimated_cost:.6f}")
    
    # Example 4: Latency-sensitive routing
    print("\n" + "=" * 80)
    print("Example 4: Latency-Sensitive Routing")
    print("=" * 80)
    
    urgent_query = "Quick summary of this article"
    
    decision_fast = router.route_query(urgent_query, max_latency=1000)
    print(f"\nUrgent query: {urgent_query}")
    print(f"Max latency: 1000ms")
    print(f"Selected: {decision_fast.selected_model}")
    print(f"Expected latency: {decision_fast.estimated_latency:.0f}ms")
    
    # Example 5: Execute with routing
    print("\n" + "=" * 80)
    print("Example 5: Execute with Automatic Routing")
    print("=" * 80)
    
    test_query = "What are the benefits of exercise?"
    
    print(f"\nQuery: {test_query}")
    print("Routing and executing...")
    
    result = router.execute_with_routing(test_query)
    
    if result["success"]:
        print(f"\nRouted to: {result['routing_decision'].selected_model}")
        print(f"Actual latency: {result['actual_latency_ms']:.0f}ms")
        print(f"Response: {result['result'][:150]}...")
    else:
        print(f"Execution failed: {result.get('error')}")
    
    # Example 6: Batch routing analysis
    print("\n" + "=" * 80)
    print("Example 6: Batch Routing Analysis")
    print("=" * 80)
    
    batch_queries = [
        "Hello, how are you?",
        "Write a sorting algorithm in Python",
        "Explain Einstein's theory of relativity",
        "What's the weather like?",
        "Design a machine learning pipeline"
    ]
    
    print(f"\nRouting {len(batch_queries)} queries:")
    print("-" * 60)
    
    model_counts = defaultdict(int)
    total_cost = 0.0
    
    for query in batch_queries:
        decision = router.route_query(query)
        model_counts[decision.selected_model] += 1
        total_cost += decision.estimated_cost
    
    print("\nRouting Distribution:")
    for model, count in model_counts.items():
        print(f"  {model}: {count} queries ({count/len(batch_queries)*100:.0f}%)")
    
    print(f"\nTotal estimated cost: ${total_cost:.6f}")
    print(f"Average cost per query: ${total_cost/len(batch_queries):.6f}")
    
    # Example 7: Performance report
    print("\n" + "=" * 80)
    print("Example 7: Routing Performance Report")
    print("=" * 80)
    
    report = router.get_performance_report()
    
    print(f"\nTotal queries routed: {report['total_queries']}")
    print("\nPer-Model Statistics:")
    print("-" * 60)
    
    for model_name, stats in report["models"].items():
        print(f"\n{model_name}:")
        print(f"  Queries routed: {stats['queries_routed']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Avg cost: ${stats['avg_cost']:.6f}")
        print(f"  Avg latency: {stats['avg_latency_ms']:.0f}ms")
    
    # Final Summary
    print("\n" + "=" * 80)
    print("🎊 CONGRATULATIONS! 🎊")
    print("=" * 80)
    print("""
✅ ALL 170 AGENTIC AI DESIGN PATTERNS IMPLEMENTED! ✅

The Model Routing & Selection pattern enables:
✓ Intelligent query-to-model matching
✓ Cost-performance optimization
✓ Domain-aware routing
✓ Latency-sensitive selection
✓ Quality-constrained routing
✓ Performance monitoring
✓ Adaptive model selection

This pattern is valuable for:
- Production AI systems
- Cost optimization
- Multi-model architectures
- Performance-critical applications
- Resource management
- Quality assurance

🎉 MILESTONE ACHIEVED: 170/170 PATTERNS COMPLETE! 🎉

This completes the comprehensive implementation of all agentic AI design
patterns in LangChain/LangGraph, covering:
- Core Architectural Patterns
- Reasoning & Planning Patterns
- Multi-Agent Patterns
- Tool Use & Action Patterns
- Memory & State Management
- Interaction & Control
- Evaluation & Optimization
- Safety & Reliability
- Advanced Hybrid Patterns
- Emerging & Research Patterns
- Domain-Specific Patterns
- Implementation Patterns
- Prompt Engineering
- Resource Management
- Testing & Quality
- Communication Patterns
- Advanced Memory, Planning, Context, Learning
- Coordination & Orchestration
- Knowledge Management
- Dialogue & Interaction
- Specialization, Control, Performance
- Error Handling, Testing & Integration
- Advanced Reasoning
- AND Emerging Paradigms!

Thank you for following this comprehensive journey through agentic AI patterns!
    """)


if __name__ == "__main__":
    demonstrate_model_routing()
