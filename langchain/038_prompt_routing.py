"""
Pattern 038: Prompt Routing

Description:
    Prompt Routing intelligently routes queries to specialized prompts or models
    based on query characteristics such as type, complexity, domain, or intent.
    This enables efficient resource utilization and improved response quality by
    matching queries with the most appropriate handler.

Components:
    - Router: Classifies and routes queries
    - Specialized Prompts: Domain or task-specific prompts
    - Routing Logic: Rules or model-based classification
    - Fallback Handler: Default route for unmatched queries
    - Performance Tracker: Monitors routing decisions

Use Cases:
    - Multi-domain customer service
    - Enterprise knowledge systems
    - Cost optimization (route to cheaper models when possible)
    - Specialized expert systems
    - Content moderation and filtering
    - Dynamic model selection

LangChain Implementation:
    Uses classification chains and conditional routing to direct queries to
    appropriate specialized prompts or models based on intent and complexity.
"""

import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class QueryType(Enum):
    """Types of queries for routing."""
    TECHNICAL = "technical"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"
    FACTUAL = "factual"
    CODE = "code"
    MATH = "math"
    GENERAL = "general"


class QueryComplexity(Enum):
    """Complexity levels for queries."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class Route:
    """A routing destination."""
    name: str
    query_types: List[QueryType]
    prompt: ChatPromptTemplate
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    description: str = ""
    priority: int = 1  # Higher priority routes checked first


@dataclass
class RoutingDecision:
    """Result of routing decision."""
    query: str
    selected_route: str
    query_type: QueryType
    complexity: QueryComplexity
    confidence: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RoutedResponse:
    """Response from routed query."""
    query: str
    response: str
    route_used: str
    routing_decision: RoutingDecision
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)


class QueryRouter:
    """
    Routes queries to specialized prompts based on classification.
    
    Features:
    - Automatic query classification
    - Rule-based and model-based routing
    - Multiple routing strategies
    - Performance tracking
    - Fallback handling
    """
    
    def __init__(
        self,
        classification_model: str = "gpt-3.5-turbo",
        use_llm_classification: bool = True
    ):
        self.routes: Dict[str, Route] = {}
        self.default_route: Optional[Route] = None
        self.use_llm_classification = use_llm_classification
        
        if use_llm_classification:
            self.classifier_llm = ChatOpenAI(
                model=classification_model,
                temperature=0.1  # Low temperature for consistent classification
            )
            
            # Classification prompt
            self.classification_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a query classifier. Analyze the query and classify it.

Available Types:
- TECHNICAL: Software, hardware, technical troubleshooting
- CREATIVE: Writing, art, brainstorming, creative ideas
- ANALYTICAL: Data analysis, research, critical thinking
- CONVERSATIONAL: Casual chat, greetings, small talk
- FACTUAL: Facts, definitions, information lookup
- CODE: Programming, code generation, debugging
- MATH: Mathematics, calculations, formulas
- GENERAL: General questions that don't fit other categories

Complexity Levels:
- SIMPLE: Straightforward, single-step questions
- MODERATE: Multi-step but manageable questions
- COMPLEX: Requires deep reasoning or multiple components
- EXPERT: Highly specialized or advanced questions

Respond in this format:
TYPE: [query type]
COMPLEXITY: [complexity level]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]"""),
                ("user", "Query: {query}\n\nClassify this query.")
            ])
        
        self.routing_history: List[RoutingDecision] = []
    
    def add_route(self, route: Route):
        """Add a routing destination."""
        self.routes[route.name] = route
    
    def set_default_route(self, route: Route):
        """Set the default fallback route."""
        self.default_route = route
    
    def classify_query(self, query: str) -> tuple[QueryType, QueryComplexity, float, str]:
        """
        Classify query using LLM or rules.
        
        Returns:
            Tuple of (query_type, complexity, confidence, reasoning)
        """
        if self.use_llm_classification and hasattr(self, 'classifier_llm'):
            # LLM-based classification
            chain = self.classification_prompt | self.classifier_llm | StrOutputParser()
            classification = chain.invoke({"query": query})
            
            # Parse classification result
            query_type, complexity, confidence, reasoning = self._parse_classification(classification)
        else:
            # Rule-based classification
            query_type, complexity, confidence, reasoning = self._rule_based_classification(query)
        
        return query_type, complexity, confidence, reasoning
    
    def _parse_classification(self, classification: str) -> tuple[QueryType, QueryComplexity, float, str]:
        """Parse LLM classification output."""
        query_type = QueryType.GENERAL
        complexity = QueryComplexity.MODERATE
        confidence = 0.5
        reasoning = ""
        
        lines = classification.split('\n')
        for line in lines:
            if line.startswith("TYPE:"):
                type_str = line.split(':')[1].strip().upper()
                try:
                    query_type = QueryType[type_str]
                except (KeyError, ValueError):
                    pass
            elif line.startswith("COMPLEXITY:"):
                complexity_str = line.split(':')[1].strip().upper()
                try:
                    complexity = QueryComplexity[complexity_str]
                except (KeyError, ValueError):
                    pass
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.split(':', 1)[1].strip()
        
        return query_type, complexity, confidence, reasoning
    
    def _rule_based_classification(self, query: str) -> tuple[QueryType, QueryComplexity, float, str]:
        """Simple rule-based classification."""
        query_lower = query.lower()
        
        # Classify by keywords
        if any(word in query_lower for word in ['code', 'program', 'function', 'class', 'debug']):
            query_type = QueryType.CODE
            reasoning = "Keywords suggest coding question"
        elif any(word in query_lower for word in ['calculate', 'solve', 'equation', 'math']):
            query_type = QueryType.MATH
            reasoning = "Keywords suggest mathematical question"
        elif any(word in query_lower for word in ['write', 'create', 'imagine', 'story', 'poem']):
            query_type = QueryType.CREATIVE
            reasoning = "Keywords suggest creative task"
        elif any(word in query_lower for word in ['analyze', 'compare', 'evaluate', 'research']):
            query_type = QueryType.ANALYTICAL
            reasoning = "Keywords suggest analytical task"
        elif any(word in query_lower for word in ['hello', 'hi', 'how are you', 'thanks']):
            query_type = QueryType.CONVERSATIONAL
            reasoning = "Conversational greeting or response"
        elif any(word in query_lower for word in ['what is', 'define', 'who is', 'when']):
            query_type = QueryType.FACTUAL
            reasoning = "Factual information request"
        else:
            query_type = QueryType.GENERAL
            reasoning = "General question"
        
        # Estimate complexity by length and question marks
        word_count = len(query.split())
        question_marks = query.count('?')
        
        if word_count < 10 and question_marks <= 1:
            complexity = QueryComplexity.SIMPLE
        elif word_count < 30:
            complexity = QueryComplexity.MODERATE
        elif word_count < 60:
            complexity = QueryComplexity.COMPLEX
        else:
            complexity = QueryComplexity.EXPERT
        
        confidence = 0.7  # Rule-based has moderate confidence
        
        return query_type, complexity, confidence, reasoning
    
    def route(self, query: str) -> RoutingDecision:
        """
        Determine the best route for a query.
        
        Args:
            query: The query to route
            
        Returns:
            RoutingDecision with selected route
        """
        # Classify query
        query_type, complexity, confidence, reasoning = self.classify_query(query)
        
        # Find matching route
        selected_route = None
        
        # Sort routes by priority
        sorted_routes = sorted(self.routes.values(), key=lambda r: r.priority, reverse=True)
        
        for route in sorted_routes:
            if query_type in route.query_types:
                selected_route = route.name
                break
        
        # Use default route if no match
        if not selected_route and self.default_route:
            selected_route = self.default_route.name
        
        decision = RoutingDecision(
            query=query,
            selected_route=selected_route or "none",
            query_type=query_type,
            complexity=complexity,
            confidence=confidence,
            reasoning=reasoning
        )
        
        self.routing_history.append(decision)
        
        return decision
    
    def execute_route(self, query: str) -> RoutedResponse:
        """
        Route and execute query.
        
        Args:
            query: The query to process
            
        Returns:
            RoutedResponse with result
        """
        start_time = datetime.now()
        
        # Get routing decision
        decision = self.route(query)
        
        # Get the selected route
        if decision.selected_route not in self.routes:
            if self.default_route:
                route = self.default_route
            else:
                return RoutedResponse(
                    query=query,
                    response="No appropriate route found for this query.",
                    route_used="none",
                    routing_decision=decision,
                    execution_time=0.0
                )
        else:
            route = self.routes[decision.selected_route]
        
        # Execute using the selected route
        llm = ChatOpenAI(model=route.model, temperature=route.temperature)
        chain = route.prompt | llm | StrOutputParser()
        
        response = chain.invoke({"query": query})
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        return RoutedResponse(
            query=query,
            response=response,
            route_used=route.name,
            routing_decision=decision,
            execution_time=execution_time
        )
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about routing decisions."""
        if not self.routing_history:
            return {"total_routes": 0}
        
        by_type = {}
        by_complexity = {}
        by_route = {}
        
        for decision in self.routing_history:
            by_type[decision.query_type.value] = by_type.get(decision.query_type.value, 0) + 1
            by_complexity[decision.complexity.value] = by_complexity.get(decision.complexity.value, 0) + 1
            by_route[decision.selected_route] = by_route.get(decision.selected_route, 0) + 1
        
        avg_confidence = sum(d.confidence for d in self.routing_history) / len(self.routing_history)
        
        return {
            "total_routes": len(self.routing_history),
            "by_type": by_type,
            "by_complexity": by_complexity,
            "by_route": by_route,
            "average_confidence": avg_confidence
        }


def demonstrate_prompt_routing():
    """
    Demonstrates prompt routing with specialized handlers for different query types.
    """
    print("=" * 80)
    print("PROMPT ROUTING DEMONSTRATION")
    print("=" * 80)
    
    # Create router
    router = QueryRouter(use_llm_classification=True)
    
    # Define specialized routes
    
    # 1. Code route
    code_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert programmer. Provide clear, well-commented code solutions."),
        ("user", "{query}")
    ])
    code_route = Route(
        name="code_expert",
        query_types=[QueryType.CODE],
        prompt=code_prompt,
        temperature=0.3,
        description="Handles programming and code questions",
        priority=10
    )
    router.add_route(code_route)
    
    # 2. Creative route
    creative_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a creative writer. Provide imaginative, engaging responses."),
        ("user", "{query}")
    ])
    creative_route = Route(
        name="creative_writer",
        query_types=[QueryType.CREATIVE],
        prompt=creative_prompt,
        temperature=0.9,
        description="Handles creative writing tasks",
        priority=9
    )
    router.add_route(creative_route)
    
    # 3. Math route
    math_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a mathematics expert. Provide step-by-step solutions with clear explanations."),
        ("user", "{query}")
    ])
    math_route = Route(
        name="math_expert",
        query_types=[QueryType.MATH],
        prompt=math_prompt,
        temperature=0.1,
        description="Handles mathematical questions",
        priority=10
    )
    router.add_route(math_route)
    
    # 4. Analytical route
    analytical_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an analytical expert. Provide thorough analysis with evidence and reasoning."),
        ("user", "{query}")
    ])
    analytical_route = Route(
        name="analytical_expert",
        query_types=[QueryType.ANALYTICAL],
        prompt=analytical_prompt,
        temperature=0.5,
        description="Handles analytical and research questions",
        priority=8
    )
    router.add_route(analytical_route)
    
    # 5. Default route
    default_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        ("user", "{query}")
    ])
    default_route = Route(
        name="general_assistant",
        query_types=[QueryType.GENERAL, QueryType.CONVERSATIONAL, QueryType.FACTUAL, QueryType.TECHNICAL],
        prompt=default_prompt,
        temperature=0.7,
        description="Handles general questions",
        priority=1
    )
    router.add_route(default_route)
    router.set_default_route(default_route)
    
    # Show available routes
    print("\n" + "=" * 80)
    print("Available Routes")
    print("=" * 80)
    
    for name, route in router.routes.items():
        print(f"\n{name}:")
        print(f"  Description: {route.description}")
        print(f"  Query Types: {[qt.value for qt in route.query_types]}")
        print(f"  Temperature: {route.temperature}")
        print(f"  Priority: {route.priority}")
    
    # Test different query types
    test_queries = [
        "Write a Python function to calculate fibonacci numbers",
        "Write a short poem about artificial intelligence",
        "What is 156 multiplied by 327?",
        "Analyze the pros and cons of renewable energy",
        "Hello! How are you today?",
    ]
    
    print("\n" + "=" * 80)
    print("Routing Queries")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-' * 80}")
        print(f"Query {i}: {query}")
        print(f"{'-' * 80}")
        
        result = router.execute_route(query)
        
        print(f"\n[Routing Decision]")
        print(f"  Route: {result.route_used}")
        print(f"  Type: {result.routing_decision.query_type.value}")
        print(f"  Complexity: {result.routing_decision.complexity.value}")
        print(f"  Confidence: {result.routing_decision.confidence:.2f}")
        print(f"  Reasoning: {result.routing_decision.reasoning}")
        
        print(f"\n[Response]")
        response_preview = result.response[:200] + "..." if len(result.response) > 200 else result.response
        print(f"  {response_preview}")
        print(f"\n  Execution Time: {result.execution_time:.2f}s")
    
    # Show routing statistics
    print("\n" + "=" * 80)
    print("Routing Statistics")
    print("=" * 80)
    
    stats = router.get_routing_stats()
    print(f"\nTotal Queries Routed: {stats['total_routes']}")
    print(f"Average Confidence: {stats['average_confidence']:.2f}")
    
    print("\nQueries by Type:")
    for qtype, count in stats['by_type'].items():
        print(f"  - {qtype}: {count}")
    
    print("\nQueries by Complexity:")
    for complexity, count in stats['by_complexity'].items():
        print(f"  - {complexity}: {count}")
    
    print("\nQueries by Route:")
    for route, count in stats['by_route'].items():
        print(f"  - {route}: {count}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Prompt Routing provides:
✓ Intelligent query classification
✓ Specialized prompt selection
✓ Model and parameter optimization
✓ Cost-effective resource allocation
✓ Domain expertise routing
✓ Performance tracking

This pattern excels at:
- Multi-domain systems
- Cost optimization
- Quality through specialization
- Dynamic model selection
- Enterprise knowledge systems
- Efficient resource use

Routing components:
1. Router: Classifies and routes queries
2. Specialized Prompts: Domain-specific handlers
3. Classification: LLM or rule-based
4. Fallback: Default handler
5. Tracker: Performance monitoring

Query classification:
- Query Type: Domain/category (code, creative, math, etc.)
- Complexity: Difficulty level (simple to expert)
- Confidence: Classification certainty
- Reasoning: Why this classification

Routing strategies:
- Rule-based: Keyword matching, patterns
- Model-based: LLM classification
- Hybrid: Combine rules and models
- Priority-based: Check high-priority routes first
- Confidence-based: Threshold for routing

Route characteristics:
- Name: Identifier
- Query Types: What it handles
- Prompt: Specialized prompt template
- Model: Which LLM to use
- Temperature: Generation parameters
- Priority: Routing precedence

Benefits:
- Specialization: Expert prompts for domains
- Efficiency: Right model for right task
- Cost: Use cheaper models when appropriate
- Quality: Optimized prompts and parameters
- Scalability: Easy to add new routes
- Monitoring: Track routing decisions

Use Prompt Routing when you need:
- Multi-domain question answering
- Cost-optimized model selection
- Specialized expert systems
- Quality through domain expertise
- Dynamic resource allocation
- Large-scale production systems
""")


if __name__ == "__main__":
    demonstrate_prompt_routing()
