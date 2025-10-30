"""
Pattern 054: Agent Specialization & Routing

Description:
    The Agent Specialization & Routing pattern creates a system of domain-specific
    expert agents with intelligent routing logic. A meta-agent classifies incoming
    tasks and routes them to the most appropriate specialist, enabling efficient
    handling of diverse tasks by leveraging specialized expertise.

Components:
    1. Specialist Agents: Domain-specific expert agents
    2. Router Agent: Classifies and routes tasks
    3. Task Classifier: Determines task domain/type
    4. Capability Registry: Tracks agent specializations
    5. Load Balancer: Distributes work efficiently

Use Cases:
    - Multi-domain customer support
    - Technical help desk routing
    - Content moderation by category
    - Code review by language
    - Medical diagnosis routing by specialty
    - Legal document analysis by practice area

LangChain Implementation:
    Uses a classifier LLM to route queries to specialized LLM instances
    configured for specific domains, with fallback to generalist agents.
"""

import os
import time
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class Domain(Enum):
    """Domains for specialized agents"""
    TECHNICAL = "technical"  # Programming, IT
    MEDICAL = "medical"  # Healthcare, diagnosis
    LEGAL = "legal"  # Law, contracts
    FINANCIAL = "financial"  # Finance, accounting
    CREATIVE = "creative"  # Writing, design
    SCIENTIFIC = "scientific"  # Research, analysis
    CUSTOMER_SERVICE = "customer_service"  # Support, relations
    GENERAL = "general"  # Default fallback


class RoutingStrategy(Enum):
    """Strategies for routing tasks"""
    BEST_MATCH = "best_match"  # Route to single best specialist
    MULTI_ROUTE = "multi_route"  # Route to multiple relevant specialists
    LOAD_BALANCED = "load_balanced"  # Balance across similar specialists
    CONFIDENCE_BASED = "confidence_based"  # Route based on confidence threshold


@dataclass
class AgentCapability:
    """Describes what an agent can do"""
    domain: Domain
    specializations: List[str]
    expertise_level: float  # 0.0-1.0
    keywords: Set[str]
    examples: List[str]
    
    def matches_query(self, query: str) -> float:
        """Calculate match score for query"""
        query_lower = query.lower()
        
        # Keyword matching
        keyword_score = sum(
            1 for keyword in self.keywords
            if keyword.lower() in query_lower
        ) / max(len(self.keywords), 1)
        
        # Specialization matching
        spec_score = sum(
            1 for spec in self.specializations
            if spec.lower() in query_lower
        ) / max(len(self.specializations), 1)
        
        # Combine scores
        match_score = (keyword_score * 0.6 + spec_score * 0.4) * self.expertise_level
        
        return min(match_score, 1.0)


@dataclass
class SpecialistAgent:
    """A specialized agent with specific capabilities"""
    agent_id: str
    domain: Domain
    capability: AgentCapability
    llm: ChatOpenAI
    system_prompt: str
    request_count: int = 0
    total_time_ms: float = 0.0
    
    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.request_count if self.request_count > 0 else 0.0


@dataclass
class RoutingDecision:
    """Decision made by router"""
    query: str
    selected_agents: List[str]
    domain: Domain
    confidence: float
    reasoning: str
    alternatives: List[tuple[str, float]]  # (agent_id, score)


@dataclass
class RoutingResult:
    """Result from specialized agent routing"""
    query: str
    routing_decision: RoutingDecision
    specialist_response: str
    agent_id: str
    domain: Domain
    execution_time_ms: float
    routing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "routed_to": self.agent_id,
            "domain": self.domain.value,
            "confidence": f"{self.routing_decision.confidence:.2f}",
            "execution_time_ms": f"{self.execution_time_ms:.1f}",
            "routing_time_ms": f"{self.routing_time_ms:.1f}"
        }


class AgentRouter:
    """
    Intelligent routing system for specialized agents.
    
    Features:
    1. Multiple domain-specific specialist agents
    2. Intelligent task classification and routing
    3. Confidence-based decision making
    4. Load balancing capabilities
    5. Fallback to generalist agent
    """
    
    def __init__(
        self,
        routing_strategy: RoutingStrategy = RoutingStrategy.BEST_MATCH,
        confidence_threshold: float = 0.6
    ):
        self.routing_strategy = routing_strategy
        self.confidence_threshold = confidence_threshold
        
        # Classifier for routing
        self.classifier = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.1  # Low temperature for consistent classification
        )
        
        # Registry of specialist agents
        self.specialists: Dict[str, SpecialistAgent] = {}
        
        # Generalist fallback
        self.generalist = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # Initialize specialists
        self._initialize_specialists()
    
    def _initialize_specialists(self):
        """Create specialized agents for different domains"""
        
        # Technical specialist
        tech_capability = AgentCapability(
            domain=Domain.TECHNICAL,
            specializations=["programming", "debugging", "algorithms", "systems"],
            expertise_level=0.9,
            keywords={"code", "program", "bug", "error", "function", "algorithm", 
                     "database", "API", "server", "debug"},
            examples=["How do I debug this code?", "Explain quicksort algorithm"]
        )
        
        self.specialists["tech_expert"] = SpecialistAgent(
            agent_id="tech_expert",
            domain=Domain.TECHNICAL,
            capability=tech_capability,
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3),
            system_prompt="""You are a senior software engineer and technical expert.
You specialize in programming, algorithms, debugging, and system design.
Provide clear, accurate technical answers with code examples when appropriate."""
        )
        
        # Medical specialist
        medical_capability = AgentCapability(
            domain=Domain.MEDICAL,
            specializations=["diagnosis", "treatment", "health", "medicine"],
            expertise_level=0.85,
            keywords={"health", "medical", "doctor", "symptom", "disease", "treatment",
                     "medicine", "diagnosis", "patient", "clinical"},
            examples=["What causes headaches?", "Explain diabetes management"]
        )
        
        self.specialists["medical_expert"] = SpecialistAgent(
            agent_id="medical_expert",
            domain=Domain.MEDICAL,
            capability=medical_capability,
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2),
            system_prompt="""You are a medical information specialist.
Provide accurate health and medical information. Always include disclaimer
that this is educational information and not medical advice."""
        )
        
        # Financial specialist
        financial_capability = AgentCapability(
            domain=Domain.FINANCIAL,
            specializations=["investing", "accounting", "economics", "trading"],
            expertise_level=0.85,
            keywords={"money", "investment", "stock", "finance", "accounting", "tax",
                     "budget", "economy", "market", "portfolio"},
            examples=["How to start investing?", "Explain compound interest"]
        )
        
        self.specialists["finance_expert"] = SpecialistAgent(
            agent_id="finance_expert",
            domain=Domain.FINANCIAL,
            capability=financial_capability,
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2),
            system_prompt="""You are a financial advisor and economics expert.
Provide clear explanations of financial concepts. Include disclaimer that
this is educational information and not financial advice."""
        )
        
        # Creative specialist
        creative_capability = AgentCapability(
            domain=Domain.CREATIVE,
            specializations=["writing", "storytelling", "design", "creative"],
            expertise_level=0.9,
            keywords={"story", "write", "creative", "design", "art", "narrative",
                     "plot", "character", "style", "aesthetic"},
            examples=["Help me write a story", "Design a logo concept"]
        )
        
        self.specialists["creative_expert"] = SpecialistAgent(
            agent_id="creative_expert",
            domain=Domain.CREATIVE,
            capability=creative_capability,
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9),
            system_prompt="""You are a creative writing and design expert.
Provide imaginative, original ideas and guidance. Be encouraging and
help users explore their creativity."""
        )
        
        # Scientific specialist
        science_capability = AgentCapability(
            domain=Domain.SCIENTIFIC,
            specializations=["research", "physics", "chemistry", "biology", "data"],
            expertise_level=0.85,
            keywords={"research", "science", "experiment", "hypothesis", "data",
                     "analysis", "physics", "chemistry", "biology", "scientific"},
            examples=["Explain quantum mechanics", "How to design an experiment"]
        )
        
        self.specialists["science_expert"] = SpecialistAgent(
            agent_id="science_expert",
            domain=Domain.SCIENTIFIC,
            capability=science_capability,
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3),
            system_prompt="""You are a research scientist with expertise across disciplines.
Provide accurate scientific explanations. Cite principles and methodologies.
Be precise and evidence-based."""
        )
    
    def _classify_query(self, query: str) -> RoutingDecision:
        """Classify query and determine routing"""
        
        start_time = time.time()
        
        # Calculate match scores for all specialists
        scores = []
        for agent_id, specialist in self.specialists.items():
            match_score = specialist.capability.matches_query(query)
            
            # Factor in load balancing
            if self.routing_strategy == RoutingStrategy.LOAD_BALANCED:
                # Penalize busy agents slightly
                load_penalty = specialist.request_count * 0.01
                match_score = max(0, match_score - load_penalty)
            
            scores.append((agent_id, match_score, specialist.domain))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Best match
        best_agent_id, best_score, best_domain = scores[0]
        
        # Use LLM for low-confidence cases
        if best_score < self.confidence_threshold:
            # Ask LLM to classify
            classifier_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a query classifier. Classify queries into domains:
- technical: Programming, IT, debugging
- medical: Health, medicine, diagnosis
- financial: Money, investing, economics
- creative: Writing, art, design
- scientific: Research, physics, experiments
- general: Everything else

Respond with just the domain name."""),
                ("user", "Classify: {query}")
            ])
            
            chain = classifier_prompt | self.classifier | StrOutputParser()
            llm_domain = chain.invoke({"query": query}).strip().lower()
            
            # Find agent for this domain
            for agent_id, specialist in self.specialists.items():
                if specialist.domain.value == llm_domain:
                    best_agent_id = agent_id
                    best_domain = specialist.domain
                    best_score = 0.7  # LLM classification gets moderate confidence
                    break
        
        # Determine routing
        if self.routing_strategy == RoutingStrategy.MULTI_ROUTE:
            # Route to top 2 specialists
            selected_agents = [scores[0][0], scores[1][0]]
        else:
            selected_agents = [best_agent_id]
        
        reasoning = f"Matched domain: {best_domain.value} (score: {best_score:.2f})"
        if best_score < self.confidence_threshold:
            reasoning += " - Used LLM classification for low confidence"
        
        return RoutingDecision(
            query=query,
            selected_agents=selected_agents,
            domain=best_domain,
            confidence=best_score,
            reasoning=reasoning,
            alternatives=[(agent_id, score) for agent_id, score, _ in scores[1:3]]
        )
    
    def _execute_with_specialist(
        self,
        specialist: SpecialistAgent,
        query: str
    ) -> str:
        """Execute query with specialist agent"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", specialist.system_prompt),
            ("user", "{query}")
        ])
        
        chain = prompt | specialist.llm | StrOutputParser()
        
        start_time = time.time()
        response = chain.invoke({"query": query})
        execution_time = (time.time() - start_time) * 1000
        
        # Update statistics
        specialist.request_count += 1
        specialist.total_time_ms += execution_time
        
        return response
    
    def route_and_execute(self, query: str) -> RoutingResult:
        """Route query to specialist and execute"""
        
        total_start = time.time()
        
        # Classify and route
        routing_start = time.time()
        routing_decision = self._classify_query(query)
        routing_time_ms = (time.time() - routing_start) * 1000
        
        # Execute with primary specialist
        primary_agent_id = routing_decision.selected_agents[0]
        
        if primary_agent_id in self.specialists:
            specialist = self.specialists[primary_agent_id]
            response = self._execute_with_specialist(specialist, query)
        else:
            # Fallback to generalist
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant providing accurate information."),
                ("user", "{query}")
            ])
            chain = prompt | self.generalist | StrOutputParser()
            response = chain.invoke({"query": query})
            primary_agent_id = "generalist"
            routing_decision.domain = Domain.GENERAL
        
        total_time_ms = (time.time() - total_start) * 1000
        execution_time_ms = total_time_ms - routing_time_ms
        
        return RoutingResult(
            query=query,
            routing_decision=routing_decision,
            specialist_response=response,
            agent_id=primary_agent_id,
            domain=routing_decision.domain,
            execution_time_ms=execution_time_ms,
            routing_time_ms=routing_time_ms
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        
        stats = {
            "total_specialists": len(self.specialists),
            "specialists": {}
        }
        
        for agent_id, specialist in self.specialists.items():
            stats["specialists"][agent_id] = {
                "domain": specialist.domain.value,
                "request_count": specialist.request_count,
                "avg_time_ms": f"{specialist.avg_time_ms:.1f}",
                "expertise": f"{specialist.capability.expertise_level:.2f}"
            }
        
        return stats


def demonstrate_agent_routing():
    """Demonstrate Agent Specialization & Routing pattern"""
    
    print("=" * 80)
    print("PATTERN 054: AGENT SPECIALIZATION & ROUTING DEMONSTRATION")
    print("=" * 80)
    print("\nIntelligent routing to domain-specific expert agents\n")
    
    # Create router
    router = AgentRouter(
        routing_strategy=RoutingStrategy.BEST_MATCH,
        confidence_threshold=0.6
    )
    
    # Test queries for different domains
    test_queries = [
        ("How do I fix a memory leak in Python?", Domain.TECHNICAL),
        ("What causes high blood pressure?", Domain.MEDICAL),
        ("Should I invest in index funds or individual stocks?", Domain.FINANCIAL),
        ("Help me write an opening for a mystery novel", Domain.CREATIVE),
        ("Explain the double-slit experiment", Domain.SCIENTIFIC),
        ("What's the weather like today?", Domain.GENERAL)  # Should route to general
    ]
    
    # Test 1: Basic routing
    print("\n" + "=" * 80)
    print("TEST 1: Routing to Specialized Agents")
    print("=" * 80)
    
    for query, expected_domain in test_queries[:3]:
        print(f"\n📝 Query: {query}")
        print(f"   Expected Domain: {expected_domain.value}")
        
        result = router.route_and_execute(query)
        
        print(f"\n   ✅ Routing Decision:")
        print(f"      Routed to: {result.agent_id}")
        print(f"      Domain: {result.domain.value}")
        print(f"      Confidence: {result.routing_decision.confidence:.2f}")
        print(f"      Reasoning: {result.routing_decision.reasoning}")
        
        if result.routing_decision.alternatives:
            print(f"\n   📊 Alternative Matches:")
            for alt_agent, alt_score in result.routing_decision.alternatives:
                print(f"      - {alt_agent}: {alt_score:.2f}")
        
        print(f"\n   💬 Response: {result.specialist_response[:150]}...")
        print(f"\n   ⏱️  Timing:")
        print(f"      Routing: {result.routing_time_ms:.1f}ms")
        print(f"      Execution: {result.execution_time_ms:.1f}ms")
    
    # Test 2: Creative vs Technical routing
    print("\n" + "=" * 80)
    print("TEST 2: Domain Contrast - Creative vs Technical")
    print("=" * 80)
    
    creative_query = "Write a poem about artificial intelligence"
    technical_query = "Implement a binary search algorithm"
    
    for query in [creative_query, technical_query]:
        print(f"\n📝 Query: {query}")
        result = router.route_and_execute(query)
        
        print(f"   Routed to: {result.agent_id} ({result.domain.value})")
        print(f"   Response: {result.specialist_response[:200]}...")
    
    # Test 3: Load balancing simulation
    print("\n" + "=" * 80)
    print("TEST 3: Load Balancing")
    print("=" * 80)
    
    # Create load-balanced router
    lb_router = AgentRouter(
        routing_strategy=RoutingStrategy.LOAD_BALANCED,
        confidence_threshold=0.5
    )
    
    # Simulate multiple requests
    tech_queries = [
        "What is recursion?",
        "Explain REST APIs",
        "How does Git work?"
    ]
    
    print("\nSimulating multiple technical queries with load balancing...")
    
    for i, query in enumerate(tech_queries, 1):
        result = lb_router.route_and_execute(query)
        print(f"\n   Request {i}: {query[:40]}...")
        print(f"      Routed to: {result.agent_id}")
        print(f"      Execution time: {result.execution_time_ms:.1f}ms")
    
    # Show statistics
    print("\n   📊 Load Distribution:")
    stats = lb_router.get_statistics()
    for agent_id, agent_stats in stats["specialists"].items():
        if agent_stats["request_count"] > 0:
            print(f"      {agent_id}: {agent_stats['request_count']} requests, "
                  f"avg {agent_stats['avg_time_ms']}ms")
    
    # Test 4: Ambiguous query routing
    print("\n" + "=" * 80)
    print("TEST 4: Handling Ambiguous Queries")
    print("=" * 80)
    
    ambiguous_queries = [
        "Tell me about neural networks",  # Could be technical or scientific
        "How to manage stress",  # Could be medical or general
    ]
    
    for query in ambiguous_queries:
        print(f"\n📝 Query: {query}")
        result = router.route_and_execute(query)
        
        print(f"   Decision:")
        print(f"      Selected: {result.agent_id} ({result.domain.value})")
        print(f"      Confidence: {result.routing_decision.confidence:.2f}")
        print(f"      Reasoning: {result.routing_decision.reasoning}")
        
        if result.routing_decision.alternatives:
            print(f"   Alternatives considered:")
            for alt_agent, alt_score in result.routing_decision.alternatives:
                specialist = router.specialists.get(alt_agent)
                domain = specialist.domain.value if specialist else "unknown"
                print(f"      - {alt_agent} ({domain}): {alt_score:.2f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("AGENT SPECIALIZATION & ROUTING PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Expert Quality: Specialized agents provide domain-specific expertise
2. Efficiency: Route to appropriate agent immediately
3. Scalability: Easy to add new specialists
4. Load Distribution: Balance work across agents
5. Maintainability: Domain-specific logic in separate agents

Implementation Features:
1. Capability Registry: Tracks agent specializations
2. Intelligent Classification: Keyword + LLM-based routing
3. Confidence Thresholding: Fallback for uncertain cases
4. Load Balancing: Distribute requests evenly
5. Statistics Tracking: Monitor agent performance

Routing Strategies:
1. Best Match: Route to single best specialist
2. Multi-Route: Send to multiple relevant specialists
3. Load Balanced: Distribute across similar specialists
4. Confidence-Based: Threshold-based routing decisions

Agent Specializations:
- Technical: Programming, debugging, algorithms
- Medical: Health, diagnosis, treatment
- Financial: Investing, accounting, economics
- Creative: Writing, design, storytelling
- Scientific: Research, experiments, analysis
- General: Fallback for unclassified queries

Classification Methods:
1. Keyword Matching: Fast, rule-based
2. LLM Classification: Accurate for ambiguous cases
3. Hybrid: Combine both approaches
4. Confidence Scoring: Quantify match quality

Use Cases:
- Customer support ticket routing
- Technical helpdesk by expertise
- Content moderation by category
- Medical triage systems
- Legal document routing
- Educational tutoring systems

Best Practices:
1. Clear domain boundaries
2. Comprehensive keyword sets
3. LLM fallback for edge cases
4. Monitor routing accuracy
5. Track specialist performance
6. Regular capability updates
7. Maintain generalist fallback

Production Considerations:
- Caching classification results
- A/B testing routing strategies
- Monitoring routing accuracy
- Specialist availability checks
- Timeout handling per agent
- Cost optimization by routing
- User feedback on routing quality

Comparison with Related Patterns:
- vs. Mixture of Agents: Routing vs aggregation
- vs. Multi-Agent: Specialized vs collaborative
- vs. Chain: Direct routing vs sequential
- vs. Hierarchical: Flat routing vs tree structure

The Agent Specialization & Routing pattern enables efficient, scalable
systems that leverage domain expertise while maintaining simplicity
through intelligent task distribution.
""")


if __name__ == "__main__":
    demonstrate_agent_routing()
