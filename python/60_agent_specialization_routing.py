"""
Pattern 54: Agent Specialization & Routing
Description:
    Routes requests to specialized agents based on task requirements,
    agent capabilities, and performance history.
Use Cases:
    - Domain-specific task handling
    - Capability-based routing
    - Load balancing across specialists
    - Dynamic agent selection
Key Features:
    - Capability matching
    - Performance-based routing
    - Dynamic specialist registration
    - Routing strategy selection
Example:
    >>> router = AgentRouter()
    >>> router.register_specialist(agent, capabilities=['python', 'data'])
    >>> result = router.route_request(task, required_capabilities=['python'])
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Callable
from enum import Enum
import time
from collections import defaultdict
import re
class RoutingStrategy(Enum):
    """Strategies for routing requests"""
    CAPABILITY_MATCH = "capability_match"
    PERFORMANCE_BASED = "performance_based"
    LOAD_BALANCED = "load_balanced"
    ROUND_ROBIN = "round_robin"
    SPECIALIZED_FIRST = "specialized_first"
    HYBRID = "hybrid"
class CapabilityLevel(Enum):
    """Proficiency levels for capabilities"""
    EXPERT = 3
    INTERMEDIATE = 2
    BEGINNER = 1
@dataclass
class Capability:
    """A capability/skill"""
    name: str
    level: CapabilityLevel
    keywords: Set[str] = field(default_factory=set)
    success_rate: float = 1.0
@dataclass
class AgentSpecialization:
    """Specialization profile for an agent"""
    agent_id: str
    agent: Any
    capabilities: Dict[str, Capability]
    performance_history: List[float] = field(default_factory=list)
    total_requests: int = 0
    successful_requests: int = 0
    average_response_time: float = 0.0
    current_load: int = 0
    max_concurrent: int = 10
@dataclass
class RoutingDecision:
    """Decision about request routing"""
    selected_agent_id: str
    match_score: float
    routing_strategy: RoutingStrategy
    matched_capabilities: List[str]
    reasoning: str
    alternatives: List[tuple[str, float]] = field(default_factory=list)
@dataclass
class RequestRequirements:
    """Requirements for a request"""
    required_capabilities: Set[str]
    preferred_capabilities: Set[str] = field(default_factory=set)
    min_capability_level: CapabilityLevel = CapabilityLevel.BEGINNER
    keywords: Set[str] = field(default_factory=set)
    priority: int = 1
class AgentRouter:
    """
    Routes requests to specialized agents
    Features:
    - Capability-based matching
    - Performance tracking
    - Multiple routing strategies
    - Load balancing
    """
    def __init__(
        self,
        default_strategy: RoutingStrategy = RoutingStrategy.HYBRID
    ):
        self.specialists: Dict[str, AgentSpecialization] = {}
        self.default_strategy = default_strategy
        self.routing_history: List[RoutingDecision] = []
        self.round_robin_index = 0
    def register_specialist(
        self,
        agent: Any,
        agent_id: str,
        capabilities: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Register a specialized agent
        Args:
            agent: Agent instance
            agent_id: Unique identifier
            capabilities: Dict of capability_name -> {level, keywords}
        Returns:
            Agent ID
        """
        capability_objects = {}
        for cap_name, cap_config in capabilities.items():
            capability_objects[cap_name] = Capability(
                name=cap_name,
                level=cap_config.get('level', CapabilityLevel.INTERMEDIATE),
                keywords=set(cap_config.get('keywords', []))
            )
        specialization = AgentSpecialization(
            agent_id=agent_id,
            agent=agent,
            capabilities=capability_objects
        )
        self.specialists[agent_id] = specialization
        return agent_id
    def route_request(
        self,
        task: str,
        requirements: Optional[RequestRequirements] = None,
        strategy: Optional[RoutingStrategy] = None
    ) -> Dict[str, Any]:
        """
        Route request to appropriate specialist
        Args:
            task: Task description
            requirements: Task requirements
            strategy: Routing strategy to use
        Returns:
            Result with routing decision and execution
        """
        # Infer requirements if not provided
        if requirements is None:
            requirements = self._infer_requirements(task)
        # Select routing strategy
        strategy = strategy or self.default_strategy
        # Find best agent
        routing_decision = self._select_agent(requirements, strategy)
        if not routing_decision:
            return {
                'success': False,
                'error': 'No suitable agent found',
                'requirements': requirements
            }
        # Execute with selected agent
        result = self._execute_with_agent(
            routing_decision.selected_agent_id,
            task,
            requirements
        )
        # Update performance metrics
        self._update_performance(
            routing_decision.selected_agent_id,
            result.get('success', False),
            result.get('execution_time', 0)
        )
        # Store routing decision
        self.routing_history.append(routing_decision)
        return {
            'success': result.get('success', False),
            'result': result.get('output'),
            'routing_decision': routing_decision,
            'execution_time': result.get('execution_time'),
            'agent_id': routing_decision.selected_agent_id
        }
    def _infer_requirements(self, task: str) -> RequestRequirements:
        """Infer requirements from task description"""
        task_lower = task.lower()
        keywords = set(re.findall(r'\w+', task_lower))
        required_caps = set()
        # Simple keyword-based inference
        capability_keywords = {
            'python': ['python', 'pandas', 'numpy', 'code'],
            'data_analysis': ['data', 'analyze', 'statistics', 'chart'],
            'web': ['web', 'http', 'api', 'server'],
            'ml': ['machine learning', 'model', 'train', 'predict'],
            'database': ['database', 'sql', 'query', 'table']
        }
        for capability, cap_keywords in capability_keywords.items():
            if any(kw in task_lower for kw in cap_keywords):
                required_caps.add(capability)
        return RequestRequirements(
            required_capabilities=required_caps,
            keywords=keywords
        )
    def _select_agent(
        self,
        requirements: RequestRequirements,
        strategy: RoutingStrategy
    ) -> Optional[RoutingDecision]:
        """Select best agent for requirements"""
        if strategy == RoutingStrategy.CAPABILITY_MATCH:
            return self._capability_match_routing(requirements)
        elif strategy == RoutingStrategy.PERFORMANCE_BASED:
            return self._performance_based_routing(requirements)
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            return self._load_balanced_routing(requirements)
        elif strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_routing()
        elif strategy == RoutingStrategy.SPECIALIZED_FIRST:
            return self._specialized_first_routing(requirements)
        elif strategy == RoutingStrategy.HYBRID:
            return self._hybrid_routing(requirements)
        return None
    def _capability_match_routing(
        self,
        requirements: RequestRequirements
    ) -> Optional[RoutingDecision]:
        """Route based on capability matching"""
        candidates = []
        for agent_id, specialist in self.specialists.items():
            match_score = self._calculate_capability_match(
                specialist,
                requirements
            )
            if match_score > 0:
                matched_caps = self._get_matched_capabilities(
                    specialist,
                    requirements
                )
                candidates.append((agent_id, match_score, matched_caps))
        if not candidates:
            return None
        # Sort by match score
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_agent_id, best_score, matched_caps = candidates[0]
        return RoutingDecision(
            selected_agent_id=best_agent_id,
            match_score=best_score,
            routing_strategy=RoutingStrategy.CAPABILITY_MATCH,
            matched_capabilities=matched_caps,
            reasoning=f"Best capability match ({best_score:.2f})",
            alternatives=[(aid, score) for aid, score, _ in candidates[1:3]]
        )
    def _performance_based_routing(
        self,
        requirements: RequestRequirements
    ) -> Optional[RoutingDecision]:
        """Route based on historical performance"""
        candidates = []
        for agent_id, specialist in self.specialists.items():
            # Check if agent has required capabilities
            has_capabilities = all(
                cap in specialist.capabilities
                for cap in requirements.required_capabilities
            )
            if not has_capabilities:
                continue
            # Calculate performance score
            success_rate = (
                specialist.successful_requests / specialist.total_requests
                if specialist.total_requests > 0 else 0.5
            )
            # Penalize slow response
            speed_score = 1.0 / (1.0 + specialist.average_response_time)
            performance_score = (success_rate * 0.7) + (speed_score * 0.3)
            matched_caps = list(requirements.required_capabilities)
            candidates.append((agent_id, performance_score, matched_caps))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_agent_id, best_score, matched_caps = candidates[0]
        return RoutingDecision(
            selected_agent_id=best_agent_id,
            match_score=best_score,
            routing_strategy=RoutingStrategy.PERFORMANCE_BASED,
            matched_capabilities=matched_caps,
            reasoning=f"Best historical performance ({best_score:.2f})",
            alternatives=[(aid, score) for aid, score, _ in candidates[1:3]]
        )
    def _load_balanced_routing(
        self,
        requirements: RequestRequirements
    ) -> Optional[RoutingDecision]:
        """Route to least loaded capable agent"""
        candidates = []
        for agent_id, specialist in self.specialists.items():
            # Check capabilities
            has_capabilities = all(
                cap in specialist.capabilities
                for cap in requirements.required_capabilities
            )
            if not has_capabilities:
                continue
            # Calculate load score (lower is better)
            load_ratio = specialist.current_load / specialist.max_concurrent
            load_score = 1.0 - load_ratio
            matched_caps = list(requirements.required_capabilities)
            candidates.append((agent_id, load_score, matched_caps))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_agent_id, best_score, matched_caps = candidates[0]
        return RoutingDecision(
            selected_agent_id=best_agent_id,
            match_score=best_score,
            routing_strategy=RoutingStrategy.LOAD_BALANCED,
            matched_capabilities=matched_caps,
            reasoning=f"Least loaded agent ({best_score:.2f} capacity available)",
            alternatives=[(aid, score) for aid, score, _ in candidates[1:3]]
        )
    def _round_robin_routing(self) -> Optional[RoutingDecision]:
        """Simple round-robin routing"""
        if not self.specialists:
            return None
        agent_ids = list(self.specialists.keys())
        selected_agent_id = agent_ids[self.round_robin_index % len(agent_ids)]
        self.round_robin_index += 1
        specialist = self.specialists[selected_agent_id]
        matched_caps = list(specialist.capabilities.keys())
        return RoutingDecision(
            selected_agent_id=selected_agent_id,
            match_score=1.0,
            routing_strategy=RoutingStrategy.ROUND_ROBIN,
            matched_capabilities=matched_caps,
            reasoning="Round-robin selection"
        )
    def _specialized_first_routing(
        self,
        requirements: RequestRequirements
    ) -> Optional[RoutingDecision]:
        """Route to most specialized agent first"""
        candidates = []
        for agent_id, specialist in self.specialists.items():
            # Check for exact capability match
            has_required = all(
                cap in specialist.capabilities
                for cap in requirements.required_capabilities
            )
            if not has_required:
                continue
            # Calculate specialization score
            total_caps = len(specialist.capabilities)
            matched_caps = len(set(specialist.capabilities.keys()) & requirements.required_capabilities)
            # More specialized = fewer total capabilities but high match
            specialization_score = matched_caps / total_caps if total_caps > 0 else 0
            matched_cap_list = list(requirements.required_capabilities)
            candidates.append((agent_id, specialization_score, matched_cap_list))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_agent_id, best_score, matched_caps = candidates[0]
        return RoutingDecision(
            selected_agent_id=best_agent_id,
            match_score=best_score,
            routing_strategy=RoutingStrategy.SPECIALIZED_FIRST,
            matched_capabilities=matched_caps,
            reasoning=f"Most specialized agent ({best_score:.2f})",
            alternatives=[(aid, score) for aid, score, _ in candidates[1:3]]
        )
    def _hybrid_routing(
        self,
        requirements: RequestRequirements
    ) -> Optional[RoutingDecision]:
        """Hybrid routing combining multiple factors"""
        candidates = []
        for agent_id, specialist in self.specialists.items():
            # Capability match score
            capability_score = self._calculate_capability_match(
                specialist,
                requirements
            )
            if capability_score == 0:
                continue
            # Performance score
            success_rate = (
                specialist.successful_requests / specialist.total_requests
                if specialist.total_requests > 0 else 0.5
            )
            # Load score
            load_ratio = specialist.current_load / specialist.max_concurrent
            load_score = 1.0 - load_ratio
            # Combined score
            combined_score = (
                capability_score * 0.5 +
                success_rate * 0.3 +
                load_score * 0.2
            )
            matched_caps = self._get_matched_capabilities(specialist, requirements)
            candidates.append((agent_id, combined_score, matched_caps))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_agent_id, best_score, matched_caps = candidates[0]
        return RoutingDecision(
            selected_agent_id=best_agent_id,
            match_score=best_score,
            routing_strategy=RoutingStrategy.HYBRID,
            matched_capabilities=matched_caps,
            reasoning=f"Hybrid score: capability + performance + load ({best_score:.2f})",
            alternatives=[(aid, score) for aid, score, _ in candidates[1:3]]
        )
    def _calculate_capability_match(
        self,
        specialist: AgentSpecialization,
        requirements: RequestRequirements
    ) -> float:
        """Calculate how well specialist matches requirements"""
        if not requirements.required_capabilities:
            return 0.5
        matched = 0
        total_score = 0.0
        for req_cap in requirements.required_capabilities:
            if req_cap in specialist.capabilities:
                capability = specialist.capabilities[req_cap]
                # Level bonus
                level_score = capability.level.value / 3.0
                # Keyword match bonus
                keyword_match = len(
                    capability.keywords & requirements.keywords
                ) / max(len(requirements.keywords), 1)
                total_score += (level_score * 0.7 + keyword_match * 0.3)
                matched += 1
        # Penalty for missing required capabilities
        coverage = matched / len(requirements.required_capabilities)
        if matched == 0:
            return 0.0
        avg_score = total_score / matched
        return avg_score * coverage
    def _get_matched_capabilities(
        self,
        specialist: AgentSpecialization,
        requirements: RequestRequirements
    ) -> List[str]:
        """Get list of matched capabilities"""
        return [
            cap for cap in requirements.required_capabilities
            if cap in specialist.capabilities
        ]
    def _execute_with_agent(
        self,
        agent_id: str,
        task: str,
        requirements: RequestRequirements
    ) -> Dict[str, Any]:
        """Execute task with selected agent"""
        specialist = self.specialists[agent_id]
        # Increment load
        specialist.current_load += 1
        start_time = time.time()
        try:
            # Execute with agent
            if hasattr(specialist.agent, 'execute'):
                output = specialist.agent.execute(task)
            elif callable(specialist.agent):
                output = specialist.agent(task)
            else:
                output = f"Agent {agent_id} processed: {task}"
            execution_time = time.time() - start_time
            return {
                'success': True,
                'output': output,
                'execution_time': execution_time
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
        finally:
            # Decrement load
            specialist.current_load = max(0, specialist.current_load - 1)
    def _update_performance(
        self,
        agent_id: str,
        success: bool,
        execution_time: float
    ):
        """Update agent performance metrics"""
        specialist = self.specialists[agent_id]
        specialist.total_requests += 1
        if success:
            specialist.successful_requests += 1
        # Update average response time
        if specialist.average_response_time == 0:
            specialist.average_response_time = execution_time
        else:
            # Moving average
            specialist.average_response_time = (
                specialist.average_response_time * 0.9 +
                execution_time * 0.1
            )
    def get_specialist_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics for a specialist"""
        if agent_id not in self.specialists:
            return {}
        specialist = self.specialists[agent_id]
        return {
            'agent_id': agent_id,
            'capabilities': list(specialist.capabilities.keys()),
            'total_requests': specialist.total_requests,
            'successful_requests': specialist.successful_requests,
            'success_rate': (
                specialist.successful_requests / specialist.total_requests
                if specialist.total_requests > 0 else 0
            ),
            'average_response_time': specialist.average_response_time,
            'current_load': specialist.current_load,
            'max_concurrent': specialist.max_concurrent,
            'load_percentage': (specialist.current_load / specialist.max_concurrent * 100)
        }
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        if not self.routing_history:
            return {'message': 'No routing history'}
        strategy_counts = defaultdict(int)
        agent_usage = defaultdict(int)
        for decision in self.routing_history:
            strategy_counts[decision.routing_strategy.value] += 1
            agent_usage[decision.selected_agent_id] += 1
        return {
            'total_routes': len(self.routing_history),
            'strategy_distribution': dict(strategy_counts),
            'agent_usage': dict(agent_usage),
            'total_specialists': len(self.specialists)
        }
class SpecializedAgent:
    """Example specialized agent"""
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
    def execute(self, task: str) -> str:
        """Execute task with specialization"""
        return f"[{self.name} - {self.specialty}]: Completed '{task}'"
def main():
    """Demonstrate agent specialization and routing"""
    print("=" * 60)
    print("Agent Specialization & Routing Demonstration")
    print("=" * 60)
    router = AgentRouter()
    print("\n1. Registering Specialized Agents")
    print("-" * 60)
    # Register various specialists
    specialists = [
        {
            'agent': SpecializedAgent("PythonExpert", "Python Development"),
            'id': "python_expert",
            'capabilities': {
                'python': {
                    'level': CapabilityLevel.EXPERT,
                    'keywords': {'python', 'code', 'programming', 'pandas', 'numpy'}
                },
                'data_analysis': {
                    'level': CapabilityLevel.INTERMEDIATE,
                    'keywords': {'data', 'analyze', 'statistics'}
                }
            }
        },
        {
            'agent': SpecializedAgent("DataScientist", "Data Science"),
            'id': "data_scientist",
            'capabilities': {
                'data_analysis': {
                    'level': CapabilityLevel.EXPERT,
                    'keywords': {'data', 'analyze', 'statistics', 'visualization'}
                },
                'ml': {
                    'level': CapabilityLevel.EXPERT,
                    'keywords': {'machine learning', 'model', 'train', 'predict'}
                },
                'python': {
                    'level': CapabilityLevel.INTERMEDIATE,
                    'keywords': {'python', 'code'}
                }
            }
        },
        {
            'agent': SpecializedAgent("WebDeveloper", "Web Development"),
            'id': "web_developer",
            'capabilities': {
                'web': {
                    'level': CapabilityLevel.EXPERT,
                    'keywords': {'web', 'http', 'api', 'server', 'frontend'}
                },
                'database': {
                    'level': CapabilityLevel.INTERMEDIATE,
                    'keywords': {'database', 'sql', 'query'}
                }
            }
        },
        {
            'agent': SpecializedAgent("Generalist", "General Tasks"),
            'id': "generalist",
            'capabilities': {
                'python': {
                    'level': CapabilityLevel.BEGINNER,
                    'keywords': {'python'}
                },
                'web': {
                    'level': CapabilityLevel.BEGINNER,
                    'keywords': {'web'}
                },
                'data_analysis': {
                    'level': CapabilityLevel.BEGINNER,
                    'keywords': {'data'}
                }
            }
        }
    ]
    for spec in specialists:
        router.register_specialist(
            agent=spec['agent'],
            agent_id=spec['id'],
            capabilities=spec['capabilities']
        )
        caps = ', '.join(spec['capabilities'].keys())
        print(f"Registered: {spec['id']} - Capabilities: {caps}")
    print("\n" + "=" * 60)
    print("2. Capability-Based Routing")
    print("=" * 60)
    tasks = [
        "Write Python code to analyze data",
        "Build a web API endpoint",
        "Train a machine learning model",
        "Optimize database queries"
    ]
    for task in tasks:
        result = router.route_request(
            task=task,
            strategy=RoutingStrategy.CAPABILITY_MATCH
        )
        print(f"\nTask: '{task}'")
        print(f"  Routed to: {result['agent_id']}")
        print(f"  Match score: {result['routing_decision'].match_score:.3f}")
        print(f"  Matched capabilities: {result['routing_decision'].matched_capabilities}")
        print(f"  Reasoning: {result['routing_decision'].reasoning}")
        if result['routing_decision'].alternatives:
            print(f"  Alternatives:")
            for alt_id, alt_score in result['routing_decision'].alternatives:
                print(f"    - {alt_id}: {alt_score:.3f}")
    print("\n" + "=" * 60)
    print("3. Performance-Based Routing")
    print("=" * 60)
    # Execute multiple requests to build performance history
    print("\nBuilding performance history...")
    for _ in range(10):
        router.route_request(
            "Analyze data with Python",
            strategy=RoutingStrategy.CAPABILITY_MATCH
        )
    # Now use performance-based routing
    result = router.route_request(
        "Analyze data with Python",
        strategy=RoutingStrategy.PERFORMANCE_BASED
    )
    print(f"\nPerformance-based routing:")
    print(f"  Selected: {result['agent_id']}")
    print(f"  Score: {result['routing_decision'].match_score:.3f}")
    print(f"  Reasoning: {result['routing_decision'].reasoning}")
    print("\n" + "=" * 60)
    print("4. Hybrid Routing")
    print("=" * 60)
    result = router.route_request(
        "Build ML model with Python",
        strategy=RoutingStrategy.HYBRID
    )
    print(f"\nHybrid routing (capability + performance + load):")
    print(f"  Selected: {result['agent_id']}")
    print(f"  Score: {result['routing_decision'].match_score:.3f}")
    print(f"  Reasoning: {result['routing_decision'].reasoning}")
    print(f"  Matched capabilities: {result['routing_decision'].matched_capabilities}")
    print("\n" + "=" * 60)
    print("5. Specialized-First Routing")
    print("=" * 60)
    result = router.route_request(
        "Machine learning task",
        strategy=RoutingStrategy.SPECIALIZED_FIRST
    )
    print(f"\nSpecialized-first routing:")
    print(f"  Selected: {result['agent_id']}")
    print(f"  Specialization score: {result['routing_decision'].match_score:.3f}")
    print(f"  Reasoning: {result['routing_decision'].reasoning}")
    print("\n" + "=" * 60)
    print("6. Specialist Performance Stats")
    print("=" * 60)
    for agent_id in router.specialists.keys():
        stats = router.get_specialist_stats(agent_id)
        print(f"\n{stats['agent_id']}:")
        print(f"  Capabilities: {', '.join(stats['capabilities'])}")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Avg response time: {stats['average_response_time']:.4f}s")
        print(f"  Current load: {stats['current_load']}/{stats['max_concurrent']} ({stats['load_percentage']:.1f}%)")
    print("\n" + "=" * 60)
    print("7. Routing Statistics")
    print("=" * 60)
    stats = router.get_routing_statistics()
    print(f"\nTotal routes: {stats['total_routes']}")
    print(f"Total specialists: {stats['total_specialists']}")
    print("\nStrategy distribution:")
    for strategy, count in stats['strategy_distribution'].items():
        percentage = count / stats['total_routes'] * 100
        print(f"  {strategy}: {count} ({percentage:.1f}%)")
    print("\nAgent usage:")
    for agent_id, count in sorted(stats['agent_usage'].items(), key=lambda x: x[1], reverse=True):
        percentage = count / stats['total_routes'] * 100
        print(f"  {agent_id}: {count} ({percentage:.1f}%)")
    print("\n" + "=" * 60)
    print("8. Load Balancing Test")
    print("=" * 60)
    # Simulate high load
    print("\nSimulating concurrent requests...")
    for i in range(15):
        result = router.route_request(
            f"Python task {i}",
            strategy=RoutingStrategy.LOAD_BALANCED
        )
        print(f"  Request {i+1} -> {result['agent_id']}")
    print("\nLoad distribution:")
    for agent_id in router.specialists.keys():
        stats = router.get_specialist_stats(agent_id)
        print(f"  {agent_id}: {stats['load_percentage']:.1f}% loaded")
    print("\n" + "=" * 60)
    print("Agent Specialization & Routing demonstration complete!")
    print("=" * 60)
if __name__ == "__main__":
    main()
