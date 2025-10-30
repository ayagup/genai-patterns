"""
Agent Capability Registry Pattern

Maintains a registry of agent capabilities and skills.
Enables capability-based agent selection and composition.

Use Cases:
- Agent discovery
- Skill matching
- Capability-based routing
- Dynamic composition

Advantages:
- Dynamic capability discovery
- Skill-based selection
- Extensibility
- Loose coupling
"""

from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class CapabilityType(Enum):
    """Types of capabilities"""
    SKILL = "skill"
    SERVICE = "service"
    RESOURCE = "resource"
    KNOWLEDGE = "knowledge"
    TOOL = "tool"


class ProficiencyLevel(Enum):
    """Proficiency levels"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5


@dataclass
class Capability:
    """Agent capability definition"""
    capability_id: str
    name: str
    capability_type: CapabilityType
    description: str
    version: str = "1.0.0"
    parameters: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCapability:
    """Agent's implementation of a capability"""
    agent_id: str
    capability_id: str
    proficiency: ProficiencyLevel
    certified: bool = False
    available: bool = True
    performance_score: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityRequirement:
    """Requirement for a capability"""
    capability_id: str
    min_proficiency: ProficiencyLevel = ProficiencyLevel.BEGINNER
    required: bool = True
    preferred_tags: List[str] = field(default_factory=list)


@dataclass
class CapabilityMatch:
    """Match between requirement and agent capability"""
    agent_id: str
    capability_id: str
    match_score: float
    proficiency: ProficiencyLevel
    reasons: List[str] = field(default_factory=list)


class CapabilityValidator:
    """Validates capability implementations"""
    
    def validate_capability(self,
                          capability: Capability,
                          implementation: Callable) -> bool:
        """
        Validate capability implementation.
        
        Args:
            capability: Capability definition
            implementation: Implementation function
            
        Returns:
            Whether implementation is valid
        """
        import inspect
        
        # Check if callable
        if not callable(implementation):
            return False
        
        # Check parameters
        sig = inspect.signature(implementation)
        params = list(sig.parameters.keys())
        
        # Verify required parameters present
        for required_param in capability.parameters:
            if required_param not in params:
                return False
        
        return True
    
    def certify_capability(self,
                          agent_capability: AgentCapability,
                          test_cases: List[Dict[str, Any]]) -> bool:
        """
        Certify agent capability through testing.
        
        Args:
            agent_capability: Agent capability to certify
            test_cases: Test cases to run
            
        Returns:
            Whether certification passed
        """
        # Simplified certification logic
        # In practice, would run actual tests
        
        if not test_cases:
            return False
        
        # Assume tests pass if proficiency is advanced or higher
        return agent_capability.proficiency.value >= ProficiencyLevel.ADVANCED.value


class CapabilityMatcher:
    """Matches requirements to agent capabilities"""
    
    def match_capability(self,
                        requirement: CapabilityRequirement,
                        agent_capability: AgentCapability,
                        capability_def: Capability) -> CapabilityMatch:
        """
        Calculate match score between requirement and capability.
        
        Args:
            requirement: Capability requirement
            agent_capability: Agent's capability
            capability_def: Capability definition
            
        Returns:
            Match result
        """
        reasons = []
        score = 0.0
        
        # Base score for having the capability
        score += 40.0
        reasons.append("Has required capability")
        
        # Proficiency match
        if agent_capability.proficiency.value >= requirement.min_proficiency.value:
            proficiency_bonus = (
                agent_capability.proficiency.value - requirement.min_proficiency.value
            ) * 10
            score += min(proficiency_bonus, 30.0)
            reasons.append("Meets proficiency requirement")
        else:
            score -= 20.0
            reasons.append("Below required proficiency")
        
        # Certification bonus
        if agent_capability.certified:
            score += 15.0
            reasons.append("Certified capability")
        
        # Availability
        if agent_capability.available:
            score += 5.0
        else:
            score -= 30.0
            reasons.append("Currently unavailable")
        
        # Performance score
        score += agent_capability.performance_score * 0.1
        
        # Tag matching
        if requirement.preferred_tags:
            matching_tags = set(requirement.preferred_tags) & set(capability_def.tags)
            tag_bonus = (len(matching_tags) / len(requirement.preferred_tags)) * 10
            score += tag_bonus
            
            if matching_tags:
                reasons.append("Matches preferred tags: {}".format(
                    ", ".join(matching_tags)
                ))
        
        # Normalize to 0-100
        score = max(0.0, min(100.0, score))
        
        return CapabilityMatch(
            agent_id=agent_capability.agent_id,
            capability_id=agent_capability.capability_id,
            match_score=score,
            proficiency=agent_capability.proficiency,
            reasons=reasons
        )
    
    def rank_agents(self,
                   requirements: List[CapabilityRequirement],
                   agent_capabilities: Dict[str, List[AgentCapability]],
                   capability_defs: Dict[str, Capability]) -> List[Tuple[str, float, List[CapabilityMatch]]]:
        """
        Rank agents by how well they match requirements.
        
        Args:
            requirements: List of capability requirements
            agent_capabilities: Agent capabilities by agent ID
            capability_defs: Capability definitions
            
        Returns:
            List of (agent_id, score, matches) tuples, sorted by score
        """
        agent_scores: Dict[str, List[CapabilityMatch]] = {}
        
        # For each agent
        for agent_id, capabilities in agent_capabilities.items():
            matches = []
            
            # Check each requirement
            for req in requirements:
                # Find agent's capability for this requirement
                agent_cap = next(
                    (c for c in capabilities if c.capability_id == req.capability_id),
                    None
                )
                
                if agent_cap:
                    cap_def = capability_defs.get(req.capability_id)
                    
                    if cap_def:
                        match = self.match_capability(req, agent_cap, cap_def)
                        matches.append(match)
                elif req.required:
                    # Missing required capability
                    matches.append(CapabilityMatch(
                        agent_id=agent_id,
                        capability_id=req.capability_id,
                        match_score=0.0,
                        proficiency=ProficiencyLevel.BEGINNER,
                        reasons=["Missing required capability"]
                    ))
            
            if matches:
                agent_scores[agent_id] = matches
        
        # Calculate overall scores
        results = []
        
        for agent_id, matches in agent_scores.items():
            # Calculate weighted average
            # Required capabilities have higher weight
            total_score = 0.0
            total_weight = 0.0
            
            for i, match in enumerate(matches):
                req = requirements[i]
                weight = 2.0 if req.required else 1.0
                
                total_score += match.match_score * weight
                total_weight += weight
            
            avg_score = total_score / total_weight if total_weight > 0 else 0.0
            
            results.append((agent_id, avg_score, matches))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results


class AgentCapabilityRegistry:
    """
    Central registry for agent capabilities.
    Manages capability definitions, agent registrations, and matching.
    """
    
    def __init__(self):
        # Capability definitions
        self.capabilities: Dict[str, Capability] = {}
        
        # Agent capabilities
        self.agent_capabilities: Dict[str, List[AgentCapability]] = {}
        
        # Components
        self.validator = CapabilityValidator()
        self.matcher = CapabilityMatcher()
        
        # Indexes
        self.capability_by_type: Dict[CapabilityType, Set[str]] = {
            ct: set() for ct in CapabilityType
        }
        self.capability_by_tag: Dict[str, Set[str]] = {}
    
    def register_capability(self, capability: Capability) -> None:
        """
        Register a capability definition.
        
        Args:
            capability: Capability to register
        """
        self.capabilities[capability.capability_id] = capability
        
        # Update indexes
        self.capability_by_type[capability.capability_type].add(
            capability.capability_id
        )
        
        for tag in capability.tags:
            if tag not in self.capability_by_tag:
                self.capability_by_tag[tag] = set()
            self.capability_by_tag[tag].add(capability.capability_id)
    
    def register_agent_capability(self,
                                 agent_id: str,
                                 capability_id: str,
                                 proficiency: ProficiencyLevel,
                                 certified: bool = False) -> bool:
        """
        Register agent's capability.
        
        Args:
            agent_id: Agent identifier
            capability_id: Capability identifier
            proficiency: Proficiency level
            certified: Whether certified
            
        Returns:
            Whether registration succeeded
        """
        # Check capability exists
        if capability_id not in self.capabilities:
            return False
        
        # Create agent capability
        agent_cap = AgentCapability(
            agent_id=agent_id,
            capability_id=capability_id,
            proficiency=proficiency,
            certified=certified
        )
        
        # Add to registry
        if agent_id not in self.agent_capabilities:
            self.agent_capabilities[agent_id] = []
        
        # Check if already registered
        existing = next(
            (c for c in self.agent_capabilities[agent_id]
             if c.capability_id == capability_id),
            None
        )
        
        if existing:
            # Update existing
            existing.proficiency = proficiency
            existing.certified = certified
        else:
            # Add new
            self.agent_capabilities[agent_id].append(agent_cap)
        
        return True
    
    def unregister_agent_capability(self,
                                   agent_id: str,
                                   capability_id: str) -> bool:
        """Unregister agent capability"""
        if agent_id not in self.agent_capabilities:
            return False
        
        self.agent_capabilities[agent_id] = [
            c for c in self.agent_capabilities[agent_id]
            if c.capability_id != capability_id
        ]
        
        return True
    
    def get_capability(self, capability_id: str) -> Optional[Capability]:
        """Get capability definition"""
        return self.capabilities.get(capability_id)
    
    def get_agent_capabilities(self, agent_id: str) -> List[AgentCapability]:
        """Get all capabilities for agent"""
        return self.agent_capabilities.get(agent_id, [])
    
    def find_capabilities(self,
                         capability_type: Optional[CapabilityType] = None,
                         tags: Optional[List[str]] = None,
                         name_pattern: Optional[str] = None) -> List[Capability]:
        """
        Find capabilities by criteria.
        
        Args:
            capability_type: Optional type filter
            tags: Optional tag filters
            name_pattern: Optional name pattern
            
        Returns:
            Matching capabilities
        """
        results = set(self.capabilities.keys())
        
        # Filter by type
        if capability_type:
            type_caps = self.capability_by_type.get(capability_type, set())
            results &= type_caps
        
        # Filter by tags
        if tags:
            for tag in tags:
                tag_caps = self.capability_by_tag.get(tag, set())
                results &= tag_caps
        
        # Get capability objects
        capabilities = [
            self.capabilities[cid] for cid in results
            if cid in self.capabilities
        ]
        
        # Filter by name pattern
        if name_pattern:
            pattern = name_pattern.lower()
            capabilities = [
                c for c in capabilities
                if pattern in c.name.lower()
            ]
        
        return capabilities
    
    def find_agents_with_capability(self,
                                   capability_id: str,
                                   min_proficiency: ProficiencyLevel = ProficiencyLevel.BEGINNER,
                                   available_only: bool = True) -> List[AgentCapability]:
        """
        Find agents with specific capability.
        
        Args:
            capability_id: Capability to find
            min_proficiency: Minimum proficiency level
            available_only: Only return available agents
            
        Returns:
            List of agent capabilities
        """
        results = []
        
        for agent_id, capabilities in self.agent_capabilities.items():
            for cap in capabilities:
                if cap.capability_id != capability_id:
                    continue
                
                if cap.proficiency.value < min_proficiency.value:
                    continue
                
                if available_only and not cap.available:
                    continue
                
                results.append(cap)
        
        return results
    
    def match_requirements(self,
                          requirements: List[CapabilityRequirement],
                          limit: int = 10) -> List[Tuple[str, float, List[CapabilityMatch]]]:
        """
        Find agents matching requirements.
        
        Args:
            requirements: List of requirements
            limit: Maximum results
            
        Returns:
            Ranked list of agents with match details
        """
        ranked = self.matcher.rank_agents(
            requirements,
            self.agent_capabilities,
            self.capabilities
        )
        
        return ranked[:limit]
    
    def update_capability_stats(self,
                               agent_id: str,
                               capability_id: str,
                               performance_score: Optional[float] = None,
                               increment_usage: bool = False) -> bool:
        """
        Update capability statistics.
        
        Args:
            agent_id: Agent identifier
            capability_id: Capability identifier
            performance_score: Optional new performance score
            increment_usage: Whether to increment usage count
            
        Returns:
            Whether update succeeded
        """
        capabilities = self.agent_capabilities.get(agent_id, [])
        
        for cap in capabilities:
            if cap.capability_id == capability_id:
                if performance_score is not None:
                    cap.performance_score = performance_score
                
                if increment_usage:
                    cap.usage_count += 1
                    cap.last_used = datetime.now()
                
                return True
        
        return False
    
    def get_capability_coverage(self) -> Dict[str, Any]:
        """Get capability coverage statistics"""
        total_capabilities = len(self.capabilities)
        total_agents = len(self.agent_capabilities)
        
        # Count agents per capability
        coverage = {}
        for cap_id in self.capabilities.keys():
            agents = self.find_agents_with_capability(cap_id, available_only=False)
            coverage[cap_id] = len(agents)
        
        # Find gaps
        gaps = [cap_id for cap_id, count in coverage.items() if count == 0]
        
        return {
            "total_capabilities": total_capabilities,
            "total_agents": total_agents,
            "coverage": coverage,
            "gaps": gaps,
            "average_agents_per_capability": (
                sum(coverage.values()) / len(coverage)
                if coverage else 0
            )
        }
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry as JSON-serializable dict"""
        return {
            "capabilities": [
                {
                    "id": c.capability_id,
                    "name": c.name,
                    "type": c.capability_type.value,
                    "description": c.description,
                    "version": c.version,
                    "tags": c.tags
                }
                for c in self.capabilities.values()
            ],
            "agent_capabilities": {
                agent_id: [
                    {
                        "capability_id": ac.capability_id,
                        "proficiency": ac.proficiency.value,
                        "certified": ac.certified,
                        "available": ac.available,
                        "performance_score": ac.performance_score,
                        "usage_count": ac.usage_count
                    }
                    for ac in capabilities
                ]
                for agent_id, capabilities in self.agent_capabilities.items()
            }
        }


def demonstrate_capability_registry():
    """Demonstrate agent capability registry"""
    print("=" * 70)
    print("Agent Capability Registry Demonstration")
    print("=" * 70)
    
    registry = AgentCapabilityRegistry()
    
    # Example 1: Register capabilities
    print("\n1. Registering Capabilities:")
    
    capabilities = [
        Capability(
            "nlp_processing",
            "Natural Language Processing",
            CapabilityType.SKILL,
            "Process and understand natural language",
            tags=["nlp", "text", "ai"]
        ),
        Capability(
            "data_analysis",
            "Data Analysis",
            CapabilityType.SKILL,
            "Analyze and visualize data",
            tags=["data", "analytics", "statistics"]
        ),
        Capability(
            "web_scraping",
            "Web Scraping",
            CapabilityType.TOOL,
            "Extract data from websites",
            tags=["web", "data", "extraction"]
        ),
        Capability(
            "api_integration",
            "API Integration",
            CapabilityType.SERVICE,
            "Integrate with external APIs",
            tags=["api", "integration", "service"]
        ),
        Capability(
            "machine_learning",
            "Machine Learning",
            CapabilityType.SKILL,
            "Train and deploy ML models",
            tags=["ml", "ai", "modeling"]
        )
    ]
    
    for cap in capabilities:
        registry.register_capability(cap)
        print("  Registered: {}".format(cap.name))
    
    # Example 2: Register agent capabilities
    print("\n2. Registering Agent Capabilities:")
    
    agents = [
        ("agent_1", [
            ("nlp_processing", ProficiencyLevel.EXPERT, True),
            ("data_analysis", ProficiencyLevel.ADVANCED, True),
            ("machine_learning", ProficiencyLevel.EXPERT, True)
        ]),
        ("agent_2", [
            ("web_scraping", ProficiencyLevel.ADVANCED, True),
            ("api_integration", ProficiencyLevel.EXPERT, True),
            ("data_analysis", ProficiencyLevel.INTERMEDIATE, False)
        ]),
        ("agent_3", [
            ("nlp_processing", ProficiencyLevel.INTERMEDIATE, False),
            ("api_integration", ProficiencyLevel.ADVANCED, True),
            ("machine_learning", ProficiencyLevel.BEGINNER, False)
        ])
    ]
    
    for agent_id, caps in agents:
        for cap_id, prof, cert in caps:
            registry.register_agent_capability(agent_id, cap_id, prof, cert)
        print("  Registered {} with {} capabilities".format(agent_id, len(caps)))
    
    # Example 3: Find capabilities
    print("\n3. Finding Capabilities:")
    
    ai_caps = registry.find_capabilities(tags=["ai"])
    print("  AI-related capabilities: {}".format(len(ai_caps)))
    
    for cap in ai_caps:
        print("    - {}".format(cap.name))
    
    # Example 4: Find agents with capability
    print("\n4. Finding Agents with NLP Capability:")
    
    nlp_agents = registry.find_agents_with_capability(
        "nlp_processing",
        min_proficiency=ProficiencyLevel.ADVANCED
    )
    
    print("  Found {} agents with advanced NLP:".format(len(nlp_agents)))
    
    for agent_cap in nlp_agents:
        print("    - {}: {} (certified: {})".format(
            agent_cap.agent_id,
            agent_cap.proficiency.name,
            agent_cap.certified
        ))
    
    # Example 5: Match requirements
    print("\n5. Matching Requirements:")
    
    requirements = [
        CapabilityRequirement(
            "nlp_processing",
            min_proficiency=ProficiencyLevel.ADVANCED,
            required=True
        ),
        CapabilityRequirement(
            "machine_learning",
            min_proficiency=ProficiencyLevel.INTERMEDIATE,
            required=True
        ),
        CapabilityRequirement(
            "data_analysis",
            min_proficiency=ProficiencyLevel.BEGINNER,
            required=False
        )
    ]
    
    matches = registry.match_requirements(requirements)
    
    print("  Top matching agents:")
    
    for agent_id, score, capability_matches in matches:
        print("\n    {} (score: {:.1f})".format(agent_id, score))
        for match in capability_matches:
            print("      - {}: {:.1f} ({})".format(
                match.capability_id,
                match.match_score,
                match.proficiency.name
            ))
    
    # Example 6: Update statistics
    print("\n6. Updating Capability Statistics:")
    
    registry.update_capability_stats(
        "agent_1",
        "nlp_processing",
        performance_score=95.5,
        increment_usage=True
    )
    
    agent1_caps = registry.get_agent_capabilities("agent_1")
    nlp_cap = next(c for c in agent1_caps if c.capability_id == "nlp_processing")
    
    print("  Updated agent_1 NLP stats:")
    print("    Performance: {:.1f}".format(nlp_cap.performance_score))
    print("    Usage count: {}".format(nlp_cap.usage_count))
    
    # Example 7: Capability coverage
    print("\n7. Capability Coverage Analysis:")
    
    coverage = registry.get_capability_coverage()
    
    print("  Total capabilities: {}".format(coverage["total_capabilities"]))
    print("  Total agents: {}".format(coverage["total_agents"]))
    print("  Average agents per capability: {:.1f}".format(
        coverage["average_agents_per_capability"]
    ))
    
    if coverage["gaps"]:
        print("  Capability gaps:")
        for gap in coverage["gaps"]:
            print("    - {}".format(gap))
    
    # Example 8: Find by type
    print("\n8. Capabilities by Type:")
    
    skill_caps = registry.find_capabilities(capability_type=CapabilityType.SKILL)
    
    print("  SKILL capabilities:")
    for cap in skill_caps:
        print("    - {}".format(cap.name))
    
    # Example 9: Complex matching
    print("\n9. Complex Requirement Matching:")
    
    complex_requirements = [
        CapabilityRequirement(
            "api_integration",
            min_proficiency=ProficiencyLevel.ADVANCED,
            required=True,
            preferred_tags=["integration", "service"]
        ),
        CapabilityRequirement(
            "data_analysis",
            min_proficiency=ProficiencyLevel.INTERMEDIATE,
            required=False
        )
    ]
    
    complex_matches = registry.match_requirements(complex_requirements)
    
    print("  Best match: {}".format(complex_matches[0][0]))
    print("  Score: {:.1f}".format(complex_matches[0][1]))
    print("  Reasons:")
    
    for match in complex_matches[0][2]:
        for reason in match.reasons:
            print("    - {}".format(reason))
    
    # Example 10: Export registry
    print("\n10. Exporting Registry:")
    
    export = registry.export_registry()
    print(json.dumps(export, indent=2))


if __name__ == "__main__":
    demonstrate_capability_registry()
