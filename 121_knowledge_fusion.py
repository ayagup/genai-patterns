"""
Pattern 121: Knowledge Fusion Agent

This pattern implements multi-source knowledge integration with conflict resolution,
consensus building, and provenance tracking.

Use Cases:
- Database integration and consolidation
- Multi-source information synthesis
- Knowledge base merging
- Collaborative knowledge construction
- Consensus-based decision making

Category: Knowledge Management (2/4 = 50%)
Complexity: Advanced
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import math


class ConflictType(Enum):
    """Types of knowledge conflicts."""
    VALUE_CONFLICT = "value_conflict"  # Different values for same attribute
    TYPE_CONFLICT = "type_conflict"  # Different data types
    CARDINALITY_CONFLICT = "cardinality_conflict"  # Single vs multiple values
    SEMANTIC_CONFLICT = "semantic_conflict"  # Different meanings
    TEMPORAL_CONFLICT = "temporal_conflict"  # Different time validity
    STRUCTURAL_CONFLICT = "structural_conflict"  # Different structures


class ResolutionStrategy(Enum):
    """Conflict resolution strategies."""
    MOST_RECENT = "most_recent"  # Use most recent information
    MOST_TRUSTED = "most_trusted"  # Use most trusted source
    MAJORITY_VOTE = "majority_vote"  # Use majority consensus
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted combination
    UNION = "union"  # Combine all values
    INTERSECTION = "intersection"  # Keep common values only
    MANUAL = "manual"  # Require human intervention


@dataclass
class Source:
    """Information source with credibility metadata."""
    source_id: str
    name: str
    credibility: float = 1.0  # 0.0 to 1.0
    reliability: float = 1.0  # Historical accuracy
    last_updated: datetime = field(default_factory=datetime.now)
    domain_expertise: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        """Make hashable using source_id."""
        return hash(self.source_id)
    
    def __eq__(self, other):
        """Compare by source_id."""
        if not isinstance(other, Source):
            return False
        return self.source_id == other.source_id


@dataclass
class KnowledgeItem:
    """Single piece of knowledge with provenance."""
    entity: str
    attribute: str
    value: Any
    source: Source
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_weight(self) -> float:
        """Calculate overall weight for fusion."""
        return (
            self.source.credibility * 0.4 +
            self.source.reliability * 0.3 +
            self.confidence * 0.3
        )


@dataclass
class Conflict:
    """Knowledge conflict between sources."""
    conflict_type: ConflictType
    entity: str
    attribute: str
    items: List[KnowledgeItem]
    severity: float = 0.5  # 0.0 to 1.0
    resolution: Optional[Any] = None
    strategy_used: Optional[ResolutionStrategy] = None


@dataclass
class FusedKnowledge:
    """Result of knowledge fusion."""
    entity: str
    attribute: str
    value: Any
    confidence: float
    sources: List[Source]
    conflicts_resolved: List[Conflict]
    provenance: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class SourceManager:
    """Manages information sources and credibility."""
    
    def __init__(self):
        self.sources: Dict[str, Source] = {}
        self.performance_history: Dict[str, List[float]] = {}
    
    def add_source(self, source: Source) -> None:
        """Add or update a source."""
        self.sources[source.source_id] = source
        if source.source_id not in self.performance_history:
            self.performance_history[source.source_id] = []
    
    def update_reliability(self, source_id: str, accuracy: float) -> None:
        """Update source reliability based on feedback."""
        if source_id in self.sources:
            history = self.performance_history[source_id]
            history.append(accuracy)
            
            # Calculate moving average
            window = min(10, len(history))
            recent = history[-window:]
            self.sources[source_id].reliability = sum(recent) / len(recent)
    
    def get_source(self, source_id: str) -> Optional[Source]:
        """Get source by ID."""
        return self.sources.get(source_id)
    
    def rank_sources(self, domain: Optional[str] = None) -> List[Source]:
        """Rank sources by credibility and reliability."""
        sources = list(self.sources.values())
        
        if domain:
            # Boost sources with domain expertise
            sources = [
                s for s in sources
                if not s.domain_expertise or domain in s.domain_expertise
            ]
        
        return sorted(
            sources,
            key=lambda s: s.credibility * s.reliability,
            reverse=True
        )


class ConflictDetector:
    """Detects conflicts between knowledge items."""
    
    def detect_conflicts(
        self,
        items: List[KnowledgeItem]
    ) -> List[Conflict]:
        """Detect conflicts among knowledge items."""
        conflicts = []
        
        # Group by entity and attribute
        groups: Dict[Tuple[str, str], List[KnowledgeItem]] = {}
        for item in items:
            key = (item.entity, item.attribute)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        # Check each group for conflicts
        for (entity, attribute), group_items in groups.items():
            if len(group_items) < 2:
                continue
            
            conflict = self._check_group_conflict(entity, attribute, group_items)
            if conflict:
                conflicts.append(conflict)
        
        return conflicts
    
    def _check_group_conflict(
        self,
        entity: str,
        attribute: str,
        items: List[KnowledgeItem]
    ) -> Optional[Conflict]:
        """Check if group has conflicts."""
        # Get unique values
        values = [item.value for item in items]
        unique_values = set(str(v) for v in values)
        
        if len(unique_values) == 1:
            return None  # No conflict
        
        # Determine conflict type
        conflict_type = self._determine_conflict_type(items)
        
        # Calculate severity
        severity = self._calculate_severity(items)
        
        return Conflict(
            conflict_type=conflict_type,
            entity=entity,
            attribute=attribute,
            items=items,
            severity=severity
        )
    
    def _determine_conflict_type(self, items: List[KnowledgeItem]) -> ConflictType:
        """Determine the type of conflict."""
        types = set(type(item.value).__name__ for item in items)
        
        if len(types) > 1:
            return ConflictType.TYPE_CONFLICT
        
        # Check temporal differences
        timestamps = [item.timestamp for item in items]
        time_range = (max(timestamps) - min(timestamps)).days
        if time_range > 30:
            return ConflictType.TEMPORAL_CONFLICT
        
        # Check if values are numeric (could be averaged)
        if all(isinstance(item.value, (int, float)) for item in items):
            return ConflictType.VALUE_CONFLICT
        
        return ConflictType.SEMANTIC_CONFLICT
    
    def _calculate_severity(self, items: List[KnowledgeItem]) -> float:
        """Calculate conflict severity."""
        # Consider weight differences
        weights = [item.get_weight() for item in items]
        weight_variance = sum((w - sum(weights)/len(weights))**2 for w in weights) / len(weights)
        
        # Consider value differences (for numeric values)
        if all(isinstance(item.value, (int, float)) for item in items):
            values = [float(item.value) for item in items]
            mean = sum(values) / len(values)
            variance = sum((v - mean)**2 for v in values) / len(values)
            normalized_variance = variance / (mean**2 + 1e-10)
            return min(1.0, normalized_variance + weight_variance)
        
        # For non-numeric, severity based on number of conflicting sources
        return min(1.0, len(items) / 10.0 + weight_variance)


class ConflictResolver:
    """Resolves conflicts between knowledge items."""
    
    def __init__(self):
        self.resolution_history: List[Conflict] = []
    
    def resolve_conflict(
        self,
        conflict: Conflict,
        strategy: Optional[ResolutionStrategy] = None
    ) -> Any:
        """Resolve a conflict using specified strategy."""
        if not strategy:
            strategy = self._choose_strategy(conflict)
        
        resolution = None
        
        if strategy == ResolutionStrategy.MOST_RECENT:
            resolution = self._resolve_most_recent(conflict)
        elif strategy == ResolutionStrategy.MOST_TRUSTED:
            resolution = self._resolve_most_trusted(conflict)
        elif strategy == ResolutionStrategy.MAJORITY_VOTE:
            resolution = self._resolve_majority_vote(conflict)
        elif strategy == ResolutionStrategy.WEIGHTED_AVERAGE:
            resolution = self._resolve_weighted_average(conflict)
        elif strategy == ResolutionStrategy.UNION:
            resolution = self._resolve_union(conflict)
        elif strategy == ResolutionStrategy.INTERSECTION:
            resolution = self._resolve_intersection(conflict)
        else:
            resolution = None  # Manual resolution needed
        
        conflict.resolution = resolution
        conflict.strategy_used = strategy
        self.resolution_history.append(conflict)
        
        return resolution
    
    def _choose_strategy(self, conflict: Conflict) -> ResolutionStrategy:
        """Choose appropriate resolution strategy."""
        if conflict.conflict_type == ConflictType.TEMPORAL_CONFLICT:
            return ResolutionStrategy.MOST_RECENT
        elif conflict.conflict_type == ConflictType.VALUE_CONFLICT:
            if all(isinstance(item.value, (int, float)) for item in conflict.items):
                return ResolutionStrategy.WEIGHTED_AVERAGE
            else:
                return ResolutionStrategy.MAJORITY_VOTE
        elif conflict.conflict_type == ConflictType.TYPE_CONFLICT:
            return ResolutionStrategy.MOST_TRUSTED
        else:
            return ResolutionStrategy.MAJORITY_VOTE
    
    def _resolve_most_recent(self, conflict: Conflict) -> Any:
        """Use most recent information."""
        most_recent = max(conflict.items, key=lambda x: x.timestamp)
        return most_recent.value
    
    def _resolve_most_trusted(self, conflict: Conflict) -> Any:
        """Use most trusted source."""
        most_trusted = max(conflict.items, key=lambda x: x.get_weight())
        return most_trusted.value
    
    def _resolve_majority_vote(self, conflict: Conflict) -> Any:
        """Use value with majority support."""
        value_votes: Dict[str, float] = {}
        
        for item in conflict.items:
            value_str = str(item.value)
            if value_str not in value_votes:
                value_votes[value_str] = 0
            value_votes[value_str] += item.get_weight()
        
        # Return value with highest vote
        winner = max(value_votes.items(), key=lambda x: x[1])
        
        # Find original value (not string)
        for item in conflict.items:
            if str(item.value) == winner[0]:
                return item.value
        
        return winner[0]
    
    def _resolve_weighted_average(self, conflict: Conflict) -> Any:
        """Calculate weighted average of numeric values."""
        if not all(isinstance(item.value, (int, float)) for item in conflict.items):
            return self._resolve_majority_vote(conflict)
        
        total_weight = sum(item.get_weight() for item in conflict.items)
        weighted_sum = sum(
            float(item.value) * item.get_weight()
            for item in conflict.items
        )
        
        return weighted_sum / total_weight if total_weight > 0 else 0
    
    def _resolve_union(self, conflict: Conflict) -> List[Any]:
        """Combine all unique values."""
        values = []
        seen = set()
        
        for item in conflict.items:
            value_str = str(item.value)
            if value_str not in seen:
                seen.add(value_str)
                values.append(item.value)
        
        return values
    
    def _resolve_intersection(self, conflict: Conflict) -> Optional[Any]:
        """Keep only common values."""
        if not conflict.items:
            return None
        
        # Find values that appear in all items
        value_counts: Dict[str, int] = {}
        for item in conflict.items:
            value_str = str(item.value)
            value_counts[value_str] = value_counts.get(value_str, 0) + 1
        
        common = [v for v, c in value_counts.items() if c == len(conflict.items)]
        
        if not common:
            return None
        
        # Return first common value
        for item in conflict.items:
            if str(item.value) == common[0]:
                return item.value
        
        return None


class ConsensusBuilder:
    """Builds consensus from multiple knowledge sources."""
    
    def __init__(self, threshold: float = 0.7):
        self.consensus_threshold = threshold
    
    def build_consensus(
        self,
        items: List[KnowledgeItem]
    ) -> Optional[FusedKnowledge]:
        """Build consensus from knowledge items."""
        if not items:
            return None
        
        # Check if consensus exists
        consensus_value, confidence = self._find_consensus(items)
        
        if confidence < self.consensus_threshold:
            return None  # No consensus
        
        # Extract sources and provenance
        sources = list(set(item.source for item in items))
        provenance = [
            f"Source {item.source.source_id}: {item.value} "
            f"(weight: {item.get_weight():.2f})"
            for item in items
        ]
        
        return FusedKnowledge(
            entity=items[0].entity,
            attribute=items[0].attribute,
            value=consensus_value,
            confidence=confidence,
            sources=sources,
            conflicts_resolved=[],
            provenance=provenance
        )
    
    def _find_consensus(
        self,
        items: List[KnowledgeItem]
    ) -> Tuple[Any, float]:
        """Find consensus value and confidence."""
        # For numeric values, use weighted average
        if all(isinstance(item.value, (int, float)) for item in items):
            total_weight = sum(item.get_weight() for item in items)
            weighted_sum = sum(
                float(item.value) * item.get_weight()
                for item in items
            )
            
            avg_value = weighted_sum / total_weight if total_weight > 0 else 0
            
            # Calculate confidence based on agreement
            variance = sum(
                item.get_weight() * (float(item.value) - avg_value)**2
                for item in items
            ) / total_weight if total_weight > 0 else 0
            
            confidence = 1.0 / (1.0 + variance)
            
            return avg_value, confidence
        
        # For categorical values, use majority
        value_weights: Dict[str, float] = {}
        for item in items:
            value_str = str(item.value)
            if value_str not in value_weights:
                value_weights[value_str] = 0
            value_weights[value_str] += item.get_weight()
        
        total_weight = sum(value_weights.values())
        max_weight = max(value_weights.values())
        
        confidence = max_weight / total_weight if total_weight > 0 else 0
        
        # Find value with max weight
        for item in items:
            if value_weights[str(item.value)] == max_weight:
                return item.value, confidence
        
        return items[0].value, 0.5


class KnowledgeFusionAgent:
    """Agent for fusing knowledge from multiple sources."""
    
    def __init__(
        self,
        consensus_threshold: float = 0.7,
        default_strategy: Optional[ResolutionStrategy] = None
    ):
        self.source_manager = SourceManager()
        self.conflict_detector = ConflictDetector()
        self.conflict_resolver = ConflictResolver()
        self.consensus_builder = ConsensusBuilder(consensus_threshold)
        self.default_strategy = default_strategy
        
        # Storage
        self.knowledge_base: Dict[str, List[KnowledgeItem]] = {}
        self.fused_knowledge: Dict[str, FusedKnowledge] = {}
        self.conflicts: List[Conflict] = []
    
    def add_source(self, source: Source) -> None:
        """Register a new information source."""
        self.source_manager.add_source(source)
    
    def ingest_knowledge(self, item: KnowledgeItem) -> None:
        """Ingest a piece of knowledge."""
        key = f"{item.entity}:{item.attribute}"
        
        if key not in self.knowledge_base:
            self.knowledge_base[key] = []
        
        self.knowledge_base[key].append(item)
    
    def fuse_knowledge(
        self,
        entity: Optional[str] = None,
        attribute: Optional[str] = None
    ) -> List[FusedKnowledge]:
        """Fuse knowledge with conflict resolution."""
        results = []
        
        # Filter items
        items_to_fuse = []
        for key, items in self.knowledge_base.items():
            e, a = key.split(':', 1)
            if (entity is None or e == entity) and (attribute is None or a == attribute):
                items_to_fuse.extend([(e, a, items)])
        
        for entity_name, attr_name, items in items_to_fuse:
            fused = self._fuse_items(entity_name, attr_name, items)
            if fused:
                key = f"{entity_name}:{attr_name}"
                self.fused_knowledge[key] = fused
                results.append(fused)
        
        return results
    
    def _fuse_items(
        self,
        entity: str,
        attribute: str,
        items: List[KnowledgeItem]
    ) -> Optional[FusedKnowledge]:
        """Fuse a set of knowledge items."""
        if not items:
            return None
        
        # Try consensus first
        consensus = self.consensus_builder.build_consensus(items)
        if consensus:
            return consensus
        
        # Detect conflicts
        conflicts = self.conflict_detector.detect_conflicts(items)
        
        if not conflicts:
            # No conflict, use weighted average/voting
            return self._simple_fusion(items)
        
        # Resolve conflicts
        resolved_conflicts = []
        for conflict in conflicts:
            resolution = self.conflict_resolver.resolve_conflict(
                conflict,
                self.default_strategy
            )
            resolved_conflicts.append(conflict)
        
        # Create fused knowledge from resolution
        main_conflict = conflicts[0]
        
        return FusedKnowledge(
            entity=entity,
            attribute=attribute,
            value=main_conflict.resolution,
            confidence=self._calculate_fusion_confidence(items, conflicts),
            sources=list(set(item.source for item in items)),
            conflicts_resolved=resolved_conflicts,
            provenance=[
                f"Resolved {len(conflicts)} conflict(s) using "
                f"{main_conflict.strategy_used.value if main_conflict.strategy_used else 'unknown'}"
            ]
        )
    
    def _simple_fusion(self, items: List[KnowledgeItem]) -> FusedKnowledge:
        """Simple fusion without conflicts."""
        # Use weighted average for numbers, voting for others
        if all(isinstance(item.value, (int, float)) for item in items):
            total_weight = sum(item.get_weight() for item in items)
            value = sum(
                float(item.value) * item.get_weight()
                for item in items
            ) / total_weight if total_weight > 0 else 0
        else:
            # Voting
            value_weights: Dict[str, float] = {}
            for item in items:
                v = str(item.value)
                value_weights[v] = value_weights.get(v, 0) + item.get_weight()
            
            winner = max(value_weights.items(), key=lambda x: x[1])
            for item in items:
                if str(item.value) == winner[0]:
                    value = item.value
                    break
        
        avg_confidence = sum(item.confidence for item in items) / len(items)
        
        return FusedKnowledge(
            entity=items[0].entity,
            attribute=items[0].attribute,
            value=value,
            confidence=avg_confidence,
            sources=list(set(item.source for item in items)),
            conflicts_resolved=[],
            provenance=[f"Fused from {len(items)} sources"]
        )
    
    def _calculate_fusion_confidence(
        self,
        items: List[KnowledgeItem],
        conflicts: List[Conflict]
    ) -> float:
        """Calculate confidence in fused result."""
        # Base confidence on average item confidence
        avg_confidence = sum(item.confidence for item in items) / len(items)
        
        # Reduce confidence based on conflict severity
        max_severity = max((c.severity for c in conflicts), default=0)
        
        return avg_confidence * (1.0 - max_severity * 0.5)
    
    def get_fused_knowledge(
        self,
        entity: str,
        attribute: str
    ) -> Optional[FusedKnowledge]:
        """Get fused knowledge for entity-attribute pair."""
        key = f"{entity}:{attribute}"
        return self.fused_knowledge.get(key)
    
    def get_provenance(
        self,
        entity: str,
        attribute: str
    ) -> List[str]:
        """Get provenance trail for fused knowledge."""
        fused = self.get_fused_knowledge(entity, attribute)
        if fused:
            return fused.provenance
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fusion statistics."""
        total_items = sum(len(items) for items in self.knowledge_base.values())
        total_fused = len(self.fused_knowledge)
        total_conflicts = len(self.conflict_resolver.resolution_history)
        
        conflict_types = {}
        for conflict in self.conflict_resolver.resolution_history:
            ct = conflict.conflict_type.value
            conflict_types[ct] = conflict_types.get(ct, 0) + 1
        
        return {
            'total_sources': len(self.source_manager.sources),
            'total_items_ingested': total_items,
            'total_fused_items': total_fused,
            'total_conflicts_detected': total_conflicts,
            'conflict_types': conflict_types,
            'avg_confidence': sum(
                f.confidence for f in self.fused_knowledge.values()
            ) / total_fused if total_fused > 0 else 0
        }


def demonstrate_knowledge_fusion():
    """Demonstrate the Knowledge Fusion Agent."""
    print("=" * 60)
    print("Knowledge Fusion Agent Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = KnowledgeFusionAgent(
        consensus_threshold=0.7,
        default_strategy=ResolutionStrategy.WEIGHTED_AVERAGE
    )
    
    # Create sources with different credibility
    sources = [
        Source("src1", "Academic Database", credibility=0.95, reliability=0.90,
               domain_expertise={"science", "research"}),
        Source("src2", "News Outlet", credibility=0.75, reliability=0.80,
               domain_expertise={"current_events"}),
        Source("src3", "Social Media", credibility=0.50, reliability=0.60),
        Source("src4", "Government Agency", credibility=0.98, reliability=0.95,
               domain_expertise={"statistics", "policy"}),
        Source("src5", "Expert Blog", credibility=0.80, reliability=0.85,
               domain_expertise={"science", "technology"}),
    ]
    
    for source in sources:
        agent.add_source(source)
    
    print("\n1. SOURCES REGISTERED")
    print("-" * 60)
    for source in sources:
        print(f"   {source.name}: Credibility={source.credibility:.2f}, "
              f"Reliability={source.reliability:.2f}")
    
    # Ingest conflicting knowledge about climate
    print("\n2. INGESTING KNOWLEDGE (with conflicts)")
    print("-" * 60)
    
    knowledge_items = [
        # Temperature rise estimates (conflict)
        KnowledgeItem("Earth", "avg_temp_rise", 1.2, sources[0], confidence=0.95),
        KnowledgeItem("Earth", "avg_temp_rise", 1.5, sources[1], confidence=0.70),
        KnowledgeItem("Earth", "avg_temp_rise", 1.3, sources[3], confidence=0.90),
        KnowledgeItem("Earth", "avg_temp_rise", 1.1, sources[4], confidence=0.85),
        
        # CO2 levels (consensus)
        KnowledgeItem("Earth", "co2_ppm", 415, sources[0], confidence=0.98),
        KnowledgeItem("Earth", "co2_ppm", 414, sources[3], confidence=0.96),
        KnowledgeItem("Earth", "co2_ppm", 416, sources[4], confidence=0.92),
        
        # Sea level rise (some conflict)
        KnowledgeItem("Earth", "sea_level_rise_mm", 3.3, sources[0], confidence=0.94),
        KnowledgeItem("Earth", "sea_level_rise_mm", 3.7, sources[1], confidence=0.65),
        KnowledgeItem("Earth", "sea_level_rise_mm", 3.4, sources[3], confidence=0.92),
    ]
    
    for item in knowledge_items:
        agent.ingest_knowledge(item)
        print(f"   {item.entity}.{item.attribute} = {item.value} "
              f"from {item.source.name}")
    
    # Fuse knowledge
    print("\n3. FUSING KNOWLEDGE")
    print("-" * 60)
    
    fused_results = agent.fuse_knowledge(entity="Earth")
    
    for result in fused_results:
        print(f"\n   Attribute: {result.attribute}")
        print(f"   Fused Value: {result.value:.2f}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Sources: {len(result.sources)} sources")
        print(f"   Conflicts Resolved: {len(result.conflicts_resolved)}")
        if result.conflicts_resolved:
            for conflict in result.conflicts_resolved:
                print(f"      - {conflict.conflict_type.value}: "
                      f"severity={conflict.severity:.2f}, "
                      f"strategy={conflict.strategy_used.value if conflict.strategy_used else 'N/A'}")
    
    # Get provenance
    print("\n4. PROVENANCE TRACKING")
    print("-" * 60)
    
    provenance = agent.get_provenance("Earth", "avg_temp_rise")
    print("   Temperature Rise Provenance:")
    for entry in provenance:
        print(f"   - {entry}")
    
    # Statistics
    print("\n5. FUSION STATISTICS")
    print("-" * 60)
    
    stats = agent.get_statistics()
    print(f"   Total Sources: {stats['total_sources']}")
    print(f"   Items Ingested: {stats['total_items_ingested']}")
    print(f"   Items Fused: {stats['total_fused_items']}")
    print(f"   Conflicts Detected: {stats['total_conflicts_detected']}")
    print(f"   Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"   Conflict Types: {stats['conflict_types']}")
    
    # Demonstrate consensus building
    print("\n6. CONSENSUS BUILDING")
    print("-" * 60)
    
    co2_fused = agent.get_fused_knowledge("Earth", "co2_ppm")
    if co2_fused:
        print(f"   CO2 Levels: {co2_fused.value:.1f} ppm")
        print(f"   Confidence: {co2_fused.confidence:.2f} (high consensus)")
        print(f"   Sources in agreement: {len(co2_fused.sources)}")
    
    print("\n" + "=" * 60)
    print("Knowledge Fusion Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_knowledge_fusion()
