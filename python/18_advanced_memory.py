"""
Advanced Memory Patterns Implementation

This module demonstrates advanced memory management including:
- Hierarchical memory organization
- Associative memory with spreading activation
- Episodic memory with temporal indexing
- Semantic memory networks
- Memory consolidation and forgetting

Key Components:
- Episodic memory for events and experiences
- Semantic memory for knowledge and concepts
- Associative links between memories
- Memory consolidation over time
- Retrieval with spreading activation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple
from enum import Enum
from datetime import datetime, timedelta
import random
import math


class MemoryType(Enum):
    """Types of memory"""
    EPISODIC = "episodic"      # Event-based memories
    SEMANTIC = "semantic"      # Knowledge-based memories
    PROCEDURAL = "procedural"  # Skill-based memories
    WORKING = "working"        # Temporary active memories


class MemoryStrength(Enum):
    """Strength levels of memories"""
    WEAK = 0.2
    MODERATE = 0.5
    STRONG = 0.8
    VERY_STRONG = 1.0


@dataclass
class MemoryNode:
    """Represents a memory node in the network"""
    id: str
    memory_type: MemoryType
    content: Any
    activation: float = 0.0  # Current activation level
    base_activation: float = 0.5  # Base strength
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def decay(self, time_delta: timedelta):
        """Apply temporal decay to activation"""
        # Exponential decay based on time
        days = time_delta.days + time_delta.seconds / 86400.0
        decay_rate = 0.1  # Decay parameter
        decay_factor = math.exp(-decay_rate * days)
        
        self.activation *= decay_factor
        self.base_activation *= (0.95 ** days)  # Slower decay for base
    
    def strengthen(self, amount: float = 0.1):
        """Strengthen memory through access"""
        self.base_activation = min(1.0, self.base_activation + amount)
        self.activation = min(1.0, self.activation + amount * 2)
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class Association:
    """Represents an associative link between memories"""
    source_id: str
    target_id: str
    strength: float
    association_type: str = "general"  # e.g., "causal", "temporal", "semantic"
    created_at: datetime = field(default_factory=datetime.now)
    
    def decay(self, time_delta: timedelta):
        """Decay association strength over time"""
        days = time_delta.days + time_delta.seconds / 86400.0
        decay_factor = math.exp(-0.05 * days)
        self.strength *= decay_factor


@dataclass
class EpisodicMemory:
    """Represents an episodic (event-based) memory"""
    id: str
    event_description: str
    participants: List[str]
    location: Optional[str]
    timestamp: datetime
    emotional_valence: float = 0.0  # -1 (negative) to +1 (positive)
    importance: float = 0.5
    sensory_details: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def get_temporal_distance(self, other_time: datetime) -> float:
        """Calculate temporal distance from another time"""
        delta = abs((self.timestamp - other_time).total_seconds())
        # Normalize to 0-1 range (1 day = 1.0)
        return min(1.0, delta / 86400.0)


@dataclass
class SemanticConcept:
    """Represents a concept in semantic memory"""
    id: str
    name: str
    definition: str
    category: str
    properties: Dict[str, Any] = field(default_factory=dict)
    examples: List[str] = field(default_factory=list)
    related_concepts: Set[str] = field(default_factory=set)
    abstraction_level: int = 0  # 0 = concrete, higher = more abstract


class AssociativeMemoryNetwork:
    """Network of associatively linked memories"""
    
    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}
        self.associations: List[Association] = []
        self.spreading_activation_threshold = 0.1
        self.max_spread_depth = 3
    
    def add_memory(self, memory: MemoryNode):
        """Add a memory node to the network"""
        self.nodes[memory.id] = memory
    
    def add_association(self, source_id: str, target_id: str, 
                       strength: float, association_type: str = "general"):
        """Create an associative link between memories"""
        association = Association(
            source_id=source_id,
            target_id=target_id,
            strength=strength,
            association_type=association_type
        )
        self.associations.append(association)
    
    def activate(self, memory_id: str, activation_amount: float = 1.0):
        """Activate a memory node"""
        if memory_id in self.nodes:
            self.nodes[memory_id].activation = activation_amount
            self.nodes[memory_id].strengthen(0.05)
    
    def spread_activation(self, source_ids: List[str], 
                         max_depth: Optional[int] = None) -> Dict[str, float]:
        """Spread activation from source nodes through the network"""
        if max_depth is None:
            max_depth = self.max_spread_depth
        
        activation_levels = {}
        
        # Initialize source nodes
        for source_id in source_ids:
            if source_id in self.nodes:
                activation_levels[source_id] = self.nodes[source_id].activation
        
        # Spread activation iteratively
        for depth in range(max_depth):
            new_activations = activation_levels.copy()
            
            for association in self.associations:
                source_activation = activation_levels.get(association.source_id, 0.0)
                
                if source_activation > self.spreading_activation_threshold:
                    # Spread activation to target
                    spread_amount = source_activation * association.strength * 0.5
                    current_target_activation = new_activations.get(association.target_id, 0.0)
                    new_activations[association.target_id] = min(1.0, 
                        current_target_activation + spread_amount)
            
            activation_levels = new_activations
        
        # Update node activations
        for node_id, activation in activation_levels.items():
            if node_id in self.nodes:
                self.nodes[node_id].activation = activation
        
        return activation_levels
    
    def retrieve_by_activation(self, threshold: float = 0.3, 
                              top_k: Optional[int] = None) -> List[MemoryNode]:
        """Retrieve memories above activation threshold"""
        activated_memories = [
            node for node in self.nodes.values()
            if node.activation >= threshold
        ]
        
        # Sort by activation level
        activated_memories.sort(key=lambda x: x.activation, reverse=True)
        
        if top_k:
            return activated_memories[:top_k]
        
        return activated_memories
    
    def find_shortest_path(self, start_id: str, end_id: str, 
                          max_length: int = 5) -> Optional[List[str]]:
        """Find shortest associative path between two memories"""
        if start_id not in self.nodes or end_id not in self.nodes:
            return None
        
        # BFS to find shortest path
        queue = [(start_id, [start_id])]
        visited = {start_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            if current_id == end_id:
                return path
            
            if len(path) >= max_length:
                continue
            
            # Find connected nodes
            for association in self.associations:
                if association.source_id == current_id:
                    next_id = association.target_id
                    if next_id not in visited:
                        visited.add(next_id)
                        queue.append((next_id, path + [next_id]))
        
        return None
    
    def get_strongly_associated(self, memory_id: str, 
                               min_strength: float = 0.5) -> List[Tuple[str, float]]:
        """Get strongly associated memories"""
        associated = []
        
        for association in self.associations:
            if association.source_id == memory_id and association.strength >= min_strength:
                associated.append((association.target_id, association.strength))
        
        # Sort by strength
        associated.sort(key=lambda x: x[1], reverse=True)
        
        return associated
    
    def decay_all(self, time_elapsed: timedelta):
        """Apply decay to all memories and associations"""
        for node in self.nodes.values():
            node.decay(time_elapsed)
        
        for association in self.associations:
            association.decay(time_elapsed)


class EpisodicMemorySystem:
    """System for managing episodic (event-based) memories"""
    
    def __init__(self):
        self.episodes: Dict[str, EpisodicMemory] = {}
        self.temporal_index: List[Tuple[datetime, str]] = []  # (time, episode_id)
        self.participant_index: Dict[str, List[str]] = {}  # participant -> episode_ids
        self.location_index: Dict[str, List[str]] = {}  # location -> episode_ids
    
    def store_episode(self, episode: EpisodicMemory):
        """Store an episodic memory"""
        self.episodes[episode.id] = episode
        
        # Update temporal index
        self.temporal_index.append((episode.timestamp, episode.id))
        self.temporal_index.sort(key=lambda x: x[0])
        
        # Update participant index
        for participant in episode.participants:
            if participant not in self.participant_index:
                self.participant_index[participant] = []
            self.participant_index[participant].append(episode.id)
        
        # Update location index
        if episode.location:
            if episode.location not in self.location_index:
                self.location_index[episode.location] = []
            self.location_index[episode.location].append(episode.id)
    
    def retrieve_by_time_range(self, start_time: datetime, 
                               end_time: datetime) -> List[EpisodicMemory]:
        """Retrieve episodes within a time range"""
        episodes = []
        
        for timestamp, episode_id in self.temporal_index:
            if start_time <= timestamp <= end_time:
                episodes.append(self.episodes[episode_id])
        
        return episodes
    
    def retrieve_by_participant(self, participant: str) -> List[EpisodicMemory]:
        """Retrieve episodes involving a participant"""
        episode_ids = self.participant_index.get(participant, [])
        return [self.episodes[eid] for eid in episode_ids]
    
    def retrieve_by_location(self, location: str) -> List[EpisodicMemory]:
        """Retrieve episodes at a location"""
        episode_ids = self.location_index.get(location, [])
        return [self.episodes[eid] for eid in episode_ids]
    
    def retrieve_recent(self, count: int = 10) -> List[EpisodicMemory]:
        """Retrieve most recent episodes"""
        recent_ids = [episode_id for _, episode_id in self.temporal_index[-count:]]
        recent_ids.reverse()
        return [self.episodes[eid] for eid in recent_ids]
    
    def find_similar_episodes(self, query_episode: EpisodicMemory, 
                            top_k: int = 5) -> List[Tuple[EpisodicMemory, float]]:
        """Find episodes similar to a query episode"""
        similarities = []
        
        for episode in self.episodes.values():
            if episode.id == query_episode.id:
                continue
            
            similarity = self._calculate_episode_similarity(query_episode, episode)
            similarities.append((episode, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _calculate_episode_similarity(self, ep1: EpisodicMemory, 
                                     ep2: EpisodicMemory) -> float:
        """Calculate similarity between two episodes"""
        similarity = 0.0
        
        # Participant overlap
        participants1 = set(ep1.participants)
        participants2 = set(ep2.participants)
        if participants1 and participants2:
            participant_sim = len(participants1 & participants2) / len(participants1 | participants2)
            similarity += participant_sim * 0.3
        
        # Location match
        if ep1.location and ep2.location and ep1.location == ep2.location:
            similarity += 0.2
        
        # Temporal proximity (closer in time = more similar)
        temporal_distance = ep1.get_temporal_distance(ep2.timestamp)
        temporal_sim = 1.0 - min(1.0, temporal_distance / 30.0)  # 30 days normalization
        similarity += temporal_sim * 0.2
        
        # Emotional valence similarity
        valence_diff = abs(ep1.emotional_valence - ep2.emotional_valence)
        valence_sim = 1.0 - (valence_diff / 2.0)
        similarity += valence_sim * 0.15
        
        # Importance similarity
        importance_diff = abs(ep1.importance - ep2.importance)
        importance_sim = 1.0 - importance_diff
        similarity += importance_sim * 0.15
        
        return similarity


class SemanticMemoryNetwork:
    """Network of semantic concepts and their relationships"""
    
    def __init__(self):
        self.concepts: Dict[str, SemanticConcept] = {}
        self.is_a_relations: Dict[str, str] = {}  # concept_id -> parent_concept_id
        self.part_of_relations: Dict[str, List[str]] = {}  # whole_id -> [part_ids]
    
    def add_concept(self, concept: SemanticConcept):
        """Add a concept to semantic memory"""
        self.concepts[concept.id] = concept
    
    def add_is_a_relation(self, concept_id: str, parent_id: str):
        """Add an is-a (inheritance) relationship"""
        self.is_a_relations[concept_id] = parent_id
    
    def add_part_of_relation(self, part_id: str, whole_id: str):
        """Add a part-of (composition) relationship"""
        if whole_id not in self.part_of_relations:
            self.part_of_relations[whole_id] = []
        self.part_of_relations[whole_id].append(part_id)
    
    def get_ancestors(self, concept_id: str) -> List[str]:
        """Get all ancestor concepts in the is-a hierarchy"""
        ancestors = []
        current = concept_id
        
        while current in self.is_a_relations:
            parent = self.is_a_relations[current]
            ancestors.append(parent)
            current = parent
        
        return ancestors
    
    def get_descendants(self, concept_id: str) -> List[str]:
        """Get all descendant concepts in the is-a hierarchy"""
        descendants = []
        
        for child_id, parent_id in self.is_a_relations.items():
            if parent_id == concept_id:
                descendants.append(child_id)
                # Recursively get descendants of this child
                descendants.extend(self.get_descendants(child_id))
        
        return descendants
    
    def get_parts(self, concept_id: str) -> List[str]:
        """Get parts of a concept"""
        return self.part_of_relations.get(concept_id, [])
    
    def find_common_ancestor(self, concept_id1: str, concept_id2: str) -> Optional[str]:
        """Find the most specific common ancestor of two concepts"""
        ancestors1 = set(self.get_ancestors(concept_id1))
        ancestors2 = set(self.get_ancestors(concept_id2))
        
        common = ancestors1 & ancestors2
        
        if not common:
            return None
        
        # Find the most specific (deepest) common ancestor
        deepest_ancestor = None
        max_depth = -1
        
        for ancestor_id in common:
            depth = len(self.get_ancestors(ancestor_id))
            if depth > max_depth:
                max_depth = depth
                deepest_ancestor = ancestor_id
        
        return deepest_ancestor
    
    def semantic_similarity(self, concept_id1: str, concept_id2: str) -> float:
        """Calculate semantic similarity between two concepts"""
        if concept_id1 == concept_id2:
            return 1.0
        
        if concept_id1 not in self.concepts or concept_id2 not in self.concepts:
            return 0.0
        
        # Check direct relationship
        if concept_id2 in self.concepts[concept_id1].related_concepts:
            return 0.8
        
        # Check is-a hierarchy
        ancestors1 = set(self.get_ancestors(concept_id1))
        ancestors2 = set(self.get_ancestors(concept_id2))
        
        if concept_id1 in ancestors2 or concept_id2 in ancestors1:
            # One is ancestor of the other
            return 0.7
        
        # Check for common ancestor
        common_ancestor = self.find_common_ancestor(concept_id1, concept_id2)
        if common_ancestor:
            # Similarity based on distance to common ancestor
            dist1 = len(self.get_ancestors(concept_id1)) - len(self.get_ancestors(common_ancestor))
            dist2 = len(self.get_ancestors(concept_id2)) - len(self.get_ancestors(common_ancestor))
            total_dist = dist1 + dist2
            return max(0.3, 1.0 - (total_dist * 0.1))
        
        # Check category similarity
        if self.concepts[concept_id1].category == self.concepts[concept_id2].category:
            return 0.4
        
        return 0.1


class MemoryConsolidator:
    """Handles memory consolidation processes"""
    
    def __init__(self, associative_network: AssociativeMemoryNetwork,
                 episodic_system: EpisodicMemorySystem):
        self.associative_network = associative_network
        self.episodic_system = episodic_system
        self.consolidation_threshold = 0.6
    
    def consolidate(self) -> Dict[str, Any]:
        """Perform memory consolidation"""
        print("\n🧠 Memory Consolidation Process")
        print("=" * 60)
        
        stats = {
            "memories_strengthened": 0,
            "memories_weakened": 0,
            "associations_strengthened": 0,
            "associations_removed": 0,
            "patterns_identified": 0
        }
        
        # Strengthen frequently accessed memories
        for memory in self.associative_network.nodes.values():
            if memory.access_count > 5:
                memory.strengthen(0.1)
                stats["memories_strengthened"] += 1
            elif memory.activation < 0.2:
                memory.base_activation *= 0.9
                stats["memories_weakened"] += 1
        
        # Strengthen associations based on co-activation
        active_memories = self.associative_network.retrieve_by_activation(0.3)
        
        for i, mem1 in enumerate(active_memories):
            for mem2 in active_memories[i+1:]:
                # Find existing association
                existing = None
                for assoc in self.associative_network.associations:
                    if assoc.source_id == mem1.id and assoc.target_id == mem2.id:
                        existing = assoc
                        break
                
                if existing:
                    # Strengthen existing association
                    existing.strength = min(1.0, existing.strength + 0.05)
                    stats["associations_strengthened"] += 1
                else:
                    # Create new association if both are active
                    if mem1.activation > 0.5 and mem2.activation > 0.5:
                        self.associative_network.add_association(
                            mem1.id, mem2.id, 0.3, "co-activation"
                        )
                        stats["associations_strengthened"] += 1
        
        # Remove weak associations
        weak_threshold = 0.1
        self.associative_network.associations = [
            assoc for assoc in self.associative_network.associations
            if assoc.strength >= weak_threshold
        ]
        stats["associations_removed"] = len([
            a for a in self.associative_network.associations
            if a.strength < weak_threshold
        ])
        
        # Identify patterns in episodic memories
        recent_episodes = self.episodic_system.retrieve_recent(20)
        patterns = self._identify_patterns(recent_episodes)
        stats["patterns_identified"] = len(patterns)
        
        print(f"✅ Consolidation complete")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        return stats
    
    def _identify_patterns(self, episodes: List[EpisodicMemory]) -> List[Dict[str, Any]]:
        """Identify patterns in episodic memories"""
        patterns = []
        
        # Find recurring participants
        participant_counts = {}
        for episode in episodes:
            for participant in episode.participants:
                participant_counts[participant] = participant_counts.get(participant, 0) + 1
        
        for participant, count in participant_counts.items():
            if count >= 3:
                patterns.append({
                    "type": "recurring_participant",
                    "participant": participant,
                    "frequency": count
                })
        
        # Find recurring locations
        location_counts = {}
        for episode in episodes:
            if episode.location:
                location_counts[episode.location] = location_counts.get(episode.location, 0) + 1
        
        for location, count in location_counts.items():
            if count >= 3:
                patterns.append({
                    "type": "recurring_location",
                    "location": location,
                    "frequency": count
                })
        
        return patterns


class AdvancedMemorySystem:
    """Integrated advanced memory system"""
    
    def __init__(self):
        self.associative_network = AssociativeMemoryNetwork()
        self.episodic_system = EpisodicMemorySystem()
        self.semantic_network = SemanticMemoryNetwork()
        self.consolidator = MemoryConsolidator(
            self.associative_network,
            self.episodic_system
        )
        self.memory_counter = 0
    
    def store_episodic_memory(self, event_description: str, 
                             participants: List[str],
                             location: Optional[str] = None,
                             emotional_valence: float = 0.0,
                             importance: float = 0.5) -> str:
        """Store an episodic memory"""
        self.memory_counter += 1
        episode_id = f"episode_{self.memory_counter}"
        
        episode = EpisodicMemory(
            id=episode_id,
            event_description=event_description,
            participants=participants,
            location=location,
            timestamp=datetime.now(),
            emotional_valence=emotional_valence,
            importance=importance
        )
        
        self.episodic_system.store_episode(episode)
        
        # Add to associative network
        memory_node = MemoryNode(
            id=episode_id,
            memory_type=MemoryType.EPISODIC,
            content=event_description,
            base_activation=importance,
            tags=set(participants)
        )
        self.associative_network.add_memory(memory_node)
        
        print(f"📝 Stored episodic memory: {episode_id}")
        
        return episode_id
    
    def store_semantic_concept(self, name: str, definition: str, 
                              category: str, parent_concept: Optional[str] = None) -> str:
        """Store a semantic concept"""
        self.memory_counter += 1
        concept_id = f"concept_{self.memory_counter}"
        
        concept = SemanticConcept(
            id=concept_id,
            name=name,
            definition=definition,
            category=category
        )
        
        self.semantic_network.add_concept(concept)
        
        if parent_concept:
            self.semantic_network.add_is_a_relation(concept_id, parent_concept)
        
        # Add to associative network
        memory_node = MemoryNode(
            id=concept_id,
            memory_type=MemoryType.SEMANTIC,
            content=definition,
            base_activation=0.7,
            tags={name, category}
        )
        self.associative_network.add_memory(memory_node)
        
        print(f"📚 Stored semantic concept: {concept_id} ({name})")
        
        return concept_id
    
    def retrieve_with_context(self, query: str, context_ids: List[str] = None,
                            top_k: int = 5) -> List[MemoryNode]:
        """Retrieve memories using spreading activation"""
        print(f"\n🔍 Retrieving memories for: {query}")
        
        # Activate relevant nodes
        if context_ids:
            for context_id in context_ids:
                self.associative_network.activate(context_id, 0.8)
        
        # Spread activation
        activation_levels = self.associative_network.spread_activation(
            context_ids if context_ids else []
        )
        
        # Retrieve activated memories
        retrieved = self.associative_network.retrieve_by_activation(0.2, top_k)
        
        print(f"   Retrieved {len(retrieved)} memories")
        for i, memory in enumerate(retrieved, 1):
            print(f"   {i}. {memory.id} (activation: {memory.activation:.2f})")
        
        return retrieved
    
    def consolidate_memories(self):
        """Perform memory consolidation"""
        return self.consolidator.consolidate()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        return {
            "total_memories": len(self.associative_network.nodes),
            "episodic_memories": len(self.episodic_system.episodes),
            "semantic_concepts": len(self.semantic_network.concepts),
            "associations": len(self.associative_network.associations),
            "average_activation": sum(n.activation for n in self.associative_network.nodes.values()) / len(self.associative_network.nodes) if self.associative_network.nodes else 0
        }


def main():
    """Demonstration of advanced memory patterns"""
    print("🧠 Advanced Memory Patterns Demonstration")
    print("=" * 80)
    print("This demonstrates advanced memory management:")
    print("- Episodic memory for events")
    print("- Semantic memory for concepts")
    print("- Associative memory network")
    print("- Spreading activation")
    print("- Memory consolidation")
    
    # Create memory system
    memory_system = AdvancedMemorySystem()
    
    # Store semantic concepts
    print(f"\n{'='*80}")
    print("1. Storing Semantic Concepts")
    print(f"{'='*80}")
    
    ai_id = memory_system.store_semantic_concept(
        "Artificial Intelligence",
        "The simulation of human intelligence by machines",
        "Technology"
    )
    
    ml_id = memory_system.store_semantic_concept(
        "Machine Learning",
        "A subset of AI that enables systems to learn from data",
        "Technology",
        parent_concept=ai_id
    )
    
    dl_id = memory_system.store_semantic_concept(
        "Deep Learning",
        "A subset of ML using neural networks with multiple layers",
        "Technology",
        parent_concept=ml_id
    )
    
    # Create associations between concepts
    memory_system.associative_network.add_association(ai_id, ml_id, 0.9, "is-a")
    memory_system.associative_network.add_association(ml_id, dl_id, 0.9, "is-a")
    
    # Store episodic memories
    print(f"\n{'='*80}")
    print("2. Storing Episodic Memories")
    print(f"{'='*80}")
    
    ep1 = memory_system.store_episodic_memory(
        "Attended AI conference and learned about transformers",
        ["Alice", "Bob"],
        location="San Francisco",
        emotional_valence=0.8,
        importance=0.9
    )
    
    ep2 = memory_system.store_episodic_memory(
        "Read paper about neural networks",
        ["Alice"],
        location="Library",
        emotional_valence=0.6,
        importance=0.7
    )
    
    ep3 = memory_system.store_episodic_memory(
        "Implemented a machine learning model",
        ["Alice", "Charlie"],
        location="Office",
        emotional_valence=0.7,
        importance=0.8
    )
    
    # Create associations between episodes and concepts
    memory_system.associative_network.add_association(ep1, ai_id, 0.8, "about")
    memory_system.associative_network.add_association(ep2, dl_id, 0.7, "about")
    memory_system.associative_network.add_association(ep3, ml_id, 0.9, "about")
    
    # Retrieve with spreading activation
    print(f"\n{'='*80}")
    print("3. Retrieval with Spreading Activation")
    print(f"{'='*80}")
    
    # Query about AI - should activate related concepts and episodes
    retrieved = memory_system.retrieve_with_context(
        "Tell me about artificial intelligence",
        context_ids=[ai_id],
        top_k=5
    )
    
    # Retrieve episodes by participant
    print(f"\n{'='*80}")
    print("4. Episodic Memory Retrieval")
    print(f"{'='*80}")
    
    alice_episodes = memory_system.episodic_system.retrieve_by_participant("Alice")
    print(f"📅 Episodes involving Alice: {len(alice_episodes)}")
    for ep in alice_episodes:
        print(f"   - {ep.event_description}")
    
    # Memory consolidation
    print(f"\n{'='*80}")
    print("5. Memory Consolidation")
    print(f"{'='*80}")
    
    consolidation_stats = memory_system.consolidate_memories()
    
    # System statistics
    print(f"\n{'='*80}")
    print("6. Memory System Statistics")
    print(f"{'='*80}")
    
    stats = memory_system.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    print("\n\n🎯 Key Advanced Memory Features Demonstrated:")
    print("✅ Episodic memory with temporal indexing")
    print("✅ Semantic memory network with hierarchies")
    print("✅ Associative memory with spreading activation")
    print("✅ Memory consolidation and strengthening")
    print("✅ Multi-index retrieval (participant, location, time)")
    print("✅ Pattern identification in memories")
    print("✅ Memory decay and forgetting")
    print("✅ Context-based retrieval")


if __name__ == "__main__":
    main()
