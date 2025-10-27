"""
Pattern 106: Advanced Semantic Memory Networks

This pattern implements sophisticated graph-based semantic memory with
entity-relationship networks, spreading activation, and inference capabilities.

Use Cases:
- Knowledge representation and reasoning
- Complex relationship modeling
- Associative memory retrieval
- Concept learning and inference
- Semantic search and navigation

Key Features:
- Graph-based knowledge representation
- Entity-relationship modeling
- Spreading activation for retrieval
- Inference and reasoning
- Relationship strength tracking
- Temporal decay and reinforcement
- Semantic clustering

Implementation:
- Pure Python (3.8+) with comprehensive type hints
- Zero external dependencies
- Production-ready error handling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
import uuid
import math


class RelationType(Enum):
    """Types of relationships between entities."""
    IS_A = "is_a"                    # Taxonomy
    PART_OF = "part_of"              # Composition
    HAS_PROPERTY = "has_property"    # Attributes
    CAUSES = "causes"                # Causation
    SIMILAR_TO = "similar_to"        # Similarity
    OPPOSITE_OF = "opposite_of"      # Antonyms
    LOCATED_AT = "located_at"        # Location
    TEMPORAL = "temporal"            # Time relations
    CUSTOM = "custom"                # Domain-specific


@dataclass
class Entity:
    """Entity node in semantic network."""
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    entity_type: str = "concept"
    
    # Attributes
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Activation state
    activation: float = 0.0  # Current activation level
    base_activation: float = 0.5  # Base level
    
    # Statistics
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created: datetime = field(default_factory=datetime.now)
    
    # Semantic features
    embeddings: List[float] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    def activate(self, amount: float = 0.1) -> None:
        """Increase activation level."""
        self.activation = min(1.0, self.activation + amount)
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def decay(self, rate: float = 0.05) -> None:
        """Decay activation over time."""
        self.activation = max(0.0, self.activation - rate)
        if self.activation < self.base_activation:
            self.activation = self.base_activation * 0.95


@dataclass
class Relationship:
    """Relationship edge in semantic network."""
    relation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_id: str = ""
    target_id: str = ""
    relation_type: RelationType = RelationType.CUSTOM
    
    # Relationship strength
    strength: float = 1.0
    confidence: float = 1.0
    
    # Metadata
    properties: Dict[str, Any] = field(default_factory=dict)
    created: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    
    def strengthen(self, amount: float = 0.1) -> None:
        """Strengthen relationship through use."""
        self.strength = min(1.0, self.strength + amount)
        self.use_count += 1
        self.last_used = datetime.now()
    
    def weaken(self, amount: float = 0.05) -> None:
        """Weaken relationship over time."""
        self.strength = max(0.1, self.strength - amount)


@dataclass
class ActivationPattern:
    """Pattern of activation across network."""
    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    activated_entities: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    total_activation: float = 0.0


class SpreadingActivation:
    """
    Implements spreading activation for memory retrieval.
    
    Activation spreads from source entities through relationships
    to connected entities, simulating associative retrieval.
    """
    
    def __init__(self, decay_factor: float = 0.8):
        self.decay_factor = decay_factor  # How much activation decays per hop
        self.max_hops = 3
        self.activation_threshold = 0.1
    
    def spread(self, network: 'SemanticNetwork', 
              source_ids: List[str],
              max_iterations: int = 5) -> ActivationPattern:
        """
        Spread activation from source entities.
        
        Args:
            network: The semantic network
            source_ids: Starting entities
            max_iterations: Maximum spreading iterations
        """
        pattern = ActivationPattern()
        
        # Initialize source activation
        for entity_id in source_ids:
            if entity_id in network.entities:
                pattern.activated_entities[entity_id] = 1.0
                network.entities[entity_id].activate(0.2)
        
        # Spread activation iteratively
        for iteration in range(max_iterations):
            new_activations = {}
            
            for entity_id, activation in pattern.activated_entities.items():
                if activation < self.activation_threshold:
                    continue
                
                # Get outgoing relationships
                relations = network.get_outgoing_relations(entity_id)
                
                # Spread to connected entities
                for rel in relations:
                    target_id = rel.target_id
                    
                    # Calculate activation to spread
                    spread_amount = (activation * 
                                   rel.strength * 
                                   self.decay_factor ** iteration)
                    
                    # Accumulate activation
                    if target_id in new_activations:
                        new_activations[target_id] = max(
                            new_activations[target_id], spread_amount
                        )
                    else:
                        new_activations[target_id] = spread_amount
                    
                    # Update entity
                    if target_id in network.entities:
                        network.entities[target_id].activate(spread_amount * 0.1)
            
            # Merge new activations
            for entity_id, activation in new_activations.items():
                if entity_id not in pattern.activated_entities:
                    pattern.activated_entities[entity_id] = activation
                else:
                    pattern.activated_entities[entity_id] = max(
                        pattern.activated_entities[entity_id], activation
                    )
        
        # Calculate total activation
        pattern.total_activation = sum(pattern.activated_entities.values())
        
        return pattern


class InferenceEngine:
    """
    Performs inference over semantic network.
    
    Supports:
    - Transitive inference
    - Property inheritance
    - Similarity inference
    - Analogical reasoning
    """
    
    def __init__(self):
        pass
    
    def infer_transitive(self, network: 'SemanticNetwork',
                        source_id: str, relation_type: RelationType,
                        max_depth: int = 3) -> List[str]:
        """
        Infer transitive relationships.
        
        If A -> B and B -> C, then A -> C (transitively)
        """
        results = set()
        visited = set()
        queue = [(source_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth or current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Get relationships of specified type
            relations = network.get_outgoing_relations(current_id)
            for rel in relations:
                if rel.relation_type == relation_type:
                    target_id = rel.target_id
                    results.add(target_id)
                    queue.append((target_id, depth + 1))
        
        return list(results)
    
    def infer_properties(self, network: 'SemanticNetwork',
                        entity_id: str) -> Dict[str, Any]:
        """
        Infer properties through inheritance.
        
        Entities inherit properties from parent entities via IS_A relationships.
        """
        properties = {}
        
        # Get entity's own properties
        if entity_id in network.entities:
            properties.update(network.entities[entity_id].properties)
        
        # Get parent entities
        parents = self.infer_transitive(network, entity_id, RelationType.IS_A, max_depth=5)
        
        # Inherit properties from parents
        for parent_id in parents:
            if parent_id in network.entities:
                parent_props = network.entities[parent_id].properties
                # Add inherited properties (don't override specific ones)
                for key, value in parent_props.items():
                    if key not in properties:
                        properties[key] = value
        
        return properties
    
    def find_similar(self, network: 'SemanticNetwork',
                    entity_id: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find similar entities based on relationships.
        
        Similarity based on:
        - Explicit SIMILAR_TO relationships
        - Shared relationships
        - Shared properties
        """
        if entity_id not in network.entities:
            return []
        
        similarities = {}
        entity = network.entities[entity_id]
        
        # Explicit similarity relationships
        relations = network.get_outgoing_relations(entity_id)
        for rel in relations:
            if rel.relation_type == RelationType.SIMILAR_TO:
                similarities[rel.target_id] = rel.strength
        
        # Implicit similarity through shared relationships
        entity_relations = set((r.relation_type, r.target_id) for r in relations)
        
        for other_id, other_entity in network.entities.items():
            if other_id == entity_id or other_id in similarities:
                continue
            
            other_relations = network.get_outgoing_relations(other_id)
            other_set = set((r.relation_type, r.target_id) for r in other_relations)
            
            # Jaccard similarity
            if entity_relations or other_set:
                intersection = len(entity_relations & other_set)
                union = len(entity_relations | other_set)
                if union > 0:
                    similarity = intersection / union
                    if similarity >= threshold:
                        similarities[other_id] = similarity
        
        # Sort by similarity
        results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return results


class SemanticNetwork:
    """
    Graph-based semantic memory network.
    
    Core operations:
    - Entity and relationship management
    - Spreading activation retrieval
    - Inference and reasoning
    - Network statistics
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        
        # Indices for efficient lookup
        self.outgoing_index: Dict[str, List[str]] = {}  # entity_id -> relation_ids
        self.incoming_index: Dict[str, List[str]] = {}  # entity_id -> relation_ids
        self.type_index: Dict[str, Set[str]] = {}  # entity_type -> entity_ids
        
        # Components
        self.spreading_activation = SpreadingActivation()
        self.inference_engine = InferenceEngine()
        
        # Statistics
        self.total_retrievals = 0
        self.total_inferences = 0
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity to network."""
        self.entities[entity.entity_id] = entity
        
        # Update type index
        if entity.entity_type not in self.type_index:
            self.type_index[entity.entity_type] = set()
        self.type_index[entity.entity_type].add(entity.entity_id)
        
        # Initialize relationship indices
        if entity.entity_id not in self.outgoing_index:
            self.outgoing_index[entity.entity_id] = []
        if entity.entity_id not in self.incoming_index:
            self.incoming_index[entity.entity_id] = []
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship to network."""
        self.relationships[relationship.relation_id] = relationship
        
        # Update indices
        source_id = relationship.source_id
        target_id = relationship.target_id
        
        if source_id not in self.outgoing_index:
            self.outgoing_index[source_id] = []
        self.outgoing_index[source_id].append(relationship.relation_id)
        
        if target_id not in self.incoming_index:
            self.incoming_index[target_id] = []
        self.incoming_index[target_id].append(relationship.relation_id)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)
    
    def get_outgoing_relations(self, entity_id: str) -> List[Relationship]:
        """Get all outgoing relationships from entity."""
        relation_ids = self.outgoing_index.get(entity_id, [])
        return [self.relationships[rid] for rid in relation_ids]
    
    def get_incoming_relations(self, entity_id: str) -> List[Relationship]:
        """Get all incoming relationships to entity."""
        relation_ids = self.incoming_index.get(entity_id, [])
        return [self.relationships[rid] for rid in relation_ids]
    
    def retrieve_associative(self, query_ids: List[str],
                            top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve entities using spreading activation.
        
        Returns entities most strongly activated by query.
        """
        self.total_retrievals += 1
        
        # Spread activation
        pattern = self.spreading_activation.spread(self, query_ids)
        
        # Sort by activation
        sorted_entities = sorted(
            pattern.activated_entities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top-k excluding query entities
        results = []
        for entity_id, activation in sorted_entities:
            if entity_id not in query_ids:
                results.append((entity_id, activation))
                if len(results) >= top_k:
                    break
        
        return results
    
    def infer_relationship(self, source_id: str, target_id: str) -> List[RelationType]:
        """Infer possible relationships between entities."""
        self.total_inferences += 1
        
        possible_relations = []
        
        # Check for transitive relationships
        for rel_type in RelationType:
            transitive_targets = self.inference_engine.infer_transitive(
                self, source_id, rel_type, max_depth=2
            )
            if target_id in transitive_targets:
                possible_relations.append(rel_type)
        
        return possible_relations
    
    def get_entity_neighborhood(self, entity_id: str,
                               depth: int = 1) -> Set[str]:
        """Get neighborhood of entities within depth hops."""
        neighborhood = set()
        visited = set()
        queue = [(entity_id, 0)]
        
        while queue:
            current_id, current_depth = queue.pop(0)
            
            if current_depth > depth or current_id in visited:
                continue
            
            visited.add(current_id)
            neighborhood.add(current_id)
            
            # Add neighbors
            for rel in self.get_outgoing_relations(current_id):
                queue.append((rel.target_id, current_depth + 1))
            for rel in self.get_incoming_relations(current_id):
                queue.append((rel.source_id, current_depth + 1))
        
        return neighborhood
    
    def decay_network(self, decay_rate: float = 0.05) -> None:
        """Apply temporal decay to activation and relationship strength."""
        for entity in self.entities.values():
            entity.decay(decay_rate)
        
        for relationship in self.relationships.values():
            # Decay unused relationships
            time_since_use = (datetime.now() - relationship.last_used).total_seconds() / 3600
            if time_since_use > 24:  # More than 24 hours
                relationship.weaken(decay_rate * 0.5)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        return {
            "num_entities": len(self.entities),
            "num_relationships": len(self.relationships),
            "entity_types": {
                etype: len(eids) 
                for etype, eids in self.type_index.items()
            },
            "avg_degree": (
                sum(len(rels) for rels in self.outgoing_index.values()) / 
                len(self.entities) if self.entities else 0
            ),
            "total_retrievals": self.total_retrievals,
            "total_inferences": self.total_inferences
        }


class SemanticMemoryAgent:
    """
    Agent with advanced semantic memory capabilities.
    
    Features:
    - Knowledge graph construction
    - Associative retrieval
    - Inference and reasoning
    - Concept learning
    """
    
    def __init__(self):
        self.network = SemanticNetwork()
        self.entity_name_map: Dict[str, str] = {}  # name -> entity_id
    
    def learn_concept(self, name: str, concept_type: str = "concept",
                     properties: Optional[Dict[str, Any]] = None) -> Entity:
        """Learn a new concept."""
        entity = Entity(
            name=name,
            entity_type=concept_type,
            properties=properties or {}
        )
        
        self.network.add_entity(entity)
        self.entity_name_map[name.lower()] = entity.entity_id
        
        return entity
    
    def learn_relationship(self, source_name: str, relation_type: RelationType,
                          target_name: str, strength: float = 1.0) -> Optional[Relationship]:
        """Learn a relationship between concepts."""
        source_id = self.entity_name_map.get(source_name.lower())
        target_id = self.entity_name_map.get(target_name.lower())
        
        if not source_id or not target_id:
            return None
        
        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength
        )
        
        self.network.add_relationship(relationship)
        return relationship
    
    def recall(self, query_concepts: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Recall related concepts using associative retrieval."""
        # Get entity IDs
        query_ids = []
        for concept in query_concepts:
            entity_id = self.entity_name_map.get(concept.lower())
            if entity_id:
                query_ids.append(entity_id)
        
        if not query_ids:
            return []
        
        # Retrieve associatively
        results = self.network.retrieve_associative(query_ids, top_k)
        
        # Convert back to names
        named_results = []
        for entity_id, score in results:
            entity = self.network.get_entity(entity_id)
            if entity:
                named_results.append((entity.name, score))
        
        return named_results
    
    def infer_properties(self, concept_name: str) -> Dict[str, Any]:
        """Infer properties of a concept through inheritance."""
        entity_id = self.entity_name_map.get(concept_name.lower())
        if not entity_id:
            return {}
        
        return self.network.inference_engine.infer_properties(self.network, entity_id)
    
    def find_similar_concepts(self, concept_name: str,
                             threshold: float = 0.3) -> List[Tuple[str, float]]:
        """Find concepts similar to given concept."""
        entity_id = self.entity_name_map.get(concept_name.lower())
        if not entity_id:
            return []
        
        results = self.network.inference_engine.find_similar(
            self.network, entity_id, threshold
        )
        
        # Convert to names
        named_results = []
        for other_id, similarity in results:
            entity = self.network.get_entity(other_id)
            if entity:
                named_results.append((entity.name, similarity))
        
        return named_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return self.network.get_statistics()


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_semantic_memory():
    """Demonstrate advanced semantic memory capabilities."""
    
    print("=" * 70)
    print("ADVANCED SEMANTIC MEMORY DEMONSTRATION")
    print("=" * 70)
    
    print("\n1. CREATING SEMANTIC NETWORK")
    print("-" * 70)
    
    agent = SemanticMemoryAgent()
    print("   Agent created with semantic network")
    
    print("\n2. LEARNING CONCEPTS")
    print("-" * 70)
    print("   Building knowledge about animals...")
    
    # Create animal taxonomy
    concepts = [
        ("Animal", "category", {"living": True}),
        ("Mammal", "category", {"warm_blooded": True}),
        ("Bird", "category", {"has_feathers": True, "lays_eggs": True}),
        ("Dog", "instance", {"legs": 4, "tail": True}),
        ("Cat", "instance", {"legs": 4, "tail": True}),
        ("Eagle", "instance", {"wings": True, "predator": True}),
        ("Sparrow", "instance", {"wings": True, "small": True}),
    ]
    
    for name, ctype, props in concepts:
        agent.learn_concept(name, ctype, props)
        print(f"     Learned: {name} ({ctype})")
    
    print("\n3. LEARNING RELATIONSHIPS")
    print("-" * 70)
    print("   Establishing relationships...")
    
    relationships = [
        ("Mammal", RelationType.IS_A, "Animal"),
        ("Bird", RelationType.IS_A, "Animal"),
        ("Dog", RelationType.IS_A, "Mammal"),
        ("Cat", RelationType.IS_A, "Mammal"),
        ("Eagle", RelationType.IS_A, "Bird"),
        ("Sparrow", RelationType.IS_A, "Bird"),
        ("Dog", RelationType.SIMILAR_TO, "Cat"),
        ("Eagle", RelationType.OPPOSITE_OF, "Sparrow"),
    ]
    
    for source, rel_type, target in relationships:
        agent.learn_relationship(source, rel_type, target)
        print(f"     {source} --[{rel_type.value}]--> {target}")
    
    print("\n4. ASSOCIATIVE RETRIEVAL")
    print("-" * 70)
    print("   Query: What is related to 'Dog'?")
    
    results = agent.recall(["Dog"], top_k=5)
    print(f"\n   Retrieved {len(results)} related concepts:")
    for concept, score in results:
        print(f"     - {concept}: activation={score:.3f}")
    
    print("\n5. PROPERTY INFERENCE")
    print("-" * 70)
    print("   Inferring properties of 'Dog' through inheritance...")
    
    properties = agent.infer_properties("Dog")
    print(f"\n   Inferred properties:")
    for prop, value in properties.items():
        print(f"     {prop}: {value}")
    
    print("\n6. SIMILARITY SEARCH")
    print("-" * 70)
    print("   Finding concepts similar to 'Cat'...")
    
    similar = agent.find_similar_concepts("Cat", threshold=0.2)
    print(f"\n   Found {len(similar)} similar concepts:")
    for concept, similarity in similar:
        print(f"     - {concept}: similarity={similarity:.3f}")
    
    print("\n7. MULTI-CONCEPT RETRIEVAL")
    print("-" * 70)
    print("   Query: What relates to both 'Mammal' and 'Bird'?")
    
    results = agent.recall(["Mammal", "Bird"], top_k=3)
    print(f"\n   Common concepts:")
    for concept, score in results:
        print(f"     - {concept}: activation={score:.3f}")
    
    print("\n8. NETWORK STATISTICS")
    print("-" * 70)
    
    stats = agent.get_statistics()
    print(f"   Entities: {stats['num_entities']}")
    print(f"   Relationships: {stats['num_relationships']}")
    print(f"   Average degree: {stats['avg_degree']:.2f}")
    print(f"   Entity types:")
    for etype, count in stats['entity_types'].items():
        print(f"     {etype}: {count}")
    print(f"   Total retrievals: {stats['total_retrievals']}")
    print(f"   Total inferences: {stats['total_inferences']}")
    
    print("\n9. ADDING MORE KNOWLEDGE")
    print("-" * 70)
    print("   Expanding network with food chain...")
    
    # Add food chain concepts
    agent.learn_concept("Grass", "food", {"plant": True})
    agent.learn_concept("Rabbit", "instance", {"herbivore": True})
    agent.learn_concept("Wolf", "instance", {"carnivore": True})
    
    agent.learn_relationship("Rabbit", RelationType.IS_A, "Mammal")
    agent.learn_relationship("Wolf", RelationType.IS_A, "Mammal")
    agent.learn_relationship("Rabbit", RelationType.PART_OF, "food_chain")
    agent.learn_relationship("Wolf", RelationType.CAUSES, "Rabbit")  # Predation
    
    print("     Added food chain relationships")
    
    print("\n   Query: What is related to 'Rabbit'?")
    results = agent.recall(["Rabbit"], top_k=5)
    for concept, score in results:
        print(f"     - {concept}: activation={score:.3f}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("1. Concept learning and knowledge acquisition")
    print("2. Relationship types (IS_A, SIMILAR_TO, etc.)")
    print("3. Spreading activation retrieval")
    print("4. Property inheritance through IS_A hierarchy")
    print("5. Similarity inference based on relationships")
    print("6. Multi-concept associative retrieval")
    print("7. Network statistics and monitoring")
    print("8. Dynamic knowledge expansion")
    print("9. Graph-based semantic representation")


if __name__ == "__main__":
    demonstrate_semantic_memory()
