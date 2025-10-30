"""
Pattern 29: Semantic Memory Networks
Description:
    Implements a graph-based semantic memory system that stores concepts
    and their relationships, enabling associative retrieval and reasoning.
Use Cases:
    - Knowledge representation
    - Concept relationship mapping
    - Semantic search and reasoning
    - Context-aware retrieval
Key Features:
    - Graph-based concept storage
    - Relationship typing and weighting
    - Associative spreading activation
    - Semantic similarity search
Example:
    >>> memory = SemanticMemoryNetwork()
    >>> memory.add_concept("dog", {"type": "animal"})
    >>> memory.add_relation("dog", "mammal", "is_a", weight=0.9)
    >>> results = memory.query("dog")
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import math
from collections import defaultdict, deque
import numpy as np
class RelationType(Enum):
    """Types of semantic relationships"""
    IS_A = "is_a"  # Taxonomic
    PART_OF = "part_of"  # Meronymic
    HAS_PROPERTY = "has_property"  # Attribute
    CAUSES = "causes"  # Causal
    SIMILAR_TO = "similar_to"  # Analogical
    OPPOSITE_OF = "opposite_of"  # Antonymic
    RELATED_TO = "related_to"  # General association
@dataclass
class Concept:
    """A concept node in semantic memory"""
    concept_id: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    activation_level: float = 0.0
    access_count: int = 0
    last_accessed: float = 0.0
    created_at: float = 0.0
@dataclass
class Relation:
    """A relationship between concepts"""
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0  # Strength of relationship
    bidirectional: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)
@dataclass
class QueryResult:
    """Result of a semantic query"""
    concept: Concept
    relevance_score: float
    path_from_query: List[str] = field(default_factory=list)
    activation_trace: Dict[str, float] = field(default_factory=dict)
class SemanticMemoryNetwork:
    """
    Graph-based semantic memory system
    Features:
    - Concept graph with typed relationships
    - Spreading activation for retrieval
    - Semantic similarity search
    - Path finding between concepts
    """
    def __init__(
        self,
        decay_rate: float = 0.1,
        activation_threshold: float = 0.1
    ):
        self.concepts: Dict[str, Concept] = {}
        self.relations: List[Relation] = []
        self.relation_index: Dict[str, List[Relation]] = defaultdict(list)
        self.decay_rate = decay_rate
        self.activation_threshold = activation_threshold
        self.embedding_dim = 128
    def add_concept(
        self,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        concept_id: Optional[str] = None
    ) -> str:
        """
        Add a concept to semantic memory
        Args:
            name: Concept name
            properties: Concept properties
            concept_id: Optional ID (generated if not provided)
        Returns:
            Concept ID
        """
        import time
        if concept_id is None:
            concept_id = f"concept_{len(self.concepts)}_{name.lower().replace(' ', '_')}"
        if concept_id in self.concepts:
            # Update existing concept
            self.concepts[concept_id].properties.update(properties or {})
        else:
            # Create new concept
            concept = Concept(
                concept_id=concept_id,
                name=name,
                properties=properties or {},
                embedding=self._generate_embedding(name, properties or {}),
                created_at=time.time()
            )
            self.concepts[concept_id] = concept
        return concept_id
    def add_relation(
        self,
        source: str,
        target: str,
        relation_type: RelationType,
        weight: float = 1.0,
        bidirectional: bool = False,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a relationship between concepts
        Args:
            source: Source concept ID or name
            target: Target concept ID or name
            relation_type: Type of relationship
            weight: Relationship strength (0-1)
            bidirectional: If True, creates reverse relation
            properties: Additional properties
        Returns:
            Success status
        """
        # Resolve concept IDs
        source_id = self._resolve_concept_id(source)
        target_id = self._resolve_concept_id(target)
        if not source_id or not target_id:
            return False
        # Create relation
        relation = Relation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            bidirectional=bidirectional,
            properties=properties or {}
        )
        self.relations.append(relation)
        self.relation_index[source_id].append(relation)
        if bidirectional:
            reverse_relation = Relation(
                source_id=target_id,
                target_id=source_id,
                relation_type=relation_type,
                weight=weight,
                bidirectional=False,
                properties=properties or {}
            )
            self.relations.append(reverse_relation)
            self.relation_index[target_id].append(reverse_relation)
        return True
    def query(
        self,
        query_text: str,
        max_results: int = 10,
        use_spreading_activation: bool = True
    ) -> List[QueryResult]:
        """
        Query semantic memory
        Args:
            query_text: Query string
            max_results: Maximum results to return
            use_spreading_activation: Use activation spreading
        Returns:
            List of query results
        """
        # Find initial concepts matching query
        initial_concepts = self._find_matching_concepts(query_text)
        if not initial_concepts:
            return []
        if use_spreading_activation:
            # Use spreading activation
            activation_map = self._spread_activation(
                initial_concepts,
                max_iterations=3
            )
        else:
            # Direct matching only
            activation_map = {
                concept_id: 1.0
                for concept_id in initial_concepts
            }
        # Create results
        results = []
        for concept_id, activation in activation_map.items():
            if activation >= self.activation_threshold:
                concept = self.concepts[concept_id]
                results.append(QueryResult(
                    concept=concept,
                    relevance_score=activation,
                    path_from_query=self._find_path(
                        initial_concepts[0], concept_id
                    ) if initial_concepts else [],
                    activation_trace=activation_map
                ))
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:max_results]
    def find_related_concepts(
        self,
        concept: str,
        relation_types: Optional[List[RelationType]] = None,
        max_depth: int = 2
    ) -> List[Tuple[Concept, List[Relation]]]:
        """
        Find concepts related to a given concept
        Args:
            concept: Concept ID or name
            relation_types: Filter by relation types
            max_depth: Maximum traversal depth
        Returns:
            List of (concept, path) tuples
        """
        concept_id = self._resolve_concept_id(concept)
        if not concept_id:
            return []
        visited = set()
        results = []
        # BFS traversal
        queue = deque([(concept_id, [], 0)])
        while queue:
            current_id, path, depth = queue.popleft()
            if current_id in visited or depth > max_depth:
                continue
            visited.add(current_id)
            if current_id != concept_id:
                results.append((self.concepts[current_id], path))
            # Explore neighbors
            for relation in self.relation_index.get(current_id, []):
                if relation_types and relation.relation_type not in relation_types:
                    continue
                target_id = relation.target_id
                if target_id not in visited:
                    new_path = path + [relation]
                    queue.append((target_id, new_path, depth + 1))
        return results
    def get_concept_neighborhood(
        self,
        concept: str,
        radius: int = 1
    ) -> Dict[str, Any]:
        """
        Get the local neighborhood of a concept
        Args:
            concept: Concept ID or name
            radius: Neighborhood radius
        Returns:
            Dictionary with concepts and relations
        """
        concept_id = self._resolve_concept_id(concept)
        if not concept_id:
            return {}
        neighborhood = {
            'center': self.concepts[concept_id],
            'concepts': {},
            'relations': []
        }
        visited = {concept_id}
        queue = deque([(concept_id, 0)])
        while queue:
            current_id, depth = queue.popleft()
            if depth >= radius:
                continue
            for relation in self.relation_index.get(current_id, []):
                target_id = relation.target_id
                neighborhood['relations'].append(relation)
                if target_id not in visited:
                    visited.add(target_id)
                    neighborhood['concepts'][target_id] = self.concepts[target_id]
                    queue.append((target_id, depth + 1))
        return neighborhood
    def _find_matching_concepts(self, query: str) -> List[str]:
        """Find concepts matching query text"""
        query_lower = query.lower()
        query_embedding = self._generate_embedding(query, {})
        matches = []
        for concept_id, concept in self.concepts.items():
            # Text matching
            if query_lower in concept.name.lower():
                matches.append((concept_id, 1.0))
            # Embedding similarity
            elif concept.embedding is not None:
                similarity = self._cosine_similarity(
                    query_embedding,
                    concept.embedding
                )
                if similarity > 0.5:
                    matches.append((concept_id, similarity))
        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        return [concept_id for concept_id, _ in matches]
    def _spread_activation(
        self,
        initial_concepts: List[str],
        max_iterations: int = 3
    ) -> Dict[str, float]:
        """
        Spread activation from initial concepts
        Args:
            initial_concepts: Starting concept IDs
            max_iterations: Number of spreading iterations
        Returns:
            Map of concept_id -> activation level
        """
        activation = defaultdict(float)
        # Initialize activation
        for concept_id in initial_concepts:
            activation[concept_id] = 1.0
        # Spread activation
        for iteration in range(max_iterations):
            new_activation = defaultdict(float)
            for concept_id, act_level in activation.items():
                if act_level < self.activation_threshold:
                    continue
                # Spread to related concepts
                for relation in self.relation_index.get(concept_id, []):
                    target_id = relation.target_id
                    # Calculate spread amount
                    spread_amount = (
                        act_level * 
                        relation.weight * 
                        (1.0 - self.decay_rate)
                    )
                    new_activation[target_id] += spread_amount
                # Decay current activation
                new_activation[concept_id] += act_level * (1.0 - self.decay_rate)
            activation = new_activation
        return dict(activation)
    def _find_path(
        self,
        source_id: str,
        target_id: str
    ) -> List[str]:
        """Find shortest path between concepts"""
        if source_id == target_id:
            return [source_id]
        visited = set()
        queue = deque([(source_id, [source_id])])
        while queue:
            current_id, path = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)
            if current_id == target_id:
                return path
            for relation in self.relation_index.get(current_id, []):
                target = relation.target_id
                if target not in visited:
                    queue.append((target, path + [target]))
        return []
    def _resolve_concept_id(self, concept: str) -> Optional[str]:
        """Resolve concept name or ID to ID"""
        if concept in self.concepts:
            return concept
        # Search by name
        for concept_id, c in self.concepts.items():
            if c.name.lower() == concept.lower():
                return concept_id
        return None
    def _generate_embedding(
        self,
        text: str,
        properties: Dict[str, Any]
    ) -> np.ndarray:
        """Generate embedding for concept (simplified)"""
        # In reality, use a proper embedding model
        # This is a simple hash-based approach for demonstration
        combined_text = text + " " + " ".join(str(v) for v in properties.values())
        # Simple hash-based embedding
        np.random.seed(hash(combined_text) % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    def _cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """Calculate cosine similarity"""
        return float(np.dot(vec1, vec2))
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory network statistics"""
        relation_counts = defaultdict(int)
        for relation in self.relations:
            relation_counts[relation.relation_type.value] += 1
        return {
            'total_concepts': len(self.concepts),
            'total_relations': len(self.relations),
            'relations_by_type': dict(relation_counts),
            'avg_connections_per_concept': (
                len(self.relations) / len(self.concepts)
                if self.concepts else 0
            ),
            'most_connected_concepts': self._get_most_connected(5)
        }
    def _get_most_connected(self, top_n: int) -> List[Tuple[str, int]]:
        """Get most connected concepts"""
        connection_counts = defaultdict(int)
        for relation in self.relations:
            connection_counts[relation.source_id] += 1
            connection_counts[relation.target_id] += 1
        sorted_concepts = sorted(
            connection_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            (self.concepts[concept_id].name, count)
            for concept_id, count in sorted_concepts[:top_n]
        ]
def main():
    """Demonstrate semantic memory networks"""
    print("=" * 60)
    print("Semantic Memory Networks Demonstration")
    print("=" * 60)
    memory = SemanticMemoryNetwork()
    print("\n1. Building Semantic Network")
    print("-" * 60)
    # Add concepts
    concepts = [
        ("dog", {"type": "animal", "lifespan": "10-13 years"}),
        ("cat", {"type": "animal", "lifespan": "12-18 years"}),
        ("mammal", {"type": "class", "warm_blooded": True}),
        ("animal", {"type": "kingdom"}),
        ("pet", {"type": "role"}),
        ("fur", {"type": "feature"}),
        ("bark", {"type": "sound"}),
        ("meow", {"type": "sound"}),
        ("wolf", {"type": "animal", "wild": True}),
        ("house", {"type": "place"}),
    ]
    for name, props in concepts:
        concept_id = memory.add_concept(name, props)
        print(f"Added concept: {name} (ID: {concept_id})")
    print("\n2. Adding Relationships")
    print("-" * 60)
    # Add relationships
    relations = [
        ("dog", "mammal", RelationType.IS_A, 1.0),
        ("cat", "mammal", RelationType.IS_A, 1.0),
        ("mammal", "animal", RelationType.IS_A, 1.0),
        ("dog", "pet", RelationType.IS_A, 0.8),
        ("cat", "pet", RelationType.IS_A, 0.9),
        ("dog", "fur", RelationType.HAS_PROPERTY, 0.9),
        ("cat", "fur", RelationType.HAS_PROPERTY, 0.9),
        ("dog", "bark", RelationType.HAS_PROPERTY, 1.0),
        ("cat", "meow", RelationType.HAS_PROPERTY, 1.0),
        ("dog", "wolf", RelationType.SIMILAR_TO, 0.7),
        ("dog", "cat", RelationType.SIMILAR_TO, 0.6),
        ("pet", "house", RelationType.RELATED_TO, 0.8),
    ]
    for source, target, rel_type, weight in relations:
        success = memory.add_relation(source, target, rel_type, weight)
        print(f"Added: {source} --[{rel_type.value}]-> {target} (weight: {weight})")
    print("\n" + "=" * 60)
    print("3. Querying Semantic Memory")
    print("=" * 60)
    queries = ["dog", "pet", "animal"]
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = memory.query(query, max_results=5)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.concept.name}")
            print(f"     Relevance: {result.relevance_score:.3f}")
            if result.path_from_query:
                path_names = [
                    memory.concepts[cid].name 
                    for cid in result.path_from_query
                ]
                print(f"     Path: {' -> '.join(path_names)}")
    print("\n" + "=" * 60)
    print("4. Finding Related Concepts")
    print("=" * 60)
    concept = "dog"
    print(f"\nConcepts related to '{concept}':")
    related = memory.find_related_concepts(
        concept,
        max_depth=2
    )
    for related_concept, path in related[:5]:
        print(f"\n  {related_concept.name}")
        print(f"    Properties: {related_concept.properties}")
        if path:
            path_str = " -> ".join(
                f"{r.source_id}[{r.relation_type.value}]"
                for r in path
            )
            print(f"    Path: {path_str}")
    print("\n" + "=" * 60)
    print("5. Concept Neighborhood")
    print("=" * 60)
    neighborhood = memory.get_concept_neighborhood("dog", radius=1)
    print(f"\nNeighborhood of '{neighborhood['center'].name}':")
    print(f"  Center properties: {neighborhood['center'].properties}")
    print(f"\n  Connected concepts ({len(neighborhood['concepts'])}):")
    for concept in list(neighborhood['concepts'].values())[:5]:
        print(f"    - {concept.name}: {concept.properties}")
    print(f"\n  Relations ({len(neighborhood['relations'])}):")
    for relation in neighborhood['relations'][:5]:
        source_name = memory.concepts[relation.source_id].name
        target_name = memory.concepts[relation.target_id].name
        print(f"    - {source_name} --[{relation.relation_type.value}]-> {target_name}")
    print("\n" + "=" * 60)
    print("6. Filtering by Relation Type")
    print("=" * 60)
    print("\nIS_A hierarchy for 'dog':")
    is_a_related = memory.find_related_concepts(
        "dog",
        relation_types=[RelationType.IS_A],
        max_depth=3
    )
    for concept, path in is_a_related:
        depth = len(path)
        indent = "  " * depth
        print(f"{indent}- {concept.name}")
    print("\n" + "=" * 60)
    print("7. Network Statistics")
    print("=" * 60)
    stats = memory.get_statistics()
    print(f"\nTotal Concepts: {stats['total_concepts']}")
    print(f"Total Relations: {stats['total_relations']}")
    print(f"Avg Connections per Concept: {stats['avg_connections_per_concept']:.2f}")
    print("\nRelations by Type:")
    for rel_type, count in stats['relations_by_type'].items():
        print(f"  {rel_type}: {count}")
    print("\nMost Connected Concepts:")
    for name, count in stats['most_connected_concepts']:
        print(f"  {name}: {count} connections")
    print("\n" + "=" * 60)
    print("Semantic Memory Networks demonstration complete!")
    print("=" * 60)
if __name__ == "__main__":
    main()
