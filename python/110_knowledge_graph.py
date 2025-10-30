"""
Pattern 110: Knowledge Graph Agent

This pattern implements knowledge graph construction, reasoning, and querying,
enabling structured knowledge representation and complex reasoning.

Use Cases:
- Knowledge base construction
- Semantic search and reasoning
- Question answering over knowledge
- Relationship discovery
- Knowledge inference and completion
- Multi-hop reasoning

Key Features:
- Graph-based knowledge representation
- RDF-like triple structure
- SPARQL-style querying
- Inference and reasoning
- Path finding and exploration
- Knowledge completion
- Ontology support

Implementation:
- Pure Python (3.8+) with comprehensive type hints
- Zero external dependencies
- Production-ready error handling

🎉 THIS IS PATTERN 110 - MILESTONE ACHIEVEMENT! 🎉
Reaching 110/170 patterns (64.7% completion)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime
from collections import defaultdict, deque
import uuid


class EntityType(Enum):
    """Types of entities in knowledge graph."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    OBJECT = "object"
    ABSTRACT = "abstract"


class RelationType(Enum):
    """Types of relationships."""
    IS_A = "is_a"                  # Taxonomy
    HAS_PROPERTY = "has_property"  # Attributes
    PART_OF = "part_of"           # Composition
    LOCATED_IN = "located_in"     # Location
    WORKS_FOR = "works_for"       # Employment
    CREATED_BY = "created_by"     # Authorship
    RELATED_TO = "related_to"     # General relation
    CAUSES = "causes"             # Causation
    PRECEDES = "precedes"         # Temporal
    SIMILAR_TO = "similar_to"     # Similarity


@dataclass
class Entity:
    """Entity (node) in knowledge graph."""
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    label: str = ""
    entity_type: EntityType = EntityType.CONCEPT
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    source: str = "system"
    
    # Aliases
    aliases: List[str] = field(default_factory=list)
    
    def __hash__(self) -> int:
        return hash(self.entity_id)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.entity_id == other.entity_id


@dataclass
class Triple:
    """Triple (edge) in knowledge graph: (subject, predicate, object)."""
    triple_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    subject: str = ""  # entity_id
    predicate: RelationType = RelationType.RELATED_TO
    obj: str = ""  # entity_id or literal value
    
    # Quality
    confidence: float = 1.0
    source: str = "system"
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_tuple(self) -> Tuple[str, RelationType, str]:
        """Convert to simple tuple."""
        return (self.subject, self.predicate, self.obj)


@dataclass
class QueryPattern:
    """SPARQL-like query pattern."""
    subject: Optional[str] = None  # Can be variable like "?x"
    predicate: Optional[RelationType] = None
    obj: Optional[str] = None
    
    def matches(self, triple: Triple) -> bool:
        """Check if triple matches pattern."""
        if self.subject and not self._match_field(self.subject, triple.subject):
            return False
        if self.predicate and triple.predicate != self.predicate:
            return False
        if self.obj and not self._match_field(self.obj, triple.obj):
            return False
        return True
    
    def _match_field(self, pattern: str, value: str) -> bool:
        """Match field allowing variables."""
        if pattern.startswith("?"):
            return True  # Variable matches anything
        return pattern == value


class KnowledgeGraph:
    """
    Core knowledge graph structure.
    
    Stores entities and triples, provides efficient querying
    and traversal operations.
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.triples: Dict[str, Triple] = {}
        
        # Indices for efficient querying
        self.subject_index: Dict[str, Set[str]] = defaultdict(set)  # subject -> triple_ids
        self.object_index: Dict[str, Set[str]] = defaultdict(set)   # object -> triple_ids
        self.predicate_index: Dict[RelationType, Set[str]] = defaultdict(set)  # predicate -> triple_ids
        
        # Label to entity mapping
        self.label_to_entity: Dict[str, str] = {}  # label -> entity_id
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity to graph."""
        self.entities[entity.entity_id] = entity
        self.label_to_entity[entity.label.lower()] = entity.entity_id
        
        # Index aliases
        for alias in entity.aliases:
            self.label_to_entity[alias.lower()] = entity.entity_id
    
    def add_triple(self, triple: Triple) -> None:
        """Add triple to graph."""
        self.triples[triple.triple_id] = triple
        
        # Update indices
        self.subject_index[triple.subject].add(triple.triple_id)
        self.object_index[triple.obj].add(triple.triple_id)
        self.predicate_index[triple.predicate].add(triple.triple_id)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)
    
    def get_entity_by_label(self, label: str) -> Optional[Entity]:
        """Get entity by label."""
        entity_id = self.label_to_entity.get(label.lower())
        return self.entities.get(entity_id) if entity_id else None
    
    def get_triples_by_subject(self, subject_id: str) -> List[Triple]:
        """Get all triples with given subject."""
        triple_ids = self.subject_index.get(subject_id, set())
        return [self.triples[tid] for tid in triple_ids]
    
    def get_triples_by_object(self, object_id: str) -> List[Triple]:
        """Get all triples with given object."""
        triple_ids = self.object_index.get(object_id, set())
        return [self.triples[tid] for tid in triple_ids]
    
    def get_triples_by_predicate(self, predicate: RelationType) -> List[Triple]:
        """Get all triples with given predicate."""
        triple_ids = self.predicate_index.get(predicate, set())
        return [self.triples[tid] for tid in triple_ids]
    
    def query(self, pattern: QueryPattern) -> List[Triple]:
        """Query graph with pattern."""
        candidates = set(self.triples.keys())
        
        # Filter by subject
        if pattern.subject and not pattern.subject.startswith("?"):
            candidates &= self.subject_index.get(pattern.subject, set())
        
        # Filter by predicate
        if pattern.predicate:
            candidates &= self.predicate_index.get(pattern.predicate, set())
        
        # Filter by object
        if pattern.obj and not pattern.obj.startswith("?"):
            candidates &= self.object_index.get(pattern.obj, set())
        
        # Apply pattern matching
        results = []
        for triple_id in candidates:
            triple = self.triples[triple_id]
            if pattern.matches(triple):
                results.append(triple)
        
        return results
    
    def find_paths(self, start_id: str, end_id: str,
                   max_length: int = 5) -> List[List[Triple]]:
        """
        Find paths between entities.
        
        Uses BFS to find shortest paths up to max_length.
        """
        if start_id == end_id:
            return [[]]
        
        paths = []
        queue: deque = deque([(start_id, [])])
        visited = {start_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            if len(path) >= max_length:
                continue
            
            # Get outgoing triples
            outgoing = self.get_triples_by_subject(current_id)
            
            for triple in outgoing:
                next_id = triple.obj
                new_path = path + [triple]
                
                if next_id == end_id:
                    paths.append(new_path)
                    continue
                
                if next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, new_path))
        
        return paths
    
    def get_neighbors(self, entity_id: str,
                     relation_type: Optional[RelationType] = None) -> List[Tuple[Entity, Triple]]:
        """Get neighboring entities."""
        neighbors = []
        
        # Outgoing edges
        outgoing = self.get_triples_by_subject(entity_id)
        for triple in outgoing:
            if relation_type is None or triple.predicate == relation_type:
                neighbor = self.get_entity(triple.obj)
                if neighbor:
                    neighbors.append((neighbor, triple))
        
        # Incoming edges
        incoming = self.get_triples_by_object(entity_id)
        for triple in incoming:
            if relation_type is None or triple.predicate == relation_type:
                neighbor = self.get_entity(triple.subject)
                if neighbor:
                    neighbors.append((neighbor, triple))
        
        return neighbors


class ReasoningEngine:
    """
    Performs inference and reasoning over knowledge graph.
    
    Supports:
    - Transitive reasoning
    - Property inheritance
    - Inverse relationships
    - Type inference
    """
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        
        # Inference rules
        self.transitive_relations = {
            RelationType.IS_A,
            RelationType.PART_OF,
            RelationType.LOCATED_IN
        }
    
    def infer_transitive(self, start_id: str,
                        relation: RelationType,
                        max_depth: int = 10) -> Set[str]:
        """
        Infer transitive closure of relationship.
        
        If A->B and B->C, then A->C (transitively).
        """
        if relation not in self.transitive_relations:
            return set()
        
        closure = set()
        visited = set()
        queue = deque([(start_id, 0)])
        
        while queue:
            current_id, depth = queue.popleft()
            
            if depth >= max_depth or current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Get triples with this relation
            triples = self.graph.get_triples_by_subject(current_id)
            
            for triple in triples:
                if triple.predicate == relation:
                    target_id = triple.obj
                    closure.add(target_id)
                    queue.append((target_id, depth + 1))
        
        return closure
    
    def infer_type(self, entity_id: str) -> Set[EntityType]:
        """Infer entity types through IS_A relationships."""
        types = set()
        
        # Get direct type
        entity = self.graph.get_entity(entity_id)
        if entity:
            types.add(entity.entity_type)
        
        # Infer from IS_A relationships
        parents = self.infer_transitive(entity_id, RelationType.IS_A)
        
        for parent_id in parents:
            parent = self.graph.get_entity(parent_id)
            if parent:
                types.add(parent.entity_type)
        
        return types
    
    def complete_knowledge(self) -> List[Triple]:
        """
        Suggest new triples based on patterns and inference.
        
        Knowledge graph completion through pattern-based inference.
        """
        suggested = []
        
        # Find patterns: if A->B and A->C, and C->D, suggest B->D
        # This is simplified pattern-based completion
        
        for entity_id in self.graph.entities.keys():
            # Get outgoing relations
            outgoing = self.graph.get_triples_by_subject(entity_id)
            
            # Check for similar entities
            for triple in outgoing:
                if triple.predicate == RelationType.SIMILAR_TO:
                    similar_id = triple.obj
                    
                    # Copy relationships from similar entity
                    similar_triples = self.graph.get_triples_by_subject(similar_id)
                    
                    for st in similar_triples:
                        # Suggest same relationship
                        if st.predicate not in [t.predicate for t in outgoing]:
                            suggested_triple = Triple(
                                subject=entity_id,
                                predicate=st.predicate,
                                obj=st.obj,
                                confidence=0.7  # Lower confidence for inferred
                            )
                            suggested.append(suggested_triple)
        
        return suggested[:10]  # Limit suggestions


class QueryEngine:
    """
    Processes complex queries over knowledge graph.
    
    Supports SPARQL-like queries and natural language questions.
    """
    
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
        self.reasoning = ReasoningEngine(graph)
    
    def ask(self, question: str) -> List[Dict[str, Any]]:
        """
        Answer natural language question.
        
        Simplified NL query processing.
        """
        question_lower = question.lower()
        results = []
        
        # Pattern: "What is X?"
        if question_lower.startswith("what is"):
            entity_name = question_lower.replace("what is", "").strip("? ")
            entity = self.graph.get_entity_by_label(entity_name)
            
            if entity:
                results.append({
                    "entity": entity.label,
                    "type": entity.entity_type.value,
                    "properties": entity.properties
                })
        
        # Pattern: "Who works for X?"
        elif "works for" in question_lower:
            org_name = question_lower.split("works for")[-1].strip("? ")
            org_entity = self.graph.get_entity_by_label(org_name)
            
            if org_entity:
                # Find people who work for this org
                triples = self.graph.get_triples_by_object(org_entity.entity_id)
                
                for triple in triples:
                    if triple.predicate == RelationType.WORKS_FOR:
                        person = self.graph.get_entity(triple.subject)
                        if person:
                            results.append({
                                "person": person.label,
                                "organization": org_entity.label
                            })
        
        # Pattern: "Where is X located?"
        elif "where is" in question_lower or "located" in question_lower:
            entity_name = question_lower.replace("where is", "").replace("located", "").strip("? ")
            entity = self.graph.get_entity_by_label(entity_name)
            
            if entity:
                # Find location
                triples = self.graph.get_triples_by_subject(entity.entity_id)
                
                for triple in triples:
                    if triple.predicate == RelationType.LOCATED_IN:
                        location = self.graph.get_entity(triple.obj)
                        if location:
                            results.append({
                                "entity": entity.label,
                                "location": location.label
                            })
        
        return results
    
    def sparql_query(self, patterns: List[QueryPattern]) -> List[Dict[str, str]]:
        """
        Execute SPARQL-like query with multiple patterns.
        
        Returns variable bindings.
        """
        if not patterns:
            return []
        
        # Start with first pattern
        results = self.graph.query(patterns[0])
        bindings = []
        
        for triple in results:
            binding = {}
            
            # Extract variable bindings
            pattern = patterns[0]
            if pattern.subject and pattern.subject.startswith("?"):
                binding[pattern.subject] = triple.subject
            if pattern.obj and pattern.obj.startswith("?"):
                binding[pattern.obj] = triple.obj
            
            bindings.append(binding)
        
        # Apply remaining patterns
        for pattern in patterns[1:]:
            new_bindings = []
            
            for binding in bindings:
                # Substitute variables in pattern
                subj = binding.get(pattern.subject, pattern.subject) if pattern.subject else None
                obj = binding.get(pattern.obj, pattern.obj) if pattern.obj else None
                
                # Query with substituted pattern
                query_pattern = QueryPattern(subj, pattern.predicate, obj)
                matches = self.graph.query(query_pattern)
                
                for match in matches:
                    new_binding = binding.copy()
                    
                    if pattern.subject and pattern.subject.startswith("?"):
                        new_binding[pattern.subject] = match.subject
                    if pattern.obj and pattern.obj.startswith("?"):
                        new_binding[pattern.obj] = match.obj
                    
                    new_bindings.append(new_binding)
            
            bindings = new_bindings
        
        return bindings


class KnowledgeGraphAgent:
    """
    Agent that builds and reasons over knowledge graphs.
    
    Features:
    - Graph construction
    - Querying and reasoning
    - Knowledge completion
    - Path finding
    - Natural language queries
    """
    
    def __init__(self):
        self.graph = KnowledgeGraph()
        self.reasoning = ReasoningEngine(self.graph)
        self.query_engine = QueryEngine(self.graph)
        
        # Statistics
        self.queries_executed = 0
        self.inferences_made = 0
    
    def add_entity(self, label: str, entity_type: EntityType,
                   properties: Optional[Dict[str, Any]] = None,
                   aliases: Optional[List[str]] = None) -> Entity:
        """Add entity to knowledge graph."""
        entity = Entity(
            label=label,
            entity_type=entity_type,
            properties=properties or {},
            aliases=aliases or []
        )
        
        self.graph.add_entity(entity)
        return entity
    
    def add_fact(self, subject: str, predicate: RelationType,
                obj: str, confidence: float = 1.0) -> Optional[Triple]:
        """Add fact (triple) to knowledge graph."""
        # Get entity IDs
        subj_entity = self.graph.get_entity_by_label(subject)
        obj_entity = self.graph.get_entity_by_label(obj)
        
        if not subj_entity:
            return None
        
        obj_id = obj_entity.entity_id if obj_entity else obj
        
        triple = Triple(
            subject=subj_entity.entity_id,
            predicate=predicate,
            obj=obj_id,
            confidence=confidence
        )
        
        self.graph.add_triple(triple)
        return triple
    
    def ask(self, question: str) -> List[Dict[str, Any]]:
        """Ask natural language question."""
        self.queries_executed += 1
        return self.query_engine.ask(question)
    
    def find_path(self, start: str, end: str) -> Optional[List[Triple]]:
        """Find path between entities."""
        start_entity = self.graph.get_entity_by_label(start)
        end_entity = self.graph.get_entity_by_label(end)
        
        if not start_entity or not end_entity:
            return None
        
        paths = self.graph.find_paths(start_entity.entity_id, end_entity.entity_id)
        return paths[0] if paths else None
    
    def infer_types(self, entity_label: str) -> Set[EntityType]:
        """Infer all types of entity."""
        entity = self.graph.get_entity_by_label(entity_label)
        if not entity:
            return set()
        
        self.inferences_made += 1
        return self.reasoning.infer_type(entity.entity_id)
    
    def suggest_facts(self) -> List[Triple]:
        """Suggest new facts through knowledge completion."""
        self.inferences_made += 1
        return self.reasoning.complete_knowledge()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "entities": len(self.graph.entities),
            "triples": len(self.graph.triples),
            "queries_executed": self.queries_executed,
            "inferences_made": self.inferences_made,
            "entity_types": len(set(e.entity_type for e in self.graph.entities.values()))
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_knowledge_graph():
    """Demonstrate knowledge graph capabilities."""
    
    print("=" * 70)
    print("🎉 PATTERN 110: KNOWLEDGE GRAPH AGENT 🎉")
    print("=" * 70)
    print("MILESTONE: 110/170 patterns implemented (64.7%)!")
    print("=" * 70)
    
    print("\n1. INITIALIZING KNOWLEDGE GRAPH")
    print("-" * 70)
    
    agent = KnowledgeGraphAgent()
    print("   Agent initialized with empty knowledge graph")
    
    print("\n2. ADDING ENTITIES")
    print("-" * 70)
    print("   Building knowledge base...")
    
    # Add people
    alice = agent.add_entity("Alice", EntityType.PERSON, {"age": 30, "skills": ["Python", "ML"]})
    bob = agent.add_entity("Bob", EntityType.PERSON, {"age": 25, "skills": ["Java"]})
    print(f"     Added: {alice.label} ({alice.entity_type.value})")
    print(f"     Added: {bob.label} ({bob.entity_type.value})")
    
    # Add organizations
    acme = agent.add_entity("Acme Corp", EntityType.ORGANIZATION, {"industry": "Technology"})
    print(f"     Added: {acme.label} ({acme.entity_type.value})")
    
    # Add locations
    nyc = agent.add_entity("New York", EntityType.LOCATION, {"country": "USA", "population": 8000000})
    sf = agent.add_entity("San Francisco", EntityType.LOCATION, {"country": "USA"})
    print(f"     Added: {nyc.label}, {sf.label}")
    
    # Add concepts
    ai = agent.add_entity("Artificial Intelligence", EntityType.CONCEPT)
    ml = agent.add_entity("Machine Learning", EntityType.CONCEPT)
    print(f"     Added: {ai.label}, {ml.label}")
    
    print("\n3. ADDING FACTS (TRIPLES)")
    print("-" * 70)
    print("   Establishing relationships...")
    
    # Work relationships
    agent.add_fact("Alice", RelationType.WORKS_FOR, "Acme Corp")
    agent.add_fact("Bob", RelationType.WORKS_FOR, "Acme Corp")
    print("     Alice works for Acme Corp")
    print("     Bob works for Acme Corp")
    
    # Location relationships
    agent.add_fact("Acme Corp", RelationType.LOCATED_IN, "San Francisco")
    agent.add_fact("Alice", RelationType.LOCATED_IN, "New York")
    print("     Acme Corp located in San Francisco")
    print("     Alice located in New York")
    
    # Concept relationships
    agent.add_fact("Machine Learning", RelationType.PART_OF, "Artificial Intelligence")
    print("     ML is part of AI")
    
    print("\n4. NATURAL LANGUAGE QUERIES")
    print("-" * 70)
    
    # Query 1
    print("   Q: Who works for Acme Corp?")
    results = agent.ask("Who works for Acme Corp?")
    print(f"   A: Found {len(results)} results")
    for r in results:
        print(f"      - {r.get('person', 'Unknown')}")
    
    # Query 2
    print("\n   Q: Where is Acme Corp located?")
    results = agent.ask("Where is Acme Corp located?")
    if results:
        print(f"   A: {results[0].get('location', 'Unknown')}")
    
    # Query 3
    print("\n   Q: What is Alice?")
    results = agent.ask("What is Alice?")
    if results:
        print(f"   A: Type: {results[0].get('type', 'Unknown')}")
        print(f"      Properties: {results[0].get('properties', {})}")
    
    print("\n5. PATH FINDING")
    print("-" * 70)
    print("   Finding connection between Alice and San Francisco...")
    
    path = agent.find_path("Alice", "San Francisco")
    if path:
        print(f"   Found path with {len(path)} steps:")
        for i, triple in enumerate(path, 1):
            subj = agent.graph.get_entity(triple.subject)
            obj = agent.graph.get_entity(triple.obj)
            print(f"      {i}. {subj.label if subj else '?'} "
                  f"--[{triple.predicate.value}]--> "
                  f"{obj.label if obj else triple.obj}")
    else:
        print("   No direct path found")
    
    print("\n6. TYPE INFERENCE")
    print("-" * 70)
    print("   Inferring types through relationships...")
    
    types = agent.infer_types("Machine Learning")
    print(f"   Machine Learning types: {[t.value for t in types]}")
    
    print("\n7. KNOWLEDGE COMPLETION")
    print("-" * 70)
    print("   Suggesting new facts based on patterns...")
    
    # Add similarity for completion
    agent.add_fact("Bob", RelationType.SIMILAR_TO, "Alice")
    
    suggestions = agent.suggest_facts()
    print(f"   Generated {len(suggestions)} suggestions:")
    for s in suggestions[:5]:
        subj = agent.graph.get_entity(s.subject)
        obj = agent.graph.get_entity(s.obj)
        print(f"     - {subj.label if subj else '?'} "
              f"{s.predicate.value} "
              f"{obj.label if obj else s.obj} "
              f"(confidence: {s.confidence:.2f})")
    
    print("\n8. GRAPH STATISTICS")
    print("-" * 70)
    
    stats = agent.get_statistics()
    print(f"   Total entities: {stats['entities']}")
    print(f"   Total triples: {stats['triples']}")
    print(f"   Entity types: {stats['entity_types']}")
    print(f"   Queries executed: {stats['queries_executed']}")
    print(f"   Inferences made: {stats['inferences_made']}")
    
    print("\n9. GRAPH EXPLORATION")
    print("-" * 70)
    print("   Exploring Alice's neighborhood...")
    
    alice_entity = agent.graph.get_entity_by_label("Alice")
    if alice_entity:
        neighbors = agent.graph.get_neighbors(alice_entity.entity_id)
        print(f"   Alice has {len(neighbors)} connections:")
        for neighbor, triple in neighbors[:5]:
            print(f"     - {neighbor.label} (via {triple.predicate.value})")
    
    print("\n" + "=" * 70)
    print("🎉 MILESTONE ACHIEVED: 110 PATTERNS COMPLETE! 🎉")
    print("=" * 70)
    print("\nProgress: 110/170 patterns (64.7%)")
    print("\nKey Features Demonstrated:")
    print("1. Entity creation with types and properties")
    print("2. Triple-based fact representation")
    print("3. Natural language question answering")
    print("4. Path finding between entities")
    print("5. Type inference through relationships")
    print("6. Knowledge graph completion")
    print("7. Graph statistics and monitoring")
    print("8. Neighborhood exploration")
    print("9. Semantic querying and reasoning")
    print("\n🚀 Ready for the next 60 patterns! 🚀")


if __name__ == "__main__":
    demonstrate_knowledge_graph()
