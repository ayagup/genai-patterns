"""
Pattern 029: Semantic Memory Networks

Description:
    Semantic memory networks represent structured knowledge as graphs with entities,
    relationships, and attributes. This pattern enables sophisticated reasoning over
    interconnected concepts, supports graph-based queries, and facilitates knowledge
    discovery through relationship traversal.

Components:
    - Entity: Nodes representing concepts, objects, or individuals
    - Relationship: Edges connecting entities with typed connections
    - Attributes: Properties associated with entities
    - Graph Store: Storage and indexing of the knowledge graph
    - Query Engine: Traversal and pattern matching over the graph
    - Reasoning Module: Inference and relationship discovery

Use Cases:
    - Knowledge management systems
    - Question answering over structured knowledge
    - Relationship discovery and analysis
    - Semantic search and retrieval
    - Expert systems and decision support

LangChain Implementation:
    Uses graph data structures, relationship modeling, and LLM-based query
    interpretation to create a semantic network that supports both explicit
    knowledge storage and implicit reasoning.
"""

import os
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
import json
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class RelationType(Enum):
    """Types of relationships between entities."""
    IS_A = "is_a"  # Inheritance/taxonomy
    PART_OF = "part_of"  # Composition
    RELATED_TO = "related_to"  # General association
    CAUSES = "causes"  # Causal relationship
    PRECEDES = "precedes"  # Temporal relationship
    LOCATED_IN = "located_in"  # Spatial relationship
    OWNS = "owns"  # Ownership
    CREATED_BY = "created_by"  # Authorship
    USES = "uses"  # Functional relationship
    SIMILAR_TO = "similar_to"  # Similarity


@dataclass
class Entity:
    """Represents a node in the semantic network."""
    id: str
    name: str
    type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Entity) and self.id == other.id


@dataclass
class Relationship:
    """Represents an edge in the semantic network."""
    source_id: str
    target_id: str
    type: RelationType
    attributes: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class QueryResult:
    """Result of a semantic query."""
    entities: List[Entity]
    relationships: List[Relationship]
    paths: List[List[str]] = field(default_factory=list)
    explanation: str = ""


class SemanticGraph:
    """
    Graph data structure for storing semantic knowledge.
    
    Supports efficient entity lookup, relationship traversal,
    and pattern matching operations.
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        # Adjacency lists for efficient traversal
        self.outgoing: Dict[str, List[Relationship]] = defaultdict(list)
        self.incoming: Dict[str, List[Relationship]] = defaultdict(list)
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        self.entities[entity.id] = entity
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph."""
        # Validate entities exist
        if relationship.source_id not in self.entities:
            raise ValueError(f"Source entity {relationship.source_id} not found")
        if relationship.target_id not in self.entities:
            raise ValueError(f"Target entity {relationship.target_id} not found")
        
        self.relationships.append(relationship)
        self.outgoing[relationship.source_id].append(relationship)
        self.incoming[relationship.target_id].append(relationship)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        return self.entities.get(entity_id)
    
    def get_related_entities(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "outgoing"
    ) -> List[Entity]:
        """
        Get entities related to the given entity.
        
        Args:
            entity_id: Source entity ID
            relation_type: Optional filter by relationship type
            direction: "outgoing", "incoming", or "both"
        """
        related = []
        
        if direction in ["outgoing", "both"]:
            for rel in self.outgoing[entity_id]:
                if relation_type is None or rel.type == relation_type:
                    related.append(self.entities[rel.target_id])
        
        if direction in ["incoming", "both"]:
            for rel in self.incoming[entity_id]:
                if relation_type is None or rel.type == relation_type:
                    related.append(self.entities[rel.source_id])
        
        return related
    
    def find_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """
        Find a path between two entities using BFS.
        
        Returns list of entity IDs forming the path, or None if no path exists.
        """
        if start_id not in self.entities or end_id not in self.entities:
            return None
        
        if start_id == end_id:
            return [start_id]
        
        # BFS
        queue = [(start_id, [start_id])]
        visited = {start_id}
        
        while queue:
            current_id, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            # Explore neighbors
            for rel in self.outgoing[current_id]:
                neighbor_id = rel.target_id
                
                if neighbor_id == end_id:
                    return path + [neighbor_id]
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        return None
    
    def get_subgraph(
        self,
        entity_ids: List[str],
        depth: int = 1
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract a subgraph around given entities up to specified depth.
        """
        included_entities = set(entity_ids)
        
        # Expand to neighbors
        for _ in range(depth):
            new_entities = set()
            for entity_id in included_entities:
                neighbors = self.get_related_entities(entity_id, direction="both")
                new_entities.update(e.id for e in neighbors)
            included_entities.update(new_entities)
        
        # Get all entities
        entities = [self.entities[eid] for eid in included_entities if eid in self.entities]
        
        # Get relationships between included entities
        relationships = [
            rel for rel in self.relationships
            if rel.source_id in included_entities and rel.target_id in included_entities
        ]
        
        return entities, relationships
    
    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None
    ) -> List[Entity]:
        """
        Search entities by name or attributes.
        Simple keyword-based search.
        """
        query_lower = query.lower()
        results = []
        
        for entity in self.entities.values():
            # Filter by type if specified
            if entity_type and entity.type != entity_type:
                continue
            
            # Check name
            if query_lower in entity.name.lower():
                results.append(entity)
                continue
            
            # Check attributes
            for value in entity.attributes.values():
                if isinstance(value, str) and query_lower in value.lower():
                    results.append(entity)
                    break
        
        return results


class SemanticMemoryAgent:
    """
    Agent that maintains and reasons over a semantic memory network.
    
    Capabilities:
    - Extract entities and relationships from text
    - Store knowledge in graph structure
    - Answer questions using graph traversal
    - Discover implicit relationships
    """
    
    def __init__(self, temperature: float = 0.1):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        self.graph = SemanticGraph()
        
        # Prompt for entity extraction
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting structured knowledge from text.
Extract entities and relationships from the following text.

Format your response as JSON:
{{
  "entities": [
    {{"id": "unique_id", "name": "Entity Name", "type": "entity_type", "attributes": {{}}}},
    ...
  ],
  "relationships": [
    {{"source": "source_id", "target": "target_id", "type": "relationship_type"}},
    ...
  ]
}}

Valid relationship types: is_a, part_of, related_to, causes, precedes, located_in, owns, created_by, uses, similar_to

Be specific and extract all meaningful entities and relationships."""),
            ("user", "{text}")
        ])
        
        # Prompt for query understanding
        self.query_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at answering questions using a knowledge graph.

Given the following knowledge graph elements:
{graph_context}

Answer this question: {question}

Provide a clear, concise answer based on the graph information."""),
            ("user", "{question}")
        ])
    
    def add_knowledge(self, text: str) -> Dict[str, Any]:
        """
        Extract and add knowledge from text to the semantic network.
        """
        # Extract entities and relationships
        chain = self.extraction_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"text": text})
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = result[start_idx:end_idx]
                knowledge = json.loads(json_str)
            else:
                knowledge = json.loads(result)
        except json.JSONDecodeError:
            # If parsing fails, return error
            return {
                "success": False,
                "error": "Failed to parse extracted knowledge",
                "raw_response": result
            }
        
        # Add entities
        entities_added = 0
        for entity_data in knowledge.get("entities", []):
            entity = Entity(
                id=entity_data["id"],
                name=entity_data["name"],
                type=entity_data["type"],
                attributes=entity_data.get("attributes", {})
            )
            self.graph.add_entity(entity)
            entities_added += 1
        
        # Add relationships
        relationships_added = 0
        for rel_data in knowledge.get("relationships", []):
            try:
                # Map string to enum
                rel_type_str = rel_data["type"].upper()
                rel_type = RelationType[rel_type_str]
                
                relationship = Relationship(
                    source_id=rel_data["source"],
                    target_id=rel_data["target"],
                    type=rel_type
                )
                self.graph.add_relationship(relationship)
                relationships_added += 1
            except (KeyError, ValueError) as e:
                # Skip invalid relationships
                continue
        
        return {
            "success": True,
            "entities_added": entities_added,
            "relationships_added": relationships_added
        }
    
    def query(self, question: str) -> str:
        """
        Answer a question using the semantic network.
        """
        # Search for relevant entities in the question
        words = question.lower().split()
        relevant_entities = set()
        
        for word in words:
            entities = self.graph.search_entities(word)
            relevant_entities.update(e.id for e in entities)
        
        if not relevant_entities:
            return "I don't have enough knowledge to answer that question."
        
        # Get subgraph around relevant entities
        entities, relationships = self.graph.get_subgraph(list(relevant_entities), depth=2)
        
        # Format graph context
        graph_context = self._format_graph_context(entities, relationships)
        
        # Generate answer
        chain = self.query_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "graph_context": graph_context,
            "question": question
        })
        
        return answer
    
    def find_relationship(
        self,
        entity1: str,
        entity2: str
    ) -> Optional[List[str]]:
        """
        Find how two entities are related through the graph.
        """
        # Find entities by name
        entities1 = self.graph.search_entities(entity1)
        entities2 = self.graph.search_entities(entity2)
        
        if not entities1 or not entities2:
            return None
        
        # Find shortest path
        path = self.graph.find_path(entities1[0].id, entities2[0].id)
        
        if path:
            # Convert IDs to names
            return [self.graph.get_entity(eid).name for eid in path]
        
        return None
    
    def get_related_concepts(
        self,
        concept: str,
        relation_type: Optional[str] = None
    ) -> List[str]:
        """
        Get concepts related to the given concept.
        """
        entities = self.graph.search_entities(concept)
        
        if not entities:
            return []
        
        # Convert relation type string to enum if provided
        rel_type_enum = None
        if relation_type:
            try:
                rel_type_enum = RelationType[relation_type.upper()]
            except KeyError:
                pass
        
        # Get related entities
        related = self.graph.get_related_entities(
            entities[0].id,
            relation_type=rel_type_enum,
            direction="both"
        )
        
        return [e.name for e in related]
    
    def _format_graph_context(
        self,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> str:
        """Format graph elements as readable context."""
        lines = ["Entities:"]
        for entity in entities[:10]:  # Limit to avoid token overflow
            attrs_str = ", ".join(f"{k}={v}" for k, v in entity.attributes.items())
            lines.append(f"  - {entity.name} ({entity.type})" + (f": {attrs_str}" if attrs_str else ""))
        
        lines.append("\nRelationships:")
        for rel in relationships[:20]:  # Limit relationships
            source = self.graph.get_entity(rel.source_id)
            target = self.graph.get_entity(rel.target_id)
            if source and target:
                lines.append(f"  - {source.name} {rel.type.value} {target.name}")
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the semantic network."""
        entity_types = defaultdict(int)
        relation_types = defaultdict(int)
        
        for entity in self.graph.entities.values():
            entity_types[entity.type] += 1
        
        for rel in self.graph.relationships:
            relation_types[rel.type.value] += 1
        
        return {
            "total_entities": len(self.graph.entities),
            "total_relationships": len(self.graph.relationships),
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types)
        }


def demonstrate_semantic_memory_networks():
    """
    Demonstrates semantic memory networks with knowledge extraction,
    graph-based queries, and relationship discovery.
    """
    print("=" * 80)
    print("SEMANTIC MEMORY NETWORKS DEMONSTRATION")
    print("=" * 80)
    
    # Create agent
    agent = SemanticMemoryAgent()
    
    # Test 1: Build knowledge base
    print("\n" + "=" * 80)
    print("Test 1: Building Knowledge Base")
    print("=" * 80)
    
    knowledge_texts = [
        """
        Python is a programming language created by Guido van Rossum.
        Python is used for web development, data analysis, and artificial intelligence.
        Python is similar to Ruby and JavaScript in terms of ease of use.
        """,
        """
        Machine Learning is a part of Artificial Intelligence.
        Machine Learning uses algorithms to analyze data and make predictions.
        TensorFlow and PyTorch are popular Machine Learning frameworks.
        TensorFlow was created by Google.
        """,
        """
        San Francisco is located in California.
        California is part of the United States.
        San Francisco is known for its technology industry.
        Many tech companies are located in San Francisco.
        """
    ]
    
    for i, text in enumerate(knowledge_texts, 1):
        print(f"\nAdding knowledge batch {i}...")
        result = agent.add_knowledge(text)
        if result["success"]:
            print(f"✓ Added {result['entities_added']} entities and {result['relationships_added']} relationships")
        else:
            print(f"✗ Error: {result.get('error', 'Unknown error')}")
    
    # Show statistics
    stats = agent.get_statistics()
    print(f"\nKnowledge base statistics:")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Total relationships: {stats['total_relationships']}")
    print(f"  Entity types: {stats['entity_types']}")
    print(f"  Relation types: {stats['relation_types']}")
    
    # Test 2: Query knowledge
    print("\n" + "=" * 80)
    print("Test 2: Querying Knowledge Base")
    print("=" * 80)
    
    questions = [
        "What is Python used for?",
        "Who created Python?",
        "What are some Machine Learning frameworks?",
        "Where is San Francisco located?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        answer = agent.query(question)
        print(f"Answer: {answer}")
    
    # Test 3: Relationship discovery
    print("\n" + "=" * 80)
    print("Test 3: Discovering Relationships")
    print("=" * 80)
    
    concept_pairs = [
        ("Python", "Artificial Intelligence"),
        ("TensorFlow", "Google"),
        ("San Francisco", "United States")
    ]
    
    for concept1, concept2 in concept_pairs:
        print(f"\nFinding relationship between '{concept1}' and '{concept2}':")
        path = agent.find_relationship(concept1, concept2)
        if path:
            print(f"  Path: {' → '.join(path)}")
        else:
            print("  No direct relationship found")
    
    # Test 4: Related concepts
    print("\n" + "=" * 80)
    print("Test 4: Finding Related Concepts")
    print("=" * 80)
    
    concepts = ["Python", "Machine Learning", "San Francisco"]
    
    for concept in concepts:
        print(f"\nConcepts related to '{concept}':")
        related = agent.get_related_concepts(concept)
        if related:
            for rel_concept in related[:5]:  # Show top 5
                print(f"  - {rel_concept}")
        else:
            print("  No related concepts found")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Semantic Memory Networks provide:
✓ Structured knowledge representation as graphs
✓ Entity and relationship extraction from text
✓ Graph-based querying and reasoning
✓ Relationship discovery through path finding
✓ Concept exploration through traversal

This pattern excels at:
- Knowledge management and organization
- Question answering over structured data
- Discovering implicit connections
- Semantic search and retrieval
- Building expert systems

Key advantages:
1. Explicit relationship modeling
2. Efficient graph traversal
3. Multi-hop reasoning support
4. Scalable knowledge storage
5. Interpretable knowledge representation

Use semantic networks when you need to:
- Maintain structured knowledge bases
- Answer complex relational queries
- Discover connections between concepts
- Build domain-specific knowledge systems
- Support inference and reasoning tasks
""")


if __name__ == "__main__":
    demonstrate_semantic_memory_networks()
