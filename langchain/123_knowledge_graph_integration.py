"""
Pattern 123: Knowledge Graph Integration

Description:
    Uses structured knowledge graphs for reasoning with graph traversal,
    pattern matching, and inference capabilities.

Components:
    - Knowledge graph
    - Graph traversal
    - Pattern matching
    - Inference engine

Use Cases:
    - Complex reasoning
    - Knowledge-intensive tasks
    - Relationship discovery

LangChain Implementation:
    Integrates knowledge graphs with LLM reasoning for enhanced understanding.
"""

import os
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass
class Node:
    """Represents a node in the knowledge graph."""
    id: str
    type: str
    properties: Dict[str, Any]


@dataclass
class Edge:
    """Represents an edge in the knowledge graph."""
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class KnowledgeGraph:
    """Knowledge graph for structured knowledge representation."""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        
    def add_node(self, node: Node):
        """Add a node to the graph."""
        self.nodes[node.id] = node
        
    def add_edge(self, edge: Edge):
        """Add an edge to the graph."""
        self.edges.append(edge)
        
    def get_neighbors(self, node_id: str, relationship: str = None) -> List[Tuple[Node, str]]:
        """Get neighboring nodes."""
        neighbors = []
        for edge in self.edges:
            if edge.source == node_id:
                if relationship is None or edge.relationship == relationship:
                    if edge.target in self.nodes:
                        neighbors.append((self.nodes[edge.target], edge.relationship))
        return neighbors
    
    def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> List[str]:
        """Find path between two nodes."""
        visited = set()
        queue = [(start_id, [start_id])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current == end_id:
                return path
            
            if len(path) >= max_depth:
                continue
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor, _ in self.get_neighbors(current):
                if neighbor.id not in visited:
                    queue.append((neighbor.id, path + [neighbor.id]))
        
        return []
    
    def query_subgraph(self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        """Extract subgraph around a node."""
        visited = set()
        subgraph_nodes = {}
        subgraph_edges = []
        
        def explore(nid, d):
            if d > depth or nid in visited:
                return
            visited.add(nid)
            
            if nid in self.nodes:
                subgraph_nodes[nid] = self.nodes[nid]
                
                for neighbor, relationship in self.get_neighbors(nid):
                    edge = Edge(nid, neighbor.id, relationship)
                    subgraph_edges.append(edge)
                    explore(neighbor.id, d + 1)
        
        explore(node_id, 0)
        
        return {
            "nodes": subgraph_nodes,
            "edges": subgraph_edges
        }


class KnowledgeGraphAgent:
    """Agent that uses knowledge graph for reasoning."""
    
    def __init__(self, kg: KnowledgeGraph, model_name: str = "gpt-4"):
        self.kg = kg
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        
    def answer_question(self, question: str) -> str:
        """Answer question using knowledge graph."""
        # Extract relevant entities from question
        entity_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract key entities from this question."),
            ("user", "Question: {question}\n\nEntities (comma-separated):")
        ])
        
        chain = entity_extraction_prompt | self.llm | StrOutputParser()
        entities_str = chain.invoke({"question": question})
        entities = [e.strip() for e in entities_str.split(",")]
        
        # Find relevant nodes
        relevant_nodes = []
        for entity in entities:
            for node_id, node in self.kg.nodes.items():
                if entity.lower() in node.properties.get("name", "").lower():
                    relevant_nodes.append(node)
        
        # Extract relevant subgraph
        kg_context = ""
        for node in relevant_nodes[:3]:  # Limit to top 3
            subgraph = self.kg.query_subgraph(node.id, depth=2)
            
            # Format subgraph info
            node_info = f"\nEntity: {node.properties.get('name', node.id)}\n"
            for edge in subgraph['edges']:
                if edge.source in subgraph['nodes'] and edge.target in subgraph['nodes']:
                    source_name = subgraph['nodes'][edge.source].properties.get('name', edge.source)
                    target_name = subgraph['nodes'][edge.target].properties.get('name', edge.target)
                    node_info += f"  {source_name} --[{edge.relationship}]--> {target_name}\n"
            
            kg_context += node_info
        
        # Answer using knowledge graph context
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question using the knowledge graph information."),
            ("user", """Knowledge Graph Context:
{kg_context}

Question: {question}

Answer:""")
        ])
        
        chain = qa_prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "kg_context": kg_context or "No relevant information found in knowledge graph",
            "question": question
        })
    
    def infer_relationships(self, entity1: str, entity2: str) -> str:
        """Infer relationships between entities."""
        # Find nodes
        node1 = None
        node2 = None
        
        for node_id, node in self.kg.nodes.items():
            if entity1.lower() in node.properties.get("name", "").lower():
                node1 = node
            if entity2.lower() in node.properties.get("name", "").lower():
                node2 = node
        
        if not node1 or not node2:
            return f"Could not find entities in knowledge graph"
        
        # Find path
        path = self.kg.find_path(node1.id, node2.id)
        
        if not path:
            return f"No direct relationship found between {entity1} and {entity2}"
        
        # Construct relationship chain
        relationships = []
        for i in range(len(path) - 1):
            for edge in self.kg.edges:
                if edge.source == path[i] and edge.target == path[i+1]:
                    source_name = self.kg.nodes[edge.source].properties.get('name', edge.source)
                    target_name = self.kg.nodes[edge.target].properties.get('name', edge.target)
                    relationships.append(f"{source_name} --[{edge.relationship}]--> {target_name}")
        
        # Generate inference
        inference_prompt = ChatPromptTemplate.from_messages([
            ("system", "Explain the relationship between entities based on this knowledge graph path."),
            ("user", """Entity 1: {entity1}
Entity 2: {entity2}

Relationship Chain:
{chain}

Explain the relationship:""")
        ])
        
        chain_str = "\n".join(relationships)
        chain = inference_prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "entity1": entity1,
            "entity2": entity2,
            "chain": chain_str
        })
    
    def discover_patterns(self) -> str:
        """Discover patterns in the knowledge graph."""
        # Analyze graph structure
        structure_info = f"Total nodes: {len(self.kg.nodes)}\n"
        structure_info += f"Total edges: {len(self.kg.edges)}\n\n"
        
        # Common relationships
        relationship_counts = {}
        for edge in self.kg.edges:
            relationship_counts[edge.relationship] = relationship_counts.get(edge.relationship, 0) + 1
        
        structure_info += "Relationship types:\n"
        for rel, count in sorted(relationship_counts.items(), key=lambda x: x[1], reverse=True):
            structure_info += f"  {rel}: {count}\n"
        
        pattern_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze this knowledge graph structure and identify interesting patterns."),
            ("user", """Graph Structure:
{structure}

Identify:
1. Central entities (hubs)
2. Common patterns
3. Interesting insights
4. Potential gaps""")
        ])
        
        chain = pattern_prompt | self.llm | StrOutputParser()
        return chain.invoke({"structure": structure_info})


def demonstrate_knowledge_graph_integration():
    """Demonstrate knowledge graph integration pattern."""
    print("=== Knowledge Graph Integration Pattern ===\n")
    
    # Create knowledge graph
    print("1. Building Knowledge Graph")
    print("-" * 50)
    
    kg = KnowledgeGraph()
    
    # Add nodes (people, companies, technologies)
    kg.add_node(Node("person1", "Person", {"name": "Alice", "role": "CEO"}))
    kg.add_node(Node("person2", "Person", {"name": "Bob", "role": "CTO"}))
    kg.add_node(Node("person3", "Person", {"name": "Charlie", "role": "Engineer"}))
    
    kg.add_node(Node("company1", "Company", {"name": "TechCorp", "industry": "Software"}))
    kg.add_node(Node("company2", "Company", {"name": "DataInc", "industry": "Analytics"}))
    
    kg.add_node(Node("tech1", "Technology", {"name": "Python", "type": "Language"}))
    kg.add_node(Node("tech2", "Technology", {"name": "Machine Learning", "type": "Field"}))
    kg.add_node(Node("tech3", "Technology", {"name": "TensorFlow", "type": "Framework"}))
    
    # Add edges (relationships)
    kg.add_edge(Edge("person1", "company1", "WORKS_AT"))
    kg.add_edge(Edge("person2", "company1", "WORKS_AT"))
    kg.add_edge(Edge("person3", "company2", "WORKS_AT"))
    
    kg.add_edge(Edge("person1", "person2", "MANAGES"))
    kg.add_edge(Edge("person2", "person3", "MENTORS"))
    
    kg.add_edge(Edge("person2", "tech1", "EXPERT_IN"))
    kg.add_edge(Edge("person2", "tech2", "EXPERT_IN"))
    kg.add_edge(Edge("person3", "tech1", "USES"))
    kg.add_edge(Edge("person3", "tech3", "USES"))
    
    kg.add_edge(Edge("tech3", "tech1", "IMPLEMENTED_IN"))
    kg.add_edge(Edge("tech3", "tech2", "USED_FOR"))
    
    kg.add_edge(Edge("company1", "tech2", "SPECIALIZES_IN"))
    kg.add_edge(Edge("company2", "tech2", "USES"))
    
    print(f"✓ Created knowledge graph with {len(kg.nodes)} nodes and {len(kg.edges)} edges")
    print()
    
    # Create agent
    agent = KnowledgeGraphAgent(kg)
    
    # Question answering
    print("2. Question Answering with Knowledge Graph")
    print("-" * 50)
    
    questions = [
        "Who works at TechCorp?",
        "What technologies does Bob know?",
        "Which companies use Machine Learning?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = agent.answer_question(question)
        print(f"A: {answer}")
    
    # Relationship inference
    print("\n\n3. Inferring Relationships")
    print("-" * 50)
    
    print("\nInferring relationship between Alice and Charlie:")
    inference = agent.infer_relationships("Alice", "Charlie")
    print(inference)
    
    print("\nInferring relationship between Bob and TensorFlow:")
    inference = agent.infer_relationships("Bob", "TensorFlow")
    print(inference)
    
    # Pattern discovery
    print("\n\n4. Discovering Patterns in Knowledge Graph")
    print("-" * 50)
    patterns = agent.discover_patterns()
    print(patterns)
    
    print("\n=== Summary ===")
    print("Knowledge graph integration demonstrated with:")
    print("- Graph construction (nodes and edges)")
    print("- Graph traversal and querying")
    print("- Question answering with graph context")
    print("- Relationship inference")
    print("- Pattern discovery")
    print("- LLM-enhanced graph reasoning")


if __name__ == "__main__":
    demonstrate_knowledge_graph_integration()
