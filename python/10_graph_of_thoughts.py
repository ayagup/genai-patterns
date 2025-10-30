"""
Graph-of-Thoughts (GoT) Pattern Implementation

This module demonstrates the Graph-of-Thoughts pattern where reasoning is represented
as a directed graph with nodes (thoughts) and edges (transformations). Unlike tree
structures, graphs allow for non-linear thinking and flexible thought combinations.

Key Components:
- ThoughtNode: Individual reasoning step with connections
- ThoughtGraph: Directed graph structure for complex reasoning
- Transformation edges between thoughts
- Aggregation and synthesis capabilities
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum
import uuid
import random
import json


class EdgeType(Enum):
    """Types of connections between thoughts"""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REFINES = "refines"
    COMBINES = "combines"
    LEADS_TO = "leads_to"
    DEPENDS_ON = "depends_on"
    ALTERNATIVE = "alternative"


@dataclass
class ThoughtNode:
    """A single thought/reasoning step in the graph"""
    id: str
    content: str
    confidence: float = 0.5
    depth: int = 0
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class ThoughtEdge:
    """Connection between two thoughts"""
    from_node: str
    to_node: str
    edge_type: EdgeType
    weight: float = 1.0
    reasoning: str = ""


class ThoughtGraph:
    """Graph structure for complex reasoning patterns"""
    
    def __init__(self):
        self.nodes: Dict[str, ThoughtNode] = {}
        self.edges: List[ThoughtEdge] = []
        self.adjacency_list: Dict[str, List[str]] = {}
        self.reverse_adjacency: Dict[str, List[str]] = {}
    
    def add_node(self, thought: ThoughtNode) -> str:
        """Add a thought node to the graph"""
        self.nodes[thought.id] = thought
        if thought.id not in self.adjacency_list:
            self.adjacency_list[thought.id] = []
        if thought.id not in self.reverse_adjacency:
            self.reverse_adjacency[thought.id] = []
        return thought.id
    
    def add_edge(self, edge: ThoughtEdge) -> bool:
        """Add an edge between two thoughts"""
        if edge.from_node not in self.nodes or edge.to_node not in self.nodes:
            return False
        
        self.edges.append(edge)
        self.adjacency_list[edge.from_node].append(edge.to_node)
        self.reverse_adjacency[edge.to_node].append(edge.from_node)
        return True
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all nodes connected from this node"""
        return self.adjacency_list.get(node_id, [])
    
    def get_predecessors(self, node_id: str) -> List[str]:
        """Get all nodes that connect to this node"""
        return self.reverse_adjacency.get(node_id, [])
    
    def get_edge(self, from_node: str, to_node: str) -> Optional[ThoughtEdge]:
        """Get edge between two nodes"""
        for edge in self.edges:
            if edge.from_node == from_node and edge.to_node == to_node:
                return edge
        return None
    
    def find_paths(self, start: str, end: str, max_depth: int = 5) -> List[List[str]]:
        """Find all paths between two nodes"""
        paths = []
        
        def dfs(current: str, target: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            if current == target:
                paths.append(path + [current])
                return
            if current in path:  # Avoid cycles
                return
            
            for neighbor in self.get_neighbors(current):
                dfs(neighbor, target, path + [current], depth + 1)
        
        dfs(start, end, [], 0)
        return paths
    
    def get_strongly_connected_components(self) -> List[List[str]]:
        """Find strongly connected components (thought clusters)"""
        visited = set()
        finished = []
        
        def dfs1(node: str):
            visited.add(node)
            for neighbor in self.get_neighbors(node):
                if neighbor not in visited:
                    dfs1(neighbor)
            finished.append(node)
        
        # First DFS to get finishing times
        for node in self.nodes:
            if node not in visited:
                dfs1(node)
        
        # Second DFS on reversed graph
        visited.clear()
        components = []
        
        def dfs2(node: str, component: List[str]):
            visited.add(node)
            component.append(node)
            for predecessor in self.get_predecessors(node):
                if predecessor not in visited:
                    dfs2(predecessor, component)
        
        for node in reversed(finished):
            if node not in visited:
                component = []
                dfs2(node, component)
                if component:
                    components.append(component)
        
        return components


class GraphOfThoughtsAgent:
    """Agent that uses graph-based reasoning"""
    
    def __init__(self, max_nodes: int = 20, confidence_threshold: float = 0.7):
        self.max_nodes = max_nodes
        self.confidence_threshold = confidence_threshold
        self.current_graph = ThoughtGraph()
        self.reasoning_history: List[ThoughtGraph] = []
    
    def _generate_initial_thoughts(self, problem: str, num_thoughts: int = 3) -> List[ThoughtNode]:
        """Generate initial thoughts about the problem"""
        thought_templates = [
            "Let me analyze this problem from multiple angles: {problem}",
            "I should consider the key constraints and requirements: {problem}",
            "What are the possible approaches to solve: {problem}",
            "Let me break down the components of: {problem}",
            "I need to understand the underlying principles of: {problem}"
        ]
        
        thoughts = []
        for i in range(num_thoughts):
            template = random.choice(thought_templates)
            content = template.format(problem=problem)
            
            thought = ThoughtNode(
                id=f"init_{i}",
                content=content,
                confidence=random.uniform(0.4, 0.7),
                depth=0,
                metadata={"type": "initial", "problem": problem}
            )
            thoughts.append(thought)
        
        return thoughts
    
    def _expand_thought(self, node: ThoughtNode, problem_context: str) -> List[ThoughtNode]:
        """Generate new thoughts based on an existing thought"""
        expansion_strategies = [
            "Building on this idea: {content}",
            "An alternative perspective: {content}",
            "This leads me to think: {content}",
            "A refinement of this thought: {content}",
            "Combining this with other ideas: {content}"
        ]
        
        new_thoughts = []
        num_expansions = random.randint(1, 3)
        
        for i in range(num_expansions):
            strategy = random.choice(expansion_strategies)
            new_content = strategy.format(content=node.content[:50] + "...")
            
            new_thought = ThoughtNode(
                id=f"exp_{node.id}_{i}",
                content=new_content,
                confidence=min(node.confidence + random.uniform(-0.2, 0.3), 1.0),
                depth=node.depth + 1,
                metadata={"type": "expansion", "parent": node.id}
            )
            new_thoughts.append(new_thought)
        
        return new_thoughts
    
    def _determine_edge_type(self, from_node: ThoughtNode, to_node: ThoughtNode) -> EdgeType:
        """Determine the type of relationship between two thoughts"""
        # Simple heuristic based on content similarity and metadata
        if to_node.metadata.get("parent") == from_node.id:
            return EdgeType.LEADS_TO
        elif "alternative" in to_node.content.lower():
            return EdgeType.ALTERNATIVE
        elif "refine" in to_node.content.lower():
            return EdgeType.REFINES
        elif "combine" in to_node.content.lower():
            return EdgeType.COMBINES
        elif to_node.confidence > from_node.confidence:
            return EdgeType.SUPPORTS
        else:
            return EdgeType.DEPENDS_ON
    
    def _evaluate_graph_coherence(self) -> float:
        """Evaluate how coherent and well-connected the thought graph is"""
        if len(self.current_graph.nodes) < 2:
            return 0.0
        
        # Calculate connectivity score
        total_possible_edges = len(self.current_graph.nodes) * (len(self.current_graph.nodes) - 1)
        actual_edges = len(self.current_graph.edges)
        connectivity_score = actual_edges / max(total_possible_edges, 1)
        
        # Calculate confidence score
        avg_confidence = sum(node.confidence for node in self.current_graph.nodes.values()) / len(self.current_graph.nodes)
        
        # Calculate depth diversity
        depths = [node.depth for node in self.current_graph.nodes.values()]
        depth_diversity = len(set(depths)) / max(len(depths), 1)
        
        return (connectivity_score * 0.4 + avg_confidence * 0.4 + depth_diversity * 0.2)
    
    def _synthesize_conclusion(self) -> str:
        """Synthesize a final conclusion from the thought graph"""
        # Find nodes with highest confidence
        high_confidence_nodes = [
            node for node in self.current_graph.nodes.values()
            if node.confidence >= self.confidence_threshold
        ]
        
        if not high_confidence_nodes:
            high_confidence_nodes = sorted(
                self.current_graph.nodes.values(),
                key=lambda x: x.confidence,
                reverse=True
            )[:3]
        
        # Find strongly connected components
        components = self.current_graph.get_strongly_connected_components()
        
        conclusion_parts = []
        
        # Add insights from high-confidence nodes
        for node in high_confidence_nodes[:3]:
            conclusion_parts.append(f"Key insight: {node.content}")
        
        # Add insights from connected components
        if components:
            largest_component = max(components, key=len)
            if len(largest_component) > 1:
                conclusion_parts.append(f"Connected reasoning involves {len(largest_component)} related thoughts")
        
        # Calculate overall reasoning quality
        coherence = self._evaluate_graph_coherence()
        conclusion_parts.append(f"Reasoning coherence: {coherence:.2f}")
        
        return "\n".join(conclusion_parts)
    
    def solve(self, problem: str, max_iterations: int = 5) -> str:
        """Solve a problem using graph-based reasoning"""
        print(f"\n🕸️ Graph-of-Thoughts Agent solving: {problem}")
        print("=" * 80)
        
        # Initialize graph
        self.current_graph = ThoughtGraph()
        
        # Generate initial thoughts
        initial_thoughts = self._generate_initial_thoughts(problem)
        print(f"\n📍 Generated {len(initial_thoughts)} initial thoughts:")
        
        for thought in initial_thoughts:
            self.current_graph.add_node(thought)
            print(f"  💭 {thought.id}: {thought.content}")
            print(f"      Confidence: {thought.confidence:.2f}")
        
        # Iteratively expand and connect thoughts
        for iteration in range(max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}")
            print("-" * 40)
            
            if len(self.current_graph.nodes) >= self.max_nodes:
                print("⚠️ Maximum nodes reached")
                break
            
            # Select nodes to expand (prefer high confidence, low depth)
            expandable_nodes = [
                node for node in self.current_graph.nodes.values()
                if node.depth < 3 and node.confidence > 0.3
            ]
            
            if not expandable_nodes:
                print("ℹ️ No more expandable nodes")
                break
            
            # Sort by potential (confidence - depth penalty)
            expandable_nodes.sort(key=lambda x: x.confidence - (x.depth * 0.1), reverse=True)
            selected_node = expandable_nodes[0]
            
            print(f"🎯 Expanding node: {selected_node.id}")
            
            # Generate new thoughts
            new_thoughts = self._expand_thought(selected_node, problem)
            
            for new_thought in new_thoughts:
                if len(self.current_graph.nodes) >= self.max_nodes:
                    break
                
                self.current_graph.add_node(new_thought)
                print(f"  ➕ {new_thought.id}: {new_thought.content}")
                print(f"      Confidence: {new_thought.confidence:.2f}")
                
                # Create edge from parent to new thought
                edge_type = self._determine_edge_type(selected_node, new_thought)
                edge = ThoughtEdge(
                    from_node=selected_node.id,
                    to_node=new_thought.id,
                    edge_type=edge_type,
                    weight=new_thought.confidence,
                    reasoning=f"Generated from {selected_node.id}"
                )
                self.current_graph.add_edge(edge)
                print(f"      Edge: {edge_type.value}")
                
                # Potentially connect to other existing nodes
                for existing_node in self.current_graph.nodes.values():
                    if (existing_node.id != new_thought.id and 
                        existing_node.id != selected_node.id and
                        random.random() < 0.3):  # 30% chance of connection
                        
                        connection_edge = ThoughtEdge(
                            from_node=existing_node.id,
                            to_node=new_thought.id,
                            edge_type=self._determine_edge_type(existing_node, new_thought),
                            weight=0.5,
                            reasoning="Cross-connection"
                        )
                        self.current_graph.add_edge(connection_edge)
                        print(f"      Cross-connected to: {existing_node.id}")
        
        # Analyze graph structure
        print(f"\n📊 Graph Analysis:")
        print(f"   Nodes: {len(self.current_graph.nodes)}")
        print(f"   Edges: {len(self.current_graph.edges)}")
        print(f"   Coherence: {self._evaluate_graph_coherence():.2f}")
        
        # Find strongly connected components
        components = self.current_graph.get_strongly_connected_components()
        print(f"   Connected components: {len(components)}")
        
        # Synthesize conclusion
        conclusion = self._synthesize_conclusion()
        print(f"\n✅ Conclusion:")
        print(conclusion)
        
        # Store in history
        self.reasoning_history.append(self.current_graph)
        
        return conclusion
    
    def visualize_graph(self) -> Dict[str, Any]:
        """Create a visualization-friendly representation of the graph"""
        nodes_data = []
        edges_data = []
        
        for node in self.current_graph.nodes.values():
            nodes_data.append({
                "id": node.id,
                "content": node.content[:50] + "..." if len(node.content) > 50 else node.content,
                "confidence": node.confidence,
                "depth": node.depth,
                "type": node.metadata.get("type", "unknown")
            })
        
        for edge in self.current_graph.edges:
            edges_data.append({
                "from": edge.from_node,
                "to": edge.to_node,
                "type": edge.edge_type.value,
                "weight": edge.weight,
                "reasoning": edge.reasoning
            })
        
        return {
            "nodes": nodes_data,
            "edges": edges_data,
            "metrics": {
                "total_nodes": len(self.current_graph.nodes),
                "total_edges": len(self.current_graph.edges),
                "coherence": self._evaluate_graph_coherence(),
                "components": len(self.current_graph.get_strongly_connected_components())
            }
        }
    
    def find_reasoning_paths(self, start_concept: str, end_concept: str) -> List[List[str]]:
        """Find reasoning paths between two concepts"""
        # Find nodes that match the concepts
        start_nodes = [
            node.id for node in self.current_graph.nodes.values()
            if start_concept.lower() in node.content.lower()
        ]
        end_nodes = [
            node.id for node in self.current_graph.nodes.values()
            if end_concept.lower() in node.content.lower()
        ]
        
        all_paths = []
        for start_node in start_nodes:
            for end_node in end_nodes:
                paths = self.current_graph.find_paths(start_node, end_node)
                all_paths.extend(paths)
        
        return all_paths


def main():
    """Demonstration of the Graph-of-Thoughts pattern"""
    print("🕸️ Graph-of-Thoughts Pattern Demonstration")
    print("=" * 80)
    print("This demonstrates reasoning as a directed graph where:")
    print("- Thoughts are nodes with connections")
    print("- Edges represent relationships between thoughts")
    print("- Non-linear reasoning paths are possible")
    print("- Synthesis emerges from graph analysis")
    
    # Create agent
    agent = GraphOfThoughtsAgent(max_nodes=15, confidence_threshold=0.6)
    
    # Test problems
    test_problems = [
        "How can we design a sustainable transportation system for urban areas?",
        "What are the key factors in developing artificial general intelligence?",
        "How should we balance economic growth with environmental protection?"
    ]
    
    for i, problem in enumerate(test_problems, 1):
        print(f"\n\n🔍 Test Case {i}")
        print("=" * 80)
        
        result = agent.solve(problem, max_iterations=4)
        
        print(f"\n📈 Graph Visualization Data:")
        viz_data = agent.visualize_graph()
        print(f"Summary: {viz_data['metrics']}")
        
        # Show some example reasoning paths if available
        if len(agent.current_graph.nodes) > 2:
            nodes = list(agent.current_graph.nodes.keys())
            if len(nodes) >= 2:
                paths = agent.current_graph.find_paths(nodes[0], nodes[-1])
                if paths:
                    print(f"\n🛤️ Example reasoning path:")
                    for step in paths[0]:
                        node = agent.current_graph.nodes[step]
                        print(f"   {step}: {node.content[:60]}...")
    
    print("\n\n🎯 Key Graph-of-Thoughts Features Demonstrated:")
    print("✅ Non-linear thought connections")
    print("✅ Multiple relationship types between thoughts")
    print("✅ Graph analysis for reasoning quality")
    print("✅ Strongly connected component identification")
    print("✅ Path finding between concepts")
    print("✅ Emergent synthesis from graph structure")
    print("✅ Iterative graph construction")
    print("✅ Cross-connections between thoughts")


if __name__ == "__main__":
    main()