"""
Graph-of-Thoughts (GoT) Pattern
Represents reasoning as a directed graph with nodes and edges
"""
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
class ThoughtType(Enum):
    INITIAL = "initial"
    INTERMEDIATE = "intermediate"
    AGGREGATED = "aggregated"
    FINAL = "final"
class EdgeType(Enum):
    SEQUENTIAL = "sequential"  # A leads to B
    PARALLEL = "parallel"      # A and B are alternatives
    MERGE = "merge"            # Multiple thoughts combine
    REFINE = "refine"          # B refines A
@dataclass
class ThoughtNode:
    """A single thought/reasoning step"""
    id: str
    content: str
    thought_type: ThoughtType
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    def __hash__(self):
        return hash(self.id)
@dataclass
class ThoughtEdge:
    """Connection between thoughts"""
    from_node: str
    to_node: str
    edge_type: EdgeType
    weight: float = 1.0
    def __hash__(self):
        return hash((self.from_node, self.to_node))
class GraphOfThoughts:
    """Graph structure for complex reasoning"""
    def __init__(self):
        self.nodes: Dict[str, ThoughtNode] = {}
        self.edges: List[ThoughtEdge] = []
        self.adjacency: Dict[str, List[str]] = {}  # node_id -> [connected_node_ids]
        self.reverse_adjacency: Dict[str, List[str]] = {}  # for backtracking
    def add_node(self, content: str, thought_type: ThoughtType, 
                 score: float = 0.0, metadata: Dict[str, Any] = None) -> ThoughtNode:
        """Add a thought node to the graph"""
        node_id = str(uuid.uuid4())[:8]
        node = ThoughtNode(
            id=node_id,
            content=content,
            thought_type=thought_type,
            score=score,
            metadata=metadata or {}
        )
        self.nodes[node_id] = node
        self.adjacency[node_id] = []
        self.reverse_adjacency[node_id] = []
        return node
    def add_edge(self, from_node: ThoughtNode, to_node: ThoughtNode, 
                 edge_type: EdgeType, weight: float = 1.0):
        """Add an edge between thoughts"""
        edge = ThoughtEdge(
            from_node=from_node.id,
            to_node=to_node.id,
            edge_type=edge_type,
            weight=weight
        )
        self.edges.append(edge)
        self.adjacency[from_node.id].append(to_node.id)
        self.reverse_adjacency[to_node.id].append(from_node.id)
    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        """Get node by ID"""
        return self.nodes.get(node_id)
    def get_successors(self, node_id: str) -> List[ThoughtNode]:
        """Get all successor nodes"""
        return [self.nodes[nid] for nid in self.adjacency.get(node_id, [])]
    def get_predecessors(self, node_id: str) -> List[ThoughtNode]:
        """Get all predecessor nodes"""
        return [self.nodes[nid] for nid in self.reverse_adjacency.get(node_id, [])]
    def find_paths(self, start_id: str, end_id: str) -> List[List[ThoughtNode]]:
        """Find all paths from start to end"""
        paths = []
        def dfs(current_id: str, path: List[str], visited: Set[str]):
            if current_id == end_id:
                paths.append([self.nodes[nid] for nid in path])
                return
            for next_id in self.adjacency.get(current_id, []):
                if next_id not in visited:
                    dfs(next_id, path + [next_id], visited | {next_id})
        dfs(start_id, [start_id], {start_id})
        return paths
    def get_best_path(self, start_id: str, end_id: str) -> List[ThoughtNode]:
        """Get highest-scoring path"""
        paths = self.find_paths(start_id, end_id)
        if not paths:
            return []
        # Score each path
        path_scores = []
        for path in paths:
            score = sum(node.score for node in path) / len(path)
            path_scores.append((score, path))
        # Return best path
        return max(path_scores, key=lambda x: x[0])[1]
    def aggregate_nodes(self, node_ids: List[str], aggregation_method: str = "mean") -> ThoughtNode:
        """Aggregate multiple nodes into one"""
        nodes = [self.nodes[nid] for nid in node_ids]
        if aggregation_method == "mean":
            score = sum(n.score for n in nodes) / len(nodes)
            content = f"Aggregation of {len(nodes)} thoughts: " + \
                     "; ".join([n.content[:50] for n in nodes])
        elif aggregation_method == "max":
            best_node = max(nodes, key=lambda n: n.score)
            score = best_node.score
            content = f"Best of {len(nodes)}: {best_node.content}"
        else:
            score = sum(n.score for n in nodes) / len(nodes)
            content = "Aggregated thought"
        return self.add_node(content, ThoughtType.AGGREGATED, score)
    def visualize(self):
        """Print graph structure"""
        print(f"\n{'='*70}")
        print("GRAPH OF THOUGHTS")
        print(f"{'='*70}")
        print(f"Nodes: {len(self.nodes)}")
        print(f"Edges: {len(self.edges)}")
        # Print nodes by type
        for thought_type in ThoughtType:
            nodes = [n for n in self.nodes.values() if n.thought_type == thought_type]
            if nodes:
                print(f"\n{thought_type.value.upper()} NODES:")
                for node in nodes:
                    print(f"  [{node.id}] {node.content[:60]}... (score: {node.score:.2f})")
        # Print edges
        print(f"\nEDGES:")
        for edge in self.edges:
            from_node = self.nodes[edge.from_node]
            to_node = self.nodes[edge.to_node]
            print(f"  {from_node.id} --[{edge.edge_type.value}]--> {to_node.id}")
class GoTAgent:
    """Agent using Graph-of-Thoughts reasoning"""
    def __init__(self):
        self.graph = GraphOfThoughts()
    def solve_problem(self, problem: str) -> Dict[str, Any]:
        """Solve problem using graph-based reasoning"""
        print(f"\n{'='*70}")
        print(f"SOLVING: {problem}")
        print(f"{'='*70}")
        # Step 1: Create initial thought
        initial = self.graph.add_node(
            content=f"Problem: {problem}",
            thought_type=ThoughtType.INITIAL,
            score=1.0
        )
        print(f"\n1. Initial thought created: {initial.id}")
        # Step 2: Generate multiple reasoning paths
        paths = self._generate_reasoning_paths(initial, problem)
        print(f"\n2. Generated {len(paths)} reasoning paths")
        # Step 3: Evaluate and refine paths
        evaluated_paths = self._evaluate_paths(paths)
        print(f"\n3. Evaluated paths")
        # Step 4: Aggregate best thoughts
        final = self._aggregate_and_conclude(evaluated_paths)
        print(f"\n4. Created final conclusion: {final.id}")
        # Visualize the graph
        self.graph.visualize()
        # Find best path to solution
        best_path = self.graph.get_best_path(initial.id, final.id)
        return {
            'problem': problem,
            'initial_node': initial.id,
            'final_node': final.id,
            'best_path': [node.content for node in best_path],
            'conclusion': final.content,
            'confidence': final.score
        }
    def _generate_reasoning_paths(self, initial: ThoughtNode, problem: str) -> List[List[ThoughtNode]]:
        """Generate multiple reasoning approaches"""
        paths = []
        # Path 1: Analytical approach
        analytical = self.graph.add_node(
            content=f"Analyze {problem} systematically",
            thought_type=ThoughtType.INTERMEDIATE,
            score=0.8
        )
        self.graph.add_edge(initial, analytical, EdgeType.SEQUENTIAL)
        analytical_step2 = self.graph.add_node(
            content=f"Break down into sub-problems",
            thought_type=ThoughtType.INTERMEDIATE,
            score=0.85
        )
        self.graph.add_edge(analytical, analytical_step2, EdgeType.SEQUENTIAL)
        paths.append([initial, analytical, analytical_step2])
        # Path 2: Creative approach
        creative = self.graph.add_node(
            content=f"Consider unconventional solutions for {problem}",
            thought_type=ThoughtType.INTERMEDIATE,
            score=0.7
        )
        self.graph.add_edge(initial, creative, EdgeType.PARALLEL)
        creative_step2 = self.graph.add_node(
            content=f"Explore analogies and patterns",
            thought_type=ThoughtType.INTERMEDIATE,
            score=0.75
        )
        self.graph.add_edge(creative, creative_step2, EdgeType.SEQUENTIAL)
        paths.append([initial, creative, creative_step2])
        # Path 3: Empirical approach
        empirical = self.graph.add_node(
            content=f"Test hypotheses for {problem}",
            thought_type=ThoughtType.INTERMEDIATE,
            score=0.82
        )
        self.graph.add_edge(initial, empirical, EdgeType.PARALLEL)
        paths.append([initial, empirical])
        return paths
    def _evaluate_paths(self, paths: List[List[ThoughtNode]]) -> List[List[ThoughtNode]]:
        """Evaluate and potentially refine paths"""
        evaluated = []
        for path in paths:
            last_node = path[-1]
            # Add refinement node
            refined = self.graph.add_node(
                content=f"Refined: {last_node.content}",
                thought_type=ThoughtType.INTERMEDIATE,
                score=min(last_node.score + 0.1, 1.0)
            )
            self.graph.add_edge(last_node, refined, EdgeType.REFINE)
            evaluated.append(path + [refined])
        return evaluated
    def _aggregate_and_conclude(self, paths: List[List[ThoughtNode]]) -> ThoughtNode:
        """Aggregate insights and create conclusion"""
        # Get terminal nodes from all paths
        terminal_nodes = [path[-1] for path in paths]
        # Create aggregation node
        aggregated = self.graph.aggregate_nodes(
            [n.id for n in terminal_nodes],
            aggregation_method="mean"
        )
        # Link terminal nodes to aggregation
        for node in terminal_nodes:
            self.graph.add_edge(node, aggregated, EdgeType.MERGE)
        # Create final conclusion
        final = self.graph.add_node(
            content=f"Final solution incorporating multiple perspectives",
            thought_type=ThoughtType.FINAL,
            score=aggregated.score + 0.05
        )
        self.graph.add_edge(aggregated, final, EdgeType.SEQUENTIAL)
        return final
# Usage
if __name__ == "__main__":
    print("="*80)
    print("GRAPH-OF-THOUGHTS PATTERN DEMONSTRATION")
    print("="*80)
    agent = GoTAgent()
    # Solve a complex problem
    result = agent.solve_problem(
        "How can we reduce carbon emissions in urban transportation?"
    )
    print(f"\n{'='*70}")
    print("SOLUTION")
    print(f"{'='*70}")
    print(f"Conclusion: {result['conclusion']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"\nBest Reasoning Path:")
    for i, step in enumerate(result['best_path'], 1):
        print(f"  {i}. {step}")
