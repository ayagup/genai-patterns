"""
Pattern 004: Graph-of-Thoughts (GoT)

Description:
    Represents reasoning as a directed graph with nodes and edges.
    Unlike Tree-of-Thoughts which explores a tree structure, GoT allows for
    more flexible graph structures with cycles, merging paths, and complex
    relationships between thoughts.

Components:
    - Thought Nodes: Individual reasoning steps or ideas
    - Transformation Edges: Operations that transform one thought to another
    - Aggregation: Combining multiple thoughts into one
    - Graph Operations: Add, merge, transform, aggregate

Use Cases:
    - Complex multi-step reasoning with dependencies
    - Document analysis with cross-references
    - Problem-solving requiring non-linear thinking
    - Knowledge synthesis from multiple sources

LangChain Implementation:
    Uses LangGraph for state graph management and custom graph operations.
"""

import os
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


class EdgeType(Enum):
    """Types of edges in the graph."""
    TRANSFORM = "transform"  # Transform one thought to another
    AGGREGATE = "aggregate"  # Combine multiple thoughts
    REFINE = "refine"  # Refine/improve a thought
    COMPARE = "compare"  # Compare two thoughts
    VALIDATE = "validate"  # Validate a thought


@dataclass
class ThoughtNode:
    """Represents a node in the graph of thoughts."""
    id: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, ThoughtNode) and self.id == other.id


@dataclass
class ThoughtEdge:
    """Represents an edge in the graph of thoughts."""
    source: str  # Source node ID
    target: str  # Target node ID
    edge_type: EdgeType
    operation: str  # Description of the operation
    weight: float = 1.0


class ThoughtGraph:
    """Graph structure for managing thoughts."""
    
    def __init__(self):
        self.nodes: Dict[str, ThoughtNode] = {}
        self.edges: List[ThoughtEdge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)
    
    def add_node(self, node: ThoughtNode):
        """Add a node to the graph."""
        self.nodes[node.id] = node
    
    def add_edge(self, edge: ThoughtEdge):
        """Add an edge to the graph."""
        self.edges.append(edge)
        self.adjacency[edge.source].append(edge.target)
        self.reverse_adjacency[edge.target].append(edge.source)
    
    def get_node(self, node_id: str) -> Optional[ThoughtNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[ThoughtNode]:
        """Get all child nodes."""
        return [self.nodes[child_id] for child_id in self.adjacency.get(node_id, [])]
    
    def get_parents(self, node_id: str) -> List[ThoughtNode]:
        """Get all parent nodes."""
        return [self.nodes[parent_id] for parent_id in self.reverse_adjacency.get(node_id, [])]
    
    def get_all_paths(self, start_id: str, end_id: str) -> List[List[str]]:
        """Find all paths from start to end node."""
        paths = []
        visited = set()
        
        def dfs(current: str, path: List[str]):
            if current == end_id:
                paths.append(path.copy())
                return
            
            visited.add(current)
            for neighbor in self.adjacency.get(current, []):
                if neighbor not in visited:
                    path.append(neighbor)
                    dfs(neighbor, path)
                    path.pop()
            visited.remove(current)
        
        dfs(start_id, [start_id])
        return paths


class GraphOfThoughtsAgent:
    """Agent that uses Graph-of-Thoughts reasoning."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        """Initialize the Graph-of-Thoughts agent."""
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.graph = ThoughtGraph()
        self.node_counter = 0
    
    def generate_initial_thoughts(self, problem: str, num_thoughts: int = 3) -> List[ThoughtNode]:
        """Generate initial thoughts for the problem."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate {num_thoughts} diverse initial approaches or perspectives 
for solving the problem. Each should be a distinct starting point.

Format your response as:
1. [First approach]
2. [Second approach]
3. [Third approach]"""),
            ("human", "Problem: {problem}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"problem": problem, "num_thoughts": num_thoughts})
        
        # Parse thoughts
        thoughts = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                content = line.split('.', 1)[-1].strip().lstrip('- ').strip()
                if content:
                    node = ThoughtNode(
                        id=f"node_{self.node_counter}",
                        content=content,
                        metadata={"type": "initial", "problem": problem}
                    )
                    self.node_counter += 1
                    thoughts.append(node)
                    self.graph.add_node(node)
        
        return thoughts[:num_thoughts]
    
    def transform_thought(self, node: ThoughtNode, transformation: str) -> ThoughtNode:
        """Transform a thought using a specific operation."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Transform the given thought by: {transformation}"),
            ("human", "Original thought: {thought}\n\nTransformed thought:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "transformation": transformation,
            "thought": node.content
        })
        
        new_node = ThoughtNode(
            id=f"node_{self.node_counter}",
            content=result.strip(),
            metadata={"type": "transformed", "parent": node.id}
        )
        self.node_counter += 1
        
        self.graph.add_node(new_node)
        self.graph.add_edge(ThoughtEdge(
            source=node.id,
            target=new_node.id,
            edge_type=EdgeType.TRANSFORM,
            operation=transformation
        ))
        
        return new_node
    
    def aggregate_thoughts(self, nodes: List[ThoughtNode], aggregation_type: str = "synthesis") -> ThoughtNode:
        """Aggregate multiple thoughts into one."""
        thoughts_text = "\n".join([f"- {node.content}" for node in nodes])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Synthesize the following thoughts into a single, coherent insight.
Combine the key ideas while maintaining logical consistency."""),
            ("human", "Thoughts to synthesize:\n{thoughts}\n\nSynthesized insight:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"thoughts": thoughts_text})
        
        new_node = ThoughtNode(
            id=f"node_{self.node_counter}",
            content=result.strip(),
            metadata={"type": "aggregated", "parents": [n.id for n in nodes]}
        )
        self.node_counter += 1
        
        self.graph.add_node(new_node)
        for node in nodes:
            self.graph.add_edge(ThoughtEdge(
                source=node.id,
                target=new_node.id,
                edge_type=EdgeType.AGGREGATE,
                operation=aggregation_type
            ))
        
        return new_node
    
    def refine_thought(self, node: ThoughtNode, feedback: str) -> ThoughtNode:
        """Refine a thought based on feedback."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Refine the thought based on this feedback: {feedback}"),
            ("human", "Original thought: {thought}\n\nRefined thought:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "feedback": feedback,
            "thought": node.content
        })
        
        new_node = ThoughtNode(
            id=f"node_{self.node_counter}",
            content=result.strip(),
            metadata={"type": "refined", "parent": node.id}
        )
        self.node_counter += 1
        
        self.graph.add_node(new_node)
        self.graph.add_edge(ThoughtEdge(
            source=node.id,
            target=new_node.id,
            edge_type=EdgeType.REFINE,
            operation="refinement"
        ))
        
        return new_node
    
    def evaluate_thought(self, node: ThoughtNode, problem: str) -> float:
        """Evaluate how well a thought addresses the problem."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Rate how well this thought addresses the problem on a scale of 0.0 to 1.0.
Consider relevance, feasibility, and completeness.
Respond with ONLY a number between 0.0 and 1.0."""),
            ("human", """Problem: {problem}

Thought: {thought}

Score (0.0-1.0):""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({"problem": problem, "thought": node.content})
            score = float(response.strip())
            node.score = max(0.0, min(1.0, score))
            return node.score
        except:
            return 0.5
    
    def solve_with_graph(self, problem: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Solve a problem using Graph-of-Thoughts approach.
        
        Args:
            problem: The problem to solve
            max_iterations: Maximum number of transformation iterations
            
        Returns:
            Dictionary with solution and graph information
        """
        print(f"\nSolving: {problem}\n")
        
        # Generate initial thoughts
        print("Step 1: Generate initial thoughts")
        initial_thoughts = self.generate_initial_thoughts(problem, num_thoughts=3)
        for i, node in enumerate(initial_thoughts, 1):
            print(f"  Thought {i}: {node.content[:70]}...")
        
        # Transform and refine thoughts
        print(f"\nStep 2: Transform and refine thoughts")
        transformed_nodes = []
        
        for node in initial_thoughts[:2]:  # Transform first two
            # Apply transformations
            transformation = "Make it more specific and actionable"
            transformed = self.transform_thought(node, transformation)
            print(f"  Transformed: {transformed.content[:70]}...")
            transformed_nodes.append(transformed)
        
        # Aggregate thoughts
        print(f"\nStep 3: Aggregate insights")
        if len(transformed_nodes) >= 2:
            aggregated = self.aggregate_thoughts(transformed_nodes[:2])
            print(f"  Aggregated: {aggregated.content[:70]}...")
            
            # Refine the aggregated thought
            print(f"\nStep 4: Refine final insight")
            refined = self.refine_thought(aggregated, "Make it more comprehensive and practical")
            print(f"  Refined: {refined.content[:70]}...")
            
            # Evaluate final thought
            final_score = self.evaluate_thought(refined, problem)
            print(f"\n  Final score: {final_score:.2f}")
            
            final_node = refined
        else:
            final_node = transformed_nodes[0] if transformed_nodes else initial_thoughts[0]
            final_score = self.evaluate_thought(final_node, problem)
        
        return {
            "problem": problem,
            "solution": final_node.content,
            "score": final_score,
            "graph_stats": {
                "total_nodes": len(self.graph.nodes),
                "total_edges": len(self.graph.edges),
                "node_types": self._count_node_types()
            }
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        """Count nodes by type."""
        counts = defaultdict(int)
        for node in self.graph.nodes.values():
            node_type = node.metadata.get("type", "unknown")
            counts[node_type] += 1
        return dict(counts)
    
    def visualize_graph(self) -> str:
        """Generate a text visualization of the graph."""
        lines = ["Graph Structure:", "=" * 60]
        
        for node in self.graph.nodes.values():
            node_type = node.metadata.get("type", "unknown")
            lines.append(f"\n[{node.id}] ({node_type})")
            lines.append(f"  Content: {node.content[:60]}...")
            lines.append(f"  Score: {node.score:.2f}")
            
            children = self.graph.get_children(node.id)
            if children:
                lines.append(f"  Children: {', '.join([c.id for c in children])}")
        
        return "\n".join(lines)


def demonstrate_got_pattern():
    """Demonstrates the Graph-of-Thoughts pattern."""
    
    print("=" * 80)
    print("PATTERN 004: Graph-of-Thoughts (GoT)")
    print("=" * 80)
    print()
    
    # Create GoT agent
    agent = GraphOfThoughtsAgent()
    
    # Test problems
    problems = [
        "Design a sustainable urban transportation system for a growing city",
        "Create a strategy to improve team collaboration in a remote work environment"
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{'=' * 80}")
        print(f"Problem {i}: {problem}")
        print('=' * 80)
        
        try:
            # Reset agent for new problem
            agent = GraphOfThoughtsAgent()
            
            result = agent.solve_with_graph(problem, max_iterations=3)
            
            print(f"\n\n{'=' * 80}")
            print("SOLUTION")
            print('=' * 80)
            print(result['solution'])
            print(f"\nFinal Score: {result['score']:.2f}")
            print(f"\nGraph Statistics:")
            print(f"  Total Nodes: {result['graph_stats']['total_nodes']}")
            print(f"  Total Edges: {result['graph_stats']['total_edges']}")
            print(f"  Node Types: {result['graph_stats']['node_types']}")
            
            # Show graph structure
            print(f"\n{agent.visualize_graph()}")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("GRAPH-OF-THOUGHTS PATTERN DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("1. Graph structure for non-linear reasoning")
    print("2. Thought transformation operations")
    print("3. Aggregation of multiple thoughts")
    print("4. Refinement based on feedback")
    print("5. Flexible graph operations (transform, aggregate, refine)")
    print()
    print("Advantages over Tree-of-Thoughts:")
    print("- More flexible thought relationships")
    print("- Can merge multiple reasoning paths")
    print("- Supports cycles and cross-references")
    print("- Better for complex, interconnected problems")
    print()
    print("LangChain Components Used:")
    print("- ChatPromptTemplate: Structures transformation prompts")
    print("- StrOutputParser: Parses LLM responses")
    print("- Custom graph structure: Manages thought relationships")
    print("- Multiple edge types: Different operations between thoughts")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    demonstrate_got_pattern()
