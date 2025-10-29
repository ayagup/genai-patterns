"""
Pattern 4: Graph-of-Thoughts (GoT)
Represents reasoning as a directed graph with nodes and edges.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, field
import networkx as nx
import matplotlib.pyplot as plt

@dataclass
class ThoughtGraphNode:
    """Node in the thought graph"""
    id: str
    content: str
    node_type: str  # 'input', 'reasoning', 'aggregation', 'output'
    metadata: Dict = field(default_factory=dict)

@dataclass
class ThoughtGraphEdge:
    """Edge in the thought graph"""
    source: str
    target: str
    edge_type: str  # 'generates', 'refines', 'supports', 'contradicts'
    weight: float = 1.0

class GraphOfThoughtsPattern:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-4")
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, ThoughtGraphNode] = {}
        self.node_counter = 0
    
    def create_node(self, content: str, node_type: str, metadata: Dict = None) -> str:
        """Create a new thought node"""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        
        node = ThoughtGraphNode(
            id=node_id,
            content=content,
            node_type=node_type,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.__dict__)
        
        return node_id
    
    def add_edge(self, source: str, target: str, edge_type: str, weight: float = 1.0):
        """Add an edge between thoughts"""
        edge = ThoughtGraphEdge(source, target, edge_type, weight)
        self.graph.add_edge(source, target, type=edge_type, weight=weight)
    
    def generate_thought(self, prompt: str, context_nodes: List[str] = None) -> str:
        """Generate a new thought based on context"""
        if context_nodes:
            context = "\n".join([
                f"- {self.nodes[nid].content}"
                for nid in context_nodes
            ])
            full_prompt = f"""Context:\n{context}\n\nBased on the above context, {prompt}"""
        else:
            full_prompt = prompt
        
        template = PromptTemplate(
            template="{prompt}",
            input_variables=["prompt"]
        )
        chain = LLMChain(llm=self.llm, prompt=template)
        return chain.run(prompt=full_prompt)
    
    def decompose_problem(self, problem: str) -> List[str]:
        """Decompose problem into sub-problems"""
        template = """Break down the following problem into 3-5 independent sub-problems:

Problem: {problem}

Sub-problems:"""
        
        prompt = PromptTemplate(template=template, input_variables=["problem"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(problem=problem)
        
        sub_problems = [sp.strip() for sp in result.split('\n') if sp.strip() and not sp.startswith('Sub-problems')]
        return sub_problems[:5]
    
    def aggregate_thoughts(self, node_ids: List[str]) -> str:
        """Aggregate multiple thoughts into a synthesis"""
        thoughts = [self.nodes[nid].content for nid in node_ids]
        thoughts_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(thoughts)])
        
        template = """Synthesize the following thoughts into a coherent conclusion:

{thoughts}

Synthesis:"""
        
        prompt = PromptTemplate(template=template, input_variables=["thoughts"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(thoughts=thoughts_text)
    
    def solve_with_graph(self, problem: str) -> Dict:
        """Solve problem using graph-based reasoning"""
        # Create input node
        input_node = self.create_node(problem, 'input')
        
        # Decompose into sub-problems
        sub_problems = self.decompose_problem(problem)
        sub_problem_nodes = []
        
        for sp in sub_problems:
            sp_node = self.create_node(sp, 'reasoning', {'level': 'sub-problem'})
            self.add_edge(input_node, sp_node, 'generates')
            sub_problem_nodes.append(sp_node)
        
        # Generate solutions for each sub-problem
        solution_nodes = []
        for sp_node in sub_problem_nodes:
            solution = self.generate_thought(
                f"Provide a solution to: {self.nodes[sp_node].content}",
                [sp_node]
            )
            sol_node = self.create_node(solution, 'reasoning', {'level': 'solution'})
            self.add_edge(sp_node, sol_node, 'generates')
            solution_nodes.append(sol_node)
        
        # Create refinement layer (thoughts can refine each other)
        for i, sol_node in enumerate(solution_nodes):
            for j, other_node in enumerate(solution_nodes):
                if i != j:
                    # Check if thoughts are related
                    self.add_edge(sol_node, other_node, 'supports', weight=0.5)
        
        # Aggregate all solutions
        final_synthesis = self.aggregate_thoughts(solution_nodes)
        output_node = self.create_node(final_synthesis, 'output')
        
        for sol_node in solution_nodes:
            self.add_edge(sol_node, output_node, 'supports')
        
        # Analyze graph structure
        return {
            'problem': problem,
            'num_nodes': len(self.nodes),
            'num_edges': self.graph.number_of_edges(),
            'sub_problems': sub_problems,
            'solutions': [self.nodes[nid].content for nid in solution_nodes],
            'final_answer': final_synthesis,
            'graph_summary': {
                'avg_degree': sum(dict(self.graph.degree()).values()) / len(self.nodes),
                'density': nx.density(self.graph),
                'is_dag': nx.is_directed_acyclic_graph(self.graph)
            }
        }
    
    def visualize_graph(self, filename: str = "thought_graph.png"):
        """Visualize the thought graph"""
        plt.figure(figsize=(15, 10))
        
        # Color nodes by type
        color_map = {
            'input': 'lightblue',
            'reasoning': 'lightgreen',
            'aggregation': 'yellow',
            'output': 'lightcoral'
        }
        
        colors = [color_map[self.nodes[node].node_type] for node in self.graph.nodes()]
        
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        nx.draw(
            self.graph,
            pos,
            node_color=colors,
            with_labels=True,
            node_size=3000,
            font_size=8,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            arrowsize=20
        )
        
        plt.title("Graph of Thoughts")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    got = GraphOfThoughtsPattern()
    
    problem = """Design a sustainable urban transportation system for a city of 1 million people."""
    
    result = got.solve_with_graph(problem)
    
    print(f"Problem: {result['problem']}\n")
    print(f"Graph Statistics:")
    print(f"  Nodes: {result['num_nodes']}")
    print(f"  Edges: {result['num_edges']}")
    print(f"  Average Degree: {result['graph_summary']['avg_degree']:.2f}")
    print(f"  Is DAG: {result['graph_summary']['is_dag']}\n")
    
    print("Sub-problems identified:")
    for i, sp in enumerate(result['sub_problems'], 1):
        print(f"  {i}. {sp}")
    
    print(f"\nFinal Answer:\n{result['final_answer']}")
    
    # Visualize
    got.visualize_graph("got_example.png")
    print("\nGraph visualization saved to got_example.png")
