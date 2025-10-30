"""
Pattern 3: Tree-of-Thoughts (ToT)
Explores multiple reasoning paths simultaneously in a tree structure.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import deque

@dataclass
class ThoughtNode:
    """Represents a thought in the tree"""
    content: str
    score: float
    depth: int
    parent: 'ThoughtNode' = None
    children: List['ThoughtNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []

class TreeOfThoughtsPattern:
    def __init__(self, branching_factor: int = 3, max_depth: int = 3):
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-4")
        self.evaluator_llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.branching_factor = branching_factor
        self.max_depth = max_depth
    
    def generate_thoughts(self, problem: str, current_state: str, num_thoughts: int) -> List[str]:
        """Generate multiple next thought steps"""
        template = """Problem: {problem}

Current reasoning state: {current_state}

Generate {num_thoughts} different possible next steps in reasoning. Each should be a distinct approach.

Next steps:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["problem", "current_state", "num_thoughts"]
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        result = chain.run(
            problem=problem,
            current_state=current_state,
            num_thoughts=num_thoughts
        )
        
        # Parse the thoughts (simplified parsing)
        thoughts = [t.strip() for t in result.split('\n') if t.strip() and not t.startswith('Next')]
        return thoughts[:num_thoughts]
    
    def evaluate_thought(self, problem: str, thought: str) -> float:
        """Evaluate the quality of a thought"""
        template = """Problem: {problem}

Reasoning step: {thought}

Rate this reasoning step on a scale of 0-10 for:
1. Logical soundness
2. Progress toward solution
3. Clarity

Provide only a single number between 0-10 as your rating."""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["problem", "thought"]
        )
        chain = LLMChain(llm=self.evaluator_llm, prompt=prompt)
        result = chain.run(problem=problem, thought=thought)
        
        try:
            score = float(result.strip().split()[0])
            return min(max(score, 0), 10)  # Clamp between 0-10
        except:
            return 5.0  # Default score
    
    def bfs_search(self, problem: str) -> Tuple[List[ThoughtNode], ThoughtNode]:
        """Breadth-First Search through thought tree"""
        root = ThoughtNode(
            content="Initial problem statement",
            score=5.0,
            depth=0
        )
        
        queue = deque([root])
        all_nodes = [root]
        best_leaf = root
        
        while queue:
            node = queue.popleft()
            
            if node.depth >= self.max_depth:
                if node.score > best_leaf.score:
                    best_leaf = node
                continue
            
            # Generate child thoughts
            thoughts = self.generate_thoughts(
                problem,
                node.content,
                self.branching_factor
            )
            
            for thought in thoughts:
                score = self.evaluate_thought(problem, thought)
                child = ThoughtNode(
                    content=thought,
                    score=score,
                    depth=node.depth + 1,
                    parent=node
                )
                node.children.append(child)
                queue.append(child)
                all_nodes.append(child)
                
                if child.depth == self.max_depth and child.score > best_leaf.score:
                    best_leaf = child
        
        return all_nodes, best_leaf
    
    def dfs_search(self, problem: str) -> Tuple[List[ThoughtNode], ThoughtNode]:
        """Depth-First Search through thought tree"""
        root = ThoughtNode(
            content="Initial problem statement",
            score=5.0,
            depth=0
        )
        
        stack = [root]
        all_nodes = [root]
        best_leaf = root
        
        while stack:
            node = stack.pop()
            
            if node.depth >= self.max_depth:
                if node.score > best_leaf.score:
                    best_leaf = node
                continue
            
            thoughts = self.generate_thoughts(
                problem,
                node.content,
                self.branching_factor
            )
            
            for thought in thoughts:
                score = self.evaluate_thought(problem, thought)
                child = ThoughtNode(
                    content=thought,
                    score=score,
                    depth=node.depth + 1,
                    parent=node
                )
                node.children.append(child)
                stack.append(child)
                all_nodes.append(child)
                
                if child.depth == self.max_depth and child.score > best_leaf.score:
                    best_leaf = child
        
        return all_nodes, best_leaf
    
    def get_path_to_root(self, node: ThoughtNode) -> List[str]:
        """Get the reasoning path from root to this node"""
        path = []
        current = node
        while current:
            path.append(current.content)
            current = current.parent
        return list(reversed(path))
    
    def solve(self, problem: str, search_strategy: str = "bfs") -> Dict:
        """Solve problem using Tree of Thoughts"""
        if search_strategy == "bfs":
            all_nodes, best_solution = self.bfs_search(problem)
        else:
            all_nodes, best_solution = self.dfs_search(problem)
        
        solution_path = self.get_path_to_root(best_solution)
        
        return {
            "problem": problem,
            "search_strategy": search_strategy,
            "total_nodes_explored": len(all_nodes),
            "best_score": best_solution.score,
            "solution_path": solution_path,
            "final_thought": best_solution.content
        }

if __name__ == "__main__":
    tot = TreeOfThoughtsPattern(branching_factor=3, max_depth=2)
    
    problem = """You have 3 boxes. One contains only apples, one contains only oranges, 
    and one contains both. All boxes are labeled incorrectly. You can pick one fruit from 
    one box. How do you determine the correct labels?"""
    
    result = tot.solve(problem, search_strategy="bfs")
    
    print(f"Problem: {result['problem']}")
    print(f"\nNodes explored: {result['total_nodes_explored']}")
    print(f"Best score: {result['best_score']}")
    print(f"\nSolution path:")
    for i, step in enumerate(result['solution_path'], 1):
        print(f"{i}. {step}")
