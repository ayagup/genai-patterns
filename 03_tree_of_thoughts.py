"""
Tree-of-Thoughts (ToT) Pattern
===============================
Explores multiple reasoning paths simultaneously in a tree structure.
Components: Thought generation, evaluation, search (BFS/DFS)
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import deque
import copy


@dataclass
class ThoughtNode:
    """Represents a thought in the tree"""
    content: str
    depth: int
    score: float
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def __repr__(self):
        return f"ThoughtNode(depth={self.depth}, score={self.score:.2f}, content='{self.content[:30]}...')"


class TreeOfThoughtsAgent:
    """Tree-of-Thoughts reasoning agent"""
    
    def __init__(self, max_depth: int = 3, branching_factor: int = 3):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.root: Optional[ThoughtNode] = None
    
    def generate_thoughts(self, current_state: str, depth: int) -> List[str]:
        """
        Generate multiple possible next thoughts
        This would typically use an LLM, but we'll simulate it
        """
        if "arrange numbers" in current_state.lower():
            # For number arrangement problem
            if depth == 0:
                return [
                    "Start by placing the largest number in a strategic position",
                    "Consider the mathematical relationships between numbers",
                    "Try different ordering strategies: ascending, descending, alternating"
                ]
            elif depth == 1:
                return [
                    "Place largest at the beginning for maximum impact",
                    "Place largest in the middle to balance the arrangement",
                    "Place largest at the end and work backwards"
                ]
            else:
                return [
                    "Verify the arrangement meets all constraints",
                    "Calculate the final result",
                    "Check for alternative arrangements"
                ]
        
        elif "game of 24" in current_state.lower():
            # For Game of 24 problem
            numbers = [4, 6, 8, 2]  # Example numbers
            if depth == 0:
                return [
                    f"Try addition first: explore {numbers[0]} + {numbers[1]}",
                    f"Try multiplication first: explore {numbers[0]} * {numbers[1]}",
                    f"Try subtraction first: explore {numbers[0]} - {numbers[1]}"
                ]
            elif depth == 1:
                return [
                    "Combine intermediate results with remaining numbers",
                    "Try different operation orders",
                    "Consider using parentheses for operation precedence"
                ]
            else:
                return [
                    "Evaluate if current path reaches 24",
                    "Backtrack if necessary",
                    "Verify the solution"
                ]
        
        else:
            # Generic problem solving
            return [
                f"Approach {depth+1}A: Analyze the problem from first principles",
                f"Approach {depth+1}B: Use analogical reasoning",
                f"Approach {depth+1}C: Break down into subproblems"
            ]
    
    def evaluate_thought(self, thought: str, depth: int) -> float:
        """
        Evaluate how promising a thought is
        Returns a score between 0 and 1
        """
        score = 0.5  # Base score
        
        # Reward thoughts that show progress
        if any(word in thought.lower() for word in ['verify', 'calculate', 'solution']):
            score += 0.3
        
        # Reward strategic thinking
        if any(word in thought.lower() for word in ['strategic', 'consider', 'analyze']):
            score += 0.2
        
        # Penalize vague thoughts
        if any(word in thought.lower() for word in ['maybe', 'perhaps', 'unsure']):
            score -= 0.2
        
        # Reward concrete actions
        if any(word in thought.lower() for word in ['place', 'combine', 'try']):
            score += 0.15
        
        # Adjust based on depth (prefer thoughts that make progress)
        score += depth * 0.05
        
        return max(0.0, min(1.0, score))
    
    def search_bfs(self, problem: str) -> List[ThoughtNode]:
        """
        Breadth-First Search through the thought tree
        Returns the path to the best solution
        """
        print(f"\n{'='*60}")
        print(f"BFS Search for: {problem}")
        print(f"{'='*60}\n")
        
        # Initialize root
        self.root = ThoughtNode(
            content=f"Problem: {problem}",
            depth=0,
            score=0.5
        )
        
        queue = deque([self.root])
        best_leaf = self.root
        best_score = 0.0
        
        iteration = 0
        while queue and iteration < 20:  # Limit iterations
            current = queue.popleft()
            iteration += 1
            
            print(f"Exploring depth {current.depth}: {current.content[:50]}...")
            
            if current.depth >= self.max_depth:
                if current.score > best_score:
                    best_score = current.score
                    best_leaf = current
                continue
            
            # Generate and evaluate child thoughts
            thoughts = self.generate_thoughts(problem, current.depth)
            
            for thought in thoughts[:self.branching_factor]:
                score = self.evaluate_thought(thought, current.depth + 1)
                child = ThoughtNode(
                    content=thought,
                    depth=current.depth + 1,
                    score=score,
                    parent=current
                )
                current.children.append(child)
                queue.append(child)
                
                print(f"  → Generated thought (score={score:.2f}): {thought[:50]}...")
        
        # Reconstruct path
        path = []
        node = best_leaf
        while node:
            path.append(node)
            node = node.parent
        path.reverse()
        
        print(f"\n✅ Best path found with score: {best_score:.2f}")
        return path
    
    def search_dfs(self, problem: str, threshold: float = 0.7) -> List[ThoughtNode]:
        """
        Depth-First Search through the thought tree
        Uses pruning based on evaluation scores
        """
        print(f"\n{'='*60}")
        print(f"DFS Search for: {problem}")
        print(f"{'='*60}\n")
        
        # Initialize root
        self.root = ThoughtNode(
            content=f"Problem: {problem}",
            depth=0,
            score=0.5
        )
        
        best_path = [self.root]
        best_score = 0.0
        
        def dfs_recursive(node: ThoughtNode, current_path: List[ThoughtNode]):
            nonlocal best_path, best_score
            
            print(f"{'  ' * node.depth}Exploring: {node.content[:40]}... (score={node.score:.2f})")
            
            if node.depth >= self.max_depth:
                if node.score > best_score:
                    best_score = node.score
                    best_path = current_path.copy()
                return
            
            # Generate child thoughts
            thoughts = self.generate_thoughts(problem, node.depth)
            
            # Evaluate and sort by score
            children_data = []
            for thought in thoughts[:self.branching_factor]:
                score = self.evaluate_thought(thought, node.depth + 1)
                children_data.append((thought, score))
            
            # Sort by score (descending) and prune low-scoring branches
            children_data.sort(key=lambda x: x[1], reverse=True)
            
            for thought, score in children_data:
                if score < threshold:
                    print(f"{'  ' * (node.depth + 1)}✂️  Pruned: {thought[:40]}... (score={score:.2f} < {threshold})")
                    continue
                
                child = ThoughtNode(
                    content=thought,
                    depth=node.depth + 1,
                    score=score,
                    parent=node
                )
                node.children.append(child)
                current_path.append(child)
                
                dfs_recursive(child, current_path)
                
                current_path.pop()
        
        dfs_recursive(self.root, [self.root])
        
        print(f"\n✅ Best path found with score: {best_score:.2f}")
        return best_path
    
    def visualize_path(self, path: List[ThoughtNode]):
        """Visualize the solution path"""
        print(f"\n{'='*60}")
        print("SOLUTION PATH")
        print(f"{'='*60}\n")
        
        for i, node in enumerate(path):
            indent = "  " * node.depth
            arrow = "→" if i > 0 else "🎯"
            print(f"{indent}{arrow} {node.content}")
            print(f"{indent}   [Score: {node.score:.2f}]\n")


def main():
    """Demonstrate Tree-of-Thoughts pattern"""
    
    agent = TreeOfThoughtsAgent(max_depth=3, branching_factor=3)
    
    # Example 1: Number arrangement problem
    print("\n" + "="*60)
    print("EXAMPLE 1: Number Arrangement Problem (BFS)")
    print("="*60)
    problem1 = "Arrange numbers [3, 7, 1, 9] to maximize the result"
    path1 = agent.search_bfs(problem1)
    agent.visualize_path(path1)
    
    # Example 2: Game of 24 (DFS with pruning)
    print("\n" + "="*60)
    print("EXAMPLE 2: Game of 24 Problem (DFS)")
    print("="*60)
    agent2 = TreeOfThoughtsAgent(max_depth=3, branching_factor=3)
    problem2 = "Use numbers [4, 6, 8, 2] to make 24 using +, -, ×, ÷ in game of 24"
    path2 = agent2.search_dfs(problem2, threshold=0.6)
    agent2.visualize_path(path2)
    
    # Example 3: Creative writing
    print("\n" + "="*60)
    print("EXAMPLE 3: Creative Problem Solving")
    print("="*60)
    agent3 = TreeOfThoughtsAgent(max_depth=2, branching_factor=3)
    problem3 = "Design a solution for reducing traffic congestion"
    path3 = agent3.search_bfs(problem3)
    agent3.visualize_path(path3)


if __name__ == "__main__":
    main()
