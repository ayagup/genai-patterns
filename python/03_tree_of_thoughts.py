"""
Tree-of-Thoughts Pattern
Explores multiple reasoning paths simultaneously
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque
@dataclass
class ThoughtNode:
    content: str
    depth: int
    score: float
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = None
    def __post_init__(self):
        if self.children is None:
            self.children = []
class TreeOfThoughtsAgent:
    def __init__(self, max_depth: int = 3, branching_factor: int = 3):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
    def generate_thoughts(self, prompt: str, current_thought: str, depth: int) -> List[str]:
        """Generate possible next thoughts"""
        # Simulated thought generation (would use LLM in real implementation)
        if "creative writing" in prompt.lower():
            return [
                f"Start with character introduction at depth {depth}",
                f"Begin with setting description at depth {depth}",
                f"Open with dialogue at depth {depth}"
            ]
        return [f"Thought variant {i} at depth {depth}" for i in range(self.branching_factor)]
    def evaluate_thought(self, thought: str, goal: str) -> float:
        """Evaluate quality of a thought (0-1)"""
        # Simulated evaluation (would use LLM or heuristic in real implementation)
        score = 0.5 + (len(thought) % 10) / 20  # Dummy scoring
        return min(1.0, score)
    def search_bfs(self, initial_prompt: str) -> List[ThoughtNode]:
        """Breadth-First Search through thought tree"""
        root = ThoughtNode(content=initial_prompt, depth=0, score=1.0)
        queue = deque([root])
        all_nodes = [root]
        while queue:
            current = queue.popleft()
            if current.depth >= self.max_depth:
                continue
            # Generate and evaluate next thoughts
            thoughts = self.generate_thoughts(initial_prompt, current.content, current.depth + 1)
            for thought in thoughts:
                score = self.evaluate_thought(thought, initial_prompt)
                child = ThoughtNode(
                    content=thought,
                    depth=current.depth + 1,
                    score=score,
                    parent=current
                )
                current.children.append(child)
                all_nodes.append(child)
                queue.append(child)
        return all_nodes
    def search_dfs(self, initial_prompt: str, max_thoughts: int = 10) -> List[ThoughtNode]:
        """Depth-First Search through thought tree"""
        root = ThoughtNode(content=initial_prompt, depth=0, score=1.0)
        stack = [root]
        all_nodes = [root]
        while stack and len(all_nodes) < max_thoughts:
            current = stack.pop()
            if current.depth >= self.max_depth:
                continue
            thoughts = self.generate_thoughts(initial_prompt, current.content, current.depth + 1)
            for thought in thoughts:
                score = self.evaluate_thought(thought, initial_prompt)
                child = ThoughtNode(
                    content=thought,
                    depth=current.depth + 1,
                    score=score,
                    parent=current
                )
                current.children.append(child)
                all_nodes.append(child)
                stack.append(child)
        return all_nodes
    def get_best_path(self, nodes: List[ThoughtNode]) -> List[ThoughtNode]:
        """Find best path from root to leaf"""
        leaves = [n for n in nodes if not n.children]
        best_leaf = max(leaves, key=lambda n: n.score)
        # Trace back to root
        path = []
        current = best_leaf
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))
# Usage
if __name__ == "__main__":
    agent = TreeOfThoughtsAgent(max_depth=3, branching_factor=2)
    prompt = "Write a creative short story opening"
    print(f"Initial Prompt: {prompt}\n")
    print("=== Breadth-First Search ===")
    nodes_bfs = agent.search_bfs(prompt)
    best_path_bfs = agent.get_best_path(nodes_bfs)
    print("\nBest path (BFS):")
    for i, node in enumerate(best_path_bfs):
        print(f"  Depth {node.depth}: {node.content} (score: {node.score:.2f})")
    print("\n=== Depth-First Search ===")
    nodes_dfs = agent.search_dfs(prompt)
    best_path_dfs = agent.get_best_path(nodes_dfs)
    print("\nBest path (DFS):")
    for i, node in enumerate(best_path_dfs):
        print(f"  Depth {node.depth}: {node.content} (score: {node.score:.2f})")
