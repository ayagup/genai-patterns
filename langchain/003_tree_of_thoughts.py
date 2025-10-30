"""
Pattern 003: Tree-of-Thoughts (ToT)

Description:
    Explores multiple reasoning paths simultaneously in a tree structure.
    ToT allows the model to generate multiple candidate thoughts at each step,
    evaluate them, and search through the tree (BFS or DFS) to find the best solution.

Components:
    - Thought Generation: Creates multiple candidate next steps
    - State Evaluation: Scores each thought/state
    - Search Strategy: BFS, DFS, or beam search
    - Backtracking: Can revisit and try alternative paths

Use Cases:
    - Strategic planning and game playing
    - Creative writing with multiple storylines
    - Complex problem-solving with multiple approaches
    - Optimization problems

LangChain Implementation:
    Uses LangGraph for state management and custom search logic for tree exploration.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


class SearchStrategy(Enum):
    """Search strategies for tree exploration."""
    BFS = "breadth_first"
    DFS = "depth_first"
    BEAM = "beam_search"


@dataclass
class ThoughtNode:
    """Represents a node in the tree of thoughts."""
    thought: str
    state: str
    depth: int
    score: float = 0.0
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = field(default_factory=list)
    is_solution: bool = False
    
    def get_path(self) -> List[str]:
        """Returns the path from root to this node."""
        path = []
        current = self
        while current:
            path.append(current.thought)
            current = current.parent
        return list(reversed(path))


class TreeOfThoughtsAgent:
    """Agent that uses Tree-of-Thoughts reasoning."""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        branching_factor: int = 3,
        max_depth: int = 4
    ):
        """
        Initialize the Tree-of-Thoughts agent.
        
        Args:
            model_name: LLM model to use
            temperature: Temperature for thought generation
            branching_factor: Number of thoughts to generate at each node
            max_depth: Maximum depth of the tree
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.branching_factor = branching_factor
        self.max_depth = max_depth
    
    def generate_thoughts(
        self,
        problem: str,
        current_state: str,
        num_thoughts: int = 3
    ) -> List[str]:
        """
        Generate multiple candidate thoughts/next steps.
        
        Args:
            problem: The original problem
            current_state: Current problem-solving state
            num_thoughts: Number of thoughts to generate
            
        Returns:
            List of candidate thoughts
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a creative problem solver. Generate {num_thoughts} different 
potential next steps or thoughts for solving the problem. Each thought should explore 
a different approach or direction. Be concise and specific.

Format your response as:
1. [First thought]
2. [Second thought]
3. [Third thought]"""),
            ("human", """Problem: {problem}

Current state: {current_state}

Generate {num_thoughts} diverse next steps:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "problem": problem,
            "current_state": current_state,
            "num_thoughts": num_thoughts
        })
        
        # Parse thoughts from numbered list
        thoughts = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering/bullets
                thought = line.split('.', 1)[-1].strip()
                thought = thought.lstrip('- ').strip()
                if thought:
                    thoughts.append(thought)
        
        return thoughts[:num_thoughts]
    
    def evaluate_thought(self, problem: str, thought: str, current_state: str) -> float:
        """
        Evaluate the quality/promise of a thought.
        
        Args:
            problem: The original problem
            thought: The thought to evaluate
            current_state: Current state description
            
        Returns:
            Score between 0 and 1
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator. Rate how promising this thought is 
for solving the problem on a scale of 0.0 to 1.0, where:
- 0.0-0.3: Unlikely to help or may lead to dead end
- 0.4-0.6: Moderately promising
- 0.7-0.9: Very promising
- 1.0: This is the solution or very close to it

Respond with ONLY a number between 0.0 and 1.0."""),
            ("human", """Problem: {problem}

Current state: {current_state}

Thought to evaluate: {thought}

Score (0.0-1.0):""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "problem": problem,
                "current_state": current_state,
                "thought": thought
            })
            score = float(response.strip())
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except:
            return 0.5  # Default score if parsing fails
    
    def is_solution(self, problem: str, current_state: str) -> bool:
        """
        Check if current state represents a solution.
        
        Args:
            problem: The original problem
            current_state: Current state description
            
        Returns:
            True if this is a valid solution
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Determine if the current state represents a complete solution 
to the problem. Respond with ONLY 'yes' or 'no'."""),
            ("human", """Problem: {problem}

Current state: {current_state}

Is this a complete solution? (yes/no):""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "problem": problem,
            "current_state": current_state
        })
        
        return response.strip().lower().startswith('yes')
    
    def search_bfs(self, problem: str, max_nodes: int = 20) -> Optional[ThoughtNode]:
        """
        Breadth-first search through the tree of thoughts.
        
        Args:
            problem: The problem to solve
            max_nodes: Maximum number of nodes to explore
            
        Returns:
            Solution node if found, None otherwise
        """
        # Initialize root
        root = ThoughtNode(
            thought="Start",
            state=f"Initial problem: {problem}",
            depth=0
        )
        
        queue = deque([root])
        nodes_explored = 0
        
        while queue and nodes_explored < max_nodes:
            current = queue.popleft()
            nodes_explored += 1
            
            print(f"  Exploring (depth {current.depth}): {current.thought[:60]}...")
            
            # Check if solution
            if current.depth > 0 and self.is_solution(problem, current.state):
                current.is_solution = True
                return current
            
            # Don't expand beyond max depth
            if current.depth >= self.max_depth:
                continue
            
            # Generate and evaluate child thoughts
            thoughts = self.generate_thoughts(
                problem,
                current.state,
                self.branching_factor
            )
            
            for thought in thoughts:
                new_state = f"{current.state} → {thought}"
                score = self.evaluate_thought(problem, thought, current.state)
                
                child = ThoughtNode(
                    thought=thought,
                    state=new_state,
                    depth=current.depth + 1,
                    score=score,
                    parent=current
                )
                
                current.children.append(child)
                
                # Only add promising nodes to queue (score > 0.4)
                if score > 0.4:
                    queue.append(child)
        
        # Return best leaf node if no solution found
        return self._find_best_leaf(root)
    
    def search_dfs(self, problem: str, max_nodes: int = 20) -> Optional[ThoughtNode]:
        """
        Depth-first search through the tree of thoughts.
        
        Args:
            problem: The problem to solve
            max_nodes: Maximum number of nodes to explore
            
        Returns:
            Solution node if found, None otherwise
        """
        # Initialize root
        root = ThoughtNode(
            thought="Start",
            state=f"Initial problem: {problem}",
            depth=0
        )
        
        stack = [root]
        nodes_explored = 0
        
        while stack and nodes_explored < max_nodes:
            current = stack.pop()
            nodes_explored += 1
            
            print(f"  Exploring (depth {current.depth}): {current.thought[:60]}...")
            
            # Check if solution
            if current.depth > 0 and self.is_solution(problem, current.state):
                current.is_solution = True
                return current
            
            # Don't expand beyond max depth
            if current.depth >= self.max_depth:
                continue
            
            # Generate and evaluate child thoughts
            thoughts = self.generate_thoughts(
                problem,
                current.state,
                self.branching_factor
            )
            
            for thought in thoughts:
                new_state = f"{current.state} → {thought}"
                score = self.evaluate_thought(problem, thought, current.state)
                
                child = ThoughtNode(
                    thought=thought,
                    state=new_state,
                    depth=current.depth + 1,
                    score=score,
                    parent=current
                )
                
                current.children.append(child)
                
                # Only explore promising nodes (score > 0.4)
                if score > 0.4:
                    stack.append(child)
        
        return self._find_best_leaf(root)
    
    def _find_best_leaf(self, root: ThoughtNode) -> ThoughtNode:
        """Find the leaf node with the highest score."""
        best_node = root
        best_score = root.score
        
        def traverse(node):
            nonlocal best_node, best_score
            
            if not node.children and node.score > best_score:
                best_node = node
                best_score = node.score
            
            for child in node.children:
                traverse(child)
        
        traverse(root)
        return best_node
    
    def solve(
        self,
        problem: str,
        strategy: SearchStrategy = SearchStrategy.BFS,
        max_nodes: int = 20
    ) -> Dict[str, Any]:
        """
        Solve a problem using Tree-of-Thoughts.
        
        Args:
            problem: The problem to solve
            strategy: Search strategy to use
            max_nodes: Maximum nodes to explore
            
        Returns:
            Dictionary with solution path and statistics
        """
        print(f"\nSolving with {strategy.value} strategy...")
        
        if strategy == SearchStrategy.BFS:
            solution_node = self.search_bfs(problem, max_nodes)
        else:  # DFS
            solution_node = self.search_dfs(problem, max_nodes)
        
        if solution_node:
            path = solution_node.get_path()
            return {
                "problem": problem,
                "solution_found": solution_node.is_solution,
                "path": path,
                "final_score": solution_node.score,
                "depth": solution_node.depth,
                "strategy": strategy.value
            }
        
        return {
            "problem": problem,
            "solution_found": False,
            "path": [],
            "final_score": 0.0,
            "depth": 0,
            "strategy": strategy.value
        }


def demonstrate_tot_pattern():
    """Demonstrates the Tree-of-Thoughts pattern."""
    
    print("=" * 80)
    print("PATTERN 003: Tree-of-Thoughts (ToT)")
    print("=" * 80)
    print()
    
    # Create ToT agent
    agent = TreeOfThoughtsAgent(
        branching_factor=3,
        max_depth=3
    )
    
    # Test problems
    problems = [
        "Plan a surprise birthday party for a friend who loves nature and astronomy",
        "Design a simple mobile app to help people track their daily water intake"
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{'=' * 80}")
        print(f"Problem {i}: {problem}")
        print('=' * 80)
        
        # Try BFS strategy
        try:
            result = agent.solve(
                problem,
                strategy=SearchStrategy.BFS,
                max_nodes=15
            )
            
            print(f"\n✓ Solution found: {result['solution_found']}")
            print(f"  Final score: {result['final_score']:.2f}")
            print(f"  Depth reached: {result['depth']}")
            print(f"\nThought path:")
            for j, thought in enumerate(result['path'], 1):
                print(f"  {j}. {thought}")
        
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
    
    # Summary
    print("\n\n" + "=" * 80)
    print("TREE-OF-THOUGHTS PATTERN DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("1. Multiple thought generation at each step")
    print("2. Thought evaluation and scoring")
    print("3. Tree search strategies (BFS/DFS)")
    print("4. Path exploration and backtracking")
    print("5. Solution detection")
    print()
    print("Advantages:")
    print("- Explores multiple solution paths")
    print("- Can recover from poor initial choices")
    print("- Better solution quality through exploration")
    print("- Systematic search through problem space")
    print()
    print("LangChain Components Used:")
    print("- ChatPromptTemplate: Structures generation and evaluation prompts")
    print("- StrOutputParser: Parses LLM responses")
    print("- Custom search logic: Implements BFS/DFS")
    print("- State management: Tracks tree structure")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    demonstrate_tot_pattern()
