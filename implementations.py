I'll create sample Python programs demonstrating key agentic AI design patterns. Due to the large number of patterns (170), I'll provide representative examples from each major category that you can adapt for other patterns.

```python patterns/01_react_pattern.py
"""
ReAct Pattern: Reasoning + Acting
Agent alternates between reasoning and taking actions
"""

from typing import List, Dict, Any
import json

class ReActAgent:
    def __init__(self, tools: Dict[str, callable]):
        self.tools = tools
        self.max_iterations = 5
        
    def think(self, question: str, context: str) -> Dict[str, str]:
        """Simulate LLM reasoning about next action"""
        # In real implementation, this would call an LLM
        thought = f"I need to find information about: {question}"
        action = "search"
        action_input = question
        
        return {
            "thought": thought,
            "action": action,
            "action_input": action_input
        }
    
    def act(self, action: str, action_input: str) -> str:
        """Execute the chosen action"""
        if action in self.tools:
            return self.tools[action](action_input)
        return "Action not found"
    
    def run(self, question: str) -> str:
        """Main ReAct loop"""
        context = ""
        
        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1} ---")
            
            # Reasoning step
            decision = self.think(question, context)
            print(f"Thought: {decision['thought']}")
            print(f"Action: {decision['action']}({decision['action_input']})")
            
            # Check if we should finish
            if decision['action'] == 'finish':
                return decision['action_input']
            
            # Acting step
            observation = self.act(decision['action'], decision['action_input'])
            print(f"Observation: {observation}")
            
            context += f"\nAction: {decision['action']}\nObservation: {observation}"
            
        return "Max iterations reached"


# Example tools
def search_tool(query: str) -> str:
    """Simulate a search tool"""
    knowledge_base = {
        "python": "Python is a high-level programming language created by Guido van Rossum",
        "ai": "Artificial Intelligence is the simulation of human intelligence by machines"
    }
    for key in knowledge_base:
        if key in query.lower():
            return knowledge_base[key]
    return "No information found"

def calculator_tool(expression: str) -> str:
    """Simulate a calculator"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except:
        return "Invalid expression"


# Usage
if __name__ == "__main__":
    tools = {
        "search": search_tool,
        "calculator": calculator_tool
    }
    
    agent = ReActAgent(tools)
    result = agent.run("What is Python?")
    print(f"\nFinal Answer: {result}")
```

```python patterns/02_chain_of_thought.py
"""
Chain-of-Thought Pattern
Breaks down complex problems into intermediate reasoning steps
"""

from typing import List, Tuple

class ChainOfThoughtAgent:
    def __init__(self):
        self.steps = []
    
    def solve_step_by_step(self, problem: str) -> str:
        """Solve problem with explicit reasoning steps"""
        
        # Example: Math word problem
        if "total" in problem.lower() and "cost" in problem.lower():
            return self._solve_math_problem(problem)
        
        return "Unknown problem type"
    
    def _solve_math_problem(self, problem: str) -> str:
        """Example: Solving a math word problem step by step"""
        
        # Step 1: Understand the problem
        step1 = "Let me understand what we're looking for: the total cost"
        self.steps.append(step1)
        print(f"Step 1: {step1}")
        
        # Step 2: Identify given information
        step2 = "Given: 3 apples at $2 each, 2 oranges at $3 each"
        self.steps.append(step2)
        print(f"Step 2: {step2}")
        
        # Step 3: Calculate apple cost
        apple_cost = 3 * 2
        step3 = f"Cost of apples: 3 × $2 = ${apple_cost}"
        self.steps.append(step3)
        print(f"Step 3: {step3}")
        
        # Step 4: Calculate orange cost
        orange_cost = 2 * 3
        step4 = f"Cost of oranges: 2 × $3 = ${orange_cost}"
        self.steps.append(step4)
        print(f"Step 4: {step4}")
        
        # Step 5: Calculate total
        total = apple_cost + orange_cost
        step5 = f"Total cost: ${apple_cost} + ${orange_cost} = ${total}"
        self.steps.append(step5)
        print(f"Step 5: {step5}")
        
        return f"The total cost is ${total}"
    
    def get_reasoning_chain(self) -> List[str]:
        """Return all reasoning steps"""
        return self.steps


# Usage
if __name__ == "__main__":
    agent = ChainOfThoughtAgent()
    problem = "If I buy 3 apples at $2 each and 2 oranges at $3 each, what is the total cost?"
    
    print("Problem:", problem)
    print("\nSolving step by step:\n")
    
    answer = agent.solve_step_by_step(problem)
    print(f"\nFinal Answer: {answer}")
    
    print("\n--- Complete Reasoning Chain ---")
    for i, step in enumerate(agent.get_reasoning_chain(), 1):
        print(f"{i}. {step}")
```

```python patterns/03_tree_of_thoughts.py
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
```

```python patterns/04_plan_and_execute.py
"""
Plan-and-Execute Pattern
Separates planning from execution phases
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: int
    description: str
    dependencies: List[int]
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None

class PlanAndExecuteAgent:
    def __init__(self):
        self.tasks: List[Task] = []
        self.task_counter = 0
        
    def plan(self, goal: str) -> List[Task]:
        """Create a plan to achieve the goal"""
        print(f"\n=== Planning Phase ===")
        print(f"Goal: {goal}\n")
        
        # Example: Planning a data analysis workflow
        if "analyze data" in goal.lower():
            self.tasks = [
                Task(0, "Load data from source", []),
                Task(1, "Clean and preprocess data", [0]),
                Task(2, "Perform statistical analysis", [1]),
                Task(3, "Create visualizations", [1]),
                Task(4, "Generate report", [2, 3])
            ]
        else:
            # Generic planning
            self.tasks = [
                Task(0, f"Step 1: Understand {goal}", []),
                Task(1, f"Step 2: Gather resources", [0]),
                Task(2, f"Step 3: Execute main task", [1]),
                Task(3, f"Step 4: Verify results", [2])
            ]
        
        print("Generated Plan:")
        for task in self.tasks:
            deps = f" (depends on: {task.dependencies})" if task.dependencies else ""
            print(f"  Task {task.id}: {task.description}{deps}")
        
        return self.tasks
    
    def can_execute(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            dep_task = self.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        return True
    
    def execute_task(self, task: Task) -> bool:
        """Execute a single task"""
        print(f"\nExecuting Task {task.id}: {task.description}")
        task.status = TaskStatus.IN_PROGRESS
        
        # Simulate task execution
        try:
            # In real implementation, this would call actual functions/tools
            if "load data" in task.description.lower():
                task.result = {"rows": 1000, "columns": 10}
            elif "clean" in task.description.lower():
                task.result = {"cleaned_rows": 950}
            elif "analysis" in task.description.lower():
                task.result = {"mean": 42.5, "std": 12.3}
            elif "visualization" in task.description.lower():
                task.result = {"charts_created": 3}
            elif "report" in task.description.lower():
                task.result = {"report_path": "/tmp/report.pdf"}
            else:
                task.result = "Task completed"
            
            task.status = TaskStatus.COMPLETED
            print(f"  ✓ Completed: {task.result}")
            return True
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            print(f"  ✗ Failed: {str(e)}")
            return False
    
    def replan(self, failed_task: Task) -> List[Task]:
        """Replan when a task fails"""
        print(f"\n=== Replanning ===")
        print(f"Task {failed_task.id} failed. Creating alternative approach...")
        
        # Insert a new task before the failed one
        new_task = Task(
            id=len(self.tasks),
            description=f"Alternative approach for: {failed_task.description}",
            dependencies=failed_task.dependencies,
            status=TaskStatus.PENDING
        )
        
        self.tasks.append(new_task)
        # Update failed task to depend on new task
        failed_task.dependencies.append(new_task.id)
        failed_task.status = TaskStatus.PENDING
        
        return self.tasks
    
    def execute(self) -> Dict[str, Any]:
        """Execute all tasks in dependency order"""
        print(f"\n=== Execution Phase ===")
        
        max_iterations = 20
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Find executable tasks
            executable = [t for t in self.tasks if 
                         t.status == TaskStatus.PENDING and 
                         self.can_execute(t)]
            
            if not executable:
                # Check if all done
                if all(t.status == TaskStatus.COMPLETED for t in self.tasks):
                    print("\n✓ All tasks completed successfully!")
                    break
                
                # Check for failures
                failed = [t for t in self.tasks if t.status == TaskStatus.FAILED]
                if failed:
                    print(f"\n✗ {len(failed)} task(s) failed")
                    break
                    
                # No tasks ready - might be stuck
                print("\n⚠ No tasks ready to execute")
                break
            
            # Execute ready tasks
            for task in executable:
                success = self.execute_task(task)
                if not success:
                    self.replan(task)
        
        # Gather results
        results = {
            "total_tasks": len(self.tasks),
            "completed": sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self.tasks if t.status == TaskStatus.FAILED),
            "task_results": {t.id: t.result for t in self.tasks if t.result}
        }
        
        return results


# Usage
if __name__ == "__main__":
    agent = PlanAndExecuteAgent()
    
    goal = "Analyze data and create a comprehensive report"
    
    # Planning phase
    plan = agent.plan(goal)
    
    # Execution phase
    results = agent.execute()
    
    # Summary
    print(f"\n=== Execution Summary ===")
    print(f"Total tasks: {results['total_tasks']}")
    print(f"Completed: {results['completed']}")
    print(f"Failed: {results['failed']}")
```

```python patterns/05_reflexion.py
"""
Reflexion Pattern
Agent reflects on past failures to improve future performance
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Experience:
    task: str
    action: str
    outcome: str
    success: bool
    timestamp: datetime
    reflection: str = ""

class ReflexionAgent:
    def __init__(self):
        self.memory: List[Experience] = []
        self.max_attempts = 3
        
    def attempt_task(self, task: str, attempt_num: int) -> Tuple[str, bool]:
        """Attempt to complete a task"""
        print(f"\n--- Attempt {attempt_num} ---")
        print(f"Task: {task}")
        
        # Retrieve relevant past experiences
        relevant_memories = self.retrieve_relevant_experiences(task)
        
        # Use reflections to improve current attempt
        strategy = self.plan_with_reflection(task, relevant_memories)
        print(f"Strategy: {strategy}")
        
        # Simulate task execution
        action, success = self.execute(task, strategy, attempt_num)
        
        return action, success
    
    def retrieve_relevant_experiences(self, task: str) -> List[Experience]:
        """Retrieve similar past experiences"""
        # Simple keyword matching (in real implementation, use embeddings)
        relevant = []
        task_keywords = set(task.lower().split())
        
        for exp in self.memory:
            exp_keywords = set(exp.task.lower().split())
            if task_keywords & exp_keywords:  # If any overlap
                relevant.append(exp)
        
        return relevant
    
    def plan_with_reflection(self, task: str, memories: List[Experience]) -> str:
        """Plan strategy based on past reflections"""
        if not memories:
            return "Try standard approach"
        
        # Learn from failures
        failures = [m for m in memories if not m.success]
        if failures:
            latest_failure = failures[-1]
            return f"Avoid: {latest_failure.reflection}. Try alternative approach."
        
        # Learn from successes
        successes = [m for m in memories if m.success]
        if successes:
            latest_success = successes[-1]
            return f"Replicate successful approach: {latest_success.reflection}"
        
        return "Try standard approach"
    
    def execute(self, task: str, strategy: str, attempt: int) -> Tuple[str, bool]:
        """Execute the task (simulated)"""
        # Simulate task execution with increasing success probability
        import random
        
        # Example: Code generation task
        if "generate code" in task.lower():
            if attempt == 1:
                action = "Generated code without error handling"
                success = False
            elif attempt == 2:
                action = "Generated code with basic error handling"
                success = random.random() > 0.3
            else:
                action = "Generated code with comprehensive error handling and tests"
                success = True
        else:
            action = f"Executed strategy: {strategy}"
            success = random.random() > (0.7 - attempt * 0.2)
        
        print(f"Action: {action}")
        print(f"Success: {success}")
        
        return action, success
    
    def reflect(self, task: str, action: str, success: bool) -> str:
        """Reflect on the outcome and learn"""
        print("\n--- Reflection ---")
        
        if success:
            reflection = f"Success! The approach '{action}' worked well for '{task}'"
        else:
            # Analyze what went wrong
            if "error handling" in action.lower() and not success:
                reflection = "Failure: Need to add more comprehensive error handling"
            elif "without" in action.lower():
                reflection = "Failure: The basic approach was insufficient, need to add safety measures"
            else:
                reflection = "Failure: Current approach didn't work, need to reconsider strategy"
        
        print(f"Reflection: {reflection}")
        return reflection
    
    def learn_from_experience(self, experience: Experience):
        """Store experience in memory"""
        self.memory.append(experience)
        print(f"Learned: Stored experience in memory (total experiences: {len(self.memory)})")
    
    def run(self, task: str) -> bool:
        """Run task with reflexion loop"""
        print(f"\n{'='*60}")
        print(f"Starting Reflexion Loop for: {task}")
        print(f"{'='*60}")
        
        for attempt in range(1, self.max_attempts + 1):
            # Attempt task
            action, success = self.attempt_task(task, attempt)
            
            # Reflect on outcome
            reflection = self.reflect(task, action, success)
            
            # Store experience
            experience = Experience(
                task=task,
                action=action,
                outcome="success" if success else "failure",
                success=success,
                timestamp=datetime.now(),
                reflection=reflection
            )
            self.learn_from_experience(experience)
            
            # If successful, stop
            if success:
                print(f"\n✓ Task completed successfully on attempt {attempt}")
                return True
            
            # If not last attempt, continue loop
            if attempt < self.max_attempts:
                print(f"\n⟲ Retrying with improved approach...")
        
        print(f"\n✗ Task failed after {self.max_attempts} attempts")
        return False


# Usage
if __name__ == "__main__":
    agent = ReflexionAgent()
    
    # First task
    task1 = "Generate code for file processing"
    agent.run(task1)
    
    print("\n" + "="*60)
    print("Now attempting a similar task...")
    print("="*60)
    
    # Similar task - should benefit from previous reflection
    task2 = "Generate code for data processing"
    agent.run(task2)
    
    # Show memory
    print(f"\n=== Agent Memory ({len(agent.memory)} experiences) ===")
    for i, exp in enumerate(agent.memory, 1):
        print(f"\n{i}. Task: {exp.task}")
        print(f"   Action: {exp.action}")
        print(f"   Success: {exp.success}")
        print(f"   Reflection: {exp.reflection}")
```

```python patterns/06_rag_pattern.py
"""
Retrieval-Augmented Generation (RAG) Pattern
Retrieves relevant information before generating response
"""

from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict
    embedding: np.ndarray = None

class SimpleVectorStore:
    """Simple in-memory vector store"""
    
    def __init__(self):
        self.documents: List[Document] = []
    
    def add_document(self, doc: Document):
        """Add document with embedding"""
        if doc.embedding is None:
            doc.embedding = self._create_embedding(doc.content)
        self.documents.append(doc)
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """Create simple embedding (in reality, use a proper embedding model)"""
        # Simple word-based embedding for demonstration
        words = text.lower().split()
        # Create a simple hash-based embedding
        embedding = np.zeros(128)
        for word in words:
            idx = hash(word) % 128
            embedding[idx] += 1
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        query_embedding = self._create_embedding(query)
        
        # Calculate cosine similarity
        results = []
        for doc in self.documents:
            similarity = np.dot(query_embedding, doc.embedding)
            results.append((doc, similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

class RAGAgent:
    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """Retrieve relevant documents"""
        print(f"\n=== Retrieval Phase ===")
        print(f"Query: {query}")
        
        results = self.vector_store.search(query, top_k)
        
        print(f"\nRetrieved {len(results)} relevant documents:")
        retrieved_docs = []
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{i}. [Score: {score:.3f}] {doc.metadata.get('title', 'Untitled')}")
            print(f"   {doc.content[:200]}...")
            retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def generate(self, query: str, context_docs: List[Document]) -> str:
        """Generate answer using retrieved context"""
        print(f"\n=== Generation Phase ===")
        
        # Combine context
        context = "\n\n".join([
            f"Source {i+1}: {doc.content}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Simulated LLM generation (in real implementation, call actual LLM)
        print("Generating answer based on retrieved context...")
        
        # Simple rule-based generation for demonstration
        answer = self._simple_generate(query, context_docs)
        
        return answer
    
    def _simple_generate(self, query: str, docs: List[Document]) -> str:
        """Simple answer generation (simulation)"""
        # Extract relevant sentences
        query_words = set(query.lower().split())
        relevant_parts = []
        
        for doc in docs:
            sentences = doc.content.split('.')
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                if query_words & sentence_words:
                    relevant_parts.append(sentence.strip())
        
        if relevant_parts:
            answer = f"Based on the retrieved information: {' '.join(relevant_parts[:2])}."
        else:
            answer = "I couldn't find specific information to answer your question."
        
        # Add source attribution
        sources = [doc.metadata.get('title', 'Unknown') for doc in docs]
        answer += f"\n\nSources: {', '.join(sources)}"
        
        return answer
    
    def query(self, question: str, top_k: int = 3) -> str:
        """Complete RAG pipeline: retrieve + generate"""
        print(f"\n{'='*60}")
        print(f"RAG Query Pipeline")
        print(f"{'='*60}")
        
        # Step 1: Retrieve relevant documents
        relevant_docs = self.retrieve(question, top_k)
        
        # Step 2: Generate answer using retrieved context
        answer = self.generate(question, relevant_docs)
        
        print(f"\n=== Final Answer ===")
        print(answer)
        
        return answer


# Usage
if __name__ == "__main__":
    # Create vector store and add documents
    vector_store = SimpleVectorStore()
    
    # Add sample documents
    documents = [
        Document(
            id="doc1",
            content="Python is a high-level, interpreted programming language. "
                   "It was created by Guido van Rossum and first released in 1991. "
                   "Python emphasizes code readability and supports multiple programming paradigms.",
            metadata={"title": "Python Programming Language", "source": "tech_wiki"}
        ),
        Document(
            id="doc2",
            content="Machine learning is a subset of artificial intelligence. "
                   "It focuses on building systems that can learn from data. "
                   "Python is widely used for machine learning due to libraries like scikit-learn and TensorFlow.",
            metadata={"title": "Machine Learning Overview", "source": "ai_guide"}
        ),
        Document(
            id="doc3",
            content="Web development with Python can be done using frameworks like Django and Flask. "
                   "Django is a high-level framework that encourages rapid development. "
                   "Flask is a lightweight micro-framework for building web applications.",
            metadata={"title": "Python Web Frameworks", "source": "web_dev_guide"}
        ),
        Document(
            id="doc4",
            content="Data science involves extracting insights from data. "
                   "Python is the most popular language for data science. "
                   "Libraries like Pandas, NumPy, and Matplotlib are essential tools.",
            metadata={"title": "Data Science with Python", "source": "data_guide"}
        ),
    ]
    
    for doc in documents:
        vector_store.add_document(doc)
    
    print(f"Loaded {len(documents)} documents into vector store\n")
    
    # Create RAG agent
    agent = RAGAgent(vector_store)
    
    # Query 1
    agent.query("What is Python and who created it?")
    
    # Query 2
    print("\n" + "="*80 + "\n")
    agent.query("How is Python used in machine learning?")
    
    # Query 3
    print("\n" + "="*80 + "\n")
    agent.query("What frameworks are available for web development?")
```

```python patterns/07_multi_agent_debate.py
"""
Multi-Agent Debate Pattern
Multiple agents debate to reach better conclusions
"""

from typing import List, Dict
from dataclasses import dataclass
from enum import Enum

class AgentRole(Enum):
    PROPOSER = "proposer"
    CRITIC = "critic"
    MODERATOR = "moderator"

@dataclass
class Argument:
    agent_id: str
    role: AgentRole
    content: str
    round_num: int

class DebateAgent:
    def __init__(self, agent_id: str, role: AgentRole, perspective: str):
        self.agent_id = agent_id
        self.role = role
        self.perspective = perspective
        self.arguments: List[Argument] = []
    
    def generate_argument(self, topic: str, round_num: int, previous_args: List[Argument]) -> Argument:
        """Generate argument based on role and previous arguments"""
        
        if self.role == AgentRole.PROPOSER:
            content = self._propose(topic, round_num, previous_args)
        elif self.role == AgentRole.CRITIC:
            content = self._critique(topic, round_num, previous_args)
        else:
            content = self._moderate(topic, round_num, previous_args)
        
        arg = Argument(
            agent_id=self.agent_id,
            role=self.role,
            content=content,
            round_num=round_num
        )
        self.arguments.append(arg)
        return arg
    
    def _propose(self, topic: str, round_num: int, previous_args: List[Argument]) -> str:
        """Generate a proposal"""
        if round_num == 1:
            return f"[{self.perspective}] I propose that {topic} because it offers significant benefits."
        else:
            # Respond to criticisms
            critics = [a for a in previous_args if a.role == AgentRole.CRITIC]
            if critics:
                latest_critique = critics[-1]
                return f"[{self.perspective}] Addressing the criticism: while there are concerns, the benefits outweigh them."
        return f"[{self.perspective}] Reiterating the core proposal with refinements."
    
    def _critique(self, topic: str, round_num: int, previous_args: List[Argument]) -> str:
        """Generate a critique"""
        proposals = [a for a in previous_args if a.role == AgentRole.PROPOSER]
        if proposals:
            latest_proposal = proposals[-1]
            return f"[{self.perspective}] I have concerns about this proposal. We need to consider potential drawbacks and risks."
        return f"[{self.perspective}] Critical analysis suggests we need more evidence."
    
    def _moderate(self, topic: str, round_num: int, previous_args: List[Argument]) -> str:
        """Moderate the debate"""
        return f"[Moderator] Let's synthesize both viewpoints and find common ground."

class MultiAgentDebate:
    def __init__(self, topic: str, max_rounds: int = 3):
        self.topic = topic
        self.max_rounds = max_rounds
        self.agents: List[DebateAgent] = []
        self.debate_history: List[Argument] = []
    
    def add_agent(self, agent: DebateAgent):
        """Add an agent to the debate"""
        self.agents.append(agent)
    
    def conduct_debate(self) -> str:
        """Run the debate for specified rounds"""
        print(f"\n{'='*70}")
        print(f"Multi-Agent Debate: {self.topic}")
        print(f"{'='*70}\n")
        
        for round_num in range(1, self.max_rounds + 1):
            print(f"\n--- Round {round_num} ---\n")
            
            # Each agent presents their argument
            for agent in self.agents:
                arg = agent.generate_argument(self.topic, round_num, self.debate_history)
                self.debate_history.append(arg)
                
                print(f"{agent.role.value.upper()} ({agent.agent_id}):")
                print(f"  {arg.content}\n")
        
        # Final synthesis
        conclusion = self._synthesize_conclusion()
        return conclusion
    
    def _synthesize_conclusion(self) -> str:
        """Synthesize final conclusion from debate"""
        print(f"\n{'='*70}")
        print("DEBATE CONCLUSION")
        print(f"{'='*70}\n")
        
        # Count arguments by role
        proposer_args = [a for a in self.debate_history if a.role == AgentRole.PROPOSER]
        critic_args = [a for a in self.debate_history if a.role == AgentRole.CRITIC]
        
        conclusion = (
            f"After {self.max_rounds} rounds of debate:\n\n"
            f"PROPOSER VIEW ({len(proposer_args)} arguments):\n"
            f"  - Emphasized benefits and positive aspects\n"
            f"  - Addressed criticisms with counter-arguments\n\n"
            f"CRITIC VIEW ({len(critic_args)} arguments):\n"
            f"  - Highlighted risks and concerns\n"
            f"  - Provided critical analysis\n\n"
            f"BALANCED CONCLUSION:\n"
            f"  The debate revealed both opportunities and challenges regarding '{self.topic}'.\n"
            f"  A nuanced approach considering both perspectives is recommended."
        )
        
        return conclusion


# Usage
if __name__ == "__main__":
    # Create debate
    debate = MultiAgentDebate(
        topic="adopting AI in healthcare",
        max_rounds=3
    )
    
    # Add agents with different perspectives
    debate.add_agent(DebateAgent(
        agent_id="agent_optimist",
        role=AgentRole.PROPOSER,
        perspective="Optimistic"
    ))
    
    debate.add_agent(DebateAgent(
        agent_id="agent_skeptic",
        role=AgentRole.CRITIC,
        perspective="Skeptical"
    ))
    
    debate.add_agent(DebateAgent(
        agent_id="agent_moderator",
        role=AgentRole.MODERATOR,
        perspective="Neutral"
    ))
    
    # Conduct debate
    conclusion = debate.conduct_debate()
    print(conclusion)
```

```python patterns/08_human_in_the_loop.py
"""
Human-in-the-Loop (HITL) Pattern
Human provides guidance at critical decision points
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"

@dataclass
class Decision:
    id: int
    description: str
    proposed_action: str
    risk_level: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    human_feedback: Optional[str] = None
    final_action: Optional[str] = None

class HumanInTheLoopAgent:
    def __init__(self, auto_approve_low_risk: bool = True):
        self.auto_approve_low_risk = auto_approve_low_risk
        self.decisions: List[Decision] = []
        self.decision_counter = 0
    
    def assess_risk(self, action: str) -> str:
        """Assess risk level of an action"""
        # Simple heuristic for demonstration
        high_risk_keywords = ["delete", "remove", "irreversible", "critical"]
        medium_risk_keywords = ["update", "modify", "change"]
        
        action_lower = action.lower()
        
        if any(keyword in action_lower for keyword in high_risk_keywords):
            return "HIGH"
        elif any(keyword in action_lower for keyword in medium_risk_keywords):
            return "MEDIUM"
        else:
            return "LOW"
    
    def propose_action(self, task: str, action: str) -> Decision:
        """Propose an action for a task"""
        risk_level = self.assess_risk(action)
        
        decision = Decision(
            id=self.decision_counter,
            description=task,
            proposed_action=action,
            risk_level=risk_level
        )
        self.decision_counter += 1
        self.decisions.append(decision)
        
        return decision
    
    def request_human_approval(self, decision: Decision) -> bool:
        """Request human approval (simulated)"""
        print(f"\n{'='*60}")
        print(f"HUMAN APPROVAL REQUIRED")
        print(f"{'='*60}")
        print(f"Decision ID: {decision.id}")
        print(f"Task: {decision.description}")
        print(f"Proposed Action: {decision.proposed_action}")
        print(f"Risk Level: {decision.risk_level}")
        print(f"\nOptions:")
        print(f"  1. Approve")
        print(f"  2. Reject")
        print(f"  3. Modify")
        
        # Simulate human input (in real implementation, get actual user input)
        # For demo, auto-approve low risk, request approval for others
        if decision.risk_level == "LOW" and self.auto_approve_low_risk:
            print(f"\n[AUTO-APPROVED: Low risk action]")
            decision.status = ApprovalStatus.APPROVED
            decision.final_action = decision.proposed_action
            return True
        
        # Simulate human decision
        import random
        choice = random.choice([1, 1, 2, 3])  # Bias toward approval
        
        if choice == 1:
            print(f"\n[HUMAN: Approved]")
            decision.status = ApprovalStatus.APPROVED
            decision.final_action = decision.proposed_action
            return True
        elif choice == 2:
            print(f"\n[HUMAN: Rejected - Too risky]")
            decision.status = ApprovalStatus.REJECTED
            decision.human_feedback = "Action rejected due to high risk"
            return False
        else:
            print(f"\n[HUMAN: Modified - Use safer alternative]")
            decision.status = ApprovalStatus.MODIFIED
            decision.final_action = f"SAFE_MODE: {decision.proposed_action}"
            decision.human_feedback = "Modified to use safer approach"
            return True
    
    def execute_with_approval(self, task: str, action: str) -> Dict[str, Any]:
        """Execute action with human approval"""
        print(f"\n--- New Task ---")
        print(f"Task: {task}")
        print(f"Proposed Action: {action}")
        
        # Create decision
        decision = self.propose_action(task, action)
        
        # Get approval
        approved = self.request_human_approval(decision)
        
        # Execute if approved
        if approved:
            result = self.execute_action(decision.final_action)
            print(f"\n✓ Executed: {decision.final_action}")
            print(f"Result: {result}")
            return {"success": True, "result": result, "decision": decision}
        else:
            print(f"\n✗ Action not executed (rejected by human)")
            return {"success": False, "result": None, "decision": decision}
    
    def execute_action(self, action: str) -> str:
        """Execute the approved action"""
        # Simulate action execution
        return f"Action '{action}' completed successfully"
    
    def get_approval_summary(self) -> Dict[str, Any]:
        """Get summary of all decisions"""
        total = len(self.decisions)
        approved = sum(1 for d in self.decisions if d.status == ApprovalStatus.APPROVED)
        rejected = sum(1 for d in self.decisions if d.status == ApprovalStatus.REJECTED)
        modified = sum(1 for d in self.decisions if d.status == ApprovalStatus.MODIFIED)
        
        return {
            "total_decisions": total,
            "approved": approved,
            "rejected": rejected,
            "modified": modified,
            "approval_rate": approved / total if total > 0 else 0
        }


# Usage
if __name__ == "__main__":
    agent = HumanInTheLoopAgent(auto_approve_low_risk=True)
    
    # Example tasks with different risk levels
    tasks = [
        ("Analyze user data", "Read and analyze user statistics"),
        ("Update user preferences", "Modify user preference settings"),
        ("Clean up database", "Delete old temporary records"),
        ("Send notification", "Send email to users"),
        ("Critical system update", "Update critical system configuration"),
    ]
    
    print("="*60)
    print("HUMAN-IN-THE-LOOP AGENT EXECUTION")
    print("="*60)
    
    for task, action in tasks:
        agent.execute_with_approval(task, action)
    
    # Summary
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    summary = agent.get_approval_summary()
    print(f"\nTotal Decisions: {summary['total_decisions']}")
    print(f"Approved: {summary['approved']}")
    print(f"Rejected: {summary['rejected']}")
    print(f"Modified: {summary['modified']}")
    print(f"Approval Rate: {summary['approval_rate']:.1%}")
    
    print(f"\n--- Decision Log ---")
    for decision in agent.decisions:
        print(f"\n{decision.id}. {decision.description}")
        print(f"   Risk: {decision.risk_level} | Status: {decision.status.value}")
        if decision.human_feedback:
            print(f"   Feedback: {decision.human_feedback}")
```

```python patterns/09_memory_management.py
"""
Memory Management Patterns
Short-term, long-term, and working memory
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json

@dataclass
class Memory:
    content: str
    timestamp: datetime
    memory_type: str  # 'episodic', 'semantic', 'procedural'
    importance: float = 0.5
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ShortTermMemory:
    """Buffer for recent context - limited capacity"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
    
    def add(self, memory: Memory):
        """Add to short-term memory"""
        self.buffer.append(memory)
    
    def get_recent(self, n: int = None) -> List[Memory]:
        """Get n most recent memories"""
        if n is None:
            return list(self.buffer)
        return list(self.buffer)[-n:]
    
    def clear(self):
        """Clear short-term memory"""
        self.buffer.clear()

class LongTermMemory:
    """Persistent storage for important information"""
    
    def __init__(self):
        self.memories: List[Memory] = []
    
    def add(self, memory: Memory):
        """Add to long-term memory"""
        self.memories.append(memory)
    
    def search(self, query: str, top_k: int = 5) -> List[Memory]:
        """Search long-term memory"""
        # Simple keyword search (in reality, use embeddings)
        query_words = set(query.lower().split())
        
        results = []
        for memory in self.memories:
            memory_words = set(memory.content.lower().split())
            overlap = len(query_words & memory_words)
            if overlap > 0:
                # Update access stats
                memory.access_count += 1
                memory.last_access = datetime.now()
                results.append((memory, overlap))
        
        # Sort by relevance and importance
        results.sort(key=lambda x: (x[1], x[0].importance), reverse=True)
        return [mem for mem, _ in results[:top_k]]
    
    def consolidate(self, short_term: ShortTermMemory, importance_threshold: float = 0.6):
        """Move important memories from short-term to long-term"""
        for memory in short_term.get_recent():
            if memory.importance >= importance_threshold:
                self.add(memory)
                print(f"Consolidated to long-term: {memory.content[:50]}...")
    
    def forget(self, max_age_days: int = 30, min_importance: float = 0.3):
        """Remove old, unimportant memories"""
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        before_count = len(self.memories)
        self.memories = [
            m for m in self.memories
            if m.last_access > cutoff_date or m.importance > min_importance
        ]
        after_count = len(self.memories)
        
        forgotten = before_count - after_count
        if forgotten > 0:
            print(f"Forgot {forgotten} old/unimportant memories")

class WorkingMemory:
    """Active workspace for current task"""
    
    def __init__(self):
        self.current_goal: Optional[str] = None
        self.active_context: Dict[str, Any] = {}
        self.intermediate_results: List[Any] = []
    
    def set_goal(self, goal: str):
        """Set current goal"""
        self.current_goal = goal
        self.active_context = {"goal": goal}
    
    def update_context(self, key: str, value: Any):
        """Update active context"""
        self.active_context[key] = value
    
    def add_result(self, result: Any):
        """Add intermediate result"""
        self.intermediate_results.append(result)
    
    def clear(self):
        """Clear working memory"""
        self.current_goal = None
        self.active_context.clear()
        self.intermediate_results.clear()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current working memory state"""
        return {
            "goal": self.current_goal,
            "context": self.active_context,
            "results_count": len(self.intermediate_results)
        }

class MemoryAgent:
    """Agent with comprehensive memory system"""
    
    def __init__(self):
        self.short_term = ShortTermMemory(max_size=10)
        self.long_term = LongTermMemory()
        self.working = WorkingMemory()
    
    def perceive(self, information: str, importance: float = 0.5, memory_type: str = "episodic"):
        """Process new information"""
        memory = Memory(
            content=information,
            timestamp=datetime.now(),
            memory_type=memory_type,
            importance=importance
        )
        
        # Add to short-term memory
        self.short_term.add(memory)
        
        # If important, also add to long-term
        if importance > 0.7:
            self.long_term.add(memory)
        
        print(f"Perceived: {information[:50]}... (importance: {importance})")
    
    def recall(self, query: str) -> List[Memory]:
        """Recall relevant memories"""
        print(f"\nRecalling memories for: {query}")
        
        # Search long-term memory
        lt_memories = self.long_term.search(query, top_k=3)
        
        # Get recent short-term memories
        st_memories = self.short_term.get_recent(5)
        
        # Combine and deduplicate
        all_memories = lt_memories + [m for m in st_memories if m not in lt_memories]
        
        print(f"Found {len(all_memories)} relevant memories")
        return all_memories
    
    def think(self, task: str):
        """Process a task using working memory"""
        print(f"\n--- Thinking about task ---")
        print(f"Task: {task}")
        
        # Set up working memory
        self.working.set_goal(task)
        
        # Recall relevant information
        relevant_memories = self.recall(task)
        
        # Use memories in working memory
        self.working.update_context("relevant_memories", len(relevant_memories))
        
        for i, memory in enumerate(relevant_memories):
            print(f"\nUsing memory {i+1}: {memory.content[:60]}...")
            self.working.add_result(f"Processed memory {i+1}")
        
        # Simulate thinking process
        conclusion = f"Completed analysis of '{task}' using {len(relevant_memories)} memories"
        self.working.add_result(conclusion)
        
        print(f"\nWorking Memory State: {self.working.get_state()}")
        
        return conclusion
    
    def consolidate_memories(self):
        """Consolidate memories from short-term to long-term"""
        print("\n--- Consolidating Memories ---")
        self.long_term.consolidate(self.short_term, importance_threshold=0.6)
    
    def sleep(self):
        """Simulate sleep - consolidate and clean memories"""
        print("\n=== Sleep Cycle ===")
        self.consolidate_memories()
        self.long_term.forget(max_age_days=30, min_importance=0.3)
        self.working.clear()
        print("Sleep cycle complete\n")


# Usage
if __name__ == "__main__":
    agent = MemoryAgent()
    
    print("="*70)
    print("MEMORY MANAGEMENT DEMONSTRATION")
    print("="*70)
    
    # Day 1: Learning
    print("\n--- Day 1: Learning Phase ---")
    agent.perceive("Python is a programming language", importance=0.8, memory_type="semantic")
    agent.perceive("I wrote a hello world program today", importance=0.5, memory_type="episodic")
    agent.perceive("Functions in Python use the 'def' keyword", importance=0.9, memory_type="semantic")
    agent.perceive("Had lunch at 12pm", importance=0.2, memory_type="episodic")
    agent.perceive("Classes in Python use the 'class' keyword", importance=0.9, memory_type="semantic")
    
    # Task execution
    agent.think("How do I define a function in Python?")
    
    # End of day
    agent.sleep()
    
    # Day 2: More learning
    print("\n--- Day 2: Continued Learning ---")
    agent.perceive("Python supports object-oriented programming", importance=0.8, memory_type="semantic")
    agent.perceive("Debugged a tricky issue today", importance=0.6, memory_type="episodic")
    
    # Another task
    agent.think("What do I know about Python programming?")
    
    # Memory stats
    print("\n" + "="*70)
    print("MEMORY STATISTICS")
    print("="*70)
    print(f"Short-term memories: {len(agent.short_term.buffer)}")
    print(f"Long-term memories: {len(agent.long_term.memories)}")
    
    print("\n--- Long-term Memory Contents ---")
    for i, mem in enumerate(agent.long_term.memories, 1):
        print(f"{i}. [{mem.memory_type}] {mem.content}")
        print(f"   Importance: {mem.importance}, Access count: {mem.access_count}")
```

```python patterns/10_self_consistency.py
"""
Self-Consistency Pattern
Generates multiple reasoning paths and selects most consistent answer
"""

from typing import List, Dict, Any, Tuple
from collections import Counter
import random

class SelfConsistencyAgent:
    def __init__(self, num_samples: int = 5):
        self.num_samples = num_samples
        self.reasoning_paths: List[Dict[str, Any]] = []
    
    def generate_reasoning_path(self, problem: str, sample_num: int) -> Dict[str, Any]:
        """Generate one reasoning path (with sampling/variation)"""
        print(f"\n--- Reasoning Path {sample_num} ---")
        
        # Simulate different reasoning approaches
        # In reality, this would use temperature > 0 for LLM sampling
        
        if "math" in problem.lower() or "calculate" in problem.lower():
            return self._solve_math_problem(problem, sample_num)
        else:
            return self._general_reasoning(problem, sample_num)
    
    def _solve_math_problem(self, problem: str, sample_num: int) -> Dict[str, Any]:
        """Solve math problem with variation"""
        # Example: "If John has 3 apples and buys 5 more, how many does he have?"
        
        # Simulate different reasoning approaches
        approaches = [
            {
                "steps": [
                    "Starting amount: 3 apples",
                    "Additional amount: 5 apples",
                    "Total = 3 + 5 = 8 apples"
                ],
                "answer": 8
            },
            {
                "steps": [
                    "Begin with 3",
                    "Add 5 more",
                    "Count: 3, 4, 5, 6, 7, 8",
                    "Final count: 8"
                ],
                "answer": 8
            },
            {
                "steps": [
                    "Total = initial + bought",
                    "Total = 3 + 5",
                    "Total = 8"
                ],
                "answer": 8
            },
            # Occasionally get wrong answer to show self-consistency filtering
            {
                "steps": [
                    "Starting with 3",
                    "Multiply by... wait, add 5",
                    "3 + 5 = 7... no 8"
                ],
                "answer": 8 if random.random() > 0.2 else 7
            }
        ]
        
        chosen_approach = random.choice(approaches)
        
        print("Steps:")
        for step in chosen_approach["steps"]:
            print(f"  - {step}")
        print(f"Answer: {chosen_approach['answer']}")
        
        return {
            "path_id": sample_num,
            "steps": chosen_approach["steps"],
            "answer": chosen_approach["answer"]
        }
    
    def _general_reasoning(self, problem: str, sample_num: int) -> Dict[str, Any]:
        """General reasoning with variation"""
        # Simulate different reasoning paths
        possible_answers = ["Answer A", "Answer B", "Answer A", "Answer A"]  # A is most consistent
        answer = possible_answers[sample_num % len(possible_answers)]
        
        steps = [
            f"Analyzing the problem from perspective {sample_num}",
            f"Considering various factors",
            f"Reaching conclusion: {answer}"
        ]
        
        print("Steps:")
        for step in steps:
            print(f"  - {step}")
        print(f"Answer: {answer}")
        
        return {
            "path_id": sample_num,
            "steps": steps,
            "answer": answer
        }
    
    def aggregate_answers(self, reasoning_paths: List[Dict[str, Any]]) -> Tuple[Any, float]:
        """Aggregate answers using majority voting"""
        answers = [path["answer"] for path in reasoning_paths]
        
        # Count occurrences
        answer_counts = Counter(answers)
        
        # Get most common answer
        most_common_answer, count = answer_counts.most_common(1)[0]
        confidence = count / len(answers)
        
        return most_common_answer, confidence
    
    def solve_with_self_consistency(self, problem: str) -> Dict[str, Any]:
        """Solve problem using self-consistency"""
        print("="*70)
        print("SELF-CONSISTENCY REASONING")
        print("="*70)
        print(f"\nProblem: {problem}")
        print(f"Generating {self.num_samples} reasoning paths...\n")
        
        # Generate multiple reasoning paths
        self.reasoning_paths = []
        for i in range(1, self.num_samples + 1):
            path = self.generate_reasoning_path(problem, i)
            self.reasoning_paths.append(path)
        
        # Aggregate answers
        print("\n" + "="*70)
        print("AGGREGATION")
        print("="*70)
        
        final_answer, confidence = self.aggregate_answers(self.reasoning_paths)
        
        # Show vote distribution
        answer_counts = Counter([path["answer"] for path in self.reasoning_paths])
        print("\nVote Distribution:")
        for answer, count in answer_counts.most_common():
            percentage = (count / self.num_samples) * 100
            print(f"  {answer}: {count}/{self.num_samples} ({percentage:.1f}%)")
        
        print(f"\n{'='*70}")
        print(f"FINAL ANSWER: {final_answer}")
        print(f"Confidence: {confidence:.1%}")
        print(f"{'='*70}")
        
        return {
            "problem": problem,
            "final_answer": final_answer,
            "confidence": confidence,
            "num_paths": self.num_samples,
            "all_answers": answer_counts
        }


# Usage
if __name__ == "__main__":
    agent = SelfConsistencyAgent(num_samples=5)
    
    # Example 1: Math problem
    problem1 = "If John has 3 apples and buys 5 more, how many apples does he have in total?"
    result1 = agent.solve_with_self_consistency(problem1)
    
    print("\n\n" + "="*80 + "\n\n")
    
    # Example 2: General reasoning
    problem2 = "What is the best approach to learn programming?"
    agent2 = SelfConsistencyAgent(num_samples=4)
    result2 = agent2.solve_with_self_consistency(problem2)
```

I've provided 10 comprehensive pattern implementations. Would you like me to continue with more patterns? I can generate examples for:

- Tool use patterns (Function Calling, Code Execution)
- Advanced multi-agent patterns (Swarm, Hierarchical)
- State management patterns
- Workflow orchestration patterns
- And any other specific patterns you're most interested in

Let me know which patterns you'd like to see next!



```python patterns/11_function_calling.py
"""
Function Calling Pattern
Structured way for LLM to call predefined functions
"""

from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
import json
import inspect

@dataclass
class FunctionDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable

class FunctionRegistry:
    """Registry of available functions"""
    
    def __init__(self):
        self.functions: Dict[str, FunctionDefinition] = {}
    
    def register(self, func: Callable, description: str, parameters: Dict[str, Any]):
        """Register a function"""
        func_def = FunctionDefinition(
            name=func.__name__,
            description=description,
            parameters=parameters,
            function=func
        )
        self.functions[func.__name__] = func_def
        print(f"Registered function: {func.__name__}")
    
    def get_function_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI-style function schemas"""
        schemas = []
        for func_def in self.functions.values():
            schema = {
                "name": func_def.name,
                "description": func_def.description,
                "parameters": func_def.parameters
            }
            schemas.append(schema)
        return schemas
    
    def call_function(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a function by name"""
        if name not in self.functions:
            raise ValueError(f"Function {name} not found")
        
        func_def = self.functions[name]
        return func_def.function(**arguments)

class FunctionCallingAgent:
    def __init__(self, registry: FunctionRegistry):
        self.registry = registry
        self.call_history: List[Dict[str, Any]] = []
    
    def decide_function_call(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Decide which function to call (simulates LLM decision)"""
        # In reality, this would be done by an LLM
        # Here we use simple keyword matching for demonstration
        
        user_lower = user_input.lower()
        
        if "weather" in user_lower:
            # Extract city
            words = user_input.split()
            city = "New York"  # Default
            for i, word in enumerate(words):
                if word.lower() in ["in", "at", "for"] and i + 1 < len(words):
                    city = words[i + 1].strip("?.,!")
                    break
            
            return {
                "name": "get_weather",
                "arguments": {"city": city}
            }
        
        elif "calculate" in user_lower or "+" in user_input or "*" in user_input:
            # Extract expression
            import re
            match = re.search(r'(\d+\s*[\+\-\*\/]\s*\d+)', user_input)
            if match:
                return {
                    "name": "calculate",
                    "arguments": {"expression": match.group(1)}
                }
        
        elif "search" in user_lower or "find" in user_lower:
            # Extract query
            query_start = user_input.lower().find("search for ")
            if query_start != -1:
                query = user_input[query_start + 11:].strip("?.,!")
            else:
                query = user_input
            
            return {
                "name": "web_search",
                "arguments": {"query": query}
            }
        
        elif "email" in user_lower or "send" in user_lower:
            return {
                "name": "send_email",
                "arguments": {
                    "to": "user@example.com",
                    "subject": "Message",
                    "body": user_input
                }
            }
        
        return None
    
    def process_request(self, user_input: str) -> str:
        """Process user request with function calling"""
        print(f"\n{'='*70}")
        print(f"User Request: {user_input}")
        print(f"{'='*70}\n")
        
        # Decide if function call is needed
        function_call = self.decide_function_call(user_input)
        
        if function_call is None:
            response = "I don't have a specific function for that request."
            print(f"Response: {response}")
            return response
        
        # Show function call decision
        print(f"Function Call Decision:")
        print(f"  Function: {function_call['name']}")
        print(f"  Arguments: {json.dumps(function_call['arguments'], indent=4)}\n")
        
        # Execute function
        try:
            print(f"Executing function...")
            result = self.registry.call_function(
                function_call['name'],
                function_call['arguments']
            )
            
            # Log the call
            self.call_history.append({
                "user_input": user_input,
                "function": function_call['name'],
                "arguments": function_call['arguments'],
                "result": result
            })
            
            print(f"Function Result: {result}\n")
            
            # Generate natural language response
            response = self.generate_response(user_input, function_call, result)
            print(f"Final Response: {response}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error executing function: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def generate_response(self, user_input: str, function_call: Dict, result: Any) -> str:
        """Generate natural language response from function result"""
        func_name = function_call['name']
        
        if func_name == "get_weather":
            city = function_call['arguments']['city']
            return f"The weather in {city} is {result['condition']} with a temperature of {result['temperature']}°F."
        
        elif func_name == "calculate":
            expr = function_call['arguments']['expression']
            return f"The result of {expr} is {result}."
        
        elif func_name == "web_search":
            return f"I found {len(result)} results for your search: {result[0] if result else 'No results'}"
        
        elif func_name == "send_email":
            return f"Email sent successfully to {function_call['arguments']['to']}."
        
        return f"Function {func_name} completed with result: {result}"


# Define actual functions to be called
def get_weather(city: str) -> Dict[str, Any]:
    """Get weather for a city"""
    # Simulated weather data
    weather_db = {
        "New York": {"temperature": 72, "condition": "Sunny"},
        "London": {"temperature": 65, "condition": "Cloudy"},
        "Tokyo": {"temperature": 78, "condition": "Rainy"},
    }
    return weather_db.get(city, {"temperature": 70, "condition": "Unknown"})

def calculate(expression: str) -> float:
    """Calculate a mathematical expression"""
    try:
        # Safe evaluation (in production, use a proper math parser)
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except:
        return "Error in calculation"

def web_search(query: str) -> List[str]:
    """Perform web search"""
    # Simulated search results
    return [
        f"Result 1 for '{query}'",
        f"Result 2 for '{query}'",
        f"Result 3 for '{query}'"
    ]

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email"""
    # Simulated email sending
    return f"Email sent to {to}"


# Usage
if __name__ == "__main__":
    # Create registry and register functions
    registry = FunctionRegistry()
    
    registry.register(
        func=get_weather,
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name"
                }
            },
            "required": ["city"]
        }
    )
    
    registry.register(
        func=calculate,
        description="Calculate a mathematical expression",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    )
    
    registry.register(
        func=web_search,
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
    )
    
    registry.register(
        func=send_email,
        description="Send an email",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"}
            },
            "required": ["to", "subject", "body"]
        }
    )
    
    # Create agent
    agent = FunctionCallingAgent(registry)
    
    # Test various requests
    requests = [
        "What's the weather in London?",
        "Calculate 15 + 27",
        "Search for artificial intelligence",
        "Send an email reminder"
    ]
    
    for request in requests:
        agent.process_request(request)
        print("\n" + "="*80 + "\n")
```

```python patterns/12_code_execution.py
"""
Code Generation & Execution Pattern
Agent writes and executes code to solve problems
"""

from typing import Dict, Any, List
import sys
from io import StringIO
import traceback
import re

class CodeExecutionAgent:
    def __init__(self):
        self.execution_history: List[Dict[str, Any]] = []
        self.global_namespace: Dict[str, Any] = {}
    
    def generate_code(self, task: str) -> str:
        """Generate code for a task (simulates LLM code generation)"""
        print(f"\n--- Code Generation ---")
        print(f"Task: {task}\n")
        
        # Simple rule-based code generation for demonstration
        # In reality, this would use an LLM
        
        task_lower = task.lower()
        
        if "fibonacci" in task_lower:
            code = '''def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

result = [fibonacci(i) for i in range(10)]
print(f"First 10 Fibonacci numbers: {result}")
'''
        
        elif "prime" in task_lower:
            code = '''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [i for i in range(2, 50) if is_prime(i)]
print(f"Prime numbers up to 50: {primes}")
'''
        
        elif "sort" in task_lower or "data" in task_lower:
            code = '''data = [64, 34, 25, 12, 22, 11, 90]
sorted_data = sorted(data)
print(f"Original: {data}")
print(f"Sorted: {sorted_data}")
print(f"Min: {min(data)}, Max: {max(data)}, Mean: {sum(data)/len(data):.2f}")
'''
        
        elif "plot" in task_lower or "graph" in task_lower:
            code = '''# Simulating plot (actual plotting would require matplotlib)
x = list(range(10))
y = [i**2 for i in x]
print(f"X values: {x}")
print(f"Y values (x^2): {y}")
print("(Plot would be displayed here)")
'''
        
        else:
            code = f'''# Generated code for: {task}
result = "Task completed"
print(result)
'''
        
        print("Generated Code:")
        print("-" * 60)
        print(code)
        print("-" * 60)
        
        return code
    
    def execute_code(self, code: str, timeout: int = 5) -> Dict[str, Any]:
        """Execute code in a controlled environment"""
        print(f"\n--- Code Execution ---")
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        result = {
            "success": False,
            "output": "",
            "error": None,
            "variables": {}
        }
        
        try:
            # Create isolated namespace
            local_namespace = {}
            
            # Execute code
            exec(code, self.global_namespace, local_namespace)
            
            # Capture output
            output = captured_output.getvalue()
            
            # Extract result variables
            result_vars = {
                k: v for k, v in local_namespace.items()
                if not k.startswith('_')
            }
            
            result.update({
                "success": True,
                "output": output,
                "variables": result_vars
            })
            
            print("✓ Execution successful")
            
        except Exception as e:
            error_msg = traceback.format_exc()
            result.update({
                "success": False,
                "error": str(e),
                "traceback": error_msg
            })
            print(f"✗ Execution failed: {str(e)}")
        
        finally:
            sys.stdout = old_stdout
        
        return result
    
    def iterative_code_improvement(self, task: str, max_iterations: int = 3) -> Dict[str, Any]:
        """Iteratively improve code based on execution results"""
        print(f"\n{'='*70}")
        print(f"Iterative Code Improvement")
        print(f"{'='*70}")
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'='*70}")
            print(f"Iteration {iteration}")
            print(f"{'='*70}")
            
            # Generate code
            if iteration == 1:
                code = self.generate_code(task)
            else:
                # In a real implementation, would use error feedback to improve
                print("Improving code based on previous feedback...")
                code = self.generate_code(task)
            
            # Execute code
            exec_result = self.execute_code(code)
            
            # Store in history
            self.execution_history.append({
                "iteration": iteration,
                "task": task,
                "code": code,
                "result": exec_result
            })
            
            # Display results
            if exec_result["success"]:
                print("\nOutput:")
                print(exec_result["output"])
                
                if exec_result["variables"]:
                    print("\nVariables:")
                    for var_name, var_value in exec_result["variables"].items():
                        print(f"  {var_name} = {var_value}")
                
                print(f"\n✓ Task completed successfully on iteration {iteration}")
                return exec_result
            
            else:
                print("\nError occurred:")
                print(exec_result["error"])
                
                if iteration < max_iterations:
                    print(f"\nRetrying with improved code...")
        
        print(f"\n✗ Failed to complete task after {max_iterations} iterations")
        return exec_result
    
    def explain_code(self, code: str) -> str:
        """Explain what the code does"""
        # Simple explanation based on code structure
        lines = code.strip().split('\n')
        explanation = "Code explanation:\n"
        
        for line in lines:
            line = line.strip()
            if line.startswith('def '):
                func_name = line.split('(')[0].replace('def ', '')
                explanation += f"- Defines function '{func_name}'\n"
            elif 'print' in line:
                explanation += f"- Outputs results\n"
            elif '=' in line and not line.startswith('#'):
                var_name = line.split('=')[0].strip()
                explanation += f"- Creates variable '{var_name}'\n"
        
        return explanation


# Usage
if __name__ == "__main__":
    agent = CodeExecutionAgent()
    
    # Test different tasks
    tasks = [
        "Generate first 10 Fibonacci numbers",
        "Find prime numbers up to 50",
        "Sort a list of numbers and show statistics"
    ]
    
    for task in tasks:
        result = agent.iterative_code_improvement(task, max_iterations=1)
        print("\n" + "="*80 + "\n")
    
    # Show execution history
    print("="*70)
    print("Execution History Summary")
    print("="*70)
    for i, record in enumerate(agent.execution_history, 1):
        status = "✓" if record["result"]["success"] else "✗"
        print(f"\n{i}. {status} {record['task']}")
        print(f"   Iteration: {record['iteration']}")
```

```python patterns/13_workflow_orchestration.py
"""
Workflow Orchestration Pattern
Manages complex multi-step workflows with dependencies
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowStep:
    id: str
    name: str
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class WorkflowContext:
    """Shared context across workflow steps"""
    data: Dict[str, Any] = field(default_factory=dict)
    
    def set(self, key: str, value: Any):
        self.data[key] = value
    
    def get(self, key: str, default=None) -> Any:
        return self.data.get(key, default)

class WorkflowOrchestrator:
    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.steps: Dict[str, WorkflowStep] = {}
        self.context = WorkflowContext()
        self.execution_order: List[str] = []
    
    def add_step(self, step: WorkflowStep):
        """Add a step to the workflow"""
        self.steps[step.id] = step
        print(f"Added step: {step.name} (ID: {step.id})")
    
    def can_execute_step(self, step: WorkflowStep) -> bool:
        """Check if step dependencies are satisfied"""
        for dep_id in step.dependencies:
            if dep_id not in self.steps:
                return False
            dep_step = self.steps[dep_id]
            if dep_step.status != StepStatus.COMPLETED:
                return False
        return True
    
    def execute_step(self, step: WorkflowStep) -> bool:
        """Execute a single workflow step"""
        print(f"\n{'='*60}")
        print(f"Executing: {step.name}")
        print(f"{'='*60}")
        
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now()
        
        try:
            # Execute the step function with context
            print(f"Running function: {step.function.__name__}")
            result = step.function(self.context)
            
            step.result = result
            step.status = StepStatus.COMPLETED
            step.end_time = datetime.now()
            
            duration = (step.end_time - step.start_time).total_seconds()
            print(f"✓ Completed in {duration:.2f}s")
            print(f"Result: {result}")
            
            return True
            
        except Exception as e:
            step.error = str(e)
            step.end_time = datetime.now()
            
            # Retry logic
            if step.retry_count < step.max_retries:
                step.retry_count += 1
                step.status = StepStatus.PENDING
                print(f"⚠ Failed (attempt {step.retry_count}/{step.max_retries}): {str(e)}")
                print(f"Will retry...")
                time.sleep(1)  # Brief delay before retry
                return self.execute_step(step)  # Recursive retry
            else:
                step.status = StepStatus.FAILED
                print(f"✗ Failed after {step.max_retries} retries: {str(e)}")
                return False
    
    def build_execution_order(self) -> List[str]:
        """Build topologically sorted execution order"""
        # Simple topological sort
        visited = set()
        order = []
        
        def visit(step_id: str):
            if step_id in visited:
                return
            visited.add(step_id)
            
            step = self.steps[step_id]
            for dep_id in step.dependencies:
                visit(dep_id)
            
            order.append(step_id)
        
        for step_id in self.steps:
            visit(step_id)
        
        return order
    
    def execute_workflow(self, parallel: bool = False) -> Dict[str, Any]:
        """Execute the entire workflow"""
        print(f"\n{'='*70}")
        print(f"WORKFLOW: {self.workflow_name}")
        print(f"{'='*70}")
        print(f"Total steps: {len(self.steps)}")
        print(f"Execution mode: {'Parallel' if parallel else 'Sequential'}")
        
        # Build execution order
        self.execution_order = self.build_execution_order()
        print(f"\nExecution order: {' -> '.join(self.execution_order)}")
        
        workflow_start = datetime.now()
        
        if parallel:
            # Parallel execution (simplified - real impl would use threading/async)
            self._execute_parallel()
        else:
            # Sequential execution
            self._execute_sequential()
        
        workflow_end = datetime.now()
        duration = (workflow_end - workflow_start).total_seconds()
        
        # Generate summary
        summary = self.get_summary()
        summary['total_duration'] = duration
        
        self._print_summary(summary)
        
        return summary
    
    def _execute_sequential(self):
        """Execute workflow sequentially"""
        for step_id in self.execution_order:
            step = self.steps[step_id]
            
            if not self.can_execute_step(step):
                print(f"\n⚠ Skipping {step.name}: dependencies not met")
                step.status = StepStatus.SKIPPED
                continue
            
            success = self.execute_step(step)
            
            if not success and step.status == StepStatus.FAILED:
                print(f"\n✗ Workflow failed at step: {step.name}")
                break
    
    def _execute_parallel(self):
        """Execute workflow with parallelization where possible"""
        # Simplified parallel execution
        executed = set()
        
        while len(executed) < len(self.steps):
            # Find steps that can execute now
            ready_steps = [
                step for step in self.steps.values()
                if step.id not in executed and
                step.status == StepStatus.PENDING and
                self.can_execute_step(step)
            ]
            
            if not ready_steps:
                break
            
            # Execute ready steps (in real impl, use threading)
            for step in ready_steps:
                self.execute_step(step)
                executed.add(step.id)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get workflow execution summary"""
        total = len(self.steps)
        completed = sum(1 for s in self.steps.values() if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.steps.values() if s.status == StepStatus.FAILED)
        skipped = sum(1 for s in self.steps.values() if s.status == StepStatus.SKIPPED)
        
        return {
            'workflow_name': self.workflow_name,
            'total_steps': total,
            'completed': completed,
            'failed': failed,
            'skipped': skipped,
            'success_rate': completed / total if total > 0 else 0,
            'steps': {
                step_id: {
                    'name': step.name,
                    'status': step.status.value,
                    'duration': (step.end_time - step.start_time).total_seconds() 
                                if step.start_time and step.end_time else None,
                    'retries': step.retry_count
                }
                for step_id, step in self.steps.items()
            }
        }
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print workflow summary"""
        print(f"\n{'='*70}")
        print(f"WORKFLOW SUMMARY")
        print(f"{'='*70}")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Completed: {summary['completed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Duration: {summary.get('total_duration', 0):.2f}s")
        
        print(f"\nStep Details:")
        for step_id, step_info in summary['steps'].items():
            duration_str = f"{step_info['duration']:.2f}s" if step_info['duration'] else "N/A"
            retry_str = f" ({step_info['retries']} retries)" if step_info['retries'] > 0 else ""
            print(f"  {step_info['name']}: {step_info['status']} - {duration_str}{retry_str}")


# Example workflow functions
def load_data(context: WorkflowContext) -> Dict[str, Any]:
    """Step 1: Load data"""
    time.sleep(0.5)  # Simulate work
    data = {"records": 1000, "source": "database"}
    context.set("raw_data", data)
    return data

def validate_data(context: WorkflowContext) -> Dict[str, Any]:
    """Step 2: Validate data"""
    time.sleep(0.3)
    raw_data = context.get("raw_data")
    validated = {"valid_records": raw_data["records"] - 10, "invalid": 10}
    context.set("validated_data", validated)
    return validated

def transform_data(context: WorkflowContext) -> Dict[str, Any]:
    """Step 3: Transform data"""
    time.sleep(0.4)
    validated = context.get("validated_data")
    transformed = {"processed_records": validated["valid_records"]}
    context.set("transformed_data", transformed)
    return transformed

def analyze_data(context: WorkflowContext) -> Dict[str, Any]:
    """Step 4: Analyze data"""
    time.sleep(0.6)
    transformed = context.get("transformed_data")
    analysis = {
        "total_processed": transformed["processed_records"],
        "insights": ["Insight 1", "Insight 2"]
    }
    context.set("analysis", analysis)
    return analysis

def generate_report(context: WorkflowContext) -> str:
    """Step 5: Generate report"""
    time.sleep(0.3)
    analysis = context.get("analysis")
    report = f"Report: Processed {analysis['total_processed']} records"
    context.set("report", report)
    return report


# Usage
if __name__ == "__main__":
    # Create workflow
    workflow = WorkflowOrchestrator("Data Processing Pipeline")
    
    # Add steps with dependencies
    workflow.add_step(WorkflowStep(
        id="load",
        name="Load Data",
        function=load_data,
        dependencies=[]
    ))
    
    workflow.add_step(WorkflowStep(
        id="validate",
        name="Validate Data",
        function=validate_data,
        dependencies=["load"]
    ))
    
    workflow.add_step(WorkflowStep(
        id="transform",
        name="Transform Data",
        function=transform_data,
        dependencies=["validate"]
    ))
    
    workflow.add_step(WorkflowStep(
        id="analyze",
        name="Analyze Data",
        function=analyze_data,
        dependencies=["transform"]
    ))
    
    workflow.add_step(WorkflowStep(
        id="report",
        name="Generate Report",
        function=generate_report,
        dependencies=["analyze"]
    ))
    
    # Execute workflow
    summary = workflow.execute_workflow(parallel=False)
```


```python patterns/14_guardrails.py
"""
Guardrails Pattern
Input/output validation and filtering for safety
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import re

class GuardrailType(Enum):
    INPUT_VALIDATION = "input_validation"
    OUTPUT_FILTERING = "output_filtering"
    CONTENT_SAFETY = "content_safety"
    RATE_LIMITING = "rate_limiting"
    DATA_PRIVACY = "data_privacy"

@dataclass
class GuardrailViolation:
    type: GuardrailType
    severity: str  # "low", "medium", "high"
    message: str
    details: Dict[str, Any]

class Guardrail:
    """Base class for guardrails"""
    
    def __init__(self, name: str, guardrail_type: GuardrailType):
        self.name = name
        self.guardrail_type = guardrail_type
    
    def check(self, content: str, context: Dict[str, Any] = None) -> Optional[GuardrailViolation]:
        """Check if content violates guardrail"""
        raise NotImplementedError

class ContentSafetyGuardrail(Guardrail):
    """Check for unsafe content"""
    
    def __init__(self):
        super().__init__("Content Safety", GuardrailType.CONTENT_SAFETY)
        self.toxic_keywords = [
            "hate", "violence", "illegal", "harmful"
        ]
        self.sensitive_patterns = [
            r'\b(?:password|secret|api[_-]?key)\b'
        ]
    
    def check(self, content: str, context: Dict[str, Any] = None) -> Optional[GuardrailViolation]:
        content_lower = content.lower()
        
        # Check for toxic keywords
        for keyword in self.toxic_keywords:
            if keyword in content_lower:
                return GuardrailViolation(
                    type=self.guardrail_type,
                    severity="high",
                    message=f"Potentially toxic content detected: '{keyword}'",
                    details={"keyword": keyword}
                )
        
        # Check for sensitive patterns
        for pattern in self.sensitive_patterns:
            if re.search(pattern, content_lower):
                return GuardrailViolation(
                    type=self.guardrail_type,
                    severity="high",
                    message="Sensitive information detected",
                    details={"pattern": pattern}
                )
        
        return None

class PIIGuardrail(Guardrail):
    """Check for Personally Identifiable Information"""
    
    def __init__(self):
        super().__init__("PII Detection", GuardrailType.DATA_PRIVACY)
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        }
    
    def check(self, content: str, context: Dict[str, Any] = None) -> Optional[GuardrailViolation]:
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                return GuardrailViolation(
                    type=self.guardrail_type,
                    severity="high",
                    message=f"PII detected: {pii_type}",
                    details={"pii_type": pii_type, "count": len(matches)}
                )
        
        return None

class InputValidationGuardrail(Guardrail):
    """Validate input format and constraints"""
    
    def __init__(self, max_length: int = 1000, min_length: int = 1):
        super().__init__("Input Validation", GuardrailType.INPUT_VALIDATION)
        self.max_length = max_length
        self.min_length = min_length
    
    def check(self, content: str, context: Dict[str, Any] = None) -> Optional[GuardrailViolation]:
        # Check length
        if len(content) > self.max_length:
            return GuardrailViolation(
                type=self.guardrail_type,
                severity="medium",
                message=f"Input exceeds maximum length of {self.max_length}",
                details={"length": len(content), "max": self.max_length}
            )
        
        if len(content) < self.min_length:
            return GuardrailViolation(
                type=self.guardrail_type,
                severity="low",
                message=f"Input below minimum length of {self.min_length}",
                details={"length": len(content), "min": self.min_length}
            )
        
        # Check for injection attempts
        injection_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'(\bUNION\b|\bSELECT\b|\bDROP\b|\bINSERT\b)',  # SQL injection
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return GuardrailViolation(
                    type=self.guardrail_type,
                    severity="high",
                    message="Potential injection attempt detected",
                    details={"pattern": pattern}
                )
        
        return None

class OutputFilteringGuardrail(Guardrail):
    """Filter and sanitize output"""
    
    def __init__(self):
        super().__init__("Output Filtering", GuardrailType.OUTPUT_FILTERING)
        self.prohibited_phrases = [
            "i am an ai",
            "as an ai language model",
            "i don't have personal opinions"
        ]
    
    def check(self, content: str, context: Dict[str, Any] = None) -> Optional[GuardrailViolation]:
        content_lower = content.lower()
        
        # Check for unwanted AI disclosure phrases
        for phrase in self.prohibited_phrases:
            if phrase in content_lower:
                return GuardrailViolation(
                    type=self.guardrail_type,
                    severity="low",
                    message="Output contains unwanted AI disclosure",
                    details={"phrase": phrase}
                )
        
        # Check for incomplete sentences
        if content and not content.rstrip().endswith(('.', '!', '?')):
            return GuardrailViolation(
                type=self.guardrail_type,
                severity="low",
                message="Output appears incomplete",
                details={"last_char": content[-1] if content else None}
            )
        
        return None

class RateLimitGuardrail(Guardrail):
    """Rate limiting guardrail"""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__("Rate Limiting", GuardrailType.RATE_LIMITING)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_log: Dict[str, List[float]] = {}
    
    def check(self, content: str, context: Dict[str, Any] = None) -> Optional[GuardrailViolation]:
        import time
        
        user_id = context.get("user_id", "default") if context else "default"
        current_time = time.time()
        
        # Initialize log for user
        if user_id not in self.request_log:
            self.request_log[user_id] = []
        
        # Clean old requests
        cutoff_time = current_time - self.window_seconds
        self.request_log[user_id] = [
            t for t in self.request_log[user_id] if t > cutoff_time
        ]
        
        # Check rate limit
        if len(self.request_log[user_id]) >= self.max_requests:
            return GuardrailViolation(
                type=self.guardrail_type,
                severity="medium",
                message=f"Rate limit exceeded: {self.max_requests} requests per {self.window_seconds}s",
                details={
                    "user_id": user_id,
                    "current_count": len(self.request_log[user_id]),
                    "limit": self.max_requests
                }
            )
        
        # Log this request
        self.request_log[user_id].append(current_time)
        
        return None

class GuardrailSystem:
    """System to manage and apply multiple guardrails"""
    
    def __init__(self):
        self.input_guardrails: List[Guardrail] = []
        self.output_guardrails: List[Guardrail] = []
        self.violations_log: List[GuardrailViolation] = []
    
    def add_input_guardrail(self, guardrail: Guardrail):
        """Add input validation guardrail"""
        self.input_guardrails.append(guardrail)
        print(f"Added input guardrail: {guardrail.name}")
    
    def add_output_guardrail(self, guardrail: Guardrail):
        """Add output filtering guardrail"""
        self.output_guardrails.append(guardrail)
        print(f"Added output guardrail: {guardrail.name}")
    
    def validate_input(self, content: str, context: Dict[str, Any] = None) -> tuple[bool, List[GuardrailViolation]]:
        """Validate input against all input guardrails"""
        print(f"\n{'='*60}")
        print("INPUT VALIDATION")
        print(f"{'='*60}")
        print(f"Content: {content[:100]}...")
        
        violations = []
        
        for guardrail in self.input_guardrails:
            print(f"\nChecking: {guardrail.name}")
            violation = guardrail.check(content, context)
            
            if violation:
                violations.append(violation)
                self.violations_log.append(violation)
                print(f"  ⚠ Violation: {violation.message} (severity: {violation.severity})")
            else:
                print(f"  ✓ Passed")
        
        is_valid = not any(v.severity == "high" for v in violations)
        
        print(f"\n{'='*60}")
        if is_valid:
            print("✓ Input validation PASSED")
        else:
            print("✗ Input validation FAILED")
        print(f"{'='*60}")
        
        return is_valid, violations
    
    def filter_output(self, content: str, context: Dict[str, Any] = None) -> tuple[str, List[GuardrailViolation]]:
        """Filter output through all output guardrails"""
        print(f"\n{'='*60}")
        print("OUTPUT FILTERING")
        print(f"{'='*60}")
        print(f"Content: {content[:100]}...")
        
        violations = []
        filtered_content = content
        
        for guardrail in self.output_guardrails:
            print(f"\nChecking: {guardrail.name}")
            violation = guardrail.check(filtered_content, context)
            
            if violation:
                violations.append(violation)
                self.violations_log.append(violation)
                print(f"  ⚠ Violation: {violation.message} (severity: {violation.severity})")
                
                # Apply filtering based on severity
                if violation.severity == "high":
                    filtered_content = "[Content filtered due to policy violation]"
                elif violation.severity == "medium":
                    # Apply sanitization
                    filtered_content = self._sanitize_content(filtered_content, violation)
            else:
                print(f"  ✓ Passed")
        
        print(f"\n{'='*60}")
        print("✓ Output filtering complete")
        print(f"{'='*60}")
        
        return filtered_content, violations
    
    def _sanitize_content(self, content: str, violation: GuardrailViolation) -> str:
        """Sanitize content based on violation"""
        # Simple sanitization - in reality, would be more sophisticated
        if violation.type == GuardrailType.DATA_PRIVACY:
            # Redact PII
            content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', content)
            content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', content)
        
        return content
    
    def get_violations_summary(self) -> Dict[str, Any]:
        """Get summary of all violations"""
        total = len(self.violations_log)
        by_severity = {
            "high": sum(1 for v in self.violations_log if v.severity == "high"),
            "medium": sum(1 for v in self.violations_log if v.severity == "medium"),
            "low": sum(1 for v in self.violations_log if v.severity == "low")
        }
        by_type = {}
        for v in self.violations_log:
            type_name = v.type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        return {
            "total_violations": total,
            "by_severity": by_severity,
            "by_type": by_type
        }


# Usage
if __name__ == "__main__":
    # Create guardrail system
    system = GuardrailSystem()
    
    # Add input guardrails
    system.add_input_guardrail(InputValidationGuardrail(max_length=500))
    system.add_input_guardrail(ContentSafetyGuardrail())
    system.add_input_guardrail(PIIGuardrail())
    system.add_input_guardrail(RateLimitGuardrail(max_requests=5, window_seconds=60))
    
    # Add output guardrails
    system.add_output_guardrail(OutputFilteringGuardrail())
    system.add_output_guardrail(PIIGuardrail())
    
    print("\n" + "="*80)
    print("GUARDRAIL SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Test cases
    test_inputs = [
        {
            "content": "What is the weather today?",
            "context": {"user_id": "user123"}
        },
        {
            "content": "My email is john@example.com and my SSN is 123-45-6789",
            "context": {"user_id": "user123"}
        },
        {
            "content": "How to create harmful software?",
            "context": {"user_id": "user123"}
        },
        {
            "content": "Tell me about Python programming",
            "context": {"user_id": "user123"}
        }
    ]
    
    for i, test in enumerate(test_inputs, 1):
        print(f"\n\n{'='*80}")
        print(f"TEST CASE {i}")
        print(f"{'='*80}")
        
        # Validate input
        is_valid, input_violations = system.validate_input(
            test["content"],
            test["context"]
        )
        
        if is_valid:
            # Simulate agent response
            response = f"I understand your question about '{test['content']}'. As an AI language model, here's my response..."
            
            # Filter output
            filtered_response, output_violations = system.filter_output(
                response,
                test["context"]
            )
            
            print(f"\nFiltered Response: {filtered_response}")
        else:
            print("\n✗ Request blocked due to input violations")
    
    # Show summary
    print(f"\n\n{'='*80}")
    print("VIOLATIONS SUMMARY")
    print(f"{'='*80}")
    
    summary = system.get_violations_summary()
    print(f"\nTotal Violations: {summary['total_violations']}")
    print(f"\nBy Severity:")
    for severity, count in summary['by_severity'].items():
        print(f"  {severity}: {count}")
    print(f"\nBy Type:")
    for type_name, count in summary['by_type'].items():
        print(f"  {type_name}: {count}")
```

```python patterns/15_ensemble_agents.py
"""
Ensemble/Committee Pattern
Multiple agents work independently, results are aggregated
"""

from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from collections import Counter
import statistics

@dataclass
class AgentPrediction:
    agent_id: str
    prediction: Any
    confidence: float
    reasoning: str

class EnsembleAgent:
    def __init__(self, agent_id: str, model_type: str):
        self.agent_id = agent_id
        self.model_type = model_type
    
    def predict(self, input_data: Any) -> AgentPrediction:
        """Make a prediction (simulated)"""
        # In reality, each agent would use a different model
        # Here we simulate different prediction strategies
        
        if self.model_type == "optimistic":
            prediction = self._optimistic_predict(input_data)
        elif self.model_type == "pessimistic":
            prediction = self._pessimistic_predict(input_data)
        elif self.model_type == "balanced":
            prediction = self._balanced_predict(input_data)
        else:
            prediction = self._random_predict(input_data)
        
        return prediction
    
    def _optimistic_predict(self, input_data: Any) -> AgentPrediction:
        """Optimistic prediction strategy"""
        return AgentPrediction(
            agent_id=self.agent_id,
            prediction="positive",
            confidence=0.85,
            reasoning="Analysis shows positive indicators"
        )
    
    def _pessimistic_predict(self, input_data: Any) -> AgentPrediction:
        """Pessimistic prediction strategy"""
        return AgentPrediction(
            agent_id=self.agent_id,
            prediction="negative",
            confidence=0.75,
            reasoning="Risk factors identified"
        )
    
    def _balanced_predict(self, input_data: Any) -> AgentPrediction:
        """Balanced prediction strategy"""
        import random
        prediction = random.choice(["positive", "negative", "neutral"])
        return AgentPrediction(
            agent_id=self.agent_id,
            prediction=prediction,
            confidence=0.70,
            reasoning="Balanced analysis of factors"
        )
    
    def _random_predict(self, input_data: Any) -> AgentPrediction:
        """Random prediction"""
        import random
        return AgentPrediction(
            agent_id=self.agent_id,
            prediction=random.choice(["positive", "negative"]),
            confidence=random.uniform(0.5, 0.9),
            reasoning="Random model prediction"
        )

class EnsembleSystem:
    """Ensemble system that aggregates multiple agent predictions"""
    
    def __init__(self, aggregation_method: str = "voting"):
        self.agents: List[EnsembleAgent] = []
        self.aggregation_method = aggregation_method
        self.prediction_history: List[Dict[str, Any]] = []
    
    def add_agent(self, agent: EnsembleAgent):
        """Add an agent to the ensemble"""
        self.agents.append(agent)
        print(f"Added agent: {agent.agent_id} ({agent.model_type})")
    
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """Get predictions from all agents and aggregate"""
        print(f"\n{'='*70}")
        print(f"ENSEMBLE PREDICTION")
        print(f"{'='*70}")
        print(f"Input: {input_data}")
        print(f"Number of agents: {len(self.agents)}")
        print(f"Aggregation method: {self.aggregation_method}\n")
        
        # Collect predictions from all agents
        predictions: List[AgentPrediction] = []
        
        print("Individual Agent Predictions:")
        print("-" * 70)
        
        for agent in self.agents:
            prediction = agent.predict(input_data)
            predictions.append(prediction)
            
            print(f"\n{prediction.agent_id} ({agent.model_type}):")
            print(f"  Prediction: {prediction.prediction}")
            print(f"  Confidence: {prediction.confidence:.2%}")
            print(f"  Reasoning: {prediction.reasoning}")
        
        # Aggregate predictions
        print(f"\n{'='*70}")
        print("AGGREGATION")
        print(f"{'='*70}\n")
        
        if self.aggregation_method == "voting":
            result = self._majority_voting(predictions)
        elif self.aggregation_method == "weighted":
            result = self._weighted_voting(predictions)
        elif self.aggregation_method == "averaging":
            result = self._averaging(predictions)
        else:
            result = self._majority_voting(predictions)
        
        # Store in history
        self.prediction_history.append({
            "input": input_data,
            "individual_predictions": predictions,
            "final_result": result
        })
        
        return result
    
    def _majority_voting(self, predictions: List[AgentPrediction]) -> Dict[str, Any]:
        """Aggregate using majority voting"""
        votes = [p.prediction for p in predictions]
        vote_counts = Counter(votes)
        
        majority_prediction, count = vote_counts.most_common(1)[0]
        agreement = count / len(predictions)
        
        print(f"Vote Distribution:")
        for prediction, count in vote_counts.most_common():
            percentage = (count / len(predictions)) * 100
            print(f"  {prediction}: {count}/{len(predictions)} ({percentage:.1f}%)")
        
        print(f"\nMajority Prediction: {majority_prediction}")
        print(f"Agreement: {agreement:.1%}")
        
        return {
            "method": "majority_voting",
            "prediction": majority_prediction,
            "confidence": agreement,
            "vote_distribution": dict(vote_counts),
            "num_agents": len(predictions)
        }
    
    def _weighted_voting(self, predictions: List[AgentPrediction]) -> Dict[str, Any]:
        """Aggregate using confidence-weighted voting"""
        weighted_votes: Dict[str, float] = {}
        total_weight = 0
        
        for pred in predictions:
            weighted_votes[pred.prediction] = weighted_votes.get(pred.prediction, 0) + pred.confidence
            total_weight += pred.confidence
        
        # Normalize and find winner
        for pred in weighted_votes:
            weighted_votes[pred] /= total_weight
        
        winner = max(weighted_votes.items(), key=lambda x: x[1])
        
        print(f"Weighted Vote Distribution:")
        for prediction, weight in sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True):
            print(f"  {prediction}: {weight:.1%}")
        
        print(f"\nWeighted Winner: {winner[0]}")
        print(f"Weight: {winner[1]:.1%}")
        
        return {
            "method": "weighted_voting",
            "prediction": winner[0],
            "confidence": winner[1],
            "weighted_distribution": weighted_votes,
            "num_agents": len(predictions)
        }
    
    def _averaging(self, predictions: List[AgentPrediction]) -> Dict[str, Any]:
        """Aggregate by averaging confidence scores"""
        confidences = [p.confidence for p in predictions]
        
        avg_confidence = statistics.mean(confidences)
        std_confidence = statistics.stdev(confidences) if len(confidences) > 1 else 0
        
        # Use majority for categorical prediction
        votes = [p.prediction for p in predictions]
        majority_prediction = Counter(votes).most_common(1)[0][0]
        
        print(f"Confidence Statistics:")
        print(f"  Mean: {avg_confidence:.2%}")
        print(f"  Std Dev: {std_confidence:.2%}")
        print(f"  Min: {min(confidences):.2%}")
        print(f"  Max: {max(confidences):.2%}")
        
        print(f"\nPrediction: {majority_prediction}")
        print(f"Average Confidence: {avg_confidence:.1%}")
        
        return {
            "method": "averaging",
            "prediction": majority_prediction,
            "confidence": avg_confidence,
            "std_dev": std_confidence,
            "num_agents": len(predictions)
        }
    
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get ensemble performance statistics"""
        if not self.prediction_history:
            return {"total_predictions": 0}
        
        total = len(self.prediction_history)
        avg_confidence = statistics.mean([
            p["final_result"]["confidence"]
            for p in self.prediction_history
        ])
        
        return {
            "total_predictions": total,
            "num_agents": len(self.agents),
            "average_confidence": avg_confidence,
            "aggregation_method": self.aggregation_method
        }


# Usage
if __name__ == "__main__":
    # Create ensemble system
    ensemble = EnsembleSystem(aggregation_method="weighted")
    
    # Add diverse agents
    ensemble.add_agent(EnsembleAgent("agent_1", "optimistic"))
    ensemble.add_agent(EnsembleAgent("agent_2", "pessimistic"))
    ensemble.add_agent(EnsembleAgent("agent_3", "balanced"))
    ensemble.add_agent(EnsembleAgent("agent_4", "balanced"))
    ensemble.add_agent(EnsembleAgent("agent_5", "optimistic"))
    
    print("\n" + "="*80)
    print("ENSEMBLE SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Make predictions
    test_cases = [
        "Market analysis for Q4",
        "Risk assessment for new project",
        "Customer sentiment analysis"
    ]
    
    for test_case in test_cases:
        result = ensemble.predict(test_case)
        print(f"\n{'='*70}")
        print(f"FINAL RESULT")
        print(f"{'='*70}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Method: {result['method']}")
        print("\n" + "="*80 + "\n")
    
    # Show ensemble stats
    stats = ensemble.get_ensemble_stats()
    print("="*70)
    print("ENSEMBLE STATISTICS")
    print("="*70)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2%}" if value < 1 else f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
```

These implementations demonstrate additional key patterns. Would you like me to continue with more patterns? I can create examples for:

- Advanced multi-agent patterns (Swarm Intelligence, Hierarchical Multi-Agent)
- State machine agents
- Monitoring & observability
- Caching patterns
- A/B testing patterns
- More specialized patterns

Let me know which ones you'd like to see next!



```python patterns/16_swarm_intelligence.py
"""
Swarm Intelligence Pattern
Many simple agents collaborate to solve problems
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import random
import math

@dataclass
class Position:
    x: float
    y: float
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Particle:
    """Individual particle in the swarm"""
    id: int
    position: Position
    velocity: Position
    best_position: Position
    best_score: float
    
    def __post_init__(self):
        if self.best_position is None:
            self.best_position = Position(self.position.x, self.position.y)

class SwarmAgent:
    """Individual agent in swarm"""
    
    def __init__(self, agent_id: int, position: Position):
        self.agent_id = agent_id
        self.position = position
        self.velocity = Position(random.uniform(-1, 1), random.uniform(-1, 1))
        self.best_position = Position(position.x, position.y)
        self.best_score = float('-inf')
        self.neighbors: List['SwarmAgent'] = []
    
    def evaluate_position(self, objective_function) -> float:
        """Evaluate current position using objective function"""
        score = objective_function(self.position.x, self.position.y)
        
        # Update personal best
        if score > self.best_score:
            self.best_score = score
            self.best_position = Position(self.position.x, self.position.y)
        
        return score
    
    def update_velocity(self, global_best: Position, w: float = 0.5, c1: float = 1.5, c2: float = 1.5):
        """Update velocity based on personal and global best"""
        # Inertia component
        inertia_x = w * self.velocity.x
        inertia_y = w * self.velocity.y
        
        # Cognitive component (personal best)
        r1 = random.random()
        cognitive_x = c1 * r1 * (self.best_position.x - self.position.x)
        cognitive_y = c1 * r1 * (self.best_position.y - self.position.y)
        
        # Social component (global best)
        r2 = random.random()
        social_x = c2 * r2 * (global_best.x - self.position.x)
        social_y = c2 * r2 * (global_best.y - self.position.y)
        
        # Update velocity
        self.velocity.x = inertia_x + cognitive_x + social_x
        self.velocity.y = inertia_y + cognitive_y + social_y
        
        # Limit velocity
        max_velocity = 2.0
        velocity_magnitude = math.sqrt(self.velocity.x**2 + self.velocity.y**2)
        if velocity_magnitude > max_velocity:
            self.velocity.x = (self.velocity.x / velocity_magnitude) * max_velocity
            self.velocity.y = (self.velocity.y / velocity_magnitude) * max_velocity
    
    def update_position(self, bounds: Tuple[float, float, float, float]):
        """Update position based on velocity"""
        min_x, max_x, min_y, max_y = bounds
        
        self.position.x += self.velocity.x
        self.position.y += self.velocity.y
        
        # Keep within bounds
        self.position.x = max(min_x, min(max_x, self.position.x))
        self.position.y = max(min_y, min(max_y, self.position.y))

class ParticleSwarmOptimizer:
    """Particle Swarm Optimization system"""
    
    def __init__(self, num_agents: int, bounds: Tuple[float, float, float, float]):
        self.num_agents = num_agents
        self.bounds = bounds  # (min_x, max_x, min_y, max_y)
        self.agents: List[SwarmAgent] = []
        self.global_best_position: Position = None
        self.global_best_score: float = float('-inf')
        self.iteration_history: List[Dict[str, Any]] = []
        
        self._initialize_swarm()
    
    def _initialize_swarm(self):
        """Initialize swarm with random positions"""
        min_x, max_x, min_y, max_y = self.bounds
        
        for i in range(self.num_agents):
            position = Position(
                x=random.uniform(min_x, max_x),
                y=random.uniform(min_y, max_y)
            )
            agent = SwarmAgent(i, position)
            self.agents.append(agent)
        
        print(f"Initialized swarm with {self.num_agents} agents")
    
    def optimize(self, objective_function, max_iterations: int = 50, verbose: bool = True) -> Dict[str, Any]:
        """Run particle swarm optimization"""
        print(f"\n{'='*70}")
        print(f"PARTICLE SWARM OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Swarm size: {self.num_agents}")
        print(f"Max iterations: {max_iterations}")
        print(f"Search bounds: {self.bounds}\n")
        
        for iteration in range(max_iterations):
            # Evaluate all agents
            scores = []
            for agent in self.agents:
                score = agent.evaluate_position(objective_function)
                scores.append(score)
                
                # Update global best
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = Position(
                        agent.position.x,
                        agent.position.y
                    )
            
            # Record iteration stats
            avg_score = sum(scores) / len(scores)
            self.iteration_history.append({
                'iteration': iteration,
                'best_score': self.global_best_score,
                'avg_score': avg_score,
                'best_position': (self.global_best_position.x, self.global_best_position.y)
            })
            
            if verbose and (iteration % 10 == 0 or iteration == max_iterations - 1):
                print(f"Iteration {iteration:3d}: Best={self.global_best_score:.4f}, "
                      f"Avg={avg_score:.4f}, "
                      f"Pos=({self.global_best_position.x:.2f}, {self.global_best_position.y:.2f})")
            
            # Update all agents
            for agent in self.agents:
                agent.update_velocity(self.global_best_position)
                agent.update_position(self.bounds)
        
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Best Score: {self.global_best_score:.4f}")
        print(f"Best Position: ({self.global_best_position.x:.4f}, {self.global_best_position.y:.4f})")
        
        return {
            'best_score': self.global_best_score,
            'best_position': (self.global_best_position.x, self.global_best_position.y),
            'iterations': max_iterations,
            'history': self.iteration_history
        }
    
    def get_swarm_state(self) -> List[Dict[str, Any]]:
        """Get current state of all agents"""
        return [
            {
                'id': agent.agent_id,
                'position': (agent.position.x, agent.position.y),
                'velocity': (agent.velocity.x, agent.velocity.y),
                'best_score': agent.best_score
            }
            for agent in self.agents
        ]

class AntColonyOptimizer:
    """Ant Colony Optimization for path finding"""
    
    def __init__(self, num_ants: int, num_nodes: int):
        self.num_ants = num_ants
        self.num_nodes = num_nodes
        self.pheromone: List[List[float]] = [[1.0] * num_nodes for _ in range(num_nodes)]
        self.best_path: List[int] = []
        self.best_distance: float = float('inf')
    
    def optimize(self, distance_matrix: List[List[float]], iterations: int = 100, 
                 alpha: float = 1.0, beta: float = 2.0, evaporation: float = 0.5) -> Dict[str, Any]:
        """Run ant colony optimization"""
        print(f"\n{'='*70}")
        print(f"ANT COLONY OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Number of ants: {self.num_ants}")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Iterations: {iterations}\n")
        
        for iteration in range(iterations):
            paths = []
            distances = []
            
            # Each ant constructs a solution
            for ant in range(self.num_ants):
                path = self._construct_path(distance_matrix, alpha, beta)
                distance = self._calculate_path_distance(path, distance_matrix)
                
                paths.append(path)
                distances.append(distance)
                
                # Update best solution
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path.copy()
            
            # Update pheromones
            self._update_pheromones(paths, distances, evaporation)
            
            if iteration % 20 == 0:
                avg_distance = sum(distances) / len(distances)
                print(f"Iteration {iteration:3d}: Best={self.best_distance:.2f}, Avg={avg_distance:.2f}")
        
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Best Path: {self.best_path}")
        print(f"Best Distance: {self.best_distance:.2f}")
        
        return {
            'best_path': self.best_path,
            'best_distance': self.best_distance,
            'iterations': iterations
        }
    
    def _construct_path(self, distance_matrix: List[List[float]], alpha: float, beta: float) -> List[int]:
        """Construct path for one ant"""
        path = [0]  # Start at node 0
        unvisited = set(range(1, self.num_nodes))
        
        while unvisited:
            current = path[-1]
            next_node = self._select_next_node(current, unvisited, distance_matrix, alpha, beta)
            path.append(next_node)
            unvisited.remove(next_node)
        
        return path
    
    def _select_next_node(self, current: int, unvisited: set, distance_matrix: List[List[float]], 
                          alpha: float, beta: float) -> int:
        """Select next node using pheromone and distance"""
        probabilities = []
        
        for node in unvisited:
            pheromone = self.pheromone[current][node] ** alpha
            distance = (1.0 / distance_matrix[current][node]) ** beta
            probabilities.append((node, pheromone * distance))
        
        # Normalize probabilities
        total = sum(p[1] for p in probabilities)
        probabilities = [(node, prob / total) for node, prob in probabilities]
        
        # Select node using roulette wheel selection
        r = random.random()
        cumulative = 0
        for node, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                return node
        
        return probabilities[-1][0]  # Fallback
    
    def _calculate_path_distance(self, path: List[int], distance_matrix: List[List[float]]) -> float:
        """Calculate total distance of path"""
        distance = 0
        for i in range(len(path) - 1):
            distance += distance_matrix[path[i]][path[i + 1]]
        # Return to start
        distance += distance_matrix[path[-1]][path[0]]
        return distance
    
    def _update_pheromones(self, paths: List[List[int]], distances: List[float], evaporation: float):
        """Update pheromone levels"""
        # Evaporation
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.pheromone[i][j] *= (1 - evaporation)
        
        # Add new pheromones
        for path, distance in zip(paths, distances):
            deposit = 1.0 / distance
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += deposit
                self.pheromone[path[i + 1]][path[i]] += deposit


# Usage
if __name__ == "__main__":
    print("="*80)
    print("SWARM INTELLIGENCE DEMONSTRATIONS")
    print("="*80)
    
    # Example 1: Particle Swarm Optimization
    print("\n" + "="*80)
    print("EXAMPLE 1: Particle Swarm Optimization")
    print("="*80)
    
    # Define objective function (find maximum)
    def objective_function(x: float, y: float) -> float:
        """Example: Find maximum of -(x^2 + y^2) + 10"""
        return -(x**2 + y**2) + 10
    
    # Create and run PSO
    pso = ParticleSwarmOptimizer(
        num_agents=30,
        bounds=(-5.0, 5.0, -5.0, 5.0)
    )
    
    result = pso.optimize(objective_function, max_iterations=50)
    
    # Example 2: Ant Colony Optimization
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Ant Colony Optimization (TSP)")
    print("="*80)
    
    # Create distance matrix for 5 cities
    num_cities = 5
    distance_matrix = [
        [0, 2, 3, 4, 5],
        [2, 0, 4, 5, 3],
        [3, 4, 0, 2, 4],
        [4, 5, 2, 0, 3],
        [5, 3, 4, 3, 0]
    ]
    
    aco = AntColonyOptimizer(num_ants=10, num_nodes=num_cities)
    result = aco.optimize(distance_matrix, iterations=100)
```

```python patterns/17_state_machine_agent.py
"""
State Machine Agent Pattern
Agent behavior defined by explicit states and transitions
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

class State(Enum):
    """Possible agent states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"

@dataclass
class Transition:
    """State transition definition"""
    from_state: State
    to_state: State
    condition: Callable[[Dict[str, Any]], bool]
    action: Optional[Callable[[Dict[str, Any]], None]] = None
    name: str = ""

@dataclass
class StateEntry:
    """Log entry for state history"""
    state: State
    timestamp: datetime
    duration: float = 0.0
    metadata: Dict[str, Any] = None

class StateMachineAgent:
    """Agent that operates as a state machine"""
    
    def __init__(self, initial_state: State = State.IDLE):
        self.current_state = initial_state
        self.transitions: List[Transition] = []
        self.state_history: List[StateEntry] = []
        self.context: Dict[str, Any] = {}
        self.state_enter_callbacks: Dict[State, List[Callable]] = {}
        self.state_exit_callbacks: Dict[State, List[Callable]] = {}
        
        # Record initial state
        self._record_state_entry(initial_state)
    
    def add_transition(self, transition: Transition):
        """Add a state transition"""
        self.transitions.append(transition)
        print(f"Added transition: {transition.from_state.value} -> {transition.to_state.value}")
    
    def on_enter_state(self, state: State, callback: Callable):
        """Register callback for entering a state"""
        if state not in self.state_enter_callbacks:
            self.state_enter_callbacks[state] = []
        self.state_enter_callbacks[state].append(callback)
    
    def on_exit_state(self, state: State, callback: Callable):
        """Register callback for exiting a state"""
        if state not in self.state_exit_callbacks:
            self.state_exit_callbacks[state] = []
        self.state_exit_callbacks[state].append(callback)
    
    def transition_to(self, new_state: State, metadata: Dict[str, Any] = None):
        """Manually transition to a new state"""
        if new_state == self.current_state:
            return
        
        print(f"\n{'='*60}")
        print(f"STATE TRANSITION: {self.current_state.value} -> {new_state.value}")
        print(f"{'='*60}")
        
        # Call exit callbacks
        if self.current_state in self.state_exit_callbacks:
            for callback in self.state_exit_callbacks[self.current_state]:
                callback(self.context)
        
        # Update state history
        if self.state_history:
            last_entry = self.state_history[-1]
            last_entry.duration = (datetime.now() - last_entry.timestamp).total_seconds()
        
        old_state = self.current_state
        self.current_state = new_state
        
        # Record new state
        self._record_state_entry(new_state, metadata)
        
        # Call enter callbacks
        if new_state in self.state_enter_callbacks:
            for callback in self.state_enter_callbacks[new_state]:
                callback(self.context)
        
        print(f"Transition complete: {old_state.value} -> {new_state.value}")
    
    def _record_state_entry(self, state: State, metadata: Dict[str, Any] = None):
        """Record state entry in history"""
        entry = StateEntry(
            state=state,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.state_history.append(entry)
    
    def update(self, event: Dict[str, Any] = None):
        """Update state machine (check for transitions)"""
        if event:
            self.context.update(event)
        
        # Check all transitions from current state
        for transition in self.transitions:
            if transition.from_state != self.current_state:
                continue
            
            # Check if transition condition is met
            if transition.condition(self.context):
                # Execute transition action if defined
                if transition.action:
                    print(f"Executing transition action: {transition.name}")
                    transition.action(self.context)
                
                # Perform transition
                self.transition_to(transition.to_state, {"trigger": transition.name})
                return True
        
        return False
    
    def run_cycle(self, max_iterations: int = 10):
        """Run state machine for multiple update cycles"""
        print(f"\n{'='*70}")
        print(f"RUNNING STATE MACHINE")
        print(f"{'='*70}")
        print(f"Initial State: {self.current_state.value}")
        print(f"Max Iterations: {max_iterations}\n")
        
        for i in range(max_iterations):
            print(f"\n--- Cycle {i + 1} ---")
            print(f"Current State: {self.current_state.value}")
            
            # Update state machine
            transitioned = self.update()
            
            if not transitioned:
                print(f"No transition occurred")
            
            # Check if we've reached a terminal state
            if self.current_state in [State.COMPLETED, State.ERROR]:
                print(f"\nReached terminal state: {self.current_state.value}")
                break
            
            # Simulate some work in current state
            import time
            time.sleep(0.1)
        
        self._print_summary()
    
    def _print_summary(self):
        """Print execution summary"""
        print(f"\n{'='*70}")
        print(f"STATE MACHINE SUMMARY")
        print(f"{'='*70}")
        print(f"Final State: {self.current_state.value}")
        print(f"Total States Visited: {len(self.state_history)}")
        
        print(f"\nState History:")
        for i, entry in enumerate(self.state_history, 1):
            duration_str = f"{entry.duration:.2f}s" if entry.duration > 0 else "current"
            print(f"  {i}. {entry.state.value} - {duration_str}")
            if entry.metadata:
                print(f"     Metadata: {entry.metadata}")
        
        # Calculate time in each state
        state_durations: Dict[State, float] = {}
        for entry in self.state_history:
            state_durations[entry.state] = state_durations.get(entry.state, 0) + entry.duration
        
        if state_durations:
            print(f"\nTime in Each State:")
            for state, duration in state_durations.items():
                print(f"  {state.value}: {duration:.2f}s")


# Example: Task Processing Agent
class TaskProcessingAgent(StateMachineAgent):
    """Agent that processes tasks through defined states"""
    
    def __init__(self):
        super().__init__(initial_state=State.IDLE)
        self._setup_state_machine()
    
    def _setup_state_machine(self):
        """Setup state transitions and callbacks"""
        
        # Define transitions
        # IDLE -> LISTENING when task received
        self.add_transition(Transition(
            from_state=State.IDLE,
            to_state=State.LISTENING,
            condition=lambda ctx: ctx.get('task_received', False),
            action=lambda ctx: print("  Action: Preparing to listen"),
            name="task_received"
        ))
        
        # LISTENING -> PROCESSING when input complete
        self.add_transition(Transition(
            from_state=State.LISTENING,
            to_state=State.PROCESSING,
            condition=lambda ctx: ctx.get('input_complete', False),
            action=lambda ctx: print("  Action: Starting processing"),
            name="input_complete"
        ))
        
        # PROCESSING -> EXECUTING when ready to execute
        self.add_transition(Transition(
            from_state=State.PROCESSING,
            to_state=State.EXECUTING,
            condition=lambda ctx: ctx.get('ready_to_execute', False),
            action=lambda ctx: print("  Action: Beginning execution"),
            name="ready_to_execute"
        ))
        
        # EXECUTING -> WAITING if needs external resource
        self.add_transition(Transition(
            from_state=State.EXECUTING,
            to_state=State.WAITING,
            condition=lambda ctx: ctx.get('needs_resource', False),
            action=lambda ctx: print("  Action: Requesting external resource"),
            name="needs_resource"
        ))
        
        # WAITING -> EXECUTING when resource available
        self.add_transition(Transition(
            from_state=State.WAITING,
            to_state=State.EXECUTING,
            condition=lambda ctx: ctx.get('resource_available', False),
            action=lambda ctx: print("  Action: Resuming execution"),
            name="resource_available"
        ))
        
        # EXECUTING -> COMPLETED when done
        self.add_transition(Transition(
            from_state=State.EXECUTING,
            to_state=State.COMPLETED,
            condition=lambda ctx: ctx.get('execution_complete', False),
            action=lambda ctx: print("  Action: Finalizing results"),
            name="execution_complete"
        ))
        
        # Any state -> ERROR on error
        for state in State:
            if state != State.ERROR:
                self.add_transition(Transition(
                    from_state=state,
                    to_state=State.ERROR,
                    condition=lambda ctx: ctx.get('error_occurred', False),
                    action=lambda ctx: print(f"  Action: Handling error - {ctx.get('error_message')}"),
                    name="error_occurred"
                ))
        
        # Register state callbacks
        self.on_enter_state(State.LISTENING, lambda ctx: print("  → Entered LISTENING state"))
        self.on_enter_state(State.PROCESSING, lambda ctx: print("  → Entered PROCESSING state"))
        self.on_enter_state(State.EXECUTING, lambda ctx: print("  → Entered EXECUTING state"))
        self.on_enter_state(State.COMPLETED, lambda ctx: print("  → Entered COMPLETED state"))
        
        self.on_exit_state(State.IDLE, lambda ctx: print("  ← Exiting IDLE state"))
    
    def process_task(self, task: str):
        """Process a task through the state machine"""
        print(f"\n{'='*70}")
        print(f"PROCESSING TASK: {task}")
        print(f"{'='*70}")
        
        # Simulate task processing with events
        events = [
            {'task_received': True, 'task_name': task},
            {'input_complete': True},
            {'ready_to_execute': True},
            {'needs_resource': True},
            {'resource_available': True},
            {'execution_complete': True}
        ]
        
        for i, event in enumerate(events):
            print(f"\n--- Event {i + 1}: {list(event.keys())[0]} ---")
            self.update(event)
            
            # Clear event flags for next iteration
            self.context = {k: v for k, v in self.context.items() 
                           if k in ['task_name']}
            
            import time
            time.sleep(0.2)


# Usage
if __name__ == "__main__":
    # Create task processing agent
    agent = TaskProcessingAgent()
    
    # Process a task
    agent.process_task("Analyze customer data")
    
    # Print final summary
    print("\n" + "="*80)
    
    # Example with error handling
    print("\n\n" + "="*80)
    print("EXAMPLE WITH ERROR")
    print("="*80)
    
    agent2 = TaskProcessingAgent()
    agent2.update({'task_received': True})
    agent2.update({'input_complete': True})
    agent2.update({'error_occurred': True, 'error_message': 'Network timeout'})
```

```python patterns/18_monitoring_observability.py
"""
Monitoring & Observability Pattern
Comprehensive tracking of agent behavior
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time
import json

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class LogEntry:
    timestamp: datetime
    level: LogLevel
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None

@dataclass
class Metric:
    name: str
    type: MetricType
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class Span:
    """Distributed tracing span"""
    span_id: str
    trace_id: str
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[LogEntry] = field(default_factory=list)
    parent_span_id: Optional[str] = None

class MetricsCollector:
    """Collects and aggregates metrics"""
    
    def __init__(self):
        self.metrics: List[Metric] = []
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
    
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric"""
        self.counters[name] = self.counters.get(name, 0) + value
        
        metric = Metric(
            name=name,
            type=MetricType.COUNTER,
            value=self.counters[name],
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.metrics.append(metric)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric"""
        self.gauges[name] = value
        
        metric = Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.metrics.append(metric)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value"""
        metric = Metric(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        self.metrics.append(metric)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            "total_metrics": len(self.metrics),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "recent_metrics": self.metrics[-10:]
        }

class Logger:
    """Structured logging system"""
    
    def __init__(self, name: str):
        self.name = name
        self.logs: List[LogEntry] = []
        self.current_trace_id: Optional[str] = None
    
    def set_trace_id(self, trace_id: str):
        """Set current trace ID for correlation"""
        self.current_trace_id = trace_id
    
    def log(self, level: LogLevel, message: str, context: Dict[str, Any] = None):
        """Log a message"""
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            context=context or {},
            trace_id=self.current_trace_id
        )
        self.logs.append(entry)
        
        # Print to console
        timestamp_str = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        trace_str = f"[{entry.trace_id}]" if entry.trace_id else ""
        context_str = f" {json.dumps(entry.context)}" if entry.context else ""
        
        print(f"{timestamp_str} [{self.name}] {trace_str} {level.value.upper()}: {message}{context_str}")
    
    def debug(self, message: str, context: Dict[str, Any] = None):
        self.log(LogLevel.DEBUG, message, context)
    
    def info(self, message: str, context: Dict[str, Any] = None):
        self.log(LogLevel.INFO, message, context)
    
    def warning(self, message: str, context: Dict[str, Any] = None):
        self.log(LogLevel.WARNING, message, context)
    
    def error(self, message: str, context: Dict[str, Any] = None):
        self.log(LogLevel.ERROR, message, context)
    
    def critical(self, message: str, context: Dict[str, Any] = None):
        self.log(LogLevel.CRITICAL, message, context)
    
    def get_logs(self, level: Optional[LogLevel] = None, limit: int = None) -> List[LogEntry]:
        """Get filtered logs"""
        logs = self.logs
        
        if level:
            logs = [log for log in logs if log.level == level]
        
        if limit:
            logs = logs[-limit:]
        
        return logs

class Tracer:
    """Distributed tracing system"""
    
    def __init__(self):
        self.spans: List[Span] = []
        self.active_spans: Dict[str, Span] = {}
    
    def start_span(self, operation_name: str, trace_id: str = None, 
                   parent_span_id: str = None, tags: Dict[str, Any] = None) -> Span:
        """Start a new span"""
        import uuid
        
        span_id = str(uuid.uuid4())
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            tags=tags or {},
            parent_span_id=parent_span_id
        )
        
        self.active_spans[span_id] = span
        return span
    
    def finish_span(self, span: Span, tags: Dict[str, Any] = None):
        """Finish a span"""
        span.end_time = datetime.now()
        span.duration_ms = (span.end_time - span.start_time).total_seconds() * 1000
        
        if tags:
            span.tags.update(tags)
        
        self.spans.append(span)
        
        if span.span_id in self.active_spans:
            del self.active_spans[span.span_id]
    
    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace"""
        return [span for span in self.spans if span.trace_id == trace_id]
    
    def print_trace(self, trace_id: str):
        """Print trace tree"""
        spans = self.get_trace(trace_id)
        
        print(f"\n{'='*70}")
        print(f"TRACE: {trace_id}")
        print(f"{'='*70}")
        
        # Build tree
        root_spans = [s for s in spans if s.parent_span_id is None]
        
        def print_span(span: Span, indent: int = 0):
            prefix = "  " * indent
            duration = f"{span.duration_ms:.2f}ms" if span.duration_ms else "active"
            print(f"{prefix}├─ {span.operation_name} ({duration})")
            
            if span.tags:
                for key, value in span.tags.items():
                    print(f"{prefix}   {key}: {value}")
            
            # Print children
            children = [s for s in spans if s.parent_span_id == span.span_id]
            for child in children:
                print_span(child, indent + 1)
        
        for root in root_spans:
            print_span(root)

class ObservableAgent:
    """Agent with comprehensive observability"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = Logger(agent_id)
        self.metrics = MetricsCollector()
        self.tracer = Tracer()
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
    
    def process_request(self, request: str) -> Dict[str, Any]:
        """Process a request with full observability"""
        # Start trace
        span = self.tracer.start_span(
            operation_name="process_request",
            tags={"agent_id": self.agent_id, "request": request}
        )
        
        self.logger.set_trace_id(span.trace_id)
        self.logger.info("Processing request", {"request": request})
        
        # Increment request counter
        self.metrics.increment_counter("requests_total", tags={"agent": self.agent_id})
        
        start_time = time.time()
        
        try:
            # Simulate request processing
            result = self._execute_request(request, span)
            
            # Record success
            self.metrics.increment_counter("requests_success", tags={"agent": self.agent_id})
            self.logger.info("Request completed successfully", {"result": result})
            
            success = True
            
        except Exception as e:
            # Record error
            self.error_count += 1
            self.metrics.increment_counter("requests_error", tags={"agent": self.agent_id})
            self.logger.error(f"Request failed: {str(e)}", {"error": str(e)})
            
            result = {"error": str(e)}
            success = False
            
            span.tags["error"] = True
            span.tags["error_message"] = str(e)
        
        finally:
            # Record latency
            latency = (time.time() - start_time) * 1000  # ms
            self.metrics.record_histogram("request_latency_ms", latency, 
                                         tags={"agent": self.agent_id})
            
            # Update gauges
            self.request_count += 1
            self.total_latency += latency
            avg_latency = self.total_latency / self.request_count
            
            self.metrics.set_gauge("requests_in_flight", 0, tags={"agent": self.agent_id})
            self.metrics.set_gauge("avg_latency_ms", avg_latency, tags={"agent": self.agent_id})
            self.metrics.set_gauge("error_rate", self.error_count / self.request_count,
                                 tags={"agent": self.agent_id})
            
            # Finish span
            self.tracer.finish_span(span, tags={"success": success, "latency_ms": latency})
        
        return result
    
    def _execute_request(self, request: str, parent_span: Span) -> Dict[str, Any]:
        """Execute request with nested spans"""
        
        # Step 1: Parse request
        parse_span = self.tracer.start_span(
            operation_name="parse_request",
            trace_id=parent_span.trace_id,
            parent_span_id=parent_span.span_id
        )
        
        self.logger.debug("Parsing request")
        time.sleep(0.1)  # Simulate work
        parsed = {"task": request, "priority": "normal"}
        
        self.tracer.finish_span(parse_span, tags={"parsed_items": len(parsed)})
        
        # Step 2: Validate
        validate_span = self.tracer.start_span(
            operation_name="validate_request",
            trace_id=parent_span.trace_id,
            parent_span_id=parent_span.span_id
        )
        
        self.logger.debug("Validating request")
        time.sleep(0.05)
        
        self.tracer.finish_span(validate_span, tags={"valid": True})
        
        # Step 3: Execute
        execute_span = self.tracer.start_span(
            operation_name="execute_task",
            trace_id=parent_span.trace_id,
            parent_span_id=parent_span.span_id
        )
        
        self.logger.info("Executing task")
        time.sleep(0.15)
        
        result = {"status": "completed", "output": f"Processed: {request}"}
        
        self.tracer.finish_span(execute_span, tags={"output_size": len(str(result))})
        
        return result
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        
        health = "healthy"
        if error_rate > 0.1:
            health = "degraded"
        if error_rate > 0.5:
            health = "unhealthy"
        
        return {
            "status": health,
            "requests_total": self.request_count,
            "errors_total": self.error_count,
            "error_rate": error_rate,
            "avg_latency_ms": avg_latency
        }
    
    def print_observability_report(self):
        """Print comprehensive observability report"""
        print(f"\n{'='*70}")
        print(f"OBSERVABILITY REPORT: {self.agent_id}")
        print(f"{'='*70}")
        
        # Health status
        health = self.get_health_status()
        print(f"\nHealth Status: {health['status'].upper()}")
        print(f"  Requests: {health['requests_total']}")
        print(f"  Errors: {health['errors_total']}")
        print(f"  Error Rate: {health['error_rate']:.1%}")
        print(f"  Avg Latency: {health['avg_latency_ms']:.2f}ms")
        
        # Metrics summary
        metrics_summary = self.metrics.get_summary()
        print(f"\nMetrics Summary:")
        print(f"  Total Metrics Collected: {metrics_summary['total_metrics']}")
        
        if metrics_summary['counters']:
            print(f"\n  Counters:")
            for name, value in metrics_summary['counters'].items():
                print(f"    {name}: {value}")
        
        if metrics_summary['gauges']:
            print(f"\n  Gauges:")
            for name, value in metrics_summary['gauges'].items():
                print(f"    {name}: {value:.2f}")
        
        # Log summary
        error_logs = self.logger.get_logs(LogLevel.ERROR)
        warning_logs = self.logger.get_logs(LogLevel.WARNING)
        
        print(f"\nLog Summary:")
        print(f"  Total Logs: {len(self.logger.logs)}")
        print(f"  Errors: {len(error_logs)}")
        print(f"  Warnings: {len(warning_logs)}")
        
        if error_logs:
            print(f"\n  Recent Errors:")
            for log in error_logs[-3:]:
                print(f"    {log.timestamp}: {log.message}")
        
        # Trace summary
        print(f"\nTrace Summary:")
        print(f"  Total Spans: {len(self.tracer.spans)}")
        print(f"  Active Spans: {len(self.tracer.active_spans)}")


# Usage
if __name__ == "__main__":
    print("="*80)
    print("MONITORING & OBSERVABILITY DEMONSTRATION")
    print("="*80)
    
    # Create observable agent
    agent = ObservableAgent("agent-001")
    
    # Process multiple requests
    requests = [
        "Analyze customer data",
        "Generate report",
        "Send notifications",
        "Update database"
    ]
    
    print("\nProcessing requests...\n")
    
    for request in requests:
        result = agent.process_request(request)
        time.sleep(0.2)
    
    # Get a trace
    if agent.tracer.spans:
        first_trace_id = agent.tracer.spans[0].trace_id
        agent.tracer.print_trace(first_trace_id)
    
    # Print comprehensive report
    agent.print_observability_report()
```

```python patterns/19_caching_patterns.py
"""
Caching Patterns
Different levels of caching for performance optimization
"""

from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import time

@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int
    ttl_seconds: Optional[int]
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl_seconds is None:
            return False
        
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

class LRUCache:
    """Least Recently Used cache"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: list = []  # Track access order
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if entry.is_expired():
            del self.cache[key]
            self.access_order.remove(key)
            return None
        
        # Update access info
        entry.accessed_at = datetime.now()
        entry.access_count += 1
        
        # Move to end (most recently used)
        self.access_order.remove(key)
        self.access_order.append(key)
        
        return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Put value in cache"""
        # Remove if already exists
        if key in self.cache:
            self.access_order.remove(key)
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
        
        # Add new entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            ttl_seconds=ttl_seconds
        )
        
        self.cache[key] = entry
        self.access_order.append(key)
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_accesses": total_accesses,
            "utilization": len(self.cache) / self.max_size
        }

class SemanticCache:
    """Cache based on semantic similarity of queries"""
    
    def __init__(self, similarity_threshold: float = 0.9):
        self.cache: Dict[str, CacheEntry] = {}
        self.similarity_threshold = similarity_threshold
    
    def _compute_embedding(self, text: str) -> list:
        """Compute simple embedding (in reality, use proper embedding model)"""
        # Simple word-based embedding for demonstration
        words = text.lower().split()
        embedding = [0] * 100
        
        for word in words:
            idx = hash(word) % 100
            embedding[idx] += 1
        
        # Normalize
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding
    
    def _cosine_similarity(self, emb1: list, emb2: list) -> float:
        """Calculate cosine similarity"""
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        return dot_product
    
    def get(self, query: str) -> Optional[Any]:
        """Get cached result for semantically similar query"""
        query_embedding = self._compute_embedding(query)
        
        best_match = None
        best_similarity = 0
        
        for key, entry in self.cache.items():
            if entry.is_expired():
                continue
            
            cached_embedding = self._compute_embedding(key)
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
        if best_similarity >= self.similarity_threshold:
            print(f"  Cache hit (similarity: {best_similarity:.2f})")
            best_match.access_count += 1
            return best_match.value
        
        print(f"  Cache miss (best similarity: {best_similarity:.2f})")
        return None
    
    def put(self, query: str, value: Any, ttl_seconds: Optional[int] = 3600):
        """Cache result for query"""
        entry = CacheEntry(
            key=query,
            value=value,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            ttl_seconds=ttl_seconds
        )
        
        self.cache[query] = entry

class PromptCache:
    """Cache for LLM prompts and responses"""
    
    def __init__(self, max_size: int = 50):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
    
    def _hash_prompt(self, prompt: str, model: str, temperature: float) -> str:
        """Create hash of prompt parameters"""
        key_data = f"{prompt}:{model}:{temperature}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, prompt: str, model: str = "default", temperature: float = 0.0) -> Optional[str]:
        """Get cached response"""
        key = self._hash_prompt(prompt, model, temperature)
        
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                print(f"  Prompt cache hit")
                entry.access_count += 1
                return entry.value
            else:
                del self.cache[key]
        
        print(f"  Prompt cache miss")
        return None
    
    def put(self, prompt: str, response: str, model: str = "default", 
            temperature: float = 0.0, ttl_seconds: int = 3600):
        """Cache prompt response"""
        key = self._hash_prompt(prompt, model, temperature)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].created_at)
            del self.cache[oldest_key]
        
        entry = CacheEntry(
            key=key,
            value=response,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            access_count=0,
            ttl_seconds=ttl_seconds
        )
        
        self.cache[key] = entry

class CachedAgent:
    """Agent with multi-level caching"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.lru_cache = LRUCache(max_size=100)
        self.semantic_cache = SemanticCache(similarity_threshold=0.85)
        self.prompt_cache = PromptCache(max_size=50)
        
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_latency_saved_ms = 0
    
    def query(self, question: str) -> str:
        """Query with caching"""
        print(f"\n{'='*60}")
        print(f"Query: {question}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Try semantic cache first
        print("\nChecking semantic cache...")
        cached_response = self.semantic_cache.get(question)
        
        if cached_response:
            self.cache_hits += 1
            latency_saved = 2000  # Assume 2s saved
            self.total_latency_saved_ms += latency_saved
            
            print(f"✓ Returning cached response")
            print(f"  Latency saved: {latency_saved}ms")
            return cached_response
        
        # Cache miss - generate response
        self.cache_misses += 1
        print("✗ No cache hit, generating response...")
        
        # Simulate LLM call
        time.sleep(0.5)  # Simulate latency
        response = f"Generated response for: {question}"
        
        # Cache the response
        self.semantic_cache.put(question, response)
        
        latency = (time.time() - start_time) * 1000
        print(f"  Generated in {latency:.0f}ms")
        
        return response
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_latency_saved_ms": self.total_latency_saved_ms,
            "lru_cache_stats": self.lru_cache.get_stats()
        }
    
    def print_cache_report(self):
        """Print cache performance report"""
        stats = self.get_cache_stats()
        
        print(f"\n{'='*70}")
        print(f"CACHE PERFORMANCE REPORT")
        print(f"{'='*70}")
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Cache Hits: {stats['cache_hits']}")
        print(f"Cache Misses: {stats['cache_misses']}")
        print(f"Hit Rate: {stats['hit_rate']:.1%}")
        print(f"Total Latency Saved: {stats['total_latency_saved_ms']:.0f}ms")
        
        if stats['total_requests'] > 0:
            avg_saved = stats['total_latency_saved_ms'] / stats['total_requests']
            print(f"Avg Latency Saved per Request: {avg_saved:.0f}ms")


# Usage
if __name__ == "__main__":
    print("="*80)
    print("CACHING PATTERNS DEMONSTRATION")
    print("="*80)
    
    agent = CachedAgent("cached-agent-001")
    
    # Test queries (some similar, some different)
    queries = [
        "What is machine learning?",
        "Explain machine learning",  # Similar to first
        "What are neural networks?",
        "Tell me about deep learning",
        "What is machine learning?",  # Exact repeat
        "How does machine learning work?",  # Similar to first
    ]
    
    for query in queries:
        response = agent.query(query)
        time.sleep(0.1)
    
    # Print cache report
    agent.print_cache_report()
```

I've now created 19 comprehensive pattern implementations covering:

1. ReAct Pattern
2. Chain-of-Thought
3. Tree-of-Thoughts
4. Plan-and-Execute
5. Reflexion
6. RAG (Retrieval-Augmented Generation)
7. Multi-Agent Debate
8. Human-in-the-Loop
9. Memory Management
10. Self-Consistency
11. Function Calling
12. Code Execution
13. Workflow Orchestration
14. Guardrails
15. Ensemble Agents
16. Swarm Intelligence
17. State Machine Agent
18. Monitoring & Observability
19. Caching Patterns

Would you like me to continue with more patterns? I can implement:
- A/B Testing patterns
- Circuit Breaker patterns
- Async/Streaming agents
- Prompt optimization
- Tool routing
- And many more from the original list

Let me know which patterns you'd like to see next!






I'll continue creating Python implementations for the remaining patterns, picking up where we left off.

```python patterns/20_circuit_breaker.py
"""
Circuit Breaker Pattern
Stops agent when error rate exceeds threshold
"""

from typing import Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import time
import random

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout_seconds: int = 60   # Time before attempting recovery
    
@dataclass
class CircuitStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    last_failure_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self.state_changed_at = datetime.now()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                print(f"  [{self.name}] Circuit transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.state_changed_at = datetime.now()
            else:
                self.stats.rejected_requests += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Rejecting request."
                )
        
        # Attempt to execute
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.stats.last_failure_time is None:
            return True
        
        elapsed = datetime.now() - self.stats.last_failure_time
        return elapsed.total_seconds() >= self.config.timeout_seconds
    
    def _on_success(self):
        """Handle successful request"""
        self.stats.total_requests += 1
        self.stats.successful_requests += 1
        self.stats.consecutive_failures = 0
        self.stats.consecutive_successes += 1
        
        print(f"  [{self.name}] ✓ Request succeeded (consecutive: {self.stats.consecutive_successes})")
        
        if self.state == CircuitState.HALF_OPEN:
            if self.stats.consecutive_successes >= self.config.success_threshold:
                print(f"  [{self.name}] Circuit closing (service recovered)")
                self.state = CircuitState.CLOSED
                self.state_changed_at = datetime.now()
                self.stats.consecutive_successes = 0
    
    def _on_failure(self):
        """Handle failed request"""
        self.stats.total_requests += 1
        self.stats.failed_requests += 1
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.last_failure_time = datetime.now()
        
        print(f"  [{self.name}] ✗ Request failed (consecutive: {self.stats.consecutive_failures})")
        
        # Check if we should open the circuit
        if self.stats.consecutive_failures >= self.config.failure_threshold:
            if self.state != CircuitState.OPEN:
                print(f"  [{self.name}] !!! Circuit opening (failure threshold reached)")
                self.state = CircuitState.OPEN
                self.state_changed_at = datetime.now()
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics"""
        total = self.stats.total_requests
        success_rate = (self.stats.successful_requests / total * 100) if total > 0 else 0
        
        return {
            'name': self.name,
            'state': self.state.value,
            'total_requests': total,
            'successful': self.stats.successful_requests,
            'failed': self.stats.failed_requests,
            'rejected': self.stats.rejected_requests,
            'success_rate': success_rate,
            'consecutive_failures': self.stats.consecutive_failures,
            'time_in_state': (datetime.now() - self.state_changed_at).total_seconds()
        }
    
    def reset(self):
        """Manually reset circuit breaker"""
        print(f"  [{self.name}] Circuit manually reset")
        self.state = CircuitState.CLOSED
        self.stats.consecutive_failures = 0
        self.stats.consecutive_successes = 0
        self.state_changed_at = datetime.now()

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class ProtectedAgent:
    """Agent with circuit breaker protection"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Create circuit breakers for different operations
        self.api_circuit = CircuitBreaker(
            "API_Circuit",
            CircuitBreakerConfig(failure_threshold=3, timeout_seconds=5)
        )
        
        self.db_circuit = CircuitBreaker(
            "DB_Circuit",
            CircuitBreakerConfig(failure_threshold=5, timeout_seconds=10)
        )
    
    def call_external_api(self, endpoint: str) -> dict:
        """Call external API with circuit breaker protection"""
        print(f"\nCalling API: {endpoint}")
        
        def api_call():
            # Simulate API call that might fail
            time.sleep(0.1)
            
            # Simulate occasional failures
            if random.random() < 0.3:  # 30% failure rate
                raise Exception("API connection timeout")
            
            return {"status": "success", "data": "API response"}
        
        try:
            result = self.api_circuit.call(api_call)
            return result
        except CircuitBreakerOpenError as e:
            print(f"  Circuit breaker prevented call: {e}")
            return {"status": "error", "message": "Circuit breaker open"}
        except Exception as e:
            print(f"  API call failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def query_database(self, query: str) -> dict:
        """Query database with circuit breaker protection"""
        print(f"\nQuerying DB: {query}")
        
        def db_query():
            time.sleep(0.05)
            
            # Simulate occasional failures
            if random.random() < 0.2:  # 20% failure rate
                raise Exception("Database connection lost")
            
            return {"status": "success", "rows": 10}
        
        try:
            result = self.db_circuit.call(db_query)
            return result
        except CircuitBreakerOpenError as e:
            print(f"  Circuit breaker prevented call: {e}")
            return {"status": "error", "message": "Circuit breaker open"}
        except Exception as e:
            print(f"  DB query failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def print_circuit_status(self):
        """Print status of all circuit breakers"""
        print(f"\n{'='*70}")
        print(f"CIRCUIT BREAKER STATUS")
        print(f"{'='*70}")
        
        for circuit in [self.api_circuit, self.db_circuit]:
            stats = circuit.get_stats()
            
            print(f"\n{stats['name']}:")
            print(f"  State: {stats['state'].upper()}")
            print(f"  Total Requests: {stats['total_requests']}")
            print(f"  Successful: {stats['successful']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Rejected: {stats['rejected']}")
            print(f"  Success Rate: {stats['success_rate']:.1f}%")
            print(f"  Consecutive Failures: {stats['consecutive_failures']}")
            print(f"  Time in Current State: {stats['time_in_state']:.1f}s")


# Usage
if __name__ == "__main__":
    print("="*80)
    print("CIRCUIT BREAKER PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = ProtectedAgent("agent-001")
    
    # Simulate multiple API calls
    print("\n" + "="*80)
    print("Testing API Circuit Breaker")
    print("="*80)
    
    for i in range(15):
        print(f"\n--- Request {i+1} ---")
        agent.call_external_api(f"/api/endpoint/{i}")
        time.sleep(0.2)
        
        # Show status after some requests
        if i == 7:
            agent.print_circuit_status()
    
    # Wait for circuit to potentially reset
    print("\n\nWaiting for circuit breaker timeout...")
    time.sleep(6)
    
    print("\n" + "="*80)
    print("Retrying after timeout")
    print("="*80)
    
    for i in range(5):
        print(f"\n--- Retry Request {i+1} ---")
        agent.call_external_api(f"/api/endpoint/retry-{i}")
        time.sleep(0.2)
    
    # Final status
    agent.print_circuit_status()
```

```python patterns/21_ab_testing.py
"""
A/B Testing Pattern
Compares different agent configurations
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random
import statistics

class Variant(Enum):
    CONTROL = "control"
    TREATMENT_A = "treatment_a"
    TREATMENT_B = "treatment_b"

@dataclass
class ExperimentConfig:
    name: str
    variants: Dict[str, float]  # variant_name -> traffic_percentage
    metrics: List[str]
    
    def __post_init__(self):
        # Validate traffic percentages sum to ~1.0
        total = sum(self.variants.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Traffic percentages must sum to 1.0, got {total}")

@dataclass
class VariantResult:
    variant_name: str
    request_count: int = 0
    success_count: int = 0
    total_latency_ms: float = 0.0
    metric_values: Dict[str, List[float]] = field(default_factory=dict)
    
    def add_result(self, success: bool, latency_ms: float, metrics: Dict[str, float] = None):
        """Record a result for this variant"""
        self.request_count += 1
        if success:
            self.success_count += 1
        self.total_latency_ms += latency_ms
        
        if metrics:
            for metric_name, value in metrics.items():
                if metric_name not in self.metric_values:
                    self.metric_values[metric_name] = []
                self.metric_values[metric_name].append(value)
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate statistics for this variant"""
        success_rate = (self.success_count / self.request_count 
                       if self.request_count > 0 else 0)
        avg_latency = (self.total_latency_ms / self.request_count 
                      if self.request_count > 0 else 0)
        
        metric_stats = {}
        for metric_name, values in self.metric_values.items():
            if values:
                metric_stats[metric_name] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values)
                }
        
        return {
            'variant': self.variant_name,
            'requests': self.request_count,
            'success_rate': success_rate,
            'avg_latency_ms': avg_latency,
            'metrics': metric_stats
        }

class ABTestingFramework:
    """Framework for running A/B tests on agents"""
    
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.results: Dict[str, VariantResult] = {
            variant: VariantResult(variant)
            for variant in experiment_config.variants.keys()
        }
        self.start_time = datetime.now()
    
    def assign_variant(self, user_id: str = None) -> str:
        """Assign a variant to a user/request"""
        # Use consistent hashing if user_id provided
        if user_id:
            hash_val = hash(user_id) % 100
            cumulative = 0
            for variant, percentage in self.config.variants.items():
                cumulative += percentage * 100
                if hash_val < cumulative:
                    return variant
        
        # Random assignment
        r = random.random()
        cumulative = 0
        for variant, percentage in self.config.variants.items():
            cumulative += percentage
            if r < cumulative:
                return variant
        
        # Fallback
        return list(self.config.variants.keys())[0]
    
    def record_result(self, variant: str, success: bool, latency_ms: float,
                     metrics: Dict[str, float] = None):
        """Record a result for a variant"""
        if variant in self.results:
            self.results[variant].add_result(success, latency_ms, metrics)
    
    def get_comparison(self) -> Dict[str, Any]:
        """Get comparison of all variants"""
        stats = {
            variant: result.get_stats()
            for variant, result in self.results.items()
        }
        
        # Determine winner based on success rate
        winner = max(stats.items(), 
                    key=lambda x: x[1]['success_rate'])
        
        return {
            'experiment': self.config.name,
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'variants': stats,
            'winner': {
                'variant': winner[0],
                'success_rate': winner[1]['success_rate']
            }
        }
    
    def print_results(self):
        """Print detailed results"""
        comparison = self.get_comparison()
        
        print(f"\n{'='*70}")
        print(f"A/B TEST RESULTS: {comparison['experiment']}")
        print(f"{'='*70}")
        print(f"Duration: {comparison['duration_seconds']:.1f}s")
        
        print(f"\nVariant Performance:")
        for variant_name, stats in comparison['variants'].items():
            print(f"\n{variant_name}:")
            print(f"  Requests: {stats['requests']}")
            print(f"  Success Rate: {stats['success_rate']:.1%}")
            print(f"  Avg Latency: {stats['avg_latency_ms']:.2f}ms")
            
            if stats['metrics']:
                print(f"  Metrics:")
                for metric_name, metric_stats in stats['metrics'].items():
                    print(f"    {metric_name}:")
                    print(f"      Mean: {metric_stats['mean']:.2f}")
                    print(f"      Median: {metric_stats['median']:.2f}")
                    print(f"      StdDev: {metric_stats['stdev']:.2f}")
        
        print(f"\n{'='*70}")
        print(f"WINNER: {comparison['winner']['variant']}")
        print(f"Success Rate: {comparison['winner']['success_rate']:.1%}")
        print(f"{'='*70}")

class ABTestedAgent:
    """Agent with A/B testing capabilities"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Setup A/B test
        self.experiment = ABTestingFramework(
            ExperimentConfig(
                name="Response Strategy Test",
                variants={
                    "control": 0.34,      # Original strategy
                    "treatment_a": 0.33,  # New strategy A
                    "treatment_b": 0.33   # New strategy B
                },
                metrics=["quality_score", "user_satisfaction"]
            )
        )
    
    def process_request(self, request: str, user_id: str = None) -> Dict[str, Any]:
        """Process request with A/B testing"""
        import time
        
        # Assign variant
        variant = self.experiment.assign_variant(user_id)
        
        print(f"\nProcessing with variant: {variant}")
        
        # Measure latency
        start_time = time.time()
        
        # Execute variant-specific logic
        if variant == "control":
            response = self._control_strategy(request)
        elif variant == "treatment_a":
            response = self._treatment_a_strategy(request)
        else:
            response = self._treatment_b_strategy(request)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Simulate metrics
        quality_score = response.get('quality', 0)
        user_satisfaction = response.get('satisfaction', 0)
        success = response.get('success', False)
        
        # Record result
        self.experiment.record_result(
            variant,
            success,
            latency_ms,
            metrics={
                'quality_score': quality_score,
                'user_satisfaction': user_satisfaction
            }
        )
        
        print(f"  Latency: {latency_ms:.2f}ms")
        print(f"  Quality: {quality_score:.2f}")
        print(f"  Satisfaction: {user_satisfaction:.2f}")
        
        return response
    
    def _control_strategy(self, request: str) -> Dict[str, Any]:
        """Original/control strategy"""
        import time
        time.sleep(0.1)  # Simulate work
        
        return {
            'success': random.random() > 0.15,  # 85% success rate
            'response': f"Control response for: {request}",
            'quality': random.uniform(0.7, 0.85),
            'satisfaction': random.uniform(0.65, 0.80)
        }
    
    def _treatment_a_strategy(self, request: str) -> Dict[str, Any]:
        """Treatment A - Optimized for speed"""
        import time
        time.sleep(0.05)  # Faster
        
        return {
            'success': random.random() > 0.12,  # 88% success rate
            'response': f"Treatment A response for: {request}",
            'quality': random.uniform(0.75, 0.90),
            'satisfaction': random.uniform(0.70, 0.85)
        }
    
    def _treatment_b_strategy(self, request: str) -> Dict[str, Any]:
        """Treatment B - Optimized for quality"""
        import time
        time.sleep(0.15)  # Slower but higher quality
        
        return {
            'success': random.random() > 0.10,  # 90% success rate
            'response': f"Treatment B response for: {request}",
            'quality': random.uniform(0.80, 0.95),
            'satisfaction': random.uniform(0.75, 0.90)
        }


# Usage
if __name__ == "__main__":
    print("="*80)
    print("A/B TESTING PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = ABTestedAgent("ab-test-agent")
    
    # Simulate requests from different users
    requests = [
        "Analyze customer sentiment",
        "Generate summary report",
        "Recommend products",
        "Answer technical question",
        "Create visualization"
    ]
    
    # Run experiment
    print("\nRunning A/B test experiment...")
    print("="*80)
    
    num_iterations = 30
    for i in range(num_iterations):
        request = random.choice(requests)
        user_id = f"user_{i % 10}"  # 10 different users
        
        agent.process_request(request, user_id)
        
        # Small delay between requests
        import time
        time.sleep(0.05)
    
    # Print results
    agent.experiment.print_results()
    
    # Statistical significance check (simplified)
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    comparison = agent.experiment.get_comparison()
    variants = comparison['variants']
    
    # Compare control vs treatments
    control_sr = variants['control']['success_rate']
    
    for variant_name in ['treatment_a', 'treatment_b']:
        variant_sr = variants[variant_name]['success_rate']
        improvement = ((variant_sr - control_sr) / control_sr * 100) if control_sr > 0 else 0
        
        print(f"\n{variant_name} vs control:")
        print(f"  Success Rate Improvement: {improvement:+.1f}%")
        print(f"  Sample Size: {variants[variant_name]['requests']}")
```

```python patterns/22_async_agent.py
"""
Asynchronous Agent Pattern
Handles multiple tasks concurrently
"""

import asyncio
from typing import List, Dict, Any, Coroutine
from dataclasses import dataclass
from datetime import datetime
import time

@dataclass
class Task:
    task_id: str
    name: str
    priority: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class TaskResult:
    task_id: str
    success: bool
    result: Any
    duration_ms: float
    completed_at: datetime

class AsyncAgent:
    """Agent that processes tasks asynchronously"""
    
    def __init__(self, agent_id: str, max_concurrent: int = 5):
        self.agent_id = agent_id
        self.max_concurrent = max_concurrent
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.results: List[TaskResult] = []
        self.active_tasks: int = 0
    
    async def add_task(self, task: Task):
        """Add task to queue"""
        await self.task_queue.put(task)
        print(f"[{self.agent_id}] Task queued: {task.name} (priority: {task.priority})")
    
    async def process_task(self, task: Task) -> TaskResult:
        """Process a single task"""
        self.active_tasks += 1
        start_time = time.time()
        
        print(f"[{self.agent_id}] Processing: {task.name}")
        
        try:
            # Simulate async work
            result = await self._execute_task(task)
            success = True
            
        except Exception as e:
            result = {"error": str(e)}
            success = False
            print(f"[{self.agent_id}] Error in {task.name}: {e}")
        
        duration_ms = (time.time() - start_time) * 1000
        
        task_result = TaskResult(
            task_id=task.task_id,
            success=success,
            result=result,
            duration_ms=duration_ms,
            completed_at=datetime.now()
        )
        
        self.results.append(task_result)
        self.active_tasks -= 1
        
        print(f"[{self.agent_id}] Completed: {task.name} ({duration_ms:.0f}ms)")
        
        return task_result
    
    async def _execute_task(self, task: Task) -> Any:
        """Execute task logic"""
        # Simulate different task types with varying delays
        if "quick" in task.name.lower():
            await asyncio.sleep(0.5)
            return {"status": "completed", "type": "quick"}
        elif "slow" in task.name.lower():
            await asyncio.sleep(2.0)
            return {"status": "completed", "type": "slow"}
        else:
            await asyncio.sleep(1.0)
            return {"status": "completed", "type": "normal"}
    
    async def worker(self, worker_id: int):
        """Worker coroutine that processes tasks from queue"""
        while True:
            try:
                # Get task from queue
                task = await self.task_queue.get()
                
                # Process task
                await self.process_task(task)
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
    
    async def run(self, num_workers: int = None):
        """Run agent with multiple workers"""
        if num_workers is None:
            num_workers = self.max_concurrent
        
        print(f"\n[{self.agent_id}] Starting {num_workers} workers...")
        
        # Create worker tasks
        workers = [
            asyncio.create_task(self.worker(i))
            for i in range(num_workers)
        ]
        
        # Wait for queue to be empty
        await self.task_queue.join()
        
        # Cancel workers
        for worker in workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*workers, return_exceptions=True)
        
        print(f"[{self.agent_id}] All tasks completed")
    
    async def process_batch(self, tasks: List[Task]) -> List[TaskResult]:
        """Process multiple tasks concurrently"""
        print(f"\n[{self.agent_id}] Processing batch of {len(tasks)} tasks")
        
        # Process tasks concurrently with limit
        results = await asyncio.gather(
            *[self.process_task(task) for task in tasks],
            return_exceptions=True
        )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful
        
        if total > 0:
            avg_duration = sum(r.duration_ms for r in self.results) / total
            total_duration = sum(r.duration_ms for r in self.results)
        else:
            avg_duration = 0
            total_duration = 0
        
        return {
            'total_tasks': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'avg_duration_ms': avg_duration,
            'total_duration_ms': total_duration,
            'active_tasks': self.active_tasks
        }

class StreamingAsyncAgent:
    """Agent that streams results as they complete"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    
    async def stream_process(self, tasks: List[Task]):
        """Process tasks and stream results as they complete"""
        print(f"\n[{self.agent_id}] Streaming processing of {len(tasks)} tasks")
        
        # Create tasks
        pending = {
            asyncio.create_task(self._process_with_delay(task)): task
            for task in tasks
        }
        
        # Process as they complete
        while pending:
            # Wait for next completion
            done, pending_set = await asyncio.wait(
                pending.keys(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Update pending
            pending = {task: pending[task] for task in pending_set}
            
            # Yield completed results
            for completed_task in done:
                task = pending.pop(completed_task, None)
                try:
                    result = await completed_task
                    print(f"[{self.agent_id}] ✓ Streamed result for: {task.name if task else 'unknown'}")
                    yield result
                except Exception as e:
                    print(f"[{self.agent_id}] ✗ Error: {e}")
    
    async def _process_with_delay(self, task: Task) -> Dict[str, Any]:
        """Process task with simulated delay"""
        delay = 0.5 + (hash(task.task_id) % 20) / 10  # 0.5-2.5 seconds
        await asyncio.sleep(delay)
        
        return {
            'task_id': task.task_id,
            'task_name': task.name,
            'result': f"Processed {task.name}",
            'duration': delay
        }


# Usage
async def main():
    print("="*80)
    print("ASYNCHRONOUS AGENT PATTERN DEMONSTRATION")
    print("="*80)
    
    # Example 1: Queue-based async processing
    print("\n" + "="*80)
    print("Example 1: Queue-Based Processing")
    print("="*80)
    
    agent = AsyncAgent("async-agent-001", max_concurrent=3)
    
    # Add tasks to queue
    tasks = [
        Task("t1", "Quick task 1"),
        Task("t2", "Slow task 1"),
        Task("t3", "Normal task 1"),
        Task("t4", "Quick task 2"),
        Task("t5", "Slow task 2"),
        Task("t6", "Normal task 2"),
        Task("t7", "Quick task 3"),
        Task("t8", "Normal task 3"),
    ]
    
    # Add all tasks
    for task in tasks:
        await agent.add_task(task)
    
    # Start processing
    start_time = time.time()
    await agent.run(num_workers=3)
    total_time = time.time() - start_time
    
    # Print stats
    stats = agent.get_stats()
    print(f"\n{'='*70}")
    print("QUEUE PROCESSING STATISTICS")
    print(f"{'='*70}")
    print(f"Total Tasks: {stats['total_tasks']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Avg Task Duration: {stats['avg_duration_ms']:.0f}ms")
    print(f"Total Processing Time: {total_time:.2f}s")
    print(f"Efficiency: {stats['total_duration_ms']/1000/total_time:.1f}x")
    
```python patterns/22_async_agent.py
# ... continuing from Example 2

    # Example 2: Batch processing
    print("\n\n" + "="*80)
    print("Example 2: Batch Processing")
    print("="*80)
    
    agent2 = AsyncAgent("async-agent-002")
    
    batch_tasks = [
        Task(f"b{i}", f"Batch task {i}")
        for i in range(10)
    ]
    
    start_time = time.time()
    results = await agent2.process_batch(batch_tasks)
    batch_time = time.time() - start_time
    
    print(f"\nBatch processing completed in {batch_time:.2f}s")
    print(f"Processed {len(results)} tasks")
    
    # Example 3: Streaming results
    print("\n\n" + "="*80)
    print("Example 3: Streaming Processing")
    print("="*80)
    
    streaming_agent = StreamingAsyncAgent("streaming-agent-001")
    
    stream_tasks = [
        Task(f"s{i}", f"Stream task {i}")
        for i in range(6)
    ]
    
    print("\nProcessing tasks with streaming output:")
    async for result in streaming_agent.stream_process(stream_tasks):
        print(f"  Received: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

```python patterns/23_graph_of_thoughts.py
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
```

```python patterns/24_hierarchical_planning.py
"""
Hierarchical Planning Pattern
Breaks down goals into hierarchical sub-goals
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

class GoalStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

class GoalLevel(Enum):
    STRATEGIC = "strategic"      # High-level, long-term
    TACTICAL = "tactical"        # Mid-level, medium-term
    OPERATIONAL = "operational"  # Low-level, immediate

@dataclass
class Goal:
    """Hierarchical goal node"""
    id: str
    description: str
    level: GoalLevel
    status: GoalStatus = GoalStatus.PENDING
    parent_id: Optional[str] = None
    children: List['Goal'] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def add_child(self, child: 'Goal'):
        """Add sub-goal"""
        child.parent_id = self.id
        self.children.append(child)
    
    def is_ready(self, completed_goals: set) -> bool:
        """Check if all dependencies are met"""
        return all(dep_id in completed_goals for dep_id in self.dependencies)
    
    def get_progress(self) -> float:
        """Calculate completion progress"""
        if not self.children:
            return 1.0 if self.status == GoalStatus.COMPLETED else 0.0
        
        return sum(child.get_progress() for child in self.children) / len(self.children)

class HierarchicalPlanner:
    """Planner that creates hierarchical goal structures"""
    
    def __init__(self):
        self.goals: Dict[str, Goal] = {}
        self.root_goals: List[Goal] = []
        self.completed_goals: set = set()
        self.goal_counter = 0
    
    def create_goal(self, description: str, level: GoalLevel, 
                    parent: Optional[Goal] = None, 
                    dependencies: List[str] = None) -> Goal:
        """Create a new goal"""
        self.goal_counter += 1
        goal_id = f"G{self.goal_counter:03d}"
        
        goal = Goal(
            id=goal_id,
            description=description,
            level=level,
            dependencies=dependencies or []
        )
        
        self.goals[goal_id] = goal
        
        if parent:
            parent.add_child(goal)
        else:
            self.root_goals.append(goal)
        
        return goal
    
    def decompose_goal(self, goal: Goal, decomposition: List[Dict[str, Any]]) -> List[Goal]:
        """Decompose a goal into sub-goals"""
        sub_goals = []
        
        # Determine sub-goal level
        if goal.level == GoalLevel.STRATEGIC:
            sub_level = GoalLevel.TACTICAL
        elif goal.level == GoalLevel.TACTICAL:
            sub_level = GoalLevel.OPERATIONAL
        else:
            sub_level = GoalLevel.OPERATIONAL
        
        for sub_goal_info in decomposition:
            sub_goal = self.create_goal(
                description=sub_goal_info['description'],
                level=sub_level,
                parent=goal,
                dependencies=sub_goal_info.get('dependencies', [])
            )
            sub_goals.append(sub_goal)
        
        print(f"Decomposed '{goal.description}' into {len(sub_goals)} sub-goals")
        
        return sub_goals
    
    def get_executable_goals(self) -> List[Goal]:
        """Get goals that are ready to execute"""
        executable = []
        
        for goal in self.goals.values():
            if (goal.status == GoalStatus.PENDING and
                not goal.children and  # Leaf node
                goal.is_ready(self.completed_goals)):
                executable.append(goal)
        
        return executable
    
    def execute_goal(self, goal: Goal) -> bool:
        """Execute a goal"""
        print(f"\nExecuting: [{goal.level.value}] {goal.description}")
        
        goal.status = GoalStatus.IN_PROGRESS
        
        # Simulate execution
        import time
        import random
        time.sleep(0.1)
        
        # Simulate success/failure
        success = random.random() > 0.1  # 90% success rate
        
        if success:
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = datetime.now()
            self.completed_goals.add(goal.id)
            print(f"  ✓ Completed")
            
            # Check if parent can be completed
            if goal.parent_id:
                self._check_parent_completion(goal.parent_id)
            
            return True
        else:
            goal.status = GoalStatus.FAILED
            print(f"  ✗ Failed")
            return False
    
    def _check_parent_completion(self, parent_id: str):
        """Check if parent goal can be marked complete"""
        parent = self.goals[parent_id]
        
        if all(child.status == GoalStatus.COMPLETED for child in parent.children):
            parent.status = GoalStatus.COMPLETED
            parent.completed_at = datetime.now()
            self.completed_goals.add(parent_id)
            print(f"  ✓ Parent goal completed: {parent.description}")
            
            # Recursively check parent's parent
            if parent.parent_id:
                self._check_parent_completion(parent.parent_id)
    
    def visualize_hierarchy(self, goal: Goal = None, indent: int = 0):
        """Print goal hierarchy"""
        if goal is None:
            print(f"\n{'='*70}")
            print("GOAL HIERARCHY")
            print(f"{'='*70}")
            for root in self.root_goals:
                self.visualize_hierarchy(root)
            return
        
        prefix = "  " * indent
        status_symbol = {
            GoalStatus.PENDING: "○",
            GoalStatus.IN_PROGRESS: "◐",
            GoalStatus.COMPLETED: "●",
            GoalStatus.FAILED: "✗",
            GoalStatus.BLOCKED: "⊗"
        }[goal.status]
        
        progress = goal.get_progress() * 100
        
        print(f"{prefix}{status_symbol} [{goal.level.value[0].upper()}] {goal.description} ({progress:.0f}%)")
        
        for child in goal.children:
            self.visualize_hierarchy(child, indent + 1)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get planning statistics"""
        total = len(self.goals)
        by_status = {status: 0 for status in GoalStatus}
        by_level = {level: 0 for level in GoalLevel}
        
        for goal in self.goals.values():
            by_status[goal.status] += 1
            by_level[goal.level] += 1
        
        return {
            'total_goals': total,
            'completed': by_status[GoalStatus.COMPLETED],
            'in_progress': by_status[GoalStatus.IN_PROGRESS],
            'pending': by_status[GoalStatus.PENDING],
            'failed': by_status[GoalStatus.FAILED],
            'by_level': {level.value: count for level, count in by_level.items()},
            'overall_progress': len(self.completed_goals) / total if total > 0 else 0
        }

class HierarchicalPlanningAgent:
    """Agent that uses hierarchical planning"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.planner = HierarchicalPlanner()
    
    def plan_and_execute(self, strategic_goal: str) -> Dict[str, Any]:
        """Create hierarchical plan and execute"""
        print(f"\n{'='*70}")
        print(f"HIERARCHICAL PLANNING: {strategic_goal}")
        print(f"{'='*70}")
        
        # Create strategic goal
        root = self.planner.create_goal(
            description=strategic_goal,
            level=GoalLevel.STRATEGIC
        )
        
        # Decompose into tactical goals
        tactical_decomposition = self._decompose_strategic(strategic_goal)
        tactical_goals = self.planner.decompose_goal(root, tactical_decomposition)
        
        # Decompose tactical into operational
        for tactical in tactical_goals:
            operational_decomposition = self._decompose_tactical(tactical.description)
            self.planner.decompose_goal(tactical, operational_decomposition)
        
        # Visualize initial plan
        self.planner.visualize_hierarchy()
        
        # Execute plan
        print(f"\n{'='*70}")
        print("EXECUTION")
        print(f"{'='*70}")
        
        max_iterations = 50
        iteration = 0
        
        while iteration < max_iterations:
            executable = self.planner.get_executable_goals()
            
            if not executable:
                # Check if done
                if root.status == GoalStatus.COMPLETED:
                    print(f"\n✓ Strategic goal achieved!")
                    break
                else:
                    print(f"\n⚠ No executable goals remaining")
                    break
            
            # Execute one goal
            goal = executable[0]
            self.planner.execute_goal(goal)
            
            iteration += 1
        
        # Final visualization
        self.planner.visualize_hierarchy()
        
        # Statistics
        stats = self.planner.get_statistics()
        
        print(f"\n{'='*70}")
        print("STATISTICS")
        print(f"{'='*70}")
        print(f"Total Goals: {stats['total_goals']}")
        print(f"Completed: {stats['completed']}")
        print(f"Failed: {stats['failed']}")
        print(f"Overall Progress: {stats['overall_progress']:.1%}")
        print(f"\nGoals by Level:")
        for level, count in stats['by_level'].items():
            print(f"  {level}: {count}")
        
        return stats
    
    def _decompose_strategic(self, strategic_goal: str) -> List[Dict[str, Any]]:
        """Decompose strategic goal into tactical goals"""
        # Simulated decomposition based on goal
        if "launch product" in strategic_goal.lower():
            return [
                {'description': "Develop product features"},
                {'description': "Create marketing campaign", 'dependencies': []},
                {'description': "Set up distribution", 'dependencies': []},
                {'description': "Train support team", 'dependencies': ['G002']},  # Depends on development
            ]
        else:
            return [
                {'description': f"Phase 1 of {strategic_goal}"},
                {'description': f"Phase 2 of {strategic_goal}", 'dependencies': []},
                {'description': f"Phase 3 of {strategic_goal}", 'dependencies': []},
            ]
    
    def _decompose_tactical(self, tactical_goal: str) -> List[Dict[str, Any]]:
        """Decompose tactical goal into operational goals"""
        # Simulated decomposition
        if "develop" in tactical_goal.lower():
            return [
                {'description': "Design architecture"},
                {'description': "Implement core features", 'dependencies': []},
                {'description': "Write tests", 'dependencies': []},
                {'description': "Conduct code review", 'dependencies': []},
            ]
        elif "marketing" in tactical_goal.lower():
            return [
                {'description': "Define target audience"},
                {'description': "Create content", 'dependencies': []},
                {'description': "Launch campaigns", 'dependencies': []},
            ]
        else:
            return [
                {'description': f"Step 1: {tactical_goal}"},
                {'description': f"Step 2: {tactical_goal}", 'dependencies': []},
                {'description': f"Step 3: {tactical_goal}", 'dependencies': []},
            ]


# Usage
if __name__ == "__main__":
    print("="*80)
    print("HIERARCHICAL PLANNING PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = HierarchicalPlanningAgent("planner-001")
    
    # Plan and execute strategic goal
    result = agent.plan_and_execute(
        "Successfully launch new AI product to market"
    )
```

```python patterns/25_least_to_most.py
"""
Least-to-Most Prompting Pattern
Solves easier sub-problems first, building up to harder ones
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class DifficultyLevel(Enum):
    TRIVIAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5

@dataclass
class SubProblem:
    """A sub-problem in the least-to-most chain"""
    id: int
    description: str
    difficulty: DifficultyLevel
    solution: Any = None
    depends_on: List[int] = None
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []

class LeastToMostSolver:
    """Solver using least-to-most prompting"""
    
    def __init__(self):
        self.sub_problems: List[SubProblem] = []
        self.solutions: Dict[int, Any] = {}
        self.context: Dict[str, Any] = {}
    
    def decompose_problem(self, problem: str) -> List[SubProblem]:
        """Decompose problem into progressively harder sub-problems"""
        print(f"\n{'='*70}")
        print(f"DECOMPOSING PROBLEM")
        print(f"{'='*70}")
        print(f"Main Problem: {problem}\n")
        
        # Simulated decomposition (in reality, use LLM)
        if "calculate compound" in problem.lower():
            sub_problems = self._decompose_compound_interest(problem)
        elif "write essay" in problem.lower():
            sub_problems = self._decompose_essay(problem)
        elif "optimize" in problem.lower():
            sub_problems = self._decompose_optimization(problem)
        else:
            sub_problems = self._generic_decomposition(problem)
        
        # Sort by difficulty
        sub_problems.sort(key=lambda x: x.difficulty.value)
        
        self.sub_problems = sub_problems
        
        print("Sub-problems (ordered by difficulty):")
        for i, sp in enumerate(sub_problems, 1):
            deps = f" (depends on: {sp.depends_on})" if sp.depends_on else ""
            print(f"  {i}. [{sp.difficulty.name}] {sp.description}{deps}")
        
        return sub_problems
    
    def _decompose_compound_interest(self, problem: str) -> List[SubProblem]:
        """Decompose compound interest problem"""
        return [
            SubProblem(
                id=1,
                description="Understand the basic formula: A = P(1 + r)^t",
                difficulty=DifficultyLevel.TRIVIAL
            ),
            SubProblem(
                id=2,
                description="Identify the given values (P, r, t)",
                difficulty=DifficultyLevel.EASY,
                depends_on=[1]
            ),
            SubProblem(
                id=3,
                description="Calculate (1 + r)",
                difficulty=DifficultyLevel.EASY,
                depends_on=[2]
            ),
            SubProblem(
                id=4,
                description="Raise to power t",
                difficulty=DifficultyLevel.MEDIUM,
                depends_on=[3]
            ),
            SubProblem(
                id=5,
                description="Multiply by principal P",
                difficulty=DifficultyLevel.MEDIUM,
                depends_on=[4]
            ),
            SubProblem(
                id=6,
                description="Interpret the result in context",
                difficulty=DifficultyLevel.HARD,
                depends_on=[5]
            )
        ]
    
    ```python patterns/25_least_to_most.py
    def _decompose_essay(self, problem: str) -> List[SubProblem]:
        """Decompose essay writing problem"""
        return [
            SubProblem(
                id=1,
                description="Understand the essay topic and requirements",
                difficulty=DifficultyLevel.TRIVIAL
            ),
            SubProblem(
                id=2,
                description="Brainstorm main ideas",
                difficulty=DifficultyLevel.EASY,
                depends_on=[1]
            ),
            SubProblem(
                id=3,
                description="Create outline with thesis statement",
                difficulty=DifficultyLevel.MEDIUM,
                depends_on=[2]
            ),
            SubProblem(
                id=4,
                description="Write introduction paragraph",
                difficulty=DifficultyLevel.MEDIUM,
                depends_on=[3]
            ),
            SubProblem(
                id=5,
                description="Develop body paragraphs with evidence",
                difficulty=DifficultyLevel.HARD,
                depends_on=[4]
            ),
            SubProblem(
                id=6,
                description="Write conclusion and refine transitions",
                difficulty=DifficultyLevel.HARD,
                depends_on=[5]
            ),
            SubProblem(
                id=7,
                description="Edit for clarity, coherence, and style",
                difficulty=DifficultyLevel.EXPERT,
                depends_on=[6]
            )
        ]
    
    def _decompose_optimization(self, problem: str) -> List[SubProblem]:
        """Decompose optimization problem"""
        return [
            SubProblem(
                id=1,
                description="Define the objective function",
                difficulty=DifficultyLevel.EASY
            ),
            SubProblem(
                id=2,
                description="Identify constraints",
                difficulty=DifficultyLevel.EASY,
                depends_on=[1]
            ),
            SubProblem(
                id=3,
                description="Choose optimization method",
                difficulty=DifficultyLevel.MEDIUM,
                depends_on=[2]
            ),
            SubProblem(
                id=4,
                description="Set up initial conditions",
                difficulty=DifficultyLevel.MEDIUM,
                depends_on=[3]
            ),
            SubProblem(
                id=5,
                description="Run optimization algorithm",
                difficulty=DifficultyLevel.HARD,
                depends_on=[4]
            ),
            SubProblem(
                id=6,
                description="Validate and interpret results",
                difficulty=DifficultyLevel.EXPERT,
                depends_on=[5]
            )
        ]
    
    def _generic_decomposition(self, problem: str) -> List[SubProblem]:
        """Generic problem decomposition"""
        return [
            SubProblem(
                id=1,
                description=f"Understand basics of: {problem}",
                difficulty=DifficultyLevel.TRIVIAL
            ),
            SubProblem(
                id=2,
                description="Gather necessary information",
                difficulty=DifficultyLevel.EASY,
                depends_on=[1]
            ),
            SubProblem(
                id=3,
                description="Apply basic techniques",
                difficulty=DifficultyLevel.MEDIUM,
                depends_on=[2]
            ),
            SubProblem(
                id=4,
                description="Solve main problem",
                difficulty=DifficultyLevel.HARD,
                depends_on=[3]
            )
        ]
    
    def solve_incrementally(self) -> Dict[str, Any]:
        """Solve sub-problems incrementally, easiest to hardest"""
        print(f"\n{'='*70}")
        print("INCREMENTAL SOLVING")
        print(f"{'='*70}\n")
        
        for sub_problem in self.sub_problems:
            print(f"\n--- Solving Sub-Problem {sub_problem.id} ---")
            print(f"Difficulty: {sub_problem.difficulty.name}")
            print(f"Description: {sub_problem.description}")
            
            # Check dependencies
            if sub_problem.depends_on:
                print(f"Using context from: {sub_problem.depends_on}")
                dependency_context = {
                    dep_id: self.solutions[dep_id]
                    for dep_id in sub_problem.depends_on
                }
            else:
                dependency_context = {}
            
            # Solve sub-problem
            solution = self._solve_sub_problem(sub_problem, dependency_context)
            
            # Store solution
            self.solutions[sub_problem.id] = solution
            sub_problem.solution = solution
            
            print(f"Solution: {solution}")
            
            # Update context
            self.context[f"step_{sub_problem.id}"] = solution
        
        # Combine solutions
        final_solution = self._combine_solutions()
        
        return {
            'sub_problems': len(self.sub_problems),
            'solutions': self.solutions,
            'final_solution': final_solution
        }
    
    def _solve_sub_problem(self, sub_problem: SubProblem, 
                          dependency_context: Dict[int, Any]) -> Any:
        """Solve individual sub-problem"""
        # Simulated solving (in reality, use LLM with context)
        import time
        time.sleep(0.1)  # Simulate thinking
        
        if sub_problem.difficulty == DifficultyLevel.TRIVIAL:
            return f"Basic understanding established"
        elif sub_problem.difficulty == DifficultyLevel.EASY:
            if dependency_context:
                return f"Simple step completed using: {list(dependency_context.keys())}"
            return f"Simple step completed"
        elif sub_problem.difficulty == DifficultyLevel.MEDIUM:
            return f"Intermediate solution building on previous steps"
        elif sub_problem.difficulty == DifficultyLevel.HARD:
            return f"Complex solution integrating: {list(dependency_context.keys())}"
        else:  # EXPERT
            return f"Advanced solution synthesizing all previous work"
    
    def _combine_solutions(self) -> str:
        """Combine all sub-solutions into final answer"""
        print(f"\n{'='*70}")
        print("COMBINING SOLUTIONS")
        print(f"{'='*70}\n")
        
        combined = "Final solution incorporating all steps:\n"
        for sp in self.sub_problems:
            combined += f"  • {sp.description}: {sp.solution}\n"
        
        return combined

class LeastToMostAgent:
    """Agent using least-to-most problem solving"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    
    def solve(self, problem: str) -> Dict[str, Any]:
        """Solve problem using least-to-most approach"""
        print(f"\n{'='*70}")
        print(f"LEAST-TO-MOST PROBLEM SOLVING")
        print(f"{'='*70}")
        print(f"Agent: {self.agent_id}")
        print(f"Problem: {problem}")
        
        solver = LeastToMostSolver()
        
        # Step 1: Decompose
        sub_problems = solver.decompose_problem(problem)
        
        # Step 2: Solve incrementally
        result = solver.solve_incrementally()
        
        # Step 3: Present solution
        print(f"\n{'='*70}")
        print("FINAL SOLUTION")
        print(f"{'='*70}")
        print(result['final_solution'])
        
        return result


# Usage
if __name__ == "__main__":
    print("="*80)
    print("LEAST-TO-MOST PROMPTING PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = LeastToMostAgent("ltm-agent-001")
    
    # Example 1: Math problem
    print("\n" + "="*80)
    print("EXAMPLE 1: Compound Interest Calculation")
    print("="*80)
    
    agent.solve("Calculate compound interest on $1000 at 5% for 10 years")
    
    # Example 2: Essay writing
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Essay Writing")
    print("="*80)
    
    agent.solve("Write an essay about the impact of AI on society")
    
    # Example 3: Optimization
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Optimization Problem")
    print("="*80)
    
    agent.solve("Optimize delivery routes for 20 locations")
```

```python patterns/26_prompt_chaining.py
"""
Prompt Chaining Pattern
Links multiple prompts sequentially, output feeds into next
"""

from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PromptStep:
    """A single step in the prompt chain"""
    name: str
    prompt_template: str
    process_output: Optional[Callable[[str], Any]] = None
    required_inputs: List[str] = None
    
    def __post_init__(self):
        if self.required_inputs is None:
            self.required_inputs = []

@dataclass
class ChainResult:
    """Result from a chain execution"""
    step_name: str
    input_data: Dict[str, Any]
    output: Any
    timestamp: datetime
    duration_ms: float

class PromptChain:
    """Chain of prompts where each output feeds into the next"""
    
    def __init__(self, chain_name: str):
        self.chain_name = chain_name
        self.steps: List[PromptStep] = []
        self.results: List[ChainResult] = []
        self.context: Dict[str, Any] = {}
    
    def add_step(self, step: PromptStep):
        """Add a step to the chain"""
        self.steps.append(step)
        print(f"Added step: {step.name}")
    
    def execute(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the entire chain"""
        print(f"\n{'='*70}")
        print(f"EXECUTING CHAIN: {self.chain_name}")
        print(f"{'='*70}")
        print(f"Steps: {len(self.steps)}")
        print(f"Initial Input: {initial_input}\n")
        
        # Initialize context
        self.context.update(initial_input)
        
        # Execute each step
        for i, step in enumerate(self.steps, 1):
            print(f"\n--- Step {i}/{len(self.steps)}: {step.name} ---")
            
            # Check required inputs
            missing = [inp for inp in step.required_inputs if inp not in self.context]
            if missing:
                raise ValueError(f"Missing required inputs: {missing}")
            
            # Prepare input
            step_input = {k: self.context[k] for k in step.required_inputs} if step.required_inputs else self.context
            print(f"Input: {step_input}")
            
            # Execute step
            import time
            start_time = time.time()
            
            output = self._execute_step(step, step_input)
            
            duration_ms = (time.time() - start_time) * 1000
            
            print(f"Output: {output}")
            print(f"Duration: {duration_ms:.2f}ms")
            
            # Store result
            result = ChainResult(
                step_name=step.name,
                input_data=step_input.copy(),
                output=output,
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )
            self.results.append(result)
            
            # Update context
            self.context[step.name] = output
        
        # Prepare final result
        final_result = {
            'chain_name': self.chain_name,
            'steps_executed': len(self.results),
            'final_output': self.results[-1].output if self.results else None,
            'context': self.context,
            'execution_trace': [
                {
                    'step': r.step_name,
                    'output': r.output,
                    'duration_ms': r.duration_ms
                }
                for r in self.results
            ]
        }
        
        return final_result
    
    def _execute_step(self, step: PromptStep, input_data: Dict[str, Any]) -> Any:
        """Execute a single step"""
        # Fill prompt template
        prompt = step.prompt_template.format(**input_data)
        
        # Simulate LLM call (in reality, call actual LLM)
        raw_output = self._simulate_llm_call(prompt)
        
        # Process output if processor provided
        if step.process_output:
            output = step.process_output(raw_output)
        else:
            output = raw_output
        
        return output
    
    def _simulate_llm_call(self, prompt: str) -> str:
        """Simulate LLM API call"""
        import time
        time.sleep(0.1)  # Simulate latency
        
        # Simple simulation based on prompt content
        if "extract" in prompt.lower():
            return "Extracted data: key points identified"
        elif "analyze" in prompt.lower():
            return "Analysis: positive sentiment, high engagement"
        elif "summarize" in prompt.lower():
            return "Summary: Main points condensed into brief overview"
        elif "generate" in prompt.lower():
            return "Generated content based on analysis"
        else:
            return f"Response to: {prompt[:50]}..."


class PromptChainingAgent:
    """Agent that uses prompt chaining for complex tasks"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    
    def analyze_document(self, document: str) -> Dict[str, Any]:
        """Analyze document using prompt chain"""
        chain = PromptChain("Document Analysis Pipeline")
        
        # Step 1: Extract key information
        chain.add_step(PromptStep(
            name="extraction",
            prompt_template="Extract key information from: {document}",
            required_inputs=["document"]
        ))
        
        # Step 2: Analyze sentiment
        chain.add_step(PromptStep(
            name="sentiment",
            prompt_template="Analyze sentiment of: {extraction}",
            required_inputs=["extraction"]
        ))
        
        # Step 3: Generate summary
        chain.add_step(PromptStep(
            name="summary",
            prompt_template="Summarize findings from extraction: {extraction} and sentiment: {sentiment}",
            required_inputs=["extraction", "sentiment"]
        ))
        
        # Step 4: Create recommendations
        chain.add_step(PromptStep(
            name="recommendations",
            prompt_template="Based on summary: {summary}, provide recommendations",
            required_inputs=["summary"]
        ))
        
        # Execute chain
        result = chain.execute({"document": document})
        
        return result
    
    def create_content(self, topic: str, audience: str) -> Dict[str, Any]:
        """Create content using prompt chain"""
        chain = PromptChain("Content Creation Pipeline")
        
        # Step 1: Research
        chain.add_step(PromptStep(
            name="research",
            prompt_template="Research key points about {topic} for {audience}",
            required_inputs=["topic", "audience"]
        ))
        
        # Step 2: Outline
        chain.add_step(PromptStep(
            name="outline",
            prompt_template="Create outline based on research: {research}",
            required_inputs=["research"]
        ))
        
        # Step 3: Draft
        chain.add_step(PromptStep(
            name="draft",
            prompt_template="Write draft following outline: {outline} for {audience}",
            required_inputs=["outline", "audience"]
        ))
        
        # Step 4: Edit
        chain.add_step(PromptStep(
            name="edited",
            prompt_template="Edit and improve draft: {draft}",
            required_inputs=["draft"]
        ))
        
        # Step 5: Format
        chain.add_step(PromptStep(
            name="formatted",
            prompt_template="Format content for {audience}: {edited}",
            required_inputs=["edited", "audience"]
        ))
        
        # Execute chain
        result = chain.execute({"topic": topic, "audience": audience})
        
        return result
    
    def solve_complex_problem(self, problem: str) -> Dict[str, Any]:
        """Solve complex problem using prompt chain"""
        chain = PromptChain("Problem Solving Pipeline")
        
        # Step 1: Understand problem
        chain.add_step(PromptStep(
            name="understanding",
            prompt_template="Clarify and restate problem: {problem}",
            required_inputs=["problem"]
        ))
        
        # Step 2: Break down
        chain.add_step(PromptStep(
            name="breakdown",
            prompt_template="Break down problem into components: {understanding}",
            required_inputs=["understanding"]
        ))
        
        # Step 3: Generate solutions
        chain.add_step(PromptStep(
            name="solutions",
            prompt_template="Generate possible solutions for: {breakdown}",
            required_inputs=["breakdown"]
        ))
        
        # Step 4: Evaluate
        chain.add_step(PromptStep(
            name="evaluation",
            prompt_template="Evaluate solutions: {solutions}",
            required_inputs=["solutions"]
        ))
        
        # Step 5: Select best
        chain.add_step(PromptStep(
            name="recommendation",
            prompt_template="Recommend best solution from evaluation: {evaluation}",
            required_inputs=["evaluation"]
        ))
        
        # Execute chain
        result = chain.execute({"problem": problem})
        
        return result


# Usage
if __name__ == "__main__":
    print("="*80)
    print("PROMPT CHAINING PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = PromptChainingAgent("chain-agent-001")
    
    # Example 1: Document Analysis
    print("\n" + "="*80)
    print("EXAMPLE 1: Document Analysis Chain")
    print("="*80)
    
    result1 = agent.analyze_document(
        "Customer feedback indicates high satisfaction with product quality "
        "but concerns about delivery times and customer support responsiveness."
    )
    
    print(f"\n{'='*70}")
    print("CHAIN SUMMARY")
    print(f"{'='*70}")
    print(f"Steps Executed: {result1['steps_executed']}")
    print(f"Final Output: {result1['final_output']}")
    
    # Example 2: Content Creation
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Content Creation Chain")
    print("="*80)
    
    result2 = agent.create_content(
        topic="Artificial Intelligence in Healthcare",
        audience="medical professionals"
    )
    
    print(f"\n{'='*70}")
    print("EXECUTION TRACE")
    print(f"{'='*70}")
    for step_info in result2['execution_trace']:
        print(f"\n{step_info['step']}:")
        print(f"  Output: {step_info['output']}")
        print(f"  Duration: {step_info['duration_ms']:.2f}ms")
    
    # Example 3: Problem Solving
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Problem Solving Chain")
    print("="*80)
    
    result3 = agent.solve_complex_problem(
        "How can we reduce customer churn in our SaaS product?"
    )
    
    print(f"\n{'='*70}")
    print("FINAL RECOMMENDATION")
    print(f"{'='*70}")
    print(result3['final_output'])
```

```python patterns/27_tool_routing.py
"""
Tool Routing Pattern
Routes queries to specialized tools/models based on query type
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import re

class ToolType(Enum):
    CALCULATOR = "calculator"
    SEARCH = "search"
    DATABASE = "database"
    CODE_EXECUTOR = "code_executor"
    TRANSLATOR = "translator"
    SUMMARIZER = "summarizer"
    GENERAL_LLM = "general_llm"

@dataclass
class Tool:
    """A tool that can be invoked"""
    name: str
    tool_type: ToolType
    description: str
    function: Callable
    cost: float = 1.0  # Relative cost
    
@dataclass
class RoutingDecision:
    """Decision about which tool to use"""
    selected_tool: Tool
    confidence: float
    reasoning: str
    alternative_tools: List[Tool] = None
    
    def __post_init__(self):
        if self.alternative_tools is None:
            self.alternative_tools = []

class ToolRouter:
    """Routes queries to appropriate tools"""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.routing_rules: List[Dict[str, Any]] = []
        self.routing_history: List[Dict[str, Any]] = []
    
    def register_tool(self, tool: Tool):
        """Register a tool"""
        self.tools[tool.name] = tool
        print(f"Registered tool: {tool.name} ({tool.tool_type.value})")
    
    def add_routing_rule(self, pattern: str, tool_name: str, priority: int = 0):
        """Add a pattern-based routing rule"""
        self.routing_rules.append({
            'pattern': pattern,
            'tool_name': tool_name,
            'priority': priority
        })
        self.routing_rules.sort(key=lambda x: x['priority'], reverse=True)
    
    def route(self, query: str, context: Dict[str, Any] = None) -> RoutingDecision:
        """Determine which tool to use for a query"""
        print(f"\n{'='*60}")
        print(f"ROUTING QUERY")
        print(f"{'='*60}")
        print(f"Query: {query}")
        
        # Try pattern-based routing first
        for rule in self.routing_rules:
            if re.search(rule['pattern'], query, re.IGNORECASE):
                tool = self.tools[rule['tool_name']]
                decision = RoutingDecision(
                    selected_tool=tool,
                    confidence=0.9,
                    reasoning=f"Matched pattern: {rule['pattern']}"
                )
                print(f"Routed to: {tool.name} (pattern match)")
                return decision
        
        # Fallback to heuristic routing
        decision = self._heuristic_route(query, context)
        print(f"Routed to: {decision.selected_tool.name} ({decision.reasoning})")
        
        return decision
    
    def _heuristic_route(self, query: str, context: Dict[str, Any] = None) -> RoutingDecision:
        """Heuristic-based routing"""
        query_lower = query.lower()
        
        # Check for math/calculation
        if any(op in query for op in ['+', '-', '*', '/', 'calculate', 'compute']):
            if 'calculator' in self.tools:
                return RoutingDecision(
                    selected_tool=self.tools['calculator'],
                    confidence=0.95,
                    reasoning="Query contains mathematical operations"
                )
        
        # Check for search keywords
        if any(word in query_lower for word in ['search', 'find', 'look up', 'what is']):
            if 'search' in self.tools:
                return RoutingDecision(
                    selected_tool=self.tools['search'],
                    confidence=0.85,
                    reasoning="Query requires information retrieval"
                )
        
        # Check for code execution
        if any(word in query_lower for word in ['run code', 'execute', 'python', 'script']):
            if 'code_executor' in self.tools:
                return RoutingDecision(
                    selected_tool=self.tools['code_executor'],
                    confidence=0.90,
                    reasoning="Query requires code execution"
                )
        
        # Check for translation
        if any(word in query_lower for word in ['translate', 'translation', 'in spanish', 'in french']):
            if 'translator' in self.tools:
                return RoutingDecision(
                    selected_tool=self.tools['translator'],
                    confidence=0.92,
                    reasoning="Query requires translation"
                )
        
        # Check for summarization
        if any(word in query_lower for word in ['summarize', 'summary', 'tldr']):
            if 'summarizer' in self.tools:
                return RoutingDecision(
                    selected_tool=self.tools['summarizer'],
                    confidence=0.88,
                    reasoning="Query requires summarization"
                )
        
        # Default to general LLM
        if 'general_llm' in self.tools:
            return RoutingDecision(
                selected_tool=self.tools['general_llm'],
                confidence=0.60,
                reasoning="No specific tool matched, using general LLM"
            )
        
        # If no tools available, raise error
        raise ValueError("No suitable tool found for query")
    
    def execute(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Route and execute query"""
        # Route to appropriate tool
        decision = self.route(query, context)
        
        # Execute tool
        print(f"\nExecuting tool: {decision.selected_tool.name}")
        result = decision.selected_tool.function(query, context)
        
        # Record in history
        self.routing_history.append({
            'query': query,
            'tool': decision.selected_tool.name,
            'confidence': decision.confidence,
            'result': result
        })
        
        return {
            'tool_used': decision.selected_tool.name,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'result': result,
            'cost': decision.selected_tool.cost
        }
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        if not self.routing_history:
            return {'total_queries': 0}
        
        tool_usage = {}
        for record in self.routing_history:
            tool = record['tool']
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        return {
            'total_queries': len(self.routing_history),
            'tool_usage': tool_usage,
            'avg_confidence': sum(r['confidence'] for r in self.routing_history) / len(self.routing_history)
        }


# Define tool functions
def calculator_tool(query: str, context: Dict[str, Any] = None) -> Any:
    """Calculator tool"""
    import time
    time.sleep(0.05)
    
    # Extract mathematical expression
    import re
    expr_match = re.search(r'[\d\s\+\-\*/\(\)\.]+', query)
    if expr_match:
        expr = expr_match.group()
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except:
            return "Error: Could not evaluate expression"
    return "No mathematical expression found"

def search_tool(query: str, context: Dict[str, Any] = None) -> Any:
    """Search tool"""
    import time
    time.sleep(0.1)
    return f"Search results for: {query} (top 3 results returned)"

def code_executor_tool(query: str, context: Dict[str, Any] = None) -> Any:
    """Code execution tool"""
    import time
    time.sleep(0.15)
    return "Code executed successfully (output simulated)"

def translator_tool(query: str, context: Dict[str, Any] = None) -> Any:
    """Translation tool"""
    import time
    time.sleep(0.08)
    return "Translation completed (simulated)"

def summarizer_tool(query: str, context: Dict[str, Any] = None) -> Any:
    """Summarization tool"""
    import time
    time.sleep(0.12)
    return "Summary: Key points extracted and condensed"

def general_llm_tool(query: str, context: Dict[str, Any] = None) -> Any:
    """General LLM tool"""
    import time
    time.sleep(0.2)
    return f"General response to: {query}"


class ToolRoutingAgent:
    """Agent with intelligent tool routing"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.router = ToolRouter()
        self._setup_tools()
    
    def _setup_tools(self):
        """Setup available tools"""
        # Register tools
        self.router.register_tool(Tool(
            name="calculator",
            tool_type=ToolType.CALCULATOR,
            description="Performs mathematical calculations",
            function=calculator_tool,
            cost=0.1
        ))
        
        self.router.register_tool(Tool(
            name="search",
            tool_type=ToolType.SEARCH,
            description="Searches for information",
            function=search_tool,
            cost=0.5
        ))
        
        self.router.register_tool(Tool(
            name="code_executor",
            tool_type=ToolType.CODE_EXECUTOR,
            description="Executes code",
            function=code_executor_tool,
            cost=1.0
        ))
        
        self.router.register_tool(Tool(
            name="translator",
            tool_type=ToolType.TRANSLATOR,
            description="Translates text",
            function=translator_tool,
            cost=0.3
        ))
        
        self.router.register_tool(Tool(
            name="summarizer",
            tool_type=ToolType.SUMMARIZER,
            description="Summarizes text",
            function=summarizer_tool,
            cost=0.4
        ))
        
        self.router.register_tool(Tool(
            name="general_llm",
            tool_type=ToolType.GENERAL_LLM,
            description="General purpose LLM",
            function=general_llm_tool,
            cost=1.5
        ))
        
        # Add routing rules
        self.router.add_routing_rule(r'\b\d+\s*[\+\-\*/]\s*\d+\b', 'calculator', priority=10)
        self.router.add_routing_rule(r'\bsearch\b|\bfind\b', 'search', priority=8)
        self.router.add_routing_rule(r'\btranslate\b', 'translator', priority=9)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process query with tool routing"""
        result = self.router.execute(query)
        
        print(f"\n{'='*60}")
        print(f"RESULT")
        print(f"{'='*60}")
        print(f"Tool: {result['tool_used']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Result: {result['result']}")
        print(f"Cost: {result['cost']}")
        
        return result


```python patterns/27_tool_routing.py
# Usage
if __name__ == "__main__":
    print("="*80)
    print("TOOL ROUTING PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = ToolRoutingAgent("router-agent-001")
    
    # Test various queries
    test_queries = [
        "What is 25 * 48?",
        "Search for the latest AI research papers",
        "Translate 'Hello, how are you?' to Spanish",
        "Summarize this article about climate change",
        "Run this Python code: print('Hello World')",
        "What is the capital of France?"
    ]
    
    print("\nProcessing queries with intelligent routing...\n")
    
    results = []
    for query in test_queries:
        print("\n" + "="*80)
        result = agent.process_query(query)
        results.append(result)
    
    # Show routing statistics
    print("\n\n" + "="*80)
    print("ROUTING STATISTICS")
    print("="*80)
    
    stats = agent.router.get_routing_stats()
    print(f"\nTotal Queries: {stats['total_queries']}")
    print(f"Average Confidence: {stats['avg_confidence']:.1%}")
    
    print(f"\nTool Usage:")
    for tool, count in stats['tool_usage'].items():
        percentage = (count / stats['total_queries']) * 100
        print(f"  {tool}: {count} ({percentage:.1f}%)")
    
    # Calculate total cost
    total_cost = sum(r['cost'] for r in results)
    print(f"\nTotal Cost: {total_cost:.2f}")
```

```python patterns/28_streaming_output.py
"""
Streaming Agent Pattern
Produces output incrementally as a stream
"""

import asyncio
from typing import AsyncGenerator, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import time

@dataclass
class StreamChunk:
    """A chunk of streamed output"""
    content: str
    timestamp: datetime
    chunk_index: int
    is_final: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class StreamingAgent:
    """Agent that streams output incrementally"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.total_chunks_sent = 0
    
    async def generate_stream(self, prompt: str, chunk_size: int = 10) -> AsyncGenerator[StreamChunk, None]:
        """Generate streaming response"""
        print(f"\n{'='*60}")
        print(f"STREAMING GENERATION")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Chunk size: {chunk_size} chars\n")
        
        # Simulate full response
        full_response = self._generate_full_response(prompt)
        
        # Stream in chunks
        chunk_index = 0
        position = 0
        
        while position < len(full_response):
            # Get next chunk
            chunk_content = full_response[position:position + chunk_size]
            position += chunk_size
            
            # Create chunk
            chunk = StreamChunk(
                content=chunk_content,
                timestamp=datetime.now(),
                chunk_index=chunk_index,
                is_final=(position >= len(full_response)),
                metadata={'total_length': len(full_response)}
            )
            
            # Simulate processing delay
            await asyncio.sleep(0.05)
            
            # Yield chunk
            yield chunk
            
            chunk_index += 1
            self.total_chunks_sent += 1
    
    def _generate_full_response(self, prompt: str) -> str:
        """Generate full response (simulated)"""
        # In reality, this would call an LLM API
        if "explain" in prompt.lower():
            return (
                "Artificial Intelligence is a field of computer science that aims to create "
                "intelligent machines capable of performing tasks that typically require human "
                "intelligence. This includes learning, reasoning, problem-solving, perception, "
                "and language understanding. AI systems use various techniques including machine "
                "learning, neural networks, and natural language processing to achieve their goals."
            )
        elif "code" in prompt.lower():
            return """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")"""
        else:
            return f"This is a streaming response to: {prompt}. " * 5
    
    async def process_with_streaming(self, query: str) -> Dict[str, Any]:
        """Process query with streaming output"""
        print(f"Query: {query}\n")
        print("Streaming output:")
        print("-" * 60)
        
        full_output = ""
        chunks_received = 0
        start_time = time.time()
        first_chunk_time = None
        
        async for chunk in self.generate_stream(query, chunk_size=15):
            # Print chunk
            print(chunk.content, end='', flush=True)
            
            # Record first chunk time (time to first token)
            if first_chunk_time is None:
                first_chunk_time = time.time()
            
            # Accumulate
            full_output += chunk.content
            chunks_received += 1
            
            # Show final marker
            if chunk.is_final:
                print("\n" + "-" * 60)
                print("[STREAM COMPLETE]")
        
        total_time = time.time() - start_time
        time_to_first_chunk = (first_chunk_time - start_time) if first_chunk_time else 0
        
        return {
            'output': full_output,
            'chunks_received': chunks_received,
            'total_time_sec': total_time,
            'time_to_first_chunk_sec': time_to_first_chunk,
            'chars_per_second': len(full_output) / total_time if total_time > 0 else 0
        }

class BufferedStreamingAgent:
    """Agent that buffers and processes streaming output"""
    
    def __init__(self, agent_id: str, buffer_size: int = 5):
        self.agent_id = agent_id
        self.buffer_size = buffer_size
        self.buffer: List[StreamChunk] = []
    
    async def process_stream_with_buffer(self, stream: AsyncGenerator[StreamChunk, None]):
        """Process stream with buffering"""
        print(f"\n{'='*60}")
        print(f"BUFFERED STREAMING")
        print(f"{'='*60}")
        print(f"Buffer size: {self.buffer_size}\n")
        
        async for chunk in stream:
            # Add to buffer
            self.buffer.append(chunk)
            
            # Process when buffer is full or stream ends
            if len(self.buffer) >= self.buffer_size or chunk.is_final:
                await self._process_buffer()
    
    async def _process_buffer(self):
        """Process buffered chunks"""
        if not self.buffer:
            return
        
        # Combine buffer
        combined = ''.join(chunk.content for chunk in self.buffer)
        
        print(f"[Buffer] Processing {len(self.buffer)} chunks: {combined}")
        
        # Simulate processing
        await asyncio.sleep(0.02)
        
        # Clear buffer
        self.buffer.clear()

class StreamAggregator:
    """Aggregates multiple streams"""
    
    def __init__(self):
        self.streams: List[AsyncGenerator] = []
    
    def add_stream(self, stream: AsyncGenerator):
        """Add a stream to aggregate"""
        self.streams.append(stream)
    
    async def aggregate(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Aggregate multiple streams"""
        # Create tasks for all streams
        tasks = []
        for i, stream in enumerate(self.streams):
            tasks.append(self._consume_stream(i, stream))
        
        # Wait for all streams
        results = await asyncio.gather(*tasks)
        
        # Yield combined results
        for result in results:
            yield result
    
    async def _consume_stream(self, stream_id: int, stream: AsyncGenerator) -> Dict[str, Any]:
        """Consume a single stream"""
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        
        full_output = ''.join(chunk.content for chunk in chunks)
        
        return {
            'stream_id': stream_id,
            'output': full_output,
            'chunk_count': len(chunks)
        }


async def demo_basic_streaming():
    """Demonstrate basic streaming"""
    print("="*80)
    print("EXAMPLE 1: Basic Streaming")
    print("="*80)
    
    agent = StreamingAgent("stream-agent-001")
    
    result = await agent.process_with_streaming(
        "Explain artificial intelligence in simple terms"
    )
    
    print(f"\n{'='*60}")
    print("STREAMING STATISTICS")
    print(f"{'='*60}")
    print(f"Total chunks: {result['chunks_received']}")
    print(f"Total time: {result['total_time_sec']:.2f}s")
    print(f"Time to first chunk: {result['time_to_first_chunk_sec']:.3f}s")
    print(f"Speed: {result['chars_per_second']:.0f} chars/sec")

async def demo_buffered_streaming():
    """Demonstrate buffered streaming"""
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Buffered Streaming")
    print("="*80)
    
    agent = StreamingAgent("stream-agent-002")
    buffered_agent = BufferedStreamingAgent("buffer-agent-001", buffer_size=3)
    
    stream = agent.generate_stream("Write a Python function for calculating fibonacci", chunk_size=20)
    await buffered_agent.process_stream_with_buffer(stream)

async def demo_parallel_streaming():
    """Demonstrate parallel streaming from multiple sources"""
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Parallel Streaming")
    print("="*80)
    
    # Create multiple agents
    agent1 = StreamingAgent("stream-agent-003")
    agent2 = StreamingAgent("stream-agent-004")
    agent3 = StreamingAgent("stream-agent-005")
    
    # Create streams
    stream1 = agent1.generate_stream("Define AI", chunk_size=10)
    stream2 = agent2.generate_stream("Define ML", chunk_size=10)
    stream3 = agent3.generate_stream("Define DL", chunk_size=10)
    
    # Aggregate
    aggregator = StreamAggregator()
    aggregator.add_stream(stream1)
    aggregator.add_stream(stream2)
    aggregator.add_stream(stream3)
    
    print("Aggregating multiple streams...\n")
    
    async for result in aggregator.aggregate():
        print(f"\nStream {result['stream_id']}:")
        print(f"  Chunks: {result['chunk_count']}")
        print(f"  Output: {result['output'][:100]}...")

async def demo_streaming_with_callback():
    """Demonstrate streaming with callbacks"""
    print("\n\n" + "="*80)
    print("EXAMPLE 4: Streaming with Callbacks")
    print("="*80)
    
    agent = StreamingAgent("stream-agent-006")
    
    # Callback for each chunk
    def on_chunk(chunk: StreamChunk):
        print(f"[{chunk.chunk_index}] Received: '{chunk.content}'")
    
    # Callback for completion
    def on_complete(full_output: str):
        print(f"\n[COMPLETE] Total length: {len(full_output)} chars")
    
    full_output = ""
    
    async for chunk in agent.generate_stream("Explain machine learning", chunk_size=12):
        on_chunk(chunk)
        full_output += chunk.content
        
        if chunk.is_final:
            on_complete(full_output)


async def main():
    """Run all streaming demonstrations"""
    print("="*80)
    print("STREAMING AGENT PATTERN DEMONSTRATION")
    print("="*80)
    
    await demo_basic_streaming()
    await demo_buffered_streaming()
    await demo_parallel_streaming()
    await demo_streaming_with_callback()


if __name__ == "__main__":
    asyncio.run(main())
```

```python patterns/29_metacognitive_monitoring.py
"""
Metacognitive Monitoring Pattern
Agent monitors its own thinking process and confidence
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random

class ConfidenceLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class UncertaintySource(Enum):
    AMBIGUOUS_INPUT = "ambiguous_input"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    CONFLICTING_INFORMATION = "conflicting_information"
    KNOWLEDGE_GAP = "knowledge_gap"
    COMPLEX_REASONING = "complex_reasoning"

@dataclass
class ThinkingStep:
    """A step in the reasoning process"""
    step_id: int
    description: str
    confidence: float
    reasoning: str
    uncertainties: List[UncertaintySource] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MetacognitiveAssessment:
    """Assessment of the thinking process"""
    overall_confidence: float
    confidence_level: ConfidenceLevel
    reasoning_quality: float
    identified_gaps: List[str]
    recommendations: List[str]
    should_request_help: bool

class MetacognitiveMonitor:
    """Monitors and assesses the reasoning process"""
    
    def __init__(self):
        self.thinking_steps: List[ThinkingStep] = []
        self.assessments: List[MetacognitiveAssessment] = []
    
    def record_step(self, description: str, confidence: float, 
                   reasoning: str, uncertainties: List[UncertaintySource] = None):
        """Record a thinking step"""
        step = ThinkingStep(
            step_id=len(self.thinking_steps) + 1,
            description=description,
            confidence=confidence,
            reasoning=reasoning,
            uncertainties=uncertainties or []
        )
        self.thinking_steps.append(step)
        
        print(f"\n[Step {step.step_id}] {description}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Reasoning: {reasoning}")
        if uncertainties:
            print(f"  Uncertainties: {[u.value for u in uncertainties]}")
        
        return step
    
    def assess_confidence(self, response: str) -> float:
        """Assess confidence in a response"""
        # Multiple factors for confidence assessment
        factors = []
        
        # Factor 1: Average confidence of steps
        if self.thinking_steps:
            avg_step_confidence = sum(s.confidence for s in self.thinking_steps) / len(self.thinking_steps)
            factors.append(avg_step_confidence)
        
        # Factor 2: Number of uncertainties
        total_uncertainties = sum(len(s.uncertainties) for s in self.thinking_steps)
        uncertainty_penalty = max(0, 1.0 - (total_uncertainties * 0.1))
        factors.append(uncertainty_penalty)
        
        # Factor 3: Consistency of confidence across steps
        if len(self.thinking_steps) > 1:
            confidences = [s.confidence for s in self.thinking_steps]
            variance = sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences)
            consistency = max(0, 1.0 - variance)
            factors.append(consistency)
        
        # Factor 4: Response completeness (simulated)
        completeness = min(1.0, len(response) / 100)  # Assume 100 chars is "complete"
        factors.append(completeness)
        
        # Combine factors
        overall_confidence = sum(factors) / len(factors) if factors else 0.5
        
        return overall_confidence
    
    def identify_knowledge_gaps(self) -> List[str]:
        """Identify gaps in knowledge or reasoning"""
        gaps = []
        
        # Check for uncertainty patterns
        uncertainty_counts = {}
        for step in self.thinking_steps:
            for uncertainty in step.uncertainties:
                uncertainty_counts[uncertainty] = uncertainty_counts.get(uncertainty, 0) + 1
        
        for uncertainty, count in uncertainty_counts.items():
            if count >= 2:
                gaps.append(f"Multiple instances of {uncertainty.value}")
        
        # Check for low confidence steps
        low_confidence_steps = [s for s in self.thinking_steps if s.confidence < 0.6]
        if low_confidence_steps:
            gaps.append(f"{len(low_confidence_steps)} steps with low confidence")
        
        return gaps
    
    def generate_recommendations(self, confidence: float, gaps: List[str]) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        if confidence < 0.5:
            recommendations.append("Consider requesting human assistance")
            recommendations.append("Gather more information before proceeding")
        elif confidence < 0.7:
            recommendations.append("Verify key assumptions")
            recommendations.append("Consider alternative approaches")
        
        if "knowledge_gap" in str(gaps):
            recommendations.append("Consult external knowledge sources")
        
        if "ambiguous_input" in str(gaps):
            recommendations.append("Request clarification from user")
        
        if len(self.thinking_steps) < 3:
            recommendations.append("Break down problem into more steps")
        
        return recommendations
    
    def perform_assessment(self, response: str) -> MetacognitiveAssessment:
        """Perform comprehensive metacognitive assessment"""
        print(f"\n{'='*60}")
        print("METACOGNITIVE ASSESSMENT")
        print(f"{'='*60}")
        
        # Assess confidence
        confidence = self.assess_confidence(response)
        
        # Determine confidence level
        if confidence >= 0.9:
            level = ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.75:
            level = ConfidenceLevel.HIGH
        elif confidence >= 0.6:
            level = ConfidenceLevel.MEDIUM
        elif confidence >= 0.4:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW
        
        # Assess reasoning quality
        reasoning_quality = self._assess_reasoning_quality()
        
        # Identify gaps
        gaps = self.identify_knowledge_gaps()
        
        # Generate recommendations
        recommendations = self.generate_recommendations(confidence, gaps)
        
        # Decide if help needed
        should_request_help = confidence < 0.5 or reasoning_quality < 0.6
        
        assessment = MetacognitiveAssessment(
            overall_confidence=confidence,
            confidence_level=level,
            reasoning_quality=reasoning_quality,
            identified_gaps=gaps,
            recommendations=recommendations,
            should_request_help=should_request_help
        )
        
        self.assessments.append(assessment)
        
        # Print assessment
        print(f"\nOverall Confidence: {confidence:.1%} ({level.value})")
        print(f"Reasoning Quality: {reasoning_quality:.1%}")
        
        if gaps:
            print(f"\nIdentified Gaps:")
            for gap in gaps:
                print(f"  • {gap}")
        
        if recommendations:
            print(f"\nRecommendations:")
            for rec in recommendations:
                print(f"  • {rec}")
        
        if should_request_help:
            print(f"\n⚠️  Recommendation: Request human assistance")
        
        return assessment
    
    def _assess_reasoning_quality(self) -> float:
        """Assess quality of reasoning process"""
        if not self.thinking_steps:
            return 0.0
        
        quality_factors = []
        
        # Factor 1: Depth of reasoning (number of steps)
        depth_score = min(1.0, len(self.thinking_steps) / 5)
        quality_factors.append(depth_score)
        
        # Factor 2: Clarity of reasoning
        clarity_score = sum(1 for s in self.thinking_steps if len(s.reasoning) > 20) / len(self.thinking_steps)
        quality_factors.append(clarity_score)
        
        # Factor 3: Logical flow (simulated - check if confidence improves)
        if len(self.thinking_steps) > 1:
            confidences = [s.confidence for s in self.thinking_steps]
            improving = sum(1 for i in range(1, len(confidences)) if confidences[i] >= confidences[i-1])
            flow_score = improving / (len(confidences) - 1)
            quality_factors.append(flow_score)
        
        return sum(quality_factors) / len(quality_factors)


class MetacognitiveAgent:
    """Agent with metacognitive monitoring capabilities"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.monitor = MetacognitiveMonitor()
    
    def solve_with_monitoring(self, problem: str) -> Dict[str, Any]:
        """Solve problem with metacognitive monitoring"""
        print(f"\n{'='*70}")
        print(f"SOLVING WITH METACOGNITIVE MONITORING")
        print(f"{'='*70}")
        print(f"Problem: {problem}\n")
        
        # Step 1: Understand the problem
        self.monitor.record_step(
            description="Understanding the problem",
            confidence=0.85,
            reasoning="Problem statement is clear and well-defined",
            uncertainties=[]
        )
        
        # Step 2: Identify approach
        uncertainties = []
        if "complex" in problem.lower() or "difficult" in problem.lower():
            uncertainties.append(UncertaintySource.COMPLEX_REASONING)
        
        self.monitor.record_step(
            description="Identifying solution approach",
            confidence=0.75,
            reasoning="Multiple approaches possible, selected most straightforward",
            uncertainties=uncertainties
        )
        
        # Step 3: Gather information
        confidence = 0.8
        uncertainties = []
        
        # Simulate knowledge gaps
        if random.random() < 0.3:
            uncertainties.append(UncertaintySource.KNOWLEDGE_GAP)
            confidence = 0.6
        
        self.monitor.record_step(
            description="Gathering necessary information",
            confidence=confidence,
            reasoning="Retrieved relevant facts and data",
            uncertainties=uncertainties
        )
        
        # Step 4: Apply reasoning
        self.monitor.record_step(
            description="Applying logical reasoning",
            confidence=0.82,
            reasoning="Step-by-step logical deduction applied",
            uncertainties=[]
        )
        
        # Step 5: Formulate response
        response = f"Solution to '{problem}': [Detailed answer would go here]"
        
        self.monitor.record_step(
            description="Formulating final response",
            confidence=0.88,
            reasoning="Response synthesized from all reasoning steps",
            uncertainties=[]
        )
        
        # Perform metacognitive assessment
        assessment = self.monitor.perform_assessment(response)
        
        return {
            'problem': problem,
            'response': response,
            'assessment': assessment,
            'thinking_steps': len(self.monitor.thinking_steps),
            'should_provide_response': not assessment.should_request_help
        }


# Usage
if __name__ == "__main__":
    print("="*80)
    print("METACOGNITIVE MONITORING PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = MetacognitiveAgent("metacog-agent-001")
    
    # Example 1: Clear problem
    print("\n" + "="*80)
    print("EXAMPLE 1: Clear Problem")
    print("="*80)
    
    result1 = agent.solve_with_monitoring(
        "Calculate the sum of the first 10 natural numbers"
    )
    
    print(f"\n{'='*60}")
    print("FINAL DECISION")
    print(f"{'='*60}")
    if result1['should_provide_response']:
        print("✓ Confidence sufficient - providing response")
        print(f"Response: {result1['response']}")
    else:
        print("⚠️  Confidence insufficient - requesting human assistance")
    
    # Example 2: Complex problem
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Complex/Ambiguous Problem")
    print("="*80)
    
    agent2 = MetacognitiveAgent("metacog-agent-002")
    result2 = agent2.solve_with_monitoring(
        "Analyze the complex socioeconomic implications of emerging technologies"
    )
    
    print(f"\n{'='*60}")
    print("FINAL DECISION")
    print(f"{'='*60}")
    if result2['should_provide_response']:
        print("✓ Confidence sufficient - providing response")
    else:
        print("⚠️  Confidence insufficient - requesting human assistance")
        print(f"Recommendations: {result2['assessment'].recommendations}")
```

I've now created 29 comprehensive pattern implementations. These cover a wide range of agentic AI patterns including:

**Recent additions:**
20. Circuit Breaker
21. A/B Testing
22. Asynchronous Agent
23. Graph-of-Thoughts
24. Hierarchical Planning
25. Least-to-Most Prompting
26. Prompt Chaining
27. Tool Routing
28. Streaming Output
29. Metacognitive Monitoring

Would you like me to continue with more patterns? I can implement additional ones such as:
- Constitutional AI
- Active Learning
- Analogical Reasoning
- Multi-modal grounding
- Curriculum learning
- And many others from the original document

Let me know which patterns you'd like to see next!






I'll continue creating Python implementations for the remaining patterns. Let me create comprehensive examples for patterns 30-50.

```30_constitutional_ai.py
"""
Constitutional AI Pattern
Agent follows explicit principles/rules in behavior
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ViolationSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Principle:
    """A constitutional principle"""
    id: str
    name: str
    description: str
    examples_comply: List[str]
    examples_violate: List[str]
    severity: ViolationSeverity = ViolationSeverity.MEDIUM

@dataclass
class Violation:
    """A violation of a principle"""
    principle: Principle
    severity: ViolationSeverity
    description: str
    original_text: str
    suggested_revision: Optional[str] = None

class Constitution:
    """Set of principles governing agent behavior"""
    
    def __init__(self, name: str):
        self.name = name
        self.principles: Dict[str, Principle] = {}
    
    def add_principle(self, principle: Principle):
        """Add a principle to the constitution"""
        self.principles[principle.id] = principle
        print(f"Added principle: {principle.name}")
    
    def check_compliance(self, text: str) -> List[Violation]:
        """Check if text complies with all principles"""
        violations = []
        
        for principle in self.principles.values():
            violation = self._check_principle(text, principle)
            if violation:
                violations.append(violation)
        
        return violations
    
    def _check_principle(self, text: str, principle: Principle) -> Optional[Violation]:
        """Check if text violates a specific principle"""
        text_lower = text.lower()
        
        # Check for violation indicators
        for violate_example in principle.examples_violate:
            if violate_example.lower() in text_lower:
                return Violation(
                    principle=principle,
                    severity=principle.severity,
                    description=f"Text appears to violate principle: {principle.name}",
                    original_text=text,
                    suggested_revision=f"Revise to comply with: {principle.description}"
                )
        
        return None

class ConstitutionalAgent:
    """Agent that follows constitutional principles"""
    
    def __init__(self, agent_id: str, constitution: Constitution):
        self.agent_id = agent_id
        self.constitution = constitution
        self.response_history: List[Dict[str, Any]] = []
    
    def generate_response(self, prompt: str, max_revisions: int = 3) -> Dict[str, Any]:
        """Generate response with constitutional checks"""
        print(f"\n{'='*70}")
        print(f"CONSTITUTIONAL GENERATION")
        print(f"{'='*70}")
        print(f"Prompt: {prompt}\n")
        
        revision_count = 0
        current_response = self._initial_generation(prompt)
        
        while revision_count < max_revisions:
            print(f"\n--- Revision {revision_count} ---")
            print(f"Response: {current_response}\n")
            
            # Check compliance
            violations = self.constitution.check_compliance(current_response)
            
            if not violations:
                print("✓ No violations found - response approved")
                break
            
            # Display violations
            print(f"⚠ Found {len(violations)} violation(s):")
            for i, violation in enumerate(violations, 1):
                print(f"\n{i}. Principle: {violation.principle.name}")
                print(f"   Severity: {violation.severity.value}")
                print(f"   Issue: {violation.description}")
            
            # Critique and revise
            current_response = self._critique_and_revise(
                current_response, 
                violations
            )
            revision_count += 1
        
        # Final validation
        final_violations = self.constitution.check_compliance(current_response)
        is_compliant = len(final_violations) == 0
        
        result = {
            'prompt': prompt,
            'final_response': current_response,
            'revision_count': revision_count,
            'is_compliant': is_compliant,
            'violations': final_violations
        }
        
        self.response_history.append(result)
        
        print(f"\n{'='*70}")
        print("FINAL RESULT")
        print(f"{'='*70}")
        print(f"Revisions: {revision_count}")
        print(f"Compliant: {is_compliant}")
        print(f"Response: {current_response}")
        
        return result
    
    def _initial_generation(self, prompt: str) -> str:
        """Generate initial response (simulated)"""
        # In reality, this would call an LLM
        # Simulate potentially problematic response
        import random
        
        responses = [
            "Here's a helpful response that follows all guidelines.",
            "I can help with that, though I should mention some limitations.",
            "Let me provide harmful information about that topic.",  # Violates safety
            "The answer is definitely X without any uncertainty.",  # Violates humility
        ]
        
        return random.choice(responses)
    
    def _critique_and_revise(self, response: str, violations: List[Violation]) -> str:
        """Critique response and generate revision"""
        print("\nCritiquing and revising...")
        
        # Simulate critique
        critique = "The response violates constitutional principles. "
        for violation in violations:
            critique += f"{violation.principle.name}: {violation.description}. "
        
        print(f"Critique: {critique}")
        
        # Simulate revision (in reality, would use LLM with critique)
        if "harmful" in response.lower():
            revised = "I can't provide harmful information, but I can suggest safe alternatives."
        elif "definitely" in response.lower() or "without any uncertainty" in response.lower():
            revised = "Based on available information, the most likely answer is X, though there may be exceptions."
        else:
            revised = "Here's a helpful and safe response that acknowledges limitations."
        
        print(f"Revised: {revised}")
        
        return revised


def create_default_constitution() -> Constitution:
    """Create a default constitution with common principles"""
    constitution = Constitution("AI Safety Constitution")
    
    # Principle 1: Harmlessness
    constitution.add_principle(Principle(
        id="harmlessness",
        name="Harmlessness",
        description="Do not provide information that could cause harm",
        examples_comply=[
            "I can't provide instructions for harmful activities",
            "That would be dangerous, here's a safe alternative"
        ],
        examples_violate=[
            "harmful information",
            "dangerous instructions",
            "how to harm"
        ],
        severity=ViolationSeverity.CRITICAL
    ))
    
    # Principle 2: Truthfulness
    constitution.add_principle(Principle(
        id="truthfulness",
        name="Truthfulness",
        description="Provide accurate information and acknowledge uncertainty",
        examples_comply=[
            "Based on available data",
            "I'm not certain, but",
            "This might not be accurate"
        ],
        examples_violate=[
            "definitely true without",
            "absolutely certain",
            "no possibility of error"
        ],
        severity=ViolationSeverity.HIGH
    ))
    
    # Principle 3: Respect
    constitution.add_principle(Principle(
        id="respect",
        name="Respect",
        description="Treat all individuals with respect and dignity",
        examples_comply=[
            "respectful response",
            "acknowledging perspective"
        ],
        examples_violate=[
            "disrespectful",
            "offensive language",
            "discriminatory"
        ],
        severity=ViolationSeverity.HIGH
    ))
    
    # Principle 4: Humility
    constitution.add_principle(Principle(
        id="humility",
        name="Humility",
        description="Acknowledge limitations and avoid overconfidence",
        examples_comply=[
            "I may not have complete information",
            "This is my understanding, but",
            "I could be wrong about"
        ],
        examples_violate=[
            "I know everything about",
            "definitely without question",
            "impossible to be wrong"
        ],
        severity=ViolationSeverity.MEDIUM
    ))
    
    # Principle 5: Privacy
    constitution.add_principle(Principle(
        id="privacy",
        name="Privacy",
        description="Respect user privacy and data protection",
        examples_comply=[
            "I don't store personal information",
            "Your privacy is important"
        ],
        examples_violate=[
            "sharing your personal data",
            "storing your information",
            "tracking your activity"
        ],
        severity=ViolationSeverity.CRITICAL
    ))
    
    return constitution


# Usage
if __name__ == "__main__":
    print("="*80)
    print("CONSTITUTIONAL AI PATTERN DEMONSTRATION")
    print("="*80)
    
    # Create constitution
    constitution = create_default_constitution()
    
    print(f"\nConstitution: {constitution.name}")
    print(f"Principles: {len(constitution.principles)}")
    
    # Create agent
    agent = ConstitutionalAgent("const-agent-001", constitution)
    
    # Test prompts
    test_prompts = [
        "How can I improve my Python coding skills?",
        "Explain quantum computing",
        "What's the best way to learn a new language?",
    ]
    
    for prompt in test_prompts:
        result = agent.generate_response(prompt)
        print("\n" + "="*80 + "\n")
    
    # Summary
    print("="*80)
    print("COMPLIANCE SUMMARY")
    print("="*80)
    
    total = len(agent.response_history)
    compliant = sum(1 for r in agent.response_history if r['is_compliant'])
    avg_revisions = sum(r['revision_count'] for r in agent.response_history) / total
    
    print(f"\nTotal Responses: {total}")
    print(f"Compliant: {compliant}")
    print(f"Compliance Rate: {compliant/total:.1%}")
    print(f"Average Revisions: {avg_revisions:.1f}")
```

```
31_active_learning.py
"""
Active Learning Pattern
Agent requests human input for uncertain cases
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import random

class QueryStrategy(Enum):
    UNCERTAINTY_SAMPLING = "uncertainty_sampling"
    QUERY_BY_COMMITTEE = "query_by_committee"
    EXPECTED_MODEL_CHANGE = "expected_model_change"
    DIVERSITY_SAMPLING = "diversity_sampling"

@dataclass
class UnlabeledSample:
    """A sample that needs labeling"""
    id: str
    data: Any
    features: Dict[str, Any]
    uncertainty_score: float = 0.0

@dataclass
class HumanQuery:
    """A query to human for labeling"""
    sample: UnlabeledSample
    question: str
    strategy: QueryStrategy
    timestamp: Any = None
    
    def __post_init__(self):
        from datetime import datetime
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class HumanResponse:
    """Human's response to query"""
    query: HumanQuery
    label: Any
    confidence: float
    feedback: Optional[str] = None

class ActiveLearningAgent:
    """Agent that actively queries humans for uncertain cases"""
    
    def __init__(self, agent_id: str, uncertainty_threshold: float = 0.7):
        self.agent_id = agent_id
        self.uncertainty_threshold = uncertainty_threshold
        self.labeled_data: List[tuple] = []
        self.unlabeled_pool: List[UnlabeledSample] = []
        self.queries: List[HumanQuery] = []
        self.query_budget = 10
        self.queries_used = 0
    
    def add_unlabeled_data(self, samples: List[UnlabeledSample]):
        """Add unlabeled samples to the pool"""
        self.unlabeled_pool.extend(samples)
        print(f"Added {len(samples)} unlabeled samples. Pool size: {len(self.unlabeled_pool)}")
    
    def predict_with_uncertainty(self, sample: UnlabeledSample) -> tuple[Any, float]:
        """Make prediction and estimate uncertainty"""
        # Simulate prediction (in reality, use actual model)
        import random
        
        # Simulate different uncertainty levels
        prediction = random.choice(["Class_A", "Class_B", "Class_C"])
        uncertainty = random.random()
        
        sample.uncertainty_score = uncertainty
        
        return prediction, uncertainty
    
    def select_query_sample(self, strategy: QueryStrategy) -> Optional[UnlabeledSample]:
        """Select which sample to query human about"""
        if not self.unlabeled_pool:
            return None
        
        if strategy == QueryStrategy.UNCERTAINTY_SAMPLING:
            return self._uncertainty_sampling()
        elif strategy == QueryStrategy.QUERY_BY_COMMITTEE:
            return self._query_by_committee()
        elif strategy == QueryStrategy.DIVERSITY_SAMPLING:
            return self._diversity_sampling()
        else:
            return random.choice(self.unlabeled_pool)
    
    def _uncertainty_sampling(self) -> UnlabeledSample:
        """Select sample with highest uncertainty"""
        # Calculate uncertainty for all samples
        for sample in self.unlabeled_pool:
            if sample.uncertainty_score == 0.0:
                _, uncertainty = self.predict_with_uncertainty(sample)
        
        # Return most uncertain
        most_uncertain = max(self.unlabeled_pool, key=lambda s: s.uncertainty_score)
        return most_uncertain
    
    def _query_by_committee(self) -> UnlabeledSample:
        """Select sample where committee disagrees most"""
        # Simulate committee of models
        disagreements = []
        
        for sample in self.unlabeled_pool:
            # Simulate multiple model predictions
            predictions = [
                random.choice(["Class_A", "Class_B", "Class_C"])
                for _ in range(3)
            ]
            
            # Measure disagreement
            unique_predictions = len(set(predictions))
            disagreement = unique_predictions / 3.0
            disagreements.append((sample, disagreement))
        
        # Return sample with most disagreement
        most_disagreed = max(disagreements, key=lambda x: x[1])
        return most_disagreed[0]
    
    def _diversity_sampling(self) -> UnlabeledSample:
        """Select diverse sample to maximize coverage"""
        # Simple diversity: select sample furthest from labeled data
        if not self.labeled_data:
            return random.choice(self.unlabeled_pool)
        
        # Simulate distance calculation
        max_distance_sample = self.unlabeled_pool[0]
        max_distance = 0
        
        for sample in self.unlabeled_pool:
            # Simulate distance to labeled data
            distance = random.random()
            if distance > max_distance:
                max_distance = distance
                max_distance_sample = sample
        
        return max_distance_sample
    
    def query_human(self, sample: UnlabeledSample, strategy: QueryStrategy) -> Optional[HumanResponse]:
        """Query human for label"""
        if self.queries_used >= self.query_budget:
            print("Query budget exhausted!")
            return None
        
        query = HumanQuery(
            sample=sample,
            question=f"Please label this sample: {sample.data}",
            strategy=strategy
        )
        
        self.queries.append(query)
        self.queries_used += 1
        
        print(f"\n{'='*60}")
        print(f"HUMAN QUERY #{self.queries_used}")
        print(f"{'='*60}")
        print(f"Strategy: {strategy.value}")
        print(f"Sample: {sample.data}")
        print(f"Uncertainty: {sample.uncertainty_score:.2%}")
        print(f"Question: {query.question}")
        
        # Simulate human response
        response = self._simulate_human_response(query)
        
        print(f"\nHuman Response:")
        print(f"  Label: {response.label}")
        print(f"  Confidence: {response.confidence:.1%}")
        if response.feedback:
            print(f"  Feedback: {response.feedback}")
        
        # Add to labeled data
        self.labeled_data.append((sample, response.label))
        self.unlabeled_pool.remove(sample)
        
        return response
    
    def _simulate_human_response(self, query: HumanQuery) -> HumanResponse:
        """Simulate human providing label"""
        # Simulate human labeling with some confidence
        label = random.choice(["Class_A", "Class_B", "Class_C"])
        confidence = random.uniform(0.7, 1.0)
        
        feedbacks = [
            "This was clear",
            "Somewhat ambiguous",
            "Difficult case",
            None
        ]
        feedback = random.choice(feedbacks)
        
        return HumanResponse(
            query=query,
            label=label,
            confidence=confidence,
            feedback=feedback
        )
    
    def active_learning_loop(self, strategy: QueryStrategy, iterations: int = 5):
        """Run active learning loop"""
        print(f"\n{'='*70}")
        print(f"ACTIVE LEARNING LOOP")
        print(f"{'='*70}")
        print(f"Strategy: {strategy.value}")
        print(f"Iterations: {iterations}")
        print(f"Query Budget: {self.query_budget}")
        print(f"Unlabeled Pool: {len(self.unlabeled_pool)} samples\n")
        
        for i in range(iterations):
            if self.queries_used >= self.query_budget:
                print("\nQuery budget exhausted. Stopping.")
                break
            
            if not self.unlabeled_pool:
                print("\nNo more unlabeled data. Stopping.")
                break
            
            print(f"\n--- Iteration {i+1} ---")
            
            # Select sample to query
            sample = self.select_query_sample(strategy)
            
            if sample is None:
                print("No suitable sample found.")
                continue
            
            # Query human
            response = self.query_human(sample, strategy)
            
            # Simulate model update (in reality, retrain model)
            print(f"\nUpdating model with new labeled sample...")
            
            # Show progress
            print(f"\nProgress:")
            print(f"  Labeled: {len(self.labeled_data)}")
            print(f"  Unlabeled: {len(self.unlabeled_pool)}")
            print(f"  Queries Used: {self.queries_used}/{self.query_budget}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get active learning statistics"""
        return {
            'total_queries': self.queries_used,
            'labeled_samples': len(self.labeled_data),
            'unlabeled_remaining': len(self.unlabeled_pool),
            'query_efficiency': len(self.labeled_data) / self.queries_used if self.queries_used > 0 else 0,
            'avg_uncertainty': sum(q.sample.uncertainty_score for q in self.queries) / len(self.queries) if self.queries else 0
        }


# Usage
if __name__ == "__main__":
    print("="*80)
    print("ACTIVE LEARNING PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = ActiveLearningAgent("active-learner-001", uncertainty_threshold=0.6)
    
    # Create unlabeled samples
    samples = [
        UnlabeledSample(
            id=f"sample_{i}",
            data=f"Data point {i}",
            features={"feature1": random.random(), "feature2": random.random()}
        )
        for i in range(20)
    ]
    
    agent.add_unlabeled_data(samples)
    
    # Run active learning with different strategies
    print("\n" + "="*80)
    print("Example 1: Uncertainty Sampling")
    print("="*80)
    
    agent.active_learning_loop(QueryStrategy.UNCERTAINTY_SAMPLING, iterations=5)
    
    stats = agent.get_learning_stats()
    
    print(f"\n{'='*70}")
    print("LEARNING STATISTICS")
    print(f"{'='*70}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Example 2: Query by Committee
    print("\n\n" + "="*80)
    print("Example 2: Query by Committee")
    print("="*80)
    
    agent2 = ActiveLearningAgent("active-learner-002")
    samples2 = [
        UnlabeledSample(
            id=f"sample_{i}",
            data=f"Data point {i}",
            features={"feature1": random.random()}
        )
        for i in range(15)
    ]
    agent2.add_unlabeled_data(samples2)
    agent2.active_learning_loop(QueryStrategy.QUERY_BY_COMMITTEE, iterations=3)
```

```
32_analogical_reasoning.py
"""
Analogical Reasoning Pattern
Uses similar past problems to solve new ones
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math

@dataclass
class Case:
    """A case from past experience"""
    id: str
    problem: str
    solution: str
    context: Dict[str, Any]
    features: Dict[str, float]
    success_rate: float = 1.0
    times_used: int = 0

@dataclass
class Analogy:
    """An analogy between current and past problem"""
    source_case: Case
    target_problem: str
    similarity_score: float
    mappings: Dict[str, str]
    adapted_solution: str

class CaseLibrary:
    """Library of past cases for analogical reasoning"""
    
    def __init__(self):
        self.cases: List[Case] = []
    
    def add_case(self, case: Case):
        """Add a case to the library"""
        self.cases.append(case)
        print(f"Added case: {case.id}")
    
    def find_similar_cases(self, problem_features: Dict[str, float], top_k: int = 3) -> List[Tuple[Case, float]]:
        """Find most similar cases"""
        similarities = []
        
        for case in self.cases:
            similarity = self._calculate_similarity(problem_features, case.features)
            similarities.append((case, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _calculate_similarity(self, features1: Dict[str, float], features2: Dict[str, float]) -> float:
        """Calculate cosine similarity between feature vectors"""
        # Get common features
        common_features = set(features1.keys()) & set(features2.keys())
        
        if not common_features:
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(features1[f] * features2[f] for f in common_features)
        mag1 = math.sqrt(sum(features1[f]**2 for f in common_features))
        mag2 = math.sqrt(sum(features2[f]**2 for f in common_features))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)

class AnalogicalReasoningAgent:
    """Agent that uses analogical reasoning"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.case_library = CaseLibrary()
        self.analogies_used: List[Analogy] = []
    
    def solve_by_analogy(self, problem: str, problem_features: Dict[str, float]) -> Dict[str, Any]:
        """Solve problem using analogical reasoning"""
        print(f"\n{'='*70}")
        print(f"ANALOGICAL REASONING")
        print(f"{'='*70}")
        print(f"New Problem: {problem}\n")
        
        # Step 1: Retrieve similar cases
        similar_cases = self.case_library.find_similar_cases(problem_features, top_k=3)
        
        print("Step 1: Retrieved Similar Cases")
        print("-" * 60)
        for i, (case, similarity) in enumerate(similar_cases, 1):
            print(f"\n{i}. Case: {case.id}")
            print(f"   Problem: {case.problem}")
            print(f"   Similarity: {similarity:.2%}")
            print(f"   Success Rate: {case.success_rate:.1%}")
        
        if not similar_cases:
            print("\nNo similar cases found!")
            return {'success': False, 'solution': None}
        
        # Step 2: Select best case
        best_case, best_similarity = similar_cases[0]
        
        print(f"\n\nStep 2: Selected Best Case")
        print("-" * 60)
        print(f"Case: {best_case.id}")
        print(f"Similarity: {best_similarity:.2%}")
        
        # Step 3: Map source to target
        mappings = self._create_mappings(best_case, problem)
        
        print(f"\n\nStep 3: Created Mappings")
        print("-" * 60)
        for source, target in mappings.items():
            print(f"  {source} → {target}")
        
        # Step 4: Adapt solution
        adapted_solution = self._adapt_solution(best_case.solution, mappings)
        
        print(f"\n\nStep 4: Adapted Solution")
        print("-" * 60)
        print(f"Original: {best_case.solution}")
        print(f"Adapted: {adapted_solution}")
        
        # Create analogy record
        analogy = Analogy(
            source_case=best_case,
            target_problem=problem,
            similarity_score=best_similarity,
            mappings=mappings,
            adapted_solution=adapted_solution
        )
        
        self.analogies_used.append(analogy)
        best_case.times_used += 1
        
        return {
            'success': True,
            'solution': adapted_solution,
            'analogy': analogy,
            'confidence': best_similarity
        }
    
    def _create_mappings(self, source_case: Case, target_problem: str) -> Dict[str, str]:
        """Create mappings from source to target"""
        # Simplified mapping (in reality, use more sophisticated methods)
        mappings = {}
        
        # Extract key terms from source and target
        source_terms = set(source_case.problem.lower().split())
        target_terms = set(target_problem.lower().split())
        
        # Create simple mappings for demonstration
        common_terms = source_terms & target_terms
        unique_source = source_terms - common_terms
        unique_target = target_terms - common_terms
        
        # Map unique terms
        unique_source_list = list(unique_source)[:3]
        unique_target_list = list(unique_target)[:3]
        
        for i, source_term in enumerate(unique_source_list):
            if i < len(unique_target_list):
                mappings[source_term] = unique_target_list[i]
        
        return mappings
    
    def _adapt_solution(self, solution: str, mappings: Dict[str, str]) -> str:
        """Adapt solution using mappings"""
        adapted = solution
        
        for source_term, target_term in mappings.items():
            adapted = adapted.replace(source_term, target_term)
        
        return adapted
    
    def learn_from_outcome(self, analogy: Analogy, success: bool):
        """Update case library based on outcome"""
        if success:
            analogy.source_case.success_rate = (
                (analogy.source_case.success_rate * analogy.source_case.times_used + 1.0) /
                (analogy.source_case.times_used + 1)
            )
            print(f"✓ Analogy successful! Updated case {analogy.source_case.id} success rate to {analogy.source_case.success_rate:.1%}")
        else:
            analogy.source_case.success_rate = (
                (analogy.source_case.success_rate * analogy.source_case.times_used) /
                (analogy.source_case.times_used + 1)
            )
            print(f"✗ Analogy failed. Updated case {analogy.source_case.id} success rate to {analogy.source_case.success_rate:.1%}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analogical reasoning statistics"""
        return {
            'total_cases': len(self.case_library.cases),
            'analogies_used': len(self.analogies_used),
            'avg_similarity': sum(a.similarity_score for a in self.analogies_used) / len(self.analogies_used) if self.analogies_used else 0,
            'most_used_case': max(self.case_library.cases, key=lambda c: c.times_used) if self.case_library.cases else None
        }


# Usage
if __name__ == "__main__":
    print("="*80)
    print("ANALOGICAL REASONING PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = AnalogicalReasoningAgent("analogy-agent-001")
    
    # Add cases to library
    cases = [
        Case(
            id="case_001",
            problem="How to optimize database query performance",
            solution="Create indexes on frequently queried columns",
            context={"domain": "database"},
            features={"technical": 0.9, "optimization": 0.8, "database": 1.0},
            success_rate=0.95
        ),
        Case(
            id="case_002",
            problem="How to improve website load time",
            solution="Implement caching and compress assets",
            context={"domain": "web"},
            features={"technical": 0.9, "optimization": 0.9, "web": 1.0},
            success_rate=0.90
        ),
        Case(
            id="case_003",
            problem="How to reduce API response latency",
            solution="Add caching layer and optimize queries",
            context={"domain": "api"},
            features={"technical": 0.8, "optimization": 0.9, "api": 1.0},
            success_rate=0.85
        ),
    ]
    
    for case in cases:
        agent.case_library.add_case(case)
    
        # Solve new problem using analogy
    print("\n" + "="*80)
    print("SOLVING NEW PROBLEM")
    print("="*80)
    
    new_problem = "How to speed up search functionality performance"
    new_features = {"technical": 0.85, "optimization": 0.9, "search": 0.8}
    
    result = agent.solve_by_analogy(new_problem, new_features)
    
    # Simulate outcome
    import random
    success = random.choice([True, True, False])  # 66% success
    
    if result['success']:
        print(f"\n{'='*70}")
        print("OUTCOME")
        print(f"{'='*70}")
        
        agent.learn_from_outcome(result['analogy'], success)
    
    # Statistics
    stats = agent.get_statistics()
    
    print(f"\n{'='*70}")
    print("STATISTICS")
    print(f"{'='*70}")
    print(f"Total Cases: {stats['total_cases']}")
    print(f"Analogies Used: {stats['analogies_used']}")
    print(f"Average Similarity: {stats['avg_similarity']:.1%}")
    if stats['most_used_case']:
        print(f"Most Used Case: {stats['most_used_case'].id} ({stats['most_used_case'].times_used} times)")
```


```
33_iterative_refinement.py
"""
Iterative Refinement Pattern
Agent iteratively improves output through feedback
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

class RefinementCriterion(Enum):
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    CONCISENESS = "conciseness"
    STYLE = "style"

@dataclass
class RefinementIteration:
    """One iteration of refinement"""
    iteration_num: int
    output: str
    scores: Dict[RefinementCriterion, float]
    feedback: str
    improvements: List[str]

class IterativeRefinementAgent:
    """Agent that iteratively refines outputs"""
    
    def __init__(self, agent_id: str, max_iterations: int = 5):
        self.agent_id = agent_id
        self.max_iterations = max_iterations
        self.iterations: List[RefinementIteration] = []
    
    def generate_with_refinement(self, task: str, criteria: List[RefinementCriterion]) -> Dict[str, Any]:
        """Generate output with iterative refinement"""
        print(f"\n{'='*70}")
        print(f"ITERATIVE REFINEMENT")
        print(f"{'='*70}")
        print(f"Task: {task}")
        print(f"Criteria: {[c.value for c in criteria]}")
        print(f"Max Iterations: {self.max_iterations}\n")
        
        # Initial generation
        current_output = self._initial_generation(task)
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}")
            print(f"{'='*60}")
            print(f"\nCurrent Output:\n{current_output}\n")
            
            # Evaluate current output
            scores = self._evaluate_output(current_output, criteria)
            
            print("Evaluation Scores:")
            for criterion, score in scores.items():
                print(f"  {criterion.value}: {score:.1%}")
            
            # Check if satisfactory
            avg_score = sum(scores.values()) / len(scores)
            
            if avg_score >= 0.9:
                print(f"\n✓ Output meets quality threshold ({avg_score:.1%})")
                break
            
            # Generate feedback
            feedback = self._generate_feedback(current_output, scores, criteria)
            print(f"\nFeedback:\n{feedback}")
            
            # Identify improvements
            improvements = self._identify_improvements(scores, criteria)
            print(f"\nImprovements to make:")
            for imp in improvements:
                print(f"  • {imp}")
            
            # Store iteration
            self.iterations.append(RefinementIteration(
                iteration_num=iteration,
                output=current_output,
                scores=scores,
                feedback=feedback,
                improvements=improvements
            ))
            
            # Refine
            current_output = self._refine_output(current_output, feedback, improvements)
        
        # Final evaluation
        final_scores = self._evaluate_output(current_output, criteria)
        
        print(f"\n{'='*70}")
        print("FINAL RESULT")
        print(f"{'='*70}")
        print(f"\nFinal Output:\n{current_output}\n")
        print("\nFinal Scores:")
        for criterion, score in final_scores.items():
            print(f"  {criterion.value}: {score:.1%}")
        
        return {
            'task': task,
            'final_output': current_output,
            'iterations': len(self.iterations),
            'final_scores': final_scores,
            'avg_final_score': sum(final_scores.values()) / len(final_scores),
            'improvement': self._calculate_improvement()
        }
    
    def _initial_generation(self, task: str) -> str:
        """Generate initial output"""
        # Simulate initial generation (would use LLM)
        if "essay" in task.lower():
            return "AI is important. It helps with tasks. Many applications exist."
        elif "code" in task.lower():
            return "def func(x):\n    return x * 2"
        else:
            return "Initial response to the task."
    
    def _evaluate_output(self, output: str, criteria: List[RefinementCriterion]) -> Dict[RefinementCriterion, float]:
        """Evaluate output against criteria"""
        scores = {}
        
        for criterion in criteria:
            if criterion == RefinementCriterion.ACCURACY:
                # Check for factual accuracy indicators
                score = 0.7 if len(output) > 50 else 0.5
            
            elif criterion == RefinementCriterion.CLARITY:
                # Check for clarity indicators
                sentences = output.split('.')
                avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
                score = 0.8 if 10 <= avg_sentence_length <= 25 else 0.6
            
            elif criterion == RefinementCriterion.COMPLETENESS:
                # Check completeness
                score = min(1.0, len(output) / 200)
            
            elif criterion == RefinementCriterion.CONCISENESS:
                # Check conciseness (not too verbose)
                score = 0.9 if len(output) < 300 else 0.7
            
            elif criterion == RefinementCriterion.STYLE:
                # Check style
                score = 0.75 if output[0].isupper() else 0.5
            
            else:
                score = 0.7
            
            scores[criterion] = score
        
        return scores
    
    def _generate_feedback(self, output: str, scores: Dict[RefinementCriterion, float], 
                          criteria: List[RefinementCriterion]) -> str:
        """Generate feedback for improvement"""
        feedback_parts = []
        
        for criterion, score in scores.items():
            if score < 0.7:
                if criterion == RefinementCriterion.ACCURACY:
                    feedback_parts.append("Add more specific details and facts")
                elif criterion == RefinementCriterion.CLARITY:
                    feedback_parts.append("Improve sentence structure and clarity")
                elif criterion == RefinementCriterion.COMPLETENESS:
                    feedback_parts.append("Expand coverage of the topic")
                elif criterion == RefinementCriterion.CONCISENESS:
                    feedback_parts.append("Remove redundancy and be more concise")
                elif criterion == RefinementCriterion.STYLE:
                    feedback_parts.append("Improve writing style and flow")
        
        return ". ".join(feedback_parts) + "." if feedback_parts else "Output is good overall."
    
    def _identify_improvements(self, scores: Dict[RefinementCriterion, float],
                               criteria: List[RefinementCriterion]) -> List[str]:
        """Identify specific improvements needed"""
        improvements = []
        
        # Sort criteria by score (lowest first)
        sorted_criteria = sorted(scores.items(), key=lambda x: x[1])
        
        for criterion, score in sorted_criteria[:2]:  # Focus on top 2 weakest
            if criterion == RefinementCriterion.ACCURACY:
                improvements.append("Add supporting evidence and examples")
            elif criterion == RefinementCriterion.CLARITY:
                improvements.append("Restructure sentences for better flow")
            elif criterion == RefinementCriterion.COMPLETENESS:
                improvements.append("Address all aspects of the topic")
            elif criterion == RefinementCriterion.CONCISENESS:
                improvements.append("Remove unnecessary words")
            elif criterion == RefinementCriterion.STYLE:
                improvements.append("Polish language and tone")
        
        return improvements
    
    def _refine_output(self, output: str, feedback: str, improvements: List[str]) -> str:
        """Refine the output based on feedback"""
        # Simulate refinement (would use LLM with feedback)
        refined = output
        
        # Simple improvements for demonstration
        if "Add supporting evidence" in str(improvements):
            refined += " For example, AI applications include natural language processing and computer vision."
        
        if "Restructure sentences" in str(improvements):
            refined = refined.replace(". ", ". Furthermore, ")
        
        if "Address all aspects" in str(improvements):
            refined += " This technology continues to evolve and impact various industries."
        
        if "Remove unnecessary words" in str(improvements):
            refined = refined.replace("very ", "").replace("really ", "")
        
        # Always make it a bit better
        if len(refined) == len(output):
            refined += " Additionally, this represents significant advancement in the field."
        
        return refined
    
    def _calculate_improvement(self) -> Dict[str, float]:
        """Calculate improvement metrics"""
        if len(self.iterations) < 2:
            return {'initial_score': 0, 'final_score': 0, 'improvement': 0}
        
        initial_scores = self.iterations[0].scores
        final_scores = self.iterations[-1].scores
        
        initial_avg = sum(initial_scores.values()) / len(initial_scores)
        final_avg = sum(final_scores.values()) / len(final_scores)
        
        return {
            'initial_score': initial_avg,
            'final_score': final_avg,
            'improvement': final_avg - initial_avg
        }


# Usage
if __name__ == "__main__":
    print("="*80)
    print("ITERATIVE REFINEMENT PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = IterativeRefinementAgent("refine-agent-001", max_iterations=4)
    
    # Example 1: Essay refinement
    result = agent.generate_with_refinement(
        task="Write a brief essay about artificial intelligence",
        criteria=[
            RefinementCriterion.ACCURACY,
            RefinementCriterion.CLARITY,
            RefinementCriterion.COMPLETENESS,
            RefinementCriterion.STYLE
        ]
    )
    
    print(f"\n{'='*70}")
    print("REFINEMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final Average Score: {result['avg_final_score']:.1%}")
    
    improvement = result['improvement']
    print(f"\nImprovement:")
    print(f"  Initial: {improvement['initial_score']:.1%}")
    print(f"  Final: {improvement['final_score']:.1%}")
    print(f"  Gain: +{improvement['improvement']:.1%}")
    
    # Show iteration progression
    print(f"\n{'='*70}")
    print("ITERATION PROGRESSION")
    print(f"{'='*70}")
    
    for iteration in agent.iterations:
        avg_score = sum(iteration.scores.values()) / len(iteration.scores)
        print(f"\nIteration {iteration.iteration_num}: {avg_score:.1%}")
        print(f"  Output length: {len(iteration.output)} chars")
        print(f"  Improvements: {len(iteration.improvements)}")
```


```
34_blackboard_system.py
"""
Blackboard System Pattern
Shared knowledge space where multiple agents contribute
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import threading

class KnowledgeType(Enum):
    HYPOTHESIS = "hypothesis"
    FACT = "fact"
    INFERENCE = "inference"
    GOAL = "goal"
    CONSTRAINT = "constraint"

@dataclass
class KnowledgeSource:
    """An agent that can contribute to the blackboard"""
    id: str
    name: str
    expertise: str
    can_trigger_on: List[KnowledgeType]
    
    def can_contribute(self, blackboard_state: Dict[str, Any]) -> bool:
        """Check if this source can contribute given current state"""
        # Override in specific knowledge sources
        return True
    
    def contribute(self, blackboard: 'Blackboard') -> Optional['KnowledgeElement']:
        """Make a contribution to the blackboard"""
        raise NotImplementedError

@dataclass
class KnowledgeElement:
    """A piece of knowledge on the blackboard"""
    id: str
    type: KnowledgeType
    content: Any
    source_id: str
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)
    supports: List[str] = field(default_factory=list)
    contradicts: List[str] = field(default_factory=list)

class Blackboard:
    """Shared knowledge space"""
    
    def __init__(self, problem: str):
        self.problem = problem
        self.knowledge: List[KnowledgeElement] = []
        self.knowledge_sources: List[KnowledgeSource] = []
        self.lock = threading.Lock()
        self.element_counter = 0
    
    def add_knowledge_source(self, source: KnowledgeSource):
        """Register a knowledge source"""
        self.knowledge_sources.append(source)
        print(f"Registered knowledge source: {source.name} ({source.expertise})")
    
    def post_knowledge(self, element: KnowledgeElement):
        """Post knowledge to blackboard"""
        with self.lock:
            self.knowledge.append(element)
            print(f"\n[BLACKBOARD] New {element.type.value}: {element.content}")
            print(f"  Source: {element.source_id}")
            print(f"  Confidence: {element.confidence:.1%}")
    
    def get_knowledge_by_type(self, knowledge_type: KnowledgeType) -> List[KnowledgeElement]:
        """Retrieve knowledge of specific type"""
        return [k for k in self.knowledge if k.type == knowledge_type]
    
    def get_recent_knowledge(self, n: int = 5) -> List[KnowledgeElement]:
        """Get most recent knowledge"""
        return self.knowledge[-n:] if len(self.knowledge) >= n else self.knowledge
    
    def get_state(self) -> Dict[str, Any]:
        """Get current blackboard state"""
        return {
            'total_elements': len(self.knowledge),
            'by_type': {
                kt: len([k for k in self.knowledge if k.type == kt])
                for kt in KnowledgeType
            },
            'avg_confidence': sum(k.confidence for k in self.knowledge) / len(self.knowledge) if self.knowledge else 0
        }

class Controller:
    """Controls the problem-solving process"""
    
    def __init__(self, blackboard: Blackboard):
        self.blackboard = blackboard
        self.iteration = 0
        self.max_iterations = 20
    
    def solve(self):
        """Run the blackboard system"""
        print(f"\n{'='*70}")
        print(f"BLACKBOARD SYSTEM - SOLVING: {self.blackboard.problem}")
        print(f"{'='*70}\n")
        
        while self.iteration < self.max_iterations:
            self.iteration += 1
            
            print(f"\n{'='*60}")
            print(f"ITERATION {self.iteration}")
            print(f"{'='*60}")
            
            # Select which knowledge source should act
            source = self._select_knowledge_source()
            
            if source is None:
                print("No knowledge source can contribute. Stopping.")
                break
            
            print(f"\nSelected: {source.name}")
            
            # Let source contribute
            contribution = source.contribute(self.blackboard)
            
            if contribution:
                self.blackboard.post_knowledge(contribution)
            
            # Check if problem is solved
            if self._is_solved():
                print(f"\n✓ Problem solved in {self.iteration} iterations!")
                break
        
        self._print_summary()
    
    def _select_knowledge_source(self) -> Optional[KnowledgeSource]:
        """Select which knowledge source should contribute next"""
        # Get current blackboard state
        state = self.blackboard.get_state()
        
        # Find sources that can contribute
        eligible = []
        
        for source in self.blackboard.knowledge_sources:
            if source.can_contribute(state):
                # Assign priority (simplified)
                priority = self._calculate_priority(source, state)
                eligible.append((source, priority))
        
        if not eligible:
            return None
        
        # Select highest priority
        eligible.sort(key=lambda x: x[1], reverse=True)
        return eligible[0][0]
    
    def _calculate_priority(self, source: KnowledgeSource, state: Dict[str, Any]) -> float:
        """Calculate priority for a knowledge source"""
        # Simple priority based on expertise and current needs
        priority = 0.5
        
        # Boost priority if we need this type of knowledge
        recent = self.blackboard.get_recent_knowledge(3)
        
        if not recent:
            priority += 0.5  # Boost initial sources
        
        return priority
    
    def _is_solved(self) -> bool:
        """Check if problem is solved"""
        # Check if we have a solution hypothesis with high confidence
        hypotheses = self.blackboard.get_knowledge_by_type(KnowledgeType.HYPOTHESIS)
        
        for hyp in hypotheses:
            if hyp.confidence >= 0.85 and "solution" in str(hyp.content).lower():
                return True
        
        return False
    
    def _print_summary(self):
        """Print solution summary"""
        state = self.blackboard.get_state()
        
        print(f"\n{'='*70}")
        print("SOLUTION SUMMARY")
        print(f"{'='*70}")
        print(f"Iterations: {self.iteration}")
        print(f"Total Knowledge Elements: {state['total_elements']}")
        print(f"Average Confidence: {state['avg_confidence']:.1%}")
        
        print(f"\nKnowledge by Type:")
        for kt, count in state['by_type'].items():
            if count > 0:
                print(f"  {kt.value}: {count}")
        
        # Show final hypotheses
        hypotheses = self.blackboard.get_knowledge_by_type(KnowledgeType.HYPOTHESIS)
        if hypotheses:
            print(f"\nFinal Hypotheses:")
            for hyp in hypotheses:
                print(f"  • {hyp.content} (confidence: {hyp.confidence:.1%})")


# Example Knowledge Sources

class InitialAnalysisSource(KnowledgeSource):
    """Analyzes the initial problem"""
    
    def __init__(self):
        super().__init__(
            id="initial_analysis",
            name="Initial Analyzer",
            expertise="problem_analysis",
            can_trigger_on=[KnowledgeType.GOAL]
        )
        self.has_contributed = False
    
    def can_contribute(self, blackboard_state: Dict[str, Any]) -> bool:
        return not self.has_contributed
    
    def contribute(self, blackboard: Blackboard) -> Optional[KnowledgeElement]:
        self.has_contributed = True
        
        blackboard.element_counter += 1
        return KnowledgeElement(
            id=f"elem_{blackboard.element_counter}",
            type=KnowledgeType.FACT,
            content=f"Problem identified: {blackboard.problem}",
            source_id=self.id,
            confidence=0.95
        )

class HypothesisGeneratorSource(KnowledgeSource):
    """Generates solution hypotheses"""
    
    def __init__(self):
        super().__init__(
            id="hypothesis_gen",
            name="Hypothesis Generator",
            expertise="hypothesis_generation",
            can_trigger_on=[KnowledgeType.FACT]
        )
        self.hypotheses_generated = 0
    
    def can_contribute(self, blackboard_state: Dict[str, Any]) -> bool:
        facts = blackboard_state['by_type'].get(KnowledgeType.FACT, 0)
        return facts > 0 and self.hypotheses_generated < 2
    
    def contribute(self, blackboard: Blackboard) -> Optional[KnowledgeElement]:
        self.hypotheses_generated += 1
        
        blackboard.element_counter += 1
        return KnowledgeElement(
            id=f"elem_{blackboard.element_counter}",
            type=KnowledgeType.HYPOTHESIS,
            content=f"Hypothesis {self.hypotheses_generated}: Possible solution approach",
            source_id=self.id,
            confidence=0.6 + (self.hypotheses_generated * 0.1)
        )

class InferenceEngineSource(KnowledgeSource):
    """Makes inferences from existing knowledge"""
    
    def __init__(self):
        super().__init__(
            id="inference_engine",
            name="Inference Engine",
            expertise="logical_inference",
            can_trigger_on=[KnowledgeType.FACT, KnowledgeType.HYPOTHESIS]
        )
        self.inferences_made = 0
    
    def can_contribute(self, blackboard_state: Dict[str, Any]) -> bool:
        facts = blackboard_state['by_type'].get(KnowledgeType.FACT, 0)
        hypotheses = blackboard_state['by_type'].get(KnowledgeType.HYPOTHESIS, 0)
        return facts > 0 and hypotheses > 0 and self.inferences_made < 3
    
    def contribute(self, blackboard: Blackboard) -> Optional[KnowledgeElement]:
        self.inferences_made += 1
        
        blackboard.element_counter += 1
        return KnowledgeElement(
            id=f"elem_{blackboard.element_counter}",
            type=KnowledgeType.INFERENCE,
            content=f"Inference: Based on facts and hypotheses, we can deduce...",
            source_id=self.id,
            confidence=0.75
        )

class SolutionValidatorSource(KnowledgeSource):
    """Validates potential solutions"""
    
    def __init__(self):
        super().__init__(
            id="solution_validator",
            name="Solution Validator",
            expertise="validation",
            can_trigger_on=[KnowledgeType.HYPOTHESIS, KnowledgeType.INFERENCE]
        )
        self.validations_done = 0
    
    def can_contribute(self, blackboard_state: Dict[str, Any]) -> bool:
        hypotheses = blackboard_state['by_type'].get(KnowledgeType.HYPOTHESIS, 0)
        inferences = blackboard_state['by_type'].get(KnowledgeType.INFERENCE, 0)
        return hypotheses > 0 and inferences > 0 and self.validations_done < 2
    
    def contribute(self, blackboard: Blackboard) -> Optional[KnowledgeElement]:
        self.validations_done += 1
        
        # Create high-confidence solution
        blackboard.element_counter += 1
        return KnowledgeElement(
            id=f"elem_{blackboard.element_counter}",
            type=KnowledgeType.HYPOTHESIS,
            content="Final solution: Integrated approach based on all evidence",
            source_id=self.id,
            confidence=0.90
        )


# Usage
if __name__ == "__main__":
    print("="*80)
    print("BLACKBOARD SYSTEM PATTERN DEMONSTRATION")
    print("="*80)
    
    # Create blackboard
    blackboard = Blackboard("Optimize system performance")
    
    # Register knowledge sources
    blackboard.add_knowledge_source(InitialAnalysisSource())
    blackboard.add_knowledge_source(HypothesisGeneratorSource())
    blackboard.add_knowledge_source(InferenceEngineSource())
    blackboard.add_knowledge_source(SolutionValidatorSource())
    
    # Create controller and solve
    controller = Controller(blackboard)
    controller.solve()
    
    # Show all knowledge
    print(f"\n{'='*70}")
    print("ALL KNOWLEDGE ON BLACKBOARD")
    print(f"{'='*70}")
    
    for i, elem in enumerate(blackboard.knowledge, 1):
        print(f"\n{i}. [{elem.type.value}] {elem.content}")
        print(f"   Source: {elem.source_id}, Confidence: {elem.confidence:.1%}")
```



I'll continue implementing the remaining patterns. Let me create implementations for patterns 35-70 and beyond.

```python patterns/35_fallback_graceful_degradation.py
"""
Fallback/Graceful Degradation Pattern
Alternative strategies when primary approach fails
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time

class FallbackLevel(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    EMERGENCY = "emergency"
    HUMAN_ESCALATION = "human_escalation"

@dataclass
class FallbackStrategy:
    """A fallback strategy"""
    level: FallbackLevel
    name: str
    handler: Callable
    timeout_seconds: float = 5.0
    max_retries: int = 1

@dataclass
class ExecutionResult:
    """Result of execution attempt"""
    success: bool
    result: Any
    level_used: FallbackLevel
    error: Optional[str] = None
    latency_ms: float = 0.0

class FallbackAgent:
    """Agent with graceful degradation"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.strategies: List[FallbackStrategy] = []
        self.execution_history: List[ExecutionResult] = []
    
    def register_strategy(self, strategy: FallbackStrategy):
        """Register a fallback strategy"""
        self.strategies.append(strategy)
        # Sort by level priority
        self.strategies.sort(key=lambda s: list(FallbackLevel).index(s.level))
        print(f"Registered {strategy.level.value} strategy: {strategy.name}")
    
    def execute_with_fallback(self, task: str, context: Dict[str, Any] = None) -> ExecutionResult:
        """Execute task with fallback strategies"""
        print(f"\n{'='*70}")
        print(f"EXECUTING WITH FALLBACK")
        print(f"{'='*70}")
        print(f"Task: {task}\n")
        
        context = context or {}
        
        for strategy in self.strategies:
            print(f"\n--- Trying {strategy.level.value}: {strategy.name} ---")
            
            for attempt in range(strategy.max_retries):
                if attempt > 0:
                    print(f"  Retry {attempt + 1}/{strategy.max_retries}")
                
                start_time = time.time()
                
                try:
                    # Execute strategy with timeout
                    result = self._execute_with_timeout(
                        strategy.handler,
                        task,
                        context,
                        strategy.timeout_seconds
                    )
                    
                    latency_ms = (time.time() - start_time) * 1000
                    
                    print(f"  ✓ Success in {latency_ms:.0f}ms")
                    
                    execution_result = ExecutionResult(
                        success=True,
                        result=result,
                        level_used=strategy.level,
                        latency_ms=latency_ms
                    )
                    
                    self.execution_history.append(execution_result)
                    return execution_result
                    
                except TimeoutError as e:
                    print(f"  ⏱ Timeout after {strategy.timeout_seconds}s")
                    if attempt < strategy.max_retries - 1:
                        continue
                    
                except Exception as e:
                    print(f"  ✗ Failed: {str(e)}")
                    if attempt < strategy.max_retries - 1:
                        continue
            
            print(f"  ⚠ {strategy.name} exhausted all retries")
        
        # All strategies failed
        print(f"\n✗ All fallback strategies failed")
        
        result = ExecutionResult(
            success=False,
            result=None,
            level_used=FallbackLevel.HUMAN_ESCALATION,
            error="All fallback strategies exhausted"
        )
        
        self.execution_history.append(result)
        return result
    
    def _execute_with_timeout(self, handler: Callable, task: str, 
                              context: Dict[str, Any], timeout: float) -> Any:
        """Execute handler with timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Execution exceeded {timeout}s")
        
        # Set timeout alarm (Unix only - for demo)
        # In production, use threading.Timer or asyncio.wait_for
        try:
            # Simple timeout simulation
            start = time.time()
            result = handler(task, context)
            elapsed = time.time() - start
            
            if elapsed > timeout:
                raise TimeoutError(f"Execution exceeded {timeout}s")
            
            return result
        except Exception as e:
            raise e
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fallback statistics"""
        if not self.execution_history:
            return {"total_executions": 0}
        
        total = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        
        level_usage = {}
        for result in self.execution_history:
            level = result.level_used.value
            level_usage[level] = level_usage.get(level, 0) + 1
        
        avg_latency = sum(r.latency_ms for r in self.execution_history if r.success) / max(successful, 1)
        
        return {
            'total_executions': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total,
            'level_usage': level_usage,
            'avg_latency_ms': avg_latency
        }


# Example strategy handlers
def primary_strategy(task: str, context: Dict[str, Any]) -> str:
    """Primary high-quality but slow strategy"""
    import random
    time.sleep(0.3)
    
    # 70% success rate
    if random.random() < 0.7:
        return f"High-quality result for: {task}"
    else:
        raise Exception("Primary strategy failed")

def secondary_strategy(task: str, context: Dict[str, Any]) -> str:
    """Secondary faster but lower quality"""
    import random
    time.sleep(0.1)
    
    # 80% success rate
    if random.random() < 0.8:
        return f"Good result for: {task}"
    else:
        raise Exception("Secondary strategy failed")

def tertiary_strategy(task: str, context: Dict[str, Any]) -> str:
    """Tertiary very fast, basic quality"""
    time.sleep(0.05)
    
    # 95% success rate
    if random.random() < 0.95:
        return f"Basic result for: {task}"
    else:
        raise Exception("Tertiary strategy failed")

def emergency_strategy(task: str, context: Dict[str, Any]) -> str:
    """Emergency always-works fallback"""
    return f"Emergency fallback result for: {task}"


# Usage
if __name__ == "__main__":
    print("="*80)
    print("FALLBACK/GRACEFUL DEGRADATION PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = FallbackAgent("fallback-agent-001")
    
    # Register strategies in priority order
    agent.register_strategy(FallbackStrategy(
        level=FallbackLevel.PRIMARY,
        name="Advanced AI Model",
        handler=primary_strategy,
        timeout_seconds=1.0,
        max_retries=2
    ))
    
    agent.register_strategy(FallbackStrategy(
        level=FallbackLevel.SECONDARY,
        name="Standard AI Model",
        handler=secondary_strategy,
        timeout_seconds=0.5,
        max_retries=2
    ))
    
    agent.register_strategy(FallbackStrategy(
        level=FallbackLevel.TERTIARY,
        name="Fast Basic Model",
        handler=tertiary_strategy,
        timeout_seconds=0.2,
        max_retries=1
    ))
    
    agent.register_strategy(FallbackStrategy(
        level=FallbackLevel.EMERGENCY,
        name="Template Response",
        handler=emergency_strategy,
        timeout_seconds=0.1,
        max_retries=1
    ))
    
    # Test with multiple tasks
    tasks = [
        "Analyze customer sentiment",
        "Generate product description",
        "Summarize document",
        "Translate text",
        "Answer question"
    ]
    
    for task in tasks:
        result = agent.execute_with_fallback(task)
        
        print(f"\n{'='*60}")
        print(f"RESULT")
        print(f"{'='*60}")
        print(f"Success: {result.success}")
        print(f"Level Used: {result.level_used.value}")
        if result.success:
            print(f"Result: {result.result}")
            print(f"Latency: {result.latency_ms:.0f}ms")
        print()
    
    # Statistics
    stats = agent.get_statistics()
    
    print(f"\n{'='*70}")
    print("FALLBACK STATISTICS")
    print(f"{'='*70}")
    print(f"Total Executions: {stats['total_executions']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Average Latency: {stats['avg_latency_ms']:.0f}ms")
    
    print(f"\nLevel Usage:")
    for level, count in stats['level_usage'].items():
        percentage = (count / stats['total_executions']) * 100
        print(f"  {level}: {count} ({percentage:.1f}%)")
```

```python patterns/36_sandboxing.py
"""
Sandboxing Pattern
Executes agent actions in isolated environment
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import subprocess
import tempfile
import os
import json

class SandboxType(Enum):
    PROCESS = "process"
    DOCKER = "docker"
    VM = "virtual_machine"
    RESTRICTED_PYTHON = "restricted_python"

@dataclass
class SandboxConfig:
    """Configuration for sandbox"""
    sandbox_type: SandboxType
    memory_limit_mb: int = 512
    cpu_limit_percent: int = 50
    timeout_seconds: int = 5
    network_enabled: bool = False
    allowed_modules: List[str] = None
    
    def __post_init__(self):
        if self.allowed_modules is None:
            self.allowed_modules = ['math', 'json', 'datetime']

@dataclass
class SandboxResult:
    """Result from sandbox execution"""
    success: bool
    output: Any
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    execution_time_ms: float = 0.0
    memory_used_mb: float = 0.0

class Sandbox:
    """Isolated execution environment"""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.execution_count = 0
    
    def execute(self, code: str, context: Dict[str, Any] = None) -> SandboxResult:
        """Execute code in sandbox"""
        self.execution_count += 1
        
        print(f"\n{'='*60}")
        print(f"SANDBOX EXECUTION #{self.execution_count}")
        print(f"{'='*60}")
        print(f"Type: {self.config.sandbox_type.value}")
        print(f"Memory Limit: {self.config.memory_limit_mb}MB")
        print(f"Timeout: {self.config.timeout_seconds}s")
        print(f"\nCode to execute:")
        print("-" * 60)
        print(code)
        print("-" * 60)
        
        if self.config.sandbox_type == SandboxType.RESTRICTED_PYTHON:
            return self._execute_restricted_python(code, context)
        elif self.config.sandbox_type == SandboxType.PROCESS:
            return self._execute_in_process(code, context)
        else:
            return SandboxResult(
                success=False,
                output=None,
                error=f"Sandbox type {self.config.sandbox_type.value} not implemented"
            )
    
    def _execute_restricted_python(self, code: str, context: Dict[str, Any]) -> SandboxResult:
        """Execute Python code with restrictions"""
        import time
        
        start_time = time.time()
        
        # Create restricted globals
        restricted_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'sum': sum,
                'min': min,
                'max': max,
            }
        }
        
        # Add allowed modules
        for module_name in self.config.allowed_modules:
            try:
                restricted_globals[module_name] = __import__(module_name)
            except ImportError:
                pass
        
        # Add context
        if context:
            restricted_globals.update(context)
        
        # Capture output
        import io
        import sys
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Execute with timeout (simplified)
            local_vars = {}
            exec(code, restricted_globals, local_vars)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Get output
            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()
            
            # Get result
            result = local_vars.get('result', stdout or "Execution completed")
            
            print(f"\n✓ Execution successful ({execution_time:.0f}ms)")
            if stdout:
                print(f"Output: {stdout}")
            
            return SandboxResult(
                success=True,
                output=result,
                stdout=stdout,
                stderr=stderr,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            print(f"\n✗ Execution failed: {str(e)}")
            
            return SandboxResult(
                success=False,
                output=None,
                error=str(e),
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time_ms=execution_time
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def _execute_in_process(self, code: str, context: Dict[str, Any]) -> SandboxResult:
        """Execute in separate process"""
        import time
        
        start_time = time.time()
        
        # Create temporary file with code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute in subprocess with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            success = result.returncode == 0
            
            if success:
                print(f"\n✓ Process execution successful ({execution_time:.0f}ms)")
            else:
                print(f"\n✗ Process execution failed (exit code: {result.returncode})")
            
            return SandboxResult(
                success=success,
                output=result.stdout,
                error=result.stderr if not success else None,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time_ms=execution_time
            )
            
        except subprocess.TimeoutExpired:
            print(f"\n✗ Execution timeout after {self.config.timeout_seconds}s")
            
            return SandboxResult(
                success=False,
                output=None,
                error=f"Timeout after {self.config.timeout_seconds}s"
            )
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)

class SandboxedAgent:
    """Agent that executes code in sandbox"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.sandbox = Sandbox(SandboxConfig(
            sandbox_type=SandboxType.RESTRICTED_PYTHON,
            memory_limit_mb=256,
            timeout_seconds=3,
            allowed_modules=['math', 'json', 'datetime', 'random']
        ))
        self.execution_history: List[SandboxResult] = []
    
    def execute_user_code(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute user-provided code safely"""
        print(f"\n{'='*70}")
        print(f"SANDBOXED CODE EXECUTION")
        print(f"{'='*70}")
        
        # Validate code first
        if not self._validate_code(code):
            return {
                'success': False,
                'error': 'Code validation failed: contains prohibited operations'
            }
        
        # Execute in sandbox
        result = self.sandbox.execute(code, context)
        self.execution_history.append(result)
        
        return {
            'success': result.success,
            'output': result.output,
            'error': result.error,
            'execution_time_ms': result.execution_time_ms
        }
    
    def _validate_code(self, code: str) -> bool:
        """Validate code for dangerous operations"""
        prohibited = [
            'import os',
            'import sys',
            'import subprocess',
            '__import__',
            'eval',
            'exec',
            'open',
            'file',
            'input',
            'raw_input',
        ]
        
        code_lower = code.lower()
        
        for prohibited_item in prohibited:
            if prohibited_item in code_lower:
                print(f"⚠️  Validation failed: contains '{prohibited_item}'")
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {'total_executions': 0}
        
        total = len(self.execution_history)
        successful = sum(1 for r in self.execution_history if r.success)
        
        avg_time = sum(r.execution_time_ms for r in self.execution_history) / total
        
        return {
            'total_executions': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total,
            'avg_execution_time_ms': avg_time
        }


# Usage
if __name__ == "__main__":
    print("="*80)
    print("SANDBOXING PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = SandboxedAgent("sandbox-agent-001")
    
    # Test Case 1: Safe code
    print("\n" + "="*80)
    print("TEST 1: Safe Mathematical Calculation")
    print("="*80)
    
    safe_code = """
import math

def calculate():
    result = math.sqrt(16) + math.pi
    print(f"Result: {result}")
    return result

result = calculate()
"""
    
    result1 = agent.execute_user_code(safe_code)
    print(f"\nResult: {result1}")
    
    # Test Case 2: Prohibited code
    print("\n\n" + "="*80)
    print("TEST 2: Dangerous Code (should be blocked)")
    print("="*80)
    
    dangerous_code = """
import os
os.system('rm -rf /')  # Dangerous!
"""
    
    result2 = agent.execute_user_code(dangerous_code)
    print(f"\nResult: {result2}")
    
    # Test Case 3: Code with context
    print("\n\n" + "="*80)
    print("TEST 3: Code with Context Variables")
    print("="*80)
    
    context_code = """
# Use provided context
total = sum(numbers)
average = total / len(numbers)
result = {'total': total, 'average': average}
print(f"Numbers: {numbers}")
print(f"Total: {total}, Average: {average}")
"""
    
    result3 = agent.execute_user_code(
        context_code,
        context={'numbers': [1, 2, 3, 4, 5]}
    )
    print(f"\nResult: {result3}")
    
    # Test Case 4: Timeout
    print("\n\n" + "="*80)
    print("TEST 4: Infinite Loop (should timeout)")
    print("="*80)
    
    timeout_code = """
while True:
    pass
"""
    
    result4 = agent.execute_user_code(timeout_code)
    print(f"\nResult: {result4}")
    
    # Statistics
    stats = agent.get_statistics()
    
    print(f"\n{'='*70}")
    print("EXECUTION STATISTICS")
    print(f"{'='*70}")
    print(f"Total Executions: {stats['total_executions']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Avg Execution Time: {stats['avg_execution_time_ms']:.0f}ms")
```

```python patterns/37_chain_of_verification.py
"""
Chain-of-Verification (CoVe) Pattern
Generates verification questions to check answers
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class VerificationQuestion:
    """A question to verify the answer"""
    question: str
    expected_answer: str
    actual_answer: Optional[str] = None
    is_correct: bool = False

@dataclass
class VerificationResult:
    """Result of verification process"""
    original_answer: str
    verification_questions: List[VerificationQuestion]
    verification_score: float
    revised_answer: Optional[str] = None
    is_verified: bool = False

class ChainOfVerificationAgent:
    """Agent that verifies answers through questioning"""
    
    def __init__(self, agent_id: str, verification_threshold: float = 0.8):
        self.agent_id = agent_id
        self.verification_threshold = verification_threshold
        self.verification_history: List[VerificationResult] = []
    
    def answer_with_verification(self, question: str) -> Dict[str, Any]:
        """Answer question with verification"""
        print(f"\n{'='*70}")
        print(f"CHAIN-OF-VERIFICATION")
        print(f"{'='*70}")
        print(f"Question: {question}\n")
        
        # Step 1: Generate initial answer
        print("Step 1: Generating initial answer...")
        initial_answer = self._generate_answer(question)
        print(f"Initial Answer: {initial_answer}\n")
        
        # Step 2: Generate verification questions
        print("Step 2: Generating verification questions...")
        verification_questions = self._generate_verification_questions(
            question, initial_answer
        )
        
        print(f"Generated {len(verification_questions)} verification questions:")
        for i, vq in enumerate(verification_questions, 1):
            print(f"  {i}. {vq.question}")
        print()
        
        # Step 3: Answer verification questions
        print("Step 3: Answering verification questions...")
        for vq in verification_questions:
            vq.actual_answer = self._answer_verification_question(vq.question)
            vq.is_correct = self._check_answer(vq.expected_answer, vq.actual_answer)
            
            status = "✓" if vq.is_correct else "✗"
            print(f"  {status} {vq.question}")
            print(f"    Expected: {vq.expected_answer}")
            print(f"    Got: {vq.actual_answer}")
        print()
        
        # Step 4: Calculate verification score
        correct_count = sum(1 for vq in verification_questions if vq.is_correct)
        verification_score = correct_count / len(verification_questions) if verification_questions else 0
        
        print(f"Step 4: Verification score: {verification_score:.1%}")
        
        # Step 5: Revise if needed
        revised_answer = None
        is_verified = verification_score >= self.verification_threshold
        
        if not is_verified:
            print("\nStep 5: Verification failed, revising answer...")
            revised_answer = self._revise_answer(
                question, 
                initial_answer, 
                verification_questions
            )
            print(f"Revised Answer: {revised_answer}")
            final_answer = revised_answer
        else:
            print("\nStep 5: Verification passed!")
            final_answer = initial_answer
        
        # Store result
        result = VerificationResult(
            original_answer=initial_answer,
            verification_questions=verification_questions,
            verification_score=verification_score,
            revised_answer=revised_answer,
            is_verified=is_verified
        )
        
        self.verification_history.append(result)
        
        print(f"\n{'='*70}")
        print("FINAL RESULT")
        print(f"{'='*70}")
        print(f"Verified: {is_verified}")
        print(f"Score: {verification_score:.1%}")
        print(f"Final Answer: {final_answer}")
        
        return {
            'question': question,
            'answer': final_answer,
            'verified': is_verified,
            'verification_score': verification_score,
            'was_revised': revised_answer is not None
        }
    
    def _generate_answer(self, question: str) -> str:
        """Generate initial answer (simulated)"""
        # In reality, this would call an LLM
        import random
        
        if "capital" in question.lower():
            # Intentionally sometimes wrong for demonstration
            capitals = {
                "france": ["Paris", "Lyon"],  # Lyon is wrong
                "spain": ["Madrid", "Barcelona"],
                "italy": ["Rome", "Milan"]
            }
            
            for country, options in capitals.items():
                if country in question.lower():
                    return random.choice(options)
        
        return "Generic answer to the question"
    
    def _generate_verification_questions(self, question: str, answer: str) -> List[VerificationQuestion]:
        """Generate questions to verify the answer"""
        verification_questions = []
        
        # Fact-checking questions
        if "capital" in question.lower() and "france" in question.lower():
            verification_questions.extend([
                VerificationQuestion(
                    question="Is Paris the capital of France?",
                    expected_answer="Yes"
                ),
                VerificationQuestion(
                    question="What country is Paris the capital of?",
                    expected_answer="France"
                ),
                VerificationQuestion(
                    question="Is Lyon the capital of France?",
                    expected_answer="No"
                )
            ])
        else:
            # Generic verification questions
            verification_questions.extend([
                VerificationQuestion(
                    question=f"Is '{answer}' a correct answer to '{question}'?",
                    expected_answer="Yes"
                ),
                VerificationQuestion(
                    question=f"Can you explain why '{answer}' is correct?",
                    expected_answer="Valid explanation"
                )
            ])
        
        return verification_questions
    
    def _answer_verification_question(self, question: str) -> str:
        """Answer a verification question"""
        # Simulated answering (would use LLM)
        question_lower = question.lower()
        
        if "is paris the capital of france" in question_lower:
            return "Yes"
        elif "what country is paris" in question_lower:
            return "France"
        elif "is lyon the capital" in question_lower:
            return "No"
        elif "is correct" in question_lower:
            return "Yes"
        else:
            return "Valid explanation provided"
    
    def _check_answer(self, expected: str, actual: str) -> bool:
        """Check if answer matches expectation"""
        # Simple check (in reality, would use semantic similarity)
        expected_lower = expected.lower()
        actual_lower = actual.lower()
        
        # Check for key words
        if "yes" in expected_lower:
            return "yes" in actual_lower
        elif "no" in expected_lower:
            return "no" in actual_lower
        else:
            # Check for overlap
            expected_words = set(expected_lower.split())
            actual_words = set(actual_lower.split())
            overlap = expected_words & actual_words
            return len(overlap) > 0
    
    def _revise_answer(self, question: str, initial_answer: str, 
                       verification_questions: List[VerificationQuestion]) -> str:
        """Revise answer based on verification failures"""
        # Analyze failures
        failed_questions = [vq for vq in verification_questions if not vq.is_correct]
        
        print(f"  Analyzing {len(failed_questions)} failed verifications...")
        
        # Generate revised answer
        # In reality, would use LLM with failed verification context
        if "capital" in question.lower():
            return "Paris (revised based on verification)"
        else:
            return f"{initial_answer} (revised)"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics"""
        if not self.verification_history:
            return {'total_verifications': 0}
        
        total = len(self.verification_history)
        verified = sum(1 for r in self.verification_history if r.is_verified)
        revised = sum(1 for r in self.verification_history if r.revised_answer is not None)
        
        avg_score = sum(r.verification_score for r in self.verification_history) / total
        
        return {
            'total_verifications': total,
            'verified_count': verified,
            'revised_count': revised,
            'verification_rate': verified / total,
            'revision_rate': revised / total,
            'avg_verification_score': avg_score
        }


# Usage
if __name__ == "__main__":
    print("="*80)
    print("CHAIN-OF-VERIFICATION PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = ChainOfVerificationAgent("cove-agent-001", verification_threshold=0.7)
    
    # Test questions
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is 2 + 2?",
    ]
    
```python patterns/37_chain_of_verification.py
# ... continuing from where we left off

    for question in questions:
        result = agent.answer_with_verification(question)
        print("\n" + "="*80 + "\n")
    
    # Statistics
    stats = agent.get_statistics()
    
    print("="*80)
    print("VERIFICATION STATISTICS")
    print("="*80)
    print(f"Total Verifications: {stats['total_verifications']}")
    print(f"Verified: {stats['verified_count']}")
    print(f"Revised: {stats['revised_count']}")
    print(f"Verification Rate: {stats['verification_rate']:.1%}")
    print(f"Revision Rate: {stats['revision_rate']:.1%}")
    print(f"Avg Verification Score: {stats['avg_verification_score']:.1%}")
```

```python patterns/38_progressive_optimization.py
"""
Progressive Optimization Pattern
Iteratively optimizes solution through generations
"""

from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import random

class OptimizationMethod(Enum):
    HILL_CLIMBING = "hill_climbing"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC_ALGORITHM = "genetic_algorithm"
    GRADIENT_DESCENT = "gradient_descent"

@dataclass
class Solution:
    """A candidate solution"""
    id: int
    parameters: Dict[str, float]
    fitness: float
    generation: int

@dataclass
class OptimizationResult:
    """Result of optimization process"""
    best_solution: Solution
    all_solutions: List[Solution]
    generations: int
    convergence_history: List[float]

class ProgressiveOptimizer:
    """Optimizer using progressive refinement"""
    
    def __init__(self, objective_function: Callable, method: OptimizationMethod = OptimizationMethod.HILL_CLIMBING):
        self.objective_function = objective_function
        self.method = method
        self.solution_counter = 0
    
    def optimize(self, initial_params: Dict[str, float], 
                 max_generations: int = 50,
                 population_size: int = 10) -> OptimizationResult:
        """Run progressive optimization"""
        print(f"\n{'='*70}")
        print(f"PROGRESSIVE OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Method: {self.method.value}")
        print(f"Max Generations: {max_generations}")
        print(f"Initial Parameters: {initial_params}\n")
        
        if self.method == OptimizationMethod.HILL_CLIMBING:
            return self._hill_climbing(initial_params, max_generations)
        elif self.method == OptimizationMethod.SIMULATED_ANNEALING:
            return self._simulated_annealing(initial_params, max_generations)
        elif self.method == OptimizationMethod.GENETIC_ALGORITHM:
            return self._genetic_algorithm(initial_params, max_generations, population_size)
        else:
            raise ValueError(f"Method {self.method} not implemented")
    
    def _hill_climbing(self, initial_params: Dict[str, float], max_generations: int) -> OptimizationResult:
        """Hill climbing optimization"""
        current = self._create_solution(initial_params, 0)
        best = current
        all_solutions = [current]
        convergence_history = [current.fitness]
        
        print("Starting Hill Climbing Optimization\n")
        
        for generation in range(1, max_generations):
            # Generate neighbors
            neighbors = self._generate_neighbors(current, generation)
            
            # Find best neighbor
            best_neighbor = max(neighbors, key=lambda s: s.fitness)
            
            # Only move if neighbor is better
            if best_neighbor.fitness > current.fitness:
                current = best_neighbor
                improvement = best_neighbor.fitness - convergence_history[-1]
                print(f"Generation {generation}: Fitness={current.fitness:.4f} (↑{improvement:.4f})")
                
                if current.fitness > best.fitness:
                    best = current
            else:
                print(f"Generation {generation}: No improvement (stuck at {current.fitness:.4f})")
            
            all_solutions.extend(neighbors)
            convergence_history.append(current.fitness)
            
            # Early stopping if no improvement for several generations
            if len(convergence_history) > 10:
                recent = convergence_history[-10:]
                if all(abs(x - recent[0]) < 0.0001 for x in recent):
                    print(f"\nConverged at generation {generation}")
                    break
        
        return OptimizationResult(
            best_solution=best,
            all_solutions=all_solutions,
            generations=generation,
            convergence_history=convergence_history
        )
    
    def _simulated_annealing(self, initial_params: Dict[str, float], max_generations: int) -> OptimizationResult:
        """Simulated annealing optimization"""
        import math
        
        current = self._create_solution(initial_params, 0)
        best = current
        all_solutions = [current]
        convergence_history = [current.fitness]
        
        temperature = 1.0
        cooling_rate = 0.95
        
        print("Starting Simulated Annealing Optimization\n")
        
        for generation in range(1, max_generations):
            # Generate random neighbor
            neighbors = self._generate_neighbors(current, generation, count=1)
            neighbor = neighbors[0]
            
            # Calculate acceptance probability
            delta = neighbor.fitness - current.fitness
            
            if delta > 0:
                # Better solution - always accept
                current = neighbor
                accept = True
            else:
                # Worse solution - accept with probability
                probability = math.exp(delta / temperature)
                accept = random.random() < probability
                if accept:
                    current = neighbor
            
            if current.fitness > best.fitness:
                best = current
            
            status = "✓ Accepted" if accept else "✗ Rejected"
            print(f"Generation {generation}: Fitness={current.fitness:.4f}, Temp={temperature:.3f} - {status}")
            
            all_solutions.append(neighbor)
            convergence_history.append(current.fitness)
            
            # Cool down
            temperature *= cooling_rate
        
        return OptimizationResult(
            best_solution=best,
            all_solutions=all_solutions,
            generations=max_generations,
            convergence_history=convergence_history
        )
    
    def _genetic_algorithm(self, initial_params: Dict[str, float], 
                           max_generations: int, population_size: int) -> OptimizationResult:
        """Genetic algorithm optimization"""
        # Initialize population
        population = [
            self._create_solution(
                self._mutate_params(initial_params, 0.3),
                0
            )
            for _ in range(population_size)
        ]
        
        best = max(population, key=lambda s: s.fitness)
        all_solutions = list(population)
        convergence_history = [best.fitness]
        
        print("Starting Genetic Algorithm Optimization\n")
        print(f"Initial Population: {population_size} solutions")
        print(f"Best Initial Fitness: {best.fitness:.4f}\n")
        
        for generation in range(1, max_generations):
            # Selection
            population.sort(key=lambda s: s.fitness, reverse=True)
            survivors = population[:population_size // 2]
            
            # Crossover and mutation
            offspring = []
            while len(offspring) < population_size - len(survivors):
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                
                child_params = self._crossover(parent1.parameters, parent2.parameters)
                child_params = self._mutate_params(child_params, 0.1)
                
                child = self._create_solution(child_params, generation)
                offspring.append(child)
            
            # New population
            population = survivors + offspring
            
            # Track best
            gen_best = max(population, key=lambda s: s.fitness)
            if gen_best.fitness > best.fitness:
                improvement = gen_best.fitness - best.fitness
                print(f"Generation {generation}: Best={gen_best.fitness:.4f} (↑{improvement:.4f}) - NEW BEST!")
                best = gen_best
            else:
                print(f"Generation {generation}: Best={gen_best.fitness:.4f}")
            
            all_solutions.extend(population)
            convergence_history.append(best.fitness)
        
        return OptimizationResult(
            best_solution=best,
            all_solutions=all_solutions,
            generations=max_generations,
            convergence_history=convergence_history
        )
    
    def _create_solution(self, parameters: Dict[str, float], generation: int) -> Solution:
        """Create a solution and evaluate it"""
        fitness = self.objective_function(parameters)
        
        self.solution_counter += 1
        return Solution(
            id=self.solution_counter,
            parameters=parameters.copy(),
            fitness=fitness,
            generation=generation
        )
    
    def _generate_neighbors(self, solution: Solution, generation: int, count: int = 5) -> List[Solution]:
        """Generate neighboring solutions"""
        neighbors = []
        
        for _ in range(count):
            # Small random perturbation
            new_params = self._mutate_params(solution.parameters, 0.1)
            neighbor = self._create_solution(new_params, generation)
            neighbors.append(neighbor)
        
        return neighbors
    
    def _mutate_params(self, parameters: Dict[str, float], mutation_rate: float) -> Dict[str, float]:
        """Mutate parameters"""
        mutated = {}
        
        for key, value in parameters.items():
            if random.random() < mutation_rate:
                # Apply random change
                change = random.gauss(0, 0.1)
                mutated[key] = max(-1.0, min(1.0, value + change))  # Clamp to [-1, 1]
            else:
                mutated[key] = value
        
        return mutated
    
    def _crossover(self, params1: Dict[str, float], params2: Dict[str, float]) -> Dict[str, float]:
        """Crossover two parameter sets"""
        child = {}
        
        for key in params1.keys():
            # Random mix of parents
            if random.random() < 0.5:
                child[key] = params1[key]
            else:
                child[key] = params2[key]
        
        return child


# Example objective functions
def sphere_function(params: Dict[str, float]) -> float:
    """Simple sphere function (minimize sum of squares)"""
    # Return negative because we're maximizing
    return -sum(x**2 for x in params.values())

def rastrigin_function(params: Dict[str, float]) -> float:
    """Rastrigin function (harder to optimize)"""
    import math
    A = 10
    n = len(params)
    return -(A * n + sum(x**2 - A * math.cos(2 * math.pi * x) for x in params.values()))


# Usage
if __name__ == "__main__":
    print("="*80)
    print("PROGRESSIVE OPTIMIZATION PATTERN DEMONSTRATION")
    print("="*80)
    
    # Initial parameters
    initial = {'x': 0.5, 'y': 0.5, 'z': 0.5}
    
    # Example 1: Hill Climbing
    print("\n" + "="*80)
    print("EXAMPLE 1: Hill Climbing")
    print("="*80)
    
    optimizer1 = ProgressiveOptimizer(sphere_function, OptimizationMethod.HILL_CLIMBING)
    result1 = optimizer1.optimize(initial, max_generations=30)
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Generations: {result1.generations}")
    print(f"Best Fitness: {result1.best_solution.fitness:.4f}")
    print(f"Best Parameters: {result1.best_solution.parameters}")
    print(f"Total Solutions Explored: {len(result1.all_solutions)}")
    
    # Example 2: Simulated Annealing
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Simulated Annealing")
    print("="*80)
    
    optimizer2 = ProgressiveOptimizer(sphere_function, OptimizationMethod.SIMULATED_ANNEALING)
    result2 = optimizer2.optimize(initial, max_generations=30)
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Generations: {result2.generations}")
    print(f"Best Fitness: {result2.best_solution.fitness:.4f}")
    print(f"Best Parameters: {result2.best_solution.parameters}")
    
    # Example 3: Genetic Algorithm
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Genetic Algorithm")
    print("="*80)
    
    optimizer3 = ProgressiveOptimizer(rastrigin_function, OptimizationMethod.GENETIC_ALGORITHM)
    result3 = optimizer3.optimize(initial, max_generations=20, population_size=20)
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*60}")
    print(f"Generations: {result3.generations}")
    print(f"Best Fitness: {result3.best_solution.fitness:.4f}")
    print(f"Best Parameters: {result3.best_solution.parameters}")
    print(f"Total Solutions Explored: {len(result3.all_solutions)}")
    
    # Plot convergence (text-based)
    print(f"\n{'='*60}")
    print("CONVERGENCE PLOT")
    print(f"{'='*60}")
    
    history = result3.convergence_history
    max_fitness = max(history)
    min_fitness = min(history)
    
    for i, fitness in enumerate(history[::5]):  # Every 5th generation
        bar_length = int(40 * (fitness - min_fitness) / (max_fitness - min_fitness + 0.001))
        bar = "█" * bar_length
        print(f"Gen {i*5:3d}: {bar} {fitness:.4f}")
```

```python patterns/39_leader_follower.py
"""
Leader-Follower Multi-Agent Pattern
One agent leads while others assist or follow
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class AgentRole(Enum):
    LEADER = "leader"
    FOLLOWER = "follower"
    SPECIALIST = "specialist"

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """A task to be completed"""
    id: str
    description: str
    required_skill: str
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    result: Any = None

@dataclass
class Agent:
    """An agent in the leader-follower system"""
    id: str
    name: str
    role: AgentRole
    skills: List[str]
    capacity: int = 3
    current_tasks: List[Task] = field(default_factory=list)
    completed_tasks: int = 0
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle task"""
        return (len(self.current_tasks) < self.capacity and 
                task.required_skill in self.skills)
    
    def assign_task(self, task: Task):
        """Assign task to agent"""
        task.assigned_to = self.id
        task.status = TaskStatus.ASSIGNED
        self.current_tasks.append(task)
    
    def complete_task(self, task: Task, result: Any):
        """Mark task as completed"""
        task.status = TaskStatus.COMPLETED
        task.result = result
        if task in self.current_tasks:
            self.current_tasks.remove(task)
        self.completed_tasks += 1

class LeaderAgent(Agent):
    """Leader agent that coordinates others"""
    
    def __init__(self, id: str, name: str, skills: List[str]):
        super().__init__(id, name, AgentRole.LEADER, skills, capacity=5)
        self.followers: List[Agent] = []
    
    def add_follower(self, follower: Agent):
        """Add a follower agent"""
        self.followers.append(follower)
        print(f"[{self.name}] Added follower: {follower.name} ({follower.role.value})")
    
    def delegate_task(self, task: Task) -> bool:
        """Delegate task to best available follower"""
        print(f"\n[{self.name}] Delegating task: {task.description}")
        print(f"  Required skill: {task.required_skill}")
        
        # Find capable followers
        capable = [f for f in self.followers if f.can_handle_task(task)]
        
        if not capable:
            print(f"  ⚠ No capable followers available")
            
            # Leader handles it if possible
            if self.can_handle_task(task):
                print(f"  → Leader will handle it")
                self.assign_task(task)
                return True
            else:
                print(f"  ✗ Cannot delegate")
                return False
        
        # Select best follower (least loaded)
        best_follower = min(capable, key=lambda f: len(f.current_tasks))
        
        print(f"  → Assigned to: {best_follower.name}")
        best_follower.assign_task(task)
        return True
    
    def coordinate(self, tasks: List[Task]):
        """Coordinate task execution"""
        print(f"\n{'='*70}")
        print(f"LEADER COORDINATION: {self.name}")
        print(f"{'='*70}")
        print(f"Total tasks: {len(tasks)}")
        print(f"Followers: {len(self.followers)}\n")
        
        # Prioritize tasks
        tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # Delegate tasks
        for task in tasks:
            self.delegate_task(task)
        
        # Execute tasks
        print(f"\n{'='*70}")
        print("EXECUTION PHASE")
        print(f"{'='*70}\n")
        
        self._execute_all_tasks()
        
        # Report results
        self._report_results(tasks)
    
    def _execute_all_tasks(self):
        """Execute all assigned tasks"""
        # Execute leader's tasks
        for task in self.current_tasks[:]:
            result = self._execute_task(task)
            self.complete_task(task, result)
        
        # Execute followers' tasks
        for follower in self.followers:
            for task in follower.current_tasks[:]:
                result = self._execute_task(task)
                follower.complete_task(task, result)
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a task"""
        print(f"[{task.assigned_to}] Executing: {task.description}")
        
        # Simulate task execution
        import time
        import random
        time.sleep(0.1)
        
        # Simulate occasional failure
        if random.random() < 0.1:
            task.status = TaskStatus.FAILED
            print(f"  ✗ Failed")
            return None
        
        result = f"Result for {task.description}"
        print(f"  ✓ Completed")
        return result
    
    def _report_results(self, tasks: List[Task]):
        """Report coordination results"""
        print(f"\n{'='*70}")
        print("COORDINATION RESULTS")
        print(f"{'='*70}\n")
        
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        
        print(f"Total Tasks: {len(tasks)}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {completed/len(tasks):.1%}")
        
        print(f"\nAgent Performance:")
        print(f"  {self.name} (Leader): {self.completed_tasks} tasks")
        
        for follower in self.followers:
            print(f"  {follower.name} ({follower.role.value}): {follower.completed_tasks} tasks")
        
        # Task breakdown by skill
        print(f"\nTasks by Skill:")
        skills = {}
        for task in tasks:
            skill = task.required_skill
            skills[skill] = skills.get(skill, 0) + 1
        
        for skill, count in skills.items():
            print(f"  {skill}: {count}")

class LeaderFollowerSystem:
    """Leader-follower multi-agent system"""
    
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.leaders: List[LeaderAgent] = []
        self.tasks: List[Task] = []
    
    def add_leader(self, leader: LeaderAgent):
        """Add leader to system"""
        self.leaders.append(leader)
        print(f"Added leader: {leader.name}")
    
    def create_team(self, leader: LeaderAgent, followers: List[Agent]):
        """Create team with leader and followers"""
        print(f"\n{'='*60}")
        print(f"CREATING TEAM: {leader.name}'s Team")
        print(f"{'='*60}\n")
        
        for follower in followers:
            leader.add_follower(follower)
        
        if leader not in self.leaders:
            self.add_leader(leader)
    
    def add_tasks(self, tasks: List[Task]):
        """Add tasks to system"""
        self.tasks.extend(tasks)
        print(f"Added {len(tasks)} tasks to system")
    
    def execute(self):
        """Execute all tasks"""
        print(f"\n{'='*70}")
        print(f"LEADER-FOLLOWER SYSTEM: {self.system_name}")
        print(f"{'='*70}\n")
        
        for leader in self.leaders:
            # Assign tasks to this leader's team
            leader.coordinate(self.tasks)


# Usage
if __name__ == "__main__":
    print("="*80)
    print("LEADER-FOLLOWER PATTERN DEMONSTRATION")
    print("="*80)
    
    # Create leader
    leader = LeaderAgent(
        id="leader_1",
        name="Project Manager",
        skills=["planning", "coordination", "coding"]
    )
    
    # Create follower agents
    followers = [
        Agent(
            id="dev_1",
            name="Backend Developer",
            role=AgentRole.FOLLOWER,
            skills=["coding", "database", "api"],
            capacity=3
        ),
        Agent(
            id="dev_2",
            name="Frontend Developer",
            role=AgentRole.FOLLOWER,
            skills=["coding", "ui", "design"],
            capacity=3
        ),
        Agent(
            id="specialist_1",
            name="DevOps Engineer",
            role=AgentRole.SPECIALIST,
            skills=["deployment", "infrastructure", "monitoring"],
            capacity=2
        ),
        Agent(
            id="tester_1",
            name="QA Tester",
            role=AgentRole.SPECIALIST,
            skills=["testing", "quality_assurance"],
            capacity=4
        )
    ]
    
    # Create system
    system = LeaderFollowerSystem("Software Development Team")
    
    # Build team
    system.create_team(leader, followers)
    
    # Create tasks
    tasks = [
        Task("T1", "Implement user authentication", "coding", priority=5),
        Task("T2", "Design user interface", "ui", priority=4),
        Task("T3", "Set up database", "database", priority=5),
        Task("T4", "Create API endpoints", "api", priority=4),
        Task("T5", "Deploy to production", "deployment", priority=3),
        Task("T6", "Write unit tests", "testing", priority=3),
        Task("T7", "Set up monitoring", "monitoring", priority=2),
        Task("T8", "Quality assurance", "quality_assurance", priority=3),
        Task("T9", "Implement dashboard", "ui", priority=2),
        Task("T10", "Optimize database queries", "database", priority=2),
    ]
    
    system.add_tasks(tasks)
    
    # Execute
    system.execute()
```

```python patterns/40_competitive_multi_agent.py
"""
Competitive Multi-Agent Pattern
Agents compete to produce best solution
"""

from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
import random
import time

class CompetitionType(Enum):
    QUALITY = "quality"
    SPEED = "speed"
    EFFICIENCY = "efficiency"
    CREATIVITY = "creativity"

@dataclass
class Submission:
    """Agent's submission"""
    agent_id: str
    solution: Any
    score: float
    time_taken_ms: float
    metadata: Dict[str, Any]

@dataclass
class CompetitionResult:
    """Result of competition"""
    winner: Submission
    all_submissions: List[Submission]
    competition_type: CompetitionType
    evaluation_criteria: Dict[str, float]

class CompetitiveAgent:
    """Agent that competes with others"""
    
    def __init__(self, agent_id: str, name: str, strategy: str):
        self.agent_id = agent_id
        self.name = name
        self.strategy = strategy
        self.wins = 0
        self.total_competitions = 0
    
    def compete(self, task: str, competition_type: CompetitionType) -> Submission:
        """Generate solution for competition"""
        print(f"\n[{self.name}] Competing with {self.strategy} strategy...")
        
        start_time = time.time()
        
        # Generate solution based on strategy
        if self.strategy == "fast":
            solution = self._fast_solution(task)
        elif self.strategy == "thorough":
            solution = self._thorough_solution(task)
        elif self.strategy == "creative":
            solution = self._creative_solution(task)
        else:
            solution = self._balanced_solution(task)
        
        time_taken = (time.time() - start_time) * 1000
        
        # Calculate score
        score = self._evaluate_solution(solution, competition_type)
        
        print(f"  Solution: {solution}")
        print(f"  Score: {score:.2f}")
        print(f"  Time: {time_taken:.0f}ms")
        
        return Submission(
            agent_id=self.agent_id,
            solution=solution,
            score=score,
            time_taken_ms=time_taken,
            metadata={"strategy": self.strategy, "agent_name": self.name}
        )
    
    def _fast_solution(self, task: str) -> str:
        """Quick but basic solution"""
        time.sleep(0.05)
        return f"Fast solution for: {task}"
    
    def _thorough_solution(self, task: str) -> str:
        """Slow but high-quality solution"""
        time.sleep(0.3)
        return f"Comprehensive and detailed solution for: {task}"
    
    def _creative_solution(self, task: str) -> str:
        """Creative approach"""
        time.sleep(0.15)
        return f"Innovative and creative solution for: {task}"
    
    def _balanced_solution(self, task: str) -> str:
        """Balanced approach"""
        time.sleep(0.1)
        return f"Well-balanced solution for: {task}"
    
    def _evaluate_solution(self, solution: str, competition_type: CompetitionType) -> float:
        """Evaluate solution quality"""
        base_score = len(solution) / 100  # Simple metric
        
        # Adjust based on strategy and competition type
        if competition_type == CompetitionType.QUALITY:
            if self.strategy == "thorough":
                return base_score * random.uniform(0.9, 1.0)
            elif self.strategy == "fast":
                return base_score * random.uniform(0.6, 0.8)
        elif competition_type == CompetitionType.SPEED:
            if self.strategy == "fast":
                return base_score * random.uniform(0.9, 1.0)
            elif self.strategy == "thorough":
                return base_score * random.uniform(0.5, 0.7)
        elif competition_type == CompetitionType.CREATIVITY:
            if self.strategy == "creative":
                return base_score * random.uniform(0.9, 1.0)
            else:
                return base_score * random.uniform(0.6, 0.8)
        
        return base_score * random.uniform(0.7, 0.9)

class CompetitionArena:
    """Arena where agents compete"""
    
    def __init__(self, arena_name: str):
        self.arena_name = arena_name
        self.agents: List[CompetitiveAgent] = []
        self.competition_history: List[CompetitionResult] = []
    
    def register_agent(self, agent: CompetitiveAgent):
        """Register agent for competition"""
        self.agents.append(agent)
        print(f"Registered: {agent.name} ({agent.strategy} strategy)")
    
    def run_competition(self, task: str, competition_type: CompetitionType) -> CompetitionResult:
        """Run competition among all agents"""
        print(f"\n{'='*70}")
        print(f"COMPETITION: {self.arena_name}")
        print(f"{'='*70}")
        print(f"Task: {task}")
        print(f"Type: {competition_type.value}")
        print(f"Competitors: {len(self.agents)}")
        
        # Get submissions from all agents
        submissions = []
        for agent in self.agents:
            submission = agent.compete(task, competition_type)
            submissions.append(submission)
            agent.total_competitions += 1
        
        # Evaluate and rank
        print(f"\n{'='*70}")
        print("EVALUATION")
        print(f"{'='*70}\n")
        
        ranked = self._rank_submissions(submissions, competition_type)
        
        # Show rankings
        for i, submission in enumerate(ranked, 1):
            agent_name = submission.metadata['agent_name']
            print(f"{i}. {agent_name}: {submission.score:.2f} points ({submission.time_taken_ms:.0f}ms)")
        
        # Determine winner
        winner = ranked[0]
        winner_agent = next(a for a in self.agents if a.agent_id == winner.agent_id)
        winner_agent.wins += 1
        
                print(f"\n🏆 Winner: {winner.metadata['agent_name']}")
        
        # Create result
        result = CompetitionResult(
            winner=winner,
            all_submissions=submissions,
            competition_type=competition_type,
            evaluation_criteria={
                "quality_weight": 0.6 if competition_type == CompetitionType.QUALITY else 0.3,
                "speed_weight": 0.6 if competition_type == CompetitionType.SPEED else 0.2,
                "creativity_weight": 0.6 if competition_type == CompetitionType.CREATIVITY else 0.2
            }
        )
        
        self.competition_history.append(result)
        return result
    
    def _rank_submissions(self, submissions: List[Submission], 
                         competition_type: CompetitionType) -> List[Submission]:
        """Rank submissions by score"""
        # Multi-criteria ranking
        ranked = sorted(submissions, key=lambda s: (
            s.score * 0.7 +  # Quality component
            (1000 / max(s.time_taken_ms, 1)) * 0.3  # Speed component
        ), reverse=True)
        
        return ranked
    
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get overall leaderboard"""
        leaderboard = []
        
        for agent in self.agents:
            win_rate = agent.wins / agent.total_competitions if agent.total_competitions > 0 else 0
            
            leaderboard.append({
                'name': agent.name,
                'strategy': agent.strategy,
                'wins': agent.wins,
                'total_competitions': agent.total_competitions,
                'win_rate': win_rate
            })
        
        # Sort by win rate
        leaderboard.sort(key=lambda x: x['win_rate'], reverse=True)
        
        return leaderboard
    
    def print_leaderboard(self):
        """Print leaderboard"""
        leaderboard = self.get_leaderboard()
        
        print(f"\n{'='*70}")
        print("OVERALL LEADERBOARD")
        print(f"{'='*70}\n")
        
        print(f"{'Rank':<6} {'Agent':<25} {'Strategy':<12} {'Wins':<6} {'Total':<6} {'Win Rate':<10}")
        print("-" * 70)
        
        for i, entry in enumerate(leaderboard, 1):
            print(f"{i:<6} {entry['name']:<25} {entry['strategy']:<12} "
                  f"{entry['wins']:<6} {entry['total_competitions']:<6} {entry['win_rate']:.1%}")


# Usage
if __name__ == "__main__":
    print("="*80)
    print("COMPETITIVE MULTI-AGENT PATTERN DEMONSTRATION")
    print("="*80)
    
    # Create arena
    arena = CompetitionArena("AI Solutions Arena")
    
    # Create competing agents
    agents = [
        CompetitiveAgent("agent_1", "SpeedDemon", "fast"),
        CompetitiveAgent("agent_2", "DeepThinker", "thorough"),
        CompetitiveAgent("agent_3", "Innovator", "creative"),
        CompetitiveAgent("agent_4", "AllRounder", "balanced"),
    ]
    
    print("\nRegistering agents...")
    for agent in agents:
        arena.register_agent(agent)
    
    # Run multiple competitions
    competitions = [
        ("Optimize database query performance", CompetitionType.QUALITY),
        ("Generate creative marketing slogan", CompetitionType.CREATIVITY),
        ("Process data pipeline quickly", CompetitionType.SPEED),
        ("Design scalable architecture", CompetitionType.QUALITY),
        ("Create engaging user interface", CompetitionType.CREATIVITY),
    ]
    
    for task, comp_type in competitions:
        arena.run_competition(task, comp_type)
        print("\n" + "="*80 + "\n")
        time.sleep(0.2)
    
    # Show final leaderboard
    arena.print_leaderboard()
    
    
    
    
```
41_cooperative_multiagency.py
"""
Cooperative Multi-Agent Pattern
Agents work together sharing information and goals
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

class MessageType(Enum):
    REQUEST_HELP = "request_help"
    OFFER_HELP = "offer_help"
    SHARE_INFO = "share_info"
    TASK_COMPLETE = "task_complete"
    COORDINATE = "coordinate"

@dataclass
class Message:
    """Message between agents"""
    sender_id: str
    receiver_id: str  # or "broadcast"
    message_type: MessageType
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SharedKnowledge:
    """Knowledge shared among agents"""
    key: str
    value: Any
    contributor_id: str
    timestamp: datetime = field(default_factory=datetime.now)

class CooperativeAgent:
    """Agent that cooperates with others"""
    
    def __init__(self, agent_id: str, name: str, expertise: List[str]):
        self.agent_id = agent_id
        self.name = name
        self.expertise = expertise
        self.messages: List[Message] = []
        self.shared_memory: Dict[str, SharedKnowledge] = {}
        self.helpers: List['CooperativeAgent'] = []
        self.tasks_completed = 0
        self.help_given = 0
        self.help_received = 0
    
    def receive_message(self, message: Message):
        """Receive message from another agent"""
        self.messages.append(message)
        
        print(f"[{self.name}] Received {message.message_type.value} from {message.sender_id}")
        
        # Process message
        if message.message_type == MessageType.REQUEST_HELP:
            self._handle_help_request(message)
        elif message.message_type == MessageType.SHARE_INFO:
            self._handle_shared_info(message)
        elif message.message_type == MessageType.TASK_COMPLETE:
            self._handle_task_complete(message)
    
    def _handle_help_request(self, message: Message):
        """Handle request for help"""
        task = message.content
        
        # Check if we have expertise
        if any(exp in task.get('required_skills', []) for exp in self.expertise):
            print(f"  → Can help with this task!")
            self.help_given += 1
            
            # Offer help
            response = Message(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.OFFER_HELP,
                content={"task_id": task.get('id'), "expertise": self.expertise}
            )
            
            # In real system, would send to sender
            print(f"  → Offering help to {message.sender_id}")
        else:
            print(f"  → Cannot help (no relevant expertise)")
    
    def _handle_shared_info(self, message: Message):
        """Handle shared information"""
        info = message.content
        
        # Store in shared memory
        key = info.get('key')
        value = info.get('value')
        
        if key:
            self.shared_memory[key] = SharedKnowledge(
                key=key,
                value=value,
                contributor_id=message.sender_id
            )
            print(f"  → Stored shared knowledge: {key}")
    
    def _handle_task_complete(self, message: Message):
        """Handle task completion notification"""
        result = message.content
        print(f"  → Noted: {message.sender_id} completed {result.get('task_id')}")
    
    def share_knowledge(self, key: str, value: Any, team: List['CooperativeAgent']):
        """Share knowledge with team"""
        print(f"[{self.name}] Sharing knowledge: {key}")
        
        message = Message(
            sender_id=self.agent_id,
            receiver_id="broadcast",
            message_type=MessageType.SHARE_INFO,
            content={"key": key, "value": value}
        )
        
        for agent in team:
            if agent.agent_id != self.agent_id:
                agent.receive_message(message)
    
    def request_help(self, task: Dict[str, Any], team: List['CooperativeAgent']) -> List['CooperativeAgent']:
        """Request help from team"""
        print(f"[{self.name}] Requesting help for task: {task.get('id')}")
        
        message = Message(
            sender_id=self.agent_id,
            receiver_id="broadcast",
            message_type=MessageType.REQUEST_HELP,
            content=task
        )
        
        helpers = []
        for agent in team:
            if agent.agent_id != self.agent_id:
                agent.receive_message(message)
                
                # Check if they offered help (simplified)
                if any(exp in task.get('required_skills', []) for exp in agent.expertise):
                    helpers.append(agent)
        
        self.help_received += len(helpers)
        return helpers
    
    def collaborate(self, task: Dict[str, Any], helpers: List['CooperativeAgent']) -> Any:
        """Collaborate with helpers on task"""
        print(f"\n[{self.name}] Collaborating on: {task.get('description')}")
        print(f"  Team size: {1 + len(helpers)}")
        
        # Gather contributions
        contributions = []
        
        # Own contribution
        my_contribution = self._contribute(task)
        contributions.append({"agent": self.name, "contribution": my_contribution})
        
        # Helper contributions
        for helper in helpers:
            helper_contribution = helper._contribute(task)
            contributions.append({"agent": helper.name, "contribution": helper_contribution})
        
        # Combine contributions
        result = self._combine_contributions(contributions, task)
        
        self.tasks_completed += 1
        
        # Notify team
        self._notify_completion(task, result, [self] + helpers)
        
        return result
    
    def _contribute(self, task: Dict[str, Any]) -> str:
        """Contribute to task based on expertise"""
        relevant_skills = [s for s in self.expertise if s in task.get('required_skills', [])]
        
        if relevant_skills:
            return f"Contribution using {', '.join(relevant_skills)}"
        else:
            return "General contribution"
    
    def _combine_contributions(self, contributions: List[Dict], task: Dict[str, Any]) -> str:
        """Combine contributions into final result"""
        combined = f"Collaborative result for {task.get('id')}:\n"
        
        for contrib in contributions:
            combined += f"  - {contrib['agent']}: {contrib['contribution']}\n"
        
        return combined
    
    def _notify_completion(self, task: Dict[str, Any], result: str, team: List['CooperativeAgent']):
        """Notify team of completion"""
        message = Message(
            sender_id=self.agent_id,
            receiver_id="broadcast",
            message_type=MessageType.TASK_COMPLETE,
            content={"task_id": task.get('id'), "result": result}
        )
        
        for agent in team:
            if agent.agent_id != self.agent_id:
                agent.receive_message(message)

class CooperativeSystem:
    """System managing cooperative agents"""
    
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.agents: List[CooperativeAgent] = []
        self.shared_workspace: Dict[str, Any] = {}
    
    def add_agent(self, agent: CooperativeAgent):
        """Add agent to system"""
        self.agents.append(agent)
        print(f"Added agent: {agent.name} (expertise: {', '.join(agent.expertise)})")
    
    def execute_task(self, task: Dict[str, Any]):
        """Execute task cooperatively"""
        print(f"\n{'='*70}")
        print(f"COOPERATIVE TASK EXECUTION")
        print(f"{'='*70}")
        print(f"Task: {task.get('description')}")
        print(f"Required skills: {', '.join(task.get('required_skills', []))}")
        
        # Find primary agent (most relevant expertise)
        primary_agent = self._select_primary_agent(task)
        
        if not primary_agent:
            print("\n✗ No suitable agent found")
            return None
        
        print(f"\nPrimary agent: {primary_agent.name}")
        
        # Request help
        helpers = primary_agent.request_help(task, self.agents)
        
        print(f"Helpers: {', '.join(h.name for h in helpers) if helpers else 'None'}")
        
        # Collaborate
        result = primary_agent.collaborate(task, helpers)
        
        print(f"\n{'='*60}")
        print("TASK RESULT")
        print(f"{'='*60}")
        print(result)
        
        return result
    
    def _select_primary_agent(self, task: Dict[str, Any]) -> Optional[CooperativeAgent]:
        """Select most suitable agent for task"""
        required_skills = set(task.get('required_skills', []))
        
        best_agent = None
        best_match = 0
        
        for agent in self.agents:
            agent_skills = set(agent.expertise)
            match_count = len(required_skills & agent_skills)
            
            if match_count > best_match:
                best_match = match_count
                best_agent = agent
        
        return best_agent
    
    def share_knowledge_across_team(self):
        """Share knowledge among all agents"""
        print(f"\n{'='*70}")
        print("KNOWLEDGE SHARING SESSION")
        print(f"{'='*70}\n")
        
        # Each agent shares something
        for agent in self.agents:
            knowledge_item = f"Best practice from {agent.name}"
            agent.share_knowledge(
                key=f"practice_{agent.agent_id}",
                value=knowledge_item,
                team=self.agents
            )
    
    def print_statistics(self):
        """Print cooperation statistics"""
        print(f"\n{'='*70}")
        print("COOPERATION STATISTICS")
        print(f"{'='*70}\n")
        
        print(f"{'Agent':<20} {'Completed':<12} {'Help Given':<12} {'Help Received':<15}")
        print("-" * 70)
        
        for agent in self.agents:
            print(f"{agent.name:<20} {agent.tasks_completed:<12} "
                  f"{agent.help_given:<12} {agent.help_received:<15}")
        
        total_help = sum(a.help_given for a in self.agents)
        total_tasks = sum(a.tasks_completed for a in self.agents)
        
        print(f"\nTotal collaborative actions: {total_help}")
        print(f"Total tasks completed: {total_tasks}")
        print(f"Average help per task: {total_help / max(total_tasks, 1):.1f}")


# Usage
if __name__ == "__main__":
    print("="*80)
    print("COOPERATIVE MULTI-AGENT PATTERN DEMONSTRATION")
    print("="*80)
    
    # Create cooperative system
    system = CooperativeSystem("Development Team")
    
    # Create agents with different expertise
    agents = [
        CooperativeAgent("agent_1", "Alice", ["python", "backend", "databases"]),
        CooperativeAgent("agent_2", "Bob", ["javascript", "frontend", "ui"]),
        CooperativeAgent("agent_3", "Carol", ["devops", "cloud", "deployment"]),
        CooperativeAgent("agent_4", "Dave", ["testing", "quality", "automation"]),
    ]
    
    print("\nBuilding cooperative team...\n")
    for agent in agents:
        system.add_agent(agent)
    
    # Knowledge sharing
    system.share_knowledge_across_team()
    
    # Execute collaborative tasks
    tasks = [
        {
            "id": "T1",
            "description": "Build full-stack web application",
            "required_skills": ["python", "javascript", "databases", "frontend"]
        },
        {
            "id": "T2",
            "description": "Deploy application to cloud",
            "required_skills": ["devops", "cloud", "deployment"]
        },
        {
            "id": "T3",
            "description": "Create automated test suite",
            "required_skills": ["testing", "automation", "python"]
        },
    ]
    
    for task in tasks:
        system.execute_task(task)
        print("\n" + "="*80 + "\n")
    
    # Print statistics
    system.print_statistics()
```

```
42_neuro_symbolic.py
"""
Neuro-Symbolic Integration Pattern
Combines neural (LLM) and symbolic (logic) reasoning
"""

from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum

class SymbolType(Enum):
    ENTITY = "entity"
    RELATION = "relation"
    RULE = "rule"
    FACT = "fact"

@dataclass
class Symbol:
    """Symbolic representation"""
    name: str
    symbol_type: SymbolType
    properties: Dict[str, Any]

@dataclass
class LogicalRule:
    """Logical inference rule"""
    premises: List[str]
    conclusion: str
    confidence: float = 1.0

@dataclass
class Fact:
    """Known fact"""
    subject: str
    predicate: str
    object: str

class SymbolicReasoner:
    """Symbolic reasoning engine"""
    
    def __init__(self):
        self.facts: List[Fact] = []
        self.rules: List[LogicalRule] = []
        self.symbols: Dict[str, Symbol] = {}
    
    def add_fact(self, subject: str, predicate: str, obj: str):
        """Add a fact to knowledge base"""
        fact = Fact(subject, predicate, obj)
        self.facts.append(fact)
        print(f"[Symbolic] Added fact: {subject} {predicate} {obj}")
    
    def add_rule(self, rule: LogicalRule):
        """Add inference rule"""
        self.rules.append(rule)
        print(f"[Symbolic] Added rule: {' AND '.join(rule.premises)} → {rule.conclusion}")
    
    def query(self, subject: str, predicate: str) -> List[str]:
        """Query knowledge base"""
        results = []
        
        # Direct facts
        for fact in self.facts:
            if fact.subject == subject and fact.predicate == predicate:
                results.append(fact.object)
        
        # Inferred facts
        inferred = self._apply_rules()
        for fact in inferred:
            if fact.subject == subject and fact.predicate == predicate:
                if fact.object not in results:
                    results.append(fact.object)
        
        return results
    
    def _apply_rules(self) -> List[Fact]:
        """Apply inference rules"""
        inferred = []
        
        for rule in self.rules:
            # Check if all premises are satisfied
            if self._check_premises(rule.premises):
                # Create inferred fact
                parts = rule.conclusion.split()
                if len(parts) == 3:
                    inferred.append(Fact(parts[0], parts[1], parts[2]))
        
        return inferred
    
    def _check_premises(self, premises: List[str]) -> bool:
        """Check if premises are satisfied"""
        for premise in premises:
            parts = premise.split()
            if len(parts) == 3:
                subject, predicate, obj = parts
                found = any(
                    f.subject == subject and 
                    f.predicate == predicate and 
                    f.object == obj
                    for f in self.facts
                )
                if not found:
                    return False
        return True

class NeuralReasoner:
    """Neural reasoning component (LLM-based)"""
    
    def __init__(self):
        self.context = ""
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract entities from text"""
        # Simulated NER (would use actual LLM)
        print(f"[Neural] Extracting entities from: {text[:50]}...")
        
        # Simple simulation
        words = text.split()
        entities = [w for w in words if w[0].isupper()]
        
        print(f"[Neural] Found entities: {entities}")
        return entities
    
    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract relations from text"""
        # Simulated relation extraction
        print(f"[Neural] Extracting relations from: {text[:50]}...")
        
        relations = []
        
        # Simple pattern matching (would use LLM)
        if "is a" in text.lower():
            parts = text.lower().split("is a")
            if len(parts) == 2:
                subject = parts[0].strip().split()[-1]
                obj = parts[1].strip().split()[0]
                relations.append((subject, "is_a", obj))
        
        if "has" in text.lower():
            parts = text.lower().split("has")
            if len(parts) == 2:
                subject = parts[0].strip().split()[-1]
                obj = parts[1].strip().split()[0]
                relations.append((subject, "has", obj))
        
        print(f"[Neural] Found relations: {relations}")
        return relations
    
    def reason_about(self, question: str, context: str) -> str:
        """Reason about question using neural approach"""
        # Simulated neural reasoning
        print(f"[Neural] Reasoning: {question}")
        
        # Would use LLM here
        if "what" in question.lower():
            return "Neural inference based on patterns in data"
        elif "why" in question.lower():
            return "Neural explanation based on learned associations"
        else:
            return "Neural response"

class NeuroSymbolicAgent:
    """Agent combining neural and symbolic reasoning"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.neural = NeuralReasoner()
        self.symbolic = SymbolicReasoner()
    
    def learn_from_text(self, text: str):
        """Extract knowledge from text using neural, store symbolically"""
        print(f"\n{'='*70}")
        print("LEARNING FROM TEXT")
        print(f"{'='*70}")
        print(f"Text: {text}\n")
        
        # Neural extraction
        entities = self.neural.extract_entities(text)
        relations = self.neural.extract_relations(text)
        
        # Store as symbolic facts
        print(f"\n[Integration] Converting to symbolic facts...")
        for subject, predicate, obj in relations:
            self.symbolic.add_fact(subject, predicate, obj)
    
    def hybrid_reasoning(self, question: str) -> Dict[str, Any]:
        """Answer question using both neural and symbolic reasoning"""
        print(f"\n{'='*70}")
        print("HYBRID REASONING")
        print(f"{'='*70}")
        print(f"Question: {question}\n")
        
        # Try symbolic reasoning first
        print("Step 1: Symbolic Reasoning")
        print("-" * 60)
        
        symbolic_answer = self._symbolic_reasoning(question)
        
        # If symbolic fails or needs enrichment, use neural
        print("\nStep 2: Neural Reasoning")
        print("-" * 60)
        
        neural_answer = self.neural.reason_about(question, "")
        
        # Combine results
        print("\nStep 3: Integration")
        print("-" * 60)
        
        combined = self._integrate_answers(symbolic_answer, neural_answer)
        
        return {
            'question': question,
            'symbolic_answer': symbolic_answer,
            'neural_answer': neural_answer,
            'combined_answer': combined
        }
    
    def _symbolic_reasoning(self, question: str) -> Optional[str]:
        """Apply symbolic reasoning"""
        # Parse question
        words = question.lower().split()
        
        if "is" in words and len(words) >= 3:
            # Query like "What is X?"
            subject = words[words.index("is") + 1] if words.index("is") + 1 < len(words) else None
            
            if subject:
                results = self.symbolic.query(subject, "is_a")
                if results:
                    answer = f"{subject} is a {results[0]}"
                    print(f"[Symbolic] Deduced: {answer}")
                    return answer
        
        print(f"[Symbolic] No definitive answer from logical rules")
        return None
    
    def _integrate_answers(self, symbolic: Optional[str], neural: str) -> str:
        """Integrate symbolic and neural answers"""
        if symbolic:
            # Symbolic reasoning succeeded
            integrated = f"Based on logical deduction: {symbolic}"
            
            # Enrich with neural insights
            if neural and "neural" in neural.lower():
                integrated += f"\nAdditional context: {neural}"
            
            print(f"[Integration] Using symbolic answer, enriched with neural context")
        else:
            # Fall back to neural
            integrated = f"Based on learned patterns: {neural}"
            print(f"[Integration] Using neural answer (no symbolic solution)")
        
        return integrated
    
    def add_logical_rules(self):
        """Add logical inference rules"""
        print(f"\n{'='*70}")
        print("ADDING LOGICAL RULES")
        print(f"{'='*70}\n")
        
        # Transitivity rule
        self.symbolic.add_rule(LogicalRule(
            premises=["X is_a Y", "Y is_a Z"],
            conclusion="X is_a Z",
            confidence=1.0
        ))
        
        # Inheritance rule
        self.symbolic.add_rule(LogicalRule(
            premises=["X is_a Y", "Y has Z"],
            conclusion="X has Z",
            confidence=0.9
        ))


# Usage
if __name__ == "__main__":
    print("="*80)
    print("NEURO-SYMBOLIC INTEGRATION PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = NeuroSymbolicAgent("neuro-symbolic-001")
    
    # Learn from text (neural extraction → symbolic storage)
    texts = [
        "A dog is a mammal",
        "A mammal is an animal",
        "Mammals have fur",
        "A cat is a mammal",
    ]
    
    for text in texts:
        agent.learn_from_text(text)
    
    # Add logical rules
    agent.add_logical_rules()
    
    # Test hybrid reasoning
    questions = [
        "What is a dog?",
        "What is a cat?",
        "Why do dogs have characteristics?",
    ]
    
    for question in questions:
        result = agent.hybrid_reasoning(question)
        
        print(f"\n{'='*70}")
        print("FINAL ANSWER")
        print(f"{'='*70}")
        print(result['combined_answer'])
        print("\n" + "="*80 + "\n")
```


```
43_rate_limiting.py

Now let me create the final remaining high-priority patterns:

```python patterns/43_rate_limiting.py
"""
Rate Limiting & Throttling Pattern
Controls frequency of agent actions
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import time

class RateLimitStrategy(Enum):
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    max_requests: int
    window_seconds: int
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_size: Optional[int] = None  # For token bucket

@dataclass
class RequestRecord:
    """Record of a request"""
    timestamp: datetime
    user_id: str
    resource: str
    allowed: bool

class RateLimiter:
    """Rate limiter implementation"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests: Dict[str, deque] = {}  # user_id -> request times
        self.tokens: Dict[str, float] = {}  # For token bucket
        self.last_refill: Dict[str, datetime] = {}
        self.request_history: List[RequestRecord] = []
    
    def is_allowed(self, user_id: str, resource: str = "default") -> bool:
        """Check if request is allowed"""
        key = f"{user_id}:{resource}"
        current_time = datetime.now()
        
        if self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            allowed = self._sliding_window_check(key, current_time)
        elif self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            allowed = self._token_bucket_check(key, current_time)
        elif self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
            allowed = self._fixed_window_check(key, current_time)
        else:
            allowed = True
        
        # Record request
        self.request_history.append(RequestRecord(
            timestamp=current_time,
            user_id=user_id,
            resource=resource,
            allowed=allowed
        ))
        
        return allowed
    
    def _sliding_window_check(self, key: str, current_time: datetime) -> bool:
        """Sliding window rate limiting"""
        if key not in self.requests:
            self.requests[key] = deque()
        
        # Remove old requests outside window
        window_start = current_time - timedelta(seconds=self.config.window_seconds)
        
        while self.requests[key] and self.requests[key][0] < window_start:
            self.requests[key].popleft()
        
        # Check if under limit
        if len(self.requests[key]) < self.config.max_requests:
            self.requests[key].append(current_time)
            return True
        
        return False
    
    def _token_bucket_check(self, key: str, current_time: datetime) -> bool:
        """Token bucket rate limiting"""
        if key not in self.tokens:
            self.tokens[key] = self.config.burst_size or self.config.max_requests
            self.last_refill[key] = current_time
        
        # Refill tokens
        time_passed = (current_time - self.last_refill[key]).total_seconds()
        refill_rate = self.config.max_requests / self.config.window_seconds
        tokens_to_add = time_passed * refill_rate
        
        max_tokens = self.config.burst_size or self.config.max_requests
        self.tokens[key] = min(max_tokens, self.tokens[key] + tokens_to_add)
        self.last_refill[key] = current_time
        
        # Try to consume token
        if self.tokens[key] >= 1.0:
            self.tokens[key] -= 1.0
            return True
        
        return False
    
    def _fixed_window_check(self, key: str, current_time: datetime) -> bool:
        """Fixed window rate limiting"""
        if key not in self.requests:
            self.requests[key] = deque()
        
        # Calculate current window
        window_start = datetime(
            current_time.year,
            current_time.month,
            current_time.day,
            current_time.hour,
            current_time.minute // (self.config.window_seconds // 60),
            0
        )
        
        # Remove requests from previous windows
        self.requests[key] = deque([
            t for t in self.requests[key] if t >= window_start
        ])
        
        # Check limit
        if len(self.requests[key]) < self.config.max_requests:
            self.requests[key].append(current_time)
            return True
        
        return False
    
    def get_remaining(self, user_id: str, resource: str = "default") -> int:
        """Get remaining requests for user"""
        key = f"{user_id}:{resource}"
        
        if key not in self.requests:
            return self.config.max_requests
        
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return int(self.tokens.get(key, 0))
        else:
            return self.config.max_requests - len(self.requests[key])
    
    def reset(self, user_id: str, resource: str = "default"):
        """Reset rate limit for user"""
        key = f"{user_id}:{resource}"
        
        if key in self.requests:
            self.requests[key].clear()
        if key in self.tokens:
            self.tokens[key] = self.config.max_requests

class RateLimitedAgent:
    """Agent with rate limiting"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Different rate limits for different operations
        self.limiters = {
            'query': RateLimiter(RateLimitConfig(
                max_requests=10,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW
            )),
            'expensive_operation': RateLimiter(RateLimitConfig(
                max_requests=3,
                window_seconds=60,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                burst_size=5
            ))
        }
    
    def process_request(self, user_id: str, operation: str, request: str) -> Dict[str, Any]:
        """Process request with rate limiting"""
        print(f"\n[{self.agent_id}] Processing {operation} for user {user_id}")
        
        limiter = self.limiters.get(operation, self.limiters['query'])
        
        # Check rate limit
        if not limiter.is_allowed(user_id, operation):
            remaining = limiter.get_remaining(user_id, operation)
            
            print(f"  ✗ Rate limit exceeded")
            print(f"  Remaining: {remaining}")
            
            return {
                'success': False,
                'error': 'Rate limit exceeded',
                'retry_after_seconds': limiter.config.window_seconds,
                'remaining': remaining
            }
        
        # Process request
        remaining = limiter.get_remaining(user_id, operation)
        
        print(f"  ✓ Request allowed")
        print(f"  Remaining: {remaining}")
        
        result = self._execute_operation(operation, request)
        
        return {
            'success': True,
            'result': result,
            'remaining': remaining
        }
    
    def _execute_operation(self, operation: str, request: str) -> Any:
        """Execute the operation"""
        import time
        time.sleep(0.1)  # Simulate work
        return f"Result for {operation}: {request}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        stats = {}
        
        for name, limiter in self.limiters.items():
            total = len(limiter.request_history)
            allowed = sum(1 for r in limiter.request_history if r.allowed)
            rejected = total - allowed
            
            stats[name] = {
                'total_requests': total,
                'allowed': allowed,
                'rejected': rejected,
                'rejection_rate': rejected / total if total > 0 else 0
            }
        
        return stats


# Usage
if __name__ == "__main__":
    print("="*80)
    print("RATE LIMITING & THROTTLING PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = RateLimitedAgent("rate-limited-agent")
    
    # Simulate requests from user
    user_id = "user_123"
    
    print("\nSimulating burst of requests...")
    
    # Test query rate limit (10 per minute)
    for i in range(15):
        result = agent.process_request(user_id, 'query', f"Query {i+1}")
        
        if not result['success']:
            print(f"  Request {i+1}: BLOCKED - {result['error']}")
        
        time.sleep(0.1)
    
    print("\n" + "="*80)
    print("\nTesting expensive operation limit (3 per minute with burst)...")
    
    # Test expensive operation limit
    for i in range(8):
        result = agent.process_request(user_id, 'expensive_operation', f"Expensive {i+1}")
        time.sleep(0.2)
    
    # Statistics
    print("\n" + "="*80)
    print("RATE LIMITING STATISTICS")
    print("="*80)
    
    stats = agent.get_statistics()
    
    for operation, stat in stats.items():
        print(f"\n{operation}:")
        print(f"  Total Requests: {stat['total_requests']}")
        print(f"  Allowed: {stat['allowed']}")
        print(f"  Rejected: {stat['rejected']}")
        print(f"  Rejection Rate: {stat['rejection_rate']:.1%}")
```


```
44_token_budget_management.py
"""
Token Budget Management Pattern
Manages token usage within limits
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time

class TokenStrategy(Enum):
    TRUNCATE = "truncate"
    SUMMARIZE = "summarize"
    COMPRESS = "compress"
    PRIORITIZE = "prioritize"

@dataclass
class TokenUsage:
    """Token usage record"""
    operation: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    timestamp: float

class TokenBudgetManager:
    """Manages token budget and optimization"""
    
    def __init__(self, max_tokens_per_request: int = 4000, daily_budget_usd: float = 10.0):
        self.max_tokens_per_request = max_tokens_per_request
        self.daily_budget_usd = daily_budget_usd
        self.usage_history: List[TokenUsage] = []
        
        # Pricing (example rates)
        self.input_price_per_1k = 0.01  # $0.01 per 1K input tokens
        self.output_price_per_1k = 0.03  # $0.03 per 1K output tokens
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Real implementation would use tiktoken or similar
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def check_budget(self, estimated_tokens: int) -> Dict[str, Any]:
        """Check if operation is within budget"""
        # Check per-request limit
        if estimated_tokens > self.max_tokens_per_request:
            return {
                'allowed': False,
                'reason': 'exceeds_per_request_limit',
                'limit': self.max_tokens_per_request,
                'estimated': estimated_tokens
            }
        
        # Check daily budget
        today_cost = self._get_today_cost()
        estimated_cost = self._estimate_cost(estimated_tokens, estimated_tokens)
        
        if today_cost + estimated_cost > self.daily_budget_usd:
            return {
                'allowed': False,
                'reason': 'exceeds_daily_budget',
                'budget': self.daily_budget_usd,
                'spent': today_cost,
                'estimated_cost': estimated_cost
            }
        
        return {
            'allowed': True,
            'estimated_tokens': estimated_tokens,
            'estimated_cost': estimated_cost,
            'remaining_budget': self.daily_budget_usd - today_cost
        }
    
    def optimize_input(self, text: str, strategy: TokenStrategy = TokenStrategy.TRUNCATE) -> str:
        """Optimize input to fit within budget"""
        current_tokens = self.estimate_tokens(text)
        
        if current_tokens <= self.max_tokens_per_request:
            return text
        
        print(f"\n[TokenManager] Input too large ({current_tokens} tokens), optimizing...")
        
        if strategy == TokenStrategy.TRUNCATE:
            return self._truncate(text)
        elif strategy == TokenStrategy.SUMMARIZE:
            return self._summarize(text)
        elif strategy == TokenStrategy.COMPRESS:
            return self._compress(text)
        elif strategy == TokenStrategy.PRIORITIZE:
            return self._prioritize(text)
        
        return text
    
    def _truncate(self, text: str) -> str:
        """Truncate text to fit budget"""
        target_chars = self.max_tokens_per_request * 4  # Rough estimate
        
        if len(text) <= target_chars:
            return text
        
        truncated = text[:target_chars] + "..."
        print(f"  Strategy: TRUNCATE ({self.estimate_tokens(truncated)} tokens)")
        
        return truncated
    
    def _summarize(self, text: str) -> str:
        """Summarize text to reduce tokens"""
        # Simplified summarization (would use actual LLM)
        sentences = text.split('.')
        summary = '. '.join(sentences[:len(sentences)//2]) + "..."
        
        print(f"  Strategy: SUMMARIZE ({self.estimate_tokens(summary)} tokens)")
        
        return summary
    
    def _compress(self, text: str) -> str:
        """Compress text by removing redundancy"""
        # Simple compression: remove extra whitespace
        compressed = ' '.join(text.split())
        
        print(f"  Strategy: COMPRESS ({self.estimate_tokens(compressed)} tokens)")
        
        return compressed
    
    def _prioritize(self, text: str) -> str:
        """Keep most important parts"""
        # Keep first and last parts (introduction and conclusion)
        paragraphs = text.split('\n\n')
        
        if len(paragraphs) <= 2:
            return text
        
        prioritized = paragraphs[0] + '\n\n...\n\n' + paragraphs[-1]
        
        print(f"  Strategy: PRIORITIZE ({self.estimate_tokens(prioritized)} tokens)")
        
        return prioritized
    
    def record_usage(self, operation: str, input_tokens: int, output_tokens: int):
        """Record token usage"""
        cost = self._estimate_cost(input_tokens, output_tokens)
        
        usage = TokenUsage(
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            timestamp=time.time()
        )
        
        self.usage_history.append(usage)
        
        print(f"\n[TokenManager] Recorded usage:")
        print(f"  Input: {input_tokens} tokens")
        print(f"  Output: {output_tokens} tokens")
        print(f"  Cost: ${cost:.4f}")
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost of operation"""
        input_cost = (input_tokens / 1000) * self.input_price_per_1k
        output_cost = (output_tokens / 1000) * self.output_price_per_1k
        
        return input_cost + output_cost
    
    def _get_today_cost(self) -> float:
        """Get today's total cost"""
        import time
        
        # Get today's start timestamp
        today_start = time.time() - (time.time() % 86400)  # Start of day
        
        today_usage = [
            u for u in self.usage_history
            if u.timestamp >= today_start
        ]
        
        return sum(u.cost_usd for u in today_usage)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        if not self.usage_history:
            return {'total_operations': 0}
        
        total_input = sum(u.input_tokens for u in self.usage_history)
        total_output = sum(u.output_tokens for u in self.usage_history)
        total_cost = sum(u.cost_usd for u in self.usage_history)
        
        today_cost = self._get_today_cost()
        
        return {
            'total_operations': len(self.usage_history),
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'total_cost_usd': total_cost,
            'today_cost_usd': today_cost,
            'budget_remaining_usd': self.daily_budget_usd - today_cost,
            'avg_tokens_per_operation': (total_input + total_output) / len(self.usage_history)
        }

class BudgetAwareAgent:
    """Agent with token budget management"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.budget_manager = TokenBudgetManager(
            max_tokens_per_request=2000,
            daily_budget_usd=5.0
        )
    
    def process(self, input_text: str, strategy: TokenStrategy = TokenStrategy.TRUNCATE) -> Dict[str, Any]:
        """Process input with budget management"""
        print(f"\n{'='*70}")
        print(f"PROCESSING WITH BUDGET MANAGEMENT")
        print(f"{'='*70}")
        
        # Estimate tokens
        estimated_tokens = self.budget_manager.estimate_tokens(input_text)
        print(f"\nInput length: {len(input_text)} chars")
        print(f"Estimated tokens: {estimated_tokens}")
        
        # Check budget
        budget_check = self.budget_manager.check_budget(estimated_tokens)
        
        if not budget_check['allowed']:
            print(f"\n✗ Budget check failed: {budget_check['reason']}")
            return {
                'success': False,
                'error': budget_check['reason'],
                'details': budget_check
            }
        
        print(f"\n✓ Budget check passed")
        print(f"  Estimated cost: ${budget_check['estimated_cost']:.4f}")
        print(f"  Remaining budget: ${budget_check['remaining_budget']:.2f}")
        
        # Optimize if needed
        optimized_input = self.budget_manager.optimize_input(input_text, strategy)
        
        # Simulate processing
        output = self._generate_output(optimized_input)
        
        # Record usage
        input_tokens = self.budget_manager.estimate_tokens(optimized_input)
        output_tokens = self.budget_manager.estimate_tokens(output)
        
        self.budget_manager.record_usage("process", input_tokens, output_tokens)
        
        return {
            'success': True,
            'output': output,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }
    
    def _generate_output(self, input_text: str) -> str:
        """Generate output (simulated)"""
        # Simulate LLM generation
        return f"Generated response based on: {input_text[:50]}..."


# Usage
if __name__ == "__main__":
    print("="*80)
    print("TOKEN BUDGET MANAGEMENT PATTERN DEMONSTRATION")
    print("="*80)
    
    agent = BudgetAwareAgent("budget-agent-001")
    
    # Test 1: Normal input
    print("\nTest 1: Normal Size Input")
    print("-" * 70)
    
    normal_input = "This is a normal size input that should fit within budget. " * 10
    
    result1 = agent.process(normal_input)
    
    # Test 2: Large input requiring optimization
    print("\n\n" + "="*80)
    print("Test 2: Large Input Requiring Optimization")
    print("-" * 70)
    
    large_input = "This is a very long input text that will exceed the token budget. " * 100
    
    result2 = agent.process(large_input, strategy=TokenStrategy.SUMMARIZE)
    
    # Test 3: Multiple operations to test daily budget
    print("\n\n" + "="*80)
    print("Test 3: Multiple Operations")
    print("-" * 70)
    
    for i in range(5):
        input_text = f"Operation {i+1}: " + ("Some text. " * 50)
        agent.process(input_text, strategy=TokenStrategy.COMPRESS)
    
    # Statistics
    stats = agent.budget_manager.get_statistics()
    
    print("\n" + "="*80)
    print("BUDGET STATISTICS")
    print("="*80)
    print(f"Total Operations: {stats['total_operations']}")
    print(f"Total Input Tokens: {stats['total_input_tokens']:,}")
    print(f"Total Output Tokens: {stats['total_output_tokens']:,}")
    print(f"Total Cost: ${stats['total_cost_usd']:.4f}")
    print(f"Today's Cost: ${stats['today_cost_usd']:.4f}")
    print(f"Remaining Budget: ${stats['budget_remaining_usd']:.2f}")
    print(f"Avg Tokens/Operation: {stats['avg_tokens_per_operation']:.0f}")
```
