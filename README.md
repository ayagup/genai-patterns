# Agentic AI Design Patterns - Python Examples

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive collection of **170+ Agentic AI Design Patterns** with practical Python implementations. This repository demonstrates the fundamental patterns used in building intelligent AI agents, from basic reasoning to complex multi-agent systems.

## 📚 Documentation

For a complete catalog of all 170 patterns, see [`agentic_ai_design_patterns.md`](./agentic_ai_design_patterns.md).

## 🎯 What's Included

This repository contains **9 complete Python examples** demonstrating the most important agentic AI patterns:

### 1. **ReAct Pattern** (`01_react_pattern.py`)
- Reasoning + Acting loop
- Tool selection and usage
- Thought → Action → Observation cycle
- **Use Cases**: Question answering, task automation

### 2. **Chain-of-Thought** (`02_chain_of_thought.py`)
- Step-by-step reasoning
- Zero-shot and Few-shot variants
- Problem decomposition
- **Use Cases**: Math problems, logic puzzles, analysis

### 3. **Tree-of-Thoughts** (`03_tree_of_thoughts.py`)
- Multiple reasoning paths exploration
- BFS and DFS search strategies
- Thought evaluation and pruning
- **Use Cases**: Strategic planning, optimization

### 4. **Plan-and-Execute** (`04_plan_and_execute.py`)
- Separate planning and execution phases
- Task dependency management
- Replanning on failures
- **Use Cases**: Complex workflows, project management

### 5. **Self-Consistency** (`05_self_consistency.py`)
- Multiple reasoning paths
- Majority voting aggregation
- Confidence scoring
- **Use Cases**: Improving accuracy, reducing errors

### 6. **Reflexion** (`06_reflexion.py`)
- Learning from mistakes
- Self-reflection and memory
- Iterative improvement
- **Use Cases**: Code generation, skill learning

### 7. **Multi-Agent Patterns** (`07_multi_agent_patterns.py`)
- **Debate**: Adversarial reasoning
- **Ensemble**: Independent classification + aggregation
- **Cooperative**: Shared problem-solving
- **Use Cases**: Complex decisions, robust predictions

### 8. **RAG and Memory** (`08_rag_and_memory.py`)
- Retrieval-Augmented Generation
- Vector search and similarity
- Short-term and long-term memory
- **Use Cases**: Knowledge-intensive tasks, personalization

### 9. **Safety and Control** (`09_safety_and_control.py`)
- **Guardrails**: Input/output filtering
- **Circuit Breaker**: Failure protection
- **Human-in-the-Loop**: Critical decision approval
- **Use Cases**: Production systems, safety-critical applications

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher required
python --version

# No external dependencies needed! All examples use only Python stdlib
```

### Running Examples

Each example is self-contained and can be run directly:

```bash
# Run ReAct pattern example
python 01_react_pattern.py

# Run Chain-of-Thought example
python 02_chain_of_thought.py

# Run Tree-of-Thoughts example
python 03_tree_of_thoughts.py

# ... and so on
```

### Example Output

```
============================================================
Task: Please calculate 25 * 4 + 10
============================================================

Iteration 1:
----------------------------------------
💭 Thought: I need to perform a calculation to answer this question.
📋 Action Plan: use calculator
👀 Observation: Result: 110

✅ Final Answer: Result: 110
```

## 📖 Pattern Categories

The complete collection includes patterns across these categories:

### Core Architectural (1-5)
- ReAct, Chain-of-Thought, Tree-of-Thoughts, Graph-of-Thoughts, Plan-and-Execute

### Reasoning & Planning (6-11)
- Hierarchical Planning, Reflexion, Self-Consistency, Least-to-Most, Analogical Reasoning

### Multi-Agent (12-19)
- Debate, Ensemble, Leader-Follower, Swarm Intelligence, Society of Mind

### Tool Use & Action (20-25)
- Tool Selection, Function Calling, Code Execution, RAG, Iterative Refinement

### Memory & State (26-32)
- Short/Long-term Memory, Working Memory, Semantic Networks, State Machines

### Interaction & Control (33-39)
- Human-in-the-Loop, Constitutional AI, Guardrails, Prompt Chaining

### Evaluation & Optimization (40-44)
- Self-Evaluation, Chain-of-Verification, Progressive Optimization

### Safety & Reliability (45-52)
- Defensive Generation, Circuit Breaker, Sandboxing, Monitoring

### Advanced Patterns (53-170)
- Cognitive Architecture, Neuro-Symbolic Integration, Knowledge Graphs, and many more!

## 💡 Usage Examples

### Basic Pattern Usage

```python
from react_pattern import ReActAgent, Calculator, SearchTool

# Create tools
tools = [Calculator(), SearchTool()]

# Create agent
agent = ReActAgent(tools=tools)

# Execute task
result = agent.run("Calculate 25 * 4 + 10")
```

### Multi-Agent Collaboration

```python
from multi_agent_patterns import CooperativeSystem, CooperativeAgent, Agent

# Create system
system = CooperativeSystem("Design a new mobile app")

# Add agents with different expertise
agents = [
    CooperativeAgent(Agent("Sarah", "Market Research")),
    CooperativeAgent(Agent("Mike", "Engineering")),
    CooperativeAgent(Agent("Lisa", "Finance"))
]

for agent in agents:
    system.add_agent(agent)

# Solve collaboratively
result = system.solve_collaboratively()
```

### RAG Implementation

```python
from rag_and_memory import RAGAgent, VectorStore, Document

# Create knowledge base
vector_store = VectorStore()
vector_store.add_document(Document(
    id="1",
    content="Python is a high-level programming language...",
    metadata={"title": "Python Intro"}
))

# Create RAG agent
agent = RAGAgent(vector_store)

# Ask question with retrieval
result = agent.answer_question("What is Python?")
```

## 🎓 Learning Path

**Beginner**: Start with these patterns
1. ReAct (01) - Basic agent architecture
2. Chain-of-Thought (02) - Step-by-step reasoning
3. Memory patterns (08) - State management

**Intermediate**: Move to these
4. Tree-of-Thoughts (03) - Multiple reasoning paths
5. Plan-and-Execute (04) - Complex task breakdown
6. Self-Consistency (05) - Improved reliability

**Advanced**: Explore these
7. Reflexion (06) - Self-improvement
8. Multi-Agent (07) - Collaborative systems
9. Safety patterns (09) - Production readiness

## 🏗️ Code Structure

Each example follows a consistent structure:

```
Pattern Name
├── Core Classes (Agent implementation)
├── Helper Classes (Tools, evaluators, etc.)
├── Demonstration Functions
└── Main function with multiple examples
```

All code includes:
- ✅ Type hints for clarity
- ✅ Detailed docstrings
- ✅ Console output for visualization
- ✅ Multiple usage examples
- ✅ No external dependencies

## 🔬 Pattern Selection Guide

| Task Type | Recommended Patterns |
|-----------|---------------------|
| Question Answering | ReAct + RAG |
| Math Problems | Chain-of-Thought + Self-Consistency |
| Strategic Planning | Tree-of-Thoughts + Plan-and-Execute |
| Code Generation | Reflexion + Self-Evaluation |
| Complex Decisions | Multi-Agent Debate + Ensemble |
| Safety-Critical | Guardrails + HITL + Circuit Breaker |
| Personalization | Memory Management + Learning |

## 🛠️ Extending the Examples

### Adding a New Tool

```python
class MyCustomTool(Tool):
    def __init__(self):
        super().__init__("my_tool", "Description of what it does")
    
    def execute(self, **kwargs) -> str:
        # Implement tool logic
        return "Tool result"
```

### Creating a Custom Agent

```python
class MyAgent:
    def __init__(self):
        self.memory = []
    
    def process(self, input_data):
        # Implement agent logic
        return result
```

## 📊 Performance Considerations

- **ReAct**: Fast for simple tasks, can be slow with many tool calls
- **Tree-of-Thoughts**: Exponential complexity, use pruning
- **Self-Consistency**: Multiple calls increase latency and cost
- **Multi-Agent**: Overhead from agent coordination
- **RAG**: Retrieval speed depends on vector store size

## 🤝 Contributing

Contributions are welcome! Areas for contribution:
- Additional pattern implementations
- Performance optimizations
- Real-world use case examples
- Integration with actual LLMs
- Documentation improvements

## 📄 License

MIT License - feel free to use these patterns in your projects!

## 🔗 Related Resources

- **Full Pattern Catalog**: [`agentic_ai_design_patterns.md`](./agentic_ai_design_patterns.md)
- **LangChain**: Production framework for many patterns
- **LlamaIndex**: RAG and data-centric patterns
- **Research Papers**: 
  - ReAct: [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)
  - Tree-of-Thoughts: [arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601)
  - Reflexion: [arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)

## 🎯 Next Steps

1. **Run all examples** to see patterns in action
2. **Read the pattern catalog** for comprehensive understanding
3. **Combine patterns** for more sophisticated agents
4. **Integrate with LLMs** (OpenAI, Anthropic, etc.) for production use
5. **Build your own agent** using these patterns as building blocks

## 📞 Questions?

- Create an issue for bug reports
- Start a discussion for questions
- Check the pattern catalog for detailed explanations

---

**Happy Building! 🚀**

*Remember: These patterns are building blocks. The real power comes from combining them creatively to solve your specific problems.*
