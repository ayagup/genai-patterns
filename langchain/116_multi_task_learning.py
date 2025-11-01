"""
Pattern 116: Multi-Task Learning

Description:
    Agent learns multiple related tasks simultaneously using shared representations
    with task-specific heads for specialization.

Components:
    - Shared representations
    - Task-specific heads
    - Multi-task optimization

Use Cases:
    - Related task sets
    - Efficient training
    - Knowledge sharing

LangChain Implementation:
    Uses multi-task prompt templates and shared context for learning across tasks.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class MultiTaskLearningAgent:
    """Agent that learns multiple related tasks simultaneously."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.tasks = {}
        self.shared_knowledge = {}
        self.task_performance = {}
        
    def register_task(self, task_name: str, task_description: str, examples: List[Dict[str, str]]):
        """Register a new task for multi-task learning."""
        self.tasks[task_name] = {
            "description": task_description,
            "examples": examples
        }
        self.task_performance[task_name] = []
        
    def learn_from_shared_representations(self):
        """Extract shared knowledge from all registered tasks."""
        shared_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze these tasks and identify shared patterns, concepts, and skills that apply across them."),
            ("user", "Tasks:\n{tasks}\n\nIdentify shared knowledge:")
        ])
        
        tasks_description = "\n\n".join([
            f"{name}: {info['description']}\nExamples: {info['examples']}"
            for name, info in self.tasks.items()
        ])
        
        chain = shared_prompt | self.llm | StrOutputParser()
        self.shared_knowledge["patterns"] = chain.invoke({"tasks": tasks_description})
        
        print("Shared Knowledge Extracted:")
        print(self.shared_knowledge["patterns"])
        print()
        
    def execute_task(self, task_name: str, input_data: str) -> str:
        """Execute a specific task using shared knowledge and task-specific head."""
        if task_name not in self.tasks:
            return f"Task '{task_name}' not registered"
        
        task_info = self.tasks[task_name]
        
        # Create task-specific prompt with shared knowledge
        task_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a multi-task learning agent.

Shared Knowledge (applies to all tasks):
{shared_knowledge}

Current Task: {task_name}
Description: {task_description}

Examples:
{examples}

Use both shared knowledge and task-specific patterns."""),
            ("user", "{input}")
        ])
        
        examples_str = "\n".join([
            f"Input: {ex['input']}\nOutput: {ex['output']}"
            for ex in task_info['examples']
        ])
        
        chain = task_prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({
            "shared_knowledge": self.shared_knowledge.get("patterns", "None extracted yet"),
            "task_name": task_name,
            "task_description": task_info['description'],
            "examples": examples_str,
            "input": input_data
        })
        
        return result
    
    def evaluate_task(self, task_name: str, test_cases: List[Dict[str, str]]):
        """Evaluate performance on a specific task."""
        if task_name not in self.tasks:
            print(f"Task '{task_name}' not registered")
            return
        
        print(f"\n=== Evaluating Task: {task_name} ===")
        correct = 0
        total = len(test_cases)
        
        for case in test_cases:
            result = self.execute_task(task_name, case['input'])
            is_correct = case['expected'].lower() in result.lower()
            if is_correct:
                correct += 1
            print(f"Input: {case['input']}")
            print(f"Expected: {case['expected']}")
            print(f"Got: {result}")
            print(f"✓" if is_correct else "✗")
            print()
        
        accuracy = correct / total if total > 0 else 0
        self.task_performance[task_name].append(accuracy)
        print(f"Accuracy: {accuracy:.2%}\n")
        
    def get_task_correlations(self) -> str:
        """Analyze how tasks benefit from shared learning."""
        correlation_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze how these tasks benefit from shared multi-task learning."),
            ("user", """Tasks and Performance:
{task_info}

Shared Knowledge:
{shared_knowledge}

Explain how tasks benefit from each other and which skills transfer across tasks.""")
        ])
        
        task_info = "\n".join([
            f"{name}: {self.tasks[name]['description']} - Performance: {self.task_performance[name]}"
            for name in self.tasks
        ])
        
        chain = correlation_prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "task_info": task_info,
            "shared_knowledge": self.shared_knowledge.get("patterns", "Not extracted")
        })


def demonstrate_multi_task_learning():
    """Demonstrate multi-task learning pattern."""
    print("=== Multi-Task Learning Pattern ===\n")
    
    agent = MultiTaskLearningAgent()
    
    # Register related NLP tasks
    print("1. Registering Multiple Related Tasks")
    print("-" * 50)
    
    # Task 1: Sentiment Analysis
    agent.register_task(
        "sentiment_analysis",
        "Classify the sentiment of text as positive, negative, or neutral",
        [
            {"input": "I love this product!", "output": "positive"},
            {"input": "This is terrible.", "output": "negative"},
            {"input": "It's okay, nothing special.", "output": "neutral"}
        ]
    )
    
    # Task 2: Text Classification
    agent.register_task(
        "text_classification",
        "Classify text into categories: technology, sports, politics, entertainment",
        [
            {"input": "New AI breakthrough announced", "output": "technology"},
            {"input": "Championship game tonight", "output": "sports"},
            {"input": "Election results are in", "output": "politics"}
        ]
    )
    
    # Task 3: Intent Detection
    agent.register_task(
        "intent_detection",
        "Detect user intent: question, command, statement, request",
        [
            {"input": "What time is it?", "output": "question"},
            {"input": "Please send the report.", "output": "command"},
            {"input": "I think it will rain.", "output": "statement"}
        ]
    )
    
    print("✓ Registered 3 related NLP tasks\n")
    
    # Extract shared knowledge
    print("2. Learning Shared Representations")
    print("-" * 50)
    agent.learn_from_shared_representations()
    
    # Test on sentiment analysis
    print("3. Testing on Sentiment Analysis Task")
    print("-" * 50)
    test_cases = [
        {"input": "This is amazing and wonderful!", "expected": "positive"},
        {"input": "Worst experience ever", "expected": "negative"},
    ]
    agent.evaluate_task("sentiment_analysis", test_cases)
    
    # Test on text classification
    print("4. Testing on Text Classification Task")
    print("-" * 50)
    test_cases = [
        {"input": "Latest smartphone release announced", "expected": "technology"},
        {"input": "Team wins the finals", "expected": "sports"},
    ]
    agent.evaluate_task("text_classification", test_cases)
    
    # Analyze task correlations
    print("5. Analyzing Task Correlations and Knowledge Transfer")
    print("-" * 50)
    correlations = agent.get_task_correlations()
    print(correlations)
    print()
    
    print("=== Summary ===")
    print("Multi-task learning demonstrated with:")
    print("- Multiple related NLP tasks")
    print("- Shared knowledge extraction")
    print("- Task-specific execution with shared representations")
    print("- Knowledge transfer analysis")


if __name__ == "__main__":
    demonstrate_multi_task_learning()
