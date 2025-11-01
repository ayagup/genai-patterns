"""
Pattern 114: Online Learning

Description:
    Online Learning enables agents to continuously learn and adapt from incoming
    data streams in real-time. Unlike batch learning, online learning updates
    the agent's knowledge incrementally as new data arrives, allowing adaptation
    to changing environments and concept drift.
    
    This pattern is crucial for agents operating in dynamic environments where
    patterns, user preferences, or optimal strategies change over time. It
    combines incremental updates, drift detection, and continuous adaptation.

Key Components:
    1. Stream Processor: Handles incoming data streams
    2. Incremental Updater: Updates knowledge incrementally
    3. Drift Detector: Identifies when patterns change
    4. Performance Monitor: Tracks learning effectiveness
    5. Model Adapter: Adjusts model based on feedback
    6. Forgetting Mechanism: Manages outdated knowledge

Learning Strategies:
    - Incremental Learning: Update with each new example
    - Mini-Batch: Update with small batches
    - Windowed: Learn from recent window
    - Adaptive Rate: Adjust learning rate dynamically
    - Ensemble: Maintain multiple learners

Use Cases:
    - Real-time recommendation systems
    - Adaptive chatbots learning from conversations
    - Fraud detection with evolving patterns
    - Personalization engines
    - Market prediction systems

LangChain Implementation:
    Uses conversation memory, feedback integration, and dynamic prompt
    adjustment to enable continuous learning from interactions.
"""

import os
from typing import List, Dict, Any, Optional, Deque
from collections import deque
from dotenv import load_dotenv
from datetime import datetime
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory

load_dotenv()


class ExampleBuffer:
    """Manages a buffer of learning examples"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.examples: Deque[Dict[str, Any]] = deque(maxlen=max_size)
        self.performance_history: List[float] = []
    
    def add_example(self, input_text: str, output_text: str, 
                    feedback_score: Optional[float] = None):
        """Add a new example to the buffer"""
        example = {
            "input": input_text,
            "output": output_text,
            "feedback": feedback_score,
            "timestamp": datetime.now().isoformat()
        }
        self.examples.append(example)
        
        if feedback_score is not None:
            self.performance_history.append(feedback_score)
    
    def get_recent_examples(self, n: int = 5) -> List[Dict[str, str]]:
        """Get n most recent examples"""
        recent = list(self.examples)[-n:]
        return [{"input": ex["input"], "output": ex["output"]} for ex in recent]
    
    def get_best_examples(self, n: int = 5) -> List[Dict[str, str]]:
        """Get n best-rated examples"""
        if not self.examples:
            return []
        
        # Filter examples with feedback
        rated = [ex for ex in self.examples if ex["feedback"] is not None]
        if not rated:
            return self.get_recent_examples(n)
        
        # Sort by feedback score
        sorted_examples = sorted(rated, key=lambda x: x["feedback"], reverse=True)
        top_n = sorted_examples[:n]
        
        return [{"input": ex["input"], "output": ex["output"]} for ex in top_n]
    
    def detect_drift(self, window_size: int = 20) -> bool:
        """Detect if performance is degrading (concept drift)"""
        if len(self.performance_history) < window_size * 2:
            return False
        
        # Compare recent window to older window
        recent_window = self.performance_history[-window_size:]
        older_window = self.performance_history[-window_size*2:-window_size]
        
        recent_avg = sum(recent_window) / len(recent_window)
        older_avg = sum(older_window) / len(older_window)
        
        # Drift detected if recent performance significantly worse
        drift_threshold = 0.15  # 15% degradation
        return (older_avg - recent_avg) > drift_threshold
    
    def get_performance_trend(self) -> str:
        """Get description of performance trend"""
        if len(self.performance_history) < 10:
            return "insufficient_data"
        
        recent = self.performance_history[-10:]
        older = self.performance_history[-20:-10] if len(self.performance_history) >= 20 else []
        
        recent_avg = sum(recent) / len(recent)
        
        if not older:
            return f"recent_avg: {recent_avg:.2f}"
        
        older_avg = sum(older) / len(older)
        
        if recent_avg > older_avg + 0.1:
            return f"improving (from {older_avg:.2f} to {recent_avg:.2f})"
        elif recent_avg < older_avg - 0.1:
            return f"declining (from {older_avg:.2f} to {recent_avg:.2f})"
        else:
            return f"stable (around {recent_avg:.2f})"


class OnlineLearningAgent:
    """Agent that learns continuously from interactions"""
    
    def __init__(self, model_name: str = "gpt-4", domain: str = "general"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.domain = domain
        self.example_buffer = ExampleBuffer(max_size=100)
        self.memory = ConversationBufferMemory()
        self.total_interactions = 0
        
        # Base system prompt
        self.base_system_prompt = f"""You are an AI assistant specialized in {domain}.
You learn continuously from interactions and user feedback to improve your responses."""
        
        # Dynamic few-shot prompt template
        self.example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ])
    
    def create_dynamic_prompt(self, use_best: bool = True) -> ChatPromptTemplate:
        """Create prompt with dynamic examples from learning buffer"""
        # Get examples from buffer
        if use_best:
            examples = self.example_buffer.get_best_examples(n=3)
        else:
            examples = self.example_buffer.get_recent_examples(n=3)
        
        if not examples:
            # No examples yet, use base prompt
            return ChatPromptTemplate.from_messages([
                ("system", self.base_system_prompt),
                ("human", "{input}")
            ])
        
        # Create few-shot prompt with learned examples
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=self.example_prompt,
            examples=examples
        )
        
        # Get performance trend
        trend = self.example_buffer.get_performance_trend()
        system_msg = f"""{self.base_system_prompt}

Learning Status: {len(self.example_buffer.examples)} examples learned
Performance Trend: {trend}

Learn from these example interactions:"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_msg),
            few_shot_prompt,
            ("human", "{input}")
        ])
    
    def respond(self, user_input: str) -> str:
        """Generate response using current learned knowledge"""
        # Create dynamic prompt with learned examples
        prompt = self.create_dynamic_prompt(use_best=True)
        
        # Generate response
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"input": user_input})
        
        self.total_interactions += 1
        return response
    
    def learn_from_feedback(self, user_input: str, response: str, 
                           feedback_score: float):
        """Learn from user feedback on a response"""
        print(f"\nLearning from feedback (score: {feedback_score})...")
        
        # Add to example buffer
        self.example_buffer.add_example(user_input, response, feedback_score)
        
        # Check for drift
        if self.example_buffer.detect_drift():
            print("⚠️  Concept drift detected! Performance declining.")
            print("Consider adjusting learning strategy or model.")
        
        # Show learning progress
        trend = self.example_buffer.get_performance_trend()
        print(f"Performance trend: {trend}")
    
    def interact_and_learn(self, user_input: str, feedback_score: Optional[float] = None) -> Dict[str, Any]:
        """Complete interaction cycle with learning"""
        print(f"\n{'='*60}")
        print(f"User: {user_input}")
        print(f"{'='*60}")
        
        # Generate response
        response = self.respond(user_input)
        print(f"\nAssistant: {response}")
        
        # Learn from feedback if provided
        if feedback_score is not None:
            self.learn_from_feedback(user_input, response, feedback_score)
        
        return {
            "input": user_input,
            "response": response,
            "feedback": feedback_score,
            "total_interactions": self.total_interactions,
            "examples_learned": len(self.example_buffer.examples),
            "performance_trend": self.example_buffer.get_performance_trend()
        }
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learning progress"""
        return {
            "total_interactions": self.total_interactions,
            "examples_in_buffer": len(self.example_buffer.examples),
            "performance_history_size": len(self.example_buffer.performance_history),
            "performance_trend": self.example_buffer.get_performance_trend(),
            "drift_detected": self.example_buffer.detect_drift()
        }


def demonstrate_online_learning():
    """Demonstrate online learning pattern"""
    print("\n" + "="*70)
    print("ONLINE LEARNING DEMONSTRATION")
    print("="*70)
    
    # Create agent for customer support domain
    agent = OnlineLearningAgent(domain="customer support")
    
    # Example 1: Initial interactions (building knowledge)
    print("\n" + "="*70)
    print("Example 1: Initial Learning Phase")
    print("="*70)
    
    interactions = [
        ("How do I reset my password?", 0.9),
        ("Where can I find my order history?", 0.85),
        ("How do I contact support?", 0.95),
        ("What is your return policy?", 0.8),
        ("How long does shipping take?", 0.9),
    ]
    
    for user_input, feedback in interactions:
        agent.interact_and_learn(user_input, feedback)
    
    # Show learning stats
    print("\n" + "-"*60)
    print("Learning Statistics After Initial Phase:")
    stats = agent.get_learning_stats()
    print(json.dumps(stats, indent=2))
    
    # Example 2: Applying learned knowledge
    print("\n" + "="*70)
    print("Example 2: Applying Learned Knowledge")
    print("="*70)
    
    new_queries = [
        "I forgot my password, what should I do?",  # Similar to learned
        "How can I see my past orders?",  # Similar to learned
    ]
    
    for query in new_queries:
        result = agent.interact_and_learn(query, feedback_score=0.9)
    
    # Example 3: Continuous learning with feedback
    print("\n" + "="*70)
    print("Example 3: Continuous Learning with Mixed Feedback")
    print("="*70)
    
    mixed_interactions = [
        ("Can I change my delivery address?", 0.7),  # Medium feedback
        ("Do you offer gift wrapping?", 0.85),  # Good feedback
        ("How do I cancel my subscription?", 0.6),  # Lower feedback
        ("What payment methods do you accept?", 0.9),  # High feedback
    ]
    
    for user_input, feedback in mixed_interactions:
        agent.interact_and_learn(user_input, feedback)
    
    # Example 4: Detecting concept drift
    print("\n" + "="*70)
    print("Example 4: Simulating Performance Degradation (Drift)")
    print("="*70)
    
    # Simulate declining performance
    drift_interactions = [
        ("Question about new feature X", 0.4),  # New topics, poor performance
        ("Tell me about feature Y", 0.3),
        ("How does feature Z work?", 0.35),
        ("Feature A questions", 0.4),
        ("Feature B support", 0.3),
    ]
    
    for user_input, feedback in drift_interactions:
        agent.interact_and_learn(user_input, feedback)
    
    # Final statistics
    print("\n" + "="*70)
    print("Final Learning Statistics:")
    print("="*70)
    final_stats = agent.get_learning_stats()
    print(json.dumps(final_stats, indent=2))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Online Learning Pattern demonstrated:

Key Features:
1. Incremental Learning - Updates knowledge with each interaction
2. Example Buffer - Maintains recent and best examples
3. Performance Tracking - Monitors learning effectiveness
4. Drift Detection - Identifies when patterns change
5. Dynamic Adaptation - Adjusts prompts based on learned examples

Learning Mechanisms:
- Few-shot learning with best examples
- Continuous feedback integration
- Performance trend analysis
- Concept drift detection
- Adaptive example selection

Applications:
- Customer support bots improving from interactions
- Personalized assistants learning user preferences
- Recommendation systems adapting to trends
- Fraud detection with evolving patterns
- Adaptive tutoring systems

Benefits:
- Continuous improvement without retraining
- Adaptation to changing patterns
- Personalization based on interaction history
- Early warning for performance degradation
- Efficient learning from limited examples

Limitations:
- Requires consistent feedback
- May need periodic full retraining
- Can accumulate biases from feedback
- Limited by context window size
    """)


if __name__ == "__main__":
    demonstrate_online_learning()
