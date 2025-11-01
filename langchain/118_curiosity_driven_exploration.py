"""
Pattern 118: Curiosity-Driven Exploration

Description:
    Agent explores environment based on intrinsic motivation driven by
    novelty, information gain, and prediction error.

Components:
    - Novelty detection
    - Information gain calculation
    - Exploration strategy
    - Intrinsic reward

Use Cases:
    - Open-ended learning
    - Exploration tasks
    - Discovery systems

LangChain Implementation:
    Uses LLM to generate curious questions and exploration strategies based on
    knowledge gaps and novelty.
"""

import os
from typing import List, Dict, Any, Set
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class CuriosityDrivenAgent:
    """Agent that explores based on intrinsic curiosity."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.8)
        self.knowledge_base = set()
        self.explored_topics = []
        self.curiosity_log = []
        
    def assess_novelty(self, topic: str) -> float:
        """Assess how novel a topic is based on existing knowledge."""
        novelty_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rate the novelty of this topic given existing knowledge. Return a score from 0.0 (completely known) to 1.0 (completely novel)."),
            ("user", """Topic: {topic}

Known Knowledge:
{knowledge}

Novelty score (0.0-1.0):""")
        ])
        
        knowledge_str = "\n".join([f"- {k}" for k in list(self.knowledge_base)[:20]]) or "Empty"
        
        chain = novelty_prompt | self.llm | StrOutputParser()
        response = chain.invoke({"topic": topic, "knowledge": knowledge_str})
        
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def generate_curious_questions(self, context: str) -> List[str]:
        """Generate questions driven by curiosity about knowledge gaps."""
        curiosity_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a curious agent exploring knowledge.
Generate 3-5 interesting questions that would expand understanding in unexplored directions.
Focus on:
- Novel connections
- Surprising implications
- Unexplored aspects
- Deep mechanisms"""),
            ("user", """Current Context: {context}

Known Knowledge:
{knowledge}

Generate curious questions:""")
        ])
        
        knowledge_str = "\n".join([f"- {k}" for k in list(self.knowledge_base)[-10:]]) or "Empty"
        
        chain = curiosity_prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "knowledge": knowledge_str})
        
        # Parse questions
        questions = [line.strip('- ').strip() for line in response.split('\n') if line.strip()]
        return [q for q in questions if '?' in q][:5]
    
    def calculate_information_gain(self, potential_exploration: str) -> float:
        """Calculate expected information gain from exploring a topic."""
        gain_prompt = ChatPromptTemplate.from_messages([
            ("system", "Estimate information gain (0.0-1.0) from exploring this topic given current knowledge."),
            ("user", """Potential Exploration: {exploration}

Current Knowledge:
{knowledge}

Information gain score (0.0-1.0):""")
        ])
        
        knowledge_str = "\n".join([f"- {k}" for k in list(self.knowledge_base)[-15:]]) or "Empty"
        
        chain = gain_prompt | self.llm | StrOutputParser()
        response = chain.invoke({"exploration": potential_exploration, "knowledge": knowledge_str})
        
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def explore(self, topic: str) -> Dict[str, Any]:
        """Explore a topic and gain knowledge."""
        print(f"\n🔍 Exploring: {topic}")
        
        # Assess novelty
        novelty = self.assess_novelty(topic)
        print(f"  Novelty: {novelty:.2f}")
        
        # Calculate information gain
        info_gain = self.calculate_information_gain(topic)
        print(f"  Information Gain: {info_gain:.2f}")
        
        # Intrinsic reward
        intrinsic_reward = (novelty + info_gain) / 2
        print(f"  Intrinsic Reward: {intrinsic_reward:.2f}")
        
        # Explore the topic
        exploration_prompt = ChatPromptTemplate.from_messages([
            ("system", "Explore this topic deeply and discover interesting insights."),
            ("user", "Topic: {topic}\n\nProvide interesting discoveries:")
        ])
        
        chain = exploration_prompt | self.llm | StrOutputParser()
        discoveries = chain.invoke({"topic": topic})
        
        # Update knowledge base
        self.knowledge_base.add(topic)
        self.explored_topics.append(topic)
        
        result = {
            "topic": topic,
            "novelty": novelty,
            "information_gain": info_gain,
            "intrinsic_reward": intrinsic_reward,
            "discoveries": discoveries
        }
        
        self.curiosity_log.append(result)
        
        print(f"\n  Discoveries:")
        for line in discoveries.split('\n')[:3]:
            if line.strip():
                print(f"    • {line.strip()}")
        
        return result
    
    def choose_next_exploration(self, candidates: List[str]) -> str:
        """Choose next exploration target based on curiosity."""
        print("\n📊 Evaluating exploration candidates...")
        
        scores = []
        for candidate in candidates:
            novelty = self.assess_novelty(candidate)
            info_gain = self.calculate_information_gain(candidate)
            curiosity_score = (novelty + info_gain) / 2
            scores.append((candidate, curiosity_score))
            print(f"  {candidate}: {curiosity_score:.2f}")
        
        # Choose highest curiosity score
        best_candidate = max(scores, key=lambda x: x[1])
        print(f"\n✨ Selected: {best_candidate[0]} (score: {best_candidate[1]:.2f})")
        
        return best_candidate[0]
    
    def get_exploration_summary(self) -> str:
        """Summarize exploration journey."""
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize this curiosity-driven exploration journey."),
            ("user", """Explored Topics:
{topics}

Curiosity Log:
{log}

Provide a summary of:
1. Key discoveries
2. Exploration patterns
3. Knowledge growth
4. Most rewarding explorations""")
        ])
        
        topics = "\n".join([f"{i+1}. {t}" for i, t in enumerate(self.explored_topics)])
        log = "\n".join([
            f"Topic: {entry['topic']}, Reward: {entry['intrinsic_reward']:.2f}"
            for entry in self.curiosity_log
        ])
        
        chain = summary_prompt | self.llm | StrOutputParser()
        return chain.invoke({"topics": topics, "log": log})


def demonstrate_curiosity_driven_exploration():
    """Demonstrate curiosity-driven exploration pattern."""
    print("=== Curiosity-Driven Exploration Pattern ===\n")
    
    agent = CuriosityDrivenAgent()
    
    # Initial topic
    print("1. Starting Exploration from Initial Topic")
    print("-" * 50)
    agent.explore("Quantum Computing")
    
    # Generate curious questions
    print("\n2. Generating Curious Questions")
    print("-" * 50)
    questions = agent.generate_curious_questions("Quantum Computing")
    print("Curious questions generated:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    
    # Explore based on curiosity
    print("\n3. Curiosity-Driven Exploration")
    print("-" * 50)
    
    # Candidate topics
    candidates = [
        "Quantum Entanglement Applications",
        "Quantum Error Correction",
        "Superconducting Qubits",
        "Quantum Algorithms for Chemistry"
    ]
    
    # Choose and explore
    for i in range(3):
        print(f"\n--- Exploration Round {i+1} ---")
        next_topic = agent.choose_next_exploration(candidates)
        agent.explore(next_topic)
        
        # Remove explored topic and add related ones
        candidates.remove(next_topic)
        
        # Generate new candidates based on discoveries
        if i < 2:
            new_candidates_prompt = ChatPromptTemplate.from_messages([
                ("system", "Suggest 2 related unexplored topics based on recent discovery."),
                ("user", "Recent exploration: {topic}\n\nSuggest new topics:")
            ])
            chain = new_candidates_prompt | agent.llm | StrOutputParser()
            new_suggestions = chain.invoke({"topic": next_topic})
            new_topics = [line.strip('- ').strip() for line in new_suggestions.split('\n') if line.strip()][:2]
            candidates.extend(new_topics)
    
    # Summary
    print("\n4. Exploration Journey Summary")
    print("-" * 50)
    summary = agent.get_exploration_summary()
    print(summary)
    
    print("\n=== Summary ===")
    print(f"Total explorations: {len(agent.explored_topics)}")
    print(f"Knowledge base size: {len(agent.knowledge_base)}")
    print(f"Average intrinsic reward: {sum(e['intrinsic_reward'] for e in agent.curiosity_log) / len(agent.curiosity_log):.2f}")
    print("\nCuriosity-driven exploration demonstrated with:")
    print("- Novelty assessment")
    print("- Information gain calculation")
    print("- Intrinsic reward-based decisions")
    print("- Autonomous exploration strategy")
    print("- Continuous knowledge expansion")


if __name__ == "__main__":
    demonstrate_curiosity_driven_exploration()
