"""
Pattern 112: Common Sense Reasoning

Description:
    Common Sense Reasoning enables agents to apply implicit, everyday knowledge
    that humans take for granted. This pattern integrates common sense knowledge
    bases (like ConceptNet) with LLM reasoning to make inferences about the
    physical world, social norms, causality, and typical behaviors.
    
    The pattern helps agents understand context, make reasonable assumptions,
    avoid absurd conclusions, and reason about unstated but obvious facts.
    It bridges the gap between explicit knowledge and implicit understanding.

Key Components:
    1. Knowledge Base: ConceptNet or similar common sense KB
    2. Reasoning Engine: Inference over common sense facts
    3. Context Integrator: Applies common sense to situations
    4. Plausibility Checker: Validates conclusions
    5. Assumption Generator: Makes reasonable assumptions

Common Sense Categories:
    - Physical: Objects fall down, water is wet
    - Social: Greetings are polite, lying is wrong
    - Causal: Rain makes things wet, fire causes heat
    - Temporal: Events have order, time moves forward
    - Spatial: Objects can't be in two places
    - Functional: Tools have purposes, actions have effects

Use Cases:
    - Natural language understanding
    - Story comprehension and generation
    - Situation interpretation
    - Question answering with implicit knowledge
    - Robotics and embodied AI

LangChain Implementation:
    Uses ConceptNet integration, reasoning chains, and validation logic to
    apply common sense knowledge to agent reasoning.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.chains import LLMChain

load_dotenv()


class CommonSenseKnowledgeBase:
    """Simulated common sense knowledge base (in practice, use ConceptNet API)"""
    
    def __init__(self):
        self.knowledge = {
            # Physical laws
            "gravity": "Objects fall downward when dropped",
            "water_wet": "Water makes things wet",
            "fire_hot": "Fire is hot and can burn",
            "ice_cold": "Ice is cold and can freeze",
            "glass_fragile": "Glass is fragile and can break",
            
            # Social norms
            "greeting_polite": "Greeting people is considered polite",
            "stealing_wrong": "Taking things without permission is wrong",
            "helping_good": "Helping others is generally good",
            "lying_negative": "Lying usually has negative consequences",
            
            # Causal relationships
            "rain_wet": "Rain makes outdoor surfaces wet",
            "eating_hunger": "Eating satisfies hunger",
            "sleep_rest": "Sleep provides rest and recovery",
            "exercise_fitness": "Exercise improves fitness",
            
            # Temporal facts
            "time_forward": "Time moves forward, not backward",
            "aging_continuous": "Living things age continuously",
            "seasons_cycle": "Seasons cycle in a predictable pattern",
            
            # Spatial facts
            "one_place": "Physical objects can't be in two places at once",
            "inside_outside": "Something inside a closed container is not outside",
            "solid_through": "Solid objects can't pass through solid walls",
            
            # Functional knowledge
            "key_lock": "Keys are used to open locks",
            "knife_cut": "Knives are used for cutting",
            "chair_sit": "Chairs are for sitting on",
            "umbrella_rain": "Umbrellas protect from rain",
        }
        
        self.categories = {
            "physical": ["gravity", "water_wet", "fire_hot", "ice_cold", "glass_fragile"],
            "social": ["greeting_polite", "stealing_wrong", "helping_good", "lying_negative"],
            "causal": ["rain_wet", "eating_hunger", "sleep_rest", "exercise_fitness"],
            "temporal": ["time_forward", "aging_continuous", "seasons_cycle"],
            "spatial": ["one_place", "inside_outside", "solid_through"],
            "functional": ["key_lock", "knife_cut", "chair_sit", "umbrella_rain"],
        }
    
    def query(self, concept: str) -> Optional[str]:
        """Query knowledge base for common sense fact"""
        return self.knowledge.get(concept)
    
    def get_category(self, category: str) -> List[str]:
        """Get all facts in a category"""
        concept_ids = self.categories.get(category, [])
        return [self.knowledge[cid] for cid in concept_ids if cid in self.knowledge]
    
    def get_relevant_facts(self, context: str) -> List[str]:
        """Get facts relevant to context"""
        # Simple keyword matching (in practice, use embeddings)
        relevant = []
        context_lower = context.lower()
        for key, fact in self.knowledge.items():
            if any(word in context_lower for word in key.split("_")):
                relevant.append(fact)
        return relevant


class CommonSenseReasoner:
    """Agent that applies common sense reasoning"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.kb = CommonSenseKnowledgeBase()
        
        # Reasoning with common sense
        self.reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a reasoning agent that applies common sense knowledge.
            
Given a question or situation, use the provided common sense facts to reason
about the answer. Make reasonable assumptions based on everyday knowledge.

Common sense facts:
{facts}

Question: {question}

Provide your reasoning and answer:"""),
            ("human", "{question}")
        ])
        
        # Plausibility checking
        self.plausibility_prompt = ChatPromptTemplate.from_messages([
            ("system", """Evaluate if a conclusion is plausible based on common sense.
            
Common sense facts:
{facts}

Proposed conclusion: {conclusion}

Rate plausibility from 0-10 and explain why:"""),
            ("human", "{conclusion}")
        ])
        
        # Assumption generation
        self.assumption_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given a scenario, generate reasonable assumptions based on common sense.
            
Scenario: {scenario}

List 5 reasonable assumptions that most people would make:"""),
            ("human", "{scenario}")
        ])
        
        self.reasoning_chain = self.reasoning_prompt | self.llm | StrOutputParser()
        self.plausibility_chain = self.plausibility_prompt | self.llm | StrOutputParser()
        self.assumption_chain = self.assumption_prompt | self.llm | StrOutputParser()
    
    def reason_with_common_sense(self, question: str) -> Dict[str, Any]:
        """Apply common sense reasoning to answer a question"""
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        # Get relevant common sense facts
        relevant_facts = self.kb.get_relevant_facts(question)
        if not relevant_facts:
            relevant_facts = list(self.kb.knowledge.values())[:10]
        
        facts_str = "\n".join(f"- {fact}" for fact in relevant_facts)
        
        print(f"\nRelevant common sense facts:")
        for fact in relevant_facts:
            print(f"  - {fact}")
        
        # Reason with facts
        reasoning = self.reasoning_chain.invoke({
            "question": question,
            "facts": facts_str
        })
        
        print(f"\nReasoning:")
        print(reasoning)
        
        return {
            "question": question,
            "facts": relevant_facts,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }
    
    def check_plausibility(self, conclusion: str, context: str = "") -> Dict[str, Any]:
        """Check if a conclusion is plausible"""
        print(f"\n{'='*60}")
        print(f"Checking plausibility: {conclusion}")
        print(f"{'='*60}")
        
        # Get relevant facts
        relevant_facts = self.kb.get_relevant_facts(conclusion + " " + context)
        if not relevant_facts:
            relevant_facts = list(self.kb.knowledge.values())[:10]
        
        facts_str = "\n".join(f"- {fact}" for fact in relevant_facts)
        
        # Check plausibility
        evaluation = self.plausibility_chain.invoke({
            "conclusion": conclusion,
            "facts": facts_str
        })
        
        print(f"\nEvaluation:")
        print(evaluation)
        
        return {
            "conclusion": conclusion,
            "facts": relevant_facts,
            "evaluation": evaluation,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_assumptions(self, scenario: str) -> Dict[str, Any]:
        """Generate reasonable assumptions for a scenario"""
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario}")
        print(f"{'='*60}")
        
        # Generate assumptions
        assumptions = self.assumption_chain.invoke({
            "scenario": scenario
        })
        
        print(f"\nGenerated assumptions:")
        print(assumptions)
        
        return {
            "scenario": scenario,
            "assumptions": assumptions,
            "timestamp": datetime.now().isoformat()
        }
    
    def explain_with_common_sense(self, observation: str) -> Dict[str, Any]:
        """Explain an observation using common sense"""
        print(f"\n{'='*60}")
        print(f"Observation: {observation}")
        print(f"{'='*60}")
        
        # Get causal facts
        causal_facts = self.kb.get_category("causal")
        facts_str = "\n".join(f"- {fact}" for fact in causal_facts)
        
        explanation_prompt = f"""Using common sense causal knowledge, explain this observation:

Observation: {observation}

Common sense causal facts:
{facts_str}

Provide a simple, common sense explanation:"""
        
        explanation = self.llm.invoke(explanation_prompt).content
        
        print(f"\nExplanation:")
        print(explanation)
        
        return {
            "observation": observation,
            "causal_facts": causal_facts,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }


def demonstrate_common_sense_reasoning():
    """Demonstrate common sense reasoning patterns"""
    print("\n" + "="*70)
    print("COMMON SENSE REASONING DEMONSTRATION")
    print("="*70)
    
    reasoner = CommonSenseReasoner()
    
    # Example 1: Question answering with common sense
    print("\n" + "="*70)
    print("Example 1: Question Answering with Common Sense")
    print("="*70)
    
    question1 = "If I drop a glass cup on the floor, what will happen?"
    result1 = reasoner.reason_with_common_sense(question1)
    
    question2 = "Should I use an umbrella when it's raining?"
    result2 = reasoner.reason_with_common_sense(question2)
    
    # Example 2: Plausibility checking
    print("\n" + "="*70)
    print("Example 2: Plausibility Checking")
    print("="*70)
    
    conclusion1 = "I can walk through a solid brick wall"
    plausibility1 = reasoner.check_plausibility(conclusion1)
    
    conclusion2 = "If I leave ice cream outside in the sun, it will melt"
    plausibility2 = reasoner.check_plausibility(conclusion2)
    
    # Example 3: Assumption generation
    print("\n" + "="*70)
    print("Example 3: Assumption Generation")
    print("="*70)
    
    scenario = "A person enters a restaurant and sits at a table"
    assumptions = reasoner.generate_assumptions(scenario)
    
    # Example 4: Explanation with common sense
    print("\n" + "="*70)
    print("Example 4: Explaining Observations")
    print("="*70)
    
    observation1 = "The sidewalk is wet even though no one watered it"
    explanation1 = reasoner.explain_with_common_sense(observation1)
    
    observation2 = "The person looks tired and is yawning"
    explanation2 = reasoner.explain_with_common_sense(observation2)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Common Sense Reasoning Pattern demonstrated:

Key Features:
1. Knowledge Base Integration - Access to common sense facts
2. Contextual Reasoning - Apply relevant facts to situations
3. Plausibility Checking - Validate conclusions against common sense
4. Assumption Generation - Make reasonable implicit assumptions
5. Causal Explanation - Explain observations with common sense

Applications:
- Natural language understanding
- Question answering systems
- Story comprehension
- Situation interpretation
- Robotics and embodied AI

Common sense reasoning helps agents bridge the gap between explicit
knowledge and implicit understanding that humans naturally possess.
    """)


if __name__ == "__main__":
    demonstrate_common_sense_reasoning()
