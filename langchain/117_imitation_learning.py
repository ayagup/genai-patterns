"""
Pattern 117: Imitation Learning

Description:
    Agent learns by observing and imitating expert behavior through
    behavioral cloning and inverse reinforcement learning approaches.

Components:
    - Expert demonstrations
    - Behavioral cloning
    - Policy learning

Use Cases:
    - Robotics
    - Complex skill acquisition
    - Learning from examples

LangChain Implementation:
    Uses few-shot learning and demonstration-based prompting to imitate expert behavior.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ImitationLearningAgent:
    """Agent that learns by observing and imitating expert demonstrations."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.expert_demonstrations = []
        self.learned_policy = None
        
    def observe_expert(self, situation: str, expert_action: str, reasoning: str):
        """Observe and record expert demonstration."""
        demonstration = {
            "situation": situation,
            "action": expert_action,
            "reasoning": reasoning
        }
        self.expert_demonstrations.append(demonstration)
        print(f"Observed demonstration {len(self.expert_demonstrations)}:")
        print(f"  Situation: {situation}")
        print(f"  Action: {expert_action}")
        print(f"  Reasoning: {reasoning}")
        print()
        
    def learn_from_demonstrations(self):
        """Learn policy from all observed demonstrations."""
        learning_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze these expert demonstrations and extract the underlying policy, patterns, and decision-making strategy."),
            ("user", """Expert Demonstrations:
{demonstrations}

Extract:
1. Common patterns in decision-making
2. Key factors considered
3. General policy rules
4. Decision strategy""")
        ])
        
        demos_text = "\n\n".join([
            f"Demo {i+1}:\nSituation: {d['situation']}\nAction: {d['action']}\nReasoning: {d['reasoning']}"
            for i, d in enumerate(self.expert_demonstrations)
        ])
        
        chain = learning_prompt | self.llm | StrOutputParser()
        self.learned_policy = chain.invoke({"demonstrations": demos_text})
        
        print("=== Learned Policy ===")
        print(self.learned_policy)
        print()
        
    def imitate_expert(self, new_situation: str) -> Dict[str, str]:
        """Apply learned policy to a new situation."""
        if not self.learned_policy:
            return {"error": "No policy learned yet. Need more demonstrations."}
        
        # Create few-shot prompt with demonstrations
        example_template = """Situation: {situation}
Expert Action: {action}
Reasoning: {reasoning}"""
        
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=ChatPromptTemplate.from_messages([
                ("user", "Situation: {situation}"),
                ("assistant", "Action: {action}\nReasoning: {reasoning}")
            ]),
            examples=self.expert_demonstrations,
        )
        
        full_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are imitating an expert's decision-making style.

Learned Policy:
{learned_policy}

Use the learned policy and demonstrated patterns to decide actions for new situations."""),
            few_shot_prompt,
            ("user", "Situation: {new_situation}\n\nWhat action should be taken? Provide action and reasoning.")
        ])
        
        chain = full_prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "learned_policy": self.learned_policy,
            "new_situation": new_situation
        })
        
        # Parse response
        lines = response.split('\n')
        action = ""
        reasoning = ""
        
        for line in lines:
            if line.startswith("Action:"):
                action = line.replace("Action:", "").strip()
            elif line.startswith("Reasoning:"):
                reasoning = line.replace("Reasoning:", "").strip()
            elif action and not reasoning:
                reasoning = line.strip()
        
        return {
            "situation": new_situation,
            "action": action or response,
            "reasoning": reasoning or "Based on learned policy"
        }
    
    def evaluate_imitation(self, test_situation: str, expected_action: str) -> bool:
        """Evaluate how well the agent imitates expert behavior."""
        result = self.imitate_expert(test_situation)
        
        print(f"Test Situation: {test_situation}")
        print(f"Expected Action: {expected_action}")
        print(f"Agent Action: {result['action']}")
        print(f"Reasoning: {result['reasoning']}")
        
        # Simple similarity check
        similarity = expected_action.lower() in result['action'].lower()
        print(f"Match: {'✓' if similarity else '✗'}")
        print()
        
        return similarity


def demonstrate_imitation_learning():
    """Demonstrate imitation learning pattern."""
    print("=== Imitation Learning Pattern ===\n")
    
    agent = ImitationLearningAgent()
    
    # Scenario: Learning customer service expert behavior
    print("1. Observing Expert Demonstrations")
    print("-" * 50)
    
    # Demonstration 1: Angry customer
    agent.observe_expert(
        situation="Customer is angry about a delayed delivery",
        expert_action="Apologize sincerely, explain the situation, offer compensation (discount or free shipping on next order)",
        reasoning="Acknowledge emotions first, show empathy, provide solution to restore trust"
    )
    
    # Demonstration 2: Confused customer
    agent.observe_expert(
        situation="Customer is confused about how to use a product feature",
        expert_action="Ask clarifying questions, provide step-by-step guidance, offer visual aids or video tutorial",
        reasoning="Understand specific confusion point, break down into simple steps, provide multiple learning formats"
    )
    
    # Demonstration 3: Billing inquiry
    agent.observe_expert(
        situation="Customer questions an unexpected charge on their bill",
        expert_action="Review the charge details, explain clearly what it's for, provide itemized breakdown if needed",
        reasoning="Be transparent about charges, ensure customer understands, address concerns directly"
    )
    
    # Demonstration 4: Feature request
    agent.observe_expert(
        situation="Customer requests a feature that doesn't exist",
        expert_action="Thank them for feedback, explain current capabilities, document request for product team, suggest workaround if possible",
        reasoning="Value customer input, manage expectations, show commitment to improvement, help immediately if possible"
    )
    
    print("✓ Observed 4 expert demonstrations\n")
    
    # Learn from demonstrations
    print("2. Learning Policy from Demonstrations")
    print("-" * 50)
    agent.learn_from_demonstrations()
    
    # Test imitation on new situations
    print("3. Testing Imitation on New Situations")
    print("-" * 50)
    
    # Test 1: Product defect
    print("Test 1: Product Defect Situation")
    result = agent.imitate_expert("Customer reports a defective product that stopped working after one week")
    print(f"Situation: {result['situation']}")
    print(f"Action: {result['action']}")
    print(f"Reasoning: {result['reasoning']}")
    print()
    
    # Test 2: Refund request
    print("Test 2: Refund Request Situation")
    result = agent.imitate_expert("Customer wants a refund for a product they bought 2 months ago and barely used")
    print(f"Situation: {result['situation']}")
    print(f"Action: {result['action']}")
    print(f"Reasoning: {result['reasoning']}")
    print()
    
    # Test 3: Complex technical issue
    print("Test 3: Technical Issue Situation")
    result = agent.imitate_expert("Customer experiencing intermittent connectivity issues with the app")
    print(f"Situation: {result['situation']}")
    print(f"Action: {result['action']}")
    print(f"Reasoning: {result['reasoning']}")
    print()
    
    # Evaluation
    print("4. Evaluating Imitation Quality")
    print("-" * 50)
    agent.evaluate_imitation(
        "Customer complaint about rude staff member",
        "Apologize for experience, escalate to manager, ensure follow-up"
    )
    
    print("=== Summary ===")
    print("Imitation learning demonstrated with:")
    print("- Expert demonstration observation")
    print("- Policy learning from demonstrations")
    print("- Behavioral cloning for new situations")
    print("- Few-shot learning approach")
    print("- Pattern extraction from expert behavior")


if __name__ == "__main__":
    demonstrate_imitation_learning()
