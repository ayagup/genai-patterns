"""
Pattern 115: Transfer Learning Agent

Description:
    Transfer Learning enables agents to leverage knowledge learned from one
    task or domain to improve performance on related tasks. This pattern
    identifies transferable knowledge, adapts it to new contexts, and combines
    it with task-specific learning for faster adaptation and better performance.
    
    Essential for agents that need to work across multiple related domains or
    tasks, transfer learning reduces the data and time needed to achieve good
    performance in new areas.

Key Components:
    1. Source Knowledge Extractor: Identifies transferable knowledge
    2. Domain Adapter: Adapts knowledge to new domain
    3. Fine-tuner: Specializes for target task
    4. Transfer Strategy Selector: Chooses transfer approach
    5. Performance Evaluator: Assesses transfer effectiveness

Transfer Strategies:
    - Feature Transfer: Use learned representations
    - Parameter Transfer: Transfer model parameters
    - Instance Transfer: Leverage source examples
    - Relational Transfer: Transfer relationships
    - Meta-transfer: Transfer learning strategies

Use Cases:
    - Multi-domain chatbots
    - Cross-lingual applications
    - Domain adaptation (e.g., medical to legal)
    - Few-shot learning in new domains
    - Skill transfer in robotics

LangChain Implementation:
    Uses prompt engineering, few-shot examples, and domain adaptation
    techniques to transfer knowledge between tasks and domains.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class DomainKnowledge:
    """Represents knowledge from a source domain"""
    
    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.examples: List[Dict[str, str]] = []
        self.patterns: List[str] = []
        self.concepts: Dict[str, str] = {}
    
    def add_example(self, input_text: str, output_text: str, pattern: str = ""):
        """Add an example from this domain"""
        self.examples.append({
            "input": input_text,
            "output": output_text,
            "pattern": pattern
        })
    
    def add_pattern(self, pattern: str):
        """Add a learned pattern"""
        if pattern not in self.patterns:
            self.patterns.append(pattern)
    
    def add_concept(self, concept: str, description: str):
        """Add a domain concept"""
        self.concepts[concept] = description


class TransferLearningAgent:
    """Agent that transfers knowledge between domains"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.source_domains: Dict[str, DomainKnowledge] = {}
        self.current_domain: Optional[str] = None
    
    def add_source_domain(self, domain: DomainKnowledge):
        """Add a source domain with learned knowledge"""
        self.source_domains[domain.domain_name] = domain
        print(f"Added source domain: {domain.domain_name}")
        print(f"  - {len(domain.examples)} examples")
        print(f"  - {len(domain.patterns)} patterns")
        print(f"  - {len(domain.concepts)} concepts")
    
    def identify_relevant_source(self, target_task: str) -> Optional[DomainKnowledge]:
        """Identify most relevant source domain for transfer"""
        if not self.source_domains:
            return None
        
        # Use LLM to identify most relevant source
        domains_desc = "\n".join([
            f"- {name}: {len(d.examples)} examples, patterns: {', '.join(d.patterns[:3])}"
            for name, d in self.source_domains.items()
        ])
        
        selection_prompt = f"""Given a target task and available source domains, identify which source domain would provide the most useful transferable knowledge.

Target task: {target_task}

Available source domains:
{domains_desc}

Which source domain is most relevant? Answer with just the domain name."""
        
        response = self.llm.invoke(selection_prompt).content.strip()
        
        # Find matching domain
        for domain_name in self.source_domains:
            if domain_name.lower() in response.lower():
                print(f"\nSelected source domain: {domain_name}")
                return self.source_domains[domain_name]
        
        # Default to first domain
        return list(self.source_domains.values())[0]
    
    def adapt_examples(self, source_domain: DomainKnowledge, 
                      target_domain: str, n: int = 3) -> List[Dict[str, str]]:
        """Adapt source examples to target domain"""
        if not source_domain.examples:
            return []
        
        # Take first n examples
        source_examples = source_domain.examples[:n]
        
        # Create adaptation prompt
        examples_str = "\n\n".join([
            f"Input: {ex['input']}\nOutput: {ex['output']}"
            for ex in source_examples
        ])
        
        adaptation_prompt = f"""Adapt these examples from {source_domain.domain_name} domain to {target_domain} domain.
Keep the same pattern and structure, but change the content to fit the target domain.

Source examples ({source_domain.domain_name}):
{examples_str}

Generate {n} adapted examples for {target_domain} domain in the same format."""
        
        response = self.llm.invoke(adaptation_prompt).content
        
        # Parse adapted examples (simplified parsing)
        adapted = []
        lines = response.split('\n')
        current_input = None
        
        for line in lines:
            if line.startswith('Input:'):
                current_input = line.replace('Input:', '').strip()
            elif line.startswith('Output:') and current_input:
                output = line.replace('Output:', '').strip()
                adapted.append({"input": current_input, "output": output})
                current_input = None
        
        return adapted[:n] if adapted else source_examples[:n]
    
    def transfer_and_respond(self, target_task: str, target_domain: str,
                            query: str) -> Dict[str, Any]:
        """Transfer knowledge and respond to query"""
        print(f"\n{'='*60}")
        print(f"Target Domain: {target_domain}")
        print(f"Target Task: {target_task}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        # Identify relevant source domain
        source_domain = self.identify_relevant_source(target_task)
        
        if source_domain:
            print(f"\nTransferring knowledge from: {source_domain.domain_name}")
            
            # Adapt examples to target domain
            adapted_examples = self.adapt_examples(source_domain, target_domain, n=3)
            print(f"Adapted {len(adapted_examples)} examples to target domain")
            
            # Extract transferable patterns
            patterns_str = "\n".join([f"- {p}" for p in source_domain.patterns])
            
            # Create transfer-enhanced prompt
            if adapted_examples:
                example_prompt = ChatPromptTemplate.from_messages([
                    ("human", "{input}"),
                    ("ai", "{output}")
                ])
                
                few_shot_prompt = FewShotChatMessagePromptTemplate(
                    example_prompt=example_prompt,
                    examples=adapted_examples
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are an AI assistant for {target_domain}.

You are leveraging transferred knowledge from {source_domain.domain_name}.

Transferable patterns:
{patterns_str}

Apply these learned patterns to the current domain:"""),
                    few_shot_prompt,
                    ("human", "{input}")
                ])
            else:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are an AI assistant for {target_domain}.

You are leveraging transferred knowledge from {source_domain.domain_name}.

Transferable patterns:
{patterns_str}

Apply these learned patterns to the current task."""),
                    ("human", "{input}")
                ])
            
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"input": query})
        else:
            # No source domain available, use base prompt
            print("\nNo source domain available, using base knowledge")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"You are an AI assistant for {target_domain}."),
                ("human", "{input}")
            ])
            
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"input": query})
        
        print(f"\nResponse:")
        print(response)
        
        return {
            "target_domain": target_domain,
            "target_task": target_task,
            "query": query,
            "source_domain": source_domain.domain_name if source_domain else None,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    def compare_with_without_transfer(self, target_task: str, target_domain: str,
                                     query: str) -> Dict[str, Any]:
        """Compare performance with and without transfer learning"""
        print(f"\n{'='*70}")
        print(f"COMPARISON: With vs Without Transfer Learning")
        print(f"{'='*70}")
        
        # Response with transfer
        print("\n--- WITH TRANSFER LEARNING ---")
        with_transfer = self.transfer_and_respond(target_task, target_domain, query)
        
        # Response without transfer (base model)
        print("\n--- WITHOUT TRANSFER LEARNING (Baseline) ---")
        base_prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are an AI assistant for {target_domain}."),
            ("human", "{input}")
        ])
        base_chain = base_prompt | self.llm | StrOutputParser()
        without_transfer = base_chain.invoke({"input": query})
        
        print(f"\nBaseline Response:")
        print(without_transfer)
        
        return {
            "with_transfer": with_transfer,
            "without_transfer": without_transfer
        }


def demonstrate_transfer_learning():
    """Demonstrate transfer learning pattern"""
    print("\n" + "="*70)
    print("TRANSFER LEARNING AGENT DEMONSTRATION")
    print("="*70)
    
    agent = TransferLearningAgent()
    
    # Create source domain: Customer Support
    print("\n" + "="*70)
    print("Setting Up Source Domain Knowledge")
    print("="*70)
    
    customer_support = DomainKnowledge("customer_support")
    
    # Add examples
    customer_support.add_example(
        "How do I reset my password?",
        "To reset your password: 1) Click 'Forgot Password' on login page, "
        "2) Enter your email, 3) Check your email for reset link, 4) Create new password.",
        "step_by_step_instructions"
    )
    customer_support.add_example(
        "Where is my order?",
        "To track your order: 1) Log into your account, 2) Go to 'Orders', "
        "3) Find your order number, 4) Click 'Track' to see current status.",
        "step_by_step_instructions"
    )
    customer_support.add_example(
        "How do I cancel my subscription?",
        "To cancel subscription: 1) Go to Account Settings, 2) Select 'Subscriptions', "
        "3) Choose subscription to cancel, 4) Confirm cancellation.",
        "step_by_step_instructions"
    )
    
    # Add patterns
    customer_support.add_pattern("Provide step-by-step instructions")
    customer_support.add_pattern("Be clear and concise")
    customer_support.add_pattern("Offer alternative solutions")
    customer_support.add_pattern("Include troubleshooting tips")
    
    # Add concepts
    customer_support.add_concept("account_management", "User account operations")
    customer_support.add_concept("order_processing", "Order lifecycle and tracking")
    
    agent.add_source_domain(customer_support)
    
    # Example 1: Transfer to Technical Support
    print("\n" + "="*70)
    print("Example 1: Transfer to Technical Support Domain")
    print("="*70)
    
    result1 = agent.transfer_and_respond(
        target_task="Help users troubleshoot technical issues",
        target_domain="technical_support",
        query="How do I fix a blue screen error?"
    )
    
    # Example 2: Transfer to Medical Assistant
    print("\n" + "="*70)
    print("Example 2: Transfer to Medical Assistant Domain")
    print("="*70)
    
    result2 = agent.transfer_and_respond(
        target_task="Provide medical information and guidance",
        target_domain="medical_assistant",
        query="How do I check my blood pressure at home?"
    )
    
    # Example 3: Compare with and without transfer
    print("\n" + "="*70)
    print("Example 3: Comparing With/Without Transfer")
    print("="*70)
    
    comparison = agent.compare_with_without_transfer(
        target_task="Help users with cooking recipes",
        target_domain="cooking_assistant",
        query="How do I make homemade pasta?"
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Transfer Learning Pattern demonstrated:

Key Features:
1. Source Domain Selection - Identify most relevant source knowledge
2. Knowledge Adaptation - Adapt examples and patterns to new domain
3. Pattern Transfer - Transfer learned patterns across domains
4. Few-Shot Enhancement - Use adapted examples for better performance
5. Comparative Evaluation - Compare with/without transfer

Transfer Mechanisms:
- Example adaptation across domains
- Pattern and structure transfer
- Concept mapping between domains
- Meta-learning of general strategies
- Domain-specific fine-tuning

Applications:
- Multi-domain chatbots (customer → technical support)
- Cross-lingual NLP (English → Spanish)
- Domain adaptation (medical → legal documentation)
- Few-shot learning in new domains
- Skill transfer in robotics

Benefits:
- Faster adaptation to new domains
- Better performance with limited data
- Leverages existing knowledge
- Reduces training time and data needs
- Improves generalization

The transfer learning pattern enables agents to leverage knowledge from
related domains, significantly improving performance on new tasks with
limited domain-specific training.
    """)


if __name__ == "__main__":
    demonstrate_transfer_learning()
