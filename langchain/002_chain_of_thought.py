"""
Pattern 002: Chain-of-Thought (CoT)

Description:
    Agent breaks down complex problems into intermediate reasoning steps.
    CoT prompting encourages the model to explain its reasoning process
    step-by-step before arriving at a final answer.

Variants:
    - Zero-shot CoT: "Let's think step by step..."
    - Few-shot CoT: Provides examples with reasoning steps
    - Auto-CoT: Automatically generates reasoning examples

Use Cases:
    - Mathematical reasoning
    - Logical puzzles
    - Complex analysis tasks
    - Multi-step problem solving

LangChain Implementation:
    Uses structured prompts and output parsers to ensure step-by-step reasoning.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()


# Few-shot examples demonstrating Chain-of-Thought reasoning
COT_EXAMPLES = [
    {
        "question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
        "reasoning": """Let me think step by step:
1. Roger starts with 5 tennis balls
2. He buys 2 cans of tennis balls
3. Each can contains 3 tennis balls
4. So he gets 2 × 3 = 6 new tennis balls
5. Total tennis balls = 5 (original) + 6 (new) = 11

Therefore, Roger has 11 tennis balls now.""",
        "answer": "11"
    },
    {
        "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "reasoning": """Let me think step by step:
1. 5 machines make 5 widgets in 5 minutes
2. This means each machine makes 1 widget in 5 minutes
3. The machines work in parallel (simultaneously)
4. So 100 machines working in parallel would still take 5 minutes
5. In 5 minutes, 100 machines would make 100 widgets (1 widget per machine)

Therefore, it would take 5 minutes.""",
        "answer": "5 minutes"
    }
]


class ChainOfThoughtAgent:
    """Agent that uses Chain-of-Thought reasoning."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0):
        """Initialize the Chain-of-Thought agent."""
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def zero_shot_cot(self, question: str) -> Dict[str, str]:
        """
        Zero-shot Chain-of-Thought: Uses "Let's think step by step" prompt.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary with reasoning and answer
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that solves problems step-by-step."),
            ("human", "{question}\n\nLet's think step by step:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        reasoning = chain.invoke({"question": question})
        
        # Extract final answer
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Based on the reasoning provided, give only the final answer concisely."),
            ("human", "Reasoning:\n{reasoning}\n\nFinal answer:")
        ])
        
        answer_chain = answer_prompt | self.llm | StrOutputParser()
        answer = answer_chain.invoke({"reasoning": reasoning})
        
        return {
            "question": question,
            "reasoning": reasoning,
            "answer": answer.strip()
        }
    
    def few_shot_cot(self, question: str, examples: List[Dict] = None) -> Dict[str, str]:
        """
        Few-shot Chain-of-Thought: Provides examples with reasoning steps.
        
        Args:
            question: The question to answer
            examples: List of example question-reasoning-answer triplets
            
        Returns:
            Dictionary with reasoning and answer
        """
        if examples is None:
            examples = COT_EXAMPLES
        
        # Create few-shot prompt template
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{question}"),
            ("ai", "{reasoning}\n\nAnswer: {answer}")
        ])
        
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples
        )
        
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that solves problems step-by-step. Show your reasoning process clearly."),
            few_shot_prompt,
            ("human", "{question}")
        ])
        
        chain = final_prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"question": question})
        
        # Parse response to separate reasoning and answer
        lines = response.strip().split('\n')
        answer_line = [line for line in lines if line.strip().startswith("Answer:")]
        
        if answer_line:
            answer = answer_line[0].replace("Answer:", "").strip()
            reasoning = response.replace(answer_line[0], "").strip()
        else:
            # If no explicit answer line, last line is the answer
            reasoning = '\n'.join(lines[:-1])
            answer = lines[-1].strip()
        
        return {
            "question": question,
            "reasoning": reasoning,
            "answer": answer
        }
    
    def auto_cot(self, question: str) -> Dict[str, str]:
        """
        Auto-CoT: Automatically generates reasoning with structured steps.
        
        Args:
            question: The question to answer
            
        Returns:
            Dictionary with reasoning and answer
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that solves problems systematically.
For each problem:
1. Break it down into clear steps
2. Number each step
3. Show your work clearly
4. Provide a final answer

Format your response as:
Step 1: [description]
Step 2: [description]
...
Final Answer: [answer]"""),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"question": question})
        
        # Parse response
        lines = response.strip().split('\n')
        final_answer_line = [line for line in lines if "final answer" in line.lower()]
        
        if final_answer_line:
            answer = final_answer_line[0].split(':', 1)[-1].strip()
            reasoning = '\n'.join([line for line in lines if line not in final_answer_line])
        else:
            reasoning = response
            answer = "Unable to extract final answer"
        
        return {
            "question": question,
            "reasoning": reasoning,
            "answer": answer
        }


def demonstrate_cot_pattern():
    """Demonstrates the Chain-of-Thought pattern with various approaches."""
    
    print("=" * 80)
    print("PATTERN 002: Chain-of-Thought (CoT)")
    print("=" * 80)
    print()
    
    # Create CoT agent
    agent = ChainOfThoughtAgent()
    
    # Test questions
    test_questions = [
        "A farmer has 17 sheep and all but 9 die. How many sheep are left?",
        "If a train travels 120 km in 2 hours, how far will it travel in 5 hours at the same speed?",
        "Lisa has 10 apples. She gives 3 to her friend and buys 5 more. Her friend gives back 1 apple. How many apples does Lisa have now?"
    ]
    
    # Demonstrate Zero-shot CoT
    print("\n" + "=" * 80)
    print("ZERO-SHOT CHAIN-OF-THOUGHT")
    print("=" * 80)
    
    for i, question in enumerate(test_questions[:2], 1):
        print(f"\n{'- ' * 40}")
        print(f"Question {i}: {question}")
        print('- ' * 40)
        
        try:
            result = agent.zero_shot_cot(question)
            print(f"\nReasoning:\n{result['reasoning']}")
            print(f"\n✓ Answer: {result['answer']}")
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
    
    # Demonstrate Few-shot CoT
    print("\n\n" + "=" * 80)
    print("FEW-SHOT CHAIN-OF-THOUGHT")
    print("=" * 80)
    
    for i, question in enumerate(test_questions[1:], 1):
        print(f"\n{'- ' * 40}")
        print(f"Question {i}: {question}")
        print('- ' * 40)
        
        try:
            result = agent.few_shot_cot(question)
            print(f"\nReasoning:\n{result['reasoning']}")
            print(f"\n✓ Answer: {result['answer']}")
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
    
    # Demonstrate Auto-CoT
    print("\n\n" + "=" * 80)
    print("AUTO-COT (STRUCTURED STEPS)")
    print("=" * 80)
    
    question = test_questions[2]
    print(f"\n{'- ' * 40}")
    print(f"Question: {question}")
    print('- ' * 40)
    
    try:
        result = agent.auto_cot(question)
        print(f"\nReasoning:\n{result['reasoning']}")
        print(f"\n✓ Answer: {result['answer']}")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
    
    # Summary
    print("\n\n" + "=" * 80)
    print("CHAIN-OF-THOUGHT PATTERN DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("CoT Variants Demonstrated:")
    print("1. Zero-shot CoT: 'Let's think step by step' prompting")
    print("2. Few-shot CoT: Examples with reasoning provided")
    print("3. Auto-CoT: Structured step-by-step format")
    print()
    print("Key Benefits:")
    print("- Improved accuracy on complex tasks")
    print("- Transparent reasoning process")
    print("- Better error detection")
    print("- Enhanced interpretability")
    print()
    print("LangChain Components Used:")
    print("- ChatPromptTemplate: Structures prompts")
    print("- FewShotChatMessagePromptTemplate: Manages few-shot examples")
    print("- StrOutputParser: Parses LLM outputs")
    print("- Chain composition: Connects components")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    demonstrate_cot_pattern()
