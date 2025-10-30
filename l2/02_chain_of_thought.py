"""
Pattern 2: Chain-of-Thought (CoT)
Agent breaks down complex problems into intermediate reasoning steps.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from typing import Dict, List

class ChainOfThoughtPattern:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    def zero_shot_cot(self, question: str) -> str:
        """Zero-shot Chain of Thought"""
        template = """Question: {question}

Let's think step by step to solve this problem."""
        
        prompt = PromptTemplate(template=template, input_variables=["question"])
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(question=question)
    
    def few_shot_cot(self, question: str) -> str:
        """Few-shot Chain of Thought with examples"""
        examples = [
            {
                "question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
                "answer": """Let's think step by step:
1. Roger starts with 5 tennis balls
2. He buys 2 cans, each with 3 balls
3. New balls = 2 cans × 3 balls/can = 6 balls
4. Total = 5 + 6 = 11 balls
Answer: 11 tennis balls"""
            },
            {
                "question": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?",
                "answer": """Let's think step by step:
1. Start with 23 apples
2. Used 20 apples: 23 - 20 = 3 apples left
3. Bought 6 more: 3 + 6 = 9 apples
Answer: 9 apples"""
            }
        ]
        
        example_template = """Question: {question}
{answer}"""
        
        example_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template=example_template
        )
        
        prefix = "Solve the following problems using step-by-step reasoning:\n\n"
        suffix = "\nQuestion: {question}\nLet's think step by step:"
        
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["question"]
        )
        
        chain = LLMChain(llm=self.llm, prompt=few_shot_prompt)
        return chain.run(question=question)
    
    def auto_cot(self, question: str, num_reasoning_paths: int = 3) -> str:
        """Auto Chain of Thought - generates multiple reasoning paths"""
        template = """Question: {question}

Generate {num_paths} different step-by-step reasoning paths to solve this problem.

Path 1:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "num_paths"]
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(question=question, num_paths=num_reasoning_paths)

if __name__ == "__main__":
    cot = ChainOfThoughtPattern()
    
    question = "If a train travels 120 miles in 2 hours, how far will it travel in 5 hours at the same speed?"
    
    print("=== Zero-Shot CoT ===")
    print(cot.zero_shot_cot(question))
    
    print("\n=== Few-Shot CoT ===")
    print(cot.few_shot_cot(question))
    
    print("\n=== Auto-CoT ===")
    print(cot.auto_cot(question))
