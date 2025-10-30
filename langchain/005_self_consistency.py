"""
Pattern 005: Self-Consistency

Description:
    Samples multiple reasoning paths for the same question and selects
    the most consistent answer through majority voting or consistency scoring.
    This approach improves reliability by generating diverse solutions and
    aggregating them.

Key Concepts:
    - Multiple Sampling: Generate several independent reasoning chains
    - Diverse Reasoning: Use temperature or prompt variations
    - Aggregation: Majority voting or consistency analysis
    - Answer Selection: Choose most frequent or highest confidence answer

Use Cases:
    - Math problems with multiple solution methods
    - Questions with ambiguous reasoning paths
    - Improving reliability of complex reasoning
    - Reducing model hallucinations

LangChain Implementation:
    Multiple chain invocations with temperature variation and result aggregation.
"""

import os
from typing import List, Dict, Any, Tuple
from collections import Counter
import re
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


class SelfConsistencyAgent:
    """Agent that uses self-consistency for reliable reasoning."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", num_samples: int = 5):
        """
        Initialize the Self-Consistency agent.
        
        Args:
            model_name: Name of the OpenAI model to use
            num_samples: Number of reasoning paths to sample
        """
        self.model_name = model_name
        self.num_samples = num_samples
        self.base_temperature = 0.7
        
    def _create_llm(self, temperature: float) -> ChatOpenAI:
        """Create an LLM instance with specified temperature."""
        return ChatOpenAI(
            model=self.model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def generate_reasoning_paths(self, question: str, context: str = "") -> List[Dict[str, str]]:
        """
        Generate multiple diverse reasoning paths for a question.
        
        Args:
            question: The question to answer
            context: Optional context information
            
        Returns:
            List of dictionaries with 'reasoning' and 'answer' keys
        """
        # Base prompt for chain-of-thought reasoning
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer the question with clear step-by-step reasoning.
Show your work and explain each step.
At the end, clearly state your final answer."""),
            ("human", "{context}\n\nQuestion: {question}\n\nLet's think step by step:")
        ])
        
        reasoning_paths = []
        
        # Generate multiple samples with varying temperatures
        for i in range(self.num_samples):
            # Vary temperature for diversity
            temperature = self.base_temperature + (i - self.num_samples // 2) * 0.1
            temperature = max(0.1, min(1.0, temperature))
            
            llm = self._create_llm(temperature)
            chain = prompt | llm | StrOutputParser()
            
            try:
                response = chain.invoke({
                    "question": question,
                    "context": context if context else "No additional context."
                })
                
                # Extract answer from response
                answer = self._extract_answer(response)
                
                reasoning_paths.append({
                    "reasoning": response,
                    "answer": answer,
                    "temperature": temperature
                })
                
            except Exception as e:
                print(f"Warning: Sample {i+1} failed: {str(e)}")
                continue
        
        return reasoning_paths
    
    def _extract_answer(self, reasoning: str) -> str:
        """Extract the final answer from reasoning text."""
        # Look for common answer patterns
        patterns = [
            r"final answer is:?\s*(.+?)(?:\n|$)",
            r"answer is:?\s*(.+?)(?:\n|$)",
            r"therefore,?\s+(?:the answer is )?\s*(.+?)(?:\n|$)",
            r"thus,?\s+(.+?)(?:\n|$)",
            r"so,?\s+(?:the answer is )?\s*(.+?)(?:\n|$)",
        ]
        
        reasoning_lower = reasoning.lower()
        for pattern in patterns:
            match = re.search(pattern, reasoning_lower, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Clean up the answer
                answer = re.sub(r'[*_]', '', answer)  # Remove markdown
                return answer
        
        # If no pattern matches, take last non-empty line
        lines = [line.strip() for line in reasoning.split('\n') if line.strip()]
        return lines[-1] if lines else reasoning[:100]
    
    def aggregate_by_majority_vote(self, reasoning_paths: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Aggregate answers using majority voting.
        
        Args:
            reasoning_paths: List of reasoning paths with answers
            
        Returns:
            Dictionary with final answer and aggregation statistics
        """
        answers = [path['answer'] for path in reasoning_paths]
        
        # Normalize answers for comparison (lowercase, strip punctuation)
        normalized_answers = []
        for ans in answers:
            normalized = ans.lower().strip().rstrip('.')
            normalized_answers.append(normalized)
        
        # Count occurrences
        answer_counts = Counter(normalized_answers)
        most_common_answer, count = answer_counts.most_common(1)[0]
        
        # Find original form of the answer
        original_answer = next(
            (answers[i] for i, norm in enumerate(normalized_answers) 
             if norm == most_common_answer),
            most_common_answer
        )
        
        return {
            "final_answer": original_answer,
            "confidence": count / len(answers),
            "vote_count": count,
            "total_samples": len(answers),
            "all_answers": answer_counts,
            "agreement_rate": count / len(answers)
        }
    
    def aggregate_by_consistency_score(self, reasoning_paths: List[Dict[str, str]], 
                                       question: str) -> Dict[str, Any]:
        """
        Aggregate answers using consistency scoring via LLM.
        
        Args:
            reasoning_paths: List of reasoning paths with answers
            question: Original question
            
        Returns:
            Dictionary with final answer and consistency scores
        """
        # Use LLM to evaluate consistency of each answer with others
        answers_text = "\n".join([
            f"{i+1}. {path['answer']}" 
            for i, path in enumerate(reasoning_paths)
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are evaluating multiple answers to the same question.
Identify which answer is most consistent with the majority of other answers.
Consider semantic similarity, not just exact matches.

Respond with the number (1, 2, 3, etc.) of the most consistent answer."""),
            ("human", """Question: {question}

Answers:
{answers}

Most consistent answer number:""")
        ])
        
        llm = self._create_llm(0.3)  # Low temperature for consistent evaluation
        chain = prompt | llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "question": question,
                "answers": answers_text
            })
            
            # Extract number
            match = re.search(r'\d+', response)
            if match:
                index = int(match.group()) - 1
                if 0 <= index < len(reasoning_paths):
                    selected_answer = reasoning_paths[index]['answer']
                else:
                    # Fallback to majority vote
                    return self.aggregate_by_majority_vote(reasoning_paths)
            else:
                return self.aggregate_by_majority_vote(reasoning_paths)
            
            return {
                "final_answer": selected_answer,
                "method": "consistency_scoring",
                "selected_index": index + 1,
                "total_samples": len(reasoning_paths)
            }
            
        except Exception as e:
            print(f"Consistency scoring failed: {e}. Falling back to majority vote.")
            return self.aggregate_by_majority_vote(reasoning_paths)
    
    def solve_with_self_consistency(self, question: str, context: str = "",
                                    method: str = "majority_vote") -> Dict[str, Any]:
        """
        Solve a question using self-consistency.
        
        Args:
            question: The question to answer
            context: Optional context
            method: Aggregation method ('majority_vote' or 'consistency_score')
            
        Returns:
            Dictionary with solution and metadata
        """
        print(f"\nQuestion: {question}\n")
        print(f"Generating {self.num_samples} reasoning paths...\n")
        
        # Generate multiple reasoning paths
        reasoning_paths = self.generate_reasoning_paths(question, context)
        
        if not reasoning_paths:
            return {
                "question": question,
                "final_answer": "Unable to generate reasoning paths",
                "error": "All samples failed"
            }
        
        print(f"Generated {len(reasoning_paths)} paths\n")
        
        # Show individual reasoning paths
        for i, path in enumerate(reasoning_paths, 1):
            print(f"Path {i} (temp={path['temperature']:.2f}):")
            print(f"  Answer: {path['answer']}")
            print()
        
        # Aggregate results
        if method == "majority_vote":
            result = self.aggregate_by_majority_vote(reasoning_paths)
        else:
            result = self.aggregate_by_consistency_score(reasoning_paths, question)
        
        result["question"] = question
        result["reasoning_paths"] = len(reasoning_paths)
        
        return result


def demonstrate_self_consistency():
    """Demonstrates the Self-Consistency pattern."""
    
    print("=" * 80)
    print("PATTERN 005: Self-Consistency")
    print("=" * 80)
    print()
    print("Self-Consistency improves answer reliability by:")
    print("1. Generating multiple independent reasoning paths")
    print("2. Sampling with different temperatures for diversity")
    print("3. Aggregating answers via majority voting or consistency scoring")
    print()
    
    # Create agent
    agent = SelfConsistencyAgent(num_samples=5)
    
    # Test cases
    test_cases = [
        {
            "question": "If a train travels 120 miles in 2 hours, then travels 180 miles in 3 hours, what is its average speed for the entire journey?",
            "context": "",
            "method": "majority_vote"
        },
        {
            "question": "A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "context": "",
            "method": "majority_vote"
        },
        {
            "question": "Is it ethical for AI to make medical diagnoses?",
            "context": "Consider patient safety, accuracy, and doctor-patient relationships.",
            "method": "consistency_score"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Example {i}: {test_case['method'].replace('_', ' ').title()}")
        print('=' * 80)
        
        try:
            result = agent.solve_with_self_consistency(
                question=test_case['question'],
                context=test_case['context'],
                method=test_case['method']
            )
            
            print(f"\n{'=' * 80}")
            print("FINAL RESULT")
            print('=' * 80)
            print(f"\nFinal Answer: {result['final_answer']}")
            
            if 'confidence' in result:
                print(f"Confidence: {result['confidence']:.1%}")
                print(f"Agreement Rate: {result['agreement_rate']:.1%}")
                print(f"Vote Distribution: {dict(result['all_answers'])}")
            
            print(f"Reasoning Paths: {result['reasoning_paths']}/{agent.num_samples}")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SELF-CONSISTENCY PATTERN DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Techniques Demonstrated:")
    print("1. Multiple Sampling: Generated 5 diverse reasoning paths")
    print("2. Temperature Variation: Used different temperatures for diversity")
    print("3. Majority Voting: Aggregated answers by frequency")
    print("4. Consistency Scoring: LLM-based semantic consistency evaluation")
    print("5. Confidence Metrics: Agreement rate and vote distribution")
    print()
    print("Benefits:")
    print("- Improved reliability through multiple reasoning paths")
    print("- Reduced impact of outlier/hallucinated answers")
    print("- Confidence estimation via agreement metrics")
    print("- Works with any reasoning task")
    print()
    print("LangChain Components Used:")
    print("- ChatPromptTemplate: Structures reasoning prompts")
    print("- StrOutputParser: Parses LLM outputs")
    print("- Temperature variation: Creates diverse samples")
    print("- Multiple chain invocations: Parallel reasoning paths")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    demonstrate_self_consistency()
