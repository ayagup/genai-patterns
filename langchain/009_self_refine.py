"""
Pattern 009: Self-Refine

Description:
    Iteratively improves outputs through self-critique and refinement.
    The agent generates an initial output, evaluates it, identifies weaknesses,
    and produces an improved version. This cycle repeats until quality threshold
    is met or max iterations reached.

Key Concepts:
    - Initial Generation: Create first draft output
    - Self-Critique: Analyze weaknesses and areas for improvement
    - Refinement: Generate improved version based on critique
    - Iteration: Repeat until satisfactory or max iterations
    - Quality Tracking: Monitor improvement across iterations

Benefits:
    - Higher quality outputs through iteration
    - Self-improvement without external feedback
    - Transparency in refinement process
    - Handles complex generation tasks

Use Cases:
    - Content creation (writing, code, reports)
    - Problem-solving with iterative improvement
    - Design tasks requiring multiple revisions
    - Any task benefiting from refinement

LangChain Implementation:
    Chained prompts for generation, critique, and refinement with iteration control.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


class IterationStatus(Enum):
    """Status of refinement iteration."""
    INITIAL = "initial"
    REFINING = "refining"
    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations_reached"


@dataclass
class Critique:
    """Represents a critique of generated output."""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    score: float = 0.0  # Quality score 0-1


@dataclass
class RefinementIteration:
    """Represents one iteration of the refinement process."""
    iteration: int
    output: str
    critique: Optional[Critique] = None
    status: IterationStatus = IterationStatus.INITIAL
    
    def __str__(self):
        return f"Iteration {self.iteration}: {self.status.value} (score: {self.critique.score if self.critique else 'N/A'})"


class SelfRefineAgent:
    """Agent that uses self-refinement to improve outputs."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", 
                 max_iterations: int = 3,
                 quality_threshold: float = 0.8):
        """
        Initialize the Self-Refine agent.
        
        Args:
            model_name: Name of the OpenAI model
            max_iterations: Maximum refinement iterations
            quality_threshold: Stop if quality score exceeds this
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.critic_llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,  # Lower temperature for more consistent critique
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.history: List[RefinementIteration] = []
    
    def generate_initial(self, task: str, context: str = "") -> str:
        """
        Generate initial output for the task.
        
        Args:
            task: The task description
            context: Additional context
            
        Returns:
            Initial generated output
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate a response to the given task.
Focus on creating a solid first draft. It doesn't need to be perfect yet."""),
            ("human", """{context}

Task: {task}

Response:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        output = chain.invoke({
            "task": task,
            "context": context if context else "No additional context."
        })
        
        return output.strip()
    
    def critique(self, task: str, output: str, iteration: int) -> Critique:
        """
        Generate critique of current output.
        
        Args:
            task: Original task
            output: Current output to critique
            iteration: Current iteration number
            
        Returns:
            Critique object with feedback
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a constructive critic. Evaluate the response to the task.

Provide:
1. Strengths: What works well (2-3 points)
2. Weaknesses: What could be improved (2-3 points)
3. Suggestions: Specific improvements to make (2-3 points)
4. Score: Overall quality score from 0.0 to 1.0

Format your response as:
STRENGTHS:
- [strength 1]
- [strength 2]

WEAKNESSES:
- [weakness 1]
- [weakness 2]

SUGGESTIONS:
- [suggestion 1]
- [suggestion 2]

SCORE: [0.0-1.0]"""),
            ("human", """Task: {task}

Current Response (Iteration {iteration}):
{output}

Critique:""")
        ])
        
        chain = prompt | self.critic_llm | StrOutputParser()
        critique_text = chain.invoke({
            "task": task,
            "output": output,
            "iteration": iteration
        })
        
        # Parse critique
        critique = Critique()
        
        sections = {
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }
        
        current_section = None
        for line in critique_text.split('\n'):
            line = line.strip()
            
            if line.upper().startswith('STRENGTHS:'):
                current_section = "strengths"
            elif line.upper().startswith('WEAKNESSES:'):
                current_section = "weaknesses"
            elif line.upper().startswith('SUGGESTIONS:'):
                current_section = "suggestions"
            elif line.upper().startswith('SCORE:'):
                try:
                    score_str = line.split(':', 1)[1].strip()
                    critique.score = float(score_str)
                except:
                    critique.score = 0.5
            elif line.startswith('-') and current_section:
                sections[current_section].append(line.lstrip('- ').strip())
        
        critique.strengths = sections["strengths"]
        critique.weaknesses = sections["weaknesses"]
        critique.suggestions = sections["suggestions"]
        
        # Ensure score is set
        if critique.score == 0.0 and (critique.strengths or not critique.weaknesses):
            critique.score = 0.6  # Default reasonable score
        
        return critique
    
    def refine(self, task: str, current_output: str, critique: Critique) -> str:
        """
        Generate refined version based on critique.
        
        Args:
            task: Original task
            current_output: Current output to refine
            critique: Critique with improvement suggestions
            
        Returns:
            Refined output
        """
        weaknesses_text = "\n".join([f"- {w}" for w in critique.weaknesses])
        suggestions_text = "\n".join([f"- {s}" for s in critique.suggestions])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Improve the response based on the critique provided.
Address the weaknesses and implement the suggestions while maintaining the strengths.
Produce a refined, higher-quality version."""),
            ("human", """Task: {task}

Current Response:
{current_output}

Weaknesses to Address:
{weaknesses}

Suggestions to Implement:
{suggestions}

Refined Response:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        refined = chain.invoke({
            "task": task,
            "current_output": current_output,
            "weaknesses": weaknesses_text,
            "suggestions": suggestions_text
        })
        
        return refined.strip()
    
    def refine_iteratively(self, task: str, context: str = "",
                          verbose: bool = True) -> Dict[str, Any]:
        """
        Perform iterative refinement on the task.
        
        Args:
            task: The task to complete
            context: Additional context
            verbose: Whether to print progress
            
        Returns:
            Dictionary with final output and refinement history
        """
        if verbose:
            print(f"\nTask: {task}\n")
            print("="*60)
            print("ITERATION 1: Initial Generation")
            print("="*60)
        
        # Generate initial output
        initial_output = self.generate_initial(task, context)
        
        iteration = RefinementIteration(
            iteration=1,
            output=initial_output,
            status=IterationStatus.INITIAL
        )
        self.history.append(iteration)
        
        if verbose:
            print(f"\nGenerated:\n{initial_output[:200]}..." if len(initial_output) > 200 else f"\nGenerated:\n{initial_output}")
        
        current_output = initial_output
        
        # Refinement loop
        for i in range(2, self.max_iterations + 2):
            if verbose:
                print(f"\n{'='*60}")
                print(f"ITERATION {i}: Critique & Refine")
                print("="*60)
            
            # Critique current output
            critique = self.critique(task, current_output, i - 1)
            
            # Update previous iteration with critique
            self.history[-1].critique = critique
            
            if verbose:
                print(f"\nCritique:")
                print(f"  Quality Score: {critique.score:.2f}")
                if critique.strengths:
                    print(f"  Strengths: {len(critique.strengths)} identified")
                if critique.weaknesses:
                    print(f"  Weaknesses: {len(critique.weaknesses)} identified")
                if critique.suggestions:
                    print(f"  Suggestions: {len(critique.suggestions)} provided")
            
            # Check if quality threshold met
            if critique.score >= self.quality_threshold:
                self.history[-1].status = IterationStatus.COMPLETED
                if verbose:
                    print(f"\n✓ Quality threshold ({self.quality_threshold}) met!")
                break
            
            # Check max iterations
            if i > self.max_iterations:
                self.history[-1].status = IterationStatus.MAX_ITERATIONS
                if verbose:
                    print(f"\n⚠ Max iterations ({self.max_iterations}) reached")
                break
            
            # Refine output
            if verbose:
                print(f"\nRefining...")
            
            refined_output = self.refine(task, current_output, critique)
            
            iteration = RefinementIteration(
                iteration=i,
                output=refined_output,
                status=IterationStatus.REFINING
            )
            self.history.append(iteration)
            
            if verbose:
                print(f"\nRefined:\n{refined_output[:200]}..." if len(refined_output) > 200 else f"\nRefined:\n{refined_output}")
            
            current_output = refined_output
        
        # Final critique
        if self.history[-1].critique is None:
            final_critique = self.critique(task, current_output, len(self.history))
            self.history[-1].critique = final_critique
        
        return {
            "task": task,
            "final_output": current_output,
            "iterations": len(self.history),
            "final_score": self.history[-1].critique.score if self.history[-1].critique else 0.0,
            "history": self.history,
            "status": self.history[-1].status.value
        }
    
    def get_improvement_summary(self) -> str:
        """Generate summary of improvements across iterations."""
        if len(self.history) < 2:
            return "Not enough iterations to compare."
        
        initial_score = self.history[0].critique.score if self.history[0].critique else 0.0
        final_score = self.history[-1].critique.score if self.history[-1].critique else 0.0
        
        improvement = final_score - initial_score
        improvement_pct = (improvement / max(initial_score, 0.1)) * 100
        
        summary = f"""
Improvement Summary:
  Initial Score: {initial_score:.2f}
  Final Score: {final_score:.2f}
  Improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)
  Total Iterations: {len(self.history)}
"""
        return summary


def demonstrate_self_refine():
    """Demonstrates the Self-Refine pattern."""
    
    print("=" * 80)
    print("PATTERN 009: Self-Refine")
    print("=" * 80)
    print()
    print("Self-Refine iteratively improves outputs through:")
    print("1. Initial Generation: Create first draft")
    print("2. Self-Critique: Analyze strengths and weaknesses")
    print("3. Refinement: Generate improved version")
    print("4. Iteration: Repeat until quality threshold met")
    print()
    
    # Create agent
    agent = SelfRefineAgent(max_iterations=3, quality_threshold=0.85)
    
    # Test tasks
    tasks = [
        {
            "task": "Write a brief explanation of quantum entanglement for a general audience",
            "context": "Make it accessible but accurate, around 100 words"
        },
        {
            "task": "Create a function to calculate the Fibonacci sequence efficiently",
            "context": "Use Python, include docstring and handle edge cases"
        }
    ]
    
    for idx, test_case in enumerate(tasks, 1):
        print(f"\n{'='*80}")
        print(f"Example {idx}")
        print('='*80)
        
        try:
            # Reset history
            agent.history = []
            
            result = agent.refine_iteratively(
                task=test_case['task'],
                context=test_case['context'],
                verbose=True
            )
            
            print(f"\n\n{'='*80}")
            print("FINAL RESULT")
            print('='*80)
            print(f"\nFinal Output:\n{result['final_output']}")
            print(f"\n{agent.get_improvement_summary()}")
            print(f"Status: {result['status']}")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("SELF-REFINE PATTERN DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("1. Iterative Improvement: Multiple refinement cycles")
    print("2. Self-Critique: Autonomous evaluation of outputs")
    print("3. Targeted Refinement: Addressing specific weaknesses")
    print("4. Quality Tracking: Score-based improvement monitoring")
    print("5. Convergence: Stops when threshold met or max iterations")
    print()
    print("Advantages:")
    print("- Higher quality outputs through iteration")
    print("- No external feedback required")
    print("- Transparent improvement process")
    print("- Handles complex generation tasks")
    print()
    print("When to use Self-Refine:")
    print("- Content creation (writing, code, reports)")
    print("- Tasks benefiting from multiple revisions")
    print("- When quality is more important than speed")
    print("- Complex generation requiring refinement")
    print()
    print("LangChain Components Used:")
    print("- ChatPromptTemplate: Generation, critique, and refinement prompts")
    print("- StrOutputParser: Parse LLM outputs")
    print("- Separate LLMs: Different temperatures for generation vs critique")
    print("- Iteration control: Quality threshold and max iterations")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    demonstrate_self_refine()
