"""
Pattern 037: Prompt Chaining

Description:
    Prompt Chaining links multiple prompts sequentially where the output of one
    prompt feeds into the next. This enables complex multi-step processing by
    breaking down tasks into smaller, manageable stages with clear data flow
    between steps.

Components:
    - Chain Nodes: Individual prompts in the sequence
    - Data Flow: Mechanism for passing outputs to inputs
    - Transformation Functions: Process data between steps
    - Error Handling: Manages failures in the chain
    - Chain Orchestrator: Coordinates execution

Use Cases:
    - Multi-step document processing
    - Complex data transformation pipelines
    - Iterative content refinement
    - Research and analysis workflows
    - Code generation and testing
    - Multi-stage decision making

LangChain Implementation:
    Uses LCEL (LangChain Expression Language) for composable chains and
    RunnableSequence for sequential prompt execution with data flow.
"""

import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass
class ChainStep:
    """A step in the prompt chain."""
    name: str
    prompt: ChatPromptTemplate
    output_parser: Any
    transform: Optional[Callable] = None  # Transform output before next step
    description: str = ""


@dataclass
class ChainResult:
    """Result from executing a prompt chain."""
    final_output: Any
    intermediate_outputs: Dict[str, Any]
    execution_time: float
    steps_executed: int
    success: bool
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class PromptChain:
    """
    Orchestrates sequential execution of prompt chains.
    
    Features:
    - Sequential prompt execution
    - Data flow between steps
    - Intermediate result storage
    - Error handling and recovery
    - Execution tracking
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        temperature: float = 0.7
    ):
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        self.steps: List[ChainStep] = []
    
    def add_step(
        self,
        name: str,
        prompt: ChatPromptTemplate,
        output_parser: Any = None,
        transform: Optional[Callable] = None,
        description: str = ""
    ):
        """Add a step to the chain."""
        if output_parser is None:
            output_parser = StrOutputParser()
        
        step = ChainStep(
            name=name,
            prompt=prompt,
            output_parser=output_parser,
            transform=transform,
            description=description
        )
        self.steps.append(step)
        return self
    
    def execute(self, initial_input: Dict[str, Any]) -> ChainResult:
        """
        Execute the prompt chain sequentially.
        
        Args:
            initial_input: Initial input dictionary
            
        Returns:
            ChainResult with final output and intermediate results
        """
        start_time = datetime.now()
        intermediate_outputs = {}
        current_data = initial_input.copy()
        
        try:
            for i, step in enumerate(self.steps):
                print(f"\n[Step {i+1}/{len(self.steps)}] {step.name}")
                if step.description:
                    print(f"Description: {step.description}")
                
                # Build and execute chain for this step
                chain = step.prompt | self.llm | step.output_parser
                
                # Execute with current data
                output = chain.invoke(current_data)
                
                # Store intermediate result
                intermediate_outputs[step.name] = output
                
                # Apply transformation if provided
                if step.transform:
                    output = step.transform(output)
                
                # Update current_data for next step
                # Merge output into data for next step
                if isinstance(output, dict):
                    current_data.update(output)
                else:
                    current_data[step.name] = output
                
                print(f"Output preview: {str(output)[:150]}...")
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return ChainResult(
                final_output=output,
                intermediate_outputs=intermediate_outputs,
                execution_time=execution_time,
                steps_executed=len(self.steps),
                success=True
            )
        
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return ChainResult(
                final_output=None,
                intermediate_outputs=intermediate_outputs,
                execution_time=execution_time,
                steps_executed=len(intermediate_outputs),
                success=False,
                error=str(e)
            )


class PromptChainLibrary:
    """
    Library of pre-built prompt chains for common tasks.
    """
    
    @staticmethod
    def create_content_generation_chain(llm: ChatOpenAI) -> PromptChain:
        """
        Chain for generating refined content through multiple stages.
        
        Steps:
        1. Generate outline
        2. Expand outline into draft
        3. Refine and improve draft
        4. Add conclusion
        """
        chain = PromptChain(llm=llm, temperature=0.7)
        
        # Step 1: Generate outline
        outline_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a content outliner. Create a detailed outline for the topic."),
            ("user", "Topic: {topic}\n\nCreate an outline with 3-5 main points.")
        ])
        chain.add_step(
            name="outline",
            prompt=outline_prompt,
            description="Generate content outline"
        )
        
        # Step 2: Expand into draft
        draft_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a content writer. Expand the outline into a full draft."),
            ("user", "Topic: {topic}\n\nOutline:\n{outline}\n\nWrite a comprehensive draft based on this outline.")
        ])
        chain.add_step(
            name="draft",
            prompt=draft_prompt,
            description="Expand outline into draft"
        )
        
        # Step 3: Refine draft
        refine_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an editor. Improve the draft's clarity, flow, and impact."),
            ("user", "Draft:\n{draft}\n\nRefine this draft to improve clarity and readability.")
        ])
        chain.add_step(
            name="refined",
            prompt=refine_prompt,
            description="Refine and improve draft"
        )
        
        # Step 4: Add conclusion
        conclusion_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a content finalizer. Add a strong conclusion."),
            ("user", "Content:\n{refined}\n\nAdd a compelling conclusion that summarizes key points.")
        ])
        chain.add_step(
            name="final",
            prompt=conclusion_prompt,
            description="Add conclusion"
        )
        
        return chain
    
    @staticmethod
    def create_code_generation_chain(llm: ChatOpenAI) -> PromptChain:
        """
        Chain for generating tested code through multiple stages.
        
        Steps:
        1. Design solution approach
        2. Generate code
        3. Generate tests
        4. Generate documentation
        """
        chain = PromptChain(llm=llm, temperature=0.3)
        
        # Step 1: Design approach
        design_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a software architect. Design a solution approach."),
            ("user", "Problem: {problem}\n\nDescribe the solution approach, key functions, and data structures.")
        ])
        chain.add_step(
            name="design",
            prompt=design_prompt,
            description="Design solution approach"
        )
        
        # Step 2: Generate code
        code_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a programmer. Implement the solution in Python."),
            ("user", "Problem: {problem}\n\nDesign:\n{design}\n\nImplement the solution with clean, well-structured code.")
        ])
        chain.add_step(
            name="code",
            prompt=code_prompt,
            description="Generate implementation"
        )
        
        # Step 3: Generate tests
        test_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a test engineer. Write comprehensive tests."),
            ("user", "Code:\n{code}\n\nWrite pytest tests covering normal cases, edge cases, and errors.")
        ])
        chain.add_step(
            name="tests",
            prompt=test_prompt,
            description="Generate tests"
        )
        
        # Step 4: Generate documentation
        docs_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a technical writer. Write clear documentation."),
            ("user", "Code:\n{code}\n\nWrite documentation including: overview, usage examples, and API reference.")
        ])
        chain.add_step(
            name="documentation",
            prompt=docs_prompt,
            description="Generate documentation"
        )
        
        return chain
    
    @staticmethod
    def create_research_chain(llm: ChatOpenAI) -> PromptChain:
        """
        Chain for conducting research through multiple stages.
        
        Steps:
        1. Identify key questions
        2. Research each question
        3. Synthesize findings
        4. Draw conclusions
        """
        chain = PromptChain(llm=llm, temperature=0.5)
        
        # Step 1: Identify questions
        questions_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research analyst. Identify key research questions."),
            ("user", "Topic: {topic}\n\nList 3-5 key questions to research about this topic.")
        ])
        chain.add_step(
            name="questions",
            prompt=questions_prompt,
            description="Identify research questions"
        )
        
        # Step 2: Research questions
        research_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a researcher. Provide detailed answers based on knowledge."),
            ("user", "Topic: {topic}\n\nQuestions:\n{questions}\n\nProvide detailed answers to each question.")
        ])
        chain.add_step(
            name="research",
            prompt=research_prompt,
            description="Research questions"
        )
        
        # Step 3: Synthesize
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an analyst. Synthesize the research into key insights."),
            ("user", "Research:\n{research}\n\nSynthesize the findings into 3-5 key insights.")
        ])
        chain.add_step(
            name="synthesis",
            prompt=synthesis_prompt,
            description="Synthesize findings"
        )
        
        # Step 4: Conclusions
        conclusion_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert. Draw actionable conclusions."),
            ("user", "Topic: {topic}\n\nInsights:\n{synthesis}\n\nDraw conclusions and provide recommendations.")
        ])
        chain.add_step(
            name="conclusions",
            prompt=conclusion_prompt,
            description="Draw conclusions"
        )
        
        return chain


def demonstrate_prompt_chaining():
    """
    Demonstrates prompt chaining with sequential multi-step workflows.
    """
    print("=" * 80)
    print("PROMPT CHAINING DEMONSTRATION")
    print("=" * 80)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    
    # Test 1: Content generation chain
    print("\n" + "=" * 80)
    print("Test 1: Content Generation Chain")
    print("=" * 80)
    print("\nSteps: Outline → Draft → Refine → Conclusion")
    
    content_chain = PromptChainLibrary.create_content_generation_chain(llm)
    
    result1 = content_chain.execute({
        "topic": "The Impact of Artificial Intelligence on Healthcare"
    })
    
    print("\n" + "-" * 80)
    print("CHAIN RESULTS")
    print("-" * 80)
    print(f"Success: {result1.success}")
    print(f"Steps Executed: {result1.steps_executed}")
    print(f"Execution Time: {result1.execution_time:.2f}s")
    
    print("\n" + "-" * 80)
    print("FINAL OUTPUT")
    print("-" * 80)
    print(result1.final_output[:500] + "..." if len(result1.final_output) > 500 else result1.final_output)
    
    # Test 2: Code generation chain
    print("\n" + "=" * 80)
    print("Test 2: Code Generation Chain")
    print("=" * 80)
    print("\nSteps: Design → Code → Tests → Documentation")
    
    code_chain = PromptChainLibrary.create_code_generation_chain(llm)
    
    result2 = code_chain.execute({
        "problem": "Write a function to find the longest palindrome substring in a string"
    })
    
    print("\n" + "-" * 80)
    print("CHAIN RESULTS")
    print("-" * 80)
    print(f"Success: {result2.success}")
    print(f"Steps Executed: {result2.steps_executed}")
    print(f"Execution Time: {result2.execution_time:.2f}s")
    
    print("\n" + "-" * 80)
    print("INTERMEDIATE OUTPUTS")
    print("-" * 80)
    for step_name, output in result2.intermediate_outputs.items():
        print(f"\n[{step_name}]")
        print(output[:200] + "..." if len(output) > 200 else output)
    
    # Test 3: Custom chain with transformations
    print("\n" + "=" * 80)
    print("Test 3: Custom Chain with Transformations")
    print("=" * 80)
    
    custom_chain = PromptChain(llm=llm)
    
    # Step 1: Extract key points
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract key points from the text."),
        ("user", "Text: {text}\n\nList the main points as bullet points.")
    ])
    custom_chain.add_step(
        name="key_points",
        prompt=extract_prompt,
        description="Extract key points"
    )
    
    # Step 2: Summarize (with transformation)
    def count_points(text: str) -> dict:
        """Transform function to count bullet points."""
        points = [line for line in text.split('\n') if line.strip().startswith(('-', '•', '*'))]
        return {"key_points": text, "point_count": len(points)}
    
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Create a brief summary."),
        ("user", "Key Points:\n{key_points}\n\nWrite a 2-sentence summary.")
    ])
    custom_chain.add_step(
        name="summary",
        prompt=summary_prompt,
        transform=count_points,
        description="Create summary with point count"
    )
    
    result3 = custom_chain.execute({
        "text": """
        Artificial Intelligence is transforming healthcare in multiple ways.
        AI systems can analyze medical images faster and more accurately than humans.
        Predictive models help identify patients at risk of diseases.
        Natural language processing enables better patient record management.
        Machine learning optimizes hospital operations and resource allocation.
        """
    })
    
    print(f"\nSuccess: {result3.success}")
    print(f"Steps Executed: {result3.steps_executed}")
    
    print("\n" + "-" * 80)
    print("OUTPUT WITH TRANSFORMATION")
    print("-" * 80)
    print(f"Final Output: {result3.final_output}")
    
    # Test 4: Research chain
    print("\n" + "=" * 80)
    print("Test 4: Research Chain")
    print("=" * 80)
    print("\nSteps: Questions → Research → Synthesis → Conclusions")
    
    research_chain = PromptChainLibrary.create_research_chain(llm)
    
    result4 = research_chain.execute({
        "topic": "Quantum Computing Applications"
    })
    
    print("\n" + "-" * 80)
    print("CHAIN RESULTS")
    print("-" * 80)
    print(f"Success: {result4.success}")
    print(f"Steps Executed: {result4.steps_executed}")
    print(f"Execution Time: {result4.execution_time:.2f}s")
    
    print("\n" + "-" * 80)
    print("RESEARCH CONCLUSIONS")
    print("-" * 80)
    print(result4.final_output[:400] + "..." if len(result4.final_output) > 400 else result4.final_output)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Prompt Chaining provides:
✓ Sequential prompt execution
✓ Clear data flow between steps
✓ Intermediate result storage
✓ Transformation functions
✓ Error handling and tracking
✓ Modular, composable workflows

This pattern excels at:
- Multi-step document processing
- Complex data transformations
- Iterative content refinement
- Research and analysis workflows
- Code generation pipelines
- Multi-stage decision making

Chain components:
1. Chain Nodes: Individual prompts
2. Data Flow: Output → Input passing
3. Transformations: Process between steps
4. Orchestrator: Coordinates execution
5. Error Handler: Manages failures

Chain types demonstrated:
1. Content Generation: Outline → Draft → Refine → Conclude
2. Code Generation: Design → Code → Tests → Docs
3. Research: Questions → Research → Synthesis → Conclusions
4. Custom: Flexible with transformations

Data flow patterns:
- Sequential: Each step uses previous output
- Accumulative: All previous outputs available
- Transformed: Process data between steps
- Selective: Choose which outputs to pass

Benefits:
- Modularity: Each step is independent
- Clarity: Explicit workflow structure
- Debugging: Intermediate outputs visible
- Reusability: Chains can be composed
- Flexibility: Easy to modify steps
- Transparency: Track execution flow

Use Prompt Chaining when you need:
- Multi-step processing workflows
- Clear separation of concerns
- Iterative refinement processes
- Complex data transformations
- Modular, maintainable pipelines
- Transparent execution tracking
""")


if __name__ == "__main__":
    demonstrate_prompt_chaining()
