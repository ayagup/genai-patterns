"""
Pattern 085: Step-by-Step Instructions Pattern

Description:
    The Step-by-Step Instructions pattern structures prompts to guide the LLM through
    a process in discrete, ordered steps. This pattern is particularly effective for
    complex tasks that require systematic execution, clear sequencing, and validation
    at each stage. By breaking down tasks into explicit steps, this pattern ensures
    thorough coverage, reduces errors, and provides clear reasoning traces.

    This pattern is essential for:
    - Complex multi-stage workflows
    - Tasks requiring specific ordering
    - Processes with validation checkpoints
    - Teaching and tutoring applications
    - Troubleshooting and debugging
    - Recipe and instruction generation

Components:
    1. Step Definition
       - Numbered sequence (1, 2, 3...)
       - Clear action verbs (calculate, analyze, verify)
       - Specific objectives per step
       - Expected outcomes

    2. Step Organization
       - Main steps (high-level stages)
       - Sub-steps (detailed actions)
       - Conditional branches (if-then logic)
       - Iteration loops (repeat until)

    3. Validation Checkpoints
       - Completion criteria per step
       - Quality checks
       - Error detection
       - Progress verification

    4. Progress Tracking
       - Current step indication
       - Completed steps summary
       - Remaining steps preview
       - Overall progress percentage

Use Cases:
    1. Problem-Solving Workflows
       - Mathematical problem solving
       - Debugging code issues
       - Troubleshooting technical problems
       - Root cause analysis
       - Decision-making frameworks

    2. Instructional Content
       - Tutorial generation
       - How-to guides
       - Recipe creation
       - Assembly instructions
       - Learning modules

    3. Analysis Tasks
       - Data analysis workflows
       - Research methodology
       - Document review processes
       - Competitive analysis
       - Risk assessment

    4. Creative Processes
       - Story development (plot, characters, scenes)
       - Content creation workflow
       - Design thinking process
       - Brainstorming structure
       - Iterative refinement

    5. Compliance and Safety
       - Safety protocols
       - Audit procedures
       - Quality assurance checks
       - Regulatory compliance steps
       - Standard operating procedures

LangChain Implementation:
    LangChain enables step-by-step patterns through:
    - Structured prompt templates
    - Chain sequencing (LCEL)
    - State management between steps
    - Conditional routing
    - Progress tracking utilities

Key Features:
    1. Explicit Step Numbering
       - Clear sequential order
       - Easy progress tracking
       - Reference-able steps
       - Skippable sections

    2. Hierarchical Structure
       - Main steps and sub-steps
       - Nested levels (steps within steps)
       - Grouped related actions
       - Flexible depth

    3. Validation Integration
       - Check results at each step
       - Verify assumptions
       - Catch errors early
       - Confirm understanding

    4. Adaptive Sequencing
       - Conditional next steps
       - Skip irrelevant steps
       - Loop for iteration
       - Branch based on results

Best Practices:
    1. Step Clarity
       - Use clear, action-oriented language
       - One main action per step
       - Avoid ambiguity
       - Provide examples when helpful

    2. Appropriate Granularity
       - Not too high-level (vague)
       - Not too detailed (overwhelming)
       - Balance based on complexity
       - Adjust for target audience

    3. Logical Ordering
       - Dependencies respected
       - Natural flow
       - Build on previous steps
       - Group related steps

    4. Completeness
       - Cover all necessary steps
       - Don't skip crucial details
       - Include error handling
       - Provide alternative paths

Trade-offs:
    Advantages:
    - Reduced cognitive load (one step at a time)
    - Fewer errors (systematic approach)
    - Better traceability (clear reasoning path)
    - Easier debugging (identify problem step)
    - Improved learning (explicit process)
    - Higher consistency (standardized approach)

    Disadvantages:
    - Can be verbose (more tokens)
    - May feel rigid (less flexibility)
    - Slower for simple tasks (overhead)
    - Requires careful design (step definition)
    - May over-constrain creativity
    - Context window consumption

Production Considerations:
    1. Step Optimization
       - Balance detail with brevity
       - Remove redundant steps
       - Combine simple steps
       - Expand complex steps

    2. Error Handling
       - Define failure conditions
       - Provide recovery steps
       - Include validation checks
       - Offer troubleshooting guidance

    3. User Experience
       - Show progress indicators
       - Allow step skipping (when safe)
       - Provide step summaries
       - Enable backtracking

    4. Performance
       - Cache intermediate results
       - Parallelize independent steps
       - Optimize step execution
       - Monitor step durations

    5. Maintenance
       - Version control for procedures
       - Regular step reviews
       - Update based on feedback
       - Test step completeness
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class StepStatus(Enum):
    """Status of a step"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Step:
    """Represents a single step in a process"""
    number: int
    title: str
    description: str
    action: str
    validation: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    sub_steps: List['Step'] = field(default_factory=list)


@dataclass
class StepByStepConfig:
    """Configuration for step-by-step instructions"""
    include_validation: bool = True
    include_examples: bool = False
    include_reasoning: bool = True
    temperature: float = 0.3
    model_name: str = "gpt-3.5-turbo"


class StepByStepAgent:
    """
    Agent that structures tasks into clear step-by-step instructions.
    
    This agent demonstrates:
    1. Generating step-by-step processes
    2. Following structured procedures
    3. Progress tracking and validation
    4. Hierarchical step organization
    """
    
    def __init__(self, config: Optional[StepByStepConfig] = None):
        """
        Initialize step-by-step agent.
        
        Args:
            config: Configuration for step-by-step processing
        """
        self.config = config or StepByStepConfig()
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature
        )
    
    def generate_steps(
        self,
        task: str,
        num_steps: Optional[int] = None
    ) -> str:
        """
        Generate step-by-step instructions for a task.
        
        Args:
            task: Task to create steps for
            num_steps: Optional target number of steps
            
        Returns:
            Step-by-step instructions
        """
        num_steps_instruction = f" in approximately {num_steps} steps" if num_steps else ""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at creating clear, actionable step-by-step instructions.

Your instructions should:
1. Break down the task into discrete, manageable steps
2. Use numbered steps (1, 2, 3, etc.)
3. Start each step with an action verb
4. Be specific and actionable
5. Include validation checks where appropriate
6. Maintain logical ordering"""),
            ("human", "Create step-by-step instructions for: {task}{num_steps_instruction}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({
            "task": task,
            "num_steps_instruction": num_steps_instruction
        })
        
        return result
    
    def solve_problem_step_by_step(
        self,
        problem: str
    ) -> str:
        """
        Solve a problem using step-by-step reasoning.
        
        Args:
            problem: Problem to solve
            
        Returns:
            Step-by-step solution
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a problem solver who thinks step-by-step.

For each problem:
1. Break it down into logical steps
2. Show your work for each step
3. Validate results as you go
4. State the final answer clearly

Use this format:
Step 1: [First action]
[Work/reasoning]

Step 2: [Second action]
[Work/reasoning]

...

Final Answer: [Conclusion]"""),
            ("human", "Solve this problem step-by-step: {problem}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"problem": problem})
        
        return result
    
    def debug_code_step_by_step(
        self,
        code: str,
        error: str
    ) -> str:
        """
        Debug code using systematic step-by-step approach.
        
        Args:
            code: Code with error
            error: Error message
            
        Returns:
            Step-by-step debugging process
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a debugging expert who follows a systematic step-by-step process.

Debugging steps:
1. Understand the error message
2. Locate the error in the code
3. Identify the root cause
4. Propose a fix
5. Explain why the fix works
6. Provide the corrected code

Follow these steps explicitly in your response."""),
            ("human", """Debug this code step-by-step:

Code:
{code}

Error:
{error}""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"code": code, "error": error})
        
        return result
    
    def create_tutorial_step_by_step(
        self,
        topic: str,
        audience: str = "beginners"
    ) -> str:
        """
        Create a tutorial with step-by-step instructions.
        
        Args:
            topic: Topic to teach
            audience: Target audience level
            
        Returns:
            Step-by-step tutorial
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert teacher creating step-by-step tutorials.

Your tutorials should:
1. Start with an overview
2. List prerequisites
3. Break content into numbered steps
4. Include examples in each step
5. Provide checkpoints to verify understanding
6. End with a summary and next steps

Adapt complexity to the audience level."""),
            ("human", "Create a step-by-step tutorial on '{topic}' for {audience}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"topic": topic, "audience": audience})
        
        return result
    
    def analyze_data_step_by_step(
        self,
        data_description: str,
        analysis_goal: str
    ) -> str:
        """
        Perform data analysis using step-by-step methodology.
        
        Args:
            data_description: Description of the data
            analysis_goal: Goal of the analysis
            
        Returns:
            Step-by-step analysis process
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analyst who follows a structured step-by-step methodology.

Standard data analysis steps:
1. Understand the data and goal
2. Data cleaning and preparation
3. Exploratory analysis
4. Apply appropriate techniques
5. Interpret results
6. Draw conclusions
7. Recommend actions

Follow these steps in your analysis."""),
            ("human", """Analyze this data step-by-step:

Data: {data_description}
Goal: {analysis_goal}""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "data_description": data_description,
            "analysis_goal": analysis_goal
        })
        
        return result
    
    def decision_framework_step_by_step(
        self,
        decision: str,
        context: str
    ) -> str:
        """
        Apply decision-making framework step-by-step.
        
        Args:
            decision: Decision to make
            context: Context and constraints
            
        Returns:
            Step-by-step decision analysis
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a decision analyst using a structured step-by-step framework.

Decision-making steps:
1. Define the decision clearly
2. Identify objectives and criteria
3. Generate alternatives
4. Evaluate each alternative against criteria
5. Assess risks and uncertainties
6. Make recommendation
7. Plan implementation

Follow these steps systematically."""),
            ("human", """Help me make this decision step-by-step:

Decision: {decision}
Context: {context}""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"decision": decision, "context": context})
        
        return result
    
    def troubleshoot_step_by_step(
        self,
        problem_description: str,
        system: str = "general"
    ) -> str:
        """
        Troubleshoot a problem using systematic steps.
        
        Args:
            problem_description: Description of the problem
            system: System or domain
            
        Returns:
            Step-by-step troubleshooting guide
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a troubleshooting expert who uses a systematic step-by-step approach.

Troubleshooting methodology:
1. Gather information about the problem
2. Identify symptoms
3. Form hypotheses about causes
4. Test hypotheses systematically
5. Isolate the root cause
6. Implement solution
7. Verify resolution
8. Document for future reference

Follow these steps in your troubleshooting process."""),
            ("human", """Troubleshoot this problem step-by-step:

Problem: {problem_description}
System: {system}""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "problem_description": problem_description,
            "system": system
        })
        
        return result


def demonstrate_step_by_step():
    """Demonstrate step-by-step instruction patterns"""
    print("=" * 80)
    print("STEP-BY-STEP INSTRUCTIONS PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = StepByStepAgent()
    
    # Example 1: Generate Instructions for Task
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Generate Step-by-Step Instructions")
    print("=" * 80)
    
    task = "Plan and execute a successful product launch"
    print(f"\nTask: {task}\n")
    instructions = agent.generate_steps(task, num_steps=7)
    print(instructions)
    
    # Example 2: Math Problem Solving
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Step-by-Step Problem Solving")
    print("=" * 80)
    
    problem = "If a train travels 120 miles in 2 hours, then slows down and travels 80 miles in 2 hours, what is its average speed for the entire journey?"
    print(f"\nProblem: {problem}\n")
    solution = agent.solve_problem_step_by_step(problem)
    print(solution)
    
    # Example 3: Code Debugging
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Step-by-Step Code Debugging")
    print("=" * 80)
    
    buggy_code = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

result = calculate_average([])
print(result)
"""
    error = "ZeroDivisionError: division by zero"
    
    print(f"Code:\n{buggy_code}")
    print(f"\nError: {error}\n")
    debug_process = agent.debug_code_step_by_step(buggy_code, error)
    print(debug_process)
    
    # Example 4: Tutorial Creation
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Step-by-Step Tutorial")
    print("=" * 80)
    
    topic = "Setting up a Python virtual environment"
    audience = "beginners"
    print(f"\nTopic: {topic}")
    print(f"Audience: {audience}\n")
    tutorial = agent.create_tutorial_step_by_step(topic, audience)
    print(tutorial)
    
    # Example 5: Data Analysis
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Step-by-Step Data Analysis")
    print("=" * 80)
    
    data_desc = "Sales data for last quarter showing revenue, units sold, and customer segments"
    goal = "Identify which customer segment has the highest growth potential"
    print(f"\nData: {data_desc}")
    print(f"Goal: {goal}\n")
    analysis = agent.analyze_data_step_by_step(data_desc, goal)
    print(analysis)
    
    # Example 6: Decision Making
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Step-by-Step Decision Framework")
    print("=" * 80)
    
    decision = "Should we expand to a new market?"
    context = "B2B software company with stable domestic revenue, considering Asian market expansion"
    print(f"\nDecision: {decision}")
    print(f"Context: {context}\n")
    decision_analysis = agent.decision_framework_step_by_step(decision, context)
    print(decision_analysis)
    
    # Example 7: Troubleshooting
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Step-by-Step Troubleshooting")
    print("=" * 80)
    
    problem = "Website is loading very slowly for users"
    system = "web application"
    print(f"\nProblem: {problem}")
    print(f"System: {system}\n")
    troubleshooting = agent.troubleshoot_step_by_step(problem, system)
    print(troubleshooting)
    
    # Summary
    print("\n" + "=" * 80)
    print("STEP-BY-STEP INSTRUCTIONS SUMMARY")
    print("=" * 80)
    print("""
Step-by-Step Pattern Benefits:
1. Reduced Complexity: Breaks down complex tasks into manageable parts
2. Clear Progress: Easy to track where you are in the process
3. Fewer Errors: Systematic approach catches mistakes early
4. Better Learning: Explicit steps aid understanding
5. Reproducibility: Others can follow the same process
6. Debugging: Easy to identify where things went wrong

Key Applications Demonstrated:
1. Task Planning: Creating structured workflows
2. Problem Solving: Mathematical and logical problems
3. Code Debugging: Systematic error identification
4. Tutorial Creation: Teaching complex topics
5. Data Analysis: Structured analytical methodology
6. Decision Making: Rational choice frameworks
7. Troubleshooting: Systematic problem diagnosis

Step-by-Step Structure Elements:
- Numbered sequence (clear ordering)
- Action verbs (calculate, analyze, verify)
- Validation checkpoints (confirm results)
- Progress indicators (current step)
- Sub-steps (hierarchical detail)
- Conditional branches (if-then logic)

Best Practices:
1. Use clear, action-oriented language
2. Maintain logical dependencies between steps
3. Include validation checks at key points
4. Keep steps at appropriate granularity
5. Number steps clearly (1, 2, 3...)
6. Start steps with action verbs
7. Provide examples within steps when helpful
8. Include error handling and alternatives

When to Use Step-by-Step:
- Complex multi-stage tasks
- Teaching and learning scenarios
- Troubleshooting and debugging
- Compliance and safety procedures
- Quality assurance processes
- Standardized workflows
- Problem-solving that requires systematic approach

Advantages Over Free-Form:
- More thorough coverage
- Easier to follow and verify
- Better for complex tasks
- Reduces cognitive load
- Improves consistency
- Facilitates collaboration

Trade-offs:
- More verbose (uses more tokens)
- Can feel rigid for simple tasks
- Requires careful step design
- May over-structure creative tasks
- Slower for trivial operations

Production Tips:
- Cache common step sequences
- Allow step skipping when safe
- Provide progress indicators
- Enable backtracking to previous steps
- Version control procedural steps
- Gather feedback on step clarity
- Optimize step granularity based on usage
""")
    
    print("\n" + "=" * 80)
    print("Pattern 085 (Step-by-Step Instructions) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_step_by_step()
