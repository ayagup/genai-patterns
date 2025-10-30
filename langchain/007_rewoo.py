"""
Pattern 007: ReWOO (Reasoning Without Observation)

Description:
    Decouples reasoning from observation by planning all tool uses upfront,
    then executing them in batch. Unlike ReAct which interleaves reasoning
    and observation, ReWOO plans all actions first, then executes them,
    reducing LLM calls and enabling parallel execution.

Key Concepts:
    - Upfront Planning: Generate complete tool usage plan first
    - Dependency Analysis: Identify dependencies between tool calls
    - Batch Execution: Execute all tools (potentially in parallel)
    - Variable System: Use variables (e.g., #E1, #E2) for results
    - Final Synthesis: Generate answer using all tool results

Benefits:
    - Fewer LLM calls (plan once, execute all)
    - Enables parallel tool execution
    - More efficient for multi-tool tasks
    - Clearer execution plan

Use Cases:
    - Multi-tool research tasks
    - Information gathering from multiple sources
    - Tasks requiring coordinated tool use
    - Scenarios where parallelization is beneficial

LangChain Implementation:
    Separate planning and execution chains with variable substitution.
"""

import os
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


@dataclass
class ToolCall:
    """Represents a planned tool call."""
    variable: str  # e.g., "#E1"
    tool_name: str
    arguments: str
    dependencies: List[str] = field(default_factory=list)  # Variable dependencies
    result: Optional[str] = None
    executed: bool = False


@dataclass
class ExecutionPlan:
    """Represents the complete tool execution plan."""
    question: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    final_reasoning: str = ""
    
    def get_executable_calls(self) -> List[ToolCall]:
        """Get tool calls that can be executed (dependencies met)."""
        executable = []
        for call in self.tool_calls:
            if not call.executed:
                # Check if all dependencies are met
                deps_met = all(
                    any(tc.variable == dep and tc.executed for tc in self.tool_calls)
                    for dep in call.dependencies
                )
                if not call.dependencies or deps_met:
                    executable.append(call)
        return executable


class ReWOOAgent:
    """Agent that uses ReWOO (Reasoning Without Observation) pattern."""
    
    def __init__(self, tools: List[Tool], model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the ReWOO agent.
        
        Args:
            tools: List of available tools
            model_name: Name of the OpenAI model
        """
        self.tools = {tool.name: tool for tool in tools}
        self.tool_descriptions = self._format_tool_descriptions()
        
        self.planner_llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,  # Low temperature for consistent planning
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.solver_llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for the planner."""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
    
    def plan(self, question: str) -> ExecutionPlan:
        """
        Create an execution plan for answering the question.
        
        Args:
            question: The question to answer
            
        Returns:
            ExecutionPlan with all tool calls planned
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a planner that creates execution plans using available tools.

Available tools:
{tools}

Create a plan to answer the question using these tools. For each step:
1. Assign a variable (#E1, #E2, etc.)
2. Specify the tool and its arguments
3. Reference previous results using their variables

Format each plan step as:
#E1 = ToolName[arguments]
#E2 = ToolName[arguments using #E1 if needed]
...

Then provide reasoning for how to combine the results:
Plan: [Brief description of how to use the results]

Example:
#E1 = Calculator[15 * 7]
#E2 = Search[capital of #E1]
Plan: Use #E1 to get the product, then search for information about #E2."""),
            ("human", "Question: {question}\n\nExecution Plan:")
        ])
        
        chain = prompt | self.planner_llm | StrOutputParser()
        response = chain.invoke({
            "question": question,
            "tools": self.tool_descriptions
        })
        
        # Parse the plan
        plan = ExecutionPlan(question=question)
        
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Parse tool calls like: #E1 = ToolName[arguments]
            match = re.match(r'#(E\d+)\s*=\s*(\w+)\[(.*?)\]', line)
            if match:
                variable = f"#{match.group(1)}"
                tool_name = match.group(2)
                arguments = match.group(3).strip()
                
                # Extract dependencies (variables referenced in arguments)
                dependencies = re.findall(r'#E\d+', arguments)
                
                tool_call = ToolCall(
                    variable=variable,
                    tool_name=tool_name,
                    arguments=arguments,
                    dependencies=dependencies
                )
                plan.tool_calls.append(tool_call)
            
            # Extract final reasoning
            elif line.lower().startswith('plan:'):
                plan.final_reasoning = line.split(':', 1)[1].strip()
        
        return plan
    
    def execute_plan(self, plan: ExecutionPlan, parallel: bool = False) -> Dict[str, Any]:
        """
        Execute the tool calls in the plan.
        
        Args:
            plan: The execution plan
            parallel: Whether to execute independent calls in parallel
            
        Returns:
            Dictionary with execution results
        """
        print(f"\nQuestion: {plan.question}\n")
        print("Execution Plan:")
        for call in plan.tool_calls:
            deps = f" (depends on: {', '.join(call.dependencies)})" if call.dependencies else ""
            print(f"  {call.variable} = {call.tool_name}[{call.arguments}]{deps}")
        print(f"\nFinal Plan: {plan.final_reasoning}\n")
        
        print("="*60)
        print("EXECUTING TOOLS")
        print("="*60)
        
        # Execute tool calls respecting dependencies
        while True:
            executable = plan.get_executable_calls()
            if not executable:
                break
            
            if parallel and len(executable) > 1:
                # Execute in parallel
                with ThreadPoolExecutor(max_workers=len(executable)) as executor:
                    futures = {
                        executor.submit(self._execute_tool_call, call, plan): call 
                        for call in executable
                    }
                    
                    for future in as_completed(futures):
                        call = futures[future]
                        try:
                            result = future.result()
                            call.result = result
                            call.executed = True
                            print(f"\n✓ {call.variable} = {result[:100]}...")
                        except Exception as e:
                            call.result = f"Error: {str(e)}"
                            call.executed = True
                            print(f"\n✗ {call.variable} failed: {str(e)}")
            else:
                # Execute sequentially
                for call in executable:
                    try:
                        result = self._execute_tool_call(call, plan)
                        call.result = result
                        call.executed = True
                        print(f"\n✓ {call.variable} = {result[:100]}...")
                    except Exception as e:
                        call.result = f"Error: {str(e)}"
                        call.executed = True
                        print(f"\n✗ {call.variable} failed: {str(e)}")
        
        # Synthesize final answer
        print(f"\n{'='*60}")
        print("SYNTHESIZING ANSWER")
        print('='*60)
        
        answer = self._synthesize_answer(plan)
        
        return {
            "question": plan.question,
            "answer": answer,
            "tool_calls": len(plan.tool_calls),
            "plan": plan.final_reasoning
        }
    
    def _execute_tool_call(self, call: ToolCall, plan: ExecutionPlan) -> str:
        """Execute a single tool call."""
        # Substitute variable references with actual results
        arguments = call.arguments
        for dep_var in call.dependencies:
            # Find the result for this dependency
            for other_call in plan.tool_calls:
                if other_call.variable == dep_var and other_call.result:
                    arguments = arguments.replace(dep_var, other_call.result)
        
        # Get the tool
        tool = self.tools.get(call.tool_name)
        if not tool:
            raise ValueError(f"Tool {call.tool_name} not found")
        
        # Execute the tool
        try:
            result = tool.func(arguments)
            return str(result)
        except Exception as e:
            raise Exception(f"Tool execution failed: {str(e)}")
    
    def _synthesize_answer(self, plan: ExecutionPlan) -> str:
        """Synthesize final answer from tool results."""
        # Build evidence from tool results
        evidence = []
        for call in plan.tool_calls:
            if call.executed and call.result:
                evidence.append(f"{call.variable}: {call.result}")
        
        evidence_text = "\n".join(evidence)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Based on the evidence gathered from tool execution, 
provide a comprehensive answer to the original question.

Use the evidence to construct a well-reasoned response."""),
            ("human", """Question: {question}

Evidence:
{evidence}

Plan: {plan}

Answer:""")
        ])
        
        chain = prompt | self.solver_llm | StrOutputParser()
        answer = chain.invoke({
            "question": plan.question,
            "evidence": evidence_text,
            "plan": plan.final_reasoning
        })
        
        return answer.strip()
    
    def solve(self, question: str, parallel: bool = False) -> Dict[str, Any]:
        """
        Solve a question using ReWOO pattern.
        
        Args:
            question: The question to answer
            parallel: Whether to use parallel execution
            
        Returns:
            Dictionary with answer and metadata
        """
        print(f"\n{'='*80}")
        print("ReWOO: PLANNING PHASE")
        print('='*80)
        
        # Create plan
        plan = self.plan(question)
        
        print(f"\n{'='*80}")
        print("ReWOO: EXECUTION PHASE")
        print('='*80)
        
        # Execute plan
        result = self.execute_plan(plan, parallel=parallel)
        
        return result


# Example tools
def calculator(expression: str) -> float:
    """Calculate a mathematical expression."""
    try:
        # Simple eval (in production, use a safe math parser)
        result = eval(expression)
        return result
    except Exception as e:
        return f"Calculation error: {str(e)}"


def search(query: str) -> str:
    """Simulate a search tool."""
    # In production, this would call a real search API
    responses = {
        "capital of france": "Paris is the capital of France.",
        "population of tokyo": "Tokyo has a population of approximately 14 million people.",
        "speed of light": "The speed of light is approximately 299,792,458 meters per second.",
        "inventor of telephone": "Alexander Graham Bell is credited with inventing the telephone.",
    }
    
    query_lower = query.lower()
    for key, value in responses.items():
        if key in query_lower:
            return value
    
    return f"Search results for: {query}"


def get_date(query: str) -> str:
    """Get current date information."""
    from datetime import datetime
    now = datetime.now()
    
    if "year" in query.lower():
        return str(now.year)
    elif "month" in query.lower():
        return now.strftime("%B")
    elif "day" in query.lower():
        return str(now.day)
    else:
        return now.strftime("%Y-%m-%d")


def demonstrate_rewoo():
    """Demonstrates the ReWOO pattern."""
    
    print("=" * 80)
    print("PATTERN 007: ReWOO (Reasoning Without Observation)")
    print("=" * 80)
    print()
    print("ReWOO decouples reasoning from observation:")
    print("1. Planning Phase: Plan all tool uses upfront")
    print("2. Dependency Analysis: Identify dependencies between calls")
    print("3. Execution Phase: Execute tools (potentially in parallel)")
    print("4. Synthesis: Generate answer from all results")
    print()
    
    # Create tools
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="Calculate mathematical expressions. Input should be a valid expression."
        ),
        Tool(
            name="Search",
            func=search,
            description="Search for information. Input should be a search query."
        ),
        Tool(
            name="GetDate",
            func=get_date,
            description="Get current date information. Input can be 'year', 'month', 'day', or 'full'."
        )
    ]
    
    # Create agent
    agent = ReWOOAgent(tools=tools)
    
    # Test questions
    questions = [
        "What is 25 * 34, and what is the capital of France?",
        "Calculate 144 / 12, then search for information about that number",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}")
        print('='*80)
        
        try:
            result = agent.solve(question, parallel=False)
            
            print(f"\n\n{'='*80}")
            print("FINAL RESULT")
            print('='*80)
            print(f"\nQuestion: {result['question']}")
            print(f"\nAnswer: {result['answer']}")
            print(f"\nTool Calls: {result['tool_calls']}")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("ReWOO PATTERN DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("1. Upfront Planning: All tool uses planned before execution")
    print("2. Variable System: Results referenced as #E1, #E2, etc.")
    print("3. Dependency Tracking: Execution order based on dependencies")
    print("4. Batch Execution: All tools executed in one phase")
    print("5. Result Synthesis: Final answer combines all evidence")
    print()
    print("Advantages over ReAct:")
    print("- Fewer LLM calls (plan once vs. after each observation)")
    print("- Enables parallel tool execution")
    print("- Clearer execution flow")
    print("- More efficient for multi-tool tasks")
    print()
    print("When to use ReWOO:")
    print("- Multi-tool information gathering")
    print("- Tasks with independent tool calls")
    print("- Scenarios benefiting from parallelization")
    print("- When you want predictable tool usage")
    print()
    print("LangChain Components Used:")
    print("- Tool abstraction for uniform tool interface")
    print("- Separate planning and execution chains")
    print("- Variable substitution system")
    print("- Optional parallel execution")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    demonstrate_rewoo()
