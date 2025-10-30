"""
Pattern 5: Plan-and-Execute
Separates planning from execution phases.
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import Tool
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class PlanStep:
    """Represents a step in the plan"""
    id: int
    description: str
    dependencies: List[int]
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None

class PlanAndExecutePattern:
    def __init__(self):
        self.planner_llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.executor_llm = ChatOpenAI(temperature=0.3, model="gpt-4")
        self.replanner_llm = ChatOpenAI(temperature=0, model="gpt-4")
        
        # Mock tools for execution
        self.tools = {
            'search': Tool(
                name="search",
                func=lambda q: f"Search results for: {q}",
                description="Search for information"
            ),
            'calculate': Tool(
                name="calculate",
                func=lambda expr: str(eval(expr)) if expr else "Error",
                description="Perform calculations"
            ),
            'analyze': Tool(
                name="analyze",
                func=lambda data: f"Analysis of: {data}",
                description="Analyze data"
            )
        }
    
    def create_plan(self, objective: str) -> List[PlanStep]:
        """Create a plan to achieve the objective"""
        template = """Create a detailed step-by-step plan to achieve the following objective:

Objective: {objective}

Provide a numbered list of steps. For each step, indicate if it depends on previous steps.
Format each step as: "Step X: [description] (depends on: [step numbers or 'none'])"

Plan:"""
        
        prompt = PromptTemplate(template=template, input_variables=["objective"])
        chain = LLMChain(llm=self.planner_llm, prompt=prompt)
        result = chain.run(objective=objective)
        
        # Parse the plan
        steps = []
        for line in result.split('\n'):
            line = line.strip()
            if line and line.startswith('Step'):
                try:
                    # Extract step number and description
                    parts = line.split(':', 1)
                    if len(parts) < 2:
                        continue
                    
                    step_num = int(parts[0].replace('Step', '').strip())
                    rest = parts[1].strip()
                    
                    # Extract dependencies
                    if '(depends on:' in rest:
                        desc, deps_part = rest.split('(depends on:', 1)
                        deps_part = deps_part.rstrip(')')
                        
                        if 'none' in deps_part.lower():
                            dependencies = []
                        else:
                            dependencies = [
                                int(d.strip())
                                for d in deps_part.split(',')
                                if d.strip().isdigit()
                            ]
                    else:
                        desc = rest
                        dependencies = []
                    
                    steps.append(PlanStep(
                        id=step_num,
                        description=desc.strip(),
                        dependencies=dependencies
                    ))
                except Exception as e:
                    print(f"Error parsing step: {line}, Error: {e}")
                    continue
        
        return steps
    
    def execute_step(self, step: PlanStep, context: Dict[str, Any]) -> str:
        """Execute a single step"""
        template = """Execute the following step:

Step: {step_description}

Context from previous steps:
{context}

Provide the result of executing this step. Be specific and actionable.

Result:"""
        
        context_str = "\n".join([
            f"Step {k}: {v}"
            for k, v in context.items()
        ])
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["step_description", "context"]
        )
        chain = LLMChain(llm=self.executor_llm, prompt=prompt)
        return chain.run(step_description=step.description, context=context_str)
    
    def can_execute_step(self, step: PlanStep, completed_steps: List[int]) -> bool:
        """Check if a step's dependencies are satisfied"""
        return all(dep in completed_steps for dep in step.dependencies)
    
    def should_replan(self, step: PlanStep, result: str) -> bool:
        """Determine if replanning is needed"""
        template = """Given the following step and its execution result, determine if we need to replan:

Step: {step}
Result: {result}

Does this result indicate a failure or issue that requires replanning? Answer YES or NO and explain briefly.

Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["step", "result"]
        )
        chain = LLMChain(llm=self.replanner_llm, prompt=prompt)
        response = chain.run(step=step.description, result=result)
        
        return response.strip().upper().startswith('YES')
    
    def replan(self, objective: str, completed_steps: List[PlanStep], 
               failed_step: PlanStep, context: Dict[str, Any]) -> List[PlanStep]:
        """Create a new plan given the current situation"""
        completed_desc = "\n".join([
            f"Step {s.id}: {s.description} -> {s.result}"
            for s in completed_steps
        ])
        
        template = """The original objective is: {objective}

Steps completed so far:
{completed_steps}

This step failed:
Step {failed_id}: {failed_description}
Error: {error}

Create a new plan to achieve the objective, taking into account what has been completed and what failed.

New Plan:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["objective", "completed_steps", "failed_id", 
                           "failed_description", "error"]
        )
        chain = LLMChain(llm=self.replanner_llm, prompt=prompt)
        result = chain.run(
            objective=objective,
            completed_steps=completed_desc,
            failed_id=failed_step.id,
            failed_description=failed_step.description,
            error=failed_step.error or "Unknown error"
        )
        
        # Parse new plan (simplified)
        return self.create_plan(objective)
    
    def execute_plan(self, objective: str, max_iterations: int = 20) -> Dict[str, Any]:
        """Execute the plan with replanning support"""
        plan = self.create_plan(objective)
        completed_steps: List[int] = []
        context: Dict[str, Any] = {}
        iteration = 0
        replanned = False
        
        execution_log = []
        
        while len(completed_steps) < len(plan) and iteration < max_iterations:
            iteration += 1
            
            # Find next executable step
            next_step = None
            for step in plan:
                if step.id not in completed_steps and self.can_execute_step(step, completed_steps):
                    next_step = step
                    break
            
            if not next_step:
                # Check if there are pending steps with unmet dependencies
                pending = [s for s in plan if s.id not in completed_steps]
                if pending:
                    print(f"Warning: Cannot proceed. Pending steps with unmet dependencies: {[s.id for s in pending]}")
                break
            
            # Execute the step
            print(f"\nExecuting Step {next_step.id}: {next_step.description}")
            next_step.status = StepStatus.IN_PROGRESS
            
            try:
                result = self.execute_step(next_step, context)
                next_step.result = result
                next_step.status = StepStatus.COMPLETED
                completed_steps.append(next_step.id)
                context[next_step.id] = result
                
                execution_log.append({
                    'step_id': next_step.id,
                    'description': next_step.description,
                    'result': result,
                    'status': 'completed'
                })
                
                print(f"✓ Completed: {result[:100]}...")
                
                # Check if replanning is needed
                if self.should_replan(next_step, result):
                    print("⚠ Replanning required...")
                    plan = self.replan(objective, 
                                     [s for s in plan if s.id in completed_steps],
                                     next_step, context)
                    replanned = True
                    
            except Exception as e:
                next_step.status = StepStatus.FAILED
                next_step.error = str(e)
                execution_log.append({
                    'step_id': next_step.id,
                    'description': next_step.description,
                    'error': str(e),
                    'status': 'failed'
                })
                print(f"✗ Failed: {e}")
                break
        
        return {
            'objective': objective,
            'plan': [{'id': s.id, 'description': s.description, 'dependencies': s.dependencies} 
                    for s in plan],
            'completed_steps': completed_steps,
            'total_steps': len(plan),
            'iterations': iteration,
            'replanned': replanned,
            'execution_log': execution_log,
            'final_context': context,
            'success': len(completed_steps) == len(plan)
        }

if __name__ == "__main__":
    pae = PlanAndExecutePattern()
    
    objective = "Research and write a report on renewable energy sources"
    
    result = pae.execute_plan(objective)
    
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    print(f"Objective: {result['objective']}")
    print(f"Total Steps: {result['total_steps']}")
    print(f"Completed: {len(result['completed_steps'])}/{result['total_steps']}")
    print(f"Success: {result['success']}")
    print(f"Replanned: {result['replanned']}")
    print(f"\nExecution Log:")
    for log in result['execution_log']:
        status = log.get('status', 'unknown')
        print(f"  Step {log['step_id']}: {status.upper()}")
