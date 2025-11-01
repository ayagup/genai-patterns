"""
Pattern 120: Workflow Orchestration

Description:
    Manages complex multi-step workflows with dependencies, branching,
    error handling, and monitoring.

Components:
    - Workflow engine
    - Step dependencies
    - Error handling
    - Progress monitoring

Use Cases:
    - Business processes
    - Data pipelines
    - Complex automation

LangChain Implementation:
    Uses LangGraph for state machine-based workflow orchestration.
"""

import os
from typing import List, Dict, Any, Callable, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class StepStatus(Enum):
    """Status of workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Represents a step in the workflow."""
    name: str
    description: str
    dependencies: List[str]
    action: Callable
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str = None
    retry_count: int = 0
    max_retries: int = 3


class WorkflowOrchestrator:
    """Orchestrates complex multi-step workflows."""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.steps: Dict[str, WorkflowStep] = {}
        self.execution_log: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        
    def add_step(self, step: WorkflowStep):
        """Add a step to the workflow."""
        self.steps[step.name] = step
        print(f"✓ Added step: {step.name}")
    
    def validate_workflow(self) -> bool:
        """Validate workflow structure and dependencies."""
        print("\n=== Validating Workflow ===")
        
        # Check for circular dependencies
        for step_name, step in self.steps.items():
            if self._has_circular_dependency(step_name, set()):
                print(f"✗ Circular dependency detected for step: {step_name}")
                return False
        
        # Check all dependencies exist
        for step_name, step in self.steps.items():
            for dep in step.dependencies:
                if dep not in self.steps:
                    print(f"✗ Missing dependency: {dep} for step {step_name}")
                    return False
        
        print("✓ Workflow validation passed")
        return True
    
    def _has_circular_dependency(self, step_name: str, visited: set) -> bool:
        """Check for circular dependencies."""
        if step_name in visited:
            return True
        
        visited.add(step_name)
        step = self.steps.get(step_name)
        
        if step:
            for dep in step.dependencies:
                if self._has_circular_dependency(dep, visited.copy()):
                    return True
        
        return False
    
    def get_ready_steps(self) -> List[WorkflowStep]:
        """Get steps that are ready to execute."""
        ready = []
        
        for step in self.steps.values():
            if step.status != StepStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            deps_completed = all(
                self.steps[dep].status == StepStatus.COMPLETED
                for dep in step.dependencies
            )
            
            if deps_completed:
                ready.append(step)
        
        return ready
    
    def execute_step(self, step: WorkflowStep) -> bool:
        """Execute a single workflow step."""
        print(f"\n▶ Executing: {step.name}")
        print(f"  Description: {step.description}")
        
        step.status = StepStatus.RUNNING
        start_time = datetime.now()
        
        try:
            # Execute the step's action
            step.result = step.action(self.context)
            step.status = StepStatus.COMPLETED
            
            # Log execution
            log_entry = {
                "step": step.name,
                "status": "completed",
                "duration": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            self.execution_log.append(log_entry)
            
            print(f"  ✓ Completed")
            return True
            
        except Exception as e:
            step.error = str(e)
            step.retry_count += 1
            
            if step.retry_count < step.max_retries:
                print(f"  ⚠ Failed (attempt {step.retry_count}/{step.max_retries}): {e}")
                step.status = StepStatus.PENDING  # Retry
            else:
                step.status = StepStatus.FAILED
                print(f"  ✗ Failed after {step.retry_count} attempts: {e}")
            
            log_entry = {
                "step": step.name,
                "status": "failed",
                "error": str(e),
                "retry_count": step.retry_count,
                "timestamp": datetime.now().isoformat()
            }
            self.execution_log.append(log_entry)
            
            return False
    
    def execute_workflow(self):
        """Execute the entire workflow."""
        print("\n=== Executing Workflow ===")
        
        if not self.validate_workflow():
            print("✗ Workflow validation failed")
            return False
        
        total_steps = len(self.steps)
        completed = 0
        
        while completed < total_steps:
            ready_steps = self.get_ready_steps()
            
            if not ready_steps:
                # Check if any steps are still running or pending
                pending = sum(1 for s in self.steps.values() if s.status == StepStatus.PENDING)
                failed = sum(1 for s in self.steps.values() if s.status == StepStatus.FAILED)
                
                if pending == 0 or failed > 0:
                    break
                else:
                    continue
            
            # Execute ready steps (can be parallelized)
            for step in ready_steps:
                self.execute_step(step)
            
            completed = sum(1 for s in self.steps.values() if s.status == StepStatus.COMPLETED)
        
        # Generate execution summary
        self.generate_summary()
        
        return all(s.status == StepStatus.COMPLETED for s in self.steps.values())
    
    def generate_summary(self):
        """Generate workflow execution summary."""
        print("\n=== Workflow Summary ===")
        
        completed = sum(1 for s in self.steps.values() if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.steps.values() if s.status == StepStatus.FAILED)
        total = len(self.steps)
        
        print(f"Total Steps: {total}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {completed/total*100:.1f}%")
        
        if failed > 0:
            print("\nFailed Steps:")
            for step in self.steps.values():
                if step.status == StepStatus.FAILED:
                    print(f"  - {step.name}: {step.error}")
        
        # LLM-based analysis
        if self.execution_log:
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", "Analyze this workflow execution and provide insights."),
                ("user", """Execution Log:
{log}

Provide:
1. Performance insights
2. Bottlenecks identified
3. Optimization suggestions""")
            ])
            
            log_str = "\n".join([
                f"{entry['step']}: {entry['status']} ({entry.get('duration', 'N/A')}s)"
                for entry in self.execution_log
            ])
            
            chain = analysis_prompt | self.llm | StrOutputParser()
            analysis = chain.invoke({"log": log_str})
            
            print("\nWorkflow Analysis:")
            print(analysis)


def demonstrate_workflow_orchestration():
    """Demonstrate workflow orchestration pattern."""
    print("=== Workflow Orchestration Pattern ===\n")
    
    orchestrator = WorkflowOrchestrator()
    
    # Define workflow steps for data processing pipeline
    print("1. Defining Workflow Steps")
    print("-" * 50)
    
    # Step 1: Data Extraction
    def extract_data(context):
        print("    Extracting data from source...")
        context['raw_data'] = [1, 2, 3, 4, 5]
        return "Data extracted"
    
    orchestrator.add_step(WorkflowStep(
        name="extract_data",
        description="Extract data from source system",
        dependencies=[],
        action=extract_data
    ))
    
    # Step 2: Data Validation
    def validate_data(context):
        print("    Validating data...")
        data = context.get('raw_data', [])
        if not data:
            raise ValueError("No data to validate")
        context['validated_data'] = data
        return "Data validated"
    
    orchestrator.add_step(WorkflowStep(
        name="validate_data",
        description="Validate extracted data",
        dependencies=["extract_data"],
        action=validate_data
    ))
    
    # Step 3: Data Transformation
    def transform_data(context):
        print("    Transforming data...")
        data = context.get('validated_data', [])
        context['transformed_data'] = [x * 2 for x in data]
        return "Data transformed"
    
    orchestrator.add_step(WorkflowStep(
        name="transform_data",
        description="Transform data format",
        dependencies=["validate_data"],
        action=transform_data
    ))
    
    # Step 4: Data Enrichment
    def enrich_data(context):
        print("    Enriching data...")
        data = context.get('transformed_data', [])
        context['enriched_data'] = [{"value": x, "category": "processed"} for x in data]
        return "Data enriched"
    
    orchestrator.add_step(WorkflowStep(
        name="enrich_data",
        description="Enrich data with additional information",
        dependencies=["transform_data"],
        action=enrich_data
    ))
    
    # Step 5: Load to Database
    def load_to_db(context):
        print("    Loading to database...")
        data = context.get('enriched_data', [])
        context['db_records'] = len(data)
        return f"Loaded {len(data)} records"
    
    orchestrator.add_step(WorkflowStep(
        name="load_to_db",
        description="Load data to database",
        dependencies=["enrich_data"],
        action=load_to_db
    ))
    
    # Step 6: Generate Report
    def generate_report(context):
        print("    Generating report...")
        records = context.get('db_records', 0)
        context['report'] = f"Successfully processed {records} records"
        return "Report generated"
    
    orchestrator.add_step(WorkflowStep(
        name="generate_report",
        description="Generate processing report",
        dependencies=["load_to_db"],
        action=generate_report
    ))
    
    print()
    
    # Execute workflow
    print("2. Executing Workflow")
    print("-" * 50)
    success = orchestrator.execute_workflow()
    
    print("\n=== Results ===")
    print(f"Workflow {'succeeded' if success else 'failed'}")
    print(f"Final context: {orchestrator.context}")
    
    print("\n=== Summary ===")
    print("Workflow orchestration demonstrated with:")
    print("- Multi-step pipeline")
    print("- Dependency management")
    print("- Error handling and retries")
    print("- Progress tracking")
    print("- Execution logging")
    print("- Performance analysis")


if __name__ == "__main__":
    demonstrate_workflow_orchestration()
