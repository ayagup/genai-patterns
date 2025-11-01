"""
Pattern 134: Task-Specific Agent

Description:
    The Task-Specific Agent pattern creates agents optimized for particular tasks
    or workflows. Unlike general-purpose agents or domain experts with broad knowledge,
    task-specific agents are highly specialized for executing specific operations with
    maximum efficiency and reliability. They have deep knowledge of task requirements,
    common pitfalls, optimization strategies, and best practices for their specific task.
    
    Task-specific agents excel at their designated task through specialized prompts,
    optimized workflows, task-specific tools, relevant examples, and failure handling
    tailored to common task issues. They sacrifice breadth for depth, becoming experts
    at their single task rather than generalists.
    
    This pattern is essential for production systems where certain tasks are performed
    repeatedly and require consistent, high-quality execution. Examples include data
    validation agents, report generation agents, code review agents, content moderation
    agents, and translation agents.

Key Components:
    1. Task Definition: Clear specification of the task
    2. Optimized Workflow: Step-by-step task execution
    3. Task-Specific Tools: Specialized utilities
    4. Example Repository: Reference cases
    5. Validation Logic: Task-specific quality checks
    6. Error Handling: Common failure patterns
    7. Performance Metrics: Task success criteria

Task Specialization:
    1. Single Responsibility: One task, done well
    2. Optimized Prompts: Tailored to task requirements
    3. Specialized Tools: Task-specific utilities
    4. Domain Knowledge: Deep task understanding
    5. Workflow Efficiency: Streamlined execution
    6. Error Recovery: Task-specific failure handling
    7. Quality Assurance: Task validation

Task Categories:
    1. Data Processing: Validation, transformation, cleaning
    2. Content Generation: Reports, summaries, descriptions
    3. Analysis: Code review, sentiment analysis, quality assessment
    4. Transformation: Translation, format conversion, restructuring
    5. Validation: Compliance checking, verification, testing
    6. Interaction: Onboarding, support, guidance
    7. Monitoring: Alert processing, anomaly detection

Use Cases:
    - Data validation and cleaning
    - Automated report generation
    - Code review and analysis
    - Content moderation
    - Language translation
    - Email response generation
    - Invoice processing
    - Resume screening

Advantages:
    - Maximum task efficiency
    - Consistent quality
    - Deep task optimization
    - Reduced error rates
    - Fast execution
    - Reliable performance
    - Easy to test and validate

Challenges:
    - Limited flexibility
    - Task scope creep
    - Adaptation to task changes
    - Over-optimization risk
    - Maintenance overhead
    - Integration complexity
    - Version management

LangChain Implementation:
    This implementation uses LangChain for:
    - Task-optimized prompts
    - Specialized tool chains
    - Validation pipelines
    - Performance tracking
    
Production Considerations:
    - Version task-specific agents
    - Monitor task success rates
    - Track execution time
    - Log validation failures
    - Implement rollback mechanisms
    - A/B test task variations
    - Collect user feedback
    - Maintain task documentation
"""

import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"


class ValidationResult(Enum):
    """Validation outcome."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class TaskSpecification:
    """
    Specification for a task.
    
    Attributes:
        task_name: Name of the task
        description: Task description
        inputs: Required input fields
        outputs: Expected output fields
        validation_rules: Rules to validate outputs
        examples: Example inputs and outputs
        success_criteria: What defines success
        common_failures: Known failure patterns
    """
    task_name: str
    description: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    common_failures: List[str] = field(default_factory=list)


@dataclass
class TaskExecution:
    """
    Record of task execution.
    
    Attributes:
        execution_id: Unique identifier
        task_name: Name of executed task
        inputs: Input data
        outputs: Generated outputs
        status: Execution status
        validation: Validation results
        duration_seconds: Execution time
        errors: Any errors encountered
        timestamp: When executed
    """
    execution_id: str
    task_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    validation: Optional[ValidationResult] = None
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class TaskSpecificAgent:
    """
    Agent specialized for a specific task.
    
    This agent is optimized for executing a single task
    with maximum efficiency and reliability.
    """
    
    def __init__(
        self,
        task_spec: TaskSpecification,
        temperature: float = 0.3
    ):
        """
        Initialize task-specific agent.
        
        Args:
            task_spec: Task specification
            temperature: LLM temperature (lower for consistency)
        """
        self.task_spec = task_spec
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.execution_history: List[TaskExecution] = []
        self.execution_counter = 0
    
    def _build_task_prompt(self, inputs: Dict[str, Any]) -> str:
        """Build optimized prompt for task."""
        prompt_parts = [
            f"Task: {self.task_spec.task_name}",
            f"Description: {self.task_spec.description}",
            ""
        ]
        
        # Add success criteria
        if self.task_spec.success_criteria:
            criteria = "\n".join([f"  - {c}" for c in self.task_spec.success_criteria])
            prompt_parts.append(f"Success Criteria:\n{criteria}\n")
        
        # Add examples
        if self.task_spec.examples:
            prompt_parts.append("Examples:")
            for i, example in enumerate(self.task_spec.examples[:2], 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"  Input: {example.get('input', 'N/A')}")
                prompt_parts.append(f"  Output: {example.get('output', 'N/A')}")
            prompt_parts.append("")
        
        # Add common failures to avoid
        if self.task_spec.common_failures:
            failures = "\n".join([f"  - {f}" for f in self.task_spec.common_failures])
            prompt_parts.append(f"Avoid These Common Mistakes:\n{failures}\n")
        
        # Add current input
        prompt_parts.append("Current Input:")
        for key, value in inputs.items():
            prompt_parts.append(f"  {key}: {value}")
        
        prompt_parts.append("\nExecute the task and provide the output:")
        
        return "\n".join(prompt_parts)
    
    def execute(self, inputs: Dict[str, Any]) -> TaskExecution:
        """
        Execute the task.
        
        Args:
            inputs: Task inputs
            
        Returns:
            Task execution record
        """
        self.execution_counter += 1
        execution = TaskExecution(
            execution_id=f"exec_{self.execution_counter}",
            task_name=self.task_spec.task_name,
            inputs=inputs,
            status=TaskStatus.IN_PROGRESS
        )
        
        start_time = datetime.now()
        
        try:
            # Build prompt
            task_prompt = self._build_task_prompt(inputs)
            
            # Execute task
            prompt = ChatPromptTemplate.from_template("{task_prompt}")
            chain = prompt | self.llm | StrOutputParser()
            
            output_text = chain.invoke({"task_prompt": task_prompt})
            
            # Parse outputs
            outputs = self._parse_output(output_text)
            execution.outputs = outputs
            execution.status = TaskStatus.COMPLETED
            
        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.errors.append(str(e))
        
        # Calculate duration
        execution.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        # Store execution
        self.execution_history.append(execution)
        
        return execution
    
    def _parse_output(self, output_text: str) -> Dict[str, Any]:
        """Parse output from text response."""
        # Simple parsing - in production, use structured output
        outputs = {"raw_output": output_text}
        
        # Try to extract key-value pairs
        lines = output_text.split('\n')
        for line in lines:
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(' ', '_')
                    value = parts[1].strip()
                    outputs[key] = value
        
        return outputs
    
    def validate(self, execution: TaskExecution) -> ValidationResult:
        """
        Validate task execution.
        
        Args:
            execution: Execution to validate
            
        Returns:
            Validation result
        """
        if execution.status != TaskStatus.COMPLETED:
            execution.validation = ValidationResult.FAILED
            return ValidationResult.FAILED
        
        # Check required outputs
        missing_outputs = []
        for required_output in self.task_spec.outputs:
            if required_output not in execution.outputs:
                missing_outputs.append(required_output)
        
        if missing_outputs:
            execution.errors.append(f"Missing outputs: {missing_outputs}")
            execution.validation = ValidationResult.FAILED
            return ValidationResult.FAILED
        
        # Apply validation rules using LLM
        if self.task_spec.validation_rules:
            validation_prompt = (
                f"Validate this task output:\n\n"
                f"Task: {self.task_spec.task_name}\n"
                f"Output: {execution.outputs}\n\n"
                f"Validation Rules:\n"
            )
            for rule in self.task_spec.validation_rules:
                validation_prompt += f"- {rule}\n"
            
            validation_prompt += "\nDoes the output pass validation? Respond with PASSED, FAILED, or WARNING."
            
            prompt = ChatPromptTemplate.from_template("{validation_prompt}")
            chain = prompt | self.llm | StrOutputParser()
            
            result_text = chain.invoke({"validation_prompt": validation_prompt})
            
            if "PASSED" in result_text.upper():
                execution.validation = ValidationResult.PASSED
                execution.status = TaskStatus.VALIDATED
                return ValidationResult.PASSED
            elif "WARNING" in result_text.upper():
                execution.validation = ValidationResult.WARNING
                return ValidationResult.WARNING
            else:
                execution.validation = ValidationResult.FAILED
                return ValidationResult.FAILED
        
        # Default to passed if no rules
        execution.validation = ValidationResult.PASSED
        execution.status = TaskStatus.VALIDATED
        return ValidationResult.PASSED
    
    def execute_and_validate(self, inputs: Dict[str, Any]) -> TaskExecution:
        """Execute task and validate result."""
        execution = self.execute(inputs)
        self.validate(execution)
        return execution
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get task performance metrics."""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "validation_rate": 0.0,
                "avg_duration": 0.0
            }
        
        total = len(self.execution_history)
        completed = sum(1 for e in self.execution_history if e.status == TaskStatus.COMPLETED)
        validated = sum(1 for e in self.execution_history if e.validation == ValidationResult.PASSED)
        
        total_duration = sum(e.duration_seconds for e in self.execution_history)
        
        return {
            "total_executions": total,
            "success_rate": completed / total if total > 0 else 0.0,
            "validation_rate": validated / total if total > 0 else 0.0,
            "avg_duration": total_duration / total if total > 0 else 0.0,
            "task_name": self.task_spec.task_name
        }


def demonstrate_task_specific():
    """Demonstrate task-specific agent pattern."""
    
    print("=" * 80)
    print("TASK-SPECIFIC AGENT PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Email Response Task
    print("\n" + "=" * 80)
    print("Example 1: Email Response Generation Task")
    print("=" * 80)
    
    email_task = TaskSpecification(
        task_name="Customer Support Email Response",
        description="Generate professional customer support email responses",
        inputs=["customer_email", "issue_type", "customer_name"],
        outputs=["subject", "body", "tone"],
        validation_rules=[
            "Response must be polite and professional",
            "Must address the customer by name",
            "Must include a clear solution or next steps",
            "Must be under 200 words"
        ],
        examples=[
            {
                "input": {
                    "customer_email": "My order hasn't arrived",
                    "issue_type": "shipping_delay",
                    "customer_name": "John"
                },
                "output": {
                    "subject": "Update on Your Order Status",
                    "body": "Dear John, I apologize for the delay...",
                    "tone": "empathetic and solution-focused"
                }
            }
        ],
        success_criteria=[
            "Customer feels heard and valued",
            "Clear action items provided",
            "Professional tone maintained"
        ],
        common_failures=[
            "Being too formal or robotic",
            "Not acknowledging the customer's frustration",
            "Providing vague next steps"
        ]
    )
    
    email_agent = TaskSpecificAgent(email_task)
    
    print(f"\nTask: {email_task.task_name}")
    print(f"Description: {email_task.description}")
    print(f"Required Inputs: {', '.join(email_task.inputs)}")
    print(f"Expected Outputs: {', '.join(email_task.outputs)}")
    
    # Example 2: Execute task
    print("\n" + "=" * 80)
    print("Example 2: Task Execution")
    print("=" * 80)
    
    customer_input = {
        "customer_email": "I'm frustrated! My package was supposed to arrive 3 days ago!",
        "issue_type": "shipping_delay",
        "customer_name": "Sarah"
    }
    
    print("\nInput:")
    for key, value in customer_input.items():
        print(f"  {key}: {value}")
    
    execution = email_agent.execute(customer_input)
    
    print(f"\nExecution Status: {execution.status.value}")
    print(f"Duration: {execution.duration_seconds:.2f}s")
    print("\nGenerated Output:")
    for key, value in execution.outputs.items():
        if key != "raw_output":
            print(f"  {key}: {value}")
    
    # Example 3: Validation
    print("\n" + "=" * 80)
    print("Example 3: Output Validation")
    print("=" * 80)
    
    validation_result = email_agent.validate(execution)
    
    print(f"\nValidation Result: {validation_result.value}")
    print(f"Final Status: {execution.status.value}")
    
    if execution.errors:
        print(f"Errors: {execution.errors}")
    
    # Example 4: Data Validation Task
    print("\n" + "=" * 80)
    print("Example 4: Data Validation Task")
    print("=" * 80)
    
    validation_task = TaskSpecification(
        task_name="User Registration Data Validation",
        description="Validate user registration data for completeness and correctness",
        inputs=["email", "username", "age", "country"],
        outputs=["is_valid", "errors", "warnings"],
        validation_rules=[
            "Email must be valid format",
            "Username must be 3-20 characters",
            "Age must be 18 or older",
            "Country must be a valid country code"
        ],
        success_criteria=[
            "All validation rules checked",
            "Clear error messages provided",
            "Actionable feedback given"
        ],
        common_failures=[
            "Missing validation checks",
            "Vague error messages",
            "Not checking all fields"
        ]
    )
    
    validator_agent = TaskSpecificAgent(validation_task, temperature=0.1)
    
    print(f"\nTask: {validation_task.task_name}")
    
    # Valid data
    valid_data = {
        "email": "user@example.com",
        "username": "johndoe",
        "age": "25",
        "country": "US"
    }
    
    print("\nValidating Valid Data:")
    print(f"  {valid_data}")
    
    execution1 = validator_agent.execute_and_validate(valid_data)
    print(f"Result: {execution1.validation.value if execution1.validation else 'N/A'}")
    
    # Invalid data
    invalid_data = {
        "email": "not-an-email",
        "username": "ab",
        "age": "15",
        "country": "XYZ"
    }
    
    print("\nValidating Invalid Data:")
    print(f"  {invalid_data}")
    
    execution2 = validator_agent.execute_and_validate(invalid_data)
    print(f"Result: {execution2.validation.value if execution2.validation else 'N/A'}")
    
    # Example 5: Code Review Task
    print("\n" + "=" * 80)
    print("Example 5: Code Review Task")
    print("=" * 80)
    
    code_review_task = TaskSpecification(
        task_name="Python Code Review",
        description="Review Python code for quality, security, and best practices",
        inputs=["code", "context"],
        outputs=["issues", "suggestions", "severity", "approval"],
        validation_rules=[
            "Must identify security issues",
            "Must check for best practices",
            "Must provide actionable feedback"
        ],
        success_criteria=[
            "All major issues identified",
            "Suggestions are specific",
            "Severity levels assigned"
        ]
    )
    
    reviewer_agent = TaskSpecificAgent(code_review_task)
    
    code_sample = """
def process_user_input(input):
    result = eval(input)  # Security issue!
    return result
"""
    
    print(f"\nReviewing Code:")
    print(code_sample)
    
    review_execution = reviewer_agent.execute({
        "code": code_sample,
        "context": "User input processing function"
    })
    
    print(f"\nReview Status: {review_execution.status.value}")
    print("Review Output:")
    print(review_execution.outputs.get("raw_output", "No output")[:500])
    
    # Example 6: Performance metrics
    print("\n" + "=" * 80)
    print("Example 6: Task Performance Metrics")
    print("=" * 80)
    
    # Execute multiple tasks
    test_inputs = [
        {"customer_email": "Test 1", "issue_type": "general", "customer_name": "User1"},
        {"customer_email": "Test 2", "issue_type": "technical", "customer_name": "User2"},
        {"customer_email": "Test 3", "issue_type": "billing", "customer_name": "User3"}
    ]
    
    for inputs in test_inputs:
        email_agent.execute_and_validate(inputs)
    
    metrics = email_agent.get_performance_metrics()
    
    print("\nPERFORMANCE METRICS:")
    print("=" * 60)
    print(f"Task: {metrics['task_name']}")
    print(f"Total Executions: {metrics['total_executions']}")
    print(f"Success Rate: {metrics['success_rate']:.1%}")
    print(f"Validation Rate: {metrics['validation_rate']:.1%}")
    print(f"Average Duration: {metrics['avg_duration']:.2f}s")
    
    # Example 7: Translation Task
    print("\n" + "=" * 80)
    print("Example 7: Translation Task")
    print("=" * 80)
    
    translation_task = TaskSpecification(
        task_name="English to Spanish Translation",
        description="Translate English text to Spanish maintaining tone and context",
        inputs=["english_text", "context"],
        outputs=["spanish_text", "confidence"],
        validation_rules=[
            "Translation must preserve meaning",
            "Must maintain appropriate tone",
            "Must be grammatically correct"
        ],
        success_criteria=[
            "Accurate translation",
            "Natural Spanish phrasing",
            "Context-appropriate"
        ]
    )
    
    translator = TaskSpecificAgent(translation_task)
    
    translation_input = {
        "english_text": "Hello, how can I help you today?",
        "context": "Customer service greeting"
    }
    
    print("\nTranslation Input:")
    print(f"  Text: {translation_input['english_text']}")
    print(f"  Context: {translation_input['context']}")
    
    translation_result = translator.execute_and_validate(translation_input)
    print(f"\nTranslation Status: {translation_result.status.value}")
    print("Translation Output:")
    for key, value in translation_result.outputs.items():
        if key != "raw_output":
            print(f"  {key}: {value}")
    
    # Example 8: Summary task
    print("\n" + "=" * 80)
    print("Example 8: Task Agent Summary")
    print("=" * 80)
    
    print("\nDEPLOYED TASK-SPECIFIC AGENTS:")
    print("=" * 60)
    
    agents_summary = [
        ("Email Response", email_agent),
        ("Data Validation", validator_agent),
        ("Code Review", reviewer_agent),
        ("Translation", translator)
    ]
    
    for name, agent in agents_summary:
        metrics = agent.get_performance_metrics()
        print(f"\n{name}:")
        print(f"  Executions: {metrics['total_executions']}")
        print(f"  Success Rate: {metrics['success_rate']:.1%}")
        print(f"  Avg Duration: {metrics['avg_duration']:.2f}s")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Task-Specific Agent Pattern")
    print("=" * 80)
    
    summary = """
    The Task-Specific Agent pattern demonstrated:
    
    1. TASK SPECIALIZATION (Example 1):
       - Single-purpose agents
       - Optimized for specific tasks
       - Task-specific prompts
       - Validation rules
       - Success criteria
       - Common failure awareness
    
    2. TASK EXECUTION (Example 2):
       - Optimized workflow
       - Input processing
       - Output generation
       - Performance tracking
       - Status monitoring
    
    3. OUTPUT VALIDATION (Example 3):
       - Rule-based validation
       - Required output checking
       - Quality assurance
       - Error detection
       - Validation feedback
    
    4. DATA VALIDATION TASK (Example 4):
       - Field validation
       - Format checking
       - Business rule enforcement
       - Error reporting
       - Actionable feedback
    
    5. CODE REVIEW TASK (Example 5):
       - Security analysis
       - Best practice checking
       - Issue identification
       - Severity assessment
       - Specific suggestions
    
    6. PERFORMANCE TRACKING (Example 6):
       - Execution metrics
       - Success rates
       - Duration monitoring
       - Quality metrics
       - Continuous improvement
    
    7. TRANSLATION TASK (Example 7):
       - Language conversion
       - Context preservation
       - Tone maintenance
       - Accuracy validation
       - Confidence scoring
    
    8. MULTI-TASK DEPLOYMENT (Example 8):
       - Multiple specialized agents
       - Independent optimization
       - Performance comparison
       - Resource allocation
       - System overview
    
    KEY BENEFITS:
    ✓ Maximum task efficiency
    ✓ Consistent quality
    ✓ Deep optimization
    ✓ Reduced errors
    ✓ Fast execution
    ✓ Reliable performance
    ✓ Easy testing
    ✓ Clear metrics
    
    USE CASES:
    • Data validation and cleaning
    • Automated report generation
    • Code review and analysis
    • Content moderation
    • Language translation
    • Email response generation
    • Invoice processing
    • Resume screening
    
    TASK CATEGORIES:
    → Data Processing: Validation, transformation
    → Content Generation: Reports, summaries
    → Analysis: Code review, quality assessment
    → Transformation: Translation, conversion
    → Validation: Compliance, verification
    → Interaction: Support, guidance
    → Monitoring: Alerts, anomaly detection
    
    SPECIALIZATION ASPECTS:
    • Single responsibility principle
    • Optimized prompts
    • Specialized tools
    • Domain knowledge
    • Workflow efficiency
    • Error recovery
    • Quality assurance
    
    BEST PRACTICES:
    1. Define clear task boundaries
    2. Optimize for single task
    3. Include comprehensive examples
    4. Implement thorough validation
    5. Track performance metrics
    6. Handle common failures
    7. Version control task specs
    8. Document expected behaviors
    
    TRADE-OFFS:
    • Specialization vs. flexibility
    • Efficiency vs. adaptability
    • Depth vs. breadth
    • Optimization vs. generalization
    
    PRODUCTION CONSIDERATIONS:
    → Version task agents independently
    → Monitor success rates per task
    → Track execution times
    → Log validation failures
    → Implement rollback capabilities
    → A/B test task variations
    → Collect user feedback
    → Maintain task documentation
    → Set performance SLAs
    → Enable task composition
    
    This pattern enables highly efficient, reliable execution of specific tasks
    by sacrificing generality for optimization, resulting in production-grade
    agents that excel at their designated operations.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_task_specific()
