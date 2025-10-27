"""
Dynamic Prompt Template Generator Pattern

Generates and optimizes prompt templates dynamically based on task requirements.
Adapts prompts based on model capabilities and performance feedback.

Use Cases:
- LLM application development
- Prompt engineering automation
- Multi-model deployment
- A/B testing prompts

Advantages:
- Automated prompt optimization
- Task-specific templates
- Model-aware formatting
- Performance-driven refinement
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import re


class TaskType(Enum):
    """Types of tasks for prompt generation"""
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    GENERATION = "generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "question_answering"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"


class ModelCapability(Enum):
    """Model capabilities and constraints"""
    INSTRUCTION_FOLLOWING = "instruction_following"
    FEW_SHOT_LEARNING = "few_shot_learning"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    JSON_OUTPUT = "json_output"
    CODE_UNDERSTANDING = "code_understanding"
    LONG_CONTEXT = "long_context"


@dataclass
class PromptTemplate:
    """Prompt template with metadata"""
    template_id: str
    task_type: TaskType
    template: str
    variables: List[str]
    model_constraints: Dict[str, Any]
    performance_score: float = 0.0
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    examples: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PromptPerformance:
    """Performance metrics for a prompt"""
    template_id: str
    task_type: TaskType
    success_rate: float
    avg_latency: float
    avg_quality_score: float
    total_runs: int
    timestamp: datetime


@dataclass
class TemplateComponent:
    """Reusable template component"""
    component_id: str
    name: str
    content: str
    variables: List[str]
    applicable_tasks: List[TaskType]


class PromptOptimizer:
    """Optimizes prompts based on performance feedback"""
    
    def __init__(self):
        self.optimization_strategies = {
            "add_examples": self._add_examples,
            "add_constraints": self._add_constraints,
            "add_format_spec": self._add_format_specification,
            "add_reasoning": self._add_reasoning_steps,
            "simplify": self._simplify_prompt
        }
    
    def optimize_prompt(self,
                       template: PromptTemplate,
                       performance: PromptPerformance) -> PromptTemplate:
        """Optimize prompt based on performance"""
        optimized = template
        
        # Low success rate
        if performance.success_rate < 0.7:
            # Try adding examples
            if len(template.examples) < 3:
                optimized = self._add_examples(optimized)
            # Try adding constraints
            else:
                optimized = self._add_constraints(optimized)
        
        # Low quality
        if performance.avg_quality_score < 0.6:
            # Add format specification
            optimized = self._add_format_specification(optimized)
        
        # High latency
        if performance.avg_latency > 5.0:
            # Simplify prompt
            optimized = self._simplify_prompt(optimized)
        
        return optimized
    
    def _add_examples(self, template: PromptTemplate) -> PromptTemplate:
        """Add few-shot examples to prompt"""
        if not template.examples:
            return template
        
        examples_text = "\n\nExamples:\n"
        for i, example in enumerate(template.examples[:3], 1):
            examples_text += "\nExample {}:\n".format(i)
            for key, value in example.items():
                examples_text += "{}: {}\n".format(key, value)
        
        new_template = template.template + examples_text
        
        return PromptTemplate(
            template_id=template.template_id + "_with_examples",
            task_type=template.task_type,
            template=new_template,
            variables=template.variables,
            model_constraints=template.model_constraints,
            examples=template.examples
        )
    
    def _add_constraints(self, template: PromptTemplate) -> PromptTemplate:
        """Add explicit constraints to prompt"""
        constraints = """

Important constraints:
- Be specific and accurate
- Follow the exact format requested
- Do not add information not present in the input
- If uncertain, indicate your uncertainty
"""
        
        new_template = template.template + constraints
        
        return PromptTemplate(
            template_id=template.template_id + "_with_constraints",
            task_type=template.task_type,
            template=new_template,
            variables=template.variables,
            model_constraints=template.model_constraints,
            examples=template.examples
        )
    
    def _add_format_specification(self,
                                   template: PromptTemplate
                                   ) -> PromptTemplate:
        """Add format specification to prompt"""
        if template.task_type == TaskType.EXTRACTION:
            format_spec = """

Output format:
Return a JSON object with the following structure:
{
  "field1": "value1",
  "field2": "value2"
}
"""
        elif template.task_type == TaskType.CLASSIFICATION:
            format_spec = """

Output format:
Return only the classification label, no explanation.
"""
        else:
            format_spec = """

Output format:
Provide your answer clearly and concisely.
"""
        
        new_template = template.template + format_spec
        
        return PromptTemplate(
            template_id=template.template_id + "_with_format",
            task_type=template.task_type,
            template=new_template,
            variables=template.variables,
            model_constraints=template.model_constraints,
            examples=template.examples
        )
    
    def _add_reasoning_steps(self,
                            template: PromptTemplate
                            ) -> PromptTemplate:
        """Add reasoning steps to prompt"""
        reasoning = """

Think step by step:
1. First, analyze the input carefully
2. Then, identify the key information
3. Finally, provide your answer based on your analysis
"""
        
        new_template = template.template + reasoning
        
        return PromptTemplate(
            template_id=template.template_id + "_with_reasoning",
            task_type=template.task_type,
            template=new_template,
            variables=template.variables,
            model_constraints=template.model_constraints,
            examples=template.examples
        )
    
    def _simplify_prompt(self, template: PromptTemplate) -> PromptTemplate:
        """Simplify prompt by removing redundancy"""
        # Remove multiple newlines
        simplified = re.sub(r'\n\n+', '\n\n', template.template)
        
        # Remove verbose phrases
        verbose_phrases = [
            "please note that",
            "it is important to",
            "make sure to",
            "remember that"
        ]
        for phrase in verbose_phrases:
            simplified = simplified.replace(phrase, "")
        
        return PromptTemplate(
            template_id=template.template_id + "_simplified",
            task_type=template.task_type,
            template=simplified,
            variables=template.variables,
            model_constraints=template.model_constraints,
            examples=template.examples
        )


class DynamicPromptTemplateGenerator:
    """
    Generates and manages dynamic prompt templates.
    Adapts templates based on task requirements and performance.
    """
    
    def __init__(self):
        self.templates: Dict[str, PromptTemplate] = {}
        self.components: Dict[str, TemplateComponent] = {}
        self.performance_history: List[PromptPerformance] = []
        self.optimizer = PromptOptimizer()
        
        # Initialize base components
        self._initialize_components()
        
        # Initialize base templates
        self._initialize_base_templates()
    
    def generate_template(self,
                         task_type: TaskType,
                         requirements: Dict[str, Any],
                         model_capabilities: Optional[List[ModelCapability]] = None
                         ) -> PromptTemplate:
        """
        Generate a prompt template for a specific task.
        
        Args:
            task_type: Type of task
            requirements: Task-specific requirements
            model_capabilities: Optional model capabilities
            
        Returns:
            Generated prompt template
        """
        if model_capabilities is None:
            model_capabilities = []
        
        # Start with base template
        base_template = self._get_base_template(task_type)
        
        # Add task-specific components
        template_parts = [base_template]
        
        # Add input specification
        if requirements.get("input_description"):
            input_spec = self._generate_input_spec(
                requirements["input_description"]
            )
            template_parts.append(input_spec)
        
        # Add output specification
        if requirements.get("output_format"):
            output_spec = self._generate_output_spec(
                requirements["output_format"]
            )
            template_parts.append(output_spec)
        
        # Add constraints
        if requirements.get("constraints"):
            constraints = self._generate_constraints(
                requirements["constraints"]
            )
            template_parts.append(constraints)
        
        # Add examples if few-shot learning supported
        examples_text = ""
        if (ModelCapability.FEW_SHOT_LEARNING in model_capabilities and
            requirements.get("examples")):
            examples_text = self._format_examples(
                requirements["examples"]
            )
            template_parts.append(examples_text)
        
        # Add chain-of-thought if supported
        if (ModelCapability.CHAIN_OF_THOUGHT in model_capabilities and
            requirements.get("require_reasoning")):
            cot = self.components["chain_of_thought"].content
            template_parts.append(cot)
        
        # Combine parts
        full_template = "\n\n".join(template_parts)
        
        # Extract variables
        variables = self._extract_variables(full_template)
        
        # Create template
        template = PromptTemplate(
            template_id="template_{}_{}".format(
                task_type.value,
                len(self.templates)
            ),
            task_type=task_type,
            template=full_template,
            variables=variables,
            model_constraints={
                "capabilities": [c.value for c in model_capabilities],
                "max_tokens": requirements.get("max_tokens", 2048)
            },
            examples=requirements.get("examples", [])
        )
        
        # Store template
        self.templates[template.template_id] = template
        
        return template
    
    def fill_template(self,
                     template: PromptTemplate,
                     values: Dict[str, str]) -> str:
        """
        Fill template with values.
        
        Args:
            template: Prompt template
            values: Variable values
            
        Returns:
            Filled prompt
        """
        prompt = template.template
        
        # Replace variables
        for variable in template.variables:
            placeholder = "{" + variable + "}"
            if variable in values:
                prompt = prompt.replace(placeholder, values[variable])
            else:
                raise ValueError("Missing value for variable: {}".format(variable))
        
        # Update usage count
        template.usage_count += 1
        
        return prompt
    
    def record_performance(self,
                          template_id: str,
                          success: bool,
                          latency: float,
                          quality_score: float) -> None:
        """
        Record template performance.
        
        Args:
            template_id: Template identifier
            success: Whether execution was successful
            latency: Execution latency
            quality_score: Quality score (0-1)
        """
        template = self.templates.get(template_id)
        if not template:
            return
        
        # Update template performance
        template.performance_score = (
            (template.performance_score * template.usage_count +
             (1.0 if success else 0.0)) /
            (template.usage_count + 1)
        )
        
        # Record in history
        performance = PromptPerformance(
            template_id=template_id,
            task_type=template.task_type,
            success_rate=template.performance_score,
            avg_latency=latency,
            avg_quality_score=quality_score,
            total_runs=template.usage_count,
            timestamp=datetime.now()
        )
        self.performance_history.append(performance)
    
    def optimize_template(self, template_id: str) -> Optional[PromptTemplate]:
        """
        Optimize a template based on performance history.
        
        Args:
            template_id: Template to optimize
            
        Returns:
            Optimized template or None
        """
        template = self.templates.get(template_id)
        if not template:
            return None
        
        # Get recent performance
        recent_performance = [
            p for p in self.performance_history[-10:]
            if p.template_id == template_id
        ]
        
        if not recent_performance:
            return None
        
        # Aggregate performance
        avg_performance = PromptPerformance(
            template_id=template_id,
            task_type=template.task_type,
            success_rate=sum(p.success_rate for p in recent_performance) / len(recent_performance),
            avg_latency=sum(p.avg_latency for p in recent_performance) / len(recent_performance),
            avg_quality_score=sum(p.avg_quality_score for p in recent_performance) / len(recent_performance),
            total_runs=sum(p.total_runs for p in recent_performance),
            timestamp=datetime.now()
        )
        
        # Optimize
        optimized = self.optimizer.optimize_prompt(template, avg_performance)
        
        # Store optimized version
        self.templates[optimized.template_id] = optimized
        
        return optimized
    
    def get_best_template(self, task_type: TaskType) -> Optional[PromptTemplate]:
        """
        Get best performing template for task type.
        
        Args:
            task_type: Task type
            
        Returns:
            Best template or None
        """
        task_templates = [
            t for t in self.templates.values()
            if t.task_type == task_type and t.usage_count > 0
        ]
        
        if not task_templates:
            return None
        
        # Sort by performance
        task_templates.sort(key=lambda t: t.performance_score, reverse=True)
        
        return task_templates[0]
    
    def _initialize_components(self) -> None:
        """Initialize reusable template components"""
        self.components = {
            "chain_of_thought": TemplateComponent(
                component_id="cot_1",
                name="Chain of Thought",
                content="Let's think step by step:\n1. {step1}\n2. {step2}\n3. {step3}",
                variables=["step1", "step2", "step3"],
                applicable_tasks=[TaskType.REASONING, TaskType.QUESTION_ANSWERING]
            ),
            "json_output": TemplateComponent(
                component_id="json_1",
                name="JSON Output Format",
                content="Return the result as a valid JSON object.",
                variables=[],
                applicable_tasks=[TaskType.EXTRACTION, TaskType.CLASSIFICATION]
            ),
            "constraints": TemplateComponent(
                component_id="constraints_1",
                name="General Constraints",
                content="Important: Be accurate and concise. Do not make up information.",
                variables=[],
                applicable_tasks=list(TaskType)
            )
        }
    
    def _initialize_base_templates(self) -> None:
        """Initialize base templates for common tasks"""
        base_templates = {
            TaskType.CLASSIFICATION: "Classify the following {input_type} into one of these categories: {categories}.\n\nInput: {input}",
            TaskType.EXTRACTION: "Extract {fields} from the following {input_type}.\n\nInput: {input}",
            TaskType.GENERATION: "Generate {output_type} based on the following requirements:\n\nRequirements: {requirements}",
            TaskType.SUMMARIZATION: "Summarize the following {input_type} in {length}.\n\nInput: {input}",
            TaskType.TRANSLATION: "Translate the following text from {source_lang} to {target_lang}.\n\nText: {input}",
            TaskType.QUESTION_ANSWERING: "Answer the following question based on the given context.\n\nContext: {context}\n\nQuestion: {question}",
            TaskType.CODE_GENERATION: "Generate {language} code to {task_description}.\n\nRequirements:\n{requirements}",
            TaskType.REASONING: "Solve the following problem using logical reasoning.\n\nProblem: {problem}"
        }
        
        for task_type, template in base_templates.items():
            variables = self._extract_variables(template)
            self.templates["base_" + task_type.value] = PromptTemplate(
                template_id="base_" + task_type.value,
                task_type=task_type,
                template=template,
                variables=variables,
                model_constraints={}
            )
    
    def _get_base_template(self, task_type: TaskType) -> str:
        """Get base template for task type"""
        template_id = "base_" + task_type.value
        template = self.templates.get(template_id)
        return template.template if template else ""
    
    def _generate_input_spec(self, description: str) -> str:
        """Generate input specification"""
        return "Input description: {}".format(description)
    
    def _generate_output_spec(self, format_spec: str) -> str:
        """Generate output specification"""
        return "Expected output format:\n{}".format(format_spec)
    
    def _generate_constraints(self, constraints: List[str]) -> str:
        """Generate constraints section"""
        return "Constraints:\n" + "\n".join(
            "- {}".format(c) for c in constraints
        )
    
    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format examples for few-shot learning"""
        formatted = "Examples:\n"
        for i, example in enumerate(examples, 1):
            formatted += "\nExample {}:\n".format(i)
            for key, value in example.items():
                formatted += "{}: {}\n".format(key, value)
        return formatted
    
    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template"""
        pattern = r'\{([^}]+)\}'
        variables = re.findall(pattern, template)
        return list(set(variables))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get template statistics"""
        task_counts = {}
        for template in self.templates.values():
            task = template.task_type.value
            task_counts[task] = task_counts.get(task, 0) + 1
        
        return {
            "total_templates": len(self.templates),
            "templates_by_task": task_counts,
            "total_components": len(self.components),
            "performance_records": len(self.performance_history),
            "avg_performance": (
                sum(t.performance_score for t in self.templates.values()
                    if t.usage_count > 0) /
                len([t for t in self.templates.values() if t.usage_count > 0])
                if any(t.usage_count > 0 for t in self.templates.values())
                else 0.0
            )
        }


def demonstrate_dynamic_prompt_template():
    """Demonstrate dynamic prompt template generator"""
    print("=" * 70)
    print("Dynamic Prompt Template Generator Demonstration")
    print("=" * 70)
    
    generator = DynamicPromptTemplateGenerator()
    
    # Example 1: Generate classification template
    print("\n1. Generating Classification Template:")
    requirements1 = {
        "input_description": "customer review text",
        "output_format": "sentiment label (positive/negative/neutral)",
        "constraints": [
            "Consider context and tone",
            "Return only the label"
        ],
        "examples": [
            {"input": "Great product!", "output": "positive"},
            {"input": "Terrible experience", "output": "negative"}
        ]
    }
    
    template1 = generator.generate_template(
        task_type=TaskType.CLASSIFICATION,
        requirements=requirements1,
        model_capabilities=[
            ModelCapability.INSTRUCTION_FOLLOWING,
            ModelCapability.FEW_SHOT_LEARNING
        ]
    )
    
    print("Template ID: {}".format(template1.template_id))
    print("Variables: {}".format(template1.variables))
    print("\nTemplate preview:")
    print("-" * 60)
    print(template1.template[:400])
    print("...")
    
    # Example 2: Fill template with values
    print("\n2. Filling Template with Values:")
    filled_prompt = generator.fill_template(
        template1,
        {
            "input_type": "customer review",
            "categories": "positive, negative, neutral",
            "input": "This is an amazing product, highly recommended!"
        }
    )
    print("Filled prompt:")
    print("-" * 60)
    print(filled_prompt[:300])
    print("...")
    
    # Example 3: Generate extraction template
    print("\n3. Generating Extraction Template:")
    requirements2 = {
        "input_description": "unstructured text containing personal information",
        "output_format": "JSON with fields: name, email, phone",
        "constraints": [
            "Extract only if information is present",
            "Return null for missing fields"
        ]
    }
    
    template2 = generator.generate_template(
        task_type=TaskType.EXTRACTION,
        requirements=requirements2,
        model_capabilities=[
            ModelCapability.INSTRUCTION_FOLLOWING,
            ModelCapability.JSON_OUTPUT
        ]
    )
    
    print("Template ID: {}".format(template2.template_id))
    print("Variables: {}".format(template2.variables))
    
    # Example 4: Record performance
    print("\n4. Recording Template Performance:")
    generator.record_performance(
        template_id=template1.template_id,
        success=True,
        latency=1.2,
        quality_score=0.85
    )
    generator.record_performance(
        template_id=template1.template_id,
        success=True,
        latency=1.5,
        quality_score=0.90
    )
    print("Recorded 2 performance samples")
    print("Current performance score: {:.2f}".format(
        template1.performance_score
    ))
    
    # Example 5: Get best template
    print("\n5. Getting Best Template for Task:")
    best = generator.get_best_template(TaskType.CLASSIFICATION)
    if best:
        print("Best template: {}".format(best.template_id))
        print("Performance: {:.2f}".format(best.performance_score))
        print("Usage count: {}".format(best.usage_count))
    
    # Example 6: Statistics
    print("\n6. Template Statistics:")
    stats = generator.get_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    demonstrate_dynamic_prompt_template()
