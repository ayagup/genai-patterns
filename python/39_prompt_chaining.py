"""
Prompt Chaining Pattern
Links multiple prompts sequentially, output feeds into next
"""
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
@dataclass
class PromptStep:
    """A single step in the prompt chain"""
    name: str
    prompt_template: str
    process_output: Optional[Callable[[str], Any]] = None
    required_inputs: List[str] = None
    def __post_init__(self):
        if self.required_inputs is None:
            self.required_inputs = []
@dataclass
class ChainResult:
    """Result from a chain execution"""
    step_name: str
    input_data: Dict[str, Any]
    output: Any
    timestamp: datetime
    duration_ms: float
class PromptChain:
    """Chain of prompts where each output feeds into the next"""
    def __init__(self, chain_name: str):
        self.chain_name = chain_name
        self.steps: List[PromptStep] = []
        self.results: List[ChainResult] = []
        self.context: Dict[str, Any] = {}
    def add_step(self, step: PromptStep):
        """Add a step to the chain"""
        self.steps.append(step)
        print(f"Added step: {step.name}")
    def execute(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the entire chain"""
        print(f"\n{'='*70}")
        print(f"EXECUTING CHAIN: {self.chain_name}")
        print(f"{'='*70}")
        print(f"Steps: {len(self.steps)}")
        print(f"Initial Input: {initial_input}\n")
        # Initialize context
        self.context.update(initial_input)
        # Execute each step
        for i, step in enumerate(self.steps, 1):
            print(f"\n--- Step {i}/{len(self.steps)}: {step.name} ---")
            # Check required inputs
            missing = [inp for inp in step.required_inputs if inp not in self.context]
            if missing:
                raise ValueError(f"Missing required inputs: {missing}")
            # Prepare input
            step_input = {k: self.context[k] for k in step.required_inputs} if step.required_inputs else self.context
            print(f"Input: {step_input}")
            # Execute step
            import time
            start_time = time.time()
            output = self._execute_step(step, step_input)
            duration_ms = (time.time() - start_time) * 1000
            print(f"Output: {output}")
            print(f"Duration: {duration_ms:.2f}ms")
            # Store result
            result = ChainResult(
                step_name=step.name,
                input_data=step_input.copy(),
                output=output,
                timestamp=datetime.now(),
                duration_ms=duration_ms
            )
            self.results.append(result)
            # Update context
            self.context[step.name] = output
        # Prepare final result
        final_result = {
            'chain_name': self.chain_name,
            'steps_executed': len(self.results),
            'final_output': self.results[-1].output if self.results else None,
            'context': self.context,
            'execution_trace': [
                {
                    'step': r.step_name,
                    'output': r.output,
                    'duration_ms': r.duration_ms
                }
                for r in self.results
            ]
        }
        return final_result
    def _execute_step(self, step: PromptStep, input_data: Dict[str, Any]) -> Any:
        """Execute a single step"""
        # Fill prompt template
        prompt = step.prompt_template.format(**input_data)
        # Simulate LLM call (in reality, call actual LLM)
        raw_output = self._simulate_llm_call(prompt)
        # Process output if processor provided
        if step.process_output:
            output = step.process_output(raw_output)
        else:
            output = raw_output
        return output
    def _simulate_llm_call(self, prompt: str) -> str:
        """Simulate LLM API call"""
        import time
        time.sleep(0.1)  # Simulate latency
        # Simple simulation based on prompt content
        if "extract" in prompt.lower():
            return "Extracted data: key points identified"
        elif "analyze" in prompt.lower():
            return "Analysis: positive sentiment, high engagement"
        elif "summarize" in prompt.lower():
            return "Summary: Main points condensed into brief overview"
        elif "generate" in prompt.lower():
            return "Generated content based on analysis"
        else:
            return f"Response to: {prompt[:50]}..."
class PromptChainingAgent:
    """Agent that uses prompt chaining for complex tasks"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    def analyze_document(self, document: str) -> Dict[str, Any]:
        """Analyze document using prompt chain"""
        chain = PromptChain("Document Analysis Pipeline")
        # Step 1: Extract key information
        chain.add_step(PromptStep(
            name="extraction",
            prompt_template="Extract key information from: {document}",
            required_inputs=["document"]
        ))
        # Step 2: Analyze sentiment
        chain.add_step(PromptStep(
            name="sentiment",
            prompt_template="Analyze sentiment of: {extraction}",
            required_inputs=["extraction"]
        ))
        # Step 3: Generate summary
        chain.add_step(PromptStep(
            name="summary",
            prompt_template="Summarize findings from extraction: {extraction} and sentiment: {sentiment}",
            required_inputs=["extraction", "sentiment"]
        ))
        # Step 4: Create recommendations
        chain.add_step(PromptStep(
            name="recommendations",
            prompt_template="Based on summary: {summary}, provide recommendations",
            required_inputs=["summary"]
        ))
        # Execute chain
        result = chain.execute({"document": document})
        return result
    def create_content(self, topic: str, audience: str) -> Dict[str, Any]:
        """Create content using prompt chain"""
        chain = PromptChain("Content Creation Pipeline")
        # Step 1: Research
        chain.add_step(PromptStep(
            name="research",
            prompt_template="Research key points about {topic} for {audience}",
            required_inputs=["topic", "audience"]
        ))
        # Step 2: Outline
        chain.add_step(PromptStep(
            name="outline",
            prompt_template="Create outline based on research: {research}",
            required_inputs=["research"]
        ))
        # Step 3: Draft
        chain.add_step(PromptStep(
            name="draft",
            prompt_template="Write draft following outline: {outline} for {audience}",
            required_inputs=["outline", "audience"]
        ))
        # Step 4: Edit
        chain.add_step(PromptStep(
            name="edited",
            prompt_template="Edit and improve draft: {draft}",
            required_inputs=["draft"]
        ))
        # Step 5: Format
        chain.add_step(PromptStep(
            name="formatted",
            prompt_template="Format content for {audience}: {edited}",
            required_inputs=["edited", "audience"]
        ))
        # Execute chain
        result = chain.execute({"topic": topic, "audience": audience})
        return result
    def solve_complex_problem(self, problem: str) -> Dict[str, Any]:
        """Solve complex problem using prompt chain"""
        chain = PromptChain("Problem Solving Pipeline")
        # Step 1: Understand problem
        chain.add_step(PromptStep(
            name="understanding",
            prompt_template="Clarify and restate problem: {problem}",
            required_inputs=["problem"]
        ))
        # Step 2: Break down
        chain.add_step(PromptStep(
            name="breakdown",
            prompt_template="Break down problem into components: {understanding}",
            required_inputs=["understanding"]
        ))
        # Step 3: Generate solutions
        chain.add_step(PromptStep(
            name="solutions",
            prompt_template="Generate possible solutions for: {breakdown}",
            required_inputs=["breakdown"]
        ))
        # Step 4: Evaluate
        chain.add_step(PromptStep(
            name="evaluation",
            prompt_template="Evaluate solutions: {solutions}",
            required_inputs=["solutions"]
        ))
        # Step 5: Select best
        chain.add_step(PromptStep(
            name="recommendation",
            prompt_template="Recommend best solution from evaluation: {evaluation}",
            required_inputs=["evaluation"]
        ))
        # Execute chain
        result = chain.execute({"problem": problem})
        return result
# Usage
if __name__ == "__main__":
    print("="*80)
    print("PROMPT CHAINING PATTERN DEMONSTRATION")
    print("="*80)
    agent = PromptChainingAgent("chain-agent-001")
    # Example 1: Document Analysis
    print("\n" + "="*80)
    print("EXAMPLE 1: Document Analysis Chain")
    print("="*80)
    result1 = agent.analyze_document(
        "Customer feedback indicates high satisfaction with product quality "
        "but concerns about delivery times and customer support responsiveness."
    )
    print(f"\n{'='*70}")
    print("CHAIN SUMMARY")
    print(f"{'='*70}")
    print(f"Steps Executed: {result1['steps_executed']}")
    print(f"Final Output: {result1['final_output']}")
    # Example 2: Content Creation
    print("\n\n" + "="*80)
    print("EXAMPLE 2: Content Creation Chain")
    print("="*80)
    result2 = agent.create_content(
        topic="Artificial Intelligence in Healthcare",
        audience="medical professionals"
    )
    print(f"\n{'='*70}")
    print("EXECUTION TRACE")
    print(f"{'='*70}")
    for step_info in result2['execution_trace']:
        print(f"\n{step_info['step']}:")
        print(f"  Output: {step_info['output']}")
        print(f"  Duration: {step_info['duration_ms']:.2f}ms")
    # Example 3: Problem Solving
    print("\n\n" + "="*80)
    print("EXAMPLE 3: Problem Solving Chain")
    print("="*80)
    result3 = agent.solve_complex_problem(
        "How can we reduce customer churn in our SaaS product?"
    )
    print(f"\n{'='*70}")
    print("FINAL RECOMMENDATION")
    print(f"{'='*70}")
    print(result3['final_output'])
