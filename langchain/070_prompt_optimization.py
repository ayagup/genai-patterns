"""
Pattern 070: Prompt Optimization/Engineering

Description:
    Prompt Optimization/Engineering automates the process of finding optimal prompts for
    specific tasks. Rather than manually crafting and testing prompts, this pattern uses
    automated search, evaluation, and refinement to discover high-performing prompt
    formulations. It explores the prompt space systematically, tests variations, measures
    effectiveness, and iteratively improves prompts to maximize task performance.
    
    This pattern is essential for building production LLM systems where prompt quality
    directly impacts output quality, cost, and latency. It can discover non-obvious
    prompt structures, optimal instruction phrasing, effective examples for few-shot
    learning, and task-specific formatting that humans might not consider.

Components:
    1. Prompt Generator: Creates candidate prompt variations
    2. Evaluation Engine: Measures prompt effectiveness on test cases
    3. Optimization Strategy: Guides search through prompt space
    4. Prompt Analyzer: Extracts features from successful prompts
    5. Meta-Prompt Designer: Creates templates for prompt generation
    6. Performance Tracker: Monitors metrics across iterations
    7. Prompt Library: Stores and manages successful prompts

Architecture:
    ```
    Initial Prompt
        ↓
    ┌─────────────────────────────┐
    │   Optimization Loop         │
    │                             │
    │  Generate Variations        │
    │         ↓                   │
    │  Evaluate on Test Cases     │
    │         ↓                   │
    │  Rank by Performance        │
    │         ↓                   │
    │  Select Best Candidates     │
    │         ↓                   │
    │  Analyze Success Patterns   │
    │         ↓                   │
    │  Generate New Variations    │
    └─────────────────────────────┘
         ↓ (iterate)
    Optimized Prompt
    ```

Use Cases:
    - Automatic prompt engineering for production systems
    - Few-shot example selection and ordering
    - Task-specific instruction optimization
    - Output format specification tuning
    - Zero-shot prompt improvement
    - Multi-task prompt optimization

Advantages:
    - Finds better prompts than manual engineering
    - Saves significant development time
    - Discovers non-obvious prompt patterns
    - Adapts to model updates automatically
    - Provides data-driven prompt decisions
    - Scales across multiple tasks

LangChain Implementation:
    Uses ChatOpenAI for prompt generation and testing. Demonstrates
    prompt variation generation, evaluation on test cases, iterative
    optimization, and performance tracking.
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import random
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class OptimizationStrategy(Enum):
    """Optimization strategies for prompt search."""
    RANDOM_SEARCH = "random_search"
    HILL_CLIMBING = "hill_climbing"
    BEAM_SEARCH = "beam_search"
    GENETIC = "genetic"
    GRADIENT_FREE = "gradient_free"


class PromptComponent(Enum):
    """Components of a prompt that can be optimized."""
    INSTRUCTION = "instruction"
    CONTEXT = "context"
    EXAMPLES = "examples"
    OUTPUT_FORMAT = "output_format"
    CONSTRAINTS = "constraints"
    ROLE = "role"


@dataclass
class TestCase:
    """Test case for prompt evaluation."""
    case_id: str
    input: str
    expected_output: str
    weight: float = 1.0  # Importance weight


@dataclass
class PromptCandidate:
    """A candidate prompt with metadata."""
    prompt_id: str
    prompt_text: str
    components: Dict[PromptComponent, str]
    score: float = 0.0
    evaluations: int = 0
    generation: int = 0
    parent_id: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result of evaluating a prompt on test cases."""
    prompt_id: str
    test_case_scores: Dict[str, float]
    average_score: float
    latency: float
    tokens_used: int


@dataclass
class OptimizationRun:
    """Complete optimization run results."""
    task_description: str
    initial_prompt: PromptCandidate
    best_prompt: PromptCandidate
    all_candidates: List[PromptCandidate]
    optimization_history: List[EvaluationResult]
    total_iterations: int
    improvement: float


class PromptOptimizer:
    """
    Automated prompt optimization system.
    """
    
    def __init__(
        self,
        strategy: OptimizationStrategy = OptimizationStrategy.BEAM_SEARCH,
        beam_width: int = 3,
        max_iterations: int = 10
    ):
        """
        Initialize prompt optimizer.
        
        Args:
            strategy: Optimization strategy to use
            beam_width: Number of candidates to keep (for beam search)
            max_iterations: Maximum optimization iterations
        """
        self.strategy = strategy
        self.beam_width = beam_width
        self.max_iterations = max_iterations
        
        # LLMs for different tasks
        self.prompt_generator = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo")
        self.evaluator_llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
        self.analyzer = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")
        
        self.parser = StrOutputParser()
        
        # Tracking
        self.prompt_library: Dict[str, PromptCandidate] = {}
        self.evaluation_cache: Dict[Tuple[str, str], float] = {}
        self.generation_count = 0
    
    def generate_prompt_variations(
        self,
        base_prompt: PromptCandidate,
        num_variations: int = 5,
        task_description: str = ""
    ) -> List[PromptCandidate]:
        """
        Generate variations of a prompt.
        
        Args:
            base_prompt: Starting prompt
            num_variations: Number of variations to generate
            task_description: Description of the task
            
        Returns:
            List of prompt variations
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a prompt engineer. Generate improved variations of prompts.

Task: {task}

Consider these optimization strategies:
1. Clarify instructions and make them more specific
2. Add helpful context or constraints
3. Improve output format specifications
4. Refine role/persona if applicable
5. Reorder or restructure components
6. Add examples or improve existing ones

Generate {num_variations} distinct variations that might perform better.
Each variation should have a clear improvement hypothesis.

Format each as:
VARIATION {n}:
{complete prompt text}
---"""),
            ("user", """Base Prompt:
{base_prompt}

Generate {num_variations} improved variations:""")
        ])
        
        chain = prompt | self.prompt_generator | self.parser
        
        try:
            result = chain.invoke({
                "task": task_description,
                "base_prompt": base_prompt.prompt_text,
                "num_variations": num_variations
            })
            
            # Parse variations
            variations = []
            variation_blocks = result.split('---')
            
            for block in variation_blocks:
                if 'VARIATION' not in block:
                    continue
                
                # Extract prompt text (remove variation header)
                lines = block.split('\n')
                prompt_lines = [line for line in lines if not line.strip().startswith('VARIATION')]
                prompt_text = '\n'.join(prompt_lines).strip()
                
                if prompt_text:
                    self.generation_count += 1
                    variation = PromptCandidate(
                        prompt_id=f"prompt_{self.generation_count}",
                        prompt_text=prompt_text,
                        components=self._extract_components(prompt_text),
                        generation=base_prompt.generation + 1,
                        parent_id=base_prompt.prompt_id
                    )
                    variations.append(variation)
            
            return variations[:num_variations]
            
        except Exception as e:
            print(f"Error generating variations: {e}")
            # Fallback: create simple variations
            variations = []
            for i in range(min(3, num_variations)):
                self.generation_count += 1
                variation = PromptCandidate(
                    prompt_id=f"prompt_{self.generation_count}",
                    prompt_text=f"{base_prompt.prompt_text}\n\nVariation {i+1}: Be more specific and detailed.",
                    components=base_prompt.components.copy(),
                    generation=base_prompt.generation + 1,
                    parent_id=base_prompt.prompt_id
                )
                variations.append(variation)
            return variations
    
    def _extract_components(self, prompt_text: str) -> Dict[PromptComponent, str]:
        """Extract components from prompt text."""
        # Simple heuristic-based extraction
        components = {}
        
        lower_text = prompt_text.lower()
        
        # Check for role
        if 'you are' in lower_text or 'act as' in lower_text:
            components[PromptComponent.ROLE] = "role specified"
        
        # Check for instructions
        if any(word in lower_text for word in ['please', 'task:', 'do', 'generate', 'write', 'analyze']):
            components[PromptComponent.INSTRUCTION] = "instructions provided"
        
        # Check for examples
        if 'example' in lower_text or 'for instance' in lower_text:
            components[PromptComponent.EXAMPLES] = "examples included"
        
        # Check for output format
        if 'format' in lower_text or 'output:' in lower_text:
            components[PromptComponent.OUTPUT_FORMAT] = "format specified"
        
        # Check for constraints
        if any(word in lower_text for word in ['must', 'should', 'constraint', 'limit', 'only']):
            components[PromptComponent.CONSTRAINTS] = "constraints specified"
        
        return components
    
    def evaluate_prompt(
        self,
        prompt_candidate: PromptCandidate,
        test_cases: List[TestCase],
        scorer: Optional[Callable[[str, str], float]] = None
    ) -> EvaluationResult:
        """
        Evaluate a prompt on test cases.
        
        Args:
            prompt_candidate: Prompt to evaluate
            test_cases: Test cases to use
            scorer: Custom scoring function (optional)
            
        Returns:
            Evaluation result with scores
        """
        start_time = time.time()
        test_case_scores = {}
        total_tokens = 0
        
        # Default scorer: simple similarity
        if scorer is None:
            def default_scorer(output: str, expected: str) -> float:
                output_lower = output.lower()
                expected_lower = expected.lower()
                # Simple keyword overlap
                output_words = set(output_lower.split())
                expected_words = set(expected_lower.split())
                if not expected_words:
                    return 0.5
                overlap = len(output_words & expected_words) / len(expected_words)
                return min(1.0, overlap)
            scorer = default_scorer
        
        for test_case in test_cases:
            # Check cache
            cache_key = (prompt_candidate.prompt_id, test_case.case_id)
            if cache_key in self.evaluation_cache:
                score = self.evaluation_cache[cache_key]
            else:
                # Run prompt on test case
                try:
                    full_prompt = ChatPromptTemplate.from_messages([
                        ("system", prompt_candidate.prompt_text),
                        ("user", test_case.input)
                    ])
                    
                    chain = full_prompt | self.evaluator_llm | self.parser
                    output = chain.invoke({})
                    
                    # Score output
                    score = scorer(output, test_case.expected_output)
                    total_tokens += len(output.split())  # Rough token estimate
                    
                    # Cache result
                    self.evaluation_cache[cache_key] = score
                    
                except Exception as e:
                    print(f"Error evaluating test case {test_case.case_id}: {e}")
                    score = 0.0
            
            test_case_scores[test_case.case_id] = score * test_case.weight
        
        # Calculate weighted average
        total_weight = sum(tc.weight for tc in test_cases)
        average_score = sum(test_case_scores.values()) / total_weight if total_weight > 0 else 0.0
        
        latency = time.time() - start_time
        
        # Update candidate
        prompt_candidate.score = average_score
        prompt_candidate.evaluations += 1
        
        return EvaluationResult(
            prompt_id=prompt_candidate.prompt_id,
            test_case_scores=test_case_scores,
            average_score=average_score,
            latency=latency,
            tokens_used=total_tokens
        )
    
    def optimize(
        self,
        task_description: str,
        initial_prompt: str,
        test_cases: List[TestCase],
        scorer: Optional[Callable[[str, str], float]] = None
    ) -> OptimizationRun:
        """
        Run prompt optimization.
        
        Args:
            task_description: Description of task to optimize for
            initial_prompt: Starting prompt
            test_cases: Test cases for evaluation
            scorer: Custom scoring function
            
        Returns:
            Optimization run results
        """
        print(f"\n{'='*60}")
        print(f"Prompt Optimization: {task_description}")
        print(f"Strategy: {self.strategy.value}")
        print(f"{'='*60}\n")
        
        # Create initial candidate
        self.generation_count += 1
        initial_candidate = PromptCandidate(
            prompt_id=f"prompt_{self.generation_count}",
            prompt_text=initial_prompt,
            components=self._extract_components(initial_prompt),
            generation=0
        )
        
        # Evaluate initial prompt
        print("Evaluating initial prompt...")
        initial_eval = self.evaluate_prompt(initial_candidate, test_cases, scorer)
        print(f"  Initial Score: {initial_eval.average_score:.3f}")
        
        # Track all candidates and history
        all_candidates = [initial_candidate]
        optimization_history = [initial_eval]
        
        # Active candidates (beam)
        active_candidates = [initial_candidate]
        best_candidate = initial_candidate
        
        # Optimization loop
        for iteration in range(self.max_iterations):
            print(f"\nIteration {iteration + 1}/{self.max_iterations}")
            
            # Generate variations from active candidates
            new_candidates = []
            for candidate in active_candidates:
                variations = self.generate_prompt_variations(
                    candidate,
                    num_variations=self.beam_width,
                    task_description=task_description
                )
                new_candidates.extend(variations)
            
            print(f"  Generated {len(new_candidates)} new candidates")
            
            # Evaluate new candidates
            print(f"  Evaluating candidates...")
            for candidate in new_candidates:
                eval_result = self.evaluate_prompt(candidate, test_cases, scorer)
                optimization_history.append(eval_result)
                all_candidates.append(candidate)
                
                # Track best
                if candidate.score > best_candidate.score:
                    best_candidate = candidate
                    print(f"    ✓ New best score: {candidate.score:.3f} (improvement: {candidate.score - initial_candidate.score:+.3f})")
            
            # Select top candidates for next iteration (beam search)
            if self.strategy == OptimizationStrategy.BEAM_SEARCH:
                all_evaluated = active_candidates + new_candidates
                all_evaluated.sort(key=lambda c: c.score, reverse=True)
                active_candidates = all_evaluated[:self.beam_width]
                print(f"  Keeping top {len(active_candidates)} candidates")
            
            # Early stopping if no improvement
            if iteration > 0:
                recent_scores = [h.average_score for h in optimization_history[-self.beam_width:]]
                if len(set(recent_scores)) == 1:
                    print(f"  Early stopping: No score variation in recent candidates")
                    break
        
        # Calculate improvement
        improvement = ((best_candidate.score - initial_candidate.score) / 
                      initial_candidate.score * 100 if initial_candidate.score > 0 else 0.0)
        
        print(f"\n{'='*60}")
        print(f"Optimization Complete")
        print(f"{'='*60}")
        print(f"Initial Score: {initial_candidate.score:.3f}")
        print(f"Best Score: {best_candidate.score:.3f}")
        print(f"Improvement: {improvement:+.1f}%")
        print(f"Total Candidates Tested: {len(all_candidates)}")
        print(f"Best Candidate Generation: {best_candidate.generation}")
        
        return OptimizationRun(
            task_description=task_description,
            initial_prompt=initial_candidate,
            best_prompt=best_candidate,
            all_candidates=all_candidates,
            optimization_history=optimization_history,
            total_iterations=iteration + 1,
            improvement=improvement
        )
    
    def analyze_success_patterns(self, optimization_run: OptimizationRun) -> Dict[str, Any]:
        """
        Analyze patterns in successful prompts.
        
        Args:
            optimization_run: Completed optimization run
            
        Returns:
            Analysis of success patterns
        """
        # Get top performers
        top_prompts = sorted(
            optimization_run.all_candidates,
            key=lambda c: c.score,
            reverse=True
        )[:5]
        
        # Analyze components
        component_scores = defaultdict(list)
        for candidate in optimization_run.all_candidates:
            for component in candidate.components:
                component_scores[component].append(candidate.score)
        
        component_analysis = {}
        for component, scores in component_scores.items():
            component_analysis[component.value] = {
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "count": len(scores)
            }
        
        return {
            "top_prompts": [
                {"id": p.prompt_id, "score": p.score, "generation": p.generation}
                for p in top_prompts
            ],
            "component_analysis": component_analysis,
            "best_generation": optimization_run.best_prompt.generation,
            "total_explored": len(optimization_run.all_candidates)
        }


def demonstrate_prompt_optimization():
    """Demonstrate Prompt Optimization pattern."""
    
    print("="*80)
    print("PROMPT OPTIMIZATION/ENGINEERING - DEMONSTRATION")
    print("="*80)
    
    optimizer = PromptOptimizer(
        strategy=OptimizationStrategy.BEAM_SEARCH,
        beam_width=3,
        max_iterations=5
    )
    
    # Test 1: Sentiment classification optimization
    print("\n" + "="*80)
    print("TEST 1: Sentiment Classification Prompt Optimization")
    print("="*80)
    
    initial_prompt1 = "Classify the sentiment of the text."
    
    test_cases1 = [
        TestCase("tc1", "I love this product!", "positive", 1.0),
        TestCase("tc2", "This is terrible and disappointing.", "negative", 1.0),
        TestCase("tc3", "It's okay, nothing special.", "neutral", 1.0),
        TestCase("tc4", "Absolutely amazing experience!", "positive", 1.0),
        TestCase("tc5", "Worst purchase ever.", "negative", 1.0),
    ]
    
    run1 = optimizer.optimize(
        task_description="Sentiment Classification",
        initial_prompt=initial_prompt1,
        test_cases=test_cases1
    )
    
    print("\n--- BEST PROMPT FOUND ---")
    print(run1.best_prompt.prompt_text)
    print(f"\nScore: {run1.best_prompt.score:.3f}")
    print(f"Generation: {run1.best_prompt.generation}")
    
    # Test 2: Text summarization optimization
    print("\n" + "="*80)
    print("TEST 2: Text Summarization Prompt Optimization")
    print("="*80)
    
    initial_prompt2 = "Summarize this text in one sentence."
    
    test_cases2 = [
        TestCase("tc1", 
                "The meeting discussed quarterly results and future plans.",
                "meeting summary quarterly", 
                1.0),
        TestCase("tc2",
                "Climate change is affecting global weather patterns significantly.",
                "climate change weather impact",
                1.0),
    ]
    
    run2 = optimizer.optimize(
        task_description="Text Summarization",
        initial_prompt=initial_prompt2,
        test_cases=test_cases2
    )
    
    print("\n--- OPTIMIZATION PROGRESS ---")
    print(f"{'Iteration':<12} {'Best Score':<12} {'Candidates':<12}")
    print("-" * 40)
    
    iteration_bests = {}
    for eval_result in run2.optimization_history:
        for candidate in run2.all_candidates:
            if candidate.prompt_id == eval_result.prompt_id:
                gen = candidate.generation
                if gen not in iteration_bests or candidate.score > iteration_bests[gen]:
                    iteration_bests[gen] = candidate.score
    
    for gen in sorted(iteration_bests.keys()):
        print(f"{gen:<12} {iteration_bests[gen]:<12.3f} {'-':<12}")
    
    # Test 3: Component analysis
    print("\n" + "="*80)
    print("TEST 3: Success Pattern Analysis")
    print("="*80)
    
    analysis = optimizer.analyze_success_patterns(run2)
    
    print("\nTop Performing Prompts:")
    for i, prompt_info in enumerate(analysis["top_prompts"], 1):
        print(f"  {i}. {prompt_info['id']}: Score={prompt_info['score']:.3f}, Gen={prompt_info['generation']}")
    
    print("\nComponent Analysis:")
    for component, stats in analysis["component_analysis"].items():
        print(f"  {component}: Avg Score={stats['avg_score']:.3f}, Count={stats['count']}")
    
    print(f"\nBest Generation: {analysis['best_generation']}")
    print(f"Total Candidates Explored: {analysis['total_explored']}")
    
    # Test 4: Compare initial vs optimized
    print("\n" + "="*80)
    print("TEST 4: Initial vs Optimized Comparison")
    print("="*80)
    
    print("\n--- INITIAL PROMPT ---")
    print(run1.initial_prompt.prompt_text)
    print(f"Score: {run1.initial_prompt.score:.3f}")
    
    print("\n--- OPTIMIZED PROMPT ---")
    print(run1.best_prompt.prompt_text)
    print(f"Score: {run1.best_prompt.score:.3f}")
    print(f"Improvement: {run1.improvement:+.1f}%")
    
    # Test 5: Optimization history visualization
    print("\n" + "="*80)
    print("TEST 5: Optimization History")
    print("="*80)
    
    print("\nScore Distribution by Generation:")
    gen_scores = defaultdict(list)
    for candidate in run1.all_candidates:
        gen_scores[candidate.generation].append(candidate.score)
    
    for gen in sorted(gen_scores.keys()):
        scores = gen_scores[gen]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        print(f"  Gen {gen}: Avg={avg_score:.3f}, Max={max_score:.3f}, Count={len(scores)}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Prompt Optimization/Engineering")
    print("="*80)
    print("""
The Prompt Optimization pattern demonstrates automated prompt improvement:

Key Features Demonstrated:
1. Prompt Generation: Creates systematic variations of prompts
2. Evaluation Engine: Measures performance on test cases
3. Beam Search: Maintains top candidates across generations
4. Component Analysis: Identifies successful prompt elements
5. Iterative Refinement: Progressively improves prompts
6. Performance Tracking: Monitors improvement over iterations
7. Pattern Analysis: Extracts insights from successful prompts

Benefits:
• Finds better prompts than manual engineering
• Saves significant development and testing time
• Discovers non-obvious prompt structures
• Data-driven prompt decisions with measurable results
• Scales across multiple tasks and models
• Adapts to model updates automatically
• Provides interpretable optimization process

Use Cases:
• Production LLM system prompt optimization
• Few-shot example selection and ordering
• Task-specific instruction refinement
• Output format specification tuning
• Multi-task prompt engineering
• Domain-specific prompt adaptation
• Cost-performance trade-off optimization
• Latency-optimized prompt design

Comparison with Manual Prompt Engineering:
┌──────────────────────┬────────────────────┬─────────────────────┐
│ Aspect               │ Manual             │ Automated           │
├──────────────────────┼────────────────────┼─────────────────────┤
│ Time Required        │ Hours to days      │ Minutes to hours    │
│ Prompts Tested       │ 5-20               │ 50-500+             │
│ Consistency          │ Variable           │ Systematic          │
│ Optimization Quality │ Good               │ Excellent           │
│ Scalability          │ Limited            │ High                │
│ Reproducibility      │ Low                │ High                │
│ Adaptation Speed     │ Slow               │ Fast                │
└──────────────────────┴────────────────────┴─────────────────────┘

Optimization Strategies:
• Random Search: Explores diverse prompt space randomly
• Hill Climbing: Iteratively improves from best candidate
• Beam Search: Maintains multiple top candidates (demonstrated)
• Genetic Algorithms: Evolves prompts through crossover/mutation
• Gradient-Free: Uses performance signals without gradients
• Bayesian Optimization: Models prompt space probabilistically

LangChain Implementation Notes:
• Multiple LLMs for generation and evaluation
• Beam search for systematic exploration
• Component extraction for pattern analysis
• Evaluation caching to avoid redundant tests
• Generational tracking for improvement monitoring
• Flexible scoring functions for different tasks

Production Considerations:
• Implement comprehensive test case suites (50-500+ cases)
• Add A/B testing for production deployment
• Implement prompt versioning and rollback
• Add cost tracking for optimization runs
• Implement human evaluation for quality checks
• Add multi-objective optimization (accuracy, cost, latency)
• Implement domain-specific prompt templates
• Add safety and bias checks for generated prompts
• Implement continuous optimization with production data
• Add monitoring for prompt performance drift
• Implement transfer learning across similar tasks
• Add prompt explanation generation for interpretability

Advanced Extensions:
• Multi-task prompt optimization with shared components
• Adversarial prompt testing for robustness
• Meta-learning for faster optimization
• Neural prompt generation using learned models
• Active learning for test case selection
• Prompt compression for cost reduction
• Cross-model prompt optimization
• Hierarchical prompt structures
• Prompt ensembling for reliability
• Uncertainty-aware prompt selection

Real-World Applications:
• Customer service chatbot optimization
• Content generation quality improvement
• Code generation prompt refinement
• Medical diagnosis prompt engineering
• Legal document analysis optimization
• Educational content generation
• Multi-lingual translation prompt tuning
• Creative writing assistant optimization
    """)


if __name__ == "__main__":
    demonstrate_prompt_optimization()
