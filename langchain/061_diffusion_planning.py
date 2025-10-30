"""
Pattern 061: Diffusion-Based Planning

Description:
    Diffusion-Based Planning uses diffusion models (inspired by diffusion processes
    in physics) for trajectory planning and action sequence generation. This approach
    can handle multimodal distributions and generate diverse, high-quality plans by
    iteratively denoising random noise into coherent action sequences.

Components:
    1. Diffusion Process: Forward (add noise) and reverse (denoise)
    2. Action Sequence Generator: Creates trajectory proposals
    3. Denoising Steps: Iterative refinement of plans
    4. Scoring Function: Evaluates plan quality
    5. Sampling Strategy: Controls exploration-exploitation

Use Cases:
    - Robotics motion planning
    - Game AI pathfinding
    - Multi-step action planning
    - Creative sequence generation
    - Trajectory optimization
    - Scenario planning

LangChain Implementation:
    Implements diffusion-inspired planning using LLM-based iterative refinement,
    noise injection, and progressive denoising for plan generation.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random
import math

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class NoiseLevel(Enum):
    """Noise levels for diffusion process"""
    HIGH = "high"  # Initial high noise
    MEDIUM = "medium"  # Mid-process
    LOW = "low"  # Near convergence
    MINIMAL = "minimal"  # Final refinement


class SamplingStrategy(Enum):
    """Strategies for plan sampling"""
    DETERMINISTIC = "deterministic"  # Single best plan
    STOCHASTIC = "stochastic"  # Sample from distribution
    BEAM_SEARCH = "beam_search"  # Keep top-k candidates
    DIVERSE = "diverse"  # Maximize diversity


@dataclass
class ActionStep:
    """Single action in a sequence"""
    step_id: int
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    
    def add_noise(self, noise_level: float) -> 'ActionStep':
        """Add noise to action for diffusion"""
        # In real implementation, this would modify action parameters
        # Here we adjust confidence as proxy for noise
        noisy_confidence = max(0.1, self.confidence * (1 - noise_level * 0.5))
        return ActionStep(
            step_id=self.step_id,
            action=self.action,
            parameters=self.parameters.copy(),
            confidence=noisy_confidence
        )


@dataclass
class Plan:
    """Complete action plan"""
    plan_id: str
    goal: str
    steps: List[ActionStep]
    noise_level: float = 0.0
    quality_score: float = 0.0
    diversity_score: float = 0.0
    
    @property
    def num_steps(self) -> int:
        return len(self.steps)
    
    def to_text(self) -> str:
        """Convert plan to text representation"""
        text = f"Goal: {self.goal}\n\nSteps:\n"
        for step in self.steps:
            text += f"{step.step_id}. {step.action}\n"
        return text


@dataclass
class DiffusionResult:
    """Result from diffusion planning"""
    goal: str
    final_plans: List[Plan]
    diffusion_steps: int
    best_plan: Plan
    diversity_score: float
    total_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal[:80] + "...",
            "num_plans": len(self.final_plans),
            "best_plan_steps": self.best_plan.num_steps,
            "best_quality": f"{self.best_plan.quality_score:.2f}",
            "diversity": f"{self.diversity_score:.2f}",
            "diffusion_steps": self.diffusion_steps,
            "total_time_ms": f"{self.total_time_ms:.1f}"
        }


class DiffusionPlanner:
    """
    Diffusion-based planning agent.
    
    Features:
    1. Iterative plan denoising
    2. Multi-modal plan generation
    3. Quality-guided refinement
    4. Diverse plan sampling
    5. Progressive improvement
    """
    
    def __init__(
        self,
        num_diffusion_steps: int = 5,
        num_candidates: int = 3,
        sampling_strategy: SamplingStrategy = SamplingStrategy.DIVERSE,
        temperature: float = 0.7
    ):
        self.num_diffusion_steps = num_diffusion_steps
        self.num_candidates = num_candidates
        self.sampling_strategy = sampling_strategy
        self.temperature = temperature
        
        # Planner LLM (higher temperature for creativity)
        self.planner = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=temperature
        )
        
        # Refiner LLM (lower temperature for quality)
        self.refiner = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # Evaluator LLM
        self.evaluator = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2
        )
    
    def _generate_noisy_plan(
        self,
        goal: str,
        noise_level: float,
        context: Optional[str] = None
    ) -> Plan:
        """Generate initial noisy plan"""
        
        # Higher noise = more randomness = more creative/diverse
        creativity_instruction = ""
        if noise_level > 0.7:
            creativity_instruction = "Be highly creative and explore unusual approaches."
        elif noise_level > 0.4:
            creativity_instruction = "Consider multiple different approaches."
        else:
            creativity_instruction = "Focus on practical, direct solutions."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are generating an action plan to achieve a goal.

{creativity_instruction}

{context if context else ""}

Generate a step-by-step plan. Format:
1. [action step]
2. [action step]
..."""),
            ("user", "Goal: {goal}\n\nPlan:")
        ])
        
        chain = prompt | self.planner | StrOutputParser()
        plan_text = chain.invoke({"goal": goal})
        
        # Parse steps
        steps = []
        for i, line in enumerate(plan_text.split('\n'), 1):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                action = line.lstrip('0123456789.-) ').strip()
                if action:
                    steps.append(ActionStep(
                        step_id=i,
                        action=action,
                        confidence=1.0 - noise_level * 0.3
                    ))
        
        return Plan(
            plan_id=f"plan_{random.randint(1000, 9999)}",
            goal=goal,
            steps=steps,
            noise_level=noise_level
        )
    
    def _denoise_plan(
        self,
        noisy_plan: Plan,
        target_noise_level: float,
        iteration: int
    ) -> Plan:
        """Denoise/refine a plan"""
        
        # Calculate denoising strength
        denoise_strength = noisy_plan.noise_level - target_noise_level
        
        if denoise_strength <= 0:
            return noisy_plan
        
        refinement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are refining an action plan to make it more coherent and effective.

Improve the plan by:
1. Making steps more specific and actionable
2. Ensuring logical flow
3. Filling in missing details
4. Improving clarity

Keep the general structure but refine the details."""),
            ("user", """Goal: {goal}

Current Plan:
{plan}

Iteration {iteration}: Refine this plan:""")
        ])
        
        chain = refinement_prompt | self.refiner | StrOutputParser()
        refined_text = chain.invoke({
            "goal": noisy_plan.goal,
            "plan": noisy_plan.to_text(),
            "iteration": iteration
        })
        
        # Parse refined steps
        steps = []
        for i, line in enumerate(refined_text.split('\n'), 1):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                action = line.lstrip('0123456789.-) ').strip()
                if action:
                    steps.append(ActionStep(
                        step_id=i,
                        action=action,
                        confidence=1.0 - target_noise_level * 0.2
                    ))
        
        # If parsing failed, keep original
        if not steps:
            steps = noisy_plan.steps
        
        return Plan(
            plan_id=noisy_plan.plan_id,
            goal=noisy_plan.goal,
            steps=steps,
            noise_level=target_noise_level
        )
    
    def _evaluate_plan(self, plan: Plan) -> float:
        """Evaluate plan quality"""
        
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Evaluate this action plan on a scale of 0.0 to 1.0.

Consider:
- Completeness: Does it achieve the goal?
- Feasibility: Is it realistic?
- Efficiency: Minimal unnecessary steps?
- Clarity: Clear and actionable?

Respond with just a number between 0.0 and 1.0"""),
            ("user", "{plan}")
        ])
        
        chain = evaluation_prompt | self.evaluator | StrOutputParser()
        
        try:
            score_str = chain.invoke({"plan": plan.to_text()})
            score = float(''.join(c for c in score_str if c.isdigit() or c == '.'))
            return max(0.0, min(1.0, score))
        except:
            # Fallback: score based on step count and confidence
            avg_confidence = sum(s.confidence for s in plan.steps) / len(plan.steps) if plan.steps else 0
            step_score = min(1.0, len(plan.steps) / 5.0)  # Prefer 5-ish steps
            return (avg_confidence + step_score) / 2
    
    def _calculate_diversity(self, plans: List[Plan]) -> float:
        """Calculate diversity among plans"""
        
        if len(plans) <= 1:
            return 0.0
        
        # Simple diversity: count unique actions
        all_actions = set()
        for plan in plans:
            for step in plan.steps:
                all_actions.add(step.action.lower()[:50])  # First 50 chars
        
        # Diversity = unique actions / total actions
        total_actions = sum(len(plan.steps) for plan in plans)
        return len(all_actions) / total_actions if total_actions > 0 else 0.0
    
    def plan(
        self,
        goal: str,
        constraints: Optional[List[str]] = None
    ) -> DiffusionResult:
        """Generate plans using diffusion process"""
        
        start_time = time.time()
        
        print(f"\n🌊 Starting diffusion-based planning...")
        print(f"   Goal: {goal}")
        print(f"   Diffusion steps: {self.num_diffusion_steps}")
        print(f"   Candidates: {self.num_candidates}")
        
        # Build context from constraints
        context = ""
        if constraints:
            context = "Constraints:\n" + "\n".join(f"- {c}" for c in constraints)
        
        # Generate initial noisy plans
        print(f"\n📍 Step 0: Generating {self.num_candidates} initial noisy plans...")
        
        current_plans = []
        for i in range(self.num_candidates):
            # Start with high noise for diversity
            initial_noise = 0.8 + random.random() * 0.2  # 0.8-1.0
            plan = self._generate_noisy_plan(goal, initial_noise, context)
            current_plans.append(plan)
            print(f"   Plan {i+1}: {len(plan.steps)} steps (noise: {initial_noise:.2f})")
        
        # Diffusion process: progressively denoise
        for step in range(1, self.num_diffusion_steps + 1):
            # Calculate target noise level (decreasing)
            target_noise = max(0.0, 1.0 - (step / self.num_diffusion_steps))
            
            print(f"\n📍 Step {step}/{self.num_diffusion_steps}: Denoising to {target_noise:.2f}")
            
            # Denoise all plans
            denoised_plans = []
            for plan in current_plans:
                denoised = self._denoise_plan(plan, target_noise, step)
                denoised_plans.append(denoised)
            
            # Evaluate plans
            for plan in denoised_plans:
                plan.quality_score = self._evaluate_plan(plan)
            
            # Select best plans based on strategy
            if self.sampling_strategy == SamplingStrategy.BEAM_SEARCH:
                # Keep top-k by quality
                denoised_plans.sort(key=lambda p: p.quality_score, reverse=True)
                current_plans = denoised_plans[:self.num_candidates]
                
            elif self.sampling_strategy == SamplingStrategy.DIVERSE:
                # Balance quality and diversity
                # Keep best plan + diverse alternatives
                denoised_plans.sort(key=lambda p: p.quality_score, reverse=True)
                current_plans = [denoised_plans[0]]  # Best plan
                
                # Add diverse plans
                for plan in denoised_plans[1:]:
                    if len(current_plans) >= self.num_candidates:
                        break
                    # Check if sufficiently different
                    is_diverse = True
                    for existing in current_plans:
                        # Simple diversity check
                        common_actions = sum(
                            1 for s1 in plan.steps
                            for s2 in existing.steps
                            if s1.action[:30] == s2.action[:30]
                        )
                        if common_actions > len(plan.steps) * 0.7:
                            is_diverse = False
                            break
                    
                    if is_diverse:
                        current_plans.append(plan)
            
            else:  # DETERMINISTIC or STOCHASTIC
                denoised_plans.sort(key=lambda p: p.quality_score, reverse=True)
                current_plans = denoised_plans[:self.num_candidates]
            
            # Show progress
            for i, plan in enumerate(current_plans, 1):
                print(f"   Plan {i}: Quality {plan.quality_score:.2f}, {len(plan.steps)} steps")
        
        # Final evaluation
        print(f"\n✅ Diffusion complete. Evaluating final plans...")
        
        for plan in current_plans:
            plan.quality_score = self._evaluate_plan(plan)
        
        # Calculate diversity
        diversity_score = self._calculate_diversity(current_plans)
        
        # Select best plan
        best_plan = max(current_plans, key=lambda p: p.quality_score)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        return DiffusionResult(
            goal=goal,
            final_plans=current_plans,
            diffusion_steps=self.num_diffusion_steps,
            best_plan=best_plan,
            diversity_score=diversity_score,
            total_time_ms=total_time_ms
        )


def demonstrate_diffusion_planning():
    """Demonstrate Diffusion-Based Planning pattern"""
    
    print("=" * 80)
    print("PATTERN 061: DIFFUSION-BASED PLANNING DEMONSTRATION")
    print("=" * 80)
    print("\nIterative plan refinement through diffusion process\n")
    
    # Test 1: Simple goal with diverse sampling
    print("\n" + "=" * 80)
    print("TEST 1: Diverse Plan Generation")
    print("=" * 80)
    
    planner1 = DiffusionPlanner(
        num_diffusion_steps=4,
        num_candidates=3,
        sampling_strategy=SamplingStrategy.DIVERSE,
        temperature=0.8
    )
    
    goal1 = "Plan a successful product launch"
    
    result1 = planner1.plan(goal1)
    
    print(f"\n📊 Results:")
    print(f"   Plans Generated: {len(result1.final_plans)}")
    print(f"   Diversity Score: {result1.diversity_score:.2f}")
    print(f"   Total Time: {result1.total_time_ms:.1f}ms")
    
    print(f"\n🏆 Best Plan (Quality: {result1.best_plan.quality_score:.2f}):")
    for step in result1.best_plan.steps:
        print(f"   {step.step_id}. {step.action}")
    
    print(f"\n🔄 Alternative Plans:")
    for i, plan in enumerate(result1.final_plans[1:], 2):
        print(f"\n   Plan {i} (Quality: {plan.quality_score:.2f}):")
        for step in plan.steps[:3]:
            print(f"      {step.step_id}. {step.action}")
        if len(plan.steps) > 3:
            print(f"      ... ({len(plan.steps) - 3} more steps)")
    
    # Test 2: Constrained planning
    print("\n" + "=" * 80)
    print("TEST 2: Constrained Diffusion Planning")
    print("=" * 80)
    
    planner2 = DiffusionPlanner(
        num_diffusion_steps=5,
        num_candidates=2,
        sampling_strategy=SamplingStrategy.BEAM_SEARCH
    )
    
    goal2 = "Organize a team-building event"
    constraints2 = [
        "Budget limited to $1000",
        "Must be completed within 2 weeks",
        "Remote team across different time zones"
    ]
    
    print(f"\n🎯 Goal: {goal2}")
    print(f"📋 Constraints:")
    for constraint in constraints2:
        print(f"   - {constraint}")
    
    result2 = planner2.plan(goal2, constraints=constraints2)
    
    print(f"\n🏆 Optimized Plan (Quality: {result2.best_plan.quality_score:.2f}):")
    for step in result2.best_plan.steps:
        print(f"   {step.step_id}. {step.action}")
    
    # Test 3: Comparing diffusion steps
    print("\n" + "=" * 80)
    print("TEST 3: Impact of Diffusion Steps")
    print("=" * 80)
    
    goal3 = "Improve customer satisfaction"
    
    for num_steps in [2, 4, 6]:
        print(f"\n   Testing with {num_steps} diffusion steps...")
        
        planner = DiffusionPlanner(
            num_diffusion_steps=num_steps,
            num_candidates=2,
            sampling_strategy=SamplingStrategy.DIVERSE
        )
        
        result = planner.plan(goal3)
        
        print(f"      Best Quality: {result.best_plan.quality_score:.2f}")
        print(f"      Diversity: {result.diversity_score:.2f}")
        print(f"      Time: {result.total_time_ms:.1f}ms")
        print(f"      Steps in plan: {result.best_plan.num_steps}")
    
    # Summary
    print("\n" + "=" * 80)
    print("DIFFUSION-BASED PLANNING PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Multi-Modal Plans: Generates diverse alternative plans
2. Progressive Refinement: Iterative quality improvement
3. Exploration-Exploitation: Balances creativity and quality
4. Robustness: Multiple candidate solutions
5. Flexible Control: Adjustable diffusion process

Diffusion Process:
1. Forward Process: Add noise (generate diverse initial plans)
2. Reverse Process: Denoise (iteratively refine plans)
3. Multiple Steps: Progressive quality improvement
4. Sampling: Control exploration vs exploitation

Components:
- Noise Injection: Creates initial diversity
- Denoising: Iterative refinement
- Quality Scoring: Evaluates plan effectiveness
- Diversity Measurement: Ensures variety
- Selection Strategy: Chooses best candidates

Sampling Strategies:
1. Deterministic: Single best path
   - Highest quality guarantee
   - No diversity

2. Stochastic: Sample from distribution
   - Controlled randomness
   - Balanced exploration

3. Beam Search: Keep top-k candidates
   - Quality-focused
   - Limited diversity

4. Diverse: Maximize variety
   - Multiple alternatives
   - Creative solutions

Noise Schedules:
- High Noise (1.0 → 0.7): Initial creativity
- Medium Noise (0.7 → 0.4): Refinement begins
- Low Noise (0.4 → 0.1): Convergence
- Minimal Noise (0.1 → 0.0): Final polish

Use Cases:
- Robotics: Motion planning, trajectory optimization
- Game AI: Pathfinding, strategy generation
- Planning: Project planning, resource allocation
- Creative Tasks: Story generation, design
- Optimization: Multi-objective planning
- Scenario Analysis: Alternative futures

Best Practices:
1. More steps = higher quality (but slower)
2. More candidates = more diversity
3. Appropriate noise schedule
4. Quality-diversity balance
5. Constraint incorporation
6. Iterative evaluation
7. Early stopping when converged

Production Considerations:
- Computational cost (multiple LLM calls)
- Caching intermediate results
- Parallel plan generation
- Quality thresholds
- Timeout handling
- Plan validation
- User preference learning

Comparison with Related Patterns:
- vs. Tree-of-Thoughts: Continuous vs discrete
- vs. Beam Search: Diffusion vs greedy
- vs. Monte Carlo: Systematic vs random
- vs. Evolutionary: Denoising vs mutation

Advanced Techniques:
1. Conditional Diffusion: Constraint-aware
2. Guided Diffusion: Quality-guided sampling
3. Classifier-Free: Balance guidance
4. Latent Diffusion: Work in abstract space
5. Cascaded: Multi-resolution planning

Research Inspiration:
- Denoising Diffusion Probabilistic Models (DDPM)
- Score-based Generative Models
- Diffusion for Planning (Diffuser)
- Planning with Diffusion for Flexible Behavior

The Diffusion-Based Planning pattern provides a powerful framework
for generating diverse, high-quality plans through iterative
refinement inspired by diffusion processes in physics.
""")


if __name__ == "__main__":
    demonstrate_diffusion_planning()
