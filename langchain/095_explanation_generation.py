"""
Pattern 095: Explanation Generation

Description:
    Explanation Generation creates human-readable explanations for agent decisions,
    reasoning processes, and actions. This pattern is crucial for transparency,
    debugging, user trust, and regulatory compliance. Good explanations help users
    understand why an agent made specific decisions, what information influenced
    those decisions, and what alternatives were considered.

    Explanation types include:
    - Decision explanations (why this choice?)
    - Reasoning chain visualization (how did we get here?)
    - Confidence rationales (why this certainty?)
    - Alternative options (what else was considered?)
    - Error explanations (what went wrong?)
    - Action justifications (why this action?)

Components:
    1. Explanation Generator
       - Natural language generation
       - Template-based explanations
       - Context-aware formatting
       - Audience adaptation
       - Multiple explanation levels

    2. Reasoning Extractor
       - Captures decision points
       - Tracks information flow
       - Identifies key factors
       - Documents alternatives
       - Maintains causal chains

    3. Visualization Builder
       - Text-based visualizations
       - Tree/graph representations
       - Timeline views
       - Step-by-step breakdowns
       - Interactive explorations

    4. Context Enricher
       - Adds relevant background
       - Includes supporting evidence
       - References sources
       - Provides examples
       - Links to documentation

Use Cases:
    1. User-Facing Applications
       - Explain recommendations
       - Justify decisions
       - Build user trust
       - Enable user feedback
       - Support learning

    2. Debugging & Development
       - Understand agent behavior
       - Identify issues
       - Validate reasoning
       - Test edge cases
       - Documentation generation

    3. Compliance & Auditing
       - Regulatory requirements
       - Audit trails
       - Accountability documentation
       - Risk assessment
       - Bias detection

    4. Education & Training
       - Teaching materials
       - Example generation
       - Best practice documentation
       - User onboarding
       - Knowledge transfer

    5. Research & Analysis
       - Understand model behavior
       - Compare approaches
       - Identify patterns
       - Publication materials
       - Reproducibility

LangChain Implementation:
    LangChain supports explanation through:
    - Chain breakdown and visualization
    - Intermediate step logging
    - Prompt templates with reasoning
    - Custom explanation chains
    - Callback handlers for tracking

Key Features:
    1. Multiple Explanation Levels
       - High-level summary (executives)
       - Medium detail (general users)
       - Technical depth (developers)
       - Expert analysis (researchers)
       - Adaptive to audience

    2. Rich Context
       - Decision factors highlighted
       - Supporting evidence provided
       - Alternatives mentioned
       - Confidence indicated
       - Sources cited

    3. Clear Structure
       - Logical flow
       - Easy to follow
       - Visual aids
       - Key points emphasized
       - Actionable insights

    4. Interactive Exploration
       - Drill-down capability
       - Follow reasoning paths
       - Explore alternatives
       - Ask follow-ups
       - Customize detail level

Best Practices:
    1. Explanation Content
       - Start with main decision/conclusion
       - Explain key factors (2-5 most important)
       - Show reasoning steps
       - Mention alternatives considered
       - Include confidence/uncertainty
       - Cite sources/evidence
       - Use concrete examples

    2. Language & Style
       - Clear, simple language
       - Avoid jargon (or define it)
       - Active voice preferred
       - Concrete over abstract
       - Structured formatting
       - Consistent terminology

    3. Audience Adaptation
       - Technical users: More details, technical terms
       - Business users: Focus on impact, ROI
       - General users: Simple language, analogies
       - Experts: Assumptions, methodology
       - Regulators: Compliance focus

    4. Visual Aids
       - Decision trees for branching logic
       - Flowcharts for processes
       - Tables for comparisons
       - Timelines for sequences
       - Graphs for relationships

Trade-offs:
    Advantages:
    - Increased transparency
    - Better user trust
    - Easier debugging
    - Compliance support
    - Knowledge sharing
    - Educational value

    Disadvantages:
    - Additional computation
    - Complexity in generation
    - May reveal limitations
    - Risk of over-explanation
    - Maintenance burden
    - Translation challenges

Production Considerations:
    1. Performance Impact
       - Cache common explanations
       - Generate asynchronously
       - Provide summary first
       - Lazy load details
       - Optimize templates

    2. Quality Assurance
       - Validate explanations
       - Test with users
       - Check for hallucinations
       - Verify accuracy
       - Regular reviews

    3. Privacy & Security
       - Redact sensitive info
       - Control detail level
       - Access restrictions
       - Audit explanation access
       - Secure storage

    4. Localization
       - Multiple languages
       - Cultural adaptation
       - Units and formats
       - Examples relevance
       - Local regulations

    5. Maintenance
       - Update with model changes
       - Refresh examples
       - Improve based on feedback
       - Version explanations
       - Document changes
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class ExplanationLevel(Enum):
    """Level of detail in explanation"""
    SUMMARY = "summary"  # High-level overview
    STANDARD = "standard"  # Balanced detail
    DETAILED = "detailed"  # Comprehensive explanation
    TECHNICAL = "technical"  # For experts


class AudienceType(Enum):
    """Type of audience for explanation"""
    GENERAL = "general"  # General public
    BUSINESS = "business"  # Business users
    TECHNICAL = "technical"  # Developers
    EXPERT = "expert"  # Domain experts
    REGULATOR = "regulator"  # Compliance/audit


@dataclass
class DecisionFactor:
    """A factor that influenced a decision"""
    name: str
    description: str
    weight: float  # 0-1, importance
    evidence: str
    impact: str  # positive, negative, neutral


@dataclass
class Alternative:
    """An alternative option that was considered"""
    option: str
    description: str
    pros: List[str]
    cons: List[str]
    score: float
    not_chosen_reason: str


@dataclass
class Explanation:
    """Complete explanation of a decision"""
    decision: str
    summary: str
    reasoning_steps: List[str]
    key_factors: List[DecisionFactor]
    alternatives: List[Alternative]
    confidence: float
    sources: List[str]
    timestamp: str
    metadata: Dict[str, Any]


class ExplanationGenerator:
    """
    Generates human-readable explanations for agent decisions.
    
    Supports multiple explanation levels and audience types.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize explanation generator.
        
        Args:
            model: LLM model to use for generation
        """
        self.llm = ChatOpenAI(model=model, temperature=0.3)
    
    def generate_simple_explanation(
        self,
        decision: str,
        context: str,
        reasoning: str
    ) -> str:
        """
        Generate a simple, clear explanation.
        
        Args:
            decision: The decision made
            context: Context/input
            reasoning: Basic reasoning
            
        Returns:
            Simple explanation
        """
        prompt = ChatPromptTemplate.from_template(
            """Generate a clear, simple explanation for this decision.

Decision: {decision}
Context: {context}
Reasoning: {reasoning}

Write a 2-3 sentence explanation that:
1. States what was decided
2. Explains why in simple terms
3. Is easy for anyone to understand

Explanation:"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "decision": decision,
            "context": context,
            "reasoning": reasoning
        })
    
    def generate_detailed_explanation(
        self,
        explanation_data: Explanation,
        level: ExplanationLevel = ExplanationLevel.STANDARD,
        audience: AudienceType = AudienceType.GENERAL
    ) -> str:
        """
        Generate detailed explanation from structured data.
        
        Args:
            explanation_data: Structured explanation data
            level: Level of detail
            audience: Target audience
            
        Returns:
            Formatted explanation
        """
        sections = []
        
        # 1. Summary
        sections.append(f"# Decision Explanation\n")
        sections.append(f"**Decision:** {explanation_data.decision}\n")
        sections.append(f"**Summary:** {explanation_data.summary}\n")
        sections.append(f"**Confidence:** {explanation_data.confidence:.0%}\n")
        
        # 2. Key Factors (always include)
        if explanation_data.key_factors:
            sections.append("\n## Key Factors\n")
            for factor in sorted(explanation_data.key_factors, key=lambda f: f.weight, reverse=True):
                sections.append(f"### {factor.name} (Weight: {factor.weight:.0%})")
                sections.append(f"- {factor.description}")
                sections.append(f"- Evidence: {factor.evidence}")
                sections.append(f"- Impact: {factor.impact}\n")
        
        # 3. Reasoning Steps (standard and above)
        if level in [ExplanationLevel.STANDARD, ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL]:
            if explanation_data.reasoning_steps:
                sections.append("\n## Reasoning Process\n")
                for i, step in enumerate(explanation_data.reasoning_steps, 1):
                    sections.append(f"{i}. {step}")
        
        # 4. Alternatives (detailed and above)
        if level in [ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL]:
            if explanation_data.alternatives:
                sections.append("\n\n## Alternatives Considered\n")
                for alt in explanation_data.alternatives:
                    sections.append(f"\n### Option: {alt.option} (Score: {alt.score:.2f})")
                    sections.append(f"{alt.description}\n")
                    sections.append("**Pros:**")
                    for pro in alt.pros:
                        sections.append(f"- {pro}")
                    sections.append("\n**Cons:**")
                    for con in alt.cons:
                        sections.append(f"- {con}")
                    sections.append(f"\n**Why not chosen:** {alt.not_chosen_reason}\n")
        
        # 5. Sources (technical)
        if level == ExplanationLevel.TECHNICAL:
            if explanation_data.sources:
                sections.append("\n## Sources & References\n")
                for i, source in enumerate(explanation_data.sources, 1):
                    sections.append(f"{i}. {source}")
        
        # 6. Metadata (technical)
        if level == ExplanationLevel.TECHNICAL:
            sections.append(f"\n\n## Technical Details\n")
            sections.append(f"- Timestamp: {explanation_data.timestamp}")
            for key, value in explanation_data.metadata.items():
                sections.append(f"- {key}: {value}")
        
        return "\n".join(sections)
    
    def generate_comparison_explanation(
        self,
        options: List[Dict[str, Any]],
        chosen: str,
        criteria: List[str]
    ) -> str:
        """
        Generate explanation comparing multiple options.
        
        Args:
            options: List of options with scores
            chosen: The chosen option
            criteria: Evaluation criteria
            
        Returns:
            Comparison explanation
        """
        prompt = ChatPromptTemplate.from_template(
            """Generate a clear explanation comparing these options.

Options evaluated:
{options}

Criteria used:
{criteria}

Chosen option: {chosen}

Provide a 3-4 paragraph explanation that:
1. Summarizes what was compared
2. Explains the key differences
3. Justifies why {chosen} was selected
4. Mentions any close alternatives

Explanation:"""
        )
        
        options_text = "\n".join([f"- {opt['name']}: {opt.get('description', '')}" for opt in options])
        criteria_text = "\n".join([f"- {c}" for c in criteria])
        
        chain = prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "options": options_text,
            "criteria": criteria_text,
            "chosen": chosen
        })
    
    def explain_error(
        self,
        error_type: str,
        error_message: str,
        context: str,
        suggestions: List[str]
    ) -> str:
        """
        Generate explanation for an error.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Context where error occurred
            suggestions: Suggested fixes
            
        Returns:
            Error explanation
        """
        sections = []
        
        sections.append(f"# Error Explanation\n")
        sections.append(f"**Error Type:** {error_type}\n")
        sections.append(f"**What Happened:**")
        sections.append(f"{error_message}\n")
        
        sections.append(f"\n**Context:**")
        sections.append(f"{context}\n")
        
        if suggestions:
            sections.append(f"\n**How to Fix:**")
            for i, suggestion in enumerate(suggestions, 1):
                sections.append(f"{i}. {suggestion}")
        
        return "\n".join(sections)
    
    def generate_step_by_step(
        self,
        task: str,
        steps: List[Dict[str, Any]]
    ) -> str:
        """
        Generate step-by-step explanation.
        
        Args:
            task: Overall task
            steps: List of steps with details
            
        Returns:
            Step-by-step explanation
        """
        sections = []
        
        sections.append(f"# How I Completed: {task}\n")
        sections.append("Here's what I did, step by step:\n")
        
        for i, step in enumerate(steps, 1):
            sections.append(f"\n## Step {i}: {step.get('name', f'Step {i}')}")
            sections.append(f"{step.get('description', '')}")
            
            if 'input' in step:
                sections.append(f"\n**Input:** {step['input']}")
            
            if 'action' in step:
                sections.append(f"**Action:** {step['action']}")
            
            if 'output' in step:
                sections.append(f"**Output:** {step['output']}")
            
            if 'reasoning' in step:
                sections.append(f"**Why:** {step['reasoning']}")
            
            sections.append("")  # Blank line
        
        return "\n".join(sections)


class ReasoningVisualizer:
    """
    Visualizes reasoning chains and decision trees.
    
    Creates text-based visualizations for better understanding.
    """
    
    def __init__(self):
        """Initialize visualizer"""
        pass
    
    def visualize_decision_tree(
        self,
        root: Dict[str, Any],
        indent: int = 0
    ) -> str:
        """
        Visualize decision tree.
        
        Args:
            root: Root node of tree
            indent: Current indentation level
            
        Returns:
            Tree visualization
        """
        lines = []
        prefix = "  " * indent
        
        # Node
        lines.append(f"{prefix}├─ {root.get('question', 'Decision')}")
        
        # Answer/Decision
        if 'answer' in root:
            lines.append(f"{prefix}│  → {root['answer']}")
        
        # Children
        for child in root.get('children', []):
            lines.extend(self.visualize_decision_tree(child, indent + 1).split('\n'))
        
        return '\n'.join(lines)
    
    def visualize_reasoning_chain(
        self,
        steps: List[str],
        show_numbers: bool = True
    ) -> str:
        """
        Visualize reasoning chain.
        
        Args:
            steps: List of reasoning steps
            show_numbers: Whether to show step numbers
            
        Returns:
            Chain visualization
        """
        lines = []
        
        for i, step in enumerate(steps, 1):
            if show_numbers:
                lines.append(f"Step {i}: {step}")
            else:
                lines.append(f"→ {step}")
            
            # Arrow to next step (except last)
            if i < len(steps):
                lines.append("   ↓")
        
        return '\n'.join(lines)
    
    def visualize_factors(
        self,
        factors: List[DecisionFactor]
    ) -> str:
        """
        Visualize decision factors.
        
        Args:
            factors: List of decision factors
            
        Returns:
            Factor visualization
        """
        lines = []
        lines.append("Decision Factors (by importance):\n")
        
        # Sort by weight
        sorted_factors = sorted(factors, key=lambda f: f.weight, reverse=True)
        
        # Find max weight for scaling
        max_weight = max(f.weight for f in sorted_factors) if sorted_factors else 1.0
        
        for factor in sorted_factors:
            # Create bar
            bar_length = int((factor.weight / max_weight) * 30)
            bar = "█" * bar_length
            
            # Format
            lines.append(f"{factor.name:20s} {bar} {factor.weight:.0%}")
            lines.append(f"{'':20s} {factor.description}")
            lines.append("")
        
        return '\n'.join(lines)


def demonstrate_explanation_generation():
    """Demonstrate explanation generation patterns"""
    print("=" * 80)
    print("EXPLANATION GENERATION DEMONSTRATION")
    print("=" * 80)
    
    generator = ExplanationGenerator()
    visualizer = ReasoningVisualizer()
    
    # Example 1: Simple Explanation
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Simple Decision Explanation")
    print("=" * 80)
    
    print("\nGenerating simple explanation...")
    simple_explanation = generator.generate_simple_explanation(
        decision="Recommended Product A",
        context="User looking for a laptop for video editing",
        reasoning="Product A has better GPU and RAM for video editing tasks"
    )
    
    print("\n" + simple_explanation)
    
    # Example 2: Detailed Explanation with Factors
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Detailed Explanation with Multiple Factors")
    print("=" * 80)
    
    # Create structured explanation data
    factors = [
        DecisionFactor(
            name="Performance",
            description="GPU and RAM capacity for video editing",
            weight=0.35,
            evidence="16GB RAM, RTX 3060 GPU",
            impact="positive"
        ),
        DecisionFactor(
            name="Price",
            description="Within user's budget",
            weight=0.25,
            evidence="$1,200 (budget: $1,500)",
            impact="positive"
        ),
        DecisionFactor(
            name="User Reviews",
            description="High satisfaction among video editors",
            weight=0.20,
            evidence="4.5/5 stars, 500+ reviews",
            impact="positive"
        ),
        DecisionFactor(
            name="Battery Life",
            description="Adequate for mobile work",
            weight=0.15,
            evidence="6 hours under load",
            impact="neutral"
        ),
        DecisionFactor(
            name="Weight",
            description="Slightly heavy for portability",
            weight=0.05,
            evidence="5.5 lbs",
            impact="negative"
        )
    ]
    
    alternatives = [
        Alternative(
            option="Product B",
            description="Lighter laptop with good specs",
            pros=["More portable (3.5 lbs)", "Better battery life (8 hours)"],
            cons=["Weaker GPU (GTX 1660)", "More expensive ($1,600)"],
            score=0.75,
            not_chosen_reason="GPU performance insufficient for video editing"
        ),
        Alternative(
            option="Product C",
            description="Budget-friendly option",
            pros=["Lowest price ($900)", "Good battery (7 hours)"],
            cons=["Much weaker GPU", "Only 8GB RAM"],
            score=0.55,
            not_chosen_reason="Performance too limited for professional video editing"
        )
    ]
    
    explanation_data = Explanation(
        decision="Recommend Product A (XPS 15)",
        summary="Product A provides the best balance of performance and value for video editing within budget.",
        reasoning_steps=[
            "Identified video editing as primary use case requiring strong GPU",
            "Filtered products by GPU capability (minimum RTX 3050)",
            "Checked RAM requirements (minimum 16GB for smooth editing)",
            "Compared prices within $1,500 budget",
            "Reviewed user feedback from video editors",
            "Evaluated trade-offs (weight vs performance)"
        ],
        key_factors=factors,
        alternatives=alternatives,
        confidence=0.85,
        sources=[
            "Product specifications database",
            "User review aggregator",
            "Video editing hardware requirements guide"
        ],
        timestamp="2025-11-01T10:30:00Z",
        metadata={
            "model": "gpt-3.5-turbo",
            "execution_time_ms": 1250,
            "user_id": "user_123"
        }
    )
    
    # Generate different explanation levels
    print("\n--- SUMMARY LEVEL ---")
    summary = generator.generate_detailed_explanation(
        explanation_data,
        level=ExplanationLevel.SUMMARY,
        audience=AudienceType.GENERAL
    )
    print(summary)
    
    print("\n\n--- STANDARD LEVEL ---")
    standard = generator.generate_detailed_explanation(
        explanation_data,
        level=ExplanationLevel.STANDARD,
        audience=AudienceType.GENERAL
    )
    print(standard)
    
    # Example 3: Factor Visualization
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Visual Factor Analysis")
    print("=" * 80)
    
    print("\n" + visualizer.visualize_factors(factors))
    
    # Example 4: Reasoning Chain Visualization
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Reasoning Chain Visualization")
    print("=" * 80)
    
    print("\n" + visualizer.visualize_reasoning_chain(explanation_data.reasoning_steps))
    
    # Example 5: Comparison Explanation
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Comparison Explanation")
    print("=" * 80)
    
    print("\nGenerating comparison explanation...")
    comparison = generator.generate_comparison_explanation(
        options=[
            {"name": "Product A", "description": "High performance, moderate price"},
            {"name": "Product B", "description": "Portable, higher price"},
            {"name": "Product C", "description": "Budget option, lower specs"}
        ],
        chosen="Product A",
        criteria=["GPU Performance", "RAM", "Price", "User Reviews", "Portability"]
    )
    print("\n" + comparison)
    
    # Example 6: Error Explanation
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Error Explanation")
    print("=" * 80)
    
    error_explanation = generator.explain_error(
        error_type="Insufficient Information",
        error_message="Cannot make a recommendation without knowing your budget.",
        context="You asked for laptop recommendations for video editing but didn't specify a budget. Budget is a critical factor as video editing laptops range from $800 to $3000+.",
        suggestions=[
            "Please specify your maximum budget",
            "Tell me if you have flexibility in budget if needed for performance",
            "Mention if you're open to refurbished/used options for better value"
        ]
    )
    print("\n" + error_explanation)
    
    # Example 7: Step-by-Step Explanation
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Step-by-Step Process Explanation")
    print("=" * 80)
    
    steps = [
        {
            "name": "Understand Requirements",
            "description": "Analyzed user's need for video editing laptop",
            "input": "User query about laptop for video editing",
            "action": "Extracted key requirement: video editing",
            "output": "Requirement profile created",
            "reasoning": "Video editing requires strong GPU and RAM"
        },
        {
            "name": "Filter by Specs",
            "description": "Filtered products meeting minimum specs",
            "input": "Full product catalog (500+ items)",
            "action": "Applied filters: GPU >= RTX 3050, RAM >= 16GB",
            "output": "15 candidate products",
            "reasoning": "These specs ensure smooth 1080p video editing"
        },
        {
            "name": "Budget Check",
            "description": "Checked prices against budget",
            "input": "15 candidate products, budget $1,500",
            "action": "Filtered by price <= $1,500",
            "output": "5 products within budget",
            "reasoning": "Must stay within user's financial constraints"
        },
        {
            "name": "Review Analysis",
            "description": "Analyzed user reviews and ratings",
            "input": "5 products with reviews",
            "action": "Calculated weighted rating score",
            "output": "Rankings by user satisfaction",
            "reasoning": "Real user experience is crucial indicator"
        },
        {
            "name": "Final Selection",
            "description": "Selected best overall option",
            "input": "Ranked products with all factors",
            "action": "Applied multi-criteria scoring",
            "output": "Product A recommended",
            "reasoning": "Best balance of performance, price, and reviews"
        }
    ]
    
    step_by_step = generator.generate_step_by_step(
        task="Find the Best Video Editing Laptop",
        steps=steps
    )
    print("\n" + step_by_step)
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPLANATION GENERATION SUMMARY")
    print("=" * 80)
    print("""
Explanation Generation Benefits:
1. Transparency: Users understand agent decisions
2. Trust: Clear reasoning builds confidence
3. Debugging: Easier to identify issues
4. Compliance: Documentation for audits
5. Learning: Users learn from explanations
6. Feedback: Better user feedback on decisions

Key Components:
1. Explanation Generator
   - Natural language generation
   - Multiple detail levels
   - Audience adaptation
   - Template-based formatting
   - Context-aware content

2. Reasoning Extractor
   - Captures decision points
   - Tracks information flow
   - Identifies key factors
   - Documents alternatives
   - Maintains causal chains

3. Visualization Builder
   - Text-based visualizations
   - Decision trees
   - Factor importance charts
   - Reasoning chain flows
   - Step-by-step breakdowns

4. Context Enricher
   - Adds background info
   - Includes evidence
   - Cites sources
   - Provides examples
   - Links documentation

Explanation Levels:
1. Summary (1-2 sentences)
   - Main decision
   - Top reason
   - Confidence
   - Best for: Quick overview, dashboards

2. Standard (2-3 paragraphs)
   - Decision + rationale
   - Key factors (2-3)
   - Basic reasoning
   - Best for: General users, reports

3. Detailed (1-2 pages)
   - Full reasoning process
   - All factors with weights
   - Alternatives considered
   - Step-by-step logic
   - Best for: Stakeholders, debugging

4. Technical (2+ pages)
   - Complete technical details
   - Model parameters
   - Data sources
   - Methodology
   - Metadata
   - Best for: Developers, researchers

Audience Types:
1. General Public
   - Simple language
   - Avoid jargon
   - Use analogies
   - Focus on outcomes
   - Concrete examples

2. Business Users
   - Impact on metrics
   - ROI considerations
   - Risk assessment
   - Strategic alignment
   - Action items

3. Technical Users
   - Implementation details
   - Algorithms used
   - Data processing
   - Performance metrics
   - Code references

4. Domain Experts
   - Assumptions made
   - Methodology choices
   - Limitations
   - Alternative approaches
   - Research references

5. Regulators/Auditors
   - Compliance evidence
   - Decision documentation
   - Bias checks
   - Accountability trails
   - Policy adherence

Explanation Structure:
```
1. Decision/Conclusion
   - What was decided
   - High-level summary
   - Confidence level

2. Key Factors (Top 3-5)
   - Factor name
   - Description
   - Evidence/data
   - Impact (positive/negative)
   - Weight/importance

3. Reasoning Process
   - Step 1: [What, Why]
   - Step 2: [What, Why]
   - ...
   - Step N: [What, Why]

4. Alternatives Considered
   - Option A: [Pros, Cons, Score]
   - Option B: [Pros, Cons, Score]
   - Why not chosen

5. Supporting Information
   - Data sources
   - References
   - Related decisions
   - Confidence factors
   - Limitations/caveats
```

Best Practices:
1. Start with Conclusion
   - Lead with the decision
   - Then explain why
   - Don't bury the lead

2. Prioritize Information
   - Most important factors first
   - Top 3-5 factors usually sufficient
   - Details available on request

3. Use Clear Language
   - Short sentences
   - Active voice
   - Concrete terms
   - Define acronyms
   - Consistent terminology

4. Provide Context
   - Background information
   - Constraints considered
   - Assumptions made
   - Scope boundaries

5. Show Alternatives
   - What else was considered
   - Why not chosen
   - Trade-offs made
   - Close calls noted

6. Indicate Uncertainty
   - Confidence levels
   - Areas of uncertainty
   - Data limitations
   - Assumptions that could change

7. Support with Evidence
   - Cite data sources
   - Show calculations
   - Reference policies
   - Link to documentation

Visualization Types:
1. Decision Trees
   - Show branching logic
   - Visualize conditions
   - Trace decision path
   - Highlight chosen branch

2. Factor Charts
   - Bar charts for importance
   - Show relative weights
   - Compare factors
   - Visual hierarchy

3. Reasoning Chains
   - Sequential flow
   - Step-by-step logic
   - Input → Process → Output
   - Causal relationships

4. Comparison Tables
   - Side-by-side options
   - Feature comparisons
   - Score breakdowns
   - Highlight differences

5. Timelines
   - Sequential events
   - Process flow
   - Before/after
   - Milestone tracking

Common Pitfalls:
❌ Too much detail upfront
✓ Start simple, offer details

❌ Technical jargon
✓ Plain language, define terms

❌ No structure
✓ Clear sections, headings

❌ Missing alternatives
✓ Show what else was considered

❌ No confidence indication
✓ State uncertainty levels

❌ Unsupported claims
✓ Cite evidence, sources

Production Tips:
1. Cache common explanations
2. Generate asynchronously
3. Provide summary first, details on request
4. A/B test explanation formats
5. Collect user feedback on clarity
6. Update with model improvements
7. Version explanations with model versions
8. Monitor for hallucinations in explanations

Integration Examples:
- Chatbots: Explain recommendations
- Search: Why these results?
- E-commerce: Product recommendations
- Healthcare: Diagnosis support
- Finance: Investment advice
- Legal: Case analysis
- Education: Tutoring feedback
- Customer service: Solution rationale

Metrics to Track:
- User satisfaction with explanations
- Time to understand
- Follow-up questions needed
- Trust indicators
- Compliance audit pass rate
- Support ticket reduction
- User engagement with details

When to Use:
✓ User-facing decisions
✓ High-stakes choices
✓ Compliance requirements
✓ Learning applications
✓ Debugging & development
✓ Research & analysis
✗ Simple, obvious decisions (maybe)
✗ Real-time critical systems (cache)
✗ Privacy-sensitive contexts (redact)

ROI Analysis:
- User trust: 30-50% increase
- Support tickets: 20-40% reduction
- Debugging time: 40-60% faster
- Compliance: Audit-ready documentation
- User satisfaction: 15-30% improvement
- Cost: Low (mostly template-based)
""")
    
    print("\n" + "=" * 80)
    print("Pattern 095 (Explanation Generation) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_explanation_generation()
