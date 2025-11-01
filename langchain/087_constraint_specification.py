"""
Pattern 087: Constraint Specification Pattern

Description:
    The Constraint Specification pattern explicitly defines boundaries, rules, and
    limitations that the LLM must adhere to when generating responses. This pattern
    is crucial for controlling LLM behavior, ensuring compliance with requirements,
    and preventing undesired outputs. Constraints can be content-based, format-based,
    style-based, or domain-specific, and they guide the model toward appropriate
    responses while filtering out inappropriate ones.

    Constraint specification is essential for:
    - Safety and compliance requirements
    - Brand voice and style consistency
    - Content appropriateness filtering
    - Domain-specific requirements
    - Length and complexity control
    - Factual accuracy boundaries

Components:
    1. Constraint Types
       - Content constraints (what to include/exclude)
       - Format constraints (structure and presentation)
       - Style constraints (tone, formality, vocabulary)
       - Length constraints (word/character limits)
       - Domain constraints (topic boundaries)
       - Safety constraints (harmful content prevention)

    2. Constraint Expression
       - Explicit rules ("must", "must not")
       - Positive constraints (requirements)
       - Negative constraints (prohibitions)
       - Conditional constraints (if-then rules)
       - Priority levels (strict vs flexible)

    3. Enforcement Strategies
       - Pre-generation (prompt engineering)
       - Post-generation (validation and filtering)
       - Iterative refinement (generate-validate-refine)
       - Guardrails (safety layers)
       - Fallback handling (when constraints violated)

    4. Validation Methods
       - Rule-based checking
       - Pattern matching
       - Content analysis
       - Length verification
       - Format validation
       - Safety scoring

Use Cases:
    1. Content Safety
       - Prevent harmful content generation
       - Filter inappropriate language
       - Block sensitive topics
       - Ensure age-appropriate content
       - Compliance with platform policies

    2. Brand Consistency
       - Maintain brand voice
       - Use approved terminology
       - Follow style guidelines
       - Consistent tone across content
       - Corporate communication standards

    3. Domain Expertise
       - Stay within knowledge boundaries
       - Acknowledge uncertainty
       - Avoid medical/legal claims
       - Respect professional standards
       - Domain-specific terminology

    4. Format and Structure
       - Length limitations (tweets, abstracts)
       - Structural requirements (sections, headings)
       - Citation requirements
       - Formatting rules (markdown, HTML)
       - Template compliance

    5. Legal and Compliance
       - GDPR compliance (no PII without consent)
       - HIPAA compliance (no PHI exposure)
       - Copyright respect
       - Regulatory requirements
       - Terms of service adherence

LangChain Implementation:
    LangChain supports constraint specification through:
    - System messages with explicit rules
    - Constitutional AI principles
    - Output validators and filters
    - Custom guardrails
    - Retry mechanisms with corrections

Key Features:
    1. Multi-Level Constraints
       - Hard constraints (must be satisfied)
       - Soft constraints (preferred but flexible)
       - Conditional constraints (context-dependent)
       - Hierarchical constraints (priority ordering)

    2. Constraint Composition
       - Multiple constraints simultaneously
       - Constraint conflicts resolution
       - Priority-based enforcement
       - Constraint inheritance

    3. Validation and Feedback
       - Immediate validation
       - Detailed violation reporting
       - Correction suggestions
       - Iterative improvement

    4. Dynamic Constraints
       - Context-aware constraints
       - User-specific rules
       - Time-based constraints
       - Adaptive boundaries

Best Practices:
    1. Clarity and Specificity
       - State constraints explicitly
       - Provide examples of violations
       - Clarify edge cases
       - Use concrete language

    2. Prioritization
       - Identify critical constraints
       - Allow flexibility where possible
       - Balance constraints with quality
       - Avoid over-constraining

    3. Testing
       - Test constraint enforcement
       - Try boundary cases
       - Verify violation detection
       - Check false positives

    4. Documentation
       - Document all constraints
       - Explain rationale
       - Provide violation examples
       - Maintain constraint catalog

Trade-offs:
    Advantages:
    - Better control over outputs
    - Improved safety and compliance
    - Consistent brand voice
    - Reduced inappropriate content
    - Clear expectations
    - Easier validation

    Disadvantages:
    - May limit creativity
    - Can reduce output quality
    - Increases complexity
    - Requires careful design
    - May need frequent updates
    - Potential over-constraint

Production Considerations:
    1. Constraint Management
       - Centralized constraint definitions
       - Version control for constraints
       - Easy updates and modifications
       - Constraint testing framework
       - Documentation and examples

    2. Performance Impact
       - Validation overhead
       - Retry costs for violations
       - Caching validated outputs
       - Efficient constraint checking
       - Batch validation

    3. User Experience
       - Clear violation messages
       - Suggestion for corrections
       - Graceful degradation
       - Transparency about constraints
       - User constraint customization

    4. Monitoring
       - Track constraint violations
       - Analyze violation patterns
       - Measure enforcement effectiveness
       - Identify constraint conflicts
       - User feedback collection

    5. Evolution
       - Regular constraint reviews
       - Adapt to new requirements
       - Learn from violations
       - Community feedback
       - A/B test constraint changes
"""

import os
import re
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ConstraintType(Enum):
    """Types of constraints"""
    CONTENT = "content"
    LENGTH = "length"
    STYLE = "style"
    FORMAT = "format"
    SAFETY = "safety"
    DOMAIN = "domain"


class ConstraintSeverity(Enum):
    """Severity levels for constraints"""
    STRICT = "strict"  # Must be satisfied
    MODERATE = "moderate"  # Should be satisfied
    FLEXIBLE = "flexible"  # Nice to have


@dataclass
class Constraint:
    """Represents a single constraint"""
    name: str
    description: str
    constraint_type: ConstraintType
    severity: ConstraintSeverity
    validation_func: Optional[Callable[[str], bool]] = None
    violation_message: str = "Constraint violated"


@dataclass
class ConstraintConfig:
    """Configuration for constraint-based generation"""
    max_retries: int = 3
    temperature: float = 0.5
    model_name: str = "gpt-3.5-turbo"
    fail_on_violation: bool = False


class ConstraintSpecificationAgent:
    """
    Agent that enforces constraints on LLM outputs.
    
    This agent demonstrates:
    1. Defining multiple types of constraints
    2. Enforcing constraints during generation
    3. Validating outputs against constraints
    4. Handling constraint violations
    """
    
    def __init__(self, config: Optional[ConstraintConfig] = None):
        """
        Initialize constraint specification agent.
        
        Args:
            config: Configuration for constraint handling
        """
        self.config = config or ConstraintConfig()
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature
        )
        self.constraints: List[Constraint] = []
    
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the agent"""
        self.constraints.append(constraint)
    
    def clear_constraints(self):
        """Remove all constraints"""
        self.constraints = []
    
    def _build_constraint_instructions(self) -> str:
        """Build constraint instructions for prompt"""
        if not self.constraints:
            return ""
        
        instructions = ["You must follow these constraints:"]
        
        for i, constraint in enumerate(self.constraints, 1):
            severity_label = {
                ConstraintSeverity.STRICT: "MUST",
                ConstraintSeverity.MODERATE: "SHOULD",
                ConstraintSeverity.FLEXIBLE: "IDEALLY"
            }[constraint.severity]
            
            instructions.append(f"{i}. {severity_label}: {constraint.description}")
        
        return "\n".join(instructions)
    
    def _validate_constraints(self, text: str) -> Dict[str, Any]:
        """
        Validate text against all constraints.
        
        Args:
            text: Text to validate
            
        Returns:
            Validation results
        """
        violations = []
        
        for constraint in self.constraints:
            if constraint.validation_func:
                if not constraint.validation_func(text):
                    violations.append({
                        "constraint": constraint.name,
                        "severity": constraint.severity.value,
                        "message": constraint.violation_message
                    })
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "num_violations": len(violations)
        }
    
    def generate_with_constraints(
        self,
        prompt: str,
        task_description: str = "Generate a response"
    ) -> Dict[str, Any]:
        """
        Generate text while enforcing constraints.
        
        Args:
            prompt: User prompt
            task_description: Description of task
            
        Returns:
            Generation result with validation info
        """
        constraint_instructions = self._build_constraint_instructions()
        
        template = ChatPromptTemplate.from_messages([
            ("system", """{task_description}

{constraints}

Ensure your response satisfies all constraints."""),
            ("human", "{prompt}")
        ])
        
        chain = template | self.llm | StrOutputParser()
        
        for attempt in range(self.config.max_retries):
            # Generate
            result = chain.invoke({
                "task_description": task_description,
                "constraints": constraint_instructions,
                "prompt": prompt
            })
            
            # Validate
            validation = self._validate_constraints(result)
            
            if validation["valid"]:
                return {
                    "success": True,
                    "text": result,
                    "attempts": attempt + 1,
                    "validation": validation
                }
            
            # If not last attempt, provide feedback for retry
            if attempt < self.config.max_retries - 1:
                violation_feedback = "\n".join([
                    f"- {v['constraint']}: {v['message']}"
                    for v in validation["violations"]
                ])
                
                # Retry with violation feedback (implicit in next iteration)
                continue
        
        # Max retries exceeded
        return {
            "success": False,
            "text": result,
            "attempts": self.config.max_retries,
            "validation": validation,
            "error": "Max retries exceeded with constraint violations"
        }
    
    def generate_with_length_constraint(
        self,
        prompt: str,
        max_words: int,
        min_words: Optional[int] = None
    ) -> str:
        """
        Generate text with length constraints.
        
        Args:
            prompt: User prompt
            max_words: Maximum word count
            min_words: Optional minimum word count
            
        Returns:
            Generated text
        """
        self.clear_constraints()
        
        # Add length constraint
        def validate_length(text: str) -> bool:
            word_count = len(text.split())
            if min_words and word_count < min_words:
                return False
            if word_count > max_words:
                return False
            return True
        
        length_desc = f"between {min_words} and {max_words} words" if min_words else f"at most {max_words} words"
        
        self.add_constraint(Constraint(
            name="length",
            description=f"Response must be {length_desc}",
            constraint_type=ConstraintType.LENGTH,
            severity=ConstraintSeverity.STRICT,
            validation_func=validate_length,
            violation_message=f"Length constraint violated (required: {length_desc})"
        ))
        
        result = self.generate_with_constraints(prompt)
        return result["text"] if result["success"] else result.get("error", "Generation failed")
    
    def generate_with_style_constraint(
        self,
        prompt: str,
        style: str = "professional"
    ) -> str:
        """
        Generate text with style constraints.
        
        Args:
            prompt: User prompt
            style: Desired style (professional, casual, formal)
            
        Returns:
            Generated text
        """
        self.clear_constraints()
        
        style_guidelines = {
            "professional": "Use professional language, avoid slang, maintain objective tone",
            "casual": "Use conversational language, contractions are fine, friendly tone",
            "formal": "Use formal language, no contractions, academic or business tone",
            "humorous": "Use humor and wit, light-hearted tone, creative wordplay"
        }
        
        guideline = style_guidelines.get(style.lower(), style_guidelines["professional"])
        
        self.add_constraint(Constraint(
            name="style",
            description=guideline,
            constraint_type=ConstraintType.STYLE,
            severity=ConstraintSeverity.MODERATE
        ))
        
        result = self.generate_with_constraints(
            prompt,
            task_description=f"Generate a {style} response"
        )
        
        return result["text"] if result["success"] else result.get("text", "")
    
    def generate_with_content_constraints(
        self,
        prompt: str,
        must_include: Optional[List[str]] = None,
        must_exclude: Optional[List[str]] = None
    ) -> str:
        """
        Generate text with content inclusion/exclusion constraints.
        
        Args:
            prompt: User prompt
            must_include: Terms that must appear
            must_exclude: Terms that must not appear
            
        Returns:
            Generated text
        """
        self.clear_constraints()
        
        # Inclusion constraint
        if must_include:
            def validate_inclusion(text: str) -> bool:
                text_lower = text.lower()
                return all(term.lower() in text_lower for term in must_include)
            
            self.add_constraint(Constraint(
                name="content_inclusion",
                description=f"Must include these terms: {', '.join(must_include)}",
                constraint_type=ConstraintType.CONTENT,
                severity=ConstraintSeverity.STRICT,
                validation_func=validate_inclusion,
                violation_message=f"Required terms missing: {', '.join(must_include)}"
            ))
        
        # Exclusion constraint
        if must_exclude:
            def validate_exclusion(text: str) -> bool:
                text_lower = text.lower()
                return not any(term.lower() in text_lower for term in must_exclude)
            
            self.add_constraint(Constraint(
                name="content_exclusion",
                description=f"Must NOT include these terms: {', '.join(must_exclude)}",
                constraint_type=ConstraintType.CONTENT,
                severity=ConstraintSeverity.STRICT,
                validation_func=validate_exclusion,
                violation_message=f"Prohibited terms found: {', '.join(must_exclude)}"
            ))
        
        result = self.generate_with_constraints(prompt)
        return result["text"] if result["success"] else result.get("text", "")
    
    def generate_with_format_constraint(
        self,
        prompt: str,
        format_type: str = "bullet_points"
    ) -> str:
        """
        Generate text with format constraints.
        
        Args:
            prompt: User prompt
            format_type: Format type (bullet_points, numbered_list, paragraphs)
            
        Returns:
            Generated text
        """
        self.clear_constraints()
        
        format_descriptions = {
            "bullet_points": "Use bullet points (- or •) for each item. No numbered lists.",
            "numbered_list": "Use numbered list format (1. 2. 3. etc.). No bullet points.",
            "paragraphs": "Write in paragraph form with clear topic sentences. No lists.",
            "single_sentence": "Provide answer as a single, concise sentence."
        }
        
        description = format_descriptions.get(format_type, format_descriptions["bullet_points"])
        
        self.add_constraint(Constraint(
            name="format",
            description=description,
            constraint_type=ConstraintType.FORMAT,
            severity=ConstraintSeverity.STRICT
        ))
        
        result = self.generate_with_constraints(
            prompt,
            task_description=f"Generate response in {format_type} format"
        )
        
        return result["text"] if result["success"] else result.get("text", "")
    
    def generate_with_safety_constraints(
        self,
        prompt: str
    ) -> str:
        """
        Generate text with safety constraints.
        
        Args:
            prompt: User prompt
            
        Returns:
            Generated text
        """
        self.clear_constraints()
        
        safety_rules = [
            "No harmful, hateful, or violent content",
            "No personal attacks or discrimination",
            "No illegal activities or advice",
            "No explicit or inappropriate content",
            "Respectful and inclusive language"
        ]
        
        for rule in safety_rules:
            self.add_constraint(Constraint(
                name=f"safety_{safety_rules.index(rule)}",
                description=rule,
                constraint_type=ConstraintType.SAFETY,
                severity=ConstraintSeverity.STRICT
            ))
        
        result = self.generate_with_constraints(
            prompt,
            task_description="Generate a safe and appropriate response"
        )
        
        return result["text"] if result["success"] else "Unable to generate safe content for this prompt."


def demonstrate_constraint_specification():
    """Demonstrate constraint specification patterns"""
    print("=" * 80)
    print("CONSTRAINT SPECIFICATION PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = ConstraintSpecificationAgent()
    
    # Example 1: Length Constraints
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Length Constraints")
    print("=" * 80)
    
    prompt = "Explain artificial intelligence"
    print(f"\nPrompt: {prompt}")
    print("\nConstraint: Maximum 50 words\n")
    
    short_response = agent.generate_with_length_constraint(prompt, max_words=50)
    print(f"Response ({len(short_response.split())} words):")
    print(short_response)
    
    print("\nConstraint: 80-120 words\n")
    medium_response = agent.generate_with_length_constraint(prompt, max_words=120, min_words=80)
    print(f"Response ({len(medium_response.split())} words):")
    print(medium_response)
    
    # Example 2: Style Constraints
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Style Constraints")
    print("=" * 80)
    
    prompt = "Describe the benefits of exercise"
    print(f"\nPrompt: {prompt}\n")
    
    styles = ["professional", "casual", "formal"]
    for style in styles:
        print(f"\n{style.upper()} STYLE:")
        print("-" * 40)
        response = agent.generate_with_style_constraint(prompt, style=style)
        print(response)
    
    # Example 3: Content Inclusion/Exclusion
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Content Constraints (Must Include/Exclude)")
    print("=" * 80)
    
    prompt = "Write about healthy eating habits"
    must_include = ["vegetables", "protein", "hydration"]
    must_exclude = ["diet pills", "supplements"]
    
    print(f"\nPrompt: {prompt}")
    print(f"Must include: {must_include}")
    print(f"Must exclude: {must_exclude}\n")
    
    response = agent.generate_with_content_constraints(
        prompt,
        must_include=must_include,
        must_exclude=must_exclude
    )
    print("Response:")
    print(response)
    
    # Example 4: Format Constraints
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Format Constraints")
    print("=" * 80)
    
    prompt = "List the steps to start a business"
    print(f"\nPrompt: {prompt}\n")
    
    formats = ["bullet_points", "numbered_list", "paragraphs"]
    for fmt in formats:
        print(f"\n{fmt.upper().replace('_', ' ')} FORMAT:")
        print("-" * 40)
        response = agent.generate_with_format_constraint(prompt, format_type=fmt)
        print(response)
    
    # Example 5: Multiple Constraints
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Multiple Simultaneous Constraints")
    print("=" * 80)
    
    agent.clear_constraints()
    
    # Add multiple constraints
    agent.add_constraint(Constraint(
        name="length",
        description="Response must be 60-80 words",
        constraint_type=ConstraintType.LENGTH,
        severity=ConstraintSeverity.STRICT,
        validation_func=lambda t: 60 <= len(t.split()) <= 80,
        violation_message="Length must be 60-80 words"
    ))
    
    agent.add_constraint(Constraint(
        name="style",
        description="Use professional, objective tone",
        constraint_type=ConstraintType.STYLE,
        severity=ConstraintSeverity.MODERATE
    ))
    
    agent.add_constraint(Constraint(
        name="content",
        description="Must mention 'innovation' and 'technology'",
        constraint_type=ConstraintType.CONTENT,
        severity=ConstraintSeverity.STRICT,
        validation_func=lambda t: "innovation" in t.lower() and "technology" in t.lower(),
        violation_message="Must include 'innovation' and 'technology'"
    ))
    
    prompt = "Describe the future of work"
    print(f"\nPrompt: {prompt}")
    print("\nConstraints:")
    print("1. Length: 60-80 words")
    print("2. Style: Professional, objective")
    print("3. Content: Must mention 'innovation' and 'technology'\n")
    
    result = agent.generate_with_constraints(prompt)
    print(f"Success: {result['success']}")
    print(f"Attempts: {result['attempts']}")
    print(f"\nResponse ({len(result['text'].split())} words):")
    print(result['text'])
    
    # Example 6: Safety Constraints
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Safety Constraints")
    print("=" * 80)
    
    safe_prompt = "Give advice on managing workplace conflicts"
    print(f"\nPrompt: {safe_prompt}")
    print("\nApplying safety constraints...\n")
    
    response = agent.generate_with_safety_constraints(safe_prompt)
    print("Response:")
    print(response)
    
    # Summary
    print("\n" + "=" * 80)
    print("CONSTRAINT SPECIFICATION SUMMARY")
    print("=" * 80)
    print("""
Constraint Specification Pattern Benefits:
1. Control: Precise control over LLM outputs
2. Compliance: Ensures regulatory and policy adherence
3. Consistency: Maintains brand voice and standards
4. Safety: Prevents inappropriate content
5. Quality: Enforces quality standards
6. Integration: Easier to integrate with systems

Constraint Types Demonstrated:
1. Length Constraints: Word/character limits
2. Style Constraints: Tone, formality, voice
3. Content Constraints: Required/prohibited terms
4. Format Constraints: Structure and presentation
5. Safety Constraints: Harmful content prevention
6. Multiple Constraints: Combined requirements

Constraint Severity Levels:
- STRICT: Must be satisfied (violations cause failure)
- MODERATE: Should be satisfied (violations logged)
- FLEXIBLE: Nice to have (soft preferences)

Enforcement Strategies:
1. Pre-generation: Include constraints in prompt
2. Validation: Check outputs against rules
3. Iteration: Retry with feedback on violations
4. Fallback: Alternative approach when constraints can't be met

Best Practices:
1. Be explicit and specific with constraints
2. Prioritize constraints (strict vs flexible)
3. Provide examples of acceptable outputs
4. Test constraint enforcement thoroughly
5. Handle violations gracefully
6. Monitor constraint effectiveness
7. Update constraints based on feedback

Common Use Cases:
- Content moderation (safety)
- Brand consistency (style)
- Legal compliance (domain rules)
- Format standardization (structure)
- Length requirements (summaries, tweets)
- Quality control (standards)

Production Considerations:
- Balance constraints with creativity
- Allow appropriate flexibility
- Provide clear violation messages
- Implement retry mechanisms
- Monitor constraint violation rates
- A/B test constraint effectiveness
- Document all constraints clearly
- Version control constraint rules

Constraint Design Tips:
1. Start with essential constraints only
2. Add constraints incrementally
3. Test against diverse inputs
4. Avoid over-constraining
5. Make constraints measurable
6. Provide violation feedback
7. Allow for constraint evolution

When to Use Constraints:
- Safety-critical applications
- Regulated industries
- Brand-sensitive content
- System integration requirements
- Quality assurance needs
- Compliance requirements

Trade-offs:
✅ Better control and safety
✅ Consistent outputs
✅ Easier validation
❌ May limit creativity
❌ Requires careful design
❌ Can be overly restrictive
""")
    
    print("\n" + "=" * 80)
    print("Pattern 087 (Constraint Specification) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_constraint_specification()
