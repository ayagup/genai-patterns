"""
Pattern 045: Defensive Generation

Description:
    Defensive Generation enables agents to generate responses with built-in
    safety considerations including content filtering, bias mitigation, toxicity
    avoidance, and harmful output prevention. Implements multiple layers of
    safety checks before, during, and after generation.

Components:
    - Safety Classifier: Detects harmful content
    - Bias Detector: Identifies biased language
    - Content Filter: Blocks inappropriate content
    - Safety Rewriter: Reformulates unsafe responses
    - Policy Enforcer: Ensures compliance with policies

Use Cases:
    - Public-facing chatbots
    - Content moderation systems
    - Educational applications
    - Healthcare assistants
    - Customer service bots
    - Enterprise AI systems

LangChain Implementation:
    Uses multi-layered safety checks with content classification, filtering,
    and reformulation to ensure generated content is safe and appropriate.
"""

import os
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class SafetyLevel(Enum):
    """Safety assessment levels."""
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"


class SafetyCategory(Enum):
    """Categories of safety concerns."""
    TOXICITY = "toxicity"
    BIAS = "bias"
    HARMFUL_CONTENT = "harmful_content"
    PRIVACY_VIOLATION = "privacy_violation"
    INAPPROPRIATE = "inappropriate"
    MISINFORMATION = "misinformation"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"


@dataclass
class SafetyViolation:
    """A detected safety violation."""
    category: SafetyCategory
    severity: float  # 0.0 to 1.0
    description: str
    location: Optional[str] = None


@dataclass
class SafetyAssessment:
    """Complete safety assessment of content."""
    content: str
    safety_level: SafetyLevel
    violations: List[SafetyViolation]
    is_safe: bool
    confidence: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DefensiveGenerationResult:
    """Result of defensive generation."""
    query: str
    raw_response: str
    final_response: str
    input_assessment: SafetyAssessment
    output_assessment: SafetyAssessment
    was_filtered: bool
    was_rewritten: bool
    safety_actions: List[str]


class DefensiveGenerationAgent:
    """
    Agent that generates responses with comprehensive safety checks.
    
    Features:
    - Input safety validation
    - Output safety checking
    - Content filtering
    - Bias detection
    - Automatic rewriting
    - Policy enforcement
    """
    
    def __init__(
        self,
        safety_threshold: float = 0.7,
        enable_rewriting: bool = True,
        temperature: float = 0.5
    ):
        self.safety_threshold = safety_threshold
        self.enable_rewriting = enable_rewriting
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        
        # Blocked keywords and patterns
        self.blocked_keywords = {
            'violence', 'hate', 'discriminatory', 'harmful', 'illegal',
            'dangerous', 'explicit', 'inappropriate'
        }
        
        # Response generation prompt
        self.generation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful, safe, and responsible AI assistant.

Guidelines:
- Provide accurate, helpful information
- Avoid harmful, biased, or inappropriate content
- Respect user privacy
- Decline inappropriate requests politely
- Be objective and fair"""),
            ("user", "{query}")
        ])
        
        # Safety assessment prompt
        self.safety_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a content safety evaluator. Assess the safety of the content.

Evaluate for:
- Toxicity and hate speech
- Bias and discrimination
- Harmful or dangerous content
- Privacy violations
- Inappropriate content
- Misinformation

Provide:
SAFETY_LEVEL: [safe/caution/unsafe/blocked]
VIOLATIONS: [List any violations with category and severity]
CONFIDENCE: [0.0-1.0]
RECOMMENDATIONS: [Safety improvements if needed]"""),
            ("user", "Content to assess:\n{content}\n\nSafety assessment:")
        ])
        
        # Rewriting prompt
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a content safety editor. Rewrite content to be safe and appropriate.

Instructions:
- Remove harmful, biased, or inappropriate content
- Maintain helpfulness and accuracy
- Be respectful and inclusive
- Preserve the core information
- Decline if request is inherently unsafe

Original issues:
{violations}"""),
            ("user", "Original content:\n{content}\n\nProvide safe rewritten version:")
        ])
        
        # Generation history
        self.generations: List[DefensiveGenerationResult] = []
    
    def assess_safety(self, content: str) -> SafetyAssessment:
        """
        Assess safety of content.
        
        Returns:
            SafetyAssessment with violations and recommendations
        """
        # Quick keyword check
        content_lower = content.lower()
        keyword_violations = []
        for keyword in self.blocked_keywords:
            if keyword in content_lower:
                keyword_violations.append(SafetyViolation(
                    category=SafetyCategory.INAPPROPRIATE,
                    severity=0.6,
                    description=f"Contains potentially problematic keyword: {keyword}"
                ))
        
        # LLM-based assessment
        chain = self.safety_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"content": content})
        
        # Parse assessment
        safety_level = SafetyLevel.SAFE
        violations = keyword_violations.copy()
        confidence = 0.8
        recommendations = []
        
        lines = result.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("SAFETY_LEVEL:"):
                level_text = line.split(':')[1].strip().lower()
                try:
                    safety_level = SafetyLevel(level_text)
                except ValueError:
                    pass
            elif line.startswith("VIOLATIONS:"):
                current_section = "violations"
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
                current_section = None
            elif line.startswith("RECOMMENDATIONS:"):
                current_section = "recommendations"
            elif line and line.startswith('-') and current_section:
                item = line[1:].strip()
                if current_section == "violations":
                    # Parse violation (format: category - description)
                    if ' - ' in item:
                        cat_text, desc = item.split(' - ', 1)
                        try:
                            category = SafetyCategory(cat_text.strip().lower())
                        except ValueError:
                            category = SafetyCategory.INAPPROPRIATE
                        
                        violations.append(SafetyViolation(
                            category=category,
                            severity=0.7,
                            description=desc
                        ))
                elif current_section == "recommendations":
                    recommendations.append(item)
        
        is_safe = safety_level in [SafetyLevel.SAFE, SafetyLevel.CAUTION]
        
        return SafetyAssessment(
            content=content,
            safety_level=safety_level,
            violations=violations,
            is_safe=is_safe,
            confidence=confidence,
            recommendations=recommendations
        )
    
    def rewrite_for_safety(
        self,
        content: str,
        violations: List[SafetyViolation]
    ) -> str:
        """
        Rewrite content to address safety violations.
        
        Returns:
            Rewritten safe content
        """
        if not violations:
            return content
        
        violations_text = "\n".join([
            f"- {v.category.value}: {v.description}"
            for v in violations
        ])
        
        chain = self.rewrite_prompt | self.llm | StrOutputParser()
        rewritten = chain.invoke({
            "content": content,
            "violations": violations_text
        })
        
        return rewritten
    
    def generate_defensively(
        self,
        query: str
    ) -> DefensiveGenerationResult:
        """
        Generate response with comprehensive safety checks.
        
        Process:
        1. Assess input safety
        2. Block if input is unsafe
        3. Generate response
        4. Assess output safety
        5. Rewrite if needed
        6. Return safest version
        
        Args:
            query: User query
            
        Returns:
            DefensiveGenerationResult with safety information
        """
        safety_actions = []
        
        # Step 1: Assess input safety
        input_assessment = self.assess_safety(query)
        
        # Step 2: Block unsafe inputs
        if input_assessment.safety_level == SafetyLevel.BLOCKED:
            safety_actions.append("Input blocked due to safety violations")
            
            blocked_response = "I apologize, but I cannot respond to that request as it violates safety guidelines. Please rephrase your question in a respectful and appropriate manner."
            
            output_assessment = SafetyAssessment(
                content=blocked_response,
                safety_level=SafetyLevel.SAFE,
                violations=[],
                is_safe=True,
                confidence=1.0,
                recommendations=[]
            )
            
            return DefensiveGenerationResult(
                query=query,
                raw_response="[BLOCKED]",
                final_response=blocked_response,
                input_assessment=input_assessment,
                output_assessment=output_assessment,
                was_filtered=True,
                was_rewritten=False,
                safety_actions=safety_actions
            )
        
        # Step 3: Generate response
        chain = self.generation_prompt | self.llm | StrOutputParser()
        raw_response = chain.invoke({"query": query})
        
        # Step 4: Assess output safety
        output_assessment = self.assess_safety(raw_response)
        
        final_response = raw_response
        was_filtered = False
        was_rewritten = False
        
        # Step 5: Handle unsafe outputs
        if not output_assessment.is_safe:
            safety_actions.append(f"Output safety violation: {output_assessment.safety_level.value}")
            
            if output_assessment.safety_level == SafetyLevel.BLOCKED:
                # Block completely
                final_response = "I apologize, but I cannot provide that information as it may be harmful or inappropriate."
                was_filtered = True
                safety_actions.append("Output blocked and replaced with safe message")
            
            elif self.enable_rewriting and output_assessment.violations:
                # Try to rewrite
                safety_actions.append("Attempting to rewrite for safety")
                rewritten = self.rewrite_for_safety(raw_response, output_assessment.violations)
                
                # Assess rewritten version
                rewritten_assessment = self.assess_safety(rewritten)
                
                if rewritten_assessment.is_safe:
                    final_response = rewritten
                    output_assessment = rewritten_assessment
                    was_rewritten = True
                    safety_actions.append("Successfully rewritten for safety")
                else:
                    # Rewriting failed, block
                    final_response = "I apologize, but I cannot provide a safe response to that query."
                    was_filtered = True
                    safety_actions.append("Rewriting failed, output blocked")
        else:
            safety_actions.append("Output passed safety checks")
        
        result = DefensiveGenerationResult(
            query=query,
            raw_response=raw_response,
            final_response=final_response,
            input_assessment=input_assessment,
            output_assessment=output_assessment,
            was_filtered=was_filtered,
            was_rewritten=was_rewritten,
            safety_actions=safety_actions
        )
        
        self.generations.append(result)
        
        return result
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get statistics about safety interventions."""
        if not self.generations:
            return {"total_generations": 0}
        
        total = len(self.generations)
        filtered = sum(1 for g in self.generations if g.was_filtered)
        rewritten = sum(1 for g in self.generations if g.was_rewritten)
        
        # Violation categories
        violation_counts = {}
        for gen in self.generations:
            for violation in gen.output_assessment.violations:
                cat = violation.category.value
                violation_counts[cat] = violation_counts.get(cat, 0) + 1
        
        return {
            "total_generations": total,
            "filtered_count": filtered,
            "rewritten_count": rewritten,
            "filter_rate": filtered / total,
            "rewrite_rate": rewritten / total,
            "violation_breakdown": violation_counts
        }


def demonstrate_defensive_generation():
    """
    Demonstrates defensive generation with safety checks.
    """
    print("=" * 80)
    print("DEFENSIVE GENERATION DEMONSTRATION")
    print("=" * 80)
    
    # Create defensive agent
    agent = DefensiveGenerationAgent(
        safety_threshold=0.7,
        enable_rewriting=True,
        temperature=0.5
    )
    
    # Test 1: Safe query
    print("\n" + "=" * 80)
    print("Test 1: Safe Query")
    print("=" * 80)
    
    query1 = "How can I improve my programming skills?"
    print(f"\nQuery: {query1}")
    
    result1 = agent.generate_defensively(query1)
    
    print("\n[Input Assessment]")
    print(f"Safety Level: {result1.input_assessment.safety_level.value}")
    print(f"Is Safe: {result1.input_assessment.is_safe}")
    print(f"Violations: {len(result1.input_assessment.violations)}")
    
    print("\n[Output Assessment]")
    print(f"Safety Level: {result1.output_assessment.safety_level.value}")
    print(f"Is Safe: {result1.output_assessment.is_safe}")
    print(f"Was Filtered: {result1.was_filtered}")
    print(f"Was Rewritten: {result1.was_rewritten}")
    
    print("\n[Final Response]")
    print(result1.final_response[:300] + "..." if len(result1.final_response) > 300 else result1.final_response)
    
    print("\n[Safety Actions]")
    for action in result1.safety_actions:
        print(f"  • {action}")
    
    # Test 2: Potentially problematic query
    print("\n" + "=" * 80)
    print("Test 2: Query Requiring Careful Response")
    print("=" * 80)
    
    query2 = "What are the effects of stress on health?"
    print(f"\nQuery: {query2}")
    
    result2 = agent.generate_defensively(query2)
    
    print("\n[Safety Analysis]")
    print(f"Input Safety: {result2.input_assessment.safety_level.value}")
    print(f"Output Safety: {result2.output_assessment.safety_level.value}")
    print(f"Interventions: Filtered={result2.was_filtered}, Rewritten={result2.was_rewritten}")
    
    print("\n[Response Preview]")
    print(result2.final_response[:250] + "..." if len(result2.final_response) > 250 else result2.final_response)
    
    # Test 3: Query with safety concerns
    print("\n" + "=" * 80)
    print("Test 3: Query with Potential Safety Issues")
    print("=" * 80)
    
    query3 = "Tell me about controversial topics in politics"
    print(f"\nQuery: {query3}")
    
    result3 = agent.generate_defensively(query3)
    
    print("\n[Safety Assessment]")
    print(f"Input Level: {result3.input_assessment.safety_level.value}")
    print(f"Output Level: {result3.output_assessment.safety_level.value}")
    
    if result3.input_assessment.violations:
        print("\n[Input Violations Detected]")
        for v in result3.input_assessment.violations:
            print(f"  • {v.category.value}: {v.description}")
    
    if result3.output_assessment.violations:
        print("\n[Output Violations Detected]")
        for v in result3.output_assessment.violations:
            print(f"  • {v.category.value}: {v.description}")
    
    print("\n[Safety Actions Taken]")
    for action in result3.safety_actions:
        print(f"  • {action}")
    
    print("\n[Final Response]")
    print(result3.final_response[:200] + "..." if len(result3.final_response) > 200 else result3.final_response)
    
    # Test 4: Multiple queries to show statistics
    print("\n" + "=" * 80)
    print("Test 4: Safety Statistics")
    print("=" * 80)
    
    # Generate a few more responses
    test_queries = [
        "What are the benefits of exercise?",
        "How does climate change affect the environment?",
        "What are some healthy eating tips?"
    ]
    
    print("\nGenerating additional responses...")
    for query in test_queries:
        agent.generate_defensively(query)
    
    # Show statistics
    stats = agent.get_safety_statistics()
    
    print("\n[Safety Statistics]")
    print(f"Total Generations: {stats['total_generations']}")
    print(f"Filtered Count: {stats['filtered_count']}")
    print(f"Rewritten Count: {stats['rewritten_count']}")
    print(f"Filter Rate: {stats['filter_rate']:.1%}")
    print(f"Rewrite Rate: {stats['rewrite_rate']:.1%}")
    
    if stats['violation_breakdown']:
        print("\n[Violation Breakdown]")
        for category, count in sorted(stats['violation_breakdown'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Defensive Generation provides:
✓ Multi-layered safety checks
✓ Input validation
✓ Output filtering
✓ Content rewriting
✓ Policy enforcement
✓ Comprehensive safety metrics

This pattern excels at:
- Public-facing applications
- Content moderation
- Healthcare and education
- Enterprise systems
- Customer service
- Sensitive domains

Safety layers:
1. Input Assessment: Check query safety
2. Blocked Keyword Detection: Quick filters
3. Generation Guidelines: Safe prompting
4. Output Assessment: Check response safety
5. Content Rewriting: Fix unsafe content
6. Final Filtering: Last line of defense

Safety categories:
- Toxicity: Offensive language
- Bias: Discriminatory content
- Harmful Content: Dangerous information
- Privacy Violations: Personal data exposure
- Inappropriate: Unsuitable content
- Misinformation: False information
- Violence: Violent content
- Hate Speech: Hateful language

Safety levels:
- SAFE: No concerns, ready to use
- CAUTION: Minor concerns, monitor
- UNSAFE: Significant issues, needs fixing
- BLOCKED: Unacceptable, reject completely

Safety actions:
- Input Blocking: Reject unsafe queries
- Output Filtering: Remove unsafe responses
- Content Rewriting: Fix problematic content
- Warning Messages: Inform users
- Policy Enforcement: Apply guidelines

Benefits:
- Safety: Prevent harmful outputs
- Compliance: Meet regulations
- Trust: Build user confidence
- Flexibility: Configurable policies
- Transparency: Clear safety actions
- Continuous: Always-on protection

Configuration options:
- safety_threshold: Sensitivity level
- enable_rewriting: Auto-fix enabled
- blocked_keywords: Custom filters
- safety_categories: Focus areas
- rewrite_strategy: How to fix

Use Defensive Generation when:
- Public-facing deployment
- Sensitive domains (health, finance)
- Enterprise applications
- Regulated industries
- User-generated content
- High-stakes decisions

Comparison with other patterns:
- vs Guardrails: More comprehensive, includes rewriting
- vs Constitutional AI: Safety-focused vs value alignment
- vs Monitoring: Prevention vs detection
""")


if __name__ == "__main__":
    demonstrate_defensive_generation()
