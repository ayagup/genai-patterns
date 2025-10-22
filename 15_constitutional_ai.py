"""
Constitutional AI Pattern Implementation

This module demonstrates Constitutional AI where an agent follows explicit principles
and rules in its behavior. It includes self-critique, revision based on constitutional
principles, and value alignment mechanisms.

Key Components:
- Constitutional principles and rules
- Self-critique against principles
- Iterative revision and improvement
- Value alignment and ethical reasoning
- Principle-based decision making
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import random
import re


class PrincipleType(Enum):
    """Types of constitutional principles"""
    ETHICAL = "ethical"              # Moral and ethical guidelines
    SAFETY = "safety"               # Safety and harm prevention
    TRUTHFULNESS = "truthfulness"   # Accuracy and honesty
    HELPFULNESS = "helpfulness"     # Being helpful and constructive
    RESPECT = "respect"             # Treating others with respect
    PRIVACY = "privacy"             # Protecting privacy and confidentiality
    FAIRNESS = "fairness"           # Fair and unbiased treatment
    TRANSPARENCY = "transparency"   # Being clear and transparent


class ViolationSeverity(Enum):
    """Severity levels for principle violations"""
    MINOR = "minor"         # Small, easily correctable issues
    MODERATE = "moderate"   # Noticeable problems requiring attention
    MAJOR = "major"         # Serious violations needing significant revision
    CRITICAL = "critical"   # Fundamental violations requiring complete rework


@dataclass
class ConstitutionalPrinciple:
    """Represents a constitutional principle or rule"""
    id: str
    name: str
    description: str
    principle_type: PrincipleType
    priority: int = 1  # 1=low, 5=high
    examples: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"principle_{random.randint(1000, 9999)}"


@dataclass
class Violation:
    """Represents a violation of a constitutional principle"""
    principle_id: str
    principle_name: str
    description: str
    severity: ViolationSeverity
    location: str  # Where in the response the violation occurs
    suggestion: str  # How to fix the violation
    confidence: float = 0.0


@dataclass
class CritiqueResult:
    """Result of critiquing a response against constitutional principles"""
    response_id: str
    violations: List[Violation] = field(default_factory=list)
    overall_score: float = 0.0  # 0-1, higher is better
    needs_revision: bool = False
    critique_summary: str = ""
    
    def get_violations_by_severity(self, severity: ViolationSeverity) -> List[Violation]:
        """Get violations of a specific severity"""
        return [v for v in self.violations if v.severity == severity]
    
    def has_critical_violations(self) -> bool:
        """Check if there are any critical violations"""
        return any(v.severity == ViolationSeverity.CRITICAL for v in self.violations)


class ConstitutionBuilder:
    """Builds and manages constitutional principles"""
    
    def __init__(self):
        self.principles: Dict[str, ConstitutionalPrinciple] = {}
        self._create_default_constitution()
    
    def _create_default_constitution(self):
        """Create a default set of constitutional principles"""
        default_principles = [
            ConstitutionalPrinciple(
                id="ethical_1",
                name="Do No Harm",
                description="Avoid providing information or advice that could cause physical, emotional, or psychological harm to individuals or groups",
                principle_type=PrincipleType.ETHICAL,
                priority=5,
                examples=[
                    "Decline to provide instructions for dangerous activities",
                    "Avoid content that promotes violence or self-harm",
                    "Refuse to help with illegal activities"
                ],
                violations=[
                    "Providing instructions for harmful activities",
                    "Encouraging dangerous behavior",
                    "Assisting with illegal actions"
                ]
            ),
            ConstitutionalPrinciple(
                id="truthfulness_1",
                name="Be Truthful and Accurate",
                description="Provide accurate, fact-based information and acknowledge uncertainty when appropriate",
                principle_type=PrincipleType.TRUTHFULNESS,
                priority=5,
                examples=[
                    "Cite reliable sources when possible",
                    "Admit when information is uncertain",
                    "Correct misinformation when identified"
                ],
                violations=[
                    "Stating false information as fact",
                    "Making up non-existent sources",
                    "Claiming certainty about uncertain topics"
                ]
            ),
            ConstitutionalPrinciple(
                id="respect_1",
                name="Treat All People with Respect",
                description="Show respect for all individuals regardless of their background, beliefs, or characteristics",
                principle_type=PrincipleType.RESPECT,
                priority=4,
                examples=[
                    "Use respectful language for all groups",
                    "Avoid stereotypes and generalizations",
                    "Acknowledge diverse perspectives"
                ],
                violations=[
                    "Using derogatory language about groups",
                    "Promoting stereotypes",
                    "Dismissing valid perspectives"
                ]
            ),
            ConstitutionalPrinciple(
                id="helpfulness_1",
                name="Be Helpful and Constructive",
                description="Provide useful, constructive assistance that genuinely helps users achieve their goals",
                principle_type=PrincipleType.HELPFULNESS,
                priority=4,
                examples=[
                    "Offer practical solutions to problems",
                    "Provide step-by-step guidance when appropriate",
                    "Suggest alternatives when direct help isn't possible"
                ],
                violations=[
                    "Refusing to help with legitimate requests",
                    "Providing unhelpful or vague responses",
                    "Being dismissive of user needs"
                ]
            ),
            ConstitutionalPrinciple(
                id="fairness_1",
                name="Be Fair and Unbiased",
                description="Treat all perspectives and groups fairly, avoiding unfair bias or discrimination",
                principle_type=PrincipleType.FAIRNESS,
                priority=4,
                examples=[
                    "Present multiple viewpoints on controversial topics",
                    "Avoid showing preference for specific groups",
                    "Base recommendations on merit, not bias"
                ],
                violations=[
                    "Showing clear bias toward specific groups",
                    "Presenting only one side of controversial issues",
                    "Making unfair generalizations"
                ]
            ),
            ConstitutionalPrinciple(
                id="privacy_1",
                name="Respect Privacy",
                description="Protect personal information and respect privacy boundaries",
                principle_type=PrincipleType.PRIVACY,
                priority=3,
                examples=[
                    "Don't request unnecessary personal information",
                    "Respect confidentiality when appropriate",
                    "Advise on privacy protection"
                ],
                violations=[
                    "Requesting sensitive personal information",
                    "Sharing private information",
                    "Ignoring privacy concerns"
                ]
            ),
            ConstitutionalPrinciple(
                id="transparency_1",
                name="Be Transparent About Limitations",
                description="Be clear about capabilities, limitations, and the nature of AI assistance",
                principle_type=PrincipleType.TRANSPARENCY,
                priority=3,
                examples=[
                    "Acknowledge when uncertain about information",
                    "Explain limitations of AI advice",
                    "Be clear about the nature of AI responses"
                ],
                violations=[
                    "Claiming capabilities beyond actual abilities",
                    "Hiding uncertainty or limitations",
                    "Pretending to be human or have human experiences"
                ]
            )
        ]
        
        for principle in default_principles:
            self.principles[principle.id] = principle
    
    def add_principle(self, principle: ConstitutionalPrinciple):
        """Add a new constitutional principle"""
        self.principles[principle.id] = principle
    
    def get_principles_by_type(self, principle_type: PrincipleType) -> List[ConstitutionalPrinciple]:
        """Get all principles of a specific type"""
        return [p for p in self.principles.values() if p.principle_type == principle_type]
    
    def get_high_priority_principles(self) -> List[ConstitutionalPrinciple]:
        """Get principles with high priority (4-5)"""
        return [p for p in self.principles.values() if p.priority >= 4]


class ConstitutionalCritic:
    """Critiques responses against constitutional principles"""
    
    def __init__(self, constitution: ConstitutionBuilder):
        self.constitution = constitution
        self.violation_patterns = self._create_violation_patterns()
    
    def _create_violation_patterns(self) -> Dict[str, List[Tuple[str, ViolationSeverity]]]:
        """Create patterns for detecting principle violations"""
        patterns = {
            "ethical_1": [
                (r"how to (harm|hurt|injure)", ViolationSeverity.CRITICAL),
                (r"instructions for (violence|dangerous)", ViolationSeverity.CRITICAL),
                (r"(illegal|unlawful) (activities|actions)", ViolationSeverity.MAJOR),
                (r"(risky|unsafe) behavior", ViolationSeverity.MODERATE)
            ],
            "truthfulness_1": [
                (r"(definitely|certainly) (true|false) that", ViolationSeverity.MAJOR),
                (r"according to (fake|made-up) source", ViolationSeverity.CRITICAL),
                (r"I'm (completely sure|100% certain)", ViolationSeverity.MODERATE),
                (r"this is (absolutely|definitely) correct", ViolationSeverity.MODERATE)
            ],
            "respect_1": [
                (r"(all|most) (.*) are (stupid|inferior|bad)", ViolationSeverity.MAJOR),
                (r"(derogatory|offensive) terms", ViolationSeverity.MAJOR),
                (r"(stereotype|generalization) about", ViolationSeverity.MODERATE),
                (r"(dismissive|contemptuous) of", ViolationSeverity.MODERATE)
            ],
            "helpfulness_1": [
                (r"I (can't|won't) help with that", ViolationSeverity.MINOR),
                (r"(vague|unhelpful) response", ViolationSeverity.MODERATE),
                (r"figure it out yourself", ViolationSeverity.MODERATE),
                (r"not my problem", ViolationSeverity.MAJOR)
            ],
            "fairness_1": [
                (r"(clearly|obviously) better than", ViolationSeverity.MODERATE),
                (r"(biased|unfair) preference for", ViolationSeverity.MAJOR),
                (r"only (one side|perspective)", ViolationSeverity.MODERATE),
                (r"(discriminatory|prejudiced)", ViolationSeverity.MAJOR)
            ],
            "privacy_1": [
                (r"what is your (address|phone|ssn)", ViolationSeverity.MAJOR),
                (r"share (personal|private) information", ViolationSeverity.MAJOR),
                (r"tell me about your (personal|private)", ViolationSeverity.MODERATE)
            ],
            "transparency_1": [
                (r"I (am|have) human (experience|emotions)", ViolationSeverity.MAJOR),
                (r"I can (see|feel|touch)", ViolationSeverity.MAJOR),
                (r"I'm (completely|totally) certain", ViolationSeverity.MODERATE),
                (r"I have (personal|real-world) experience", ViolationSeverity.MAJOR)
            ]
        }
        return patterns
    
    def critique_response(self, response: str, response_id: Optional[str] = None) -> CritiqueResult:
        """Critique a response against all constitutional principles"""
        if response_id is None:
            response_id = f"response_{random.randint(1000, 9999)}"
        
        violations = []
        
        # Check each principle
        for principle_id, principle in self.constitution.principles.items():
            principle_violations = self._check_principle_violations(
                response, principle_id, principle
            )
            violations.extend(principle_violations)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(violations)
        
        # Determine if revision is needed
        needs_revision = (
            overall_score < 0.7 or
            any(v.severity in [ViolationSeverity.CRITICAL, ViolationSeverity.MAJOR] for v in violations)
        )
        
        # Generate critique summary
        critique_summary = self._generate_critique_summary(violations, overall_score)
        
        return CritiqueResult(
            response_id=response_id,
            violations=violations,
            overall_score=overall_score,
            needs_revision=needs_revision,
            critique_summary=critique_summary
        )
    
    def _check_principle_violations(self, response: str, principle_id: str, 
                                  principle: ConstitutionalPrinciple) -> List[Violation]:
        """Check for violations of a specific principle"""
        violations = []
        response_lower = response.lower()
        
        # Check pattern-based violations
        if principle_id in self.violation_patterns:
            for pattern, severity in self.violation_patterns[principle_id]:
                matches = re.finditer(pattern, response_lower)
                for match in matches:
                    violation = Violation(
                        principle_id=principle_id,
                        principle_name=principle.name,
                        description=f"Potential violation: '{match.group()}' may violate {principle.name}",
                        severity=severity,
                        location=f"Characters {match.start()}-{match.end()}",
                        suggestion=self._get_violation_suggestion(principle_id, pattern),
                        confidence=0.7 + random.uniform(-0.2, 0.2)
                    )
                    violations.append(violation)
        
        # Check against known violation examples
        for violation_example in principle.violations:
            if violation_example.lower() in response_lower:
                violation = Violation(
                    principle_id=principle_id,
                    principle_name=principle.name,
                    description=f"Response contains known violation: {violation_example}",
                    severity=ViolationSeverity.MAJOR,
                    location="Content analysis",
                    suggestion=f"Remove or revise content related to: {violation_example}",
                    confidence=0.8 + random.uniform(-0.1, 0.1)
                )
                violations.append(violation)
        
        return violations
    
    def _get_violation_suggestion(self, principle_id: str, pattern: str) -> str:
        """Get suggestions for fixing specific violation patterns"""
        suggestions = {
            "ethical_1": "Remove harmful content and focus on safe, constructive alternatives",
            "truthfulness_1": "Add appropriate uncertainty qualifiers and verify facts",
            "respect_1": "Use respectful language and avoid generalizations about groups",
            "helpfulness_1": "Provide constructive assistance or explain limitations politely",
            "fairness_1": "Present balanced perspectives and avoid showing bias",
            "privacy_1": "Respect privacy boundaries and avoid requesting personal information",
            "transparency_1": "Be clear about AI limitations and avoid claiming human attributes"
        }
        return suggestions.get(principle_id, "Review and revise to align with constitutional principles")
    
    def _calculate_overall_score(self, violations: List[Violation]) -> float:
        """Calculate overall constitutional compliance score"""
        if not violations:
            return 1.0
        
        # Weight violations by severity
        severity_weights = {
            ViolationSeverity.MINOR: 0.1,
            ViolationSeverity.MODERATE: 0.3,
            ViolationSeverity.MAJOR: 0.6,
            ViolationSeverity.CRITICAL: 1.0
        }
        
        total_penalty = sum(severity_weights[v.severity] for v in violations)
        max_penalty = len(violations)  # If all were critical
        
        # Calculate score (higher is better)
        score = max(0.0, 1.0 - (total_penalty / max(max_penalty, 1)))
        return score
    
    def _generate_critique_summary(self, violations: List[Violation], overall_score: float) -> str:
        """Generate a summary of the constitutional critique"""
        if not violations:
            return "✅ Response fully complies with constitutional principles"
        
        summary_parts = []
        summary_parts.append(f"📊 Constitutional Compliance Score: {overall_score:.1%}")
        
        # Count violations by severity
        severity_counts = {}
        for v in violations:
            severity_counts[v.severity] = severity_counts.get(v.severity, 0) + 1
        
        if severity_counts:
            summary_parts.append("🚨 Violations found:")
            for severity, count in severity_counts.items():
                emoji = {"critical": "🔴", "major": "🟠", "moderate": "🟡", "minor": "🟢"}
                summary_parts.append(f"   {emoji.get(severity.value, '⚪')} {severity.value.title()}: {count}")
        
        # Top recommendations
        if violations:
            summary_parts.append("💡 Key improvements needed:")
            for v in violations[:3]:  # Top 3 most important
                summary_parts.append(f"   • {v.suggestion}")
        
        return "\n".join(summary_parts)


class ConstitutionalReviser:
    """Revises responses to align with constitutional principles"""
    
    def __init__(self):
        self.revision_strategies = {
            ViolationSeverity.MINOR: self._minor_revision,
            ViolationSeverity.MODERATE: self._moderate_revision,
            ViolationSeverity.MAJOR: self._major_revision,
            ViolationSeverity.CRITICAL: self._critical_revision
        }
    
    def revise_response(self, original_response: str, critique: CritiqueResult) -> Tuple[str, str]:
        """Revise a response based on constitutional critique"""
        if not critique.violations:
            return original_response, "No revisions needed - response is constitutional"
        
        revised_response = original_response
        revision_notes = []
        
        # Sort violations by severity (handle critical first)
        sorted_violations = sorted(critique.violations, 
                                 key=lambda v: list(ViolationSeverity).index(v.severity), 
                                 reverse=True)
        
        for violation in sorted_violations:
            if violation.severity in self.revision_strategies:
                revised_response, note = self.revision_strategies[violation.severity](
                    revised_response, violation
                )
                revision_notes.append(note)
        
        revision_summary = "\n".join([f"• {note}" for note in revision_notes])
        return revised_response, revision_summary
    
    def _critical_revision(self, response: str, violation: Violation) -> Tuple[str, str]:
        """Handle critical violations with complete rework"""
        # For critical violations, we need to completely reframe the response
        revised = "I understand you're looking for information, but I'm not able to provide content that could potentially cause harm. Instead, let me help you find safe and constructive alternatives."
        
        if "illegal" in violation.description.lower():
            revised += " If you're interested in this topic for educational purposes, I can suggest legitimate resources for learning about law and ethics."
        elif "harm" in violation.description.lower():
            revised += " If you're facing difficulties, I'd be happy to help you find appropriate support resources."
        
        return revised, f"Critical revision: Completely reframed response to address {violation.principle_name}"
    
    def _major_revision(self, response: str, violation: Violation) -> Tuple[str, str]:
        """Handle major violations with significant changes"""
        # Remove problematic content and add safeguards
        revised = response
        
        if "truthfulness" in violation.principle_id:
            revised = re.sub(r"(definitely|certainly|100%)\s+", "likely ", revised, flags=re.IGNORECASE)
            revised += "\n\nNote: This information should be verified from authoritative sources."
        
        elif "respect" in violation.principle_id:
            # Remove generalizations and add respectful framing
            revised = re.sub(r"(all|most)\s+(\w+)\s+are", r"some \2 may be", revised, flags=re.IGNORECASE)
            revised += "\n\nIt's important to recognize the diversity within any group."
        
        elif "fairness" in violation.principle_id:
            revised += "\n\nThere are multiple perspectives on this topic, and different viewpoints should be considered."
        
        return revised, f"Major revision: Addressed {violation.principle_name} concerns"
    
    def _moderate_revision(self, response: str, violation: Violation) -> Tuple[str, str]:
        """Handle moderate violations with targeted fixes"""
        revised = response
        
        if "uncertainty" in violation.description.lower():
            revised = re.sub(r"I'm (sure|certain)", "I believe", revised, flags=re.IGNORECASE)
            revised = re.sub(r"This is (definitely|certainly)", "This appears to be", revised, flags=re.IGNORECASE)
        
        if "transparency" in violation.principle_id:
            revised += "\n\nAs an AI, I should note that my responses are based on training data and may have limitations."
        
        return revised, f"Moderate revision: Improved {violation.principle_name} compliance"
    
    def _minor_revision(self, response: str, violation: Violation) -> Tuple[str, str]:
        """Handle minor violations with small adjustments"""
        revised = response
        
        # Add polite qualifiers
        if "helpfulness" in violation.principle_id:
            revised = re.sub(r"I can't help", "I'm not able to help directly, but", revised, flags=re.IGNORECASE)
        
        return revised, f"Minor revision: Small adjustment for {violation.principle_name}"


class ConstitutionalAI:
    """Main Constitutional AI agent"""
    
    def __init__(self):
        self.constitution = ConstitutionBuilder()
        self.critic = ConstitutionalCritic(self.constitution)
        self.reviser = ConstitutionalReviser()
        self.revision_history: List[Dict[str, Any]] = []
    
    def generate_constitutional_response(self, prompt: str, max_revisions: int = 3) -> Dict[str, Any]:
        """Generate a response that complies with constitutional principles"""
        print(f"\n⚖️ Constitutional AI Response Generation")
        print("=" * 60)
        print(f"Prompt: {prompt}")
        
        # Generate initial response (simulated)
        initial_response = self._generate_initial_response(prompt)
        print(f"\n📝 Initial Response:")
        print(initial_response)
        
        current_response = initial_response
        revision_count = 0
        revision_log = []
        
        while revision_count < max_revisions:
            print(f"\n🔍 Constitutional Review #{revision_count + 1}")
            print("-" * 40)
            
            # Critique current response
            critique = self.critic.critique_response(current_response, f"v{revision_count}")
            
            print(critique.critique_summary)
            
            if not critique.needs_revision:
                print("✅ Response meets constitutional standards")
                break
            
            print(f"\n🔧 Revision needed (Score: {critique.overall_score:.1%})")
            
            # Revise response
            revised_response, revision_notes = self.reviser.revise_response(current_response, critique)
            
            revision_log.append({
                "version": revision_count,
                "critique_score": critique.overall_score,
                "violations": len(critique.violations),
                "revision_notes": revision_notes
            })
            
            print(f"Revision notes:\n{revision_notes}")
            
            current_response = revised_response
            revision_count += 1
        
        # Final critique
        final_critique = self.critic.critique_response(current_response, "final")
        
        result = {
            "prompt": prompt,
            "initial_response": initial_response,
            "final_response": current_response,
            "revisions_made": revision_count,
            "final_score": final_critique.overall_score,
            "final_violations": len(final_critique.violations),
            "revision_log": revision_log,
            "constitutional_compliance": final_critique.overall_score >= 0.8,
            "critique_details": final_critique
        }
        
        self.revision_history.append(result)
        
        print(f"\n✅ Final Response (Score: {final_critique.overall_score:.1%}):")
        print(current_response)
        
        return result
    
    def _generate_initial_response(self, prompt: str) -> str:
        """Simulate generating an initial response that may have constitutional issues"""
        # Simulate different types of potentially problematic responses
        problematic_responses = {
            "harmful": "Here's how to create dangerous substances: First, you need to...",
            "biased": "All people from that group are generally known to be less intelligent and...",
            "overconfident": "I am 100% certain that this is definitely true and there's no doubt about it...",
            "privacy": "Please provide your social security number and home address so I can help...",
            "dishonest": "According to the completely reliable source I just made up, the facts are...",
            "unhelpful": "I can't help with that. Figure it out yourself. It's not my problem.",
            "unfair": "This political party is clearly superior and anyone who disagrees is wrong..."
        }
        
        # For demonstration, introduce different types of issues based on prompt content
        if any(word in prompt.lower() for word in ["dangerous", "harmful", "illegal"]):
            base_response = problematic_responses["harmful"]
        elif any(word in prompt.lower() for word in ["group", "people", "demographic"]):
            base_response = problematic_responses["biased"]
        elif any(word in prompt.lower() for word in ["fact", "true", "certain"]):
            base_response = problematic_responses["overconfident"]
        elif any(word in prompt.lower() for word in ["personal", "private"]):
            base_response = problematic_responses["privacy"]
        else:
            # Generate a mixed response with some issues
            base_response = f"I definitely know the answer to '{prompt}'. All experts agree that this is 100% certain. People who disagree are usually wrong about these things."
        
        return base_response
    
    def add_custom_principle(self, principle: ConstitutionalPrinciple):
        """Add a custom constitutional principle"""
        self.constitution.add_principle(principle)
        # Update the critic with new patterns if needed
        self.critic = ConstitutionalCritic(self.constitution)
    
    def get_constitution_summary(self) -> Dict[str, Any]:
        """Get summary of the current constitution"""
        principles_by_type = {}
        for principle in self.constitution.principles.values():
            type_name = principle.principle_type.value
            if type_name not in principles_by_type:
                principles_by_type[type_name] = []
            principles_by_type[type_name].append({
                "name": principle.name,
                "priority": principle.priority,
                "description": principle.description[:100] + "..." if len(principle.description) > 100 else principle.description
            })
        
        return {
            "total_principles": len(self.constitution.principles),
            "principles_by_type": principles_by_type,
            "high_priority_count": len(self.constitution.get_high_priority_principles()),
            "revision_sessions": len(self.revision_history)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of constitutional AI performance"""
        if not self.revision_history:
            return {"message": "No constitutional responses generated yet"}
        
        total_sessions = len(self.revision_history)
        average_revisions = sum(r["revisions_made"] for r in self.revision_history) / total_sessions
        average_final_score = sum(r["final_score"] for r in self.revision_history) / total_sessions
        compliance_rate = len([r for r in self.revision_history if r["constitutional_compliance"]]) / total_sessions
        
        return {
            "total_sessions": total_sessions,
            "average_revisions_needed": average_revisions,
            "average_final_score": average_final_score,
            "constitutional_compliance_rate": compliance_rate,
            "improvement_trend": self._calculate_improvement_trend()
        }
    
    def _calculate_improvement_trend(self) -> str:
        """Calculate if performance is improving over time"""
        if len(self.revision_history) < 3:
            return "insufficient_data"
        
        recent_scores = [r["final_score"] for r in self.revision_history[-5:]]
        earlier_scores = [r["final_score"] for r in self.revision_history[-10:-5]]
        
        if not earlier_scores:
            return "insufficient_data"
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        earlier_avg = sum(earlier_scores) / len(earlier_scores)
        
        if recent_avg > earlier_avg + 0.05:
            return "improving"
        elif recent_avg < earlier_avg - 0.05:
            return "declining"
        else:
            return "stable"


def main():
    """Demonstration of the Constitutional AI pattern"""
    print("⚖️ Constitutional AI Pattern Demonstration")
    print("=" * 80)
    print("This demonstrates AI behavior guided by constitutional principles:")
    print("- Explicit ethical and behavioral principles")
    print("- Self-critique against constitutional standards")
    print("- Iterative revision for compliance")
    print("- Value alignment and transparency")
    
    # Create Constitutional AI agent
    agent = ConstitutionalAI()
    
    # Show constitution summary
    print(f"\n📜 Constitutional Framework:")
    constitution_summary = agent.get_constitution_summary()
    print(f"Total principles: {constitution_summary['total_principles']}")
    print(f"High priority principles: {constitution_summary['high_priority_count']}")
    
    print(f"\n📋 Principles by Type:")
    for principle_type, principles in constitution_summary["principles_by_type"].items():
        print(f"  {principle_type.title()}:")
        for p in principles[:2]:  # Show first 2 of each type
            print(f"    • {p['name']} (Priority: {p['priority']})")
        if len(principles) > 2:
            print(f"    ... and {len(principles) - 2} more")
    
    # Test prompts with different potential issues
    test_prompts = [
        "How can I create dangerous explosives at home?",
        "What do you think about people from different cultural backgrounds?",
        "Is climate change real? Give me a definitive answer.",
        "Can you help me write a fair analysis of different political viewpoints?",
        "I need help with a programming problem. Can you provide some guidance?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n\n🔍 Test Case {i}")
        print("=" * 80)
        
        result = agent.generate_constitutional_response(prompt, max_revisions=3)
        
        print(f"\n📊 Constitutional Response Results:")
        print(f"Revisions needed: {result['revisions_made']}")
        print(f"Final compliance score: {result['final_score']:.1%}")
        print(f"Constitutional compliance: {result['constitutional_compliance']}")
        print(f"Final violations: {result['final_violations']}")
        
        if result['revision_log']:
            print(f"\n📝 Revision Progress:")
            for log_entry in result['revision_log']:
                print(f"  Version {log_entry['version']}: Score {log_entry['critique_score']:.1%}, {log_entry['violations']} violations")
    
    # Performance summary
    print(f"\n\n📈 Constitutional AI Performance Summary")
    print("=" * 80)
    
    performance = agent.get_performance_summary()
    for key, value in performance.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    print("\n\n🎯 Key Constitutional AI Features Demonstrated:")
    print("✅ Explicit constitutional principles and rules")
    print("✅ Automated constitutional compliance checking")
    print("✅ Iterative self-revision based on principles")
    print("✅ Multiple violation severity levels")
    print("✅ Transparent critique and revision process")
    print("✅ Performance tracking and improvement")
    print("✅ Customizable principle frameworks")
    print("✅ Value alignment through constitutional constraints")


if __name__ == "__main__":
    main()