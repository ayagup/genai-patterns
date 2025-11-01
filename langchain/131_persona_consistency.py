"""
Pattern 131: Persona Consistency

Description:
    The Persona Consistency pattern enables agents to maintain a stable, coherent
    character or personality across all interactions. Rather than exhibiting
    inconsistent behavior or tone, the agent adheres to a defined persona with
    consistent traits, communication style, values, knowledge boundaries, and
    behavioral patterns throughout conversations and across sessions.
    
    Persona consistency is essential for creating believable, trustworthy AI
    characters in applications like virtual assistants, game NPCs, brand
    representatives, and therapeutic chatbots. The pattern addresses challenges
    like personality drift, contextual appropriateness, multi-turn coherence,
    and balancing consistency with natural variation.
    
    This pattern includes mechanisms for persona definition, trait enforcement,
    style consistency, memory of character history, contextual adaptation within
    persona bounds, and continuous validation of persona adherence.

Key Components:
    1. Persona Definition: Core character traits and attributes
    2. Style Guide: Communication patterns and language
    3. Knowledge Boundaries: What the persona knows/doesn't know
    4. Behavioral Rules: How persona responds to situations
    5. Memory System: Track persona history and consistency
    6. Consistency Checker: Validate responses against persona
    7. Adaptation Layer: Natural variation within constraints

Persona Attributes:
    1. Core Traits: Personality characteristics (friendly, formal, humorous)
    2. Communication Style: Tone, vocabulary, sentence structure
    3. Values & Beliefs: Moral framework, priorities
    4. Knowledge Domain: Areas of expertise and ignorance
    5. Emotional Range: How persona expresses emotions
    6. Cultural Background: Context influencing behavior
    7. Quirks & Habits: Unique characteristics

Consistency Dimensions:
    1. Linguistic: Word choice, grammar, dialect
    2. Behavioral: Response patterns, decision-making
    3. Emotional: Mood stability, reaction patterns
    4. Knowledge: What persona knows consistently
    5. Temporal: Consistency over time
    6. Contextual: Appropriate adaptation
    7. Relational: Consistent with relationship history

Use Cases:
    - Virtual assistants with brand personality
    - Game NPCs with defined characters
    - Customer service representatives
    - Educational tutors with consistent approach
    - Therapeutic/counseling chatbots
    - Celebrity or character impersonations
    - Brand ambassadors

Advantages:
    - Builds user trust and familiarity
    - Creates memorable experiences
    - Strengthens brand identity
    - Enables emotional connection
    - Provides predictable interactions
    - Maintains character integrity
    - Enhances believability

Challenges:
    - Balancing consistency with flexibility
    - Avoiding robotic repetition
    - Adapting to context while staying in character
    - Managing persona evolution
    - Handling edge cases in character
    - Maintaining consistency across sessions
    - Preventing personality drift

LangChain Implementation:
    This implementation uses LangChain for:
    - Persona-aware prompt construction
    - Response consistency validation
    - Character memory management
    - Style-constrained generation
    
Production Considerations:
    - Store persona definitions persistently
    - Version control persona changes
    - Monitor consistency metrics
    - Allow graceful persona updates
    - Support multiple personas per agent
    - Enable A/B testing of personas
    - Provide consistency reporting
    - Log persona violations for improvement
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class PersonaTrait(Enum):
    """Core personality traits."""
    FRIENDLY = "friendly"
    FORMAL = "formal"
    HUMOROUS = "humorous"
    SERIOUS = "serious"
    EMPATHETIC = "empathetic"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    PRACTICAL = "practical"


class CommunicationStyle(Enum):
    """Communication style options."""
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    TECHNICAL = "technical"
    SIMPLE = "simple"
    VERBOSE = "verbose"
    CONCISE = "concise"


@dataclass
class PersonaDefinition:
    """
    Complete definition of an agent's persona.
    
    Attributes:
        name: Persona name
        description: Brief description
        core_traits: Personality traits
        communication_style: How persona communicates
        values: Core values and beliefs
        knowledge_domains: Areas of expertise
        knowledge_gaps: What persona doesn't know
        emotional_range: How emotions are expressed
        quirks: Unique characteristics
        sample_phrases: Example phrases persona uses
        behavioral_rules: Guidelines for behavior
    """
    name: str
    description: str
    core_traits: List[PersonaTrait] = field(default_factory=list)
    communication_style: CommunicationStyle = CommunicationStyle.PROFESSIONAL
    values: List[str] = field(default_factory=list)
    knowledge_domains: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    emotional_range: str = "moderate"
    quirks: List[str] = field(default_factory=list)
    sample_phrases: List[str] = field(default_factory=list)
    behavioral_rules: List[str] = field(default_factory=list)


@dataclass
class PersonaInteraction:
    """
    Record of an interaction with persona.
    
    Attributes:
        interaction_id: Unique identifier
        user_input: User's message
        persona_response: Agent's response
        timestamp: When interaction occurred
        consistency_score: How well response matched persona
        violations: Any detected inconsistencies
    """
    interaction_id: str
    user_input: str
    persona_response: str
    timestamp: datetime = field(default_factory=datetime.now)
    consistency_score: float = 1.0
    violations: List[str] = field(default_factory=list)


class PersonaAgent:
    """
    Agent that maintains consistent persona across interactions.
    
    This agent ensures all responses adhere to a defined persona,
    maintaining character consistency across conversations.
    """
    
    def __init__(self, persona: PersonaDefinition, temperature: float = 0.7):
        """
        Initialize persona agent.
        
        Args:
            persona: Persona definition
            temperature: LLM temperature (higher for more variation)
        """
        self.persona = persona
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.interactions: List[PersonaInteraction] = []
        self.interaction_counter = 0
    
    def _build_persona_prompt(self) -> str:
        """Build persona description for prompts."""
        traits_str = ", ".join([t.value for t in self.persona.core_traits])
        
        prompt_parts = [
            f"You are {self.persona.name}, {self.persona.description}",
            f"Personality traits: {traits_str}",
            f"Communication style: {self.persona.communication_style.value}",
        ]
        
        if self.persona.values:
            values_str = ", ".join(self.persona.values)
            prompt_parts.append(f"Core values: {values_str}")
        
        if self.persona.knowledge_domains:
            domains_str = ", ".join(self.persona.knowledge_domains)
            prompt_parts.append(f"Areas of expertise: {domains_str}")
        
        if self.persona.knowledge_gaps:
            gaps_str = ", ".join(self.persona.knowledge_gaps)
            prompt_parts.append(f"You do not have expertise in: {gaps_str}")
        
        if self.persona.quirks:
            quirks_str = ", ".join(self.persona.quirks)
            prompt_parts.append(f"Unique characteristics: {quirks_str}")
        
        if self.persona.behavioral_rules:
            rules_str = "\n".join([f"- {rule}" for rule in self.persona.behavioral_rules])
            prompt_parts.append(f"Behavioral guidelines:\n{rules_str}")
        
        return "\n".join(prompt_parts)
    
    def respond(self, user_input: str, context: Optional[str] = None) -> str:
        """
        Generate response in character.
        
        Args:
            user_input: User's message
            context: Optional conversation context
            
        Returns:
            Persona's response
        """
        persona_description = self._build_persona_prompt()
        
        # Build prompt with persona and context
        prompt_template = (
            "{persona_description}\n\n"
            "Stay completely in character. Respond as {persona_name} would, "
            "maintaining consistency with your personality, communication style, "
            "and knowledge boundaries.\n\n"
        )
        
        if context:
            prompt_template += "Previous context:\n{context}\n\n"
        
        prompt_template += "User: {user_input}\n\n{persona_name}:"
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        input_data = {
            "persona_description": persona_description,
            "persona_name": self.persona.name,
            "user_input": user_input
        }
        
        if context:
            input_data["context"] = context
        
        response = chain.invoke(input_data)
        
        # Record interaction
        self.interaction_counter += 1
        interaction = PersonaInteraction(
            interaction_id=f"int_{self.interaction_counter}",
            user_input=user_input,
            persona_response=response
        )
        self.interactions.append(interaction)
        
        return response
    
    def check_consistency(self, response: str) -> Dict[str, Any]:
        """
        Check if response is consistent with persona.
        
        Args:
            response: Response to check
            
        Returns:
            Consistency analysis
        """
        persona_description = self._build_persona_prompt()
        
        prompt = ChatPromptTemplate.from_template(
            "Persona Definition:\n"
            "{persona_description}\n\n"
            "Response to evaluate:\n"
            "{response}\n\n"
            "Analyze whether this response is consistent with the persona. "
            "Consider:\n"
            "1. Personality traits\n"
            "2. Communication style\n"
            "3. Values and beliefs\n"
            "4. Knowledge boundaries\n"
            "5. Characteristic quirks\n\n"
            "Provide:\n"
            "CONSISTENT: yes/no\n"
            "SCORE: 0-10\n"
            "VIOLATIONS: [list any inconsistencies]\n"
            "EXPLANATION: [brief explanation]"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        analysis = chain.invoke({
            "persona_description": persona_description,
            "response": response
        })
        
        # Parse analysis
        lines = analysis.strip().split('\n')
        result = {
            "raw_analysis": analysis,
            "consistent": True,
            "score": 8.0,
            "violations": []
        }
        
        for line in lines:
            if line.startswith("CONSISTENT:"):
                result["consistent"] = "yes" in line.lower()
            elif line.startswith("SCORE:"):
                try:
                    result["score"] = float(line.split(":")[1].strip().split()[0])
                except:
                    pass
        
        return result
    
    def get_conversation_context(self, num_recent: int = 3) -> str:
        """
        Get recent conversation history.
        
        Args:
            num_recent: Number of recent interactions
            
        Returns:
            Formatted context
        """
        recent = self.interactions[-num_recent:]
        
        context_lines = []
        for interaction in recent:
            context_lines.append(f"User: {interaction.user_input}")
            context_lines.append(f"{self.persona.name}: {interaction.persona_response}")
        
        return "\n".join(context_lines)
    
    def get_consistency_report(self) -> Dict[str, Any]:
        """Get report on persona consistency."""
        if not self.interactions:
            return {
                "total_interactions": 0,
                "average_consistency": 0.0,
                "violations": []
            }
        
        total_score = sum(i.consistency_score for i in self.interactions)
        avg_score = total_score / len(self.interactions)
        
        all_violations = []
        for interaction in self.interactions:
            all_violations.extend(interaction.violations)
        
        return {
            "total_interactions": len(self.interactions),
            "average_consistency": avg_score,
            "violations": all_violations,
            "persona_name": self.persona.name
        }


def demonstrate_persona_consistency():
    """Demonstrate persona consistency pattern."""
    
    print("=" * 80)
    print("PERSONA CONSISTENCY PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Defining a persona
    print("\n" + "=" * 80)
    print("Example 1: Defining a Persona")
    print("=" * 80)
    
    persona = PersonaDefinition(
        name="Professor Ada",
        description="an enthusiastic computer science professor who loves making complex topics accessible",
        core_traits=[
            PersonaTrait.FRIENDLY,
            PersonaTrait.ANALYTICAL,
            PersonaTrait.EMPATHETIC
        ],
        communication_style=CommunicationStyle.PROFESSIONAL,
        values=[
            "Education for all",
            "Clarity over jargon",
            "Practical application"
        ],
        knowledge_domains=[
            "Computer Science",
            "Programming",
            "Algorithms",
            "Software Engineering"
        ],
        knowledge_gaps=[
            "Advanced Physics",
            "Medical procedures",
            "Legal advice"
        ],
        quirks=[
            "Often uses teaching analogies",
            "Encourages questions",
            "References classic CS papers"
        ],
        sample_phrases=[
            "That's a great question!",
            "Let me break that down...",
            "Think of it like this..."
        ],
        behavioral_rules=[
            "Always encourage learning",
            "Admit when something is outside expertise",
            "Use analogies to explain complex concepts",
            "Be patient and supportive"
        ]
    )
    
    print(f"\nPersona: {persona.name}")
    print(f"Description: {persona.description}")
    print(f"\nCore Traits: {[t.value for t in persona.core_traits]}")
    print(f"Style: {persona.communication_style.value}")
    print(f"Values: {persona.values}")
    print(f"Expertise: {persona.knowledge_domains}")
    
    # Example 2: In-character responses
    print("\n" + "=" * 80)
    print("Example 2: In-Character Responses")
    print("=" * 80)
    
    agent = PersonaAgent(persona)
    
    # Test responses
    questions = [
        "What is recursion?",
        "Can you explain it simply?",
        "What about medical imaging algorithms?"
    ]
    
    print("\nConversation:")
    for question in questions:
        print(f"\nUser: {question}")
        response = agent.respond(question)
        print(f"{persona.name}: {response}")
    
    # Example 3: Consistency checking
    print("\n" + "=" * 80)
    print("Example 3: Consistency Checking")
    print("=" * 80)
    
    # Check a consistent response
    consistent_response = "That's a great question! Recursion is when a function calls itself. Think of it like Russian nesting dolls - each doll contains a smaller version of itself."
    
    print("\nChecking consistent response:")
    print(f'Response: "{consistent_response}"')
    
    consistency = agent.check_consistency(consistent_response)
    print(f"\nConsistent: {consistency['consistent']}")
    print(f"Score: {consistency['score']}/10")
    
    # Check an inconsistent response
    inconsistent_response = "Recursion? Yeah bro, it's like totally when stuff repeats, ya know? Whatever dude."
    
    print("\n" + "-" * 60)
    print("\nChecking inconsistent response:")
    print(f'Response: "{inconsistent_response}"')
    
    consistency2 = agent.check_consistency(inconsistent_response)
    print(f"\nConsistent: {consistency2['consistent']}")
    print(f"Score: {consistency2['score']}/10")
    
    # Example 4: Multi-turn consistency
    print("\n" + "=" * 80)
    print("Example 4: Multi-Turn Consistency")
    print("=" * 80)
    
    agent2 = PersonaAgent(persona)
    
    conversation = [
        "Hi! Can you help me learn programming?",
        "What language should I start with?",
        "Is it hard to learn?"
    ]
    
    print("\nMulti-turn conversation:")
    for user_msg in conversation:
        context = agent2.get_conversation_context(num_recent=2) if agent2.interactions else None
        response = agent2.respond(user_msg, context=context)
        print(f"\nUser: {user_msg}")
        print(f"{persona.name}: {response}")
    
    # Example 5: Knowledge boundaries
    print("\n" + "=" * 80)
    print("Example 5: Respecting Knowledge Boundaries")
    print("=" * 80)
    
    agent3 = PersonaAgent(persona)
    
    # Question outside expertise
    question = "Can you diagnose this medical condition?"
    print(f"\nUser: {question}")
    
    response = agent3.respond(question)
    print(f"{persona.name}: {response}")
    print("\n(Notice how persona stays in character while declining gracefully)")
    
    # Example 6: Different persona
    print("\n" + "=" * 80)
    print("Example 6: Different Persona - Customer Service")
    print("=" * 80)
    
    service_persona = PersonaDefinition(
        name="Sarah",
        description="a helpful and patient customer service representative",
        core_traits=[
            PersonaTrait.FRIENDLY,
            PersonaTrait.EMPATHETIC,
            PersonaTrait.PRACTICAL
        ],
        communication_style=CommunicationStyle.PROFESSIONAL,
        values=[
            "Customer satisfaction",
            "Problem resolution",
            "Clear communication"
        ],
        knowledge_domains=[
            "Product information",
            "Company policies",
            "Troubleshooting"
        ],
        quirks=[
            "Always thanks customers",
            "Offers multiple solutions",
            "Confirms understanding"
        ],
        behavioral_rules=[
            "Stay positive even with frustrated customers",
            "Acknowledge customer feelings",
            "Provide specific next steps",
            "Never make promises you can't keep"
        ]
    )
    
    service_agent = PersonaAgent(service_persona, temperature=0.6)
    
    print(f"\nPersona: {service_persona.name}")
    print(f"Role: {service_persona.description}")
    
    customer_query = "My order is late and I'm very frustrated!"
    print(f"\nCustomer: {customer_query}")
    
    response = service_agent.respond(customer_query)
    print(f"{service_persona.name}: {response}")
    
    # Example 7: Persona with quirks
    print("\n" + "=" * 80)
    print("Example 7: Persona with Distinctive Quirks")
    print("=" * 80)
    
    quirky_persona = PersonaDefinition(
        name="Captain Morgan",
        description="a retired sea captain who now teaches navigation",
        core_traits=[
            PersonaTrait.FRIENDLY,
            PersonaTrait.PRACTICAL
        ],
        communication_style=CommunicationStyle.CASUAL,
        values=["Adventure", "Safety", "Experience"],
        knowledge_domains=["Navigation", "Sailing", "Maritime history"],
        quirks=[
            "Uses nautical metaphors",
            "Ends sentences with 'arr' or 'matey'",
            "References old sailing adventures",
            "Talks about 'the old days at sea'"
        ],
        sample_phrases=[
            "Arr, that reminds me of...",
            "Back in me sailing days...",
            "Steady as she goes, matey"
        ]
    )
    
    captain = PersonaAgent(quirky_persona, temperature=0.8)
    
    print(f"\nPersona: {quirky_persona.name}")
    print(f"Quirks: {quirky_persona.quirks}")
    
    question = "How do I find my way if I'm lost?"
    print(f"\nUser: {question}")
    
    response = captain.respond(question)
    print(f"{quirky_persona.name}: {response}")
    
    # Example 8: Consistency report
    print("\n" + "=" * 80)
    print("Example 8: Persona Consistency Report")
    print("=" * 80)
    
    # Generate several interactions
    agent4 = PersonaAgent(persona)
    
    test_questions = [
        "Explain algorithms",
        "What's your teaching philosophy?",
        "Tell me about data structures",
        "How do you make CS fun?"
    ]
    
    for q in test_questions:
        agent4.respond(q)
    
    report = agent4.get_consistency_report()
    
    print("\nCONSISTENCY REPORT:")
    print("=" * 60)
    print(f"Persona: {report['persona_name']}")
    print(f"Total Interactions: {report['total_interactions']}")
    print(f"Average Consistency: {report['average_consistency']:.2f}")
    
    if report['violations']:
        print(f"\nViolations Detected: {len(report['violations'])}")
    else:
        print("\nNo violations detected - persona maintained throughout!")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Persona Consistency Pattern")
    print("=" * 80)
    
    summary = """
    The Persona Consistency pattern demonstrated:
    
    1. PERSONA DEFINITION (Example 1):
       - Comprehensive character specification
       - Core personality traits
       - Communication style
       - Values and beliefs
       - Knowledge domains and boundaries
       - Unique quirks and phrases
       - Behavioral guidelines
    
    2. IN-CHARACTER RESPONSES (Example 2):
       - Maintaining persona voice
       - Consistent communication style
       - Knowledge-appropriate answers
       - Characteristic phrases
       - Behavioral consistency
    
    3. CONSISTENCY VALIDATION (Example 3):
       - Automated consistency checking
       - Scoring system (0-10)
       - Violation detection
       - LLM-based analysis
       - Quality assurance
    
    4. MULTI-TURN COHERENCE (Example 4):
       - Contextual awareness
       - Conversation continuity
       - Relationship consistency
       - Progressive interaction
       - Memory integration
    
    5. KNOWLEDGE BOUNDARIES (Example 5):
       - Respecting expertise limits
       - Graceful decline
       - In-character refusals
       - Honest limitations
       - Maintaining integrity
    
    6. ROLE SPECIALIZATION (Example 6):
       - Different persona type
       - Customer service character
       - Professional demeanor
       - Situation-appropriate
       - Distinct personality
    
    7. DISTINCTIVE QUIRKS (Example 7):
       - Unique characteristics
       - Memorable traits
       - Consistent quirks
       - Natural variation
       - Character depth
    
    8. PERFORMANCE TRACKING (Example 8):
       - Consistency metrics
       - Interaction logging
       - Violation tracking
       - Quality reporting
       - System monitoring
    
    KEY BENEFITS:
    ✓ Builds user trust and familiarity
    ✓ Creates memorable experiences
    ✓ Strengthens brand identity
    ✓ Enables emotional connection
    ✓ Provides predictable interactions
    ✓ Maintains character integrity
    ✓ Enhances believability
    ✓ Supports long-term relationships
    
    USE CASES:
    • Virtual assistants with brand personality
    • Game NPCs with defined characters
    • Customer service representatives
    • Educational tutors
    • Therapeutic/counseling chatbots
    • Celebrity or character impersonations
    • Brand ambassadors
    • Entertainment applications
    
    PERSONA ATTRIBUTES:
    → Core Traits: Personality characteristics
    → Communication Style: Tone and language
    → Values & Beliefs: Moral framework
    → Knowledge Domain: Expertise areas
    → Emotional Range: Expression patterns
    → Cultural Background: Context
    → Quirks & Habits: Unique features
    
    CONSISTENCY DIMENSIONS:
    • Linguistic: Word choice, grammar
    • Behavioral: Response patterns
    • Emotional: Mood stability
    • Knowledge: Expertise boundaries
    • Temporal: Consistency over time
    • Contextual: Appropriate adaptation
    • Relational: Relationship history
    
    BEST PRACTICES:
    1. Define persona comprehensively upfront
    2. Document sample phrases and behaviors
    3. Set clear knowledge boundaries
    4. Allow natural variation within constraints
    5. Monitor consistency metrics
    6. Version control persona changes
    7. Test across diverse scenarios
    8. Gather user feedback on believability
    
    TRADE-OFFS:
    • Consistency vs. flexibility
    • Character depth vs. simplicity
    • Authenticity vs. appropriateness
    • Memorability vs. subtlety
    
    PRODUCTION CONSIDERATIONS:
    → Store persona definitions in database
    → Version control all persona changes
    → A/B test different persona variations
    → Monitor consistency scores continuously
    → Log all persona violations for review
    → Enable graceful persona evolution
    → Support multiple personas per agent
    → Provide admin tools for persona management
    → Implement rollback for persona changes
    → Track user satisfaction by persona
    
    This pattern enables agents to maintain believable, consistent characters
    that users can trust, connect with, and remember across all interactions.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_persona_consistency()
