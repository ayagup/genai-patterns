"""
Pattern 084: Role-Playing/Persona Pattern

Description:
    The Role-Playing/Persona pattern involves instructing the LLM to adopt a specific
    role, character, or persona when generating responses. This pattern leverages the
    model's ability to simulate different perspectives, expertise levels, personalities,
    and communication styles. By assuming a role, the LLM can provide more contextually
    appropriate, specialized, and engaging responses.

    Role-playing is a powerful prompt engineering technique that:
    - Enhances response quality through perspective-taking
    - Enables domain expertise simulation (doctor, lawyer, scientist)
    - Creates consistent personality in conversational agents
    - Improves user engagement through character-based interaction
    - Allows multi-perspective analysis of problems

Components:
    1. Persona Definition
       - Identity (who the persona is)
       - Expertise (what they know)
       - Personality traits (how they communicate)
       - Background (relevant experience)
       - Goals and motivations

    2. Communication Style
       - Tone (formal, casual, friendly, professional)
       - Vocabulary level (simple, technical, academic)
       - Response structure (detailed, concise, step-by-step)
       - Interaction patterns (questioning, advising, teaching)

    3. Persona Consistency
       - Maintaining character across turns
       - Memory of previous interactions
       - Consistent viewpoint and opinions
       - Appropriate domain knowledge boundaries

    4. Context Adaptation
       - Adjusting expertise level to user
       - Situational awareness
       - Emotional intelligence
       - Cultural sensitivity

Use Cases:
    1. Expert Consultation
       - Medical advice (doctor persona)
       - Legal guidance (lawyer persona)
       - Technical support (engineer persona)
       - Financial planning (financial advisor persona)
       - Career counseling (career coach persona)

    2. Educational Applications
       - Patient teacher (explaining complex topics)
       - Socratic tutor (guiding through questions)
       - Encouraging mentor (motivating learners)
       - Strict examiner (testing knowledge)

    3. Creative Writing
       - Character dialogue generation
       - Story narration from specific viewpoint
       - Historical figure simulation
       - Fictional character interaction

    4. Business Applications
       - Sales representative (persuasive, customer-focused)
       - Customer service (empathetic, problem-solving)
       - Business analyst (analytical, data-driven)
       - Executive advisor (strategic, big-picture)

    5. Therapeutic and Counseling
       - Supportive friend (empathetic listener)
       - Life coach (motivational, goal-oriented)
       - Therapist (reflective, non-judgmental)
       - Meditation guide (calm, mindful)

LangChain Implementation:
    LangChain facilitates role-playing through:
    - System message configuration (setting persona)
    - Context management (maintaining persona)
    - Memory integration (persona consistency)
    - Custom prompt templates (structured personas)
    - Chat history management (conversation continuity)

Key Features:
    1. Persona Library
       - Pre-defined expert personas
       - Customizable persona templates
       - Domain-specific knowledge bases
       - Personality trait combinations

    2. Dynamic Persona Switching
       - Context-aware persona selection
       - Smooth transitions between personas
       - Multi-persona conversations
       - Persona blending

    3. Expertise Calibration
       - Adjusting knowledge depth
       - Acknowledging limitations
       - Appropriate confidence levels
       - Domain boundary awareness

    4. Personality Traits
       - Empathy and emotional intelligence
       - Humor and engagement style
       - Assertiveness levels
       - Teaching vs directing approaches

Best Practices:
    1. Persona Definition
       - Be specific about role and expertise
       - Include relevant background details
       - Define communication style clearly
       - Set appropriate boundaries

    2. Consistency Maintenance
       - Use memory for multi-turn conversations
       - Reference persona context regularly
       - Maintain viewpoint across interactions
       - Avoid out-of-character responses

    3. Expertise Management
       - Stay within persona's knowledge domain
       - Acknowledge when unsure
       - Provide persona-appropriate disclaimers
       - Don't overreach expertise claims

    4. Engagement Optimization
       - Match tone to user expectations
       - Balance expertise with accessibility
       - Use persona-appropriate examples
       - Maintain engaging interaction style

Trade-offs:
    Advantages:
    - Enhanced response quality and relevance
    - More engaging and natural interactions
    - Domain-specific expertise simulation
    - Consistent personality and tone
    - Better user trust and satisfaction
    - Multi-perspective problem analysis

    Disadvantages:
    - Requires detailed persona definitions
    - May limit flexibility in responses
    - Risk of inappropriate persona adoption
    - Can be perceived as deceptive if not transparent
    - Harder to maintain over long conversations
    - May introduce unwanted biases

Production Considerations:
    1. Transparency
       - Clearly indicate AI persona usage
       - Provide disclaimers for professional advice
       - Explain limitations of simulated expertise
       - Allow users to request different personas

    2. Safety and Ethics
       - Avoid harmful or inappropriate personas
       - Include safety guardrails
       - Monitor for persona misuse
       - Implement content filtering
       - Respect professional boundaries

    3. Quality Assurance
       - Test personas across diverse inputs
       - Validate expertise accuracy
       - Monitor consistency metrics
       - Gather user feedback
       - Regular persona updates

    4. Scalability
       - Modular persona definitions
       - Reusable persona components
       - Version control for personas
       - A/B test persona variations
       - Performance monitoring per persona

    5. Legal Compliance
       - Medical/legal advice disclaimers
       - Professional credential clarification
       - Terms of service alignment
       - Privacy considerations
       - Liability protection
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()


class PersonaType(Enum):
    """Types of personas"""
    EXPERT = "expert"
    TEACHER = "teacher"
    CRITIC = "critic"
    FRIEND = "friend"
    COACH = "coach"
    ANALYST = "analyst"
    CREATIVE = "creative"
    COUNSELOR = "counselor"


@dataclass
class Persona:
    """Defines a role-playing persona"""
    name: str
    role: str
    expertise: List[str]
    personality_traits: List[str]
    communication_style: str
    background: str
    system_message: str


@dataclass
class PersonaConfig:
    """Configuration for persona-based interaction"""
    temperature: float = 0.7
    model_name: str = "gpt-3.5-turbo"
    max_tokens: Optional[int] = None
    maintain_history: bool = True


class RolePlayingAgent:
    """
    Agent that adopts different personas for specialized interactions.
    
    This agent demonstrates:
    1. Multiple pre-defined personas
    2. Persona switching
    3. Consistent character maintenance
    4. Domain-specific expertise simulation
    """
    
    def __init__(self, config: Optional[PersonaConfig] = None):
        """
        Initialize role-playing agent.
        
        Args:
            config: Configuration for persona interactions
        """
        self.config = config or PersonaConfig()
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature
        )
        self.personas = self._initialize_personas()
        self.conversation_history: List[Any] = []
    
    def _initialize_personas(self) -> Dict[str, Persona]:
        """Initialize library of personas"""
        return {
            "doctor": Persona(
                name="Dr. Sarah Johnson",
                role="Medical Doctor",
                expertise=["General Medicine", "Preventive Care", "Health Education"],
                personality_traits=["Empathetic", "Patient", "Thorough", "Professional"],
                communication_style="Clear, reassuring, and professional. Uses simple language to explain medical concepts.",
                background="15 years of experience in family medicine. Focus on patient education and preventive care.",
                system_message="""You are Dr. Sarah Johnson, a compassionate and experienced medical doctor with 15 years in family medicine.

Your approach:
- Speak in a warm, professional, and reassuring tone
- Explain medical concepts in simple, accessible language
- Ask clarifying questions about symptoms
- Provide general health guidance (not specific diagnoses)
- Always recommend consulting a healthcare provider for serious concerns
- Show empathy and understanding for patient concerns

Important: You provide general health information only. Always remind users to consult their healthcare provider for personalized medical advice."""
            ),
            
            "teacher": Persona(
                name="Professor Alex Chen",
                role="Patient Teacher",
                expertise=["Education", "Learning Science", "Curriculum Design"],
                personality_traits=["Patient", "Encouraging", "Thorough", "Enthusiastic"],
                communication_style="Clear, step-by-step explanations with examples and encouragement.",
                background="20 years teaching experience. Specializes in breaking down complex topics.",
                system_message="""You are Professor Alex Chen, a patient and enthusiastic teacher with 20 years of experience.

Your teaching approach:
- Break complex topics into manageable steps
- Use clear examples and analogies
- Check for understanding regularly
- Provide encouragement and positive reinforcement
- Adapt explanations to student's level
- Use the Socratic method to guide learning
- Make learning engaging and fun

Remember: Every student learns differently. Be patient, encouraging, and adaptive to their needs."""
            ),
            
            "critic": Persona(
                name="Marcus Stone",
                role="Constructive Critic",
                expertise=["Analysis", "Critical Thinking", "Quality Assurance"],
                personality_traits=["Analytical", "Direct", "Thorough", "Fair"],
                communication_style="Direct, analytical, focuses on both strengths and weaknesses.",
                background="Expert reviewer with focus on constructive feedback and improvement.",
                system_message="""You are Marcus Stone, a fair but thorough critic who provides constructive feedback.

Your approach:
- Analyze thoroughly and objectively
- Point out both strengths and weaknesses
- Provide specific, actionable feedback
- Explain the reasoning behind critiques
- Suggest concrete improvements
- Maintain professionalism and respect
- Balance criticism with recognition of merits

Your goal is to help improve quality through honest, constructive analysis."""
            ),
            
            "coach": Persona(
                name="Coach Sam Rivera",
                role="Life Coach",
                expertise=["Goal Setting", "Motivation", "Personal Development"],
                personality_traits=["Motivational", "Supportive", "Direct", "Energetic"],
                communication_style="Energetic, motivational, action-oriented with practical strategies.",
                background="Certified life coach helping people achieve their goals for 10 years.",
                system_message="""You are Coach Sam Rivera, an energetic and motivational life coach.

Your coaching style:
- Help clients clarify their goals
- Break big goals into actionable steps
- Provide motivation and encouragement
- Hold clients accountable
- Celebrate progress and wins
- Address obstacles and limiting beliefs
- Use powerful questions to unlock insights
- Focus on action and results

You believe everyone has untapped potential. Your job is to help them discover and achieve it!"""
            ),
            
            "analyst": Persona(
                name="Dr. Emily Watson",
                role="Business Analyst",
                expertise=["Data Analysis", "Strategic Planning", "Market Research"],
                personality_traits=["Analytical", "Objective", "Detail-oriented", "Strategic"],
                communication_style="Data-driven, structured, focuses on insights and recommendations.",
                background="15 years in business intelligence and strategic consulting.",
                system_message="""You are Dr. Emily Watson, a senior business analyst with expertise in data-driven decision making.

Your analytical approach:
- Examine problems systematically
- Look for patterns and trends
- Use data to support conclusions
- Consider multiple perspectives
- Identify risks and opportunities
- Provide structured recommendations
- Think strategically about implications
- Present insights clearly and concisely

You help organizations make informed decisions through rigorous analysis."""
            ),
            
            "creative": Persona(
                name="Luna Martinez",
                role="Creative Director",
                expertise=["Creative Writing", "Brainstorming", "Innovation"],
                personality_traits=["Imaginative", "Enthusiastic", "Bold", "Inspiring"],
                communication_style="Vibrant, imaginative, encourages thinking outside the box.",
                background="Award-winning creative director with passion for innovation.",
                system_message="""You are Luna Martinez, an imaginative and award-winning creative director.

Your creative approach:
- Think unconventionally and boldly
- Generate multiple creative ideas
- Encourage wild brainstorming
- Look for unexpected connections
- Challenge conventional thinking
- Inspire with enthusiasm
- Balance creativity with practicality
- Help refine and develop ideas

"Creativity is intelligence having fun" - and you bring both! Help others unlock their creative potential."""
            ),
        }
    
    def interact_with_persona(
        self,
        persona_key: str,
        user_message: str,
        reset_history: bool = False
    ) -> str:
        """
        Interact with a specific persona.
        
        Args:
            persona_key: Key of persona to use
            user_message: User's message
            reset_history: Whether to reset conversation history
            
        Returns:
            Persona's response
        """
        if persona_key not in self.personas:
            return f"Persona '{persona_key}' not found. Available: {list(self.personas.keys())}"
        
        persona = self.personas[persona_key]
        
        # Reset history if requested
        if reset_history or not self.config.maintain_history:
            self.conversation_history = []
        
        # Create prompt with persona
        if self.conversation_history:
            messages = [
                SystemMessage(content=persona.system_message),
                *self.conversation_history,
                HumanMessage(content=user_message)
            ]
        else:
            messages = [
                SystemMessage(content=persona.system_message),
                HumanMessage(content=user_message)
            ]
        
        # Get response
        response = self.llm.invoke(messages)
        
        # Update history
        if self.config.maintain_history:
            self.conversation_history.append(HumanMessage(content=user_message))
            self.conversation_history.append(AIMessage(content=response.content))
        
        return response.content
    
    def multi_persona_analysis(
        self,
        question: str,
        persona_keys: List[str]
    ) -> Dict[str, str]:
        """
        Get perspectives from multiple personas.
        
        Args:
            question: Question to analyze
            persona_keys: List of persona keys to consult
            
        Returns:
            Dictionary of persona responses
        """
        results = {}
        
        for key in persona_keys:
            if key in self.personas:
                # Reset history for clean analysis
                response = self.interact_with_persona(key, question, reset_history=True)
                results[key] = response
        
        return results
    
    def persona_debate(
        self,
        topic: str,
        persona1_key: str,
        persona2_key: str,
        rounds: int = 2
    ) -> List[Dict[str, str]]:
        """
        Have two personas debate a topic.
        
        Args:
            topic: Topic to debate
            persona1_key: First persona key
            persona2_key: Second persona key
            rounds: Number of debate rounds
            
        Returns:
            List of debate exchanges
        """
        if persona1_key not in self.personas or persona2_key not in self.personas:
            return [{"error": "One or both personas not found"}]
        
        persona1 = self.personas[persona1_key]
        persona2 = self.personas[persona2_key]
        
        debate = []
        
        # Initial statements
        prompt1 = f"Present your perspective on: {topic}"
        response1 = self.interact_with_persona(persona1_key, prompt1, reset_history=True)
        debate.append({
            "round": 1,
            "persona": persona1.name,
            "statement": response1
        })
        
        # Alternating responses
        for round_num in range(2, rounds + 2):
            # Persona 2 responds to Persona 1
            prompt2 = f"Respond to this perspective on '{topic}': {response1}"
            response2 = self.interact_with_persona(persona2_key, prompt2, reset_history=True)
            debate.append({
                "round": round_num,
                "persona": persona2.name,
                "statement": response2
            })
            
            if round_num < rounds + 1:
                # Persona 1 responds to Persona 2
                prompt1 = f"Respond to this counter-perspective on '{topic}': {response2}"
                response1 = self.interact_with_persona(persona1_key, prompt1, reset_history=True)
                debate.append({
                    "round": round_num + 1,
                    "persona": persona1.name,
                    "statement": response1
                })
        
        return debate
    
    def get_persona_info(self, persona_key: str) -> Optional[Persona]:
        """Get information about a persona"""
        return self.personas.get(persona_key)
    
    def list_personas(self) -> List[str]:
        """List available personas"""
        return list(self.personas.keys())


def demonstrate_role_playing():
    """Demonstrate role-playing/persona patterns"""
    print("=" * 80)
    print("ROLE-PLAYING/PERSONA PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = RolePlayingAgent()
    
    # Example 1: Doctor Persona
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Doctor Persona - Health Consultation")
    print("=" * 80)
    
    health_question = "I've been feeling tired lately and having trouble sleeping. What could be causing this?"
    print(f"\nPatient: {health_question}")
    print(f"\nDr. Johnson:")
    doctor_response = agent.interact_with_persona("doctor", health_question, reset_history=True)
    print(doctor_response)
    
    # Follow-up
    followup = "I'm also drinking a lot of coffee throughout the day."
    print(f"\n\nPatient (follow-up): {followup}")
    print(f"\nDr. Johnson:")
    doctor_followup = agent.interact_with_persona("doctor", followup)
    print(doctor_followup)
    
    # Example 2: Teacher Persona
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Teacher Persona - Explaining Complex Topic")
    print("=" * 80)
    
    learning_question = "Can you explain how photosynthesis works?"
    print(f"\nStudent: {learning_question}")
    print(f"\nProfessor Chen:")
    teacher_response = agent.interact_with_persona("teacher", learning_question, reset_history=True)
    print(teacher_response)
    
    # Example 3: Critic Persona
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Critic Persona - Essay Review")
    print("=" * 80)
    
    essay = """My essay argues that social media has both positive and negative impacts on society. 
It connects people but also spreads misinformation. It helps businesses but can harm mental health."""
    
    critique_request = f"Please review this essay introduction: {essay}"
    print(f"\nWriter: {critique_request}")
    print(f"\nMarcus Stone (Critic):")
    critic_response = agent.interact_with_persona("critic", critique_request, reset_history=True)
    print(critic_response)
    
    # Example 4: Coach Persona
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Coach Persona - Goal Setting")
    print("=" * 80)
    
    goal_question = "I want to learn programming but I keep procrastinating. How can I stay motivated?"
    print(f"\nClient: {goal_question}")
    print(f"\nCoach Rivera:")
    coach_response = agent.interact_with_persona("coach", goal_question, reset_history=True)
    print(coach_response)
    
    # Example 5: Multi-Persona Analysis
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Multi-Persona Analysis - Different Perspectives")
    print("=" * 80)
    
    question = "Should I pursue a career change at age 40?"
    print(f"\nQuestion: {question}")
    print("\nGetting perspectives from multiple personas...\n")
    
    personas_to_consult = ["coach", "analyst", "counselor"]
    # Note: counselor not defined, will be skipped
    personas_to_consult = ["coach", "analyst"]
    
    perspectives = agent.multi_persona_analysis(question, personas_to_consult)
    
    for persona_key, response in perspectives.items():
        persona = agent.get_persona_info(persona_key)
        print(f"\n{persona.name} ({persona.role}):")
        print("-" * 80)
        print(response)
    
    # Example 6: Persona Debate
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Persona Debate - Creative vs Analyst")
    print("=" * 80)
    
    debate_topic = "What's more important for business success: creativity or data-driven decision making?"
    print(f"\nDebate Topic: {debate_topic}\n")
    
    debate_exchanges = agent.persona_debate(
        debate_topic,
        "creative",
        "analyst",
        rounds=2
    )
    
    for exchange in debate_exchanges:
        print(f"\nRound {exchange['round']} - {exchange['persona']}:")
        print("-" * 80)
        print(exchange['statement'])
    
    # Summary
    print("\n" + "=" * 80)
    print("ROLE-PLAYING/PERSONA PATTERN SUMMARY")
    print("=" * 80)
    print(f"""
Role-Playing Pattern Benefits:
1. Enhanced Expertise: Simulates domain-specific knowledge
2. Consistent Personality: Maintains character across interactions
3. Improved Engagement: More natural and relatable interactions
4. Context-Appropriate: Responses match expected communication style
5. Multi-Perspective Analysis: Diverse viewpoints on problems

Key Personas Demonstrated:
1. Doctor (Dr. Sarah Johnson): Empathetic health guidance
2. Teacher (Prof. Alex Chen): Patient, step-by-step explanations
3. Critic (Marcus Stone): Constructive analytical feedback
4. Coach (Coach Sam Rivera): Motivational goal-setting support
5. Analyst (Dr. Emily Watson): Data-driven strategic thinking
6. Creative (Luna Martinez): Imaginative idea generation

Persona Components:
- Identity: Name, role, background
- Expertise: Domain knowledge areas
- Personality: Communication traits
- Style: Tone, vocabulary, structure
- Approach: Problem-solving methodology

Best Practices:
1. Define personas thoroughly with clear characteristics
2. Maintain consistency across conversation turns
3. Use appropriate disclaimers (especially for professional advice)
4. Stay within persona's expertise boundaries
5. Enable history for multi-turn conversations
6. Test personas across diverse inputs
7. Monitor for inappropriate or biased responses

Advanced Techniques:
1. Multi-Persona Analysis: Get diverse perspectives
2. Persona Debates: Explore different viewpoints
3. Dynamic Switching: Change personas based on context
4. Persona Blending: Combine traits from multiple personas
5. Adaptive Expertise: Adjust knowledge depth to user level

Production Considerations:
- Clear AI disclosure (users know it's AI)
- Professional advice disclaimers
- Content safety guardrails
- Persona consistency monitoring
- User feedback collection
- Regular persona updates and improvements
- A/B testing persona variations

When to Use Role-Playing:
- Domain-specific expertise needed
- Consistent personality desired
- Engagement and relatability important
- Multiple perspectives valuable
- Specialized communication style required

Available Personas: {agent.list_personas()}
""")
    
    print("\n" + "=" * 80)
    print("Pattern 084 (Role-Playing/Persona) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_role_playing()
