"""
Pattern 065: Social Agent Patterns

Description:
    Social agents possess social intelligence - the ability to understand and
    interact with humans in socially appropriate ways. This includes theory of
    mind (understanding others' mental states), empathy, politeness, social norms,
    and collaborative communication. These capabilities enable more natural and
    effective human-AI interaction.

Components:
    1. Theory of Mind: Understanding others' beliefs, desires, intentions
    2. Empathy Module: Recognizing and responding to emotions
    3. Politeness Engine: Socially appropriate communication
    4. Social Norm Awareness: Understanding context-appropriate behavior
    5. Collaborative Communication: Turn-taking, clarification, common ground
    6. Personality Model: Consistent behavioral traits

Use Cases:
    - Virtual assistants and companions
    - Customer service chatbots
    - Educational tutors
    - Healthcare support agents
    - Social robots and NPCs
    - Collaborative work assistants

LangChain Implementation:
    Implements social intelligence using LLM-based mental state modeling,
    emotion recognition, politeness strategies, and collaborative dialogue.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class EmotionType(Enum):
    """Types of emotions"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"
    EXCITEMENT = "excitement"
    FRUSTRATION = "frustration"
    CONFUSION = "confusion"


class PolitenessLevel(Enum):
    """Levels of politeness"""
    FORMAL = "formal"
    POLITE = "polite"
    CASUAL = "casual"
    INFORMAL = "informal"


@dataclass
class MentalState:
    """Representation of someone's mental state"""
    beliefs: Dict[str, Any] = field(default_factory=dict)
    desires: List[str] = field(default_factory=list)
    intentions: List[str] = field(default_factory=list)
    emotions: Dict[EmotionType, float] = field(default_factory=dict)
    
    def to_text(self) -> str:
        text = "Mental State:\n"
        if self.beliefs:
            text += f"  Beliefs: {self.beliefs}\n"
        if self.desires:
            text += f"  Desires: {', '.join(self.desires)}\n"
        if self.intentions:
            text += f"  Intentions: {', '.join(self.intentions)}\n"
        if self.emotions:
            emotions_str = ", ".join(f"{e.value}={v:.2f}" for e, v in self.emotions.items())
            text += f"  Emotions: {emotions_str}\n"
        return text


@dataclass
class SocialContext:
    """Context for social interaction"""
    relationship: str  # e.g., "stranger", "friend", "professional"
    setting: str  # e.g., "formal meeting", "casual chat"
    cultural_context: Optional[str] = None
    power_dynamic: Optional[str] = None  # e.g., "equal", "superior", "subordinate"


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    speaker: str
    utterance: str
    emotion: Optional[EmotionType] = None
    intention: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class SocialAgent:
    """
    Socially intelligent agent.
    
    Features:
    1. Theory of mind modeling
    2. Emotion recognition and empathy
    3. Politeness and social appropriateness
    4. Collaborative communication
    5. Personality consistency
    """
    
    def __init__(
        self,
        personality: str = "helpful and friendly",
        default_politeness: PolitenessLevel = PolitenessLevel.POLITE
    ):
        self.personality = personality
        self.default_politeness = default_politeness
        self.conversation_history: List[ConversationTurn] = []
        
        # Theory of mind model
        self.user_mental_state = MentalState()
        
        # LLM for social reasoning
        self.social_reasoner = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.4
        )
        
        # LLM for empathetic response
        self.empathy_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.6
        )
    
    def recognize_emotion(self, text: str) -> Tuple[EmotionType, float]:
        """Recognize emotion in user's message"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the emotion in the following text.

Identify:
1. Primary emotion (joy, sadness, anger, fear, surprise, disgust, neutral, excitement, frustration, confusion)
2. Intensity (0.0 to 1.0)

Format: emotion,intensity
Example: joy,0.8"""),
            ("user", "{text}")
        ])
        
        chain = prompt | self.social_reasoner | StrOutputParser()
        
        try:
            result = chain.invoke({"text": text})
            parts = result.strip().split(',')
            
            emotion_str = parts[0].strip().lower()
            intensity = float(parts[1].strip()) if len(parts) > 1 else 0.5
            
            # Map to enum
            for emotion in EmotionType:
                if emotion.value in emotion_str:
                    return emotion, intensity
            
            return EmotionType.NEUTRAL, intensity
        except:
            return EmotionType.NEUTRAL, 0.5
    
    def infer_mental_state(self, utterance: str, context: str = "") -> MentalState:
        """Infer user's mental state (theory of mind)"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze what the user is thinking and feeling.

Infer:
1. Beliefs: What does the user believe?
2. Desires: What does the user want?
3. Intentions: What is the user trying to do?

{context}

Provide your analysis."""),
            ("user", "User said: {utterance}\n\nMental State:")
        ])
        
        chain = prompt | self.social_reasoner | StrOutputParser()
        analysis = chain.invoke({
            "utterance": utterance,
            "context": context
        })
        
        # Parse analysis (simplified)
        mental_state = MentalState()
        
        lines = analysis.split('\n')
        for line in lines:
            line_lower = line.lower()
            if 'belief' in line_lower or 'think' in line_lower:
                mental_state.beliefs['inferred'] = line.strip()
            elif 'desire' in line_lower or 'want' in line_lower:
                mental_state.desires.append(line.strip())
            elif 'intention' in line_lower or 'trying' in line_lower:
                mental_state.intentions.append(line.strip())
        
        return mental_state
    
    def generate_empathetic_response(
        self,
        user_utterance: str,
        emotion: EmotionType,
        intensity: float
    ) -> str:
        """Generate empathetic response"""
        
        empathy_guidance = {
            EmotionType.JOY: "Share in their happiness and be enthusiastic",
            EmotionType.SADNESS: "Show compassion and support",
            EmotionType.ANGER: "Acknowledge their feelings and be calming",
            EmotionType.FEAR: "Be reassuring and supportive",
            EmotionType.FRUSTRATION: "Show understanding and offer help",
            EmotionType.CONFUSION: "Be patient and clarifying",
            EmotionType.EXCITEMENT: "Match their energy positively"
        }
        
        guidance = empathy_guidance.get(emotion, "Be supportive and understanding")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a {self.personality} assistant with high emotional intelligence.

The user is experiencing {emotion.value} (intensity: {intensity:.1f}/1.0).

Response strategy: {guidance}

Show empathy by:
1. Acknowledging their emotion
2. Validating their feelings
3. Responding appropriately
4. Offering support if needed

Be genuine and warm."""),
            ("user", "{utterance}")
        ])
        
        chain = prompt | self.empathy_model | StrOutputParser()
        return chain.invoke({"utterance": user_utterance})
    
    def adjust_politeness(
        self,
        message: str,
        target_politeness: PolitenessLevel,
        social_context: SocialContext
    ) -> str:
        """Adjust message politeness level"""
        
        politeness_guides = {
            PolitenessLevel.FORMAL: "Use formal language, titles, and respectful phrasing",
            PolitenessLevel.POLITE: "Use polite language with 'please' and 'thank you'",
            PolitenessLevel.CASUAL: "Use friendly, conversational language",
            PolitenessLevel.INFORMAL: "Use relaxed, casual language"
        }
        
        guide = politeness_guides[target_politeness]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Adjust the following message to be {target_politeness.value}.

Context:
- Relationship: {social_context.relationship}
- Setting: {social_context.setting}

Strategy: {guide}

Maintain the core meaning while adjusting tone."""),
            ("user", "Original: {message}\n\nAdjusted:")
        ])
        
        chain = prompt | self.social_reasoner | StrOutputParser()
        return chain.invoke({"message": message})
    
    def collaborate(
        self,
        user_utterance: str,
        social_context: SocialContext
    ) -> str:
        """Collaborative response with social awareness"""
        
        # Recognize emotion
        emotion, intensity = self.recognize_emotion(user_utterance)
        
        # Update mental state
        self.user_mental_state = self.infer_mental_state(user_utterance)
        
        # Add to conversation history
        turn = ConversationTurn(
            speaker="user",
            utterance=user_utterance,
            emotion=emotion,
            intention=self.user_mental_state.intentions[0] if self.user_mental_state.intentions else None
        )
        self.conversation_history.append(turn)
        
        # Generate empathetic response
        if intensity > 0.6:  # Strong emotion
            response = self.generate_empathetic_response(user_utterance, emotion, intensity)
        else:
            # Standard response
            context = f"Relationship: {social_context.relationship}, Setting: {social_context.setting}"
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""You are a {self.personality} assistant.

Context: {context}

User's mental state:
{self.user_mental_state.to_text()}

Respond in a socially appropriate way:
1. Show understanding of their perspective
2. Be collaborative and helpful
3. Maintain conversation flow
4. Ask clarifying questions if needed"""),
                ("user", "{utterance}")
            ])
            
            chain = prompt | self.social_reasoner | StrOutputParser()
            response = chain.invoke({"utterance": user_utterance})
        
        # Adjust politeness
        response = self.adjust_politeness(response, self.default_politeness, social_context)
        
        # Add agent response to history
        self.conversation_history.append(ConversationTurn(
            speaker="agent",
            utterance=response
        ))
        
        return response
    
    def demonstrate_theory_of_mind(
        self,
        scenario: str
    ) -> Dict[str, Any]:
        """Demonstrate theory of mind reasoning"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze this scenario using theory of mind.

For each person in the scenario, identify:
1. What they BELIEVE (their knowledge/beliefs)
2. What they KNOW vs DON'T KNOW
3. What they might THINK about the situation
4. What they might DO next

This requires understanding different perspectives."""),
            ("user", "Scenario: {scenario}\n\nTheory of Mind Analysis:")
        ])
        
        chain = prompt | self.social_reasoner | StrOutputParser()
        analysis = chain.invoke({"scenario": scenario})
        
        return {
            "scenario": scenario,
            "analysis": analysis
        }


def demonstrate_social_agent():
    """Demonstrate Social Agent Patterns"""
    
    print("=" * 80)
    print("PATTERN 065: SOCIAL AGENT PATTERNS DEMONSTRATION")
    print("=" * 80)
    print("\nSocial intelligence: emotion, empathy, politeness, theory of mind\n")
    
    # Test 1: Emotion recognition and empathy
    print("\n" + "=" * 80)
    print("TEST 1: Emotion Recognition and Empathetic Response")
    print("=" * 80)
    
    agent = SocialAgent(
        personality="warm and understanding",
        default_politeness=PolitenessLevel.POLITE
    )
    
    emotional_messages = [
        ("I just got promoted at work! I can't believe it!", "joyful announcement"),
        ("I'm really struggling with this project. Nothing seems to work.", "frustrated"),
        ("I don't understand what you mean. This is confusing.", "confused"),
    ]
    
    social_ctx = SocialContext(
        relationship="friend",
        setting="casual chat"
    )
    
    for message, description in emotional_messages:
        print(f"\n📝 User ({description}): {message}")
        
        emotion, intensity = agent.recognize_emotion(message)
        print(f"   Detected: {emotion.value} (intensity: {intensity:.2f})")
        
        response = agent.collaborate(message, social_ctx)
        print(f"\n💬 Agent: {response}")
    
    # Test 2: Theory of mind
    print("\n" + "=" * 80)
    print("TEST 2: Theory of Mind Reasoning")
    print("=" * 80)
    
    agent2 = SocialAgent()
    
    scenario = """
    Sally puts her ball in the basket and leaves the room.
    While Sally is gone, Anne moves the ball from the basket to the box.
    Sally comes back to the room.
    Where will Sally look for her ball?
    """
    
    print(f"\n🧠 Classic Theory of Mind Test (Sally-Anne):")
    print(f"   {scenario.strip()}")
    
    tom_result = agent2.demonstrate_theory_of_mind(scenario)
    
    print(f"\n   Analysis:")
    for line in tom_result['analysis'].split('\n')[:8]:
        if line.strip():
            print(f"   {line}")
    
    # Test 3: Politeness adjustment
    print("\n" + "=" * 80)
    print("TEST 3: Politeness Level Adjustment")
    print("=" * 80)
    
    message = "I need the report by tomorrow."
    
    contexts = [
        (PolitenessLevel.FORMAL, SocialContext("subordinate to boss", "formal meeting")),
        (PolitenessLevel.POLITE, SocialContext("colleague", "office")),
        (PolitenessLevel.CASUAL, SocialContext("friend", "casual chat")),
    ]
    
    print(f"\n📄 Original message: '{message}'")
    print(f"\n   Adjusted for different contexts:\n")
    
    for politeness, context in contexts:
        adjusted = agent.adjust_politeness(message, politeness, context)
        print(f"   {politeness.value.upper()} ({context.relationship}):")
        print(f"      {adjusted}")
        print()
    
    # Test 4: Mental state inference
    print("\n" + "=" * 80)
    print("TEST 4: Mental State Inference")
    print("=" * 80)
    
    utterances = [
        "I've been trying to figure this out for hours but I'm not making progress.",
        "Maybe we should consider a different approach?",
        "I think I understand now, but could you clarify the last part?",
    ]
    
    print(f"\n🎭 Inferring mental states from utterances:\n")
    
    for utterance in utterances:
        print(f"   User: {utterance}")
        mental_state = agent.infer_mental_state(utterance)
        print(f"   Inferred State:")
        if mental_state.beliefs:
            print(f"      Beliefs: {mental_state.beliefs}")
        if mental_state.desires:
            print(f"      Desires: {mental_state.desires}")
        if mental_state.intentions:
            print(f"      Intentions: {mental_state.intentions}")
        print()
    
    # Test 5: Collaborative conversation
    print("\n" + "=" * 80)
    print("TEST 5: Collaborative Conversation")
    print("=" * 80)
    
    agent3 = SocialAgent(personality="helpful and patient tutor")
    
    conversation = [
        "I'm trying to learn Python but it's overwhelming.",
        "I've tried online tutorials but they move too fast.",
        "Maybe something more interactive would help?",
    ]
    
    context = SocialContext(
        relationship="student to tutor",
        setting="educational session"
    )
    
    print(f"\n🗣️  Collaborative conversation:\n")
    
    for user_msg in conversation:
        print(f"   Student: {user_msg}")
        response = agent3.collaborate(user_msg, context)
        print(f"   Tutor: {response}\n")
    
    print(f"   Conversation history: {len(agent3.conversation_history)} turns")
    
    # Summary
    print("\n" + "=" * 80)
    print("SOCIAL AGENT PATTERNS SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Natural Interaction: More human-like communication
2. Emotional Intelligence: Recognize and respond to emotions
3. Social Appropriateness: Context-aware behavior
4. Better Understanding: Theory of mind for perspective-taking
5. User Satisfaction: More pleasant and effective interactions

Social Intelligence Components:

1. Theory of Mind (ToM):
   - Understanding others' mental states
   - Beliefs, desires, intentions
   - Perspective-taking ability
   - Predicting behavior
   - Key for collaboration

2. Emotion Recognition:
   - Detect emotions in text/speech
   - Intensity estimation
   - Multi-emotion handling
   - Context-sensitive
   - Real-time updates

3. Empathy:
   - Emotional resonance
   - Appropriate responses
   - Validation of feelings
   - Supportive communication
   - Compassionate behavior

4. Politeness Strategies:
   - Formal vs informal register
   - Indirect speech acts
   - Face-saving behaviors
   - Cultural sensitivity
   - Power dynamics awareness

5. Social Norms:
   - Context-appropriate behavior
   - Turn-taking in conversation
   - Topic management
   - Disclosure appropriateness
   - Boundary respect

6. Collaborative Communication:
   - Common ground building
   - Clarification requests
   - Confirmation checks
   - Shared understanding
   - Cooperative principles

Emotion Categories:
- Basic: Joy, sadness, anger, fear, surprise, disgust
- Complex: Frustration, confusion, excitement, disappointment
- Intensity: Weak, moderate, strong
- Mixed: Multiple emotions simultaneously

Politeness Levels:
1. Formal: Official, respectful, distant
2. Polite: Courteous, considerate
3. Casual: Friendly, relaxed
4. Informal: Very casual, intimate

Theory of Mind Stages:
1. Basic: Recognize that others have mental states
2. Intermediate: Infer beliefs from behavior
3. Advanced: Recursive reasoning (I think you think I think...)
4. False Belief: Understand mistaken beliefs

Social Contexts:
- Relationship: Stranger, acquaintance, friend, family, professional
- Setting: Formal meeting, casual chat, educational, healthcare
- Power: Equal, superior, subordinate
- Culture: Individualist vs collectivist, direct vs indirect

Use Cases:
- Virtual Assistants: Alexa, Siri, Google Assistant
- Customer Service: Empathetic support bots
- Education: Patient tutoring agents
- Healthcare: Supportive health coaches
- Gaming: Believable NPCs
- Companionship: Social robots
- Workplace: Collaborative assistants

Challenges:
1. Ambiguity: Emotions and intentions unclear
2. Cultural Differences: Norms vary across cultures
3. Sarcasm/Irony: Difficult to detect
4. Context Dependency: Same words, different meanings
5. Individual Differences: Personal preferences
6. Privacy: Balancing personalization and privacy

Best Practices:
1. Start with emotion recognition
2. Validate user's feelings
3. Adjust tone to context
4. Be consistent in personality
5. Ask clarifying questions
6. Respect boundaries
7. Learn from interactions

Production Considerations:
- Real-time emotion detection
- Personality consistency
- Cultural adaptation
- Privacy protection
- Bias mitigation
- Graceful degradation
- User feedback integration

Advanced Techniques:
1. Multimodal Emotion Recognition
   - Text + voice + facial expressions
   - Improved accuracy

2. Dynamic Personality
   - Adapt to user preferences
   - Context-dependent traits

3. Cultural Adaptation
   - Learn cultural norms
   - Localization

4. Conversation State Tracking
   - Maintain context
   - Track goals

5. Proactive Empathy
   - Anticipate needs
   - Offer support preemptively

Evaluation Metrics:
- User Satisfaction: Ratings and feedback
- Engagement: Conversation length, return rate
- Task Success: Goal achievement
- Emotion Recognition Accuracy: Precision/recall
- Appropriateness: Human judgment
- Trust: Willingness to share information

Comparison with Related Patterns:
- vs. Basic Chatbot: Social awareness vs scripted
- vs. Task-Focused: Relationship vs transaction
- vs. Rule-Based: Flexible vs rigid
- vs. Reactive: Proactive social intelligence

Integration with Other Patterns:
- Memory: Remember user preferences and history
- Personalization: Adapt to individual users
- Multi-Agent: Social coordination
- Active Learning: Learn from corrections

Psychological Foundations:
- Social Cognition: How we understand others
- Emotional Intelligence: Perceive, use, understand, manage emotions
- Pragmatics: Language use in context
- Conversation Analysis: Structure of interaction

Research Directions:
- Artificial empathy effectiveness
- Cultural intelligence in AI
- Long-term relationship building
- Ethical boundaries of social AI
- Measuring social intelligence

Design Guidelines:
1. Personality: Consistent and appropriate
2. Boundaries: Clear role definition
3. Transparency: Honest about AI nature
4. Respect: User autonomy and feelings
5. Safety: Protect vulnerable users
6. Inclusivity: Diverse users and contexts

The Social Agent pattern enables AI systems to interact with
humans in more natural, appropriate, and satisfying ways through
social intelligence, emotional awareness, and collaborative communication.
""")


if __name__ == "__main__":
    demonstrate_social_agent()
