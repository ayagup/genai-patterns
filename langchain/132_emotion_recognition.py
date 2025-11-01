"""
Pattern 132: Emotion Recognition & Response

Description:
    The Emotion Recognition & Response pattern enables agents to detect emotional
    states from user input and respond with appropriate empathy, tone, and content.
    Rather than treating all interactions uniformly, the agent recognizes emotions
    like frustration, joy, sadness, or confusion and adapts its behavior to be
    contextually appropriate and emotionally intelligent.
    
    This pattern is critical for creating human-centered AI that can build rapport,
    de-escalate conflicts, celebrate successes, and provide appropriate support
    based on user emotional state. It combines emotion detection (analyzing text
    for emotional cues), sentiment analysis, empathetic response generation, and
    emotional context tracking over conversations.
    
    The pattern addresses challenges like subtle emotion detection, cultural
    differences in expression, appropriate emotional responses, avoiding
    manipulation, and maintaining authenticity while being supportive.

Key Components:
    1. Emotion Detection: Identify emotions from text
    2. Sentiment Analysis: Classify overall tone (positive/negative/neutral)
    3. Emotional Intensity: Measure emotion strength
    4. Context Tracking: Track emotional arc over conversation
    5. Empathetic Response: Generate appropriate reactions
    6. Tone Adaptation: Adjust agent tone based on user emotion
    7. Intervention Logic: Special handling for extreme emotions

Emotion Categories:
    1. Primary Emotions: Joy, sadness, anger, fear, surprise, disgust
    2. Complex Emotions: Frustration, excitement, confusion, satisfaction
    3. Social Emotions: Gratitude, embarrassment, pride, shame
    4. Cognitive States: Confidence, uncertainty, curiosity, boredom

Detection Signals:
    1. Explicit Expressions: "I'm frustrated", "This is great!"
    2. Exclamation Marks: Multiple !!! or ???
    3. Capitalization: ALL CAPS for emphasis
    4. Word Choice: Emotional vocabulary
    5. Sentence Structure: Short angry bursts vs. rambling confusion
    6. Emoticons/Emoji: 😊 😠 😢 🤔
    7. Negation: "This isn't working", "No luck"

Response Strategies:
    1. Validation: Acknowledge the emotion
    2. Empathy: Show understanding
    3. Support: Offer help or encouragement
    4. De-escalation: Calm strong negative emotions
    5. Celebration: Share in positive emotions
    6. Clarification: Address confusion
    7. Patience: Adjust pace for overwhelm

Use Cases:
    - Customer support (handling complaints)
    - Mental health chatbots
    - Educational tutors (frustration detection)
    - Healthcare assistants (patient anxiety)
    - HR assistants (employee concerns)
    - Social companions
    - Crisis intervention

Advantages:
    - Improved user satisfaction
    - Better conflict resolution
    - Stronger emotional connection
    - Appropriate tone matching
    - Early problem detection
    - Personalized support
    - Enhanced trust

Challenges:
    - Ambiguous emotional signals
    - Cultural differences in expression
    - Sarcasm and irony detection
    - Over-interpretation risk
    - Privacy and sensitivity
    - Avoiding emotional manipulation
    - Maintaining authenticity

LangChain Implementation:
    This implementation uses LangChain for:
    - Emotion detection from text
    - Sentiment classification
    - Empathetic response generation
    - Context-aware emotional tracking
    
Production Considerations:
    - Use multiple emotion detection methods
    - Track emotional patterns over time
    - Implement crisis detection triggers
    - Provide human escalation paths
    - Respect cultural differences
    - Avoid over-familiarity
    - Log emotion data securely
    - Enable emotion detection toggles
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class Emotion(Enum):
    """Primary emotion categories."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    FRUSTRATION = "frustration"
    EXCITEMENT = "excitement"
    CONFUSION = "confusion"
    SATISFACTION = "satisfaction"
    GRATITUDE = "gratitude"
    ANXIETY = "anxiety"
    NEUTRAL = "neutral"


class Sentiment(Enum):
    """Overall sentiment classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class ResponseStrategy(Enum):
    """Strategy for responding to emotions."""
    VALIDATE = "validate"
    EMPATHIZE = "empathize"
    SUPPORT = "support"
    DE_ESCALATE = "de_escalate"
    CELEBRATE = "celebrate"
    CLARIFY = "clarify"
    ENCOURAGE = "encourage"


@dataclass
class EmotionalState:
    """
    Detected emotional state from user input.
    
    Attributes:
        primary_emotion: Main emotion detected
        secondary_emotions: Other emotions present
        sentiment: Overall sentiment
        intensity: Emotion strength (0-1)
        confidence: Detection confidence (0-1)
        signals: Textual cues that indicated emotion
        timestamp: When detected
    """
    primary_emotion: Emotion
    secondary_emotions: List[Emotion] = field(default_factory=list)
    sentiment: Sentiment = Sentiment.NEUTRAL
    intensity: float = 0.5
    confidence: float = 0.5
    signals: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EmpatheticResponse:
    """
    Response tailored to emotional state.
    
    Attributes:
        response_text: Generated response
        strategy: Strategy used
        acknowledged_emotion: Emotion that was addressed
        tone: Response tone
        follow_up_actions: Suggested next steps
    """
    response_text: str
    strategy: ResponseStrategy
    acknowledged_emotion: Emotion
    tone: str
    follow_up_actions: List[str] = field(default_factory=list)


class EmotionalAgent:
    """
    Agent with emotion recognition and empathetic response capabilities.
    
    This agent detects user emotions and responds with appropriate
    empathy, tone, and support.
    """
    
    def __init__(self, temperature: float = 0.7):
        """
        Initialize emotional agent.
        
        Args:
            temperature: LLM temperature for response generation
        """
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.emotional_history: List[EmotionalState] = []
        
        # Emotion keywords for rule-based detection
        self.emotion_keywords = {
            Emotion.JOY: ["happy", "great", "wonderful", "excellent", "love", "amazing", "perfect"],
            Emotion.SADNESS: ["sad", "disappointed", "unhappy", "down", "depressed", "terrible"],
            Emotion.ANGER: ["angry", "furious", "mad", "annoyed", "irritated", "outraged"],
            Emotion.FRUSTRATION: ["frustrated", "stuck", "can't", "won't work", "difficult", "struggle"],
            Emotion.CONFUSION: ["confused", "don't understand", "unclear", "lost", "puzzled"],
            Emotion.GRATITUDE: ["thank", "thanks", "appreciate", "grateful", "helpful"],
            Emotion.ANXIETY: ["worried", "anxious", "nervous", "scared", "concerned"],
            Emotion.EXCITEMENT: ["excited", "can't wait", "awesome", "fantastic", "thrilled"],
        }
    
    def detect_emotion_simple(self, text: str) -> EmotionalState:
        """
        Simple rule-based emotion detection.
        
        Args:
            text: User input text
            
        Returns:
            Detected emotional state
        """
        text_lower = text.lower()
        detected_emotions = []
        signals = []
        
        # Check keywords
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_emotions.append(emotion)
                    signals.append(f"Keyword: '{keyword}'")
        
        # Check for caps (anger/excitement)
        if text.isupper() and len(text) > 10:
            detected_emotions.append(Emotion.ANGER)
            signals.append("ALL CAPS")
        
        # Check for exclamation marks
        exclamation_count = text.count('!')
        if exclamation_count >= 2:
            detected_emotions.append(Emotion.EXCITEMENT)
            signals.append(f"Multiple exclamations ({exclamation_count})")
        
        # Check for question marks (confusion)
        question_count = text.count('?')
        if question_count >= 2:
            detected_emotions.append(Emotion.CONFUSION)
            signals.append(f"Multiple questions ({question_count})")
        
        # Determine primary emotion
        if detected_emotions:
            primary = detected_emotions[0]
            secondary = detected_emotions[1:] if len(detected_emotions) > 1 else []
        else:
            primary = Emotion.NEUTRAL
            secondary = []
        
        # Determine sentiment
        positive_emotions = {Emotion.JOY, Emotion.GRATITUDE, Emotion.EXCITEMENT, Emotion.SATISFACTION}
        negative_emotions = {Emotion.SADNESS, Emotion.ANGER, Emotion.FRUSTRATION, Emotion.ANXIETY, Emotion.FEAR}
        
        if primary in positive_emotions:
            sentiment = Sentiment.POSITIVE
        elif primary in negative_emotions:
            sentiment = Sentiment.NEGATIVE
        else:
            sentiment = Sentiment.NEUTRAL
        
        # Estimate intensity
        intensity = 0.5
        if exclamation_count > 0 or text.isupper():
            intensity = 0.8
        elif primary != Emotion.NEUTRAL:
            intensity = 0.6
        
        return EmotionalState(
            primary_emotion=primary,
            secondary_emotions=secondary,
            sentiment=sentiment,
            intensity=intensity,
            confidence=0.7 if signals else 0.3,
            signals=signals
        )
    
    def detect_emotion_llm(self, text: str) -> EmotionalState:
        """
        LLM-based emotion detection.
        
        Args:
            text: User input text
            
        Returns:
            Detected emotional state
        """
        prompt = ChatPromptTemplate.from_template(
            "Analyze the emotional content of this message:\n\n"
            "\"{text}\"\n\n"
            "Provide:\n"
            "PRIMARY_EMOTION: [joy/sadness/anger/fear/frustration/confusion/gratitude/excitement/neutral/etc.]\n"
            "SENTIMENT: [positive/negative/neutral/mixed]\n"
            "INTENSITY: [0.0-1.0, where 0 is no emotion and 1 is extreme]\n"
            "SIGNALS: [what textual cues indicate this emotion]\n"
            "CONFIDENCE: [0.0-1.0, how certain are you]"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        analysis = chain.invoke({"text": text})
        
        # Parse analysis
        lines = analysis.strip().split('\n')
        
        primary_emotion = Emotion.NEUTRAL
        sentiment = Sentiment.NEUTRAL
        intensity = 0.5
        confidence = 0.5
        signals = []
        
        for line in lines:
            if line.startswith("PRIMARY_EMOTION:"):
                emotion_str = line.split(":")[1].strip().lower()
                # Map to enum
                for emotion in Emotion:
                    if emotion.value in emotion_str:
                        primary_emotion = emotion
                        break
            
            elif line.startswith("SENTIMENT:"):
                sentiment_str = line.split(":")[1].strip().lower()
                if "positive" in sentiment_str:
                    sentiment = Sentiment.POSITIVE
                elif "negative" in sentiment_str:
                    sentiment = Sentiment.NEGATIVE
                elif "mixed" in sentiment_str:
                    sentiment = Sentiment.MIXED
            
            elif line.startswith("INTENSITY:"):
                try:
                    intensity = float(line.split(":")[1].strip().split()[0])
                except:
                    pass
            
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":")[1].strip().split()[0])
                except:
                    pass
            
            elif line.startswith("SIGNALS:"):
                signals.append(line.split(":")[1].strip())
        
        return EmotionalState(
            primary_emotion=primary_emotion,
            sentiment=sentiment,
            intensity=intensity,
            confidence=confidence,
            signals=signals
        )
    
    def select_response_strategy(self, emotional_state: EmotionalState) -> ResponseStrategy:
        """
        Select appropriate response strategy.
        
        Args:
            emotional_state: Detected emotional state
            
        Returns:
            Response strategy
        """
        emotion = emotional_state.primary_emotion
        intensity = emotional_state.intensity
        
        # High-intensity negative emotions need de-escalation
        if intensity > 0.7 and emotional_state.sentiment == Sentiment.NEGATIVE:
            return ResponseStrategy.DE_ESCALATE
        
        # Map emotions to strategies
        strategy_map = {
            Emotion.JOY: ResponseStrategy.CELEBRATE,
            Emotion.EXCITEMENT: ResponseStrategy.CELEBRATE,
            Emotion.GRATITUDE: ResponseStrategy.VALIDATE,
            Emotion.SADNESS: ResponseStrategy.EMPATHIZE,
            Emotion.ANGER: ResponseStrategy.DE_ESCALATE,
            Emotion.FRUSTRATION: ResponseStrategy.SUPPORT,
            Emotion.CONFUSION: ResponseStrategy.CLARIFY,
            Emotion.ANXIETY: ResponseStrategy.SUPPORT,
            Emotion.NEUTRAL: ResponseStrategy.VALIDATE
        }
        
        return strategy_map.get(emotion, ResponseStrategy.VALIDATE)
    
    def generate_empathetic_response(
        self,
        user_input: str,
        emotional_state: EmotionalState,
        strategy: ResponseStrategy
    ) -> EmpatheticResponse:
        """
        Generate response tailored to emotional state.
        
        Args:
            user_input: User's message
            emotional_state: Detected emotional state
            strategy: Response strategy to use
            
        Returns:
            Empathetic response
        """
        # Build context about emotion
        emotion_context = (
            f"The user is expressing {emotional_state.primary_emotion.value} "
            f"with {emotional_state.intensity:.1f} intensity. "
            f"Overall sentiment is {emotional_state.sentiment.value}."
        )
        
        # Strategy-specific instructions
        strategy_instructions = {
            ResponseStrategy.VALIDATE: "Acknowledge and validate their feelings.",
            ResponseStrategy.EMPATHIZE: "Show deep empathy and understanding. Be gentle and supportive.",
            ResponseStrategy.SUPPORT: "Offer concrete support and encouragement. Focus on solutions.",
            ResponseStrategy.DE_ESCALATE: "Be calm and soothing. Acknowledge frustration and offer to help resolve issues.",
            ResponseStrategy.CELEBRATE: "Share in their positive emotions! Be enthusiastic and encouraging.",
            ResponseStrategy.CLARIFY: "Help them understand. Be patient and clear. Break things down.",
            ResponseStrategy.ENCOURAGE: "Provide motivation and confidence. Highlight their strengths."
        }
        
        instruction = strategy_instructions.get(strategy, "Respond appropriately.")
        
        prompt = ChatPromptTemplate.from_template(
            "User message: \"{user_input}\"\n\n"
            "Emotional Context: {emotion_context}\n\n"
            "Response Strategy: {strategy}\n"
            "Instruction: {instruction}\n\n"
            "Generate an empathetic, helpful response that appropriately addresses "
            "the user's emotional state. Be genuine and human-like."
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        response_text = chain.invoke({
            "user_input": user_input,
            "emotion_context": emotion_context,
            "strategy": strategy.value,
            "instruction": instruction
        })
        
        # Determine tone
        tone_map = {
            ResponseStrategy.CELEBRATE: "enthusiastic",
            ResponseStrategy.EMPATHIZE: "gentle",
            ResponseStrategy.DE_ESCALATE: "calm",
            ResponseStrategy.CLARIFY: "patient",
            ResponseStrategy.SUPPORT: "encouraging",
            ResponseStrategy.VALIDATE: "understanding"
        }
        
        return EmpatheticResponse(
            response_text=response_text,
            strategy=strategy,
            acknowledged_emotion=emotional_state.primary_emotion,
            tone=tone_map.get(strategy, "neutral")
        )
    
    def respond(self, user_input: str, use_llm_detection: bool = True) -> str:
        """
        Process input and generate empathetic response.
        
        Args:
            user_input: User's message
            use_llm_detection: Use LLM vs. rule-based detection
            
        Returns:
            Empathetic response text
        """
        # Detect emotion
        if use_llm_detection:
            emotional_state = self.detect_emotion_llm(user_input)
        else:
            emotional_state = self.detect_emotion_simple(user_input)
        
        # Store in history
        self.emotional_history.append(emotional_state)
        
        # Select strategy
        strategy = self.select_response_strategy(emotional_state)
        
        # Generate response
        response = self.generate_empathetic_response(user_input, emotional_state, strategy)
        
        return response.response_text
    
    def get_emotional_trajectory(self) -> List[Dict[str, Any]]:
        """Get emotional arc over conversation."""
        return [
            {
                "emotion": state.primary_emotion.value,
                "sentiment": state.sentiment.value,
                "intensity": state.intensity,
                "timestamp": state.timestamp.isoformat()
            }
            for state in self.emotional_history
        ]
    
    def get_current_emotional_state(self) -> Optional[EmotionalState]:
        """Get most recent emotional state."""
        return self.emotional_history[-1] if self.emotional_history else None


def demonstrate_emotion_recognition():
    """Demonstrate emotion recognition and response pattern."""
    
    print("=" * 80)
    print("EMOTION RECOGNITION & RESPONSE PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic emotion detection
    print("\n" + "=" * 80)
    print("Example 1: Basic Emotion Detection")
    print("=" * 80)
    
    agent = EmotionalAgent()
    
    test_messages = [
        "I'm so frustrated! This isn't working at all!",
        "Thank you so much! This is exactly what I needed!",
        "I'm a bit confused about how this works...",
        "This is terrible. I'm very disappointed.",
        "WOW! This is AMAZING!!!"
    ]
    
    print("\nDetecting emotions in various messages:")
    for msg in test_messages:
        emotion_state = agent.detect_emotion_simple(msg)
        print(f"\nMessage: \"{msg}\"")
        print(f"  Primary Emotion: {emotion_state.primary_emotion.value}")
        print(f"  Sentiment: {emotion_state.sentiment.value}")
        print(f"  Intensity: {emotion_state.intensity:.2f}")
        print(f"  Signals: {emotion_state.signals}")
    
    # Example 2: LLM-based detection
    print("\n" + "=" * 80)
    print("Example 2: LLM-Based Emotion Detection")
    print("=" * 80)
    
    complex_message = "I've been trying this for hours and nothing works. I'm starting to think I'm just not smart enough for this."
    
    print(f"\nAnalyzing complex message:")
    print(f'"{complex_message}"')
    
    emotion_state = agent.detect_emotion_llm(complex_message)
    print(f"\nPrimary Emotion: {emotion_state.primary_emotion.value}")
    print(f"Sentiment: {emotion_state.sentiment.value}")
    print(f"Intensity: {emotion_state.intensity:.2f}")
    print(f"Confidence: {emotion_state.confidence:.2f}")
    print(f"Signals: {emotion_state.signals}")
    
    # Example 3: Strategy selection
    print("\n" + "=" * 80)
    print("Example 3: Response Strategy Selection")
    print("=" * 80)
    
    emotions_and_strategies = [
        (Emotion.FRUSTRATION, 0.8, Sentiment.NEGATIVE),
        (Emotion.JOY, 0.9, Sentiment.POSITIVE),
        (Emotion.CONFUSION, 0.6, Sentiment.NEUTRAL),
        (Emotion.ANGER, 0.9, Sentiment.NEGATIVE),
    ]
    
    print("\nStrategy selection for different emotional states:")
    for emotion, intensity, sentiment in emotions_and_strategies:
        state = EmotionalState(
            primary_emotion=emotion,
            sentiment=sentiment,
            intensity=intensity
        )
        strategy = agent.select_response_strategy(state)
        print(f"\n{emotion.value} (intensity={intensity}, {sentiment.value})")
        print(f"  → Strategy: {strategy.value}")
    
    # Example 4: Empathetic responses
    print("\n" + "=" * 80)
    print("Example 4: Empathetic Response Generation")
    print("=" * 80)
    
    agent2 = EmotionalAgent()
    
    scenarios = [
        "I'm so frustrated! I've been stuck on this problem for hours!",
        "This is absolutely perfect! You've made my day!",
        "I don't understand how this feature works. Can you help?"
    ]
    
    print("\nGenerating empathetic responses:")
    for scenario in scenarios:
        print(f"\nUser: {scenario}")
        response = agent2.respond(scenario)
        
        # Get emotional state
        state = agent2.get_current_emotional_state()
        print(f"Detected: {state.primary_emotion.value} ({state.intensity:.2f} intensity)")
        print(f"Agent: {response}")
    
    # Example 5: De-escalation
    print("\n" + "=" * 80)
    print("Example 5: De-escalation of Strong Negative Emotions")
    print("=" * 80)
    
    agent3 = EmotionalAgent()
    
    angry_message = "This is completely unacceptable! I've wasted so much time and nothing works! I want a refund NOW!"
    
    print(f"\nAngry User: {angry_message}")
    
    response = agent3.respond(angry_message)
    state = agent3.get_current_emotional_state()
    
    print(f"\nDetected Emotion: {state.primary_emotion.value}")
    print(f"Intensity: {state.intensity:.2f}")
    print(f"Sentiment: {state.sentiment.value}")
    print(f"\nAgent (De-escalation): {response}")
    
    # Example 6: Celebration
    print("\n" + "=" * 80)
    print("Example 6: Celebrating Positive Emotions")
    print("=" * 80)
    
    agent4 = EmotionalAgent()
    
    excited_message = "I finally got it working!!! This is amazing! Thank you so much!"
    
    print(f"\nExcited User: {excited_message}")
    
    response = agent4.respond(excited_message)
    state = agent4.get_current_emotional_state()
    
    print(f"\nDetected Emotion: {state.primary_emotion.value}")
    print(f"Intensity: {state.intensity:.2f}")
    print(f"\nAgent (Celebration): {response}")
    
    # Example 7: Emotional trajectory
    print("\n" + "=" * 80)
    print("Example 7: Tracking Emotional Trajectory")
    print("=" * 80)
    
    agent5 = EmotionalAgent()
    
    conversation = [
        "I'm trying to set this up but it's not working.",
        "I've tried everything! This is so frustrating!",
        "Oh wait, I think I see the issue now.",
        "Yes! It's working now! Thank you!"
    ]
    
    print("\nConversation with emotional tracking:")
    for msg in conversation:
        print(f"\nUser: {msg}")
        response = agent5.respond(msg)
        state = agent5.get_current_emotional_state()
        print(f"[{state.primary_emotion.value}, intensity={state.intensity:.2f}]")
        print(f"Agent: {response}")
    
    print("\n" + "-" * 60)
    print("\nEMOTIONAL TRAJECTORY:")
    trajectory = agent5.get_emotional_trajectory()
    for i, point in enumerate(trajectory, 1):
        print(f"{i}. {point['emotion']} ({point['sentiment']}) - intensity: {point['intensity']:.2f}")
    
    # Example 8: Mixed emotions
    print("\n" + "=" * 80)
    print("Example 8: Handling Mixed/Complex Emotions")
    print("=" * 80)
    
    agent6 = EmotionalAgent()
    
    mixed_message = "I'm excited about this new feature, but also worried I won't be able to learn it in time."
    
    print(f"\nUser: {mixed_message}")
    
    response = agent6.respond(mixed_message)
    state = agent6.get_current_emotional_state()
    
    print(f"\nDetected Primary: {state.primary_emotion.value}")
    print(f"Secondary: {[e.value for e in state.secondary_emotions]}")
    print(f"Sentiment: {state.sentiment.value}")
    print(f"\nAgent: {response}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Emotion Recognition & Response Pattern")
    print("=" * 80)
    
    summary = """
    The Emotion Recognition & Response pattern demonstrated:
    
    1. EMOTION DETECTION (Examples 1-2):
       - Rule-based detection (keywords, punctuation, caps)
       - LLM-based detection (contextual understanding)
       - Primary and secondary emotions
       - Intensity measurement (0-1 scale)
       - Confidence scoring
       - Signal identification
    
    2. SENTIMENT ANALYSIS (Example 1):
       - Positive/negative/neutral classification
       - Overall tone assessment
       - Mixed sentiment detection
       - Intensity consideration
    
    3. STRATEGY SELECTION (Example 3):
       - Emotion-to-strategy mapping
       - Intensity consideration
       - Context-appropriate responses
       - Validate, empathize, support, de-escalate, celebrate, clarify
    
    4. EMPATHETIC RESPONSES (Example 4):
       - Emotion-aware generation
       - Appropriate tone matching
       - Genuine, human-like language
       - Contextual support
    
    5. DE-ESCALATION (Example 5):
       - High-intensity negative emotion handling
       - Calming language
       - Problem-solving focus
       - Validation of feelings
       - Conflict resolution
    
    6. CELEBRATION (Example 6):
       - Positive emotion amplification
       - Enthusiastic responses
       - Shared joy
       - Encouragement
       - Relationship building
    
    7. EMOTIONAL TRAJECTORY (Example 7):
       - Tracking emotions over time
       - Conversation arc analysis
       - Progress monitoring
       - Pattern recognition
       - Intervention timing
    
    8. COMPLEX EMOTIONS (Example 8):
       - Mixed emotion handling
       - Nuanced understanding
       - Multiple emotion acknowledgment
       - Balanced responses
    
    KEY BENEFITS:
    ✓ Improved user satisfaction
    ✓ Better conflict resolution
    ✓ Stronger emotional connections
    ✓ Appropriate tone matching
    ✓ Early problem detection
    ✓ Personalized support
    ✓ Enhanced trust and rapport
    ✓ Human-centered AI
    
    USE CASES:
    • Customer support (complaint handling)
    • Mental health chatbots
    • Educational tutors (frustration detection)
    • Healthcare assistants (anxiety support)
    • HR assistants (employee concerns)
    • Social companions
    • Crisis intervention
    • Coaching and counseling
    
    EMOTION CATEGORIES:
    → Primary: Joy, sadness, anger, fear, surprise
    → Complex: Frustration, excitement, confusion
    → Social: Gratitude, embarrassment, pride
    → Cognitive: Confidence, uncertainty, curiosity
    
    DETECTION SIGNALS:
    • Explicit expressions ("I'm frustrated")
    • Exclamation marks (!!!)
    • Capitalization (ALL CAPS)
    • Emotional vocabulary
    • Sentence structure
    • Emoticons/emoji
    • Negation patterns
    
    RESPONSE STRATEGIES:
    1. VALIDATE: Acknowledge feelings
    2. EMPATHIZE: Show understanding
    3. SUPPORT: Offer help
    4. DE-ESCALATE: Calm emotions
    5. CELEBRATE: Share positive feelings
    6. CLARIFY: Address confusion
    7. ENCOURAGE: Provide motivation
    
    BEST PRACTICES:
    1. Use multiple detection methods (rule + LLM)
    2. Track emotional patterns over time
    3. Adjust response to intensity
    4. Respect cultural differences
    5. Avoid over-interpretation
    6. Maintain authenticity
    7. Provide human escalation for crises
    8. Log emotional data securely
    
    TRADE-OFFS:
    • Accuracy vs. speed
    • Empathy vs. over-familiarity
    • Automation vs. authenticity
    • Sensitivity vs. robustness
    
    PRODUCTION CONSIDERATIONS:
    → Implement crisis detection triggers
    → Provide human escalation paths
    → Monitor emotional patterns for abuse
    → Respect privacy in emotion logging
    → Support multiple languages/cultures
    → A/B test empathetic approaches
    → Track satisfaction by emotional state
    → Enable emotion detection toggles
    → Comply with mental health regulations
    → Train on diverse emotional expressions
    
    This pattern enables agents to be emotionally intelligent, responding
    appropriately to user feelings and building stronger, more human-like
    connections through empathy and understanding.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_emotion_recognition()
