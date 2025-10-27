"""
Pattern 129: Emotion Recognition Agent

This pattern implements sentiment analysis, emotion detection,
and empathetic response generation for emotionally-aware interactions.

Use Cases:
- Customer service
- Mental health support
- User experience optimization
- Conflict resolution
- Personalized interactions

Category: Dialogue & Interaction (3/4 = 75%)
Complexity: Advanced
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
import re


class EmotionType(Enum):
    """Primary emotion types (Ekman's basic emotions + extended)."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


class Sentiment(Enum):
    """Sentiment polarity."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class EmotionalIntensity(Enum):
    """Intensity of emotion."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class EmpathyLevel(Enum):
    """Level of empathy in response."""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class EmotionDetection:
    """Detected emotion in text."""
    emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    confidence: float
    indicators: List[str]  # Words/phrases that indicated this emotion


@dataclass
class SentimentAnalysis:
    """Sentiment analysis result."""
    sentiment: Sentiment
    polarity_score: float  # -1.0 to 1.0
    subjectivity: float  # 0.0 to 1.0
    confidence: float


@dataclass
class EmotionalState:
    """Complete emotional state of user."""
    timestamp: datetime
    primary_emotion: EmotionType
    secondary_emotions: List[EmotionType]
    sentiment: Sentiment
    intensity: EmotionalIntensity
    emotional_trajectory: str  # "stable", "escalating", "de-escalating"


@dataclass
class EmpathicResponse:
    """Empathetic response to user emotion."""
    response_text: str
    empathy_level: EmpathyLevel
    addresses_emotion: EmotionType
    validation_included: bool
    support_offered: bool


class EmotionLexicon:
    """Lexicon of emotion-indicating words."""
    
    def __init__(self):
        self.emotion_keywords = {
            EmotionType.JOY: {
                'happy', 'glad', 'joyful', 'delighted', 'pleased', 'wonderful',
                'excited', 'thrilled', 'great', 'awesome', 'fantastic', 'love',
                'enjoy', 'appreciate', 'excellent'
            },
            EmotionType.SADNESS: {
                'sad', 'unhappy', 'depressed', 'miserable', 'disappointed',
                'upset', 'down', 'blue', 'sorry', 'regret', 'miss', 'cry',
                'unfortunate', 'terrible'
            },
            EmotionType.ANGER: {
                'angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated',
                'outraged', 'hate', 'terrible', 'awful', 'unacceptable',
                'disgusting', 'worst'
            },
            EmotionType.FEAR: {
                'afraid', 'scared', 'worried', 'anxious', 'nervous', 'terrified',
                'concerned', 'fear', 'panic', 'dread', 'uncertain', 'insecure'
            },
            EmotionType.SURPRISE: {
                'surprised', 'shocked', 'amazed', 'astonished', 'unexpected',
                'wow', 'incredible', 'unbelievable', 'sudden'
            },
            EmotionType.DISGUST: {
                'disgusted', 'revolted', 'sick', 'gross', 'nasty', 'appalled',
                'repulsive', 'awful'
            },
            EmotionType.TRUST: {
                'trust', 'believe', 'confident', 'reliable', 'depend', 'faith',
                'secure', 'certain', 'sure'
            },
            EmotionType.ANTICIPATION: {
                'hope', 'expect', 'anticipate', 'looking forward', 'eager',
                'ready', 'prepare', 'await', 'soon'
            }
        }
        
        # Sentiment words
        self.positive_words = {
            'good', 'great', 'excellent', 'wonderful', 'fantastic', 'amazing',
            'love', 'best', 'perfect', 'beautiful', 'nice', 'helpful',
            'thank', 'appreciate', 'satisfied'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'disappointed', 'poor', 'useless', 'waste', 'fail', 'problem',
            'issue', 'complaint', 'wrong'
        }
        
        # Intensity modifiers
        self.intensifiers = {
            'very', 'extremely', 'absolutely', 'totally', 'completely',
            'really', 'so', 'incredibly', 'utterly', 'quite'
        }
        
        self.diminishers = {
            'slightly', 'somewhat', 'a bit', 'little', 'barely',
            'hardly', 'kind of', 'sort of'
        }


class EmotionDetector:
    """Detects emotions in text."""
    
    def __init__(self):
        self.lexicon = EmotionLexicon()
    
    def detect_emotions(self, text: str) -> List[EmotionDetection]:
        """Detect all emotions present in text."""
        words = text.lower().split()
        emotions = []
        
        for emotion_type, keywords in self.lexicon.emotion_keywords.items():
            matches = []
            base_intensity = 0.0
            
            # Find matching keywords
            for i, word in enumerate(words):
                word_clean = re.sub(r'[^\w\s]', '', word)
                if word_clean in keywords:
                    matches.append(word)
                    
                    # Check for intensifiers/diminishers
                    intensity_modifier = 1.0
                    if i > 0:
                        prev_word = words[i-1]
                        if prev_word in self.lexicon.intensifiers:
                            intensity_modifier = 1.5
                        elif prev_word in self.lexicon.diminishers:
                            intensity_modifier = 0.5
                    
                    base_intensity += intensity_modifier
            
            if matches:
                # Calculate normalized intensity
                intensity = min(1.0, base_intensity / 3.0)
                confidence = min(1.0, len(matches) / 2.0)
                
                emotions.append(EmotionDetection(
                    emotion=emotion_type,
                    intensity=intensity,
                    confidence=confidence,
                    indicators=matches
                ))
        
        # Sort by intensity
        emotions.sort(key=lambda e: e.intensity, reverse=True)
        
        # If no emotions detected, return neutral
        if not emotions:
            emotions.append(EmotionDetection(
                emotion=EmotionType.NEUTRAL,
                intensity=0.5,
                confidence=0.8,
                indicators=[]
            ))
        
        return emotions
    
    def get_primary_emotion(self, text: str) -> EmotionDetection:
        """Get primary (strongest) emotion."""
        emotions = self.detect_emotions(text)
        return emotions[0] if emotions else EmotionDetection(
            emotion=EmotionType.NEUTRAL,
            intensity=0.5,
            confidence=0.8,
            indicators=[]
        )


class SentimentAnalyzer:
    """Analyzes sentiment of text."""
    
    def __init__(self):
        self.lexicon = EmotionLexicon()
    
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Analyze sentiment of text."""
        words = text.lower().split()
        
        # Count positive and negative words
        positive_count = 0
        negative_count = 0
        
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^\w\s]', '', word)
            
            # Check for negation
            is_negated = False
            if i > 0 and words[i-1] in ['not', 'no', 'never', "n't", 'dont', "don't"]:
                is_negated = True
            
            if word_clean in self.lexicon.positive_words:
                if is_negated:
                    negative_count += 1
                else:
                    positive_count += 1
            
            elif word_clean in self.lexicon.negative_words:
                if is_negated:
                    positive_count += 1
                else:
                    negative_count += 1
        
        # Calculate polarity score
        total = positive_count + negative_count
        if total == 0:
            polarity_score = 0.0
        else:
            polarity_score = (positive_count - negative_count) / total
        
        # Determine sentiment
        if polarity_score <= -0.6:
            sentiment = Sentiment.VERY_NEGATIVE
        elif polarity_score <= -0.2:
            sentiment = Sentiment.NEGATIVE
        elif polarity_score <= 0.2:
            sentiment = Sentiment.NEUTRAL
        elif polarity_score <= 0.6:
            sentiment = Sentiment.POSITIVE
        else:
            sentiment = Sentiment.VERY_POSITIVE
        
        # Calculate subjectivity (opinion vs fact)
        opinion_indicators = ['feel', 'think', 'believe', 'seem', 'should', 'must']
        subjectivity = min(1.0, sum(1 for word in words if word in opinion_indicators) / max(1, len(words) / 10))
        
        # Confidence based on number of sentiment words
        confidence = min(1.0, total / max(1, len(words) / 5))
        
        return SentimentAnalysis(
            sentiment=sentiment,
            polarity_score=polarity_score,
            subjectivity=subjectivity,
            confidence=max(0.5, confidence)
        )


class EmotionalStateTracker:
    """Tracks emotional state over time."""
    
    def __init__(self):
        self.state_history: List[EmotionalState] = []
    
    def track_state(
        self,
        primary_emotion: EmotionType,
        secondary_emotions: List[EmotionType],
        sentiment: Sentiment,
        intensity: float
    ) -> EmotionalState:
        """Track new emotional state."""
        # Determine intensity level
        if intensity < 0.2:
            intensity_level = EmotionalIntensity.VERY_LOW
        elif intensity < 0.4:
            intensity_level = EmotionalIntensity.LOW
        elif intensity < 0.6:
            intensity_level = EmotionalIntensity.MEDIUM
        elif intensity < 0.8:
            intensity_level = EmotionalIntensity.HIGH
        else:
            intensity_level = EmotionalIntensity.VERY_HIGH
        
        # Determine trajectory
        trajectory = "stable"
        if len(self.state_history) >= 2:
            prev_intensity = self._get_emotion_intensity(self.state_history[-1].primary_emotion)
            curr_intensity = intensity
            
            if curr_intensity > prev_intensity + 0.2:
                trajectory = "escalating"
            elif curr_intensity < prev_intensity - 0.2:
                trajectory = "de-escalating"
        
        state = EmotionalState(
            timestamp=datetime.now(),
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            sentiment=sentiment,
            intensity=intensity_level,
            emotional_trajectory=trajectory
        )
        
        self.state_history.append(state)
        return state
    
    def _get_emotion_intensity(self, emotion: EmotionType) -> float:
        """Get intensity for emotion type (simplified)."""
        negative_emotions = {EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR, EmotionType.DISGUST}
        if emotion in negative_emotions:
            return 0.7
        elif emotion == EmotionType.JOY:
            return 0.8
        else:
            return 0.5
    
    def get_trajectory_summary(self) -> str:
        """Get summary of emotional trajectory."""
        if not self.state_history:
            return "No emotional history"
        
        recent_states = self.state_history[-5:]
        emotions = [s.primary_emotion.value for s in recent_states]
        trajectory = recent_states[-1].emotional_trajectory
        
        return f"Recent emotions: {' → '.join(emotions)} (trend: {trajectory})"


class EmpathyGenerator:
    """Generates empathetic responses."""
    
    def __init__(self):
        self.response_templates = {
            EmotionType.JOY: {
                EmpathyLevel.MODERATE: [
                    "That's great to hear!",
                    "I'm glad you're feeling positive about this."
                ],
                EmpathyLevel.HIGH: [
                    "That's wonderful! I'm really happy for you.",
                    "How exciting! Your joy is evident and well-deserved."
                ]
            },
            EmotionType.SADNESS: {
                EmpathyLevel.MODERATE: [
                    "I understand this is difficult for you.",
                    "I'm sorry you're going through this."
                ],
                EmpathyLevel.HIGH: [
                    "I can see this is really affecting you, and I want you to know that your feelings are completely valid.",
                    "This sounds genuinely difficult. I'm here to support you through this."
                ]
            },
            EmotionType.ANGER: {
                EmpathyLevel.MODERATE: [
                    "I can see why you're frustrated.",
                    "Your frustration is understandable."
                ],
                EmpathyLevel.HIGH: [
                    "I completely understand your frustration, and your feelings are absolutely valid. Let's work together to address this.",
                    "You have every right to feel upset about this situation. I'm committed to helping resolve this for you."
                ]
            },
            EmotionType.FEAR: {
                EmpathyLevel.MODERATE: [
                    "I understand your concerns.",
                    "It's natural to feel uncertain about this."
                ],
                EmpathyLevel.HIGH: [
                    "I hear your concerns, and they're completely valid. Let me help address what's worrying you.",
                    "Your worries are understandable. I'm here to provide clarity and support."
                ]
            },
            EmotionType.NEUTRAL: {
                EmpathyLevel.MODERATE: [
                    "I appreciate you sharing this with me.",
                    "Thank you for your message."
                ]
            }
        }
        
        self.validation_phrases = [
            "Your feelings are valid",
            "I hear what you're saying",
            "I understand where you're coming from",
            "That makes complete sense"
        ]
        
        self.support_phrases = [
            "I'm here to help",
            "Let's work through this together",
            "I'll do my best to assist you",
            "We'll figure this out"
        ]
    
    def generate_empathic_response(
        self,
        emotion: EmotionType,
        intensity: float,
        context: str = ""
    ) -> EmpathicResponse:
        """Generate empathetic response."""
        # Determine empathy level based on intensity
        if intensity > 0.7:
            empathy_level = EmpathyLevel.VERY_HIGH
        elif intensity > 0.5:
            empathy_level = EmpathyLevel.HIGH
        else:
            empathy_level = EmpathyLevel.MODERATE
        
        # Get template
        templates = self.response_templates.get(emotion, {}).get(empathy_level, [])
        
        if not templates:
            # Fallback to moderate level or default
            templates = self.response_templates.get(emotion, {}).get(EmpathyLevel.MODERATE, [
                "I understand.",
                "Thank you for sharing."
            ])
        
        response_text = templates[0] if templates else "I hear you."
        
        # Add validation for negative emotions
        validation_included = False
        if emotion in [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR] and intensity > 0.5:
            response_text += f" {self.validation_phrases[0]}."
            validation_included = True
        
        # Add support offer for high intensity
        support_offered = False
        if intensity > 0.6:
            response_text += f" {self.support_phrases[0]}."
            support_offered = True
        
        return EmpathicResponse(
            response_text=response_text,
            empathy_level=empathy_level,
            addresses_emotion=emotion,
            validation_included=validation_included,
            support_offered=support_offered
        )


class EmotionRecognitionAgent:
    """Agent for emotion recognition and empathetic response."""
    
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.state_tracker = EmotionalStateTracker()
        self.empathy_generator = EmpathyGenerator()
        self.interaction_count = 0
    
    def process_message(self, text: str) -> Tuple[EmotionalState, EmpathicResponse]:
        """Process message and generate empathetic response."""
        self.interaction_count += 1
        
        # Detect emotions
        emotions = self.emotion_detector.detect_emotions(text)
        primary_emotion = emotions[0]
        secondary_emotions = [e.emotion for e in emotions[1:3]]
        
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze_sentiment(text)
        
        # Track emotional state
        state = self.state_tracker.track_state(
            primary_emotion.emotion,
            secondary_emotions,
            sentiment.sentiment,
            primary_emotion.intensity
        )
        
        # Generate empathetic response
        response = self.empathy_generator.generate_empathic_response(
            primary_emotion.emotion,
            primary_emotion.intensity,
            text
        )
        
        return state, response
    
    def get_emotion_report(self, text: str) -> str:
        """Get detailed emotion analysis report."""
        emotions = self.emotion_detector.detect_emotions(text)
        sentiment = self.sentiment_analyzer.analyze_sentiment(text)
        
        lines = [
            f"Emotion Analysis Report",
            f"=" * 60,
            f"Text: {text}",
            f"\nSentiment: {sentiment.sentiment.value}",
            f"Polarity: {sentiment.polarity_score:.2f}",
            f"Subjectivity: {sentiment.subjectivity:.2f}",
            f"Confidence: {sentiment.confidence:.2f}",
            f"\nDetected Emotions:"
        ]
        
        for i, emotion in enumerate(emotions[:3], 1):
            lines.append(f"  {i}. {emotion.emotion.value.upper()}")
            lines.append(f"     Intensity: {emotion.intensity:.2f}")
            lines.append(f"     Confidence: {emotion.confidence:.2f}")
            if emotion.indicators:
                lines.append(f"     Indicators: {', '.join(emotion.indicators)}")
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get emotion recognition statistics."""
        if not self.state_tracker.state_history:
            return {
                'total_interactions': self.interaction_count,
                'emotions_tracked': 0
            }
        
        emotion_counts = {}
        for state in self.state_tracker.state_history:
            emotion = state.primary_emotion.value
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        most_common = max(emotion_counts.items(), key=lambda x: x[1]) if emotion_counts else ('neutral', 0)
        
        return {
            'total_interactions': self.interaction_count,
            'emotions_tracked': len(self.state_tracker.state_history),
            'unique_emotions': len(emotion_counts),
            'most_common_emotion': most_common[0],
            'emotion_distribution': emotion_counts,
            'trajectory': self.state_tracker.get_trajectory_summary()
        }


def demonstrate_emotion_recognition():
    """Demonstrate the Emotion Recognition Agent."""
    print("=" * 60)
    print("Emotion Recognition Agent Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = EmotionRecognitionAgent()
    
    print("\n1. ANALYZING VARIOUS EMOTIONS")
    print("-" * 60)
    
    test_messages = [
        "I'm so happy with your service! Everything is wonderful!",
        "I'm extremely frustrated and angry about this delay.",
        "I'm worried that this might not work out.",
        "Thank you so much! This is exactly what I needed.",
        "This is absolutely terrible and unacceptable.",
    ]
    
    for message in test_messages:
        print(f"\nUser: {message}")
        state, response = agent.process_message(message)
        
        print(f"  Emotion: {state.primary_emotion.value} ({state.intensity.value})")
        print(f"  Sentiment: {state.sentiment.value}")
        print(f"  Agent Response: {response.response_text}")
        print(f"  Empathy Level: {response.empathy_level.value}")
    
    print("\n\n2. DETAILED EMOTION ANALYSIS")
    print("-" * 60)
    
    analysis_text = "I'm really disappointed and sad about the results, but I'm also a bit hopeful things will improve."
    print(f"\n{agent.get_emotion_report(analysis_text)}")
    
    print("\n\n3. EMOTIONAL TRAJECTORY")
    print("-" * 60)
    
    trajectory_messages = [
        "I'm a bit annoyed",
        "This is getting frustrating",
        "I'm really angry now!",
        "Thank you for helping, I feel better",
        "I really appreciate your support"
    ]
    
    print("Processing conversation with emotional trajectory:")
    for msg in trajectory_messages:
        state, response = agent.process_message(msg)
        print(f"\n  User: {msg}")
        print(f"  Emotion: {state.primary_emotion.value} - Trajectory: {state.emotional_trajectory}")
        print(f"  Agent: {response.response_text}")
    
    print(f"\n\nTrajectory Summary: {agent.state_tracker.get_trajectory_summary()}")
    
    print("\n\n4. SENTIMENT POLARITY")
    print("-" * 60)
    
    polarity_tests = [
        ("This is absolutely amazing!", "Very positive"),
        ("This is okay, nothing special", "Neutral"),
        ("This is terrible and I hate it", "Very negative"),
        ("Not bad at all", "Positive with negation"),
    ]
    
    for text, label in polarity_tests:
        sentiment = agent.sentiment_analyzer.analyze_sentiment(text)
        print(f"\n  {label}:")
        print(f"  Text: {text}")
        print(f"  Sentiment: {sentiment.sentiment.value}")
        print(f"  Polarity: {sentiment.polarity_score:.2f}")
    
    print("\n\n5. EMPATHY LEVELS")
    print("-" * 60)
    
    empathy_tests = [
        (EmotionType.JOY, 0.4, "Low intensity joy"),
        (EmotionType.JOY, 0.9, "High intensity joy"),
        (EmotionType.SADNESS, 0.8, "High intensity sadness"),
        (EmotionType.ANGER, 0.9, "High intensity anger"),
    ]
    
    for emotion, intensity, description in empathy_tests:
        response = agent.empathy_generator.generate_empathic_response(emotion, intensity)
        print(f"\n  {description}:")
        print(f"  Response: {response.response_text}")
        print(f"  Empathy Level: {response.empathy_level.value}")
        print(f"  Validation: {response.validation_included}, Support: {response.support_offered}")
    
    print("\n\n6. STATISTICS")
    print("-" * 60)
    
    stats = agent.get_statistics()
    print(f"  Total Interactions: {stats['total_interactions']}")
    print(f"  Emotions Tracked: {stats['emotions_tracked']}")
    print(f"  Unique Emotions: {stats['unique_emotions']}")
    print(f"  Most Common: {stats['most_common_emotion']}")
    print(f"\n  Emotion Distribution:")
    for emotion, count in sorted(stats['emotion_distribution'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {emotion}: {count}")
    print(f"\n  {stats['trajectory']}")
    
    print("\n" + "=" * 60)
    print("🎉 Pattern 129 Complete!")
    print("Dialogue & Interaction Category: 75%")
    print("129/170 patterns implemented (75.9%)!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_emotion_recognition()
