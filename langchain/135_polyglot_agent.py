"""
Pattern 135: Polyglot Agent

Description:
    The Polyglot Agent pattern creates agents capable of understanding and
    generating content in multiple languages. Unlike single-language agents,
    polyglot agents can seamlessly switch between languages, translate content,
    detect language automatically, and maintain context across language boundaries.
    They understand cultural nuances, idioms, and context-specific language usage.
    
    Polyglot agents are essential for global applications where users speak different
    languages, content needs translation, or multilingual support is required. They
    go beyond simple translation to understand context, maintain consistency across
    languages, and adapt tone appropriately for different cultures.
    
    This pattern enables truly global AI applications that can serve users in their
    native languages while maintaining quality and coherence across all interactions.

Key Components:
    1. Language Detection: Automatic language identification
    2. Translation Engine: High-quality translation
    3. Context Preservation: Maintain meaning across languages
    4. Cultural Adaptation: Culturally appropriate responses
    5. Language Memory: Track language preferences
    6. Multilingual Knowledge: Language-specific information
    7. Quality Validation: Translation accuracy checks

Language Capabilities:
    1. Detection: Identify input language
    2. Translation: Convert between languages
    3. Generation: Create content in any language
    4. Code-Switching: Handle mixed-language input
    5. Localization: Adapt to regional variants
    6. Transcreation: Culturally adapt content
    7. Validation: Check translation quality

Supported Features:
    1. Multiple Languages: Support for many languages
    2. Automatic Detection: Identify language without prompting
    3. Context Awareness: Understand conversation flow
    4. Cultural Sensitivity: Respect cultural differences
    5. Consistent Terminology: Maintain vocabulary across languages
    6. Bidirectional: Handle any language pair
    7. Real-time: Fast language processing

Use Cases:
    - Global customer support
    - Multilingual content generation
    - International e-commerce
    - Language learning applications
    - Global documentation
    - Multilingual chatbots
    - International business communication
    - Content localization

Advantages:
    - Global reach
    - Better user experience
    - Market expansion
    - Cultural inclusivity
    - Reduced translation costs
    - Real-time multilingual support
    - Consistent brand voice

Challenges:
    - Translation accuracy
    - Cultural nuances
    - Idiom handling
    - Context preservation
    - Language model quality
    - Regional variations
    - Technical terminology

LangChain Implementation:
    This implementation uses LangChain for:
    - Language detection
    - Translation chains
    - Multilingual generation
    - Context tracking
    
Production Considerations:
    - Support major languages first
    - Provide human translation fallback
    - Track translation quality
    - Handle rare languages gracefully
    - Cache common translations
    - Monitor language coverage
    - Respect regional preferences
    - Enable language selection override
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE = "zh"
    ARABIC = "ar"
    HINDI = "hi"
    UNKNOWN = "unknown"


@dataclass
class LanguagePreference:
    """
    User language preferences.
    
    Attributes:
        primary_language: Main language
        secondary_languages: Alternative languages
        regional_variant: Regional dialect (e.g., en-US, en-GB)
        formality_level: Formal or informal
    """
    primary_language: Language
    secondary_languages: List[Language] = field(default_factory=list)
    regional_variant: Optional[str] = None
    formality_level: str = "neutral"


@dataclass
class MultilingualMessage:
    """
    Message with language information.
    
    Attributes:
        message_id: Unique identifier
        content: Message content
        detected_language: Detected language
        target_language: Desired output language
        translation: Translated content if applicable
        confidence: Language detection confidence
        timestamp: When created
    """
    message_id: str
    content: str
    detected_language: Language
    target_language: Optional[Language] = None
    translation: Optional[str] = None
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class PolyglotAgent:
    """
    Agent with multilingual capabilities.
    
    This agent can understand and generate content in multiple
    languages with context preservation and cultural awareness.
    """
    
    def __init__(self, temperature: float = 0.5):
        """
        Initialize polyglot agent.
        
        Args:
            temperature: LLM temperature
        """
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.conversation_history: List[MultilingualMessage] = []
        self.message_counter = 0
        self.user_preferences: Dict[str, LanguagePreference] = {}
        
        # Common phrases in different languages for detection
        self.language_patterns = {
            Language.ENGLISH: ["hello", "thank you", "please", "the", "is", "are"],
            Language.SPANISH: ["hola", "gracias", "por favor", "el", "la", "es"],
            Language.FRENCH: ["bonjour", "merci", "s'il vous plaît", "le", "la", "est"],
            Language.GERMAN: ["hallo", "danke", "bitte", "der", "die", "ist"],
            Language.ITALIAN: ["ciao", "grazie", "per favore", "il", "la", "è"],
            Language.PORTUGUESE: ["olá", "obrigado", "por favor", "o", "a", "é"],
        }
    
    def detect_language(self, text: str) -> Tuple[Language, float]:
        """
        Detect language of text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language and confidence
        """
        text_lower = text.lower()
        
        # Simple pattern matching for demo
        language_scores = {}
        for language, patterns in self.language_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                language_scores[language] = score
        
        if language_scores:
            detected = max(language_scores, key=language_scores.get)
            max_score = language_scores[detected]
            confidence = min(0.9, max_score / len(self.language_patterns[detected]))
            return detected, confidence
        
        # Fallback to LLM detection
        prompt = ChatPromptTemplate.from_template(
            "Detect the language of this text: \"{text}\"\n\n"
            "Respond with just the language code (en, es, fr, de, it, pt, ru, ja, ko, zh, ar, hi)"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"text": text})
        
        # Map result to Language enum
        code = result.strip().lower()[:2]
        for lang in Language:
            if lang.value == code:
                return lang, 0.7
        
        return Language.UNKNOWN, 0.3
    
    def translate(
        self,
        text: str,
        source_language: Language,
        target_language: Language,
        context: Optional[str] = None
    ) -> str:
        """
        Translate text between languages.
        
        Args:
            text: Text to translate
            source_language: Source language
            target_language: Target language
            context: Optional context
            
        Returns:
            Translated text
        """
        if source_language == target_language:
            return text
        
        prompt_template = (
            "Translate the following text from {source} to {target}.\n"
        )
        
        if context:
            prompt_template += "Context: {context}\n"
        
        prompt_template += (
            "Maintain the tone, meaning, and cultural appropriateness.\n\n"
            "Text: {text}\n\n"
            "Translation:"
        )
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        input_data = {
            "text": text,
            "source": source_language.name,
            "target": target_language.name
        }
        
        if context:
            input_data["context"] = context
        
        translation = chain.invoke(input_data)
        return translation.strip()
    
    def respond(
        self,
        user_input: str,
        target_language: Optional[Language] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Respond to user in appropriate language.
        
        Args:
            user_input: User's message
            target_language: Desired response language (auto-detect if None)
            user_id: User identifier for preferences
            
        Returns:
            Response in target language
        """
        # Detect input language
        detected_lang, confidence = self.detect_language(user_input)
        
        # Determine target language
        if target_language is None:
            # Check user preferences
            if user_id and user_id in self.user_preferences:
                target_language = self.user_preferences[user_id].primary_language
            else:
                # Default to input language
                target_language = detected_lang
        
        # Store message
        self.message_counter += 1
        message = MultilingualMessage(
            message_id=f"msg_{self.message_counter}",
            content=user_input,
            detected_language=detected_lang,
            target_language=target_language,
            confidence=confidence
        )
        self.conversation_history.append(message)
        
        # Generate response
        prompt = ChatPromptTemplate.from_template(
            "Respond to this message in {target_language}:\n\n"
            "User message: {user_input}\n"
            "(Detected language: {detected_language})\n\n"
            "Provide a helpful response in {target_language}, maintaining "
            "appropriate cultural context and tone."
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "user_input": user_input,
            "target_language": target_language.name,
            "detected_language": detected_lang.name
        })
        
        return response
    
    def set_user_preference(self, user_id: str, preference: LanguagePreference):
        """Set language preferences for a user."""
        self.user_preferences[user_id] = preference
    
    def handle_mixed_language(self, text: str) -> Dict[str, Any]:
        """
        Handle text with multiple languages (code-switching).
        
        Args:
            text: Mixed-language text
            
        Returns:
            Analysis of language segments
        """
        prompt = ChatPromptTemplate.from_template(
            "This text contains multiple languages:\n\n"
            "\"{text}\"\n\n"
            "Identify the languages used and provide:\n"
            "1. List of languages detected\n"
            "2. Dominant language\n"
            "3. Translation to English if needed"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        analysis = chain.invoke({"text": text})
        
        return {
            "original_text": text,
            "analysis": analysis
        }
    
    def localize_content(
        self,
        content: str,
        target_language: Language,
        region: Optional[str] = None
    ) -> str:
        """
        Localize content for specific language and region.
        
        Args:
            content: Content to localize
            target_language: Target language
            region: Regional variant (e.g., US, UK, BR)
            
        Returns:
            Localized content
        """
        prompt_template = (
            "Localize this content for {target_language}"
        )
        
        if region:
            prompt_template += " ({region} region)"
        
        prompt_template += (
            ".\n\n"
            "Original content:\n"
            "{content}\n\n"
            "Adapt idioms, cultural references, units, dates, and formats "
            "to be appropriate for the target audience.\n\n"
            "Localized content:"
        )
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        
        input_data = {
            "content": content,
            "target_language": target_language.name
        }
        
        if region:
            input_data["region"] = region
        
        localized = chain.invoke(input_data)
        return localized.strip()
    
    def get_language_statistics(self) -> Dict[str, Any]:
        """Get statistics about language usage."""
        if not self.conversation_history:
            return {
                "total_messages": 0,
                "languages_detected": [],
                "most_common": None
            }
        
        language_counts = {}
        for msg in self.conversation_history:
            lang = msg.detected_language
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        most_common = max(language_counts, key=language_counts.get) if language_counts else None
        
        return {
            "total_messages": len(self.conversation_history),
            "languages_detected": list(language_counts.keys()),
            "language_distribution": {lang.name: count for lang, count in language_counts.items()},
            "most_common": most_common.name if most_common else None
        }


def demonstrate_polyglot():
    """Demonstrate polyglot agent pattern."""
    
    print("=" * 80)
    print("POLYGLOT AGENT PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = PolyglotAgent()
    
    # Example 1: Language detection
    print("\n" + "=" * 80)
    print("Example 1: Automatic Language Detection")
    print("=" * 80)
    
    test_phrases = [
        "Hello, how are you?",
        "Hola, ¿cómo estás?",
        "Bonjour, comment allez-vous?",
        "Hallo, wie geht es dir?",
        "Ciao, come stai?"
    ]
    
    print("\nDetecting languages:")
    for phrase in test_phrases:
        detected, confidence = agent.detect_language(phrase)
        print(f'\n"{phrase}"')
        print(f"  → Language: {detected.name} (confidence: {confidence:.2f})")
    
    # Example 2: Translation
    print("\n" + "=" * 80)
    print("Example 2: Translation Between Languages")
    print("=" * 80)
    
    original = "I love learning new languages!"
    source = Language.ENGLISH
    targets = [Language.SPANISH, Language.FRENCH, Language.GERMAN]
    
    print(f"\nOriginal ({source.name}): {original}")
    print("\nTranslations:")
    
    for target in targets:
        translation = agent.translate(original, source, target)
        print(f"  {target.name}: {translation}")
    
    # Example 3: Multilingual conversation
    print("\n" + "=" * 80)
    print("Example 3: Multilingual Conversation")
    print("=" * 80)
    
    print("\nConversation with automatic language switching:")
    
    messages = [
        ("Hello! Can you help me?", None),
        ("Sí, ¿en qué puedo ayudarte?", Language.SPANISH),
        ("Thanks! Do you speak French too?", Language.ENGLISH)
    ]
    
    for msg, target_lang in messages:
        detected, conf = agent.detect_language(msg)
        print(f'\nUser ({detected.name}): {msg}')
        response = agent.respond(msg, target_language=target_lang)
        print(f'Agent: {response}')
    
    # Example 4: User language preferences
    print("\n" + "=" * 80)
    print("Example 4: User Language Preferences")
    print("=" * 80)
    
    # Set preference for user
    user_pref = LanguagePreference(
        primary_language=Language.SPANISH,
        secondary_languages=[Language.ENGLISH],
        regional_variant="es-MX",
        formality_level="formal"
    )
    
    agent.set_user_preference("user123", user_pref)
    
    print("\nUser Preference Set:")
    print(f"  Primary: {user_pref.primary_language.name}")
    print(f"  Variant: {user_pref.regional_variant}")
    print(f"  Formality: {user_pref.formality_level}")
    
    print("\nUser sends message in English:")
    user_msg = "What's the weather like?"
    print(f'User: {user_msg}')
    
    response = agent.respond(user_msg, user_id="user123")
    print(f'Agent (responds in Spanish): {response}')
    
    # Example 5: Code-switching handling
    print("\n" + "=" * 80)
    print("Example 5: Handling Code-Switching")
    print("=" * 80)
    
    mixed_text = "I went to the mercado to buy some légumes and Brot"
    
    print(f'\nMixed-language text: "{mixed_text}"')
    print("(English + Spanish + French + German)")
    
    analysis = agent.handle_mixed_language(mixed_text)
    print("\nAnalysis:")
    print(analysis["analysis"])
    
    # Example 6: Content localization
    print("\n" + "=" * 80)
    print("Example 6: Content Localization")
    print("=" * 80)
    
    us_content = "The conference starts at 9 AM on 12/25. Temperature will be 72°F."
    
    print(f"\nOriginal (US English):")
    print(f"  {us_content}")
    
    print("\nLocalized versions:")
    
    # UK English
    uk_version = agent.localize_content(us_content, Language.ENGLISH, region="UK")
    print(f"\nUK English:")
    print(f"  {uk_version}")
    
    # Spanish (Mexico)
    mx_version = agent.localize_content(us_content, Language.SPANISH, region="MX")
    print(f"\nSpanish (Mexico):")
    print(f"  {mx_version}")
    
    # Example 7: Multilingual customer support
    print("\n" + "=" * 80)
    print("Example 7: Multilingual Customer Support")
    print("=" * 80)
    
    support_queries = [
        ("My order hasn't arrived yet", Language.ENGLISH),
        ("¿Dónde está mi pedido?", Language.SPANISH),
        ("Je n'ai pas reçu ma commande", Language.FRENCH)
    ]
    
    print("\nHandling support queries in multiple languages:")
    
    for query, response_lang in support_queries:
        detected, _ = agent.detect_language(query)
        print(f'\n[{detected.name}] Customer: {query}')
        response = agent.respond(query, target_language=response_lang)
        print(f'[{response_lang.name}] Agent: {response}')
    
    # Example 8: Language statistics
    print("\n" + "=" * 80)
    print("Example 8: Language Usage Statistics")
    print("=" * 80)
    
    stats = agent.get_language_statistics()
    
    print("\nLANGUAGE STATISTICS:")
    print("=" * 60)
    print(f"Total Messages: {stats['total_messages']}")
    print(f"\nLanguages Detected: {', '.join([lang.name for lang in stats['languages_detected']])}")
    
    if stats.get('language_distribution'):
        print("\nDistribution:")
        for lang, count in stats['language_distribution'].items():
            percentage = (count / stats['total_messages']) * 100
            print(f"  {lang}: {count} ({percentage:.1f}%)")
    
    print(f"\nMost Common: {stats['most_common']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Polyglot Agent Pattern")
    print("=" * 80)
    
    summary = """
    The Polyglot Agent pattern demonstrated:
    
    1. LANGUAGE DETECTION (Example 1):
       - Automatic identification
       - Confidence scoring
       - Multiple language support
       - Pattern recognition
       - Fallback mechanisms
    
    2. TRANSLATION (Example 2):
       - High-quality conversion
       - Context preservation
       - Multiple target languages
       - Tone maintenance
       - Cultural awareness
    
    3. MULTILINGUAL CONVERSATION (Example 3):
       - Seamless language switching
       - Context preservation
       - Natural flow
       - Automatic adaptation
       - Real-time processing
    
    4. USER PREFERENCES (Example 4):
       - Language preference storage
       - Regional variants
       - Formality levels
       - Automatic application
       - Personalization
    
    5. CODE-SWITCHING (Example 5):
       - Mixed-language handling
       - Language segment identification
       - Dominant language detection
       - Comprehensive translation
       - Context understanding
    
    6. LOCALIZATION (Example 6):
       - Regional adaptation
       - Cultural appropriateness
       - Unit conversion
       - Date format adjustment
       - Idiomatic expressions
    
    7. CUSTOMER SUPPORT (Example 7):
       - Multilingual queries
       - Appropriate responses
       - Language matching
       - Service quality
       - Global reach
    
    8. USAGE ANALYTICS (Example 8):
       - Language distribution
       - Usage patterns
       - Statistics tracking
       - Popular languages
       - Coverage analysis
    
    KEY BENEFITS:
    ✓ Global reach
    ✓ Better user experience
    ✓ Market expansion
    ✓ Cultural inclusivity
    ✓ Reduced translation costs
    ✓ Real-time support
    ✓ Consistent quality
    ✓ Wider accessibility
    
    USE CASES:
    • Global customer support
    • Multilingual content generation
    • International e-commerce
    • Language learning apps
    • Global documentation
    • Multilingual chatbots
    • Business communication
    • Content localization
    
    LANGUAGE CAPABILITIES:
    → Detection: Identify input language
    → Translation: Convert between languages
    → Generation: Create multilingual content
    → Code-Switching: Handle mixed languages
    → Localization: Regional adaptation
    → Transcreation: Cultural adaptation
    → Validation: Quality checking
    
    SUPPORTED FEATURES:
    • Multiple languages (10+)
    • Automatic detection
    • Context awareness
    • Cultural sensitivity
    • Consistent terminology
    • Bidirectional translation
    • Real-time processing
    
    BEST PRACTICES:
    1. Support major languages first
    2. Provide human translation fallback
    3. Track translation quality
    4. Handle rare languages gracefully
    5. Cache common translations
    6. Monitor language coverage
    7. Respect regional preferences
    8. Enable language override
    
    TRADE-OFFS:
    • Coverage vs. quality
    • Speed vs. accuracy
    • Automation vs. human review
    • Generalization vs. specialization
    
    PRODUCTION CONSIDERATIONS:
    → Prioritize high-traffic languages
    → Implement quality monitoring
    → Cache frequent translations
    → Provide confidence scores
    → Enable human escalation
    → Track accuracy metrics
    → Support regional variants
    → Handle cultural sensitivities
    → Maintain terminology consistency
    → Update language models regularly
    
    This pattern enables truly global AI applications that can serve users
    in their native languages while maintaining quality and cultural
    appropriateness across all interactions.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_polyglot()
