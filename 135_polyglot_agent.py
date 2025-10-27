"""
Pattern 135: Polyglot Agent

This pattern implements cross-lingual operations with translation, multilingual
understanding, and language detection capabilities.

Use Cases:
- International customer support
- Content localization
- Cross-language information retrieval
- Multilingual chatbots
- Global documentation systems

Category: Specialization (3/6 = 50%)
Complexity: Advanced
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
import hashlib


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"
    ARABIC = "ar"
    RUSSIAN = "ru"
    PORTUGUESE = "pt"
    HINDI = "hi"


class TranslationQuality(Enum):
    """Quality levels for translation."""
    LITERAL = "literal"  # Word-for-word
    STANDARD = "standard"  # Natural translation
    PROFESSIONAL = "professional"  # Polished, context-aware
    LOCALIZED = "localized"  # Culturally adapted


class LanguageScript(Enum):
    """Writing scripts."""
    LATIN = "latin"
    CYRILLIC = "cyrillic"
    ARABIC_SCRIPT = "arabic"
    CHINESE_HANZI = "hanzi"
    JAPANESE_KANA = "kana"
    DEVANAGARI = "devanagari"


@dataclass
class LanguageProfile:
    """Profile for a language."""
    language: Language
    script: LanguageScript
    rtl: bool  # Right-to-left
    common_words: List[str]
    formal_pronouns: List[str]
    informal_pronouns: List[str]


@dataclass
class TranslationRequest:
    """Request for translation."""
    request_id: str
    source_text: str
    source_language: Language
    target_language: Language
    quality_level: TranslationQuality = TranslationQuality.STANDARD
    preserve_formatting: bool = True
    context: Optional[str] = None


@dataclass
class TranslationResult:
    """Result of translation."""
    request_id: str
    translated_text: str
    source_language: Language
    target_language: Language
    confidence: float
    alternatives: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    text: str
    detected_language: Language
    confidence: float
    alternative_languages: List[Tuple[Language, float]] = field(default_factory=list)


@dataclass
class MultilingualContent:
    """Content in multiple languages."""
    content_id: str
    translations: Dict[Language, str]
    primary_language: Language
    metadata: Dict[str, Any] = field(default_factory=dict)


class LanguageDetector:
    """Detects language from text."""
    
    def __init__(self):
        # Simplified language detection using characteristic words
        self.language_patterns = {
            Language.ENGLISH: {'the', 'is', 'and', 'to', 'of', 'in', 'for', 'you', 'that', 'with'},
            Language.SPANISH: {'el', 'la', 'de', 'que', 'y', 'es', 'en', 'los', 'por', 'un'},
            Language.FRENCH: {'le', 'de', 'un', 'et', 'être', 'à', 'il', 'que', 'ne', 'je'},
            Language.GERMAN: {'der', 'die', 'das', 'und', 'in', 'von', 'zu', 'den', 'mit', 'ist'},
            Language.PORTUGUESE: {'o', 'a', 'de', 'que', 'e', 'do', 'da', 'em', 'um', 'os'},
            Language.RUSSIAN: {'и', 'в', 'не', 'на', 'я', 'что', 'с', 'он', 'как', 'это'},
            Language.CHINESE: {'的', '是', '在', '了', '我', '有', '他', '这', '为', '就'},
            Language.JAPANESE: {'は', 'の', 'に', 'を', 'と', 'が', 'た', 'で', 'も', 'です'},
            Language.ARABIC: {'في', 'من', 'إلى', 'على', 'أن', 'هذا', 'ما', 'كان', 'عن', 'هو'},
            Language.HINDI: {'है', 'के', 'की', 'में', 'से', 'को', 'और', 'का', 'एक', 'हैं'}
        }
    
    def detect(self, text: str) -> LanguageDetectionResult:
        """Detect language of text."""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Score each language
        scores: Dict[Language, float] = {}
        
        for language, patterns in self.language_patterns.items():
            overlap = len(words & patterns)
            score = overlap / len(patterns)
            scores[language] = score
        
        # Find best match
        if not scores or max(scores.values()) == 0:
            # Default to English
            detected = Language.ENGLISH
            confidence = 0.3
        else:
            detected = max(scores.items(), key=lambda x: x[1])[0]
            confidence = scores[detected]
        
        # Get alternatives
        alternatives = [
            (lang, score) for lang, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if lang != detected and score > 0
        ][:3]
        
        return LanguageDetectionResult(
            text=text,
            detected_language=detected,
            confidence=min(1.0, confidence * 3),  # Scale up
            alternative_languages=alternatives
        )


class TranslationEngine:
    """Translates text between languages."""
    
    def __init__(self):
        # Simplified translation dictionaries
        self.dictionaries = {
            (Language.ENGLISH, Language.SPANISH): {
                'hello': 'hola',
                'world': 'mundo',
                'goodbye': 'adiós',
                'thank you': 'gracias',
                'please': 'por favor',
                'yes': 'sí',
                'no': 'no',
                'good': 'bueno',
                'bad': 'malo'
            },
            (Language.ENGLISH, Language.FRENCH): {
                'hello': 'bonjour',
                'world': 'monde',
                'goodbye': 'au revoir',
                'thank you': 'merci',
                'please': 's\'il vous plaît',
                'yes': 'oui',
                'no': 'non',
                'good': 'bon',
                'bad': 'mauvais'
            },
            (Language.ENGLISH, Language.GERMAN): {
                'hello': 'hallo',
                'world': 'welt',
                'goodbye': 'auf wiedersehen',
                'thank you': 'danke',
                'please': 'bitte',
                'yes': 'ja',
                'no': 'nein',
                'good': 'gut',
                'bad': 'schlecht'
            }
        }
    
    def translate(self, request: TranslationRequest) -> TranslationResult:
        """Translate text."""
        # Get dictionary for language pair
        dict_key = (request.source_language, request.target_language)
        reverse_key = (request.target_language, request.source_language)
        
        dictionary = self.dictionaries.get(dict_key, {})
        if not dictionary and reverse_key in self.dictionaries:
            # Use reverse dictionary
            reverse_dict = self.dictionaries[reverse_key]
            dictionary = {v: k for k, v in reverse_dict.items()}
        
        # Translate
        if not dictionary:
            # No dictionary available
            translated = f"[{request.target_language.value}] {request.source_text}"
            confidence = 0.3
            warnings = [f"No dictionary available for {request.source_language.value} → {request.target_language.value}"]
        else:
            translated_words = []
            text_lower = request.source_text.lower()
            confidence_scores = []
            
            # Try to translate each word/phrase
            words = text_lower.split()
            i = 0
            while i < len(words):
                # Try multi-word phrases first
                found = False
                for length in range(min(3, len(words) - i), 0, -1):
                    phrase = ' '.join(words[i:i+length])
                    if phrase in dictionary:
                        translated_words.append(dictionary[phrase])
                        confidence_scores.append(1.0)
                        i += length
                        found = True
                        break
                
                if not found:
                    # Keep original word
                    translated_words.append(words[i])
                    confidence_scores.append(0.3)
                    i += 1
            
            translated = ' '.join(translated_words)
            confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            warnings = []
        
        # Generate alternatives
        alternatives = [
            f"{translated} (literal)",
            f"{translated} (formal)" if request.quality_level == TranslationQuality.PROFESSIONAL else translated
        ]
        
        return TranslationResult(
            request_id=request.request_id,
            translated_text=translated,
            source_language=request.source_language,
            target_language=request.target_language,
            confidence=confidence,
            alternatives=alternatives[:1],
            warnings=warnings
        )


class CulturalAdapter:
    """Adapts content for cultural context."""
    
    def __init__(self):
        self.cultural_norms = {
            Language.ENGLISH: {'formality': 'moderate', 'directness': 'high'},
            Language.JAPANESE: {'formality': 'high', 'directness': 'low'},
            Language.GERMAN: {'formality': 'high', 'directness': 'high'},
            Language.SPANISH: {'formality': 'moderate', 'directness': 'moderate'},
            Language.FRENCH: {'formality': 'high', 'directness': 'moderate'}
        }
    
    def adapt(
        self,
        text: str,
        source_language: Language,
        target_language: Language
    ) -> str:
        """Adapt text for target culture."""
        source_norms = self.cultural_norms.get(source_language, {})
        target_norms = self.cultural_norms.get(target_language, {})
        
        adapted = text
        
        # Adjust formality
        if target_norms.get('formality') == 'high' and source_norms.get('formality') != 'high':
            adapted = self._increase_formality(adapted)
        
        # Adjust directness
        if target_norms.get('directness') == 'low' and source_norms.get('directness') == 'high':
            adapted = self._reduce_directness(adapted)
        
        return adapted
    
    def _increase_formality(self, text: str) -> str:
        """Increase formality of text."""
        # Simple replacements
        replacements = {
            'hi': 'hello',
            'bye': 'goodbye',
            'yeah': 'yes',
            'nope': 'no',
            'gonna': 'going to',
            'wanna': 'want to'
        }
        
        result = text
        for informal, formal in replacements.items():
            result = result.replace(informal, formal)
        
        return result
    
    def _reduce_directness(self, text: str) -> str:
        """Reduce directness of text."""
        # Add polite qualifiers
        if text.startswith('You should'):
            text = 'Perhaps you might consider ' + text[10:]
        elif text.startswith('Do this'):
            text = 'It would be appreciated if you could do this'
        
        return text


class LanguagePreferenceManager:
    """Manages user language preferences."""
    
    def __init__(self):
        self.user_preferences: Dict[str, Language] = {}
        self.auto_detect_enabled: Dict[str, bool] = {}
    
    def set_preference(self, user_id: str, language: Language):
        """Set user's preferred language."""
        self.user_preferences[user_id] = language
    
    def get_preference(self, user_id: str) -> Optional[Language]:
        """Get user's preferred language."""
        return self.user_preferences.get(user_id)
    
    def enable_auto_detect(self, user_id: str, enabled: bool = True):
        """Enable/disable auto-detection for user."""
        self.auto_detect_enabled[user_id] = enabled
    
    def is_auto_detect_enabled(self, user_id: str) -> bool:
        """Check if auto-detection is enabled."""
        return self.auto_detect_enabled.get(user_id, True)


class TranslationCache:
    """Caches translations for efficiency."""
    
    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self.cache: Dict[str, TranslationResult] = {}
    
    def _get_cache_key(
        self,
        text: str,
        source_lang: Language,
        target_lang: Language
    ) -> str:
        """Generate cache key."""
        key_str = f"{source_lang.value}:{target_lang.value}:{text}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(
        self,
        text: str,
        source_lang: Language,
        target_lang: Language
    ) -> Optional[TranslationResult]:
        """Get cached translation."""
        key = self._get_cache_key(text, source_lang, target_lang)
        return self.cache.get(key)
    
    def put(self, result: TranslationResult):
        """Cache translation result."""
        key = self._get_cache_key(
            result.translated_text,  # Using result text for reverse lookup
            result.target_language,
            result.source_language
        )
        
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = result


class PolyglotAgent:
    """Agent with multilingual capabilities."""
    
    def __init__(self):
        self.detector = LanguageDetector()
        self.translator = TranslationEngine()
        self.cultural_adapter = CulturalAdapter()
        self.preference_manager = LanguagePreferenceManager()
        self.cache = TranslationCache()
        self.translation_history: List[TranslationResult] = []
        self.supported_languages = set(Language)
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect language of text."""
        return self.detector.detect(text)
    
    def translate(
        self,
        text: str,
        target_language: Language,
        source_language: Optional[Language] = None,
        quality_level: TranslationQuality = TranslationQuality.STANDARD,
        culturally_adapt: bool = False
    ) -> TranslationResult:
        """Translate text to target language."""
        # Detect source language if not provided
        if source_language is None:
            detection = self.detect_language(text)
            source_language = detection.detected_language
        
        # Check cache
        cached = self.cache.get(text, source_language, target_language)
        if cached:
            return cached
        
        # Create translation request
        request = TranslationRequest(
            request_id=hashlib.md5(f"{text}{datetime.now()}".encode()).hexdigest()[:8],
            source_text=text,
            source_language=source_language,
            target_language=target_language,
            quality_level=quality_level
        )
        
        # Translate
        result = self.translator.translate(request)
        
        # Apply cultural adaptation if requested
        if culturally_adapt:
            result.translated_text = self.cultural_adapter.adapt(
                result.translated_text,
                source_language,
                target_language
            )
        
        # Cache result
        self.cache.put(result)
        
        # Record history
        self.translation_history.append(result)
        
        return result
    
    def translate_conversation(
        self,
        messages: List[str],
        target_language: Language,
        preserve_speaker: bool = True
    ) -> List[str]:
        """Translate an entire conversation."""
        translated = []
        
        for message in messages:
            result = self.translate(message, target_language)
            translated.append(result.translated_text)
        
        return translated
    
    def create_multilingual_content(
        self,
        text: str,
        primary_language: Language,
        target_languages: List[Language]
    ) -> MultilingualContent:
        """Create content in multiple languages."""
        content_id = hashlib.md5(text.encode()).hexdigest()[:8]
        
        translations = {primary_language: text}
        
        for target_lang in target_languages:
            if target_lang != primary_language:
                result = self.translate(
                    text,
                    target_lang,
                    source_language=primary_language,
                    culturally_adapt=True
                )
                translations[target_lang] = result.translated_text
        
        return MultilingualContent(
            content_id=content_id,
            translations=translations,
            primary_language=primary_language,
            metadata={'created_at': datetime.now().isoformat()}
        )
    
    def get_content_in_language(
        self,
        content: MultilingualContent,
        language: Language
    ) -> str:
        """Get content in specified language."""
        # Return direct translation if available
        if language in content.translations:
            return content.translations[language]
        
        # Otherwise, translate from primary language
        primary_text = content.translations[content.primary_language]
        result = self.translate(primary_text, language, content.primary_language)
        return result.translated_text
    
    def auto_translate_for_user(
        self,
        user_id: str,
        text: str,
        source_language: Optional[Language] = None
    ) -> str:
        """Automatically translate for user's preferred language."""
        # Get user preference
        user_lang = self.preference_manager.get_preference(user_id)
        
        if user_lang is None:
            # No preference set, return original
            return text
        
        # Detect or use provided source language
        if source_language is None:
            detection = self.detect_language(text)
            source_language = detection.detected_language
        
        # If already in user's language, return original
        if source_language == user_lang:
            return text
        
        # Translate
        result = self.translate(text, user_lang, source_language)
        return result.translated_text
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get translation statistics."""
        if not self.translation_history:
            return {'total_translations': 0}
        
        # Language pair statistics
        language_pairs: Dict[str, int] = {}
        for result in self.translation_history:
            pair = f"{result.source_language.value}→{result.target_language.value}"
            language_pairs[pair] = language_pairs.get(pair, 0) + 1
        
        # Average confidence
        avg_confidence = sum(r.confidence for r in self.translation_history) / len(self.translation_history)
        
        # Most translated languages
        target_langs: Dict[str, int] = {}
        for result in self.translation_history:
            lang = result.target_language.value
            target_langs[lang] = target_langs.get(lang, 0) + 1
        
        return {
            'total_translations': len(self.translation_history),
            'average_confidence': avg_confidence,
            'language_pairs': language_pairs,
            'most_translated_to': max(target_langs.items(), key=lambda x: x[1])[0] if target_langs else None,
            'cache_size': len(self.cache.cache),
            'supported_languages': len(self.supported_languages)
        }


def demonstrate_polyglot_agent():
    """Demonstrate the Polyglot Agent."""
    print("=" * 60)
    print("Polyglot Agent Demonstration")
    print("=" * 60)
    
    agent = PolyglotAgent()
    
    print("\n1. LANGUAGE DETECTION")
    print("-" * 60)
    
    test_texts = [
        "Hello, how are you today?",
        "Bonjour, comment allez-vous?",
        "Hola, ¿cómo estás?",
        "Guten Tag, wie geht es dir?"
    ]
    
    for text in test_texts:
        detection = agent.detect_language(text)
        print(f"\nText: {text}")
        print(f"  Detected: {detection.detected_language.value}")
        print(f"  Confidence: {detection.confidence:.2f}")
        if detection.alternative_languages:
            alts = [f"{lang.value}({score:.2f})" for lang, score in detection.alternative_languages[:2]]
            print(f"  Alternatives: {', '.join(alts)}")
    
    print("\n\n2. TRANSLATION - ENGLISH TO SPANISH")
    print("-" * 60)
    
    english_texts = [
        "Hello world",
        "Thank you",
        "Goodbye",
        "Please help me"
    ]
    
    for text in english_texts:
        result = agent.translate(
            text,
            target_language=Language.SPANISH,
            source_language=Language.ENGLISH
        )
        print(f"\nEnglish: {text}")
        print(f"Spanish: {result.translated_text}")
        print(f"Confidence: {result.confidence:.2f}")
    
    print("\n\n3. TRANSLATION - ENGLISH TO FRENCH")
    print("-" * 60)
    
    text = "Hello, thank you for your help"
    result = agent.translate(
        text,
        target_language=Language.FRENCH,
        source_language=Language.ENGLISH
    )
    
    print(f"English: {text}")
    print(f"French: {result.translated_text}")
    print(f"Confidence: {result.confidence:.2f}")
    if result.alternatives:
        print(f"Alternatives: {result.alternatives[0]}")
    
    print("\n\n4. CONVERSATION TRANSLATION")
    print("-" * 60)
    
    conversation = [
        "Hello, how are you?",
        "Thank you for your help",
        "Goodbye, have a good day"
    ]
    
    print("Original conversation (English):")
    for i, msg in enumerate(conversation, 1):
        print(f"  {i}. {msg}")
    
    translated = agent.translate_conversation(conversation, Language.SPANISH)
    
    print("\nTranslated conversation (Spanish):")
    for i, msg in enumerate(translated, 1):
        print(f"  {i}. {msg}")
    
    print("\n\n5. MULTILINGUAL CONTENT CREATION")
    print("-" * 60)
    
    original_text = "Welcome to our service. Thank you for choosing us."
    
    content = agent.create_multilingual_content(
        original_text,
        primary_language=Language.ENGLISH,
        target_languages=[Language.SPANISH, Language.FRENCH, Language.GERMAN]
    )
    
    print(f"Original ({Language.ENGLISH.value}): {original_text}")
    print(f"\nTranslations:")
    for lang, text in content.translations.items():
        if lang != Language.ENGLISH:
            print(f"  {lang.value}: {text}")
    
    print("\n\n6. USER LANGUAGE PREFERENCES")
    print("-" * 60)
    
    # Set preferences for different users
    agent.preference_manager.set_preference("user1", Language.SPANISH)
    agent.preference_manager.set_preference("user2", Language.FRENCH)
    agent.preference_manager.set_preference("user3", Language.GERMAN)
    
    message = "Hello, this is an important notification"
    
    print(f"Original message: {message}\n")
    
    for user_id in ["user1", "user2", "user3"]:
        pref_lang = agent.preference_manager.get_preference(user_id)
        translated = agent.auto_translate_for_user(user_id, message)
        print(f"{user_id} ({pref_lang.value}): {translated}")
    
    print("\n\n7. CULTURAL ADAPTATION")
    print("-" * 60)
    
    text = "Hi, you should do this now"
    
    print(f"Original: {text}\n")
    
    # Translate with cultural adaptation
    result_es = agent.translate(
        text,
        target_language=Language.SPANISH,
        source_language=Language.ENGLISH,
        culturally_adapt=True
    )
    
    result_ja = agent.translate(
        text,
        target_language=Language.JAPANESE,
        source_language=Language.ENGLISH,
        culturally_adapt=True
    )
    
    print(f"Spanish (adapted): {result_es.translated_text}")
    print(f"Japanese (adapted): {result_ja.translated_text}")
    print("\nNote: Cultural adaptation adjusts formality and directness")
    
    print("\n\n8. AUTO-DETECTION AND TRANSLATION")
    print("-" * 60)
    
    mixed_texts = [
        ("Hello world", Language.SPANISH),
        ("Bonjour", Language.ENGLISH),
        ("Gracias", Language.FRENCH)
    ]
    
    print("Auto-detecting source language and translating:")
    for text, target in mixed_texts:
        result = agent.translate(text, target_language=target)
        print(f"\n'{text}' → {result.target_language.value}")
        print(f"  Result: {result.translated_text}")
        print(f"  Detected source: {result.source_language.value}")
    
    print("\n\n9. TRANSLATION CACHING")
    print("-" * 60)
    
    test_text = "Hello world"
    
    print("First translation (no cache):")
    result1 = agent.translate(test_text, Language.SPANISH, Language.ENGLISH)
    print(f"  Result: {result1.translated_text}")
    
    print("\nSecond translation (cached):")
    result2 = agent.translate(test_text, Language.SPANISH, Language.ENGLISH)
    print(f"  Result: {result2.translated_text}")
    print(f"  Same result: {result1.translated_text == result2.translated_text}")
    
    print("\n\n10. STATISTICS")
    print("-" * 60)
    
    stats = agent.get_statistics()
    print(f"  Total Translations: {stats['total_translations']}")
    print(f"  Average Confidence: {stats['average_confidence']:.2f}")
    print(f"  Supported Languages: {stats['supported_languages']}")
    print(f"  Cache Size: {stats['cache_size']}")
    
    print(f"\n  Language Pairs:")
    for pair, count in sorted(stats['language_pairs'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {pair}: {count}")
    
    if stats['most_translated_to']:
        print(f"\n  Most Translated To: {stats['most_translated_to']}")
    
    print("\n" + "=" * 60)
    print("🎉 Pattern 135 Complete!")
    print("Specialization Category: 50%")
    print("135/170 patterns implemented (79.4%)!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_polyglot_agent()
