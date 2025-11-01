"""
Pattern 136: Accessibility-Focused Agent

Description:
    The Accessibility-Focused Agent pattern creates agents designed to be
    inclusive and usable by people with diverse abilities and needs. These
    agents prioritize accessibility features such as clear communication,
    support for assistive technologies, content adaptation, and inclusive
    design principles to ensure everyone can benefit from AI assistance.
    
    Accessibility-focused agents go beyond basic usability to accommodate
    visual, auditory, cognitive, motor, and other impairments. They provide
    alternative formats, clear language, patience with interaction, and
    features that make AI technology available to all users regardless of
    their abilities or circumstances.
    
    This pattern embodies inclusive design principles, ensuring AI agents
    are not just functional but truly accessible to diverse populations
    including people with disabilities, elderly users, non-native speakers,
    and users with temporary limitations.

Key Components:
    1. Clear Communication: Simple, unambiguous language
    2. Alternative Formats: Multiple content representations
    3. Assistive Tech Support: Screen readers, voice control
    4. Flexible Interaction: Multiple input/output methods
    5. Content Adaptation: Adjust to user needs
    6. Error Tolerance: Patient, forgiving interactions
    7. Inclusive Design: Universal accessibility

Accessibility Features:
    1. Visual: Screen reader support, high contrast, text sizing
    2. Auditory: Text alternatives, captions, visual alerts
    3. Cognitive: Clear language, simple structure, consistency
    4. Motor: Keyboard navigation, voice control, large targets
    5. Language: Plain language, translation, glossaries
    6. Context: Help text, examples, clear feedback
    7. Time: No time pressure, save progress, pause

User Accommodations:
    1. Vision impaired: Audio descriptions, clear structure
    2. Hearing impaired: Text alternatives, visual cues
    3. Cognitive differences: Simple language, clear layout
    4. Motor limitations: Alternative inputs, no precision requirements
    5. Language barriers: Clear, simple language, visual aids
    6. Elderly users: Larger text, slower pace, clear instructions
    7. Temporary limitations: Flexible interaction methods

Use Cases:
    - Public service AI assistants
    - Educational platforms
    - Healthcare information systems
    - Government services
    - E-learning applications
    - Banking and finance
    - Emergency services
    - General consumer applications

Advantages:
    - Inclusive user base
    - Legal compliance (ADA, WCAG)
    - Better UX for everyone
    - Broader market reach
    - Social responsibility
    - Reduced support burden
    - Positive brand image

Challenges:
    - Balancing features and complexity
    - Multiple accommodation needs
    - Testing with diverse users
    - Performance overhead
    - Maintaining accessibility
    - Cost of implementation
    - Technical limitations

LangChain Implementation:
    This implementation uses LangChain for:
    - Clear, simple language generation
    - Content format adaptation
    - Flexible interaction patterns
    - Accessible error messages
    
Production Considerations:
    - Follow WCAG guidelines
    - Test with assistive technologies
    - Provide multiple interaction modes
    - Include accessibility settings
    - Document accessibility features
    - Collect user feedback
    - Regular accessibility audits
    - Train on inclusive practices
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


class AccessibilityNeed(Enum):
    """Types of accessibility needs."""
    VISUAL_IMPAIRMENT = "visual_impairment"
    HEARING_IMPAIRMENT = "hearing_impairment"
    COGNITIVE_DIFFERENCE = "cognitive_difference"
    MOTOR_LIMITATION = "motor_limitation"
    LANGUAGE_BARRIER = "language_barrier"
    ELDERLY = "elderly"
    NONE = "none"


class ContentFormat(Enum):
    """Content output formats."""
    TEXT = "text"
    AUDIO_DESCRIPTION = "audio_description"
    SIMPLIFIED_TEXT = "simplified_text"
    STRUCTURED_DATA = "structured_data"
    BULLET_POINTS = "bullet_points"
    STEP_BY_STEP = "step_by_step"


class ReadingLevel(Enum):
    """Reading comprehension levels."""
    ELEMENTARY = "elementary"
    MIDDLE_SCHOOL = "middle_school"
    HIGH_SCHOOL = "high_school"
    COLLEGE = "college"
    ADVANCED = "advanced"


@dataclass
class AccessibilityProfile:
    """
    User accessibility profile.
    
    Attributes:
        user_id: User identifier
        accessibility_needs: List of accessibility requirements
        preferred_format: Preferred content format
        reading_level: Reading comprehension level
        font_size: Text size preference (multiplier)
        high_contrast: High contrast mode
        audio_enabled: Audio output enabled
        reduced_motion: Reduce animations
        keyboard_only: Keyboard-only navigation
        screen_reader: Screen reader in use
    """
    user_id: str
    accessibility_needs: List[AccessibilityNeed] = field(default_factory=list)
    preferred_format: ContentFormat = ContentFormat.TEXT
    reading_level: ReadingLevel = ReadingLevel.HIGH_SCHOOL
    font_size: float = 1.0
    high_contrast: bool = False
    audio_enabled: bool = False
    reduced_motion: bool = False
    keyboard_only: bool = False
    screen_reader: bool = False


@dataclass
class AccessibleResponse:
    """
    Accessible response with multiple formats.
    
    Attributes:
        response_id: Unique identifier
        primary_content: Main response
        alternative_formats: Alternative representations
        metadata: Accessibility metadata
        reading_time: Estimated reading time (seconds)
        complexity_score: Content complexity (0-1)
        timestamp: When generated
    """
    response_id: str
    primary_content: str
    alternative_formats: Dict[ContentFormat, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reading_time: float = 0.0
    complexity_score: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)


class AccessibilityAgent:
    """
    Agent focused on accessibility and inclusive design.
    
    This agent adapts content and interaction to accommodate
    diverse user needs and abilities.
    """
    
    def __init__(self, temperature: float = 0.5):
        """
        Initialize accessibility-focused agent.
        
        Args:
            temperature: LLM temperature
        """
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.user_profiles: Dict[str, AccessibilityProfile] = {}
        self.response_history: List[AccessibleResponse] = []
        self.response_counter = 0
    
    def set_user_profile(self, profile: AccessibilityProfile):
        """Set accessibility profile for a user."""
        self.user_profiles[profile.user_id] = profile
    
    def simplify_text(self, text: str, reading_level: ReadingLevel) -> str:
        """
        Simplify text to appropriate reading level.
        
        Args:
            text: Original text
            reading_level: Target reading level
            
        Returns:
            Simplified text
        """
        level_descriptions = {
            ReadingLevel.ELEMENTARY: "a 3rd grader",
            ReadingLevel.MIDDLE_SCHOOL: "a 7th grader",
            ReadingLevel.HIGH_SCHOOL: "a high school student",
            ReadingLevel.COLLEGE: "a college student",
            ReadingLevel.ADVANCED: "an expert"
        }
        
        prompt = ChatPromptTemplate.from_template(
            "Rewrite this text so {target_audience} can understand it easily:\n\n"
            "{text}\n\n"
            "Use:\n"
            "- Simple, common words\n"
            "- Short sentences\n"
            "- Clear structure\n"
            "- Active voice\n"
            "- Concrete examples\n\n"
            "Simplified version:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        simplified = chain.invoke({
            "text": text,
            "target_audience": level_descriptions[reading_level]
        })
        
        return simplified.strip()
    
    def create_audio_description(self, content: str) -> str:
        """
        Create audio-friendly description.
        
        Args:
            content: Visual content to describe
            
        Returns:
            Audio description
        """
        prompt = ChatPromptTemplate.from_template(
            "Create an audio description for screen readers:\n\n"
            "{content}\n\n"
            "Provide a clear, verbal description that:\n"
            "- Describes visual elements\n"
            "- Explains layout and structure\n"
            "- Indicates interactive elements\n"
            "- Uses clear navigation cues\n\n"
            "Audio description:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        description = chain.invoke({"content": content})
        return description.strip()
    
    def format_as_steps(self, content: str) -> List[str]:
        """
        Format content as clear steps.
        
        Args:
            content: Content to format
            
        Returns:
            List of steps
        """
        prompt = ChatPromptTemplate.from_template(
            "Break this into clear, numbered steps:\n\n"
            "{content}\n\n"
            "Each step should:\n"
            "- Be one clear action\n"
            "- Use simple language\n"
            "- Start with an action verb\n"
            "- Be easy to follow\n\n"
            "List each step on a new line starting with the number."
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({"content": content})
        
        # Parse steps
        steps = []
        for line in result.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering/bullets
                step = line.lstrip('0123456789.-• ')
                if step:
                    steps.append(step)
        
        return steps
    
    def respond(
        self,
        user_input: str,
        user_id: Optional[str] = None
    ) -> AccessibleResponse:
        """
        Generate accessible response.
        
        Args:
            user_input: User's question/request
            user_id: User identifier for profile
            
        Returns:
            Accessible response with multiple formats
        """
        self.response_counter += 1
        response_id = f"resp_{self.response_counter}"
        
        # Get user profile
        profile = self.user_profiles.get(user_id) if user_id else None
        reading_level = profile.reading_level if profile else ReadingLevel.HIGH_SCHOOL
        
        # Generate primary response
        prompt = ChatPromptTemplate.from_template(
            "Provide a clear, helpful response to this question:\n\n"
            "{user_input}\n\n"
            "Use:\n"
            "- Clear, simple language\n"
            "- Short paragraphs\n"
            "- Specific examples\n"
            "- Organized structure\n\n"
            "Response:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        primary_content = chain.invoke({"user_input": user_input})
        
        # Create alternative formats
        alternative_formats = {}
        
        # Simplified version
        alternative_formats[ContentFormat.SIMPLIFIED_TEXT] = self.simplify_text(
            primary_content,
            ReadingLevel.MIDDLE_SCHOOL
        )
        
        # Step-by-step if applicable
        if any(word in user_input.lower() for word in ['how', 'steps', 'guide', 'do']):
            steps = self.format_as_steps(primary_content)
            alternative_formats[ContentFormat.STEP_BY_STEP] = '\n'.join(
                f"{i+1}. {step}" for i, step in enumerate(steps)
            )
        
        # Audio description for screen readers
        if profile and profile.screen_reader:
            alternative_formats[ContentFormat.AUDIO_DESCRIPTION] = \
                self.create_audio_description(primary_content)
        
        # Estimate reading time (average 200 words/min)
        word_count = len(primary_content.split())
        reading_time = (word_count / 200) * 60  # seconds
        
        # Create response
        response = AccessibleResponse(
            response_id=response_id,
            primary_content=primary_content,
            alternative_formats=alternative_formats,
            reading_time=reading_time,
            metadata={
                "user_id": user_id,
                "reading_level": reading_level.value if reading_level else None,
                "word_count": word_count
            }
        )
        
        self.response_history.append(response)
        return response
    
    def provide_help(self, context: str = "general") -> str:
        """
        Provide accessible help information.
        
        Args:
            context: Help context
            
        Returns:
            Help text
        """
        prompt = ChatPromptTemplate.from_template(
            "Provide clear, beginner-friendly help for: {context}\n\n"
            "Include:\n"
            "- What it does (simple explanation)\n"
            "- How to use it (step-by-step)\n"
            "- Common issues and solutions\n"
            "- Where to get more help\n\n"
            "Use very simple language. Be encouraging and supportive."
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        help_text = chain.invoke({"context": context})
        return help_text
    
    def check_accessibility(self, content: str) -> Dict[str, Any]:
        """
        Check content for accessibility issues.
        
        Args:
            content: Content to check
            
        Returns:
            Accessibility assessment
        """
        prompt = ChatPromptTemplate.from_template(
            "Review this content for accessibility:\n\n"
            "{content}\n\n"
            "Check for:\n"
            "1. Reading level (is it clear and simple?)\n"
            "2. Structure (is it well organized?)\n"
            "3. Language (avoids jargon and complex terms?)\n"
            "4. Length (appropriate, not overwhelming?)\n"
            "5. Clarity (easy to understand?)\n\n"
            "Provide:\n"
            "SCORE: 0-10\n"
            "ISSUES: [list any problems]\n"
            "SUGGESTIONS: [how to improve]"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        
        assessment = chain.invoke({"content": content})
        
        # Parse assessment
        score = 7.0  # default
        lines = assessment.split('\n')
        for line in lines:
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(":")[1].strip().split()[0])
                except:
                    pass
        
        return {
            "score": score,
            "assessment": assessment,
            "accessible": score >= 7.0
        }
    
    def get_accessibility_report(self) -> Dict[str, Any]:
        """Get accessibility usage report."""
        if not self.response_history:
            return {
                "total_responses": 0,
                "avg_reading_time": 0.0,
                "formats_provided": []
            }
        
        total_reading_time = sum(r.reading_time for r in self.response_history)
        avg_reading_time = total_reading_time / len(self.response_history)
        
        # Count formats
        format_counts = {}
        for response in self.response_history:
            for format_type in response.alternative_formats.keys():
                format_counts[format_type] = format_counts.get(format_type, 0) + 1
        
        return {
            "total_responses": len(self.response_history),
            "avg_reading_time": avg_reading_time,
            "formats_provided": list(format_counts.keys()),
            "format_usage": {fmt.value: count for fmt, count in format_counts.items()},
            "users_served": len(self.user_profiles)
        }


def demonstrate_accessibility():
    """Demonstrate accessibility-focused agent pattern."""
    
    print("=" * 80)
    print("ACCESSIBILITY-FOCUSED AGENT PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = AccessibilityAgent()
    
    # Example 1: User profiles
    print("\n" + "=" * 80)
    print("Example 1: Accessibility Profiles")
    print("=" * 80)
    
    profiles = [
        AccessibilityProfile(
            user_id="user1",
            accessibility_needs=[AccessibilityNeed.VISUAL_IMPAIRMENT],
            screen_reader=True,
            audio_enabled=True,
            reading_level=ReadingLevel.MIDDLE_SCHOOL
        ),
        AccessibilityProfile(
            user_id="user2",
            accessibility_needs=[AccessibilityNeed.COGNITIVE_DIFFERENCE],
            preferred_format=ContentFormat.STEP_BY_STEP,
            reading_level=ReadingLevel.ELEMENTARY,
            reduced_motion=True
        ),
        AccessibilityProfile(
            user_id="user3",
            accessibility_needs=[AccessibilityNeed.ELDERLY],
            font_size=1.5,
            high_contrast=True,
            reading_level=ReadingLevel.HIGH_SCHOOL
        )
    ]
    
    for profile in profiles:
        agent.set_user_profile(profile)
        print(f"\nProfile: {profile.user_id}")
        print(f"  Needs: {[need.value for need in profile.accessibility_needs]}")
        print(f"  Reading Level: {profile.reading_level.value}")
        print(f"  Screen Reader: {profile.screen_reader}")
    
    # Example 2: Simplified text
    print("\n" + "=" * 80)
    print("Example 2: Text Simplification")
    print("=" * 80)
    
    complex_text = """
    The implementation leverages sophisticated algorithms to optimize 
    performance characteristics through parallelization and caching mechanisms,
    thereby enhancing computational efficiency and reducing latency.
    """
    
    print("\nOriginal (Complex):")
    print(complex_text)
    
    simplified = agent.simplify_text(complex_text, ReadingLevel.MIDDLE_SCHOOL)
    print("\nSimplified (Middle School Level):")
    print(simplified)
    
    # Example 3: Step-by-step format
    print("\n" + "=" * 80)
    print("Example 3: Step-by-Step Instructions")
    print("=" * 80)
    
    content = "To reset your password, access the settings, find security options, and select password reset."
    
    print("\nOriginal:")
    print(content)
    
    steps = agent.format_as_steps(content)
    print("\nAs Steps:")
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}")
    
    # Example 4: Accessible responses
    print("\n" + "=" * 80)
    print("Example 4: Accessible Responses with Multiple Formats")
    print("=" * 80)
    
    question = "How do I change my account settings?"
    
    print(f"\nQuestion: {question}")
    print(f"User: user2 (needs: cognitive support, step-by-step)")
    
    response = agent.respond(question, user_id="user2")
    
    print(f"\nPrimary Response:")
    print(response.primary_content[:200] + "...")
    
    print(f"\nAlternative Formats Available:")
    for format_type in response.alternative_formats.keys():
        print(f"  - {format_type.value}")
    
    if ContentFormat.STEP_BY_STEP in response.alternative_formats:
        print(f"\nStep-by-Step Version:")
        print(response.alternative_formats[ContentFormat.STEP_BY_STEP])
    
    print(f"\nReading Time: {response.reading_time:.0f} seconds")
    
    # Example 5: Screen reader support
    print("\n" + "=" * 80)
    print("Example 5: Screen Reader Audio Descriptions")
    print("=" * 80)
    
    visual_content = "A login form with email field, password field, and login button"
    
    print(f"\nVisual Content: {visual_content}")
    
    audio_desc = agent.create_audio_description(visual_content)
    print(f"\nAudio Description for Screen Readers:")
    print(audio_desc)
    
    # Example 6: Help system
    print("\n" + "=" * 80)
    print("Example 6: Accessible Help System")
    print("=" * 80)
    
    print("\nRequesting help for: 'uploading files'")
    
    help_text = agent.provide_help("uploading files")
    print(f"\nHelp Content:")
    print(help_text[:400] + "...")
    
    # Example 7: Accessibility checking
    print("\n" + "=" * 80)
    print("Example 7: Content Accessibility Assessment")
    print("=" * 80)
    
    test_content = "This is a simple, clear explanation that anyone can understand."
    
    print(f"\nChecking content: \"{test_content}\"")
    
    assessment = agent.check_accessibility(test_content)
    print(f"\nAccessibility Score: {assessment['score']}/10")
    print(f"Accessible: {assessment['accessible']}")
    
    # Example 8: Usage report
    print("\n" + "=" * 80)
    print("Example 8: Accessibility Usage Report")
    print("=" * 80)
    
    # Generate some responses
    test_questions = [
        "What is AI?",
        "How do I sign up?",
        "Explain machine learning"
    ]
    
    for q in test_questions:
        agent.respond(q, user_id="user1")
    
    report = agent.get_accessibility_report()
    
    print("\nACCESSIBILITY REPORT:")
    print("=" * 60)
    print(f"Total Responses: {report['total_responses']}")
    print(f"Average Reading Time: {report['avg_reading_time']:.0f} seconds")
    print(f"Users Served: {report['users_served']}")
    print(f"\nFormats Provided:")
    for format_name, count in report.get('format_usage', {}).items():
        print(f"  {format_name}: {count}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Accessibility-Focused Agent Pattern")
    print("=" * 80)
    
    summary = """
    The Accessibility-Focused Agent pattern demonstrated:
    
    1. USER PROFILES (Example 1):
       - Accessibility needs tracking
       - Individual preferences
       - Assistive technology support
       - Reading level customization
       - Visual preferences
    
    2. TEXT SIMPLIFICATION (Example 2):
       - Reading level adaptation
       - Complex to simple conversion
       - Clear language
       - Appropriate vocabulary
       - Comprehension support
    
    3. STRUCTURED FORMATS (Example 3):
       - Step-by-step instructions
       - Clear organization
       - Easy-to-follow format
       - Actionable guidance
       - Sequential clarity
    
    4. MULTIPLE FORMATS (Example 4):
       - Primary content
       - Alternative representations
       - Format flexibility
       - User choice
       - Comprehensive options
    
    5. SCREEN READER SUPPORT (Example 5):
       - Audio descriptions
       - Verbal navigation
       - Structure explanation
       - Accessible interaction
       - Non-visual access
    
    6. HELP SYSTEM (Example 6):
       - Clear, simple help
       - Beginner-friendly
       - Encouraging tone
       - Practical guidance
       - Support resources
    
    7. QUALITY ASSURANCE (Example 7):
       - Accessibility scoring
       - Content assessment
       - Issue identification
       - Improvement suggestions
       - Standards compliance
    
    8. MONITORING (Example 8):
       - Usage tracking
       - Format analytics
       - Performance metrics
       - User statistics
       - Continuous improvement
    
    KEY BENEFITS:
    ✓ Inclusive user base
    ✓ Legal compliance (ADA, WCAG)
    ✓ Better UX for everyone
    ✓ Broader market reach
    ✓ Social responsibility
    ✓ Reduced support burden
    ✓ Positive brand image
    ✓ Universal usability
    
    USE CASES:
    • Public service AI assistants
    • Educational platforms
    • Healthcare information
    • Government services
    • E-learning applications
    • Banking and finance
    • Emergency services
    • Consumer applications
    
    ACCESSIBILITY FEATURES:
    → Visual: Screen readers, high contrast, text sizing
    → Auditory: Text alternatives, captions, visual alerts
    → Cognitive: Clear language, simple structure
    → Motor: Keyboard navigation, voice control
    → Language: Plain language, translation
    → Context: Help text, examples, clear feedback
    → Time: No pressure, save progress, pause
    
    USER ACCOMMODATIONS:
    • Vision impaired: Audio descriptions
    • Hearing impaired: Text alternatives
    • Cognitive differences: Simple language
    • Motor limitations: Alternative inputs
    • Language barriers: Clear communication
    • Elderly users: Larger text, slower pace
    • Temporary limitations: Flexible methods
    
    BEST PRACTICES:
    1. Follow WCAG guidelines
    2. Test with assistive technologies
    3. Provide multiple interaction modes
    4. Include accessibility settings
    5. Document accessibility features
    6. Collect user feedback
    7. Regular accessibility audits
    8. Train on inclusive practices
    
    TRADE-OFFS:
    • Features vs. simplicity
    • Customization vs. consistency
    • Performance vs. alternatives
    • Automation vs. user control
    
    PRODUCTION CONSIDERATIONS:
    → Comply with ADA and WCAG standards
    → Test with real assistive technologies
    → Provide clear documentation
    → Support keyboard-only navigation
    → Ensure color contrast ratios
    → Include skip navigation links
    → Provide text alternatives
    → Enable font size adjustment
    → Support screen magnification
    → Test with diverse users
    
    This pattern ensures AI agents are truly inclusive, making technology
    accessible to all users regardless of their abilities, creating a more
    equitable and usable experience for everyone.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_accessibility()
