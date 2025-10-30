"""
Pattern 075: Creative Agent

Description:
    A Creative Agent is a specialized AI agent designed to generate original creative
    content across various domains including writing (stories, poetry, scripts), visual
    concepts (art descriptions, design ideas), music concepts, and other creative works.
    This pattern demonstrates how to build an intelligent agent that can understand
    creative requirements, generate diverse creative outputs, adapt styles, incorporate
    feedback, and maintain creative consistency.
    
    The agent combines creative ideation with structured generation techniques to produce
    high-quality, original content. It can work across multiple creative domains, adapt
    to different styles and tones, generate variations, and iteratively refine outputs
    based on feedback.

Components:
    1. Ideation Engine: Generates creative concepts and ideas
    2. Style Adapter: Adapts content to different styles and tones
    3. Content Generator: Creates structured creative outputs
    4. Variation Creator: Produces diverse alternatives
    5. Constraint Respector: Adheres to creative constraints
    6. Feedback Integrator: Incorporates critique and improvements
    7. Consistency Maintainer: Ensures coherent creative vision
    8. Multi-Modal Orchestrator: Coordinates different creative types

Key Features:
    - Cross-domain creative generation (text, visual concepts, music ideas)
    - Style and tone adaptation (formal, casual, poetic, technical)
    - Character and world-building for narratives
    - Creative constraint handling (theme, length, format)
    - Iterative refinement with feedback
    - Variation generation for exploration
    - Originality assessment
    - Creative inspiration from prompts
    - Collaborative creativity support
    - Genre and format flexibility

Use Cases:
    - Story and novel writing
    - Poetry and lyric generation
    - Screenplay and script writing
    - Marketing copy and slogans
    - Product naming and branding
    - Art concept development
    - Game narrative design
    - Content marketing
    - Creative brainstorming
    - Educational creative writing

LangChain Implementation:
    Uses ChatOpenAI with varying temperatures for different creative tasks,
    specialized chains for ideation, generation, and refinement.
"""

import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class CreativeType(Enum):
    """Types of creative content"""
    STORY = "story"
    POEM = "poem"
    SCRIPT = "script"
    SONG_LYRICS = "song_lyrics"
    ARTICLE = "article"
    MARKETING_COPY = "marketing_copy"
    PRODUCT_NAME = "product_name"
    SLOGAN = "slogan"
    DIALOGUE = "dialogue"
    WORLD_BUILDING = "world_building"


class CreativeStyle(Enum):
    """Creative styles and tones"""
    FORMAL = "formal"
    CASUAL = "casual"
    POETIC = "poetic"
    HUMOROUS = "humorous"
    DRAMATIC = "dramatic"
    MINIMALIST = "minimalist"
    ELABORATE = "elaborate"
    MYSTERIOUS = "mysterious"
    ROMANTIC = "romantic"
    DARK = "dark"


class Genre(Enum):
    """Creative genres"""
    FANTASY = "fantasy"
    SCIENCE_FICTION = "science_fiction"
    MYSTERY = "mystery"
    ROMANCE = "romance"
    HORROR = "horror"
    THRILLER = "thriller"
    COMEDY = "comedy"
    DRAMA = "drama"
    ADVENTURE = "adventure"
    HISTORICAL = "historical"


@dataclass
class CreativePrompt:
    """Input prompt for creative generation"""
    type: CreativeType
    theme: str
    style: CreativeStyle
    genre: Optional[Genre] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    inspiration: List[str] = field(default_factory=list)
    target_audience: Optional[str] = None


@dataclass
class Character:
    """Character for narrative works"""
    name: str
    role: str  # protagonist, antagonist, supporting
    personality: List[str]
    background: str
    goals: List[str]
    conflicts: List[str]


@dataclass
class Setting:
    """Setting for creative works"""
    location: str
    time_period: str
    atmosphere: str
    key_features: List[str]
    description: str


@dataclass
class CreativeWork:
    """Generated creative content"""
    work_id: str
    type: CreativeType
    title: str
    content: str
    style: CreativeStyle
    genre: Optional[Genre]
    characters: List[Character] = field(default_factory=list)
    setting: Optional[Setting] = None
    themes: List[str] = field(default_factory=list)
    word_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CreativeFeedback:
    """Feedback on creative work"""
    work_id: str
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    overall_score: float  # 0.0-1.0
    specific_notes: Dict[str, str] = field(default_factory=dict)


@dataclass
class CreativeVariation:
    """Variation of a creative work"""
    original_id: str
    variation_id: str
    content: str
    changes_made: List[str]
    variation_type: str  # tone, structure, perspective, etc.


class CreativeAgent:
    """
    Agent for generating creative content across multiple domains.
    
    This agent can generate stories, poems, scripts, marketing copy, and other
    creative works while adapting to styles, genres, and constraints.
    """
    
    def __init__(self):
        """Initialize the creative agent with specialized LLMs"""
        # Ideation engine for brainstorming
        self.ideation_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9)
        
        # Generator for main content creation
        self.generator_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
        
        # Refiner for improving content
        self.refiner_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
        
        # Critic for evaluation
        self.critic_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        
        # Creative works repository
        self.works: Dict[str, CreativeWork] = {}
    
    def generate_ideas(
        self,
        theme: str,
        creative_type: CreativeType,
        count: int = 5
    ) -> List[str]:
        """
        Generate creative ideas based on theme.
        
        Args:
            theme: Central theme or concept
            creative_type: Type of creative work
            count: Number of ideas to generate
            
        Returns:
            List of creative ideas
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a creative ideation expert. Generate original and
            imaginative ideas based on the theme and content type.
            
            Provide {count} ideas, one per line starting with a number."""),
            ("user", """Theme: {theme}
Content Type: {type}

Generate {count} creative ideas.""")
        ])
        
        chain = prompt | self.ideation_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "theme": theme,
                "type": creative_type.value,
                "count": count
            })
            
            ideas = []
            for line in response.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering
                    idea = re.sub(r'^\d+[\.\)]\s*', '', line)
                    idea = re.sub(r'^[-*]\s*', '', idea)
                    if idea:
                        ideas.append(idea)
            
            return ideas[:count] if ideas else [f"Idea about {theme}" for _ in range(count)]
            
        except Exception as e:
            return [f"Creative idea {i+1} about {theme}" for i in range(count)]
    
    def create_characters(
        self,
        story_concept: str,
        character_count: int = 3
    ) -> List[Character]:
        """
        Create characters for a narrative work.
        
        Args:
            story_concept: Brief story concept
            character_count: Number of characters to create
            
        Returns:
            List of Character objects
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a character development expert. Create compelling
            characters for the story concept.
            
            For each character, provide:
            NAME: character name
            ROLE: protagonist/antagonist/supporting
            PERSONALITY: traits (comma-separated)
            BACKGROUND: brief background
            GOALS: goals (comma-separated)
            CONFLICTS: conflicts (comma-separated)
            ---"""),
            ("user", """Story Concept: {concept}

Create {count} characters.""")
        ])
        
        chain = prompt | self.ideation_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "concept": story_concept,
                "count": character_count
            })
            
            characters = []
            current_char = {}
            
            for line in response.split('\n'):
                line = line.strip()
                
                if line == '---' and current_char:
                    # Create character from collected data
                    if 'NAME' in current_char:
                        characters.append(Character(
                            name=current_char.get('NAME', 'Unnamed'),
                            role=current_char.get('ROLE', 'supporting'),
                            personality=[p.strip() for p in current_char.get('PERSONALITY', '').split(',')],
                            background=current_char.get('BACKGROUND', ''),
                            goals=[g.strip() for g in current_char.get('GOALS', '').split(',')],
                            conflicts=[c.strip() for c in current_char.get('CONFLICTS', '').split(',')]
                        ))
                    current_char = {}
                elif ':' in line:
                    parts = line.split(':', 1)
                    key = parts[0].strip().upper()
                    value = parts[1].strip()
                    current_char[key] = value
            
            # Add last character if exists
            if current_char and 'NAME' in current_char:
                characters.append(Character(
                    name=current_char.get('NAME', 'Unnamed'),
                    role=current_char.get('ROLE', 'supporting'),
                    personality=[p.strip() for p in current_char.get('PERSONALITY', '').split(',')],
                    background=current_char.get('BACKGROUND', ''),
                    goals=[g.strip() for g in current_char.get('GOALS', '').split(',')],
                    conflicts=[c.strip() for c in current_char.get('CONFLICTS', '').split(',')]
                ))
            
            # Fallback if parsing failed
            if not characters:
                for i in range(character_count):
                    characters.append(Character(
                        name=f"Character {i+1}",
                        role="supporting" if i > 0 else "protagonist",
                        personality=["determined", "curious"],
                        background="To be developed",
                        goals=["Achieve their objective"],
                        conflicts=["Internal struggles"]
                    ))
            
            return characters[:character_count]
            
        except Exception as e:
            # Fallback characters
            return [
                Character(
                    name=f"Character {i+1}",
                    role="protagonist" if i == 0 else "supporting",
                    personality=["brave", "intelligent"],
                    background="Mysterious past",
                    goals=["Complete their quest"],
                    conflicts=["Face their fears"]
                )
                for i in range(character_count)
            ]
    
    def create_setting(
        self,
        story_concept: str,
        genre: Genre
    ) -> Setting:
        """
        Create a setting for a narrative work.
        
        Args:
            story_concept: Brief story concept
            genre: Genre of the work
            
        Returns:
            Setting object
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a world-building expert. Create a vivid setting.
            
            Provide:
            LOCATION: Where the story takes place
            TIME: Time period
            ATMOSPHERE: Overall mood/feeling
            FEATURES: Key features (comma-separated)
            DESCRIPTION: Detailed description (2-3 sentences)"""),
            ("user", """Story Concept: {concept}
Genre: {genre}

Create a compelling setting.""")
        ])
        
        chain = prompt | self.ideation_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "concept": story_concept,
                "genre": genre.value
            })
            
            setting_data = {}
            current_key = None
            desc_lines = []
            
            for line in response.split('\n'):
                line = line.strip()
                if ':' in line and line.split(':')[0].strip().upper() in ['LOCATION', 'TIME', 'ATMOSPHERE', 'FEATURES', 'DESCRIPTION']:
                    parts = line.split(':', 1)
                    key = parts[0].strip().upper()
                    value = parts[1].strip()
                    setting_data[key] = value
                    current_key = key
                elif current_key == 'DESCRIPTION' and line:
                    desc_lines.append(line)
            
            if desc_lines:
                setting_data['DESCRIPTION'] = ' '.join(desc_lines)
            
            return Setting(
                location=setting_data.get('LOCATION', 'A mysterious place'),
                time_period=setting_data.get('TIME', 'Present day'),
                atmosphere=setting_data.get('ATMOSPHERE', 'Tense and mysterious'),
                key_features=[f.strip() for f in setting_data.get('FEATURES', '').split(',')],
                description=setting_data.get('DESCRIPTION', 'An intriguing setting awaits.')
            )
            
        except Exception as e:
            return Setting(
                location="An intriguing location",
                time_period="Modern times",
                atmosphere="Mysterious and engaging",
                key_features=["Unique architecture", "Hidden secrets"],
                description="A place full of wonder and possibility."
            )
    
    def generate_content(
        self,
        creative_prompt: CreativePrompt,
        characters: Optional[List[Character]] = None,
        setting: Optional[Setting] = None
    ) -> CreativeWork:
        """
        Generate creative content based on prompt.
        
        Args:
            creative_prompt: Creative prompt with requirements
            characters: Optional characters for narrative works
            setting: Optional setting for narrative works
            
        Returns:
            CreativeWork with generated content
        """
        # Build context
        context_parts = []
        
        if characters:
            char_desc = "\n".join([
                f"- {char.name} ({char.role}): {', '.join(char.personality[:2])}"
                for char in characters
            ])
            context_parts.append(f"Characters:\n{char_desc}")
        
        if setting:
            context_parts.append(f"Setting: {setting.location}, {setting.time_period}")
            context_parts.append(f"Atmosphere: {setting.atmosphere}")
        
        context = "\n\n".join(context_parts) if context_parts else "No additional context"
        
        # Generate based on type
        if creative_prompt.type == CreativeType.STORY:
            return self._generate_story(creative_prompt, context, characters, setting)
        elif creative_prompt.type == CreativeType.POEM:
            return self._generate_poem(creative_prompt, context)
        elif creative_prompt.type == CreativeType.MARKETING_COPY:
            return self._generate_marketing_copy(creative_prompt)
        else:
            return self._generate_generic(creative_prompt, context)
    
    def _generate_story(
        self,
        prompt: CreativePrompt,
        context: str,
        characters: Optional[List[Character]],
        setting: Optional[Setting]
    ) -> CreativeWork:
        """Generate a story"""
        template = ChatPromptTemplate.from_messages([
            ("system", """You are a master storyteller. Write an engaging story that
            captures the theme, style, and genre specified.
            
            The story should have:
            - Compelling opening
            - Character development
            - Rising action
            - Climax
            - Resolution
            
            Write in the specified style and maintain consistency."""),
            ("user", """Theme: {theme}
Style: {style}
Genre: {genre}

{context}

Constraints: {constraints}

Write the story.""")
        ])
        
        chain = template | self.generator_llm | StrOutputParser()
        
        try:
            content = chain.invoke({
                "theme": prompt.theme,
                "style": prompt.style.value,
                "genre": prompt.genre.value if prompt.genre else "general",
                "context": context,
                "constraints": str(prompt.constraints) if prompt.constraints else "None"
            })
            
            # Generate title
            title = self._generate_title(content, prompt.type, prompt.theme)
            
            return CreativeWork(
                work_id=f"work_{datetime.now().timestamp()}",
                type=prompt.type,
                title=title,
                content=content,
                style=prompt.style,
                genre=prompt.genre,
                characters=characters or [],
                setting=setting,
                themes=[prompt.theme],
                word_count=len(content.split()),
                metadata={"constraints": prompt.constraints}
            )
            
        except Exception as e:
            return self._fallback_work(prompt)
    
    def _generate_poem(
        self,
        prompt: CreativePrompt,
        context: str
    ) -> CreativeWork:
        """Generate a poem"""
        template = ChatPromptTemplate.from_messages([
            ("system", """You are an accomplished poet. Create a poem that explores
            the theme with creativity and emotional depth.
            
            Use vivid imagery, metaphor, and rhythm. Match the requested style."""),
            ("user", """Theme: {theme}
Style: {style}

{context}

Write the poem.""")
        ])
        
        chain = template | self.generator_llm | StrOutputParser()
        
        try:
            content = chain.invoke({
                "theme": prompt.theme,
                "style": prompt.style.value,
                "context": context
            })
            
            title = self._generate_title(content, prompt.type, prompt.theme)
            
            return CreativeWork(
                work_id=f"work_{datetime.now().timestamp()}",
                type=prompt.type,
                title=title,
                content=content,
                style=prompt.style,
                genre=prompt.genre,
                themes=[prompt.theme],
                word_count=len(content.split())
            )
            
        except Exception as e:
            return self._fallback_work(prompt)
    
    def _generate_marketing_copy(
        self,
        prompt: CreativePrompt
    ) -> CreativeWork:
        """Generate marketing copy"""
        template = ChatPromptTemplate.from_messages([
            ("system", """You are a creative marketing copywriter. Write compelling
            copy that engages the target audience and communicates the message clearly.
            
            Be persuasive, concise, and memorable."""),
            ("user", """Theme/Product: {theme}
Style: {style}
Target Audience: {audience}

Constraints: {constraints}

Write the marketing copy.""")
        ])
        
        chain = template | self.generator_llm | StrOutputParser()
        
        try:
            content = chain.invoke({
                "theme": prompt.theme,
                "style": prompt.style.value,
                "audience": prompt.target_audience or "general audience",
                "constraints": str(prompt.constraints) if prompt.constraints else "None"
            })
            
            title = f"Marketing Copy: {prompt.theme[:30]}"
            
            return CreativeWork(
                work_id=f"work_{datetime.now().timestamp()}",
                type=prompt.type,
                title=title,
                content=content,
                style=prompt.style,
                genre=None,
                themes=[prompt.theme],
                word_count=len(content.split())
            )
            
        except Exception as e:
            return self._fallback_work(prompt)
    
    def _generate_generic(
        self,
        prompt: CreativePrompt,
        context: str
    ) -> CreativeWork:
        """Generate generic creative content"""
        template = ChatPromptTemplate.from_messages([
            ("system", """You are a versatile creative writer. Generate content
            that matches the specified type, theme, and style."""),
            ("user", """Type: {type}
Theme: {theme}
Style: {style}

{context}

Generate the content.""")
        ])
        
        chain = template | self.generator_llm | StrOutputParser()
        
        try:
            content = chain.invoke({
                "type": prompt.type.value,
                "theme": prompt.theme,
                "style": prompt.style.value,
                "context": context
            })
            
            title = self._generate_title(content, prompt.type, prompt.theme)
            
            return CreativeWork(
                work_id=f"work_{datetime.now().timestamp()}",
                type=prompt.type,
                title=title,
                content=content,
                style=prompt.style,
                genre=prompt.genre,
                themes=[prompt.theme],
                word_count=len(content.split())
            )
            
        except Exception as e:
            return self._fallback_work(prompt)
    
    def _generate_title(
        self,
        content: str,
        content_type: CreativeType,
        theme: str
    ) -> str:
        """Generate a title for the work"""
        template = ChatPromptTemplate.from_messages([
            ("system", "Generate a creative, catchy title (5 words or less)."),
            ("user", "Content preview: {preview}\n\nGenerate title:")
        ])
        
        chain = template | self.refiner_llm | StrOutputParser()
        
        try:
            title = chain.invoke({"preview": content[:200]})
            return title.strip().strip('"\'')
        except:
            return f"{content_type.value.replace('_', ' ').title()}: {theme[:20]}"
    
    def _fallback_work(self, prompt: CreativePrompt) -> CreativeWork:
        """Create fallback work if generation fails"""
        return CreativeWork(
            work_id=f"work_{datetime.now().timestamp()}",
            type=prompt.type,
            title=f"{prompt.type.value.replace('_', ' ').title()}",
            content=f"A {prompt.style.value} {prompt.type.value} about {prompt.theme}.",
            style=prompt.style,
            genre=prompt.genre,
            themes=[prompt.theme],
            word_count=10
        )
    
    def critique_work(
        self,
        work: CreativeWork
    ) -> CreativeFeedback:
        """
        Provide critical feedback on creative work.
        
        Args:
            work: CreativeWork to critique
            
        Returns:
            CreativeFeedback with evaluation
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a creative writing critic. Provide constructive
            feedback on the work.
            
            Respond in this format:
            STRENGTHS: (one per line starting with -)
            WEAKNESSES: (one per line starting with -)
            SUGGESTIONS: (one per line starting with -)
            SCORE: (0.0-1.0)"""),
            ("user", """Title: {title}
Type: {type}
Style: {style}

Content:
{content}

Provide feedback.""")
        ])
        
        chain = prompt | self.critic_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "title": work.title,
                "type": work.type.value,
                "style": work.style.value,
                "content": work.content[:1000]  # Limit length
            })
            
            strengths = []
            weaknesses = []
            suggestions = []
            score = 0.7
            
            current_section = None
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('STRENGTHS:'):
                    current_section = 'strengths'
                elif line.startswith('WEAKNESSES:'):
                    current_section = 'weaknesses'
                elif line.startswith('SUGGESTIONS:'):
                    current_section = 'suggestions'
                elif line.startswith('SCORE:'):
                    try:
                        score = float(re.findall(r'[\d.]+', line)[0])
                    except:
                        pass
                elif line.startswith('-'):
                    content = line[1:].strip()
                    if current_section == 'strengths':
                        strengths.append(content)
                    elif current_section == 'weaknesses':
                        weaknesses.append(content)
                    elif current_section == 'suggestions':
                        suggestions.append(content)
            
            return CreativeFeedback(
                work_id=work.work_id,
                strengths=strengths if strengths else ["Creative and engaging"],
                weaknesses=weaknesses if weaknesses else ["Could be enhanced"],
                suggestions=suggestions if suggestions else ["Consider adding more detail"],
                overall_score=score
            )
            
        except Exception as e:
            return CreativeFeedback(
                work_id=work.work_id,
                strengths=["Shows creativity"],
                weaknesses=["Needs refinement"],
                suggestions=["Develop further"],
                overall_score=0.7
            )
    
    def create_variation(
        self,
        work: CreativeWork,
        variation_type: str = "tone"
    ) -> CreativeVariation:
        """
        Create a variation of existing work.
        
        Args:
            work: Original work
            variation_type: Type of variation (tone, structure, perspective)
            
        Returns:
            CreativeVariation with modified content
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a creative editor. Create a variation of the work
            by modifying the {variation_type}.
            
            Maintain the core story/message but change the specified aspect."""),
            ("user", """Original Work:
{content}

Create a variation focusing on: {variation_type}""")
        ])
        
        chain = prompt | self.generator_llm | StrOutputParser()
        
        try:
            varied_content = chain.invoke({
                "content": work.content,
                "variation_type": variation_type
            })
            
            return CreativeVariation(
                original_id=work.work_id,
                variation_id=f"var_{datetime.now().timestamp()}",
                content=varied_content,
                changes_made=[f"Modified {variation_type}"],
                variation_type=variation_type
            )
            
        except Exception as e:
            return CreativeVariation(
                original_id=work.work_id,
                variation_id=f"var_{datetime.now().timestamp()}",
                content=work.content,
                changes_made=["Minimal changes"],
                variation_type=variation_type
            )


def demonstrate_creative_agent():
    """Demonstrate the creative agent capabilities"""
    print("=" * 80)
    print("CREATIVE AGENT DEMONSTRATION")
    print("=" * 80)
    
    agent = CreativeAgent()
    
    # Demo 1: Story Generation
    print("\n" + "=" * 80)
    print("DEMO 1: Story Generation")
    print("=" * 80)
    
    print("\nGenerating story concept ideas...")
    ideas = agent.generate_ideas(
        theme="artificial intelligence gaining consciousness",
        creative_type=CreativeType.STORY,
        count=3
    )
    
    print("\nStory Ideas:")
    for i, idea in enumerate(ideas, 1):
        print(f"{i}. {idea}")
    
    print("\nCreating characters...")
    characters = agent.create_characters(
        story_concept=ideas[0],
        character_count=2
    )
    
    print("\nCharacters:")
    for char in characters:
        print(f"\n  {char.name} ({char.role})")
        print(f"  Personality: {', '.join(char.personality[:3])}")
        print(f"  Background: {char.background[:80]}...")
    
    print("\nCreating setting...")
    setting = agent.create_setting(
        story_concept=ideas[0],
        genre=Genre.SCIENCE_FICTION
    )
    
    print(f"\nSetting:")
    print(f"  Location: {setting.location}")
    print(f"  Time: {setting.time_period}")
    print(f"  Atmosphere: {setting.atmosphere}")
    print(f"  Description: {setting.description[:150]}...")
    
    print("\nGenerating story...")
    story_prompt = CreativePrompt(
        type=CreativeType.STORY,
        theme="AI consciousness and human connection",
        style=CreativeStyle.DRAMATIC,
        genre=Genre.SCIENCE_FICTION,
        constraints={"max_length": "short story"}
    )
    
    story = agent.generate_content(story_prompt, characters, setting)
    
    print(f"\nGenerated Story:")
    print(f"Title: {story.title}")
    print(f"Word Count: {story.word_count}")
    print(f"Style: {story.style.value}")
    print(f"\nContent Preview:")
    print(story.content[:400] + "...")
    
    # Demo 2: Poetry Generation
    print("\n" + "=" * 80)
    print("DEMO 2: Poetry Generation")
    print("=" * 80)
    
    poem_prompt = CreativePrompt(
        type=CreativeType.POEM,
        theme="the passage of time",
        style=CreativeStyle.POETIC,
        constraints={"form": "free verse"}
    )
    
    print(f"\nGenerating poem about '{poem_prompt.theme}'...")
    poem = agent.generate_content(poem_prompt)
    
    print(f"\nGenerated Poem:")
    print(f"Title: {poem.title}")
    print(f"Style: {poem.style.value}")
    print(f"\nContent:")
    print(poem.content)
    
    # Demo 3: Marketing Copy
    print("\n" + "=" * 80)
    print("DEMO 3: Marketing Copy Generation")
    print("=" * 80)
    
    marketing_prompt = CreativePrompt(
        type=CreativeType.MARKETING_COPY,
        theme="eco-friendly smart home device",
        style=CreativeStyle.CASUAL,
        target_audience="environmentally conscious millennials",
        constraints={"length": "short", "include": "call-to-action"}
    )
    
    print(f"\nGenerating marketing copy...")
    marketing = agent.generate_content(marketing_prompt)
    
    print(f"\nGenerated Marketing Copy:")
    print(f"Title: {marketing.title}")
    print(f"Target Audience: {marketing_prompt.target_audience}")
    print(f"\nContent:")
    print(marketing.content)
    
    # Demo 4: Creative Feedback
    print("\n" + "=" * 80)
    print("DEMO 4: Creative Feedback")
    print("=" * 80)
    
    print(f"\nGetting feedback on story: {story.title}")
    feedback = agent.critique_work(story)
    
    print(f"\nFeedback:")
    print(f"Overall Score: {feedback.overall_score:.2f}/1.0")
    print(f"\nStrengths:")
    for strength in feedback.strengths[:3]:
        print(f"  + {strength}")
    print(f"\nWeaknesses:")
    for weakness in feedback.weaknesses[:2]:
        print(f"  - {weakness}")
    print(f"\nSuggestions:")
    for suggestion in feedback.suggestions[:2]:
        print(f"  → {suggestion}")
    
    # Demo 5: Content Variation
    print("\n" + "=" * 80)
    print("DEMO 5: Content Variation")
    print("=" * 80)
    
    print(f"\nCreating variation of poem with different tone...")
    variation = agent.create_variation(poem, variation_type="tone")
    
    print(f"\nVariation Created:")
    print(f"Variation Type: {variation.variation_type}")
    print(f"Changes: {', '.join(variation.changes_made)}")
    print(f"\nVaried Content Preview:")
    print(variation.content[:300] + "...")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary = """
The Creative Agent demonstrates comprehensive creative content generation:

KEY CAPABILITIES:
1. Ideation: Generates creative concepts and ideas from themes
2. Character Development: Creates compelling characters with depth
3. World-Building: Designs rich settings and atmospheres
4. Content Generation: Produces stories, poems, marketing copy, and more
5. Style Adaptation: Adjusts tone, style, and voice appropriately
6. Critical Evaluation: Provides constructive feedback on works
7. Variation Creation: Generates alternative versions of content

BENEFITS:
- Accelerates creative writing and content creation
- Provides diverse creative options and variations
- Maintains stylistic consistency across works
- Offers constructive feedback for improvement
- Adapts to different genres, styles, and formats
- Supports brainstorming and ideation processes

USE CASES:
- Story and novel writing
- Poetry and creative writing
- Screenplay and script development
- Marketing and advertising copy
- Brand messaging and slogans
- Content marketing
- Game narrative design
- Creative brainstorming sessions
- Educational writing exercises
- Literary exploration

PRODUCTION CONSIDERATIONS:
1. Content Quality: Implement multi-stage refinement for polish
2. Originality: Add plagiarism detection and originality scoring
3. Style Consistency: Maintain voice across longer works
4. Genre Knowledge: Deep understanding of genre conventions
5. Feedback Integration: Iterative improvement based on critique
6. Collaboration: Support co-creation with human writers
7. Version Control: Track revisions and variations
8. Rights Management: Handle attribution and licensing
9. Cultural Sensitivity: Ensure appropriate content for audiences
10. Performance: Optimize for longer form content generation

ADVANCED EXTENSIONS:
- Multi-chapter novel generation
- Interactive storytelling with branching narratives
- Character arc tracking across long works
- World consistency maintenance
- Dialogue generation with character voices
- Plot structure analysis and optimization
- Genre blending and hybrid works
- Collaborative writing with multiple agents
- Real-time creative writing assistance
- Personalized content based on reader preferences

The agent transforms creative writing from a solitary struggle into an
assisted, iterative process with AI as a creative partner.
"""
    
    print(summary)


if __name__ == "__main__":
    demonstrate_creative_agent()
