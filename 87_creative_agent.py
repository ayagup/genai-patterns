"""
Creative Agent Pattern

Enables agents to perform creative tasks by generating novel outputs through
various creative processes and techniques.

Key Concepts:
- Divergent thinking (generating many ideas)
- Convergent thinking (refining ideas)
- Creative constraints
- Style transfer
- Iterative refinement

Use Cases:
- Creative writing (stories, poems, scripts)
- Art generation
- Music composition
- Design ideation
- Brainstorming
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random
import uuid


class CreativeMode(Enum):
    """Modes of creative thinking."""
    DIVERGENT = "divergent"  # Generate many ideas
    CONVERGENT = "convergent"  # Refine and select
    ASSOCIATIVE = "associative"  # Make connections
    TRANSFORMATIVE = "transformative"  # Transform existing ideas
    COMBINATORIAL = "combinatorial"  # Combine elements


class CreativeDomain(Enum):
    """Creative domains."""
    WRITING = "writing"
    ART = "art"
    MUSIC = "music"
    DESIGN = "design"
    PRODUCT = "product"
    CONCEPT = "concept"


class StyleAttribute(Enum):
    """Style attributes for creative work."""
    FORMAL = "formal"
    CASUAL = "casual"
    POETIC = "poetic"
    MINIMALIST = "minimalist"
    ELABORATE = "elaborate"
    PLAYFUL = "playful"
    SERIOUS = "serious"
    ABSTRACT = "abstract"
    CONCRETE = "concrete"


@dataclass
class CreativeConstraint:
    """Constraint for creative generation."""
    constraint_type: str
    value: Any
    description: str
    
    def is_satisfied(self, output: Any) -> bool:
        """Check if output satisfies constraint."""
        # Simplified constraint checking
        if self.constraint_type == "length":
            if isinstance(output, str):
                return len(output.split()) <= self.value
        elif self.constraint_type == "theme":
            if isinstance(output, str):
                return self.value.lower() in output.lower()
        return True


@dataclass
class CreativeIdea:
    """A creative idea or concept."""
    idea_id: str
    content: str
    domain: CreativeDomain
    novelty_score: float  # 0-1
    quality_score: float  # 0-1
    style_attributes: Set[StyleAttribute] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    parent_ideas: List[str] = field(default_factory=list)  # For tracking lineage
    
    def overall_score(self, novelty_weight: float = 0.5) -> float:
        """Calculate overall creative score."""
        quality_weight = 1.0 - novelty_weight
        return (
            novelty_weight * self.novelty_score +
            quality_weight * self.quality_score
        )


@dataclass
class CreativePrompt:
    """Prompt for creative generation."""
    text: str
    domain: CreativeDomain
    constraints: List[CreativeConstraint] = field(default_factory=list)
    style_attributes: Set[StyleAttribute] = field(default_factory=set)
    reference_works: List[str] = field(default_factory=list)


class CreativeMemory:
    """Memory system for creative agent."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.ideas: Dict[str, CreativeIdea] = {}
        self.inspiration_pool: List[str] = []
    
    def store_idea(self, idea: CreativeIdea) -> None:
        """Store a creative idea."""
        self.ideas[idea.idea_id] = idea
        
        # Add to inspiration pool
        if len(self.inspiration_pool) < self.capacity:
            self.inspiration_pool.append(idea.content)
        else:
            # Replace random element
            idx = random.randint(0, len(self.inspiration_pool) - 1)
            self.inspiration_pool[idx] = idea.content
    
    def get_inspiration(self, n: int = 3) -> List[str]:
        """Get random inspiration from memory."""
        return random.sample(
            self.inspiration_pool,
            min(n, len(self.inspiration_pool))
        )
    
    def find_similar_ideas(
        self,
        domain: CreativeDomain,
        min_score: float = 0.5
    ) -> List[CreativeIdea]:
        """Find similar high-quality ideas."""
        return [
            idea for idea in self.ideas.values()
            if idea.domain == domain and idea.overall_score() >= min_score
        ]


class CreativeAgent:
    """Agent that performs creative tasks."""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.memory = CreativeMemory()
        self.current_mode = CreativeMode.DIVERGENT
        
        # Creative parameters
        self.temperature = 0.8  # Higher = more creative/random
        self.novelty_preference = 0.6  # Weight given to novelty vs quality
    
    def generate(
        self,
        prompt: CreativePrompt,
        num_ideas: int = 5,
        mode: Optional[CreativeMode] = None
    ) -> List[CreativeIdea]:
        """Generate creative ideas based on prompt."""
        mode = mode or self.current_mode
        self.current_mode = mode
        
        print(f"\n[{self.name}] Generating {num_ideas} ideas in {mode.value} mode")
        print(f"Prompt: {prompt.text}")
        
        ideas = []
        
        if mode == CreativeMode.DIVERGENT:
            ideas = self._divergent_generation(prompt, num_ideas)
        elif mode == CreativeMode.CONVERGENT:
            ideas = self._convergent_generation(prompt, num_ideas)
        elif mode == CreativeMode.ASSOCIATIVE:
            ideas = self._associative_generation(prompt, num_ideas)
        elif mode == CreativeMode.COMBINATORIAL:
            ideas = self._combinatorial_generation(prompt, num_ideas)
        else:
            ideas = self._transformative_generation(prompt, num_ideas)
        
        # Store in memory
        for idea in ideas:
            self.memory.store_idea(idea)
        
        return ideas
    
    def _divergent_generation(
        self,
        prompt: CreativePrompt,
        num_ideas: int
    ) -> List[CreativeIdea]:
        """Generate diverse ideas through divergent thinking."""
        ideas = []
        
        # Generate varied ideas
        variations = [
            "unconventional",
            "traditional with a twist",
            "futuristic",
            "minimalist",
            "maximalist",
            "abstract",
            "literal",
            "metaphorical"
        ]
        
        for i in range(num_ideas):
            variation = variations[i % len(variations)]
            
            if prompt.domain == CreativeDomain.WRITING:
                content = self._generate_story_idea(prompt, variation)
            elif prompt.domain == CreativeDomain.ART:
                content = self._generate_art_concept(prompt, variation)
            elif prompt.domain == CreativeDomain.MUSIC:
                content = self._generate_music_idea(prompt, variation)
            else:
                content = self._generate_design_idea(prompt, variation)
            
            idea = CreativeIdea(
                idea_id=str(uuid.uuid4()),
                content=content,
                domain=prompt.domain,
                novelty_score=random.uniform(0.6, 0.9),
                quality_score=random.uniform(0.4, 0.7),
                style_attributes=prompt.style_attributes
            )
            
            ideas.append(idea)
        
        return ideas
    
    def _convergent_generation(
        self,
        prompt: CreativePrompt,
        num_ideas: int
    ) -> List[CreativeIdea]:
        """Refine and improve existing ideas."""
        # Get best previous ideas
        similar = self.memory.find_similar_ideas(prompt.domain)
        
        if not similar:
            # Fall back to divergent if no history
            return self._divergent_generation(prompt, num_ideas)
        
        # Refine top ideas
        ideas = []
        for i in range(min(num_ideas, len(similar))):
            base_idea = similar[i]
            
            refined_content = f"{base_idea.content} [Refined: Enhanced with {prompt.text}]"
            
            idea = CreativeIdea(
                idea_id=str(uuid.uuid4()),
                content=refined_content,
                domain=prompt.domain,
                novelty_score=base_idea.novelty_score * 0.9,  # Slightly less novel
                quality_score=min(1.0, base_idea.quality_score * 1.2),  # Higher quality
                parent_ideas=[base_idea.idea_id]
            )
            
            ideas.append(idea)
        
        return ideas
    
    def _associative_generation(
        self,
        prompt: CreativePrompt,
        num_ideas: int
    ) -> List[CreativeIdea]:
        """Generate ideas through association."""
        ideas = []
        
        # Get inspiration from memory
        inspiration = self.memory.get_inspiration(3)
        
        for i in range(num_ideas):
            # Make associations
            if inspiration:
                inspo = inspiration[i % len(inspiration)]
                content = f"Combining '{prompt.text}' with inspiration from '{inspo[:50]}...'"
            else:
                content = f"Association-based idea for: {prompt.text}"
            
            idea = CreativeIdea(
                idea_id=str(uuid.uuid4()),
                content=content,
                domain=prompt.domain,
                novelty_score=random.uniform(0.5, 0.8),
                quality_score=random.uniform(0.5, 0.8)
            )
            
            ideas.append(idea)
        
        return ideas
    
    def _combinatorial_generation(
        self,
        prompt: CreativePrompt,
        num_ideas: int
    ) -> List[CreativeIdea]:
        """Combine existing elements in novel ways."""
        ideas = []
        
        elements = [
            "Element A: Classic structure",
            "Element B: Modern twist",
            "Element C: Unexpected detail",
            "Element D: Emotional core"
        ]
        
        for i in range(num_ideas):
            # Combine random elements
            selected = random.sample(elements, k=min(3, len(elements)))
            content = f"{prompt.text} - Combining: {', '.join(selected)}"
            
            idea = CreativeIdea(
                idea_id=str(uuid.uuid4()),
                content=content,
                domain=prompt.domain,
                novelty_score=random.uniform(0.6, 0.85),
                quality_score=random.uniform(0.5, 0.75)
            )
            
            ideas.append(idea)
        
        return ideas
    
    def _transformative_generation(
        self,
        prompt: CreativePrompt,
        num_ideas: int
    ) -> List[CreativeIdea]:
        """Transform existing ideas into new forms."""
        ideas = []
        
        transformations = [
            "inverting the concept",
            "changing the perspective",
            "scaling up/down",
            "changing the medium",
            "adding constraints"
        ]
        
        for i in range(num_ideas):
            transform = transformations[i % len(transformations)]
            content = f"{prompt.text} - Transformed by {transform}"
            
            idea = CreativeIdea(
                idea_id=str(uuid.uuid4()),
                content=content,
                domain=prompt.domain,
                novelty_score=random.uniform(0.7, 0.9),
                quality_score=random.uniform(0.4, 0.7)
            )
            
            ideas.append(idea)
        
        return ideas
    
    def _generate_story_idea(self, prompt: CreativePrompt, variation: str) -> str:
        """Generate a story idea."""
        themes = ["adventure", "mystery", "romance", "conflict", "discovery"]
        theme = random.choice(themes)
        return f"Story ({variation}): {prompt.text} - A {theme}-driven narrative with unexpected twists"
    
    def _generate_art_concept(self, prompt: CreativePrompt, variation: str) -> str:
        """Generate an art concept."""
        styles = ["impressionist", "abstract", "surreal", "photorealistic", "expressionist"]
        style = random.choice(styles)
        return f"Art ({variation}): {prompt.text} - {style} interpretation with bold colors"
    
    def _generate_music_idea(self, prompt: CreativePrompt, variation: str) -> str:
        """Generate a music idea."""
        moods = ["melancholic", "uplifting", "mysterious", "energetic", "peaceful"]
        mood = random.choice(moods)
        return f"Music ({variation}): {prompt.text} - {mood} composition in unconventional time signature"
    
    def _generate_design_idea(self, prompt: CreativePrompt, variation: str) -> str:
        """Generate a design idea."""
        principles = ["symmetry", "asymmetry", "minimalism", "ornate detail", "negative space"]
        principle = random.choice(principles)
        return f"Design ({variation}): {prompt.text} - Emphasizing {principle}"
    
    def evaluate_ideas(
        self,
        ideas: List[CreativeIdea],
        select_top_n: int = 3
    ) -> List[CreativeIdea]:
        """Evaluate and rank creative ideas."""
        print(f"\n[{self.name}] Evaluating {len(ideas)} ideas")
        
        # Sort by overall score
        ranked = sorted(
            ideas,
            key=lambda x: x.overall_score(self.novelty_preference),
            reverse=True
        )
        
        top_ideas = ranked[:select_top_n]
        
        print(f"Selected top {len(top_ideas)} ideas:")
        for i, idea in enumerate(top_ideas, 1):
            print(f"  {i}. Score: {idea.overall_score(self.novelty_preference):.2f} "
                  f"(N: {idea.novelty_score:.2f}, Q: {idea.quality_score:.2f})")
            print(f"     {idea.content[:80]}...")
        
        return top_ideas
    
    def iterate_on_idea(
        self,
        idea: CreativeIdea,
        feedback: str
    ) -> CreativeIdea:
        """Iterate and improve an idea based on feedback."""
        print(f"\n[{self.name}] Iterating on idea with feedback: {feedback}")
        
        improved_content = f"{idea.content}\n[Iteration: {feedback}]"
        
        # Create improved version
        new_idea = CreativeIdea(
            idea_id=str(uuid.uuid4()),
            content=improved_content,
            domain=idea.domain,
            novelty_score=idea.novelty_score * 0.95,
            quality_score=min(1.0, idea.quality_score * 1.15),
            style_attributes=idea.style_attributes,
            parent_ideas=[idea.idea_id]
        )
        
        self.memory.store_idea(new_idea)
        
        return new_idea


def demonstrate_creative_agent():
    """Demonstrate creative agent pattern."""
    print("=" * 60)
    print("CREATIVE AGENT DEMONSTRATION")
    print("=" * 60)
    
    # Create creative agent
    agent = CreativeAgent("creative1", "CreativeAI")
    
    # Example 1: Divergent story generation
    print("\n" + "=" * 60)
    print("1. Divergent Story Generation")
    print("=" * 60)
    
    story_prompt = CreativePrompt(
        text="A detective who can see into alternate timelines",
        domain=CreativeDomain.WRITING,
        style_attributes={StyleAttribute.SERIOUS, StyleAttribute.ABSTRACT}
    )
    
    story_ideas = agent.generate(
        story_prompt,
        num_ideas=5,
        mode=CreativeMode.DIVERGENT
    )
    
    for i, idea in enumerate(story_ideas, 1):
        print(f"\n  Idea {i}: {idea.content}")
    
    # Example 2: Select and refine best ideas
    print("\n" + "=" * 60)
    print("2. Convergent Refinement")
    print("=" * 60)
    
    top_stories = agent.evaluate_ideas(story_ideas, select_top_n=2)
    
    refined_ideas = agent.generate(
        story_prompt,
        num_ideas=2,
        mode=CreativeMode.CONVERGENT
    )
    
    # Example 3: Art concept generation
    print("\n" + "=" * 60)
    print("3. Art Concept Generation (Combinatorial)")
    print("=" * 60)
    
    art_prompt = CreativePrompt(
        text="Urban landscape meets natural elements",
        domain=CreativeDomain.ART,
        style_attributes={StyleAttribute.ABSTRACT, StyleAttribute.MINIMALIST}
    )
    
    art_ideas = agent.generate(
        art_prompt,
        num_ideas=4,
        mode=CreativeMode.COMBINATORIAL
    )
    
    for i, idea in enumerate(art_ideas, 1):
        print(f"\n  Concept {i}: {idea.content}")
    
    # Example 4: Iterative improvement
    print("\n" + "=" * 60)
    print("4. Iterative Improvement")
    print("=" * 60)
    
    best_art = agent.evaluate_ideas(art_ideas, select_top_n=1)[0]
    print(f"\nStarting with: {best_art.content[:80]}...")
    
    iteration1 = agent.iterate_on_idea(
        best_art,
        "Add more emphasis on the contrast between urban and natural"
    )
    
    iteration2 = agent.iterate_on_idea(
        iteration1,
        "Incorporate interactive elements for viewer engagement"
    )
    
    print(f"\nFinal version: {iteration2.content[:120]}...")
    
    # Show memory statistics
    print("\n" + "=" * 60)
    print("5. Creative Memory Statistics")
    print("=" * 60)
    
    print(f"\nTotal ideas generated: {len(agent.memory.ideas)}")
    print(f"Inspiration pool size: {len(agent.memory.inspiration_pool)}")


if __name__ == "__main__":
    demonstrate_creative_agent()
