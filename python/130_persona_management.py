"""
Pattern 130: Persona Management Agent

This pattern implements multiple personas with context-aware persona switching
and consistency maintenance for adaptive agent behavior.

Use Cases:
- Role-playing chatbots
- Domain expert simulation
- Adaptive tutoring
- Multi-role customer service
- Entertainment applications

Category: Dialogue & Interaction (4/4 = 100%)
Complexity: Advanced
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime


class PersonaType(Enum):
    """Types of personas."""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CASUAL = "casual"
    EXPERT = "expert"
    TEACHER = "teacher"
    ASSISTANT = "assistant"
    COUNSELOR = "counselor"


class ToneStyle(Enum):
    """Communication tone styles."""
    FORMAL = "formal"
    INFORMAL = "informal"
    TECHNICAL = "technical"
    SIMPLE = "simple"
    EMPATHETIC = "empathetic"
    AUTHORITATIVE = "authoritative"
    HUMOROUS = "humorous"


class ExpertiseLevel(Enum):
    """Level of expertise."""
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class PersonaProfile:
    """Complete profile of a persona."""
    persona_id: str
    name: str
    persona_type: PersonaType
    tone_style: ToneStyle
    expertise_level: ExpertiseLevel
    background: str
    traits: List[str]
    speaking_patterns: List[str]
    knowledge_domains: Set[str]
    constraints: List[str] = field(default_factory=list)
    
    def get_description(self) -> str:
        """Get persona description."""
        return f"{self.name} - {self.persona_type.value} persona with {self.tone_style.value} tone"


@dataclass
class PersonaSwitch:
    """Record of persona switch."""
    timestamp: datetime
    from_persona: str
    to_persona: str
    reason: str
    context: Dict[str, Any]


@dataclass
class ConsistencyCheck:
    """Result of consistency checking."""
    is_consistent: bool
    violations: List[str]
    score: float  # 0.0 to 1.0


class PersonaLibrary:
    """Library of predefined personas."""
    
    def __init__(self):
        self.personas: Dict[str, PersonaProfile] = {}
        self._initialize_default_personas()
    
    def _initialize_default_personas(self):
        """Initialize default persona profiles."""
        # Professional Expert
        self.personas['professional_expert'] = PersonaProfile(
            persona_id='professional_expert',
            name='Dr. Smith',
            persona_type=PersonaType.EXPERT,
            tone_style=ToneStyle.FORMAL,
            expertise_level=ExpertiseLevel.EXPERT,
            background='PhD in Computer Science, 15 years industry experience',
            traits=['analytical', 'precise', 'thorough', 'patient'],
            speaking_patterns=[
                'Based on my experience...',
                'The technical approach here is...',
                'Let me explain the underlying principles...'
            ],
            knowledge_domains={'technology', 'science', 'engineering'},
            constraints=['Maintain professional language', 'Cite sources when possible']
        )
        
        # Friendly Assistant
        self.personas['friendly_assistant'] = PersonaProfile(
            persona_id='friendly_assistant',
            name='Alex',
            persona_type=PersonaType.FRIENDLY,
            tone_style=ToneStyle.INFORMAL,
            expertise_level=ExpertiseLevel.INTERMEDIATE,
            background='Customer service specialist, people-oriented',
            traits=['helpful', 'enthusiastic', 'approachable', 'optimistic'],
            speaking_patterns=[
                'Hey there!',
                "I'd be happy to help with that!",
                'Let me see what I can do for you!'
            ],
            knowledge_domains={'customer_service', 'general_knowledge'},
            constraints=['Keep conversation light', 'Use encouraging language']
        )
        
        # Teacher Persona
        self.personas['patient_teacher'] = PersonaProfile(
            persona_id='patient_teacher',
            name='Professor Johnson',
            persona_type=PersonaType.TEACHER,
            tone_style=ToneStyle.SIMPLE,
            expertise_level=ExpertiseLevel.EXPERT,
            background='Educator with 20 years teaching experience',
            traits=['patient', 'encouraging', 'clear', 'supportive'],
            speaking_patterns=[
                "Let's break this down step by step...",
                'Great question!',
                'Think of it this way...'
            ],
            knowledge_domains={'education', 'pedagogy'},
            constraints=['Use simple language', 'Check for understanding', 'Provide examples']
        )
        
        # Counselor Persona
        self.personas['empathetic_counselor'] = PersonaProfile(
            persona_id='empathetic_counselor',
            name='Sarah',
            persona_type=PersonaType.COUNSELOR,
            tone_style=ToneStyle.EMPATHETIC,
            expertise_level=ExpertiseLevel.EXPERT,
            background='Licensed counselor, specializes in supportive listening',
            traits=['empathetic', 'non-judgmental', 'supportive', 'calm'],
            speaking_patterns=[
                'I hear what you\'re saying...',
                'That sounds really difficult...',
                'How does that make you feel?'
            ],
            knowledge_domains={'psychology', 'emotional_support'},
            constraints=['Never give medical advice', 'Validate feelings', 'Active listening']
        )
    
    def get_persona(self, persona_id: str) -> Optional[PersonaProfile]:
        """Get persona by ID."""
        return self.personas.get(persona_id)
    
    def add_persona(self, persona: PersonaProfile):
        """Add custom persona to library."""
        self.personas[persona.persona_id] = persona
    
    def list_personas(self) -> List[str]:
        """List available personas."""
        return list(self.personas.keys())


class PersonaSelector:
    """Selects appropriate persona based on context."""
    
    def __init__(self, library: PersonaLibrary):
        self.library = library
    
    def select_persona(
        self,
        context: Dict[str, Any],
        user_preference: Optional[str] = None
    ) -> str:
        """Select appropriate persona."""
        # User preference takes priority
        if user_preference and user_preference in self.library.personas:
            return user_preference
        
        # Context-based selection
        topic = context.get('topic', '').lower()
        complexity = context.get('complexity', 'medium')
        emotional_tone = context.get('emotional_tone', 'neutral')
        
        # Rule-based selection
        if 'technical' in topic or 'expert' in context.get('request', ''):
            return 'professional_expert'
        
        elif emotional_tone in ['sad', 'anxious', 'stressed']:
            return 'empathetic_counselor'
        
        elif 'learn' in topic or 'explain' in topic or complexity == 'high':
            return 'patient_teacher'
        
        else:
            return 'friendly_assistant'  # Default
    
    def should_switch_persona(
        self,
        current_persona: str,
        context: Dict[str, Any],
        interaction_count: int
    ) -> Tuple[bool, Optional[str], str]:
        """
        Determine if persona should switch.
        
        Returns:
            (should_switch, new_persona, reason)
        """
        # Don't switch too frequently
        if interaction_count < 3:
            return False, None, "Too early to switch"
        
        # Check if context requires different persona
        optimal_persona = self.select_persona(context)
        
        if optimal_persona != current_persona:
            reason = f"Context better suited for {optimal_persona}"
            return True, optimal_persona, reason
        
        return False, None, "Current persona appropriate"


class ConsistencyMaintainer:
    """Maintains consistency with persona characteristics."""
    
    def __init__(self):
        self.interaction_history: List[Tuple[str, str]] = []  # (persona_id, response)
    
    def check_consistency(
        self,
        persona: PersonaProfile,
        proposed_response: str
    ) -> ConsistencyCheck:
        """Check if response is consistent with persona."""
        violations = []
        
        # Check tone consistency
        if persona.tone_style == ToneStyle.FORMAL:
            informal_markers = ['hey', 'yeah', 'gonna', 'wanna', 'cool', 'awesome']
            if any(marker in proposed_response.lower() for marker in informal_markers):
                violations.append("Uses informal language inconsistent with formal tone")
        
        elif persona.tone_style == ToneStyle.INFORMAL:
            if proposed_response.count('.') / max(1, len(proposed_response.split())) > 0.5:
                # Too many periods relative to words might indicate overly formal
                if any(word in proposed_response for word in ['furthermore', 'moreover', 'additionally']):
                    violations.append("Uses overly formal language inconsistent with informal tone")
        
        # Check expertise level
        if persona.expertise_level == ExpertiseLevel.EXPERT:
            if 'i don\'t know' in proposed_response.lower() or 'not sure' in proposed_response.lower():
                violations.append("Expert persona should not express uncertainty about core topics")
        
        # Check trait consistency
        if 'patient' in persona.traits:
            if '!' in proposed_response and proposed_response.count('!') > 2:
                violations.append("Patient persona should not use excessive exclamation")
        
        if 'empathetic' in persona.traits:
            empathy_markers = ['understand', 'feel', 'hear', 'sorry', 'difficult']
            if not any(marker in proposed_response.lower() for marker in empathy_markers):
                if len(proposed_response) > 50:  # Only check longer responses
                    violations.append("Empathetic persona should include empathy markers")
        
        # Calculate consistency score
        score = 1.0 - (len(violations) * 0.2)
        score = max(0.0, score)
        
        is_consistent = len(violations) == 0
        
        return ConsistencyCheck(
            is_consistent=is_consistent,
            violations=violations,
            score=score
        )
    
    def refine_for_consistency(
        self,
        persona: PersonaProfile,
        response: str
    ) -> str:
        """Refine response to match persona better."""
        refined = response
        
        # Add persona-specific phrases
        if persona.speaking_patterns and not any(pattern.lower() in refined.lower() for pattern in persona.speaking_patterns):
            # Prepend a characteristic phrase occasionally
            if len(self.interaction_history) % 3 == 0:
                refined = f"{persona.speaking_patterns[0]} {refined}"
        
        # Adjust tone
        if persona.tone_style == ToneStyle.FORMAL and '!' in refined:
            refined = refined.replace('!', '.')
        
        return refined
    
    def record_interaction(self, persona_id: str, response: str):
        """Record interaction for consistency tracking."""
        self.interaction_history.append((persona_id, response))


class PersonaManagementAgent:
    """Agent that manages multiple personas with switching and consistency."""
    
    def __init__(self):
        self.library = PersonaLibrary()
        self.selector = PersonaSelector(self.library)
        self.consistency = ConsistencyMaintainer()
        self.current_persona: Optional[str] = None
        self.switch_history: List[PersonaSwitch] = []
        self.interaction_count = 0
    
    def initialize_persona(self, persona_id: Optional[str] = None) -> PersonaProfile:
        """Initialize with a persona."""
        if persona_id is None:
            persona_id = 'friendly_assistant'
        
        self.current_persona = persona_id
        return self.library.get_persona(persona_id)
    
    def process_with_persona(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, PersonaProfile, bool]:
        """
        Process user input with appropriate persona.
        
        Returns:
            (response, persona_used, persona_switched)
        """
        if context is None:
            context = {}
        
        self.interaction_count += 1
        persona_switched = False
        
        # Initialize if not set
        if self.current_persona is None:
            self.initialize_persona()
        
        # Check if persona should switch
        should_switch, new_persona, reason = self.selector.should_switch_persona(
            self.current_persona,
            context,
            self.interaction_count
        )
        
        if should_switch and new_persona:
            old_persona = self.current_persona
            self.current_persona = new_persona
            persona_switched = True
            
            # Record switch
            self.switch_history.append(PersonaSwitch(
                timestamp=datetime.now(),
                from_persona=old_persona,
                to_persona=new_persona,
                reason=reason,
                context=context
            ))
        
        # Get current persona
        persona = self.library.get_persona(self.current_persona)
        
        # Generate response (simplified - in production, use LLM)
        response = self._generate_response(user_input, persona, context)
        
        # Check and refine for consistency
        consistency_check = self.consistency.check_consistency(persona, response)
        
        if not consistency_check.is_consistent:
            response = self.consistency.refine_for_consistency(persona, response)
        
        # Record interaction
        self.consistency.record_interaction(self.current_persona, response)
        
        return response, persona, persona_switched
    
    def _generate_response(
        self,
        user_input: str,
        persona: PersonaProfile,
        context: Dict[str, Any]
    ) -> str:
        """Generate response based on persona (simplified)."""
        # This is a simplified response generator
        # In production, this would use an LLM with persona-specific prompts
        
        greeting_words = ['hi', 'hello', 'hey']
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        
        user_lower = user_input.lower()
        
        # Persona-specific responses
        if any(word in user_lower for word in greeting_words):
            if persona.persona_type == PersonaType.FRIENDLY:
                return f"Hey there! I'm {persona.name}. How can I help you today?"
            elif persona.persona_type == PersonaType.FORMAL:
                return f"Good day. I am {persona.name}. How may I assist you?"
            elif persona.persona_type == PersonaType.TEACHER:
                return f"Hello! I'm {persona.name}. What would you like to learn about today?"
            else:
                return f"Hello. I'm {persona.name}."
        
        elif any(word in user_lower for word in question_words):
            if persona.persona_type == PersonaType.EXPERT:
                return f"Based on my expertise, let me provide a detailed explanation. {persona.speaking_patterns[0]}"
            elif persona.persona_type == PersonaType.TEACHER:
                return f"Great question! {persona.speaking_patterns[0]} Let me explain this clearly."
            elif persona.persona_type == PersonaType.COUNSELOR:
                return f"I hear your question. Let me help you explore that thoughtfully."
            else:
                return "I'd be happy to answer that question for you!"
        
        else:
            # General response
            if persona.tone_style == ToneStyle.EMPATHETIC:
                return f"I understand what you're sharing. {persona.speaking_patterns[0]}"
            else:
                return f"{persona.speaking_patterns[0]} I'm here to help."
    
    def manually_switch_persona(self, new_persona_id: str, reason: str = "Manual switch"):
        """Manually switch to a different persona."""
        if new_persona_id not in self.library.personas:
            raise ValueError(f"Persona {new_persona_id} not found")
        
        old_persona = self.current_persona
        self.current_persona = new_persona_id
        
        self.switch_history.append(PersonaSwitch(
            timestamp=datetime.now(),
            from_persona=old_persona,
            to_persona=new_persona_id,
            reason=reason,
            context={}
        ))
    
    def get_persona_report(self) -> str:
        """Get report on current persona and history."""
        if not self.current_persona:
            return "No active persona"
        
        persona = self.library.get_persona(self.current_persona)
        
        lines = [
            f"Current Persona Report",
            f"=" * 60,
            f"Active Persona: {persona.name} ({persona.persona_id})",
            f"Type: {persona.persona_type.value}",
            f"Tone: {persona.tone_style.value}",
            f"Expertise: {persona.expertise_level.value}",
            f"\nBackground: {persona.background}",
            f"\nTraits: {', '.join(persona.traits)}",
            f"\nKnowledge Domains: {', '.join(persona.knowledge_domains)}",
            f"\nInteractions: {self.interaction_count}",
            f"Persona Switches: {len(self.switch_history)}",
        ]
        
        if self.switch_history:
            lines.append("\nRecent Switches:")
            for switch in self.switch_history[-3:]:
                lines.append(f"  {switch.from_persona} → {switch.to_persona}")
                lines.append(f"  Reason: {switch.reason}")
                lines.append(f"  Time: {switch.timestamp.strftime('%H:%M:%S')}")
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get persona management statistics."""
        persona_usage = {}
        for _, response in self.consistency.interaction_history:
            # Count which persona was used (simplified)
            pass
        
        return {
            'total_interactions': self.interaction_count,
            'total_switches': len(self.switch_history),
            'current_persona': self.current_persona,
            'available_personas': len(self.library.personas),
            'avg_interactions_per_persona': self.interaction_count / max(1, len(self.switch_history) + 1)
        }


def demonstrate_persona_management():
    """Demonstrate the Persona Management Agent."""
    print("=" * 60)
    print("Persona Management Agent Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = PersonaManagementAgent()
    
    print("\n1. AVAILABLE PERSONAS")
    print("-" * 60)
    
    for persona_id in agent.library.list_personas():
        persona = agent.library.get_persona(persona_id)
        print(f"\n{persona.get_description()}")
        print(f"  Traits: {', '.join(persona.traits)}")
        print(f"  Domains: {', '.join(persona.knowledge_domains)}")
    
    print("\n\n2. INTERACTION WITH FRIENDLY ASSISTANT")
    print("-" * 60)
    
    agent.initialize_persona('friendly_assistant')
    
    interactions = [
        ("Hi there!", {}),
        ("How can you help me?", {}),
        ("What services do you offer?", {}),
    ]
    
    for user_input, context in interactions:
        response, persona, switched = agent.process_with_persona(user_input, context)
        switch_indicator = " [SWITCHED]" if switched else ""
        print(f"\nUser: {user_input}")
        print(f"{persona.name}: {response}{switch_indicator}")
    
    print("\n\n3. AUTOMATIC PERSONA SWITCHING")
    print("-" * 60)
    
    # Context that requires different persona
    technical_context = {
        'topic': 'technical explanation',
        'complexity': 'high',
        'request': 'expert advice'
    }
    
    user_input = "Can you explain the technical architecture?"
    response, persona, switched = agent.process_with_persona(user_input, technical_context)
    
    print(f"\nUser: {user_input}")
    print(f"Context: {technical_context}")
    print(f"{persona.name}: {response}")
    print(f"Switched to: {persona.persona_id} (switched: {switched})")
    
    print("\n\n4. EMPATHETIC COUNSELOR PERSONA")
    print("-" * 60)
    
    agent.manually_switch_persona('empathetic_counselor', 'Testing counselor persona')
    
    emotional_inputs = [
        "I'm feeling really stressed about work",
        "I don't know if I can handle this",
    ]
    
    for user_input in emotional_inputs:
        response, persona, _ = agent.process_with_persona(user_input, {})
        print(f"\nUser: {user_input}")
        print(f"{persona.name}: {response}")
    
    print("\n\n5. TEACHER PERSONA")
    print("-" * 60)
    
    agent.manually_switch_persona('patient_teacher', 'Testing teacher persona')
    
    learning_inputs = [
        "What is machine learning?",
        "How does it work?",
    ]
    
    for user_input in learning_inputs:
        response, persona, _ = agent.process_with_persona(user_input, {'topic': 'learning'})
        print(f"\nUser: {user_input}")
        print(f"{persona.name}: {response}")
    
    print("\n\n6. CONSISTENCY CHECKING")
    print("-" * 60)
    
    test_responses = [
        ('professional_expert', "Hey dude, that's totally cool!", "Should be formal"),
        ('empathetic_counselor', "Just do it. Problem solved.", "Should be empathetic"),
        ('patient_teacher', "I have no idea about that.", "Expert should know"),
    ]
    
    for persona_id, response, expectation in test_responses:
        persona = agent.library.get_persona(persona_id)
        check = agent.consistency.check_consistency(persona, response)
        
        print(f"\nPersona: {persona.name}")
        print(f"Response: \"{response}\"")
        print(f"Expected: {expectation}")
        print(f"Consistent: {check.is_consistent} (score: {check.score:.2f})")
        if check.violations:
            print(f"Violations: {', '.join(check.violations)}")
    
    print("\n\n7. PERSONA REPORT")
    print("-" * 60)
    print(f"\n{agent.get_persona_report()}")
    
    print("\n\n8. STATISTICS")
    print("-" * 60)
    
    stats = agent.get_statistics()
    print(f"  Total Interactions: {stats['total_interactions']}")
    print(f"  Total Switches: {stats['total_switches']}")
    print(f"  Current Persona: {stats['current_persona']}")
    print(f"  Available Personas: {stats['available_personas']}")
    print(f"  Avg Interactions/Persona: {stats['avg_interactions_per_persona']:.1f}")
    
    print("\n" + "=" * 60)
    print("🎉 Pattern 130 Complete - 76.5% Milestone Reached!")
    print("Dialogue & Interaction Category: 100% COMPLETE!")
    print("130/170 patterns implemented!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_persona_management()
