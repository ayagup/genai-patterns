"""
Pattern 127: Dialogue Management Agent

This pattern implements multi-turn dialogue state tracking, context management,
intent recognition, and conversation flow control.

Use Cases:
- Conversational AI
- Customer service bots
- Virtual assistants
- Interactive tutorials
- Multi-turn task completion

Category: Dialogue & Interaction (1/4 = 25%)
Complexity: Advanced
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
import re


class IntentType(Enum):
    """Types of user intents."""
    GREETING = "greeting"
    QUESTION = "question"
    REQUEST = "request"
    CONFIRMATION = "confirmation"
    DENIAL = "denial"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"


class DialogueState(Enum):
    """States in dialogue flow."""
    INITIAL = "initial"
    GATHERING_INFO = "gathering_info"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_CLARIFICATION = "waiting_clarification"


class EntityType(Enum):
    """Types of entities in dialogue."""
    PERSON = "person"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    PRODUCT = "product"
    NUMBER = "number"
    ORGANIZATION = "organization"


@dataclass
class Entity:
    """Extracted entity from user input."""
    entity_type: EntityType
    value: str
    confidence: float
    span: Tuple[int, int]  # Start and end position in text


@dataclass
class Intent:
    """Recognized intent from user input."""
    intent_type: IntentType
    confidence: float
    entities: List[Entity] = field(default_factory=list)
    slots: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    """Single turn in dialogue."""
    turn_id: int
    timestamp: datetime
    speaker: str  # "user" or "agent"
    utterance: str
    intent: Optional[Intent] = None
    state: Optional[DialogueState] = None


@dataclass
class DialogueContext:
    """Context maintained throughout dialogue."""
    conversation_id: str
    current_state: DialogueState
    current_intent: Optional[IntentType]
    slots: Dict[str, Any]  # Information collected
    history: List[Turn]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_last_user_turn(self) -> Optional[Turn]:
        """Get last user turn."""
        for turn in reversed(self.history):
            if turn.speaker == "user":
                return turn
        return None
    
    def get_last_agent_turn(self) -> Optional[Turn]:
        """Get last agent turn."""
        for turn in reversed(self.history):
            if turn.speaker == "agent":
                return turn
        return None


@dataclass
class DialoguePolicy:
    """Policy defining dialogue behavior."""
    name: str
    required_slots: List[str]
    optional_slots: List[str] = field(default_factory=list)
    max_turns: int = 20
    confirmation_required: bool = True


class IntentRecognizer:
    """Recognizes user intents."""
    
    def __init__(self):
        # Simple keyword-based recognition (in production, use ML models)
        self.intent_patterns = {
            IntentType.GREETING: [
                r'\b(hi|hello|hey|greetings)\b',
                r'\bgood (morning|afternoon|evening)\b'
            ],
            IntentType.QUESTION: [
                r'\b(what|when|where|who|why|how)\b',
                r'\?$'
            ],
            IntentType.REQUEST: [
                r'\b(please|could you|can you|I want|I need)\b',
                r'\b(book|reserve|schedule|order|buy)\b'
            ],
            IntentType.CONFIRMATION: [
                r'\b(yes|yeah|correct|right|exactly|sure|ok|okay)\b',
                r'^y$'
            ],
            IntentType.DENIAL: [
                r'\b(no|nope|not|never|wrong)\b',
                r'^n$'
            ],
            IntentType.CLARIFICATION: [
                r'\b(what do you mean|I don\'t understand|clarify|explain)\b',
                r'\?.*\?'  # Multiple question marks suggest confusion
            ],
            IntentType.FEEDBACK: [
                r'\b(thanks|thank you|good|great|excellent|bad|poor)\b'
            ],
            IntentType.GOODBYE: [
                r'\b(bye|goodbye|see you|farewell|exit|quit)\b'
            ]
        }
    
    def recognize_intent(self, utterance: str) -> Intent:
        """Recognize intent from utterance."""
        utterance_lower = utterance.lower()
        
        # Check each intent pattern
        intent_scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            matches = 0
            for pattern in patterns:
                if re.search(pattern, utterance_lower, re.IGNORECASE):
                    matches += 1
            
            if matches > 0:
                intent_scores[intent_type] = matches / len(patterns)
        
        if not intent_scores:
            return Intent(
                intent_type=IntentType.UNKNOWN,
                confidence=0.0
            )
        
        # Get intent with highest score
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        return Intent(
            intent_type=best_intent[0],
            confidence=best_intent[1]
        )


class EntityExtractor:
    """Extracts entities from text."""
    
    def __init__(self):
        self.entity_patterns = {
            EntityType.DATE: r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|today|tomorrow|yesterday)\b',
            EntityType.TIME: r'\b(\d{1,2}:\d{2}|\d{1,2}\s*(am|pm))\b',
            EntityType.NUMBER: r'\b\d+\b',
            EntityType.LOCATION: r'\b(in|at|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text."""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = Entity(
                    entity_type=entity_type,
                    value=match.group(0),
                    confidence=0.8,
                    span=(match.start(), match.end())
                )
                entities.append(entity)
        
        return entities


class SlotFiller:
    """Fills slots with extracted information."""
    
    def __init__(self):
        self.slot_extractors = {
            'date': self._extract_date,
            'time': self._extract_time,
            'location': self._extract_location,
            'quantity': self._extract_number,
            'name': self._extract_name
        }
    
    def _extract_date(self, entities: List[Entity]) -> Optional[str]:
        """Extract date from entities."""
        for entity in entities:
            if entity.entity_type == EntityType.DATE:
                return entity.value
        return None
    
    def _extract_time(self, entities: List[Entity]) -> Optional[str]:
        """Extract time from entities."""
        for entity in entities:
            if entity.entity_type == EntityType.TIME:
                return entity.value
        return None
    
    def _extract_location(self, entities: List[Entity]) -> Optional[str]:
        """Extract location from entities."""
        for entity in entities:
            if entity.entity_type == EntityType.LOCATION:
                return entity.value
        return None
    
    def _extract_number(self, entities: List[Entity]) -> Optional[int]:
        """Extract number from entities."""
        for entity in entities:
            if entity.entity_type == EntityType.NUMBER:
                try:
                    return int(entity.value)
                except ValueError:
                    continue
        return None
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract name from text (simple capitalized word)."""
        match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        if match:
            return match.group(1)
        return None
    
    def fill_slots(
        self,
        utterance: str,
        entities: List[Entity],
        current_slots: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fill slots based on utterance and entities."""
        new_slots = current_slots.copy()
        
        # Try each slot extractor
        for slot_name, extractor in self.slot_extractors.items():
            if slot_name not in new_slots or new_slots[slot_name] is None:
                if slot_name == 'name':
                    value = extractor(utterance)
                else:
                    value = extractor(entities)
                
                if value is not None:
                    new_slots[slot_name] = value
        
        return new_slots


class StateManager:
    """Manages dialogue state transitions."""
    
    def __init__(self):
        self.transition_rules = {
            DialogueState.INITIAL: {
                IntentType.GREETING: DialogueState.GATHERING_INFO,
                IntentType.REQUEST: DialogueState.GATHERING_INFO,
                IntentType.QUESTION: DialogueState.GATHERING_INFO,
            },
            DialogueState.GATHERING_INFO: {
                IntentType.CONFIRMATION: DialogueState.CONFIRMING,
                IntentType.CLARIFICATION: DialogueState.WAITING_CLARIFICATION,
                IntentType.GOODBYE: DialogueState.COMPLETED,
            },
            DialogueState.WAITING_CLARIFICATION: {
                IntentType.CONFIRMATION: DialogueState.GATHERING_INFO,
                IntentType.DENIAL: DialogueState.GATHERING_INFO,
            },
            DialogueState.CONFIRMING: {
                IntentType.CONFIRMATION: DialogueState.EXECUTING,
                IntentType.DENIAL: DialogueState.GATHERING_INFO,
            },
            DialogueState.EXECUTING: {
                IntentType.FEEDBACK: DialogueState.COMPLETED,
            }
        }
    
    def transition(
        self,
        current_state: DialogueState,
        intent: IntentType,
        slots_filled: bool
    ) -> DialogueState:
        """Determine next state based on current state and intent."""
        # Check if we have enough information to confirm
        if current_state == DialogueState.GATHERING_INFO and slots_filled:
            return DialogueState.CONFIRMING
        
        # Use transition rules
        if current_state in self.transition_rules:
            rules = self.transition_rules[current_state]
            if intent in rules:
                return rules[intent]
        
        # Default: stay in current state
        return current_state


class ResponseGenerator:
    """Generates agent responses."""
    
    def __init__(self):
        self.templates = {
            DialogueState.INITIAL: [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?"
            ],
            DialogueState.GATHERING_INFO: {
                'missing_date': "When would you like to schedule this?",
                'missing_time': "What time works best for you?",
                'missing_location': "Where would you like this to be?",
                'missing_quantity': "How many would you like?",
                'missing_name': "May I have your name, please?",
            },
            DialogueState.CONFIRMING: "Let me confirm: {summary}. Is this correct?",
            DialogueState.EXECUTING: "Great! I'm processing your request...",
            DialogueState.COMPLETED: "All done! Is there anything else I can help you with?",
            DialogueState.WAITING_CLARIFICATION: "I'm not sure I understand. Could you please rephrase that?",
            DialogueState.FAILED: "I'm sorry, I couldn't complete your request. Would you like to try again?"
        }
    
    def generate_response(
        self,
        state: DialogueState,
        context: DialogueContext,
        policy: DialoguePolicy
    ) -> str:
        """Generate appropriate response."""
        if state == DialogueState.INITIAL:
            return self.templates[state][0]
        
        elif state == DialogueState.GATHERING_INFO:
            # Find missing required slot
            for slot in policy.required_slots:
                if slot not in context.slots or context.slots[slot] is None:
                    template_key = f'missing_{slot}'
                    if template_key in self.templates[state]:
                        return self.templates[state][template_key]
            
            return "Could you provide more details?"
        
        elif state == DialogueState.CONFIRMING:
            summary = self._create_summary(context.slots, policy)
            return self.templates[state].format(summary=summary)
        
        elif state in self.templates:
            template = self.templates[state]
            if isinstance(template, str):
                return template
            return template[0] if isinstance(template, list) else str(template)
        
        return "I'm processing your request."
    
    def _create_summary(self, slots: Dict[str, Any], policy: DialoguePolicy) -> str:
        """Create summary of filled slots."""
        parts = []
        for slot in policy.required_slots:
            if slot in slots and slots[slot] is not None:
                parts.append(f"{slot}={slots[slot]}")
        
        return ", ".join(parts)


class ConversationManager:
    """Manages conversation history and context."""
    
    def __init__(self):
        self.active_conversations: Dict[str, DialogueContext] = {}
        self.turn_counter = 0
    
    def create_conversation(self, conversation_id: str) -> DialogueContext:
        """Create new conversation."""
        context = DialogueContext(
            conversation_id=conversation_id,
            current_state=DialogueState.INITIAL,
            current_intent=None,
            slots={},
            history=[]
        )
        self.active_conversations[conversation_id] = context
        return context
    
    def add_turn(
        self,
        conversation_id: str,
        speaker: str,
        utterance: str,
        intent: Optional[Intent] = None,
        state: Optional[DialogueState] = None
    ) -> Turn:
        """Add turn to conversation."""
        if conversation_id not in self.active_conversations:
            self.create_conversation(conversation_id)
        
        context = self.active_conversations[conversation_id]
        
        self.turn_counter += 1
        turn = Turn(
            turn_id=self.turn_counter,
            timestamp=datetime.now(),
            speaker=speaker,
            utterance=utterance,
            intent=intent,
            state=state
        )
        
        context.history.append(turn)
        
        if state:
            context.current_state = state
        if intent:
            context.current_intent = intent.intent_type
        
        return turn
    
    def get_context(self, conversation_id: str) -> Optional[DialogueContext]:
        """Get conversation context."""
        return self.active_conversations.get(conversation_id)
    
    def get_conversation_summary(self, conversation_id: str) -> str:
        """Get summary of conversation."""
        context = self.get_context(conversation_id)
        if not context:
            return "No conversation found"
        
        lines = [
            f"Conversation: {conversation_id}",
            f"State: {context.current_state.value}",
            f"Turns: {len(context.history)}",
            f"Slots filled: {len([v for v in context.slots.values() if v is not None])}/{len(context.slots)}",
            "\nRecent turns:"
        ]
        
        for turn in context.history[-5:]:
            speaker_icon = "👤" if turn.speaker == "user" else "🤖"
            lines.append(f"  {speaker_icon} {turn.utterance}")
        
        return "\n".join(lines)


class DialogueManagementAgent:
    """Agent for managing multi-turn dialogues."""
    
    def __init__(self, policy: DialoguePolicy):
        self.policy = policy
        self.intent_recognizer = IntentRecognizer()
        self.entity_extractor = EntityExtractor()
        self.slot_filler = SlotFiller()
        self.state_manager = StateManager()
        self.response_generator = ResponseGenerator()
        self.conversation_manager = ConversationManager()
    
    def start_conversation(self, conversation_id: str) -> str:
        """Start a new conversation."""
        context = self.conversation_manager.create_conversation(conversation_id)
        
        # Initialize required slots
        for slot in self.policy.required_slots:
            context.slots[slot] = None
        
        # Generate greeting
        response = self.response_generator.generate_response(
            DialogueState.INITIAL,
            context,
            self.policy
        )
        
        self.conversation_manager.add_turn(
            conversation_id,
            "agent",
            response,
            state=DialogueState.INITIAL
        )
        
        return response
    
    def process_user_input(self, conversation_id: str, utterance: str) -> str:
        """Process user input and generate response."""
        context = self.conversation_manager.get_context(conversation_id)
        
        if not context:
            return self.start_conversation(conversation_id)
        
        # Recognize intent
        intent = self.intent_recognizer.recognize_intent(utterance)
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(utterance)
        intent.entities = entities
        
        # Fill slots
        context.slots = self.slot_filler.fill_slots(
            utterance,
            entities,
            context.slots
        )
        
        # Add user turn
        self.conversation_manager.add_turn(
            conversation_id,
            "user",
            utterance,
            intent=intent
        )
        
        # Check if all required slots are filled
        slots_filled = all(
            slot in context.slots and context.slots[slot] is not None
            for slot in self.policy.required_slots
        )
        
        # Determine next state
        next_state = self.state_manager.transition(
            context.current_state,
            intent.intent_type,
            slots_filled
        )
        
        # Generate response
        response = self.response_generator.generate_response(
            next_state,
            context,
            self.policy
        )
        
        # Add agent turn
        self.conversation_manager.add_turn(
            conversation_id,
            "agent",
            response,
            state=next_state
        )
        
        return response
    
    def get_dialogue_state(self, conversation_id: str) -> Dict[str, Any]:
        """Get current dialogue state."""
        context = self.conversation_manager.get_context(conversation_id)
        
        if not context:
            return {'status': 'not_found'}
        
        return {
            'conversation_id': conversation_id,
            'state': context.current_state.value,
            'intent': context.current_intent.value if context.current_intent else None,
            'slots': context.slots,
            'turn_count': len(context.history),
            'completed': context.current_state in [DialogueState.COMPLETED, DialogueState.FAILED]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dialogue statistics."""
        total_conversations = len(self.conversation_manager.active_conversations)
        total_turns = sum(
            len(ctx.history)
            for ctx in self.conversation_manager.active_conversations.values()
        )
        
        completed = sum(
            1 for ctx in self.conversation_manager.active_conversations.values()
            if ctx.current_state == DialogueState.COMPLETED
        )
        
        return {
            'total_conversations': total_conversations,
            'total_turns': total_turns,
            'completed_conversations': completed,
            'average_turns_per_conversation': total_turns / total_conversations if total_conversations > 0 else 0
        }


def demonstrate_dialogue_management():
    """Demonstrate the Dialogue Management Agent."""
    print("=" * 60)
    print("Dialogue Management Agent Demonstration")
    print("=" * 60)
    
    # Create policy for appointment booking
    policy = DialoguePolicy(
        name="appointment_booking",
        required_slots=['date', 'time', 'name'],
        optional_slots=['location'],
        confirmation_required=True
    )
    
    # Create agent
    agent = DialogueManagementAgent(policy)
    
    print("\n1. STARTING CONVERSATION")
    print("-" * 60)
    
    conversation_id = "conv_001"
    response = agent.start_conversation(conversation_id)
    print(f"Agent: {response}")
    
    print("\n\n2. MULTI-TURN DIALOGUE")
    print("-" * 60)
    
    # Simulate conversation
    user_inputs = [
        "Hi, I'd like to book an appointment",
        "Tomorrow at 2pm",
        "My name is John Smith",
        "Yes, that's correct",
    ]
    
    for user_input in user_inputs:
        print(f"\nUser: {user_input}")
        response = agent.process_user_input(conversation_id, user_input)
        print(f"Agent: {response}")
        
        # Show dialogue state
        state = agent.get_dialogue_state(conversation_id)
        print(f"  [State: {state['state']}, Slots filled: {sum(1 for v in state['slots'].values() if v is not None)}/{len(policy.required_slots)}]")
    
    print("\n\n3. CONVERSATION SUMMARY")
    print("-" * 60)
    summary = agent.conversation_manager.get_conversation_summary(conversation_id)
    print(summary)
    
    print("\n\n4. ANOTHER CONVERSATION WITH CLARIFICATION")
    print("-" * 60)
    
    conversation_id_2 = "conv_002"
    response = agent.start_conversation(conversation_id_2)
    print(f"Agent: {response}")
    
    clarification_inputs = [
        "I need help with booking",
        "What do you mean?",  # Clarification
        "I want to schedule for next Monday at 10am",
        "Alice Johnson",
        "Yes",
    ]
    
    for user_input in clarification_inputs:
        print(f"\nUser: {user_input}")
        response = agent.process_user_input(conversation_id_2, user_input)
        print(f"Agent: {response}")
    
    print("\n\n5. DIALOGUE STATE TRACKING")
    print("-" * 60)
    
    for conv_id in [conversation_id, conversation_id_2]:
        state = agent.get_dialogue_state(conv_id)
        print(f"\n{conv_id}:")
        print(f"  State: {state['state']}")
        print(f"  Turns: {state['turn_count']}")
        print(f"  Completed: {state['completed']}")
        print(f"  Slots: {state['slots']}")
    
    print("\n\n6. STATISTICS")
    print("-" * 60)
    
    stats = agent.get_statistics()
    print(f"  Total Conversations: {stats['total_conversations']}")
    print(f"  Total Turns: {stats['total_turns']}")
    print(f"  Completed: {stats['completed_conversations']}")
    print(f"  Avg Turns/Conversation: {stats['average_turns_per_conversation']:.1f}")
    
    print("\n" + "=" * 60)
    print("🎉 Pattern 127 Complete!")
    print("🆕 NEW CATEGORY: Dialogue & Interaction (25%)")
    print("127/170 patterns implemented (74.7%)!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_dialogue_management()
