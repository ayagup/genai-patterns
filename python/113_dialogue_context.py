"""
Pattern 113: Dialogue Context Management

This pattern demonstrates comprehensive dialogue context tracking including
conversation state, turn-taking, discourse coherence, and topic management.

Key concepts:
- Dialogue state tracking
- Turn-taking management
- Discourse coherence maintenance
- Topic tracking and shifting
- Conversational flow control

Use cases:
- Conversational AI systems
- Multi-turn dialogue agents
- Customer service bots
- Interactive tutoring systems
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
import time
import uuid


class SpeakerRole(Enum):
    """Roles in dialogue"""
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    MODERATOR = "moderator"


class UtteranceType(Enum):
    """Types of utterances"""
    QUESTION = "question"
    ANSWER = "answer"
    STATEMENT = "statement"
    REQUEST = "request"
    ACKNOWLEDGMENT = "acknowledgment"
    CLARIFICATION = "clarification"
    GREETING = "greeting"
    FAREWELL = "farewell"


class DialogueAct(Enum):
    """Dialogue acts (speech acts)"""
    INFORM = "inform"
    REQUEST_INFO = "request_info"
    CONFIRM = "confirm"
    DENY = "deny"
    OFFER = "offer"
    ACCEPT = "accept"
    REJECT = "reject"
    GREET = "greet"
    THANK = "thank"
    APOLOGIZE = "apologize"


class TopicTransitionType(Enum):
    """Types of topic transitions"""
    CONTINUATION = "continuation"
    SHIFT = "shift"
    RETURN = "return"
    DIGRESSION = "digression"


@dataclass
class Utterance:
    """A single utterance in dialogue"""
    speaker: SpeakerRole
    text: str
    utterance_type: UtteranceType
    dialogue_acts: List[DialogueAct]
    timestamp: float
    turn_number: int
    references: List[str] = field(default_factory=list)  # References to previous turns
    topics: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class Turn:
    """A dialogue turn (may contain multiple utterances)"""
    speaker: SpeakerRole
    utterances: List[Utterance]
    turn_number: int
    timestamp: float
    duration: float = 0.0
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class Topic:
    """A topic in the dialogue"""
    name: str
    keywords: Set[str]
    introduced_at: int  # Turn number
    last_mentioned: int
    mention_count: int = 1
    subtopics: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class CoherenceLink:
    """Link establishing coherence between utterances"""
    from_utterance: str  # Utterance ID
    to_utterance: str    # Utterance ID
    link_type: str       # Type of coherence relation
    strength: float      # Strength of the link
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


class ConversationState:
    """Tracks the current state of the conversation"""
    
    def __init__(self):
        self.turn_count: int = 0
        self.current_speaker: Optional[SpeakerRole] = None
        self.last_speaker: Optional[SpeakerRole] = None
        self.active_topics: Set[str] = set()
        self.pending_questions: List[str] = []
        self.pending_requests: List[str] = []
        self.discussed_topics: Set[str] = set()
        self.grounding_status: Dict[str, bool] = {}  # Topic -> grounded?
    
    def update(self, turn: Turn):
        """Update state based on new turn"""
        self.turn_count += 1
        self.last_speaker = self.current_speaker
        self.current_speaker = turn.speaker
        
        # Update topics
        for utterance in turn.utterances:
            self.active_topics.update(utterance.topics)
            self.discussed_topics.update(utterance.topics)
            
            # Track questions and requests
            if utterance.utterance_type == UtteranceType.QUESTION:
                self.pending_questions.append(utterance.id)
            elif utterance.utterance_type == UtteranceType.REQUEST:
                self.pending_requests.append(utterance.id)
            elif utterance.utterance_type == UtteranceType.ANSWER:
                # Clear pending questions (simplified)
                if self.pending_questions:
                    self.pending_questions.pop(0)


class TurnTakingManager:
    """Manages turn-taking in dialogue"""
    
    def __init__(self):
        self.turns: List[Turn] = []
        self.expected_next_speaker: Optional[SpeakerRole] = None
        self.turn_duration_history: List[float] = []
        self.interruptions: int = 0
    
    def add_turn(self, speaker: SpeakerRole, utterances: List[Utterance],
                 duration: float = 0.0) -> Turn:
        """Add a new turn"""
        turn = Turn(
            speaker=speaker,
            utterances=utterances,
            turn_number=len(self.turns) + 1,
            timestamp=time.time(),
            duration=duration
        )
        
        self.turns.append(turn)
        self.turn_duration_history.append(duration)
        
        # Check for interruption
        if self.expected_next_speaker and speaker != self.expected_next_speaker:
            self.interruptions += 1
        
        # Predict next speaker (simple alternation)
        self.expected_next_speaker = self._predict_next_speaker(speaker)
        
        return turn
    
    def _predict_next_speaker(self, current_speaker: SpeakerRole) -> SpeakerRole:
        """Predict next speaker"""
        # Simple rule: alternate between user and agent
        if current_speaker == SpeakerRole.USER:
            return SpeakerRole.AGENT
        elif current_speaker == SpeakerRole.AGENT:
            return SpeakerRole.USER
        else:
            return current_speaker
    
    def get_turn_statistics(self) -> Dict[str, Any]:
        """Get turn-taking statistics"""
        if not self.turns:
            return {}
        
        speaker_turns = {}
        for turn in self.turns:
            speaker = turn.speaker.value
            speaker_turns[speaker] = speaker_turns.get(speaker, 0) + 1
        
        return {
            "total_turns": len(self.turns),
            "by_speaker": speaker_turns,
            "avg_turn_duration": sum(self.turn_duration_history) / len(self.turn_duration_history)
                                if self.turn_duration_history else 0.0,
            "interruptions": self.interruptions,
            "expected_next": self.expected_next_speaker.value if self.expected_next_speaker else None
        }


class TopicTracker:
    """Tracks topics throughout dialogue"""
    
    def __init__(self):
        self.topics: Dict[str, Topic] = {}
        self.topic_history: List[Tuple[int, str]] = []  # (turn_number, topic)
        self.topic_transitions: List[Dict[str, Any]] = []
    
    def add_topic(self, topic_name: str, keywords: Set[str], turn_number: int) -> Topic:
        """Add or update a topic"""
        if topic_name in self.topics:
            topic = self.topics[topic_name]
            topic.last_mentioned = turn_number
            topic.mention_count += 1
            topic.keywords.update(keywords)
        else:
            topic = Topic(
                name=topic_name,
                keywords=keywords,
                introduced_at=turn_number,
                last_mentioned=turn_number
            )
            self.topics[topic_name] = topic
        
        self.topic_history.append((turn_number, topic_name))
        
        # Detect topic transition
        if len(self.topic_history) > 1:
            prev_topic = self.topic_history[-2][1]
            if prev_topic != topic_name:
                transition_type = self._classify_transition(prev_topic, topic_name)
                self.topic_transitions.append({
                    "from": prev_topic,
                    "to": topic_name,
                    "turn": turn_number,
                    "type": transition_type.value
                })
        
        return topic
    
    def _classify_transition(self, from_topic: str, to_topic: str) -> TopicTransitionType:
        """Classify type of topic transition"""
        # Check if returning to previous topic
        recent_topics = [t for _, t in self.topic_history[-5:]]
        if to_topic in recent_topics[:-1]:
            return TopicTransitionType.RETURN
        
        # Check topic relatedness (simplified)
        from_keywords = self.topics[from_topic].keywords
        to_keywords = self.topics[to_topic].keywords
        
        overlap = len(from_keywords & to_keywords)
        if overlap > 0:
            return TopicTransitionType.CONTINUATION
        else:
            return TopicTransitionType.SHIFT
    
    def get_active_topics(self, recency_window: int = 3) -> List[str]:
        """Get currently active topics"""
        if not self.topic_history:
            return []
        
        recent_mentions = self.topic_history[-recency_window:]
        active = list(dict.fromkeys([topic for _, topic in recent_mentions]))
        return active
    
    def get_topic_statistics(self) -> Dict[str, Any]:
        """Get topic statistics"""
        return {
            "total_topics": len(self.topics),
            "active_topics": len(self.get_active_topics()),
            "topic_shifts": len([t for t in self.topic_transitions
                                if t["type"] in ["shift", "digression"]]),
            "most_discussed": max(self.topics.values(),
                                 key=lambda t: t.mention_count).name
                             if self.topics else None
        }


class CoherenceAnalyzer:
    """Analyzes and maintains discourse coherence"""
    
    def __init__(self):
        self.coherence_links: List[CoherenceLink] = []
        self.utterance_index: Dict[str, Utterance] = {}
    
    def add_utterance(self, utterance: Utterance):
        """Add utterance and establish coherence links"""
        self.utterance_index[utterance.id] = utterance
        
        # Establish links with recent utterances
        recent_utterances = list(self.utterance_index.values())[-5:]
        
        for prev_utterance in recent_utterances[:-1]:
            link = self._establish_link(prev_utterance, utterance)
            if link:
                self.coherence_links.append(link)
    
    def _establish_link(self, from_utt: Utterance,
                       to_utt: Utterance) -> Optional[CoherenceLink]:
        """Establish coherence link between utterances"""
        
        # Check for explicit references
        if from_utt.id in to_utt.references:
            return CoherenceLink(
                from_utterance=from_utt.id,
                to_utterance=to_utt.id,
                link_type="explicit_reference",
                strength=1.0
            )
        
        # Check for question-answer pairs
        if from_utt.utterance_type == UtteranceType.QUESTION and \
           to_utt.utterance_type == UtteranceType.ANSWER:
            return CoherenceLink(
                from_utterance=from_utt.id,
                to_utterance=to_utt.id,
                link_type="question_answer",
                strength=0.9
            )
        
        # Check for topic continuity
        topic_overlap = len(from_utt.topics & to_utt.topics)
        if topic_overlap > 0:
            return CoherenceLink(
                from_utterance=from_utt.id,
                to_utterance=to_utt.id,
                link_type="topic_continuity",
                strength=0.7
            )
        
        # Check for temporal adjacency (weak link)
        if to_utt.timestamp - from_utt.timestamp < 5.0:
            return CoherenceLink(
                from_utterance=from_utt.id,
                to_utterance=to_utt.id,
                link_type="temporal_adjacency",
                strength=0.3
            )
        
        return None
    
    def compute_coherence_score(self, window: int = 10) -> float:
        """Compute overall coherence score"""
        if not self.utterance_index:
            return 1.0
        
        recent_utterances = list(self.utterance_index.values())[-window:]
        recent_ids = {u.id for u in recent_utterances}
        
        # Count links within window
        relevant_links = [l for l in self.coherence_links
                         if l.from_utterance in recent_ids and l.to_utterance in recent_ids]
        
        if len(recent_utterances) <= 1:
            return 1.0
        
        # Coherence score: ratio of strong links to possible links
        possible_links = len(recent_utterances) - 1
        strong_links = [l for l in relevant_links if l.strength > 0.6]
        
        score = len(strong_links) / possible_links if possible_links > 0 else 1.0
        return min(score, 1.0)
    
    def get_coherence_report(self) -> Dict[str, Any]:
        """Get coherence report"""
        link_types = {}
        for link in self.coherence_links:
            link_types[link.link_type] = link_types.get(link.link_type, 0) + 1
        
        return {
            "total_links": len(self.coherence_links),
            "coherence_score": self.compute_coherence_score(),
            "link_types": link_types,
            "avg_link_strength": sum(l.strength for l in self.coherence_links) / len(self.coherence_links)
                                if self.coherence_links else 0.0
        }


class DialogueContextManager:
    """
    Complete dialogue context management system that tracks conversation state,
    manages turn-taking, monitors topics, and maintains coherence.
    """
    
    def __init__(self):
        self.state = ConversationState()
        self.turn_manager = TurnTakingManager()
        self.topic_tracker = TopicTracker()
        self.coherence_analyzer = CoherenceAnalyzer()
        self.dialogue_history: List[Utterance] = []
    
    def add_utterance(self, speaker: SpeakerRole, text: str,
                     utterance_type: UtteranceType,
                     dialogue_acts: Optional[List[DialogueAct]] = None,
                     topics: Optional[Set[str]] = None,
                     references: Optional[List[str]] = None,
                     turn_duration: float = 0.0) -> Dict[str, Any]:
        """Add utterance and update all context components"""
        
        # Create utterance
        utterance = Utterance(
            speaker=speaker,
            text=text,
            utterance_type=utterance_type,
            dialogue_acts=dialogue_acts or [],
            timestamp=time.time(),
            turn_number=self.state.turn_count + 1,
            references=references or [],
            topics=topics or set()
        )
        
        self.dialogue_history.append(utterance)
        
        # Add to coherence analyzer
        self.coherence_analyzer.add_utterance(utterance)
        
        # Create turn
        turn = self.turn_manager.add_turn(speaker, [utterance], turn_duration)
        
        # Update state
        self.state.update(turn)
        
        # Track topics
        for topic in utterance.topics:
            keywords = set(text.lower().split())  # Simplified keyword extraction
            self.topic_tracker.add_topic(topic, keywords, turn.turn_number)
        
        # Compile response
        response = {
            "utterance_id": utterance.id,
            "turn_number": turn.turn_number,
            "active_topics": list(self.state.active_topics),
            "coherence_score": self.coherence_analyzer.compute_coherence_score(),
            "pending_questions": len(self.state.pending_questions),
            "expected_next_speaker": self.turn_manager.expected_next_speaker.value
                                    if self.turn_manager.expected_next_speaker else None
        }
        
        return response
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get comprehensive context summary"""
        return {
            "conversation_state": {
                "turns": self.state.turn_count,
                "current_speaker": self.state.current_speaker.value if self.state.current_speaker else None,
                "active_topics": list(self.state.active_topics),
                "pending_questions": len(self.state.pending_questions),
                "pending_requests": len(self.state.pending_requests)
            },
            "turn_taking": self.turn_manager.get_turn_statistics(),
            "topics": self.topic_tracker.get_topic_statistics(),
            "coherence": self.coherence_analyzer.get_coherence_report()
        }
    
    def get_dialogue_flow(self) -> List[Dict[str, Any]]:
        """Get dialogue flow summary"""
        flow = []
        
        for utterance in self.dialogue_history[-10:]:  # Last 10 utterances
            flow.append({
                "turn": utterance.turn_number,
                "speaker": utterance.speaker.value,
                "type": utterance.utterance_type.value,
                "text": utterance.text[:50] + "..." if len(utterance.text) > 50 else utterance.text,
                "topics": list(utterance.topics)
            })
        
        return flow
    
    def check_coherence(self, window: int = 5) -> Dict[str, Any]:
        """Check recent coherence"""
        score = self.coherence_analyzer.compute_coherence_score(window)
        
        status = "high" if score > 0.7 else "medium" if score > 0.4 else "low"
        
        return {
            "score": score,
            "status": status,
            "window_size": window,
            "recommendation": self._get_coherence_recommendation(score)
        }
    
    def _get_coherence_recommendation(self, score: float) -> str:
        """Get recommendation based on coherence score"""
        if score > 0.7:
            return "Dialogue is coherent. Continue naturally."
        elif score > 0.4:
            return "Moderate coherence. Consider explicit topic markers."
        else:
            return "Low coherence. Recommend explicit transitions or clarifications."


# Demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("PATTERN 113: DIALOGUE CONTEXT MANAGEMENT")
    print("Demonstration of conversation state, turn-taking, and coherence tracking")
    print("=" * 80)
    
    # Create manager
    manager = DialogueContextManager()
    
    print("\n1. Simulating Multi-Turn Dialogue")
    print("-" * 40)
    
    # Turn 1: User greeting
    result = manager.add_utterance(
        SpeakerRole.USER,
        "Hello! I need help with my Python project.",
        UtteranceType.GREETING,
        dialogue_acts=[DialogueAct.GREET, DialogueAct.REQUEST_INFO],
        topics={"greeting", "python", "help"},
        turn_duration=2.0
    )
    print(f"Turn {result['turn_number']} [USER]: Greeting + request")
    print(f"  Active topics: {result['active_topics']}")
    print(f"  Coherence: {result['coherence_score']:.2f}")
    
    # Turn 2: Agent response
    result = manager.add_utterance(
        SpeakerRole.AGENT,
        "Hello! I'd be happy to help with your Python project. What specifically do you need?",
        UtteranceType.QUESTION,
        dialogue_acts=[DialogueAct.GREET, DialogueAct.REQUEST_INFO],
        topics={"python", "help"},
        turn_duration=1.5
    )
    print(f"Turn {result['turn_number']} [AGENT]: Greeting + clarification question")
    print(f"  Expected next: {result['expected_next_speaker']}")
    
    # Turn 3: User provides details
    result = manager.add_utterance(
        SpeakerRole.USER,
        "I'm working on a machine learning model and getting errors with data preprocessing.",
        UtteranceType.STATEMENT,
        dialogue_acts=[DialogueAct.INFORM],
        topics={"python", "machine_learning", "errors", "preprocessing"},
        turn_duration=3.0
    )
    print(f"Turn {result['turn_number']} [USER]: Provides details")
    print(f"  Active topics: {result['active_topics']}")
    print(f"  Coherence: {result['coherence_score']:.2f}")
    
    # Turn 4: Agent asks for clarification
    result = manager.add_utterance(
        SpeakerRole.AGENT,
        "What kind of preprocessing errors are you seeing?",
        UtteranceType.QUESTION,
        dialogue_acts=[DialogueAct.REQUEST_INFO],
        topics={"preprocessing", "errors"},
        turn_duration=1.0
    )
    print(f"Turn {result['turn_number']} [AGENT]: Clarification question")
    print(f"  Pending questions: {result['pending_questions']}")
    
    # Turn 5: User answers
    result = manager.add_utterance(
        SpeakerRole.USER,
        "I'm getting ValueError when trying to normalize the features.",
        UtteranceType.ANSWER,
        dialogue_acts=[DialogueAct.INFORM],
        topics={"errors", "preprocessing", "normalization"},
        turn_duration=2.5
    )
    print(f"Turn {result['turn_number']} [USER]: Answers question")
    print(f"  Coherence: {result['coherence_score']:.2f}")
    
    # Turn 6: Agent provides solution
    result = manager.add_utterance(
        SpeakerRole.AGENT,
        "The ValueError usually means there are NaN values. Try using df.fillna() first.",
        UtteranceType.ANSWER,
        dialogue_acts=[DialogueAct.INFORM, DialogueAct.OFFER],
        topics={"errors", "solution", "preprocessing"},
        turn_duration=2.0
    )
    print(f"Turn {result['turn_number']} [AGENT]: Provides solution")
    
    # Turn 7: User thanks and shifts topic
    result = manager.add_utterance(
        SpeakerRole.USER,
        "Thanks! Also, do you know about deployment options for ML models?",
        UtteranceType.QUESTION,
        dialogue_acts=[DialogueAct.THANK, DialogueAct.REQUEST_INFO],
        topics={"deployment", "machine_learning"},
        turn_duration=2.5
    )
    print(f"Turn {result['turn_number']} [USER]: Thanks + topic shift")
    print(f"  Active topics: {result['active_topics']}")
    
    # Get comprehensive summary
    print("\n2. Conversation State Summary")
    print("-" * 40)
    summary = manager.get_context_summary()
    
    conv_state = summary["conversation_state"]
    print(f"Total turns: {conv_state['turns']}")
    print(f"Current speaker: {conv_state['current_speaker']}")
    print(f"Active topics: {', '.join(conv_state['active_topics'])}")
    print(f"Pending questions: {conv_state['pending_questions']}")
    
    # Turn-taking statistics
    print("\n3. Turn-Taking Statistics")
    print("-" * 40)
    turn_stats = summary["turn_taking"]
    print(f"Total turns: {turn_stats['total_turns']}")
    print(f"By speaker:")
    for speaker, count in turn_stats["by_speaker"].items():
        print(f"  {speaker}: {count} turns")
    print(f"Average turn duration: {turn_stats['avg_turn_duration']:.1f}s")
    print(f"Interruptions: {turn_stats['interruptions']}")
    print(f"Expected next: {turn_stats['expected_next']}")
    
    # Topic statistics
    print("\n4. Topic Statistics")
    print("-" * 40)
    topic_stats = summary["topics"]
    print(f"Total topics discussed: {topic_stats['total_topics']}")
    print(f"Currently active: {topic_stats['active_topics']}")
    print(f"Topic shifts: {topic_stats['topic_shifts']}")
    print(f"Most discussed: {topic_stats['most_discussed']}")
    
    # Coherence analysis
    print("\n5. Coherence Analysis")
    print("-" * 40)
    coherence = summary["coherence"]
    print(f"Overall coherence score: {coherence['coherence_score']:.2f}")
    print(f"Total coherence links: {coherence['total_links']}")
    print(f"Average link strength: {coherence['avg_link_strength']:.2f}")
    
    if coherence["link_types"]:
        print("\nLink types:")
        for link_type, count in coherence["link_types"].items():
            print(f"  {link_type}: {count}")
    
    # Check current coherence
    print("\n6. Current Coherence Check")
    print("-" * 40)
    coherence_check = manager.check_coherence(window=5)
    print(f"Recent coherence score: {coherence_check['score']:.2f}")
    print(f"Status: {coherence_check['status']}")
    print(f"Recommendation: {coherence_check['recommendation']}")
    
    # Dialogue flow
    print("\n7. Recent Dialogue Flow")
    print("-" * 40)
    flow = manager.get_dialogue_flow()
    for entry in flow:
        print(f"Turn {entry['turn']} [{entry['speaker'].upper()}] ({entry['type']}):")
        print(f"  \"{entry['text']}\"")
        if entry['topics']:
            print(f"  Topics: {', '.join(entry['topics'])}")
    
    print("\n" + "=" * 80)
    print("✓ Dialogue context management demonstration complete!")
    print("  Context & Grounding category now 100% COMPLETE! 🎉")
    print("=" * 80)
