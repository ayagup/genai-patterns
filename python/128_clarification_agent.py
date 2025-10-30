"""
Pattern 128: Clarification Agent

This pattern implements ambiguity detection, clarifying questions generation,
and user intent refinement through interactive clarification.

Use Cases:
- Ambiguous query handling
- User intent refinement
- Information gap filling
- Misunderstanding resolution
- Interactive question answering

Category: Dialogue & Interaction (2/4 = 50%)
Complexity: Advanced
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime


class AmbiguityType(Enum):
    """Types of ambiguity."""
    LEXICAL = "lexical"  # Word has multiple meanings
    SYNTACTIC = "syntactic"  # Sentence structure ambiguous
    SEMANTIC = "semantic"  # Meaning unclear
    REFERENTIAL = "referential"  # Unclear what is being referenced
    SCOPE = "scope"  # Unclear scope of statement
    INTENT = "intent"  # Unclear user intention


class ClarificationStrategy(Enum):
    """Strategies for clarification."""
    ASK_DIRECTLY = "ask_directly"
    OFFER_OPTIONS = "offer_options"
    REQUEST_EXAMPLE = "request_example"
    REPHRASE_CONFIRM = "rephrase_confirm"
    PARTIAL_UNDERSTANDING = "partial_understanding"


class ConfidenceLevel(Enum):
    """Confidence in understanding."""
    VERY_LOW = "very_low"  # < 0.3
    LOW = "low"  # 0.3 - 0.5
    MEDIUM = "medium"  # 0.5 - 0.7
    HIGH = "high"  # 0.7 - 0.9
    VERY_HIGH = "very_high"  # > 0.9


@dataclass
class AmbiguityDetection:
    """Detected ambiguity in user input."""
    ambiguity_type: AmbiguityType
    ambiguous_element: str
    possible_interpretations: List[str]
    confidence: float
    context: str


@dataclass
class ClarificationQuestion:
    """Question to clarify ambiguity."""
    question_id: str
    question_text: str
    strategy: ClarificationStrategy
    ambiguity: AmbiguityDetection
    options: Optional[List[str]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ClarificationResponse:
    """Response to clarification question."""
    question_id: str
    user_response: str
    selected_option: Optional[str]
    resolved: bool
    refined_intent: Optional[str] = None


@dataclass
class UnderstandingState:
    """Current understanding of user intent."""
    original_query: str
    current_interpretation: str
    confidence: float
    ambiguities: List[AmbiguityDetection]
    clarifications_asked: int = 0
    resolved_ambiguities: int = 0


class AmbiguityDetector:
    """Detects ambiguities in user input."""
    
    def __init__(self):
        # Common ambiguous words and their meanings
        self.ambiguous_words = {
            'bank': ['financial institution', 'river bank', 'tilt'],
            'left': ['direction', 'remaining', 'departed'],
            'right': ['direction', 'correct', 'entitlement'],
            'light': ['illumination', 'not heavy', 'ignite'],
            'can': ['able to', 'container', 'preserve'],
            'book': ['publication', 'reserve'],
            'date': ['calendar date', 'romantic meeting', 'fruit'],
            'spring': ['season', 'water source', 'coil', 'jump'],
        }
        
        # Pronouns that may cause referential ambiguity
        self.ambiguous_pronouns = {'it', 'this', 'that', 'they', 'them', 'these', 'those'}
        
        # Words indicating potential scope ambiguity
        self.scope_indicators = {'all', 'some', 'every', 'each', 'any', 'most'}
    
    def detect_ambiguities(self, text: str) -> List[AmbiguityDetection]:
        """Detect ambiguities in text."""
        ambiguities = []
        words = text.lower().split()
        
        # Lexical ambiguity
        for word in words:
            if word in self.ambiguous_words:
                ambiguity = AmbiguityDetection(
                    ambiguity_type=AmbiguityType.LEXICAL,
                    ambiguous_element=word,
                    possible_interpretations=self.ambiguous_words[word],
                    confidence=0.7,
                    context=text
                )
                ambiguities.append(ambiguity)
        
        # Referential ambiguity
        for word in words:
            if word in self.ambiguous_pronouns:
                # Check if clear antecedent
                ambiguity = AmbiguityDetection(
                    ambiguity_type=AmbiguityType.REFERENTIAL,
                    ambiguous_element=word,
                    possible_interpretations=['unclear reference'],
                    confidence=0.6,
                    context=text
                )
                ambiguities.append(ambiguity)
        
        # Scope ambiguity
        for word in words:
            if word in self.scope_indicators:
                ambiguity = AmbiguityDetection(
                    ambiguity_type=AmbiguityType.SCOPE,
                    ambiguous_element=word,
                    possible_interpretations=['scope unclear'],
                    confidence=0.5,
                    context=text
                )
                ambiguities.append(ambiguity)
        
        # Intent ambiguity - detect questions vs statements
        if '?' not in text and any(word in words for word in ['can', 'could', 'would', 'should']):
            ambiguity = AmbiguityDetection(
                ambiguity_type=AmbiguityType.INTENT,
                ambiguous_element='sentence type',
                possible_interpretations=['question', 'statement', 'request'],
                confidence=0.6,
                context=text
            )
            ambiguities.append(ambiguity)
        
        return ambiguities
    
    def calculate_understanding_confidence(
        self,
        text: str,
        ambiguities: List[AmbiguityDetection]
    ) -> float:
        """Calculate confidence in understanding."""
        if not ambiguities:
            return 0.95  # High confidence if no ambiguities
        
        # Reduce confidence based on ambiguities
        base_confidence = 0.8
        
        for ambiguity in ambiguities:
            penalty = 0.15 if ambiguity.ambiguity_type in [AmbiguityType.INTENT, AmbiguityType.SEMANTIC] else 0.1
            base_confidence -= penalty
        
        return max(0.1, base_confidence)


class ClarificationQuestionGenerator:
    """Generates clarification questions."""
    
    def __init__(self):
        self.question_counter = 0
    
    def generate_question(
        self,
        ambiguity: AmbiguityDetection,
        strategy: Optional[ClarificationStrategy] = None
    ) -> ClarificationQuestion:
        """Generate clarification question for ambiguity."""
        self.question_counter += 1
        question_id = f"cq_{self.question_counter}"
        
        # Choose strategy based on ambiguity type
        if strategy is None:
            strategy = self._choose_strategy(ambiguity)
        
        question_text = self._generate_question_text(ambiguity, strategy)
        options = self._generate_options(ambiguity, strategy)
        
        return ClarificationQuestion(
            question_id=question_id,
            question_text=question_text,
            strategy=strategy,
            ambiguity=ambiguity,
            options=options
        )
    
    def _choose_strategy(self, ambiguity: AmbiguityDetection) -> ClarificationStrategy:
        """Choose best clarification strategy."""
        if ambiguity.ambiguity_type == AmbiguityType.LEXICAL:
            return ClarificationStrategy.OFFER_OPTIONS
        elif ambiguity.ambiguity_type == AmbiguityType.REFERENTIAL:
            return ClarificationStrategy.ASK_DIRECTLY
        elif ambiguity.ambiguity_type == AmbiguityType.INTENT:
            return ClarificationStrategy.REPHRASE_CONFIRM
        else:
            return ClarificationStrategy.REQUEST_EXAMPLE
    
    def _generate_question_text(
        self,
        ambiguity: AmbiguityDetection,
        strategy: ClarificationStrategy
    ) -> str:
        """Generate question text based on strategy."""
        element = ambiguity.ambiguous_element
        
        if strategy == ClarificationStrategy.OFFER_OPTIONS:
            return f"When you mention '{element}', which meaning do you intend?"
        
        elif strategy == ClarificationStrategy.ASK_DIRECTLY:
            if ambiguity.ambiguity_type == AmbiguityType.REFERENTIAL:
                return f"What does '{element}' refer to?"
            return f"Could you clarify what you mean by '{element}'?"
        
        elif strategy == ClarificationStrategy.REQUEST_EXAMPLE:
            return f"Could you provide an example of what you mean?"
        
        elif strategy == ClarificationStrategy.REPHRASE_CONFIRM:
            return f"Just to confirm, are you asking about...?"
        
        elif strategy == ClarificationStrategy.PARTIAL_UNDERSTANDING:
            return f"I understand part of your request, but could you clarify '{element}'?"
        
        return "Could you please clarify your request?"
    
    def _generate_options(
        self,
        ambiguity: AmbiguityDetection,
        strategy: ClarificationStrategy
    ) -> Optional[List[str]]:
        """Generate options for clarification."""
        if strategy == ClarificationStrategy.OFFER_OPTIONS:
            return ambiguity.possible_interpretations
        return None


class IntentRefiner:
    """Refines user intent through clarification."""
    
    def __init__(self):
        self.refinement_history: List[Dict[str, Any]] = []
    
    def refine_intent(
        self,
        original_query: str,
        clarification_response: ClarificationResponse,
        ambiguity: AmbiguityDetection
    ) -> str:
        """Refine intent based on clarification."""
        refined = original_query
        
        if clarification_response.selected_option:
            # Replace ambiguous element with selected interpretation
            refined = refined.replace(
                ambiguity.ambiguous_element,
                f"{ambiguity.ambiguous_element} ({clarification_response.selected_option})"
            )
        
        # Record refinement
        self.refinement_history.append({
            'original': original_query,
            'ambiguity': ambiguity.ambiguous_element,
            'clarification': clarification_response.user_response,
            'refined': refined,
            'timestamp': datetime.now()
        })
        
        return refined
    
    def get_refinement_path(self, query: str) -> List[str]:
        """Get refinement path for a query."""
        path = [query]
        
        for refinement in self.refinement_history:
            if refinement['original'] == path[-1]:
                path.append(refinement['refined'])
        
        return path


class ConfidenceTracker:
    """Tracks confidence in understanding over clarifications."""
    
    def __init__(self):
        self.confidence_history: List[Tuple[float, str]] = []
    
    def update_confidence(
        self,
        current_confidence: float,
        clarification_provided: bool,
        ambiguity_resolved: bool
    ) -> float:
        """Update confidence after clarification."""
        new_confidence = current_confidence
        
        if clarification_provided:
            if ambiguity_resolved:
                # Significant boost if ambiguity resolved
                new_confidence = min(1.0, current_confidence + 0.2)
            else:
                # Small boost for any clarification
                new_confidence = min(1.0, current_confidence + 0.05)
        
        self.confidence_history.append((
            new_confidence,
            "resolved" if ambiguity_resolved else "clarified"
        ))
        
        return new_confidence
    
    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to level."""
        if confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.5:
            return ConfidenceLevel.LOW
        elif confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH


class ClarificationAgent:
    """Agent for handling ambiguities through clarification."""
    
    def __init__(self, max_clarifications: int = 3):
        self.max_clarifications = max_clarifications
        self.ambiguity_detector = AmbiguityDetector()
        self.question_generator = ClarificationQuestionGenerator()
        self.intent_refiner = IntentRefiner()
        self.confidence_tracker = ConfidenceTracker()
        self.active_sessions: Dict[str, UnderstandingState] = {}
        self.pending_questions: Dict[str, ClarificationQuestion] = {}
    
    def process_query(self, session_id: str, query: str) -> Tuple[bool, Optional[ClarificationQuestion], str]:
        """
        Process user query and determine if clarification needed.
        
        Returns:
            (needs_clarification, clarification_question, message)
        """
        # Detect ambiguities
        ambiguities = self.ambiguity_detector.detect_ambiguities(query)
        
        # Calculate confidence
        confidence = self.ambiguity_detector.calculate_understanding_confidence(
            query,
            ambiguities
        )
        
        # Create or update understanding state
        state = UnderstandingState(
            original_query=query,
            current_interpretation=query,
            confidence=confidence,
            ambiguities=ambiguities
        )
        
        self.active_sessions[session_id] = state
        
        # Determine if clarification needed
        confidence_level = self.confidence_tracker.get_confidence_level(confidence)
        
        if confidence_level in [ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW]:
            # Need clarification
            if state.clarifications_asked >= self.max_clarifications:
                return False, None, "I'm having trouble understanding. Could you rephrase your entire request?"
            
            # Generate clarification question for highest-confidence ambiguity
            if ambiguities:
                primary_ambiguity = max(ambiguities, key=lambda a: a.confidence)
                question = self.question_generator.generate_question(primary_ambiguity)
                self.pending_questions[question.question_id] = question
                state.clarifications_asked += 1
                
                # Format question with options if available
                if question.options:
                    options_text = "\n".join(f"  {i+1}. {opt}" for i, opt in enumerate(question.options))
                    message = f"{question.question_text}\n{options_text}"
                else:
                    message = question.question_text
                
                return True, question, message
        
        # Confidence acceptable, proceed
        message = f"I understand your request (confidence: {confidence:.2f})"
        if ambiguities:
            message += f", though I detected {len(ambiguities)} potential ambiguities."
        
        return False, None, message
    
    def handle_clarification_response(
        self,
        session_id: str,
        question_id: str,
        response: str
    ) -> Tuple[bool, str]:
        """
        Handle user's response to clarification question.
        
        Returns:
            (resolved, message)
        """
        if question_id not in self.pending_questions:
            return False, "No pending clarification found"
        
        if session_id not in self.active_sessions:
            return False, "Session not found"
        
        question = self.pending_questions[question_id]
        state = self.active_sessions[session_id]
        
        # Parse response
        selected_option = None
        if question.options:
            # Try to extract option number
            response_lower = response.lower().strip()
            for i, option in enumerate(question.options, 1):
                if str(i) in response_lower or option.lower() in response_lower:
                    selected_option = option
                    break
        
        # Create clarification response
        clarification_response = ClarificationResponse(
            question_id=question_id,
            user_response=response,
            selected_option=selected_option,
            resolved=selected_option is not None
        )
        
        # Refine intent
        refined_intent = self.intent_refiner.refine_intent(
            state.current_interpretation,
            clarification_response,
            question.ambiguity
        )
        
        state.current_interpretation = refined_intent
        
        # Update confidence
        state.confidence = self.confidence_tracker.update_confidence(
            state.confidence,
            clarification_provided=True,
            ambiguity_resolved=clarification_response.resolved
        )
        
        if clarification_response.resolved:
            state.resolved_ambiguities += 1
            message = f"Thank you! I now understand you mean: {refined_intent}"
        else:
            message = f"I see. Current understanding: {refined_intent}"
        
        # Clean up
        del self.pending_questions[question_id]
        
        return clarification_response.resolved, message
    
    def get_understanding_report(self, session_id: str) -> str:
        """Get report on current understanding."""
        if session_id not in self.active_sessions:
            return "No active session found"
        
        state = self.active_sessions[session_id]
        confidence_level = self.confidence_tracker.get_confidence_level(state.confidence)
        
        lines = [
            f"Understanding Report for Session {session_id}",
            f"=" * 60,
            f"Original Query: {state.original_query}",
            f"Current Interpretation: {state.current_interpretation}",
            f"Confidence: {state.confidence:.2f} ({confidence_level.value})",
            f"Ambiguities Detected: {len(state.ambiguities)}",
            f"Clarifications Asked: {state.clarifications_asked}",
            f"Ambiguities Resolved: {state.resolved_ambiguities}",
        ]
        
        if state.ambiguities:
            lines.append("\nDetected Ambiguities:")
            for i, amb in enumerate(state.ambiguities, 1):
                lines.append(f"  {i}. {amb.ambiguity_type.value}: '{amb.ambiguous_element}'")
                lines.append(f"     Interpretations: {', '.join(amb.possible_interpretations)}")
        
        return "\n".join(lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get clarification statistics."""
        total_sessions = len(self.active_sessions)
        total_ambiguities = sum(len(s.ambiguities) for s in self.active_sessions.values())
        total_clarifications = sum(s.clarifications_asked for s in self.active_sessions.values())
        total_resolved = sum(s.resolved_ambiguities for s in self.active_sessions.values())
        
        return {
            'total_sessions': total_sessions,
            'total_ambiguities_detected': total_ambiguities,
            'total_clarifications_asked': total_clarifications,
            'total_resolved': total_resolved,
            'resolution_rate': total_resolved / total_clarifications if total_clarifications > 0 else 0,
            'avg_confidence': sum(s.confidence for s in self.active_sessions.values()) / total_sessions if total_sessions > 0 else 0
        }


def demonstrate_clarification():
    """Demonstrate the Clarification Agent."""
    print("=" * 60)
    print("Clarification Agent Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = ClarificationAgent(max_clarifications=3)
    
    print("\n1. PROCESSING AMBIGUOUS QUERY")
    print("-" * 60)
    
    # Query with lexical ambiguity
    session1 = "session_001"
    query1 = "I need to book a table at the bank"
    
    print(f"User Query: {query1}")
    needs_clarification, question, message = agent.process_query(session1, query1)
    
    print(f"\nAgent Response: {message}")
    print(f"Needs Clarification: {needs_clarification}")
    
    if needs_clarification and question:
        print(f"\nClarification Strategy: {question.strategy.value}")
        print(f"Ambiguity Type: {question.ambiguity.ambiguity_type.value}")
        
        # Provide clarification
        print("\n--- User responds ---")
        clarification = "1"  # Select financial institution
        print(f"User: {clarification}")
        
        resolved, response = agent.handle_clarification_response(
            session1,
            question.question_id,
            clarification
        )
        
        print(f"\nAgent: {response}")
        print(f"Resolved: {resolved}")
    
    # Query with referential ambiguity
    print("\n\n2. REFERENTIAL AMBIGUITY")
    print("-" * 60)
    
    session2 = "session_002"
    query2 = "I saw it yesterday and want to buy it"
    
    print(f"User Query: {query2}")
    needs_clarification, question, message = agent.process_query(session2, query2)
    
    print(f"\nAgent Response: {message}")
    
    if needs_clarification and question:
        print(f"Clarification needed for: {question.ambiguity.ambiguous_element}")
        
        print("\n--- User responds ---")
        clarification = "The blue jacket from the store window"
        print(f"User: {clarification}")
        
        resolved, response = agent.handle_clarification_response(
            session2,
            question.question_id,
            clarification
        )
        
        print(f"\nAgent: {response}")
    
    # Clear query (no ambiguity)
    print("\n\n3. CLEAR QUERY (NO CLARIFICATION NEEDED)")
    print("-" * 60)
    
    session3 = "session_003"
    query3 = "Please send the report to john@example.com by Friday"
    
    print(f"User Query: {query3}")
    needs_clarification, question, message = agent.process_query(session3, query3)
    
    print(f"\nAgent Response: {message}")
    print(f"Needs Clarification: {needs_clarification}")
    
    # Multiple ambiguities
    print("\n\n4. MULTIPLE AMBIGUITIES")
    print("-" * 60)
    
    session4 = "session_004"
    query4 = "Can you tell me about that light spring book?"
    
    print(f"User Query: {query4}")
    needs_clarification, question, message = agent.process_query(session4, query4)
    
    print(f"\nAgent Response: {message}")
    
    if needs_clarification and question:
        print(f"\nFirst ambiguity to resolve: {question.ambiguity.ambiguous_element}")
        
        # Resolve first ambiguity
        print("\n--- User responds ---")
        clarification = "2"  # Choose interpretation
        print(f"User: {clarification}")
        
        resolved, response = agent.handle_clarification_response(
            session4,
            question.question_id,
            clarification
        )
        
        print(f"\nAgent: {response}")
    
    # Understanding reports
    print("\n\n5. UNDERSTANDING REPORTS")
    print("-" * 60)
    
    for session in [session1, session2, session4]:
        print(f"\n{agent.get_understanding_report(session)}")
        print()
    
    # Intent refinement path
    print("\n\n6. INTENT REFINEMENT PATH")
    print("-" * 60)
    
    refinement_path = agent.intent_refiner.get_refinement_path(query1)
    print("Refinement path for first query:")
    for i, step in enumerate(refinement_path):
        print(f"  Step {i+1}: {step}")
    
    # Statistics
    print("\n\n7. STATISTICS")
    print("-" * 60)
    
    stats = agent.get_statistics()
    print(f"  Total Sessions: {stats['total_sessions']}")
    print(f"  Ambiguities Detected: {stats['total_ambiguities_detected']}")
    print(f"  Clarifications Asked: {stats['total_clarifications_asked']}")
    print(f"  Resolved: {stats['total_resolved']}")
    print(f"  Resolution Rate: {stats['resolution_rate']:.1%}")
    print(f"  Average Confidence: {stats['avg_confidence']:.2f}")
    
    print("\n" + "=" * 60)
    print("🎉 Pattern 128 Complete!")
    print("Dialogue & Interaction Category: 50%")
    print("128/170 patterns implemented (75.3%)!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_clarification()
