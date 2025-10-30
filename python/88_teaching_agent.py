"""
Teaching/Tutoring Agent Pattern

Enables agents to act as tutors, providing personalized instruction and guidance
through Socratic questioning, adaptive difficulty, and pedagogical strategies.

Key Concepts:
- Socratic method (learning through questioning)
- Adaptive difficulty
- Knowledge assessment
- Personalized learning paths
- Scaffolding

Use Cases:
- Educational tutoring
- Skill training
- Concept explanation
- Problem-solving guidance
- Assessment and feedback
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class QuestionType(Enum):
    """Types of pedagogical questions."""
    CLARIFYING = "clarifying"  # "Can you explain what you mean?"
    PROBING = "probing"  # "Why do you think that?"
    HYPOTHETICAL = "hypothetical"  # "What if...?"
    CONNECTING = "connecting"  # "How does this relate to...?"
    EVALUATING = "evaluating"  # "What evidence supports this?"
    REDIRECTING = "redirecting"  # "Let's look at it differently"


class DifficultyLevel(Enum):
    """Difficulty levels for content."""
    BEGINNER = 1
    ELEMENTARY = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5


class LearningStyle(Enum):
    """Learning style preferences."""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"


class ConceptStatus(Enum):
    """Student understanding status."""
    NOT_INTRODUCED = "not_introduced"
    INTRODUCED = "introduced"
    PRACTICING = "practicing"
    UNDERSTOOD = "understood"
    MASTERED = "mastered"


@dataclass
class Concept:
    """A concept to be learned."""
    concept_id: str
    name: str
    description: str
    difficulty: DifficultyLevel
    prerequisites: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    common_misconceptions: List[str] = field(default_factory=list)


@dataclass
class StudentProfile:
    """Profile of a student."""
    student_id: str
    name: str
    current_level: DifficultyLevel
    learning_style: LearningStyle
    concept_knowledge: Dict[str, ConceptStatus] = field(default_factory=dict)
    strengths: Set[str] = field(default_factory=set)
    areas_for_improvement: Set[str] = field(default_factory=set)
    learning_pace: float = 1.0  # Multiplier for content progression
    
    def get_mastery_level(self) -> float:
        """Calculate overall mastery as percentage."""
        if not self.concept_knowledge:
            return 0.0
        
        status_values = {
            ConceptStatus.NOT_INTRODUCED: 0.0,
            ConceptStatus.INTRODUCED: 0.2,
            ConceptStatus.PRACTICING: 0.5,
            ConceptStatus.UNDERSTOOD: 0.8,
            ConceptStatus.MASTERED: 1.0
        }
        
        total = sum(status_values[status] for status in self.concept_knowledge.values())
        return total / len(self.concept_knowledge)


@dataclass
class LearningSession:
    """A tutoring session."""
    session_id: str
    student_id: str
    concept_id: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    assessment_score: Optional[float] = None
    
    def add_interaction(self, speaker: str, message: str, question_type: Optional[QuestionType] = None):
        """Add interaction to session."""
        self.interactions.append({
            "timestamp": datetime.now(),
            "speaker": speaker,
            "message": message,
            "question_type": question_type.value if question_type else None
        })


class KnowledgeBase:
    """Knowledge base for teaching content."""
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.concept_relationships: Dict[str, List[str]] = {}
    
    def add_concept(self, concept: Concept) -> None:
        """Add concept to knowledge base."""
        self.concepts[concept.concept_id] = concept
        
        # Build relationship graph
        for prereq in concept.prerequisites:
            if prereq not in self.concept_relationships:
                self.concept_relationships[prereq] = []
            self.concept_relationships[prereq].append(concept.concept_id)
    
    def get_next_concepts(self, mastered_concepts: Set[str]) -> List[Concept]:
        """Get concepts ready to learn based on prerequisites."""
        ready = []
        
        for concept in self.concepts.values():
            if concept.concept_id in mastered_concepts:
                continue
            
            # Check if all prerequisites are mastered
            prereqs_met = all(
                prereq in mastered_concepts 
                for prereq in concept.prerequisites
            )
            
            if prereqs_met:
                ready.append(concept)
        
        return sorted(ready, key=lambda c: c.difficulty.value)
    
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get concept by ID."""
        return self.concepts.get(concept_id)


class TeachingAgent:
    """Agent that tutors students using pedagogical strategies."""
    
    def __init__(self, agent_id: str, name: str, subject: str):
        self.agent_id = agent_id
        self.name = name
        self.subject = subject
        self.knowledge_base = KnowledgeBase()
        self.student_profiles: Dict[str, StudentProfile] = {}
        self.active_sessions: Dict[str, LearningSession] = {}
    
    def register_student(self, profile: StudentProfile) -> None:
        """Register a new student."""
        self.student_profiles[profile.student_id] = profile
        print(f"[{self.name}] Registered student: {profile.name}")
    
    def start_session(
        self,
        student_id: str,
        concept_id: str
    ) -> Optional[LearningSession]:
        """Start a tutoring session."""
        if student_id not in self.student_profiles:
            return None
        
        concept = self.knowledge_base.get_concept(concept_id)
        if not concept:
            return None
        
        session = LearningSession(
            session_id=str(uuid.uuid4()),
            student_id=student_id,
            concept_id=concept_id
        )
        
        self.active_sessions[session.session_id] = session
        
        student = self.student_profiles[student_id]
        print(f"\n[{self.name}] Starting session with {student.name}")
        print(f"Topic: {concept.name}")
        print(f"Student level: {student.current_level.name}")
        
        return session
    
    def introduce_concept(
        self,
        session: LearningSession
    ) -> str:
        """Introduce a new concept to student."""
        concept = self.knowledge_base.get_concept(session.concept_id)
        if not concept:
            return "Concept not found"
        
        student = self.student_profiles[session.student_id]
        
        # Adapt introduction to learning style
        intro = self._adapt_to_learning_style(
            concept.description,
            student.learning_style
        )
        
        # Update student knowledge
        student.concept_knowledge[concept.concept_id] = ConceptStatus.INTRODUCED
        
        session.add_interaction("tutor", intro)
        print(f"\n[{self.name}]: {intro}")
        
        return intro
    
    def ask_socratic_question(
        self,
        session: LearningSession,
        question_type: QuestionType,
        context: str = ""
    ) -> str:
        """Ask a Socratic question to guide learning."""
        concept = self.knowledge_base.get_concept(session.concept_id)
        if not concept:
            return "Concept not found"
        
        questions = {
            QuestionType.CLARIFYING: [
                f"Can you explain in your own words what {concept.name} means?",
                f"What do you understand by {concept.name}?",
                "Could you clarify what you mean by that?"
            ],
            QuestionType.PROBING: [
                f"Why do you think {concept.name} works this way?",
                "What makes you say that?",
                "Can you give me a reason for your answer?"
            ],
            QuestionType.HYPOTHETICAL: [
                f"What would happen if we changed this aspect of {concept.name}?",
                "Imagine a scenario where this doesn't apply. What would that look like?",
                "What if we tried a different approach?"
            ],
            QuestionType.CONNECTING: [
                f"How does {concept.name} relate to what we learned earlier?",
                "Can you see any patterns or connections here?",
                "Where else might you use this concept?"
            ],
            QuestionType.EVALUATING: [
                "What evidence supports your conclusion?",
                "How would you test if this is correct?",
                "What are the strengths and weaknesses of this approach?"
            ],
            QuestionType.REDIRECTING: [
                "Let's think about it from a different angle.",
                "What if we started with a simpler example?",
                "Could there be another way to look at this?"
            ]
        }
        
        import random
        question = random.choice(questions[question_type])
        
        if context:
            question = f"{context} {question}"
        
        session.add_interaction("tutor", question, question_type)
        print(f"\n[{self.name}] ({question_type.value}): {question}")
        
        return question
    
    def provide_feedback(
        self,
        session: LearningSession,
        student_response: str,
        is_correct: bool
    ) -> str:
        """Provide constructive feedback."""
        concept = self.knowledge_base.get_concept(session.concept_id)
        if not concept:
            return "Concept not found"
        
        if is_correct:
            feedback = self._positive_feedback(student_response)
            
            # Update student progress
            student = self.student_profiles[session.student_id]
            current_status = student.concept_knowledge.get(
                concept.concept_id,
                ConceptStatus.NOT_INTRODUCED
            )
            
            if current_status == ConceptStatus.INTRODUCED:
                student.concept_knowledge[concept.concept_id] = ConceptStatus.PRACTICING
            elif current_status == ConceptStatus.PRACTICING:
                student.concept_knowledge[concept.concept_id] = ConceptStatus.UNDERSTOOD
        else:
            feedback = self._corrective_feedback(student_response, concept)
        
        session.add_interaction("tutor", feedback)
        print(f"\n[{self.name}]: {feedback}")
        
        return feedback
    
    def provide_hint(
        self,
        session: LearningSession,
        difficulty: int = 1
    ) -> str:
        """Provide a scaffolded hint."""
        concept = self.knowledge_base.get_concept(session.concept_id)
        if not concept:
            return "Concept not found"
        
        hints = [
            f"Let's start with what we know about {concept.name}...",
            f"Remember the key characteristic of {concept.name}...",
            f"Think about the examples we discussed earlier.",
            "Try breaking the problem into smaller steps.",
            f"Here's a direct clue: {concept.examples[0] if concept.examples else 'Consider the definition.'}"
        ]
        
        hint = hints[min(difficulty - 1, len(hints) - 1)]
        
        session.add_interaction("tutor", f"Hint: {hint}")
        print(f"\n[{self.name}] (Hint): {hint}")
        
        return hint
    
    def assess_understanding(
        self,
        session: LearningSession
    ) -> float:
        """Assess student's understanding of concept."""
        # Simplified assessment based on session interactions
        total_questions = sum(
            1 for i in session.interactions 
            if i.get("question_type") is not None
        )
        
        if total_questions == 0:
            score = 0.5
        else:
            # Mock scoring based on interaction count (in real system, would analyze responses)
            score = min(1.0, len(session.interactions) / (total_questions * 3))
        
        session.assessment_score = score
        
        # Update student profile
        student = self.student_profiles[session.student_id]
        concept = self.knowledge_base.get_concept(session.concept_id)
        
        if concept:
            if score >= 0.8:
                student.concept_knowledge[concept.concept_id] = ConceptStatus.MASTERED
                student.strengths.add(concept.concept_id)
            elif score >= 0.6:
                student.concept_knowledge[concept.concept_id] = ConceptStatus.UNDERSTOOD
            else:
                student.areas_for_improvement.add(concept.concept_id)
        
        print(f"\n[{self.name}] Assessment score: {score:.2f}")
        
        return score
    
    def recommend_next_topic(
        self,
        student_id: str
    ) -> Optional[Concept]:
        """Recommend next topic based on student progress."""
        student = self.student_profiles[student_id]
        
        # Get mastered concepts
        mastered = {
            cid for cid, status in student.concept_knowledge.items()
            if status in {ConceptStatus.UNDERSTOOD, ConceptStatus.MASTERED}
        }
        
        # Find next available concepts
        next_concepts = self.knowledge_base.get_next_concepts(mastered)
        
        if not next_concepts:
            return None
        
        # Select based on difficulty and student level
        for concept in next_concepts:
            if concept.difficulty.value <= student.current_level.value + 1:
                print(f"\n[{self.name}] Recommending next: {concept.name}")
                return concept
        
        return next_concepts[0] if next_concepts else None
    
    def _adapt_to_learning_style(
        self,
        content: str,
        style: LearningStyle
    ) -> str:
        """Adapt content presentation to learning style."""
        adaptations = {
            LearningStyle.VISUAL: "Let me draw a picture for you: ",
            LearningStyle.AUDITORY: "Listen to this explanation: ",
            LearningStyle.KINESTHETIC: "Let's work through this hands-on: ",
            LearningStyle.READING_WRITING: "Here's the written explanation: "
        }
        
        return adaptations.get(style, "") + content
    
    def _positive_feedback(self, response: str) -> str:
        """Generate positive feedback."""
        encouragements = [
            "Excellent! You've grasped the key concept.",
            "Great job! Your understanding is clear.",
            "Perfect! That's exactly right.",
            "Well done! You're making great progress.",
            "Correct! I can see you've really thought about this."
        ]
        
        import random
        return random.choice(encouragements)
    
    def _corrective_feedback(self, response: str, concept: Concept) -> str:
        """Generate corrective feedback."""
        # Check for common misconceptions
        if concept.common_misconceptions:
            return f"Not quite. A common misconception is thinking that... Let's revisit the core idea of {concept.name}."
        
        return f"Let's think about this differently. Remember that {concept.name} involves..."
    
    def end_session(self, session_id: str) -> None:
        """End tutoring session."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.end_time = datetime.now()
            
            # Final assessment
            score = self.assess_understanding(session)
            
            student = self.student_profiles[session.student_id]
            print(f"\n[{self.name}] Session ended")
            print(f"Overall mastery: {student.get_mastery_level() * 100:.1f}%")
            
            del self.active_sessions[session_id]


def demonstrate_teaching_agent():
    """Demonstrate teaching/tutoring agent pattern."""
    print("=" * 60)
    print("TEACHING/TUTORING AGENT DEMONSTRATION")
    print("=" * 60)
    
    # Create teaching agent
    tutor = TeachingAgent("tutor1", "Professor AI", "Computer Science")
    
    # Build knowledge base
    print("\n" + "=" * 60)
    print("1. Building Knowledge Base")
    print("=" * 60)
    
    # Add concepts
    variables = Concept(
        concept_id="variables",
        name="Variables",
        description="Variables are containers for storing data values",
        difficulty=DifficultyLevel.BEGINNER,
        examples=["x = 5", "name = 'Alice'"],
        common_misconceptions=["Variables are permanent", "Variable names must be single letters"]
    )
    
    loops = Concept(
        concept_id="loops",
        name="Loops",
        description="Loops allow you to repeat code multiple times",
        difficulty=DifficultyLevel.ELEMENTARY,
        prerequisites=["variables"],
        examples=["for i in range(10):", "while condition:"],
        common_misconceptions=["Infinite loops are always bad", "You can't break out of loops"]
    )
    
    functions = Concept(
        concept_id="functions",
        name="Functions",
        description="Functions are reusable blocks of code that perform specific tasks",
        difficulty=DifficultyLevel.INTERMEDIATE,
        prerequisites=["variables", "loops"],
        examples=["def greet(name):", "return value"],
        common_misconceptions=["Functions must always return something", "Parameters and arguments are the same"]
    )
    
    tutor.knowledge_base.add_concept(variables)
    tutor.knowledge_base.add_concept(loops)
    tutor.knowledge_base.add_concept(functions)
    
    print(f"Added {len(tutor.knowledge_base.concepts)} concepts")
    
    # Create student profile
    print("\n" + "=" * 60)
    print("2. Student Registration")
    print("=" * 60)
    
    student = StudentProfile(
        student_id="student1",
        name="Alice",
        current_level=DifficultyLevel.BEGINNER,
        learning_style=LearningStyle.VISUAL
    )
    
    tutor.register_student(student)
    
    # Start learning session on variables
    print("\n" + "=" * 60)
    print("3. Teaching Session: Variables")
    print("=" * 60)
    
    session1 = tutor.start_session("student1", "variables")
    if not session1:
        print("Failed to start session")
        return
    
    # Introduce concept
    tutor.introduce_concept(session1)
    
    # Ask Socratic questions
    tutor.ask_socratic_question(session1, QuestionType.CLARIFYING)
    session1.add_interaction("student", "A variable is like a box that holds a value")
    
    tutor.provide_feedback(session1, "A variable is like a box", is_correct=True)
    
    tutor.ask_socratic_question(session1, QuestionType.PROBING)
    session1.add_interaction("student", "Because we need to store and reuse data")
    
    tutor.provide_feedback(session1, "store and reuse", is_correct=True)
    
    tutor.ask_socratic_question(session1, QuestionType.HYPOTHETICAL)
    session1.add_interaction("student", "We could use it but the name wouldn't be clear")
    
    # Provide hint for practice
    tutor.provide_hint(session1, difficulty=1)
    
    # End session and assess
    tutor.end_session(session1.session_id)
    
    # Recommend next topic
    print("\n" + "=" * 60)
    print("4. Personalized Learning Path")
    print("=" * 60)
    
    next_topic = tutor.recommend_next_topic("student1")
    
    if next_topic:
        print(f"Ready to learn: {next_topic.name}")
        print(f"Prerequisites met: {next_topic.prerequisites}")
    
    # Start session on loops
    print("\n" + "=" * 60)
    print("5. Teaching Session: Loops")
    print("=" * 60)
    
    session2 = tutor.start_session("student1", "loops")
    if not session2:
        print("Failed to start session")
        return
    
    tutor.introduce_concept(session2)
    tutor.ask_socratic_question(session2, QuestionType.CONNECTING)
    session2.add_interaction("student", "Loops use variables to count iterations")
    
    tutor.provide_feedback(session2, "use variables to count", is_correct=True)
    
    tutor.end_session(session2.session_id)
    
    # Show student progress
    print("\n" + "=" * 60)
    print("6. Student Progress Report")
    print("=" * 60)
    
    print(f"\nStudent: {student.name}")
    print(f"Overall Mastery: {student.get_mastery_level() * 100:.1f}%")
    print(f"\nConcepts learned:")
    for concept_id, status in student.concept_knowledge.items():
        concept = tutor.knowledge_base.get_concept(concept_id)
        if concept:
            print(f"  - {concept.name}: {status.value}")
    
    if student.strengths:
        print(f"\nStrengths: {', '.join(student.strengths)}")


if __name__ == "__main__":
    demonstrate_teaching_agent()
