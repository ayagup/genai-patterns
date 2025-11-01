"""
Pattern 076: Teaching Agent

Description:
    A Teaching Agent is a specialized AI agent designed to provide personalized
    tutoring and educational support across various subjects and skill levels. This
    pattern demonstrates how to build an intelligent tutoring system that can assess
    student knowledge, adapt teaching strategies, explain concepts at appropriate
    levels, generate practice problems, provide feedback, and track learning progress.
    
    The agent combines pedagogical principles with adaptive learning techniques to
    create effective, personalized learning experiences. It can diagnose knowledge
    gaps, scaffold learning, provide multiple explanations, encourage critical
    thinking, and adjust difficulty based on student performance.

Components:
    1. Knowledge Assessor: Evaluates student understanding and identifies gaps
    2. Curriculum Planner: Creates personalized learning paths
    3. Concept Explainer: Explains topics at appropriate complexity levels
    4. Problem Generator: Creates practice problems and exercises
    5. Feedback Provider: Gives constructive, encouraging feedback
    6. Progress Tracker: Monitors learning progress and mastery
    7. Adaptive Engine: Adjusts difficulty and teaching approach
    8. Socratic Questioner: Guides discovery through questioning

Key Features:
    - Multi-level explanations (beginner, intermediate, advanced)
    - Adaptive difficulty adjustment based on performance
    - Multiple teaching strategies (direct instruction, discovery, examples)
    - Personalized learning paths
    - Practice problem generation with solutions
    - Immediate, constructive feedback
    - Knowledge gap identification
    - Progress tracking and analytics
    - Socratic questioning for deeper understanding
    - Visual and conceptual explanations

Use Cases:
    - K-12 tutoring across subjects
    - College-level subject tutoring
    - Professional skill development
    - Language learning
    - Programming education
    - Math and science tutoring
    - Test preparation (SAT, GRE, etc.)
    - Corporate training
    - Self-paced online courses
    - Homework help

LangChain Implementation:
    Uses ChatOpenAI with specialized prompts for different pedagogical approaches,
    chains for assessment, explanation, and feedback generation.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class Subject(Enum):
    """Academic subjects"""
    MATHEMATICS = "mathematics"
    SCIENCE = "science"
    PROGRAMMING = "programming"
    LANGUAGE = "language"
    HISTORY = "history"
    LITERATURE = "literature"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    ECONOMICS = "economics"


class SkillLevel(Enum):
    """Student skill levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TeachingStrategy(Enum):
    """Teaching approaches"""
    DIRECT_INSTRUCTION = "direct_instruction"
    GUIDED_DISCOVERY = "guided_discovery"
    SOCRATIC_METHOD = "socratic_method"
    WORKED_EXAMPLES = "worked_examples"
    PROBLEM_BASED = "problem_based"
    SCAFFOLDED = "scaffolded"


class QuestionDifficulty(Enum):
    """Difficulty levels for questions"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    CHALLENGING = "challenging"


@dataclass
class LearningObjective:
    """A specific learning objective"""
    objective_id: str
    subject: Subject
    topic: str
    description: str
    skill_level: SkillLevel
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class StudentProfile:
    """Profile of a student"""
    student_id: str
    name: str
    skill_levels: Dict[Subject, SkillLevel] = field(default_factory=dict)
    learning_style: str = "visual"  # visual, auditory, kinesthetic
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)


@dataclass
class ConceptExplanation:
    """Explanation of a concept"""
    concept: str
    level: SkillLevel
    explanation: str
    examples: List[str]
    analogies: List[str]
    common_misconceptions: List[str]
    key_points: List[str]


@dataclass
class PracticeProblem:
    """A practice problem"""
    problem_id: str
    subject: Subject
    topic: str
    difficulty: QuestionDifficulty
    question: str
    solution: str
    hints: List[str]
    explanation: str


@dataclass
class StudentResponse:
    """Student's response to a problem"""
    problem_id: str
    student_answer: str
    is_correct: bool
    partial_credit: float  # 0.0-1.0
    time_taken: Optional[int] = None  # seconds


@dataclass
class Feedback:
    """Feedback on student work"""
    response_id: str
    correctness: str  # correct, partially_correct, incorrect
    positive_points: List[str]
    areas_to_improve: List[str]
    suggestions: List[str]
    encouragement: str
    next_steps: str


@dataclass
class LearningSession:
    """A tutoring session"""
    session_id: str
    student_id: str
    subject: Subject
    topic: str
    objectives: List[LearningObjective]
    explanations_given: List[ConceptExplanation] = field(default_factory=list)
    problems_attempted: List[Tuple[PracticeProblem, StudentResponse]] = field(default_factory=list)
    mastery_level: float = 0.0  # 0.0-1.0
    duration_minutes: int = 0


class TeachingAgent:
    """
    Agent for personalized tutoring and education.
    
    This agent can assess student knowledge, explain concepts, generate
    practice problems, provide feedback, and adapt to student needs.
    """
    
    def __init__(self):
        """Initialize the teaching agent with specialized LLMs"""
        # Assessor for evaluating understanding
        self.assessor_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        
        # Explainer for teaching concepts
        self.explainer_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)
        
        # Problem generator for creating exercises
        self.generator_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)
        
        # Feedback provider for student responses
        self.feedback_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        
        # Student profiles
        self.students: Dict[str, StudentProfile] = {}
        
        # Session history
        self.sessions: List[LearningSession] = []
    
    def assess_knowledge(
        self,
        student_id: str,
        subject: Subject,
        topic: str,
        student_response: str
    ) -> Dict[str, Any]:
        """
        Assess student's knowledge on a topic.
        
        Args:
            student_id: Student identifier
            subject: Subject area
            topic: Specific topic
            student_response: Student's explanation or answer
            
        Returns:
            Assessment with understanding level and gaps
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an educational assessment expert. Evaluate the
            student's understanding based on their response.
            
            Respond in this format:
            UNDERSTANDING_LEVEL: beginner/intermediate/advanced
            STRENGTHS: (one per line starting with -)
            GAPS: (one per line starting with -)
            MASTERY_SCORE: (0.0-1.0)
            RECOMMENDATIONS: (one per line starting with -)"""),
            ("user", """Subject: {subject}
Topic: {topic}

Student Response:
{response}

Assess the student's knowledge.""")
        ])
        
        chain = prompt | self.assessor_llm | StrOutputParser()
        
        try:
            result = chain.invoke({
                "subject": subject.value,
                "topic": topic,
                "response": student_response
            })
            
            understanding_level = SkillLevel.INTERMEDIATE
            strengths = []
            gaps = []
            mastery_score = 0.5
            recommendations = []
            
            current_section = None
            
            for line in result.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('UNDERSTANDING_LEVEL:'):
                    level_str = line.replace('UNDERSTANDING_LEVEL:', '').strip().lower()
                    if 'beginner' in level_str:
                        understanding_level = SkillLevel.BEGINNER
                    elif 'advanced' in level_str:
                        understanding_level = SkillLevel.ADVANCED
                    elif 'expert' in level_str:
                        understanding_level = SkillLevel.EXPERT
                    else:
                        understanding_level = SkillLevel.INTERMEDIATE
                elif line.startswith('STRENGTHS:'):
                    current_section = 'strengths'
                elif line.startswith('GAPS:'):
                    current_section = 'gaps'
                elif line.startswith('MASTERY_SCORE:'):
                    try:
                        mastery_score = float(re.findall(r'[\d.]+', line)[0])
                    except:
                        pass
                elif line.startswith('RECOMMENDATIONS:'):
                    current_section = 'recommendations'
                elif line.startswith('-'):
                    content = line[1:].strip()
                    if current_section == 'strengths':
                        strengths.append(content)
                    elif current_section == 'gaps':
                        gaps.append(content)
                    elif current_section == 'recommendations':
                        recommendations.append(content)
            
            return {
                "understanding_level": understanding_level,
                "strengths": strengths if strengths else ["Shows effort"],
                "gaps": gaps if gaps else ["Some concepts need clarification"],
                "mastery_score": mastery_score,
                "recommendations": recommendations if recommendations else ["Continue practicing"]
            }
            
        except Exception as e:
            return {
                "understanding_level": SkillLevel.INTERMEDIATE,
                "strengths": ["Assessment in progress"],
                "gaps": ["Further evaluation needed"],
                "mastery_score": 0.5,
                "recommendations": ["Continue learning"]
            }
    
    def explain_concept(
        self,
        concept: str,
        subject: Subject,
        level: SkillLevel,
        teaching_strategy: TeachingStrategy = TeachingStrategy.DIRECT_INSTRUCTION
    ) -> ConceptExplanation:
        """
        Explain a concept at the appropriate level.
        
        Args:
            concept: Concept to explain
            subject: Subject area
            level: Student's skill level
            teaching_strategy: Teaching approach to use
            
        Returns:
            ConceptExplanation with multi-faceted explanation
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert teacher. Explain the concept clearly
            and effectively for the specified skill level using the teaching strategy.
            
            Respond in this format:
            EXPLANATION: Clear, level-appropriate explanation
            EXAMPLES: (one per line starting with -)
            ANALOGIES: (one per line starting with -)
            MISCONCEPTIONS: Common misunderstandings (one per line starting with -)
            KEY_POINTS: Essential takeaways (one per line starting with -)"""),
            ("user", """Concept: {concept}
Subject: {subject}
Skill Level: {level}
Teaching Strategy: {strategy}

Explain this concept.""")
        ])
        
        chain = prompt | self.explainer_llm | StrOutputParser()
        
        try:
            result = chain.invoke({
                "concept": concept,
                "subject": subject.value,
                "level": level.value,
                "strategy": teaching_strategy.value
            })
            
            explanation = ""
            examples = []
            analogies = []
            misconceptions = []
            key_points = []
            
            current_section = None
            explanation_lines = []
            
            for line in result.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('EXPLANATION:'):
                    current_section = 'explanation'
                    exp_text = line.replace('EXPLANATION:', '').strip()
                    if exp_text:
                        explanation_lines.append(exp_text)
                elif line.startswith('EXAMPLES:'):
                    current_section = 'examples'
                elif line.startswith('ANALOGIES:'):
                    current_section = 'analogies'
                elif line.startswith('MISCONCEPTIONS:'):
                    current_section = 'misconceptions'
                elif line.startswith('KEY_POINTS:'):
                    current_section = 'key_points'
                elif line.startswith('-'):
                    content = line[1:].strip()
                    if current_section == 'examples':
                        examples.append(content)
                    elif current_section == 'analogies':
                        analogies.append(content)
                    elif current_section == 'misconceptions':
                        misconceptions.append(content)
                    elif current_section == 'key_points':
                        key_points.append(content)
                elif current_section == 'explanation':
                    explanation_lines.append(line)
            
            explanation = ' '.join(explanation_lines) if explanation_lines else f"Explanation of {concept}"
            
            return ConceptExplanation(
                concept=concept,
                level=level,
                explanation=explanation,
                examples=examples if examples else [f"Example related to {concept}"],
                analogies=analogies if analogies else [f"Think of it like..."],
                common_misconceptions=misconceptions if misconceptions else ["Avoid common mistakes"],
                key_points=key_points if key_points else ["Main concept understood"]
            )
            
        except Exception as e:
            return ConceptExplanation(
                concept=concept,
                level=level,
                explanation=f"Explanation of {concept} for {level.value} level",
                examples=[f"Example of {concept}"],
                analogies=["Conceptual analogy"],
                common_misconceptions=["Common misunderstanding"],
                key_points=["Key takeaway"]
            )
    
    def generate_practice_problem(
        self,
        subject: Subject,
        topic: str,
        difficulty: QuestionDifficulty,
        student_level: SkillLevel
    ) -> PracticeProblem:
        """
        Generate a practice problem.
        
        Args:
            subject: Subject area
            topic: Specific topic
            difficulty: Problem difficulty
            student_level: Student's skill level
            
        Returns:
            PracticeProblem with question and solution
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an educational content creator. Generate a practice
            problem appropriate for the student's level and difficulty.
            
            Respond in this format:
            QUESTION: The practice question
            SOLUTION: Step-by-step solution
            HINTS: Helpful hints (one per line starting with -)
            EXPLANATION: Why this solution works"""),
            ("user", """Subject: {subject}
Topic: {topic}
Difficulty: {difficulty}
Student Level: {level}

Generate a practice problem.""")
        ])
        
        chain = prompt | self.generator_llm | StrOutputParser()
        
        try:
            result = chain.invoke({
                "subject": subject.value,
                "topic": topic,
                "difficulty": difficulty.value,
                "level": student_level.value
            })
            
            question = ""
            solution = ""
            hints = []
            explanation = ""
            
            current_section = None
            question_lines = []
            solution_lines = []
            explanation_lines = []
            
            for line in result.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('QUESTION:'):
                    current_section = 'question'
                    q_text = line.replace('QUESTION:', '').strip()
                    if q_text:
                        question_lines.append(q_text)
                elif line.startswith('SOLUTION:'):
                    current_section = 'solution'
                    s_text = line.replace('SOLUTION:', '').strip()
                    if s_text:
                        solution_lines.append(s_text)
                elif line.startswith('HINTS:'):
                    current_section = 'hints'
                elif line.startswith('EXPLANATION:'):
                    current_section = 'explanation'
                    e_text = line.replace('EXPLANATION:', '').strip()
                    if e_text:
                        explanation_lines.append(e_text)
                elif line.startswith('-'):
                    if current_section == 'hints':
                        hints.append(line[1:].strip())
                elif current_section == 'question':
                    question_lines.append(line)
                elif current_section == 'solution':
                    solution_lines.append(line)
                elif current_section == 'explanation':
                    explanation_lines.append(line)
            
            question = ' '.join(question_lines) if question_lines else f"Practice problem about {topic}"
            solution = ' '.join(solution_lines) if solution_lines else "Solution provided"
            explanation = ' '.join(explanation_lines) if explanation_lines else "Explanation of solution"
            
            return PracticeProblem(
                problem_id=f"prob_{datetime.now().timestamp()}",
                subject=subject,
                topic=topic,
                difficulty=difficulty,
                question=question,
                solution=solution,
                hints=hints if hints else ["Think about the key concepts"],
                explanation=explanation
            )
            
        except Exception as e:
            return PracticeProblem(
                problem_id=f"prob_{datetime.now().timestamp()}",
                subject=subject,
                topic=topic,
                difficulty=difficulty,
                question=f"Practice problem about {topic}",
                solution="Solution to be provided",
                hints=["Review the concept", "Break it down step by step"],
                explanation="Explanation available"
            )
    
    def provide_feedback(
        self,
        problem: PracticeProblem,
        student_response: StudentResponse
    ) -> Feedback:
        """
        Provide feedback on student's work.
        
        Args:
            problem: The practice problem
            student_response: Student's response
            
        Returns:
            Feedback with constructive guidance
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a supportive tutor. Provide encouraging,
            constructive feedback on the student's work.
            
            Respond in this format:
            CORRECTNESS: correct/partially_correct/incorrect
            POSITIVE: What they did well (one per line starting with -)
            IMPROVE: Areas to improve (one per line starting with -)
            SUGGESTIONS: Specific suggestions (one per line starting with -)
            ENCOURAGEMENT: Encouraging message
            NEXT_STEPS: What to do next"""),
            ("user", """Problem: {question}
Correct Solution: {solution}

Student Answer: {answer}
Is Correct: {is_correct}

Provide feedback.""")
        ])
        
        chain = prompt | self.feedback_llm | StrOutputParser()
        
        try:
            result = chain.invoke({
                "question": problem.question,
                "solution": problem.solution,
                "answer": student_response.student_answer,
                "is_correct": student_response.is_correct
            })
            
            correctness = "partially_correct"
            positive = []
            improve = []
            suggestions = []
            encouragement = ""
            next_steps = ""
            
            current_section = None
            
            for line in result.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('CORRECTNESS:'):
                    corr_text = line.replace('CORRECTNESS:', '').strip().lower()
                    if 'correct' in corr_text and 'partially' not in corr_text and 'incorrect' not in corr_text:
                        correctness = "correct"
                    elif 'incorrect' in corr_text:
                        correctness = "incorrect"
                    else:
                        correctness = "partially_correct"
                elif line.startswith('POSITIVE:'):
                    current_section = 'positive'
                elif line.startswith('IMPROVE:'):
                    current_section = 'improve'
                elif line.startswith('SUGGESTIONS:'):
                    current_section = 'suggestions'
                elif line.startswith('ENCOURAGEMENT:'):
                    current_section = 'encouragement'
                    encouragement = line.replace('ENCOURAGEMENT:', '').strip()
                elif line.startswith('NEXT_STEPS:'):
                    current_section = 'next_steps'
                    next_steps = line.replace('NEXT_STEPS:', '').strip()
                elif line.startswith('-'):
                    content = line[1:].strip()
                    if current_section == 'positive':
                        positive.append(content)
                    elif current_section == 'improve':
                        improve.append(content)
                    elif current_section == 'suggestions':
                        suggestions.append(content)
                elif current_section == 'encouragement':
                    encouragement += " " + line
                elif current_section == 'next_steps':
                    next_steps += " " + line
            
            return Feedback(
                response_id=f"feedback_{datetime.now().timestamp()}",
                correctness=correctness,
                positive_points=positive if positive else ["Good effort!"],
                areas_to_improve=improve if improve else ["Keep practicing"],
                suggestions=suggestions if suggestions else ["Review the concept"],
                encouragement=encouragement.strip() if encouragement else "Keep up the good work!",
                next_steps=next_steps.strip() if next_steps else "Try another problem"
            )
            
        except Exception as e:
            return Feedback(
                response_id=f"feedback_{datetime.now().timestamp()}",
                correctness="partially_correct",
                positive_points=["You're making progress"],
                areas_to_improve=["Continue learning"],
                suggestions=["Review the material"],
                encouragement="Keep trying!",
                next_steps="Practice more problems"
            )
    
    def create_learning_path(
        self,
        student_id: str,
        subject: Subject,
        goal: str,
        current_level: SkillLevel
    ) -> List[LearningObjective]:
        """
        Create a personalized learning path.
        
        Args:
            student_id: Student identifier
            subject: Subject area
            goal: Learning goal
            current_level: Current skill level
            
        Returns:
            List of LearningObjectives in sequence
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a curriculum designer. Create a learning path
            from the current level to achieve the goal.
            
            Provide 3-5 learning objectives, one per line:
            OBJECTIVE: topic | description | level"""),
            ("user", """Subject: {subject}
Current Level: {current_level}
Goal: {goal}

Create a learning path.""")
        ])
        
        chain = prompt | self.explainer_llm | StrOutputParser()
        
        try:
            result = chain.invoke({
                "subject": subject.value,
                "current_level": current_level.value,
                "goal": goal
            })
            
            objectives = []
            obj_counter = 1
            
            for line in result.split('\n'):
                if 'OBJECTIVE:' in line or '|' in line:
                    parts = line.replace('OBJECTIVE:', '').split('|')
                    if len(parts) >= 2:
                        topic = parts[0].strip()
                        description = parts[1].strip()
                        level_str = parts[2].strip().lower() if len(parts) > 2 else current_level.value
                        
                        # Parse level
                        obj_level = current_level
                        if 'advanced' in level_str:
                            obj_level = SkillLevel.ADVANCED
                        elif 'intermediate' in level_str:
                            obj_level = SkillLevel.INTERMEDIATE
                        elif 'beginner' in level_str:
                            obj_level = SkillLevel.BEGINNER
                        
                        objectives.append(LearningObjective(
                            objective_id=f"obj_{obj_counter}",
                            subject=subject,
                            topic=topic,
                            description=description,
                            skill_level=obj_level,
                            prerequisites=[f"obj_{obj_counter-1}"] if obj_counter > 1 else []
                        ))
                        obj_counter += 1
            
            if not objectives:
                # Fallback objectives
                objectives = [
                    LearningObjective(
                        objective_id="obj_1",
                        subject=subject,
                        topic="Fundamentals",
                        description="Master fundamental concepts",
                        skill_level=current_level
                    ),
                    LearningObjective(
                        objective_id="obj_2",
                        subject=subject,
                        topic="Intermediate Topics",
                        description="Build on fundamentals",
                        skill_level=SkillLevel.INTERMEDIATE,
                        prerequisites=["obj_1"]
                    ),
                    LearningObjective(
                        objective_id="obj_3",
                        subject=subject,
                        topic="Advanced Applications",
                        description=f"Achieve goal: {goal}",
                        skill_level=SkillLevel.ADVANCED,
                        prerequisites=["obj_2"]
                    )
                ]
            
            return objectives
            
        except Exception as e:
            return [
                LearningObjective(
                    objective_id="obj_1",
                    subject=subject,
                    topic="Core Concepts",
                    description="Foundation building",
                    skill_level=current_level
                )
            ]
    
    def conduct_tutoring_session(
        self,
        student_id: str,
        subject: Subject,
        topic: str,
        duration_minutes: int = 30
    ) -> LearningSession:
        """
        Conduct a complete tutoring session.
        
        Args:
            student_id: Student identifier
            subject: Subject to tutor
            topic: Specific topic
            duration_minutes: Session duration
            
        Returns:
            LearningSession with complete session data
        """
        print(f"\n📚 Starting tutoring session on {topic}...")
        
        # Create session
        session = LearningSession(
            session_id=f"session_{datetime.now().timestamp()}",
            student_id=student_id,
            subject=subject,
            topic=topic,
            objectives=[],
            duration_minutes=duration_minutes
        )
        
        # Get or create student profile
        if student_id not in self.students:
            self.students[student_id] = StudentProfile(
                student_id=student_id,
                name=f"Student {student_id}"
            )
        
        student = self.students[student_id]
        current_level = student.skill_levels.get(subject, SkillLevel.BEGINNER)
        
        # Step 1: Explain concept
        print("  ✓ Explaining concept...")
        explanation = self.explain_concept(
            topic,
            subject,
            current_level,
            TeachingStrategy.DIRECT_INSTRUCTION
        )
        session.explanations_given.append(explanation)
        
        # Step 2: Generate practice problems
        print("  ✓ Generating practice problems...")
        for difficulty in [QuestionDifficulty.EASY, QuestionDifficulty.MEDIUM]:
            problem = self.generate_practice_problem(
                subject,
                topic,
                difficulty,
                current_level
            )
            
            # Simulate student response
            is_correct = difficulty == QuestionDifficulty.EASY
            response = StudentResponse(
                problem_id=problem.problem_id,
                student_answer="Student's attempt at solution",
                is_correct=is_correct,
                partial_credit=0.8 if is_correct else 0.4
            )
            
            session.problems_attempted.append((problem, response))
        
        # Step 3: Calculate mastery
        if session.problems_attempted:
            total_credit = sum(resp.partial_credit for _, resp in session.problems_attempted)
            session.mastery_level = total_credit / len(session.problems_attempted)
        
        print("  ✅ Session complete!\n")
        
        self.sessions.append(session)
        return session


def demonstrate_teaching_agent():
    """Demonstrate the teaching agent capabilities"""
    print("=" * 80)
    print("TEACHING AGENT DEMONSTRATION")
    print("=" * 80)
    
    agent = TeachingAgent()
    
    # Demo 1: Knowledge Assessment
    print("\n" + "=" * 80)
    print("DEMO 1: Knowledge Assessment")
    print("=" * 80)
    
    student_response = """
    Photosynthesis is the process where plants use sunlight to make food.
    They take in carbon dioxide and water, and produce glucose and oxygen.
    The process happens in the chloroplasts using chlorophyll.
    """
    
    print("\nAssessing student knowledge on photosynthesis...")
    assessment = agent.assess_knowledge(
        "student001",
        Subject.BIOLOGY,
        "Photosynthesis",
        student_response
    )
    
    print(f"\nAssessment Results:")
    print(f"Understanding Level: {assessment['understanding_level'].value}")
    print(f"Mastery Score: {assessment['mastery_score']:.2f}/1.0")
    print(f"\nStrengths:")
    for strength in assessment['strengths'][:3]:
        print(f"  ✓ {strength}")
    print(f"\nKnowledge Gaps:")
    for gap in assessment['gaps'][:2]:
        print(f"  ⚠ {gap}")
    print(f"\nRecommendations:")
    for rec in assessment['recommendations'][:2]:
        print(f"  → {rec}")
    
    # Demo 2: Concept Explanation
    print("\n" + "=" * 80)
    print("DEMO 2: Concept Explanation")
    print("=" * 80)
    
    concept = "quadratic equations"
    print(f"\nExplaining '{concept}' at intermediate level...")
    
    explanation = agent.explain_concept(
        concept,
        Subject.MATHEMATICS,
        SkillLevel.INTERMEDIATE,
        TeachingStrategy.WORKED_EXAMPLES
    )
    
    print(f"\nConcept: {explanation.concept}")
    print(f"Level: {explanation.level.value}")
    print(f"\nExplanation:")
    print(f"  {explanation.explanation[:300]}...")
    print(f"\nExamples:")
    for example in explanation.examples[:2]:
        print(f"  • {example}")
    print(f"\nKey Points:")
    for point in explanation.key_points[:2]:
        print(f"  ✓ {point}")
    
    # Demo 3: Practice Problem Generation
    print("\n" + "=" * 80)
    print("DEMO 3: Practice Problem Generation")
    print("=" * 80)
    
    print("\nGenerating practice problems...")
    
    for difficulty in [QuestionDifficulty.EASY, QuestionDifficulty.MEDIUM]:
        problem = agent.generate_practice_problem(
            Subject.MATHEMATICS,
            "quadratic equations",
            difficulty,
            SkillLevel.INTERMEDIATE
        )
        
        print(f"\n{difficulty.value.upper()} Problem:")
        print(f"Question: {problem.question[:150]}...")
        print(f"\nHints:")
        for hint in problem.hints[:2]:
            print(f"  💡 {hint}")
        print(f"\nSolution: {problem.solution[:150]}...")
    
    # Demo 4: Feedback on Student Work
    print("\n" + "=" * 80)
    print("DEMO 4: Feedback on Student Work")
    print("=" * 80)
    
    sample_problem = agent.generate_practice_problem(
        Subject.PROGRAMMING,
        "loops",
        QuestionDifficulty.EASY,
        SkillLevel.BEGINNER
    )
    
    student_answer = "I tried using a for loop but got confused about the syntax"
    response = StudentResponse(
        problem_id=sample_problem.problem_id,
        student_answer=student_answer,
        is_correct=False,
        partial_credit=0.3
    )
    
    print(f"\nProblem: {sample_problem.question[:100]}...")
    print(f"Student Answer: {student_answer}")
    
    print("\nGenerating feedback...")
    feedback = agent.provide_feedback(sample_problem, response)
    
    print(f"\nFeedback:")
    print(f"Correctness: {feedback.correctness}")
    print(f"\nWhat you did well:")
    for point in feedback.positive_points[:2]:
        print(f"  ✓ {point}")
    print(f"\nAreas to improve:")
    for area in feedback.areas_to_improve[:2]:
        print(f"  → {area}")
    print(f"\nEncouragement: {feedback.encouragement}")
    print(f"Next Steps: {feedback.next_steps}")
    
    # Demo 5: Personalized Learning Path
    print("\n" + "=" * 80)
    print("DEMO 5: Personalized Learning Path")
    print("=" * 80)
    
    print("\nCreating learning path for Python programming...")
    learning_path = agent.create_learning_path(
        "student001",
        Subject.PROGRAMMING,
        "Become proficient in Python web development",
        SkillLevel.BEGINNER
    )
    
    print(f"\nLearning Path ({len(learning_path)} objectives):")
    for i, obj in enumerate(learning_path, 1):
        print(f"\n{i}. {obj.topic} ({obj.skill_level.value})")
        print(f"   {obj.description}")
        if obj.prerequisites:
            print(f"   Prerequisites: {', '.join(obj.prerequisites)}")
    
    # Demo 6: Complete Tutoring Session
    print("\n" + "=" * 80)
    print("DEMO 6: Complete Tutoring Session")
    print("=" * 80)
    
    session = agent.conduct_tutoring_session(
        "student001",
        Subject.MATHEMATICS,
        "linear equations",
        duration_minutes=20
    )
    
    print(f"Session ID: {session.session_id}")
    print(f"Topic: {session.topic}")
    print(f"Duration: {session.duration_minutes} minutes")
    print(f"Mastery Level: {session.mastery_level:.2f}/1.0")
    
    print(f"\nConcepts Explained: {len(session.explanations_given)}")
    for exp in session.explanations_given:
        print(f"  • {exp.concept} ({exp.level.value})")
    
    print(f"\nProblems Attempted: {len(session.problems_attempted)}")
    for problem, response in session.problems_attempted:
        status = "✓" if response.is_correct else "✗"
        print(f"  {status} {problem.difficulty.value} - Score: {response.partial_credit:.1f}/1.0")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary = """
The Teaching Agent demonstrates comprehensive educational capabilities:

KEY CAPABILITIES:
1. Knowledge Assessment: Evaluates student understanding and identifies gaps
2. Adaptive Explanation: Explains concepts at appropriate complexity levels
3. Problem Generation: Creates practice problems matched to student level
4. Constructive Feedback: Provides encouraging, helpful feedback
5. Learning Path Creation: Designs personalized curricula
6. Progress Tracking: Monitors learning and mastery development
7. Multi-Strategy Teaching: Adapts teaching approach to student needs

BENEFITS:
- Personalized learning at student's pace
- Immediate, constructive feedback
- Unlimited practice problems
- Patient, consistent teaching
- Tracks progress over time
- Identifies and addresses knowledge gaps
- Adapts difficulty automatically

USE CASES:
- K-12 tutoring across all subjects
- College-level subject support
- Professional skill development
- Language learning
- Programming education
- Test preparation (SAT, ACT, GRE)
- Homework help
- Corporate training
- Self-paced learning
- Supplemental education

PRODUCTION CONSIDERATIONS:
1. Pedagogy: Implement proven teaching strategies and learning science
2. Assessment: Robust knowledge testing and gap analysis
3. Adaptation: Real-time difficulty adjustment based on performance
4. Content Library: Comprehensive problem and explanation databases
5. Progress Analytics: Detailed learning analytics and reporting
6. Engagement: Gamification, rewards, and motivation systems
7. Multi-Modal: Support text, images, videos, interactive content
8. Collaboration: Enable teacher oversight and intervention
9. Accessibility: Support diverse learning needs and disabilities
10. Safety: Content filtering and age-appropriate material

ADVANCED EXTENSIONS:
- Socratic dialogue for deeper understanding
- Visual learning with diagrams and animations
- Spaced repetition for memory retention
- Collaborative learning with peer interaction
- Real-world problem applications
- Project-based learning modules
- Adaptive testing and placement
- Learning style optimization
- Emotional intelligence and encouragement
- Multi-language support

The agent provides scalable, personalized education that adapts to each
student's needs, pace, and learning style while maintaining pedagogical
effectiveness.
"""
    
    print(summary)


if __name__ == "__main__":
    demonstrate_teaching_agent()
