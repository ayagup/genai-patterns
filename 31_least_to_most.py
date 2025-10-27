"""
Least-to-Most Prompting Pattern Implementation

This module demonstrates the least-to-most prompting approach where complex problems
are solved by first tackling easier sub-problems and gradually building up to
harder ones. This creates a foundation of understanding that enables solving
the more complex aspects.

Key Components:
- Problem decomposition from complex to simple
- Sequential solving with context building
- Progressive difficulty ramping
- Knowledge accumulation and transfer
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import random
import math


class DifficultyLevel(Enum):
    """Difficulty levels for problem decomposition"""
    TRIVIAL = "trivial"      # Very basic, fundamental concepts
    SIMPLE = "simple"        # Single-step, straightforward
    MODERATE = "moderate"    # Multi-step, some complexity
    COMPLEX = "complex"      # Multiple concepts, reasoning required
    ADVANCED = "advanced"    # High-level integration, expert knowledge


@dataclass
class SubProblem:
    """Represents a sub-problem in the decomposition"""
    id: str
    description: str
    difficulty: DifficultyLevel
    prerequisites: List[str] = field(default_factory=list)
    concepts_taught: List[str] = field(default_factory=list)
    solution: str = ""
    explanation: str = ""
    confidence: float = 0.0
    solved: bool = False
    attempts: int = 0
    
    def __post_init__(self):
        if not self.id:
            self.id = f"subproblem_{random.randint(1000, 9999)}"


@dataclass
class LearningContext:
    """Maintains context and knowledge accumulated during problem solving"""
    learned_concepts: List[str] = field(default_factory=list)
    solved_problems: List[str] = field(default_factory=list)
    difficulty_progression: List[DifficultyLevel] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    knowledge_base: Dict[str, str] = field(default_factory=dict)
    
    def add_concept(self, concept: str, definition: str):
        """Add a newly learned concept"""
        if concept not in self.learned_concepts:
            self.learned_concepts.append(concept)
            self.knowledge_base[concept] = definition
    
    def has_prerequisite(self, concept: str) -> bool:
        """Check if a prerequisite concept has been learned"""
        return concept in self.learned_concepts
    
    def get_average_confidence(self) -> float:
        """Get average confidence across solved problems"""
        return sum(self.confidence_history) / len(self.confidence_history) if self.confidence_history else 0.0


class ProblemDecomposer:
    """Decomposes complex problems into simpler sub-problems"""
    
    def __init__(self):
        self.decomposition_strategies = {
            "mathematical": self._decompose_mathematical,
            "logical": self._decompose_logical,
            "procedural": self._decompose_procedural,
            "conceptual": self._decompose_conceptual
        }
    
    def decompose(self, problem: str, problem_type: str = "general") -> List[SubProblem]:
        """Decompose a problem into sub-problems of increasing difficulty"""
        if problem_type in self.decomposition_strategies:
            return self.decomposition_strategies[problem_type](problem)
        else:
            return self._decompose_general(problem)
    
    def _decompose_mathematical(self, problem: str) -> List[SubProblem]:
        """Decompose mathematical problems"""
        sub_problems = []
        
        # Example: Complex equation solving
        if "equation" in problem.lower() or "solve" in problem.lower():
            sub_problems = [
                SubProblem(
                    id="math_1",
                    description="Understand basic arithmetic operations (+, -, ×, ÷)",
                    difficulty=DifficultyLevel.TRIVIAL,
                    concepts_taught=["arithmetic", "operations"],
                    solution="Basic operations: addition, subtraction, multiplication, division",
                    explanation="These are fundamental mathematical operations"
                ),
                SubProblem(
                    id="math_2", 
                    description="Learn about variables and constants",
                    difficulty=DifficultyLevel.SIMPLE,
                    prerequisites=["arithmetic"],
                    concepts_taught=["variables", "constants"],
                    solution="Variables (x, y) represent unknown values; constants are fixed numbers",
                    explanation="Variables allow us to represent unknown quantities"
                ),
                SubProblem(
                    id="math_3",
                    description="Understand simple linear equations (ax + b = c)",
                    difficulty=DifficultyLevel.MODERATE,
                    prerequisites=["variables", "arithmetic"],
                    concepts_taught=["linear_equations", "isolation"],
                    solution="Isolate variable by performing same operations on both sides",
                    explanation="Linear equations have variables to the first power only"
                ),
                SubProblem(
                    id="math_4",
                    description="Solve systems of equations",
                    difficulty=DifficultyLevel.COMPLEX,
                    prerequisites=["linear_equations"],
                    concepts_taught=["systems", "substitution", "elimination"],
                    solution="Use substitution or elimination methods",
                    explanation="Systems involve multiple equations with multiple variables"
                ),
                SubProblem(
                    id="math_5",
                    description="Apply to complex word problems",
                    difficulty=DifficultyLevel.ADVANCED,
                    prerequisites=["systems", "isolation"],
                    concepts_taught=["modeling", "interpretation"],
                    solution="Translate real-world scenarios into mathematical equations",
                    explanation="Word problems require translating language into mathematics"
                )
            ]
        
        return sub_problems
    
    def _decompose_logical(self, problem: str) -> List[SubProblem]:
        """Decompose logical reasoning problems"""
        return [
            SubProblem(
                id="logic_1",
                description="Understand basic logical statements (true/false)",
                difficulty=DifficultyLevel.TRIVIAL,
                concepts_taught=["truth_values", "statements"],
                solution="Statements are either true or false",
                explanation="Foundation of logical reasoning"
            ),
            SubProblem(
                id="logic_2",
                description="Learn logical operators (AND, OR, NOT)",
                difficulty=DifficultyLevel.SIMPLE,
                prerequisites=["truth_values"],
                concepts_taught=["operators", "conjunction", "disjunction"],
                solution="AND: both must be true; OR: at least one true; NOT: opposite",
                explanation="These operators combine logical statements"
            ),
            SubProblem(
                id="logic_3",
                description="Understand conditional statements (if-then)",
                difficulty=DifficultyLevel.MODERATE,
                prerequisites=["operators"],
                concepts_taught=["conditionals", "implications"],
                solution="If P then Q: Q must be true when P is true",
                explanation="Conditionals express dependencies between statements"
            ),
            SubProblem(
                id="logic_4",
                description="Apply logical reasoning to arguments",
                difficulty=DifficultyLevel.COMPLEX,
                prerequisites=["conditionals"],
                concepts_taught=["validity", "soundness"],
                solution="Check if conclusions follow logically from premises",
                explanation="Valid arguments have conclusions that must follow from premises"
            )
        ]
    
    def _decompose_procedural(self, problem: str) -> List[SubProblem]:
        """Decompose procedural/algorithmic problems"""
        return [
            SubProblem(
                id="proc_1",
                description="Identify the goal and constraints",
                difficulty=DifficultyLevel.TRIVIAL,
                concepts_taught=["goals", "constraints"],
                solution="Clearly state what needs to be accomplished and any limitations",
                explanation="Understanding the problem is the first step"
            ),
            SubProblem(
                id="proc_2",
                description="Break down into sequential steps",
                difficulty=DifficultyLevel.SIMPLE,
                prerequisites=["goals"],
                concepts_taught=["sequencing", "steps"],
                solution="List actions in the order they need to be performed",
                explanation="Most procedures are sequences of actions"
            ),
            SubProblem(
                id="proc_3",
                description="Handle decision points and branching",
                difficulty=DifficultyLevel.MODERATE,
                prerequisites=["sequencing"],
                concepts_taught=["decisions", "branching"],
                solution="Use if-then logic for different scenarios",
                explanation="Procedures often require different paths based on conditions"
            ),
            SubProblem(
                id="proc_4",
                description="Optimize and handle edge cases",
                difficulty=DifficultyLevel.COMPLEX,
                prerequisites=["decisions"],
                concepts_taught=["optimization", "edge_cases"],
                solution="Consider unusual inputs and improve efficiency",
                explanation="Robust procedures handle all possible scenarios"
            )
        ]
    
    def _decompose_conceptual(self, problem: str) -> List[SubProblem]:
        """Decompose conceptual understanding problems"""
        return [
            SubProblem(
                id="concept_1",
                description="Define key terms and vocabulary",
                difficulty=DifficultyLevel.TRIVIAL,
                concepts_taught=["terminology", "definitions"],
                solution="Learn precise meanings of important terms",
                explanation="Shared vocabulary is essential for understanding"
            ),
            SubProblem(
                id="concept_2",
                description="Understand relationships between concepts",
                difficulty=DifficultyLevel.SIMPLE,
                prerequisites=["terminology"],
                concepts_taught=["relationships", "connections"],
                solution="Map how concepts relate to and influence each other",
                explanation="Concepts rarely exist in isolation"
            ),
            SubProblem(
                id="concept_3",
                description="Apply concepts to simple examples",
                difficulty=DifficultyLevel.MODERATE,
                prerequisites=["relationships"],
                concepts_taught=["application", "examples"],
                solution="Use concrete examples to illustrate abstract concepts",
                explanation="Examples make abstract ideas concrete"
            ),
            SubProblem(
                id="concept_4",
                description="Synthesize into comprehensive understanding",
                difficulty=DifficultyLevel.COMPLEX,
                prerequisites=["application"],
                concepts_taught=["synthesis", "integration"],
                solution="Combine all concepts into unified understanding",
                explanation="Deep understanding integrates all components"
            )
        ]
    
    def _decompose_general(self, problem: str) -> List[SubProblem]:
        """Generic decomposition for unspecified problem types"""
        return [
            SubProblem(
                id="gen_1",
                description="Understand the problem statement",
                difficulty=DifficultyLevel.TRIVIAL,
                concepts_taught=["comprehension"],
                solution="Read carefully and identify what is being asked",
                explanation="Problem comprehension is the foundation"
            ),
            SubProblem(
                id="gen_2",
                description="Identify relevant information and concepts",
                difficulty=DifficultyLevel.SIMPLE,
                prerequisites=["comprehension"],
                concepts_taught=["analysis", "relevance"],
                solution="Separate relevant from irrelevant information",
                explanation="Focus on what matters for the solution"
            ),
            SubProblem(
                id="gen_3",
                description="Develop a solution approach",
                difficulty=DifficultyLevel.MODERATE,
                prerequisites=["analysis"],
                concepts_taught=["strategy", "approach"],
                solution="Choose appropriate methods and create a plan",
                explanation="Strategic thinking guides problem solving"
            ),
            SubProblem(
                id="gen_4",
                description="Implement and verify the solution",
                difficulty=DifficultyLevel.COMPLEX,
                prerequisites=["strategy"],
                concepts_taught=["implementation", "verification"],
                solution="Execute the plan and check the results",
                explanation="Solutions must be implemented and validated"
            )
        ]


class ProgressiveTeacher:
    """Teaches concepts progressively from simple to complex"""
    
    def __init__(self):
        self.teaching_strategies = {
            DifficultyLevel.TRIVIAL: self._teach_trivial,
            DifficultyLevel.SIMPLE: self._teach_simple,
            DifficultyLevel.MODERATE: self._teach_moderate,
            DifficultyLevel.COMPLEX: self._teach_complex,
            DifficultyLevel.ADVANCED: self._teach_advanced
        }
    
    def teach_concept(self, sub_problem: SubProblem, context: LearningContext) -> Tuple[bool, str, float]:
        """Teach a concept at the appropriate difficulty level"""
        # Check prerequisites
        for prereq in sub_problem.prerequisites:
            if not context.has_prerequisite(prereq):
                return False, f"Missing prerequisite: {prereq}", 0.0
        
        # Apply appropriate teaching strategy
        if sub_problem.difficulty in self.teaching_strategies:
            success, explanation, confidence = self.teaching_strategies[sub_problem.difficulty](
                sub_problem, context
            )
        else:
            success, explanation, confidence = self._teach_simple(sub_problem, context)
        
        # Update context if successful
        if success:
            for concept in sub_problem.concepts_taught:
                context.add_concept(concept, sub_problem.explanation)
            context.solved_problems.append(sub_problem.id)
            context.difficulty_progression.append(sub_problem.difficulty)
            context.confidence_history.append(confidence)
        
        return success, explanation, confidence
    
    def _teach_trivial(self, sub_problem: SubProblem, context: LearningContext) -> Tuple[bool, str, float]:
        """Teach trivial concepts with simple exposition"""
        explanation = f"📚 Basic Concept: {sub_problem.description}\n"
        explanation += f"💡 Key Idea: {sub_problem.solution}\n"
        explanation += f"🔍 Why Important: {sub_problem.explanation}"
        
        # High confidence for trivial concepts
        confidence = 0.9 + random.uniform(-0.1, 0.1)
        return True, explanation, confidence
    
    def _teach_simple(self, sub_problem: SubProblem, context: LearningContext) -> Tuple[bool, str, float]:
        """Teach simple concepts with examples"""
        explanation = f"📖 Learning: {sub_problem.description}\n"
        explanation += f"✅ Solution: {sub_problem.solution}\n"
        explanation += f"📝 Explanation: {sub_problem.explanation}\n"
        
        # Reference previous concepts
        if sub_problem.prerequisites:
            explanation += f"🔗 Building on: {', '.join(sub_problem.prerequisites)}"
        
        confidence = 0.8 + random.uniform(-0.15, 0.15)
        return True, explanation, confidence
    
    def _teach_moderate(self, sub_problem: SubProblem, context: LearningContext) -> Tuple[bool, str, float]:
        """Teach moderate concepts with detailed examples"""
        explanation = f"🎯 Moderate Challenge: {sub_problem.description}\n"
        explanation += f"🛠️ Approach: {sub_problem.solution}\n"
        explanation += f"📖 Understanding: {sub_problem.explanation}\n"
        
        # Show connection to learned concepts
        relevant_concepts = [c for c in context.learned_concepts if c in sub_problem.prerequisites]
        if relevant_concepts:
            explanation += f"🧠 Using what we learned: {', '.join(relevant_concepts)}\n"
        
        explanation += "💪 This builds your problem-solving skills!"
        
        confidence = 0.7 + random.uniform(-0.2, 0.2)
        return True, explanation, confidence
    
    def _teach_complex(self, sub_problem: SubProblem, context: LearningContext) -> Tuple[bool, str, float]:
        """Teach complex concepts with comprehensive analysis"""
        explanation = f"🧩 Complex Problem: {sub_problem.description}\n"
        explanation += f"🎯 Strategy: {sub_problem.solution}\n"
        explanation += f"🔬 Deep Dive: {sub_problem.explanation}\n"
        
        # Show the learning journey
        if context.difficulty_progression:
            explanation += f"📈 Your Progress: {' → '.join([d.value for d in context.difficulty_progression])}\n"
        
        # Calculate confidence based on foundation strength
        foundation_strength = len([c for c in context.learned_concepts if c in sub_problem.prerequisites]) / max(len(sub_problem.prerequisites), 1)
        base_confidence = 0.6 + (foundation_strength * 0.2)
        confidence = base_confidence + random.uniform(-0.15, 0.15)
        
        explanation += f"🎪 This integrates {len(sub_problem.prerequisites)} concepts you've mastered!"
        
        return True, explanation, confidence
    
    def _teach_advanced(self, sub_problem: SubProblem, context: LearningContext) -> Tuple[bool, str, float]:
        """Teach advanced concepts with synthesis and application"""
        explanation = f"🚀 Advanced Application: {sub_problem.description}\n"
        explanation += f"🏆 Expert Approach: {sub_problem.solution}\n"
        explanation += f"🎓 Mastery Level: {sub_problem.explanation}\n"
        
        # Show comprehensive understanding
        avg_confidence = context.get_average_confidence()
        explanation += f"📊 Your Learning Journey: {len(context.solved_problems)} problems solved\n"
        explanation += f"💪 Average Confidence: {avg_confidence:.1%}\n"
        
        # Advanced problems require strong foundation
        foundation_score = avg_confidence * len(context.learned_concepts) / 10
        confidence = min(0.5 + foundation_score, 0.9) + random.uniform(-0.1, 0.1)
        
        explanation += "🎉 You're now applying expert-level thinking!"
        
        return True, explanation, confidence


class LeastToMostAgent:
    """Main agent implementing least-to-most prompting"""
    
    def __init__(self):
        self.decomposer = ProblemDecomposer()
        self.teacher = ProgressiveTeacher()
        self.context = LearningContext()
        self.learning_history: List[Dict[str, Any]] = []
    
    def solve_problem(self, problem: str, problem_type: str = "general") -> Dict[str, Any]:
        """Solve a problem using least-to-most approach"""
        print(f"\n🎯 Least-to-Most Problem Solving")
        print("=" * 60)
        print(f"Problem: {problem}")
        print(f"Type: {problem_type}")
        
        # Reset context for new problem
        self.context = LearningContext()
        
        # Decompose problem
        print(f"\n📋 Decomposing problem into sub-problems...")
        sub_problems = self.decomposer.decompose(problem, problem_type)
        
        if not sub_problems:
            return {"error": "Could not decompose problem"}
        
        print(f"Generated {len(sub_problems)} sub-problems:")
        for i, sp in enumerate(sub_problems, 1):
            print(f"  {i}. [{sp.difficulty.value}] {sp.description}")
        
        # Solve sub-problems in order of increasing difficulty
        results = []
        
        for i, sub_problem in enumerate(sub_problems, 1):
            print(f"\n📚 Step {i}: {sub_problem.description}")
            print("-" * 40)
            
            success, explanation, confidence = self.teacher.teach_concept(sub_problem, self.context)
            
            sub_problem.solved = success
            sub_problem.confidence = confidence
            sub_problem.attempts = 1
            
            result = {
                "step": i,
                "sub_problem": sub_problem.description,
                "difficulty": sub_problem.difficulty.value,
                "success": success,
                "confidence": confidence,
                "explanation": explanation,
                "concepts_learned": sub_problem.concepts_taught,
                "prerequisites": sub_problem.prerequisites
            }
            results.append(result)
            
            print(explanation)
            print(f"✅ Success: {success} | Confidence: {confidence:.1%}")
            
            if not success:
                print("❌ Failed to learn this step. Cannot proceed to more complex concepts.")
                break
        
        # Generate final summary
        final_result = {
            "original_problem": problem,
            "problem_type": problem_type,
            "total_steps": len(sub_problems),
            "completed_steps": len([r for r in results if r["success"]]),
            "overall_success": all(r["success"] for r in results),
            "average_confidence": sum(r["confidence"] for r in results) / len(results) if results else 0,
            "concepts_mastered": len(self.context.learned_concepts),
            "learning_progression": [r["difficulty"] for r in results],
            "step_results": results,
            "knowledge_base": dict(self.context.knowledge_base)
        }
        
        # Store in history
        self.learning_history.append(final_result)
        
        print(f"\n📈 Final Summary:")
        print(f"Completed: {final_result['completed_steps']}/{final_result['total_steps']} steps")
        print(f"Overall Success: {final_result['overall_success']}")
        print(f"Average Confidence: {final_result['average_confidence']:.1%}")
        print(f"Concepts Mastered: {final_result['concepts_mastered']}")
        print(f"Learning Progression: {' → '.join(final_result['learning_progression'])}")
        
        return final_result
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of all learning sessions"""
        if not self.learning_history:
            return {"message": "No learning sessions completed"}
        
        total_problems = len(self.learning_history)
        successful_problems = len([h for h in self.learning_history if h["overall_success"]])
        
        all_concepts = set()
        all_difficulties = []
        all_confidences = []
        
        for history in self.learning_history:
            all_concepts.update(history["knowledge_base"].keys())
            all_difficulties.extend(history["learning_progression"])
            all_confidences.append(history["average_confidence"])
        
        difficulty_counts = {}
        for diff in all_difficulties:
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        
        return {
            "total_problems_attempted": total_problems,
            "successful_problems": successful_problems,
            "success_rate": successful_problems / total_problems,
            "unique_concepts_learned": len(all_concepts),
            "average_session_confidence": sum(all_confidences) / len(all_confidences),
            "difficulty_distribution": difficulty_counts,
            "concepts_mastered": list(all_concepts)
        }


def main():
    """Demonstration of the Least-to-Most Prompting pattern"""
    print("📚 Least-to-Most Prompting Pattern Demonstration")
    print("=" * 80)
    print("This demonstrates progressive learning from simple to complex:")
    print("- Problem decomposition by difficulty")
    print("- Sequential learning with prerequisite checking")
    print("- Context building and knowledge accumulation")
    print("- Confidence tracking and adaptive teaching")
    
    # Create agent
    agent = LeastToMostAgent()
    
    # Test problems of different types
    test_problems = [
        ("Solve the system of equations: 2x + 3y = 12 and x - y = 1", "mathematical"),
        ("Determine if the argument 'All birds fly. Penguins are birds. Therefore penguins fly.' is valid", "logical"),
        ("Design an algorithm to sort a list of numbers", "procedural"),
        ("Explain how photosynthesis works in plants", "conceptual")
    ]
    
    for i, (problem, problem_type) in enumerate(test_problems, 1):
        print(f"\n\n🔍 Test Case {i}")
        print("=" * 80)
        
        result = agent.solve_problem(problem, problem_type)
        
        print(f"\n📊 Detailed Results:")
        print(f"Problem Type: {result['problem_type']}")
        print(f"Steps Completed: {result['completed_steps']}/{result['total_steps']}")
        print(f"Success Rate: {result['overall_success']}")
        print(f"Concepts Learned: {result['concepts_mastered']}")
        
        if result['step_results']:
            print(f"\n📝 Learning Steps:")
            for step_result in result['step_results']:
                emoji = "✅" if step_result['success'] else "❌"
                print(f"  {emoji} Step {step_result['step']}: {step_result['sub_problem']}")
                print(f"     Difficulty: {step_result['difficulty']}, Confidence: {step_result['confidence']:.1%}")
                if step_result['concepts_learned']:
                    print(f"     Learned: {', '.join(step_result['concepts_learned'])}")
    
    # Overall learning summary
    print(f"\n\n📈 Overall Learning Summary")
    print("=" * 80)
    
    summary = agent.get_learning_summary()
    for key, value in summary.items():
        if key != "concepts_mastered":  # Skip the long list
            print(f"{key}: {value}")
    
    print(f"\n🧠 Key Concepts Mastered:")
    for concept in summary.get("concepts_mastered", [])[:10]:  # Show first 10
        print(f"  • {concept}")
    if len(summary.get("concepts_mastered", [])) > 10:
        print(f"  ... and {len(summary['concepts_mastered']) - 10} more")
    
    print("\n\n🎯 Key Least-to-Most Features Demonstrated:")
    print("✅ Progressive difficulty decomposition")
    print("✅ Prerequisite checking and enforcement")
    print("✅ Context building and knowledge accumulation")
    print("✅ Adaptive confidence tracking")
    print("✅ Multiple problem type support")
    print("✅ Learning history and progress tracking")
    print("✅ Foundation-based teaching strategies")
    print("✅ Incremental concept mastery")


if __name__ == "__main__":
    main()