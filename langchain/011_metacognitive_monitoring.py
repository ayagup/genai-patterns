"""
Pattern 011: Metacognitive Monitoring

Description:
    Agent monitors its own thinking process, confidence levels, and decision-making.
    Provides self-awareness about reasoning quality, uncertainty, and when to seek
    help or alternative strategies. Essential for safe and reliable AI systems.

Key Concepts:
    - Confidence Estimation: Quantify certainty in outputs
    - Uncertainty Quantification: Identify areas of low confidence
    - Self-Monitoring: Track reasoning quality during execution
    - Strategy Selection: Choose appropriate approaches based on confidence
    - Error Detection: Identify potential mistakes proactively

Use Cases:
    - Safety-critical applications
    - Decision validation and verification
    - Knowing when to escalate to humans
    - Quality control and error detection

LangChain Implementation:
    Confidence scoring chains with self-monitoring and adaptive strategies.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class MetacognitiveState:
    """Represents the agent's metacognitive state."""
    task: str
    current_step: str
    confidence: float = 0.5  # 0.0 to 1.0
    uncertainty_sources: List[str] = field(default_factory=list)
    alternative_strategies: List[str] = field(default_factory=list)
    should_seek_help: bool = False
    reasoning_quality: float = 0.5
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """Convert numeric confidence to level."""
        if self.confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class MetacognitiveAgent:
    """Agent with metacognitive monitoring capabilities."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo",
                 confidence_threshold: float = 0.6,
                 help_threshold: float = 0.4):
        """
        Initialize the Metacognitive Agent.
        
        Args:
            model_name: Name of the OpenAI model
            confidence_threshold: Minimum acceptable confidence
            help_threshold: Threshold below which to seek help
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.confidence_threshold = confidence_threshold
        self.help_threshold = help_threshold
        self.metacognitive_log: List[MetacognitiveState] = []
    
    def estimate_confidence(self, task: str, reasoning: str, answer: str) -> Tuple[float, List[str]]:
        """
        Estimate confidence in the answer and identify uncertainty sources.
        
        Args:
            task: The task being solved
            reasoning: The reasoning process
            answer: The proposed answer
            
        Returns:
            Tuple of (confidence_score, uncertainty_sources)
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Evaluate your confidence in this answer. Consider:
1. Completeness of reasoning
2. Availability of relevant information
3. Complexity of the task
4. Potential ambiguities or uncertainties

Provide:
CONFIDENCE: [0.0-1.0]
UNCERTAINTY_SOURCES:
- [source 1]
- [source 2]
...

Be honest about limitations and uncertainties."""),
            ("human", """Task: {task}

Reasoning:
{reasoning}

Answer: {answer}

Confidence Evaluation:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "task": task,
            "reasoning": reasoning,
            "answer": answer
        })
        
        # Parse confidence and uncertainty sources
        confidence = 0.5
        uncertainty_sources = []
        
        for line in response.split('\n'):
            line = line.strip()
            if line.upper().startswith('CONFIDENCE:'):
                try:
                    conf_str = line.split(':', 1)[1].strip()
                    confidence = float(conf_str)
                except:
                    pass
            elif line.startswith('-') and 'UNCERTAINTY_SOURCES' in response.upper():
                source = line.lstrip('- ').strip()
                if source:
                    uncertainty_sources.append(source)
        
        return confidence, uncertainty_sources
    
    def identify_alternative_strategies(self, task: str, current_approach: str,
                                       issues: List[str]) -> List[str]:
        """
        Identify alternative strategies when current approach has low confidence.
        
        Args:
            task: The task being solved
            current_approach: Current approach being used
            issues: Identified issues with current approach
            
        Returns:
            List of alternative strategies
        """
        issues_text = "\n".join([f"- {issue}" for issue in issues])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Suggest alternative strategies for solving this task given the identified issues.
Provide 2-3 concrete alternative approaches.

Format:
1. [Strategy 1]
2. [Strategy 2]
3. [Strategy 3]"""),
            ("human", """Task: {task}

Current Approach: {current_approach}

Issues:
{issues}

Alternative Strategies:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "task": task,
            "current_approach": current_approach,
            "issues": issues_text
        })
        
        # Parse strategies
        strategies = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                strategy = line.split('.', 1)[-1].strip().lstrip('- ').strip()
                if strategy:
                    strategies.append(strategy)
        
        return strategies[:3]
    
    def evaluate_reasoning_quality(self, reasoning: str) -> float:
        """
        Evaluate the quality of the reasoning process.
        
        Args:
            reasoning: The reasoning to evaluate
            
        Returns:
            Quality score 0.0-1.0
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Evaluate the quality of this reasoning. Consider:
- Logical consistency
- Completeness
- Clarity
- Rigor

Provide a quality score from 0.0 (poor) to 1.0 (excellent).
Respond with just the number."""),
            ("human", """Reasoning:
{reasoning}

Quality Score (0.0-1.0):""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            response = chain.invoke({"reasoning": reasoning})
            score = float(response.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    def solve_with_monitoring(self, task: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Solve a task with metacognitive monitoring.
        
        Args:
            task: The task to solve
            verbose: Whether to print monitoring information
            
        Returns:
            Dictionary with solution and metacognitive information
        """
        if verbose:
            print(f"\nTask: {task}\n")
            print("="*60)
            print("METACOGNITIVE MONITORING ACTIVE")
            print("="*60)
        
        # Step 1: Initial reasoning
        if verbose:
            print("\nStep 1: Initial Reasoning")
        
        reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", "Solve this task with clear step-by-step reasoning."),
            ("human", "{task}")
        ])
        
        chain = reasoning_prompt | self.llm | StrOutputParser()
        reasoning = chain.invoke({"task": task})
        
        # Extract answer
        answer_lines = reasoning.strip().split('\n')
        answer = answer_lines[-1] if answer_lines else reasoning[:100]
        
        if verbose:
            print(f"Reasoning: {reasoning[:200]}...")
            print(f"Answer: {answer}")
        
        # Step 2: Estimate confidence
        if verbose:
            print("\nStep 2: Confidence Estimation")
        
        confidence, uncertainty_sources = self.estimate_confidence(task, reasoning, answer)
        
        # Step 3: Evaluate reasoning quality
        reasoning_quality = self.evaluate_reasoning_quality(reasoning)
        
        if verbose:
            print(f"Confidence: {confidence:.2f} ({ConfidenceLevel(confidence >= 0.9 and 'very_high' or confidence >= 0.7 and 'high' or confidence >= 0.5 and 'medium' or confidence >= 0.3 and 'low' or 'very_low').value})")
            print(f"Reasoning Quality: {reasoning_quality:.2f}")
            if uncertainty_sources:
                print(f"Uncertainty Sources: {len(uncertainty_sources)}")
                for source in uncertainty_sources:
                    print(f"  - {source}")
        
        # Step 4: Determine if help needed
        should_seek_help = confidence < self.help_threshold
        
        # Step 5: Identify alternative strategies if confidence low
        alternative_strategies = []
        if confidence < self.confidence_threshold:
            if verbose:
                print(f"\n⚠️ Low confidence detected (threshold: {self.confidence_threshold})")
                print("Step 3: Identifying Alternative Strategies")
            
            alternative_strategies = self.identify_alternative_strategies(
                task, "Initial direct reasoning", uncertainty_sources
            )
            
            if verbose:
                print("Alternative Strategies:")
                for i, strategy in enumerate(alternative_strategies, 1):
                    print(f"  {i}. {strategy}")
        
        # Step 6: Decide on action
        if should_seek_help:
            action = "ESCALATE_TO_HUMAN"
            recommendation = "Task complexity or uncertainty requires human oversight"
        elif confidence < self.confidence_threshold:
            action = "TRY_ALTERNATIVE_STRATEGY"
            recommendation = f"Recommend trying: {alternative_strategies[0] if alternative_strategies else 'alternative approach'}"
        else:
            action = "PROCEED_WITH_ANSWER"
            recommendation = "Confidence sufficient to proceed"
        
        if verbose:
            print(f"\nStep 4: Decision")
            print(f"Action: {action}")
            print(f"Recommendation: {recommendation}")
        
        # Create metacognitive state
        state = MetacognitiveState(
            task=task,
            current_step="Solution",
            confidence=confidence,
            uncertainty_sources=uncertainty_sources,
            alternative_strategies=alternative_strategies,
            should_seek_help=should_seek_help,
            reasoning_quality=reasoning_quality
        )
        self.metacognitive_log.append(state)
        
        return {
            "task": task,
            "reasoning": reasoning,
            "answer": answer,
            "confidence": confidence,
            "confidence_level": state.get_confidence_level().value,
            "reasoning_quality": reasoning_quality,
            "uncertainty_sources": uncertainty_sources,
            "alternative_strategies": alternative_strategies,
            "should_seek_help": should_seek_help,
            "recommended_action": action,
            "recommendation": recommendation
        }


def demonstrate_metacognitive_monitoring():
    """Demonstrates the Metacognitive Monitoring pattern."""
    
    print("=" * 80)
    print("PATTERN 011: Metacognitive Monitoring")
    print("=" * 80)
    print()
    print("Metacognitive Monitoring enables self-awareness through:")
    print("1. Confidence Estimation: Quantify certainty in outputs")
    print("2. Uncertainty Detection: Identify sources of uncertainty")
    print("3. Quality Assessment: Evaluate reasoning quality")
    print("4. Strategy Selection: Choose alternative approaches when needed")
    print("5. Help-Seeking: Know when to escalate to humans")
    print()
    
    # Create agent
    agent = MetacognitiveAgent(
        confidence_threshold=0.6,
        help_threshold=0.4
    )
    
    # Test tasks with varying complexity
    tasks = [
        "What is 15 + 27?",  # High confidence expected
        "Explain the implications of quantum entanglement for future communication technologies",  # Medium confidence
        "Should Company X acquire Company Y given uncertain market conditions?"  # Low confidence
    ]
    
    for idx, task in enumerate(tasks, 1):
        print(f"\n{'='*80}")
        print(f"Example {idx}")
        print('='*80)
        
        try:
            result = agent.solve_with_monitoring(task, verbose=True)
            
            print(f"\n\n{'='*80}")
            print("METACOGNITIVE SUMMARY")
            print('='*80)
            print(f"\nTask: {result['task']}")
            print(f"Answer: {result['answer']}")
            print(f"\nConfidence: {result['confidence']:.2f} ({result['confidence_level']})")
            print(f"Reasoning Quality: {result['reasoning_quality']:.2f}")
            print(f"Should Seek Help: {'Yes' if result['should_seek_help'] else 'No'}")
            print(f"\nRecommended Action: {result['recommended_action']}")
            print(f"Recommendation: {result['recommendation']}")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n\n" + "=" * 80)
    print("METACOGNITIVE MONITORING PATTERN DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("1. Confidence Estimation: Quantified certainty levels")
    print("2. Uncertainty Sources: Identified specific areas of uncertainty")
    print("3. Quality Assessment: Evaluated reasoning quality")
    print("4. Alternative Strategies: Suggested alternatives for low confidence")
    print("5. Decision Making: Determined appropriate actions based on confidence")
    print()
    print("Advantages:")
    print("- Self-awareness of limitations")
    print("- Proactive error detection")
    print("- Intelligent help-seeking")
    print("- Adaptive strategy selection")
    print()
    print("When to use Metacognitive Monitoring:")
    print("- Safety-critical applications")
    print("- High-stakes decision-making")
    print("- Uncertain or ambiguous tasks")
    print("- When transparency is required")
    print()
    print("LangChain Components Used:")
    print("- ChatPromptTemplate: Confidence estimation and evaluation")
    print("- StrOutputParser: Parse monitoring outputs")
    print("- Threshold-based decision making")
    print("- Structured metacognitive state tracking")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set it in your .env file or environment.")
        exit(1)
    
    demonstrate_metacognitive_monitoring()
