"""
Pattern 034: Active Learning

Description:
    Active Learning enables agents to strategically request human input for the most
    informative or uncertain cases. Rather than passively learning from all data,
    the agent actively selects examples that will maximize learning efficiency,
    focusing human effort on cases where it matters most.

Components:
    - Uncertainty Estimator: Measures confidence in predictions
    - Query Strategy: Selects most informative examples
    - Oracle: Human expert providing labels/feedback
    - Learning Loop: Incorporates feedback to improve
    - Sample Selector: Chooses examples for annotation

Use Cases:
    - Training data collection with limited budget
    - Model improvement in production
    - Handling ambiguous or edge cases
    - Domain adaptation
    - Quality assurance and verification

LangChain Implementation:
    Uses confidence estimation and strategic sampling to identify cases where
    human feedback will be most valuable, optimizing the learning process.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import random
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class QueryStrategy(Enum):
    """Strategies for selecting samples for human annotation."""
    UNCERTAINTY_SAMPLING = "uncertainty_sampling"  # Least confident examples
    MARGIN_SAMPLING = "margin_sampling"  # Smallest margin between top predictions
    ENTROPY_SAMPLING = "entropy_sampling"  # Highest entropy
    QUERY_BY_COMMITTEE = "query_by_committee"  # Maximum disagreement
    DIVERSITY_SAMPLING = "diversity_sampling"  # Representative diversity
    EXPECTED_ERROR_REDUCTION = "expected_error_reduction"  # Max expected improvement


@dataclass
class Prediction:
    """A prediction with confidence information."""
    input: str
    output: str
    confidence: float
    alternatives: List[Tuple[str, float]] = field(default_factory=list)  # (output, confidence)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnnotationRequest:
    """Request for human annotation."""
    id: str
    input: str
    predicted_output: str
    confidence: float
    query_strategy: QueryStrategy
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Human response
    human_label: Optional[str] = None
    feedback: Optional[str] = None
    annotation_time: Optional[datetime] = None


@dataclass
class TrainingExample:
    """Labeled training example."""
    input: str
    output: str
    source: str  # "human", "synthetic", "model"
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


class Oracle:
    """
    Simulates human expert for annotation (in production, would be real human).
    """
    
    def __init__(self, auto_annotate: bool = True):
        self.auto_annotate = auto_annotate
        self.annotation_count = 0
    
    def annotate(self, request: AnnotationRequest) -> AnnotationRequest:
        """
        Provide annotation for the request.
        
        In simulation mode, provides realistic labels.
        In production, would wait for human annotation.
        """
        self.annotation_count += 1
        
        if self.auto_annotate:
            # Simulate human annotation
            # In practice, this would involve actual human judgment
            request.human_label = f"[HUMAN ANNOTATED] {request.predicted_output}"
            request.feedback = "Annotation provided"
            request.annotation_time = datetime.now()
        
        return request


class ActiveLearningAgent:
    """
    Agent that uses active learning to improve through strategic human feedback.
    
    Features:
    - Multiple query strategies
    - Confidence estimation
    - Strategic sample selection
    - Learning from feedback
    """
    
    def __init__(
        self,
        query_strategy: QueryStrategy = QueryStrategy.UNCERTAINTY_SAMPLING,
        oracle: Optional[Oracle] = None,
        temperature: float = 0.7
    ):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        self.query_strategy = query_strategy
        self.oracle = oracle or Oracle()
        
        # Training data
        self.training_examples: List[TrainingExample] = []
        self.unlabeled_pool: List[str] = []
        self.annotation_history: List[AnnotationRequest] = []
        
        self._next_request_id = 1
        
        # Prompt for making predictions with confidence
        self.prediction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that provides predictions with confidence scores.

Previous examples (for reference):
{examples}

Task: {input}

Provide your answer and estimate your confidence (0.0-1.0).

Format:
ANSWER: [your answer]
CONFIDENCE: [0.0-1.0]
ALTERNATIVES: [alternative answer 1] (confidence), [alternative answer 2] (confidence)"""),
            ("user", "{input}")
        ])
    
    def predict_with_confidence(self, input_text: str) -> Prediction:
        """
        Make a prediction with confidence estimation.
        """
        # Format examples from training data
        examples_text = self._format_examples(self.training_examples[:5])
        
        # Generate prediction
        chain = self.prediction_prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "input": input_text,
            "examples": examples_text
        })
        
        # Parse result
        answer, confidence, alternatives = self._parse_prediction(result)
        
        return Prediction(
            input=input_text,
            output=answer,
            confidence=confidence,
            alternatives=alternatives
        )
    
    def select_for_annotation(
        self,
        predictions: List[Prediction],
        budget: int = 5
    ) -> List[Prediction]:
        """
        Select most informative examples for human annotation.
        
        Args:
            predictions: List of predictions to choose from
            budget: Number of examples to select
        
        Returns:
            Selected predictions for annotation
        """
        if not predictions:
            return []
        
        # Apply query strategy
        if self.query_strategy == QueryStrategy.UNCERTAINTY_SAMPLING:
            # Select examples with lowest confidence
            selected = sorted(predictions, key=lambda p: p.confidence)[:budget]
        
        elif self.query_strategy == QueryStrategy.MARGIN_SAMPLING:
            # Select examples with smallest margin between top 2 predictions
            selected = []
            for pred in predictions:
                if len(pred.alternatives) >= 2:
                    margin = abs(pred.alternatives[0][1] - pred.alternatives[1][1])
                    selected.append((margin, pred))
                else:
                    selected.append((0.0, pred))
            
            selected = [p for _, p in sorted(selected, key=lambda x: x[0])[:budget]]
        
        elif self.query_strategy == QueryStrategy.DIVERSITY_SAMPLING:
            # Select diverse examples (simple: random sampling)
            selected = random.sample(predictions, min(budget, len(predictions)))
        
        else:
            # Default: uncertainty sampling
            selected = sorted(predictions, key=lambda p: p.confidence)[:budget]
        
        return selected
    
    def request_annotations(
        self,
        predictions: List[Prediction]
    ) -> List[AnnotationRequest]:
        """
        Request human annotations for selected predictions.
        """
        requests = []
        
        for pred in predictions:
            request = AnnotationRequest(
                id=f"ann_{self._next_request_id:04d}",
                input=pred.input,
                predicted_output=pred.output,
                confidence=pred.confidence,
                query_strategy=self.query_strategy
            )
            self._next_request_id += 1
            
            # Get annotation from oracle
            request = self.oracle.annotate(request)
            
            requests.append(request)
            self.annotation_history.append(request)
        
        return requests
    
    def incorporate_feedback(
        self,
        annotations: List[AnnotationRequest]
    ) -> int:
        """
        Incorporate human annotations into training data.
        
        Returns number of examples added.
        """
        added = 0
        
        for annotation in annotations:
            if annotation.human_label:
                example = TrainingExample(
                    input=annotation.input,
                    output=annotation.human_label,
                    source="human",
                    confidence=1.0
                )
                self.training_examples.append(example)
                added += 1
        
        return added
    
    def active_learning_cycle(
        self,
        unlabeled_data: List[str],
        annotation_budget: int = 5
    ) -> Dict[str, Any]:
        """
        Perform one cycle of active learning.
        
        Steps:
        1. Make predictions on unlabeled data
        2. Select most informative examples
        3. Request human annotations
        4. Incorporate feedback into training data
        
        Returns:
            Statistics about the cycle
        """
        # Step 1: Predict on unlabeled data
        predictions = []
        for input_text in unlabeled_data:
            pred = self.predict_with_confidence(input_text)
            predictions.append(pred)
        
        # Step 2: Select for annotation
        selected = self.select_for_annotation(predictions, annotation_budget)
        
        # Step 3: Request annotations
        annotations = self.request_annotations(selected)
        
        # Step 4: Incorporate feedback
        added = self.incorporate_feedback(annotations)
        
        return {
            "cycle_complete": True,
            "predictions_made": len(predictions),
            "selected_for_annotation": len(selected),
            "annotations_received": len(annotations),
            "examples_added": added,
            "total_training_examples": len(self.training_examples),
            "avg_confidence": sum(p.confidence for p in predictions) / len(predictions) if predictions else 0
        }
    
    def _format_examples(self, examples: List[TrainingExample]) -> str:
        """Format training examples for prompt."""
        if not examples:
            return "No examples yet."
        
        lines = []
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"  Input: {ex.input}")
            lines.append(f"  Output: {ex.output}")
        
        return "\n".join(lines)
    
    def _parse_prediction(self, result: str) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Parse prediction result."""
        lines = result.strip().split('\n')
        
        answer = ""
        confidence = 0.5
        alternatives = []
        
        for line in lines:
            if line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.replace("CONFIDENCE:", "").strip()
                    confidence = float(conf_str)
                except ValueError:
                    pass
            elif line.startswith("ALTERNATIVES:"):
                # Parse alternatives (simple parsing)
                alt_str = line.replace("ALTERNATIVES:", "").strip()
                # This is simplified - in practice, would need better parsing
        
        if not answer:
            answer = result
        
        return answer, confidence, alternatives
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get active learning statistics."""
        if not self.annotation_history:
            return {
                "total_annotations": 0,
                "training_examples": len(self.training_examples),
                "avg_confidence_at_annotation": 0.0
            }
        
        avg_conf = sum(a.confidence for a in self.annotation_history) / len(self.annotation_history)
        
        return {
            "total_annotations": len(self.annotation_history),
            "training_examples": len(self.training_examples),
            "avg_confidence_at_annotation": avg_conf,
            "oracle_calls": self.oracle.annotation_count,
            "query_strategy": self.query_strategy.value
        }


def demonstrate_active_learning():
    """
    Demonstrates active learning with strategic sample selection
    and human feedback integration.
    """
    print("=" * 80)
    print("ACTIVE LEARNING DEMONSTRATION")
    print("=" * 80)
    
    # Create oracle (simulated human expert)
    oracle = Oracle(auto_annotate=True)
    
    # Test 1: Uncertainty Sampling
    print("\n" + "=" * 80)
    print("Test 1: Uncertainty Sampling Strategy")
    print("=" * 80)
    
    agent1 = ActiveLearningAgent(
        query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
        oracle=oracle
    )
    
    # Simulated unlabeled data
    unlabeled_data = [
        "What is machine learning?",
        "Explain neural networks",
        "How does backpropagation work?",
        "What is deep learning?",
        "Describe convolutional neural networks",
        "What is transfer learning?",
        "Explain reinforcement learning",
        "What are transformers in AI?"
    ]
    
    print(f"\nUnlabeled data pool: {len(unlabeled_data)} examples")
    print("\nStarting active learning cycle...")
    
    result1 = agent1.active_learning_cycle(unlabeled_data, annotation_budget=3)
    
    print(f"\nCycle Results:")
    print(f"  Predictions made: {result1['predictions_made']}")
    print(f"  Selected for annotation: {result1['selected_for_annotation']}")
    print(f"  Annotations received: {result1['annotations_received']}")
    print(f"  Examples added to training: {result1['examples_added']}")
    print(f"  Total training examples: {result1['total_training_examples']}")
    print(f"  Average confidence: {result1['avg_confidence']:.2f}")
    
    # Show which examples were selected
    print(f"\nSelected examples (lowest confidence):")
    for i, annotation in enumerate(agent1.annotation_history[:3], 1):
        print(f"  {i}. \"{annotation.input}\" (confidence: {annotation.confidence:.2f})")
    
    # Test 2: Multiple cycles
    print("\n" + "=" * 80)
    print("Test 2: Multiple Active Learning Cycles")
    print("=" * 80)
    
    agent2 = ActiveLearningAgent(
        query_strategy=QueryStrategy.UNCERTAINTY_SAMPLING,
        oracle=oracle
    )
    
    # Run 3 cycles
    for cycle in range(1, 4):
        print(f"\n--- Cycle {cycle} ---")
        result = agent2.active_learning_cycle(unlabeled_data, annotation_budget=2)
        print(f"Training examples: {result['total_training_examples']}")
        print(f"Average confidence: {result['avg_confidence']:.2f}")
    
    # Test 3: Compare strategies
    print("\n" + "=" * 80)
    print("Test 3: Comparing Query Strategies")
    print("=" * 80)
    
    strategies = [
        QueryStrategy.UNCERTAINTY_SAMPLING,
        QueryStrategy.MARGIN_SAMPLING,
        QueryStrategy.DIVERSITY_SAMPLING
    ]
    
    for strategy in strategies:
        agent = ActiveLearningAgent(
            query_strategy=strategy,
            oracle=Oracle(auto_annotate=True)
        )
        
        result = agent.active_learning_cycle(unlabeled_data[:5], annotation_budget=2)
        
        print(f"\n{strategy.value}:")
        print(f"  Examples added: {result['examples_added']}")
        print(f"  Avg confidence: {result['avg_confidence']:.2f}")
    
    # Show final statistics
    print("\n" + "=" * 80)
    print("Final Statistics")
    print("=" * 80)
    
    stats = agent1.get_statistics()
    print(f"\nAgent 1 (Uncertainty Sampling):")
    print(f"  Total annotations: {stats['total_annotations']}")
    print(f"  Training examples: {stats['training_examples']}")
    print(f"  Avg confidence at annotation: {stats['avg_confidence_at_annotation']:.2f}")
    print(f"  Oracle calls: {stats['oracle_calls']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Active Learning provides:
✓ Strategic sample selection for annotation
✓ Confidence-based uncertainty estimation
✓ Multiple query strategies
✓ Efficient use of human expertise
✓ Continuous improvement through feedback
✓ Cost-effective training data collection

This pattern excels at:
- Maximizing learning from limited annotation budget
- Focusing human effort on most informative examples
- Handling ambiguous or edge cases
- Improving model performance efficiently
- Domain adaptation with minimal labels

Query strategies:
1. UNCERTAINTY_SAMPLING: Select least confident predictions
2. MARGIN_SAMPLING: Smallest margin between top predictions
3. ENTROPY_SAMPLING: Highest prediction entropy
4. QUERY_BY_COMMITTEE: Maximum disagreement among models
5. DIVERSITY_SAMPLING: Representative sample diversity
6. EXPECTED_ERROR_REDUCTION: Maximum expected improvement

Active learning cycle:
1. Make predictions on unlabeled data
2. Estimate confidence for each prediction
3. Select most informative examples using query strategy
4. Request human annotations for selected examples
5. Incorporate feedback into training data
6. Repeat until budget exhausted or performance satisfactory

Benefits:
- Efficiency: Fewer labels needed for same performance
- Focus: Human effort on valuable examples
- Adaptability: Continuously improves with feedback
- Cost-effective: Reduces annotation costs
- Quality: Better than random sampling

Use active learning when you need:
- Limited annotation budget
- Efficient model training
- Production model improvement
- Domain adaptation
- Edge case handling
- Cost-effective data collection
""")


if __name__ == "__main__":
    demonstrate_active_learning()
