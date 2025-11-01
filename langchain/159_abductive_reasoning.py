"""
Pattern 159: Abductive Reasoning

Description:
    Abductive reasoning infers the best explanation for a set of observations.
    Unlike deductive reasoning (which derives consequences from premises) or
    inductive reasoning (which generalizes from examples), abduction works
    backwards from observations to find the most likely cause or explanation.

Components:
    - Observation collection
    - Hypothesis generation
    - Explanation evaluation
    - Likelihood ranking
    - Plausibility assessment

Use Cases:
    - Diagnostic reasoning (medical, technical)
    - Root cause analysis
    - Mystery solving
    - Troubleshooting
    - Scientific hypothesis formation

Benefits:
    - Finds likely explanations
    - Handles incomplete information
    - Supports decision-making
    - Enables inference to best explanation

Trade-offs:
    - Conclusions not guaranteed
    - Multiple possible explanations
    - Requires domain knowledge
    - Computationally intensive

LangChain Implementation:
    Uses ChatOpenAI for hypothesis generation, evaluation scoring,
    and ranking of explanations by plausibility
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

load_dotenv()


class ConfidenceLevel(Enum):
    """Confidence levels for hypotheses"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class Observation:
    """Single observation or evidence"""
    description: str
    importance: float = 1.0  # 0-1 scale
    timestamp: Optional[str] = None
    source: Optional[str] = None


@dataclass
class Hypothesis:
    """Potential explanation"""
    explanation: str
    confidence: float  # 0-1 scale
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    plausibility_score: float = 0.0


@dataclass
class AbductiveAnalysis:
    """Result of abductive reasoning"""
    observations: List[Observation]
    hypotheses: List[Hypothesis]
    best_explanation: Optional[Hypothesis]
    reasoning_trace: List[str] = field(default_factory=list)


class AbductiveReasoningAgent:
    """Agent that performs abductive reasoning"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize abductive reasoning agent
        
        Args:
            model_name: LLM model to use
        """
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.evaluator_llm = ChatOpenAI(model=model_name, temperature=0)
    
    def reason(self, observations: List[Observation], 
               domain_context: Optional[str] = None,
               num_hypotheses: int = 5) -> AbductiveAnalysis:
        """
        Perform abductive reasoning on observations
        
        Args:
            observations: List of observations
            domain_context: Optional domain-specific context
            num_hypotheses: Number of hypotheses to generate
            
        Returns:
            AbductiveAnalysis with ranked hypotheses
        """
        reasoning_trace = []
        
        # Step 1: Generate hypotheses
        reasoning_trace.append("Generating hypotheses from observations...")
        hypotheses = self._generate_hypotheses(observations, domain_context, num_hypotheses)
        reasoning_trace.append(f"Generated {len(hypotheses)} hypotheses")
        
        # Step 2: Evaluate each hypothesis
        reasoning_trace.append("Evaluating hypotheses...")
        for hypothesis in hypotheses:
            self._evaluate_hypothesis(hypothesis, observations)
        
        # Step 3: Rank hypotheses
        hypotheses.sort(key=lambda h: h.plausibility_score, reverse=True)
        reasoning_trace.append("Ranked hypotheses by plausibility")
        
        # Step 4: Select best explanation
        best_explanation = hypotheses[0] if hypotheses else None
        
        return AbductiveAnalysis(
            observations=observations,
            hypotheses=hypotheses,
            best_explanation=best_explanation,
            reasoning_trace=reasoning_trace
        )
    
    def _generate_hypotheses(self, observations: List[Observation],
                            domain_context: Optional[str],
                            num_hypotheses: int) -> List[Hypothesis]:
        """Generate potential hypotheses"""
        # Format observations
        obs_text = "\n".join([
            f"- {obs.description} (importance: {obs.importance:.1f})"
            for obs in observations
        ])
        
        # Create prompt for hypothesis generation
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at abductive reasoning - inferring the best explanation for observations.
Generate {num_hypotheses} plausible hypotheses that could explain the given observations.
For each hypothesis, provide:
1. Clear explanation
2. Supporting reasoning
3. Key assumptions

{domain_context}

Return as JSON array: [{{"explanation": "...", "reasoning": "...", "assumptions": ["..."]}}]"""),
            ("user", "Observations:\n{observations}\n\nGenerate {num_hypotheses} hypotheses:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        context = f"Domain context: {domain_context}" if domain_context else "Consider general knowledge."
        
        response = chain.invoke({
            "observations": obs_text,
            "num_hypotheses": num_hypotheses,
            "domain_context": context
        })
        
        # Parse response
        hypotheses = self._parse_hypotheses(response)
        return hypotheses
    
    def _parse_hypotheses(self, response: str) -> List[Hypothesis]:
        """Parse hypotheses from LLM response"""
        try:
            # Try to extract JSON
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                hypotheses = []
                for item in data:
                    hypothesis = Hypothesis(
                        explanation=item.get('explanation', ''),
                        confidence=0.5,  # Initial neutral confidence
                        assumptions=item.get('assumptions', [])
                    )
                    hypotheses.append(hypothesis)
                return hypotheses
        except Exception as e:
            print(f"Error parsing hypotheses: {e}")
        
        # Fallback: split by numbered items
        lines = response.split('\n')
        hypotheses = []
        current_explanation = []
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                if current_explanation:
                    explanation_text = ' '.join(current_explanation)
                    if explanation_text:
                        hypotheses.append(Hypothesis(
                            explanation=explanation_text,
                            confidence=0.5
                        ))
                    current_explanation = []
                # Remove numbering/bullet
                clean_line = line.lstrip('0123456789.-) ')
                current_explanation.append(clean_line)
            elif current_explanation:
                current_explanation.append(line)
        
        # Add last hypothesis
        if current_explanation:
            explanation_text = ' '.join(current_explanation)
            if explanation_text:
                hypotheses.append(Hypothesis(
                    explanation=explanation_text,
                    confidence=0.5
                ))
        
        return hypotheses[:5]  # Limit to 5
    
    def _evaluate_hypothesis(self, hypothesis: Hypothesis, 
                            observations: List[Observation]):
        """Evaluate a hypothesis against observations"""
        # Format observations
        obs_text = "\n".join([obs.description for obs in observations])
        
        # Create evaluation prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Evaluate how well this hypothesis explains the observations.
Provide:
1. Supporting evidence (which observations support it)
2. Contradicting evidence (which observations contradict it)
3. Confidence score (0-1)
4. Plausibility assessment

Return as JSON: {{
    "supporting": ["..."],
    "contradicting": ["..."],
    "confidence": 0.0-1.0,
    "plausibility": 0.0-1.0,
    "reasoning": "..."
}}"""),
            ("user", """Hypothesis: {hypothesis}

Observations:
{observations}

Evaluate:""")
        ])
        
        chain = prompt | self.evaluator_llm | StrOutputParser()
        
        response = chain.invoke({
            "hypothesis": hypothesis.explanation,
            "observations": obs_text
        })
        
        # Parse evaluation
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                hypothesis.supporting_evidence = data.get('supporting', [])
                hypothesis.contradicting_evidence = data.get('contradicting', [])
                hypothesis.confidence = float(data.get('confidence', 0.5))
                hypothesis.plausibility_score = float(data.get('plausibility', 0.5))
        except Exception as e:
            print(f"Error parsing evaluation: {e}")
            # Use default values
            hypothesis.confidence = 0.5
            hypothesis.plausibility_score = 0.5
    
    def explain_reasoning(self, analysis: AbductiveAnalysis) -> str:
        """Generate natural language explanation of reasoning"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Explain the abductive reasoning process in clear, natural language."),
            ("user", """Observations:
{observations}

Best Explanation: {best_explanation}
Confidence: {confidence:.2%}

Supporting Evidence:
{supporting}

Explain why this is the best explanation:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        obs_text = "\n".join([f"- {obs.description}" for obs in analysis.observations])
        supporting_text = "\n".join([f"- {s}" for s in analysis.best_explanation.supporting_evidence])
        
        explanation = chain.invoke({
            "observations": obs_text,
            "best_explanation": analysis.best_explanation.explanation,
            "confidence": analysis.best_explanation.confidence,
            "supporting": supporting_text
        })
        
        return explanation


def demonstrate_abductive_reasoning():
    """Demonstrate abductive reasoning pattern"""
    print("=" * 80)
    print("ABDUCTIVE REASONING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = AbductiveReasoningAgent()
    
    # Example 1: Medical diagnosis
    print("\n" + "="*80)
    print("EXAMPLE 1: Medical Diagnosis")
    print("="*80)
    observations = [
        Observation("Patient has a fever of 101°F", importance=0.9),
        Observation("Patient reports sore throat", importance=0.8),
        Observation("Patient has swollen lymph nodes", importance=0.7),
        Observation("Patient has been in contact with sick individuals", importance=0.6)
    ]
    
    analysis = agent.reason(
        observations,
        domain_context="Medical diagnosis",
        num_hypotheses=4
    )
    
    print("Observations:")
    for obs in observations:
        print(f"  - {obs.description}")
    
    print(f"\nGenerated {len(analysis.hypotheses)} hypotheses:")
    for i, hyp in enumerate(analysis.hypotheses, 1):
        print(f"\n{i}. {hyp.explanation}")
        print(f"   Confidence: {hyp.confidence:.2%}")
        print(f"   Plausibility: {hyp.plausibility_score:.2%}")
    
    print(f"\nBest Explanation:")
    print(f"  {analysis.best_explanation.explanation}")
    print(f"  Confidence: {analysis.best_explanation.confidence:.2%}")
    
    # Example 2: Technical troubleshooting
    print("\n" + "="*80)
    print("EXAMPLE 2: Technical Troubleshooting")
    print("="*80)
    observations = [
        Observation("Website is loading slowly", importance=1.0),
        Observation("Database query times are normal", importance=0.8),
        Observation("CPU usage is at 95%", importance=0.9),
        Observation("Recent code deployment yesterday", importance=0.7),
        Observation("Memory usage is normal", importance=0.6)
    ]
    
    analysis = agent.reason(
        observations,
        domain_context="Software system performance troubleshooting",
        num_hypotheses=3
    )
    
    print("Observations:")
    for obs in observations:
        print(f"  - {obs.description}")
    
    print(f"\nTop 3 Hypotheses:")
    for i, hyp in enumerate(analysis.hypotheses[:3], 1):
        print(f"\n{i}. {hyp.explanation}")
        print(f"   Plausibility: {hyp.plausibility_score:.2%}")
        if hyp.supporting_evidence:
            print(f"   Supporting: {', '.join(hyp.supporting_evidence[:2])}")
    
    # Example 3: Mystery scenario
    print("\n" + "="*80)
    print("EXAMPLE 3: Mystery Investigation")
    print("="*80)
    observations = [
        Observation("Window is broken from the inside", importance=0.9),
        Observation("Valuable items are missing", importance=1.0),
        Observation("No signs of forced entry", importance=0.8),
        Observation("Alarm system was deactivated", importance=0.9),
        Observation("Security code was used correctly", importance=0.95)
    ]
    
    analysis = agent.reason(
        observations,
        domain_context="Criminal investigation",
        num_hypotheses=4
    )
    
    print("Observations:")
    for obs in observations:
        print(f"  - {obs.description}")
    
    print(f"\nBest Explanation:")
    best = analysis.best_explanation
    print(f"  {best.explanation}")
    print(f"  Confidence: {best.confidence:.2%}")
    print(f"  Plausibility: {best.plausibility_score:.2%}")
    
    # Generate natural language explanation
    explanation = agent.explain_reasoning(analysis)
    print(f"\nReasoning:")
    print(f"  {explanation[:300]}...")
    
    # Example 4: Business analysis
    print("\n" + "="*80)
    print("EXAMPLE 4: Business Analysis")
    print("="*80)
    observations = [
        Observation("Sales dropped 30% this month", importance=1.0),
        Observation("Competitor launched new product", importance=0.8),
        Observation("Customer complaints increased", importance=0.7),
        Observation("Website traffic is unchanged", importance=0.6),
        Observation("Pricing remained the same", importance=0.5)
    ]
    
    analysis = agent.reason(
        observations,
        domain_context="Business performance analysis",
        num_hypotheses=3
    )
    
    print("Observations:")
    for obs in observations:
        print(f"  - {obs.description}")
    
    print(f"\nHypotheses (ranked by plausibility):")
    for i, hyp in enumerate(analysis.hypotheses, 1):
        print(f"\n{i}. {hyp.explanation}")
        print(f"   Plausibility: {hyp.plausibility_score:.2%}")
    
    # Example 5: Scientific reasoning
    print("\n" + "="*80)
    print("EXAMPLE 5: Scientific Observation")
    print("="*80)
    observations = [
        Observation("Plant leaves are turning yellow", importance=0.9),
        Observation("Soil is consistently wet", importance=0.8),
        Observation("Plant is in low light area", importance=0.7),
        Observation("No visible pests or diseases", importance=0.6)
    ]
    
    analysis = agent.reason(
        observations,
        domain_context="Plant biology and care",
        num_hypotheses=4
    )
    
    print("Observations:")
    for obs in observations:
        print(f"  - {obs.description}")
    
    print(f"\nTop Hypothesis:")
    best = analysis.hypotheses[0]
    print(f"  Explanation: {best.explanation}")
    print(f"  Confidence: {best.confidence:.2%}")
    if best.assumptions:
        print(f"  Assumptions: {', '.join(best.assumptions)}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Abductive Reasoning Best Practices")
    print("="*80)
    print("""
1. OBSERVATION COLLECTION:
   - Gather all available evidence
   - Note importance/relevance of each observation
   - Include timing and context
   - Consider source reliability

2. HYPOTHESIS GENERATION:
   - Generate multiple plausible explanations
   - Consider diverse possibilities
   - Don't prematurely eliminate options
   - Use domain knowledge

3. EVALUATION CRITERIA:
   - Explanatory power (how well it explains observations)
   - Simplicity (Occam's Razor)
   - Consistency with known facts
   - Plausibility given context

4. RANKING FACTORS:
   - Supporting evidence count and quality
   - Contradicting evidence
   - Prior probability
   - Completeness of explanation

5. REASONING PROCESS:
   - Work backwards from observations
   - Consider alternative explanations
   - Evaluate each hypothesis rigorously
   - Update beliefs with new evidence

6. APPLICATIONS:
   - Diagnostic reasoning
   - Root cause analysis
   - Mystery solving
   - Scientific hypothesis formation
   - Decision support systems

Benefits:
✓ Finds most likely explanations
✓ Handles incomplete information
✓ Supports diagnostic tasks
✓ Enables inference to best explanation
✓ Practical for real-world problems

Limitations:
- Conclusions not guaranteed correct
- Multiple explanations may be equally plausible
- Requires domain knowledge
- Sensitive to observation quality
- May miss novel explanations
    """)


if __name__ == "__main__":
    demonstrate_abductive_reasoning()
