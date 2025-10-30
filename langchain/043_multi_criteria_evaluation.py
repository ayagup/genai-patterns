"""
Pattern 043: Multi-Criteria Evaluation

Description:
    Multi-Criteria Evaluation enables agents to evaluate outputs across multiple
    dimensions simultaneously, balancing trade-offs between competing objectives
    like accuracy, relevance, safety, cost, and latency. Uses weighted scoring,
    Pareto optimization, and multi-objective decision making.

Components:
    - Criteria Definitions: Define evaluation dimensions
    - Multi-Dimensional Scorer: Evaluates across all criteria
    - Weight Manager: Balances criterion importance
    - Trade-off Analyzer: Identifies Pareto-optimal solutions
    - Aggregator: Combines scores into final decision

Use Cases:
    - Model selection and comparison
    - Response ranking and filtering
    - Production system optimization
    - A/B testing and experimentation
    - Resource allocation decisions
    - Quality vs cost trade-offs

LangChain Implementation:
    Uses multi-dimensional scoring chains to evaluate outputs across multiple
    criteria, identify trade-offs, and select Pareto-optimal solutions.
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class EvaluationCriterion(Enum):
    """Standard evaluation criteria."""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    SAFETY = "safety"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    EFFICIENCY = "efficiency"
    COST = "cost"
    LATENCY = "latency"
    CREATIVITY = "creativity"
    COHERENCE = "coherence"


@dataclass
class CriterionScore:
    """Score for a single criterion."""
    criterion: EvaluationCriterion
    score: float  # 0.0 to 1.0
    weight: float  # Importance weight
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def weighted_score(self) -> float:
        """Calculate weighted score."""
        return self.score * self.weight


@dataclass
class MultiCriteriaEvaluation:
    """Complete multi-criteria evaluation result."""
    candidate_id: str
    criterion_scores: List[CriterionScore]
    overall_score: float
    weighted_score: float
    rankings: Dict[EvaluationCriterion, int]
    trade_offs: List[str]
    is_pareto_optimal: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_score(self, criterion: EvaluationCriterion) -> Optional[float]:
        """Get score for specific criterion."""
        for cs in self.criterion_scores:
            if cs.criterion == criterion:
                return cs.score
        return None
    
    def get_weighted_score(self, criterion: EvaluationCriterion) -> Optional[float]:
        """Get weighted score for specific criterion."""
        for cs in self.criterion_scores:
            if cs.criterion == criterion:
                return cs.weighted_score
        return None


@dataclass
class ComparisonResult:
    """Result of comparing multiple candidates."""
    candidates: List[MultiCriteriaEvaluation]
    pareto_frontier: List[MultiCriteriaEvaluation]
    best_overall: MultiCriteriaEvaluation
    best_per_criterion: Dict[EvaluationCriterion, MultiCriteriaEvaluation]
    trade_off_analysis: List[str]


class MultiCriteriaEvaluator:
    """
    Evaluates outputs across multiple criteria with configurable weights.
    
    Features:
    - Multi-dimensional scoring
    - Configurable criterion weights
    - Pareto optimality detection
    - Trade-off analysis
    - Comprehensive comparison
    """
    
    def __init__(
        self,
        criteria_weights: Optional[Dict[EvaluationCriterion, float]] = None,
        temperature: float = 0.3
    ):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        
        # Default equal weights if not specified
        if criteria_weights is None:
            criteria_weights = {
                EvaluationCriterion.ACCURACY: 1.0,
                EvaluationCriterion.RELEVANCE: 1.0,
                EvaluationCriterion.SAFETY: 1.0,
                EvaluationCriterion.COMPLETENESS: 0.8,
                EvaluationCriterion.CLARITY: 0.8
            }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(criteria_weights.values())
        self.criteria_weights = {
            k: v / total_weight for k, v in criteria_weights.items()
        }
        
        # Criterion-specific evaluation prompt
        self.criterion_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator for the criterion: {criterion}

Evaluation focus:
{criterion_description}

Rate the response on a scale of 0.0 (poor) to 1.0 (excellent).

Provide:
SCORE: [0.0-1.0]
REASONING: [Detailed justification]
STRENGTHS: [What works well]
WEAKNESSES: [What could be improved]"""),
            ("user", """Query: {query}

Response to evaluate:
{response}

Evaluate on {criterion}:""")
        ])
        
        # Trade-off analysis prompt
        self.tradeoff_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze trade-offs between different evaluation criteria.

Identify:
1. Conflicts between criteria
2. Areas where improving one criterion hurts another
3. Opportunities for win-win improvements
4. Pareto-optimal characteristics

Provide trade-off insights."""),
            ("user", """Evaluation scores:
{scores}

Analyze trade-offs:""")
        ])
        
        # Criterion descriptions
        self.criterion_descriptions = {
            EvaluationCriterion.ACCURACY: "Factual correctness and precision of information",
            EvaluationCriterion.RELEVANCE: "How well the response addresses the query",
            EvaluationCriterion.SAFETY: "Absence of harmful, biased, or inappropriate content",
            EvaluationCriterion.COMPLETENESS: "Coverage of all relevant aspects",
            EvaluationCriterion.CLARITY: "Clear, understandable communication",
            EvaluationCriterion.EFFICIENCY: "Conciseness without losing important information",
            EvaluationCriterion.COST: "Resource efficiency (tokens, compute)",
            EvaluationCriterion.LATENCY: "Speed of generation",
            EvaluationCriterion.CREATIVITY: "Originality and novel insights",
            EvaluationCriterion.COHERENCE: "Logical flow and consistency"
        }
        
        # Evaluation history
        self.evaluations: List[MultiCriteriaEvaluation] = []
    
    def evaluate_criterion(
        self,
        query: str,
        response: str,
        criterion: EvaluationCriterion
    ) -> CriterionScore:
        """
        Evaluate response on a single criterion.
        
        Returns:
            CriterionScore object
        """
        chain = self.criterion_prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "query": query,
            "response": response,
            "criterion": criterion.value,
            "criterion_description": self.criterion_descriptions[criterion]
        })
        
        # Parse result
        score = 0.5
        reasoning = ""
        strengths = []
        weaknesses = []
        
        lines = result.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score = float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif line.startswith("REASONING:"):
                reasoning = line.split(':', 1)[1].strip()
                current_section = "reasoning"
            elif line.startswith("STRENGTHS:"):
                current_section = "strengths"
            elif line.startswith("WEAKNESSES:"):
                current_section = "weaknesses"
            elif line and current_section:
                if current_section == "reasoning" and not line.startswith(('-', 'STRENGTHS', 'WEAKNESSES')):
                    reasoning += " " + line
                elif line.startswith('-'):
                    item = line[1:].strip()
                    if current_section == "strengths":
                        strengths.append(item)
                    elif current_section == "weaknesses":
                        weaknesses.append(item)
        
        weight = self.criteria_weights.get(criterion, 1.0)
        
        return CriterionScore(
            criterion=criterion,
            score=score,
            weight=weight,
            reasoning=reasoning,
            metadata={
                "strengths": strengths,
                "weaknesses": weaknesses
            }
        )
    
    def evaluate(
        self,
        candidate_id: str,
        query: str,
        response: str,
        criteria: Optional[List[EvaluationCriterion]] = None
    ) -> MultiCriteriaEvaluation:
        """
        Perform multi-criteria evaluation.
        
        Args:
            candidate_id: Identifier for this candidate
            query: The original query
            response: The response to evaluate
            criteria: List of criteria to evaluate (uses all weighted criteria if None)
            
        Returns:
            MultiCriteriaEvaluation object
        """
        if criteria is None:
            criteria = list(self.criteria_weights.keys())
        
        # Evaluate each criterion
        criterion_scores = []
        for criterion in criteria:
            score = self.evaluate_criterion(query, response, criterion)
            criterion_scores.append(score)
        
        # Calculate overall scores
        overall_score = sum(cs.score for cs in criterion_scores) / len(criterion_scores)
        weighted_score = sum(cs.weighted_score for cs in criterion_scores)
        
        # Calculate rankings (1 = best)
        rankings = {}
        for criterion in criteria:
            rankings[criterion] = 1  # Will be updated in comparison
        
        # Identify trade-offs
        trade_offs = self._analyze_trade_offs(criterion_scores)
        
        evaluation = MultiCriteriaEvaluation(
            candidate_id=candidate_id,
            criterion_scores=criterion_scores,
            overall_score=overall_score,
            weighted_score=weighted_score,
            rankings=rankings,
            trade_offs=trade_offs
        )
        
        self.evaluations.append(evaluation)
        
        return evaluation
    
    def _analyze_trade_offs(
        self,
        criterion_scores: List[CriterionScore]
    ) -> List[str]:
        """Analyze trade-offs between criteria."""
        trade_offs = []
        
        # Find high and low scoring criteria
        high_scores = [cs for cs in criterion_scores if cs.score >= 0.7]
        low_scores = [cs for cs in criterion_scores if cs.score < 0.5]
        
        if high_scores and low_scores:
            high_names = [cs.criterion.value for cs in high_scores]
            low_names = [cs.criterion.value for cs in low_scores]
            trade_offs.append(
                f"Trade-off: Strong in {', '.join(high_names)} but weak in {', '.join(low_names)}"
            )
        
        # Check specific trade-offs
        accuracy_score = next((cs.score for cs in criterion_scores if cs.criterion == EvaluationCriterion.ACCURACY), None)
        creativity_score = next((cs.score for cs in criterion_scores if cs.criterion == EvaluationCriterion.CREATIVITY), None)
        
        if accuracy_score and creativity_score:
            if accuracy_score > 0.7 and creativity_score < 0.5:
                trade_offs.append("Trade-off: High accuracy but low creativity")
            elif creativity_score > 0.7 and accuracy_score < 0.5:
                trade_offs.append("Trade-off: High creativity but lower accuracy")
        
        return trade_offs
    
    def compare(
        self,
        evaluations: List[MultiCriteriaEvaluation]
    ) -> ComparisonResult:
        """
        Compare multiple candidates and identify Pareto frontier.
        
        Args:
            evaluations: List of evaluations to compare
            
        Returns:
            ComparisonResult with rankings and Pareto frontier
        """
        # Update rankings for each criterion
        criteria = set()
        for eval in evaluations:
            criteria.update(eval.rankings.keys())
        
        for criterion in criteria:
            # Sort by score for this criterion (descending)
            sorted_evals = sorted(
                evaluations,
                key=lambda e: e.get_score(criterion) or 0,
                reverse=True
            )
            
            # Assign rankings
            for rank, eval in enumerate(sorted_evals, 1):
                eval.rankings[criterion] = rank
        
        # Find Pareto frontier
        pareto_frontier = self._find_pareto_frontier(evaluations)
        
        # Mark Pareto-optimal candidates
        pareto_ids = {e.candidate_id for e in pareto_frontier}
        for eval in evaluations:
            eval.is_pareto_optimal = eval.candidate_id in pareto_ids
        
        # Find best overall (by weighted score)
        best_overall = max(evaluations, key=lambda e: e.weighted_score)
        
        # Find best per criterion
        best_per_criterion = {}
        for criterion in criteria:
            best = max(evaluations, key=lambda e: e.get_score(criterion) or 0)
            best_per_criterion[criterion] = best
        
        # Overall trade-off analysis
        trade_off_analysis = self._compare_trade_offs(evaluations)
        
        return ComparisonResult(
            candidates=evaluations,
            pareto_frontier=pareto_frontier,
            best_overall=best_overall,
            best_per_criterion=best_per_criterion,
            trade_off_analysis=trade_off_analysis
        )
    
    def _find_pareto_frontier(
        self,
        evaluations: List[MultiCriteriaEvaluation]
    ) -> List[MultiCriteriaEvaluation]:
        """
        Find Pareto-optimal solutions (non-dominated candidates).
        
        A candidate is Pareto-optimal if no other candidate is better
        in all criteria.
        """
        pareto_frontier = []
        
        for candidate in evaluations:
            is_dominated = False
            
            for other in evaluations:
                if other.candidate_id == candidate.candidate_id:
                    continue
                
                # Check if other dominates candidate
                # (better or equal in all criteria, strictly better in at least one)
                better_count = 0
                worse_count = 0
                
                for criterion in candidate.rankings.keys():
                    cand_score = candidate.get_score(criterion) or 0
                    other_score = other.get_score(criterion) or 0
                    
                    if other_score > cand_score:
                        better_count += 1
                    elif other_score < cand_score:
                        worse_count += 1
                
                if better_count > 0 and worse_count == 0:
                    # other dominates candidate
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_frontier.append(candidate)
        
        return pareto_frontier
    
    def _compare_trade_offs(
        self,
        evaluations: List[MultiCriteriaEvaluation]
    ) -> List[str]:
        """Analyze trade-offs across all candidates."""
        analysis = []
        
        if len(evaluations) < 2:
            return analysis
        
        # Find criteria with high variance
        criteria = list(evaluations[0].rankings.keys())
        
        for criterion in criteria:
            scores = [e.get_score(criterion) or 0 for e in evaluations]
            variance = max(scores) - min(scores)
            
            if variance > 0.3:
                analysis.append(
                    f"High variance in {criterion.value}: "
                    f"best {max(scores):.2f}, worst {min(scores):.2f}"
                )
        
        # Pareto analysis
        pareto_count = sum(1 for e in evaluations if e.is_pareto_optimal)
        if pareto_count > 1:
            analysis.append(
                f"{pareto_count} candidates on Pareto frontier - "
                f"no single best solution across all criteria"
            )
        
        return analysis


def demonstrate_multi_criteria_evaluation():
    """
    Demonstrates multi-criteria evaluation with trade-off analysis.
    """
    print("=" * 80)
    print("MULTI-CRITERIA EVALUATION DEMONSTRATION")
    print("=" * 80)
    
    # Test 1: Evaluate multiple responses with different strengths
    print("\n" + "=" * 80)
    print("Test 1: Comparing Multiple Responses")
    print("=" * 80)
    
    # Create evaluator with custom weights
    evaluator = MultiCriteriaEvaluator(
        criteria_weights={
            EvaluationCriterion.ACCURACY: 1.5,
            EvaluationCriterion.RELEVANCE: 1.3,
            EvaluationCriterion.SAFETY: 1.0,
            EvaluationCriterion.CLARITY: 0.8,
            EvaluationCriterion.COMPLETENESS: 0.7
        },
        temperature=0.2
    )
    
    query = "Explain how photosynthesis works."
    
    responses = {
        "response_a": """Photosynthesis is the process by which plants convert sunlight into chemical energy. During photosynthesis, plants take in carbon dioxide from the air and water from the soil. Using chlorophyll in their leaves, they capture light energy and convert it into glucose (sugar) and oxygen. The oxygen is released into the atmosphere, while the glucose is used by the plant for energy and growth.""",
        
        "response_b": """Photosynthesis: 6CO2 + 6H2O + light → C6H12O6 + 6O2. This occurs in chloroplasts through light-dependent and light-independent reactions. Light reactions produce ATP and NADPH, while the Calvin cycle fixes CO2 into glucose.""",
        
        "response_c": """Plants are like solar panels! They catch sunlight and turn it into food. They breathe in CO2 (which we breathe out) and breathe out oxygen (which we need). It's nature's perfect recycling system! The green color in leaves, called chlorophyll, does all the magic."""
    }
    
    print("\nEvaluating 3 different responses...")
    print(f"\nQuery: {query}")
    print(f"\nCriteria weights:")
    for criterion, weight in evaluator.criteria_weights.items():
        print(f"  {criterion.value}: {weight:.2f}")
    
    # Evaluate all responses
    evaluations = []
    for resp_id, response in responses.items():
        print(f"\nEvaluating {resp_id}...")
        evaluation = evaluator.evaluate(resp_id, query, response)
        evaluations.append(evaluation)
    
    # Compare responses
    comparison = evaluator.compare(evaluations)
    
    print("\n[Individual Evaluations]")
    for eval in evaluations:
        print(f"\n{eval.candidate_id}:")
        print(f"  Overall Score: {eval.overall_score:.3f}")
        print(f"  Weighted Score: {eval.weighted_score:.3f}")
        print(f"  Pareto Optimal: {'Yes ✓' if eval.is_pareto_optimal else 'No'}")
        
        print("  Criterion Scores:")
        for cs in eval.criterion_scores:
            print(f"    {cs.criterion.value}: {cs.score:.3f} (weighted: {cs.weighted_score:.3f})")
    
    print("\n[Comparison Results]")
    print(f"\nBest Overall (by weighted score): {comparison.best_overall.candidate_id}")
    print(f"  Weighted Score: {comparison.best_overall.weighted_score:.3f}")
    
    print("\nBest per Criterion:")
    for criterion, best_eval in comparison.best_per_criterion.items():
        score = best_eval.get_score(criterion)
        print(f"  {criterion.value}: {best_eval.candidate_id} ({score:.3f})")
    
    print(f"\nPareto Frontier ({len(comparison.pareto_frontier)} candidates):")
    for eval in comparison.pareto_frontier:
        print(f"  {eval.candidate_id} - Weighted: {eval.weighted_score:.3f}, Overall: {eval.overall_score:.3f}")
    
    if comparison.trade_off_analysis:
        print("\nTrade-off Analysis:")
        for analysis in comparison.trade_off_analysis:
            print(f"  - {analysis}")
    
    # Test 2: Custom criteria evaluation
    print("\n" + "=" * 80)
    print("Test 2: Custom Criteria Configuration")
    print("=" * 80)
    
    # Prioritize safety and clarity over other factors
    safety_focused = MultiCriteriaEvaluator(
        criteria_weights={
            EvaluationCriterion.SAFETY: 2.0,
            EvaluationCriterion.CLARITY: 1.5,
            EvaluationCriterion.ACCURACY: 1.0,
            EvaluationCriterion.RELEVANCE: 0.8
        },
        temperature=0.2
    )
    
    query2 = "How do I handle conflicts at work?"
    response2 = "Stay calm, listen actively, focus on the issue not the person, seek common ground, and involve HR if needed."
    
    print(f"\nQuery: {query2}")
    print("\nPriority: Safety and Clarity")
    
    eval2 = safety_focused.evaluate("response_safety", query2, response2)
    
    print("\n[Evaluation Results]")
    print(f"Overall Score: {eval2.overall_score:.3f}")
    print(f"Weighted Score: {eval2.weighted_score:.3f}")
    
    print("\nDetailed Scores:")
    for cs in sorted(eval2.criterion_scores, key=lambda x: x.weighted_score, reverse=True):
        print(f"  {cs.criterion.value}:")
        print(f"    Raw: {cs.score:.3f} | Weight: {cs.weight:.2f} | Weighted: {cs.weighted_score:.3f}")
        print(f"    {cs.reasoning[:100]}...")
    
    # Test 3: Pareto frontier visualization
    print("\n" + "=" * 80)
    print("Test 3: Pareto Frontier Analysis")
    print("=" * 80)
    
    print("\n[Score Matrix]")
    print(f"{'Response':<12} ", end="")
    for criterion in evaluations[0].rankings.keys():
        print(f"{criterion.value[:8]:<10}", end="")
    print()
    
    for eval in evaluations:
        print(f"{eval.candidate_id:<12} ", end="")
        for criterion in eval.rankings.keys():
            score = eval.get_score(criterion) or 0
            marker = "★" if eval.is_pareto_optimal else " "
            print(f"{score:.2f}{marker:<7}", end="")
        print()
    
    print("\n★ = Pareto Optimal")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Multi-Criteria Evaluation provides:
✓ Multi-dimensional scoring
✓ Configurable criterion weights
✓ Pareto optimality detection
✓ Trade-off analysis
✓ Comprehensive comparison
✓ Balanced decision making

This pattern excels at:
- Model selection and comparison
- Response ranking
- Production optimization
- A/B testing
- Resource allocation
- Quality vs cost trade-offs

Standard evaluation criteria:
- Accuracy: Factual correctness
- Relevance: Addresses query
- Safety: No harmful content
- Completeness: Comprehensive coverage
- Clarity: Clear communication
- Efficiency: Concise
- Cost: Resource usage
- Latency: Speed
- Creativity: Originality
- Coherence: Logical flow

Evaluation process:
1. Define criteria and weights
2. Score each criterion
3. Calculate weighted scores
4. Identify trade-offs
5. Find Pareto frontier
6. Compare candidates

Pareto optimality:
- Non-dominated solutions
- No candidate better in all criteria
- Multiple optimal trade-offs
- No single "best" solution

Weight configuration:
- Higher weight = more important
- Normalized to sum to 1.0
- Domain-specific priorities
- Stakeholder preferences

Benefits:
- Balanced: Consider multiple factors
- Transparent: Clear criteria
- Flexible: Configurable weights
- Comprehensive: Holistic evaluation
- Trade-offs: Explicit conflicts
- Optimal: Pareto solutions

Use Multi-Criteria Evaluation when:
- Multiple objectives matter
- Trade-offs exist
- Need balanced decisions
- Comparing alternatives
- Production deployment
- Stakeholder alignment needed

Comparison with other patterns:
- vs Self-Evaluation: Multiple dimensions vs single quality
- vs Benchmark-Driven: Balanced evaluation vs single metric
- vs Progressive Optimization: Evaluation vs improvement
""")


if __name__ == "__main__":
    demonstrate_multi_criteria_evaluation()
