"""
Pattern 43: Multi-Criteria Evaluation
Description:
    Evaluates agent outputs across multiple dimensions with configurable
    weights and scoring methods to make holistic quality assessments.
Use Cases:
    - Complex decision-making with trade-offs
    - Multi-objective optimization
    - Comparing alternative solutions
    - Quality assurance with multiple factors
Key Features:
    - Configurable evaluation criteria
    - Weighted scoring system
    - Pareto optimization support
    - Trade-off analysis
Example:
    >>> evaluator = MultiCriteriaEvaluator()
    >>> scores = evaluator.evaluate(output, criteria)
    >>> best = evaluator.select_best([output1, output2, output3])
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import statistics
class CriterionType(Enum):
    """Types of evaluation criteria"""
    MAXIMIZE = "maximize"  # Higher is better
    MINIMIZE = "minimize"  # Lower is better
    TARGET = "target"      # Closer to target is better
class AggregationMethod(Enum):
    """Methods for aggregating scores"""
    WEIGHTED_SUM = "weighted_sum"
    WEIGHTED_PRODUCT = "weighted_product"
    MIN_MAX = "min_max"
    PARETO = "pareto"
@dataclass
class EvaluationCriterion:
    """Single evaluation criterion"""
    name: str
    description: str
    criterion_type: CriterionType
    weight: float = 1.0
    target_value: Optional[float] = None
    min_value: float = 0.0
    max_value: float = 1.0
    scoring_function: Optional[Callable[[Any], float]] = None
@dataclass
class CriterionScore:
    """Score for a single criterion"""
    criterion_name: str
    raw_value: float
    normalized_score: float  # 0.0 to 1.0
    weighted_score: float
    explanation: str = ""
@dataclass
class MultiCriteriaScore:
    """Complete multi-criteria evaluation result"""
    item_id: str
    criterion_scores: Dict[str, CriterionScore]
    aggregate_score: float
    method: AggregationMethod
    is_pareto_optimal: bool = False
    dominated_by: List[str] = field(default_factory=list)
    rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
@dataclass
class TradeOffAnalysis:
    """Analysis of trade-offs between criteria"""
    improved_criteria: List[str]
    degraded_criteria: List[str]
    net_improvement: float
    recommendation: str
class MultiCriteriaEvaluator:
    """
    Evaluates items across multiple criteria
    Features:
    - Flexible criterion definition
    - Multiple aggregation methods
    - Pareto optimization
    - Trade-off analysis
    """
    def __init__(
        self,
        criteria: Optional[List[EvaluationCriterion]] = None,
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_SUM
    ):
        self.criteria = criteria or []
        self.aggregation_method = aggregation_method
        self.evaluation_history: List[MultiCriteriaScore] = []
    def add_criterion(self, criterion: EvaluationCriterion):
        """Add an evaluation criterion"""
        self.criteria.append(criterion)
    def evaluate(
        self,
        item: Any,
        item_id: str,
        custom_values: Optional[Dict[str, float]] = None
    ) -> MultiCriteriaScore:
        """
        Evaluate an item across all criteria
        Args:
            item: Item to evaluate
            item_id: Unique identifier
            custom_values: Optional pre-computed criterion values
        Returns:
            Multi-criteria score
        """
        criterion_scores = {}
        for criterion in self.criteria:
            # Get raw value
            if custom_values and criterion.name in custom_values:
                raw_value = custom_values[criterion.name]
            elif criterion.scoring_function:
                raw_value = criterion.scoring_function(item)
            else:
                raw_value = self._default_scoring(item, criterion)
            # Normalize score
            normalized_score = self._normalize_score(
                raw_value, criterion
            )
            # Apply weight
            weighted_score = normalized_score * criterion.weight
            # Create score object
            criterion_scores[criterion.name] = CriterionScore(
                criterion_name=criterion.name,
                raw_value=raw_value,
                normalized_score=normalized_score,
                weighted_score=weighted_score,
                explanation=self._explain_score(raw_value, criterion)
            )
        # Aggregate scores
        aggregate_score = self._aggregate_scores(
            criterion_scores,
            self.aggregation_method
        )
        result = MultiCriteriaScore(
            item_id=item_id,
            criterion_scores=criterion_scores,
            aggregate_score=aggregate_score,
            method=self.aggregation_method,
            metadata={'item': str(item)[:100]}
        )
        self.evaluation_history.append(result)
        return result
    def evaluate_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> List[MultiCriteriaScore]:
        """
        Evaluate multiple items
        Args:
            items: List of {item_id, item, custom_values}
        Returns:
            List of scores
        """
        scores = []
        for item_data in items:
            score = self.evaluate(
                item=item_data['item'],
                item_id=item_data['item_id'],
                custom_values=item_data.get('custom_values')
            )
            scores.append(score)
        # Identify Pareto optimal solutions
        if self.aggregation_method == AggregationMethod.PARETO:
            self._identify_pareto_optimal(scores)
        # Rank solutions
        self._rank_solutions(scores)
        return scores
    def select_best(
        self,
        items: List[Dict[str, Any]],
        top_n: int = 1
    ) -> List[MultiCriteriaScore]:
        """
        Select best item(s) based on evaluation
        Args:
            items: Items to evaluate
            top_n: Number of top items to return
        Returns:
        """
        scores = self.evaluate_batch(items)
        # Sort by aggregate score (descending)
        sorted_scores = sorted(
            scores,
            key=lambda x: x.aggregate_score,
            reverse=True
        )
        return sorted_scores[:top_n]
    def analyze_tradeoffs(
        self,
        item1_id: str,
        item2_id: str
    ) -> TradeOffAnalysis:
        """
        Analyze trade-offs between two evaluated items
        Args:
            item1_id: First item ID
            item2_id: Second item ID
        Returns:
            Trade-off analysis
        """
        # Find scores
        score1 = next(
            (s for s in self.evaluation_history if s.item_id == item1_id),
            None
        )
        score2 = next(
            (s for s in self.evaluation_history if s.item_id == item2_id),
            None
        )
        if not score1 or not score2:
            raise ValueError("Both items must be evaluated first")
        improved_criteria = []
        degraded_criteria = []
        net_improvement = 0.0
        for criterion_name in score1.criterion_scores.keys():
            score1_val = score1.criterion_scores[criterion_name].normalized_score
            score2_val = score2.criterion_scores[criterion_name].normalized_score
            diff = score2_val - score1_val
            if diff > 0.05:  # Significant improvement
                improved_criteria.append(criterion_name)
                net_improvement += diff
            elif diff < -0.05:  # Significant degradation
                degraded_criteria.append(criterion_name)
                net_improvement += diff
        # Generate recommendation
        if net_improvement > 0.1:
            recommendation = f"Choose {item2_id}: Overall improvement of {net_improvement:.2%}"
        elif net_improvement < -0.1:
            recommendation = f"Choose {item1_id}: {item2_id} shows degradation of {-net_improvement:.2%}"
        else:
            recommendation = "Items are comparable; choose based on priority criteria"
        return TradeOffAnalysis(
            improved_criteria=improved_criteria,
            degraded_criteria=degraded_criteria,
            net_improvement=net_improvement,
            recommendation=recommendation
        )
    def _normalize_score(
        self,
        raw_value: float,
        criterion: EvaluationCriterion
    ) -> float:
        """Normalize score to 0-1 range"""
        if criterion.criterion_type == CriterionType.MAXIMIZE:
            # Higher is better
            if criterion.max_value == criterion.min_value:
                return 1.0
            normalized = (raw_value - criterion.min_value) / (
                criterion.max_value - criterion.min_value
            )
            return max(0.0, min(1.0, normalized))
        elif criterion.criterion_type == CriterionType.MINIMIZE:
            # Lower is better
            if criterion.max_value == criterion.min_value:
                return 1.0
            normalized = (criterion.max_value - raw_value) / (
                criterion.max_value - criterion.min_value
            )
            return max(0.0, min(1.0, normalized))
        elif criterion.criterion_type == CriterionType.TARGET:
            # Closer to target is better
            if criterion.target_value is None:
                return 0.5
            distance = abs(raw_value - criterion.target_value)
            max_distance = max(
                abs(criterion.max_value - criterion.target_value),
                abs(criterion.min_value - criterion.target_value)
            )
            if max_distance == 0:
                return 1.0
            normalized = 1.0 - (distance / max_distance)
            return max(0.0, min(1.0, normalized))
        return 0.5
    def _aggregate_scores(
        self,
        criterion_scores: Dict[str, CriterionScore],
        method: AggregationMethod
    ) -> float:
        """Aggregate criterion scores using specified method"""
        if not criterion_scores:
            return 0.0
        if method == AggregationMethod.WEIGHTED_SUM:
            total_weight = sum(
                score.weighted_score
                for score in criterion_scores.values()
            )
            weight_sum = sum(
                self._get_criterion_weight(name)
                for name in criterion_scores.keys()
            )
            return total_weight / weight_sum if weight_sum > 0 else 0.0
        elif method == AggregationMethod.WEIGHTED_PRODUCT:
            product = 1.0
            total_weight = 0.0
            for score in criterion_scores.values():
                weight = self._get_criterion_weight(score.criterion_name)
                product *= (score.normalized_score ** weight)
                total_weight += weight
            return product ** (1.0 / total_weight) if total_weight > 0 else 0.0
        elif method == AggregationMethod.MIN_MAX:
            # Return minimum score (conservative)
            return min(
                score.normalized_score
                for score in criterion_scores.values()
            )
        elif method == AggregationMethod.PARETO:
            # For Pareto, use weighted sum but mark as needing Pareto analysis
            return self._aggregate_scores(
                criterion_scores,
                AggregationMethod.WEIGHTED_SUM
            )
        return 0.0
    def _identify_pareto_optimal(self, scores: List[MultiCriteriaScore]):
        """Identify Pareto optimal solutions"""
        for i, score1 in enumerate(scores):
            is_dominated = False
            dominated_by = []
            for j, score2 in enumerate(scores):
                if i == j:
                    continue
                if self._dominates(score2, score1):
                    is_dominated = True
                    dominated_by.append(score2.item_id)
            score1.is_pareto_optimal = not is_dominated
            score1.dominated_by = dominated_by
    def _dominates(
        self,
        score1: MultiCriteriaScore,
        score2: MultiCriteriaScore
    ) -> bool:
        """Check if score1 Pareto-dominates score2"""
        at_least_one_better = False
        for criterion_name in score1.criterion_scores.keys():
            val1 = score1.criterion_scores[criterion_name].normalized_score
            val2 = score2.criterion_scores[criterion_name].normalized_score
            if val1 < val2:  # score1 is worse in this criterion
                return False
            elif val1 > val2:  # score1 is better in this criterion
                at_least_one_better = True
        return at_least_one_better
    def _rank_solutions(self, scores: List[MultiCriteriaScore]):
        """Assign ranks to solutions"""
        sorted_scores = sorted(
            scores,
            key=lambda x: x.aggregate_score,
            reverse=True
        )
        for rank, score in enumerate(sorted_scores, 1):
            score.rank = rank
    def _get_criterion_weight(self, criterion_name: str) -> float:
        """Get weight for a criterion"""
        for criterion in self.criteria:
            if criterion.name == criterion_name:
                return criterion.weight
        return 1.0
    def _default_scoring(self, item: Any, criterion: EvaluationCriterion) -> float:
        """Default scoring function"""
        # Simple heuristic based on item attributes
        if hasattr(item, criterion.name):
            return float(getattr(item, criterion.name))
        return 0.5
    def _explain_score(
        self,
        raw_value: float,
        criterion: EvaluationCriterion
    ) -> str:
        """Generate explanation for score"""
        if criterion.criterion_type == CriterionType.MAXIMIZE:
            return f"Raw value: {raw_value:.3f} (higher is better, max: {criterion.max_value})"
        elif criterion.criterion_type == CriterionType.MINIMIZE:
            return f"Raw value: {raw_value:.3f} (lower is better, min: {criterion.min_value})"
        elif criterion.criterion_type == CriterionType.TARGET:
            return f"Raw value: {raw_value:.3f} (target: {criterion.target_value})"
        return f"Raw value: {raw_value:.3f}"
    def get_criterion_rankings(self) -> Dict[str, List[str]]:
        """Get item rankings for each criterion"""
        rankings = {}
        for criterion in self.criteria:
            # Sort by this criterion
            sorted_items = sorted(
                self.evaluation_history,
                key=lambda x: x.criterion_scores[criterion.name].normalized_score,
                reverse=True
            )
            rankings[criterion.name] = [item.item_id for item in sorted_items]
        return rankings
    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        if not self.evaluation_history:
            return {'message': 'No evaluations performed yet'}
        # Overall statistics
        aggregate_scores = [s.aggregate_score for s in self.evaluation_history]
        # Per-criterion statistics
        criterion_stats = {}
        for criterion in self.criteria:
            scores = [
                s.criterion_scores[criterion.name].normalized_score
                for s in self.evaluation_history
                if criterion.name in s.criterion_scores
            ]
            if scores:
                criterion_stats[criterion.name] = {
                    'mean': statistics.mean(scores),
                    'median': statistics.median(scores),
                    'stdev': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    'min': min(scores),
                    'max': max(scores)
                }
        # Pareto optimal count
        pareto_count = sum(
            1 for s in self.evaluation_history if s.is_pareto_optimal
        )
        return {
            'total_evaluations': len(self.evaluation_history),
            'aggregation_method': self.aggregation_method.value,
            'num_criteria': len(self.criteria),
            'aggregate_score_stats': {
                'mean': statistics.mean(aggregate_scores),
                'median': statistics.median(aggregate_scores),
                'min': min(aggregate_scores),
                'max': max(aggregate_scores)
            },
            'criterion_statistics': criterion_stats,
            'pareto_optimal_count': pareto_count,
            'pareto_optimal_percentage': pareto_count / len(self.evaluation_history) * 100
        }
class LLMOutputEvaluator(MultiCriteriaEvaluator):
    """
    Specialized evaluator for LLM outputs
    Pre-configured with common LLM evaluation criteria
    """
    def __init__(self):
        criteria = [
            EvaluationCriterion(
                name="accuracy",
                description="Factual correctness of the output",
                criterion_type=CriterionType.MAXIMIZE,
                weight=3.0,
                min_value=0.0,
                max_value=1.0
            ),
            EvaluationCriterion(
                name="relevance",
                description="How well output addresses the query",
                criterion_type=CriterionType.MAXIMIZE,
                weight=2.5,
                min_value=0.0,
                max_value=1.0
            ),
            EvaluationCriterion(
                name="completeness",
                description="Coverage of all required aspects",
                criterion_type=CriterionType.MAXIMIZE,
                weight=2.0,
                min_value=0.0,
                max_value=1.0
            ),
            EvaluationCriterion(
                name="clarity",
                description="Ease of understanding",
                criterion_type=CriterionType.MAXIMIZE,
                weight=1.5,
                min_value=0.0,
                max_value=1.0
            ),
            EvaluationCriterion(
                name="conciseness",
                description="Brevity without losing content",
                criterion_type=CriterionType.MAXIMIZE,
                weight=1.0,
                min_value=0.0,
                max_value=1.0
            ),
            EvaluationCriterion(
                name="latency",
                description="Response generation time",
                criterion_type=CriterionType.MINIMIZE,
                weight=1.0,
                min_value=0.0,
                max_value=10.0  # seconds
            ),
            EvaluationCriterion(
                name="token_count",
                description="Number of tokens used",
                criterion_type=CriterionType.MINIMIZE,
                weight=0.5,
                min_value=0,
                max_value=4000
            )
        ]
        super().__init__(criteria, AggregationMethod.WEIGHTED_SUM)
def main():
    """Demonstrate multi-criteria evaluation pattern"""
    print("=" * 60)
    print("Multi-Criteria Evaluation Demonstration")
    print("=" * 60)
    # Create LLM output evaluator
    evaluator = LLMOutputEvaluator()
    print("\n1. Evaluation Criteria")
    print("-" * 60)
    for criterion in evaluator.criteria:
        print(f"\n{criterion.name.upper()}")
        print(f"  Description: {criterion.description}")
        print(f"  Type: {criterion.criterion_type.value}")
        print(f"  Weight: {criterion.weight}")
        print(f"  Range: {criterion.min_value} - {criterion.max_value}")
    print("\n" + "=" * 60)
    print("2. Evaluating Multiple LLM Outputs")
    print("=" * 60)
    # Simulate different LLM outputs with varying characteristics
    outputs = [
        {
            'item_id': 'output_1',
            'item': 'Detailed and accurate response with examples',
            'custom_values': {
                'accuracy': 0.95,
                'relevance': 0.90,
                'completeness': 0.85,
                'clarity': 0.80,
                'conciseness': 0.60,  # Very detailed
                'latency': 2.5,
                'token_count': 800
            }
        },
        {
            'item_id': 'output_2',
            'item': 'Concise but less complete response',
            'custom_values': {
                'accuracy': 0.85,
                'relevance': 0.85,
                'completeness': 0.65,
                'clarity': 0.90,
                'conciseness': 0.95,  # Very concise
                'latency': 0.8,
                'token_count': 200
            }
        },
        {
            'item_id': 'output_3',
            'item': 'Balanced response',
            'custom_values': {
                'accuracy': 0.88,
                'relevance': 0.92,
                'completeness': 0.78,
                'clarity': 0.85,
                'conciseness': 0.75,
                'latency': 1.5,
                'token_count': 450
            }
        },
        {
            'item_id': 'output_4',
            'item': 'Fast but less accurate',
            'custom_values': {
                'accuracy': 0.70,
                'relevance': 0.75,
                'completeness': 0.60,
                'clarity': 0.75,
                'conciseness': 0.85,
                'latency': 0.5,
                'token_count': 150
            }
        }
    ]
    # Evaluate all outputs
    scores = evaluator.evaluate_batch(outputs)
    print("\nEvaluation Results:")
    print("-" * 60)
    for score in sorted(scores, key=lambda x: x.rank):
        print(f"\n{score.item_id} (Rank #{score.rank})")
        print(f"  Aggregate Score: {score.aggregate_score:.3f}")
        print(f"  Pareto Optimal: {score.is_pareto_optimal}")
        print("  Criterion Scores:")
        for criterion_name, criterion_score in score.criterion_scores.items():
            print(f"    {criterion_name}: {criterion_score.normalized_score:.3f} "
                  f"(weighted: {criterion_score.weighted_score:.3f})")
    print("\n" + "=" * 60)
    print("3. Best Output Selection")
    print("=" * 60)
    best = evaluator.select_best(outputs, top_n=1)[0]
    print(f"\nBest Output: {best.item_id}")
    print(f"Aggregate Score: {best.aggregate_score:.3f}")
    print(f"Item: {best.metadata['item']}")
    print("\nTop Criteria Scores:")
    sorted_criteria = sorted(
        best.criterion_scores.items(),
        key=lambda x: x[1].normalized_score,
        reverse=True
    )
    for criterion_name, criterion_score in sorted_criteria[:3]:
        print(f"  {criterion_name}: {criterion_score.normalized_score:.3f}")
        print(f"    {criterion_score.explanation}")
    print("\n" + "=" * 60)
    print("4. Trade-off Analysis")
    print("=" * 60)
    # Compare output_1 (detailed) vs output_2 (concise)
    tradeoff = evaluator.analyze_tradeoffs('output_1', 'output_2')
    print(f"\nComparing: output_1 vs output_2")
    print(f"\nImproved Criteria in output_2:")
    for criterion in tradeoff.improved_criteria:
        print(f"  - {criterion}")
    print(f"\nDegraded Criteria in output_2:")
    for criterion in tradeoff.degraded_criteria:
        print(f"  - {criterion}")
    print(f"\nNet Improvement: {tradeoff.net_improvement:+.3f}")
    print(f"Recommendation: {tradeoff.recommendation}")
    print("\n" + "=" * 60)
    print("5. Criterion Rankings")
    print("=" * 60)
    rankings = evaluator.get_criterion_rankings()
    print("\nBest performers by criterion:")
    for criterion_name, ranked_items in rankings.items():
        print(f"\n{criterion_name}:")
        for i, item_id in enumerate(ranked_items[:3], 1):
            score = next(s for s in scores if s.item_id == item_id)
            criterion_score = score.criterion_scores[criterion_name]
            print(f"  {i}. {item_id}: {criterion_score.normalized_score:.3f}")
    print("\n" + "=" * 60)
    print("6. Pareto Optimal Solutions")
    print("=" * 60)
    pareto_optimal = [s for s in scores if s.is_pareto_optimal]
    print(f"\nPareto Optimal Solutions: {len(pareto_optimal)}")
    for score in pareto_optimal:
        print(f"\n{score.item_id}:")
        print(f"  Aggregate Score: {score.aggregate_score:.3f}")
        print(f"  Not dominated by any other solution")
    non_pareto = [s for s in scores if not s.is_pareto_optimal]
    if non_pareto:
        print(f"\nDominated Solutions:")
        for score in non_pareto:
            print(f"\n{score.item_id}:")
            print(f"  Dominated by: {', '.join(score.dominated_by)}")
    print("\n" + "=" * 60)
    print("7. Overall Statistics")
    print("=" * 60)
    stats = evaluator.get_statistics()
    print(f"\nTotal Evaluations: {stats['total_evaluations']}")
    print(f"Aggregation Method: {stats['aggregation_method']}")
    print(f"Number of Criteria: {stats['num_criteria']}")
    print(f"\nAggregate Score Statistics:")
    agg_stats = stats['aggregate_score_stats']
    print(f"  Mean: {agg_stats['mean']:.3f}")
    print(f"  Median: {agg_stats['median']:.3f}")
    print(f"  Range: {agg_stats['min']:.3f} - {agg_stats['max']:.3f}")
    print(f"\nCriterion Statistics:")
    for criterion_name, crit_stats in stats['criterion_statistics'].items():
        print(f"\n  {criterion_name}:")
        print(f"    Mean: {crit_stats['mean']:.3f}")
        print(f"    Std Dev: {crit_stats['stdev']:.3f}")
        print(f"    Range: {crit_stats['min']:.3f} - {crit_stats['max']:.3f}")
    print(f"\nPareto Optimal: {stats['pareto_optimal_count']} "
          f"({stats['pareto_optimal_percentage']:.1f}%)")
    print("\n" + "=" * 60)
    print("Multi-Criteria Evaluation demonstration complete!")
    print("=" * 60)
if __name__ == "__main__":
    main()
