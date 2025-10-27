"""
Pattern 124: Interpretability Framework

This pattern implements model interpretation, feature importance analysis,
attention visualization, and decision boundary analysis.

Use Cases:
- Understanding model behavior
- Feature engineering guidance
- Model debugging and validation
- Stakeholder communication
- Regulatory compliance

Category: Explainability & Transparency (2/4 = 50%)
Complexity: Advanced
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime
import math
import random


class InterpretationMethod(Enum):
    """Methods for model interpretation."""
    FEATURE_IMPORTANCE = "feature_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    SHAP_VALUES = "shap_values"  # Simplified SHAP-like
    ATTENTION_WEIGHTS = "attention_weights"
    ACTIVATION_ANALYSIS = "activation_analysis"
    DECISION_BOUNDARY = "decision_boundary"


class ImportanceMethod(Enum):
    """Methods for calculating feature importance."""
    PERMUTATION = "permutation"
    GRADIENT_BASED = "gradient_based"
    MODEL_SPECIFIC = "model_specific"
    CORRELATION = "correlation"


@dataclass
class Feature:
    """Feature with metadata."""
    name: str
    value: Any
    feature_type: str  # numeric, categorical, text
    range: Optional[Tuple[float, float]] = None
    categories: Optional[List[str]] = None


@dataclass
class FeatureImportance:
    """Feature importance score."""
    feature_name: str
    importance: float
    method: ImportanceMethod
    confidence: float = 1.0
    rank: Optional[int] = None
    
    def __lt__(self, other):
        """For sorting by importance."""
        return self.importance < other.importance


@dataclass
class PartialDependence:
    """Partial dependence of prediction on feature."""
    feature_name: str
    values: List[float]
    predictions: List[float]
    baseline: float


@dataclass
class AttentionWeights:
    """Attention weights for inputs."""
    input_tokens: List[str]
    attention_scores: List[float]
    layer: int
    head: Optional[int] = None


@dataclass
class InterpretationResult:
    """Result of interpretation analysis."""
    method: InterpretationMethod
    analysis: Dict[str, Any]
    visualizations: List[str]
    insights: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class FeatureImportanceAnalyzer:
    """Analyzes feature importance."""
    
    def __init__(self, prediction_fn: Callable[[Dict[str, Any]], float]):
        self.prediction_fn = prediction_fn
        self.baseline_predictions: Dict[str, float] = {}
    
    def analyze_importance(
        self,
        features: Dict[str, Any],
        method: ImportanceMethod = ImportanceMethod.PERMUTATION,
        n_iterations: int = 100
    ) -> List[FeatureImportance]:
        """Analyze feature importance."""
        if method == ImportanceMethod.PERMUTATION:
            return self._permutation_importance(features, n_iterations)
        elif method == ImportanceMethod.GRADIENT_BASED:
            return self._gradient_importance(features)
        elif method == ImportanceMethod.CORRELATION:
            return self._correlation_importance(features)
        else:
            return self._permutation_importance(features, n_iterations)
    
    def _permutation_importance(
        self,
        features: Dict[str, Any],
        n_iterations: int
    ) -> List[FeatureImportance]:
        """Calculate importance via permutation."""
        baseline_pred = self.prediction_fn(features)
        importances = {}
        
        for feature_name in features.keys():
            importance_scores = []
            
            for _ in range(n_iterations):
                # Permute this feature
                permuted_features = features.copy()
                
                # Simple permutation: add noise or swap
                if isinstance(features[feature_name], (int, float)):
                    original = float(features[feature_name])
                    noise = random.gauss(0, abs(original) * 0.2 + 0.1)
                    permuted_features[feature_name] = original + noise
                else:
                    # For non-numeric, use default
                    permuted_features[feature_name] = None
                
                # Measure impact
                permuted_pred = self.prediction_fn(permuted_features)
                impact = abs(baseline_pred - permuted_pred)
                importance_scores.append(impact)
            
            # Average impact
            avg_importance = sum(importance_scores) / len(importance_scores)
            std_importance = math.sqrt(
                sum((s - avg_importance)**2 for s in importance_scores) / len(importance_scores)
            )
            
            importances[feature_name] = FeatureImportance(
                feature_name=feature_name,
                importance=avg_importance,
                method=ImportanceMethod.PERMUTATION,
                confidence=1.0 / (std_importance + 0.01)
            )
        
        # Rank features
        ranked = sorted(importances.values(), reverse=True)
        for rank, imp in enumerate(ranked, 1):
            imp.rank = rank
        
        return ranked
    
    def _gradient_importance(
        self,
        features: Dict[str, Any]
    ) -> List[FeatureImportance]:
        """Calculate gradient-based importance."""
        importances = []
        baseline_pred = self.prediction_fn(features)
        
        for feature_name, value in features.items():
            if not isinstance(value, (int, float)):
                continue
            
            # Approximate gradient
            epsilon = abs(value) * 0.01 + 0.001
            features_plus = features.copy()
            features_plus[feature_name] = value + epsilon
            
            pred_plus = self.prediction_fn(features_plus)
            gradient = (pred_plus - baseline_pred) / epsilon
            
            importances.append(FeatureImportance(
                feature_name=feature_name,
                importance=abs(gradient),
                method=ImportanceMethod.GRADIENT_BASED,
                confidence=0.9
            ))
        
        # Rank
        importances.sort(reverse=True)
        for rank, imp in enumerate(importances, 1):
            imp.rank = rank
        
        return importances
    
    def _correlation_importance(
        self,
        features: Dict[str, Any]
    ) -> List[FeatureImportance]:
        """Simple correlation-based importance."""
        baseline_pred = self.prediction_fn(features)
        importances = []
        
        for feature_name, value in features.items():
            if not isinstance(value, (int, float)):
                continue
            
            # Correlation proxy: how much feature correlates with prediction
            correlation = abs(value * baseline_pred) / (abs(value) + abs(baseline_pred) + 1)
            
            importances.append(FeatureImportance(
                feature_name=feature_name,
                importance=correlation,
                method=ImportanceMethod.CORRELATION,
                confidence=0.7
            ))
        
        importances.sort(reverse=True)
        for rank, imp in enumerate(importances, 1):
            imp.rank = rank
        
        return importances


class PartialDependenceAnalyzer:
    """Analyzes partial dependence of predictions."""
    
    def __init__(self, prediction_fn: Callable[[Dict[str, Any]], float]):
        self.prediction_fn = prediction_fn
    
    def analyze(
        self,
        features: Dict[str, Any],
        target_feature: str,
        n_points: int = 20
    ) -> PartialDependence:
        """Analyze partial dependence for a feature."""
        if target_feature not in features:
            raise ValueError(f"Feature {target_feature} not found")
        
        baseline_pred = self.prediction_fn(features)
        original_value = features[target_feature]
        
        # Generate range of values
        if isinstance(original_value, (int, float)):
            min_val = original_value * 0.5
            max_val = original_value * 1.5
            values = [
                min_val + (max_val - min_val) * i / (n_points - 1)
                for i in range(n_points)
            ]
        else:
            # For non-numeric, can't do partial dependence easily
            values = [original_value]
        
        # Calculate predictions
        predictions = []
        for val in values:
            modified_features = features.copy()
            modified_features[target_feature] = val
            pred = self.prediction_fn(modified_features)
            predictions.append(pred)
        
        return PartialDependence(
            feature_name=target_feature,
            values=values,
            predictions=predictions,
            baseline=baseline_pred
        )


class AttentionAnalyzer:
    """Analyzes attention patterns."""
    
    def __init__(self):
        self.attention_patterns: List[AttentionWeights] = []
    
    def compute_attention(
        self,
        inputs: List[str],
        query: str,
        layer: int = 0
    ) -> AttentionWeights:
        """Compute attention weights for inputs."""
        # Simplified attention: based on similarity to query
        attention_scores = []
        
        for inp in inputs:
            # Simple similarity: word overlap
            query_words = set(query.lower().split())
            input_words = set(inp.lower().split())
            
            overlap = len(query_words & input_words)
            similarity = overlap / (len(query_words) + 0.1)
            
            # Add some noise for realism
            noise = random.gauss(0, 0.1)
            score = max(0, min(1, similarity + noise))
            attention_scores.append(score)
        
        # Normalize
        total = sum(attention_scores)
        if total > 0:
            attention_scores = [s / total for s in attention_scores]
        
        weights = AttentionWeights(
            input_tokens=inputs,
            attention_scores=attention_scores,
            layer=layer
        )
        
        self.attention_patterns.append(weights)
        return weights
    
    def visualize_attention(self, attention: AttentionWeights) -> str:
        """Create text visualization of attention."""
        lines = [f"Attention Weights (Layer {attention.layer}):"]
        
        # Sort by score
        sorted_items = sorted(
            zip(attention.input_tokens, attention.attention_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        for token, score in sorted_items:
            bar_length = int(score * 40)
            bar = "█" * bar_length
            lines.append(f"  {token:20s} {bar} {score:.3f}")
        
        return "\n".join(lines)


class DecisionBoundaryAnalyzer:
    """Analyzes decision boundaries."""
    
    def __init__(self, prediction_fn: Callable[[Dict[str, Any]], float]):
        self.prediction_fn = prediction_fn
    
    def analyze_boundary(
        self,
        features: Dict[str, Any],
        feature1: str,
        feature2: str,
        n_points: int = 10,
        decision_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Analyze decision boundary in 2D feature space."""
        if feature1 not in features or feature2 not in features:
            raise ValueError("Features not found")
        
        val1 = features[feature1]
        val2 = features[feature2]
        
        if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
            raise ValueError("Both features must be numeric")
        
        # Generate grid
        range1 = (val1 * 0.5, val1 * 1.5)
        range2 = (val2 * 0.5, val2 * 1.5)
        
        grid_predictions = []
        for i in range(n_points):
            row = []
            for j in range(n_points):
                v1 = range1[0] + (range1[1] - range1[0]) * i / (n_points - 1)
                v2 = range2[0] + (range2[1] - range2[0]) * j / (n_points - 1)
                
                modified_features = features.copy()
                modified_features[feature1] = v1
                modified_features[feature2] = v2
                
                pred = self.prediction_fn(modified_features)
                row.append(pred)
            
            grid_predictions.append(row)
        
        # Find boundary points
        boundary_points = []
        for i in range(n_points - 1):
            for j in range(n_points - 1):
                # Check if boundary crosses this cell
                values = [
                    grid_predictions[i][j],
                    grid_predictions[i+1][j],
                    grid_predictions[i][j+1],
                    grid_predictions[i+1][j+1]
                ]
                
                if (min(values) < decision_threshold < max(values)):
                    v1 = range1[0] + (range1[1] - range1[0]) * i / (n_points - 1)
                    v2 = range2[0] + (range2[1] - range2[0]) * j / (n_points - 1)
                    boundary_points.append((v1, v2))
        
        return {
            'feature1': feature1,
            'feature2': feature2,
            'grid_predictions': grid_predictions,
            'boundary_points': boundary_points,
            'grid_size': n_points
        }
    
    def visualize_boundary(self, boundary_analysis: Dict[str, Any]) -> str:
        """Create text visualization of decision boundary."""
        lines = [
            f"Decision Boundary Analysis:",
            f"Features: {boundary_analysis['feature1']} vs {boundary_analysis['feature2']}",
            f"Boundary points found: {len(boundary_analysis['boundary_points'])}"
        ]
        
        # Simple grid visualization
        grid = boundary_analysis['grid_predictions']
        n = boundary_analysis['grid_size']
        
        lines.append("\nPrediction Grid (█ = high, ░ = low):")
        for row in grid:
            vis_row = ""
            for val in row:
                if val > 0.7:
                    vis_row += "██"
                elif val > 0.5:
                    vis_row += "▓▓"
                elif val > 0.3:
                    vis_row += "▒▒"
                else:
                    vis_row += "░░"
            lines.append("  " + vis_row)
        
        return "\n".join(lines)


class InterpretabilityFramework:
    """Framework for model interpretability."""
    
    def __init__(self, prediction_fn: Callable[[Dict[str, Any]], float]):
        self.prediction_fn = prediction_fn
        self.feature_analyzer = FeatureImportanceAnalyzer(prediction_fn)
        self.pd_analyzer = PartialDependenceAnalyzer(prediction_fn)
        self.attention_analyzer = AttentionAnalyzer()
        self.boundary_analyzer = DecisionBoundaryAnalyzer(prediction_fn)
        
        # History
        self.interpretations: List[InterpretationResult] = []
    
    def interpret(
        self,
        features: Dict[str, Any],
        method: InterpretationMethod,
        **kwargs
    ) -> InterpretationResult:
        """Run interpretation analysis."""
        insights = []
        visualizations = []
        
        if method == InterpretationMethod.FEATURE_IMPORTANCE:
            importance_method = kwargs.get('importance_method', ImportanceMethod.PERMUTATION)
            importances = self.feature_analyzer.analyze_importance(features, importance_method)
            
            analysis = {
                'importances': [
                    {'feature': imp.feature_name, 'importance': imp.importance, 'rank': imp.rank}
                    for imp in importances
                ]
            }
            
            # Generate insights
            top_feature = importances[0]
            insights.append(
                f"Most important feature: {top_feature.feature_name} "
                f"(importance: {top_feature.importance:.3f})"
            )
            
            if len(importances) > 1:
                importance_ratio = importances[0].importance / (importances[1].importance + 0.001)
                if importance_ratio > 2:
                    insights.append(
                        f"{top_feature.feature_name} is significantly more important "
                        f"than other features ({importance_ratio:.1f}x)"
                    )
            
            # Visualization
            viz = self._visualize_feature_importance(importances)
            visualizations.append(viz)
        
        elif method == InterpretationMethod.PARTIAL_DEPENDENCE:
            target_feature = kwargs.get('target_feature')
            if not target_feature:
                raise ValueError("target_feature required for partial dependence")
            
            pd = self.pd_analyzer.analyze(features, target_feature)
            
            analysis = {
                'feature': pd.feature_name,
                'values': pd.values,
                'predictions': pd.predictions,
                'baseline': pd.baseline
            }
            
            # Analyze trend
            if len(pd.predictions) > 1:
                increasing = sum(
                    1 for i in range(len(pd.predictions)-1)
                    if pd.predictions[i+1] > pd.predictions[i]
                )
                
                if increasing > len(pd.predictions) * 0.7:
                    insights.append(
                        f"Prediction increases with {pd.feature_name} "
                        f"(positive relationship)"
                    )
                elif increasing < len(pd.predictions) * 0.3:
                    insights.append(
                        f"Prediction decreases with {pd.feature_name} "
                        f"(negative relationship)"
                    )
                else:
                    insights.append(
                        f"Prediction has non-linear relationship with {pd.feature_name}"
                    )
            
            viz = self._visualize_partial_dependence(pd)
            visualizations.append(viz)
        
        elif method == InterpretationMethod.ATTENTION_WEIGHTS:
            inputs = kwargs.get('inputs', [])
            query = kwargs.get('query', '')
            
            attention = self.attention_analyzer.compute_attention(inputs, query)
            
            analysis = {
                'tokens': attention.input_tokens,
                'scores': attention.attention_scores
            }
            
            # Top attended tokens
            top_indices = sorted(
                range(len(attention.attention_scores)),
                key=lambda i: attention.attention_scores[i],
                reverse=True
            )[:3]
            
            top_tokens = [attention.input_tokens[i] for i in top_indices]
            insights.append(f"Most attended tokens: {', '.join(top_tokens)}")
            
            viz = self.attention_analyzer.visualize_attention(attention)
            visualizations.append(viz)
        
        elif method == InterpretationMethod.DECISION_BOUNDARY:
            feature1 = kwargs.get('feature1')
            feature2 = kwargs.get('feature2')
            
            if not feature1 or not feature2:
                raise ValueError("feature1 and feature2 required for boundary analysis")
            
            boundary = self.boundary_analyzer.analyze_boundary(features, feature1, feature2)
            
            analysis = boundary
            
            insights.append(
                f"Found {len(boundary['boundary_points'])} boundary points "
                f"in {feature1}-{feature2} space"
            )
            
            viz = self.boundary_analyzer.visualize_boundary(boundary)
            visualizations.append(viz)
        
        else:
            analysis = {}
            insights.append(f"Method {method.value} not yet implemented")
        
        result = InterpretationResult(
            method=method,
            analysis=analysis,
            visualizations=visualizations,
            insights=insights
        )
        
        self.interpretations.append(result)
        return result
    
    def _visualize_feature_importance(self, importances: List[FeatureImportance]) -> str:
        """Visualize feature importance."""
        lines = ["Feature Importance:"]
        
        max_importance = importances[0].importance
        
        for imp in importances[:10]:  # Top 10
            bar_length = int((imp.importance / max_importance) * 40)
            bar = "█" * bar_length
            lines.append(
                f"  {imp.rank}. {imp.feature_name:20s} {bar} {imp.importance:.4f}"
            )
        
        return "\n".join(lines)
    
    def _visualize_partial_dependence(self, pd: PartialDependence) -> str:
        """Visualize partial dependence."""
        lines = [f"Partial Dependence: {pd.feature_name}"]
        
        # Simple ASCII plot
        min_pred = min(pd.predictions)
        max_pred = max(pd.predictions)
        range_pred = max_pred - min_pred
        
        if range_pred < 0.001:
            lines.append("  (constant prediction)")
            return "\n".join(lines)
        
        # Create plot
        height = 10
        for h in range(height, -1, -1):
            line = "  "
            threshold = min_pred + (range_pred * h / height)
            
            for pred in pd.predictions:
                if pred >= threshold:
                    line += "█"
                else:
                    line += " "
            
            lines.append(line)
        
        # X-axis
        lines.append("  " + "─" * len(pd.predictions))
        lines.append(f"  Value range: {pd.values[0]:.2f} to {pd.values[-1]:.2f}")
        
        return "\n".join(lines)
    
    def comprehensive_analysis(
        self,
        features: Dict[str, Any]
    ) -> Dict[InterpretationMethod, InterpretationResult]:
        """Run comprehensive interpretation analysis."""
        results = {}
        
        # Feature importance
        results[InterpretationMethod.FEATURE_IMPORTANCE] = self.interpret(
            features,
            InterpretationMethod.FEATURE_IMPORTANCE
        )
        
        # Partial dependence for top features
        if results[InterpretationMethod.FEATURE_IMPORTANCE].analysis['importances']:
            top_feature = results[InterpretationMethod.FEATURE_IMPORTANCE].analysis['importances'][0]
            
            if isinstance(features.get(top_feature['feature']), (int, float)):
                results[InterpretationMethod.PARTIAL_DEPENDENCE] = self.interpret(
                    features,
                    InterpretationMethod.PARTIAL_DEPENDENCE,
                    target_feature=top_feature['feature']
                )
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get interpretation statistics."""
        if not self.interpretations:
            return {}
        
        method_counts = {}
        for interp in self.interpretations:
            method = interp.method.value
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            'total_interpretations': len(self.interpretations),
            'methods_used': method_counts,
            'total_insights': sum(len(i.insights) for i in self.interpretations)
        }


def demonstrate_interpretability():
    """Demonstrate the Interpretability Framework."""
    print("=" * 60)
    print("Interpretability Framework Demonstration")
    print("=" * 60)
    
    # Define a sample prediction function
    def loan_prediction(features: Dict[str, Any]) -> float:
        """Simple loan approval prediction."""
        score = 0.0
        
        credit = features.get('credit_score', 0)
        income = features.get('annual_income', 0)
        debt_ratio = features.get('debt_to_income_ratio', 0)
        
        # Simple scoring
        score += (credit - 600) / 200 * 0.5  # Credit score weight
        score += min(income / 100000, 1.0) * 0.3  # Income weight
        score += (0.5 - debt_ratio) * 0.2  # Debt ratio weight (lower is better)
        
        return max(0, min(1, score))
    
    # Create framework
    framework = InterpretabilityFramework(loan_prediction)
    
    # Sample features
    features = {
        'credit_score': 720,
        'annual_income': 75000,
        'debt_to_income_ratio': 0.30,
        'employment_years': 5,
        'loan_amount': 200000
    }
    
    print("\n1. SAMPLE LOAN APPLICATION")
    print("-" * 60)
    for feature, value in features.items():
        print(f"   {feature}: {value}")
    
    baseline_pred = loan_prediction(features)
    print(f"\n   Predicted approval score: {baseline_pred:.3f}")
    
    # Feature importance
    print("\n2. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 60)
    
    importance_result = framework.interpret(
        features,
        InterpretationMethod.FEATURE_IMPORTANCE,
        importance_method=ImportanceMethod.PERMUTATION
    )
    
    print(importance_result.visualizations[0])
    print("\n   Insights:")
    for insight in importance_result.insights:
        print(f"   - {insight}")
    
    # Partial dependence
    print("\n3. PARTIAL DEPENDENCE ANALYSIS")
    print("-" * 60)
    
    pd_result = framework.interpret(
        features,
        InterpretationMethod.PARTIAL_DEPENDENCE,
        target_feature='credit_score'
    )
    
    print(pd_result.visualizations[0])
    print("\n   Insights:")
    for insight in pd_result.insights:
        print(f"   - {insight}")
    
    # Attention analysis
    print("\n4. ATTENTION ANALYSIS")
    print("-" * 60)
    
    inputs = [
        "high credit score",
        "stable employment",
        "low debt ratio",
        "sufficient income",
        "previous loans"
    ]
    
    attention_result = framework.interpret(
        features,
        InterpretationMethod.ATTENTION_WEIGHTS,
        inputs=inputs,
        query="loan approval factors"
    )
    
    print(attention_result.visualizations[0])
    print("\n   Insights:")
    for insight in attention_result.insights:
        print(f"   - {insight}")
    
    # Decision boundary
    print("\n5. DECISION BOUNDARY ANALYSIS")
    print("-" * 60)
    
    boundary_result = framework.interpret(
        features,
        InterpretationMethod.DECISION_BOUNDARY,
        feature1='credit_score',
        feature2='annual_income'
    )
    
    print(boundary_result.visualizations[0])
    print("\n   Insights:")
    for insight in boundary_result.insights:
        print(f"   - {insight}")
    
    # Statistics
    print("\n6. INTERPRETATION STATISTICS")
    print("-" * 60)
    
    stats = framework.get_statistics()
    print(f"   Total Interpretations: {stats['total_interpretations']}")
    print(f"   Total Insights Generated: {stats['total_insights']}")
    print(f"   Methods Used:")
    for method, count in stats['methods_used'].items():
        print(f"   - {method}: {count}")
    
    print("\n" + "=" * 60)
    print("Explainability & Transparency Category: 50% Complete!")
    print("Pattern 124 advances the category!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_interpretability()
