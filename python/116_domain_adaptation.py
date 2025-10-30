"""
Agentic Design Pattern: Domain Adaptation Agent

This pattern implements an agent that can adapt knowledge and skills from one domain
(source) to another domain (target) through feature alignment, adversarial training,
and transfer learning techniques.

Key Components:
1. DomainFeatureExtractor - Extracts domain-invariant features
2. DomainClassifier - Distinguishes between source and target domains
3. AdversarialAdapter - Adversarial training for domain confusion
4. TransferValidator - Validates transfer quality
5. DomainAdaptationAgent - Main orchestrator

Features:
- Cross-domain feature alignment
- Adversarial domain adaptation
- Progressive transfer learning
- Domain distance measurement
- Adaptation quality monitoring
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from collections import defaultdict
import random
import math


class DomainType(Enum):
    """Types of domains for adaptation."""
    SOURCE = "source"
    TARGET = "target"
    AUXILIARY = "auxiliary"


class AdaptationStrategy(Enum):
    """Strategies for domain adaptation."""
    ADVERSARIAL = "adversarial"
    FEATURE_ALIGNMENT = "feature_alignment"
    SELF_TRAINING = "self_training"
    JOINT_TRAINING = "joint_training"
    PROGRESSIVE = "progressive"


class TransferType(Enum):
    """Types of knowledge transfer."""
    FEATURE_SPACE = "feature_space"
    LABEL_SPACE = "label_space"
    MODEL_PARAMETERS = "model_parameters"
    REPRESENTATION = "representation"


@dataclass
class DomainSample:
    """Represents a sample from a domain."""
    sample_id: str
    domain: DomainType
    features: List[float]
    label: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainStatistics:
    """Statistical properties of a domain."""
    domain: DomainType
    sample_count: int
    feature_means: List[float]
    feature_variances: List[float]
    label_distribution: Dict[str, int]
    density_estimate: float = 0.0


@dataclass
class AdaptationResult:
    """Results of domain adaptation process."""
    strategy: AdaptationStrategy
    source_domain: DomainType
    target_domain: DomainType
    adapted_features: List[List[float]]
    alignment_score: float
    transfer_quality: float
    domain_confusion: float  # Higher = better domain invariance
    iterations: int
    converged: bool


@dataclass
class TransferValidation:
    """Validation results for transfer learning."""
    transfer_type: TransferType
    source_performance: float
    target_performance: float
    transfer_ratio: float  # target_perf / source_perf
    domain_distance: float
    confidence: float
    warnings: List[str] = field(default_factory=list)


class DomainFeatureExtractor:
    """Extracts domain-invariant features."""
    
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 64):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.weights_layer1: List[List[float]] = []
        self.weights_layer2: List[List[float]] = []
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights (simple random initialization)."""
        # Layer 1: input_dim -> hidden_dim (will be set dynamically)
        # Layer 2: hidden_dim -> feature_dim
        for _ in range(self.hidden_dim):
            self.weights_layer2.append([
                random.gauss(0, 0.1) for _ in range(self.feature_dim)
            ])
    
    def extract_features(self, raw_features: List[float]) -> List[float]:
        """
        Extract domain-invariant features.
        
        Args:
            raw_features: Raw input features
            
        Returns:
            Extracted domain-invariant features
        """
        # Initialize layer 1 weights if needed
        if not self.weights_layer1:
            input_dim = len(raw_features)
            for _ in range(input_dim):
                self.weights_layer1.append([
                    random.gauss(0, 0.1) for _ in range(self.hidden_dim)
                ])
        
        # Simple two-layer neural network simulation
        # Layer 1: raw_features -> hidden
        hidden = [0.0] * self.hidden_dim
        for i in range(len(raw_features)):
            for j in range(self.hidden_dim):
                if i < len(self.weights_layer1):
                    hidden[j] += raw_features[i] * self.weights_layer1[i][j]
        
        # Activation (ReLU)
        hidden = [max(0, h) for h in hidden]
        
        # Layer 2: hidden -> features
        features = [0.0] * self.feature_dim
        for i in range(self.hidden_dim):
            for j in range(self.feature_dim):
                features[j] += hidden[i] * self.weights_layer2[i][j]
        
        # Normalize
        norm = math.sqrt(sum(f * f for f in features)) + 1e-8
        features = [f / norm for f in features]
        
        return features
    
    def update_weights(self, gradient: List[float], learning_rate: float = 0.01):
        """Update weights based on gradient (simplified)."""
        # Update layer 2 weights (simplified gradient descent)
        for i in range(self.hidden_dim):
            for j in range(self.feature_dim):
                if j < len(gradient):
                    self.weights_layer2[i][j] -= learning_rate * gradient[j]


class DomainClassifier:
    """Classifies samples into domains (for adversarial training)."""
    
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.weights: List[float] = [random.gauss(0, 0.1) for _ in range(feature_dim)]
        self.bias: float = 0.0
        
    def predict_domain(self, features: List[float]) -> Tuple[float, DomainType]:
        """
        Predict domain probability.
        
        Args:
            features: Feature vector
            
        Returns:
            Tuple of (probability, predicted_domain)
        """
        # Linear classifier with sigmoid
        score = self.bias
        for i, f in enumerate(features):
            if i < len(self.weights):
                score += f * self.weights[i]
        
        # Sigmoid activation
        prob_source = 1.0 / (1.0 + math.exp(-score))
        
        predicted = DomainType.SOURCE if prob_source > 0.5 else DomainType.TARGET
        confidence = max(prob_source, 1 - prob_source)
        
        return confidence, predicted
    
    def update(self, features: List[float], true_domain: DomainType, learning_rate: float = 0.01):
        """Update classifier weights."""
        # Get prediction
        prob_source, _ = self.predict_domain(features)
        
        # Compute error
        target = 1.0 if true_domain == DomainType.SOURCE else 0.0
        error = target - prob_source
        
        # Update weights
        for i in range(len(features)):
            if i < len(self.weights):
                self.weights[i] += learning_rate * error * features[i]
        self.bias += learning_rate * error


class AdversarialAdapter:
    """Performs adversarial domain adaptation."""
    
    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.feature_extractor = DomainFeatureExtractor(feature_dim)
        self.domain_classifier = DomainClassifier(feature_dim)
        self.adaptation_history: List[Dict[str, float]] = []
        
    def adapt_features(
        self,
        source_samples: List[DomainSample],
        target_samples: List[DomainSample],
        max_iterations: int = 100,
        convergence_threshold: float = 0.01
    ) -> AdaptationResult:
        """
        Perform adversarial domain adaptation.
        
        The goal is to learn features where:
        1. Task classifier performs well (feature extractor helps task)
        2. Domain classifier performs poorly (features are domain-invariant)
        
        Args:
            source_samples: Labeled source domain samples
            target_samples: Target domain samples (may be unlabeled)
            max_iterations: Maximum training iterations
            convergence_threshold: Convergence threshold for domain confusion
            
        Returns:
            AdaptationResult with adapted features
        """
        adapted_features_source = []
        adapted_features_target = []
        
        for iteration in range(max_iterations):
            # Extract features from both domains
            source_features = [
                self.feature_extractor.extract_features(s.features)
                for s in source_samples
            ]
            target_features = [
                self.feature_extractor.extract_features(s.features)
                for s in target_samples
            ]
            
            # Train domain classifier to distinguish domains
            # (We want to maximize its ability to distinguish)
            for features, domain in zip(
                source_features + target_features,
                [DomainType.SOURCE] * len(source_features) + [DomainType.TARGET] * len(target_features)
            ):
                self.domain_classifier.update(features, domain, learning_rate=0.01)
            
            # Evaluate domain confusion
            domain_confusion = self._compute_domain_confusion(source_features, target_features)
            
            # Update feature extractor to confuse domain classifier
            # (We want to minimize its ability to distinguish - adversarial)
            gradient = self._compute_adversarial_gradient(source_features, target_features)
            self.feature_extractor.update_weights(gradient, learning_rate=0.001)
            
            # Track progress
            self.adaptation_history.append({
                'iteration': iteration,
                'domain_confusion': domain_confusion,
                'alignment_score': self._compute_alignment_score(source_features, target_features)
            })
            
            # Check convergence (high domain confusion means good adaptation)
            if domain_confusion > (1.0 - convergence_threshold):
                adapted_features_source = source_features
                adapted_features_target = target_features
                break
        
        # Final feature extraction
        if not adapted_features_source:
            adapted_features_source = [
                self.feature_extractor.extract_features(s.features)
                for s in source_samples
            ]
            adapted_features_target = [
                self.feature_extractor.extract_features(s.features)
                for s in target_samples
            ]
        
        # Compute final metrics
        alignment_score = self._compute_alignment_score(adapted_features_source, adapted_features_target)
        domain_confusion = self._compute_domain_confusion(adapted_features_source, adapted_features_target)
        transfer_quality = (alignment_score + domain_confusion) / 2.0
        
        return AdaptationResult(
            strategy=AdaptationStrategy.ADVERSARIAL,
            source_domain=DomainType.SOURCE,
            target_domain=DomainType.TARGET,
            adapted_features=adapted_features_source + adapted_features_target,
            alignment_score=alignment_score,
            transfer_quality=transfer_quality,
            domain_confusion=domain_confusion,
            iterations=iteration + 1,
            converged=(domain_confusion > (1.0 - convergence_threshold))
        )
    
    def _compute_domain_confusion(self, source_features: List[List[float]], 
                                   target_features: List[List[float]]) -> float:
        """
        Compute how confused the domain classifier is.
        Higher values = better (more domain-invariant features).
        """
        total_error = 0.0
        total_samples = 0
        
        # Check source samples (should predict SOURCE but we want confusion)
        for features in source_features:
            prob, pred = self.domain_classifier.predict_domain(features)
            # Low confidence or wrong prediction = good (domain confusion)
            if pred == DomainType.SOURCE:
                total_error += (1.0 - prob)  # Reward low confidence
            else:
                total_error += 1.0  # Reward wrong prediction
            total_samples += 1
        
        # Check target samples
        for features in target_features:
            prob, pred = self.domain_classifier.predict_domain(features)
            if pred == DomainType.TARGET:
                total_error += (1.0 - prob)
            else:
                total_error += 1.0
            total_samples += 1
        
        return total_error / max(total_samples, 1)
    
    def _compute_alignment_score(self, source_features: List[List[float]],
                                  target_features: List[List[float]]) -> float:
        """Compute feature distribution alignment (lower distance = better)."""
        if not source_features or not target_features:
            return 0.0
        
        # Compute mean features for each domain
        source_mean = [
            sum(features[i] for features in source_features) / len(source_features)
            for i in range(len(source_features[0]))
        ]
        target_mean = [
            sum(features[i] for features in target_features) / len(target_features)
            for i in range(len(target_features[0]))
        ]
        
        # Euclidean distance between means
        distance = math.sqrt(sum((s - t) ** 2 for s, t in zip(source_mean, target_mean)))
        
        # Convert to similarity score (0 to 1)
        alignment = 1.0 / (1.0 + distance)
        return alignment
    
    def _compute_adversarial_gradient(self, source_features: List[List[float]],
                                      target_features: List[List[float]]) -> List[float]:
        """Compute gradient for adversarial training (simplified)."""
        # Simplified gradient computation
        # Real implementation would use backpropagation
        gradient = [0.0] * self.feature_dim
        
        for features in source_features + target_features:
            prob, _ = self.domain_classifier.predict_domain(features)
            # Gradient encourages domain confusion
            for i, f in enumerate(features):
                if i < self.feature_dim:
                    gradient[i] += f * (0.5 - prob)  # Push towards 0.5 probability
        
        # Normalize
        norm = math.sqrt(sum(g * g for g in gradient)) + 1e-8
        gradient = [g / norm for g in gradient]
        
        return gradient


class TransferValidator:
    """Validates quality of domain transfer."""
    
    def __init__(self):
        self.validation_history: List[TransferValidation] = []
        
    def validate_transfer(
        self,
        source_samples: List[DomainSample],
        target_samples: List[DomainSample],
        adapted_features: List[List[float]],
        transfer_type: TransferType = TransferType.FEATURE_SPACE
    ) -> TransferValidation:
        """
        Validate transfer learning quality.
        
        Args:
            source_samples: Source domain samples
            target_samples: Target domain samples
            adapted_features: Adapted feature representations
            transfer_type: Type of transfer being validated
            
        Returns:
            TransferValidation with quality metrics
        """
        warnings = []
        
        # Compute source domain performance (simplified - based on label consistency)
        source_performance = self._estimate_performance(
            source_samples[:len(source_samples)],
            adapted_features[:len(source_samples)]
        )
        
        # Compute target domain performance
        target_performance = self._estimate_performance(
            target_samples,
            adapted_features[len(source_samples):]
        )
        
        # Compute transfer ratio
        transfer_ratio = target_performance / max(source_performance, 0.01)
        
        # Compute domain distance
        domain_distance = self._compute_domain_distance(
            adapted_features[:len(source_samples)],
            adapted_features[len(source_samples):]
        )
        
        # Generate warnings
        if transfer_ratio < 0.5:
            warnings.append("Low transfer ratio - significant performance drop")
        if domain_distance > 0.5:
            warnings.append("High domain distance - domains may be too different")
        if target_performance < 0.3:
            warnings.append("Low target performance - consider more adaptation")
        
        # Compute confidence
        confidence = min(1.0, (transfer_ratio + (1.0 - domain_distance)) / 2.0)
        
        validation = TransferValidation(
            transfer_type=transfer_type,
            source_performance=source_performance,
            target_performance=target_performance,
            transfer_ratio=transfer_ratio,
            domain_distance=domain_distance,
            confidence=confidence,
            warnings=warnings
        )
        
        self.validation_history.append(validation)
        return validation
    
    def _estimate_performance(self, samples: List[DomainSample], 
                             features: List[List[float]]) -> float:
        """Estimate performance based on feature quality (simplified)."""
        if not samples or not features:
            return 0.0
        
        # Measure feature consistency and separation
        # Higher variance = better separation = better performance
        total_variance = 0.0
        for dim in range(len(features[0])):
            values = [f[dim] for f in features]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            total_variance += variance
        
        # Normalize to 0-1
        performance = min(1.0, total_variance)
        return performance
    
    def _compute_domain_distance(self, source_features: List[List[float]],
                                 target_features: List[List[float]]) -> float:
        """Compute statistical distance between domains."""
        if not source_features or not target_features:
            return 1.0
        
        # Maximum Mean Discrepancy (MMD) - simplified
        source_mean = [
            sum(f[i] for f in source_features) / len(source_features)
            for i in range(len(source_features[0]))
        ]
        target_mean = [
            sum(f[i] for f in target_features) / len(target_features)
            for i in range(len(target_features[0]))
        ]
        
        distance = math.sqrt(sum((s - t) ** 2 for s, t in zip(source_mean, target_mean)))
        return min(1.0, distance)


class DomainAdaptationAgent:
    """
    Main agent for domain adaptation.
    
    Orchestrates the adaptation process including:
    - Feature extraction and alignment
    - Adversarial training
    - Transfer validation
    - Progressive adaptation
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        adaptation_strategy: AdaptationStrategy = AdaptationStrategy.ADVERSARIAL
    ):
        self.feature_dim = feature_dim
        self.adaptation_strategy = adaptation_strategy
        
        # Components
        self.adversarial_adapter = AdversarialAdapter(feature_dim)
        self.transfer_validator = TransferValidator()
        
        # State
        self.source_samples: List[DomainSample] = []
        self.target_samples: List[DomainSample] = []
        self.domain_statistics: Dict[DomainType, DomainStatistics] = {}
        self.adaptation_results: List[AdaptationResult] = []
        self.current_adaptation: Optional[AdaptationResult] = None
        
    def add_source_samples(self, samples: List[DomainSample]):
        """Add source domain samples."""
        self.source_samples.extend(samples)
        self._update_domain_statistics(DomainType.SOURCE)
        
    def add_target_samples(self, samples: List[DomainSample]):
        """Add target domain samples."""
        self.target_samples.extend(samples)
        self._update_domain_statistics(DomainType.TARGET)
        
    def adapt_domains(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 0.01
    ) -> AdaptationResult:
        """
        Perform domain adaptation.
        
        Args:
            max_iterations: Maximum adaptation iterations
            convergence_threshold: Convergence threshold
            
        Returns:
            AdaptationResult with adaptation metrics
        """
        if not self.source_samples:
            raise ValueError("No source samples available for adaptation")
        if not self.target_samples:
            raise ValueError("No target samples available for adaptation")
        
        print(f"\n{'='*80}")
        print(f"🔄 Starting Domain Adaptation: {len(self.source_samples)} source → {len(self.target_samples)} target samples")
        print(f"Strategy: {self.adaptation_strategy.value}")
        print(f"{'='*80}\n")
        
        # Perform adaptation based on strategy
        if self.adaptation_strategy == AdaptationStrategy.ADVERSARIAL:
            result = self.adversarial_adapter.adapt_features(
                self.source_samples,
                self.target_samples,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold
            )
        else:
            # For other strategies, use simplified feature alignment
            result = self._simple_feature_alignment()
        
        self.current_adaptation = result
        self.adaptation_results.append(result)
        
        print(f"✓ Adaptation Complete!")
        print(f"  Iterations: {result.iterations}")
        print(f"  Alignment Score: {result.alignment_score:.3f}")
        print(f"  Domain Confusion: {result.domain_confusion:.3f}")
        print(f"  Transfer Quality: {result.transfer_quality:.3f}")
        print(f"  Converged: {result.converged}")
        
        return result
    
    def validate_transfer(self, transfer_type: TransferType = TransferType.FEATURE_SPACE) -> TransferValidation:
        """Validate the current domain adaptation."""
        if not self.current_adaptation:
            raise ValueError("No adaptation to validate - run adapt_domains first")
        
        print(f"\n{'='*80}")
        print(f"🔍 Validating Transfer: {transfer_type.value}")
        print(f"{'='*80}\n")
        
        validation = self.transfer_validator.validate_transfer(
            self.source_samples,
            self.target_samples,
            self.current_adaptation.adapted_features,
            transfer_type
        )
        
        print(f"📊 Validation Results:")
        print(f"  Source Performance: {validation.source_performance:.3f}")
        print(f"  Target Performance: {validation.target_performance:.3f}")
        print(f"  Transfer Ratio: {validation.transfer_ratio:.3f}")
        print(f"  Domain Distance: {validation.domain_distance:.3f}")
        print(f"  Confidence: {validation.confidence:.3f}")
        
        if validation.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in validation.warnings:
                print(f"  • {warning}")
        
        return validation
    
    def predict_target(self, target_sample: DomainSample) -> Tuple[str, float]:
        """
        Predict label for target domain sample using adapted features.
        
        Args:
            target_sample: Sample from target domain
            
        Returns:
            Tuple of (predicted_label, confidence)
        """
        if not self.current_adaptation:
            raise ValueError("No adaptation available - run adapt_domains first")
        
        # Extract features
        target_features = self.adversarial_adapter.feature_extractor.extract_features(
            target_sample.features
        )
        
        # Find nearest neighbor in source domain (simplified prediction)
        best_label = "unknown"
        best_similarity = -1.0
        
        for i, source_sample in enumerate(self.source_samples):
            if i >= len(self.current_adaptation.adapted_features):
                break
            if not source_sample.label:
                continue
            
            source_features = self.current_adaptation.adapted_features[i]
            
            # Compute cosine similarity
            similarity = sum(s * t for s, t in zip(source_features, target_features))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_label = source_sample.label
        
        confidence = max(0.0, min(1.0, best_similarity))
        return best_label, confidence
    
    def get_domain_shift_analysis(self) -> Dict[str, Any]:
        """Analyze the domain shift between source and target."""
        if DomainType.SOURCE not in self.domain_statistics or \
           DomainType.TARGET not in self.domain_statistics:
            return {"error": "Insufficient domain statistics"}
        
        source_stats = self.domain_statistics[DomainType.SOURCE]
        target_stats = self.domain_statistics[DomainType.TARGET]
        
        # Compute feature distribution differences
        mean_shift = math.sqrt(sum(
            (s - t) ** 2
            for s, t in zip(source_stats.feature_means, target_stats.feature_means)
        ))
        
        variance_shift = math.sqrt(sum(
            (s - t) ** 2
            for s, t in zip(source_stats.feature_variances, target_stats.feature_variances)
        ))
        
        # Compute label distribution difference (if available)
        label_shift = self._compute_label_shift(
            source_stats.label_distribution,
            target_stats.label_distribution
        )
        
        return {
            'mean_shift': mean_shift,
            'variance_shift': variance_shift,
            'label_shift': label_shift,
            'overall_shift': (mean_shift + variance_shift + label_shift) / 3.0,
            'source_samples': source_stats.sample_count,
            'target_samples': target_stats.sample_count
        }
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of all adaptation attempts."""
        if not self.adaptation_results:
            return {"message": "No adaptations performed yet"}
        
        return {
            'total_adaptations': len(self.adaptation_results),
            'average_quality': sum(r.transfer_quality for r in self.adaptation_results) / len(self.adaptation_results),
            'average_iterations': sum(r.iterations for r in self.adaptation_results) / len(self.adaptation_results),
            'convergence_rate': sum(1 for r in self.adaptation_results if r.converged) / len(self.adaptation_results),
            'best_quality': max(r.transfer_quality for r in self.adaptation_results),
            'validations_performed': len(self.transfer_validator.validation_history),
            'current_strategy': self.adaptation_strategy.value
        }
    
    def _update_domain_statistics(self, domain: DomainType):
        """Update statistics for a domain."""
        samples = self.source_samples if domain == DomainType.SOURCE else self.target_samples
        
        if not samples:
            return
        
        # Compute feature statistics
        feature_dim = len(samples[0].features)
        feature_means = [0.0] * feature_dim
        feature_variances = [0.0] * feature_dim
        
        for sample in samples:
            for i, f in enumerate(sample.features):
                feature_means[i] += f
        
        for i in range(feature_dim):
            feature_means[i] /= len(samples)
        
        for sample in samples:
            for i, f in enumerate(sample.features):
                feature_variances[i] += (f - feature_means[i]) ** 2
        
        for i in range(feature_dim):
            feature_variances[i] /= len(samples)
        
        # Label distribution
        label_distribution: Dict[str, int] = defaultdict(int)
        for sample in samples:
            if sample.label:
                label_distribution[sample.label] += 1
        
        self.domain_statistics[domain] = DomainStatistics(
            domain=domain,
            sample_count=len(samples),
            feature_means=feature_means,
            feature_variances=feature_variances,
            label_distribution=dict(label_distribution)
        )
    
    def _compute_label_shift(self, source_dist: Dict[str, int], 
                            target_dist: Dict[str, int]) -> float:
        """Compute difference in label distributions."""
        all_labels = set(source_dist.keys()) | set(target_dist.keys())
        
        if not all_labels:
            return 0.0
        
        source_total = sum(source_dist.values())
        target_total = sum(target_dist.values())
        
        if source_total == 0 or target_total == 0:
            return 1.0
        
        # KL divergence (simplified)
        divergence = 0.0
        for label in all_labels:
            p = source_dist.get(label, 0) / source_total
            q = target_dist.get(label, 0) / target_total
            
            if p > 0:
                if q > 0:
                    divergence += p * math.log(p / q)
                else:
                    divergence += p * math.log(p / 0.01)  # Smoothing
        
        return min(1.0, divergence)
    
    def _simple_feature_alignment(self) -> AdaptationResult:
        """Perform simple feature alignment without adversarial training."""
        # Extract features
        source_features = [
            self.adversarial_adapter.feature_extractor.extract_features(s.features)
            for s in self.source_samples
        ]
        target_features = [
            self.adversarial_adapter.feature_extractor.extract_features(s.features)
            for s in self.target_samples
        ]
        
        # Compute alignment
        alignment_score = self.adversarial_adapter._compute_alignment_score(
            source_features, target_features
        )
        
        return AdaptationResult(
            strategy=self.adaptation_strategy,
            source_domain=DomainType.SOURCE,
            target_domain=DomainType.TARGET,
            adapted_features=source_features + target_features,
            alignment_score=alignment_score,
            transfer_quality=alignment_score,
            domain_confusion=0.5,
            iterations=1,
            converged=True
        )


# Demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("DOMAIN ADAPTATION AGENT DEMONSTRATION")
    print("=" * 80)
    
    # Create agent
    agent = DomainAdaptationAgent(
        feature_dim=32,
        adaptation_strategy=AdaptationStrategy.ADVERSARIAL
    )
    
    # Scenario: Adapting from Product Reviews (source) to Movie Reviews (target)
    print("\n📝 Scenario: Product Reviews → Movie Reviews")
    print("   Source: E-commerce product reviews (labeled)")
    print("   Target: Movie reviews (unlabeled)")
    print("   Goal: Transfer sentiment classification")
    
    # Create source domain samples (product reviews - labeled)
    source_samples = []
    print("\n📦 Adding Source Domain Samples (Product Reviews):")
    
    # Positive product reviews
    for i in range(15):
        features = [
            random.gauss(0.7, 0.2),  # High positive sentiment
            random.gauss(0.3, 0.15), # Low negative sentiment
            random.gauss(0.6, 0.2),  # Product quality
            random.gauss(0.5, 0.2),  # Price satisfaction
        ]
        sample = DomainSample(
            sample_id=f"product_pos_{i}",
            domain=DomainType.SOURCE,
            features=features,
            label="positive",
            confidence=0.9,
            metadata={'category': 'electronics', 'rating': random.randint(4, 5)}
        )
        source_samples.append(sample)
    
    # Negative product reviews
    for i in range(15):
        features = [
            random.gauss(0.2, 0.15), # Low positive sentiment
            random.gauss(0.8, 0.2),  # High negative sentiment
            random.gauss(0.3, 0.2),  # Product quality
            random.gauss(0.2, 0.15), # Price satisfaction
        ]
        sample = DomainSample(
            sample_id=f"product_neg_{i}",
            domain=DomainType.SOURCE,
            features=features,
            label="negative",
            confidence=0.9,
            metadata={'category': 'electronics', 'rating': random.randint(1, 2)}
        )
        source_samples.append(sample)
    
    agent.add_source_samples(source_samples)
    print(f"✓ Added {len(source_samples)} labeled product reviews")
    
    # Create target domain samples (movie reviews - unlabeled)
    target_samples = []
    print("\n🎬 Adding Target Domain Samples (Movie Reviews):")
    
    # Movie reviews (similar sentiment but different features)
    for i in range(20):
        # Movies have different feature distributions
        is_positive = i < 10
        features = [
            random.gauss(0.7 if is_positive else 0.3, 0.25),  # Sentiment
            random.gauss(0.3 if is_positive else 0.7, 0.25),  # Negative sentiment
            random.gauss(0.6 if is_positive else 0.4, 0.2),   # Plot quality
            random.gauss(0.6 if is_positive else 0.3, 0.2),   # Entertainment
        ]
        sample = DomainSample(
            sample_id=f"movie_{i}",
            domain=DomainType.TARGET,
            features=features,
            label="positive" if is_positive else "negative",  # Hidden labels for validation
            confidence=0.0,
            metadata={'genre': random.choice(['action', 'drama', 'comedy']), 'year': 2023}
        )
        target_samples.append(sample)
    
    agent.add_target_samples(target_samples)
    print(f"✓ Added {len(target_samples)} unlabeled movie reviews")
    
    # Analyze domain shift
    print("\n" + "="*80)
    print("📊 DOMAIN SHIFT ANALYSIS")
    print("="*80)
    
    shift_analysis = agent.get_domain_shift_analysis()
    print(f"Mean Shift: {shift_analysis['mean_shift']:.4f}")
    print(f"Variance Shift: {shift_analysis['variance_shift']:.4f}")
    print(f"Label Shift: {shift_analysis['label_shift']:.4f}")
    print(f"Overall Shift: {shift_analysis['overall_shift']:.4f}")
    
    # Perform domain adaptation
    adaptation_result = agent.adapt_domains(
        max_iterations=50,
        convergence_threshold=0.02
    )
    
    # Validate transfer
    validation = agent.validate_transfer(TransferType.FEATURE_SPACE)
    
    # Test predictions on target domain
    print("\n" + "="*80)
    print("🎯 TESTING PREDICTIONS ON TARGET DOMAIN")
    print("="*80)
    
    correct = 0
    total = 0
    
    for i, sample in enumerate(target_samples[:8]):
        predicted_label, confidence = agent.predict_target(sample)
        true_label = sample.label
        is_correct = predicted_label == true_label
        
        if is_correct:
            correct += 1
        total += 1
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Movie {i+1}: Predicted={predicted_label:8s} (conf={confidence:.2f}) | True={true_label:8s}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n📈 Accuracy: {correct}/{total} = {accuracy:.1%}")
    
    # Get adaptation summary
    print("\n" + "="*80)
    print("📋 ADAPTATION SUMMARY")
    print("="*80)
    
    summary = agent.get_adaptation_summary()
    print(f"Total Adaptations: {summary['total_adaptations']}")
    print(f"Average Quality: {summary['average_quality']:.3f}")
    print(f"Average Iterations: {summary['average_iterations']:.1f}")
    print(f"Convergence Rate: {summary['convergence_rate']:.1%}")
    print(f"Best Quality: {summary['best_quality']:.3f}")
    print(f"Validations Performed: {summary['validations_performed']}")
    print(f"Strategy: {summary['current_strategy']}")
    
    print("\n" + "="*80)
    print("✅ Domain Adaptation Agent demonstration complete!")
    print("="*80)
    print("\nKey Achievements:")
    print("• Adversarial domain adaptation with feature alignment")
    print("• Cross-domain transfer from products to movies")
    print("• Domain confusion for invariant features")
    print("• Transfer validation and quality metrics")
    print(f"• Achieved {adaptation_result.domain_confusion:.1%} domain confusion")
    print(f"• Transfer quality: {adaptation_result.transfer_quality:.3f}")
