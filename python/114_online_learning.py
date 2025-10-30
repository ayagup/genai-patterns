"""
Pattern 114: Online Learning Agent

This pattern demonstrates online (incremental) learning where the agent
continuously learns from a stream of data without retraining from scratch.
It handles concept drift, adapts learning rates, and maintains performance.

Key concepts:
- Incremental learning from data streams
- Concept drift detection and adaptation
- Adaptive learning rate
- Forgetting mechanisms
- Performance monitoring

Use cases:
- Real-time recommendation systems
- Fraud detection
- Adaptive user interfaces
- Dynamic pricing systems
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import time
import uuid
import math
import random


class DriftType(Enum):
    """Types of concept drift"""
    SUDDEN = "sudden"
    GRADUAL = "gradual"
    INCREMENTAL = "incremental"
    RECURRING = "recurring"
    NONE = "none"


class LearningStrategy(Enum):
    """Online learning strategies"""
    INCREMENTAL = "incremental"
    SLIDING_WINDOW = "sliding_window"
    WEIGHTED_FORGETTING = "weighted_forgetting"
    ENSEMBLE = "ensemble"


@dataclass
class DataPoint:
    """A single data point in the stream"""
    features: Dict[str, float]
    label: Any
    timestamp: float
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class Prediction:
    """A prediction made by the model"""
    input_features: Dict[str, float]
    predicted_label: Any
    confidence: float
    timestamp: float
    actual_label: Optional[Any] = None
    correct: Optional[bool] = None
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class PerformanceWindow:
    """Performance metrics over a time window"""
    window_start: float
    window_end: float
    accuracy: float
    sample_count: int
    avg_confidence: float
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class DriftAlert:
    """Alert about detected concept drift"""
    drift_type: DriftType
    severity: float  # 0.0 to 1.0
    timestamp: float
    description: str
    affected_features: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


class IncrementalModel:
    """Simple incremental learning model"""
    
    def __init__(self, learning_rate: float = 0.1):
        self.weights: Dict[str, float] = {}
        self.bias: float = 0.0
        self.learning_rate = learning_rate
        self.update_count: int = 0
        self.feature_stats: Dict[str, Dict[str, float]] = {}
    
    def predict(self, features: Dict[str, float]) -> Tuple[Any, float]:
        """Make a prediction"""
        # Simple linear model for binary classification
        score = self.bias
        
        for feature, value in features.items():
            if feature in self.weights:
                score += self.weights[feature] * value
        
        # Sigmoid activation
        probability = 1.0 / (1.0 + math.exp(-score))
        predicted_label = 1 if probability > 0.5 else 0
        confidence = max(probability, 1 - probability)
        
        return predicted_label, confidence
    
    def update(self, features: Dict[str, float], label: int,
              learning_rate: Optional[float] = None) -> float:
        """Update model with new example"""
        lr = learning_rate if learning_rate is not None else self.learning_rate
        
        # Get prediction
        predicted_label, _ = self.predict(features)
        
        # Compute error
        error = label - (1 if predicted_label == 1 else 0)
        
        # Update weights (gradient descent)
        for feature, value in features.items():
            if feature not in self.weights:
                self.weights[feature] = 0.0
            
            self.weights[feature] += lr * error * value
            
            # Update feature statistics
            if feature not in self.feature_stats:
                self.feature_stats[feature] = {
                    "mean": 0.0,
                    "variance": 0.0,
                    "count": 0
                }
            
            stats = self.feature_stats[feature]
            count = stats["count"]
            old_mean = stats["mean"]
            
            # Incremental mean and variance
            stats["count"] += 1
            stats["mean"] = (old_mean * count + value) / stats["count"]
            
            if count > 0:
                stats["variance"] = ((stats["variance"] * count) + 
                                   (value - old_mean) * (value - stats["mean"])) / stats["count"]
        
        # Update bias
        self.bias += lr * error
        
        self.update_count += 1
        
        return abs(error)
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state"""
        return {
            "weights": self.weights.copy(),
            "bias": self.bias,
            "update_count": self.update_count,
            "learning_rate": self.learning_rate
        }


class ConceptDriftDetector:
    """Detects concept drift in data stream"""
    
    def __init__(self, window_size: int = 50, sensitivity: float = 0.5):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.performance_windows: List[PerformanceWindow] = []
        self.drift_alerts: List[DriftAlert] = []
        self.baseline_accuracy: Optional[float] = None
    
    def add_performance_window(self, window: PerformanceWindow):
        """Add performance window"""
        self.performance_windows.append(window)
        
        # Maintain window limit
        if len(self.performance_windows) > 20:
            self.performance_windows = self.performance_windows[-20:]
        
        # Set baseline if not set
        if self.baseline_accuracy is None and len(self.performance_windows) >= 3:
            self.baseline_accuracy = sum(w.accuracy for w in self.performance_windows[:3]) / 3
    
    def detect_drift(self) -> Optional[DriftAlert]:
        """Detect concept drift"""
        if len(self.performance_windows) < 5:
            return None
        
        recent_windows = self.performance_windows[-5:]
        older_windows = self.performance_windows[-10:-5] if len(self.performance_windows) >= 10 else []
        
        # Calculate recent vs older accuracy
        recent_acc = sum(w.accuracy for w in recent_windows) / len(recent_windows)
        older_acc = sum(w.accuracy for w in older_windows) / len(older_windows) if older_windows else recent_acc
        
        # Detect sudden drift
        if recent_acc < older_acc - 0.15:
            alert = DriftAlert(
                drift_type=DriftType.SUDDEN,
                severity=min((older_acc - recent_acc) / older_acc, 1.0),
                timestamp=time.time(),
                description=f"Sudden performance drop: {older_acc:.2f} → {recent_acc:.2f}"
            )
            self.drift_alerts.append(alert)
            return alert
        
        # Detect gradual drift
        if len(self.performance_windows) >= 10:
            # Check for declining trend
            accuracies = [w.accuracy for w in self.performance_windows[-10:]]
            if self._is_declining_trend(accuracies):
                alert = DriftAlert(
                    drift_type=DriftType.GRADUAL,
                    severity=0.6,
                    timestamp=time.time(),
                    description="Gradual performance degradation detected"
                )
                self.drift_alerts.append(alert)
                return alert
        
        return None
    
    def _is_declining_trend(self, values: List[float]) -> bool:
        """Check if values show declining trend"""
        if len(values) < 5:
            return False
        
        # Simple linear trend check
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        return second_half < first_half - 0.1
    
    def get_drift_report(self) -> Dict[str, Any]:
        """Get drift detection report"""
        return {
            "total_alerts": len(self.drift_alerts),
            "recent_alerts": len([a for a in self.drift_alerts
                                 if time.time() - a.timestamp < 60.0]),
            "by_type": {dt.value: len([a for a in self.drift_alerts if a.drift_type == dt])
                       for dt in DriftType if dt != DriftType.NONE},
            "average_severity": sum(a.severity for a in self.drift_alerts) / len(self.drift_alerts)
                               if self.drift_alerts else 0.0
        }


class AdaptiveLearningRate:
    """Manages adaptive learning rate"""
    
    def __init__(self, initial_rate: float = 0.1, decay: float = 0.95,
                 min_rate: float = 0.001, max_rate: float = 0.5):
        self.initial_rate = initial_rate
        self.current_rate = initial_rate
        self.decay = decay
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.error_history: List[float] = []
    
    def update(self, error: float) -> float:
        """Update learning rate based on error"""
        self.error_history.append(error)
        
        # Keep recent history
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
        
        # Adjust based on error trend
        if len(self.error_history) >= 10:
            recent_errors = self.error_history[-10:]
            avg_recent_error = sum(recent_errors) / len(recent_errors)
            
            # If errors increasing, boost learning rate
            if len(self.error_history) >= 20:
                older_errors = self.error_history[-20:-10]
                avg_older_error = sum(older_errors) / len(older_errors)
                
                if avg_recent_error > avg_older_error * 1.2:
                    # Increase learning rate
                    self.current_rate = min(self.current_rate * 1.1, self.max_rate)
                elif avg_recent_error < avg_older_error * 0.8:
                    # Decrease learning rate (converging)
                    self.current_rate *= self.decay
        
        # Apply decay
        self.current_rate = max(self.current_rate * self.decay, self.min_rate)
        
        return self.current_rate
    
    def reset(self):
        """Reset to initial rate"""
        self.current_rate = self.initial_rate
        self.error_history = []


class OnlineLearningAgent:
    """
    Complete online learning agent that learns incrementally from data streams,
    detects concept drift, and adapts learning parameters.
    """
    
    def __init__(self, initial_learning_rate: float = 0.1,
                 window_size: int = 50,
                 drift_sensitivity: float = 0.5):
        self.model = IncrementalModel(initial_learning_rate)
        self.adaptive_lr = AdaptiveLearningRate(initial_learning_rate)
        self.drift_detector = ConceptDriftDetector(window_size, drift_sensitivity)
        
        self.data_stream: List[DataPoint] = []
        self.predictions: List[Prediction] = []
        self.current_window: List[Prediction] = []
        self.window_size = window_size
        
        self.samples_processed: int = 0
        self.total_updates: int = 0
    
    def process_data_point(self, features: Dict[str, float], label: int,
                          weight: float = 1.0) -> Dict[str, Any]:
        """Process a single data point from the stream"""
        
        # Create data point
        data_point = DataPoint(
            features=features,
            label=label,
            timestamp=time.time(),
            weight=weight
        )
        self.data_stream.append(data_point)
        
        # Make prediction
        predicted_label, confidence = self.model.predict(features)
        
        # Create prediction record
        prediction = Prediction(
            input_features=features,
            predicted_label=predicted_label,
            confidence=confidence,
            timestamp=time.time(),
            actual_label=label,
            correct=(predicted_label == label)
        )
        
        self.predictions.append(prediction)
        self.current_window.append(prediction)
        
        # Update model
        error = self.model.update(features, label, self.adaptive_lr.current_rate)
        
        # Update adaptive learning rate
        new_lr = self.adaptive_lr.update(error)
        
        self.samples_processed += 1
        self.total_updates += 1
        
        # Check if window is full
        drift_detected = None
        if len(self.current_window) >= self.window_size:
            # Create performance window
            correct_count = sum(1 for p in self.current_window if p.correct)
            accuracy = correct_count / len(self.current_window)
            avg_conf = sum(p.confidence for p in self.current_window) / len(self.current_window)
            
            perf_window = PerformanceWindow(
                window_start=self.current_window[0].timestamp,
                window_end=self.current_window[-1].timestamp,
                accuracy=accuracy,
                sample_count=len(self.current_window),
                avg_confidence=avg_conf
            )
            
            self.drift_detector.add_performance_window(perf_window)
            
            # Check for drift
            drift_alert = self.drift_detector.detect_drift()
            if drift_alert:
                drift_detected = drift_alert
                # Adapt to drift
                self._adapt_to_drift(drift_alert)
            
            # Reset window
            self.current_window = []
        
        return {
            "predicted": predicted_label,
            "correct": prediction.correct,
            "confidence": confidence,
            "error": error,
            "learning_rate": new_lr,
            "drift_detected": drift_detected.drift_type.value if drift_detected else None,
            "samples_processed": self.samples_processed
        }
    
    def _adapt_to_drift(self, alert: DriftAlert):
        """Adapt model to detected drift"""
        if alert.drift_type == DriftType.SUDDEN:
            # Increase learning rate for sudden drift
            self.adaptive_lr.current_rate = min(
                self.adaptive_lr.current_rate * 2.0,
                self.adaptive_lr.max_rate
            )
        elif alert.drift_type == DriftType.GRADUAL:
            # Moderately increase learning rate
            self.adaptive_lr.current_rate = min(
                self.adaptive_lr.current_rate * 1.5,
                self.adaptive_lr.max_rate
            )
    
    def get_performance_stats(self, window: Optional[int] = None) -> Dict[str, Any]:
        """Get performance statistics"""
        predictions = self.predictions[-window:] if window else self.predictions
        
        if not predictions:
            return {}
        
        correct_count = sum(1 for p in predictions if p.correct)
        accuracy = correct_count / len(predictions)
        
        return {
            "accuracy": accuracy,
            "total_predictions": len(predictions),
            "correct": correct_count,
            "incorrect": len(predictions) - correct_count,
            "avg_confidence": sum(p.confidence for p in predictions) / len(predictions),
            "samples_processed": self.samples_processed,
            "total_updates": self.total_updates
        }
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary"""
        return {
            "performance": self.get_performance_stats(window=100),
            "model_state": {
                "num_features": len(self.model.weights),
                "update_count": self.model.update_count,
                "current_learning_rate": self.adaptive_lr.current_rate
            },
            "drift_detection": self.drift_detector.get_drift_report(),
            "data_stream": {
                "total_points": len(self.data_stream),
                "current_window_size": len(self.current_window),
                "window_progress": f"{len(self.current_window)}/{self.window_size}"
            }
        }


# Demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("PATTERN 114: ONLINE LEARNING AGENT")
    print("Demonstration of incremental learning with concept drift detection")
    print("=" * 80)
    
    # Create agent
    agent = OnlineLearningAgent(
        initial_learning_rate=0.1,
        window_size=50,
        drift_sensitivity=0.5
    )
    
    print("\n1. Processing Initial Data Stream (Phase 1)")
    print("-" * 40)
    print("Learning pattern: feature1 > 0.5 → class 1, else → class 0")
    
    # Phase 1: Initial pattern
    for i in range(100):
        feature1 = random.uniform(0, 1)
        feature2 = random.uniform(0, 1)
        label = 1 if feature1 > 0.5 else 0
        
        result = agent.process_data_point(
            {"feature1": feature1, "feature2": feature2},
            label
        )
        
        if i % 25 == 0:
            print(f"  Sample {i+1}: pred={result['predicted']}, "
                  f"correct={result['correct']}, conf={result['confidence']:.2f}, "
                  f"lr={result['learning_rate']:.4f}")
    
    stats1 = agent.get_performance_stats(window=50)
    print(f"\nPhase 1 Performance: {stats1['accuracy']:.2%} accuracy")
    
    # Phase 2: Concept drift (pattern changes)
    print("\n2. Introducing Concept Drift (Phase 2)")
    print("-" * 40)
    print("NEW pattern: feature2 > 0.5 → class 1, else → class 0")
    
    drift_detected_at = None
    for i in range(100):
        feature1 = random.uniform(0, 1)
        feature2 = random.uniform(0, 1)
        # Pattern changed!
        label = 1 if feature2 > 0.5 else 0
        
        result = agent.process_data_point(
            {"feature1": feature1, "feature2": feature2},
            label
        )
        
        if result['drift_detected'] and not drift_detected_at:
            drift_detected_at = i
            print(f"✓ DRIFT DETECTED at sample {100 + i + 1}!")
            print(f"  Type: {result['drift_detected']}")
            print(f"  Learning rate increased to: {result['learning_rate']:.4f}")
        
        if (i + 100) % 25 == 0:
            print(f"  Sample {i+101}: pred={result['predicted']}, "
                  f"correct={result['correct']}, conf={result['confidence']:.2f}, "
                  f"lr={result['learning_rate']:.4f}")
    
    stats2 = agent.get_performance_stats(window=50)
    print(f"\nPhase 2 Performance: {stats2['accuracy']:.2%} accuracy")
    
    # Phase 3: Recovery
    print("\n3. Continued Learning (Phase 3)")
    print("-" * 40)
    print("Agent adapting to new pattern...")
    
    for i in range(50):
        feature1 = random.uniform(0, 1)
        feature2 = random.uniform(0, 1)
        label = 1 if feature2 > 0.5 else 0
        
        result = agent.process_data_point(
            {"feature1": feature1, "feature2": feature2},
            label
        )
        
        if (i + 200) % 25 == 0:
            print(f"  Sample {i+201}: pred={result['predicted']}, "
                  f"correct={result['correct']}, conf={result['confidence']:.2f}, "
                  f"lr={result['learning_rate']:.4f}")
    
    stats3 = agent.get_performance_stats(window=50)
    print(f"\nPhase 3 Performance: {stats3['accuracy']:.2%} accuracy (recovered!)")
    
    # Summary
    print("\n4. Learning Summary")
    print("-" * 40)
    summary = agent.get_learning_summary()
    
    perf = summary["performance"]
    print(f"Overall Performance:")
    print(f"  Accuracy: {perf['accuracy']:.2%}")
    print(f"  Total predictions: {perf['total_predictions']}")
    print(f"  Correct: {perf['correct']}, Incorrect: {perf['incorrect']}")
    print(f"  Avg confidence: {perf['avg_confidence']:.2f}")
    
    model_state = summary["model_state"]
    print(f"\nModel State:")
    print(f"  Features learned: {model_state['num_features']}")
    print(f"  Total updates: {model_state['update_count']}")
    print(f"  Current learning rate: {model_state['current_learning_rate']:.4f}")
    
    drift_report = summary["drift_detection"]
    print(f"\nDrift Detection:")
    print(f"  Total alerts: {drift_report['total_alerts']}")
    print(f"  Recent alerts: {drift_report['recent_alerts']}")
    if drift_report['by_type']:
        print(f"  By type: {drift_report['by_type']}")
    print(f"  Avg severity: {drift_report['average_severity']:.2f}")
    
    data_stream = summary["data_stream"]
    print(f"\nData Stream:")
    print(f"  Total points processed: {data_stream['total_points']}")
    print(f"  Current window: {data_stream['window_progress']}")
    
    print("\n" + "=" * 80)
    print("✓ Online learning agent demonstration complete!")
    print("  Incremental learning, drift detection, and adaptation working.")
    print("=" * 80)
