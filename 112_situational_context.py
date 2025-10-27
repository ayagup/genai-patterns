"""
Pattern 112: Situational Context Agent

This pattern demonstrates dynamic situational context management where the agent
tracks, predicts, and detects anomalies in evolving contexts. It maintains
awareness of how situations change over time and adapts accordingly.

Key concepts:
- Context evolution tracking
- Situation prediction
- Context anomaly detection
- Dynamic context updating
- Temporal context modeling

Use cases:
- Adaptive dialogue systems
- Autonomous navigation
- Real-time monitoring
- Dynamic task planning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
import time
import uuid
import math


class ContextDimension(Enum):
    """Dimensions of situational context"""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SOCIAL = "social"
    TASK = "task"
    ENVIRONMENTAL = "environmental"
    COGNITIVE = "cognitive"


class ContextChangeType(Enum):
    """Types of context changes"""
    INCREMENTAL = "incremental"
    SUDDEN = "sudden"
    PERIODIC = "periodic"
    GRADUAL = "gradual"
    CYCLIC = "cyclic"


class AnomalyType(Enum):
    """Types of context anomalies"""
    OUTLIER = "outlier"
    DEVIATION = "deviation"
    UNEXPECTED_CHANGE = "unexpected_change"
    MISSING_EXPECTED = "missing_expected"
    CONTRADICTION = "contradiction"


@dataclass
class ContextSnapshot:
    """A snapshot of context at a specific time"""
    timestamp: float
    dimension: ContextDimension
    state: Dict[str, Any]
    confidence: float
    source: str = "observation"
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class ContextTransition:
    """Transition between context states"""
    from_snapshot: ContextSnapshot
    to_snapshot: ContextSnapshot
    change_type: ContextChangeType
    magnitude: float
    duration: float
    trigger: Optional[str] = None
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class ContextPrediction:
    """Prediction of future context state"""
    dimension: ContextDimension
    predicted_state: Dict[str, Any]
    time_horizon: float
    confidence: float
    basis: str  # What the prediction is based on
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class ContextAnomaly:
    """Detected context anomaly"""
    anomaly_type: AnomalyType
    dimension: ContextDimension
    expected: Any
    observed: Any
    severity: float  # 0.0 to 1.0
    timestamp: float
    description: str
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


class ContextEvolutionTracker:
    """Tracks how context evolves over time"""
    
    def __init__(self, max_history: int = 100):
        self.snapshots: Dict[ContextDimension, List[ContextSnapshot]] = {}
        self.transitions: List[ContextTransition] = []
        self.max_history = max_history
        
        # Initialize dimensions
        for dim in ContextDimension:
            self.snapshots[dim] = []
    
    def add_snapshot(self, dimension: ContextDimension, state: Dict[str, Any],
                    confidence: float = 1.0, source: str = "observation") -> ContextSnapshot:
        """Add a context snapshot"""
        snapshot = ContextSnapshot(
            timestamp=time.time(),
            dimension=dimension,
            state=state.copy(),
            confidence=confidence,
            source=source
        )
        
        self.snapshots[dimension].append(snapshot)
        
        # Maintain history limit
        if len(self.snapshots[dimension]) > self.max_history:
            self.snapshots[dimension] = self.snapshots[dimension][-self.max_history:]
        
        # Detect transition if previous snapshot exists
        if len(self.snapshots[dimension]) >= 2:
            prev_snapshot = self.snapshots[dimension][-2]
            transition = self._detect_transition(prev_snapshot, snapshot)
            if transition:
                self.transitions.append(transition)
        
        return snapshot
    
    def _detect_transition(self, from_snap: ContextSnapshot,
                          to_snap: ContextSnapshot) -> Optional[ContextTransition]:
        """Detect transition between snapshots"""
        
        # Calculate change magnitude
        magnitude = self._compute_state_distance(from_snap.state, to_snap.state)
        duration = to_snap.timestamp - from_snap.timestamp
        
        # Classify change type
        if magnitude < 0.1:
            return None  # Too small to be significant
        
        if duration < 0.5 and magnitude > 0.5:
            change_type = ContextChangeType.SUDDEN
        elif magnitude / duration < 0.1:
            change_type = ContextChangeType.GRADUAL
        elif self._is_periodic_change(from_snap, to_snap):
            change_type = ContextChangeType.CYCLIC
        elif magnitude / duration > 0.5:
            change_type = ContextChangeType.INCREMENTAL
        else:
            change_type = ContextChangeType.PERIODIC
        
        transition = ContextTransition(
            from_snapshot=from_snap,
            to_snapshot=to_snap,
            change_type=change_type,
            magnitude=magnitude,
            duration=duration
        )
        
        return transition
    
    def _compute_state_distance(self, state1: Dict[str, Any],
                                state2: Dict[str, Any]) -> float:
        """Compute distance between two states"""
        distance = 0.0
        all_keys = set(state1.keys()) | set(state2.keys())
        
        for key in all_keys:
            val1 = state1.get(key)
            val2 = state2.get(key)
            
            if val1 is None or val2 is None:
                distance += 1.0
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                distance += abs(val1 - val2)
            elif val1 != val2:
                distance += 1.0
        
        return distance / max(len(all_keys), 1)
    
    def _is_periodic_change(self, from_snap: ContextSnapshot,
                           to_snap: ContextSnapshot) -> bool:
        """Check if change is periodic"""
        # Simple heuristic: check if we've seen similar state before
        recent_snapshots = self.snapshots[from_snap.dimension][-10:]
        
        for snapshot in recent_snapshots[:-2]:
            if self._compute_state_distance(snapshot.state, to_snap.state) < 0.2:
                return True
        
        return False
    
    def get_evolution_pattern(self, dimension: ContextDimension,
                             window: Optional[int] = None) -> Dict[str, Any]:
        """Get evolution pattern for a dimension"""
        snapshots = self.snapshots[dimension]
        if window:
            snapshots = snapshots[-window:]
        
        if not snapshots:
            return {}
        
        # Analyze transitions
        relevant_transitions = [t for t in self.transitions
                               if t.from_snapshot.dimension == dimension]
        
        if window:
            relevant_transitions = relevant_transitions[-window:]
        
        # Calculate statistics
        change_types = [t.change_type.value for t in relevant_transitions]
        magnitudes = [t.magnitude for t in relevant_transitions]
        
        pattern = {
            "dimension": dimension.value,
            "snapshots": len(snapshots),
            "transitions": len(relevant_transitions),
            "change_types": {},
            "avg_magnitude": sum(magnitudes) / len(magnitudes) if magnitudes else 0.0,
            "max_magnitude": max(magnitudes) if magnitudes else 0.0,
            "stability": 1.0 - (sum(magnitudes) / len(magnitudes) if magnitudes else 0.0)
        }
        
        # Count change types
        for ct in set(change_types):
            pattern["change_types"][ct] = change_types.count(ct)
        
        return pattern


class SituationPredictor:
    """Predicts future context states"""
    
    def __init__(self):
        self.predictions: List[ContextPrediction] = []
        self.prediction_accuracy: List[float] = []
    
    def predict_context(self, tracker: ContextEvolutionTracker,
                       dimension: ContextDimension,
                       time_horizon: float = 5.0) -> ContextPrediction:
        """Predict future context state"""
        
        snapshots = tracker.snapshots[dimension]
        
        if len(snapshots) < 2:
            # Not enough history, predict current state
            current = snapshots[-1] if snapshots else None
            if current:
                return ContextPrediction(
                    dimension=dimension,
                    predicted_state=current.state.copy(),
                    time_horizon=time_horizon,
                    confidence=0.5,
                    basis="insufficient_history"
                )
        
        # Analyze recent trend
        recent_snapshots = snapshots[-5:]
        trend = self._analyze_trend(recent_snapshots)
        
        # Get current state
        current_state = snapshots[-1].state.copy()
        
        # Predict based on trend
        predicted_state = self._extrapolate_state(current_state, trend, time_horizon)
        
        # Calculate confidence based on trend consistency
        confidence = self._calculate_prediction_confidence(recent_snapshots, trend)
        
        # Generate alternatives
        alternatives = self._generate_alternatives(current_state, trend, time_horizon)
        
        prediction = ContextPrediction(
            dimension=dimension,
            predicted_state=predicted_state,
            time_horizon=time_horizon,
            confidence=confidence,
            basis="trend_extrapolation",
            alternatives=alternatives
        )
        
        self.predictions.append(prediction)
        return prediction
    
    def _analyze_trend(self, snapshots: List[ContextSnapshot]) -> Dict[str, float]:
        """Analyze trend in snapshots"""
        if len(snapshots) < 2:
            return {}
        
        trend = {}
        
        # Get all numeric keys
        first_state = snapshots[0].state
        numeric_keys = [k for k, v in first_state.items()
                       if isinstance(v, (int, float))]
        
        for key in numeric_keys:
            values = [s.state.get(key, 0) for s in snapshots]
            if len(values) >= 2:
                # Simple linear trend
                trend[key] = (values[-1] - values[0]) / len(values)
        
        return trend
    
    def _extrapolate_state(self, current_state: Dict[str, Any],
                          trend: Dict[str, float],
                          time_horizon: float) -> Dict[str, Any]:
        """Extrapolate future state"""
        predicted = current_state.copy()
        
        for key, rate in trend.items():
            if key in predicted and isinstance(predicted[key], (int, float)):
                predicted[key] = predicted[key] + (rate * time_horizon)
        
        return predicted
    
    def _calculate_prediction_confidence(self, snapshots: List[ContextSnapshot],
                                        trend: Dict[str, float]) -> float:
        """Calculate prediction confidence"""
        if len(snapshots) < 3:
            return 0.5
        
        # Check trend consistency
        consistency = 0.0
        
        for key in trend.keys():
            values = [s.state.get(key, 0) for s in snapshots]
            if len(values) >= 3:
                # Calculate variance
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                # Lower variance = higher confidence
                consistency += 1.0 / (1.0 + variance)
        
        confidence = consistency / max(len(trend), 1) if trend else 0.5
        return min(max(confidence, 0.1), 0.95)
    
    def _generate_alternatives(self, current_state: Dict[str, Any],
                              trend: Dict[str, float],
                              time_horizon: float) -> List[Dict[str, Any]]:
        """Generate alternative predictions"""
        alternatives = []
        
        # Optimistic alternative (trend * 1.5)
        optimistic = current_state.copy()
        for key, rate in trend.items():
            if key in optimistic and isinstance(optimistic[key], (int, float)):
                optimistic[key] = optimistic[key] + (rate * time_horizon * 1.5)
        alternatives.append({"scenario": "optimistic", "state": optimistic})
        
        # Pessimistic alternative (trend * 0.5)
        pessimistic = current_state.copy()
        for key, rate in trend.items():
            if key in pessimistic and isinstance(pessimistic[key], (int, float)):
                pessimistic[key] = pessimistic[key] + (rate * time_horizon * 0.5)
        alternatives.append({"scenario": "pessimistic", "state": pessimistic})
        
        return alternatives
    
    def validate_prediction(self, prediction: ContextPrediction,
                           actual_state: Dict[str, Any]) -> float:
        """Validate prediction against actual state"""
        predicted = prediction.predicted_state
        
        # Compute accuracy
        total_error = 0.0
        count = 0
        
        for key in predicted.keys():
            if key in actual_state:
                pred_val = predicted[key]
                actual_val = actual_state[key]
                
                if isinstance(pred_val, (int, float)) and isinstance(actual_val, (int, float)):
                    error = abs(pred_val - actual_val)
                    total_error += error
                    count += 1
                elif pred_val == actual_val:
                    count += 1
                else:
                    total_error += 1.0
                    count += 1
        
        accuracy = 1.0 - (total_error / count) if count > 0 else 0.0
        self.prediction_accuracy.append(max(accuracy, 0.0))
        return accuracy


class ContextAnomalyDetector:
    """Detects anomalies in context"""
    
    def __init__(self, sensitivity: float = 0.5):
        self.sensitivity = sensitivity
        self.anomalies: List[ContextAnomaly] = []
        self.normal_ranges: Dict[Tuple[ContextDimension, str], Tuple[float, float]] = {}
    
    def learn_normal_behavior(self, tracker: ContextEvolutionTracker,
                             dimension: ContextDimension):
        """Learn normal behavior for a dimension"""
        snapshots = tracker.snapshots[dimension]
        
        if len(snapshots) < 5:
            return
        
        # For each numeric key, learn normal range
        first_state = snapshots[0].state
        numeric_keys = [k for k, v in first_state.items()
                       if isinstance(v, (int, float))]
        
        for key in numeric_keys:
            values = [s.state.get(key, 0) for s in snapshots
                     if key in s.state and isinstance(s.state[key], (int, float))]
            
            if values:
                mean = sum(values) / len(values)
                std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
                
                # Normal range: mean ± 2*std
                lower = mean - 2 * std
                upper = mean + 2 * std
                
                self.normal_ranges[(dimension, key)] = (lower, upper)
    
    def detect_anomalies(self, snapshot: ContextSnapshot,
                        predictor: Optional[SituationPredictor] = None) -> List[ContextAnomaly]:
        """Detect anomalies in snapshot"""
        anomalies = []
        
        # Check for outliers
        for key, value in snapshot.state.items():
            if isinstance(value, (int, float)):
                range_key = (snapshot.dimension, key)
                if range_key in self.normal_ranges:
                    lower, upper = self.normal_ranges[range_key]
                    
                    if value < lower or value > upper:
                        severity = self._calculate_severity(value, lower, upper)
                        
                        anomaly = ContextAnomaly(
                            anomaly_type=AnomalyType.OUTLIER,
                            dimension=snapshot.dimension,
                            expected=(lower + upper) / 2,
                            observed=value,
                            severity=severity,
                            timestamp=snapshot.timestamp,
                            description=f"{key} value {value:.2f} outside normal range [{lower:.2f}, {upper:.2f}]"
                        )
                        anomalies.append(anomaly)
        
        # Check predictions if available
        if predictor and predictor.predictions:
            recent_predictions = [p for p in predictor.predictions
                                 if p.dimension == snapshot.dimension]
            if recent_predictions:
                latest_prediction = recent_predictions[-1]
                
                # Check if observation differs significantly from prediction
                for key in latest_prediction.predicted_state.keys():
                    if key in snapshot.state:
                        predicted = latest_prediction.predicted_state[key]
                        observed = snapshot.state[key]
                        
                        if isinstance(predicted, (int, float)) and isinstance(observed, (int, float)):
                            diff = abs(predicted - observed)
                            threshold = 0.3 * abs(predicted) if predicted != 0 else 0.5
                            
                            if diff > threshold:
                                anomaly = ContextAnomaly(
                                    anomaly_type=AnomalyType.DEVIATION,
                                    dimension=snapshot.dimension,
                                    expected=predicted,
                                    observed=observed,
                                    severity=min(diff / max(abs(predicted), 1.0), 1.0),
                                    timestamp=snapshot.timestamp,
                                    description=f"{key} deviated from prediction: expected {predicted:.2f}, got {observed:.2f}"
                                )
                                anomalies.append(anomaly)
        
        self.anomalies.extend(anomalies)
        return anomalies
    
    def _calculate_severity(self, value: float, lower: float, upper: float) -> float:
        """Calculate anomaly severity"""
        if value < lower:
            severity = (lower - value) / max(abs(lower), 1.0)
        else:
            severity = (value - upper) / max(abs(upper), 1.0)
        
        return min(severity, 1.0)
    
    def get_anomaly_report(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get anomaly report"""
        anomalies = self.anomalies
        
        if time_window:
            current_time = time.time()
            anomalies = [a for a in anomalies
                        if current_time - a.timestamp < time_window]
        
        report = {
            "total_anomalies": len(anomalies),
            "by_type": {},
            "by_dimension": {},
            "high_severity": 0,
            "average_severity": 0.0
        }
        
        if not anomalies:
            return report
        
        # Count by type
        for anomaly in anomalies:
            atype = anomaly.anomaly_type.value
            report["by_type"][atype] = report["by_type"].get(atype, 0) + 1
            
            dim = anomaly.dimension.value
            report["by_dimension"][dim] = report["by_dimension"].get(dim, 0) + 1
            
            if anomaly.severity > 0.7:
                report["high_severity"] += 1
        
        report["average_severity"] = sum(a.severity for a in anomalies) / len(anomalies)
        
        return report


class SituationalContextAgent:
    """
    Complete situational context agent that tracks, predicts, and detects
    anomalies in dynamically evolving contexts.
    """
    
    def __init__(self, max_history: int = 100, anomaly_sensitivity: float = 0.5):
        self.tracker = ContextEvolutionTracker(max_history)
        self.predictor = SituationPredictor()
        self.anomaly_detector = ContextAnomalyDetector(anomaly_sensitivity)
        self.active_predictions: Dict[ContextDimension, ContextPrediction] = {}
    
    def update_context(self, dimension: ContextDimension, state: Dict[str, Any],
                      confidence: float = 1.0) -> Dict[str, Any]:
        """Update context and analyze"""
        
        # Add snapshot
        snapshot = self.tracker.add_snapshot(dimension, state, confidence)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(snapshot, self.predictor)
        
        # Validate predictions if they exist
        prediction_accuracy = None
        if dimension in self.active_predictions:
            prediction = self.active_predictions[dimension]
            prediction_accuracy = self.predictor.validate_prediction(prediction, state)
        
        # Generate new prediction
        new_prediction = self.predictor.predict_context(self.tracker, dimension)
        self.active_predictions[dimension] = new_prediction
        
        # Learn normal behavior
        if len(self.tracker.snapshots[dimension]) >= 10:
            self.anomaly_detector.learn_normal_behavior(self.tracker, dimension)
        
        return {
            "snapshot_id": snapshot.id,
            "anomalies": len(anomalies),
            "anomaly_details": [{"type": a.anomaly_type.value, "severity": a.severity}
                               for a in anomalies],
            "prediction_accuracy": prediction_accuracy,
            "new_prediction": {
                "state": new_prediction.predicted_state,
                "confidence": new_prediction.confidence,
                "time_horizon": new_prediction.time_horizon
            }
        }
    
    def get_situation_summary(self) -> Dict[str, Any]:
        """Get comprehensive situation summary"""
        summary = {
            "tracked_dimensions": [],
            "evolution_patterns": {},
            "active_predictions": {},
            "anomaly_report": {}
        }
        
        # Tracked dimensions
        for dim in ContextDimension:
            if self.tracker.snapshots[dim]:
                summary["tracked_dimensions"].append(dim.value)
                
                # Evolution pattern
                pattern = self.tracker.get_evolution_pattern(dim)
                summary["evolution_patterns"][dim.value] = pattern
        
        # Active predictions
        for dim, pred in self.active_predictions.items():
            summary["active_predictions"][dim.value] = {
                "confidence": pred.confidence,
                "time_horizon": pred.time_horizon,
                "basis": pred.basis
            }
        
        # Anomaly report
        summary["anomaly_report"] = self.anomaly_detector.get_anomaly_report()
        
        return summary


# Demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("PATTERN 112: SITUATIONAL CONTEXT AGENT")
    print("Demonstration of dynamic context tracking, prediction, and anomaly detection")
    print("=" * 80)
    
    # Create agent
    agent = SituationalContextAgent(max_history=50, anomaly_sensitivity=0.6)
    
    print("\n1. Simulating Context Evolution")
    print("-" * 40)
    
    # Simulate temporal context evolution (time of day, activity level)
    print("Tracking TEMPORAL dimension (activity levels over time)...")
    for i in range(10):
        # Simulate daily activity pattern
        hour = 8 + i
        activity_level = 50 + 30 * math.sin(i * 0.5)  # Cyclic pattern
        
        result = agent.update_context(
            ContextDimension.TEMPORAL,
            {
                "hour": hour,
                "activity_level": activity_level,
                "day": "Monday"
            },
            confidence=0.95
        )
        
        if i % 3 == 0:
            print(f"  Hour {hour}: activity={activity_level:.1f}, "
                  f"anomalies={result['anomalies']}, "
                  f"pred_conf={result['new_prediction']['confidence']:.2f}")
    
    # Simulate spatial context (location, crowd density)
    print("\nTracking SPATIAL dimension (location and crowd)...")
    locations = ["home", "commute", "office", "office", "lunch", "office", "office", "commute", "home"]
    for i, location in enumerate(locations):
        crowd_density = {"home": 2, "commute": 50, "office": 30, "lunch": 80}.get(location, 10)
        
        result = agent.update_context(
            ContextDimension.SPATIAL,
            {
                "location": location,
                "crowd_density": crowd_density,
                "noise_level": crowd_density * 0.8
            },
            confidence=0.90
        )
        
        if i % 3 == 0:
            print(f"  Step {i+1}: {location}, crowd={crowd_density}, "
                  f"anomalies={result['anomalies']}")
    
    # Inject anomaly
    print("\n2. Injecting Context Anomaly")
    print("-" * 40)
    print("Sudden spike in activity level (anomaly)...")
    result = agent.update_context(
        ContextDimension.TEMPORAL,
        {
            "hour": 18,
            "activity_level": 150,  # Anomalous spike
            "day": "Monday"
        },
        confidence=0.95
    )
    
    print(f"✓ Detected {result['anomalies']} anomalies")
    if result['anomaly_details']:
        for detail in result['anomaly_details']:
            print(f"  - Type: {detail['type']}, Severity: {detail['severity']:.2f}")
    
    # Check predictions
    print("\n3. Context Predictions")
    print("-" * 40)
    summary = agent.get_situation_summary()
    
    for dim, pred_info in summary['active_predictions'].items():
        print(f"\n{dim.upper()}:")
        print(f"  Confidence: {pred_info['confidence']:.2f}")
        print(f"  Time horizon: {pred_info['time_horizon']:.1f}s")
        print(f"  Basis: {pred_info['basis']}")
    
    # Evolution patterns
    print("\n4. Context Evolution Patterns")
    print("-" * 40)
    
    for dim, pattern in summary['evolution_patterns'].items():
        if pattern:
            print(f"\n{dim.upper()}:")
            print(f"  Snapshots: {pattern['snapshots']}")
            print(f"  Transitions: {pattern['transitions']}")
            print(f"  Stability: {pattern['stability']:.2f}")
            print(f"  Avg magnitude: {pattern['avg_magnitude']:.2f}")
            if pattern['change_types']:
                print(f"  Change types: {pattern['change_types']}")
    
    # Anomaly report
    print("\n5. Anomaly Report")
    print("-" * 40)
    anomaly_report = summary['anomaly_report']
    print(f"Total anomalies: {anomaly_report['total_anomalies']}")
    print(f"High severity: {anomaly_report['high_severity']}")
    print(f"Average severity: {anomaly_report['average_severity']:.2f}")
    
    if anomaly_report['by_type']:
        print("\nBy type:")
        for atype, count in anomaly_report['by_type'].items():
            print(f"  {atype}: {count}")
    
    if anomaly_report['by_dimension']:
        print("\nBy dimension:")
        for dim, count in anomaly_report['by_dimension'].items():
            print(f"  {dim}: {count}")
    
    # Final summary
    print("\n6. Overall Situation Summary")
    print("-" * 40)
    print(f"Tracked dimensions: {len(summary['tracked_dimensions'])}")
    print(f"Active predictions: {len(summary['active_predictions'])}")
    print(f"Total snapshots: {sum(len(agent.tracker.snapshots[d]) for d in ContextDimension)}")
    print(f"Total transitions: {len(agent.tracker.transitions)}")
    
    print("\n" + "=" * 80)
    print("✓ Situational context agent demonstration complete!")
    print("  Context evolution tracking, prediction, and anomaly detection working.")
    print("=" * 80)
