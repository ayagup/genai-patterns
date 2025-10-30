"""
Ensemble/Committee Pattern
Multiple agents work independently, results are aggregated
"""
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from collections import Counter
import statistics
@dataclass
class AgentPrediction:
    agent_id: str
    prediction: Any
    confidence: float
    reasoning: str
class EnsembleAgent:
    def __init__(self, agent_id: str, model_type: str):
        self.agent_id = agent_id
        self.model_type = model_type
    def predict(self, input_data: Any) -> AgentPrediction:
        """Make a prediction (simulated)"""
        # In reality, each agent would use a different model
        # Here we simulate different prediction strategies
        if self.model_type == "optimistic":
            prediction = self._optimistic_predict(input_data)
        elif self.model_type == "pessimistic":
            prediction = self._pessimistic_predict(input_data)
        elif self.model_type == "balanced":
            prediction = self._balanced_predict(input_data)
        else:
            prediction = self._random_predict(input_data)
        return prediction
    def _optimistic_predict(self, input_data: Any) -> AgentPrediction:
        """Optimistic prediction strategy"""
        return AgentPrediction(
            agent_id=self.agent_id,
            prediction="positive",
            confidence=0.85,
            reasoning="Analysis shows positive indicators"
        )
    def _pessimistic_predict(self, input_data: Any) -> AgentPrediction:
        """Pessimistic prediction strategy"""
        return AgentPrediction(
            agent_id=self.agent_id,
            prediction="negative",
            confidence=0.75,
            reasoning="Risk factors identified"
        )
    def _balanced_predict(self, input_data: Any) -> AgentPrediction:
        """Balanced prediction strategy"""
        import random
        prediction = random.choice(["positive", "negative", "neutral"])
        return AgentPrediction(
            agent_id=self.agent_id,
            prediction=prediction,
            confidence=0.70,
            reasoning="Balanced analysis of factors"
        )
    def _random_predict(self, input_data: Any) -> AgentPrediction:
        """Random prediction"""
        import random
        return AgentPrediction(
            agent_id=self.agent_id,
            prediction=random.choice(["positive", "negative"]),
            confidence=random.uniform(0.5, 0.9),
            reasoning="Random model prediction"
        )
class EnsembleSystem:
    """Ensemble system that aggregates multiple agent predictions"""
    def __init__(self, aggregation_method: str = "voting"):
        self.agents: List[EnsembleAgent] = []
        self.aggregation_method = aggregation_method
        self.prediction_history: List[Dict[str, Any]] = []
    def add_agent(self, agent: EnsembleAgent):
        """Add an agent to the ensemble"""
        self.agents.append(agent)
        print(f"Added agent: {agent.agent_id} ({agent.model_type})")
    def predict(self, input_data: Any) -> Dict[str, Any]:
        """Get predictions from all agents and aggregate"""
        print(f"\n{'='*70}")
        print(f"ENSEMBLE PREDICTION")
        print(f"{'='*70}")
        print(f"Input: {input_data}")
        print(f"Number of agents: {len(self.agents)}")
        print(f"Aggregation method: {self.aggregation_method}\n")
        # Collect predictions from all agents
        predictions: List[AgentPrediction] = []
        print("Individual Agent Predictions:")
        print("-" * 70)
        for agent in self.agents:
            prediction = agent.predict(input_data)
            predictions.append(prediction)
            print(f"\n{prediction.agent_id} ({agent.model_type}):")
            print(f"  Prediction: {prediction.prediction}")
            print(f"  Confidence: {prediction.confidence:.2%}")
            print(f"  Reasoning: {prediction.reasoning}")
        # Aggregate predictions
        print(f"\n{'='*70}")
        print("AGGREGATION")
        print(f"{'='*70}\n")
        if self.aggregation_method == "voting":
            result = self._majority_voting(predictions)
        elif self.aggregation_method == "weighted":
            result = self._weighted_voting(predictions)
        elif self.aggregation_method == "averaging":
            result = self._averaging(predictions)
        else:
            result = self._majority_voting(predictions)
        # Store in history
        self.prediction_history.append({
            "input": input_data,
            "individual_predictions": predictions,
            "final_result": result
        })
        return result
    def _majority_voting(self, predictions: List[AgentPrediction]) -> Dict[str, Any]:
        """Aggregate using majority voting"""
        votes = [p.prediction for p in predictions]
        vote_counts = Counter(votes)
        majority_prediction, count = vote_counts.most_common(1)[0]
        agreement = count / len(predictions)
        print(f"Vote Distribution:")
        for prediction, count in vote_counts.most_common():
            percentage = (count / len(predictions)) * 100
            print(f"  {prediction}: {count}/{len(predictions)} ({percentage:.1f}%)")
        print(f"\nMajority Prediction: {majority_prediction}")
        print(f"Agreement: {agreement:.1%}")
        return {
            "method": "majority_voting",
            "prediction": majority_prediction,
            "confidence": agreement,
            "vote_distribution": dict(vote_counts),
            "num_agents": len(predictions)
        }
    def _weighted_voting(self, predictions: List[AgentPrediction]) -> Dict[str, Any]:
        """Aggregate using confidence-weighted voting"""
        weighted_votes: Dict[str, float] = {}
        total_weight = 0
        for pred in predictions:
            weighted_votes[pred.prediction] = weighted_votes.get(pred.prediction, 0) + pred.confidence
            total_weight += pred.confidence
        # Normalize and find winner
        for pred in weighted_votes:
            weighted_votes[pred] /= total_weight
        winner = max(weighted_votes.items(), key=lambda x: x[1])
        print(f"Weighted Vote Distribution:")
        for prediction, weight in sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True):
            print(f"  {prediction}: {weight:.1%}")
        print(f"\nWeighted Winner: {winner[0]}")
        print(f"Weight: {winner[1]:.1%}")
        return {
            "method": "weighted_voting",
            "prediction": winner[0],
            "confidence": winner[1],
            "weighted_distribution": weighted_votes,
            "num_agents": len(predictions)
        }
    def _averaging(self, predictions: List[AgentPrediction]) -> Dict[str, Any]:
        """Aggregate by averaging confidence scores"""
        confidences = [p.confidence for p in predictions]
        avg_confidence = statistics.mean(confidences)
        std_confidence = statistics.stdev(confidences) if len(confidences) > 1 else 0
        # Use majority for categorical prediction
        votes = [p.prediction for p in predictions]
        majority_prediction = Counter(votes).most_common(1)[0][0]
        print(f"Confidence Statistics:")
        print(f"  Mean: {avg_confidence:.2%}")
        print(f"  Std Dev: {std_confidence:.2%}")
        print(f"  Min: {min(confidences):.2%}")
        print(f"  Max: {max(confidences):.2%}")
        print(f"\nPrediction: {majority_prediction}")
        print(f"Average Confidence: {avg_confidence:.1%}")
        return {
            "method": "averaging",
            "prediction": majority_prediction,
            "confidence": avg_confidence,
            "std_dev": std_confidence,
            "num_agents": len(predictions)
        }
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get ensemble performance statistics"""
        if not self.prediction_history:
            return {"total_predictions": 0}
        total = len(self.prediction_history)
        avg_confidence = statistics.mean([
            p["final_result"]["confidence"]
            for p in self.prediction_history
        ])
        return {
            "total_predictions": total,
            "num_agents": len(self.agents),
            "average_confidence": avg_confidence,
            "aggregation_method": self.aggregation_method
        }
# Usage
if __name__ == "__main__":
    # Create ensemble system
    ensemble = EnsembleSystem(aggregation_method="weighted")
    # Add diverse agents
    ensemble.add_agent(EnsembleAgent("agent_1", "optimistic"))
    ensemble.add_agent(EnsembleAgent("agent_2", "pessimistic"))
    ensemble.add_agent(EnsembleAgent("agent_3", "balanced"))
    ensemble.add_agent(EnsembleAgent("agent_4", "balanced"))
    ensemble.add_agent(EnsembleAgent("agent_5", "optimistic"))
    print("\n" + "="*80)
    print("ENSEMBLE SYSTEM DEMONSTRATION")
    print("="*80)
    # Make predictions
    test_cases = [
        "Market analysis for Q4",
        "Risk assessment for new project",
        "Customer sentiment analysis"
    ]
    for test_case in test_cases:
        result = ensemble.predict(test_case)
        print(f"\n{'='*70}")
        print(f"FINAL RESULT")
        print(f"{'='*70}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Method: {result['method']}")
        print("\n" + "="*80 + "\n")
    # Show ensemble stats
    stats = ensemble.get_ensemble_stats()
    print("="*70)
    print("ENSEMBLE STATISTICS")
    print("="*70)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2%}" if value < 1 else f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
