"""
Agent Reputation System Pattern

Tracks and manages agent reputations based on behavior and performance.
Implements trust metrics and reputation-based decision making.

Use Cases:
- Trust management
- Partner selection
- Quality control
- Fraud prevention

Advantages:
- Promotes good behavior
- Identifies reliable agents
- Reduces risk
- Enables trust-based cooperation
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import math


class ReputationModel(Enum):
    """Reputation calculation models"""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    BETA_DISTRIBUTION = "beta_distribution"
    EIGENTRUST = "eigentrust"


class InteractionType(Enum):
    """Types of interactions"""
    TRANSACTION = "transaction"
    COLLABORATION = "collaboration"
    REVIEW = "review"
    RECOMMENDATION = "recommendation"


class RatingType(Enum):
    """Rating types"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class Interaction:
    """Record of agent interaction"""
    interaction_id: str
    from_agent: str
    to_agent: str
    interaction_type: InteractionType
    timestamp: datetime
    rating: float  # 0-1 scale
    rating_type: RatingType
    context: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False


@dataclass
class ReputationScore:
    """Agent reputation score"""
    agent_id: str
    overall_score: float  # 0-1 scale
    confidence: float  # Confidence in score
    num_interactions: int
    last_updated: datetime
    category_scores: Dict[str, float] = field(default_factory=dict)
    trend: float = 0.0  # Positive/negative trend


@dataclass
class TrustRelationship:
    """Trust relationship between agents"""
    from_agent: str
    to_agent: str
    trust_score: float  # 0-1 scale
    established_at: datetime
    last_interaction: datetime
    interaction_count: int


@dataclass
class ReputationHistory:
    """Historical reputation data"""
    agent_id: str
    timestamp: datetime
    score: float
    num_interactions: int


class ReputationCalculator:
    """Calculates reputation scores"""
    
    def __init__(self, model: ReputationModel = ReputationModel.WEIGHTED_AVERAGE):
        self.model = model
    
    def calculate_reputation(self,
                            interactions: List[Interaction],
                            decay_factor: float = 0.95) -> Tuple[float, float]:
        """
        Calculate reputation score from interactions.
        
        Args:
            interactions: List of interactions
            decay_factor: Time decay factor
            
        Returns:
            (score, confidence) tuple
        """
        if not interactions:
            return 0.5, 0.0  # Neutral with no confidence
        
        if self.model == ReputationModel.SIMPLE_AVERAGE:
            return self._simple_average(interactions)
        elif self.model == ReputationModel.WEIGHTED_AVERAGE:
            return self._weighted_average(interactions, decay_factor)
        elif self.model == ReputationModel.BETA_DISTRIBUTION:
            return self._beta_distribution(interactions)
        else:
            return self._weighted_average(interactions, decay_factor)
    
    def _simple_average(self,
                       interactions: List[Interaction]) -> Tuple[float, float]:
        """Simple average of ratings"""
        total = sum(i.rating for i in interactions)
        avg = total / len(interactions)
        
        # Confidence based on number of interactions
        confidence = min(len(interactions) / 10.0, 1.0)
        
        return avg, confidence
    
    def _weighted_average(self,
                         interactions: List[Interaction],
                         decay_factor: float) -> Tuple[float, float]:
        """Time-weighted average"""
        if not interactions:
            return 0.5, 0.0
        
        # Sort by timestamp
        sorted_interactions = sorted(
            interactions,
            key=lambda x: x.timestamp,
            reverse=True
        )
        
        now = datetime.now()
        total_weight = 0.0
        weighted_sum = 0.0
        
        for i, interaction in enumerate(sorted_interactions):
            # Time-based weight
            age_days = (now - interaction.timestamp).days
            time_weight = decay_factor ** age_days
            
            # Recency weight (more recent = higher weight)
            recency_weight = decay_factor ** i
            
            # Combined weight
            weight = time_weight * recency_weight
            
            # Verified interactions get higher weight
            if interaction.verified:
                weight *= 1.5
            
            weighted_sum += interaction.rating * weight
            total_weight += weight
        
        score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        # Confidence based on number and recency
        confidence = min(
            (len(interactions) / 20.0) * (total_weight / len(interactions)),
            1.0
        )
        
        return score, confidence
    
    def _beta_distribution(self,
                          interactions: List[Interaction]) -> Tuple[float, float]:
        """Beta distribution model"""
        # Count positive and negative interactions
        alpha = 1  # Prior for positive
        beta = 1   # Prior for negative
        
        for interaction in interactions:
            if interaction.rating >= 0.7:
                alpha += 1
            elif interaction.rating <= 0.3:
                beta += 1
            else:
                # Neutral ratings contribute fractionally
                alpha += interaction.rating
                beta += (1 - interaction.rating)
        
        # Expected value of beta distribution
        score = alpha / (alpha + beta)
        
        # Variance as inverse of confidence
        variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        confidence = 1.0 - math.sqrt(variance)
        
        return score, confidence
    
    def calculate_trend(self,
                       history: List[ReputationHistory],
                       window_size: int = 10) -> float:
        """
        Calculate reputation trend.
        
        Args:
            history: Historical scores
            window_size: Window for trend calculation
            
        Returns:
            Trend value (-1 to 1)
        """
        if len(history) < 2:
            return 0.0
        
        # Sort by timestamp
        sorted_history = sorted(history, key=lambda x: x.timestamp)
        
        # Take recent window
        recent = sorted_history[-window_size:]
        
        if len(recent) < 2:
            return 0.0
        
        # Calculate linear trend
        n = len(recent)
        x_mean = (n - 1) / 2
        y_mean = sum(h.score for h in recent) / n
        
        numerator = sum(
            (i - x_mean) * (h.score - y_mean)
            for i, h in enumerate(recent)
        )
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        
        # Normalize to -1 to 1
        return max(-1.0, min(1.0, slope * 10))


class TrustNetwork:
    """Manages trust relationships between agents"""
    
    def __init__(self):
        self.relationships: Dict[Tuple[str, str], TrustRelationship] = {}
    
    def update_trust(self,
                    from_agent: str,
                    to_agent: str,
                    interaction: Interaction) -> None:
        """
        Update trust relationship based on interaction.
        
        Args:
            from_agent: Trusting agent
            to_agent: Trusted agent
            interaction: Recent interaction
        """
        key = (from_agent, to_agent)
        
        if key not in self.relationships:
            # Create new relationship
            self.relationships[key] = TrustRelationship(
                from_agent=from_agent,
                to_agent=to_agent,
                trust_score=interaction.rating,
                established_at=interaction.timestamp,
                last_interaction=interaction.timestamp,
                interaction_count=1
            )
        else:
            # Update existing relationship
            rel = self.relationships[key]
            
            # Exponential moving average
            alpha = 0.3  # Learning rate
            rel.trust_score = (
                alpha * interaction.rating +
                (1 - alpha) * rel.trust_score
            )
            
            rel.last_interaction = interaction.timestamp
            rel.interaction_count += 1
    
    def get_trust_score(self,
                       from_agent: str,
                       to_agent: str) -> float:
        """Get trust score between two agents"""
        key = (from_agent, to_agent)
        
        if key in self.relationships:
            return self.relationships[key].trust_score
        
        return 0.5  # Neutral for unknown relationships
    
    def get_transitive_trust(self,
                            from_agent: str,
                            to_agent: str,
                            max_hops: int = 3) -> float:
        """
        Calculate transitive trust through network.
        
        Args:
            from_agent: Starting agent
            to_agent: Target agent
            max_hops: Maximum path length
            
        Returns:
            Transitive trust score
        """
        # BFS to find trust paths
        visited = {from_agent}
        queue = [(from_agent, 1.0, 0)]  # (agent, trust, hops)
        max_trust = 0.0
        
        while queue:
            current, trust, hops = queue.pop(0)
            
            if current == to_agent:
                max_trust = max(max_trust, trust)
                continue
            
            if hops >= max_hops:
                continue
            
            # Find neighbors
            for (from_a, to_a), rel in self.relationships.items():
                if from_a == current and to_a not in visited:
                    # Propagate trust
                    new_trust = trust * rel.trust_score
                    queue.append((to_a, new_trust, hops + 1))
                    visited.add(to_a)
        
        return max_trust


class ReputationAggregator:
    """Aggregates reputation from multiple sources"""
    
    def aggregate_scores(self,
                        scores: List[Tuple[float, float]],
                        weights: Optional[List[float]] = None) -> Tuple[float, float]:
        """
        Aggregate multiple reputation scores.
        
        Args:
            scores: List of (score, confidence) tuples
            weights: Optional weights for each score
            
        Returns:
            Aggregated (score, confidence)
        """
        if not scores:
            return 0.5, 0.0
        
        if weights is None:
            weights = [1.0] * len(scores)
        
        # Weight by confidence
        total_weight = 0.0
        weighted_sum = 0.0
        
        for (score, confidence), weight in zip(scores, weights):
            effective_weight = weight * confidence
            weighted_sum += score * effective_weight
            total_weight += effective_weight
        
        if total_weight == 0:
            return 0.5, 0.0
        
        aggregated_score = weighted_sum / total_weight
        
        # Aggregate confidence (weighted average)
        aggregated_confidence = sum(
            c * w for (_, c), w in zip(scores, weights)
        ) / sum(weights)
        
        return aggregated_score, aggregated_confidence


class AgentReputationSystem:
    """
    Comprehensive reputation management system for agents.
    Tracks interactions, calculates reputations, and manages trust.
    """
    
    def __init__(self,
                 model: ReputationModel = ReputationModel.WEIGHTED_AVERAGE,
                 decay_factor: float = 0.95):
        self.model = model
        self.decay_factor = decay_factor
        
        # Components
        self.calculator = ReputationCalculator(model)
        self.trust_network = TrustNetwork()
        self.aggregator = ReputationAggregator()
        
        # State
        self.interactions: Dict[str, List[Interaction]] = {}
        self.reputations: Dict[str, ReputationScore] = {}
        self.history: Dict[str, List[ReputationHistory]] = {}
        
        self.interaction_counter = 0
    
    def record_interaction(self,
                          from_agent: str,
                          to_agent: str,
                          interaction_type: InteractionType,
                          rating: float,
                          context: Optional[Dict[str, Any]] = None,
                          verified: bool = False) -> str:
        """
        Record interaction between agents.
        
        Args:
            from_agent: Rating agent
            to_agent: Rated agent
            interaction_type: Type of interaction
            rating: Rating value (0-1)
            context: Optional context
            verified: Whether interaction is verified
            
        Returns:
            Interaction ID
        """
        if context is None:
            context = {}
        
        # Determine rating type
        if rating >= 0.7:
            rating_type = RatingType.POSITIVE
        elif rating <= 0.3:
            rating_type = RatingType.NEGATIVE
        else:
            rating_type = RatingType.NEUTRAL
        
        # Create interaction
        interaction = Interaction(
            interaction_id="int_{}".format(self.interaction_counter),
            from_agent=from_agent,
            to_agent=to_agent,
            interaction_type=interaction_type,
            timestamp=datetime.now(),
            rating=rating,
            rating_type=rating_type,
            context=context,
            verified=verified
        )
        
        self.interaction_counter += 1
        
        # Store interaction
        if to_agent not in self.interactions:
            self.interactions[to_agent] = []
        self.interactions[to_agent].append(interaction)
        
        # Update trust network
        self.trust_network.update_trust(from_agent, to_agent, interaction)
        
        # Update reputation
        self._update_reputation(to_agent)
        
        return interaction.interaction_id
    
    def get_reputation(self, agent_id: str) -> Optional[ReputationScore]:
        """Get reputation score for agent"""
        return self.reputations.get(agent_id)
    
    def get_category_reputation(self,
                               agent_id: str,
                               category: str) -> float:
        """Get reputation for specific category"""
        reputation = self.reputations.get(agent_id)
        
        if not reputation:
            return 0.5
        
        return reputation.category_scores.get(category, reputation.overall_score)
    
    def compare_agents(self,
                      agent_ids: List[str],
                      category: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Compare reputation of multiple agents.
        
        Args:
            agent_ids: Agents to compare
            category: Optional specific category
            
        Returns:
            Sorted list of (agent_id, score) tuples
        """
        scores = []
        
        for agent_id in agent_ids:
            if category:
                score = self.get_category_reputation(agent_id, category)
            else:
                reputation = self.get_reputation(agent_id)
                score = reputation.overall_score if reputation else 0.5
            
            scores.append((agent_id, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores
    
    def get_trust_score(self,
                       from_agent: str,
                       to_agent: str,
                       use_transitive: bool = True) -> float:
        """
        Get trust score between agents.
        
        Args:
            from_agent: Trusting agent
            to_agent: Trusted agent
            use_transitive: Whether to use transitive trust
            
        Returns:
            Trust score
        """
        direct_trust = self.trust_network.get_trust_score(from_agent, to_agent)
        
        if not use_transitive:
            return direct_trust
        
        # Get transitive trust
        transitive_trust = self.trust_network.get_transitive_trust(
            from_agent,
            to_agent
        )
        
        # Combine direct and transitive
        if direct_trust > 0.5:  # Have direct experience
            return 0.7 * direct_trust + 0.3 * transitive_trust
        else:  # Rely more on transitive
            return 0.3 * direct_trust + 0.7 * transitive_trust
    
    def recommend_agents(self,
                        requester_id: str,
                        category: Optional[str] = None,
                        min_reputation: float = 0.6,
                        top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Recommend agents based on reputation and trust.
        
        Args:
            requester_id: Agent requesting recommendations
            category: Optional category filter
            min_reputation: Minimum reputation threshold
            top_n: Number of recommendations
            
        Returns:
            List of (agent_id, score) recommendations
        """
        candidates = []
        
        for agent_id, reputation in self.reputations.items():
            if agent_id == requester_id:
                continue
            
            # Get reputation score
            if category:
                rep_score = self.get_category_reputation(agent_id, category)
            else:
                rep_score = reputation.overall_score
            
            if rep_score < min_reputation:
                continue
            
            # Get trust score
            trust_score = self.get_trust_score(requester_id, agent_id)
            
            # Combined score (weighted average)
            combined_score = (
                0.6 * rep_score +
                0.4 * trust_score
            )
            
            candidates.append((agent_id, combined_score))
        
        # Sort and return top N
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:top_n]
    
    def detect_reputation_fraud(self,
                               agent_id: str,
                               threshold: float = 0.3) -> Tuple[bool, str]:
        """
        Detect potential reputation manipulation.
        
        Args:
            agent_id: Agent to check
            threshold: Detection threshold
            
        Returns:
            (is_fraud, reason) tuple
        """
        interactions = self.interactions.get(agent_id, [])
        
        if len(interactions) < 5:
            return False, "Insufficient data"
        
        # Check for sudden reputation spike
        recent = interactions[-10:]
        older = interactions[-20:-10] if len(interactions) >= 20 else []
        
        if older:
            recent_avg = sum(i.rating for i in recent) / len(recent)
            older_avg = sum(i.rating for i in older) / len(older)
            
            if recent_avg - older_avg > threshold:
                return True, "Sudden reputation spike detected"
        
        # Check for clustering of ratings from same agents
        raters = {}
        for interaction in recent:
            rater = interaction.from_agent
            raters[rater] = raters.get(rater, 0) + 1
        
        if raters:
            max_count = max(raters.values())
            if max_count > len(recent) * 0.5:
                return True, "Suspicious rating concentration"
        
        # Check for unverified high ratings
        unverified_high = sum(
            1 for i in recent
            if i.rating > 0.8 and not i.verified
        )
        
        if unverified_high > len(recent) * 0.7:
            return True, "Too many unverified high ratings"
        
        return False, "No fraud detected"
    
    def _update_reputation(self, agent_id: str) -> None:
        """Update reputation score for agent"""
        interactions = self.interactions.get(agent_id, [])
        
        if not interactions:
            return
        
        # Calculate overall reputation
        score, confidence = self.calculator.calculate_reputation(
            interactions,
            self.decay_factor
        )
        
        # Calculate category-specific reputations
        category_scores = {}
        categories = set(
            i.context.get("category")
            for i in interactions
            if i.context.get("category")
        )
        
        for category in categories:
            category_interactions = [
                i for i in interactions
                if i.context.get("category") == category
            ]
            
            if category_interactions:
                cat_score, _ = self.calculator.calculate_reputation(
                    category_interactions,
                    self.decay_factor
                )
                category_scores[category] = cat_score
        
        # Calculate trend
        history = self.history.get(agent_id, [])
        trend = self.calculator.calculate_trend(history)
        
        # Update or create reputation score
        reputation = ReputationScore(
            agent_id=agent_id,
            overall_score=score,
            confidence=confidence,
            num_interactions=len(interactions),
            last_updated=datetime.now(),
            category_scores=category_scores,
            trend=trend
        )
        
        self.reputations[agent_id] = reputation
        
        # Add to history
        if agent_id not in self.history:
            self.history[agent_id] = []
        
        self.history[agent_id].append(ReputationHistory(
            agent_id=agent_id,
            timestamp=datetime.now(),
            score=score,
            num_interactions=len(interactions)
        ))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_agents = len(self.reputations)
        total_interactions = sum(
            len(ints) for ints in self.interactions.values()
        )
        
        avg_reputation = (
            sum(r.overall_score for r in self.reputations.values()) /
            total_agents
        ) if total_agents > 0 else 0.0
        
        high_reputation = sum(
            1 for r in self.reputations.values()
            if r.overall_score >= 0.7
        )
        
        low_reputation = sum(
            1 for r in self.reputations.values()
            if r.overall_score <= 0.3
        )
        
        return {
            "model": self.model.value,
            "total_agents": total_agents,
            "total_interactions": total_interactions,
            "avg_reputation": avg_reputation,
            "high_reputation_agents": high_reputation,
            "low_reputation_agents": low_reputation,
            "trust_relationships": len(self.trust_network.relationships)
        }


def demonstrate_reputation_system():
    """Demonstrate agent reputation system"""
    print("=" * 70)
    print("Agent Reputation System Demonstration")
    print("=" * 70)
    
    system = AgentReputationSystem(
        model=ReputationModel.WEIGHTED_AVERAGE,
        decay_factor=0.95
    )
    
    # Example 1: Record interactions
    print("\n1. Recording Interactions:")
    
    # Simulate various interactions
    interactions_data = [
        ("buyer_1", "seller_1", InteractionType.TRANSACTION, 0.9, {"category": "electronics"}),
        ("buyer_2", "seller_1", InteractionType.TRANSACTION, 0.85, {"category": "electronics"}),
        ("buyer_3", "seller_1", InteractionType.TRANSACTION, 0.8, {"category": "books"}),
        ("buyer_1", "seller_2", InteractionType.TRANSACTION, 0.6, {"category": "electronics"}),
        ("buyer_2", "seller_2", InteractionType.TRANSACTION, 0.5, {"category": "electronics"}),
        ("buyer_1", "seller_3", InteractionType.TRANSACTION, 0.95, {"category": "books"}),
        ("buyer_2", "seller_3", InteractionType.TRANSACTION, 0.9, {"category": "books"}),
        ("buyer_3", "seller_3", InteractionType.TRANSACTION, 0.92, {"category": "books"}),
    ]
    
    for from_a, to_a, int_type, rating, context in interactions_data:
        int_id = system.record_interaction(
            from_a, to_a, int_type, rating, context, verified=True
        )
        print("  Recorded: {} -> {} (rating: {:.2f})".format(
            from_a, to_a, rating
        ))
    
    # Example 2: Get reputation scores
    print("\n2. Reputation Scores:")
    
    for seller_id in ["seller_1", "seller_2", "seller_3"]:
        reputation = system.get_reputation(seller_id)
        if reputation:
            print("\n  {}:".format(seller_id))
            print("    Overall score: {:.2%}".format(reputation.overall_score))
            print("    Confidence: {:.2%}".format(reputation.confidence))
            print("    Interactions: {}".format(reputation.num_interactions))
            print("    Trend: {:.2f}".format(reputation.trend))
            
            if reputation.category_scores:
                print("    Category scores:")
                for cat, score in reputation.category_scores.items():
                    print("      {}: {:.2%}".format(cat, score))
    
    # Example 3: Compare agents
    print("\n3. Comparing Sellers:")
    
    comparison = system.compare_agents(
        ["seller_1", "seller_2", "seller_3"],
        category="electronics"
    )
    
    print("  Electronics category:")
    for rank, (agent_id, score) in enumerate(comparison, 1):
        print("    {}. {}: {:.2%}".format(rank, agent_id, score))
    
    # Example 4: Trust scores
    print("\n4. Trust Scores:")
    
    trust_score = system.get_trust_score("buyer_1", "seller_1")
    print("  buyer_1 -> seller_1: {:.2%}".format(trust_score))
    
    trust_score = system.get_trust_score("buyer_1", "seller_2")
    print("  buyer_1 -> seller_2: {:.2%}".format(trust_score))
    
    # Example 5: Recommendations
    print("\n5. Agent Recommendations for buyer_1:")
    
    recommendations = system.recommend_agents(
        "buyer_1",
        category="books",
        min_reputation=0.7,
        top_n=3
    )
    
    for rank, (agent_id, score) in enumerate(recommendations, 1):
        print("  {}. {}: {:.2%}".format(rank, agent_id, score))
    
    # Example 6: Fraud detection
    print("\n6. Reputation Fraud Detection:")
    
    # Add suspicious interactions
    for i in range(5):
        system.record_interaction(
            "suspicious_buyer_{}".format(i),
            "suspicious_seller",
            InteractionType.TRANSACTION,
            0.95,
            {"category": "test"},
            verified=False
        )
    
    is_fraud, reason = system.detect_reputation_fraud("suspicious_seller")
    print("  suspicious_seller:")
    print("    Fraud detected: {}".format(is_fraud))
    print("    Reason: {}".format(reason))
    
    # Example 7: System statistics
    print("\n7. System Statistics:")
    stats = system.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Example 8: Different reputation models
    print("\n8. Comparing Reputation Models:")
    
    models = [
        ReputationModel.SIMPLE_AVERAGE,
        ReputationModel.WEIGHTED_AVERAGE,
        ReputationModel.BETA_DISTRIBUTION
    ]
    
    for model in models:
        test_system = AgentReputationSystem(model=model)
        
        # Add same interactions
        for from_a, to_a, int_type, rating, context in interactions_data[:3]:
            test_system.record_interaction(
                from_a, to_a, int_type, rating, context
            )
        
        reputation = test_system.get_reputation("seller_1")
        
        print("\n  Model: {}".format(model.value))
        if reputation:
            print("    Score: {:.2%}".format(reputation.overall_score))
            print("    Confidence: {:.2%}".format(reputation.confidence))


if __name__ == "__main__":
    demonstrate_reputation_system()
