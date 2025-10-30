"""
Pattern 039: Feedback Loops

Description:
    Feedback Loops enable agents to learn from the outcomes of their actions through
    immediate feedback (real-time corrections) and delayed feedback (outcome-based
    rewards). The agent uses feedback to improve performance, adjust strategies,
    and adapt behavior over time.

Components:
    - Feedback Collector: Gathers outcome data
    - Performance Tracker: Monitors metrics over time
    - Learning Mechanism: Incorporates feedback into behavior
    - Reward Signal: Quantifies outcome quality
    - Adaptation Strategy: How to adjust based on feedback

Use Cases:
    - Continuous improvement systems
    - Reinforcement learning agents
    - User preference learning
    - Model fine-tuning
    - A/B testing and optimization
    - Adaptive recommendation systems

LangChain Implementation:
    Uses feedback collection, performance tracking, and iterative improvement
    mechanisms to enable agents to learn from action outcomes.
"""

import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from statistics import mean, stdev
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class FeedbackType(Enum):
    """Types of feedback."""
    IMMEDIATE = "immediate"  # Real-time corrections
    DELAYED = "delayed"  # Outcome-based rewards
    EXPLICIT = "explicit"  # Direct user feedback
    IMPLICIT = "implicit"  # Inferred from behavior
    COMPARATIVE = "comparative"  # A/B testing results


class FeedbackSignal(Enum):
    """Feedback signals."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


@dataclass
class Feedback:
    """Feedback on an action or response."""
    action_id: str
    signal: FeedbackSignal
    score: float  # -1.0 to 1.0
    feedback_type: FeedbackType
    details: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """An action taken by the agent."""
    id: str
    query: str
    response: str
    strategy: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics over time."""
    total_actions: int
    average_score: float
    positive_feedback_rate: float
    negative_feedback_rate: float
    improvement_rate: float
    recent_scores: List[float]
    best_strategy: str
    timestamp: datetime = field(default_factory=datetime.now)


class FeedbackCollector:
    """
    Collects and stores feedback on agent actions.
    """
    
    def __init__(self):
        self.actions: Dict[str, Action] = {}
        self.feedback: Dict[str, List[Feedback]] = {}  # action_id -> feedbacks
    
    def record_action(self, action: Action):
        """Record an action taken by the agent."""
        self.actions[action.id] = action
        self.feedback[action.id] = []
    
    def add_feedback(self, feedback: Feedback):
        """Add feedback for an action."""
        if feedback.action_id not in self.feedback:
            self.feedback[feedback.action_id] = []
        
        self.feedback[feedback.action_id].append(feedback)
    
    def get_action_feedback(self, action_id: str) -> List[Feedback]:
        """Get all feedback for an action."""
        return self.feedback.get(action_id, [])
    
    def get_average_score(self, action_ids: Optional[List[str]] = None) -> float:
        """Get average feedback score."""
        if action_ids is None:
            action_ids = list(self.actions.keys())
        
        scores = []
        for action_id in action_ids:
            feedbacks = self.get_action_feedback(action_id)
            if feedbacks:
                avg_score = mean([f.score for f in feedbacks])
                scores.append(avg_score)
        
        return mean(scores) if scores else 0.0


class PerformanceTracker:
    """
    Tracks performance metrics over time.
    """
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.score_history: List[tuple[datetime, float]] = []
        self.strategy_scores: Dict[str, List[float]] = {}
    
    def record_score(self, score: float, strategy: str = "default"):
        """Record a performance score."""
        self.score_history.append((datetime.now(), score))
        
        if strategy not in self.strategy_scores:
            self.strategy_scores[strategy] = []
        self.strategy_scores[strategy].append(score)
    
    def get_recent_performance(self) -> PerformanceMetrics:
        """Get recent performance metrics."""
        if not self.score_history:
            return PerformanceMetrics(
                total_actions=0,
                average_score=0.0,
                positive_feedback_rate=0.0,
                negative_feedback_rate=0.0,
                improvement_rate=0.0,
                recent_scores=[],
                best_strategy="none"
            )
        
        # Get recent scores
        recent_scores = [score for _, score in self.score_history[-self.window_size:]]
        
        # Calculate metrics
        total_actions = len(self.score_history)
        average_score = mean(recent_scores)
        positive_rate = sum(1 for s in recent_scores if s > 0) / len(recent_scores)
        negative_rate = sum(1 for s in recent_scores if s < 0) / len(recent_scores)
        
        # Calculate improvement rate
        if len(recent_scores) >= 2:
            first_half = recent_scores[:len(recent_scores)//2]
            second_half = recent_scores[len(recent_scores)//2:]
            improvement_rate = mean(second_half) - mean(first_half)
        else:
            improvement_rate = 0.0
        
        # Find best strategy
        best_strategy = "default"
        best_avg = -float('inf')
        for strategy, scores in self.strategy_scores.items():
            if scores:
                avg = mean(scores)
                if avg > best_avg:
                    best_avg = avg
                    best_strategy = strategy
        
        return PerformanceMetrics(
            total_actions=total_actions,
            average_score=average_score,
            positive_feedback_rate=positive_rate,
            negative_feedback_rate=negative_rate,
            improvement_rate=improvement_rate,
            recent_scores=recent_scores,
            best_strategy=best_strategy
        )
    
    def get_strategy_performance(self, strategy: str) -> Optional[float]:
        """Get average performance for a strategy."""
        if strategy in self.strategy_scores and self.strategy_scores[strategy]:
            return mean(self.strategy_scores[strategy])
        return None


class FeedbackLoopAgent:
    """
    Agent that learns from feedback to improve performance.
    
    Features:
    - Immediate and delayed feedback
    - Performance tracking
    - Strategy adaptation
    - Continuous improvement
    - Multi-strategy experimentation
    """
    
    def __init__(
        self,
        strategies: Optional[Dict[str, ChatPromptTemplate]] = None,
        temperature: float = 0.7,
        exploration_rate: float = 0.2  # Rate of trying new strategies
    ):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        self.feedback_collector = FeedbackCollector()
        self.performance_tracker = PerformanceTracker()
        self.exploration_rate = exploration_rate
        
        # Initialize strategies
        if strategies is None:
            strategies = self._default_strategies()
        self.strategies = strategies
        
        # Current best strategy
        self.current_strategy = list(strategies.keys())[0]
        
        # Action counter for IDs
        self.action_count = 0
    
    def _default_strategies(self) -> Dict[str, ChatPromptTemplate]:
        """Default response strategies."""
        return {
            "detailed": ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Provide detailed, comprehensive answers."),
                ("user", "{query}")
            ]),
            "concise": ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Provide concise, to-the-point answers."),
                ("user", "{query}")
            ]),
            "friendly": ChatPromptTemplate.from_messages([
                ("system", "You are a friendly, conversational assistant. Provide warm, engaging answers."),
                ("user", "{query}")
            ]),
            "technical": ChatPromptTemplate.from_messages([
                ("system", "You are a technical expert. Provide precise, technical answers with specifics."),
                ("user", "{query}")
            ]),
        }
    
    def select_strategy(self) -> str:
        """
        Select a strategy using epsilon-greedy exploration.
        
        Returns:
            Strategy name
        """
        import random
        
        # Exploration: try random strategy
        if random.random() < self.exploration_rate:
            return random.choice(list(self.strategies.keys()))
        
        # Exploitation: use best performing strategy
        metrics = self.performance_tracker.get_recent_performance()
        if metrics.best_strategy in self.strategies:
            return metrics.best_strategy
        
        return self.current_strategy
    
    def process_query(
        self,
        query: str,
        strategy: Optional[str] = None
    ) -> Action:
        """
        Process a query using selected or specified strategy.
        
        Args:
            query: The query to process
            strategy: Optional specific strategy to use
            
        Returns:
            Action object with response
        """
        # Select strategy
        if strategy is None:
            strategy = self.select_strategy()
        
        # Get prompt for strategy
        prompt = self.strategies.get(strategy, self.strategies[self.current_strategy])
        
        # Generate response
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query})
        
        # Create action
        self.action_count += 1
        action = Action(
            id=f"action_{self.action_count}",
            query=query,
            response=response,
            strategy=strategy,
            metadata={"exploration": strategy != self.current_strategy}
        )
        
        # Record action
        self.feedback_collector.record_action(action)
        
        return action
    
    def provide_feedback(
        self,
        action_id: str,
        signal: FeedbackSignal,
        score: float,
        feedback_type: FeedbackType = FeedbackType.EXPLICIT,
        details: str = ""
    ):
        """
        Provide feedback on an action.
        
        Args:
            action_id: ID of the action
            signal: Positive, negative, or neutral
            score: Numerical score (-1.0 to 1.0)
            feedback_type: Type of feedback
            details: Additional details
        """
        feedback = Feedback(
            action_id=action_id,
            signal=signal,
            score=score,
            feedback_type=feedback_type,
            details=details
        )
        
        self.feedback_collector.add_feedback(feedback)
        
        # Update performance tracking
        action = self.feedback_collector.actions.get(action_id)
        if action:
            self.performance_tracker.record_score(score, action.strategy)
    
    def learn_from_feedback(self):
        """
        Update strategy selection based on accumulated feedback.
        """
        metrics = self.performance_tracker.get_recent_performance()
        
        # Update current strategy to best performing
        if metrics.best_strategy in self.strategies:
            self.current_strategy = metrics.best_strategy
        
        # Adjust exploration rate based on improvement
        if metrics.improvement_rate > 0:
            # Doing well, reduce exploration
            self.exploration_rate = max(0.1, self.exploration_rate * 0.9)
        else:
            # Not improving, increase exploration
            self.exploration_rate = min(0.5, self.exploration_rate * 1.1)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        metrics = self.performance_tracker.get_recent_performance()
        
        # Strategy breakdown
        strategy_performance = {}
        for strategy in self.strategies.keys():
            perf = self.performance_tracker.get_strategy_performance(strategy)
            if perf is not None:
                strategy_performance[strategy] = perf
        
        return {
            "total_actions": metrics.total_actions,
            "average_score": metrics.average_score,
            "positive_feedback_rate": metrics.positive_feedback_rate,
            "negative_feedback_rate": metrics.negative_feedback_rate,
            "improvement_rate": metrics.improvement_rate,
            "current_strategy": self.current_strategy,
            "best_strategy": metrics.best_strategy,
            "exploration_rate": self.exploration_rate,
            "strategy_performance": strategy_performance,
            "recent_scores": metrics.recent_scores
        }


def demonstrate_feedback_loops():
    """
    Demonstrates feedback loops with learning from outcomes.
    """
    print("=" * 80)
    print("FEEDBACK LOOPS DEMONSTRATION")
    print("=" * 80)
    
    # Create feedback loop agent
    agent = FeedbackLoopAgent(exploration_rate=0.3)
    
    # Show available strategies
    print("\n" + "=" * 80)
    print("Available Strategies")
    print("=" * 80)
    
    for strategy_name in agent.strategies.keys():
        print(f"  - {strategy_name}")
    
    print(f"\nInitial Strategy: {agent.current_strategy}")
    print(f"Exploration Rate: {agent.exploration_rate:.2f}")
    
    # Simulate interactions with feedback
    print("\n" + "=" * 80)
    print("Simulation: Learning from Feedback")
    print("=" * 80)
    
    # Test queries with different preference patterns
    test_scenarios = [
        # Users prefer concise answers for these
        ("What is Python?", "concise", 0.8),
        ("What is machine learning?", "concise", 0.9),
        ("Define API", "concise", 0.85),
        
        # Users prefer detailed answers for these
        ("How do neural networks work?", "detailed", 0.9),
        ("Explain quantum computing", "detailed", 0.85),
        ("How does blockchain work?", "detailed", 0.8),
        
        # Users prefer friendly tone for these
        ("Can you help me?", "friendly", 0.9),
        ("Thanks for your help!", "friendly", 0.95),
        
        # Technical questions prefer technical answers
        ("Explain Big O notation", "technical", 0.9),
        ("What is polymorphism?", "technical", 0.85),
    ]
    
    print("\nSimulating 10 interactions with feedback...")
    
    for i, (query, preferred_strategy, ideal_score) in enumerate(test_scenarios, 1):
        print(f"\n[Interaction {i}]")
        print(f"Query: {query}")
        
        # Agent processes query (may explore)
        action = agent.process_query(query)
        
        print(f"Strategy Used: {action.strategy}")
        print(f"Response Preview: {action.response[:100]}...")
        
        # Provide feedback based on whether strategy matched preference
        if action.strategy == preferred_strategy:
            # Good match
            score = ideal_score
            signal = FeedbackSignal.POSITIVE
            details = "Strategy matched user preference"
        else:
            # Suboptimal match
            score = ideal_score * 0.5  # Reduced score
            signal = FeedbackSignal.NEUTRAL
            details = "Strategy didn't match user preference"
        
        agent.provide_feedback(
            action_id=action.id,
            signal=signal,
            score=score,
            feedback_type=FeedbackType.IMMEDIATE,
            details=details
        )
        
        print(f"Feedback: {signal.value} (score: {score:.2f})")
        
        # Periodically learn from feedback
        if i % 3 == 0:
            agent.learn_from_feedback()
            print(f"\n  → Learning applied. Current strategy: {agent.current_strategy}")
            print(f"  → Exploration rate adjusted to: {agent.exploration_rate:.2f}")
    
    # Show performance report
    print("\n" + "=" * 80)
    print("Performance Report")
    print("=" * 80)
    
    report = agent.get_performance_report()
    
    print(f"\nTotal Actions: {report['total_actions']}")
    print(f"Average Score: {report['average_score']:.3f}")
    print(f"Positive Feedback Rate: {report['positive_feedback_rate']:.1%}")
    print(f"Negative Feedback Rate: {report['negative_feedback_rate']:.1%}")
    print(f"Improvement Rate: {report['improvement_rate']:+.3f}")
    
    print(f"\nCurrent Strategy: {report['current_strategy']}")
    print(f"Best Strategy: {report['best_strategy']}")
    print(f"Exploration Rate: {report['exploration_rate']:.2f}")
    
    print("\nStrategy Performance:")
    for strategy, score in sorted(report['strategy_performance'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {strategy}: {score:.3f}")
    
    print("\nRecent Scores:")
    for i, score in enumerate(report['recent_scores'][-5:], 1):
        print(f"  {i}. {score:.3f}")
    
    # Test learned behavior
    print("\n" + "=" * 80)
    print("Testing Learned Behavior")
    print("=" * 80)
    
    # Reduce exploration to see what agent learned
    agent.exploration_rate = 0.0
    
    test_queries = [
        "What is Docker?",  # Should use concise (learned preference)
        "How does cryptography work?",  # Should use detailed
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        action = agent.process_query(query)
        print(f"Selected Strategy: {action.strategy}")
        print(f"Response: {action.response[:150]}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Feedback Loops provide:
✓ Learning from action outcomes
✓ Performance tracking over time
✓ Strategy adaptation based on feedback
✓ Exploration vs exploitation balance
✓ Continuous improvement
✓ Multi-strategy experimentation

This pattern excels at:
- Adaptive systems
- Preference learning
- Continuous optimization
- A/B testing automation
- Reinforcement learning
- User personalization

Feedback components:
1. Feedback Collector: Gathers outcome data
2. Performance Tracker: Monitors metrics
3. Learning Mechanism: Updates behavior
4. Reward Signal: Quantifies quality
5. Adaptation Strategy: Adjusts approach

Feedback types:
- IMMEDIATE: Real-time corrections
- DELAYED: Outcome-based rewards
- EXPLICIT: Direct user feedback
- IMPLICIT: Inferred from behavior
- COMPARATIVE: A/B test results

Feedback signals:
- POSITIVE: Good outcome
- NEGATIVE: Bad outcome
- NEUTRAL: Acceptable outcome
- MIXED: Some good, some bad

Learning strategies:
- Epsilon-greedy: Balance exploration/exploitation
- Performance-based: Use best performing
- Adaptive exploration: Adjust based on improvement
- Multi-armed bandit: Test multiple strategies
- Gradient-based: Continuous optimization

Performance metrics:
- Average score: Overall quality
- Positive/negative rates: Feedback distribution
- Improvement rate: Learning trajectory
- Strategy performance: Compare approaches
- Recent trends: Current performance

Benefits:
- Adaptation: Improves over time
- Personalization: Learns preferences
- Optimization: Finds best strategies
- Resilience: Adapts to changes
- Efficiency: Focuses on what works
- Transparency: Track improvement

Use Feedback Loops when you need:
- Continuous improvement systems
- Adaptive recommendation engines
- Reinforcement learning agents
- User preference learning
- A/B testing automation
- Self-optimizing systems
""")


if __name__ == "__main__":
    demonstrate_feedback_loops()
