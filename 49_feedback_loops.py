"""
Pattern 39: Feedback Loops
Description:
    Implements continuous improvement mechanisms through systematic feedback
    collection, analysis, and integration into agent behavior.
Use Cases:
    - Continuous learning from user interactions
    - Performance-based adaptation
    - Quality improvement over time
    - User preference learning
Key Features:
    - Multiple feedback sources (user, system, peer)
    - Feedback aggregation and analysis
    - Automatic behavior adjustment
    - Performance tracking over time
Example:
    >>> agent = FeedbackLoopAgent()
    >>> result = agent.execute_with_feedback("task")
    >>> agent.process_feedback(result['id'], feedback)
    >>> stats = agent.get_improvement_metrics()
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import time
from collections import defaultdict, deque
import statistics
class FeedbackType(Enum):
    """Types of feedback"""
    USER_RATING = "user_rating"
    USER_CORRECTION = "user_correction"
    SYSTEM_METRIC = "system_metric"
    PEER_REVIEW = "peer_review"
    AUTOMATED_TEST = "automated_test"
class FeedbackSentiment(Enum):
    """Sentiment of feedback"""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
@dataclass
class Feedback:
    """Individual feedback item"""
    feedback_id: str
    execution_id: str
    feedback_type: FeedbackType
    sentiment: FeedbackSentiment
    rating: Optional[float] = None  # 0.0 to 1.0
    comment: Optional[str] = None
    corrections: Optional[Dict[str, Any]] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
@dataclass
class ExecutionRecord:
    """Record of an execution for feedback tracking"""
    execution_id: str
    task: str
    output: Any
    context: Dict[str, Any]
    timestamp: float
    feedback_items: List[Feedback] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
@dataclass
class ImprovementMetrics:
    """Metrics tracking improvement over time"""
    period_start: float
    period_end: float
    total_executions: int
    avg_rating: float
    rating_trend: float  # positive = improving
    feedback_count: int
    corrections_applied: int
    performance_improvement: Dict[str, float] = field(default_factory=dict)
class FeedbackAggregator:
    """Aggregates and analyzes feedback"""
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.recent_ratings: deque = deque(maxlen=window_size)
        self.feedback_by_type: Dict[FeedbackType, List[Feedback]] = defaultdict(list)
    def add_feedback(self, feedback: Feedback):
        """Add feedback to aggregator"""
        if feedback.rating is not None:
            self.recent_ratings.append(feedback.rating)
        self.feedback_by_type[feedback.feedback_type].append(feedback)
    def get_average_rating(self) -> float:
        """Get average rating from recent feedback"""
        if not self.recent_ratings:
            return 0.5
        return statistics.mean(self.recent_ratings)
    def get_rating_trend(self) -> float:
        """Calculate trend in ratings (positive = improving)"""
        if len(self.recent_ratings) < 2:
            return 0.0
        # Simple linear regression
        recent = list(self.recent_ratings)
        n = len(recent)
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(recent)
        numerator = sum((x[i] - x_mean) * (recent[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        slope = numerator / denominator
        return slope
    def get_common_corrections(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get most common corrections from feedback"""
        corrections_list = []
        for feedback_items in self.feedback_by_type.values():
            for feedback in feedback_items:
                if feedback.corrections:
                    corrections_list.append(feedback.corrections)
        # Count correction types
        correction_counts = defaultdict(int)
        for correction in corrections_list:
            for key in correction.keys():
                correction_counts[key] += 1
        # Sort by frequency
        sorted_corrections = sorted(
            correction_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [
            {'type': corr_type, 'count': count}
            for corr_type, count in sorted_corrections[:top_n]
        ]
    def get_sentiment_distribution(self) -> Dict[str, float]:
        """Get distribution of feedback sentiments"""
        sentiment_counts = defaultdict(int)
        total = 0
        for feedback_items in self.feedback_by_type.values():
            for feedback in feedback_items:
                sentiment_counts[feedback.sentiment.value] += 1
                total += 1
        if total == 0:
            return {}
        return {
            sentiment: count / total
            for sentiment, count in sentiment_counts.items()
        }
class BehaviorAdapter:
    """Adapts agent behavior based on feedback"""
    def __init__(self):
        self.learned_patterns: Dict[str, Any] = {}
        self.adaptation_rules: List[Callable] = []
        self.parameter_adjustments: Dict[str, float] = defaultdict(float)
    def learn_from_feedback(self, feedback: Feedback, execution: ExecutionRecord):
        """Learn from a single feedback item"""
        if feedback.feedback_type == FeedbackType.USER_CORRECTION:
            self._learn_correction(feedback, execution)
        elif feedback.feedback_type == FeedbackType.USER_RATING:
            self._adjust_confidence(feedback, execution)
        elif feedback.feedback_type == FeedbackType.SYSTEM_METRIC:
            self._optimize_performance(feedback, execution)
    def _learn_correction(self, feedback: Feedback, execution: ExecutionRecord):
        """Learn from user corrections"""
        if not feedback.corrections:
            return
        # Store pattern: task type -> correction
        task_type = execution.context.get('task_type', 'general')
        if task_type not in self.learned_patterns:
            self.learned_patterns[task_type] = []
        self.learned_patterns[task_type].append({
            'original_output': execution.output,
            'corrections': feedback.corrections,
            'timestamp': feedback.timestamp
        })
    def _adjust_confidence(self, feedback: Feedback, execution: ExecutionRecord):
        """Adjust confidence thresholds based on rating"""
        if feedback.rating is None:
            return
        task_type = execution.context.get('task_type', 'general')
        # If low rating, increase caution
        if feedback.rating < 0.5:
            self.parameter_adjustments[f"{task_type}_min_confidence"] += 0.05
        # If high rating, can be more confident
        elif feedback.rating > 0.8:
            self.parameter_adjustments[f"{task_type}_min_confidence"] -= 0.02
    def _optimize_performance(self, feedback: Feedback, execution: ExecutionRecord):
        """Optimize based on performance metrics"""
        if not feedback.metadata.get('metrics'):
            return
        metrics = feedback.metadata['metrics']
        # Adjust parameters to improve metrics
        if metrics.get('latency', 0) > 1.0:  # High latency
            self.parameter_adjustments['max_iterations'] -= 1
        if metrics.get('accuracy', 1.0) < 0.7:  # Low accuracy
            self.parameter_adjustments['temperature'] -= 0.1
    def apply_adaptations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned adaptations to current context"""
        adapted_context = context.copy()
        task_type = context.get('task_type', 'general')
        # Apply learned patterns
        if task_type in self.learned_patterns:
            patterns = self.learned_patterns[task_type]
            adapted_context['learned_patterns'] = patterns[-5:]  # Last 5
        # Apply parameter adjustments
        for param, adjustment in self.parameter_adjustments.items():
            if param.startswith(task_type):
                param_name = param.replace(f"{task_type}_", "")
                current_value = adapted_context.get(param_name, 0.5)
                adapted_context[param_name] = max(0.0, min(1.0, current_value + adjustment))
        return adapted_context
class FeedbackLoopAgent:
    """
    Agent with continuous feedback loops for improvement
    Features:
    - Multi-source feedback collection
    - Automatic behavior adaptation
    - Performance tracking
    - Continuous learning
    """
    def __init__(
        self,
        agent_id: str = "feedback_agent",
        window_size: int = 100
    ):
        self.agent_id = agent_id
        self.execution_history: Dict[str, ExecutionRecord] = {}
        self.aggregator = FeedbackAggregator(window_size)
        self.adapter = BehaviorAdapter()
        self.improvement_history: List[ImprovementMetrics] = []
        self.execution_counter = 0
    def execute_with_feedback(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute task with feedback tracking
        Args:
            task: Task to execute
            context: Execution context
        Returns:
            Result with execution ID for feedback
        """
        self.execution_counter += 1
        execution_id = f"exec_{self.execution_counter}_{int(time.time())}"
        # Apply learned adaptations
        adapted_context = self.adapter.apply_adaptations(context or {})
        # Execute task
        start_time = time.time()
        output = self._execute_task(task, adapted_context)
        execution_time = time.time() - start_time
        # Record execution
        record = ExecutionRecord(
            execution_id=execution_id,
            task=task,
            output=output,
            context=adapted_context,
            timestamp=time.time(),
            performance_metrics={
                'execution_time': execution_time,
                'context_adaptations': len(adapted_context) - len(context or {})
            }
        )
        self.execution_history[execution_id] = record
        return {
            'execution_id': execution_id,
            'output': output,
            'request_feedback': True,
            'adapted_parameters': adapted_context
        }
    def process_feedback(
        self,
        execution_id: str,
        feedback_type: FeedbackType,
        rating: Optional[float] = None,
        comment: Optional[str] = None,
        corrections: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process received feedback
        Args:
            execution_id: ID of execution to provide feedback for
            feedback_type: Type of feedback
            rating: Optional rating (0.0 to 1.0)
            comment: Optional comment
            corrections: Optional corrections
            metadata: Optional additional data
        Returns:
            Feedback processing result
        """
        if execution_id not in self.execution_history:
            return {'error': 'Execution ID not found'}
        # Determine sentiment
        sentiment = self._determine_sentiment(rating, comment, corrections)
        # Create feedback object
        feedback = Feedback(
            feedback_id=f"fb_{len(self.aggregator.recent_ratings)}",
            execution_id=execution_id,
            feedback_type=feedback_type,
            sentiment=sentiment,
            rating=rating,
            comment=comment,
            corrections=corrections,
            metadata=metadata or {}
        )
        # Add to execution record
        execution = self.execution_history[execution_id]
        execution.feedback_items.append(feedback)
        # Aggregate feedback
        self.aggregator.add_feedback(feedback)
        # Learn from feedback
        self.adapter.learn_from_feedback(feedback, execution)
        return {
            'feedback_id': feedback.feedback_id,
            'processed': True,
            'adaptations_applied': len(self.adapter.learned_patterns),
            'current_avg_rating': self.aggregator.get_average_rating()
        }
    def get_improvement_metrics(
        self,
        period_hours: float = 24.0
    ) -> ImprovementMetrics:
        """
        Calculate improvement metrics for a time period
        Args:
            period_hours: Time period to analyze
        Returns:
            Improvement metrics
        """
        period_start = time.time() - (period_hours * 3600)
        period_end = time.time()
        # Filter executions in period
        period_executions = [
            exec_rec for exec_rec in self.execution_history.values()
            if period_start <= exec_rec.timestamp <= period_end
        ]
        if not period_executions:
            return ImprovementMetrics(
                period_start=period_start,
                period_end=period_end,
                total_executions=0,
                avg_rating=0.0,
                rating_trend=0.0,
                feedback_count=0,
                corrections_applied=0
            )
        # Calculate metrics
        all_ratings = []
        total_feedback = 0
        corrections_count = 0
        for execution in period_executions:
            for feedback in execution.feedback_items:
                total_feedback += 1
                if feedback.rating is not None:
                    all_ratings.append(feedback.rating)
                if feedback.corrections:
                    corrections_count += 1
        avg_rating = statistics.mean(all_ratings) if all_ratings else 0.0
        rating_trend = self.aggregator.get_rating_trend()
        # Performance improvements
        performance_improvement = {}
        if len(period_executions) > 1:
            early_executions = period_executions[:len(period_executions)//2]
            late_executions = period_executions[len(period_executions)//2:]
            early_time = statistics.mean(
                e.performance_metrics.get('execution_time', 0)
                for e in early_executions
            )
            late_time = statistics.mean(
                e.performance_metrics.get('execution_time', 0)
                for e in late_executions
            )
            if early_time > 0:
                performance_improvement['execution_time_reduction'] = (
                    (early_time - late_time) / early_time
                )
        metrics = ImprovementMetrics(
            period_start=period_start,
            period_end=period_end,
            total_executions=len(period_executions),
            avg_rating=avg_rating,
            rating_trend=rating_trend,
            feedback_count=total_feedback,
            corrections_applied=corrections_count,
            performance_improvement=performance_improvement
        )
        self.improvement_history.append(metrics)
        return metrics
    def _determine_sentiment(
        self,
        rating: Optional[float],
        comment: Optional[str],
        corrections: Optional[Dict[str, Any]]
    ) -> FeedbackSentiment:
        """Determine sentiment from feedback components"""
        # Rating-based sentiment
        if rating is not None:
            if rating >= 0.7:
                return FeedbackSentiment.POSITIVE
            elif rating <= 0.4:
                return FeedbackSentiment.NEGATIVE
        # Corrections indicate issues
        if corrections:
            return FeedbackSentiment.NEGATIVE
        # Comment-based (simple keyword matching)
        if comment:
            comment_lower = comment.lower()
            positive_words = ['good', 'great', 'excellent', 'perfect', 'correct']
            negative_words = ['bad', 'wrong', 'incorrect', 'poor', 'error']
            if any(word in comment_lower for word in positive_words):
                return FeedbackSentiment.POSITIVE
            elif any(word in comment_lower for word in negative_words):
                return FeedbackSentiment.NEGATIVE
        return FeedbackSentiment.NEUTRAL
    def _execute_task(self, task: str, context: Dict[str, Any]) -> str:
        """Execute the actual task (simulated)"""
        # In reality, this would call an LLM or other processing
        # Check for learned patterns
        task_type = context.get('task_type', 'general')
        if task_type in self.adapter.learned_patterns:
            patterns = self.adapter.learned_patterns[task_type]
            if patterns:
                # Apply most recent correction
                last_pattern = patterns[-1]
                return f"Task result (improved based on feedback): {task} - Applied {len(patterns)} learned patterns"
        # Check confidence threshold
        min_confidence = context.get('min_confidence', 0.5)
        result = f"Result for: {task}"
        if min_confidence > 0.7:
            result += " (high confidence mode)"
        return result
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        total_feedback = sum(
            len(record.feedback_items)
            for record in self.execution_history.values()
        )
        sentiment_dist = self.aggregator.get_sentiment_distribution()
        common_corrections = self.aggregator.get_common_corrections()
        return {
            'total_executions': len(self.execution_history),
            'total_feedback_items': total_feedback,
            'avg_rating': self.aggregator.get_average_rating(),
            'rating_trend': self.aggregator.get_rating_trend(),
            'sentiment_distribution': sentiment_dist,
            'common_corrections': common_corrections,
            'learned_patterns': len(self.adapter.learned_patterns),
            'parameter_adjustments': dict(self.adapter.parameter_adjustments),
            'improvement_periods': len(self.improvement_history)
        }
def main():
    """Demonstrate feedback loops pattern"""
    print("=" * 60)
    print("Feedback Loops Agent Demonstration")
    print("=" * 60)
    # Create agent
    agent = FeedbackLoopAgent()
    print("\n1. Initial Executions (Before Feedback)")
    print("-" * 60)
    # Execute some tasks
    tasks = [
        ("Calculate 15 + 27", {'task_type': 'math'}),
        ("Explain photosynthesis", {'task_type': 'explanation'}),
        ("Translate 'hello' to Spanish", {'task_type': 'translation'}),
    ]
    execution_ids = []
    for task, context in tasks:
        result = agent.execute_with_feedback(task, context)
        execution_ids.append(result['execution_id'])
        print(f"\nTask: {task}")
        print(f"Output: {result['output']}")
        print(f"Execution ID: {result['execution_id']}")
    print("\n" + "=" * 60)
    print("2. Processing User Feedback")
    print("=" * 60)
    # Provide various types of feedback
    feedback_items = [
        {
            'execution_id': execution_ids[0],
            'type': FeedbackType.USER_RATING,
            'rating': 0.9,
            'comment': "Great answer!"
        },
        {
            'execution_id': execution_ids[1],
            'type': FeedbackType.USER_CORRECTION,
            'rating': 0.6,
            'corrections': {'detail_level': 'add_more_examples'},
            'comment': "Good but needs more examples"
        },
        {
            'execution_id': execution_ids[2],
            'type': FeedbackType.USER_RATING,
            'rating': 0.3,
            'comment': "Incorrect translation"
        }
    ]
    for feedback_item in feedback_items:
        result = agent.process_feedback(
            execution_id=feedback_item['execution_id'],
            feedback_type=feedback_item['type'],
            rating=feedback_item.get('rating'),
            comment=feedback_item.get('comment'),
            corrections=feedback_item.get('corrections')
        )
        print(f"\nFeedback processed: {result['feedback_id']}")
        print(f"Adaptations applied: {result['adaptations_applied']}")
        print(f"Current avg rating: {result['current_avg_rating']:.2f}")
    print("\n" + "=" * 60)
    print("3. Re-executing Tasks (With Learned Adaptations)")
    print("=" * 60)
    # Execute same tasks again to see improvements
    for task, context in tasks[:2]:
        result = agent.execute_with_feedback(task, context)
        print(f"\nTask: {task}")
        print(f"Output: {result['output']}")
        print(f"Adapted parameters: {result.get('adapted_parameters', {}).keys()}")
    print("\n" + "=" * 60)
    print("4. More Feedback (Building History)")
    print("=" * 60)
    # Add more feedback
    for i in range(5):
        task = f"Task {i+4}"
        result = agent.execute_with_feedback(task, {'task_type': 'general'})
        # Simulate improving ratings
        rating = 0.5 + (i * 0.1)
        agent.process_feedback(
            execution_id=result['execution_id'],
            feedback_type=FeedbackType.USER_RATING,
            rating=rating
        )
    print(f"\nAdded 5 more executions with improving ratings")
    print("\n" + "=" * 60)
    print("5. Improvement Metrics")
    print("=" * 60)
    metrics = agent.get_improvement_metrics(period_hours=1.0)
    print(f"\nPeriod: Last 1 hour")
    print(f"Total Executions: {metrics.total_executions}")
    print(f"Average Rating: {metrics.avg_rating:.2f}")
    print(f"Rating Trend: {metrics.rating_trend:+.4f} (positive = improving)")
    print(f"Feedback Count: {metrics.feedback_count}")
    print(f"Corrections Applied: {metrics.corrections_applied}")
    if metrics.performance_improvement:
        print("\nPerformance Improvements:")
        for metric, improvement in metrics.performance_improvement.items():
            print(f"  {metric}: {improvement:+.2%}")
    print("\n" + "=" * 60)
    print("6. Overall Statistics")
    print("=" * 60)
    stats = agent.get_statistics()
    print(f"\nTotal Executions: {stats['total_executions']}")
    print(f"Total Feedback Items: {stats['total_feedback_items']}")
    print(f"Average Rating: {stats['avg_rating']:.2f}")
    print(f"Rating Trend: {stats['rating_trend']:+.4f}")
    print("\nSentiment Distribution:")
    for sentiment, percentage in stats['sentiment_distribution'].items():
        print(f"  {sentiment}: {percentage:.1%}")
    if stats['common_corrections']:
        print("\nMost Common Corrections:")
        for correction in stats['common_corrections']:
            print(f"  {correction['type']}: {correction['count']} times")
    print(f"\nLearned Patterns (by type): {stats['learned_patterns']}")
    if stats['parameter_adjustments']:
        print("\nParameter Adjustments:")
        for param, adjustment in stats['parameter_adjustments'].items():
            print(f"  {param}: {adjustment:+.3f}")
    print("\n" + "=" * 60)
    print("Feedback Loops demonstration complete!")
    print("=" * 60)
if __name__ == "__main__":
    main()
