"""
Pattern 134: Task-Specific Agent

This pattern implements an agent optimized for a single, specific task with
high efficiency, specialized capabilities, and task-focused design.

Use Cases:
- Email classification
- Sentiment analysis
- Named entity recognition
- Code formatting
- Image captioning
- Specific API integration

Category: Specialization (2/6 = 33.3%)
Complexity: Intermediate
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import hashlib


class TaskType(Enum):
    """Types of specific tasks."""
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    GENERATION = "generation"
    ANALYSIS = "analysis"


class OptimizationLevel(Enum):
    """Optimization levels for task execution."""
    STANDARD = "standard"
    OPTIMIZED = "optimized"
    HIGH_PERFORMANCE = "high_performance"
    MAXIMUM = "maximum"


@dataclass
class TaskSpecification:
    """Specification of the specific task."""
    task_id: str
    task_type: TaskType
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    constraints: List[str] = field(default_factory=list)
    performance_targets: Dict[str, float] = field(default_factory=dict)


@dataclass
class TaskInput:
    """Input for task execution."""
    input_id: str
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskOutput:
    """Output from task execution."""
    input_id: str
    result: Any
    confidence: float
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """Performance metrics for task execution."""
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_execution_time_ms: float
    average_confidence: float
    throughput_per_second: float


class TaskCache:
    """Caches task results for optimization."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, TaskOutput] = {}
        self.access_count: Dict[str, int] = {}
    
    def get_cache_key(self, input_data: Any) -> str:
        """Generate cache key from input."""
        data_str = str(input_data)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, input_data: Any) -> Optional[TaskOutput]:
        """Get cached result."""
        key = self.get_cache_key(input_data)
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def put(self, input_data: Any, output: TaskOutput):
        """Cache result."""
        key = self.get_cache_key(input_data)
        
        # Evict least-used if cache full
        if len(self.cache) >= self.max_size:
            least_used_key = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_used_key]
            del self.access_count[least_used_key]
        
        self.cache[key] = output
        self.access_count[key] = 0
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()


class TaskValidator:
    """Validates task inputs and outputs."""
    
    def __init__(self, task_spec: TaskSpecification):
        self.task_spec = task_spec
    
    def validate_input(self, task_input: TaskInput) -> tuple[bool, Optional[str]]:
        """Validate input against schema."""
        # Simplified validation
        if task_input.data is None:
            return False, "Input data cannot be None"
        
        # Check type requirements
        expected_type = self.task_spec.input_schema.get('type')
        if expected_type:
            actual_type = type(task_input.data).__name__
            if actual_type != expected_type:
                return False, f"Expected {expected_type}, got {actual_type}"
        
        # Check constraints
        for constraint in self.task_spec.constraints:
            if not self._check_constraint(task_input.data, constraint):
                return False, f"Constraint violated: {constraint}"
        
        return True, None
    
    def validate_output(self, task_output: TaskOutput) -> tuple[bool, Optional[str]]:
        """Validate output against schema."""
        if task_output.result is None:
            return False, "Output result cannot be None"
        
        # Check output type
        expected_type = self.task_spec.output_schema.get('type')
        if expected_type:
            actual_type = type(task_output.result).__name__
            if actual_type != expected_type:
                return False, f"Expected {expected_type}, got {actual_type}"
        
        # Check confidence bounds
        if not 0.0 <= task_output.confidence <= 1.0:
            return False, "Confidence must be between 0 and 1"
        
        return True, None
    
    def _check_constraint(self, data: Any, constraint: str) -> bool:
        """Check if data meets constraint."""
        # Simplified constraint checking
        if constraint.startswith('min_length:'):
            min_len = int(constraint.split(':')[1])
            return len(str(data)) >= min_len
        elif constraint.startswith('max_length:'):
            max_len = int(constraint.split(':')[1])
            return len(str(data)) <= max_len
        elif constraint == 'not_empty':
            return bool(data)
        return True


class TaskOptimizer:
    """Optimizes task execution."""
    
    def __init__(self, optimization_level: OptimizationLevel):
        self.optimization_level = optimization_level
        self.optimization_stats = {
            'cached_hits': 0,
            'batch_optimizations': 0,
            'fast_path_taken': 0
        }
    
    def should_use_cache(self) -> bool:
        """Check if caching should be used."""
        return self.optimization_level in [
            OptimizationLevel.OPTIMIZED,
            OptimizationLevel.HIGH_PERFORMANCE,
            OptimizationLevel.MAXIMUM
        ]
    
    def should_use_batch_processing(self) -> bool:
        """Check if batch processing should be used."""
        return self.optimization_level in [
            OptimizationLevel.HIGH_PERFORMANCE,
            OptimizationLevel.MAXIMUM
        ]
    
    def should_use_fast_path(self, input_data: Any) -> bool:
        """Check if fast path can be used for simple cases."""
        if self.optimization_level != OptimizationLevel.MAXIMUM:
            return False
        
        # Simple heuristic: small inputs can use fast path
        data_size = len(str(input_data))
        return data_size < 100
    
    def optimize_execution(self, execution_func: Callable, *args, **kwargs) -> Any:
        """Apply optimizations to execution."""
        # Check for fast path
        if args and self.should_use_fast_path(args[0]):
            self.optimization_stats['fast_path_taken'] += 1
            # In real implementation, this would use optimized code path
        
        return execution_func(*args, **kwargs)


class TaskExecutor:
    """Executes the specific task."""
    
    def __init__(
        self,
        task_spec: TaskSpecification,
        execution_func: Callable[[Any], Any]
    ):
        self.task_spec = task_spec
        self.execution_func = execution_func
        self.execution_history: List[tuple[TaskInput, TaskOutput]] = []
    
    def execute(self, task_input: TaskInput) -> TaskOutput:
        """Execute the task."""
        start_time = datetime.now()
        
        # Execute task-specific function
        result = self.execution_func(task_input.data)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Calculate confidence (simplified)
        confidence = self._calculate_confidence(task_input.data, result)
        
        # Create output
        output = TaskOutput(
            input_id=task_input.input_id,
            result=result,
            confidence=confidence,
            execution_time_ms=execution_time,
            metadata={'task_type': self.task_spec.task_type.value}
        )
        
        # Record execution
        self.execution_history.append((task_input, output))
        
        return output
    
    def _calculate_confidence(self, input_data: Any, result: Any) -> float:
        """Calculate confidence in result."""
        # Simplified confidence calculation
        if result is None:
            return 0.0
        
        # Base confidence on result characteristics
        if isinstance(result, (int, float)):
            return 0.9
        elif isinstance(result, str):
            return 0.85 if len(result) > 0 else 0.3
        elif isinstance(result, (list, dict)):
            return 0.8 if result else 0.5
        
        return 0.7


class PerformanceMonitor:
    """Monitors task performance."""
    
    def __init__(self):
        self.execution_times: List[float] = []
        self.confidences: List[float] = []
        self.success_count = 0
        self.failure_count = 0
        self.start_time = datetime.now()
    
    def record_execution(self, output: TaskOutput, success: bool):
        """Record execution metrics."""
        self.execution_times.append(output.execution_time_ms)
        self.confidences.append(output.confidence)
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        total = self.success_count + self.failure_count
        
        avg_time = sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0
        avg_confidence = sum(self.confidences) / len(self.confidences) if self.confidences else 0
        
        # Calculate throughput
        elapsed_seconds = (datetime.now() - self.start_time).total_seconds()
        throughput = total / elapsed_seconds if elapsed_seconds > 0 else 0
        
        return PerformanceMetrics(
            total_executions=total,
            successful_executions=self.success_count,
            failed_executions=self.failure_count,
            average_execution_time_ms=avg_time,
            average_confidence=avg_confidence,
            throughput_per_second=throughput
        )
    
    def check_performance_targets(self, targets: Dict[str, float]) -> Dict[str, bool]:
        """Check if performance targets are met."""
        metrics = self.get_metrics()
        results = {}
        
        if 'max_execution_time_ms' in targets:
            results['execution_time'] = metrics.average_execution_time_ms <= targets['max_execution_time_ms']
        
        if 'min_confidence' in targets:
            results['confidence'] = metrics.average_confidence >= targets['min_confidence']
        
        if 'min_success_rate' in targets:
            success_rate = metrics.successful_executions / max(1, metrics.total_executions)
            results['success_rate'] = success_rate >= targets['min_success_rate']
        
        return results


class TaskSpecificAgent:
    """Agent optimized for a specific task."""
    
    def __init__(
        self,
        task_spec: TaskSpecification,
        execution_func: Callable[[Any], Any],
        optimization_level: OptimizationLevel = OptimizationLevel.OPTIMIZED
    ):
        self.task_spec = task_spec
        self.validator = TaskValidator(task_spec)
        self.executor = TaskExecutor(task_spec, execution_func)
        self.optimizer = TaskOptimizer(optimization_level)
        self.cache = TaskCache() if self.optimizer.should_use_cache() else None
        self.monitor = PerformanceMonitor()
    
    def process(self, input_data: Any, input_id: Optional[str] = None) -> TaskOutput:
        """Process input with the specific task."""
        # Create task input
        if input_id is None:
            input_id = hashlib.md5(str(input_data).encode()).hexdigest()[:8]
        
        task_input = TaskInput(input_id=input_id, data=input_data)
        
        # Validate input
        is_valid, error_msg = self.validator.validate_input(task_input)
        if not is_valid:
            raise ValueError(f"Invalid input: {error_msg}")
        
        # Check cache
        if self.cache:
            cached_output = self.cache.get(input_data)
            if cached_output:
                self.optimizer.optimization_stats['cached_hits'] += 1
                self.monitor.record_execution(cached_output, True)
                return cached_output
        
        # Execute task with optimizations
        output = self.optimizer.optimize_execution(self.executor.execute, task_input)
        
        # Validate output
        is_valid, error_msg = self.validator.validate_output(output)
        if not is_valid:
            self.monitor.record_execution(output, False)
            raise ValueError(f"Invalid output: {error_msg}")
        
        # Cache result
        if self.cache:
            self.cache.put(input_data, output)
        
        # Record metrics
        self.monitor.record_execution(output, True)
        
        return output
    
    def batch_process(self, input_list: List[Any]) -> List[TaskOutput]:
        """Process multiple inputs."""
        if self.optimizer.should_use_batch_processing():
            self.optimizer.optimization_stats['batch_optimizations'] += 1
            # In real implementation, this would use batch-optimized processing
        
        results = []
        for input_data in input_list:
            try:
                output = self.process(input_data)
                results.append(output)
            except Exception as e:
                # Create error output
                error_output = TaskOutput(
                    input_id="error",
                    result=None,
                    confidence=0.0,
                    execution_time_ms=0.0,
                    metadata={'error': str(e)}
                )
                results.append(error_output)
        
        return results
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        return self.monitor.get_metrics()
    
    def check_performance(self) -> Dict[str, bool]:
        """Check if performance targets are met."""
        return self.monitor.check_performance_targets(
            self.task_spec.performance_targets
        )
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            'optimization_level': self.optimizer.optimization_level.value,
            'cache_enabled': self.cache is not None,
            'cache_size': len(self.cache.cache) if self.cache else 0,
            **self.optimizer.optimization_stats
        }
    
    def reset_cache(self):
        """Reset the cache."""
        if self.cache:
            self.cache.clear()


# Task-specific implementations

def create_email_classifier() -> TaskSpecificAgent:
    """Create email classification agent."""
    task_spec = TaskSpecification(
        task_id="email_classifier",
        task_type=TaskType.CLASSIFICATION,
        name="Email Classifier",
        description="Classifies emails into categories",
        input_schema={'type': 'str'},
        output_schema={'type': 'str'},
        constraints=['not_empty', 'min_length:5'],
        performance_targets={
            'max_execution_time_ms': 50.0,
            'min_confidence': 0.7,
            'min_success_rate': 0.95
        }
    )
    
    def classify_email(email_text: str) -> str:
        """Classify email into category."""
        email_lower = email_text.lower()
        
        # Simple keyword-based classification
        if any(word in email_lower for word in ['urgent', 'asap', 'important', 'critical']):
            return 'urgent'
        elif any(word in email_lower for word in ['meeting', 'schedule', 'calendar']):
            return 'meeting'
        elif any(word in email_lower for word in ['invoice', 'payment', 'bill']):
            return 'finance'
        elif any(word in email_lower for word in ['support', 'help', 'issue', 'problem']):
            return 'support'
        else:
            return 'general'
    
    return TaskSpecificAgent(
        task_spec=task_spec,
        execution_func=classify_email,
        optimization_level=OptimizationLevel.HIGH_PERFORMANCE
    )


def create_sentiment_analyzer() -> TaskSpecificAgent:
    """Create sentiment analysis agent."""
    task_spec = TaskSpecification(
        task_id="sentiment_analyzer",
        task_type=TaskType.ANALYSIS,
        name="Sentiment Analyzer",
        description="Analyzes sentiment of text",
        input_schema={'type': 'str'},
        output_schema={'type': 'dict'},
        constraints=['not_empty'],
        performance_targets={
            'max_execution_time_ms': 30.0,
            'min_confidence': 0.75
        }
    )
    
    def analyze_sentiment(text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        text_lower = text.lower()
        
        # Simple sentiment analysis
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'poor'}
        
        words = set(text_lower.split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        
        if pos_count > neg_count:
            sentiment = 'positive'
            score = min(1.0, 0.5 + (pos_count - neg_count) * 0.1)
        elif neg_count > pos_count:
            sentiment = 'negative'
            score = max(-1.0, -0.5 - (neg_count - pos_count) * 0.1)
        else:
            sentiment = 'neutral'
            score = 0.0
        
        return {
            'sentiment': sentiment,
            'score': score,
            'positive_words': pos_count,
            'negative_words': neg_count
        }
    
    return TaskSpecificAgent(
        task_spec=task_spec,
        execution_func=analyze_sentiment,
        optimization_level=OptimizationLevel.MAXIMUM
    )


def create_entity_extractor() -> TaskSpecificAgent:
    """Create named entity extraction agent."""
    task_spec = TaskSpecification(
        task_id="entity_extractor",
        task_type=TaskType.EXTRACTION,
        name="Entity Extractor",
        description="Extracts named entities from text",
        input_schema={'type': 'str'},
        output_schema={'type': 'dict'},
        constraints=['not_empty'],
        performance_targets={
            'max_execution_time_ms': 40.0,
            'min_confidence': 0.65
        }
    )
    
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """Extract named entities."""
        words = text.split()
        
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': []
        }
        
        # Simple pattern-based extraction
        for i, word in enumerate(words):
            # Capitalized words might be entities
            if word and word[0].isupper() and len(word) > 1:
                # Check context for type
                prev_word = words[i-1].lower() if i > 0 else ''
                
                if prev_word in ['mr', 'ms', 'dr', 'prof']:
                    entities['persons'].append(word)
                elif prev_word in ['at', 'in', 'from', 'to']:
                    entities['locations'].append(word)
                elif prev_word in ['company', 'corp', 'inc']:
                    entities['organizations'].append(word)
                elif word in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                    entities['dates'].append(word)
        
        return entities
    
    return TaskSpecificAgent(
        task_spec=task_spec,
        execution_func=extract_entities,
        optimization_level=OptimizationLevel.OPTIMIZED
    )


def create_text_formatter() -> TaskSpecificAgent:
    """Create text formatting agent."""
    task_spec = TaskSpecification(
        task_id="text_formatter",
        task_type=TaskType.TRANSFORMATION,
        name="Text Formatter",
        description="Formats text according to rules",
        input_schema={'type': 'str'},
        output_schema={'type': 'str'},
        constraints=['not_empty'],
        performance_targets={
            'max_execution_time_ms': 20.0,
            'min_success_rate': 0.99
        }
    )
    
    def format_text(text: str) -> str:
        """Format text (capitalize sentences, fix spacing)."""
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Capitalize first letter of sentences
        sentences = text.split('.')
        formatted = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                formatted.append(sentence)
        
        return '. '.join(formatted) + ('.' if formatted else '')
    
    return TaskSpecificAgent(
        task_spec=task_spec,
        execution_func=format_text,
        optimization_level=OptimizationLevel.MAXIMUM
    )


def demonstrate_task_specific_agent():
    """Demonstrate the Task-Specific Agent pattern."""
    print("=" * 60)
    print("Task-Specific Agent Demonstration")
    print("=" * 60)
    
    print("\n1. EMAIL CLASSIFIER")
    print("-" * 60)
    
    email_classifier = create_email_classifier()
    
    emails = [
        "URGENT: Server down, need immediate attention!",
        "Meeting scheduled for tomorrow at 10am",
        "Invoice #12345 for payment due next week",
        "Help needed with login issue",
        "Just checking in to say hello"
    ]
    
    print(f"Classifying {len(emails)} emails...")
    for email in emails:
        output = email_classifier.process(email)
        print(f"\nEmail: {email[:40]}...")
        print(f"  Category: {output.result}")
        print(f"  Confidence: {output.confidence:.2f}")
        print(f"  Time: {output.execution_time_ms:.2f}ms")
    
    metrics = email_classifier.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Total: {metrics.total_executions}")
    print(f"  Success Rate: {metrics.successful_executions/metrics.total_executions*100:.1f}%")
    print(f"  Avg Time: {metrics.average_execution_time_ms:.2f}ms")
    print(f"  Avg Confidence: {metrics.average_confidence:.2f}")
    
    targets_met = email_classifier.check_performance()
    print(f"  Performance Targets: {'✓ Met' if all(targets_met.values()) else '✗ Not Met'}")
    
    print("\n\n2. SENTIMENT ANALYZER")
    print("-" * 60)
    
    sentiment_analyzer = create_sentiment_analyzer()
    
    texts = [
        "This product is amazing! I love it so much!",
        "Terrible experience, worst purchase ever.",
        "It's okay, nothing special about it.",
        "Great quality and excellent customer service!"
    ]
    
    print(f"Analyzing sentiment of {len(texts)} texts...")
    for text in texts:
        output = sentiment_analyzer.process(text)
        result = output.result
        print(f"\nText: {text[:40]}...")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Score: {result['score']:.2f}")
        print(f"  Positive words: {result['positive_words']}")
        print(f"  Negative words: {result['negative_words']}")
        print(f"  Confidence: {output.confidence:.2f}")
    
    print("\n\n3. ENTITY EXTRACTOR")
    print("-" * 60)
    
    entity_extractor = create_entity_extractor()
    
    texts = [
        "Dr Smith from Microsoft will visit New York on Monday",
        "The meeting at Google in California is confirmed",
        "Prof Johnson teaches at Stanford University"
    ]
    
    print(f"Extracting entities from {len(texts)} texts...")
    for text in texts:
        output = entity_extractor.process(text)
        entities = output.result
        print(f"\nText: {text}")
        print(f"  Persons: {entities['persons']}")
        print(f"  Organizations: {entities['organizations']}")
        print(f"  Locations: {entities['locations']}")
        print(f"  Dates: {entities['dates']}")
        print(f"  Confidence: {output.confidence:.2f}")
    
    print("\n\n4. TEXT FORMATTER")
    print("-" * 60)
    
    text_formatter = create_text_formatter()
    
    texts = [
        "hello world. this is a test. multiple   spaces   here.",
        "first sentence.second sentence without space.",
        "no ending punctuation"
    ]
    
    print("Formatting texts...")
    for text in texts:
        output = text_formatter.process(text)
        print(f"\nOriginal: {text}")
        print(f"Formatted: {output.result}")
        print(f"Time: {output.execution_time_ms:.2f}ms")
    
    print("\n\n5. BATCH PROCESSING")
    print("-" * 60)
    
    batch_emails = [
        "URGENT deadline approaching",
        "Schedule a meeting for next week",
        "Invoice payment reminder",
        "Technical support required",
        "General inquiry about product"
    ]
    
    print(f"Batch processing {len(batch_emails)} emails...")
    batch_results = email_classifier.batch_process(batch_emails)
    
    categories_count = {}
    for output in batch_results:
        if output.result:
            categories_count[output.result] = categories_count.get(output.result, 0) + 1
    
    print(f"\nCategory Distribution:")
    for category, count in sorted(categories_count.items()):
        print(f"  {category}: {count}")
    
    print("\n\n6. CACHING DEMONSTRATION")
    print("-" * 60)
    
    # Process same email twice to demonstrate caching
    test_email = "URGENT: Critical system failure!"
    
    print("First execution (no cache):")
    output1 = email_classifier.process(test_email)
    print(f"  Time: {output1.execution_time_ms:.2f}ms")
    
    print("\nSecond execution (cached):")
    output2 = email_classifier.process(test_email)
    print(f"  Time: {output2.execution_time_ms:.2f}ms")
    
    opt_stats = email_classifier.get_optimization_stats()
    print(f"\nOptimization Stats:")
    print(f"  Cache Enabled: {opt_stats['cache_enabled']}")
    print(f"  Cache Hits: {opt_stats['cached_hits']}")
    print(f"  Cache Size: {opt_stats['cache_size']}")
    print(f"  Batch Optimizations: {opt_stats['batch_optimizations']}")
    
    print("\n\n7. OPTIMIZATION LEVELS COMPARISON")
    print("-" * 60)
    
    optimization_levels = [
        OptimizationLevel.STANDARD,
        OptimizationLevel.OPTIMIZED,
        OptimizationLevel.HIGH_PERFORMANCE,
        OptimizationLevel.MAXIMUM
    ]
    
    test_text = "This is a test message for optimization comparison"
    
    print("Testing same task with different optimization levels:\n")
    for level in optimization_levels:
        # Create temporary agent with specific optimization level
        temp_spec = TaskSpecification(
            task_id="temp",
            task_type=TaskType.ANALYSIS,
            name="Temp",
            description="Temp",
            input_schema={'type': 'str'},
            output_schema={'type': 'dict'}
        )
        
        temp_agent = TaskSpecificAgent(
            task_spec=temp_spec,
            execution_func=lambda x: {'result': len(x)},
            optimization_level=level
        )
        
        output = temp_agent.process(test_text)
        opt_stats = temp_agent.get_optimization_stats()
        
        print(f"{level.value}:")
        print(f"  Cache Enabled: {opt_stats['cache_enabled']}")
        print(f"  Execution Time: {output.execution_time_ms:.2f}ms")
    
    print("\n\n8. OVERALL STATISTICS")
    print("-" * 60)
    
    print("\nEmail Classifier:")
    metrics = email_classifier.get_performance_metrics()
    print(f"  Executions: {metrics.total_executions}")
    print(f"  Success Rate: {metrics.successful_executions/max(1, metrics.total_executions)*100:.1f}%")
    print(f"  Avg Time: {metrics.average_execution_time_ms:.2f}ms")
    print(f"  Throughput: {metrics.throughput_per_second:.2f} ops/sec")
    
    print("\nSentiment Analyzer:")
    metrics = sentiment_analyzer.get_performance_metrics()
    print(f"  Executions: {metrics.total_executions}")
    print(f"  Avg Confidence: {metrics.average_confidence:.2f}")
    print(f"  Avg Time: {metrics.average_execution_time_ms:.2f}ms")
    
    print("\n" + "=" * 60)
    print("🎉 Pattern 134 Complete!")
    print("Specialization Category: 33.3%")
    print("134/170 patterns implemented (78.8%)!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_task_specific_agent()
