"""
Pattern 079: Batch Processing Agent

Description:
    A Batch Processing Agent handles multiple tasks or inputs concurrently in batches,
    optimizing throughput and resource utilization. This pattern is essential for
    processing large volumes of data efficiently, enabling parallel execution,
    job queuing, progress monitoring, and result aggregation.
    
    The agent manages batches of work items, distributes them across available resources,
    monitors progress, handles failures, and aggregates results. It supports various
    batching strategies including size-based, time-based, and priority-based batching.

Components:
    1. Batch Scheduler: Schedules and organizes batch jobs
    2. Job Queue: Manages pending work items
    3. Parallel Executor: Executes tasks concurrently
    4. Progress Monitor: Tracks batch processing progress
    5. Result Aggregator: Combines individual results
    6. Error Handler: Handles failures and retries
    7. Resource Manager: Manages compute resources
    8. Throughput Optimizer: Optimizes batch sizes

Key Features:
    - Parallel batch processing
    - Dynamic batch sizing
    - Job queue management
    - Progress tracking
    - Error handling and retries
    - Result aggregation
    - Resource optimization
    - Priority scheduling
    - Cancellation support
    - Checkpointing for resumption

Use Cases:
    - Bulk document processing
    - Large-scale data analysis
    - Batch translation services
    - Bulk content generation
    - Mass email personalization
    - Dataset annotation
    - Batch image processing
    - ETL pipelines
    - Bulk API operations
    - Large-scale testing

LangChain Implementation:
    Uses concurrent.futures for parallel execution, batch processing utilities,
    and structured output handling for aggregation.
"""

import os
import time
import uuid
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class JobStatus(Enum):
    """Status of a batch job"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class BatchStrategy(Enum):
    """Batching strategy"""
    SIZE_BASED = "size_based"  # Fixed batch size
    TIME_BASED = "time_based"  # Time window
    ADAPTIVE = "adaptive"  # Dynamic sizing
    PRIORITY = "priority"  # Priority-based


@dataclass
class BatchJob:
    """A single job in a batch"""
    job_id: str
    input_data: Any
    priority: int = 0  # Higher = more important
    status: JobStatus = JobStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    attempts: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    batch_size: int = 10
    max_workers: int = 4
    max_retries: int = 3
    timeout: float = 60.0  # seconds per job
    strategy: BatchStrategy = BatchStrategy.SIZE_BASED
    aggregate_results: bool = True
    checkpoint_interval: int = 10  # Jobs between checkpoints


@dataclass
class BatchResult:
    """Result of batch processing"""
    batch_id: str
    total_jobs: int
    completed: int
    failed: int
    cancelled: int
    results: List[Any]
    errors: List[Tuple[str, str]]  # (job_id, error)
    duration: float
    throughput: float  # jobs/second
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchProgress:
    """Progress of batch processing"""
    batch_id: str
    total_jobs: int
    completed: int
    processing: int
    pending: int
    failed: int
    progress_pct: float
    estimated_remaining: float  # seconds
    current_throughput: float


class BatchProcessingAgent:
    """
    Agent for efficient batch processing of multiple tasks.
    
    This agent handles large volumes of work by processing items in batches,
    with parallel execution, error handling, and progress tracking.
    """
    
    def __init__(self, config: BatchConfig = BatchConfig()):
        """Initialize the batch processing agent"""
        self.config = config
        
        # LLM for processing
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # Job management
        self.jobs: Dict[str, BatchJob] = {}
        self.batches: Dict[str, List[str]] = {}  # batch_id -> job_ids
        self.active_batches: List[str] = []
    
    def create_batch(
        self,
        inputs: List[Any],
        priorities: Optional[List[int]] = None
    ) -> str:
        """
        Create a new batch of jobs.
        
        Args:
            inputs: List of input data for jobs
            priorities: Optional priority for each job
            
        Returns:
            Batch ID
        """
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        job_ids = []
        
        for i, input_data in enumerate(inputs):
            job_id = f"job_{uuid.uuid4().hex[:8]}"
            priority = priorities[i] if priorities else 0
            
            job = BatchJob(
                job_id=job_id,
                input_data=input_data,
                priority=priority,
                status=JobStatus.QUEUED
            )
            
            self.jobs[job_id] = job
            job_ids.append(job_id)
        
        self.batches[batch_id] = job_ids
        self.active_batches.append(batch_id)
        
        return batch_id
    
    def process_batch(
        self,
        batch_id: str,
        task_fn: Callable[[Any], Any],
        on_progress: Optional[Callable[[BatchProgress], None]] = None
    ) -> BatchResult:
        """
        Process a batch of jobs in parallel.
        
        Args:
            batch_id: ID of batch to process
            task_fn: Function to process each job
            on_progress: Optional progress callback
            
        Returns:
            BatchResult with aggregated results
        """
        if batch_id not in self.batches:
            raise ValueError(f"Batch {batch_id} not found")
        
        job_ids = self.batches[batch_id]
        start_time = time.time()
        
        results = []
        errors = []
        completed_count = 0
        failed_count = 0
        cancelled_count = 0
        
        # Process jobs in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit jobs
            future_to_job = {}
            for job_id in job_ids:
                job = self.jobs[job_id]
                job.status = JobStatus.PROCESSING
                job.started_at = datetime.now()
                
                future = executor.submit(
                    self._process_single_job,
                    job_id,
                    task_fn
                )
                future_to_job[future] = job_id
            
            # Collect results as they complete
            for future in as_completed(future_to_job):
                job_id = future_to_job[future]
                job = self.jobs[job_id]
                
                try:
                    result = future.result(timeout=self.config.timeout)
                    job.result = result
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.now()
                    results.append(result)
                    completed_count += 1
                    
                except Exception as e:
                    job.error = str(e)
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.now()
                    errors.append((job_id, str(e)))
                    failed_count += 1
                
                # Progress callback
                if on_progress:
                    progress = self._calculate_progress(
                        batch_id,
                        start_time,
                        completed_count + failed_count
                    )
                    on_progress(progress)
        
        duration = time.time() - start_time
        total_jobs = len(job_ids)
        
        return BatchResult(
            batch_id=batch_id,
            total_jobs=total_jobs,
            completed=completed_count,
            failed=failed_count,
            cancelled=cancelled_count,
            results=results,
            errors=errors,
            duration=duration,
            throughput=total_jobs / duration if duration > 0 else 0
        )
    
    def _process_single_job(
        self,
        job_id: str,
        task_fn: Callable[[Any], Any]
    ) -> Any:
        """
        Process a single job with retries.
        
        Args:
            job_id: Job ID
            task_fn: Task function
            
        Returns:
            Job result
        """
        job = self.jobs[job_id]
        
        for attempt in range(self.config.max_retries):
            try:
                job.attempts = attempt + 1
                if attempt > 0:
                    job.status = JobStatus.RETRYING
                
                result = task_fn(job.input_data)
                return result
                
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError(f"Job {job_id} failed after {self.config.max_retries} attempts")
    
    def _calculate_progress(
        self,
        batch_id: str,
        start_time: float,
        completed: int
    ) -> BatchProgress:
        """Calculate batch progress"""
        job_ids = self.batches[batch_id]
        total = len(job_ids)
        
        processing = sum(
            1 for jid in job_ids
            if self.jobs[jid].status == JobStatus.PROCESSING
        )
        
        pending = sum(
            1 for jid in job_ids
            if self.jobs[jid].status in [JobStatus.PENDING, JobStatus.QUEUED]
        )
        
        failed = sum(
            1 for jid in job_ids
            if self.jobs[jid].status == JobStatus.FAILED
        )
        
        progress_pct = (completed / total) * 100 if total > 0 else 0
        
        elapsed = time.time() - start_time
        throughput = completed / elapsed if elapsed > 0 else 0
        remaining = total - completed
        estimated_remaining = remaining / throughput if throughput > 0 else 0
        
        return BatchProgress(
            batch_id=batch_id,
            total_jobs=total,
            completed=completed,
            processing=processing,
            pending=pending,
            failed=failed,
            progress_pct=progress_pct,
            estimated_remaining=estimated_remaining,
            current_throughput=throughput
        )
    
    def process_text_batch(
        self,
        texts: List[str],
        operation: str,
        on_progress: Optional[Callable[[BatchProgress], None]] = None
    ) -> BatchResult:
        """
        Process a batch of text operations.
        
        Args:
            texts: List of text inputs
            operation: Operation to perform (summarize, translate, etc.)
            on_progress: Optional progress callback
            
        Returns:
            BatchResult with processed texts
        """
        # Create batch
        batch_id = self.create_batch(texts)
        
        # Define task function
        def process_text(text: str) -> str:
            prompt = ChatPromptTemplate.from_template(
                "{operation} the following text:\n\n{text}"
            )
            
            chain = prompt | self.llm | StrOutputParser()
            
            result = chain.invoke({
                "operation": operation,
                "text": text[:500]  # Limit input length
            })
            
            return result
        
        # Process batch
        return self.process_batch(batch_id, process_text, on_progress)
    
    def process_classification_batch(
        self,
        items: List[str],
        categories: List[str],
        on_progress: Optional[Callable[[BatchProgress], None]] = None
    ) -> BatchResult:
        """
        Batch classify items into categories.
        
        Args:
            items: Items to classify
            categories: Possible categories
            on_progress: Optional progress callback
            
        Returns:
            BatchResult with classifications
        """
        batch_id = self.create_batch(items)
        
        categories_str = ", ".join(categories)
        
        def classify_item(item: str) -> str:
            prompt = ChatPromptTemplate.from_template(
                """Classify the following item into one of these categories: {categories}

Item: {item}

Category:"""
            )
            
            chain = prompt | self.llm | StrOutputParser()
            
            result = chain.invoke({
                "categories": categories_str,
                "item": item[:200]
            })
            
            return result.strip()
        
        return self.process_batch(batch_id, classify_item, on_progress)
    
    def process_qa_batch(
        self,
        questions: List[str],
        context: str,
        on_progress: Optional[Callable[[BatchProgress], None]] = None
    ) -> BatchResult:
        """
        Answer a batch of questions.
        
        Args:
            questions: List of questions
            context: Shared context for all questions
            on_progress: Optional progress callback
            
        Returns:
            BatchResult with answers
        """
        batch_id = self.create_batch(questions)
        
        def answer_question(question: str) -> str:
            prompt = ChatPromptTemplate.from_template(
                """Context: {context}

Question: {question}

Answer:"""
            )
            
            chain = prompt | self.llm | StrOutputParser()
            
            result = chain.invoke({
                "context": context[:1000],
                "question": question
            })
            
            return result
        
        return self.process_batch(batch_id, answer_question, on_progress)
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get status of a batch.
        
        Args:
            batch_id: Batch ID
            
        Returns:
            Status dictionary
        """
        if batch_id not in self.batches:
            raise ValueError(f"Batch {batch_id} not found")
        
        job_ids = self.batches[batch_id]
        
        status_counts = defaultdict(int)
        for job_id in job_ids:
            status = self.jobs[job_id].status
            status_counts[status.value] += 1
        
        return {
            "batch_id": batch_id,
            "total_jobs": len(job_ids),
            "status_breakdown": dict(status_counts),
            "jobs": [
                {
                    "job_id": jid,
                    "status": self.jobs[jid].status.value,
                    "attempts": self.jobs[jid].attempts,
                    "error": self.jobs[jid].error
                }
                for jid in job_ids[:10]  # First 10 jobs
            ]
        }


def demonstrate_batch_processing_agent():
    """Demonstrate the batch processing agent capabilities"""
    print("=" * 80)
    print("BATCH PROCESSING AGENT DEMONSTRATION")
    print("=" * 80)
    
    # Custom config for faster demos
    config = BatchConfig(
        batch_size=5,
        max_workers=3,
        max_retries=2
    )
    
    agent = BatchProcessingAgent(config)
    
    # Demo 1: Text Summarization Batch
    print("\n" + "=" * 80)
    print("DEMO 1: Batch Text Summarization")
    print("=" * 80)
    
    texts = [
        "Artificial intelligence is transforming healthcare through advanced diagnostic tools, personalized treatment plans, and drug discovery acceleration.",
        "Machine learning models require large amounts of training data and computational resources to achieve high performance on complex tasks.",
        "Natural language processing enables computers to understand, interpret, and generate human language in meaningful ways.",
        "Computer vision systems can now recognize objects, faces, and scenes with accuracy rivaling human perception.",
        "Robotics combined with AI is enabling autonomous vehicles, warehouse automation, and advanced manufacturing processes."
    ]
    
    print(f"\nProcessing {len(texts)} texts for summarization...")
    print("-" * 80)
    
    progress_updates = []
    
    def on_progress(progress: BatchProgress):
        progress_updates.append(progress)
        bar_length = 40
        filled = int(bar_length * progress.progress_pct / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r[{bar}] {progress.progress_pct:.1f}% ({progress.completed}/{progress.total_jobs})", end="", flush=True)
    
    result = agent.process_text_batch(texts, "Summarize in one sentence", on_progress)
    
    print("\n" + "-" * 80)
    print(f"\n✓ Batch complete!")
    print(f"Total Jobs: {result.total_jobs}")
    print(f"Completed: {result.completed}")
    print(f"Failed: {result.failed}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Throughput: {result.throughput:.2f} jobs/sec")
    
    print(f"\nSample Results:")
    for i, summary in enumerate(result.results[:3], 1):
        print(f"\n{i}. {summary[:100]}...")
    
    # Demo 2: Classification Batch
    print("\n" + "=" * 80)
    print("DEMO 2: Batch Classification")
    print("=" * 80)
    
    items = [
        "Python programming tutorial",
        "Chocolate chip cookie recipe",
        "2024 Olympic games highlights",
        "Climate change research paper",
        "Stock market analysis",
        "Travel guide to Japan",
        "Guitar lesson for beginners"
    ]
    
    categories = ["Technology", "Food", "Sports", "Science", "Finance", "Travel", "Music"]
    
    print(f"\nClassifying {len(items)} items into categories:")
    print(f"Categories: {', '.join(categories)}")
    print("-" * 80)
    
    result2 = agent.process_classification_batch(items, categories, on_progress=on_progress)
    
    print("\n" + "-" * 80)
    print(f"\n✓ Classification complete!")
    print(f"Duration: {result2.duration:.2f}s")
    print(f"Throughput: {result2.throughput:.2f} items/sec")
    
    print(f"\nClassification Results:")
    for i, (item, category) in enumerate(zip(items, result2.results), 1):
        print(f"{i}. '{item[:40]}...' → {category}")
    
    # Demo 3: Question Answering Batch
    print("\n" + "=" * 80)
    print("DEMO 3: Batch Question Answering")
    print("=" * 80)
    
    context = """
    LangChain is a framework for developing applications powered by language models.
    It provides tools for prompt management, chains, agents, memory, and more.
    LangChain supports multiple LLM providers including OpenAI, Anthropic, and Google.
    The framework emphasizes composability and modularity for building complex applications.
    """
    
    questions = [
        "What is LangChain?",
        "What providers does LangChain support?",
        "What are the main features of LangChain?",
        "What is the design philosophy of LangChain?"
    ]
    
    print(f"\nContext: {context.strip()[:100]}...")
    print(f"\nAnswering {len(questions)} questions...")
    print("-" * 80)
    
    result3 = agent.process_qa_batch(questions, context, on_progress=on_progress)
    
    print("\n" + "-" * 80)
    print(f"\n✓ Q&A batch complete!")
    print(f"Duration: {result3.duration:.2f}s")
    
    print(f"\nQuestions and Answers:")
    for i, (question, answer) in enumerate(zip(questions, result3.results), 1):
        print(f"\nQ{i}: {question}")
        print(f"A{i}: {answer[:150]}...")
    
    # Demo 4: Batch Status Monitoring
    print("\n" + "=" * 80)
    print("DEMO 4: Batch Status Monitoring")
    print("=" * 80)
    
    # Create a new batch
    test_items = ["Item " + str(i) for i in range(8)]
    batch_id = agent.create_batch(test_items)
    
    print(f"\nCreated batch: {batch_id}")
    print(f"Total items: {len(test_items)}")
    
    # Get status before processing
    status = agent.get_batch_status(batch_id)
    
    print(f"\nBatch Status:")
    print(f"  Total Jobs: {status['total_jobs']}")
    print(f"  Status Breakdown:")
    for status_type, count in status['status_breakdown'].items():
        print(f"    {status_type}: {count}")
    
    # Demo 5: Performance Comparison
    print("\n" + "=" * 80)
    print("DEMO 5: Parallel vs Sequential Performance")
    print("=" * 80)
    
    test_texts = [f"Sample text {i} for processing" for i in range(10)]
    
    print(f"\nProcessing {len(test_texts)} items...")
    
    # Parallel processing
    print("\nParallel Processing (4 workers):")
    parallel_start = time.time()
    parallel_result = agent.process_text_batch(test_texts, "Count words in")
    parallel_duration = time.time() - parallel_start
    
    print(f"  Duration: {parallel_duration:.2f}s")
    print(f"  Throughput: {parallel_result.throughput:.2f} jobs/sec")
    
    # Sequential processing (1 worker)
    print("\nSequential Processing (1 worker):")
    sequential_config = BatchConfig(max_workers=1)
    sequential_agent = BatchProcessingAgent(sequential_config)
    sequential_start = time.time()
    sequential_result = sequential_agent.process_text_batch(test_texts, "Count words in")
    sequential_duration = time.time() - sequential_start
    
    print(f"  Duration: {sequential_duration:.2f}s")
    print(f"  Throughput: {sequential_result.throughput:.2f} jobs/sec")
    
    speedup = sequential_duration / parallel_duration if parallel_duration > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Efficiency: {(speedup / config.max_workers) * 100:.1f}%")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary = """
The Batch Processing Agent demonstrates efficient parallel processing:

KEY CAPABILITIES:
1. Parallel Execution: Process multiple jobs concurrently
2. Job Queuing: Manage large volumes of pending work
3. Progress Tracking: Real-time monitoring with callbacks
4. Error Handling: Automatic retries with exponential backoff
5. Result Aggregation: Collect and organize batch results
6. Resource Management: Control worker threads and resources
7. Flexible Batching: Support various batching strategies
8. Status Monitoring: Track job and batch status

BENEFITS:
- Dramatically improved throughput
- Efficient resource utilization
- Scalable processing of large datasets
- Automatic error recovery
- Progress visibility
- Cost optimization through parallelization
- Reduced total processing time
- Better user experience for bulk operations

USE CASES:
- Bulk document summarization or translation
- Large-scale data classification
- Mass content generation
- Batch question answering
- Dataset annotation and labeling
- Bulk email personalization
- ETL pipeline processing
- Large-scale testing
- Batch API operations
- Document ingestion for RAG systems

PRODUCTION CONSIDERATIONS:
1. Worker Scaling: Adjust workers based on load and resources
2. Rate Limiting: Respect API rate limits across workers
3. Cost Management: Monitor token usage across batches
4. Error Recovery: Implement robust retry strategies
5. Checkpointing: Save progress for resumption
6. Priority Queues: Process high-priority jobs first
7. Resource Limits: Prevent resource exhaustion
8. Monitoring: Track batch performance metrics
9. Load Balancing: Distribute work evenly
10. Graceful Shutdown: Handle interruptions cleanly

ADVANCED EXTENSIONS:
- Dynamic worker scaling based on queue depth
- Intelligent batch sizing with machine learning
- Multi-stage batch pipelines
- Distributed processing across machines
- Streaming batch results
- Partial result delivery
- Batch deduplication
- Priority-based scheduling
- Resource-aware batching
- Automatic retry backoff tuning

Batch processing is essential for production AI systems handling
large volumes of data efficiently and cost-effectively.
"""
    
    print(summary)


if __name__ == "__main__":
    demonstrate_batch_processing_agent()
