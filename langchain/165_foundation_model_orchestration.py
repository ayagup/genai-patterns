"""
Pattern 165: Foundation Model Orchestration

Description:
    The Foundation Model Orchestration pattern coordinates multiple foundation models
    (LLMs with different capabilities, sizes, and specializations) to solve complex tasks
    optimally. The orchestrator routes tasks to the most appropriate model based on
    requirements, capabilities, cost, and performance trade-offs.

Components:
    1. Model Registry: Catalog of available models and their capabilities
    2. Task Analyzer: Analyzes task requirements and complexity
    3. Model Selector: Chooses optimal model for each task
    4. Result Aggregator: Combines results from multiple models
    5. Performance Monitor: Tracks model performance and costs
    6. Fallback Manager: Handles model failures

Use Cases:
    - Cost-optimized AI pipelines
    - Quality-performance trade-offs
    - Specialized task routing
    - Multi-model ensemble systems
    - Adaptive model selection
    - Resource optimization

Benefits:
    - Optimal cost-performance balance
    - Better quality through specialization
    - Fault tolerance with fallbacks
    - Flexible model upgrades
    - Resource efficiency

Trade-offs:
    - Orchestration complexity
    - Latency overhead
    - Model compatibility issues
    - Cost tracking complexity
    - Requires multiple API keys

LangChain Implementation:
    Uses LangChain's model abstraction to coordinate multiple models. Implements
    intelligent routing logic based on task characteristics and model capabilities.
"""

import os
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class ModelCapability(Enum):
    """Model capability types"""
    REASONING = "reasoning"
    CODING = "coding"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    FAST_RESPONSE = "fast_response"
    LONG_CONTEXT = "long_context"


class ModelTier(Enum):
    """Model tier for cost/performance trade-off"""
    NANO = "nano"  # Fastest, cheapest, basic
    SMALL = "small"  # Fast, cheap, good quality
    MEDIUM = "medium"  # Balanced
    LARGE = "large"  # High quality, slower, expensive
    PREMIUM = "premium"  # Best quality, slowest, most expensive


@dataclass
class ModelProfile:
    """Profile of a foundation model"""
    name: str
    model_id: str
    tier: ModelTier
    capabilities: List[ModelCapability]
    max_tokens: int
    cost_per_1k_tokens: float
    avg_latency_ms: float
    quality_score: float = 0.8  # 0-1 scale
    available: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """A task to be processed"""
    id: str
    content: str
    required_capabilities: List[ModelCapability]
    priority: str = "medium"  # low, medium, high
    max_cost: Optional[float] = None
    max_latency: Optional[float] = None
    min_quality: float = 0.7


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    model_used: str
    result: str
    execution_time: float
    estimated_cost: float
    quality_estimate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class FoundationModelOrchestrator:
    """Orchestrates multiple foundation models for optimal task execution"""
    
    def __init__(self):
        """Initialize the orchestrator with model registry"""
        self.models: Dict[str, ModelProfile] = {}
        self.llm_instances: Dict[str, ChatOpenAI] = {}
        self.performance_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "tasks_completed": 0,
            "total_cost": 0.0,
            "total_latency": 0.0,
            "failures": 0
        })
        
        # Register models
        self._register_default_models()
        
        # Task routing prompt
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a task analysis expert. Analyze the task and determine
            which capabilities are most important for successful completion."""),
            ("user", """Task: {task}

Analyze this task and identify the key capabilities needed (choose from: reasoning,
coding, creative, analysis, summarization, translation, fast_response, long_context).

Respond with just the capability names, comma-separated.""")
        ])
    
    def _register_default_models(self):
        """Register default model profiles"""
        # Nano tier - fast and cheap
        self.register_model(ModelProfile(
            name="GPT-3.5 Turbo",
            model_id="gpt-3.5-turbo",
            tier=ModelTier.SMALL,
            capabilities=[
                ModelCapability.FAST_RESPONSE,
                ModelCapability.SUMMARIZATION,
                ModelCapability.ANALYSIS
            ],
            max_tokens=4096,
            cost_per_1k_tokens=0.002,
            avg_latency_ms=800,
            quality_score=0.75
        ))
        
        # Medium tier - balanced
        self.register_model(ModelProfile(
            name="GPT-4 Turbo",
            model_id="gpt-4-turbo-preview",
            tier=ModelTier.LARGE,
            capabilities=[
                ModelCapability.REASONING,
                ModelCapability.CODING,
                ModelCapability.CREATIVE,
                ModelCapability.ANALYSIS,
                ModelCapability.LONG_CONTEXT
            ],
            max_tokens=128000,
            cost_per_1k_tokens=0.01,
            avg_latency_ms=2000,
            quality_score=0.95
        ))
        
        # Specialized models (simulated with different configs)
        self.register_model(ModelProfile(
            name="Code Specialist",
            model_id="gpt-3.5-turbo",  # Using same model but different profile
            tier=ModelTier.MEDIUM,
            capabilities=[
                ModelCapability.CODING,
                ModelCapability.REASONING
            ],
            max_tokens=4096,
            cost_per_1k_tokens=0.003,
            avg_latency_ms=1000,
            quality_score=0.85
        ))
    
    def register_model(self, profile: ModelProfile):
        """Register a model in the orchestrator"""
        self.models[profile.name] = profile
        # Create LLM instance
        try:
            self.llm_instances[profile.name] = ChatOpenAI(
                model=profile.model_id,
                temperature=0.7,
                max_tokens=1000
            )
        except Exception as e:
            print(f"Warning: Could not create LLM instance for {profile.name}: {e}")
            profile.available = False
    
    def execute_task(self, task: Task, use_fallback: bool = True) -> TaskResult:
        """Execute a task using the most appropriate model"""
        # Select best model for task
        selected_model = self._select_model(task)
        
        if not selected_model:
            raise ValueError("No suitable model found for task")
        
        # Execute task
        start_time = time.time()
        
        try:
            result = self._execute_on_model(task, selected_model)
            execution_time = time.time() - start_time
            
            # Calculate cost
            estimated_tokens = len(task.content.split()) * 1.3  # Rough estimate
            estimated_cost = (estimated_tokens / 1000) * selected_model.cost_per_1k_tokens
            
            # Update stats
            self._update_stats(selected_model.name, execution_time, estimated_cost, success=True)
            
            return TaskResult(
                task_id=task.id,
                model_used=selected_model.name,
                result=result,
                execution_time=execution_time,
                estimated_cost=estimated_cost,
                quality_estimate=selected_model.quality_score
            )
            
        except Exception as e:
            self._update_stats(selected_model.name, 0, 0, success=False)
            
            if use_fallback:
                # Try fallback model
                fallback_model = self._select_fallback_model(task, selected_model)
                if fallback_model:
                    print(f"Primary model failed, trying fallback: {fallback_model.name}")
                    task_copy = Task(
                        id=task.id,
                        content=task.content,
                        required_capabilities=task.required_capabilities,
                        priority=task.priority
                    )
                    return self.execute_task(task_copy, use_fallback=False)
            
            raise RuntimeError(f"Task execution failed: {e}")
    
    def execute_batch(self, tasks: List[Task], parallel: bool = False) -> List[TaskResult]:
        """Execute multiple tasks"""
        results = []
        
        for task in tasks:
            try:
                result = self.execute_task(task)
                results.append(result)
            except Exception as e:
                print(f"Task {task.id} failed: {e}")
        
        return results
    
    def execute_with_ensemble(self, task: Task, num_models: int = 3) -> Dict[str, Any]:
        """Execute task on multiple models and aggregate results"""
        # Select top N models
        candidates = self._rank_models_for_task(task)[:num_models]
        
        results = []
        for model_profile in candidates:
            try:
                result = self._execute_on_model(task, model_profile)
                results.append({
                    "model": model_profile.name,
                    "result": result,
                    "quality": model_profile.quality_score
                })
            except Exception as e:
                print(f"Model {model_profile.name} failed: {e}")
        
        if not results:
            raise RuntimeError("All models failed")
        
        # Aggregate results (simple voting or quality-weighted selection)
        best_result = max(results, key=lambda x: x["quality"])
        
        return {
            "aggregated_result": best_result["result"],
            "all_results": results,
            "models_used": [r["model"] for r in results]
        }
    
    def _select_model(self, task: Task) -> Optional[ModelProfile]:
        """Select the best model for a task"""
        candidates = self._rank_models_for_task(task)
        
        if not candidates:
            return None
        
        # Filter by constraints
        for model in candidates:
            # Check cost constraint
            if task.max_cost:
                estimated_tokens = len(task.content.split()) * 1.3
                estimated_cost = (estimated_tokens / 1000) * model.cost_per_1k_tokens
                if estimated_cost > task.max_cost:
                    continue
            
            # Check latency constraint
            if task.max_latency and model.avg_latency_ms > task.max_latency:
                continue
            
            # Check quality constraint
            if model.quality_score < task.min_quality:
                continue
            
            # Check availability
            if not model.available:
                continue
            
            return model
        
        # Return best available if no constraints met
        return candidates[0] if candidates else None
    
    def _rank_models_for_task(self, task: Task) -> List[ModelProfile]:
        """Rank models by suitability for task"""
        scores = []
        
        for model in self.models.values():
            if not model.available:
                continue
            
            score = 0.0
            
            # Capability match
            matching_caps = sum(
                1 for cap in task.required_capabilities 
                if cap in model.capabilities
            )
            if task.required_capabilities:
                capability_score = matching_caps / len(task.required_capabilities)
            else:
                capability_score = 0.5
            
            score += capability_score * 40  # 40% weight
            
            # Quality score
            score += model.quality_score * 30  # 30% weight
            
            # Cost efficiency (lower is better)
            cost_score = 1.0 - min(model.cost_per_1k_tokens / 0.05, 1.0)
            score += cost_score * 15  # 15% weight
            
            # Speed (lower latency is better)
            speed_score = 1.0 - min(model.avg_latency_ms / 5000, 1.0)
            score += speed_score * 15  # 15% weight
            
            # Priority adjustment
            if task.priority == "high":
                score += model.quality_score * 10
            elif task.priority == "low":
                score += cost_score * 10
            
            scores.append((score, model))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        return [model for _, model in scores]
    
    def _select_fallback_model(self, task: Task, failed_model: ModelProfile) -> Optional[ModelProfile]:
        """Select a fallback model after primary fails"""
        candidates = self._rank_models_for_task(task)
        
        # Return first available model that's not the failed one
        for model in candidates:
            if model.name != failed_model.name and model.available:
                return model
        
        return None
    
    def _execute_on_model(self, task: Task, model: ModelProfile) -> str:
        """Execute task on specific model"""
        llm = self.llm_instances.get(model.name)
        
        if not llm:
            raise RuntimeError(f"LLM instance not available for {model.name}")
        
        # Create execution prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are processing this task using {model.name}."),
            ("user", "{task_content}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"task_content": task.content})
        
        return result
    
    def _update_stats(self, model_name: str, execution_time: float, 
                     cost: float, success: bool):
        """Update performance statistics"""
        stats = self.performance_stats[model_name]
        
        if success:
            stats["tasks_completed"] += 1
            stats["total_cost"] += cost
            stats["total_latency"] += execution_time
        else:
            stats["failures"] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for all models"""
        report = {}
        
        for model_name, stats in self.performance_stats.items():
            total_tasks = stats["tasks_completed"] + stats["failures"]
            
            if total_tasks > 0:
                report[model_name] = {
                    "tasks_completed": stats["tasks_completed"],
                    "failures": stats["failures"],
                    "success_rate": stats["tasks_completed"] / total_tasks,
                    "total_cost": stats["total_cost"],
                    "avg_latency": stats["total_latency"] / stats["tasks_completed"] 
                                  if stats["tasks_completed"] > 0 else 0,
                    "cost_per_task": stats["total_cost"] / stats["tasks_completed"]
                                    if stats["tasks_completed"] > 0 else 0
                }
        
        return report


def demonstrate_foundation_model_orchestration():
    """Demonstrate foundation model orchestration"""
    print("=" * 80)
    print("FOUNDATION MODEL ORCHESTRATION PATTERN DEMONSTRATION")
    print("=" * 80)
    
    orchestrator = FoundationModelOrchestrator()
    
    # Example 1: Simple task routing
    print("\n" + "=" * 80)
    print("Example 1: Simple Task Routing")
    print("=" * 80)
    
    tasks = [
        Task(
            id="task1",
            content="Summarize this article in 2 sentences: AI has transformed modern technology...",
            required_capabilities=[ModelCapability.SUMMARIZATION, ModelCapability.FAST_RESPONSE],
            priority="medium"
        ),
        Task(
            id="task2",
            content="Write a Python function to calculate fibonacci numbers recursively",
            required_capabilities=[ModelCapability.CODING],
            priority="high"
        ),
        Task(
            id="task3",
            content="Write a creative short story about a robot learning to paint",
            required_capabilities=[ModelCapability.CREATIVE],
            priority="low"
        )
    ]
    
    print("\nExecuting tasks with automatic model selection:")
    for task in tasks:
        try:
            result = orchestrator.execute_task(task)
            print(f"\nTask: {task.id}")
            print(f"Required capabilities: {[c.value for c in task.required_capabilities]}")
            print(f"Selected model: {result.model_used}")
            print(f"Execution time: {result.execution_time:.3f}s")
            print(f"Estimated cost: ${result.estimated_cost:.6f}")
            print(f"Quality estimate: {result.quality_estimate:.2f}")
            print(f"Result preview: {result.result[:100]}...")
        except Exception as e:
            print(f"\nTask {task.id} failed: {e}")
    
    # Example 2: Cost-constrained execution
    print("\n" + "=" * 80)
    print("Example 2: Cost-Constrained Execution")
    print("=" * 80)
    
    budget_task = Task(
        id="budget_task",
        content="Analyze the sentiment of this review: The product was okay but overpriced.",
        required_capabilities=[ModelCapability.ANALYSIS],
        priority="medium",
        max_cost=0.001  # Very tight budget
    )
    
    try:
        result = orchestrator.execute_task(budget_task)
        print(f"\nTask with max cost ${budget_task.max_cost:.6f}:")
        print(f"Selected model: {result.model_used}")
        print(f"Actual cost: ${result.estimated_cost:.6f}")
        print(f"Result: {result.result[:150]}...")
    except Exception as e:
        print(f"Task failed: {e}")
    
    # Example 3: Quality-focused execution
    print("\n" + "=" * 80)
    print("Example 3: Quality-Focused Execution")
    print("=" * 80)
    
    quality_task = Task(
        id="quality_task",
        content="Solve this logic puzzle: If all A are B, and some B are C, what can we conclude?",
        required_capabilities=[ModelCapability.REASONING],
        priority="high",
        min_quality=0.9  # High quality requirement
    )
    
    try:
        result = orchestrator.execute_task(quality_task)
        print(f"\nTask with min quality {quality_task.min_quality}:")
        print(f"Selected model: {result.model_used}")
        print(f"Quality estimate: {result.quality_estimate}")
        print(f"Result: {result.result[:200]}...")
    except Exception as e:
        print(f"Task failed: {e}")
    
    # Example 4: Ensemble execution
    print("\n" + "=" * 80)
    print("Example 4: Ensemble Execution")
    print("=" * 80)
    
    ensemble_task = Task(
        id="ensemble_task",
        content="What are the three most important factors in climate change?",
        required_capabilities=[ModelCapability.ANALYSIS, ModelCapability.REASONING],
        priority="high"
    )
    
    try:
        ensemble_result = orchestrator.execute_with_ensemble(ensemble_task, num_models=2)
        print(f"\nTask executed with ensemble approach:")
        print(f"Models used: {', '.join(ensemble_result['models_used'])}")
        print(f"\nAggregated result:")
        print(ensemble_result['aggregated_result'][:200] + "...")
        
        print(f"\nAll model results:")
        for i, r in enumerate(ensemble_result['all_results'], 1):
            print(f"\n  Model {i}: {r['model']} (quality: {r['quality']})")
            print(f"  Result: {r['result'][:100]}...")
    except Exception as e:
        print(f"Ensemble execution failed: {e}")
    
    # Example 5: Batch processing
    print("\n" + "=" * 80)
    print("Example 5: Batch Processing")
    print("=" * 80)
    
    batch_tasks = [
        Task(f"batch_{i}", f"Process item {i}: analyze and summarize", 
             [ModelCapability.ANALYSIS], "medium")
        for i in range(3)
    ]
    
    print(f"\nProcessing batch of {len(batch_tasks)} tasks:")
    batch_results = orchestrator.execute_batch(batch_tasks)
    
    print(f"Completed: {len(batch_results)}/{len(batch_tasks)} tasks")
    for result in batch_results:
        print(f"  {result.task_id}: {result.model_used} in {result.execution_time:.3f}s")
    
    # Example 6: Performance reporting
    print("\n" + "=" * 80)
    print("Example 6: Performance Reporting")
    print("=" * 80)
    
    report = orchestrator.get_performance_report()
    
    print("\nPerformance Report:")
    print("-" * 60)
    for model_name, stats in report.items():
        print(f"\n{model_name}:")
        print(f"  Tasks completed: {stats['tasks_completed']}")
        print(f"  Failures: {stats['failures']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Total cost: ${stats['total_cost']:.6f}")
        print(f"  Avg latency: {stats['avg_latency']:.3f}s")
        print(f"  Cost per task: ${stats['cost_per_task']:.6f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Foundation Model Orchestration pattern enables:
✓ Intelligent routing of tasks to optimal models
✓ Cost-performance trade-off optimization
✓ Quality-focused model selection
✓ Ensemble execution for critical tasks
✓ Fallback handling for reliability
✓ Performance monitoring and optimization
✓ Batch processing with automatic routing

This pattern is valuable for:
- Production AI systems with cost constraints
- Multi-model architectures
- Quality-performance optimization
- Resource-efficient AI pipelines
- Fault-tolerant AI systems
- Adaptive model selection
    """)


if __name__ == "__main__":
    demonstrate_foundation_model_orchestration()
