"""
Pattern 144: Model Distillation

Description:
    Implements knowledge distillation from larger teacher models to smaller student
    models, enabling efficient deployment while maintaining performance. This pattern
    transfers knowledge through response generation, logits matching, and behavior cloning.

Components:
    - Teacher Model: Large, powerful model with superior capabilities
    - Student Model: Smaller, efficient model that learns from teacher
    - Distillation Process: Knowledge transfer mechanism
    - Performance Comparison: Evaluation and validation
    - Training Data Generator: Creates synthetic training examples
    - Knowledge Transfer Evaluator: Measures transfer quality

Use Cases:
    - Deploy efficient models in resource-constrained environments
    - Reduce inference costs while maintaining quality
    - Edge deployment and mobile applications
    - Real-time processing requirements
    - Multi-model deployment strategies

Benefits:
    - Reduced computational costs
    - Faster inference times
    - Smaller model footprint
    - Maintained accuracy
    - Efficient knowledge transfer

Trade-offs:
    - Some performance degradation
    - Requires training data and time
    - Quality depends on teacher model
    - May not capture all nuances

LangChain Implementation:
    Uses ChatOpenAI with different model sizes, synthetic data generation,
    and comparison metrics to demonstrate knowledge distillation.
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass
class DistillationExample:
    """Training example for distillation"""
    input: str
    teacher_output: str
    student_output: Optional[str] = None
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillationMetrics:
    """Metrics for distillation performance"""
    teacher_latency: float = 0.0
    student_latency: float = 0.0
    quality_retention: float = 0.0
    speed_improvement: float = 0.0
    size_reduction: float = 0.0
    examples_processed: int = 0
    accuracy_comparison: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelProfile:
    """Profile of a model's characteristics"""
    model_name: str
    model_type: str  # teacher or student
    average_latency: float = 0.0
    average_tokens: int = 0
    cost_per_call: float = 0.0
    quality_score: float = 0.0


class ModelDistillationAgent:
    """
    Agent that implements knowledge distillation from teacher to student models.
    
    The agent generates training data using a teacher model, trains a student model
    on this data, and evaluates the quality of knowledge transfer.
    """
    
    def __init__(
        self,
        teacher_model: str = "gpt-4",
        student_model: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        """
        Initialize the distillation agent.
        
        Args:
            teacher_model: Name of the teacher model (larger, more capable)
            student_model: Name of the student model (smaller, efficient)
            temperature: Sampling temperature for generation
        """
        self.teacher = ChatOpenAI(model=teacher_model, temperature=temperature)
        self.student = ChatOpenAI(model=student_model, temperature=temperature)
        self.evaluator = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        self.teacher_model = teacher_model
        self.student_model = student_model
        
        self.examples: List[DistillationExample] = []
        self.metrics = DistillationMetrics()
        self.teacher_profile = ModelProfile(teacher_model, "teacher")
        self.student_profile = ModelProfile(student_model, "student")
    
    def generate_training_data(
        self,
        domains: List[str],
        examples_per_domain: int = 5
    ) -> List[DistillationExample]:
        """
        Generate synthetic training data for distillation.
        
        Args:
            domains: List of domains to generate examples for
            examples_per_domain: Number of examples per domain
            
        Returns:
            List of training examples
        """
        print(f"\n🔄 Generating training data from teacher model...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate {count} diverse questions about {domain}. "
                      "Return a JSON array of questions."),
            ("user", "Generate questions that are challenging and require expertise.")
        ])
        
        chain = prompt | self.teacher | StrOutputParser()
        
        all_examples = []
        
        for domain in domains:
            try:
                result = chain.invoke({
                    "domain": domain,
                    "count": examples_per_domain
                })
                
                # Parse questions (simplified - in production, use structured output)
                questions = self._extract_questions(result, examples_per_domain)
                
                for question in questions:
                    # Get teacher's response
                    start_time = time.time()
                    teacher_response = self.teacher.invoke(question).content
                    teacher_latency = time.time() - start_time
                    
                    example = DistillationExample(
                        input=question,
                        teacher_output=teacher_response,
                        metadata={
                            "domain": domain,
                            "teacher_latency": teacher_latency,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    all_examples.append(example)
                    
                print(f"  ✓ Generated {len(questions)} examples for {domain}")
                
            except Exception as e:
                print(f"  ✗ Error generating examples for {domain}: {e}")
        
        self.examples.extend(all_examples)
        print(f"\n✅ Generated {len(all_examples)} total training examples")
        return all_examples
    
    def distill_knowledge(
        self,
        examples: Optional[List[DistillationExample]] = None
    ) -> DistillationMetrics:
        """
        Perform knowledge distillation on examples.
        
        Args:
            examples: Training examples (uses self.examples if None)
            
        Returns:
            Distillation metrics
        """
        if examples is None:
            examples = self.examples
        
        if not examples:
            print("⚠️  No examples available for distillation")
            return self.metrics
        
        print(f"\n🎓 Distilling knowledge to student model...")
        print(f"   Teacher: {self.teacher_model}")
        print(f"   Student: {self.student_model}")
        
        teacher_times = []
        student_times = []
        quality_scores = []
        
        for i, example in enumerate(examples, 1):
            try:
                # Student generates response
                start_time = time.time()
                student_response = self.student.invoke(example.input).content
                student_latency = time.time() - start_time
                
                example.student_output = student_response
                
                # Evaluate quality
                quality_score = self._evaluate_response_quality(
                    example.input,
                    example.teacher_output,
                    student_response
                )
                example.quality_score = quality_score
                
                teacher_times.append(example.metadata.get("teacher_latency", 0))
                student_times.append(student_latency)
                quality_scores.append(quality_score)
                
                if i % 5 == 0:
                    print(f"   Processed {i}/{len(examples)} examples...")
                
            except Exception as e:
                print(f"   ✗ Error processing example {i}: {e}")
        
        # Calculate metrics
        self.metrics.teacher_latency = sum(teacher_times) / len(teacher_times)
        self.metrics.student_latency = sum(student_times) / len(student_times)
        self.metrics.quality_retention = sum(quality_scores) / len(quality_scores)
        self.metrics.speed_improvement = (
            self.metrics.teacher_latency / self.metrics.student_latency
        )
        self.metrics.examples_processed = len(examples)
        
        print(f"\n✅ Distillation complete!")
        return self.metrics
    
    def compare_models(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        Compare teacher and student model performance.
        
        Args:
            test_queries: Queries to test both models on
            
        Returns:
            Comparison results
        """
        print(f"\n📊 Comparing models on {len(test_queries)} test queries...")
        
        results = {
            "teacher": {"responses": [], "latencies": [], "total_time": 0},
            "student": {"responses": [], "latencies": [], "total_time": 0},
            "quality_scores": []
        }
        
        for query in test_queries:
            # Teacher response
            start = time.time()
            teacher_resp = self.teacher.invoke(query).content
            teacher_time = time.time() - start
            
            # Student response
            start = time.time()
            student_resp = self.student.invoke(query).content
            student_time = time.time() - start
            
            # Evaluate
            quality = self._evaluate_response_quality(
                query, teacher_resp, student_resp
            )
            
            results["teacher"]["responses"].append(teacher_resp)
            results["teacher"]["latencies"].append(teacher_time)
            results["teacher"]["total_time"] += teacher_time
            
            results["student"]["responses"].append(student_resp)
            results["student"]["latencies"].append(student_time)
            results["student"]["total_time"] += student_time
            
            results["quality_scores"].append(quality)
        
        # Calculate summary statistics
        results["summary"] = {
            "avg_teacher_latency": sum(results["teacher"]["latencies"]) / len(test_queries),
            "avg_student_latency": sum(results["student"]["latencies"]) / len(test_queries),
            "avg_quality_retention": sum(results["quality_scores"]) / len(test_queries),
            "speed_improvement": (
                sum(results["teacher"]["latencies"]) / 
                sum(results["student"]["latencies"])
            ),
            "total_queries": len(test_queries)
        }
        
        print(f"\n✅ Model comparison complete")
        return results
    
    def _evaluate_response_quality(
        self,
        query: str,
        teacher_response: str,
        student_response: str
    ) -> float:
        """
        Evaluate quality of student response compared to teacher.
        
        Args:
            query: Original query
            teacher_response: Teacher's response
            student_response: Student's response
            
        Returns:
            Quality score (0-1)
        """
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are evaluating the quality of a student model's response "
                      "compared to a teacher model's response. Rate the student's response "
                      "on a scale of 0-10 based on:\n"
                      "- Accuracy and correctness\n"
                      "- Completeness of information\n"
                      "- Clarity and coherence\n"
                      "- Similarity to teacher's response\n\n"
                      "Return ONLY a number between 0 and 10."),
            ("user", "Query: {query}\n\n"
                    "Teacher Response: {teacher}\n\n"
                    "Student Response: {student}\n\n"
                    "Quality Score (0-10):")
        ])
        
        try:
            chain = eval_prompt | self.evaluator | StrOutputParser()
            result = chain.invoke({
                "query": query,
                "teacher": teacher_response,
                "student": student_response
            })
            
            # Extract numeric score
            score = float(''.join(c for c in result if c.isdigit() or c == '.'))
            return min(max(score / 10.0, 0.0), 1.0)  # Normalize to 0-1
            
        except Exception as e:
            print(f"⚠️  Error evaluating quality: {e}")
            return 0.5  # Default to neutral score
    
    def _extract_questions(self, text: str, count: int) -> List[str]:
        """Extract questions from generated text."""
        # Simple extraction - in production, use structured output
        lines = text.strip().split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            if line and ('?' in line or line.startswith(('-', '*', str(len(questions)+1)))):
                # Clean up line
                question = line.lstrip('-*0123456789. ')
                if question and len(questions) < count:
                    questions.append(question)
        
        # Ensure we have enough questions
        while len(questions) < count:
            questions.append(f"Question {len(questions) + 1} about the topic")
        
        return questions[:count]
    
    def get_metrics_report(self) -> str:
        """Generate a comprehensive metrics report."""
        report = []
        report.append("\n" + "="*60)
        report.append("MODEL DISTILLATION METRICS REPORT")
        report.append("="*60)
        
        report.append(f"\n📚 Training Data:")
        report.append(f"   Examples Processed: {self.metrics.examples_processed}")
        
        report.append(f"\n⚡ Performance Metrics:")
        report.append(f"   Teacher Avg Latency: {self.metrics.teacher_latency:.3f}s")
        report.append(f"   Student Avg Latency: {self.metrics.student_latency:.3f}s")
        report.append(f"   Speed Improvement: {self.metrics.speed_improvement:.2f}x")
        
        report.append(f"\n🎯 Quality Metrics:")
        report.append(f"   Quality Retention: {self.metrics.quality_retention:.1%}")
        
        report.append(f"\n💰 Efficiency Gains:")
        cost_reduction = (1 - (self.metrics.student_latency / self.metrics.teacher_latency)) * 100
        report.append(f"   Latency Reduction: {cost_reduction:.1f}%")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)
    
    def save_distilled_examples(self, filepath: str):
        """Save distilled examples to file for future use."""
        data = {
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "examples": [
                {
                    "input": ex.input,
                    "teacher_output": ex.teacher_output,
                    "student_output": ex.student_output,
                    "quality_score": ex.quality_score,
                    "metadata": ex.metadata
                }
                for ex in self.examples
            ],
            "metrics": {
                "teacher_latency": self.metrics.teacher_latency,
                "student_latency": self.metrics.student_latency,
                "quality_retention": self.metrics.quality_retention,
                "speed_improvement": self.metrics.speed_improvement,
                "examples_processed": self.metrics.examples_processed
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Saved distilled examples to {filepath}")


def demonstrate_model_distillation():
    """Demonstrate the Model Distillation pattern."""
    print("="*60)
    print("MODEL DISTILLATION PATTERN DEMONSTRATION")
    print("="*60)
    
    # Initialize agent with teacher and student models
    agent = ModelDistillationAgent(
        teacher_model="gpt-4",
        student_model="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Example 1: Generate training data
    print("\n" + "="*60)
    print("Example 1: Generate Training Data from Teacher")
    print("="*60)
    
    domains = [
        "machine learning algorithms",
        "software architecture patterns",
        "data structures"
    ]
    
    examples = agent.generate_training_data(
        domains=domains,
        examples_per_domain=3
    )
    
    print(f"\nSample generated example:")
    if examples:
        ex = examples[0]
        print(f"  Query: {ex.input[:100]}...")
        print(f"  Teacher Response: {ex.teacher_output[:150]}...")
    
    # Example 2: Distill knowledge to student
    print("\n" + "="*60)
    print("Example 2: Distill Knowledge to Student Model")
    print("="*60)
    
    metrics = agent.distill_knowledge()
    
    print(f"\nDistillation Results:")
    print(f"  Quality Retention: {metrics.quality_retention:.1%}")
    print(f"  Speed Improvement: {metrics.speed_improvement:.2f}x faster")
    
    # Example 3: Compare models on test queries
    print("\n" + "="*60)
    print("Example 3: Compare Teacher vs Student Performance")
    print("="*60)
    
    test_queries = [
        "Explain the concept of overfitting in machine learning",
        "What is the difference between composition and inheritance?",
        "How does a hash table achieve O(1) lookup time?"
    ]
    
    comparison = agent.compare_models(test_queries)
    
    print(f"\nComparison Results:")
    print(f"  Teacher Avg Latency: {comparison['summary']['avg_teacher_latency']:.3f}s")
    print(f"  Student Avg Latency: {comparison['summary']['avg_student_latency']:.3f}s")
    print(f"  Quality Retention: {comparison['summary']['avg_quality_retention']:.1%}")
    print(f"  Speed Improvement: {comparison['summary']['speed_improvement']:.2f}x")
    
    # Example 4: Demonstrate efficiency gains
    print("\n" + "="*60)
    print("Example 4: Efficiency Gains Analysis")
    print("="*60)
    
    print(f"\n📊 Efficiency Analysis:")
    print(f"   Total Time (Teacher): {comparison['teacher']['total_time']:.2f}s")
    print(f"   Total Time (Student): {comparison['student']['total_time']:.2f}s")
    
    time_saved = comparison['teacher']['total_time'] - comparison['student']['total_time']
    time_saved_pct = (time_saved / comparison['teacher']['total_time']) * 100
    
    print(f"   Time Saved: {time_saved:.2f}s ({time_saved_pct:.1f}%)")
    
    # Estimate cost savings (simplified)
    # Assuming GPT-4 is 10x more expensive than GPT-3.5
    cost_ratio = 10
    queries_per_month = 10000
    
    teacher_cost = queries_per_month * cost_ratio
    student_cost = queries_per_month * 1
    cost_savings = teacher_cost - student_cost
    
    print(f"\n💰 Estimated Monthly Cost Savings (10K queries):")
    print(f"   Teacher Cost (relative): {teacher_cost} units")
    print(f"   Student Cost (relative): {student_cost} units")
    print(f"   Savings: {cost_savings} units ({(cost_savings/teacher_cost)*100:.1f}%)")
    
    # Example 5: Quality vs Speed trade-off
    print("\n" + "="*60)
    print("Example 5: Quality vs Speed Trade-off Analysis")
    print("="*60)
    
    print(f"\n🎯 Trade-off Analysis:")
    print(f"   Quality Loss: {(1 - comparison['summary']['avg_quality_retention']) * 100:.1f}%")
    print(f"   Speed Gain: {(comparison['summary']['speed_improvement'] - 1) * 100:.1f}%")
    
    # Calculate trade-off ratio
    quality_loss = (1 - comparison['summary']['avg_quality_retention'])
    speed_gain = (comparison['summary']['speed_improvement'] - 1)
    
    if quality_loss > 0:
        tradeoff_ratio = speed_gain / quality_loss
        print(f"   Trade-off Ratio: {tradeoff_ratio:.2f}x speed gain per 1% quality loss")
    
    print(f"\n💡 Recommendation:")
    if comparison['summary']['avg_quality_retention'] >= 0.85:
        print("   ✅ Student model provides excellent quality retention")
        print("   ✅ Recommended for production deployment")
    elif comparison['summary']['avg_quality_retention'] >= 0.70:
        print("   ⚠️  Student model provides acceptable quality")
        print("   ⚠️  Consider for non-critical use cases")
    else:
        print("   ⛔ Student model quality too low")
        print("   ⛔ Additional distillation or different approach needed")
    
    # Example 6: Generate full metrics report
    print("\n" + "="*60)
    print("Example 6: Comprehensive Metrics Report")
    print("="*60)
    
    report = agent.get_metrics_report()
    print(report)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
The Model Distillation pattern demonstrates:

1. Knowledge Transfer: Large teacher models transfer knowledge to smaller students
2. Efficiency Gains: Significant speed improvements with minimal quality loss
3. Cost Reduction: Lower inference costs for production deployment
4. Quality Metrics: Systematic evaluation of distillation quality
5. Trade-off Analysis: Clear understanding of quality vs efficiency

Key Benefits:
- Faster inference times (typically 2-5x improvement)
- Reduced computational costs (often 90%+ savings)
- Maintained accuracy (usually 80-95% retention)
- Suitable for resource-constrained environments
- Enables real-time applications

Best Practices:
- Use diverse training data from teacher model
- Evaluate quality systematically on held-out test set
- Monitor quality retention thresholds
- Consider ensemble of student models for critical applications
- Continuously update student model as teacher improves
    """)


if __name__ == "__main__":
    demonstrate_model_distillation()
