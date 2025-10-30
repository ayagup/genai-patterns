"""
Pattern 144: Model Distillation Agent

This pattern implements model distillation where a smaller, faster "student" model
learns from a larger, more capable "teacher" model. Knowledge is transferred through
soft targets, intermediate representations, and behavior mimicry.

Category: Performance Optimization
Use Cases:
- Deploy faster models to production
- Reduce inference costs
- Enable edge deployment
- Maintain quality with smaller models
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime
import random
import math


class DistillationType(Enum):
    """Types of knowledge distillation"""
    RESPONSE = "response"  # Soft targets from final outputs
    FEATURE = "feature"    # Intermediate layer matching
    ATTENTION = "attention"  # Attention transfer
    RELATION = "relation"  # Relational knowledge
    HYBRID = "hybrid"      # Multiple approaches


class ModelSize(Enum):
    """Relative model sizes"""
    TINY = "tiny"       # 10x smaller
    SMALL = "small"     # 5x smaller
    MEDIUM = "medium"   # 3x smaller
    LARGE = "large"     # 2x smaller
    XLARGE = "xlarge"   # Similar size


class TrainingPhase(Enum):
    """Phases of distillation training"""
    INITIALIZATION = "initialization"
    WARM_UP = "warm_up"
    DISTILLATION = "distillation"
    FINE_TUNING = "fine_tuning"
    VALIDATION = "validation"
    COMPLETE = "complete"


@dataclass
class ModelSpec:
    """Specification for a model"""
    name: str
    size: ModelSize
    parameters: int
    layers: int
    hidden_dim: int
    attention_heads: int
    inference_time_ms: float
    memory_mb: float
    capabilities: List[str] = field(default_factory=list)


@dataclass
class DistillationConfig:
    """Configuration for distillation process"""
    distillation_type: DistillationType
    temperature: float = 3.0  # Softmax temperature
    alpha: float = 0.5  # Balance between hard and soft targets
    learning_rate: float = 1e-4
    epochs: int = 50
    batch_size: int = 32
    feature_matching: bool = True
    attention_transfer: bool = False
    use_hints: bool = True


@dataclass
class TrainingExample:
    """Example for distillation training"""
    input_text: str
    task_type: str
    ground_truth: Optional[str] = None
    teacher_output: Optional[Dict[str, Any]] = None
    teacher_logits: Optional[List[float]] = None
    teacher_features: Optional[Dict[str, List[float]]] = None
    teacher_attention: Optional[List[List[float]]] = None


@dataclass
class DistillationMetrics:
    """Metrics for distillation process"""
    epoch: int
    phase: TrainingPhase
    distillation_loss: float
    task_loss: float
    total_loss: float
    teacher_agreement: float
    student_accuracy: float
    compression_ratio: float
    speedup_factor: float
    quality_retention: float
    timestamp: datetime = field(default_factory=datetime.now)


class TeacherModel:
    """Large, capable teacher model"""
    
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.inference_count = 0
        
    def predict(self, input_text: str, task_type: str) -> Dict[str, Any]:
        """Generate predictions with full model"""
        self.inference_count += 1
        
        # Simulate teacher model inference (would be actual model in practice)
        logits = self._compute_logits(input_text, task_type)
        features = self._extract_features(input_text)
        attention = self._compute_attention(input_text)
        
        # Generate output
        output = self._generate_output(logits, task_type)
        
        return {
            'output': output,
            'logits': logits,
            'features': features,
            'attention': attention,
            'confidence': self._compute_confidence(logits)
        }
    
    def _compute_logits(self, input_text: str, task_type: str) -> List[float]:
        """Compute output logits (before softmax)"""
        # Simulate logits based on input complexity
        vocab_size = 50000
        complexity = len(input_text.split()) / 10.0
        
        # Create distribution with high confidence on correct answer
        logits = [random.gauss(-5, 1) for _ in range(min(100, vocab_size))]
        # Make top answer much higher
        top_idx = random.randint(0, len(logits) - 1)
        logits[top_idx] += 10 * complexity
        
        return logits
    
    def _extract_features(self, input_text: str) -> Dict[str, List[float]]:
        """Extract intermediate layer features"""
        features = {}
        # Sample a few layers for efficiency
        sample_layers = min(3, self.spec.layers)
        # Use smaller feature dimension for simulation
        feature_dim = min(64, self.spec.hidden_dim)
        
        for layer in range(sample_layers):
            # Simulate layer features
            features[f'layer_{layer}'] = [
                random.gauss(0, 1) for _ in range(feature_dim)
            ]
        return features
    
    def _compute_attention(self, input_text: str) -> List[List[float]]:
        """Compute attention weights"""
        seq_len = min(10, len(input_text.split()))  # Limit sequence length
        attention = []
        
        # Sample a few attention heads for efficiency
        sample_heads = min(4, self.spec.attention_heads)
        
        for head in range(sample_heads):
            # Simulate attention pattern
            weights = [random.random() for _ in range(seq_len)]
            # Normalize
            total = sum(weights)
            weights = [w / total for w in weights]
            attention.append(weights)
        
        return attention
    
    def _generate_output(self, logits: List[float], task_type: str) -> str:
        """Generate output from logits"""
        # Simulate output generation
        if task_type == "classification":
            return f"class_{logits.index(max(logits))}"
        elif task_type == "qa":
            return "The answer is based on the context provided."
        elif task_type == "summarization":
            return "Summary of the key points."
        else:
            return "Generated response based on input."
    
    def _compute_confidence(self, logits: List[float]) -> float:
        """Compute prediction confidence"""
        # Apply softmax and return max probability
        exp_logits = [math.exp(l) for l in logits]
        total = sum(exp_logits)
        probs = [e / total for e in exp_logits]
        return max(probs)


class StudentModel:
    """Smaller, faster student model"""
    
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.trained = False
        self.training_steps = 0
        self.performance = 0.0
        
    def predict(self, input_text: str, task_type: str) -> Dict[str, Any]:
        """Generate predictions (initially poor, improves with training)"""
        # Performance improves with training
        quality_factor = self.performance if self.trained else 0.3
        
        # Simulate faster but initially less accurate inference
        logits = self._compute_logits(input_text, task_type, quality_factor)
        output = self._generate_output(logits, task_type)
        
        return {
            'output': output,
            'logits': logits,
            'confidence': self._compute_confidence(logits)
        }
    
    def _compute_logits(self, input_text: str, task_type: str, quality: float) -> List[float]:
        """Compute logits with quality factor"""
        # Smaller vocabulary/output space
        logits = [random.gauss(-3, 2) for _ in range(50)]
        
        # Quality affects how good the top prediction is
        top_idx = random.randint(0, len(logits) - 1)
        logits[top_idx] += 5 * quality
        
        return logits
    
    def _generate_output(self, logits: List[float], task_type: str) -> str:
        """Generate output from logits"""
        return f"student_output_{logits.index(max(logits))}"
    
    def _compute_confidence(self, logits: List[float]) -> float:
        """Compute prediction confidence"""
        exp_logits = [math.exp(l) for l in logits]
        total = sum(exp_logits)
        probs = [e / total for e in exp_logits]
        return max(probs)
    
    def update_from_distillation(self, loss: float, agreement: float):
        """Update student based on distillation"""
        self.training_steps += 1
        # Performance improves with training
        improvement = (1 - self.performance) * 0.05 * agreement
        self.performance = min(0.95, self.performance + improvement)
        self.trained = True


class DistillationLoss:
    """Computes various distillation losses"""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        
    def compute_distillation_loss(
        self,
        teacher_logits: List[float],
        student_logits: List[float]
    ) -> float:
        """Compute KL divergence between teacher and student distributions"""
        # Apply temperature scaling
        temp = self.config.temperature
        
        # Compute softmax with temperature
        teacher_probs = self._softmax_with_temp(teacher_logits, temp)
        student_probs = self._softmax_with_temp(student_logits, temp)
        
        # KL divergence
        kl_div = sum(
            t * math.log(t / (s + 1e-10) + 1e-10)
            for t, s in zip(teacher_probs, student_probs)
            if t > 0
        )
        
        return kl_div * (temp ** 2)  # Scale by temperature squared
    
    def compute_task_loss(
        self,
        student_output: str,
        ground_truth: str
    ) -> float:
        """Compute loss on actual task (hard targets)"""
        # Simulate task loss (would be cross-entropy in practice)
        if student_output == ground_truth:
            return 0.1
        elif ground_truth in student_output:
            return 0.5
        else:
            return 1.0
    
    def compute_feature_loss(
        self,
        teacher_features: Dict[str, List[float]],
        student_features: Dict[str, List[float]]
    ) -> float:
        """Compute loss on intermediate features"""
        if not self.config.feature_matching:
            return 0.0
        
        total_loss = 0.0
        count = 0
        
        # Match corresponding layers
        for layer_name in teacher_features:
            if layer_name in student_features:
                t_feat = teacher_features[layer_name]
                s_feat = student_features[layer_name]
                
                # MSE loss
                loss = sum((t - s) ** 2 for t, s in zip(t_feat, s_feat))
                total_loss += loss / len(t_feat)
                count += 1
        
        return total_loss / count if count > 0 else 0.0
    
    def compute_attention_loss(
        self,
        teacher_attention: List[List[float]],
        student_attention: List[List[float]]
    ) -> float:
        """Compute loss on attention patterns"""
        if not self.config.attention_transfer:
            return 0.0
        
        total_loss = 0.0
        
        # Match attention heads
        for t_attn, s_attn in zip(teacher_attention, student_attention):
            # MSE on attention weights
            loss = sum((t - s) ** 2 for t, s in zip(t_attn, s_attn))
            total_loss += loss / len(t_attn)
        
        return total_loss / len(teacher_attention)
    
    def compute_total_loss(
        self,
        teacher_example: TrainingExample,
        student_output: Dict[str, Any],
        ground_truth: Optional[str] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Compute combined loss"""
        losses = {}
        
        # Distillation loss (soft targets)
        if teacher_example.teacher_logits:
            losses['distillation'] = self.compute_distillation_loss(
                teacher_example.teacher_logits,
                student_output['logits']
            )
        else:
            losses['distillation'] = 0.0
        
        # Task loss (hard targets)
        if ground_truth:
            losses['task'] = self.compute_task_loss(
                student_output['output'],
                ground_truth
            )
        else:
            losses['task'] = 0.0
        
        # Feature matching loss
        if teacher_example.teacher_features:
            losses['feature'] = self.compute_feature_loss(
                teacher_example.teacher_features,
                {}  # Would extract from student
            )
        else:
            losses['feature'] = 0.0
        
        # Attention transfer loss
        if teacher_example.teacher_attention:
            losses['attention'] = self.compute_attention_loss(
                teacher_example.teacher_attention,
                []  # Would extract from student
            )
        else:
            losses['attention'] = 0.0
        
        # Weighted combination
        alpha = self.config.alpha
        total = (
            alpha * losses['distillation'] +
            (1 - alpha) * losses['task'] +
            0.1 * losses['feature'] +
            0.05 * losses['attention']
        )
        
        losses['total'] = total
        return total, losses
    
    def _softmax_with_temp(self, logits: List[float], temperature: float) -> List[float]:
        """Apply softmax with temperature scaling"""
        scaled = [l / temperature for l in logits]
        exp_scaled = [math.exp(s) for s in scaled]
        total = sum(exp_scaled)
        return [e / total for e in exp_scaled]


class PerformanceEvaluator:
    """Evaluates distilled model performance"""
    
    def __init__(self, teacher: TeacherModel, student: StudentModel):
        self.teacher = teacher
        self.student = student
        
    def evaluate(self, test_examples: List[TrainingExample]) -> Dict[str, float]:
        """Evaluate student against teacher"""
        agreements = []
        student_correct = 0
        teacher_correct = 0
        
        speedups = []
        
        for example in test_examples:
            # Teacher prediction
            teacher_pred = self.teacher.predict(
                example.input_text,
                example.task_type
            )
            
            # Student prediction
            student_pred = self.student.predict(
                example.input_text,
                example.task_type
            )
            
            # Compare outputs
            agreement = self._compute_agreement(
                teacher_pred['output'],
                student_pred['output']
            )
            agreements.append(agreement)
            
            # Check correctness if ground truth available
            if example.ground_truth:
                if teacher_pred['output'] == example.ground_truth:
                    teacher_correct += 1
                if student_pred['output'] == example.ground_truth:
                    student_correct += 1
            
            # Speedup
            speedup = self.teacher.spec.inference_time_ms / self.student.spec.inference_time_ms
            speedups.append(speedup)
        
        # Compute metrics
        metrics = {
            'teacher_agreement': sum(agreements) / len(agreements),
            'student_accuracy': student_correct / len(test_examples) if any(e.ground_truth for e in test_examples) else 0.0,
            'teacher_accuracy': teacher_correct / len(test_examples) if any(e.ground_truth for e in test_examples) else 0.0,
            'compression_ratio': self.teacher.spec.parameters / self.student.spec.parameters,
            'speedup_factor': sum(speedups) / len(speedups),
            'memory_reduction': (1 - self.student.spec.memory_mb / self.teacher.spec.memory_mb) * 100
        }
        
        # Quality retention
        if metrics['teacher_accuracy'] > 0:
            metrics['quality_retention'] = metrics['student_accuracy'] / metrics['teacher_accuracy']
        else:
            metrics['quality_retention'] = metrics['teacher_agreement']
        
        return metrics
    
    def _compute_agreement(self, teacher_output: str, student_output: str) -> float:
        """Compute agreement between outputs"""
        if teacher_output == student_output:
            return 1.0
        
        # Partial credit for similar outputs
        teacher_tokens = set(teacher_output.lower().split())
        student_tokens = set(student_output.lower().split())
        
        if not teacher_tokens or not student_tokens:
            return 0.0
        
        intersection = teacher_tokens.intersection(student_tokens)
        union = teacher_tokens.union(student_tokens)
        
        return len(intersection) / len(union) if union else 0.0


class DistillationTrainer:
    """Manages distillation training process"""
    
    def __init__(
        self,
        teacher: TeacherModel,
        student: StudentModel,
        config: DistillationConfig
    ):
        self.teacher = teacher
        self.student = student
        self.config = config
        self.loss_calculator = DistillationLoss(config)
        self.evaluator = PerformanceEvaluator(teacher, student)
        self.training_history: List[DistillationMetrics] = []
        self.current_phase = TrainingPhase.INITIALIZATION
        
    def train(self, training_data: List[TrainingExample]) -> List[DistillationMetrics]:
        """Run full distillation training"""
        print("\n" + "="*60)
        print("STARTING DISTILLATION TRAINING")
        print("="*60)
        
        # Phase 1: Initialization
        self.current_phase = TrainingPhase.INITIALIZATION
        print(f"\nPhase: {self.current_phase.value}")
        self._initialize_student()
        
        # Phase 2: Warm-up (train on hard labels only)
        self.current_phase = TrainingPhase.WARM_UP
        print(f"\nPhase: {self.current_phase.value}")
        for epoch in range(2):  # Reduced for demo
            metrics = self._train_epoch(training_data, warm_up=True, epoch=epoch)
            self.training_history.append(metrics)
            print(f"  Epoch {epoch}: Loss={metrics.total_loss:.4f}, Accuracy={metrics.student_accuracy:.4f}")
        
        # Phase 3: Distillation (soft labels from teacher)
        self.current_phase = TrainingPhase.DISTILLATION
        print(f"\nPhase: {self.current_phase.value}")
        for epoch in range(self.config.epochs):
            metrics = self._train_epoch(training_data, warm_up=False, epoch=epoch)
            self.training_history.append(metrics)
            
            if epoch % 3 == 0:  # Print every 3 epochs
                print(f"  Epoch {epoch}: "
                      f"Loss={metrics.total_loss:.4f}, "
                      f"Agreement={metrics.teacher_agreement:.4f}, "
                      f"Quality={metrics.quality_retention:.4f}")
        
        # Phase 4: Fine-tuning
        self.current_phase = TrainingPhase.FINE_TUNING
        print(f"\nPhase: {self.current_phase.value}")
        for epoch in range(3):  # Reduced for demo
            metrics = self._train_epoch(training_data, warm_up=True, epoch=epoch)
            self.training_history.append(metrics)
        
        # Phase 5: Final validation
        self.current_phase = TrainingPhase.VALIDATION
        print(f"\nPhase: {self.current_phase.value}")
        final_metrics = self._validate(training_data)
        self.training_history.append(final_metrics)
        
        self.current_phase = TrainingPhase.COMPLETE
        print(f"\nTraining complete!")
        print(f"Final agreement with teacher: {final_metrics.teacher_agreement:.2%}")
        print(f"Quality retention: {final_metrics.quality_retention:.2%}")
        print(f"Speedup: {final_metrics.speedup_factor:.1f}x")
        print(f"Compression: {final_metrics.compression_ratio:.1f}x")
        
        return self.training_history
    
    def _initialize_student(self):
        """Initialize student model"""
        self.student.performance = 0.3  # Start with low performance
        print(f"  Initialized student model: {self.student.spec.name}")
        print(f"  Parameters: {self.student.spec.parameters:,}")
        print(f"  Compression ratio: {self.teacher.spec.parameters / self.student.spec.parameters:.1f}x")
    
    def _train_epoch(
        self,
        training_data: List[TrainingExample],
        warm_up: bool,
        epoch: int
    ) -> DistillationMetrics:
        """Train for one epoch"""
        total_loss = 0.0
        distillation_loss = 0.0
        task_loss = 0.0
        
        # Process batches
        for i in range(0, len(training_data), self.config.batch_size):
            batch = training_data[i:i + self.config.batch_size]
            
            for example in batch:
                # Get student prediction
                student_output = self.student.predict(
                    example.input_text,
                    example.task_type
                )
                
                # Compute loss
                if warm_up:
                    # Only task loss during warm-up
                    ground_truth_value = example.ground_truth
                    if not ground_truth_value and example.teacher_output:
                        ground_truth_value = example.teacher_output['output']
                    
                    loss = self.loss_calculator.compute_task_loss(
                        student_output['output'],
                        ground_truth_value or ""
                    )
                    task_loss += loss
                    total_loss += loss
                else:
                    # Full distillation loss
                    loss, components = self.loss_calculator.compute_total_loss(
                        example,
                        student_output,
                        example.ground_truth
                    )
                    distillation_loss += components['distillation']
                    task_loss += components['task']
                    total_loss += loss
                
                # Update student (simulated gradient descent)
                agreement = 0.7 + (epoch / self.config.epochs) * 0.25  # Improve over time
                self.student.update_from_distillation(loss, agreement)
        
        # Evaluate on training data
        eval_metrics = self.evaluator.evaluate(training_data[:20])  # Small sample for speed
        
        # Create metrics
        metrics = DistillationMetrics(
            epoch=epoch,
            phase=self.current_phase,
            distillation_loss=distillation_loss / len(training_data),
            task_loss=task_loss / len(training_data),
            total_loss=total_loss / len(training_data),
            teacher_agreement=eval_metrics['teacher_agreement'],
            student_accuracy=eval_metrics['student_accuracy'],
            compression_ratio=eval_metrics['compression_ratio'],
            speedup_factor=eval_metrics['speedup_factor'],
            quality_retention=eval_metrics['quality_retention']
        )
        
        return metrics
    
    def _validate(self, test_data: List[TrainingExample]) -> DistillationMetrics:
        """Final validation"""
        eval_metrics = self.evaluator.evaluate(test_data)
        
        return DistillationMetrics(
            epoch=-1,
            phase=TrainingPhase.VALIDATION,
            distillation_loss=0.0,
            task_loss=0.0,
            total_loss=0.0,
            teacher_agreement=eval_metrics['teacher_agreement'],
            student_accuracy=eval_metrics['student_accuracy'],
            compression_ratio=eval_metrics['compression_ratio'],
            speedup_factor=eval_metrics['speedup_factor'],
            quality_retention=eval_metrics['quality_retention']
        )


class ModelDistillationAgent:
    """
    Main agent that orchestrates model distillation.
    Creates efficient student models from capable teacher models.
    """
    
    def __init__(self):
        self.distillations: Dict[str, DistillationTrainer] = {}
        self.models: Dict[str, Any] = {}
        
    def create_teacher_student_pair(
        self,
        teacher_spec: ModelSpec,
        student_spec: ModelSpec
    ) -> Tuple[TeacherModel, StudentModel]:
        """Create teacher and student models"""
        teacher = TeacherModel(teacher_spec)
        student = StudentModel(student_spec)
        
        # Store models
        self.models[teacher_spec.name] = teacher
        self.models[student_spec.name] = student
        
        return teacher, student
    
    def distill_model(
        self,
        teacher: TeacherModel,
        student: StudentModel,
        training_data: List[TrainingExample],
        config: DistillationConfig
    ) -> DistillationTrainer:
        """Distill knowledge from teacher to student"""
        # Prepare training data with teacher predictions
        print("Generating teacher predictions for training data...")
        for example in training_data:
            teacher_output = teacher.predict(example.input_text, example.task_type)
            example.teacher_output = teacher_output
            example.teacher_logits = teacher_output['logits']
            example.teacher_features = teacher_output['features']
            example.teacher_attention = teacher_output['attention']
        
        # Create and run trainer
        trainer = DistillationTrainer(teacher, student, config)
        trainer.train(training_data)
        
        # Store distillation
        key = f"{teacher.spec.name}_to_{student.spec.name}"
        self.distillations[key] = trainer
        
        return trainer
    
    def compare_models(
        self,
        teacher: TeacherModel,
        student: StudentModel,
        test_data: List[TrainingExample]
    ) -> Dict[str, Any]:
        """Compare teacher and student performance"""
        evaluator = PerformanceEvaluator(teacher, student)
        metrics = evaluator.evaluate(test_data)
        
        comparison = {
            'teacher': {
                'name': teacher.spec.name,
                'parameters': teacher.spec.parameters,
                'inference_time_ms': teacher.spec.inference_time_ms,
                'memory_mb': teacher.spec.memory_mb,
                'accuracy': metrics['teacher_accuracy']
            },
            'student': {
                'name': student.spec.name,
                'parameters': student.spec.parameters,
                'inference_time_ms': student.spec.inference_time_ms,
                'memory_mb': student.spec.memory_mb,
                'accuracy': metrics['student_accuracy']
            },
            'improvements': {
                'compression_ratio': metrics['compression_ratio'],
                'speedup_factor': metrics['speedup_factor'],
                'memory_reduction_pct': metrics['memory_reduction'],
                'quality_retention': metrics['quality_retention']
            },
            'agreement': metrics['teacher_agreement']
        }
        
        return comparison
    
    def get_distillation_report(self) -> Dict[str, Any]:
        """Get comprehensive distillation report"""
        report = {
            'total_distillations': len(self.distillations),
            'models': len(self.models),
            'distillations': {}
        }
        
        for key, trainer in self.distillations.items():
            final_metrics = trainer.training_history[-1]
            
            report['distillations'][key] = {
                'teacher': trainer.teacher.spec.name,
                'student': trainer.student.spec.name,
                'training_epochs': len(trainer.training_history),
                'final_agreement': final_metrics.teacher_agreement,
                'quality_retention': final_metrics.quality_retention,
                'compression_ratio': final_metrics.compression_ratio,
                'speedup_factor': final_metrics.speedup_factor,
                'phase': trainer.current_phase.value
            }
        
        return report


def demonstrate_model_distillation():
    """Demonstrate model distillation pattern"""
    print("\n" + "="*60)
    print("MODEL DISTILLATION PATTERN DEMONSTRATION")
    print("="*60)
    
    agent = ModelDistillationAgent()
    
    # Scenario 1: Create teacher model (large, capable)
    print("\n" + "-"*60)
    print("Scenario 1: Create Teacher and Student Models")
    print("-"*60)
    
    teacher_spec = ModelSpec(
        name="GPT-Large",
        size=ModelSize.XLARGE,
        parameters=1_000_000_000,  # 1B parameters
        layers=24,
        hidden_dim=1024,
        attention_heads=16,
        inference_time_ms=500.0,
        memory_mb=4000.0,
        capabilities=["reasoning", "knowledge", "generation"]
    )
    
    student_spec = ModelSpec(
        name="GPT-Tiny",
        size=ModelSize.TINY,
        parameters=100_000_000,  # 100M parameters (10x smaller)
        layers=6,
        hidden_dim=256,
        attention_heads=4,
        inference_time_ms=50.0,
        memory_mb=400.0,
        capabilities=["generation"]
    )
    
    teacher, student = agent.create_teacher_student_pair(teacher_spec, student_spec)
    
    print(f"Teacher Model: {teacher.spec.name}")
    print(f"  Parameters: {teacher.spec.parameters:,}")
    print(f"  Inference Time: {teacher.spec.inference_time_ms}ms")
    print(f"  Memory: {teacher.spec.memory_mb}MB")
    
    print(f"\nStudent Model: {student.spec.name}")
    print(f"  Parameters: {student.spec.parameters:,}")
    print(f"  Inference Time: {student.spec.inference_time_ms}ms")
    print(f"  Memory: {student.spec.memory_mb}MB")
    
    print(f"\nPotential Speedup: {teacher.spec.inference_time_ms / student.spec.inference_time_ms:.1f}x")
    print(f"Compression Ratio: {teacher.spec.parameters / student.spec.parameters:.1f}x")
    
    # Scenario 2: Generate training data
    print("\n" + "-"*60)
    print("Scenario 2: Prepare Training Data")
    print("-"*60)
    
    training_examples = [
        TrainingExample(
            input_text="What is the capital of France?",
            task_type="qa",
            ground_truth="Paris"
        ),
        TrainingExample(
            input_text="Explain photosynthesis in simple terms",
            task_type="explanation",
            ground_truth="Plants convert sunlight into energy"
        ),
        TrainingExample(
            input_text="Classify sentiment: This movie was amazing!",
            task_type="classification",
            ground_truth="positive"
        ),
        TrainingExample(
            input_text="Summarize: AI is transforming how we work",
            task_type="summarization",
            ground_truth="AI transforms work"
        ),
    ] * 10  # Repeat to create dataset (40 examples - reduced for demo)
    
    print(f"Created {len(training_examples)} training examples")
    print(f"Task types: {set(e.task_type for e in training_examples)}")
    
    # Scenario 3: Configure distillation
    print("\n" + "-"*60)
    print("Scenario 3: Configure Distillation")
    print("-"*60)
    
    config = DistillationConfig(
        distillation_type=DistillationType.HYBRID,
        temperature=3.0,
        alpha=0.5,
        learning_rate=1e-4,
        epochs=10,  # Reduced for demo
        batch_size=16,
        feature_matching=True,
        attention_transfer=True
    )
    
    print(f"Distillation Type: {config.distillation_type.value}")
    print(f"Temperature: {config.temperature}")
    print(f"Alpha (soft/hard balance): {config.alpha}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch Size: {config.batch_size}")
    
    # Scenario 4: Run distillation
    print("\n" + "-"*60)
    print("Scenario 4: Distill Knowledge")
    print("-"*60)
    
    trainer = agent.distill_model(teacher, student, training_examples, config)
    
    # Scenario 5: Compare models
    print("\n" + "-"*60)
    print("Scenario 5: Compare Teacher and Student")
    print("-"*60)
    
    test_examples = training_examples[:20]  # Use subset for testing
    comparison = agent.compare_models(teacher, student, test_examples)
    
    print("\nTeacher Performance:")
    print(f"  Accuracy: {comparison['teacher']['accuracy']:.2%}")
    print(f"  Inference: {comparison['teacher']['inference_time_ms']}ms")
    print(f"  Memory: {comparison['teacher']['memory_mb']}MB")
    
    print("\nStudent Performance:")
    print(f"  Accuracy: {comparison['student']['accuracy']:.2%}")
    print(f"  Inference: {comparison['student']['inference_time_ms']}ms")
    print(f"  Memory: {comparison['student']['memory_mb']}MB")
    
    print("\nImprovements:")
    print(f"  Speedup: {comparison['improvements']['speedup_factor']:.1f}x faster")
    print(f"  Compression: {comparison['improvements']['compression_ratio']:.1f}x smaller")
    print(f"  Memory Saved: {comparison['improvements']['memory_reduction_pct']:.1f}%")
    print(f"  Quality Retained: {comparison['improvements']['quality_retention']:.1%}")
    
    # Scenario 6: Get final report
    print("\n" + "-"*60)
    print("Scenario 6: Distillation Report")
    print("-"*60)
    
    report = agent.get_distillation_report()
    
    print(f"\nTotal Distillations: {report['total_distillations']}")
    print(f"Models Managed: {report['models']}")
    
    for key, info in report['distillations'].items():
        print(f"\n{key}:")
        print(f"  Training Epochs: {info['training_epochs']}")
        print(f"  Final Agreement: {info['final_agreement']:.2%}")
        print(f"  Quality Retention: {info['quality_retention']:.1%}")
        print(f"  Speedup: {info['speedup_factor']:.1f}x")
        print(f"  Status: {info['phase']}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Successfully distilled {teacher.spec.name} to {student.spec.name}")
    print(f"✓ Achieved {comparison['improvements']['speedup_factor']:.1f}x speedup")
    print(f"✓ Reduced model size by {comparison['improvements']['compression_ratio']:.1f}x")
    print(f"✓ Retained {comparison['improvements']['quality_retention']:.1%} of quality")
    print(f"✓ Saved {comparison['improvements']['memory_reduction_pct']:.1f}% memory")
    print("\nDistilled model ready for production deployment!")


if __name__ == "__main__":
    demonstrate_model_distillation()
