"""
Pattern 145: Quantization & Compression Agent

This pattern implements model quantization and compression techniques to reduce
model size and computational requirements while maintaining acceptable performance.
Techniques include weight quantization, pruning, and knowledge distillation.

Category: Performance Optimization
Use Cases:
- Deploy models on edge devices
- Reduce inference costs
- Enable mobile deployment
- Optimize resource usage
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime
import random
import math


class QuantizationType(Enum):
    """Types of quantization"""
    INT8 = "int8"           # 8-bit integer
    INT4 = "int4"           # 4-bit integer
    INT2 = "int2"           # 2-bit integer
    FP16 = "fp16"           # 16-bit float
    BF16 = "bf16"           # Brain float 16
    MIXED = "mixed"         # Mixed precision
    DYNAMIC = "dynamic"     # Dynamic quantization


class PruningStrategy(Enum):
    """Pruning strategies"""
    MAGNITUDE = "magnitude"       # Remove small weights
    STRUCTURED = "structured"     # Remove entire structures
    UNSTRUCTURED = "unstructured" # Remove individual weights
    GRADUAL = "gradual"          # Gradually increase sparsity
    LOTTERY = "lottery"          # Lottery ticket hypothesis
    LAYER_WISE = "layer_wise"    # Per-layer pruning


class CompressionTechnique(Enum):
    """Compression techniques"""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    TENSOR_DECOMPOSITION = "tensor_decomposition"
    WEIGHT_SHARING = "weight_sharing"
    HUFFMAN_CODING = "huffman_coding"


@dataclass
class ModelWeights:
    """Model weights with metadata"""
    layer_name: str
    weights: List[float]
    shape: Tuple[int, ...]
    dtype: str = "float32"
    sparsity: float = 0.0
    quantized: bool = False


@dataclass
class QuantizationConfig:
    """Configuration for quantization"""
    quantization_type: QuantizationType
    per_channel: bool = False
    symmetric: bool = True
    calibration_samples: int = 100
    outlier_threshold: float = 6.0


@dataclass
class PruningConfig:
    """Configuration for pruning"""
    strategy: PruningStrategy
    target_sparsity: float = 0.5
    gradual_steps: int = 10
    initial_sparsity: float = 0.0
    layer_wise_ratios: Optional[Dict[str, float]] = None


@dataclass
class CompressionMetrics:
    """Metrics for compression"""
    technique: CompressionTechnique
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    accuracy_before: float
    accuracy_after: float
    accuracy_loss: float
    inference_speedup: float
    memory_reduction_pct: float
    timestamp: datetime = field(default_factory=datetime.now)


class WeightQuantizer:
    """Quantizes model weights to lower precision"""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.scale_factors: Dict[str, float] = {}
        self.zero_points: Dict[str, float] = {}
        
    def quantize_weights(self, weights: ModelWeights) -> ModelWeights:
        """Quantize weights to target precision"""
        if self.config.quantization_type == QuantizationType.INT8:
            return self._quantize_int8(weights)
        elif self.config.quantization_type == QuantizationType.INT4:
            return self._quantize_int4(weights)
        elif self.config.quantization_type == QuantizationType.FP16:
            return self._quantize_fp16(weights)
        elif self.config.quantization_type == QuantizationType.MIXED:
            return self._quantize_mixed(weights)
        else:
            return weights
    
    def _quantize_int8(self, weights: ModelWeights) -> ModelWeights:
        """Quantize to 8-bit integers"""
        # Find range
        min_val = min(weights.weights)
        max_val = max(weights.weights)
        
        if self.config.symmetric:
            # Symmetric quantization
            abs_max = max(abs(min_val), abs(max_val))
            scale = abs_max / 127.0
            zero_point = 0
        else:
            # Asymmetric quantization
            scale = (max_val - min_val) / 255.0
            zero_point = -min_val / scale
        
        # Store calibration
        self.scale_factors[weights.layer_name] = scale
        self.zero_points[weights.layer_name] = zero_point
        
        # Quantize
        quantized = []
        for w in weights.weights:
            q = round(w / scale + zero_point)
            q = max(-128, min(127, q))  # Clip to int8 range
            # Dequantize for simulation
            dequant = (q - zero_point) * scale
            quantized.append(dequant)
        
        return ModelWeights(
            layer_name=weights.layer_name,
            weights=quantized,
            shape=weights.shape,
            dtype="int8",
            sparsity=weights.sparsity,
            quantized=True
        )
    
    def _quantize_int4(self, weights: ModelWeights) -> ModelWeights:
        """Quantize to 4-bit integers"""
        # 4-bit quantization (even more aggressive)
        min_val = min(weights.weights)
        max_val = max(weights.weights)
        
        scale = (max_val - min_val) / 15.0  # 4 bits = 16 levels
        zero_point = -min_val / scale
        
        self.scale_factors[weights.layer_name] = scale
        self.zero_points[weights.layer_name] = zero_point
        
        quantized = []
        for w in weights.weights:
            q = round(w / scale + zero_point)
            q = max(0, min(15, q))  # Clip to 4-bit range
            dequant = (q - zero_point) * scale
            quantized.append(dequant)
        
        return ModelWeights(
            layer_name=weights.layer_name,
            weights=quantized,
            shape=weights.shape,
            dtype="int4",
            sparsity=weights.sparsity,
            quantized=True
        )
    
    def _quantize_fp16(self, weights: ModelWeights) -> ModelWeights:
        """Quantize to 16-bit floating point"""
        # FP16 has less precision than FP32
        # Simulate by rounding to fewer significant digits
        quantized = [round(w, 4) for w in weights.weights]
        
        return ModelWeights(
            layer_name=weights.layer_name,
            weights=quantized,
            shape=weights.shape,
            dtype="fp16",
            sparsity=weights.sparsity,
            quantized=True
        )
    
    def _quantize_mixed(self, weights: ModelWeights) -> ModelWeights:
        """Mixed precision quantization"""
        # Use different precision for different value ranges
        quantized = []
        
        for w in weights.weights:
            if abs(w) > self.config.outlier_threshold:
                # Keep outliers in FP16
                quantized.append(round(w, 4))
            else:
                # Quantize normal values to INT8
                scale = self.config.outlier_threshold / 127.0
                q = round(w / scale)
                q = max(-128, min(127, q))
                dequant = q * scale
                quantized.append(dequant)
        
        return ModelWeights(
            layer_name=weights.layer_name,
            weights=quantized,
            shape=weights.shape,
            dtype="mixed",
            sparsity=weights.sparsity,
            quantized=True
        )


class ModelPruner:
    """Prunes model weights to increase sparsity"""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.pruning_masks: Dict[str, List[bool]] = {}
        
    def prune_weights(self, weights: ModelWeights) -> ModelWeights:
        """Prune weights based on strategy"""
        if self.config.strategy == PruningStrategy.MAGNITUDE:
            return self._prune_by_magnitude(weights)
        elif self.config.strategy == PruningStrategy.STRUCTURED:
            return self._prune_structured(weights)
        elif self.config.strategy == PruningStrategy.GRADUAL:
            return self._prune_gradual(weights)
        else:
            return self._prune_by_magnitude(weights)
    
    def _prune_by_magnitude(self, weights: ModelWeights) -> ModelWeights:
        """Prune smallest magnitude weights"""
        # Get layer-specific sparsity target
        if self.config.layer_wise_ratios and weights.layer_name in self.config.layer_wise_ratios:
            target_sparsity = self.config.layer_wise_ratios[weights.layer_name]
        else:
            target_sparsity = self.config.target_sparsity
        
        # Sort weights by magnitude
        indexed_weights = [(i, abs(w)) for i, w in enumerate(weights.weights)]
        indexed_weights.sort(key=lambda x: x[1])
        
        # Determine pruning threshold
        num_to_prune = int(len(weights.weights) * target_sparsity)
        prune_indices = set(idx for idx, _ in indexed_weights[:num_to_prune])
        
        # Create pruning mask
        mask = [i not in prune_indices for i in range(len(weights.weights))]
        self.pruning_masks[weights.layer_name] = mask
        
        # Apply pruning
        pruned = [w if mask[i] else 0.0 for i, w in enumerate(weights.weights)]
        
        # Calculate actual sparsity
        actual_sparsity = sum(1 for w in pruned if w == 0.0) / len(pruned)
        
        return ModelWeights(
            layer_name=weights.layer_name,
            weights=pruned,
            shape=weights.shape,
            dtype=weights.dtype,
            sparsity=actual_sparsity,
            quantized=weights.quantized
        )
    
    def _prune_structured(self, weights: ModelWeights) -> ModelWeights:
        """Structured pruning (e.g., entire rows/columns)"""
        # For demonstration, prune in blocks
        block_size = 4
        pruned = list(weights.weights)
        
        target_sparsity = self.config.target_sparsity
        num_blocks = len(pruned) // block_size
        num_to_prune = int(num_blocks * target_sparsity)
        
        # Calculate block magnitudes
        blocks = []
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size
            block_magnitude = sum(abs(w) for w in pruned[start:end])
            blocks.append((i, block_magnitude))
        
        # Sort and prune smallest blocks
        blocks.sort(key=lambda x: x[1])
        prune_block_ids = set(idx for idx, _ in blocks[:num_to_prune])
        
        # Zero out pruned blocks
        for block_id in prune_block_ids:
            start = block_id * block_size
            end = start + block_size
            for i in range(start, end):
                if i < len(pruned):
                    pruned[i] = 0.0
        
        actual_sparsity = sum(1 for w in pruned if w == 0.0) / len(pruned)
        
        return ModelWeights(
            layer_name=weights.layer_name,
            weights=pruned,
            shape=weights.shape,
            dtype=weights.dtype,
            sparsity=actual_sparsity,
            quantized=weights.quantized
        )
    
    def _prune_gradual(self, weights: ModelWeights) -> ModelWeights:
        """Gradual pruning with increasing sparsity"""
        # Simulate gradual pruning (would be over multiple steps in practice)
        current_sparsity = weights.sparsity
        target_sparsity = self.config.target_sparsity
        
        # Calculate sparsity for this step
        step_sparsity = current_sparsity + (target_sparsity - current_sparsity) / self.config.gradual_steps
        
        # Create temporary config for this step
        temp_config = PruningConfig(
            strategy=PruningStrategy.MAGNITUDE,
            target_sparsity=step_sparsity
        )
        temp_pruner = ModelPruner(temp_config)
        
        return temp_pruner._prune_by_magnitude(weights)


class TensorDecomposer:
    """Performs tensor decomposition for compression"""
    
    def decompose_layer(self, weights: ModelWeights) -> Tuple[ModelWeights, Optional[ModelWeights]]:
        """Decompose weight matrix into low-rank factors"""
        # Simulate SVD-based decomposition
        # In practice, would use actual SVD
        
        # Assume 2D weight matrix
        if len(weights.shape) != 2:
            return weights, None
        
        rows, cols = weights.shape
        rank = min(rows, cols) // 2  # Use half rank for compression
        
        # Create low-rank factors (simulated)
        factor1 = [random.gauss(0, 1) for _ in range(rows * rank)]
        factor2 = [random.gauss(0, 1) for _ in range(rank * cols)]
        
        return (
            ModelWeights(
                layer_name=f"{weights.layer_name}_U",
                weights=factor1,
                shape=(rows, rank),
                dtype=weights.dtype
            ),
            ModelWeights(
                layer_name=f"{weights.layer_name}_V",
                weights=factor2,
                shape=(rank, cols),
                dtype=weights.dtype
            )
        )


class CompressionAnalyzer:
    """Analyzes compression effects"""
    
    def calculate_size(self, weights: ModelWeights) -> float:
        """Calculate size in MB"""
        num_params = len(weights.weights)
        
        # Bytes per parameter based on dtype
        if weights.dtype == "float32":
            bytes_per_param = 4
        elif weights.dtype in ["float16", "fp16", "bf16"]:
            bytes_per_param = 2
        elif weights.dtype == "int8":
            bytes_per_param = 1
        elif weights.dtype == "int4":
            bytes_per_param = 0.5
        else:
            bytes_per_param = 4
        
        # Account for sparsity (sparse storage)
        if weights.sparsity > 0:
            # Sparse storage: indices + values
            active_params = num_params * (1 - weights.sparsity)
            bytes_per_param += 4 / num_params  # Index overhead
            return (active_params * bytes_per_param) / (1024 * 1024)
        
        return (num_params * bytes_per_param) / (1024 * 1024)
    
    def estimate_speedup(self, weights: ModelWeights) -> float:
        """Estimate inference speedup"""
        speedup = 1.0
        
        # Quantization speedup
        if weights.quantized:
            if weights.dtype == "int8":
                speedup *= 2.0  # INT8 is ~2x faster than FP32
            elif weights.dtype == "int4":
                speedup *= 3.0  # INT4 is ~3x faster
            elif weights.dtype in ["fp16", "bf16"]:
                speedup *= 1.5  # FP16 is ~1.5x faster
        
        # Sparsity speedup (if hardware supports it)
        if weights.sparsity > 0.5:
            speedup *= 1.0 + weights.sparsity  # Up to 2x with 100% sparsity
        
        return speedup
    
    def estimate_accuracy_loss(
        self,
        original: ModelWeights,
        compressed: ModelWeights
    ) -> float:
        """Estimate accuracy loss from compression"""
        # Calculate weight difference
        differences = [
            abs(o - c) for o, c in zip(original.weights, compressed.weights)
        ]
        
        # Average relative error
        avg_diff = sum(differences) / len(differences)
        avg_magnitude = sum(abs(w) for w in original.weights) / len(original.weights)
        
        if avg_magnitude > 0:
            relative_error = avg_diff / avg_magnitude
        else:
            relative_error = 0.0
        
        # Estimate accuracy loss (heuristic)
        # More aggressive compression = more accuracy loss
        accuracy_loss = min(0.15, relative_error * 0.1)  # Cap at 15% loss
        
        return accuracy_loss


class QuantizationCompressionAgent:
    """
    Main agent that orchestrates model quantization and compression.
    Reduces model size and computational requirements.
    """
    
    def __init__(self):
        self.original_weights: Dict[str, ModelWeights] = {}
        self.compressed_weights: Dict[str, ModelWeights] = {}
        self.metrics: List[CompressionMetrics] = []
        self.analyzer = CompressionAnalyzer()
        
    def add_layer(self, layer_name: str, weights: List[float], shape: Tuple[int, ...]):
        """Add a layer to the model"""
        self.original_weights[layer_name] = ModelWeights(
            layer_name=layer_name,
            weights=weights,
            shape=shape
        )
    
    def quantize_model(
        self,
        config: QuantizationConfig,
        layers: Optional[List[str]] = None
    ) -> Dict[str, ModelWeights]:
        """Quantize model weights"""
        quantizer = WeightQuantizer(config)
        
        if layers is None:
            layers = list(self.original_weights.keys())
        
        quantized = {}
        total_original_size = 0.0
        total_compressed_size = 0.0
        total_accuracy_loss = 0.0
        
        for layer_name in layers:
            if layer_name not in self.original_weights:
                continue
            
            original = self.original_weights[layer_name]
            quantized_weights = quantizer.quantize_weights(original)
            quantized[layer_name] = quantized_weights
            self.compressed_weights[layer_name] = quantized_weights
            
            # Calculate metrics
            orig_size = self.analyzer.calculate_size(original)
            comp_size = self.analyzer.calculate_size(quantized_weights)
            speedup = self.analyzer.estimate_speedup(quantized_weights)
            acc_loss = self.analyzer.estimate_accuracy_loss(original, quantized_weights)
            
            total_original_size += orig_size
            total_compressed_size += comp_size
            total_accuracy_loss += acc_loss
            
            # Store metrics
            metrics = CompressionMetrics(
                technique=CompressionTechnique.QUANTIZATION,
                original_size_mb=orig_size,
                compressed_size_mb=comp_size,
                compression_ratio=orig_size / comp_size if comp_size > 0 else 1.0,
                accuracy_before=1.0,
                accuracy_after=1.0 - acc_loss,
                accuracy_loss=acc_loss,
                inference_speedup=speedup,
                memory_reduction_pct=(1 - comp_size / orig_size) * 100 if orig_size > 0 else 0
            )
            self.metrics.append(metrics)
        
        return quantized
    
    def prune_model(
        self,
        config: PruningConfig,
        layers: Optional[List[str]] = None
    ) -> Dict[str, ModelWeights]:
        """Prune model weights"""
        pruner = ModelPruner(config)
        
        if layers is None:
            layers = list(self.original_weights.keys())
        
        pruned = {}
        total_original_size = 0.0
        total_compressed_size = 0.0
        total_accuracy_loss = 0.0
        
        for layer_name in layers:
            if layer_name not in self.original_weights:
                continue
            
            original = self.original_weights[layer_name]
            pruned_weights = pruner.prune_weights(original)
            pruned[layer_name] = pruned_weights
            self.compressed_weights[layer_name] = pruned_weights
            
            # Calculate metrics
            orig_size = self.analyzer.calculate_size(original)
            comp_size = self.analyzer.calculate_size(pruned_weights)
            speedup = self.analyzer.estimate_speedup(pruned_weights)
            acc_loss = self.analyzer.estimate_accuracy_loss(original, pruned_weights)
            
            total_original_size += orig_size
            total_compressed_size += comp_size
            total_accuracy_loss += acc_loss
            
            # Store metrics
            metrics = CompressionMetrics(
                technique=CompressionTechnique.PRUNING,
                original_size_mb=orig_size,
                compressed_size_mb=comp_size,
                compression_ratio=orig_size / comp_size if comp_size > 0 else 1.0,
                accuracy_before=1.0,
                accuracy_after=1.0 - acc_loss,
                accuracy_loss=acc_loss,
                inference_speedup=speedup,
                memory_reduction_pct=(1 - comp_size / orig_size) * 100 if orig_size > 0 else 0
            )
            self.metrics.append(metrics)
        
        return pruned
    
    def compress_hybrid(
        self,
        quant_config: QuantizationConfig,
        prune_config: PruningConfig,
        layers: Optional[List[str]] = None
    ) -> Dict[str, ModelWeights]:
        """Apply both quantization and pruning"""
        if layers is None:
            layers = list(self.original_weights.keys())
        
        # First prune
        pruned = self.prune_model(prune_config, layers)
        
        # Then quantize the pruned weights
        quantizer = WeightQuantizer(quant_config)
        compressed = {}
        
        for layer_name, pruned_weights in pruned.items():
            quantized = quantizer.quantize_weights(pruned_weights)
            compressed[layer_name] = quantized
            self.compressed_weights[layer_name] = quantized
        
        return compressed
    
    def get_compression_report(self) -> Dict[str, Any]:
        """Get comprehensive compression report"""
        if not self.metrics:
            return {"error": "No compression performed yet"}
        
        # Aggregate metrics
        total_techniques = len(set(m.technique for m in self.metrics))
        total_layers = len(self.compressed_weights)
        
        avg_compression = sum(m.compression_ratio for m in self.metrics) / len(self.metrics)
        avg_speedup = sum(m.inference_speedup for m in self.metrics) / len(self.metrics)
        avg_accuracy_loss = sum(m.accuracy_loss for m in self.metrics) / len(self.metrics)
        avg_memory_reduction = sum(m.memory_reduction_pct for m in self.metrics) / len(self.metrics)
        
        # Calculate total sizes
        total_original = sum(
            self.analyzer.calculate_size(w) for w in self.original_weights.values()
        )
        total_compressed = sum(
            self.analyzer.calculate_size(w) for w in self.compressed_weights.values()
        )
        
        report = {
            'summary': {
                'techniques_applied': total_techniques,
                'layers_compressed': total_layers,
                'total_original_size_mb': total_original,
                'total_compressed_size_mb': total_compressed,
                'overall_compression_ratio': total_original / total_compressed if total_compressed > 0 else 1.0
            },
            'averages': {
                'compression_ratio': avg_compression,
                'inference_speedup': avg_speedup,
                'accuracy_loss': avg_accuracy_loss,
                'memory_reduction_pct': avg_memory_reduction
            },
            'by_technique': {}
        }
        
        # Group by technique
        for technique in CompressionTechnique:
            technique_metrics = [m for m in self.metrics if m.technique == technique]
            if technique_metrics:
                report['by_technique'][technique.value] = {
                    'count': len(technique_metrics),
                    'avg_compression': sum(m.compression_ratio for m in technique_metrics) / len(technique_metrics),
                    'avg_speedup': sum(m.inference_speedup for m in technique_metrics) / len(technique_metrics),
                    'avg_accuracy_loss': sum(m.accuracy_loss for m in technique_metrics) / len(technique_metrics)
                }
        
        return report
    
    def compare_techniques(self) -> Dict[str, Any]:
        """Compare different compression techniques"""
        comparison = {}
        
        for technique in CompressionTechnique:
            technique_metrics = [m for m in self.metrics if m.technique == technique]
            if technique_metrics:
                comparison[technique.value] = {
                    'layers': len(technique_metrics),
                    'compression_ratio': sum(m.compression_ratio for m in technique_metrics) / len(technique_metrics),
                    'speedup': sum(m.inference_speedup for m in technique_metrics) / len(technique_metrics),
                    'accuracy_retention': 1.0 - sum(m.accuracy_loss for m in technique_metrics) / len(technique_metrics),
                    'memory_saved_pct': sum(m.memory_reduction_pct for m in technique_metrics) / len(technique_metrics)
                }
        
        return comparison


def demonstrate_quantization_compression():
    """Demonstrate quantization and compression pattern"""
    print("\n" + "="*60)
    print("QUANTIZATION & COMPRESSION PATTERN DEMONSTRATION")
    print("="*60)
    
    agent = QuantizationCompressionAgent()
    
    # Scenario 1: Create model layers
    print("\n" + "-"*60)
    print("Scenario 1: Create Model Layers")
    print("-"*60)
    
    # Generate synthetic weights for demonstration (scaled down for speed)
    layers = {
        'embedding': (1000, 128),      # Reduced from (50000, 768)
        'attention.q': (128, 128),
        'attention.k': (128, 128),
        'attention.v': (128, 128),
        'ffn.dense1': (128, 512),      # Reduced from (768, 3072)
        'ffn.dense2': (512, 128),      # Reduced from (3072, 768)
        'output': (128, 1000)          # Reduced from (768, 50000)
    }
    
    for layer_name, shape in layers.items():
        num_params = shape[0] * shape[1]
        # Generate random weights
        weights = [random.gauss(0, 0.1) for _ in range(num_params)]
        agent.add_layer(layer_name, weights, shape)
        print(f"✓ Added layer: {layer_name} - Shape: {shape}, Params: {num_params:,}")
    
    total_params = sum(len(w.weights) for w in agent.original_weights.values())
    total_size = sum(agent.analyzer.calculate_size(w) for w in agent.original_weights.values())
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Total Size: {total_size:.2f} MB")
    
    # Scenario 2: INT8 Quantization
    print("\n" + "-"*60)
    print("Scenario 2: INT8 Quantization")
    print("-"*60)
    
    quant_config_int8 = QuantizationConfig(
        quantization_type=QuantizationType.INT8,
        symmetric=True
    )
    
    quantized_int8 = agent.quantize_model(quant_config_int8)
    
    print(f"Quantized {len(quantized_int8)} layers to INT8")
    
    # Show one example
    example_layer = list(quantized_int8.keys())[0]
    example = quantized_int8[example_layer]
    print(f"\nExample: {example_layer}")
    print(f"  Dtype: {example.dtype}")
    print(f"  Quantized: {example.quantized}")
    print(f"  Original weights sample: {agent.original_weights[example_layer].weights[:5]}")
    print(f"  Quantized weights sample: {example.weights[:5]}")
    
    # Scenario 3: FP16 Quantization
    print("\n" + "-"*60)
    print("Scenario 3: FP16 Quantization")
    print("-"*60)
    
    # Reset for new compression
    agent.compressed_weights.clear()
    agent.metrics.clear()
    
    quant_config_fp16 = QuantizationConfig(
        quantization_type=QuantizationType.FP16
    )
    
    quantized_fp16 = agent.quantize_model(quant_config_fp16)
    
    print(f"Quantized {len(quantized_fp16)} layers to FP16")
    
    # Scenario 4: Magnitude-based Pruning
    print("\n" + "-"*60)
    print("Scenario 4: Magnitude-Based Pruning")
    print("-"*60)
    
    # Reset
    agent.compressed_weights.clear()
    agent.metrics.clear()
    
    prune_config = PruningConfig(
        strategy=PruningStrategy.MAGNITUDE,
        target_sparsity=0.5  # 50% sparsity
    )
    
    pruned = agent.prune_model(prune_config)
    
    print(f"Pruned {len(pruned)} layers to 50% sparsity")
    
    for layer_name, weights in pruned.items():
        print(f"  {layer_name}: {weights.sparsity:.1%} sparsity")
    
    # Scenario 5: Structured Pruning
    print("\n" + "-"*60)
    print("Scenario 5: Structured Pruning")
    print("-"*60)
    
    # Reset
    agent.compressed_weights.clear()
    agent.metrics.clear()
    
    prune_config_structured = PruningConfig(
        strategy=PruningStrategy.STRUCTURED,
        target_sparsity=0.3  # 30% sparsity
    )
    
    pruned_structured = agent.prune_model(prune_config_structured)
    
    print(f"Applied structured pruning to {len(pruned_structured)} layers")
    avg_sparsity = sum(w.sparsity for w in pruned_structured.values()) / len(pruned_structured)
    print(f"Average sparsity: {avg_sparsity:.1%}")
    
    # Scenario 6: Hybrid Compression
    print("\n" + "-"*60)
    print("Scenario 6: Hybrid Compression (Prune + Quantize)")
    print("-"*60)
    
    # Reset
    agent.compressed_weights.clear()
    agent.metrics.clear()
    
    hybrid_quant = QuantizationConfig(
        quantization_type=QuantizationType.INT8
    )
    
    hybrid_prune = PruningConfig(
        strategy=PruningStrategy.MAGNITUDE,
        target_sparsity=0.6  # 60% sparsity
    )
    
    hybrid = agent.compress_hybrid(hybrid_quant, hybrid_prune)
    
    print(f"Applied hybrid compression to {len(hybrid)} layers")
    
    example_hybrid = hybrid[list(hybrid.keys())[0]]
    print(f"\nExample: {example_hybrid.layer_name}")
    print(f"  Dtype: {example_hybrid.dtype}")
    print(f"  Sparsity: {example_hybrid.sparsity:.1%}")
    print(f"  Quantized: {example_hybrid.quantized}")
    
    # Scenario 7: Compression Report
    print("\n" + "-"*60)
    print("Scenario 7: Compression Report")
    print("-"*60)
    
    report = agent.get_compression_report()
    
    print("\nSummary:")
    print(f"  Techniques Applied: {report['summary']['techniques_applied']}")
    print(f"  Layers Compressed: {report['summary']['layers_compressed']}")
    print(f"  Original Size: {report['summary']['total_original_size_mb']:.2f} MB")
    print(f"  Compressed Size: {report['summary']['total_compressed_size_mb']:.2f} MB")
    print(f"  Overall Compression: {report['summary']['overall_compression_ratio']:.2f}x")
    
    print("\nAverages:")
    print(f"  Compression Ratio: {report['averages']['compression_ratio']:.2f}x")
    print(f"  Inference Speedup: {report['averages']['inference_speedup']:.2f}x")
    print(f"  Accuracy Loss: {report['averages']['accuracy_loss']:.2%}")
    print(f"  Memory Reduction: {report['averages']['memory_reduction_pct']:.1f}%")
    
    print("\nBy Technique:")
    for technique, metrics in report['by_technique'].items():
        print(f"  {technique}:")
        print(f"    Layers: {metrics['count']}")
        print(f"    Avg Compression: {metrics['avg_compression']:.2f}x")
        print(f"    Avg Speedup: {metrics['avg_speedup']:.2f}x")
        print(f"    Avg Accuracy Loss: {metrics['avg_accuracy_loss']:.2%}")
    
    # Scenario 8: Compare Techniques
    print("\n" + "-"*60)
    print("Scenario 8: Technique Comparison")
    print("-"*60)
    
    comparison = agent.compare_techniques()
    
    print("\nTechnique Performance:")
    for technique, metrics in comparison.items():
        print(f"\n{technique.upper()}:")
        print(f"  Compression Ratio: {metrics['compression_ratio']:.2f}x")
        print(f"  Speedup: {metrics['speedup']:.2f}x")
        print(f"  Accuracy Retention: {metrics['accuracy_retention']:.1%}")
        print(f"  Memory Saved: {metrics['memory_saved_pct']:.1f}%")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Compressed {report['summary']['layers_compressed']} model layers")
    print(f"✓ Achieved {report['summary']['overall_compression_ratio']:.2f}x compression")
    print(f"✓ {report['averages']['inference_speedup']:.2f}x faster inference")
    print(f"✓ Saved {report['averages']['memory_reduction_pct']:.1f}% memory")
    print(f"✓ Minimal accuracy loss: {report['averages']['accuracy_loss']:.2%}")
    print("\n✅ Performance Optimization Category: COMPLETE (5/5 patterns)")
    print("Ready for efficient edge deployment!")


if __name__ == "__main__":
    demonstrate_quantization_compression()
