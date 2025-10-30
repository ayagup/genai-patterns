"""
Pattern 53: Mixture of Agents (MoA)
Description:
    Combines outputs from multiple specialized agents using layered
    aggregation and synthesis to produce superior results.
Use Cases:
    - Complex problem solving requiring diverse expertise
    - Quality improvement through multi-perspective analysis
    - Reducing individual model biases
    - Ensemble learning for LLM outputs
Key Features:
    - Layered agent architecture
    - Multiple aggregation strategies
    - Quality-weighted synthesis
    - Iterative refinement through layers
Example:
    >>> moa = MixtureOfAgents()
    >>> moa.add_layer([agent1, agent2, agent3])
    >>> moa.add_layer([synthesizer_agent])
    >>> result = moa.process(query)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import time
from collections import defaultdict
class AggregationMethod(Enum):
    """Methods for aggregating agent outputs"""
    CONCATENATE = "concatenate"
    WEIGHTED_VOTE = "weighted_vote"
    BEST_OF_N = "best_of_n"
    CONSENSUS = "consensus"
    SYNTHESIS = "synthesis"
class AgentRole(Enum):
    """Roles agents can play in MoA"""
    PROPOSER = "proposer"  # Generate initial responses
    AGGREGATOR = "aggregator"  # Combine responses
    REFINER = "refiner"  # Improve combined output
    VALIDATOR = "validator"  # Verify quality
@dataclass
class AgentConfig:
    """Configuration for an agent in MoA"""
    agent_id: str
    agent: Any
    role: AgentRole
    weight: float = 1.0
    specialty: Optional[str] = None
    quality_threshold: float = 0.5
@dataclass
class LayerOutput:
    """Output from a layer in MoA"""
    layer_index: int
    agent_outputs: Dict[str, Any]
    aggregated_output: Any
    aggregation_method: AggregationMethod
    quality_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
@dataclass
class MoAResult:
    """Final result from Mixture of Agents"""
    final_output: Any
    layer_outputs: List[LayerOutput]
    total_agents_used: int
    processing_time: float
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
class MixtureOfAgents:
    """
    Mixture of Agents system with layered architecture
    Features:
    - Multi-layer agent architecture
    - Flexible aggregation strategies
    - Quality-based weighting
    - Iterative refinement
    """
    def __init__(self):
        self.layers: List[List[AgentConfig]] = []
        self.layer_aggregation_methods: List[AggregationMethod] = []
        self.execution_history: List[MoAResult] = []
    def add_layer(
        self,
        agents: List[AgentConfig],
        aggregation_method: AggregationMethod = AggregationMethod.SYNTHESIS
    ):
        """
        Add a layer of agents
        Args:
            agents: List of agent configurations
            aggregation_method: How to aggregate outputs in this layer
        """
        self.layers.append(agents)
        self.layer_aggregation_methods.append(aggregation_method)
    def process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MoAResult:
        """
        Process query through all layers
        Args:
            query: Input query
            context: Additional context
        Returns:
            MoA result with final output
        """
        start_time = time.time()
        layer_outputs = []
        current_input = query
        total_agents = 0
        # Process through each layer
        for layer_idx, (layer_agents, agg_method) in enumerate(
            zip(self.layers, self.layer_aggregation_methods)
        ):
            layer_output = self._process_layer(
                layer_idx,
                layer_agents,
                current_input,
                context or {},
                agg_method
            )
            layer_outputs.append(layer_output)
            total_agents += len(layer_agents)
            # Use aggregated output as input for next layer
            current_input = layer_output.aggregated_output
        # Calculate final quality score
        final_quality = self._calculate_final_quality(layer_outputs)
        result = MoAResult(
            final_output=current_input,
            layer_outputs=layer_outputs,
            total_agents_used=total_agents,
            processing_time=time.time() - start_time,
            quality_score=final_quality,
            metadata={
                'num_layers': len(self.layers),
                'query': query
            }
        )
        self.execution_history.append(result)
        return result
    def _process_layer(
        self,
        layer_idx: int,
        agents: List[AgentConfig],
        input_data: Any,
        context: Dict[str, Any],
        aggregation_method: AggregationMethod
    ) -> LayerOutput:
        """Process input through a single layer"""
        agent_outputs = {}
        quality_scores = {}
        # Get output from each agent in parallel
        for agent_config in agents:
            output = self._execute_agent(
                agent_config,
                input_data,
                context
            )
            agent_outputs[agent_config.agent_id] = output
            # Calculate quality score
            quality = self._evaluate_output_quality(
                output,
                agent_config
            )
            quality_scores[agent_config.agent_id] = quality
        # Aggregate outputs
        aggregated = self._aggregate_outputs(
            agent_outputs,
            quality_scores,
            aggregation_method
        )
        return LayerOutput(
            layer_index=layer_idx,
            agent_outputs=agent_outputs,
            aggregated_output=aggregated,
            aggregation_method=aggregation_method,
            quality_scores=quality_scores,
            metadata={
                'num_agents': len(agents),
                'avg_quality': sum(quality_scores.values()) / len(quality_scores)
            }
        )
    def _execute_agent(
        self,
        agent_config: AgentConfig,
        input_data: Any,
        context: Dict[str, Any]
    ) -> Any:
        """Execute a single agent"""
        # If agent has execute method, use it
        if hasattr(agent_config.agent, 'execute'):
            return agent_config.agent.execute(input_data, context)
        # Otherwise, treat as callable
        if callable(agent_config.agent):
            return agent_config.agent(input_data, context)
        # Fallback: simulate agent execution
        role_prefix = {
            AgentRole.PROPOSER: "Initial analysis:",
            AgentRole.AGGREGATOR: "Combined perspective:",
            AgentRole.REFINER: "Refined answer:",
            AgentRole.VALIDATOR: "Validated response:"
        }
        prefix = role_prefix.get(agent_config.role, "Response:")
        if isinstance(input_data, str):
            return f"{prefix} {agent_config.specialty or agent_config.agent_id} processed: {input_data}"
        else:
            return {
                'role': agent_config.role.value,
                'agent_id': agent_config.agent_id,
                'processed_input': str(input_data)[:100]
            }
    def _evaluate_output_quality(
        self,
        output: Any,
        agent_config: AgentConfig
    ) -> float:
        """Evaluate quality of agent output"""
        # Simple heuristic-based quality scoring
        quality = 0.5  # Base quality
        # Length bonus (but not too long)
        if isinstance(output, str):
            length = len(output)
            if 50 <= length <= 500:
                quality += 0.2
            elif length > 500:
                quality += 0.1
        # Role-based quality adjustment
        if agent_config.role == AgentRole.VALIDATOR:
            quality += 0.1  # Validators get bonus
        # Specialty bonus
        if agent_config.specialty:
            quality += 0.1
        return min(quality, 1.0)
    def _aggregate_outputs(
        self,
        outputs: Dict[str, Any],
        quality_scores: Dict[str, float],
        method: AggregationMethod
    ) -> Any:
        """Aggregate outputs from multiple agents"""
        if method == AggregationMethod.CONCATENATE:
            # Concatenate all outputs
            combined = []
            for agent_id, output in outputs.items():
                combined.append(f"[{agent_id}]: {output}")
            return "\n\n".join(combined)
        elif method == AggregationMethod.WEIGHTED_VOTE:
            # Weight outputs by quality scores
            weighted_outputs = []
            for agent_id, output in outputs.items():
                weight = quality_scores.get(agent_id, 0.5)
                weighted_outputs.append((weight, output))
            # Return highest weighted output
            weighted_outputs.sort(key=lambda x: x[0], reverse=True)
            return weighted_outputs[0][1] if weighted_outputs else ""
        elif method == AggregationMethod.BEST_OF_N:
            # Return output with highest quality
            best_agent = max(quality_scores.items(), key=lambda x: x[1])[0]
            return outputs[best_agent]
        elif method == AggregationMethod.CONSENSUS:
            # Find common elements (simplified)
            if all(isinstance(o, str) for o in outputs.values()):
                # For strings, look for common phrases
                all_outputs = list(outputs.values())
                return all_outputs[0] if all_outputs else ""
            return list(outputs.values())[0] if outputs else ""
        elif method == AggregationMethod.SYNTHESIS:
            # Synthesize outputs into coherent response
            synthesis_parts = []
            # Sort by quality
            sorted_outputs = sorted(
                outputs.items(),
                key=lambda x: quality_scores.get(x[0], 0),
                reverse=True
            )
            synthesis_parts.append("Synthesized response incorporating multiple perspectives:\n")
            for i, (agent_id, output) in enumerate(sorted_outputs[:3], 1):
                quality = quality_scores.get(agent_id, 0)
                synthesis_parts.append(
                    f"\n{i}. From {agent_id} (quality: {quality:.2f}):\n{output}\n"
                )
            return "".join(synthesis_parts)
        return ""
    def _calculate_final_quality(
        self,
        layer_outputs: List[LayerOutput]
    ) -> float:
        """Calculate final quality score from all layers"""
        if not layer_outputs:
            return 0.0
        # Weight later layers more heavily
        total_score = 0.0
        total_weight = 0.0
        for i, layer_output in enumerate(layer_outputs):
            layer_weight = (i + 1) / len(layer_outputs)  # Later layers weighted more
            avg_quality = layer_output.metadata.get('avg_quality', 0.5)
            total_score += avg_quality * layer_weight
            total_weight += layer_weight
        return total_score / total_weight if total_weight > 0 else 0.0
    def get_statistics(self) -> Dict[str, Any]:
        """Get MoA execution statistics"""
        if not self.execution_history:
            return {'message': 'No executions yet'}
        avg_processing_time = sum(
            r.processing_time for r in self.execution_history
        ) / len(self.execution_history)
        avg_quality = sum(
            r.quality_score for r in self.execution_history
        ) / len(self.execution_history)
        total_agents = sum(
            r.total_agents_used for r in self.execution_history
        )
        return {
            'total_executions': len(self.execution_history),
            'num_layers': len(self.layers),
            'total_agents_configured': sum(len(layer) for layer in self.layers),
            'avg_processing_time': avg_processing_time,
            'avg_quality_score': avg_quality,
            'total_agents_executed': total_agents
        }
class SimpleAgent:
    """Simple agent for demonstration"""
    def __init__(self, name: str, specialty: Optional[str] = None):
        self.name = name
        self.specialty = specialty
    def execute(self, query: str, context: Dict[str, Any]) -> str:
        """Execute agent logic"""
        if self.specialty:
            return f"[{self.name} - {self.specialty}]: Analyzed '{query}' with specialized knowledge in {self.specialty}"
        return f"[{self.name}]: Processed '{query}'"
def main():
    """Demonstrate Mixture of Agents pattern"""
    print("=" * 60)
    print("Mixture of Agents (MoA) Demonstration")
    print("=" * 60)
    # Create MoA system
    moa = MixtureOfAgents()
    print("\n1. Building 3-Layer MoA Architecture")
    print("-" * 60)
    # Layer 1: Diverse proposers with different specialties
    layer1_agents = [
        AgentConfig(
            agent_id="technical_expert",
            agent=SimpleAgent("TechExpert", "technical analysis"),
            role=AgentRole.PROPOSER,
            weight=1.0,
            specialty="technical"
        ),
        AgentConfig(
            agent_id="creative_thinker",
            agent=SimpleAgent("CreativeThinker", "creative solutions"),
            role=AgentRole.PROPOSER,
            weight=0.9,
            specialty="creative"
        ),
        AgentConfig(
            agent_id="practical_advisor",
            agent=SimpleAgent("PracticalAdvisor", "practical implementation"),
            role=AgentRole.PROPOSER,
            weight=1.1,
            specialty="practical"
        ),
    ]
    moa.add_layer(layer1_agents, AggregationMethod.CONCATENATE)
    print("Layer 1: 3 specialized proposer agents (CONCATENATE)")
    for agent in layer1_agents:
        print(f"  - {agent.agent_id}: {agent.specialty}")
    # Layer 2: Aggregators
    layer2_agents = [
        AgentConfig(
            agent_id="synthesizer_1",
            agent=SimpleAgent("Synthesizer1"),
            role=AgentRole.AGGREGATOR,
            weight=1.0
        ),
        AgentConfig(
            agent_id="synthesizer_2",
            agent=SimpleAgent("Synthesizer2"),
            role=AgentRole.AGGREGATOR,
            weight=1.0
        ),
    ]
    moa.add_layer(layer2_agents, AggregationMethod.SYNTHESIS)
    print("\nLayer 2: 2 synthesizer agents (SYNTHESIS)")
    # Layer 3: Final refiner
    layer3_agents = [
        AgentConfig(
            agent_id="final_refiner",
            agent=SimpleAgent("FinalRefiner", "quality refinement"),
            role=AgentRole.REFINER,
            weight=1.2
        ),
    ]
    moa.add_layer(layer3_agents, AggregationMethod.BEST_OF_N)
    print("\nLayer 3: 1 final refiner agent (BEST_OF_N)")
    print("\n" + "=" * 60)
    print("2. Processing Query Through MoA")
    print("=" * 60)
    query = "How can we improve system performance?"
    print(f"\nQuery: '{query}'")
    result = moa.process(query)
    print(f"\nProcessing complete!")
    print(f"Total agents used: {result.total_agents_used}")
    print(f"Processing time: {result.processing_time:.4f}s")
    print(f"Quality score: {result.quality_score:.3f}")
    print("\n" + "=" * 60)
    print("3. Layer-by-Layer Analysis")
    print("=" * 60)
    for layer_output in result.layer_outputs:
        print(f"\nLayer {layer_output.layer_index + 1}:")
        print(f"  Aggregation method: {layer_output.aggregation_method.value}")
        print(f"  Number of agents: {layer_output.metadata['num_agents']}")
        print(f"  Average quality: {layer_output.metadata['avg_quality']:.3f}")
        print(f"\n  Agent outputs:")
        for agent_id, output in layer_output.agent_outputs.items():
            quality = layer_output.quality_scores[agent_id]
            output_preview = str(output)[:80] + "..." if len(str(output)) > 80 else str(output)
            print(f"    {agent_id} (Q: {quality:.2f}):")
            print(f"      {output_preview}")
        print(f"\n  Aggregated output:")
        agg_preview = str(layer_output.aggregated_output)[:150]
        print(f"    {agg_preview}...")
    print("\n" + "=" * 60)
    print("4. Final Output")
    print("=" * 60)
    print(f"\n{result.final_output[:300]}...")
    print("\n" + "=" * 60)
    print("5. Testing Different Aggregation Methods")
    print("=" * 60)
    # Create simpler MoA for testing
    test_moa = MixtureOfAgents()
    test_agents = [
        AgentConfig(
            agent_id=f"agent_{i}",
            agent=SimpleAgent(f"Agent{i}"),
            role=AgentRole.PROPOSER,
            weight=1.0 + (i * 0.1)
        )
        for i in range(4)
    ]
    aggregation_methods = [
        AggregationMethod.CONCATENATE,
        AggregationMethod.WEIGHTED_VOTE,
        AggregationMethod.BEST_OF_N,
        AggregationMethod.SYNTHESIS
    ]
    for method in aggregation_methods:
        test_moa = MixtureOfAgents()
        test_moa.add_layer(test_agents, method)
        result = test_moa.process("Test query")
        print(f"\n{method.value}:")
        print(f"  Output length: {len(str(result.final_output))} chars")
        print(f"  Quality: {result.quality_score:.3f}")
        print(f"  Preview: {str(result.final_output)[:100]}...")
    print("\n" + "=" * 60)
    print("6. Multiple Queries Processing")
    print("=" * 60)
    queries = [
        "Explain quantum computing",
        "Design a scalable architecture",
        "Optimize database queries",
    ]
    results = []
    for query in queries:
        result = moa.process(query)
        results.append(result)
        print(f"\nQuery: {query}")
        print(f"  Quality: {result.quality_score:.3f}")
        print(f"  Time: {result.processing_time:.4f}s")
    print("\n" + "=" * 60)
    print("7. MoA Statistics")
    print("=" * 60)
    stats = moa.get_statistics()
    print(f"\nTotal Executions: {stats['total_executions']}")
    print(f"Number of Layers: {stats['num_layers']}")
    print(f"Total Agents Configured: {stats['total_agents_configured']}")
    print(f"Average Processing Time: {stats['avg_processing_time']:.4f}s")
    print(f"Average Quality Score: {stats['avg_quality_score']:.3f}")
    print(f"Total Agents Executed: {stats['total_agents_executed']}")
    print("\n" + "=" * 60)
    print("8. Quality Improvement Across Layers")
    print("=" * 60)
    # Analyze quality progression
    if results:
        sample_result = results[0]
        print(f"\nQuality progression for: '{queries[0]}'")
        for i, layer_output in enumerate(sample_result.layer_outputs, 1):
            avg_quality = layer_output.metadata['avg_quality']
            print(f"  Layer {i}: {avg_quality:.3f}")
        print(f"  Final: {sample_result.quality_score:.3f}")
    print("\n" + "=" * 60)
    print("Mixture of Agents demonstration complete!")
    print("=" * 60)
if __name__ == "__main__":
    main()
