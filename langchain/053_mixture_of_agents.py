"""
Pattern 053: Mixture of Agents (MoA)

Description:
    The Mixture of Agents pattern combines outputs from multiple specialized agents
    using learned or rule-based aggregation strategies. Each agent brings specific
    expertise, and their outputs are intelligently combined to produce superior
    results that leverage the strengths of each individual agent.

Components:
    1. Specialized Agents: Multiple agents with different capabilities/models
    2. Proposer Agents: Generate initial candidate responses
    3. Aggregator Agent: Combines and synthesizes multiple responses
    4. Confidence Scoring: Weights agent contributions
    5. Quality Evaluation: Assesses individual and combined outputs

Use Cases:
    - Complex tasks requiring diverse expertise
    - Multi-domain question answering
    - Code generation with multiple approaches
    - Creative content requiring varied perspectives
    - Research synthesis from multiple sources
    - Decision-making with multiple viewpoints

LangChain Implementation:
    Uses multiple LLM instances with different configurations or models,
    then aggregates their outputs using a meta-agent that synthesizes
    the best aspects of each response.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class AgentRole(Enum):
    """Roles for agents in MoA"""
    PROPOSER = "proposer"  # Generates candidate responses
    AGGREGATOR = "aggregator"  # Combines multiple responses
    CRITIC = "critic"  # Evaluates quality
    SPECIALIST = "specialist"  # Domain expert


class AggregationStrategy(Enum):
    """Strategies for combining agent outputs"""
    SYNTHESIS = "synthesis"  # Synthesize best parts
    VOTING = "voting"  # Majority vote or ranking
    WEIGHTED = "weighted"  # Weighted by confidence
    SEQUENTIAL = "sequential"  # Sequential refinement
    BEST_OF = "best_of"  # Select single best


@dataclass
class AgentOutput:
    """Output from a single agent"""
    agent_id: str
    agent_role: AgentRole
    response: str
    confidence: float  # 0.0-1.0
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.agent_role.value,
            "response": self.response[:150] + "..." if len(self.response) > 150 else self.response,
            "confidence": f"{self.confidence:.2f}",
            "reasoning": self.reasoning[:100] + "..." if self.reasoning and len(self.reasoning) > 100 else self.reasoning
        }


@dataclass
class MoAResult:
    """Result from Mixture of Agents execution"""
    query: str
    final_response: str
    aggregation_strategy: AggregationStrategy
    agent_outputs: List[AgentOutput]
    aggregator_reasoning: str
    confidence_score: float
    execution_time_ms: float
    layer_count: int
    
    @property
    def num_agents(self) -> int:
        return len(self.agent_outputs)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "final_response": self.final_response[:200] + "..." if len(self.final_response) > 200 else self.final_response,
            "strategy": self.aggregation_strategy.value,
            "num_agents": self.num_agents,
            "layer_count": self.layer_count,
            "confidence": f"{self.confidence_score:.2f}",
            "execution_time_ms": f"{self.execution_time_ms:.1f}"
        }


class MixtureOfAgents:
    """
    Mixture of Agents implementation with multiple aggregation strategies.
    
    This implementation provides:
    1. Multiple proposer agents generating diverse responses
    2. Aggregator agent synthesizing best combined response
    3. Layered architecture for iterative refinement
    4. Confidence-based weighting
    5. Multiple aggregation strategies
    """
    
    def __init__(
        self,
        num_proposers: int = 3,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.SYNTHESIS,
        temperature: float = 0.7,
        use_layers: bool = True,
        num_layers: int = 2
    ):
        self.num_proposers = num_proposers
        self.aggregation_strategy = aggregation_strategy
        self.temperature = temperature
        self.use_layers = use_layers
        self.num_layers = num_layers if use_layers else 1
        
        # Create proposer agents with varied temperatures for diversity
        self.proposers = [
            ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=temperature + (i * 0.1)  # Vary temperature
            )
            for i in range(num_proposers)
        ]
        
        # Aggregator agent (lower temperature for focused synthesis)
        self.aggregator = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )
    
    def _generate_proposer_response(
        self,
        agent_id: str,
        llm: ChatOpenAI,
        query: str,
        context: Optional[str] = None
    ) -> AgentOutput:
        """Generate response from a proposer agent"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert assistant providing thoughtful, comprehensive answers.
Provide your best response to the user's question."""),
            ("user", "{query}")
        ])
        
        if context:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert assistant. You have access to previous responses from other agents.
Consider their insights but provide your own unique perspective and improvements.

Previous responses:
{context}"""),
                ("user", "Now provide your response to: {query}")
            ])
        
        try:
            chain = prompt | llm | StrOutputParser()
            
            if context:
                response = chain.invoke({"query": query, "context": context})
            else:
                response = chain.invoke({"query": query})
            
            # Estimate confidence based on response length and coherence
            confidence = min(0.9, 0.5 + (len(response) / 1000))
            
            return AgentOutput(
                agent_id=agent_id,
                agent_role=AgentRole.PROPOSER,
                response=response,
                confidence=confidence
            )
            
        except Exception as e:
            return AgentOutput(
                agent_id=agent_id,
                agent_role=AgentRole.PROPOSER,
                response=f"Error: {str(e)}",
                confidence=0.0,
                metadata={"error": True}
            )
    
    def _aggregate_synthesis(
        self,
        query: str,
        agent_outputs: List[AgentOutput]
    ) -> Tuple[str, str, float]:
        """Synthesize multiple responses into one cohesive answer"""
        
        # Prepare responses for aggregator
        responses_text = "\n\n".join([
            f"Agent {i+1} (Confidence: {output.confidence:.2f}):\n{output.response}"
            for i, output in enumerate(agent_outputs)
            if not output.metadata.get("error")
        ])
        
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert aggregator. You receive multiple responses to the same question.

Your task:
1. Identify the best insights from each response
2. Synthesize them into a single, comprehensive answer
3. Resolve any contradictions
4. Provide a cohesive, well-structured response

Be concise but thorough. Leverage the strengths of each response."""),
            ("user", """Question: {query}

Multiple Agent Responses:
{responses}

Synthesize these into the best possible answer:""")
        ])
        
        chain = synthesis_prompt | self.aggregator | StrOutputParser()
        
        try:
            final_response = chain.invoke({
                "query": query,
                "responses": responses_text
            })
            
            # Calculate confidence as weighted average
            valid_outputs = [o for o in agent_outputs if not o.metadata.get("error")]
            avg_confidence = sum(o.confidence for o in valid_outputs) / len(valid_outputs) if valid_outputs else 0.0
            
            reasoning = "Synthesized best insights from all agents"
            
            return final_response, reasoning, avg_confidence
            
        except Exception as e:
            # Fallback to best individual response
            best = max(agent_outputs, key=lambda x: x.confidence)
            return best.response, f"Synthesis failed: {str(e)}", best.confidence
    
    def _aggregate_voting(
        self,
        query: str,
        agent_outputs: List[AgentOutput]
    ) -> Tuple[str, str, float]:
        """Use voting to select best response"""
        
        # For voting, we'll use the aggregator to rank responses
        responses_text = "\n\n".join([
            f"Response {i+1}:\n{output.response}"
            for i, output in enumerate(agent_outputs)
            if not output.metadata.get("error")
        ])
        
        voting_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are evaluating multiple responses to select the best one.

Rank the responses and select the winner. Explain your choice briefly.

Format:
Winner: [Response number]
Reason: [Brief explanation]"""),
            ("user", """Question: {query}

Responses to evaluate:
{responses}

Select the best response:""")
        ])
        
        chain = voting_prompt | self.aggregator | StrOutputParser()
        
        try:
            evaluation = chain.invoke({
                "query": query,
                "responses": responses_text
            })
            
            # Parse winner (simplified)
            winner_idx = 0
            for i in range(len(agent_outputs)):
                if f"Response {i+1}" in evaluation or f"#{i+1}" in evaluation:
                    winner_idx = i
                    break
            
            winner = agent_outputs[winner_idx]
            
            return winner.response, evaluation, winner.confidence
            
        except Exception as e:
            best = max(agent_outputs, key=lambda x: x.confidence)
            return best.response, f"Voting failed: {str(e)}", best.confidence
    
    def _aggregate_weighted(
        self,
        query: str,
        agent_outputs: List[AgentOutput]
    ) -> Tuple[str, str, float]:
        """Weighted combination based on confidence scores"""
        
        # Similar to synthesis but explicitly weights by confidence
        valid_outputs = [o for o in agent_outputs if not o.metadata.get("error")]
        
        if not valid_outputs:
            return "No valid responses", "All agents failed", 0.0
        
        # Sort by confidence
        sorted_outputs = sorted(valid_outputs, key=lambda x: x.confidence, reverse=True)
        
        # Give more weight to high-confidence responses
        weighted_text = "\n\n".join([
            f"Agent {i+1} (High Confidence - Weight: {output.confidence:.2f}):\n{output.response}"
            if output.confidence > 0.7 else
            f"Agent {i+1} (Moderate Confidence - Weight: {output.confidence:.2f}):\n{output.response}"
            for i, output in enumerate(sorted_outputs)
        ])
        
        weighted_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are combining multiple responses weighted by their confidence scores.

Give more importance to high-confidence responses but don't ignore lower-confidence ones
if they contain unique valuable insights.

Produce a balanced synthesis."""),
            ("user", """Question: {query}

Weighted Responses:
{responses}

Produce weighted synthesis:""")
        ])
        
        chain = weighted_prompt | self.aggregator | StrOutputParser()
        
        final_response = chain.invoke({
            "query": query,
            "responses": weighted_text
        })
        
        # Confidence is weighted average favoring high-confidence responses
        total_weight = sum(o.confidence ** 2 for o in sorted_outputs)
        weighted_confidence = sum(
            (o.confidence ** 2) * o.confidence for o in sorted_outputs
        ) / total_weight if total_weight > 0 else 0.5
        
        reasoning = "Weighted synthesis favoring high-confidence responses"
        
        return final_response, reasoning, weighted_confidence
    
    def _aggregate_sequential(
        self,
        query: str,
        agent_outputs: List[AgentOutput]
    ) -> Tuple[str, str, float]:
        """Sequential refinement - each response builds on previous"""
        
        # Start with first response
        current_response = agent_outputs[0].response
        
        for i, output in enumerate(agent_outputs[1:], 1):
            if output.metadata.get("error"):
                continue
            
            refine_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are refining an answer by incorporating new insights.

Current answer:
{current}

New perspective:
{new}

Produce an improved, refined answer that combines both:"""),
                ("user", "Refined answer:")
            ])
            
            chain = refine_prompt | self.aggregator | StrOutputParser()
            
            current_response = chain.invoke({
                "current": current_response,
                "new": output.response
            })
        
        avg_confidence = sum(o.confidence for o in agent_outputs) / len(agent_outputs)
        reasoning = f"Sequential refinement through {len(agent_outputs)} agents"
        
        return current_response, reasoning, avg_confidence
    
    def execute(self, query: str) -> MoAResult:
        """Execute Mixture of Agents with multiple layers"""
        
        start_time = time.time()
        
        all_agent_outputs = []
        current_context = None
        
        # Execute layers
        for layer in range(self.num_layers):
            layer_outputs = []
            
            # Generate responses from all proposers
            for i, proposer in enumerate(self.proposers):
                agent_id = f"layer{layer+1}_agent{i+1}"
                
                output = self._generate_proposer_response(
                    agent_id,
                    proposer,
                    query,
                    context=current_context
                )
                
                layer_outputs.append(output)
                all_agent_outputs.append(output)
            
            # Prepare context for next layer
            if layer < self.num_layers - 1:
                valid_responses = [
                    o.response for o in layer_outputs
                    if not o.metadata.get("error")
                ]
                current_context = "\n\n".join(valid_responses[:2])  # Top 2 responses
        
        # Aggregate final responses
        final_layer_outputs = all_agent_outputs[-self.num_proposers:]
        
        if self.aggregation_strategy == AggregationStrategy.SYNTHESIS:
            final_response, reasoning, confidence = self._aggregate_synthesis(
                query, final_layer_outputs
            )
        elif self.aggregation_strategy == AggregationStrategy.VOTING:
            final_response, reasoning, confidence = self._aggregate_voting(
                query, final_layer_outputs
            )
        elif self.aggregation_strategy == AggregationStrategy.WEIGHTED:
            final_response, reasoning, confidence = self._aggregate_weighted(
                query, final_layer_outputs
            )
        elif self.aggregation_strategy == AggregationStrategy.SEQUENTIAL:
            final_response, reasoning, confidence = self._aggregate_sequential(
                query, final_layer_outputs
            )
        else:  # BEST_OF
            best = max(final_layer_outputs, key=lambda x: x.confidence)
            final_response = best.response
            reasoning = f"Selected best individual response (confidence: {best.confidence:.2f})"
            confidence = best.confidence
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return MoAResult(
            query=query,
            final_response=final_response,
            aggregation_strategy=self.aggregation_strategy,
            agent_outputs=all_agent_outputs,
            aggregator_reasoning=reasoning,
            confidence_score=confidence,
            execution_time_ms=execution_time_ms,
            layer_count=self.num_layers
        )


def demonstrate_mixture_of_agents():
    """Demonstrate Mixture of Agents pattern"""
    
    print("=" * 80)
    print("PATTERN 053: MIXTURE OF AGENTS (MoA) DEMONSTRATION")
    print("=" * 80)
    print("\nCombining multiple specialized agents for superior results\n")
    
    # Test 1: Basic MoA with synthesis
    print("\n" + "=" * 80)
    print("TEST 1: Basic MoA with Synthesis Aggregation")
    print("=" * 80)
    
    moa1 = MixtureOfAgents(
        num_proposers=3,
        aggregation_strategy=AggregationStrategy.SYNTHESIS,
        use_layers=False
    )
    
    query1 = "What are the pros and cons of remote work?"
    
    print(f"\nQuery: {query1}")
    print(f"Number of Proposer Agents: {moa1.num_proposers}")
    print(f"Aggregation Strategy: {moa1.aggregation_strategy.value}")
    
    result1 = moa1.execute(query1)
    
    print(f"\n📊 Individual Agent Responses:")
    for i, output in enumerate(result1.agent_outputs, 1):
        print(f"\n  Agent {i} (Confidence: {output.confidence:.2f}):")
        print(f"    {output.response[:120]}...")
    
    print(f"\n✨ Final Synthesized Response:")
    print(f"  {result1.final_response[:300]}...")
    print(f"\n  Aggregator Reasoning: {result1.aggregator_reasoning}")
    print(f"  Final Confidence: {result1.confidence_score:.2f}")
    print(f"  Execution Time: {result1.execution_time_ms:.1f}ms")
    
    # Test 2: Layered MoA
    print("\n" + "=" * 80)
    print("TEST 2: Layered MoA (Iterative Refinement)")
    print("=" * 80)
    
    moa2 = MixtureOfAgents(
        num_proposers=3,
        aggregation_strategy=AggregationStrategy.SYNTHESIS,
        use_layers=True,
        num_layers=2
    )
    
    query2 = "Explain quantum computing in simple terms"
    
    print(f"\nQuery: {query2}")
    print(f"Architecture: {moa2.num_layers} layers with {moa2.num_proposers} agents per layer")
    
    result2 = moa2.execute(query2)
    
    print(f"\n📊 Layer-by-Layer Generation:")
    layer1_agents = result2.agent_outputs[:moa2.num_proposers]
    layer2_agents = result2.agent_outputs[moa2.num_proposers:]
    
    print(f"\n  Layer 1 (Initial Responses):")
    for i, output in enumerate(layer1_agents, 1):
        print(f"    Agent {i}: {output.response[:80]}...")
    
    if layer2_agents:
        print(f"\n  Layer 2 (Refined with Context):")
        for i, output in enumerate(layer2_agents, 1):
            print(f"    Agent {i}: {output.response[:80]}...")
    
    print(f"\n✨ Final Layered Response:")
    print(f"  {result2.final_response[:300]}...")
    print(f"  Confidence: {result2.confidence_score:.2f}")
    
    # Test 3: Different aggregation strategies
    print("\n" + "=" * 80)
    print("TEST 3: Comparing Aggregation Strategies")
    print("=" * 80)
    
    query3 = "What's the best way to learn programming?"
    
    strategies = [
        AggregationStrategy.SYNTHESIS,
        AggregationStrategy.VOTING,
        AggregationStrategy.WEIGHTED
    ]
    
    print(f"\nQuery: {query3}")
    print(f"\nTesting different aggregation strategies...")
    
    results = []
    
    for strategy in strategies:
        moa = MixtureOfAgents(
            num_proposers=3,
            aggregation_strategy=strategy,
            use_layers=False
        )
        
        result = moa.execute(query3)
        results.append((strategy, result))
        
        print(f"\n  {strategy.value.upper()}:")
        print(f"    Response: {result.final_response[:100]}...")
        print(f"    Confidence: {result.confidence_score:.2f}")
        print(f"    Reasoning: {result.aggregator_reasoning[:80]}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("MIXTURE OF AGENTS PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Diverse Perspectives: Multiple agents provide varied viewpoints
2. Error Reduction: Aggregation filters out individual mistakes
3. Robustness: Less sensitive to single agent failures
4. Quality Improvement: Combined output better than any individual
5. Expertise Combination: Leverage multiple specializations

Implementation Features:
1. Multiple proposer agents with diverse configurations
2. Aggregator agent for intelligent synthesis
3. Layered architecture for iterative refinement
4. Confidence-based weighting
5. Multiple aggregation strategies

Aggregation Strategies:
1. Synthesis: Combines best insights from all responses
2. Voting: Democratic selection of best response
3. Weighted: Confidence-based combination
4. Sequential: Iterative refinement chain
5. Best-of: Simple selection of highest quality

Layered Architecture:
- Layer 1: Initial independent responses
- Layer 2+: Responses informed by previous layer
- Each layer refines and improves outputs
- Context flows from layer to layer

Use Cases:
- Complex questions requiring multiple perspectives
- Creative tasks needing diverse ideas
- Technical problems with multiple solutions
- Research synthesis from various sources
- Decision-making with high stakes
- Content generation requiring variety

Best Practices:
1. Use 3-5 proposer agents (balance diversity vs cost)
2. Vary agent configurations for diversity
3. Match aggregation strategy to use case
4. Use layers for complex reasoning tasks
5. Monitor confidence scores for quality
6. A/B test different strategies
7. Cache common query results

Production Considerations:
- Parallel execution for latency (async)
- Cost vs quality tradeoffs
- Model diversity (different providers)
- Caching for repeated queries
- Load balancing across agents
- Timeout handling
- Fallback strategies

Comparison with Related Patterns:
- vs. Ensemble: Synthesis vs simple voting
- vs. Redundancy: Quality improvement vs fault tolerance
- vs. Multi-Agent Debate: Aggregation vs deliberation
- vs. Leader-Follower: Parallel vs hierarchical

The Mixture of Agents pattern is powerful for tasks requiring high quality
outputs that benefit from multiple perspectives and expert combination.
""")


if __name__ == "__main__":
    demonstrate_mixture_of_agents()
