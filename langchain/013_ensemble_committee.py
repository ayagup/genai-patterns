"""
Pattern 013: Ensemble/Committee

Description:
    The Ensemble/Committee pattern uses multiple agents working independently on the same
    task, then aggregates their outputs to reach a final decision. This pattern leverages
    diversity to improve robustness, reduce errors, and provide more reliable results.

Components:
    - Multiple independent agents (diverse models, prompts, or strategies)
    - Aggregation mechanism (voting, averaging, weighted combination)
    - Result synthesis and confidence scoring

Use Cases:
    - Classification tasks requiring high accuracy
    - Prediction problems where uncertainty quantification is important
    - Reducing bias through diverse perspectives
    - Quality assurance through redundancy

LangChain Implementation:
    Uses multiple ChatOpenAI instances with different configurations, implements
    various aggregation strategies (majority voting, weighted voting, consensus),
    and provides confidence metrics based on agreement levels.

Key Features:
    - Diverse agent configurations (temperature, models, prompts)
    - Multiple aggregation strategies
    - Confidence scoring based on agreement
    - Detailed breakdown of individual votes
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import Counter
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class AggregationStrategy(Enum):
    """Aggregation strategies for combining agent outputs."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    CONSENSUS = "consensus"
    AVERAGING = "averaging"


@dataclass
class AgentVote:
    """Represents a single agent's vote/output."""
    agent_id: str
    output: str
    confidence: float
    reasoning: Optional[str] = None


@dataclass
class EnsembleResult:
    """Result from ensemble aggregation."""
    final_output: str
    confidence: float
    individual_votes: List[AgentVote]
    agreement_score: float
    strategy_used: str


class EnsembleAgent:
    """
    Ensemble agent that combines outputs from multiple independent agents.
    
    This implementation creates diverse agents through different configurations
    (temperature, system prompts) and provides multiple aggregation strategies.
    """
    
    def __init__(
        self,
        num_agents: int = 3,
        base_model: str = "gpt-3.5-turbo",
        temperatures: Optional[List[float]] = None
    ):
        """
        Initialize ensemble with multiple agents.
        
        Args:
            num_agents: Number of agents in ensemble
            base_model: Base model for all agents
            temperatures: List of temperatures for diversity (default: [0.3, 0.7, 1.0])
        """
        self.num_agents = num_agents
        self.base_model = base_model
        
        # Create diverse temperature settings if not provided
        if temperatures is None:
            temperatures = [0.3, 0.5, 0.7, 0.9, 1.0][:num_agents]
        
        # Create diverse agents with different temperatures
        self.agents = []
        for i, temp in enumerate(temperatures[:num_agents]):
            agent = ChatOpenAI(
                model=base_model,
                temperature=temp,
                model_kwargs={"seed": i * 42}  # Different seeds for diversity
            )
            self.agents.append(agent)
        
        # Create a meta-agent for confidence estimation
        self.meta_agent = ChatOpenAI(model=base_model, temperature=0.2)
    
    def _estimate_confidence(
        self,
        question: str,
        answer: str,
        reasoning: str
    ) -> float:
        """Estimate confidence score for an individual answer."""
        confidence_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at evaluating answer quality and confidence."),
            ("user", """Given this question and answer, estimate the confidence level (0.0-1.0).
Consider clarity, completeness, and certainty in the reasoning.

Question: {question}

Answer: {answer}

Reasoning: {reasoning}

Provide only a number between 0.0 and 1.0.""")
        ])
        
        chain = confidence_prompt | self.meta_agent | StrOutputParser()
        
        try:
            confidence_str = chain.invoke({
                "question": question,
                "answer": answer,
                "reasoning": reasoning
            })
            confidence = float(confidence_str.strip())
            return max(0.0, min(1.0, confidence))
        except:
            return 0.5  # Default moderate confidence
    
    def query_agents(
        self,
        question: str,
        system_prompt: Optional[str] = None
    ) -> List[AgentVote]:
        """
        Query all agents independently and collect their votes.
        
        Args:
            question: Question to ask all agents
            system_prompt: Optional system prompt (default: general assistant)
            
        Returns:
            List of agent votes with outputs and confidence scores
        """
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant. Provide clear, accurate answers."
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\nProvide your answer followed by your reasoning."),
            ("user", "{question}")
        ])
        
        votes = []
        
        for i, agent in enumerate(self.agents):
            chain = prompt_template | agent | StrOutputParser()
            
            try:
                response = chain.invoke({"question": question})
                
                # Split response into answer and reasoning
                parts = response.split("\n", 1)
                answer = parts[0].strip()
                reasoning = parts[1].strip() if len(parts) > 1 else ""
                
                # Estimate confidence
                confidence = self._estimate_confidence(question, answer, reasoning)
                
                vote = AgentVote(
                    agent_id=f"agent_{i}",
                    output=answer,
                    confidence=confidence,
                    reasoning=reasoning
                )
                votes.append(vote)
                
            except Exception as e:
                print(f"Agent {i} error: {e}")
                # Add a low-confidence fallback vote
                votes.append(AgentVote(
                    agent_id=f"agent_{i}",
                    output="Error generating response",
                    confidence=0.0,
                    reasoning=f"Error: {str(e)}"
                ))
        
        return votes
    
    def aggregate_majority_vote(
        self,
        votes: List[AgentVote]
    ) -> Tuple[str, float]:
        """
        Aggregate using simple majority voting.
        
        Returns:
            (winning answer, agreement score)
        """
        # Count votes for each unique answer
        answer_counts = Counter(vote.output for vote in votes)
        most_common = answer_counts.most_common(1)[0]
        winning_answer = most_common[0]
        vote_count = most_common[1]
        
        agreement_score = vote_count / len(votes)
        
        return winning_answer, agreement_score
    
    def aggregate_weighted_vote(
        self,
        votes: List[AgentVote]
    ) -> Tuple[str, float]:
        """
        Aggregate using confidence-weighted voting.
        
        Returns:
            (winning answer, weighted agreement score)
        """
        # Weight each answer by confidence
        answer_weights: Dict[str, float] = {}
        total_confidence = sum(vote.confidence for vote in votes)
        
        for vote in votes:
            if vote.output in answer_weights:
                answer_weights[vote.output] += vote.confidence
            else:
                answer_weights[vote.output] = vote.confidence
        
        # Find answer with highest weighted vote
        winning_answer = max(answer_weights.items(), key=lambda x: x[1])[0]
        winning_weight = answer_weights[winning_answer]
        
        # Calculate weighted agreement score
        agreement_score = winning_weight / total_confidence if total_confidence > 0 else 0.0
        
        return winning_answer, agreement_score
    
    def aggregate_consensus(
        self,
        votes: List[AgentVote],
        question: str
    ) -> Tuple[str, float]:
        """
        Generate consensus answer by synthesizing all votes.
        
        Returns:
            (consensus answer, confidence score)
        """
        # Compile all answers and reasoning
        vote_summary = "\n\n".join([
            f"Agent {vote.agent_id} (confidence: {vote.confidence:.2f}):\n"
            f"Answer: {vote.output}\n"
            f"Reasoning: {vote.reasoning}"
            for vote in votes
        ])
        
        consensus_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at synthesizing multiple perspectives into a consensus."),
            ("user", """Given the following question and multiple agent responses, create a 
consensus answer that incorporates the best elements from all responses.

Question: {question}

Agent Responses:
{vote_summary}

Provide a consensus answer that:
1. Incorporates accurate information from all agents
2. Resolves any contradictions
3. Provides a clear, unified response

Consensus Answer:""")
        ])
        
        chain = consensus_prompt | self.meta_agent | StrOutputParser()
        
        consensus_answer = chain.invoke({
            "question": question,
            "vote_summary": vote_summary
        })
        
        # Calculate confidence based on average agent confidence
        avg_confidence = sum(vote.confidence for vote in votes) / len(votes)
        
        return consensus_answer.strip(), avg_confidence
    
    def solve(
        self,
        question: str,
        strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_VOTE,
        system_prompt: Optional[str] = None
    ) -> EnsembleResult:
        """
        Solve a problem using the ensemble of agents.
        
        Args:
            question: Question to solve
            strategy: Aggregation strategy to use
            system_prompt: Optional custom system prompt
            
        Returns:
            EnsembleResult with final answer and metadata
        """
        # Get votes from all agents
        votes = self.query_agents(question, system_prompt)
        
        # Aggregate based on strategy
        if strategy == AggregationStrategy.MAJORITY_VOTE:
            final_output, agreement_score = self.aggregate_majority_vote(votes)
            
        elif strategy == AggregationStrategy.WEIGHTED_VOTE:
            final_output, agreement_score = self.aggregate_weighted_vote(votes)
            
        elif strategy == AggregationStrategy.CONSENSUS:
            final_output, agreement_score = self.aggregate_consensus(votes, question)
            
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        
        # Calculate overall confidence
        confidence = agreement_score
        
        return EnsembleResult(
            final_output=final_output,
            confidence=confidence,
            individual_votes=votes,
            agreement_score=agreement_score,
            strategy_used=strategy.value
        )


def demonstrate_ensemble():
    """Demonstrate the Ensemble/Committee pattern with various examples."""
    
    print("=" * 80)
    print("ENSEMBLE/COMMITTEE PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Create ensemble with 5 agents
    ensemble = EnsembleAgent(num_agents=5)
    
    # Test 1: Classification task
    print("\n" + "=" * 80)
    print("TEST 1: Sentiment Classification")
    print("=" * 80)
    
    sentiment_question = """Classify the sentiment of this review as Positive, Negative, or Neutral:

"The product works okay, but the customer service was terrible. Mixed feelings overall."

Your answer should be exactly one word: Positive, Negative, or Neutral."""
    
    result = ensemble.solve(
        sentiment_question,
        strategy=AggregationStrategy.WEIGHTED_VOTE
    )
    
    print(f"\nQuestion: Sentiment classification of mixed review")
    print(f"\nFinal Answer: {result.final_output}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Agreement Score: {result.agreement_score:.2f}")
    print(f"Strategy: {result.strategy_used}")
    
    print("\nIndividual Agent Votes:")
    for vote in result.individual_votes:
        print(f"  {vote.agent_id}: {vote.output} (confidence: {vote.confidence:.2f})")
    
    # Test 2: Factual question with consensus
    print("\n" + "=" * 80)
    print("TEST 2: Factual Question with Consensus")
    print("=" * 80)
    
    factual_question = "What are the main causes of climate change?"
    
    result = ensemble.solve(
        factual_question,
        strategy=AggregationStrategy.CONSENSUS
    )
    
    print(f"\nQuestion: {factual_question}")
    print(f"\nConsensus Answer:\n{result.final_output}")
    print(f"\nConfidence: {result.confidence:.2f}")
    print(f"Agreement Score: {result.agreement_score:.2f}")
    
    # Test 3: Multiple choice with majority vote
    print("\n" + "=" * 80)
    print("TEST 3: Multiple Choice with Majority Vote")
    print("=" * 80)
    
    mc_question = """Which of the following is the capital of Australia?
A) Sydney
B) Melbourne
C) Canberra
D) Brisbane

Answer with only the letter (A, B, C, or D)."""
    
    result = ensemble.solve(
        mc_question,
        strategy=AggregationStrategy.MAJORITY_VOTE
    )
    
    print(f"\nQuestion: Capital of Australia (multiple choice)")
    print(f"\nFinal Answer: {result.final_output}")
    print(f"Agreement Score: {result.agreement_score:.2f}")
    
    print("\nIndividual Agent Votes:")
    for vote in result.individual_votes:
        print(f"  {vote.agent_id}: {vote.output}")
    
    # Test 4: Comparison of strategies
    print("\n" + "=" * 80)
    print("TEST 4: Strategy Comparison")
    print("=" * 80)
    
    comparison_question = "Is artificial intelligence dangerous? Answer Yes or No, then explain."
    
    print(f"\nQuestion: {comparison_question}\n")
    
    for strategy in [AggregationStrategy.MAJORITY_VOTE, 
                     AggregationStrategy.WEIGHTED_VOTE,
                     AggregationStrategy.CONSENSUS]:
        
        result = ensemble.solve(comparison_question, strategy=strategy)
        
        print(f"\nStrategy: {strategy.value}")
        print(f"Answer: {result.final_output[:150]}...")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Agreement: {result.agreement_score:.2f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Ensemble/Committee pattern demonstrates several key benefits:

1. **Robustness**: Multiple agents reduce the impact of individual errors
2. **Diversity**: Different temperatures create varied perspectives
3. **Confidence Scoring**: Agreement levels indicate reliability
4. **Flexibility**: Multiple aggregation strategies for different use cases

Aggregation Strategies:
- **Majority Vote**: Simple, works well for discrete choices
- **Weighted Vote**: Considers confidence, better for uncertain domains
- **Consensus**: Synthesizes responses, best for complex questions

Use Cases:
- High-stakes classification tasks
- Fact-checking and validation
- Reducing model bias
- Quality assurance through redundancy

The pattern is particularly effective when:
- Individual agent accuracy is moderate but independent
- Diversity in approaches improves results
- Confidence/uncertainty quantification is important
- Explainability of decisions is needed
""")


if __name__ == "__main__":
    demonstrate_ensemble()
