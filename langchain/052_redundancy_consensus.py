"""
Pattern 052: Redundancy & Consensus

Description:
    The Redundancy & Consensus pattern executes the same operation across multiple
    independent agents or models, then uses voting mechanisms to reach consensus.
    This pattern provides fault tolerance, error reduction, and improved reliability
    through N-version programming and majority voting.

Components:
    1. Redundant Executors: Multiple independent agents/models
    2. Voting Mechanism: Majority vote, weighted vote, ranked choice
    3. Consensus Builder: Aggregates results to reach agreement
    4. Disagreement Resolver: Handles cases with no clear consensus
    5. Quality Scorer: Evaluates individual responses

Use Cases:
    - Critical decisions requiring high reliability
    - Error-prone tasks needing validation
    - Safety-critical applications
    - Medical diagnosis support
    - Financial predictions
    - Content moderation at scale

LangChain Implementation:
    Uses multiple LLM instances or models to generate responses independently,
    then applies voting algorithms to reach consensus on the final answer.
"""

import os
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import Counter

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class VotingStrategy(Enum):
    """Voting strategies for consensus"""
    MAJORITY = "majority"  # Simple majority wins
    UNANIMOUS = "unanimous"  # All must agree
    WEIGHTED = "weighted"  # Weighted by confidence/quality
    RANKED_CHOICE = "ranked_choice"  # Ranked voting
    THRESHOLD = "threshold"  # Must meet minimum agreement threshold


class AgreementLevel(Enum):
    """Level of agreement among voters"""
    UNANIMOUS = "unanimous"  # 100% agreement
    STRONG_CONSENSUS = "strong_consensus"  # 80%+
    CONSENSUS = "consensus"  # 60-79%
    MAJORITY = "majority"  # 51-59%
    SPLIT = "split"  # No clear majority
    DISAGREEMENT = "disagreement"  # High conflict


@dataclass
class AgentResponse:
    """Response from a single agent"""
    agent_id: str
    response: str
    confidence: float  # 0.0-1.0
    execution_time_ms: float
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "response": self.response[:100] + "..." if len(self.response) > 100 else self.response,
            "confidence": f"{self.confidence:.2f}",
            "execution_time_ms": f"{self.execution_time_ms:.1f}",
            "model": self.model_name
        }


@dataclass
class VoteResult:
    """Result of voting process"""
    winner: str
    vote_count: int
    total_votes: int
    agreement_level: AgreementLevel
    confidence: float
    all_responses: List[str]
    vote_distribution: Dict[str, int]
    
    @property
    def agreement_percentage(self) -> float:
        return (self.vote_count / max(1, self.total_votes)) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "winner": self.winner[:100] + "..." if len(self.winner) > 100 else self.winner,
            "votes": f"{self.vote_count}/{self.total_votes}",
            "agreement": f"{self.agreement_percentage:.1f}%",
            "agreement_level": self.agreement_level.value,
            "confidence": f"{self.confidence:.2f}",
            "unique_responses": len(set(self.all_responses)),
            "vote_distribution": self.vote_distribution
        }


@dataclass
class ConsensusResult:
    """Result from consensus-based execution"""
    query: str
    final_answer: str
    voting_strategy: VotingStrategy
    vote_result: VoteResult
    individual_responses: List[AgentResponse]
    execution_time_ms: float
    redundancy_factor: int
    disagreements: List[str]
    
    @property
    def reliability_score(self) -> float:
        """Calculate reliability based on agreement and confidence"""
        agreement_factor = self.vote_result.agreement_percentage / 100
        confidence_factor = self.vote_result.confidence
        redundancy_factor = min(1.0, self.redundancy_factor / 5)  # Cap at 5
        
        return (agreement_factor * 0.5 + confidence_factor * 0.3 + redundancy_factor * 0.2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query[:80] + "..." if len(self.query) > 80 else self.query,
            "final_answer": self.final_answer[:150] + "..." if len(self.final_answer) > 150 else self.final_answer,
            "voting_strategy": self.voting_strategy.value,
            "vote_result": self.vote_result.to_dict(),
            "execution_time_ms": f"{self.execution_time_ms:.1f}",
            "redundancy_factor": self.redundancy_factor,
            "reliability_score": f"{self.reliability_score:.2f}",
            "had_disagreements": len(self.disagreements) > 0
        }


class ConsensusAgent:
    """
    Agent that uses redundancy and voting for reliable decisions.
    
    This implementation provides:
    1. Multiple independent executions (N-version programming)
    2. Various voting strategies (majority, weighted, unanimous)
    3. Consensus building with conflict resolution
    4. Reliability scoring based on agreement
    5. Disagreement detection and handling
    """
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        redundancy_factor: int = 3,
        voting_strategy: VotingStrategy = VotingStrategy.MAJORITY,
        temperature: float = 0.7,
        min_confidence_threshold: float = 0.6
    ):
        self.model_name = model_name
        self.redundancy_factor = redundancy_factor
        self.voting_strategy = voting_strategy
        self.temperature = temperature
        self.min_confidence_threshold = min_confidence_threshold
        
        # Create multiple independent agents
        self.agents = [
            ChatOpenAI(model_name=model_name, temperature=temperature)
            for _ in range(redundancy_factor)
        ]
    
    def _execute_single_agent(
        self,
        agent_id: str,
        llm: ChatOpenAI,
        query: str
    ) -> AgentResponse:
        """Execute query on single agent"""
        import time
        
        start_time = time.time()
        
        try:
            response = llm.invoke(query)
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Estimate confidence (simplified - production would use actual model confidence)
            confidence = 0.7 + (0.3 * (len(response.content) / max(1, len(query))))
            confidence = min(1.0, confidence)
            
            return AgentResponse(
                agent_id=agent_id,
                response=response.content,
                confidence=confidence,
                execution_time_ms=execution_time_ms,
                model_name=self.model_name
            )
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            return AgentResponse(
                agent_id=agent_id,
                response=f"ERROR: {str(e)}",
                confidence=0.0,
                execution_time_ms=execution_time_ms,
                model_name=self.model_name,
                metadata={"error": True}
            )
    
    def _normalize_response(self, response: str) -> str:
        """Normalize response for comparison"""
        # Remove extra whitespace, lowercase, strip punctuation
        normalized = response.lower().strip()
        normalized = " ".join(normalized.split())
        return normalized
    
    def _calculate_similarity(self, resp1: str, resp2: str) -> float:
        """Calculate similarity between two responses"""
        # Simple word overlap similarity
        words1 = set(self._normalize_response(resp1).split())
        words2 = set(self._normalize_response(resp2).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _majority_vote(
        self,
        responses: List[AgentResponse],
        similarity_threshold: float = 0.7
    ) -> VoteResult:
        """Majority voting with similarity grouping"""
        # Group similar responses
        groups: List[List[AgentResponse]] = []
        
        for response in responses:
            added_to_group = False
            
            for group in groups:
                # Check similarity with group representative
                similarity = self._calculate_similarity(
                    response.response,
                    group[0].response
                )
                
                if similarity >= similarity_threshold:
                    group.append(response)
                    added_to_group = True
                    break
            
            if not added_to_group:
                groups.append([response])
        
        # Find largest group
        largest_group = max(groups, key=len)
        vote_count = len(largest_group)
        total_votes = len(responses)
        
        # Select best response from winning group (highest confidence)
        winner = max(largest_group, key=lambda r: r.confidence)
        
        # Calculate average confidence of winning group
        avg_confidence = sum(r.confidence for r in largest_group) / len(largest_group)
        
        # Determine agreement level
        agreement_pct = (vote_count / total_votes) * 100
        
        if agreement_pct == 100:
            agreement_level = AgreementLevel.UNANIMOUS
        elif agreement_pct >= 80:
            agreement_level = AgreementLevel.STRONG_CONSENSUS
        elif agreement_pct >= 60:
            agreement_level = AgreementLevel.CONSENSUS
        elif agreement_pct >= 51:
            agreement_level = AgreementLevel.MAJORITY
        elif len(groups) == len(responses):
            agreement_level = AgreementLevel.DISAGREEMENT
        else:
            agreement_level = AgreementLevel.SPLIT
        
        # Create vote distribution
        vote_distribution = {}
        for i, group in enumerate(groups, 1):
            key = f"response_group_{i}"
            vote_distribution[key] = len(group)
        
        return VoteResult(
            winner=winner.response,
            vote_count=vote_count,
            total_votes=total_votes,
            agreement_level=agreement_level,
            confidence=avg_confidence,
            all_responses=[r.response for r in responses],
            vote_distribution=vote_distribution
        )
    
    def _weighted_vote(self, responses: List[AgentResponse]) -> VoteResult:
        """Weighted voting based on confidence scores"""
        # Weight responses by confidence
        weighted_responses = []
        
        for response in responses:
            # Weight = confidence score
            weight = response.confidence
            weighted_responses.append((response, weight))
        
        # Find response with highest total weight (considering similarity)
        best_response = None
        best_weight = 0.0
        
        for response, base_weight in weighted_responses:
            total_weight = base_weight
            
            # Add weights from similar responses
            for other_response, other_weight in weighted_responses:
                if response.agent_id != other_response.agent_id:
                    similarity = self._calculate_similarity(
                        response.response,
                        other_response.response
                    )
                    if similarity >= 0.7:
                        total_weight += other_weight * similarity
            
            if total_weight > best_weight:
                best_weight = total_weight
                best_response = response
        
        # Calculate agreement level based on weight distribution
        total_weight = sum(w for _, w in weighted_responses)
        agreement_pct = (best_weight / total_weight) * 100
        
        if agreement_pct >= 90:
            agreement_level = AgreementLevel.STRONG_CONSENSUS
        elif agreement_pct >= 70:
            agreement_level = AgreementLevel.CONSENSUS
        elif agreement_pct >= 50:
            agreement_level = AgreementLevel.MAJORITY
        else:
            agreement_level = AgreementLevel.SPLIT
        
        return VoteResult(
            winner=best_response.response,
            vote_count=1,  # Weighted doesn't use simple counts
            total_votes=len(responses),
            agreement_level=agreement_level,
            confidence=best_response.confidence,
            all_responses=[r.response for r in responses],
            vote_distribution={"weighted_winner": 1}
        )
    
    def _unanimous_vote(self, responses: List[AgentResponse]) -> VoteResult:
        """Unanimous voting - all must agree"""
        # Check if all responses are similar
        if not responses:
            raise ValueError("No responses to vote on")
        
        reference = responses[0].response
        all_agree = all(
            self._calculate_similarity(r.response, reference) >= 0.9
            for r in responses[1:]
        )
        
        if all_agree:
            avg_confidence = sum(r.confidence for r in responses) / len(responses)
            
            return VoteResult(
                winner=reference,
                vote_count=len(responses),
                total_votes=len(responses),
                agreement_level=AgreementLevel.UNANIMOUS,
                confidence=avg_confidence,
                all_responses=[r.response for r in responses],
                vote_distribution={"unanimous": len(responses)}
            )
        else:
            # No consensus - return best individual response
            best = max(responses, key=lambda r: r.confidence)
            
            return VoteResult(
                winner=best.response,
                vote_count=1,
                total_votes=len(responses),
                agreement_level=AgreementLevel.DISAGREEMENT,
                confidence=best.confidence,
                all_responses=[r.response for r in responses],
                vote_distribution={"no_consensus": len(responses)}
            )
    
    def _apply_voting_strategy(
        self,
        responses: List[AgentResponse]
    ) -> VoteResult:
        """Apply configured voting strategy"""
        if self.voting_strategy == VotingStrategy.MAJORITY:
            return self._majority_vote(responses)
        elif self.voting_strategy == VotingStrategy.WEIGHTED:
            return self._weighted_vote(responses)
        elif self.voting_strategy == VotingStrategy.UNANIMOUS:
            return self._unanimous_vote(responses)
        else:
            # Default to majority
            return self._majority_vote(responses)
    
    def execute_with_consensus(self, query: str) -> ConsensusResult:
        """Execute query with redundancy and consensus"""
        import time
        
        start_time = time.time()
        
        # Execute across all redundant agents
        responses = []
        for i, agent in enumerate(self.agents):
            agent_id = f"agent_{i+1}"
            response = self._execute_single_agent(agent_id, agent, query)
            responses.append(response)
        
        # Filter out errors
        valid_responses = [r for r in responses if not r.metadata.get("error")]
        
        if not valid_responses:
            # All failed - return error consensus
            execution_time_ms = (time.time() - start_time) * 1000
            
            return ConsensusResult(
                query=query,
                final_answer="ERROR: All agents failed to respond",
                voting_strategy=self.voting_strategy,
                vote_result=VoteResult(
                    winner="ERROR",
                    vote_count=0,
                    total_votes=len(responses),
                    agreement_level=AgreementLevel.DISAGREEMENT,
                    confidence=0.0,
                    all_responses=[r.response for r in responses],
                    vote_distribution={"errors": len(responses)}
                ),
                individual_responses=responses,
                execution_time_ms=execution_time_ms,
                redundancy_factor=self.redundancy_factor,
                disagreements=[]
            )
        
        # Apply voting strategy
        vote_result = self._apply_voting_strategy(valid_responses)
        
        # Identify disagreements
        disagreements = []
        winner_normalized = self._normalize_response(vote_result.winner)
        
        for response in valid_responses:
            similarity = self._calculate_similarity(response.response, vote_result.winner)
            if similarity < 0.5:  # Significant disagreement
                disagreements.append(response.response[:100])
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return ConsensusResult(
            query=query,
            final_answer=vote_result.winner,
            voting_strategy=self.voting_strategy,
            vote_result=vote_result,
            individual_responses=responses,
            execution_time_ms=execution_time_ms,
            redundancy_factor=self.redundancy_factor,
            disagreements=disagreements
        )


def demonstrate_redundancy_consensus():
    """Demonstrate redundancy and consensus pattern"""
    
    print("=" * 80)
    print("PATTERN 052: REDUNDANCY & CONSENSUS DEMONSTRATION")
    print("=" * 80)
    print("\nDemonstrating fault-tolerant decision making through redundancy\n")
    
    # Test 1: Majority voting
    print("\n" + "=" * 80)
    print("TEST 1: Majority Voting with Redundancy")
    print("=" * 80)
    
    agent1 = ConsensusAgent(
        redundancy_factor=5,
        voting_strategy=VotingStrategy.MAJORITY,
        temperature=0.3  # Low temperature for more consistent responses
    )
    
    query1 = "What is the capital of France?"
    
    print(f"\nQuery: {query1}")
    print(f"Redundancy: {agent1.redundancy_factor} agents")
    print(f"Strategy: {agent1.voting_strategy.value}")
    
    result1 = agent1.execute_with_consensus(query1)
    
    print(f"\n✓ Consensus reached:")
    print(f"  Final Answer: {result1.final_answer}")
    print(f"  Agreement: {result1.vote_result.agreement_percentage:.1f}% ({result1.vote_result.agreement_level.value})")
    print(f"  Confidence: {result1.vote_result.confidence:.2f}")
    print(f"  Reliability Score: {result1.reliability_score:.2f}")
    print(f"  Execution Time: {result1.execution_time_ms:.1f}ms")
    
    print(f"\n  Individual Responses:")
    for i, resp in enumerate(result1.individual_responses, 1):
        print(f"    Agent {i}: {resp.response[:60]}... (confidence: {resp.confidence:.2f})")
    
    # Test 2: Weighted voting
    print("\n" + "=" * 80)
    print("TEST 2: Weighted Voting by Confidence")
    print("=" * 80)
    
    agent2 = ConsensusAgent(
        redundancy_factor=3,
        voting_strategy=VotingStrategy.WEIGHTED,
        temperature=0.5
    )
    
    query2 = "Is water wet?"
    
    print(f"\nQuery: {query2}")
    print(f"Strategy: {agent2.voting_strategy.value}")
    
    result2 = agent2.execute_with_consensus(query2)
    
    print(f"\n✓ Weighted consensus:")
    print(f"  Final Answer: {result2.final_answer[:120]}...")
    print(f"  Agreement Level: {result2.vote_result.agreement_level.value}")
    print(f"  Confidence: {result2.vote_result.confidence:.2f}")
    
    # Test 3: Unanimous voting (strict)
    print("\n" + "=" * 80)
    print("TEST 3: Unanimous Voting (Strict Consensus)")
    print("=" * 80)
    
    agent3 = ConsensusAgent(
        redundancy_factor=3,
        voting_strategy=VotingStrategy.UNANIMOUS,
        temperature=0.1  # Very low for consistency
    )
    
    query3 = "What is 2+2?"
    
    print(f"\nQuery: {query3}")
    print(f"Strategy: {agent3.voting_strategy.value}")
    
    result3 = agent3.execute_with_consensus(query3)
    
    print(f"\n✓ Unanimous check:")
    print(f"  Final Answer: {result3.final_answer}")
    print(f"  Agreement: {result3.vote_result.agreement_level.value}")
    print(f"  All Agreed: {result3.vote_result.agreement_level == AgreementLevel.UNANIMOUS}")
    
    # Test 4: Disagreement handling
    print("\n" + "=" * 80)
    print("TEST 4: Handling Disagreements")
    print("=" * 80)
    
    agent4 = ConsensusAgent(
        redundancy_factor=5,
        voting_strategy=VotingStrategy.MAJORITY,
        temperature=0.9  # High temperature for diverse responses
    )
    
    query4 = "What's the best programming language?"
    
    print(f"\nQuery: {query4}")
    print("(Intentionally subjective to generate disagreements)")
    
    result4 = agent4.execute_with_consensus(query4)
    
    print(f"\n📊 Consensus Analysis:")
    print(f"  Final Answer: {result4.final_answer[:100]}...")
    print(f"  Agreement: {result4.vote_result.agreement_percentage:.1f}%")
    print(f"  Agreement Level: {result4.vote_result.agreement_level.value}")
    print(f"  Unique Responses: {len(set(result4.vote_result.all_responses))}")
    print(f"  Disagreements Found: {len(result4.disagreements)}")
    
    if result4.disagreements:
        print(f"\n  Minority Opinions:")
        for i, disagreement in enumerate(result4.disagreements[:2], 1):
            print(f"    {i}. {disagreement}...")
    
    print(f"\n  Vote Distribution:")
    for group, count in result4.vote_result.vote_distribution.items():
        print(f"    {group}: {count} votes")
    
    # Test 5: Reliability comparison
    print("\n" + "=" * 80)
    print("TEST 5: Reliability Score Comparison")
    print("=" * 80)
    
    test_queries = [
        "What is the speed of light?",
        "Who wrote Hamlet?",
        "What's the meaning of life?"
    ]
    
    print(f"\nComparing reliability across different query types...")
    
    reliability_scores = []
    
    for query in test_queries:
        result = agent1.execute_with_consensus(query)
        reliability_scores.append((query, result.reliability_score, result.vote_result.agreement_level))
        
        print(f"\nQuery: {query}")
        print(f"  Reliability: {result.reliability_score:.2f}")
        print(f"  Agreement: {result.vote_result.agreement_level.value}")
        print(f"  Confidence: {result.vote_result.confidence:.2f}")
    
    # Show ranking
    print(f"\n📊 Reliability Ranking:")
    sorted_scores = sorted(reliability_scores, key=lambda x: x[1], reverse=True)
    
    for i, (query, score, agreement) in enumerate(sorted_scores, 1):
        print(f"  {i}. {query}")
        print(f"     Score: {score:.2f}, Agreement: {agreement.value}")
    
    # Summary
    print("\n" + "=" * 80)
    print("REDUNDANCY & CONSENSUS PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Fault Tolerance: Multiple agents provide backup if one fails
2. Error Reduction: Voting filters out individual mistakes
3. Reliability: Higher confidence through consensus
4. Robustness: Less sensitive to individual model quirks
5. Quality Assurance: Cross-validation of responses

Implementation Features:
1. N-version programming with multiple agents
2. Multiple voting strategies (majority, weighted, unanimous)
3. Similarity-based response grouping
4. Disagreement detection and tracking
5. Reliability scoring based on agreement and confidence
6. Configurable redundancy factor

Voting Strategies:
1. Majority Vote: Most common response wins
2. Weighted Vote: Responses weighted by confidence
3. Unanimous: All agents must agree (strictest)
4. Ranked Choice: Ordered preference voting
5. Threshold: Minimum agreement percentage required

Agreement Levels:
- Unanimous (100%): Perfect agreement
- Strong Consensus (80%+): Very high agreement
- Consensus (60-79%): Good agreement
- Majority (51-59%): Simple majority
- Split: No clear majority
- Disagreement: High conflict

Use Cases:
- Medical diagnosis support systems
- Financial predictions and recommendations
- Safety-critical decision making
- Content moderation at scale
- Fraud detection systems
- Legal document analysis
- Scientific research validation

Best Practices:
1. Use redundancy factor 3-7 (balance cost vs reliability)
2. Match voting strategy to use case criticality
3. Monitor disagreement patterns for insights
4. Consider computational cost vs reliability gain
5. Use lower temperature for factual queries
6. Track reliability scores over time
7. Escalate high-disagreement cases to humans
8. Document when consensus failed

Production Considerations:
- Parallel execution for latency (async processing)
- Cost vs reliability tradeoffs
- Caching for repeated queries
- Timeout handling for slow agents
- Load balancing across models
- Model diversity (different providers)
- Fallback strategies for no consensus
- A/B testing of voting strategies

Comparison with Related Patterns:
- vs. Ensemble: Voting vs averaging predictions
- vs. Self-Consistency: Multiple executions vs one with sampling
- vs. Multi-Agent Debate: Consensus vs deliberation
- vs. Fallback: Redundancy vs sequential alternatives

The Redundancy & Consensus pattern is essential for high-reliability AI systems
where decisions must be trustworthy and errors minimized through cross-validation
and democratic decision-making among multiple independent agents.
""")


if __name__ == "__main__":
    demonstrate_redundancy_consensus()
