"""
Pattern 52: Redundancy & Consensus
Description:
    Uses multiple redundant agents to verify outputs through consensus
    mechanisms, improving reliability and accuracy.
Use Cases:
    - Critical decision validation
    - Fault tolerance
    - Error detection
    - Quality assurance
Key Features:
    - Multiple verification methods
    - Byzantine fault tolerance
    - Consensus algorithms
    - Confidence scoring
Example:
    >>> consensus = ConsensusSystem()
    >>> consensus.add_verifier(agent1)
    >>> consensus.add_verifier(agent2)
    >>> result = consensus.reach_consensus(task)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set
from enum import Enum
import time
from collections import Counter, defaultdict
import statistics
class ConsensusMethod(Enum):
    """Methods for reaching consensus"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    UNANIMOUS = "unanimous"
    QUORUM = "quorum"
    BYZANTINE_FAULT_TOLERANT = "byzantine_fault_tolerant"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
class VerificationStatus(Enum):
    """Status of verification"""
    CONSENSUS_REACHED = "consensus_reached"
    NO_CONSENSUS = "no_consensus"
    INSUFFICIENT_VERIFIERS = "insufficient_verifiers"
    BYZANTINE_FAILURE = "byzantine_failure"
@dataclass
class VerifierConfig:
    """Configuration for a verifier agent"""
    verifier_id: str
    agent: Any
    weight: float = 1.0
    reliability_score: float = 1.0
    specialization: Optional[str] = None
@dataclass
class VerificationResult:
    """Result from a single verifier"""
    verifier_id: str
    output: Any
    confidence: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
@dataclass
class ConsensusResult:
    """Result of consensus process"""
    consensus_output: Any
    status: VerificationStatus
    agreement_level: float
    verifier_results: List[VerificationResult]
    consensus_method: ConsensusMethod
    total_verifiers: int
    agreeing_verifiers: int
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
class ConsensusSystem:
    """
    Redundancy and consensus system
    Features:
    - Multiple consensus methods
    - Byzantine fault tolerance
    - Confidence-based weighting
    - Verification tracking
    """
    def __init__(
        self,
        consensus_method: ConsensusMethod = ConsensusMethod.MAJORITY_VOTE,
        min_verifiers: int = 3,
        consensus_threshold: float = 0.66
    ):
        self.consensus_method = consensus_method
        self.min_verifiers = min_verifiers
        self.consensus_threshold = consensus_threshold
        self.verifiers: Dict[str, VerifierConfig] = {}
        self.verification_history: List[ConsensusResult] = []
    def add_verifier(
        self,
        agent: Any,
        verifier_id: str,
        weight: float = 1.0,
        reliability_score: float = 1.0,
        specialization: Optional[str] = None
    ):
        """Add a verifier agent"""
        config = VerifierConfig(
            verifier_id=verifier_id,
            agent=agent,
            weight=weight,
            reliability_score=reliability_score,
            specialization=specialization
        )
        self.verifiers[verifier_id] = config
    def reach_consensus(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        method: Optional[ConsensusMethod] = None
    ) -> ConsensusResult:
        """
        Reach consensus on a task
        Args:
            task: Task to verify
            context: Additional context
            method: Consensus method to use
        Returns:
            Consensus result
        """
        method = method or self.consensus_method
        # Check minimum verifiers
        if len(self.verifiers) < self.min_verifiers:
            return ConsensusResult(
                consensus_output=None,
                status=VerificationStatus.INSUFFICIENT_VERIFIERS,
                agreement_level=0.0,
                verifier_results=[],
                consensus_method=method,
                total_verifiers=len(self.verifiers),
                agreeing_verifiers=0,
                confidence_score=0.0,
                metadata={'required_verifiers': self.min_verifiers}
            )
        # Get verifications from all verifiers
        verifier_results = self._collect_verifications(task, context or {})
        # Apply consensus method
        if method == ConsensusMethod.MAJORITY_VOTE:
            consensus = self._majority_vote_consensus(verifier_results)
        elif method == ConsensusMethod.WEIGHTED_VOTE:
            consensus = self._weighted_vote_consensus(verifier_results)
        elif method == ConsensusMethod.UNANIMOUS:
            consensus = self._unanimous_consensus(verifier_results)
        elif method == ConsensusMethod.QUORUM:
            consensus = self._quorum_consensus(verifier_results)
        elif method == ConsensusMethod.BYZANTINE_FAULT_TOLERANT:
            consensus = self._byzantine_consensus(verifier_results)
        elif method == ConsensusMethod.CONFIDENCE_WEIGHTED:
            consensus = self._confidence_weighted_consensus(verifier_results)
        else:
            consensus = self._majority_vote_consensus(verifier_results)
        # Add to history
        self.verification_history.append(consensus)
        return consensus
    def _collect_verifications(
        self,
        task: str,
        context: Dict[str, Any]
    ) -> List[VerificationResult]:
        """Collect verifications from all verifiers"""
        results = []
        for verifier_id, config in self.verifiers.items():
            start_time = time.time()
            try:
                # Execute verifier
                output = self._execute_verifier(config, task, context)
                # Calculate confidence
                confidence = self._calculate_confidence(output, config)
                result = VerificationResult(
                    verifier_id=verifier_id,
                    output=output,
                    confidence=confidence,
                    execution_time=time.time() - start_time,
                    metadata={
                        'weight': config.weight,
                        'reliability': config.reliability_score
                    }
                )
                results.append(result)
            except Exception as e:
                # Record failure
                result = VerificationResult(
                    verifier_id=verifier_id,
                    output=None,
                    confidence=0.0,
                    execution_time=time.time() - start_time,
                    metadata={'error': str(e)}
                )
                results.append(result)
        return results
    def _execute_verifier(
        self,
        config: VerifierConfig,
        task: str,
        context: Dict[str, Any]
    ) -> Any:
        """Execute a single verifier"""
        if hasattr(config.agent, 'verify'):
            return config.agent.verify(task, context)
        elif hasattr(config.agent, 'execute'):
            return config.agent.execute(task)
        elif callable(config.agent):
            return config.agent(task)
        else:
            # Simulate verification
            return f"Verification by {config.verifier_id}: {task}"
    def _calculate_confidence(
        self,
        output: Any,
        config: VerifierConfig
    ) -> float:
        """Calculate confidence score for output"""
        # Base confidence from reliability score
        confidence = config.reliability_score
        # Adjust based on output characteristics
        if isinstance(output, str):
            # Longer, more detailed responses get higher confidence
            length_bonus = min(len(output) / 200, 0.2)
            confidence += length_bonus
        return min(confidence, 1.0)
    def _majority_vote_consensus(
        self,
        results: List[VerificationResult]
    ) -> ConsensusResult:
        """Simple majority vote"""
        # Count outputs
        output_counts = Counter(str(r.output) for r in results)
        if not output_counts:
            return self._no_consensus_result(results, ConsensusMethod.MAJORITY_VOTE)
        # Find majority
        most_common_output, count = output_counts.most_common(1)[0]
        # Check if majority threshold met
        agreement_level = count / len(results)
        if agreement_level >= self.consensus_threshold:
            # Find actual output object
            consensus_output = next(
                r.output for r in results
                if str(r.output) == most_common_output
            )
            # Calculate confidence
            agreeing_results = [r for r in results if str(r.output) == most_common_output]
            avg_confidence = statistics.mean(r.confidence for r in agreeing_results)
            return ConsensusResult(
                consensus_output=consensus_output,
                status=VerificationStatus.CONSENSUS_REACHED,
                agreement_level=agreement_level,
                verifier_results=results,
                consensus_method=ConsensusMethod.MAJORITY_VOTE,
                total_verifiers=len(results),
                agreeing_verifiers=count,
                confidence_score=avg_confidence
            )
        return self._no_consensus_result(results, ConsensusMethod.MAJORITY_VOTE)
    def _weighted_vote_consensus(
        self,
        results: List[VerificationResult]
    ) -> ConsensusResult:
        """Weighted vote based on verifier weights"""
        # Calculate weighted votes
        weighted_votes = defaultdict(float)
        total_weight = 0.0
        for result in results:
            weight = result.metadata.get('weight', 1.0)
            weighted_votes[str(result.output)] += weight
            total_weight += weight
        if not weighted_votes:
            return self._no_consensus_result(results, ConsensusMethod.WEIGHTED_VOTE)
        # Find highest weighted output
        best_output, best_weight = max(weighted_votes.items(), key=lambda x: x[1])
        agreement_level = best_weight / total_weight
        if agreement_level >= self.consensus_threshold:
            consensus_output = next(
                r.output for r in results
                if str(r.output) == best_output
            )
            agreeing_count = sum(
                1 for r in results if str(r.output) == best_output
            )
            return ConsensusResult(
                consensus_output=consensus_output,
                status=VerificationStatus.CONSENSUS_REACHED,
                agreement_level=agreement_level,
                verifier_results=results,
                consensus_method=ConsensusMethod.WEIGHTED_VOTE,
                total_verifiers=len(results),
                agreeing_verifiers=agreeing_count,
                confidence_score=agreement_level
            )
        return self._no_consensus_result(results, ConsensusMethod.WEIGHTED_VOTE)
    def _unanimous_consensus(
        self,
        results: List[VerificationResult]
    ) -> ConsensusResult:
        """Require unanimous agreement"""
        if not results:
            return self._no_consensus_result(results, ConsensusMethod.UNANIMOUS)
        # Check if all outputs are the same
        first_output = str(results[0].output)
        all_agree = all(str(r.output) == first_output for r in results)
        if all_agree:
            avg_confidence = statistics.mean(r.confidence for r in results)
            return ConsensusResult(
                consensus_output=results[0].output,
                status=VerificationStatus.CONSENSUS_REACHED,
                agreement_level=1.0,
                verifier_results=results,
                consensus_method=ConsensusMethod.UNANIMOUS,
                total_verifiers=len(results),
                agreeing_verifiers=len(results),
                confidence_score=avg_confidence
            )
        return self._no_consensus_result(results, ConsensusMethod.UNANIMOUS)
    def _quorum_consensus(
        self,
        results: List[VerificationResult]
    ) -> ConsensusResult:
        """Require quorum (2/3) agreement"""
        output_counts = Counter(str(r.output) for r in results)
        if not output_counts:
            return self._no_consensus_result(results, ConsensusMethod.QUORUM)
        most_common_output, count = output_counts.most_common(1)[0]
        # Quorum is 2/3
        quorum_threshold = 2.0 / 3.0
        agreement_level = count / len(results)
        if agreement_level >= quorum_threshold:
            consensus_output = next(
                r.output for r in results
                if str(r.output) == most_common_output
            )
            agreeing_results = [r for r in results if str(r.output) == most_common_output]
            avg_confidence = statistics.mean(r.confidence for r in agreeing_results)
            return ConsensusResult(
                consensus_output=consensus_output,
                status=VerificationStatus.CONSENSUS_REACHED,
                agreement_level=agreement_level,
                verifier_results=results,
                consensus_method=ConsensusMethod.QUORUM,
                total_verifiers=len(results),
                agreeing_verifiers=count,
                confidence_score=avg_confidence
            )
        return self._no_consensus_result(results, ConsensusMethod.QUORUM)
    def _byzantine_consensus(
        self,
        results: List[VerificationResult]
    ) -> ConsensusResult:
        """Byzantine fault tolerant consensus"""
        # Byzantine consensus requires at least 3f+1 nodes to tolerate f failures
        # We'll assume up to 1/3 can be faulty
        n = len(results)
        f = n // 3  # Maximum faulty nodes
        required_agreement = n - f  # Need agreement from at least n-f nodes
        output_counts = Counter(str(r.output) for r in results)
        if not output_counts:
            return ConsensusResult(
                consensus_output=None,
                status=VerificationStatus.BYZANTINE_FAILURE,
                agreement_level=0.0,
                verifier_results=results,
                consensus_method=ConsensusMethod.BYZANTINE_FAULT_TOLERANT,
                total_verifiers=n,
                agreeing_verifiers=0,
                confidence_score=0.0
            )
        most_common_output, count = output_counts.most_common(1)[0]
        if count >= required_agreement:
            consensus_output = next(
                r.output for r in results
                if str(r.output) == most_common_output
            )
            agreement_level = count / n
            agreeing_results = [r for r in results if str(r.output) == most_common_output]
            avg_confidence = statistics.mean(r.confidence for r in agreeing_results)
            return ConsensusResult(
                consensus_output=consensus_output,
                status=VerificationStatus.CONSENSUS_REACHED,
                agreement_level=agreement_level,
                verifier_results=results,
                consensus_method=ConsensusMethod.BYZANTINE_FAULT_TOLERANT,
                total_verifiers=n,
                agreeing_verifiers=count,
                confidence_score=avg_confidence,
                metadata={
                    'max_faulty_nodes': f,
                    'required_agreement': required_agreement
                }
            )
        return ConsensusResult(
            consensus_output=None,
            status=VerificationStatus.BYZANTINE_FAILURE,
            agreement_level=count / n,
            verifier_results=results,
            consensus_method=ConsensusMethod.BYZANTINE_FAULT_TOLERANT,
            total_verifiers=n,
            agreeing_verifiers=count,
            confidence_score=0.0,
            metadata={
                'max_faulty_nodes': f,
                'required_agreement': required_agreement,
                'actual_agreement': count
            }
        )
    def _confidence_weighted_consensus(
        self,
        results: List[VerificationResult]
    ) -> ConsensusResult:
        """Consensus weighted by confidence scores"""
        # Weight votes by confidence
        confidence_votes = defaultdict(float)
        total_confidence = 0.0
        for result in results:
            confidence_votes[str(result.output)] += result.confidence
            total_confidence += result.confidence
        if not confidence_votes or total_confidence == 0:
            return self._no_consensus_result(results, ConsensusMethod.CONFIDENCE_WEIGHTED)
        best_output, best_confidence = max(confidence_votes.items(), key=lambda x: x[1])
        agreement_level = best_confidence / total_confidence
        if agreement_level >= self.consensus_threshold:
            consensus_output = next(
                r.output for r in results
                if str(r.output) == best_output
            )
            agreeing_count = sum(
                1 for r in results if str(r.output) == best_output
            )
            return ConsensusResult(
                consensus_output=consensus_output,
                status=VerificationStatus.CONSENSUS_REACHED,
                agreement_level=agreement_level,
                verifier_results=results,
                consensus_method=ConsensusMethod.CONFIDENCE_WEIGHTED,
                total_verifiers=len(results),
                agreeing_verifiers=agreeing_count,
                confidence_score=agreement_level
            )
        return self._no_consensus_result(results, ConsensusMethod.CONFIDENCE_WEIGHTED)
    def _no_consensus_result(
        self,
        results: List[VerificationResult],
        method: ConsensusMethod
    ) -> ConsensusResult:
        """Create no consensus result"""
        return ConsensusResult(
            consensus_output=None,
            status=VerificationStatus.NO_CONSENSUS,
            agreement_level=0.0,
            verifier_results=results,
            consensus_method=method,
            total_verifiers=len(results),
            agreeing_verifiers=0,
            confidence_score=0.0
        )
    def get_statistics(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        if not self.verification_history:
            return {'message': 'No verification history'}
        successful = sum(
            1 for r in self.verification_history
            if r.status == VerificationStatus.CONSENSUS_REACHED
        )
        avg_agreement = statistics.mean(
            r.agreement_level for r in self.verification_history
        )
        avg_confidence = statistics.mean(
            r.confidence_score for r in self.verification_history
            if r.confidence_score > 0
        )
        method_counts = Counter(
            r.consensus_method.value for r in self.verification_history
        )
        return {
            'total_verifications': len(self.verification_history),
            'successful_consensus': successful,
            'success_rate': successful / len(self.verification_history),
            'avg_agreement_level': avg_agreement,
            'avg_confidence': avg_confidence,
            'total_verifiers': len(self.verifiers),
            'method_distribution': dict(method_counts)
        }
class SimpleVerifier:
    """Simple verifier agent for demonstration"""
    def __init__(self, name: str, bias: Optional[str] = None):
        self.name = name
        self.bias = bias  # Optional bias for testing
    def verify(self, task: str, context: Dict[str, Any]) -> str:
        """Verify a task"""
        if self.bias:
            return f"{self.bias} verification result"
        return f"Standard verification: {task}"
def main():
    """Demonstrate redundancy and consensus pattern"""
    print("=" * 60)
    print("Redundancy & Consensus Demonstration")
    print("=" * 60)
    print("\n1. Setting Up Consensus System")
    print("-" * 60)
    consensus = ConsensusSystem(
        consensus_method=ConsensusMethod.MAJORITY_VOTE,
        min_verifiers=3,
        consensus_threshold=0.66
    )
    # Add verifiers
    verifiers = [
        ("verifier_1", SimpleVerifier("V1"), 1.0, 0.9),
        ("verifier_2", SimpleVerifier("V2"), 1.0, 0.95),
        ("verifier_3", SimpleVerifier("V3"), 1.0, 0.85),
        ("verifier_4", SimpleVerifier("V4"), 0.8, 0.8),
        ("verifier_5", SimpleVerifier("V5"), 1.2, 1.0),
    ]
    for vid, agent, weight, reliability in verifiers:
        consensus.add_verifier(
            agent=agent,
            verifier_id=vid,
            weight=weight,
            reliability_score=reliability
        )
        print(f"Added {vid}: weight={weight}, reliability={reliability}")
    print(f"\nTotal verifiers: {len(consensus.verifiers)}")
    print("\n" + "=" * 60)
    print("2. Majority Vote Consensus")
    print("=" * 60)
    result = consensus.reach_consensus(
        "Verify calculation: 2 + 2 = 4",
        method=ConsensusMethod.MAJORITY_VOTE
    )
    print(f"\nStatus: {result.status.value}")
    print(f"Agreement Level: {result.agreement_level:.2%}")
    print(f"Agreeing Verifiers: {result.agreeing_verifiers}/{result.total_verifiers}")
    print(f"Confidence Score: {result.confidence_score:.3f}")
    print(f"Consensus Output: {result.consensus_output}")
    print("\nIndividual verifier results:")
    for vr in result.verifier_results:
        print(f"  {vr.verifier_id}: {vr.output[:50]}... (confidence: {vr.confidence:.2f})")
    print("\n" + "=" * 60)
    print("3. Weighted Vote Consensus")
    print("=" * 60)
    result = consensus.reach_consensus(
        "Validate solution approach",
        method=ConsensusMethod.WEIGHTED_VOTE
    )
    print(f"\nStatus: {result.status.value}")
    print(f"Agreement Level: {result.agreement_level:.2%}")
    print(f"Confidence: {result.confidence_score:.3f}")
    print("\n" + "=" * 60)
    print("4. Byzantine Fault Tolerant Consensus")
    print("=" * 60)
    # Add a faulty verifier
    consensus.add_verifier(
        agent=SimpleVerifier("Faulty", bias="INCORRECT"),
        verifier_id="faulty_1",
        weight=1.0,
        reliability_score=0.3
    )
    result = consensus.reach_consensus(
        "Critical security decision",
        method=ConsensusMethod.BYZANTINE_FAULT_TOLERANT
    )
    print(f"\nStatus: {result.status.value}")
    print(f"Agreement Level: {result.agreement_level:.2%}")
    print(f"Total Verifiers: {result.total_verifiers}")
    print(f"Agreeing: {result.agreeing_verifiers}")
    if 'max_faulty_nodes' in result.metadata:
        print(f"Max Faulty Nodes Tolerated: {result.metadata['max_faulty_nodes']}")
        print(f"Required Agreement: {result.metadata['required_agreement']}")
    print("\n" + "=" * 60)
    print("5. Unanimous Consensus")
    print("=" * 60)
    result = consensus.reach_consensus(
        "Must have complete agreement",
        method=ConsensusMethod.UNANIMOUS
    )
    print(f"\nStatus: {result.status.value}")
    if result.status == VerificationStatus.CONSENSUS_REACHED:
        print(f"All {result.total_verifiers} verifiers agree!")
        print(f"Confidence: {result.confidence_score:.3f}")
    else:
        print(f"Could not reach unanimous consensus")
        print(f"Agreement level: {result.agreement_level:.2%}")
    print("\n" + "=" * 60)
    print("6. Quorum Consensus (2/3 Required)")
    print("=" * 60)
    result = consensus.reach_consensus(
        "Quorum-based decision",
        method=ConsensusMethod.QUORUM
    )
    print(f"\nStatus: {result.status.value}")
    print(f"Agreement Level: {result.agreement_level:.2%}")
    print(f"Quorum Threshold: 66.7%")
    print(f"Agreeing: {result.agreeing_verifiers}/{result.total_verifiers}")
    print("\n" + "=" * 60)
    print("7. Confidence-Weighted Consensus")
    print("=" * 60)
    result = consensus.reach_consensus(
        "Weight by confidence scores",
        method=ConsensusMethod.CONFIDENCE_WEIGHTED
    )
    print(f"\nStatus: {result.status.value}")
    print(f"Weighted Agreement: {result.agreement_level:.2%}")
    print(f"Overall Confidence: {result.confidence_score:.3f}")
    print("\nVerifier contributions:")
    for vr in sorted(result.verifier_results, key=lambda x: x.confidence, reverse=True):
        print(f"  {vr.verifier_id}: confidence={vr.confidence:.2f}")
    print("\n" + "=" * 60)
    print("8. Multiple Consensus Rounds")
    print("=" * 60)
    tasks = [
        "Task A: Mathematical proof",
        "Task B: Code review",
        "Task C: Design decision",
        "Task D: Security audit"
    ]
    for task in tasks:
        result = consensus.reach_consensus(task)
        status_icon = "✓" if result.status == VerificationStatus.CONSENSUS_REACHED else "✗"
        print(f"{status_icon} {task}: {result.agreement_level:.1%} agreement")
    print("\n" + "=" * 60)
    print("9. Consensus Statistics")
    print("=" * 60)
    stats = consensus.get_statistics()
    print(f"\nTotal Verifications: {stats['total_verifications']}")
    print(f"Successful Consensus: {stats['successful_consensus']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Avg Agreement Level: {stats['avg_agreement_level']:.2%}")
    print(f"Avg Confidence: {stats['avg_confidence']:.3f}")
    print(f"Total Verifiers: {stats['total_verifiers']}")
    print("\nMethod Distribution:")
    for method, count in stats['method_distribution'].items():
        print(f"  {method}: {count}")
    print("\n" + "=" * 60)
    print("Redundancy & Consensus demonstration complete!")
    print("=" * 60)
if __name__ == "__main__":
    main()
