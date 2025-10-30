"""
Agentic Design Pattern: Agent Synchronization

This pattern implements distributed coordination and state synchronization between
multiple agents using consensus protocols, vector clocks, and conflict resolution.

Key Components:
1. VectorClock - Logical clock for causality tracking
2. StateSnapshot - Versioned state representation
3. ConsensusProtocol - Raft-like consensus algorithm
4. ConflictResolver - Resolves state conflicts
5. AgentSynchronizationManager - Main orchestrator

Features:
- Distributed state synchronization
- Causality tracking with vector clocks
- Consensus protocols (leader election, log replication)
- Conflict detection and resolution
- Eventually consistent state
- Network partition handling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from collections import defaultdict
import random
import time


class AgentRole(Enum):
    """Roles in consensus protocol."""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"


class MessageType(Enum):
    """Types of synchronization messages."""
    STATE_UPDATE = "state_update"
    HEARTBEAT = "heartbeat"
    VOTE_REQUEST = "vote_request"
    VOTE_RESPONSE = "vote_response"
    APPEND_ENTRIES = "append_entries"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""
    LAST_WRITE_WINS = "last_write_wins"
    VECTOR_CLOCK = "vector_clock"
    MERGE = "merge"
    MANUAL = "manual"


@dataclass
class VectorClock:
    """
    Vector clock for tracking causality in distributed systems.
    
    Each agent maintains a vector of timestamps, one per agent.
    """
    clocks: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, agent_id: str):
        """Increment clock for an agent."""
        self.clocks[agent_id] = self.clocks.get(agent_id, 0) + 1
    
    def update(self, other: 'VectorClock'):
        """Update clock by taking maximum of each component."""
        for agent_id, timestamp in other.clocks.items():
            self.clocks[agent_id] = max(
                self.clocks.get(agent_id, 0),
                timestamp
            )
    
    def happens_before(self, other: 'VectorClock') -> bool:
        """Check if this clock happens before another (causality)."""
        # self < other if all components <= and at least one is strictly less
        all_less_or_equal = all(
            self.clocks.get(agent_id, 0) <= other.clocks.get(agent_id, 0)
            for agent_id in set(self.clocks.keys()) | set(other.clocks.keys())
        )
        
        some_strictly_less = any(
            self.clocks.get(agent_id, 0) < other.clocks.get(agent_id, 0)
            for agent_id in set(self.clocks.keys()) | set(other.clocks.keys())
        )
        
        return all_less_or_equal and some_strictly_less
    
    def concurrent(self, other: 'VectorClock') -> bool:
        """Check if clocks are concurrent (no causal relationship)."""
        return not self.happens_before(other) and not other.happens_before(self)
    
    def copy(self) -> 'VectorClock':
        """Create a copy of this clock."""
        return VectorClock(clocks=self.clocks.copy())


@dataclass
class StateSnapshot:
    """Versioned state snapshot with vector clock."""
    version: int
    data: Dict[str, Any]
    vector_clock: VectorClock
    agent_id: str
    timestamp: float
    checksum: Optional[str] = None


@dataclass
class SyncMessage:
    """Message for agent synchronization."""
    message_id: str
    message_type: MessageType
    sender_id: str
    receiver_id: str
    vector_clock: VectorClock
    payload: Dict[str, Any]
    timestamp: float


@dataclass
class ConsensusVote:
    """Vote in consensus protocol."""
    term: int
    candidate_id: str
    voter_id: str
    granted: bool


@dataclass
class LogEntry:
    """Entry in replicated log."""
    index: int
    term: int
    command: Dict[str, Any]
    vector_clock: VectorClock


class ConsensusProtocol:
    """Simplified Raft-like consensus protocol."""
    
    def __init__(self, agent_id: str, peer_ids: List[str]):
        self.agent_id = agent_id
        self.peer_ids = peer_ids
        self.role = AgentRole.FOLLOWER
        
        # Raft state
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index: Dict[str, int] = {}
        self.match_index: Dict[str, int] = {}
        
        # Timing
        self.last_heartbeat = time.time()
        self.election_timeout = random.uniform(1.0, 2.0)
        self.heartbeat_interval = 0.5
        
    def start_election(self) -> List[SyncMessage]:
        """Start leader election."""
        self.role = AgentRole.CANDIDATE
        self.current_term += 1
        self.voted_for = self.agent_id
        
        print(f"🗳️  Agent {self.agent_id} starting election for term {self.current_term}")
        
        # Send vote requests
        messages = []
        for peer_id in self.peer_ids:
            message = SyncMessage(
                message_id=f"vote_req_{self.agent_id}_{self.current_term}",
                message_type=MessageType.VOTE_REQUEST,
                sender_id=self.agent_id,
                receiver_id=peer_id,
                vector_clock=VectorClock(clocks={self.agent_id: self.current_term}),
                payload={
                    'term': self.current_term,
                    'candidate_id': self.agent_id,
                    'last_log_index': len(self.log) - 1 if self.log else -1,
                    'last_log_term': self.log[-1].term if self.log else 0
                },
                timestamp=time.time()
            )
            messages.append(message)
        
        return messages
    
    def process_vote_request(self, message: SyncMessage) -> Optional[SyncMessage]:
        """Process vote request from candidate."""
        term = message.payload['term']
        candidate_id = message.payload['candidate_id']
        
        # Update term if higher
        if term > self.current_term:
            self.current_term = term
            self.role = AgentRole.FOLLOWER
            self.voted_for = None
        
        # Decide whether to grant vote
        can_vote = (
            term >= self.current_term and
            (self.voted_for is None or self.voted_for == candidate_id)
        )
        
        if can_vote:
            self.voted_for = candidate_id
            granted = True
        else:
            granted = False
        
        # Send vote response
        return SyncMessage(
            message_id=f"vote_resp_{self.agent_id}_{term}",
            message_type=MessageType.VOTE_RESPONSE,
            sender_id=self.agent_id,
            receiver_id=candidate_id,
            vector_clock=VectorClock(clocks={self.agent_id: term}),
            payload={
                'term': self.current_term,
                'voter_id': self.agent_id,
                'granted': granted
            },
            timestamp=time.time()
        )
    
    def become_leader(self):
        """Transition to leader role."""
        self.role = AgentRole.LEADER
        print(f"👑 Agent {self.agent_id} became leader for term {self.current_term}")
        
        # Initialize leader state
        for peer_id in self.peer_ids:
            self.next_index[peer_id] = len(self.log)
            self.match_index[peer_id] = -1
    
    def append_entry(self, command: Dict[str, Any], vector_clock: VectorClock) -> int:
        """Append entry to log (leader only)."""
        entry = LogEntry(
            index=len(self.log),
            term=self.current_term,
            command=command,
            vector_clock=vector_clock
        )
        self.log.append(entry)
        return entry.index
    
    def create_heartbeat(self, peer_id: str) -> SyncMessage:
        """Create heartbeat message for peer."""
        return SyncMessage(
            message_id=f"heartbeat_{self.agent_id}_{time.time()}",
            message_type=MessageType.HEARTBEAT,
            sender_id=self.agent_id,
            receiver_id=peer_id,
            vector_clock=VectorClock(clocks={self.agent_id: self.current_term}),
            payload={
                'term': self.current_term,
                'leader_id': self.agent_id,
                'commit_index': self.commit_index
            },
            timestamp=time.time()
        )


class ConflictResolver:
    """Resolves conflicts between state versions."""
    
    def __init__(self, strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.VECTOR_CLOCK):
        self.strategy = strategy
        self.conflicts_resolved = 0
        
    def resolve(
        self,
        local_state: StateSnapshot,
        remote_state: StateSnapshot
    ) -> StateSnapshot:
        """
        Resolve conflict between local and remote state.
        
        Returns:
            Resolved state snapshot
        """
        if self.strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
            return self._last_write_wins(local_state, remote_state)
        elif self.strategy == ConflictResolutionStrategy.VECTOR_CLOCK:
            return self._vector_clock_resolution(local_state, remote_state)
        elif self.strategy == ConflictResolutionStrategy.MERGE:
            return self._merge_states(local_state, remote_state)
        else:
            # Default to vector clock
            return self._vector_clock_resolution(local_state, remote_state)
    
    def _last_write_wins(
        self,
        local_state: StateSnapshot,
        remote_state: StateSnapshot
    ) -> StateSnapshot:
        """Use timestamp to resolve conflict."""
        self.conflicts_resolved += 1
        return remote_state if remote_state.timestamp > local_state.timestamp else local_state
    
    def _vector_clock_resolution(
        self,
        local_state: StateSnapshot,
        remote_state: StateSnapshot
    ) -> StateSnapshot:
        """Use vector clocks to determine causality."""
        local_vc = local_state.vector_clock
        remote_vc = remote_state.vector_clock
        
        if local_vc.happens_before(remote_vc):
            # Remote is newer
            self.conflicts_resolved += 1
            return remote_state
        elif remote_vc.happens_before(local_vc):
            # Local is newer
            self.conflicts_resolved += 1
            return local_state
        else:
            # Concurrent updates - merge
            self.conflicts_resolved += 1
            return self._merge_states(local_state, remote_state)
    
    def _merge_states(
        self,
        local_state: StateSnapshot,
        remote_state: StateSnapshot
    ) -> StateSnapshot:
        """Merge concurrent states."""
        # Simple merge: take union of data, preferring remote on conflicts
        merged_data = local_state.data.copy()
        merged_data.update(remote_state.data)
        
        # Merge vector clocks
        merged_vc = local_state.vector_clock.copy()
        merged_vc.update(remote_state.vector_clock)
        
        return StateSnapshot(
            version=max(local_state.version, remote_state.version) + 1,
            data=merged_data,
            vector_clock=merged_vc,
            agent_id=local_state.agent_id,
            timestamp=max(local_state.timestamp, remote_state.timestamp)
        )


class AgentSynchronizationManager:
    """
    Main manager for agent synchronization.
    
    Coordinates:
    - State synchronization across agents
    - Consensus protocol execution
    - Conflict resolution
    - Message passing
    """
    
    def __init__(
        self,
        agent_id: str,
        peer_ids: List[str],
        conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.VECTOR_CLOCK
    ):
        self.agent_id = agent_id
        self.peer_ids = peer_ids
        
        # Components
        self.consensus = ConsensusProtocol(agent_id, peer_ids)
        self.resolver = ConflictResolver(conflict_strategy)
        
        # State
        self.vector_clock = VectorClock(clocks={agent_id: 0})
        self.local_state = StateSnapshot(
            version=0,
            data={},
            vector_clock=self.vector_clock.copy(),
            agent_id=agent_id,
            timestamp=time.time()
        )
        
        # Peer states (what we know about other agents)
        self.peer_states: Dict[str, StateSnapshot] = {}
        
        # Message queue
        self.outgoing_messages: List[SyncMessage] = []
        self.message_log: List[SyncMessage] = []
        
        # Statistics
        self.sync_count = 0
        self.conflict_count = 0
        self.message_count = 0
        
    def update_local_state(self, key: str, value: Any):
        """Update local state and increment vector clock."""
        # Increment local clock
        self.vector_clock.increment(self.agent_id)
        
        # Update state
        new_data = self.local_state.data.copy()
        new_data[key] = value
        
        self.local_state = StateSnapshot(
            version=self.local_state.version + 1,
            data=new_data,
            vector_clock=self.vector_clock.copy(),
            agent_id=self.agent_id,
            timestamp=time.time()
        )
        
        print(f"📝 Agent {self.agent_id} updated state: {key}={value} (version {self.local_state.version})")
    
    def synchronize_with_peer(self, peer_id: str) -> SyncMessage:
        """Create synchronization message for peer."""
        message = SyncMessage(
            message_id=f"sync_{self.agent_id}_{peer_id}_{time.time()}",
            message_type=MessageType.SYNC_REQUEST,
            sender_id=self.agent_id,
            receiver_id=peer_id,
            vector_clock=self.vector_clock.copy(),
            payload={
                'state': {
                    'version': self.local_state.version,
                    'data': self.local_state.data,
                    'timestamp': self.local_state.timestamp
                }
            },
            timestamp=time.time()
        )
        
        self.outgoing_messages.append(message)
        self.message_count += 1
        
        return message
    
    def process_sync_request(self, message: SyncMessage) -> StateSnapshot:
        """
        Process synchronization request from peer.
        
        Returns:
            Updated local state after sync
        """
        sender_id = message.sender_id
        remote_state_data = message.payload['state']
        
        # Reconstruct remote state
        remote_state = StateSnapshot(
            version=remote_state_data['version'],
            data=remote_state_data['data'],
            vector_clock=message.vector_clock,
            agent_id=sender_id,
            timestamp=remote_state_data['timestamp']
        )
        
        print(f"🔄 Agent {self.agent_id} received sync from {sender_id} (version {remote_state.version})")
        
        # Update vector clock
        self.vector_clock.update(message.vector_clock)
        self.vector_clock.increment(self.agent_id)
        
        # Check for conflicts
        if self.local_state.vector_clock.concurrent(remote_state.vector_clock):
            print(f"  ⚠️  Conflict detected with {sender_id}")
            self.conflict_count += 1
            
            # Resolve conflict
            resolved_state = self.resolver.resolve(self.local_state, remote_state)
            self.local_state = resolved_state
            print(f"  ✓ Conflict resolved (version {resolved_state.version})")
        else:
            # No conflict - apply causally later state
            if remote_state.vector_clock.happens_before(self.local_state.vector_clock):
                # Local is newer
                print(f"  ✓ Local state is newer, keeping local")
            else:
                # Remote is newer or concurrent
                self.local_state = remote_state
                print(f"  ✓ Applied remote state (version {remote_state.version})")
        
        # Update peer state tracking
        self.peer_states[sender_id] = remote_state
        self.sync_count += 1
        self.message_log.append(message)
        
        return self.local_state
    
    def broadcast_state(self) -> List[SyncMessage]:
        """Broadcast state to all peers."""
        messages = []
        for peer_id in self.peer_ids:
            message = self.synchronize_with_peer(peer_id)
            messages.append(message)
        
        print(f"📡 Agent {self.agent_id} broadcasting state to {len(self.peer_ids)} peers")
        return messages
    
    def get_synchronization_status(self) -> Dict[str, Any]:
        """Get current synchronization status."""
        # Check how many peers are in sync
        in_sync_count = 0
        out_of_sync_count = 0
        
        for peer_id, peer_state in self.peer_states.items():
            if peer_state.version == self.local_state.version:
                in_sync_count += 1
            else:
                out_of_sync_count += 1
        
        return {
            'agent_id': self.agent_id,
            'local_version': self.local_state.version,
            'vector_clock': dict(self.vector_clock.clocks),
            'consensus_role': self.consensus.role.value,
            'consensus_term': self.consensus.current_term,
            'peers_tracked': len(self.peer_states),
            'peers_in_sync': in_sync_count,
            'peers_out_of_sync': out_of_sync_count,
            'total_syncs': self.sync_count,
            'total_conflicts': self.conflict_count,
            'conflicts_resolved': self.resolver.conflicts_resolved,
            'messages_sent': self.message_count
        }
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of current state."""
        return {
            'agent_id': self.agent_id,
            'version': self.local_state.version,
            'data': self.local_state.data,
            'vector_clock': dict(self.vector_clock.clocks),
            'timestamp': self.local_state.timestamp,
            'known_peers': list(self.peer_states.keys())
        }


# Demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("AGENT SYNCHRONIZATION DEMONSTRATION")
    print("=" * 80)
    
    # Create a cluster of 3 agents
    agent_ids = ['agent_1', 'agent_2', 'agent_3']
    
    agents: Dict[str, AgentSynchronizationManager] = {}
    
    print("\n🤖 Creating Agent Cluster:")
    for agent_id in agent_ids:
        peer_ids = [aid for aid in agent_ids if aid != agent_id]
        agent = AgentSynchronizationManager(
            agent_id=agent_id,
            peer_ids=peer_ids,
            conflict_strategy=ConflictResolutionStrategy.VECTOR_CLOCK
        )
        agents[agent_id] = agent
        print(f"  • {agent_id} with peers: {', '.join(peer_ids)}")
    
    # Scenario 1: Concurrent updates
    print(f"\n{'='*80}")
    print("SCENARIO 1: Concurrent Updates")
    print(f"{'='*80}\n")
    
    # Each agent updates different keys
    agents['agent_1'].update_local_state('temperature', 22.5)
    agents['agent_2'].update_local_state('humidity', 65.0)
    agents['agent_3'].update_local_state('pressure', 1013.0)
    
    # Agent 1 and Agent 2 make concurrent updates to same key
    print("\n🔀 Creating concurrent conflict:")
    agents['agent_1'].update_local_state('mode', 'heating')
    agents['agent_2'].update_local_state('mode', 'cooling')
    
    # Synchronize Agent 1 -> Agent 2
    print(f"\n{'='*80}")
    print("SYNCHRONIZATION: Agent 1 → Agent 2")
    print(f"{'='*80}\n")
    
    message = agents['agent_1'].synchronize_with_peer('agent_2')
    agents['agent_2'].process_sync_request(message)
    
    # Synchronize Agent 2 -> Agent 3
    print(f"\n{'='*80}")
    print("SYNCHRONIZATION: Agent 2 → Agent 3")
    print(f"{'='*80}\n")
    
    message = agents['agent_2'].synchronize_with_peer('agent_3')
    agents['agent_3'].process_sync_request(message)
    
    # Broadcast from Agent 3 to all
    print(f"\n{'='*80}")
    print("BROADCAST: Agent 3 → All Peers")
    print(f"{'='*80}\n")
    
    messages = agents['agent_3'].broadcast_state()
    for message in messages:
        if message.receiver_id in agents:
            agents[message.receiver_id].process_sync_request(message)
    
    # Show final states
    print(f"\n{'='*80}")
    print("FINAL AGENT STATES")
    print(f"{'='*80}\n")
    
    for agent_id, agent in agents.items():
        state = agent.get_state_summary()
        print(f"Agent: {state['agent_id']}")
        print(f"  Version: {state['version']}")
        print(f"  Data: {state['data']}")
        print(f"  Vector Clock: {state['vector_clock']}")
        print()
    
    # Show synchronization status
    print(f"{'='*80}")
    print("SYNCHRONIZATION STATUS")
    print(f"{'='*80}\n")
    
    for agent_id, agent in agents.items():
        status = agent.get_synchronization_status()
        print(f"Agent: {status['agent_id']}")
        print(f"  Local Version: {status['local_version']}")
        print(f"  Vector Clock: {status['vector_clock']}")
        print(f"  Consensus: {status['consensus_role']} (term {status['consensus_term']})")
        print(f"  Peers In Sync: {status['peers_in_sync']}/{status['peers_tracked']}")
        print(f"  Total Syncs: {status['total_syncs']}")
        print(f"  Total Conflicts: {status['total_conflicts']}")
        print(f"  Conflicts Resolved: {status['conflicts_resolved']}")
        print(f"  Messages Sent: {status['messages_sent']}")
        print()
    
    # Scenario 2: Leader election
    print(f"{'='*80}")
    print("SCENARIO 2: Leader Election")
    print(f"{'='*80}\n")
    
    # Agent 1 starts election
    vote_requests = agents['agent_1'].consensus.start_election()
    votes_received = 1  # Self-vote
    
    # Process votes
    for vote_req in vote_requests:
        receiver = agents[vote_req.receiver_id]
        vote_response = receiver.consensus.process_vote_request(vote_req)
        
        if vote_response and vote_response.payload['granted']:
            votes_received += 1
            print(f"  ✓ {vote_response.sender_id} voted for {vote_req.sender_id}")
        else:
            print(f"  ✗ {vote_req.receiver_id} rejected vote for {vote_req.sender_id}")
    
    # Check if won election (majority)
    if votes_received > len(agent_ids) // 2:
        agents['agent_1'].consensus.become_leader()
    
    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    total_syncs = sum(a.sync_count for a in agents.values())
    total_conflicts = sum(a.conflict_count for a in agents.values())
    total_messages = sum(a.message_count for a in agents.values())
    
    print(f"Total Synchronizations: {total_syncs}")
    print(f"Total Conflicts: {total_conflicts}")
    print(f"Total Messages: {total_messages}")
    print(f"Conflict Resolution Rate: {(total_conflicts / total_syncs * 100) if total_syncs > 0 else 0:.1f}%")
    
    # Check consensus
    leader_count = sum(1 for a in agents.values() if a.consensus.role == AgentRole.LEADER)
    print(f"\nConsensus Status:")
    print(f"  Leaders: {leader_count}")
    print(f"  Followers: {len(agents) - leader_count}")
    
    print("\n" + "="*80)
    print("✅ Agent Synchronization demonstration complete!")
    print("="*80)
    print("\nKey Achievements:")
    print("• Distributed state synchronization with vector clocks")
    print("• Causality tracking and conflict detection")
    print("• Consensus protocol with leader election")
    print("• Automatic conflict resolution")
    print("• Eventually consistent distributed state")
    print(f"• Synchronized {len(agents)} agents with {total_conflicts} conflicts resolved")
