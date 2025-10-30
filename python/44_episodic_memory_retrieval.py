"""
Pattern 30: Episodic Memory Retrieval
Description:
    Stores and retrieves specific experiences/episodes with temporal
    and contextual information for experience-based reasoning.
Use Cases:
    - Experience-based learning
    - Contextual recall
    - Temporal reasoning
    - Case-based reasoning
Key Features:
    - Temporal episode storage
    - Context-aware retrieval
    - Similarity-based matching
    - Temporal decay modeling
Example:
    >>> memory = EpisodicMemory()
    >>> memory.store_episode(event, context)
    >>> episodes = memory.retrieve_similar(query, context)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time
import math
from collections import defaultdict
import numpy as np
class EpisodeType(Enum):
    """Types of episodes"""
    INTERACTION = "interaction"
    TASK_COMPLETION = "task_completion"
    ERROR = "error"
    SUCCESS = "success"
    OBSERVATION = "observation"
@dataclass
class TemporalContext:
    """Temporal information for an episode"""
    timestamp: float
    time_of_day: str  # morning, afternoon, evening, night
    day_of_week: Optional[str] = None
    duration: Optional[float] = None
@dataclass
class SpatialContext:
    """Spatial/environmental context"""
    location: Optional[str] = None
    environment_type: Optional[str] = None
    related_entities: List[str] = field(default_factory=list)
@dataclass
class EmotionalContext:
    """Emotional/motivational context"""
    sentiment: float = 0.0  # -1 to 1
    confidence: float = 0.5
    importance: float = 0.5
@dataclass
class Episode:
    """An episodic memory"""
    episode_id: str
    episode_type: EpisodeType
    content: Any
    temporal_context: TemporalContext
    spatial_context: Optional[SpatialContext] = None
    emotional_context: Optional[EmotionalContext] = None
    tags: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    access_count: int = 0
    last_accessed: float = 0.0
    related_episodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
@dataclass
class RetrievalResult:
    """Result of episodic retrieval"""
    episode: Episode
    similarity_score: float
    temporal_distance: float
    context_match: float
    overall_score: float
    retrieval_cues: Dict[str, float] = field(default_factory=dict)
class EpisodicMemory:
    """
    Episodic memory system for storing and retrieving experiences
    Features:
    - Temporal organization
    - Context-aware retrieval
    - Similarity-based matching
    - Temporal decay
    """
    def __init__(
        self,
        decay_rate: float = 0.01,
        embedding_dim: int = 128
    ):
        self.episodes: Dict[str, Episode] = {}
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)  # date -> episodes
        self.type_index: Dict[EpisodeType, List[str]] = defaultdict(list)
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self.decay_rate = decay_rate
        self.embedding_dim = embedding_dim
        self.episode_counter = 0
    def store_episode(
        self,
        content: Any,
        episode_type: EpisodeType,
        temporal_context: Optional[TemporalContext] = None,
        spatial_context: Optional[SpatialContext] = None,
        emotional_context: Optional[EmotionalContext] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a new episode
        Args:
            content: Episode content
            episode_type: Type of episode
            temporal_context: When it happened
            spatial_context: Where it happened
            emotional_context: How it felt
            tags: Categorization tags
            metadata: Additional metadata
        Returns:
            Episode ID
        """
        self.episode_counter += 1
        episode_id = f"episode_{self.episode_counter}_{int(time.time())}"
        # Create temporal context if not provided
        if temporal_context is None:
            temporal_context = self._create_temporal_context()
        # Generate embedding
        embedding = self._generate_embedding(content, tags or [])
        # Create episode
        episode = Episode(
            episode_id=episode_id,
            episode_type=episode_type,
            content=content,
            temporal_context=temporal_context,
            spatial_context=spatial_context,
            emotional_context=emotional_context,
            tags=tags or [],
            embedding=embedding,
            metadata=metadata or {}
        )
        # Store episode
        self.episodes[episode_id] = episode
        # Index episode
        self._index_episode(episode)
        return episode_id
    def retrieve_similar(
        self,
        query: Any,
        context: Optional[Dict[str, Any]] = None,
        max_results: int = 10,
        time_window: Optional[float] = None,
        episode_types: Optional[List[EpisodeType]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve episodes similar to query
        Args:
            query: Query content
            context: Query context
            max_results: Maximum results
            time_window: Time window in seconds (None = all time)
            episode_types: Filter by episode types
        Returns:
            List of retrieval results
        """
        current_time = time.time()
        query_embedding = self._generate_embedding(query, context.get('tags', []) if context else [])
        results = []
        for episode in self.episodes.values():
            # Filter by type
            if episode_types and episode.episode_type not in episode_types:
                continue
            # Filter by time window
            if time_window:
                time_diff = current_time - episode.temporal_context.timestamp
                if time_diff > time_window:
                    continue
            # Calculate similarity
            similarity = self._calculate_similarity(query_embedding, episode.embedding)
            # Calculate temporal distance
            temporal_distance = current_time - episode.temporal_context.timestamp
            # Calculate context match
            context_match = self._calculate_context_match(episode, context or {})
            # Apply temporal decay
            decay_factor = math.exp(-self.decay_rate * temporal_distance / 86400)  # days
            # Calculate overall score
            overall_score = (
                similarity * 0.5 +
                context_match * 0.3 +
                decay_factor * 0.2
            )
            results.append(RetrievalResult(
                episode=episode,
                similarity_score=similarity,
                temporal_distance=temporal_distance,
                context_match=context_match,
                overall_score=overall_score,
                retrieval_cues={
                    'similarity': similarity,
                    'context': context_match,
                    'recency': decay_factor
                }
            ))
        # Sort by overall score
        results.sort(key=lambda x: x.overall_score, reverse=True)
        # Update access counts
        for result in results[:max_results]:
            result.episode.access_count += 1
            result.episode.last_accessed = current_time
        return results[:max_results]
    def retrieve_by_time_range(
        self,
        start_time: float,
        end_time: float,
        episode_types: Optional[List[EpisodeType]] = None
    ) -> List[Episode]:
        """Retrieve episodes within a time range"""
        results = []
        for episode in self.episodes.values():
            timestamp = episode.temporal_context.timestamp
            if start_time <= timestamp <= end_time:
                if not episode_types or episode.episode_type in episode_types:
                    results.append(episode)
        # Sort by time
        results.sort(key=lambda x: x.temporal_context.timestamp)
        return results
    def retrieve_by_tags(
        self,
        tags: List[str],
        match_all: bool = False
    ) -> List[Episode]:
        """Retrieve episodes by tags"""
        if match_all:
            # Find episodes with all tags
            episode_sets = [set(self.tag_index[tag]) for tag in tags]
            if episode_sets:                
                matching_ids = set.intersection(*episode_sets)
            else:
                matching_ids = set()
        else:
            # Find episodes with any tag
            matching_ids = set()
            for tag in tags:
                matching_ids.update(self.tag_index[tag])
        return [self.episodes[eid] for eid in matching_ids]
    def retrieve_recent(
        self,
        n: int = 10,
        episode_types: Optional[List[EpisodeType]] = None
    ) -> List[Episode]:
        """Retrieve most recent episodes"""
        filtered_episodes = self.episodes.values()
        if episode_types:
            filtered_episodes = [
                ep for ep in filtered_episodes
                if ep.episode_type in episode_types
            ]
        # Sort by timestamp
        sorted_episodes = sorted(
            filtered_episodes,
            key=lambda x: x.temporal_context.timestamp,
            reverse=True
        )
        return sorted_episodes[:n]
    def retrieve_most_accessed(
        self,
        n: int = 10
    ) -> List[Episode]:
        """Retrieve most frequently accessed episodes"""
        sorted_episodes = sorted(
            self.episodes.values(),
            key=lambda x: x.access_count,
            reverse=True
        )
        return sorted_episodes[:n]
    def find_related_episodes(
        self,
        episode_id: str,
        max_results: int = 5
    ) -> List[RetrievalResult]:
        """Find episodes related to a given episode"""
        if episode_id not in self.episodes:
            return []
        source_episode = self.episodes[episode_id]
        # Use the episode content as query
        return self.retrieve_similar(
            query=source_episode.content,
            context={
                'tags': source_episode.tags,
                'type': source_episode.episode_type
            },
            max_results=max_results + 1  # +1 to exclude source
        )[1:]  # Skip first result (will be the source episode itself)
    def consolidate_episodes(
        self,
        episode_ids: List[str],
        new_type: EpisodeType,
        consolidation_strategy: str = 'merge'
    ) -> str:
        """
        Consolidate multiple episodes into one
        Args:
            episode_ids: Episodes to consolidate
            new_type: Type for consolidated episode
            consolidation_strategy: 'merge' or 'summarize'
        Returns:
            ID of consolidated episode
        """
        episodes_to_consolidate = [
            self.episodes[eid] for eid in episode_ids
            if eid in self.episodes
        ]
        if not episodes_to_consolidate:
            return ""
        # Merge content
        if consolidation_strategy == 'merge':
            consolidated_content = {
                'episodes': [ep.content for ep in episodes_to_consolidate],
                'count': len(episodes_to_consolidate)
            }
        else:  # summarize
            # Simple summarization (in practice, use LLM)
            consolidated_content = {
                'summary': f"Consolidation of {len(episodes_to_consolidate)} episodes",
                'episode_types': list(set(ep.episode_type for ep in episodes_to_consolidate)),
                'time_span': (
                    max(ep.temporal_context.timestamp for ep in episodes_to_consolidate) -
                    min(ep.temporal_context.timestamp for ep in episodes_to_consolidate)
                )
            }
        # Merge tags
        all_tags = set()
        for ep in episodes_to_consolidate:
            all_tags.update(ep.tags)
        # Create temporal context (use earliest timestamp)
        earliest_episode = min(
            episodes_to_consolidate,
            key=lambda x: x.temporal_context.timestamp
        )
        # Store consolidated episode
        consolidated_id = self.store_episode(
            content=consolidated_content,
            episode_type=new_type,
            temporal_context=earliest_episode.temporal_context,
            tags=list(all_tags),
            metadata={
                'consolidated_from': episode_ids,
                'consolidation_strategy': consolidation_strategy
            }
        )
        # Link episodes
        for episode_id in episode_ids:
            if episode_id in self.episodes:
                self.episodes[episode_id].related_episodes.append(consolidated_id)
        return consolidated_id
    def _create_temporal_context(self) -> TemporalContext:
        """Create temporal context for current time"""
        import datetime
        now = time.time()
        dt = datetime.datetime.fromtimestamp(now)
        hour = dt.hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        elif 17 <= hour < 21:
            time_of_day = "evening"
        else:
            time_of_day = "night"
        return TemporalContext(
            timestamp=now,
            time_of_day=time_of_day,
            day_of_week=dt.strftime("%A")
        )
    def _index_episode(self, episode: Episode):
        """Index episode for retrieval"""
        # Index by date
        import datetime
        dt = datetime.datetime.fromtimestamp(episode.temporal_context.timestamp)
        date_key = dt.strftime("%Y-%m-%d")
        self.temporal_index[date_key].append(episode.episode_id)
        # Index by type
        self.type_index[episode.episode_type].append(episode.episode_id)
        # Index by tags
        for tag in episode.tags:
            self.tag_index[tag].append(episode.episode_id)
    def _generate_embedding(
        self,
        content: Any,
        tags: List[str]
    ) -> np.ndarray:
        """Generate embedding for content"""
        # Simple hash-based embedding for demonstration
        text = str(content) + " ".join(tags)
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    def _calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity"""
        return float(np.dot(embedding1, embedding2))
    def _calculate_context_match(
        self,
        episode: Episode,
        query_context: Dict[str, Any]
    ) -> float:
        """Calculate how well episode context matches query context"""
        matches = []
        # Tag overlap
        if 'tags' in query_context and episode.tags:
            query_tags = set(query_context['tags'])
            episode_tags = set(episode.tags)
            if query_tags and episode_tags:
                overlap = len(query_tags & episode_tags) / len(query_tags | episode_tags)
                matches.append(overlap)
        # Type match
        if 'type' in query_context:
            type_match = 1.0 if episode.episode_type == query_context['type'] else 0.0
            matches.append(type_match)
        # Spatial context match
        if 'location' in query_context and episode.spatial_context:
            location_match = 1.0 if episode.spatial_context.location == query_context['location'] else 0.0
            matches.append(location_match)
        # Emotional context match
        if 'sentiment' in query_context and episode.emotional_context:
            sentiment_diff = abs(episode.emotional_context.sentiment - query_context['sentiment'])
            sentiment_match = 1.0 - min(sentiment_diff / 2.0, 1.0)
            matches.append(sentiment_match)
        return sum(matches) / len(matches) if matches else 0.5
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        if not self.episodes:
            return {'message': 'No episodes stored'}
        # Calculate statistics
        type_counts = defaultdict(int)
        for episode in self.episodes.values():
            type_counts[episode.episode_type.value] += 1
        timestamps = [ep.temporal_context.timestamp for ep in self.episodes.values()]
        oldest = min(timestamps)
        newest = max(timestamps)
        access_counts = [ep.access_count for ep in self.episodes.values()]
        return {
            'total_episodes': len(self.episodes),
            'episodes_by_type': dict(type_counts),
            'time_span_days': (newest - oldest) / 86400,
            'avg_access_count': sum(access_counts) / len(access_counts),
            'most_accessed': max(access_counts),
            'total_tags': len(self.tag_index),
            'avg_tags_per_episode': sum(len(ep.tags) for ep in self.episodes.values()) / len(self.episodes)
        }
def main():
    """Demonstrate episodic memory retrieval"""
    print("=" * 60)
    print("Episodic Memory Retrieval Demonstration")
    print("=" * 60)
    memory = EpisodicMemory()
    print("\n1. Storing Episodes")
    print("-" * 60)
    # Store various episodes
    episodes_data = [
        {
            'content': 'User asked about Python programming',
            'type': EpisodeType.INTERACTION,
            'tags': ['programming', 'python', 'question'],
            'emotional': EmotionalContext(sentiment=0.5, importance=0.7)
        },
        {
            'content': 'Successfully helped user debug code',
            'type': EpisodeType.SUCCESS,
            'tags': ['programming', 'debugging', 'success'],
            'emotional': EmotionalContext(sentiment=0.9, importance=0.8)
        },
        {
            'content': 'User reported error in explanation',
            'type': EpisodeType.ERROR,
            'tags': ['error', 'feedback'],
            'emotional': EmotionalContext(sentiment=-0.3, importance=0.9)
        },
        {
            'content': 'Observed user struggling with async programming',
            'type': EpisodeType.OBSERVATION,
            'tags': ['programming', 'async', 'difficulty'],
            'emotional': EmotionalContext(sentiment=0.0, importance=0.6)
        },
        {
            'content': 'Completed task: Explain neural networks',
            'type': EpisodeType.TASK_COMPLETION,
            'tags': ['ai', 'neural_networks', 'explanation'],
            'emotional': EmotionalContext(sentiment=0.7, importance=0.8)
        }
    ]
    episode_ids = []
    for i, ep_data in enumerate(episodes_data):
        time.sleep(0.1)  # Small delay to create temporal separation
        episode_id = memory.store_episode(
            content=ep_data['content'],
            episode_type=ep_data['type'],
            tags=ep_data['tags'],
            emotional_context=ep_data['emotional'],
            spatial_context=SpatialContext(
                location=f"context_{i}",
                environment_type="chat"
            )
        )
        episode_ids.append(episode_id)
        print(f"Stored: {ep_data['content'][:50]}...")
    print(f"\nTotal episodes stored: {len(episode_ids)}")
    print("\n" + "=" * 60)
    print("2. Similarity-Based Retrieval")
    print("=" * 60)
    query = "User needs help with programming"
    print(f"\nQuery: '{query}'")
    results = memory.retrieve_similar(
        query=query,
        context={'tags': ['programming']},
        max_results=3
    )
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.episode.content}")
        print(f"   Type: {result.episode.episode_type.value}")
        print(f"   Overall Score: {result.overall_score:.3f}")
        print(f"   Similarity: {result.similarity_score:.3f}")
        print(f"   Context Match: {result.context_match:.3f}")
        print(f"   Age: {result.temporal_distance:.2f} seconds ago")
        print(f"   Tags: {', '.join(result.episode.tags)}")
    print("\n" + "=" * 60)
    print("3. Retrieving Recent Episodes")
    print("=" * 60)
    recent = memory.retrieve_recent(n=3)
    print("\nMost recent episodes:")
    for i, episode in enumerate(recent, 1):
        print(f"\n{i}. {episode.content}")
        print(f"   Type: {episode.episode_type.value}")
        print(f"   Time: {episode.temporal_context.time_of_day}")
        print(f"   Sentiment: {episode.emotional_context.sentiment if episode.emotional_context else 'N/A'}")
    print("\n" + "=" * 60)
    print("4. Tag-Based Retrieval")
    print("=" * 60)
    print("\nEpisodes tagged 'programming':")
    programming_episodes = memory.retrieve_by_tags(['programming'])
    for episode in programming_episodes:
        print(f"  - {episode.content}")
        print(f"    Additional tags: {', '.join(ep_tag for ep_tag in episode.tags if ep_tag != 'programming')}")
    print("\nEpisodes with both 'programming' AND 'success':")
    success_programming = memory.retrieve_by_tags(
        ['programming', 'success'],
        match_all=True
    )
    for episode in success_programming:
        print(f"  - {episode.content}")
    print("\n" + "=" * 60)
    print("5. Finding Related Episodes")
    print("=" * 60)
    source_episode_id = episode_ids[0]
    source_episode = memory.episodes[source_episode_id]
    print(f"\nFinding episodes related to:")
    print(f"'{source_episode.content}'")
    related = memory.find_related_episodes(source_episode_id, max_results=3)
    for i, result in enumerate(related, 1):
        print(f"\n{i}. {result.episode.content}")
        print(f"   Similarity: {result.similarity_score:.3f}")
        print(f"   Shared tags: {set(source_episode.tags) & set(result.episode.tags)}")
    print("\n" + "=" * 60)
    print("6. Temporal Retrieval")
    print("=" * 60)
    # Get time range for last 1 second
    end_time = time.time()
    start_time = end_time - 1.0
    print(f"\nEpisodes in last 1 second:")
    temporal_episodes = memory.retrieve_by_time_range(
        start_time,
        end_time
    )
    for episode in temporal_episodes:
        print(f"  - {episode.content[:60]}...")
        print(f"    Type: {episode.episode_type.value}")
    print("\n" + "=" * 60)
    print("7. Episode Consolidation")
    print("=" * 60)
    # Consolidate programming-related episodes
    programming_ids = [ep.episode_id for ep in programming_episodes[:3]]
    print(f"\nConsolidating {len(programming_ids)} programming episodes...")
    consolidated_id = memory.consolidate_episodes(
        programming_ids,
        EpisodeType.OBSERVATION,
        consolidation_strategy='summarize'
    )
    if consolidated_id:
        consolidated = memory.episodes[consolidated_id]
        print(f"\nConsolidated episode created:")
        print(f"  ID: {consolidated_id}")
        print(f"  Content: {consolidated.content}")
        print(f"  Tags: {', '.join(consolidated.tags)}")
        print(f"  Metadata: {consolidated.metadata}")
    print("\n" + "=" * 60)
    print("8. Access Patterns")
    print("=" * 60)
    # Retrieve same query multiple times to build access patterns
    for _ in range(3):
        memory.retrieve_similar("programming help", max_results=2)
    most_accessed = memory.retrieve_most_accessed(n=3)
    print("\nMost frequently accessed episodes:")
    for i, episode in enumerate(most_accessed, 1):
        print(f"\n{i}. {episode.content[:60]}...")
        print(f"   Access count: {episode.access_count}")
        print(f"   Last accessed: {time.time() - episode.last_accessed:.2f}s ago")
    print("\n" + "=" * 60)
    print("9. Memory Statistics")
    print("=" * 60)
    stats = memory.get_statistics()
    print(f"\nTotal Episodes: {stats['total_episodes']}")
    print(f"Time Span: {stats['time_span_days']:.4f} days")
    print(f"Average Access Count: {stats['avg_access_count']:.2f}")
    print(f"Total Tags: {stats['total_tags']}")
    print(f"Average Tags per Episode: {stats['avg_tags_per_episode']:.2f}")
    print("\nEpisodes by Type:")
    for ep_type, count in stats['episodes_by_type'].items():
        print(f"  {ep_type}: {count}")
    print("\n" + "=" * 60)
    print("10. Context-Aware Retrieval")
    print("=" * 60)
    # Retrieve with emotional context
    print("\nRetrieving positive experiences:")
    positive_results = memory.retrieve_similar(
        query="successful interaction",
        context={
            'sentiment': 0.8,
            'tags': ['success']
        },
        max_results=3
    )
    for result in positive_results:
        emotion = result.episode.emotional_context
        if emotion:
            print(f"\n  - {result.episode.content}")
            print(f"    Sentiment: {emotion.sentiment:.2f}")
            print(f"    Importance: {emotion.importance:.2f}")
            print(f"    Match score: {result.context_match:.3f}")
    print("\n" + "=" * 60)
    print("Episodic Memory Retrieval demonstration complete!")
    print("=" * 60)
if __name__ == "__main__":
    main()
