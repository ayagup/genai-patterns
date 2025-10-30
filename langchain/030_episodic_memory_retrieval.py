"""
Pattern 030: Episodic Memory & Retrieval

Description:
    Episodic memory stores and retrieves specific experiences or events with temporal
    and contextual information. This pattern enables agents to recall relevant past
    experiences based on similarity, temporal proximity, and contextual relevance,
    supporting experience-based reasoning and learning.

Components:
    - Episode: Time-stamped experience with context, actions, and outcomes
    - Episode Store: Storage system for episodes with indexing
    - Retrieval System: Similarity-based and temporal filtering
    - Context Matcher: Identifies relevant episodes for current situation
    - Experience Reconstructor: Rebuilds complete episode details

Use Cases:
    - Learning from past interactions
    - Experience-based decision making
    - Personalization through history
    - Debugging and analysis
    - Temporal reasoning

LangChain Implementation:
    Uses vector similarity search for content-based retrieval combined with
    temporal and contextual filtering to find relevant past experiences.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import math
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class EpisodeType(Enum):
    """Types of episodes that can be stored."""
    CONVERSATION = "conversation"
    TASK_EXECUTION = "task_execution"
    DECISION = "decision"
    ERROR = "error"
    SUCCESS = "success"
    LEARNING = "learning"
    INTERACTION = "interaction"


@dataclass
class Episode:
    """
    Represents a single episodic memory.
    
    An episode captures a complete experience including context,
    actions taken, outcomes, and temporal information.
    """
    id: str
    type: EpisodeType
    timestamp: datetime
    context: Dict[str, Any]  # Situational context
    content: str  # Main content/description
    actions: List[str] = field(default_factory=list)  # Actions taken
    outcome: Optional[str] = None  # Result of the episode
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5  # 0-1 scale
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_text(self) -> str:
        """Convert episode to searchable text representation."""
        parts = [
            f"Type: {self.type.value}",
            f"Content: {self.content}",
        ]
        
        if self.context:
            parts.append(f"Context: {json.dumps(self.context)}")
        
        if self.actions:
            parts.append(f"Actions: {', '.join(self.actions)}")
        
        if self.outcome:
            parts.append(f"Outcome: {self.outcome}")
        
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        
        return " | ".join(parts)
    
    def mark_accessed(self):
        """Mark episode as accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class RetrievalQuery:
    """Query for retrieving episodes."""
    query: str
    episode_types: Optional[List[EpisodeType]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    min_importance: float = 0.0
    max_results: int = 5
    tags: Optional[List[str]] = None


@dataclass
class RetrievalResult:
    """Result of episode retrieval with relevance scoring."""
    episode: Episode
    relevance_score: float  # Overall relevance (0-1)
    similarity_score: float  # Content similarity (0-1)
    recency_score: float  # Temporal relevance (0-1)
    importance_score: float  # Importance weight (0-1)


class EpisodicMemoryStore:
    """
    Stores and retrieves episodic memories.
    
    Supports multiple retrieval strategies:
    - Similarity-based (content matching)
    - Temporal filtering (time-based)
    - Importance weighting
    - Tag-based filtering
    """
    
    def __init__(self, recency_decay_hours: float = 168.0):  # Default 1 week
        self.episodes: Dict[str, Episode] = {}
        self.recency_decay_hours = recency_decay_hours
        self._next_id = 1
    
    def add_episode(self, episode: Episode) -> str:
        """Add an episode to memory."""
        if not episode.id:
            episode.id = f"ep_{self._next_id:04d}"
            self._next_id += 1
        
        self.episodes[episode.id] = episode
        return episode.id
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve a specific episode by ID."""
        episode = self.episodes.get(episode_id)
        if episode:
            episode.mark_accessed()
        return episode
    
    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Retrieve relevant episodes based on query.
        
        Combines multiple scoring factors:
        - Similarity to query content
        - Recency (temporal decay)
        - Importance weighting
        """
        candidates = self._filter_episodes(query)
        
        if not candidates:
            return []
        
        # Score each candidate
        results = []
        current_time = datetime.now()
        
        for episode in candidates:
            # Similarity score (simple keyword matching)
            similarity_score = self._compute_similarity(query.query, episode)
            
            # Recency score (exponential decay)
            recency_score = self._compute_recency(episode.timestamp, current_time)
            
            # Importance score
            importance_score = episode.importance
            
            # Combined relevance score (weighted average)
            relevance_score = (
                0.5 * similarity_score +
                0.3 * recency_score +
                0.2 * importance_score
            )
            
            result = RetrievalResult(
                episode=episode,
                relevance_score=relevance_score,
                similarity_score=similarity_score,
                recency_score=recency_score,
                importance_score=importance_score
            )
            results.append(result)
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Mark accessed
        for result in results[:query.max_results]:
            result.episode.mark_accessed()
        
        return results[:query.max_results]
    
    def _filter_episodes(self, query: RetrievalQuery) -> List[Episode]:
        """Apply filters to episodes."""
        filtered = []
        
        for episode in self.episodes.values():
            # Type filter
            if query.episode_types and episode.type not in query.episode_types:
                continue
            
            # Importance filter
            if episode.importance < query.min_importance:
                continue
            
            # Time range filter
            if query.time_range:
                start_time, end_time = query.time_range
                if not (start_time <= episode.timestamp <= end_time):
                    continue
            
            # Tag filter
            if query.tags:
                if not any(tag in episode.tags for tag in query.tags):
                    continue
            
            filtered.append(episode)
        
        return filtered
    
    def _compute_similarity(self, query: str, episode: Episode) -> float:
        """
        Compute similarity between query and episode.
        Simple keyword-based similarity (can be replaced with embeddings).
        """
        query_words = set(query.lower().split())
        episode_text = episode.to_text().lower()
        episode_words = set(episode_text.split())
        
        if not query_words:
            return 0.0
        
        # Jaccard similarity
        intersection = query_words & episode_words
        union = query_words | episode_words
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _compute_recency(self, timestamp: datetime, current_time: datetime) -> float:
        """
        Compute recency score with exponential decay.
        More recent episodes score higher.
        """
        hours_ago = (current_time - timestamp).total_seconds() / 3600
        
        if hours_ago < 0:
            return 1.0
        
        # Exponential decay
        decay_rate = math.log(2) / self.recency_decay_hours
        return math.exp(-decay_rate * hours_ago)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about episodic memory."""
        if not self.episodes:
            return {
                "total_episodes": 0,
                "episode_types": {},
                "avg_importance": 0.0,
                "most_accessed": None
            }
        
        type_counts = {}
        for episode in self.episodes.values():
            type_counts[episode.type.value] = type_counts.get(episode.type.value, 0) + 1
        
        avg_importance = sum(e.importance for e in self.episodes.values()) / len(self.episodes)
        
        most_accessed = max(self.episodes.values(), key=lambda e: e.access_count)
        
        return {
            "total_episodes": len(self.episodes),
            "episode_types": type_counts,
            "avg_importance": avg_importance,
            "most_accessed": {
                "id": most_accessed.id,
                "access_count": most_accessed.access_count,
                "content": most_accessed.content[:50] + "..."
            }
        }


class EpisodicMemoryAgent:
    """
    Agent that uses episodic memory to inform its responses.
    
    Capabilities:
    - Store experiences as episodes
    - Retrieve relevant past experiences
    - Learn from past successes and failures
    - Provide context-aware responses
    """
    
    def __init__(self, temperature: float = 0.7):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        self.memory_store = EpisodicMemoryStore()
        
        # Prompt for responding with episodic context
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant with episodic memory.
You can remember past experiences and use them to inform your responses.

Relevant past experiences:
{episodes}

Use these experiences to provide context-aware, informed responses.
Reference past experiences when relevant."""),
            ("user", "{query}")
        ])
        
        # Prompt for extracting learnings
        self.learning_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze this experience and extract key learnings.

Experience: {experience}

What are the key takeaways? What worked? What didn't?
Provide 2-3 concise learnings."""),
            ("user", "Extract learnings")
        ])
    
    def record_episode(
        self,
        episode_type: EpisodeType,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        actions: Optional[List[str]] = None,
        outcome: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> str:
        """Record a new episodic memory."""
        episode = Episode(
            id="",  # Will be auto-generated
            type=episode_type,
            timestamp=datetime.now(),
            context=context or {},
            content=content,
            actions=actions or [],
            outcome=outcome,
            importance=importance,
            tags=tags or []
        )
        
        return self.memory_store.add_episode(episode)
    
    def respond_with_memory(self, query: str) -> Tuple[str, List[Episode]]:
        """
        Generate response using relevant episodic memories.
        
        Returns:
            Tuple of (response, relevant_episodes)
        """
        # Retrieve relevant episodes
        retrieval_query = RetrievalQuery(
            query=query,
            max_results=3
        )
        
        results = self.memory_store.retrieve(retrieval_query)
        
        # Format episodes for context
        if results:
            episodes_text = "\n\n".join([
                f"Episode {i+1} (relevance: {r.relevance_score:.2f}):\n"
                f"Type: {r.episode.type.value}\n"
                f"Time: {r.episode.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
                f"Content: {r.episode.content}\n"
                f"Outcome: {r.episode.outcome or 'N/A'}"
                for i, r in enumerate(results)
            ])
        else:
            episodes_text = "No relevant past experiences found."
        
        # Generate response
        chain = self.response_prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "episodes": episodes_text,
            "query": query
        })
        
        # Extract episodes
        episodes = [r.episode for r in results]
        
        return response, episodes
    
    def learn_from_experience(self, episode_id: str) -> Optional[str]:
        """
        Extract learnings from a specific episode.
        """
        episode = self.memory_store.get_episode(episode_id)
        
        if not episode:
            return None
        
        # Format experience
        experience_text = (
            f"Type: {episode.type.value}\n"
            f"Context: {json.dumps(episode.context)}\n"
            f"Content: {episode.content}\n"
            f"Actions: {', '.join(episode.actions)}\n"
            f"Outcome: {episode.outcome or 'N/A'}"
        )
        
        # Extract learnings
        chain = self.learning_prompt | self.llm | StrOutputParser()
        learnings = chain.invoke({"experience": experience_text})
        
        return learnings
    
    def get_similar_experiences(
        self,
        reference_episode_id: str,
        max_results: int = 5
    ) -> List[RetrievalResult]:
        """
        Find episodes similar to a reference episode.
        """
        reference = self.memory_store.get_episode(reference_episode_id)
        
        if not reference:
            return []
        
        query = RetrievalQuery(
            query=reference.content,
            episode_types=[reference.type],
            max_results=max_results
        )
        
        results = self.memory_store.retrieve(query)
        
        # Exclude the reference episode itself
        return [r for r in results if r.episode.id != reference_episode_id]
    
    def get_recent_episodes(
        self,
        hours: int = 24,
        episode_type: Optional[EpisodeType] = None
    ) -> List[Episode]:
        """Get episodes from recent time period."""
        current_time = datetime.now()
        start_time = current_time - timedelta(hours=hours)
        
        query = RetrievalQuery(
            query="",  # Match all
            episode_types=[episode_type] if episode_type else None,
            time_range=(start_time, current_time),
            max_results=100  # Get all recent
        )
        
        results = self.memory_store.retrieve(query)
        return [r.episode for r in results]


def demonstrate_episodic_memory():
    """
    Demonstrates episodic memory with storage, retrieval,
    and experience-based reasoning.
    """
    print("=" * 80)
    print("EPISODIC MEMORY & RETRIEVAL DEMONSTRATION")
    print("=" * 80)
    
    # Create agent
    agent = EpisodicMemoryAgent()
    
    # Test 1: Record various episodes
    print("\n" + "=" * 80)
    print("Test 1: Recording Episodes")
    print("=" * 80)
    
    episodes_to_record = [
        {
            "type": EpisodeType.CONVERSATION,
            "content": "User asked about Python programming best practices",
            "context": {"topic": "programming", "language": "Python"},
            "actions": ["Provided advice on PEP 8", "Recommended tools"],
            "outcome": "User found the advice helpful",
            "importance": 0.7,
            "tags": ["programming", "python", "advice"]
        },
        {
            "type": EpisodeType.TASK_EXECUTION,
            "content": "Helped user debug a database connection issue",
            "context": {"topic": "debugging", "database": "PostgreSQL"},
            "actions": ["Checked connection string", "Verified credentials", "Tested connection"],
            "outcome": "Successfully resolved the connection issue",
            "importance": 0.8,
            "tags": ["debugging", "database", "postgresql"]
        },
        {
            "type": EpisodeType.ERROR,
            "content": "Failed to parse user's malformed JSON input",
            "context": {"error_type": "JSONDecodeError"},
            "actions": ["Attempted to parse JSON", "Provided error message"],
            "outcome": "User corrected the JSON format",
            "importance": 0.6,
            "tags": ["error", "json", "parsing"]
        },
        {
            "type": EpisodeType.SUCCESS,
            "content": "Successfully explained machine learning concepts to beginner",
            "context": {"topic": "machine learning", "user_level": "beginner"},
            "actions": ["Used simple analogies", "Provided examples", "Answered follow-up questions"],
            "outcome": "User understood the concepts clearly",
            "importance": 0.9,
            "tags": ["teaching", "machine_learning", "success"]
        },
        {
            "type": EpisodeType.CONVERSATION,
            "content": "Discussed best practices for API design",
            "context": {"topic": "api design", "focus": "RESTful APIs"},
            "actions": ["Explained REST principles", "Discussed versioning", "Covered authentication"],
            "outcome": "User appreciated the comprehensive explanation",
            "importance": 0.75,
            "tags": ["api", "design", "rest"]
        }
    ]
    
    episode_ids = []
    for ep_data in episodes_to_record:
        ep_id = agent.record_episode(**ep_data)
        episode_ids.append(ep_id)
        print(f"✓ Recorded episode: {ep_data['content'][:50]}...")
    
    # Show statistics
    stats = agent.memory_store.get_statistics()
    print(f"\nMemory statistics:")
    print(f"  Total episodes: {stats['total_episodes']}")
    print(f"  Episode types: {stats['episode_types']}")
    print(f"  Average importance: {stats['avg_importance']:.2f}")
    
    # Test 2: Query with episodic memory
    print("\n" + "=" * 80)
    print("Test 2: Responding with Episodic Memory")
    print("=" * 80)
    
    queries = [
        "How should I approach learning programming?",
        "I'm having database connection problems, any advice?",
        "Can you explain machine learning to me?"
    ]
    
    for query in queries:
        print(f"\nUser: {query}")
        response, relevant_episodes = agent.respond_with_memory(query)
        print(f"\nAgent: {response}")
        print(f"\nRelevant past experiences: {len(relevant_episodes)}")
        for ep in relevant_episodes:
            print(f"  - {ep.type.value}: {ep.content[:60]}...")
    
    # Test 3: Learn from experiences
    print("\n" + "=" * 80)
    print("Test 3: Learning from Past Experiences")
    print("=" * 80)
    
    print(f"\nAnalyzing episode: {episode_ids[1]}")
    learnings = agent.learn_from_experience(episode_ids[1])
    if learnings:
        print(f"Learnings extracted:\n{learnings}")
    
    # Test 4: Find similar experiences
    print("\n" + "=" * 80)
    print("Test 4: Finding Similar Experiences")
    print("=" * 80)
    
    reference_id = episode_ids[0]
    print(f"\nFinding experiences similar to: {episode_ids[0]}")
    similar = agent.get_similar_experiences(reference_id, max_results=3)
    
    if similar:
        print(f"Found {len(similar)} similar experiences:")
        for result in similar:
            print(f"  - Relevance: {result.relevance_score:.2f}")
            print(f"    {result.episode.content[:70]}...")
    else:
        print("No similar experiences found")
    
    # Test 5: Recent activity
    print("\n" + "=" * 80)
    print("Test 5: Recent Activity Review")
    print("=" * 80)
    
    recent = agent.get_recent_episodes(hours=24)
    print(f"\nEpisodes from last 24 hours: {len(recent)}")
    for ep in recent[:3]:
        print(f"  - [{ep.timestamp.strftime('%H:%M')}] {ep.type.value}: {ep.content[:60]}...")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Episodic Memory & Retrieval provides:
✓ Time-stamped experience storage
✓ Multi-factor relevance scoring
✓ Context-aware retrieval
✓ Learning from past experiences
✓ Temporal and semantic filtering

This pattern excels at:
- Personalizing responses based on history
- Learning from successes and failures
- Providing context-aware recommendations
- Supporting temporal reasoning
- Building experiential knowledge

Key advantages:
1. Rich contextual information
2. Temporal awareness
3. Experience-based learning
4. Flexible retrieval strategies
5. Automatic relevance scoring

Retrieval factors:
- Similarity: Content matching (50% weight)
- Recency: Temporal decay (30% weight)
- Importance: Episode significance (20% weight)

Use episodic memory when you need to:
- Remember specific past interactions
- Learn from historical outcomes
- Provide personalized experiences
- Support case-based reasoning
- Track agent development over time
""")


if __name__ == "__main__":
    demonstrate_episodic_memory()
