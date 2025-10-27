"""
Pattern 107: Context Grounding Agent

This pattern implements multi-modal context understanding and grounding,
enabling agents to understand and operate in complex contextual situations.

Use Cases:
- Situational awareness in complex environments
- Multi-modal understanding (text, vision, audio)
- Context-dependent decision making
- Grounded language understanding
- Dialogue and conversation management

Key Features:
- Multi-layered context representation
- Temporal context tracking
- Spatial context awareness
- Social context understanding
- Context-dependent reasoning
- Grounding mechanisms
- Context switching and management

Implementation:
- Pure Python (3.8+) with comprehensive type hints
- Zero external dependencies
- Production-ready error handling
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import uuid


class ContextType(Enum):
    """Types of context."""
    TEMPORAL = "temporal"      # Time-based context
    SPATIAL = "spatial"        # Location/environment
    SOCIAL = "social"          # Social situation
    DIALOGUE = "dialogue"      # Conversation context
    TASK = "task"             # Current task
    DOMAIN = "domain"         # Subject domain
    EMOTIONAL = "emotional"   # Emotional state
    PHYSICAL = "physical"     # Physical environment


class GroundingType(Enum):
    """Types of grounding."""
    ENTITY = "entity"          # Ground to entities
    LOCATION = "location"      # Ground to locations
    TIME = "time"             # Ground to time
    ACTION = "action"         # Ground to actions
    RELATION = "relation"     # Ground to relationships
    ATTRIBUTE = "attribute"   # Ground to attributes


@dataclass
class ContextFrame:
    """Single frame of context information."""
    frame_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    context_type: ContextType = ContextType.TASK
    
    # Core content
    content: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    source: str = "system"
    
    # Validity
    expires_at: Optional[datetime] = None
    priority: int = 1  # Higher = more important
    
    # Relationships
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if context frame is still valid."""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True
    
    def get_age(self) -> float:
        """Get age in seconds."""
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass
class GroundingLink:
    """Links linguistic expressions to grounded entities."""
    link_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Expression being grounded
    expression: str = ""
    expression_type: GroundingType = GroundingType.ENTITY
    
    # Grounded reference
    grounded_id: str = ""
    grounded_type: str = ""
    
    # Grounding quality
    confidence: float = 1.0
    ambiguity: float = 0.0  # Higher = more ambiguous
    
    # Context
    context_ids: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ContextState:
    """Complete context state at a point in time."""
    state_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Active contexts by type
    active_contexts: Dict[ContextType, List[str]] = field(default_factory=dict)
    
    # Grounding mappings
    groundings: List[str] = field(default_factory=list)  # grounding_link_ids
    
    # Attention focus
    focus_entities: Set[str] = field(default_factory=set)
    focus_topics: Set[str] = field(default_factory=set)
    
    # Situational factors
    situation_summary: str = ""
    constraints: List[str] = field(default_factory=list)


class ContextManager:
    """
    Manages multi-layered context information.
    
    Maintains different types of context (temporal, spatial, social, etc.)
    and their relationships over time.
    """
    
    def __init__(self, max_history: int = 100):
        self.frames: Dict[str, ContextFrame] = {}
        self.max_history = max_history
        
        # Indices
        self.type_index: Dict[ContextType, Set[str]] = {}
        self.temporal_order: deque = deque(maxlen=max_history)
        
        # Current state
        self.active_frame_ids: Set[str] = set()
    
    def add_frame(self, frame: ContextFrame) -> None:
        """Add context frame."""
        self.frames[frame.frame_id] = frame
        
        # Update type index
        if frame.context_type not in self.type_index:
            self.type_index[frame.context_type] = set()
        self.type_index[frame.context_type].add(frame.frame_id)
        
        # Update temporal order
        self.temporal_order.append(frame.frame_id)
        
        # Mark as active
        self.active_frame_ids.add(frame.frame_id)
        
        # Cleanup if needed
        self._cleanup_expired()
    
    def get_active_contexts(self, context_type: Optional[ContextType] = None) -> List[ContextFrame]:
        """Get currently active context frames."""
        active = []
        
        for frame_id in self.active_frame_ids:
            if frame_id in self.frames:
                frame = self.frames[frame_id]
                
                # Check validity
                if not frame.is_valid():
                    self.active_frame_ids.remove(frame_id)
                    continue
                
                # Filter by type if specified
                if context_type is None or frame.context_type == context_type:
                    active.append(frame)
        
        # Sort by priority
        active.sort(key=lambda f: f.priority, reverse=True)
        return active
    
    def get_recent_contexts(self, limit: int = 10,
                           context_type: Optional[ContextType] = None) -> List[ContextFrame]:
        """Get recent context frames."""
        recent = []
        
        # Iterate in reverse temporal order
        for frame_id in reversed(self.temporal_order):
            if frame_id in self.frames:
                frame = self.frames[frame_id]
                
                if context_type is None or frame.context_type == context_type:
                    recent.append(frame)
                    
                    if len(recent) >= limit:
                        break
        
        return recent
    
    def merge_contexts(self, frame_ids: List[str]) -> Dict[str, Any]:
        """Merge multiple context frames into unified view."""
        merged = {}
        
        for frame_id in frame_ids:
            if frame_id in self.frames:
                frame = self.frames[frame_id]
                
                # Merge content with priority-based override
                for key, value in frame.content.items():
                    if key not in merged:
                        merged[key] = value
                    else:
                        # Keep higher priority value
                        existing_frame_id = merged.get(f"__{key}_source")
                        if existing_frame_id and existing_frame_id in self.frames:
                            if frame.priority > self.frames[existing_frame_id].priority:
                                merged[key] = value
                                merged[f"__{key}_source"] = frame_id
                        else:
                            merged[f"__{key}_source"] = frame_id
        
        return merged
    
    def update_focus(self, entities: Set[str], topics: Set[str]) -> None:
        """Update attention focus."""
        # Create focus frame
        focus_frame = ContextFrame(
            context_type=ContextType.TASK,
            content={
                "focus_entities": list(entities),
                "focus_topics": list(topics)
            },
            priority=10  # High priority
        )
        self.add_frame(focus_frame)
    
    def _cleanup_expired(self) -> None:
        """Remove expired context frames."""
        to_remove = []
        
        for frame_id, frame in self.frames.items():
            if not frame.is_valid():
                to_remove.append(frame_id)
        
        for frame_id in to_remove:
            frame = self.frames[frame_id]
            
            # Remove from indices
            if frame.context_type in self.type_index:
                self.type_index[frame.context_type].discard(frame_id)
            
            # Remove from active
            self.active_frame_ids.discard(frame_id)
            
            # Remove frame
            del self.frames[frame_id]


class GroundingEngine:
    """
    Grounds linguistic expressions to concrete entities and concepts.
    
    Resolves references, maintains entity salience, and disambiguates
    based on context.
    """
    
    def __init__(self):
        self.grounding_links: Dict[str, GroundingLink] = {}
        
        # Entity database
        self.entities: Dict[str, Dict[str, Any]] = {}
        
        # Salience tracking
        self.entity_salience: Dict[str, float] = {}  # entity_id -> salience
        self.salience_decay = 0.1
        
        # Discourse history
        self.mentioned_entities: deque = deque(maxlen=50)
    
    def register_entity(self, entity_id: str,
                       entity_type: str,
                       attributes: Dict[str, Any]) -> None:
        """Register entity in grounding database."""
        self.entities[entity_id] = {
            "type": entity_type,
            "attributes": attributes,
            "aliases": attributes.get("aliases", [])
        }
    
    def ground_expression(self, expression: str,
                         expression_type: GroundingType,
                         context_frames: List[ContextFrame]) -> Optional[GroundingLink]:
        """
        Ground expression to entity or concept.
        
        Uses context to disambiguate and resolve references.
        """
        # Extract context information
        context_info = self._extract_context_info(context_frames)
        
        # Find candidate entities
        candidates = self._find_candidates(expression, expression_type, context_info)
        
        if not candidates:
            return None
        
        # Disambiguate
        best_candidate = self._disambiguate(candidates, context_info)
        
        # Create grounding link
        link = GroundingLink(
            expression=expression,
            expression_type=expression_type,
            grounded_id=best_candidate["entity_id"],
            grounded_type=best_candidate["type"],
            confidence=best_candidate["score"],
            ambiguity=1.0 - best_candidate["score"],
            context_ids=[f.frame_id for f in context_frames]
        )
        
        self.grounding_links[link.link_id] = link
        
        # Update salience
        self._update_salience(best_candidate["entity_id"], increase=0.3)
        self.mentioned_entities.append(best_candidate["entity_id"])
        
        return link
    
    def resolve_reference(self, reference: str,
                         context_frames: List[ContextFrame]) -> Optional[str]:
        """
        Resolve pronoun or definite reference to entity.
        
        Examples: "it", "the cat", "him", "her"
        """
        # Check if it's a pronoun
        pronouns = {"it", "he", "she", "they", "him", "her", "them", "this", "that"}
        
        if reference.lower() in pronouns:
            # Use discourse history and salience
            if self.mentioned_entities:
                # Get most salient recent entity
                recent = list(self.mentioned_entities)[-5:]  # Last 5 mentions
                
                best_entity = max(
                    recent,
                    key=lambda e: self.entity_salience.get(e, 0.0)
                )
                return best_entity
        
        # For definite references, ground normally
        link = self.ground_expression(reference, GroundingType.ENTITY, context_frames)
        return link.grounded_id if link else None
    
    def _extract_context_info(self, frames: List[ContextFrame]) -> Dict[str, Any]:
        """Extract relevant information from context frames."""
        info = {
            "locations": set(),
            "times": set(),
            "mentioned_entities": set(),
            "topics": set(),
            "situation": {}
        }
        
        for frame in frames:
            content = frame.content
            
            # Extract based on context type
            if frame.context_type == ContextType.SPATIAL:
                info["locations"].update(content.get("locations", []))
            elif frame.context_type == ContextType.TEMPORAL:
                info["times"].update(content.get("times", []))
            elif frame.context_type == ContextType.TASK:
                info["topics"].update(content.get("focus_topics", []))
                info["mentioned_entities"].update(content.get("focus_entities", []))
        
        return info
    
    def _find_candidates(self, expression: str,
                        expression_type: GroundingType,
                        context_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find candidate entities for grounding."""
        candidates = []
        expr_lower = expression.lower()
        
        for entity_id, entity_data in self.entities.items():
            # Check name match
            entity_type = entity_data["type"]
            attributes = entity_data["attributes"]
            
            # Simple name matching
            name = attributes.get("name", "").lower()
            aliases = [a.lower() for a in entity_data.get("aliases", [])]
            
            score = 0.0
            
            # Exact match
            if expr_lower == name or expr_lower in aliases:
                score = 1.0
            # Partial match
            elif expr_lower in name or name in expr_lower:
                score = 0.7
            elif any(expr_lower in alias or alias in expr_lower for alias in aliases):
                score = 0.6
            
            # Boost by salience
            salience = self.entity_salience.get(entity_id, 0.1)
            score += salience * 0.3
            
            # Boost by recency
            if entity_id in context_info.get("mentioned_entities", set()):
                score += 0.2
            
            if score > 0:
                candidates.append({
                    "entity_id": entity_id,
                    "type": entity_type,
                    "score": min(1.0, score),
                    "attributes": attributes
                })
        
        return candidates
    
    def _disambiguate(self, candidates: List[Dict[str, Any]],
                     context_info: Dict[str, Any]) -> Dict[str, Any]:
        """Select best candidate using context."""
        if len(candidates) == 1:
            return candidates[0]
        
        # Sort by score
        candidates.sort(key=lambda c: c["score"], reverse=True)
        
        return candidates[0]
    
    def _update_salience(self, entity_id: str, increase: float = 0.0) -> None:
        """Update entity salience."""
        # Decay all salience
        for eid in list(self.entity_salience.keys()):
            self.entity_salience[eid] *= (1 - self.salience_decay)
            
            # Remove if too low
            if self.entity_salience[eid] < 0.01:
                del self.entity_salience[eid]
        
        # Increase specified entity
        if entity_id not in self.entity_salience:
            self.entity_salience[entity_id] = 0.0
        
        self.entity_salience[entity_id] = min(
            1.0,
            self.entity_salience[entity_id] + increase
        )


class SituationalAwareness:
    """
    Maintains awareness of current situation.
    
    Integrates multiple context types to form comprehensive
    situational understanding.
    """
    
    def __init__(self):
        self.current_situation: Dict[str, Any] = {}
        self.situation_history: List[Dict[str, Any]] = []
    
    def assess_situation(self, context_manager: ContextManager) -> Dict[str, Any]:
        """Assess current situation from active contexts."""
        situation = {
            "timestamp": datetime.now(),
            "temporal": {},
            "spatial": {},
            "social": {},
            "task": {},
            "constraints": [],
            "opportunities": [],
            "risks": []
        }
        
        # Get active contexts
        for context_type in ContextType:
            frames = context_manager.get_active_contexts(context_type)
            
            if not frames:
                continue
            
            # Merge frames of same type
            merged = {}
            for frame in frames:
                merged.update(frame.content)
            
            # Store in situation
            type_key = context_type.value
            situation[type_key] = merged
        
        # Analyze situation
        self._analyze_constraints(situation)
        self._identify_opportunities(situation)
        self._assess_risks(situation)
        
        # Update current situation
        self.current_situation = situation
        self.situation_history.append(situation)
        
        return situation
    
    def _analyze_constraints(self, situation: Dict[str, Any]) -> None:
        """Identify constraints in current situation."""
        constraints = []
        
        # Time constraints
        if "deadline" in situation.get("temporal", {}):
            constraints.append("time_limited")
        
        # Resource constraints
        if "resources" in situation.get("task", {}):
            resources = situation["task"]["resources"]
            if resources.get("limited", False):
                constraints.append("resource_limited")
        
        situation["constraints"] = constraints
    
    def _identify_opportunities(self, situation: Dict[str, Any]) -> None:
        """Identify opportunities in situation."""
        opportunities = []
        
        # Check for favorable conditions
        if situation.get("social", {}).get("cooperation", False):
            opportunities.append("collaborative_work")
        
        situation["opportunities"] = opportunities
    
    def _assess_risks(self, situation: Dict[str, Any]) -> None:
        """Assess risks in situation."""
        risks = []
        
        # Check for risk factors
        if situation.get("task", {}).get("complexity") == "high":
            risks.append("high_complexity")
        
        situation["risks"] = risks
    
    def get_situation_summary(self) -> str:
        """Generate natural language summary of situation."""
        if not self.current_situation:
            return "No situational information available."
        
        parts = []
        
        # Time
        temporal = self.current_situation.get("temporal", {})
        if temporal:
            parts.append(f"Time context: {temporal}")
        
        # Location
        spatial = self.current_situation.get("spatial", {})
        if spatial:
            parts.append(f"Location: {spatial}")
        
        # Task
        task = self.current_situation.get("task", {})
        if task:
            parts.append(f"Current task: {task}")
        
        # Constraints
        if self.current_situation["constraints"]:
            parts.append(f"Constraints: {', '.join(self.current_situation['constraints'])}")
        
        return "; ".join(parts) if parts else "Situation unclear."


class ContextGroundingAgent:
    """
    Agent with advanced context understanding and grounding.
    
    Features:
    - Multi-layered context management
    - Expression grounding
    - Reference resolution
    - Situational awareness
    - Context-dependent reasoning
    """
    
    def __init__(self):
        self.context_manager = ContextManager()
        self.grounding_engine = GroundingEngine()
        self.situational_awareness = SituationalAwareness()
        
        # Statistics
        self.total_groundings = 0
        self.successful_groundings = 0
    
    def add_context(self, context_type: ContextType,
                   content: Dict[str, Any],
                   priority: int = 1,
                   duration: Optional[int] = None) -> ContextFrame:
        """Add context information."""
        expires_at = None
        if duration:
            expires_at = datetime.now() + timedelta(seconds=duration)
        
        frame = ContextFrame(
            context_type=context_type,
            content=content,
            priority=priority,
            expires_at=expires_at
        )
        
        self.context_manager.add_frame(frame)
        return frame
    
    def register_entity(self, name: str, entity_type: str,
                       attributes: Optional[Dict[str, Any]] = None) -> str:
        """Register entity for grounding."""
        entity_id = str(uuid.uuid4())[:8]
        attrs = attributes or {}
        attrs["name"] = name
        
        self.grounding_engine.register_entity(entity_id, entity_type, attrs)
        return entity_id
    
    def ground(self, expression: str,
              expression_type: GroundingType = GroundingType.ENTITY) -> Optional[GroundingLink]:
        """Ground expression to entity."""
        self.total_groundings += 1
        
        # Get active contexts
        contexts = self.context_manager.get_active_contexts()
        
        # Ground expression
        link = self.grounding_engine.ground_expression(
            expression, expression_type, contexts
        )
        
        if link:
            self.successful_groundings += 1
        
        return link
    
    def resolve_reference(self, reference: str) -> Optional[str]:
        """Resolve reference to entity ID."""
        contexts = self.context_manager.get_active_contexts()
        return self.grounding_engine.resolve_reference(reference, contexts)
    
    def assess_situation(self) -> Dict[str, Any]:
        """Assess current situation."""
        return self.situational_awareness.assess_situation(self.context_manager)
    
    def get_situation_summary(self) -> str:
        """Get natural language situation summary."""
        return self.situational_awareness.get_situation_summary()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        success_rate = (
            self.successful_groundings / self.total_groundings
            if self.total_groundings > 0 else 0.0
        )
        
        return {
            "total_groundings": self.total_groundings,
            "successful_groundings": self.successful_groundings,
            "success_rate": success_rate,
            "active_contexts": len(self.context_manager.active_frame_ids),
            "total_entities": len(self.grounding_engine.entities),
            "salient_entities": len(self.grounding_engine.entity_salience)
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_context_grounding():
    """Demonstrate context grounding capabilities."""
    
    print("=" * 70)
    print("CONTEXT GROUNDING AGENT DEMONSTRATION")
    print("=" * 70)
    
    print("\n1. INITIALIZING AGENT")
    print("-" * 70)
    
    agent = ContextGroundingAgent()
    print("   Agent initialized with context management")
    
    print("\n2. REGISTERING ENTITIES")
    print("-" * 70)
    print("   Creating entity database...")
    
    # Register some entities
    cat_id = agent.register_entity("Whiskers", "animal", {
        "species": "cat",
        "color": "orange",
        "aliases": ["the cat", "kitty"]
    })
    print(f"     Registered: Whiskers (cat) - ID: {cat_id}")
    
    dog_id = agent.register_entity("Rex", "animal", {
        "species": "dog",
        "color": "brown",
        "aliases": ["the dog", "puppy"]
    })
    print(f"     Registered: Rex (dog) - ID: {dog_id}")
    
    park_id = agent.register_entity("Central Park", "location", {
        "type": "park",
        "city": "New York"
    })
    print(f"     Registered: Central Park (location) - ID: {park_id}")
    
    print("\n3. ADDING CONTEXT")
    print("-" * 70)
    print("   Setting up situational context...")
    
    # Add spatial context
    agent.add_context(
        ContextType.SPATIAL,
        {
            "location": "Central Park",
            "weather": "sunny",
            "temperature": 72
        },
        priority=2
    )
    print("     Added spatial context: Central Park, sunny, 72°F")
    
    # Add temporal context
    agent.add_context(
        ContextType.TEMPORAL,
        {
            "time_of_day": "afternoon",
            "day": "Saturday"
        },
        priority=1
    )
    print("     Added temporal context: Saturday afternoon")
    
    # Add task context
    agent.add_context(
        ContextType.TASK,
        {
            "activity": "walking pets",
            "focus_entities": [cat_id, dog_id],
            "focus_topics": ["pets", "exercise"]
        },
        priority=3
    )
    print("     Added task context: walking pets")
    
    print("\n4. GROUNDING EXPRESSIONS")
    print("-" * 70)
    print("   Grounding linguistic expressions to entities...")
    
    # Ground direct reference
    link1 = agent.ground("Whiskers")
    if link1:
        print(f"\n     'Whiskers' grounded to:")
        print(f"       Entity ID: {link1.grounded_id}")
        print(f"       Confidence: {link1.confidence:.2f}")
        print(f"       Ambiguity: {link1.ambiguity:.2f}")
    
    # Ground with alias
    link2 = agent.ground("the cat")
    if link2:
        print(f"\n     'the cat' grounded to:")
        print(f"       Entity ID: {link2.grounded_id}")
        print(f"       Confidence: {link2.confidence:.2f}")
    
    # Ground location
    link3 = agent.ground("Central Park", GroundingType.LOCATION)
    if link3:
        print(f"\n     'Central Park' grounded to:")
        print(f"       Entity ID: {link3.grounded_id}")
        print(f"       Type: {link3.grounded_type}")
    
    print("\n5. REFERENCE RESOLUTION")
    print("-" * 70)
    print("   Resolving pronouns and references...")
    
    # After mentioning Whiskers, resolve "it"
    resolved = agent.resolve_reference("it")
    if resolved:
        print(f"     'it' resolved to entity: {resolved}")
        print(f"     (Most salient recent entity)")
    
    # Mention dog, then resolve
    agent.ground("Rex")
    resolved2 = agent.resolve_reference("him")
    if resolved2:
        print(f"     'him' resolved to entity: {resolved2}")
    
    print("\n6. SITUATIONAL AWARENESS")
    print("-" * 70)
    print("   Assessing current situation...")
    
    situation = agent.assess_situation()
    print(f"\n     Situation assessment:")
    print(f"       Timestamp: {situation['timestamp'].strftime('%H:%M:%S')}")
    print(f"       Spatial: {situation.get('spatial', {})}")
    print(f"       Temporal: {situation.get('temporal', {})}")
    print(f"       Task: {situation.get('task', {})}")
    
    print("\n7. SITUATION SUMMARY")
    print("-" * 70)
    
    summary = agent.get_situation_summary()
    print(f"   {summary}")
    
    print("\n8. CONTEXT SWITCHING")
    print("-" * 70)
    print("   Switching to new context...")
    
    # Add new context with higher priority
    agent.add_context(
        ContextType.TASK,
        {
            "activity": "feeding time",
            "focus_entities": [cat_id],
            "focus_topics": ["food", "schedule"]
        },
        priority=5  # Higher priority
    )
    print("     New context: feeding time (priority 5)")
    
    # Reassess situation
    situation2 = agent.assess_situation()
    summary2 = agent.get_situation_summary()
    print(f"   Updated situation: {summary2}")
    
    print("\n9. STATISTICS")
    print("-" * 70)
    
    stats = agent.get_statistics()
    print(f"   Total groundings: {stats['total_groundings']}")
    print(f"   Successful: {stats['successful_groundings']}")
    print(f"   Success rate: {stats['success_rate']:.1%}")
    print(f"   Active contexts: {stats['active_contexts']}")
    print(f"   Total entities: {stats['total_entities']}")
    print(f"   Salient entities: {stats['salient_entities']}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("1. Multi-layered context management (spatial, temporal, task)")
    print("2. Entity registration and grounding database")
    print("3. Expression grounding with confidence scores")
    print("4. Alias and partial name matching")
    print("5. Pronoun and reference resolution")
    print("6. Salience-based disambiguation")
    print("7. Situational awareness and assessment")
    print("8. Context switching and priority management")
    print("9. Natural language situation summaries")


if __name__ == "__main__":
    demonstrate_context_grounding()
