"""
Pattern 111: Multi-Modal Grounding Agent

This pattern demonstrates multi-modal context grounding where the agent
integrates information from multiple modalities (text, spatial, temporal, visual)
to ground references and understand situations holistically.

Key concepts:
- Multi-modal representation and fusion
- Cross-modal reference resolution
- Perceptual grounding
- Modality alignment
- Unified semantic space

Use cases:
- Visual question answering
- Robot navigation with natural language
- Multi-modal dialogue systems
- Augmented reality applications
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from enum import Enum
import time
import uuid


class Modality(Enum):
    """Supported modalities"""
    TEXT = "text"
    VISUAL = "visual"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    AUDIO = "audio"
    HAPTIC = "haptic"


class ReferenceType(Enum):
    """Types of cross-modal references"""
    ENTITY = "entity"
    LOCATION = "location"
    TIME = "time"
    EVENT = "event"
    PROPERTY = "property"
    RELATION = "relation"


@dataclass
class ModalityData:
    """Data from a specific modality"""
    modality: Modality
    content: Any
    timestamp: float
    confidence: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class PerceptualFeature:
    """Feature extracted from perceptual input"""
    feature_type: str
    value: Any
    modality: Modality
    confidence: float
    location: Optional[Tuple[float, float, float]] = None  # 3D coordinates
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class GroundedReference:
    """A reference grounded across modalities"""
    reference_text: str
    reference_type: ReferenceType
    modalities: Set[Modality]
    groundings: Dict[Modality, Any]  # Modality -> grounded representation
    confidence: float
    ambiguity_score: float = 0.0
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


@dataclass
class CrossModalMapping:
    """Mapping between representations in different modalities"""
    source_modality: Modality
    target_modality: Modality
    source_data: Any
    target_data: Any
    alignment_score: float
    transformation: Optional[str] = None
    
    def __post_init__(self):
        self.id = str(uuid.uuid4())[:8]


class ModalityRepresentation:
    """Manages representation for a single modality"""
    
    def __init__(self, modality: Modality):
        self.modality = modality
        self.data: List[ModalityData] = []
        self.features: List[PerceptualFeature] = []
        self.index: Dict[str, ModalityData] = {}
    
    def add_data(self, content: Any, confidence: float = 1.0,
                 source: str = "input", metadata: Optional[Dict] = None) -> ModalityData:
        """Add data to this modality representation"""
        data = ModalityData(
            modality=self.modality,
            content=content,
            timestamp=time.time(),
            confidence=confidence,
            source=source,
            metadata=metadata or {}
        )
        self.data.append(data)
        self.index[data.id] = data
        return data
    
    def extract_features(self, data: ModalityData) -> List[PerceptualFeature]:
        """Extract features from modality data"""
        features = []
        
        if self.modality == Modality.TEXT:
            # Text features: entities, keywords, sentiment
            if isinstance(data.content, str):
                words = data.content.lower().split()
                for word in words:
                    if len(word) > 3:  # Simple keyword extraction
                        feature = PerceptualFeature(
                            feature_type="keyword",
                            value=word,
                            modality=self.modality,
                            confidence=0.8,
                            timestamp=data.timestamp
                        )
                        features.append(feature)
        
        elif self.modality == Modality.VISUAL:
            # Visual features: objects, colors, shapes
            if isinstance(data.content, dict):
                for obj_type, obj_data in data.content.items():
                    feature = PerceptualFeature(
                        feature_type=obj_type,
                        value=obj_data.get("label", obj_type),
                        modality=self.modality,
                        confidence=obj_data.get("confidence", 0.9),
                        location=obj_data.get("location"),
                        timestamp=data.timestamp
                    )
                    features.append(feature)
        
        elif self.modality == Modality.SPATIAL:
            # Spatial features: positions, distances, orientations
            if isinstance(data.content, dict):
                for location_name, coords in data.content.items():
                    feature = PerceptualFeature(
                        feature_type="location",
                        value=location_name,
                        modality=self.modality,
                        confidence=0.95,
                        location=coords,
                        timestamp=data.timestamp
                    )
                    features.append(feature)
        
        elif self.modality == Modality.TEMPORAL:
            # Temporal features: events, durations, sequences
            if isinstance(data.content, dict):
                feature = PerceptualFeature(
                    feature_type="event",
                    value=data.content.get("event", "unknown"),
                    modality=self.modality,
                    confidence=0.9,
                    timestamp=data.content.get("timestamp", data.timestamp)
                )
                features.append(feature)
        
        self.features.extend(features)
        return features
    
    def get_recent_data(self, window: float = 10.0) -> List[ModalityData]:
        """Get recent data within time window"""
        current_time = time.time()
        return [d for d in self.data if current_time - d.timestamp < window]


class ModalityFusion:
    """Fuses information from multiple modalities"""
    
    def __init__(self):
        self.fusion_strategies: Dict[Tuple[Modality, Modality], str] = {}
        self.mappings: List[CrossModalMapping] = []
    
    def align_modalities(self, source: ModalityRepresentation,
                        target: ModalityRepresentation) -> List[CrossModalMapping]:
        """Align representations between two modalities"""
        mappings = []
        
        source_features = source.features[-10:]  # Recent features
        target_features = target.features[-10:]
        
        for s_feat in source_features:
            for t_feat in target_features:
                # Simple alignment: check if values match or are related
                alignment_score = self._compute_alignment(s_feat, t_feat)
                
                if alignment_score > 0.5:
                    mapping = CrossModalMapping(
                        source_modality=source.modality,
                        target_modality=target.modality,
                        source_data=s_feat,
                        target_data=t_feat,
                        alignment_score=alignment_score
                    )
                    mappings.append(mapping)
        
        self.mappings.extend(mappings)
        return mappings
    
    def _compute_alignment(self, feat1: PerceptualFeature,
                          feat2: PerceptualFeature) -> float:
        """Compute alignment score between features"""
        score = 0.0
        
        # Same feature type increases alignment
        if feat1.feature_type == feat2.feature_type:
            score += 0.3
        
        # Value similarity
        if feat1.value == feat2.value:
            score += 0.5
        elif isinstance(feat1.value, str) and isinstance(feat2.value, str):
            if feat1.value.lower() in feat2.value.lower() or \
               feat2.value.lower() in feat1.value.lower():
                score += 0.3
        
        # Temporal proximity
        if feat1.timestamp and feat2.timestamp:
            time_diff = abs(feat1.timestamp - feat2.timestamp)
            if time_diff < 1.0:
                score += 0.2 * (1.0 - time_diff)
        
        # Spatial proximity (if both have locations)
        if feat1.location and feat2.location:
            distance = sum((a - b) ** 2 for a, b in zip(feat1.location, feat2.location)) ** 0.5
            if distance < 5.0:
                score += 0.2 * (1.0 - distance / 5.0)
        
        return min(score, 1.0)
    
    def fuse(self, modalities: Dict[Modality, ModalityRepresentation],
             fusion_type: str = "weighted") -> Dict[str, Any]:
        """Fuse information from multiple modalities"""
        fused_representation = {
            "entities": {},
            "locations": {},
            "events": {},
            "properties": {},
            "confidence": 0.0
        }
        
        # Collect all features
        all_features: List[PerceptualFeature] = []
        for mod_rep in modalities.values():
            all_features.extend(mod_rep.features)
        
        # Group features by type
        feature_groups: Dict[str, List[PerceptualFeature]] = {}
        for feat in all_features:
            if feat.feature_type not in feature_groups:
                feature_groups[feat.feature_type] = []
            feature_groups[feat.feature_type].append(feat)
        
        # Fuse each group
        for feat_type, feats in feature_groups.items():
            if feat_type in ["keyword", "object", "entity"]:
                for feat in feats:
                    if feat.value not in fused_representation["entities"]:
                        fused_representation["entities"][feat.value] = {
                            "modalities": set(),
                            "confidence": 0.0,
                            "features": []
                        }
                    fused_representation["entities"][feat.value]["modalities"].add(feat.modality)
                    fused_representation["entities"][feat.value]["confidence"] = max(
                        fused_representation["entities"][feat.value]["confidence"],
                        feat.confidence
                    )
                    fused_representation["entities"][feat.value]["features"].append(feat)
            
            elif feat_type == "location":
                for feat in feats:
                    if feat.value not in fused_representation["locations"]:
                        fused_representation["locations"][feat.value] = {
                            "coordinates": feat.location,
                            "confidence": feat.confidence,
                            "modality": feat.modality
                        }
            
            elif feat_type == "event":
                for feat in feats:
                    event_key = f"{feat.value}_{feat.timestamp}"
                    if event_key not in fused_representation["events"]:
                        fused_representation["events"][event_key] = {
                            "event": feat.value,
                            "timestamp": feat.timestamp,
                            "confidence": feat.confidence
                        }
        
        # Overall confidence (average of all feature confidences)
        if all_features:
            fused_representation["confidence"] = sum(f.confidence for f in all_features) / len(all_features)
        
        return fused_representation


class CrossModalReferenceResolver:
    """Resolves references across different modalities"""
    
    def __init__(self):
        self.grounded_references: Dict[str, GroundedReference] = {}
        self.resolution_history: List[Dict[str, Any]] = []
    
    def resolve_reference(self, text_reference: str,
                         modalities: Dict[Modality, ModalityRepresentation],
                         reference_type: ReferenceType) -> Optional[GroundedReference]:
        """Resolve a text reference using available modalities"""
        
        # Find potential groundings in each modality
        groundings: Dict[Modality, Any] = {}
        confidence_scores: List[float] = []
        
        for modality, mod_rep in modalities.items():
            grounding = self._find_grounding(text_reference, mod_rep, reference_type)
            if grounding:
                groundings[modality] = grounding
                confidence_scores.append(grounding.get("confidence", 0.5))
        
        if not groundings:
            return None
        
        # Create grounded reference
        overall_confidence = sum(confidence_scores) / len(confidence_scores)
        ambiguity_score = 1.0 - overall_confidence
        
        grounded_ref = GroundedReference(
            reference_text=text_reference,
            reference_type=reference_type,
            modalities=set(groundings.keys()),
            groundings=groundings,
            confidence=overall_confidence,
            ambiguity_score=ambiguity_score
        )
        
        self.grounded_references[grounded_ref.id] = grounded_ref
        
        # Record resolution
        self.resolution_history.append({
            "reference": text_reference,
            "type": reference_type.value,
            "modalities": [m.value for m in groundings.keys()],
            "confidence": overall_confidence,
            "timestamp": time.time()
        })
        
        return grounded_ref
    
    def _find_grounding(self, reference: str, mod_rep: ModalityRepresentation,
                       ref_type: ReferenceType) -> Optional[Dict[str, Any]]:
        """Find grounding for reference in a specific modality"""
        
        reference_lower = reference.lower()
        
        if mod_rep.modality == Modality.TEXT:
            # Text grounding: find matching keywords/entities
            for feat in mod_rep.features:
                if feat.feature_type in ["keyword", "entity"]:
                    if isinstance(feat.value, str) and reference_lower in feat.value.lower():
                        return {
                            "feature": feat,
                            "match_type": "exact",
                            "confidence": feat.confidence
                        }
        
        elif mod_rep.modality == Modality.VISUAL:
            # Visual grounding: find matching objects
            for feat in mod_rep.features:
                if isinstance(feat.value, str) and reference_lower in feat.value.lower():
                    return {
                        "feature": feat,
                        "location": feat.location,
                        "confidence": feat.confidence
                    }
        
        elif mod_rep.modality == Modality.SPATIAL:
            # Spatial grounding: find matching locations
            for feat in mod_rep.features:
                if feat.feature_type == "location":
                    if isinstance(feat.value, str) and reference_lower in feat.value.lower():
                        return {
                            "location": feat.value,
                            "coordinates": feat.location,
                            "confidence": feat.confidence
                        }
        
        elif mod_rep.modality == Modality.TEMPORAL:
            # Temporal grounding: find matching events
            for feat in mod_rep.features:
                if feat.feature_type == "event":
                    if isinstance(feat.value, str) and reference_lower in feat.value.lower():
                        return {
                            "event": feat.value,
                            "timestamp": feat.timestamp,
                            "confidence": feat.confidence
                        }
        
        return None
    
    def get_disambiguation_options(self, reference: str) -> List[Dict[str, Any]]:
        """Get disambiguation options for ambiguous reference"""
        options = []
        
        for grounded_ref in self.grounded_references.values():
            if grounded_ref.reference_text.lower() == reference.lower():
                option = {
                    "reference_id": grounded_ref.id,
                    "modalities": [m.value for m in grounded_ref.modalities],
                    "confidence": grounded_ref.confidence,
                    "ambiguity": grounded_ref.ambiguity_score,
                    "groundings": {}
                }
                
                for modality, grounding in grounded_ref.groundings.items():
                    option["groundings"][modality.value] = str(grounding)
                
                options.append(option)
        
        return sorted(options, key=lambda x: x["confidence"], reverse=True)


class MultiModalGroundingAgent:
    """
    Complete multi-modal grounding agent that integrates information
    from multiple sensory modalities to ground references and understand context.
    """
    
    def __init__(self):
        self.modalities: Dict[Modality, ModalityRepresentation] = {}
        self.fusion_engine = ModalityFusion()
        self.reference_resolver = CrossModalReferenceResolver()
        self.unified_representation: Optional[Dict[str, Any]] = None
    
    def register_modality(self, modality: Modality) -> ModalityRepresentation:
        """Register a new modality"""
        if modality not in self.modalities:
            self.modalities[modality] = ModalityRepresentation(modality)
        return self.modalities[modality]
    
    def add_perception(self, modality: Modality, content: Any,
                      confidence: float = 1.0, metadata: Optional[Dict] = None) -> ModalityData:
        """Add perceptual input from a modality"""
        if modality not in self.modalities:
            self.register_modality(modality)
        
        mod_rep = self.modalities[modality]
        data = mod_rep.add_data(content, confidence, metadata=metadata)
        
        # Extract features
        mod_rep.extract_features(data)
        
        return data
    
    def align_all_modalities(self) -> List[CrossModalMapping]:
        """Align all registered modalities"""
        mappings = []
        modality_list = list(self.modalities.values())
        
        for i in range(len(modality_list)):
            for j in range(i + 1, len(modality_list)):
                alignments = self.fusion_engine.align_modalities(
                    modality_list[i],
                    modality_list[j]
                )
                mappings.extend(alignments)
        
        return mappings
    
    def fuse_modalities(self, fusion_type: str = "weighted") -> Dict[str, Any]:
        """Fuse all modality information into unified representation"""
        self.unified_representation = self.fusion_engine.fuse(
            self.modalities,
            fusion_type
        )
        return self.unified_representation
    
    def ground_reference(self, text_reference: str,
                        reference_type: ReferenceType) -> Optional[GroundedReference]:
        """Ground a text reference using all available modalities"""
        return self.reference_resolver.resolve_reference(
            text_reference,
            self.modalities,
            reference_type
        )
    
    def query_unified_space(self, query: str) -> Dict[str, Any]:
        """Query the unified semantic space"""
        if not self.unified_representation:
            self.fuse_modalities()
        
        results = {
            "query": query,
            "entities": [],
            "locations": [],
            "events": []
        }
        
        if not self.unified_representation:
            return results
        
        query_lower = query.lower()
        
        # Search entities
        for entity_name, entity_data in self.unified_representation["entities"].items():
            if query_lower in entity_name.lower():
                results["entities"].append({
                    "name": entity_name,
                    "modalities": [m.value for m in entity_data["modalities"]],
                    "confidence": entity_data["confidence"]
                })
        
        # Search locations
        for loc_name, loc_data in self.unified_representation["locations"].items():
            if query_lower in loc_name.lower():
                results["locations"].append({
                    "name": loc_name,
                    "coordinates": loc_data["coordinates"],
                    "confidence": loc_data["confidence"]
                })
        
        # Search events
        for event_key, event_data in self.unified_representation["events"].items():
            if query_lower in event_data["event"].lower():
                results["events"].append({
                    "event": event_data["event"],
                    "timestamp": event_data["timestamp"],
                    "confidence": event_data["confidence"]
                })
        
        return results
    
    def get_perceptual_summary(self) -> Dict[str, Any]:
        """Get summary of current perceptual state"""
        summary = {
            "active_modalities": [m.value for m in self.modalities.keys()],
            "total_features": sum(len(mr.features) for mr in self.modalities.values()),
            "cross_modal_mappings": len(self.fusion_engine.mappings),
            "grounded_references": len(self.reference_resolver.grounded_references),
            "modality_details": {}
        }
        
        for modality, mod_rep in self.modalities.items():
            summary["modality_details"][modality.value] = {
                "data_points": len(mod_rep.data),
                "features": len(mod_rep.features),
                "recent_data": len(mod_rep.get_recent_data())
            }
        
        return summary


# Demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("PATTERN 111: MULTI-MODAL GROUNDING AGENT")
    print("Demonstration of cross-modal reference grounding and fusion")
    print("=" * 80)
    
    # Create agent
    agent = MultiModalGroundingAgent()
    
    # Register modalities
    print("\n1. Registering Modalities")
    print("-" * 40)
    agent.register_modality(Modality.TEXT)
    agent.register_modality(Modality.VISUAL)
    agent.register_modality(Modality.SPATIAL)
    agent.register_modality(Modality.TEMPORAL)
    print("✓ Registered: TEXT, VISUAL, SPATIAL, TEMPORAL")
    
    # Add perceptual inputs
    print("\n2. Adding Perceptual Inputs")
    print("-" * 40)
    
    # Text input
    agent.add_perception(
        Modality.TEXT,
        "The red ball is on the table near the window",
        confidence=0.95
    )
    print("✓ TEXT: 'The red ball is on the table near the window'")
    
    # Visual input (simulated object detection)
    agent.add_perception(
        Modality.VISUAL,
        {
            "ball": {"label": "ball", "color": "red", "confidence": 0.92, "location": (1.0, 2.0, 0.5)},
            "table": {"label": "table", "color": "brown", "confidence": 0.88, "location": (1.0, 2.0, 0.0)},
            "window": {"label": "window", "confidence": 0.85, "location": (0.5, 2.5, 1.5)}
        },
        confidence=0.90
    )
    print("✓ VISUAL: Detected 3 objects (ball, table, window) with locations")
    
    # Spatial input
    agent.add_perception(
        Modality.SPATIAL,
        {
            "table": (1.0, 2.0, 0.0),
            "window": (0.5, 2.5, 1.5),
            "door": (3.0, 0.0, 0.0)
        },
        confidence=0.95
    )
    print("✓ SPATIAL: 3 locations mapped (table, window, door)")
    
    # Temporal input
    agent.add_perception(
        Modality.TEMPORAL,
        {
            "event": "ball_placed",
            "timestamp": time.time() - 5.0
        },
        confidence=0.80
    )
    print("✓ TEMPORAL: Event 'ball_placed' 5 seconds ago")
    
    # Align modalities
    print("\n3. Aligning Modalities")
    print("-" * 40)
    mappings = agent.align_all_modalities()
    print(f"✓ Created {len(mappings)} cross-modal mappings")
    
    # Show top mappings
    top_mappings = sorted(mappings, key=lambda m: m.alignment_score, reverse=True)[:5]
    for i, mapping in enumerate(top_mappings, 1):
        print(f"  {i}. {mapping.source_modality.value} ↔ {mapping.target_modality.value}: "
              f"score={mapping.alignment_score:.2f}")
    
    # Fuse modalities
    print("\n4. Fusing Modalities into Unified Representation")
    print("-" * 40)
    unified = agent.fuse_modalities()
    print(f"✓ Unified representation created")
    print(f"  - Entities: {len(unified['entities'])}")
    print(f"  - Locations: {len(unified['locations'])}")
    print(f"  - Events: {len(unified['events'])}")
    print(f"  - Overall confidence: {unified['confidence']:.2f}")
    
    # Show entities
    print("\n  Detected Entities:")
    for entity_name, entity_data in unified['entities'].items():
        modalities = [m.value for m in entity_data['modalities']]
        print(f"    • {entity_name}: {', '.join(modalities)} (conf={entity_data['confidence']:.2f})")
    
    # Ground references
    print("\n5. Grounding Text References")
    print("-" * 40)
    
    references = [
        ("ball", ReferenceType.ENTITY),
        ("table", ReferenceType.LOCATION),
        ("window", ReferenceType.ENTITY),
        ("ball_placed", ReferenceType.EVENT)
    ]
    
    grounded_count = 0
    for ref_text, ref_type in references:
        grounded = agent.ground_reference(ref_text, ref_type)
        if grounded:
            grounded_count += 1
            modalities = [m.value for m in grounded.modalities]
            print(f"✓ '{ref_text}' ({ref_type.value})")
            print(f"  Grounded in: {', '.join(modalities)}")
            print(f"  Confidence: {grounded.confidence:.2f}, Ambiguity: {grounded.ambiguity_score:.2f}")
    
    print(f"\nSuccessfully grounded {grounded_count}/{len(references)} references")
    
    # Query unified space
    print("\n6. Querying Unified Semantic Space")
    print("-" * 40)
    
    queries = ["ball", "table", "window"]
    for query in queries:
        results = agent.query_unified_space(query)
        print(f"\nQuery: '{query}'")
        if results["entities"]:
            for entity in results["entities"]:
                print(f"  Entity: {entity['name']} [{', '.join(entity['modalities'])}]")
        if results["locations"]:
            for loc in results["locations"]:
                print(f"  Location: {loc['name']} at {loc['coordinates']}")
    
    # Summary
    print("\n7. Perceptual State Summary")
    print("-" * 40)
    summary = agent.get_perceptual_summary()
    print(f"Active Modalities: {len(summary['active_modalities'])}")
    print(f"Total Features: {summary['total_features']}")
    print(f"Cross-Modal Mappings: {summary['cross_modal_mappings']}")
    print(f"Grounded References: {summary['grounded_references']}")
    
    print("\nModality Details:")
    for modality, details in summary['modality_details'].items():
        print(f"  {modality.upper()}:")
        print(f"    - Data points: {details['data_points']}")
        print(f"    - Features: {details['features']}")
        print(f"    - Recent: {details['recent_data']}")
    
    print("\n" + "=" * 80)
    print("✓ Multi-modal grounding demonstration complete!")
    print("  Cross-modal alignment, fusion, and reference resolution working.")
    print("=" * 80)
