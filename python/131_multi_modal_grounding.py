"""
Pattern 131: Multi-Modal Grounding Agent

This pattern implements multi-modal input processing (text, image, audio),
modality fusion, and unified understanding across different input types.

Use Cases:
- Visual question answering
- Audio-visual analysis
- Multi-modal content understanding
- Cross-modal retrieval
- Accessibility applications

Category: Context & Grounding (2/4 = 50%)
Complexity: Advanced
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
import hashlib


class ModalityType(Enum):
    """Types of modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SENSOR = "sensor"


class FusionStrategy(Enum):
    """Strategies for multi-modal fusion."""
    EARLY_FUSION = "early_fusion"  # Fuse at feature level
    LATE_FUSION = "late_fusion"  # Fuse at decision level
    HYBRID_FUSION = "hybrid_fusion"  # Mix of both
    ATTENTION_FUSION = "attention_fusion"  # Attention-weighted


class AlignmentType(Enum):
    """Types of cross-modal alignment."""
    TEMPORAL = "temporal"  # Time-based alignment
    SPATIAL = "spatial"  # Space-based alignment
    SEMANTIC = "semantic"  # Meaning-based alignment
    REFERENCE = "reference"  # Reference-based alignment


@dataclass
class ModalityInput:
    """Input from a specific modality."""
    modality: ModalityType
    data: Any
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_id(self) -> str:
        """Get unique identifier."""
        content_str = str(self.data)[:100] + str(self.timestamp)
        return hashlib.md5(content_str.encode()).hexdigest()


@dataclass
class ModalityFeatures:
    """Extracted features from modality."""
    modality: ModalityType
    features: Dict[str, Any]
    confidence: float
    source_id: str


@dataclass
class CrossModalAlignment:
    """Alignment between modalities."""
    modality_1: ModalityType
    modality_2: ModalityType
    alignment_type: AlignmentType
    alignment_score: float
    aligned_elements: List[Tuple[Any, Any]]


@dataclass
class FusedRepresentation:
    """Fused multi-modal representation."""
    modalities: Set[ModalityType]
    fusion_strategy: FusionStrategy
    representation: Dict[str, Any]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class TextProcessor:
    """Processes text modality."""
    
    def __init__(self):
        self.vocab_size = 10000
    
    def extract_features(self, text: str) -> ModalityFeatures:
        """Extract text features."""
        # Simplified feature extraction
        words = text.lower().split()
        
        features = {
            'word_count': len(words),
            'unique_words': len(set(words)),
            'avg_word_length': sum(len(w) for w in words) / max(1, len(words)),
            'has_question': '?' in text,
            'has_numbers': any(c.isdigit() for c in text),
            'sentiment': self._simple_sentiment(text),
            'entities': self._extract_entities(text),
            'keywords': words[:10]  # Top 10 words as keywords
        }
        
        return ModalityFeatures(
            modality=ModalityType.TEXT,
            features=features,
            confidence=0.9,
            source_id=hashlib.md5(text.encode()).hexdigest()
        )
    
    def _simple_sentiment(self, text: str) -> str:
        """Simple sentiment analysis."""
        positive_words = {'good', 'great', 'excellent', 'happy', 'wonderful'}
        negative_words = {'bad', 'terrible', 'awful', 'sad', 'poor'}
        
        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simplified)."""
        # Simple capitalized word extraction
        words = text.split()
        entities = [w for w in words if w[0].isupper() and len(w) > 1]
        return entities[:5]  # Top 5


class ImageProcessor:
    """Processes image modality."""
    
    def __init__(self):
        self.image_size = (224, 224)
    
    def extract_features(self, image_data: Dict[str, Any]) -> ModalityFeatures:
        """Extract image features."""
        # Simplified feature extraction (in production, use CNN)
        
        features = {
            'width': image_data.get('width', 0),
            'height': image_data.get('height', 0),
            'aspect_ratio': image_data.get('width', 1) / max(1, image_data.get('height', 1)),
            'format': image_data.get('format', 'unknown'),
            'color_mode': image_data.get('color_mode', 'RGB'),
            'detected_objects': image_data.get('objects', []),
            'scene': image_data.get('scene', 'unknown'),
            'dominant_colors': image_data.get('colors', []),
            'has_faces': image_data.get('has_faces', False),
            'text_in_image': image_data.get('ocr_text', '')
        }
        
        return ModalityFeatures(
            modality=ModalityType.IMAGE,
            features=features,
            confidence=0.85,
            source_id=str(image_data.get('path', 'unknown'))
        )


class AudioProcessor:
    """Processes audio modality."""
    
    def __init__(self):
        self.sample_rate = 16000
    
    def extract_features(self, audio_data: Dict[str, Any]) -> ModalityFeatures:
        """Extract audio features."""
        # Simplified feature extraction (in production, use audio processing)
        
        features = {
            'duration': audio_data.get('duration', 0),
            'sample_rate': audio_data.get('sample_rate', self.sample_rate),
            'channels': audio_data.get('channels', 1),
            'transcript': audio_data.get('transcript', ''),
            'speech_rate': audio_data.get('speech_rate', 0),
            'volume_level': audio_data.get('volume', 0),
            'pitch_range': audio_data.get('pitch_range', 'medium'),
            'speaker_count': audio_data.get('speakers', 1),
            'emotion': audio_data.get('emotion', 'neutral'),
            'language': audio_data.get('language', 'en')
        }
        
        return ModalityFeatures(
            modality=ModalityType.AUDIO,
            features=features,
            confidence=0.8,
            source_id=str(audio_data.get('path', 'unknown'))
        )


class CrossModalAligner:
    """Aligns information across modalities."""
    
    def __init__(self):
        self.alignment_threshold = 0.5
    
    def align_text_image(
        self,
        text_features: ModalityFeatures,
        image_features: ModalityFeatures
    ) -> CrossModalAlignment:
        """Align text and image."""
        aligned_elements = []
        alignment_score = 0.0
        
        # Check for objects mentioned in text
        text_words = set(text_features.features.get('keywords', []))
        image_objects = set(image_features.features.get('detected_objects', []))
        
        common_elements = text_words & image_objects
        
        for element in common_elements:
            aligned_elements.append((element, element))
        
        # Check for text in image (OCR)
        ocr_text = image_features.features.get('text_in_image', '').lower()
        text_entities = set(e.lower() for e in text_features.features.get('entities', []))
        
        for entity in text_entities:
            if entity in ocr_text:
                aligned_elements.append((entity, f"text_in_image:{entity}"))
        
        # Calculate alignment score
        if len(text_words) + len(image_objects) > 0:
            alignment_score = len(aligned_elements) / (len(text_words) + len(image_objects))
        
        return CrossModalAlignment(
            modality_1=ModalityType.TEXT,
            modality_2=ModalityType.IMAGE,
            alignment_type=AlignmentType.SEMANTIC,
            alignment_score=min(1.0, alignment_score * 2),  # Normalize
            aligned_elements=aligned_elements
        )
    
    def align_audio_text(
        self,
        audio_features: ModalityFeatures,
        text_features: ModalityFeatures
    ) -> CrossModalAlignment:
        """Align audio and text."""
        aligned_elements = []
        
        # Compare transcript with text
        transcript = audio_features.features.get('transcript', '').lower()
        text_words = set(text_features.features.get('keywords', []))
        
        for word in text_words:
            if word.lower() in transcript:
                aligned_elements.append((word, f"transcript:{word}"))
        
        # Calculate temporal alignment score
        alignment_score = len(aligned_elements) / max(1, len(text_words))
        
        return CrossModalAlignment(
            modality_1=ModalityType.AUDIO,
            modality_2=ModalityType.TEXT,
            alignment_type=AlignmentType.TEMPORAL,
            alignment_score=alignment_score,
            aligned_elements=aligned_elements
        )


class ModalityFusion:
    """Fuses multiple modalities."""
    
    def __init__(self, strategy: FusionStrategy = FusionStrategy.HYBRID_FUSION):
        self.strategy = strategy
    
    def fuse_features(
        self,
        features_list: List[ModalityFeatures],
        alignments: List[CrossModalAlignment]
    ) -> FusedRepresentation:
        """Fuse features from multiple modalities."""
        if self.strategy == FusionStrategy.EARLY_FUSION:
            return self._early_fusion(features_list)
        elif self.strategy == FusionStrategy.LATE_FUSION:
            return self._late_fusion(features_list)
        elif self.strategy == FusionStrategy.ATTENTION_FUSION:
            return self._attention_fusion(features_list, alignments)
        else:  # HYBRID_FUSION
            return self._hybrid_fusion(features_list, alignments)
    
    def _early_fusion(self, features_list: List[ModalityFeatures]) -> FusedRepresentation:
        """Early fusion - combine at feature level."""
        combined_features = {}
        modalities = set()
        total_confidence = 0.0
        
        for features in features_list:
            modalities.add(features.modality)
            # Namespace features by modality
            for key, value in features.features.items():
                combined_features[f"{features.modality.value}_{key}"] = value
            total_confidence += features.confidence
        
        avg_confidence = total_confidence / len(features_list)
        
        return FusedRepresentation(
            modalities=modalities,
            fusion_strategy=FusionStrategy.EARLY_FUSION,
            representation=combined_features,
            confidence=avg_confidence
        )
    
    def _late_fusion(self, features_list: List[ModalityFeatures]) -> FusedRepresentation:
        """Late fusion - combine at decision level."""
        decisions = {}
        modalities = set()
        
        for features in features_list:
            modalities.add(features.modality)
            # Extract high-level decision from features
            decision = self._extract_decision(features)
            decisions[features.modality.value] = decision
        
        # Vote or aggregate decisions
        final_decision = self._aggregate_decisions(decisions)
        
        return FusedRepresentation(
            modalities=modalities,
            fusion_strategy=FusionStrategy.LATE_FUSION,
            representation={'decisions': decisions, 'final': final_decision},
            confidence=0.8
        )
    
    def _attention_fusion(
        self,
        features_list: List[ModalityFeatures],
        alignments: List[CrossModalAlignment]
    ) -> FusedRepresentation:
        """Attention-based fusion - weight by alignment."""
        # Calculate attention weights based on alignments
        weights = self._calculate_attention_weights(features_list, alignments)
        
        weighted_features = {}
        modalities = set()
        
        for features, weight in zip(features_list, weights):
            modalities.add(features.modality)
            for key, value in features.features.items():
                weighted_key = f"{features.modality.value}_{key}"
                if isinstance(value, (int, float)):
                    weighted_features[weighted_key] = value * weight
                else:
                    weighted_features[weighted_key] = value
        
        weighted_features['attention_weights'] = {
            f.modality.value: w for f, w in zip(features_list, weights)
        }
        
        return FusedRepresentation(
            modalities=modalities,
            fusion_strategy=FusionStrategy.ATTENTION_FUSION,
            representation=weighted_features,
            confidence=sum(f.confidence * w for f, w in zip(features_list, weights))
        )
    
    def _hybrid_fusion(
        self,
        features_list: List[ModalityFeatures],
        alignments: List[CrossModalAlignment]
    ) -> FusedRepresentation:
        """Hybrid fusion - combine early and late fusion."""
        # Early fusion for aligned features
        early_fused = self._early_fusion(features_list)
        
        # Late fusion for decisions
        late_fused = self._late_fusion(features_list)
        
        # Combine
        hybrid_representation = {
            'early_features': early_fused.representation,
            'late_decisions': late_fused.representation,
            'alignments': [
                {
                    'modalities': f"{a.modality_1.value}-{a.modality_2.value}",
                    'score': a.alignment_score
                }
                for a in alignments
            ]
        }
        
        return FusedRepresentation(
            modalities=early_fused.modalities,
            fusion_strategy=FusionStrategy.HYBRID_FUSION,
            representation=hybrid_representation,
            confidence=(early_fused.confidence + late_fused.confidence) / 2
        )
    
    def _extract_decision(self, features: ModalityFeatures) -> Dict[str, Any]:
        """Extract high-level decision from features."""
        if features.modality == ModalityType.TEXT:
            return {
                'sentiment': features.features.get('sentiment', 'neutral'),
                'has_question': features.features.get('has_question', False),
                'category': 'text_content'
            }
        elif features.modality == ModalityType.IMAGE:
            return {
                'scene': features.features.get('scene', 'unknown'),
                'has_objects': len(features.features.get('detected_objects', [])) > 0,
                'category': 'visual_content'
            }
        elif features.modality == ModalityType.AUDIO:
            return {
                'emotion': features.features.get('emotion', 'neutral'),
                'has_speech': len(features.features.get('transcript', '')) > 0,
                'category': 'audio_content'
            }
        return {}
    
    def _aggregate_decisions(self, decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate decisions from multiple modalities."""
        aggregated = {
            'modality_count': len(decisions),
            'categories': [d.get('category', 'unknown') for d in decisions.values()]
        }
        
        # Aggregate sentiment/emotion
        sentiments = []
        for decision in decisions.values():
            if 'sentiment' in decision:
                sentiments.append(decision['sentiment'])
            if 'emotion' in decision:
                sentiments.append(decision['emotion'])
        
        if sentiments:
            aggregated['overall_sentiment'] = max(set(sentiments), key=sentiments.count)
        
        return aggregated
    
    def _calculate_attention_weights(
        self,
        features_list: List[ModalityFeatures],
        alignments: List[CrossModalAlignment]
    ) -> List[float]:
        """Calculate attention weights based on alignments."""
        # Base weights from confidence
        weights = [f.confidence for f in features_list]
        
        # Adjust based on alignment scores
        for alignment in alignments:
            for i, features in enumerate(features_list):
                if features.modality in [alignment.modality_1, alignment.modality_2]:
                    weights[i] *= (1 + alignment.alignment_score)
        
        # Normalize
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        return weights


class MultiModalGroundingAgent:
    """Agent for multi-modal input processing and grounding."""
    
    def __init__(self, fusion_strategy: FusionStrategy = FusionStrategy.HYBRID_FUSION):
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.audio_processor = AudioProcessor()
        self.aligner = CrossModalAligner()
        self.fusion = ModalityFusion(fusion_strategy)
        self.processing_history: List[Dict[str, Any]] = []
    
    def process_multi_modal_input(
        self,
        inputs: List[ModalityInput]
    ) -> Tuple[FusedRepresentation, List[CrossModalAlignment]]:
        """Process multi-modal inputs."""
        # Extract features from each modality
        features_list = []
        
        for input_data in inputs:
            if input_data.modality == ModalityType.TEXT:
                features = self.text_processor.extract_features(input_data.data)
            elif input_data.modality == ModalityType.IMAGE:
                features = self.image_processor.extract_features(input_data.data)
            elif input_data.modality == ModalityType.AUDIO:
                features = self.audio_processor.extract_features(input_data.data)
            else:
                continue
            
            features_list.append(features)
        
        # Align modalities
        alignments = self._align_all_modalities(features_list)
        
        # Fuse features
        fused = self.fusion.fuse_features(features_list, alignments)
        
        # Record processing
        self.processing_history.append({
            'timestamp': datetime.now(),
            'modalities': [f.modality.value for f in features_list],
            'alignments': len(alignments),
            'confidence': fused.confidence
        })
        
        return fused, alignments
    
    def _align_all_modalities(
        self,
        features_list: List[ModalityFeatures]
    ) -> List[CrossModalAlignment]:
        """Align all pairs of modalities."""
        alignments = []
        
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                feat1, feat2 = features_list[i], features_list[j]
                
                if (feat1.modality == ModalityType.TEXT and 
                    feat2.modality == ModalityType.IMAGE):
                    alignment = self.aligner.align_text_image(feat1, feat2)
                    alignments.append(alignment)
                
                elif (feat1.modality == ModalityType.IMAGE and 
                      feat2.modality == ModalityType.TEXT):
                    alignment = self.aligner.align_text_image(feat2, feat1)
                    alignments.append(alignment)
                
                elif (feat1.modality == ModalityType.AUDIO and 
                      feat2.modality == ModalityType.TEXT):
                    alignment = self.aligner.align_audio_text(feat1, feat2)
                    alignments.append(alignment)
                
                elif (feat1.modality == ModalityType.TEXT and 
                      feat2.modality == ModalityType.AUDIO):
                    alignment = self.aligner.align_audio_text(feat2, feat1)
                    alignments.append(alignment)
        
        return alignments
    
    def answer_question(
        self,
        question: str,
        context_inputs: List[ModalityInput]
    ) -> str:
        """Answer question using multi-modal context."""
        # Add question as text modality
        all_inputs = [
            ModalityInput(
                modality=ModalityType.TEXT,
                data=question,
                timestamp=datetime.now()
            )
        ] + context_inputs
        
        # Process all inputs
        fused, alignments = self.process_multi_modal_input(all_inputs)
        
        # Generate answer based on fused representation
        answer = self._generate_answer(question, fused, alignments)
        
        return answer
    
    def _generate_answer(
        self,
        question: str,
        fused: FusedRepresentation,
        alignments: List[CrossModalAlignment]
    ) -> str:
        """Generate answer from fused representation."""
        # Simplified answer generation
        answer_parts = [f"Based on {len(fused.modalities)} modalities:"]
        
        # Extract insights from fused representation
        if 'early_features' in fused.representation:
            # Hybrid fusion
            early = fused.representation['early_features']
            late = fused.representation['late_decisions']
            
            # Check for image content
            if any('image' in k for k in early.keys()):
                objects = early.get('image_detected_objects', [])
                if objects:
                    answer_parts.append(f"The image shows: {', '.join(objects[:3])}")
            
            # Check for audio content
            if any('audio' in k for k in early.keys()):
                transcript = early.get('audio_transcript', '')
                if transcript:
                    answer_parts.append(f"Audio transcript: '{transcript[:50]}...'")
        
        # Include alignment information
        if alignments:
            strong_alignments = [a for a in alignments if a.alignment_score > 0.5]
            if strong_alignments:
                answer_parts.append(
                    f"Found {len(strong_alignments)} strong cross-modal alignments"
                )
        
        # Add confidence
        answer_parts.append(f"(Confidence: {fused.confidence:.2f})")
        
        return " ".join(answer_parts)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get multi-modal processing statistics."""
        if not self.processing_history:
            return {'total_processed': 0}
        
        modality_counts = {}
        for record in self.processing_history:
            for modality in record['modalities']:
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        avg_confidence = sum(r['confidence'] for r in self.processing_history) / len(self.processing_history)
        
        return {
            'total_processed': len(self.processing_history),
            'modality_counts': modality_counts,
            'average_confidence': avg_confidence,
            'total_alignments': sum(r['alignments'] for r in self.processing_history)
        }


def demonstrate_multi_modal_grounding():
    """Demonstrate the Multi-Modal Grounding Agent."""
    print("=" * 60)
    print("Multi-Modal Grounding Agent Demonstration")
    print("=" * 60)
    
    # Create agent
    agent = MultiModalGroundingAgent(FusionStrategy.HYBRID_FUSION)
    
    print("\n1. TEXT-ONLY PROCESSING")
    print("-" * 60)
    
    text_input = ModalityInput(
        modality=ModalityType.TEXT,
        data="What is the weather like today? It looks sunny outside.",
        timestamp=datetime.now()
    )
    
    fused, alignments = agent.process_multi_modal_input([text_input])
    print(f"Processed text input")
    print(f"Confidence: {fused.confidence:.2f}")
    print(f"Fusion strategy: {fused.fusion_strategy.value}")
    
    print("\n\n2. TEXT + IMAGE PROCESSING")
    print("-" * 60)
    
    text_input = ModalityInput(
        modality=ModalityType.TEXT,
        data="Show me the cat and dog in this picture",
        timestamp=datetime.now()
    )
    
    image_input = ModalityInput(
        modality=ModalityType.IMAGE,
        data={
            'width': 800,
            'height': 600,
            'format': 'JPEG',
            'objects': ['cat', 'dog', 'tree'],
            'scene': 'outdoor',
            'has_faces': False,
            'ocr_text': 'PETS'
        },
        timestamp=datetime.now()
    )
    
    fused, alignments = agent.process_multi_modal_input([text_input, image_input])
    
    print(f"Processed {len(fused.modalities)} modalities: {', '.join(m.value for m in fused.modalities)}")
    print(f"Confidence: {fused.confidence:.2f}")
    print(f"Alignments found: {len(alignments)}")
    
    for alignment in alignments:
        print(f"\nAlignment: {alignment.modality_1.value} ↔ {alignment.modality_2.value}")
        print(f"  Score: {alignment.alignment_score:.2f}")
        print(f"  Type: {alignment.alignment_type.value}")
        print(f"  Aligned elements: {len(alignment.aligned_elements)}")
        if alignment.aligned_elements:
            print(f"  Examples: {alignment.aligned_elements[:3]}")
    
    print("\n\n3. TEXT + AUDIO PROCESSING")
    print("-" * 60)
    
    text_input = ModalityInput(
        modality=ModalityType.TEXT,
        data="What did the speaker say about climate change?",
        timestamp=datetime.now()
    )
    
    audio_input = ModalityInput(
        modality=ModalityType.AUDIO,
        data={
            'duration': 30.5,
            'sample_rate': 16000,
            'channels': 1,
            'transcript': 'Climate change is affecting weather patterns globally',
            'speech_rate': 150,
            'emotion': 'serious',
            'language': 'en',
            'speakers': 1
        },
        timestamp=datetime.now()
    )
    
    fused, alignments = agent.process_multi_modal_input([text_input, audio_input])
    
    print(f"Processed {len(fused.modalities)} modalities")
    print(f"Confidence: {fused.confidence:.2f}")
    
    for alignment in alignments:
        print(f"\nAlignment: {alignment.modality_1.value} ↔ {alignment.modality_2.value}")
        print(f"  Score: {alignment.alignment_score:.2f}")
        print(f"  Aligned words: {len(alignment.aligned_elements)}")
    
    print("\n\n4. TRI-MODAL PROCESSING (TEXT + IMAGE + AUDIO)")
    print("-" * 60)
    
    text_input = ModalityInput(
        modality=ModalityType.TEXT,
        data="Describe the presentation about renewable energy",
        timestamp=datetime.now()
    )
    
    image_input = ModalityInput(
        modality=ModalityType.IMAGE,
        data={
            'width': 1920,
            'height': 1080,
            'format': 'PNG',
            'objects': ['solar panel', 'wind turbine', 'graph'],
            'scene': 'presentation',
            'ocr_text': 'Renewable Energy 2025',
            'has_faces': True
        },
        timestamp=datetime.now()
    )
    
    audio_input = ModalityInput(
        modality=ModalityType.AUDIO,
        data={
            'duration': 120,
            'transcript': 'Solar and wind energy are key to renewable future',
            'emotion': 'enthusiastic',
            'language': 'en',
            'speakers': 1
        },
        timestamp=datetime.now()
    )
    
    fused, alignments = agent.process_multi_modal_input([text_input, image_input, audio_input])
    
    print(f"Processed {len(fused.modalities)} modalities: {', '.join(m.value for m in fused.modalities)}")
    print(f"Confidence: {fused.confidence:.2f}")
    print(f"Total alignments: {len(alignments)}")
    
    # Question answering
    print("\n\n5. MULTI-MODAL QUESTION ANSWERING")
    print("-" * 60)
    
    question = "What renewable energy sources are shown?"
    answer = agent.answer_question(question, [image_input, audio_input])
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    question2 = "What is the speaker's emotion?"
    answer2 = agent.answer_question(question2, [audio_input])
    
    print(f"\nQuestion: {question2}")
    print(f"Answer: {answer2}")
    
    print("\n\n6. FUSION STRATEGIES COMPARISON")
    print("-" * 60)
    
    strategies = [
        FusionStrategy.EARLY_FUSION,
        FusionStrategy.LATE_FUSION,
        FusionStrategy.ATTENTION_FUSION,
        FusionStrategy.HYBRID_FUSION
    ]
    
    for strategy in strategies:
        agent_test = MultiModalGroundingAgent(strategy)
        fused, _ = agent_test.process_multi_modal_input([text_input, image_input])
        print(f"\n{strategy.value}:")
        print(f"  Confidence: {fused.confidence:.2f}")
        print(f"  Representation keys: {len(fused.representation)}")
    
    print("\n\n7. STATISTICS")
    print("-" * 60)
    
    stats = agent.get_statistics()
    print(f"  Total Processed: {stats['total_processed']}")
    print(f"  Average Confidence: {stats['average_confidence']:.2f}")
    print(f"  Total Alignments: {stats['total_alignments']}")
    print(f"\n  Modality Usage:")
    for modality, count in sorted(stats['modality_counts'].items()):
        print(f"    {modality}: {count}")
    
    print("\n" + "=" * 60)
    print("🎉 Pattern 131 Complete!")
    print("Context & Grounding Category: 50%")
    print("131/170 patterns implemented (77.1%)!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_multi_modal_grounding()
