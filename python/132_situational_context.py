"""
Pattern 132: Situational Context Agent

This pattern implements real-time context awareness, environmental understanding,
and adaptive behavior based on situational factors.

Use Cases:
- Smart home automation
- Mobile assistant context switching
- Real-time recommendation systems
- Context-aware notifications
- Adaptive user interfaces

Category: Context & Grounding (3/4 = 75%)
Complexity: Advanced
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from datetime import datetime, time
import hashlib


class ContextDimension(Enum):
    """Dimensions of situational context."""
    TEMPORAL = "temporal"  # Time-based
    SPATIAL = "spatial"  # Location-based
    SOCIAL = "social"  # People/interaction-based
    ENVIRONMENTAL = "environmental"  # Physical environment
    ACTIVITY = "activity"  # User activity
    DEVICE = "device"  # Device/platform context


class TimeOfDay(Enum):
    """Time of day categories."""
    EARLY_MORNING = "early_morning"  # 5-8 AM
    MORNING = "morning"  # 8-12 PM
    AFTERNOON = "afternoon"  # 12-5 PM
    EVENING = "evening"  # 5-9 PM
    NIGHT = "night"  # 9 PM-5 AM


class LocationType(Enum):
    """Types of locations."""
    HOME = "home"
    WORK = "work"
    COMMUTE = "commute"
    STORE = "store"
    RESTAURANT = "restaurant"
    GYM = "gym"
    OUTDOOR = "outdoor"
    OTHER = "other"


class ActivityType(Enum):
    """Types of activities."""
    SLEEPING = "sleeping"
    WORKING = "working"
    COMMUTING = "commuting"
    EXERCISING = "exercising"
    EATING = "eating"
    SOCIALIZING = "socializing"
    RELAXING = "relaxing"
    SHOPPING = "shopping"


@dataclass
class TemporalContext:
    """Temporal context information."""
    current_time: datetime
    time_of_day: TimeOfDay
    day_of_week: str
    is_weekend: bool
    is_holiday: bool = False
    
    @classmethod
    def from_datetime(cls, dt: datetime) -> 'TemporalContext':
        """Create from datetime."""
        hour = dt.hour
        
        if 5 <= hour < 8:
            tod = TimeOfDay.EARLY_MORNING
        elif 8 <= hour < 12:
            tod = TimeOfDay.MORNING
        elif 12 <= hour < 17:
            tod = TimeOfDay.AFTERNOON
        elif 17 <= hour < 21:
            tod = TimeOfDay.EVENING
        else:
            tod = TimeOfDay.NIGHT
        
        return cls(
            current_time=dt,
            time_of_day=tod,
            day_of_week=dt.strftime('%A'),
            is_weekend=dt.weekday() >= 5
        )


@dataclass
class SpatialContext:
    """Spatial context information."""
    location_type: LocationType
    latitude: float
    longitude: float
    accuracy: float
    is_indoor: bool
    nearby_places: List[str] = field(default_factory=list)


@dataclass
class SocialContext:
    """Social context information."""
    people_nearby: List[str]
    interaction_mode: str  # alone, with_one, with_group
    social_situation: str  # private, public, professional
    in_meeting: bool = False
    available: bool = True


@dataclass
class EnvironmentalContext:
    """Environmental context information."""
    weather: str
    temperature: float  # Celsius
    noise_level: str  # quiet, moderate, loud
    light_level: str  # dark, dim, normal, bright
    connectivity: str  # wifi, cellular, offline


@dataclass
class ActivityContext:
    """Activity context information."""
    current_activity: ActivityType
    activity_confidence: float
    duration: int  # Minutes in current activity
    previous_activity: Optional[ActivityType] = None


@dataclass
class DeviceContext:
    """Device context information."""
    device_type: str  # phone, tablet, laptop, desktop, watch
    battery_level: float
    is_charging: bool
    screen_on: bool
    network_type: str
    in_motion: bool = False


@dataclass
class SituationalContext:
    """Complete situational context."""
    context_id: str
    timestamp: datetime
    temporal: TemporalContext
    spatial: SpatialContext
    social: SocialContext
    environmental: EnvironmentalContext
    activity: ActivityContext
    device: DeviceContext
    
    def get_context_summary(self) -> str:
        """Get readable summary."""
        parts = [
            f"Time: {self.temporal.time_of_day.value}",
            f"Location: {self.spatial.location_type.value}",
            f"Activity: {self.activity.current_activity.value}",
            f"Social: {self.social.interaction_mode}"
        ]
        return " | ".join(parts)


class ContextSensor:
    """Senses and collects context information."""
    
    def __init__(self):
        self.last_reading: Optional[SituationalContext] = None
    
    def sense_context(
        self,
        user_data: Dict[str, Any]
    ) -> SituationalContext:
        """Sense current situational context."""
        # Temporal
        current_time = datetime.now()
        temporal = TemporalContext.from_datetime(current_time)
        
        # Spatial
        spatial = SpatialContext(
            location_type=LocationType[user_data.get('location_type', 'HOME').upper()],
            latitude=user_data.get('latitude', 0.0),
            longitude=user_data.get('longitude', 0.0),
            accuracy=user_data.get('location_accuracy', 10.0),
            is_indoor=user_data.get('is_indoor', True),
            nearby_places=user_data.get('nearby_places', [])
        )
        
        # Social
        social = SocialContext(
            people_nearby=user_data.get('people_nearby', []),
            interaction_mode=user_data.get('interaction_mode', 'alone'),
            social_situation=user_data.get('social_situation', 'private'),
            in_meeting=user_data.get('in_meeting', False),
            available=user_data.get('available', True)
        )
        
        # Environmental
        environmental = EnvironmentalContext(
            weather=user_data.get('weather', 'clear'),
            temperature=user_data.get('temperature', 20.0),
            noise_level=user_data.get('noise_level', 'quiet'),
            light_level=user_data.get('light_level', 'normal'),
            connectivity=user_data.get('connectivity', 'wifi')
        )
        
        # Activity
        activity = ActivityContext(
            current_activity=ActivityType[user_data.get('activity', 'RELAXING').upper()],
            activity_confidence=user_data.get('activity_confidence', 0.8),
            duration=user_data.get('activity_duration', 10),
            previous_activity=self.last_reading.activity.current_activity if self.last_reading else None
        )
        
        # Device
        device = DeviceContext(
            device_type=user_data.get('device_type', 'phone'),
            battery_level=user_data.get('battery_level', 80.0),
            is_charging=user_data.get('is_charging', False),
            screen_on=user_data.get('screen_on', True),
            network_type=user_data.get('network_type', 'wifi'),
            in_motion=user_data.get('in_motion', False)
        )
        
        # Create context
        context_id = hashlib.md5(
            f"{current_time.isoformat()}{spatial.latitude}{spatial.longitude}".encode()
        ).hexdigest()[:8]
        
        context = SituationalContext(
            context_id=context_id,
            timestamp=current_time,
            temporal=temporal,
            spatial=spatial,
            social=social,
            environmental=environmental,
            activity=activity,
            device=device
        )
        
        self.last_reading = context
        return context


class ContextAnalyzer:
    """Analyzes context for patterns and insights."""
    
    def __init__(self):
        self.context_history: List[SituationalContext] = []
    
    def analyze_context(self, context: SituationalContext) -> Dict[str, Any]:
        """Analyze current context."""
        self.context_history.append(context)
        
        analysis = {
            'disruption_score': self._calculate_disruption_score(context),
            'urgency_level': self._calculate_urgency_level(context),
            'privacy_level': self._calculate_privacy_level(context),
            'attention_availability': self._calculate_attention(context),
            'preferred_notification': self._get_notification_preference(context),
            'interaction_style': self._get_interaction_style(context)
        }
        
        return analysis
    
    def _calculate_disruption_score(self, context: SituationalContext) -> float:
        """Calculate how disruptive interaction would be (0-1)."""
        score = 0.0
        
        # Activity disruption
        if context.activity.current_activity == ActivityType.SLEEPING:
            score += 0.4
        elif context.activity.current_activity == ActivityType.WORKING:
            score += 0.3
        elif context.activity.current_activity in [ActivityType.EATING, ActivityType.SOCIALIZING]:
            score += 0.2
        
        # Social context
        if context.social.in_meeting:
            score += 0.3
        elif context.social.interaction_mode == 'with_group':
            score += 0.2
        
        # Environmental
        if context.environmental.noise_level == 'loud':
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_urgency_level(self, context: SituationalContext) -> str:
        """Determine urgency level based on context."""
        # Time-based
        if context.temporal.time_of_day == TimeOfDay.NIGHT:
            return 'low'
        elif context.temporal.time_of_day in [TimeOfDay.MORNING, TimeOfDay.AFTERNOON]:
            return 'normal'
        
        # Activity-based
        if context.activity.current_activity == ActivityType.WORKING:
            return 'high'
        
        return 'normal'
    
    def _calculate_privacy_level(self, context: SituationalContext) -> str:
        """Determine privacy level."""
        if context.social.social_situation == 'public':
            return 'low'
        elif context.social.social_situation == 'professional':
            return 'medium'
        else:
            return 'high'
    
    def _calculate_attention(self, context: SituationalContext) -> float:
        """Calculate available attention (0-1)."""
        attention = 1.0
        
        if context.activity.current_activity == ActivityType.SLEEPING:
            attention = 0.0
        elif context.activity.current_activity == ActivityType.WORKING:
            attention = 0.5
        elif context.social.in_meeting:
            attention = 0.3
        elif context.device.in_motion:
            attention = 0.6
        
        return attention
    
    def _get_notification_preference(self, context: SituationalContext) -> str:
        """Get preferred notification mode."""
        if context.activity.current_activity == ActivityType.SLEEPING:
            return 'silent'
        elif context.social.in_meeting or context.social.interaction_mode == 'with_group':
            return 'vibrate'
        elif context.environmental.noise_level == 'loud':
            return 'visual'
        else:
            return 'sound'
    
    def _get_interaction_style(self, context: SituationalContext) -> str:
        """Get preferred interaction style."""
        if context.device.device_type == 'watch':
            return 'glanceable'
        elif context.device.in_motion or context.activity.current_activity == ActivityType.COMMUTING:
            return 'voice'
        elif context.spatial.location_type == LocationType.WORK:
            return 'concise'
        else:
            return 'detailed'


class AdaptiveBehaviorEngine:
    """Adapts behavior based on context."""
    
    def __init__(self):
        self.adaptation_history: List[Dict[str, Any]] = []
    
    def adapt_behavior(
        self,
        context: SituationalContext,
        analysis: Dict[str, Any],
        intent: str
    ) -> Dict[str, Any]:
        """Adapt behavior for current context."""
        adaptation = {
            'original_intent': intent,
            'adapted_response': self._adapt_response(context, analysis, intent),
            'timing': self._adapt_timing(context, analysis),
            'modality': self._adapt_modality(context, analysis),
            'verbosity': self._adapt_verbosity(context, analysis),
            'personalization': self._adapt_personalization(context)
        }
        
        self.adaptation_history.append({
            'timestamp': context.timestamp,
            'context_id': context.context_id,
            'adaptation': adaptation
        })
        
        return adaptation
    
    def _adapt_response(
        self,
        context: SituationalContext,
        analysis: Dict[str, Any],
        intent: str
    ) -> str:
        """Adapt response content."""
        style = analysis.get('interaction_style', 'detailed')
        
        if style == 'glanceable':
            return f"[Brief] {intent[:30]}..."
        elif style == 'concise':
            return f"[Quick] {intent[:50]}"
        elif style == 'voice':
            return f"[Voice-optimized] {intent}"
        else:
            return f"[Detailed] {intent} - with full context"
    
    def _adapt_timing(
        self,
        context: SituationalContext,
        analysis: Dict[str, Any]
    ) -> str:
        """Adapt timing of interaction."""
        disruption = analysis.get('disruption_score', 0.5)
        
        if disruption > 0.7:
            return 'defer'  # Postpone to better time
        elif disruption > 0.4:
            return 'batch'  # Batch with other notifications
        else:
            return 'immediate'
    
    def _adapt_modality(
        self,
        context: SituationalContext,
        analysis: Dict[str, Any]
    ) -> str:
        """Adapt interaction modality."""
        notification = analysis.get('preferred_notification', 'sound')
        
        if context.device.device_type == 'watch':
            return 'haptic'
        elif notification == 'silent':
            return 'none'
        elif notification == 'visual':
            return 'visual_only'
        else:
            return notification
    
    def _adapt_verbosity(
        self,
        context: SituationalContext,
        analysis: Dict[str, Any]
    ) -> str:
        """Adapt verbosity level."""
        attention = analysis.get('attention_availability', 1.0)
        
        if attention < 0.3:
            return 'minimal'
        elif attention < 0.6:
            return 'compact'
        else:
            return 'full'
    
    def _adapt_personalization(self, context: SituationalContext) -> Dict[str, Any]:
        """Adapt personalization based on context."""
        return {
            'greet_by_name': context.social.interaction_mode == 'alone',
            'use_casual_tone': context.spatial.location_type == LocationType.HOME,
            'show_details': context.activity.current_activity == ActivityType.RELAXING,
            'enable_proactive': context.temporal.is_weekend
        }


class ContextPredictor:
    """Predicts future context based on patterns."""
    
    def __init__(self):
        self.context_patterns: Dict[str, List[SituationalContext]] = {}
    
    def learn_pattern(self, context: SituationalContext):
        """Learn from context pattern."""
        # Group by time of day and day of week
        key = f"{context.temporal.time_of_day.value}_{context.temporal.day_of_week}"
        
        if key not in self.context_patterns:
            self.context_patterns[key] = []
        
        self.context_patterns[key].append(context)
    
    def predict_context(
        self,
        current_context: SituationalContext,
        minutes_ahead: int = 60
    ) -> Optional[Dict[str, Any]]:
        """Predict likely context in future."""
        key = f"{current_context.temporal.time_of_day.value}_{current_context.temporal.day_of_week}"
        
        if key not in self.context_patterns or len(self.context_patterns[key]) < 2:
            return None
        
        # Analyze patterns
        patterns = self.context_patterns[key]
        
        # Most common next location
        locations = [c.spatial.location_type for c in patterns]
        predicted_location = max(set(locations), key=locations.count)
        
        # Most common next activity
        activities = [c.activity.current_activity for c in patterns]
        predicted_activity = max(set(activities), key=activities.count)
        
        return {
            'predicted_location': predicted_location.value,
            'predicted_activity': predicted_activity.value,
            'confidence': len(patterns) / 10.0,  # More patterns = more confidence
            'minutes_ahead': minutes_ahead
        }


class SituationalContextAgent:
    """Agent for situational context awareness and adaptation."""
    
    def __init__(self):
        self.sensor = ContextSensor()
        self.analyzer = ContextAnalyzer()
        self.behavior_engine = AdaptiveBehaviorEngine()
        self.predictor = ContextPredictor()
        self.current_context: Optional[SituationalContext] = None
    
    def update_context(self, user_data: Dict[str, Any]) -> SituationalContext:
        """Update with new context data."""
        context = self.sensor.sense_context(user_data)
        self.current_context = context
        
        # Learn pattern for prediction
        self.predictor.learn_pattern(context)
        
        return context
    
    def process_intent(
        self,
        intent: str,
        user_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process intent with context awareness."""
        # Update context
        context = self.update_context(user_data)
        
        # Analyze context
        analysis = self.analyzer.analyze_context(context)
        
        # Adapt behavior
        adaptation = self.behavior_engine.adapt_behavior(context, analysis, intent)
        
        # Predict future context
        prediction = self.predictor.predict_context(context)
        
        return {
            'context': context,
            'analysis': analysis,
            'adaptation': adaptation,
            'prediction': prediction,
            'context_summary': context.get_context_summary()
        }
    
    def should_interrupt(self, urgency: str = 'normal') -> bool:
        """Determine if interruption is appropriate."""
        if not self.current_context:
            return True
        
        analysis = self.analyzer.analyze_context(self.current_context)
        disruption = analysis.get('disruption_score', 0.5)
        
        if urgency == 'critical':
            return True
        elif urgency == 'high':
            return disruption < 0.7
        elif urgency == 'normal':
            return disruption < 0.4
        else:  # low urgency
            return disruption < 0.2
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get context statistics."""
        if not self.analyzer.context_history:
            return {'total_contexts': 0}
        
        contexts = self.analyzer.context_history
        
        # Activity distribution
        activities = {}
        for ctx in contexts:
            act = ctx.activity.current_activity.value
            activities[act] = activities.get(act, 0) + 1
        
        # Location distribution
        locations = {}
        for ctx in contexts:
            loc = ctx.spatial.location_type.value
            locations[loc] = locations.get(loc, 0) + 1
        
        # Average disruption
        disruptions = []
        for ctx in contexts:
            analysis = self.analyzer.analyze_context(ctx)
            disruptions.append(analysis['disruption_score'])
        
        avg_disruption = sum(disruptions) / len(disruptions) if disruptions else 0
        
        return {
            'total_contexts': len(contexts),
            'activity_distribution': activities,
            'location_distribution': locations,
            'average_disruption': avg_disruption,
            'adaptations_made': len(self.behavior_engine.adaptation_history),
            'patterns_learned': len(self.predictor.context_patterns)
        }


def demonstrate_situational_context():
    """Demonstrate the Situational Context Agent."""
    print("=" * 60)
    print("Situational Context Agent Demonstration")
    print("=" * 60)
    
    agent = SituationalContextAgent()
    
    print("\n1. MORNING AT HOME - WORKING")
    print("-" * 60)
    
    result = agent.process_intent(
        "Check my schedule for today",
        {
            'location_type': 'home',
            'activity': 'working',
            'is_indoor': True,
            'noise_level': 'quiet',
            'device_type': 'laptop',
            'battery_level': 95.0,
            'screen_on': True,
            'people_nearby': [],
            'interaction_mode': 'alone',
            'social_situation': 'private'
        }
    )
    
    print(f"Context: {result['context_summary']}")
    print(f"Disruption Score: {result['analysis']['disruption_score']:.2f}")
    print(f"Attention Available: {result['analysis']['attention_availability']:.2f}")
    print(f"Notification: {result['analysis']['preferred_notification']}")
    print(f"Interaction Style: {result['analysis']['interaction_style']}")
    print(f"Timing: {result['adaptation']['timing']}")
    print(f"Response: {result['adaptation']['adapted_response']}")
    
    print("\n\n2. COMMUTING - IN MOTION")
    print("-" * 60)
    
    result = agent.process_intent(
        "What's the weather today?",
        {
            'location_type': 'commute',
            'activity': 'commuting',
            'is_indoor': False,
            'noise_level': 'loud',
            'device_type': 'phone',
            'battery_level': 45.0,
            'screen_on': True,
            'in_motion': True,
            'people_nearby': ['stranger1', 'stranger2'],
            'interaction_mode': 'alone',
            'social_situation': 'public',
            'weather': 'rainy',
            'temperature': 15.0
        }
    )
    
    print(f"Context: {result['context_summary']}")
    print(f"Privacy Level: {result['analysis']['privacy_level']}")
    print(f"Notification: {result['analysis']['preferred_notification']}")
    print(f"Interaction Style: {result['analysis']['interaction_style']}")
    print(f"Verbosity: {result['adaptation']['verbosity']}")
    print(f"Response: {result['adaptation']['adapted_response']}")
    
    print("\n\n3. AT WORK - IN MEETING")
    print("-" * 60)
    
    result = agent.process_intent(
        "Reminder: Call John at 3pm",
        {
            'location_type': 'work',
            'activity': 'working',
            'is_indoor': True,
            'noise_level': 'moderate',
            'device_type': 'phone',
            'battery_level': 60.0,
            'in_meeting': True,
            'people_nearby': ['colleague1', 'colleague2', 'boss'],
            'interaction_mode': 'with_group',
            'social_situation': 'professional',
            'available': False
        }
    )
    
    print(f"Context: {result['context_summary']}")
    print(f"Disruption Score: {result['analysis']['disruption_score']:.2f}")
    print(f"Notification: {result['analysis']['preferred_notification']}")
    print(f"Timing: {result['adaptation']['timing']}")
    print(f"Modality: {result['adaptation']['modality']}")
    print(f"Response: {result['adaptation']['adapted_response']}")
    
    print("\n\n4. NIGHT AT HOME - RELAXING")
    print("-" * 60)
    
    result = agent.process_intent(
        "Show me movie recommendations",
        {
            'location_type': 'home',
            'activity': 'relaxing',
            'is_indoor': True,
            'noise_level': 'quiet',
            'light_level': 'dim',
            'device_type': 'tablet',
            'battery_level': 80.0,
            'people_nearby': ['family_member'],
            'interaction_mode': 'with_one',
            'social_situation': 'private'
        }
    )
    
    print(f"Context: {result['context_summary']}")
    print(f"Attention Available: {result['analysis']['attention_availability']:.2f}")
    print(f"Urgency Level: {result['analysis']['urgency_level']}")
    print(f"Interaction Style: {result['analysis']['interaction_style']}")
    print(f"Verbosity: {result['adaptation']['verbosity']}")
    print(f"Personalization:")
    for key, value in result['adaptation']['personalization'].items():
        print(f"  {key}: {value}")
    
    print("\n\n5. INTERRUPTION MANAGEMENT")
    print("-" * 60)
    
    scenarios = [
        ('critical', 'Emergency alert'),
        ('high', 'Important meeting reminder'),
        ('normal', 'News update'),
        ('low', 'Social media notification')
    ]
    
    # Update to meeting context
    agent.update_context({
        'location_type': 'work',
        'activity': 'working',
        'in_meeting': True,
        'social_situation': 'professional'
    })
    
    print("Current context: In meeting at work")
    print("\nInterruption decisions:")
    for urgency, message in scenarios:
        should_interrupt = agent.should_interrupt(urgency)
        print(f"  {urgency.upper()}: '{message}' -> {'✓ Interrupt' if should_interrupt else '✗ Defer'}")
    
    print("\n\n6. CONTEXT PREDICTION")
    print("-" * 60)
    
    # Add more contexts for pattern learning
    for _ in range(3):
        agent.update_context({
            'location_type': 'home',
            'activity': 'relaxing'
        })
    
    prediction = None
    if agent.current_context:
        prediction = agent.predictor.predict_context(agent.current_context)
    
    if prediction:
        print(f"Predicted location: {prediction['predicted_location']}")
        print(f"Predicted activity: {prediction['predicted_activity']}")
        print(f"Confidence: {prediction['confidence']:.2f}")
        print(f"Time ahead: {prediction['minutes_ahead']} minutes")
    
    print("\n\n7. STATISTICS")
    print("-" * 60)
    
    stats = agent.get_statistics()
    print(f"  Total Contexts: {stats['total_contexts']}")
    print(f"  Average Disruption: {stats['average_disruption']:.2f}")
    print(f"  Adaptations Made: {stats['adaptations_made']}")
    print(f"  Patterns Learned: {stats['patterns_learned']}")
    
    print(f"\n  Activity Distribution:")
    for activity, count in sorted(stats['activity_distribution'].items()):
        print(f"    {activity}: {count}")
    
    print(f"\n  Location Distribution:")
    for location, count in sorted(stats['location_distribution'].items()):
        print(f"    {location}: {count}")
    
    print("\n" + "=" * 60)
    print("🎉 Pattern 132 Complete!")
    print("Context & Grounding Category: 75%")
    print("132/170 patterns implemented (77.6%)!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_situational_context()
