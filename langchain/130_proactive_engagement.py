"""
Pattern 130: Proactive Engagement

Description:
    The Proactive Engagement pattern enables agents to anticipate user needs and
    initiate interactions rather than passively waiting for user input. Instead of
    purely reactive behavior, proactive agents monitor context, identify opportunities
    for assistance, predict upcoming needs, and engage users at appropriate moments
    to provide value before being explicitly asked.
    
    Proactive engagement is essential for creating helpful, attentive AI assistants
    that feel intelligent and considerate. The pattern addresses challenges like
    determining when to interrupt, avoiding being annoying, predicting actual needs
    vs. false positives, timing interventions appropriately, and personalizing
    proactivity levels to user preferences.
    
    This pattern includes mechanisms for need prediction, opportunity detection,
    engagement timing, interruption management, value assessment, personalization,
    and learning from user responses to proactive suggestions.

Key Components:
    1. Context Monitor: Tracks user activity and environment
    2. Need Predictor: Anticipates upcoming needs
    3. Opportunity Detector: Identifies engagement moments
    4. Timing Manager: Determines when to engage
    5. Value Assessor: Evaluates potential intervention value
    6. Personalization Engine: Adapts to user preferences
    7. Feedback Learner: Learns from user responses

Engagement Triggers:
    1. Predicted Needs: Anticipate what user will need
    2. Detected Problems: Identify issues proactively
    3. Opportunities: Suggest improvements or shortcuts
    4. Reminders: Timely notifications
    5. Contextual Tips: Relevant guidance
    6. Status Updates: Important changes
    7. Recommendations: Personalized suggestions

Timing Strategies:
    1. Immediate: Critical or time-sensitive
    2. Opportune Moment: Natural break points
    3. Scheduled: Predetermined times
    4. Threshold-Based: When confidence is high
    5. User-Initiated Flow: During interactions
    6. Idle Detection: When user is inactive

Use Cases:
    - Smart assistants (Alexa, Siri, Google Assistant)
    - Customer service chatbots
    - Healthcare monitoring systems
    - Educational tutors
    - Personal productivity assistants
    - Smart home automation
    - Enterprise knowledge management

Advantages:
    - Improves user experience through anticipation
    - Reduces cognitive load on users
    - Catches issues before they escalate
    - Provides timely, relevant assistance
    - Increases perceived intelligence
    - Builds trust through helpfulness
    - Enhances productivity

Challenges:
    - Avoiding annoying interruptions
    - False positive predictions
    - Privacy and monitoring concerns
    - Determining appropriate timing
    - Balancing proactivity with passivity
    - Personalizing to individual preferences
    - Managing user attention

LangChain Implementation:
    This implementation uses LangChain for:
    - Context analysis and need prediction
    - Opportunity identification
    - Value assessment of interventions
    - Personalized engagement strategies
    
Production Considerations:
    - Implement user preference controls
    - Track engagement acceptance rates
    - Use machine learning for need prediction
    - Respect "do not disturb" modes
    - Provide easy dismissal options
    - Learn from user feedback
    - Monitor intervention frequency
    - Enable granular notification settings
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class EngagementType(Enum):
    """Types of proactive engagements."""
    PREDICTED_NEED = "predicted_need"
    PROBLEM_DETECTION = "problem_detection"
    OPPORTUNITY = "opportunity"
    REMINDER = "reminder"
    TIP = "tip"
    STATUS_UPDATE = "status_update"
    RECOMMENDATION = "recommendation"


class EngagementPriority(Enum):
    """Priority levels for engagements."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class UserResponse(Enum):
    """User responses to proactive engagements."""
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    DEFERRED = "deferred"
    IGNORED = "ignored"


@dataclass
class ContextSignal:
    """
    A signal from the user context.
    
    Attributes:
        signal_type: Type of signal
        content: Signal content
        timestamp: When detected
        confidence: Detection confidence
        metadata: Additional information
    """
    signal_type: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProactiveEngagement:
    """
    A proactive engagement with the user.
    
    Attributes:
        engagement_id: Unique identifier
        engagement_type: Type of engagement
        priority: Urgency level
        message: Engagement message
        suggested_action: Recommended action
        value_score: Estimated value to user
        timing_score: Appropriateness of timing
        triggers: What triggered this
        created_at: When created
        engaged_at: When delivered to user
        response: User response
    """
    engagement_id: str
    engagement_type: EngagementType
    priority: EngagementPriority
    message: str
    suggested_action: Optional[str] = None
    value_score: float = 0.5
    timing_score: float = 0.5
    triggers: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    engaged_at: Optional[datetime] = None
    response: Optional[UserResponse] = None


@dataclass
class UserPreferences:
    """
    User preferences for proactive engagement.
    
    Attributes:
        enabled: Whether proactive engagement is on
        min_priority: Minimum priority level to show
        quiet_hours: Hours when engagement is disabled
        max_frequency: Maximum engagements per hour
        preferred_types: Preferred engagement types
    """
    enabled: bool = True
    min_priority: EngagementPriority = EngagementPriority.MEDIUM
    quiet_hours: List[int] = field(default_factory=list)
    max_frequency: int = 5
    preferred_types: List[EngagementType] = field(default_factory=list)


class ProactiveAgent:
    """
    Proactive engagement agent that anticipates needs and initiates interactions.
    
    This agent monitors context, predicts needs, and proactively engages
    users at appropriate moments with helpful suggestions.
    """
    
    def __init__(self, temperature: float = 0.3):
        """
        Initialize proactive agent.
        
        Args:
            temperature: LLM temperature
        """
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.context_signals: List[ContextSignal] = []
        self.engagements: List[ProactiveEngagement] = []
        self.user_preferences = UserPreferences()
        self.engagement_counter = 0
    
    def add_context_signal(
        self,
        signal_type: str,
        content: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContextSignal:
        """
        Add a context signal from monitoring.
        
        Args:
            signal_type: Type of signal
            content: Signal content
            confidence: Detection confidence
            metadata: Additional information
            
        Returns:
            Created signal
        """
        signal = ContextSignal(
            signal_type=signal_type,
            content=content,
            confidence=confidence,
            metadata=metadata or {}
        )
        self.context_signals.append(signal)
        return signal
    
    def predict_need(self, context_window: int = 5) -> Optional[str]:
        """
        Predict user need based on recent context.
        
        Args:
            context_window: Number of recent signals to consider
            
        Returns:
            Predicted need description
        """
        recent_signals = self.context_signals[-context_window:]
        
        if not recent_signals:
            return None
        
        # Build context
        context = "\n".join([
            f"{s.signal_type}: {s.content}"
            for s in recent_signals
        ])
        
        prompt = ChatPromptTemplate.from_template(
            "Based on these recent user activities and context signals:\n\n"
            "{context}\n\n"
            "Predict what the user might need help with next. "
            "Consider patterns, incomplete tasks, and logical next steps.\n\n"
            "Provide:\n"
            "1. Predicted need (one sentence)\n"
            "2. Confidence (0-1)\n"
            "3. Brief reasoning"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        prediction = chain.invoke({"context": context})
        
        return prediction
    
    def detect_opportunity(self, context: str) -> Optional[Dict[str, Any]]:
        """
        Detect opportunity for proactive engagement.
        
        Args:
            context: Current context description
            
        Returns:
            Opportunity details if found
        """
        prompt = ChatPromptTemplate.from_template(
            "Analyze this user context:\n\n"
            "{context}\n\n"
            "Is there an opportunity for helpful proactive engagement?\n"
            "Consider:\n"
            "- Shortcuts or efficiency improvements\n"
            "- Potential problems to avoid\n"
            "- Helpful tips or information\n"
            "- Relevant recommendations\n\n"
            "Respond with:\n"
            "OPPORTUNITY: yes/no\n"
            "TYPE: [predicted_need/problem_detection/opportunity/tip/recommendation]\n"
            "MESSAGE: [brief helpful message]\n"
            "VALUE: [0-1 score]"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"context": context})
        
        # Parse result
        lines = result.strip().split('\n')
        opportunity_dict = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                opportunity_dict[key.strip().lower()] = value.strip()
        
        if opportunity_dict.get('opportunity', '').lower() == 'yes':
            return opportunity_dict
        
        return None
    
    def assess_timing(self, engagement_type: EngagementType) -> float:
        """
        Assess if timing is appropriate for engagement.
        
        Args:
            engagement_type: Type of engagement
            
        Returns:
            Timing appropriateness score (0-1)
        """
        current_hour = datetime.now().hour
        
        # Check quiet hours
        if current_hour in self.user_preferences.quiet_hours:
            return 0.1
        
        # Check recent engagement frequency
        recent_hour = datetime.now() - timedelta(hours=1)
        recent_engagements = [
            e for e in self.engagements
            if e.engaged_at and e.engaged_at > recent_hour
        ]
        
        if len(recent_engagements) >= self.user_preferences.max_frequency:
            return 0.2
        
        # Critical engagements always have good timing
        # (override frequency limits for critical items)
        
        # Higher score for fewer recent engagements
        frequency_score = 1.0 - (len(recent_engagements) / self.user_preferences.max_frequency)
        
        return max(0.3, frequency_score)
    
    def create_engagement(
        self,
        engagement_type: EngagementType,
        message: str,
        priority: EngagementPriority = EngagementPriority.MEDIUM,
        suggested_action: Optional[str] = None,
        triggers: Optional[List[str]] = None
    ) -> ProactiveEngagement:
        """
        Create a proactive engagement.
        
        Args:
            engagement_type: Type of engagement
            message: Engagement message
            priority: Priority level
            suggested_action: Suggested action
            triggers: What triggered this
            
        Returns:
            Created engagement
        """
        self.engagement_counter += 1
        
        # Assess value and timing
        timing_score = self.assess_timing(engagement_type)
        
        # Simple value heuristic (could be ML model in production)
        value_score = {
            EngagementPriority.CRITICAL: 0.9,
            EngagementPriority.HIGH: 0.7,
            EngagementPriority.MEDIUM: 0.5,
            EngagementPriority.LOW: 0.3
        }.get(priority, 0.5)
        
        engagement = ProactiveEngagement(
            engagement_id=f"eng_{self.engagement_counter}",
            engagement_type=engagement_type,
            priority=priority,
            message=message,
            suggested_action=suggested_action,
            value_score=value_score,
            timing_score=timing_score,
            triggers=triggers or []
        )
        
        self.engagements.append(engagement)
        return engagement
    
    def should_engage(self, engagement: ProactiveEngagement) -> bool:
        """
        Determine if agent should deliver engagement.
        
        Args:
            engagement: Engagement to evaluate
            
        Returns:
            Whether to engage
        """
        # Check if proactive engagement is enabled
        if not self.user_preferences.enabled:
            return False
        
        # Check priority threshold
        priority_levels = {
            EngagementPriority.CRITICAL: 4,
            EngagementPriority.HIGH: 3,
            EngagementPriority.MEDIUM: 2,
            EngagementPriority.LOW: 1
        }
        
        if priority_levels.get(engagement.priority, 0) < \
           priority_levels.get(self.user_preferences.min_priority, 2):
            return False
        
        # Check preferred types (if any specified)
        if self.user_preferences.preferred_types and \
           engagement.engagement_type not in self.user_preferences.preferred_types:
            return False
        
        # Check timing score
        if engagement.timing_score < 0.3:
            return False
        
        # Combine value and timing for final decision
        combined_score = (engagement.value_score + engagement.timing_score) / 2
        
        # Critical items always go through
        if engagement.priority == EngagementPriority.CRITICAL:
            return True
        
        return combined_score > 0.5
    
    def engage_user(self, engagement: ProactiveEngagement) -> bool:
        """
        Deliver engagement to user.
        
        Args:
            engagement: Engagement to deliver
            
        Returns:
            Whether engagement was delivered
        """
        if not self.should_engage(engagement):
            return False
        
        engagement.engaged_at = datetime.now()
        return True
    
    def record_response(
        self,
        engagement_id: str,
        response: UserResponse
    ):
        """
        Record user response to engagement.
        
        Args:
            engagement_id: Engagement identifier
            response: User response
        """
        for engagement in self.engagements:
            if engagement.engagement_id == engagement_id:
                engagement.response = response
                break
    
    def get_acceptance_rate(self) -> float:
        """Get acceptance rate for delivered engagements."""
        delivered = [e for e in self.engagements if e.engaged_at]
        
        if not delivered:
            return 0.0
        
        accepted = [e for e in delivered if e.response == UserResponse.ACCEPTED]
        return len(accepted) / len(delivered)
    
    def get_engagement_summary(self) -> Dict[str, Any]:
        """Get summary of engagement activity."""
        delivered = [e for e in self.engagements if e.engaged_at]
        
        return {
            "total_created": len(self.engagements),
            "total_delivered": len(delivered),
            "delivery_rate": len(delivered) / len(self.engagements) if self.engagements else 0,
            "acceptance_rate": self.get_acceptance_rate(),
            "by_type": {
                et.value: len([e for e in delivered if e.engagement_type == et])
                for et in EngagementType
            },
            "by_priority": {
                pr.value: len([e for e in delivered if e.priority == pr])
                for pr in EngagementPriority
            }
        }


def demonstrate_proactive_engagement():
    """Demonstrate proactive engagement pattern."""
    
    print("=" * 80)
    print("PROACTIVE ENGAGEMENT PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Need prediction
    print("\n" + "=" * 80)
    print("Example 1: Predicting User Needs")
    print("=" * 80)
    
    agent = ProactiveAgent()
    
    # Add context signals
    agent.add_context_signal("user_action", "Opened email application")
    agent.add_context_signal("user_action", "Viewed calendar")
    agent.add_context_signal("user_action", "Searched for 'meeting notes template'")
    agent.add_context_signal("time_context", "10 minutes before scheduled meeting")
    
    print("\nContext signals collected:")
    for signal in agent.context_signals:
        print(f"  • {signal.signal_type}: {signal.content}")
    
    print("\nPredicting user need...")
    prediction = agent.predict_need()
    print(f"\nPrediction:\n{prediction}")
    
    # Example 2: Opportunity detection
    print("\n" + "=" * 80)
    print("Example 2: Detecting Engagement Opportunities")
    print("=" * 80)
    
    agent2 = ProactiveAgent()
    
    context = """
    User is writing a report in Word.
    Document has multiple spelling errors highlighted.
    User has been working for 2 hours without saving.
    Battery level: 15%
    """
    
    print(f"\nContext:\n{context}")
    print("\nDetecting opportunities...")
    
    opportunity = agent2.detect_opportunity(context)
    
    if opportunity:
        print("\nOpportunity detected!")
        print(f"  Type: {opportunity.get('type', 'N/A')}")
        print(f"  Message: {opportunity.get('message', 'N/A')}")
        print(f"  Value: {opportunity.get('value', 'N/A')}")
    
    # Example 3: Creating engagements
    print("\n" + "=" * 80)
    print("Example 3: Creating Proactive Engagements")
    print("=" * 80)
    
    agent3 = ProactiveAgent()
    
    # Create various engagements
    eng1 = agent3.create_engagement(
        EngagementType.REMINDER,
        "Your meeting starts in 5 minutes",
        priority=EngagementPriority.HIGH,
        suggested_action="Join video call",
        triggers=["calendar_event", "time_proximity"]
    )
    
    eng2 = agent3.create_engagement(
        EngagementType.TIP,
        "You can use Ctrl+S to save your document",
        priority=EngagementPriority.LOW,
        triggers=["unsaved_document", "long_edit_session"]
    )
    
    eng3 = agent3.create_engagement(
        EngagementType.PROBLEM_DETECTION,
        "Low battery detected. Please plug in your charger.",
        priority=EngagementPriority.CRITICAL,
        suggested_action="Connect charger",
        triggers=["battery_low"]
    )
    
    print("\nCreated engagements:")
    for eng in agent3.engagements:
        print(f"\n  {eng.engagement_id}:")
        print(f"    Type: {eng.engagement_type.value}")
        print(f"    Priority: {eng.priority.value}")
        print(f"    Message: {eng.message}")
        print(f"    Value Score: {eng.value_score:.2f}")
        print(f"    Timing Score: {eng.timing_score:.2f}")
    
    # Example 4: Timing assessment
    print("\n" + "=" * 80)
    print("Example 4: Timing Assessment")
    print("=" * 80)
    
    agent4 = ProactiveAgent()
    
    # Set quiet hours
    agent4.user_preferences.quiet_hours = [22, 23, 0, 1, 2, 3, 4, 5, 6]
    agent4.user_preferences.max_frequency = 3
    
    print("\nUser preferences:")
    print(f"  Quiet hours: {agent4.user_preferences.quiet_hours}")
    print(f"  Max frequency: {agent4.user_preferences.max_frequency}/hour")
    
    # Create and assess engagement
    eng = agent4.create_engagement(
        EngagementType.RECOMMENDATION,
        "Check out this relevant article",
        priority=EngagementPriority.MEDIUM
    )
    
    print(f"\nEngagement timing score: {eng.timing_score:.2f}")
    print(f"Should engage: {agent4.should_engage(eng)}")
    
    # Example 5: Delivery decision
    print("\n" + "=" * 80)
    print("Example 5: Engagement Delivery Decision")
    print("=" * 80)
    
    agent5 = ProactiveAgent()
    
    # Create different priority engagements
    engagements = [
        agent5.create_engagement(
            EngagementType.TIP,
            "Low priority tip",
            priority=EngagementPriority.LOW
        ),
        agent5.create_engagement(
            EngagementType.REMINDER,
            "Medium priority reminder",
            priority=EngagementPriority.MEDIUM
        ),
        agent5.create_engagement(
            EngagementType.PROBLEM_DETECTION,
            "Critical issue detected",
            priority=EngagementPriority.CRITICAL
        )
    ]
    
    print("\nDelivery decisions:")
    for eng in engagements:
        delivered = agent5.engage_user(eng)
        status = "✓ DELIVERED" if delivered else "✗ HELD"
        print(f"  {eng.priority.value}: {status}")
    
    # Example 6: User feedback learning
    print("\n" + "=" * 80)
    print("Example 6: Learning from User Feedback")
    print("=" * 80)
    
    agent6 = ProactiveAgent()
    
    # Create and deliver engagements
    for i in range(5):
        eng = agent6.create_engagement(
            EngagementType.TIP,
            f"Helpful tip #{i+1}",
            priority=EngagementPriority.MEDIUM
        )
        agent6.engage_user(eng)
    
    # Record user responses
    agent6.record_response(agent6.engagements[0].engagement_id, UserResponse.ACCEPTED)
    agent6.record_response(agent6.engagements[1].engagement_id, UserResponse.DISMISSED)
    agent6.record_response(agent6.engagements[2].engagement_id, UserResponse.ACCEPTED)
    agent6.record_response(agent6.engagements[3].engagement_id, UserResponse.ACCEPTED)
    agent6.record_response(agent6.engagements[4].engagement_id, UserResponse.DISMISSED)
    
    print("\nUser responses recorded:")
    for eng in agent6.engagements:
        if eng.response:
            print(f"  {eng.engagement_id}: {eng.response.value}")
    
    acceptance_rate = agent6.get_acceptance_rate()
    print(f"\nAcceptance rate: {acceptance_rate:.0%}")
    
    # Example 7: Preference management
    print("\n" + "=" * 80)
    print("Example 7: User Preference Management")
    print("=" * 80)
    
    agent7 = ProactiveAgent()
    
    # User configures preferences
    agent7.user_preferences.enabled = True
    agent7.user_preferences.min_priority = EngagementPriority.HIGH
    agent7.user_preferences.preferred_types = [
        EngagementType.REMINDER,
        EngagementType.PROBLEM_DETECTION
    ]
    
    print("\nUser preferences:")
    print(f"  Enabled: {agent7.user_preferences.enabled}")
    print(f"  Min priority: {agent7.user_preferences.min_priority.value}")
    print(f"  Preferred types: {[t.value for t in agent7.user_preferences.preferred_types]}")
    
    # Test different engagements
    test_engagements = [
        ("TIP (LOW)", EngagementType.TIP, EngagementPriority.LOW),
        ("REMINDER (HIGH)", EngagementType.REMINDER, EngagementPriority.HIGH),
        ("OPPORTUNITY (HIGH)", EngagementType.OPPORTUNITY, EngagementPriority.HIGH),
        ("PROBLEM (CRITICAL)", EngagementType.PROBLEM_DETECTION, EngagementPriority.CRITICAL)
    ]
    
    print("\nEngagement filtering:")
    for name, eng_type, priority in test_engagements:
        eng = agent7.create_engagement(eng_type, f"Test message", priority=priority)
        would_deliver = agent7.should_engage(eng)
        status = "✓ Would deliver" if would_deliver else "✗ Would filter"
        print(f"  {name}: {status}")
    
    # Example 8: System summary
    print("\n" + "=" * 80)
    print("Example 8: Engagement System Summary")
    print("=" * 80)
    
    agent8 = ProactiveAgent()
    
    # Simulate various engagements
    for i in range(10):
        eng_type = [EngagementType.TIP, EngagementType.REMINDER, EngagementType.RECOMMENDATION][i % 3]
        priority = [EngagementPriority.LOW, EngagementPriority.MEDIUM, EngagementPriority.HIGH][i % 3]
        
        eng = agent8.create_engagement(eng_type, f"Message {i+1}", priority=priority)
        
        if agent8.engage_user(eng):
            # Simulate responses
            response = UserResponse.ACCEPTED if i % 2 == 0 else UserResponse.DISMISSED
            agent8.record_response(eng.engagement_id, response)
    
    summary = agent8.get_engagement_summary()
    
    print("\nENGAGEMENT SUMMARY:")
    print("=" * 60)
    print(f"Total Created: {summary['total_created']}")
    print(f"Total Delivered: {summary['total_delivered']}")
    print(f"Delivery Rate: {summary['delivery_rate']:.0%}")
    print(f"Acceptance Rate: {summary['acceptance_rate']:.0%}")
    
    print("\nBy Type:")
    for eng_type, count in summary['by_type'].items():
        if count > 0:
            print(f"  {eng_type}: {count}")
    
    print("\nBy Priority:")
    for priority, count in summary['by_priority'].items():
        if count > 0:
            print(f"  {priority}: {count}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Proactive Engagement Pattern")
    print("=" * 80)
    
    summary_text = """
    The Proactive Engagement pattern demonstrated:
    
    1. NEED PREDICTION (Example 1):
       - Context signal collection
       - Pattern analysis
       - LLM-based prediction
       - Confidence estimation
       - Anticipatory behavior
    
    2. OPPORTUNITY DETECTION (Example 2):
       - Context analysis
       - Opportunity identification
       - Value assessment
       - Type classification
       - Actionable suggestions
    
    3. ENGAGEMENT CREATION (Example 3):
       - Multiple engagement types
       - Priority assignment
       - Value scoring
       - Timing assessment
       - Trigger tracking
    
    4. TIMING ASSESSMENT (Example 4):
       - Quiet hour respect
       - Frequency limiting
       - Context awareness
       - Appropriateness scoring
       - User preference consideration
    
    5. DELIVERY DECISIONS (Example 5):
       - Priority-based filtering
       - Threshold evaluation
       - Critical override
       - Smart gatekeeping
       - Controlled delivery
    
    6. FEEDBACK LEARNING (Example 6):
       - Response tracking
       - Acceptance rate calculation
       - Pattern learning
       - Continuous improvement
       - Adaptive behavior
    
    7. PREFERENCE MANAGEMENT (Example 7):
       - User control
       - Type filtering
       - Priority thresholds
       - Personalization
       - Preference respect
    
    8. SYSTEM INTEGRATION (Example 8):
       - Comprehensive metrics
       - Type distribution
       - Priority analysis
       - Performance tracking
       - System overview
    
    KEY BENEFITS:
    ✓ Improves UX through anticipation
    ✓ Reduces cognitive load on users
    ✓ Catches issues before they escalate
    ✓ Provides timely, relevant assistance
    ✓ Increases perceived intelligence
    ✓ Builds trust through helpfulness
    ✓ Enhances productivity
    ✓ Creates delightful experiences
    
    USE CASES:
    • Smart assistants (Alexa, Siri)
    • Customer service chatbots
    • Healthcare monitoring
    • Educational tutors
    • Productivity assistants
    • Smart home automation
    • Enterprise knowledge management
    • Preventive maintenance
    
    ENGAGEMENT TYPES:
    → Predicted Needs: Anticipate requirements
    → Problem Detection: Identify issues
    → Opportunities: Suggest improvements
    → Reminders: Timely notifications
    → Tips: Contextual guidance
    → Status Updates: Important changes
    → Recommendations: Personalized suggestions
    
    TIMING STRATEGIES:
    • Immediate: Critical/time-sensitive
    • Opportune Moment: Natural breaks
    • Scheduled: Predetermined times
    • Threshold-Based: High confidence
    • User-Initiated Flow: During interactions
    • Idle Detection: When inactive
    
    BEST PRACTICES:
    1. Implement user preference controls
    2. Track engagement acceptance rates
    3. Use ML for need prediction
    4. Respect "do not disturb" modes
    5. Provide easy dismissal options
    6. Learn from user feedback
    7. Monitor intervention frequency
    8. Enable granular notification settings
    
    TRADE-OFFS:
    • Helpfulness vs. annoyance
    • Proactivity vs. privacy
    • Frequency vs. relevance
    • Automation vs. control
    
    PRODUCTION CONSIDERATIONS:
    → Implement A/B testing for engagement strategies
    → Use ML models for need prediction
    → Track detailed engagement metrics
    → Provide fine-grained user controls
    → Respect platform notification guidelines
    → Implement gradual rollout for new features
    → Monitor user satisfaction scores
    → Enable easy opt-out mechanisms
    → Log all engagement decisions
    → Personalize based on user history
    
    This pattern enables agents to be helpful and attentive by anticipating
    user needs and engaging proactively at appropriate moments, creating a
    more intelligent and considerate user experience.
    """
    
    print(summary_text)


if __name__ == "__main__":
    demonstrate_proactive_engagement()
