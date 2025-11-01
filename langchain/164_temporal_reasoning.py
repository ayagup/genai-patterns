"""
Pattern 164: Temporal Reasoning

Description:
    The Temporal Reasoning pattern enables agents to understand and reason about time-based
    relationships, sequences, durations, and temporal logic. This pattern helps agents solve
    problems involving timelines, schedules, causality, and temporal constraints.

Components:
    1. Timeline Manager: Maintains ordered sequences of events
    2. Temporal Relation Analyzer: Identifies before/after/during/overlapping relationships
    3. Duration Calculator: Computes time intervals and durations
    4. Temporal Constraint Checker: Validates temporal constraints
    5. Sequence Analyzer: Analyzes event sequences and patterns
    6. Temporal Query Engine: Answers time-based questions

Use Cases:
    - Schedule planning and optimization
    - Event sequence analysis
    - Causal reasoning
    - Historical timeline construction
    - Project timeline management
    - Deadline tracking
    - Temporal pattern detection

Benefits:
    - Enables time-aware reasoning
    - Supports scheduling and planning
    - Detects temporal patterns
    - Validates temporal constraints
    - Better understanding of causality

Trade-offs:
    - Complex time representation
    - Timezone handling challenges
    - Requires precise temporal data
    - May need calendar logic
    - Limited by data accuracy

LangChain Implementation:
    Uses LLM for temporal relationship interpretation with structured datetime
    management and interval calculations. Combines symbolic temporal logic with
    natural language understanding for temporal queries.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class TemporalRelation(Enum):
    """Types of temporal relationships (Allen's interval algebra)"""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    CONTAINS = "contains"
    OVERLAPS = "overlaps"
    OVERLAPPED_BY = "overlapped_by"
    MEETS = "meets"
    MET_BY = "met_by"
    STARTS = "starts"
    STARTED_BY = "started_by"
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUALS = "equals"


class TimeUnit(Enum):
    """Units of time measurement"""
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


@dataclass
class TemporalEvent:
    """Represents an event with temporal properties"""
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate duration of the event"""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_point_event(self) -> bool:
        """Check if this is a point event (no duration)"""
        return self.end_time is None or self.start_time == self.end_time


@dataclass
class Timeline:
    """Represents a timeline with multiple events"""
    name: str
    events: List[TemporalEvent] = field(default_factory=list)
    
    def add_event(self, event: TemporalEvent):
        """Add an event to the timeline"""
        self.events.append(event)
        self.events.sort(key=lambda e: e.start_time)
    
    def get_events_in_range(self, start: datetime, end: datetime) -> List[TemporalEvent]:
        """Get all events within a time range"""
        return [e for e in self.events 
                if e.start_time <= end and (e.end_time is None or e.end_time >= start)]


@dataclass
class TemporalQuery:
    """A temporal reasoning query"""
    question: str
    events: List[TemporalEvent]
    context: str = ""


@dataclass
class TemporalAnalysis:
    """Result of temporal reasoning"""
    answer: str
    relationships: List[str]
    timeline: str
    sequences: List[List[str]]
    reasoning: str


class TemporalReasoningAgent:
    """Agent that performs temporal reasoning tasks"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the temporal reasoning agent"""
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Prompts for temporal reasoning
        self.temporal_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a temporal reasoning expert. Analyze events, timelines,
            and temporal relationships. Consider durations, sequences, and causal relationships.
            Be precise about time-based logic."""),
            ("user", """Events and their times:
{event_descriptions}

Question: {question}

Provide a clear answer based on temporal analysis.""")
        ])
        
        self.sequence_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at analyzing event sequences and patterns.
            Identify patterns, detect anomalies, and reason about causality."""),
            ("user", """Event sequence:
{sequence}

Analysis task: {task}

Describe patterns, relationships, and insights.""")
        ])
        
        self.temporal_chain = self.temporal_prompt | self.llm | StrOutputParser()
        self.sequence_chain = self.sequence_prompt | self.llm | StrOutputParser()
    
    def analyze_temporal_query(self, query: TemporalQuery) -> TemporalAnalysis:
        """Analyze a temporal query and return comprehensive analysis"""
        # Identify temporal relationships
        relationships = self._identify_temporal_relationships(query.events)
        
        # Create timeline visualization
        timeline_viz = self._create_timeline_visualization(query.events)
        
        # Identify sequences
        sequences = self._identify_sequences(query.events)
        
        # Create event descriptions
        event_descriptions = self._create_event_descriptions(query.events)
        
        # Get LLM reasoning
        reasoning = self.temporal_chain.invoke({
            "event_descriptions": event_descriptions,
            "question": query.question
        })
        
        return TemporalAnalysis(
            answer=reasoning,
            relationships=relationships,
            timeline=timeline_viz,
            sequences=sequences,
            reasoning=reasoning
        )
    
    def check_temporal_relation(self, event1: TemporalEvent, 
                               event2: TemporalEvent,
                               relation: TemporalRelation) -> bool:
        """Check if a specific temporal relationship holds between two events"""
        e1_start = event1.start_time
        e1_end = event1.end_time or event1.start_time
        e2_start = event2.start_time
        e2_end = event2.end_time or event2.start_time
        
        if relation == TemporalRelation.BEFORE:
            return e1_end < e2_start
        elif relation == TemporalRelation.AFTER:
            return e1_start > e2_end
        elif relation == TemporalRelation.DURING:
            return e1_start > e2_start and e1_end < e2_end
        elif relation == TemporalRelation.CONTAINS:
            return e1_start < e2_start and e1_end > e2_end
        elif relation == TemporalRelation.OVERLAPS:
            return e1_start < e2_start and e1_end > e2_start and e1_end < e2_end
        elif relation == TemporalRelation.OVERLAPPED_BY:
            return e2_start < e1_start and e2_end > e1_start and e2_end < e1_end
        elif relation == TemporalRelation.MEETS:
            return e1_end == e2_start
        elif relation == TemporalRelation.MET_BY:
            return e2_end == e1_start
        elif relation == TemporalRelation.STARTS:
            return e1_start == e2_start and e1_end < e2_end
        elif relation == TemporalRelation.STARTED_BY:
            return e1_start == e2_start and e1_end > e2_end
        elif relation == TemporalRelation.FINISHES:
            return e1_start > e2_start and e1_end == e2_end
        elif relation == TemporalRelation.FINISHED_BY:
            return e1_start < e2_start and e1_end == e2_end
        elif relation == TemporalRelation.EQUALS:
            return e1_start == e2_start and e1_end == e2_end
        
        return False
    
    def find_overlapping_events(self, events: List[TemporalEvent]) -> List[Tuple[str, str]]:
        """Find all pairs of overlapping events"""
        overlaps = []
        
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                if self._events_overlap(event1, event2):
                    overlaps.append((event1.name, event2.name))
        
        return overlaps
    
    def find_gaps(self, events: List[TemporalEvent]) -> List[Tuple[datetime, datetime]]:
        """Find time gaps between consecutive events"""
        if not events:
            return []
        
        sorted_events = sorted(events, key=lambda e: e.start_time)
        gaps = []
        
        for i in range(len(sorted_events) - 1):
            current_end = sorted_events[i].end_time or sorted_events[i].start_time
            next_start = sorted_events[i + 1].start_time
            
            if current_end < next_start:
                gaps.append((current_end, next_start))
        
        return gaps
    
    def calculate_total_duration(self, events: List[TemporalEvent],
                                merge_overlaps: bool = False) -> timedelta:
        """Calculate total duration of events"""
        if not events:
            return timedelta(0)
        
        if not merge_overlaps:
            # Simple sum of all durations
            total = timedelta(0)
            for event in events:
                if event.duration:
                    total += event.duration
            return total
        else:
            # Merge overlapping intervals first
            intervals = []
            for event in events:
                start = event.start_time
                end = event.end_time or event.start_time
                intervals.append((start, end))
            
            # Sort and merge
            intervals.sort()
            merged = []
            current_start, current_end = intervals[0]
            
            for start, end in intervals[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = start, end
            
            merged.append((current_start, current_end))
            
            # Calculate total
            total = timedelta(0)
            for start, end in merged:
                total += end - start
            
            return total
    
    def detect_temporal_patterns(self, events: List[TemporalEvent]) -> Dict[str, Any]:
        """Detect patterns in temporal data"""
        if not events:
            return {"patterns": []}
        
        sorted_events = sorted(events, key=lambda e: e.start_time)
        
        patterns = {
            "total_events": len(events),
            "time_span": {
                "start": sorted_events[0].start_time,
                "end": sorted_events[-1].end_time or sorted_events[-1].start_time
            },
            "regular_intervals": self._detect_regular_intervals(sorted_events),
            "clusters": self._detect_time_clusters(sorted_events),
            "gaps": len(self.find_gaps(events)),
            "overlaps": len(self.find_overlapping_events(events))
        }
        
        return patterns
    
    def _events_overlap(self, event1: TemporalEvent, event2: TemporalEvent) -> bool:
        """Check if two events overlap in time"""
        e1_start = event1.start_time
        e1_end = event1.end_time or event1.start_time
        e2_start = event2.start_time
        e2_end = event2.end_time or event2.start_time
        
        return not (e1_end < e2_start or e2_end < e1_start)
    
    def _identify_temporal_relationships(self, events: List[TemporalEvent]) -> List[str]:
        """Identify temporal relationships between events"""
        relationships = []
        
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                # Check key relationships
                if self.check_temporal_relation(event1, event2, TemporalRelation.BEFORE):
                    relationships.append(f"{event1.name} before {event2.name}")
                elif self.check_temporal_relation(event1, event2, TemporalRelation.DURING):
                    relationships.append(f"{event1.name} during {event2.name}")
                elif self.check_temporal_relation(event1, event2, TemporalRelation.OVERLAPS):
                    relationships.append(f"{event1.name} overlaps {event2.name}")
                elif self.check_temporal_relation(event1, event2, TemporalRelation.MEETS):
                    relationships.append(f"{event1.name} meets {event2.name}")
        
        return relationships
    
    def _identify_sequences(self, events: List[TemporalEvent]) -> List[List[str]]:
        """Identify sequences of consecutive events"""
        if not events:
            return []
        
        sorted_events = sorted(events, key=lambda e: e.start_time)
        sequences = []
        current_sequence = [sorted_events[0].name]
        
        for i in range(1, len(sorted_events)):
            prev_end = sorted_events[i-1].end_time or sorted_events[i-1].start_time
            curr_start = sorted_events[i].start_time
            
            # Check if events are consecutive (within 1 hour)
            if (curr_start - prev_end) < timedelta(hours=1):
                current_sequence.append(sorted_events[i].name)
            else:
                if len(current_sequence) > 1:
                    sequences.append(current_sequence)
                current_sequence = [sorted_events[i].name]
        
        if len(current_sequence) > 1:
            sequences.append(current_sequence)
        
        return sequences
    
    def _detect_regular_intervals(self, sorted_events: List[TemporalEvent]) -> Optional[timedelta]:
        """Detect if events occur at regular intervals"""
        if len(sorted_events) < 3:
            return None
        
        intervals = []
        for i in range(1, len(sorted_events)):
            interval = sorted_events[i].start_time - sorted_events[i-1].start_time
            intervals.append(interval)
        
        # Check if intervals are similar (within 10% variance)
        avg_interval = sum(intervals, timedelta(0)) / len(intervals)
        variance = sum((i - avg_interval).total_seconds()**2 for i in intervals) / len(intervals)
        
        if variance < (avg_interval.total_seconds() * 0.1)**2:
            return avg_interval
        
        return None
    
    def _detect_time_clusters(self, sorted_events: List[TemporalEvent]) -> int:
        """Detect clusters of events happening close together"""
        if len(sorted_events) < 2:
            return len(sorted_events)
        
        clusters = 1
        for i in range(1, len(sorted_events)):
            gap = sorted_events[i].start_time - sorted_events[i-1].start_time
            if gap > timedelta(hours=2):  # New cluster if gap > 2 hours
                clusters += 1
        
        return clusters
    
    def _create_event_descriptions(self, events: List[TemporalEvent]) -> str:
        """Create textual descriptions of events"""
        descriptions = []
        for event in sorted(events, key=lambda e: e.start_time):
            desc = f"{event.name}: {event.start_time.strftime('%Y-%m-%d %H:%M')}"
            if event.end_time:
                desc += f" to {event.end_time.strftime('%Y-%m-%d %H:%M')}"
                desc += f" (duration: {event.duration})"
            if event.description:
                desc += f" - {event.description}"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def _create_timeline_visualization(self, events: List[TemporalEvent]) -> str:
        """Create ASCII timeline visualization"""
        if not events:
            return "No events to visualize"
        
        sorted_events = sorted(events, key=lambda e: e.start_time)
        
        lines = ["Timeline:", "=" * 60]
        
        for event in sorted_events:
            time_str = event.start_time.strftime('%Y-%m-%d %H:%M')
            if event.end_time:
                duration_str = f" [{event.duration}]"
            else:
                duration_str = " [instant]"
            
            lines.append(f"  {time_str} | {event.name}{duration_str}")
        
        return "\n".join(lines)


def demonstrate_temporal_reasoning():
    """Demonstrate temporal reasoning capabilities"""
    print("=" * 80)
    print("TEMPORAL REASONING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = TemporalReasoningAgent()
    
    # Example 1: Project timeline analysis
    print("\n" + "=" * 80)
    print("Example 1: Project Timeline Analysis")
    print("=" * 80)
    
    base_time = datetime(2024, 1, 1, 9, 0)
    
    project_events = [
        TemporalEvent("Planning", base_time, base_time + timedelta(days=7), "Initial planning phase"),
        TemporalEvent("Design", base_time + timedelta(days=7), base_time + timedelta(days=14), "Design phase"),
        TemporalEvent("Development", base_time + timedelta(days=14), base_time + timedelta(days=45), "Development phase"),
        TemporalEvent("Testing", base_time + timedelta(days=40), base_time + timedelta(days=52), "Testing phase overlaps with dev"),
        TemporalEvent("Deployment", base_time + timedelta(days=52), base_time + timedelta(days=53), "Final deployment")
    ]
    
    query = TemporalQuery(
        question="What is the timeline of the project and are there any overlapping phases?",
        events=project_events,
        context="Software project timeline"
    )
    
    analysis = agent.analyze_temporal_query(query)
    
    print(f"\nQuery: {query.question}")
    print(f"\n{analysis.timeline}")
    print(f"\nTemporal Relationships:")
    for rel in analysis.relationships[:5]:
        print(f"  - {rel}")
    print(f"\nAnalysis:\n{analysis.answer}")
    
    # Example 2: Overlapping events detection
    print("\n" + "=" * 80)
    print("Example 2: Overlapping Events Detection")
    print("=" * 80)
    
    meeting_events = [
        TemporalEvent("Team Meeting", datetime(2024, 1, 15, 10, 0), datetime(2024, 1, 15, 11, 0)),
        TemporalEvent("Client Call", datetime(2024, 1, 15, 10, 30), datetime(2024, 1, 15, 11, 30)),
        TemporalEvent("Lunch Break", datetime(2024, 1, 15, 12, 0), datetime(2024, 1, 15, 13, 0)),
        TemporalEvent("Code Review", datetime(2024, 1, 15, 14, 0), datetime(2024, 1, 15, 15, 0))
    ]
    
    overlaps = agent.find_overlapping_events(meeting_events)
    
    print("\nSchedule for January 15, 2024:")
    for event in sorted(meeting_events, key=lambda e: e.start_time):
        print(f"  {event.start_time.strftime('%H:%M')}-{event.end_time.strftime('%H:%M')}: {event.name}")
    
    if overlaps:
        print(f"\nConflicts detected:")
        for event1, event2 in overlaps:
            print(f"  ⚠ {event1} overlaps with {event2}")
    else:
        print("\n✓ No scheduling conflicts detected")
    
    # Example 3: Temporal relationship checking
    print("\n" + "=" * 80)
    print("Example 3: Temporal Relationship Checking")
    print("=" * 80)
    
    event_a = TemporalEvent("Event A", datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 12, 0))
    event_b = TemporalEvent("Event B", datetime(2024, 1, 1, 11, 0), datetime(2024, 1, 1, 11, 30))
    event_c = TemporalEvent("Event C", datetime(2024, 1, 1, 12, 0), datetime(2024, 1, 1, 13, 0))
    
    relations_to_check = [
        (event_a, event_b, TemporalRelation.CONTAINS),
        (event_b, event_a, TemporalRelation.DURING),
        (event_a, event_c, TemporalRelation.MEETS),
        (event_c, event_a, TemporalRelation.MET_BY),
        (event_a, event_b, TemporalRelation.OVERLAPS)
    ]
    
    print("\nChecking temporal relationships:")
    for evt1, evt2, relation in relations_to_check:
        holds = agent.check_temporal_relation(evt1, evt2, relation)
        symbol = "✓" if holds else "✗"
        print(f"  {symbol} {evt1.name} {relation.value} {evt2.name}: {holds}")
    
    # Example 4: Duration calculations
    print("\n" + "=" * 80)
    print("Example 4: Duration Calculations")
    print("=" * 80)
    
    work_events = [
        TemporalEvent("Task 1", datetime(2024, 1, 1, 9, 0), datetime(2024, 1, 1, 11, 0)),
        TemporalEvent("Task 2", datetime(2024, 1, 1, 10, 30), datetime(2024, 1, 1, 12, 0)),
        TemporalEvent("Task 3", datetime(2024, 1, 1, 13, 0), datetime(2024, 1, 1, 15, 0)),
    ]
    
    total_simple = agent.calculate_total_duration(work_events, merge_overlaps=False)
    total_merged = agent.calculate_total_duration(work_events, merge_overlaps=True)
    
    print("\nWork events:")
    for event in work_events:
        print(f"  {event.name}: {event.duration}")
    
    print(f"\nTotal duration (simple sum): {total_simple}")
    print(f"Total duration (merged overlaps): {total_merged}")
    print(f"Overlap time: {total_simple - total_merged}")
    
    # Example 5: Gap detection
    print("\n" + "=" * 80)
    print("Example 5: Gap Detection")
    print("=" * 80)
    
    sparse_events = [
        TemporalEvent("Morning Session", datetime(2024, 1, 1, 9, 0), datetime(2024, 1, 1, 10, 0)),
        TemporalEvent("Afternoon Session", datetime(2024, 1, 1, 14, 0), datetime(2024, 1, 1, 15, 0)),
        TemporalEvent("Evening Session", datetime(2024, 1, 1, 19, 0), datetime(2024, 1, 1, 20, 0))
    ]
    
    gaps = agent.find_gaps(sparse_events)
    
    print("\nEvents:")
    for event in sparse_events:
        print(f"  {event.start_time.strftime('%H:%M')}-{event.end_time.strftime('%H:%M')}: {event.name}")
    
    print(f"\nGaps detected: {len(gaps)}")
    for start, end in gaps:
        duration = end - start
        print(f"  Gap from {start.strftime('%H:%M')} to {end.strftime('%H:%M')} ({duration})")
    
    # Example 6: Pattern detection
    print("\n" + "=" * 80)
    print("Example 6: Pattern Detection")
    print("=" * 80)
    
    regular_events = [
        TemporalEvent(f"Daily Standup {i}", datetime(2024, 1, i, 9, 0), datetime(2024, 1, i, 9, 15))
        for i in range(1, 6)
    ]
    
    patterns = agent.detect_temporal_patterns(regular_events)
    
    print("\nEvents:")
    for event in regular_events:
        print(f"  {event.start_time.strftime('%Y-%m-%d %H:%M')}: {event.name}")
    
    print(f"\nPattern Analysis:")
    print(f"  Total events: {patterns['total_events']}")
    print(f"  Time span: {patterns['time_span']['start'].strftime('%Y-%m-%d')} to {patterns['time_span']['end'].strftime('%Y-%m-%d')}")
    if patterns['regular_intervals']:
        print(f"  Regular interval detected: {patterns['regular_intervals']}")
    print(f"  Number of clusters: {patterns['clusters']}")
    print(f"  Gaps: {patterns['gaps']}")
    print(f"  Overlaps: {patterns['overlaps']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Temporal Reasoning pattern enables agents to:
✓ Understand temporal relationships (before, after, during, overlaps)
✓ Detect scheduling conflicts and overlaps
✓ Calculate durations and time gaps
✓ Identify temporal patterns and sequences
✓ Validate temporal constraints
✓ Reason about causality and event sequences
✓ Analyze timelines and schedules

This pattern is valuable for:
- Project and task scheduling
- Calendar and meeting management
- Event sequence analysis
- Timeline construction and validation
- Causal reasoning
- Temporal pattern detection
- Deadline and milestone tracking
    """)


if __name__ == "__main__":
    demonstrate_temporal_reasoning()
