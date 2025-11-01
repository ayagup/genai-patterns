"""
Pattern 113: Contextual Adaptation

Description:
    Contextual Adaptation enables agents to dynamically adjust their behavior,
    communication style, and strategies based on the detected context. The
    pattern monitors multiple context dimensions (user preferences, environment,
    task type, constraints) and adapts agent behavior accordingly.
    
    This pattern is essential for creating flexible, user-friendly agents that
    can operate effectively across diverse situations and user needs. It combines
    context detection, preference learning, and dynamic behavior adjustment.

Key Components:
    1. Context Detector: Identifies current context dimensions
    2. Preference Learner: Learns user/situation preferences
    3. Behavior Adapter: Adjusts agent behavior
    4. Style Adapter: Modifies communication style
    5. Strategy Selector: Chooses appropriate strategies
    6. Constraint Manager: Handles situational constraints

Context Dimensions:
    - User: Experience level, preferences, goals
    - Environment: Location, time, available resources
    - Task: Type, complexity, urgency, importance
    - Social: Formality, relationship, cultural context
    - Technical: Device, bandwidth, capabilities
    - Domain: Field-specific conventions and norms

Use Cases:
    - Personalized assistants
    - Adaptive learning systems
    - Multi-domain chatbots
    - Context-aware recommendations
    - International applications

LangChain Implementation:
    Uses dynamic prompting, context tracking, and conditional routing to
    adapt agent behavior based on detected context.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.memory import ConversationBufferMemory

load_dotenv()


class UserExperience(Enum):
    """User experience levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class Formality(Enum):
    """Communication formality levels"""
    CASUAL = "casual"
    NEUTRAL = "neutral"
    FORMAL = "formal"
    VERY_FORMAL = "very_formal"


class TaskUrgency(Enum):
    """Task urgency levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Context:
    """Represents current context"""
    
    def __init__(self):
        # User context
        self.user_experience = UserExperience.INTERMEDIATE
        self.user_preferences = {}
        self.user_goals = []
        
        # Environment context
        self.time_of_day = "morning"
        self.location = "office"
        self.available_resources = []
        
        # Task context
        self.task_type = "general"
        self.task_complexity = "medium"
        self.task_urgency = TaskUrgency.MEDIUM
        
        # Social context
        self.formality = Formality.NEUTRAL
        self.relationship = "professional"
        
        # Technical context
        self.device = "desktop"
        self.bandwidth = "high"
        
        # Domain context
        self.domain = "general"
        self.domain_conventions = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "user": {
                "experience": self.user_experience.value,
                "preferences": self.user_preferences,
                "goals": self.user_goals
            },
            "environment": {
                "time": self.time_of_day,
                "location": self.location,
                "resources": self.available_resources
            },
            "task": {
                "type": self.task_type,
                "complexity": self.task_complexity,
                "urgency": self.task_urgency.value
            },
            "social": {
                "formality": self.formality.value,
                "relationship": self.relationship
            },
            "technical": {
                "device": self.device,
                "bandwidth": self.bandwidth
            },
            "domain": {
                "name": self.domain,
                "conventions": self.domain_conventions
            }
        }
    
    def summarize(self) -> str:
        """Generate human-readable context summary"""
        return f"""
Current Context:
- User: {self.user_experience.value} level, wants {', '.join(self.user_goals) if self.user_goals else 'assistance'}
- Environment: {self.time_of_day} at {self.location}
- Task: {self.task_type} ({self.task_complexity} complexity, {self.task_urgency.value} urgency)
- Communication: {self.formality.value} style, {self.relationship} relationship
- Technical: {self.device} with {self.bandwidth} bandwidth
- Domain: {self.domain}
"""


class ContextualAdaptationAgent:
    """Agent that adapts behavior based on context"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.context = Context()
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Adaptive response generation
        self.adaptive_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an adaptive AI assistant that adjusts your behavior based on context.

{context_summary}

Adaptation Guidelines:
- For beginners: Use simple language, explain concepts, provide examples
- For experts: Be concise, use technical terms, assume background knowledge
- For casual contexts: Be friendly and conversational
- For formal contexts: Be professional and structured
- For urgent tasks: Be direct and action-oriented
- For complex tasks: Break down steps and provide detailed guidance

Adapt your response to match the context appropriately."""),
            ("human", "{query}")
        ])
        
        # Context detection prompt
        self.context_detector = ChatPromptTemplate.from_messages([
            ("system", """Analyze the user's query and extract context information.

Determine:
1. User experience level (beginner/intermediate/advanced/expert)
2. Formality expected (casual/neutral/formal/very_formal)
3. Task urgency (low/medium/high/critical)
4. Task type and complexity

Return JSON with: experience, formality, urgency, task_type, complexity"""),
            ("human", "{query}")
        ])
        
        self.response_chain = self.adaptive_prompt | self.llm | StrOutputParser()
        self.detector_chain = self.context_detector | self.llm | JsonOutputParser()
    
    def detect_context(self, query: str) -> Dict[str, Any]:
        """Detect context from user query"""
        try:
            detected = self.detector_chain.invoke({"query": query})
            
            # Update context based on detection
            if "experience" in detected:
                exp_map = {
                    "beginner": UserExperience.BEGINNER,
                    "intermediate": UserExperience.INTERMEDIATE,
                    "advanced": UserExperience.ADVANCED,
                    "expert": UserExperience.EXPERT
                }
                self.context.user_experience = exp_map.get(
                    detected["experience"], UserExperience.INTERMEDIATE
                )
            
            if "formality" in detected:
                form_map = {
                    "casual": Formality.CASUAL,
                    "neutral": Formality.NEUTRAL,
                    "formal": Formality.FORMAL,
                    "very_formal": Formality.VERY_FORMAL
                }
                self.context.formality = form_map.get(
                    detected["formality"], Formality.NEUTRAL
                )
            
            if "urgency" in detected:
                urg_map = {
                    "low": TaskUrgency.LOW,
                    "medium": TaskUrgency.MEDIUM,
                    "high": TaskUrgency.HIGH,
                    "critical": TaskUrgency.CRITICAL
                }
                self.context.task_urgency = urg_map.get(
                    detected["urgency"], TaskUrgency.MEDIUM
                )
            
            if "task_type" in detected:
                self.context.task_type = detected["task_type"]
            
            if "complexity" in detected:
                self.context.task_complexity = detected["complexity"]
            
            return detected
        except Exception as e:
            print(f"Context detection error: {e}")
            return {}
    
    def update_context(self, updates: Dict[str, Any]):
        """Manually update context"""
        for key, value in updates.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)
    
    def respond(self, query: str, auto_detect: bool = True) -> Dict[str, Any]:
        """Generate context-adapted response"""
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        # Auto-detect context if enabled
        if auto_detect:
            detected = self.detect_context(query)
            print(f"\nDetected context: {detected}")
        
        # Show current context
        print(self.context.summarize())
        
        # Generate adapted response
        response = self.response_chain.invoke({
            "context_summary": self.context.summarize(),
            "query": query
        })
        
        print(f"\nAdapted Response:")
        print(response)
        
        return {
            "query": query,
            "context": self.context.to_dict(),
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    def demonstrate_adaptation(self, query: str, contexts: List[Dict[str, Any]]):
        """Demonstrate how response changes with different contexts"""
        print(f"\n{'='*70}")
        print(f"Demonstrating Context Adaptation")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        responses = []
        for i, context_updates in enumerate(contexts, 1):
            print(f"\n--- Context {i}: {context_updates.get('description', '')} ---")
            
            # Reset context
            self.context = Context()
            
            # Apply updates
            for key, value in context_updates.items():
                if key != "description" and hasattr(self.context, key):
                    setattr(self.context, key, value)
            
            # Generate response
            result = self.respond(query, auto_detect=False)
            responses.append(result)
        
        return responses


def demonstrate_contextual_adaptation():
    """Demonstrate contextual adaptation pattern"""
    print("\n" + "="*70)
    print("CONTEXTUAL ADAPTATION DEMONSTRATION")
    print("="*70)
    
    agent = ContextualAdaptationAgent()
    
    # Example 1: Auto-detection of context
    print("\n" + "="*70)
    print("Example 1: Auto-Detection of Context")
    print("="*70)
    
    query1 = "I'm new to Python. Can you explain what a variable is?"
    result1 = agent.respond(query1, auto_detect=True)
    
    query2 = "Explain the time complexity of quicksort's partitioning scheme"
    result2 = agent.respond(query2, auto_detect=True)
    
    # Example 2: Adaptation to different user levels
    print("\n" + "="*70)
    print("Example 2: Adapting to User Experience Level")
    print("="*70)
    
    query = "How does machine learning work?"
    
    contexts = [
        {
            "description": "Beginner",
            "user_experience": UserExperience.BEGINNER,
            "formality": Formality.CASUAL
        },
        {
            "description": "Expert",
            "user_experience": UserExperience.EXPERT,
            "formality": Formality.NEUTRAL
        }
    ]
    
    responses = agent.demonstrate_adaptation(query, contexts)
    
    # Example 3: Adaptation to urgency
    print("\n" + "="*70)
    print("Example 3: Adapting to Task Urgency")
    print("="*70)
    
    query_urgent = "My server is down! How do I restart it?"
    
    contexts_urgent = [
        {
            "description": "Critical Urgency",
            "task_urgency": TaskUrgency.CRITICAL,
            "task_type": "troubleshooting"
        },
        {
            "description": "Low Urgency",
            "task_urgency": TaskUrgency.LOW,
            "task_type": "learning"
        }
    ]
    
    responses_urgent = agent.demonstrate_adaptation(query_urgent, contexts_urgent)
    
    # Example 4: Adaptation to formality
    print("\n" + "="*70)
    print("Example 4: Adapting to Formality Level")
    print("="*70)
    
    query_formal = "What are your capabilities?"
    
    contexts_formal = [
        {
            "description": "Casual",
            "formality": Formality.CASUAL,
            "relationship": "friendly"
        },
        {
            "description": "Very Formal",
            "formality": Formality.VERY_FORMAL,
            "relationship": "professional"
        }
    ]
    
    responses_formal = agent.demonstrate_adaptation(query_formal, contexts_formal)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Contextual Adaptation Pattern demonstrated:

Key Features:
1. Context Detection - Automatically detect context from input
2. Multi-Dimensional Adaptation - Adapt across multiple context dimensions
3. Dynamic Behavior - Adjust communication style and strategy
4. User Personalization - Adapt to user experience and preferences
5. Situational Awareness - Respond appropriately to urgency and formality

Context Dimensions Handled:
- User experience level (beginner to expert)
- Communication formality (casual to very formal)
- Task urgency (low to critical)
- Task complexity and type
- Environment and technical constraints
- Domain-specific conventions

Applications:
- Personalized assistants adapting to user expertise
- Customer service bots adjusting tone
- Educational systems matching student level
- Multi-domain chatbots switching contexts
- International applications respecting cultural norms

Contextual adaptation makes agents more natural, effective, and user-friendly
by tailoring their behavior to the specific situation and user needs.
    """)


if __name__ == "__main__":
    demonstrate_contextual_adaptation()
