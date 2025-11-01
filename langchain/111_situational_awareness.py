"""
Pattern 111: Situational Awareness

Description:
    Situational Awareness enables agents to maintain comprehensive understanding
    of their current context, environment, and state. The pattern continuously
    monitors and updates a model of the situation, including relevant entities,
    relationships, constraints, opportunities, and threats. This awareness informs
    decision-making and enables context-appropriate behavior.
    
    The pattern is crucial for adaptive agents operating in dynamic environments
    where context changes frequently. It combines perception, state estimation,
    context modeling, and predictive awareness to maintain an accurate and
    up-to-date understanding of the situation.
    
    Situational awareness includes environmental monitoring, state tracking,
    context interpretation, threat detection, opportunity identification, and
    predictive situation assessment.

Key Components:
    1. Perception Module: Gathers situational data
    2. State Estimator: Tracks current state
    3. Context Modeler: Interprets situation meaning
    4. Change Detector: Identifies state changes
    5. Threat Assessor: Evaluates risks
    6. Opportunity Detector: Identifies opportunities
    7. Predictor: Anticipates future states

Awareness Dimensions:
    - Spatial: Where things are
    - Temporal: When things happen
    - Causal: Why things occur
    - Social: Who is involved
    - Functional: What capabilities exist
    - Intentional: What goals are active

Use Cases:
    - Autonomous vehicles understanding traffic
    - Security monitoring systems
    - Personal assistants with context awareness
    - Game AI with environmental awareness
    - Crisis management systems

LangChain Implementation:
    Uses memory systems, context tracking, state estimation chains, and
    predictive models to maintain situational awareness.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.memory import ConversationBufferMemory

load_dotenv()


class SituationStatus(Enum):
    """Enumeration of situation statuses."""
    NORMAL = "normal"
    ALERT = "alert"
    WARNING = "warning"
    CRITICAL = "critical"
    OPPORTUNITY = "opportunity"


class SituationalAwarenessAgent:
    """
    Agent that maintains comprehensive awareness of current situation.
    """
    
    def __init__(self, agent_role: str = "general_agent"):
        """Initialize with role and context."""
        self.llm = ChatOpenAI(temperature=0.3, model="gpt-4")
        self.role = agent_role
        self.situation_memory = ConversationBufferMemory(
            memory_key="situation_history",
            return_messages=True
        )
        
        # Situational state
        self.current_context = {}
        self.active_entities = {}
        self.threats = []
        self.opportunities = []
        self.status = SituationStatus.NORMAL
    
    def perceive_environment(
        self,
        observations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process environmental observations to update awareness.
        
        Args:
            observations: Dictionary of observed facts
            
        Returns:
            Updated situational model
        """
        print(f"\n{'='*60}")
        print("PERCEPTION & STATE ESTIMATION")
        print(f"{'='*60}")
        print(f"Processing {len(observations)} observations...")
        
        # Update context with observations
        self.current_context.update(observations)
        self.current_context['last_update'] = datetime.now().isoformat()
        
        # Analyze observations for entities
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a situational awareness system analyzing observations.
Extract key entities, their states, and relationships from the observations.

Provide a JSON response with:
- entities: List of identified entities with properties
- relationships: List of relationships between entities
- key_facts: Important facts about the situation"""),
            ("user", "Observations: {observations}")
        ])
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            analysis = chain.invoke({"observations": str(observations)})
            
            # Update entities
            if 'entities' in analysis:
                for entity in analysis['entities']:
                    entity_id = entity.get('id', entity.get('name', 'unknown'))
                    self.active_entities[entity_id] = entity
            
            print(f"Identified {len(self.active_entities)} entities")
            print(f"Key facts: {analysis.get('key_facts', [])}")
            
            return {
                "perception": analysis,
                "context_updated": True,
                "timestamp": self.current_context['last_update']
            }
        except:
            # Fallback if JSON parsing fails
            return {
                "perception": observations,
                "context_updated": True,
                "timestamp": self.current_context['last_update']
            }
    
    def assess_situation(self) -> Dict[str, Any]:
        """
        Assess current situation for threats and opportunities.
        
        Returns:
            Situation assessment with status and recommendations
        """
        print(f"\n{'='*60}")
        print("SITUATION ASSESSMENT")
        print(f"{'='*60}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a situational assessment expert. Analyze the current
situation and provide:
1. Overall situation status (normal/alert/warning/critical/opportunity)
2. Identified threats or risks
3. Identified opportunities
4. Recommended actions
5. Confidence in assessment (0-1)

Format as: STATUS | Threats: [...] | Opportunities: [...] | Actions: [...] | Confidence: X.X"""),
            ("user", """Current Context: {context}
Active Entities: {entities}
Situation History: {history}

Assess this situation comprehensively.""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        assessment_text = chain.invoke({
            "context": str(self.current_context),
            "entities": str(self.active_entities),
            "history": self.situation_memory.load_memory_variables({}).get('situation_history', [])
        })
        
        # Parse assessment
        parts = assessment_text.split('|')
        status_text = parts[0].strip().lower()
        
        # Update status
        if 'critical' in status_text:
            self.status = SituationStatus.CRITICAL
        elif 'warning' in status_text:
            self.status = SituationStatus.WARNING
        elif 'alert' in status_text:
            self.status = SituationStatus.ALERT
        elif 'opportunity' in status_text:
            self.status = SituationStatus.OPPORTUNITY
        else:
            self.status = SituationStatus.NORMAL
        
        result = {
            "status": self.status.value,
            "assessment": assessment_text,
            "context_size": len(self.current_context),
            "entities_tracked": len(self.active_entities),
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"Status: {self.status.value.upper()}")
        print(f"Assessment:\n{assessment_text}")
        
        # Store in memory
        self.situation_memory.save_context(
            {"input": "situation_assessment"},
            {"output": assessment_text}
        )
        
        return result
    
    def predict_changes(
        self,
        time_horizon: str = "near future"
    ) -> Dict[str, Any]:
        """
        Predict likely changes to the situation.
        
        Args:
            time_horizon: Time frame for prediction
            
        Returns:
            Predictions about situation evolution
        """
        print(f"\n{'='*60}")
        print("PREDICTIVE AWARENESS")
        print(f"{'='*60}")
        print(f"Predicting changes for: {time_horizon}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a predictive analyst. Based on current situation,
predict likely changes and developments.

Provide:
1. Most likely changes
2. Probability estimates
3. Potential triggers
4. Recommended preparations"""),
            ("user", """Current Situation:
Context: {context}
Entities: {entities}
Current Status: {status}

Predict changes for: {time_horizon}""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        predictions = chain.invoke({
            "context": str(self.current_context),
            "entities": str(self.active_entities),
            "status": self.status.value,
            "time_horizon": time_horizon
        })
        
        result = {
            "time_horizon": time_horizon,
            "predictions": predictions,
            "based_on_entities": len(self.active_entities),
            "confidence": "medium"  # Could be calculated
        }
        
        print(f"Predictions:\n{predictions}")
        
        return result
    
    def context_appropriate_response(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        Generate response appropriate to current situational context.
        
        Args:
            query: Query or request
            
        Returns:
            Context-aware response
        """
        print(f"\n{'='*60}")
        print("CONTEXT-AWARE RESPONSE")
        print(f"{'='*60}")
        print(f"Query: {query}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an agent with full situational awareness. 
Respond to queries considering the complete situational context.

Your response should:
1. Acknowledge relevant context
2. Address the query appropriately
3. Consider current status and constraints
4. Provide context-aware recommendations"""),
            ("user", """Query: {query}

Current Situation:
- Status: {status}
- Context: {context}
- Active Entities: {entities}

Provide a situationally appropriate response.""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "query": query,
            "status": self.status.value,
            "context": str(self.current_context)[:500],  # Limit size
            "entities": str(list(self.active_entities.keys()))
        })
        
        result = {
            "query": query,
            "response": response,
            "context_considered": True,
            "status_at_response": self.status.value
        }
        
        print(f"Response:\n{response}")
        
        return result
    
    def get_situation_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of current situation."""
        return {
            "status": self.status.value,
            "context": self.current_context,
            "entities": self.active_entities,
            "role": self.role,
            "last_update": self.current_context.get('last_update', 'never')
        }


def demonstrate_situational_awareness():
    """Demonstrate situational awareness capabilities."""
    print("=" * 60)
    print("SITUATIONAL AWARENESS AGENT DEMONSTRATION")
    print("=" * 60)
    
    agent = SituationalAwarenessAgent(agent_role="security_monitor")
    
    # Example 1: Initial Perception
    print("\n" + "=" * 60)
    print("Example 1: Environmental Perception")
    print("=" * 60)
    
    observations1 = {
        "time": "14:30",
        "location": "Building A, Floor 3",
        "people_count": 15,
        "noise_level": "normal",
        "access_points": ["door_301", "door_302"],
        "cameras_active": True
    }
    
    result1 = agent.perceive_environment(observations1)
    
    # Example 2: Situation Assessment
    print("\n" + "=" * 60)
    print("Example 2: Situation Assessment")
    print("=" * 60)
    
    result2 = agent.assess_situation()
    
    # Example 3: Changed Situation
    print("\n" + "=" * 60)
    print("Example 3: Detecting Changes")
    print("=" * 60)
    
    observations2 = {
        "time": "14:35",
        "people_count": 25,  # Increased
        "noise_level": "elevated",
        "unauthorized_access_attempt": "door_304",
        "alarm_triggered": True
    }
    
    result3 = agent.perceive_environment(observations2)
    result4 = agent.assess_situation()
    
    # Example 4: Predictive Awareness
    print("\n" + "=" * 60)
    print("Example 4: Predicting Changes")
    print("=" * 60)
    
    result5 = agent.predict_changes("next 15 minutes")
    
    # Example 5: Context-Aware Response
    print("\n" + "=" * 60)
    print("Example 5: Context-Appropriate Response")
    print("=" * 60)
    
    result6 = agent.context_appropriate_response(
        "Should I allow the scheduled maintenance crew to enter?"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SITUATION SUMMARY")
    print("=" * 60)
    summary = agent.get_situation_summary()
    print(f"Current Status: {summary['status'].upper()}")
    print(f"Entities Tracked: {len(summary['entities'])}")
    print(f"Context Elements: {len(summary['context'])}")
    print(f"Agent Role: {summary['role']}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("✓ Processed environmental observations")
    print("✓ Maintained situational context")
    print("✓ Detected and assessed changes")
    print("✓ Predicted future developments")
    print("✓ Generated context-aware responses")


if __name__ == "__main__":
    demonstrate_situational_awareness()
