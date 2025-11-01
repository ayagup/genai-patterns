"""
Pattern 128: Multi-Turn Dialogue Management

Description:
    Manages coherent multi-turn conversations with context tracking, dialogue state,
    and appropriate responses across multiple exchanges.

Components:
    - Dialogue state tracking
    - Context management
    - Intent recognition
    - Response generation

Use Cases:
    - Conversational AI assistants
    - Customer service chatbots
    - Interactive tutoring systems

LangChain Implementation:
    Uses LangChain memory, state management, and conversation chains.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()


class DialogueState(str, Enum):
    """Dialogue states"""
    GREETING = "greeting"
    INFO_GATHERING = "info_gathering"
    CLARIFICATION = "clarification"
    PROCESSING = "processing"
    RESPONSE = "response"
    CLOSING = "closing"


class UserIntent(BaseModel):
    """Detected user intent"""
    intent: str = Field(description="Primary intent")
    confidence: float = Field(description="Confidence score")
    entities: Dict[str, Any] = Field(description="Extracted entities")


class DialogueContext(BaseModel):
    """Dialogue context"""
    state: str = Field(description="Current dialogue state")
    topic: Optional[str] = Field(description="Current topic")
    user_goal: Optional[str] = Field(description="User's goal")
    collected_info: Dict[str, Any] = Field(default_factory=dict)


class MultiTurnDialogueAgent:
    """Agent for managing multi-turn conversations"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.conversation_history: List[Dict[str, str]] = []
        self.context: DialogueContext = DialogueContext(
            state=DialogueState.GREETING.value,
            topic=None,
            user_goal=None,
            collected_info={}
        )
        
        # Intent detection prompt
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the user's message and detect their intent.
Common intents: greeting, question, request_info, provide_info, complaint, thanks, goodbye

Return JSON:
{{
  "intent": "intent_name",
  "confidence": 0.9,
  "entities": {{"key": "value"}}
}}"""),
            ("user", "{message}")
        ])
        
        # Response generation with context
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful conversational AI assistant.
Current conversation context:
- State: {state}
- Topic: {topic}
- User Goal: {user_goal}
- Collected Info: {collected_info}

Provide a natural, contextually appropriate response that:
1. Acknowledges the user's message
2. Maintains conversation flow
3. Helps achieve the user's goal
4. Asks follow-up questions if needed"""),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{message}")
        ])
        
        self.intent_parser = JsonOutputParser()
        self.response_parser = StrOutputParser()
    
    def detect_intent(self, message: str) -> UserIntent:
        """Detect user intent from message"""
        try:
            chain = self.intent_prompt | self.llm | self.intent_parser
            result = chain.invoke({"message": message})
            
            return UserIntent(
                intent=result["intent"],
                confidence=result["confidence"],
                entities=result.get("entities", {})
            )
        except Exception as e:
            print(f"Intent detection error: {e}")
            return UserIntent(intent="unknown", confidence=0.0, entities={})
    
    def update_context(self, intent: UserIntent, message: str):
        """Update dialogue context based on intent"""
        # Update state based on intent
        if intent.intent == "greeting":
            self.context.state = DialogueState.GREETING.value
        elif intent.intent in ["question", "request_info"]:
            self.context.state = DialogueState.INFO_GATHERING.value
        elif intent.intent == "provide_info":
            self.context.state = DialogueState.PROCESSING.value
            # Store provided information
            self.context.collected_info.update(intent.entities)
        elif intent.intent == "goodbye":
            self.context.state = DialogueState.CLOSING.value
        
        # Extract topic if not set
        if not self.context.topic and len(self.conversation_history) > 0:
            # Simple topic extraction (could be enhanced)
            for entity_key in intent.entities:
                if entity_key not in ["time", "date", "number"]:
                    self.context.topic = intent.entities[entity_key]
                    break
    
    def generate_response(self, message: str) -> str:
        """Generate contextually appropriate response"""
        # Detect intent
        intent = self.detect_intent(message)
        print(f"\n🎯 Detected Intent: {intent.intent} (confidence: {intent.confidence:.2f})")
        
        # Update context
        self.update_context(intent, message)
        print(f"📊 Dialogue State: {self.context.state}")
        if self.context.topic:
            print(f"💡 Topic: {self.context.topic}")
        
        # Prepare conversation history for prompt
        history = []
        for turn in self.conversation_history[-5:]:  # Last 5 turns
            if turn["role"] == "user":
                history.append(HumanMessage(content=turn["content"]))
            else:
                history.append(AIMessage(content=turn["content"]))
        
        # Generate response
        chain = self.response_prompt | self.llm | self.response_parser
        
        response = chain.invoke({
            "message": message,
            "state": self.context.state,
            "topic": self.context.topic or "not set",
            "user_goal": self.context.user_goal or "not determined",
            "collected_info": str(self.context.collected_info),
            "history": history
        })
        
        return response
    
    def chat(self, user_message: str) -> str:
        """Process user message and return response"""
        print(f"\n{'='*60}")
        print(f"👤 User: {user_message}")
        
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate response
        response = self.generate_response(user_message)
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"🤖 Assistant: {response}")
        
        return response
    
    def get_conversation_summary(self) -> str:
        """Generate summary of conversation"""
        if not self.conversation_history:
            return "No conversation yet."
        
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following conversation in 2-3 sentences."),
            ("user", "{conversation}")
        ])
        
        conversation_text = "\n".join([
            f"{turn['role']}: {turn['content']}"
            for turn in self.conversation_history
        ])
        
        chain = summary_prompt | self.llm | StrOutputParser()
        summary = chain.invoke({"conversation": conversation_text})
        
        return summary
    
    def reset_conversation(self):
        """Reset conversation state"""
        self.conversation_history = []
        self.context = DialogueContext(
            state=DialogueState.GREETING.value,
            topic=None,
            user_goal=None,
            collected_info={}
        )
        print("\n🔄 Conversation reset")


def demonstrate_multi_turn_dialogue():
    """Demonstrate multi-turn dialogue management"""
    print("=" * 80)
    print("Pattern 128: Multi-Turn Dialogue Management")
    print("=" * 80)
    
    agent = MultiTurnDialogueAgent()
    
    # Example 1: Customer service conversation
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Customer Service Conversation")
    print("=" * 80)
    
    conversation_1 = [
        "Hi, I need help with my order",
        "My order number is 12345",
        "It was supposed to arrive yesterday but I haven't received it",
        "Yes, my address is 123 Main Street",
        "Thank you for your help!"
    ]
    
    for message in conversation_1:
        agent.chat(message)
    
    print("\n" + "-" * 60)
    print("📝 Conversation Summary:")
    print(agent.get_conversation_summary())
    
    # Example 2: Technical support
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Technical Support Conversation")
    print("=" * 80)
    
    agent.reset_conversation()
    
    conversation_2 = [
        "Hello, my computer won't start",
        "It's a laptop, about 2 years old",
        "When I press the power button, nothing happens at all",
        "No, there are no lights or sounds",
        "Okay, I'll try that. Thanks!"
    ]
    
    for message in conversation_2:
        agent.chat(message)
    
    print("\n" + "-" * 60)
    print("📝 Conversation Summary:")
    print(agent.get_conversation_summary())
    
    # Example 3: Information gathering
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Information Gathering Conversation")
    print("=" * 80)
    
    agent.reset_conversation()
    
    conversation_3 = [
        "I want to book a flight",
        "From New York to London",
        "Next week, preferably Tuesday",
        "Economy class is fine",
        "Great, please proceed with the booking"
    ]
    
    for message in conversation_3:
        agent.chat(message)
    
    print("\n" + "-" * 60)
    print("📝 Conversation Summary:")
    print(agent.get_conversation_summary())
    
    print("\n" + "-" * 60)
    print("📊 Final Context State:")
    print(f"  State: {agent.context.state}")
    print(f"  Topic: {agent.context.topic}")
    print(f"  Collected Info: {agent.context.collected_info}")
    print(f"  Total Turns: {len(agent.conversation_history)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Multi-Turn Dialogue Management Pattern:
- Maintains context across conversation turns
- Tracks dialogue state and user intent
- Generates contextually appropriate responses
- Manages information collection

Key Features:
✓ Intent detection
✓ State tracking
✓ Context maintenance
✓ Natural conversation flow
✓ Information extraction

Dialogue States:
• Greeting - Initial interaction
• Info Gathering - Collecting information
• Clarification - Resolving ambiguities
• Processing - Handling requests
• Response - Providing answers
• Closing - Ending conversation

Benefits:
✓ Coherent multi-turn conversations
✓ Context-aware responses
✓ Better user experience
✓ Goal-oriented dialogue
✓ Efficient information gathering
    """)


if __name__ == "__main__":
    demonstrate_multi_turn_dialogue()
