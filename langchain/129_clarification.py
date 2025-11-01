"""
Pattern 129: Clarification & Disambiguation

Description:
    Detects ambiguous or unclear user inputs and asks targeted clarifying questions
    to resolve uncertainties before proceeding.

Components:
    - Ambiguity detection
    - Clarification question generation
    - Context tracking
    - Progressive refinement

Use Cases:
    - Conversational agents handling ambiguous queries
    - Complex instruction following
    - Reducing errors from misunderstanding

LangChain Implementation:
    Uses LangChain to detect ambiguity and generate appropriate clarification questions.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()


class AmbiguityAnalysis(BaseModel):
    """Analysis of ambiguity in user input"""
    is_ambiguous: bool = Field(description="Whether input is ambiguous")
    ambiguity_type: str = Field(description="Type of ambiguity: lexical, syntactic, semantic, referential")
    confidence: float = Field(description="Confidence in ambiguity detection")
    ambiguous_elements: List[str] = Field(description="Specific ambiguous elements")
    clarification_needed: str = Field(description="What needs clarification")


class ClarificationQuestion(BaseModel):
    """A clarification question"""
    question: str = Field(description="The clarification question")
    options: Optional[List[str]] = Field(description="Possible options if applicable")
    priority: int = Field(description="Priority (1=high, 3=low)")


class ClarificationAgent:
    """Agent for handling ambiguity and generating clarifications"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.clarification_history: List[Dict[str, Any]] = []
        
        # Ambiguity detection prompt
        self.ambiguity_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the user's input for ambiguity or unclear elements.

Types of ambiguity:
- Lexical: Multiple meanings of a word (e.g., "bank" - financial or river)
- Syntactic: Sentence structure unclear (e.g., "I saw the man with the telescope")
- Semantic: Vague or imprecise meaning (e.g., "nearby", "soon")
- Referential: Unclear what something refers to (e.g., "it", "that thing")

Return JSON:
{{
  "is_ambiguous": true/false,
  "ambiguity_type": "type",
  "confidence": 0.9,
  "ambiguous_elements": ["element1", "element2"],
  "clarification_needed": "what specifically needs clarification"
}}"""),
            ("user", "Input: {input}\n\nPrevious context: {context}")
        ])
        
        # Clarification generation prompt
        self.clarification_prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate a helpful clarification question to resolve the ambiguity.

Make the question:
1. Specific and targeted
2. Easy to answer
3. Include options when helpful
4. Natural and conversational

Return JSON:
{{
  "question": "the clarification question",
  "options": ["option1", "option2"] or null,
  "priority": 1-3
}}"""),
            ("user", """Input: {input}
Ambiguity: {ambiguity}
Elements needing clarification: {elements}

Generate a clarification question.""")
        ])
        
        self.json_parser = JsonOutputParser()
    
    def detect_ambiguity(self, user_input: str, context: str = "") -> AmbiguityAnalysis:
        """Detect if input is ambiguous"""
        try:
            chain = self.ambiguity_prompt | self.llm | self.json_parser
            result = chain.invoke({"input": user_input, "context": context})
            
            return AmbiguityAnalysis(**result)
        except Exception as e:
            print(f"Ambiguity detection error: {e}")
            return AmbiguityAnalysis(
                is_ambiguous=False,
                ambiguity_type="unknown",
                confidence=0.0,
                ambiguous_elements=[],
                clarification_needed=""
            )
    
    def generate_clarification(
        self,
        user_input: str,
        analysis: AmbiguityAnalysis
    ) -> ClarificationQuestion:
        """Generate clarification question"""
        try:
            chain = self.clarification_prompt | self.llm | self.json_parser
            result = chain.invoke({
                "input": user_input,
                "ambiguity": analysis.ambiguity_type,
                "elements": ", ".join(analysis.ambiguous_elements)
            })
            
            return ClarificationQuestion(**result)
        except Exception as e:
            print(f"Clarification generation error: {e}")
            return ClarificationQuestion(
                question="Could you please provide more details?",
                options=None,
                priority=2
            )
    
    def process_input(self, user_input: str, context: str = "") -> Dict[str, Any]:
        """Process input and return clarification if needed"""
        print(f"\n{'='*70}")
        print(f"📝 Input: {user_input}")
        
        # Detect ambiguity
        analysis = self.detect_ambiguity(user_input, context)
        
        print(f"\n🔍 Analysis:")
        print(f"   Ambiguous: {analysis.is_ambiguous}")
        print(f"   Confidence: {analysis.confidence:.2f}")
        
        result = {
            "input": user_input,
            "is_ambiguous": analysis.is_ambiguous,
            "analysis": analysis,
            "clarification": None
        }
        
        if analysis.is_ambiguous and analysis.confidence > 0.6:
            print(f"   Type: {analysis.ambiguity_type}")
            print(f"   Ambiguous elements: {', '.join(analysis.ambiguous_elements)}")
            
            # Generate clarification
            clarification = self.generate_clarification(user_input, analysis)
            result["clarification"] = clarification
            
            print(f"\n❓ Clarification needed:")
            print(f"   {clarification.question}")
            if clarification.options:
                print(f"   Options:")
                for i, opt in enumerate(clarification.options, 1):
                    print(f"     {i}. {opt}")
            
            # Store in history
            self.clarification_history.append({
                "input": user_input,
                "analysis": analysis.dict(),
                "clarification": clarification.dict()
            })
        else:
            print(f"\n✅ Input is clear, no clarification needed")
        
        return result
    
    def resolve_with_clarification(
        self,
        original_input: str,
        clarification_response: str
    ) -> str:
        """Process clarified input"""
        resolution_prompt = ChatPromptTemplate.from_messages([
            ("system", "Combine the original input with the clarification to form a complete, unambiguous statement."),
            ("user", """Original: {original}
Clarification response: {clarification}

Generate complete, unambiguous statement:""")
        ])
        
        chain = resolution_prompt | self.llm | StrOutputParser()
        resolved = chain.invoke({
            "original": original_input,
            "clarification": clarification_response
        })
        
        print(f"\n✅ Resolved statement:")
        print(f"   {resolved}")
        
        return resolved


def demonstrate_clarification():
    """Demonstrate clarification and disambiguation"""
    print("=" * 80)
    print("Pattern 129: Clarification & Disambiguation")
    print("=" * 80)
    
    agent = ClarificationAgent()
    
    # Example 1: Lexical ambiguity
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Lexical Ambiguity")
    print("=" * 80)
    
    ambiguous_inputs = [
        "I need to go to the bank",
        "Can you get me the bat?",
        "Show me the rock music"
    ]
    
    for input_text in ambiguous_inputs:
        result = agent.process_input(input_text)
        
        if result["clarification"]:
            # Simulate user providing clarification
            print(f"\n   💬 User provides clarification: Financial bank")
            resolved = agent.resolve_with_clarification(
                input_text,
                "The financial institution"
            )
    
    # Example 2: Referential ambiguity
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Referential Ambiguity")
    print("=" * 80)
    
    referential_inputs = [
        ("John told Mike he should leave", "Previous: John and Mike were arguing"),
        ("Put it on the table", "Previous: User has a box and a book"),
        ("I want the blue one", "Previous: Looking at shirts")
    ]
    
    for input_text, context in referential_inputs:
        agent.process_input(input_text, context)
    
    # Example 3: Semantic vagueness
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Semantic Vagueness")
    print("=" * 80)
    
    vague_inputs = [
        "Schedule a meeting soon",
        "I need a lot of storage",
        "Find a nearby restaurant"
    ]
    
    for input_text in vague_inputs:
        agent.process_input(input_text)
    
    # Example 4: Complex ambiguity
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Complex Ambiguous Request")
    print("=" * 80)
    
    complex_input = "Book me a nice hotel room with a good view for my trip"
    result = agent.process_input(complex_input)
    
    if result["clarification"]:
        print(f"\n   💬 User response: Paris, June 15-20, budget up to $200/night")
        resolved = agent.resolve_with_clarification(
            complex_input,
            "Paris, France from June 15-20, 2024, budget $200 per night"
        )
    
    # Example 5: Clear input (no clarification needed)
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Clear Input (No Clarification)")
    print("=" * 80)
    
    clear_inputs = [
        "Set an alarm for 7:00 AM tomorrow",
        "What is the capital of France?",
        "Calculate 15% tip on $85.50"
    ]
    
    for input_text in clear_inputs:
        agent.process_input(input_text)
    
    # Statistics
    print("\n" + "=" * 80)
    print("CLARIFICATION STATISTICS")
    print("=" * 80)
    
    print(f"\nTotal clarifications asked: {len(agent.clarification_history)}")
    
    if agent.clarification_history:
        print(f"\nAmbiguity types encountered:")
        types = {}
        for item in agent.clarification_history:
            amb_type = item['analysis']['ambiguity_type']
            types[amb_type] = types.get(amb_type, 0) + 1
        
        for amb_type, count in types.items():
            print(f"  - {amb_type}: {count}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Clarification & Disambiguation Pattern:
- Detects ambiguous or unclear user inputs
- Generates targeted clarification questions
- Resolves ambiguity through progressive refinement
- Improves accuracy and user satisfaction

Types of Ambiguity Handled:
✓ Lexical - Multiple word meanings
✓ Syntactic - Unclear sentence structure
✓ Semantic - Vague or imprecise terms
✓ Referential - Unclear references

Key Benefits:
✓ Reduces errors from misunderstanding
✓ Improves task completion accuracy
✓ Better user experience
✓ Handles complex, real-world inputs
✓ Proactive problem prevention

Best Practices:
• Ask specific, targeted questions
• Provide options when helpful
• Keep questions natural and conversational
• Prioritize most important clarifications
• Track clarification history for learning
    """)


if __name__ == "__main__":
    demonstrate_clarification()
