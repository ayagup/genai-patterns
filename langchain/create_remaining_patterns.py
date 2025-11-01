"""
Script to generate remaining pattern implementations efficiently
"""

patterns_data = [
    # Dialogue & Interaction (130-132)
    {
        "num": 130,
        "name": "proactive_engagement",
        "title": "Proactive Engagement",
        "description": "Agent initiates interactions when appropriate, detecting opportunities and user needs",
        "components": ["Opportunity detection", "Timing optimization", "Proactive suggestions", "User need prediction"],
        "use_cases": ["Virtual assistants", "Monitoring systems", "Proactive customer service"]
    },
    {
        "num": 131,
        "name": "persona_consistency",
        "title": "Persona Consistency",
        "description": "Maintains consistent personality, tone, and style across all interactions",
        "components": ["Persona definition", "Style enforcement", "Tone consistency", "Character maintenance"],
        "use_cases": ["Brand consistency", "Character agents", "Conversational AI"]
    },
    {
        "num": 132,
        "name": "emotion_recognition",
        "title": "Emotion Recognition & Response",
        "description": "Detects and responds appropriately to user emotions",
        "components": ["Sentiment analysis", "Emotion detection", "Empathetic responses", "Affect adaptation"],
        "use_cases": ["Customer service", "Mental health support", "Education"]
    },
    
    # Specialization (133-136)
    {
        "num": 133,
        "name": "domain_expert",
        "title": "Domain Expert Agent",
        "description": "Deep specialization in a specific domain with expert-level knowledge",
        "components": ["Domain knowledge", "Expert reasoning", "Specialized vocabulary", "Domain-specific tools"],
        "use_cases": ["Medical diagnosis", "Legal advice", "Financial analysis"]
    },
    {
        "num": 134,
        "name": "task_specific",
        "title": "Task-Specific Agent",
        "description": "Optimized for a single specific task or function",
        "components": ["Task optimization", "Specialized processing", "Performance tuning", "Focused capabilities"],
        "use_cases": ["Summarization", "Translation", "Classification"]
    },
    {
        "num": 135,
        "name": "polyglot_agent",
        "title": "Polyglot Agent",
        "description": "Operates across multiple languages with translation and cross-lingual understanding",
        "components": ["Multi-language support", "Translation", "Cross-lingual reasoning", "Cultural adaptation"],
        "use_cases": ["International customer service", "Multilingual content", "Global applications"]
    },
    {
        "num": 136,
        "name": "accessibility_focused",
        "title": "Accessibility-Focused Agent",
        "description": "Designed for users with disabilities, ensuring inclusive access",
        "components": ["Screen reader support", "Voice control", "Simplified interfaces", "Alternative formats"],
        "use_cases": ["Inclusive applications", "Accessibility compliance", "Assistive technology"]
    },
]

def generate_pattern_file(pattern_data):
    """Generate pattern implementation file"""
    
    template = f'''"""
Pattern {pattern_data['num']}: {pattern_data['title']}

Description:
    {pattern_data['description']}

Components:
    - {chr(10)+'    - '.join(pattern_data['components'])}

Use Cases:
    - {chr(10)+'    - '.join(pattern_data['use_cases'])}

LangChain Implementation:
    Demonstrates {pattern_data['title'].lower()} using LangChain.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class {pattern_data['title'].replace(' ', '').replace('&', 'And').replace('-', '')}Agent:
    """Agent implementing {pattern_data['title'].lower()}"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        
        # Main processing prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at {pattern_data['title'].lower()}.
            
{pattern_data['description']}

Key capabilities:
{chr(10).join(['- ' + comp for comp in pattern_data['components']])}"""),
            ("user", "{{input}}")
        ])
        
        self.parser = StrOutputParser()
    
    def process(self, input_text: str) -> str:
        """Process input and return result"""
        chain = self.prompt | self.llm | self.parser
        result = chain.invoke({{"input": input_text}})
        return result
    
    def demonstrate(self):
        """Demonstrate the pattern"""
        print(f"\\n{{'='*60}}")
        print(f"Demonstrating {pattern_data['title']}")
        print(f"{{'='*60}}")
        
        # Example demonstration
        test_input = "Example input for {pattern_data['title'].lower()}"
        result = self.process(test_input)
        
        print(f"\\nInput: {{test_input}}")
        print(f"Output: {{result}}")


def demonstrate_{pattern_data['name']}():
    """Demonstrate {pattern_data['title']}"""
    print("=" * 80)
    print("Pattern {pattern_data['num']}: {pattern_data['title']}")
    print("=" * 80)
    
    agent = {pattern_data['title'].replace(' ', '').replace('&', 'And').replace('-', '')}Agent()
    
    # Example 1
    print("\\n" + "=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)
    
    agent.demonstrate()
    
    # Summary
    print("\\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
{pattern_data['title']} Pattern:
{pattern_data['description']}

Key Components:
{chr(10).join(['✓ ' + comp for comp in pattern_data['components']])}

Use Cases:
{chr(10).join(['• ' + uc for uc in pattern_data['use_cases']])}

Benefits:
✓ Specialized capability
✓ Optimized performance
✓ Domain expertise
✓ Better user experience
    """)


if __name__ == "__main__":
    demonstrate_{pattern_data['name']}()
'''
    
    return template


# Generate files
def main():
    import os
    
    base_path = os.path.dirname(__file__)
    created_count = 0
    
    for pattern in patterns_data:
        filename = f"{pattern['num']:03d}_{pattern['name']}.py"
        filepath = os.path.join(base_path, filename)
        
        if not os.path.exists(filepath):
            content = generate_pattern_file(pattern)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ Created: {filename}")
            created_count += 1
        else:
            print(f"⊘ Skipped (exists): {filename}")
    
    print(f"\n{'='*60}")
    print(f"Created {created_count} new pattern files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
