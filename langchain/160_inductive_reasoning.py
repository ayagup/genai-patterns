"""
Pattern 160: Inductive Reasoning

Description:
    Inductive reasoning generalizes from specific observations to broader patterns
    or rules. It moves from particular instances to general conclusions, inferring
    patterns that likely hold across similar cases.

Components:
    - Example collection
    - Pattern identification
    - Rule generalization
    - Confidence assessment
    - Counter-example checking

Use Cases:
    - Pattern discovery
    - Rule learning
    - Trend analysis
    - Scientific generalization
    - Predictive modeling

Benefits:
    - Discovers patterns from data
    - Enables prediction
    - Learns from examples
    - Practical applicability

Trade-offs:
    - Conclusions not guaranteed
    - Vulnerable to biased samples
    - Requires sufficient examples
    - May over-generalize

LangChain Implementation:
    Uses ChatOpenAI for pattern recognition, rule extraction,
    and validation through additional examples
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json

load_dotenv()


@dataclass
class Example:
    """Single example/observation"""
    input: str
    output: str
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pattern:
    """Discovered pattern"""
    description: str
    rule: str
    confidence: float  # 0-1
    supporting_examples: List[str] = field(default_factory=list)
    counter_examples: List[str] = field(default_factory=list)
    generality: float = 0.5  # How general vs specific


@dataclass
class InductiveAnalysis:
    """Result of inductive reasoning"""
    examples: List[Example]
    patterns: List[Pattern]
    generalizations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_trace: List[str] = field(default_factory=list)


class InductiveReasoningAgent:
    """Agent that performs inductive reasoning"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize inductive reasoning agent
        
        Args:
            model_name: LLM model to use
        """
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.validator_llm = ChatOpenAI(model=model_name, temperature=0)
    
    def reason(self, examples: List[Example],
               domain: Optional[str] = None,
               min_confidence: float = 0.5) -> InductiveAnalysis:
        """
        Perform inductive reasoning on examples
        
        Args:
            examples: List of examples to learn from
            domain: Optional domain context
            min_confidence: Minimum confidence threshold
            
        Returns:
            InductiveAnalysis with discovered patterns
        """
        reasoning_trace = []
        
        # Step 1: Identify patterns
        reasoning_trace.append(f"Analyzing {len(examples)} examples...")
        patterns = self._identify_patterns(examples, domain)
        reasoning_trace.append(f"Identified {len(patterns)} potential patterns")
        
        # Step 2: Validate patterns
        reasoning_trace.append("Validating patterns against examples...")
        for pattern in patterns:
            self._validate_pattern(pattern, examples)
        
        # Step 3: Filter by confidence
        patterns = [p for p in patterns if p.confidence >= min_confidence]
        reasoning_trace.append(f"Retained {len(patterns)} high-confidence patterns")
        
        # Step 4: Generate generalizations
        generalizations = self._generate_generalizations(patterns, examples)
        reasoning_trace.append(f"Generated {len(generalizations)} generalizations")
        
        # Step 5: Calculate overall confidence
        if patterns:
            avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
        else:
            avg_confidence = 0.0
        
        return InductiveAnalysis(
            examples=examples,
            patterns=patterns,
            generalizations=generalizations,
            confidence=avg_confidence,
            reasoning_trace=reasoning_trace
        )
    
    def _identify_patterns(self, examples: List[Example],
                          domain: Optional[str]) -> List[Pattern]:
        """Identify patterns from examples"""
        # Format examples
        examples_text = "\n".join([
            f"Input: {ex.input} → Output: {ex.output}"
            for ex in examples
        ])
        
        # Create pattern identification prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at identifying patterns from examples.
Analyze the given input-output examples and identify patterns or rules.

{domain_context}

For each pattern, provide:
1. Clear description
2. General rule
3. How general vs specific it is (0-1)

Return as JSON array: [{{"description": "...", "rule": "...", "generality": 0.0-1.0}}]"""),
            ("user", "Examples:\n{examples}\n\nIdentify patterns:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        domain_context = f"Domain: {domain}" if domain else "Consider general patterns."
        
        response = chain.invoke({
            "examples": examples_text,
            "domain_context": domain_context
        })
        
        # Parse patterns
        patterns = self._parse_patterns(response)
        return patterns
    
    def _parse_patterns(self, response: str) -> List[Pattern]:
        """Parse patterns from LLM response"""
        try:
            # Try to extract JSON
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                patterns = []
                for item in data:
                    pattern = Pattern(
                        description=item.get('description', ''),
                        rule=item.get('rule', ''),
                        confidence=0.5,  # Initial, will validate
                        generality=float(item.get('generality', 0.5))
                    )
                    patterns.append(pattern)
                return patterns
        except Exception as e:
            print(f"Error parsing patterns: {e}")
        
        # Fallback parsing
        patterns = []
        lines = response.split('\n')
        current_pattern = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                if current_pattern.get('description'):
                    patterns.append(Pattern(
                        description=current_pattern.get('description', ''),
                        rule=current_pattern.get('rule', ''),
                        confidence=0.5,
                        generality=0.5
                    ))
                current_pattern = {'description': line[2:].strip()}
            elif 'rule:' in line.lower():
                current_pattern['rule'] = line.split(':', 1)[1].strip()
        
        # Add last pattern
        if current_pattern.get('description'):
            patterns.append(Pattern(
                description=current_pattern.get('description', ''),
                rule=current_pattern.get('rule', ''),
                confidence=0.5,
                generality=0.5
            ))
        
        return patterns[:5]  # Limit to 5
    
    def _validate_pattern(self, pattern: Pattern, examples: List[Example]):
        """Validate pattern against examples"""
        supporting = 0
        contradicting = 0
        
        for example in examples:
            # Check if pattern applies to this example
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Determine if the pattern/rule applies to this example. Respond with 'yes' or 'no'."),
                ("user", """Pattern: {pattern}
Rule: {rule}

Example - Input: {input}, Output: {output}

Does the pattern apply?""")
            ])
            
            chain = prompt | self.validator_llm | StrOutputParser()
            
            response = chain.invoke({
                "pattern": pattern.description,
                "rule": pattern.rule,
                "input": example.input,
                "output": example.output
            })
            
            if 'yes' in response.lower():
                supporting += 1
                pattern.supporting_examples.append(f"{example.input} → {example.output}")
            else:
                contradicting += 1
                pattern.counter_examples.append(f"{example.input} → {example.output}")
        
        # Calculate confidence
        total = supporting + contradicting
        if total > 0:
            pattern.confidence = supporting / total
        else:
            pattern.confidence = 0.0
    
    def _generate_generalizations(self, patterns: List[Pattern],
                                 examples: List[Example]) -> List[str]:
        """Generate general rules from patterns"""
        if not patterns:
            return []
        
        # Format patterns
        patterns_text = "\n".join([
            f"- {p.description} (confidence: {p.confidence:.0%})"
            for p in patterns
        ])
        
        # Create generalization prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Synthesize the patterns into broader generalizations or principles."),
            ("user", """Patterns discovered:
{patterns}

Generate 2-3 general rules or principles that capture these patterns:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({"patterns": patterns_text})
        
        # Parse generalizations
        generalizations = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                clean_line = line.lstrip('0123456789.-) ')
                if clean_line:
                    generalizations.append(clean_line)
        
        return generalizations
    
    def predict(self, analysis: InductiveAnalysis, new_input: str) -> Dict[str, Any]:
        """Use learned patterns to make a prediction"""
        # Format patterns
        patterns_text = "\n".join([
            f"- {p.rule} (confidence: {p.confidence:.0%})"
            for p in analysis.patterns
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Use the learned patterns to predict the output for the new input.
Explain which pattern(s) you're applying."""),
            ("user", """Learned patterns:
{patterns}

New input: {input}

Predict the output and explain your reasoning:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "patterns": patterns_text,
            "input": new_input
        })
        
        return {
            "input": new_input,
            "prediction": response,
            "based_on_patterns": len(analysis.patterns),
            "confidence": analysis.confidence
        }


def demonstrate_inductive_reasoning():
    """Demonstrate inductive reasoning pattern"""
    print("=" * 80)
    print("INDUCTIVE REASONING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = InductiveReasoningAgent()
    
    # Example 1: Number patterns
    print("\n" + "="*80)
    print("EXAMPLE 1: Number Pattern Discovery")
    print("="*80)
    examples = [
        Example(input="2", output="4"),
        Example(input="3", output="9"),
        Example(input="4", output="16"),
        Example(input="5", output="25"),
        Example(input="6", output="36"),
    ]
    
    analysis = agent.reason(examples, domain="Mathematics")
    
    print("Examples:")
    for ex in examples:
        print(f"  {ex.input} → {ex.output}")
    
    print(f"\nDiscovered Patterns:")
    for i, pattern in enumerate(analysis.patterns, 1):
        print(f"\n{i}. {pattern.description}")
        print(f"   Rule: {pattern.rule}")
        print(f"   Confidence: {pattern.confidence:.0%}")
        print(f"   Generality: {pattern.generality:.1f}")
    
    # Test prediction
    if analysis.patterns:
        prediction = agent.predict(analysis, "7")
        print(f"\nPrediction for '7': {prediction['prediction'][:200]}...")
    
    # Example 2: Text transformation
    print("\n" + "="*80)
    print("EXAMPLE 2: Text Transformation Pattern")
    print("="*80)
    examples = [
        Example(input="cat", output="cats"),
        Example(input="dog", output="dogs"),
        Example(input="house", output="houses"),
        Example(input="car", output="cars"),
        Example(input="book", output="books"),
    ]
    
    analysis = agent.reason(examples, domain="English grammar")
    
    print("Examples:")
    for ex in examples:
        print(f"  {ex.input} → {ex.output}")
    
    print(f"\nPattern Analysis:")
    for pattern in analysis.patterns:
        print(f"\n- {pattern.description}")
        print(f"  Confidence: {pattern.confidence:.0%}")
        if pattern.supporting_examples:
            print(f"  Supporting: {len(pattern.supporting_examples)} examples")
    
    print(f"\nGeneralizations:")
    for gen in analysis.generalizations:
        print(f"  - {gen}")
    
    # Example 3: Category classification
    print("\n" + "="*80)
    print("EXAMPLE 3: Classification Pattern")
    print("="*80)
    examples = [
        Example(input="apple", output="fruit"),
        Example(input="banana", output="fruit"),
        Example(input="carrot", output="vegetable"),
        Example(input="broccoli", output="vegetable"),
        Example(input="orange", output="fruit"),
        Example(input="spinach", output="vegetable"),
    ]
    
    analysis = agent.reason(examples, domain="Food classification")
    
    print("Examples:")
    for ex in examples:
        print(f"  {ex.input} → {ex.output}")
    
    print(f"\nDiscovered Rules:")
    for pattern in analysis.patterns:
        print(f"\n- {pattern.rule}")
        print(f"  Confidence: {pattern.confidence:.0%}")
    
    # Test with new input
    prediction = agent.predict(analysis, "tomato")
    print(f"\nPrediction for 'tomato':")
    print(f"  {prediction['prediction'][:300]}...")
    
    # Example 4: Sequence patterns
    print("\n" + "="*80)
    print("EXAMPLE 4: Sequence Pattern")
    print("="*80)
    examples = [
        Example(input="position 1", output="2"),
        Example(input="position 2", output="4"),
        Example(input="position 3", output="6"),
        Example(input="position 4", output="8"),
        Example(input="position 5", output="10"),
    ]
    
    analysis = agent.reason(examples, domain="Sequences")
    
    print("Examples:")
    for ex in examples:
        print(f"  {ex.input} → {ex.output}")
    
    print(f"\nPattern:")
    if analysis.patterns:
        pattern = analysis.patterns[0]
        print(f"  {pattern.description}")
        print(f"  Rule: {pattern.rule}")
        print(f"  Confidence: {pattern.confidence:.0%}")
    
    # Example 5: Behavior pattern
    print("\n" + "="*80)
    print("EXAMPLE 5: Behavioral Pattern")
    print("="*80)
    examples = [
        Example(input="Monday morning", output="heavy traffic"),
        Example(input="Tuesday morning", output="heavy traffic"),
        Example(input="Wednesday morning", output="heavy traffic"),
        Example(input="Saturday morning", output="light traffic"),
        Example(input="Sunday morning", output="light traffic"),
    ]
    
    analysis = agent.reason(examples, domain="Traffic patterns")
    
    print("Examples:")
    for ex in examples:
        print(f"  {ex.input} → {ex.output}")
    
    print(f"\nInduced Patterns:")
    for i, pattern in enumerate(analysis.patterns, 1):
        print(f"\n{i}. {pattern.description}")
        print(f"   Confidence: {pattern.confidence:.0%}")
    
    print(f"\nGeneralizations:")
    for gen in analysis.generalizations:
        print(f"  - {gen}")
    
    # Prediction
    prediction = agent.predict(analysis, "Thursday morning")
    print(f"\nPrediction for 'Thursday morning':")
    print(f"  {prediction['prediction'][:200]}...")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Inductive Reasoning Best Practices")
    print("="*80)
    print("""
1. EXAMPLE COLLECTION:
   - Gather diverse, representative examples
   - Ensure sufficient quantity
   - Include edge cases
   - Verify example quality

2. PATTERN IDENTIFICATION:
   - Look for commonalities
   - Consider multiple levels of abstraction
   - Identify exceptions
   - Test alternative patterns

3. VALIDATION:
   - Check against all examples
   - Count supporting vs contradicting evidence
   - Calculate confidence scores
   - Seek counter-examples

4. GENERALIZATION:
   - Move from specific to general
   - Define scope of applicability
   - Acknowledge limitations
   - Avoid over-generalization

5. CONFIDENCE ASSESSMENT:
   - Based on supporting evidence ratio
   - Consider sample size
   - Account for example diversity
   - Update with new evidence

6. APPLICATIONS:
   - Pattern discovery in data
   - Rule learning from examples
   - Trend analysis
   - Predictive modeling
   - Scientific generalization

Benefits:
✓ Discovers patterns from data
✓ Learns from examples
✓ Enables predictions
✓ Practical and applicable
✓ Foundation for machine learning

Limitations:
- Conclusions not guaranteed
- Sensitive to sample bias
- May over-generalize
- Requires sufficient examples
- Exception handling needed
    """)


if __name__ == "__main__":
    demonstrate_inductive_reasoning()
