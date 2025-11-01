"""
Pattern 083: Few-Shot Learning Pattern

Description:
    The Few-Shot Learning pattern uses example-based prompting to teach the LLM
    how to perform tasks through demonstrations. Instead of relying on zero-shot
    capabilities, this pattern provides the model with a few examples of input-output
    pairs that illustrate the desired behavior. This approach leverages the LLM's
    in-context learning abilities to generalize from examples.

    Few-shot learning is one of the most fundamental prompt engineering techniques
    and is particularly effective when:
    - The task format is non-standard or complex
    - Consistent output format is required
    - Specific style or tone needs to be maintained
    - Domain-specific knowledge needs to be demonstrated
    - Edge cases need to be handled in specific ways

Components:
    1. Example Selection
       - Static examples (fixed set)
       - Dynamic examples (selected based on similarity)
       - Diverse examples (covering different scenarios)
       - Representative examples (typical cases)

    2. Example Formatting
       - Clear input-output structure
       - Consistent formatting across examples
       - Annotations and explanations (when needed)
       - Order considerations (easy to hard, chronological)

    3. Example Storage
       - In-prompt storage (small example sets)
       - Vector database storage (large example sets)
       - Retrieval mechanisms (similarity-based)
       - Example caching strategies

    4. Generalization Strategy
       - Pattern recognition from examples
       - Analogical reasoning
       - Template extraction
       - Feature learning

Use Cases:
    1. Text Classification
       - Sentiment analysis with custom categories
       - Intent recognition for domain-specific applications
       - Content moderation with nuanced rules
       - Topic categorization with specific taxonomies

    2. Data Extraction
       - Structured information extraction from unstructured text
       - Entity recognition with custom entity types
       - Relationship extraction between entities
       - Attribute extraction with specific formats

    3. Code Generation
       - API usage examples for specific libraries
       - Code style consistency (following project conventions)
       - Design pattern implementation
       - Test case generation

    4. Creative Writing
       - Style mimicry (writing in specific author's style)
       - Format consistency (poems, stories, articles)
       - Tone control (formal, casual, humorous)
       - Genre-specific writing (sci-fi, mystery, romance)

    5. Translation and Transformation
       - Domain-specific translation (legal, medical, technical)
       - Format conversion (JSON to YAML, CSV to markdown)
       - Data normalization (various formats to standard)
       - Code refactoring patterns

LangChain Implementation:
    LangChain provides excellent support for few-shot learning through:
    - FewShotChatMessagePromptTemplate for chat models
    - FewShotPromptTemplate for completion models
    - Example selectors (semantic similarity, length-based, max marginal relevance)
    - Vector store-backed example selection
    - Custom example formatting

Key Features:
    1. Static Few-Shot Learning
       - Predefined set of examples
       - Fast and deterministic
       - Good for stable tasks
       - No retrieval overhead

    2. Dynamic Few-Shot Learning
       - Examples selected based on input similarity
       - More adaptive to diverse inputs
       - Requires vector database
       - Better generalization

    3. Example Quality Management
       - Diverse examples covering edge cases
       - Clear and unambiguous examples
       - Representative of target distribution
       - Regularly updated based on performance

    4. Performance Optimization
       - Token budget management (limiting example count)
       - Example compression (removing redundancy)
       - Caching frequently used example sets
       - Lazy loading for large example sets

Best Practices:
    1. Example Selection
       - Use 3-5 examples for most tasks (sweet spot)
       - Include diverse examples covering different patterns
       - Order examples from simple to complex
       - Ensure examples are correct and high-quality

    2. Example Formatting
       - Use consistent structure across all examples
       - Clearly separate input from output
       - Add brief explanations for complex cases
       - Use delimiters to avoid confusion

    3. Dynamic Selection
       - Use semantic similarity for input-dependent selection
       - Balance diversity and relevance in selection
       - Cache embeddings for better performance
       - Set appropriate k value (number of examples)

    4. Iteration and Improvement
       - Start with few examples, add more if needed
       - A/B test different example sets
       - Monitor performance across different inputs
       - Update examples based on failure cases

Trade-offs:
    Advantages:
    - Significantly improves task performance vs zero-shot
    - More reliable than instructions alone
    - Enables consistent output formatting
    - Works well for complex or ambiguous tasks
    - Reduces need for fine-tuning

    Disadvantages:
    - Increases token usage (more expensive)
    - May not scale to tasks requiring many examples
    - Examples can bias the model inappropriately
    - Requires careful example curation
    - Dynamic selection adds latency and complexity

Production Considerations:
    1. Token Budget Management
       - Monitor tokens used by examples
       - Balance example count with context window
       - Use example compression when possible
       - Consider model's context limit

    2. Example Maintenance
       - Version control for example sets
       - Regular review and updates
       - Performance tracking per example
       - A/B testing of example variations

    3. Scalability
       - Cache example embeddings
       - Use efficient vector stores
       - Batch process when possible
       - Consider example pruning strategies

    4. Quality Assurance
       - Validate examples are correct
       - Test examples independently
       - Monitor output quality metrics
       - Handle example selection failures gracefully

    5. Cost Optimization
       - Balance example count with quality
       - Use dynamic selection judiciously
       - Cache common example sets
       - Monitor cost per query
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate
)
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.example_selectors import (
    SemanticSimilarityExampleSelector,
    MaxMarginalRelevanceExampleSelector
)
from langchain_chroma import Chroma

load_dotenv()


class ExampleSelectionStrategy(Enum):
    """Strategy for selecting examples"""
    STATIC = "static"  # Fixed set of examples
    SEMANTIC_SIMILARITY = "semantic_similarity"  # Based on input similarity
    MAX_MARGINAL_RELEVANCE = "max_marginal_relevance"  # Diverse + relevant


@dataclass
class FewShotExample:
    """Represents a single few-shot example"""
    input: str
    output: str
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FewShotConfig:
    """Configuration for few-shot learning"""
    num_examples: int = 3
    selection_strategy: ExampleSelectionStrategy = ExampleSelectionStrategy.STATIC
    temperature: float = 0.3
    model_name: str = "gpt-3.5-turbo"
    include_explanations: bool = False


class FewShotLearningAgent:
    """
    Agent that uses few-shot learning to teach LLM through examples.
    
    This agent demonstrates various few-shot learning approaches:
    1. Static few-shot (fixed examples)
    2. Dynamic few-shot (similarity-based selection)
    3. Task-specific few-shot applications
    """
    
    def __init__(self, config: Optional[FewShotConfig] = None):
        """
        Initialize few-shot learning agent.
        
        Args:
            config: Configuration for few-shot learning
        """
        self.config = config or FewShotConfig()
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature
        )
        self.embeddings = OpenAIEmbeddings()
    
    def create_static_few_shot_chain(
        self,
        examples: List[FewShotExample],
        task_instruction: str,
        input_variable: str = "input"
    ):
        """
        Create a chain with static few-shot examples.
        
        Args:
            examples: List of few-shot examples
            task_instruction: Instruction for the task
            input_variable: Name of input variable
            
        Returns:
            Runnable chain with few-shot prompting
        """
        # Format examples as dictionaries
        example_dicts = [
            {"input": ex.input, "output": ex.output}
            for ex in examples
        ]
        
        # Create example prompt template
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ])
        
        # Create few-shot prompt
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=example_dicts
        )
        
        # Create final prompt
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", task_instruction),
            few_shot_prompt,
            ("human", "{" + input_variable + "}")
        ])
        
        # Create chain
        chain = final_prompt | self.llm | StrOutputParser()
        
        return chain
    
    def create_dynamic_few_shot_chain(
        self,
        examples: List[FewShotExample],
        task_instruction: str,
        input_variable: str = "input",
        k: int = 3
    ):
        """
        Create a chain with dynamic example selection based on similarity.
        
        Args:
            examples: Pool of examples to select from
            task_instruction: Instruction for the task
            input_variable: Name of input variable
            k: Number of examples to select
            
        Returns:
            Runnable chain with dynamic few-shot prompting
        """
        # Format examples as dictionaries
        example_dicts = [
            {"input": ex.input, "output": ex.output}
            for ex in examples
        ]
        
        # Create example selector using semantic similarity
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            example_dicts,
            self.embeddings,
            Chroma,
            k=k,
            input_keys=["input"]
        )
        
        # Create example prompt template
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ])
        
        # Create few-shot prompt with selector
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            example_selector=example_selector,
            input_variables=[input_variable]
        )
        
        # Create final prompt
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", task_instruction),
            few_shot_prompt,
            ("human", "{" + input_variable + "}")
        ])
        
        # Create chain
        chain = final_prompt | self.llm | StrOutputParser()
        
        return chain
    
    def sentiment_analysis_few_shot(
        self,
        text: str,
        examples: Optional[List[FewShotExample]] = None,
        use_dynamic: bool = False
    ) -> str:
        """
        Perform sentiment analysis using few-shot learning.
        
        Args:
            text: Text to analyze
            examples: Optional custom examples
            use_dynamic: Whether to use dynamic selection
            
        Returns:
            Sentiment classification
        """
        # Default examples if none provided
        if examples is None:
            examples = [
                FewShotExample(
                    input="This movie was absolutely fantastic! I loved every minute.",
                    output="Positive"
                ),
                FewShotExample(
                    input="The product broke after one day. Terrible quality.",
                    output="Negative"
                ),
                FewShotExample(
                    input="It's okay, nothing special but not bad either.",
                    output="Neutral"
                ),
                FewShotExample(
                    input="I'm thrilled with this purchase! Exceeded expectations.",
                    output="Positive"
                ),
                FewShotExample(
                    input="Complete waste of money. Very disappointed.",
                    output="Negative"
                )
            ]
        
        task_instruction = """You are a sentiment analyzer. Classify the sentiment of the given text as Positive, Negative, or Neutral.
Look at the examples to understand the task."""
        
        # Create chain based on selection strategy
        if use_dynamic:
            chain = self.create_dynamic_few_shot_chain(
                examples,
                task_instruction,
                k=3
            )
        else:
            # Use first 3 examples for static
            chain = self.create_static_few_shot_chain(
                examples[:3],
                task_instruction
            )
        
        result = chain.invoke({"input": text})
        return result
    
    def entity_extraction_few_shot(
        self,
        text: str,
        examples: Optional[List[FewShotExample]] = None
    ) -> str:
        """
        Extract entities using few-shot learning.
        
        Args:
            text: Text to extract entities from
            examples: Optional custom examples
            
        Returns:
            Extracted entities in JSON format
        """
        # Default examples
        if examples is None:
            examples = [
                FewShotExample(
                    input="Apple Inc. announced a new iPhone model in Cupertino on September 12th.",
                    output='{"organization": "Apple Inc.", "product": "iPhone", "location": "Cupertino", "date": "September 12th"}'
                ),
                FewShotExample(
                    input="Elon Musk tweeted about Tesla's production numbers from the Fremont factory.",
                    output='{"person": "Elon Musk", "organization": "Tesla", "location": "Fremont factory"}'
                ),
                FewShotExample(
                    input="The meeting at Microsoft headquarters in Seattle was attended by Satya Nadella.",
                    output='{"organization": "Microsoft", "location": "Seattle", "person": "Satya Nadella"}'
                )
            ]
        
        task_instruction = """Extract entities from the text and format them as JSON.
Include: person, organization, location, date, product (if present).
Follow the format shown in the examples."""
        
        chain = self.create_static_few_shot_chain(
            examples,
            task_instruction
        )
        
        result = chain.invoke({"input": text})
        return result
    
    def code_generation_few_shot(
        self,
        description: str,
        examples: Optional[List[FewShotExample]] = None
    ) -> str:
        """
        Generate code using few-shot learning.
        
        Args:
            description: Description of desired code
            examples: Optional custom examples
            
        Returns:
            Generated code
        """
        # Default examples
        if examples is None:
            examples = [
                FewShotExample(
                    input="Create a function that returns the sum of two numbers",
                    output="""def add_numbers(a, b):
    \"\"\"Return the sum of two numbers.\"\"\"
    return a + b"""
                ),
                FewShotExample(
                    input="Create a function that checks if a string is a palindrome",
                    output="""def is_palindrome(s):
    \"\"\"Check if a string is a palindrome.\"\"\"
    return s == s[::-1]"""
                ),
                FewShotExample(
                    input="Create a function that finds the maximum value in a list",
                    output="""def find_max(lst):
    \"\"\"Find the maximum value in a list.\"\"\"
    if not lst:
        return None
    return max(lst)"""
                )
            ]
        
        task_instruction = """Generate Python code based on the description.
Follow the style and format shown in the examples.
Include docstrings and proper formatting."""
        
        chain = self.create_static_few_shot_chain(
            examples,
            task_instruction,
            input_variable="input"
        )
        
        result = chain.invoke({"input": description})
        return result
    
    def style_transfer_few_shot(
        self,
        text: str,
        target_style: str,
        examples: Optional[List[FewShotExample]] = None
    ) -> str:
        """
        Transfer text to a specific style using few-shot learning.
        
        Args:
            text: Text to transform
            target_style: Target style (e.g., "formal", "casual", "poetic")
            examples: Optional custom examples
            
        Returns:
            Transformed text
        """
        # Style-specific examples
        style_examples = {
            "formal": [
                FewShotExample(
                    input="Hey, can you help me out with this?",
                    output="I would appreciate your assistance with this matter."
                ),
                FewShotExample(
                    input="That's totally wrong!",
                    output="I respectfully disagree with that assessment."
                ),
                FewShotExample(
                    input="Thanks a lot for everything!",
                    output="I extend my sincere gratitude for your comprehensive support."
                )
            ],
            "casual": [
                FewShotExample(
                    input="I would like to request your assistance.",
                    output="Hey, could you help me out?"
                ),
                FewShotExample(
                    input="This is highly problematic.",
                    output="This is a real problem."
                ),
                FewShotExample(
                    input="I appreciate your effort.",
                    output="Thanks for trying!"
                )
            ],
            "poetic": [
                FewShotExample(
                    input="The sun set over the mountains.",
                    output="Golden rays bid farewell, as peaks embrace the fading light."
                ),
                FewShotExample(
                    input="Rain started falling.",
                    output="Heaven's tears descended, kissing the earth below."
                ),
                FewShotExample(
                    input="The bird flew away.",
                    output="On wings of freedom, it soared beyond the horizon's edge."
                )
            ]
        }
        
        examples = examples or style_examples.get(target_style.lower(), [])
        
        if not examples:
            return "Style not supported. Available: formal, casual, poetic"
        
        task_instruction = f"""Transform the text into {target_style} style.
Follow the transformation pattern shown in the examples."""
        
        chain = self.create_static_few_shot_chain(
            examples,
            task_instruction
        )
        
        result = chain.invoke({"input": text})
        return result
    
    def compare_zero_shot_vs_few_shot(
        self,
        text: str,
        task: str = "sentiment"
    ) -> Dict[str, str]:
        """
        Compare zero-shot vs few-shot performance.
        
        Args:
            text: Input text
            task: Task to perform
            
        Returns:
            Dictionary with both results
        """
        # Zero-shot prompt
        zero_shot_prompt = ChatPromptTemplate.from_messages([
            ("system", "Classify the sentiment as Positive, Negative, or Neutral."),
            ("human", "{input}")
        ])
        zero_shot_chain = zero_shot_prompt | self.llm | StrOutputParser()
        zero_shot_result = zero_shot_chain.invoke({"input": text})
        
        # Few-shot result
        few_shot_result = self.sentiment_analysis_few_shot(text)
        
        return {
            "zero_shot": zero_shot_result,
            "few_shot": few_shot_result
        }


def demonstrate_few_shot_learning():
    """Demonstrate few-shot learning patterns"""
    print("=" * 80)
    print("FEW-SHOT LEARNING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = FewShotLearningAgent()
    
    # Example 1: Sentiment Analysis
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Sentiment Analysis with Few-Shot Learning")
    print("=" * 80)
    
    test_texts = [
        "This restaurant exceeded all my expectations! Amazing food!",
        "The service was slow and the food was cold. Not going back.",
        "It was fine. Nothing special but acceptable."
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        sentiment = agent.sentiment_analysis_few_shot(text)
        print(f"Sentiment: {sentiment}")
    
    # Example 2: Entity Extraction
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Entity Extraction with Few-Shot Learning")
    print("=" * 80)
    
    test_text = "Jeff Bezos announced Amazon's expansion into the healthcare sector at a conference in Las Vegas last Monday."
    print(f"\nText: {test_text}")
    entities = agent.entity_extraction_few_shot(test_text)
    print(f"Entities: {entities}")
    
    # Example 3: Code Generation
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Code Generation with Few-Shot Learning")
    print("=" * 80)
    
    code_requests = [
        "Create a function that calculates factorial of a number",
        "Create a function that reverses a string"
    ]
    
    for request in code_requests:
        print(f"\nRequest: {request}")
        code = agent.code_generation_few_shot(request)
        print(f"Generated Code:\n{code}")
    
    # Example 4: Style Transfer
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Style Transfer with Few-Shot Learning")
    print("=" * 80)
    
    original_text = "I need your help with this problem."
    styles = ["formal", "casual", "poetic"]
    
    print(f"\nOriginal: {original_text}")
    for style in styles:
        transformed = agent.style_transfer_few_shot(original_text, style)
        print(f"{style.title()} style: {transformed}")
    
    # Example 5: Zero-Shot vs Few-Shot Comparison
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Zero-Shot vs Few-Shot Comparison")
    print("=" * 80)
    
    ambiguous_text = "The movie was interesting and thought-provoking, though quite challenging to follow."
    print(f"\nAmbiguous Text: {ambiguous_text}")
    
    comparison = agent.compare_zero_shot_vs_few_shot(ambiguous_text)
    print(f"\nZero-Shot Result: {comparison['zero_shot']}")
    print(f"Few-Shot Result: {comparison['few_shot']}")
    print("\nNote: Few-shot learning often provides more consistent and format-compliant outputs")
    
    # Example 6: Dynamic Example Selection
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Dynamic Example Selection (Similarity-Based)")
    print("=" * 80)
    
    test_text = "I'm so excited about this purchase! Best decision ever!"
    print(f"\nText: {test_text}")
    print("\nStatic Selection:")
    static_result = agent.sentiment_analysis_few_shot(test_text, use_dynamic=False)
    print(f"Result: {static_result}")
    
    print("\nDynamic Selection (picks most similar examples):")
    dynamic_result = agent.sentiment_analysis_few_shot(test_text, use_dynamic=True)
    print(f"Result: {dynamic_result}")
    
    # Summary
    print("\n" + "=" * 80)
    print("FEW-SHOT LEARNING SUMMARY")
    print("=" * 80)
    print("""
Few-Shot Learning Pattern Benefits:
1. Improved Accuracy: Examples guide the model to better performance
2. Consistency: Output format matches examples consistently
3. Task Clarity: Examples disambiguate task requirements
4. Style Control: Examples demonstrate desired tone and style
5. No Fine-Tuning: Achieves good results without model retraining

Key Techniques Demonstrated:
1. Static Few-Shot: Fixed examples for consistent tasks
2. Dynamic Few-Shot: Similarity-based example selection
3. Task-Specific Examples: Domain-appropriate demonstrations
4. Style Transfer: Teaching specific writing styles
5. Format Control: Ensuring structured outputs

Best Practices:
1. Use 3-5 examples for most tasks (sweet spot)
2. Ensure examples are diverse and representative
3. Order examples from simple to complex
4. Include edge cases in examples
5. Keep examples concise to save tokens
6. Use dynamic selection for varied inputs
7. Validate examples are correct and high-quality
8. Monitor token usage and costs

When to Use Few-Shot vs Zero-Shot:
- Use few-shot when: output format is specific, task is ambiguous, consistency matters
- Use zero-shot when: task is simple, tokens are limited, speed is critical
- Use dynamic selection when: inputs are diverse, quality is paramount

Production Considerations:
- Cache example embeddings for performance
- Monitor token usage (examples increase costs)
- A/B test different example sets
- Version control example datasets
- Update examples based on failure cases
- Balance example count with quality needs
""")
    
    print("\n" + "=" * 80)
    print("Pattern 083 (Few-Shot Learning) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_few_shot_learning()
