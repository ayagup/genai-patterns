"""
Pattern 086: Output Format Specification Pattern

Description:
    The Output Format Specification pattern explicitly defines the structure, format,
    and schema of the LLM's response. This pattern is crucial for integrating LLM
    outputs with downstream systems, APIs, and applications that require consistent,
    parseable, and structured data. By specifying the exact format upfront, this
    pattern eliminates ambiguity and ensures reliable programmatic processing.

    Format specification is essential for:
    - API integration and data exchange
    - Structured data extraction
    - Multi-step workflows with defined interfaces
    - Database storage and retrieval
    - User interface rendering
    - Automated testing and validation

Components:
    1. Format Type
       - JSON (JavaScript Object Notation)
       - XML (Extensible Markup Language)
       - YAML (YAML Ain't Markup Language)
       - CSV (Comma-Separated Values)
       - Markdown (formatted text)
       - Custom formats (domain-specific)

    2. Schema Definition
       - Field names and types
       - Required vs optional fields
       - Value constraints and ranges
       - Nested structures
       - Array/list specifications

    3. Format Enforcement
       - Clear examples in prompts
       - Schema validation
       - Error handling for format violations
       - Retry mechanisms
       - Fallback strategies

    4. Parsing and Validation
       - Automated parsing
       - Schema compliance checking
       - Type validation
       - Constraint verification
       - Error reporting

Use Cases:
    1. Data Extraction
       - Entity extraction from text → JSON
       - Information retrieval → structured records
       - Document parsing → database entries
       - Web scraping → CSV data
       - Form filling → key-value pairs

    2. API Integration
       - API request generation → JSON payloads
       - Response formatting → standardized structures
       - Webhook payloads → event schemas
       - Configuration files → YAML/JSON
       - Data exchange → XML documents

    3. Report Generation
       - Business reports → Markdown tables
       - Data summaries → formatted text
       - Analytics → JSON metrics
       - Documentation → structured sections
       - Dashboards → visualization specs

    4. Code Generation
       - Function signatures → specific syntax
       - Configuration files → language-specific format
       - Test cases → structured test specs
       - API documentation → OpenAPI/Swagger
       - Database schemas → SQL/DDL

    5. Content Creation
       - Blog posts → Markdown with frontmatter
       - Product descriptions → structured fields
       - Email templates → HTML with placeholders
       - Social media → platform-specific formats
       - Translations → parallel text structures

LangChain Implementation:
    LangChain provides excellent format control through:
    - StructuredOutputParser for schema-based parsing
    - JsonOutputParser for JSON responses
    - PydanticOutputParser for type-safe outputs
    - Custom output parsers for specific formats
    - Output fixing parser for auto-correction

Key Features:
    1. Type Safety
       - Pydantic models for validation
       - Automatic type checking
       - Type coercion when possible
       - Clear error messages

    2. Schema Validation
       - JSON Schema compliance
       - Custom validation rules
       - Nested object validation
       - Array element validation

    3. Error Recovery
       - Automatic retry with corrections
       - Format fixing parsers
       - Fallback to alternative formats
       - Graceful degradation

    4. Multiple Format Support
       - JSON (most common)
       - XML for legacy systems
       - CSV for tabular data
       - Markdown for documents
       - Custom formats as needed

Best Practices:
    1. Clear Format Specification
       - Provide explicit format examples
       - Use schema definitions when possible
       - Show complete structure (not partial)
       - Include edge cases in examples

    2. Validation Strategy
       - Validate immediately after generation
       - Provide clear error messages
       - Implement retry logic
       - Log format violations

    3. Error Handling
       - Graceful fallbacks
       - User-friendly error messages
       - Automatic format correction when possible
       - Manual intervention for critical failures

    4. Documentation
       - Document expected formats
       - Provide format examples
       - Explain field meanings
       - Note format versions

Trade-offs:
    Advantages:
    - Reliable programmatic processing
    - Easy integration with systems
    - Reduced parsing errors
    - Type safety and validation
    - Better error detection
    - Automated testing possible

    Disadvantages:
    - More rigid (less flexibility)
    - Format violations require handling
    - Increases prompt complexity
    - May constrain natural language quality
    - Requires careful schema design
    - Parsing overhead

Production Considerations:
    1. Schema Evolution
       - Version format schemas
       - Backward compatibility
       - Migration strategies
       - Deprecation policies

    2. Performance
       - Parsing efficiency
       - Validation overhead
       - Caching strategies
       - Batch processing

    3. Reliability
       - Retry logic for format errors
       - Format fixing mechanisms
       - Fallback formats
       - Monitoring and alerting

    4. Maintenance
       - Schema documentation
       - Format validation tests
       - Breaking change detection
       - Client SDK generation

    5. Security
       - Input sanitization
       - Injection prevention
       - Size limits
       - Type validation
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser
)
from langchain_openai import ChatOpenAI

load_dotenv()


class OutputFormat(Enum):
    """Supported output formats"""
    JSON = "json"
    XML = "xml"
    MARKDOWN = "markdown"
    CSV = "csv"
    YAML = "yaml"


# Pydantic models for structured outputs
class Person(BaseModel):
    """Person entity"""
    name: str = Field(description="Full name of the person")
    age: Optional[int] = Field(description="Age in years", ge=0, le=150)
    email: Optional[str] = Field(description="Email address")
    occupation: Optional[str] = Field(description="Job title or occupation")


class Product(BaseModel):
    """Product information"""
    name: str = Field(description="Product name")
    category: str = Field(description="Product category")
    price: float = Field(description="Price in USD", gt=0)
    in_stock: bool = Field(description="Availability status")
    rating: Optional[float] = Field(description="Average rating", ge=0, le=5)


class Article(BaseModel):
    """Article/content structure"""
    title: str = Field(description="Article title")
    author: str = Field(description="Author name")
    summary: str = Field(description="Brief summary")
    tags: List[str] = Field(description="Content tags")
    word_count: int = Field(description="Approximate word count", gt=0)


class SentimentAnalysis(BaseModel):
    """Sentiment analysis result"""
    text: str = Field(description="Analyzed text")
    sentiment: str = Field(description="Sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score", ge=0, le=1)
    keywords: List[str] = Field(description="Key terms identified")


@dataclass
class FormatConfig:
    """Configuration for format specification"""
    format_type: OutputFormat = OutputFormat.JSON
    strict_mode: bool = True
    temperature: float = 0.1  # Lower for more consistent formatting
    model_name: str = "gpt-3.5-turbo"


class OutputFormatAgent:
    """
    Agent that enforces specific output formats.
    
    This agent demonstrates:
    1. JSON format specification
    2. Pydantic-based type-safe outputs
    3. Multiple format support
    4. Schema validation
    """
    
    def __init__(self, config: Optional[FormatConfig] = None):
        """
        Initialize output format agent.
        
        Args:
            config: Configuration for format handling
        """
        self.config = config or FormatConfig()
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature
        )
    
    def extract_as_json(
        self,
        text: str,
        schema_description: str
    ) -> Dict[str, Any]:
        """
        Extract information in JSON format.
        
        Args:
            text: Text to extract from
            schema_description: Description of desired JSON structure
            
        Returns:
            Extracted data as dictionary
        """
        json_parser = JsonOutputParser()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract information from the text in JSON format.

Format requirements:
- Valid JSON only
- Follow the schema described
- Use appropriate data types
- Include all requested fields

{format_instructions}"""),
            ("human", "Extract from this text: {text}\n\nSchema: {schema}")
        ])
        
        chain = prompt | self.llm | json_parser
        
        try:
            result = chain.invoke({
                "text": text,
                "schema": schema_description,
                "format_instructions": json_parser.get_format_instructions()
            })
            return result
        except Exception as e:
            return {"error": str(e), "raw_text": text}
    
    def extract_person_info(self, text: str) -> Person:
        """
        Extract person information with type safety.
        
        Args:
            text: Text containing person information
            
        Returns:
            Person object with validated fields
        """
        parser = PydanticOutputParser(pydantic_object=Person)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract person information from the text.

{format_instructions}

Provide valid data for all available fields."""),
            ("human", "{text}")
        ])
        
        chain = prompt | self.llm | parser
        
        result = chain.invoke({
            "text": text,
            "format_instructions": parser.get_format_instructions()
        })
        
        return result
    
    def extract_product_info(self, text: str) -> Product:
        """
        Extract product information with validation.
        
        Args:
            text: Text containing product information
            
        Returns:
            Product object with validated fields
        """
        parser = PydanticOutputParser(pydantic_object=Product)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract product information from the text.

{format_instructions}

Ensure all data types are correct and constraints are met."""),
            ("human", "{text}")
        ])
        
        chain = prompt | self.llm | parser
        
        result = chain.invoke({
            "text": text,
            "format_instructions": parser.get_format_instructions()
        })
        
        return result
    
    def analyze_sentiment_structured(self, text: str) -> SentimentAnalysis:
        """
        Perform sentiment analysis with structured output.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentAnalysis object with validated fields
        """
        parser = PydanticOutputParser(pydantic_object=SentimentAnalysis)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the sentiment of the text and provide results in the specified format.

{format_instructions}

- Sentiment must be: positive, negative, or neutral
- Confidence must be between 0 and 1
- Extract 3-5 relevant keywords"""),
            ("human", "{text}")
        ])
        
        chain = prompt | self.llm | parser
        
        result = chain.invoke({
            "text": text,
            "format_instructions": parser.get_format_instructions()
        })
        
        return result
    
    def generate_markdown_report(
        self,
        topic: str,
        sections: List[str]
    ) -> str:
        """
        Generate report in markdown format.
        
        Args:
            topic: Report topic
            sections: List of section titles
            
        Returns:
            Markdown formatted report
        """
        sections_list = "\n".join([f"- {s}" for s in sections])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate a report in Markdown format.

Format requirements:
- Use # for main title
- Use ## for section headers
- Use **bold** for emphasis
- Use bullet points (- ) for lists
- Use code blocks (```) for code
- Maintain consistent formatting"""),
            ("human", """Create a report on: {topic}

Include these sections:
{sections}""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({
            "topic": topic,
            "sections": sections_list
        })
        
        return result
    
    def generate_csv_format(
        self,
        data_description: str,
        num_rows: int = 5
    ) -> str:
        """
        Generate data in CSV format.
        
        Args:
            data_description: Description of data to generate
            num_rows: Number of data rows
            
        Returns:
            CSV formatted data
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate data in CSV format.

Format requirements:
- First row: column headers
- Subsequent rows: data values
- Use commas as separators
- Quote strings containing commas
- No extra whitespace
- Valid CSV syntax"""),
            ("human", """Generate {num_rows} rows of CSV data for: {description}""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({
            "description": data_description,
            "num_rows": num_rows
        })
        
        return result
    
    def generate_xml_format(self, data_description: str) -> str:
        """
        Generate data in XML format.
        
        Args:
            data_description: Description of data structure
            
        Returns:
            XML formatted data
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Generate data in XML format.

Format requirements:
- Proper XML declaration
- Well-formed tags (open and close)
- Proper nesting
- Attribute syntax when appropriate
- Valid XML structure"""),
            ("human", "Generate XML for: {description}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        result = chain.invoke({"description": data_description})
        
        return result
    
    def compare_formats(self, data: str) -> Dict[str, str]:
        """
        Generate same data in multiple formats for comparison.
        
        Args:
            data: Data description
            
        Returns:
            Dictionary with data in different formats
        """
        formats = {}
        
        # JSON format
        json_result = self.extract_as_json(
            data,
            "Extract as JSON with appropriate fields"
        )
        formats["json"] = json.dumps(json_result, indent=2)
        
        # XML format
        formats["xml"] = self.generate_xml_format(data)
        
        # CSV format (if tabular)
        formats["csv"] = self.generate_csv_format(data, num_rows=3)
        
        return formats


def demonstrate_output_format():
    """Demonstrate output format specification patterns"""
    print("=" * 80)
    print("OUTPUT FORMAT SPECIFICATION PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = OutputFormatAgent()
    
    # Example 1: JSON Extraction
    print("\n" + "=" * 80)
    print("EXAMPLE 1: JSON Format Extraction")
    print("=" * 80)
    
    text = "Apple Inc. released the iPhone 15 Pro in September 2023 priced at $999."
    schema = "Extract product name, company, release date, and price"
    print(f"\nText: {text}")
    print(f"Schema: {schema}\n")
    
    json_result = agent.extract_as_json(text, schema)
    print("JSON Output:")
    print(json.dumps(json_result, indent=2))
    
    # Example 2: Type-Safe Person Extraction
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Type-Safe Person Information (Pydantic)")
    print("=" * 80)
    
    person_text = "Dr. Sarah Johnson is a 45-year-old cardiologist at Mayo Clinic. Her email is sarah.j@mayo.edu"
    print(f"\nText: {person_text}\n")
    
    person = agent.extract_person_info(person_text)
    print("Extracted Person (Type-Safe):")
    print(f"  Name: {person.name}")
    print(f"  Age: {person.age}")
    print(f"  Occupation: {person.occupation}")
    print(f"  Email: {person.email}")
    print(f"\nObject type: {type(person)}")
    
    # Example 3: Product Information
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Structured Product Information")
    print("=" * 80)
    
    product_text = "The UltraBook Pro laptop in the Electronics category costs $1,299. Currently in stock with 4.5 star rating."
    print(f"\nText: {product_text}\n")
    
    product = agent.extract_product_info(product_text)
    print("Extracted Product:")
    print(f"  Name: {product.name}")
    print(f"  Category: {product.category}")
    print(f"  Price: ${product.price}")
    print(f"  In Stock: {product.in_stock}")
    print(f"  Rating: {product.rating}")
    
    # Example 4: Sentiment Analysis with Structure
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Structured Sentiment Analysis")
    print("=" * 80)
    
    review = "This product is absolutely amazing! Best purchase I've made this year. Highly recommended!"
    print(f"\nReview: {review}\n")
    
    sentiment = agent.analyze_sentiment_structured(review)
    print("Sentiment Analysis Result:")
    print(f"  Sentiment: {sentiment.sentiment}")
    print(f"  Confidence: {sentiment.confidence:.2f}")
    print(f"  Keywords: {', '.join(sentiment.keywords)}")
    
    # Example 5: Markdown Report
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Markdown Formatted Report")
    print("=" * 80)
    
    topic = "Quarterly Sales Performance"
    sections = ["Executive Summary", "Key Metrics", "Regional Analysis", "Recommendations"]
    print(f"\nTopic: {topic}")
    print(f"Sections: {sections}\n")
    
    markdown_report = agent.generate_markdown_report(topic, sections)
    print("Markdown Output:")
    print(markdown_report)
    
    # Example 6: CSV Format
    print("\n" + "=" * 80)
    print("EXAMPLE 6: CSV Format Generation")
    print("=" * 80)
    
    data_desc = "Employee records with name, department, salary, and hire date"
    print(f"\nData: {data_desc}\n")
    
    csv_data = agent.generate_csv_format(data_desc, num_rows=5)
    print("CSV Output:")
    print(csv_data)
    
    # Example 7: XML Format
    print("\n" + "=" * 80)
    print("EXAMPLE 7: XML Format Generation")
    print("=" * 80)
    
    xml_desc = "Book catalog with title, author, ISBN, and publication year"
    print(f"\nData: {xml_desc}\n")
    
    xml_data = agent.generate_xml_format(xml_desc)
    print("XML Output:")
    print(xml_data)
    
    # Summary
    print("\n" + "=" * 80)
    print("OUTPUT FORMAT SPECIFICATION SUMMARY")
    print("=" * 80)
    print("""
Output Format Pattern Benefits:
1. Programmatic Processing: Easy to parse and use in code
2. Type Safety: Validation ensures correct data types
3. Integration Ready: Works seamlessly with APIs and systems
4. Error Detection: Schema validation catches issues early
5. Documentation: Format serves as documentation
6. Testing: Enables automated testing

Key Formats Demonstrated:
1. JSON: Most common, flexible, widely supported
2. Pydantic Models: Type-safe Python objects with validation
3. Markdown: Human-readable formatted documents
4. CSV: Tabular data for spreadsheets
5. XML: Structured data for legacy systems

Format Specification Techniques:
1. Explicit Examples: Show exact format in prompt
2. Schema Definition: Use JSON Schema or Pydantic
3. Format Instructions: Parser-generated instructions
4. Validation: Automatic type and constraint checking
5. Error Recovery: Retry and fix malformed outputs

Best Practices:
1. Choose appropriate format for use case
   - JSON: APIs, web apps, general data
   - CSV: Tabular data, spreadsheets
   - XML: Legacy systems, complex hierarchies
   - Markdown: Documentation, reports
   
2. Use type safety when possible (Pydantic)
   - Automatic validation
   - IDE support
   - Clear error messages
   
3. Provide clear format instructions
   - Show complete examples
   - Specify all fields
   - Define data types and constraints
   
4. Handle format errors gracefully
   - Retry with corrections
   - Fallback formats
   - Clear error messages
   
5. Validate outputs
   - Schema compliance
   - Type checking
   - Constraint verification

Production Considerations:
- Lower temperature (0.1-0.3) for consistent formatting
- Implement retry logic for format violations
- Cache format instructions to save tokens
- Version schemas for evolution
- Monitor parsing success rates
- Log format errors for debugging
- Test edge cases thoroughly

Format Selection Guide:
- JSON: Default choice for most applications
- Pydantic: When type safety is critical
- CSV: For tabular data and spreadsheet integration
- XML: For legacy system compatibility
- Markdown: For human-readable documents
- YAML: For configuration files

Common Pitfalls:
- Insufficient format examples in prompts
- No validation of generated outputs
- Overly complex schemas
- Missing error handling
- Temperature too high (inconsistent outputs)
- Not handling optional fields properly
""")
    
    print("\n" + "=" * 80)
    print("Pattern 086 (Output Format Specification) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_output_format()
