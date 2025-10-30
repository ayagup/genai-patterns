"""
Pattern 071: Code Agent Patterns

Description:
    Code Agent Patterns represent specialized agents designed for software development tasks
    including code analysis, generation, refactoring, debugging, testing, and documentation.
    These agents understand programming languages, design patterns, best practices, and can
    interact with codebases to assist developers. They combine static analysis, dynamic
    execution, code understanding, and generation capabilities to provide intelligent
    programming assistance.
    
    Code agents go beyond simple code completion to understand context, intent, architecture,
    and can reason about code quality, performance, security, and maintainability. They can
    explain existing code, suggest improvements, generate tests, refactor implementations,
    and even help with system design.

Components:
    1. Code Parser: Analyzes syntax and structure
    2. Code Generator: Creates new code based on specifications
    3. Code Analyzer: Identifies issues, patterns, and opportunities
    4. Code Refactorer: Improves code structure and quality
    5. Test Generator: Creates unit and integration tests
    6. Documentation Generator: Produces code documentation
    7. Debugger Assistant: Helps identify and fix bugs

Architecture:
    ```
    Code Task Request
        ↓
    Task Classifier → [Analyze/Generate/Refactor/Debug/Test/Document]
        ↓
    Code Understanding → [Parse, analyze structure & semantics]
        ↓
    Task Executor → [Execute specialized task]
        ↓
    Code Validator → [Check correctness & quality]
        ↓
    Result + Explanation
    ```

Use Cases:
    - Automated code review and quality assessment
    - Intelligent code completion and generation
    - Refactoring suggestions and automation
    - Bug detection and debugging assistance
    - Test generation and coverage improvement
    - Documentation generation and maintenance
    - Code translation between languages
    - Architecture and design assistance

Advantages:
    - Accelerates development workflow
    - Improves code quality and consistency
    - Reduces bugs and technical debt
    - Provides learning opportunities for developers
    - Automates repetitive coding tasks
    - Maintains coding standards automatically

LangChain Implementation:
    Uses ChatOpenAI for code understanding and generation. Demonstrates
    code analysis, generation, refactoring, test creation, and documentation.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class CodeTaskType(Enum):
    """Types of code-related tasks."""
    ANALYZE = "analyze"
    GENERATE = "generate"
    REFACTOR = "refactor"
    DEBUG = "debug"
    TEST = "test"
    DOCUMENT = "document"
    REVIEW = "review"
    EXPLAIN = "explain"


class CodeLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"


class CodeQualityIssue(Enum):
    """Types of code quality issues."""
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    COMPLEXITY = "complexity"
    DUPLICATION = "duplication"
    MAINTAINABILITY = "maintainability"


@dataclass
class CodeSnippet:
    """Represents a piece of code."""
    code: str
    language: CodeLanguage
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass
class CodeIssue:
    """Represents a code quality issue."""
    issue_type: CodeQualityIssue
    severity: str  # "critical", "high", "medium", "low"
    description: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class CodeAnalysis:
    """Results of code analysis."""
    snippet: CodeSnippet
    issues: List[CodeIssue]
    complexity_score: float
    maintainability_score: float
    summary: str


@dataclass
class GeneratedCode:
    """Generated code with metadata."""
    code: str
    language: CodeLanguage
    explanation: str
    test_code: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class RefactoredCode:
    """Refactored code with changes."""
    original: str
    refactored: str
    changes: List[str]
    improvements: List[str]


class CodeAgent:
    """
    Specialized agent for code-related tasks.
    """
    
    def __init__(self, default_language: CodeLanguage = CodeLanguage.PYTHON):
        """
        Initialize code agent.
        
        Args:
            default_language: Default programming language
        """
        self.default_language = default_language
        
        # Specialized LLMs for different tasks
        self.analyzer = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
        self.generator = ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo")
        self.refactorer = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
        self.explainer = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo")
        
        self.parser = StrOutputParser()
    
    def analyze_code(self, snippet: CodeSnippet) -> CodeAnalysis:
        """
        Analyze code for quality issues.
        
        Args:
            snippet: Code to analyze
            
        Returns:
            Code analysis with issues and metrics
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a code quality expert. Analyze the code and identify issues.

Consider:
1. Bugs and potential errors
2. Security vulnerabilities
3. Performance issues
4. Code style and best practices
5. Complexity and maintainability
6. Code duplication

Provide:
Issues: [type:severity:line:description:suggestion, ...]
Complexity: [1-10 score]
Maintainability: [1-10 score]
Summary: [brief overall assessment]

Format each issue as: type:severity:line:description:suggestion"""),
            ("user", """Language: {language}
Code:
```
{code}
```

Analyze this code:""")
        ])
        
        chain = prompt | self.analyzer | self.parser
        
        try:
            result = chain.invoke({
                "language": snippet.language.value,
                "code": snippet.code
            })
            
            # Parse result
            issues = []
            complexity_score = 5.0
            maintainability_score = 5.0
            summary = ""
            
            for line in result.split('\n'):
                line = line.strip()
                if line.startswith("Issues:"):
                    issues_str = line.replace("Issues:", "").strip()
                    if issues_str.lower() not in ['none', '']:
                        # Parse issues
                        for issue_str in issues_str.split(','):
                            parts = issue_str.strip().split(':')
                            if len(parts) >= 5:
                                try:
                                    issue_type = CodeQualityIssue(parts[0].strip().lower())
                                except ValueError:
                                    issue_type = CodeQualityIssue.BUG
                                
                                issue = CodeIssue(
                                    issue_type=issue_type,
                                    severity=parts[1].strip(),
                                    description=parts[3].strip(),
                                    line_number=int(parts[2].strip()) if parts[2].strip().isdigit() else None,
                                    suggestion=parts[4].strip() if len(parts) > 4 else None
                                )
                                issues.append(issue)
                
                elif line.startswith("Complexity:"):
                    try:
                        complexity_score = float(re.search(r'\d+', line).group())
                    except:
                        pass
                
                elif line.startswith("Maintainability:"):
                    try:
                        maintainability_score = float(re.search(r'\d+', line).group())
                    except:
                        pass
                
                elif line.startswith("Summary:"):
                    summary = line.replace("Summary:", "").strip()
            
            if not summary:
                summary = "Code analysis completed."
            
            return CodeAnalysis(
                snippet=snippet,
                issues=issues,
                complexity_score=complexity_score,
                maintainability_score=maintainability_score,
                summary=summary
            )
            
        except Exception as e:
            print(f"Error analyzing code: {e}")
            return CodeAnalysis(
                snippet=snippet,
                issues=[],
                complexity_score=5.0,
                maintainability_score=5.0,
                summary="Analysis failed."
            )
    
    def generate_code(
        self,
        description: str,
        language: Optional[CodeLanguage] = None,
        include_tests: bool = False
    ) -> GeneratedCode:
        """
        Generate code from description.
        
        Args:
            description: What the code should do
            language: Target programming language
            include_tests: Whether to generate tests
            
        Returns:
            Generated code with explanation
        """
        language = language or self.default_language
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert programmer. Generate clean, efficient, well-documented code.

Follow best practices:
- Clear naming conventions
- Proper error handling
- Type hints (where applicable)
- Docstrings/comments
- Modular design

Provide:
Code: [complete implementation]
Explanation: [how it works]
Dependencies: [required libraries, comma-separated]
{test_instruction}"""),
            ("user", """Language: {language}
Task: {description}

Generate the code:""")
        ])
        
        test_instruction = ""
        if include_tests:
            test_instruction = "Tests: [unit tests for the code]"
        
        chain = prompt | self.generator | self.parser
        
        try:
            result = chain.invoke({
                "language": language.value,
                "description": description,
                "test_instruction": test_instruction
            })
            
            # Parse result
            code = ""
            explanation = ""
            test_code = None
            dependencies = []
            
            current_section = None
            section_lines = []
            
            for line in result.split('\n'):
                if line.startswith("Code:"):
                    if current_section:
                        self._save_section(current_section, section_lines, 
                                         locals())
                    current_section = "code"
                    section_lines = []
                elif line.startswith("Explanation:"):
                    if current_section:
                        self._save_section(current_section, section_lines,
                                         {"code": code})
                        code = '\n'.join(section_lines)
                    current_section = "explanation"
                    section_lines = [line.replace("Explanation:", "").strip()]
                elif line.startswith("Dependencies:"):
                    if current_section == "explanation":
                        explanation = '\n'.join(section_lines)
                    deps_str = line.replace("Dependencies:", "").strip()
                    if deps_str.lower() not in ['none', '']:
                        dependencies = [d.strip() for d in deps_str.split(',')]
                    current_section = "dependencies"
                    section_lines = []
                elif line.startswith("Tests:"):
                    if current_section == "code":
                        code = '\n'.join(section_lines)
                    current_section = "tests"
                    section_lines = []
                else:
                    section_lines.append(line)
            
            # Save last section
            if current_section == "code":
                code = '\n'.join(section_lines)
            elif current_section == "explanation":
                explanation = '\n'.join(section_lines)
            elif current_section == "tests":
                test_code = '\n'.join(section_lines)
            
            # Clean up code (remove markdown formatting)
            code = self._clean_code_block(code)
            if test_code:
                test_code = self._clean_code_block(test_code)
            
            if not explanation:
                explanation = "Generated code based on description."
            
            return GeneratedCode(
                code=code,
                language=language,
                explanation=explanation,
                test_code=test_code,
                dependencies=dependencies
            )
            
        except Exception as e:
            print(f"Error generating code: {e}")
            return GeneratedCode(
                code="# Error generating code",
                language=language,
                explanation="Code generation failed."
            )
    
    def _clean_code_block(self, code: str) -> str:
        """Remove markdown code block formatting."""
        # Remove ```language and ``` markers
        code = re.sub(r'^```\w*\n', '', code)
        code = re.sub(r'\n```$', '', code)
        code = code.strip()
        return code
    
    def _save_section(self, section: str, lines: List[str], context: Dict):
        """Helper to save parsed sections."""
        content = '\n'.join(lines)
        if section in context:
            context[section] = content
    
    def refactor_code(self, snippet: CodeSnippet, focus: str = "general") -> RefactoredCode:
        """
        Refactor code for better quality.
        
        Args:
            snippet: Code to refactor
            focus: Focus area (general, performance, readability, etc.)
            
        Returns:
            Refactored code with explanation
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a refactoring expert. Improve the code while preserving functionality.

Focus on:
- {focus}
- Code clarity and readability
- Best practices and patterns
- Performance optimization
- Maintainability

Provide:
Refactored: [improved code]
Changes: [list of changes made, comma-separated]
Improvements: [benefits of refactoring, comma-separated]"""),
            ("user", """Language: {language}
Original Code:
```
{code}
```

Refactor this code:""")
        ])
        
        chain = prompt | self.refactorer | self.parser
        
        try:
            result = chain.invoke({
                "language": snippet.language.value,
                "code": snippet.code,
                "focus": focus
            })
            
            # Parse result
            refactored = ""
            changes = []
            improvements = []
            
            for line in result.split('\n'):
                if line.startswith("Refactored:"):
                    # Start capturing refactored code
                    refactored_lines = []
                    continue
                elif line.startswith("Changes:"):
                    changes_str = line.replace("Changes:", "").strip()
                    if changes_str.lower() not in ['none', '']:
                        changes = [c.strip() for c in changes_str.split(',')]
                elif line.startswith("Improvements:"):
                    imp_str = line.replace("Improvements:", "").strip()
                    if imp_str.lower() not in ['none', '']:
                        improvements = [i.strip() for i in imp_str.split(',')]
                elif refactored == "" and not line.startswith("Changes") and not line.startswith("Improvements"):
                    refactored += line + '\n'
            
            refactored = self._clean_code_block(refactored)
            
            if not changes:
                changes = ["Code refactored"]
            if not improvements:
                improvements = ["Improved code quality"]
            
            return RefactoredCode(
                original=snippet.code,
                refactored=refactored if refactored else snippet.code,
                changes=changes,
                improvements=improvements
            )
            
        except Exception as e:
            print(f"Error refactoring code: {e}")
            return RefactoredCode(
                original=snippet.code,
                refactored=snippet.code,
                changes=["Refactoring failed"],
                improvements=[]
            )
    
    def explain_code(self, snippet: CodeSnippet, detail_level: str = "medium") -> str:
        """
        Explain what code does.
        
        Args:
            snippet: Code to explain
            detail_level: Level of detail (brief, medium, detailed)
            
        Returns:
            Explanation of the code
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a code educator. Explain code clearly and comprehensively.

Detail Level: {detail_level}

For brief: High-level summary in 2-3 sentences
For medium: Explain main components and flow
For detailed: Line-by-line or block-by-block explanation

Provide clear, educational explanation."""),
            ("user", """Language: {language}
Code:
```
{code}
```

Explain this code:""")
        ])
        
        chain = prompt | self.explainer | self.parser
        
        try:
            result = chain.invoke({
                "language": snippet.language.value,
                "code": snippet.code,
                "detail_level": detail_level
            })
            
            return result.strip()
            
        except Exception as e:
            print(f"Error explaining code: {e}")
            return "Unable to generate explanation."
    
    def generate_tests(self, snippet: CodeSnippet) -> str:
        """
        Generate unit tests for code.
        
        Args:
            snippet: Code to test
            
        Returns:
            Test code
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a testing expert. Generate comprehensive unit tests.

Include:
- Test for normal cases
- Edge cases
- Error handling
- Different input types

Use appropriate testing framework for the language."""),
            ("user", """Language: {language}
Code to Test:
```
{code}
```

Generate unit tests:""")
        ])
        
        chain = prompt | self.generator | self.parser
        
        try:
            result = chain.invoke({
                "language": snippet.language.value,
                "code": snippet.code
            })
            
            return self._clean_code_block(result)
            
        except Exception as e:
            print(f"Error generating tests: {e}")
            return "# Error generating tests"


def demonstrate_code_agent():
    """Demonstrate Code Agent pattern."""
    
    print("="*80)
    print("CODE AGENT PATTERNS - DEMONSTRATION")
    print("="*80)
    
    agent = CodeAgent(default_language=CodeLanguage.PYTHON)
    
    # Test 1: Code Analysis
    print("\n" + "="*80)
    print("TEST 1: Code Analysis")
    print("="*80)
    
    sample_code = """
def calculate_average(numbers):
    sum = 0
    for num in numbers:
        sum = sum + num
    return sum / len(numbers)
"""
    
    snippet1 = CodeSnippet(code=sample_code, language=CodeLanguage.PYTHON)
    analysis = agent.analyze_code(snippet1)
    
    print(f"\nComplexity Score: {analysis.complexity_score}/10")
    print(f"Maintainability Score: {analysis.maintainability_score}/10")
    print(f"\nSummary: {analysis.summary}")
    
    if analysis.issues:
        print(f"\nIssues Found: {len(analysis.issues)}")
        for issue in analysis.issues[:3]:
            print(f"  • {issue.issue_type.value.upper()} ({issue.severity}): {issue.description}")
            if issue.suggestion:
                print(f"    Suggestion: {issue.suggestion}")
    
    # Test 2: Code Generation
    print("\n" + "="*80)
    print("TEST 2: Code Generation")
    print("="*80)
    
    generated = agent.generate_code(
        description="Create a function to validate email addresses using regex",
        language=CodeLanguage.PYTHON,
        include_tests=True
    )
    
    print("\n--- Generated Code ---")
    print(generated.code[:300] + "..." if len(generated.code) > 300 else generated.code)
    
    print(f"\n--- Explanation ---")
    print(generated.explanation[:200] + "..." if len(generated.explanation) > 200 else generated.explanation)
    
    if generated.dependencies:
        print(f"\n--- Dependencies ---")
        print(", ".join(generated.dependencies))
    
    # Test 3: Code Refactoring
    print("\n" + "="*80)
    print("TEST 3: Code Refactoring")
    print("="*80)
    
    messy_code = """
def process(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
    return result
"""
    
    snippet2 = CodeSnippet(code=messy_code, language=CodeLanguage.PYTHON)
    refactored = agent.refactor_code(snippet2, focus="readability")
    
    print("\n--- Original Code ---")
    print(refactored.original)
    
    print("\n--- Refactored Code ---")
    print(refactored.refactored)
    
    print("\n--- Changes Made ---")
    for change in refactored.changes:
        print(f"  • {change}")
    
    print("\n--- Improvements ---")
    for improvement in refactored.improvements:
        print(f"  • {improvement}")
    
    # Test 4: Code Explanation
    print("\n" + "="*80)
    print("TEST 4: Code Explanation")
    print("="*80)
    
    complex_code = """
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 2:
        return 1
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]
"""
    
    snippet3 = CodeSnippet(code=complex_code, language=CodeLanguage.PYTHON)
    explanation = agent.explain_code(snippet3, detail_level="medium")
    
    print("\n--- Code ---")
    print(complex_code)
    
    print("\n--- Explanation ---")
    print(explanation)
    
    # Test 5: Test Generation
    print("\n" + "="*80)
    print("TEST 5: Test Generation")
    print("="*80)
    
    function_code = """
def is_palindrome(text):
    cleaned = ''.join(c.lower() for c in text if c.isalnum())
    return cleaned == cleaned[::-1]
"""
    
    snippet4 = CodeSnippet(code=function_code, language=CodeLanguage.PYTHON)
    tests = agent.generate_tests(snippet4)
    
    print("\n--- Function to Test ---")
    print(function_code)
    
    print("\n--- Generated Tests ---")
    print(tests[:400] + "..." if len(tests) > 400 else tests)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Code Agent Patterns")
    print("="*80)
    print("""
The Code Agent pattern demonstrates specialized AI assistance for software development:

Key Features Demonstrated:
1. Code Analysis: Identifies bugs, security issues, performance problems
2. Code Generation: Creates functions from natural language descriptions
3. Code Refactoring: Improves code structure and readability
4. Code Explanation: Provides clear educational explanations
5. Test Generation: Creates comprehensive unit tests
6. Quality Metrics: Assesses complexity and maintainability
7. Multi-language Support: Works across programming languages

Benefits:
• Accelerates development workflow significantly
• Improves code quality and consistency
• Reduces bugs and technical debt
• Provides learning opportunities for developers
• Automates repetitive coding tasks
• Maintains coding standards automatically
• Assists with code reviews and documentation

Use Cases:
• Automated code review in CI/CD pipelines
• Intelligent IDE code completion and suggestions
• Refactoring legacy code for maintainability
• Bug detection and debugging assistance
• Test generation for improving coverage
• Documentation generation and maintenance
• Code translation between languages
• Learning and education for new developers

Production Considerations:
• Implement actual static analysis tools integration
• Add support for project-wide context and dependencies
• Implement incremental analysis for large codebases
• Add caching for repeated analysis tasks
• Integrate with version control for change analysis
• Implement security scanning and vulnerability detection
• Add performance profiling and optimization suggestions
• Support for framework-specific best practices
• Integration with existing dev tools and IDEs
• Human review for critical code changes
• Metrics tracking for code quality improvements
• Team-specific style guide enforcement

Advanced Extensions:
• Multi-file refactoring with dependency tracking
• Architectural pattern detection and suggestions
• Automated bug fixing with verification
• Code clone detection and deduplication
• Performance optimization through profiling
• Security vulnerability patching
• API usage pattern analysis
• Code smell detection and remediation
    """)


if __name__ == "__main__":
    demonstrate_code_agent()
