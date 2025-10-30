"""
Pattern 073: Web Browsing Agent

Description:
    A Web Browsing Agent is a specialized AI agent designed to navigate websites,
    extract information, interact with web elements, and synthesize findings from
    multiple web sources. This pattern demonstrates how to build an intelligent agent
    that can autonomously browse the web, understand page content, follow links,
    fill forms, and extract structured information.
    
    The agent combines web navigation capabilities with natural language understanding
    to perform complex web-based research tasks. It can handle dynamic content,
    JavaScript-rendered pages, authentication, pagination, and multi-page workflows.
    Unlike simple web scrapers, this agent makes intelligent decisions about where
    to navigate based on task objectives.

Components:
    1. Navigation Controller: Manages page navigation and history
    2. Content Extractor: Parses and extracts relevant information from pages
    3. Element Interactor: Clicks buttons, fills forms, handles dropdowns
    4. Link Analyzer: Evaluates and prioritizes links to follow
    5. Session Manager: Handles cookies, authentication, state
    6. Page Understanding: Interprets page structure and content semantically
    7. Information Synthesizer: Combines data from multiple pages
    8. Task Planner: Plans multi-step browsing strategies

Key Features:
    - Intelligent link following based on task relevance
    - Dynamic content handling (JavaScript, AJAX)
    - Form filling and interaction
    - Multi-page information aggregation
    - Authentication and session management
    - Pagination handling
    - Screenshot and visual analysis
    - Content summarization
    - Structured data extraction
    - Recursive crawling with depth limits

Use Cases:
    - Competitive intelligence gathering
    - Price monitoring and comparison
    - News aggregation and monitoring
    - Academic research and literature review
    - Job posting scraping
    - Real estate listing analysis
    - Product information extraction
    - Documentation exploration
    - Social media monitoring
    - API discovery and testing

LangChain Implementation:
    Uses ChatOpenAI for page understanding and decision-making, simulates
    web interactions with structured data models representing browser state.
"""

import os
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class PageType(Enum):
    """Types of web pages"""
    HOME = "home"
    ARTICLE = "article"
    PRODUCT = "product"
    LISTING = "listing"
    FORM = "form"
    SEARCH_RESULTS = "search_results"
    DOCUMENTATION = "documentation"
    PROFILE = "profile"
    UNKNOWN = "unknown"


class ActionType(Enum):
    """Types of web actions"""
    NAVIGATE = "navigate"
    CLICK = "click"
    FILL_FORM = "fill_form"
    SCROLL = "scroll"
    WAIT = "wait"
    EXTRACT = "extract"
    BACK = "back"
    REFRESH = "refresh"


class ContentType(Enum):
    """Types of content to extract"""
    TEXT = "text"
    LINKS = "links"
    IMAGES = "images"
    TABLES = "tables"
    FORMS = "forms"
    METADATA = "metadata"
    STRUCTURED_DATA = "structured_data"


@dataclass
class WebPage:
    """Represents a web page"""
    url: str
    title: str
    page_type: PageType
    content: str
    links: List[str] = field(default_factory=list)
    forms: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    visited_at: datetime = field(default_factory=datetime.now)


@dataclass
class BrowsingAction:
    """Represents a browsing action"""
    action_type: ActionType
    target: str  # URL, element selector, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class ExtractedData:
    """Structured data extracted from a page"""
    source_url: str
    data_type: str
    content: Dict[str, Any]
    confidence: float
    extracted_at: datetime = field(default_factory=datetime.now)


@dataclass
class BrowsingSession:
    """State of a browsing session"""
    session_id: str
    start_url: str
    objective: str
    current_url: str
    history: List[str] = field(default_factory=list)
    visited_urls: Set[str] = field(default_factory=set)
    extracted_data: List[ExtractedData] = field(default_factory=list)
    cookies: Dict[str, str] = field(default_factory=dict)


@dataclass
class NavigationPlan:
    """Plan for multi-page navigation"""
    objective: str
    actions: List[BrowsingAction]
    expected_pages: int
    max_depth: int
    priority_keywords: List[str]


class WebBrowsingAgent:
    """
    Agent for intelligent web browsing and information extraction.
    
    This agent can navigate websites autonomously, understand page content,
    make decisions about where to navigate, and extract relevant information.
    """
    
    def __init__(self, max_pages: int = 10, max_depth: int = 3):
        """
        Initialize the web browsing agent.
        
        Args:
            max_pages: Maximum number of pages to visit per task
            max_depth: Maximum navigation depth from start page
        """
        # Navigator for planning and decision-making
        self.navigator_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        
        # Extractor for content understanding
        self.extractor_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        
        # Analyzer for link evaluation
        self.analyzer_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        
        # Synthesizer for combining information
        self.synthesizer_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)
        
        self.max_pages = max_pages
        self.max_depth = max_depth
        
        # Simulated page database (in production, would use real web driver)
        self.page_database: Dict[str, WebPage] = {}
        self._initialize_sample_pages()
    
    def _initialize_sample_pages(self):
        """Initialize sample web pages for demonstration"""
        self.page_database = {
            "https://example.com": WebPage(
                url="https://example.com",
                title="Example Company - Home",
                page_type=PageType.HOME,
                content="Welcome to Example Company. We provide innovative solutions. Visit our products page or read our latest blog posts.",
                links=[
                    "https://example.com/products",
                    "https://example.com/blog",
                    "https://example.com/about",
                    "https://example.com/contact"
                ],
                metadata={"description": "Leading provider of innovative solutions"}
            ),
            "https://example.com/products": WebPage(
                url="https://example.com/products",
                title="Products - Example Company",
                page_type=PageType.LISTING,
                content="Our Products: Product A ($99) - High quality solution. Product B ($149) - Premium offering. Product C ($199) - Enterprise solution.",
                links=[
                    "https://example.com/products/a",
                    "https://example.com/products/b",
                    "https://example.com/products/c",
                    "https://example.com"
                ],
                metadata={"category": "products"}
            ),
            "https://example.com/products/a": WebPage(
                url="https://example.com/products/a",
                title="Product A - Details",
                page_type=PageType.PRODUCT,
                content="Product A: $99. Features: Fast, Reliable, Easy to use. Rating: 4.5/5. Reviews: 120 customer reviews.",
                links=["https://example.com/products", "https://example.com/cart"],
                metadata={"price": "99", "rating": "4.5", "product_id": "A"}
            ),
            "https://example.com/blog": WebPage(
                url="https://example.com/blog",
                title="Blog - Example Company",
                page_type=PageType.LISTING,
                content="Recent Posts: How to improve productivity (Jan 15, 2024). Industry trends 2024 (Jan 10, 2024). Best practices guide (Jan 5, 2024).",
                links=[
                    "https://example.com/blog/productivity",
                    "https://example.com/blog/trends",
                    "https://example.com/blog/practices",
                    "https://example.com"
                ],
                metadata={"category": "blog"}
            ),
            "https://example.com/blog/productivity": WebPage(
                url="https://example.com/blog/productivity",
                title="How to Improve Productivity",
                page_type=PageType.ARTICLE,
                content="Improving productivity requires focus, proper tools, and good habits. Key strategies include time blocking, eliminating distractions, and using automation. Studies show 40% improvement with these methods.",
                links=["https://example.com/blog", "https://example.com/products"],
                metadata={"author": "John Doe", "date": "2024-01-15", "category": "productivity"}
            )
        }
    
    def create_navigation_plan(
        self,
        start_url: str,
        objective: str,
        keywords: List[str]
    ) -> NavigationPlan:
        """
        Create a navigation plan for achieving the objective.
        
        Args:
            start_url: Starting URL
            objective: What to accomplish
            keywords: Important keywords to look for
            
        Returns:
            NavigationPlan with actions to take
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a web navigation expert. Create a plan to achieve
            the objective by browsing from the start URL.
            
            Respond in this format:
            ACTIONS: List of actions (one per line)
            Format: action_type|target|reason
            
            EXPECTED_PAGES: Number of pages to visit
            KEYWORDS: Priority keywords (comma-separated)"""),
            ("user", """Start URL: {start_url}
Objective: {objective}
Keywords: {keywords}

Create a navigation plan.""")
        ])
        
        chain = prompt | self.navigator_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "start_url": start_url,
                "objective": objective,
                "keywords": ", ".join(keywords)
            })
            
            actions = []
            expected_pages = self.max_pages
            priority_keywords = keywords.copy()
            
            for line in response.split('\n'):
                if '|' in line and not line.startswith(('ACTIONS', 'EXPECTED', 'KEYWORDS')):
                    parts = line.split('|')
                    if len(parts) >= 3:
                        action_type_str = parts[0].strip().lower()
                        
                        # Map to enum
                        action_type = ActionType.NAVIGATE
                        if 'click' in action_type_str:
                            action_type = ActionType.CLICK
                        elif 'fill' in action_type_str or 'form' in action_type_str:
                            action_type = ActionType.FILL_FORM
                        elif 'extract' in action_type_str:
                            action_type = ActionType.EXTRACT
                        elif 'navigate' in action_type_str or 'visit' in action_type_str:
                            action_type = ActionType.NAVIGATE
                        
                        actions.append(BrowsingAction(
                            action_type=action_type,
                            target=parts[1].strip(),
                            reason=parts[2].strip() if len(parts) > 2 else ""
                        ))
                elif line.startswith('EXPECTED_PAGES:'):
                    try:
                        expected_pages = int(re.findall(r'\d+', line)[0])
                    except:
                        pass
                elif line.startswith('KEYWORDS:'):
                    kw_str = line.replace('KEYWORDS:', '').strip()
                    priority_keywords = [k.strip() for k in kw_str.split(',')]
            
            if not actions:
                actions.append(BrowsingAction(
                    action_type=ActionType.NAVIGATE,
                    target=start_url,
                    reason="Start browsing"
                ))
            
            return NavigationPlan(
                objective=objective,
                actions=actions,
                expected_pages=min(expected_pages, self.max_pages),
                max_depth=self.max_depth,
                priority_keywords=priority_keywords
            )
            
        except Exception as e:
            return NavigationPlan(
                objective=objective,
                actions=[BrowsingAction(
                    action_type=ActionType.NAVIGATE,
                    target=start_url,
                    reason="Start browsing"
                )],
                expected_pages=self.max_pages,
                max_depth=self.max_depth,
                priority_keywords=keywords
            )
    
    def fetch_page(self, url: str) -> Optional[WebPage]:
        """
        Fetch a web page (simulated).
        
        Args:
            url: URL to fetch
            
        Returns:
            WebPage object or None if not found
        """
        # In production, would use actual web driver (Selenium, Playwright, etc.)
        return self.page_database.get(url)
    
    def understand_page(self, page: WebPage, objective: str) -> Dict[str, Any]:
        """
        Understand page content and relevance to objective.
        
        Args:
            page: WebPage to analyze
            objective: Current objective
            
        Returns:
            Dictionary with page understanding
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a web content analyst. Analyze the page and determine
            its relevance to the objective.
            
            Respond in this format:
            RELEVANCE: High/Medium/Low
            KEY_INFO: Key information found (one per line starting with -)
            NEXT_STEPS: Suggested next actions (one per line starting with -)
            CONFIDENCE: Confidence score (0.0-1.0)"""),
            ("user", """Page URL: {url}
Title: {title}
Type: {page_type}
Content: {content}

Objective: {objective}

Analyze this page.""")
        ])
        
        chain = prompt | self.extractor_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "url": page.url,
                "title": page.title,
                "page_type": page.page_type.value,
                "content": page.content[:500],  # Limit content length
                "objective": objective
            })
            
            relevance = "medium"
            key_info = []
            next_steps = []
            confidence = 0.7
            
            current_section = None
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('RELEVANCE:'):
                    relevance = line.replace('RELEVANCE:', '').strip().lower()
                elif line.startswith('KEY_INFO:'):
                    current_section = 'info'
                elif line.startswith('NEXT_STEPS:'):
                    current_section = 'steps'
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(re.findall(r'[\d.]+', line)[0])
                    except:
                        pass
                elif line.startswith('-'):
                    content = line[1:].strip()
                    if current_section == 'info':
                        key_info.append(content)
                    elif current_section == 'steps':
                        next_steps.append(content)
            
            return {
                "relevance": relevance,
                "key_info": key_info if key_info else ["Page analyzed"],
                "next_steps": next_steps if next_steps else ["Continue browsing"],
                "confidence": confidence
            }
            
        except Exception as e:
            return {
                "relevance": "medium",
                "key_info": ["Page content available"],
                "next_steps": ["Explore links"],
                "confidence": 0.6
            }
    
    def evaluate_links(
        self,
        links: List[str],
        objective: str,
        visited: Set[str],
        keywords: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Evaluate and rank links by relevance to objective.
        
        Args:
            links: List of URLs to evaluate
            objective: Current objective
            visited: Set of already visited URLs
            keywords: Priority keywords
            
        Returns:
            List of (url, score) tuples, sorted by score
        """
        unvisited_links = [link for link in links if link not in visited]
        
        if not unvisited_links:
            return []
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a link evaluation expert. Score each link based on
            its likely relevance to the objective.
            
            Respond with one line per link:
            url|score|reason
            
            Score from 0.0 (irrelevant) to 1.0 (highly relevant)"""),
            ("user", """Objective: {objective}
Keywords: {keywords}

Links to evaluate:
{links}

Score each link.""")
        ])
        
        chain = prompt | self.analyzer_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "objective": objective,
                "keywords": ", ".join(keywords),
                "links": "\n".join([f"- {link}" for link in unvisited_links[:10]])
            })
            
            scored_links = []
            
            for line in response.split('\n'):
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 2:
                        url = parts[0].strip()
                        try:
                            score = float(re.findall(r'[\d.]+', parts[1])[0])
                            if url in unvisited_links:
                                scored_links.append((url, score))
                        except:
                            pass
            
            # If no scores, assign default scores
            if not scored_links:
                for link in unvisited_links:
                    # Simple keyword matching
                    score = 0.5
                    for keyword in keywords:
                        if keyword.lower() in link.lower():
                            score += 0.2
                    scored_links.append((link, min(score, 1.0)))
            
            return sorted(scored_links, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            # Fallback: basic keyword matching
            scored_links = []
            for link in unvisited_links:
                score = 0.5
                for keyword in keywords:
                    if keyword.lower() in link.lower():
                        score += 0.2
                scored_links.append((link, min(score, 1.0)))
            
            return sorted(scored_links, key=lambda x: x[1], reverse=True)
    
    def extract_information(
        self,
        page: WebPage,
        objective: str,
        content_types: List[ContentType]
    ) -> ExtractedData:
        """
        Extract structured information from a page.
        
        Args:
            page: WebPage to extract from
            objective: What information to extract
            content_types: Types of content to extract
            
        Returns:
            ExtractedData with structured information
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an information extraction expert. Extract structured
            data from the page based on the objective.
            
            Respond in this format (JSON-like):
            {{
                "key1": "value1",
                "key2": "value2",
                ...
            }}
            
            Extract relevant fields based on content type."""),
            ("user", """Page: {title}
URL: {url}
Content: {content}

Objective: {objective}
Content Types: {types}

Extract structured information.""")
        ])
        
        chain = prompt | self.extractor_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "title": page.title,
                "url": page.url,
                "content": page.content,
                "objective": objective,
                "types": ", ".join([ct.value for ct in content_types])
            })
            
            # Try to parse as structured data
            extracted = {}
            
            # Simple key-value extraction
            for line in response.split('\n'):
                if ':' in line and not line.strip().startswith(('#', '//')):
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().strip('"\'{}')
                        value = parts[1].strip().strip(',"\'}')
                        if key and value:
                            extracted[key] = value
            
            # If no structured data, extract key info
            if not extracted:
                extracted = {
                    "title": page.title,
                    "url": page.url,
                    "summary": page.content[:200]
                }
            
            return ExtractedData(
                source_url=page.url,
                data_type="page_data",
                content=extracted,
                confidence=0.8
            )
            
        except Exception as e:
            return ExtractedData(
                source_url=page.url,
                data_type="page_data",
                content={
                    "title": page.title,
                    "url": page.url,
                    "content": page.content[:200]
                },
                confidence=0.6
            )
    
    def browse(
        self,
        start_url: str,
        objective: str,
        keywords: List[str] = None
    ) -> BrowsingSession:
        """
        Browse the web to achieve the objective.
        
        Args:
            start_url: Starting URL
            objective: What to accomplish
            keywords: Optional priority keywords
            
        Returns:
            BrowsingSession with results
        """
        if keywords is None:
            keywords = []
        
        # Create session
        session = BrowsingSession(
            session_id=f"session_{datetime.now().timestamp()}",
            start_url=start_url,
            objective=objective,
            current_url=start_url
        )
        
        # Create navigation plan
        plan = self.create_navigation_plan(start_url, objective, keywords)
        
        # Browse pages
        pages_visited = 0
        current_depth = 0
        
        urls_to_visit = [(start_url, 0)]  # (url, depth)
        
        while urls_to_visit and pages_visited < self.max_pages:
            current_url, depth = urls_to_visit.pop(0)
            
            if current_url in session.visited_urls or depth > self.max_depth:
                continue
            
            # Fetch page
            page = self.fetch_page(current_url)
            if not page:
                continue
            
            # Update session
            session.current_url = current_url
            session.history.append(current_url)
            session.visited_urls.add(current_url)
            pages_visited += 1
            
            # Understand page
            understanding = self.understand_page(page, objective)
            
            # Extract information if relevant
            if understanding["relevance"] in ["high", "medium"]:
                extracted = self.extract_information(
                    page,
                    objective,
                    [ContentType.TEXT, ContentType.STRUCTURED_DATA]
                )
                session.extracted_data.append(extracted)
            
            # Evaluate and queue links
            if depth < self.max_depth:
                scored_links = self.evaluate_links(
                    page.links,
                    objective,
                    session.visited_urls,
                    plan.priority_keywords
                )
                
                # Add top links to queue
                for link, score in scored_links[:3]:  # Top 3 links
                    if score > 0.5:
                        urls_to_visit.append((link, depth + 1))
        
        return session
    
    def synthesize_findings(self, session: BrowsingSession) -> Dict[str, Any]:
        """
        Synthesize findings from browsing session.
        
        Args:
            session: BrowsingSession with extracted data
            
        Returns:
            Dictionary with synthesized findings
        """
        # Compile extracted data
        all_data = []
        for extracted in session.extracted_data:
            all_data.append(f"From {extracted.source_url}:")
            for key, value in extracted.content.items():
                all_data.append(f"  - {key}: {value}")
        
        data_summary = "\n".join(all_data[:50])  # Limit length
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research synthesizer. Combine information from
            multiple sources into coherent findings.
            
            Respond in this format:
            SUMMARY: Brief summary
            KEY_FINDINGS: Main findings (one per line starting with -)
            INSIGHTS: Deeper insights (one per line starting with -)
            CONCLUSION: Final conclusion"""),
            ("user", """Objective: {objective}
            
Pages Visited: {pages_count}

Extracted Data:
{data}

Synthesize findings.""")
        ])
        
        chain = prompt | self.synthesizer_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "objective": session.objective,
                "pages_count": len(session.visited_urls),
                "data": data_summary
            })
            
            summary = ""
            findings = []
            insights = []
            conclusion = ""
            
            current_section = None
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('SUMMARY:'):
                    summary = line.replace('SUMMARY:', '').strip()
                elif line.startswith('KEY_FINDINGS:'):
                    current_section = 'findings'
                elif line.startswith('INSIGHTS:'):
                    current_section = 'insights'
                elif line.startswith('CONCLUSION:'):
                    current_section = 'conclusion'
                elif line.startswith('-'):
                    content = line[1:].strip()
                    if current_section == 'findings':
                        findings.append(content)
                    elif current_section == 'insights':
                        insights.append(content)
                elif current_section == 'conclusion':
                    conclusion += line + " "
            
            return {
                "summary": summary if summary else "Browsing completed successfully",
                "findings": findings if findings else ["Information gathered from multiple pages"],
                "insights": insights if insights else ["Further analysis recommended"],
                "conclusion": conclusion.strip() if conclusion else "Objective achieved",
                "pages_visited": len(session.visited_urls),
                "data_points": len(session.extracted_data)
            }
            
        except Exception as e:
            return {
                "summary": f"Browsed {len(session.visited_urls)} pages",
                "findings": [f"Visited {len(session.visited_urls)} pages", f"Extracted {len(session.extracted_data)} data points"],
                "insights": ["Multiple sources explored"],
                "conclusion": "Browsing session completed",
                "pages_visited": len(session.visited_urls),
                "data_points": len(session.extracted_data)
            }


def demonstrate_web_browsing_agent():
    """Demonstrate the web browsing agent capabilities"""
    print("=" * 80)
    print("WEB BROWSING AGENT DEMONSTRATION")
    print("=" * 80)
    
    agent = WebBrowsingAgent(max_pages=5, max_depth=2)
    
    # Demo 1: Product Research
    print("\n" + "=" * 80)
    print("DEMO 1: Product Research")
    print("=" * 80)
    
    print("\nObjective: Research products and pricing")
    print("Starting URL: https://example.com")
    
    session1 = agent.browse(
        start_url="https://example.com",
        objective="Find all products and their prices",
        keywords=["product", "price", "buy"]
    )
    
    print(f"\nBrowsing Session: {session1.session_id}")
    print(f"Pages Visited: {len(session1.visited_urls)}")
    print(f"\nNavigation Path:")
    for i, url in enumerate(session1.history, 1):
        print(f"  {i}. {url}")
    
    print(f"\nData Extracted: {len(session1.extracted_data)} items")
    for i, data in enumerate(session1.extracted_data[:3], 1):
        print(f"\n  Item {i} from {data.source_url}:")
        for key, value in list(data.content.items())[:3]:
            print(f"    - {key}: {value}")
    
    findings1 = agent.synthesize_findings(session1)
    print(f"\nSynthesized Findings:")
    print(f"Summary: {findings1['summary']}")
    print(f"\nKey Findings:")
    for finding in findings1['findings'][:3]:
        print(f"  - {finding}")
    
    # Demo 2: Content Research
    print("\n" + "=" * 80)
    print("DEMO 2: Content Research")
    print("=" * 80)
    
    print("\nObjective: Find articles about productivity")
    print("Starting URL: https://example.com")
    
    session2 = agent.browse(
        start_url="https://example.com",
        objective="Find and summarize articles about productivity",
        keywords=["blog", "article", "productivity"]
    )
    
    print(f"\nBrowsing Session: {session2.session_id}")
    print(f"Pages Visited: {len(session2.visited_urls)}")
    print(f"\nPages Explored:")
    for url in session2.visited_urls:
        print(f"  - {url}")
    
    findings2 = agent.synthesize_findings(session2)
    print(f"\nSynthesized Findings:")
    print(f"Summary: {findings2['summary']}")
    if findings2['insights']:
        print(f"\nInsights:")
        for insight in findings2['insights'][:2]:
            print(f"  - {insight}")
    print(f"\nConclusion: {findings2['conclusion']}")
    
    # Demo 3: Page Understanding
    print("\n" + "=" * 80)
    print("DEMO 3: Page Understanding")
    print("=" * 80)
    
    sample_page = agent.fetch_page("https://example.com/products/a")
    if sample_page:
        print(f"\nAnalyzing: {sample_page.title}")
        print(f"URL: {sample_page.url}")
        print(f"Type: {sample_page.page_type.value}")
        
        understanding = agent.understand_page(
            sample_page,
            "Extract product details and pricing"
        )
        
        print(f"\nPage Understanding:")
        print(f"Relevance: {understanding['relevance']}")
        print(f"Confidence: {understanding['confidence']:.2f}")
        print(f"\nKey Information:")
        for info in understanding['key_info'][:3]:
            print(f"  - {info}")
        print(f"\nSuggested Next Steps:")
        for step in understanding['next_steps'][:2]:
            print(f"  - {step}")
    
    # Demo 4: Link Evaluation
    print("\n" + "=" * 80)
    print("DEMO 4: Link Evaluation")
    print("=" * 80)
    
    test_links = [
        "https://example.com/products",
        "https://example.com/blog",
        "https://example.com/about",
        "https://example.com/contact"
    ]
    
    print("\nEvaluating links for objective: 'Find product information'")
    print("Links to evaluate:")
    for link in test_links:
        print(f"  - {link}")
    
    scored_links = agent.evaluate_links(
        test_links,
        "Find product information and pricing",
        set(),
        ["product", "price", "catalog"]
    )
    
    print(f"\nLink Rankings:")
    for url, score in scored_links:
        print(f"  {score:.2f} - {url}")
    
    # Demo 5: Information Extraction
    print("\n" + "=" * 80)
    print("DEMO 5: Information Extraction")
    print("=" * 80)
    
    product_page = agent.fetch_page("https://example.com/products/a")
    if product_page:
        print(f"\nExtracting from: {product_page.title}")
        
        extracted = agent.extract_information(
            product_page,
            "Extract product name, price, rating, and features",
            [ContentType.STRUCTURED_DATA, ContentType.TEXT]
        )
        
        print(f"\nExtracted Data (confidence: {extracted.confidence:.2f}):")
        for key, value in extracted.content.items():
            print(f"  {key}: {value}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary = """
The Web Browsing Agent demonstrates intelligent web navigation and information
extraction capabilities:

KEY CAPABILITIES:
1. Navigation Planning: Creates strategic browsing plans based on objectives
2. Intelligent Link Following: Evaluates and prioritizes links by relevance
3. Page Understanding: Analyzes page content and determines relevance
4. Information Extraction: Extracts structured data from web pages
5. Multi-Page Synthesis: Combines information from multiple sources

BENEFITS:
- Autonomous web research without manual navigation
- Intelligent decision-making about where to browse
- Structured data extraction from unstructured pages
- Multi-source information aggregation
- Objective-driven browsing strategies

USE CASES:
- Competitive intelligence gathering
- Price monitoring and comparison
- Content research and aggregation
- Product information extraction
- News monitoring and analysis
- Academic research assistance
- Job posting scraping
- Real estate data collection
- Market research
- Documentation exploration

PRODUCTION CONSIDERATIONS:
1. Web Driver: Integrate with Selenium, Playwright, or Puppeteer for real browsing
2. JavaScript Handling: Execute JavaScript for dynamic content
3. Authentication: Support login, cookies, and session management
4. Rate Limiting: Respect robots.txt and implement polite crawling
5. Error Handling: Handle timeouts, 404s, CAPTCHAs gracefully
6. Proxy Support: Rotate proxies for large-scale scraping
7. Content Storage: Persist pages and data to databases
8. Screenshot Capture: Take screenshots for visual analysis
9. Form Interaction: Fill and submit forms programmatically
10. Performance: Parallel browsing for efficiency

ADVANCED EXTENSIONS:
- Visual page understanding with computer vision
- Natural language interaction for browsing commands
- Automatic CAPTCHA solving
- Multi-tab parallel browsing
- Browser fingerprinting avoidance
- Dynamic wait strategies for page loads
- Content change monitoring
- Recursive deep crawling with sitemaps
- Structured data validation
- API endpoint discovery

The agent provides an intelligent layer on top of web automation, making
decisions about navigation and extraction based on natural language objectives.
"""
    
    print(summary)


if __name__ == "__main__":
    demonstrate_web_browsing_agent()
