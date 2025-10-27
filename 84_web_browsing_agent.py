"""
Web Browsing Agent Pattern

Enables agents to navigate, interact with, and extract information from web pages
through automated browsing capabilities.

Key Concepts:
- Web navigation
- Element interaction
- Content extraction
- Form filling
- Session management

Use Cases:
- Web scraping
- Automated testing
- Data collection
- Web automation
- Information retrieval
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re
from urllib.parse import urljoin, urlparse


class ActionType(Enum):
    """Types of web browsing actions."""
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    SELECT = "select"
    EXTRACT = "extract"
    SUBMIT = "submit"
    WAIT = "wait"


class ElementType(Enum):
    """HTML element types."""
    LINK = "link"
    BUTTON = "button"
    INPUT = "input"
    TEXTAREA = "textarea"
    SELECT = "select"
    IMAGE = "image"
    DIV = "div"
    SPAN = "span"
    TABLE = "table"


@dataclass
class WebElement:
    """Represents a web page element."""
    element_id: str
    element_type: ElementType
    tag: str
    text: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    children: List['WebElement'] = field(default_factory=list)
    
    def get_attribute(self, name: str) -> Optional[str]:
        """Get element attribute value."""
        return self.attributes.get(name)
    
    def has_text(self, text: str, case_sensitive: bool = False) -> bool:
        """Check if element contains text."""
        if case_sensitive:
            return text in self.text
        return text.lower() in self.text.lower()
    
    def matches_selector(self, selector: str) -> bool:
        """Simple selector matching."""
        # Simplified selector matching (id, class, tag)
        if selector.startswith('#'):
            return self.attributes.get('id') == selector[1:]
        elif selector.startswith('.'):
            classes = self.attributes.get('class', '').split()
            return selector[1:] in classes
        else:
            return self.tag == selector


@dataclass
class WebPage:
    """Represents a web page."""
    url: str
    title: str
    content: str
    elements: List[WebElement] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    forms: List[Dict[str, Any]] = field(default_factory=list)
    loaded_at: datetime = field(default_factory=datetime.now)
    
    def find_element(self, selector: str) -> Optional[WebElement]:
        """Find first element matching selector."""
        for element in self.elements:
            if element.matches_selector(selector):
                return element
        return None
    
    def find_elements(self, selector: str) -> List[WebElement]:
        """Find all elements matching selector."""
        return [e for e in self.elements if e.matches_selector(selector)]
    
    def find_by_text(self, text: str, element_type: Optional[ElementType] = None) -> List[WebElement]:
        """Find elements containing text."""
        matches = []
        for element in self.elements:
            if element_type and element.element_type != element_type:
                continue
            if element.has_text(text):
                matches.append(element)
        return matches
    
    def extract_text(self, selector: Optional[str] = None) -> str:
        """Extract text from page or specific elements."""
        if selector:
            elements = self.find_elements(selector)
            return " ".join(e.text for e in elements)
        return self.content


@dataclass
class BrowsingAction:
    """Represents a browsing action."""
    action_type: ActionType
    target: Optional[str] = None  # URL, selector, or element ID
    value: Optional[str] = None  # For type, select actions
    timestamp: datetime = field(default_factory=datetime.now)
    result: Optional[str] = None


@dataclass
class ExtractionRule:
    """Rule for extracting data from pages."""
    name: str
    selector: str
    attribute: Optional[str] = None  # Extract attribute instead of text
    pattern: Optional[str] = None  # Regex pattern for extraction
    multiple: bool = False  # Extract multiple matches


class BrowserSession:
    """Manages a browsing session."""
    
    def __init__(self):
        self.current_page: Optional[WebPage] = None
        self.history: List[WebPage] = []
        self.cookies: Dict[str, str] = {}
        self.session_id = f"session_{datetime.now().timestamp()}"
    
    def navigate_to(self, url: str) -> WebPage:
        """Navigate to a URL."""
        # Simulate page load
        page = self._load_page(url)
        
        if self.current_page:
            self.history.append(self.current_page)
        
        self.current_page = page
        return page
    
    def go_back(self) -> Optional[WebPage]:
        """Go back to previous page."""
        if not self.history:
            return None
        
        self.current_page = self.history.pop()
        return self.current_page
    
    def refresh(self) -> WebPage:
        """Refresh current page."""
        if not self.current_page:
            raise ValueError("No current page to refresh")
        
        return self.navigate_to(self.current_page.url)
    
    def _load_page(self, url: str) -> WebPage:
        """Simulate loading a web page."""
        # This is a mock implementation
        # In practice, this would fetch actual HTML
        
        domain = urlparse(url).netloc
        
        # Create mock page based on URL
        if "github.com" in domain:
            return self._create_github_page(url)
        elif "wikipedia.org" in domain:
            return self._create_wikipedia_page(url)
        else:
            return self._create_generic_page(url)
    
    def _create_github_page(self, url: str) -> WebPage:
        """Create mock GitHub page."""
        elements = [
            WebElement("nav1", ElementType.LINK, "a", "Pull requests",
                      {"href": "/pulls", "class": "nav-link"}),
            WebElement("nav2", ElementType.LINK, "a", "Issues",
                      {"href": "/issues", "class": "nav-link"}),
            WebElement("repo-title", ElementType.DIV, "div", "awesome-project",
                      {"class": "repo-title"}),
            WebElement("star-button", ElementType.BUTTON, "button", "Star",
                      {"class": "btn-star"}),
            WebElement("readme", ElementType.DIV, "div",
                      "This is an awesome project for demonstrating web browsing agents.",
                      {"id": "readme", "class": "markdown-body"}),
        ]
        
        return WebPage(
            url=url,
            title="awesome-project - GitHub",
            content="GitHub repository page content",
            elements=elements,
            links=["/pulls", "/issues", "/stargazers"]
        )
    
    def _create_wikipedia_page(self, url: str) -> WebPage:
        """Create mock Wikipedia page."""
        elements = [
            WebElement("title", ElementType.DIV, "h1", "Artificial Intelligence",
                      {"id": "firstHeading", "class": "title"}),
            WebElement("toc", ElementType.DIV, "div", "Contents",
                      {"id": "toc", "class": "toc"}),
            WebElement("content1", ElementType.DIV, "p",
                      "Artificial intelligence (AI) is intelligence demonstrated by machines.",
                      {"class": "content"}),
            WebElement("content2", ElementType.DIV, "p",
                      "AI applications include advanced web search, recommendation systems, and autonomous vehicles.",
                      {"class": "content"}),
            WebElement("edit-link", ElementType.LINK, "a", "Edit",
                      {"href": "/edit", "class": "edit-page"}),
        ]
        
        return WebPage(
            url=url,
            title="Artificial Intelligence - Wikipedia",
            content="Wikipedia article about AI",
            elements=elements,
            links=["/wiki/Machine_learning", "/wiki/Deep_learning"]
        )
    
    def _create_generic_page(self, url: str) -> WebPage:
        """Create mock generic page."""
        elements = [
            WebElement("header", ElementType.DIV, "header", "Welcome",
                      {"class": "header"}),
            WebElement("search-input", ElementType.INPUT, "input", "",
                      {"id": "search", "type": "text", "placeholder": "Search..."}),
            WebElement("submit-btn", ElementType.BUTTON, "button", "Submit",
                      {"class": "btn-primary"}),
        ]
        
        return WebPage(
            url=url,
            title="Generic Web Page",
            content="Generic page content",
            elements=elements,
            links=["/about", "/contact"]
        )


class WebBrowsingAgent:
    """Agent capable of browsing and interacting with web pages."""
    
    def __init__(self, name: str):
        self.name = name
        self.session = BrowserSession()
        self.action_history: List[BrowsingAction] = []
        self.extracted_data: Dict[str, Any] = {}
    
    def navigate(self, url: str) -> WebPage:
        """Navigate to a URL."""
        print(f"[{self.name}] Navigating to: {url}")
        
        action = BrowsingAction(ActionType.NAVIGATE, target=url)
        page = self.session.navigate_to(url)
        action.result = f"Loaded: {page.title}"
        
        self.action_history.append(action)
        print(f"  ✓ Loaded: {page.title}")
        
        return page
    
    def click(self, selector: str) -> bool:
        """Click an element."""
        if not self.session.current_page:
            return False
        
        print(f"[{self.name}] Clicking: {selector}")
        
        element = self.session.current_page.find_element(selector)
        if not element:
            print(f"  ✗ Element not found: {selector}")
            return False
        
        action = BrowsingAction(ActionType.CLICK, target=selector)
        
        # Simulate click behavior
        if element.element_type == ElementType.LINK:
            href = element.get_attribute('href')
            if href:
                # Navigate to link
                base_url = self.session.current_page.url
                full_url = urljoin(base_url, href)
                self.navigate(full_url)
        
        action.result = f"Clicked: {element.text}"
        self.action_history.append(action)
        print(f"  ✓ Clicked: {element.text}")
        
        return True
    
    def type_text(self, selector: str, text: str) -> bool:
        """Type text into an input field."""
        if not self.session.current_page:
            return False
        
        print(f"[{self.name}] Typing into {selector}: '{text}'")
        
        element = self.session.current_page.find_element(selector)
        if not element:
            print(f"  ✗ Element not found: {selector}")
            return False
        
        if element.element_type not in [ElementType.INPUT, ElementType.TEXTAREA]:
            print(f"  ✗ Element is not an input field")
            return False
        
        action = BrowsingAction(ActionType.TYPE, target=selector, value=text)
        element.text = text  # Update element text
        action.result = "Text entered"
        
        self.action_history.append(action)
        print(f"  ✓ Text entered")
        
        return True
    
    def extract_text(self, selector: str, name: Optional[str] = None) -> Optional[str]:
        """Extract text from element(s)."""
        if not self.session.current_page:
            return None
        
        print(f"[{self.name}] Extracting text from: {selector}")
        
        text = self.session.current_page.extract_text(selector)
        
        action = BrowsingAction(ActionType.EXTRACT, target=selector)
        action.result = f"Extracted {len(text)} characters"
        self.action_history.append(action)
        
        if name:
            self.extracted_data[name] = text
        
        print(f"  ✓ Extracted: {text[:50]}...")
        
        return text
    
    def extract_with_rules(self, rules: List[ExtractionRule]) -> Dict[str, Any]:
        """Extract data using multiple rules."""
        if not self.session.current_page:
            return {}
        
        print(f"[{self.name}] Extracting data with {len(rules)} rules")
        
        results = {}
        
        for rule in rules:
            elements = self.session.current_page.find_elements(rule.selector)
            
            if rule.multiple:
                values = []
                for element in elements:
                    value = self._extract_value(element, rule)
                    if value:
                        values.append(value)
                results[rule.name] = values
            else:
                if elements:
                    value = self._extract_value(elements[0], rule)
                    results[rule.name] = value
        
        self.extracted_data.update(results)
        print(f"  ✓ Extracted {len(results)} fields")
        
        return results
    
    def _extract_value(self, element: WebElement, rule: ExtractionRule) -> Any:
        """Extract value from element based on rule."""
        if rule.attribute:
            value = element.get_attribute(rule.attribute)
        else:
            value = element.text
        
        if value and rule.pattern:
            match = re.search(rule.pattern, value)
            if match:
                value = match.group(1) if match.groups() else match.group(0)
        
        return value
    
    def find_and_click(self, text: str, element_type: Optional[ElementType] = None) -> bool:
        """Find element by text and click it."""
        if not self.session.current_page:
            return False
        
        elements = self.session.current_page.find_by_text(text, element_type)
        
        if not elements:
            print(f"[{self.name}] No element found with text: {text}")
            return False
        
        element = elements[0]
        selector = f"#{element.element_id}"
        return self.click(selector)
    
    def get_all_links(self) -> List[str]:
        """Get all links on current page."""
        if not self.session.current_page:
            return []
        
        links = []
        for element in self.session.current_page.elements:
            if element.element_type == ElementType.LINK:
                href = element.get_attribute('href')
                if href:
                    links.append(href)
        
        return links
    
    def get_summary(self) -> Dict[str, Any]:
        """Get browsing session summary."""
        return {
            "agent": self.name,
            "current_url": self.session.current_page.url if self.session.current_page else None,
            "current_title": self.session.current_page.title if self.session.current_page else None,
            "actions_performed": len(self.action_history),
            "pages_visited": len(self.session.history) + (1 if self.session.current_page else 0),
            "data_extracted": len(self.extracted_data)
        }


def demonstrate_web_browsing():
    """Demonstrate web browsing agent."""
    print("=" * 60)
    print("WEB BROWSING AGENT DEMONSTRATION")
    print("=" * 60)
    
    # Create agent
    agent = WebBrowsingAgent("WebBot")
    
    # Demonstration 1: Navigate and extract from GitHub
    print("\n" + "=" * 60)
    print("1. Navigating GitHub Repository")
    print("=" * 60)
    
    page = agent.navigate("https://github.com/user/awesome-project")
    
    # Extract repository information
    print("\nExtracting repository information...")
    rules = [
        ExtractionRule("repo_name", ".repo-title", multiple=False),
        ExtractionRule("readme", "#readme", multiple=False),
        ExtractionRule("navigation", ".nav-link", multiple=True)
    ]
    
    data = agent.extract_with_rules(rules)
    print("\nExtracted data:")
    for key, value in data.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {str(value)[:50]}...")
    
    # Click on Issues link
    print("\nClicking on Issues link...")
    agent.find_and_click("Issues", ElementType.LINK)
    
    # Demonstration 2: Browse Wikipedia
    print("\n" + "=" * 60)
    print("2. Browsing Wikipedia Article")
    print("=" * 60)
    
    page = agent.navigate("https://en.wikipedia.org/wiki/Artificial_Intelligence")
    
    # Extract article content
    title = agent.extract_text(".title", "article_title")
    content = agent.extract_text(".content", "article_content")
    
    print(f"\nArticle title: {agent.extracted_data.get('article_title', 'N/A')}")
    print(f"Content preview: {agent.extracted_data.get('article_content', 'N/A')[:100]}...")
    
    # Get all links
    links = agent.get_all_links()
    print(f"\nFound {len(links)} links on page")
    
    # Demonstration 3: Form interaction
    print("\n" + "=" * 60)
    print("3. Form Interaction")
    print("=" * 60)
    
    page = agent.navigate("https://example.com/search")
    
    # Fill out search form
    agent.type_text("#search", "agentic AI patterns")
    agent.find_and_click("Submit", ElementType.BUTTON)
    
    # Session Summary
    print("\n" + "=" * 60)
    print("Browsing Session Summary")
    print("=" * 60)
    
    summary = agent.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\nAction History:")
    for i, action in enumerate(agent.action_history[-5:], 1):
        print(f"  {i}. {action.action_type.value}: {action.target or 'N/A'}")
        if action.result:
            print(f"     Result: {action.result}")


if __name__ == "__main__":
    demonstrate_web_browsing()
