"""
Research Agent Pattern

Enables agents to conduct comprehensive research through literature review,
information synthesis, and citation management.

Key Concepts:
- Literature search and retrieval
- Document analysis
- Information synthesis
- Citation management
- Knowledge organization

Use Cases:
- Academic research
- Market research
- Competitive analysis
- Knowledge discovery
- Report generation
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
import re


class SourceType(Enum):
    """Types of research sources."""
    ACADEMIC_PAPER = "academic_paper"
    BOOK = "book"
    WEB_ARTICLE = "web_article"
    REPORT = "report"
    PATENT = "patent"
    DATASET = "dataset"
    CONFERENCE = "conference"


class ResearchPhase(Enum):
    """Phases of research process."""
    SEARCH = "search"
    RETRIEVE = "retrieve"
    READ = "read"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    CITE = "cite"
    REPORT = "report"


@dataclass
class Citation:
    """Represents a bibliographic citation."""
    citation_id: str
    authors: List[str]
    title: str
    year: int
    source_type: SourceType
    venue: Optional[str] = None  # Journal, conference, etc.
    doi: Optional[str] = None
    url: Optional[str] = None
    
    def format_apa(self) -> str:
        """Format citation in APA style."""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += ", et al."
        
        return f"{authors_str} ({self.year}). {self.title}. {self.venue or 'Source'}."
    
    def format_mla(self) -> str:
        """Format citation in MLA style."""
        if self.authors:
            first_author = self.authors[0]
            return f"{first_author}, et al. \"{self.title}.\" {self.venue}, {self.year}."
        return f"\"{self.title}.\" {self.venue}, {self.year}."


@dataclass
class Document:
    """Represents a research document."""
    doc_id: str
    title: str
    authors: List[str]
    abstract: str
    content: str
    source_type: SourceType
    year: int
    keywords: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)  # References to other docs
    citation_count: int = 0
    relevance_score: float = 0.0
    
    def get_snippet(self, max_length: int = 200) -> str:
        """Get content snippet."""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    def extract_key_concepts(self) -> List[str]:
        """Extract key concepts from document."""
        # Simple keyword extraction (in practice, use NLP)
        words = re.findall(r'\b[a-z]{4,}\b', self.content.lower())
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # Return top concepts
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:10]]


@dataclass
class ResearchQuery:
    """Represents a research query."""
    query_text: str
    keywords: List[str]
    date_range: Optional[Tuple[int, int]] = None
    source_types: Optional[List[SourceType]] = None
    max_results: int = 10


@dataclass
class ResearchFinding:
    """Represents a research finding or insight."""
    finding_id: str
    topic: str
    summary: str
    supporting_documents: List[str]  # Document IDs
    confidence: float
    contradictions: List[str] = field(default_factory=list)


@dataclass
class ResearchReport:
    """Comprehensive research report."""
    title: str
    research_question: str
    methodology: str
    findings: List[ResearchFinding]
    synthesis: str
    citations: List[Citation]
    created_at: datetime = field(default_factory=datetime.now)


class DocumentStore:
    """Stores and indexes research documents."""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.index: Dict[str, Set[str]] = defaultdict(set)  # keyword -> doc_ids
    
    def add_document(self, document: Document) -> None:
        """Add document to store."""
        self.documents[document.doc_id] = document
        
        # Index by keywords
        for keyword in document.keywords:
            self.index[keyword.lower()].add(document.doc_id)
        
        # Index by title words
        for word in document.title.lower().split():
            if len(word) > 3:
                self.index[word].add(document.doc_id)
    
    def search(self, query: ResearchQuery) -> List[Document]:
        """Search for documents matching query."""
        matching_doc_ids: Set[str] = set()
        
        # Search by keywords
        for keyword in query.keywords:
            matching_doc_ids.update(self.index.get(keyword.lower(), set()))
        
        # Filter results
        results = []
        for doc_id in matching_doc_ids:
            doc = self.documents[doc_id]
            
            # Apply filters
            if query.source_types and doc.source_type not in query.source_types:
                continue
            
            if query.date_range:
                min_year, max_year = query.date_range
                if not (min_year <= doc.year <= max_year):
                    continue
            
            results.append(doc)
        
        # Sort by relevance (citation count as proxy)
        results.sort(key=lambda d: d.citation_count, reverse=True)
        
        return results[:query.max_results]
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by ID."""
        return self.documents.get(doc_id)


class CitationManager:
    """Manages citations and bibliographies."""
    
    def __init__(self):
        self.citations: Dict[str, Citation] = {}
        self.citation_graph: Dict[str, List[str]] = defaultdict(list)
    
    def add_citation(self, citation: Citation) -> None:
        """Add citation to manager."""
        self.citations[citation.citation_id] = citation
    
    def create_from_document(self, document: Document) -> Citation:
        """Create citation from document."""
        citation = Citation(
            citation_id=document.doc_id,
            authors=document.authors,
            title=document.title,
            year=document.year,
            source_type=document.source_type
        )
        self.add_citation(citation)
        return citation
    
    def add_relationship(self, citing_id: str, cited_id: str) -> None:
        """Record citation relationship."""
        self.citation_graph[citing_id].append(cited_id)
    
    def get_bibliography(self, doc_ids: List[str], style: str = "APA") -> List[str]:
        """Generate bibliography for documents."""
        bibliography = []
        
        for doc_id in sorted(set(doc_ids)):
            if doc_id in self.citations:
                citation = self.citations[doc_id]
                if style == "APA":
                    bibliography.append(citation.format_apa())
                elif style == "MLA":
                    bibliography.append(citation.format_mla())
        
        return bibliography


class ResearchAgent:
    """Agent capable of conducting research."""
    
    def __init__(self, name: str):
        self.name = name
        self.document_store = DocumentStore()
        self.citation_manager = CitationManager()
        self.research_notes: Dict[str, str] = {}
        self.current_phase = ResearchPhase.SEARCH
    
    def initialize_corpus(self, documents: List[Document]) -> None:
        """Initialize document corpus."""
        print(f"[{self.name}] Initializing corpus with {len(documents)} documents")
        for doc in documents:
            self.document_store.add_document(doc)
            self.citation_manager.create_from_document(doc)
    
    def search(self, query: ResearchQuery) -> List[Document]:
        """Search for relevant documents."""
        print(f"\n[{self.name}] Searching: {query.query_text}")
        print(f"  Keywords: {', '.join(query.keywords)}")
        
        self.current_phase = ResearchPhase.SEARCH
        results = self.document_store.search(query)
        
        print(f"  ✓ Found {len(results)} documents")
        return results
    
    def read_document(self, doc_id: str) -> Optional[Document]:
        """Read and analyze a document."""
        self.current_phase = ResearchPhase.READ
        document = self.document_store.get_document(doc_id)
        
        if not document:
            return None
        
        print(f"\n[{self.name}] Reading: {document.title}")
        print(f"  Authors: {', '.join(document.authors)}")
        print(f"  Year: {document.year}")
        
        # Extract and store key concepts
        concepts = document.extract_key_concepts()
        self.research_notes[doc_id] = f"Key concepts: {', '.join(concepts)}"
        
        return document
    
    def analyze_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Analyze a collection of documents."""
        print(f"\n[{self.name}] Analyzing {len(documents)} documents")
        self.current_phase = ResearchPhase.ANALYZE
        
        # Extract common themes
        all_keywords = []
        for doc in documents:
            all_keywords.extend(doc.keywords)
        
        keyword_freq = defaultdict(int)
        for keyword in all_keywords:
            keyword_freq[keyword] += 1
        
        common_themes = sorted(
            keyword_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Year distribution
        year_dist = defaultdict(int)
        for doc in documents:
            year_dist[doc.year] += 1
        
        analysis = {
            "total_documents": len(documents),
            "common_themes": [theme for theme, count in common_themes],
            "year_range": (
                min(doc.year for doc in documents),
                max(doc.year for doc in documents)
            ),
            "source_types": list(set(doc.source_type for doc in documents)),
            "most_cited": max(documents, key=lambda d: d.citation_count) if documents else None
        }
        
        print(f"  Common themes: {', '.join(analysis['common_themes'])}")
        print(f"  Year range: {analysis['year_range'][0]}-{analysis['year_range'][1]}")
        
        return analysis
    
    def synthesize_findings(
        self,
        documents: List[Document],
        research_question: str
    ) -> List[ResearchFinding]:
        """Synthesize findings from documents."""
        print(f"\n[{self.name}] Synthesizing findings")
        self.current_phase = ResearchPhase.SYNTHESIZE
        
        findings = []
        
        # Group documents by theme
        theme_docs = defaultdict(list)
        for doc in documents:
            for keyword in doc.keywords:
                theme_docs[keyword].append(doc.doc_id)
        
        # Create findings for major themes
        for theme, doc_ids in list(theme_docs.items())[:5]:
            if len(doc_ids) >= 2:  # At least 2 documents support theme
                finding = ResearchFinding(
                    finding_id=f"finding_{len(findings) + 1}",
                    topic=theme,
                    summary=f"Multiple sources discuss {theme} in relation to {research_question}",
                    supporting_documents=doc_ids,
                    confidence=min(0.9, len(doc_ids) * 0.2)
                )
                findings.append(finding)
        
        print(f"  ✓ Generated {len(findings)} findings")
        
        return findings
    
    def generate_report(
        self,
        research_question: str,
        documents: List[Document],
        findings: List[ResearchFinding]
    ) -> ResearchReport:
        """Generate comprehensive research report."""
        print(f"\n[{self.name}] Generating research report")
        self.current_phase = ResearchPhase.REPORT
        
        # Create synthesis
        synthesis_parts = []
        for finding in findings:
            doc_count = len(finding.supporting_documents)
            synthesis_parts.append(
                f"{finding.topic.title()} emerged as a significant theme, "
                f"supported by {doc_count} sources with {finding.confidence:.0%} confidence."
            )
        
        synthesis = " ".join(synthesis_parts)
        
        # Gather citations
        citations = []
        for doc in documents:
            citation = self.citation_manager.citations.get(doc.doc_id)
            if citation:
                citations.append(citation)
        
        report = ResearchReport(
            title=f"Research Report: {research_question}",
            research_question=research_question,
            methodology="Systematic literature review with thematic analysis",
            findings=findings,
            synthesis=synthesis,
            citations=citations
        )
        
        print(f"  ✓ Report generated with {len(findings)} findings and {len(citations)} citations")
        
        return report
    
    def conduct_research(self, research_question: str, keywords: List[str]) -> ResearchReport:
        """Conduct complete research workflow."""
        print("=" * 60)
        print(f"RESEARCH PROJECT: {research_question}")
        print("=" * 60)
        
        # Step 1: Search
        query = ResearchQuery(
            query_text=research_question,
            keywords=keywords,
            max_results=10
        )
        documents = self.search(query)
        
        # Step 2: Read key documents
        for doc in documents[:3]:
            self.read_document(doc.doc_id)
        
        # Step 3: Analyze
        analysis = self.analyze_documents(documents)
        
        # Step 4: Synthesize
        findings = self.synthesize_findings(documents, research_question)
        
        # Step 5: Generate report
        report = self.generate_report(research_question, documents, findings)
        
        return report


def demonstrate_research_agent():
    """Demonstrate research agent capabilities."""
    print("=" * 60)
    print("RESEARCH AGENT DEMONSTRATION")
    print("=" * 60)
    
    # Create research agent
    agent = ResearchAgent("ResearchBot")
    
    # Create sample document corpus
    documents = [
        Document(
            doc_id="doc1",
            title="Deep Learning for Natural Language Processing",
            authors=["Smith, J.", "Jones, A."],
            abstract="Survey of deep learning methods for NLP tasks",
            content="Deep learning has revolutionized NLP through transformers and attention mechanisms.",
            source_type=SourceType.ACADEMIC_PAPER,
            year=2021,
            keywords=["deep learning", "NLP", "transformers", "attention"],
            citation_count=150
        ),
        Document(
            doc_id="doc2",
            title="Attention Mechanisms in Neural Networks",
            authors=["Brown, K."],
            abstract="Comprehensive review of attention mechanisms",
            content="Attention mechanisms allow models to focus on relevant parts of input sequences.",
            source_type=SourceType.ACADEMIC_PAPER,
            year=2020,
            keywords=["attention", "neural networks", "transformers"],
            citation_count=200
        ),
        Document(
            doc_id="doc3",
            title="Large Language Models and Few-Shot Learning",
            authors=["Davis, M.", "Wilson, R."],
            abstract="Analysis of few-shot learning capabilities in LLMs",
            content="Large language models demonstrate remarkable few-shot learning abilities.",
            source_type=SourceType.ACADEMIC_PAPER,
            year=2022,
            keywords=["language models", "few-shot learning", "transformers"],
            citation_count=180
        ),
        Document(
            doc_id="doc4",
            title="Agentic AI Systems: Design and Implementation",
            authors=["Taylor, S."],
            abstract="Framework for building agentic AI systems",
            content="Agentic AI systems combine reasoning, planning, and tool use for autonomous operation.",
            source_type=SourceType.REPORT,
            year=2023,
            keywords=["agentic AI", "reasoning", "planning", "autonomy"],
            citation_count=50
        ),
        Document(
            doc_id="doc5",
            title="Transformer Architecture Evolution",
            authors=["Lee, H.", "Chen, Y."],
            abstract="Historical development of transformer models",
            content="Transformers evolved from RNNs to become the dominant architecture in modern AI.",
            source_type=SourceType.ACADEMIC_PAPER,
            year=2021,
            keywords=["transformers", "architecture", "deep learning"],
            citation_count=120
        ),
    ]
    
    # Initialize corpus
    agent.initialize_corpus(documents)
    
    # Conduct research
    report = agent.conduct_research(
        research_question="What are the key components of modern AI systems?",
        keywords=["transformers", "attention", "language models"]
    )
    
    # Display report
    print("\n" + "=" * 60)
    print("RESEARCH REPORT")
    print("=" * 60)
    
    print(f"\nTitle: {report.title}")
    print(f"Research Question: {report.research_question}")
    print(f"Methodology: {report.methodology}")
    
    print(f"\nFindings ({len(report.findings)}):")
    for i, finding in enumerate(report.findings, 1):
        print(f"\n{i}. {finding.topic.title()}")
        print(f"   Summary: {finding.summary}")
        print(f"   Confidence: {finding.confidence:.0%}")
        print(f"   Supporting documents: {len(finding.supporting_documents)}")
    
    print(f"\nSynthesis:")
    print(f"  {report.synthesis}")
    
    print(f"\nBibliography ({len(report.citations)} sources):")
    bibliography = agent.citation_manager.get_bibliography(
        [c.citation_id for c in report.citations],
        style="APA"
    )
    for i, citation in enumerate(bibliography, 1):
        print(f"{i}. {citation}")


if __name__ == "__main__":
    demonstrate_research_agent()
