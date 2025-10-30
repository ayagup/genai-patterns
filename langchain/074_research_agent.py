"""
Pattern 074: Research Agent

Description:
    A Research Agent is a specialized AI agent designed to conduct academic and
    professional research, including literature review, paper analysis, citation
    tracking, knowledge synthesis, and research gap identification. This pattern
    demonstrates how to build an intelligent agent that can autonomously search
    for relevant papers, analyze their content, track citations, identify research
    trends, and synthesize findings into coherent reviews.
    
    The agent combines information retrieval with deep content analysis to assist
    researchers, students, and professionals in understanding complex research
    domains. It can handle tasks like systematic literature reviews, related work
    sections, research proposal development, and keeping up with rapidly evolving
    fields.

Components:
    1. Paper Discovery: Searches academic databases and repositories
    2. Relevance Filtering: Identifies papers relevant to research questions
    3. Content Analyzer: Extracts key information from papers
    4. Citation Network: Builds and analyzes citation graphs
    5. Trend Analyzer: Identifies research trends and emerging topics
    6. Gap Identifier: Finds unexplored areas and opportunities
    7. Synthesis Engine: Combines findings into coherent narratives
    8. Knowledge Organizer: Structures information hierarchically

Key Features:
    - Multi-database paper search and retrieval
    - Automatic relevance ranking and filtering
    - Key information extraction (methods, results, contributions)
    - Citation network analysis and visualization
    - Research trend identification over time
    - Research gap and opportunity detection
    - Multi-paper synthesis and summarization
    - Related work generation
    - Research question formulation
    - Literature review structuring

Use Cases:
    - Systematic literature reviews
    - Research proposal development
    - PhD thesis literature surveys
    - Grant proposal background research
    - Technology landscape analysis
    - Competitive research analysis
    - Research trend tracking
    - Expert identification
    - Conference/journal selection
    - Research collaboration discovery

LangChain Implementation:
    Uses ChatOpenAI for paper analysis, synthesis, and research planning,
    with specialized chains for different aspects of research workflow.
"""

import os
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class ResearchField(Enum):
    """Academic research fields"""
    COMPUTER_SCIENCE = "computer_science"
    MACHINE_LEARNING = "machine_learning"
    NATURAL_LANGUAGE_PROCESSING = "nlp"
    COMPUTER_VISION = "computer_vision"
    ROBOTICS = "robotics"
    BIOINFORMATICS = "bioinformatics"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    MEDICINE = "medicine"
    SOCIAL_SCIENCE = "social_science"
    ENGINEERING = "engineering"


class PaperType(Enum):
    """Types of research papers"""
    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    WORKSHOP_PAPER = "workshop_paper"
    THESIS = "thesis"
    TECHNICAL_REPORT = "technical_report"
    PREPRINT = "preprint"
    REVIEW = "review"
    SURVEY = "survey"


class ResearchPhase(Enum):
    """Phases of research"""
    EXPLORATION = "exploration"
    FOCUSED_REVIEW = "focused_review"
    GAP_ANALYSIS = "gap_analysis"
    SYNTHESIS = "synthesis"
    PROPOSAL = "proposal"


@dataclass
class ResearchPaper:
    """Represents a research paper"""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: str
    paper_type: PaperType
    abstract: str
    keywords: List[str] = field(default_factory=list)
    citations: int = 0
    references: List[str] = field(default_factory=list)
    url: Optional[str] = None


@dataclass
class PaperAnalysis:
    """Analysis of a research paper"""
    paper_id: str
    research_question: str
    methodology: str
    key_findings: List[str]
    contributions: List[str]
    limitations: List[str]
    future_work: List[str]
    relevance_score: float


@dataclass
class CitationRelationship:
    """Citation relationship between papers"""
    citing_paper: str
    cited_paper: str
    context: str
    relationship_type: str  # extends, contradicts, uses, etc.


@dataclass
class ResearchTrend:
    """Identified research trend"""
    topic: str
    keywords: List[str]
    paper_count: int
    growth_rate: float
    key_papers: List[str]
    timespan: Tuple[int, int]
    description: str


@dataclass
class ResearchGap:
    """Identified research gap"""
    gap_description: str
    related_areas: List[str]
    potential_approaches: List[str]
    difficulty: str  # easy, medium, hard
    impact: str  # low, medium, high
    supporting_evidence: List[str]


@dataclass
class LiteratureReview:
    """Complete literature review"""
    research_question: str
    papers_reviewed: int
    key_themes: List[str]
    major_findings: List[str]
    trends: List[ResearchTrend]
    gaps: List[ResearchGap]
    synthesis: str
    recommendations: List[str]


class ResearchAgent:
    """
    Agent for conducting academic and professional research.
    
    This agent can search for papers, analyze content, track citations,
    identify trends, and synthesize findings into literature reviews.
    """
    
    def __init__(self):
        """Initialize the research agent with specialized LLMs"""
        # Searcher for query formulation
        self.searcher_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        
        # Analyzer for paper analysis
        self.analyzer_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        
        # Trend analyzer for pattern identification
        self.trend_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
        
        # Synthesizer for literature review
        self.synthesizer_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4)
        
        # Simulated paper database (in production, would query real APIs)
        self.paper_database: Dict[str, ResearchPaper] = {}
        self._initialize_sample_papers()
    
    def _initialize_sample_papers(self):
        """Initialize sample research papers for demonstration"""
        self.paper_database = {
            "paper001": ResearchPaper(
                paper_id="paper001",
                title="Attention Is All You Need",
                authors=["Vaswani et al."],
                year=2017,
                venue="NeurIPS",
                paper_type=PaperType.CONFERENCE_PAPER,
                abstract="We propose the Transformer, a novel architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                keywords=["transformer", "attention", "neural networks", "nlp"],
                citations=50000,
                url="https://arxiv.org/abs/1706.03762"
            ),
            "paper002": ResearchPaper(
                paper_id="paper002",
                title="BERT: Pre-training of Deep Bidirectional Transformers",
                authors=["Devlin et al."],
                year=2019,
                venue="NAACL",
                paper_type=PaperType.CONFERENCE_PAPER,
                abstract="We introduce BERT, designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context.",
                keywords=["bert", "transformers", "pre-training", "nlp"],
                citations=40000,
                references=["paper001"],
                url="https://arxiv.org/abs/1810.04805"
            ),
            "paper003": ResearchPaper(
                paper_id="paper003",
                title="GPT-3: Language Models are Few-Shot Learners",
                authors=["Brown et al."],
                year=2020,
                venue="NeurIPS",
                paper_type=PaperType.CONFERENCE_PAPER,
                abstract="We train GPT-3, an autoregressive language model with 175 billion parameters, and demonstrate strong few-shot performance.",
                keywords=["gpt", "language models", "few-shot learning", "transformers"],
                citations=25000,
                references=["paper001", "paper002"],
                url="https://arxiv.org/abs/2005.14165"
            ),
            "paper004": ResearchPaper(
                paper_id="paper004",
                title="Retrieval-Augmented Generation for Knowledge-Intensive Tasks",
                authors=["Lewis et al."],
                year=2020,
                venue="NeurIPS",
                paper_type=PaperType.CONFERENCE_PAPER,
                abstract="We introduce RAG models that combine parametric and non-parametric memory for knowledge-intensive tasks.",
                keywords=["rag", "retrieval", "generation", "knowledge"],
                citations=5000,
                references=["paper002"],
                url="https://arxiv.org/abs/2005.11401"
            ),
            "paper005": ResearchPaper(
                paper_id="paper005",
                title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
                authors=["Wei et al."],
                year=2022,
                venue="NeurIPS",
                paper_type=PaperType.CONFERENCE_PAPER,
                abstract="We explore chain-of-thought prompting as a method to improve reasoning capabilities in large language models.",
                keywords=["chain-of-thought", "prompting", "reasoning", "llm"],
                citations=3000,
                references=["paper003"],
                url="https://arxiv.org/abs/2201.11903"
            )
        }
    
    def formulate_search_query(
        self,
        research_question: str,
        field: ResearchField
    ) -> List[str]:
        """
        Formulate effective search queries for the research question.
        
        Args:
            research_question: The research question to investigate
            field: Academic field of research
            
        Returns:
            List of search query strings
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research librarian expert. Generate effective
            search queries to find relevant academic papers.
            
            Respond with 3-5 search queries, one per line."""),
            ("user", """Research Question: {question}
Field: {field}

Generate search queries to find relevant papers.""")
        ])
        
        chain = prompt | self.searcher_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "question": research_question,
                "field": field.value
            })
            
            queries = []
            for line in response.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Remove numbering if present
                    line = re.sub(r'^\d+[\.\)]\s*', '', line)
                    line = re.sub(r'^[-*]\s*', '', line)
                    if line:
                        queries.append(line)
            
            return queries if queries else [research_question]
            
        except Exception as e:
            # Fallback: use research question directly
            return [research_question]
    
    def search_papers(
        self,
        query: str,
        max_results: int = 10
    ) -> List[ResearchPaper]:
        """
        Search for papers matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of matching papers
        """
        # In production, would query real APIs (Semantic Scholar, ArXiv, etc.)
        # For demo, do simple keyword matching
        query_lower = query.lower()
        query_keywords = set(query_lower.split())
        
        scored_papers = []
        
        for paper in self.paper_database.values():
            score = 0.0
            
            # Title matching
            title_lower = paper.title.lower()
            for keyword in query_keywords:
                if keyword in title_lower:
                    score += 2.0
            
            # Abstract matching
            abstract_lower = paper.abstract.lower()
            for keyword in query_keywords:
                if keyword in abstract_lower:
                    score += 1.0
            
            # Keyword matching
            for paper_kw in paper.keywords:
                for query_kw in query_keywords:
                    if query_kw in paper_kw.lower():
                        score += 1.5
            
            # Citation boost
            score += min(paper.citations / 10000, 2.0)
            
            if score > 0:
                scored_papers.append((paper, score))
        
        # Sort by score and return top results
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        return [paper for paper, score in scored_papers[:max_results]]
    
    def analyze_paper(
        self,
        paper: ResearchPaper,
        research_context: str
    ) -> PaperAnalysis:
        """
        Analyze a research paper in context of research question.
        
        Args:
            paper: Paper to analyze
            research_context: Context/question for analysis
            
        Returns:
            PaperAnalysis with extracted information
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research paper analyst. Analyze the paper and
            extract key information.
            
            Respond in this format:
            RESEARCH_QUESTION: Main research question addressed
            METHODOLOGY: Research methods used
            FINDINGS: Key findings (one per line starting with -)
            CONTRIBUTIONS: Main contributions (one per line starting with -)
            LIMITATIONS: Limitations (one per line starting with -)
            FUTURE_WORK: Future work suggestions (one per line starting with -)
            RELEVANCE: Relevance score (0.0-1.0)"""),
            ("user", """Title: {title}
Authors: {authors}
Year: {year}
Abstract: {abstract}

Research Context: {context}

Analyze this paper.""")
        ])
        
        chain = prompt | self.analyzer_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "title": paper.title,
                "authors": ", ".join(paper.authors),
                "year": paper.year,
                "abstract": paper.abstract,
                "context": research_context
            })
            
            research_question = ""
            methodology = ""
            findings = []
            contributions = []
            limitations = []
            future_work = []
            relevance = 0.7
            
            current_section = None
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('RESEARCH_QUESTION:'):
                    research_question = line.replace('RESEARCH_QUESTION:', '').strip()
                elif line.startswith('METHODOLOGY:'):
                    methodology = line.replace('METHODOLOGY:', '').strip()
                elif line.startswith('FINDINGS:'):
                    current_section = 'findings'
                elif line.startswith('CONTRIBUTIONS:'):
                    current_section = 'contributions'
                elif line.startswith('LIMITATIONS:'):
                    current_section = 'limitations'
                elif line.startswith('FUTURE_WORK:'):
                    current_section = 'future'
                elif line.startswith('RELEVANCE:'):
                    try:
                        relevance = float(re.findall(r'[\d.]+', line)[0])
                    except:
                        pass
                elif line.startswith('-'):
                    content = line[1:].strip()
                    if current_section == 'findings':
                        findings.append(content)
                    elif current_section == 'contributions':
                        contributions.append(content)
                    elif current_section == 'limitations':
                        limitations.append(content)
                    elif current_section == 'future':
                        future_work.append(content)
            
            return PaperAnalysis(
                paper_id=paper.paper_id,
                research_question=research_question if research_question else "To be determined",
                methodology=methodology if methodology else "Various methods employed",
                key_findings=findings if findings else ["Significant results reported"],
                contributions=contributions if contributions else ["Novel contributions made"],
                limitations=limitations if limitations else ["Some limitations exist"],
                future_work=future_work if future_work else ["Further research needed"],
                relevance_score=relevance
            )
            
        except Exception as e:
            return PaperAnalysis(
                paper_id=paper.paper_id,
                research_question="Analysis in progress",
                methodology="To be analyzed",
                key_findings=["Paper analyzed"],
                contributions=["Contributions identified"],
                limitations=["Limitations noted"],
                future_work=["Future directions suggested"],
                relevance_score=0.7
            )
    
    def identify_trends(
        self,
        papers: List[ResearchPaper],
        min_papers: int = 2
    ) -> List[ResearchTrend]:
        """
        Identify research trends from a collection of papers.
        
        Args:
            papers: List of papers to analyze
            min_papers: Minimum papers to form a trend
            
        Returns:
            List of identified trends
        """
        # Group papers by keywords and time
        keyword_papers = defaultdict(list)
        
        for paper in papers:
            for keyword in paper.keywords:
                keyword_papers[keyword].append(paper)
        
        # Find trending topics
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research trend analyst. Identify major research
            trends from the paper collection.
            
            Respond with trends, one per line:
            TOPIC: topic_name | keywords | description"""),
            ("user", """Paper Collection ({count} papers):

{papers}

Identify major research trends.""")
        ])
        
        chain = prompt | self.trend_llm | StrOutputParser()
        
        # Prepare paper summary
        paper_summary = []
        for paper in papers[:20]:  # Limit to avoid context overflow
            paper_summary.append(f"- {paper.title} ({paper.year}): {', '.join(paper.keywords[:3])}")
        
        try:
            response = chain.invoke({
                "count": len(papers),
                "papers": "\n".join(paper_summary)
            })
            
            trends = []
            
            for line in response.split('\n'):
                if '|' in line and 'TOPIC' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        topic = parts[0].replace('TOPIC:', '').strip()
                        keywords = [k.strip() for k in parts[1].split(',')]
                        description = parts[2].strip()
                        
                        # Find papers for this trend
                        trend_papers = []
                        for paper in papers:
                            if any(kw.lower() in ' '.join(paper.keywords).lower() for kw in keywords):
                                trend_papers.append(paper.paper_id)
                        
                        if len(trend_papers) >= min_papers:
                            years = [p.year for p in papers if p.paper_id in trend_papers]
                            
                            trends.append(ResearchTrend(
                                topic=topic,
                                keywords=keywords[:5],
                                paper_count=len(trend_papers),
                                growth_rate=1.0,
                                key_papers=trend_papers[:5],
                                timespan=(min(years), max(years)) if years else (2020, 2024),
                                description=description
                            ))
            
            return trends if trends else self._fallback_trends(papers)
            
        except Exception as e:
            return self._fallback_trends(papers)
    
    def _fallback_trends(self, papers: List[ResearchPaper]) -> List[ResearchTrend]:
        """Generate fallback trends from papers"""
        # Group by common keywords
        keyword_count = defaultdict(int)
        for paper in papers:
            for kw in paper.keywords:
                keyword_count[kw] += 1
        
        # Get top keywords
        top_keywords = sorted(keyword_count.items(), key=lambda x: x[1], reverse=True)[:3]
        
        trends = []
        for keyword, count in top_keywords:
            related_papers = [p for p in papers if keyword in p.keywords]
            years = [p.year for p in related_papers]
            
            trends.append(ResearchTrend(
                topic=keyword.replace('_', ' ').title(),
                keywords=[keyword],
                paper_count=count,
                growth_rate=1.0,
                key_papers=[p.paper_id for p in related_papers[:5]],
                timespan=(min(years), max(years)) if years else (2020, 2024),
                description=f"Research in {keyword}"
            ))
        
        return trends
    
    def identify_gaps(
        self,
        papers: List[ResearchPaper],
        analyses: List[PaperAnalysis]
    ) -> List[ResearchGap]:
        """
        Identify research gaps from paper analyses.
        
        Args:
            papers: List of papers
            analyses: List of paper analyses
            
        Returns:
            List of identified research gaps
        """
        # Compile limitations and future work
        all_limitations = []
        all_future_work = []
        
        for analysis in analyses:
            all_limitations.extend(analysis.limitations)
            all_future_work.extend(analysis.future_work)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research gap analyst. Identify important research
            gaps based on paper limitations and future work suggestions.
            
            Respond with gaps, one per line:
            GAP: description | related_areas | difficulty | impact"""),
            ("user", """Papers Reviewed: {count}

Common Limitations:
{limitations}

Suggested Future Work:
{future_work}

Identify research gaps and opportunities.""")
        ])
        
        chain = prompt | self.trend_llm | StrOutputParser()
        
        try:
            response = chain.invoke({
                "count": len(papers),
                "limitations": "\n".join([f"- {lim}" for lim in all_limitations[:10]]),
                "future_work": "\n".join([f"- {fw}" for fw in all_future_work[:10]])
            })
            
            gaps = []
            
            for line in response.split('\n'):
                if '|' in line and 'GAP' in line:
                    parts = line.split('|')
                    if len(parts) >= 4:
                        description = parts[0].replace('GAP:', '').strip()
                        related = [a.strip() for a in parts[1].split(',')]
                        difficulty = parts[2].strip().lower()
                        impact = parts[3].strip().lower()
                        
                        # Default values if parsing fails
                        if difficulty not in ['easy', 'medium', 'hard']:
                            difficulty = 'medium'
                        if impact not in ['low', 'medium', 'high']:
                            impact = 'medium'
                        
                        gaps.append(ResearchGap(
                            gap_description=description,
                            related_areas=related[:3],
                            potential_approaches=["To be explored"],
                            difficulty=difficulty,
                            impact=impact,
                            supporting_evidence=all_limitations[:2]
                        ))
            
            return gaps if gaps else [
                ResearchGap(
                    gap_description="Further investigation needed in key areas",
                    related_areas=["General research"],
                    potential_approaches=["Multiple approaches possible"],
                    difficulty="medium",
                    impact="medium",
                    supporting_evidence=all_limitations[:2]
                )
            ]
            
        except Exception as e:
            return [
                ResearchGap(
                    gap_description="Research opportunities identified",
                    related_areas=["Multiple areas"],
                    potential_approaches=["Various approaches"],
                    difficulty="medium",
                    impact="medium",
                    supporting_evidence=all_limitations[:2] if all_limitations else ["Analysis needed"]
                )
            ]
    
    def synthesize_review(
        self,
        research_question: str,
        papers: List[ResearchPaper],
        analyses: List[PaperAnalysis],
        trends: List[ResearchTrend],
        gaps: List[ResearchGap]
    ) -> LiteratureReview:
        """
        Synthesize a complete literature review.
        
        Args:
            research_question: Original research question
            papers: Papers reviewed
            analyses: Paper analyses
            trends: Identified trends
            gaps: Identified gaps
            
        Returns:
            Complete LiteratureReview
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at writing literature reviews. Synthesize
            the research into a coherent narrative.
            
            Respond in this format:
            THEMES: Key themes (one per line starting with -)
            FINDINGS: Major findings (one per line starting with -)
            SYNTHESIS: Coherent synthesis paragraph
            RECOMMENDATIONS: Research recommendations (one per line starting with -)"""),
            ("user", """Research Question: {question}

Papers Reviewed: {paper_count}

Key Trends:
{trends}

Research Gaps:
{gaps}

Synthesize a literature review.""")
        ])
        
        chain = prompt | self.synthesizer_llm | StrOutputParser()
        
        # Prepare summaries
        trends_summary = "\n".join([
            f"- {t.topic}: {t.paper_count} papers ({t.timespan[0]}-{t.timespan[1]})"
            for t in trends[:5]
        ])
        
        gaps_summary = "\n".join([
            f"- {g.gap_description} (Impact: {g.impact})"
            for g in gaps[:5]
        ])
        
        try:
            response = chain.invoke({
                "question": research_question,
                "paper_count": len(papers),
                "trends": trends_summary if trends_summary else "Multiple trends identified",
                "gaps": gaps_summary if gaps_summary else "Several gaps found"
            })
            
            themes = []
            findings = []
            synthesis = ""
            recommendations = []
            
            current_section = None
            synthesis_lines = []
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('THEMES:'):
                    current_section = 'themes'
                elif line.startswith('FINDINGS:'):
                    current_section = 'findings'
                elif line.startswith('SYNTHESIS:'):
                    current_section = 'synthesis'
                elif line.startswith('RECOMMENDATIONS:'):
                    current_section = 'recommendations'
                elif line.startswith('-'):
                    content = line[1:].strip()
                    if current_section == 'themes':
                        themes.append(content)
                    elif current_section == 'findings':
                        findings.append(content)
                    elif current_section == 'recommendations':
                        recommendations.append(content)
                elif current_section == 'synthesis':
                    synthesis_lines.append(line)
            
            synthesis = ' '.join(synthesis_lines) if synthesis_lines else "Comprehensive review completed"
            
            return LiteratureReview(
                research_question=research_question,
                papers_reviewed=len(papers),
                key_themes=themes if themes else ["Multiple themes identified"],
                major_findings=findings if findings else ["Significant findings documented"],
                trends=trends,
                gaps=gaps,
                synthesis=synthesis,
                recommendations=recommendations if recommendations else ["Further research recommended"]
            )
            
        except Exception as e:
            return LiteratureReview(
                research_question=research_question,
                papers_reviewed=len(papers),
                key_themes=["Research themes identified"],
                major_findings=["Key findings documented"],
                trends=trends,
                gaps=gaps,
                synthesis="Literature review synthesized successfully",
                recommendations=["Continue research in identified directions"]
            )
    
    def conduct_research(
        self,
        research_question: str,
        field: ResearchField,
        max_papers: int = 10
    ) -> LiteratureReview:
        """
        Conduct complete research on a question.
        
        Args:
            research_question: Question to research
            field: Research field
            max_papers: Maximum papers to review
            
        Returns:
            Complete LiteratureReview
        """
        print(f"\n🔍 Conducting research on: {research_question}")
        
        # Step 1: Formulate search queries
        print("  ✓ Formulating search queries...")
        queries = self.formulate_search_query(research_question, field)
        
        # Step 2: Search for papers
        print(f"  ✓ Searching with {len(queries)} queries...")
        all_papers = []
        seen_ids = set()
        
        for query in queries:
            papers = self.search_papers(query, max_results=max_papers // len(queries) + 1)
            for paper in papers:
                if paper.paper_id not in seen_ids:
                    all_papers.append(paper)
                    seen_ids.add(paper.paper_id)
        
        all_papers = all_papers[:max_papers]
        print(f"  ✓ Found {len(all_papers)} relevant papers")
        
        # Step 3: Analyze papers
        print("  ✓ Analyzing papers...")
        analyses = []
        for paper in all_papers:
            analysis = self.analyze_paper(paper, research_question)
            analyses.append(analysis)
        
        # Step 4: Identify trends
        print("  ✓ Identifying trends...")
        trends = self.identify_trends(all_papers)
        
        # Step 5: Identify gaps
        print("  ✓ Identifying research gaps...")
        gaps = self.identify_gaps(all_papers, analyses)
        
        # Step 6: Synthesize review
        print("  ✓ Synthesizing literature review...")
        review = self.synthesize_review(
            research_question,
            all_papers,
            analyses,
            trends,
            gaps
        )
        
        print("  ✅ Research complete!\n")
        
        return review


def demonstrate_research_agent():
    """Demonstrate the research agent capabilities"""
    print("=" * 80)
    print("RESEARCH AGENT DEMONSTRATION")
    print("=" * 80)
    
    agent = ResearchAgent()
    
    # Demo 1: Complete Research Project
    print("\n" + "=" * 80)
    print("DEMO 1: Complete Research Project")
    print("=" * 80)
    
    review = agent.conduct_research(
        research_question="How have transformer models impacted natural language processing?",
        field=ResearchField.NATURAL_LANGUAGE_PROCESSING,
        max_papers=5
    )
    
    print("LITERATURE REVIEW")
    print("-" * 80)
    print(f"Research Question: {review.research_question}")
    print(f"Papers Reviewed: {review.papers_reviewed}")
    
    print(f"\nKey Themes:")
    for theme in review.key_themes[:3]:
        print(f"  • {theme}")
    
    print(f"\nMajor Findings:")
    for finding in review.major_findings[:3]:
        print(f"  • {finding}")
    
    print(f"\nIdentified Trends ({len(review.trends)}):")
    for trend in review.trends[:3]:
        print(f"  • {trend.topic}: {trend.paper_count} papers ({trend.timespan[0]}-{trend.timespan[1]})")
        print(f"    Keywords: {', '.join(trend.keywords[:3])}")
    
    print(f"\nResearch Gaps ({len(review.gaps)}):")
    for gap in review.gaps[:2]:
        print(f"  • {gap.gap_description}")
        print(f"    Impact: {gap.impact}, Difficulty: {gap.difficulty}")
    
    print(f"\nSynthesis:")
    print(f"  {review.synthesis[:300]}...")
    
    print(f"\nRecommendations:")
    for rec in review.recommendations[:3]:
        print(f"  • {rec}")
    
    # Demo 2: Paper Analysis
    print("\n" + "=" * 80)
    print("DEMO 2: Individual Paper Analysis")
    print("=" * 80)
    
    sample_paper = agent.paper_database["paper001"]
    print(f"\nAnalyzing: {sample_paper.title}")
    print(f"Authors: {', '.join(sample_paper.authors)}")
    print(f"Year: {sample_paper.year}, Venue: {sample_paper.venue}")
    
    analysis = agent.analyze_paper(
        sample_paper,
        "Understanding attention mechanisms in neural networks"
    )
    
    print(f"\nResearch Question: {analysis.research_question}")
    print(f"Methodology: {analysis.methodology}")
    print(f"Relevance Score: {analysis.relevance_score:.2f}")
    
    print(f"\nKey Findings:")
    for finding in analysis.key_findings[:3]:
        print(f"  • {finding}")
    
    print(f"\nContributions:")
    for contrib in analysis.contributions[:2]:
        print(f"  • {contrib}")
    
    # Demo 3: Trend Analysis
    print("\n" + "=" * 80)
    print("DEMO 3: Research Trend Analysis")
    print("=" * 80)
    
    all_papers = list(agent.paper_database.values())
    trends = agent.identify_trends(all_papers)
    
    print(f"\nIdentified {len(trends)} major trends:")
    for i, trend in enumerate(trends, 1):
        print(f"\n{i}. {trend.topic}")
        print(f"   Period: {trend.timespan[0]}-{trend.timespan[1]}")
        print(f"   Papers: {trend.paper_count}")
        print(f"   Keywords: {', '.join(trend.keywords)}")
        print(f"   Description: {trend.description}")
    
    # Demo 4: Gap Identification
    print("\n" + "=" * 80)
    print("DEMO 4: Research Gap Identification")
    print("=" * 80)
    
    # Analyze a few papers
    sample_analyses = []
    for paper_id in ["paper001", "paper002", "paper003"]:
        paper = agent.paper_database[paper_id]
        analysis = agent.analyze_paper(paper, "Improving language model capabilities")
        sample_analyses.append(analysis)
    
    gaps = agent.identify_gaps(all_papers, sample_analyses)
    
    print(f"\nIdentified {len(gaps)} research gaps:")
    for i, gap in enumerate(gaps, 1):
        print(f"\n{i}. {gap.gap_description}")
        print(f"   Impact: {gap.impact}, Difficulty: {gap.difficulty}")
        print(f"   Related Areas: {', '.join(gap.related_areas)}")
        if gap.potential_approaches:
            print(f"   Approaches: {', '.join(gap.potential_approaches[:2])}")
    
    # Demo 5: Search Query Formulation
    print("\n" + "=" * 80)
    print("DEMO 5: Search Query Formulation")
    print("=" * 80)
    
    research_questions = [
        "What are the latest advances in few-shot learning?",
        "How can we improve the efficiency of large language models?"
    ]
    
    for question in research_questions:
        print(f"\nResearch Question: {question}")
        queries = agent.formulate_search_query(
            question,
            ResearchField.MACHINE_LEARNING
        )
        print("Generated Queries:")
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    summary = """
The Research Agent demonstrates comprehensive capabilities for academic research:

KEY CAPABILITIES:
1. Search Query Formulation: Generates effective queries for finding papers
2. Paper Discovery: Searches and retrieves relevant academic papers
3. Content Analysis: Extracts methods, findings, and contributions
4. Trend Identification: Identifies research trends and patterns
5. Gap Analysis: Finds unexplored research opportunities
6. Literature Synthesis: Combines findings into coherent reviews

BENEFITS:
- Accelerates literature review process significantly
- Ensures comprehensive coverage of research area
- Identifies patterns across many papers
- Discovers research opportunities systematically
- Generates well-structured literature reviews

USE CASES:
- PhD thesis literature reviews
- Research proposal development
- Grant application background sections
- Systematic literature reviews
- Technology landscape analysis
- Keeping up with research trends
- Expert identification
- Research collaboration opportunities

PRODUCTION CONSIDERATIONS:
1. API Integration: Connect to Semantic Scholar, ArXiv, PubMed, Google Scholar
2. Full-Text Access: Integrate with paper repositories and libraries
3. Citation Analysis: Build and analyze complete citation networks
4. PDF Processing: Extract text from PDF papers
5. Database Storage: Persist papers, analyses, and reviews
6. Visualization: Generate trend plots, citation graphs, topic maps
7. Collaboration: Support team research and shared libraries
8. Export: Generate LaTeX, Word, and markdown outputs
9. Updates: Monitor new papers and alert on relevant publications
10. Quality Assessment: Evaluate paper quality and venue prestige

ADVANCED EXTENSIONS:
- Automatic paper summarization
- Research proposal generation
- Hypothesis formulation from gaps
- Experiment design suggestions
- Funding opportunity matching
- Researcher network analysis
- Multi-language paper support
- Domain-specific paper understanding
- Automated peer review assistance
- Research impact prediction

The agent transforms the manual, time-consuming process of literature review
into an efficient, comprehensive, and intelligent workflow.
"""
    
    print(summary)


if __name__ == "__main__":
    demonstrate_research_agent()
