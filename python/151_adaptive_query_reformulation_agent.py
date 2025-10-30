"""
Adaptive Query Reformulation Pattern

Automatically reformulates queries based on context, feedback, and search results.
Improves information retrieval through iterative query refinement.

Use Cases:
- Search engines
- Database querying
- Information retrieval systems
- Chatbots with search capabilities

Advantages:
- Improved search accuracy
- Context-aware reformulation
- Learning from feedback
- Multi-turn query optimization
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
import re


class ReformulationType(Enum):
    """Types of query reformulation"""
    EXPANSION = "expansion"  # Add related terms
    SPECIALIZATION = "specialization"  # Make more specific
    GENERALIZATION = "generalization"  # Make more general
    SYNONYM_REPLACEMENT = "synonym_replacement"  # Replace with synonyms
    SPELLING_CORRECTION = "spelling_correction"  # Fix spelling
    CONTEXT_INJECTION = "context_injection"  # Add contextual terms
    NEGATIVE_FILTERING = "negative_filtering"  # Add negative terms


@dataclass
class QueryContext:
    """Context information for query reformulation"""
    user_id: str
    session_id: str
    previous_queries: List[str]
    clicked_results: List[str]
    rejected_results: List[str]
    user_preferences: Dict[str, Any]
    domain: Optional[str] = None
    language: str = "en"


@dataclass
class ReformulationResult:
    """Result of query reformulation"""
    original_query: str
    reformulated_query: str
    reformulation_type: ReformulationType
    confidence: float
    explanation: str
    added_terms: List[str]
    removed_terms: List[str]
    timestamp: datetime


@dataclass
class SearchFeedback:
    """Feedback on search results"""
    query: str
    results_count: int
    clicked_results: List[str]
    time_to_click: Optional[float]
    user_satisfied: bool
    relevance_scores: Dict[str, float]


class QueryAnalyzer:
    """Analyzes queries to determine reformulation needs"""
    
    def __init__(self):
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was'
        }
        
    def extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        # Lowercase and split
        words = query.lower().split()
        
        # Remove stopwords and punctuation
        keywords = []
        for word in words:
            cleaned = re.sub(r'[^\w\s]', '', word)
            if cleaned and cleaned not in self.stopwords:
                keywords.append(cleaned)
        
        return keywords
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent"""
        query_lower = query.lower()
        
        intent = {
            "type": "informational",
            "specificity": "medium",
            "temporal": False,
            "local": False,
            "transactional": False
        }
        
        # Check for question words
        question_words = ['what', 'where', 'when', 'who', 'why', 'how']
        if any(word in query_lower for word in question_words):
            intent["type"] = "informational"
        
        # Check for transactional intent
        transactional_words = ['buy', 'purchase', 'order', 'price', 'cost']
        if any(word in query_lower for word in transactional_words):
            intent["type"] = "transactional"
            intent["transactional"] = True
        
        # Check for local intent
        local_words = ['near', 'nearby', 'location', 'address', 'directions']
        if any(word in query_lower for word in local_words):
            intent["local"] = True
        
        # Check for temporal intent
        temporal_words = ['today', 'yesterday', 'tomorrow', 'now', 'recent', 'latest']
        if any(word in query_lower for word in temporal_words):
            intent["temporal"] = True
        
        # Assess specificity
        keywords = self.extract_keywords(query)
        if len(keywords) > 4:
            intent["specificity"] = "high"
        elif len(keywords) < 2:
            intent["specificity"] = "low"
        
        return intent
    
    def detect_issues(self, query: str, results_count: int) -> List[str]:
        """Detect potential query issues"""
        issues = []
        
        # Too few results
        if results_count < 5:
            issues.append("too_specific")
        
        # Too many results
        if results_count > 10000:
            issues.append("too_general")
        
        # Very short query
        if len(query.split()) < 2:
            issues.append("too_short")
        
        # Very long query
        if len(query.split()) > 15:
            issues.append("too_long")
        
        # Possible spelling issues (simplistic check)
        words = query.split()
        for word in words:
            if len(word) > 15 or (len(word) > 5 and word.count('x') > 2):
                issues.append("possible_typo")
                break
        
        return issues


class ThesaurusService:
    """Provides synonyms and related terms"""
    
    def __init__(self):
        # Simplified thesaurus
        self.synonyms = {
            'buy': ['purchase', 'acquire', 'obtain'],
            'fast': ['quick', 'rapid', 'swift'],
            'cheap': ['affordable', 'inexpensive', 'budget'],
            'good': ['excellent', 'quality', 'superior'],
            'best': ['top', 'premier', 'optimal'],
            'car': ['vehicle', 'automobile', 'auto'],
            'phone': ['mobile', 'smartphone', 'cell'],
            'computer': ['pc', 'laptop', 'desktop']
        }
        
        self.related_terms = {
            'python': ['programming', 'code', 'software', 'development'],
            'cooking': ['recipe', 'ingredients', 'kitchen', 'food'],
            'travel': ['vacation', 'trip', 'destination', 'tourism'],
            'fitness': ['exercise', 'workout', 'health', 'training']
        }
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word"""
        return self.synonyms.get(word.lower(), [])
    
    def get_related_terms(self, word: str) -> List[str]:
        """Get related terms for a word"""
        return self.related_terms.get(word.lower(), [])
    
    def expand_query(self, query: str, max_expansions: int = 3) -> List[str]:
        """Expand query with synonyms and related terms"""
        words = query.lower().split()
        expansions = []
        
        for word in words:
            synonyms = self.get_synonyms(word)
            if synonyms:
                expansions.extend(synonyms[:max_expansions])
            
            related = self.get_related_terms(word)
            if related:
                expansions.extend(related[:max_expansions])
        
        return list(set(expansions))[:max_expansions * 2]


class AdaptiveQueryReformulator:
    """
    Agent that adaptively reformulates queries based on context and feedback.
    Learns from user behavior to improve search results.
    """
    
    def __init__(self):
        self.analyzer = QueryAnalyzer()
        self.thesaurus = ThesaurusService()
        self.reformulation_history: List[ReformulationResult] = []
        self.feedback_history: List[SearchFeedback] = []
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        
    def reformulate_query(self,
                         query: str,
                         context: QueryContext,
                         search_feedback: Optional[SearchFeedback] = None
                         ) -> List[ReformulationResult]:
        """
        Reformulate query based on context and feedback.
        
        Args:
            query: Original search query
            context: Query context information
            search_feedback: Optional feedback from previous search
            
        Returns:
            List of reformulation results
        """
        reformulations = []
        
        # Analyze query
        keywords = self.analyzer.extract_keywords(query)
        intent = self.analyzer.analyze_query_intent(query)
        
        # Detect issues if feedback provided
        issues = []
        if search_feedback:
            issues = self.analyzer.detect_issues(
                query, search_feedback.results_count
            )
        
        # Apply reformulation strategies
        
        # 1. Query Expansion
        if "too_specific" in issues or intent["specificity"] == "high":
            expansion = self._expand_query(query, keywords, context)
            if expansion:
                reformulations.append(expansion)
        
        # 2. Query Specialization
        if "too_general" in issues or intent["specificity"] == "low":
            specialization = self._specialize_query(query, keywords, context)
            if specialization:
                reformulations.append(specialization)
        
        # 3. Synonym Replacement
        synonym_reform = self._replace_with_synonyms(query, keywords)
        if synonym_reform:
            reformulations.append(synonym_reform)
        
        # 4. Context Injection
        if context.previous_queries or context.clicked_results:
            context_reform = self._inject_context(query, context)
            if context_reform:
                reformulations.append(context_reform)
        
        # 5. Spelling Correction
        if "possible_typo" in issues:
            spelling_reform = self._correct_spelling(query)
            if spelling_reform:
                reformulations.append(spelling_reform)
        
        # 6. Negative Filtering (based on rejected results)
        if context.rejected_results:
            negative_reform = self._add_negative_terms(query, context)
            if negative_reform:
                reformulations.append(negative_reform)
        
        # Store reformulations
        self.reformulation_history.extend(reformulations)
        
        # Sort by confidence
        reformulations.sort(key=lambda x: x.confidence, reverse=True)
        
        return reformulations
    
    def learn_from_feedback(self, feedback: SearchFeedback) -> None:
        """
        Learn from search feedback to improve future reformulations.
        
        Args:
            feedback: Feedback on search results
        """
        self.feedback_history.append(feedback)
        
        # Update user preferences based on successful queries
        if feedback.user_satisfied and feedback.clicked_results:
            query_keywords = self.analyzer.extract_keywords(feedback.query)
            
            # Extract patterns from successful searches
            for keyword in query_keywords:
                if keyword not in self.user_preferences:
                    self.user_preferences[keyword] = {
                        "success_count": 0,
                        "common_additions": [],
                        "preferred_specificity": "medium"
                    }
                
                self.user_preferences[keyword]["success_count"] += 1
    
    def get_best_reformulation(self,
                               query: str,
                               context: QueryContext,
                               feedback: Optional[SearchFeedback] = None
                               ) -> Optional[ReformulationResult]:
        """
        Get the best single reformulation.
        
        Args:
            query: Original query
            context: Query context
            feedback: Optional search feedback
            
        Returns:
            Best reformulation or None
        """
        reformulations = self.reformulate_query(query, context, feedback)
        return reformulations[0] if reformulations else None
    
    def _expand_query(self,
                     query: str,
                     keywords: List[str],
                     context: QueryContext) -> Optional[ReformulationResult]:
        """Expand query with related terms"""
        expansions = []
        
        # Get related terms from thesaurus
        for keyword in keywords:
            related = self.thesaurus.get_related_terms(keyword)
            expansions.extend(related[:2])
        
        if not expansions:
            return None
        
        # Create expanded query
        expanded_query = "{} {}".format(query, " ".join(expansions[:3]))
        
        return ReformulationResult(
            original_query=query,
            reformulated_query=expanded_query,
            reformulation_type=ReformulationType.EXPANSION,
            confidence=0.7,
            explanation="Added related terms to broaden search",
            added_terms=expansions[:3],
            removed_terms=[],
            timestamp=datetime.now()
        )
    
    def _specialize_query(self,
                         query: str,
                         keywords: List[str],
                         context: QueryContext) -> Optional[ReformulationResult]:
        """Make query more specific"""
        additional_terms = []
        
        # Add terms from user preferences
        if context.user_preferences:
            domain = context.user_preferences.get('domain')
            if domain:
                additional_terms.append(domain)
        
        # Add temporal constraint if not present
        intent = self.analyzer.analyze_query_intent(query)
        if not intent["temporal"] and context.user_preferences.get('prefer_recent'):
            additional_terms.append("recent")
        
        if not additional_terms:
            # Add first clicked result term
            if context.clicked_results:
                clicked_terms = self.analyzer.extract_keywords(
                    context.clicked_results[0]
                )
                if clicked_terms:
                    additional_terms.append(clicked_terms[0])
        
        if not additional_terms:
            return None
        
        specialized_query = "{} {}".format(query, " ".join(additional_terms))
        
        return ReformulationResult(
            original_query=query,
            reformulated_query=specialized_query,
            reformulation_type=ReformulationType.SPECIALIZATION,
            confidence=0.8,
            explanation="Made query more specific to narrow results",
            added_terms=additional_terms,
            removed_terms=[],
            timestamp=datetime.now()
        )
    
    def _replace_with_synonyms(self,
                               query: str,
                               keywords: List[str]
                               ) -> Optional[ReformulationResult]:
        """Replace terms with synonyms"""
        words = query.split()
        replaced = False
        replaced_terms = []
        new_query_words = []
        
        for word in words:
            synonyms = self.thesaurus.get_synonyms(word.lower())
            if synonyms and word.lower() in keywords:
                new_query_words.append(synonyms[0])
                replaced_terms.append(word.lower())
                replaced = True
            else:
                new_query_words.append(word)
        
        if not replaced:
            return None
        
        new_query = " ".join(new_query_words)
        
        return ReformulationResult(
            original_query=query,
            reformulated_query=new_query,
            reformulation_type=ReformulationType.SYNONYM_REPLACEMENT,
            confidence=0.65,
            explanation="Replaced terms with synonyms for alternative results",
            added_terms=[w for w in new_query_words if w not in words],
            removed_terms=replaced_terms,
            timestamp=datetime.now()
        )
    
    def _inject_context(self,
                       query: str,
                       context: QueryContext) -> Optional[ReformulationResult]:
        """Inject contextual terms from session"""
        contextual_terms = []
        
        # Extract terms from previous queries
        if context.previous_queries:
            for prev_query in context.previous_queries[-2:]:
                prev_keywords = self.analyzer.extract_keywords(prev_query)
                query_keywords = self.analyzer.extract_keywords(query)
                
                # Find terms not in current query
                for keyword in prev_keywords:
                    if keyword not in query_keywords:
                        contextual_terms.append(keyword)
                        break
        
        # Extract terms from clicked results
        if context.clicked_results and not contextual_terms:
            clicked_keywords = self.analyzer.extract_keywords(
                context.clicked_results[0]
            )
            query_keywords = self.analyzer.extract_keywords(query)
            
            for keyword in clicked_keywords:
                if keyword not in query_keywords:
                    contextual_terms.append(keyword)
                    break
        
        if not contextual_terms:
            return None
        
        context_query = "{} {}".format(query, contextual_terms[0])
        
        return ReformulationResult(
            original_query=query,
            reformulated_query=context_query,
            reformulation_type=ReformulationType.CONTEXT_INJECTION,
            confidence=0.75,
            explanation="Added contextual terms from session history",
            added_terms=contextual_terms[:1],
            removed_terms=[],
            timestamp=datetime.now()
        )
    
    def _correct_spelling(self, query: str) -> Optional[ReformulationResult]:
        """Correct potential spelling errors"""
        # Simplified spelling correction
        common_corrections = {
            'teh': 'the',
            'adn': 'and',
            'recieve': 'receive',
            'seperate': 'separate',
            'occured': 'occurred'
        }
        
        words = query.split()
        corrected = False
        corrected_terms = []
        new_words = []
        
        for word in words:
            if word.lower() in common_corrections:
                new_words.append(common_corrections[word.lower()])
                corrected_terms.append(word.lower())
                corrected = True
            else:
                new_words.append(word)
        
        if not corrected:
            return None
        
        corrected_query = " ".join(new_words)
        
        return ReformulationResult(
            original_query=query,
            reformulated_query=corrected_query,
            reformulation_type=ReformulationType.SPELLING_CORRECTION,
            confidence=0.9,
            explanation="Corrected potential spelling errors",
            added_terms=[],
            removed_terms=corrected_terms,
            timestamp=datetime.now()
        )
    
    def _add_negative_terms(self,
                           query: str,
                           context: QueryContext) -> Optional[ReformulationResult]:
        """Add negative terms based on rejected results"""
        if not context.rejected_results:
            return None
        
        # Extract common terms from rejected results
        rejected_keywords = []
        for result in context.rejected_results[:3]:
            keywords = self.analyzer.extract_keywords(result)
            rejected_keywords.extend(keywords)
        
        # Find most common rejected term
        if not rejected_keywords:
            return None
        
        from collections import Counter
        common_rejected = Counter(rejected_keywords).most_common(1)
        if not common_rejected:
            return None
        
        negative_term = common_rejected[0][0]
        negative_query = "{} -{}".format(query, negative_term)
        
        return ReformulationResult(
            original_query=query,
            reformulated_query=negative_query,
            reformulation_type=ReformulationType.NEGATIVE_FILTERING,
            confidence=0.7,
            explanation="Added negative terms to filter unwanted results",
            added_terms=["-{}".format(negative_term)],
            removed_terms=[],
            timestamp=datetime.now()
        )
    
    def get_reformulation_statistics(self) -> Dict[str, Any]:
        """Get statistics about reformulations"""
        if not self.reformulation_history:
            return {"message": "No reformulations yet"}
        
        type_counts = {}
        for reform in self.reformulation_history:
            type_name = reform.reformulation_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        avg_confidence = (
            sum(r.confidence for r in self.reformulation_history) /
            len(self.reformulation_history)
        )
        
        return {
            "total_reformulations": len(self.reformulation_history),
            "reformulation_types": type_counts,
            "average_confidence": round(avg_confidence, 2),
            "feedback_samples": len(self.feedback_history)
        }


def demonstrate_adaptive_query_reformulation():
    """Demonstrate adaptive query reformulation"""
    print("=" * 70)
    print("Adaptive Query Reformulation Agent Demonstration")
    print("=" * 70)
    
    agent = AdaptiveQueryReformulator()
    
    # Example 1: Query too specific (few results)
    print("\n1. Reformulating Too-Specific Query:")
    query1 = "red leather jacket vintage 1950s mens"
    context1 = QueryContext(
        user_id="user123",
        session_id="session456",
        previous_queries=[],
        clicked_results=[],
        rejected_results=[],
        user_preferences={}
    )
    feedback1 = SearchFeedback(
        query=query1,
        results_count=2,  # Too few
        clicked_results=[],
        time_to_click=None,
        user_satisfied=False,
        relevance_scores={}
    )
    
    reformulations1 = agent.reformulate_query(query1, context1, feedback1)
    print("Original: {}".format(query1))
    print("Found {} reformulations:".format(len(reformulations1)))
    for i, reform in enumerate(reformulations1[:3], 1):
        print("\n  {}. Type: {}".format(i, reform.reformulation_type.value))
        print("     Reformulated: {}".format(reform.reformulated_query))
        print("     Confidence: {:.2f}".format(reform.confidence))
        print("     Explanation: {}".format(reform.explanation))
    
    # Example 2: Query too general (many results)
    print("\n" + "=" * 70)
    print("2. Reformulating Too-General Query:")
    query2 = "phone"
    context2 = QueryContext(
        user_id="user123",
        session_id="session789",
        previous_queries=["best smartphones 2024"],
        clicked_results=["iPhone 15 Pro review"],
        rejected_results=["landline phones"],
        user_preferences={"domain": "mobile", "prefer_recent": True}
    )
    feedback2 = SearchFeedback(
        query=query2,
        results_count=15000,  # Too many
        clicked_results=[],
        time_to_click=None,
        user_satisfied=False,
        relevance_scores={}
    )
    
    reformulations2 = agent.reformulate_query(query2, context2, feedback2)
    print("Original: {}".format(query2))
    print("Context: Previous queries = {}".format(context2.previous_queries))
    print("         Clicked results = {}".format(context2.clicked_results))
    print("\nFound {} reformulations:".format(len(reformulations2)))
    for i, reform in enumerate(reformulations2[:3], 1):
        print("\n  {}. Type: {}".format(i, reform.reformulation_type.value))
        print("     Reformulated: {}".format(reform.reformulated_query))
        print("     Confidence: {:.2f}".format(reform.confidence))
        print("     Added terms: {}".format(reform.added_terms))
    
    # Example 3: Query with context from session
    print("\n" + "=" * 70)
    print("3. Context-Aware Reformulation:")
    query3 = "best restaurants"
    context3 = QueryContext(
        user_id="user456",
        session_id="session101",
        previous_queries=["italian food", "pasta recipes"],
        clicked_results=["italian restaurant near me"],
        rejected_results=["fast food chains"],
        user_preferences={"domain": "italian"}
    )
    
    best_reform = agent.get_best_reformulation(query3, context3)
    if best_reform:
        print("Original: {}".format(query3))
        print("Best reformulation:")
        print("  Type: {}".format(best_reform.reformulation_type.value))
        print("  Query: {}".format(best_reform.reformulated_query))
        print("  Confidence: {:.2f}".format(best_reform.confidence))
        print("  Explanation: {}".format(best_reform.explanation))
    
    # Example 4: Learning from feedback
    print("\n" + "=" * 70)
    print("4. Learning from Feedback:")
    successful_feedback = SearchFeedback(
        query="best italian restaurants",
        results_count=50,
        clicked_results=["Authentic Italian Trattoria"],
        time_to_click=2.5,
        user_satisfied=True,
        relevance_scores={"result_1": 0.95}
    )
    
    agent.learn_from_feedback(successful_feedback)
    print("Learned from successful search:")
    print("  Query: {}".format(successful_feedback.query))
    print("  User satisfied: {}".format(successful_feedback.user_satisfied))
    print("  Clicked: {}".format(successful_feedback.clicked_results))
    
    # Example 5: Statistics
    print("\n" + "=" * 70)
    print("5. Reformulation Statistics:")
    stats = agent.get_reformulation_statistics()
    print(json.dumps(stats, indent=2))
    
    # Example 6: Synonym replacement
    print("\n" + "=" * 70)
    print("6. Synonym Replacement:")
    query6 = "buy cheap car"
    context6 = QueryContext(
        user_id="user789",
        session_id="session202",
        previous_queries=[],
        clicked_results=[],
        rejected_results=[],
        user_preferences={}
    )
    
    reformulations6 = agent.reformulate_query(query6, context6)
    print("Original: {}".format(query6))
    synonym_reforms = [r for r in reformulations6 
                      if r.reformulation_type == ReformulationType.SYNONYM_REPLACEMENT]
    if synonym_reforms:
        reform = synonym_reforms[0]
        print("Synonym replacement:")
        print("  Reformulated: {}".format(reform.reformulated_query))
        print("  Removed: {}".format(reform.removed_terms))
        print("  Confidence: {:.2f}".format(reform.confidence))


if __name__ == "__main__":
    demonstrate_adaptive_query_reformulation()
