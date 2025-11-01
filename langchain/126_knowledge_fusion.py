"""
Pattern 126: Knowledge Fusion

Description:
    Combines knowledge from multiple heterogeneous sources, resolving conflicts
    and creating a unified, consistent knowledge representation.

Components:
    - Multi-source integration
    - Conflict detection and resolution
    - Consistency checking
    - Knowledge merging strategies

Use Cases:
    - Multi-source data integration
    - Fact verification across sources
    - Building comprehensive knowledge bases

LangChain Implementation:
    Uses LangChain for analyzing, comparing, and merging knowledge from multiple sources.
"""

import os
from typing import List, Dict, Any, Optional, Set
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()


class KnowledgeSource(BaseModel):
    """Represents a knowledge source"""
    source_id: str = Field(description="Unique source identifier")
    name: str = Field(description="Source name")
    reliability: float = Field(description="Source reliability score 0-1")
    data: Dict[str, Any] = Field(description="Knowledge data")


class FusedKnowledge(BaseModel):
    """Fused knowledge item"""
    key: str = Field(description="Knowledge key/identifier")
    value: Any = Field(description="Fused value")
    confidence: float = Field(description="Confidence in fused value")
    sources: List[str] = Field(description="Contributing sources")
    conflicts: List[Dict[str, Any]] = Field(description="Detected conflicts", default_factory=list)


class KnowledgeFusionAgent:
    """Agent for fusing knowledge from multiple sources"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.sources: List[KnowledgeSource] = []
        self.fused_knowledge: Dict[str, FusedKnowledge] = {}
        
        # Conflict resolution prompt
        self.conflict_resolution_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at resolving conflicting information from multiple sources.
Given conflicting facts from different sources with their reliability scores, determine:
1. Which fact is most likely correct
2. If a synthesis/combination is possible
3. Confidence level in the resolution

Return JSON with format:
{{
  "resolved_value": "the resolved/synthesized value",
  "confidence": 0.85,
  "reasoning": "explanation of resolution",
  "method": "voting|weighted|synthesis|newest"
}}"""),
            ("user", """Conflicting information about: {key}

Sources and values:
{conflicts}

Resolve the conflict.""")
        ])
        
        self.parser = JsonOutputParser()
    
    def add_source(self, source: KnowledgeSource):
        """Add a knowledge source"""
        self.sources.append(source)
        print(f"✓ Added source: {source.name} (reliability: {source.reliability})")
    
    def detect_conflicts(self, key: str, values: List[Dict[str, Any]]) -> bool:
        """Check if there are conflicts for a key"""
        if len(values) <= 1:
            return False
        
        # Check if all values are the same
        unique_values = set()
        for v in values:
            # Convert to string for comparison
            val_str = str(v['value'])
            unique_values.add(val_str)
        
        return len(unique_values) > 1
    
    def simple_fusion(self, key: str, values: List[Dict[str, Any]]) -> FusedKnowledge:
        """Simple fusion without conflicts (voting or consensus)"""
        if len(values) == 1:
            # Single source - use as is
            return FusedKnowledge(
                key=key,
                value=values[0]['value'],
                confidence=values[0]['reliability'],
                sources=[values[0]['source_id']]
            )
        
        # Multiple sources with same value - average confidence
        avg_confidence = sum(v['reliability'] for v in values) / len(values)
        sources = [v['source_id'] for v in values]
        
        return FusedKnowledge(
            key=key,
            value=values[0]['value'],
            confidence=avg_confidence,
            sources=sources
        )
    
    def resolve_conflict(self, key: str, values: List[Dict[str, Any]]) -> FusedKnowledge:
        """Resolve conflicting values using LLM"""
        print(f"\n⚠️  Conflict detected for '{key}'")
        
        # Prepare conflict description
        conflicts_text = ""
        conflict_list = []
        for v in values:
            conflicts_text += f"- Source '{v['source_name']}' (reliability: {v['reliability']}): {v['value']}\n"
            conflict_list.append({
                "source": v['source_name'],
                "value": v['value'],
                "reliability": v['reliability']
            })
        
        print(f"   Conflicting values:\n{conflicts_text}")
        
        try:
            # Use LLM to resolve
            chain = self.conflict_resolution_prompt | self.llm | self.parser
            result = chain.invoke({"key": key, "conflicts": conflicts_text})
            
            print(f"   Resolution: {result['resolved_value']}")
            print(f"   Method: {result['method']}, Confidence: {result['confidence']}")
            print(f"   Reasoning: {result['reasoning']}")
            
            return FusedKnowledge(
                key=key,
                value=result['resolved_value'],
                confidence=result['confidence'],
                sources=[v['source_id'] for v in values],
                conflicts=conflict_list
            )
        except Exception as e:
            print(f"   Error resolving conflict: {e}")
            # Fallback: use most reliable source
            best_source = max(values, key=lambda x: x['reliability'])
            return FusedKnowledge(
                key=key,
                value=best_source['value'],
                confidence=best_source['reliability'] * 0.8,  # Reduce confidence
                sources=[v['source_id'] for v in values],
                conflicts=conflict_list
            )
    
    def fuse_knowledge(self) -> Dict[str, FusedKnowledge]:
        """Fuse knowledge from all sources"""
        print("\n" + "=" * 80)
        print("🔄 Starting Knowledge Fusion Process")
        print("=" * 80)
        
        # Group knowledge by key
        knowledge_by_key = defaultdict(list)
        
        for source in self.sources:
            for key, value in source.data.items():
                knowledge_by_key[key].append({
                    'source_id': source.source_id,
                    'source_name': source.name,
                    'reliability': source.reliability,
                    'value': value
                })
        
        print(f"\n📊 Found {len(knowledge_by_key)} unique knowledge keys")
        
        # Fuse each key
        conflicts_detected = 0
        for key, values in knowledge_by_key.items():
            if self.detect_conflicts(key, values):
                conflicts_detected += 1
                self.fused_knowledge[key] = self.resolve_conflict(key, values)
            else:
                self.fused_knowledge[key] = self.simple_fusion(key, values)
        
        print(f"\n✓ Fusion complete!")
        print(f"  - Total keys: {len(self.fused_knowledge)}")
        print(f"  - Conflicts resolved: {conflicts_detected}")
        
        return self.fused_knowledge
    
    def query(self, key: str) -> Optional[FusedKnowledge]:
        """Query fused knowledge"""
        return self.fused_knowledge.get(key)
    
    def get_high_confidence_facts(self, threshold: float = 0.8) -> List[FusedKnowledge]:
        """Get facts with confidence above threshold"""
        return [
            fact for fact in self.fused_knowledge.values()
            if fact.confidence >= threshold
        ]
    
    def get_conflicted_facts(self) -> List[FusedKnowledge]:
        """Get facts that had conflicts"""
        return [
            fact for fact in self.fused_knowledge.values()
            if fact.conflicts
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fusion statistics"""
        total_facts = len(self.fused_knowledge)
        conflicted = len(self.get_conflicted_facts())
        high_confidence = len(self.get_high_confidence_facts())
        
        avg_confidence = sum(f.confidence for f in self.fused_knowledge.values()) / total_facts if total_facts > 0 else 0
        
        return {
            "total_sources": len(self.sources),
            "total_facts": total_facts,
            "conflicted_facts": conflicted,
            "high_confidence_facts": high_confidence,
            "average_confidence": avg_confidence
        }


def demonstrate_knowledge_fusion():
    """Demonstrate knowledge fusion from multiple sources"""
    print("=" * 80)
    print("Pattern 126: Knowledge Fusion")
    print("=" * 80)
    
    agent = KnowledgeFusionAgent()
    
    # Example 1: Add multiple knowledge sources
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Adding Knowledge Sources")
    print("=" * 80)
    
    # Source 1: Wikipedia (high reliability)
    wikipedia = KnowledgeSource(
        source_id="wikipedia",
        name="Wikipedia",
        reliability=0.85,
        data={
            "eiffel_tower_height": "330 meters",
            "eiffel_tower_built": "1889",
            "eiffel_tower_location": "Paris, France",
            "mount_everest_height": "8,849 meters",
            "python_creator": "Guido van Rossum"
        }
    )
    
    # Source 2: Tourist website (medium reliability)
    tourist_site = KnowledgeSource(
        source_id="tourist_site",
        name="Tourism Website",
        reliability=0.65,
        data={
            "eiffel_tower_height": "324 meters",  # Conflict!
            "eiffel_tower_built": "1889",
            "eiffel_tower_location": "Paris, France",
            "eiffel_tower_visitors": "7 million per year"
        }
    )
    
    # Source 3: News article (medium reliability)
    news = KnowledgeSource(
        source_id="news",
        name="News Article",
        reliability=0.70,
        data={
            "eiffel_tower_height": "330 meters including antenna",  # Conflict!
            "mount_everest_height": "8,848.86 meters",  # Conflict!
            "python_creator": "Guido van Rossum",
            "python_first_release": "1991"
        }
    )
    
    agent.add_source(wikipedia)
    agent.add_source(tourist_site)
    agent.add_source(news)
    
    # Example 2: Fuse knowledge
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Fusing Knowledge")
    print("=" * 80)
    
    fused = agent.fuse_knowledge()
    
    # Example 3: Query fused knowledge
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Querying Fused Knowledge")
    print("=" * 80)
    
    queries = [
        "eiffel_tower_height",
        "eiffel_tower_location",
        "python_creator",
        "mount_everest_height"
    ]
    
    for query in queries:
        result = agent.query(query)
        if result:
            print(f"\n🔍 Query: {query}")
            print(f"   Value: {result.value}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Sources: {', '.join(result.sources)}")
            if result.conflicts:
                print(f"   ⚠️  Had conflicts from {len(result.conflicts)} sources")
    
    # Example 4: High confidence facts
    print("\n" + "=" * 80)
    print("EXAMPLE 4: High Confidence Facts")
    print("=" * 80)
    
    high_conf = agent.get_high_confidence_facts(threshold=0.75)
    print(f"\n✓ Found {len(high_conf)} high-confidence facts (>0.75):\n")
    for fact in high_conf:
        print(f"  - {fact.key}: {fact.value} (confidence: {fact.confidence:.2f})")
    
    # Example 5: Conflicted facts
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Facts with Conflicts")
    print("=" * 80)
    
    conflicted = agent.get_conflicted_facts()
    print(f"\n⚠️  Found {len(conflicted)} facts with conflicts:\n")
    for fact in conflicted:
        print(f"  - {fact.key}:")
        print(f"      Resolved to: {fact.value}")
        print(f"      Original values:")
        for conflict in fact.conflicts:
            print(f"        • {conflict['source']}: {conflict['value']}")
    
    # Example 6: Statistics
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Fusion Statistics")
    print("=" * 80)
    
    stats = agent.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"  Total Sources: {stats['total_sources']}")
    print(f"  Total Facts: {stats['total_facts']}")
    print(f"  Conflicted Facts: {stats['conflicted_facts']}")
    print(f"  High Confidence Facts: {stats['high_confidence_facts']}")
    print(f"  Average Confidence: {stats['average_confidence']:.2f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Knowledge Fusion Pattern:
- Integrates knowledge from multiple heterogeneous sources
- Detects and resolves conflicting information
- Uses source reliability scores for weighting
- Creates unified, consistent knowledge representation
- Maintains provenance and conflict information

Conflict Resolution Strategies:
✓ Voting (majority wins)
✓ Weighted by source reliability
✓ LLM-based synthesis
✓ Temporal (newest information)
✓ Consensus building

Key Benefits:
✓ Comprehensive knowledge coverage
✓ Improved accuracy through verification
✓ Conflict awareness and resolution
✓ Source tracking and attribution
✓ Confidence scoring
    """)


if __name__ == "__main__":
    demonstrate_knowledge_fusion()
