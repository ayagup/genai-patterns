"""
Pattern 125: Knowledge Extraction & Mining

Description:
    Automatically extracts structured knowledge from unstructured data sources using NER,
    relation extraction, and information extraction techniques.

Components:
    - Named Entity Recognition (NER)
    - Relation extraction
    - Information extraction pipelines
    - Knowledge base building

Use Cases:
    - Building knowledge bases from documents
    - Extracting facts from text
    - Document processing and indexing

LangChain Implementation:
    Uses LangChain for entity extraction, relation identification, and knowledge structuring.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()


# Pydantic models for structured output
class Entity(BaseModel):
    """An entity extracted from text"""
    text: str = Field(description="The entity text")
    type: str = Field(description="Entity type (PERSON, ORG, LOCATION, DATE, etc.)")
    confidence: float = Field(description="Confidence score 0-1")


class Relation(BaseModel):
    """A relation between two entities"""
    subject: str = Field(description="Subject entity")
    predicate: str = Field(description="Relation type")
    object: str = Field(description="Object entity")
    confidence: float = Field(description="Confidence score 0-1")


class KnowledgeFact(BaseModel):
    """A structured knowledge fact"""
    entities: List[Entity] = Field(description="Entities in the fact")
    relations: List[Relation] = Field(description="Relations between entities")
    source_text: str = Field(description="Original text")
    timestamp: str = Field(description="Extraction timestamp")


class KnowledgeExtractionAgent:
    """Agent for extracting structured knowledge from text"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.knowledge_base: List[KnowledgeFact] = []
        
        # Entity extraction chain
        self.entity_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at Named Entity Recognition (NER).
Extract all entities from the text and classify them into types:
PERSON, ORGANIZATION, LOCATION, DATE, EVENT, PRODUCT, TECHNOLOGY, CONCEPT, etc.

Return JSON with format:
{{
  "entities": [
    {{"text": "entity name", "type": "TYPE", "confidence": 0.95}}
  ]
}}"""),
            ("user", "Text: {text}")
        ])
        
        # Relation extraction chain
        self.relation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting semantic relations between entities.
Given a text and its entities, extract meaningful relations in the form:
Subject - Predicate - Object

Common relation types:
- works_for, founded_by, located_in, part_of
- born_in, created_on, acquired_by
- invented, developed, collaborated_with

Return JSON with format:
{{
  "relations": [
    {{"subject": "Entity1", "predicate": "relation_type", "object": "Entity2", "confidence": 0.9}}
  ]
}}"""),
            ("user", "Text: {text}\n\nEntities: {entities}")
        ])
        
        self.entity_parser = JsonOutputParser()
        self.relation_parser = JsonOutputParser()
        
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text"""
        try:
            chain = self.entity_prompt | self.llm | self.entity_parser
            result = chain.invoke({"text": text})
            
            entities = []
            for e in result.get("entities", []):
                entities.append(Entity(
                    text=e["text"],
                    type=e["type"],
                    confidence=e.get("confidence", 0.8)
                ))
            return entities
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations between entities"""
        try:
            entity_str = ", ".join([f"{e.text} ({e.type})" for e in entities])
            
            chain = self.relation_prompt | self.llm | self.relation_parser
            result = chain.invoke({"text": text, "entities": entity_str})
            
            relations = []
            for r in result.get("relations", []):
                relations.append(Relation(
                    subject=r["subject"],
                    predicate=r["predicate"],
                    object=r["object"],
                    confidence=r.get("confidence", 0.8)
                ))
            return relations
        except Exception as e:
            print(f"Relation extraction error: {e}")
            return []
    
    def extract_knowledge(self, text: str) -> KnowledgeFact:
        """Extract complete knowledge from text"""
        print(f"\n🔍 Extracting knowledge from text...")
        
        # Step 1: Extract entities
        entities = self.extract_entities(text)
        print(f"\n📌 Extracted {len(entities)} entities:")
        for e in entities:
            print(f"  - {e.text} ({e.type}) - confidence: {e.confidence:.2f}")
        
        # Step 2: Extract relations
        relations = self.extract_relations(text, entities)
        print(f"\n🔗 Extracted {len(relations)} relations:")
        for r in relations:
            print(f"  - {r.subject} --[{r.predicate}]--> {r.object}")
        
        # Create knowledge fact
        fact = KnowledgeFact(
            entities=entities,
            relations=relations,
            source_text=text,
            timestamp=datetime.now().isoformat()
        )
        
        # Add to knowledge base
        self.knowledge_base.append(fact)
        
        return fact
    
    def batch_extract(self, texts: List[str]) -> List[KnowledgeFact]:
        """Extract knowledge from multiple texts"""
        print(f"\n📚 Processing {len(texts)} documents...")
        facts = []
        
        for i, text in enumerate(texts, 1):
            print(f"\n--- Document {i}/{len(texts)} ---")
            fact = self.extract_knowledge(text)
            facts.append(fact)
        
        return facts
    
    def query_knowledge_base(self, entity_name: str) -> List[Dict[str, Any]]:
        """Query knowledge base for entity"""
        results = []
        
        for fact in self.knowledge_base:
            # Check if entity appears in this fact
            for entity in fact.entities:
                if entity_name.lower() in entity.text.lower():
                    # Find related entities and relations
                    related = {
                        "entity": entity.text,
                        "type": entity.type,
                        "relations": [],
                        "source": fact.source_text[:100] + "..."
                    }
                    
                    for rel in fact.relations:
                        if entity_name.lower() in rel.subject.lower():
                            related["relations"].append({
                                "type": rel.predicate,
                                "target": rel.object
                            })
                        elif entity_name.lower() in rel.object.lower():
                            related["relations"].append({
                                "type": f"inverse_{rel.predicate}",
                                "target": rel.subject
                            })
                    
                    if related["relations"]:
                        results.append(related)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        total_entities = sum(len(f.entities) for f in self.knowledge_base)
        total_relations = sum(len(f.relations) for f in self.knowledge_base)
        
        entity_types = {}
        for fact in self.knowledge_base:
            for entity in fact.entities:
                entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        
        relation_types = {}
        for fact in self.knowledge_base:
            for relation in fact.relations:
                relation_types[relation.predicate] = relation_types.get(relation.predicate, 0) + 1
        
        return {
            "total_documents": len(self.knowledge_base),
            "total_entities": total_entities,
            "total_relations": total_relations,
            "entity_types": entity_types,
            "relation_types": relation_types
        }


def demonstrate_knowledge_extraction():
    """Demonstrate knowledge extraction and mining"""
    print("=" * 80)
    print("Pattern 125: Knowledge Extraction & Mining")
    print("=" * 80)
    
    agent = KnowledgeExtractionAgent()
    
    # Example 1: Extract from single document
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Single Document Extraction")
    print("=" * 80)
    
    text1 = """
    OpenAI was founded in December 2015 by Sam Altman, Elon Musk, and others in San Francisco.
    The company developed ChatGPT, which was released on November 30, 2022. ChatGPT uses GPT-4,
    a large language model trained on massive amounts of text data.
    """
    
    fact1 = agent.extract_knowledge(text1)
    
    # Example 2: Batch extraction
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Batch Document Processing")
    print("=" * 80)
    
    documents = [
        """Microsoft was founded by Bill Gates and Paul Allen in 1975 in Albuquerque, New Mexico.
        The company later moved to Redmond, Washington.""",
        
        """Google was founded by Larry Page and Sergey Brin in 1998 while they were Ph.D. students
        at Stanford University. The company is headquartered in Mountain View, California.""",
        
        """Apple Inc. was co-founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
        The company introduced the iPhone in 2007, revolutionizing the smartphone industry."""
    ]
    
    facts = agent.batch_extract(documents)
    
    # Example 3: Query knowledge base
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Querying Knowledge Base")
    print("=" * 80)
    
    queries = ["OpenAI", "Bill Gates", "iPhone"]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = agent.query_knowledge_base(query)
        
        if results:
            for result in results:
                print(f"\n  Entity: {result['entity']} ({result['type']})")
                print(f"  Relations:")
                for rel in result['relations']:
                    print(f"    - {rel['type']}: {rel['target']}")
        else:
            print("  No results found")
    
    # Example 4: Knowledge base statistics
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Knowledge Base Statistics")
    print("=" * 80)
    
    stats = agent.get_statistics()
    
    print(f"\n📊 Statistics:")
    print(f"  Total Documents: {stats['total_documents']}")
    print(f"  Total Entities: {stats['total_entities']}")
    print(f"  Total Relations: {stats['total_relations']}")
    
    print(f"\n📌 Entity Types:")
    for entity_type, count in sorted(stats['entity_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {entity_type}: {count}")
    
    print(f"\n🔗 Relation Types:")
    for rel_type, count in sorted(stats['relation_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {rel_type}: {count}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Knowledge Extraction & Mining Pattern:
- Automatically extracts structured knowledge from text
- Identifies entities (people, organizations, locations, etc.)
- Discovers relations between entities
- Builds queryable knowledge base
- Enables semantic search and knowledge discovery

Key Benefits:
✓ Automated knowledge acquisition
✓ Structured representation of information
✓ Relationship discovery
✓ Scalable processing
✓ Knowledge base building
    """)


if __name__ == "__main__":
    demonstrate_knowledge_extraction()
