"""
Analogical Reasoning Pattern Implementation

This module demonstrates analogical reasoning where an agent uses similar past 
problems to solve new ones. It includes case retrieval, mapping between source
and target domains, and adaptation of solutions.

Key Components:
- Case-based reasoning with similarity matching
- Structural mapping between source and target
- Solution adaptation and validation
- Learning from analogical successes and failures
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Tuple
from enum import Enum
import math
import random
import json


class SimilarityType(Enum):
    """Types of similarity for analogical reasoning"""
    SURFACE = "surface"          # Superficial similarities
    STRUCTURAL = "structural"    # Relational structure similarities
    FUNCTIONAL = "functional"    # Goal/purpose similarities
    CAUSAL = "causal"           # Cause-effect relationships
    PRAGMATIC = "pragmatic"     # Context and constraints


@dataclass
class Concept:
    """Represents a concept in a domain"""
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: List[str] = field(default_factory=list)
    category: str = ""
    
    def __str__(self):
        return self.name


@dataclass
class Relation:
    """Represents a relationship between concepts"""
    name: str
    source: str
    target: str
    strength: float = 1.0
    relation_type: str = "general"
    
    def __str__(self):
        return f"{self.source} --{self.name}--> {self.target}"


@dataclass
class Domain:
    """Represents a problem domain with concepts and relations"""
    name: str
    concepts: Dict[str, Concept] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    
    def add_concept(self, concept: Concept):
        """Add a concept to the domain"""
        self.concepts[concept.name] = concept
    
    def add_relation(self, relation: Relation):
        """Add a relation to the domain"""
        self.relations.append(relation)
    
    def get_concept_relations(self, concept_name: str) -> List[Relation]:
        """Get all relations involving a concept"""
        return [r for r in self.relations if r.source == concept_name or r.target == concept_name]


@dataclass
class AnalogicalCase:
    """Represents a case for analogical reasoning"""
    case_id: str
    domain: Domain
    problem_description: str
    solution: str
    solution_steps: List[str] = field(default_factory=list)
    success_rate: float = 1.0
    usage_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.case_id:
            self.case_id = f"case_{random.randint(1000, 9999)}"


@dataclass
class Mapping:
    """Represents a mapping between source and target domains"""
    concept_mappings: Dict[str, str] = field(default_factory=dict)  # source -> target
    relation_mappings: Dict[str, str] = field(default_factory=dict)  # source -> target
    confidence: float = 0.0
    similarity_scores: Dict[SimilarityType, float] = field(default_factory=dict)
    
    def add_concept_mapping(self, source_concept: str, target_concept: str):
        """Add a concept mapping"""
        self.concept_mappings[source_concept] = target_concept
    
    def add_relation_mapping(self, source_relation: str, target_relation: str):
        """Add a relation mapping"""
        self.relation_mappings[source_relation] = target_relation


class SimilarityCalculator:
    """Calculates similarity between domains and cases"""
    
    def __init__(self):
        self.similarity_weights = {
            SimilarityType.SURFACE: 0.1,
            SimilarityType.STRUCTURAL: 0.3,
            SimilarityType.FUNCTIONAL: 0.25,
            SimilarityType.CAUSAL: 0.25,
            SimilarityType.PRAGMATIC: 0.1
        }
    
    def calculate_concept_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """Calculate similarity between two concepts"""
        similarities = []
        
        # Name similarity (surface)
        name_sim = self._string_similarity(concept1.name, concept2.name)
        similarities.append(name_sim * 0.3)
        
        # Category similarity
        if concept1.category and concept2.category:
            cat_sim = 1.0 if concept1.category == concept2.category else 0.0
            similarities.append(cat_sim * 0.4)
        
        # Attribute similarity
        attr_sim = self._attribute_similarity(concept1.attributes, concept2.attributes)
        similarities.append(attr_sim * 0.3)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def calculate_structural_similarity(self, domain1: Domain, domain2: Domain) -> float:
        """Calculate structural similarity between domains"""
        if not domain1.relations or not domain2.relations:
            return 0.0
        
        # Compare relation patterns
        relations1 = set((r.name, r.source, r.target) for r in domain1.relations)
        relations2 = set((r.name, r.source, r.target) for r in domain2.relations)
        
        if not relations1 and not relations2:
            return 1.0
        
        intersection = len(relations1.intersection(relations2))
        union = len(relations1.union(relations2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_functional_similarity(self, case1: AnalogicalCase, 
                                      case2: AnalogicalCase) -> float:
        """Calculate functional similarity between cases"""
        # Compare goals
        goals1 = set(case1.domain.goals)
        goals2 = set(case2.domain.goals)
        
        if not goals1 and not goals2:
            return 1.0
        
        goal_intersection = len(goals1.intersection(goals2))
        goal_union = len(goals1.union(goals2))
        goal_similarity = goal_intersection / goal_union if goal_union > 0 else 0.0
        
        # Compare problem descriptions
        desc_similarity = self._string_similarity(case1.problem_description, case2.problem_description)
        
        return (goal_similarity + desc_similarity) / 2
    
    def calculate_case_similarity(self, source_case: AnalogicalCase, 
                                target_case: AnalogicalCase) -> Dict[SimilarityType, float]:
        """Calculate comprehensive similarity between cases"""
        similarities = {}
        
        # Surface similarity
        similarities[SimilarityType.SURFACE] = self._string_similarity(
            source_case.problem_description, target_case.problem_description
        )
        
        # Structural similarity
        similarities[SimilarityType.STRUCTURAL] = self.calculate_structural_similarity(
            source_case.domain, target_case.domain
        )
        
        # Functional similarity
        similarities[SimilarityType.FUNCTIONAL] = self.calculate_functional_similarity(
            source_case, target_case
        )
        
        # Causal similarity (simplified)
        causal_relations1 = [r for r in source_case.domain.relations if "cause" in r.name.lower()]
        causal_relations2 = [r for r in target_case.domain.relations if "cause" in r.name.lower()]
        similarities[SimilarityType.CAUSAL] = len(causal_relations1) / max(len(causal_relations2), 1)
        similarities[SimilarityType.CAUSAL] = min(similarities[SimilarityType.CAUSAL], 1.0)
        
        # Pragmatic similarity (context)
        context_sim = self._context_similarity(source_case.context, target_case.context)
        similarities[SimilarityType.PRAGMATIC] = context_sim
        
        return similarities
    
    def calculate_overall_similarity(self, similarities: Dict[SimilarityType, float]) -> float:
        """Calculate weighted overall similarity"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for sim_type, score in similarities.items():
            weight = self.similarity_weights.get(sim_type, 0.0)
            total_weighted_score += score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Jaccard similarity of words"""
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _attribute_similarity(self, attrs1: Dict[str, Any], attrs2: Dict[str, Any]) -> float:
        """Calculate similarity between attribute dictionaries"""
        if not attrs1 and not attrs2:
            return 1.0
        
        common_keys = set(attrs1.keys()).intersection(set(attrs2.keys()))
        all_keys = set(attrs1.keys()).union(set(attrs2.keys()))
        
        if not all_keys:
            return 1.0
        
        matches = 0
        for key in common_keys:
            if attrs1[key] == attrs2[key]:
                matches += 1
        
        return matches / len(all_keys)
    
    def _context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between context dictionaries"""
        return self._attribute_similarity(context1, context2)


class AnalogicalMapper:
    """Maps between source and target domains"""
    
    def __init__(self, similarity_calculator: SimilarityCalculator):
        self.similarity_calculator = similarity_calculator
    
    def create_mapping(self, source_domain: Domain, target_domain: Domain) -> Mapping:
        """Create a mapping between source and target domains"""
        mapping = Mapping()
        
        # Map concepts
        concept_mappings = self._map_concepts(source_domain, target_domain)
        mapping.concept_mappings = concept_mappings
        
        # Map relations
        relation_mappings = self._map_relations(source_domain, target_domain, concept_mappings)
        mapping.relation_mappings = relation_mappings
        
        # Calculate mapping confidence
        mapping.confidence = self._calculate_mapping_confidence(mapping, source_domain, target_domain)
        
        return mapping
    
    def _map_concepts(self, source_domain: Domain, target_domain: Domain) -> Dict[str, str]:
        """Map concepts between domains"""
        mappings = {}
        used_targets = set()
        
        # Calculate all pairwise similarities
        concept_similarities = []
        
        for source_name, source_concept in source_domain.concepts.items():
            for target_name, target_concept in target_domain.concepts.items():
                similarity = self.similarity_calculator.calculate_concept_similarity(
                    source_concept, target_concept
                )
                concept_similarities.append((similarity, source_name, target_name))
        
        # Sort by similarity and create mappings
        concept_similarities.sort(reverse=True)
        
        for similarity, source_name, target_name in concept_similarities:
            if (source_name not in mappings and 
                target_name not in used_targets and 
                similarity > 0.3):  # Threshold for acceptable mapping
                mappings[source_name] = target_name
                used_targets.add(target_name)
        
        return mappings
    
    def _map_relations(self, source_domain: Domain, target_domain: Domain, 
                      concept_mappings: Dict[str, str]) -> Dict[str, str]:
        """Map relations between domains based on concept mappings"""
        mappings = {}
        
        for source_rel in source_domain.relations:
            best_match = None
            best_score = 0.0
            
            for target_rel in target_domain.relations:
                score = self._relation_mapping_score(source_rel, target_rel, concept_mappings)
                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = target_rel
            
            if best_match:
                mappings[f"{source_rel.source}-{source_rel.name}-{source_rel.target}"] = \
                    f"{best_match.source}-{best_match.name}-{best_match.target}"
        
        return mappings
    
    def _relation_mapping_score(self, source_rel: Relation, target_rel: Relation, 
                               concept_mappings: Dict[str, str]) -> float:
        """Calculate score for mapping two relations"""
        score = 0.0
        
        # Relation name similarity
        name_sim = self.similarity_calculator._string_similarity(source_rel.name, target_rel.name)
        score += name_sim * 0.4
        
        # Source concept mapping
        if (source_rel.source in concept_mappings and 
            concept_mappings[source_rel.source] == target_rel.source):
            score += 0.3
        
        # Target concept mapping
        if (source_rel.target in concept_mappings and 
            concept_mappings[source_rel.target] == target_rel.target):
            score += 0.3
        
        return score
    
    def _calculate_mapping_confidence(self, mapping: Mapping, 
                                    source_domain: Domain, target_domain: Domain) -> float:
        """Calculate confidence in the mapping"""
        concept_coverage = len(mapping.concept_mappings) / max(len(source_domain.concepts), 1)
        relation_coverage = len(mapping.relation_mappings) / max(len(source_domain.relations), 1)
        
        return (concept_coverage + relation_coverage) / 2


class SolutionAdapter:
    """Adapts solutions from source to target domain"""
    
    def __init__(self):
        self.adaptation_strategies = [
            self._direct_substitution,
            self._structural_adaptation,
            self._constraint_adaptation,
            self._goal_adaptation
        ]
    
    def adapt_solution(self, source_case: AnalogicalCase, mapping: Mapping, 
                      target_problem: str) -> Tuple[str, List[str], float]:
        """Adapt a solution from source to target domain"""
        adapted_solution = source_case.solution
        adapted_steps = source_case.solution_steps.copy()
        adaptation_confidence = mapping.confidence
        
        # Apply adaptation strategies
        for strategy in self.adaptation_strategies:
            adapted_solution, adapted_steps = strategy(
                adapted_solution, adapted_steps, mapping
            )
        
        # Adjust confidence based on adaptation complexity
        adaptation_penalty = len(mapping.concept_mappings) * 0.05
        adaptation_confidence = max(0.0, adaptation_confidence - adaptation_penalty)
        
        return adapted_solution, adapted_steps, adaptation_confidence
    
    def _direct_substitution(self, solution: str, steps: List[str], 
                           mapping: Mapping) -> Tuple[str, List[str]]:
        """Apply direct concept substitution"""
        adapted_solution = solution
        adapted_steps = steps.copy()
        
        for source_concept, target_concept in mapping.concept_mappings.items():
            adapted_solution = adapted_solution.replace(source_concept, target_concept)
            adapted_steps = [step.replace(source_concept, target_concept) for step in adapted_steps]
        
        return adapted_solution, adapted_steps
    
    def _structural_adaptation(self, solution: str, steps: List[str], 
                             mapping: Mapping) -> Tuple[str, List[str]]:
        """Adapt based on structural mappings"""
        # This would involve more complex structural transformations
        # For now, we'll add a note about structural considerations
        if mapping.relation_mappings:
            adapted_solution = solution + " (adapted for different structural relationships)"
            adapted_steps = steps + ["Consider structural differences in target domain"]
        else:
            adapted_solution = solution
            adapted_steps = steps
        
        return adapted_solution, adapted_steps
    
    def _constraint_adaptation(self, solution: str, steps: List[str], 
                             mapping: Mapping) -> Tuple[str, List[str]]:
        """Adapt based on different constraints"""
        # Add constraint consideration step
        adapted_steps = steps + ["Verify solution satisfies target domain constraints"]
        return solution, adapted_steps
    
    def _goal_adaptation(self, solution: str, steps: List[str], 
                       mapping: Mapping) -> Tuple[str, List[str]]:
        """Adapt based on different goals"""
        # Add goal verification step
        adapted_steps = steps + ["Ensure solution achieves target domain goals"]
        return solution, adapted_steps


class AnalogicalReasoningAgent:
    """Main agent for analogical reasoning"""
    
    def __init__(self):
        self.case_library: List[AnalogicalCase] = []
        self.similarity_calculator = SimilarityCalculator()
        self.mapper = AnalogicalMapper(self.similarity_calculator)
        self.adapter = SolutionAdapter()
        self.reasoning_history: List[Dict[str, Any]] = []
    
    def add_case(self, case: AnalogicalCase):
        """Add a case to the case library"""
        self.case_library.append(case)
    
    def retrieve_similar_cases(self, target_case: AnalogicalCase, 
                             top_k: int = 3) -> List[Tuple[AnalogicalCase, float]]:
        """Retrieve the most similar cases from the library"""
        similarities = []
        
        for source_case in self.case_library:
            similarity_scores = self.similarity_calculator.calculate_case_similarity(
                source_case, target_case
            )
            overall_similarity = self.similarity_calculator.calculate_overall_similarity(
                similarity_scores
            )
            similarities.append((source_case, overall_similarity))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def solve_by_analogy(self, target_problem: str, target_domain: Domain) -> Dict[str, Any]:
        """Solve a problem using analogical reasoning"""
        print(f"\n🧠 Solving by Analogy: {target_problem}")
        print("=" * 60)
        
        # Create target case
        target_case = AnalogicalCase(
            case_id="target",
            domain=target_domain,
            problem_description=target_problem,
            solution="",  # To be determined
            context={"problem_type": "new"}
        )
        
        # Retrieve similar cases
        print("🔍 Retrieving similar cases...")
        similar_cases = self.retrieve_similar_cases(target_case)
        
        if not similar_cases:
            return {"error": "No similar cases found in library"}
        
        print(f"Found {len(similar_cases)} similar cases:")
        for i, (case, similarity) in enumerate(similar_cases, 1):
            print(f"  {i}. {case.case_id}: {similarity:.3f} similarity")
            print(f"     Problem: {case.problem_description[:60]}...")
        
        # Use the most similar case
        best_case, best_similarity = similar_cases[0]
        print(f"\n🎯 Using best match: {best_case.case_id} (similarity: {best_similarity:.3f})")
        
        # Create mapping
        print("🗺️ Creating domain mapping...")
        mapping = self.mapper.create_mapping(best_case.domain, target_domain)
        
        print(f"Mapping confidence: {mapping.confidence:.3f}")
        print(f"Concept mappings: {len(mapping.concept_mappings)}")
        print(f"Relation mappings: {len(mapping.relation_mappings)}")
        
        if mapping.concept_mappings:
            print("Key concept mappings:")
            for source, target in list(mapping.concept_mappings.items())[:3]:
                print(f"  {source} → {target}")
        
        # Adapt solution
        print("\n🔧 Adapting solution...")
        adapted_solution, adapted_steps, adaptation_confidence = self.adapter.adapt_solution(
            best_case, mapping, target_problem
        )
        
        # Create result
        result = {
            "target_problem": target_problem,
            "source_case": {
                "id": best_case.case_id,
                "problem": best_case.problem_description,
                "solution": best_case.solution
            },
            "similarity_score": best_similarity,
            "mapping_confidence": mapping.confidence,
            "adaptation_confidence": adaptation_confidence,
            "adapted_solution": adapted_solution,
            "solution_steps": adapted_steps,
            "concept_mappings": mapping.concept_mappings,
            "overall_confidence": (best_similarity + mapping.confidence + adaptation_confidence) / 3
        }
        
        # Store in history
        self.reasoning_history.append(result)
        
        print(f"\n✅ Solution adapted with confidence: {result['overall_confidence']:.3f}")
        print(f"Adapted solution: {adapted_solution}")
        
        return result
    
    def learn_from_feedback(self, case_id: str, success: bool):
        """Learn from feedback on analogical reasoning"""
        case = next((c for c in self.case_library if c.case_id == case_id), None)
        if case:
            case.usage_count += 1
            if success:
                case.success_rate = (case.success_rate * (case.usage_count - 1) + 1.0) / case.usage_count
            else:
                case.success_rate = (case.success_rate * (case.usage_count - 1) + 0.0) / case.usage_count
            
            print(f"📈 Updated case {case_id}: success_rate={case.success_rate:.2f}, usage_count={case.usage_count}")
    
    def get_case_library_summary(self) -> Dict[str, Any]:
        """Get summary of the case library"""
        if not self.case_library:
            return {"message": "Case library is empty"}
        
        domains = set(case.domain.name for case in self.case_library)
        avg_success_rate = sum(case.success_rate for case in self.case_library) / len(self.case_library)
        total_usage = sum(case.usage_count for case in self.case_library)
        
        return {
            "total_cases": len(self.case_library),
            "domains": list(domains),
            "average_success_rate": avg_success_rate,
            "total_usage": total_usage,
            "most_used_case": max(self.case_library, key=lambda x: x.usage_count).case_id
        }


def create_sample_cases() -> List[AnalogicalCase]:
    """Create sample cases for demonstration"""
    cases = []
    
    # Case 1: Solar system analogy for atomic structure
    solar_domain = Domain("solar_system")
    solar_domain.add_concept(Concept("sun", {"mass": "large", "position": "center"}, category="star"))
    solar_domain.add_concept(Concept("planet", {"mass": "small", "motion": "orbital"}, category="celestial_body"))
    solar_domain.add_relation(Relation("orbits", "planet", "sun", 1.0, "gravitational"))
    solar_domain.add_relation(Relation("attracts", "sun", "planet", 1.0, "gravitational"))
    solar_domain.goals = ["maintain stable orbits", "balance forces"]
    
    cases.append(AnalogicalCase(
        case_id="solar_system",
        domain=solar_domain,
        problem_description="Explain the structure and stability of the solar system",
        solution="Planets orbit the sun due to gravitational attraction, maintaining stable paths through balanced forces",
        solution_steps=[
            "Identify central massive body (sun)",
            "Identify orbiting bodies (planets)",
            "Analyze gravitational forces",
            "Verify force balance for stable orbits"
        ],
        success_rate=0.9
    ))
    
    # Case 2: Water flow analogy for electrical circuits
    water_domain = Domain("water_flow")
    water_domain.add_concept(Concept("pump", {"function": "pressure_source"}, category="device"))
    water_domain.add_concept(Concept("pipe", {"resistance": "variable"}, category="conduit"))
    water_domain.add_concept(Concept("valve", {"control": "flow_rate"}, category="controller"))
    water_domain.add_relation(Relation("flows_through", "water", "pipe", 1.0, "physical"))
    water_domain.add_relation(Relation("powered_by", "water", "pump", 1.0, "causal"))
    water_domain.goals = ["control flow rate", "manage pressure"]
    
    cases.append(AnalogicalCase(
        case_id="water_flow",
        domain=water_domain,
        problem_description="Design a water flow system with controllable flow rate",
        solution="Use pump for pressure, pipes for conduction, and valves for flow control",
        solution_steps=[
            "Install pressure source (pump)",
            "Connect conducting medium (pipes)",
            "Add flow control mechanisms (valves)",
            "Test and adjust flow rates"
        ],
        success_rate=0.85
    ))
    
    # Case 3: Library analogy for database systems
    library_domain = Domain("library")
    library_domain.add_concept(Concept("book", {"content": "information"}, category="resource"))
    library_domain.add_concept(Concept("catalog", {"function": "indexing"}, category="system"))
    library_domain.add_concept(Concept("librarian", {"role": "facilitator"}, category="agent"))
    library_domain.add_relation(Relation("indexes", "catalog", "book", 1.0, "organizational"))
    library_domain.add_relation(Relation("retrieves", "librarian", "book", 1.0, "functional"))
    library_domain.goals = ["organize information", "enable quick retrieval"]
    
    cases.append(AnalogicalCase(
        case_id="library_system",
        domain=library_domain,
        problem_description="Organize and provide access to large collection of information",
        solution="Use systematic cataloging and trained staff to organize and retrieve information efficiently",
        solution_steps=[
            "Catalog all resources systematically",
            "Create searchable index system",
            "Train staff for retrieval assistance",
            "Implement access procedures"
        ],
        success_rate=0.95
    ))
    
    return cases


def main():
    """Demonstration of the Analogical Reasoning pattern"""
    print("🧠 Analogical Reasoning Pattern Demonstration")
    print("=" * 80)
    print("This demonstrates solving problems using analogies:")
    print("- Case retrieval based on similarity")
    print("- Domain mapping between source and target")
    print("- Solution adaptation and validation")
    print("- Learning from analogical successes")
    
    # Create agent and populate with sample cases
    agent = AnalogicalReasoningAgent()
    sample_cases = create_sample_cases()
    
    for case in sample_cases:
        agent.add_case(case)
    
    print(f"\n📚 Loaded {len(sample_cases)} cases into case library")
    
    # Test problem: Understanding atomic structure
    print(f"\n\n🎯 Test Problem 1: Atomic Structure")
    print("=" * 80)
    
    atomic_domain = Domain("atomic_structure")
    atomic_domain.add_concept(Concept("nucleus", {"mass": "large", "position": "center", "charge": "positive"}, category="particle"))
    atomic_domain.add_concept(Concept("electron", {"mass": "small", "motion": "orbital", "charge": "negative"}, category="particle"))
    atomic_domain.add_relation(Relation("orbits", "electron", "nucleus", 1.0, "electromagnetic"))
    atomic_domain.add_relation(Relation("attracts", "nucleus", "electron", 1.0, "electromagnetic"))
    atomic_domain.goals = ["maintain stable structure", "balance forces"]
    
    result1 = agent.solve_by_analogy(
        "Explain the structure and stability of an atom",
        atomic_domain
    )
    
    print(f"\n📊 Result Summary:")
    print(f"Overall confidence: {result1['overall_confidence']:.3f}")
    print(f"Adapted solution: {result1['adapted_solution']}")
    
    # Simulate feedback
    agent.learn_from_feedback(result1['source_case']['id'], True)
    
    # Test problem: Database design
    print(f"\n\n🎯 Test Problem 2: Database Design")
    print("=" * 80)
    
    database_domain = Domain("database")
    database_domain.add_concept(Concept("record", {"content": "data"}, category="resource"))
    database_domain.add_concept(Concept("index", {"function": "lookup"}, category="system"))
    database_domain.add_concept(Concept("query_engine", {"role": "facilitator"}, category="agent"))
    database_domain.add_relation(Relation("indexes", "index", "record", 1.0, "organizational"))
    database_domain.add_relation(Relation("retrieves", "query_engine", "record", 1.0, "functional"))
    database_domain.goals = ["organize data", "enable fast queries"]
    
    result2 = agent.solve_by_analogy(
        "Design an efficient database system for storing and retrieving large amounts of data",
        database_domain
    )
    
    print(f"\n📊 Result Summary:")
    print(f"Overall confidence: {result2['overall_confidence']:.3f}")
    print(f"Adapted solution: {result2['adapted_solution']}")
    
    # Simulate feedback
    agent.learn_from_feedback(result2['source_case']['id'], True)
    
    # Case library summary
    print(f"\n📈 Case Library Summary:")
    summary = agent.get_case_library_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n\n🎯 Key Analogical Reasoning Features Demonstrated:")
    print("✅ Case-based similarity matching")
    print("✅ Multi-type similarity calculation (surface, structural, functional)")
    print("✅ Domain mapping between source and target")
    print("✅ Solution adaptation strategies")
    print("✅ Confidence estimation for analogies")
    print("✅ Learning from feedback")
    print("✅ Case library management")
    print("✅ Structural and functional alignment")


if __name__ == "__main__":
    main()