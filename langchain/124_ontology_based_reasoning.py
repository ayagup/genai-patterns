"""
Pattern 124: Ontology-Based Reasoning

Description:
    Uses formal ontologies (OWL, RDF, SKOS) for domain knowledge
    representation and semantic reasoning.

Components:
    - Ontology definition
    - Class hierarchies
    - Property definitions
    - Inference rules

Use Cases:
    - Semantic reasoning
    - Domain expertise
    - Formal logic

LangChain Implementation:
    Implements ontology concepts with LLM-based reasoning over formal structures.
"""

import os
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


@dataclass
class OntologyClass:
    """Represents a class in the ontology."""
    name: str
    parent: str = None
    properties: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = []


@dataclass
class OntologyProperty:
    """Represents a property in the ontology."""
    name: str
    domain: str  # Class it belongs to
    range: str   # Type of value
    description: str = ""


@dataclass
class OntologyInstance:
    """Represents an instance of a class."""
    id: str
    class_name: str
    properties: Dict[str, Any]


class Ontology:
    """Formal ontology for domain knowledge."""
    
    def __init__(self, name: str):
        self.name = name
        self.classes: Dict[str, OntologyClass] = {}
        self.properties: Dict[str, OntologyProperty] = {}
        self.instances: Dict[str, OntologyInstance] = {}
        
    def add_class(self, ontology_class: OntologyClass):
        """Add a class to the ontology."""
        self.classes[ontology_class.name] = ontology_class
        
    def add_property(self, ontology_property: OntologyProperty):
        """Add a property to the ontology."""
        self.properties[ontology_property.name] = ontology_property
        
    def add_instance(self, instance: OntologyInstance):
        """Add an instance to the ontology."""
        self.instances[instance.id] = instance
        
    def get_subclasses(self, class_name: str) -> List[str]:
        """Get all subclasses of a class."""
        subclasses = []
        for cls_name, cls in self.classes.items():
            if cls.parent == class_name:
                subclasses.append(cls_name)
                # Recursively get subclasses
                subclasses.extend(self.get_subclasses(cls_name))
        return subclasses
    
    def get_superclasses(self, class_name: str) -> List[str]:
        """Get all superclasses of a class."""
        superclasses = []
        current = self.classes.get(class_name)
        while current and current.parent:
            superclasses.append(current.parent)
            current = self.classes.get(current.parent)
        return superclasses
    
    def is_subclass_of(self, class1: str, class2: str) -> bool:
        """Check if class1 is a subclass of class2."""
        return class2 in self.get_superclasses(class1)
    
    def get_instances_of_class(self, class_name: str, include_subclasses: bool = True) -> List[OntologyInstance]:
        """Get all instances of a class."""
        instances = [inst for inst in self.instances.values() if inst.class_name == class_name]
        
        if include_subclasses:
            subclasses = self.get_subclasses(class_name)
            for subclass in subclasses:
                instances.extend([inst for inst in self.instances.values() if inst.class_name == subclass])
        
        return instances


class OntologyReasoningAgent:
    """Agent that performs reasoning using ontologies."""
    
    def __init__(self, ontology: Ontology, model_name: str = "gpt-4"):
        self.ontology = ontology
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        
    def classify_instance(self, instance_description: str) -> str:
        """Classify an instance into ontology classes."""
        # Get ontology structure
        ontology_structure = "Ontology Classes:\n"
        for cls_name, cls in self.ontology.classes.items():
            ontology_structure += f"- {cls_name}"
            if cls.parent:
                ontology_structure += f" (subclass of {cls.parent})"
            ontology_structure += f": {cls.description}\n"
        
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", "Classify this instance into the most appropriate ontology class."),
            ("user", """{ontology_structure}

Instance to classify: {instance}

Most appropriate class:""")
        ])
        
        chain = classification_prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "ontology_structure": ontology_structure,
            "instance": instance_description
        })
    
    def infer_properties(self, instance_id: str) -> Dict[str, Any]:
        """Infer properties of an instance based on ontology."""
        instance = self.ontology.instances.get(instance_id)
        if not instance:
            return {}
        
        # Get class hierarchy
        superclasses = self.ontology.get_superclasses(instance.class_name)
        
        # Collect inherited properties
        inherited_properties = {}
        for superclass in superclasses:
            cls = self.ontology.classes.get(superclass)
            if cls:
                for prop in cls.properties:
                    if prop not in instance.properties:
                        inherited_properties[prop] = f"inherited from {superclass}"
        
        inference_prompt = ChatPromptTemplate.from_messages([
            ("system", "Infer additional properties for this instance based on ontology."),
            ("user", """Instance: {instance_id}
Class: {class_name}
Superclasses: {superclasses}
Current Properties: {current_props}

Infer likely values for these properties:
{inherited_props}""")
        ])
        
        chain = inference_prompt | self.llm | StrOutputParser()
        inferences = chain.invoke({
            "instance_id": instance_id,
            "class_name": instance.class_name,
            "superclasses": ", ".join(superclasses),
            "current_props": str(instance.properties),
            "inherited_props": ", ".join(inherited_properties.keys())
        })
        
        return {"inferences": inferences, "inherited": inherited_properties}
    
    def answer_semantic_query(self, query: str) -> str:
        """Answer semantic queries using ontology."""
        # Serialize ontology info
        ontology_info = f"Ontology: {self.ontology.name}\n\n"
        ontology_info += "Classes:\n"
        for cls_name, cls in self.ontology.classes.items():
            ontology_info += f"- {cls_name}: {cls.description}\n"
        
        ontology_info += "\nInstances:\n"
        for inst_id, inst in list(self.ontology.instances.items())[:10]:  # Limit to 10
            ontology_info += f"- {inst_id} ({inst.class_name}): {inst.properties}\n"
        
        query_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer this semantic query using the ontology knowledge."),
            ("user", """{ontology_info}

Query: {query}

Answer:""")
        ])
        
        chain = query_prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "ontology_info": ontology_info,
            "query": query
        })
    
    def check_consistency(self) -> str:
        """Check ontology consistency."""
        consistency_prompt = ChatPromptTemplate.from_messages([
            ("system", "Check this ontology for logical consistency and conflicts."),
            ("user", """Ontology Structure:
{structure}

Check for:
1. Circular class hierarchies
2. Property domain/range conflicts
3. Instance classification errors
4. Missing definitions""")
        ])
        
        structure = ""
        for cls_name, cls in self.ontology.classes.items():
            structure += f"{cls_name} -> {cls.parent}\n"
        
        chain = consistency_prompt | self.llm | StrOutputParser()
        return chain.invoke({"structure": structure})


def demonstrate_ontology_based_reasoning():
    """Demonstrate ontology-based reasoning pattern."""
    print("=== Ontology-Based Reasoning Pattern ===\n")
    
    # Create ontology
    print("1. Building Domain Ontology")
    print("-" * 50)
    
    ontology = Ontology("VehicleOntology")
    
    # Define class hierarchy
    ontology.add_class(OntologyClass(
        name="Vehicle",
        description="Any mode of transportation"
    ))
    
    ontology.add_class(OntologyClass(
        name="LandVehicle",
        parent="Vehicle",
        description="Vehicles that travel on land"
    ))
    
    ontology.add_class(OntologyClass(
        name="WaterVehicle",
        parent="Vehicle",
        description="Vehicles that travel on water"
    ))
    
    ontology.add_class(OntologyClass(
        name="AirVehicle",
        parent="Vehicle",
        description="Vehicles that travel in air"
    ))
    
    ontology.add_class(OntologyClass(
        name="Car",
        parent="LandVehicle",
        properties=["num_wheels", "fuel_type", "num_doors"],
        description="Four-wheeled motor vehicle"
    ))
    
    ontology.add_class(OntologyClass(
        name="Bicycle",
        parent="LandVehicle",
        properties=["num_wheels", "has_motor"],
        description="Two-wheeled human-powered vehicle"
    ))
    
    ontology.add_class(OntologyClass(
        name="Boat",
        parent="WaterVehicle",
        properties=["propulsion_type", "max_speed"],
        description="Watercraft of any size"
    ))
    
    # Add instances
    ontology.add_instance(OntologyInstance(
        id="vehicle1",
        class_name="Car",
        properties={"num_wheels": 4, "fuel_type": "electric", "num_doors": 4}
    ))
    
    ontology.add_instance(OntologyInstance(
        id="vehicle2",
        class_name="Bicycle",
        properties={"num_wheels": 2, "has_motor": False}
    ))
    
    print(f"✓ Created ontology with {len(ontology.classes)} classes")
    print(f"✓ Added {len(ontology.instances)} instances")
    print()
    
    # Create reasoning agent
    agent = OntologyReasoningAgent(ontology)
    
    # Classification
    print("2. Instance Classification")
    print("-" * 50)
    
    test_descriptions = [
        "A motorized two-wheeler used for transportation",
        "A watercraft with sails",
        "A four-wheeled vehicle with an engine"
    ]
    
    for desc in test_descriptions:
        print(f"\nDescription: {desc}")
        classification = agent.classify_instance(desc)
        print(f"Classification: {classification}")
    
    # Property inference
    print("\n\n3. Property Inference")
    print("-" * 50)
    
    inferred = agent.infer_properties("vehicle1")
    print(f"\nInferred properties for vehicle1:")
    print(inferred)
    
    # Semantic queries
    print("\n\n4. Semantic Queries")
    print("-" * 50)
    
    queries = [
        "What are all the types of land vehicles?",
        "What properties do cars have?",
        "How are bicycles and cars related?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        answer = agent.answer_semantic_query(query)
        print(f"Answer: {answer}")
    
    # Consistency check
    print("\n\n5. Ontology Consistency Check")
    print("-" * 50)
    consistency = agent.check_consistency()
    print(consistency)
    
    print("\n=== Summary ===")
    print("Ontology-based reasoning demonstrated with:")
    print("- Formal class hierarchies")
    print("- Property definitions")
    print("- Instance classification")
    print("- Property inference")
    print("- Semantic queries")
    print("- Consistency checking")


if __name__ == "__main__":
    demonstrate_ontology_based_reasoning()
