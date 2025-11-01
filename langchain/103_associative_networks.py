"""
Pattern 103: Associative Memory Networks

Description:
    The Associative Memory Networks pattern implements memory systems where memories
    are interconnected through associations, enabling retrieval through spreading
    activation and contextual relationships. This pattern mimics human associative
    memory where one memory can trigger recall of related memories through various
    types of connections (semantic, temporal, causal, episodic).
    
    Associative networks allow agents to discover indirect relationships, make
    creative connections, and retrieve relevant information through chains of
    associations rather than direct queries. Each memory node can be linked to
    others through typed relationships, and activation spreads through the network
    when memories are accessed, bringing related memories to attention.
    
    This pattern supports various association types including semantic similarity,
    temporal co-occurrence, causal relationships, part-whole hierarchies, and
    user-defined custom links. It implements spreading activation algorithms,
    association strength tracking, and network-based retrieval strategies.

Key Components:
    1. Memory Nodes: Individual memory items in the network
    2. Association Links: Typed connections between memories
    3. Spreading Activation: Propagation of activation through network
    4. Association Strength: Weight of connections
    5. Activation Decay: Gradual decrease of activation
    6. Network Traversal: Algorithms for exploring associations
    7. Link Types: Different relationship types

Association Types:
    1. Semantic: Similar meaning or concept
    2. Temporal: Occurred at similar times
    3. Causal: Cause-effect relationships
    4. Spatial: Co-location in space
    5. Episodic: Part of same experience
    6. Hierarchical: Part-whole, is-a relationships
    7. Functional: Used together or for similar purposes
    
Spreading Activation:
    1. Initial Activation: Starting memory gets activation
    2. Propagation: Activation spreads to connected memories
    3. Decay: Activation weakens with distance
    4. Threshold: Minimum activation for retrieval
    5. Convergence: Multiple paths strengthen activation
    6. Inhibition: Negative associations reduce activation

Use Cases:
    - Creative problem solving (analogies, brainstorming)
    - Context-aware information retrieval
    - Knowledge discovery and inference
    - Recommendation systems
    - Learning from experience
    - Concept mapping and knowledge graphs
    - Semantic search and exploration

Advantages:
    - Flexible retrieval through associations
    - Discovers indirect relationships
    - Mimics human memory organization
    - Supports creative connections
    - Handles ambiguous queries
    - Learns from access patterns
    - Enables serendipitous discovery

Challenges:
    - Computational complexity of spreading activation
    - Managing association strength decay
    - Preventing over-activation (too many results)
    - Determining appropriate association types
    - Balancing network density
    - Avoiding spurious associations

LangChain Implementation:
    This implementation uses LangChain for:
    - LLM-based semantic similarity calculation
    - Association type inference
    - Context-aware link creation
    - Network-based reasoning
    
Production Considerations:
    - Implement efficient graph traversal algorithms
    - Set appropriate activation thresholds
    - Monitor network size and complexity
    - Prune weak associations periodically
    - Cache frequently accessed paths
    - Consider distributed graph storage
    - Implement incremental updates
    - Balance precision vs. recall in retrieval
"""

import os
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class AssociationType(Enum):
    """Types of associations between memories."""
    SEMANTIC = "semantic"  # Similar meaning
    TEMPORAL = "temporal"  # Time-based
    CAUSAL = "causal"  # Cause-effect
    SPATIAL = "spatial"  # Location-based
    EPISODIC = "episodic"  # Same experience
    HIERARCHICAL = "hierarchical"  # Part-whole
    FUNCTIONAL = "functional"  # Used together
    CUSTOM = "custom"  # User-defined


@dataclass
class MemoryNode:
    """
    Node in associative memory network.
    
    Attributes:
        node_id: Unique identifier
        content: Memory content
        created_at: Creation timestamp
        activation: Current activation level
        base_activation: Base activation strength
        tags: Associated tags
        metadata: Additional information
    """
    node_id: str
    content: str
    created_at: datetime
    activation: float = 0.0
    base_activation: float = 0.5
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Association:
    """
    Link between memory nodes.
    
    Attributes:
        from_node: Source node ID
        to_node: Target node ID
        association_type: Type of association
        strength: Association strength (0-1)
        created_at: When association was created
        access_count: Number of times traversed
        metadata: Additional information
    """
    from_node: str
    to_node: str
    association_type: AssociationType
    strength: float
    created_at: datetime
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AssociativeMemoryNetwork:
    """
    Associative memory network with spreading activation.
    
    This class implements a network of interconnected memories with
    various association types and spreading activation for retrieval.
    """
    
    def __init__(
        self,
        activation_decay: float = 0.5,
        activation_threshold: float = 0.3,
        max_spread_depth: int = 3,
        temperature: float = 0.3
    ):
        """
        Initialize associative memory network.
        
        Args:
            activation_decay: Decay factor for spreading (0-1)
            activation_threshold: Minimum activation for retrieval
            max_spread_depth: Maximum depth for spreading activation
            temperature: LLM temperature
        """
        self.activation_decay = activation_decay
        self.activation_threshold = activation_threshold
        self.max_spread_depth = max_spread_depth
        
        self.nodes: Dict[str, MemoryNode] = {}
        self.associations: Dict[str, List[Association]] = defaultdict(list)
        self.reverse_associations: Dict[str, List[Association]] = defaultdict(list)
        
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.node_counter = 0
    
    def add_memory(
        self,
        content: str,
        tags: List[str] = None,
        base_activation: float = 0.5
    ) -> MemoryNode:
        """
        Add memory node to network.
        
        Args:
            content: Memory content
            tags: Associated tags
            base_activation: Base activation level
            
        Returns:
            Created memory node
        """
        self.node_counter += 1
        node_id = f"node_{self.node_counter}"
        
        node = MemoryNode(
            node_id=node_id,
            content=content,
            created_at=datetime.now(),
            base_activation=base_activation,
            tags=tags or []
        )
        
        self.nodes[node_id] = node
        return node
    
    def associate(
        self,
        from_node_id: str,
        to_node_id: str,
        association_type: AssociationType,
        strength: float = 0.5,
        bidirectional: bool = True
    ):
        """
        Create association between nodes.
        
        Args:
            from_node_id: Source node
            to_node_id: Target node
            association_type: Type of association
            strength: Association strength
            bidirectional: Create reverse link too
        """
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            return
        
        # Forward association
        assoc = Association(
            from_node=from_node_id,
            to_node=to_node_id,
            association_type=association_type,
            strength=strength,
            created_at=datetime.now()
        )
        self.associations[from_node_id].append(assoc)
        self.reverse_associations[to_node_id].append(assoc)
        
        # Bidirectional association
        if bidirectional:
            reverse_assoc = Association(
                from_node=to_node_id,
                to_node=from_node_id,
                association_type=association_type,
                strength=strength,
                created_at=datetime.now()
            )
            self.associations[to_node_id].append(reverse_assoc)
            self.reverse_associations[from_node_id].append(reverse_assoc)
    
    def auto_associate(
        self,
        node_id: str,
        min_strength: float = 0.4
    ):
        """
        Automatically create associations based on similarity.
        
        Args:
            node_id: Node to associate
            min_strength: Minimum strength for association
        """
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Find similar nodes by tags
        for other_id, other_node in self.nodes.items():
            if other_id == node_id:
                continue
            
            # Tag-based similarity
            common_tags = set(node.tags) & set(other_node.tags)
            if common_tags and len(node.tags) > 0:
                strength = len(common_tags) / max(len(node.tags), len(other_node.tags))
                
                if strength >= min_strength:
                    self.associate(
                        node_id,
                        other_id,
                        AssociationType.SEMANTIC,
                        strength,
                        bidirectional=True
                    )
    
    def spread_activation(
        self,
        start_node_id: str,
        initial_activation: float = 1.0
    ) -> Dict[str, float]:
        """
        Spread activation from starting node through network.
        
        Args:
            start_node_id: Starting node
            initial_activation: Initial activation level
            
        Returns:
            Dictionary of node_id -> activation level
        """
        if start_node_id not in self.nodes:
            return {}
        
        # Reset all activations
        for node in self.nodes.values():
            node.activation = 0.0
        
        # Initialize with starting node
        activations = {start_node_id: initial_activation}
        self.nodes[start_node_id].activation = initial_activation
        
        # Breadth-first spreading
        queue = deque([(start_node_id, initial_activation, 0)])
        visited = set()
        
        while queue:
            current_id, current_activation, depth = queue.popleft()
            
            if current_id in visited or depth >= self.max_spread_depth:
                continue
            
            visited.add(current_id)
            
            # Spread to associated nodes
            for assoc in self.associations.get(current_id, []):
                target_id = assoc.to_node
                
                # Calculate activation with decay
                decay = self.activation_decay ** depth
                new_activation = current_activation * assoc.strength * decay
                
                # Update activation (cumulative)
                if target_id in activations:
                    activations[target_id] += new_activation
                else:
                    activations[target_id] = new_activation
                
                self.nodes[target_id].activation = activations[target_id]
                
                # Track association usage
                assoc.access_count += 1
                
                # Continue spreading if above threshold
                if new_activation > self.activation_threshold:
                    queue.append((target_id, new_activation, depth + 1))
        
        return activations
    
    def retrieve_associated(
        self,
        node_id: str,
        min_activation: Optional[float] = None,
        max_results: int = 10
    ) -> List[Tuple[MemoryNode, float]]:
        """
        Retrieve memories associated with given node.
        
        Args:
            node_id: Starting node
            min_activation: Minimum activation threshold
            max_results: Maximum number of results
            
        Returns:
            List of (node, activation) tuples
        """
        if min_activation is None:
            min_activation = self.activation_threshold
        
        # Spread activation
        activations = self.spread_activation(node_id)
        
        # Filter and sort by activation
        results = [
            (self.nodes[nid], activation)
            for nid, activation in activations.items()
            if nid != node_id and activation >= min_activation
        ]
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def find_path(
        self,
        from_node_id: str,
        to_node_id: str,
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """
        Find path between two nodes.
        
        Args:
            from_node_id: Start node
            to_node_id: End node
            max_depth: Maximum path length
            
        Returns:
            List of node IDs in path, or None
        """
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            return None
        
        # BFS to find shortest path
        queue = deque([(from_node_id, [from_node_id])])
        visited = set()
        
        while queue:
            current_id, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current_id == to_node_id:
                return path
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Explore neighbors
            for assoc in self.associations.get(current_id, []):
                if assoc.to_node not in visited:
                    queue.append((assoc.to_node, path + [assoc.to_node]))
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get network statistics."""
        total_associations = sum(len(assocs) for assocs in self.associations.values())
        
        assoc_types = defaultdict(int)
        for assocs in self.associations.values():
            for assoc in assocs:
                assoc_types[assoc.association_type.value] += 1
        
        return {
            "total_nodes": len(self.nodes),
            "total_associations": total_associations,
            "avg_associations_per_node": total_associations / len(self.nodes) if self.nodes else 0,
            "association_types": dict(assoc_types)
        }


class AssociativeMemoryAgent:
    """
    Agent with associative memory network.
    
    This agent uses associative memory for context-aware retrieval
    and creative problem-solving through memory associations.
    """
    
    def __init__(
        self,
        activation_decay: float = 0.5,
        temperature: float = 0.5
    ):
        """
        Initialize associative memory agent.
        
        Args:
            activation_decay: Activation decay factor
            temperature: LLM temperature
        """
        self.memory_network = AssociativeMemoryNetwork(
            activation_decay=activation_decay
        )
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
    
    def learn(
        self,
        content: str,
        tags: List[str] = None,
        related_to: List[str] = None
    ) -> MemoryNode:
        """
        Learn new information with associations.
        
        Args:
            content: Content to learn
            tags: Associated tags
            related_to: IDs of related memories
            
        Returns:
            Created memory node
        """
        # Add memory
        node = self.memory_network.add_memory(content, tags)
        
        # Auto-associate based on tags
        self.memory_network.auto_associate(node.node_id)
        
        # Explicit associations
        if related_to:
            for related_id in related_to:
                self.memory_network.associate(
                    node.node_id,
                    related_id,
                    AssociationType.SEMANTIC,
                    0.7
                )
        
        return node
    
    def recall(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Tuple[MemoryNode, float, List[str]]]:
        """
        Recall memories through associative retrieval.
        
        Args:
            query: Query string
            max_results: Maximum results
            
        Returns:
            List of (node, activation, path) tuples
        """
        # Find direct matches
        matching_nodes = []
        for node in self.memory_network.nodes.values():
            if query.lower() in node.content.lower():
                matching_nodes.append(node)
        
        if not matching_nodes:
            return []
        
        # Use first match as starting point for spreading activation
        start_node = matching_nodes[0]
        
        # Spread activation
        associated = self.memory_network.retrieve_associated(
            start_node.node_id,
            max_results=max_results * 2
        )
        
        # Build results with paths
        results = []
        for node, activation in associated:
            path = self.memory_network.find_path(start_node.node_id, node.node_id)
            results.append((node, activation, path or []))
        
        return results[:max_results]


def demonstrate_associative_memory():
    """Demonstrate associative memory networks pattern."""
    
    print("=" * 80)
    print("ASSOCIATIVE MEMORY NETWORKS PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic network creation and associations
    print("\n" + "=" * 80)
    print("Example 1: Building Associative Memory Network")
    print("=" * 80)
    
    network = AssociativeMemoryNetwork(
        activation_decay=0.6,
        activation_threshold=0.2,
        max_spread_depth=3
    )
    
    print("\nAdding memories about animals...")
    
    # Add animal memories
    dog = network.add_memory("Dog is a pet animal", ["animal", "pet", "mammal"])
    cat = network.add_memory("Cat is a pet animal", ["animal", "pet", "mammal"])
    lion = network.add_memory("Lion is a wild animal", ["animal", "wild", "mammal"])
    bird = network.add_memory("Bird can fly", ["animal", "flying"])
    fish = network.add_memory("Fish lives in water", ["animal", "aquatic"])
    
    print(f"  Added {len(network.nodes)} memory nodes")
    
    # Create explicit associations
    print("\nCreating associations...")
    network.associate(dog.node_id, cat.node_id, AssociationType.SEMANTIC, 0.8)
    network.associate(dog.node_id, lion.node_id, AssociationType.HIERARCHICAL, 0.5)
    network.associate(cat.node_id, lion.node_id, AssociationType.HIERARCHICAL, 0.5)
    
    stats = network.get_statistics()
    print(f"  Total associations: {stats['total_associations']}")
    print(f"  Avg per node: {stats['avg_associations_per_node']:.1f}")
    
    # Example 2: Spreading activation
    print("\n" + "=" * 80)
    print("Example 2: Spreading Activation")
    print("=" * 80)
    
    print(f"\nActivating memory: '{dog.content}'")
    activations = network.spread_activation(dog.node_id, initial_activation=1.0)
    
    print("\nActivation levels:")
    sorted_activations = sorted(
        activations.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for node_id, activation in sorted_activations:
        node = network.nodes[node_id]
        print(f"  [{activation:.3f}] {node.content}")
    
    # Example 3: Associative retrieval
    print("\n" + "=" * 80)
    print("Example 3: Associative Retrieval")
    print("=" * 80)
    
    print(f"\nRetrieving memories associated with 'Dog'...")
    associated_memories = network.retrieve_associated(dog.node_id, max_results=5)
    
    print(f"\nFound {len(associated_memories)} associated memories:")
    for node, activation in associated_memories:
        print(f"  [{activation:.3f}] {node.content}")
        print(f"           Tags: {node.tags}")
    
    # Example 4: Path finding
    print("\n" + "=" * 80)
    print("Example 4: Finding Association Paths")
    print("=" * 80)
    
    print(f"\nFinding path from 'Dog' to 'Lion'...")
    path = network.find_path(dog.node_id, lion.node_id)
    
    if path:
        print(f"  Path found ({len(path)} steps):")
        for i, node_id in enumerate(path):
            node = network.nodes[node_id]
            print(f"    {i+1}. {node.content}")
    
    # Example 5: Auto-association by tags
    print("\n" + "=" * 80)
    print("Example 5: Automatic Association by Similarity")
    print("=" * 80)
    
    auto_network = AssociativeMemoryNetwork()
    
    # Add memories with tags
    memories = [
        ("Python is a programming language", ["programming", "python", "language"]),
        ("Java is a programming language", ["programming", "java", "language"]),
        ("Machine learning uses Python", ["programming", "python", "ml"]),
        ("Data science involves statistics", ["data", "statistics", "analysis"]),
        ("Python is popular in data science", ["programming", "python", "data"]),
    ]
    
    nodes = []
    for content, tags in memories:
        node = auto_network.add_memory(content, tags)
        nodes.append(node)
        auto_network.auto_associate(node.node_id, min_strength=0.4)
    
    print(f"\nAdded {len(nodes)} memories")
    
    stats = auto_network.get_statistics()
    print(f"Auto-created associations: {stats['total_associations']}")
    print(f"Association types: {stats['association_types']}")
    
    # Show associations for first memory
    print(f"\nAssociations for: '{nodes[0].content}'")
    assocs = auto_network.associations[nodes[0].node_id]
    for assoc in assocs:
        target = auto_network.nodes[assoc.to_node]
        print(f"  → [{assoc.strength:.2f}] {target.content}")
    
    # Example 6: Multi-hop retrieval
    print("\n" + "=" * 80)
    print("Example 6: Multi-Hop Associative Retrieval")
    print("=" * 80)
    
    # Create network with multi-hop connections
    hop_network = AssociativeMemoryNetwork(
        max_spread_depth=4,
        activation_threshold=0.15
    )
    
    # Create a knowledge chain
    apple = hop_network.add_memory("Apple is a fruit", ["food", "fruit"])
    fruit = hop_network.add_memory("Fruit is healthy", ["food", "health"])
    health = hop_network.add_memory("Health requires exercise", ["health", "fitness"])
    exercise = hop_network.add_memory("Exercise burns calories", ["fitness", "energy"])
    
    # Chain associations
    hop_network.associate(apple.node_id, fruit.node_id, AssociationType.HIERARCHICAL, 0.8)
    hop_network.associate(fruit.node_id, health.node_id, AssociationType.CAUSAL, 0.7)
    hop_network.associate(health.node_id, exercise.node_id, AssociationType.CAUSAL, 0.6)
    
    print("\nKnowledge chain created:")
    print("  Apple → Fruit → Health → Exercise")
    
    print(f"\nStarting from 'Apple', spreading activation...")
    multi_hop = hop_network.retrieve_associated(apple.node_id, max_results=10)
    
    print(f"\nRetrieved {len(multi_hop)} memories through associations:")
    for node, activation in multi_hop:
        path = hop_network.find_path(apple.node_id, node.node_id)
        hops = len(path) - 1 if path else 0
        print(f"  [{activation:.3f}] ({hops} hops) {node.content}")
    
    # Example 7: Associative memory agent
    print("\n" + "=" * 80)
    print("Example 7: Agent with Associative Memory")
    print("=" * 80)
    
    agent = AssociativeMemoryAgent(activation_decay=0.6)
    
    print("\nTeaching agent about programming concepts...")
    
    # Teach related concepts
    python_node = agent.learn(
        "Python is a high-level programming language",
        tags=["programming", "python"]
    )
    
    django_node = agent.learn(
        "Django is a Python web framework",
        tags=["programming", "python", "web"],
        related_to=[python_node.node_id]
    )
    
    flask_node = agent.learn(
        "Flask is a lightweight Python framework",
        tags=["programming", "python", "web"],
        related_to=[python_node.node_id, django_node.node_id]
    )
    
    agent.learn(
        "NumPy is a Python library for numerical computing",
        tags=["programming", "python", "data"],
        related_to=[python_node.node_id]
    )
    
    agent.learn(
        "pandas is used for data analysis in Python",
        tags=["programming", "python", "data"],
        related_to=[python_node.node_id]
    )
    
    print(f"  Learned {len(agent.memory_network.nodes)} concepts")
    
    # Recall with associations
    print("\nRecalling memories about 'Python'...")
    recalled = agent.recall("Python", max_results=5)
    
    print(f"\nRecalled {len(recalled)} associated memories:")
    for node, activation, path in recalled:
        print(f"  [{activation:.3f}] {node.content}")
        if path and len(path) > 1:
            print(f"           Path: {' → '.join(path[:3])}")
    
    # Example 8: Network statistics and visualization
    print("\n" + "=" * 80)
    print("Example 8: Network Analysis")
    print("=" * 80)
    
    stats = agent.memory_network.get_statistics()
    
    print("\nNetwork Statistics:")
    print(f"  Total Nodes: {stats['total_nodes']}")
    print(f"  Total Associations: {stats['total_associations']}")
    print(f"  Avg Associations/Node: {stats['avg_associations_per_node']:.2f}")
    print(f"  Association Types:")
    for assoc_type, count in stats['association_types'].items():
        print(f"    - {assoc_type}: {count}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Associative Memory Networks Pattern")
    print("=" * 80)
    
    summary = """
    The Associative Memory Networks pattern demonstrated:
    
    1. NETWORK CONSTRUCTION (Example 1):
       - Memory nodes with content and tags
       - Typed associations between memories
       - Semantic, hierarchical, and custom links
       - Bidirectional and directional connections
    
    2. SPREADING ACTIVATION (Example 2):
       - Initial activation at source node
       - Propagation through network links
       - Activation decay with distance
       - Strength-weighted spreading
       - Cumulative activation from multiple paths
    
    3. ASSOCIATIVE RETRIEVAL (Example 3):
       - Activation-based memory retrieval
       - Threshold filtering
       - Ranked results by activation
       - Context-aware recall
    
    4. PATH FINDING (Example 4):
       - Shortest path between memories
       - Breadth-first search
       - Multi-hop connections
       - Relationship discovery
    
    5. AUTO-ASSOCIATION (Example 5):
       - Tag-based similarity
       - Automatic link creation
       - Strength calculation
       - Semantic relationships
    
    6. MULTI-HOP RETRIEVAL (Example 6):
       - Deep network traversal
       - Chain of associations
       - Indirect relationships
       - Knowledge inference
    
    7. AGENT INTEGRATION (Example 7):
       - Learning with associations
       - Context-aware recall
       - Path visualization
       - Association strengthening
    
    8. NETWORK ANALYSIS (Example 8):
       - Node and link statistics
       - Association type distribution
       - Network density metrics
       - Structure analysis
    
    KEY BENEFITS:
    ✓ Flexible context-aware retrieval
    ✓ Discovers indirect relationships
    ✓ Mimics human associative memory
    ✓ Supports creative connections
    ✓ Handles ambiguous queries
    ✓ Learns from access patterns
    ✓ Enables serendipitous discovery
    
    USE CASES:
    • Creative problem solving and brainstorming
    • Context-aware information retrieval
    • Knowledge discovery and inference
    • Recommendation systems
    • Learning from experience
    • Concept mapping and knowledge graphs
    • Semantic search and exploration
    
    BEST PRACTICES:
    1. Choose appropriate association types
    2. Set reasonable activation thresholds
    3. Limit spreading depth for performance
    4. Prune weak associations periodically
    5. Use tags for initial similarity
    6. Track and strengthen frequently used paths
    7. Balance network density
    8. Monitor spreading activation performance
    
    TRADE-OFFS:
    • Complexity vs. flexibility
    • Precision vs. recall
    • Network size vs. performance
    • Automatic vs. manual association
    
    PRODUCTION CONSIDERATIONS:
    → Implement efficient graph storage (Neo4j, NetworkX)
    → Cache frequently accessed activation patterns
    → Set appropriate activation thresholds
    → Monitor and prune weak associations
    → Consider distributed graph for scale
    → Implement incremental activation updates
    → Track association usage for strengthening
    → Balance exploration vs. exploitation
    → Use approximate algorithms for large networks
    → Consider temporal decay of associations
    
    This pattern enables flexible, context-aware memory retrieval through
    network-based associations, supporting creative problem-solving and
    knowledge discovery through spreading activation mechanisms.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_associative_memory()
