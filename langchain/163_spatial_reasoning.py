"""
Pattern 163: Spatial Reasoning

Description:
    The Spatial Reasoning pattern enables agents to understand and reason about spatial
    relationships, positions, orientations, and geometric properties. This pattern helps
    agents solve problems involving physical space, layouts, navigation, and spatial
    configurations.

Components:
    1. Coordinate System: Tracks positions and locations
    2. Relationship Analyzer: Identifies spatial relationships (above, below, left, right, etc.)
    3. Distance Calculator: Computes distances and proximity
    4. Orientation Tracker: Manages directions and rotations
    5. Spatial Query Engine: Answers spatial questions
    6. Visualization Helper: Generates spatial descriptions

Use Cases:
    - Navigation and pathfinding
    - Layout design and optimization
    - Object placement and arrangement
    - Map understanding and interpretation
    - Geometry problem solving
    - Spatial puzzle solving
    - Floor plan analysis

Benefits:
    - Enables spatial understanding
    - Supports geometric reasoning
    - Facilitates navigation tasks
    - Improves layout optimization
    - Better scene understanding

Trade-offs:
    - Complex coordinate management
    - Requires clear spatial representation
    - May need visualization capabilities
    - Challenging for abstract spaces
    - Limited by 2D/3D representation

LangChain Implementation:
    Uses LLM for spatial relationship interpretation with structured coordinate
    systems and distance calculations. Combines symbolic representation with
    natural language understanding for spatial queries.
"""

import os
import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class Direction(Enum):
    """Cardinal and ordinal directions"""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    NORTHEAST = "northeast"
    NORTHWEST = "northwest"
    SOUTHEAST = "southeast"
    SOUTHWEST = "southwest"
    UP = "up"
    DOWN = "down"


class SpatialRelation(Enum):
    """Types of spatial relationships"""
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left of"
    RIGHT_OF = "right of"
    IN_FRONT_OF = "in front of"
    BEHIND = "behind"
    INSIDE = "inside"
    OUTSIDE = "outside"
    ADJACENT = "adjacent"
    NEAR = "near"
    FAR = "far"
    BETWEEN = "between"


@dataclass
class SpatialObject:
    """Represents an object in space"""
    name: str
    x: float
    y: float
    z: float = 0.0
    width: float = 1.0
    height: float = 1.0
    depth: float = 1.0
    orientation: float = 0.0  # degrees


@dataclass
class SpatialQuery:
    """A spatial reasoning query"""
    question: str
    objects: List[SpatialObject]
    context: str = ""


@dataclass
class SpatialAnalysis:
    """Result of spatial reasoning"""
    answer: str
    relationships: List[str]
    distances: Dict[str, float]
    reasoning: str
    visualization: str


class SpatialReasoningAgent:
    """Agent that performs spatial reasoning tasks"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the spatial reasoning agent"""
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Prompts for different spatial reasoning tasks
        self.relation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a spatial reasoning expert. Analyze the positions 
            of objects and describe their spatial relationships clearly and accurately.
            Consider distances, directions, and relative positions."""),
            ("user", """Objects and their positions:
{object_descriptions}

Question: {question}

Provide a clear answer based on the spatial relationships.""")
        ])
        
        self.path_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a navigation expert. Given object positions, 
            determine optimal paths and directions."""),
            ("user", """Objects and their positions:
{object_descriptions}

Navigation task: {task}

Describe the path or navigation instructions.""")
        ])
        
        self.relation_chain = self.relation_prompt | self.llm | StrOutputParser()
        self.path_chain = self.path_prompt | self.llm | StrOutputParser()
    
    def analyze_spatial_query(self, query: SpatialQuery) -> SpatialAnalysis:
        """Analyze a spatial query and return comprehensive analysis"""
        # Calculate all pairwise distances
        distances = self._calculate_distances(query.objects)
        
        # Identify spatial relationships
        relationships = self._identify_relationships(query.objects)
        
        # Create object descriptions
        object_descriptions = self._create_object_descriptions(query.objects)
        
        # Get LLM reasoning
        reasoning = self.relation_chain.invoke({
            "object_descriptions": object_descriptions,
            "question": query.question
        })
        
        # Create visualization
        visualization = self._create_visualization(query.objects)
        
        return SpatialAnalysis(
            answer=reasoning,
            relationships=relationships,
            distances=distances,
            reasoning=reasoning,
            visualization=visualization
        )
    
    def find_path(self, start: SpatialObject, end: SpatialObject, 
                  obstacles: List[SpatialObject]) -> Dict[str, Any]:
        """Find path from start to end avoiding obstacles"""
        # Calculate direct distance
        direct_distance = self._distance(start, end)
        
        # Calculate direction
        direction = self._get_direction(start, end)
        
        # Check for obstacles in path
        obstacles_in_path = self._check_obstacles(start, end, obstacles)
        
        # Get navigation instructions
        object_descriptions = self._create_object_descriptions(
            [start, end] + obstacles
        )
        
        instructions = self.path_chain.invoke({
            "object_descriptions": object_descriptions,
            "task": f"Navigate from {start.name} to {end.name}"
        })
        
        return {
            "direct_distance": direct_distance,
            "direction": direction,
            "obstacles_in_path": obstacles_in_path,
            "instructions": instructions,
            "path_clear": len(obstacles_in_path) == 0
        }
    
    def find_nearest(self, reference: SpatialObject, 
                     candidates: List[SpatialObject]) -> Tuple[SpatialObject, float]:
        """Find the nearest object to reference from candidates"""
        nearest = None
        min_distance = float('inf')
        
        for candidate in candidates:
            distance = self._distance(reference, candidate)
            if distance < min_distance:
                min_distance = distance
                nearest = candidate
        
        return nearest, min_distance
    
    def check_relationship(self, obj1: SpatialObject, obj2: SpatialObject,
                          relation: SpatialRelation) -> bool:
        """Check if a specific spatial relationship holds between two objects"""
        if relation == SpatialRelation.ABOVE:
            return obj1.z > obj2.z
        elif relation == SpatialRelation.BELOW:
            return obj1.z < obj2.z
        elif relation == SpatialRelation.LEFT_OF:
            return obj1.x < obj2.x
        elif relation == SpatialRelation.RIGHT_OF:
            return obj1.x > obj2.x
        elif relation == SpatialRelation.IN_FRONT_OF:
            return obj1.y > obj2.y
        elif relation == SpatialRelation.BEHIND:
            return obj1.y < obj2.y
        elif relation == SpatialRelation.NEAR:
            distance = self._distance(obj1, obj2)
            return distance < 5.0  # Threshold for "near"
        elif relation == SpatialRelation.FAR:
            distance = self._distance(obj1, obj2)
            return distance > 10.0  # Threshold for "far"
        elif relation == SpatialRelation.ADJACENT:
            distance = self._distance(obj1, obj2)
            return distance < 2.0  # Threshold for "adjacent"
        else:
            return False
    
    def get_objects_in_region(self, objects: List[SpatialObject],
                              center: Tuple[float, float, float],
                              radius: float) -> List[SpatialObject]:
        """Get all objects within a spherical region"""
        result = []
        cx, cy, cz = center
        
        for obj in objects:
            distance = math.sqrt(
                (obj.x - cx)**2 + 
                (obj.y - cy)**2 + 
                (obj.z - cz)**2
            )
            if distance <= radius:
                result.append(obj)
        
        return result
    
    def _distance(self, obj1: SpatialObject, obj2: SpatialObject) -> float:
        """Calculate Euclidean distance between two objects"""
        return math.sqrt(
            (obj1.x - obj2.x)**2 + 
            (obj1.y - obj2.y)**2 + 
            (obj1.z - obj2.z)**2
        )
    
    def _calculate_distances(self, objects: List[SpatialObject]) -> Dict[str, float]:
        """Calculate all pairwise distances"""
        distances = {}
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                key = f"{obj1.name} to {obj2.name}"
                distances[key] = self._distance(obj1, obj2)
        return distances
    
    def _identify_relationships(self, objects: List[SpatialObject]) -> List[str]:
        """Identify spatial relationships between objects"""
        relationships = []
        
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                # Check various relationships
                if self.check_relationship(obj1, obj2, SpatialRelation.ABOVE):
                    relationships.append(f"{obj1.name} is above {obj2.name}")
                if self.check_relationship(obj1, obj2, SpatialRelation.BELOW):
                    relationships.append(f"{obj1.name} is below {obj2.name}")
                if self.check_relationship(obj1, obj2, SpatialRelation.LEFT_OF):
                    relationships.append(f"{obj1.name} is left of {obj2.name}")
                if self.check_relationship(obj1, obj2, SpatialRelation.RIGHT_OF):
                    relationships.append(f"{obj1.name} is right of {obj2.name}")
                if self.check_relationship(obj1, obj2, SpatialRelation.NEAR):
                    relationships.append(f"{obj1.name} is near {obj2.name}")
                if self.check_relationship(obj1, obj2, SpatialRelation.ADJACENT):
                    relationships.append(f"{obj1.name} is adjacent to {obj2.name}")
        
        return relationships
    
    def _get_direction(self, from_obj: SpatialObject, 
                       to_obj: SpatialObject) -> Direction:
        """Determine cardinal direction from one object to another"""
        dx = to_obj.x - from_obj.x
        dy = to_obj.y - from_obj.y
        
        angle = math.atan2(dy, dx) * 180 / math.pi
        
        # Normalize to 0-360
        if angle < 0:
            angle += 360
        
        # Determine direction
        if 22.5 <= angle < 67.5:
            return Direction.NORTHEAST
        elif 67.5 <= angle < 112.5:
            return Direction.NORTH
        elif 112.5 <= angle < 157.5:
            return Direction.NORTHWEST
        elif 157.5 <= angle < 202.5:
            return Direction.WEST
        elif 202.5 <= angle < 247.5:
            return Direction.SOUTHWEST
        elif 247.5 <= angle < 292.5:
            return Direction.SOUTH
        elif 292.5 <= angle < 337.5:
            return Direction.SOUTHEAST
        else:
            return Direction.EAST
    
    def _check_obstacles(self, start: SpatialObject, end: SpatialObject,
                        obstacles: List[SpatialObject]) -> List[str]:
        """Check which obstacles are in the path between start and end"""
        obstacles_in_path = []
        
        for obstacle in obstacles:
            # Simple line intersection check
            if self._is_near_line(start, end, obstacle):
                obstacles_in_path.append(obstacle.name)
        
        return obstacles_in_path
    
    def _is_near_line(self, start: SpatialObject, end: SpatialObject,
                      point: SpatialObject, threshold: float = 1.0) -> bool:
        """Check if a point is near the line between start and end"""
        # Calculate perpendicular distance from point to line
        x1, y1 = start.x, start.y
        x2, y2 = end.x, end.y
        x0, y0 = point.x, point.y
        
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2-y1)**2 + (x2-x1)**2)
        
        if denominator == 0:
            return False
        
        distance = numerator / denominator
        
        # Also check if point is between start and end
        between = (min(x1, x2) <= x0 <= max(x1, x2) and 
                  min(y1, y2) <= y0 <= max(y1, y2))
        
        return distance < threshold and between
    
    def _create_object_descriptions(self, objects: List[SpatialObject]) -> str:
        """Create textual descriptions of object positions"""
        descriptions = []
        for obj in objects:
            desc = f"{obj.name}: position ({obj.x:.1f}, {obj.y:.1f}, {obj.z:.1f})"
            if obj.width != 1.0 or obj.height != 1.0:
                desc += f", size ({obj.width:.1f}x{obj.height:.1f}x{obj.depth:.1f})"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def _create_visualization(self, objects: List[SpatialObject]) -> str:
        """Create ASCII visualization of object positions"""
        if not objects:
            return "No objects to visualize"
        
        # Find bounds
        min_x = min(obj.x for obj in objects) - 2
        max_x = max(obj.x for obj in objects) + 2
        min_y = min(obj.y for obj in objects) - 2
        max_y = max(obj.y for obj in objects) + 2
        
        # Create grid
        width = int(max_x - min_x) + 1
        height = int(max_y - min_y) + 1
        
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        # Place objects
        for i, obj in enumerate(objects):
            x = int(obj.x - min_x)
            y = int(obj.y - min_y)
            if 0 <= x < width and 0 <= y < height:
                grid[height - 1 - y][x] = chr(65 + i)  # A, B, C, ...
        
        # Create visualization string
        vis = "\n" + "\n".join("".join(row) for row in grid)
        vis += "\n\nLegend: " + ", ".join(
            f"{chr(65+i)}={obj.name}" for i, obj in enumerate(objects)
        )
        
        return vis


def demonstrate_spatial_reasoning():
    """Demonstrate spatial reasoning capabilities"""
    print("=" * 80)
    print("SPATIAL REASONING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = SpatialReasoningAgent()
    
    # Example 1: Room layout analysis
    print("\n" + "=" * 80)
    print("Example 1: Room Layout Analysis")
    print("=" * 80)
    
    room_objects = [
        SpatialObject("Table", 5, 5, 0, 2, 1, 3),
        SpatialObject("Chair1", 4, 5, 0, 1, 1, 1),
        SpatialObject("Chair2", 6, 5, 0, 1, 1, 1),
        SpatialObject("Lamp", 5, 5, 2, 0.5, 0.5, 1),
        SpatialObject("Door", 0, 5, 0, 1, 2, 0.1),
        SpatialObject("Window", 10, 5, 1, 2, 1, 0.1)
    ]
    
    query = SpatialQuery(
        question="What is the spatial arrangement of furniture in the room?",
        objects=room_objects,
        context="Living room layout"
    )
    
    analysis = agent.analyze_spatial_query(query)
    
    print(f"\nQuery: {query.question}")
    print(f"\nAnswer: {analysis.answer}")
    print(f"\nKey Relationships:")
    for rel in analysis.relationships[:5]:
        print(f"  - {rel}")
    print(f"\nVisualization:{analysis.visualization}")
    
    # Example 2: Navigation pathfinding
    print("\n" + "=" * 80)
    print("Example 2: Navigation Pathfinding")
    print("=" * 80)
    
    start = SpatialObject("Start", 0, 0, 0)
    end = SpatialObject("Goal", 10, 10, 0)
    obstacles = [
        SpatialObject("Obstacle1", 5, 5, 0, 2, 2, 1),
        SpatialObject("Obstacle2", 7, 3, 0, 1, 1, 1)
    ]
    
    path_result = agent.find_path(start, end, obstacles)
    
    print(f"\nNavigation from {start.name} to {end.name}")
    print(f"Direct distance: {path_result['direct_distance']:.2f} units")
    print(f"Direction: {path_result['direction'].value}")
    print(f"Path clear: {path_result['path_clear']}")
    if path_result['obstacles_in_path']:
        print(f"Obstacles in path: {', '.join(path_result['obstacles_in_path'])}")
    print(f"\nInstructions:\n{path_result['instructions']}")
    
    # Example 3: Nearest neighbor search
    print("\n" + "=" * 80)
    print("Example 3: Nearest Neighbor Search")
    print("=" * 80)
    
    reference = SpatialObject("Home", 5, 5, 0)
    stores = [
        SpatialObject("Grocery Store", 3, 4, 0),
        SpatialObject("Pharmacy", 8, 6, 0),
        SpatialObject("Coffee Shop", 5, 7, 0),
        SpatialObject("Bookstore", 10, 10, 0)
    ]
    
    nearest, distance = agent.find_nearest(reference, stores)
    
    print(f"\nFinding nearest store to {reference.name}")
    print(f"Nearest: {nearest.name}")
    print(f"Distance: {distance:.2f} units")
    
    print("\nAll distances:")
    for store in stores:
        dist = agent._distance(reference, store)
        direction = agent._get_direction(reference, store)
        print(f"  {store.name}: {dist:.2f} units {direction.value}")
    
    # Example 4: Spatial relationship checking
    print("\n" + "=" * 80)
    print("Example 4: Spatial Relationship Checking")
    print("=" * 80)
    
    ceiling_light = SpatialObject("Ceiling Light", 5, 5, 8)
    floor_lamp = SpatialObject("Floor Lamp", 5, 5, 0)
    couch = SpatialObject("Couch", 3, 5, 0)
    
    relations_to_check = [
        (ceiling_light, floor_lamp, SpatialRelation.ABOVE),
        (floor_lamp, ceiling_light, SpatialRelation.BELOW),
        (couch, floor_lamp, SpatialRelation.NEAR),
        (couch, floor_lamp, SpatialRelation.ADJACENT)
    ]
    
    print("\nChecking spatial relationships:")
    for obj1, obj2, relation in relations_to_check:
        holds = agent.check_relationship(obj1, obj2, relation)
        print(f"  {obj1.name} is {relation.value} {obj2.name}: {holds}")
    
    # Example 5: Regional search
    print("\n" + "=" * 80)
    print("Example 5: Regional Search")
    print("=" * 80)
    
    all_objects = [
        SpatialObject("Object1", 2, 2, 0),
        SpatialObject("Object2", 5, 5, 0),
        SpatialObject("Object3", 8, 8, 0),
        SpatialObject("Object4", 3, 7, 0),
        SpatialObject("Object5", 10, 2, 0)
    ]
    
    search_center = (5, 5, 0)
    search_radius = 4
    
    objects_in_region = agent.get_objects_in_region(
        all_objects, search_center, search_radius
    )
    
    print(f"\nSearching for objects within {search_radius} units of {search_center}")
    print(f"Found {len(objects_in_region)} objects:")
    for obj in objects_in_region:
        dist = math.sqrt(
            (obj.x - search_center[0])**2 + 
            (obj.y - search_center[1])**2 + 
            (obj.z - search_center[2])**2
        )
        print(f"  {obj.name} at ({obj.x}, {obj.y}, {obj.z}) - {dist:.2f} units away")
    
    # Example 6: Complex spatial query
    print("\n" + "=" * 80)
    print("Example 6: Complex Spatial Query")
    print("=" * 80)
    
    city_objects = [
        SpatialObject("City Hall", 5, 5, 0, 3, 3, 10),
        SpatialObject("Park", 8, 8, 0, 5, 5, 0),
        SpatialObject("School", 2, 8, 0, 2, 2, 5),
        SpatialObject("Hospital", 9, 2, 0, 3, 3, 8),
        SpatialObject("Library", 1, 2, 0, 2, 2, 4)
    ]
    
    complex_query = SpatialQuery(
        question="Which public facilities are located in the northern part of the city and which are in the south?",
        objects=city_objects,
        context="City planning analysis"
    )
    
    complex_analysis = agent.analyze_spatial_query(complex_query)
    
    print(f"\nQuery: {complex_query.question}")
    print(f"\nAnalysis:\n{complex_analysis.answer}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The Spatial Reasoning pattern enables agents to:
✓ Understand spatial relationships and positions
✓ Calculate distances and directions
✓ Find optimal paths and navigation routes
✓ Identify nearest neighbors
✓ Search within spatial regions
✓ Analyze complex spatial arrangements
✓ Combine geometric calculations with semantic understanding

This pattern is valuable for:
- Navigation and routing systems
- Layout and design optimization
- Geographic information systems
- Robotics and autonomous vehicles
- Game AI and virtual environments
- Spatial data analysis
- Map-based applications
    """)


if __name__ == "__main__":
    demonstrate_spatial_reasoning()
