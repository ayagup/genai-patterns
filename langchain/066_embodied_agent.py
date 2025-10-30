"""
Pattern 066: Embodied Agent

Description:
    Embodied agents have a physical or virtual body that exists in an environment,
    with sensors for perception and actuators for action. Embodiment provides grounded
    understanding of the world through sensorimotor interaction, spatial reasoning,
    and physical constraints. This pattern bridges abstract reasoning with physical reality.

Components:
    1. Sensory System: Vision, hearing, touch, proprioception
    2. Motor System: Movement, manipulation, locomotion
    3. Body Schema: Internal model of body and capabilities
    4. Spatial Reasoning: Navigation, object relations, distance estimation
    5. Affordance Detection: Understanding what actions are possible
    6. Sensorimotor Integration: Coordinating perception and action

Use Cases:
    - Robotics (physical embodiment)
    - Virtual agents in simulations
    - Embodied AI assistants
    - Game NPCs with physical presence
    - Virtual reality interactions
    - Drone navigation

LangChain Implementation:
    Implements embodied cognition using LLM-based spatial reasoning,
    action planning with physical constraints, and grounded perception.
"""

import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class SensorType(Enum):
    """Types of sensors"""
    VISION = "vision"
    HEARING = "hearing"
    TOUCH = "touch"
    PROPRIOCEPTION = "proprioception"  # Body position
    DISTANCE = "distance"  # Range sensor


class ActionType(Enum):
    """Types of actions"""
    MOVE = "move"
    TURN = "turn"
    GRASP = "grasp"
    RELEASE = "release"
    LOOK = "look"
    SPEAK = "speak"


@dataclass
class Position:
    """3D position"""
    x: float
    y: float
    z: float = 0.0
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance"""
        return math.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2
        )
    
    def __str__(self) -> str:
        if self.z == 0:
            return f"({self.x:.1f}, {self.y:.1f})"
        return f"({self.x:.1f}, {self.y:.1f}, {self.z:.1f})"


@dataclass
class Perception:
    """Sensory perception"""
    sensor: SensorType
    data: Any
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0


@dataclass
class PhysicalObject:
    """Object in environment"""
    object_id: str
    name: str
    position: Position
    size: Tuple[float, float, float]  # width, height, depth
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def is_reachable(self, agent_position: Position, reach_distance: float) -> bool:
        """Check if object is within reach"""
        return agent_position.distance_to(self.position) <= reach_distance


@dataclass
class BodyState:
    """State of agent's body"""
    position: Position
    orientation: float  # degrees
    velocity: float = 0.0
    holding_object: Optional[str] = None
    energy_level: float = 1.0


class EmbodiedAgent:
    """
    Embodied agent with physical presence.
    
    Features:
    1. Spatial awareness and navigation
    2. Object interaction with constraints
    3. Grounded perception
    4. Physical action planning
    5. Affordance recognition
    """
    
    def __init__(
        self,
        initial_position: Position = Position(0, 0, 0),
        reach_distance: float = 1.5,
        move_speed: float = 1.0
    ):
        self.body = BodyState(position=initial_position, orientation=0)
        self.reach_distance = reach_distance
        self.move_speed = move_speed
        
        # Environment knowledge
        self.perceived_objects: Dict[str, PhysicalObject] = {}
        self.perceptions: List[Perception] = []
        
        # Cognitive LLM
        self.cognition = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )
        
        # Spatial reasoning LLM
        self.spatial_reasoner = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.2
        )
    
    def perceive(
        self,
        sensor: SensorType,
        environment_description: str
    ) -> Perception:
        """Perceive environment through sensors"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are processing sensory information from a {sensor.value} sensor.

Current body state:
- Position: {self.body.position}
- Orientation: {self.body.orientation}°

Extract relevant information from the environment description."""),
            ("user", "Environment: {environment}\n\nWhat do you perceive?")
        ])
        
        chain = prompt | self.cognition | StrOutputParser()
        perception_data = chain.invoke({"environment": environment_description})
        
        perception = Perception(
            sensor=sensor,
            data=perception_data,
            confidence=0.85
        )
        
        self.perceptions.append(perception)
        return perception
    
    def identify_affordances(
        self,
        obj: PhysicalObject
    ) -> List[str]:
        """Identify what actions are possible with object"""
        
        # Check physical constraints
        is_reachable = obj.is_reachable(self.body.position, self.reach_distance)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Identify affordances (possible actions) for this object.

Affordances are action possibilities based on physical properties.

Consider:
1. Object properties
2. Physical reachability
3. Agent capabilities
4. Current state"""),
            ("user", """Object: {name}
Size: {size}
Properties: {properties}
Reachable: {reachable}
Agent holding: {holding}

What actions are possible?""")
        ])
        
        chain = prompt | self.spatial_reasoner | StrOutputParser()
        affordances_text = chain.invoke({
            "name": obj.name,
            "size": f"{obj.size[0]}×{obj.size[1]}×{obj.size[2]}",
            "properties": str(obj.properties),
            "reachable": "Yes" if is_reachable else "No",
            "holding": self.body.holding_object or "Nothing"
        })
        
        # Parse affordances
        affordances = []
        for line in affordances_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•')):
                action = line.lstrip('-•').strip()
                if action:
                    affordances.append(action)
        
        return affordances
    
    def plan_navigation(
        self,
        target_position: Position,
        obstacles: List[PhysicalObject]
    ) -> List[Dict[str, Any]]:
        """Plan navigation path considering obstacles"""
        
        current = self.body.position
        distance = current.distance_to(target_position)
        
        # Build obstacle description
        obstacle_desc = ""
        if obstacles:
            obstacle_desc = "Obstacles:\n"
            for obs in obstacles:
                dist = current.distance_to(obs.position)
                obstacle_desc += f"- {obs.name} at {obs.position} (distance: {dist:.1f}m)\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Plan a navigation path from current position to target.

Consider:
1. Obstacles to avoid
2. Efficient path
3. Physical constraints (can't move through walls)

Provide step-by-step movement plan."""),
            ("user", """Current position: {current}
Target position: {target}
Distance: {distance:.1f}m

{obstacles}

Navigation plan:""")
        ])
        
        chain = prompt | self.spatial_reasoner | StrOutputParser()
        plan_text = chain.invoke({
            "current": str(current),
            "target": str(target_position),
            "distance": distance,
            "obstacles": obstacle_desc
        })
        
        # Parse plan into actions
        actions = []
        for i, line in enumerate(plan_text.split('\n'), 1):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                action_text = line.lstrip('0123456789.-) ').strip()
                if action_text:
                    actions.append({
                        "step": i,
                        "action": "move",
                        "description": action_text
                    })
        
        return actions
    
    def execute_action(
        self,
        action_type: ActionType,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute physical action"""
        
        result = {
            "action": action_type.value,
            "success": False,
            "message": ""
        }
        
        if action_type == ActionType.MOVE:
            # Simulate movement
            target = parameters.get("target_position")
            if target:
                distance = self.body.position.distance_to(target)
                time_needed = distance / self.move_speed
                
                # Simple movement (in reality would be more complex)
                self.body.position = target
                self.body.energy_level -= 0.1 * distance
                
                result["success"] = True
                result["message"] = f"Moved to {target} ({time_needed:.1f}s)"
        
        elif action_type == ActionType.GRASP:
            object_id = parameters.get("object_id")
            if object_id in self.perceived_objects:
                obj = self.perceived_objects[object_id]
                
                if obj.is_reachable(self.body.position, self.reach_distance):
                    if not self.body.holding_object:
                        self.body.holding_object = object_id
                        result["success"] = True
                        result["message"] = f"Grasped {obj.name}"
                    else:
                        result["message"] = f"Already holding {self.body.holding_object}"
                else:
                    result["message"] = f"{obj.name} is out of reach"
            else:
                result["message"] = "Object not found"
        
        elif action_type == ActionType.RELEASE:
            if self.body.holding_object:
                released = self.body.holding_object
                self.body.holding_object = None
                result["success"] = True
                result["message"] = f"Released {released}"
            else:
                result["message"] = "Not holding anything"
        
        return result
    
    def ground_understanding(
        self,
        abstract_instruction: str
    ) -> str:
        """Ground abstract instruction in physical terms"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Translate abstract instruction into concrete physical actions.

Current embodied state:
- Position: {self.body.position}
- Holding: {self.body.holding_object or 'nothing'}
- Energy: {self.body.energy_level:.0%}

Known objects:
{self._get_objects_description()}

Provide specific physical actions with spatial details."""),
            ("user", "Instruction: {instruction}\n\nPhysical actions:")
        ])
        
        chain = prompt | self.cognition | StrOutputParser()
        return chain.invoke({"instruction": abstract_instruction})
    
    def _get_objects_description(self) -> str:
        """Get description of perceived objects"""
        if not self.perceived_objects:
            return "No objects perceived yet"
        
        desc = ""
        for obj in self.perceived_objects.values():
            dist = self.body.position.distance_to(obj.position)
            desc += f"- {obj.name} at {obj.position} (distance: {dist:.1f}m)\n"
        return desc


def demonstrate_embodied_agent():
    """Demonstrate Embodied Agent pattern"""
    
    print("=" * 80)
    print("PATTERN 066: EMBODIED AGENT DEMONSTRATION")
    print("=" * 80)
    print("\nPhysical embodiment: sensors, actuators, spatial reasoning\n")
    
    # Test 1: Perception and spatial awareness
    print("\n" + "=" * 80)
    print("TEST 1: Perception and Spatial Awareness")
    print("=" * 80)
    
    agent = EmbodiedAgent(
        initial_position=Position(0, 0, 0),
        reach_distance=1.5,
        move_speed=1.0
    )
    
    print(f"\n🤖 Agent initialized at position {agent.body.position}")
    
    # Perceive environment
    environment = """
    You are in a room. To your left (at -2, 0) is a table with a cup on it.
    Straight ahead (at 0, 3) is a door. To your right (at 2, 0) is a chair.
    """
    
    print(f"\n👁️  Perceiving environment...")
    perception = agent.perceive(SensorType.VISION, environment)
    print(f"   Sensor: {perception.sensor.value}")
    print(f"   Perceived: {perception.data[:150]}...")
    
    # Add objects to agent's knowledge
    agent.perceived_objects["table"] = PhysicalObject(
        "table", "table", Position(-2, 0, 0), (1.0, 0.8, 0.1)
    )
    agent.perceived_objects["cup"] = PhysicalObject(
        "cup", "cup", Position(-2, 0, 0.8), (0.1, 0.15, 0.1),
        properties={"containable": True, "graspable": True}
    )
    agent.perceived_objects["door"] = PhysicalObject(
        "door", "door", Position(0, 3, 0), (1.0, 2.0, 0.1)
    )
    
    print(f"\n📍 Objects in environment:")
    for obj_id, obj in agent.perceived_objects.items():
        distance = agent.body.position.distance_to(obj.position)
        reachable = "✓" if distance <= agent.reach_distance else "✗"
        print(f"   {reachable} {obj.name} at {obj.position} (distance: {distance:.1f}m)")
    
    # Test 2: Affordance detection
    print("\n" + "=" * 80)
    print("TEST 2: Affordance Detection")
    print("=" * 80)
    
    cup = agent.perceived_objects["cup"]
    
    print(f"\n🔍 Identifying affordances for: {cup.name}")
    print(f"   Position: {cup.position}")
    print(f"   Size: {cup.size[0]}×{cup.size[1]}×{cup.size[2]}m")
    print(f"   Distance from agent: {agent.body.position.distance_to(cup.position):.1f}m")
    
    affordances = agent.identify_affordances(cup)
    print(f"\n   Possible actions:")
    for affordance in affordances:
        print(f"      • {affordance}")
    
    # Test 3: Navigation planning
    print("\n" + "=" * 80)
    print("TEST 3: Navigation Planning")
    print("=" * 80)
    
    target = Position(-2, 0, 0)  # Table position
    obstacles = [agent.perceived_objects["chair"]]
    
    print(f"\n🗺️  Planning path:")
    print(f"   From: {agent.body.position}")
    print(f"   To: {target}")
    print(f"   Obstacles: {[o.name for o in obstacles]}")
    
    nav_plan = agent.plan_navigation(target, obstacles)
    
    print(f"\n   Navigation plan ({len(nav_plan)} steps):")
    for action in nav_plan[:5]:  # Show first 5
        print(f"      {action['step']}. {action['description']}")
    if len(nav_plan) > 5:
        print(f"      ... ({len(nav_plan) - 5} more steps)")
    
    # Test 4: Action execution
    print("\n" + "=" * 80)
    print("TEST 4: Physical Action Execution")
    print("=" * 80)
    
    print(f"\n🎮 Executing actions:\n")
    
    # Move to table
    print(f"   Action 1: Move to table")
    result1 = agent.execute_action(
        ActionType.MOVE,
        {"target_position": Position(-2, 0, 0)}
    )
    print(f"      Result: {result1['message']}")
    print(f"      New position: {agent.body.position}")
    
    # Try to grasp cup
    print(f"\n   Action 2: Grasp cup")
    result2 = agent.execute_action(
        ActionType.GRASP,
        {"object_id": "cup"}
    )
    print(f"      Result: {result2['message']}")
    print(f"      Holding: {agent.body.holding_object}")
    
    # Try to grasp something else (should fail)
    print(f"\n   Action 3: Try to grasp door (should fail)")
    result3 = agent.execute_action(
        ActionType.GRASP,
        {"object_id": "door"}
    )
    print(f"      Result: {result3['message']}")
    
    # Release cup
    print(f"\n   Action 4: Release cup")
    result4 = agent.execute_action(ActionType.RELEASE, {})
    print(f"      Result: {result4['message']}")
    print(f"      Holding: {agent.body.holding_object or 'nothing'}")
    
    # Test 5: Grounding abstract instructions
    print("\n" + "=" * 80)
    print("TEST 5: Grounding Abstract Instructions")
    print("=" * 80)
    
    agent2 = EmbodiedAgent(initial_position=Position(0, 0, 0))
    agent2.perceived_objects = {
        "book": PhysicalObject("book", "book", Position(1, 1, 0.5), (0.2, 0.3, 0.05)),
        "shelf": PhysicalObject("shelf", "shelf", Position(3, 0, 1.0), (1.0, 2.0, 0.3))
    }
    
    instructions = [
        "Get the book",
        "Put it on the shelf",
        "Return to starting position"
    ]
    
    print(f"\n📝 Grounding abstract instructions in physical terms:\n")
    
    for instruction in instructions:
        print(f"   Abstract: '{instruction}'")
        grounded = agent2.ground_understanding(instruction)
        print(f"   Grounded:")
        for line in grounded.split('\n')[:3]:
            if line.strip():
                print(f"      {line.strip()}")
        print()
    
    # Summary
    print("\n" + "=" * 80)
    print("EMBODIED AGENT PATTERN SUMMARY")
    print("=" * 80)
    print("""
Key Benefits:
1. Grounded Understanding: Physical basis for abstract concepts
2. Spatial Reasoning: Navigate and manipulate in 3D space
3. Action Constraints: Realistic physical limitations
4. Sensorimotor Integration: Coordinated perception-action
5. Natural Interaction: Physical presence enables natural tasks

Embodiment Components:

1. Sensory System:
   - Vision: Object detection, scene understanding
   - Hearing: Sound localization, speech
   - Touch: Contact, texture, force
   - Proprioception: Body position, posture
   - Distance: Range sensing, depth

2. Motor System:
   - Locomotion: Walking, rolling, flying
   - Manipulation: Grasping, moving objects
   - Gestures: Pointing, waving
   - Facial expressions (if humanoid)
   - Tool use

3. Body Schema:
   - Size and shape awareness
   - Reach distance
   - Movement capabilities
   - Center of mass
   - Degrees of freedom

4. Spatial Representation:
   - Egocentric: Relative to body ("to my left")
   - Allocentric: World-centered ("north of the tree")
   - Route knowledge: Paths between places
   - Survey knowledge: Map-like representation

5. Affordances:
   - Graspable: Can be picked up
   - Sittable: Can sit on
   - Pushable: Can be moved
   - Openable: Can be opened
   - Context-dependent

6. Physical Constraints:
   - Reachability: Within arm's reach
   - Traversability: Can move through
   - Stability: Won't fall over
   - Energy: Battery/fuel limitations
   - Payload: Weight capacity

Grounding Language in Action:
- "Pick up": Approach + reach + grasp + lift
- "Put on": Move + position + align + release
- "Go to": Plan path + navigate + avoid obstacles
- "Look at": Turn head/body + focus vision

Use Cases:
- Physical Robotics:
  - Warehouse robots
  - Service robots
  - Manufacturing robots
  - Exploration rovers

- Virtual Embodiment:
  - Game characters
  - Virtual assistants with avatars
  - Simulation agents
  - Training environments

- Human-Robot Interaction:
  - Collaborative robots (cobots)
  - Social robots
  - Assistive robots
  - Telepresence

Challenges:
1. Sensor Noise: Imperfect perception
2. Motor Control: Precise actuation
3. Real-time Processing: Fast reactions needed
4. Sim-to-Real Transfer: Simulations differ from reality
5. Safety: Physical harm risks
6. Complexity: Many degrees of freedom

Best Practices:
1. Start with simplified sensing
2. Use world models for planning
3. Implement safety constraints
4. Test in simulation first
5. Incremental complexity
6. Fallback behaviors
7. Human oversight for learning

Production Considerations:
- Sensor calibration and maintenance
- Safety zones and limits
- Emergency stop mechanisms
- Predictable failure modes
- Logging for debugging
- Remote monitoring
- Regular testing

Spatial Reasoning Tasks:
- Object localization
- Path planning
- Obstacle avoidance
- Reachability analysis
- Collision detection
- Distance estimation
- Orientation tracking

Integration with AI:
- Vision: Computer vision, object detection
- Planning: Motion planning, SLAM
- Control: PID, MPC, RL
- Learning: Imitation learning, RL
- Language: Instruction grounding

Comparison with Related Patterns:
- vs. Abstract Agent: Physical vs symbolic
- vs. Disembodied: Grounded vs abstract
- vs. Simulated: Real physics vs approximation
- vs. Teleoperated: Autonomous vs controlled

Advanced Techniques:
1. Visual Servoing: Vision-guided control
2. Force Control: Manipulation with force feedback
3. Dynamic Balance: Maintaining stability
4. Compliant Motion: Adaptive to contact
5. Multi-Modal Fusion: Combine sensor types

Embodied Cognition Theory:
- Mind is shaped by body
- Cognition is for action
- Environment as external memory
- Situated activity
- Sensorimotor contingencies

Research Directions:
- Sim-to-real transfer
- Learning from demonstration
- Embodied language understanding
- Social embodiment
- Morphological computation

Design Principles:
1. Sense-Plan-Act cycle
2. Reactive behaviors for safety
3. Deliberative planning for complex tasks
4. Hierarchical control
5. Graceful degradation

Testing Approaches:
- Unit tests for each component
- Integration tests for behaviors
- Simulation before deployment
- Staged real-world testing
- Safety-critical validation

The Embodied Agent pattern grounds AI systems in physical
reality through sensors, actuators, and spatial reasoning,
enabling natural interaction with the physical world.
""")


if __name__ == "__main__":
    demonstrate_embodied_agent()
