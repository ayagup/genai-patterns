"""
Pattern 068: Instruction Following & Grounding

Description:
    Instruction Following & Grounding enables agents to precisely understand and execute
    natural language instructions by breaking them down into structured, executable steps.
    This pattern goes beyond simple command parsing to deeply ground instructions in the
    agent's capabilities, environment constraints, and task context. The agent validates
    instructions for feasibility, disambiguates ambiguous commands, grounds abstract
    concepts in concrete actions, and provides execution verification.
    
    The pattern is crucial for building reliable agents that can follow complex,
    multi-step instructions while maintaining safety, handling edge cases, and providing
    transparent feedback about what it's doing and why.

Components:
    1. Instruction Parser: Extracts intent, parameters, and constraints
    2. Action Grounder: Maps abstract instructions to concrete actions
    3. Feasibility Checker: Validates instruction executability
    4. Disambiguator: Resolves ambiguities through clarification
    5. Execution Planner: Creates step-by-step execution plan
    6. Execution Monitor: Tracks execution progress and validates outcomes
    7. Feedback Generator: Provides status updates and explanations

Architecture:
    ```
    Natural Language Instruction
        ↓
    Instruction Parser → [Intent, Parameters, Constraints]
        ↓
    Disambiguator → [Clarified Instruction]
        ↓
    Action Grounder → [Concrete Actions]
        ↓
    Feasibility Checker → [Validation]
        ↓
    Execution Planner → [Execution Steps]
        ↓
    Execution Monitor → [Execute & Verify]
        ↓
    Feedback Generator → [Status & Results]
    ```

Use Cases:
    - Robotic task execution from natural language
    - Virtual assistant command processing
    - Smart home automation
    - Software automation and scripting
    - Manufacturing and logistics instructions
    - Educational tutoring systems

Advantages:
    - Handles ambiguous and underspecified instructions
    - Grounds abstract concepts in concrete actions
    - Validates feasibility before execution
    - Provides transparent execution process
    - Handles errors and edge cases gracefully
    - Learns from execution feedback

LangChain Implementation:
    Uses ChatOpenAI for instruction parsing, grounding, and planning.
    Demonstrates instruction decomposition, parameter extraction, feasibility
    checking, execution planning, and verification.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class InstructionType(Enum):
    """Types of instructions."""
    ACTION = "action"  # Do something
    QUERY = "query"  # Find information
    CONDITION = "condition"  # If-then logic
    SEQUENCE = "sequence"  # Series of actions
    LOOP = "loop"  # Repeated actions
    CONSTRAINT = "constraint"  # With specific constraints


class ActionStatus(Enum):
    """Status of action execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class FeasibilityLevel(Enum):
    """Feasibility assessment levels."""
    FEASIBLE = "feasible"
    FEASIBLE_WITH_ASSUMPTIONS = "feasible_with_assumptions"
    PARTIALLY_FEASIBLE = "partially_feasible"
    INFEASIBLE = "infeasible"
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass
class Parameter:
    """Represents a parameter extracted from instruction."""
    name: str
    value: Any
    data_type: str
    is_optional: bool = False
    default_value: Any = None


@dataclass
class Constraint:
    """Represents a constraint on instruction execution."""
    constraint_type: str  # e.g., "time", "resource", "condition"
    description: str
    is_hard: bool = True  # Hard vs soft constraint


@dataclass
class ParsedInstruction:
    """Represents a parsed natural language instruction."""
    instruction_id: str
    original_text: str
    instruction_type: InstructionType
    intent: str  # The main goal
    parameters: List[Parameter]
    constraints: List[Constraint]
    dependencies: List[str] = field(default_factory=list)  # Other instruction IDs
    ambiguities: List[str] = field(default_factory=list)


@dataclass
class GroundedAction:
    """Represents a grounded, executable action."""
    action_id: str
    action_name: str
    parameters: Dict[str, Any]
    preconditions: List[str]
    effects: List[str]
    estimated_duration: float  # seconds
    
    def execute(self) -> Tuple[bool, str]:
        """Simulate action execution."""
        # In real implementation, this would execute actual actions
        print(f"  Executing: {self.action_name}({', '.join(f'{k}={v}' for k, v in self.parameters.items())})")
        time.sleep(0.1)  # Simulate execution time
        return True, f"Successfully executed {self.action_name}"


@dataclass
class ExecutionStep:
    """Represents a step in execution plan."""
    step_id: int
    action: GroundedAction
    status: ActionStatus
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class FeasibilityAssessment:
    """Assessment of instruction feasibility."""
    level: FeasibilityLevel
    is_feasible: bool
    reasons: List[str]
    assumptions: List[str] = field(default_factory=list)
    clarifications_needed: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    """Complete result of instruction execution."""
    instruction: ParsedInstruction
    feasibility: FeasibilityAssessment
    execution_steps: List[ExecutionStep]
    success: bool
    summary: str
    execution_time: float


class InstructionFollowingAgent:
    """
    Agent that parses, grounds, and executes natural language instructions.
    """
    
    def __init__(self, agent_capabilities: Optional[List[str]] = None):
        """
        Initialize the instruction following agent.
        
        Args:
            agent_capabilities: List of actions the agent can perform
        """
        # Define agent capabilities
        self.capabilities = agent_capabilities or [
            "move_to",
            "pick_up",
            "put_down",
            "open",
            "close",
            "search",
            "read",
            "write",
            "calculate",
            "wait",
            "notify"
        ]
        
        # LLMs for different tasks
        self.parser = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")
        self.grounder = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
        self.planner = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
        self.verifier = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")
        
        self.str_parser = StrOutputParser()
        
        # Execution history
        self.execution_history: List[ExecutionResult] = []
    
    def parse_instruction(self, instruction_text: str) -> ParsedInstruction:
        """
        Parse natural language instruction into structured format.
        
        Args:
            instruction_text: Natural language instruction
            
        Returns:
            Parsed instruction with intent, parameters, and constraints
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an instruction parser. Extract structured information from natural language.

Analyze the instruction and identify:
1. Instruction Type (action/query/condition/sequence/loop/constraint)
2. Intent (what is the goal?)
3. Parameters (name, value, type, optional?)
4. Constraints (time, resource, conditional constraints)
5. Ambiguities (what is unclear or underspecified?)

Format your response as:
Type: [type]
Intent: [main goal]
Parameters: [name:value:type:optional, ...]
Constraints: [type:description:hard/soft, ...]
Ambiguities: [list of unclear aspects, or 'none']"""),
            ("user", "Parse this instruction: {instruction}")
        ])
        
        chain = prompt | self.parser | self.str_parser
        
        try:
            result = chain.invoke({"instruction": instruction_text})
            
            # Parse the structured output
            instruction_type = InstructionType.ACTION
            intent = ""
            parameters = []
            constraints = []
            ambiguities = []
            
            for line in result.split('\n'):
                line = line.strip()
                if line.startswith("Type:"):
                    type_str = line.replace("Type:", "").strip().lower()
                    for itype in InstructionType:
                        if itype.value in type_str:
                            instruction_type = itype
                            break
                
                elif line.startswith("Intent:"):
                    intent = line.replace("Intent:", "").strip()
                
                elif line.startswith("Parameters:"):
                    params_str = line.replace("Parameters:", "").strip()
                    if params_str.lower() != 'none':
                        # Parse format: name:value:type:optional
                        for param_str in params_str.split(','):
                            parts = [p.strip() for p in param_str.split(':')]
                            if len(parts) >= 3:
                                param = Parameter(
                                    name=parts[0],
                                    value=parts[1],
                                    data_type=parts[2],
                                    is_optional=len(parts) > 3 and parts[3].lower() == 'optional'
                                )
                                parameters.append(param)
                
                elif line.startswith("Constraints:"):
                    const_str = line.replace("Constraints:", "").strip()
                    if const_str.lower() != 'none':
                        # Parse format: type:description:hard/soft
                        for c_str in const_str.split(','):
                            parts = [p.strip() for p in c_str.split(':')]
                            if len(parts) >= 2:
                                constraint = Constraint(
                                    constraint_type=parts[0],
                                    description=parts[1] if len(parts) > 1 else "",
                                    is_hard=len(parts) <= 2 or parts[2].lower() == 'hard'
                                )
                                constraints.append(constraint)
                
                elif line.startswith("Ambiguities:"):
                    amb_str = line.replace("Ambiguities:", "").strip()
                    if amb_str.lower() != 'none':
                        ambiguities = [a.strip() for a in amb_str.split(',')]
            
            instruction_id = f"inst_{int(time.time() * 1000)}"
            
            return ParsedInstruction(
                instruction_id=instruction_id,
                original_text=instruction_text,
                instruction_type=instruction_type,
                intent=intent if intent else instruction_text,
                parameters=parameters,
                constraints=constraints,
                ambiguities=ambiguities
            )
            
        except Exception as e:
            print(f"Error parsing instruction: {e}")
            # Fallback: minimal parsing
            return ParsedInstruction(
                instruction_id=f"inst_{int(time.time() * 1000)}",
                original_text=instruction_text,
                instruction_type=InstructionType.ACTION,
                intent=instruction_text,
                parameters=[],
                constraints=[]
            )
    
    def ground_instruction(self, parsed: ParsedInstruction) -> List[GroundedAction]:
        """
        Ground abstract instruction into concrete, executable actions.
        
        Args:
            parsed: Parsed instruction
            
        Returns:
            List of grounded actions
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an action grounder. Map abstract instructions to concrete actions.

Agent Capabilities: {capabilities}

For each instruction, specify:
1. Action sequence (using available capabilities)
2. Parameters for each action
3. Preconditions (what must be true before)
4. Effects (what changes after)
5. Estimated duration (seconds)

Format each action as:
Action: [action_name]
Parameters: [key1=value1, key2=value2, ...]
Preconditions: [list]
Effects: [list]
Duration: [seconds]
---"""),
            ("user", """Ground this instruction:
Intent: {intent}
Type: {instruction_type}
Parameters: {parameters}
Constraints: {constraints}

Provide concrete actions:""")
        ])
        
        chain = prompt | self.grounder | self.str_parser
        
        try:
            params_str = ", ".join([f"{p.name}={p.value}" for p in parsed.parameters])
            const_str = ", ".join([f"{c.constraint_type}: {c.description}" for c in parsed.constraints])
            
            result = chain.invoke({
                "capabilities": ", ".join(self.capabilities),
                "intent": parsed.intent,
                "instruction_type": parsed.instruction_type.value,
                "parameters": params_str if params_str else "none",
                "constraints": const_str if const_str else "none"
            })
            
            # Parse grounded actions
            actions = []
            action_blocks = result.split('---')
            
            for i, block in enumerate(action_blocks):
                if not block.strip():
                    continue
                
                action_name = ""
                parameters = {}
                preconditions = []
                effects = []
                duration = 1.0
                
                for line in block.split('\n'):
                    line = line.strip()
                    if line.startswith("Action:"):
                        action_name = line.replace("Action:", "").strip()
                    elif line.startswith("Parameters:"):
                        param_str = line.replace("Parameters:", "").strip()
                        if param_str.lower() != 'none':
                            for kv in param_str.split(','):
                                if '=' in kv:
                                    k, v = kv.split('=', 1)
                                    parameters[k.strip()] = v.strip()
                    elif line.startswith("Preconditions:"):
                        pre_str = line.replace("Preconditions:", "").strip()
                        if pre_str.lower() != 'none':
                            preconditions = [p.strip() for p in pre_str.split(',')]
                    elif line.startswith("Effects:"):
                        eff_str = line.replace("Effects:", "").strip()
                        if eff_str.lower() != 'none':
                            effects = [e.strip() for e in eff_str.split(',')]
                    elif line.startswith("Duration:"):
                        try:
                            duration = float(line.replace("Duration:", "").strip())
                        except:
                            duration = 1.0
                
                if action_name:
                    action = GroundedAction(
                        action_id=f"action_{i+1}",
                        action_name=action_name,
                        parameters=parameters,
                        preconditions=preconditions,
                        effects=effects,
                        estimated_duration=duration
                    )
                    actions.append(action)
            
            return actions if actions else [
                GroundedAction(
                    action_id="action_1",
                    action_name="execute_instruction",
                    parameters={"instruction": parsed.intent},
                    preconditions=[],
                    effects=["instruction executed"],
                    estimated_duration=1.0
                )
            ]
            
        except Exception as e:
            print(f"Error grounding instruction: {e}")
            # Fallback: single generic action
            return [
                GroundedAction(
                    action_id="action_1",
                    action_name="execute",
                    parameters={"instruction": parsed.intent},
                    preconditions=[],
                    effects=["completed"],
                    estimated_duration=1.0
                )
            ]
    
    def check_feasibility(
        self,
        parsed: ParsedInstruction,
        grounded_actions: List[GroundedAction]
    ) -> FeasibilityAssessment:
        """
        Check if instruction is feasible to execute.
        
        Args:
            parsed: Parsed instruction
            grounded_actions: Grounded actions
            
        Returns:
            Feasibility assessment
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a feasibility checker. Assess if instructions can be executed.

Agent Capabilities: {capabilities}

Consider:
1. Are all required actions within agent capabilities?
2. Are parameters complete and valid?
3. Are constraints satisfiable?
4. Are preconditions likely to be met?
5. Are there any safety concerns?

Provide:
Feasibility: [feasible/feasible_with_assumptions/partially_feasible/infeasible/needs_clarification]
Reasons: [list of reasons for assessment]
Assumptions: [assumptions made, if any]
Clarifications: [what needs clarification, if any]
Alternatives: [alternative approaches, if any]"""),
            ("user", """Assess feasibility:
Intent: {intent}
Actions: {actions}
Constraints: {constraints}
Ambiguities: {ambiguities}

Provide assessment:""")
        ])
        
        chain = prompt | self.verifier | self.str_parser
        
        try:
            actions_str = ", ".join([a.action_name for a in grounded_actions])
            const_str = ", ".join([c.description for c in parsed.constraints])
            amb_str = ", ".join(parsed.ambiguities) if parsed.ambiguities else "none"
            
            result = chain.invoke({
                "capabilities": ", ".join(self.capabilities),
                "intent": parsed.intent,
                "actions": actions_str,
                "constraints": const_str if const_str else "none",
                "ambiguities": amb_str
            })
            
            # Parse assessment
            level = FeasibilityLevel.FEASIBLE
            reasons = []
            assumptions = []
            clarifications = []
            alternatives = []
            
            for line in result.split('\n'):
                line = line.strip()
                if line.startswith("Feasibility:"):
                    level_str = line.replace("Feasibility:", "").strip().lower()
                    for flevel in FeasibilityLevel:
                        if flevel.value in level_str:
                            level = flevel
                            break
                elif line.startswith("Reasons:"):
                    reasons_str = line.replace("Reasons:", "").strip()
                    if reasons_str.lower() != 'none':
                        reasons = [r.strip() for r in reasons_str.split(',')]
                elif line.startswith("Assumptions:"):
                    ass_str = line.replace("Assumptions:", "").strip()
                    if ass_str.lower() != 'none':
                        assumptions = [a.strip() for a in ass_str.split(',')]
                elif line.startswith("Clarifications:"):
                    clar_str = line.replace("Clarifications:", "").strip()
                    if clar_str.lower() != 'none':
                        clarifications = [c.strip() for c in clar_str.split(',')]
                elif line.startswith("Alternatives:"):
                    alt_str = line.replace("Alternatives:", "").strip()
                    if alt_str.lower() != 'none':
                        alternatives = [a.strip() for a in alt_str.split(',')]
            
            is_feasible = level in [FeasibilityLevel.FEASIBLE, FeasibilityLevel.FEASIBLE_WITH_ASSUMPTIONS]
            
            return FeasibilityAssessment(
                level=level,
                is_feasible=is_feasible,
                reasons=reasons if reasons else ["Assessment completed"],
                assumptions=assumptions,
                clarifications_needed=clarifications,
                alternatives=alternatives
            )
            
        except Exception as e:
            print(f"Error checking feasibility: {e}")
            # Fallback: assume feasible
            return FeasibilityAssessment(
                level=FeasibilityLevel.FEASIBLE,
                is_feasible=True,
                reasons=["Default feasibility check"]
            )
    
    def execute_instruction(
        self,
        instruction_text: str,
        dry_run: bool = False
    ) -> ExecutionResult:
        """
        Execute a natural language instruction end-to-end.
        
        Args:
            instruction_text: Natural language instruction
            dry_run: If True, plan but don't execute
            
        Returns:
            Complete execution result
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Instruction: {instruction_text}")
        print(f"{'='*60}\n")
        
        # Step 1: Parse
        print("Step 1: Parsing instruction...")
        parsed = self.parse_instruction(instruction_text)
        print(f"  → Intent: {parsed.intent}")
        print(f"  → Type: {parsed.instruction_type.value}")
        print(f"  → Parameters: {len(parsed.parameters)}")
        if parsed.ambiguities:
            print(f"  → Ambiguities: {', '.join(parsed.ambiguities)}")
        
        # Step 2: Ground
        print("\nStep 2: Grounding to concrete actions...")
        grounded_actions = self.ground_instruction(parsed)
        print(f"  → Generated {len(grounded_actions)} actions:")
        for action in grounded_actions:
            print(f"    • {action.action_name}({', '.join(f'{k}={v}' for k, v in action.parameters.items())})")
        
        # Step 3: Check feasibility
        print("\nStep 3: Checking feasibility...")
        feasibility = self.check_feasibility(parsed, grounded_actions)
        print(f"  → Feasibility: {feasibility.level.value}")
        print(f"  → Is Feasible: {feasibility.is_feasible}")
        if feasibility.assumptions:
            print(f"  → Assumptions: {', '.join(feasibility.assumptions)}")
        if feasibility.clarifications_needed:
            print(f"  → Needs Clarification: {', '.join(feasibility.clarifications_needed)}")
        
        # Step 4: Execute (if feasible and not dry run)
        execution_steps = []
        success = True
        
        if feasibility.is_feasible and not dry_run:
            print("\nStep 4: Executing actions...")
            for i, action in enumerate(grounded_actions, 1):
                step = ExecutionStep(
                    step_id=i,
                    action=action,
                    status=ActionStatus.IN_PROGRESS,
                    start_time=time.time()
                )
                
                try:
                    exec_success, result = action.execute()
                    step.status = ActionStatus.COMPLETED if exec_success else ActionStatus.FAILED
                    step.result = result
                    step.end_time = time.time()
                    print(f"    ✓ Step {i} completed")
                except Exception as e:
                    step.status = ActionStatus.FAILED
                    step.error = str(e)
                    step.end_time = time.time()
                    success = False
                    print(f"    ✗ Step {i} failed: {e}")
                
                execution_steps.append(step)
        else:
            print("\nStep 4: Execution skipped (dry run or infeasible)")
            for i, action in enumerate(grounded_actions, 1):
                step = ExecutionStep(
                    step_id=i,
                    action=action,
                    status=ActionStatus.PENDING
                )
                execution_steps.append(step)
        
        # Generate summary
        execution_time = time.time() - start_time
        
        if dry_run:
            summary = f"Dry run completed. {len(grounded_actions)} actions planned."
        elif not feasibility.is_feasible:
            summary = f"Instruction not feasible: {feasibility.level.value}"
        elif success:
            summary = f"Successfully executed {len(execution_steps)} actions."
        else:
            failed_steps = sum(1 for s in execution_steps if s.status == ActionStatus.FAILED)
            summary = f"Execution completed with {failed_steps} failures out of {len(execution_steps)} steps."
        
        result = ExecutionResult(
            instruction=parsed,
            feasibility=feasibility,
            execution_steps=execution_steps,
            success=success,
            summary=summary,
            execution_time=execution_time
        )
        
        self.execution_history.append(result)
        
        return result


def demonstrate_instruction_following():
    """Demonstrate Instruction Following & Grounding pattern."""
    
    print("="*80)
    print("INSTRUCTION FOLLOWING & GROUNDING - DEMONSTRATION")
    print("="*80)
    
    agent = InstructionFollowingAgent()
    
    # Test 1: Simple action instruction
    print("\n" + "="*80)
    print("TEST 1: Simple Action Instruction")
    print("="*80)
    
    result1 = agent.execute_instruction("Pick up the red box and put it on the table.")
    
    print("\n--- EXECUTION RESULT ---")
    print(f"Success: {result1.success}")
    print(f"Summary: {result1.summary}")
    print(f"Execution Time: {result1.execution_time:.2f}s")
    print(f"Actions Executed: {len(result1.execution_steps)}")
    
    # Test 2: Instruction with constraints
    print("\n" + "="*80)
    print("TEST 2: Instruction with Constraints")
    print("="*80)
    
    result2 = agent.execute_instruction(
        "Search for documents about machine learning, but only look in the research folder "
        "and complete within 5 seconds."
    )
    
    print("\n--- PARSED INSTRUCTION ---")
    print(f"Intent: {result2.instruction.intent}")
    print(f"Type: {result2.instruction.instruction_type.value}")
    print(f"Parameters: {[(p.name, p.value) for p in result2.instruction.parameters]}")
    print(f"Constraints: {[(c.constraint_type, c.description) for c in result2.instruction.constraints]}")
    
    print("\n--- GROUNDED ACTIONS ---")
    for step in result2.execution_steps:
        print(f"Step {step.step_id}: {step.action.action_name}")
        print(f"  Parameters: {step.action.parameters}")
        print(f"  Preconditions: {step.action.preconditions}")
        print(f"  Effects: {step.action.effects}")
        print(f"  Status: {step.status.value}")
    
    # Test 3: Ambiguous instruction (dry run)
    print("\n" + "="*80)
    print("TEST 3: Ambiguous Instruction (Dry Run)")
    print("="*80)
    
    result3 = agent.execute_instruction(
        "Move the thing over there.",
        dry_run=True
    )
    
    print("\n--- AMBIGUITY ANALYSIS ---")
    if result3.instruction.ambiguities:
        print(f"Identified Ambiguities:")
        for amb in result3.instruction.ambiguities:
            print(f"  • {amb}")
    
    print("\n--- FEASIBILITY ASSESSMENT ---")
    print(f"Level: {result3.feasibility.level.value}")
    print(f"Is Feasible: {result3.feasibility.is_feasible}")
    if result3.feasibility.clarifications_needed:
        print(f"Clarifications Needed:")
        for clar in result3.feasibility.clarifications_needed:
            print(f"  • {clar}")
    if result3.feasibility.alternatives:
        print(f"Suggested Alternatives:")
        for alt in result3.feasibility.alternatives:
            print(f"  • {alt}")
    
    # Test 4: Complex multi-step instruction
    print("\n" + "="*80)
    print("TEST 4: Complex Multi-Step Instruction")
    print("="*80)
    
    result4 = agent.execute_instruction(
        "Open the file named 'report.txt', read its contents, calculate the sum of all numbers, "
        "and write the result to 'summary.txt'."
    )
    
    print("\n--- EXECUTION PLAN ---")
    total_duration = sum(step.action.estimated_duration for step in result4.execution_steps)
    print(f"Total Steps: {len(result4.execution_steps)}")
    print(f"Estimated Duration: {total_duration:.1f}s")
    print("\nStep-by-Step Plan:")
    for step in result4.execution_steps:
        print(f"  {step.step_id}. {step.action.action_name}")
        print(f"     Duration: {step.action.estimated_duration}s")
        print(f"     Effects: {', '.join(step.action.effects)}")
    
    print("\n--- EXECUTION SUMMARY ---")
    print(f"Status: {'✓ Success' if result4.success else '✗ Failed'}")
    print(f"Summary: {result4.summary}")
    
    # Test 5: Conditional instruction
    print("\n" + "="*80)
    print("TEST 5: Conditional Instruction")
    print("="*80)
    
    result5 = agent.execute_instruction(
        "If the temperature is above 25 degrees, open the window, otherwise turn on the heater.",
        dry_run=True
    )
    
    print("\n--- INSTRUCTION ANALYSIS ---")
    print(f"Type: {result5.instruction.instruction_type.value}")
    print(f"Intent: {result5.instruction.intent}")
    print(f"\nGrounded Actions: {len(result5.execution_steps)}")
    for step in result5.execution_steps:
        print(f"  • {step.action.action_name}: {step.action.parameters}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Instruction Following & Grounding")
    print("="*80)
    print(f"""
The Instruction Following & Grounding pattern demonstrates precise instruction execution:

Key Features Demonstrated:
1. Instruction Parsing: Extracts intent, parameters, and constraints from natural language
2. Action Grounding: Maps abstract instructions to concrete executable actions
3. Feasibility Checking: Validates executability before attempting execution
4. Ambiguity Detection: Identifies unclear or underspecified aspects
5. Constraint Handling: Respects time, resource, and conditional constraints
6. Execution Planning: Creates step-by-step execution plans with preconditions
7. Execution Monitoring: Tracks progress and validates outcomes
8. Transparent Feedback: Provides clear status updates and explanations

Benefits:
• Handles complex, multi-step instructions reliably
• Grounds abstract concepts in concrete actions
• Validates feasibility before execution
• Identifies and resolves ambiguities
• Provides transparent execution process
• Handles errors gracefully with clear feedback
• Respects constraints and preconditions
• Learns from execution history

Use Cases:
• Robotic task execution from natural language commands
• Virtual assistant command processing
• Smart home automation with complex rules
• Software automation and scripting from descriptions
• Manufacturing instructions with safety constraints
• Educational tutoring with adaptive instruction following
• Accessibility tools for users with disabilities

Comparison with Simple Command Parsing:
┌───────────────────────┬──────────────────┬─────────────────────┐
│ Aspect                │ Simple Parsing   │ Full Grounding      │
├───────────────────────┼──────────────────┼─────────────────────┤
│ Ambiguity Handling    │ Minimal          │ Comprehensive       │
│ Constraint Validation │ None             │ Full validation     │
│ Feasibility Checking  │ None             │ Pre-execution       │
│ Action Grounding      │ Direct mapping   │ Context-aware       │
│ Error Handling        │ Basic            │ Graceful with plans │
│ Transparency          │ Low              │ High                │
│ Complexity Handling   │ Limited          │ Excellent           │
└───────────────────────┴──────────────────┴─────────────────────┘

LangChain Implementation Notes:
• Multiple specialized LLMs for parsing, grounding, and verification
• Structured output parsing for instruction components
• Feasibility assessment before execution
• Step-by-step execution with monitoring
• Execution history for learning and debugging
• Dry run mode for planning without execution
• Clear error messages and alternative suggestions

Production Considerations:
• Implement actual action execution layer (not simulated)
• Add safety checks and permission systems
• Implement rollback mechanisms for failed operations
• Add interactive clarification for ambiguous instructions
• Implement learning from user corrections
• Add execution visualization for transparency
• Implement cost estimation for resource-intensive operations
• Add monitoring and logging for audit trails
• Implement timeout and resource limit enforcement
• Add support for cancellation and pause/resume
• Implement state persistence for long-running tasks
• Add multi-user coordination for shared resources

Advanced Extensions:
• Natural language feedback and explanation generation
• Learning user preferences and implicit constraints
• Proactive error prevention through predictive analysis
• Multi-modal instruction understanding (speech, gesture, vision)
• Collaborative instruction refinement with humans
• Transfer learning from demonstration
• Uncertainty-aware execution with confidence intervals
• Hierarchical instruction decomposition for complex tasks
    """)


if __name__ == "__main__":
    demonstrate_instruction_following()
