"""
Pattern 109: Adaptive Planning

Description:
    The Adaptive Planning pattern enables agents to dynamically adjust their plans
    in response to execution feedback, environmental changes, and unexpected events.
    Rather than rigidly following a predetermined plan, adaptive planners monitor
    execution, detect deviations from expected outcomes, diagnose problems, and
    modify plans on the fly to maintain progress toward goals.
    
    Adaptive planning is essential in dynamic, uncertain environments where initial
    assumptions may become invalid, resources may change, or unexpected opportunities
    and obstacles may arise. The pattern combines continuous monitoring, deviation
    detection, root cause analysis, and replanning strategies to ensure resilience
    and goal achievement despite changing conditions.
    
    This pattern includes mechanisms for execution monitoring, deviation detection,
    impact assessment, root cause diagnosis, plan repair strategies, replanning
    triggers, and learning from outcomes to improve future plans.

Key Components:
    1. Execution Monitor: Tracks plan execution
    2. Deviation Detector: Identifies plan-reality gaps
    3. Impact Assessor: Evaluates deviation significance
    4. Root Cause Analyzer: Diagnoses problems
    5. Plan Repairer: Fixes plans locally
    6. Replanner: Creates new plans when needed
    7. Learning Module: Improves from experience

Adaptation Strategies:
    1. Plan Repair: Minor modifications to existing plan
    2. Plan Reuse: Apply similar past plan
    3. Plan Revision: Significant restructuring
    4. Replanning: Create entirely new plan
    5. Reactive Adaptation: Immediate adjustments
    6. Proactive Adaptation: Anticipate changes
    7. Learning-based: Use past outcomes

Monitoring Aspects:
    1. Progress: Task completion vs. expected
    2. Resources: Availability and consumption
    3. Time: Schedule adherence
    4. Quality: Output quality measures
    5. Cost: Budget tracking
    6. Risks: Emerging threats
    7. Opportunities: New possibilities

Use Cases:
    - Robot navigation in dynamic environments
    - Project management with changing requirements
    - Manufacturing with equipment failures
    - Supply chain disruptions
    - Military operations in uncertain terrain
    - Autonomous vehicle route planning
    - Resource allocation in cloud systems

Advantages:
    - Robust to uncertainty
    - Maintains progress despite changes
    - Exploits new opportunities
    - Reduces plan failures
    - Learns from experience
    - Handles unexpected events
    - Flexible goal achievement

Challenges:
    - Monitoring overhead
    - Replanning latency
    - Plan stability vs. adaptability
    - Knowing when to replan
    - Avoiding thrashing (excessive replanning)
    - Preserving progress
    - Computational cost

LangChain Implementation:
    This implementation uses LangChain for:
    - LLM-based deviation analysis
    - Root cause diagnosis
    - Plan repair suggestions
    - Natural language explanations
    
Production Considerations:
    - Implement efficient monitoring
    - Use event-driven detection
    - Cache replan results
    - Set replanning thresholds
    - Limit replanning frequency
    - Preserve partial progress
    - Log all adaptations
    - Enable replay for debugging
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class StepStatus(Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DeviationType(Enum):
    """Types of plan deviations."""
    DELAY = "delay"
    RESOURCE_SHORTAGE = "resource_shortage"
    QUALITY_ISSUE = "quality_issue"
    COST_OVERRUN = "cost_overrun"
    DEPENDENCY_FAILURE = "dependency_failure"
    ENVIRONMENTAL_CHANGE = "environmental_change"


class AdaptationAction(Enum):
    """Types of adaptation actions."""
    CONTINUE = "continue"
    REPAIR = "repair"
    REVISE = "revise"
    REPLAN = "replan"
    ABORT = "abort"


@dataclass
class PlanStep:
    """
    A step in an adaptive plan.
    
    Attributes:
        step_id: Unique identifier
        description: Step description
        expected_duration: Expected time to complete
        expected_cost: Expected cost
        dependencies: Prerequisite steps
        status: Current status
        started_at: When started
        completed_at: When completed
        actual_duration: Actual time taken
        actual_cost: Actual cost
    """
    step_id: str
    description: str
    expected_duration: float = 1.0
    expected_cost: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actual_duration: Optional[float] = None
    actual_cost: Optional[float] = None


@dataclass
class Deviation:
    """
    Detected deviation from plan.
    
    Attributes:
        deviation_id: Unique identifier
        deviation_type: Type of deviation
        affected_step: Step with deviation
        severity: Severity score (0-10)
        description: Human-readable description
        detected_at: When detected
        resolved: Whether resolved
    """
    deviation_id: str
    deviation_type: DeviationType
    affected_step: str
    severity: float
    description: str
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False


@dataclass
class Adaptation:
    """
    Adaptation made to plan.
    
    Attributes:
        adaptation_id: Unique identifier
        action: Type of adaptation
        reason: Why adaptation was made
        changes: Description of changes
        timestamp: When adaptation occurred
        success: Whether adaptation succeeded
    """
    adaptation_id: str
    action: AdaptationAction
    reason: str
    changes: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: Optional[bool] = None


class AdaptivePlanner:
    """
    Adaptive planning system with execution monitoring and replanning.
    
    This planner monitors execution, detects deviations, and adapts
    plans dynamically to maintain progress toward goals.
    """
    
    def __init__(self, temperature: float = 0.3):
        """
        Initialize adaptive planner.
        
        Args:
            temperature: LLM temperature
        """
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.steps: Dict[str, PlanStep] = {}
        self.deviations: List[Deviation] = []
        self.adaptations: List[Adaptation] = []
        self.step_counter = 0
        self.deviation_counter = 0
        self.adaptation_counter = 0
        self.replanning_threshold = 5.0  # Severity threshold for replanning
    
    def create_plan(self, goal: str, num_steps: int = 5) -> List[PlanStep]:
        """
        Create initial plan using LLM.
        
        Args:
            goal: Goal description
            num_steps: Number of steps
            
        Returns:
            List of plan steps
        """
        prompt = ChatPromptTemplate.from_template(
            "Create a {num_steps}-step plan to achieve this goal:\n\n"
            "Goal: {goal}\n\n"
            "For each step, provide:\n"
            "- Description\n"
            "- Expected duration (hours)\n"
            "- Expected cost (units)\n\n"
            "Format as:\n"
            "Step N: [description]\n"
            "Duration: [hours]\n"
            "Cost: [units]\n"
            "---\n"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({"goal": goal, "num_steps": num_steps})
        
        # Parse steps
        steps = []
        step_blocks = result.split("---")
        
        for block in step_blocks:
            if not block.strip():
                continue
            
            lines = [line.strip() for line in block.strip().split('\n')]
            description = ""
            duration = 1.0
            cost = 1.0
            
            for line in lines:
                if line.startswith("Step"):
                    description = line.split(":", 1)[1].strip() if ":" in line else line
                elif line.startswith("Duration:"):
                    try:
                        duration = float(line.split(":")[1].strip().split()[0])
                    except:
                        duration = 1.0
                elif line.startswith("Cost:"):
                    try:
                        cost = float(line.split(":")[1].strip().split()[0])
                    except:
                        cost = 1.0
            
            if description:
                self.step_counter += 1
                step = PlanStep(
                    step_id=f"step_{self.step_counter}",
                    description=description,
                    expected_duration=duration,
                    expected_cost=cost
                )
                self.steps[step.step_id] = step
                steps.append(step)
        
        return steps
    
    def execute_step(
        self,
        step_id: str,
        actual_duration: Optional[float] = None,
        actual_cost: Optional[float] = None,
        success: bool = True
    ) -> bool:
        """
        Execute a plan step.
        
        Args:
            step_id: Step to execute
            actual_duration: Actual time taken
            actual_cost: Actual cost
            success: Whether execution succeeded
            
        Returns:
            Success status
        """
        if step_id not in self.steps:
            return False
        
        step = self.steps[step_id]
        
        # Start execution
        if step.status == StepStatus.PENDING:
            step.status = StepStatus.IN_PROGRESS
            step.started_at = datetime.now()
        
        # Complete execution
        if step.status == StepStatus.IN_PROGRESS:
            if success:
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.now()
                step.actual_duration = actual_duration or step.expected_duration
                step.actual_cost = actual_cost or step.expected_cost
            else:
                step.status = StepStatus.FAILED
        
        return success
    
    def monitor_execution(self) -> List[Deviation]:
        """
        Monitor plan execution and detect deviations.
        
        Returns:
            List of new deviations
        """
        new_deviations = []
        
        for step in self.steps.values():
            if step.status == StepStatus.COMPLETED:
                # Check for delays
                if step.actual_duration and step.actual_duration > step.expected_duration * 1.2:
                    self.deviation_counter += 1
                    deviation = Deviation(
                        deviation_id=f"dev_{self.deviation_counter}",
                        deviation_type=DeviationType.DELAY,
                        affected_step=step.step_id,
                        severity=min(10.0, (step.actual_duration / step.expected_duration - 1) * 10),
                        description=f"Step took {step.actual_duration:.1f}h vs expected {step.expected_duration:.1f}h"
                    )
                    self.deviations.append(deviation)
                    new_deviations.append(deviation)
                
                # Check for cost overruns
                if step.actual_cost and step.actual_cost > step.expected_cost * 1.2:
                    self.deviation_counter += 1
                    deviation = Deviation(
                        deviation_id=f"dev_{self.deviation_counter}",
                        deviation_type=DeviationType.COST_OVERRUN,
                        affected_step=step.step_id,
                        severity=min(10.0, (step.actual_cost / step.expected_cost - 1) * 10),
                        description=f"Step cost {step.actual_cost:.1f} vs expected {step.expected_cost:.1f}"
                    )
                    self.deviations.append(deviation)
                    new_deviations.append(deviation)
            
            elif step.status == StepStatus.FAILED:
                # Failure is high severity
                self.deviation_counter += 1
                deviation = Deviation(
                    deviation_id=f"dev_{self.deviation_counter}",
                    deviation_type=DeviationType.DEPENDENCY_FAILURE,
                    affected_step=step.step_id,
                    severity=9.0,
                    description=f"Step {step.step_id} failed"
                )
                self.deviations.append(deviation)
                new_deviations.append(deviation)
        
        return new_deviations
    
    def diagnose_deviation(self, deviation: Deviation) -> str:
        """
        Diagnose root cause of deviation using LLM.
        
        Args:
            deviation: Deviation to diagnose
            
        Returns:
            Diagnosis text
        """
        step = self.steps[deviation.affected_step]
        
        prompt = ChatPromptTemplate.from_template(
            "Analyze this plan deviation and suggest root causes:\n\n"
            "Step: {step_description}\n"
            "Deviation: {deviation_type}\n"
            "Details: {deviation_description}\n"
            "Severity: {severity}/10\n\n"
            "Provide:\n"
            "1. Likely root cause\n"
            "2. Impact on remaining plan\n"
            "3. Recommended action (continue/repair/replan)\n"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        diagnosis = chain.invoke({
            "step_description": step.description,
            "deviation_type": deviation.deviation_type.value,
            "deviation_description": deviation.description,
            "severity": deviation.severity
        })
        
        return diagnosis
    
    def decide_adaptation(self, deviation: Deviation) -> AdaptationAction:
        """
        Decide what adaptation action to take.
        
        Args:
            deviation: Deviation triggering adaptation
            
        Returns:
            Recommended action
        """
        # Simple severity-based decision
        if deviation.severity < 3.0:
            return AdaptationAction.CONTINUE
        elif deviation.severity < 5.0:
            return AdaptationAction.REPAIR
        elif deviation.severity < 8.0:
            return AdaptationAction.REVISE
        else:
            return AdaptationAction.REPLAN
    
    def repair_plan(self, deviation: Deviation) -> Adaptation:
        """
        Repair plan by adjusting expectations.
        
        Args:
            deviation: Deviation to repair
            
        Returns:
            Adaptation record
        """
        step = self.steps[deviation.affected_step]
        
        # Adjust expectations for remaining steps
        remaining_steps = [
            s for s in self.steps.values()
            if s.status == StepStatus.PENDING
        ]
        
        if deviation.deviation_type == DeviationType.DELAY:
            # Increase duration estimates
            for s in remaining_steps:
                s.expected_duration *= 1.2
            changes = f"Increased duration estimates by 20% for {len(remaining_steps)} remaining steps"
        
        elif deviation.deviation_type == DeviationType.COST_OVERRUN:
            # Increase cost estimates
            for s in remaining_steps:
                s.expected_cost *= 1.2
            changes = f"Increased cost estimates by 20% for {len(remaining_steps)} remaining steps"
        
        else:
            changes = "No specific repair applied"
        
        self.adaptation_counter += 1
        adaptation = Adaptation(
            adaptation_id=f"adapt_{self.adaptation_counter}",
            action=AdaptationAction.REPAIR,
            reason=deviation.description,
            changes=changes,
            success=True
        )
        self.adaptations.append(adaptation)
        
        return adaptation
    
    def replan_from_current(self, goal: str) -> List[PlanStep]:
        """
        Create new plan from current state.
        
        Args:
            goal: Original goal
            
        Returns:
            New plan steps
        """
        # Preserve completed steps
        completed = [s for s in self.steps.values() if s.status == StepStatus.COMPLETED]
        
        # Create new steps for remaining work
        prompt = ChatPromptTemplate.from_template(
            "The original plan encountered issues. Create a revised plan.\n\n"
            "Goal: {goal}\n"
            "Completed steps: {completed_count}\n"
            "Previous issues: {issues}\n\n"
            "Create 3-5 steps to complete the remaining work.\n"
            "Consider lessons learned from previous issues.\n\n"
            "Format as:\n"
            "Step N: [description]\n"
            "Duration: [hours]\n"
            "Cost: [units]\n"
            "---\n"
        )
        
        issues = ", ".join([d.description for d in self.deviations[-3:]])
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "goal": goal,
            "completed_count": len(completed),
            "issues": issues or "None"
        })
        
        # Parse new steps
        new_steps = []
        step_blocks = result.split("---")
        
        for block in step_blocks:
            if not block.strip():
                continue
            
            lines = [line.strip() for line in block.strip().split('\n')]
            description = ""
            duration = 1.0
            cost = 1.0
            
            for line in lines:
                if line.startswith("Step"):
                    description = line.split(":", 1)[1].strip() if ":" in line else line
                elif line.startswith("Duration:"):
                    try:
                        duration = float(line.split(":")[1].strip().split()[0])
                    except:
                        duration = 1.0
                elif line.startswith("Cost:"):
                    try:
                        cost = float(line.split(":")[1].strip().split()[0])
                    except:
                        cost = 1.0
            
            if description:
                self.step_counter += 1
                step = PlanStep(
                    step_id=f"step_{self.step_counter}",
                    description=description,
                    expected_duration=duration,
                    expected_cost=cost
                )
                self.steps[step.step_id] = step
                new_steps.append(step)
        
        # Record adaptation
        self.adaptation_counter += 1
        adaptation = Adaptation(
            adaptation_id=f"adapt_{self.adaptation_counter}",
            action=AdaptationAction.REPLAN,
            reason="Significant deviations detected",
            changes=f"Created {len(new_steps)} new steps",
            success=True
        )
        self.adaptations.append(adaptation)
        
        return new_steps
    
    def adapt(self, goal: str) -> Optional[Adaptation]:
        """
        Monitor and adapt plan as needed.
        
        Args:
            goal: Original goal
            
        Returns:
            Adaptation if made, None otherwise
        """
        # Monitor for deviations
        new_deviations = self.monitor_execution()
        
        if not new_deviations:
            return None
        
        # Check most severe deviation
        most_severe = max(new_deviations, key=lambda d: d.severity)
        
        # Decide action
        action = self.decide_adaptation(most_severe)
        
        # Execute adaptation
        if action == AdaptationAction.CONTINUE:
            return None
        elif action == AdaptationAction.REPAIR:
            return self.repair_plan(most_severe)
        elif action == AdaptationAction.REPLAN:
            self.replan_from_current(goal)
            return self.adaptations[-1]
        
        return None
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get plan progress summary."""
        total = len(self.steps)
        completed = sum(1 for s in self.steps.values() if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.steps.values() if s.status == StepStatus.FAILED)
        in_progress = sum(1 for s in self.steps.values() if s.status == StepStatus.IN_PROGRESS)
        
        total_expected_duration = sum(s.expected_duration for s in self.steps.values())
        total_actual_duration = sum(
            s.actual_duration for s in self.steps.values()
            if s.actual_duration is not None
        )
        
        return {
            "total_steps": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": total - completed - failed - in_progress,
            "completion_rate": completed / total if total > 0 else 0,
            "total_deviations": len(self.deviations),
            "total_adaptations": len(self.adaptations),
            "expected_duration": total_expected_duration,
            "actual_duration": total_actual_duration
        }


def demonstrate_adaptive_planning():
    """Demonstrate adaptive planning pattern."""
    
    print("=" * 80)
    print("ADAPTIVE PLANNING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic adaptive planning
    print("\n" + "=" * 80)
    print("Example 1: Plan Creation and Execution")
    print("=" * 80)
    
    planner = AdaptivePlanner()
    
    print("\nCreating initial plan...")
    steps = planner.create_plan("Develop a mobile app", num_steps=5)
    
    print(f"\nCreated {len(steps)} steps:")
    for step in steps:
        print(f"  {step.step_id}: {step.description}")
        print(f"    Expected: {step.expected_duration}h, ${step.expected_cost}")
    
    print("\nExecuting steps...")
    for i, step in enumerate(steps[:2]):
        planner.execute_step(step.step_id)
        print(f"  ✓ Completed: {step.description}")
    
    progress = planner.get_progress_summary()
    print(f"\nProgress: {progress['completed']}/{progress['total_steps']} steps completed")
    
    # Example 2: Deviation detection
    print("\n" + "=" * 80)
    print("Example 2: Deviation Detection")
    print("=" * 80)
    
    planner2 = AdaptivePlanner()
    planner2.create_plan("Build website", num_steps=4)
    
    # Execute with delays
    steps2 = list(planner2.steps.values())
    planner2.execute_step(steps2[0].step_id, actual_duration=3.0, actual_cost=1.0)
    steps2[0].expected_duration = 2.0  # Set expected for comparison
    
    print("\nExecuted step with delay:")
    print(f"  Expected: {steps2[0].expected_duration}h")
    print(f"  Actual: {steps2[0].actual_duration}h")
    print(f"  Delay: {steps2[0].actual_duration - steps2[0].expected_duration:.1f}h")
    
    print("\nMonitoring for deviations...")
    deviations = planner2.monitor_execution()
    
    print(f"\nDetected {len(deviations)} deviation(s):")
    for dev in deviations:
        print(f"  • Type: {dev.deviation_type.value}")
        print(f"    Severity: {dev.severity:.1f}/10")
        print(f"    Description: {dev.description}")
    
    # Example 3: Root cause diagnosis
    print("\n" + "=" * 80)
    print("Example 3: Root Cause Diagnosis")
    print("=" * 80)
    
    if deviations:
        dev = deviations[0]
        print(f"\nDiagnosing deviation: {dev.description}")
        print("\nLLM Analysis:")
        diagnosis = planner2.diagnose_deviation(dev)
        print(diagnosis)
    
    # Example 4: Plan repair
    print("\n" + "=" * 80)
    print("Example 4: Plan Repair")
    print("=" * 80)
    
    planner3 = AdaptivePlanner()
    planner3.create_plan("Complete project", num_steps=5)
    
    # Simulate deviation
    steps3 = list(planner3.steps.values())
    planner3.execute_step(steps3[0].step_id, actual_duration=5.0)
    steps3[0].expected_duration = 2.0
    
    deviations3 = planner3.monitor_execution()
    
    if deviations3:
        print("\nDeviation detected:")
        print(f"  Severity: {deviations3[0].severity:.1f}/10")
        
        print("\nDeciding adaptation action...")
        action = planner3.decide_adaptation(deviations3[0])
        print(f"  Recommended: {action.value}")
        
        print("\nRepairing plan...")
        adaptation = planner3.repair_plan(deviations3[0])
        print(f"  Changes: {adaptation.changes}")
    
    # Example 5: Full replanning
    print("\n" + "=" * 80)
    print("Example 5: Full Replanning")
    print("=" * 80)
    
    planner4 = AdaptivePlanner()
    original_steps = planner4.create_plan("Launch product", num_steps=4)
    
    print(f"\nOriginal plan: {len(original_steps)} steps")
    
    # Simulate major failure
    steps4 = list(planner4.steps.values())
    planner4.execute_step(steps4[0].step_id)
    planner4.execute_step(steps4[1].step_id, success=False)
    steps4[1].status = StepStatus.FAILED
    
    print("\nMajor failure detected!")
    deviations4 = planner4.monitor_execution()
    
    if deviations4:
        severe_dev = max(deviations4, key=lambda d: d.severity)
        print(f"  Severity: {severe_dev.severity:.1f}/10")
        
        action = planner4.decide_adaptation(severe_dev)
        print(f"  Action: {action.value}")
        
        if action == AdaptationAction.REPLAN:
            print("\nReplanning from current state...")
            new_steps = planner4.replan_from_current("Launch product")
            print(f"  Created {len(new_steps)} new steps")
    
    # Example 6: Adaptation decision tree
    print("\n" + "=" * 80)
    print("Example 6: Adaptation Decision Making")
    print("=" * 80)
    
    planner5 = AdaptivePlanner()
    
    # Test different severity levels
    test_cases = [
        (2.0, "Minor delay"),
        (4.5, "Moderate cost overrun"),
        (7.0, "Significant quality issue"),
        (9.5, "Critical failure")
    ]
    
    print("\nAdaptation decisions by severity:")
    for severity, description in test_cases:
        planner5.step_counter += 1
        test_step = PlanStep(
            step_id=f"step_{planner5.step_counter}",
            description="Test step"
        )
        planner5.steps[test_step.step_id] = test_step
        
        planner5.deviation_counter += 1
        test_dev = Deviation(
            deviation_id=f"dev_{planner5.deviation_counter}",
            deviation_type=DeviationType.DELAY,
            affected_step=test_step.step_id,
            severity=severity,
            description=description
        )
        
        action = planner5.decide_adaptation(test_dev)
        print(f"  Severity {severity:.1f} ({description}) → {action.value}")
    
    # Example 7: Automatic adaptation loop
    print("\n" + "=" * 80)
    print("Example 7: Automatic Adaptation Loop")
    print("=" * 80)
    
    planner6 = AdaptivePlanner()
    planner6.create_plan("Deploy service", num_steps=4)
    
    print("\nExecuting with automatic adaptation...")
    
    steps6 = list(planner6.steps.values())
    
    # Step 1: Normal
    planner6.execute_step(steps6[0].step_id, actual_duration=1.0)
    adaptation = planner6.adapt("Deploy service")
    print(f"  Step 1: Completed normally - {adaptation or 'No adaptation needed'}")
    
    # Step 2: Minor delay
    planner6.execute_step(steps6[1].step_id, actual_duration=3.0)
    steps6[1].expected_duration = 2.0
    adaptation = planner6.adapt("Deploy service")
    print(f"  Step 2: Minor delay - {adaptation.action.value if adaptation else 'Continue'}")
    
    # Step 3: Major issue
    planner6.execute_step(steps6[2].step_id, actual_duration=10.0)
    steps6[2].expected_duration = 2.0
    adaptation = planner6.adapt("Deploy service")
    print(f"  Step 3: Major issue - {adaptation.action.value if adaptation else 'Continue'}")
    
    print(f"\nTotal adaptations: {len(planner6.adaptations)}")
    
    # Example 8: Learning from experience
    print("\n" + "=" * 80)
    print("Example 8: Progress Tracking and Summary")
    print("=" * 80)
    
    planner7 = AdaptivePlanner()
    planner7.create_plan("Complete migration", num_steps=6)
    
    # Execute with various outcomes
    steps7 = list(planner7.steps.values())
    planner7.execute_step(steps7[0].step_id, actual_duration=1.5, actual_cost=1.0)
    steps7[0].expected_duration = 1.0
    planner7.execute_step(steps7[1].step_id, actual_duration=3.0, actual_cost=2.0)
    steps7[1].expected_duration = 2.0
    steps7[1].expected_cost = 1.5
    planner7.execute_step(steps7[2].step_id, actual_duration=1.0, actual_cost=1.0)
    
    # Monitor and adapt
    planner7.adapt("Complete migration")
    
    print("\nFINAL SUMMARY:")
    print("=" * 60)
    
    progress = planner7.get_progress_summary()
    for key, value in progress.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\nDeviations:")
    for dev in planner7.deviations:
        print(f"  • {dev.deviation_type.value} (severity: {dev.severity:.1f})")
    
    print("\nAdaptations:")
    for adapt in planner7.adaptations:
        print(f"  • {adapt.action.value}: {adapt.changes}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Adaptive Planning Pattern")
    print("=" * 80)
    
    summary = """
    The Adaptive Planning pattern demonstrated:
    
    1. PLAN CREATION (Example 1):
       - LLM-based plan generation
       - Step-by-step structure
       - Duration and cost estimation
       - Progress tracking
       - Execution monitoring
    
    2. DEVIATION DETECTION (Example 2):
       - Automatic monitoring
       - Delay detection (20% threshold)
       - Cost overrun detection
       - Severity scoring
       - Real-time alerts
    
    3. ROOT CAUSE DIAGNOSIS (Example 3):
       - LLM-based analysis
       - Context-aware diagnosis
       - Impact assessment
       - Actionable recommendations
       - Natural language explanations
    
    4. PLAN REPAIR (Example 4):
       - Local plan adjustments
       - Expectation updates
       - Propagation to remaining steps
       - Minimal disruption
       - Quick fixes
    
    5. FULL REPLANNING (Example 5):
       - New plan generation
       - Progress preservation
       - Lesson incorporation
       - Context-aware replanning
       - Fresh start when needed
    
    6. DECISION MAKING (Example 6):
       - Severity-based decisions
       - Action selection (continue/repair/revise/replan)
       - Graduated response
       - Threshold-based triggers
       - Balanced approach
    
    7. AUTOMATIC ADAPTATION (Example 7):
       - Continuous monitoring
       - Automatic response
       - Multi-step adaptation
       - Event-driven updates
       - Hands-free operation
    
    8. PROGRESS TRACKING (Example 8):
       - Comprehensive metrics
       - Deviation history
       - Adaptation log
       - Performance summary
       - Complete audit trail
    
    KEY BENEFITS:
    ✓ Robust to uncertainty and changes
    ✓ Maintains progress despite issues
    ✓ Exploits new opportunities
    ✓ Reduces plan failure rate
    ✓ Learns from experience
    ✓ Handles unexpected events
    ✓ Flexible goal achievement
    ✓ Automatic problem response
    
    USE CASES:
    • Robot navigation (dynamic environments)
    • Project management (changing requirements)
    • Manufacturing (equipment failures)
    • Supply chain (disruptions)
    • Military operations (uncertain terrain)
    • Autonomous vehicles (route planning)
    • Cloud resource allocation
    • Emergency response coordination
    
    ADAPTATION STRATEGIES:
    → Plan Repair: Minor modifications
    → Plan Reuse: Apply similar past plan
    → Plan Revision: Significant restructuring
    → Replanning: Create entirely new plan
    → Reactive: Immediate adjustments
    → Proactive: Anticipate changes
    
    MONITORING ASPECTS:
    • Progress: Completion vs. expected
    • Resources: Availability and consumption
    • Time: Schedule adherence
    • Quality: Output measures
    • Cost: Budget tracking
    • Risks: Emerging threats
    
    BEST PRACTICES:
    1. Implement efficient monitoring
    2. Use event-driven detection
    3. Cache replan results
    4. Set replanning thresholds
    5. Limit replanning frequency
    6. Preserve partial progress
    7. Log all adaptations
    8. Enable replay for debugging
    
    TRADE-OFFS:
    • Monitoring overhead vs. responsiveness
    • Plan stability vs. adaptability
    • Replanning cost vs. plan quality
    • Automation vs. human control
    
    PRODUCTION CONSIDERATIONS:
    → Monitor continuously without blocking
    → Use asynchronous event processing
    → Implement replanning rate limits
    → Cache similar plan patterns
    → Log all deviations and adaptations
    → Support manual override
    → Enable plan playback
    → Track adaptation success rate
    → Alert on excessive replanning
    → Preserve completed work
    
    This pattern enables plans to evolve dynamically in response to
    execution feedback and environmental changes, maintaining progress
    toward goals despite uncertainty and unexpected events.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_adaptive_planning()
