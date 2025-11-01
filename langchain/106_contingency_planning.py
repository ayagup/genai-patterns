"""
Pattern 106: Contingency Planning

Description:
    The Contingency Planning pattern prepares agents to handle unexpected events
    and failures by creating alternative plans before they're needed. This pattern
    involves anticipating potential problems, creating backup plans, and enabling
    quick adaptation when things don't go as expected.
    
    Contingency planning is crucial for reliable agent systems that operate in
    uncertain, dynamic environments. Rather than reacting to failures after they
    occur, agents proactively identify risks, assess their likelihood and impact,
    and prepare alternative strategies. This reduces downtime, improves robustness,
    and enables graceful degradation when primary plans fail.
    
    The pattern includes risk identification, scenario planning, plan B/C/D
    creation, trigger condition specification, and rapid plan switching. It
    balances the cost of planning for contingencies against the benefit of
    being prepared for failures.

Key Components:
    1. Risk Identifier: Detects potential failure points
    2. Scenario Generator: Creates alternative scenarios
    3. Contingency Creator: Develops backup plans
    4. Trigger Monitor: Watches for contingency conditions
    5. Plan Switcher: Activates alternative plans
    6. Impact Assessor: Evaluates risk severity
    7. Recovery Planner: Plans return to primary plan

Contingency Types:
    1. Alternative Action: Different way to achieve same goal
    2. Fallback Goal: Reduced/modified objective
    3. Resource Substitution: Use different resources
    4. Path Alternative: Different sequence of steps
    5. Graceful Degradation: Accept reduced functionality
    6. Abort and Recover: Abandon task, clean up

Risk Assessment:
    1. Likelihood: Probability of occurrence (low/medium/high)
    2. Impact: Severity of consequences (minor/moderate/critical)
    3. Priority: Likelihood × Impact
    4. Mitigation Cost: Resources to prepare contingency
    
Planning Phases:
    1. Risk Identification: What could go wrong?
    2. Impact Analysis: How bad would it be?
    3. Contingency Development: What's plan B?
    4. Trigger Specification: When to switch plans?
    5. Monitoring: Watch for triggers
    6. Activation: Switch to contingency
    7. Recovery: Return to normal operations

Use Cases:
    - Mission-critical operations
    - Uncertain environments
    - Resource-constrained systems
    - High-stakes decision making
    - Autonomous systems
    - Real-time operations
    - Safety-critical applications

Advantages:
    - Reduces response time to failures
    - Improves system reliability
    - Enables graceful degradation
    - Proactive vs reactive approach
    - Builds resilience
    - Reduces costly downtime
    - Increases confidence

Challenges:
    - Computational cost of planning
    - Predicting all possible failures
    - Balancing thoroughness vs. efficiency
    - Maintaining multiple plans
    - Detecting when to switch
    - Avoiding over-planning
    - Resource allocation for contingencies

LangChain Implementation:
    This implementation uses LangChain for:
    - LLM-based risk identification
    - Alternative plan generation
    - Scenario analysis
    - Natural language contingency description
    
Production Considerations:
    - Focus on high-impact, high-probability risks
    - Use tiered contingency levels (B, C, D plans)
    - Implement fast plan switching
    - Monitor trigger conditions efficiently
    - Test contingency plans regularly
    - Document all contingencies
    - Balance planning cost vs. benefit
    - Enable human override when needed
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class RiskLevel(Enum):
    """Risk likelihood levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImpactLevel(Enum):
    """Impact severity levels."""
    MINOR = "minor"
    MODERATE = "moderate"
    CRITICAL = "critical"


class PlanStatus(Enum):
    """Status of a plan."""
    PRIMARY = "primary"
    CONTINGENCY = "contingency"
    ACTIVE = "active"
    FAILED = "failed"
    SUCCEEDED = "succeeded"


@dataclass
class Risk:
    """
    Represents a potential risk or failure point.
    
    Attributes:
        risk_id: Unique identifier
        description: Risk description
        likelihood: Probability of occurrence
        impact: Severity of consequences
        trigger_condition: Condition that indicates risk occurred
        affected_steps: Plan steps affected by this risk
    """
    risk_id: str
    description: str
    likelihood: RiskLevel
    impact: ImpactLevel
    trigger_condition: str
    affected_steps: List[str] = field(default_factory=list)
    
    @property
    def priority(self) -> int:
        """Calculate risk priority."""
        likelihood_scores = {RiskLevel.LOW: 1, RiskLevel.MEDIUM: 2, RiskLevel.HIGH: 3}
        impact_scores = {ImpactLevel.MINOR: 1, ImpactLevel.MODERATE: 2, ImpactLevel.CRITICAL: 3}
        return likelihood_scores[self.likelihood] * impact_scores[self.impact]


@dataclass
class Plan:
    """
    Represents a plan (primary or contingency).
    
    Attributes:
        plan_id: Unique identifier
        description: Plan description
        steps: List of action steps
        status: Plan status
        parent_plan_id: ID of plan this is contingency for
        triggers: Conditions that activate this plan
        cost: Estimated cost/resources
        success_criteria: How to know if plan succeeded
    """
    plan_id: str
    description: str
    steps: List[str]
    status: PlanStatus
    parent_plan_id: Optional[str] = None
    triggers: List[str] = field(default_factory=list)
    cost: float = 1.0
    success_criteria: List[str] = field(default_factory=list)
    activated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ContingencyPlanner:
    """
    Contingency planning system that creates and manages backup plans.
    
    This planner identifies risks, creates alternative plans, monitors
    for trigger conditions, and switches to contingencies when needed.
    """
    
    def __init__(self, temperature: float = 0.4):
        """
        Initialize contingency planner.
        
        Args:
            temperature: LLM temperature
        """
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")
        self.plans: Dict[str, Plan] = {}
        self.risks: Dict[str, Risk] = {}
        self.active_plan_id: Optional[str] = None
        self.plan_counter = 0
        self.risk_counter = 0
        self.event_log: List[Dict[str, Any]] = []
    
    def create_primary_plan(
        self,
        description: str,
        steps: List[str]
    ) -> Plan:
        """
        Create primary plan.
        
        Args:
            description: Plan description
            steps: Plan steps
            
        Returns:
            Created plan
        """
        self.plan_counter += 1
        plan = Plan(
            plan_id=f"plan_{self.plan_counter}",
            description=description,
            steps=steps,
            status=PlanStatus.PRIMARY
        )
        
        self.plans[plan.plan_id] = plan
        self.active_plan_id = plan.plan_id
        
        return plan
    
    def identify_risks(self, plan: Plan) -> List[Risk]:
        """
        Identify potential risks in a plan using LLM.
        
        Args:
            plan: Plan to analyze
            
        Returns:
            List of identified risks
        """
        prompt = ChatPromptTemplate.from_template(
            "Analyze this plan and identify 3-5 potential risks or failure points:\n\n"
            "Plan: {description}\n"
            "Steps:\n{steps}\n\n"
            "For each risk, provide:\n"
            "- Description of what could go wrong\n"
            "- Likelihood (low/medium/high)\n"
            "- Impact (minor/moderate/critical)\n"
            "- Trigger condition (how to detect it happened)\n\n"
            "Format each risk as:\n"
            "Risk: [description]\n"
            "Likelihood: [level]\n"
            "Impact: [level]\n"
            "Trigger: [condition]\n"
            "---\n"
        )
        
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan.steps))
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "description": plan.description,
            "steps": steps_text
        })
        
        # Parse risks from LLM output
        risks = []
        risk_blocks = result.split("---")
        
        for block in risk_blocks:
            if not block.strip():
                continue
            
            lines = [line.strip() for line in block.strip().split('\n')]
            risk_desc = ""
            likelihood = RiskLevel.MEDIUM
            impact = ImpactLevel.MODERATE
            trigger = ""
            
            for line in lines:
                if line.startswith("Risk:"):
                    risk_desc = line[5:].strip()
                elif line.startswith("Likelihood:"):
                    level = line[11:].strip().lower()
                    likelihood = RiskLevel(level) if level in [l.value for l in RiskLevel] else RiskLevel.MEDIUM
                elif line.startswith("Impact:"):
                    level = line[7:].strip().lower()
                    impact = ImpactLevel(level) if level in [i.value for i in ImpactLevel] else ImpactLevel.MODERATE
                elif line.startswith("Trigger:"):
                    trigger = line[8:].strip()
            
            if risk_desc and trigger:
                self.risk_counter += 1
                risk = Risk(
                    risk_id=f"risk_{self.risk_counter}",
                    description=risk_desc,
                    likelihood=likelihood,
                    impact=impact,
                    trigger_condition=trigger
                )
                self.risks[risk.risk_id] = risk
                risks.append(risk)
        
        return risks
    
    def create_contingency_plan(
        self,
        primary_plan: Plan,
        risk: Risk,
        level: str = "B"
    ) -> Plan:
        """
        Create contingency plan for a specific risk.
        
        Args:
            primary_plan: Primary plan
            risk: Risk to mitigate
            level: Contingency level (B, C, D, etc.)
            
        Returns:
            Contingency plan
        """
        prompt = ChatPromptTemplate.from_template(
            "Create a contingency plan (Plan {level}) for this situation:\n\n"
            "Primary Plan: {plan_description}\n"
            "Risk: {risk_description}\n"
            "Trigger: {trigger}\n\n"
            "Design an alternative plan that:\n"
            "1. Achieves the same or similar goal\n"
            "2. Avoids or mitigates the identified risk\n"
            "3. Is practical and feasible\n\n"
            "Provide 3-5 concrete steps for the contingency plan.\n"
            "List each step on a new line, numbered.\n\n"
            "Steps:"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "level": level,
            "plan_description": primary_plan.description,
            "risk_description": risk.description,
            "trigger": risk.trigger_condition
        })
        
        # Parse steps
        lines = [line.strip() for line in result.strip().split('\n') if line.strip()]
        steps = []
        
        for line in lines:
            # Remove numbering
            step = line
            for prefix in ["1.", "2.", "3.", "4.", "5.", "1)", "2)", "3)", "4)", "5)", "-", "*"]:
                if step.startswith(prefix):
                    step = step[len(prefix):].strip()
                    break
            if step:
                steps.append(step)
        
        # Create contingency plan
        self.plan_counter += 1
        contingency = Plan(
            plan_id=f"plan_{self.plan_counter}",
            description=f"Plan {level}: {primary_plan.description} (contingency for {risk.description})",
            steps=steps,
            status=PlanStatus.CONTINGENCY,
            parent_plan_id=primary_plan.plan_id,
            triggers=[risk.trigger_condition]
        )
        
        self.plans[contingency.plan_id] = contingency
        
        return contingency
    
    def create_tiered_contingencies(
        self,
        primary_plan: Plan,
        num_tiers: int = 2
    ) -> List[Plan]:
        """
        Create multiple tiers of contingency plans.
        
        Args:
            primary_plan: Primary plan
            num_tiers: Number of contingency tiers
            
        Returns:
            List of contingency plans
        """
        # Identify risks
        risks = self.identify_risks(primary_plan)
        
        # Sort by priority
        risks.sort(key=lambda r: r.priority, reverse=True)
        
        # Create contingencies for top risks
        contingencies = []
        levels = ["B", "C", "D", "E", "F"]
        
        for i in range(min(num_tiers, len(risks), len(levels))):
            contingency = self.create_contingency_plan(
                primary_plan,
                risks[i],
                level=levels[i]
            )
            contingencies.append(contingency)
        
        return contingencies
    
    def check_trigger_conditions(
        self,
        observed_events: List[str]
    ) -> Optional[Plan]:
        """
        Check if any contingency should be activated.
        
        Args:
            observed_events: List of observed events/conditions
            
        Returns:
            Contingency plan to activate, or None
        """
        # Check contingency plans for matching triggers
        contingency_plans = [
            p for p in self.plans.values()
            if p.status == PlanStatus.CONTINGENCY
        ]
        
        for plan in contingency_plans:
            for trigger in plan.triggers:
                for event in observed_events:
                    if trigger.lower() in event.lower() or event.lower() in trigger.lower():
                        return plan
        
        return None
    
    def activate_contingency(
        self,
        contingency_plan: Plan,
        reason: str
    ):
        """
        Activate a contingency plan.
        
        Args:
            contingency_plan: Contingency to activate
            reason: Reason for activation
        """
        # Deactivate current plan
        if self.active_plan_id and self.active_plan_id in self.plans:
            self.plans[self.active_plan_id].status = PlanStatus.FAILED
        
        # Activate contingency
        contingency_plan.status = PlanStatus.ACTIVE
        contingency_plan.activated_at = datetime.now()
        self.active_plan_id = contingency_plan.plan_id
        
        # Log event
        self.event_log.append({
            "timestamp": datetime.now(),
            "event": "contingency_activated",
            "plan_id": contingency_plan.plan_id,
            "reason": reason
        })
    
    def simulate_execution(
        self,
        plan: Plan,
        simulate_failure: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Simulate plan execution.
        
        Args:
            plan: Plan to execute
            simulate_failure: Simulate failure at this step
            
        Returns:
            Execution result
        """
        results = []
        
        for i, step in enumerate(plan.steps):
            # Simulate failure
            if simulate_failure and simulate_failure.lower() in step.lower():
                return {
                    "success": False,
                    "completed_steps": results,
                    "failed_at": step,
                    "reason": f"Simulated failure at: {step}"
                }
            
            # Successful step
            results.append({
                "step": step,
                "status": "completed",
                "timestamp": datetime.now()
            })
        
        plan.status = PlanStatus.SUCCEEDED
        plan.completed_at = datetime.now()
        
        return {
            "success": True,
            "completed_steps": results,
            "plan_id": plan.plan_id
        }
    
    def get_plan_summary(self) -> str:
        """Get summary of all plans."""
        lines = []
        
        # Primary plans
        primary = [p for p in self.plans.values() if p.status in [PlanStatus.PRIMARY, PlanStatus.ACTIVE]]
        if primary:
            lines.append("PRIMARY PLAN:")
            for plan in primary:
                active = " (ACTIVE)" if plan.plan_id == self.active_plan_id else ""
                lines.append(f"  {plan.plan_id}{active}: {plan.description}")
                for i, step in enumerate(plan.steps, 1):
                    lines.append(f"    {i}. {step}")
        
        # Contingencies
        contingencies = [p for p in self.plans.values() if p.status == PlanStatus.CONTINGENCY]
        if contingencies:
            lines.append("\nCONTINGENCY PLANS:")
            for plan in contingencies:
                lines.append(f"  {plan.plan_id}: {plan.description}")
                if plan.triggers:
                    lines.append(f"    Triggers: {', '.join(plan.triggers[:2])}...")
                for i, step in enumerate(plan.steps, 1):
                    lines.append(f"    {i}. {step}")
        
        return "\n".join(lines)
    
    def get_risk_summary(self) -> str:
        """Get summary of identified risks."""
        lines = ["IDENTIFIED RISKS (by priority):"]
        
        risks = sorted(self.risks.values(), key=lambda r: r.priority, reverse=True)
        
        for risk in risks:
            priority = risk.priority
            lines.append(
                f"\n  [{priority}] {risk.description}\n"
                f"      Likelihood: {risk.likelihood.value}, Impact: {risk.impact.value}\n"
                f"      Trigger: {risk.trigger_condition}"
            )
        
        return "\n".join(lines)


def demonstrate_contingency_planning():
    """Demonstrate contingency planning pattern."""
    
    print("=" * 80)
    print("CONTINGENCY PLANNING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Basic contingency planning
    print("\n" + "=" * 80)
    print("Example 1: Creating Primary and Contingency Plans")
    print("=" * 80)
    
    planner = ContingencyPlanner()
    
    # Create primary plan
    primary = planner.create_primary_plan(
        "Deploy new software version",
        [
            "Run automated tests",
            "Build production package",
            "Deploy to staging",
            "Run smoke tests",
            "Deploy to production"
        ]
    )
    
    print("\nPrimary Plan:")
    print(f"  {primary.description}")
    for i, step in enumerate(primary.steps, 1):
        print(f"  {i}. {step}")
    
    print("\nIdentifying risks...")
    risks = planner.identify_risks(primary)
    
    print(f"\nIdentified {len(risks)} risks:")
    for risk in risks:
        print(f"  • {risk.description}")
        print(f"    Likelihood: {risk.likelihood.value}, Impact: {risk.impact.value}")
        print(f"    Priority: {risk.priority}")
    
    # Create contingency for highest priority risk
    if risks:
        top_risk = max(risks, key=lambda r: r.priority)
        contingency = planner.create_contingency_plan(primary, top_risk, "B")
        
        print(f"\nContingency Plan B (for: {top_risk.description}):")
        for i, step in enumerate(contingency.steps, 1):
            print(f"  {i}. {step}")
    
    # Example 2: Tiered contingencies
    print("\n" + "=" * 80)
    print("Example 2: Multi-Tier Contingency Planning")
    print("=" * 80)
    
    planner2 = ContingencyPlanner()
    
    primary2 = planner2.create_primary_plan(
        "Complete important presentation",
        [
            "Prepare slides",
            "Rehearse presentation",
            "Set up projector",
            "Deliver presentation"
        ]
    )
    
    print("\nCreating tiered contingencies...")
    contingencies = planner2.create_tiered_contingencies(primary2, num_tiers=3)
    
    print(f"\nCreated {len(contingencies)} contingency tiers")
    print(planner2.get_plan_summary())
    
    # Example 3: Trigger detection and activation
    print("\n" + "=" * 80)
    print("Example 3: Contingency Activation")
    print("=" * 80)
    
    planner3 = ContingencyPlanner()
    
    primary3 = planner3.create_primary_plan(
        "Travel to meeting",
        [
            "Drive to meeting location",
            "Park car",
            "Enter building"
        ]
    )
    
    # Create simple contingency
    planner3.risk_counter += 1
    traffic_risk = Risk(
        risk_id=f"risk_{planner3.risk_counter}",
        description="Heavy traffic delays arrival",
        likelihood=RiskLevel.MEDIUM,
        impact=ImpactLevel.MODERATE,
        trigger_condition="traffic jam detected"
    )
    planner3.risks[traffic_risk.risk_id] = traffic_risk
    
    contingency3 = planner3.create_contingency_plan(primary3, traffic_risk, "B")
    
    print("\nPrimary Plan:")
    print(f"  {primary3.description}")
    
    print("\nContingency Plan B:")
    print(f"  {contingency3.description}")
    
    print("\nSimulating execution with problem...")
    observed_events = ["traffic jam detected", "delay expected"]
    
    triggered_plan = planner3.check_trigger_conditions(observed_events)
    
    if triggered_plan:
        print(f"\n⚠️  Trigger detected! Activating: {triggered_plan.plan_id}")
        planner3.activate_contingency(triggered_plan, "Traffic jam encountered")
        print(f"  Now executing: {triggered_plan.description}")
    
    # Example 4: Risk prioritization
    print("\n" + "=" * 80)
    print("Example 4: Risk Assessment and Prioritization")
    print("=" * 80)
    
    planner4 = ContingencyPlanner()
    
    primary4 = planner4.create_primary_plan(
        "Launch product campaign",
        [
            "Finalize marketing materials",
            "Schedule social media posts",
            "Send press releases",
            "Monitor campaign metrics"
        ]
    )
    
    print("\nAnalyzing risks...")
    risks4 = planner4.identify_risks(primary4)
    
    print(planner4.get_risk_summary())
    
    # Example 5: Execution with fallback
    print("\n" + "=" * 80)
    print("Example 5: Execution with Automatic Fallback")
    print("=" * 80)
    
    planner5 = ContingencyPlanner()
    
    primary5 = planner5.create_primary_plan(
        "Process data file",
        [
            "Download file from server",
            "Parse file contents",
            "Validate data",
            "Store in database"
        ]
    )
    
    # Create contingency
    planner5.risk_counter += 1
    download_risk = Risk(
        risk_id=f"risk_{planner5.risk_counter}",
        description="File download fails",
        likelihood=RiskLevel.MEDIUM,
        impact=ImpactLevel.MODERATE,
        trigger_condition="download fails"
    )
    
    contingency5 = Plan(
        plan_id="plan_backup",
        description="Plan B: Use cached file",
        steps=[
            "Retrieve file from cache",
            "Parse file contents",
            "Validate data",
            "Store in database"
        ],
        status=PlanStatus.CONTINGENCY,
        parent_plan_id=primary5.plan_id,
        triggers=["download fails"]
    )
    planner5.plans[contingency5.plan_id] = contingency5
    
    print("\nAttempting primary plan...")
    result = planner5.simulate_execution(primary5, simulate_failure="download")
    
    if not result["success"]:
        print(f"❌ Primary plan failed: {result['reason']}")
        
        # Check for contingency
        events = ["download fails"]
        triggered = planner5.check_trigger_conditions(events)
        
        if triggered:
            print(f"\n✓ Activating contingency: {triggered.plan_id}")
            planner5.activate_contingency(triggered, result["reason"])
            
            print("\nAttempting contingency plan...")
            result2 = planner5.simulate_execution(triggered)
            
            if result2["success"]:
                print(f"✅ Contingency plan succeeded!")
    
    # Example 6: Graceful degradation
    print("\n" + "=" * 80)
    print("Example 6: Graceful Degradation Strategy")
    print("=" * 80)
    
    planner6 = ContingencyPlanner()
    
    primary6 = planner6.create_primary_plan(
        "Generate comprehensive report",
        [
            "Collect data from all sources",
            "Perform advanced analytics",
            "Create interactive visualizations",
            "Generate PDF report"
        ]
    )
    
    # Degraded plan
    degraded = Plan(
        plan_id="plan_degraded",
        description="Plan B: Generate basic report (degraded)",
        steps=[
            "Collect data from primary source only",
            "Perform basic statistics",
            "Create simple charts",
            "Generate text report"
        ],
        status=PlanStatus.CONTINGENCY,
        parent_plan_id=primary6.plan_id,
        triggers=["data source unavailable", "time constraint"]
    )
    planner6.plans[degraded.plan_id] = degraded
    
    print("\nPrimary Plan (Full Functionality):")
    for i, step in enumerate(primary6.steps, 1):
        print(f"  {i}. {step}")
    
    print("\nDegraded Plan (Reduced Functionality):")
    for i, step in enumerate(degraded.steps, 1):
        print(f"  {i}. {step}")
    
    print("\nStrategy: Accept reduced functionality to meet deadline")
    
    # Example 7: Multiple contingency levels
    print("\n" + "=" * 80)
    print("Example 7: Cascade of Contingencies")
    print("=" * 80)
    
    planner7 = ContingencyPlanner()
    
    primary7 = planner7.create_primary_plan(
        "Connect to service",
        ["Connect to primary server"]
    )
    
    plan_b = Plan(
        plan_id="plan_b",
        description="Plan B: Connect to backup server",
        steps=["Connect to backup server"],
        status=PlanStatus.CONTINGENCY,
        parent_plan_id=primary7.plan_id,
        triggers=["primary server down"]
    )
    
    plan_c = Plan(
        plan_id="plan_c",
        description="Plan C: Use cached data",
        steps=["Load cached data from local storage"],
        status=PlanStatus.CONTINGENCY,
        parent_plan_id=plan_b.plan_id,
        triggers=["backup server down"]
    )
    
    plan_d = Plan(
        plan_id="plan_d",
        description="Plan D: Operate in offline mode",
        steps=["Enable offline mode", "Queue operations"],
        status=PlanStatus.CONTINGENCY,
        parent_plan_id=plan_c.plan_id,
        triggers=["no cached data"]
    )
    
    planner7.plans.update({
        plan_b.plan_id: plan_b,
        plan_c.plan_id: plan_c,
        plan_d.plan_id: plan_d
    })
    
    print("\nContingency Cascade:")
    print("  Primary: Connect to primary server")
    print("  Plan B: Connect to backup server (if primary fails)")
    print("  Plan C: Use cached data (if backup fails)")
    print("  Plan D: Operate offline (if no cache)")
    print("\nStrategy: Progressive degradation with multiple fallback levels")
    
    # Example 8: Contingency metrics
    print("\n" + "=" * 80)
    print("Example 8: Contingency Planning Metrics")
    print("=" * 80)
    
    planner8 = ContingencyPlanner()
    
    primary8 = planner8.create_primary_plan(
        "Complete project milestone",
        [
            "Finish feature implementation",
            "Write documentation",
            "Conduct code review",
            "Merge to main branch"
        ]
    )
    
    # Create multiple contingencies
    planner8.create_tiered_contingencies(primary8, num_tiers=3)
    
    print("\nContingency Planning Statistics:")
    print(f"  Total Plans: {len(planner8.plans)}")
    print(f"  Primary Plans: {sum(1 for p in planner8.plans.values() if p.status == PlanStatus.PRIMARY)}")
    print(f"  Contingency Plans: {sum(1 for p in planner8.plans.values() if p.status == PlanStatus.CONTINGENCY)}")
    print(f"  Identified Risks: {len(planner8.risks)}")
    print(f"  High-Priority Risks: {sum(1 for r in planner8.risks.values() if r.priority >= 6)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Contingency Planning Pattern")
    print("=" * 80)
    
    summary = """
    The Contingency Planning pattern demonstrated:
    
    1. BASIC CONTINGENCIES (Example 1):
       - Risk identification using LLM
       - Likelihood and impact assessment
       - Priority calculation
       - Alternative plan creation
       - Trigger condition specification
    
    2. TIERED PLANNING (Example 2):
       - Multiple contingency levels (B, C, D)
       - Risk-based prioritization
       - Comprehensive backup strategies
       - Structured alternatives
    
    3. TRIGGER ACTIVATION (Example 3):
       - Real-time monitoring
       - Condition matching
       - Automatic plan switching
       - Event-driven adaptation
       - Seamless transition
    
    4. RISK PRIORITIZATION (Example 4):
       - Priority scoring (likelihood × impact)
       - Risk ranking
       - Resource allocation guidance
       - Focus on critical risks
    
    5. AUTOMATIC FALLBACK (Example 5):
       - Failure detection
       - Contingency activation
       - Execution retry
       - Success tracking
       - Resilient operation
    
    6. GRACEFUL DEGRADATION (Example 6):
       - Reduced functionality plans
       - Partial goal achievement
       - Quality-time tradeoffs
       - Acceptable alternatives
    
    7. CONTINGENCY CASCADE (Example 7):
       - Multi-level fallbacks
       - Progressive degradation
       - Chain of alternatives
       - Comprehensive coverage
    
    8. METRICS TRACKING (Example 8):
       - Plan statistics
       - Risk counts
       - Priority distribution
       - Coverage assessment
    
    KEY BENEFITS:
    ✓ Reduces response time to failures
    ✓ Improves system reliability
    ✓ Enables graceful degradation
    ✓ Proactive preparation
    ✓ Builds resilience
    ✓ Reduces costly downtime
    ✓ Increases confidence
    ✓ Supports high-stakes operations
    
    USE CASES:
    • Mission-critical operations
    • Uncertain environments
    • Resource-constrained systems
    • High-stakes decision making
    • Autonomous systems
    • Real-time operations
    • Safety-critical applications
    • Production deployments
    
    CONTINGENCY TYPES:
    → Alternative Action: Different approach
    → Fallback Goal: Modified objective
    → Resource Substitution: Different resources
    → Path Alternative: Different sequence
    → Graceful Degradation: Reduced functionality
    → Abort and Recover: Clean exit
    
    BEST PRACTICES:
    1. Focus on high-impact, high-probability risks
    2. Use tiered contingency levels (B, C, D)
    3. Implement fast plan switching
    4. Monitor trigger conditions efficiently
    5. Test contingency plans regularly
    6. Document all contingencies clearly
    7. Balance planning cost vs. benefit
    8. Enable human override capability
    
    TRADE-OFFS:
    • Planning cost vs. preparedness benefit
    • Coverage thoroughness vs. efficiency
    • Multiple plans vs. maintenance burden
    • Proactive vs. reactive approach
    
    PRODUCTION CONSIDERATIONS:
    → Implement efficient trigger monitoring
    → Cache contingency plans for quick access
    → Test contingencies in simulation
    → Monitor plan activation frequency
    → Track contingency success rates
    → Update plans based on experience
    → Provide clear plan switching notifications
    → Enable rollback to primary plan
    → Log all plan transitions
    → Support human decision overrides
    
    This pattern enables agents to handle unexpected failures
    gracefully by preparing alternative plans before they're needed,
    monitoring for trigger conditions, and switching quickly when
    problems occur.
    """
    
    print(summary)


if __name__ == "__main__":
    demonstrate_contingency_planning()
