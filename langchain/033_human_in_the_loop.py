"""
Pattern 033: Human-in-the-Loop (HITL)

Description:
    Human-in-the-Loop integrates human oversight and intervention at critical
    decision points in agent workflows. This pattern enables humans to approve,
    correct, or guide agent actions, ensuring safety, quality, and alignment
    with human values in high-stakes or ambiguous situations.

Components:
    - Intervention Points: Designated moments for human input
    - Approval Gates: Require human approval before proceeding
    - Correction Mechanism: Allow humans to fix agent errors
    - Guidance System: Humans provide direction when agent is uncertain
    - Escalation Logic: Automatic escalation on confidence thresholds
    - Feedback Integration: Learn from human corrections

Use Cases:
    - High-stakes decision making (medical, financial, legal)
    - Content moderation and review
    - Quality assurance for generated content
    - Training data collection and refinement
    - Compliance and regulatory requirements
    - Handling edge cases and ambiguity

LangChain Implementation:
    Uses callback handlers and approval mechanisms to integrate human
    oversight into agent workflows, with configurable intervention points
    and escalation criteria.
"""

import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


class InterventionType(Enum):
    """Types of human intervention."""
    APPROVAL = "approval"  # Human must approve action
    CORRECTION = "correction"  # Human can correct output
    GUIDANCE = "guidance"  # Human provides direction
    REVIEW = "review"  # Human reviews but doesn't block
    ESCALATION = "escalation"  # Automatic escalation to human


class InterventionStatus(Enum):
    """Status of intervention request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CORRECTED = "corrected"
    SKIPPED = "skipped"


@dataclass
class InterventionRequest:
    """Request for human intervention."""
    id: str
    type: InterventionType
    context: str
    proposed_action: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Response
    status: InterventionStatus = InterventionStatus.PENDING
    human_response: Optional[str] = None
    correction: Optional[str] = None
    response_time: Optional[datetime] = None


@dataclass
class EscalationCriteria:
    """Criteria for automatic escalation to human."""
    min_confidence: float = 0.7  # Below this, escalate
    sensitive_keywords: List[str] = field(default_factory=list)
    high_stakes_actions: List[str] = field(default_factory=list)
    require_approval_for: List[str] = field(default_factory=list)


class HumanInterface:
    """
    Interface for human interaction (simulated or real).
    
    In production, this would connect to a UI, API, or messaging system.
    """
    
    def __init__(self, auto_approve: bool = False):
        self.auto_approve = auto_approve
        self.pending_requests: List[InterventionRequest] = []
    
    def request_intervention(
        self,
        request: InterventionRequest
    ) -> InterventionRequest:
        """
        Request human intervention.
        
        In simulation mode, auto-responds. In production, would wait for human.
        """
        self.pending_requests.append(request)
        
        if self.auto_approve:
            # Simulate human response
            if request.type == InterventionType.APPROVAL:
                request.status = InterventionStatus.APPROVED
                request.human_response = "Approved by human"
            elif request.type == InterventionType.CORRECTION:
                # Simulate a correction
                request.status = InterventionStatus.CORRECTED
                request.correction = f"Corrected version: {request.proposed_action} [HUMAN EDITED]"
            elif request.type == InterventionType.GUIDANCE:
                request.status = InterventionStatus.APPROVED
                request.human_response = "Proceed with suggested approach"
            else:
                request.status = InterventionStatus.APPROVED
            
            request.response_time = datetime.now()
        
        return request
    
    def get_pending_count(self) -> int:
        """Get number of pending intervention requests."""
        return sum(1 for r in self.pending_requests if r.status == InterventionStatus.PENDING)
    
    def get_all_requests(self) -> List[InterventionRequest]:
        """Get all intervention requests."""
        return self.pending_requests.copy()


class HITLAgent:
    """
    Agent with Human-in-the-Loop integration.
    
    Features:
    - Configurable intervention points
    - Automatic escalation based on criteria
    - Approval gates for sensitive actions
    - Learning from human feedback
    """
    
    def __init__(
        self,
        human_interface: HumanInterface,
        escalation_criteria: Optional[EscalationCriteria] = None,
        temperature: float = 0.7
    ):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
        self.human_interface = human_interface
        self.escalation_criteria = escalation_criteria or EscalationCriteria()
        
        self._next_request_id = 1
        self.intervention_history: List[InterventionRequest] = []
        
        # Prompt for generating response
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant that works with human oversight.

Task: {task}

Generate a response and provide a confidence score (0.0-1.0) for your answer.

Respond in this format:
CONFIDENCE: [score]
RESPONSE: [your response]"""),
            ("user", "{task}")
        ])
        
        # Prompt for confidence assessment
        self.confidence_prompt = ChatPromptTemplate.from_messages([
            ("system", """Assess your confidence in this response on a scale of 0.0 to 1.0.

Task: {task}
Response: {response}

Consider:
- Certainty of facts
- Complexity of task
- Potential for error
- Stakes involved

Respond with just a number between 0.0 and 1.0."""),
            ("user", "Assess confidence")
        ])
    
    def process_with_oversight(
        self,
        task: str,
        require_approval: bool = False
    ) -> Dict[str, Any]:
        """
        Process task with human oversight.
        
        Args:
            task: Task to process
            require_approval: Force human approval regardless of confidence
        
        Returns:
            Dictionary with response and intervention details
        """
        # Generate initial response
        chain = self.response_prompt | self.llm | StrOutputParser()
        result = chain.invoke({"task": task})
        
        # Parse response
        response, confidence = self._parse_response(result)
        
        # Check if escalation is needed
        should_escalate = (
            require_approval or
            self._should_escalate(task, response, confidence)
        )
        
        if should_escalate:
            # Request human intervention
            intervention_type = (
                InterventionType.APPROVAL if require_approval or confidence < 0.5
                else InterventionType.REVIEW
            )
            
            request = InterventionRequest(
                id=f"req_{self._next_request_id:04d}",
                type=intervention_type,
                context=task,
                proposed_action=response,
                confidence=confidence
            )
            self._next_request_id += 1
            
            # Get human input
            request = self.human_interface.request_intervention(request)
            self.intervention_history.append(request)
            
            # Process human response
            if request.status == InterventionStatus.APPROVED:
                final_response = response
                used_hitl = True
                hitl_result = "approved"
            elif request.status == InterventionStatus.CORRECTED:
                final_response = request.correction or response
                used_hitl = True
                hitl_result = "corrected"
            elif request.status == InterventionStatus.REJECTED:
                final_response = "Action rejected by human oversight"
                used_hitl = True
                hitl_result = "rejected"
            else:
                final_response = response
                used_hitl = True
                hitl_result = "pending"
        else:
            # No human intervention needed
            final_response = response
            used_hitl = False
            hitl_result = "not_required"
        
        return {
            "response": final_response,
            "confidence": confidence,
            "used_hitl": used_hitl,
            "hitl_result": hitl_result,
            "original_response": response if used_hitl else None
        }
    
    def _parse_response(self, result: str) -> tuple[str, float]:
        """Parse response and confidence from LLM output."""
        lines = result.strip().split('\n')
        confidence = 0.5  # Default
        response = result
        
        for line in lines:
            if line.startswith("CONFIDENCE:"):
                try:
                    conf_str = line.replace("CONFIDENCE:", "").strip()
                    confidence = float(conf_str)
                except ValueError:
                    pass
            elif line.startswith("RESPONSE:"):
                response = line.replace("RESPONSE:", "").strip()
        
        # If format wasn't followed, estimate confidence
        if confidence == 0.5 and "CONFIDENCE:" not in result:
            confidence = self._estimate_confidence(result)
        
        return response, confidence
    
    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence from response content."""
        # Simple heuristic based on uncertainty phrases
        uncertainty_phrases = [
            "i think", "maybe", "possibly", "uncertain", "not sure",
            "might", "could be", "perhaps", "probably"
        ]
        
        response_lower = response.lower()
        uncertainty_count = sum(1 for phrase in uncertainty_phrases if phrase in response_lower)
        
        # More uncertainty phrases = lower confidence
        confidence = max(0.3, 1.0 - (uncertainty_count * 0.15))
        
        return confidence
    
    def _should_escalate(self, task: str, response: str, confidence: float) -> bool:
        """Determine if task should be escalated to human."""
        # Low confidence
        if confidence < self.escalation_criteria.min_confidence:
            return True
        
        # Sensitive keywords in task or response
        text = f"{task} {response}".lower()
        for keyword in self.escalation_criteria.sensitive_keywords:
            if keyword.lower() in text:
                return True
        
        # High-stakes actions
        for action in self.escalation_criteria.high_stakes_actions:
            if action.lower() in text:
                return True
        
        return False
    
    def request_correction(
        self,
        task: str,
        initial_response: str,
        issue_description: str
    ) -> Dict[str, Any]:
        """
        Request human correction for a specific issue.
        """
        request = InterventionRequest(
            id=f"req_{self._next_request_id:04d}",
            type=InterventionType.CORRECTION,
            context=f"Task: {task}\nIssue: {issue_description}",
            proposed_action=initial_response,
            confidence=0.0,  # Explicitly requesting correction
            metadata={"issue": issue_description}
        )
        self._next_request_id += 1
        
        # Get human correction
        request = self.human_interface.request_intervention(request)
        self.intervention_history.append(request)
        
        return {
            "original": initial_response,
            "corrected": request.correction or initial_response,
            "status": request.status.value
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about human interventions."""
        if not self.intervention_history:
            return {
                "total_interventions": 0,
                "by_type": {},
                "by_status": {},
                "avg_confidence": 0.0
            }
        
        by_type = {}
        by_status = {}
        
        for request in self.intervention_history:
            # Count by type
            type_key = request.type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
            
            # Count by status
            status_key = request.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1
        
        avg_confidence = sum(r.confidence for r in self.intervention_history) / len(self.intervention_history)
        
        return {
            "total_interventions": len(self.intervention_history),
            "by_type": by_type,
            "by_status": by_status,
            "avg_confidence": avg_confidence
        }


def demonstrate_human_in_the_loop():
    """
    Demonstrates Human-in-the-Loop with various intervention scenarios.
    """
    print("=" * 80)
    print("HUMAN-IN-THE-LOOP DEMONSTRATION")
    print("=" * 80)
    
    # Create human interface (simulated)
    human_interface = HumanInterface(auto_approve=True)
    
    # Configure escalation criteria
    escalation_criteria = EscalationCriteria(
        min_confidence=0.7,
        sensitive_keywords=["medical", "financial", "legal", "delete", "confidential"],
        high_stakes_actions=["approve payment", "delete data", "grant access"]
    )
    
    # Create HITL agent
    agent = HITLAgent(
        human_interface=human_interface,
        escalation_criteria=escalation_criteria
    )
    
    # Test 1: High confidence task (no intervention)
    print("\n" + "=" * 80)
    print("Test 1: High Confidence Task (No Intervention Expected)")
    print("=" * 80)
    
    task1 = "What is the capital of France?"
    print(f"\nTask: {task1}")
    result1 = agent.process_with_oversight(task1)
    print(f"Response: {result1['response']}")
    print(f"Confidence: {result1['confidence']:.2f}")
    print(f"HITL Used: {result1['used_hitl']}")
    print(f"HITL Result: {result1['hitl_result']}")
    
    # Test 2: Sensitive keyword (automatic escalation)
    print("\n" + "=" * 80)
    print("Test 2: Sensitive Content (Automatic Escalation)")
    print("=" * 80)
    
    task2 = "Should I approve this medical procedure for the patient?"
    print(f"\nTask: {task2}")
    result2 = agent.process_with_oversight(task2)
    print(f"Response: {result2['response']}")
    print(f"Confidence: {result2['confidence']:.2f}")
    print(f"HITL Used: {result2['used_hitl']}")
    print(f"HITL Result: {result2['hitl_result']}")
    if result2['original_response']:
        print(f"Original Response: {result2['original_response'][:100]}...")
    
    # Test 3: Forced approval
    print("\n" + "=" * 80)
    print("Test 3: Forced Human Approval")
    print("=" * 80)
    
    task3 = "Delete all user data from the database"
    print(f"\nTask: {task3}")
    result3 = agent.process_with_oversight(task3, require_approval=True)
    print(f"Response: {result3['response']}")
    print(f"Confidence: {result3['confidence']:.2f}")
    print(f"HITL Used: {result3['used_hitl']}")
    print(f"HITL Result: {result3['hitl_result']}")
    
    # Test 4: Request correction
    print("\n" + "=" * 80)
    print("Test 4: Human Correction Request")
    print("=" * 80)
    
    task4 = "Calculate the total revenue for Q4 2024"
    initial_response = "The total revenue was $1.5M"
    issue = "The calculation seems incorrect based on our records"
    
    print(f"\nTask: {task4}")
    print(f"Initial Response: {initial_response}")
    print(f"Issue: {issue}")
    
    correction_result = agent.request_correction(task4, initial_response, issue)
    print(f"Corrected Response: {correction_result['corrected']}")
    print(f"Status: {correction_result['status']}")
    
    # Test 5: Low confidence (escalation)
    print("\n" + "=" * 80)
    print("Test 5: Low Confidence Response")
    print("=" * 80)
    
    task5 = "I'm not sure about this complex legal interpretation..."
    print(f"\nTask: {task5}")
    result5 = agent.process_with_oversight(task5)
    print(f"Response: {result5['response']}")
    print(f"Confidence: {result5['confidence']:.2f}")
    print(f"HITL Used: {result5['used_hitl']}")
    print(f"HITL Result: {result5['hitl_result']}")
    
    # Show statistics
    print("\n" + "=" * 80)
    print("Intervention Statistics")
    print("=" * 80)
    
    stats = agent.get_statistics()
    print(f"\nTotal Interventions: {stats['total_interventions']}")
    print(f"By Type: {stats['by_type']}")
    print(f"By Status: {stats['by_status']}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    
    print(f"\nPending Requests: {human_interface.get_pending_count()}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Human-in-the-Loop (HITL) provides:
✓ Human oversight at critical decision points
✓ Automatic escalation based on confidence
✓ Approval gates for sensitive actions
✓ Correction mechanisms for error fixing
✓ Guidance integration for ambiguous cases
✓ Audit trail of human interventions

This pattern excels at:
- High-stakes decision making (medical, financial, legal)
- Quality assurance and content moderation
- Handling edge cases and ambiguity
- Regulatory compliance requirements
- Training data collection
- Building trust and accountability

Intervention types:
1. APPROVAL: Human must approve before proceeding
2. CORRECTION: Human can fix errors or improve output
3. GUIDANCE: Human provides direction when uncertain
4. REVIEW: Human reviews but doesn't block execution
5. ESCALATION: Automatic routing to human based on criteria

Escalation triggers:
- Low confidence scores (< 0.7 by default)
- Sensitive keywords (medical, financial, legal, etc.)
- High-stakes actions (delete data, approve payments, etc.)
- Explicit requirement for human approval

Benefits:
- Safety: Prevents high-risk errors
- Quality: Human expertise improves outputs
- Trust: Builds confidence through oversight
- Learning: Agent learns from corrections
- Compliance: Meets regulatory requirements
- Accountability: Clear audit trail

Use HITL when you need:
- Safety in high-stakes scenarios
- Quality assurance for critical outputs
- Regulatory compliance
- Human expertise for edge cases
- Trust and transparency
- Gradual automation with human backup
""")


if __name__ == "__main__":
    demonstrate_human_in_the_loop()
