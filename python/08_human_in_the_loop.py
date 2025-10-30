"""
Human-in-the-Loop (HITL) Pattern
Human provides guidance at critical decision points
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
@dataclass
class Decision:
    id: int
    description: str
    proposed_action: str
    risk_level: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    human_feedback: Optional[str] = None
    final_action: Optional[str] = None
class HumanInTheLoopAgent:
    def __init__(self, auto_approve_low_risk: bool = True):
        self.auto_approve_low_risk = auto_approve_low_risk
        self.decisions: List[Decision] = []
        self.decision_counter = 0
    def assess_risk(self, action: str) -> str:
        """Assess risk level of an action"""
        # Simple heuristic for demonstration
        high_risk_keywords = ["delete", "remove", "irreversible", "critical"]
        medium_risk_keywords = ["update", "modify", "change"]
        action_lower = action.lower()
        if any(keyword in action_lower for keyword in high_risk_keywords):
            return "HIGH"
        elif any(keyword in action_lower for keyword in medium_risk_keywords):
            return "MEDIUM"
        else:
            return "LOW"
    def propose_action(self, task: str, action: str) -> Decision:
        """Propose an action for a task"""
        risk_level = self.assess_risk(action)
        decision = Decision(
            id=self.decision_counter,
            description=task,
            proposed_action=action,
            risk_level=risk_level
        )
        self.decision_counter += 1
        self.decisions.append(decision)
        return decision
    def request_human_approval(self, decision: Decision) -> bool:
        """Request human approval (simulated)"""
        print(f"\n{'='*60}")
        print(f"HUMAN APPROVAL REQUIRED")
        print(f"{'='*60}")
        print(f"Decision ID: {decision.id}")
        print(f"Task: {decision.description}")
        print(f"Proposed Action: {decision.proposed_action}")
        print(f"Risk Level: {decision.risk_level}")
        print(f"\nOptions:")
        print(f"  1. Approve")
        print(f"  2. Reject")
        print(f"  3. Modify")
        # Simulate human input (in real implementation, get actual user input)
        # For demo, auto-approve low risk, request approval for others
        if decision.risk_level == "LOW" and self.auto_approve_low_risk:
            print(f"\n[AUTO-APPROVED: Low risk action]")
            decision.status = ApprovalStatus.APPROVED
            decision.final_action = decision.proposed_action
            return True
        # Simulate human decision
        import random
        choice = random.choice([1, 1, 2, 3])  # Bias toward approval
        if choice == 1:
            print(f"\n[HUMAN: Approved]")
            decision.status = ApprovalStatus.APPROVED
            decision.final_action = decision.proposed_action
            return True
        elif choice == 2:
            print(f"\n[HUMAN: Rejected - Too risky]")
            decision.status = ApprovalStatus.REJECTED
            decision.human_feedback = "Action rejected due to high risk"
            return False
        else:
            print(f"\n[HUMAN: Modified - Use safer alternative]")
            decision.status = ApprovalStatus.MODIFIED
            decision.final_action = f"SAFE_MODE: {decision.proposed_action}"
            decision.human_feedback = "Modified to use safer approach"
            return True
    def execute_with_approval(self, task: str, action: str) -> Dict[str, Any]:
        """Execute action with human approval"""
        print(f"\n--- New Task ---")
        print(f"Task: {task}")
        print(f"Proposed Action: {action}")
        # Create decision
        decision = self.propose_action(task, action)
        # Get approval
        approved = self.request_human_approval(decision)
        # Execute if approved
        if approved:
            result = self.execute_action(decision.final_action)
            print(f"\n✓ Executed: {decision.final_action}")
            print(f"Result: {result}")
            return {"success": True, "result": result, "decision": decision}
        else:
            print(f"\n✗ Action not executed (rejected by human)")
            return {"success": False, "result": None, "decision": decision}
    def execute_action(self, action: str) -> str:
        """Execute the approved action"""
        # Simulate action execution
        return f"Action '{action}' completed successfully"
    def get_approval_summary(self) -> Dict[str, Any]:
        """Get summary of all decisions"""
        total = len(self.decisions)
        approved = sum(1 for d in self.decisions if d.status == ApprovalStatus.APPROVED)
        rejected = sum(1 for d in self.decisions if d.status == ApprovalStatus.REJECTED)
        modified = sum(1 for d in self.decisions if d.status == ApprovalStatus.MODIFIED)
        return {
            "total_decisions": total,
            "approved": approved,
            "rejected": rejected,
            "modified": modified,
            "approval_rate": approved / total if total > 0 else 0
        }
# Usage
if __name__ == "__main__":
    agent = HumanInTheLoopAgent(auto_approve_low_risk=True)
    # Example tasks with different risk levels
    tasks = [
        ("Analyze user data", "Read and analyze user statistics"),
        ("Update user preferences", "Modify user preference settings"),
        ("Clean up database", "Delete old temporary records"),
        ("Send notification", "Send email to users"),
        ("Critical system update", "Update critical system configuration"),
    ]
    print("="*60)
    print("HUMAN-IN-THE-LOOP AGENT EXECUTION")
    print("="*60)
    for task, action in tasks:
        agent.execute_with_approval(task, action)
    # Summary
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    summary = agent.get_approval_summary()
    print(f"\nTotal Decisions: {summary['total_decisions']}")
    print(f"Approved: {summary['approved']}")
    print(f"Rejected: {summary['rejected']}")
    print(f"Modified: {summary['modified']}")
    print(f"Approval Rate: {summary['approval_rate']:.1%}")
    print(f"\n--- Decision Log ---")
    for decision in agent.decisions:
        print(f"\n{decision.id}. {decision.description}")
        print(f"   Risk: {decision.risk_level} | Status: {decision.status.value}")
        if decision.human_feedback:
            print(f"   Feedback: {decision.human_feedback}")
