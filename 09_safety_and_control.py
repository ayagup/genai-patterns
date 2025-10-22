"""
Safety and Control Patterns
============================
Demonstrates patterns for ensuring safe and controlled agent behavior.
Includes: Guardrails, Circuit Breaker, Human-in-the-Loop
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime


# ============================================================================
# 1. GUARDRAILS PATTERN
# ============================================================================

class GuardrailType(Enum):
    INPUT_VALIDATION = "input_validation"
    OUTPUT_FILTERING = "output_filtering"
    CONTENT_SAFETY = "content_safety"
    CONSTRAINT_CHECK = "constraint_check"


@dataclass
class GuardrailViolation:
    """Represents a guardrail violation"""
    type: GuardrailType
    severity: str  # "low", "medium", "high"
    message: str
    blocked_content: Optional[str] = None


class ContentGuardrail:
    """Filters unsafe or inappropriate content"""
    
    def __init__(self):
        self.blocked_words = ['hack', 'exploit', 'illegal', 'dangerous']
        self.toxic_patterns = ['hate', 'violence', 'threat']
    
    def check_input(self, text: str) -> Optional[GuardrailViolation]:
        """Check if input violates content policies"""
        text_lower = text.lower()
        
        # Check for blocked words
        for word in self.blocked_words:
            if word in text_lower:
                return GuardrailViolation(
                    type=GuardrailType.CONTENT_SAFETY,
                    severity="high",
                    message=f"Input contains blocked content: '{word}'",
                    blocked_content=word
                )
        
        # Check for toxic patterns
        for pattern in self.toxic_patterns:
            if pattern in text_lower:
                return GuardrailViolation(
                    type=GuardrailType.CONTENT_SAFETY,
                    severity="medium",
                    message=f"Input contains potentially toxic content: '{pattern}'",
                    blocked_content=pattern
                )
        
        return None
    
    def filter_output(self, text: str) -> tuple[str, Optional[GuardrailViolation]]:
        """Filter output to ensure safety"""
        filtered = text
        violations = []
        
        # Remove sensitive patterns
        sensitive_info = ['password', 'ssn', 'credit card']
        for pattern in sensitive_info:
            if pattern in text.lower():
                filtered = filtered.replace(pattern, "[REDACTED]")
                violations.append(GuardrailViolation(
                    type=GuardrailType.OUTPUT_FILTERING,
                    severity="high",
                    message=f"Removed sensitive information: {pattern}"
                ))
        
        return filtered, violations[0] if violations else None


class ConstraintGuardrail:
    """Enforces operational constraints"""
    
    def __init__(self, max_length: int = 1000, max_tokens: int = 500):
        self.max_length = max_length
        self.max_tokens = max_tokens
    
    def check_constraints(self, text: str) -> Optional[GuardrailViolation]:
        """Check if text violates constraints"""
        if len(text) > self.max_length:
            return GuardrailViolation(
                type=GuardrailType.CONSTRAINT_CHECK,
                severity="medium",
                message=f"Text exceeds maximum length: {len(text)} > {self.max_length}"
            )
        
        token_count = len(text.split())
        if token_count > self.max_tokens:
            return GuardrailViolation(
                type=GuardrailType.CONSTRAINT_CHECK,
                severity="medium",
                message=f"Text exceeds maximum tokens: {token_count} > {self.max_tokens}"
            )
        
        return None


class GuardrailsAgent:
    """Agent with multiple guardrails"""
    
    def __init__(self):
        self.content_guardrail = ContentGuardrail()
        self.constraint_guardrail = ConstraintGuardrail()
        self.violations_log: List[GuardrailViolation] = []
    
    def process_safely(self, user_input: str) -> Dict:
        """Process input with guardrails"""
        print(f"\n{'='*70}")
        print(f"Processing: {user_input[:60]}...")
        print(f"{'='*70}\n")
        
        # Input validation
        print("🛡️  Checking input guardrails...")
        
        content_violation = self.content_guardrail.check_input(user_input)
        if content_violation:
            self.violations_log.append(content_violation)
            print(f"❌ Blocked: {content_violation.message}")
            return {
                "success": False,
                "error": "Input blocked by guardrails",
                "violation": content_violation
            }
        
        constraint_violation = self.constraint_guardrail.check_constraints(user_input)
        if constraint_violation:
            self.violations_log.append(constraint_violation)
            print(f"⚠️  Warning: {constraint_violation.message}")
        
        print("✅ Input passed guardrails")
        
        # Generate response
        print("\n💭 Generating response...")
        response = self._generate_response(user_input)
        
        # Output filtering
        print("🛡️  Filtering output...")
        filtered_response, output_violation = self.content_guardrail.filter_output(response)
        
        if output_violation:
            self.violations_log.append(output_violation)
            print(f"⚠️  Filtered: {output_violation.message}")
        else:
            print("✅ Output passed guardrails")
        
        print(f"\n📤 Safe Response: {filtered_response}")
        
        return {
            "success": True,
            "response": filtered_response,
            "violations": [output_violation] if output_violation else []
        }
    
    def _generate_response(self, input_text: str) -> str:
        """Generate response (simulated)"""
        return f"I can help with that. Your request about '{input_text[:30]}...' has been processed."


# ============================================================================
# 2. CIRCUIT BREAKER PATTERN
# ============================================================================

class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        # Check if circuit is OPEN
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                print("🔄 Circuit HALF-OPEN: Attempting recovery...")
                self.state = CircuitState.HALF_OPEN
            else:
                print("⛔ Circuit OPEN: Request blocked")
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            # Attempt the function call
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                print("✅ Circuit CLOSED: Service recovered")
                self.state = CircuitState.CLOSED
                self.success_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            print(f"🚨 Circuit OPEN: Too many failures ({self.failure_count})")
            self.state = CircuitState.OPEN
            self.success_count = 0
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def get_state(self) -> Dict:
        """Get current circuit breaker state"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count
        }


class ResilientAgent:
    """Agent with circuit breaker for external service calls"""
    
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5,  # Short timeout for demo
            success_threshold=2
        )
        self.call_count = 0
    
    def call_external_service(self, data: str) -> str:
        """Call external service with circuit breaker protection"""
        self.call_count += 1
        
        print(f"\n📞 Call #{self.call_count}: Attempting external service call...")
        print(f"   Circuit state: {self.circuit_breaker.state.value}")
        
        try:
            result = self.circuit_breaker.call(
                self._simulate_external_call,
                data
            )
            print(f"   ✅ Success: {result}")
            return result
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            return f"Error: {str(e)}"
    
    def _simulate_external_call(self, data: str) -> str:
        """Simulate external service call that might fail"""
        # Simulate intermittent failures
        if self.call_count in [2, 3, 4]:  # These calls will fail
            raise Exception("External service error")
        
        return f"Processed: {data}"


# ============================================================================
# 3. HUMAN-IN-THE-LOOP (HITL) PATTERN
# ============================================================================

class ApprovalRequired(Exception):
    """Exception raised when human approval is needed"""
    pass


class HITLDecision:
    """Represents a decision requiring human input"""
    
    def __init__(self, context: str, options: List[str], recommendation: Optional[str] = None):
        self.context = context
        self.options = options
        self.recommendation = recommendation
        self.human_choice: Optional[str] = None
        self.timestamp = datetime.now()


class HumanInTheLoopAgent:
    """Agent that requests human approval for critical decisions"""
    
    def __init__(self, auto_approve_threshold: float = 0.9):
        self.auto_approve_threshold = auto_approve_threshold
        self.pending_decisions: List[HITLDecision] = []
        self.decision_history: List[HITLDecision] = []
    
    def make_decision(self, task: str, confidence: float) -> Dict:
        """Make a decision, requesting human input if needed"""
        print(f"\n{'='*70}")
        print(f"Task: {task}")
        print(f"Agent Confidence: {confidence:.2f}")
        print(f"{'='*70}\n")
        
        # Determine if human approval is needed
        if confidence >= self.auto_approve_threshold:
            print(f"✅ Auto-approved (confidence {confidence:.2f} >= {self.auto_approve_threshold})")
            decision = "proceed"
            approved_by = "agent"
        else:
            print(f"⏸️  Human approval required (confidence {confidence:.2f} < {self.auto_approve_threshold})")
            decision = self._request_human_approval(task, confidence)
            approved_by = "human"
        
        return {
            "task": task,
            "decision": decision,
            "confidence": confidence,
            "approved_by": approved_by
        }
    
    def _request_human_approval(self, task: str, confidence: float) -> str:
        """Request human approval for decision"""
        options = ["proceed", "modify", "reject"]
        recommendation = "proceed" if confidence > 0.7 else "review carefully"
        
        decision = HITLDecision(
            context=f"Task: {task} (confidence: {confidence:.2f})",
            options=options,
            recommendation=recommendation
        )
        
        print(f"\n{'─'*70}")
        print("HUMAN APPROVAL REQUESTED")
        print(f"{'─'*70}")
        print(f"Context: {decision.context}")
        print(f"Options: {', '.join(decision.options)}")
        print(f"Recommendation: {decision.recommendation}")
        print(f"{'─'*70}")
        
        # Simulate human decision (in real scenario, would wait for human input)
        human_choice = self._simulate_human_choice(confidence)
        decision.human_choice = human_choice
        
        print(f"👤 Human Decision: {human_choice}")
        
        self.decision_history.append(decision)
        return human_choice
    
    def _simulate_human_choice(self, confidence: float) -> str:
        """Simulate human decision making"""
        # Higher confidence → more likely to approve
        if confidence > 0.75:
            return "proceed"
        elif confidence > 0.5:
            return "modify"
        else:
            return "reject"
    
    def get_approval_stats(self) -> Dict:
        """Get statistics on approval decisions"""
        total = len(self.decision_history)
        if total == 0:
            return {"total_decisions": 0}
        
        choices = [d.human_choice for d in self.decision_history]
        return {
            "total_decisions": total,
            "proceeded": choices.count("proceed"),
            "modified": choices.count("modify"),
            "rejected": choices.count("reject")
        }


def main():
    """Demonstrate safety and control patterns"""
    
    # ========================================================================
    # EXAMPLE 1: GUARDRAILS PATTERN
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 1: GUARDRAILS PATTERN")
    print("="*70)
    
    agent = GuardrailsAgent()
    
    # Safe input
    result1 = agent.process_safely("Tell me about Python programming")
    
    # Input with blocked content
    print("\n")
    result2 = agent.process_safely("How to hack into a system?")
    
    # Input with sensitive output
    print("\n")
    result3 = agent.process_safely("What is my password information?")
    
    # ========================================================================
    # EXAMPLE 2: CIRCUIT BREAKER PATTERN
    # ========================================================================
    print("\n\n" + "="*70)
    print("EXAMPLE 2: CIRCUIT BREAKER PATTERN")
    print("="*70)
    
    resilient_agent = ResilientAgent()
    
    # Make multiple calls - some will fail
    for i in range(8):
        resilient_agent.call_external_service(f"data_{i}")
        time.sleep(0.1)  # Small delay between calls
    
    print(f"\n{'─'*70}")
    print(f"Final Circuit State: {resilient_agent.circuit_breaker.get_state()}")
    
    # ========================================================================
    # EXAMPLE 3: HUMAN-IN-THE-LOOP PATTERN
    # ========================================================================
    print("\n\n" + "="*70)
    print("EXAMPLE 3: HUMAN-IN-THE-LOOP (HITL) PATTERN")
    print("="*70)
    
    hitl_agent = HumanInTheLoopAgent(auto_approve_threshold=0.9)
    
    # Various tasks with different confidence levels
    tasks = [
        ("Send automated email response", 0.95),
        ("Delete user account", 0.75),
        ("Update system configuration", 0.60),
        ("Generate routine report", 0.92),
        ("Transfer funds", 0.50)
    ]
    
    for task, confidence in tasks:
        hitl_agent.make_decision(task, confidence)
    
    # Show approval statistics
    print(f"\n{'='*70}")
    print("APPROVAL STATISTICS")
    print(f"{'='*70}")
    stats = hitl_agent.get_approval_stats()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Summary
    print(f"\n\n{'='*70}")
    print("SAFETY PATTERNS SUMMARY")
    print(f"{'='*70}")
    
    print("\n1. GUARDRAILS:")
    print("   ✓ Input validation prevents harmful requests")
    print("   ✓ Output filtering protects sensitive information")
    print("   ✓ Constraint checking enforces operational limits")
    
    print("\n2. CIRCUIT BREAKER:")
    print("   ✓ Prevents cascading failures")
    print("   ✓ Automatically recovers when service improves")
    print("   ✓ Provides graceful degradation")
    
    print("\n3. HUMAN-IN-THE-LOOP:")
    print("   ✓ Escalates uncertain decisions to humans")
    print("   ✓ Maintains safety in high-stakes scenarios")
    print("   ✓ Balances automation with human oversight")


if __name__ == "__main__":
    main()
