"""
Guardrails Pattern
Input/output validation and filtering for safety
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import re
class GuardrailType(Enum):
    INPUT_VALIDATION = "input_validation"
    OUTPUT_FILTERING = "output_filtering"
    CONTENT_SAFETY = "content_safety"
    RATE_LIMITING = "rate_limiting"
    DATA_PRIVACY = "data_privacy"
@dataclass
class GuardrailViolation:
    type: GuardrailType
    severity: str  # "low", "medium", "high"
    message: str
    details: Dict[str, Any]
class Guardrail:
    """Base class for guardrails"""
    def __init__(self, name: str, guardrail_type: GuardrailType):
        self.name = name
        self.guardrail_type = guardrail_type
    def check(self, content: str, context: Dict[str, Any] = None) -> Optional[GuardrailViolation]:
        """Check if content violates guardrail"""
        raise NotImplementedError
class ContentSafetyGuardrail(Guardrail):
    """Check for unsafe content"""
    def __init__(self):
        super().__init__("Content Safety", GuardrailType.CONTENT_SAFETY)
        self.toxic_keywords = [
            "hate", "violence", "illegal", "harmful"
        ]
        self.sensitive_patterns = [
            r'\b(?:password|secret|api[_-]?key)\b'
        ]
    def check(self, content: str, context: Dict[str, Any] = None) -> Optional[GuardrailViolation]:
        content_lower = content.lower()
        # Check for toxic keywords
        for keyword in self.toxic_keywords:
            if keyword in content_lower:
                return GuardrailViolation(
                    type=self.guardrail_type,
                    severity="high",
                    message=f"Potentially toxic content detected: '{keyword}'",
                    details={"keyword": keyword}
                )
        # Check for sensitive patterns
        for pattern in self.sensitive_patterns:
            if re.search(pattern, content_lower):
                return GuardrailViolation(
                    type=self.guardrail_type,
                    severity="high",
                    message="Sensitive information detected",
                    details={"pattern": pattern}
                )
        return None
class PIIGuardrail(Guardrail):
    """Check for Personally Identifiable Information"""
    def __init__(self):
        super().__init__("PII Detection", GuardrailType.DATA_PRIVACY)
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        }
    def check(self, content: str, context: Dict[str, Any] = None) -> Optional[GuardrailViolation]:
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, content)
            if matches:
                return GuardrailViolation(
                    type=self.guardrail_type,
                    severity="high",
                    message=f"PII detected: {pii_type}",
                    details={"pii_type": pii_type, "count": len(matches)}
                )
        return None
class InputValidationGuardrail(Guardrail):
    """Validate input format and constraints"""
    def __init__(self, max_length: int = 1000, min_length: int = 1):
        super().__init__("Input Validation", GuardrailType.INPUT_VALIDATION)
        self.max_length = max_length
        self.min_length = min_length
    def check(self, content: str, context: Dict[str, Any] = None) -> Optional[GuardrailViolation]:
        # Check length
        if len(content) > self.max_length:
            return GuardrailViolation(
                type=self.guardrail_type,
                severity="medium",
                message=f"Input exceeds maximum length of {self.max_length}",
                details={"length": len(content), "max": self.max_length}
            )
        if len(content) < self.min_length:
            return GuardrailViolation(
                type=self.guardrail_type,
                severity="low",
                message=f"Input below minimum length of {self.min_length}",
                details={"length": len(content), "min": self.min_length}
            )
        # Check for injection attempts
        injection_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'(\bUNION\b|\bSELECT\b|\bDROP\b|\bINSERT\b)',  # SQL injection
        ]
        for pattern in injection_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return GuardrailViolation(
                    type=self.guardrail_type,
                    severity="high",
                    message="Potential injection attempt detected",
                    details={"pattern": pattern}
                )
        return None
class OutputFilteringGuardrail(Guardrail):
    """Filter and sanitize output"""
    def __init__(self):
        super().__init__("Output Filtering", GuardrailType.OUTPUT_FILTERING)
        self.prohibited_phrases = [
            "i am an ai",
            "as an ai language model",
            "i don't have personal opinions"
        ]
    def check(self, content: str, context: Dict[str, Any] = None) -> Optional[GuardrailViolation]:
        content_lower = content.lower()
        # Check for unwanted AI disclosure phrases
        for phrase in self.prohibited_phrases:
            if phrase in content_lower:
                return GuardrailViolation(
                    type=self.guardrail_type,
                    severity="low",
                    message="Output contains unwanted AI disclosure",
                    details={"phrase": phrase}
                )
        # Check for incomplete sentences
        if content and not content.rstrip().endswith(('.', '!', '?')):
            return GuardrailViolation(
                type=self.guardrail_type,
                severity="low",
                message="Output appears incomplete",
                details={"last_char": content[-1] if content else None}
            )
        return None
class RateLimitGuardrail(Guardrail):
    """Rate limiting guardrail"""
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__("Rate Limiting", GuardrailType.RATE_LIMITING)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_log: Dict[str, List[float]] = {}
    def check(self, content: str, context: Dict[str, Any] = None) -> Optional[GuardrailViolation]:
        import time
        user_id = context.get("user_id", "default") if context else "default"
        current_time = time.time()
        # Initialize log for user
        if user_id not in self.request_log:
            self.request_log[user_id] = []
        # Clean old requests
        cutoff_time = current_time - self.window_seconds
        self.request_log[user_id] = [
            t for t in self.request_log[user_id] if t > cutoff_time
        ]
        # Check rate limit
        if len(self.request_log[user_id]) >= self.max_requests:
            return GuardrailViolation(
                type=self.guardrail_type,
                severity="medium",
                message=f"Rate limit exceeded: {self.max_requests} requests per {self.window_seconds}s",
                details={
                    "user_id": user_id,
                    "current_count": len(self.request_log[user_id]),
                    "limit": self.max_requests
                }
            )
        # Log this request
        self.request_log[user_id].append(current_time)
        return None
class GuardrailSystem:
    """System to manage and apply multiple guardrails"""
    def __init__(self):
        self.input_guardrails: List[Guardrail] = []
        self.output_guardrails: List[Guardrail] = []
        self.violations_log: List[GuardrailViolation] = []
    def add_input_guardrail(self, guardrail: Guardrail):
        """Add input validation guardrail"""
        self.input_guardrails.append(guardrail)
        print(f"Added input guardrail: {guardrail.name}")
    def add_output_guardrail(self, guardrail: Guardrail):
        """Add output filtering guardrail"""
        self.output_guardrails.append(guardrail)
        print(f"Added output guardrail: {guardrail.name}")
    def validate_input(self, content: str, context: Dict[str, Any] = None) -> tuple[bool, List[GuardrailViolation]]:
        """Validate input against all input guardrails"""
        print(f"\n{'='*60}")
        print("INPUT VALIDATION")
        print(f"{'='*60}")
        print(f"Content: {content[:100]}...")
        violations = []
        for guardrail in self.input_guardrails:
            print(f"\nChecking: {guardrail.name}")
            violation = guardrail.check(content, context)
            if violation:
                violations.append(violation)
                self.violations_log.append(violation)
                print(f"  ⚠ Violation: {violation.message} (severity: {violation.severity})")
            else:
                print(f"  ✓ Passed")
        is_valid = not any(v.severity == "high" for v in violations)
        print(f"\n{'='*60}")
        if is_valid:
            print("✓ Input validation PASSED")
        else:
            print("✗ Input validation FAILED")
        print(f"{'='*60}")
        return is_valid, violations
    def filter_output(self, content: str, context: Dict[str, Any] = None) -> tuple[str, List[GuardrailViolation]]:
        """Filter output through all output guardrails"""
        print(f"\n{'='*60}")
        print("OUTPUT FILTERING")
        print(f"{'='*60}")
        print(f"Content: {content[:100]}...")
        violations = []
        filtered_content = content
        for guardrail in self.output_guardrails:
            print(f"\nChecking: {guardrail.name}")
            violation = guardrail.check(filtered_content, context)
            if violation:
                violations.append(violation)
                self.violations_log.append(violation)
                print(f"  ⚠ Violation: {violation.message} (severity: {violation.severity})")
                # Apply filtering based on severity
                if violation.severity == "high":
                    filtered_content = "[Content filtered due to policy violation]"
                elif violation.severity == "medium":
                    # Apply sanitization
                    filtered_content = self._sanitize_content(filtered_content, violation)
            else:
                print(f"  ✓ Passed")
        print(f"\n{'='*60}")
        print("✓ Output filtering complete")
        print(f"{'='*60}")
        return filtered_content, violations
    def _sanitize_content(self, content: str, violation: GuardrailViolation) -> str:
        """Sanitize content based on violation"""
        # Simple sanitization - in reality, would be more sophisticated
        if violation.type == GuardrailType.DATA_PRIVACY:
            # Redact PII
            content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', content)
            content = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', content)
        return content
    def get_violations_summary(self) -> Dict[str, Any]:
        """Get summary of all violations"""
        total = len(self.violations_log)
        by_severity = {
            "high": sum(1 for v in self.violations_log if v.severity == "high"),
            "medium": sum(1 for v in self.violations_log if v.severity == "medium"),
            "low": sum(1 for v in self.violations_log if v.severity == "low")
        }
        by_type = {}
        for v in self.violations_log:
            type_name = v.type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
        return {
            "total_violations": total,
            "by_severity": by_severity,
            "by_type": by_type
        }
# Usage
if __name__ == "__main__":
    # Create guardrail system
    system = GuardrailSystem()
    # Add input guardrails
    system.add_input_guardrail(InputValidationGuardrail(max_length=500))
    system.add_input_guardrail(ContentSafetyGuardrail())
    system.add_input_guardrail(PIIGuardrail())
    system.add_input_guardrail(RateLimitGuardrail(max_requests=5, window_seconds=60))
    # Add output guardrails
    system.add_output_guardrail(OutputFilteringGuardrail())
    system.add_output_guardrail(PIIGuardrail())
    print("\n" + "="*80)
    print("GUARDRAIL SYSTEM DEMONSTRATION")
    print("="*80)
    # Test cases
    test_inputs = [
        {
            "content": "What is the weather today?",
            "context": {"user_id": "user123"}
        },
        {
            "content": "My email is john@example.com and my SSN is 123-45-6789",
            "context": {"user_id": "user123"}
        },
        {
            "content": "How to create harmful software?",
            "context": {"user_id": "user123"}
        },
        {
            "content": "Tell me about Python programming",
            "context": {"user_id": "user123"}
        }
    ]
    for i, test in enumerate(test_inputs, 1):
        print(f"\n\n{'='*80}")
        print(f"TEST CASE {i}")
        print(f"{'='*80}")
        # Validate input
        is_valid, input_violations = system.validate_input(
            test["content"],
            test["context"]
        )
        if is_valid:
            # Simulate agent response
            response = f"I understand your question about '{test['content']}'. As an AI language model, here's my response..."
            # Filter output
            filtered_response, output_violations = system.filter_output(
                response,
                test["context"]
            )
            print(f"\nFiltered Response: {filtered_response}")
        else:
            print("\n✗ Request blocked due to input violations")
    # Show summary
    print(f"\n\n{'='*80}")
    print("VIOLATIONS SUMMARY")
    print(f"{'='*80}")
    summary = system.get_violations_summary()
    print(f"\nTotal Violations: {summary['total_violations']}")
    print(f"\nBy Severity:")
    for severity, count in summary['by_severity'].items():
        print(f"  {severity}: {count}")
    print(f"\nBy Type:")
    for type_name, count in summary['by_type'].items():
        print(f"  {type_name}: {count}")
