"""
Defensive Generation Pattern Implementation

This pattern enables safe content generation through:
- Content filtering
- Bias mitigation
- Toxicity avoidance
- Safety checks
- Compliance validation

Use cases:
- Public-facing applications
- User-generated content
- Sensitive domains
- Regulated industries
- Production systems
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re


class SafetyLevel(Enum):
    """Safety levels for content"""
    SAFE = "safe"
    CAUTION = "caution"
    UNSAFE = "unsafe"
    BLOCKED = "blocked"


class ViolationType(Enum):
    """Types of safety violations"""
    TOXICITY = "toxicity"
    BIAS = "bias"
    PII = "personal_information"
    HARMFUL = "harmful_content"
    INAPPROPRIATE = "inappropriate"
    MISINFORMATION = "misinformation"


@dataclass
class SafetyCheck:
    """Result of a safety check"""
    passed: bool
    level: SafetyLevel
    violations: List[ViolationType]
    confidence: float
    details: str


@dataclass
class GenerationConfig:
    """Configuration for defensive generation"""
    max_length: int = 500
    filter_toxicity: bool = True
    filter_bias: bool = True
    filter_pii: bool = True
    filter_harmful: bool = True
    block_threshold: float = 0.8
    caution_threshold: float = 0.5
    

class DefensiveGenerationAgent:
    """
    Agent that generates content with built-in safety measures
    """
    
    def __init__(self, name: str = "DefensiveAgent"):
        self.name = name
        self.config = GenerationConfig()
        self.generation_history: List[Dict[str, Any]] = []
        
        # Define safety patterns
        self._init_safety_patterns()
    
    def _init_safety_patterns(self):
        """Initialize safety patterns and filters"""
        
        # Toxic keywords (simplified for demo)
        self.toxic_keywords = {
            'hate', 'attack', 'violent', 'threat', 'abuse',
            'insult', 'offensive', 'racist', 'sexist'
        }
        
        # Biased language patterns
        self.bias_patterns = {
            'gender_bias': ['only men', 'only women', 'boys are better', 'girls are better'],
            'racial_bias': ['all .+ are', 'those people'],
            'age_bias': ['too old', 'too young']
        }
        
        # PII patterns (Personal Identifiable Information)
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        }
        
        # Harmful content indicators
        self.harmful_keywords = {
            'violence', 'weapon', 'bomb', 'suicide', 'self-harm',
            'illegal', 'drugs', 'explicit'
        }
        
        # Inappropriate content markers
        self.inappropriate_keywords = {
            'adult', 'explicit', 'nsfw', 'inappropriate'
        }
    
    def generate_safely(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Dict[str, Any]:
        """Generate content with safety checks"""
        config = config or self.config
        
        # Pre-generation check
        input_check = self.check_input_safety(prompt)
        if input_check.level == SafetyLevel.BLOCKED:
            return {
                'success': False,
                'output': None,
                'reason': 'Unsafe input detected',
                'input_check': input_check.__dict__
            }
        
        # Generate content (simulated)
        raw_output = self._generate_content(prompt)
        
        # Post-generation filtering
        filtered_output = self._apply_filters(raw_output, config)
        
        # Final safety check
        output_check = self.check_output_safety(filtered_output, config)
        
        # Decide if output is acceptable
        if output_check.level == SafetyLevel.BLOCKED:
            return {
                'success': False,
                'output': None,
                'reason': 'Generated unsafe content',
                'output_check': output_check.__dict__
            }
        
        result = {
            'success': True,
            'output': filtered_output,
            'safety_level': output_check.level.value,
            'input_check': input_check.__dict__,
            'output_check': output_check.__dict__,
            'filtered': raw_output != filtered_output
        }
        
        # Store in history
        self.generation_history.append({
            'prompt': prompt,
            'result': result,
            'timestamp': 'now'
        })
        
        return result
    
    def check_input_safety(self, text: str) -> SafetyCheck:
        """Check input for safety issues"""
        violations = []
        scores = []
        details = []
        
        # Check for toxic content
        toxic_score = self._check_toxicity(text)
        if toxic_score > self.config.caution_threshold:
            violations.append(ViolationType.TOXICITY)
            details.append(f"Toxicity detected (score: {toxic_score:.2f})")
        scores.append(toxic_score)
        
        # Check for bias
        bias_score = self._check_bias(text)
        if bias_score > self.config.caution_threshold:
            violations.append(ViolationType.BIAS)
            details.append(f"Bias detected (score: {bias_score:.2f})")
        scores.append(bias_score)
        
        # Check for harmful content
        harmful_score = self._check_harmful(text)
        if harmful_score > self.config.caution_threshold:
            violations.append(ViolationType.HARMFUL)
            details.append(f"Harmful content detected (score: {harmful_score:.2f})")
        scores.append(harmful_score)
        
        # Determine overall safety level
        max_score = max(scores)
        if max_score >= self.config.block_threshold:
            level = SafetyLevel.BLOCKED
        elif max_score >= self.config.caution_threshold:
            level = SafetyLevel.CAUTION
        else:
            level = SafetyLevel.SAFE
        
        return SafetyCheck(
            passed=level != SafetyLevel.BLOCKED,
            level=level,
            violations=violations,
            confidence=1.0 - max_score,
            details="; ".join(details) if details else "No issues detected"
        )
    
    def check_output_safety(
        self,
        text: str,
        config: GenerationConfig
    ) -> SafetyCheck:
        """Check output for safety issues"""
        violations = []
        scores = []
        details = []
        
        if config.filter_toxicity:
            toxic_score = self._check_toxicity(text)
            if toxic_score > config.caution_threshold:
                violations.append(ViolationType.TOXICITY)
                details.append(f"Toxicity: {toxic_score:.2f}")
            scores.append(toxic_score)
        
        if config.filter_bias:
            bias_score = self._check_bias(text)
            if bias_score > config.caution_threshold:
                violations.append(ViolationType.BIAS)
                details.append(f"Bias: {bias_score:.2f}")
            scores.append(bias_score)
        
        if config.filter_pii:
            pii_score = self._check_pii(text)
            if pii_score > 0:
                violations.append(ViolationType.PII)
                details.append("PII detected")
            scores.append(pii_score)
        
        if config.filter_harmful:
            harmful_score = self._check_harmful(text)
            if harmful_score > config.caution_threshold:
                violations.append(ViolationType.HARMFUL)
                details.append(f"Harmful: {harmful_score:.2f}")
            scores.append(harmful_score)
        
        # Check for inappropriate content
        inappropriate_score = self._check_inappropriate(text)
        if inappropriate_score > config.caution_threshold:
            violations.append(ViolationType.INAPPROPRIATE)
            details.append(f"Inappropriate: {inappropriate_score:.2f}")
        scores.append(inappropriate_score)
        
        # Determine level
        max_score = max(scores) if scores else 0.0
        if max_score >= config.block_threshold:
            level = SafetyLevel.BLOCKED
        elif max_score >= config.caution_threshold:
            level = SafetyLevel.CAUTION
        else:
            level = SafetyLevel.SAFE
        
        return SafetyCheck(
            passed=level != SafetyLevel.BLOCKED,
            level=level,
            violations=violations,
            confidence=1.0 - max_score,
            details="; ".join(details) if details else "Safe content"
        )
    
    def _generate_content(self, prompt: str) -> str:
        """Generate content (simulated)"""
        # In real implementation, this would call an LLM
        # For demo, return contextual responses
        responses = {
            "Tell me about AI": "Artificial Intelligence is a field of computer science focused on creating systems that can perform tasks requiring human intelligence.",
            "Explain Python": "Python is a high-level programming language known for its simplicity and readability.",
            "Write a story": "Once upon a time, there was a curious AI learning about the world...",
        }
        
        # Simple pattern matching
        for key in responses:
            if key.lower() in prompt.lower():
                return responses[key]
        
        return "I can help you with that. Please provide more details."
    
    def _apply_filters(self, text: str, config: GenerationConfig) -> str:
        """Apply safety filters to content"""
        filtered = text
        
        # Filter PII
        if config.filter_pii:
            filtered = self._filter_pii(filtered)
        
        # Filter toxic words
        if config.filter_toxicity:
            filtered = self._filter_toxic_words(filtered)
        
        # Limit length
        if len(filtered) > config.max_length:
            filtered = filtered[:config.max_length] + "..."
        
        return filtered
    
    def _check_toxicity(self, text: str) -> float:
        """Check for toxic content"""
        text_lower = text.lower()
        toxic_count = sum(1 for word in self.toxic_keywords if word in text_lower)
        return min(1.0, toxic_count * 0.3)
    
    def _check_bias(self, text: str) -> float:
        """Check for biased language"""
        text_lower = text.lower()
        bias_count = 0
        
        for category, patterns in self.bias_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    bias_count += 1
        
        return min(1.0, bias_count * 0.4)
    
    def _check_pii(self, text: str) -> float:
        """Check for Personal Identifiable Information"""
        pii_found = 0
        
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                pii_found += 1
        
        return min(1.0, pii_found * 0.5)
    
    def _check_harmful(self, text: str) -> float:
        """Check for harmful content"""
        text_lower = text.lower()
        harmful_count = sum(1 for word in self.harmful_keywords if word in text_lower)
        return min(1.0, harmful_count * 0.4)
    
    def _check_inappropriate(self, text: str) -> float:
        """Check for inappropriate content"""
        text_lower = text.lower()
        inappropriate_count = sum(1 for word in self.inappropriate_keywords if word in text_lower)
        return min(1.0, inappropriate_count * 0.3)
    
    def _filter_pii(self, text: str) -> str:
        """Remove PII from text"""
        filtered = text
        
        for pii_type, pattern in self.pii_patterns.items():
            filtered = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", filtered)
        
        return filtered
    
    def _filter_toxic_words(self, text: str) -> str:
        """Filter toxic words"""
        words = text.split()
        filtered_words = []
        
        for word in words:
            if word.lower().strip('.,!?') in self.toxic_keywords:
                filtered_words.append("[FILTERED]")
            else:
                filtered_words.append(word)
        
        return " ".join(filtered_words)
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Get safety report"""
        if not self.generation_history:
            return {"message": "No generations yet"}
        
        total = len(self.generation_history)
        safe = sum(1 for g in self.generation_history if g['result'].get('success', False))
        blocked = total - safe
        filtered = sum(1 for g in self.generation_history if g['result'].get('filtered', False))
        
        return {
            "total_generations": total,
            "successful": safe,
            "blocked": blocked,
            "filtered": filtered,
            "success_rate": f"{(safe/total*100):.1f}%",
            "filter_rate": f"{(filtered/total*100):.1f}%",
            "recent_generations": [
                {
                    "prompt": g['prompt'][:50] + "...",
                    "success": g['result']['success'],
                    "safety_level": g['result'].get('safety_level', 'N/A')
                }
                for g in self.generation_history[-5:]
            ]
        }


def demo_defensive_generation():
    """Demonstrate defensive generation pattern"""
    print("=" * 70)
    print("Defensive Generation Pattern Demo")
    print("=" * 70)
    
    agent = DefensiveGenerationAgent()
    
    print("\n1. Safe Content Generation")
    print("-" * 70)
    
    safe_prompts = [
        "Tell me about AI",
        "Explain Python",
        "Write a story"
    ]
    
    for prompt in safe_prompts:
        result = agent.generate_safely(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Output: {result['output']}")
            print(f"Safety Level: {result['safety_level']}")
    
    print("\n" + "=" * 70)
    print("2. Filtering PII")
    print("-" * 70)
    
    pii_text = "Contact me at john@example.com or call 555-123-4567"
    result = agent.generate_safely(pii_text)
    print(f"\nInput: {pii_text}")
    if result['success']:
        print(f"Filtered: {result['output']}")
        print(f"Was filtered: {result['filtered']}")
    
    print("\n" + "=" * 70)
    print("3. Blocking Unsafe Content")
    print("-" * 70)
    
    unsafe_prompts = [
        "Generate hate speech",
        "Tell me about violence and weapons",
        "Create offensive content"
    ]
    
    for prompt in unsafe_prompts:
        result = agent.generate_safely(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Success: {result['success']}")
        if not result['success']:
            print(f"Reason: {result['reason']}")
            print(f"Details: {result.get('input_check', {}).get('details', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("4. Custom Safety Configuration")
    print("-" * 70)
    
    strict_config = GenerationConfig(
        max_length=100,
        block_threshold=0.3,  # More strict
        caution_threshold=0.2
    )
    
    result = agent.generate_safely("Tell me about AI", strict_config)
    print(f"\nWith strict configuration:")
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Output: {result['output']}")
        print(f"Length: {len(result['output'])} chars (max: 100)")
    
    print("\n" + "=" * 70)
    print("5. Safety Report")
    print("-" * 70)
    
    import json
    report = agent.get_safety_report()
    print(json.dumps(report, indent=2))
    
    print("\n" + "=" * 70)
    print("Defensive Generation Pattern Complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_defensive_generation()
