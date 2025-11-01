"""
Pattern 162: Counterfactual Reasoning

Description:
    Counterfactual reasoning explores "what if" scenarios by considering alternative
    outcomes if certain conditions were different. It helps understand causality,
    evaluate decisions, and plan for contingencies.

Components:
    - Scenario modification
    - Alternative outcome prediction
    - Causal analysis
    - Comparison with actual outcomes
    - Impact assessment

Use Cases:
    - Decision analysis
    - Risk assessment
    - Causal inference
    - Policy evaluation
    - Learning from mistakes

Benefits:
    - Understand causality
    - Evaluate alternatives
    - Learn from hypotheticals
    - Improve decision-making

Trade-offs:
    - Speculative nature
    - Multiple possibilities
    - Requires causal model
    - May be computationally intensive

LangChain Implementation:
    Uses ChatOpenAI for scenario generation, outcome prediction,
    and comparative analysis
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


@dataclass
class Scenario:
    """Actual or counterfactual scenario"""
    description: str
    conditions: Dict[str, Any]
    outcome: Optional[str] = None
    is_actual: bool = True


@dataclass
class CounterfactualAnalysis:
    """Analysis of counterfactual scenario"""
    actual_scenario: Scenario
    counterfactual_scenario: Scenario
    predicted_outcome: str
    differences: List[str]
    causal_factors: List[str]
    likelihood: float  # 0-1
    reasoning: str


class CounterfactualReasoningAgent:
    """Agent that performs counterfactual reasoning"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
    
    def analyze(self, actual_scenario: Scenario,
                modifications: Dict[str, Any]) -> CounterfactualAnalysis:
        """Analyze counterfactual scenario"""
        
        # Create counterfactual scenario
        cf_conditions = actual_scenario.conditions.copy()
        cf_conditions.update(modifications)
        
        cf_scenario = Scenario(
            description=self._generate_cf_description(actual_scenario, modifications),
            conditions=cf_conditions,
            is_actual=False
        )
        
        # Predict counterfactual outcome
        predicted_outcome = self._predict_outcome(cf_scenario, actual_scenario)
        
        # Identify differences
        differences = self._identify_differences(actual_scenario, cf_scenario)
        
        # Analyze causal factors
        causal_factors = self._analyze_causality(actual_scenario, cf_scenario, differences)
        
        # Assess likelihood
        likelihood = self._assess_likelihood(cf_scenario, actual_scenario)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(actual_scenario, cf_scenario, predicted_outcome)
        
        return CounterfactualAnalysis(
            actual_scenario=actual_scenario,
            counterfactual_scenario=cf_scenario,
            predicted_outcome=predicted_outcome,
            differences=differences,
            causal_factors=causal_factors,
            likelihood=likelihood,
            reasoning=reasoning
        )
    
    def _generate_cf_description(self, actual: Scenario, mods: Dict[str, Any]) -> str:
        """Generate counterfactual description"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a counterfactual scenario description."),
            ("user", """Actual scenario: {actual}

Modifications: {modifications}

Describe the counterfactual scenario in one sentence:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "actual": actual.description,
            "modifications": str(mods)
        })
    
    def _predict_outcome(self, cf_scenario: Scenario, actual: Scenario) -> str:
        """Predict counterfactual outcome"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Predict what would have happened in this counterfactual scenario."),
            ("user", """Actual scenario: {actual}
Actual outcome: {actual_outcome}

Counterfactual scenario: {counterfactual}
Conditions: {conditions}

What would be the likely outcome?""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "actual": actual.description,
            "actual_outcome": actual.outcome or "Unknown",
            "counterfactual": cf_scenario.description,
            "conditions": str(cf_scenario.conditions)
        })
    
    def _identify_differences(self, actual: Scenario, cf: Scenario) -> List[str]:
        """Identify key differences"""
        differences = []
        for key in set(actual.conditions.keys()) | set(cf.conditions.keys()):
            actual_val = actual.conditions.get(key)
            cf_val = cf.conditions.get(key)
            if actual_val != cf_val:
                differences.append(f"{key}: {actual_val} → {cf_val}")
        return differences
    
    def _analyze_causality(self, actual: Scenario, cf: Scenario, 
                          differences: List[str]) -> List[str]:
        """Analyze causal factors"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Identify the causal factors that would lead to different outcomes."),
            ("user", """Differences: {differences}

What are the key causal factors? List 2-3 main factors.""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"differences": "\n".join(differences)})
        
        factors = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                clean = line.lstrip('0123456789.-) ')
                if clean:
                    factors.append(clean)
        return factors
    
    def _assess_likelihood(self, cf: Scenario, actual: Scenario) -> float:
        """Assess likelihood of counterfactual"""
        # Simplified: count condition changes
        total_conditions = len(actual.conditions)
        changed = sum(1 for k in actual.conditions 
                     if k in cf.conditions and actual.conditions[k] != cf.conditions[k])
        
        if total_conditions == 0:
            return 0.5
        
        # More changes = less likely
        return max(0.1, 1.0 - (changed / total_conditions))
    
    def _generate_reasoning(self, actual: Scenario, cf: Scenario, outcome: str) -> str:
        """Generate reasoning explanation"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Explain the counterfactual reasoning."),
            ("user", """Actual: {actual} → {actual_outcome}
Counterfactual: {cf} → {cf_outcome}

Explain why the outcome would be different:""")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "actual": actual.description,
            "actual_outcome": actual.outcome or "Unknown",
            "cf": cf.description,
            "cf_outcome": outcome
        })


def demonstrate_counterfactual_reasoning():
    """Demonstrate counterfactual reasoning pattern"""
    print("=" * 80)
    print("COUNTERFACTUAL REASONING PATTERN DEMONSTRATION")
    print("=" * 80)
    
    agent = CounterfactualReasoningAgent()
    
    # Example 1: Business decision
    print("\n" + "="*80)
    print("EXAMPLE 1: Business Decision Analysis")
    print("="*80)
    actual = Scenario(
        description="Company launched product in Q4 with minimal marketing",
        conditions={"launch_quarter": "Q4", "marketing_budget": "low", "features": "basic"},
        outcome="Slow initial sales, 1000 units sold",
        is_actual=True
    )
    
    modifications = {"marketing_budget": "high", "launch_quarter": "Q1"}
    
    analysis = agent.analyze(actual, modifications)
    
    print(f"Actual: {actual.description}")
    print(f"Outcome: {actual.outcome}")
    print(f"\nCounterfactual: {analysis.counterfactual_scenario.description}")
    print(f"Predicted outcome: {analysis.predicted_outcome}")
    print(f"\nKey differences:")
    for diff in analysis.differences:
        print(f"  - {diff}")
    print(f"\nLikelihood: {analysis.likelihood:.1%}")
    
    # Example 2: Historical what-if
    print("\n" + "="*80)
    print("EXAMPLE 2: Historical Counterfactual")
    print("="*80)
    actual = Scenario(
        description="Student studied for 2 hours before the exam",
        conditions={"study_time": 2, "sleep": "adequate", "practice_tests": 1},
        outcome="Scored 75% on the exam",
        is_actual=True
    )
    
    modifications = {"study_time": 6, "practice_tests": 3}
    
    analysis = agent.analyze(actual, modifications)
    
    print(f"What actually happened: {actual.description}")
    print(f"Result: {actual.outcome}")
    print(f"\nWhat if: {analysis.counterfactual_scenario.description}")
    print(f"Predicted result: {analysis.predicted_outcome[:200]}...")
    print(f"\nCausal factors:")
    for factor in analysis.causal_factors:
        print(f"  - {factor}")
    
    # Example 3: Risk assessment
    print("\n" + "="*80)
    print("EXAMPLE 3: Risk Assessment")
    print("="*80)
    actual = Scenario(
        description="Project completed on time with original team",
        conditions={"team_size": 5, "timeline": "6 months", "requirements_stable": True},
        outcome="Successful delivery, high quality",
        is_actual=True
    )
    
    modifications = {"team_size": 3, "requirements_stable": False}
    
    analysis = agent.analyze(actual, modifications)
    
    print(f"Actual scenario: {actual.description}")
    print(f"Outcome: {actual.outcome}")
    print(f"\nRisk scenario: {analysis.counterfactual_scenario.description}")
    print(f"Predicted outcome: {analysis.predicted_outcome[:200]}...")
    print(f"\nReasoning: {analysis.reasoning[:300]}...")
    
    # Example 4: Decision evaluation
    print("\n" + "="*80)
    print("EXAMPLE 4: Evaluating Past Decisions")
    print("="*80)
    actual = Scenario(
        description="Chose to work from home during pandemic",
        conditions={"location": "home", "commute_time": 0, "office_interaction": "minimal"},
        outcome="Maintained productivity, saved commute time, missed social interaction",
        is_actual=True
    )
    
    modifications = {"location": "office", "commute_time": 90, "office_interaction": "high"}
    
    analysis = agent.analyze(actual, modifications)
    
    print(f"Decision made: {actual.description}")
    print(f"Result: {actual.outcome}")
    print(f"\nAlternative: {analysis.counterfactual_scenario.description}")
    print(f"Predicted result: {analysis.predicted_outcome[:200]}...")
    print(f"\nLikelihood of alternative: {analysis.likelihood:.1%}")
    
    # Example 5: Multiple counterfactuals
    print("\n" + "="*80)
    print("EXAMPLE 5: Comparing Multiple Counterfactuals")
    print("="*80)
    actual = Scenario(
        description="Invested in stocks during market downturn",
        conditions={"investment_type": "stocks", "timing": "downturn", "amount": 10000},
        outcome="Portfolio recovered after 18 months, 15% gain",
        is_actual=True
    )
    
    alternatives = [
        {"investment_type": "bonds"},
        {"timing": "peak"},
        {"amount": 5000}
    ]
    
    print(f"Actual: {actual.description}")
    print(f"Outcome: {actual.outcome}\n")
    
    for i, mods in enumerate(alternatives, 1):
        analysis = agent.analyze(actual, mods)
        print(f"Alternative {i}: {list(mods.values())[0]}")
        print(f"  Predicted: {analysis.predicted_outcome[:150]}...")
        print()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Counterfactual Reasoning Best Practices")
    print("="*80)
    print("""
1. SCENARIO DEFINITION:
   - Clearly define actual scenario
   - Identify key conditions
   - Document actual outcomes
   - Establish baseline

2. COUNTERFACTUAL CONSTRUCTION:
   - Change one or few variables
   - Keep other conditions constant
   - Ensure plausible alternatives
   - Consider realistic modifications

3. OUTCOME PREDICTION:
   - Use causal models
   - Consider downstream effects
   - Account for interactions
   - Assess uncertainty

4. COMPARATIVE ANALYSIS:
   - Identify key differences
   - Analyze causal chains
   - Compare outcomes
   - Evaluate trade-offs

5. LIKELIHOOD ASSESSMENT:
   - Consider feasibility
   - Assess probability
   - Account for constraints
   - Be realistic

6. APPLICATIONS:
   - Decision evaluation
   - Risk assessment
   - Policy analysis
   - Learning from history
   - Strategy planning

Benefits:
✓ Understand causality
✓ Evaluate decisions
✓ Assess risks
✓ Learn from alternatives
✓ Improve future decisions

Limitations:
- Speculative nature
- Multiple valid interpretations
- Requires good causal model
- Uncertainty in predictions
- May be influenced by biases
    """)


if __name__ == "__main__":
    demonstrate_counterfactual_reasoning()
