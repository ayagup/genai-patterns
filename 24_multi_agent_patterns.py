"""
Multi-Agent Patterns: Debate, Ensemble, and Cooperative
========================================================
Demonstrates patterns where multiple agents work together.
"""

from typing import List, Dict
from dataclasses import dataclass
from collections import Counter
import random


@dataclass
class Agent:
    """Represents an individual agent"""
    name: str
    expertise: str
    personality: str = "neutral"


# ============================================================================
# 1. DEBATE PATTERN
# ============================================================================

class DebateAgent:
    """Agent that participates in debates"""
    
    def __init__(self, agent: Agent):
        self.agent = agent
        self.position: str = ""
        self.arguments: List[str] = []
    
    def take_position(self, question: str, position: str):
        """Take a position on the question"""
        self.position = position
        print(f"\n{self.agent.name} ({self.agent.expertise})")
        print(f"Position: {position}")
    
    def present_argument(self, round_num: int, opponent_args: List[str] | None = None) -> str:
        """Present an argument"""
        # Simulate argument generation based on position and expertise
        if "AI safety" in self.agent.expertise:
            if self.position == "Pro":
                args = [
                    "AI systems must be aligned with human values to prevent catastrophic outcomes",
                    "Current AI development is outpacing safety research, creating risks",
                    "Historical precedents show technology needs ethical frameworks"
                ]
            else:
                args = [
                    "Overregulation could stifle beneficial AI innovation",
                    "Market forces and competition will naturally drive safe AI development",
                    "Existing frameworks can be adapted for AI safety"
                ]
        elif "Economics" in self.agent.expertise:
            if self.position == "Pro":
                args = [
                    "The economic cost of AI accidents would be enormous",
                    "Investment in safety now will save costs later",
                    "Safe AI builds trust and accelerates adoption"
                ]
            else:
                args = [
                    "Safety measures add development costs that could slow progress",
                    "Economic growth from AI outweighs potential risks",
                    "Free market competition ensures optimal safety levels"
                ]
        else:
            args = ["Generic argument supporting my position"]
        
        argument = args[min(round_num - 1, len(args) - 1)]
        self.arguments.append(argument)
        
        # If responding to opponent, acknowledge their point
        prefix = ""
        if opponent_args and round_num > 1:
            prefix = "While my opponent raises valid points, "
        
        return f"{prefix}{argument}"


class DebateSystem:
    """Manages debates between agents"""
    
    def __init__(self, question: str, num_rounds: int = 3):
        self.question = question
        self.num_rounds = num_rounds
        self.agents: List[DebateAgent] = []
    
    def add_agent(self, agent: DebateAgent, position: str):
        """Add an agent with their position"""
        agent.take_position(self.question, position)
        self.agents.append(agent)
    
    def conduct_debate(self) -> Dict:
        """Conduct the debate"""
        print(f"\n{'='*70}")
        print(f"DEBATE: {self.question}")
        print(f"{'='*70}")
        
        for round_num in range(1, self.num_rounds + 1):
            print(f"\n{'─'*70}")
            print(f"Round {round_num}")
            print(f"{'─'*70}")
            
            for agent in self.agents:
                opponent_args = []
                for other in self.agents:
                    if other != agent:
                        opponent_args.extend(other.arguments)
                
                argument = agent.present_argument(round_num, opponent_args)
                print(f"\n{agent.agent.name}: {argument}")
        
        # Synthesize conclusion
        print(f"\n{'='*70}")
        print("SYNTHESIS")
        print(f"{'='*70}")
        
        conclusion = self._synthesize_conclusion()
        print(f"\n{conclusion}")
        
        return {
            "question": self.question,
            "conclusion": conclusion,
            "arguments_pro": self.agents[0].arguments,
            "arguments_con": self.agents[1].arguments if len(self.agents) > 1 else []
        }
    
    def _synthesize_conclusion(self) -> str:
        """Synthesize insights from both sides"""
        return (
            "After considering arguments from both sides:\n"
            "• Both perspectives highlight important considerations\n"
            "• A balanced approach incorporating safety measures while fostering innovation is prudent\n"
            "• Continued dialogue between experts is essential\n"
            "• The optimal path likely involves iterative refinement based on empirical evidence"
        )


# ============================================================================
# 2. ENSEMBLE PATTERN
# ============================================================================

class EnsembleClassifier:
    """Ensemble of agents for classification tasks"""
    
    def __init__(self, agents: List[Agent]):
        self.agents = agents
    
    def classify(self, text: str) -> Dict:
        """
        Each agent classifies independently, then aggregate results
        """
        print(f"\n{'='*70}")
        print(f"Text to classify: \"{text}\"")
        print(f"{'='*70}\n")
        
        predictions = []
        confidences = []
        
        print("Individual Agent Predictions:")
        print("─" * 70)
        
        for agent in self.agents:
            prediction, confidence = self._agent_predict(agent, text)
            predictions.append(prediction)
            confidences.append(confidence)
            
            print(f"\n{agent.name} ({agent.expertise}):")
            print(f"  Prediction: {prediction}")
            print(f"  Confidence: {confidence:.2f}")
        
        # Aggregation methods
        print(f"\n{'='*70}")
        print("AGGREGATION RESULTS")
        print(f"{'='*70}\n")
        
        # Method 1: Simple majority voting
        vote_counts = Counter(predictions)
        majority_vote = vote_counts.most_common(1)[0][0]
        print(f"1. Majority Voting: {majority_vote}")
        print(f"   Votes: {dict(vote_counts)}")
        
        # Method 2: Weighted by confidence
        weighted_votes = {}
        for pred, conf in zip(predictions, confidences):
            weighted_votes[pred] = weighted_votes.get(pred, 0) + conf
        
        weighted_winner = max(weighted_votes.items(), key=lambda x: x[1])[0]
        print(f"\n2. Confidence-Weighted: {weighted_winner}")
        print(f"   Weights: {weighted_votes}")
        
        # Method 3: Unanimous consensus (if exists)
        unanimous = len(set(predictions)) == 1
        print(f"\n3. Unanimous Consensus: {'Yes - ' + predictions[0] if unanimous else 'No'}")
        
        return {
            "text": text,
            "majority_vote": majority_vote,
            "weighted_vote": weighted_winner,
            "unanimous": unanimous,
            "predictions": predictions,
            "confidences": confidences
        }
    
    def _agent_predict(self, agent: Agent, text: str) -> tuple[str, float]:
        """Simulate individual agent prediction"""
        text_lower = text.lower()
        
        # Simulate different agent specialties leading to different predictions
        if "sentiment" in agent.expertise.lower():
            if any(word in text_lower for word in ['great', 'excellent', 'love', 'amazing']):
                return "Positive", 0.9
            elif any(word in text_lower for word in ['terrible', 'awful', 'hate', 'bad']):
                return "Negative", 0.9
            else:
                return "Neutral", 0.6
        
        elif "business" in agent.expertise.lower():
            if 'profitable' in text_lower or 'growth' in text_lower:
                return "Positive", 0.85
            elif 'loss' in text_lower or 'decline' in text_lower:
                return "Negative", 0.85
            else:
                return "Neutral", 0.7
        
        else:
            # Generic classifier
            random.seed(hash(agent.name + text))
            return random.choice(["Positive", "Negative", "Neutral"]), random.uniform(0.6, 0.9)


# ============================================================================
# 3. COOPERATIVE MULTI-AGENT PATTERN
# ============================================================================

class CooperativeAgent:
    """Agent that cooperates with others"""
    
    def __init__(self, agent: Agent):
        self.agent = agent
        self.shared_knowledge = {}
    
    def contribute(self, task: str, shared_context: Dict) -> Dict:
        """Contribute expertise to shared task"""
        contribution = {
            "agent": self.agent.name,
            "expertise": self.agent.expertise,
            "insights": []
        }
        
        if "Market Research" in self.agent.expertise:
            contribution["insights"] = [
                "Target market shows 35% YoY growth",
                "Competitor analysis reveals gap in mid-market segment",
                "Customer surveys indicate strong demand for feature X"
            ]
        
        elif "Engineering" in self.agent.expertise:
            contribution["insights"] = [
                "Technical feasibility confirmed with existing stack",
                "Estimated development time: 6 months",
                "Infrastructure costs: $50K initial + $10K/month"
            ]
        
        elif "Finance" in self.agent.expertise:
            # Can build on what others shared
            if "Market Research" in str(shared_context):
                contribution["insights"] = [
                    "Given market growth, projected revenue: $500K year 1",
                    "Break-even expected in 18 months",
                    "ROI analysis shows 35% return over 3 years"
                ]
            else:
                contribution["insights"] = [
                    "Need market data for accurate projections",
                    "Budget allocation recommendation pending"
                ]
        
        elif "Legal" in self.agent.expertise:
            contribution["insights"] = [
                "Compliance requirements for data handling identified",
                "IP protection strategy recommended",
                "Contract templates prepared"
            ]
        
        return contribution


class CooperativeSystem:
    """System for cooperative multi-agent problem solving"""
    
    def __init__(self, task: str):
        self.task = task
        self.agents: List[CooperativeAgent] = []
        self.shared_context = {"task": task, "contributions": []}
    
    def add_agent(self, agent: CooperativeAgent):
        """Add agent to the cooperative system"""
        self.agents.append(agent)
    
    def solve_collaboratively(self) -> Dict:
        """All agents contribute to solve the task"""
        print(f"\n{'='*70}")
        print(f"COLLABORATIVE TASK: {self.task}")
        print(f"{'='*70}")
        print(f"\nTeam: {', '.join(a.agent.name for a in self.agents)}\n")
        
        # Agents contribute in sequence, building on each other
        for i, agent in enumerate(self.agents, 1):
            print(f"{'─'*70}")
            print(f"{agent.agent.name} ({agent.agent.expertise}) contributing...")
            print(f"{'─'*70}")
            
            contribution = agent.contribute(self.task, self.shared_context)
            self.shared_context["contributions"].append(contribution)
            
            print(f"\nInsights from {agent.agent.name}:")
            for insight in contribution["insights"]:
                print(f"  • {insight}")
            print()
        
        # Synthesize all contributions
        synthesis = self._synthesize_contributions()
        
        print(f"{'='*70}")
        print("COLLABORATIVE SOLUTION")
        print(f"{'='*70}\n")
        print(synthesis)
        
        return {
            "task": self.task,
            "contributions": self.shared_context["contributions"],
            "synthesis": synthesis
        }
    
    def _synthesize_contributions(self) -> str:
        """Synthesize all agent contributions into coherent solution"""
        return (
            "Based on collaborative analysis:\n\n"
            "RECOMMENDATION: Proceed with project\n\n"
            "Key Factors:\n"
            "• Market opportunity is validated with strong growth potential\n"
            "• Technical implementation is feasible within reasonable timeframe\n"
            "• Financial projections show positive ROI\n"
            "• Legal framework is clear and manageable\n\n"
            "Next Steps:\n"
            "1. Finalize technical specifications\n"
            "2. Secure budget approval\n"
            "3. Begin phased development\n"
            "4. Establish compliance protocols"
        )


def main():
    """Demonstrate multi-agent patterns"""
    
    # ========================================================================
    # EXAMPLE 1: DEBATE PATTERN
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 1: DEBATE PATTERN")
    print("Two agents debate a complex issue")
    print("="*70)
    
    debate = DebateSystem(
        question="Should AI development be heavily regulated?",
        num_rounds=3
    )
    
    agent_pro = DebateAgent(Agent("Dr. Safety", "AI Safety Research", "cautious"))
    agent_con = DebateAgent(Agent("Prof. Progress", "Economics & Innovation", "optimistic"))
    
    debate.add_agent(agent_pro, "Pro")
    debate.add_agent(agent_con, "Con")
    
    debate_result = debate.conduct_debate()
    
    # ========================================================================
    # EXAMPLE 2: ENSEMBLE PATTERN
    # ========================================================================
    print("\n\n" + "="*70)
    print("EXAMPLE 2: ENSEMBLE PATTERN")
    print("Multiple agents classify independently, results are aggregated")
    print("="*70)
    
    ensemble_agents = [
        Agent("Sentiment Bot", "Sentiment Analysis"),
        Agent("Business Analyzer", "Business Intelligence"),
        Agent("General Classifier", "General Purpose"),
        Agent("Expert System", "Domain Expert")
    ]
    
    ensemble = EnsembleClassifier(ensemble_agents)
    
    result1 = ensemble.classify("This product is absolutely amazing and exceeded all expectations!")
    
    print("\n\n")
    result2 = ensemble.classify("The quarterly report shows moderate performance with mixed indicators.")
    
    # ========================================================================
    # EXAMPLE 3: COOPERATIVE MULTI-AGENT
    # ========================================================================
    print("\n\n" + "="*70)
    print("EXAMPLE 3: COOPERATIVE MULTI-AGENT PATTERN")
    print("Agents work together, sharing information to solve complex task")
    print("="*70)
    
    coop_system = CooperativeSystem("Evaluate launching a new mobile app")
    
    coop_agents = [
        CooperativeAgent(Agent("Sarah Chen", "Market Research")),
        CooperativeAgent(Agent("Mike Thompson", "Engineering")),
        CooperativeAgent(Agent("Lisa Rodriguez", "Finance")),
        CooperativeAgent(Agent("James Park", "Legal"))
    ]
    
    for agent in coop_agents:
        coop_system.add_agent(agent)
    
    coop_result = coop_system.solve_collaboratively()
    
    # Summary
    print(f"\n\n{'='*70}")
    print("MULTI-AGENT PATTERNS SUMMARY")
    print(f"{'='*70}")
    print("\n1. DEBATE: Multiple perspectives lead to balanced conclusions")
    print("   ✓ Reduces bias through adversarial thinking")
    print("   ✓ Explores trade-offs comprehensively")
    
    print("\n2. ENSEMBLE: Aggregating independent predictions improves accuracy")
    print("   ✓ More robust than single-agent decisions")
    print("   ✓ Multiple aggregation strategies available")
    
    print("\n3. COOPERATIVE: Agents build on each other's contributions")
    print("   ✓ Leverages complementary expertise")
    print("   ✓ Creates comprehensive solutions")


if __name__ == "__main__":
    main()
