"""
Scientific Discovery Agent Pattern

Enables agents to conduct scientific research through hypothesis generation,
experiment design, data analysis, and iterative refinement of theories.

Key Concepts:
- Hypothesis generation
- Experiment design
- Data collection and analysis
- Theory formation
- Peer review simulation

Use Cases:
- Automated research
- Hypothesis testing
- Experimental design
- Scientific reasoning
- Literature synthesis
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random
import uuid


class ResearchPhase(Enum):
    """Phases of scientific research."""
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    EXPERIMENTATION = "experimentation"
    ANALYSIS = "analysis"
    CONCLUSION = "conclusion"
    PUBLICATION = "publication"


class HypothesisStatus(Enum):
    """Status of a hypothesis."""
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class ExperimentType(Enum):
    """Types of experiments."""
    OBSERVATIONAL = "observational"
    CONTROLLED = "controlled"
    COMPARATIVE = "comparative"
    LONGITUDINAL = "longitudinal"
    SIMULATION = "simulation"


@dataclass
class Observation:
    """A scientific observation."""
    observation_id: str
    description: str
    measurements: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """A scientific hypothesis."""
    hypothesis_id: str
    statement: str
    variables: List[str]
    predicted_outcome: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    confidence: float = 0.5  # 0-1
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def update_confidence(self) -> None:
        """Update confidence based on evidence."""
        total = len(self.supporting_evidence) + len(self.contradicting_evidence)
        if total > 0:
            self.confidence = len(self.supporting_evidence) / total


@dataclass
class Experiment:
    """A scientific experiment."""
    experiment_id: str
    name: str
    experiment_type: ExperimentType
    hypothesis_id: str
    independent_variables: Dict[str, Any]
    dependent_variables: List[str]
    control_variables: Dict[str, Any]
    procedure: List[str]
    expected_results: Optional[str] = None
    actual_results: Optional[Dict[str, Any]] = None
    status: str = "designed"
    
    def run(self) -> Dict[str, Any]:
        """Simulate running the experiment."""
        # Simplified simulation
        results = {}
        for var in self.dependent_variables:
            # Generate mock results
            results[var] = random.uniform(0, 100)
        
        self.actual_results = results
        self.status = "completed"
        return results


@dataclass
class ResearchPaper:
    """A research paper."""
    paper_id: str
    title: str
    abstract: str
    hypotheses: List[str]
    experiments: List[str]
    conclusions: List[str]
    confidence_score: float
    created_at: datetime = field(default_factory=datetime.now)


class ScientificKnowledgeBase:
    """Knowledge base for scientific research."""
    
    def __init__(self):
        self.observations: List[Observation] = []
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.experiments: Dict[str, Experiment] = {}
        self.theories: List[str] = []
        self.facts: Set[str] = set()
    
    def add_observation(self, observation: Observation) -> None:
        """Add observation to knowledge base."""
        self.observations.append(observation)
    
    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Add hypothesis to knowledge base."""
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis
    
    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Get hypothesis by ID."""
        return self.hypotheses.get(hypothesis_id)
    
    def add_experiment(self, experiment: Experiment) -> None:
        """Add experiment to knowledge base."""
        self.experiments[experiment.experiment_id] = experiment
    
    def get_supported_hypotheses(self) -> List[Hypothesis]:
        """Get all supported hypotheses."""
        return [
            h for h in self.hypotheses.values()
            if h.status == HypothesisStatus.SUPPORTED
        ]
    
    def promote_to_theory(self, hypothesis_id: str) -> bool:
        """Promote well-supported hypothesis to theory."""
        hypothesis = self.hypotheses.get(hypothesis_id)
        if hypothesis and hypothesis.confidence >= 0.8:
            self.theories.append(hypothesis.statement)
            return True
        return False


class ScientificDiscoveryAgent:
    """Agent that conducts scientific research."""
    
    def __init__(self, agent_id: str, name: str, field: str):
        self.agent_id = agent_id
        self.name = name
        self.field = field
        self.knowledge_base = ScientificKnowledgeBase()
        self.current_phase = ResearchPhase.OBSERVATION
        self.research_questions: List[str] = []
    
    def make_observation(
        self,
        description: str,
        measurements: Dict[str, float]
    ) -> Observation:
        """Make and record an observation."""
        observation = Observation(
            observation_id=str(uuid.uuid4()),
            description=description,
            measurements=measurements
        )
        
        self.knowledge_base.add_observation(observation)
        print(f"\n[{self.name}] Observation recorded:")
        print(f"  {description}")
        print(f"  Measurements: {measurements}")
        
        return observation
    
    def generate_hypotheses(
        self,
        num_hypotheses: int = 3
    ) -> List[Hypothesis]:
        """Generate hypotheses from observations."""
        print(f"\n[{self.name}] Generating {num_hypotheses} hypotheses...")
        
        hypotheses = []
        
        # Analyze recent observations
        recent_obs = self.knowledge_base.observations[-5:]
        
        for i in range(num_hypotheses):
            # Generate hypothesis based on patterns in observations
            hypothesis = self._create_hypothesis_from_observations(recent_obs, i)
            self.knowledge_base.add_hypothesis(hypothesis)
            hypotheses.append(hypothesis)
            
            print(f"\n  Hypothesis {i+1}:")
            print(f"    {hypothesis.statement}")
            print(f"    Predicted: {hypothesis.predicted_outcome}")
        
        self.current_phase = ResearchPhase.HYPOTHESIS
        return hypotheses
    
    def _create_hypothesis_from_observations(
        self,
        observations: List[Observation],
        index: int
    ) -> Hypothesis:
        """Create a hypothesis from observations."""
        # Simplified hypothesis generation
        templates = [
            "Increasing variable X will increase variable Y",
            "Variable A is positively correlated with variable B",
            "The effect is mediated by variable Z",
            "There is a causal relationship between X and Y",
            "The phenomenon follows a linear pattern"
        ]
        
        variables = []
        if observations:
            # Extract variables from observations
            for obs in observations[:2]:
                variables.extend(obs.measurements.keys())
        
        variables = list(set(variables))[:3] if variables else ["X", "Y", "Z"]
        
        statement = templates[index % len(templates)]
        
        return Hypothesis(
            hypothesis_id=str(uuid.uuid4()),
            statement=statement,
            variables=variables,
            predicted_outcome=f"Expected change in {variables[0] if variables else 'outcome'}"
        )
    
    def design_experiment(
        self,
        hypothesis: Hypothesis,
        experiment_type: ExperimentType = ExperimentType.CONTROLLED
    ) -> Experiment:
        """Design an experiment to test a hypothesis."""
        print(f"\n[{self.name}] Designing {experiment_type.value} experiment...")
        
        # Extract variables from hypothesis
        independent_vars = {hypothesis.variables[0]: "manipulated"} if hypothesis.variables else {"X": "manipulated"}
        dependent_vars = hypothesis.variables[1:2] if len(hypothesis.variables) > 1 else ["Y"]
        control_vars = {"temperature": 25.0, "pressure": 1.0}
        
        # Design procedure
        procedure = [
            "Set up experimental apparatus",
            f"Control for: {', '.join(control_vars.keys())}",
            f"Manipulate independent variable: {list(independent_vars.keys())[0]}",
            f"Measure dependent variable: {dependent_vars[0]}",
            "Record results",
            "Repeat for statistical significance"
        ]
        
        experiment = Experiment(
            experiment_id=str(uuid.uuid4()),
            name=f"Test: {hypothesis.statement[:50]}",
            experiment_type=experiment_type,
            hypothesis_id=hypothesis.hypothesis_id,
            independent_variables=independent_vars,
            dependent_variables=dependent_vars,
            control_variables=control_vars,
            procedure=procedure,
            expected_results=hypothesis.predicted_outcome
        )
        
        self.knowledge_base.add_experiment(experiment)
        
        print(f"  Experiment: {experiment.name}")
        print(f"  Independent vars: {independent_vars}")
        print(f"  Dependent vars: {dependent_vars}")
        
        self.current_phase = ResearchPhase.EXPERIMENTATION
        return experiment
    
    def run_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """Execute an experiment."""
        print(f"\n[{self.name}] Running experiment: {experiment.name}")
        
        results = experiment.run()
        
        print(f"  Results: {results}")
        
        return results
    
    def analyze_results(
        self,
        experiment: Experiment,
        hypothesis: Hypothesis
    ) -> Dict[str, Any]:
        """Analyze experiment results."""
        print(f"\n[{self.name}] Analyzing results...")
        
        if not experiment.actual_results:
            return {"error": "No results to analyze"}
        
        # Simplified analysis
        analysis = {
            "experiment_id": experiment.experiment_id,
            "hypothesis_id": hypothesis.hypothesis_id,
            "results_summary": experiment.actual_results,
            "statistical_significance": random.choice([True, False]),
            "effect_size": random.uniform(0.1, 0.9)
        }
        
        # Update hypothesis based on results
        if analysis["statistical_significance"]:
            hypothesis.supporting_evidence.append(experiment.experiment_id)
            hypothesis.status = HypothesisStatus.SUPPORTED
            print(f"  ✓ Hypothesis SUPPORTED")
        else:
            hypothesis.contradicting_evidence.append(experiment.experiment_id)
            if len(hypothesis.contradicting_evidence) > 2:
                hypothesis.status = HypothesisStatus.REFUTED
                print(f"  ✗ Hypothesis REFUTED")
            else:
                hypothesis.status = HypothesisStatus.INCONCLUSIVE
                print(f"  ? Results INCONCLUSIVE")
        
        hypothesis.update_confidence()
        print(f"  Confidence: {hypothesis.confidence:.2f}")
        
        self.current_phase = ResearchPhase.ANALYSIS
        return analysis
    
    def draw_conclusions(
        self,
        hypotheses: List[Hypothesis]
    ) -> List[str]:
        """Draw scientific conclusions from tested hypotheses."""
        print(f"\n[{self.name}] Drawing conclusions...")
        
        conclusions = []
        
        for hypothesis in hypotheses:
            if hypothesis.status == HypothesisStatus.SUPPORTED:
                conclusion = f"Evidence supports that {hypothesis.statement.lower()}"
                conclusions.append(conclusion)
                
                # Consider promoting to theory
                if hypothesis.confidence >= 0.8:
                    if self.knowledge_base.promote_to_theory(hypothesis.hypothesis_id):
                        conclusions.append(f"Elevated to theory: {hypothesis.statement}")
            
            elif hypothesis.status == HypothesisStatus.REFUTED:
                conclusion = f"Evidence refutes that {hypothesis.statement.lower()}"
                conclusions.append(conclusion)
        
        print(f"\n  Found {len(conclusions)} conclusions:")
        for i, conc in enumerate(conclusions, 1):
            print(f"    {i}. {conc}")
        
        self.current_phase = ResearchPhase.CONCLUSION
        return conclusions
    
    def publish_findings(
        self,
        title: str,
        hypotheses: List[Hypothesis],
        experiments: List[Experiment],
        conclusions: List[str]
    ) -> ResearchPaper:
        """Publish research findings as a paper."""
        print(f"\n[{self.name}] Publishing research paper...")
        
        # Generate abstract
        abstract = self._generate_abstract(hypotheses, experiments, conclusions)
        
        # Calculate overall confidence
        avg_confidence = sum(h.confidence for h in hypotheses) / len(hypotheses) if hypotheses else 0.0
        
        paper = ResearchPaper(
            paper_id=str(uuid.uuid4()),
            title=title,
            abstract=abstract,
            hypotheses=[h.hypothesis_id for h in hypotheses],
            experiments=[e.experiment_id for e in experiments],
            conclusions=conclusions,
            confidence_score=avg_confidence
        )
        
        print(f"\n  Title: {paper.title}")
        print(f"  Confidence Score: {paper.confidence_score:.2f}")
        print(f"\n  Abstract:\n  {paper.abstract}")
        
        self.current_phase = ResearchPhase.PUBLICATION
        return paper
    
    def _generate_abstract(
        self,
        hypotheses: List[Hypothesis],
        experiments: List[Experiment],
        conclusions: List[str]
    ) -> str:
        """Generate paper abstract."""
        abstract_parts = []
        
        abstract_parts.append(
            f"This study investigated {len(hypotheses)} hypotheses in the field of {self.field}."
        )
        
        abstract_parts.append(
            f"A total of {len(experiments)} experiments were conducted using various methodologies."
        )
        
        supported = sum(1 for h in hypotheses if h.status == HypothesisStatus.SUPPORTED)
        abstract_parts.append(
            f"Results provided strong support for {supported} of {len(hypotheses)} hypotheses."
        )
        
        if conclusions:
            abstract_parts.append(
                f"Key finding: {conclusions[0]}"
            )
        
        return " ".join(abstract_parts)
    
    def peer_review(self, paper: ResearchPaper) -> Dict[str, Any]:
        """Simulate peer review of research."""
        print(f"\n[{self.name}] Conducting peer review...")
        
        # Simplified peer review
        review = {
            "methodology_score": random.uniform(0.6, 1.0),
            "significance_score": random.uniform(0.5, 1.0),
            "clarity_score": random.uniform(0.7, 1.0),
            "reproducibility_score": random.uniform(0.6, 0.9),
            "comments": [],
            "recommendation": ""
        }
        
        avg_score = sum([
            review["methodology_score"],
            review["significance_score"],
            review["clarity_score"],
            review["reproducibility_score"]
        ]) / 4
        
        if avg_score >= 0.8:
            review["recommendation"] = "Accept"
        elif avg_score >= 0.6:
            review["recommendation"] = "Minor Revisions"
        else:
            review["recommendation"] = "Major Revisions"
        
        review["comments"].append(f"Overall scientific rigor: {avg_score:.2f}")
        review["comments"].append(f"Confidence in findings: {paper.confidence_score:.2f}")
        
        print(f"  Recommendation: {review['recommendation']}")
        print(f"  Overall Score: {avg_score:.2f}")
        
        return review


def demonstrate_scientific_discovery():
    """Demonstrate scientific discovery agent pattern."""
    print("=" * 60)
    print("SCIENTIFIC DISCOVERY AGENT DEMONSTRATION")
    print("=" * 60)
    
    # Create scientific discovery agent
    scientist = ScientificDiscoveryAgent(
        "scientist1",
        "Dr. AI",
        "Cognitive Science"
    )
    
    # Phase 1: Observations
    print("\n" + "=" * 60)
    print("1. Making Observations")
    print("=" * 60)
    
    obs1 = scientist.make_observation(
        "Memory recall improves with repetition",
        {"repetitions": 5.0, "recall_accuracy": 0.75}
    )
    
    obs2 = scientist.make_observation(
        "Spaced repetition shows stronger effects",
        {"repetitions": 5.0, "recall_accuracy": 0.85, "spacing_hours": 24.0}
    )
    
    obs3 = scientist.make_observation(
        "Active recall outperforms passive review",
        {"active_recall": 1.0, "recall_accuracy": 0.80}
    )
    
    # Phase 2: Hypothesis Generation
    print("\n" + "=" * 60)
    print("2. Generating Hypotheses")
    print("=" * 60)
    
    hypotheses = scientist.generate_hypotheses(num_hypotheses=2)
    
    # Phase 3: Experiment Design and Execution
    print("\n" + "=" * 60)
    print("3. Designing and Running Experiments")
    print("=" * 60)
    
    experiments = []
    analyses = []
    
    for hypothesis in hypotheses:
        # Design experiment
        experiment = scientist.design_experiment(
            hypothesis,
            ExperimentType.CONTROLLED
        )
        experiments.append(experiment)
        
        # Run experiment
        scientist.run_experiment(experiment)
        
        # Analyze results
        analysis = scientist.analyze_results(experiment, hypothesis)
        analyses.append(analysis)
    
    # Phase 4: Draw Conclusions
    print("\n" + "=" * 60)
    print("4. Drawing Conclusions")
    print("=" * 60)
    
    conclusions = scientist.draw_conclusions(hypotheses)
    
    # Phase 5: Publish Findings
    print("\n" + "=" * 60)
    print("5. Publishing Research")
    print("=" * 60)
    
    paper = scientist.publish_findings(
        title="The Effects of Repetition and Spacing on Memory Recall",
        hypotheses=hypotheses,
        experiments=experiments,
        conclusions=conclusions
    )
    
    # Phase 6: Peer Review
    print("\n" + "=" * 60)
    print("6. Peer Review")
    print("=" * 60)
    
    review = scientist.peer_review(paper)
    
    # Summary
    print("\n" + "=" * 60)
    print("7. Research Summary")
    print("=" * 60)
    
    print(f"\nField: {scientist.field}")
    print(f"Observations made: {len(scientist.knowledge_base.observations)}")
    print(f"Hypotheses tested: {len(hypotheses)}")
    print(f"Experiments conducted: {len(experiments)}")
    print(f"Theories established: {len(scientist.knowledge_base.theories)}")
    print(f"Papers published: 1")
    print(f"Review outcome: {review['recommendation']}")


if __name__ == "__main__":
    demonstrate_scientific_discovery()
