"""
Agent Coalition Formation Pattern

Agents form coalitions to achieve shared goals.
Implements coalition formation algorithms and payoff distribution.

Use Cases:
- Resource pooling
- Task collaboration
- Power aggregation
- Collective bargaining

Advantages:
- Increased capabilities
- Risk sharing
- Resource efficiency
- Collective strength
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import random
from itertools import combinations


class CoalitionFormationAlgorithm(Enum):
    """Coalition formation algorithms"""
    GREEDY = "greedy"
    CORE_BASED = "core_based"
    SHAPLEY_VALUE = "shapley_value"
    KERNEL_BASED = "kernel_based"


class PayoffDistribution(Enum):
    """Methods for distributing coalition payoff"""
    EQUAL_SPLIT = "equal_split"
    PROPORTIONAL = "proportional"
    SHAPLEY = "shapley"
    NUCLEOLUS = "nucleolus"


@dataclass
class CoalitionAgent:
    """Agent capable of joining coalitions"""
    agent_id: str
    capabilities: Dict[str, float]  # capability -> level
    resources: Dict[str, float]  # resource -> amount
    preferences: Dict[str, Any]
    current_coalitions: Set[str] = field(default_factory=set)
    reputation: float = 1.0


@dataclass
class Task:
    """Task requiring coalition"""
    task_id: str
    name: str
    required_capabilities: Dict[str, float]
    required_resources: Dict[str, float]
    reward: float
    deadline: Optional[datetime] = None
    complexity: float = 1.0


@dataclass
class Coalition:
    """Coalition of agents"""
    coalition_id: str
    members: Set[str]
    tasks: List[str]
    formed_at: datetime
    total_capability: Dict[str, float] = field(default_factory=dict)
    total_resources: Dict[str, float] = field(default_factory=dict)
    total_reward: float = 0.0
    payoff_distribution: Dict[str, float] = field(default_factory=dict)
    stability_score: float = 0.0


@dataclass
class CoalitionProposal:
    """Proposal to form coalition"""
    proposal_id: str
    proposer: str
    proposed_members: Set[str]
    target_tasks: List[str]
    expected_reward: float
    proposed_distribution: Dict[str, float]
    timestamp: datetime
    accepted_by: Set[str] = field(default_factory=set)


class CoalitionValueCalculator:
    """Calculates value of coalitions"""
    
    def calculate_coalition_value(self,
                                  agents: List[CoalitionAgent],
                                  tasks: List[Task]) -> float:
        """
        Calculate total value coalition can generate.
        
        Args:
            agents: Coalition members
            tasks: Available tasks
            
        Returns:
            Total value
        """
        # Sum capabilities and resources
        total_capabilities = {}
        total_resources = {}
        
        for agent in agents:
            for cap, level in agent.capabilities.items():
                total_capabilities[cap] = total_capabilities.get(cap, 0) + level
            
            for res, amount in agent.resources.items():
                total_resources[res] = total_resources.get(res, 0) + amount
        
        # Calculate which tasks can be completed
        total_value = 0.0
        
        for task in tasks:
            can_complete = True
            
            # Check capabilities
            for req_cap, req_level in task.required_capabilities.items():
                if total_capabilities.get(req_cap, 0) < req_level:
                    can_complete = False
                    break
            
            # Check resources
            if can_complete:
                for req_res, req_amount in task.required_resources.items():
                    if total_resources.get(req_res, 0) < req_amount:
                        can_complete = False
                        break
            
            if can_complete:
                total_value += task.reward
        
        return total_value
    
    def calculate_shapley_value(self,
                               agent_id: str,
                               all_agents: List[CoalitionAgent],
                               tasks: List[Task]) -> float:
        """
        Calculate Shapley value for an agent.
        
        Args:
            agent_id: Agent to calculate for
            all_agents: All agents
            tasks: Available tasks
            
        Returns:
            Shapley value
        """
        agent = next((a for a in all_agents if a.agent_id == agent_id), None)
        if not agent:
            return 0.0
        
        other_agents = [a for a in all_agents if a.agent_id != agent_id]
        
        shapley = 0.0
        
        # Calculate marginal contribution for all possible coalitions
        for r in range(len(other_agents) + 1):
            for coalition_members in combinations(other_agents, r):
                coalition_list = list(coalition_members)
                
                # Value without agent
                value_without = self.calculate_coalition_value(
                    coalition_list,
                    tasks
                )
                
                # Value with agent
                value_with = self.calculate_coalition_value(
                    coalition_list + [agent],
                    tasks
                )
                
                marginal = value_with - value_without
                
                # Weight by probability
                n = len(all_agents)
                s = len(coalition_list)
                weight = (
                    (1.0 / n) *
                    (1.0 / self._binomial(n - 1, s))
                )
                
                shapley += weight * marginal
        
        return shapley
    
    def _binomial(self, n: int, k: int) -> int:
        """Calculate binomial coefficient"""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        
        result = 1
        for i in range(min(k, n - k)):
            result = result * (n - i) // (i + 1)
        
        return result


class CoalitionStabilityAnalyzer:
    """Analyzes coalition stability"""
    
    def __init__(self, value_calculator: CoalitionValueCalculator):
        self.value_calculator = value_calculator
    
    def is_stable(self,
                 coalition: Coalition,
                 all_agents: Dict[str, CoalitionAgent],
                 tasks: List[Task]) -> Tuple[bool, float]:
        """
        Check if coalition is stable.
        
        Args:
            coalition: Coalition to check
            all_agents: All available agents
            tasks: Available tasks
            
        Returns:
            (is_stable, stability_score) tuple
        """
        coalition_agents = [
            all_agents[aid] for aid in coalition.members
            if aid in all_agents
        ]
        
        # Calculate coalition value
        coalition_value = self.value_calculator.calculate_coalition_value(
            coalition_agents,
            tasks
        )
        
        # Check if any subset can do better
        max_defection_gain = 0.0
        
        for subset_size in range(1, len(coalition.members)):
            for subset in combinations(coalition.members, subset_size):
                subset_agents = [
                    all_agents[aid] for aid in subset
                    if aid in all_agents
                ]
                
                subset_value = self.value_calculator.calculate_coalition_value(
                    subset_agents,
                    tasks
                )
                
                # Current payoff for subset
                current_payoff = sum(
                    coalition.payoff_distribution.get(aid, 0)
                    for aid in subset
                )
                
                # Potential gain from defection
                defection_gain = subset_value - current_payoff
                
                if defection_gain > max_defection_gain:
                    max_defection_gain = defection_gain
        
        # Stability score (higher is more stable)
        stability_score = 1.0 / (1.0 + max_defection_gain)
        
        return max_defection_gain <= 0, stability_score


class CoalitionFormationEngine:
    """Forms coalitions of agents"""
    
    def __init__(self,
                 algorithm: CoalitionFormationAlgorithm,
                 value_calculator: CoalitionValueCalculator):
        self.algorithm = algorithm
        self.value_calculator = value_calculator
    
    def form_coalitions(self,
                       agents: List[CoalitionAgent],
                       tasks: List[Task]) -> List[Coalition]:
        """
        Form coalitions for tasks.
        
        Args:
            agents: Available agents
            tasks: Tasks to complete
            
        Returns:
            List of formed coalitions
        """
        if self.algorithm == CoalitionFormationAlgorithm.GREEDY:
            return self._greedy_formation(agents, tasks)
        elif self.algorithm == CoalitionFormationAlgorithm.SHAPLEY_VALUE:
            return self._shapley_formation(agents, tasks)
        else:
            return self._greedy_formation(agents, tasks)
    
    def _greedy_formation(self,
                         agents: List[CoalitionAgent],
                         tasks: List[Task]) -> List[Coalition]:
        """Greedy coalition formation"""
        coalitions = []
        available_agents = set(a.agent_id for a in agents)
        remaining_tasks = tasks.copy()
        
        for task in remaining_tasks:
            # Find minimal coalition for task
            best_coalition = None
            best_size = len(agents) + 1
            
            # Try all possible coalitions
            for size in range(1, len(available_agents) + 1):
                for agent_combo in combinations(available_agents, size):
                    combo_agents = [
                        a for a in agents
                        if a.agent_id in agent_combo
                    ]
                    
                    # Check if can complete task
                    value = self.value_calculator.calculate_coalition_value(
                        combo_agents,
                        [task]
                    )
                    
                    if value >= task.reward and size < best_size:
                        best_coalition = agent_combo
                        best_size = size
                
                if best_coalition:
                    break
            
            if best_coalition:
                # Create coalition
                coalition_agents = [
                    a for a in agents
                    if a.agent_id in best_coalition
                ]
                
                coalition = Coalition(
                    coalition_id="coalition_{}".format(len(coalitions)),
                    members=set(best_coalition),
                    tasks=[task.task_id],
                    formed_at=datetime.now(),
                    total_reward=task.reward
                )
                
                # Calculate capabilities and resources
                for agent in coalition_agents:
                    for cap, level in agent.capabilities.items():
                        coalition.total_capability[cap] = (
                            coalition.total_capability.get(cap, 0) + level
                        )
                    for res, amount in agent.resources.items():
                        coalition.total_resources[res] = (
                            coalition.total_resources.get(res, 0) + amount
                        )
                
                coalitions.append(coalition)
                
                # Remove agents from available pool
                available_agents -= set(best_coalition)
        
        return coalitions
    
    def _shapley_formation(self,
                          agents: List[CoalitionAgent],
                          tasks: List[Task]) -> List[Coalition]:
        """Formation based on Shapley values"""
        # Form grand coalition
        coalition = Coalition(
            coalition_id="grand_coalition",
            members=set(a.agent_id for a in agents),
            tasks=[t.task_id for t in tasks],
            formed_at=datetime.now()
        )
        
        # Calculate total value
        total_value = self.value_calculator.calculate_coalition_value(
            agents,
            tasks
        )
        
        coalition.total_reward = total_value
        
        # Calculate Shapley values for distribution
        for agent in agents:
            shapley = self.value_calculator.calculate_shapley_value(
                agent.agent_id,
                agents,
                tasks
            )
            coalition.payoff_distribution[agent.agent_id] = shapley
        
        return [coalition]


class AgentCoalitionSystem:
    """
    System managing agent coalitions.
    Facilitates coalition formation, payoff distribution, and stability.
    """
    
    def __init__(self,
                 formation_algorithm: CoalitionFormationAlgorithm = CoalitionFormationAlgorithm.GREEDY,
                 payoff_method: PayoffDistribution = PayoffDistribution.SHAPLEY):
        self.formation_algorithm = formation_algorithm
        self.payoff_method = payoff_method
        
        # Components
        self.value_calculator = CoalitionValueCalculator()
        self.stability_analyzer = CoalitionStabilityAnalyzer(
            self.value_calculator
        )
        self.formation_engine = CoalitionFormationEngine(
            formation_algorithm,
            self.value_calculator
        )
        
        # State
        self.agents: Dict[str, CoalitionAgent] = {}
        self.tasks: Dict[str, Task] = {}
        self.coalitions: Dict[str, Coalition] = {}
        self.proposals: List[CoalitionProposal] = []
    
    def register_agent(self,
                      agent_id: str,
                      capabilities: Dict[str, float],
                      resources: Dict[str, float],
                      preferences: Optional[Dict[str, Any]] = None) -> None:
        """
        Register agent in system.
        
        Args:
            agent_id: Agent identifier
            capabilities: Agent capabilities
            resources: Agent resources
            preferences: Agent preferences
        """
        if preferences is None:
            preferences = {}
        
        agent = CoalitionAgent(
            agent_id=agent_id,
            capabilities=capabilities,
            resources=resources,
            preferences=preferences
        )
        
        self.agents[agent_id] = agent
    
    def add_task(self,
                task_id: str,
                name: str,
                required_capabilities: Dict[str, float],
                required_resources: Dict[str, float],
                reward: float,
                deadline: Optional[datetime] = None) -> None:
        """
        Add task requiring coalition.
        
        Args:
            task_id: Task identifier
            name: Task name
            required_capabilities: Required capabilities
            required_resources: Required resources
            reward: Task reward
            deadline: Optional deadline
        """
        task = Task(
            task_id=task_id,
            name=name,
            required_capabilities=required_capabilities,
            required_resources=required_resources,
            reward=reward,
            deadline=deadline
        )
        
        self.tasks[task_id] = task
    
    def form_coalitions_for_tasks(self) -> List[str]:
        """
        Form coalitions for available tasks.
        
        Returns:
            List of coalition IDs
        """
        available_agents = [
            agent for agent in self.agents.values()
            if not agent.current_coalitions
        ]
        
        available_tasks = list(self.tasks.values())
        
        # Form coalitions
        new_coalitions = self.formation_engine.form_coalitions(
            available_agents,
            available_tasks
        )
        
        coalition_ids = []
        
        for coalition in new_coalitions:
            # Distribute payoff
            self._distribute_payoff(coalition, available_agents)
            
            # Analyze stability
            is_stable, stability = self.stability_analyzer.is_stable(
                coalition,
                self.agents,
                available_tasks
            )
            coalition.stability_score = stability
            
            # Register coalition
            self.coalitions[coalition.coalition_id] = coalition
            coalition_ids.append(coalition.coalition_id)
            
            # Update agent memberships
            for agent_id in coalition.members:
                if agent_id in self.agents:
                    self.agents[agent_id].current_coalitions.add(
                        coalition.coalition_id
                    )
        
        return coalition_ids
    
    def propose_coalition(self,
                         proposer_id: str,
                         member_ids: List[str],
                         task_ids: List[str]) -> str:
        """
        Propose formation of coalition.
        
        Args:
            proposer_id: Agent proposing
            member_ids: Proposed members
            task_ids: Target tasks
            
        Returns:
            Proposal ID
        """
        # Calculate expected value
        proposed_agents = [
            self.agents[aid] for aid in member_ids
            if aid in self.agents
        ]
        
        proposed_tasks = [
            self.tasks[tid] for tid in task_ids
            if tid in self.tasks
        ]
        
        expected_reward = self.value_calculator.calculate_coalition_value(
            proposed_agents,
            proposed_tasks
        )
        
        # Calculate proposed distribution
        distribution = {}
        if self.payoff_method == PayoffDistribution.EQUAL_SPLIT:
            share = expected_reward / len(member_ids)
            distribution = {aid: share for aid in member_ids}
        else:
            # Use Shapley values
            for agent_id in member_ids:
                shapley = self.value_calculator.calculate_shapley_value(
                    agent_id,
                    proposed_agents,
                    proposed_tasks
                )
                distribution[agent_id] = shapley
        
        proposal = CoalitionProposal(
            proposal_id="proposal_{}".format(len(self.proposals)),
            proposer=proposer_id,
            proposed_members=set(member_ids),
            target_tasks=task_ids,
            expected_reward=expected_reward,
            proposed_distribution=distribution,
            timestamp=datetime.now()
        )
        
        self.proposals.append(proposal)
        
        return proposal.proposal_id
    
    def _distribute_payoff(self,
                          coalition: Coalition,
                          all_agents: List[CoalitionAgent]) -> None:
        """Distribute coalition payoff among members"""
        if self.payoff_method == PayoffDistribution.EQUAL_SPLIT:
            share = coalition.total_reward / len(coalition.members)
            coalition.payoff_distribution = {
                aid: share for aid in coalition.members
            }
        
        elif self.payoff_method == PayoffDistribution.SHAPLEY:
            tasks = [
                self.tasks[tid] for tid in coalition.tasks
                if tid in self.tasks
            ]
            
            for agent_id in coalition.members:
                shapley = self.value_calculator.calculate_shapley_value(
                    agent_id,
                    all_agents,
                    tasks
                )
                coalition.payoff_distribution[agent_id] = shapley
        
        else:  # PROPORTIONAL
            total_capability = sum(
                sum(self.agents[aid].capabilities.values())
                for aid in coalition.members
                if aid in self.agents
            )
            
            for agent_id in coalition.members:
                if agent_id in self.agents:
                    agent_capability = sum(
                        self.agents[agent_id].capabilities.values()
                    )
                    share = (
                        coalition.total_reward *
                        agent_capability / total_capability
                    )
                    coalition.payoff_distribution[agent_id] = share
    
    def get_coalition_info(self, coalition_id: str) -> Dict[str, Any]:
        """Get information about coalition"""
        coalition = self.coalitions.get(coalition_id)
        if not coalition:
            return {}
        
        return {
            "coalition_id": coalition_id,
            "members": list(coalition.members),
            "num_members": len(coalition.members),
            "tasks": coalition.tasks,
            "total_reward": coalition.total_reward,
            "payoff_distribution": coalition.payoff_distribution,
            "stability_score": coalition.stability_score,
            "formed_at": coalition.formed_at.isoformat()
        }
    
    def get_agent_coalitions(self, agent_id: str) -> List[str]:
        """Get coalitions agent belongs to"""
        agent = self.agents.get(agent_id)
        if not agent:
            return []
        
        return list(agent.current_coalitions)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_agents = len(self.agents)
        agents_in_coalitions = sum(
            1 for a in self.agents.values()
            if a.current_coalitions
        )
        
        total_value = sum(c.total_reward for c in self.coalitions.values())
        
        avg_coalition_size = (
            sum(len(c.members) for c in self.coalitions.values()) /
            len(self.coalitions)
        ) if self.coalitions else 0
        
        avg_stability = (
            sum(c.stability_score for c in self.coalitions.values()) /
            len(self.coalitions)
        ) if self.coalitions else 0
        
        return {
            "total_agents": total_agents,
            "agents_in_coalitions": agents_in_coalitions,
            "total_coalitions": len(self.coalitions),
            "total_tasks": len(self.tasks),
            "total_value_generated": total_value,
            "avg_coalition_size": avg_coalition_size,
            "avg_stability_score": avg_stability,
            "formation_algorithm": self.formation_algorithm.value,
            "payoff_method": self.payoff_method.value
        }


def demonstrate_agent_coalition():
    """Demonstrate agent coalition system"""
    print("=" * 70)
    print("Agent Coalition Formation Demonstration")
    print("=" * 70)
    
    system = AgentCoalitionSystem(
        formation_algorithm=CoalitionFormationAlgorithm.GREEDY,
        payoff_method=PayoffDistribution.SHAPLEY
    )
    
    # Example 1: Register agents with capabilities
    print("\n1. Registering Agents:")
    
    agents_data = [
        ("agent_1", {"coding": 0.8, "design": 0.3}, {"time": 40, "budget": 5000}),
        ("agent_2", {"coding": 0.5, "design": 0.9}, {"time": 30, "budget": 3000}),
        ("agent_3", {"coding": 0.9, "testing": 0.7}, {"time": 35, "budget": 4000}),
        ("agent_4", {"design": 0.6, "testing": 0.8}, {"time": 25, "budget": 2000}),
        ("agent_5", {"coding": 0.7, "design": 0.5, "testing": 0.6}, {"time": 45, "budget": 6000})
    ]
    
    for agent_id, capabilities, resources in agents_data:
        system.register_agent(agent_id, capabilities, resources)
        print("  {}: {} capabilities, {} resources".format(
            agent_id,
            len(capabilities),
            len(resources)
        ))
    
    # Example 2: Add tasks
    print("\n2. Adding Tasks:")
    
    tasks_data = [
        ("task_1", "Web App", {"coding": 1.0, "design": 0.5}, {"time": 50, "budget": 8000}, 15000),
        ("task_2", "Mobile App", {"coding": 0.8, "design": 0.9, "testing": 0.5}, {"time": 60, "budget": 10000}, 20000),
        ("task_3", "Testing Suite", {"testing": 1.0}, {"time": 30, "budget": 3000}, 8000)
    ]
    
    for task_id, name, req_cap, req_res, reward in tasks_data:
        system.add_task(task_id, name, req_cap, req_res, reward)
        print("  {}: Reward ${:,.0f}".format(name, reward))
    
    # Example 3: Form coalitions
    print("\n3. Forming Coalitions:")
    
    coalition_ids = system.form_coalitions_for_tasks()
    
    print("  Formed {} coalition(s)".format(len(coalition_ids)))
    
    for coalition_id in coalition_ids:
        info = system.get_coalition_info(coalition_id)
        print("\n  Coalition {}:".format(coalition_id))
        print("    Members: {}".format(", ".join(info["members"])))
        print("    Tasks: {}".format(", ".join(info["tasks"])))
        print("    Total reward: ${:,.0f}".format(info["total_reward"]))
        print("    Stability: {:.2%}".format(info["stability_score"]))
        print("    Payoff distribution:")
        for agent_id, payoff in info["payoff_distribution"].items():
            print("      {}: ${:,.2f}".format(agent_id, payoff))
    
    # Example 4: Coalition proposal
    print("\n4. Proposing New Coalition:")
    
    proposal_id = system.propose_coalition(
        proposer_id="agent_1",
        member_ids=["agent_1", "agent_2", "agent_3"],
        task_ids=["task_1"]
    )
    
    proposal = system.proposals[-1]
    print("  Proposal ID: {}".format(proposal_id))
    print("  Proposer: {}".format(proposal.proposer))
    print("  Proposed members: {}".format(", ".join(proposal.proposed_members)))
    print("  Expected reward: ${:,.0f}".format(proposal.expected_reward))
    
    # Example 5: Agent coalition membership
    print("\n5. Agent Coalition Memberships:")
    
    for agent_id in ["agent_1", "agent_2", "agent_3"]:
        coalitions = system.get_agent_coalitions(agent_id)
        print("  {}: {} coalition(s)".format(agent_id, len(coalitions)))
    
    # Example 6: System statistics
    print("\n6. System Statistics:")
    stats = system.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Example 7: Compare formation algorithms
    print("\n7. Comparing Formation Algorithms:")
    
    algorithms = [
        CoalitionFormationAlgorithm.GREEDY,
        CoalitionFormationAlgorithm.SHAPLEY_VALUE
    ]
    
    for algo in algorithms:
        test_system = AgentCoalitionSystem(
            formation_algorithm=algo,
            payoff_method=PayoffDistribution.SHAPLEY
        )
        
        # Register same agents and tasks
        for agent_id, capabilities, resources in agents_data[:3]:
            test_system.register_agent(agent_id, capabilities, resources)
        
        for task_id, name, req_cap, req_res, reward in tasks_data[:2]:
            test_system.add_task(task_id, name, req_cap, req_res, reward)
        
        # Form coalitions
        test_coalitions = test_system.form_coalitions_for_tasks()
        test_stats = test_system.get_statistics()
        
        print("\n  Algorithm: {}".format(algo.value))
        print("    Coalitions formed: {}".format(test_stats["total_coalitions"]))
        print("    Total value: ${:,.0f}".format(test_stats["total_value_generated"]))
        print("    Avg stability: {:.2%}".format(test_stats["avg_stability_score"]))


if __name__ == "__main__":
    demonstrate_agent_coalition()
