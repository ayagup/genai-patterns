"""
Agentic AI Design Pattern: Resource-Constrained Planning

This pattern implements planning algorithms that operate under resource constraints
such as budget limits, capacity restrictions, and inventory management.

Key Concepts:
1. Resource Types: Budget, capacity, inventory, personnel
2. Constraint Satisfaction: Meeting limits while achieving goals
3. Optimization: Maximizing utility under constraints
4. Allocation: Distributing limited resources efficiently
5. Feasibility Checking: Validating plans against constraints

Use Cases:
- Project management with budget constraints
- Manufacturing with limited capacity
- Supply chain planning
- Resource allocation in organizations
- Scheduling with limited resources
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import uuid
from datetime import datetime, timedelta


class ResourceType(Enum):
    """Types of resources that can be constrained"""
    BUDGET = "budget"
    CAPACITY = "capacity"
    INVENTORY = "inventory"
    PERSONNEL = "personnel"
    TIME = "time"
    EQUIPMENT = "equipment"


class PlanningStrategy(Enum):
    """Strategy for resource-constrained planning"""
    GREEDY = "greedy"  # Allocate to highest priority first
    OPTIMAL = "optimal"  # Find optimal allocation
    HEURISTIC = "heuristic"  # Use heuristic rules
    BALANCED = "balanced"  # Balance across constraints


@dataclass
class Resource:
    """Represents a constrained resource"""
    name: str
    resource_type: ResourceType
    total_capacity: float
    available: float
    unit_cost: float = 0.0
    
    def allocate(self, amount: float) -> bool:
        """Allocate resource if available"""
        if amount <= self.available:
            self.available -= amount
            return True
        return False
    
    def release(self, amount: float) -> None:
        """Release allocated resource"""
        self.available = min(self.available + amount, self.total_capacity)
    
    def utilization(self) -> float:
        """Calculate resource utilization percentage"""
        if self.total_capacity == 0:
            return 0.0
        return ((self.total_capacity - self.available) / self.total_capacity) * 100


@dataclass
class ResourcePool:
    """Manages a collection of resources"""
    resources: Dict[str, Resource] = field(default_factory=dict)
    
    def add_resource(self, resource: Resource) -> None:
        """Add resource to pool"""
        self.resources[resource.name] = resource
    
    def check_availability(self, requirements: Dict[str, float]) -> bool:
        """Check if required resources are available"""
        for resource_name, amount in requirements.items():
            if resource_name not in self.resources:
                return False
            if self.resources[resource_name].available < amount:
                return False
        return True
    
    def allocate_resources(self, requirements: Dict[str, float]) -> bool:
        """Allocate multiple resources atomically"""
        # Check all resources first
        if not self.check_availability(requirements):
            return False
        
        # Allocate all resources
        allocated = []
        try:
            for resource_name, amount in requirements.items():
                if self.resources[resource_name].allocate(amount):
                    allocated.append((resource_name, amount))
                else:
                    # Rollback on failure
                    for name, amt in allocated:
                        self.resources[name].release(amt)
                    return False
            return True
        except Exception:
            # Rollback on error
            for name, amt in allocated:
                self.resources[name].release(amt)
            return False
    
    def release_resources(self, requirements: Dict[str, float]) -> None:
        """Release multiple resources"""
        for resource_name, amount in requirements.items():
            if resource_name in self.resources:
                self.resources[resource_name].release(amount)
    
    def total_cost(self, requirements: Dict[str, float]) -> float:
        """Calculate total cost of resource requirements"""
        cost = 0.0
        for resource_name, amount in requirements.items():
            if resource_name in self.resources:
                cost += self.resources[resource_name].unit_cost * amount
        return cost
    
    def get_utilization_report(self) -> Dict[str, float]:
        """Get utilization report for all resources"""
        return {name: resource.utilization() 
                for name, resource in self.resources.items()}


@dataclass
class ConstrainedTask:
    """Task with resource requirements and constraints"""
    task_id: str
    name: str
    priority: int  # Higher number = higher priority
    resource_requirements: Dict[str, float]
    duration: timedelta
    value: float = 0.0  # Value/benefit of completing task
    dependencies: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


@dataclass
class ResourceAllocation:
    """Represents allocation of resources to a task"""
    task_id: str
    task_name: str
    allocated_resources: Dict[str, float]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    cost: float = 0.0
    feasible: bool = True
    
    def __str__(self) -> str:
        return (f"Allocation(task={self.task_name}, "
                f"resources={self.allocated_resources}, "
                f"cost=${self.cost:.2f}, feasible={self.feasible})")


@dataclass
class ResourcePlan:
    """Complete plan with resource allocations"""
    allocations: List[ResourceAllocation] = field(default_factory=list)
    total_cost: float = 0.0
    total_value: float = 0.0
    feasible: bool = True
    unallocated_tasks: List[str] = field(default_factory=list)
    
    def add_allocation(self, allocation: ResourceAllocation) -> None:
        """Add allocation to plan"""
        self.allocations.append(allocation)
        self.total_cost += allocation.cost
    
    def efficiency_ratio(self) -> float:
        """Calculate value-to-cost ratio"""
        if self.total_cost == 0:
            return float('inf') if self.total_value > 0 else 0.0
        return self.total_value / self.total_cost
    
    def __str__(self) -> str:
        return (f"ResourcePlan(tasks={len(self.allocations)}, "
                f"cost=${self.total_cost:.2f}, value={self.total_value:.2f}, "
                f"efficiency={self.efficiency_ratio():.2f}, feasible={self.feasible})")


class ResourceConstrainedPlanner:
    """Plans task execution under resource constraints"""
    
    def __init__(self, resource_pool: ResourcePool, strategy: PlanningStrategy):
        self.resource_pool = resource_pool
        self.strategy = strategy
    
    def create_plan(self, tasks: List[ConstrainedTask]) -> ResourcePlan:
        """Create plan based on strategy"""
        if self.strategy == PlanningStrategy.GREEDY:
            return self._greedy_plan(tasks)
        elif self.strategy == PlanningStrategy.OPTIMAL:
            return self._optimal_plan(tasks)
        elif self.strategy == PlanningStrategy.HEURISTIC:
            return self._heuristic_plan(tasks)
        elif self.strategy == PlanningStrategy.BALANCED:
            return self._balanced_plan(tasks)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _greedy_plan(self, tasks: List[ConstrainedTask]) -> ResourcePlan:
        """Greedy allocation by priority"""
        plan = ResourcePlan()
        
        # Sort tasks by priority (descending)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        for task in sorted_tasks:
            # Check resource availability
            if self.resource_pool.check_availability(task.resource_requirements):
                # Allocate resources
                if self.resource_pool.allocate_resources(task.resource_requirements):
                    cost = self.resource_pool.total_cost(task.resource_requirements)
                    allocation = ResourceAllocation(
                        task_id=task.task_id,
                        task_name=task.name,
                        allocated_resources=task.resource_requirements.copy(),
                        cost=cost,
                        feasible=True
                    )
                    plan.add_allocation(allocation)
                    plan.total_value += task.value
                else:
                    plan.unallocated_tasks.append(task.task_id)
            else:
                plan.unallocated_tasks.append(task.task_id)
        
        plan.feasible = len(plan.unallocated_tasks) == 0
        return plan
    
    def _optimal_plan(self, tasks: List[ConstrainedTask]) -> ResourcePlan:
        """Find optimal allocation (simplified knapsack approach)"""
        plan = ResourcePlan()
        
        # Sort by value-to-cost ratio (descending)
        tasks_with_cost = []
        for task in tasks:
            cost = self.resource_pool.total_cost(task.resource_requirements)
            ratio = task.value / cost if cost > 0 else float('inf')
            tasks_with_cost.append((task, cost, ratio))
        
        sorted_tasks = sorted(tasks_with_cost, key=lambda x: x[2], reverse=True)
        
        for task, cost, ratio in sorted_tasks:
            if self.resource_pool.check_availability(task.resource_requirements):
                if self.resource_pool.allocate_resources(task.resource_requirements):
                    allocation = ResourceAllocation(
                        task_id=task.task_id,
                        task_name=task.name,
                        allocated_resources=task.resource_requirements.copy(),
                        cost=cost,
                        feasible=True
                    )
                    plan.add_allocation(allocation)
                    plan.total_value += task.value
                else:
                    plan.unallocated_tasks.append(task.task_id)
            else:
                plan.unallocated_tasks.append(task.task_id)
        
        plan.feasible = len(plan.unallocated_tasks) == 0
        return plan
    
    def _heuristic_plan(self, tasks: List[ConstrainedTask]) -> ResourcePlan:
        """Use heuristic rules for allocation"""
        plan = ResourcePlan()
        
        # Heuristic: prioritize by deadline and priority
        def heuristic_score(task: ConstrainedTask) -> float:
            score = task.priority * 100
            if task.deadline:
                # Urgency factor (sooner deadline = higher score)
                hours_until_deadline = (task.deadline - datetime.now()).total_seconds() / 3600
                if hours_until_deadline > 0:
                    score += 1000 / hours_until_deadline
            return score
        
        sorted_tasks = sorted(tasks, key=heuristic_score, reverse=True)
        
        for task in sorted_tasks:
            if self.resource_pool.check_availability(task.resource_requirements):
                if self.resource_pool.allocate_resources(task.resource_requirements):
                    cost = self.resource_pool.total_cost(task.resource_requirements)
                    allocation = ResourceAllocation(
                        task_id=task.task_id,
                        task_name=task.name,
                        allocated_resources=task.resource_requirements.copy(),
                        cost=cost,
                        feasible=True
                    )
                    plan.add_allocation(allocation)
                    plan.total_value += task.value
                else:
                    plan.unallocated_tasks.append(task.task_id)
            else:
                plan.unallocated_tasks.append(task.task_id)
        
        plan.feasible = len(plan.unallocated_tasks) == 0
        return plan
    
    def _balanced_plan(self, tasks: List[ConstrainedTask]) -> ResourcePlan:
        """Balance across multiple criteria"""
        plan = ResourcePlan()
        
        # Balance: priority, value, resource efficiency
        def balanced_score(task: ConstrainedTask) -> float:
            cost = self.resource_pool.total_cost(task.resource_requirements)
            value_score = task.value / max(cost, 1.0)
            priority_score = task.priority
            # Calculate resource diversity (prefer tasks using different resources)
            resource_diversity = len(task.resource_requirements)
            
            return (value_score * 0.4 + priority_score * 0.4 + 
                    resource_diversity * 0.2)
        
        sorted_tasks = sorted(tasks, key=balanced_score, reverse=True)
        
        for task in sorted_tasks:
            if self.resource_pool.check_availability(task.resource_requirements):
                if self.resource_pool.allocate_resources(task.resource_requirements):
                    cost = self.resource_pool.total_cost(task.resource_requirements)
                    allocation = ResourceAllocation(
                        task_id=task.task_id,
                        task_name=task.name,
                        allocated_resources=task.resource_requirements.copy(),
                        cost=cost,
                        feasible=True
                    )
                    plan.add_allocation(allocation)
                    plan.total_value += task.value
                else:
                    plan.unallocated_tasks.append(task.task_id)
            else:
                plan.unallocated_tasks.append(task.task_id)
        
        plan.feasible = len(plan.unallocated_tasks) == 0
        return plan
    
    def validate_plan(self, plan: ResourcePlan) -> Tuple[bool, List[str]]:
        """Validate plan against constraints"""
        issues = []
        
        # Check if plan is feasible
        if not plan.feasible:
            issues.append("Plan has unallocated tasks")
        
        # Check resource allocations
        resource_usage = {name: 0.0 for name in self.resource_pool.resources.keys()}
        
        for allocation in plan.allocations:
            for resource_name, amount in allocation.allocated_resources.items():
                if resource_name not in resource_usage:
                    issues.append(f"Unknown resource: {resource_name}")
                    continue
                resource_usage[resource_name] += amount
        
        # Check if any resource is over-allocated
        for resource_name, used in resource_usage.items():
            resource = self.resource_pool.resources[resource_name]
            if used > resource.total_capacity:
                issues.append(
                    f"Resource {resource_name} over-allocated: "
                    f"{used:.2f} > {resource.total_capacity:.2f}"
                )
        
        return len(issues) == 0, issues
    
    def optimize_plan(self, plan: ResourcePlan, tasks: List[ConstrainedTask],
                     iterations: int = 10) -> ResourcePlan:
        """Optimize plan through iterative improvement"""
        best_plan = plan
        best_efficiency = plan.efficiency_ratio()
        
        for _ in range(iterations):
            # Reset resource pool
            for resource in self.resource_pool.resources.values():
                resource.available = resource.total_capacity
            
            # Try different task orderings
            import random
            shuffled_tasks = tasks.copy()
            random.shuffle(shuffled_tasks)
            
            # Create new plan with shuffled order
            new_plan = self.create_plan(shuffled_tasks)
            new_efficiency = new_plan.efficiency_ratio()
            
            # Keep better plan
            if new_efficiency > best_efficiency:
                best_plan = new_plan
                best_efficiency = new_efficiency
        
        return best_plan


def demonstrate_resource_constrained_planning():
    """Demonstrate resource-constrained planning"""
    print("=" * 60)
    print("RESOURCE-CONSTRAINED PLANNING DEMONSTRATION")
    print("=" * 60)
    
    # Create resource pool
    resource_pool = ResourcePool()
    resource_pool.add_resource(Resource(
        name="budget",
        resource_type=ResourceType.BUDGET,
        total_capacity=10000.0,
        available=10000.0,
        unit_cost=1.0
    ))
    resource_pool.add_resource(Resource(
        name="developers",
        resource_type=ResourceType.PERSONNEL,
        total_capacity=5.0,
        available=5.0,
        unit_cost=100.0
    ))
    resource_pool.add_resource(Resource(
        name="servers",
        resource_type=ResourceType.EQUIPMENT,
        total_capacity=10.0,
        available=10.0,
        unit_cost=50.0
    ))
    
    print("\n1. RESOURCE POOL INITIALIZED")
    print(f"   Budget: ${resource_pool.resources['budget'].total_capacity:.2f}")
    print(f"   Developers: {resource_pool.resources['developers'].total_capacity:.0f}")
    print(f"   Servers: {resource_pool.resources['servers'].total_capacity:.0f}")
    
    # Define tasks
    tasks = [
        ConstrainedTask(
            task_id="t1",
            name="Frontend Development",
            priority=3,
            resource_requirements={"budget": 2000, "developers": 2},
            duration=timedelta(days=14),
            value=1000.0
        ),
        ConstrainedTask(
            task_id="t2",
            name="Backend Development",
            priority=5,
            resource_requirements={"budget": 3000, "developers": 2, "servers": 3},
            duration=timedelta(days=21),
            value=1500.0
        ),
        ConstrainedTask(
            task_id="t3",
            name="Database Setup",
            priority=4,
            resource_requirements={"budget": 1500, "developers": 1, "servers": 4},
            duration=timedelta(days=7),
            value=800.0
        ),
        ConstrainedTask(
            task_id="t4",
            name="Testing",
            priority=2,
            resource_requirements={"budget": 1000, "developers": 1, "servers": 2},
            duration=timedelta(days=10),
            value=500.0
        ),
        ConstrainedTask(
            task_id="t5",
            name="Documentation",
            priority=1,
            resource_requirements={"budget": 500, "developers": 1},
            duration=timedelta(days=5),
            value=300.0
        ),
    ]
    
    print(f"\n2. TASKS DEFINED: {len(tasks)} tasks")
    for task in tasks:
        print(f"   {task.name}: priority={task.priority}, value=${task.value:.0f}")
    
    # Test different strategies
    strategies = [
        PlanningStrategy.GREEDY,
        PlanningStrategy.OPTIMAL,
        PlanningStrategy.HEURISTIC,
        PlanningStrategy.BALANCED
    ]
    
    print("\n3. COMPARING PLANNING STRATEGIES")
    print("-" * 60)
    
    for strategy in strategies:
        # Reset resource pool
        for resource in resource_pool.resources.values():
            resource.available = resource.total_capacity
        
        planner = ResourceConstrainedPlanner(resource_pool, strategy)
        plan = planner.create_plan(tasks)
        
        print(f"\n   Strategy: {strategy.value.upper()}")
        print(f"   Allocated Tasks: {len(plan.allocations)}/{len(tasks)}")
        print(f"   Total Cost: ${plan.total_cost:.2f}")
        print(f"   Total Value: ${plan.total_value:.2f}")
        print(f"   Efficiency Ratio: {plan.efficiency_ratio():.2f}")
        print(f"   Feasible: {plan.feasible}")
        
        if plan.unallocated_tasks:
            print(f"   Unallocated: {len(plan.unallocated_tasks)} tasks")
        
        # Show resource utilization
        utilization = resource_pool.get_utilization_report()
        print(f"   Resource Utilization:")
        for resource_name, util in utilization.items():
            print(f"     {resource_name}: {util:.1f}%")
    
    # Demonstrate plan validation
    print("\n4. PLAN VALIDATION")
    print("-" * 60)
    
    # Reset and create optimal plan
    for resource in resource_pool.resources.values():
        resource.available = resource.total_capacity
    
    planner = ResourceConstrainedPlanner(resource_pool, PlanningStrategy.OPTIMAL)
    plan = planner.create_plan(tasks)
    
    valid, issues = planner.validate_plan(plan)
    print(f"   Plan Valid: {valid}")
    if issues:
        print("   Issues:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print("   No issues found")
    
    # Demonstrate optimization
    print("\n5. PLAN OPTIMIZATION")
    print("-" * 60)
    
    # Reset resources
    for resource in resource_pool.resources.values():
        resource.available = resource.total_capacity
    
    initial_plan = planner.create_plan(tasks)
    print(f"   Initial Plan: {initial_plan}")
    
    # Reset resources for optimization
    for resource in resource_pool.resources.values():
        resource.available = resource.total_capacity
    
    optimized_plan = planner.optimize_plan(initial_plan, tasks, iterations=20)
    print(f"   Optimized Plan: {optimized_plan}")
    
    improvement = ((optimized_plan.efficiency_ratio() - initial_plan.efficiency_ratio()) 
                   / initial_plan.efficiency_ratio() * 100)
    print(f"   Improvement: {improvement:.1f}%")
    
    # Show final allocation details
    print("\n6. FINAL ALLOCATION DETAILS")
    print("-" * 60)
    
    for allocation in optimized_plan.allocations:
        print(f"\n   Task: {allocation.task_name}")
        print(f"   Resources:")
        for resource_name, amount in allocation.allocated_resources.items():
            print(f"     {resource_name}: {amount:.2f}")
        print(f"   Cost: ${allocation.cost:.2f}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"\nKey Takeaways:")
    print("1. Different strategies produce different allocations")
    print("2. Optimal strategy maximizes value-to-cost ratio")
    print("3. Resource utilization varies by strategy")
    print("4. Plan validation ensures feasibility")
    print("5. Optimization can improve efficiency")


if __name__ == "__main__":
    demonstrate_resource_constrained_planning()
