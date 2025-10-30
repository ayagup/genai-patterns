"""
Pattern 90: Load Balancing
Description:
    Distributes requests across multiple agent instances to optimize
    resource utilization, minimize latency, and ensure high availability.
Use Cases:
    - High-throughput agent systems
    - Auto-scaling agent deployments
    - Fault-tolerant architectures
    - Resource optimization
Key Features:
    - Multiple load balancing strategies
    - Health monitoring
    - Auto-scaling support
    - Request queue management
Example:
    >>> balancer = LoadBalancer()
    >>> balancer.add_agent(agent1)
    >>> balancer.add_agent(agent2)
    >>> result = balancer.execute(request)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import time
import threading
from collections import deque
import random
import statistics
class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    RANDOM = "random"
    IP_HASH = "ip_hash"
    LEAST_LOAD = "least_load"
class AgentHealth(Enum):
    """Health status of agent instances"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
@dataclass
class AgentInstance:
    """Agent instance in the pool"""
    instance_id: str
    agent: Any
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    health_status: AgentHealth = AgentHealth.UNKNOWN
    last_health_check: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    def get_avg_response_time(self) -> float:
        """Get average response time"""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    def get_success_rate(self) -> float:
        """Get success rate"""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests
    def get_load(self) -> float:
        """Get current load (0.0 to 1.0)"""
        if self.max_connections == 0:
            return 1.0
        return self.current_connections / self.max_connections
@dataclass
class HealthCheckConfig:
    """Configuration for health checks"""
    interval: float = 10.0  # seconds
    timeout: float = 5.0
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    check_function: Optional[Callable] = None
@dataclass
class LoadBalancerMetrics:
    """Metrics for load balancer"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    requests_per_second: float = 0.0
    active_connections: int = 0
    healthy_instances: int = 0
    total_instances: int = 0
class LoadBalancer:
    """
    Load balancer for distributing requests across agent instances
    Features:
    - Multiple balancing strategies
    - Health monitoring
    - Auto-scaling triggers
    - Connection pooling
    """
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS,
        health_check_config: Optional[HealthCheckConfig] = None
    ):
        self.strategy = strategy
        self.health_check_config = health_check_config or HealthCheckConfig()
        self.instances: List[AgentInstance] = []
        self.current_index = 0  # For round-robin
        self.request_queue: deque = deque()
        self.metrics = LoadBalancerMetrics()
        self.lock = threading.Lock()
        self.running = False
        self.health_check_thread: Optional[threading.Thread] = None
    def add_agent(
        self,
        agent: Any,
        instance_id: Optional[str] = None,
        weight: float = 1.0,
        max_connections: int = 100
    ) -> str:
        """
        Add agent instance to the pool
        Args:
            agent: Agent instance
            instance_id: Unique identifier
            weight: Weight for weighted strategies
            max_connections: Maximum concurrent connections
        Returns:
            Instance ID
        """
        if instance_id is None:
            instance_id = f"agent_{len(self.instances)}_{int(time.time())}"
        instance = AgentInstance(
            instance_id=instance_id,
            agent=agent,
            weight=weight,
            max_connections=max_connections
        )
        with self.lock:
            self.instances.append(instance)
            self.metrics.total_instances += 1
        return instance_id
    def remove_agent(self, instance_id: str) -> bool:
        """Remove agent instance from pool"""
        with self.lock:
            for i, instance in enumerate(self.instances):
                if instance.instance_id == instance_id:
                    self.instances.pop(i)
                    self.metrics.total_instances -= 1
                    return True
        return False
    def execute(
        self,
        request: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute request using load balancing
        Args:
            request: Request to execute
            timeout: Request timeout
        Returns:
            Response from agent
        """
        start_time = time.time()
        # Select agent instance
        instance = self._select_instance(request)
        if instance is None:
            self.metrics.failed_requests += 1
            return {
                'success': False,
                'error': 'No healthy agent instances available',
                'execution_time': time.time() - start_time
            }
        # Execute request
        try:
            with self.lock:
                instance.current_connections += 1
                self.metrics.active_connections += 1
            result = self._execute_on_instance(instance, request, timeout)
            execution_time = time.time() - start_time
            with self.lock:
                instance.response_times.append(execution_time)
                instance.total_requests += 1
                self.metrics.total_requests += 1
                if result.get('success', False):
                    self.metrics.successful_requests += 1
                else:
                    instance.failed_requests += 1
                    self.metrics.failed_requests += 1
            result['execution_time'] = execution_time
            result['instance_id'] = instance.instance_id
            return result
        except Exception as e:
            with self.lock:
                instance.failed_requests += 1
                self.metrics.failed_requests += 1
            return {
                'success': False,
                'error': str(e),
                'instance_id': instance.instance_id,
                'execution_time': time.time() - start_time
            }
        finally:
            with self.lock:
                instance.current_connections -= 1
                self.metrics.active_connections -= 1
    def _select_instance(self, request: Dict[str, Any]) -> Optional[AgentInstance]:
        """Select agent instance based on strategy"""
        with self.lock:
            healthy_instances = [
                inst for inst in self.instances
                if inst.health_status == AgentHealth.HEALTHY
                and inst.current_connections < inst.max_connections
            ]
        if not healthy_instances:
            return None
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
            return self._response_time_select(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.IP_HASH:
            return self._ip_hash_select(healthy_instances, request)
        elif self.strategy == LoadBalancingStrategy.LEAST_LOAD:
            return self._least_load_select(healthy_instances)
        return healthy_instances[0]
    def _round_robin_select(self, instances: List[AgentInstance]) -> AgentInstance:
        """Round-robin selection"""
        with self.lock:
            instance = instances[self.current_index % len(instances)]
            self.current_index += 1
        return instance
    def _least_connections_select(self, instances: List[AgentInstance]) -> AgentInstance:
        """Select instance with least connections"""
        return min(instances, key=lambda x: x.current_connections)
    def _weighted_round_robin_select(self, instances: List[AgentInstance]) -> AgentInstance:
        """Weighted round-robin selection"""
        total_weight = sum(inst.weight for inst in instances)
        if total_weight == 0:
            return instances[0]
        # Generate random number
        rand = random.uniform(0, total_weight)
        cumulative_weight = 0
        for instance in instances:
            cumulative_weight += instance.weight
            if rand <= cumulative_weight:
                return instance
        return instances[-1]
    def _response_time_select(self, instances: List[AgentInstance]) -> AgentInstance:
        """Select instance with best response time"""
        return min(instances, key=lambda x: x.get_avg_response_time() or float('inf'))
    def _ip_hash_select(
        self,
        instances: List[AgentInstance],
        request: Dict[str, Any]
    ) -> AgentInstance:
        """Hash-based selection for session affinity"""
        client_id = request.get('client_id', '')
        hash_value = hash(client_id) % len(instances)
        return instances[hash_value]
    def _least_load_select(self, instances: List[AgentInstance]) -> AgentInstance:
        """Select instance with least load"""
        return min(instances, key=lambda x: x.get_load())
    def _execute_on_instance(
        self,
        instance: AgentInstance,
        request: Dict[str, Any],
        timeout: Optional[float]
    ) -> Dict[str, Any]:
        """Execute request on specific instance"""
        # In reality, this would call the agent's execution method
        # For now, simulate execution
        if hasattr(instance.agent, 'execute'):
            return instance.agent.execute(request.get('task', ''))
        # Simulated execution
        time.sleep(random.uniform(0.01, 0.1))
        # Simulate occasional failures
        if random.random() < 0.05:  # 5% failure rate
            return {'success': False, 'error': 'Simulated failure'}
        return {
            'success': True,
            'result': f"Processed by {instance.instance_id}",
            'task': request.get('task', '')
        }
    def start_health_checks(self):
        """Start background health check thread"""
        if self.running:
            return
        self.running = True
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self.health_check_thread.start()
    def stop_health_checks(self):
        """Stop health check thread"""
        self.running = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5.0)
    def _health_check_loop(self):
        """Background health check loop"""
        while self.running:
            self._perform_health_checks()
            time.sleep(self.health_check_config.interval)
    def _perform_health_checks(self):
        """Perform health checks on all instances"""
        with self.lock:
            instances_to_check = list(self.instances)
        for instance in instances_to_check:
            is_healthy = self._check_instance_health(instance)
            with self.lock:
                instance.last_health_check = time.time()
                if is_healthy:
                    instance.health_status = AgentHealth.HEALTHY
                else:
                    # Check failure rate
                    success_rate = instance.get_success_rate()
                    if success_rate < 0.5:
                        instance.health_status = AgentHealth.UNHEALTHY
                    elif success_rate < 0.8:
                        instance.health_status = AgentHealth.DEGRADED
                    else:
                        instance.health_status = AgentHealth.HEALTHY
        # Update metrics
        with self.lock:
            self.metrics.healthy_instances = sum(
                1 for inst in self.instances
                if inst.health_status == AgentHealth.HEALTHY
            )
    def _check_instance_health(self, instance: AgentInstance) -> bool:
        """Check health of a single instance"""
        if self.health_check_config.check_function:
            try:
                return self.health_check_config.check_function(instance.agent)
            except Exception:
                return False
        # Default health check: check if not overloaded
        return instance.get_load() < 0.9 and instance.get_success_rate() > 0.7
    def get_metrics(self) -> LoadBalancerMetrics:
        """Get current load balancer metrics"""
        with self.lock:
            # Calculate requests per second
            if self.metrics.total_requests > 0:
                # Simple approximation
                self.metrics.requests_per_second = self.metrics.total_requests / max(
                    time.time() - (self.instances[0].last_health_check if self.instances else time.time()),
                    1.0
                )
            # Calculate average response time
            all_times = []
            for instance in self.instances:
                all_times.extend(instance.response_times)
            if all_times:
                self.metrics.avg_response_time = statistics.mean(all_times)
            return LoadBalancerMetrics(
                total_requests=self.metrics.total_requests,
                successful_requests=self.metrics.successful_requests,
                failed_requests=self.metrics.failed_requests,
                avg_response_time=self.metrics.avg_response_time,
                requests_per_second=self.metrics.requests_per_second,
                active_connections=self.metrics.active_connections,
                healthy_instances=self.metrics.healthy_instances,
                total_instances=self.metrics.total_instances
            )
    def get_instance_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for each instance"""
        stats = []
        with self.lock:
            for instance in self.instances:
                stats.append({
                    'instance_id': instance.instance_id,
                    'health_status': instance.health_status.value,
                    'current_connections': instance.current_connections,
                    'total_requests': instance.total_requests,
                    'failed_requests': instance.failed_requests,
                    'success_rate': instance.get_success_rate(),
                    'avg_response_time': instance.get_avg_response_time(),
                    'load': instance.get_load(),
                    'weight': instance.weight
                })
        return stats
    def scale_up(self, agent_factory: Callable, count: int = 1) -> List[str]:
        """
        Scale up by adding more instances
        Args:
            agent_factory: Function that creates new agent instances
            count: Number of instances to add
        Returns:
            List of new instance IDs
        """
        new_ids = []
        for _ in range(count):
            new_agent = agent_factory()
            instance_id = self.add_agent(new_agent)
            new_ids.append(instance_id)
        return new_ids
    def scale_down(self, count: int = 1) -> List[str]:
        """
        Scale down by removing instances with least load
        Args:
            count: Number of instances to remove
        Returns:
            List of removed instance IDs
        """
        removed_ids = []
        with self.lock:
            # Sort by load (ascending)
            sorted_instances = sorted(self.instances, key=lambda x: x.get_load())
            for i in range(min(count, len(sorted_instances))):
                instance = sorted_instances[i]
                if instance.current_connections == 0:  # Only remove idle instances
                    removed_ids.append(instance.instance_id)
        for instance_id in removed_ids:
            self.remove_agent(instance_id)
        return removed_ids
    def should_scale_up(self) -> bool:
        """Determine if scaling up is needed"""
        metrics = self.get_metrics()
        # Scale up if:
        # - Average load > 70%
        # - Or response time is degrading
        avg_load = sum(inst.get_load() for inst in self.instances) / max(len(self.instances), 1)
        return (avg_load > 0.7 or 
                metrics.avg_response_time > 1.0 or
                metrics.healthy_instances < metrics.total_instances * 0.5)
    def should_scale_down(self) -> bool:
        """Determine if scaling down is possible"""
        # Scale down if average load < 30% and we have more than 1 instance
        avg_load = sum(inst.get_load() for inst in self.instances) / max(len(self.instances), 1)
        return avg_load < 0.3 and len(self.instances) > 1
# Simple mock agent for demonstration
class SimpleAgent:
    """Simple agent for load balancer demonstration"""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    def execute(self, task: str) -> Dict[str, Any]:
        """Execute a task"""
        time.sleep(random.uniform(0.01, 0.05))
        return {
            'success': True,
            'result': f"Agent {self.agent_id} processed: {task}",
            'agent_id': self.agent_id
        }
def main():
    """Demonstrate load balancing pattern"""
    print("=" * 60)
    print("Load Balancing Pattern Demonstration")
    print("=" * 60)
    print("\n1. Creating Load Balancer with Multiple Strategies")
    print("-" * 60)
    # Test different strategies
    strategies = [
        LoadBalancingStrategy.ROUND_ROBIN,
        LoadBalancingStrategy.LEAST_CONNECTIONS,
        LoadBalancingStrategy.RANDOM
    ]
    for strategy in strategies:
        print(f"\nTesting {strategy.value}:")
        balancer = LoadBalancer(strategy=strategy)
        # Add agents
        for i in range(3):
            agent = SimpleAgent(f"agent_{i}")
            balancer.add_agent(agent, weight=i+1)  # Different weights
        # Set all to healthy
        for instance in balancer.instances:
            instance.health_status = AgentHealth.HEALTHY
        # Execute requests
        results = []
        for i in range(10):
            result = balancer.execute({'task': f'task_{i}'})
            results.append(result['instance_id'])
        # Show distribution
        from collections import Counter
        distribution = Counter(results)
        print(f"  Distribution: {dict(distribution)}")
    print("\n" + "=" * 60)
    print("2. Load Balancer with Health Checks")
    print("=" * 60)
    balancer = LoadBalancer(
        strategy=LoadBalancingStrategy.LEAST_CONNECTIONS
    )
    # Add agents
    agent_ids = []
    for i in range(4):
        agent = SimpleAgent(f"agent_{i}")
        instance_id = balancer.add_agent(agent, max_connections=10)
        agent_ids.append(instance_id)
    # Start health checks
    balancer.start_health_checks()
    # Manually set health statuses for demonstration
    balancer.instances[0].health_status = AgentHealth.HEALTHY
    balancer.instances[1].health_status = AgentHealth.HEALTHY
    balancer.instances[2].health_status = AgentHealth.DEGRADED
    balancer.instances[3].health_status = AgentHealth.UNHEALTHY
    print("\nInstance Health Status:")
    for instance in balancer.instances:
        print(f"  {instance.instance_id}: {instance.health_status.value}")
    # Execute requests
    print("\nExecuting 20 requests...")
    for i in range(20):
        balancer.execute({'task': f'task_{i}'})
    metrics = balancer.get_metrics()
    print(f"\nMetrics after execution:")
    print(f"  Total Requests: {metrics.total_requests}")
    print(f"  Successful: {metrics.successful_requests}")
    print(f"  Failed: {metrics.failed_requests}")
    print(f"  Active Connections: {metrics.active_connections}")
    print(f"  Healthy Instances: {metrics.healthy_instances}/{metrics.total_instances}")
    print(f"  Avg Response Time: {metrics.avg_response_time:.4f}s")
    balancer.stop_health_checks()
    print("\n" + "=" * 60)
    print("3. Instance Statistics")
    print("=" * 60)
    stats = balancer.get_instance_stats()
    for stat in sorted(stats, key=lambda x: x['total_requests'], reverse=True):
        print(f"\n{stat['instance_id']}:")
        print(f"  Health: {stat['health_status']}")
        print(f"  Requests: {stat['total_requests']}")
        print(f"  Success Rate: {stat['success_rate']:.2%}")
        print(f"  Avg Response Time: {stat['avg_response_time']:.4f}s")
        print(f"  Current Load: {stat['load']:.2%}")
    print("\n" + "=" * 60)
    print("4. Auto-scaling")
    print("=" * 60)
    def agent_factory():
        return SimpleAgent(f"autoscaled_{random.randint(1000, 9999)}")
    print(f"\nCurrent instances: {len(balancer.instances)}")
    print(f"Should scale up: {balancer.should_scale_up()}")
    print(f"Should scale down: {balancer.should_scale_down()}")
    # Simulate high load
    for instance in balancer.instances:
        instance.current_connections = 8  # 80% of max (10)
    print(f"\nAfter simulating high load:")
    print(f"Should scale up: {balancer.should_scale_up()}")
    if balancer.should_scale_up():
        new_ids = balancer.scale_up(agent_factory, count=2)
        print(f"Scaled up: Added {len(new_ids)} instances")
        print(f"New instances: {new_ids}")
    print(f"Total instances now: {len(balancer.instances)}")
    # Simulate low load
    for instance in balancer.instances:
        instance.current_connections = 0
    print(f"\nAfter simulating low load:")
    print(f"Should scale down: {balancer.should_scale_down()}")
    if balancer.should_scale_down():
        removed_ids = balancer.scale_down(count=2)
        print(f"Scaled down: Removed {len(removed_ids)} instances")
        print(f"Removed instances: {removed_ids}")
    print(f"Total instances now: {len(balancer.instances)}")
    print("\n" + "=" * 60)
    print("5. Weighted Round Robin")
    print("=" * 60)
    weighted_balancer = LoadBalancer(
        strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN
    )
    # Add agents with different weights
    weights = [1.0, 2.0, 3.0]
    for i, weight in enumerate(weights):
        agent = SimpleAgent(f"weighted_agent_{i}")
        weighted_balancer.add_agent(agent, weight=weight)
        weighted_balancer.instances[i].health_status = AgentHealth.HEALTHY
    # Execute requests
    results = []
    for i in range(60):
        result = weighted_balancer.execute({'task': f'task_{i}'})
        results.append(result['instance_id'])
    # Show distribution
    from collections import Counter
    distribution = Counter(results)
    print("\nWeighted Distribution (60 requests):")
    print(f"  Weights: {weights}")
    print(f"  Expected ratio: 1:2:3")
    print(f"  Actual distribution:")
    for instance_id, count in sorted(distribution.items()):
        percentage = count / 60 * 100
        print(f"    {instance_id}: {count} requests ({percentage:.1f}%)")
    print("\n" + "=" * 60)
    print("Load Balancing demonstration complete!")
    print("=" * 60)
if __name__ == "__main__":
    main()
