"""
Agent Resource Allocation Pattern

Manages allocation of limited resources among competing agents.
Implements fair allocation, priority-based distribution, and optimization.

Use Cases:
- Computational resource management
- Task scheduling
- Budget allocation
- Capacity planning

Advantages:
- Fair resource distribution
- Priority handling
- Resource optimization
- Contention resolution
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import heapq
import json


class ResourceType(Enum):
    """Types of resources"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    API_CALLS = "api_calls"
    TOKENS = "tokens"
    CUSTOM = "custom"


class AllocationStrategy(Enum):
    """Resource allocation strategies"""
    FAIR_SHARE = "fair_share"
    PRIORITY_BASED = "priority_based"
    PROPORTIONAL = "proportional"
    ROUND_ROBIN = "round_robin"
    DYNAMIC = "dynamic"


class RequestStatus(Enum):
    """Resource request status"""
    PENDING = "pending"
    ALLOCATED = "allocated"
    DENIED = "denied"
    RELEASED = "released"
    EXPIRED = "expired"


@dataclass
class Resource:
    """Resource definition"""
    resource_id: str
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceRequest:
    """Request for resources"""
    request_id: str
    agent_id: str
    resource_type: ResourceType
    amount: float
    priority: int  # Higher = more priority
    requested_at: datetime
    expires_at: Optional[datetime] = None
    status: RequestStatus = RequestStatus.PENDING
    allocated_amount: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceAllocation:
    """Allocated resource"""
    allocation_id: str
    request_id: str
    agent_id: str
    resource_type: ResourceType
    amount: float
    allocated_at: datetime
    expires_at: Optional[datetime] = None
    released: bool = False


@dataclass
class AgentQuota:
    """Resource quota for agent"""
    agent_id: str
    resource_type: ResourceType
    max_allocation: float
    current_usage: float = 0.0
    reserved: float = 0.0


class ResourcePool:
    """Manages resource pool"""
    
    def __init__(self):
        self.resources: Dict[ResourceType, Resource] = {}
        self.allocations: Dict[str, ResourceAllocation] = {}
    
    def add_resource(self,
                    resource_type: ResourceType,
                    capacity: float,
                    unit: str = "") -> str:
        """
        Add resource to pool.
        
        Args:
            resource_type: Type of resource
            capacity: Total capacity
            unit: Unit of measurement
            
        Returns:
            Resource ID
        """
        resource_id = "res_{}".format(resource_type.value)
        
        resource = Resource(
            resource_id=resource_id,
            resource_type=resource_type,
            total_capacity=capacity,
            available_capacity=capacity,
            unit=unit
        )
        
        self.resources[resource_type] = resource
        
        return resource_id
    
    def allocate(self,
                allocation: ResourceAllocation) -> bool:
        """
        Allocate resources.
        
        Args:
            allocation: Allocation to make
            
        Returns:
            Whether allocation succeeded
        """
        resource = self.resources.get(allocation.resource_type)
        
        if not resource:
            return False
        
        if resource.available_capacity < allocation.amount:
            return False
        
        # Update availability
        resource.available_capacity -= allocation.amount
        
        # Store allocation
        self.allocations[allocation.allocation_id] = allocation
        
        return True
    
    def release(self, allocation_id: str) -> bool:
        """
        Release allocated resources.
        
        Args:
            allocation_id: Allocation to release
            
        Returns:
            Whether release succeeded
        """
        allocation = self.allocations.get(allocation_id)
        
        if not allocation or allocation.released:
            return False
        
        resource = self.resources.get(allocation.resource_type)
        
        if not resource:
            return False
        
        # Return to pool
        resource.available_capacity += allocation.amount
        allocation.released = True
        
        return True
    
    def get_availability(self, resource_type: ResourceType) -> float:
        """Get available capacity for resource type"""
        resource = self.resources.get(resource_type)
        return resource.available_capacity if resource else 0.0
    
    def get_utilization(self, resource_type: ResourceType) -> float:
        """Get resource utilization percentage"""
        resource = self.resources.get(resource_type)
        
        if not resource or resource.total_capacity == 0:
            return 0.0
        
        used = resource.total_capacity - resource.available_capacity
        return (used / resource.total_capacity) * 100


class RequestQueue:
    """Priority queue for resource requests"""
    
    def __init__(self):
        self.heap: List[Tuple[int, str, ResourceRequest]] = []
        self.requests: Dict[str, ResourceRequest] = {}
    
    def add_request(self, request: ResourceRequest) -> None:
        """Add request to queue"""
        # Use negative priority for max heap
        heapq.heappush(
            self.heap,
            (-request.priority, request.requested_at.timestamp(), request)
        )
        self.requests[request.request_id] = request
    
    def get_next(self) -> Optional[ResourceRequest]:
        """Get highest priority request"""
        while self.heap:
            _, _, request = heapq.heappop(self.heap)
            
            # Check if still pending
            if request.request_id in self.requests and request.status == RequestStatus.PENDING:
                return request
        
        return None
    
    def remove_request(self, request_id: str) -> bool:
        """Remove request from queue"""
        if request_id in self.requests:
            del self.requests[request_id]
            return True
        return False
    
    def get_pending_count(self) -> int:
        """Get count of pending requests"""
        return sum(
            1 for r in self.requests.values()
            if r.status == RequestStatus.PENDING
        )


class QuotaManager:
    """Manages agent resource quotas"""
    
    def __init__(self):
        self.quotas: Dict[Tuple[str, ResourceType], AgentQuota] = {}
    
    def set_quota(self,
                 agent_id: str,
                 resource_type: ResourceType,
                 max_allocation: float) -> None:
        """
        Set resource quota for agent.
        
        Args:
            agent_id: Agent identifier
            resource_type: Resource type
            max_allocation: Maximum allowed allocation
        """
        key = (agent_id, resource_type)
        
        if key in self.quotas:
            quota = self.quotas[key]
            quota.max_allocation = max_allocation
        else:
            self.quotas[key] = AgentQuota(
                agent_id=agent_id,
                resource_type=resource_type,
                max_allocation=max_allocation
            )
    
    def check_quota(self,
                   agent_id: str,
                   resource_type: ResourceType,
                   amount: float) -> bool:
        """
        Check if allocation would exceed quota.
        
        Args:
            agent_id: Agent identifier
            resource_type: Resource type
            amount: Amount to allocate
            
        Returns:
            Whether allocation is within quota
        """
        key = (agent_id, resource_type)
        
        if key not in self.quotas:
            return True  # No quota set
        
        quota = self.quotas[key]
        
        return (quota.current_usage + quota.reserved + amount) <= quota.max_allocation
    
    def update_usage(self,
                    agent_id: str,
                    resource_type: ResourceType,
                    delta: float) -> None:
        """Update current usage"""
        key = (agent_id, resource_type)
        
        if key in self.quotas:
            quota = self.quotas[key]
            quota.current_usage += delta
            quota.current_usage = max(0, quota.current_usage)
    
    def reserve(self,
               agent_id: str,
               resource_type: ResourceType,
               amount: float) -> bool:
        """Reserve quota space"""
        key = (agent_id, resource_type)
        
        if not self.check_quota(agent_id, resource_type, amount):
            return False
        
        if key in self.quotas:
            self.quotas[key].reserved += amount
        
        return True
    
    def release_reservation(self,
                           agent_id: str,
                           resource_type: ResourceType,
                           amount: float) -> None:
        """Release reserved quota"""
        key = (agent_id, resource_type)
        
        if key in self.quotas:
            quota = self.quotas[key]
            quota.reserved -= amount
            quota.reserved = max(0, quota.reserved)


class AllocationEngine:
    """Executes resource allocation strategies"""
    
    def __init__(self,
                 resource_pool: ResourcePool,
                 quota_manager: QuotaManager):
        self.resource_pool = resource_pool
        self.quota_manager = quota_manager
    
    def allocate_fair_share(self,
                           requests: List[ResourceRequest]) -> List[ResourceAllocation]:
        """
        Allocate resources using fair share strategy.
        
        Args:
            requests: List of requests
            
        Returns:
            List of allocations
        """
        allocations = []
        
        if not requests:
            return allocations
        
        # Group by resource type
        by_type: Dict[ResourceType, List[ResourceRequest]] = {}
        for req in requests:
            if req.resource_type not in by_type:
                by_type[req.resource_type] = []
            by_type[req.resource_type].append(req)
        
        # Allocate each resource type
        for resource_type, type_requests in by_type.items():
            available = self.resource_pool.get_availability(resource_type)
            
            # Calculate fair share
            total_requested = sum(r.amount for r in type_requests)
            
            if total_requested <= available:
                # Can satisfy all requests
                share = 1.0
            else:
                # Proportional allocation
                share = available / total_requested
            
            # Allocate to each request
            for req in type_requests:
                allocated_amount = req.amount * share
                
                # Check quota
                if not self.quota_manager.check_quota(
                    req.agent_id,
                    req.resource_type,
                    allocated_amount
                ):
                    continue
                
                allocation = ResourceAllocation(
                    allocation_id="alloc_{}_{}".format(
                        req.request_id,
                        datetime.now().timestamp()
                    ),
                    request_id=req.request_id,
                    agent_id=req.agent_id,
                    resource_type=req.resource_type,
                    amount=allocated_amount,
                    allocated_at=datetime.now(),
                    expires_at=req.expires_at
                )
                
                if self.resource_pool.allocate(allocation):
                    allocations.append(allocation)
                    req.status = RequestStatus.ALLOCATED
                    req.allocated_amount = allocated_amount
                    
                    # Update quota
                    self.quota_manager.update_usage(
                        req.agent_id,
                        req.resource_type,
                        allocated_amount
                    )
        
        return allocations
    
    def allocate_priority_based(self,
                               requests: List[ResourceRequest]) -> List[ResourceAllocation]:
        """
        Allocate resources based on priority.
        
        Args:
            requests: List of requests
            
        Returns:
            List of allocations
        """
        allocations = []
        
        # Sort by priority (descending)
        sorted_requests = sorted(
            requests,
            key=lambda r: r.priority,
            reverse=True
        )
        
        for req in sorted_requests:
            # Check availability
            available = self.resource_pool.get_availability(req.resource_type)
            
            if available < req.amount:
                req.status = RequestStatus.DENIED
                continue
            
            # Check quota
            if not self.quota_manager.check_quota(
                req.agent_id,
                req.resource_type,
                req.amount
            ):
                req.status = RequestStatus.DENIED
                continue
            
            # Allocate
            allocation = ResourceAllocation(
                allocation_id="alloc_{}_{}".format(
                    req.request_id,
                    datetime.now().timestamp()
                ),
                request_id=req.request_id,
                agent_id=req.agent_id,
                resource_type=req.resource_type,
                amount=req.amount,
                allocated_at=datetime.now(),
                expires_at=req.expires_at
            )
            
            if self.resource_pool.allocate(allocation):
                allocations.append(allocation)
                req.status = RequestStatus.ALLOCATED
                req.allocated_amount = req.amount
                
                # Update quota
                self.quota_manager.update_usage(
                    req.agent_id,
                    req.resource_type,
                    req.amount
                )
        
        return allocations


class AgentResourceAllocator:
    """
    Comprehensive resource allocation system for agents.
    Manages resources, quotas, and allocation strategies.
    """
    
    def __init__(self, strategy: AllocationStrategy = AllocationStrategy.FAIR_SHARE):
        self.strategy = strategy
        
        # Components
        self.resource_pool = ResourcePool()
        self.quota_manager = QuotaManager()
        self.allocation_engine = AllocationEngine(
            self.resource_pool,
            self.quota_manager
        )
        self.request_queue = RequestQueue()
        
        # State
        self.request_counter = 0
    
    def add_resource(self,
                    resource_type: ResourceType,
                    capacity: float,
                    unit: str = "") -> str:
        """Add resource to pool"""
        return self.resource_pool.add_resource(resource_type, capacity, unit)
    
    def set_quota(self,
                 agent_id: str,
                 resource_type: ResourceType,
                 max_allocation: float) -> None:
        """Set resource quota for agent"""
        self.quota_manager.set_quota(agent_id, resource_type, max_allocation)
    
    def request_resource(self,
                        agent_id: str,
                        resource_type: ResourceType,
                        amount: float,
                        priority: int = 5,
                        ttl: Optional[int] = None) -> str:
        """
        Request resource allocation.
        
        Args:
            agent_id: Requesting agent
            resource_type: Type of resource
            amount: Amount requested
            priority: Request priority (1-10)
            ttl: Time to live in seconds
            
        Returns:
            Request ID
        """
        request_id = "req_{}".format(self.request_counter)
        self.request_counter += 1
        
        expires_at = None
        if ttl:
            expires_at = datetime.now() + timedelta(seconds=ttl)
        
        request = ResourceRequest(
            request_id=request_id,
            agent_id=agent_id,
            resource_type=resource_type,
            amount=amount,
            priority=priority,
            requested_at=datetime.now(),
            expires_at=expires_at
        )
        
        self.request_queue.add_request(request)
        
        return request_id
    
    def process_requests(self) -> List[ResourceAllocation]:
        """
        Process pending resource requests.
        
        Returns:
            List of allocations made
        """
        # Collect pending requests
        pending_requests = []
        
        while True:
            request = self.request_queue.get_next()
            if not request:
                break
            
            # Check expiration
            if request.expires_at and datetime.now() > request.expires_at:
                request.status = RequestStatus.EXPIRED
                continue
            
            pending_requests.append(request)
        
        # Execute allocation strategy
        if self.strategy == AllocationStrategy.FAIR_SHARE:
            return self.allocation_engine.allocate_fair_share(pending_requests)
        elif self.strategy == AllocationStrategy.PRIORITY_BASED:
            return self.allocation_engine.allocate_priority_based(pending_requests)
        else:
            return self.allocation_engine.allocate_fair_share(pending_requests)
    
    def release_resource(self, allocation_id: str) -> bool:
        """
        Release allocated resource.
        
        Args:
            allocation_id: Allocation to release
            
        Returns:
            Whether release succeeded
        """
        allocation = self.resource_pool.allocations.get(allocation_id)
        
        if not allocation:
            return False
        
        # Release from pool
        success = self.resource_pool.release(allocation_id)
        
        if success:
            # Update quota
            self.quota_manager.update_usage(
                allocation.agent_id,
                allocation.resource_type,
                -allocation.amount
            )
        
        return success
    
    def get_agent_allocations(self, agent_id: str) -> List[ResourceAllocation]:
        """Get all allocations for agent"""
        return [
            alloc for alloc in self.resource_pool.allocations.values()
            if alloc.agent_id == agent_id and not alloc.released
        ]
    
    def get_resource_status(self, resource_type: ResourceType) -> Dict[str, Any]:
        """Get status of resource"""
        resource = self.resource_pool.resources.get(resource_type)
        
        if not resource:
            return {}
        
        utilization = self.resource_pool.get_utilization(resource_type)
        
        # Count allocations
        active_allocations = sum(
            1 for alloc in self.resource_pool.allocations.values()
            if alloc.resource_type == resource_type and not alloc.released
        )
        
        return {
            "resource_type": resource_type.value,
            "total_capacity": resource.total_capacity,
            "available_capacity": resource.available_capacity,
            "utilization_percent": utilization,
            "active_allocations": active_allocations,
            "unit": resource.unit
        }
    
    def get_agent_quota_status(self,
                              agent_id: str,
                              resource_type: ResourceType) -> Dict[str, Any]:
        """Get quota status for agent"""
        key = (agent_id, resource_type)
        quota = self.quota_manager.quotas.get(key)
        
        if not quota:
            return {}
        
        return {
            "agent_id": agent_id,
            "resource_type": resource_type.value,
            "max_allocation": quota.max_allocation,
            "current_usage": quota.current_usage,
            "reserved": quota.reserved,
            "available": quota.max_allocation - quota.current_usage - quota.reserved
        }
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired allocations.
        
        Returns:
            Number of allocations cleaned up
        """
        count = 0
        now = datetime.now()
        
        for allocation in list(self.resource_pool.allocations.values()):
            if allocation.expires_at and now > allocation.expires_at:
                if self.release_resource(allocation.allocation_id):
                    count += 1
        
        return count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get allocation statistics"""
        total_allocations = len(self.resource_pool.allocations)
        active_allocations = sum(
            1 for a in self.resource_pool.allocations.values()
            if not a.released
        )
        
        pending_requests = self.request_queue.get_pending_count()
        
        resource_stats = {}
        for resource_type in self.resource_pool.resources.keys():
            resource_stats[resource_type.value] = self.get_resource_status(
                resource_type
            )
        
        return {
            "total_allocations": total_allocations,
            "active_allocations": active_allocations,
            "pending_requests": pending_requests,
            "resource_types": len(self.resource_pool.resources),
            "resources": resource_stats
        }


def demonstrate_resource_allocation():
    """Demonstrate agent resource allocation"""
    print("=" * 70)
    print("Agent Resource Allocation Demonstration")
    print("=" * 70)
    
    allocator = AgentResourceAllocator(strategy=AllocationStrategy.FAIR_SHARE)
    
    # Example 1: Add resources
    print("\n1. Adding Resources to Pool:")
    
    allocator.add_resource(ResourceType.CPU, 100.0, "cores")
    allocator.add_resource(ResourceType.MEMORY, 1024.0, "GB")
    allocator.add_resource(ResourceType.TOKENS, 10000.0, "tokens")
    
    print("  Added 3 resource types")
    
    # Example 2: Set quotas
    print("\n2. Setting Agent Quotas:")
    
    agents = ["agent_1", "agent_2", "agent_3"]
    
    for agent_id in agents:
        allocator.set_quota(agent_id, ResourceType.CPU, 40.0)
        allocator.set_quota(agent_id, ResourceType.MEMORY, 400.0)
        allocator.set_quota(agent_id, ResourceType.TOKENS, 5000.0)
        print("  Set quotas for {}".format(agent_id))
    
    # Example 3: Request resources
    print("\n3. Requesting Resources:")
    
    requests = [
        ("agent_1", ResourceType.CPU, 20.0, 8),
        ("agent_2", ResourceType.CPU, 30.0, 6),
        ("agent_3", ResourceType.CPU, 40.0, 7),
        ("agent_1", ResourceType.MEMORY, 200.0, 5),
        ("agent_2", ResourceType.MEMORY, 300.0, 9),
    ]
    
    for agent_id, resource_type, amount, priority in requests:
        req_id = allocator.request_resource(
            agent_id,
            resource_type,
            amount,
            priority
        )
        print("  {} requested {:.0f} {} (priority: {})".format(
            agent_id, amount, resource_type.value, priority
        ))
    
    # Example 4: Process requests
    print("\n4. Processing Resource Requests:")
    
    allocations = allocator.process_requests()
    
    print("  Processed {} allocations".format(len(allocations)))
    
    for alloc in allocations:
        print("    {} allocated {:.2f} {}".format(
            alloc.agent_id,
            alloc.amount,
            alloc.resource_type.value
        ))
    
    # Example 5: Check resource status
    print("\n5. Resource Status:")
    
    for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.TOKENS]:
        status = allocator.get_resource_status(resource_type)
        
        print("\n  {}:".format(resource_type.value.upper()))
        print("    Total: {:.0f} {}".format(
            status["total_capacity"],
            status["unit"]
        ))
        print("    Available: {:.0f} {}".format(
            status["available_capacity"],
            status["unit"]
        ))
        print("    Utilization: {:.1f}%".format(status["utilization_percent"]))
    
    # Example 6: Check agent quotas
    print("\n6. Agent Quota Status:")
    
    for agent_id in agents:
        quota_status = allocator.get_agent_quota_status(
            agent_id,
            ResourceType.CPU
        )
        
        if quota_status:
            print("\n  {}:".format(agent_id))
            print("    Max allocation: {:.0f}".format(quota_status["max_allocation"]))
            print("    Current usage: {:.0f}".format(quota_status["current_usage"]))
            print("    Available: {:.0f}".format(quota_status["available"]))
    
    # Example 7: Release resources
    print("\n7. Releasing Resources:")
    
    agent_allocations = allocator.get_agent_allocations("agent_1")
    
    if agent_allocations:
        first_alloc = agent_allocations[0]
        success = allocator.release_resource(first_alloc.allocation_id)
        
        if success:
            print("  Released {:.0f} {} from agent_1".format(
                first_alloc.amount,
                first_alloc.resource_type.value
            ))
    
    # Example 8: Priority-based allocation
    print("\n8. Priority-Based Allocation:")
    
    # Switch to priority-based strategy
    allocator.strategy = AllocationStrategy.PRIORITY_BASED
    
    # New requests
    allocator.request_resource("agent_1", ResourceType.TOKENS, 2000, priority=10)
    allocator.request_resource("agent_2", ResourceType.TOKENS, 3000, priority=3)
    allocator.request_resource("agent_3", ResourceType.TOKENS, 6000, priority=7)
    
    priority_allocs = allocator.process_requests()
    
    print("  Priority-based allocations:")
    for alloc in priority_allocs:
        print("    {} got {:.0f} tokens".format(
            alloc.agent_id,
            alloc.amount
        ))
    
    # Example 9: Statistics
    print("\n9. Allocation Statistics:")
    stats = allocator.get_statistics()
    print(json.dumps(stats, indent=2))
    
    # Example 10: Cleanup
    print("\n10. Resource Cleanup:")
    
    expired = allocator.cleanup_expired()
    print("  Cleaned up {} expired allocations".format(expired))


if __name__ == "__main__":
    demonstrate_resource_allocation()
