"""
Simulation Testing Pattern

Tests agent in simulated environments with synthetic users, mock APIs, and 
environment simulators. Enables safe, reproducible testing of complex scenarios
and edge cases without real-world consequences.

Use Cases:
- Testing autonomous agents before deployment
- Validating multi-agent interactions
- Edge case and failure scenario testing
- Performance and stress testing
- Safety-critical system validation

Benefits:
- Safe testing: No real-world impact
- Reproducibility: Consistent test conditions
- Comprehensive coverage: Test rare edge cases
- Cost-effective: No real API/resource costs
- Parallel testing: Run multiple scenarios simultaneously
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import time


class SimulationState(Enum):
    """States of simulation execution"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class EntityType(Enum):
    """Types of entities in simulation"""
    USER = "user"
    API = "api"
    DATABASE = "database"
    ENVIRONMENT = "environment"
    AGENT = "agent"


@dataclass
class SimulatedEntity:
    """A simulated entity in the environment"""
    entity_id: str
    entity_type: EntityType
    behavior_model: str
    state: Dict[str, Any] = field(default_factory=dict)
    interaction_count: int = 0
    
    def interact(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate interaction with this entity"""
        self.interaction_count += 1
        
        # Simulate behavior based on entity type
        if self.entity_type == EntityType.API:
            return self._simulate_api_response(action, params)
        elif self.entity_type == EntityType.USER:
            return self._simulate_user_response(action, params)
        elif self.entity_type == EntityType.DATABASE:
            return self._simulate_database_response(action, params)
        else:
            return {"status": "success", "data": None}
    
    def _simulate_api_response(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate API response"""
        # Simulate latency
        time.sleep(random.uniform(0.01, 0.05))
        
        # Simulate occasional failures
        if random.random() < 0.05:  # 5% failure rate
            return {
                "status": "error",
                "error": "API_TIMEOUT",
                "message": "Request timed out"
            }
        
        return {
            "status": "success",
            "data": {"result": f"API response for {action}"},
            "latency_ms": random.randint(50, 200)
        }
    
    def _simulate_user_response(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate user behavior"""
        # Simulate user response time
        time.sleep(random.uniform(0.1, 0.3))
        
        # Simulate user behavior patterns
        behaviors = ["cooperative", "confused", "adversarial", "busy"]
        behavior = random.choice(behaviors)
        
        return {
            "status": "success",
            "response": f"User {behavior} response to {action}",
            "behavior": behavior
        }
    
    def _simulate_database_response(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate database operation"""
        time.sleep(random.uniform(0.02, 0.08))
        
        return {
            "status": "success",
            "affected_rows": random.randint(0, 10),
            "query_time_ms": random.randint(20, 100)
        }


@dataclass
class TestScenario:
    """A test scenario configuration"""
    scenario_id: str
    name: str
    description: str
    initial_state: Dict[str, Any]
    entities: List[SimulatedEntity]
    expected_outcomes: List[Dict[str, Any]]
    timeout_seconds: float = 60.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of a test scenario execution"""
    scenario_id: str
    success: bool
    duration_seconds: float
    outcomes_achieved: List[Dict[str, Any]]
    outcomes_missed: List[Dict[str, Any]]
    interactions_log: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class MockAPI:
    """Mock API for testing"""
    
    def __init__(self, name: str, failure_rate: float = 0.0):
        self.name = name
        self.failure_rate = failure_rate
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []
    
    def call(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Simulate API call"""
        self.call_count += 1
        
        call_record = {
            "timestamp": time.time(),
            "endpoint": endpoint,
            "params": kwargs
        }
        self.call_history.append(call_record)
        
        # Simulate failure
        if random.random() < self.failure_rate:
            return {
                "success": False,
                "error": "API_ERROR",
                "message": f"Mock API {self.name} failed"
            }
        
        # Simulate latency
        time.sleep(random.uniform(0.01, 0.05))
        
        return {
            "success": True,
            "data": {
                "endpoint": endpoint,
                "result": f"Mock response from {self.name}"
            }
        }
    
    def reset(self) -> None:
        """Reset mock API state"""
        self.call_count = 0
        self.call_history.clear()


class SyntheticUser:
    """Synthetic user for testing"""
    
    def __init__(self, user_id: str, behavior_profile: str = "normal"):
        self.user_id = user_id
        self.behavior_profile = behavior_profile
        self.interaction_history: List[Dict[str, Any]] = []
    
    def respond_to(self, agent_action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate user response based on behavior profile"""
        response_data = {
            "user_id": self.user_id,
            "timestamp": time.time(),
            "action": agent_action,
            "context": context
        }
        
        if self.behavior_profile == "cooperative":
            response_data["response"] = "User provides helpful information"
            response_data["satisfaction"] = random.uniform(0.7, 1.0)
        
        elif self.behavior_profile == "confused":
            response_data["response"] = "User asks for clarification"
            response_data["satisfaction"] = random.uniform(0.3, 0.6)
            response_data["needs_help"] = True
        
        elif self.behavior_profile == "adversarial":
            response_data["response"] = "User provides misleading information"
            response_data["satisfaction"] = random.uniform(0.0, 0.4)
            response_data["adversarial"] = True
        
        else:  # normal
            response_data["response"] = "User responds normally"
            response_data["satisfaction"] = random.uniform(0.5, 0.8)
        
        self.interaction_history.append(response_data)
        return response_data


class EnvironmentSimulator:
    """Simulates environmental conditions"""
    
    def __init__(self):
        self.current_state: Dict[str, Any] = {
            "network_latency": 50,  # ms
            "server_load": 0.5,  # 0-1
            "error_rate": 0.01,  # 1%
            "time_of_day": "morning"
        }
        self.state_history: List[Dict[str, Any]] = []
    
    def update_conditions(self, changes: Dict[str, Any]) -> None:
        """Update environmental conditions"""
        self.state_history.append(self.current_state.copy())
        self.current_state.update(changes)
    
    def simulate_network_issues(self) -> bool:
        """Simulate network problems"""
        return random.random() < self.current_state["error_rate"]
    
    def get_latency(self) -> float:
        """Get current network latency"""
        base_latency = self.current_state["network_latency"]
        variation = random.uniform(-10, 10)
        return max(0, base_latency + variation) / 1000  # Convert to seconds
    
    def simulate_high_load(self) -> None:
        """Simulate high server load"""
        self.update_conditions({
            "server_load": 0.9,
            "network_latency": 200,
            "error_rate": 0.1
        })
    
    def simulate_normal_conditions(self) -> None:
        """Reset to normal conditions"""
        self.update_conditions({
            "server_load": 0.5,
            "network_latency": 50,
            "error_rate": 0.01
        })


class SimulationTestingFramework:
    """
    Simulation Testing Framework
    
    Comprehensive testing framework for agents using simulated
    environments, synthetic users, and mock services.
    """
    
    def __init__(self, name: str = "Agent Simulation"):
        self.name = name
        self.scenarios: Dict[str, TestScenario] = {}
        self.results: List[TestResult] = []
        self.mock_apis: Dict[str, MockAPI] = {}
        self.synthetic_users: Dict[str, SyntheticUser] = {}
        self.environment = EnvironmentSimulator()
        self.state = SimulationState.CREATED
    
    def register_scenario(self, scenario: TestScenario) -> None:
        """Register a test scenario"""
        print(f"\n[Scenario] Registered: {scenario.name}")
        print(f"  ID: {scenario.scenario_id}")
        print(f"  Entities: {len(scenario.entities)}")
        print(f"  Expected outcomes: {len(scenario.expected_outcomes)}")
        
        self.scenarios[scenario.scenario_id] = scenario
    
    def add_mock_api(self, name: str, failure_rate: float = 0.0) -> MockAPI:
        """Add a mock API to the simulation"""
        mock = MockAPI(name, failure_rate)
        self.mock_apis[name] = mock
        print(f"\n[Mock API] Added: {name} (failure_rate: {failure_rate:.1%})")
        return mock
    
    def add_synthetic_user(self, user_id: str, behavior: str = "normal") -> SyntheticUser:
        """Add a synthetic user"""
        user = SyntheticUser(user_id, behavior)
        self.synthetic_users[user_id] = user
        print(f"\n[Synthetic User] Added: {user_id} (behavior: {behavior})")
        return user
    
    def run_scenario(
        self,
        scenario_id: str,
        agent_fn: Callable[[TestScenario, Dict[str, Any]], List[Dict[str, Any]]]
    ) -> TestResult:
        """
        Run a test scenario
        
        Args:
            scenario_id: ID of scenario to run
            agent_fn: Function that executes agent behavior
                     Takes (scenario, resources) and returns list of outcomes
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        scenario = self.scenarios[scenario_id]
        
        print(f"\n{'=' * 70}")
        print(f"RUNNING SCENARIO: {scenario.name}")
        print(f"{'=' * 70}")
        
        self.state = SimulationState.RUNNING
        start_time = time.time()
        
        # Prepare resources for agent
        resources = {
            "mock_apis": self.mock_apis,
            "synthetic_users": self.synthetic_users,
            "environment": self.environment,
            "entities": {e.entity_id: e for e in scenario.entities}
        }
        
        # Initialize test state
        interactions_log: List[Dict[str, Any]] = []
        errors: List[str] = []
        
        try:
            # Execute agent behavior
            outcomes = agent_fn(scenario, resources)
            
            # Collect interaction logs
            for entity in scenario.entities:
                if entity.interaction_count > 0:
                    interactions_log.append({
                        "entity_id": entity.entity_id,
                        "entity_type": entity.entity_type.value,
                        "interaction_count": entity.interaction_count
                    })
            
            # Check expected outcomes
            outcomes_achieved = []
            outcomes_missed = []
            
            for expected in scenario.expected_outcomes:
                achieved = any(
                    self._matches_outcome(outcome, expected)
                    for outcome in outcomes
                )
                
                if achieved:
                    outcomes_achieved.append(expected)
                else:
                    outcomes_missed.append(expected)
            
            success = len(outcomes_missed) == 0
            
        except Exception as e:
            errors.append(str(e))
            outcomes_achieved = []
            outcomes_missed = scenario.expected_outcomes
            success = False
        
        duration = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            "total_interactions": sum(e.interaction_count for e in scenario.entities),
            "api_calls": sum(api.call_count for api in self.mock_apis.values()),
            "user_interactions": len([u for u in self.synthetic_users.values() 
                                     if u.interaction_history]),
            "duration_seconds": duration
        }
        
        result = TestResult(
            scenario_id=scenario_id,
            success=success,
            duration_seconds=duration,
            outcomes_achieved=outcomes_achieved,
            outcomes_missed=outcomes_missed,
            interactions_log=interactions_log,
            errors=errors,
            metrics=metrics
        )
        
        self.results.append(result)
        self.state = SimulationState.COMPLETED
        
        # Print results
        self._print_result(result)
        
        return result
    
    def _matches_outcome(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Check if actual outcome matches expected"""
        for key, value in expected.items():
            if key not in actual or actual[key] != value:
                return False
        return True
    
    def _print_result(self, result: TestResult) -> None:
        """Print test result"""
        print(f"\n{'=' * 70}")
        print(f"TEST RESULT: {'✓ PASSED' if result.success else '✗ FAILED'}")
        print(f"{'=' * 70}")
        
        print(f"\nDuration: {result.duration_seconds:.2f}s")
        print(f"Outcomes achieved: {len(result.outcomes_achieved)}/{len(result.outcomes_achieved) + len(result.outcomes_missed)}")
        
        if result.outcomes_missed:
            print(f"\nMissed outcomes:")
            for outcome in result.outcomes_missed:
                print(f"  - {outcome}")
        
        if result.errors:
            print(f"\nErrors:")
            for error in result.errors:
                print(f"  - {error}")
        
        print(f"\nMetrics:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")
    
    def run_all_scenarios(
        self,
        agent_fn: Callable[[TestScenario, Dict[str, Any]], List[Dict[str, Any]]]
    ) -> List[TestResult]:
        """Run all registered scenarios"""
        results = []
        
        for scenario_id in self.scenarios:
            result = self.run_scenario(scenario_id, agent_fn)
            results.append(result)
            
            # Reset state between scenarios
            self._reset_simulation()
        
        self._print_summary(results)
        return results
    
    def _reset_simulation(self) -> None:
        """Reset simulation state between scenarios"""
        for api in self.mock_apis.values():
            api.reset()
        
        self.synthetic_users.clear()
        self.environment.simulate_normal_conditions()
    
    def _print_summary(self, results: List[TestResult]) -> None:
        """Print test summary"""
        print(f"\n{'=' * 70}")
        print("TEST SUMMARY")
        print(f"{'=' * 70}")
        
        total = len(results)
        passed = sum(1 for r in results if r.success)
        failed = total - passed
        
        print(f"\nTotal scenarios: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        
        total_duration = sum(r.duration_seconds for r in results)
        print(f"\nTotal duration: {total_duration:.2f}s")
        print(f"Average duration: {total_duration/total:.2f}s")


def demonstrate_simulation_testing():
    """
    Demonstrate Simulation Testing pattern
    """
    print("=" * 70)
    print("SIMULATION TESTING PATTERN DEMONSTRATION")
    print("=" * 70)
    
    # Create testing framework
    framework = SimulationTestingFramework("Agent Test Suite")
    
    # Example 1: API Integration Testing
    print("\n" + "=" * 70)
    print("EXAMPLE 1: API Integration Testing")
    print("=" * 70)
    
    # Add mock APIs
    weather_api = framework.add_mock_api("weather_api", failure_rate=0.1)
    database_api = framework.add_mock_api("database", failure_rate=0.05)
    
    # Create test scenario
    scenario1 = TestScenario(
        scenario_id="api_integration_001",
        name="Weather Query with Retry",
        description="Test agent handling of API failures with retry logic",
        initial_state={"query": "What's the weather in Paris?"},
        entities=[
            SimulatedEntity("weather_api", EntityType.API, "retry_on_failure"),
            SimulatedEntity("database", EntityType.DATABASE, "persistent")
        ],
        expected_outcomes=[
            {"type": "api_call", "api": "weather_api", "success": True},
            {"type": "data_stored", "database": "database"}
        ]
    )
    
    framework.register_scenario(scenario1)
    
    # Example agent function
    def weather_agent(scenario: TestScenario, resources: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple agent that queries weather API"""
        outcomes = []
        
        # Try to call weather API with retry
        max_retries = 3
        for attempt in range(max_retries):
            result = resources["mock_apis"]["weather_api"].call("/weather", city="Paris")
            
            if result["success"]:
                outcomes.append({
                    "type": "api_call",
                    "api": "weather_api",
                    "success": True,
                    "attempt": attempt + 1
                })
                
                # Store in database
                db_result = resources["mock_apis"]["database"].call("/store", data=result["data"])
                if db_result["success"]:
                    outcomes.append({
                        "type": "data_stored",
                        "database": "database"
                    })
                break
        
        return outcomes
    
    # Run scenario
    result1 = framework.run_scenario("api_integration_001", weather_agent)
    
    # Example 2: User Interaction Testing
    print("\n" + "=" * 70)
    print("EXAMPLE 2: User Interaction Testing")
    print("=" * 70)
    
    # Add synthetic users
    framework.add_synthetic_user("user_001", behavior="cooperative")
    framework.add_synthetic_user("user_002", behavior="confused")
    framework.add_synthetic_user("user_003", behavior="adversarial")
    
    scenario2 = TestScenario(
        scenario_id="user_interaction_001",
        name="Multi-User Support",
        description="Test agent handling different user behaviors",
        initial_state={"mode": "customer_support"},
        entities=[
            SimulatedEntity("user_001", EntityType.USER, "cooperative"),
            SimulatedEntity("user_002", EntityType.USER, "confused"),
            SimulatedEntity("user_003", EntityType.USER, "adversarial")
        ],
        expected_outcomes=[
            {"type": "user_satisfied", "user_id": "user_001"},
            {"type": "clarification_provided", "user_id": "user_002"},
            {"type": "escalated", "user_id": "user_003"}
        ]
    )
    
    framework.register_scenario(scenario2)
    
    def support_agent(scenario: TestScenario, resources: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Agent handling multiple user types"""
        outcomes = []
        
        for user_id, user in resources["synthetic_users"].items():
            response = user.respond_to("greeting", {})
            
            if response.get("satisfaction", 0) > 0.7:
                outcomes.append({"type": "user_satisfied", "user_id": user_id})
            elif response.get("needs_help"):
                outcomes.append({"type": "clarification_provided", "user_id": user_id})
            elif response.get("adversarial"):
                outcomes.append({"type": "escalated", "user_id": user_id})
        
        return outcomes
    
    result2 = framework.run_scenario("user_interaction_001", support_agent)


def demonstrate_advanced_scenarios():
    """Demonstrate advanced testing scenarios"""
    print("\n" + "=" * 70)
    print("ADVANCED SIMULATION SCENARIOS")
    print("=" * 70)
    
    print("\n1. STRESS TESTING:")
    print("   - Simulate high load conditions")
    print("   - Test performance under pressure")
    print("   - Identify breaking points")
    
    print("\n2. FAILURE SCENARIO TESTING:")
    print("   - Network failures")
    print("   - API timeouts")
    print("   - Database outages")
    print("   - Cascading failures")
    
    print("\n3. EDGE CASE TESTING:")
    print("   - Unusual input combinations")
    print("   - Boundary conditions")
    print("   - Race conditions")
    print("   - State inconsistencies")
    
    print("\n4. INTEGRATION TESTING:")
    print("   - Multi-agent coordination")
    print("   - Cross-service communication")
    print("   - End-to-end workflows")
    
    print("\n5. REGRESSION TESTING:")
    print("   - Automated test suites")
    print("   - Version comparison")
    print("   - Performance regression detection")


if __name__ == "__main__":
    demonstrate_simulation_testing()
    demonstrate_advanced_scenarios()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Simulation testing enables safe, reproducible agent testing
2. Mock APIs and synthetic users provide controlled test environments
3. Environment simulators test under various conditions
4. Comprehensive coverage of edge cases and failures
5. Cost-effective and parallel testing capabilities

Best Practices:
- Design realistic test scenarios
- Cover both happy paths and failure cases
- Use synthetic users with diverse behaviors
- Simulate realistic network conditions
- Track comprehensive metrics
- Automate regression testing
- Test edge cases systematically
- Maintain test scenario library
    """)
