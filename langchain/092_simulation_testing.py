"""
Pattern 092: Simulation Testing

Description:
    Simulation Testing creates controlled, reproducible environments to test agent behavior
    under various scenarios, conditions, and edge cases. Unlike production testing,
    simulations allow you to test dangerous, expensive, or rare scenarios safely and
    repeatedly. This pattern is essential for validating agent behavior before deployment
    and for stress testing system limits.

    Simulation testing enables:
    - Safe testing of dangerous scenarios
    - Reproducible test conditions
    - Stress and load testing
    - Edge case exploration
    - Multi-agent interaction testing
    - Cost-effective experimentation

Components:
    1. Simulation Environment
       - Mock external services (APIs, databases)
       - Configurable conditions (delays, errors, loads)
       - State management and reset
       - Recording and replay capabilities
       - Deterministic execution

    2. Scenario Definition
       - Initial state setup
       - Event sequences
       - Expected outcomes
       - Success criteria
       - Failure conditions

    3. Test Harness
       - Scenario execution engine
       - Result validation
       - Metrics collection
       - Logging and tracing
       - Report generation

    4. Mock Components
       - LLM response mocking
       - Tool/API simulation
       - User interaction simulation
       - External system simulation
       - Network condition simulation

Use Cases:
    1. Safety Testing
       - Dangerous scenarios without risk
       - Edge case handling
       - Failure mode analysis
       - Recovery testing
       - Adversarial inputs

    2. Performance Testing
       - Load testing (many concurrent requests)
       - Stress testing (beyond capacity)
       - Endurance testing (long duration)
       - Spike testing (sudden load changes)
       - Scalability validation

    3. Integration Testing
       - Multi-component interaction
       - External service integration
       - Data flow validation
       - Error propagation
       - State consistency

    4. Regression Testing
       - Validate fixes don't break features
       - Ensure consistent behavior
       - Track performance changes
       - Verify improvements
       - Document behavior changes

    5. Exploratory Testing
       - Discover edge cases
       - Test rare scenarios
       - Validate assumptions
       - Find unexpected behaviors
       - Stress system boundaries

LangChain Implementation:
    LangChain supports simulation through:
    - Mock LLM implementations
    - Custom callbacks for control
    - Test chain compositions
    - Fake embeddings for testing
    - Memory simulation

Key Features:
    1. Controllable Environment
       - Deterministic responses
       - Configurable delays
       - Error injection
       - State manipulation
       - Reproducible conditions

    2. Comprehensive Scenarios
       - Normal operations
       - Error conditions
       - Edge cases
       - Load patterns
       - Attack scenarios

    3. Detailed Observation
       - Action logging
       - State tracking
       - Performance metrics
       - Error recording
       - Decision paths

    4. Flexible Validation
       - Multiple success criteria
       - Partial success handling
       - Performance thresholds
       - State assertions
       - Behavior verification

Best Practices:
    1. Scenario Design
       - Start simple, add complexity
       - Cover happy path first
       - Add error scenarios
       - Include edge cases
       - Test boundaries

    2. Mock Fidelity
       - Balance realism vs simplicity
       - Accurate error simulation
       - Realistic timing
       - State consistency
       - Clear documentation

    3. Test Independence
       - Each test isolated
       - Clean state between tests
       - No shared dependencies
       - Deterministic execution
       - Parallel-safe

    4. Performance
       - Fast execution (< 1s per test)
       - Efficient mocking
       - Minimal overhead
       - Parallel execution
       - Resource cleanup

Trade-offs:
    Advantages:
    - Safe testing of dangerous scenarios
    - Reproducible conditions
    - Cost-effective (no real API calls)
    - Fast execution
    - Comprehensive coverage
    - Early issue detection

    Disadvantages:
    - Mock/reality gap
    - Maintenance overhead
    - Complexity in setup
    - May miss real-world issues
    - Requires good scenario design
    - Can give false confidence

Production Considerations:
    1. Test Coverage
       - Normal operation paths
       - Common error scenarios
       - Known edge cases
       - Performance limits
       - Security boundaries

    2. Realism Balance
       - Accurate enough for confidence
       - Simple enough to maintain
       - Fast enough for CI/CD
       - Flexible for experimentation
       - Close to production behavior

    3. Maintenance
       - Update with code changes
       - Sync with production behavior
       - Regular review
       - Refactor as needed
       - Document assumptions

    4. Integration with CI/CD
       - Automated test runs
       - Fast feedback loops
       - Quality gates
       - Regression prevention
       - Performance baselines

    5. Hybrid Approach
       - Simulation for development
       - Integration tests for validation
       - Shadow testing in production
       - A/B testing for changes
       - Monitoring in production
"""

import os
import time
import random
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class ScenarioType(Enum):
    """Types of test scenarios"""
    NORMAL = "normal"
    ERROR = "error"
    EDGE_CASE = "edge_case"
    LOAD = "load"
    ADVERSARIAL = "adversarial"


class OutcomeType(Enum):
    """Test outcome types"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class MockResponse:
    """Represents a mock LLM response"""
    content: str
    delay: float = 0.0  # Simulated delay in seconds
    should_error: bool = False
    error_message: str = ""


@dataclass
class SimulationScenario:
    """Defines a simulation test scenario"""
    id: str
    name: str
    description: str
    scenario_type: ScenarioType
    initial_state: Dict[str, Any]
    events: List[Dict[str, Any]]
    expected_outcomes: List[str]
    success_criteria: Callable[[Dict[str, Any]], bool]


@dataclass
class SimulationResult:
    """Result of a simulation test"""
    scenario_id: str
    outcome: OutcomeType
    duration: float
    events_executed: int
    success_criteria_met: bool
    observations: List[str]
    metrics: Dict[str, Any]
    errors: List[str] = field(default_factory=list)


class MockLLM:
    """
    Mock LLM for testing that returns pre-configured responses.
    
    Allows testing without making actual API calls.
    """
    
    def __init__(self, responses: Optional[Dict[str, MockResponse]] = None):
        """
        Initialize mock LLM.
        
        Args:
            responses: Dictionary mapping inputs to mock responses
        """
        self.responses = responses or {}
        self.default_response = MockResponse(
            content="This is a mock response",
            delay=0.1
        )
        self.call_count = 0
        self.call_history: List[Dict[str, Any]] = []
    
    def invoke(self, input: str) -> str:
        """
        Mock invoke method.
        
        Args:
            input: Input string
            
        Returns:
            Mock response
        """
        self.call_count += 1
        self.call_history.append({
            "input": input,
            "timestamp": datetime.now()
        })
        
        # Find matching response
        response = self.responses.get(input, self.default_response)
        
        # Simulate delay
        if response.delay > 0:
            time.sleep(response.delay)
        
        # Simulate error
        if response.should_error:
            raise Exception(response.error_message or "Mock error")
        
        return response.content
    
    def add_response(self, input: str, response: MockResponse):
        """Add a response pattern"""
        self.responses[input] = response
    
    def reset(self):
        """Reset mock state"""
        self.call_count = 0
        self.call_history.clear()


class MockTool:
    """
    Mock tool for testing agent tool use.
    
    Simulates external tools without actual execution.
    """
    
    def __init__(
        self,
        name: str,
        behavior: Callable[[str], str],
        should_fail: bool = False,
        failure_rate: float = 0.0
    ):
        """
        Initialize mock tool.
        
        Args:
            name: Tool name
            behavior: Function defining tool behavior
            should_fail: Whether tool should always fail
            failure_rate: Probability of failure (0-1)
        """
        self.name = name
        self.behavior = behavior
        self.should_fail = should_fail
        self.failure_rate = failure_rate
        self.call_count = 0
    
    def execute(self, input: str) -> str:
        """
        Execute tool with mock behavior.
        
        Args:
            input: Tool input
            
        Returns:
            Tool output
        """
        self.call_count += 1
        
        # Simulate failure
        if self.should_fail or random.random() < self.failure_rate:
            raise Exception(f"{self.name} failed")
        
        return self.behavior(input)


class SimulationEnvironment:
    """
    Controlled environment for simulation testing.
    
    Provides mock components and state management.
    """
    
    def __init__(self):
        """Initialize simulation environment"""
        self.mock_llm = MockLLM()
        self.mock_tools: Dict[str, MockTool] = {}
        self.state: Dict[str, Any] = {}
        self.event_log: List[Dict[str, Any]] = []
    
    def setup_mock_llm(self, responses: Dict[str, MockResponse]):
        """Configure mock LLM responses"""
        self.mock_llm = MockLLM(responses)
    
    def add_mock_tool(self, tool: MockTool):
        """Add a mock tool"""
        self.mock_tools[tool.name] = tool
    
    def set_state(self, key: str, value: Any):
        """Set environment state"""
        self.state[key] = value
    
    def get_state(self, key: str) -> Any:
        """Get environment state"""
        return self.state.get(key)
    
    def log_event(self, event_type: str, details: Dict[str, Any]):
        """Log an event"""
        self.event_log.append({
            "timestamp": datetime.now(),
            "type": event_type,
            "details": details
        })
    
    def reset(self):
        """Reset environment to initial state"""
        self.mock_llm.reset()
        for tool in self.mock_tools.values():
            tool.call_count = 0
        self.state.clear()
        self.event_log.clear()


class SimulationTester:
    """
    Executes and validates simulation tests.
    
    Runs scenarios in controlled environment and collects results.
    """
    
    def __init__(self, environment: SimulationEnvironment):
        """
        Initialize tester.
        
        Args:
            environment: Simulation environment
        """
        self.environment = environment
    
    def run_scenario(self, scenario: SimulationScenario) -> SimulationResult:
        """
        Run a simulation scenario.
        
        Args:
            scenario: Scenario to run
            
        Returns:
            Simulation result
        """
        # Reset environment
        self.environment.reset()
        
        # Setup initial state
        for key, value in scenario.initial_state.items():
            self.environment.set_state(key, value)
        
        start_time = time.time()
        observations = []
        errors = []
        events_executed = 0
        
        try:
            # Execute events
            for event in scenario.events:
                event_type = event.get("type")
                
                if event_type == "llm_call":
                    input_text = event.get("input", "")
                    try:
                        response = self.environment.mock_llm.invoke(input_text)
                        observations.append(f"LLM called: {input_text[:50]}...")
                        observations.append(f"LLM response: {response[:50]}...")
                        events_executed += 1
                    except Exception as e:
                        errors.append(f"LLM error: {str(e)}")
                
                elif event_type == "tool_call":
                    tool_name = event.get("tool")
                    tool_input = event.get("input", "")
                    
                    if tool_name in self.environment.mock_tools:
                        try:
                            result = self.environment.mock_tools[tool_name].execute(tool_input)
                            observations.append(f"Tool {tool_name} called")
                            observations.append(f"Tool result: {result[:50]}...")
                            events_executed += 1
                        except Exception as e:
                            errors.append(f"Tool error: {str(e)}")
                
                elif event_type == "state_update":
                    key = event.get("key")
                    value = event.get("value")
                    self.environment.set_state(key, value)
                    observations.append(f"State updated: {key}={value}")
                    events_executed += 1
                
                # Log event
                self.environment.log_event(event_type, event)
            
            # Evaluate success criteria
            success = scenario.success_criteria(self.environment.state)
            
            # Determine outcome
            if errors:
                outcome = OutcomeType.ERROR
            elif success:
                outcome = OutcomeType.SUCCESS
            elif events_executed < len(scenario.events):
                outcome = OutcomeType.PARTIAL
            else:
                outcome = OutcomeType.FAILURE
            
        except Exception as e:
            errors.append(f"Scenario error: {str(e)}")
            outcome = OutcomeType.ERROR
            success = False
        
        duration = time.time() - start_time
        
        # Collect metrics
        metrics = {
            "llm_calls": self.environment.mock_llm.call_count,
            "tool_calls": sum(t.call_count for t in self.environment.mock_tools.values()),
            "events_logged": len(self.environment.event_log)
        }
        
        return SimulationResult(
            scenario_id=scenario.id,
            outcome=outcome,
            duration=duration,
            events_executed=events_executed,
            success_criteria_met=success,
            observations=observations,
            metrics=metrics,
            errors=errors
        )
    
    def run_scenarios(
        self,
        scenarios: List[SimulationScenario]
    ) -> List[SimulationResult]:
        """
        Run multiple scenarios.
        
        Args:
            scenarios: List of scenarios
            
        Returns:
            List of results
        """
        results = []
        for scenario in scenarios:
            result = self.run_scenario(scenario)
            results.append(result)
        return results


def demonstrate_simulation_testing():
    """Demonstrate simulation testing"""
    print("=" * 80)
    print("SIMULATION TESTING DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Setup Simulation Environment
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Creating Simulation Environment")
    print("=" * 80)
    
    env = SimulationEnvironment()
    
    # Configure mock LLM responses
    env.setup_mock_llm({
        "What is 2+2?": MockResponse(content="4", delay=0.1),
        "What is the capital of France?": MockResponse(content="Paris", delay=0.1),
        "Error test": MockResponse(
            content="",
            should_error=True,
            error_message="Simulated LLM error"
        )
    })
    
    # Add mock tools
    calculator = MockTool(
        name="calculator",
        behavior=lambda x: str(eval(x)),  # Simple eval for demo
        failure_rate=0.0
    )
    env.add_mock_tool(calculator)
    
    search = MockTool(
        name="search",
        behavior=lambda x: f"Search results for: {x}",
        failure_rate=0.0
    )
    env.add_mock_tool(search)
    
    print("\nSimulation environment configured:")
    print(f"  Mock LLM: {len(env.mock_llm.responses)} responses")
    print(f"  Mock Tools: {len(env.mock_tools)} tools")
    print(f"    - calculator")
    print(f"    - search")
    
    # Example 2: Normal Operation Scenario
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Normal Operation Scenario")
    print("=" * 80)
    
    normal_scenario = SimulationScenario(
        id="scenario_001",
        name="Normal Q&A Flow",
        description="Test normal question answering",
        scenario_type=ScenarioType.NORMAL,
        initial_state={"question_count": 0},
        events=[
            {"type": "llm_call", "input": "What is 2+2?"},
            {"type": "state_update", "key": "question_count", "value": 1},
            {"type": "llm_call", "input": "What is the capital of France?"},
            {"type": "state_update", "key": "question_count", "value": 2}
        ],
        success_criteria=lambda state: state.get("question_count", 0) == 2
    )
    
    tester = SimulationTester(env)
    result = tester.run_scenario(normal_scenario)
    
    print(f"\nScenario: {normal_scenario.name}")
    print(f"Type: {normal_scenario.scenario_type.value}")
    print(f"Outcome: {result.outcome.value}")
    print(f"Duration: {result.duration:.3f}s")
    print(f"Events executed: {result.events_executed}/{len(normal_scenario.events)}")
    print(f"Success criteria met: {result.success_criteria_met}")
    print(f"\nMetrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")
    
    # Example 3: Error Handling Scenario
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Error Handling Scenario")
    print("=" * 80)
    
    error_scenario = SimulationScenario(
        id="scenario_002",
        name="LLM Error Recovery",
        description="Test error handling when LLM fails",
        scenario_type=ScenarioType.ERROR,
        initial_state={"error_handled": False},
        events=[
            {"type": "llm_call", "input": "Error test"},
            {"type": "state_update", "key": "error_handled", "value": True}
        ],
        success_criteria=lambda state: state.get("error_handled", False)
    )
    
    error_result = tester.run_scenario(error_scenario)
    
    print(f"\nScenario: {error_scenario.name}")
    print(f"Outcome: {error_result.outcome.value}")
    print(f"Errors encountered: {len(error_result.errors)}")
    for error in error_result.errors:
        print(f"  - {error}")
    
    # Example 4: Tool Usage Scenario
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Tool Usage Scenario")
    print("=" * 80)
    
    tool_scenario = SimulationScenario(
        id="scenario_003",
        name="Calculator Tool Usage",
        description="Test agent using calculator tool",
        scenario_type=ScenarioType.NORMAL,
        initial_state={"calculations_done": 0},
        events=[
            {"type": "tool_call", "tool": "calculator", "input": "10 + 5"},
            {"type": "state_update", "key": "calculations_done", "value": 1},
            {"type": "tool_call", "tool": "calculator", "input": "20 * 3"},
            {"type": "state_update", "key": "calculations_done", "value": 2}
        ],
        success_criteria=lambda state: state.get("calculations_done", 0) == 2
    )
    
    tool_result = tester.run_scenario(tool_scenario)
    
    print(f"\nScenario: {tool_scenario.name}")
    print(f"Outcome: {tool_result.outcome.value}")
    print(f"Tool calls: {tool_result.metrics['tool_calls']}")
    print(f"Success: {tool_result.success_criteria_met}")
    
    print("\nObservations:")
    for obs in tool_result.observations:
        print(f"  {obs}")
    
    # Example 5: Edge Case Scenario
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Edge Case Scenario")
    print("=" * 80)
    
    edge_scenario = SimulationScenario(
        id="scenario_004",
        name="Empty Input Handling",
        description="Test handling of empty inputs",
        scenario_type=ScenarioType.EDGE_CASE,
        initial_state={"edge_case_handled": False},
        events=[
            {"type": "llm_call", "input": ""},
            {"type": "state_update", "key": "edge_case_handled", "value": True}
        ],
        success_criteria=lambda state: state.get("edge_case_handled", False)
    )
    
    edge_result = tester.run_scenario(edge_scenario)
    
    print(f"\nScenario: {edge_scenario.name}")
    print(f"Type: {edge_scenario.scenario_type.value}")
    print(f"Outcome: {edge_result.outcome.value}")
    print(f"Success: {edge_result.success_criteria_met}")
    
    # Example 6: Load Testing Simulation
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Load Testing Simulation")
    print("=" * 80)
    
    print("\nSimulating high load scenario...")
    
    load_scenarios = []
    for i in range(10):
        scenario = SimulationScenario(
            id=f"load_test_{i}",
            name=f"Load Test {i}",
            description="Concurrent request simulation",
            scenario_type=ScenarioType.LOAD,
            initial_state={},
            events=[
                {"type": "llm_call", "input": f"Query {i}"},
            ],
            success_criteria=lambda state: True
        )
        load_scenarios.append(scenario)
    
    start_time = time.time()
    load_results = tester.run_scenarios(load_scenarios)
    total_time = time.time() - start_time
    
    successful = sum(1 for r in load_results if r.outcome == OutcomeType.SUCCESS)
    
    print(f"\nLoad test completed:")
    print(f"  Total scenarios: {len(load_scenarios)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(load_scenarios) - successful}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Avg time per scenario: {total_time/len(load_scenarios):.3f}s")
    
    # Example 7: Comprehensive Test Suite
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Running Comprehensive Test Suite")
    print("=" * 80)
    
    test_suite = [normal_scenario, error_scenario, tool_scenario, edge_scenario]
    
    print(f"\nRunning {len(test_suite)} scenarios...\n")
    
    suite_results = tester.run_scenarios(test_suite)
    
    # Analyze results
    by_outcome = {}
    for result in suite_results:
        outcome = result.outcome.value
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
    
    print("Test Suite Results:")
    print(f"  Total: {len(suite_results)}")
    for outcome, count in by_outcome.items():
        print(f"  {outcome.upper()}: {count}")
    
    success_rate = (by_outcome.get("success", 0) / len(suite_results)) * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    print("\nDetailed Results:")
    for result in suite_results:
        status = "✓" if result.outcome == OutcomeType.SUCCESS else "✗"
        print(f"  {status} {result.scenario_id}: {result.outcome.value}")
        print(f"     Duration: {result.duration:.3f}s")
        print(f"     Events: {result.events_executed}")
        if result.errors:
            print(f"     Errors: {len(result.errors)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SIMULATION TESTING SUMMARY")
    print("=" * 80)
    print("""
Simulation Testing Benefits:
1. Safety: Test dangerous scenarios without risk
2. Reproducibility: Consistent test conditions
3. Speed: Fast execution without API calls
4. Cost: No real API costs during testing
5. Control: Complete control over conditions
6. Coverage: Test rare and edge cases easily

Simulation Components:
1. Mock LLM
   - Pre-configured responses
   - Error simulation
   - Delay simulation
   - Call tracking
   - Deterministic behavior

2. Mock Tools
   - Simulated tool behavior
   - Failure injection
   - Performance simulation
   - Call counting
   - State management

3. Simulation Environment
   - State management
   - Event logging
   - Component coordination
   - Reset capabilities
   - Observation collection

Scenario Types:
1. Normal Operation
   - Happy path testing
   - Expected workflows
   - Common use cases
   - Standard behavior
   - Success criteria validation

2. Error Handling
   - LLM failures
   - Tool failures
   - Network errors
   - Timeout scenarios
   - Recovery mechanisms

3. Edge Cases
   - Boundary conditions
   - Empty inputs
   - Large inputs
   - Unusual patterns
   - Corner cases

4. Load Testing
   - Concurrent requests
   - High volume
   - Resource limits
   - Performance degradation
   - Scalability validation

5. Adversarial
   - Malicious inputs
   - Attack scenarios
   - Security testing
   - Abuse detection
   - Safety boundaries

Best Practices:
1. Scenario Design
   - Clear objectives
   - Measurable outcomes
   - Realistic conditions
   - Comprehensive coverage
   - Maintainable tests

2. Mock Fidelity
   - Accurate behavior
   - Realistic timing
   - Error patterns
   - State consistency
   - Documentation

3. Test Organization
   - Logical grouping
   - Clear naming
   - Independent tests
   - Reusable components
   - Version control

4. Continuous Testing
   - CI/CD integration
   - Automated runs
   - Fast feedback
   - Regression detection
   - Quality gates

When to Use Simulation:
✓ Development phase testing
✓ Dangerous scenario validation
✓ Rare case exploration
✓ Cost-sensitive testing
✓ Performance testing
✓ Edge case discovery
✗ Production validation (use real testing)
✗ API behavior verification (use integration tests)

Limitations:
- Mock/reality gap
- May miss real-world issues
- Requires maintenance
- Setup complexity
- False confidence risk

Complementary Approaches:
1. Simulation (Development)
   - Fast feedback
   - Safe exploration
   - Cost-effective

2. Integration Tests (Pre-production)
   - Real API calls
   - Actual behavior
   - Environment validation

3. Shadow Testing (Production)
   - Real traffic
   - No user impact
   - Behavior comparison

4. A/B Testing (Production)
   - User feedback
   - Real performance
   - Gradual rollout

Production Tips:
- Use simulation for development
- Combine with integration tests
- Validate mocks against reality
- Update scenarios regularly
- Monitor real vs simulated gaps
- Document assumptions
- Review test coverage
- Fast test execution (< 5s per scenario)
- Parallel execution for large suites
- Clear failure messages
""")
    
    print("\n" + "=" * 80)
    print("Pattern 092 (Simulation Testing) demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_simulation_testing()
