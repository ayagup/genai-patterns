# Renumbering Script for Agentic Patterns
# This script renames all pattern files to have unique sequential numbers 01-64

$basePath = "c:\Users\Lenovo\Documents\code\python\agentic_patterns"

# First, rename files to temporary names to avoid conflicts
Write-Host "Step 1: Renaming files to temporary names..."

$renamePairs = @(
    # Move duplicates to temp names first (100+ series)
    @("05_self_consistency.py", "temp_122_self_consistency.py"),
    @("06_reflexion.py", "temp_123_reflexion.py"),
    @("07_multi_agent_patterns.py", "temp_124_multi_agent_patterns.py"),
    @("08_rag_and_memory.py", "temp_125_rag_and_memory.py"),
    @("09_safety_and_control.py", "temp_126_safety_and_control.py"),
    @("10_self_consistency.py", "temp_127_self_consistency.py"),
    @("11_hierarchical_planning.py", "temp_128_hierarchical_planning.py"),
    @("12_metacognitive_monitoring.py", "temp_129_metacognitive_monitoring.py"),
    @("13_workflow_orchestration.py", "temp_130_workflow_orchestration.py"),
    @("14_least_to_most.py", "temp_131_least_to_most.py"),
    @("15_ensemble_agents.py", "temp_132_ensemble_agents.py"),
    @("16_swarm_intelligence.py", "temp_133_swarm_intelligence.py"),
    @("17_state_machine_agent.py", "temp_134_state_machine_agent.py"),
    @("18_monitoring_observability.py", "temp_135_monitoring_observability.py"),
    @("19_tool_selection.py", "temp_136_tool_selection.py"),
    @("23_graph_of_thoughts.py", "temp_137_graph_of_thoughts.py"),
    @("24_hierarchical_planning.py", "temp_138_hierarchical_planning.py"),
    @("26_prompt_chaining.py", "temp_139_prompt_chaining.py"),
    @("27_tool_routing.py", "temp_140_tool_routing.py"),
    @("28_streaming_output.py", "temp_141_streaming_output.py"),
    @("29_metacognitive_monitoring.py", "temp_142_metacognitive_monitoring.py"),
    @("29_semantic_memory_networks.py", "temp_143_semantic_memory_networks.py"),
    @("30_episodic_memory_retrieval.py", "temp_144_episodic_memory_retrieval.py"),
    @("31_memory_consolidation.py", "temp_145_memory_consolidation.py"),
    @("35_fallback_graceful_degradation.py", "temp_146_fallback_graceful_degradation.py"),
    @("36_sandboxing.py", "temp_147_sandboxing.py"),
    @("38_progressive_optimization.py", "temp_148_progressive_optimization.py"),
    @("39_feedback_loops.py", "temp_149_feedback_loops.py"),
    @("39_leader_follower.py", "temp_150_leader_follower.py"),
    @("40_competitive_multi_agent.py", "temp_151_competitive_multi_agent.py"),
    @("40_self_evaluation.py", "temp_152_self_evaluation.py"),
    @("43_multi_criteria_evaluation.py", "temp_153_multi_criteria_evaluation.py"),
    @("43_rate_limiting.py", "temp_154_rate_limiting.py"),
    @("44_benchmark_driven_development.py", "temp_155_benchmark_driven_development.py"),
    @("45_defensive_generation.py", "temp_156_defensive_generation.py"),
    @("50_adversarial_testing.py", "temp_157_adversarial_testing.py"),
    @("52_redundancy_consensus.py", "temp_158_redundancy_consensus.py"),
    @("53_mixture_of_agents.py", "temp_159_mixture_of_agents.py"),
    @("54_agent_specialization_routing.py", "temp_160_agent_specialization_routing.py"),
    @("55_cognitive_architecture.py", "temp_161_cognitive_architecture.py"),
    @("56_blackboard_system.py", "temp_162_blackboard_system.py"),
    @("79_batch_processing.py", "temp_163_batch_processing.py"),
    @("90_load_balancing.py", "temp_164_load_balancing.py")
)

foreach ($pair in $renamePairs) {
    $oldName = Join-Path $basePath $pair[0]
    $newName = Join-Path $basePath $pair[1]
    if (Test-Path $oldName) {
        Rename-Item -Path $oldName -NewName $pair[1]
        Write-Host "  Renamed: $($pair[0]) -> $($pair[1])"
    }
}

Write-Host "`nStep 2: Renaming temporary files to final sequential numbers..."

$finalRenames = @(
    # Now rename temp files to their final numbers
    @("temp_122_self_consistency.py", "22_self_consistency.py"),
    @("temp_123_reflexion.py", "23_reflexion_enhanced.py"),
    @("temp_124_multi_agent_patterns.py", "24_multi_agent_patterns.py"),
    @("temp_125_rag_and_memory.py", "25_rag_and_memory.py"),
    @("temp_126_safety_and_control.py", "26_safety_and_control.py"),
    @("temp_127_self_consistency.py", "27_self_consistency_variant.py"),
    @("temp_128_hierarchical_planning.py", "28_hierarchical_planning.py"),
    @("temp_129_metacognitive_monitoring.py", "29_metacognitive_monitoring.py"),
    @("temp_130_workflow_orchestration.py", "30_workflow_orchestration.py"),
    @("temp_131_least_to_most.py", "31_least_to_most.py"),
    @("temp_132_ensemble_agents.py", "32_ensemble_agents.py"),
    @("temp_133_swarm_intelligence.py", "33_swarm_intelligence.py"),
    @("temp_134_state_machine_agent.py", "34_state_machine_agent.py"),
    @("temp_135_monitoring_observability.py", "35_monitoring_observability.py"),
    @("temp_136_tool_selection.py", "36_tool_selection.py"),
    @("temp_137_graph_of_thoughts.py", "37_graph_of_thoughts_extracted.py"),
    @("temp_138_hierarchical_planning.py", "38_hierarchical_planning_extracted.py"),
    @("temp_139_prompt_chaining.py", "39_prompt_chaining.py"),
    @("temp_140_tool_routing.py", "40_tool_routing.py"),
    @("temp_141_streaming_output.py", "41_streaming_output.py"),
    @("temp_142_metacognitive_monitoring.py", "42_metacognitive_monitoring_variant.py"),
    @("temp_143_semantic_memory_networks.py", "43_semantic_memory_networks.py"),
    @("temp_144_episodic_memory_retrieval.py", "44_episodic_memory_retrieval.py"),
    @("temp_145_memory_consolidation.py", "45_memory_consolidation.py"),
    @("temp_146_fallback_graceful_degradation.py", "46_fallback_graceful_degradation.py"),
    @("temp_147_sandboxing.py", "47_sandboxing.py"),
    @("temp_148_progressive_optimization.py", "48_progressive_optimization.py"),
    @("temp_149_feedback_loops.py", "49_feedback_loops.py"),
    @("temp_150_leader_follower.py", "50_leader_follower.py"),
    @("temp_151_competitive_multi_agent.py", "51_competitive_multi_agent.py"),
    @("temp_152_self_evaluation.py", "52_self_evaluation.py"),
    @("temp_153_multi_criteria_evaluation.py", "53_multi_criteria_evaluation.py"),
    @("temp_154_rate_limiting.py", "54_rate_limiting.py"),
    @("temp_155_benchmark_driven_development.py", "55_benchmark_driven_development.py"),
    @("temp_156_defensive_generation.py", "56_defensive_generation.py"),
    @("temp_157_adversarial_testing.py", "57_adversarial_testing.py"),
    @("temp_158_redundancy_consensus.py", "58_redundancy_consensus.py"),
    @("temp_159_mixture_of_agents.py", "59_mixture_of_agents.py"),
    @("temp_160_agent_specialization_routing.py", "60_agent_specialization_routing.py"),
    @("temp_161_cognitive_architecture.py", "61_cognitive_architecture.py"),
    @("temp_162_blackboard_system.py", "62_blackboard_system.py"),
    @("temp_163_batch_processing.py", "63_batch_processing.py"),
    @("temp_164_load_balancing.py", "64_load_balancing.py")
)

foreach ($pair in $finalRenames) {
    $oldName = Join-Path $basePath $pair[0]
    $newName = Join-Path $basePath $pair[1]
    if (Test-Path $oldName) {
        Rename-Item -Path $oldName -NewName $pair[1]
        Write-Host "  Renamed: $($pair[0]) -> $($pair[1])"
    }
}

Write-Host "`nStep 3: Verification - Listing all pattern files..."
Get-ChildItem -Path $basePath -Filter "*_*.py" | Where-Object { $_.Name -match '^\d{2}_' } | Sort-Object Name | Select-Object Name

Write-Host "`nRenumbering complete! All files now have unique numbers 01-64."
