"""
Run All Examples - Agentic AI Design Patterns
==============================================
This script runs all example patterns to demonstrate their functionality.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_example(script_name: str) -> bool:
    """Run a single example script"""
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n✅ {script_name} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {script_name} failed with error\n")
        return False
    except FileNotFoundError:
        print(f"\n⚠️  {script_name} not found\n")
        return False


def main():
    """Run all examples"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         Agentic AI Design Patterns - Running All Examples           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    examples = [
        ("01_react_pattern.py", "ReAct (Reasoning + Acting)"),
        ("02_chain_of_thought.py", "Chain-of-Thought Reasoning"),
        ("03_tree_of_thoughts.py", "Tree-of-Thoughts Exploration"),
        ("04_plan_and_execute.py", "Plan-and-Execute Pattern"),
        ("05_self_consistency.py", "Self-Consistency with Voting"),
        ("06_reflexion.py", "Reflexion (Learning from Mistakes)"),
        ("07_multi_agent_patterns.py", "Multi-Agent Patterns"),
        ("08_rag_and_memory.py", "RAG and Memory Management"),
        ("09_safety_and_control.py", "Safety and Control Patterns"),
        ("10_graph_of_thoughts.py", "Graph-of-Thoughts Reasoning"),
        ("11_hierarchical_planning.py", "Hierarchical Planning"),
        ("12_metacognitive_monitoring.py", "Metacognitive Monitoring"),
        ("13_analogical_reasoning.py", "Analogical Reasoning"),
        ("14_least_to_most.py", "Least-to-Most Prompting"),
        ("15_constitutional_ai.py", "Constitutional AI"),
        ("16_chain_of_verification.py", "Chain-of-Verification (CoVe)"),
        ("17_advanced_rag.py", "Advanced RAG with Multi-hop Reasoning"),
        ("18_advanced_memory.py", "Advanced Memory Patterns"),
        ("19_tool_selection.py", "Dynamic Tool Selection"),
    ]
    
    results = []
    
    print("\nAvailable Examples:")
    print("─" * 80)
    for i, (script, description) in enumerate(examples, 1):
        status = "✓" if Path(script).exists() else "✗"
        print(f"{i}. {status} {description:<40} ({script})")
    
    print("\n" + "─" * 80)
    print("\nOptions:")
    print("  Enter a number (1-19) to run a specific example")
    print("  Enter 'all' to run all examples")
    print("  Enter 'q' to quit")
    print()
    
    choice = input("Your choice: ").strip().lower()
    
    if choice == 'q':
        print("\n👋 Goodbye!")
        return
    
    elif choice == 'all':
        print("\n🚀 Running all examples...\n")
        for script, description in examples:
            if Path(script).exists():
                success = run_example(script)
                results.append((description, success))
                input("\nPress Enter to continue to next example...")
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}\n")
        for description, success in results:
            status = "✅" if success else "❌"
            print(f"{status} {description}")
        
        total = len(results)
        successful = sum(1 for _, success in results if success)
        print(f"\nCompleted: {successful}/{total} examples")
    
    elif choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(examples):
            script, description = examples[idx]
            if Path(script).exists():
                run_example(script)
            else:
                print(f"\n⚠️  {script} not found")
        else:
            print(f"\n⚠️  Invalid choice. Please enter a number between 1 and {len(examples)}")
    
    else:
        print("\n⚠️  Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
        sys.exit(0)
