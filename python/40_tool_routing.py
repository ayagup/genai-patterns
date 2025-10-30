# Usage
if __name__ == "__main__":
    print("="*80)
    print("TOOL ROUTING PATTERN DEMONSTRATION")
    print("="*80)
    agent = ToolRoutingAgent("router-agent-001")
    # Test various queries
    test_queries = [
        "What is 25 * 48?",
        "Search for the latest AI research papers",
        "Translate 'Hello, how are you?' to Spanish",
        "Summarize this article about climate change",
        "Run this Python code: print('Hello World')",
        "What is the capital of France?"
    ]
    print("\nProcessing queries with intelligent routing...\n")
    results = []
    for query in test_queries:
        print("\n" + "="*80)
        result = agent.process_query(query)
        results.append(result)
    # Show routing statistics
    print("\n\n" + "="*80)
    print("ROUTING STATISTICS")
    print("="*80)
    stats = agent.router.get_routing_stats()
    print(f"\nTotal Queries: {stats['total_queries']}")
    print(f"Average Confidence: {stats['avg_confidence']:.1%}")
    print(f"\nTool Usage:")
    for tool, count in stats['tool_usage'].items():
        percentage = (count / stats['total_queries']) * 100
        print(f"  {tool}: {count} ({percentage:.1f}%)")
    # Calculate total cost
    total_cost = sum(r['cost'] for r in results)
    print(f"\nTotal Cost: {total_cost:.2f}")
