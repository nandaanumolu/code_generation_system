#!/usr/bin/env python3
"""
Test memory integration in orchestrator
"""

import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent))
from shared.memory import get_memory_service, MemoryEntry


def test_memory_integration():
    """Test memory service integration."""
    
    print("🧪 Testing Memory Integration...\n")
    
    # Get memory service
    memory = get_memory_service("test_integration.db")
    
    # Pre-populate with some test memories
    print("1️⃣ Pre-populating memory with examples:")
    
    test_memories = [
        {
            "request": "Create a function to calculate fibonacci numbers",
            "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "quality": 0.85
        },
        {
            "request": "Write a function to sort a list",
            "code": "def sort_list(lst):\n    return sorted(lst)",
            "quality": 0.9
        },
        {
            "request": "Create a class for a bank account",
            "code": "class BankAccount:\n    def __init__(self, balance=0):\n        self.balance = balance",
            "quality": 0.8
        }
    ]
    
    for mem in test_memories:
        entry = MemoryEntry(
            category="code_generation",
            agent_name="test_agent",
            data={
                "original_request": mem["request"],
                "generated_code": mem["code"]
            },
            quality_score=mem["quality"],
            tags=["test", "example"]
        )
        memory.store(entry)
        print(f"   ✅ Stored: {mem['request'][:50]}...")
    
    # Test similarity search
    print("\n2️⃣ Testing similarity search:")
    
    test_queries = [
        "Create a fibonacci function",
        "Make a function for sorting",
        "Build a bank account system",
        "Create a web scraper"  # This should not match
    ]
    
    for query in test_queries:
        similar = memory.search_similar(query, threshold=0.5)
        print(f"\n   Query: '{query}'")
        if similar:
            best = similar[0]
            print(f"   ✅ Found match (score: {best.data.get('similarity_score', 0):.2f})")
            print(f"      Original: {best.data.get('original_request', '')}")
        else:
            print(f"   ❌ No similar requests found")
    
    # Test agent history
    print("\n3️⃣ Testing agent history:")
    history = memory.get_agent_history("test_agent", days=1)
    print(f"   Total memories: {history['total_memories']}")
    print(f"   By category: {history['by_category']}")
    
    # Cleanup
    Path("test_integration.db").unlink(missing_ok=True)
    print("\n✅ Memory integration tests completed!")
    
    print("\n📝 How memory works in the orchestrator:")
    print("   1. Check for similar past requests")
    print("   2. Suggest reusing high-quality matches")
    print("   3. Save successful generations")
    print("   4. Learn from past successes")


if __name__ == "__main__":
    test_memory_integration()