"""
Parallel Coordinator Agent
Manages parallel execution of multiple code generation agents
Note: Simplified version that doesn't use ParallelAgent directly
"""

from google.adk import Agent
from google.adk.tools import FunctionTool
import sys
from pathlib import Path

# Add shared to path
sys.path.append(str(Path(__file__).parent.parent.parent))


# Tool to compare and select best code
def compare_generated_codes(results: list) -> dict:
    """
    Compare results from multiple generators and select the best.
    
    Args:
        results: List of generation results
        
    Returns:
        Comparison summary and selection
    """
    if not results:
        return {"status": "error", "message": "No results to compare"}
    
    # Simple selection strategy - could be more sophisticated
    # For now, prefer: 1) Longest code, 2) First result
    best_result = max(results, key=lambda r: len(r.get("code", "")))
    
    return {
        "status": "compared",
        "total_results": len(results),
        "selected_generator": best_result.get("generator", "unknown"),
        "selection_reason": "Most comprehensive implementation",
        "comparison_summary": f"Compared {len(results)} implementations"
    }


# Tool to simulate parallel generation
def simulate_parallel_generation(request: str) -> dict:
    """
    Simulate parallel code generation from multiple agents.
    In production, this would actually call the agents.
    
    Args:
        request: Code generation request
        
    Returns:
        Simulated results from multiple generators
    """
    # This is a placeholder - in real implementation, 
    # we would call gemini_agent, gpt4_agent, claude_gen_agent
    
    return {
        "status": "generated",
        "results": [
            {
                "generator": "gemini_agent",
                "code": f"# Gemini implementation\n# {request}\ndef solution():\n    pass",
                "lines": 4
            },
            {
                "generator": "gpt4_agent", 
                "code": f"# GPT-4 implementation\n# {request}\ndef solution():\n    # TODO: implement\n    pass",
                "lines": 5
            },
            {
                "generator": "claude_agent",
                "code": f"# Claude implementation\n# {request}\ndef solution():\n    '''Docstring'''\n    pass",
                "lines": 5
            }
        ]
    }


# Create the parallel coordinator agent
root_agent = Agent(
    name="parallel_coordinator",
    model="gemini-2.0-flash",
    description="Coordinates parallel code generation from multiple agents",
    instruction="""You are the Parallel Generation Coordinator. Your role is to:

1. Take code generation requests
2. Distribute them to multiple generation agents (simulated for now)
3. Collect and compare results
4. Select the best implementation

When you receive a code generation request:
- Use simulate_parallel_generation to get results from multiple generators
- Use compare_generated_codes to analyze and select the best
- Present the selected code with explanation

Note: In the current implementation, we're simulating the parallel generation.
In production, this would actually call the real agents.

Selection criteria:
1. Completeness - does it fully address requirements?
2. Code quality - is it well-structured?
3. Documentation - are there good comments/docstrings?

Always explain why you selected a particular implementation.""",
    tools=[simulate_parallel_generation, compare_generated_codes]
)


# Test function
if __name__ == "__main__":
    print("🔄 Parallel Coordinator Agent Ready!")
    print("\nThis agent coordinates:")
    print("- Gemini Agent (Google style)")
    print("- GPT-4 Agent (Comprehensive)")  
    print("- Claude Agent (Elegant)")
    
    print("\nNote: Currently using simulated generation")
    print("In production, would call actual agents")
    
    print("\nTo test interactively:")
    print("1. Run: adk run agents/parallel_coordinator")
    print("2. Or use: adk web")
    
    print("\nExample prompt:")
    print("Create a Python function to merge two sorted lists")