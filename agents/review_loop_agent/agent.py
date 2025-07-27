"""
Review Loop Agent
Manages the iterative cycle between Critic and Refactor agents
Note: Simplified version
"""

from google.adk import Agent
from google.adk.tools import FunctionTool
import sys
from pathlib import Path

# Add shared to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.tools import analyze_code


# Tool to track loop iterations
def track_iteration(
    iteration: int,
    max_iterations: int,
    has_issues: bool,
    quality_score: float
) -> dict:
    """
    Track the progress of review-refactor iterations.
    
    Args:
        iteration: Current iteration number
        max_iterations: Maximum allowed iterations
        has_issues: Whether issues remain
        quality_score: Current quality score (0-1)
        
    Returns:
        Iteration tracking info
    """
    should_continue = (
        iteration < max_iterations and 
        has_issues and 
        quality_score < 0.9  # Stop if quality is good enough
    )
    
    return {
        "status": "tracked",
        "current_iteration": iteration,
        "max_iterations": max_iterations,
        "should_continue": should_continue,
        "quality_score": quality_score,
        "progress_percentage": (iteration / max_iterations) * 100
    }


# Tool to simulate review and refactor
def simulate_review_refactor(code: str, iteration: int) -> dict:
    """
    Simulate the review-refactor process.
    In production, this would call critic and refactor agents.
    
    Args:
        code: Code to review and refactor
        iteration: Current iteration number
        
    Returns:
        Simulated review and refactor results
    """
    # Simple simulation - in real implementation,
    # we would call critic_agent and refactor_agent
    
    # Simulate finding fewer issues each iteration
    issues_count = max(0, 5 - iteration)
    quality_score = 0.6 + (iteration * 0.15)
    
    # Simulate refactored code
    refactored = code
    if iteration == 1:
        refactored += "\n    # Added error handling"
    elif iteration == 2:
        refactored += "\n    # Added type hints"
    
    return {
        "status": "reviewed_and_refactored",
        "iteration": iteration,
        "issues_found": issues_count,
        "quality_score": min(quality_score, 1.0),
        "has_critical_issues": issues_count > 3,
        "refactored_code": refactored,
        "changes_made": [f"Fixed {issues_count} issues"] if issues_count > 0 else ["No changes needed"]
    }


# Create the review loop agent
root_agent = Agent(
    name="review_loop_agent",
    model="gemini-2.0-flash",
    description="Manages iterative code review and refactoring",
    instruction="""You are the Review Loop Coordinator. Your role is to:

1. Take generated code
2. Manage iterative review and refactoring
3. Track quality improvements
4. Know when to stop

Process for each iteration:
1. Use simulate_review_refactor to review and improve code
2. Use track_iteration to check if we should continue
3. Stop when quality is good enough or max iterations reached

Note: In the current implementation, we're simulating the review-refactor process.
In production, this would actually call the critic and refactor agents.

Exit criteria:
- No critical issues remain AND quality score >= 0.8
- OR maximum 5 iterations reached

Always:
- Track the iteration count
- Monitor quality improvement
- Report what improvements were made

Present the final improved code with a summary of the iterations.""",
    tools=[simulate_review_refactor, track_iteration]
)


# Test function
if __name__ == "__main__":
    print("🔁 Review Loop Agent Ready!")
    print("\nThis agent manages:")
    print("- Iterative code review")
    print("- Refactoring based on issues")
    print("- Quality tracking")
    
    print("\nNote: Currently using simulated review/refactor")
    print("In production, would call actual critic and refactor agents")
    
    print("\nTo test interactively:")
    print("1. Run: adk run agents/review_loop_agent")
    print("2. Or use: adk web")
    
    print("\nExample prompt:")
    print("Review and improve this code: def div(a,b): return a/b")