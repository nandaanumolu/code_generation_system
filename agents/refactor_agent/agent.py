"""
Refactor Agent
Improves code based on review feedback while maintaining functionality
"""

from google.adk import Agent
from google.adk.tools import FunctionTool
import sys
from pathlib import Path

# Add shared to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.tools import analyze_code, wrap_code_in_markdown


# Tool to track refactoring changes
def document_changes(
    original_code: str,
    refactored_code: str,
    changes_made: list
) -> dict:
    """
    Document what changes were made during refactoring.
    
    Args:
        original_code: Code before refactoring
        refactored_code: Code after refactoring
        changes_made: List of changes implemented
        
    Returns:
        Change documentation
    """
    original_metrics = analyze_code(original_code)
    new_metrics = analyze_code(refactored_code)
    
    return {
        "status": "documented",
        "changes_made": changes_made[:10],  # Limit to 10
        "metrics_before": {
            "lines": original_metrics["lines_total"],
            "has_functions": original_metrics["has_functions"]
        },
        "metrics_after": {
            "lines": new_metrics["lines_total"],
            "has_functions": new_metrics["has_functions"]
        },
        "improvement_summary": f"Refactored {len(changes_made)} issues"
    }


# Tool to validate refactored code
def validate_refactoring(code: str, test_cases: list = None) -> dict:
    """
    Basic validation that refactored code is still valid.
    
    Args:
        code: Refactored code to validate
        test_cases: Optional test cases to consider
        
    Returns:
        Validation results
    """
    metrics = analyze_code(code)
    
    # Basic syntax check (very simple)
    looks_valid = (
        metrics["lines_non_empty"] > 0 and
        not code.strip().endswith(":") and  # Not incomplete
        code.count("(") == code.count(")")  # Balanced parentheses
    )
    
    return {
        "status": "validated",
        "appears_valid": looks_valid,
        "has_structure": metrics["has_functions"] or metrics["has_classes"],
        "line_count": metrics["lines_non_empty"],
        "ready_for_delivery": looks_valid
    }


# Create the Refactor agent
root_agent = Agent(
    name="refactor_agent",
    model="gemini-2.0-flash",  # Fast for refactoring tasks
    description="Code refactoring specialist",
    instruction="""You are a Code Refactoring Specialist. Your role is to:

1. Take code with identified issues
2. Fix the problems while maintaining functionality
3. Improve code quality and structure
4. Document what changes were made

When refactoring code based on review feedback:
- Preserve the original functionality exactly
- Fix critical issues first (security, bugs)
- Then address major issues (performance, structure)
- Finally handle minor issues (style, naming)
- Add comments explaining significant changes

Refactoring priorities:
1. Security vulnerabilities - MUST fix
2. Logic errors and bugs - MUST fix
3. Performance issues - SHOULD fix
4. Code structure - SHOULD improve
5. Style and naming - NICE to fix

Process:
1. Understand the original code's purpose
2. Review the feedback carefully
3. Plan the refactoring approach
4. Implement fixes systematically
5. Use validate_refactoring to check the result
6. Use document_changes to list what was fixed

Always:
- Maintain backward compatibility
- Keep the same function/class interfaces
- Improve without over-engineering
- Add brief comments for non-obvious changes

Wrap the refactored code in markdown code blocks.""",
    tools=[validate_refactoring, document_changes]
)


# Test function for standalone testing
if __name__ == "__main__":
    print("🔧 Refactor Agent Ready!")
    print("\nRefactoring Capabilities:")
    print("- Fix security vulnerabilities")
    print("- Correct logic errors")
    print("- Improve performance")
    print("- Enhance code structure")
    print("- Better naming and style")
    
    print("\nTo test interactively:")
    print("1. Run: adk run agents/refactor_agent")
    print("2. Or use: adk web (and select refactor_agent)")
    
    print("\nExample test prompt:")
    print("---")
    print("Please refactor this code that has a SQL injection vulnerability:")
    print("")
    print("def get_user(user_id):")
    print("    query = f'SELECT * FROM users WHERE id = {user_id}'")
    print("    return database.execute(query)")
    print("")
    print("Issues to fix:")
    print("- SQL injection vulnerability")
    print("- No error handling")
    print("- No input validation")
    print("---")