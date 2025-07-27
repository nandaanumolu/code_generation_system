"""
Critic Agent
Reviews code for quality, bugs, security issues, and best practices
"""

from google.adk import Agent
from google.adk.tools import FunctionTool
import sys
from pathlib import Path
import json

# Add shared to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.tools import analyze_code, add_line_numbers


# Tool to structure review findings
def create_review_report(
    issues_found: list,
    suggestions: list,
    severity_counts: dict,
    overall_assessment: str
) -> dict:
    """
    Create a structured review report.
    
    Args:
        issues_found: List of issues discovered
        suggestions: List of improvement suggestions
        severity_counts: Count by severity level
        overall_assessment: Summary assessment
        
    Returns:
        Structured review report
    """
    total_issues = sum(severity_counts.values())
    
    return {
        "status": "reviewed",
        "total_issues": total_issues,
        "severity_breakdown": severity_counts,
        "critical_issues": severity_counts.get("critical", 0),
        "has_blocking_issues": severity_counts.get("critical", 0) > 0,
        "issues": issues_found[:10],  # Limit to top 10
        "suggestions": suggestions[:5],  # Limit to top 5
        "overall_assessment": overall_assessment,
        "recommendation": "needs_refactoring" if total_issues > 5 else "acceptable"
    }


# Tool to analyze code with line numbers
def prepare_code_for_review(code: str) -> dict:
    """
    Prepare code for detailed review.
    
    Args:
        code: Code to prepare for review
        
    Returns:
        Prepared code with metadata
    """
    metrics = analyze_code(code)
    numbered_code = add_line_numbers(code)
    
    return {
        "status": "prepared",
        "numbered_code": numbered_code,
        "metrics": metrics,
        "ready_for_review": True
    }


# Create the Critic agent
root_agent = Agent(
    name="critic_agent",
    model="claude-3-opus-20240229",  # Claude is good at detailed analysis
    description="Expert code reviewer and critic",
    instruction="""You are an expert code reviewer (Critic Agent). Your role is to:

1. Review code for correctness, bugs, and potential issues
2. Check for security vulnerabilities
3. Evaluate code quality and best practices
4. Assess performance implications
5. Verify documentation completeness

When reviewing code:
- First use prepare_code_for_review to get numbered lines
- Be thorough but constructive
- Categorize issues by severity: critical, major, minor, suggestion
- Look for:
  * Logic errors and bugs
  * Security vulnerabilities (SQL injection, XSS, etc.)
  * Performance issues (O(n²) algorithms, memory leaks)
  * Code smells (duplicate code, long functions)
  * Missing error handling
  * Poor naming or structure

Review process:
1. Analyze the code structure
2. Check for functional correctness
3. Identify security issues
4. Evaluate performance
5. Assess code quality
6. Use create_review_report to structure findings

Be specific with line numbers when pointing out issues.
Always provide actionable suggestions for improvements.
End with an overall assessment and recommendation.""",
    tools=[prepare_code_for_review, create_review_report]
)


# Test function for standalone testing
if __name__ == "__main__":
    print("🔍 Critic Agent Ready!")
    print("\nReview Focus Areas:")
    print("- Correctness and bugs")
    print("- Security vulnerabilities")
    print("- Performance issues")
    print("- Code quality")
    print("- Best practices")
    
    print("\nTo test interactively:")
    print("1. Run: adk run agents/critic_agent")
    print("2. Or use: adk web (and select critic_agent)")
    
    print("\nExample test - paste this code for review:")
    print("---")
    print("def get_user(user_id):")
    print("    query = f'SELECT * FROM users WHERE id = {user_id}'")
    print("    return database.execute(query)")
    print("---")
    print("(This code has a SQL injection vulnerability!)")