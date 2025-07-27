"""
GPT-4 Code Generation Agent
Specializes in comprehensive solutions with detailed error handling
"""

from google.adk import Agent
from google.adk.tools import FunctionTool
import sys
from pathlib import Path

# Add shared to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.tools import analyze_code, wrap_code_in_markdown


# Tool to validate code completeness
def check_code_completeness(code: str, requirements: str) -> dict:
    """
    Check if generated code seems complete.
    
    Args:
        code: Generated code
        requirements: Original requirements
        
    Returns:
        Completeness check results
    """
    metrics = analyze_code(code)
    
    return {
        "status": "checked",
        "has_functions": metrics["has_functions"],
        "has_classes": metrics["has_classes"],
        "line_count": metrics["lines_non_empty"],
        "looks_complete": metrics["lines_non_empty"] > 3
    }


# Create the GPT-4 code generation agent  
root_agent = Agent(
    name="gpt4_agent",
    model="gpt-4",  # Will use gpt-4 via litellm
    description="GPT-4 powered code generation expert",
    instruction="""You are a GPT-4 powered code generation expert. Your strengths are:

1. Creating comprehensive, production-ready solutions
2. Implementing robust error handling and edge cases
3. Writing detailed documentation and comments
4. Providing multiple implementation approaches

When generating code:
- Think through edge cases and potential errors
- Add comprehensive error handling
- Include detailed docstrings
- Consider performance implications
- Provide usage examples

For each code generation request:
1. Analyze requirements thoroughly
2. Consider multiple approaches
3. Generate complete, robust code
4. Use check_code_completeness to verify
5. Include usage examples and notes

Always provide well-commented code with clear explanations.
Consider "what could go wrong" and handle those cases.""",
    tools=[check_code_completeness]
)


# Test function for standalone testing
if __name__ == "__main__":
    print("🚀 GPT-4 Code Generation Agent Ready!")
    print("\nSpecialties:")
    print("- Comprehensive solutions")
    print("- Robust error handling")
    print("- Detailed documentation")
    print("- Edge case consideration")
    
    print("\nTo test interactively:")
    print("1. Run: adk run agents/gpt4_agent")
    print("2. Or use: adk web (and select gpt4_agent)")
    
    print("\nExample prompts to try:")
    print("- 'Create a thread-safe cache implementation in Python'")
    print("- 'Write a function to parse CSV files with error handling'")
    print("- 'Build a retry decorator with exponential backoff'")