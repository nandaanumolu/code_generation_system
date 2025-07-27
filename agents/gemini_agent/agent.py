"""
Gemini Code Generation Agent
Specializes in clean, efficient code with Google Cloud best practices
"""

from google.adk import Agent
from google.adk.tools import FunctionTool
import sys
from pathlib import Path

# Add shared to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.tools import analyze_code, wrap_code_in_markdown


# Tool to analyze generated code
def check_code_quality(code: str) -> dict:
    """
    Quick quality check of generated code.
    
    Args:
        code: Generated code to check
        
    Returns:
        Basic quality metrics
    """
    metrics = analyze_code(code)
    
    return {
        "status": "analyzed",
        "metrics": metrics,
        "ready": metrics["lines_non_empty"] > 0
    }


# Create the Gemini code generation agent
root_agent = Agent(
    name="gemini_agent",
    model="gemini-2.0-flash",
    description="Gemini-powered code generation expert",
    instruction="""You are a Gemini-powered code generation expert. Your strengths are:

1. Writing clean, efficient, and well-structured code
2. Following Google's style guides and best practices
3. Creating scalable solutions
4. Adding helpful comments and docstrings

When generating code:
- Focus on clarity and maintainability
- Use descriptive variable names
- Include error handling
- Add type hints for Python code
- Follow PEP 8 style guidelines

For each code generation request:
1. Understand the requirements
2. Plan the structure
3. Generate clean code
4. Use check_code_quality to verify basics
5. Explain key design decisions

Always wrap your code in proper markdown code blocks.
End with a brief explanation of the code's approach.""",
    tools=[check_code_quality]
)


# Test function for standalone testing
if __name__ == "__main__":
    print("🚀 Gemini Code Generation Agent Ready!")
    print("\nSpecialties:")
    print("- Clean, efficient code")
    print("- Google Cloud best practices")
    print("- Scalable solutions")
    print("- Comprehensive documentation")
    
    print("\nTo test interactively:")
    print("1. Run: adk run agents/gemini_agent")
    print("2. Or use: adk web (and select gemini_agent)")
    
    print("\nExample prompts to try:")
    print("- 'Create a Python function to validate email addresses'")
    print("- 'Write a class for managing a task queue'")
    print("- 'Generate a REST API endpoint for user registration'")