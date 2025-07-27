"""
Claude Code Generation Agent
Specializes in elegant, well-documented solutions with strong type safety
"""

from google.adk import Agent
from google.adk.tools import FunctionTool
import sys
from pathlib import Path

# Add shared to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.tools import analyze_code, add_line_numbers


# Tool to check code elegance and documentation
def evaluate_code_quality(code: str) -> dict:
    """
    Evaluate code for quality and documentation.
    
    Args:
        code: Generated code to evaluate
        
    Returns:
        Quality evaluation results
    """
    metrics = analyze_code(code)
    
    # Simple checks for Claude's focus areas
    has_type_hints = ": " in code and "->" in code
    has_docstrings = '"""' in code or "'''" in code
    
    return {
        "status": "evaluated",
        "has_type_hints": has_type_hints,
        "has_docstrings": has_docstrings,
        "line_count": metrics["lines_non_empty"],
        "appears_elegant": has_type_hints and has_docstrings
    }


# Create the Claude code generation agent
root_agent = Agent(
    name="claude_gen_agent", 
    model="claude-3-opus-20240229",  # Will use Claude via litellm
    description="Claude-powered code generation expert",
    instruction="""You are a Claude-powered code generation expert. Your strengths are:

1. Writing elegant, pythonic, and idiomatic code
2. Creating comprehensive type hints and annotations
3. Designing clean APIs and interfaces
4. Following SOLID principles and design patterns
5. Writing exceptional documentation

When generating code:
- Prioritize code elegance and readability
- Use descriptive names that tell a story
- Add comprehensive type hints
- Include detailed docstrings with examples
- Consider long-term maintainability

For each code generation request:
1. Think about the most elegant solution
2. Design clean interfaces
3. Generate well-structured code
4. Use evaluate_code_quality to check quality
5. Explain design patterns used

Focus on creating code that is a joy to read and maintain.
Prefer composition over inheritance, and simplicity over complexity.""",
    tools=[evaluate_code_quality]
)


# Test function for standalone testing
if __name__ == "__main__":
    print("🚀 Claude Code Generation Agent Ready!")
    print("\nSpecialties:")
    print("- Elegant, idiomatic code")
    print("- Strong type safety")
    print("- Clean architecture")
    print("- Exceptional documentation")
    print("- Design patterns")
    
    print("\nTo test interactively:")
    print("1. Run: adk run agents/claude_gen_agent")
    print("2. Or use: adk web (and select claude_gen_agent)")
    
    print("\nExample prompts to try:")
    print("- 'Create a type-safe configuration manager class'")
    print("- 'Design a clean API for a payment processing system'")
    print("- 'Implement a builder pattern for complex object creation'")