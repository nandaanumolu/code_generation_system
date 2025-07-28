"""
Gemini Code Generation Agent
Specializes in clean, efficient code with Google Cloud best practices
"""

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
import sys
from pathlib import Path

# Add shared to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import all our shared tools
from shared.tools import (
    analyze_code,
    wrap_code_in_markdown,
    add_line_numbers,
    extract_functions,
    estimate_complexity,
    validate_python_syntax,
    format_code_output,
    clean_code_string
)

from shared.guardrails import (
    validate_output_code,
    check_code_safety
)


# Create the Gemini code generation agent with full toolset
root_agent = LlmAgent(
    name="gemini_agent",
    model="gemini-2.0-flash",
    description="Gemini-powered code generation expert with comprehensive analysis tools",
    instruction="""You are a Gemini-powered code generation expert. You have complete authority to assess code quality and make decisions about the generated code.

Your strengths:
1. Writing clean, efficient, and well-structured code
2. Following Google's style guides and best practices
3. Creating scalable solutions
4. Comprehensive code analysis and quality assessment

Available tools:
- analyze_code: Get detailed metrics (lines, functions, classes, complexity, etc.)
- estimate_complexity: Analyze cyclomatic complexity and nesting
- validate_python_syntax: Check if Python code is syntactically valid
- extract_functions: Extract function signatures from code
- validate_output_code: Security and safety validation
- check_code_safety: Quick safety check
- add_line_numbers: Add line numbers for review
- format_code_output: Format code with metadata
- wrap_code_in_markdown: Wrap code in markdown blocks
- clean_code_string: Clean and normalize code

Your process:
1. Understand the requirements thoroughly
2. Generate the code solution
3. YOU decide which tools to use for analysis - you have full authority
4. Assess the code quality based on YOUR standards
5. If the code doesn't meet YOUR quality standards, regenerate it
6. Provide detailed analysis and reasoning

Quality standards you enforce:
- Clean, readable code with descriptive names
- Proper error handling
- Type hints for Python
- Comprehensive docstrings
- Reasonable complexity (cyclomatic complexity < 10)
- No security vulnerabilities
- Efficient algorithms

You have COMPLETE AUTHORITY to:
- Reject and regenerate code that doesn't meet standards
- Decide which analysis tools to use
- Set quality thresholds
- Make architectural decisions
- Choose implementation approaches

Always provide:
1. The generated code
2. Your quality assessment
3. Reasoning for your decisions
4. Any concerns or recommendations""",
    tools=[
        FunctionTool(analyze_code),
        FunctionTool(estimate_complexity),
        FunctionTool(validate_python_syntax),
        FunctionTool(extract_functions),
        FunctionTool(validate_output_code),
        FunctionTool(check_code_safety),
        FunctionTool(add_line_numbers),
        FunctionTool(format_code_output),
        FunctionTool(wrap_code_in_markdown),
        FunctionTool(clean_code_string)
    ]
)


# Test function for standalone testing
if __name__ == "__main__":
    print("🚀 Gemini Code Generation Agent Ready!")
    print("\nThis agent has FULL AUTHORITY over code quality decisions")
    print("\nAvailable tools:")
    print("- Code analysis (metrics, complexity, syntax)")
    print("- Security validation")
    print("- Code formatting and cleanup")
    print("- Function extraction and analysis")
    
    print("\nThe agent will:")
    print("- Generate code based on requirements")
    print("- Analyze it using appropriate tools")
    print("- Make quality decisions autonomously")
    print("- Regenerate if standards aren't met")
    
    print("\nTo test interactively:")
    print("1. Run: adk run agents/gemini_agent")
    print("2. Or use: adk web (and select gemini_agent)")
    
    print("\nExample prompts to try:")
    print("- 'Create a secure user authentication class'")
    print("- 'Build a rate limiter with proper error handling'")
    print("- 'Generate a data validation module with full analysis'")