"""
Delivery Agent
Packages final code with documentation and metadata
"""

from google.adk import Agent
from google.adk.tools import FunctionTool
from datetime import datetime
import json


# Tool for formatting the final output with guardrail
def format_final_code(code: str, metadata: dict) -> dict:
    """
    Format code with metadata and apply output guardrail.
    
    Args:
        code: The final code to deliver
        metadata: Additional metadata about the code
        
    Returns:
        Formatted package ready for delivery with guardrail validation
    """
    # For now, we'll do basic validation without async
    # In production, ADK would handle async properly
    
    # Basic security check (simplified version)
    dangerous_patterns = ['rm -rf', 'os.system', 'eval(', 'exec(']
    security_issues = []
    
    for pattern in dangerous_patterns:
        if pattern in code:
            security_issues.append(pattern)
    
    # If security issues found, comment them out
    final_code = code
    if security_issues:
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if any(pattern in line for pattern in dangerous_patterns):
                lines[i] = f"# SECURITY: {line}"
        final_code = '\n'.join(lines)
    
    # Calculate basic quality score
    lines = final_code.split('\n')
    quality_score = 1.0
    if len(lines) < 2:
        quality_score -= 0.2
    if not any('def ' in line or 'class ' in line for line in lines):
        quality_score -= 0.1
    if 'TODO' in final_code or 'FIXME' in final_code:
        quality_score -= 0.1
    quality_score = max(0.0, quality_score)
    
    # Build response
    response = {
        "status": "success",
        "delivered_at": datetime.now().isoformat(),
        "code": final_code,
        "metadata": metadata,
        "statistics": {
            "total_lines": len(lines),
            "total_characters": len(final_code),
            "non_empty_lines": len([l for l in lines if l.strip()])
        },
        "guardrail_validation": {
            "passed": len(security_issues) == 0,
            "quality_score": quality_score,
            "was_sanitized": final_code != code
        }
    }
    
    # Add warnings if any
    if security_issues:
        response["security_warnings"] = f"Code was sanitized due to security issues: {security_issues}"
    
    return response


# Create the delivery agent
root_agent = Agent(
    name="delivery_agent",
    model="gemini-2.0-flash",
    description="Agent responsible for packaging and delivering final code",
    instruction="""You are the Final Delivery Agent. Your role is to:

1. Take the final approved code and package it nicely
2. Apply output guardrail for security validation
3. Add a brief summary of what the code does
4. Include any important usage notes
5. Format everything for clean presentation

When you receive code to deliver:
- Summarize its purpose in 1-2 sentences
- Note the programming language
- Mention any key features or functions
- Use the format_final_code tool to package it
- The tool will automatically apply security checks

Important: The output guardrail will:
- Check for security issues
- Sanitize dangerous code if needed
- Calculate quality score
- Add validation metadata

If code is sanitized, mention it in your response.
Always report the quality score.
End with: "✅ Code delivered successfully!"
""",
    tools=[format_final_code]
)


# Test function for standalone testing
if __name__ == "__main__":
    print("🚀 Delivery Agent Ready!")
    print("This agent packages final code for delivery.")
    print("\nTo test interactively:")
    print("1. Run: adk run agents/delivery_agent")
    print("2. Or use: adk web (and select delivery_agent)")
    
    # Example of what this agent handles
    example_code = '''def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)'''
    
    print("\nExample input:")
    print(f"Please deliver this code:\n{example_code}")