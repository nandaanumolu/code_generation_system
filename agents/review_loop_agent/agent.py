# agents/review_loop_agent/agent.py
"""
Review-Refactor Loop Agent using ADK's LoopAgent
Orchestrates iterative code improvement through review and refactoring cycles

This agent:
1. Takes initial code as input
2. Runs critic agent to review the code
3. Runs refactor agent to fix identified issues
4. Loops until code quality meets standards or max iterations reached
"""

from google.adk.agents import Agent, LoopAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator, Dict, Any, List, Optional
import sys
from pathlib import Path
import json
import re

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the critic and refactor agents
try:
    # Import critic agent
    from critic_agent.agent import root_agent as critic_agent_instance
    print("✓ Successfully imported critic agent")
except ImportError as e:
    print(f"Error importing critic agent: {e}")
    critic_agent_instance = None

try:
    # Import refactor agent (adjust the import path based on your folder structure)
    from refactor_agent.agent import root_agent as refactor_agent_instance
    print("✓ Successfully imported refactor agent")
except ImportError as e:
    print(f"Error importing refactor agent: {e}")
    refactor_agent_instance = None

# Import shared tools
from shared.tools import (
    analyze_code,
    wrap_code_in_markdown,
    clean_code_string,
    format_code_output
)


def prepare_code_for_review(code: str) -> Dict[str, Any]:
    """
    Prepare code for the review-refactor loop.
    
    Args:
        code: The code to be reviewed and refactored
        
    Returns:
        Prepared context for the loop
    """
    cleaned_code = clean_code_string(code)
    return {
        "status": "prepared",
        "original_code": code,
        "cleaned_code": cleaned_code,
        "ready_for_loop": True
    }


def extract_review_results(critic_output: str) -> Dict[str, Any]:
    """
    Extract structured review results from critic agent output.
    
    Args:
        critic_output: Raw output from critic agent
        
    Returns:
        Structured review results
    """
    try:
        # Try to parse as JSON first
        if "{" in critic_output and "}" in critic_output:
            # Extract JSON from the output
            json_start = critic_output.find("{")
            json_end = critic_output.rfind("}") + 1
            json_str = critic_output[json_start:json_end]
            result = json.loads(json_str)
            
            # Extract key metrics
            return {
                "has_critical_issues": result.get("review_decision", {}).get("blocking", False),
                "severity_counts": result.get("structured_feedback", {}).get("severity_breakdown", {}),
                "quality_score": result.get("review_analysis", {}).get("overall_quality_score", 0.0),
                "issues": result.get("review_analysis", {}).get("all_issues", []),
                "recommendations": result.get("structured_feedback", {}).get("structured_recommendations", [])
            }
    except:
        pass
    
    # Fallback: Parse text output
    has_critical = bool(re.search(r'critical|CRITICAL|blocking|BLOCK', critic_output, re.IGNORECASE))
    has_major = bool(re.search(r'major|MAJOR|significant', critic_output, re.IGNORECASE))
    
    return {
        "has_critical_issues": has_critical,
        "has_major_issues": has_major,
        "raw_output": critic_output,
        "parsed": False
    }


def extract_refactored_code(refactor_output: str) -> str:
    """
    Extract the refactored code from refactor agent output.
    
    Args:
        refactor_output: Raw output from refactor agent
        
    Returns:
        Extracted code
    """
    # Look for code blocks
    if "```python" in refactor_output:
        match = re.search(r'```python\n(.*?)```', refactor_output, re.DOTALL)
        if match:
            return match.group(1).strip()
    elif "```" in refactor_output:
        match = re.search(r'```\n(.*?)```', refactor_output, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Return as-is if no code block found
    return refactor_output


def create_stop_condition(quality_threshold: float = 0.7) -> callable:
    """
    Create a stop condition function for the loop.
    
    Args:
        quality_threshold: Minimum quality score to stop the loop
        
    Returns:
        Stop condition function
    """
    def should_stop(context: InvocationContext) -> bool:
        """
        Determine if the loop should stop based on the latest review results.
        """
        # Get the last event from context
        last_messages = context.get_messages()
        if not last_messages:
            return False
        
        # Look for critic agent output in recent messages
        for msg in reversed(last_messages[-5:]):  # Check last 5 messages
            if msg.get("author") == "critic_agent":
                content = msg.get("content", {}).get("parts", [{}])[0].get("text", "")
                review_results = extract_review_results(content)
                
                # Stop if no critical issues and quality is good
                if not review_results.get("has_critical_issues", True):
                    severity_counts = review_results.get("severity_counts", {})
                    if severity_counts.get("critical", 0) == 0 and severity_counts.get("major", 0) <= 1:
                        print("✓ Loop stopping: Code quality acceptable")
                        return True
                    
                # Stop if quality score is above threshold
                if review_results.get("quality_score", 0) >= quality_threshold:
                    print(f"✓ Loop stopping: Quality score {review_results['quality_score']:.2f} >= {quality_threshold}")
                    return True
        
        return False
    
    return should_stop


def format_loop_results(loop_output: str, iterations: int) -> Dict[str, Any]:
    """
    Format the final results from the loop.
    
    Args:
        loop_output: Raw output from the loop
        iterations: Number of iterations completed
        
    Returns:
        Formatted results
    """
    # Extract final code
    final_code = extract_refactored_code(loop_output)
    
    # Extract final review
    final_review = extract_review_results(loop_output)
    
    return {
        "status": "loop_completed",
        "iterations_completed": iterations,
        "final_code": final_code,
        "final_quality_score": final_review.get("quality_score", 0.0),
        "remaining_issues": {
            "critical": final_review.get("severity_counts", {}).get("critical", 0),
            "major": final_review.get("severity_counts", {}).get("major", 0),
            "minor": final_review.get("severity_counts", {}).get("minor", 0)
        },
        "improvement_achieved": iterations > 0,
        "ready_for_production": not final_review.get("has_critical_issues", True)
    }


# Create wrapper agents if direct instances aren't available
if not critic_agent_instance:
    # Create a minimal critic agent for testing
    critic_agent_instance = LlmAgent(
        name="critic_agent",
        model=LiteLlm(model="openai/gpt-4o"),
        description="Code reviewer that identifies issues",
        instruction="""Review the provided code and identify:
        1. Critical issues (security vulnerabilities, bugs)
        2. Major issues (performance, design flaws)
        3. Minor issues (style, documentation)
        
        Provide a structured review with severity levels."""
    )

if not refactor_agent_instance:
    # Create a minimal refactor agent for testing
    refactor_agent_instance = LlmAgent(
        name="refactor_agent",
        model=LiteLlm(model="openai/gpt-4o"),
        description="Code refactoring specialist",
        instruction="""Refactor the code based on the review feedback.
        Focus on fixing critical and major issues first.
        Provide the improved code in a code block."""
    )


# Create the main Loop Agent
root_agent = LoopAgent(
    name="review_refactor_loop",
    description="Iteratively reviews and refactors code until quality standards are met",
    sub_agents=[critic_agent_instance, refactor_agent_instance],
    max_iterations=5
)


# Create the root agent that orchestrates everything
#  root_agent = Agent(
#     name="review_loop_agent",
#     model=LiteLlm(model="openai/gpt-4o"),
#     description="Orchestrates iterative code review and refactoring until quality standards are met",
#     instruction="""You are the Review-Refactor Loop Orchestrator. Your job is to:

# 1. Take the user's code that needs improvement
# 2. Prepare it for the review-refactor loop
# 3. Run the loop (which automatically handles critic → refactor iterations)
# 4. Present the final improved code and summary

# The loop will automatically:
# - Run the critic agent to review code
# - Run the refactor agent to fix issues
# - Repeat until quality is acceptable or max iterations reached

# IMPORTANT: 
# - First call prepare_code_for_review() to prepare the code
# - Then simply state that you're starting the iterative review-refactor process
# - The LoopAgent sub-agent will handle the iterations automatically
# - After the loop completes, format and present the results

# Example response:
# "I'll help improve your code through iterative review and refactoring. Let me prepare it first...
# [Call prepare_code_for_review]
# Now I'll run the review-refactor loop to iteratively improve the code...
# [The loop runs automatically]
# Here's the improved code after X iterations:
# [Present final code and summary]"
# """,
#     tools=[
#         FunctionTool(prepare_code_for_review),
#         FunctionTool(extract_review_results),
#         FunctionTool(extract_refactored_code),
#         FunctionTool(format_loop_results),
#         FunctionTool(analyze_code),
#         FunctionTool(wrap_code_in_markdown),
#         FunctionTool(format_code_output)
#     ],
#     sub_agents=[loop_agent]  # The loop agent is a sub-agent
# )


# Test function
if __name__ == "__main__":
    print("🔄 Review-Refactor Loop Agent Ready!")
    print("\n✨ Features:")
    print("- Automatic iteration between critic and refactor agents")
    print("- Stops when code quality is acceptable")
    print("- Maximum 5 iterations to prevent infinite loops")
    print("- Structured output with improvement tracking")
    
    print("\n📋 Components:")
    print(f"- Critic Agent: {'✓ Loaded' if critic_agent_instance else '✗ Using fallback'}")
    print(f"- Refactor Agent: {'✓ Loaded' if refactor_agent_instance else '✗ Using fallback'}")
    print("- Loop Agent: ✓ Configured")
    
    print("\n🚀 Usage:")
    print("1. Run: adk run agents/review_loop_agent")
    print("2. Provide code that needs improvement")
    print("3. Watch the iterative improvement process")
    
    print("\n💡 Example Test:")
    print('Try: "Please improve this code:"')
    print('def authenticate(u, p):')
    print('    return u == "admin" and p == "password123"')