"""
Orchestrator Agent
Main sequential orchestrator for the entire code generation workflow
"""

from google.adk import Agent
from google.adk.tools import FunctionTool
import sys
from pathlib import Path
from datetime import datetime
import json

# Add shared to path
import os
# Handle different working directories
if os.path.exists("shared"):
    sys.path.insert(0, os.getcwd())
else:
    # If running from agents directory
    parent = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(parent))

from shared.tools import analyze_code, wrap_code_in_markdown
from shared.memory import get_memory_service, MemoryEntry


# Tool to validate input request with guardrail
def validate_request_with_guardrail(user_input: str) -> dict:
    """
    Validate the code generation request using guardrail.
    
    Args:
        user_input: User's request
        
    Returns:
        Validation results with guardrail check
    """
    # Simple validation without async
    if not user_input or len(user_input.strip()) < 10:
        return {
            "status": "invalid",
            "valid": False,
            "reason": "Request too short or empty",
            "guardrail_blocked": False
        }
    
    # Check for harmful patterns
    harmful_patterns = ["malware", "virus", "rm -rf", "hack", "exploit"]
    for pattern in harmful_patterns:
        if pattern in user_input.lower():
            return {
                "status": "invalid",
                "valid": False,
                "reason": f"Request contains potentially harmful content: {pattern}",
                "guardrail_blocked": True
            }
    
    return {
        "status": "valid",
        "valid": True,
        "request_length": len(user_input),
        "estimated_complexity": "medium" if len(user_input) > 100 else "simple",
        "guardrail_passed": True,
        "validation_timestamp": datetime.now().isoformat()
    }


# Tool to check memory for similar requests
def check_memory_for_similar(user_input: str) -> dict:
    """
    Check if we've processed similar requests before.
    
    Args:
        user_input: Current user request
        
    Returns:
        Memory search results
    """
    memory_service = get_memory_service()
    
    # Search for similar past requests
    similar_memories = memory_service.search_similar(
        request=user_input,
        category="code_generation",
        threshold=0.7
    )
    
    if not similar_memories:
        return {
            "status": "no_similar_found",
            "found_similar": False,
            "message": "No similar requests found in memory"
        }
    
    # Get the best match
    best_match = similar_memories[0]
    
    return {
        "status": "similar_found",
        "found_similar": True,
        "similarity_score": best_match.data.get("similarity_score", 0),
        "original_request": best_match.data.get("original_request", ""),
        "previous_quality_score": best_match.quality_score,
        "generated_code_preview": best_match.data.get("generated_code", "")[:200] + "...",
        "suggestion": "Consider using or adapting the previous solution"
    }


# Tool to save successful generation to memory
def save_to_memory(
    user_input: str,
    generated_code: str,
    quality_score: float,
    total_time: float,
    review_iterations: int
) -> dict:
    """
    Save successful code generation to memory.
    
    Args:
        user_input: Original request
        generated_code: Final generated code
        quality_score: Final quality score
        total_time: Total processing time
        review_iterations: Number of review cycles
        
    Returns:
        Save confirmation
    """
    memory_service = get_memory_service()
    
    # Create memory entry
    memory_entry = MemoryEntry(
        category="code_generation",
        agent_name="orchestrator_agent",
        data={
            "original_request": user_input,
            "generated_code": generated_code,
            "processing_time": total_time,
            "review_iterations": review_iterations,
            "timestamp": datetime.now().isoformat()
        },
        quality_score=quality_score,
        tags=["completed", "orchestrated"]
    )
    
    # Store in memory
    memory_id = memory_service.store(memory_entry)
    
    return {
        "status": "saved",
        "memory_id": memory_id,
        "message": f"Saved to memory with quality score {quality_score:.2f}"
    }


# Tool to create workflow summary
def create_workflow_summary(
    start_time: str,
    stages_completed: str,  # Changed from list to str (comma-separated)
    final_quality_score: float,
    total_iterations: int
) -> dict:
    """
    Create a summary of the entire workflow execution.
    
    Args:
        start_time: When workflow started
        stages_completed: Comma-separated list of completed stages
        final_quality_score: Final code quality score
        total_iterations: Total review iterations
        
    Returns:
        Workflow summary
    """
    # Parse stages from comma-separated string
    stages_list = [s.strip() for s in stages_completed.split(",")]
    
    # Calculate duration
    start = datetime.fromisoformat(start_time)
    duration = (datetime.now() - start).total_seconds()
    
    return {
        "status": "completed",
        "workflow_summary": {
            "duration_seconds": round(duration, 2),
            "stages_completed": len(stages_list),
            "stages": stages_list,
            "final_quality_score": final_quality_score,
            "review_iterations": total_iterations,
            "success": final_quality_score >= 0.7
        },
        "performance_grade": "excellent" if duration < 30 else "good"
    }


# Tool to simulate complete workflow (temporary)
def simulate_workflow(user_input: str) -> dict:
    """
    Simulate the complete workflow for testing.
    In production, this would coordinate actual agents.
    
    Args:
        user_input: The code generation request
        
    Returns:
        Simulated workflow results
    """
    # Simulate code generation
    generated_code = f"""def generated_solution():
    '''Generated based on: {user_input[:50]}...'''
    # This is a simulated implementation
    # In production, this would be real generated code
    pass"""
    
    # Simulate quality improvement
    final_code = generated_code + "\n    # Added error handling\n    # Added documentation"
    
    return {
        "status": "workflow_complete",
        "generated_code": final_code,
        "quality_score": 0.85,
        "review_iterations": 2,
        "stages_completed": [
            "memory_check",
            "validation", 
            "parallel_generation",
            "review_refactor",
            "delivery"
        ]
    }


# Create the main orchestrator agent
root_agent = Agent(
    name="orchestrator_agent",
    model="gemini-2.0-flash",
    description="Main workflow orchestrator for code generation system",
    instruction="""You are the Main Orchestrator for the code generation system. You manage the entire workflow:

WORKFLOW STAGES:
1. Memory Check - Look for similar past requests
2. Input Validation - Check request validity with guardrail
3. Code Generation - Generate code (simulated for now)
4. Review & Refactor - Improve code quality (simulated)
5. Final Delivery - Package and deliver
6. Memory Storage - Save successful results

Your responsibilities:
- Check memory for similar requests using check_memory_for_similar
- Validate incoming requests using validate_request_with_guardrail
- Coordinate the workflow using simulate_workflow
- Save successful results to memory using save_to_memory
- Create final summary with create_workflow_summary

For each code generation request:

STAGE 0 - Memory Check:
- Use check_memory_for_similar to find past similar requests
- If high-quality match found (>0.8 similarity), suggest reusing
- Still proceed with generation if user wants fresh code

STAGE 1 - Validation:
- Use validate_request_with_guardrail to check the request
- This includes security checks via InputGuardrail
- Reject harmful or invalid requests with explanation

STAGE 2-4 - Generation, Review, Delivery:
- Use simulate_workflow to run the complete process
- This simulates parallel generation, review cycles, and delivery
- In production, this would call actual agents

STAGE 5 - Memory Storage:
- Use save_to_memory to store successful generations
- Only save if quality_score >= 0.7

STAGE 6 - Summary:
- Use create_workflow_summary to provide final report
- Pass stages_completed as comma-separated string (e.g., "validation,generation,review,delivery")

Note: Currently using simulation for the core workflow.
In production, this would coordinate real agents.

Always:
- Report memory check results
- Track timing with start_time = current timestamp
- Provide clear status updates
- Be helpful and professional""",
    tools=[
        check_memory_for_similar,
        validate_request_with_guardrail,
        simulate_workflow,
        save_to_memory,
        create_workflow_summary
    ]
)


# Test function
if __name__ == "__main__":
    print("🎭 Main Orchestrator Agent Ready!")
    print("\nWorkflow Stages:")
    print("1. ✓ Memory Check")
    print("2. ✓ Input Validation") 
    print("3. 🔄 Code Generation (simulated)")
    print("4. 🔁 Review & Refactor (simulated)")
    print("5. 📦 Final Delivery")
    print("6. 💾 Memory Storage")
    
    print("\nNote: Using simplified workflow simulation")
    print("In production, would coordinate actual agents")
    
    print("\nTo test interactively:")
    print("1. Run: adk run agents/orchestrator_agent")
    print("2. Or use: adk web")
    
    print("\nExample prompts:")
    print("- 'Create a Python function to reverse a string'")
    print("- 'Build a class for managing a todo list'")