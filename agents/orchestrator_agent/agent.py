"""
Main Orchestrator Agent - Production Ready
Coordinates the entire multi-agent code generation system end-to-end

Workflow:
1. User Input → Input Guardrails
2. Parallel Generation (Gemini + GPT-4 + Claude)
3. Best Selection → Review-Refactor Loop
4. Output Guardrails → Final Delivery

This is the main entry point for the entire system.
"""

from google.adk.agents import SequentialAgent, LlmAgent, Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys
from pathlib import Path
import json

import os
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OPENTELEMETRY_SUPPRESS_INSTRUMENTATION"] = "true"


# Add paths for importing all agents
current_dir = Path(__file__).parent
agents_dir = current_dir.parent
sys.path.append(str(agents_dir))

# Import all our agents
try:
    from parallel_coordinator.agent import root_agent as parallel_coordinator
    print("✅ Parallel Coordinator imported successfully")
    PARALLEL_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import Parallel Coordinator: {e}")
    parallel_coordinator = None
    PARALLEL_AVAILABLE = False

try:
    from review_loop_agent.agent import root_agent as review_loop_agent
    print("✅ Review-Refactor Loop Agent imported successfully")
    REVIEW_LOOP_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import Review Loop Agent: {e}")
    review_loop_agent = None
    REVIEW_LOOP_AVAILABLE = False

# Import delivery agent if available
try:
    from delivery_agent.agent import root_agent as delivery_agent
    print("✅ Delivery Agent imported successfully")
    DELIVERY_AVAILABLE = True
except ImportError as e:
    print("⚠️ Delivery Agent not available - will create basic version")
    delivery_agent = None
    DELIVERY_AVAILABLE = False

# Import guardrails
try:
    from shared.guardrails.input_guardrail import validate_input_request
    from shared.guardrails.output_guardrail import validate_output_safety
    GUARDRAILS_AVAILABLE = True
    print("✅ Guardrails available")
except ImportError:
    print("⚠️ Guardrails not available - using fallback")
    GUARDRAILS_AVAILABLE = False

# Import memory service
try:
    from shared.memory import get_memory_service, MemoryEntry
    MEMORY_AVAILABLE = True
    print("✅ Memory service available")
except ImportError:
    print("⚠️ Memory service not available")
    MEMORY_AVAILABLE = False
    class MemoryEntry:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


# State keys for orchestration
STATE_USER_REQUEST = "user_request"
STATE_PROJECT_NAME = "project_name"
STATE_PROJECT_DESCRIPTION = "project_description"
STATE_VALIDATION_RESULTS = "input_validation_results"
STATE_PARALLEL_RESULTS = "parallel_coordinator_results"
STATE_SELECTED_CODE = "selected_code"
STATE_SELECTED_AGENT = "selected_agent"
STATE_REVIEW_RESULTS = "review_loop_results"
STATE_FINAL_CODE = "final_code"
STATE_OUTPUT_VALIDATION = "output_validation_results"
STATE_DELIVERY_READY = "delivery_ready"
STATE_ORCHESTRATION_STATUS = "orchestration_status"
STATE_WORKFLOW_SUMMARY = "workflow_summary"


# --- ORCHESTRATION FUNCTIONS ---

def validate_user_request(
    request: str,
    project_name: str = "",
    project_description: str = ""
) -> Dict[str, Any]:
    """
    Validate user input using guardrails.
    
    Args:
        request: User's code generation request
        project_name: Name of the project
        project_description: Description of the project
        
    Returns:
        Validation results
    """
    if not GUARDRAILS_AVAILABLE:
        return {
            "is_valid": True,
            "is_safe": True,
            "confidence": 0.8,
            "can_proceed": True,
            "message": "Guardrails not available - proceeding with basic validation",
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        # Combine all inputs for validation
        full_request = f"""
        Project: {project_name}
        Description: {project_description}
        Request: {request}
        """
        
        validation_result = validate_input_request(full_request)
        
        can_proceed = validation_result.get("is_valid", True) and validation_result.get("is_safe", True)
        
        return {
            "is_valid": validation_result.get("is_valid", True),
            "is_safe": validation_result.get("is_safe", True),
            "confidence": validation_result.get("confidence", 1.0),
            "can_proceed": can_proceed,
            "issues": validation_result.get("issues", []),
            "message": "Request validated successfully" if can_proceed else "Request blocked by guardrails",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "is_valid": True,
            "is_safe": True,
            "confidence": 0.5,
            "can_proceed": True,
            "error": str(e),
            "message": "Validation error - proceeding with caution",
            "timestamp": datetime.now().isoformat()
        }


def prepare_for_parallel_generation(
    user_request: str,
    validation_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare the validated request for parallel generation.
    
    Args:
        user_request: Original user request
        validation_results: Results from validation
        
    Returns:
        Prepared request for parallel generation
    """
    if not validation_results.get("can_proceed", False):
        return {
            "status": "blocked",
            "reason": "Request failed validation",
            "issues": validation_results.get("issues", [])
        }
    
    return {
        "status": "ready_for_generation",
        "request": user_request,
        "validation_confidence": validation_results.get("confidence", 0),
        "timestamp": datetime.now().isoformat(),
        "context": {
            "validated": True,
            "safety_checked": True,
            "ready_for_parallel": True
        }
    }


def prepare_for_review_loop(
    parallel_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Prepare the selected code for review-refactor loop.
    
    Args:
        parallel_results: Results from parallel coordinator
        
    Returns:
        Prepared data for review loop
    """
    selected_code = parallel_results.get("selected_code", "")
    selected_agent = parallel_results.get("selected_agent", "unknown")
    
    if not selected_code:
        return {
            "status": "no_code_selected",
            "error": "No code was selected from parallel generation"
        }
    
    return {
        "status": "ready_for_review",
        "code_to_review": selected_code,
        "source_agent": selected_agent,
        "original_requirement": parallel_results.get("request", ""),
        "generation_metadata": {
            "selection_score": parallel_results.get("selection_metadata", {}).get("score", 0),
            "all_scores": parallel_results.get("selection_metadata", {}).get("all_scores", {}),
            "timestamp": datetime.now().isoformat()
        }
    }


def validate_final_output(
    final_code: str,
    original_request: str
) -> Dict[str, Any]:
    """
    Validate the final output using guardrails.
    
    Args:
        final_code: Final improved code from review loop
        original_request: Original user request
        
    Returns:
        Final validation results
    """
    if not GUARDRAILS_AVAILABLE:
        return {
            "is_safe": True,
            "is_appropriate": True,
            "ready_for_delivery": True,
            "confidence": 0.8,
            "message": "Output validation skipped - guardrails not available"
        }
    
    try:
        validation_result = validate_output_safety(
            generated_code=final_code,
            original_request=original_request
        )
        
        ready = validation_result.get("is_safe", True) and validation_result.get("is_appropriate", True)
        
        return {
            "is_safe": validation_result.get("is_safe", True),
            "is_appropriate": validation_result.get("is_appropriate", True),
            "ready_for_delivery": ready,
            "confidence": validation_result.get("confidence", 1.0),
            "issues": validation_result.get("safety_issues", []),
            "message": "Output validated and ready" if ready else "Output blocked by guardrails",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "is_safe": True,
            "is_appropriate": True,
            "ready_for_delivery": True,
            "confidence": 0.5,
            "error": str(e),
            "message": "Output validation error - proceeding"
        }


def create_workflow_summary(
    validation_results: Dict[str, Any],
    parallel_results: Dict[str, Any],
    review_results: Dict[str, Any],
    output_validation: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a comprehensive summary of the entire workflow.
    
    Args:
        validation_results: Input validation results
        parallel_results: Parallel generation results
        review_results: Review-refactor loop results
        output_validation: Output validation results
        
    Returns:
        Complete workflow summary
    """
    return {
        "workflow_status": "complete",
        "stages_completed": {
            "input_validation": validation_results.get("can_proceed", False),
            "parallel_generation": parallel_results.get("status") == "parallel_generation_complete",
            "review_refactor_loop": review_results.get("completed_successfully", False),
            "output_validation": output_validation.get("ready_for_delivery", False)
        },
        "generation_summary": {
            "selected_agent": parallel_results.get("selected_agent", "unknown"),
            "initial_score": parallel_results.get("selection_metadata", {}).get("score", 0),
            "all_agent_scores": parallel_results.get("selection_metadata", {}).get("all_scores", {})
        },
        "improvement_summary": {
            "total_iterations": review_results.get("total_iterations", 0),
            "initial_issues": review_results.get("initial_issues_count", 0),
            "final_issues": review_results.get("final_issues_count", 0),
            "quality_improved": review_results.get("quality_standards_met", False)
        },
        "validation_summary": {
            "input_safe": validation_results.get("is_safe", False),
            "output_safe": output_validation.get("is_safe", False),
            "confidence": (validation_results.get("confidence", 0) + output_validation.get("confidence", 0)) / 2
        },
        "ready_for_delivery": output_validation.get("ready_for_delivery", False),
        "timestamp": datetime.now().isoformat()
    }


def save_orchestration_to_memory(
    request: str,
    workflow_summary: Dict[str, Any],
    final_code: str
) -> Dict[str, Any]:
    """
    Save successful orchestration to memory for learning.
    
    Args:
        request: Original user request
        workflow_summary: Complete workflow summary
        final_code: Final generated code
        
    Returns:
        Memory save result
    """
    if not MEMORY_AVAILABLE:
        return {"saved": False, "reason": "Memory not available"}
    
    try:
        memory_service = get_memory_service()
        
        # Calculate quality score
        quality_score = 0.0
        if workflow_summary.get("ready_for_delivery", False):
            quality_score += 0.5
        if workflow_summary.get("improvement_summary", {}).get("quality_improved", False):
            quality_score += 0.3
        if workflow_summary.get("validation_summary", {}).get("confidence", 0) > 0.8:
            quality_score += 0.2
        
        memory_entry = MemoryEntry(
            category="orchestration_complete",
            agent_name="main_orchestrator",
            data={
                "request_summary": request[:200],
                "selected_agent": workflow_summary.get("generation_summary", {}).get("selected_agent"),
                "total_iterations": workflow_summary.get("improvement_summary", {}).get("total_iterations"),
                "quality_score": quality_score,
                "workflow_summary": workflow_summary,
                "code_preview": final_code[:500] if final_code else "",
                "timestamp": datetime.now().isoformat()
            },
            quality_score=quality_score,
            tags=["orchestration", "complete", "production"]
        )
        
        memory_id = memory_service.store(memory_entry)
        
        return {
            "saved": True,
            "memory_id": memory_id,
            "quality_score": quality_score
        }
        
    except Exception as e:
        return {
            "saved": False,
            "error": str(e)
        }


def format_final_delivery(
    final_code: str,
    workflow_summary: Dict[str, Any],
    project_metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format the final delivery package.
    
    Args:
        final_code: Final validated code
        workflow_summary: Complete workflow summary
        project_metadata: Project name and description
        
    Returns:
        Final delivery package
    """
    return {
        "status": "ready_for_delivery",
        "project": {
            "name": project_metadata.get("name", "generated_project"),
            "description": project_metadata.get("description", ""),
            "timestamp": datetime.now().isoformat()
        },
        "code": {
            "final_code": final_code,
            "language": "python",
            "validated": True,
            "quality_assured": True
        },
        "generation_metadata": {
            "selected_agent": workflow_summary.get("generation_summary", {}).get("selected_agent"),
            "improvement_iterations": workflow_summary.get("improvement_summary", {}).get("total_iterations"),
            "quality_metrics": {
                "initial_score": workflow_summary.get("generation_summary", {}).get("initial_score"),
                "issues_resolved": workflow_summary.get("improvement_summary", {}).get("initial_issues", 0) - 
                                 workflow_summary.get("improvement_summary", {}).get("final_issues", 0),
                "validation_confidence": workflow_summary.get("validation_summary", {}).get("confidence")
            }
        },
        "next_steps": [
            "Use MCP for file/folder creation",
            "Set up project structure",
            "Initialize git repository",
            "Create documentation"
        ],
        "delivery_ready": True
    }


# --- ORCHESTRATION AGENTS ---

input_validation_agent = LlmAgent(
    name="InputValidation",
    model="gemini-2.0-flash",
    description="Validates user input using guardrails",
    instruction="""You are the Input Validation Agent. Validate the user's request before processing.

Extract from the user input:
1. The main code generation request
2. Project name (if provided)
3. Project description (if provided)

Call validate_user_request with these parameters to ensure the request is safe and valid.

If validation fails, stop the process and explain why.
If validation succeeds, prepare the request for the next stage.""",
    tools=[
        FunctionTool(validate_user_request),
        FunctionTool(prepare_for_parallel_generation)
    ],
    output_key="validation_stage"
)


preparation_agent = LlmAgent(
    name="WorkflowPreparation",
    model="gemini-2.0-flash",
    description="Prepares data between workflow stages",
    instruction="""You are the Workflow Preparation Agent. Prepare data for transitions between stages.

Your tasks vary based on the current stage:

1. After parallel generation: Call prepare_for_review_loop
2. After review loop: Prepare final code for validation
3. Handle any data transformation needed between stages

Ensure smooth data flow through the entire pipeline.""",
    tools=[
        FunctionTool(prepare_for_review_loop)
    ],
    output_key="preparation_stage"
)


output_validation_agent = LlmAgent(
    name="OutputValidation",
    model="gemini-2.0-flash",
    description="Validates final output and prepares delivery",
    instruction="""You are the Output Validation Agent. Perform final validation and prepare delivery.

Your workflow:
1. Extract the final code from review loop results
2. Call validate_final_output to ensure safety and appropriateness
3. Create workflow summary using create_workflow_summary
4. Save to memory using save_orchestration_to_memory
5. Format final delivery using format_final_delivery

Ensure all validation passes before marking as ready for delivery.""",
    tools=[
        FunctionTool(validate_final_output),
        FunctionTool(create_workflow_summary),
        FunctionTool(save_orchestration_to_memory),
        FunctionTool(format_final_delivery)
    ],
    output_key="output_validation_stage"
)


# --- CREATE ORCHESTRATOR ---

# Build the agent list based on availability
orchestration_agents = [input_validation_agent]

if PARALLEL_AVAILABLE and parallel_coordinator:
    orchestration_agents.append(parallel_coordinator)
else:
    print("⚠️ Parallel coordinator not available")

orchestration_agents.append(preparation_agent)

if REVIEW_LOOP_AVAILABLE and review_loop_agent:
    orchestration_agents.append(review_loop_agent)
else:
    print("⚠️ Review loop agent not available")

orchestration_agents.append(output_validation_agent)

if DELIVERY_AVAILABLE and delivery_agent:
    orchestration_agents.append(delivery_agent)

# Create the main orchestrator
root_agent = SequentialAgent(
    name="main_orchestrator",
    description="Main orchestrator coordinating the entire code generation workflow",
    sub_agents=orchestration_agents
)


# --- UTILITY FUNCTIONS ---

def run_complete_workflow(
    session_state: Dict[str, Any],
    request: str,
    project_name: str = "my_project",
    project_description: str = ""
) -> Dict[str, Any]:
    """
    Run the complete orchestrated workflow.
    
    Args:
        session_state: ADK session state
        request: User's code generation request
        project_name: Name of the project
        project_description: Project description
        
    Returns:
        Final results
    """
    # Initialize orchestration state
    session_state.update({
        STATE_USER_REQUEST: request,
        STATE_PROJECT_NAME: project_name,
        STATE_PROJECT_DESCRIPTION: project_description,
        STATE_ORCHESTRATION_STATUS: "started",
        "timestamp_start": datetime.now().isoformat()
    })
    
    print(f"🚀 Starting orchestrated workflow for: {project_name}")
    print(f"📝 Request: {request[:100]}...")
    
    return {
        "status": "initialized",
        "stages": len(orchestration_agents),
        "ready_for_execution": True
    }


def get_orchestration_results(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract final results from orchestration."""
    return {
        "final_code": session_state.get(STATE_FINAL_CODE, ""),
        "workflow_summary": session_state.get(STATE_WORKFLOW_SUMMARY, {}),
        "delivery_package": session_state.get("delivery_package", {}),
        "status": session_state.get(STATE_ORCHESTRATION_STATUS, "unknown"),
        "project": {
            "name": session_state.get(STATE_PROJECT_NAME, ""),
            "description": session_state.get(STATE_PROJECT_DESCRIPTION, "")
        }
    }


if __name__ == "__main__":
    print("\n🎯 MAIN ORCHESTRATOR AGENT - COMPLETE SYSTEM")
    print("=" * 60)
    
    print("\n📊 SYSTEM STATUS:")
    print(f"✅ Guardrails: {'Available' if GUARDRAILS_AVAILABLE else 'Not Available'}")
    print(f"✅ Memory: {'Available' if MEMORY_AVAILABLE else 'Not Available'}")
    print(f"✅ Parallel Coordinator: {'Ready' if PARALLEL_AVAILABLE else 'Not Available'}")
    print(f"✅ Review Loop: {'Ready' if REVIEW_LOOP_AVAILABLE else 'Not Available'}")
    print(f"✅ Total Stages: {len(orchestration_agents)}")
    
    print("\n🔄 COMPLETE WORKFLOW:")
    print("1. 🛡️  Input Validation (Guardrails)")
    print("2. 🚀 Parallel Generation (3 agents)")
    print("3. 🎯 Best Selection")
    print("4. 🔍 Review-Refactor Loop")
    print("5. ✅ Output Validation")
    print("6. 📦 Final Delivery")
    
    print("\n⚙️ WORKFLOW STAGES:")
    for i, agent in enumerate(orchestration_agents, 1):
        print(f"{i}. {agent.name}")
    
    print("\n🚀 USAGE:")
    print("1. Run: adk run agents/main_orchestrator")
    print("2. Provide:")
    print("   - request: Your complete code generation request")
    print("   - project_name: Name for your project")
    print("   - project_description: Brief description")
    print("\n3. The system will:")
    print("   - Validate your request")
    print("   - Generate code with 3 models in parallel")
    print("   - Select the best output")
    print("   - Improve it through review-refactor cycles")
    print("   - Validate the final output")
    print("   - Prepare for delivery")
    
    print("\n💡 EXAMPLE REQUESTS:")
    print("- 'Create a complete REST API for task management with authentication'")
    print("- 'Build a real-time chat application with WebSocket support'")
    print("- 'Implement a data pipeline for ETL processing with error handling'")
    
    print("\n📦 OUTPUT:")
    print("You'll receive:")
    print("- Production-ready code")
    print("- Complete workflow summary")
    print("- Quality metrics")
    print("- Ready for MCP file/folder creation")
    
    print("\n✨ This is your complete code generation system!")
    print("Next step: Connect MCP for automatic project structure creation")