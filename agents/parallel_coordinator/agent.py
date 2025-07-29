"""
Parallel Coordinator Agent - Production Ready
Runs Gemini, GPT-4, and Claude generation agents in parallel and selects the best output

Features:
- Parallel execution of all three generation agents
- Intelligent selection of best generated code
- Memory integration for learning patterns
- Guardrails for input/output validation
- Comprehensive comparison and scoring
"""

from google.adk.agents import ParallelAgent, LlmAgent, Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys
from pathlib import Path
import json

# Add paths for importing other agents
current_dir = Path(__file__).parent
agents_dir = current_dir.parent
sys.path.append(str(agents_dir))

# Import generation agents with error handling
try:
    from gemini_agent.agent import root_agent as gemini_agent
    print("✅ Gemini Generation Agent imported successfully")
    GEMINI_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import Gemini Agent: {e}")
    gemini_agent = None
    GEMINI_AVAILABLE = False

try:
    from gpt4_agent.agent import root_agent as gpt4_agent
    print("✅ GPT-4 Generation Agent imported successfully")
    GPT4_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import GPT-4 Agent: {e}")
    gpt4_agent = None
    GPT4_AVAILABLE = False

try:
    from claude_gen_agent.agent import root_agent as claude_agent
    print("✅ Claude Generation Agent imported successfully")
    CLAUDE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import Claude Agent: {e}")
    claude_agent = None
    CLAUDE_AVAILABLE = False

# Import shared tools
try:
    from shared.tools import (
        analyze_code,
        validate_python_syntax,
        estimate_complexity,
        format_code_output
    )
except ImportError:
    print("⚠️ Some shared tools not available")

# Import memory and guardrails
try:
    from shared.memory import get_memory_service, MemoryEntry
    MEMORY_AVAILABLE = True
except ImportError:
    print("⚠️ Memory service not available")
    MEMORY_AVAILABLE = False
    class MemoryEntry:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

try:
    from shared.guardrails.input_guardrail import validate_input_request
    from shared.guardrails.output_guardrail import validate_output_safety
    GUARDRAILS_AVAILABLE = True
except ImportError:
    print("⚠️ Guardrails not available")
    GUARDRAILS_AVAILABLE = False


# State keys for parallel coordination
STATE_USER_REQUEST = "user_request"
STATE_GEMINI_OUTPUT = "gemini_output"
STATE_GPT4_OUTPUT = "gpt4_output"
STATE_CLAUDE_OUTPUT = "claude_output"
STATE_BEST_SELECTION = "best_selection"
STATE_COMPARISON_RESULTS = "comparison_results"
STATE_PARALLEL_STATUS = "parallel_status"


# --- INITIALIZATION FUNCTIONS ---

def initialize_parallel_state(
    user_request: str,
    additional_context: str = ""
) -> Dict[str, Any]:
    """
    Initialize state for parallel generation.
    
    Args:
        user_request: The user's code generation request
        additional_context: Any additional context or requirements
        
    Returns:
        Initial state dictionary
    """
    return {
        STATE_USER_REQUEST: user_request,
        "additional_context": additional_context,
        "timestamp": datetime.now().isoformat(),
        STATE_PARALLEL_STATUS: "initialized",
        STATE_GEMINI_OUTPUT: None,
        STATE_GPT4_OUTPUT: None,
        STATE_CLAUDE_OUTPUT: None,
        STATE_BEST_SELECTION: None,
        STATE_COMPARISON_RESULTS: None,
        "request_validated": False,
        "outputs_validated": False
    }


def validate_generation_request(request: str) -> Dict[str, Any]:
    """
    Validate the generation request using guardrails.
    
    Args:
        request: User's generation request
        
    Returns:
        Validation results
    """
    if not GUARDRAILS_AVAILABLE:
        return {
            "is_valid": True,
            "is_safe": True,
            "confidence": 0.8,
            "message": "Guardrails not available - proceeding with basic validation"
        }
    
    try:
        validation_result = validate_input_request(request)
        return {
            "is_valid": validation_result.get("is_valid", True),
            "is_safe": validation_result.get("is_safe", True),
            "confidence": validation_result.get("confidence", 1.0),
            "issues": validation_result.get("issues", []),
            "blocked": not validation_result.get("is_safe", True)
        }
    except Exception as e:
        return {
            "is_valid": True,
            "is_safe": True,
            "confidence": 0.5,
            "error": str(e)
        }


# --- COMPARISON AND SELECTION FUNCTIONS ---

def analyze_generated_code(
    code: str,
    agent_name: str
) -> Dict[str, Any]:
    """
    Analyze generated code from an agent.
    
    Args:
        code: Generated code
        agent_name: Name of the agent that generated it
        
    Returns:
        Analysis results with scores
    """
    try:
        # Basic validation
        syntax_result = validate_python_syntax(code)
        is_valid = syntax_result.get("is_valid", True)
        
        # Code metrics
        metrics = analyze_code(code)
        complexity = estimate_complexity(code)
        
        # Calculate scores based on agent specialization
        scores = {
            "syntax_valid": 1.0 if is_valid else 0.0,
            "completeness": min(1.0, metrics.get("lines_non_empty", 0) / 20),  # Assume 20 lines is complete
            "has_functions": 0.2 if metrics.get("has_functions", False) else 0.0,
            "has_classes": 0.2 if metrics.get("has_classes", False) else 0.0,
            "has_docstrings": 0.2 if metrics.get("has_docstrings", False) else 0.0,
            "complexity_score": max(0, 1.0 - (complexity.get("complexity", 0) / 20)),  # Lower is better
            "has_error_handling": 0.2 if "try" in code and "except" in code else 0.0
        }
        
        # Agent-specific bonuses
        if agent_name == "gemini_agent":
            # Gemini excels at clean, efficient code
            scores["clean_code_bonus"] = 0.1 if metrics.get("lines_non_empty", 0) < 50 else 0.0
        elif agent_name == "gpt4_agent":
            # GPT-4 excels at robustness
            scores["robustness_bonus"] = 0.1 if scores["has_error_handling"] > 0 else 0.0
        elif agent_name == "claude_agent":
            # Claude excels at documentation
            scores["documentation_bonus"] = 0.1 if scores["has_docstrings"] > 0 else 0.0
        
        # Calculate total score
        total_score = sum(scores.values())
        
        return {
            "agent": agent_name,
            "is_valid": is_valid,
            "scores": scores,
            "total_score": total_score,
            "metrics": metrics,
            "complexity": complexity.get("complexity", 0),
            "syntax_errors": syntax_result.get("errors", [])
        }
        
    except Exception as e:
        return {
            "agent": agent_name,
            "is_valid": False,
            "total_score": 0.0,
            "error": str(e)
        }


def compare_generated_outputs(
    gemini_output: Dict[str, Any],
    gpt4_output: Dict[str, Any],
    claude_output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare outputs from all three agents and rank them.
    
    Args:
        gemini_output: Output from Gemini agent
        gpt4_output: Output from GPT-4 agent
        claude_output: Output from Claude agent
        
    Returns:
        Comparison results with rankings
    """
    analyses = []
    
    # Analyze each output
    for agent_name, output in [
        ("gemini_agent", gemini_output),
        ("gpt4_agent", gpt4_output),
        ("claude_agent", claude_output)
    ]:
        if output and isinstance(output, dict):
            code = output.get("generated_code", output.get("code", ""))
            if code:
                analysis = analyze_generated_code(code, agent_name)
                analysis["output"] = output
                analyses.append(analysis)
            else:
                print(f"⚠️ No code found in {agent_name} output")
        else:
            print(f"⚠️ Invalid output from {agent_name}")
    
    # Sort by total score
    analyses.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    
    # Create comparison summary
    comparison = {
        "rankings": [
            {
                "rank": i + 1,
                "agent": analysis["agent"],
                "score": analysis["total_score"],
                "is_valid": analysis["is_valid"],
                "key_strengths": _identify_strengths(analysis)
            }
            for i, analysis in enumerate(analyses)
        ],
        "best_agent": analyses[0]["agent"] if analyses else None,
        "best_score": analyses[0]["total_score"] if analyses else 0,
        "all_analyses": analyses,
        "recommendation": _generate_recommendation(analyses)
    }
    
    return comparison


def _identify_strengths(analysis: Dict[str, Any]) -> List[str]:
    """Identify key strengths of the generated code."""
    strengths = []
    scores = analysis.get("scores", {})
    
    if scores.get("syntax_valid", 0) > 0:
        strengths.append("Valid syntax")
    if scores.get("has_error_handling", 0) > 0:
        strengths.append("Error handling")
    if scores.get("has_docstrings", 0) > 0:
        strengths.append("Well documented")
    if scores.get("complexity_score", 0) > 0.7:
        strengths.append("Low complexity")
    if scores.get("clean_code_bonus", 0) > 0:
        strengths.append("Clean and efficient")
    if scores.get("robustness_bonus", 0) > 0:
        strengths.append("Robust implementation")
    
    return strengths


def _generate_recommendation(analyses: List[Dict[str, Any]]) -> str:
    """Generate recommendation based on comparison results."""
    if not analyses:
        return "No valid outputs to compare"
    
    best = analyses[0]
    if best["total_score"] > 1.5:
        return f"Strongly recommend {best['agent']} output - high quality code"
    elif best["total_score"] > 1.0:
        return f"Recommend {best['agent']} output - good quality code"
    else:
        return f"Use {best['agent']} output with review - may need improvements"


def select_best_output(
    comparison_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Select the best output based on comparison results.
    
    Args:
        comparison_results: Results from compare_generated_outputs
        
    Returns:
        Selected best output with metadata
    """
    best_agent = comparison_results.get("best_agent")
    all_analyses = comparison_results.get("all_analyses", [])
    
    if not all_analyses:
        return {
            "status": "no_valid_outputs",
            "selected_agent": None,
            "selected_code": None,
            "reason": "No valid outputs to select from"
        }
    
    # Get the best analysis
    best_analysis = all_analyses[0]
    best_output = best_analysis.get("output", {})
    
    # Extract the code
    selected_code = best_output.get("generated_code", best_output.get("code", ""))
    
    # Validate the selected code
    if GUARDRAILS_AVAILABLE:
        try:
            validation = validate_output_safety(
                generated_code=selected_code,
                original_request="Code generation request"
            )
            is_safe = validation.get("is_safe", True)
        except:
            is_safe = True
    else:
        is_safe = True
    
    return {
        "status": "selection_complete",
        "selected_agent": best_agent,
        "selected_code": selected_code,
        "selection_score": best_analysis["total_score"],
        "is_safe": is_safe,
        "reason": f"Selected {best_agent} with score {best_analysis['total_score']:.2f}",
        "runner_up": all_analyses[1]["agent"] if len(all_analyses) > 1 else None,
        "all_scores": {
            analysis["agent"]: analysis["total_score"] 
            for analysis in all_analyses
        }
    }


def save_parallel_results_to_memory(
    request: str,
    comparison_results: Dict[str, Any],
    selected_output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Save successful parallel generation results to memory.
    
    Args:
        request: Original user request
        comparison_results: Comparison results
        selected_output: Selected best output
        
    Returns:
        Memory save result
    """
    if not MEMORY_AVAILABLE:
        return {"saved": False, "reason": "Memory not available"}
    
    try:
        memory_service = get_memory_service()
        
        memory_entry = MemoryEntry(
            category="parallel_generation",
            agent_name="parallel_coordinator",
            data={
                "request": request[:200],
                "selected_agent": selected_output.get("selected_agent"),
                "selection_score": selected_output.get("selection_score", 0),
                "all_scores": selected_output.get("all_scores", {}),
                "timestamp": datetime.now().isoformat(),
                "comparison_summary": comparison_results.get("recommendation", "")
            },
            quality_score=selected_output.get("selection_score", 0) / 2.0,  # Normalize
            tags=["parallel", "generation", selected_output.get("selected_agent", "unknown")]
        )
        
        memory_id = memory_service.store(memory_entry)
        
        return {
            "saved": True,
            "memory_id": memory_id,
            "message": "Parallel generation results saved"
        }
        
    except Exception as e:
        return {
            "saved": False,
            "error": str(e)
        }


def format_parallel_results(
    selected_output: Dict[str, Any],
    comparison_results: Dict[str, Any],
    validation_results: Dict[str, Any],
    memory_save: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format final parallel generation results.
    
    Args:
        selected_output: Selected best output
        comparison_results: Comparison results
        validation_results: Input validation results
        memory_save: Memory save results
        
    Returns:
        Formatted final output
    """
    return {
        "status": "parallel_generation_complete",
        "selected_code": selected_output.get("selected_code", ""),
        "selected_agent": selected_output.get("selected_agent", "unknown"),
        "selection_metadata": {
            "score": selected_output.get("selection_score", 0),
            "reason": selected_output.get("reason", ""),
            "runner_up": selected_output.get("runner_up", None),
            "all_scores": selected_output.get("all_scores", {})
        },
        "comparison_summary": {
            "rankings": comparison_results.get("rankings", []),
            "recommendation": comparison_results.get("recommendation", "")
        },
        "validation": {
            "input_valid": validation_results.get("is_valid", True),
            "output_safe": selected_output.get("is_safe", True)
        },
        "memory_integration": memory_save,
        "next_stage": "review_loop",
        "ready_for_review": True
    }


# --- PREPARATION AGENT ---

parallel_preparation_agent = LlmAgent(
    name="ParallelPreparation",
    model="gemini-2.0-flash",
    description="Prepares state for parallel generation",
    instruction="""You are the Parallel Preparation Agent. Set up the parallel generation process.

Your tasks:
1. Call initialize_parallel_state with the user's request
2. Call validate_generation_request to ensure the request is safe
3. Prepare the state for parallel execution

The user will provide:
- request: The code generation request
- context (optional): Additional context or requirements

Initialize the state and validate the request, then confirm readiness for parallel execution.""",
    tools=[
        FunctionTool(initialize_parallel_state),
        FunctionTool(validate_generation_request)
    ],
    output_key="parallel_preparation"
)


# --- COMPARISON AGENT ---

comparison_agent = LlmAgent(
    name="ComparisonAgent",
    model="gemini-2.0-flash",
    description="Compares outputs from all generation agents",
    instruction="""You are the Comparison Agent. Compare and analyze outputs from all three generation agents.

Your workflow:
1. Extract outputs from session state:
   - gemini_output
   - gpt4_output
   - claude_output

2. Call compare_generated_outputs with all three outputs to analyze and rank them

3. Call select_best_output with the comparison results to choose the best code

4. Save results to memory using save_parallel_results_to_memory

5. Format final results with format_parallel_results

Remember agent specializations:
- Gemini: Clean, efficient code
- GPT-4: Robust, comprehensive solutions
- Claude: Elegant, well-documented code

Select the best output based on overall quality, correctness, and fitness for purpose.""",
    tools=[
        FunctionTool(compare_generated_outputs),
        FunctionTool(select_best_output),
        FunctionTool(save_parallel_results_to_memory),
        FunctionTool(format_parallel_results)
    ],
    output_key="comparison_results"
)


# --- CREATE PARALLEL AGENT ---

# Only include available agents
generation_agents = []
if GEMINI_AVAILABLE and gemini_agent:
    generation_agents.append(gemini_agent)
    print("✅ Added Gemini agent to parallel execution")
else:
    print("⚠️ Gemini agent not available for parallel execution")

if GPT4_AVAILABLE and gpt4_agent:
    generation_agents.append(gpt4_agent)
    print("✅ Added GPT-4 agent to parallel execution")
else:
    print("⚠️ GPT-4 agent not available for parallel execution")

if CLAUDE_AVAILABLE and claude_agent:
    generation_agents.append(claude_agent)
    print("✅ Added Claude agent to parallel execution")
else:
    print("⚠️ Claude agent not available for parallel execution")

if len(generation_agents) >= 2:
    # Create parallel agent only if we have at least 2 agents
    parallel_generation_agent = ParallelAgent(
        name="ParallelGeneration",
        description="Runs generation agents in parallel",
        sub_agents=generation_agents
    )
    print(f"✅ Created ParallelAgent with {len(generation_agents)} generation agents")
else:
    print("❌ Not enough generation agents available for parallel execution")
    parallel_generation_agent = None


# --- ROOT AGENT ---

if parallel_generation_agent:
    root_agent = Agent(
        name="parallel_coordinator",
        model=LiteLlm(model="openai/gpt-4o"),
        description="Coordinates parallel execution of generation agents and selects best output",
        instruction="""You are the Parallel Coordinator Agent. You orchestrate the parallel execution of Gemini, GPT-4, and Claude generation agents.

Your workflow:
1. **Preparation Phase**: The preparation agent will set up the state
2. **Parallel Generation**: All three agents generate code simultaneously
3. **Comparison Phase**: Compare all outputs and select the best one

Key responsibilities:
- Ensure all generation agents receive the same request
- Wait for all parallel executions to complete
- Compare outputs based on quality metrics
- Select the best code for the review loop
- Save results for future learning

The selected code will be passed to the review-refactor loop for quality improvement.

Agent specializations to consider:
- **Gemini**: Excels at clean, efficient Google Cloud best practices
- **GPT-4**: Excels at comprehensive, robust solutions with error handling
- **Claude**: Excels at elegant, well-documented solutions with type safety

Your selection should consider:
- Code correctness and syntax validity
- Completeness of the solution
- Error handling and robustness
- Documentation and readability
- Alignment with the specific request

Remember: The goal is to select the best starting point for the review-refactor loop.""",
        tools=[
            FunctionTool(parallel_preparation_agent),
            FunctionTool(comparison_agent)
        ],
        output_key="parallel_coordinator_results"
    )
else:
    # Fallback for testing
    root_agent = Agent(
        name="parallel_coordinator",
        model=LiteLlm(model="openai/gpt-4o"),
        description="Parallel coordinator in test mode",
        instruction="Parallel generation not available - need at least 2 generation agents",
        tools=[],
        output_key="parallel_coordinator_results"
    )


# --- UTILITY FUNCTIONS ---

def run_parallel_generation(
    session_state: Dict[str, Any],
    request: str,
    context: str = ""
) -> Dict[str, Any]:
    """
    Run parallel generation programmatically.
    
    Args:
        session_state: ADK session state
        request: Code generation request
        context: Additional context
        
    Returns:
        Results from parallel generation
    """
    # Initialize state
    initial_state = initialize_parallel_state(request, context)
    session_state.update(initial_state)
    
    print(f"🚀 Starting parallel generation for: {request[:50]}...")
    
    # The actual execution would be done by ADK
    return {
        "status": "ready_for_execution",
        "request": request,
        "agents_available": len(generation_agents)
    }


def get_parallel_results(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract results from parallel execution."""
    return {
        "selected_code": session_state.get(STATE_BEST_SELECTION, {}).get("selected_code", ""),
        "selected_agent": session_state.get(STATE_BEST_SELECTION, {}).get("selected_agent", ""),
        "comparison_results": session_state.get(STATE_COMPARISON_RESULTS, {}),
        "all_outputs": {
            "gemini": session_state.get(STATE_GEMINI_OUTPUT, {}),
            "gpt4": session_state.get(STATE_GPT4_OUTPUT, {}),
            "claude": session_state.get(STATE_CLAUDE_OUTPUT, {})
        },
        "status": session_state.get(STATE_PARALLEL_STATUS, "unknown")
    }


if __name__ == "__main__":
    print("\n🎯 Parallel Coordinator Agent Ready!")
    print("=" * 50)
    
    print("\n📊 CONFIGURATION:")
    print(f"- Gemini Agent: {'✅ Available' if GEMINI_AVAILABLE else '❌ Not Available'}")
    print(f"- GPT-4 Agent: {'✅ Available' if GPT4_AVAILABLE else '❌ Not Available'}")
    print(f"- Claude Agent: {'✅ Available' if CLAUDE_AVAILABLE else '❌ Not Available'}")
    print(f"- Total Agents: {len(generation_agents)}")
    print(f"- Parallel Execution: {'✅ Ready' if parallel_generation_agent else '❌ Not Possible'}")
    
    print("\n🏗️ ARCHITECTURE:")
    print("1. Parallel Preparation Agent")
    print("   - Initializes state")
    print("   - Validates request")
    print("2. ParallelAgent")
    print("   - Runs all generation agents simultaneously")
    print("   - Collects outputs")
    print("3. Comparison Agent")
    print("   - Analyzes all outputs")
    print("   - Selects best code")
    print("   - Saves results to memory")
    
    print("\n⚙️ SELECTION CRITERIA:")
    print("- Syntax validity")
    print("- Code completeness")
    print("- Error handling")
    print("- Documentation quality")
    print("- Complexity score")
    print("- Agent-specific strengths")
    
    print("\n🚀 USAGE:")
    print("1. Run: adk run agents/parallel_coordinator")
    print("2. Provide:")
    print("   - request: Your code generation request")
    print("   - context: Additional requirements (optional)")
    print("\n3. The agent will:")
    print("   - Run all generation agents in parallel")
    print("   - Compare outputs")
    print("   - Select the best code")
    print("   - Prepare it for review-refactor loop")
    
    print("\n💡 EXAMPLE REQUESTS:")
    print("- 'Create a REST API with FastAPI for user management'")
    print("- 'Build a data processing pipeline with error handling'")
    print("- 'Implement a caching system with TTL support'")
    
    if len(generation_agents) < 3:
        print("\n⚠️ WARNING: Not all generation agents are available")
        print("The parallel coordinator will work with available agents only")
    
    print("\n✨ Benefits of Parallel Generation:")
    print("- Get the best of all three models")
    print("- Automatic quality comparison")
    print("- Learn from selection patterns")
    print("- Higher quality starting code for review loop")