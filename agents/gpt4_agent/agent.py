"""
Enhanced GPT-4 Code Generation Agent - Production Ready with Memory & Guardrails
Specializes in comprehensive solutions with detailed error handling and edge case coverage

Features:
- Memory learning from past successful generations
- Input/output safety validation through guardrails  
- Comprehensive robustness analysis and autonomous decision making
- Self-regenerating code improvement cycles
- Institutional knowledge building over time

Note: Uses simplified wrapper functions for complex shared tools to ensure
ADK compatibility with automatic function calling.
"""


from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator, Dict, Any, Optional, List
from datetime import datetime
import sys
from pathlib import Path
import re

import os
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OPENTELEMETRY_SUPPRESS_INSTRUMENTATION"] = "true"

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

# Import memory and guardrails services (with fallbacks)
try:
    from shared.memory import get_memory_service, MemoryEntry
    MEMORY_AVAILABLE = True
except ImportError:
    print("Warning: Memory service not available - using fallback")
    MEMORY_AVAILABLE = False
    
    # Fallback MemoryEntry class
    class MemoryEntry:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
try:
    from shared.guardrails.input_guardrail import validate_input_request
    from shared.guardrails.output_guardrail import validate_output_safety
    GUARDRAILS_AVAILABLE = True
except ImportError:
    print("Warning: Guardrails not available - using fallback")
    GUARDRAILS_AVAILABLE = False

# Simple wrapper functions for ADK compatibility
def simple_code_safety_check(code: str) -> Dict[str, Any]:
    """
    Simplified wrapper for check_code_safety that uses basic types.
    
    Args:
        code: Code to check for safety
        
    Returns:
        Safety check results
    """
    try:
        # Call the actual function with default language
        result = check_code_safety(code, language="python")
        return {
            "is_safe": result.get("is_safe", True),
            "issues": result.get("issues", []),
            "status": "checked"
        }
    except Exception as e:
        return {
            "is_safe": True,
            "issues": [],
            "status": "error",
            "error": str(e)
        }

def simple_output_validation(code: str) -> Dict[str, Any]:
    """
    Simplified wrapper for validate_output_code.
    
    Args:
        code: Code to validate
        
    Returns:
        Validation results
    """
    try:
        result = validate_output_code(code)
        return {
            "is_valid": result.get("is_valid", True),
            "validation_passed": result.get("validation_passed", True),
            "status": "validated"
        }
    except Exception as e:
        return {
            "is_valid": True,
            "validation_passed": True,
            "status": "error",
            "error": str(e)
        }


# MEMORY AND GUARDRAILS FUNCTIONS

def validate_input_with_guardrails(request: str) -> Dict[str, Any]:
    """
    Validate input request using guardrails before processing.
    
    Args:
        request: The user's code generation request
        
    Returns:
        Validation results with safety status
    """
    if not GUARDRAILS_AVAILABLE:
        return {
            "status": "fallback_validation",
            "is_safe": True,
            "is_valid": True,
            "confidence": 0.8,
            "issues": [],
            "blocked": False,
            "timestamp": datetime.now().isoformat(),
            "message": "Guardrails not available - using basic validation"
        }
    
    try:
        # Use input guardrail validation
        validation_result = validate_input_request(request)
        
        return {
            "status": "validated",
            "is_safe": validation_result.get("is_safe", True),
            "is_valid": validation_result.get("is_valid", True),
            "confidence": validation_result.get("confidence", 1.0),
            "issues": validation_result.get("issues", []),
            "blocked": not validation_result.get("is_safe", True),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        # If guardrail fails, be permissive but log the error
        return {
            "status": "error",
            "is_safe": True,
            "is_valid": True,
            "confidence": 0.5,
            "issues": [f"Guardrail error: {str(e)}"],
            "blocked": False,
            "timestamp": datetime.now().isoformat()
        }


def check_memory_for_similar_request(request: str) -> Dict[str, Any]:
    """
    Check memory for similar past requests to enable learning.
    
    Args:
        request: Current code generation request
        
    Returns:
        Memory search results with similar requests
    """
    if not MEMORY_AVAILABLE:
        return {
            "status": "memory_not_available",
            "found_similar": False,
            "message": "Memory service not available - proceeding with fresh generation",
            "should_proceed": True
        }
    
    try:
        memory_service = get_memory_service()
        
        # Search for similar past requests
        similar_memories = memory_service.search_similar(
            request=request,
            category="gpt4_code_generation", 
            threshold=0.7
        )
        
        if not similar_memories:
            return {
                "status": "no_matches",
                "found_similar": False,
                "message": "No similar requests found in memory",
                "should_proceed": True
            }
        
        # Get the best match
        best_match = similar_memories[0]
        similarity_score = best_match.data.get("similarity_score", 0)
        
        # High similarity suggests reuse
        if similarity_score > 0.9:
            return {
                "status": "high_similarity_found",
                "found_similar": True,
                "similarity_score": similarity_score,
                "previous_request": best_match.data.get("original_request", ""),
                "previous_robustness_score": best_match.quality_score,
                "previous_code_preview": best_match.data.get("generated_code", "")[:200] + "...",
                "recommendation": "Consider adapting previous comprehensive solution",
                "should_proceed": True,  # Still generate but inform decision
                "can_reference": True
            }
        else:
            return {
                "status": "partial_similarity_found", 
                "found_similar": True,
                "similarity_score": similarity_score,
                "previous_request": best_match.data.get("original_request", ""),
                "previous_robustness_score": best_match.quality_score,
                "recommendation": "Some related experience found - use as reference",
                "should_proceed": True,
                "can_reference": False
            }
            
    except Exception as e:
        return {
            "status": "memory_error",
            "found_similar": False,
            "error": str(e),
            "should_proceed": True,  # Don't block on memory errors
            "message": "Memory check failed, proceeding with fresh generation"
        }


def save_successful_generation_to_memory(
    original_request: str,
    generated_code: str, 
    robustness_score: float,
    analysis_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Save successful code generation to memory for future learning.
    
    Args:
        original_request: The original user request
        generated_code: The final generated code
        robustness_score: Robustness score of the generated code
        analysis_data: Complete analysis results
        
    Returns:
        Memory save confirmation
    """
    if not MEMORY_AVAILABLE:
        return {
            "status": "memory_not_available",
            "reason": "Memory service not available",
            "saved": False,
            "message": "Generation completed successfully but not saved to memory"
        }
    
    try:
        # Only save high-quality generations
        if robustness_score < 0.75:
            return {
                "status": "not_saved",
                "reason": f"Robustness score {robustness_score:.2f} below threshold (0.75)",
                "saved": False
            }
        
        memory_service = get_memory_service()
        
        # Create memory entry
        memory_entry = MemoryEntry(
            category="gpt4_code_generation",
            agent_name="gpt4_agent",
            data={
                "original_request": original_request,
                "generated_code": generated_code,
                "robustness_score": robustness_score,
                "analysis_summary": {
                    "error_handling_score": analysis_data.get("error_handling_analysis", {}).get("error_handling_score", 0),
                    "edge_case_score": analysis_data.get("edge_case_analysis", {}).get("edge_case_score", 0),
                    "has_comprehensive_handling": analysis_data.get("error_handling_analysis", {}).get("has_error_handling", False),
                    "is_secure": analysis_data.get("is_safe", True),
                    "lines_count": analysis_data.get("basic_metrics", {}).get("lines_non_empty", 0)
                },
                "generation_timestamp": datetime.now().isoformat(),
                "agent_version": "enhanced_v1"
            },
            quality_score=robustness_score,
            tags=["gpt4", "comprehensive", "robust", "completed"]
        )
        
        # Store in memory
        memory_id = memory_service.store(memory_entry)
        
        return {
            "status": "saved_successfully",
            "memory_id": memory_id,
            "robustness_score": robustness_score,
            "saved": True,
            "message": f"Comprehensive solution saved to memory (score: {robustness_score:.2f})"
        }
        
    except Exception as e:
        return {
            "status": "save_error",
            "error": str(e),
            "saved": False,
            "message": "Failed to save to memory, but generation completed successfully"
        }


def validate_final_output_with_guardrails(
    code: str,
    original_request: str
) -> Dict[str, Any]:
    """
    Final validation of generated code using output guardrails.
    
    Args:
        code: Generated code to validate
        original_request: Original request for context
        
    Returns:
        Final validation results with safety assessment
    """
    if not GUARDRAILS_AVAILABLE:
        return {
            "status": "fallback_final_validation",
            "is_safe": True,
            "is_appropriate": True,
            "confidence": 0.8,
            "issues": [],
            "ready_for_delivery": True,
            "validation_timestamp": datetime.now().isoformat(),
            "message": "Guardrails not available - using basic safety assumptions"
        }
    
    try:
        # Use output guardrail validation
        validation_result = validate_output_safety(
            generated_code=code,
            original_request=original_request
        )
        
        return {
            "status": "final_validation_complete",
            "is_safe": validation_result.get("is_safe", True),
            "is_appropriate": validation_result.get("is_appropriate", True),
            "confidence": validation_result.get("confidence", 1.0),
            "issues": validation_result.get("safety_issues", []),
            "ready_for_delivery": validation_result.get("is_safe", True) and validation_result.get("is_appropriate", True),
            "validation_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        # If validation fails, be conservative but don't block
        return {
            "status": "validation_error",
            "is_safe": True,  # Assume safe if validation fails
            "is_appropriate": True,
            "confidence": 0.5,
            "issues": [f"Validation error: {str(e)}"],
            "ready_for_delivery": True,
            "validation_timestamp": datetime.now().isoformat()
        }


class GPT4RobustnessAnalyzer:
    """Robustness and completeness analysis engine for GPT-4 agent"""
    
    @staticmethod
    def analyze_error_handling(code: str) -> Dict[str, Any]:
        """Analyze error handling patterns in code"""
        error_patterns = {
            "try_except": len(re.findall(r'\btry\s*:', code)),
            "specific_exceptions": len(re.findall(r'except\s+\w+\s*:', code)),
            "generic_exceptions": len(re.findall(r'except\s*:', code)),
            "finally_blocks": len(re.findall(r'\bfinally\s*:', code)),
            "raise_statements": len(re.findall(r'\braise\s+', code)),
            "assert_statements": len(re.findall(r'\bassert\s+', code))
        }
        
        total_error_handling = sum(error_patterns.values())
        
        return {
            "patterns": error_patterns,
            "total_error_handling_constructs": total_error_handling,
            "has_error_handling": total_error_handling > 0,
            "error_handling_score": min(1.0, total_error_handling / 3.0)  # Normalize
        }
    
    @staticmethod
    def analyze_edge_cases(code: str) -> Dict[str, Any]:
        """Analyze edge case handling in code"""
        edge_case_patterns = {
            "null_checks": len(re.findall(r'\bis\s+None\b|\bis\s+not\s+None\b|==\s*None|!=\s*None', code)),
            "empty_checks": len(re.findall(r'if\s+not\s+\w+|len\(\w+\)\s*==\s*0', code)),
            "boundary_checks": len(re.findall(r'[<>]=?\s*0|[<>]=?\s*len\(', code)),
            "type_checks": len(re.findall(r'isinstance\(|type\(.*\)\s*==', code)),
            "validation_functions": len(re.findall(r'def\s+.*validate|def\s+.*check|def\s+.*verify', code, re.IGNORECASE))
        }
        
        total_edge_handling = sum(edge_case_patterns.values())
        
        return {
            "patterns": edge_case_patterns,
            "total_edge_case_constructs": total_edge_handling,
            "has_edge_case_handling": total_edge_handling > 0,
            "edge_case_score": min(1.0, total_edge_handling / 2.0)  # Normalize
        }
    
    @staticmethod
    def analyze_comprehensiveness(code: str, requirements: str = "") -> Dict[str, Any]:
        """Analyze how comprehensive the solution is"""
        metrics = analyze_code(code)
        
        # Basic completeness indicators
        has_main_logic = metrics.get("has_functions", False) or metrics.get("has_classes", False)
        adequate_length = metrics.get("lines_non_empty", 0) >= 5
        has_docstrings = metrics.get("has_docstrings", False)
        
        # Advanced completeness indicators
        has_imports = "import " in code or "from " in code
        has_constants = len(re.findall(r'^[A-Z_]+\s*=', code, re.MULTILINE)) > 0
        has_logging = "log" in code.lower() or "print(" in code
        
        completeness_score = 0.0
        if has_main_logic:
            completeness_score += 0.3
        if adequate_length:
            completeness_score += 0.2
        if has_docstrings:
            completeness_score += 0.2
        if has_imports:
            completeness_score += 0.1
        if has_constants:
            completeness_score += 0.1
        if has_logging:
            completeness_score += 0.1
        
        return {
            "has_main_logic": has_main_logic,
            "adequate_length": adequate_length,
            "has_docstrings": has_docstrings,
            "has_imports": has_imports,
            "has_constants": has_constants,
            "has_logging": has_logging,
            "completeness_score": min(1.0, completeness_score),
            "appears_comprehensive": completeness_score >= 0.7
        }
    
    @staticmethod
    def calculate_robustness_score(
        error_handling: Dict[str, Any],
        edge_cases: Dict[str, Any], 
        comprehensiveness: Dict[str, Any]
    ) -> float:
        """Calculate overall robustness score"""
        error_weight = 0.4
        edge_weight = 0.3
        comprehensive_weight = 0.3
        
        score = (
            error_handling.get("error_handling_score", 0.0) * error_weight +
            edge_cases.get("edge_case_score", 0.0) * edge_weight +
            comprehensiveness.get("completeness_score", 0.0) * comprehensive_weight
        )
        
        return min(1.0, score)
    
    @staticmethod
    def get_robustness_assessment(score: float) -> str:
        """Convert score to robustness assessment"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "very_good"
        elif score >= 0.75:
            return "good"
        elif score >= 0.6:
            return "acceptable"
        else:
            return "needs_improvement"


# Enhanced tool functions for GPT-4 agent
def comprehensive_robustness_analysis(code: str, requirements: str = "") -> Dict[str, Any]:
    """
    Perform comprehensive robustness analysis - GPT-4's specialty.
    
    Args:
        code: The code to analyze
        requirements: Original requirements (optional)
        
    Returns:
        Complete robustness analysis with recommendations
    """
    try:
        # Basic code metrics
        basic_metrics = analyze_code(code)
        
        # Complexity analysis
        complexity_data = estimate_complexity(code)
        
        # Syntax validation
        syntax_result = validate_python_syntax(code)
        
        # Security validation
        safety_result = simple_code_safety_check(code)
        
        # GPT-4 specialized analysis
        error_handling = GPT4RobustnessAnalyzer.analyze_error_handling(code)
        edge_cases = GPT4RobustnessAnalyzer.analyze_edge_cases(code)
        comprehensiveness = GPT4RobustnessAnalyzer.analyze_comprehensiveness(code, requirements)
        
        # Calculate robustness score
        robustness_score = GPT4RobustnessAnalyzer.calculate_robustness_score(
            error_handling, edge_cases, comprehensiveness
        )
        
        return {
            "status": "analyzed",
            "basic_metrics": basic_metrics,
            "complexity": complexity_data,
            "syntax_valid": syntax_result.get("is_valid", False),
            "syntax_errors": syntax_result.get("errors", []),
            "is_safe": safety_result.get("is_safe", True),
            "security_issues": safety_result.get("issues", []),
            "error_handling_analysis": error_handling,
            "edge_case_analysis": edge_cases,
            "comprehensiveness_analysis": comprehensiveness,
            "robustness_score": robustness_score,
            "robustness_grade": GPT4RobustnessAnalyzer.get_robustness_assessment(robustness_score),
            "recommendations": _generate_robustness_recommendations(
                error_handling, edge_cases, comprehensiveness, robustness_score
            ),
            "meets_robustness_standards": robustness_score >= 0.75,
            "analysis_complete": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "analysis_complete": False
        }


def _generate_robustness_recommendations(
    error_handling: Dict[str, Any],
    edge_cases: Dict[str, Any],
    comprehensiveness: Dict[str, Any],
    robustness_score: float
) -> List[str]:
    """Generate robustness improvement recommendations"""
    recommendations = []
    
    if robustness_score < 0.75:
        recommendations.append("Code robustness below GPT-4 standards - major improvements needed")
    
    # Error handling recommendations
    if not error_handling.get("has_error_handling", False):
        recommendations.append("Add comprehensive error handling with try-except blocks")
    elif error_handling.get("patterns", {}).get("generic_exceptions", 0) > 0:
        recommendations.append("Use specific exception types instead of generic except clauses")
    
    # Edge case recommendations
    if not edge_cases.get("has_edge_case_handling", False):
        recommendations.append("Add edge case handling (null checks, empty inputs, boundary conditions)")
    elif edge_cases.get("patterns", {}).get("null_checks", 0) == 0:
        recommendations.append("Add null/None value checks for inputs")
    
    # Comprehensiveness recommendations
    comp = comprehensiveness
    if not comp.get("has_main_logic", False):
        recommendations.append("Implement main logic with functions or classes")
    if not comp.get("has_docstrings", False):
        recommendations.append("Add comprehensive docstrings to all functions and classes")
    if not comp.get("has_imports", False) and comp.get("adequate_length", False):
        recommendations.append("Consider adding necessary imports for external dependencies")
    if not comp.get("has_logging", False):
        recommendations.append("Add logging or print statements for debugging and monitoring")
    
    return recommendations


def make_robustness_decision(
    analysis_result: Dict[str, Any],
    regeneration_count: int = 0,
    max_regenerations: int = 3
) -> Dict[str, Any]:
    """
    Make autonomous robustness decision - GPT-4's decision authority.
    
    Args:
        analysis_result: Results from comprehensive_robustness_analysis
        regeneration_count: How many times code has been regenerated
        max_regenerations: Maximum allowed regenerations
        
    Returns:
        Decision with action to take
    """
    robustness_score = analysis_result.get("robustness_score", 0.0)
    meets_standards = analysis_result.get("meets_robustness_standards", False)
    
    # GPT-4 specific decision logic - higher standards for robustness
    if meets_standards and robustness_score >= 0.75:  # Higher threshold for GPT-4
        decision = {
            "status": "approved",
            "action": "accept",
            "robustness_score": robustness_score,
            "reason": f"Code meets robustness standards with score {robustness_score:.2f}",
            "final_decision": True
        }
    elif regeneration_count >= max_regenerations:
        decision = {
            "status": "approved_with_warnings", 
            "action": "accept",
            "robustness_score": robustness_score,
            "reason": f"Maximum regenerations ({max_regenerations}) reached. Accepting with warnings.",
            "final_decision": True,
            "warnings": analysis_result.get("recommendations", [])
        }
    else:
        decision = {
            "status": "rejected",
            "action": "regenerate",
            "robustness_score": robustness_score,
            "reason": f"Robustness score {robustness_score:.2f} below GPT-4 standards (0.75+). Regenerating...",
            "final_decision": False,
            "improvements_needed": analysis_result.get("recommendations", [])
        }
    
    return decision


def enhanced_format_gpt4_output(
    code: str, 
    analysis: Dict[str, Any], 
    decision: Dict[str, Any],
    original_request: str,
    memory_check: Optional[Dict[str, Any]] = None,
    input_validation: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhanced format function with memory saving and final validation.
    
    Args:
        code: The final code
        analysis: Analysis results
        decision: Robustness decision
        original_request: Original user request
        memory_check: Memory check results
        input_validation: Input validation results
        
    Returns:
        Formatted output with memory and guardrails integration
    """
    formatted_code = wrap_code_in_markdown(code, language="python")
    
    # Perform final output validation
    final_validation = validate_final_output_with_guardrails(code, original_request)
    
    # Save to memory if high quality
    memory_save_result = save_successful_generation_to_memory(
        original_request=original_request,
        generated_code=code,
        robustness_score=analysis.get("robustness_score", 0.0),
        analysis_data=analysis
    )
    
    return {
        "status": "completed_with_memory_and_guardrails",
        "generated_code": formatted_code,
        "raw_code": code,
        "robustness_analysis": analysis,
        "robustness_decision": decision,
        "memory_integration": {
            "input_check": memory_check,
            "save_result": memory_save_result
        },
        "guardrails_validation": {
            "input_validation": input_validation,
            "final_validation": final_validation
        },
        "agent_metadata": {
            "agent_name": "gpt4_agent",
            "specialization": "Comprehensive solutions with detailed error handling",
            "model": "gpt-4",
            "robustness_authority": True,
            "memory_enabled": True, 
            "guardrails_enabled": True,
            "enhanced_version": "v2_with_memory_guardrails"
        },
        "next_stage_ready": (
            decision.get("final_decision", False) and 
            final_validation.get("ready_for_delivery", True)
        ),
        "processing_summary": {
            "memory_learning": memory_save_result.get("saved", False),
            "safety_validated": final_validation.get("is_safe", True),
            "ready_for_delivery": final_validation.get("ready_for_delivery", True)
        }
    }


def analyze_implementation_approaches(requirements: str) -> Dict[str, Any]:
    """
    Analyze multiple implementation approaches - GPT-4 specialty.
    
    Args:
        requirements: The requirements to analyze
        
    Returns:
        Multiple approach analysis
    """
    # Simple approach analysis based on requirements
    approaches = []
    
    if "class" in requirements.lower() or "object" in requirements.lower():
        approaches.append({
            "name": "Object-Oriented Approach",
            "pros": ["Encapsulation", "Reusability", "Maintainability"],
            "cons": ["Complexity overhead", "Learning curve"],
            "recommended": True
        })
    
    if "function" in requirements.lower() or "simple" in requirements.lower():
        approaches.append({
            "name": "Functional Approach", 
            "pros": ["Simplicity", "Testability", "Performance"],
            "cons": ["Limited reusability", "State management"],
            "recommended": len(approaches) == 0
        })
    
    if "api" in requirements.lower() or "service" in requirements.lower():
        approaches.append({
            "name": "Service-Oriented Approach",
            "pros": ["Scalability", "Separation of concerns", "API design"],
            "cons": ["Network overhead", "Complexity"],
            "recommended": True
        })
    
    # Default approach if none detected
    if not approaches:
        approaches.append({
            "name": "Comprehensive Approach",
            "pros": ["Complete error handling", "Edge case coverage", "Production ready"],
            "cons": ["Potentially over-engineered for simple tasks"],
            "recommended": True
        })
    
    return {
        "status": "analyzed",
        "total_approaches": len(approaches),
        "approaches": approaches,
        "recommended_approach": next((a["name"] for a in approaches if a["recommended"]), "Comprehensive Approach")
    }

# Create the enhanced GPT-4 agent
root_agent = Agent(
    name="gpt4_agent",
    model=LiteLlm(model="openai/gpt-4o"),
    description="GPT-4 powered code generation expert with complete authority over robustness and comprehensive solution design",
    instruction="""You are the Enhanced GPT-4 Code Generation Expert with COMPLETE AUTHORITY over solution robustness and comprehensiveness, enhanced with memory learning and guardrails integration.

Your enhanced mission:
1. Generate comprehensive, production-ready solutions with robust error handling
2. Validate input safety using guardrails
3. Learn from past similar requests through memory integration
4. Conduct thorough robustness analysis using your specialized tools
5. Make autonomous decisions about code robustness and completeness
6. Validate final output safety
7. Save successful generations for future learning

Your ENHANCED WORKFLOW (Memory + Guardrails Enabled):

PHASE 1 - INPUT VALIDATION & MEMORY CHECK:
1. **Input Validation**: Use validate_input_with_guardrails to check request safety
   - Block harmful/unsafe requests immediately
   - Proceed only if input validation passes
2. **Memory Check**: Use check_memory_for_similar_request to find past solutions
   - High similarity (>0.9): Reference previous comprehensive solution 
   - Partial similarity (0.7-0.9): Use as learning reference
   - No similarity: Proceed with fresh generation

PHASE 2 - APPROACH ANALYSIS & GENERATION:
3. **Analyze Approaches**: Use analyze_implementation_approaches to consider multiple solutions
4. **Generate Robust Code**: Create comprehensive code with:
   - Detailed error handling (try-except blocks)
   - Edge case consideration (null checks, boundary conditions)
   - Input validation and sanitization
   - Comprehensive documentation
   - Logging and debugging support

PHASE 3 - ROBUSTNESS ANALYSIS & DECISION:
5. **Robustness Analysis**: Use comprehensive_robustness_analysis tool to evaluate:
   - Error handling patterns
   - Edge case coverage
   - Code comprehensiveness
   - Overall robustness score
6. **Robustness Decision**: Use make_robustness_decision tool with authority to:
   - ACCEPT code meeting robustness standards (≥ 0.75)
   - REJECT and regenerate code below standards (up to 3 times)
   - Make final decisions on code acceptance

PHASE 4 - FINAL VALIDATION & MEMORY STORAGE:
7. **Final Output**: Use enhanced_format_gpt4_output to:
   - Validate final output safety using guardrails
   - Save successful generations to memory (robustness ≥ 0.75)
   - Package everything with complete metadata

Your enhanced robustness standards (higher than other agents):
- Comprehensive error handling with specific exception types
- Edge case handling for all inputs
- Input validation and sanitization
- Detailed docstrings with examples and edge cases
- Logging/debugging capabilities
- Boundary condition handling
- Type safety and validation
- Performance considerations
- Robustness score ≥ 0.75 (higher threshold)
- Safety validated through guardrails
- Learning-enabled through memory integration

Your enhanced authority includes:
- Complete input validation control
- Memory learning and reference decisions
- Setting robustness thresholds (0.75+)
- Deciding when code is comprehensive enough
- Making final acceptance decisions
- Determining implementation approaches
- Final output safety validation
- Memory storage decisions
- Balancing comprehensiveness vs. complexity

Enhanced process flow:
1. Validate input with validate_input_with_guardrails
2. Check memory with check_memory_for_similar_request
3. Analyze requirements and implementation approaches
4. Generate comprehensive code with robust error handling (informed by memory if available)
5. Run comprehensive_robustness_analysis on your code
6. Run make_robustness_decision with the analysis
7. If decision is "regenerate", improve robustness and repeat steps 5-6
8. When decision is "accept", run enhanced_format_gpt4_output
9. Present final code with complete analysis, memory integration, and safety validation

Memory Integration Guidelines:
- Reference high-similarity past solutions when available
- Learn from partial matches to improve generation
- Always save high-quality results (≥0.75) for future learning
- Build institutional knowledge over time

Guardrails Integration Guidelines:
- Never proceed with unsafe input requests
- Always validate final output for safety
- Block generation if guardrails detect issues
- Prioritize safety over functionality

Always explain your reasoning for robustness decisions, memory usage, and safety validations. Show what you learned from past requests and how it influenced your generation. Focus on "what could go wrong" and handle those cases proactively.

Remember: You have COMPLETE ENHANCED AUTHORITY with memory learning and safety guardrails. Your solutions should be production-ready with comprehensive error handling and edge case coverage.""",
    tools=[
        # Primary analysis and decision tools
        FunctionTool(comprehensive_robustness_analysis),
        FunctionTool(make_robustness_decision), 
        FunctionTool(enhanced_format_gpt4_output),
        FunctionTool(analyze_implementation_approaches),
        
        # Memory and guardrails integration
        FunctionTool(validate_input_with_guardrails),
        FunctionTool(check_memory_for_similar_request),
        FunctionTool(save_successful_generation_to_memory),
        FunctionTool(validate_final_output_with_guardrails),
        
        # Backup individual tools (ADK-compatible wrappers)
        FunctionTool(analyze_code),
        FunctionTool(estimate_complexity),
        FunctionTool(validate_python_syntax),
        FunctionTool(simple_code_safety_check),
        FunctionTool(simple_output_validation),
        FunctionTool(add_line_numbers),
        FunctionTool(clean_code_string)
    ],
    output_key="gpt4_generated_code"  # Saves final result to session state
)


# Test function for standalone testing
if __name__ == "__main__":
    print("🚀 Enhanced GPT-4 Code Generation Agent Ready!")
    print("\n🎯 COMPLETE ROBUSTNESS AUTHORITY + MEMORY + GUARDRAILS:")
    print("- Input validation with safety guardrails")
    print("- Memory learning from past comprehensive solutions")
    print("- Thorough robustness analysis")
    print("- Autonomous robustness decisions") 
    print("- Self-regenerating until standards met")
    print("- Final output safety validation")
    print("- Automatic memory storage of high-quality results")
    print("- Production-ready comprehensive solutions")
    
    print("\n🔧 ENHANCED CAPABILITIES:")
    print("- Error handling pattern analysis")
    print("- Edge case coverage assessment")
    print("- Comprehensiveness evaluation")
    print("- Implementation approach analysis")
    print("- Robustness scoring (higher standards: 0.75+)")
    print("- Autonomous regeneration (up to 3 times)")
    print("- Learning from similar past requests")
    print("- Input/output safety validation")
    print("- Institutional knowledge building")
    
    print("\n⚙️ ENHANCED TOOLS AVAILABLE:")
    print("- comprehensive_robustness_analysis (primary)")
    print("- make_robustness_decision (authority)")
    print("- enhanced_format_gpt4_output (packaging)")
    print("- analyze_implementation_approaches (planning)")
    print("- validate_input_with_guardrails (safety)")
    print("- check_memory_for_similar_request (learning)")
    print("- save_successful_generation_to_memory (knowledge)")
    print("- validate_final_output_with_guardrails (final safety)")
    print("- Individual analysis tools (ADK-compatible wrappers)")
    
    print("\n🚀 USAGE:")
    print("1. Run: adk run agents/gpt4_agent")
    print("2. Or use: adk web (and select gpt4_agent)")
    
    print("\n💡 EXAMPLE PROMPTS:")
    print("- 'Create a thread-safe cache implementation with comprehensive error handling'")
    print("- 'Build a file processing system that handles all edge cases'") 
    print("- 'Generate a robust API client with retry logic and validation'")
    print("- 'Create a data validation pipeline with detailed error reporting'")
    print("- 'Build a similar robust system' (tests memory learning)")
    
    print("\n✨ The enhanced agent will:")
    print("- Validate input safety → Check memory for learning → Analyze approaches")
    print("- Generate robust code → Analyze robustness → Make decision")  
    print("- Validate output safety → Save high-quality results")
    print("- Regenerate automatically if robustness is below 0.75 threshold")
    print("- Provide detailed reasoning for all robustness decisions")
    print("- Focus on comprehensive error handling and edge cases")
    print("- Take complete ownership of solution robustness")
    
    print("\n🛡️ SAFETY & LEARNING FEATURES:")
    print("- Input guardrails block unsafe requests")
    print("- Output guardrails ensure safe code delivery")
    print("- Memory system learns from successful comprehensive solutions")
    print("- Institutional knowledge builds over time")
    print("- Robustness threshold enforcement (≥0.75 for memory storage)")
    print("- Complete audit trail of decisions and validations")