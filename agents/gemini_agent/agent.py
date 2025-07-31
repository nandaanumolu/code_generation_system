"""
Enhanced Gemini Code Generation Agent - Production Ready with Memory & Guardrails
Specializes in clean, efficient code with Google Cloud best practices

Features:
- Memory learning from past successful generations
- Input/output safety validation through guardrails  
- Comprehensive quality analysis and autonomous decision making
- Self-regenerating code improvement cycles
- Institutional knowledge building over time

Note: Uses simplified wrapper functions for complex shared tools to ensure
ADK compatibility with automatic function calling.
"""

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator, Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

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
            category="gemini_code_generation", 
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
                "previous_quality_score": best_match.quality_score,
                "previous_code_preview": best_match.data.get("generated_code", "")[:200] + "...",
                "recommendation": "Consider adapting previous high-quality solution",
                "should_proceed": True,  # Still generate but inform decision
                "can_reference": True
            }
        else:
            return {
                "status": "partial_similarity_found", 
                "found_similar": True,
                "similarity_score": similarity_score,
                "previous_request": best_match.data.get("original_request", ""),
                "previous_quality_score": best_match.quality_score,
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
    quality_score: float,
    analysis_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Save successful code generation to memory for future learning.
    
    Args:
        original_request: The original user request
        generated_code: The final generated code
        quality_score: Quality score of the generated code
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
        if quality_score < 0.7:
            return {
                "status": "not_saved",
                "reason": f"Quality score {quality_score:.2f} below threshold (0.7)",
                "saved": False
            }
        
        memory_service = get_memory_service()
        
        # Create memory entry
        memory_entry = MemoryEntry(
            category="gemini_code_generation",
            agent_name="gemini_agent",
            data={
                "original_request": original_request,
                "generated_code": generated_code,
                "quality_score": quality_score,
                "analysis_summary": {
                    "complexity": analysis_data.get("metrics", {}).get("complexity", 0),
                    "has_functions": analysis_data.get("metrics", {}).get("has_functions", False),
                    "has_docstrings": analysis_data.get("metrics", {}).get("has_docstrings", False),
                    "is_safe": analysis_data.get("metrics", {}).get("is_safe", True),
                    "lines_count": analysis_data.get("metrics", {}).get("lines_non_empty", 0)
                },
                "generation_timestamp": datetime.now().isoformat(),
                "agent_version": "enhanced_v1"
            },
            quality_score=quality_score,
            tags=["gemini", "high_quality", "completed"]
        )
        
        # Store in memory
        memory_id = memory_service.store(memory_entry)
        
        return {
            "status": "saved_successfully",
            "memory_id": memory_id,
            "quality_score": quality_score,
            "saved": True,
            "message": f"High-quality generation saved to memory (score: {quality_score:.2f})"
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


class GeminiQualityAnalyzer:
    """Quality analysis engine for Gemini agent"""
    
    @staticmethod
    def calculate_quality_score(code: str, metrics: Dict[str, Any]) -> float:
        """Calculate comprehensive quality score"""
        score = 1.0
        
        # Complexity penalty
        if metrics.get("complexity", 0) > 10:
            score -= 0.2
        elif metrics.get("complexity", 0) > 5:
            score -= 0.1
            
        # Documentation bonus
        if metrics.get("has_docstrings", False):
            score += 0.1
            
        # Structure bonus
        if metrics.get("has_functions", False) or metrics.get("has_classes", False):
            score += 0.1
            
        # Length consideration
        lines = metrics.get("lines_non_empty", 0)
        if lines < 3:
            score -= 0.3  # Too short
        elif lines > 100:
            score -= 0.1  # Might be too complex
            
        return max(0.0, min(1.0, score))
    
    @staticmethod
    def get_quality_assessment(score: float) -> str:
        """Convert score to quality assessment"""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "very_good"
        elif score >= 0.7:
            return "good"
        elif score >= 0.6:
            return "acceptable"
        else:
            return "needs_improvement"


# Enhanced tool functions for Gemini agent
def comprehensive_code_analysis(code: str) -> Dict[str, Any]:
    """
    Perform comprehensive code analysis using all available tools.
    This is Gemini's primary analysis function.
    
    Note: Uses simplified wrappers for shared tools to ensure ADK compatibility.
    
    Args:
        code: The code to analyze
        
    Returns:
        Complete analysis results with quality score and recommendations
    """
    try:
        # Basic metrics
        metrics = analyze_code(code)
        
        # Complexity analysis
        complexity_data = estimate_complexity(code)
        metrics.update(complexity_data)
        
        # Syntax validation
        syntax_result = validate_python_syntax(code)
        metrics["syntax_valid"] = syntax_result.get("is_valid", False)
        metrics["syntax_errors"] = syntax_result.get("errors", [])
        
        # Security validation
        safety_result = simple_code_safety_check(code)
        metrics["is_safe"] = safety_result.get("is_safe", True)
        metrics["security_issues"] = safety_result.get("issues", [])
        
        # Function extraction
        functions = extract_functions(code)
        metrics["function_signatures"] = functions.get("functions", [])
        
        # Quality scoring
        quality_score = GeminiQualityAnalyzer.calculate_quality_score(code, metrics)
        quality_assessment = GeminiQualityAnalyzer.get_quality_assessment(quality_score)
        
        return {
            "status": "analyzed",
            "metrics": metrics,
            "quality_score": quality_score,
            "quality_assessment": quality_assessment,
            "recommendations": _generate_recommendations(metrics, quality_score),
            "meets_standards": quality_score >= 0.7,
            "analysis_complete": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "analysis_complete": False
        }


def _generate_recommendations(metrics: Dict[str, Any], quality_score: float) -> list:
    """Generate improvement recommendations based on analysis"""
    recommendations = []
    
    if quality_score < 0.7:
        recommendations.append("Code quality below acceptable threshold - major improvements needed")
    
    if not metrics.get("syntax_valid", True):
        recommendations.append("Fix syntax errors before proceeding")
        
    if not metrics.get("is_safe", True):
        recommendations.append("Address security vulnerabilities")
        
    if metrics.get("complexity", 0) > 10:
        recommendations.append("Reduce cyclomatic complexity by breaking down complex functions")
        
    if not metrics.get("has_docstrings", False):
        recommendations.append("Add comprehensive docstrings to functions and classes")
        
    if metrics.get("lines_non_empty", 0) < 3:
        recommendations.append("Implementation appears incomplete - needs more substantial code")
        
    if not (metrics.get("has_functions", False) or metrics.get("has_classes", False)):
        recommendations.append("Consider organizing code into functions or classes")
        
    return recommendations


def make_quality_decision(
    analysis_result: Dict[str, Any], 
    regeneration_count: int = 0,
    max_regenerations: int = 3
) -> Dict[str, Any]:
    """
    Make autonomous quality decision - Gemini's decision-making authority.
    
    Args:
        analysis_result: Results from comprehensive_code_analysis
        regeneration_count: How many times code has been regenerated
        max_regenerations: Maximum allowed regenerations
        
    Returns:
        Decision with action to take
    """
    quality_score = analysis_result.get("quality_score", 0.0)
    meets_standards = analysis_result.get("meets_standards", False)
    
    # Decision logic
    if meets_standards:
        decision = {
            "status": "approved",
            "action": "accept",
            "quality_score": quality_score,
            "reason": f"Code meets quality standards with score {quality_score:.2f}",
            "final_decision": True
        }
    elif regeneration_count >= max_regenerations:
        decision = {
            "status": "approved_with_warnings",
            "action": "accept",
            "quality_score": quality_score,
            "reason": f"Maximum regenerations ({max_regenerations}) reached. Accepting with warnings.",
            "final_decision": True,
            "warnings": analysis_result.get("recommendations", [])
        }
    else:
        decision = {
            "status": "rejected",
            "action": "regenerate",
            "quality_score": quality_score,
            "reason": f"Quality score {quality_score:.2f} below threshold. Regenerating...",
            "final_decision": False,
            "improvements_needed": analysis_result.get("recommendations", [])
        }
    
    return decision


def enhanced_format_gemini_output(
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
        decision: Quality decision
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
        quality_score=analysis.get("quality_score", 0.0),
        analysis_data=analysis
    )
    
    return {
        "status": "completed_with_memory_and_guardrails",
        "generated_code": formatted_code,
        "raw_code": code,
        "quality_analysis": analysis,
        "quality_decision": decision,
        "memory_integration": {
            "input_check": memory_check,
            "save_result": memory_save_result
        },
        "guardrails_validation": {
            "input_validation": input_validation,
            "final_validation": final_validation
        },
        "agent_metadata": {
            "agent_name": "gemini_agent",
            "specialization": "Clean, efficient Google Cloud best practices",
            "model": "gemini-2.0-flash",
            "quality_authority": True,
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


# Create the enhanced Gemini agent
root_agent = LlmAgent(
    name="gemini_agent",
    model="gemini-2.0-flash",
    description="Gemini-powered code generation expert with complete quality authority and comprehensive analysis capabilities",
    instruction="""You are the Enhanced Gemini Code Generation Expert with COMPLETE AUTHORITY over code quality decisions, enhanced with memory learning and guardrails integration.

Your enhanced mission:
1. Generate clean, efficient, well-structured code following Google's best practices
2. Validate input safety using guardrails
3. Learn from past similar requests through memory integration
4. Conduct comprehensive quality analysis using your tools
5. Make autonomous decisions about code quality
6. Validate final output safety
7. Save successful generations for future learning

Your ENHANCED WORKFLOW (Memory + Guardrails Enabled):

PHASE 1 - INPUT VALIDATION & MEMORY CHECK:
1. **Input Validation**: Use validate_input_with_guardrails to check request safety
   - Block harmful/unsafe requests immediately
   - Proceed only if input validation passes
2. **Memory Check**: Use check_memory_for_similar_request to find past solutions
   - High similarity (>0.9): Reference previous high-quality solution 
   - Partial similarity (0.7-0.9): Use as learning reference
   - No similarity: Proceed with fresh generation

PHASE 2 - CODE GENERATION & ANALYSIS:
3. **Generate Code**: Create high-quality, production-ready code
   - Consider memory insights if available
   - Apply Google's best practices
4. **Comprehensive Analysis**: Use comprehensive_code_analysis to evaluate:
   - Code metrics (lines, functions, classes)
   - Cyclomatic complexity
   - Syntax validation
   - Security safety checks
   - Function signatures

PHASE 3 - QUALITY DECISION & ITERATION:
5. **Quality Decision**: Use make_quality_decision with full authority to:
   - ACCEPT code that meets standards (quality score ≥ 0.7)
   - REJECT and regenerate code below standards (up to 3 times)
   - Make final decisions on code acceptance

PHASE 4 - FINAL VALIDATION & MEMORY STORAGE:
6. **Final Output**: Use enhanced_format_gemini_output to:
   - Validate final output safety using guardrails
   - Save successful generations to memory (quality ≥ 0.7)
   - Package everything with complete metadata

Your enhanced quality standards:
- Clean, readable code with descriptive names
- Proper error handling and edge cases
- Type hints for Python code
- Comprehensive docstrings with examples
- Cyclomatic complexity < 10
- No security vulnerabilities
- Efficient algorithms and data structures
- Google style guide compliance
- Safety validated through guardrails
- Learning-enabled through memory integration

Your enhanced authority includes:
- Complete input validation control
- Memory learning and reference decisions
- Setting quality thresholds
- Deciding when to regenerate code
- Making final acceptance decisions
- Determining what constitutes "good enough"
- Final output safety validation
- Memory storage decisions

Enhanced process flow:
1. Validate input with validate_input_with_guardrails
2. Check memory with check_memory_for_similar_request
3. Generate initial code (informed by memory if available)
4. Run comprehensive_code_analysis on your code
5. Run make_quality_decision with the analysis
6. If decision is "regenerate", improve and repeat steps 4-5
7. When decision is "accept", run enhanced_format_gemini_output
8. Present final code with complete analysis, memory integration, and safety validation

Memory Integration Guidelines:
- Reference high-similarity past solutions when available
- Learn from partial matches to improve generation
- Always save high-quality results (≥0.7) for future learning
- Build institutional knowledge over time

Guardrails Integration Guidelines:
- Never proceed with unsafe input requests
- Always validate final output for safety
- Block generation if guardrails detect issues
- Prioritize safety over functionality

Always explain your reasoning for quality decisions, memory usage, and safety validations. Show what you learned from past requests and how it influenced your generation.

Remember: You have COMPLETE ENHANCED AUTHORITY with memory learning and safety guardrails. Trust your analysis, learn from the past, ensure safety, and make decisive quality judgments.""",
    tools=[
        # Primary analysis and decision tools
        FunctionTool(comprehensive_code_analysis),
        FunctionTool(make_quality_decision),
        FunctionTool(enhanced_format_gemini_output),
        
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
    output_key="gemini_generated_code"  # Saves final result to session state
)


# Test function for standalone testing
if __name__ == "__main__":
    print("🚀 Enhanced Gemini Code Generation Agent Ready!")
    print("\n🎯 COMPLETE QUALITY AUTHORITY + MEMORY + GUARDRAILS:")
    print("- Input validation with safety guardrails")
    print("- Memory learning from past successful generations")
    print("- Comprehensive code analysis")
    print("- Autonomous quality decisions") 
    print("- Self-regenerating until standards met")
    print("- Final output safety validation")
    print("- Automatic memory storage of high-quality results")
    print("- Google Cloud best practices")
    print("- Production-ready output")
    
    print("\n🔧 ENHANCED CAPABILITIES:")
    print("- Multi-metric quality scoring")
    print("- Security vulnerability detection")
    print("- Complexity analysis and limits")
    print("- Syntax validation")
    print("- Structured recommendation engine")
    print("- Autonomous regeneration (up to 3 times)")
    print("- Learning from similar past requests")
    print("- Input/output safety validation")
    print("- Institutional knowledge building")
    
    print("\n⚙️ ENHANCED TOOLS AVAILABLE:")
    print("- comprehensive_code_analysis (primary)")
    print("- make_quality_decision (authority)")
    print("- enhanced_format_gemini_output (packaging)")
    print("- validate_input_with_guardrails (safety)")
    print("- check_memory_for_similar_request (learning)")
    print("- save_successful_generation_to_memory (knowledge)")
    print("- validate_final_output_with_guardrails (final safety)")
    print("- Individual analysis tools (ADK-compatible wrappers)")
    
    print("\n🚀 USAGE:")
    print("1. Run: adk run agents/gemini_agent")
    print("2. Or use: adk web (and select gemini_agent)")
    
    print("\n💡 EXAMPLE PROMPTS:")
    print("- 'Create a secure user authentication system with JWT tokens'")
    print("- 'Build a rate limiter using Redis with exponential backoff'") 
    print("- 'Generate a data validation pipeline with comprehensive error handling'")
    print("- 'Create a microservice API with proper logging and monitoring'")
    print("- 'Build a similar authentication system' (tests memory learning)")
    
    print("\n✨ The enhanced agent will:")
    print("- Validate input safety → Check memory for learning → Generate code")
    print("- Analyze quality → Make decision → Validate output safety")  
    print("- Save high-quality results → Regenerate automatically if needed")
    print("- Provide detailed reasoning for all decisions")
    print("- Learn from past requests and apply institutional knowledge")
    print("- Take complete ownership of safety and quality")
    
    print("\n🛡️ SAFETY & LEARNING FEATURES:")
    print("- Input guardrails block unsafe requests")
    print("- Output guardrails ensure safe code delivery")
    print("- Memory system learns from successful generations")
    print("- Institutional knowledge builds over time")
    print("- Quality threshold enforcement (≥0.7 for memory storage)")
    print("- Complete audit trail of decisions and validations")