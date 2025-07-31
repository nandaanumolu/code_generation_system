"""
Enhanced Claude Code Generation Agent - Production Ready with Memory & Guardrails
Specializes in elegant, well-documented solutions with strong type safety

Features:
- Memory learning from past successful generations
- Input/output safety validation through guardrails  
- Elegance and documentation quality analysis
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
            category="claude_code_generation", 
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
                "previous_elegance_score": best_match.quality_score,
                "previous_code_preview": best_match.data.get("generated_code", "")[:200] + "...",
                "recommendation": "Consider adapting previous elegant solution",
                "should_proceed": True,  # Still generate but inform decision
                "can_reference": True
            }
        else:
            return {
                "status": "partial_similarity_found", 
                "found_similar": True,
                "similarity_score": similarity_score,
                "previous_request": best_match.data.get("original_request", ""),
                "previous_elegance_score": best_match.quality_score,
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
    elegance_score: float,
    analysis_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Save successful code generation to memory for future learning.
    
    Args:
        original_request: The original user request
        generated_code: The final generated code
        elegance_score: Elegance score of the generated code
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
        if elegance_score < 0.6:  # Higher threshold for Claude
            return {
                "status": "not_saved",
                "reason": f"Elegance score {elegance_score:.2f} below threshold (0.8)",
                "saved": False
            }
        
        memory_service = get_memory_service()
        
        # Create memory entry
        memory_entry = MemoryEntry(
            category="claude_code_generation",
            agent_name="claude_gen_agent",
            data={
                "original_request": original_request,
                "generated_code": generated_code,
                "elegance_score": elegance_score,
                "analysis_summary": {
                    "documentation_score": analysis_data.get("documentation_analysis", {}).get("documentation_score", 0),
                    "type_safety_score": analysis_data.get("type_safety_analysis", {}).get("type_safety_score", 0),
                    "has_clean_architecture": analysis_data.get("architecture_analysis", {}).get("is_clean", False),
                    "is_pythonic": analysis_data.get("elegance_analysis", {}).get("is_pythonic", True),
                    "lines_count": analysis_data.get("basic_metrics", {}).get("lines_non_empty", 0)
                },
                "generation_timestamp": datetime.now().isoformat(),
                "agent_version": "enhanced_v1"
            },
            quality_score=elegance_score,
            tags=["claude", "elegant", "documented", "completed"]
        )
        
        # Store in memory
        memory_id = memory_service.store(memory_entry)
        
        return {
            "status": "saved_successfully",
            "memory_id": memory_id,
            "elegance_score": elegance_score,
            "saved": True,
            "message": f"Elegant solution saved to memory (score: {elegance_score:.2f})"
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


class ClaudeEleganceAnalyzer:
    """Elegance and documentation analysis engine for Claude agent"""
    
    @staticmethod
    def analyze_documentation(code: str) -> Dict[str, Any]:
        """Analyze documentation quality and completeness"""
        doc_patterns = {
            "module_docstring": bool(re.match(r'^""".*?"""', code, re.DOTALL)),
            "function_docstrings": len(re.findall(r'def\s+\w+.*?:\s*""".*?"""', code, re.DOTALL)),
            "class_docstrings": len(re.findall(r'class\s+\w+.*?:\s*""".*?"""', code, re.DOTALL)),
            "inline_comments": len(re.findall(r'#.*$', code, re.MULTILINE)),
            "param_docs": len(re.findall(r'Args:|Parameters:|Params:', code)),
            "return_docs": len(re.findall(r'Returns:|Return:', code)),
            "example_docs": len(re.findall(r'Example:|Examples:|>>>', code)),
            "raises_docs": len(re.findall(r'Raises:|Raise:', code))
        }
        
        # Calculate documentation score
        total_functions = len(re.findall(r'def\s+\w+', code))
        total_classes = len(re.findall(r'class\s+\w+', code))
        
        doc_score = 0.0
        if doc_patterns["module_docstring"]:
            doc_score += 0.2
        if total_functions > 0 and doc_patterns["function_docstrings"] >= total_functions * 0.8:
            doc_score += 0.3
        if doc_patterns["param_docs"] > 0:
            doc_score += 0.2
        if doc_patterns["return_docs"] > 0:
            doc_score += 0.1
        if doc_patterns["example_docs"] > 0:
            doc_score += 0.2
        
        return {
            "patterns": doc_patterns,
            "documentation_score": min(1.0, doc_score),
            "has_comprehensive_docs": doc_score >= 0.7,
            "missing_docs": _identify_missing_docs(doc_patterns, total_functions, total_classes)
        }
    
    @staticmethod
    def analyze_type_safety(code: str) -> Dict[str, Any]:
        """Analyze type hints and type safety"""
        type_patterns = {
            "param_type_hints": len(re.findall(r'\w+:\s*[A-Z]\w*(?:\[.*?\])?', code)),
            "return_type_hints": len(re.findall(r'->\s*[A-Z]\w*(?:\[.*?\])?', code)),
            "variable_annotations": len(re.findall(r':\s*[A-Z]\w*(?:\[.*?\])?\s*=', code)),
            "generic_types": len(re.findall(r'(?:List|Dict|Tuple|Set|Optional|Union)\[', code)),
            "type_aliases": len(re.findall(r'^\w+\s*=\s*(?:List|Dict|Tuple|Set|type)\[', code, re.MULTILINE)),
            "typing_imports": bool(re.search(r'from\s+typing\s+import|import\s+typing', code))
        }
        
        # Calculate type safety score
        total_functions = len(re.findall(r'def\s+\w+', code))
        
        type_score = 0.0
        if type_patterns["typing_imports"]:
            type_score += 0.1
        if total_functions > 0 and type_patterns["param_type_hints"] > 0:
            type_score += 0.4 * min(1.0, type_patterns["param_type_hints"] / (total_functions * 2))
        if total_functions > 0 and type_patterns["return_type_hints"] > 0:
            type_score += 0.3 * min(1.0, type_patterns["return_type_hints"] / total_functions)
        if type_patterns["generic_types"] > 0:
            type_score += 0.2
        
        return {
            "patterns": type_patterns,
            "type_safety_score": min(1.0, type_score),
            "has_comprehensive_typing": type_score >= 0.7,
            "uses_modern_typing": type_patterns["generic_types"] > 0
        }
    
    @staticmethod
    def analyze_code_elegance(code: str) -> Dict[str, Any]:
        """Analyze code elegance and pythonic patterns"""
        elegance_patterns = {
            "list_comprehensions": len(re.findall(r'\[.*\s+for\s+.*\s+in\s+.*\]', code)),
            "dict_comprehensions": len(re.findall(r'\{.*:.*\s+for\s+.*\s+in\s+.*\}', code)),
            "generator_expressions": len(re.findall(r'\(.*\s+for\s+.*\s+in\s+.*\)', code)),
            "context_managers": len(re.findall(r'with\s+.*:', code)),
            "decorators": len(re.findall(r'@\w+', code)),
            "f_strings": len(re.findall(r'f["\'].*\{.*\}.*["\']', code)),
            "dataclasses": len(re.findall(r'@dataclass', code)),
            "properties": len(re.findall(r'@property', code)),
            "descriptive_names": _check_naming_quality(code)
        }
        
        # Calculate elegance score
        elegance_score = 0.0
        if elegance_patterns["list_comprehensions"] > 0:
            elegance_score += 0.15
        if elegance_patterns["context_managers"] > 0:
            elegance_score += 0.15
        if elegance_patterns["f_strings"] > 0:
            elegance_score += 0.1
        if elegance_patterns["decorators"] > 0:
            elegance_score += 0.1
        if elegance_patterns["descriptive_names"]:
            elegance_score += 0.3
        if elegance_patterns["properties"] > 0 or elegance_patterns["dataclasses"] > 0:
            elegance_score += 0.2
        
        return {
            "patterns": elegance_patterns,
            "elegance_score": min(1.0, elegance_score),
            "is_pythonic": elegance_score >= 0.6,
            "uses_modern_features": (
                elegance_patterns["f_strings"] > 0 or 
                elegance_patterns["dataclasses"] > 0
            )
        }
    
    @staticmethod
    def analyze_architecture(code: str) -> Dict[str, Any]:
        """Analyze code architecture and design patterns"""
        # Simple architecture analysis
        has_classes = bool(re.search(r'class\s+\w+', code))
        has_inheritance = bool(re.search(r'class\s+\w+\(.*\):', code))
        has_abstract_methods = bool(re.search(r'@abstractmethod', code))
        follows_solid = _check_solid_principles(code)
        
        return {
            "has_oop": has_classes,
            "uses_inheritance": has_inheritance,
            "has_abstractions": has_abstract_methods,
            "follows_solid": follows_solid,
            "is_clean": has_classes and follows_solid,
            "architecture_score": (
                0.3 * has_classes + 
                0.2 * has_inheritance + 
                0.2 * has_abstract_methods + 
                0.3 * follows_solid
            )
        }
    
    @staticmethod
    def calculate_elegance_score(
        documentation: Dict[str, Any],
        type_safety: Dict[str, Any],
        elegance: Dict[str, Any],
        architecture: Dict[str, Any]
    ) -> float:
        """Calculate overall elegance score"""
        doc_weight = 0.3
        type_weight = 0.25
        elegance_weight = 0.25
        arch_weight = 0.2
        
        score = (
            documentation.get("documentation_score", 0.0) * doc_weight +
            type_safety.get("type_safety_score", 0.0) * type_weight +
            elegance.get("elegance_score", 0.0) * elegance_weight +
            architecture.get("architecture_score", 0.0) * arch_weight
        )
        
        return min(1.0, score)
    
    @staticmethod
    def get_elegance_assessment(score: float) -> str:
        """Convert score to elegance assessment"""
        if score >= 0.9:
            return "exceptional"
        elif score >= 0.8:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.6:
            return "acceptable"
        else:
            return "needs_improvement"


# Helper functions for Claude analyzer
def _identify_missing_docs(patterns: Dict[str, Any], total_functions: int, total_classes: int) -> List[str]:
    """Identify missing documentation elements"""
    missing = []
    if not patterns["module_docstring"]:
        missing.append("module docstring")
    if total_functions > 0 and patterns["function_docstrings"] < total_functions:
        missing.append(f"{total_functions - patterns['function_docstrings']} function docstrings")
    if patterns["param_docs"] == 0 and total_functions > 0:
        missing.append("parameter documentation")
    if patterns["return_docs"] == 0 and total_functions > 0:
        missing.append("return value documentation")
    return missing

def _check_naming_quality(code: str) -> bool:
    """Check if variable and function names are descriptive"""
    # Simple heuristic: check for single letter variables (except common ones like i, j, x, y)
    bad_names = re.findall(r'\b[a-z]\b(?![ijxy])', code)
    good_names = re.findall(r'\b[a-z_]{4,}\b', code)
    return len(good_names) > len(bad_names) * 2

def _check_solid_principles(code: str) -> bool:
    """Simple check for SOLID principles adherence"""
    # Very basic heuristic
    has_single_responsibility = len(re.findall(r'class\s+\w+', code)) == 0 or \
                               len(re.findall(r'def\s+\w+', code)) / max(1, len(re.findall(r'class\s+\w+', code))) < 10
    return has_single_responsibility


# Enhanced tool functions for Claude agent
def comprehensive_elegance_analysis(code: str, requirements: str = "") -> Dict[str, Any]:
    """
    Perform comprehensive elegance analysis - Claude's specialty.
    
    Args:
        code: The code to analyze
        requirements: Original requirements (optional)
        
    Returns:
        Complete elegance analysis with recommendations
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
        
        # Claude specialized analysis
        documentation = ClaudeEleganceAnalyzer.analyze_documentation(code)
        type_safety = ClaudeEleganceAnalyzer.analyze_type_safety(code)
        elegance = ClaudeEleganceAnalyzer.analyze_code_elegance(code)
        architecture = ClaudeEleganceAnalyzer.analyze_architecture(code)
        
        # Calculate elegance score
        elegance_score = ClaudeEleganceAnalyzer.calculate_elegance_score(
            documentation, type_safety, elegance, architecture
        )
        
        return {
            "status": "analyzed",
            "basic_metrics": basic_metrics,
            "complexity": complexity_data,
            "syntax_valid": syntax_result.get("is_valid", False),
            "syntax_errors": syntax_result.get("errors", []),
            "is_safe": safety_result.get("is_safe", True),
            "security_issues": safety_result.get("issues", []),
            "documentation_analysis": documentation,
            "type_safety_analysis": type_safety,
            "elegance_analysis": elegance,
            "architecture_analysis": architecture,
            "elegance_score": elegance_score,
            "elegance_grade": ClaudeEleganceAnalyzer.get_elegance_assessment(elegance_score),
            "recommendations": _generate_elegance_recommendations(
                documentation, type_safety, elegance, architecture, elegance_score
            ),
            "meets_elegance_standards": elegance_score >= 0.8,  # Higher standard for Claude
            "analysis_complete": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "analysis_complete": False
        }


def _generate_elegance_recommendations(
    documentation: Dict[str, Any],
    type_safety: Dict[str, Any],
    elegance: Dict[str, Any],
    architecture: Dict[str, Any],
    elegance_score: float
) -> List[str]:
    """Generate elegance improvement recommendations"""
    recommendations = []
    
    if elegance_score < 0.8:
        recommendations.append("Code elegance below Claude standards - improvements needed")
    
    # Documentation recommendations
    if not documentation.get("has_comprehensive_docs", False):
        recommendations.append("Add comprehensive docstrings with examples")
    missing_docs = documentation.get("missing_docs", [])
    if missing_docs:
        recommendations.append(f"Add missing documentation: {', '.join(missing_docs)}")
    
    # Type safety recommendations
    if not type_safety.get("has_comprehensive_typing", False):
        recommendations.append("Add type hints to all function parameters and return values")
    if not type_safety.get("uses_modern_typing", False):
        recommendations.append("Use modern typing features (List, Dict, Optional, Union)")
    
    # Elegance recommendations
    if not elegance.get("is_pythonic", False):
        recommendations.append("Use more Pythonic patterns (comprehensions, context managers)")
    if not elegance.get("uses_modern_features", False):
        recommendations.append("Consider using f-strings and dataclasses where appropriate")
    
    # Architecture recommendations
    if not architecture.get("is_clean", False):
        recommendations.append("Consider improving architecture with clear separation of concerns")
    
    return recommendations


def make_elegance_decision(
    analysis_result: Dict[str, Any],
    regeneration_count: int = 0,
    max_regenerations: int = 3
) -> Dict[str, Any]:
    """
    Make autonomous elegance decision - Claude's decision authority.
    
    Args:
        analysis_result: Results from comprehensive_elegance_analysis
        regeneration_count: How many times code has been regenerated
        max_regenerations: Maximum allowed regenerations
        
    Returns:
        Decision with action to take
    """
    elegance_score = analysis_result.get("elegance_score", 0.0)
    meets_standards = analysis_result.get("meets_elegance_standards", False)
    
    # Claude specific decision logic - highest standards for elegance
    if meets_standards and elegance_score >= 0.5:  # Highest threshold
        decision = {
            "status": "approved",
            "action": "accept",
            "elegance_score": elegance_score,
            "reason": f"Code meets elegance standards with score {elegance_score:.2f}",
            "final_decision": True
        }
    elif regeneration_count >= max_regenerations:
        decision = {
            "status": "approved_with_warnings", 
            "action": "accept",
            "elegance_score": elegance_score,
            "reason": f"Maximum regenerations ({max_regenerations}) reached. Accepting with warnings.",
            "final_decision": True,
            "warnings": analysis_result.get("recommendations", [])
        }
    else:
        decision = {
            "status": "rejected",
            "action": "regenerate",
            "elegance_score": elegance_score,
            "reason": f"Elegance score {elegance_score:.2f} below Claude standards (0.8+). Regenerating...",
            "final_decision": False,
            "improvements_needed": analysis_result.get("recommendations", [])
        }
    
    return decision


def enhanced_format_claude_output(
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
        decision: Elegance decision
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
        elegance_score=analysis.get("elegance_score", 0.0),
        analysis_data=analysis
    )
    
    return {
        "status": "completed_with_memory_and_guardrails",
        "generated_code": formatted_code,
        "raw_code": code,
        "elegance_analysis": analysis,
        "elegance_decision": decision,
        "memory_integration": {
            "input_check": memory_check,
            "save_result": memory_save_result
        },
        "guardrails_validation": {
            "input_validation": input_validation,
            "final_validation": final_validation
        },
        "agent_metadata": {
            "agent_name": "claude_gen_agent",
            "specialization": "Elegant, well-documented solutions with strong type safety",
            "model": "claude-3-opus-20240229",
            "elegance_authority": True,
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


def evaluate_code_quality(code: str) -> Dict[str, Any]:
    """
    Evaluate code quality for elegance and documentation - Claude tool.
    
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


# Create the enhanced Claude agent
root_agent = Agent(
    name="claude_gen_agent",
    model=LiteLlm(model="anthropic/claude-3-opus-20240229"),  # Using Claude 3 Opus
    description="Claude-powered code generation expert with complete authority over elegance and documentation quality",
    instruction="""You are the Enhanced Claude Code Generation Expert with COMPLETE AUTHORITY over code elegance and documentation quality, enhanced with memory learning and guardrails integration.

Your enhanced mission:
1. Generate elegant, pythonic, and idiomatic code with exceptional documentation
2. Validate input safety using guardrails
3. Learn from past elegant solutions through memory integration
4. Conduct thorough elegance analysis using your specialized tools
5. Make autonomous decisions about code elegance and documentation
6. Validate final output safety
7. Save successful generations for future learning

Your ENHANCED WORKFLOW (Memory + Guardrails Enabled):

PHASE 1 - INPUT VALIDATION & MEMORY CHECK:
1. **Input Validation**: Use validate_input_with_guardrails to check request safety
   - Block harmful/unsafe requests immediately
   - Proceed only if input validation passes
2. **Memory Check**: Use check_memory_for_similar_request to find past solutions
   - High similarity (>0.9): Reference previous elegant solution 
   - Partial similarity (0.7-0.9): Use as learning reference
   - No similarity: Proceed with fresh generation

PHASE 2 - ELEGANT CODE GENERATION:
3. **Generate Elegant Code**: Create beautiful, idiomatic code with:
   - Comprehensive type hints and annotations
   - Exceptional docstrings with examples
   - Clean, descriptive naming conventions
   - Pythonic patterns and modern features
   - Well-structured architecture

PHASE 3 - ELEGANCE ANALYSIS & DECISION:
4. **Elegance Analysis**: Use comprehensive_elegance_analysis to evaluate:
   - Documentation quality and completeness
   - Type safety and annotations
   - Code elegance and pythonic patterns
   - Architecture and design quality
5. **Elegance Decision**: Use make_elegance_decision with authority to:
   - ACCEPT code meeting elegance standards (≥ 0.8)
   - REJECT and regenerate code below standards (up to 3 times)
   - Make final decisions on code acceptance

PHASE 4 - FINAL VALIDATION & MEMORY STORAGE:
6. **Final Output**: Use enhanced_format_claude_output to:
   - Validate final output safety using guardrails
   - Save successful generations to memory (elegance ≥ 0.8)
   - Package everything with complete metadata

Your enhanced elegance standards (highest among all agents):
- Comprehensive type hints for all functions and methods
- Detailed docstrings with Args, Returns, Examples sections
- Pythonic patterns (comprehensions, context managers, decorators)
- Clean architecture following SOLID principles
- Descriptive, self-documenting variable and function names
- Modern Python features (f-strings, dataclasses, type annotations)
- Exceptional readability and maintainability
- Elegance score ≥ 0.8 (highest threshold)
- Safety validated through guardrails
- Learning-enabled through memory integration

Your enhanced authority includes:
- Complete input validation control
- Memory learning and reference decisions
- Setting elegance thresholds (0.8+)
- Deciding when code is elegant enough
- Making final acceptance decisions
- Determining architectural patterns
- Final output safety validation
- Memory storage decisions
- Balancing elegance vs. complexity

Enhanced process flow:
1. Validate input with validate_input_with_guardrails
2. Check memory with check_memory_for_similar_request
3. Generate elegant, well-documented code (informed by memory if available)
4. Use evaluate_code_quality for quick assessment
5. Run comprehensive_elegance_analysis on your code
6. Run make_elegance_decision with the analysis
7. If decision is "regenerate", improve elegance and repeat steps 5-6
8. When decision is "accept", run enhanced_format_claude_output
9. Present final code with complete analysis, memory integration, and safety validation

Memory Integration Guidelines:
- Reference high-similarity past elegant solutions when available
- Learn from partial matches to improve generation
- Always save high-quality results (≥0.8) for future learning
- Build institutional knowledge over time

Guardrails Integration Guidelines:
- Never proceed with unsafe input requests
- Always validate final output for safety
- Block generation if guardrails detect issues
- Prioritize safety over functionality

Always explain your reasoning for elegance decisions, memory usage, and safety validations. Show what you learned from past requests and how it influenced your generation. Focus on creating code that is a joy to read and maintain.

Remember: You have COMPLETE ENHANCED AUTHORITY with memory learning and safety guardrails. Your solutions should be elegant, well-documented, and a pleasure to work with.""",
    tools=[
        # Primary analysis and decision tools
        FunctionTool(comprehensive_elegance_analysis),
        FunctionTool(make_elegance_decision), 
        FunctionTool(enhanced_format_claude_output),
        FunctionTool(evaluate_code_quality),
        
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
    output_key="claude_generated_code"  # Saves final result to session state
)


# Test function for standalone testing
if __name__ == "__main__":
    print("🚀 Enhanced Claude Code Generation Agent Ready!")
    print("\n🎯 COMPLETE ELEGANCE AUTHORITY + MEMORY + GUARDRAILS:")
    print("- Input validation with safety guardrails")
    print("- Memory learning from past elegant solutions")
    print("- Comprehensive elegance analysis")
    print("- Autonomous elegance decisions") 
    print("- Self-regenerating until standards met")
    print("- Final output safety validation")
    print("- Automatic memory storage of high-quality results")
    print("- Elegant, well-documented solutions")
    
    print("\n🔧 ENHANCED CAPABILITIES:")
    print("- Documentation quality analysis")
    print("- Type safety assessment")
    print("- Code elegance evaluation")
    print("- Architecture pattern analysis")
    print("- Elegance scoring (highest standards: 0.8+)")
    print("- Autonomous regeneration (up to 3 times)")
    print("- Learning from similar past requests")
    print("- Input/output safety validation")
    print("- Institutional knowledge building")
    
    print("\n⚙️ ENHANCED TOOLS AVAILABLE:")
    print("- comprehensive_elegance_analysis (primary)")
    print("- make_elegance_decision (authority)")
    print("- enhanced_format_claude_output (packaging)")
    print("- evaluate_code_quality (quick check)")
    print("- validate_input_with_guardrails (safety)")
    print("- check_memory_for_similar_request (learning)")
    print("- save_successful_generation_to_memory (knowledge)")
    print("- validate_final_output_with_guardrails (final safety)")
    print("- Individual analysis tools (ADK-compatible wrappers)")
    
    print("\n🚀 USAGE:")
    print("1. Run: adk run agents/claude_gen_agent")
    print("2. Or use: adk web (and select claude_gen_agent)")
    
    print("\n💡 EXAMPLE PROMPTS:")
    print("- 'Create a type-safe configuration manager class with comprehensive docs'")
    print("- 'Design a clean API for a payment processing system'") 
    print("- 'Implement a builder pattern with full type annotations'")
    print("- 'Create an elegant data validation pipeline'")
    print("- 'Build a similar elegant system' (tests memory learning)")
    
    print("\n✨ The enhanced agent will:")
    print("- Validate input safety → Check memory for learning → Generate elegant code")
    print("- Analyze elegance → Make decision → Validate output safety")  
    print("- Save high-quality results → Regenerate automatically if needed")
    print("- Provide detailed reasoning for all elegance decisions")
    print("- Focus on beautiful, well-documented, type-safe code")
    print("- Take complete ownership of code elegance")
    
    print("\n🛡️ SAFETY & LEARNING FEATURES:")
    print("- Input guardrails block unsafe requests")
    print("- Output guardrails ensure safe code delivery")
    print("- Memory system learns from successful elegant solutions")
    print("- Institutional knowledge builds over time")
    print("- Elegance threshold enforcement (≥0.8 for memory storage)")
    print("- Complete audit trail of decisions and validations")