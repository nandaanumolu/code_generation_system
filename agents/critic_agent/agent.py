"""
Enhanced Critic Agent - Production Ready with Memory & Guardrails
Specializes in comprehensive code review and analysis from all generation agents

Features:
- Comprehensive multi-dimensional code review
- Reviews code from Gemini, GPT-4, and Claude agents with understanding of their specialties
- Memory learning from past review patterns and successful critiques
- Input/output safety validation through guardrails  
- Structured feedback with severity classification
- Institutional knowledge building for better reviews over time

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

def validate_review_input_with_guardrails(code_to_review: str, review_request: str = "") -> Dict[str, Any]:
    """
    Validate code review input using guardrails.
    
    Args:
        code_to_review: The code that needs to be reviewed
        review_request: Optional specific review request
        
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
        # Combine code and request for validation
        combined_input = f"Review request: {review_request}\nCode to review:\n{code_to_review}"
        validation_result = validate_input_request(combined_input)
        
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
        return {
            "status": "error",
            "is_safe": True,
            "is_valid": True,
            "confidence": 0.5,
            "issues": [f"Guardrail error: {str(e)}"],
            "blocked": False,
            "timestamp": datetime.now().isoformat()
        }


def check_memory_for_similar_reviews(code_to_review: str, source_agent: str = "") -> Dict[str, Any]:
    """
    Check memory for similar past code reviews to enable learning.
    
    Args:
        code_to_review: Current code to review
        source_agent: Which agent generated the code (gemini, gpt4, claude)
        
    Returns:
        Memory search results with similar reviews
    """
    if not MEMORY_AVAILABLE:
        return {
            "status": "memory_not_available",
            "found_similar": False,
            "message": "Memory service not available - proceeding with fresh review",
            "should_proceed": True
        }
    
    try:
        memory_service = get_memory_service()
        
        # Search for similar past reviews
        similar_memories = memory_service.search_similar(
            request=code_to_review,
            category="critic_code_reviews", 
            threshold=0.6  # Lower threshold for code similarity
        )
        
        if not similar_memories:
            return {
                "status": "no_matches",
                "found_similar": False,
                "message": "No similar reviews found in memory",
                "should_proceed": True
            }
        
        # Get the best match
        best_match = similar_memories[0]
        similarity_score = best_match.data.get("similarity_score", 0)
        
        # High similarity suggests learning from past review
        if similarity_score > 0.8:
            return {
                "status": "high_similarity_found",
                "found_similar": True,
                "similarity_score": similarity_score,
                "previous_code": best_match.data.get("reviewed_code", "")[:200] + "...",
                "previous_review_quality": best_match.quality_score,
                "previous_issues_found": best_match.data.get("issues_count", 0),
                "previous_severity_breakdown": best_match.data.get("severity_breakdown", {}),
                "recommendation": "Learn from previous similar review patterns",
                "should_proceed": True,
                "can_reference": True
            }
        else:
            return {
                "status": "partial_similarity_found", 
                "found_similar": True,
                "similarity_score": similarity_score,
                "previous_review_quality": best_match.quality_score,
                "recommendation": "Some related review experience found - use as reference",
                "should_proceed": True,
                "can_reference": False
            }
            
    except Exception as e:
        return {
            "status": "memory_error",
            "found_similar": False,
            "error": str(e),
            "should_proceed": True,
            "message": "Memory check failed, proceeding with fresh review"
        }


def save_successful_review_to_memory(
    reviewed_code: str,
    review_analysis: Dict[str, Any],
    source_agent: str,
    review_quality_score: float
) -> Dict[str, Any]:
    """
    Save successful code review to memory for future learning.
    
    Args:
        reviewed_code: The code that was reviewed
        review_analysis: Complete review analysis results
        source_agent: Which agent generated the code
        review_quality_score: Quality score of the review process
        
    Returns:
        Memory save confirmation
    """
    if not MEMORY_AVAILABLE:
        return {
            "status": "memory_not_available",
            "reason": "Memory service not available",
            "saved": False,
            "message": "Review completed successfully but not saved to memory"
        }
    
    try:
        # Only save high-quality reviews
        if review_quality_score < 0.7:
            return {
                "status": "not_saved",
                "reason": f"Review quality score {review_quality_score:.2f} below threshold (0.7)",
                "saved": False
            }
        
        memory_service = get_memory_service()
        
        # Create memory entry
        memory_entry = MemoryEntry(
            category="critic_code_reviews",
            agent_name="critic_agent",
            data={
                "reviewed_code": reviewed_code,
                "source_agent": source_agent,
                "review_quality_score": review_quality_score,
                "issues_count": review_analysis.get("total_issues", 0),
                "severity_breakdown": review_analysis.get("severity_counts", {}),
                "review_summary": {
                    "correctness_score": review_analysis.get("correctness_analysis", {}).get("correctness_score", 0),
                    "security_score": review_analysis.get("security_analysis", {}).get("security_score", 0),
                    "performance_score": review_analysis.get("performance_analysis", {}).get("performance_score", 0),
                    "maintainability_score": review_analysis.get("maintainability_analysis", {}).get("maintainability_score", 0),
                    "overall_code_quality": review_analysis.get("overall_quality_score", 0)
                },
                "review_timestamp": datetime.now().isoformat(),
                "agent_version": "enhanced_v1"
            },
            quality_score=review_quality_score,
            tags=["critic", "comprehensive_review", "completed", source_agent]
        )
        
        # Store in memory
        memory_id = memory_service.store(memory_entry)
        
        return {
            "status": "saved_successfully",
            "memory_id": memory_id,
            "review_quality_score": review_quality_score,
            "saved": True,
            "message": f"Comprehensive review saved to memory (score: {review_quality_score:.2f})"
        }
        
    except Exception as e:
        return {
            "status": "save_error",
            "error": str(e),
            "saved": False,
            "message": "Failed to save to memory, but review completed successfully"
        }


def validate_review_output_with_guardrails(
    review_feedback: str,
    original_code: str
) -> Dict[str, Any]:
    """
    Final validation of review feedback using output guardrails.
    
    Args:
        review_feedback: Generated review feedback
        original_code: Original code that was reviewed
        
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
            generated_code=review_feedback,
            original_request=f"Code review of: {original_code[:200]}..."
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
        return {
            "status": "validation_error",
            "is_safe": True,
            "is_appropriate": True,
            "confidence": 0.5,
            "issues": [f"Validation error: {str(e)}"],
            "ready_for_delivery": True,
            "validation_timestamp": datetime.now().isoformat()
        }


class CriticAnalysisEngine:
    """Comprehensive code review and analysis engine for Critic agent"""
    
    @staticmethod
    def analyze_correctness(code: str) -> Dict[str, Any]:
        """Analyze code correctness and logic issues"""
        issues = []
        
        # Basic syntax check
        syntax_result = validate_python_syntax(code)
        if not syntax_result.get("is_valid", True):
            issues.extend([{"type": "syntax_error", "severity": "critical", "description": err} 
                          for err in syntax_result.get("errors", [])])
        
        # Logic issue patterns
        logic_patterns = {
            "unreachable_code": len(re.findall(r'return.*\n.*\S', code)),
            "infinite_loops": len(re.findall(r'while\s+True.*(?!\n.*break)', code, re.DOTALL)),
            "unused_variables": len(re.findall(r'^\s*(\w+)\s*=.*(?!\n.*\1)', code, re.MULTILINE)),
            "missing_return": len(re.findall(r'def\s+\w+\([^)]*\):(?:(?!\n\s*def|\n\s*class|\nclass|\ndef).)*?(?!\n.*return)', code, re.DOTALL))
        }
        
        for pattern, count in logic_patterns.items():
            if count > 0:
                issues.append({
                    "type": pattern,
                    "severity": "major",
                    "count": count,
                    "description": f"Found {count} instances of {pattern.replace('_', ' ')}"
                })
        
        correctness_score = max(0.0, 1.0 - (len(issues) * 0.1))
        
        return {
            "correctness_score": correctness_score,
            "syntax_valid": syntax_result.get("is_valid", True),
            "logic_issues": issues,
            "total_correctness_issues": len(issues),
            "patterns_analyzed": logic_patterns
        }
    
    @staticmethod
    def analyze_security(code: str) -> Dict[str, Any]:
        """Analyze security vulnerabilities and risks"""
        security_issues = []
        
        # Security vulnerability patterns
        security_patterns = {
            "sql_injection": re.findall(r'execute\s*\(\s*[\'"].*%.*[\'"]|execute\s*\(\s*f[\'"]', code),
            "eval_usage": re.findall(r'\beval\s*\(|\bexec\s*\(', code),
            "shell_injection": re.findall(r'os\.system\s*\(|subprocess\.[^(]*\([^)]*shell\s*=\s*True', code),
            "hardcoded_secrets": re.findall(r'password\s*=\s*[\'"][^\'"]+[\'"]|api_key\s*=\s*[\'"][^\'"]+[\'"]', code, re.IGNORECASE),
            "unsafe_pickle": re.findall(r'pickle\.loads?\s*\(|cPickle\.loads?\s*\(', code),
            "unsafe_yaml": re.findall(r'yaml\.load\s*\([^)]*(?!Loader)', code)
        }
        
        for vuln_type, matches in security_patterns.items():
            if matches:
                security_issues.append({
                    "type": vuln_type,
                    "severity": "critical" if vuln_type in ["sql_injection", "eval_usage", "shell_injection"] else "major",
                    "instances": len(matches),
                    "examples": matches[:3],  # First 3 examples
                    "description": f"Potential {vuln_type.replace('_', ' ')} vulnerability"
                })
        
        # Security best practices check
        has_input_validation = bool(re.search(r'\bvalidate\w*\(|\bcheck\w*\(|isinstance\s*\(', code))
        has_error_handling = bool(re.search(r'\btry\s*:|except\s*\w*:', code))
        
        security_score = max(0.0, 1.0 - (len(security_issues) * 0.15))
        if has_input_validation:
            security_score += 0.1
        if has_error_handling:
            security_score += 0.1
        security_score = min(1.0, security_score)
        
        return {
            "security_score": security_score,
            "security_issues": security_issues,
            "total_security_issues": len(security_issues),
            "has_input_validation": has_input_validation,
            "has_error_handling": has_error_handling,
            "patterns_checked": list(security_patterns.keys())
        }
    
    @staticmethod
    def analyze_performance(code: str) -> Dict[str, Any]:
        """Analyze performance issues and optimization opportunities"""
        performance_issues = []
        
        # Performance anti-patterns
        perf_patterns = {
            "nested_loops": len(re.findall(r'for\s+\w+.*:\s*.*for\s+\w+', code, re.DOTALL)),
            "inefficient_string_concat": len(re.findall(r'\w+\s*\+=\s*[\'"]|\w+\s*=\s*\w+\s*\+\s*[\'"]', code)),
            "global_variables": len(re.findall(r'^\s*global\s+\w+', code, re.MULTILINE)),
            "repeated_calculations": len(re.findall(r'(\w+\([^)]*\)).*\1', code)),
            "large_iterations": len(re.findall(r'for\s+\w+\s+in\s+range\s*\(\s*\d{4,}', code))
        }
        
        for pattern, count in perf_patterns.items():
            if count > 0:
                severity = "major" if pattern in ["nested_loops", "large_iterations"] else "minor"
                performance_issues.append({
                    "type": pattern,
                    "severity": severity,
                    "count": count,
                    "description": f"Found {count} instances of {pattern.replace('_', ' ')}"
                })
        
        # Complexity analysis
        complexity_data = estimate_complexity(code)
        complexity_score = complexity_data.get("complexity", 0)
        
        if complexity_score > 15:
            performance_issues.append({
                "type": "high_complexity",
                "severity": "major",
                "value": complexity_score,
                "description": f"Cyclomatic complexity ({complexity_score}) is high"
            })
        
        performance_score = max(0.0, 1.0 - (len(performance_issues) * 0.1))
        
        return {
            "performance_score": performance_score,
            "performance_issues": performance_issues,
            "total_performance_issues": len(performance_issues),
            "complexity_score": complexity_score,
            "patterns_analyzed": perf_patterns
        }
    
    @staticmethod
    def analyze_maintainability(code: str) -> Dict[str, Any]:
        """Analyze code maintainability and best practices"""
        maintainability_issues = []
        
        # Get basic metrics
        metrics = analyze_code(code)
        
        # Maintainability checks
        has_docstrings = metrics.get("has_docstrings", False)
        has_functions = metrics.get("has_functions", False)
        has_classes = metrics.get("has_classes", False)
        line_count = metrics.get("lines_non_empty", 0)
        
        # Code organization issues
        if not has_docstrings and (has_functions or has_classes):
            maintainability_issues.append({
                "type": "missing_documentation",
                "severity": "major",
                "description": "Functions/classes lack docstrings"
            })
        
        if line_count > 50 and not (has_functions or has_classes):
            maintainability_issues.append({
                "type": "monolithic_code",
                "severity": "major",
                "description": "Large code block should be organized into functions/classes"
            })
        
        # Naming convention issues
        bad_names = re.findall(r'\b[a-z][a-z0-9]*[A-Z]|\b[A-Z][a-z]*_|\bdef\s+[A-Z]', code)
        if bad_names:
            maintainability_issues.append({
                "type": "naming_convention",
                "severity": "minor",
                "count": len(bad_names),
                "description": f"Found {len(bad_names)} naming convention violations"
            })
        
        # Magic numbers
        magic_numbers = re.findall(r'\b\d{2,}\b', code)
        if len(magic_numbers) > 3:
            maintainability_issues.append({
                "type": "magic_numbers",
                "severity": "minor",
                "count": len(magic_numbers),
                "description": "Consider using named constants for numeric literals"
            })
        
        maintainability_score = 1.0
        if has_docstrings:
            maintainability_score += 0.2
        if has_functions or has_classes:
            maintainability_score += 0.2
        maintainability_score -= len(maintainability_issues) * 0.1
        maintainability_score = max(0.0, min(1.0, maintainability_score))
        
        return {
            "maintainability_score": maintainability_score,
            "maintainability_issues": maintainability_issues,
            "total_maintainability_issues": len(maintainability_issues),
            "has_documentation": has_docstrings,
            "is_well_organized": has_functions or has_classes,
            "code_metrics": metrics
        }
    
    @staticmethod
    def calculate_overall_quality_score(
        correctness: Dict[str, Any],
        security: Dict[str, Any],
        performance: Dict[str, Any],
        maintainability: Dict[str, Any]
    ) -> float:
        """Calculate overall code quality score"""
        # Weighted scoring
        weights = {
            "correctness": 0.35,
            "security": 0.30,
            "performance": 0.20,
            "maintainability": 0.15
        }
        
        score = (
            correctness.get("correctness_score", 0.0) * weights["correctness"] +
            security.get("security_score", 0.0) * weights["security"] +
            performance.get("performance_score", 0.0) * weights["performance"] +
            maintainability.get("maintainability_score", 0.0) * weights["maintainability"]
        )
        
        return min(1.0, score)
    
    @staticmethod
    def get_quality_grade(score: float) -> str:
        """Convert score to quality grade"""
        if score >= 0.9:
            return "A (Excellent)"
        elif score >= 0.8:
            return "B (Very Good)"
        elif score >= 0.7:
            return "C (Good)"
        elif score >= 0.6:
            return "D (Acceptable)"
        else:
            return "F (Needs Major Improvement)"


# Enhanced tool functions for Critic agent
def comprehensive_code_review(
    code: str, 
    source_agent: str = "unknown",
    review_focus: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Perform comprehensive multi-dimensional code review.
    
    Args:
        code: The code to review
        source_agent: Which agent generated the code (gemini, gpt4, claude)
        review_focus: Type of review (comprehensive, security, performance, etc.)
        
    Returns:
        Complete review analysis with structured feedback
    """
    try:
        # Add line numbers for precise feedback
        numbered_code = add_line_numbers(code)
        
        # Multi-dimensional analysis
        correctness = CriticAnalysisEngine.analyze_correctness(code)
        security = CriticAnalysisEngine.analyze_security(code)
        performance = CriticAnalysisEngine.analyze_performance(code)
        maintainability = CriticAnalysisEngine.analyze_maintainability(code)
        
        # Calculate overall quality
        overall_quality = CriticAnalysisEngine.calculate_overall_quality_score(
            correctness, security, performance, maintainability
        )
        
        # Aggregate all issues
        all_issues = []
        all_issues.extend(correctness.get("logic_issues", []))
        all_issues.extend(security.get("security_issues", []))
        all_issues.extend(performance.get("performance_issues", []))
        all_issues.extend(maintainability.get("maintainability_issues", []))
        
        # Count by severity
        severity_counts = {
            "critical": len([i for i in all_issues if i.get("severity") == "critical"]),
            "major": len([i for i in all_issues if i.get("severity") == "major"]),
            "minor": len([i for i in all_issues if i.get("severity") == "minor"])
        }
        
        return {
            "status": "reviewed",
            "source_agent": source_agent,
            "numbered_code": numbered_code.get("numbered_code", code),
            "correctness_analysis": correctness,
            "security_analysis": security,
            "performance_analysis": performance,
            "maintainability_analysis": maintainability,
            "overall_quality_score": overall_quality,
            "quality_grade": CriticAnalysisEngine.get_quality_grade(overall_quality),
            "all_issues": all_issues,
            "total_issues": len(all_issues),
            "severity_counts": severity_counts,
            "has_critical_issues": severity_counts["critical"] > 0,
            "has_blocking_issues": severity_counts["critical"] > 0 or severity_counts["major"] > 3,
            "review_complete": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "review_complete": False
        }


def generate_structured_feedback(
    review_analysis: Dict[str, Any],
    target_audience: str = "developer"
) -> Dict[str, Any]:
    """
    Generate structured, actionable feedback from review analysis.
    
    Args:
        review_analysis: Results from comprehensive_code_review
        target_audience: Who the feedback is for (developer, refactor_agent, etc.)
        
    Returns:
        Structured feedback with priorities and recommendations
    """
    try:
        all_issues = review_analysis.get("all_issues", [])
        severity_counts = review_analysis.get("severity_counts", {})
        overall_quality = review_analysis.get("overall_quality_score", 0.0)
        
        # Prioritize feedback
        critical_issues = [i for i in all_issues if i.get("severity") == "critical"]
        major_issues = [i for i in all_issues if i.get("severity") == "major"]
        minor_issues = [i for i in all_issues if i.get("severity") == "minor"]
        
        # Generate recommendations
        recommendations = []
        
        if critical_issues:
            recommendations.append({
                "priority": "IMMEDIATE",
                "category": "Critical Issues",
                "action": "Fix all critical issues before deployment",
                "issues": critical_issues[:5]  # Top 5 critical
            })
        
        if major_issues:
            recommendations.append({
                "priority": "HIGH",
                "category": "Major Issues", 
                "action": "Address major issues to improve code quality",
                "issues": major_issues[:5]  # Top 5 major
            })
        
        if minor_issues and overall_quality > 0.7:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "Minor Improvements",
                "action": "Consider addressing for better maintainability",
                "issues": minor_issues[:3]  # Top 3 minor
            })
        
        # Overall assessment
        if overall_quality >= 0.8:
            overall_assessment = "Code quality is good with minor improvements needed."
        elif overall_quality >= 0.6:
            overall_assessment = "Code quality is acceptable but requires attention to identified issues."
        else:
            overall_assessment = "Code quality needs significant improvement before deployment."
        
        return {
            "status": "feedback_generated",
            "overall_assessment": overall_assessment,
            "quality_score": overall_quality,
            "quality_grade": review_analysis.get("quality_grade", "Unknown"),
            "total_issues_found": len(all_issues),
            "severity_breakdown": severity_counts,
            "structured_recommendations": recommendations,
            "next_actions": _generate_next_actions(severity_counts, overall_quality),
            "review_summary": _generate_review_summary(review_analysis),
            "feedback_complete": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "feedback_complete": False
        }


def _generate_next_actions(severity_counts: Dict[str, int], quality_score: float) -> List[str]:
    """Generate recommended next actions based on review results"""
    actions = []
    
    if severity_counts.get("critical", 0) > 0:
        actions.append("🚨 BLOCK deployment - critical issues must be fixed")
        actions.append("🔧 Refactor code to address security/correctness issues")
    elif severity_counts.get("major", 0) > 3:
        actions.append("⚠️  Significant refactoring recommended")
        actions.append("🔍 Focus on major issues before proceeding")
    elif quality_score >= 0.8:
        actions.append("✅ Code is ready with minor improvements")
        actions.append("📝 Consider addressing minor issues for polish")
    else:
        actions.append("🔄 Moderate refactoring needed")
        actions.append("📊 Re-review after improvements")
    
    return actions


def _generate_review_summary(review_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Generate executive summary of the review"""
    return {
        "correctness": f"Score: {review_analysis.get('correctness_analysis', {}).get('correctness_score', 0):.2f}",
        "security": f"Score: {review_analysis.get('security_analysis', {}).get('security_score', 0):.2f}",
        "performance": f"Score: {review_analysis.get('performance_analysis', {}).get('performance_score', 0):.2f}",
        "maintainability": f"Score: {review_analysis.get('maintainability_analysis', {}).get('maintainability_score', 0):.2f}",
        "recommendation": "APPROVE" if review_analysis.get("overall_quality_score", 0) >= 0.7 else "NEEDS_WORK"
    }


def make_review_decision(
    review_analysis: Dict[str, Any],
    quality_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Make autonomous decision about code quality and next steps.
    
    Args:
        review_analysis: Results from comprehensive_code_review
        quality_threshold: Minimum quality score for approval
        
    Returns:
        Decision with recommended actions
    """
    overall_quality = review_analysis.get("overall_quality_score", 0.0)
    has_critical_issues = review_analysis.get("has_critical_issues", False)
    has_blocking_issues = review_analysis.get("has_blocking_issues", False)
    
    if has_critical_issues:
        decision = {
            "status": "rejected",
            "action": "requires_immediate_refactoring",
            "quality_score": overall_quality,
            "reason": "Critical issues found - code must be refactored before deployment",
            "blocking": True,
            "refactor_priority": "CRITICAL"
        }
    elif has_blocking_issues:
        decision = {
            "status": "rejected",
            "action": "requires_major_refactoring",
            "quality_score": overall_quality,
            "reason": "Multiple major issues found - significant refactoring needed",
            "blocking": True,
            "refactor_priority": "HIGH"
        }
    elif overall_quality >= quality_threshold:
        decision = {
            "status": "approved",
            "action": "ready_for_delivery",
            "quality_score": overall_quality,
            "reason": f"Code meets quality standards (score: {overall_quality:.2f})",
            "blocking": False,
            "refactor_priority": "OPTIONAL"
        }
    else:
        decision = {
            "status": "conditional_approval",
            "action": "minor_refactoring_recommended",
            "quality_score": overall_quality,
            "reason": f"Code is functional but could benefit from improvements (score: {overall_quality:.2f})",
            "blocking": False,
            "refactor_priority": "MEDIUM"
        }
    
    return decision


def enhanced_format_critic_output(
    review_analysis: Dict[str, Any],
    structured_feedback: Dict[str, Any],
    review_decision: Dict[str, Any],
    original_code: str,
    source_agent: str,
    memory_check: Optional[Dict[str, Any]] = None,
    input_validation: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhanced format function with memory saving and final validation.
    Now with robust type checking to handle LLM mistakes.
    """
    
    # 🚨 CRITICAL FIX: Add type checking for all dict parameters
    # Convert strings to dicts if needed
    if isinstance(review_analysis, str):
        print(f"⚠️ WARNING: review_analysis is string, attempting to parse or create default")
        # Try to create a minimal valid structure
        review_analysis = {
            "review_complete": False,
            "total_issues": 0,
            "overall_quality_score": 0.5,
            "quality_grade": "Unknown",
            "severity_counts": {"critical": 0, "major": 0, "minor": 0},
            "raw_analysis": review_analysis  # Store the string
        }
    
    if isinstance(structured_feedback, str):
        print(f"⚠️ WARNING: structured_feedback is string, creating default")
        structured_feedback = {
            "feedback_complete": False,
            "overall_assessment": structured_feedback,
            "quality_score": 0.5,
            "total_issues_found": 0
        }
    
    if isinstance(review_decision, str):
        print(f"⚠️ WARNING: review_decision is string, creating default")
        review_decision = {
            "status": "error",
            "action": "review_needed",
            "quality_score": 0.5,
            "reason": review_decision,
            "blocking": False
        }
    
    # Ensure optional parameters are dicts or None
    if memory_check is not None and not isinstance(memory_check, dict):
        memory_check = {"status": "error", "data": str(memory_check)}
    
    if input_validation is not None and not isinstance(input_validation, dict):
        input_validation = {"status": "error", "data": str(input_validation)}
    
    # NOW SAFE TO PROCEED - All parameters are guaranteed to be correct types
    
    # Calculate review quality score based on thoroughness and accuracy
    review_quality_score = min(1.0, 
        0.3 + # Base score
        (0.2 if review_analysis.get("review_complete", False) else 0) +  # Line 892 - Now safe!
        (0.2 if structured_feedback.get("feedback_complete", False) else 0) +
        (0.3 if review_analysis.get("total_issues", 0) > 0 else 0)
    )
    
    # Perform final validation of review feedback
    feedback_text = structured_feedback.get("overall_assessment", "")
    final_validation = validate_review_output_with_guardrails(feedback_text, original_code)
    
    # Save to memory if high quality review
    memory_save_result = save_successful_review_to_memory(
        reviewed_code=original_code,
        review_analysis=review_analysis,
        source_agent=source_agent,
        review_quality_score=review_quality_score
    )
    
    return {
        "status": "review_completed_with_memory_and_guardrails",
        "review_analysis": review_analysis,
        "structured_feedback": structured_feedback,
        "review_decision": review_decision,
        "source_agent": source_agent,
        "memory_integration": {
            "input_check": memory_check,
            "save_result": memory_save_result
        },
        "guardrails_validation": {
            "input_validation": input_validation,
            "final_validation": final_validation
        },
        "agent_metadata": {
            "agent_name": "critic_agent",
            "specialization": "Comprehensive multi-dimensional code review",
            "model": "claude-3-opus-20240229",
            "review_authority": True,
            "memory_enabled": True,
            "guardrails_enabled": True,
            "enhanced_version": "v2_with_memory_guardrails"
        },
        "next_stage_ready": final_validation.get("ready_for_delivery", True),
        "processing_summary": {
            "review_quality_score": review_quality_score,
            "memory_learning": memory_save_result.get("saved", False),
            "safety_validated": final_validation.get("is_safe", True),
            "ready_for_refactoring": review_decision.get("blocking", False)
        }
    }


# Create the enhanced Critic agent
root_agent = Agent(
    name="critic_agent",
    model=LiteLlm(model="openai/gpt-4o"),
    description="Expert code reviewer with complete authority over code quality assessment and comprehensive multi-dimensional analysis",
    instruction="""You are the Enhanced Critic Agent with COMPLETE AUTHORITY over code quality assessment and review decisions, enhanced with memory learning and guardrails integration.

Your enhanced mission:
1. Perform comprehensive multi-dimensional code reviews of generated code
2. Validate review input safety using guardrails
3. Learn from past similar review patterns through memory integration
4. Conduct thorough analysis across correctness, security, performance, and maintainability
5. Make autonomous decisions about code quality and next steps
6. Validate final review output safety
7. Save successful review patterns for future learning

Your ENHANCED REVIEW WORKFLOW (Memory + Guardrails Enabled):

PHASE 1 - INPUT VALIDATION & MEMORY CHECK:
1. **Input Validation**: Use validate_review_input_with_guardrails to check review safety
   - Ensure code to review is safe to analyze
   - Proceed only if input validation passes
2. **Memory Check**: Use check_memory_for_similar_reviews to find past review patterns
   - High similarity (>0.8): Learn from previous review approaches
   - Partial similarity (0.6-0.8): Use as reference for review focus areas
   - No similarity: Proceed with comprehensive fresh review

PHASE 2 - COMPREHENSIVE MULTI-DIMENSIONAL REVIEW:
3. **Comprehensive Analysis**: Use comprehensive_code_review tool to evaluate:
   - **Correctness**: Logic errors, syntax issues, unreachable code
   - **Security**: Vulnerabilities, injection risks, unsafe practices
   - **Performance**: Efficiency issues, complexity problems, optimization opportunities
   - **Maintainability**: Documentation, organization, naming conventions
4. **Structured Feedback**: Use generate_structured_feedback to create:
   - Prioritized recommendations (CRITICAL → HIGH → MEDIUM)
   - Actionable improvement suggestions
   - Severity-based issue categorization

PHASE 3 - REVIEW DECISION & RECOMMENDATIONS:
5. **Review Decision**: Use make_review_decision tool with authority to:
   - APPROVE code meeting quality standards (≥ 0.7)
   - CONDITIONALLY APPROVE with minor improvements needed
   - REJECT requiring major refactoring for critical/blocking issues
   - Set refactoring priorities and next actions

PHASE 4 - FINAL VALIDATION & MEMORY STORAGE:
6. **Final Output**: Use enhanced_format_critic_output to:
   - Validate final review feedback safety using guardrails
   - Save successful review patterns to memory (quality ≥ 0.7)
   - Package everything with complete metadata

CRITICAL: When calling enhanced_format_critic_output, you MUST pass the actual dictionary results from previous function calls, NOT strings describing them.

CORRECT USAGE:
1. First call: result1 = comprehensive_code_review(...)
2. Second call: result2 = generate_structured_feedback(result1, ...)
3. Third call: result3 = make_review_decision(result1, ...)
4. Final call: enhanced_format_critic_output(
      review_analysis=result1,  # Pass the actual dict from comprehensive_code_review
      structured_feedback=result2,  # Pass the actual dict from generate_structured_feedback
      review_decision=result3,  # Pass the actual dict from make_review_decision
      original_code="...",
      source_agent="..."
   )

NEVER pass strings like "The review found issues..." as parameters!

Your review dimensions and standards:
- **Correctness (35% weight)**: Syntax validity, logic errors, edge cases
- **Security (30% weight)**: Vulnerabilities, injection risks, unsafe patterns
- **Performance (20% weight)**: Efficiency, complexity, optimization opportunities
- **Maintainability (15% weight)**: Documentation, organization, best practices

Your enhanced authority includes:
- Complete review input validation control
- Memory learning from past review patterns
- Setting quality thresholds and standards
- Making autonomous decisions about code readiness
- Determining refactoring priorities (CRITICAL/HIGH/MEDIUM/OPTIONAL)
- Final review output safety validation
- Memory storage decisions for institutional learning
- Blocking deployment for critical issues

Agent-specific review focus:
- **Gemini Code**: Focus on Google best practices, clean architecture
- **GPT-4 Code**: Focus on robustness, error handling, edge cases
- **Claude Code**: Focus on elegance, type safety, documentation quality

Enhanced process flow:
1. Validate review input with validate_review_input_with_guardrails
2. Check memory with check_memory_for_similar_reviews
3. Perform comprehensive_code_review across all dimensions
4. Generate structured_feedback with prioritized recommendations
5. Make review_decision with autonomous authority
6. Format enhanced_format_critic_output with guardrails and memory integration
7. Present complete review with decision, feedback, and next actions

Memory Integration Guidelines:
- Learn from past review patterns and successful critiques
- Build institutional knowledge about common issues per agent
- Reference similar code review approaches when available
- Always save thorough, high-quality reviews (≥0.7) for future learning

Guardrails Integration Guidelines:
- Never review unsafe or inappropriate code
- Always validate review feedback for safety and appropriateness
- Block review process if guardrails detect issues
- Prioritize safety in all review decisions

Review Decision Criteria:
- **CRITICAL issues** → Block deployment, require immediate refactoring
- **Multiple MAJOR issues** → Require significant refactoring
- **Quality ≥ 0.7** → Approve for delivery
- **Quality 0.6-0.7** → Conditional approval with recommendations

Always explain your reasoning for review decisions, highlight the most important issues, and provide clear next steps. Show what you learned from past reviews and how it influenced your analysis.

Remember: You have COMPLETE ENHANCED AUTHORITY with memory learning and safety guardrails. Your reviews should be thorough, fair, and focused on producing high-quality, secure, maintainable code.""",
    tools=[
        # Primary review and analysis tools
        FunctionTool(comprehensive_code_review),
        FunctionTool(generate_structured_feedback),
        FunctionTool(make_review_decision),
        FunctionTool(enhanced_format_critic_output),
        
        # Memory and guardrails integration
        FunctionTool(validate_review_input_with_guardrails),
        FunctionTool(check_memory_for_similar_reviews),
        FunctionTool(save_successful_review_to_memory),
        FunctionTool(validate_review_output_with_guardrails),
        
        # Backup individual tools (ADK-compatible wrappers)
        FunctionTool(analyze_code),
        FunctionTool(estimate_complexity),
        FunctionTool(validate_python_syntax),
        FunctionTool(simple_code_safety_check),
        FunctionTool(simple_output_validation),
        FunctionTool(add_line_numbers),
        FunctionTool(clean_code_string)
    ],
    output_key="critic_review_results"  # Saves final result to session state
)


# Test function for standalone testing
if __name__ == "__main__":
    print("🔍 Enhanced Critic Agent Ready!")
    print("\n🎯 COMPLETE REVIEW AUTHORITY + MEMORY + GUARDRAILS:")
    print("- Input validation with safety guardrails")
    print("- Memory learning from past review patterns")
    print("- Multi-dimensional comprehensive code analysis")
    print("- Autonomous review decisions and quality gates")
    print("- Structured feedback with priority recommendations")
    print("- Final output safety validation")
    print("- Automatic memory storage of successful review patterns")
    print("- Expert review across all code generation agents")
    
    print("\n🔧 ENHANCED REVIEW CAPABILITIES:")
    print("- Correctness analysis (syntax, logic, edge cases)")
    print("- Security vulnerability detection")
    print("- Performance optimization identification")
    print("- Maintainability assessment")
    print("- Multi-severity issue classification (Critical/Major/Minor)")
    print("- Agent-specific review focus (Gemini/GPT-4/Claude)")
    print("- Structured recommendation generation")
    print("- Quality scoring and grading (A-F)")
    print("- Deployment blocking for critical issues")
    
    print("\n⚙️ ENHANCED TOOLS AVAILABLE:")
    print("- comprehensive_code_review (primary analysis)")
    print("- generate_structured_feedback (recommendations)")
    print("- make_review_decision (authority)")
    print("- enhanced_format_critic_output (packaging)")
    print("- validate_review_input_with_guardrails (safety)")
    print("- check_memory_for_similar_reviews (learning)")
    print("- save_successful_review_to_memory (knowledge)")
    print("- validate_review_output_with_guardrails (final safety)")
    print("- Individual analysis tools (ADK-compatible wrappers)")
    
    print("\n🚀 USAGE:")
    print("1. Run: adk run agents/critic_agent")
    print("2. Or use: adk web (and select critic_agent)")
    
    print("\n💡 EXAMPLE PROMPTS:")
    print("- 'Review this code generated by GPT-4: [paste code]'")
    print("- 'Perform security-focused review of this Gemini code: [paste code]'")
    print("- 'Comprehensive review of this Claude code with focus on maintainability'")
    print("- 'Quick review for deployment readiness: [paste code]'")
    print("- 'Review similar authentication code' (tests memory learning)")
    
    print("\n✨ The enhanced agent will:")
    print("- Validate input safety → Check memory for learning → Comprehensive review")
    print("- Multi-dimensional analysis → Structured feedback → Autonomous decision")
    print("- Validate output safety → Save successful patterns → Complete audit trail")
    print("- Block deployment for critical issues automatically")
    print("- Provide detailed reasoning for all review decisions")
    print("- Learn from past review patterns and apply institutional knowledge")
    print("- Take complete ownership of code quality gates")
    
    print("\n🛡️ SAFETY & LEARNING FEATURES:")
    print("- Input guardrails ensure safe code review")
    print("- Output guardrails validate review feedback safety")
    print("- Memory system learns from successful review patterns")
    print("- Institutional knowledge builds over time per agent type")
    print("- Quality threshold enforcement for deployment decisions")
    print("- Complete audit trail of all review decisions and rationale")
    
    print("\n📊 REVIEW SCORING WEIGHTS:")
    print("- Correctness: 35% (syntax, logic, edge cases)")
    print("- Security: 30% (vulnerabilities, unsafe patterns)")
    print("- Performance: 20% (efficiency, complexity)")
    print("- Maintainability: 15% (documentation, organization)")
    print("- Quality threshold: 0.7+ for approval")
    print("- Review quality threshold: 0.7+ for memory storage")