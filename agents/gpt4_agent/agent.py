"""
Enhanced GPT-4 Code Generation Agent - Production Ready
Specializes in comprehensive solutions with detailed error handling and edge case coverage

Note: Uses simplified wrapper functions for complex shared tools to ensure
ADK compatibility with automatic function calling.
"""

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator, Dict, Any, Optional, List
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


class GPT4CompletenessAnalyzer:
    """Completeness and robustness analysis engine for GPT-4 agent"""
    
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
        error_handling = GPT4CompletenessAnalyzer.analyze_error_handling(code)
        edge_cases = GPT4CompletenessAnalyzer.analyze_edge_cases(code)
        comprehensiveness = GPT4CompletenessAnalyzer.analyze_comprehensiveness(code, requirements)
        
        # Calculate robustness score
        robustness_score = GPT4CompletenessAnalyzer.calculate_robustness_score(
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
            "robustness_grade": _get_robustness_grade(robustness_score),
            "recommendations": _generate_robustness_recommendations(
                error_handling, edge_cases, comprehensiveness, robustness_score
            ),
            "meets_robustness_standards": robustness_score >= 0.7,
            "analysis_complete": True
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "analysis_complete": False
        }


def _get_robustness_grade(score: float) -> str:
    """Convert robustness score to grade"""
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


def _generate_robustness_recommendations(
    error_handling: Dict[str, Any],
    edge_cases: Dict[str, Any],
    comprehensiveness: Dict[str, Any],
    robustness_score: float
) -> List[str]:
    """Generate robustness improvement recommendations"""
    recommendations = []
    
    if robustness_score < 0.7:
        recommendations.append("Code robustness below acceptable threshold - major improvements needed")
    
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


def format_gpt4_output(
    code: str,
    analysis: Dict[str, Any],
    decision: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format the final output from GPT-4 agent with all metadata.
    
    Args:
        code: The final code
        analysis: Robustness analysis results
        decision: Robustness decision
        
    Returns:
        Formatted output ready for next stage
    """
    formatted_code = wrap_code_in_markdown(code, language="python")
    
    return {
        "status": "completed",
        "generated_code": formatted_code.get("formatted_code", code),
        "raw_code": code,
        "robustness_analysis": analysis,
        "robustness_decision": decision,
        "agent_metadata": {
            "agent_name": "gpt4_agent",
            "specialization": "Comprehensive solutions with detailed error handling",
            "model": "gpt-4",
            "robustness_authority": True
        },
        "next_stage_ready": decision.get("final_decision", False)
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
            "name": "Procedural Approach",
            "pros": ["Straightforward", "Easy to understand"],
            "cons": ["Limited scalability"],
            "recommended": True
        })
    
    return {
        "status": "analyzed",
        "total_approaches": len(approaches),
        "approaches": approaches,
        "recommended_approach": next((a["name"] for a in approaches if a["recommended"]), "Procedural Approach")
    }


# Create the enhanced GPT-4 agent
root_agent = LlmAgent(
    name="gpt4_agent",
    model="gpt-4.1-2025-04-14",
    description="GPT-4 powered code generation expert with complete authority over robustness and comprehensive solution design",
    instruction="""You are the GPT-4 Code Generation Expert with COMPLETE AUTHORITY over solution robustness and comprehensiveness.

Your core mission:
1. Generate comprehensive, production-ready solutions with robust error handling
2. Conduct thorough robustness analysis using your specialized tools
3. Make autonomous decisions about code robustness and completeness
4. Consider multiple implementation approaches and edge cases
5. Take full ownership of the generation-analysis-decision cycle

Your complete process:
1. **Understand Requirements**: Analyze the user's request comprehensively
2. **Analyze Approaches**: Use analyze_implementation_approaches to consider multiple solutions
3. **Generate Robust Code**: Create comprehensive code with:
   - Detailed error handling (try-except blocks)
   - Edge case consideration (null checks, boundary conditions)
   - Input validation and sanitization
   - Comprehensive documentation
   - Logging and debugging support
4. **Robustness Analysis**: Use comprehensive_robustness_analysis tool to evaluate:
   - Error handling patterns
   - Edge case coverage
   - Code comprehensiveness
   - Overall robustness score
5. **Robustness Decision**: Use make_robustness_decision tool with authority to:
   - ACCEPT code meeting robustness standards (≥ 0.75)
   - REJECT and regenerate code below standards (up to 3 times)
   - Make final decisions on code acceptance
6. **Format Output**: Use format_gpt4_output to package everything

Your robustness standards (higher than other agents):
- Comprehensive error handling with specific exception types
- Edge case handling for all inputs
- Input validation and sanitization
- Detailed docstrings with examples and edge cases
- Logging/debugging capabilities
- Boundary condition handling
- Type safety and validation
- Performance considerations
- Robustness score ≥ 0.75 (higher threshold)

Your authority includes:
- Setting robustness thresholds (0.75+)
- Deciding when code is comprehensive enough
- Making final acceptance decisions
- Determining implementation approaches
- Balancing comprehensiveness vs. complexity

Process flow:
1. Analyze requirements and implementation approaches
2. Generate comprehensive code with robust error handling
3. Run comprehensive_robustness_analysis on your code
4. Run make_robustness_decision with the analysis
5. If decision is "regenerate", improve robustness and repeat steps 3-4
6. When decision is "accept", run format_gpt4_output
7. Present final code with complete analysis and reasoning

Always explain your reasoning for robustness decisions and what improvements you made during regeneration cycles. Focus on "what could go wrong" and handle those cases proactively.

Remember: You are the ROBUSTNESS AUTHORITY. Your solutions should be production-ready with comprehensive error handling and edge case coverage.""",
    tools=[
        FunctionTool(comprehensive_robustness_analysis),
        FunctionTool(make_robustness_decision), 
        FunctionTool(format_gpt4_output),
        FunctionTool(analyze_implementation_approaches),
        # Backup individual tools for specific needs (using safe wrappers)
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
    print("\n🎯 COMPLETE ROBUSTNESS AUTHORITY:")
    print("- Comprehensive robustness analysis")
    print("- Autonomous robustness decisions")
    print("- Self-regenerating until standards met (0.75+ threshold)")
    print("- Multiple approach consideration")
    print("- Production-ready comprehensive solutions")
    
    print("\n🔧 ENHANCED CAPABILITIES:")
    print("- Error handling pattern analysis")
    print("- Edge case coverage assessment")
    print("- Comprehensiveness evaluation")
    print("- Implementation approach analysis")
    print("- Robustness scoring (higher standards)")
    print("- Autonomous regeneration (up to 3 times)")
    
    print("\n⚙️ TOOLS AVAILABLE:")
    print("- comprehensive_robustness_analysis (primary)")
    print("- make_robustness_decision (authority)")
    print("- format_gpt4_output (packaging)")
    print("- analyze_implementation_approaches (planning)")
    print("- Individual analysis tools (ADK-compatible wrappers)")
    print("- simple_code_safety_check (security)")
    print("- simple_output_validation (validation)")
    
    print("\n🚀 USAGE:")
    print("1. Run: adk run agents/gpt4_agent")
    print("2. Or use: adk web (and select gpt4_agent)")
    
    print("\n💡 EXAMPLE PROMPTS:")
    print("- 'Create a thread-safe cache implementation with comprehensive error handling'")
    print("- 'Build a file processing system that handles all edge cases'")
    print("- 'Generate a robust API client with retry logic and validation'")
    print("- 'Create a data validation pipeline with detailed error reporting'")
    
    print("\n✨ The agent will:")
    print("- Analyze approaches → Generate robust code → Analyze robustness → Make decision → Format output")
    print("- Regenerate automatically if robustness is below 0.75 threshold")
    print("- Provide detailed reasoning for all robustness decisions")
    print("- Focus on comprehensive error handling and edge cases")
    print("- Take complete ownership of solution robustness")