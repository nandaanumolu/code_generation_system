"""
Code analysis tool for examining and evaluating code quality.
Used by multiple agents for code assessment.
"""

import re
import ast
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class CodeMetrics:
    """Container for code metrics."""
    lines_total: int
    lines_code: int
    lines_comments: int
    lines_blank: int
    function_count: int
    class_count: int
    complexity_score: float
    has_type_hints: bool
    has_docstrings: bool
    has_error_handling: bool


def analyze_code(code: str) -> Dict[str, Any]:
    """
    Analyze code and return comprehensive metrics.
    This is the main function used by agents.
    
    Args:
        code: Source code to analyze
        
    Returns:
        Dictionary containing various code metrics
    """
    if not code:
        return _empty_metrics()
    
    lines = code.split('\n')
    
    # Count different types of lines
    code_lines = []
    comment_lines = []
    blank_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            blank_lines.append(line)
        elif stripped.startswith('#'):
            comment_lines.append(line)
        else:
            code_lines.append(line)
    
    # Detect code patterns
    has_functions = bool(re.search(r'def\s+\w+\s*\(', code))
    has_classes = bool(re.search(r'class\s+\w+\s*[\(:]', code))
    has_docstrings = '"""' in code or "'''" in code
    has_type_hints = _detect_type_hints(code)
    has_error_handling = bool(re.search(r'try\s*:|except\s*[\w\s]*:', code))
    has_imports = bool(re.search(r'^\s*(import|from)\s+', code, re.MULTILINE))
    
    # Count elements
    function_count = len(re.findall(r'def\s+\w+\s*\(', code))
    class_count = len(re.findall(r'class\s+\w+\s*[\(:]', code))
    
    # Estimate complexity
    complexity = estimate_complexity(code)
    
    return {
        "lines_total": len(lines),
        "lines_non_empty": len([l for l in lines if l.strip()]),
        "lines_code": len(code_lines),
        "lines_comments": len(comment_lines),
        "lines_blank": len(blank_lines),
        "has_functions": has_functions,
        "has_classes": has_classes,
        "has_docstrings": has_docstrings,
        "has_imports": has_imports,
        "has_type_hints": has_type_hints,
        "has_error_handling": has_error_handling,
        "function_count": function_count,
        "class_count": class_count,
        "complexity": complexity
    }


def extract_functions(code: str) -> List[Tuple[str, str]]:
    """
    Extract function names and their signatures from code.
    
    Args:
        code: Source code to extract functions from
        
    Returns:
        List of tuples (function_name, full_signature)
    """
    functions = []
    
    # More comprehensive regex for function definitions
    pattern = r'def\s+(\w+)\s*(\([^)]*\))(\s*->\s*[^:]+)?:'
    matches = re.finditer(pattern, code, re.MULTILINE)
    
    for match in matches:
        func_name = match.group(1)
        params = match.group(2)
        return_type = match.group(3) if match.group(3) else ""
        
        full_signature = f"def {func_name}{params}{return_type}"
        functions.append((func_name, full_signature.strip()))
    
    return functions


def estimate_complexity(code: str) -> Dict[str, Any]:
    """
    Estimate code complexity using various metrics.
    
    Args:
        code: Source code to analyze
        
    Returns:
        Dictionary with complexity metrics
    """
    if not code.strip():
        return {"cyclomatic": 1, "nesting": 0, "rating": "low"}
    
    try:
        tree = ast.parse(code)
        complexity = _calculate_cyclomatic_complexity(tree)
        max_nesting = _calculate_max_nesting(tree)
        
        # Determine rating
        if complexity > 20 or max_nesting > 5:
            rating = "high"
        elif complexity > 10 or max_nesting > 3:
            rating = "medium"
        else:
            rating = "low"
        
        return {
            "cyclomatic": complexity,
            "nesting": max_nesting,
            "rating": rating,
            "maintainability_index": _calculate_maintainability_index(code, complexity)
        }
    except:
        # Fallback to simple analysis if AST parsing fails
        return _simple_complexity_estimate(code)


def validate_python_syntax(code: str) -> Dict[str, Any]:
    """
    Validate Python syntax without executing the code.
    
    Args:
        code: Python code to validate
        
    Returns:
        Dictionary with validation results
    """
    try:
        ast.parse(code)
        compile(code, '<string>', 'exec')
        return {
            "valid": True,
            "error": None,
            "line_number": None,
            "error_type": None
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "error": str(e.msg),
            "line_number": e.lineno,
            "error_type": "SyntaxError",
            "offset": e.offset
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "line_number": None,
            "error_type": type(e).__name__
        }


def get_code_metrics(code: str) -> CodeMetrics:
    """
    Get structured code metrics.
    
    Args:
        code: Source code to analyze
        
    Returns:
        CodeMetrics dataclass instance
    """
    analysis = analyze_code(code)
    complexity = estimate_complexity(code)
    
    return CodeMetrics(
        lines_total=analysis["lines_total"],
        lines_code=analysis["lines_code"],
        lines_comments=analysis["lines_comments"],
        lines_blank=analysis["lines_blank"],
        function_count=analysis["function_count"],
        class_count=analysis["class_count"],
        complexity_score=complexity["cyclomatic"],
        has_type_hints=analysis["has_type_hints"],
        has_docstrings=analysis["has_docstrings"],
        has_error_handling=analysis["has_error_handling"]
    )


# Private helper functions

def _empty_metrics() -> Dict[str, Any]:
    """Return empty metrics for empty code."""
    return {
        "lines_total": 0,
        "lines_non_empty": 0,
        "lines_code": 0,
        "lines_comments": 0,
        "lines_blank": 0,
        "has_functions": False,
        "has_classes": False,
        "has_docstrings": False,
        "has_imports": False,
        "has_type_hints": False,
        "has_error_handling": False,
        "function_count": 0,
        "class_count": 0,
        "complexity": {"cyclomatic": 0, "nesting": 0, "rating": "low"}
    }


def _detect_type_hints(code: str) -> bool:
    """Detect if code uses type hints."""
    # Check for parameter type hints
    param_hints = bool(re.search(r':\s*(str|int|float|bool|List|Dict|Any|Optional|Union|Tuple)', code))
    # Check for return type hints
    return_hints = bool(re.search(r'->\s*(str|int|float|bool|List|Dict|Any|Optional|Union|Tuple|None)', code))
    return param_hints or return_hints


def _calculate_cyclomatic_complexity(tree: ast.AST) -> int:
    """Calculate cyclomatic complexity from AST."""
    complexity = 1  # Base complexity
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
        elif isinstance(node, ast.ExceptHandler):
            complexity += 1
    
    return complexity


def _calculate_max_nesting(tree: ast.AST) -> int:
    """Calculate maximum nesting level from AST."""
    max_depth = 0
    
    def _get_depth(node: ast.AST, depth: int = 0) -> int:
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                _get_depth(child, depth + 1)
            else:
                _get_depth(child, depth)
    
    _get_depth(tree)
    return max_depth


def _calculate_maintainability_index(code: str, complexity: int) -> float:
    """Calculate a simple maintainability index (0-100)."""
    # Simplified version of maintainability index
    lines = len(code.split('\n'))
    
    # Base score
    score = 100.0
    
    # Deduct for complexity
    score -= complexity * 2
    
    # Deduct for length
    if lines > 100:
        score -= (lines - 100) * 0.1
    
    # Bonus for documentation
    if '"""' in code or "'''" in code:
        score += 5
    
    return max(0, min(100, score))


def _simple_complexity_estimate(code: str) -> Dict[str, Any]:
    """Simple complexity estimate when AST parsing fails."""
    lines = code.split('\n')
    
    # Count control structures
    if_count = len(re.findall(r'\bif\s+', code))
    loop_count = len(re.findall(r'\b(for|while)\s+', code))
    try_count = len(re.findall(r'\btry\s*:', code))
    
    complexity = 1 + if_count + loop_count + try_count
    
    # Estimate nesting by indentation
    max_indent = 0
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent // 4)
    
    return {
        "cyclomatic": complexity,
        "nesting": max_indent,
        "rating": "high" if complexity > 10 else "medium" if complexity > 5 else "low",
        "maintainability_index": 100 - (complexity * 5)
    }