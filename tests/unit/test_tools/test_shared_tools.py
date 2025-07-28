"""
Unit tests for shared tools.
Run with: pytest tests/unit/test_tools/test_shared_tools.py -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.tools import (
    analyze_code, 
    wrap_code_in_markdown, 
    add_line_numbers,
    extract_functions,
    format_code_output,
    estimate_complexity,
    validate_python_syntax
)


class TestCodeAnalysis:
    """Test code analysis functions."""
    
    def test_analyze_empty_code(self):
        """Test analyzing empty code."""
        result = analyze_code("")
        assert result["lines_total"] == 0
        assert result["has_functions"] == False
        assert result["has_classes"] == False
    
    def test_analyze_simple_function(self):
        """Test analyzing a simple function."""
        code = """
def hello():
    print("Hello, World!")
"""
        result = analyze_code(code)
        assert result["has_functions"] == True
        assert result["function_count"] == 1
        assert result["has_classes"] == False
    
    def test_analyze_with_type_hints(self):
        """Test analyzing code with type hints."""
        code = """
def add(a: int, b: int) -> int:
    return a + b
"""
        result = analyze_code(code)
        assert result["has_type_hints"] == True
        assert result["has_functions"] == True
    
    def test_analyze_with_error_handling(self):
        """Test analyzing code with try/except."""
        code = """
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
"""
        result = analyze_code(code)
        assert result["has_error_handling"] == True


class TestFunctionExtraction:
    """Test function extraction."""
    
    def test_extract_simple_function(self):
        """Test extracting a simple function."""
        code = "def foo(): pass"
        functions = extract_functions(code)
        assert len(functions) == 1
        assert functions[0][0] == "foo"
        assert functions[0][1] == "def foo()"
    
    def test_extract_function_with_params(self):
        """Test extracting function with parameters."""
        code = "def add(a: int, b: int) -> int: return a + b"
        functions = extract_functions(code)
        assert len(functions) == 1
        assert functions[0][0] == "add"
        assert "-> int" in functions[0][1]
    
    def test_extract_multiple_functions(self):
        """Test extracting multiple functions."""
        code = """
def first():
    pass

def second(x, y):
    return x + y

def third() -> None:
    print("test")
"""
        functions = extract_functions(code)
        assert len(functions) == 3
        assert functions[0][0] == "first"
        assert functions[1][0] == "second"
        assert functions[2][0] == "third"


class TestFormatting:
    """Test formatting functions."""
    
    def test_wrap_code_in_markdown(self):
        """Test markdown wrapping."""
        code = "print('hello')"
        result = wrap_code_in_markdown(code)
        assert result.startswith("```python")
        assert result.endswith("```")
        assert "print('hello')" in result
    
    def test_wrap_empty_code(self):
        """Test wrapping empty code."""
        result = wrap_code_in_markdown("")
        assert "No code provided" in result
    
    def test_add_line_numbers_simple(self):
        """Test adding line numbers."""
        code = "line1\nline2\nline3"
        result = add_line_numbers(code)
        assert "1 | line1" in result
        assert "2 | line2" in result
        assert "3 | line3" in result
    
    def test_add_line_numbers_with_highlight(self):
        """Test adding line numbers with highlights."""
        code = "line1\nline2\nline3"
        result = add_line_numbers(code, highlight_lines=[2])
        assert "1 | line1" in result
        assert "2>| line2" in result
        assert "3 | line3" in result
    
    def test_format_code_output_complete(self):
        """Test complete code formatting."""
        code = "def test(): pass"
        result = format_code_output(
            code,
            title="Test Function",
            show_line_numbers=True,
            metadata={"author": "Test", "version": "1.0"}
        )
        assert "## Test Function" in result
        assert "author: Test" in result
        assert "1 | def test(): pass" in result


class TestComplexity:
    """Test complexity estimation."""
    
    def test_simple_complexity(self):
        """Test complexity of simple code."""
        code = "x = 1"
        result = estimate_complexity(code)
        assert result["cyclomatic"] == 1
        assert result["rating"] == "low"
    
    def test_medium_complexity(self):
        """Test complexity with control flow."""
        code = """
def check(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
"""
        result = estimate_complexity(code)
        assert result["cyclomatic"] > 1
        assert result["rating"] in ["low", "medium"]
    
    def test_high_complexity(self):
        """Test high complexity code."""
        code = """
def complex_function(data):
    result = []
    for item in data:
        if item > 0:
            if item % 2 == 0:
                for i in range(item):
                    if i % 3 == 0:
                        result.append(i)
                    elif i % 5 == 0:
                        result.append(i * 2)
            else:
                try:
                    result.append(item / 2)
                except:
                    pass
    return result
"""
        result = estimate_complexity(code)
        assert result["cyclomatic"] > 5
        assert result["nesting"] > 2


class TestSyntaxValidation:
    """Test syntax validation."""
    
    def test_valid_syntax(self):
        """Test valid Python syntax."""
        code = "def foo(): return 42"
        result = validate_python_syntax(code)
        assert result["valid"] == True
        assert result["error"] is None
    
    def test_invalid_syntax(self):
        """Test invalid Python syntax."""
        code = "def foo() return 42"  # Missing colon
        result = validate_python_syntax(code)
        assert result["valid"] == False
        assert result["error"] is not None
        assert result["error_type"] == "SyntaxError"
    
    def test_syntax_error_line_number(self):
        """Test that line numbers are reported correctly."""
        code = """
def foo():
    return 42
    
def bar()  # Missing colon
    return 24
"""
        result = validate_python_syntax(code)
        assert result["valid"] == False
        assert result["line_number"] == 5


# Quick test runner for development
if __name__ == "__main__":
    # Run a quick test
    print("Running quick test of shared tools...")
    
    test_code = """
def fibonacci(n: int) -> int:
    '''Calculate fibonacci number'''
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
    
    print("\n=== Code Analysis ===")
    analysis = analyze_code(test_code)
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    print("\n=== Function Extraction ===")
    functions = extract_functions(test_code)
    for name, sig in functions:
        print(f"Function: {name}")
        print(f"Signature: {sig}")
    
    print("\n=== Syntax Validation ===")
    validation = validate_python_syntax(test_code)
    print(f"Valid: {validation['valid']}")
    
    print("\n=== Line Numbers ===")
    print(add_line_numbers(test_code))
    
    print("\nTo run full test suite: pytest tests/unit/test_tools/test_shared_tools.py -v")