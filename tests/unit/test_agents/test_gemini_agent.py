"""
Unit tests for Gemini Agent
Run with: pytest tests/unit/test_agents/test_gemini_agent.py -v
Or direct: python tests/unit/test_agents/test_gemini_agent.py
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the agent and tools
from agents.gemini_agent.agent import root_agent
from shared.tools import analyze_code, estimate_complexity, validate_python_syntax
from shared.guardrails import validate_output_code


class TestGeminiAgentConfiguration:
    """Test Gemini agent configuration and setup."""
    
    def test_agent_exists(self):
        """Test that the agent is properly defined."""
        assert root_agent is not None
        assert root_agent.name == "gemini_agent"
        assert root_agent.model == "gemini-2.0-flash"
    
    def test_agent_has_tools(self):
        """Test that the agent has all necessary tools."""
        assert len(root_agent.tools) == 10  # We added 10 tools
        
        # Check for specific important tools
        tool_names = [tool.function.__name__ for tool in root_agent.tools]
        assert "analyze_code" in tool_names
        assert "validate_output_code" in tool_names
        assert "estimate_complexity" in tool_names
    
    def test_agent_description(self):
        """Test agent description and instruction."""
        assert "comprehensive analysis tools" in root_agent.description
        assert "COMPLETE AUTHORITY" in root_agent.instruction


class TestGeminiAgentTools:
    """Test the tools available to Gemini agent."""
    
    @pytest.fixture
    def sample_code(self):
        """Sample code for testing."""
        return """
def calculate_factorial(n: int) -> int:
    '''Calculate factorial of n recursively.'''
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)

def fibonacci(n: int) -> int:
    '''Calculate nth Fibonacci number.'''
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
"""
    
    def test_analyze_code_tool(self, sample_code):
        """Test the analyze_code tool."""
        result = analyze_code(sample_code)
        assert result['function_count'] == 2
        assert result['has_type_hints'] == True
        assert result['has_error_handling'] == True
        assert result['has_docstrings'] == True
    
    def test_complexity_tool(self, sample_code):
        """Test the estimate_complexity tool."""
        result = estimate_complexity(sample_code)
        assert 'cyclomatic' in result
        assert 'nesting' in result
        assert 'rating' in result
        assert result['rating'] in ['low', 'medium', 'high']
    
    def test_syntax_validation_tool(self, sample_code):
        """Test the validate_python_syntax tool."""
        result = validate_python_syntax(sample_code)
        assert result['valid'] == True
        assert result['error'] is None
    
    def test_security_validation_tool(self, sample_code):
        """Test the validate_output_code tool."""
        result = validate_output_code(sample_code)
        assert result.is_safe == True
        assert result.syntax_valid == True
    
    def test_dangerous_code_detection(self):
        """Test that dangerous code is detected."""
        dangerous_code = "import os\nos.system('rm -rf /')"
        result = validate_output_code(dangerous_code)
        assert result.is_safe == False
        assert len(result.security_issues) > 0


# Direct execution tests (without pytest)
def test_agent_configuration():
    """Test the Gemini agent configuration."""
    print("=== Testing Gemini Agent Configuration ===")
    print(f"Agent Name: {root_agent.name}")
    print(f"Model: {root_agent.model}")
    print(f"Number of tools: {len(root_agent.tools)}")
    print("\nAvailable tools:")
    for tool in root_agent.tools:
        print(f"  - {tool.function.__name__}")
    
    assert root_agent.name == "gemini_agent", "Agent name mismatch"
    assert len(root_agent.tools) == 10, f"Expected 10 tools, got {len(root_agent.tools)}"
    print("\n✅ Configuration test passed!")


def test_tools_functionality():
    """Test individual tools the agent can use."""
    print("\n=== Testing Tool Functionality ===")
    
    # Test code sample
    test_code = """
def add(a: int, b: int) -> int:
    '''Add two numbers.'''
    return a + b
"""
    
    # Test analyze_code
    print("\n1. Testing analyze_code:")
    analysis = analyze_code(test_code)
    assert analysis['has_functions'] == True
    assert analysis['has_type_hints'] == True
    print("   ✅ Code analysis working")
    
    # Test syntax validation
    print("\n2. Testing validate_python_syntax:")
    syntax_result = validate_python_syntax(test_code)
    assert syntax_result['valid'] == True
    print("   ✅ Syntax validation working")
    
    # Test security validation
    print("\n3. Testing validate_output_code:")
    security_result = validate_output_code(test_code)
    assert security_result.is_safe == True
    print("   ✅ Security validation working")
    
    print("\n✅ All tool tests passed!")


def test_agent_authority():
    """Test agent authority concepts."""
    print("\n=== Testing Agent Authority Concepts ===")
    
    # Check instruction contains authority keywords
    instruction = root_agent.instruction.lower()
    authority_keywords = ["authority", "decide", "standards", "autonomous"]
    
    found_keywords = [kw for kw in authority_keywords if kw in instruction]
    print(f"Authority keywords found: {found_keywords}")
    
    assert len(found_keywords) >= 2, "Agent instruction should emphasize authority"
    print("✅ Agent has proper authority instructions")


if __name__ == "__main__":
    # Run tests directly (without pytest)
    print("Running Gemini Agent Tests...\n")
    
    try:
        test_agent_configuration()
        test_tools_functionality()
        test_agent_authority()
        
        print("\n" + "="*50)
        print("🎉 All tests passed!")
        print("="*50)
        
        print("\nTo run with pytest:")
        print("  pytest tests/unit/test_agents/test_gemini_agent.py -v")
        
        print("\nTo test the agent interactively:")
        print("  adk run agents/gemini_agent")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)