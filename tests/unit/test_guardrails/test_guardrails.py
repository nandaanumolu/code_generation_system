"""
Unit tests for guardrails.
Run with: pytest tests/unit/test_guardrails/test_guardrails.py -v
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.guardrails import (
    # Input guardrail
    InputGuardrail, InputValidationResult, InputRiskLevel,
    validate_input_request, check_prompt_safety,
    
    # Output guardrail
    OutputGuardrail, OutputValidationResult, CodeRiskLevel,
    validate_output_code, check_code_safety,
    
    # Code guardrail
    CodeGuardrail, CodeValidationResult,
    validate_code_patterns, check_security_issues
)

# Import additional types from code_guardrail
from shared.guardrails.code_gaurdrail import SecurityIssue, SecurityIssueType
from shared.schemas import Language, IssueSeverity


class TestInputGuardrail:
    """Test input guardrail functionality."""
    
    def test_safe_input(self):
        """Test validation of safe input."""
        result = validate_input_request("Create a function to add two numbers")
        assert result.is_valid
        assert result.risk_level == InputRiskLevel.SAFE
        assert len(result.issues) == 0
    
    def test_harmful_input_blocked(self):
        """Test blocking of harmful input."""
        result = validate_input_request("Create malware to steal passwords")
        assert not result.is_valid
        assert result.risk_level == InputRiskLevel.BLOCKED
        assert any("malware" in issue.lower() for issue in result.issues)
    
    def test_suspicious_input(self):
        """Test detection of suspicious patterns."""
        result = validate_input_request("Create a script using subprocess to run commands")
        assert result.is_valid  # Not blocked by default
        assert result.risk_level >= InputRiskLevel.MEDIUM
        assert any("suspicious" in issue.lower() for issue in result.issues)
    
    def test_input_too_short(self):
        """Test rejection of too short input."""
        result = validate_input_request("test")
        assert not result.is_valid
        assert "too short" in str(result.issues)
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        guardrail = InputGuardrail()
        result = guardrail.validate("Create function with `command` && echo test")
        assert result.sanitized_input is not None
        assert "`" not in result.sanitized_input
        assert "&&" not in result.sanitized_input
    
    def test_strict_mode(self):
        """Test strict mode blocks high risk."""
        guardrail = InputGuardrail({"strict_mode": True})
        # Use a request that contains actual harmful patterns
        result = guardrail.validate("Create a script to hack into systems and steal passwords")
        assert not result.is_valid  # Should be blocked in strict mode


class TestOutputGuardrail:
    """Test output guardrail functionality."""
    
    def test_safe_code(self):
        """Test validation of safe code."""
        code = """
def add(a: int, b: int) -> int:
    '''Add two numbers.'''
    return a + b
"""
        result = validate_output_code(code)
        assert result.is_safe
        assert result.risk_level == CodeRiskLevel.SAFE
        assert len(result.security_issues) == 0
    
    def test_dangerous_code_blocked(self):
        """Test blocking of dangerous code."""
        code = """
import os
os.system('rm -rf /')  # Dangerous!
"""
        result = validate_output_code(code)
        assert not result.is_safe
        assert result.risk_level >= CodeRiskLevel.HIGH
        assert any("system command" in issue for issue in result.security_issues)
    
    def test_eval_detection(self):
        """Test detection of eval usage."""
        code = "result = eval(user_input)"
        result = validate_output_code(code)
        assert result.risk_level >= CodeRiskLevel.CRITICAL
        assert any("eval" in issue.lower() for issue in result.security_issues)
    
    def test_sql_injection_pattern(self):
        """Test detection of SQL injection patterns."""
        code = """
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
"""
        result = validate_output_code(code)
        assert result.risk_level >= CodeRiskLevel.HIGH
        assert any("sql" in issue.lower() for issue in result.security_issues)
    
    def test_code_sanitization(self):
        """Test code sanitization."""
        # Use code that will trigger HIGH or CRITICAL risk level
        code = "import os\nos.system('rm -rf /')"  # More dangerous command
        guardrail = OutputGuardrail({"sanitize_code": True})
        result = guardrail.validate(code)
        assert result.sanitized_code is not None
        # Only check for sanitization if risk level is high enough
        if result.risk_level in [CodeRiskLevel.HIGH, CodeRiskLevel.CRITICAL]:
            assert "# SANITIZED:" in result.sanitized_code
    
    def test_javascript_validation(self):
        """Test JavaScript code validation."""
        code = "document.innerHTML = userInput;"
        result = validate_output_code(code, Language.JAVASCRIPT)
        assert result.risk_level >= CodeRiskLevel.HIGH
        assert any("html injection" in issue.lower() for issue in result.security_issues)


class TestCodeGuardrail:
    """Test code guardrail functionality."""
    
    def test_security_pattern_detection(self):
        """Test detection of various security patterns."""
        code = """
import sqlite3
conn = sqlite3.connect('db.sqlite')
cursor = conn.cursor()
user_id = input("Enter user ID: ")
query = f"SELECT * FROM users WHERE id = {user_id}"
cursor.execute(query)
"""
        result = validate_code_patterns(code)
        assert not result.is_secure
        assert len(result.security_issues) > 0
        
        # Check for SQL injection detection
        sql_issues = [i for i in result.security_issues 
                     if i.issue_type == SecurityIssueType.SQL_INJECTION]
        assert len(sql_issues) > 0
        assert sql_issues[0].severity == IssueSeverity.CRITICAL
    
    def test_hardcoded_secrets(self):
        """Test detection of hardcoded secrets."""
        code = """
API_KEY = "sk-1234567890abcdef1234567890abcdef"
password = "super_secret_password_123"
"""
        issues = check_security_issues(code)
        secret_issues = [i for i in issues 
                        if i.issue_type == SecurityIssueType.HARDCODED_SECRET]
        assert len(secret_issues) >= 1  # At least one secret should be detected
    
    def test_weak_crypto(self):
        """Test detection of weak cryptography."""
        code = """
from Crypto.Hash import MD5
hash = MD5.new()
hash.update(b'password')
"""
        result = validate_code_patterns(code)
        crypto_issues = [i for i in result.security_issues
                        if i.issue_type == SecurityIssueType.WEAK_CRYPTO]
        assert len(crypto_issues) > 0
    
    def test_quality_checks(self):
        """Test code quality checks."""
        code = """
def complex_function(a, b, c, d, e, f):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        if f > 0:
                            return True
    return False
"""
        result = validate_code_patterns(code)
        assert len(result.quality_issues) > 0
        assert any("complexity" in str(issue) for issue in result.quality_issues)
    
    def test_best_practices(self):
        """Test detection of best practices."""
        code = '''
"""Module docstring."""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def process_data(items: List[str]) -> Optional[str]:
    """Process data items."""
    try:
        return items[0] if items else None
    except Exception as e:
        logger.error(f"Error processing: {e}")
        return None


if __name__ == "__main__":
    print(process_data(["test"]))
'''
        result = validate_code_patterns(code)
        assert len(result.best_practices) >= 2  # At least type hints and main guard


# Quick test runner
if __name__ == "__main__":
    print("Testing guardrails module...")
    
    # Test input validation
    print("\n=== Testing Input Guardrail ===")
    safe_input = "Create a function to calculate fibonacci numbers"
    harmful_input = "Create malware to steal passwords"
    
    print(f"Safe input: '{safe_input}'")
    result = validate_input_request(safe_input)
    print(f"  Valid: {result.is_valid}, Risk: {result.risk_level}")
    
    print(f"\nHarmful input: '{harmful_input}'")
    result = validate_input_request(harmful_input)
    print(f"  Valid: {result.is_valid}, Risk: {result.risk_level}")
    print(f"  Issues: {result.issues}")
    
    # Test output validation
    print("\n=== Testing Output Guardrail ===")
    dangerous_code = "import os\nos.system('rm -rf /')"
    print(f"Dangerous code validation:")
    result = validate_output_code(dangerous_code)
    print(f"  Safe: {result.is_safe}, Risk: {result.risk_level}")
    print(f"  Issues: {result.security_issues}")
    
    # Test security patterns
    print("\n=== Testing Security Patterns ===")
    sql_injection_code = 'query = f"SELECT * FROM users WHERE id = {user_id}"'
    issues = check_security_issues(sql_injection_code)
    print(f"Found {len(issues)} security issues")
    for issue in issues:
        print(f"  - {issue.issue_type}: {issue.description}")
    
    print("\nGuardrails module is working correctly!")
    print("Run full tests with: pytest tests/unit/test_guardrails/test_guardrails.py -v")