"""
Output guardrail for validating and sanitizing generated code.
Ensures generated code is safe and appropriate.
"""

import re
import ast
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from ..schemas import GeneratedCode, Language
from ..tools import validate_python_syntax


logger = logging.getLogger(__name__)


class CodeRiskLevel(str, Enum):
    """Risk levels for generated code."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OutputValidationResult:
    """Result of output validation."""
    is_safe: bool
    risk_level: CodeRiskLevel
    security_issues: List[str]
    sanitized_code: Optional[str] = None
    modifications: List[str] = None
    syntax_valid: bool = True
    
    def __post_init__(self):
        if self.modifications is None:
            self.modifications = []


class OutputGuardrail:
    """
    Validates and sanitizes generated code for safety.
    Checks for dangerous patterns, security vulnerabilities, and malicious code.
    """
    
    # Dangerous Python patterns
    DANGEROUS_PATTERNS = [
        # System operations
        (r'os\.system\s*\(', "Direct system command execution"),
        (r'subprocess\.(call|run|Popen)\s*\(', "Subprocess execution"),
        (r'eval\s*\(', "Eval usage (code injection risk)"),
        (r'exec\s*\(', "Exec usage (code injection risk)"),
        (r'__import__\s*\(', "Dynamic import (can load malicious modules)"),
        
        # File operations
        (r'open\s*\([^,)]*["\']w["\']', "File write operation"),
        (r'os\.remove|os\.unlink', "File deletion"),
        (r'shutil\.rmtree', "Directory deletion"),
        (r'os\.chmod', "File permission modification"),
        
        # Network operations
        (r'socket\.socket', "Raw socket creation"),
        (r'urllib\.request\.urlopen', "URL fetching"),
        (r'requests\.(get|post|put|delete)', "HTTP requests"),
        
        # Dangerous modules
        (r'import\s+(ctypes|pickle|marshal)', "Potentially dangerous module import"),
        (r'from\s+(ctypes|pickle|marshal)', "Potentially dangerous module import"),
    ]
    
    # Suspicious patterns (not blocked but flagged)
    SUSPICIOUS_PATTERNS = [
        # Environment access
        (r'os\.environ', "Environment variable access"),
        (r'os\.getenv', "Environment variable access"),
        
        # File system navigation
        (r'os\.listdir', "Directory listing"),
        (r'os\.walk', "Directory traversal"),
        (r'glob\.glob', "File pattern matching"),
        
        # Process operations
        (r'os\.getpid|os\.kill', "Process manipulation"),
        (r'signal\.\w+', "Signal handling"),
        
        # Encoding/Decoding
        (r'base64\.(encode|decode)', "Base64 operations"),
        (r'\.encode\(["\']hex["\']', "Hex encoding"),
    ]
    
    # Security vulnerabilities
    SECURITY_PATTERNS = [
        # SQL Injection
        (r'["\']SELECT.*WHERE.*%s["\']', "Potential SQL injection"),
        (r'f["\']SELECT.*WHERE.*{', "SQL injection via f-string"),
        (r'\.format\(\).*SELECT.*WHERE', "SQL injection via format"),
        
        # Path Traversal
        (r'\.\./', "Path traversal pattern"),
        (r'os\.path\.join\([^,]+,\s*user_input', "Potential path traversal"),
        
        # Command Injection
        (r'os\.system.*\+.*input', "Command injection risk"),
        (r'subprocess.*shell=True', "Shell injection risk"),
        
        # XSS (for web code)
        (r'innerHTML\s*=.*user', "Potential XSS vulnerability"),
        (r'document\.write\(.*user', "Potential XSS vulnerability"),
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional configuration."""
        self.config = config or {}
        self.block_dangerous = self.config.get("block_dangerous", True)
        self.sanitize_code = self.config.get("sanitize_code", True)
        self.check_syntax = self.config.get("check_syntax", True)
        self.max_code_length = self.config.get("max_code_length", 10000)
    
    def validate(self, code: str, language: Language = Language.PYTHON) -> OutputValidationResult:
        """
        Validate generated code for safety.
        
        Args:
            code: The generated code to validate
            language: Programming language of the code
            
        Returns:
            Validation result with risk assessment
        """
        security_issues = []
        risk_level = CodeRiskLevel.SAFE
        
        # Check code length
        if len(code) > self.max_code_length:
            security_issues.append(f"Code exceeds maximum length ({self.max_code_length} chars)")
            code = code[:self.max_code_length]
        
        # Language-specific validation
        if language == Language.PYTHON:
            result = self._validate_python(code)
        elif language == Language.JAVASCRIPT:
            result = self._validate_javascript(code)
        else:
            # Basic validation for other languages
            result = self._validate_generic(code)
        
        security_issues.extend(result["issues"])
        risk_level = result["risk_level"]
        
        # Check syntax if enabled
        syntax_valid = True
        if self.check_syntax and language == Language.PYTHON:
            syntax_result = validate_python_syntax(code)
            syntax_valid = syntax_result["valid"]
            if not syntax_valid:
                security_issues.append(f"Syntax error: {syntax_result['error']}")
        
        # Sanitize code if needed
        sanitized_code = code
        modifications = []
        if self.sanitize_code and risk_level in [CodeRiskLevel.HIGH, CodeRiskLevel.CRITICAL]:
            sanitized_code, modifications = self._sanitize_code(code, language)
        
        # Determine if code is safe
        is_safe = risk_level in [CodeRiskLevel.SAFE, CodeRiskLevel.LOW]
        if self.block_dangerous and risk_level == CodeRiskLevel.CRITICAL:
            is_safe = False
        
        return OutputValidationResult(
            is_safe=is_safe,
            risk_level=risk_level,
            security_issues=security_issues,
            sanitized_code=sanitized_code,
            modifications=modifications,
            syntax_valid=syntax_valid
        )
    
    def validate_generated_code(self, generated: GeneratedCode) -> OutputValidationResult:
        """
        Validate a GeneratedCode object.
        
        Args:
            generated: The generated code object
            
        Returns:
            Validation result
        """
        return self.validate(generated.code, generated.language)
    
    def _validate_python(self, code: str) -> Dict[str, Any]:
        """Validate Python code specifically."""
        issues = []
        risk_level = CodeRiskLevel.SAFE
        
        # Check dangerous patterns
        for pattern, description in self.DANGEROUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Dangerous: {description}")
                risk_level = max(risk_level, CodeRiskLevel.HIGH)
                
                # Some patterns are critical
                if any(keyword in description.lower() for keyword in ["eval", "exec", "system"]):
                    risk_level = CodeRiskLevel.CRITICAL
        
        # Check suspicious patterns
        suspicious_count = 0
        for pattern, description in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Suspicious: {description}")
                suspicious_count += 1
        
        if suspicious_count >= 3:
            risk_level = max(risk_level, CodeRiskLevel.MEDIUM)
        
        # Check security vulnerabilities
        for pattern, description in self.SECURITY_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE | re.DOTALL):
                issues.append(f"Security: {description}")
                risk_level = max(risk_level, CodeRiskLevel.HIGH)
        
        # AST-based analysis for Python
        try:
            tree = ast.parse(code)
            ast_issues = self._analyze_python_ast(tree)
            issues.extend(ast_issues)
            if ast_issues:
                risk_level = max(risk_level, CodeRiskLevel.MEDIUM)
        except:
            pass  # Syntax errors handled separately
        
        return {"issues": issues, "risk_level": risk_level}
    
    def _validate_javascript(self, code: str) -> Dict[str, Any]:
        """Validate JavaScript code specifically."""
        issues = []
        risk_level = CodeRiskLevel.SAFE
        
        # JavaScript-specific dangerous patterns
        js_dangerous = [
            (r'eval\s*\(', "Eval usage"),
            (r'new\s+Function\s*\(', "Dynamic function creation"),
            (r'innerHTML\s*=', "Direct HTML injection"),
            (r'document\.write\s*\(', "Document.write usage"),
            (r'child_process', "Child process execution"),
            (r'fs\.(unlink|rmdir|rm)', "File deletion"),
        ]
        
        for pattern, description in js_dangerous:
            if re.search(pattern, code):
                issues.append(f"Dangerous: {description}")
                risk_level = max(risk_level, CodeRiskLevel.HIGH)
        
        return {"issues": issues, "risk_level": risk_level}
    
    def _validate_generic(self, code: str) -> Dict[str, Any]:
        """Generic validation for any language."""
        issues = []
        risk_level = CodeRiskLevel.SAFE
        
        # Generic dangerous patterns
        generic_dangerous = [
            (r'rm\s+-rf', "Dangerous file deletion"),
            (r'format\s+c:', "Disk formatting"),
            (r':(){ :|:& };:', "Fork bomb"),
            (r'telnet|netcat|nc\s', "Network tools"),
        ]
        
        for pattern, description in generic_dangerous:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append(f"Dangerous: {description}")
                risk_level = CodeRiskLevel.HIGH
        
        return {"issues": issues, "risk_level": risk_level}
    
    def _analyze_python_ast(self, tree: ast.AST) -> List[str]:
        """Analyze Python AST for dangerous patterns."""
        issues = []
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                        issues.append(f"AST: Dangerous function call: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['system', 'popen', 'remove', 'rmtree']:
                        issues.append(f"AST: Dangerous method call: {node.func.attr}")
            
            # Check for import of dangerous modules
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['os', 'subprocess', 'socket', 'ctypes']:
                        issues.append(f"AST: Import of sensitive module: {alias.name}")
        
        return issues
    
    def _sanitize_code(self, code: str, language: Language) -> Tuple[str, List[str]]:
        """
        Sanitize code by commenting out dangerous lines.
        
        Args:
            code: Code to sanitize
            language: Programming language
            
        Returns:
            Tuple of (sanitized_code, modifications)
        """
        modifications = []
        lines = code.split('\n')
        sanitized_lines = []
        
        comment_prefix = "#" if language == Language.PYTHON else "//"
        
        for i, line in enumerate(lines):
            # Check if line contains dangerous patterns
            is_dangerous = False
            for pattern, description in self.DANGEROUS_PATTERNS:
                if re.search(pattern, line):
                    is_dangerous = True
                    modifications.append(f"Commented line {i+1}: {description}")
                    break
            
            if is_dangerous:
                sanitized_lines.append(f"{comment_prefix} SANITIZED: {line}")
            else:
                sanitized_lines.append(line)
        
        return '\n'.join(sanitized_lines), modifications


# Convenience functions

def validate_output_code(
    code: str,
    language: Language = Language.PYTHON,
    strict: bool = True
) -> OutputValidationResult:
    """
    Validate generated code using default guardrail.
    
    Args:
        code: Code to validate
        language: Programming language
        strict: Whether to block dangerous code
        
    Returns:
        Validation result
    """
    guardrail = OutputGuardrail({"block_dangerous": strict})
    return guardrail.validate(code, language)


def check_code_safety(code: str, language: Language = Language.PYTHON) -> bool:
    """
    Quick check if code is safe.
    
    Args:
        code: Code to check
        language: Programming language
        
    Returns:
        True if safe, False otherwise
    """
    result = validate_output_code(code, language)
    return result.is_safe and result.risk_level in [CodeRiskLevel.SAFE, CodeRiskLevel.LOW]


def sanitize_code(code: str, language: Language = Language.PYTHON) -> str:
    """
    Sanitize code using default guardrail.
    
    Args:
        code: Code to sanitize
        language: Programming language
        
    Returns:
        Sanitized code
    """
    guardrail = OutputGuardrail()
    result = guardrail.validate(code, language)
    return result.sanitized_code or code