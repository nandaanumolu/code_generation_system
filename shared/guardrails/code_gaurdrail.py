"""
Code-specific guardrail for detailed pattern validation.
Focuses on code quality, security patterns, and best practices.
"""

import re
import ast
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..schemas import Language, IssueSeverity, IssueCategory
from ..tools import analyze_code, estimate_complexity


logger = logging.getLogger(__name__)


class SecurityIssueType(str, Enum):
    """Types of security issues."""
    SQL_INJECTION = "sql_injection"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    XSS = "cross_site_scripting"
    HARDCODED_SECRET = "hardcoded_secret"
    WEAK_CRYPTO = "weak_cryptography"
    INSECURE_RANDOM = "insecure_random"
    XXE = "xml_external_entity"
    SSRF = "server_side_request_forgery"
    LDAP_INJECTION = "ldap_injection"


@dataclass
class SecurityIssue:
    """A security issue found in code."""
    issue_type: SecurityIssueType
    severity: IssueSeverity
    line_number: Optional[int]
    description: str
    recommendation: str
    cwe_id: Optional[str] = None  # Common Weakness Enumeration ID


@dataclass
class CodeValidationResult:
    """Result of code validation."""
    is_secure: bool
    security_issues: List[SecurityIssue] = field(default_factory=list)
    quality_issues: List[Dict[str, Any]] = field(default_factory=list)
    best_practices: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0.0 (safe) to 1.0 (critical)
    
    @property
    def total_issues(self) -> int:
        return len(self.security_issues) + len(self.quality_issues)
    
    @property
    def has_critical_issues(self) -> bool:
        return any(issue.severity == IssueSeverity.CRITICAL for issue in self.security_issues)


class CodeGuardrail:
    """
    Comprehensive code validation for security and quality.
    Performs deep analysis of code patterns and potential vulnerabilities.
    """
    
    # Security patterns by language
    PYTHON_SECURITY_PATTERNS = [
        # SQL Injection
        (
            r'cursor\.execute\s*\(\s*["\'][^"\']*%[sdf][^"\']*["\'].*%.*\)',
            SecurityIssueType.SQL_INJECTION,
            "Using string formatting in SQL queries",
            "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
            "CWE-89"
        ),
        (
            r'f["\'].*SELECT.*WHERE.*{.*}',
            SecurityIssueType.SQL_INJECTION,
            "Using f-strings in SQL queries",
            "Use parameterized queries instead of f-strings",
            "CWE-89"
        ),
        
        # Command Injection
        (
            r'os\.system\s*\([^)]*\+[^)]*\)',
            SecurityIssueType.COMMAND_INJECTION,
            "Concatenating user input in os.system",
            "Use subprocess with list arguments instead",
            "CWE-78"
        ),
        (
            r'subprocess\..*shell\s*=\s*True.*\+',
            SecurityIssueType.COMMAND_INJECTION,
            "Using shell=True with user input",
            "Use shell=False and pass arguments as list",
            "CWE-78"
        ),
        
        # Path Traversal
        (
            r'open\s*\([^)]*\+[^)]*["\']\.\.\/["\']',
            SecurityIssueType.PATH_TRAVERSAL,
            "Potential path traversal in file operations",
            "Validate and sanitize file paths",
            "CWE-22"
        ),
        
        # Hardcoded Secrets
        (
            r'(password|api_key|secret|token)\s*=\s*["\'][^"\']{8,}["\']',
            SecurityIssueType.HARDCODED_SECRET,
            "Hardcoded credential or secret",
            "Use environment variables or secure credential storage",
            "CWE-798"
        ),
        
        # Weak Crypto
        (
            r'from\s+Crypto\.Hash\s+import\s+(MD5|SHA1)',
            SecurityIssueType.WEAK_CRYPTO,
            "Using weak cryptographic hash function",
            "Use SHA256 or stronger hash functions",
            "CWE-327"
        ),
        
        # Insecure Random
        (
            r'random\.(random|randint|choice)\s*\(',
            SecurityIssueType.INSECURE_RANDOM,
            "Using insecure random for security purposes",
            "Use secrets module for cryptographic randomness",
            "CWE-330"
        ),
    ]
    
    JAVASCRIPT_SECURITY_PATTERNS = [
        # XSS
        (
            r'innerHTML\s*=\s*[^;]*\+',
            SecurityIssueType.XSS,
            "Direct innerHTML assignment with concatenation",
            "Use textContent or properly escape HTML",
            "CWE-79"
        ),
        (
            r'document\.write\s*\([^)]*\+',
            SecurityIssueType.XSS,
            "Using document.write with user input",
            "Use safe DOM manipulation methods",
            "CWE-79"
        ),
        
        # Eval
        (
            r'eval\s*\([^)]*\+',
            SecurityIssueType.COMMAND_INJECTION,
            "Using eval with user input",
            "Parse JSON or use safe alternatives",
            "CWE-94"
        ),
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional configuration."""
        self.config = config or {}
        self.check_security = self.config.get("check_security", True)
        self.check_quality = self.config.get("check_quality", True)
        self.check_complexity = self.config.get("check_complexity", True)
        self.max_complexity = self.config.get("max_complexity", 10)
        self.min_quality_score = self.config.get("min_quality_score", 0.7)
    
    def validate(self, code: str, language: Language = Language.PYTHON) -> CodeValidationResult:
        """
        Validate code for security and quality issues.
        
        Args:
            code: Code to validate
            language: Programming language
            
        Returns:
            Comprehensive validation result
        """
        security_issues = []
        quality_issues = []
        best_practices = []
        
        # Security validation
        if self.check_security:
            security_issues = self._check_security_patterns(code, language)
        
        # Quality validation
        if self.check_quality:
            quality_issues = self._check_code_quality(code, language)
        
        # Complexity validation
        if self.check_complexity and language == Language.PYTHON:
            complexity_issues = self._check_complexity(code)
            quality_issues.extend(complexity_issues)
        
        # Best practices
        best_practices = self._check_best_practices(code, language)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(security_issues, quality_issues)
        
        # Determine if secure
        is_secure = (
            not any(issue.severity == IssueSeverity.CRITICAL for issue in security_issues) and
            risk_score < 0.7
        )
        
        return CodeValidationResult(
            is_secure=is_secure,
            security_issues=security_issues,
            quality_issues=quality_issues,
            best_practices=best_practices,
            risk_score=risk_score
        )
    
    def _check_security_patterns(self, code: str, language: Language) -> List[SecurityIssue]:
        """Check for security patterns in code."""
        issues = []
        
        # Select patterns based on language
        if language == Language.PYTHON:
            patterns = self.PYTHON_SECURITY_PATTERNS
        elif language == Language.JAVASCRIPT:
            patterns = self.JAVASCRIPT_SECURITY_PATTERNS
        else:
            return issues  # No patterns for other languages yet
        
        # Check each pattern
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, issue_type, description, recommendation, cwe_id in patterns:
                if re.search(pattern, line):
                    # Determine severity based on issue type
                    if issue_type in [SecurityIssueType.SQL_INJECTION, 
                                     SecurityIssueType.COMMAND_INJECTION]:
                        severity = IssueSeverity.CRITICAL
                    elif issue_type in [SecurityIssueType.XSS, 
                                       SecurityIssueType.PATH_TRAVERSAL]:
                        severity = IssueSeverity.MAJOR
                    else:
                        severity = IssueSeverity.MINOR
                    
                    issues.append(SecurityIssue(
                        issue_type=issue_type,
                        severity=severity,
                        line_number=line_num,
                        description=description,
                        recommendation=recommendation,
                        cwe_id=cwe_id
                    ))
        
        # AST-based analysis for Python
        if language == Language.PYTHON:
            try:
                tree = ast.parse(code)
                ast_issues = self._analyze_python_ast_security(tree)
                issues.extend(ast_issues)
            except:
                pass
        
        return issues
    
    def _analyze_python_ast_security(self, tree: ast.AST) -> List[SecurityIssue]:
        """Analyze Python AST for security issues."""
        issues = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.issues = []
            
            def visit_Call(self, node):
                # Check for pickle usage
                if (isinstance(node.func, ast.Attribute) and 
                    node.func.attr in ['loads', 'load'] and
                    isinstance(node.func.value, ast.Name) and
                    node.func.value.id == 'pickle'):
                    self.issues.append(SecurityIssue(
                        issue_type=SecurityIssueType.COMMAND_INJECTION,
                        severity=IssueSeverity.MAJOR,
                        line_number=node.lineno,
                        description="Unsafe pickle deserialization",
                        recommendation="Avoid pickle for untrusted data, use JSON instead",
                        cwe_id="CWE-502"
                    ))
                
                self.generic_visit(node)
        
        visitor = SecurityVisitor()
        visitor.visit(tree)
        return visitor.issues
    
    def _check_code_quality(self, code: str, language: Language) -> List[Dict[str, Any]]:
        """Check code quality issues."""
        issues = []
        
        # Analyze code metrics
        metrics = analyze_code(code)
        
        # Check for missing docstrings
        if language == Language.PYTHON and not metrics["has_docstrings"]:
            issues.append({
                "type": "documentation",
                "severity": "minor",
                "message": "Missing docstrings",
                "recommendation": "Add docstrings to functions and classes"
            })
        
        # Check for missing type hints (Python)
        if language == Language.PYTHON and not metrics["has_type_hints"]:
            issues.append({
                "type": "type_safety",
                "severity": "minor",
                "message": "Missing type hints",
                "recommendation": "Add type hints for better code clarity"
            })
        
        # Check for missing error handling
        if not metrics["has_error_handling"] and metrics["has_functions"]:
            issues.append({
                "type": "error_handling",
                "severity": "major",
                "message": "No error handling found",
                "recommendation": "Add try-except blocks for error handling"
            })
        
        return issues
    
    def _check_complexity(self, code: str) -> List[Dict[str, Any]]:
        """Check code complexity."""
        issues = []
        
        complexity = estimate_complexity(code)
        
        if complexity["cyclomatic"] > self.max_complexity:
            issues.append({
                "type": "complexity",
                "severity": "major",
                "message": f"High cyclomatic complexity: {complexity['cyclomatic']}",
                "recommendation": "Refactor to reduce complexity"
            })
        
        if complexity["nesting"] > 4:
            issues.append({
                "type": "complexity",
                "severity": "minor",
                "message": f"Deep nesting level: {complexity['nesting']}",
                "recommendation": "Reduce nesting by extracting functions"
            })
        
        return issues
    
    def _check_best_practices(self, code: str, language: Language) -> List[str]:
        """Check for best practices."""
        practices = []
        
        if language == Language.PYTHON:
            # Check for good practices
            if 'if __name__ == "__main__":' in code:
                practices.append("Good: Uses proper main guard")
            
            if 'logging' in code and not 'print(' in code:
                practices.append("Good: Uses logging instead of print")
            
            if 'typing' in code or '->' in code:
                practices.append("Good: Uses type hints")
            
            if 'pytest' in code or 'unittest' in code:
                practices.append("Good: Includes tests")
        
        return practices
    
    def _calculate_risk_score(
        self, 
        security_issues: List[SecurityIssue], 
        quality_issues: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall risk score (0.0 to 1.0)."""
        score = 0.0
        
        # Weight security issues heavily
        for issue in security_issues:
            if issue.severity == IssueSeverity.CRITICAL:
                score += 0.3
            elif issue.severity == IssueSeverity.MAJOR:
                score += 0.15
            elif issue.severity == IssueSeverity.MINOR:
                score += 0.05
        
        # Weight quality issues less
        for issue in quality_issues:
            if issue.get("severity") == "major":
                score += 0.05
            else:
                score += 0.02
        
        return min(1.0, score)


# Convenience functions

def validate_code_patterns(
    code: str,
    language: Language = Language.PYTHON
) -> CodeValidationResult:
    """
    Validate code patterns using default guardrail.
    
    Args:
        code: Code to validate
        language: Programming language
        
    Returns:
        Validation result
    """
    guardrail = CodeGuardrail()
    return guardrail.validate(code, language)


def check_security_issues(
    code: str,
    language: Language = Language.PYTHON
) -> List[SecurityIssue]:
    """
    Check for security issues in code.
    
    Args:
        code: Code to check
        language: Programming language
        
    Returns:
        List of security issues
    """
    result = validate_code_patterns(code, language)
    return result.security_issues


def get_code_risk_level(
    code: str,
    language: Language = Language.PYTHON
) -> str:
    """
    Get risk level of code.
    
    Args:
        code: Code to analyze
        language: Programming language
        
    Returns:
        Risk level string
    """
    result = validate_code_patterns(code, language)
    
    if result.risk_score >= 0.8:
        return "critical"
    elif result.risk_score >= 0.6:
        return "high"
    elif result.risk_score >= 0.4:
        return "medium"
    elif result.risk_score >= 0.2:
        return "low"
    else:
        return "safe"