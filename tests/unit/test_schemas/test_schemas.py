"""
Unit tests for schemas.
Run with: pytest tests/unit/test_schemas/test_schemas.py -v
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.schemas import (
    # Common
    Language, CodeComplexity, QualityMetrics,
    AgentResponse, ErrorResponse, SuccessResponse,
    
    # Code generation
    CodeRequest, CodeResponse, GenerationConfig,
    GeneratedCode, ParallelGenerationResult,
    
    # Review
    CodeReview, ReviewIssue, IssueSeverity, IssueCategory,
    RefactorRequest, RefactorResponse
)


class TestCommonSchemas:
    """Test common schema classes."""
    
    def test_language_enum(self):
        """Test Language enum."""
        assert Language.PYTHON.value == "python"
        assert Language.from_string("Python") == Language.PYTHON
        assert Language.from_string("js") == Language.JAVASCRIPT
        assert Language.from_string("unknown_lang") == Language.UNKNOWN
    
    def test_quality_metrics(self):
        """Test QualityMetrics."""
        metrics = QualityMetrics(
            lines_of_code=100,
            cyclomatic_complexity=8,
            maintainability_index=75.0,
            documentation_score=0.8,
            type_hint_coverage=0.9
        )
        
        assert metrics.complexity_level == CodeComplexity.MEDIUM
        assert 0 <= metrics.overall_score <= 1
    
    def test_agent_response(self):
        """Test AgentResponse."""
        response = AgentResponse(
            success=True,
            message="Test successful",
            agent_name="test_agent"
        )
        
        assert response.success == True
        assert response.agent_name == "test_agent"
        assert isinstance(response.timestamp, datetime)


class TestCodeGenerationSchemas:
    """Test code generation schemas."""
    
    def test_code_request_validation(self):
        """Test CodeRequest validation."""
        # Valid request
        request = CodeRequest(
            prompt="Create a function to add two numbers"
        )
        assert request.prompt == "Create a function to add two numbers"
        assert request.config.language == Language.PYTHON
        
        # Invalid - empty prompt
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            CodeRequest(prompt="")
        
        # Invalid - too short
        with pytest.raises(ValueError, match="Prompt too short"):
            CodeRequest(prompt="test")
    
    def test_generated_code(self):
        """Test GeneratedCode."""
        code = GeneratedCode(
            code="def add(a, b): return a + b",
            language=Language.PYTHON,
            generator_agent="test_agent",
            generation_time_ms=100,
            confidence_score=0.95
        )
        
        assert code.line_count == 1
        assert code.confidence_score == 0.95
        
        # Invalid - empty code
        with pytest.raises(ValueError, match="Generated code cannot be empty"):
            GeneratedCode(
                code="",
                language=Language.PYTHON,
                generator_agent="test",
                generation_time_ms=100,
                confidence_score=0.5
            )
    
    def test_code_response(self):
        """Test CodeResponse."""
        generated = GeneratedCode(
            code="def add(a, b): return a + b",
            language=Language.PYTHON,
            generator_agent="test_agent",
            generation_time_ms=100,
            confidence_score=0.95
        )
        
        response = CodeResponse(
            success=True,
            message="Generated successfully",
            agent_name="test_agent",
            generated_code=generated,
            suggestions=["Add type hints"]
        )
        
        assert response.primary_code == "def add(a, b): return a + b"
        assert len(response.suggestions) == 1


class TestReviewSchemas:
    """Test review schemas."""
    
    def test_review_issue(self):
        """Test ReviewIssue."""
        issue = ReviewIssue(
            severity=IssueSeverity.MAJOR,
            category=IssueCategory.ERROR_HANDLING,
            message="No error handling for division by zero",
            line_number=5,
            suggestion="Add zero check"
        )
        
        assert issue.severity.priority == 3  # Major priority
        assert issue.affects_lines == [5]
        
        # Test line range
        issue_range = ReviewIssue(
            severity=IssueSeverity.MINOR,
            category=IssueCategory.STYLE,
            message="Lines too long",
            line_range=(10, 15)
        )
        assert len(issue_range.affects_lines) == 6
    
    def test_code_review(self):
        """Test CodeReview."""
        issue1 = ReviewIssue(
            severity=IssueSeverity.CRITICAL,
            category=IssueCategory.SECURITY,
            message="SQL injection vulnerability"
        )
        
        issue2 = ReviewIssue(
            severity=IssueSeverity.MINOR,
            category=IssueCategory.STYLE,
            message="Line too long"
        )
        
        metrics = QualityMetrics(
            lines_of_code=50,
            cyclomatic_complexity=5,
            maintainability_index=80.0,
            documentation_score=0.7,
            type_hint_coverage=0.8
        )
        
        review = CodeReview(
            success=True,
            message="Review completed",
            agent_name="critic_agent",
            code_analyzed="def test(): pass",
            language=Language.PYTHON,
            issues=[issue1, issue2],
            quality_metrics=metrics,
            summary="Code has security issues",
            review_config=ReviewConfig()
        )
        
        assert review.total_issues == 2
        assert review.has_blocking_issues == True
        assert len(review.critical_issues) == 1
        assert review.overall_rating == "needs_critical_fixes"
    
    def test_refactor_response(self):
        """Test RefactorResponse."""
        metrics = QualityMetrics(
            lines_of_code=50,
            cyclomatic_complexity=5,
            maintainability_index=85.0,
            documentation_score=0.8,
            type_hint_coverage=0.9
        )
        
        response = RefactorResponse(
            success=True,
            message="Refactoring complete",
            agent_name="refactor_agent",
            refactored_code="def divide(a, b):\n    if b == 0:\n        raise ValueError()\n    return a / b",
            changes_made=[],
            issues_addressed=[],
            issues_remaining=[],
            new_metrics=metrics,
            improvement_summary="Added error handling"
        )
        
        assert response.success == True
        assert "ValueError" in response.refactored_code


# Quick test runner
if __name__ == "__main__":
    print("Testing schemas module...")
    
    # Test Language enum
    print("\n=== Testing Language Enum ===")
    print(f"Python: {Language.PYTHON}")
    print(f"From 'js': {Language.from_string('js')}")
    print(f"From 'C++': {Language.from_string('C++')}")
    
    # Test CodeRequest
    print("\n=== Testing CodeRequest ===")
    request = CodeRequest(
        prompt="Create a Python function to calculate fibonacci numbers",
        constraints=["Use recursion", "Add memoization"]
    )
    print(f"Prompt: {request.prompt}")
    print(f"Language: {request.config.language}")
    print(f"Constraints: {request.constraints}")
    
    # Test ReviewIssue
    print("\n=== Testing ReviewIssue ===")
    issue = ReviewIssue(
        severity=IssueSeverity.MAJOR,
        category=IssueCategory.PERFORMANCE,
        message="Recursive fibonacci without memoization is O(2^n)",
        line_number=3,
        suggestion="Add memoization or use iterative approach"
    )
    print(f"Issue: {issue.message}")
    print(f"Severity: {issue.severity} (priority: {issue.severity.priority})")
    print(f"Suggestion: {issue.suggestion}")
    
    print("\nSchemas module is working correctly!")
    print("Run full tests with: pytest tests/unit/test_schemas/test_schemas.py -v")