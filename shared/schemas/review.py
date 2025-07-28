"""
Schemas for code review and refactoring operations.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict

from .common import AgentResponse, Language, QualityMetrics


class IssueSeverity(str, Enum):
    """Severity levels for code issues."""
    CRITICAL = "critical"    # Must fix - security, crashes, data loss
    MAJOR = "major"         # Should fix - bugs, performance issues
    MINOR = "minor"         # Nice to fix - style, minor improvements
    SUGGESTION = "suggestion"  # Optional improvements
    
    @property
    def priority(self) -> int:
        """Get numeric priority (higher = more severe)."""
        priorities = {
            self.CRITICAL: 4,
            self.MAJOR: 3,
            self.MINOR: 2,
            self.SUGGESTION: 1
        }
        return priorities.get(self, 0)


class IssueCategory(str, Enum):
    """Categories of code issues."""
    SECURITY = "security"
    BUG = "bug"
    PERFORMANCE = "performance"
    STYLE = "style"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"
    TYPE_SAFETY = "type_safety"
    ERROR_HANDLING = "error_handling"
    LOGIC = "logic"
    NAMING = "naming"
    STRUCTURE = "structure"
    DUPLICATION = "duplication"
    COMPLEXITY = "complexity"
    TESTABILITY = "testability"
    OTHER = "other"


class ReviewIssue(BaseModel):
    """A single issue found during code review."""
    
    severity: IssueSeverity = Field(description="Issue severity")
    category: IssueCategory = Field(description="Issue category")
    message: str = Field(description="Description of the issue")
    line_number: Optional[int] = Field(None, ge=1, description="Line number where issue occurs")
    line_range: Optional[Tuple[int, int]] = Field(None, description="Range of lines affected")
    code_snippet: Optional[str] = Field(None, description="Relevant code snippet")
    suggestion: Optional[str] = Field(None, description="Suggested fix or improvement")
    rule_id: Optional[str] = Field(None, description="ID of violated rule/standard")
    
    @field_validator('line_range')
    def validate_line_range(cls, v):
        if v and v[0] > v[1]:
            raise ValueError("Invalid line range: start must be <= end")
        return v
    
    @field_validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError("Issue message cannot be empty")
        return v.strip()
    
    @property
    def affects_lines(self) -> List[int]:
        """Get all affected line numbers."""
        if self.line_number:
            return [self.line_number]
        elif self.line_range:
            return list(range(self.line_range[0], self.line_range[1] + 1))
        return []
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "severity": "major",
                "category": "security",
                "message": "SQL injection vulnerability: user input directly concatenated into query",
                "line_number": 15,
                "code_snippet": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
                "suggestion": "Use parameterized queries or prepared statements"
            }
        }
    )


class ReviewConfig(BaseModel):
    """Configuration for code review."""
    
    check_security: bool = Field(True, description="Check for security vulnerabilities")
    check_performance: bool = Field(True, description="Check for performance issues")
    check_style: bool = Field(True, description="Check code style and formatting")
    check_complexity: bool = Field(True, description="Check code complexity")
    check_documentation: bool = Field(True, description="Check documentation completeness")
    check_type_safety: bool = Field(True, description="Check type annotations (Python)")
    check_error_handling: bool = Field(True, description="Check error handling")
    max_line_length: int = Field(100, ge=80, le=120, description="Maximum line length")
    max_function_length: int = Field(50, ge=20, le=100, description="Maximum function length")
    max_complexity: int = Field(10, ge=5, le=20, description="Maximum cyclomatic complexity")
    custom_rules: List[str] = Field(default_factory=list, description="Custom review rules")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "check_security": True,
                "check_performance": True,
                "max_line_length": 100,
                "max_complexity": 10
            }
        }
    )


class CodeReview(AgentResponse):
    """Complete code review results."""
    
    code_analyzed: str = Field(description="The code that was reviewed")
    language: Language = Field(description="Programming language")
    issues: List[ReviewIssue] = Field(default_factory=list, description="Issues found")
    quality_metrics: QualityMetrics = Field(description="Code quality metrics")
    summary: str = Field(description="Executive summary of the review")
    strengths: List[str] = Field(default_factory=list, description="Code strengths")
    review_config: ReviewConfig = Field(description="Configuration used for review")
    
    @property
    def total_issues(self) -> int:
        """Get total number of issues."""
        return len(self.issues)
    
    @property
    def issues_by_severity(self) -> Dict[IssueSeverity, int]:
        """Get issue count by severity."""
        counts = {severity: 0 for severity in IssueSeverity}
        for issue in self.issues:
            counts[issue.severity] += 1
        return counts
    
    @property
    def critical_issues(self) -> List[ReviewIssue]:
        """Get only critical issues."""
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]
    
    @property
    def has_blocking_issues(self) -> bool:
        """Check if there are any blocking issues."""
        return any(i.severity == IssueSeverity.CRITICAL for i in self.issues)
    
    @property
    def overall_rating(self) -> str:
        """Get overall rating based on issues and metrics."""
        if self.has_blocking_issues:
            return "needs_critical_fixes"
        elif self.issues_by_severity[IssueSeverity.MAJOR] > 3:
            return "needs_major_improvements"
        elif self.total_issues > 10:
            return "needs_improvements"
        elif self.quality_metrics.overall_score > 0.8:
            return "excellent"
        else:
            return "good"
    
    def get_issues_for_lines(self, start: int, end: int) -> List[ReviewIssue]:
        """Get issues affecting a range of lines."""
        result = []
        for issue in self.issues:
            affected = issue.affects_lines
            if any(start <= line <= end for line in affected):
                result.append(issue)
        return result
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Code review completed",
                "agent_name": "critic_agent",
                "code_analyzed": "def divide(a, b): return a / b",
                "language": "python",
                "issues": [
                    {
                        "severity": "major",
                        "category": "error_handling",
                        "message": "No handling for division by zero",
                        "line_number": 1
                    }
                ],
                "summary": "Code needs error handling improvements"
            }
        }
    )


class RefactorRequest(BaseModel):
    """Request to refactor code based on review."""
    
    original_code: str = Field(description="Original code to refactor")
    language: Language = Field(description="Programming language")
    review: CodeReview = Field(description="Review containing issues to address")
    priorities: List[IssueSeverity] = Field(
        default_factory=lambda: list(IssueSeverity),
        description="Priority order for fixing issues"
    )
    preserve_functionality: bool = Field(True, description="Ensure functionality remains the same")
    max_changes: Optional[int] = Field(None, description="Maximum number of changes to make")
    target_metrics: Optional[QualityMetrics] = Field(None, description="Target quality metrics")
    
    @field_validator('original_code')
    def validate_code(cls, v):
        if not v.strip():
            raise ValueError("Original code cannot be empty")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "original_code": "def divide(a, b): return a / b",
                "language": "python",
                "review": {"issues": [{"severity": "major", "category": "error_handling"}]},
                "priorities": ["critical", "major", "minor"]
            }
        }
    )


class RefactorChange(BaseModel):
    """A single change made during refactoring."""
    
    issue_addressed: ReviewIssue = Field(description="The issue this change addresses")
    original_lines: List[str] = Field(description="Original code lines")
    new_lines: List[str] = Field(description="New code lines")
    line_range: Tuple[int, int] = Field(description="Line range affected")
    description: str = Field(description="Description of the change")
    
    @property
    def diff_preview(self) -> str:
        """Get a simple diff preview."""
        result = []
        for line in self.original_lines:
            result.append(f"- {line}")
        for line in self.new_lines:
            result.append(f"+ {line}")
        return "\n".join(result)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "issue_addressed": {"severity": "major", "message": "No error handling"},
                "original_lines": ["def divide(a, b): return a / b"],
                "new_lines": [
                    "def divide(a, b):",
                    "    if b == 0:",
                    "        raise ValueError('Cannot divide by zero')",
                    "    return a / b"
                ],
                "line_range": (1, 1),
                "description": "Added zero division check"
            }
        }
    )


class RefactorResponse(AgentResponse):
    """Response containing refactored code."""
    
    refactored_code: str = Field(description="The refactored code")
    changes_made: List[RefactorChange] = Field(description="List of changes made")
    issues_addressed: List[ReviewIssue] = Field(description="Issues that were addressed")
    issues_remaining: List[ReviewIssue] = Field(description="Issues not addressed")
    new_metrics: QualityMetrics = Field(description="Quality metrics after refactoring")
    improvement_summary: str = Field(description="Summary of improvements")
    
    @property
    def total_changes(self) -> int:
        """Get total number of changes made."""
        return len(self.changes_made)
    
    @property
    def addressed_rate(self) -> float:
        """Get percentage of issues addressed."""
        total = len(self.issues_addressed) + len(self.issues_remaining)
        if total == 0:
            return 1.0
        return len(self.issues_addressed) / total
    
    @property
    def quality_improvement(self) -> float:
        """Get quality score improvement (requires original metrics)."""
        # This would need original metrics passed in
        return 0.0
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Code refactored successfully",
                "agent_name": "refactor_agent",
                "refactored_code": "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b",
                "changes_made": [
                    {
                        "description": "Added zero division check",
                        "line_range": (1, 1)
                    }
                ],
                "improvement_summary": "Added error handling for division by zero"
            }
        }
    )


class ReviewRefactorCycle(BaseModel):
    """Result of a complete review-refactor cycle."""
    
    iteration: int = Field(description="Iteration number")
    initial_code: str = Field(description="Code at start of iteration")
    review: CodeReview = Field(description="Review results")
    refactor_response: Optional[RefactorResponse] = Field(None, description="Refactoring results")
    final_code: str = Field(description="Code at end of iteration")
    quality_before: float = Field(description="Quality score before")
    quality_after: float = Field(description="Quality score after")
    should_continue: bool = Field(description="Whether another iteration is needed")
    
    @property
    def quality_delta(self) -> float:
        """Get quality improvement in this iteration."""
        return self.quality_after - self.quality_before
    
    @property
    def was_successful(self) -> bool:
        """Check if iteration improved the code."""
        return self.quality_delta > 0
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "iteration": 1,
                "initial_code": "def divide(a, b): return a / b",
                "quality_before": 0.6,
                "quality_after": 0.85,
                "should_continue": False
            }
        }
    )