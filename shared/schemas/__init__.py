"""
Schemas for data validation across the code generation system.
Uses Pydantic for runtime type checking and validation.
"""

from .common import (
    AgentResponse,
    ErrorResponse,
    SuccessResponse,
    Language,
    CodeComplexity,
    QualityMetrics
)

from .code_generation import (
    CodeRequest,
    CodeResponse,
    GenerationConfig,
    CodeExample,
    GeneratedCode,
    ParallelGenerationResult
)

from .review import (
    CodeReview,
    ReviewIssue,
    IssueSeverity,
    IssueCategory,
    RefactorRequest,
    RefactorResponse,
    ReviewConfig
)

__all__ = [
    # Common schemas
    'AgentResponse',
    'ErrorResponse',
    'SuccessResponse',
    'Language',
    'CodeComplexity',
    'QualityMetrics',
    
    # Code generation schemas
    'CodeRequest',
    'CodeResponse',
    'GenerationConfig',
    'CodeExample',
    'GeneratedCode',
    'ParallelGenerationResult',
    
    # Review schemas
    'CodeReview',
    'ReviewIssue',
    'IssueSeverity',
    'IssueCategory',
    'RefactorRequest',
    'RefactorResponse',
    'ReviewConfig',
]