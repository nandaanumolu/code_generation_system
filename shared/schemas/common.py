"""
Common schemas used across the code generation system.
"""

from typing import Dict, List, Any, Optional, Union, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict


class Language(str, Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, value: str) -> 'Language':
        """Create from string, defaulting to UNKNOWN."""
        try:
            return cls(value.lower())
        except ValueError:
            # Try to match common variations
            mappings = {
                "c++": cls.CPP,
                "c#": cls.CSHARP,
                "js": cls.JAVASCRIPT,
                "ts": cls.TYPESCRIPT,
                "py": cls.PYTHON,
            }
            return mappings.get(value.lower(), cls.UNKNOWN)


class CodeComplexity(str, Enum):
    """Code complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class QualityMetrics(BaseModel):
    """Metrics for code quality assessment."""
    
    lines_of_code: int = Field(ge=0, description="Total lines of code")
    cyclomatic_complexity: int = Field(ge=1, description="Cyclomatic complexity score")
    maintainability_index: float = Field(ge=0, le=100, description="Maintainability index (0-100)")
    test_coverage: Optional[float] = Field(None, ge=0, le=100, description="Test coverage percentage")
    documentation_score: float = Field(ge=0, le=1, description="Documentation completeness (0-1)")
    type_hint_coverage: float = Field(ge=0, le=1, description="Type hint coverage for Python (0-1)")
    
    @property
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        scores = [
            self.maintainability_index / 100,
            self.documentation_score,
            self.type_hint_coverage,
        ]
        if self.test_coverage is not None:
            scores.append(self.test_coverage / 100)
        
        return sum(scores) / len(scores)
    
    @property
    def complexity_level(self) -> CodeComplexity:
        """Determine complexity level from cyclomatic complexity."""
        if self.cyclomatic_complexity <= 5:
            return CodeComplexity.LOW
        elif self.cyclomatic_complexity <= 10:
            return CodeComplexity.MEDIUM
        elif self.cyclomatic_complexity <= 20:
            return CodeComplexity.HIGH
        else:
            return CodeComplexity.VERY_HIGH
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lines_of_code": 150,
                "cyclomatic_complexity": 8,
                "maintainability_index": 75.5,
                "test_coverage": 85.0,
                "documentation_score": 0.9,
                "type_hint_coverage": 0.95
            }
        }
    )


class AgentResponse(BaseModel):
    """Base response from any agent."""
    
    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Human-readable message")
    agent_name: str = Field(description="Name of the responding agent")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class ErrorResponse(AgentResponse):
    """Error response from an agent."""
    
    success: Literal[False] = Field(default=False)
    error_code: str = Field(description="Error code for programmatic handling")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")
    traceback: Optional[str] = Field(None, description="Stack trace if available")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "message": "Failed to generate code: syntax error",
                "agent_name": "gemini_agent",
                "error_code": "SYNTAX_ERROR",
                "error_details": {"line": 5, "column": 10}
            }
        }
    )


class SuccessResponse(AgentResponse):
    """Success response from an agent."""
    
    success: Literal[True] = Field(default=True)
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Code generated successfully",
                "agent_name": "gemini_agent",
                "data": {"code": "def hello(): pass"}
            }
        }
    )


class ValidationError(BaseModel):
    """Validation error details."""
    
    field: str = Field(description="Field that failed validation")
    message: str = Field(description="Error message")
    value: Any = Field(description="The invalid value")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "field": "max_tokens",
                "message": "Value must be between 1 and 4000",
                "value": 5000
            }
        }
    )


class WorkflowState(BaseModel):
    """State information for workflow execution."""
    
    workflow_id: str = Field(description="Unique workflow identifier")
    current_stage: str = Field(description="Current stage in the workflow")
    stages_completed: List[str] = Field(default_factory=list, description="Completed stages")
    start_time: datetime = Field(default_factory=datetime.now, description="Workflow start time")
    end_time: Optional[datetime] = Field(None, description="Workflow end time")
    data: Dict[str, Any] = Field(default_factory=dict, description="Workflow data")
    errors: List[ErrorResponse] = Field(default_factory=list, description="Any errors encountered")
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate workflow duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def is_complete(self) -> bool:
        """Check if workflow is complete."""
        return self.end_time is not None
    
    def add_stage(self, stage: str):
        """Mark a stage as completed."""
        if stage not in self.stages_completed:
            self.stages_completed.append(stage)
        self.current_stage = stage
    
    def complete(self):
        """Mark workflow as complete."""
        self.end_time = datetime.now()
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )


class BatchRequest(BaseModel):
    """Request for batch processing."""
    
    items: List[Dict[str, Any]] = Field(description="Items to process")
    parallel: bool = Field(default=True, description="Process items in parallel")
    max_concurrent: int = Field(default=3, ge=1, le=10, description="Max concurrent processing")
    stop_on_error: bool = Field(default=False, description="Stop batch on first error")
    
    @field_validator('items')
    def validate_items(cls, v):
        if not v:
            raise ValueError("Items list cannot be empty")
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 items")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {"request": "Create a function to add two numbers"},
                    {"request": "Create a function to multiply two numbers"}
                ],
                "parallel": True,
                "max_concurrent": 3
            }
        }
    )


class BatchResponse(BaseModel):
    """Response from batch processing."""
    
    total_items: int = Field(description="Total number of items processed")
    successful: int = Field(description="Number of successful items")
    failed: int = Field(description="Number of failed items")
    results: List[Union[SuccessResponse, ErrorResponse]] = Field(description="Individual results")
    processing_time_ms: int = Field(description="Total processing time")
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_items == 0:
            return 0.0
        return self.successful / self.total_items
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_items": 2,
                "successful": 2,
                "failed": 0,
                "results": [
                    {"success": True, "message": "Generated add function"},
                    {"success": True, "message": "Generated multiply function"}
                ],
                "processing_time_ms": 1500
            }
        }
    )