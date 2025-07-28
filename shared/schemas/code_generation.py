"""
Schemas for code generation requests and responses.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict

from .common import Language, CodeComplexity, QualityMetrics, AgentResponse


class GenerationConfig(BaseModel):
    """Configuration for code generation."""
    
    language: Language = Field(Language.PYTHON, description="Target programming language")
    max_tokens: int = Field(2000, ge=100, le=4000, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0, le=2, description="Sampling temperature")
    include_tests: bool = Field(False, description="Include unit tests")
    include_docstrings: bool = Field(True, description="Include documentation")
    include_type_hints: bool = Field(True, description="Include type hints (Python)")
    include_error_handling: bool = Field(True, description="Include error handling")
    style_guide: Optional[str] = Field(None, description="Style guide to follow (e.g., PEP8, Google)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "language": "python",
                "max_tokens": 2000,
                "temperature": 0.7,
                "include_tests": True,
                "include_docstrings": True
            }
        }
    )


class CodeExample(BaseModel):
    """Example code for context or demonstration."""
    
    code: str = Field(description="Example code snippet")
    language: Language = Field(description="Programming language")
    description: Optional[str] = Field(None, description="Description of the example")
    is_input: bool = Field(False, description="Whether this is input example")
    is_output: bool = Field(False, description="Whether this is expected output")
    
    @field_validator('code')
    def validate_code_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Code cannot be empty")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "def add(a: int, b: int) -> int:\n    return a + b",
                "language": "python",
                "description": "Simple addition function"
            }
        }
    )


class CodeRequest(BaseModel):
    """Request for code generation."""
    
    prompt: str = Field(description="Natural language description of what to generate")
    config: GenerationConfig = Field(default_factory=GenerationConfig, description="Generation configuration")
    examples: List[CodeExample] = Field(default_factory=list, description="Example code for context")
    context: Optional[str] = Field(None, description="Additional context or requirements")
    constraints: List[str] = Field(default_factory=list, description="Specific constraints or requirements")
    previous_attempt: Optional[str] = Field(None, description="Previous code attempt to improve upon")
    
    @field_validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError("Prompt cannot be empty")
        if len(v) < 10:
            raise ValueError("Prompt too short - please provide more detail")
        if len(v) > 2000:
            raise ValueError("Prompt too long - please be more concise")
        return v.strip()
    
    @field_validator('constraints')
    def validate_constraints(cls, v):
        if len(v) > 10:
            raise ValueError("Too many constraints - maximum 10 allowed")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "Create a Python function to calculate the factorial of a number",
                "config": {
                    "language": "python",
                    "include_tests": True
                },
                "constraints": [
                    "Use recursion",
                    "Handle negative numbers"
                ]
            }
        }
    )


class GeneratedCode(BaseModel):
    """Generated code with metadata."""
    
    code: str = Field(description="The generated code")
    language: Language = Field(description="Programming language")
    generator_agent: str = Field(description="Agent that generated this code")
    generation_time_ms: int = Field(description="Time taken to generate")
    quality_metrics: Optional[QualityMetrics] = Field(None, description="Quality metrics")
    confidence_score: float = Field(ge=0, le=1, description="Confidence in the generation")
    explanation: Optional[str] = Field(None, description="Explanation of the approach")
    
    @field_validator('code')
    def validate_code_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Generated code cannot be empty")
        return v
    
    @property
    def line_count(self) -> int:
        """Get number of lines in the code."""
        return len(self.code.split('\n'))
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
                "language": "python",
                "generator_agent": "gemini_agent",
                "generation_time_ms": 450,
                "confidence_score": 0.95
            }
        }
    )


class CodeResponse(AgentResponse):
    """Response containing generated code."""
    
    generated_code: GeneratedCode = Field(description="The generated code and metadata")
    alternatives: List[GeneratedCode] = Field(default_factory=list, description="Alternative implementations")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    warnings: List[str] = Field(default_factory=list, description="Any warnings about the code")
    
    @property
    def primary_code(self) -> str:
        """Get the primary generated code."""
        return self.generated_code.code
    
    @property
    def best_quality_code(self) -> GeneratedCode:
        """Get the highest quality code (primary or alternative)."""
        all_codes = [self.generated_code] + self.alternatives
        
        # Sort by quality metrics if available, otherwise by confidence
        def sort_key(gc: GeneratedCode):
            if gc.quality_metrics:
                return gc.quality_metrics.overall_score
            return gc.confidence_score
        
        return max(all_codes, key=sort_key)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "Code generated successfully",
                "agent_name": "gemini_agent",
                "generated_code": {
                    "code": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
                    "language": "python",
                    "generator_agent": "gemini_agent",
                    "generation_time_ms": 300,
                    "confidence_score": 0.9
                },
                "suggestions": ["Consider adding input validation"]
            }
        }
    )


class ParallelGenerationResult(BaseModel):
    """Result from parallel code generation by multiple agents."""
    
    request: CodeRequest = Field(description="Original request")
    results: Dict[str, GeneratedCode] = Field(description="Results keyed by agent name")
    selected_agent: str = Field(description="Agent whose code was selected")
    selection_reason: str = Field(description="Reason for selection")
    comparison_metrics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, 
        description="Comparison metrics for each agent"
    )
    total_time_ms: int = Field(description="Total generation time")
    
    @property
    def selected_code(self) -> GeneratedCode:
        """Get the selected code."""
        return self.results[self.selected_agent]
    
    @property
    def all_codes(self) -> List[GeneratedCode]:
        """Get all generated codes."""
        return list(self.results.values())
    
    @field_validator('selected_agent')
    def validate_selected_agent(cls, v, info):
        results = info.data.get('results', {})
        if v not in results:
            raise ValueError(f"Selected agent '{v}' not in results")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request": {"prompt": "Create a sorting function"},
                "results": {
                    "gemini_agent": {"code": "def sort(arr): return sorted(arr)", "language": "python"},
                    "gpt4_agent": {"code": "def sort(arr):\n    # Quick sort implementation\n    ...", "language": "python"}
                },
                "selected_agent": "gpt4_agent",
                "selection_reason": "More comprehensive implementation",
                "total_time_ms": 1200
            }
        }
    )


class CodeGenerationBatch(BaseModel):
    """Batch of code generation requests."""
    
    requests: List[CodeRequest] = Field(description="List of generation requests")
    common_config: Optional[GenerationConfig] = Field(None, description="Common config for all requests")
    parallel_execution: bool = Field(True, description="Execute in parallel")
    
    @field_validator('requests')
    def validate_requests(cls, v):
        if not v:
            raise ValueError("Requests list cannot be empty")
        if len(v) > 20:
            raise ValueError("Batch size cannot exceed 20 requests")
        return v
    
    def apply_common_config(self):
        """Apply common config to all requests if specified."""
        if self.common_config:
            for request in self.requests:
                # Merge common config with request config
                for field, value in self.common_config.dict(exclude_unset=True).items():
                    setattr(request.config, field, value)