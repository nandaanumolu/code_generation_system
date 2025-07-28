"""
Shared components for the code generation system.
This module provides common tools, schemas, guardrails, and utilities
used across all agents.
"""

# Version info
__version__ = "0.1.0"
__author__ = "Code Generation System Team"

# Make key components easily accessible
from . import tools
from . import memory
from . import schemas
from . import guardrails

# TODO: Import these as we implement them
# from . import state
# from . import utils

# Expose commonly used items at package level for convenience
from .tools import analyze_code, wrap_code_in_markdown, add_line_numbers
from .memory import get_memory_service, MemoryEntry
from .schemas import CodeRequest, CodeResponse, CodeReview
from .guardrails import validate_input_request, validate_output_code

__all__ = [
    # Modules
    'tools',
    'memory',
    'schemas',
    'guardrails',
    # TODO: Add these back as we implement them
    # 'state',
    # 'utils',
    
    # Commonly used functions
    'analyze_code',
    'wrap_code_in_markdown', 
    'add_line_numbers',
    'get_memory_service',
    'MemoryEntry',
    'CodeRequest',
    'CodeResponse',
    'CodeReview',
    'validate_input_request',
    'validate_output_code',
]