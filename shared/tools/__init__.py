"""
Tools module for code generation system.
Provides various tools for code analysis, formatting, and processing.
"""

# Import from individual tool modules
from .code_analysis_tool import (
    analyze_code,
    extract_functions,
    estimate_complexity,
    validate_python_syntax,
    get_code_metrics
)

from .formatting_tools import (
    wrap_code_in_markdown,
    add_line_numbers,
    format_code_output,
    clean_code_string,
    indent_code
)

# TODO: Uncomment these imports as we implement the files
# from .code_execution_tool import (
#     execute_code_safely,
#     validate_code_safety,
#     create_sandbox_environment
# )

# from .search_tools import (
#     search_google,
#     search_documentation,
#     search_stack_overflow
# )

# from .mcp_ingestion_tool import (
#     ingest_to_memory,
#     prepare_for_ingestion,
#     validate_mcp_format
# )

# Expose main functions at package level
__all__ = [
    # Code analysis
    'analyze_code',
    'extract_functions',
    'estimate_complexity',
    'validate_python_syntax',
    'get_code_metrics',
    
    # Formatting
    'wrap_code_in_markdown',
    'add_line_numbers',
    'format_code_output',
    'clean_code_string',
    'indent_code',
    
    # TODO: Add these back as we implement the files
    # # Code execution
    # 'execute_code_safely',
    # 'validate_code_safety',
    # 'create_sandbox_environment',
    
    # # Search
    # 'search_google',
    # 'search_documentation',
    # 'search_stack_overflow',
    
    # # MCP ingestion
    # 'ingest_to_memory',
    # 'prepare_for_ingestion',
    # 'validate_mcp_format',
]