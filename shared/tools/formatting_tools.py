"""
Code formatting utilities for the code generation system.
Provides tools for formatting code output in various ways.
"""

import re
import textwrap
from typing import Optional, List, Dict, Any


def wrap_code_in_markdown(code: str, language: str = "python") -> str:
    """
    Wrap code in markdown code blocks for display.
    
    Args:
        code: Code to wrap
        language: Programming language for syntax highlighting
        
    Returns:
        Code wrapped in markdown blocks
    """
    # Clean up code first
    code = code.strip()
    
    # Ensure no triple backticks in the code itself
    if '```' in code:
        code = code.replace('```', '` ` `')
    
    # Handle empty code
    if not code:
        return f"```{language}\n# No code provided\n```"
    
    return f"```{language}\n{code}\n```"


def add_line_numbers(code: str, start_line: int = 1, highlight_lines: Optional[List[int]] = None) -> str:
    """
    Add line numbers to code for review purposes.
    
    Args:
        code: Code to add line numbers to
        start_line: Starting line number (default: 1)
        highlight_lines: List of line numbers to highlight with '>'
        
    Returns:
        Code with line numbers prefixed
    """
    if not code:
        return ""
    
    lines = code.split('\n')
    numbered_lines = []
    
    # Calculate width for line numbers
    end_line = start_line + len(lines) - 1
    width = len(str(end_line))
    
    highlight_lines = highlight_lines or []
    
    for i, line in enumerate(lines, start_line):
        # Add highlight marker if needed
        marker = '>' if i in highlight_lines else ' '
        numbered_lines.append(f"{i:>{width}}{marker}| {line}")
    
    return '\n'.join(numbered_lines)


def format_code_output(
    code: str,
    title: Optional[str] = None,
    language: str = "python",
    show_line_numbers: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format code output with optional title and metadata.
    
    Args:
        code: Code to format
        title: Optional title for the code block
        language: Programming language
        show_line_numbers: Whether to add line numbers
        metadata: Optional metadata to display
        
    Returns:
        Formatted code output
    """
    output_parts = []
    
    # Add title if provided
    if title:
        output_parts.append(f"## {title}")
        output_parts.append("")
    
    # Add metadata if provided
    if metadata:
        output_parts.append("**Metadata:**")
        for key, value in metadata.items():
            output_parts.append(f"- {key}: {value}")
        output_parts.append("")
    
    # Add code with optional line numbers
    if show_line_numbers:
        code_with_numbers = add_line_numbers(code)
        output_parts.append("```")
        output_parts.append(code_with_numbers)
        output_parts.append("```")
    else:
        output_parts.append(wrap_code_in_markdown(code, language))
    
    return '\n'.join(output_parts)


def clean_code_string(code: str, remove_comments: bool = False, remove_docstrings: bool = False) -> str:
    """
    Clean and normalize code string.
    
    Args:
        code: Code to clean
        remove_comments: Whether to remove comments
        remove_docstrings: Whether to remove docstrings
        
    Returns:
        Cleaned code
    """
    if not code:
        return ""
    
    # Remove leading/trailing whitespace
    code = code.strip()
    
    # Remove comments if requested
    if remove_comments:
        # Remove single-line comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    
    # Remove docstrings if requested
    if remove_docstrings:
        # Remove triple-quoted strings
        code = re.sub(r'"""[\s\S]*?"""', '', code)
        code = re.sub(r"'''[\s\S]*?'''", '', code)
    
    # Remove multiple blank lines
    code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
    
    # Ensure consistent line endings
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    
    return code.strip()


def indent_code(code: str, spaces: int = 4, skip_first_line: bool = False) -> str:
    """
    Indent code by specified number of spaces.
    
    Args:
        code: Code to indent
        spaces: Number of spaces to indent
        skip_first_line: Whether to skip indenting the first line
        
    Returns:
        Indented code
    """
    if not code:
        return ""
    
    indent = ' ' * spaces
    lines = code.split('\n')
    
    if skip_first_line and lines:
        # Don't indent first line
        indented_lines = [lines[0]]
        indented_lines.extend(indent + line if line.strip() else line 
                            for line in lines[1:])
    else:
        indented_lines = [indent + line if line.strip() else line 
                         for line in lines]
    
    return '\n'.join(indented_lines)


def extract_code_blocks(text: str, language: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Extract code blocks from markdown text.
    
    Args:
        text: Text containing markdown code blocks
        language: Optional language filter
        
    Returns:
        List of dictionaries with 'code' and 'language' keys
    """
    # Pattern to match code blocks with optional language
    pattern = r'```(\w*)\n(.*?)\n```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    blocks = []
    for lang, code in matches:
        # If language filter specified, only include matching blocks
        if language is None or lang == language:
            blocks.append({
                'language': lang or 'plaintext',
                'code': code.strip()
            })
    
    return blocks


def format_function_signature(
    name: str,
    params: List[str],
    return_type: Optional[str] = None,
    decorators: Optional[List[str]] = None,
    docstring: Optional[str] = None
) -> str:
    """
    Format a function signature with proper styling.
    
    Args:
        name: Function name
        params: List of parameter strings
        return_type: Optional return type annotation
        decorators: Optional list of decorator strings
        docstring: Optional docstring
        
    Returns:
        Formatted function signature
    """
    lines = []
    
    # Add decorators
    if decorators:
        for decorator in decorators:
            lines.append(f"@{decorator}")
    
    # Build function signature
    if len(params) <= 3 and sum(len(p) for p in params) < 50:
        # Single line for short parameter lists
        param_str = ", ".join(params)
        signature = f"def {name}({param_str})"
    else:
        # Multi-line for long parameter lists
        lines.append(f"def {name}(")
        for i, param in enumerate(params):
            comma = "," if i < len(params) - 1 else ""
            lines.append(f"    {param}{comma}")
        signature = "\n".join(lines) + "\n)"
        lines = [signature]
    
    # Add return type if specified
    if return_type:
        lines[-1] += f" -> {return_type}"
    
    lines[-1] += ":"
    
    # Add docstring if provided
    if docstring:
        lines.append(f'    """{docstring}"""')
    
    return '\n'.join(lines)


def create_code_diff(original: str, modified: str, context_lines: int = 3) -> str:
    """
    Create a simple diff representation between two code versions.
    
    Args:
        original: Original code
        modified: Modified code
        context_lines: Number of context lines to show
        
    Returns:
        Diff representation
    """
    original_lines = original.split('\n')
    modified_lines = modified.split('\n')
    
    diff_lines = []
    diff_lines.append("```diff")
    
    # Simple line-by-line comparison
    max_lines = max(len(original_lines), len(modified_lines))
    
    for i in range(max_lines):
        if i < len(original_lines) and i < len(modified_lines):
            if original_lines[i] != modified_lines[i]:
                diff_lines.append(f"- {original_lines[i]}")
                diff_lines.append(f"+ {modified_lines[i]}")
            else:
                diff_lines.append(f"  {original_lines[i]}")
        elif i < len(original_lines):
            diff_lines.append(f"- {original_lines[i]}")
        else:
            diff_lines.append(f"+ {modified_lines[i]}")
    
    diff_lines.append("```")
    
    return '\n'.join(diff_lines)


def truncate_code(code: str, max_lines: int = 50, message: str = "... (truncated)") -> str:
    """
    Truncate code to a maximum number of lines.
    
    Args:
        code: Code to truncate
        max_lines: Maximum number of lines
        message: Message to append if truncated
        
    Returns:
        Truncated code
    """
    lines = code.split('\n')
    
    if len(lines) <= max_lines:
        return code
    
    truncated_lines = lines[:max_lines]
    truncated_lines.append(message)
    
    return '\n'.join(truncated_lines)