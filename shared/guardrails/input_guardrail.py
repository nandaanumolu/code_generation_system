"""
Input guardrail for validating and sanitizing user inputs.
Prevents harmful, inappropriate, or malicious requests.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from ..schemas import CodeRequest


logger = logging.getLogger(__name__)


class InputRiskLevel(str, Enum):
    """Risk levels for input validation."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    BLOCKED = "blocked"


@dataclass
class InputValidationResult:
    """Result of input validation."""
    is_valid: bool
    risk_level: InputRiskLevel
    issues: List[str]
    sanitized_input: Optional[str] = None
    modifications: List[str] = None
    
    def __post_init__(self):
        if self.modifications is None:
            self.modifications = []


class InputGuardrail:
    """
    Validates and sanitizes user inputs for code generation.
    Checks for harmful patterns, inappropriate content, and security risks.
    """
    
    # Harmful patterns that should be blocked
    HARMFUL_PATTERNS = [
        # Malware/virus related
        (r'\b(malware|virus|trojan|worm|ransomware|keylogger)\b', "malware-related content"),
        (r'\b(hack|exploit|backdoor|rootkit)\b', "hacking/exploitation content"),
        
        # System damage
        (r'rm\s+-rf\s+/', "dangerous file deletion command"),
        (r':(){ :|:& };:', "fork bomb pattern"),
        (r'dd\s+if=/dev/(zero|random)', "disk overwrite command"),
        
        # Credential theft
        (r'\b(steal|grab|harvest)\s+(password|credential|token|key)', "credential theft intent"),
        (r'\b(phishing|spoof|fake)\s+(site|page|login)', "phishing-related content"),
        
        # Illegal activities
        (r'\b(crack|bypass|circumvent)\s+(license|drm|protection)', "circumvention request"),
        (r'\b(illegal|pirate|torrent)\s+(download|software|content)', "illegal content request"),
    ]
    
    # Suspicious patterns that raise the risk level
    SUSPICIOUS_PATTERNS = [
        # Network operations
        (r'\b(socket|netcat|nc|telnet)\b', "network operations"),
        (r'\b(download|wget|curl)\s+http', "remote download"),
        
        # System operations
        (r'\b(subprocess|os\.system|exec|eval)\b', "system execution"),
        (r'\b(admin|root|sudo|privilege)\b', "privilege escalation"),
        
        # Data operations
        (r'\b(encrypt|decrypt|cipher|hash)\b', "cryptographic operations"),
        (r'\b(scrape|crawl|harvest)\s+(data|website)', "data scraping"),
        
        # Obfuscation
        (r'\b(obfuscate|hide|conceal|encode)\b', "obfuscation request"),
        (r'base64|hex|binary', "encoding operations"),
    ]
    
    # Inappropriate content patterns
    INAPPROPRIATE_PATTERNS = [
        (r'\b(adult|explicit|nsfw)\b', "adult content"),
        (r'\b(hate|racist|discriminat)', "hate speech"),
        (r'\b(violence|weapon|bomb)\b', "violent content"),
        (r'\b(drug|narcotic|illegal substance)\b', "drug-related content"),
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional configuration."""
        self.config = config or {}
        self.strict_mode = self.config.get("strict_mode", False)
        self.log_violations = self.config.get("log_violations", True)
        self.max_input_length = self.config.get("max_input_length", 5000)
        self.min_input_length = self.config.get("min_input_length", 10)
    
    def validate(self, input_text: str) -> InputValidationResult:
        """
        Validate input text for safety and appropriateness.
        
        Args:
            input_text: The input text to validate
            
        Returns:
            Validation result with risk assessment
        """
        issues = []
        risk_level = InputRiskLevel.SAFE
        
        # Check length
        if len(input_text) < self.min_input_length:
            issues.append(f"Input too short (minimum {self.min_input_length} characters)")
            return InputValidationResult(
                is_valid=False,
                risk_level=InputRiskLevel.BLOCKED,
                issues=issues
            )
        
        if len(input_text) > self.max_input_length:
            issues.append(f"Input too long (maximum {self.max_input_length} characters)")
            input_text = input_text[:self.max_input_length]
        
        # Check for harmful patterns (immediate block)
        for pattern, description in self.HARMFUL_PATTERNS:
            if re.search(pattern, input_text, re.IGNORECASE):
                issues.append(f"Blocked: {description}")
                risk_level = InputRiskLevel.BLOCKED
                
                if self.log_violations:
                    logger.warning(f"Blocked input containing: {description}")
                
                return InputValidationResult(
                    is_valid=False,
                    risk_level=risk_level,
                    issues=issues
                )
        
        # Check for inappropriate content
        for pattern, description in self.INAPPROPRIATE_PATTERNS:
            if re.search(pattern, input_text, re.IGNORECASE):
                issues.append(f"Inappropriate: {description}")
                risk_level = max(risk_level, InputRiskLevel.HIGH)
        
        # Check for suspicious patterns
        suspicious_count = 0
        for pattern, description in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, input_text, re.IGNORECASE):
                issues.append(f"Suspicious: {description}")
                suspicious_count += 1
        
        # Adjust risk level based on suspicious patterns
        if suspicious_count >= 3:
            risk_level = max(risk_level, InputRiskLevel.HIGH)
        elif suspicious_count >= 1:
            risk_level = max(risk_level, InputRiskLevel.MEDIUM)
        
        # Sanitize if needed
        sanitized_input, modifications = self._sanitize_input(input_text)
        
        # In strict mode, block HIGH risk
        if self.strict_mode and risk_level == InputRiskLevel.HIGH:
            return InputValidationResult(
                is_valid=False,
                risk_level=risk_level,
                issues=issues,
                sanitized_input=sanitized_input,
                modifications=modifications
            )
        
        return InputValidationResult(
            is_valid=risk_level != InputRiskLevel.BLOCKED,
            risk_level=risk_level,
            issues=issues,
            sanitized_input=sanitized_input,
            modifications=modifications
        )
    
    def validate_code_request(self, request: CodeRequest) -> InputValidationResult:
        """
        Validate a CodeRequest object.
        
        Args:
            request: The code generation request
            
        Returns:
            Validation result
        """
        # Validate prompt
        prompt_result = self.validate(request.prompt)
        
        # Check constraints
        for constraint in request.constraints:
            constraint_result = self.validate(constraint)
            if not constraint_result.is_valid:
                prompt_result.issues.extend(constraint_result.issues)
                prompt_result.risk_level = max(prompt_result.risk_level, constraint_result.risk_level)
        
        # Check context if provided
        if request.context:
            context_result = self.validate(request.context)
            if not context_result.is_valid:
                prompt_result.issues.extend(context_result.issues)
                prompt_result.risk_level = max(prompt_result.risk_level, context_result.risk_level)
        
        return prompt_result
    
    def _sanitize_input(self, text: str) -> Tuple[str, List[str]]:
        """
        Sanitize input text by removing or modifying problematic content.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Tuple of (sanitized_text, list_of_modifications)
        """
        modifications = []
        sanitized = text
        
        # Remove potential command injection characters
        dangerous_chars = ['`', '$', '\\', '&&', '||', ';', '|', '>', '<']
        for char in dangerous_chars:
            if char in sanitized:
                sanitized = sanitized.replace(char, '')
                modifications.append(f"Removed '{char}' characters")
        
        # Remove potential path traversal
        if '../' in sanitized or '..\\' in sanitized:
            sanitized = sanitized.replace('../', '').replace('..\\', '')
            modifications.append("Removed path traversal patterns")
        
        # Limit consecutive whitespace
        if '  ' in sanitized:
            sanitized = re.sub(r'\s+', ' ', sanitized)
            modifications.append("Normalized whitespace")
        
        # Remove null bytes
        if '\x00' in sanitized:
            sanitized = sanitized.replace('\x00', '')
            modifications.append("Removed null bytes")
        
        return sanitized.strip(), modifications


# Convenience functions

def validate_input_request(
    input_text: str,
    strict: bool = False
) -> InputValidationResult:
    """
    Validate an input request using default guardrail.
    
    Args:
        input_text: Text to validate
        strict: Whether to use strict mode
        
    Returns:
        Validation result
    """
    guardrail = InputGuardrail({"strict_mode": strict})
    return guardrail.validate(input_text)


def check_prompt_safety(prompt: str) -> bool:
    """
    Quick check if a prompt is safe.
    
    Args:
        prompt: Prompt to check
        
    Returns:
        True if safe, False otherwise
    """
    result = validate_input_request(prompt)
    return result.is_valid and result.risk_level in [InputRiskLevel.SAFE, InputRiskLevel.LOW]


def sanitize_input(text: str) -> str:
    """
    Sanitize input text using default guardrail.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    guardrail = InputGuardrail()
    result = guardrail.validate(text)
    return result.sanitized_input or text