"""Validation utilities for input and output safety."""

import re
from typing import Tuple, Optional

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class InputValidator:
    """Validates user inputs for safety and quality."""

    # Safety patterns to detect
    UNSAFE_PATTERNS = [
        r"exec\s*\(",
        r"eval\s*\(",
        r"__import__",
        r"subprocess",
        r"os\.system",
    ]

    # Minimum/maximum constraints
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 2000
    MIN_FILE_SIZE = 1  # bytes
    MAX_FILE_SIZE = 104857600  # 100 MB

    @classmethod
    def validate_query(cls, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user query for safety and quality.

        Args:
            query: User query string

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query:
            return False, "Query cannot be empty"

        query = query.strip()

        if len(query) < cls.MIN_QUERY_LENGTH:
            return False, f"Query must be at least {cls.MIN_QUERY_LENGTH} characters"

        if len(query) > cls.MAX_QUERY_LENGTH:
            return False, f"Query must not exceed {cls.MAX_QUERY_LENGTH} characters"

        # Check for unsafe patterns
        for pattern in cls.UNSAFE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                logger.warning("Unsafe query pattern detected", pattern=pattern)
                return False, "Query contains unsafe patterns"

        # Check for SQL injection patterns
        if re.search(
            r"(UNION|SELECT|DELETE|INSERT|UPDATE|DROP)\s", query, re.IGNORECASE
        ):
            logger.warning("SQL injection pattern detected")
            return False, "Query contains potentially harmful patterns"

        return True, None

    @classmethod
    def validate_filename(cls, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Validate filename for safety.

        Args:
            filename: Filename to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not filename:
            return False, "Filename cannot be empty"

        # Check for path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            return False, "Filename contains invalid characters"

        # Check for very long filenames
        if len(filename) > 255:
            return False, "Filename is too long"

        return True, None

    @classmethod
    def validate_file_size(cls, size_bytes: int) -> Tuple[bool, Optional[str]]:
        """
        Validate file size.

        Args:
            size_bytes: File size in bytes

        Returns:
            Tuple of (is_valid, error_message)
        """
        if size_bytes < cls.MIN_FILE_SIZE:
            return False, "File is too small"

        if size_bytes > cls.MAX_FILE_SIZE:
            return (
                False,
                f"File exceeds maximum size of {cls.MAX_FILE_SIZE / (1024*1024):.0f} MB",
            )

        return True, None


class OutputSanitizer:
    """Sanitizes generated outputs for safety."""

    @staticmethod
    def sanitize_response(response: str) -> str:
        """
        Sanitize LLM response.

        Args:
            response: Generated response text

        Returns:
            Sanitized response
        """
        if not response:
            return response

        # Remove potentially harmful content
        sanitized = response

        # Remove code execution attempts
        dangerous_keywords = ["exec", "eval", "__import__", "subprocess", "os.system"]
        for keyword in dangerous_keywords:
            sanitized = re.sub(
                rf"\b{re.escape(keyword)}\s*\(",
                "[code_removed]",
                sanitized,
                flags=re.IGNORECASE,
            )

        return sanitized

    @staticmethod
    def truncate_response(response: str, max_length: int = 10000) -> str:
        """
        Truncate response to maximum length.

        Args:
            response: Response text
            max_length: Maximum length

        Returns:
            Truncated response
        """
        if len(response) <= max_length:
            return response

        return response[:max_length] + "...[truncated]"
