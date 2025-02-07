"""NL2SQL error handling components.

This module provides error handling functionality for the NL2SQL system including
custom exceptions, error tracking, and logging integration.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.config import ERROR_MESSAGES

logger = logging.getLogger(__name__)


class NL2SQLError(Exception):
    """Base exception class for NL2SQL system errors."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        """Initialize base error.
        
        Args:
            message: Error message string
            error_code: Optional error code identifier
        """
        self.message = message
        self.error_code = error_code
        self.timestamp = datetime.now()
        super().__init__(self.message)


class InputProcessingError(NL2SQLError):
    """Exception raised for errors in input processing."""
    pass


class SchemaError(NL2SQLError):
    """Exception raised for database schema related errors."""
    pass


class QueryGenerationError(NL2SQLError):
    """Exception raised for errors during SQL query generation."""
    pass


class ValidationError(NL2SQLError):
    """Exception raised for query validation errors."""
    pass


class ExecutionError(NL2SQLError):
    """Exception raised for query execution errors."""
    pass


@dataclass
class ErrorRecord:
    """Data class for storing error information."""

    timestamp: datetime
    error_type: str
    message: str
    error_code: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class ErrorHandler:
    """Manages error handling and tracking for NL2SQL system."""

    def __init__(self):
        """Initialize error handler."""
        self.errors: List[ErrorRecord] = []
        self.logger = logging.getLogger(__name__)

    def log_error(self, error: NL2SQLError, context: Optional[Dict[str, Any]] = None) -> None:
        """Log error and store in error history.
        
        Args:
            error: NL2SQLError instance
            context: Optional context information dictionary
        """
        error_record = ErrorRecord(
            timestamp=error.timestamp,
            error_type=error.__class__.__name__,
            message=error.message,
            error_code=error.error_code,
            context=context
        )
        self.errors.append(error_record)
        self.logger.error(
            f"Error occurred: {error.message}",
            extra={"error_type": error.__class__.__name__, "error_code": error.error_code}
        )

    def get_error_history(self) -> List[ErrorRecord]:
        """Retrieve list of recorded errors.
        
        Returns:
            List of ErrorRecord instances
        """
        return self.errors

    def clear_errors(self) -> None:
        """Clear error history."""
        self.errors = []

    @staticmethod
    def format_error_message(template_key: str, **kwargs) -> str:
        """Format error message using template.
        
        Args:
            template_key: Key for error message template
            **kwargs: Format parameters for template
            
        Returns:
            Formatted error message string
        """
        template = ERROR_MESSAGES.get(template_key, "Unknown error occurred")
        return template.format(**kwargs)

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Handle and log any exception.
        
        Args:
            error: Exception instance
            context: Optional context information
        """
        if isinstance(error, NL2SQLError):
            self.log_error(error, context)
        else:
            wrapped_error = NL2SQLError(str(error))
            self.log_error(wrapped_error, context)
