"""NL2SQL base agent interface.

This module defines the base abstract class and interfaces that all NL2SQL
agents must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class GenerationResult:
    """Container for query generation results."""
    
    query: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass 
class ValidationResult:
    """Container for query validation results."""
    
    is_valid: bool
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """Abstract base class for NL2SQL agents.
    
    All NL2SQL agent implementations must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base agent.
        
        Args:
            config: Optional configuration dictionary for agent settings
        """
        self.config = config or {}
    
    @abstractmethod
    async def generate(
        self,
        text: str,
        schema: Dict[str, Any],
        **kwargs
    ) -> GenerationResult:
        """Generate SQL query from natural language input.
        
        Args:
            text: Natural language input text
            schema: Database schema information
            **kwargs: Additional keyword arguments
            
        Returns:
            GenerationResult containing the generated query and metadata
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError
        
    @abstractmethod
    def validate(
        self,
        query: str,
        schema: Dict[str, Any],
        **kwargs
    ) -> ValidationResult:
        """Validate generated SQL query.
        
        Args:
            query: Generated SQL query string
            schema: Database schema information
            **kwargs: Additional keyword arguments
            
        Returns:
            ValidationResult containing validation status and any errors/warnings
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        """Get agent name.
        
        Returns:
            String identifier for the agent
        """
        return self.__class__.__name__
    
    def __repr__(self) -> str:
        """Get string representation.
        
        Returns:
            String representation of agent instance
        """
        return f"{self.name}(config={self.config})"