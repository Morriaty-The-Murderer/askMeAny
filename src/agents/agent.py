"""NL2SQL agent implementation.

This module implements the main NL2SQLAgent class supporting multiple model backends,
connection pooling, request queueing and async processing capabilities.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from src.agents.base import BaseAgent, GenerationResult, ValidationResult
from src.agents.models import (
    BaseModel,
    OpenAIModel,
    TransformersModel,
    ModelError
)
from src.agents.config import (
    BaseConfig,
    YAMLConfig,
    DBConfig,
    ConfigError,
    create_config
)
from src.components.error_handler import NL2SQLError

logger = logging.getLogger(__name__)


class AgentError(NL2SQLError):
    """Base exception for agent-related errors."""
    pass


class RequestQueue:
    """Queue system for managing concurrent requests."""
    
    def __init__(self, max_size: int = 100):
        """Initialize request queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.queue = asyncio.Queue(maxsize=max_size)
        self.processing = {}
        self.results = {}
        
    async def add_request(
        self,
        request_id: str,
        text: str,
        schema: Dict[str, Any],
        priority: int = 0
    ):
        """Add request to queue.
        
        Args:
            request_id: Unique request identifier
            text: Input text
            schema: Database schema
            priority: Request priority (higher = more important)
        """
        await self.queue.put((priority, request_id, text, schema))
        self.processing[request_id] = datetime.now()
        
    def get_result(self, request_id: str) -> Optional[GenerationResult]:
        """Get result for request if available.
        
        Args:
            request_id: Request identifier
            
        Returns:
            GenerationResult if available, None otherwise
        """
        return self.results.get(request_id)


class NL2SQLAgent(BaseAgent):
    """Main agent implementation for NL2SQL conversion."""

    def __init__(
        self,
        config_type: str = "yaml",
        config_path: Optional[str] = None,
        db_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize NL2SQL agent.
        
        Args:
            config_type: Type of config to use ('yaml' or 'db')
            config_path: Path to YAML config file
            db_config: Database configuration dictionary
        """
        # Initialize config
        config_args = {"path": config_path} if config_path else {"db_config": db_config}
        self.config_manager = create_config(config_type, **config_args)
        config = self.config_manager.load()
        super().__init__(config)
        
        # Initialize components
        self.model = self._create_model()
        self.request_queue = RequestQueue(
            max_size=self.config.get("queue_size", 100)
        )
        
        # Start background task
        self.processing_task = asyncio.create_task(self._process_queue())
        
    def _create_model(self) -> BaseModel:
        """Create model instance based on configuration.
        
        Returns:
            Model instance
        """
        model_config = self.config.get("model", {})
        model_type = model_config.get("type", "openai")
        
        if model_type == "openai":
            return OpenAIModel(model_config)
        elif model_type == "transformers":
            return TransformersModel(model_config)
        else:
            raise AgentError(f"Unknown model type: {model_type}")

    async def generate(
        self,
        text: str,
        schema: Dict[str, Any],
        **kwargs
    ) -> GenerationResult:
        """Generate SQL from natural language asynchronously.
        
        Args:
            text: Natural language input
            schema: Database schema
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with generated query
        """
        try:
            # Add request to queue
            request_id = f"{datetime.now().timestamp()}"
            await self.request_queue.add_request(request_id, text, schema)
            
            # Wait for result
            while True:
                if result := self.request_queue.get_result(request_id):
                    return result
                await asyncio.sleep(0.1)
                
        except Exception as e:
            raise AgentError(f"Generation failed: {str(e)}")

    def validate(
        self,
        query: str,
        schema: Dict[str, Any],
        **kwargs
    ) -> ValidationResult:
        """Validate generated SQL query.
        
        Args:
            query: SQL query string
            schema: Database schema
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationResult with validation status
        """
        try:
            # Basic validation - check query structure and schema references
            errors = []
            warnings = []
            
            # Check query syntax using sqlparse
            import sqlparse
            parsed = sqlparse.parse(query)
            if not parsed or not parsed[0].tokens:
                errors.append("Invalid SQL syntax")
            
            # Check schema references
            tables = {t.lower() for t in schema.get("tables", {})}
            columns = {c.lower() for c in schema.get("columns", {})}
            
            # Extract table/column references and validate
            refs = self._extract_references(query)
            for table in refs["tables"]:
                if table.lower() not in tables:
                    errors.append(f"Unknown table: {table}")
                    
            for column in refs["columns"]:
                if column.lower() not in columns:
                    warnings.append(f"Potential unknown column: {column}")
                    
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            raise AgentError(f"Validation failed: {str(e)}")

    def _extract_references(self, query: str) -> Dict[str, List[str]]:
        """Extract table and column references from query.
        
        Args:
            query: SQL query string
            
        Returns:
            Dictionary with table and column references
        """
        # Basic extraction - could be improved with proper SQL parsing
        tables = []
        columns = []
        
        # Split into words and check for references
        words = query.replace("(", " ").replace(")", " ").split()
        
        for i, word in enumerate(words):
            if word.lower() == "from" and i < len(words) - 1:
                tables.append(words[i + 1].strip(","))
            elif word.lower() in ("select", "where", "group by", "order by"):
                # Next words are likely columns
                j = i + 1
                while j < len(words) and words[j].lower() not in ("from", "where"):
                    columns.extend(c.strip(",") for c in words[j].split("."))
                    j += 1
                    
        return {
            "tables": list(set(tables)),
            "columns": list(set(columns))
        }
                
    async def _process_queue(self):
        """Background task to process request queue."""
        while True:
            try:
                # Get next request
                priority, request_id, text, schema = await self.request_queue.queue.get()
                
                # Generate query
                prompt = self._create_prompt(text, schema)
                query = await self.model.generate(prompt)
                
                # Store result
                self.request_queue.results[request_id] = GenerationResult(
                    query=query,
                    confidence=0.8  # TODO: Implement proper confidence scoring
                )
                
                # Cleanup
                del self.request_queue.processing[request_id]
                self.request_queue.queue.task_done()
                
            except Exception as e:
                logger.error(f"Queue processing error: {str(e)}")
                await asyncio.sleep(1)
                
    def _create_prompt(self, text: str, schema: Dict[str, Any]) -> str:
        """Create model prompt from input and schema.
        
        Args:
            text: Natural language input
            schema: Database schema
            
        Returns:
            Formatted prompt string
        """
        schema_str = "\n".join(
            f"Table {table}: {', '.join(cols)}"
            for table, cols in schema.get("tables", {}).items()
        )
        
        return f"""Given the following database schema:

{schema_str}

Convert this natural language query to SQL:
{text}

SQL query:"""

    async def close(self):
        """Clean up resources."""
        if hasattr(self, "processing_task"):
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()