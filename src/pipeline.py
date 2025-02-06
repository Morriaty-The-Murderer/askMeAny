"""NL2SQL system pipeline orchestration.

This module implements the main processing pipeline that coordinates component
interactions and handles the end-to-end flow of the NL2SQL system.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.components.processors import (
    InputProcessor,
    SchemaManager,
    QueryValidator,
    QueryExecutor
)
from src.components.error_handler import (
    ErrorHandler,
    NL2SQLError,
    InputProcessingError,
    SchemaError,
    QueryGenerationError,
    ValidationError,
    ExecutionError
)
from src.agents.agent import NL2SQLAgent
from src.config import DATABASE, MODEL_CONFIG, LOGGING_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for pipeline execution results."""
    
    success: bool
    query: Optional[str] = None
    results: Optional[List[Tuple]] = None
    columns: Optional[List[str]] = None
    error: Optional[NL2SQLError] = None
    metadata: Optional[Dict[str, Any]] = None


class Pipeline:
    """Main processing pipeline for NL2SQL system."""
    
    def __init__(
        self,
        db_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize pipeline and components.
        
        Args:
            db_config: Optional database configuration override
            model_config: Optional model configuration override
        """
        # Initialize components
        self.input_processor = InputProcessor()
        self.schema_manager = SchemaManager(db_config or DATABASE)
        self.agent = NL2SQLAgent(config_type='yaml', db_config=db_config or DATABASE)
        self.query_validator = QueryValidator()
        self.query_executor = QueryExecutor(db_config or DATABASE)
        self.error_handler = ErrorHandler()
        
        # Setup logging
        logging.config.dictConfig(LOGGING_CONFIG)
        self.logger = logging.getLogger(__name__)

    def _process_input(self, text: str) -> Dict[str, Any]:
        """Process natural language input.
        
        Args:
            text: Raw input text
            
        Returns:
            Processed input metadata
            
        Raises:
            InputProcessingError: If input processing fails
        """
        try:
            return self.input_processor.process(text)
        except Exception as e:
            raise InputProcessingError(f"Input processing failed: {str(e)}")

    def _get_schema(self, table_name: str) -> Dict[str, Any]:
        """Retrieve database schema information.
        
        Args:
            table_name: Target table name
            
        Returns:
            Table schema metadata
            
        Raises:
            SchemaError: If schema retrieval fails
        """
        try:
            return self.schema_manager.get_table_schema(table_name)
        except Exception as e:
            raise SchemaError(f"Schema retrieval failed: {str(e)}")

    async def _generate_query(self, processed_input: Dict[str, Any], schema: Dict[str, Any]) -> str:
        """Generate SQL query from processed input.
        
        Args:
            processed_input: Processed natural language input
            schema: Database schema information
            
        Returns:
            Generated SQL query
            
        Raises:
            QueryGenerationError: If query generation fails
        """
        try:
            result = await self.agent.generate(processed_input['normalized'], schema)
            return result.query
        except Exception as e:
            raise QueryGenerationError(f"Query generation failed: {str(e)}")

    def _validate_query(self, query: str) -> None:
        """Validate generated SQL query.
        
        Args:
            query: SQL query to validate
            
        Raises:
            ValidationError: If validation fails
        """
        if not self.query_validator.validate_syntax(query):
            raise ValidationError("Invalid SQL syntax")
        
        is_safe, message = self.query_validator.validate_safety(query)
        if not is_safe:
            raise ValidationError(f"Query safety check failed: {message}")

    def _execute_query(self, query: str) -> Tuple[List[Tuple], List[str]]:
        """Execute validated SQL query.
        
        Args:
            query: Validated SQL query
            
        Returns:
            Tuple of (results, column_names)
            
        Raises:
            ExecutionError: If query execution fails
        """
        try:
            return self.query_executor.execute(query)
        except Exception as e:
            raise ExecutionError(f"Query execution failed: {str(e)}")

    async def process(
        self, 
        input_text: str,
        table_name: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """Execute complete NL2SQL pipeline.
        
        Args:
            input_text: Natural language input text
            table_name: Target database table
            context: Optional additional context
            
        Returns:
            PipelineResult containing execution results or error
        """
        try:
            # Step 1: Process input
            processed_input = self._process_input(input_text)
            
            # Step 2: Get schema
            schema = self._get_schema(table_name)
            
            # Step 3: Generate query
            query = await self._generate_query(processed_input, schema)
            
            # Step 4: Validate query
            self._validate_query(query)
            
            # Step 5: Execute query
            results, columns = self._execute_query(query)
            
            return PipelineResult(
                success=True,
                query=query,
                results=results,
                columns=columns,
                metadata={
                    "input_metadata": processed_input,
                    "schema": schema,
                    "context": context
                }
            )
            
        except NL2SQLError as e:
            self.error_handler.log_error(e, context)
            return PipelineResult(success=False, error=e)
            
        except Exception as e:
            error = NL2SQLError(f"Unexpected error: {str(e)}")
            self.error_handler.log_error(error, context)
            return PipelineResult(success=False, error=error)

    def get_error_history(self) -> List[Dict[str, Any]]:
        """Retrieve error history.
        
        Returns:
            List of recorded errors
        """
        return self.error_handler.get_error_history()

    def clear_errors(self) -> None:
        """Clear error history."""
        self.error_handler.clear_errors()