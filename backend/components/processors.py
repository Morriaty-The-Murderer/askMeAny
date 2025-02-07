"""NL2SQL system processor components.

This module contains the core processing classes for input handling, schema management,
query generation, validation and execution in the NL2SQL system.
"""

from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import sqlparse

from backend.config import DATABASE, MODEL_CONFIG, ERROR_MESSAGES
from backend.logger_conf import get_logger

logger = get_logger(__name__)


class InputProcessor:
    """Handles preprocessing and normalization of natural language input."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the input processor.
        
        Args:
            config: Optional configuration dictionary for processor settings
        """
        self.config = config or {}
        self.logger = get_logger(__name__)

    def normalize_text(self, text: str) -> str:
        """Normalize input text by removing extra whitespace and standardizing case.
        
        Args:
            text: Raw input text
            
        Returns:
            Normalized text string
        """
        return " ".join(text.lower().split())

    def process(self, input_text: str) -> Dict[str, Any]:
        """Process raw input text into structured format.
        
        Args:
            input_text: Raw natural language input
            
        Returns:
            Processed input metadata dictionary
        """
        normalized = self.normalize_text(input_text)
        return {
            "original": input_text,
            "normalized": normalized,
            "tokens": normalized.split(),
            "length": len(normalized)
        }


class SchemaManager:
    """Manages database schema information and related operations."""

    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        """Initialize schema manager.
        
        Args:
            db_config: Optional database configuration override
        """
        self.db_config = db_config or DATABASE
        self.connection = None
        self._schema_cache = {}
        self.logger = get_logger(__name__)

    def connect(self) -> None:
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(**self.db_config)
        except Exception as e:
            self.logger.error(ERROR_MESSAGES["db_connection"].format(error=str(e)))
            raise

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Retrieve schema information for specified table.
        
        Args:
            table_name: Name of database table
            
        Returns:
            Table schema metadata dictionary
        """
        if table_name in self._schema_cache:
            return self._schema_cache[table_name]

        if not self.connection:
            self.connect()

        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = %s
            """, (table_name,))
            columns = cursor.fetchall()

        schema = {
            "table_name": table_name,
            "columns": [{"name": col[0], "type": col[1], "nullable": col[2]} for col in columns]
        }
        self._schema_cache[table_name] = schema
        return schema


class QueryGenerator:
    """Generates SQL queries from processed natural language input."""

    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """Initialize query generator.
        
        Args:
            model_config: Optional model configuration override
        """
        self.model_config = model_config or MODEL_CONFIG
        self.logger = get_logger(__name__)

    def generate(self, processed_input: Dict[str, Any], schema: Dict[str, Any]) -> str:
        """Generate SQL query from processed input and schema.
        
        Args:
            processed_input: Processed natural language input
            schema: Database schema information
            
        Returns:
            Generated SQL query string
        """
        # Placeholder for actual generation logic
        # Would integrate with OpenAI API or other model here
        raise NotImplementedError("Query generation not implemented")


class QueryValidator:
    """Validates generated SQL queries for correctness and safety."""

    def __init__(self):
        """Initialize query validator."""
        self.logger = get_logger(__name__)

    def validate_syntax(self, query: str) -> bool:
        """Check if query has valid SQL syntax.
        
        Args:
            query: SQL query string to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            parsed = sqlparse.parse(query)
            return bool(parsed)
        except Exception as e:
            self.logger.error(f"Syntax validation error: {str(e)}")
            return False

    def validate_safety(self, query: str) -> Tuple[bool, str]:
        """Check query for potentially unsafe operations.
        
        Args:
            query: SQL query string to validate
            
        Returns:
            Tuple of (is_safe, message)
        """
        dangerous_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER"]
        query_upper = query.upper()

        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False, f"Query contains dangerous keyword: {keyword}"

        return True, "Query passed safety validation"


class QueryExecutor:
    """Executes validated SQL queries and handles results."""

    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        """Initialize query executor.
        
        Args:
            db_config: Optional database configuration override
        """
        self.db_config = db_config or DATABASE
        self.connection = None
        self.logger = get_logger(__name__)

    def connect(self) -> None:
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(**self.db_config)
        except Exception as e:
            self.logger.error(ERROR_MESSAGES["db_connection"].format(error=str(e)))
            raise

    def execute(self, query: str) -> Tuple[List[Tuple], List[str]]:
        """Execute validated SQL query.
        
        Args:
            query: Validated SQL query string
            
        Returns:
            Tuple of (results, column_names)
        """
        if not self.connection:
            self.connect()

        with self.connection.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            return results, column_names
