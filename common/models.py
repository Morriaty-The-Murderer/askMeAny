"""Core model classes for NL2SQL query generation and results handling.

This module contains the base classes for NL2SQL operations, including natural language
parsing, SQL generation, and query result management.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from sqlalchemy import text


class SQLQueryGenerator:
    """Generates SQL queries from natural language input.
    
    Handles the complete pipeline of parsing natural language questions,
    converting them to SQL queries, and validating the generated SQL.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize query generator with specified language model.
        
        Args:
            model_name: Name of the language model to use for query generation
        """
        self.model_name = model_name
        
    def parse_natural_language(self, text: str) -> Dict[str, Union[str, List[str]]]:
        """Parse natural language query into structured components.
        
        Args:
            text: Natural language query text
            
        Returns:
            Dictionary containing parsed query components:
            {
                'intent': str,
                'entities': List[str],
                'conditions': List[str],
                'aggregations': List[str]
            }
            
        Raises:
            ValueError: If input text is empty or cannot be parsed
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
            
        # TODO: Implement actual NLP parsing logic
        parsed = {
            'intent': '',
            'entities': [],
            'conditions': [],
            'aggregations': []
        }
        return parsed
        
    def generate_sql(self, parsed_query: Dict[str, Union[str, List[str]]]) -> str:
        """Generate SQL query from parsed components.
        
        Args:
            parsed_query: Dictionary containing parsed query components
            
        Returns:
            Generated SQL query string
            
        Raises:
            ValueError: If parsed query is invalid or missing required components
        """
        if not parsed_query:
            raise ValueError("Parsed query cannot be empty")
            
        # TODO: Implement SQL generation logic
        sql = "SELECT * FROM table"
        return sql
        
    def validate_sql(self, sql: str) -> bool:
        """Validate generated SQL query for syntax and safety.
        
        Args:
            sql: SQL query string to validate
            
        Returns:
            True if query is valid, False otherwise
        """
        if not sql.strip():
            return False
            
        try:
            # Basic syntax validation using SQLAlchemy
            text(sql)
            return True
        except Exception:
            return False


class QueryResult:
    """Handles and formats query execution results.
    
    Provides methods for storing, accessing, and formatting query results,
    including metadata about the query execution.
    """
    
    def __init__(self, 
                 query: str,
                 data: Optional[pd.DataFrame] = None,
                 error: Optional[str] = None):
        """Initialize query result container.
        
        Args:
            query: Executed SQL query
            data: Query result data (if successful)
            error: Error message (if query failed)
        """
        self.query = query
        self.data = data
        self.error = error
        self.execution_time: Optional[float] = None
        
    def to_dict(self) -> Dict[str, Union[str, pd.DataFrame, float, None]]:
        """Convert query result to dictionary format.
        
        Returns:
            Dictionary containing query result data and metadata
        """
        return {
            'query': self.query,
            'data': self.data,
            'error': self.error,
            'execution_time': self.execution_time
        }
        
    def is_successful(self) -> bool:
        """Check if query execution was successful.
        
        Returns:
            True if query executed successfully, False otherwise
        """
        return self.error is None and self.data is not None
        
    def get_row_count(self) -> int:
        """Get number of rows in result set.
        
        Returns:
            Number of rows in result data, or 0 if no data
        """
        if self.data is None:
            return 0
        return len(self.data)