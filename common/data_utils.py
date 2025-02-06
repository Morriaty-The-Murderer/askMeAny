"""Core data handling utilities for NL2SQL system.

This module provides utility functions for loading, validating and extracting metadata
from data files used in the NL2SQL system.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union

import pandas as pd
from pandas.errors import EmptyDataError
from loguru import logger


def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV file into a pandas DataFrame.
    
    Args:
        filepath: Path to the CSV file

    Returns:
        pandas DataFrame containing the loaded data

    Raises:
        FileNotFoundError: If the specified file does not exist
        EmptyDataError: If the CSV file is empty
        ValueError: If the file is not a valid CSV
    """
    file_path = Path(filepath)
    
    if not file_path.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"No file found at {filepath}")
        
    try:
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        if df.empty:
            logger.error("Loaded CSV file is empty")
            raise EmptyDataError("The CSV file contains no data")
            
        logger.info(f"Successfully loaded {len(df)} rows of data")
        return df
        
    except pd.errors.ParserError as e:
        logger.exception("Error parsing CSV file")
        raise ValueError(f"Invalid CSV file format: {e}")


def validate_data_schema(df: pd.DataFrame) -> bool:
    """Validate the schema of the input DataFrame.
    
    Checks for required columns, data types, and data quality issues.
    
    Args:
        df: pandas DataFrame to validate

    Returns:
        bool: True if validation passes, False otherwise
    """
    required_columns = {'id', 'name', 'department', 'salary', 'hire_date'}
    
    # Check required columns
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logger.error(f"Missing required columns: {missing}")
        return False
        
    # Check for null values
    null_counts = df[list(required_columns)].isnull().sum()
    if null_counts.any():
        logger.warning(f"Found null values in columns: \n{null_counts[null_counts > 0]}")
        return False
        
    # Validate data types
    try:
        df['id'] = pd.to_numeric(df['id'])
        df['salary'] = pd.to_numeric(df['salary'])
        df['hire_date'] = pd.to_datetime(df['hire_date'])
    except (ValueError, TypeError) as e:
        logger.error(f"Data type validation failed: {e}")
        return False
        
    logger.info("Data schema validation passed")
    return True


def get_column_metadata(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Extract metadata about DataFrame columns.
    
    Args:
        df: pandas DataFrame to analyze

    Returns:
        Dictionary containing metadata for each column:
        {
            'column_name': {
                'dtype': str,
                'null_count': int,
                'unique_values': int,
                'sample_values': list
            }
        }
    """
    metadata = {}
    
    try:
        for column in df.columns:
            metadata[column] = {
                'dtype': str(df[column].dtype),
                'null_count': int(df[column].isnull().sum()),
                'unique_values': int(df[column].nunique()),
                'sample_values': df[column].dropna().head(5).tolist()
            }
            
        logger.info("Successfully extracted column metadata")
        return metadata
        
    except Exception as e:
        logger.exception("Error extracting column metadata")
        raise ValueError(f"Failed to extract column metadata: {e}")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to a consistent format.
    
    Converts column names to lowercase, replaces spaces with underscores,
    and removes special characters.
    
    Args:
        df: pandas DataFrame to normalize

    Returns:
        DataFrame with normalized column names
    """
    try:
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(' ', '_')
        df.columns = df.columns.str.replace('[^a-z0-9_]', '', regex=True)
        
        logger.info("Successfully normalized column names")
        return df
        
    except Exception as e:
        logger.exception("Error normalizing column names")
        raise ValueError(f"Failed to normalize column names: {e}")