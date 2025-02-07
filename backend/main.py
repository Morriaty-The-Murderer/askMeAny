#!/usr/bin/env python3
"""Main entry point for NL2SQL application.

This module serves as the primary entry point for the Natural Language to SQL
query conversion system. It handles the initialization of components and
orchestrates the processing flow.

Example:
    To run the application:
        $ python main.py
"""

import os
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from backend.config import DATABASE, MODEL_CONFIG
from backend.services.pipeline import Pipeline
from backend.components.error_handler import NL2SQLError
from backend.logger_conf import get_logger

def setup_environment():
    """Initialize application environment and configurations."""
    # Set up logging
    try:
        logger = get_logger(__name__)
        logger.info("Initializing application environment")
        
        # Initialize database connection
        pipeline = Pipeline(db_config=DATABASE, model_config=MODEL_CONFIG)
        return pipeline
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        sys.exit(1)

def main():
    """Main execution function."""
    logger = get_logger(__name__)
    
    try:
        # Initialize environment and pipeline
        pipeline = setup_environment()
        logger.info("Starting NL2SQL application")
        
        # Main application loop
        while True:
            try:
                # Get user input
                print("\nEnter natural language query (or 'quit' to exit):")
                text = input("> ").strip()
                
                # Check for commands
                if text.lower() == 'quit':
                    break
                elif text.lower() == 'hot_reload':
                    print("\nReloading pipeline configuration...")
                    pipeline = setup_environment()
                    print("Pipeline reloaded successfully")
                    continue
                    
                # Process through pipeline
                table_name = "users"  # TODO: Make this configurable
                result = pipeline.process(text, table_name)
                
                if result.success:
                    print(f"\nGenerated SQL: {result.query}")
                    print("\nResults:")
                    if result.columns:
                        print(", ".join(result.columns))
                    if result.results:
                        for row in result.results:
                            print(row)
                else:
                    print(f"\nError: {result.error.message}")
            except Exception as e:
                logger.error(f"Query processing error: {e}")
                print(f"\nError processing query: {e}")
                continue

    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())