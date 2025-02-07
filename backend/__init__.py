"""Python module for NL2SQL library

This package provides natural language to SQL conversion functionality with
key components organized into api and services layers:

- api: Web interface and request handling
- services: Core business logic and agent coordination
- components: Shared utilities and error handling

Copyright (c) 2023
All rights reserved.
"""

from backend.services.pipeline import Pipeline
from backend.components.error_handler import NL2SQLError
from backend.api import __version__ as api_version
from backend.logger_conf import logger, get_logger

__version__ = '0.1.0'
__author__ = 'The Author'
__all__ = [
    'Agent',
    'AgentConfig',
    'AgentQueue',
    'DataProcessor',
    'NL2SQLError',
    'Pipeline',
    'logger',
    'get_logger',
]

# Version compatibility check
API_MIN_VERSION = '0.1.0'
assert api_version >= API_MIN_VERSION, f'API version {api_version} is lower than minimum required {API_MIN_VERSION}'

# Initialize logger for package level
logger = get_logger(__name__)
