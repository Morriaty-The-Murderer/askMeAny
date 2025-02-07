"""
Configuration settings for NL2SQL application.

This module contains all configuration parameters including database settings,
model parameters, API credentials and data source URLs. Values are loaded from
environment variables where appropriate.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"

# Ensure required directories exist
for directory in [DATA_DIR, MODELS_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True)

# Database configurations
DATABASE = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "nl2sql_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "min_connections": 1,
    "max_connections": 10,
    "connection_timeout": 30,
}

# Model parameters
MODEL_CONFIG = {
    "model_name": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 150,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "max_retries": 3,
    "timeout": 30,
}

# Training parameters
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 10,
    "validation_split": 0.2,
    "early_stopping_patience": 3,
}

# Data source URLs
DATA_SOURCES = {
    "gdp": {
        "world_bank": "https://api.worldbank.org/v2/country/all/indicator/NY.GDP.MKTP.CD",
        "imf": "https://www.imf.org/external/datamapper/NGDP_RPCH@WEO/OEMDC/ADVEC/WEOWORLD",
    },
    "stock": {
        "alpha_vantage_base": "https://www.alphavantage.co/query",
        "yahoo_finance_base": "https://query1.finance.yahoo.com/v8/finance/chart/",
    }
}

# API Keys and credentials
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
    "world_bank": os.getenv("WORLD_BANK_API_KEY"),
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": "nl2sql.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": True,
        }
    },
}

# Cache settings
CACHE_CONFIG = {
    "type": "filesystem",
    "directory": str(CACHE_DIR),
    "expiration_time": 3600,  # 1 hour in seconds
    "max_size": 1e9,  # 1GB in bytes
}

# Error message templates
ERROR_MESSAGES = {
    "db_connection": "Failed to connect to database: {error}",
    "api_error": "API request failed: {error}",
    "model_loading": "Failed to load model: {error}",
    "invalid_input": "Invalid input format: {error}",
}

def get_db_uri():
    """Construct database URI from configuration."""
    return f"postgresql://{DATABASE['user']}:{DATABASE['password']}@{DATABASE['host']}:{DATABASE['port']}/{DATABASE['database']}"

def validate_config():
    """Validate critical configuration settings."""
    required_vars = [
        ("OPENAI_API_KEY", API_KEYS["openai"]),
        ("DB_PASSWORD", DATABASE["password"]),
    ]
    
    missing_vars = [var for var, value in required_vars if not value]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

def init_config():
    """Initialize and validate configuration."""
    validate_config()
    return True