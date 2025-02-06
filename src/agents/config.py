"""NL2SQL system configuration management.

This module provides configuration management classes for both local YAML-based
development and database-backed production environments. Includes connection pooling
and validation functionality.
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import yaml
import psycopg2
from psycopg2 import pool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "default_provider": "openai",
    "fallback_providers": ["anthropic", "google"],
    "openai": {
        "enabled": True,
        "models": {
            "default": "gpt-4",
            "alternatives": ["gpt-4-turbo", "gpt-3.5-turbo"]
        },
        "parameters": {
            "max_tokens": 2048,
            "temperature": 0.7,
            "timeout": 30,
            "retries": 3
        }
    },
    "anthropic": {
        "enabled": True,
        "models": {
            "default": "claude-2",
            "alternatives": ["claude-instant"]
        },
        "parameters": {
            "max_tokens": 2048,
            "temperature": 0.7
        }
    },
    "google": {
        "enabled": True,
        "models": {
            "default": "gemini-pro",
            "alternatives": ["gemini-pro-vision"]
        },
        "parameters": {
            "max_tokens": 2048,
            "temperature": 0.7
        }
    }
}

class ConfigError(Exception):
    """Base exception for configuration errors."""
    pass

class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""
    pass

class ConfigLoadError(ConfigError):
    """Raised when loading configuration fails."""
    pass

class RetryStrategy(BaseModel):
    """Retry configuration for API requests."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    retryable_errors: List[str] = Field(default_factory=lambda: ['timeout', 'rate_limit'])

class ModelMapping(BaseModel):
    """Model mapping configuration."""
    default: str
    alternatives: List[str] = Field(default_factory=list)
    version_map: Dict[str, str] = Field(default_factory=dict)
    capabilities: List[str] = Field(default_factory=list)

class ProviderConfig(BaseModel):
    """Provider-specific configuration schema."""
    enabled: bool = True
    api_key: Optional[str] = None
    api_base: str
    api_version: Optional[str] = None
    models: ModelMapping
    parameters: Dict[str, Any]
    retry_strategy: RetryStrategy = Field(default_factory=RetryStrategy)

class ConfigSchema(BaseModel):
    """Pydantic model for config validation."""
    default_provider: str = "openai"
    fallback_providers: List[str] = Field(default_factory=list)
    openai: ProviderConfig
    anthropic: Optional[ProviderConfig] = None
    google: Optional[ProviderConfig] = None

class BaseConfig(ABC):
    """Abstract base class for configuration implementations."""
    
    def __init__(self):
        """Initialize base config."""
        self.logger = logging.getLogger(__name__)
        self._config: Dict[str, Any] = {}
        self._pool: Optional[pool.SimpleConnectionPool] = None
        
    def get_active_providers(self) -> List[str]:
        """Get list of enabled providers.
        
        Returns:
            List of enabled provider names
        """
        providers = []
        for provider in ['openai', 'anthropic', 'google']:
            if self._config.get(provider, {}).get('enabled', False):
                providers.append(provider)
        return providers
        
    def get_provider_config(self, provider: str) -> ProviderConfig:
        """Get configuration for specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Provider configuration object
            
        Raises:
            ConfigError: If provider not found or disabled
        """
        if provider not in self._config:
            raise ConfigError(f"Provider '{provider}' not found in configuration")
            
        if not self._config[provider].get('enabled', False):
            raise ConfigError(f"Provider '{provider}' is disabled")
            
        return ProviderConfig(**self._config[provider])

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load configuration data."""
        raise NotImplementedError
        
def validate(self, config: Dict[str, Any]) -> None:
        """Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigValidationError: If validation fails
        """
        try:
            schema = ConfigSchema(**config)
            
            # Validate provider configuration
            if not config.get(schema.default_provider, {}).get('enabled', False):
                raise ConfigValidationError(f"Default provider '{schema.default_provider}' is not enabled")
                
            for provider in schema.fallback_providers:
                if provider not in config:
                    raise ConfigValidationError(f"Fallback provider '{provider}' not configured")
                if not config[provider].get('enabled', False):
                    raise ConfigValidationError(f"Fallback provider '{provider}' is not enabled")
                    
            # Validate API endpoints
            for provider in [schema.default_provider] + schema.fallback_providers:
                if provider in config and config[provider].get('enabled'):
                    if not config[provider].get('api_base'):
                        raise ConfigValidationError(f"Missing API endpoint for provider '{provider}'")
        except Exception as e:
            raise ConfigValidationError(f"Config validation failed: {e}")

    def get_pool(self) -> pool.SimpleConnectionPool:
        """Get or create database connection pool.
        
        Returns:
            Database connection pool instance
        """
        if not self._pool:
            db_config = self._config["database"]
            self._pool = pool.SimpleConnectionPool(
                db_config["pool_min"],
                db_config["pool_max"],
                host=db_config.get("host", "localhost"),
                port=db_config.get("port", 5432),
                dbname=db_config["name"],
                user=db_config["user"],
                password=db_config["password"],
                connect_timeout=db_config["connect_timeout"]
            )
        return self._pool

class YAMLConfig(BaseConfig):
    """YAML-based configuration for local development."""
    
    def __init__(self, path: str):
        """Initialize YAML config.
        
        Args:
            path: Path to YAML config file
        """
        super().__init__()
        self.path = path

    def load(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigLoadError: If loading fails
        """
        try:
            with open(self.path) as f:
                config = yaml.safe_load(f)
            self.validate(config)
            self._config = config
            return config
        except Exception as e:
            raise ConfigLoadError(f"Failed to load YAML config: {e}")

class DBConfig(BaseConfig):
    """Database-backed configuration for production."""
    
    def __init__(self, db_config: Dict[str, Any]):
        """Initialize database config.
        
        Args:
            db_config: Database connection parameters
        """
        super().__init__()
        self.db_config = db_config

    def load(self) -> Dict[str, Any]:
        """Load configuration from database.
        
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigLoadError: If loading fails
        """
        try:
            with self.get_pool().getconn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT config FROM system_config WHERE active = true")
                    config = cur.fetchone()[0]
            self.validate(config)
            self._config = config
            return config
        except Exception as e:
            raise ConfigLoadError(f"Failed to load DB config: {e}")

def create_config(config_type: str, **kwargs) -> BaseConfig:
    """Factory function to create config instance.
    
    Args:
        config_type: Type of config to create ('yaml' or 'db')
        **kwargs: Additional arguments for config constructor
        
    Returns:
        Configuration instance
    """
    if config_type == "yaml":
        return YAMLConfig(kwargs["path"])
    elif config_type == "db":
        return DBConfig(kwargs["db_config"])
    else:
        raise ValueError(f"Unknown config type: {config_type}")