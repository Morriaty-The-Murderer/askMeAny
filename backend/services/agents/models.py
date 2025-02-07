"""Models for NL2SQL agent system.

This module implements model classes for different backends (OpenAI API and local
transformers) with common utilities for tokenization and text processing.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass

import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from backend.components.error_handler import NL2SQLError, QueryGenerationError
from backend.logger_conf import get_logger

logger = get_logger(__name__)


class ModelError(NL2SQLError):
    """Base error class for model-related errors."""
    pass


class TokenizationError(ModelError):
    """Error raised for tokenization failures."""
    pass


class ApiError(ModelError):
    """Error raised for API communication failures."""
    pass


class ModelLoadError(ModelError):
    """Error raised when model loading fails."""
    pass


@dataclass
class TokenizerResult:
    """Container for tokenization results."""
    
    tokens: List[str]
    token_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None


class BaseModel(ABC):
    """Abstract base class for model implementations."""

def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize base model.
        
        Args:
            config: Optional model configuration
        """
        self.config = config or {}
        self.logger = get_logger(__name__)

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        **kwargs
    ) -> str:
        """Generate text completion for given prompt.
        
        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text completion
        """
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, text: str) -> TokenizerResult:
        """Tokenize input text.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            TokenizerResult with tokens and ids
        """
        raise NotImplementedError


class BaseAPIModel(BaseModel):
    """Base class for API-based model implementations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize API model.
        
        Args:
            config: Optional API configuration 
        """
        super().__init__(config)
        self.provider = self.config.get("provider", "openai")
        self.api_key = self.config.get(f"{self.provider}_api_key")
        self.api_base = self.config.get(f"{self.provider}_api_base")
        self.api_version = self.config.get(f"{self.provider}_api_version")
        self.fallback_providers = self.config.get("fallback_providers", [])
        
        if not self.api_key:
            raise ModelError(f"{self.provider} API key not provided")
            
        self.model = self.config.get("model_name",
                                   self.config.get("models", {}).get("default"))
        
        # Initialize API clients
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None
        self.tokenizer = None
        
        self._setup_clients()

    def _setup_clients(self):
        """Initialize API clients for enabled providers."""
        try:
            if "openai" in [self.provider] + self.fallback_providers:
                import openai
                self.openai_client = openai.Client(
                    api_key=self.config.get("openai_api_key"),
                    base_url=self.config.get("openai_api_base"),
                    api_version=self.config.get("openai_api_version")
                )
            
            if "anthropic" in [self.provider] + self.fallback_providers:
                import anthropic
                self.anthropic_client = anthropic.Client(api_key=self.config.get("anthropic_api_key"))
            
            if "google" in [self.provider] + self.fallback_providers:
                import google.generativeai as genai
                genai.configure(api_key=self.config.get("google_api_key"))
                self.google_client = genai
        except Exception as e:
            self.logger.warning(f"Failed to initialize some API clients: {e}")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate completion with fallback support.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Generated text

        Raises:
            ApiError: If all providers fail
        """
        errors = []
        providers = [self.provider] + self.fallback_providers

        for provider in providers:
            try:
                return await self._generate_with_provider(
                    provider=provider,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
            except Exception as e:
                self.logger.warning(f"{provider} generation failed: {e}")
                errors.append(f"{provider}: {str(e)}")

        raise ApiError(f"All providers failed: {'; '.join(errors)}")
                async def _generate_with_provider(self, provider: str, prompt: str, max_tokens: int, temperature: float, **kwargs) -> str:
                    """Generate completion using specified provider.

                    Args:
                        provider: Provider name
                        prompt: Input prompt
                        max_tokens: Maximum tokens
                        temperature: Sampling temperature
                        **kwargs: Additional parameters

                    Returns:
                        Generated text

                    Raises:
                        ApiError: If generation fails
                    """
                    try:
                        if provider == "openai" and self.openai_client:
                            response = await self.openai_client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=max_tokens,
                                temperature=temperature,
                                **kwargs
                            )
                            return response.choices[0].message.content

            elif provider == "anthropic" and self.anthropic_client:
                response = await self.anthropic_client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    **kwargs
                )
                return response.content[0].text

            elif provider == "google" and self.google_client:
                model = self.google_client.GenerativeModel(self.model)
                response = await model.generate_content(
                    prompt,
                    generation_config={
                        "max_output_tokens": max_tokens,
                        "temperature": temperature,
                        **kwargs
                    }
                )
                return response.text
                
            else:
                raise ApiError(f"Unsupported or unconfigured provider: {provider}")
        except Exception as e:
            raise ApiError(f"{provider} API error: {str(e)}")
            def tokenize(self, text: str) -> TokenizerResult:
                """Tokenize text using provider's tokenizer.

                Args:
                    text: Input text to tokenize

                Returns:
                    TokenizerResult with tokens and ids

                Raises:
                    TokenizationError: If tokenization fails
                """
                try:
                    if self.provider == "openai" and self.openai_client:
                        tokens = self.openai_client.tokenize(text)
                        token_ids = [t.id for t in tokens]

                    elif self.provider == "anthropic" and self.anthropic_client:
                        tokens = self.anthropic_client.get_tokenizer().tokenize(text)
                        token_ids = [t.id for t in tokens]

                    elif self.provider == "google" and self.google_client:
                        model = self.google_client.GenerativeModel(self.model)
                        token_info = model.count_tokens(text)
                        tokens = token_info.tokens
                        token_ids = list(range(len(tokens)))

                    else:
                        raise TokenizationError(f"Unsupported provider: {self.provider}")

                    return TokenizerResult(
                        tokens=tokens,
                        token_ids=token_ids
                    )
                except Exception as e:
                    raise TokenizationError(f"Tokenization failed: {str(e)}")


class TransformersModel(BaseModel):
    """Hugging Face Transformers model implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize transformers model.
        
        Args:
            config: Optional model configuration
        """
        super().__init__(config) 
        self.model_name = self.config.get("model_name", 
                                        self.config.get("models", {}).get("default", "gpt2"))
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.provider = "transformers"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {self.model_name}: {str(e)}")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate completion using local model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum completion length 
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text completion
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise ModelError(f"Generation failed: {str(e)}")

    def tokenize(self, text: str) -> TokenizerResult:
        """Tokenize text using model tokenizer.
        
        Args:
            text: Input text
            
        Returns:
            TokenizerResult with tokens and attention masks
            
        Raises:
            TokenizationError: If tokenization fails
        """
        try:
            encoding = self.tokenizer(
                text,
                return_tensors=None,
                return_attention_mask=True,
                return_token_type_ids=True
            )
            
            return TokenizerResult(
                tokens=self.tokenizer.convert_ids_to_tokens(encoding["input_ids"]),
                token_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                token_type_ids=encoding["token_type_ids"]
            )
        except Exception as e:
            raise TokenizationError(f"Tokenization failed: {str(e)}")