"""Unified LLM service supporting multiple providers."""

from typing import Optional, Any, List, Dict
import json

from app.core.logging_config import get_logger
from app.core.providers import provider_manager, ProviderType

logger = get_logger(__name__)

# Lazy imports for optional dependencies
_openai_client = None


def _get_openai_client():
    """Get or initialize OpenAI client."""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            provider = provider_manager.get_provider(ProviderType.OPENAI)
            _openai_client = OpenAI(api_key=provider.api_key.strip('"'))
            logger.info("Initialized OpenAI client")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    return _openai_client


class LLMService:
    """Unified service for interacting with multiple LLM providers."""

    def __init__(self, provider_type: Optional[ProviderType] = None):
        """
        Initialize LLM service.

        Args:
            provider_type: Provider to use (defaults to primary enabled provider)
        """
        if provider_type is None:
            self.provider = provider_manager.get_primary_provider()
        else:
            if not provider_manager.is_provider_enabled(provider_type):
                raise ValueError(f"Provider {provider_type.value} is not enabled")
            self.provider = provider_manager.get_provider(provider_type)

        logger.info(f"Initialized LLM service with {self.provider.provider_type.value}")

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate completion from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Provider-specific arguments

        Returns:
            Generated text
        """
        provider_type = self.provider.provider_type

        try:
            if provider_type == ProviderType.OPENAI:
                return self._complete_openai(messages, temperature, max_tokens, **kwargs)
            elif provider_type == ProviderType.LOCAL:
                return self._complete_local(messages, temperature, max_tokens, **kwargs)
            else:
                raise ValueError(f"Unsupported provider type: {provider_type.value}. Only OpenAI and Local LLM are supported.")
        except Exception as e:
            logger.error(f"Completion failed: {e}", provider=provider_type.value)
            raise

    def _complete_openai(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> str:
        """OpenAI completion."""
        client = _get_openai_client()
        model_lower = self.provider.model.lower()
        is_reasoning_model = model_lower.startswith("o1") or model_lower.startswith("o3") or "gpt-5" in model_lower

        params = {
            "model": self.provider.model,
            "messages": messages,
        }

        if is_reasoning_model:
            # Reasoning models do not support temperature, and use max_completion_tokens instead of max_tokens
            # We set a higher limit to accommodate both internal reasoning tokens and final output tokens.
            if max_tokens is not None:
                params["max_completion_tokens"] = max(max_tokens, 4000)
        else:
            params["temperature"] = temperature
            if max_tokens is not None:
                params["max_tokens"] = max_tokens

        params.update(kwargs)

        try:
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            err_msg = str(e)
            if "max_tokens" in err_msg and "max_completion_tokens" in err_msg:
                # Dynamic fallback for unsupported max_tokens
                logger.info("Retrying with max_completion_tokens fallback", model=self.provider.model)
                if "max_tokens" in params:
                    params["max_completion_tokens"] = params.pop("max_tokens")
                if "temperature" in params:
                    params.pop("temperature")
                response = client.chat.completions.create(**params)
                return response.choices[0].message.content
            raise

    def _complete_local(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> str:
        """Local LLM completion (assumes OpenAI-compatible API)."""
        import requests

        payload = {
            "messages": messages,
            "temperature": temperature,
            "model": self.provider.model,
            "stream": False,  # Local LLM does not support streaming
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        response = requests.post(
            f"{self.provider.base_url}/api/chat",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        result = response.json()
        return result["message"]["content"]

    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available (enabled) providers."""
        return [p.provider_type.value for p in provider_manager.get_enabled_providers()]

    @staticmethod
    def get_provider_status() -> Dict[str, bool]:
        """Get status of all providers."""
        return {
            provider_type.value: provider_manager.is_provider_enabled(provider_type)
            for provider_type in ProviderType
        }


# Create default instance
_default_service = None


def get_llm_service(provider_type: Optional[ProviderType] = None) -> LLMService:
    """Get LLM service instance."""
    global _default_service
    if _default_service is None:
        _default_service = LLMService(provider_type)
    return _default_service
