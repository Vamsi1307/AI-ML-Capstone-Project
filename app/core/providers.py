"""LLM Provider configurations and factory."""

import os
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class ProviderType(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    LOCAL = "local"  # Ollama or other local LLM servers


@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider."""
    provider_type: ProviderType
    enabled: bool = False
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


class ProviderManager:
    """Manage and validate LLM provider configurations."""

    def __init__(self):
        """Initialize provider configurations from environment variables."""
        self.providers: Dict[ProviderType, LLMProviderConfig] = {}
        self._load_providers()
        self._validate_at_least_one_enabled()

    def _load_providers(self) -> None:
        """Load provider configurations from environment variables."""
        # OpenAI
        openai_enabled = os.getenv("OPENAI_ENABLED", "false").lower() == "true"
        self.providers[ProviderType.OPENAI] = LLMProviderConfig(
            provider_type=ProviderType.OPENAI,
            enabled=openai_enabled and bool(os.getenv("OPENAI_API_KEY")),
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
        )

        # Local LLM (Ollama or other compatible servers - no API key required)
        local_enabled = os.getenv("LOCAL_LLM_ENABLED", "false").lower() == "true"
        self.providers[ProviderType.LOCAL] = LLMProviderConfig(
            provider_type=ProviderType.LOCAL,
            enabled=local_enabled,
            base_url=os.getenv("LOCAL_LLM_URL", "http://localhost:11434"),
            model=os.getenv("LOCAL_LLM_MODEL", "llama2"),
        )

    def _validate_at_least_one_enabled(self) -> None:
        """Validate that at least one provider is enabled."""
        enabled_providers = [p for p in self.providers.values() if p.enabled]

        if not enabled_providers:
            error_msg = (
                "No LLM providers are enabled. "
                "Please enable at least one provider in your .env file:\n"
                "  - OPENAI_ENABLED=true (requires OPENAI_API_KEY)\n"
                "  - LOCAL_LLM_ENABLED=true (requires LOCAL_LLM_URL)"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_enabled_providers(self) -> List[LLMProviderConfig]:
        """Get list of enabled providers."""
        return [p for p in self.providers.values() if p.enabled]

    def get_primary_provider(self) -> LLMProviderConfig:
        """Get the first enabled provider (primary)."""
        enabled = self.get_enabled_providers()
        if not enabled:
            raise ValueError("No enabled LLM providers found")
        return enabled[0]

    def get_provider(self, provider_type: ProviderType) -> LLMProviderConfig:
        """Get provider by type."""
        return self.providers[provider_type]

    def is_provider_enabled(self, provider_type: ProviderType) -> bool:
        """Check if provider is enabled."""
        return self.providers[provider_type].enabled

    def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all providers."""
        return {p.provider_type.value: p.enabled for p in self.providers.values()}

    def log_provider_status(self) -> None:
        """Log the status of all providers."""
        logger.info("LLM Provider Status:")
        for provider_type, config in self.providers.items():
            status = "✓ Enabled" if config.enabled else "✗ Disabled"
            logger.info(f"  {provider_type.value}: {status}")

        primary = self.get_primary_provider()
        logger.info(f"Primary provider: {primary.provider_type.value}")


# Global instance
provider_manager = ProviderManager()
