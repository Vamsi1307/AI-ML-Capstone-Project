"""Embedding service for generating vector representations."""

from typing import List, Union, Optional
import numpy as np
import requests

from app.core.config import settings
from app.core.logging_config import get_logger
from app.core.providers import provider_manager, ProviderType

logger = get_logger(__name__)

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class EmbeddingService:
    """Service for generating vector embeddings from text."""

    def __init__(self, model: Optional[str] = None, provider_type: Optional[ProviderType] = None):
        """
        Initialize embedding service.

        Args:
            model: Model name (default from settings or provider)
            provider_type: Provider to use (defaults to first enabled provider)
        """
        # Determine provider based on availability if not explicitly specified
        if provider_type is None:
            provider_type = self._get_default_provider()

        self.provider_type = provider_type
        self.model = model
        self.client = None
        self._initialize_client()

    def _get_default_provider(self) -> ProviderType:
        """
        Get the default provider from enabled providers.
        Prefers LOCAL, then falls back to OPENAI.

        Returns:
            ProviderType: First enabled provider
        """
        # Prefer LOCAL if enabled
        if provider_manager.is_provider_enabled(ProviderType.LOCAL):
            logger.info("Using LOCAL provider for embeddings")
            return ProviderType.LOCAL

        # Fall back to OpenAI if enabled
        if provider_manager.is_provider_enabled(ProviderType.OPENAI):
            logger.info("Using OPENAI provider for embeddings")
            return ProviderType.OPENAI

        raise ValueError(
            "No embedding provider is enabled. "
            "Please enable at least one provider in your .env file: "
            "LOCAL_LLM_ENABLED=true or OPENAI_ENABLED=true"
        )

    def _get_model_name(self) -> str:
        """
        Get embedding model name from environment or provider configuration.

        Returns:
            Model name for embeddings
        """
        # If explicitly provided, use it
        if self.model:
            return self.model

        # Get from provider configuration based on provider type
        provider = provider_manager.get_provider(self.provider_type)
        if provider.model:
            return provider.model

        # Fallback defaults based on provider
        if self.provider_type == ProviderType.OPENAI:
            return "text-embedding-ada-002"
        elif self.provider_type == ProviderType.LOCAL:
            return "nomic-embed-text"  # Popular open-source embedding model
        else:
            return "nomic-embed-text"

    def _initialize_client(self) -> None:
        """Initialize the appropriate embedding client."""
        try:
            # For OpenAI embeddings
            if self.provider_type == ProviderType.OPENAI:
                if not HAS_OPENAI:
                    raise ImportError(
                        "OpenAI client not installed. Install with: pip install openai"
                    )
                if not provider_manager.is_provider_enabled(ProviderType.OPENAI):
                    raise ValueError("OpenAI provider is not enabled")

                provider = provider_manager.get_provider(ProviderType.OPENAI)
                self.model = self._get_model_name()

                self.client = OpenAI(api_key=provider.api_key.strip('"'))
                logger.info("Initialized OpenAI embedding client", model=self.model)

            # For Local LLM (Ollama), use the /api/embed endpoint
            elif self.provider_type == ProviderType.LOCAL:
                if not provider_manager.is_provider_enabled(ProviderType.LOCAL):
                    raise ValueError("Local LLM provider is not enabled")

                provider = provider_manager.get_provider(ProviderType.LOCAL)
                self.model = self._get_model_name()
                self.base_url = provider.base_url

                # Store client info (no actual client object needed for HTTP API)
                self.client = None
                logger.info(f"Initialized LOCAL embedding client (Ollama) with model {self.model}")

        except Exception as e:
            logger.error(f"Embedding client initialization failed: {e}", provider=self.provider_type)
            raise

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text.

        Args:
            text: Single string or list of strings to embed

        Returns:
            Numpy array of embeddings
        """
        # Handle single string
        if isinstance(text, str):
            text = [text]
            single_input = True
        else:
            single_input = False

        try:
            if self.provider_type == ProviderType.OPENAI:
                embeddings = self._embed_with_openai(text)
            elif self.provider_type == ProviderType.LOCAL:
                embeddings = self._embed_with_ollama(text)
            else:
                raise ValueError(f"Unsupported provider type: {self.provider_type}")

            logger.info(
                "Text embedding completed",
                count=len(text),
                embedding_dim=embeddings.shape[1],
            )

            # Return single embedding if input was single string
            if single_input:
                return embeddings[0]

            return embeddings

        except Exception as e:
            logger.error("Text embedding failed", error=str(e))
            raise

    def _embed_with_openai(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using OpenAI API.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings
        """
        response = self.client.embeddings.create(input=texts, model=self.model)

        embeddings = []
        for item in response.data:
            embeddings.append(item.embedding)

        return np.array(embeddings, dtype=np.float32)

    def _embed_with_ollama(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings using Ollama /api/embed endpoint.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings (N x D)
        """
        embeddings = []

        for i, text in enumerate(texts):
            try:
                logger.debug(f"Embedding text {i+1}/{len(texts)}")
                response = requests.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model,
                        "input": text
                    },
                    timeout=180
                )
                response.raise_for_status()
                data = response.json()

                logger.debug(f"Ollama response keys: {data.keys()}")

                # Ollama returns "embeddings" key with list of vectors
                if "embeddings" in data:
                    embedding = data["embeddings"]
                    logger.debug(f"Got 'embeddings' with type {type(embedding)}, length {len(embedding) if isinstance(embedding, list) else 'N/A'}")

                    # If it's a list of lists (multiple embeddings), take first one
                    if isinstance(embedding, list) and len(embedding) > 0:
                        if isinstance(embedding[0], list):
                            logger.warning(f"Embeddings is list of lists, taking first embedding")
                            embedding = embedding[0]

                        embeddings.append(embedding)
                    else:
                        raise ValueError(f"Embeddings is empty or invalid: {embedding}")
                else:
                    raise ValueError(f"Response missing 'embeddings' key. Keys present: {list(data.keys())}, Full response: {data}")

            except requests.exceptions.RequestException as e:
                logger.error(f"Ollama embedding request failed: {e}")
                raise
            except Exception as e:
                logger.error(f"Error processing Ollama embedding response: {e}")
                raise

        result = np.array(embeddings, dtype=np.float32)
        logger.info(f"Created embedding array with shape {result.shape}")
        return result

    def get_embedding_dimension(self) -> int:
        """
        Get dimension of embeddings.

        Returns:
            Embedding dimension
        """
        if self.provider_type == ProviderType.OPENAI:
            # Common OpenAI embedding dimensions
            if "text-embedding-3-large" in self.model:
                return 3072
            elif "text-embedding-3-small" in self.model:
                return 512
            else:  # text-embedding-ada-002
                return 1536
        elif self.provider_type == ProviderType.LOCAL:
            # Get dimension from Ollama by embedding a test string
            test_embedding = self.embed_text("test")
            return len(test_embedding)
        else:
            raise ValueError(f"Unsupported provider type: {self.provider_type}")
