"""Core configuration management."""

import os
from typing import Optional
from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Application settings from environment variables."""

    # API Configuration
    API_TITLE: str = "GenAI Document Assistant"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Production-ready RAG + Agentic AI system"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Server Configuration
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

    # Vector Storage
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "app/data/vectors")
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "faiss")

    # Document Processing
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "200"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "overlap")

    # Retrieval Configuration
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

    # Agent Configuration
    MAX_REACT_ITERATIONS: int = int(os.getenv("MAX_REACT_ITERATIONS", "8"))

    # Preprocessing Configuration
    ENABLE_CHUNK_SUMMARIES: bool = os.getenv("ENABLE_CHUNK_SUMMARIES", "false").lower() == "true"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/app.log")

    @classmethod
    def validate(cls) -> None:
        """Validate critical configuration."""
        # Provider validation is now handled in providers.py
        pass

    def __init__(self):
        """Initialize settings and validate."""
        self.validate()


settings = Settings()

