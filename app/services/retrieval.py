"""Document retrieval service for semantic search."""

from typing import List, Dict, Any
import numpy as np

from app.core.config import settings
from app.core.logging_config import get_logger
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService

logger = get_logger(__name__)


class RetrievalService:
    """Service for retrieving relevant document chunks using semantic search."""

    def __init__(
        self,
        embedding_service: EmbeddingService = None,
        vector_store: VectorStoreService = None,
        top_k: int = None,
        similarity_threshold: float = None,
    ):
        """
        Initialize retrieval service.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Service for vector storage and search
            top_k: Number of results to return (default from settings)
            similarity_threshold: Minimum similarity score (default from settings)
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStoreService()
        self.top_k = top_k or settings.TOP_K_RESULTS
        self.similarity_threshold = (
            similarity_threshold or settings.SIMILARITY_THRESHOLD
        )

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query string
            top_k: Number of results (uses default if not provided)

        Returns:
            List of relevant document chunks with metadata and scores
        """
        top_k = top_k or self.top_k

        logger.info("Retrieving documents", query=query, top_k=top_k)

        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_text(query)

            # Search vector store
            results, distances = self.vector_store.search(query_embedding, k=top_k)

            # Convert distances to similarity scores (lower distance = higher similarity)
            # For L2 distance, convert to similarity using 1/(1+distance)
            processed_results = []
            for result in results:
                distance = result.pop("distance", 0)
                similarity = 1 / (1 + distance)

                # Filter by threshold
                if similarity >= self.similarity_threshold:
                    result["similarity_score"] = float(similarity)
                    processed_results.append(result)

            logger.info(
                "Document retrieval successful",
                query=query,
                result_count=len(processed_results),
            )
            return processed_results

        except Exception as e:
            logger.error("Document retrieval failed", query=query, error=str(e))
            raise

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with retrieval statistics
        """
        stats = {
            "total_vectors": self.vector_store.get_vector_count(),
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "embedding_model": self.embedding_service.model,
        }
        return stats
