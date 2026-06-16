"""Vector store service for storing and retrieving embeddings."""

import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pickle

import numpy as np
from faiss import write_index, read_index, IndexFlatL2
import faiss

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class VectorStoreService:
    """Service for managing vector storage using FAISS."""

    def __init__(self, store_path: str = None):
        """
        Initialize vector store.

        Args:
            store_path: Path to store FAISS index (default from settings)
        """
        if store_path is None:
            from app.core.providers import provider_manager, ProviderType
            if provider_manager.is_provider_enabled(ProviderType.LOCAL):
                provider_name = "local"
            elif provider_manager.is_provider_enabled(ProviderType.OPENAI):
                provider_name = "openai"
            else:
                provider_name = "default"
            self.store_path = os.path.join(settings.VECTOR_STORE_PATH, provider_name)
        else:
            self.store_path = store_path

        Path(self.store_path).mkdir(parents=True, exist_ok=True)

        self.index_path = os.path.join(self.store_path, "index.faiss")
        self.metadata_path = os.path.join(self.store_path, "metadata.pkl")

        self.index = None
        self.metadata = {}
        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        """Load existing index or create new one."""
        if os.path.exists(self.index_path):
            try:
                logger.info("Loading existing FAISS index", path=self.index_path)
                self.index = read_index(self.index_path)

                # Load metadata
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, "rb") as f:
                        self.metadata = pickle.load(f)
                logger.info("Index loaded successfully", vector_count=self.index.ntotal)
            except Exception as e:
                logger.error("Failed to load index", error=str(e))
                self.index = None
                self.metadata = {}
        else:
            logger.info("Creating new FAISS index")

    def add_vectors(
        self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]
    ) -> None:
        """
        Add vectors to the index.

        Args:
            vectors: Numpy array of embeddings (N x D)
            metadata_list: List of metadata dictionaries corresponding to vectors
        """
        if vectors.size == 0:
            logger.warning("No vectors to add")
            return

        vectors = vectors.astype(np.float32)

        # Ensure vectors is 2D
        if len(vectors.shape) == 1:
            logger.warning("Vectors is 1D, reshaping to 2D", shape_before=vectors.shape)
            vectors = vectors.reshape(1, -1)

        dimension = vectors.shape[1]
        logger.info("Processing vectors", shape=vectors.shape, dimension=dimension)

        # Initialize index if not exists
        if self.index is None:
            logger.info("Initializing FAISS index with dimension", dimension=dimension)
            self.index = IndexFlatL2(dimension)

        start_id = self.index.ntotal
        self.index.add(vectors)

        # Store metadata
        for idx, metadata in enumerate(metadata_list):
            self.metadata[start_id + idx] = metadata

        logger.info(
            "Vectors added to index",
            count=len(vectors),
            total_vectors=self.index.ntotal,
        )
        self._save_index()

    def search(
        self, query_vector: np.ndarray, k: int = settings.TOP_K_RESULTS
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding (1 x D)
            k: Number of results to return

        Returns:
            Tuple of (metadata_list, distances_list)
        """
        if self.index is None:
            logger.warning("Index is empty, returning empty results")
            return [], []

        query_vector = query_vector.astype(np.float32).reshape(1, -1)

        # Ensure k doesn't exceed index size
        k = min(k, self.index.ntotal)

        distances, indices = self.index.search(query_vector, k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if int(idx) in self.metadata:
                metadata = self.metadata[int(idx)].copy()
                metadata["distance"] = float(distance)
                results.append(metadata)

        logger.info("Search completed", query_count=k, result_count=len(results))
        return results, distances[0].tolist()

    def _save_index(self) -> None:
        """Save index and metadata to disk."""
        try:
            if self.index is not None:
                write_index(self.index, self.index_path)

            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)

            logger.info("Index saved to disk", path=self.index_path)
        except Exception as e:
            logger.error("Failed to save index", error=str(e))
            raise

    def get_vector_count(self) -> int:
        """
        Get total number of vectors in index.

        Returns:
            Number of vectors
        """
        return self.index.ntotal if self.index else 0

    def reset(self) -> None:
        """Reset the index and metadata."""
        logger.info("Resetting vector store")
        self.index = None
        self.metadata = {}

        # Remove files
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
