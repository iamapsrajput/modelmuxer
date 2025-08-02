# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.
"""
Embedding management for ModelMuxer classification system.

This module provides utilities for generating, caching, and managing
embeddings used in prompt classification and semantic routing.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from ..core.exceptions import ClassificationError
from ..core.utils import hash_prompt

logger = structlog.get_logger(__name__)


class EmbeddingManager:
    """
    Manages embeddings for prompt classification and semantic routing.

    This class handles embedding generation, caching, similarity calculations,
    and provides utilities for working with semantic representations of text.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
    ):
        self.model_name = model_name
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache/embeddings")

        # Initialize the sentence transformer
        try:
            self.encoder = SentenceTransformer(model_name)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
            logger.info(
                "embedding_manager_initialized", model=model_name, dimension=self.embedding_dim
            )
        except Exception as e:
            raise ClassificationError(f"Failed to initialize sentence transformer: {e}")

        # Create cache directory
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for frequently used embeddings
        self.memory_cache: Dict[str, np.ndarray] = {}
        self.cache_stats = {"hits": 0, "misses": 0, "disk_loads": 0, "disk_saves": 0}

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"{self.model_name}_{text_hash[:16]}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cached embedding."""
        return self.cache_dir / f"{cache_key}.pkl"

    async def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text with caching support.

        Args:
            text: Input text to embed

        Returns:
            Numpy array containing the embedding
        """
        if not text.strip():
            return np.zeros(self.embedding_dim)

        cache_key = self._get_cache_key(text)

        # Check memory cache first
        if cache_key in self.memory_cache:
            self.cache_stats["hits"] += 1
            return self.memory_cache[cache_key]

        # Check disk cache
        if self.enable_cache:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        embedding = pickle.load(f)

                    # Store in memory cache
                    self.memory_cache[cache_key] = embedding
                    self.cache_stats["hits"] += 1
                    self.cache_stats["disk_loads"] += 1

                    return embedding
                except Exception as e:
                    logger.warning("failed_to_load_cached_embedding", error=str(e))

        # Generate new embedding
        try:
            embedding = self.encoder.encode([text])[0]

            # Cache the embedding
            self.memory_cache[cache_key] = embedding
            self.cache_stats["misses"] += 1

            # Save to disk cache
            if self.enable_cache:
                try:
                    cache_path = self._get_cache_path(cache_key)
                    with open(cache_path, "wb") as f:
                        pickle.dump(embedding, f)
                    self.cache_stats["disk_saves"] += 1
                except Exception as e:
                    logger.warning("failed_to_save_embedding_cache", error=str(e))

            return embedding

        except Exception as e:
            raise ClassificationError(f"Failed to generate embedding: {e}")

    async def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts efficiently.

        Args:
            texts: List of input texts

        Returns:
            List of numpy arrays containing embeddings
        """
        if not texts:
            return []

        # Check which texts are already cached
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if not text.strip():
                cached_embeddings[i] = np.zeros(self.embedding_dim)
                continue

            cache_key = self._get_cache_key(text)

            # Check memory cache
            if cache_key in self.memory_cache:
                cached_embeddings[i] = self.memory_cache[cache_key]
                self.cache_stats["hits"] += 1
                continue

            # Check disk cache
            if self.enable_cache:
                cache_path = self._get_cache_path(cache_key)
                if cache_path.exists():
                    try:
                        with open(cache_path, "rb") as f:
                            embedding = pickle.load(f)

                        cached_embeddings[i] = embedding
                        self.memory_cache[cache_key] = embedding
                        self.cache_stats["hits"] += 1
                        self.cache_stats["disk_loads"] += 1
                        continue
                    except Exception as e:
                        logger.warning("failed_to_load_cached_embedding", error=str(e))

            # Text needs to be embedded
            uncached_texts.append(text)
            uncached_indices.append(i)

        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                new_embeddings = self.encoder.encode(uncached_texts)

                for idx, embedding in zip(uncached_indices, new_embeddings):
                    cached_embeddings[idx] = embedding

                    # Cache the embedding
                    text = texts[idx]
                    cache_key = self._get_cache_key(text)
                    self.memory_cache[cache_key] = embedding
                    self.cache_stats["misses"] += 1

                    # Save to disk cache
                    if self.enable_cache:
                        try:
                            cache_path = self._get_cache_path(cache_key)
                            with open(cache_path, "wb") as f:
                                pickle.dump(embedding, f)
                            self.cache_stats["disk_saves"] += 1
                        except Exception as e:
                            logger.warning("failed_to_save_embedding_cache", error=str(e))

            except Exception as e:
                raise ClassificationError(f"Failed to generate batch embeddings: {e}")

        # Return embeddings in original order
        return [cached_embeddings[i] for i in range(len(texts))]

    def calculate_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray, method: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            method: Similarity method ("cosine", "euclidean", "dot")

        Returns:
            Similarity score
        """
        if method == "cosine":
            return self._cosine_similarity(embedding1, embedding2)
        elif method == "euclidean":
            return self._euclidean_similarity(embedding1, embedding2)
        elif method == "dot":
            return np.dot(embedding1, embedding2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _euclidean_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate euclidean similarity (inverse of distance) between two embeddings."""
        distance = np.linalg.norm(embedding1 - embedding2)
        return 1.0 / (1.0 + distance)  # Convert distance to similarity

    async def find_most_similar(
        self, query_text: str, candidate_texts: List[str], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find the most similar texts to a query.

        Args:
            query_text: Query text
            candidate_texts: List of candidate texts
            top_k: Number of top results to return

        Returns:
            List of (text, similarity_score) tuples
        """
        if not candidate_texts:
            return []

        # Get embeddings
        query_embedding = await self.get_embedding(query_text)
        candidate_embeddings = await self.get_embeddings_batch(candidate_texts)

        # Calculate similarities
        similarities = []
        for text, embedding in zip(candidate_texts, candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, embedding)
            similarities.append((text, similarity))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def cluster_embeddings(
        self, embeddings: List[np.ndarray], n_clusters: int = 5, method: str = "kmeans"
    ) -> List[int]:
        """
        Cluster embeddings into groups.

        Args:
            embeddings: List of embeddings to cluster
            n_clusters: Number of clusters
            method: Clustering method ("kmeans")

        Returns:
            List of cluster labels
        """
        if not embeddings:
            return []

        try:
            from sklearn.cluster import KMeans

            embeddings_array = np.array(embeddings)

            if method == "kmeans":
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(embeddings_array)
                return labels.tolist()
            else:
                raise ValueError(f"Unknown clustering method: {method}")

        except ImportError:
            raise ClassificationError("scikit-learn is required for clustering")
        except Exception as e:
            raise ClassificationError(f"Clustering failed: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / max(total_requests, 1)

        return {
            "cache_stats": self.cache_stats.copy(),
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache),
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
        }

    def clear_cache(self, memory_only: bool = False) -> None:
        """Clear embedding cache."""
        self.memory_cache.clear()

        if not memory_only and self.enable_cache:
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(
                        "failed_to_delete_cache_file", file=str(cache_file), error=str(e)
                    )

        # Reset stats
        self.cache_stats = {"hits": 0, "misses": 0, "disk_loads": 0, "disk_saves": 0}

        logger.info("embedding_cache_cleared", memory_only=memory_only)
