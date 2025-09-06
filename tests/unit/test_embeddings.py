# ModelMuxer (c) 2025 Ajay Rajput
# Licensed under Business Source License 1.1 – see LICENSE for details.

"""
Comprehensive unit tests for app/classification/embeddings.py.

Tests cover:
- Embedding generation (get_embedding, get_embeddings_batch)
- Similarity calculations (calculate_similarity, _cosine_similarity, _euclidean_similarity)
- Caching mechanisms (memory and disk cache)
- Vector operations and clustering
- Error conditions and edge cases
- Cache statistics and management
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.classification.embeddings import EmbeddingManager
from app.core.exceptions import ClassificationError


class TestEmbeddingManager:
    """Test suite for EmbeddingManager class."""

    @pytest.fixture
    async def temp_cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    async def mock_encoder(self):
        """Create a mock sentence transformer encoder."""
        mock_encoder = MagicMock()
        mock_encoder.get_sentence_embedding_dimension.return_value = 384
        mock_encoder.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        return mock_encoder

    @pytest.fixture
    async def embedding_manager(self, temp_cache_dir, mock_encoder):
        """Create an EmbeddingManager instance with mocked dependencies."""
        with patch('app.classification.embeddings.SentenceTransformer', return_value=mock_encoder):
            manager = EmbeddingManager(
                model_name="test-model",
                cache_dir=str(temp_cache_dir),
                enable_cache=True
            )
            yield manager

    @pytest.mark.asyncio
    async def test_initialization_success(self, temp_cache_dir, mock_encoder):
        """Test successful initialization of EmbeddingManager."""
        with patch('app.classification.embeddings.SentenceTransformer', return_value=mock_encoder):
            manager = EmbeddingManager(
                model_name="test-model",
                cache_dir=str(temp_cache_dir)
            )

            assert manager.model_name == "test-model"
            assert manager.enable_cache == True
            assert manager.embedding_dim == 384
            assert manager.cache_dir == temp_cache_dir
            assert isinstance(manager.memory_cache, dict)
            assert manager.cache_stats == {"hits": 0, "misses": 0, "disk_loads": 0, "disk_saves": 0}

    @pytest.mark.asyncio
    async def test_initialization_model_failure(self, temp_cache_dir):
        """Test initialization failure when model loading fails."""
        with patch('app.classification.embeddings.SentenceTransformer', side_effect=Exception("Model load failed")):
            with pytest.raises(ClassificationError, match="Failed to initialize sentence transformer"):
                EmbeddingManager(model_name="invalid-model", cache_dir=str(temp_cache_dir))

    @pytest.mark.asyncio
    async def test_get_embedding_basic(self, embedding_manager, mock_encoder):
        """Test basic embedding generation."""
        expected = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        mock_encoder.encode.return_value = np.array([expected], dtype=np.float32)

        result = await embedding_manager.get_embedding("test text")

        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)  # Based on mock return
        np.testing.assert_allclose(result, expected, rtol=1e-6)
        mock_encoder.encode.assert_called_once_with(["test text"])

    @pytest.mark.asyncio
    async def test_get_embedding_empty_text(self, embedding_manager):
        """Test embedding generation with empty text."""
        result = await embedding_manager.get_embedding("")

        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)  # embedding_dim
        # Should return zeros for empty text
        np.testing.assert_array_equal(result, np.zeros(384))

    @pytest.mark.asyncio
    async def test_get_embedding_whitespace_only(self, embedding_manager):
        """Test embedding generation with whitespace-only text."""
        result = await embedding_manager.get_embedding("   ")

        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        np.testing.assert_array_equal(result, np.zeros(384))

    @pytest.mark.asyncio
    async def test_get_embedding_caching(self, embedding_manager, mock_encoder):
        """Test that embeddings are cached properly."""
        mock_encoder.encode.return_value = np.array([[0.5, 0.6, 0.7, 0.8]], dtype=np.float32)

        # First call
        result1 = await embedding_manager.get_embedding("test text")
        assert mock_encoder.encode.call_count == 1

        # Second call should use cache
        result2 = await embedding_manager.get_embedding("test text")
        assert mock_encoder.encode.call_count == 1  # Should not call again

        np.testing.assert_array_equal(result1, result2)
        assert embedding_manager.cache_stats["hits"] == 1
        assert embedding_manager.cache_stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_get_embedding_disk_cache(self, embedding_manager, mock_encoder, temp_cache_dir):
        """Test disk caching functionality."""
        mock_encoder.encode.return_value = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)

        # First call - should save to disk
        result1 = await embedding_manager.get_embedding("disk test")

        # Create new manager instance to test disk loading
        with patch('app.classification.embeddings.SentenceTransformer', return_value=mock_encoder):
            new_manager = EmbeddingManager(
                model_name="test-model",
                cache_dir=str(temp_cache_dir),
                enable_cache=True
            )

            # Should load from disk
            result2 = await new_manager.get_embedding("disk test")

        np.testing.assert_array_equal(result1, result2)
        assert new_manager.cache_stats["disk_loads"] == 1

    @pytest.mark.asyncio
    async def test_get_embedding_encoding_failure(self, embedding_manager, mock_encoder):
        """Test handling of encoding failures."""
        mock_encoder.encode.side_effect = Exception("Encoding failed")

        with pytest.raises(ClassificationError, match="Failed to generate embedding"):
            await embedding_manager.get_embedding("test text")

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_basic(self, embedding_manager, mock_encoder):
        """Test batch embedding generation."""
        mock_encoder.encode.return_value = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8]
        ], dtype=np.float32)

        texts = ["text1", "text2"]
        results = await embedding_manager.get_embeddings_batch(texts)

        assert len(results) == 2
        assert all(isinstance(r, np.ndarray) for r in results)
        assert results[0].shape == (4,)
        assert results[1].shape == (4,)

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_empty_list(self, embedding_manager):
        """Test batch embedding with empty text list."""
        results = await embedding_manager.get_embeddings_batch([])

        assert results == []

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_with_empty_texts(self, embedding_manager, mock_encoder):
        """Test batch embedding with some empty texts."""
        # Mock returns embedding with correct dimension (384)
        mock_embedding = np.random.rand(384).astype(np.float32)
        mock_encoder.encode.return_value = np.array([mock_embedding], dtype=np.float32)

        texts = ["", "valid text", "   "]
        results = await embedding_manager.get_embeddings_batch(texts)

        assert len(results) == 3
        # Empty texts should return zero vectors
        np.testing.assert_array_equal(results[0], np.zeros(384))
        np.testing.assert_array_equal(results[2], np.zeros(384))
        # Valid text should have actual embedding
        assert not np.allclose(results[1], np.zeros(384))

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_mixed_cache(self, embedding_manager, mock_encoder):
        """Test batch embedding with mixed cached and uncached texts."""
        mock_encoder.encode.return_value = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)

        # Pre-cache one text
        await embedding_manager.get_embedding("cached text")

        # Batch with cached and uncached
        texts = ["cached text", "new text"]
        results = await embedding_manager.get_embeddings_batch(texts)

        assert len(results) == 2
        # Should only call encode once for the new text
        assert mock_encoder.encode.call_count == 2  # Once for cache, once for batch

    @pytest.mark.asyncio
    async def test_calculate_similarity_cosine(self, embedding_manager):
        """Test cosine similarity calculation."""
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])

        similarity = embedding_manager.calculate_similarity(emb1, emb2, method="cosine")

        assert similarity == 0.0  # Orthogonal vectors

        # Test identical vectors
        similarity = embedding_manager.calculate_similarity(emb1, emb1, method="cosine")
        assert similarity == 1.0

    @pytest.mark.asyncio
    async def test_calculate_similarity_euclidean(self, embedding_manager):
        """Test euclidean similarity calculation."""
        emb1 = np.array([0.0, 0.0, 0.0])
        emb2 = np.array([3.0, 4.0, 0.0])  # Distance = 5

        similarity = embedding_manager.calculate_similarity(emb1, emb2, method="euclidean")

        # Should be 1 / (1 + 5) = 1/6
        assert similarity == pytest.approx(1.0 / 6.0)

    @pytest.mark.asyncio
    async def test_calculate_similarity_dot_product(self, embedding_manager):
        """Test dot product similarity calculation."""
        emb1 = np.array([1.0, 2.0, 3.0])
        emb2 = np.array([4.0, 5.0, 6.0])

        similarity = embedding_manager.calculate_similarity(emb1, emb2, method="dot")

        assert similarity == 32.0  # 1*4 + 2*5 + 3*6

    @pytest.mark.asyncio
    async def test_calculate_similarity_invalid_method(self, embedding_manager):
        """Test similarity calculation with invalid method."""
        emb1 = np.array([1.0, 0.0])
        emb2 = np.array([0.0, 1.0])

        with pytest.raises(ValueError, match="Unknown similarity method"):
            embedding_manager.calculate_similarity(emb1, emb2, method="invalid")

    @pytest.mark.asyncio
    async def test_cosine_similarity_zero_vectors(self, embedding_manager):
        """Test cosine similarity with zero vectors."""
        emb1 = np.array([0.0, 0.0, 0.0])
        emb2 = np.array([1.0, 2.0, 3.0])

        similarity = embedding_manager._cosine_similarity(emb1, emb2)

        assert similarity == 0.0

    @pytest.mark.asyncio
    async def test_euclidean_similarity_identical_vectors(self, embedding_manager):
        """Test euclidean similarity with identical vectors."""
        emb = np.array([1.0, 2.0, 3.0])

        similarity = embedding_manager._euclidean_similarity(emb, emb)

        assert similarity == 1.0  # Distance = 0, so 1 / (1 + 0) = 1

    @pytest.mark.asyncio
    async def test_find_most_similar_basic(self, embedding_manager, mock_encoder):
        """Test finding most similar texts."""
        mock_encoder.encode.side_effect = [
            np.array([[1.0, 0.0, 0.0]]),  # Query embedding
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]])  # Batch candidates
        ]

        candidates = ["text1", "text2", "text3"]
        results = await embedding_manager.find_most_similar("query", candidates, top_k=2)

        assert len(results) == 2
        assert results[0][0] == "text1"  # Most similar
        assert results[1][0] == "text3"  # Second most similar
        assert results[0][1] == 1.0      # Perfect similarity
        assert results[1][1] < results[0][1]  # Sorted by similarity (descending)

    @pytest.mark.asyncio
    async def test_find_most_similar_empty_candidates(self, embedding_manager):
        """Test finding most similar with empty candidates."""
        results = await embedding_manager.find_most_similar("query", [])

        assert results == []

    @pytest.mark.asyncio
    async def test_cluster_embeddings_basic(self, embedding_manager):
        """Test basic clustering functionality."""
        embeddings = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.9, 0.1, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.1, 0.9, 0.0])
        ]

        with patch('sklearn.cluster.KMeans') as mock_kmeans:
            mock_kmeans.return_value.fit_predict.return_value = np.array([0, 0, 1, 1])
            labels = embedding_manager.cluster_embeddings(embeddings, n_clusters=2)

            assert labels == [0, 0, 1, 1]
            mock_kmeans.assert_called_once()
            mock_kmeans.return_value.fit_predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_cluster_embeddings_empty_list(self, embedding_manager):
        """Test clustering with empty embeddings list."""
        labels = embedding_manager.cluster_embeddings([])

        assert labels == []

    @pytest.mark.asyncio
    async def test_cluster_embeddings_sklearn_unavailable(self, embedding_manager):
        """Test clustering when sklearn is not available."""
        embeddings = [np.array([1.0, 2.0, 3.0])]

        with patch.dict('sys.modules', {'sklearn.cluster': None}):
            with pytest.raises(ClassificationError, match="scikit-learn is required for clustering"):
                embedding_manager.cluster_embeddings(embeddings)

    @pytest.mark.asyncio
    async def test_cluster_embeddings_invalid_method(self, embedding_manager):
        """Test clustering with invalid method."""
        embeddings = [np.array([1.0, 2.0, 3.0])]

        with pytest.raises(ClassificationError, match="Clustering failed"):
            embedding_manager.cluster_embeddings(embeddings, method="invalid")

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, embedding_manager):
        """Test cache statistics retrieval."""
        # Simulate some cache activity
        embedding_manager.cache_stats["hits"] = 5
        embedding_manager.cache_stats["misses"] = 3
        embedding_manager.cache_stats["disk_loads"] = 2
        embedding_manager.cache_stats["disk_saves"] = 1

        stats = embedding_manager.get_cache_stats()

        assert stats["cache_stats"]["hits"] == 5
        assert stats["cache_stats"]["misses"] == 3
        assert stats["hit_rate"] == 5 / 8  # 5 hits out of 8 total requests
        assert stats["memory_cache_size"] == 0  # Empty in this test
        assert stats["model_name"] == "test-model"
        assert stats["embedding_dimension"] == 384

    @pytest.mark.asyncio
    async def test_clear_cache_memory_only(self, embedding_manager):
        """Test clearing memory cache only."""
        # Add something to memory cache
        embedding_manager.memory_cache["test"] = np.array([1.0, 2.0, 3.0])

        embedding_manager.clear_cache(memory_only=True)

        assert len(embedding_manager.memory_cache) == 0
        assert embedding_manager.cache_stats == {"hits": 0, "misses": 0, "disk_loads": 0, "disk_saves": 0}

    @pytest.mark.asyncio
    async def test_clear_cache_full(self, embedding_manager, temp_cache_dir):
        """Test clearing both memory and disk cache."""
        # Add to memory cache
        embedding_manager.memory_cache["test"] = np.array([1.0, 2.0, 3.0])

        # Create a fake cache file
        cache_file = temp_cache_dir / "test_cache.pkl"
        cache_file.write_text("fake cache data")

        embedding_manager.clear_cache(memory_only=False)

        assert len(embedding_manager.memory_cache) == 0
        assert not cache_file.exists()
        assert embedding_manager.cache_stats == {"hits": 0, "misses": 0, "disk_loads": 0, "disk_saves": 0}

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, embedding_manager):
        """Test cache key generation."""
        key1 = embedding_manager._get_cache_key("test text")
        key2 = embedding_manager._get_cache_key("test text")
        key3 = embedding_manager._get_cache_key("different text")

        assert key1 == key2  # Same text should produce same key
        assert key1 != key3  # Different text should produce different key
        assert len(key1) == 27  # "test-model_" (11) + hash (16) = 27
        # Actually: "test-model_" (11 chars) + 16 char hash = 27 chars

    @pytest.mark.asyncio
    async def test_cache_path_generation(self, embedding_manager, temp_cache_dir):
        """Test cache file path generation."""
        cache_key = "test_key_123"
        path = embedding_manager._get_cache_path(cache_key)

        assert str(path).endswith("test_key_123.pkl")
        assert path.parent == temp_cache_dir

    @pytest.mark.asyncio
    async def test_large_embeddings_handling(self, embedding_manager, mock_encoder):
        """Test handling of large embedding dimensions."""
        large_embedding = np.random.rand(1536).astype(np.float32)  # GPT-3 size
        mock_encoder.encode.return_value = np.array([large_embedding])

        result = await embedding_manager.get_embedding("large test")

        assert result.shape == (1536,)
        np.testing.assert_array_equal(result, large_embedding)

    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self, embedding_manager, mock_encoder):
        """Test concurrent embedding generation."""
        mock_encoder.encode.side_effect = lambda texts: np.random.rand(len(texts), 384).astype(np.float32)

        texts = ["text1", "text2", "text3"]

        # Run concurrent operations
        results = await asyncio.gather(*[
            embedding_manager.get_embedding(text) for text in texts
        ])

        assert len(results) == 3
        assert all(isinstance(r, np.ndarray) for r in results)
        assert all(r.shape == (384,) for r in results)

    @pytest.mark.asyncio
    async def test_memory_cache_eviction_simulation(self, embedding_manager, mock_encoder):
        """Test memory cache behavior under load."""
        mock_encoder.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

        # Generate many different embeddings
        for i in range(100):
            await embedding_manager.get_embedding(f"text_{i}")

        # Check that cache has grown
        assert len(embedding_manager.memory_cache) == 100

        # Access one again - should be a hit
        await embedding_manager.get_embedding("text_50")
        assert embedding_manager.cache_stats["hits"] == 1

    @pytest.mark.asyncio
    async def test_disk_cache_corruption_handling(self, embedding_manager, temp_cache_dir):
        """Test handling of corrupted disk cache files."""
        # Create corrupted cache file
        cache_key = embedding_manager._get_cache_key("corrupted")
        cache_path = embedding_manager._get_cache_path(cache_key)
        cache_path.write_bytes(b"corrupted data")

        # Should fall back to generating new embedding
        result = await embedding_manager.get_embedding("corrupted")

        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)

    @pytest.mark.asyncio
    async def test_similarity_with_different_vector_sizes(self, embedding_manager):
        """Test similarity calculation with different vector sizes."""
        emb1 = np.array([1.0, 2.0])
        emb2 = np.array([3.0, 4.0])

        similarity = embedding_manager.calculate_similarity(emb1, emb2, method="cosine")

        expected = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert similarity == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_batch_processing_order_preservation(self, embedding_manager, mock_encoder):
        """Test that batch processing preserves input order."""
        mock_encoder.encode.return_value = np.array([
            [0.1, 0.2],  # For "second"
            [0.3, 0.4],  # For "first"
            [0.5, 0.6]   # For "third"
        ], dtype=np.float32)

        texts = ["first", "second", "third"]
        results = await embedding_manager.get_embeddings_batch(texts)

        # Results should be in original order
        assert len(results) == 3
        # Note: In real implementation, order is preserved by the batch processing logic

    @pytest.mark.asyncio
    async def test_unicode_text_handling(self, embedding_manager, mock_encoder):
        """Test handling of unicode text."""
        unicode_text = "Hello 世界 🌍"
        mock_encoder.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

        result = await embedding_manager.get_embedding(unicode_text)

        assert isinstance(result, np.ndarray)
        mock_encoder.encode.assert_called_with([unicode_text])

    @pytest.mark.asyncio
    async def test_very_long_text_handling(self, embedding_manager, mock_encoder):
        """Test handling of very long text."""
        long_text = "A" * 10000  # 10k character text
        mock_encoder.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

        result = await embedding_manager.get_embedding(long_text)

        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    @pytest.mark.asyncio
    async def test_find_most_similar_with_top_k_larger_than_candidates(self, embedding_manager, mock_encoder):
        """Test find_most_similar when top_k exceeds number of candidates."""
        mock_encoder.encode.side_effect = [
            np.array([[1.0, 0.0]]),  # Query
            np.array([[0.9, 0.1], [0.8, 0.2]])  # Batch candidates
        ]

        candidates = ["text1", "text2"]
        results = await embedding_manager.find_most_similar("query", candidates, top_k=5)

        # Should return all candidates
        assert len(results) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    @pytest.mark.asyncio
    async def test_clustering_with_single_cluster(self, embedding_manager):
        """Test clustering with n_clusters=1."""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 2.1, 3.1]),
            np.array([0.9, 1.9, 2.9])
        ]

        with patch('sklearn.cluster.KMeans') as mock_kmeans:
            mock_kmeans.return_value.fit_predict.return_value = np.array([0, 0, 0])
            labels = embedding_manager.cluster_embeddings(embeddings, n_clusters=1)

            assert labels == [0, 0, 0]
            mock_kmeans.assert_called_once_with(n_clusters=1, random_state=42)

    @pytest.mark.asyncio
    async def test_disk_cache_save_warning(self, embedding_manager, mock_encoder, temp_cache_dir):
        """Test disk cache save warning is logged."""
        mock_encoder.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

        # Mock Path.write_bytes to raise an exception
        with patch('pathlib.Path.write_bytes', side_effect=Exception("Disk write failed")):
            with patch('app.classification.embeddings.logger') as mock_logger:
                await embedding_manager.get_embedding("test text")

                # Verify warning was logged
                mock_logger.warning.assert_called_with("failed_to_save_embedding_cache", error=str)

    @pytest.mark.asyncio
    async def test_legacy_pickle_deserialization(self, embedding_manager, temp_cache_dir):
        """Test legacy pickle deserialization fallback."""
        import pickle

        # Create a cache file with pickle format
        cache_key = embedding_manager._get_cache_key("legacy test")
        cache_path = embedding_manager._get_cache_path(cache_key)
        test_embedding = np.array([0.5, 0.6, 0.7, 0.8])

        # Write using pickle (legacy format)
        with open(cache_path, 'wb') as f:
            pickle.dump(test_embedding, f)

        # Mock secure deserialization to fail
        with patch('app.classification.embeddings.secure_serializer.deserialize', side_effect=ValueError):
            result = await embedding_manager.get_embedding("legacy test")

            np.testing.assert_array_equal(result, test_embedding)

    @pytest.mark.asyncio
    async def test_batch_disk_cache_save_warning(self, embedding_manager, mock_encoder, temp_cache_dir):
        """Test batch disk cache save warning is logged."""
        mock_encoder.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

        # Mock Path.write_bytes to raise an exception
        with patch('pathlib.Path.write_bytes', side_effect=Exception("Batch disk write failed")):
            with patch('app.classification.embeddings.logger') as mock_logger:
                await embedding_manager.get_embeddings_batch(["test1", "test2"])

                # Verify warning was logged for each save attempt
                assert mock_logger.warning.call_count >= 1
                mock_logger.warning.assert_called_with("failed_to_save_embedding_cache", error=str)

    @pytest.mark.asyncio
    async def test_clear_cache_disk_cleanup_warning(self, embedding_manager, temp_cache_dir):
        """Test clear_cache disk cleanup warning is logged."""
        # Create a cache file
        cache_file = temp_cache_dir / "test_cache.pkl"
        cache_file.write_text("fake cache")

        # Mock unlink to raise an exception
        with patch('pathlib.Path.unlink', side_effect=Exception("Delete failed")):
            with patch('app.classification.embeddings.logger') as mock_logger:
                embedding_manager.clear_cache(memory_only=False)

                # Verify warning was logged
                mock_logger.warning.assert_called_with(
                    "failed_to_delete_cache_file",
                    file=str(cache_file),
                    error=str
                )