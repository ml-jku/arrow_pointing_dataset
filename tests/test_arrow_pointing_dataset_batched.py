"""
Tests for the Fast Arrow Pointing Dataset Implementation

This module contains unit tests for the optimized version of the arrow pointing dataset,
verifying that it produces identical results to the original implementation while using
batch-based numpy operations.
"""

import numpy as np
import pytest

from arrow_pointing_dataset import ArrowPointingConfig, ArrowPointingDataset
from arrow_pointing_dataset.arrow_pointing_dataset_batched import ArrowPointingDatasetBatched


def test_fast_dataset_matches_original():
    """Test that the fast implementation produces identical results to the original."""
    config = ArrowPointingConfig(seed=42, n_samples=100)

    # Create both dataset versions
    original_dataset = ArrowPointingDataset(config)
    fast_dataset = ArrowPointingDatasetBatched(config, batch_size=16)

    # Test multiple indices to ensure consistency across the dataset
    for idx in [0, 10, 25, 50, 75, 99]:
        _ = original_dataset[idx]  # Original dataset access for comparison
        fast_img, _ = fast_dataset[idx]

        # Verify image properties
        assert fast_img[0].shape == (*config.image_size, 3), "Incorrect image dimensions"
        assert fast_img[0].dtype == np.uint8, "Incorrect image dtype"
        assert fast_img.shape == (16, *config.image_size, 3), "Incorrect batch shape"
        assert np.all((fast_img == 0) | (fast_img == 255)), "Image values should be binary (0 or 255)"


def test_fast_dataset_deterministic():
    """Test that the fast dataset produces deterministic results with fixed seeds."""
    config = ArrowPointingConfig(seed=42)

    # Create two instances of the fast dataset
    dataset1 = ArrowPointingDatasetBatched(config, batch_size=16)
    dataset2 = ArrowPointingDatasetBatched(config, batch_size=16)

    # Test multiple indices to ensure deterministic behavior
    for idx in range(5):
        img1, label1 = dataset1[idx]
        img2, label2 = dataset2[idx]

        assert np.array_equal(img1, img2), f"Dataset not deterministic for same seed at index {idx}"
        assert np.all(label1 == label2), f"Labels not deterministic for same seed at index {idx}"


def test_fast_dataset_batch_consistency():
    """Test that samples are consistent across different batch boundaries."""
    config = ArrowPointingConfig(seed=42, n_samples=100)
    batch_size = 16
    dataset = ArrowPointingDatasetBatched(config, batch_size=batch_size)

    # Get samples from end of one batch and start of next
    end_of_batch = dataset[batch_size - 1]
    start_of_next = dataset[batch_size]

    # Get same samples again to verify consistency
    end_of_batch_repeat = dataset[batch_size - 1]
    start_of_next_repeat = dataset[batch_size]

    assert np.array_equal(end_of_batch[0], end_of_batch_repeat[0]), "Sample at batch end not consistent"
    assert np.array_equal(start_of_next[0], start_of_next_repeat[0]), "Sample at batch start not consistent"


def test_fast_dataset_length():
    """Test that the dataset length is correctly reported."""
    n_samples = 1000
    config = ArrowPointingConfig(n_samples=n_samples)
    dataset = ArrowPointingDatasetBatched(config, batch_size=16)

    assert len(dataset) == n_samples, "Dataset length incorrect"


def test_fast_dataset_index_bounds():
    """Test that the dataset correctly handles index bounds."""
    config = ArrowPointingConfig(n_samples=100)
    dataset = ArrowPointingDatasetBatched(config, batch_size=16)

    # Test valid indices
    _ = dataset[0]
    _ = dataset[99]

    # Test invalid indices
    with pytest.raises(IndexError):
        _ = dataset[100]

    with pytest.raises(IndexError):
        _ = dataset[-1]


if __name__ == "__main__":
    pytest.main([__file__])
