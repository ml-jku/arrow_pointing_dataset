"""
Tests for Grain Dataset Implementation

This module contains tests for the Grain dataset transforms, verifying that they correctly
generate samples and maintain consistent behavior across different configurations.
"""

import numpy as np
import pytest
from numpy.random import default_rng

from arrow_pointing_dataset import ArrowPointingConfig
from arrow_pointing_dataset.grain_dataset import ArrowPointingMap, ArrowPointingRandomMap


def test_random_map_transform():
    """Test random map transform."""
    config = ArrowPointingConfig(image_size=(224, 224), n_samples=10, seed=42)
    transform = ArrowPointingRandomMap(config)

    # Test with fixed RNG
    rng = default_rng(42)
    sample = transform.random_map(0, rng)

    # Check sample properties
    assert "image" in sample and "label" in sample
    assert sample["image"].shape == (224, 224, 3)
    assert sample["image"].dtype == np.uint8
    assert np.all((sample["image"] == 0) | (sample["image"] == 255))
    assert sample["label"].dtype == np.int32

    # Test randomness - two calls should give different results
    sample2 = transform.random_map(0, rng)
    assert not np.array_equal(sample["image"], sample2["image"])


def test_deterministic_map_transform():
    """Test deterministic map transform."""
    config = ArrowPointingConfig(image_size=(224, 224), n_samples=10, seed=42)
    transform = ArrowPointingMap(config)

    # Test deterministic behavior - same index should give same result
    rng = default_rng(42)  # Not used by the transform
    sample1 = transform.map(0, rng)
    sample2 = transform.map(0, rng)

    assert np.array_equal(sample1["image"], sample2["image"])
    assert sample1["label"] == sample2["label"]

    # Different indices should give different results
    sample3 = transform.map(1, rng)
    assert not np.array_equal(sample1["image"], sample3["image"])

    # Test intersection ratio
    config = ArrowPointingConfig(image_size=(224, 224), n_samples=1000, seed=42, intersection_ratio=0.7)
    transform = ArrowPointingMap(config)

    # Generate many samples and check if intersection ratio is approximately correct
    labels = []
    for i in range(100):
        sample = transform.map(i, rng)
        labels.append(sample["label"])

    intersection_rate = np.mean(labels)
    assert 0.6 < intersection_rate < 0.8, f"Intersection rate {intersection_rate} too far from configured ratio 0.7"


if __name__ == "__main__":
    pytest.main([__file__])
