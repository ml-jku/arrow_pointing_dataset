"""
Tests for the Arrow Pointing Dataset

This module contains unit tests for the core functionality of the arrow pointing dataset,
including dataset generation, position calculation, and image drawing functions.
"""

import numpy as np
import pytest

try:
    from arrow_pointing_dataset import (
        ArrowPointingConfig,
        ArrowPointingDataset,
        draw_arrow_and_circle,
        generate_random_positions,
    )
except ImportError:
    import os
    import sys

    sys.path.append(os.path.split(os.path.abspath(__file__))[0])
    from arrow_pointing_dataset import (
        ArrowPointingConfig,
        ArrowPointingDataset,
        draw_arrow_and_circle,
        generate_random_positions,
    )


def test_arrow_pointing_dataset():
    """Test basic dataset functionality with default configuration."""
    # Test with default config
    config = ArrowPointingConfig()
    dataset = ArrowPointingDataset(config)

    # Test dataset item generation
    image, label = dataset[0]

    # Check image properties
    assert image.shape == (*config.image_size, 3), "Incorrect image dimensions"
    assert image.dtype == np.uint8, "Incorrect image dtype"
    assert np.all((image == 0) | (image == 255)), "Image values should be binary (0 or 255)"

    # Check label properties
    assert isinstance(label, np.ndarray), "Label should be int32"
    assert label.dtype == np.int32


def test_generate_random_positions():
    """Test generation of random positions for arrows and circles."""
    # Test intersecting case
    pos = generate_random_positions(
        intersecting=True,
        min_dist_x_factor=2.0,
        min_dist_y_factor=2.0,
        min_radius=15,
        max_radius=30,
        image_size=(224, 224),
        intersection_factor=1.2,
        boundary_padding=30,
    )

    ax, ay, _, cx, cy, r = pos  # phi not used in bounds checking

    # Check position bounds
    assert 30 <= ax <= 194, "Arrow x position out of bounds"
    assert 30 <= ay <= 194, "Arrow y position out of bounds"
    assert 30 <= cx <= 194, "Circle x position out of bounds"
    assert 30 <= cy <= 194, "Circle y position out of bounds"
    assert 15 <= r <= 30, "Circle radius out of bounds"

    # Test non-intersecting case
    pos_non = generate_random_positions(
        intersecting=False,
        min_dist_x_factor=2.0,
        min_dist_y_factor=2.0,
        min_radius=15,
        max_radius=30,
        image_size=(224, 224),
        intersection_factor=1.2,
        boundary_padding=30,
    )

    # Verify different configurations are generated
    assert pos != pos_non, "Intersecting and non-intersecting configurations should differ"


def test_draw_arrow_and_circle():
    """Test the arrow and circle drawing functionality."""
    # Test drawing function
    image = draw_arrow_and_circle(
        arrow_position_x=112,
        arrow_position_y=112,
        arrow_direction_angle=0,  # pointing right
        circle_position_x=162,
        circle_position_y=112,
        circle_radius=20,
        image_size=(224, 224),
        arrow_length=30,
        arrow_width=3,
    )

    # Check image properties
    assert image.shape == (224, 224, 3), "Incorrect image dimensions"
    assert image.dtype == np.uint8, "Incorrect image dtype"
    assert np.all((image == 0) | (image == 255)), "Image values should be binary (0 or 255)"

    # Verify arrow and circle are drawn (should have some black pixels)
    assert np.any(image == 0), "No black pixels found in the image"


def test_dataset_deterministic():
    """Test that the dataset produces deterministic results with fixed seeds."""
    # Test seed reproducibility
    config = ArrowPointingConfig(seed=42)
    dataset1 = ArrowPointingDataset(config)
    dataset2 = ArrowPointingDataset(config)

    # Same index should produce identical results
    img1, label1 = dataset1[0]
    img2, label2 = dataset2[0]

    assert np.array_equal(img1, img2), "Dataset not deterministic for same seed"
    assert label1 == label2, "Labels not deterministic for same seed"


if __name__ == "__main__":
    pytest.main([__file__])
