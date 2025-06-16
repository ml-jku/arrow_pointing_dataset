"""
Tests for Framework-Specific Dataset Implementations

This module contains tests for the PyTorch and TensorFlow dataset wrappers,
verifying that they correctly handle data loading, batching, and maintain
consistent behavior across framework-specific implementations.
"""

import pytest
import tensorflow as tf
import torch
from torchvision.transforms import v2 as T

from arrow_pointing_dataset import ArrowPointingConfig, ArrowPointingTFDataset, ArrowPointingTorchDataset


def test_torch_dataset():
    """Test PyTorch dataset wrapper."""
    config = ArrowPointingConfig(image_size=(224, 224), n_samples=10, seed=42)
    dataset = ArrowPointingTorchDataset(config)

    # Test length
    assert len(dataset) == 10, "Incorrect dataset length"

    # Test single item
    image, label = dataset[0]

    # Check image properties
    assert image.shape == (3, 224, 224), "Incorrect image shape (should be CHW)"
    assert image.dtype == torch.float32, "Incorrect image dtype"
    assert 0 <= image.min() <= image.max() <= 1, "Image values outside [0,1]"

    # Check label properties
    assert isinstance(label, torch.Tensor), "Label should be a tensor"
    assert label.dtype == torch.int32, "Label should be int32"
    assert label.shape == (), "Label should be a scalar"

    # Test deterministic behavior
    dataset2 = ArrowPointingTorchDataset(config)
    image2, label2 = dataset2[0]
    assert torch.equal(image, image2), "Dataset not deterministic for same seed"
    assert torch.equal(label, label2), "Labels not deterministic for same seed"

    # Test deterministic behavior
    dataset3 = ArrowPointingTorchDataset(config, T.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)))
    image3, _ = dataset3[0]
    assert not torch.allclose(image2, image3)


def test_tensorflow_generator():
    """Test TensorFlow dataset generator."""
    config = ArrowPointingConfig(image_size=(224, 224), n_samples=10, seed=42)
    dataset_wrapper = ArrowPointingTFDataset(config)

    # Test generator output
    items = list(dataset_wrapper._generator())
    assert len(items) == 10, "Generator yielded incorrect number of items"

    # Check first item properties
    image, label = items[0]
    assert isinstance(image, tf.Tensor), "Image should be a TensorFlow tensor"
    assert isinstance(label, tf.Tensor), "Label should be a TensorFlow tensor"
    assert image.shape == (224, 224, 3), "Incorrect image shape"
    assert image.dtype == tf.float32, "Incorrect image dtype"
    assert 0 <= float(tf.reduce_min(image)) <= float(tf.reduce_max(image)) <= 1, "Image values outside [0,1]"
    assert label.dtype == tf.int32, "Incorrect label dtype"
    assert label.shape == (), "Incorrect label shape"

    # Test deterministic behavior
    items2 = list(dataset_wrapper._generator())
    for (image1, label1), (image2, label2) in zip(items, items2):
        assert tf.reduce_all(image1 == image2), "Generator not deterministic for same seed"
        assert tf.reduce_all(label1 == label2), "Labels not deterministic for same seed"


def test_tensorflow_dataset():
    """Test TensorFlow dataset wrapper."""
    config = ArrowPointingConfig(image_size=(224, 224), n_samples=10, seed=42)
    dataset_wrapper = ArrowPointingTFDataset(config)

    # Test without batching
    dataset = dataset_wrapper.get_dataset()

    first_image = None
    # Test first item
    for image, label in dataset.take(1):
        # Check image properties
        first_image = image
        assert image.shape == (224, 224, 3), "Incorrect image shape (should be HWC)"
        assert image.dtype == tf.float32, "Incorrect image dtype"
        assert 0 <= tf.reduce_min(image) <= tf.reduce_max(image) <= 1, "Image values outside [0,1]"

        # Check label properties
        assert label.dtype == tf.int32, "Label should be int32"
        assert label.shape == (), "Label should be a scalar"
        break

    # Test with batching
    batched_dataset = dataset_wrapper.get_dataset(batch_size=2)
    for images, labels in batched_dataset.take(1):
        assert images.shape == (2, 224, 224, 3), "Incorrect batch shape"
        assert labels.shape == (2,), "Incorrect labels shape"
        break

    # Test deterministic behavior
    dataset_wrapper2 = ArrowPointingTFDataset(config)
    for (image1, label1), (image2, label2) in zip(dataset.take(1), dataset_wrapper2.get_dataset().take(1)):
        assert tf.reduce_all(image1 == image2), "Dataset not deterministic for same seed"
        assert tf.reduce_all(label1 == label2), "Labels not deterministic for same seed"
        break

    dataset = dataset_wrapper.get_dataset(shuffle=True)
    for image3, _ in dataset.take(1):
        assert not tf.reduce_all(image3 == first_image), "Shuffled Dataset should return other first element"
        break


if __name__ == "__main__":
    pytest.main([__file__])
