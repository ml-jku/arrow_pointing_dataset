"""
Tests for the Batched PyTorch Dataset Implementation

This module contains tests for the PyTorch integration of the batched arrow pointing dataset,
focusing on proper batch handling, worker coordination, and data consistency.
"""

import pytest
import torch

from arrow_pointing_dataset.arrow_pointing_dataset_batched import ArrowPointingConfig
from arrow_pointing_dataset.torch_dataset_batched import (
    ArrowPointingTorchDatasetBatched,
    collate_as_concat,
    create_dataloader,
)


def test_batch_shapes():
    """Test that batches have correct shapes and types."""
    config = ArrowPointingConfig(n_samples=100, seed=42)
    dataset = ArrowPointingTorchDatasetBatched(config, batch_size=16)

    # Test single batch
    images, labels = dataset[0]
    assert isinstance(images, torch.Tensor), "Images should be a torch tensor"
    assert isinstance(labels, torch.Tensor), "Labels should be a torch tensor"
    assert images.shape == (16, 3, 224, 224), "Incorrect image batch shape"
    assert labels.shape == (16,), "Incorrect label batch shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.int32, "Labels should be int32"


def test_dataloader_single_worker():
    """Test dataloader with a single worker."""
    config = ArrowPointingConfig(n_samples=100, seed=42)
    batch_size = 32

    dataloader = create_dataloader(config=config, batch_size=batch_size, num_workers=0, shuffle=False)

    # Get first batch
    images, labels = next(iter(dataloader))
    assert images.shape == (batch_size, 3, 224, 224), "Incorrect batch shape"
    assert labels.shape == (batch_size,), "Incorrect labels shape"


def test_dataloader_multi_worker():
    """Test dataloader with multiple workers."""
    config = ArrowPointingConfig(n_samples=128, seed=42)
    batch_size = 32
    num_workers = 4

    dataloader = create_dataloader(config=config, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Collect all batches
    all_images = []
    all_labels = []
    for images, labels in dataloader:
        all_images.append(images)
        all_labels.append(labels)

        # Check batch shapes
        assert images.shape == (batch_size, 3, 224, 224), "Incorrect batch shape"
        assert labels.shape == (batch_size,), "Incorrect labels shape"

    # Check total number of samples
    total_images = torch.cat(all_images, dim=0)
    total_labels = torch.cat(all_labels, dim=0)
    assert len(total_images) == 128, "Incorrect total number of samples"
    assert len(total_labels) == 128, "Incorrect total number of labels"


def test_deterministic_batches():
    """Test that batches are deterministic with fixed seeds."""
    config = ArrowPointingConfig(n_samples=64, seed=42)
    batch_size = 32

    # Create two dataloaders with same config
    dataloader1 = create_dataloader(config=config, batch_size=batch_size, num_workers=2, shuffle=False)

    dataloader2 = create_dataloader(config=config, batch_size=batch_size, num_workers=2, shuffle=False)

    # Compare batches
    for (images1, labels1), (images2, labels2) in zip(dataloader1, dataloader2):
        assert torch.allclose(images1, images2), "Images not deterministic"
        assert torch.all(labels1 == labels2), "Labels not deterministic"


def test_collate_function():
    """Test the collate_as_concat function."""
    # Create dummy batches
    batch1 = (torch.ones(8, 3, 224, 224), torch.zeros(8, dtype=torch.int32))
    batch2 = (torch.zeros(8, 3, 224, 224), torch.ones(8, dtype=torch.int32))

    # Test concatenation
    images, labels = collate_as_concat([batch1, batch2])
    assert images.shape == (16, 3, 224, 224), "Incorrect concatenated shape"
    assert labels.shape == (16,), "Incorrect concatenated labels shape"

    # Verify concatenation order
    assert torch.all(images[:8] == 1), "First batch incorrect"
    assert torch.all(images[8:] == 0), "Second batch incorrect"
    assert torch.all(labels[:8] == 0), "First labels incorrect"
    assert torch.all(labels[8:] == 1), "Second labels incorrect"


# def test_shuffle_consistency():
#     """Test that shuffling maintains batch consistency."""
#     config = ArrowPointingConfig(n_samples=100, seed=42)
#     batch_size = 32

#     dataloader = create_dataloader(config=config, batch_size=batch_size, num_workers=2, shuffle=True)

#     # Get two batches
#     images1, labels1 = next(iter(dataloader))
#     images2, labels2 = next(iter(dataloader))

#     # Verify batch consistency
#     assert images1.shape == (batch_size, 3, 224, 224), "Incorrect shuffled batch shape"
#     assert labels1.shape == (batch_size,), "Incorrect shuffled labels shape"
#     assert not torch.allclose(images1, images2), "Shuffled batches should be different"


if __name__ == "__main__":
    pytest.main([__file__])
