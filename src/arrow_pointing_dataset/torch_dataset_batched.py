"""
PyTorch Dataset Integration Module for Batched Arrow Pointing Dataset

This module provides a PyTorch-compatible dataset implementation that leverages the
batched generation capabilities of ArrowPointingDatasetBatched. Instead of returning
single samples, it returns pre-generated batches for maximum efficiency.
"""

from functools import partial
from typing import Callable, Optional

import numpy as np
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

from .arrow_pointing_dataset_batched import ArrowPointingConfig, ArrowPointingDatasetBatched


class ArrowPointingTorchDatasetBatched(Dataset):
    """PyTorch wrapper for ArrowPointingDatasetBatched that returns pre-generated batches.

    This implementation returns full batches instead of individual samples, leveraging
    the efficient batch generation of ArrowPointingDatasetBatched. When using this dataset:

    1. The DataLoader's batch_size should be 1 since we return full batches
    2. The actual batch size is determined by this dataset's batch_size parameter
    3. Transforms should expect batched inputs (B, H, W, C)
    """

    def __init__(self, config: ArrowPointingConfig, batch_size: int = 32, transform: Optional[T.Transform] = None):
        """Initialize the dataset.

        Args:
            config (ArrowPointingConfig): Configuration object containing dataset parameters
            batch_size (int): Size of batches to generate
            transform (T.Transform | None): Optional transform to apply to batches of images
        """
        self.config = config
        self.batch_size = batch_size
        self.dataset = ArrowPointingDatasetBatched(config, batch_size=batch_size)
        self.transform = transform

        # Calculate number of complete batches
        self.n_batches = len(self.dataset) // batch_size

    def __len__(self) -> int:
        """Return the number of batches (not individual samples)."""
        return self.n_batches

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of samples from the dataset.

        Args:
            idx: Index of the batch to get

        Returns:
            tuple: (images, labels) where:
                  - images is a torch.Tensor of shape (batch_size, 3, H, W)
                  - labels is a torch.Tensor of shape (batch_size,)
        """
        if idx < 0 or idx >= self.n_batches:
            raise IndexError("Batch index out of range")

        # Get a batch worth of samples
        # Stack into batch tensors
        images, labels = self.dataset[idx]
        images = torch.from_numpy(images).float()  # (B, H, W, 3)
        images = images.permute(0, 3, 1, 2)  # (B, 3, H, W)
        images = images / 255.0

        labels = torch.tensor(labels, dtype=torch.int32)  # (B,)

        if self.transform is not None:
            images = self.transform(images)

        return images, labels


def collate_as_concat(
    list_of_batches: list[tuple[np.ndarray, np.ndarray]], batch_wrapper_fn: Optional[Callable] = None
):
    """
    Collator for the dataloader using batches of batches of classification images.
    """
    out = (
        torch.cat([b[0] for b in list_of_batches], axis=0),
        torch.cat([b[1] for b in list_of_batches], axis=0),
    )
    if batch_wrapper_fn is not None:
        return batch_wrapper_fn(out)
    return out


def create_dataloader(
    config: ArrowPointingConfig,
    batch_size: int = 32,
    num_workers: int = 1,
    transform: Optional[T.Transform] = None,
    shuffle: bool = True,
    batch_wrapper_fn: Optional[Callable] = None,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for the batched dataset.

    Args:
        config (ArrowPointingConfig): Dataset configuration
        batch_size (int): Size of batches to generate
        num_workers (int): Number of worker processes for data loading
        transform (T.Transform | None): Optional transform to apply to batches
        shuffle (bool): Whether to shuffle the batches
        batch_wrapper_fn (Callable | None): Optional function to wrap batches

    Returns:
        torch.utils.data.DataLoader: Configured data loader that yields batches
    """
    worker_batch_size = batch_size // num_workers if num_workers > 0 else batch_size
    dataset = ArrowPointingTorchDatasetBatched(config=config, batch_size=worker_batch_size, transform=transform)

    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=partial(collate_as_concat, batch_wrapper_fn=batch_wrapper_fn),
        num_workers=num_workers,
        batch_size=num_workers if num_workers > 0 else 1,
        **kwargs,
    )
