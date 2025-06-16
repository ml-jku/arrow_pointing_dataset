"""
PyTorch Dataset Integration Module

This module provides a PyTorch-compatible dataset implementation for the arrow pointing task.
It wraps the base ArrowPointingDataset to make it compatible with PyTorch's data loading
pipeline, enabling efficient data loading and preprocessing for PyTorch-based models.
"""

from typing import Optional

import numpy as np
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset, get_worker_info

from .arrow_pointing_dataset import ArrowPointingConfig, ArrowPointingDataset


def get_worker_seed(base_seed: int):
    """
    Get the unique seed for a single worker based on the worker id and the base seed.
    """
    worker_info = get_worker_info()

    if worker_info is not None:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        # worker_seed = worker_info.seed
    else:
        worker_id = 0
        num_workers = 1
        # worker_seed = base_seed
    seed_seq = np.random.SeedSequence(base_seed)
    return int(seed_seq.generate_state(num_workers)[worker_id])


class ArrowPointingTorchDataset(Dataset):
    """PyTorch wrapper for ArrowPointingDataset."""

    def __init__(self, config: ArrowPointingConfig, transform: Optional[T.Transform] = None):
        """Initialize the dataset.

        Args:
            config (ArrowPointingConfig): Configuration object containing dataset parameters
                such as image size, number of samples, and random seed.
        """
        self.config = config
        self.dataset = ArrowPointingDataset(config)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to get

        Returns:
            tuple: (image, label) where image is a torch.Tensor of shape (3, H, W)
                  normalized to [0, 1] and label is a integer tensor
        """
        image, label = self.dataset[idx]

        # Convert from HWC to CHW format and normalize to [0, 1]
        image = torch.from_numpy(image).float()  # (H, W, 3)
        image = image.permute(2, 0, 1)  # (3, H, W)
        image = image / 255.0

        label = torch.tensor(label, dtype=torch.int32)
        if self.transform is not None:
            image = self.transform(image)

        return image, label
