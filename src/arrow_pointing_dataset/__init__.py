"""
Arrow Pointing Dataset

A dataset generator for creating images of arrows pointing to circles,
with support for PyTorch, TensorFlow, and JAX/Grain integration.
"""

from typing import Callable, Optional, Type

from .arrow_pointing_dataset import (
    ArrowPointingConfig,
    ArrowPointingDataset,
    draw_arrow_and_circle,
    generate_random_positions,
)

# Conditionally import framework-specific implementations
ArrowPointingTorchDataset: Optional[Type] = None
try:
    from .torch_dataset import ArrowPointingTorchDataset
except ImportError:
    pass

ArrowPointingTorchDatasetBatched: Optional[Type] = None
create_dataloader: Optional[Callable] = None
try:
    from .torch_dataset_batched import ArrowPointingTorchDatasetBatched, create_dataloader
except ImportError:
    pass

ArrowPointingTFDataset: Optional[Type] = None
try:
    from .tensorflow_dataset import ArrowPointingTFDataset
except ImportError:
    pass

ArrowPointingMap: Optional[Type] = None
ArrowPointingRandomMap: Optional[Type] = None
try:
    from .grain_dataset import ArrowPointingMap, ArrowPointingRandomMap
except ImportError:
    pass

__version__ = "0.1.0"
__author__ = "Korbinian Pöppel"
__email__ = "poeppel@ml.jku.at"

__all__ = [
    "ArrowPointingDataset",
    "ArrowPointingConfig",
    "generate_random_positions",
    "draw_arrow_and_circle",
    "ArrowPointingTorchDataset",
    "ArrowPointingTorchDatasetBatched",
    "ArrowPointingTFDataset",
    "ArrowPointingRandomMap",
    "ArrowPointingMap",
    "create_dataloader",
]
