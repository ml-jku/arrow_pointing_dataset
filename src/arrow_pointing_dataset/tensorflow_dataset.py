"""
TensorFlow Dataset Integration Module

This module provides a TensorFlow-compatible dataset implementation for the arrow pointing task.
It wraps the base ArrowPointingDataset to make it compatible with TensorFlow's data loading
pipeline, enabling efficient data loading and preprocessing for TensorFlow-based models.
"""

from typing import Optional

import tensorflow as tf

from .arrow_pointing_dataset import ArrowPointingConfig, ArrowPointingDataset


class ArrowPointingTFDataset:
    """TensorFlow wrapper for ArrowPointingDataset."""

    def __init__(self, config: ArrowPointingConfig):
        """Initialize the dataset.

        Args:
            config: Configuration for the dataset
        """
        self.config = config
        self.dataset = ArrowPointingDataset(config)

    def get_dataset(self, batch_size: Optional[int] = None, shuffle: bool = False) -> tf.data.Dataset:
        """Returns a tf.data.Dataset.

        Args:
            batch_size: Optional batch size for the dataset
            shuffle: Whether to shuffle the dataset

        Returns:
            A tf.data.Dataset instance
        """
        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=(
                tf.TensorSpec(shape=(*self.config.image_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.dataset))
        if batch_size is not None:
            dataset = dataset.batch(batch_size)

        return dataset.prefetch(tf.data.AUTOTUNE)

    def _generator(self):
        """Generator function for tf.data.Dataset."""
        for _, (image, label) in enumerate(self.dataset):
            # Keep HWC format and normalize to [0, 1]
            image = tf.convert_to_tensor(image, dtype=tf.float32) / 255.0
            label = tf.convert_to_tensor(label, dtype=tf.int32)
            yield image, label
