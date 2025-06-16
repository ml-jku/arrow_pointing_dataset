"""
Grain Dataset Integration Module

This module provides a Grain-compatible dataset implementation for the arrow pointing task.
It wraps the base ArrowPointingDataset to make it compatible with the Grain data loading
framework, enabling efficient data loading and preprocessing for JAX-based models.
"""

from dataclasses import asdict

import numpy as np
from grain.python import MapTransform, RandomMapTransform
from numpy.random import default_rng

from .arrow_pointing_dataset import ArrowPointingConfig, draw_arrow_and_circle, generate_random_positions


class ArrowPointingRandomMap(RandomMapTransform):
    """Random map transform for generating arrow pointing samples.

    This transform generates random samples where arrows either intersect or don't
    intersect with circles, with the intersection probability determined randomly.
    """

    config_class = ArrowPointingConfig

    def __init__(self, config: ArrowPointingConfig):
        """Initialize the transform.

        Args:
            config: Configuration object containing dataset parameters
        """
        self.config = config

    def random_map(self, idx: int, rng: np.random.Generator):
        """Generate a random sample. Solely uses the rng, not the index.

        Args:
            idx: Index of the sample
            rng: Random number generator

        Returns:
            Dictionary containing the generated image and intersection label
        """
        _ = idx
        intersecting = rng.uniform(low=0, high=1)
        image = draw_arrow_and_circle(
            *generate_random_positions(
                intersecting=intersecting < self.config.intersection_ratio, rng=rng, **asdict(self.config)
            ),
            **asdict(self.config),
        )
        return {
            "image": image,
            "label": np.array(intersecting < self.config.intersection_ratio, dtype=np.int32),
        }


class ArrowPointingMap(MapTransform):
    """Deterministic map transform for generating arrow pointing samples.

    This transform generates deterministic samples where arrows either intersect or don't
    intersect with circles, with the intersection determined by the sample index and seed.
    """

    config_class = ArrowPointingConfig

    def __init__(self, config: ArrowPointingConfig):
        """Initialize the transform.

        Args:
            config: Configuration object containing dataset parameters
        """
        self.config = config

    def map(self, idx: int, rng: np.random.Generator):
        """Generate a deterministic sample.

        Args:
            idx: Index of the sample
            rng: Random number generator (not used, we create our own seeded RNG)

        Returns:
            Dictionary containing the generated image and intersection label
        """
        rng = default_rng(seed=self.config.seed * idx)
        intersecting = rng.uniform(low=0, high=1)
        image = draw_arrow_and_circle(
            *generate_random_positions(
                intersecting=intersecting < self.config.intersection_ratio, rng=rng, **asdict(self.config)
            ),
            **asdict(self.config),
        )
        return {
            "image": image,
            "label": np.array(intersecting < self.config.intersection_ratio, dtype=np.int32),
        }
