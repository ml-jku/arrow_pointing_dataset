"""
Arrow Pointing Dataset Fast Implementation

This module provides an optimized version of the arrow pointing dataset that generates
batches of images directly using numpy operations, avoiding PIL/draw dependencies and
memory shuffling bottlenecks.
"""

from dataclasses import dataclass
from math import pi

import numpy as np
from numpy.random import default_rng


# pylint: disable=duplicate-code
@dataclass
class ArrowPointingConfig:
    """Configuration class for the Arrow Pointing Dataset.

    This class defines all the parameters needed to generate the dataset, including
    geometric constraints, image properties, and dataset size settings.

    Attributes:
        min_dist_x_factor (float): Minimum distance factor in x direction between arrow and circle
        min_dist_y_factor (float): Minimum distance factor in y direction between arrow and circle
        min_radius (int): Minimum radius of the circle in pixels
        max_radius (int): Maximum radius of the circle in pixels
        image_size (tuple[int, int]): Size of the output image (width, height)
        intersection_factor (float): Factor controlling arrow-circle intersection probability
        boundary_padding (int): Padding from image boundaries in pixels
        arrow_length (int): Length of the arrow in pixels
        arrow_width (int): Width of the arrow line in pixels
        n_samples (int): Number of samples in the dataset
        seed (int): Random seed for reproducibility
    """

    min_dist_x_factor: float = 2.0
    min_dist_y_factor: float = 2.0
    min_radius: int = 15
    max_radius: int = 30
    image_size: tuple[int, int] = (224, 224)
    intersection_factor: float = 1.2
    boundary_padding: int = 30
    arrow_length_min: int = 10
    arrow_length_max: int = 30  # 10  # 30
    arrow_width_min: int = 2
    arrow_width_max: int = 3

    intersection_ratio: float = 0.5
    n_samples: int = 50000
    seed: int = 42


# pylint: enable=duplicate-code


def generate_random_positions_batch(batch_size: int, config: ArrowPointingConfig, rng: np.random.Generator):
    """Generate random positions for a batch of arrow and circle pairs.

    Args:
        batch_size (int): Number of samples to generate
        config (ArrowPointingConfig): Dataset configuration
        rng (numpy.random.Generator): Random number generator

    Returns:
        tuple: Arrays of positions and parameters for the batch
    """
    sx = config.image_size[0] - 2 * config.boundary_padding
    sy = config.image_size[1] - 2 * config.boundary_padding

    # Generate random arrow positions
    ax = sx * rng.random(batch_size)
    ay = sy * rng.random(batch_size)

    # Generate random circle radii
    r = config.min_radius + (config.max_radius - config.min_radius) * rng.random(batch_size)

    # Generate circle positions
    offset_x = config.min_dist_x_factor * r
    windowx = sx - 2 * offset_x
    cx = ax + offset_x + windowx * rng.random(batch_size)
    cx = np.where(cx > sx, cx - sx, cx)

    offset_y = config.min_dist_y_factor * r
    windowy = sy - 2 * offset_y
    cy = ay + offset_y + windowy * rng.random(batch_size)
    cy = np.where(cy > sy, cy - sy, cy)

    # Calculate distances and angles
    dx = cx - ax
    dy = cy - ay
    dist_squared = dx**2 + dy**2

    # Generate intersection labels
    intersecting = rng.random(batch_size) < config.intersection_ratio
    rd = np.where(intersecting, r / config.intersection_factor, r * config.intersection_factor)

    # Calculate angles
    cos_phi = np.sqrt(np.abs((dist_squared - rd**2) / dist_squared))
    phi0 = np.arctan2(dy, dx)

    phi_range = np.where(intersecting, 2 * np.arccos(cos_phi), 2 * pi - 2 * np.arccos(cos_phi))
    dphi = rng.random(batch_size) * phi_range
    phi = phi0 + dphi + np.where(intersecting, -np.arccos(cos_phi), np.arccos(cos_phi))

    # Add boundary padding
    ax += config.boundary_padding
    ay += config.boundary_padding
    cx += config.boundary_padding
    cy += config.boundary_padding

    return ax, ay, phi, cx, cy, r, intersecting


def draw_arrows_and_circles_batch(
    positions, arrow_length: np.ndarray, arrow_width: np.ndarray, config: ArrowPointingConfig
):
    """Draw a batch of arrows and circles using vectorized numpy operations.

    Args:
        positions: Tuple of position arrays from generate_random_positions_batch
        config (ArrowPointingConfig): Dataset configuration

    Returns:
        numpy.ndarray: Batch of generated images
    """
    ax, ay, phi, cx, cy, r, _ = positions
    batch_size = len(ax)

    # Create batch of white images
    images = np.ones((batch_size, *config.image_size, 3), dtype=np.uint8) * 255

    # Create coordinate grids for vectorized operations (match PIL's coordinate system)
    x, y = np.meshgrid(np.arange(config.image_size[0]), np.arange(config.image_size[1]))
    x = x.T  # Transpose to match PIL's coordinate system
    y = y.T

    # Expand coordinates for broadcasting with batch dimension
    x_batch = x[None, :, :]  # Shape: (batch_size, height, width)
    y_batch = y[None, :, :]

    # Draw circles (vectorized across batch)
    circle_mask = ((x_batch - cx[:, None, None]) ** 2 + (y_batch - cy[:, None, None]) ** 2) <= r[:, None, None] ** 2
    images[circle_mask] = 0

    # Compute arrow parameters for all batches
    cos_phi = np.cos(phi)[:, None, None]
    sin_phi = np.sin(phi)[:, None, None]
    ax_batch = ax[:, None, None]
    ay_batch = ay[:, None, None]

    # Arrow stems
    start_x = ax_batch - arrow_length[:, None, None] // 2 * cos_phi
    start_y = ay_batch - arrow_length[:, None, None] // 2 * sin_phi
    end_x = ax_batch + arrow_length[:, None, None] // 2 * cos_phi
    end_y = ay_batch + arrow_length[:, None, None] // 2 * sin_phi

    # Draw arrow stems (vectorized)
    dx = end_x - start_x
    dy = end_y - start_y
    line_dist = np.abs(dy * (x_batch - start_x) - dx * (y_batch - start_y)) / np.sqrt(dx**2 + dy**2)
    line_bounds = ((x_batch - start_x) * dx + (y_batch - start_y) * dy) / (dx**2 + dy**2)
    line_mask = (line_dist <= arrow_width[:, None, None] / 2) & (line_bounds >= 0) & (line_bounds <= 1)
    images[line_mask] = 0

    # Arrow heads
    head_length = arrow_length[:, None, None] // 2.5
    head_angle = 0.5

    # Left head lines (vectorized)
    head_start_x_left = end_x - head_length * np.cos(phi[:, None, None] - head_angle)
    head_start_y_left = end_y - head_length * np.sin(phi[:, None, None] - head_angle)
    dx_left = end_x - head_start_x_left
    dy_left = end_y - head_start_y_left
    head_dist_left = np.abs(
        dy_left * (x_batch - head_start_x_left) - dx_left * (y_batch - head_start_y_left)
    ) / np.sqrt(dx_left**2 + dy_left**2)
    head_bounds_left = ((x_batch - head_start_x_left) * dx_left + (y_batch - head_start_y_left) * dy_left) / (
        dx_left**2 + dy_left**2
    )
    head_mask_left = (
        (head_dist_left <= arrow_width[:, None, None] / 2) & (head_bounds_left >= 0) & (head_bounds_left <= 1)
    )
    images[head_mask_left] = 0

    # Right head lines (vectorized)
    head_start_x_right = end_x - head_length * np.cos(phi[:, None, None] + head_angle)
    head_start_y_right = end_y - head_length * np.sin(phi[:, None, None] + head_angle)
    dx_right = end_x - head_start_x_right
    dy_right = end_y - head_start_y_right
    head_dist_right = np.abs(
        dy_right * (x_batch - head_start_x_right) - dx_right * (y_batch - head_start_y_right)
    ) / np.sqrt(dx_right**2 + dy_right**2)
    head_bounds_right = ((x_batch - head_start_x_right) * dx_right + (y_batch - head_start_y_right) * dy_right) / (
        dx_right**2 + dy_right**2
    )
    head_mask_right = (
        (head_dist_right <= arrow_width[:, None, None] / 2) & (head_bounds_right >= 0) & (head_bounds_right <= 1)
    )
    images[head_mask_right] = 0

    return images


class ArrowPointingDatasetBatched:
    """Optimized dataset class for generating arrow-pointing-to-circle images.

    This class implements a dataset that generates batches of images containing arrows
    either intersecting or not intersecting with circles. It uses pure numpy operations
    for improved performance.
    """

    config_class = ArrowPointingConfig

    def __init__(self, config: ArrowPointingConfig, batch_size: int):
        """Initialize the dataset with the given configuration.

        Args:
            config (ArrowPointingConfig): Configuration object containing dataset parameters
        """
        self.config = config
        self.batch_size = batch_size

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Number of samples as specified in the configuration
        """
        return self.config.n_samples

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to generate

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing the image array and intersection label
        """
        if idx < 0 or idx >= self.config.n_samples:
            raise IndexError("Dataset index out of range")

        rng = default_rng(seed=self.config.seed)
        rng = default_rng(seed=rng.integers(1 << 62) + idx)
        positions = generate_random_positions_batch(self.batch_size, self.config, rng)

        arrow_length = rng.uniform(
            size=(self.batch_size,), low=self.config.arrow_length_min, high=self.config.arrow_length_max
        )
        arrow_width = rng.uniform(
            size=(self.batch_size,), low=self.config.arrow_width_min, high=self.config.arrow_width_max
        )

        images = draw_arrows_and_circles_batch(positions, arrow_length, arrow_width, self.config)

        return images, positions[6]
