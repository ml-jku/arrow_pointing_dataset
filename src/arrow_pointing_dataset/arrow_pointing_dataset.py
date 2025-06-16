"""
Arrow Pointing Dataset Core Module

This module provides the core functionality for generating a dataset of images containing
arrows pointing to circles. The dataset can be used for training machine learning models
to detect whether an arrow intersects with a circle.

The module includes utilities for:
- Generating random positions for arrows and circles
- Drawing arrows and circles on images
- Creating dataset instances with configurable parameters
"""

from dataclasses import asdict, dataclass
from math import acos, atan2, cos, pi, sin

import numpy as np
from numpy.random import default_rng
from PIL import Image, ImageDraw


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
        intersection_factor (float): Factor controlling arrow-circle intersection distinguishablity
        boundary_padding (int): Padding from image boundaries in pixels
        arrow_length_min (int): Length of the arrow in pixels
        arrow_length_max (int): Length of the arrow in pixels
        arrow_width_min (int): Width of the arrow line in pixels
        arrow_width_max (int): Width of the arrow line in pixels
        n_samples (int): Number of samples in the dataset
        seed (int): Random seed for reproducibility
    """

    min_dist_x_factor: float = 2.0
    min_dist_y_factor: float = 2.0
    min_radius: int = 15  # 5  # 15
    max_radius: int = 30  # 20  # 30
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


def generate_random_positions(
    intersecting: bool = True,
    min_dist_x_factor=2.0,
    min_dist_y_factor=2.0,
    min_radius=5,
    max_radius=30,
    rng=default_rng(),
    image_size=(224, 224),
    intersection_factor=1.2,
    boundary_padding=30,
    **kwargs,
):
    """Generate random positions for an arrow and circle pair.

    Args:
        intersecting (bool): Whether the arrow should intersect with the circle
        min_dist_x_factor (float): Minimum distance factor in x direction
        min_dist_y_factor (float): Minimum distance factor in y direction
        min_radius (int): Minimum circle radius
        max_radius (int): Maximum circle radius
        rng (numpy.random.Generator): Random number generator
        image_size (tuple[int, int]): Image dimensions (width, height)
        intersection_factor (float): Factor controlling intersection probability
        boundary_padding (int): Padding from image boundaries
        **kwargs: Additional keyword arguments

    Returns:
        tuple: Arrow x, y coordinates, direction angle, circle x, y coordinates, and radius
    """
    _ = kwargs
    sx, sy = image_size[0] - 2 * boundary_padding, image_size[1] - 2 * boundary_padding
    ax, ay = sx * rng.random(), sx * rng.random()

    r = min_radius + (max_radius - min_radius) * rng.random()

    offset_x = min_dist_x_factor * r
    windowx = sx - 2 * offset_x
    cx = ax + offset_x + windowx * rng.random()
    cx = (cx - sx) if (cx > sx) else cx

    offset_y = min_dist_y_factor * r
    windowy = sy - 2 * offset_y
    cy = ay + offset_y + windowy * rng.random()
    cy = (cy - sy) if (cy > sy) else cy

    dx = cx - ax
    dy = cy - ay

    rd = r / intersection_factor if intersecting else intersection_factor * r

    cosphi = acos(np.abs(((dx**2 + dy**2 - rd**2) / (dx**2 + dy**2))) ** (1 / 2))

    phi0 = atan2(dy, dx)

    phi_range = 2 * cosphi if intersecting else 2 * pi - 2 * cosphi
    dphi = rng.random() * phi_range

    phi = phi0 + dphi + (-cosphi if intersecting else cosphi)

    return ax + boundary_padding, ay + boundary_padding, phi, cx + boundary_padding, cy + boundary_padding, r


def draw_arrow_and_circle(
    arrow_position_x,
    arrow_position_y,
    arrow_direction_angle,
    circle_position_x,
    circle_position_y,
    circle_radius,
    image_size=(224, 224),
    arrow_length=8,
    arrow_width=2,
    **kwargs,
):
    """Draw an arrow and circle on a white background image.

    Args:
        arrow_position_x (float): X coordinate of arrow base
        arrow_position_y (float): Y coordinate of arrow base
        arrow_direction_angle (float): Angle of arrow direction in radians
        circle_position_x (float): X coordinate of circle center
        circle_position_y (float): Y coordinate of circle center
        circle_radius (float): Radius of the circle
        image_size (tuple[int, int]): Size of output image (width, height)
        arrow_length (int): Length of the arrow
        arrow_width (int): Width of the arrow line
        **kwargs: Additional keyword arguments

    Returns:
        numpy.ndarray: Generated image as a numpy array
    """
    _ = kwargs
    image = Image.fromarray(np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255)
    draw = ImageDraw.Draw(image)

    apx, apy, ada, cpx, cpy, cr, al, aw = (
        arrow_position_x,
        arrow_position_y,
        arrow_direction_angle,
        circle_position_x,
        circle_position_y,
        circle_radius,
        arrow_length,
        arrow_width,
    )

    draw.line(
        (apx - al // 2 * cos(ada), apy - al // 2 * sin(ada), apx + al // 2 * cos(ada), apy + al // 2 * sin(ada)),
        fill=(0, 0, 0),
        width=aw,
    )
    draw.line(
        (
            apx + al // 2 * cos(ada) - al // 2.5 * cos(ada - 0.5),
            apy + al // 2 * sin(ada) - al // 2.5 * sin(ada - 0.5),
            apx + al // 2 * cos(ada),
            apy + al // 2 * sin(ada),
        ),
        fill=(0, 0, 0),
        width=aw,
    )
    draw.line(
        (
            apx + al // 2 * cos(ada) - al // 2.5 * cos(ada + 0.5),
            apy + al // 2 * sin(ada) - al // 2.5 * sin(ada + 0.5),
            apx + al // 2 * cos(ada),
            apy + al // 2 * sin(ada),
        ),
        fill=(0, 0, 0),
        width=aw,
    )

    ada = ((ada / 2 / pi) - round(ada / 2 / pi)) * 2 * pi

    draw.ellipse((cpx - cr, cpy - cr, cpx + cr, cpy + cr), fill=(0, 0, 0))
    image_array = np.array(image)
    return image_array


class ArrowPointingDataset:
    """Dataset class for generating arrow-pointing-to-circle images.

    This class implements a dataset that generates images containing arrows either
    intersecting or not intersecting with circles. It can be used to train machine
    learning models for intersection detection tasks.

    The dataset is configurable through the ArrowPointingConfig class and generates
    images on-the-fly when indexed.
    """

    config_class = ArrowPointingConfig

    def __init__(self, config: ArrowPointingConfig):
        """Initialize the dataset with the given configuration.

        Args:
            config (ArrowPointingConfig): Configuration object containing dataset parameters
        """
        self.config = config

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
        if idx >= self.config.n_samples:
            raise IndexError

        rng = default_rng(seed=self.config.seed)
        gen_seed = rng.integers(1 << 62)
        rng = default_rng(seed=gen_seed + idx)
        intersecting = rng.uniform(low=0, high=1)
        arrow_length = int(rng.uniform(low=self.config.arrow_length_min, high=self.config.arrow_length_max))
        arrow_width = int(rng.uniform(low=self.config.arrow_width_min, high=self.config.arrow_width_max))
        # cfg_kwargs = asdict(self.config)
        # if "arrow_length" in cfg_kwargs:
        #     cfg_kwargs.pop("arrow_length")
        # if "arrow_width" in cfg_kwargs:
        #     cfg_kwargs.pop("arrow_width")

        arrow_x, arrow_y, phi, circle_x, circle_y, radius = generate_random_positions(
            intersecting=intersecting < self.config.intersection_ratio, rng=rng, **asdict(self.config)
        )

        return (
            draw_arrow_and_circle(
                arrow_x,
                arrow_y,
                phi,
                circle_x,
                circle_y,
                radius,
                arrow_length=arrow_length,
                arrow_width=arrow_width,
                **asdict(self.config),
            ),
            np.array(intersecting < self.config.intersection_ratio, dtype=np.int32),
        )
