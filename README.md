# Arrow Pointing Dataset

A customizable dataset generator for creating images of arrows pointing to circles, useful for machine learning experiments in visual reasoning and spatial relationships.

## Installation

```bash
# Basic installation
pip install arrow_pointing_dataset

# With PyTorch support
pip install arrow_pointing_dataset[torch]

# With TensorFlow support
pip install arrow_pointing_dataset[tensorflow]
```

## Usage

### Basic Usage
```python
from arrow_pointing_dataset import ArrowPointingDataset, ArrayPointingConfig

# Create dataset with default configuration
config = ArrayPointingConfig(
    image_size=(224, 224),
    n_samples=1000
)
dataset = ArrowPointingDataset(config)

# Get a sample
image, label = dataset[0]  # Returns numpy array and integer (0/1) label
```

### PyTorch Integration
```python
from arrow_pointing_dataset.torch import ArrowPointingTorchDataset
import torch

# Create PyTorch dataset
dataset = ArrowPointingTorchDataset(
    image_size=(224, 224),
    n_samples=1000,
    seed=42
)

# Use with DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# With transforms
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ArrowPointingTorchDataset(
    transform=transform,
    image_size=(224, 224)
)
```

### TensorFlow Integration
```python
from arrow_pointing_dataset.tensorflow import ArrowPointingTFDataset

# Create TensorFlow dataset
dataset = ArrowPointingTFDataset(
    image_size=(224, 224),
    n_samples=1000,
    seed=42
)

# Get tf.data.Dataset
tf_dataset = dataset.get_dataset()

# Apply batching and shuffling
tf_dataset = tf_dataset.shuffle(1000).batch(32)
```

## Configuration Options

The dataset can be customized using `ArrayPointingConfig`:

```python
config = ArrayPointingConfig(
    image_size=(224, 224),      # Size of output images
    min_radius=15,              # Minimum circle radius
    max_radius=30,              # Maximum circle radius
    arrow_length=30,            # Length of the arrow
    arrow_width=3,              # Width of the arrow
    boundary_padding=30,        # Padding from image boundaries
    n_samples=1000,             # Number of samples to generate
    seed=42                     # Random seed for reproducibility
)
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{arrow_pointing_dataset,
  author = {Pöppel, Korbinian},
  title = {Arrow Pointing Dataset},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ml-jku/arrow_pointing_data}
}
```
