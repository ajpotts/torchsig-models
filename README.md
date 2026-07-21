# TorchSig Models

**TorchSig Models** provides pre-trained models and utilities for working with [TorchSig](https://github.com/TorchDSP/torchsig) data. This library includes model architectures, training scripts, and adapters for integrating TorchSig with popular deep learning frameworks like YOLO and Anomalib.

## Features

- **Pre-trained models** for narrowband and wideband signal classification
- **Adapters** for YOLO, Anomalib, and other frameworks
- **Training utilities** including metrics tracking and reproducibility tools
- **Easy integration** with existing TorchSig datasets

## Getting Started

### Prerequisites

The datasets generated can be very large (562 GB for narrowband and 396 GB for wideband), so ensure your device has adequate storage before running the scripts.

- Ubuntu &ge; 22.04 (recommended)
- Hard drive storage with 1 TB (for full datasets)
- CPU with &ge; 4 cores
- GPU with &ge; 16 GB storage (recommended for training)
- Python &ge; 3.10

We recommend using Ubuntu or a Docker container for best compatibility.

### Installation

#### From Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/TorchDSP/torchsig-models.git
cd torchsig-models
pip install -e ".[dev]"
```

#### Using Docker

Build and run the Docker image with GPU support:

```bash
docker build -t torchsig-models -f docker/Dockerfile .
docker run -it --gpus all torchsig-models
```

Note: The Docker image size is ~20.5 GB.

## Docker

Run TorchSig Models within Docker by building the Docker image:
```bash
docker build -t torchsig-models -f Dockerfile .
```
Note that the Docker image size is ~20.5 GB.

## Available Models

### Narrowband Models

| Model | Description | Paper |
|-------|-------------|-------|
| `XCiTClassifier` | 1D Version of the XCiT Model for signal classification | [arXiv:2106.09681](https://arxiv.org/abs/2106.09681) |

### Wideband Models

| Model | Description | Paper |
|-------|-------------|-------|
| YOLO11n | YOLO-based detector for wideband signal detection | [arXiv:1506.02640](https://arxiv.org/abs/1506.02640) |

## Usage

### Quick Start

```python
import torchsig_models
from torchsig_models.models import XCiTClassifier
from torchsig_models.adapters import yolo_train, yolo_infer
from torchsig_models.utils import ClassifierMetricsTracker

# Initialize a model
model = XCiTClassifier(num_classes=24)

# Use training utilities
metrics_tracker = ClassifierMetricsTracker(num_classes=24)
```

### Using YOLO Adapter

```python
from pathlib import Path
from torchsig_models.adapters import yolo_train, yolo_infer

# Train a YOLO model
config_path = Path("path/to/yolo_config.yaml")
output_dir = Path("output/directory")
model_path = Path("yolo11n.pt")

yolo_train(
    model_filepath=model_path,
    config=config_path,
    output_dir=output_dir,
    epochs=25
)

# Run inference
yolo_infer(
    output_dir=output_dir,
    config=config_path,
    split="val"
)
```

### Using Anomalib Adapter

```python
from torchsig_models.adapters import TorchSigAnomalibDataset

# Create an Anomalib-compatible dataset from TorchSig data
dataset = TorchSigAnomalibDataset(
    root="path/to/torchsig/dataset",
    task="classification",
    split="train"
)
```

## Package Structure

```
torchsig-models/
├── torchsig_models/
│   ├── __init__.py          # Main package exports
│   ├── models/              # Model architectures
│   │   └── iq_models/       # IQ signal models
│   │       └── xcit/        # XCiT model implementation
│   ├── adapters/            # Framework adapters
│   │   ├── yolo_utils.py    # YOLO utilities
│   │   ├── yolo_train_detector.py
│   │   ├── yolo_inference_detector.py
│   │   └── anomalib_utils.py # Anomalib integration
│   └── util/                # Training utilities
│       ├── classifier_metrics_tracker.py
│       └── training.py      # Training helpers
├── scripts/                 # Training and inference scripts
├── tests/                   # Unit tests
└── pyproject.toml           # Package configuration
```

## Development Workflow

To simplify environment setup and maintain code quality, this project uses a `Makefile`. This provides a standardized set of shortcuts for common development tasks, ensuring consistency across different environments.

### Common Commands

| Command | Description | Tool Used |
| :--- | :--- | :--- |
| `make install` | Installs dependencies and the package in editable mode. | `pip` |
| `make test` | Runs the full suite of tests. | `pytest` |
| `make test-cov` | Runs tests and generates a detailed coverage report. | `pytest-cov` |
| `make test-notebooks` | Executes all Jupyter notebooks to verify they run without errors. | `jupyter` |
| `make test-notebooks-clean` | Removes stamp files created by notebook execution. | `shell` |
| `make clean-notebooks` | Removes all output from executed notebooks. | `jupyter` |
| `make lint` | Performs static analysis to find bugs and style issues. | `ruff` |
| `make format` | Automatically formats the codebase to project standards. | `ruff` |
| `make fix` | Automatically fixes linting errors and formats the code. | `ruff` |
| `make clean` | Wipes `__pycache__`, test caches, and `/tmp` artifacts. | `shell` |

For a full list of available targets and descriptions, run:
```bash
make help
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use TorchSig Models in your research, please cite:

```bibtex
@misc{torchsig-models,
  author = {TorchSig Team},
  title = {TorchSig Models: Pre-trained models for signal processing},
  year = {2026},
  url = {https://github.com/TorchDSP/torchsig-models},
}
```

## Support

For questions, issues, or feature requests, please open an issue on our [GitHub repository](https://github.com/TorchDSP/torchsig-models/issues).

