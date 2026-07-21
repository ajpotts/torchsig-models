# TorchSig Models Docker

This directory contains Docker configurations for running TorchSig Models with GPU support.

## Available Dockerfiles

| File | Description | GPU Support | Size |
|------|-------------|-------------|------|
| [`Dockerfile`](Dockerfile) | **Main Dockerfile** - Optimized multi-stage build with CUDA 11.8, Python 3.10, and all dependencies. Reduced image size through efficient layer caching and multi-stage builds. | ✅ Yes | ~15-18 GB |

## Quick Start

### Build the Docker Image

```bash
# From the repository root
docker build -t torchsig-models -f docker/Dockerfile .
```

**Note:** The optimized image size is approximately **15-18 GB** (reduced from 20.5 GB) due to:
- Multi-stage build removing build dependencies
- Efficient layer caching
- Minimal runtime dependencies

### Run the Container

#### With GPU Support (Recommended)

```bash
# Run with all available GPUs
docker run --gpus all -it torchsig-models

# Run with specific GPUs (e.g., GPU 0 and 1)
docker run --gpus '"device=0,1"' -it torchsig-models
```

#### Without GPU (CPU-only)

```bash
docker run -it torchsig-models
```

## Dockerfile Features

The optimized [`Dockerfile`](Dockerfile) uses a **multi-stage build** approach:

### Stage 1: Builder
- **Base:** `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- **Purpose:** Install build dependencies and compile packages
- **Includes:**
  - Python 3.10 with development tools
  - Build essentials (gcc, make, cmake)
  - Git and curl
  - **uv** for faster pip installations
  - Library dependencies (libffi-dev, libssl-dev)
- **Result:** Virtual environment with all dependencies pre-built using uv

### Stage 2: Runtime
- **Base:** `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` (clean)
- **Purpose:** Minimal image with only runtime dependencies
- **Includes:**
  - Python 3.10 runtime
  - **uv** for faster pip operations
  - Minimal system libraries (libgl1, libsm6, libxrender1, etc.)
  - Virtual environment copied from builder stage
  - Application code
- **Result:** Production-ready image with uv available

### Optimization Benefits

| Optimization | Benefit |
|--------------|---------|
| Multi-stage build | Removes unnecessary build tools from final image |
| Separate dependency copying | Better layer caching for faster rebuilds |
| uv in final image | Faster pip operations available at runtime |
| uv for pip installations | Faster dependency resolution and installation |
| Clean apt cache | Reduces image size |
| Virtual environment | Isolated Python environment |
| .dockerignore | Excludes unnecessary files from image |

### Environment Configuration
- `PYTHONDONTWRITEBYTECODE=1` - Prevents `.pyc` files
- `PYTHONUNBUFFERED=1` - Ensures Python output is not buffered
- `PYTHONPATH=/workspace:${PYTHONPATH}` - Proper Python path

### Package Installation
- TorchSig Models installed in **editable mode** with dev dependencies
- **OpenCV fix:** Uses `opencv-python-headless==4.12.0.88` for ultralytics compatibility

### Verification
The Dockerfile includes a verification step that checks:
- PyTorch installation and CUDA availability
- TorchSig installation
- TorchSig Models installation

## Usage Examples

### Basic Usage

```bash
# Build the image
docker build -t torchsig-models -f docker/Dockerfile .

# Start a container
docker run --gpus all -it torchsig-models

# Inside the container, you can:
python -c "import torchsig_models; print(torchsig_models.__version__)"
```

### Mounting Local Data

```bash
# Mount a local directory for data access
docker run --gpus all -it \
  -v /path/to/local/data:/workspace/data \
  torchsig-models
```

### Running Tests Inside Container

```bash
# Build and run tests
docker build -t torchsig-models -f docker/Dockerfile .
docker run --gpus all -it torchsig-models pytest tests/ -v
```

### Using Jupyter Notebook

```bash
# Start a container with Jupyter
docker run --gpus all -it \
  -p 8888:8888 \
  torchsig-models

# Inside the container:
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Development Mode

```bash
# Mount the current directory for development
docker run --gpus all -it \
  -v $(pwd):/workspace \
  torchsig-models bash

# Changes made locally will be reflected in the container
# Note: For full rebuild after code changes, rebuild the image
```

## Customization

### Using a Different Python Version

To use a different Python version, modify both stages in the Dockerfile:

```dockerfile
# In both Stage 1 and Stage 2, change:
RUN apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-pip \
    python3.11-venv \
    python-is-python3
```

### Using a Different CUDA Version

To use a different CUDA version, change the base image in both stages:

```dockerfile
# For CUDA 12.1
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as builder
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# For CUDA 11.7
FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu22.04 as builder
FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu22.04
```

### Installing Additional System Packages

Add packages to the runtime stage (Stage 2):

```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    # Add your packages here
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*
```

### Installing Additional Python Packages

Add packages to the builder stage (Stage 1) before the venv copy:

```dockerfile
# In Stage 1, after pip install -e ".[dev]"
RUN pip install pandas numpy
```

## Build Optimization Tips

### Use Build Cache

Docker automatically caches layers. To maximize cache usage:

```bash
# Copy only dependency files first (already done in Dockerfile)
# Then copy the rest

# For development, use:
docker build --cache-from torchsig-models -t torchsig-models -f docker/Dockerfile .
```

### Use BuildKit

Enable BuildKit for faster builds and better caching:

```bash
DOCKER_BUILDKIT=1 docker build -t torchsig-models -f docker/Dockerfile .
```

### Squash Layers (Optional)

To reduce the number of layers (not recommended for development):

```bash
docker build --squash -t torchsig-models -f docker/Dockerfile .
```

## Troubleshooting

### CUDA Not Available

**Problem:** `torch.cuda.is_available()` returns `False`

**Solution:**
1. Ensure you have NVIDIA drivers installed on your host machine
2. Use `--gpus all` flag when running the container
3. Check with `nvidia-smi` that GPUs are visible
4. Verify NVIDIA Container Toolkit is installed

```bash
# Check NVIDIA drivers
nvidia-smi

# Check NVIDIA Container Toolkit
nvidia-container-cli --version

# Run with GPU support
docker run --gpus all -it torchsig-models
```

### Permission Issues

**Problem:** Permission denied when mounting volumes

**Solution:** Use the `--user` flag or adjust permissions:

```bash
# Run as current user
docker run --gpus all -it --user $(id -u):$(id -g) torchsig-models

# Or change permissions on the mounted directory
chmod -R a+rw /path/to/local/data
```

### Out of Memory

**Problem:** Docker build fails due to memory constraints

**Solution:**
1. Increase Docker memory allocation in Docker Desktop settings
2. Use `--memory` flag for docker run
3. Use BuildKit for more efficient memory usage

```bash
# Set Docker to use more memory (in Docker Desktop)
# Or use BuildKit
DOCKER_BUILDKIT=1 docker build -t torchsig-models -f docker/Dockerfile .
```

### Build Fails Due to Network Issues

**Problem:** pip install fails due to network timeouts

**Solution:**
1. Use a mirror or proxy
2. Increase timeout
3. Retry the build

```bash
# Use a pip mirror
# Edit Dockerfile and add --index-url to pip commands
# Or use a local pip cache
```

### Slow Dependency Installation

**Problem:** pip installation is slow

**Solution:** The Dockerfile already uses caching. For faster rebuilds:
- Only change files that are copied after the dependency installation
- Use BuildKit
- Consider using a local pip cache

## Docker Compose (Optional)

For more complex setups, you can create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  torchsig-models:
    build:
      context: .
      dockerfile: docker/Dockerfile
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    volumes:
      - ./data:/workspace/data
      - ./output:/workspace/output
    ports:
      - "8888:8888"
    tty: true
    stdin_open: true
```

Run with:

```bash
docker-compose build
docker-compose up -d
docker-compose exec torchsig-models bash
```

## Cleanup

### Remove Docker Image

```bash
# List images
docker images

# Remove specific image
docker rmi torchsig-models

# Remove all unused images
docker image prune -a
```

### Remove Containers

```bash
# List containers
docker ps -a

# Remove specific container
docker rm <container_id>

# Remove all stopped containers
docker container prune
```

### Remove Volumes

```bash
# List volumes
docker volume ls

# Remove all unused volumes
docker volume prune
```

### Remove Build Cache

```bash
# Remove build cache
docker builder prune
```

## Best Practices

1. **Use Tags**: Always tag your Docker images with version numbers
   ```bash
   docker build -t torchsig-models:1.0.0 -f docker/Dockerfile .
   ```

2. **Clean Up**: Regularly clean up unused images and containers
   ```bash
   docker system prune
   ```

3. **Use .dockerignore**: Create a `.dockerignore` file to exclude unnecessary files
   ```text
   .venv
   .git
   __pycache__
   *.pyc
   *.pt
   *.h5
   .pytest_cache
   ```

4. **Security**: Don't run containers as root in production

5. **Minimal Images**: Use the optimized multi-stage build for production

6. **Cache Dependencies**: The Dockerfile is already optimized for caching

## Image Size Comparison

| Configuration | Size | Notes |
|---------------|------|-------|
| Original single-stage | ~20.5 GB | Included build tools in final image |
| **Optimized multi-stage** | **~15-18 GB** | **Build tools removed, minimal runtime** |

The optimization reduces the image size by **2-5 GB** while maintaining all functionality.

## References

- [NVIDIA CUDA Docker Images](https://hub.docker.com/r/nvidia/cuda)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)
- [Docker BuildKit](https://docs.docker.com/build/buildkit/)

## Support

For issues or questions related to Docker configurations:
- Check the [main README](../README.md) for general information
- Open an issue on [GitHub](https://github.com/TorchDSP/torchsig-models/issues)

---

**Last Updated:** June 2025

**Maintainers:** TorchSig Team
