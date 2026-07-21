# TorchSig Models Dockerfile
# Optimized GPU-enabled image for running torchsig-models with CUDA support
# Multi-stage build to reduce final image size

# ============================================================================
# Stage 1: Builder - Install build dependencies and compile packages
# ============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS builder

# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures Python output is not buffered
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    build-essential \
    cmake \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency resolution
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install build tools
RUN python3.10 -m venv /opt/venv
ENV PATH="/root/.local/bin:/opt/venv/bin:${PATH}"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel build

# Copy only requirements first for better caching
COPY pyproject.toml .
COPY torchsig_models/ ./torchsig_models/

# Install dependencies in builder stage using uv for speed
RUN uv pip install --no-cache-dir --upgrade pip && \
    uv pip install -e ".[dev]" && \
    # Fix OpenCV version for ultralytics compatibility
    uv pip uninstall -y opencv-python opencv-python-headless && \
    uv pip install --no-cache-dir opencv-python-headless==4.12.0.88

# ============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# ============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Ensures Python output is not buffered
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    curl \
    make \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb1 \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code (excluding files in .dockerignore)
COPY . .

# Install uv and packages in a single RUN to ensure PATH is set correctly
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:/opt/venv/bin:${PATH}" && \
    uv pip install --no-cache-dir -e ".[dev]" && \
    # Ensure correct OpenCV version
    uv pip uninstall -y opencv-python opencv-python-headless && \
    uv pip install --no-cache-dir opencv-python-headless==4.12.0.88

# Set PATH for subsequent commands
ENV PATH="/root/.local/bin:/opt/venv/bin:${PATH}"

# Set environment variables
ENV PYTHONPATH=/workspace

# ============================================================================
# Verification
# ============================================================================
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && \
    python -c "import torchsig; print(f'TorchSig: {torchsig.__version__}')" && \
    python -c "import torchsig_models; print(f'TorchSig Models: {torchsig_models.__version__}')"

# ============================================================================
# Default Command
# ============================================================================
CMD ["bash"]
