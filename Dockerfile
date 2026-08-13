# Dockerfile for the main Voice RAG application
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install Python and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Upgrade pip and install setuptools/wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install dependencies first (for docker caching)
COPY pyproject.toml .
RUN pip install -e .

# Copy application code
COPY . .

# Expose API port
EXPOSE 8080

# Run FastAPI server
CMD ["python", "-m", "api.server"]
