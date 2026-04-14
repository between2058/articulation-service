# =============================================================================
# Articulation Service — Docker Image
#
# CPU-only Python 3.10 image for GLB parsing + USD export.
# No GPU needed — geometry/material operations only.
#
# Build:  docker build -t articulation-service:latest .
# Run:    docker run -p 52071:52071 articulation-service:latest
# =============================================================================

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    http_proxy=http://proxy.intra:80 \
    https_proxy=http://proxy.intra:80 \
    no_proxy=localhost,127.0.0.1

# System deps for Pillow / trimesh / OpenGL (headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/     /app/app/
COPY scripts/ /app/scripts/

# Create runtime directories
RUN mkdir -p /app/outputs /app/outputs/textures /app/uploads /app/logs

EXPOSE 52071

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=30s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:52071/api/health')" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "52071", \
     "--workers", "1", \
     "--log-level", "info"]
