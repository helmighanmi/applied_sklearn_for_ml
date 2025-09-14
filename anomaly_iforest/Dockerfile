# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# (Optional) system deps; you can remove build-essential if wheels cover your platform
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Leverage layer caching: install requirements first
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY README.md .

# Non-root user for safety
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Mount points for data/models at runtime
VOLUME ["/app/data", "/app/models"]

# Default command can be overridden on `docker run`
ENTRYPOINT ["python"]
CMD ["scripts/inference.py", "--data", "data/new_data.csv", "--config", "configs/config.yaml", "--out", "scored.csv"]

# ---------- (Optional) Notebook target ----------
FROM base AS notebook
RUN pip install jupyterlab
EXPOSE 8888
CMD ["python", "-m", "jupyterlab", "--ip=0.0.0.0", "--no-browser", "--NotebookApp.token=", "--NotebookApp.password="]
