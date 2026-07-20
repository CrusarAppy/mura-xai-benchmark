# Reproducible environment for the XAI benchmark.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e .

COPY . .
CMD ["python", "scripts/run_experiment.py", "--config", "configs/densenet_gradcam.yaml"]
