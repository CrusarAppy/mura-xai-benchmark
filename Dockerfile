# Reproducible environment for the XAI-MURA benchmark (proposal 3.10.6 / 4.2).
# CUDA 12.1 + PyTorch base so training and explanation run on GPU identically to Kaggle/Colab.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /workspace

# Pinned scientific stack (torch/torchvision come from the base image).
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

# Install the benchmark package.
COPY . /workspace
RUN pip install --no-cache-dir -e .

# Default: run the smoke tests to verify the image.
CMD ["python", "-m", "pytest", "-q", "tests/"]
