# Base image matches the architecture plan's PyTorch framework choice
# (used for the optional Appendix B.1 finetuning path / HFLocalClient backend).
# The core ReAct evaluation loop itself is API-driven and does not require a GPU.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# System deps
RUN apt-get update && apt-get install -y git wget build-essential curl && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project
WORKDIR /workspace
COPY . .
RUN pip install -e .

# Default: show help
CMD ["python", "evaluate.py", "--help"]
