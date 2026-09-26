# Step 13 - Container image for the Knowledge Copilot API (api/main.py).
#
# Image     = a frozen package: Linux + Python + libraries + our code. Built once.
# Container = a running instance of an image. Start, stop, delete, start again - always identical.
#
# Every instruction below creates a LAYER. Docker caches layers and rebuilds only from the first
# layer whose input changed. So: things that change rarely (Python, libraries) come first,
# things that change often (our code) come last. Editing api/main.py then rebuilds in seconds.
#
# Build:   docker build -t knowledge-copilot-api .
# Run:     see the commands in the chat / README

FROM python:3.12-slim

# PYTHONUNBUFFERED: print log lines immediately (otherwise `docker logs` shows them late)
# HF_HOME:          where Hugging Face stores downloaded models -> we mount a volume there
# LLM_MODEL / RERANKER: defaults, overridable at run time with -e (same variables as in Step 12)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
    RERANKER=large

WORKDIR /app

# 1) PyTorch, CPU-only build. A plain `pip install torch` on Linux pulls the CUDA version with
#    several GB of NVIDIA libraries that a CPU container never uses.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) All other dependencies. torch is already installed, so pip keeps the CPU version.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Our code and the knowledge base - only what the API needs (no PDFs, eval, results).
COPY src/ src/
COPY api/ api/
COPY data/chunks.jsonl data/chunks.jsonl

# 4) Do not run as root: if someone ever broke into the app, they would not own the container.
RUN useradd --create-home appuser \
    && mkdir -p /cache/huggingface \
    && chown -R appuser /cache
USER appuser

EXPOSE 8000

# Docker asks /health every 30 s and marks the container "healthy" or "unhealthy".
# start-period is long because the first start downloads the models.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# --host 0.0.0.0: listen on ALL network interfaces of the container. With the default 127.0.0.1
# the server would only accept connections from inside the container itself - not from your browser.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
