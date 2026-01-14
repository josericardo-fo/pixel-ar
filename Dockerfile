FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# Dependências de sistema comuns para PIL/OpenCV-like stacks
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copia o projeto
COPY . /app

# Instala dependências Python (somente worker; evita deps de demo)
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-worker.txt

# flash-attn é opcional (há um wheel no repo, mas pode não bater com o Python do container)
RUN if ls /app/flash_attn-*.whl >/dev/null 2>&1; then \
    pip install --no-cache-dir /app/flash_attn-*.whl || true; \
    fi

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf \
    TRANSFORMERS_CACHE=/data/hf/transformers \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONPATH=/app

CMD ["python", "-m", "janus_worker.redis_worker"]
