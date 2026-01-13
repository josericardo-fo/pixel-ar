# workers/switti/Dockerfile
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala Git e libs gráficas
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HOME="/data/hf"
RUN mkdir -p /data/hf && chmod 777 /data/hf

WORKDIR /app

# --- 1. Copia o MODELO ---
COPY models/switti /app/switti

# Remove o .git quebrado do submódulo para não confundir
RUN rm -rf /app/switti/.git

# Instala dependências do Switti
WORKDIR /app/switti
RUN pip install --no-cache-dir -r requirements.txt

# Adiciona ao PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app/switti"

# --- 2. Copia o WORKER ---
WORKDIR /app

COPY workers/switti/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY workers/switti/src .

# --- CORREÇÃO FINAL DO GIT ---
# Inicializamos o git na pasta /app (onde o python roda)
# Isso engana o arg_util.py independente de onde ele esteja
RUN git config --global --add safe.directory /app && \
    git init && \
    git config user.email "docker@dummy.com" && \
    git config user.name "Docker Worker" && \
    git add . && \
    git commit -m "Dummy commit para satisfazer arg_util.py"

CMD ["python", "main.py"]