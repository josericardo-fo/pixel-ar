# pixel-ar

Projeto com workers de inferência por modelo (um container por modelo) orquestrados via Docker Compose e filas no Redis.

## Visão geral

- Cada modelo roda em um **worker dedicado** (container separado).
- O Redis atua como **broker** de filas.
- As imagens geradas são gravadas em `outputs/` (montado como volume).

## Serviços (docker-compose)

- `redis`: broker de filas.
- `janus-pro-1b-worker` e `janus-3b-worker`: workers do Janus (modelos diferentes), consumindo `queue:{MODEL_NAME}` e escrevendo resultados em `results:{MODEL_NAME}`.
- `switti-worker`: worker do Switti (fila configurável via `QUEUE_NAME`).

## Como subir

Build + start:

`docker compose up --build`

Para GPU:

- Linux: requer NVIDIA Container Toolkit.
- macOS: Docker Desktop não oferece CUDA nativa; use uma máquina Linux/VM com GPU para execução real.

## Variáveis de ambiente úteis

Janus:

- `MODEL_NAME`, `MODEL_PATH`, `DEVICE`, `DTYPE`, `ATTN_IMPL`, `OUTPUT_DIR`

Switti:

- `MODEL_PATH`, `QUEUE_NAME`, `DEVICE`, `DTYPE`, `OUTPUT_DIR`
