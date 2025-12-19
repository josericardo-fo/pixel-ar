#!/usr/bin/env bash
set -euo pipefail

PROMPT='A realistic stone statue of a male figure, carved from coarse-grained grey granite. The texture is rough, granular, with visible speckles of quartz and feldspar catching the light. Weathered surface, soft natural lighting, neutral blurred background. Serene expression. High resolution close-up emphasizing the grain.'

# Ajustes opcionais (podem ser sobrescritos via env)
# Quantidade de imagens geradas por job
PARALLEL_SIZE="${PARALLEL_SIZE:-10}"
GUIDANCE="${GUIDANCE:-5}"
TEMPERATURE="${TEMPERATURE:-1}"
WIDTH="${WIDTH:-384}"
HEIGHT="${HEIGHT:-384}"
TIMEOUT_SECS="${TIMEOUT_SECS:-900}"

# Filas (por modelo)
QUEUE_PRO="queue:janus-pro-1b"
RESULTS_PRO="results:janus-pro-1b"
QUEUE_JANUS="queue:janus-1.3b"
RESULTS_JANUS="results:janus-1.3b"

# Gera job JSON de forma segura (sem sofrer com escaping do prompt)
make_job_json() {
  local job_id="$1"
  PROMPT="$PROMPT" \
  PARALLEL_SIZE="$PARALLEL_SIZE" \
  GUIDANCE="$GUIDANCE" \
  TEMPERATURE="$TEMPERATURE" \
  WIDTH="$WIDTH" \
  HEIGHT="$HEIGHT" \
  python3 - "$job_id" <<'PY'
import json
import os
import sys

job_id = sys.argv[1]
prompt = os.environ["PROMPT"]

job = {
    "job_id": job_id,
    "input": {
        "task": "text_to_image",
        "prompt": prompt,
        "parallel_size": int(os.environ["PARALLEL_SIZE"]),
        "guidance": float(os.environ["GUIDANCE"]),
        "temperature": float(os.environ["TEMPERATURE"]),
        "width": int(os.environ["WIDTH"]),
        "height": int(os.environ["HEIGHT"]),
    },
}

print(json.dumps(job, ensure_ascii=False))
PY
}

JOB_ID_PRO="t2i_pro_$(date +%Y%m%d_%H%M%S)"
JOB_ID_JANUS="t2i_janus_$(date +%Y%m%d_%H%M%S)"

JOB_JSON_PRO="$(make_job_json "$JOB_ID_PRO")"
JOB_JSON_JANUS="$(make_job_json "$JOB_ID_JANUS")"

echo "Enfileirando..."
echo "- Pro 1B:   $QUEUE_PRO (job_id=$JOB_ID_PRO)"
echo "- Janus:    $QUEUE_JANUS (job_id=$JOB_ID_JANUS)"

# Enfileira nos dois modelos
docker exec janus-redis redis-cli RPUSH "$QUEUE_PRO" "$JOB_JSON_PRO" >/dev/null
docker exec janus-redis redis-cli RPUSH "$QUEUE_JANUS" "$JOB_JSON_JANUS" >/dev/null

echo "Aguardando resultados (timeout=${TIMEOUT_SECS}s)..."

echo "\n[RESULT Pro 1B]"
docker exec janus-redis redis-cli BLPOP "$RESULTS_PRO" "$TIMEOUT_SECS"

echo "\n[RESULT Janus 1.3B]"
docker exec janus-redis redis-cli BLPOP "$RESULTS_JANUS" "$TIMEOUT_SECS"

echo "\nDica: as imagens também ficam em ./outputs (volume montado)."
