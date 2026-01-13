set -euo pipefail

# --- CONFIGURAÇÕES ---

# Container do Redis (O mesmo usado pelo Janus no seu exemplo)
REDIS_CONTAINER="${REDIS_CONTAINER:-janus-redis}"

# Nome da fila definida no docker-compose do Switti
QUEUE_NAME="switti_queue"

# Prompt Padrão
PROMPT='A realistic stone statue of a male figure, carved from coarse-grained grey granite. The texture is rough, granular, with visible speckles of quartz and feldspar catching the light. Weathered surface, soft natural lighting, neutral blurred background. Serene expression. High resolution close-up emphasizing the grain.'

# Ajustes do Switti (Baseados no README oficial)
# CFG Scale (Classifier Free Guidance)
CFG="${CFG:-6.0}" 
# Seed (Para reprodução). Se deixar vazio, o Python gera aleatório.
SEED="${SEED:-42}" 
# Timeout para desistir de esperar
TIMEOUT_SECS="${TIMEOUT_SECS:-300}"

# --- FUNÇÕES ---

# Gera o JSON seguro para o Switti
make_job_json() {
  local job_id="$1"
  
  # Passamos variáveis para o Python via env para evitar problemas de aspas no bash
  PROMPT="$PROMPT" \
  CFG="$CFG" \
  SEED="$SEED" \
  python3 - "$job_id" <<'PY'
import json
import os
import sys

job_id = sys.argv[1]
prompt = os.environ["PROMPT"]
cfg = float(os.environ["CFG"])
seed_env = os.environ.get("SEED", "")

job = {
    "id": job_id,
    "prompt": prompt,
    "cfg": cfg,
    # Parâmetros extras sugeridos pelo Switti
    "top_k": 400,
    "top_p": 0.95,
    "more_smooth": True
}

if seed_env:
    job["seed"] = int(seed_env)

print(json.dumps(job, ensure_ascii=False))
PY
}

# --- EXECUÇÃO ---

# Gera um ID único
JOB_ID="switti_task_$(date +%Y%m%d_%H%M%S)"
JOB_JSON="$(make_job_json "$JOB_ID")"

echo "=============================================="
echo "🚀 Enviando Job para o SWITTI"
echo "ID:       $JOB_ID"
echo "Fila:     $QUEUE_NAME"
echo "Prompt:   ${PROMPT:0:50}..."
echo "Container:$REDIS_CONTAINER"
echo "=============================================="

# Envia para o Redis
docker exec "$REDIS_CONTAINER" redis-cli RPUSH "$QUEUE_NAME" "$JOB_JSON" >/dev/null

echo "⏳ Aguardando resultado (verificando result:$JOB_ID)..."

# Loop de espera (Polling)
# Diferente do BLPOP, aqui verificamos se a chave existe porque o Switti usa SET
START_TIME=$(date +%s)
while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    if [ "$ELAPSED" -gt "$TIMEOUT_SECS" ]; then
        echo "❌ Timeout atingido ($TIMEOUT_SECS segundos)."
        exit 1
    fi

    # Tenta pegar o resultado
    RESULT=$(docker exec "$REDIS_CONTAINER" redis-cli GET "result:$JOB_ID")

    if [ "$RESULT" != "" ]; then
        echo ""
        echo "✅ Sucesso! Imagem gerada em ${ELAPSED}s."
        
        # Opcional: Extrair o caminho da imagem do JSON de resposta usando python/jq
        IMG_PATH=$(echo "$RESULT" | python3 -c "import sys, json; print(json.load(sys.stdin).get('image_path', 'Desconhecido'))")
        echo "📂 Caminho no container: $IMG_PATH"
        echo "🖼️  Verifique sua pasta ./outputs/switti"
        break
    fi

    # Barra de progresso simples
    printf "."
    sleep 1
done