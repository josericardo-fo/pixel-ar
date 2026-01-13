import os
import json
import time
import base64
import io
import logging
import torch
import redis
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SWITTI] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- IMPORTAÇÃO ESPECÍFICA DO SWITTI ---
# Baseado no README deles: "from models import SwittiPipeline"
try:
    from models import SwittiPipeline
except ImportError:
    # Fallback caso a estrutura de pastas seja diferente
    try:
        from switti.models import SwittiPipeline
    except ImportError:
        logger.critical("ERRO: Não foi possível importar 'SwittiPipeline'. Verifique se 'models/switti' foi copiado corretamente.")
        exit(1)

# Variáveis de Ambiente
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/0')
QUEUE_NAME = os.getenv('QUEUE_NAME', 'switti_queue')
MODEL_PATH = os.getenv('MODEL_PATH', 'yresearch/Switti-1024') # Valor default corrigido
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/data/outputs/switti')
DEVICE = os.getenv('DEVICE', 'cuda')
DTYPE_STR = os.getenv('DTYPE', 'bf16')

# Configura precisão
if DTYPE_STR == 'bf16' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float16

os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # 1. Conexão Redis
    try:
        r = redis.from_url(REDIS_URL, decode_responses=True)
        r.ping()
    except Exception as e:
        logger.error(f"Erro Redis: {e}")
        exit(1)

    # 2. Carregamento do Modelo
    logger.info(f"Carregando Switti de '{MODEL_PATH}'...")
    try:
        pipe = SwittiPipeline.from_pretrained(
            MODEL_PATH, 
            device=DEVICE, 
            torch_dtype=TORCH_DTYPE
        )
        logger.info("Modelo carregado com sucesso!")
    except Exception as e:
        logger.critical(f"Erro ao carregar modelo: {e}")
        exit(1)

    logger.info(f"Worker SWITTI ouvindo: {QUEUE_NAME}")

    while True:
        task = r.blpop(QUEUE_NAME, timeout=5)
        if not task: continue
        
        _, data_str = task
        try:
            d = json.loads(data_str)
            task_id = d.get('id')
            prompt = d.get('prompt')

            if not prompt: continue
            
            logger.info(f"[{task_id}] Gerando: {prompt[:40]}...")
            start = time.time()

            # --- INFERÊNCIA COM PARÂMETROS DO README ---
            # O Switti tem parâmetros específicos para melhorar a qualidade
            # Você pode tornar esses valores dinâmicos lendo do JSON da task (d.get...)
            with torch.no_grad():
                images = pipe(
                    [prompt], # O pipe espera uma lista
                    cfg=d.get('cfg', 6.0),
                    top_k=d.get('top_k', 400),
                    top_p=d.get('top_p', 0.95),
                    more_smooth=d.get('more_smooth', True),
                    return_pil=True,
                    # Parâmetros avançados sugeridos no README:
                    smooth_start_si=2,
                    turn_on_cfg_start_si=0,
                    turn_off_cfg_start_si=11,
                    last_scale_temp=0.1,
                    seed=d.get('seed', None) # Se None, é aleatório
                )
                
                # Pega a primeira imagem
                img = images[0]

            # Salvar e Converter
            filename = f"{task_id}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            img.save(filepath)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            result = {
                "id": task_id,
                "status": "completed",
                "image_path": filepath,
                "image_base64": b64,
                "model": "switti-1024"
            }
            r.set(f"result:{task_id}", json.dumps(result))
            r.expire(f"result:{task_id}", 3600)
            
            logger.info(f"[{task_id}] Concluído em {time.time() - start:.2f}s")

        except Exception as e:
            logger.error(f"Erro task: {e}")
            # Opcional: reportar erro no Redis

if __name__ == "__main__":
    main()