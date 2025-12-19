import json
import os
import signal
import sys
import time
from typing import Any, Dict, Optional

import redis

from janus_worker.model_runner import (
    load_model_context,
    run_multimodal_understanding,
    run_text_to_image,
)


JsonDict = Dict[str, Any]


def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Variável de ambiente obrigatória não definida: {name}")
    return v


def _parse_job(raw: str) -> JsonDict:
    try:
        job = json.loads(raw)
    except Exception as e:
        raise ValueError(f"job não é JSON válido: {e}")

    if not isinstance(job, dict):
        raise ValueError("job deve ser um objeto JSON")

    if "job_id" not in job:
        raise ValueError("job deve conter 'job_id'")
    if "input" not in job:
        raise ValueError("job deve conter 'input'")

    return job


def _result_ok(job_id: str, output: Any) -> str:
    return json.dumps({"job_id": job_id, "status": "ok", "output": output}, ensure_ascii=False)


def _result_err(job_id: str, error: str) -> str:
    return json.dumps({"job_id": job_id, "status": "error", "error": error}, ensure_ascii=False)


def main() -> int:
    model_name = _env("MODEL_NAME", "janus")
    model_path = _env("MODEL_PATH", "deepseek-ai/Janus-1.3B")
    redis_url = _env("REDIS_URL", "redis://redis:6379/0")

    queue_key = f"queue:{model_name}"
    results_key = f"results:{model_name}"

    out_dir = _env("OUTPUT_DIR", "/data/outputs")

    # Configs opcionais
    device = os.getenv("DEVICE")  # ex: cuda, cpu, cuda:0
    dtype = os.getenv("DTYPE")  # ex: bf16, fp16, fp32
    attn_impl = os.getenv("ATTN_IMPL", "eager")

    # Padrões do job para T2I
    default_parallel_size = int(os.getenv("T2I_PARALLEL_SIZE", "4"))
    default_guidance = float(os.getenv("T2I_GUIDANCE", "5"))
    default_temperature = float(os.getenv("T2I_TEMPERATURE", "1"))
    default_width = int(os.getenv("T2I_WIDTH", "384"))
    default_height = int(os.getenv("T2I_HEIGHT", "384"))

    # Padrões do job para VLM
    default_max_new_tokens = int(os.getenv("VLM_MAX_NEW_TOKENS", "512"))

    stop = {"value": False}

    def _handle_sigterm(_signum, _frame):
        stop["value"] = True

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    print(
        f"[worker] starting | model_name={model_name} model_path={model_path} queue={queue_key} results={results_key}",
        flush=True,
    )

    # Carrega modelo uma única vez
    ctx = load_model_context(
        model_name=model_name,
        model_path=model_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )

    print(
        f"[worker] model loaded | device={ctx.device} dtype={ctx.dtype} sft_format={ctx.processor.sft_format}",
        flush=True,
    )

    r = redis.Redis.from_url(redis_url, decode_responses=True)

    # Loop principal
    while not stop["value"]:
        try:
            # BRPOP é bloqueante; usa timeout curto para checar sinal de parada
            item = r.brpop(queue_key, timeout=2)
            if item is None:
                continue

            _key, raw = item
            job_id = "unknown"
            try:
                job = _parse_job(raw)
                job_id = str(job["job_id"])
                job_input = job["input"]

                # input pode ser str (atalho) ou dict
                task = None
                payload: JsonDict
                if isinstance(job_input, str):
                    task = "text_to_image"
                    payload = {"prompt": job_input}
                elif isinstance(job_input, dict):
                    task = str(job_input.get("task") or job_input.get("type") or "text_to_image")
                    payload = job_input
                else:
                    raise ValueError("input deve ser string ou objeto")

                if task in {"text_to_image", "t2i"}:
                    prompt = str(payload.get("prompt") or payload.get("text") or "")
                    if not prompt:
                        raise ValueError("T2I requer 'prompt'")

                    seed = payload.get("seed")
                    seed_int = int(seed) if seed is not None else None

                    paths = run_text_to_image(
                        ctx,
                        prompt=prompt,
                        out_dir=out_dir,
                        seed=seed_int,
                        guidance=float(payload.get("guidance", default_guidance)),
                        temperature=float(payload.get("temperature", default_temperature)),
                        parallel_size=int(payload.get("parallel_size", default_parallel_size)),
                        width=int(payload.get("width", default_width)),
                        height=int(payload.get("height", default_height)),
                    )
                    r.lpush(results_key, _result_ok(job_id, {"images": paths}))

                elif task in {"multimodal_understanding", "vlm", "vqa"}:
                    question = str(payload.get("question") or payload.get("prompt") or "")
                    if not question:
                        raise ValueError("VLM requer 'question' (ou 'prompt')")

                    image = payload.get("image")
                    if image is None:
                        # também aceita lista para manter compatibilidade
                        images = payload.get("images")
                        if isinstance(images, list) and images:
                            image = images[0]
                    if image is None:
                        raise ValueError("VLM requer 'image' (path ou data:image base64)")

                    answer = run_multimodal_understanding(
                        ctx,
                        question=question,
                        image=image,
                        max_new_tokens=int(payload.get("max_new_tokens", default_max_new_tokens)),
                        do_sample=bool(payload.get("do_sample", False)),
                        temperature=float(payload.get("temperature", 0.0)),
                        top_p=float(payload.get("top_p", 0.95)),
                    )
                    r.lpush(results_key, _result_ok(job_id, {"text": answer}))

                else:
                    raise ValueError(f"task desconhecida: {task}")

            except Exception as e:
                r.lpush(results_key, _result_err(job_id, str(e)))

        except redis.exceptions.ConnectionError as e:
            print(f"[worker] redis connection error: {e} (retrying)", file=sys.stderr, flush=True)
            time.sleep(1)
        except Exception as e:
            print(f"[worker] unexpected error: {e}", file=sys.stderr, flush=True)
            time.sleep(0.2)

    print("[worker] shutdown", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
