import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from janus.utils.conversation import get_conv_template


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ModelContext:
    model_name: str
    model_path: str
    device: torch.device
    dtype: torch.dtype
    tokenizer: Any
    processor: VLChatProcessor
    model: MultiModalityCausalLM


def _pick_device(device_str: Optional[str]) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pick_dtype(dtype_str: Optional[str], device: torch.device) -> torch.dtype:
    if dtype_str:
        v = dtype_str.lower().strip()
        if v in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if v in {"fp16", "float16"}:
            return torch.float16
        if v in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"dtype inválido: {dtype_str}")

    # Padrão: bf16 em CUDA, fp32 em CPU (mais compatível)
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def load_model_context(
    *,
    model_name: str,
    model_path: str,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
) -> ModelContext:
    dev = _pick_device(device)
    dt = _pick_dtype(dtype, dev)

    # Alguns modelos exigem configurar a implementação de atenção
    config = AutoConfig.from_pretrained(model_path)
    if hasattr(config, "language_config") and config.language_config is not None:
        language_config = config.language_config
        if attn_implementation:
            language_config._attn_implementation = attn_implementation
    else:
        language_config = None

    processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        language_config=language_config,
    )
    model = model.to(dev)
    if dev.type == "cuda":
        model = model.to(dt)
    model.eval()

    return ModelContext(
        model_name=model_name,
        model_path=model_path,
        device=dev,
        dtype=dt,
        tokenizer=tokenizer,
        processor=processor,
        model=model,
    )


@torch.inference_mode()
def run_multimodal_understanding(
    ctx: ModelContext,
    *,
    question: str,
    image: Union[str, JsonDict],
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 0.95,
) -> str:
    # Reutiliza util do repo, que suporta path e data:image;base64
    conv = get_conv_template(ctx.processor.sft_format)
    user_role = conv.roles[0]
    assistant_role = conv.roles[1]
    conversation = [
        {
            "role": user_role,
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": assistant_role, "content": ""},
    ]
    pil_images = load_pil_images(conversation)

    prepare_inputs = ctx.processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
    ).to(ctx.device, dtype=ctx.dtype)

    inputs_embeds = ctx.model.prepare_inputs_embeds(**prepare_inputs)

    outputs = ctx.model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=ctx.tokenizer.eos_token_id,
        bos_token_id=ctx.tokenizer.bos_token_id,
        eos_token_id=ctx.tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample if temperature and temperature > 0 else False,
        use_cache=True,
        temperature=temperature if temperature and temperature > 0 else None,
        top_p=top_p if temperature and temperature > 0 else None,
    )

    return ctx.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)


def _t2i_generate_tokens_and_patches(
    ctx: ModelContext,
    *,
    input_ids: torch.LongTensor,
    width: int,
    height: int,
    temperature: float = 1.0,
    parallel_size: int = 4,
    cfg_weight: float = 5.0,
    image_token_num_per_image: int = 576,
    patch_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(ctx.device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = ctx.processor.pad_id

    inputs_embeds = ctx.model.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(ctx.device)

    pkv = None
    for i in range(image_token_num_per_image):
        outputs = ctx.model.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=pkv,
        )
        pkv = outputs.past_key_values
        hidden_states = outputs.last_hidden_state

        logits = ctx.model.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1
        ).view(-1)
        img_embeds = ctx.model.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    patches = ctx.model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, width // patch_size, height // patch_size],
    )
    return generated_tokens.to(dtype=torch.int), patches


def _t2i_unpack(patches: torch.Tensor, *, width: int, height: int, parallel_size: int) -> np.ndarray:
    dec = patches.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    return visual_img


@torch.inference_mode()
def run_text_to_image(
    ctx: ModelContext,
    *,
    prompt: str,
    out_dir: str,
    seed: Optional[int] = None,
    guidance: float = 5.0,
    temperature: float = 1.0,
    parallel_size: int = 4,
    width: int = 384,
    height: int = 384,
) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)

    # Seeds
    if seed is not None:
        torch.manual_seed(seed)
        if ctx.device.type == "cuda":
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)

    conv = get_conv_template(ctx.processor.sft_format)
    user_role = conv.roles[0]
    assistant_role = conv.roles[1]
    messages = [
        {"role": user_role, "content": prompt},
        {"role": assistant_role, "content": ""},
    ]
    text = ctx.processor.apply_sft_template_for_multi_turn_prompts(
        conversations=messages,
        sft_format=ctx.processor.sft_format,
        system_prompt="",
    )
    text = text + ctx.processor.image_start_tag

    input_ids = torch.LongTensor(ctx.tokenizer.encode(text)).to(ctx.device)

    # Normaliza dimensões para múltiplos de 16
    w = (width // 16) * 16
    h = (height // 16) * 16

    _, patches = _t2i_generate_tokens_and_patches(
        ctx,
        input_ids=input_ids,
        width=w,
        height=h,
        temperature=temperature,
        parallel_size=parallel_size,
        cfg_weight=guidance,
    )

    images = _t2i_unpack(patches, width=w, height=h, parallel_size=parallel_size)

    ts = time.strftime("%Y%m%d-%H%M%S")
    batch_id = uuid.uuid4().hex[:8]

    saved: List[str] = []
    for i in range(parallel_size):
        img = Image.fromarray(images[i])
        filename = f"t2i_{ts}_{batch_id}_{i}.png"
        path = os.path.join(out_dir, filename)
        img.save(path)
        saved.append(path)

    return saved
