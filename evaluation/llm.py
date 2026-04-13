from pathlib import Path
from typing import Optional

_QWEN_MODEL = None
_QWEN_TOKENIZER = None

QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
QWEN_LOCAL_MODEL_PATH = "/content/drive/MyDrive/Qwen2.5-7B-Instruct"


def resolve_qwen_model_source(
    local_model_path: Optional[str] = QWEN_LOCAL_MODEL_PATH,
    fallback_model_id: str = QWEN_MODEL_ID,
) -> str:
    if local_model_path:
        candidate = Path(local_model_path)
        if candidate.exists():
            return str(candidate)
        print(f"[안내] 로컬 모델 경로를 찾지 못해 Hugging Face 모델로 fallback합니다: {candidate}")
    return fallback_model_id


def load_qwen_7b_quantized(
    model_id: str = QWEN_MODEL_ID,
    local_model_path: Optional[str] = QWEN_LOCAL_MODEL_PATH,
):
    global _QWEN_MODEL, _QWEN_TOKENIZER
    if _QWEN_MODEL is not None and _QWEN_TOKENIZER is not None:
        return _QWEN_MODEL, _QWEN_TOKENIZER

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_source = resolve_qwen_model_source(local_model_path, model_id)
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    _QWEN_TOKENIZER = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    _QWEN_MODEL = AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        local_files_only=Path(model_source).exists(),
    )

    if _QWEN_TOKENIZER.pad_token is None:
        _QWEN_TOKENIZER.pad_token = _QWEN_TOKENIZER.eos_token

    print(f"[모델 로드 완료] {model_source}")
    return _QWEN_MODEL, _QWEN_TOKENIZER


def qwen_chat(system_prompt: str, user_prompt: str, max_new_tokens: int = 700) -> str:
    import torch

    model, tokenizer = load_qwen_7b_quantized()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = generated[0][model_inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
