import json
import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .config import MODEL_ID, DRIVE_PATH

tokenizer = None
model = None
MODEL_INPUT_DEVICE = None


def load_model(drive_path: str = DRIVE_PATH, model_id: str = MODEL_ID):
    global tokenizer, model, MODEL_INPUT_DEVICE

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    if os.path.exists(drive_path):
        print(f"Drive 캐시에서 로드 중: {drive_path}")
        tokenizer = AutoTokenizer.from_pretrained(drive_path)
        model = AutoModelForCausalLM.from_pretrained(
            drive_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        print(f"HuggingFace에서 다운로드 중: {model_id}")
        print("처음 1회만 실행됩니다. Drive에 저장 후 다음부터 빠르게 로드됩니다.")
        os.makedirs(drive_path, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        print(f"Drive에 저장 중: {drive_path}")
        tokenizer.save_pretrained(drive_path)
        model.save_pretrained(drive_path)
        print("Drive 저장 완료")

    model.eval()
    MODEL_INPUT_DEVICE = next(model.parameters()).device
    print(f"로드 완료 | device: {MODEL_INPUT_DEVICE} | 4bit quantized")


def load_model_colab():
    """Google Colab 환경에서 Drive 마운트 후 모델 로드."""
    from google.colab import drive
    drive.mount("/content/drive")
    load_model()


def qwen_chat(
    system_prompt: str,
    user_msg: str,
    max_new_tokens: int = 512,
    assistant_prefix: str = "",
) -> str:
    """
    Qwen2.5-Instruct 채팅 형식으로 추론.
    assistant_prefix: 어시스턴트 턴을 이 문자열로 시작 강제 (JSON 파싱 보장용)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if assistant_prefix:
        text = text + assistant_prefix
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(MODEL_INPUT_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_json(raw: str) -> dict:
    """LLM 출력에서 JSON 추출. 코드블록 래핑 자동 제거."""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if m:
        raw = m.group(1)
    m2 = re.search(r"(\{[\s\S]+\})", raw)
    if m2:
        raw = m2.group(1)
    return json.loads(raw)
