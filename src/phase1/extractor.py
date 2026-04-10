"""
extractor.py — Qwen 7B 기반 r/g 수치 추출기

토론 발언 텍스트에서 두 가지 수치를 추출한다:
    r (입장 강도/논리성): 0~40 사이 정수
    g (공격성/타격력):    5~50 사이 정수

모델 로딩은 최초 호출 시 1회만 수행한다 (모듈 레벨 싱글톤).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)

# ── 모델 설정 ─────────────────────────────────────────────────────────────────
_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
_MAX_NEW_TOKENS = 64
_R_MIN, _R_MAX = 0, 40
_G_MIN, _G_MAX = 5, 50
_R_DEFAULT, _G_DEFAULT = 20, 25  # 추출 실패 시 중간값 사용

# 싱글톤 컨테이너
_tokenizer = None
_model = None
_device = None
_load_attempted: bool = False  # 로딩 시도 여부 (실패 시 재시도 방지)


# ── 전용 추출 프롬프트 ────────────────────────────────────────────────────────
_EXTRACTION_SYSTEM = (
    "당신은 토론 발언을 분석하는 채점 시스템입니다. "
    "반드시 JSON 형식으로만 응답하고, 다른 텍스트는 절대 출력하지 마세요."
)

_EXTRACTION_TEMPLATE = """다음 토론 발언을 읽고 두 가지 수치를 추출하세요.

r (입장 강도/논리성): 0~40 사이 정수
  0  = 논거 없음, 순수 감정 발언
  20 = 보통 수준의 논리, 근거 일부 제시
  40 = 명확한 논리 구조, 강력한 근거 및 사례 제시

g (공격성/타격력): 5~50 사이 정수
  5  = 매우 온화, 단순 의견 진술
  25 = 중간 수준 반박, 상대 논리 일부 지적
  50 = 강력한 공격, 상대 핵심 논거 완전 반박

발언:
\"\"\"{speech}\"\"\"

JSON 형식으로만 응답하세요:
{{"r": <정수>, "g": <정수>}}"""


# ── 모델 초기화 ───────────────────────────────────────────────────────────────

def _load_model() -> None:
    """Qwen 7B 모델을 최초 1회 로드한다. 실패 시 재시도하지 않는다."""
    global _tokenizer, _model, _device, _load_attempted

    if _load_attempted:
        return
    _load_attempted = True

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Qwen 7B 모델 로딩 중... (device=%s)", _device)

        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_NAME,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        _model.eval()
        logger.info("Qwen 7B 모델 로딩 완료.")

    except Exception as exc:  # noqa: BLE001
        logger.warning("Qwen 7B 로딩 실패 (%s). 폴백 추출기를 사용합니다.", exc)
        _model = None
        _tokenizer = None


# ── 내부 파싱 유틸 ────────────────────────────────────────────────────────────

def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _parse_rg(raw: str) -> Tuple[int, int]:
    """LLM 출력 문자열에서 r, g를 파싱한다.

    JSON 블록 → 정규식 순으로 폴백한다.
    """
    # 1차: JSON 파싱
    try:
        # ```json ... ``` 블록 제거
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        data = json.loads(clean)
        r = _clamp(int(data["r"]), _R_MIN, _R_MAX)
        g = _clamp(int(data["g"]), _G_MIN, _G_MAX)
        return r, g
    except Exception:  # noqa: BLE001
        pass

    # 2차: 정규식 폴백 — "r": 숫자, "g": 숫자 패턴 탐색
    r_match = re.search(r'"r"\s*:\s*(\d+)', raw)
    g_match = re.search(r'"g"\s*:\s*(\d+)', raw)
    if r_match and g_match:
        r = _clamp(int(r_match.group(1)), _R_MIN, _R_MAX)
        g = _clamp(int(g_match.group(1)), _G_MIN, _G_MAX)
        return r, g

    logger.warning("r/g 파싱 실패. 기본값(%d, %d) 사용. raw=%r", _R_DEFAULT, _G_DEFAULT, raw[:120])
    return _R_DEFAULT, _G_DEFAULT


# ── 공개 API ──────────────────────────────────────────────────────────────────

def extract_rg(speech: str) -> Tuple[int, int]:
    """토론 발언 텍스트에서 r, g를 추출한다.

    Qwen 7B 모델이 사용 가능하면 LLM 추론을 수행하고,
    그렇지 않으면 규칙 기반 폴백 추출기를 사용한다.

    Args:
        speech: 단일 발언 텍스트 (한국어/영어 무관)

    Returns:
        (r, g) — r: 0~40, g: 5~50 정수 튜플
    """
    _load_model()

    if _model is not None and _tokenizer is not None:
        return _extract_via_llm(speech)

    return _extract_fallback(speech)


def _extract_via_llm(speech: str) -> Tuple[int, int]:
    """Qwen 7B Chat 형식으로 r, g를 추출한다."""
    import torch

    prompt = _EXTRACTION_TEMPLATE.format(speech=speech[:800])  # 토큰 절약

    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    try:
        # Qwen2.5-Instruct apply_chat_template
        text = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = _tokenizer(text, return_tensors="pt").to(_device)

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=_tokenizer.eos_token_id,
            )

        # 입력 토큰 제외 → 생성 부분만 디코딩
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        raw = _tokenizer.decode(generated, skip_special_tokens=True).strip()
        return _parse_rg(raw)

    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM 추출 중 오류 (%s). 폴백 사용.", exc)
        return _extract_fallback(speech)


# ── 판세 판정 프롬프트 ────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "당신은 토론 판세를 분석하는 심판입니다. "
    "반드시 JSON 형식으로만 응답하고, 다른 텍스트는 절대 출력하지 마세요."
)

_JUDGE_TEMPLATE = """다음 토론 발언 두 개를 읽고, 어느 쪽이 더 설득력 있는지 판정하세요.

[찬성 발언]
{pro_text}

[반대 발언]
{con_text}

판정 기준 (발언 내용만 보세요. 숫자 점수는 무시하세요):
- 논리 구조가 명확한가
- 구체적 근거·사례·수치를 제시했는가
- 상대 주장을 효과적으로 반박했는가

우세 점수 기준 (0~5):
  0 = 완전 무승부
  1 = 아주 근소하게 앞섬
  2 = 약간 앞섬
  3 = 뚜렷하게 앞섬
  4 = 크게 앞섬
  5 = 압도적 우위

JSON 형식으로만 응답하세요:
{{"winner": "찬성" 또는 "반대", "score": 0~5 정수, "reason": "발언 내용 근거로 한 한 줄 한글 설명"}}"""


def judge_turn(
    score_summary: str,
    v: float,
    dominance: str,
    speeches_text: str = "",
) -> str:
    """발언 원문만으로 판세를 판정하고 결과를 반환한다.

    Args:
        score_summary : (미사용 — 발언 원문만으로 판정)
        v             : 폴백용 대립 지수
        dominance     : 폴백용 우세 진영
        speeches_text : "찬성:...\n반대:..." 형식 발언 원문

    Returns:
        "[찬성/반대]이 [score]/5로 우세 — [reason]" 형식 문자열
    """
    _load_model()

    if _model is not None and _tokenizer is not None:
        return _judge_via_llm(speeches_text)

    return _judge_fallback(v, dominance)


def _split_speeches(speeches_text: str):
    """speeches_text에서 찬성/반대 발언을 분리한다.

    "찬성:...\n반대:..." 또는 "PRO:...\nCON:..." 형식을 처리한다.
    분리에 실패하면 ("", "") 을 반환한다.
    """
    pro_text, con_text = "", ""

    # 한국어 레이블 우선
    pro_match = re.search(r"찬성\s*:\s*(.+?)(?=반대\s*:|$)", speeches_text, re.DOTALL)
    con_match = re.search(r"반대\s*:\s*(.+?)$", speeches_text, re.DOTALL)

    if pro_match and con_match:
        pro_text = pro_match.group(1).strip()
        con_text = con_match.group(1).strip()
    else:
        # 영어 레이블 폴백
        pro_match = re.search(r"PRO\s*:\s*(.+?)(?=CON\s*:|$)", speeches_text, re.DOTALL)
        con_match = re.search(r"CON\s*:\s*(.+?)$", speeches_text, re.DOTALL)
        if pro_match and con_match:
            pro_text = pro_match.group(1).strip()
            con_text = con_match.group(1).strip()

    return pro_text, con_text


def _judge_via_llm(speeches_text: str) -> str:
    """Qwen 7B로 발언 원문만 보고 판세 판정."""
    import torch

    # speeches_text에서 찬성/반대 분리
    pro_text, con_text = _split_speeches(speeches_text)

    prompt = _JUDGE_TEMPLATE.format(
        pro_text=pro_text[:600],
        con_text=con_text[:600],
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    try:
        text = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _tokenizer(text, return_tensors="pt").to(_device)
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=_tokenizer.eos_token_id,
            )
        raw = _tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        data = json.loads(clean)
        winner = data["winner"]
        score = int(data["score"])
        reason = data["reason"]
        return f"[{winner}]이 {score}/5로 우세 — {reason}"

    except Exception as exc:  # noqa: BLE001
        logger.warning("판세 판정 LLM 오류 (%s). 발언 원문 판정 불가.", exc)
        return "판세 판정 실패 — LLM 오류"


def _judge_fallback(v: float, dominance: str) -> str:
    """규칙 기반 폴백 판세 판정 (0~5 점수 척도)."""
    abs_v = abs(v)
    if abs_v < 0.5:
        score = 0
        reason = "팽팽한 접전"
    elif abs_v < 2.0:
        score = 1
        reason = f"{'찬성' if dominance == 'PRO' else '반대'}측이 논리성에서 근소하게 앞섬"
    elif abs_v < 5.0:
        score = 2
        reason = f"{'찬성' if dominance == 'PRO' else '반대'}측 논거가 약간 우세"
    elif abs_v < 10.0:
        score = 3
        reason = f"{'찬성' if dominance == 'PRO' else '반대'}측 입장 강도와 논거가 뚜렷이 앞섬"
    elif abs_v < 20.0:
        score = 4
        reason = f"{'찬성' if dominance == 'PRO' else '반대'}측이 논리성·공격성에서 크게 앞섬"
    else:
        score = 5
        reason = f"{'찬성' if dominance == 'PRO' else '반대'}측이 논리성·공격성 모두에서 압도적 우위"

    winner_kor = "찬성" if dominance == "PRO" else "반대"
    return f"[{winner_kor}]이 {score}/5로 우세 — {reason}"


def _extract_fallback(speech: str) -> Tuple[int, int]:
    """모델 없이 동작하는 규칙 기반 폴백 추출기.

    발언 길이·키워드를 기반으로 r, g를 근사한다.
    실제 모델 대비 정확도는 낮지만 시스템 동작을 보장한다.
    """
    text = speech.strip()
    length = len(text)

    # r: 발언 길이 비례 (짧을수록 논거 부족 경향)
    r = _clamp(int(length / 10), _R_MIN, _R_MAX)

    # g: 공격성 키워드 탐지
    aggressive_keywords = [
        "잘못", "틀렸", "반박", "오류", "모순", "부당", "근거 없",
        "왜곡", "비논리", "억지", "wrong", "incorrect", "fallacy",
        "contradiction", "refute", "invalid",
    ]
    hit_count = sum(1 for kw in aggressive_keywords if kw in text)
    g = _clamp(_G_MIN + hit_count * 5, _G_MIN, _G_MAX)

    return r, g
