"""
generator.py — HuggingFace transformers 기반 토론 주제 생성기

모델: Qwen/Qwen3.5-9B (HuggingFace 직접 로드)
- thinking 모드 비활성화 (enable_thinking=False) → 깔끔한 JSON 출력
- 모델은 최초 1회만 로드하고 모듈 레벨 싱글톤으로 재사용
주제 형식: 찬/반이 명확한 단호한 평서문(Declarative Statement)
"""

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple

MODEL_ID = "Qwen/Qwen3.5-9B"

# ── 시스템 지시사항 (프롬프트 레벨 안전 가드레일 포함) ─────────────────────────
_SYSTEM_PROMPT = """\
당신은 전문 토론 주제 생성 AI입니다. 아래 규칙을 절대적으로 준수하세요.

[언어 규칙 — 최우선 규칙]
- title과 description은 반드시 순수한 한국어로만 작성합니다.
- 영어, 일본어(가타카나·히라가나·한자), 중국어, 아랍어 등 어떤 외국어도 절대 포함하지 마세요.
- 외래어는 한국어 표기법에 따라 표기합니다. 예: Bangladesh → 방글라데시, AI → AI(영문 약어는 허용)

[형식 규칙 — 반드시 지켜야 함]
1. title은 반드시 "단호한 평서문(Declarative Statement)" 형태로 작성합니다.
   ✅ 올바른 예: "생성형 AI는 창작 산업의 일자리를 대체할 것이다"
   ✅ 올바른 예: "기본소득제는 자동화 시대의 필수 정책이다"
   ✅ 올바른 예: "핵에너지는 탄소중립 달성을 위한 불가피한 선택이다"
   ❌ 절대 금지: "~해야 하는가?", "A vs B", "~일까요?", "어떻게 생각하시나요?"
   ❌ 절대 금지: 의문문, 선택형, 비교형 표현
2. 각 주제는 찬성(PRO)과 반대(CON) 입장이 명확히 나뉠 수 있어야 합니다.
3. description은 100~200자 한국어로, 주제의 배경과 핵심 쟁점을 설명합니다.

[안전 규칙 — 절대 위반 금지]
- 폭력, 살인, 테러, 전쟁 미화와 관련된 논제를 생성하지 마세요.
- 혐오, 특정 인종·민족·종교·성별·집단 비하 논제를 생성하지 마세요.
- 자살, 자해, 정신건강 위기를 조장하는 논제를 생성하지 마세요.
- 마약, 불법 약물 조장 논제를 생성하지 마세요.
- 성착취, 음란, 성범죄 관련 논제를 생성하지 마세요.
- 아동 학대, 인신매매 관련 논제를 생성하지 마세요.
- 범죄를 미화하거나 조장하는 논제를 생성하지 마세요.

[출력 형식 — JSON만 출력]
다른 텍스트 없이 반드시 아래 형식의 유효한 JSON만 출력하세요:
{
  "topics": [
    {
      "title": "단호한 평서문 형태의 토론 주제 (한국어)",
      "description": "해당 주제의 배경과 핵심 쟁점 설명 (100~200자, 한국어)"
    }
  ]
}"""

# ── 모듈 레벨 싱글톤 (카테고리마다 재로드 방지) ───────────────────────────────
_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None


def _load_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """모델과 토크나이저를 최초 1회만 로드한다."""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        print(f"  [모델 로드] {MODEL_ID} 로딩 중... (최초 1회)")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _model.eval()
        print(f"  [모델 로드] 완료")
    return _model, _tokenizer


def _build_user_message(category: str, articles: List[Dict]) -> str:
    """카테고리와 수집 기사를 바탕으로 유저 메시지를 구성한다."""
    headlines: List[str] = []
    for art in articles[:20]:
        title = (art.get("title") or "").strip()
        desc  = (art.get("description") or "").strip()
        if title:
            snippet = f"- {title}: {desc[:80]}" if desc else f"- {title}"
            headlines.append(snippet)

    news_block = "\n".join(headlines) if headlines else "관련 뉴스 없음"

    return (
        f"[카테고리]: {category}\n\n"
        f"[최근 30일 관련 뉴스 헤드라인 (영문 원문)]:\n{news_block}\n\n"
        f"위 뉴스를 참고하여 '{category}' 카테고리의 토론 주제를 정확히 3개 생성하세요.\n"
        f"모든 title은 반드시 단호한 평서문이어야 하며, JSON 형식으로만 응답하세요."
    )


def _extract_json(text: str) -> str:
    """응답 텍스트에서 JSON 블록만 추출한다."""
    # 마크다운 코드 블록 제거
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # 첫 번째 { ... } 블록 추출
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


def generate_topics(category: str, articles: List[Dict]) -> List[Dict]:
    """
    HuggingFace Qwen/Qwen3.5-9B 모델로 카테고리별 토론 주제를 생성한다.

    Returns:
        [{"title": "...", "description": "..."}, ...] 형태의 리스트.
        오류 발생 시 빈 리스트 반환.
    """
    model, tokenizer = _load_model()

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _build_user_message(category, articles)},
    ]

    # enable_thinking=False: Qwen3의 <think> 블록 비활성화 → 순수 JSON 출력
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 입력 토큰 제외하고 생성된 부분만 디코딩
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    try:
        json_str = _extract_json(raw_response)
        data = json.loads(json_str)
        topics = data.get("topics", [])
        valid = [t for t in topics if t.get("title") and t.get("description")]
        return valid
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"  [경고] JSON 파싱 실패 ({category}): {e}")
        print(f"         원본 응답 (앞 300자): {raw_response[:300]}")
        return []
