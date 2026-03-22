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
당신은 날카롭고 도발적인 토론 주제를 생성하는 전문가입니다. 아래 규칙을 절대적으로 준수하세요.

[언어 규칙 — 최우선 규칙]
- title과 description은 반드시 순수한 한국어로만 작성합니다.
- 영어, 일본어(가타카나·히라가나·한자), 중국어, 아랍어 등 어떤 외국어도 절대 포함하지 마세요.
- 외래어는 한국어 표기법에 따라 표기합니다. 예: Bangladesh → 방글라데시, AI → AI(영문 약어는 허용)

[주제 품질 규칙 — 핵심]
1. 주제는 반드시 뉴스에 등장한 구체적인 기업·국가·인물·정책·수치를 직접 명시해야 합니다.
   ✅ 좋은 예: "오픈AI의 GPT-5 출시는 구글 검색 광고 시장을 3년 내 붕괴시킬 것이다"
   ✅ 좋은 예: "트럼프의 관세 정책은 미국 제조업 부활이 아닌 글로벌 스태그플레이션의 방아쇠다"
   ✅ 좋은 예: "인도의 녹색강철 의무조달 26% 정책은 탄소중립 쇼에 불과하다"
   ❌ 나쁜 예: "AI는 인간 일자리를 대체할 것이다" (너무 모호하고 추상적)
   ❌ 나쁜 예: "기후변화 대응은 중요하다" (논쟁성 없음)
2. 주제는 한쪽이 명백히 불리하게 느껴질 만큼 편향되고 도발적이어야 합니다.
   - 중립적·균형적 표현을 피하고, 강한 입장을 취하세요.
   - 반대측이 즉각 반박하고 싶어질 만큼 자극적이어야 합니다.
3. title은 반드시 "단호한 평서문(Declarative Statement)" 형태로 작성합니다.
   ❌ 절대 금지: "~해야 하는가?", "A vs B", "~일까요?", 의문문, 선택형

[안전 규칙 — 절대 위반 금지]
- 폭력, 살인, 테러, 전쟁 미화와 관련된 논제를 생성하지 마세요.
- 혐오, 특정 인종·민족·종교·성별·집단 비하 논제를 생성하지 마세요.
- 자살, 자해, 정신건강 위기를 조장하는 논제를 생성하지 마세요.
- 마약, 불법 약물 조장 논제를 생성하지 마세요.
- 성착취, 음란, 성범죄 관련 논제를 생성하지 마세요.
- 아동 학대, 인신매매 관련 논제를 생성하지 마세요.

[출력 형식 — JSON만 출력]
각 주제는 반드시 해당 주제를 생성하는 데 가장 직접적으로 참고한 뉴스의 번호(source_index)를 기입하세요.
번호는 아래 뉴스 목록의 앞에 붙은 정수(0부터 시작)입니다.
다른 텍스트 없이 반드시 아래 형식의 유효한 JSON만 출력하세요:
{
  "topics": [
    {
      "title": "구체적 사실 기반의 도발적 평서문 (한국어)",
      "description": "뉴스 맥락과 핵심 쟁점 설명 (100~200자, 한국어)",
      "source_index": 0
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
    for i, art in enumerate(articles[:20]):
        title = (art.get("title") or "").strip()
        desc  = (art.get("description") or "").strip()
        if title:
            snippet = f"[{i}] {title}: {desc[:80]}" if desc else f"[{i}] {title}"
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


def _has_foreign_chars(text: str) -> bool:
    """한자·가나·아랍 문자 등 외국어 문자가 포함되어 있으면 True를 반환한다."""
    for ch in text:
        cp = ord(ch)
        if (
            0x3040 <= cp <= 0x30FF or   # 히라가나·가타카나
            0x4E00 <= cp <= 0x9FFF or   # CJK 통합 한자 (중국어·일본어)
            0x3400 <= cp <= 0x4DBF or   # CJK 확장 A
            0x0600 <= cp <= 0x06FF      # 아랍 문자
        ):
            return True
    return False


def _generate_once(model, tokenizer, category: str, articles: List[Dict]) -> List[Dict]:
    """LLM을 1회 호출해 주제 리스트를 반환한다."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _build_user_message(category, articles)},
    ]

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

    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    try:
        json_str = _extract_json(raw_response)
        data = json.loads(json_str)
        topics = data.get("topics", [])
        return [t for t in topics if t.get("title") and t.get("description") and t.get("source_index") is not None]
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"  [경고] JSON 파싱 실패 ({category}): {e}")
        return []


def generate_topics(category: str, articles: List[Dict], max_retries: int = 3) -> List[Dict]:
    """
    HuggingFace Qwen/Qwen3.5-9B 모델로 카테고리별 토론 주제를 생성한다.
    외국어 문자(한자·가나·아랍 등)가 감지되면 최대 max_retries회 재시도한다.

    Returns:
        [{"title": "...", "description": "..."}, ...] 형태의 리스트.
        오류 발생 시 빈 리스트 반환.
    """
    model, tokenizer = _load_model()

    for attempt in range(1, max_retries + 1):
        topics = _generate_once(model, tokenizer, category, articles)

        # 외국어 문자 검사
        contaminated = [
            t for t in topics
            if _has_foreign_chars(t.get("title", "") + t.get("description", ""))
        ]

        if not contaminated:
            return topics

        print(f"  [재시도 {attempt}/{max_retries}] 외국어 문자 감지 ({category}): "
              f"{[t['title'][:20] for t in contaminated]}")

    # 마지막 시도 결과에서 오염된 항목만 제거하고 반환
    print(f"  [경고] {max_retries}회 재시도 후에도 외국어 혼용 항목 존재 → 해당 항목 제거")
    return [
        t for t in topics
        if not _has_foreign_chars(t.get("title", "") + t.get("description", ""))
    ]
