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
당신은 토론 대회 출제위원입니다. 아래 뉴스를 읽고 토론 논제를 출제하세요.

토론 논제란 한 문장을 읽었을 때 사람들이 즉각 찬성파와 반대파로 갈리는 주장입니다.
아래 패턴 중에서 골라 사용하되, 같은 카테고리 내 논제들은 서로 다른 패턴을 사용하세요:
  패턴A: "[주체]의 [행동]은 [X]가 아닌 [Y]다"
  패턴B: "[주체]의 [정책]은 [강한 부정적 평가]에 불과하다"
  패턴C: "[사건]은 [결과]의 결정적 기폭제/분기점/발단/촉매다" (도화선 사용 금지)
  패턴D: "[사건]은 [표면적 명분]이 아닌 [숨겨진 의도]의 산물이다"
  패턴E: "[결과]의 책임은 [A]가 아니라 [B]에 있다"
  패턴F: "[행동]은 [가치]를 침해한다"
  패턴G: "[정책/행동]은 오히려 [부정적 결과]를 초래한다"
  패턴H: "[X]보다 [Y]가 우선되어야 한다"

예시 (뉴스 → 논제 변환):
  뉴스: "트럼프, 중국산 수입품에 추가 관세 부과"
  논제: "트럼프 관세 정책은 제조업 부활이 아닌 글로벌 스태그플레이션의 기폭제다"  ← 패턴A

  뉴스: "인도 정부, 공공 프로젝트 녹색강철 26% 의무 구매 발표"
  논제: "인도 녹색강철 의무조달은 탄소중립이 아닌 보호무역주의 장벽에 불과하다"  ← 패턴B

  뉴스: "미국 법원, 펜타곤 언론 접근 제한 정책 위헌 판결"
  논제: "트럼프의 펜타곤 언론 통제 정책은 언론의 자유를 침해한다"  ← 패턴K

  뉴스: "이란 전쟁으로 글로벌 유가 급등"
  논제: "유가 폭등의 책임은 이란이 아니라 미국의 중동 개입에 있다"  ← 패턴F

논제 작성 규칙:
- 뉴스에 실제 등장한 기업·국가·인물·정책만 사용 (없는 수치·사실 생성 금지)
- 한국어 50자 이내, 의문문 금지
- 미사여구·수식어 사용 금지 — 주체·행동·평가만 남겨 군더더기 없이 쓸 것
- 생성하는 논제들은 반드시 서로 다른 사건·정책·주체를 다뤄야 함 (같은 소재 반복 금지)
- 폭력·혐오·자살·마약·성착취·아동학대 논제 생성 금지

description: "[찬성 측 주장]과 [반대 측 주장]이 맞서는 주제입니다." 형식으로 한 문장 작성. 출처 번호 언급 금지
source_indices: 참고한 뉴스 번호 배열 (0부터 시작, 최대 3개)

다른 텍스트 없이 아래 JSON만 출력하세요:
{
  "topics": [
    {
      "title": "...",
      "description": "...",
      "source_indices": [0, 3]
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
            dtype=torch.float16,
            device_map="auto",
        )
        _model.eval()
        print(f"  [모델 로드] 완료")
    return _model, _tokenizer


def _build_user_message(
    category: str,
    articles: List[Dict],
    exclude_titles: Optional[List[str]] = None,
    n: int = 3,
) -> str:
    """카테고리와 수집 기사를 바탕으로 유저 메시지를 구성한다."""
    headlines: List[str] = []
    for i, art in enumerate(articles[:20]):
        title = (art.get("title") or "").strip()
        desc  = (art.get("description") or "").strip()
        if title:
            snippet = f"[{i}] {title}: {desc[:80]}" if desc else f"[{i}] {title}"
            headlines.append(snippet)

    news_block = "\n".join(headlines) if headlines else "관련 뉴스 없음"

    exclude_block = ""
    if exclude_titles:
        titles_str = "\n".join(f"  - {t}" for t in exclude_titles)
        exclude_block = (
            f"\n[이미 생성된 주제 — 반드시 제외]\n"
            f"아래 주제와 동일하거나 유사한 주제는 절대 생성하지 마세요:\n{titles_str}\n"
        )

    return (
        f"[카테고리]: {category}\n\n"
        f"[최근 30일 관련 뉴스 헤드라인 (영문 원문)]:\n{news_block}\n"
        f"{exclude_block}\n"
        f"위 뉴스를 참고하여 '{category}' 카테고리의 토론 주제를 정확히 {n}개 생성하세요.\n"
        f"모든 title은 반드시 단호한 평서문이어야 하며, JSON 형식으로만 응답하세요."
    )


def _extract_json(text: str) -> str:
    """응답 텍스트에서 JSON 블록만 추출한다."""
    # 마크다운 코드 블록 제거
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # 첫 번째 { ... } 블록 추출
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text


_QUESTION_ENDINGS = ("인가", "은가", "까요", "나요", "할까", "일까", "는가", "ㄴ가", "?")

def _is_question(title: str) -> bool:
    """title이 의문문이면 True를 반환한다."""
    t = title.rstrip(" .。")
    return any(t.endswith(e) for e in _QUESTION_ENDINGS)


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


def _generate_once(
    model,
    tokenizer,
    category: str,
    articles: List[Dict],
    exclude_titles: Optional[List[str]] = None,
    n: int = 3,
) -> List[Dict]:
    """LLM을 1회 호출해 주제 리스트를 반환한다."""
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _build_user_message(category, articles, exclude_titles, n)},
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
        return [
            t for t in topics
            if t.get("title") and t.get("description") and t.get("source_indices") is not None
            and not _is_question(t["title"])
        ]
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"  [경고] JSON 파싱 실패 ({category}): {e}")
        return []


def generate_topics(
    category: str,
    articles: List[Dict],
    max_retries: int = 3,
    exclude_titles: Optional[List[str]] = None,
    n: int = 3,
) -> List[Dict]:
    """
    HuggingFace Qwen/Qwen3.5-9B 모델로 카테고리별 토론 주제를 생성한다.
    외국어 문자(한자·가나·아랍 등)가 감지되면 최대 max_retries회 재시도한다.

    Args:
        exclude_titles: 이미 채택된 주제 제목 목록 — LLM에게 전달해 중복 방지.
        n:              생성할 주제 수 (기본 3).

    Returns:
        [{"title": "...", "description": "..."}, ...] 형태의 리스트.
        오류 발생 시 빈 리스트 반환.
    """
    model, tokenizer = _load_model()

    for attempt in range(1, max_retries + 1):
        topics = _generate_once(model, tokenizer, category, articles, exclude_titles, n)

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
