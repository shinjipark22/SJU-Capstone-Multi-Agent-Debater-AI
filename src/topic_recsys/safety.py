"""
safety.py — 룰베이스(Rule-based) 안전 필터

LLM이 생성한 토론 주제의 title/description에 금지어가 포함된 경우
해당 항목을 리스트에서 제거한다. (2중 필터링의 코드 레벨 담당)
"""

from typing import Dict, List

# ── 금지어 리스트 ──────────────────────────────────────────────────────────────
# LLM 프롬프트 레벨 필터의 보완재로, 명시적 위험 키워드를 하드코딩으로 차단한다.
BANNED_WORDS: List[str] = [
    # 생명·신체 위협
    "자살", "자해", "자살방조", "자살충동",
    "살인", "살해", "살상", "학살", "집단학살", "종족청소",
    "테러", "테러리즘", "폭탄", "폭발물", "생화학무기", "핵무기사용",
    # 범죄·불법
    "마약", "마약류", "필로폰", "헤로인", "코카인", "대마",
    "인신매매", "납치조장", "불법무기",
    # 성·아동 착취
    "성착취", "성매매", "음란", "포르노", "아동포르노",
    "아동학대", "아동성범죄",
    # 혐오·차별
    "혐오", "인종혐오", "성혐오", "장애혐오",
    "유대인혐오", "이슬람혐오", "동성애혐오",
]


def is_safe(topic: Dict) -> bool:
    """title과 description 중 금지어가 없으면 True를 반환한다."""
    combined = f"{topic.get('title', '')} {topic.get('description', '')}"
    return not any(word in combined for word in BANNED_WORDS)


def filter_topics(topics: List[Dict], category: str = "") -> List[Dict]:
    """
    금지어가 포함된 주제를 제거한 안전한 리스트를 반환한다.

    Args:
        topics:   LLM이 생성한 주제 딕셔너리 리스트.
        category: 로그 출력용 카테고리명 (선택).

    Returns:
        금지어가 없는 주제만 담긴 리스트.
    """
    safe_topics = [t for t in topics if is_safe(t)]
    removed = len(topics) - len(safe_topics)

    if removed > 0:
        label = f"[{category}] " if category else ""
        print(f"  [안전 필터] {label}{removed}개 주제 금지어 감지 → 제거됨")

    return safe_topics
