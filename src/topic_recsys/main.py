"""
main.py — Topic RecSys 전체 파이프라인 실행 진입점

실행 흐름:
  1. NewsAPI로 4개 카테고리 뉴스 크롤링 (과거 30일)
  2. Qwen/Qwen3.5-9B (HuggingFace)으로 카테고리별 토론 주제 생성
  3. 룰베이스 안전 필터링 적용
  4. 카테고리별 고유 ID 발급 (tech_001, econ_001, poli_001, env_001)
  5. topics_YYYYMMDD.json 으로 저장

사용:
  python -m src.topic_recsys.main
  또는
  python src/topic_recsys/main.py
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List

# ── 직접 실행(python main.py)과 모듈 실행(-m) 양쪽 import 지원 ─────────────────
try:
    from .crawler import NewsCrawler
    from .generator import generate_topics
    from .safety import filter_topics
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.topic_recsys.crawler import NewsCrawler
    from src.topic_recsys.generator import generate_topics
    from src.topic_recsys.safety import filter_topics

# ── 카테고리 → ID 접두사 매핑 ──────────────────────────────────────────────────
CATEGORY_PREFIX: Dict[str, str] = {
    "기술/AI":   "tech",
    "경제/산업": "econ",
    "정치/사회": "poli",
    "과학/환경": "env",
}

# 출력 디렉토리: 프로젝트 루트의 output/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────

def _assign_ids(
    topics: List[Dict],
    prefix: str,
    articles: List[Dict],
    start: int = 1,
) -> List[Dict]:
    """
    주제 리스트에 고유 ID를 발급하고, source_title로 기사를 매칭해 source를 첨부한다.

    예: prefix="tech", start=1 → "tech_001", "tech_002", ...
    """
    result = []
    for i, topic in enumerate(topics):
        idx = topic.get("source_index")
        matched = articles[idx] if isinstance(idx, int) and 0 <= idx < len(articles) else None
        source = (
            {
                "title":       matched.get("title", ""),
                "url":         matched.get("url", ""),
                "publishedAt": matched.get("publishedAt", ""),
            }
            if matched else
            {"title": "", "url": "", "publishedAt": ""}
        )
        result.append({
            "id":          f"{prefix}_{(start + i):03d}",
            "title":       topic.get("title", "").strip(),
            "description": topic.get("description", "").strip(),
            "source":      source,
        })
    return result


def _validate_minimum(categories: Dict[str, List[Dict]]) -> None:
    """카테고리당 최소 3개 주제가 있는지 확인하고 경고를 출력한다."""
    for cat, topics in categories.items():
        if len(topics) < 3:
            print(
                f"  [경고] '{cat}' 카테고리의 주제가 {len(topics)}개입니다 "
                f"(최소 3개 권장). LLM 재시도 또는 안전 필터 결과를 확인하세요."
            )


# ── 메인 파이프라인 ───────────────────────────────────────────────────────────

def run() -> Dict:
    """
    Topic RecSys 전체 파이프라인을 실행한다.

    Returns:
        저장된 JSON과 동일한 딕셔너리 구조.
    """
    # 환경변수에서 NEWS_API_KEY 로드 (.env 지원)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv 미설치 시 os.environ에서 직접 읽음

    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GNEWS_API_KEY 환경변수가 설정되지 않았습니다.\n"
            "프로젝트 루트의 .env 파일에 GNEWS_API_KEY=<your_key> 를 추가하세요.\n"
            "발급: https://gnews.io"
        )

    # ── 1단계: 뉴스 크롤링 ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  [1단계] NewsAPI 뉴스 크롤링 (최근 30일)")
    print("=" * 55)
    crawler = NewsCrawler(api_key)
    news_data = crawler.fetch_all()

    # ── 2단계: LLM 주제 생성 → 안전 필터링 → ID 발급 ──────────────────────────
    print("\n" + "=" * 55)
    print("  [2단계] LLM 토론 주제 생성 · 필터링 · ID 발급")
    print("=" * 55)

    categories_result: Dict[str, List[Dict]] = {}

    for category, prefix in CATEGORY_PREFIX.items():
        print(f"\n  ▶ [{category}]")

        articles = news_data.get(category, [])
        print(f"    수집 기사 수: {len(articles)}건")

        # LLM 호출
        raw_topics = generate_topics(category, articles)
        print(f"    LLM 생성 주제: {len(raw_topics)}개")

        # 룰베이스 안전 필터링
        safe_topics = filter_topics(raw_topics, category)

        # 고유 ID 발급 + source_title로 기사 1:1 매칭
        final_topics = _assign_ids(safe_topics, prefix, articles)
        categories_result[category] = final_topics

        print(f"    최종 확정 주제: {len(final_topics)}개")
        for t in final_topics:
            matched = "✅" if t["source"].get("url") else "⚠️ 미매칭"
            print(f"      [{t['id']}] {t['title']} ({matched})")

    # 최소 주제 수 검증
    _validate_minimum(categories_result)

    # ── 3단계: JSON 저장 ───────────────────────────────────────────────────────
    today_str = datetime.now().strftime("%Y%m%d")
    output_payload = {"categories": categories_result}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"topics_{today_str}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 55)
    print(f"  [완료] 결과 저장: {output_path}")
    print("=" * 55 + "\n")

    return output_payload


if __name__ == "__main__":
    run()
