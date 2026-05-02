"""
ingest_wikipedia.py — 각 토론 토픽 관련 Wikipedia 페이지를 Pinecone 에 인덱싱.

Tavily 호출 0 으로 vector DB 다양성을 늘리는 사전 워밍 스크립트.
토픽별로 큐레이션된 Wikipedia 검색어 리스트로 페이지를 fetch 한 뒤
기존 src.graph.vector_store.upsert_search_results() API 로 저장한다.

실행:
    python scripts/ingest_wikipedia.py                    # 12개 토픽 전체
    python scripts/ingest_wikipedia.py --topic tech_001   # 특정 토픽만
    python scripts/ingest_wikipedia.py --dry-run          # fetch 만, upsert 안 함
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

# .env 로드 + import path
# 스크립트 위치: scripts/wiki/ingest_wikipedia.py → ROOT 는 두 단계 위
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("wiki_ingest")

from src.graph.vector_store import upsert_search_results  # noqa: E402


# ============================================================
# 토픽 ID → Wikipedia 검색어 큐레이션 리스트
# - 한국어 키워드 위주 (영어 키워드는 한국어 위키에서 잘 안 잡힘)
# - 너무 일반적인 키워드(중동·곡물·철강 등) 지양 — 토픽 핵심에 집중
# - 토픽당 5개. search API 가 fuzzy 매칭하니 정확한 페이지 제목 아니어도 OK
# ============================================================
TOPIC_WIKIPEDIA_KEYWORDS: Dict[str, List[str]] = {
    # ── 기술/AI ─────────────────────────────────────────────
    "tech_001": [  # AI 발전 vs 고용
        "챗봇",
        "생성형 인공지능",
        "기술적 실업",
        "인공지능의 고용 영향",
    ],
    "tech_002": [  # 데이터센터 vs 환경/전력
        "데이터 센터",
        "인공지능 환경문제",
        "그린 컴퓨팅",
    ],
    "tech_003": [  # AI 윤리 vs 모델 성능
        "인공지능의 윤리",
        "챗봇 정신병",
        "인공 일반 지능의 실존적 위험",
        "AI 안전",
        "딥페이크 포르노그래피",
        "인공지능 규제",
        "인공지능 규제법",
    ],
    # ── 경제/산업 ─────────────────────────────────────────
    "econ_001": [  # 이란 전쟁 책임
        "미국-이란 관계",
        "2025년~2026년 미국-이란 협상",
        "2026년 이란 전쟁",
        "JCPOA",
    ],
    "econ_002": [  # 트럼프 관세 vs 스태그플레이션
        "트럼프 2기 행정부 관세",
        "무역 전쟁",
    ],
    "econ_003": [  # 중동 위기: 원유 vs 식량
        
    ],
    # ── 정치/사회 ─────────────────────────────────────────
    "poli_001": [  # 펜타곤 언론 접근
      
    ],
    "poli_002": [  # 인도 녹색강철 의무화
        "탄소세",
        "탄소배출권 거래",
        "탄소 배출",
    ],
    "poli_003": [  # GM 작물 확대
        "유전자 변형 작물",
        "GMO 음모론",
    ],
    # ── 과학/환경 ─────────────────────────────────────────
    "env_001": [  # 기후난민의 원인
        "생태학적 난민",
    ],
    "env_002": [  # 인플레이션과 에너지
        "에너지 위기",
        "재생 가능 에너지",
        "탈원전",
    ],
    "env_003": [  # 기후위기 대응: 교육 vs 정치
        "기후 운동",
    ],
}


WIKI_API_BASE = "https://{lang}.wikipedia.org/w/api.php"

# 단일 session 재사용 — 연결 오버헤드 감소 + 일관된 헤더
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "SJU-debate-cache-warmer/1.0 (research)"})


def _wiki_request(
    params: Dict,
    lang: str,
    max_retries: int = 4,
) -> Optional[Dict]:
    """rate-limit / 429 / 일시 오류 재시도 포함 wiki API 호출.

    - 429: Retry-After 헤더 또는 지수 백오프(5/10/20/40s)
    - 그 외 오류: 지수 백오프(1/2/4/8s)
    """
    for attempt in range(1, max_retries + 1):
        try:
            r = _SESSION.get(
                WIKI_API_BASE.format(lang=lang),
                params=params,
                timeout=15,
            )
            if r.status_code == 429:
                # Retry-After 헤더 우선, 없으면 지수 백오프
                retry_after = r.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else 5 * (2 ** (attempt - 1))
                if attempt < max_retries:
                    logger.info("  [wiki/%s] 429 — %ds 대기 후 재시도(%d/%d)", lang, wait, attempt, max_retries)
                    time.sleep(wait)
                    continue
                logger.warning("  [wiki/%s] 429 재시도 한계 도달", lang)
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))
                continue
            logger.warning("  [wiki/%s] HTTP 실패: %s", lang, e)
    return None


def search_best_title(keyword: str, lang: str = "ko") -> Optional[str]:
    """list=search 로 키워드와 가장 가까운 페이지의 정확한 title 을 찾는다."""
    data = _wiki_request(
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": keyword,
            "srlimit": 1,
        },
        lang=lang,
    )
    if not data:
        return None
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None
    return hits[0].get("title")


def fetch_wikipedia_page(
    title: str,
    lang: str = "ko",
    min_chars: int = 400,
    max_retries: int = 3,
) -> Optional[Dict[str, str]]:
    """주어진 정확한 title 로 wiki 페이지를 fetch — 없거나 stub 이면 None."""
    data = _wiki_request(
        {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
        },
        lang=lang,
        max_retries=max_retries,
    )
    if not data:
        return None
    pages = data.get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if pid == "-1":
            return None
        extract = (page.get("extract") or "").strip()
        if len(extract) < min_chars:
            return None
        actual_title = page.get("title", title)
        url = f"https://{lang}.wikipedia.org/wiki/{actual_title.replace(' ', '_')}"
        return {"url": url, "title": actual_title, "content": extract}
    return None


def find_and_fetch(
    keyword: str,
    lang: str = "ko",
    min_chars: int = 400,
) -> Optional[Dict[str, str]]:
    """키워드로 위키 페이지를 찾아 fetch.

    1) 우선 keyword 자체를 정확한 title 로 시도 (redirect 자동 추적)
       — 사용자가 정확한 title 을 알고 있는 경우 search 거치지 않고 바로 가져옴
    2) 그게 없으면 search API 로 best title 찾아 fetch (fuzzy 매칭)
    """
    # 1) exact title 시도
    page = fetch_wikipedia_page(keyword, lang=lang, min_chars=min_chars)
    if page:
        return page
    # 2) search fallback
    best = search_best_title(keyword, lang=lang)
    if not best:
        return None
    return fetch_wikipedia_page(best, lang=lang, min_chars=min_chars)


def fetch_with_fallback(keyword: str) -> Optional[Dict[str, str]]:
    """ko 위키에서 exact→search → 없으면 en 위키에서 exact→search."""
    page = find_and_fetch(keyword, lang="ko")
    if page:
        return page
    return find_and_fetch(keyword, lang="en")


# 기본 로컬 저장 위치 — Pinecone 에 들어간 게 뭔지 검증/백업 용도.
# 각 페이지를 {topic_id}__{kw}.txt 로 저장 (메타데이터 헤더 + 본문).
DEFAULT_SAVE_DIR = ROOT / "data" / "wiki_pages"


def _save_page_to_disk(
    save_dir: Path,
    topic_id: str,
    keyword: str,
    page: Dict[str, str],
) -> Path:
    """fetch 한 위키 페이지를 로컬 텍스트 파일로 저장."""
    save_dir.mkdir(parents=True, exist_ok=True)
    safe_kw = keyword.replace("/", "_").replace(" ", "_")
    out = save_dir / f"{topic_id}__{safe_kw}.txt"
    header = (
        f"# topic_id: {topic_id}\n"
        f"# keyword:  {keyword}\n"
        f"# title:    {page['title']}\n"
        f"# url:      {page['url']}\n"
        f"# length:   {len(page['content'])} chars\n"
        f"---\n"
    )
    out.write_text(header + page["content"], encoding="utf-8")
    return out


def main(
    topic_filter: Optional[str] = None,
    dry_run: bool = False,
    save_dir: Optional[Path] = None,
    skip_save: bool = False,
) -> None:
    total_pages_fetched = 0
    total_pages_skipped = 0
    total_vectors = 0

    if save_dir is None:
        save_dir = DEFAULT_SAVE_DIR

    for topic_id, keywords in TOPIC_WIKIPEDIA_KEYWORDS.items():
        if topic_filter and topic_id != topic_filter:
            continue

        logger.info("")
        logger.info("=== %s — 검색어 %d개 ===", topic_id, len(keywords))

        for kw in keywords:
            page = fetch_with_fallback(kw)
            time.sleep(1.5)  # 위키 API rate-limit 회피 (search+extract 두 번 호출)
            if not page:
                logger.warning("  [skip] %s — 페이지 없음/stub", kw)
                total_pages_skipped += 1
                continue

            logger.info(
                "  [ok]   %s (%d chars) — %s",
                page["title"],
                len(page["content"]),
                page["url"],
            )
            total_pages_fetched += 1

            # 로컬 백업 저장 (dry-run 일 때도 저장 — 어떤 페이지가 매칭됐는지 검증용)
            if not skip_save:
                try:
                    out = _save_page_to_disk(save_dir, topic_id, kw, page)
                    logger.info("       └ saved: %s", out.relative_to(ROOT))
                except Exception as e:
                    logger.warning("       └ 로컬 저장 실패: %s", e)

            if dry_run:
                continue

            try:
                n = upsert_search_results(
                    query=f"wiki:{topic_id}:{kw}",
                    results=[page],
                )
                total_vectors += n
                logger.info("       └ upsert: %d vectors", n)
            except Exception as e:
                logger.error("       └ upsert 실패: %s", e)

    logger.info("")
    logger.info("=== 완료 ===")
    logger.info("  fetched  : %d page(s)", total_pages_fetched)
    logger.info("  skipped  : %d", total_pages_skipped)
    if not skip_save:
        logger.info("  saved    : %s", save_dir.relative_to(ROOT))
    if not dry_run:
        logger.info("  upserted : %d vector(s)", total_vectors)
    else:
        logger.info("  (dry-run — upsert 안 함)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topic", help="특정 토픽만 처리 (예: tech_001)")
    p.add_argument("--dry-run", action="store_true", help="fetch 만, upsert 안 함")
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help=f"위키 페이지 로컬 저장 경로 (기본: {DEFAULT_SAVE_DIR.relative_to(ROOT)})",
    )
    p.add_argument(
        "--skip-save",
        action="store_true",
        help="로컬 저장 비활성 (Pinecone 에만 upsert)",
    )
    args = p.parse_args()
    main(
        topic_filter=args.topic,
        dry_run=args.dry_run,
        save_dir=args.save_dir,
        skip_save=args.skip_save,
    )
