"""
collector.py — 웹 검색 + 기사 본문 추출

토픽별로 DuckDuckGo 검색 → URL 방문 → 본문 추출의 파이프라인을 수행한다.
참고문헌(references) URL을 우선 수집하고, 부족분은 DDGS 검색으로 채운다.

[의존성]
    - ddgs: DuckDuckGo 검색 (필수)
    - trafilatura: 기사 본문 추출 (권장, 없으면 BeautifulSoup 폴백)
    - requests + beautifulsoup4: 폴백 추출 (필수)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests
from ddgs import DDGS

logger = logging.getLogger(__name__)

# trafilatura 선택적 임포트
try:
    import trafilatura
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False
    logger.warning(
        "trafilatura 미설치 — 본문 추출 품질이 제한됩니다. "
        "pip install trafilatura 로 설치를 권장합니다."
    )

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


# ── 검색 쿼리 생성 ────────────────────────────────────────────────────────────

def generate_search_queries(topic: dict) -> List[str]:
    """토픽의 찬/반/중립 관점으로 다양한 검색 쿼리를 생성한다."""
    title = topic["title"]
    pro = topic.get("pro", "")
    con = topic.get("con", "")

    return [
        f"{title} 찬성 근거 데이터",
        f"{title} 반대 근거 데이터",
        f"{title} 통계 연구 결과",
        f"{title} 전문가 의견",
        f"{pro} 사례",
        f"{con} 사례",
    ]


# ── URL 본문 추출 ─────────────────────────────────────────────────────────────

def _extract_with_trafilatura(html: str) -> Optional[str]:
    """trafilatura로 기사 본문을 추출한다."""
    if not _HAS_TRAFILATURA:
        return None
    text = trafilatura.extract(html, include_comments=False, include_tables=False)
    return text if text and len(text) > 200 else None


def _extract_with_bs4(html: str) -> Optional[str]:
    """BeautifulSoup으로 기사 본문을 추출한다 (폴백)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]

    if not paragraphs:
        return None
    return "\n\n".join(paragraphs[:30])


def extract_text_from_url(url: str, timeout: int = 10) -> Optional[str]:
    """URL에서 기사 본문을 추출한다. trafilatura → BS4 순으로 폴백."""
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    resp = None
    for verify in (True, False):
        try:
            resp = requests.get(
                url, timeout=timeout, headers=_REQUEST_HEADERS, verify=verify,
            )
            resp.raise_for_status()
            break
        except requests.exceptions.SSLError:
            if verify:
                continue  # SSL 실패 → verify=False로 재시도
            logger.warning("[collector] SSL 재시도도 실패 (%s)", url[:60])
            return None
        except Exception as e:
            logger.warning("[collector] HTTP 요청 실패 (%s): %s", url[:60], e)
            return None

    if resp is None:
        return None

    # trafilatura 우선
    text = _extract_with_trafilatura(resp.text)
    if text:
        return text

    # BS4 폴백
    return _extract_with_bs4(resp.text)


# ── DDGS 검색 ─────────────────────────────────────────────────────────────────

def _search_ddgs(query: str, max_results: int = 3) -> List[Dict]:
    """DuckDuckGo 텍스트 검색을 수행한다."""
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        logger.warning("[collector] DDGS 검색 실패 (%s): %s", query[:40], e)
        return []


# ── 토픽별 수집 ───────────────────────────────────────────────────────────────

def collect_for_topic(
    topic: dict,
    target_count: int = 20,
    search_delay: float = 0.5,
) -> List[Dict]:
    """하나의 토픽에 대해 기사를 수집한다.

    1단계: 토픽의 참고문헌(references) URL을 우선 수집
    2단계: DDGS 검색으로 추가 기사 수집
    3단계: 본문 추출 실패 시 DDGS snippet을 폴백으로 사용

    Args:
        topic:        topics.json의 단일 항목
        target_count: 목표 기사 수 (실제 수집량은 청킹 여유를 위해 2배 시도)
        search_delay: DDGS 검색 간 대기 시간 (초, rate-limit 방지)

    Returns:
        수집된 기사 리스트 [{url, title, text, source_type, collected_at}]
    """
    topic_id = topic.get("id", "unknown")
    raw_articles: List[Dict] = []
    seen_urls: set = set()
    fetch_target = target_count * 2  # 청킹 후 선별을 위해 여유분 확보

    # ── 1. 참고문헌 URL ──────────────────────────────────────────────────────
    for ref in topic.get("references", []):
        url = ref.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        text = extract_text_from_url(url)
        if text:
            raw_articles.append({
                "url": url,
                "title": ref.get("title", ""),
                "text": text,
                "source_type": "report",
                "collected_at": datetime.now().isoformat(),
            })
            logger.info("[collector] 참고문헌 수집: %s", ref.get("title", url[:40]))

    # ── 2. DDGS 검색 ─────────────────────────────────────────────────────────
    queries = generate_search_queries(topic)
    for query in queries:
        if len(raw_articles) >= fetch_target:
            break

        results = _search_ddgs(query, max_results=3)
        for r in results:
            if len(raw_articles) >= fetch_target:
                break

            url = r.get("href", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            text = extract_text_from_url(url)
            if not text:
                # 폴백: DDGS snippet 사용
                snippet = r.get("body", "")
                title = r.get("title", "")
                if len(snippet) > 80:
                    text = f"{title}. {snippet}" if title else snippet

            if text:
                raw_articles.append({
                    "url": url,
                    "title": r.get("title", ""),
                    "text": text,
                    "source_type": "news",
                    "collected_at": datetime.now().isoformat(),
                })

        time.sleep(search_delay)

    logger.info("[collector] %s: %d개 기사 수집 완료", topic_id, len(raw_articles))
    return raw_articles
