"""
collector.py — 웹 검색 + 기사 본문 추출

토픽별로 다중 검색 엔진을 사용하여 근거 기사를 수집한다.

[검색 우선순위]
    1. 참고문헌(references) URL — 토픽에 명시된 신뢰 출처
    2. Tavily Search — AI 최적화 검색, 본문 자동 추출
    3. Naver News API — 한국어 뉴스 최적화
    4. DuckDuckGo (DDGS) — 폴백

[의존성]
    - tavily-python: Tavily 검색 (TAVILY_API_KEY 필요)
    - requests: Naver API + URL 본문 추출
    - trafilatura: 기사 본문 추출 (권장)
    - ddgs: 폴백 검색
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── 선택적 임포트 ────────────────────────────────────────────────────────────

try:
    from tavily import TavilyClient
    _tavily_client: Optional[TavilyClient] = None

    def _get_tavily() -> Optional[TavilyClient]:
        global _tavily_client
        if _tavily_client is None:
            key = os.environ.get("TAVILY_API_KEY")
            if key:
                _tavily_client = TavilyClient(api_key=key)
            else:
                logger.warning("[collector] TAVILY_API_KEY 미설정 — Tavily 검색 비활성화")
        return _tavily_client
except ImportError:
    def _get_tavily():
        return None

try:
    import trafilatura
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

try:
    from ddgs import DDGS
    _HAS_DDGS = True
except ImportError:
    _HAS_DDGS = False


_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Naver API 인증
_NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID")
_NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")


# ── 토픽별 영문 키워드 매핑 ────────────────────────────────────────────────────
# 국제 이슈가 많아 영문 검색이 품질/양 모두 유리함
_TOPIC_EN_KEYWORDS: Dict[str, List[str]] = {
    "tech_001": ["AI job creation vs job displacement", "artificial intelligence employment impact"],
    "tech_002": ["data center energy regulation vs AI infrastructure", "data center environmental impact"],
    "tech_003": ["AI safety vs model performance", "AI ethics trustworthiness competitiveness"],
    "econ_001": ["Iran war economic impact US Middle East intervention", "Iran geopolitical conflict global economy"],
    "econ_002": ["Trump tariff policy stagflation", "US tariff manufacturing reshoring impact"],
    "econ_003": ["Middle East crisis food security vs oil supply", "global food insecurity conflict"],
    "poli_001": ["Pentagon press access restriction freedom of press", "military media access First Amendment"],
    "poli_002": ["India green steel mandatory procurement", "green steel policy industry monopoly"],
    "poli_003": ["India GM crop food security vs agriculture tradition", "genetically modified crops debate"],
    "env_001": ["climate refugee vs conflict poverty migration", "climate change displacement cause"],
    "env_002": ["inflation energy policy vs climate change", "energy price inflation cause"],
    "env_003": ["climate crisis political participation vs science education", "climate action civic engagement"],
}


# ── 검색 쿼리 생성 ────────────────────────────────────────────────────────────

def generate_search_queries(topic: dict) -> List[str]:
    """토픽 정보를 기반으로 한국어 + 영어 다각적 검색 쿼리를 생성한다.

    [쿼리 카테고리]
        - 찬반 근거:   직접적인 찬성/반대 논거
        - 통계/데이터:  수치 기반 근거
        - 배경 지식:    논제의 맥락과 역사
        - 유사 사례:    해외/과거 사례
        - 전문가 의견:  기관/학자 발언
        - 반론:        상대방 예상 반론과 그에 대한 재반박
        - 영문 검색:    국제 이슈 대응 (Tavily에서 특히 효과적)
    """
    title = topic["title"]
    pro = topic.get("pro", "")
    con = topic.get("con", "")
    desc = topic.get("description_long", "")
    topic_id = topic.get("id", "")

    desc_first = desc.split("\n")[0].strip() if desc else ""
    short_title = (
        title.split("은 ")[0] if "은 " in title
        else title.split("의 ")[0] if "의 " in title
        else title
    )

    queries = [
        # ── 한국어: 찬반 근거 ────────────────────────────
        f"{title} 찬성 근거",
        f"{title} 반대 근거",
        f'"{pro}" 연구 결과',
        f'"{con}" 연구 결과',

        # ── 한국어: 통계/데이터 ──────────────────────────
        f"{short_title} 통계 수치 보고서",
        f"{short_title} 2024 2025 2026 데이터",

        # ── 한국어: 배경 지식 ────────────────────────────
        f"{short_title} 배경 현황 분석",
        f"{short_title} 원인 구조 메커니즘",

        # ── 한국어: 유사 사례 ────────────────────────────
        f"{short_title} 해외 사례 비교",
        f"{short_title} 성공 실패 사례",

        # ── 한국어: 전문가 의견 ──────────────────────────
        f"{short_title} 전문가 학자 견해",
        f"{short_title} 국제기구 보고서",

        # ── 한국어: 반론/논쟁 ────────────────────────────
        f"{pro} 비판 반론",
        f"{con} 비판 반론",
    ]

    if desc_first and len(desc_first) > 20:
        queries.append(f"{desc_first[:60]} 관련 논의")

    # ── 영문 쿼리 ────────────────────────────────────────
    en_keywords = _TOPIC_EN_KEYWORDS.get(topic_id, [])
    for kw in en_keywords:
        queries.append(f"{kw} statistics report 2024 2025 2026")
        queries.append(f"{kw} pros and cons evidence")

    return queries


# ── URL 본문 추출 ─────────────────────────────────────────────────────────────

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
                continue
            logger.warning("[collector] SSL 재시도도 실패 (%s)", url[:60])
            return None
        except Exception as e:
            logger.warning("[collector] HTTP 요청 실패 (%s): %s", url[:60], e)
            return None

    if resp is None:
        return None

    # trafilatura 우선
    if _HAS_TRAFILATURA:
        text = trafilatura.extract(resp.text, include_comments=False, include_tables=False)
        if text and len(text) > 200:
            return text

    # BS4 폴백
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]
    if not paragraphs:
        return None
    return "\n\n".join(paragraphs[:30])


# ── 검색 엔진별 구현 ─────────────────────────────────────────────────────────

def _search_tavily(query: str, max_results: int = 5) -> List[Dict]:
    """Tavily Search — AI 최적화 검색, 본문 자동 포함."""
    client = _get_tavily()
    if not client:
        return []

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            include_raw_content=True,
            search_depth="advanced",
        )
        articles = []
        for r in response.get("results", []):
            text = r.get("raw_content") or r.get("content", "")
            if text and len(text) > 100:
                articles.append({
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "text": text,
                    "source_type": "news",
                    "source": "tavily",
                })
        return articles
    except Exception as e:
        logger.warning("[collector] Tavily 검색 실패 (%s): %s", query[:40], e)
        return []


def _search_naver_news(query: str, max_results: int = 5) -> List[Dict]:
    """Naver News API — 한국어 뉴스 검색."""
    if not _NAVER_CLIENT_ID or not _NAVER_CLIENT_SECRET:
        return []

    try:
        resp = requests.get(
            "https://openapi.naver.com/v1/search/news.json",
            params={"query": query, "display": max_results, "sort": "sim"},
            headers={
                "X-Naver-Client-Id": _NAVER_CLIENT_ID,
                "X-Naver-Client-Secret": _NAVER_CLIENT_SECRET,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for item in data.get("items", []):
            # 원본 링크로 본문 추출 시도
            url = item.get("originallink") or item.get("link", "")
            text = extract_text_from_url(url) if url else None

            if not text:
                # 폴백: API의 description 사용
                import re
                desc = re.sub(r'<[^>]+>', '', item.get("description", ""))
                title = re.sub(r'<[^>]+>', '', item.get("title", ""))
                if len(desc) > 80:
                    text = f"{title}. {desc}"

            if text and len(text) > 100:
                articles.append({
                    "url": url,
                    "title": re.sub(r'<[^>]+>', '', item.get("title", "")),
                    "text": text,
                    "source_type": "news",
                    "source": "naver",
                })
        return articles
    except Exception as e:
        logger.warning("[collector] Naver 검색 실패 (%s): %s", query[:40], e)
        return []


def _search_ddgs(query: str, max_results: int = 3) -> List[Dict]:
    """DuckDuckGo 검색 — 폴백."""
    if not _HAS_DDGS:
        return []

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        articles = []
        for r in results:
            url = r.get("href", "")
            text = extract_text_from_url(url) if url else None

            if not text:
                snippet = r.get("body", "")
                title = r.get("title", "")
                if len(snippet) > 80:
                    text = f"{title}. {snippet}" if title else snippet

            if text:
                articles.append({
                    "url": url,
                    "title": r.get("title", ""),
                    "text": text,
                    "source_type": "news",
                    "source": "ddgs",
                })
        return articles
    except Exception as e:
        logger.warning("[collector] DDGS 검색 실패 (%s): %s", query[:40], e)
        return []


# ── 토픽별 수집 (통합) ───────────────────────────────────────────────────────

def collect_for_topic(
    topic: dict,
    target_count: int = 20,
    search_delay: float = 0.3,
) -> List[Dict]:
    """하나의 토픽에 대해 다중 소스로 기사를 수집한다.

    [수집 순서]
        1. 참고문헌(references) URL
        2. 쿼리별: Tavily → Naver News → DDGS 순으로 시도
        3. 중복 URL 자동 제거

    Args:
        topic:        topics.json의 단일 항목
        target_count: 목표 기사 수 (여유분 포함 2배 시도)
        search_delay: 검색 간 대기 시간 (초)

    Returns:
        수집된 기사 리스트
    """
    topic_id = topic.get("id", "unknown")
    raw_articles: List[Dict] = []
    seen_urls: set = set()
    fetch_target = target_count * 2

    def _add_article(article: Dict) -> bool:
        """중복 확인 후 기사를 추가한다. 추가되면 True."""
        url = article.get("url", "")
        if url and url in seen_urls:
            return False
        if url:
            seen_urls.add(url)
        raw_articles.append({
            "url": url,
            "title": article.get("title", ""),
            "text": article["text"],
            "source_type": article.get("source_type", "news"),
            "source": article.get("source", "unknown"),
            "collected_at": datetime.now().isoformat(),
        })
        return True

    # ── 1. 참고문헌 URL ──────────────────────────────────────────────────────
    for ref in topic.get("references", []):
        url = ref.get("url", "")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        text = extract_text_from_url(url)
        if text:
            _add_article({
                "url": url,
                "title": ref.get("title", ""),
                "text": text,
                "source_type": "report",
                "source": "reference",
            })
            logger.info("[collector] 참고문헌 수집: %s", ref.get("title", url[:40]))

    # ── 2. 쿼리별 다중 소스 검색 ─────────────────────────────────────────────
    queries = generate_search_queries(topic)

    # 검색 엔진 활성 상태 로그
    engines = []
    if _get_tavily():
        engines.append("Tavily")
    if _NAVER_CLIENT_ID:
        engines.append("Naver")
    if _HAS_DDGS:
        engines.append("DDGS")
    logger.info("[collector] %s: 활성 검색 엔진: %s", topic_id, engines)

    for query in queries:
        if len(raw_articles) >= fetch_target:
            break

        # Tavily (최우선 — 본문 자동 포함, 품질 높음)
        if _get_tavily() and len(raw_articles) < fetch_target:
            for article in _search_tavily(query, max_results=3):
                if len(raw_articles) >= fetch_target:
                    break
                _add_article(article)

        # Naver News (한국어 뉴스)
        if _NAVER_CLIENT_ID and len(raw_articles) < fetch_target:
            for article in _search_naver_news(query, max_results=3):
                if len(raw_articles) >= fetch_target:
                    break
                _add_article(article)

        # DDGS (폴백)
        if _HAS_DDGS and len(raw_articles) < fetch_target:
            for article in _search_ddgs(query, max_results=3):
                if len(raw_articles) >= fetch_target:
                    break
                _add_article(article)

        time.sleep(search_delay)

    # 소스별 통계
    source_counts: Dict[str, int] = {}
    for a in raw_articles:
        src = a.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    logger.info(
        "[collector] %s: %d개 수집 완료 — %s",
        topic_id, len(raw_articles), source_counts,
    )
    return raw_articles
