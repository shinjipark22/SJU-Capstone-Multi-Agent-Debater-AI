"""
crawler.py — GNews API 뉴스 크롤러

실행일 기준 과거 30일간의 뉴스를 4개 카테고리별로 수집한다.
GNews API 무료 플랜: 카테고리당 최대 10건, 100 req/일
"""

import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List

# ── 카테고리별 검색 쿼리 ───────────────────────────────────────────────────────
CATEGORY_QUERIES: Dict[str, str] = {
    "기술/AI":   "artificial intelligence",
    "경제/산업": "global economy",
    "정치/사회": "government policy",
    "과학/환경": "climate change",
}

GNEWS_API_BASE = "https://gnews.io/api/v4/search"
MAX_ARTICLES = 10  # 무료 플랜 최대치


class NewsCrawler:
    """GNews API 기반 뉴스 수집기."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def _fetch_category(self, query: str, from_dt: str, to_dt: str) -> List[Dict]:
        params = {
            "q":       query,
            "lang":    "en",
            "max":     MAX_ARTICLES,
            "from":    from_dt,   # ISO 8601: YYYY-MM-DDTHH:MM:SSZ
            "to":      to_dt,
            "sortby":  "relevance",
            "token":   self.api_key,
        }
        resp = requests.get(GNEWS_API_BASE, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json().get("articles", [])

    @staticmethod
    def _deduplicate(articles: List[Dict]) -> List[Dict]:
        """URL 기준으로 중복 기사를 제거한다."""
        seen: set = set()
        unique = []
        for art in articles:
            url = art.get("url", "")
            if url and url not in seen:
                seen.add(url)
                unique.append(art)
        return unique

    def fetch_all(self) -> Dict[str, List[Dict]]:
        """4개 카테고리 전체 뉴스를 수집해 반환한다."""
        now      = datetime.now(timezone.utc)
        to_dt    = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        from_dt  = (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

        results: Dict[str, List[Dict]] = {}
        for category, query in CATEGORY_QUERIES.items():
            print(f"  [크롤링] {category}  ({from_dt[:10]} ~ {to_dt[:10]}) ...")
            try:
                articles = self._fetch_category(query, from_dt, to_dt)
                articles = self._deduplicate(articles)
            except requests.RequestException as e:
                print(f"  [경고] {category} 뉴스 수집 실패: {e}")
                articles = []
            results[category] = articles
            print(f"          → {len(articles)}건 수집 완료")

        return results
