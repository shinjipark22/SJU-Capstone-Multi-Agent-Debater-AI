"""토픽 기반 참고 링크 수집 — semantic cache 우선 + Tavily fallback.

DebateAssistant의 안내문 하단 "관련 링크 첨부" 영역을 채우기 위해 사용.
사용자가 ctx.links 를 manual 로 넘기면 그게 우선이고, 비어 있을 때만 이 모듈로
자동 수집한다.

토론 쪽 search_web 과 동일한 인덱스·인프라를 그대로 활용해 중복 노력 회피.

두 종류의 함수:
- fetch_topic_links: 토픽+진영 기반 (역할반전 등 focus_area 없을 때)
- fetch_focus_assets: focus_area 기반 — content(LLM 인용용) + links(UI용) 동시 반환
                      한 번 Tavily 호출로 tips와 링크가 같은 출처를 가리키도록.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from src.graph.llm import _tavily_client


logger = logging.getLogger(__name__)


_TAVILY_EXCLUDE = [
    "blog.naver.com",
    "m.blog.naver.com",
    "tistory.com",
    "brunch.co.kr",
    "linkedin.com",
    "medium.com",
    "velog.io",
    "daum.net",
]


# 어시스턴트와 토론 측이 동일 Pinecone namespace (search-cache) 를 공유한다.
# 분리했을 때 명분 (오염 방지) 대비 사용자가 올린 자료 (위키 등) 가 어시스턴트
# 측에 안 보이는 손실이 더 컸음 — 통합으로 자료 풀 일원화.
# 기존 'assistant-cache' 벡터는 일회 마이그레이션 후 비워짐.
ASSISTANT_NAMESPACE = "search-cache"

# 어시스턴트용 의미 유사도 임계값.
# 0.85 로 두니 phase 간 stance 만 다른 유사 쿼리 (찬성/반대 vs 동일 토픽 50자)에
# 캐시 hit 이 너무 많이 걸려 링크가 phase 마다 동일했음. 살짝 올려 phase 별
# 다양성을 확보.
ASSISTANT_SIMILARITY_THRESHOLD = 0.86


def _save_to_assistant_cache(query: str, items: list) -> None:
    """Tavily 결과를 어시스턴트 namespace 에 저장. 실패해도 조용히 넘어감."""
    if not items:
        return
    try:
        from src.graph.vector_store import upsert_search_results

        upsert_search_results(query, items, namespace=ASSISTANT_NAMESPACE)
    except Exception as e:
        logger.warning("[assistant_cache] upsert 실패: %s", e)


# text_guide.GuideLink 와 호환되는 dict 형태로 반환 (그쪽에서 dataclass 로 변환)
def fetch_topic_links(topic: str, stance: str, n: int = 3) -> List[dict]:
    """토픽+진영 기반으로 참고 링크 N개 수집.

    1) Pinecone 의미 캐시 lookup (이미 토론에서 쓴 자료 풀)
    2) miss 면 Tavily 직접 호출
    실패 시 빈 리스트 반환 — 호출자는 placeholder 노출.

    Parameters
    ----------
    topic : 토론 주제
    stance : "PRO" / "CON" — 검색 쿼리에 진영 관점을 살짝 더해 다양화
    n : 반환할 링크 수
    """
    stance_kr = "찬성" if stance == "PRO" else "반대"
    query = f"{topic} {stance_kr} 논거 통계 자료"

    # 1) semantic cache 우선 — 어시스턴트 전용 namespace 에서 조회
    try:
        from src.graph.vector_store import semantic_cache_lookup

        cached = semantic_cache_lookup(
            query, max_urls=n, threshold=ASSISTANT_SIMILARITY_THRESHOLD,
            namespace=ASSISTANT_NAMESPACE,
        )
        if cached:
            out = []
            for r in cached:
                url = r.get("url") or ""
                title = (r.get("title") or url or "참고 자료").strip()
                if not url:
                    continue
                out.append({"title": title[:80], "url": url, "summary": None})
            if out:
                return out[:n]
    except Exception as e:
        logger.warning("[fetch_topic_links] cache lookup 실패: %s", e)

    # 2) Tavily 직접 호출 → 결과를 어시스턴트 캐시에 저장
    try:
        results = _tavily_client.search(
            query,
            max_results=max(n + 2, 5),  # 필터 감안 여유분
            search_depth="advanced",
            exclude_domains=_TAVILY_EXCLUDE,
        )
        items = results.get("results", []) or []
        # 다음 사용자 요청에서 재사용할 수 있도록 어시스턴트 namespace 에 누적
        _save_to_assistant_cache(query, items)
        out: List[dict] = []
        for r in items:
            url = (r.get("url") or "").strip()
            if not url:
                continue
            title = (r.get("title") or url).strip()
            out.append({"title": title[:80], "url": url, "summary": None})
            if len(out) >= n:
                break
        return out
    except Exception as e:
        logger.warning("[fetch_topic_links] Tavily 실패: %s", e)
        return []


def fetch_focus_assets(focus_area: str, n: int = 3) -> Dict:
    """focus_area 키워드로 한 번 검색 → tips 본문 + 링크 동시 반환.

    한 번의 Tavily/Pinecone 호출로 두 결과를 동시에 만들어,
    어시스턴트 tips 에 인용된 자료원과 링크 영역에 노출되는 URL 이 동일하게 묶이도록 한다.

    Returns
    -------
    {
        "content": "[검색 결과]\\n- 본문 스니펫...\\n- ...",   # tips 사전 검색용 (LLM 인용)
        "links":   [{"title": ..., "url": ..., "summary": None}, ...]   # 링크 영역용
    }
    실패 시 두 필드 모두 비어 있음.
    """
    fa = (focus_area or "").strip()
    if not fa:
        return {"content": "", "links": []}

    # 1) Pinecone 의미 캐시 우선 — 어시스턴트 전용 namespace 에서 조회
    try:
        from src.graph.vector_store import semantic_cache_lookup

        cached = semantic_cache_lookup(
            fa, max_urls=n, threshold=ASSISTANT_SIMILARITY_THRESHOLD,
            namespace=ASSISTANT_NAMESPACE,
        )
        if cached:
            content_lines = [
                f"- {r.get('content', '')[:300]}"
                for r in cached if r.get("content")
            ]
            links_out = [
                {
                    "title": (r.get("title") or r.get("url") or "참고 자료").strip()[:80],
                    "url": r.get("url", ""),
                    "summary": None,
                }
                for r in cached if r.get("url")
            ]
            return {
                "content": ("[검색 결과]\n" + "\n".join(content_lines)) if content_lines else "",
                "links": links_out[:n],
            }
    except Exception as e:
        logger.warning("[fetch_focus_assets] cache lookup 실패: %s", e)

    # 2) Tavily 직접 호출 → 결과를 어시스턴트 캐시에 저장 + 반환
    try:
        results = _tavily_client.search(
            fa,
            max_results=max(n + 2, 5),
            search_depth="advanced",
            exclude_domains=_TAVILY_EXCLUDE,
        )
        items = results.get("results", []) or []
        # 다음 사용자 요청에서 재사용할 수 있도록 어시스턴트 namespace 에 누적
        _save_to_assistant_cache(fa, items)
        # URL 있는 것만, 최대 n개
        chosen = []
        for r in items:
            url = (r.get("url") or "").strip()
            if not url:
                continue
            chosen.append(r)
            if len(chosen) >= n:
                break

        content_lines = [f"- {r.get('content', '')[:300]}" for r in chosen if r.get("content")]
        links_out = [
            {
                "title": (r.get("title") or r.get("url")).strip()[:80],
                "url": r.get("url"),
                "summary": None,
            }
            for r in chosen
        ]
        return {
            "content": ("[검색 결과]\n" + "\n".join(content_lines)) if content_lines else "",
            "links": links_out,
        }
    except Exception as e:
        logger.warning("[fetch_focus_assets] Tavily 실패: %s", e)
        return {"content": "", "links": []}
