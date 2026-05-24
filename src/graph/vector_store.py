"""
vector_store.py — Pinecone 벡터 캐시 래퍼

Integrated Embedding (multilingual-e5-large, max 512 tokens) 사용.
긴 content는 청크 + 오버랩으로 저장 후 URL 기준 dedup 조회.

- 저장: 1 URL → N 청크 (500자 × 100 overlap)
- 조회: top_k chunks → URL별 best-score 1건만 + 원본 full_content 반환
- REST API 직접 호출 (urllib3 1.x latin-1 body 버그 회피)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from typing import Dict, Iterable, List, Optional, Set

import requests

logger = logging.getLogger(__name__)

# 진영별 사용된 URL 누적 (한 세션 = 한 subprocess 내).
# PRO 진영은 같은 진영 에이전트들끼리만 URL 중복 방지되고,
# CON 진영은 CON 진영끼리. 상대 진영은 각자 독립적으로 같은 URL 참조 가능
# (신뢰도 재검증·반박 근거 조사 등 목적).
# 각 subprocess는 모듈을 새로 import하므로 세션 간 자동 격리.
_used_urls_by_stance: Dict[str, Set[str]] = {}
_current_stance: Optional[str] = None
_url_lock = threading.Lock()


def set_current_stance(stance: Optional[str]) -> None:
    """현재 발언자의 진영을 설정 (에이전트 턴 진입 시 호출).

    "PRO" | "CON" | None (unset/공용). None이면 URL 제외 로직 비활성.
    """
    global _current_stance
    with _url_lock:
        _current_stance = stance


def get_current_stance() -> Optional[str]:
    with _url_lock:
        return _current_stance


def get_excluded_urls() -> Set[str]:
    """현재 발언자 진영에서 이미 사용된 URL 집합. stance 없으면 빈 집합."""
    with _url_lock:
        if not _current_stance:
            return set()
        return set(_used_urls_by_stance.get(_current_stance, set()))


def mark_urls_used(urls: Iterable[str]) -> None:
    """새로 반환된 URL을 현재 진영 사용 집합에 추가. stance 없으면 no-op."""
    with _url_lock:
        if not _current_stance:
            return
        s = _used_urls_by_stance.setdefault(_current_stance, set())
        for u in urls:
            if u:
                s.add(u)


def reset_session_urls() -> None:
    """같은 프로세스 내 재사용 시 호출."""
    global _current_stance
    with _url_lock:
        _used_urls_by_stance.clear()
        _current_stance = None

_SIMILARITY_THRESHOLD = float(os.environ.get("PINECONE_THRESHOLD", "0.85"))
_CHUNK_SIZE = int(os.environ.get("PINECONE_CHUNK_SIZE", "500"))
_CHUNK_OVERLAP = int(os.environ.get("PINECONE_CHUNK_OVERLAP", "100"))
_DEFAULT_NAMESPACE = "search-cache"
_INDEX_HOST: Optional[str] = None
_INIT_FAILED = False


def _resolve_host() -> Optional[str]:
    """Pinecone Control API로 index host 해석 (최초 1회만)."""
    global _INDEX_HOST, _INIT_FAILED
    if _INDEX_HOST is not None:
        return _INDEX_HOST
    if _INIT_FAILED:
        return None
    key = os.environ.get("PINECONE_API_KEY", "").strip()
    name = os.environ.get("PINECONE_INDEX", "debate-search-cache").strip()
    if not key:
        _INIT_FAILED = True
        logger.info("[vector_store] PINECONE_API_KEY 없음 — 캐시 비활성")
        return None
    try:
        r = requests.get(
            f"https://api.pinecone.io/indexes/{name}",
            headers={"Api-Key": key, "X-Pinecone-API-Version": "2025-04"},
            timeout=10,
        )
        r.raise_for_status()
        host = r.json().get("host", "")
        if not host.startswith("http"):
            host = f"https://{host}"
        _INDEX_HOST = host
        logger.info("[vector_store] 인덱스 호스트: %s", host)
        return host
    except Exception as e:
        _INIT_FAILED = True
        logger.warning("[vector_store] 호스트 해석 실패: %s", e)
        return None


def _headers(json_mode: bool = True) -> Dict[str, str]:
    key = os.environ.get("PINECONE_API_KEY", "")
    h = {
        "Api-Key": key,
        "X-Pinecone-API-Version": "2025-04",
    }
    h["Content-Type"] = "application/json" if json_mode else "application/x-ndjson"
    return h


def _url_to_id(url: str) -> str:
    return "doc_" + hashlib.md5(url.encode("utf-8", "ignore")).hexdigest()


def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> List[str]:
    """긴 텍스트를 size자 청크로 분할 (overlap자 겹침). 짧으면 1개 반환."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
        # 마지막 청크가 너무 짧으면 이전 청크와 합치기 (size의 30% 미만 시)
        if len(text) - start < size * 0.3 and chunks:
            # 남은 부분을 마지막 청크에 합치지 말고, 한 번 더 size로 끊기
            if start < len(text):
                chunks.append(text[start:])
            break
    return chunks


def upsert_search_results(
    query: str,
    results: List[Dict],
    namespace: str = _DEFAULT_NAMESPACE,
) -> int:
    """Tavily 검색 결과를 청크 단위로 Pinecone에 저장.

    각 결과(URL)당 content를 청크 여러개로 분할해 여러 vector로 저장.
    _id = `doc_{md5(url)}_c{i}` (URL unique + chunk index).
    `full_content` metadata에 원본 보존 (조회 시 반환용).

    Returns:
        저장된 총 vector(chunk) 수.
    """
    host = _resolve_host()
    if not host:
        return 0
    records = []
    for r in results:
        url = (r.get("url") or "").strip()
        content = (r.get("content") or "").strip()
        if not url or not content:
            continue
        base_id = _url_to_id(url)
        title = (r.get("title") or "")[:300]
        chunks = _chunk_text(content)
        full_content = content[:2200]  # metadata 저장용 (조회 시 반환)
        for i, ch in enumerate(chunks):
            records.append({
                "_id": f"{base_id}_c{i}",
                "text": ch,                    # 임베딩 대상
                "url": url,
                "title": title,
                "full_content": full_content,  # 조회 시 이거 반환
                "chunk_idx": i,
                "total_chunks": len(chunks),
                "query": query[:200],
            })
    if not records:
        return 0
    try:
        body = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
        resp = requests.post(
            f"{host}/records/namespaces/{namespace}/upsert",
            headers=_headers(json_mode=False),
            data=body.encode("utf-8"),
            timeout=30,
        )
        if resp.status_code >= 400:
            logger.warning("[vector_store upsert] HTTP %d: %s", resp.status_code, resp.text[:200])
            return 0
        logger.info("[vector_store] %d chunks 저장 (query=%s)", len(records), query[:30])
        return len(records)
    except Exception as e:
        logger.warning("[vector_store upsert] 실패: %s", e)
        return 0


def semantic_cache_lookup(
    query: str,
    top_k: int = 10,
    threshold: float = _SIMILARITY_THRESHOLD,
    max_urls: int = 3,
    namespace: str = _DEFAULT_NAMESPACE,
) -> Optional[List[Dict]]:
    """의미 유사도 검색 후 URL 단위 dedup.

    1) top_k 청크 검색
    2) URL별 best score 청크만 유지
    3) threshold 이상 URL만 반환 (최대 max_urls개)
    4) 반환할 때 full_content (원본) 사용

    Returns:
        [{url, title, content, score}, ...] 또는 None (miss)
    """
    host = _resolve_host()
    if not host:
        return None
    try:
        body = json.dumps({
            "query": {"top_k": int(top_k), "inputs": {"text": query[:2000]}},
        }, ensure_ascii=False)
        resp = requests.post(
            f"{host}/records/namespaces/{namespace}/search",
            headers=_headers(json_mode=True),
            data=body.encode("utf-8"),
            timeout=15,
        )
        if resp.status_code >= 400:
            logger.warning("[vector_store lookup] HTTP %d: %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        hits = data.get("result", {}).get("hits", []) or []
        if not hits:
            return None

        # URL별 best score 청크만 유지 (이미 사용된 URL 제외)
        excluded = get_excluded_urls()
        by_url: Dict[str, Dict] = {}
        for h in hits:
            score = float(h.get("_score", 0) or 0)
            fields = h.get("fields", {}) or {}
            url = fields.get("url", "")
            if not url or url in excluded:
                continue
            cur = by_url.get(url)
            if cur is None or score > cur["_score"]:
                by_url[url] = {"_score": score, "_fields": fields}

        # threshold 필터 + 상위 max_urls
        ranked = sorted(by_url.values(), key=lambda x: x["_score"], reverse=True)
        ranked = [x for x in ranked if x["_score"] >= threshold][:max_urls]
        if not ranked:
            return None

        out = []
        for x in ranked:
            f = x["_fields"]
            out.append({
                "url": f.get("url", ""),
                "title": f.get("title", ""),
                "content": f.get("full_content") or f.get("text", ""),  # 원본 우선
                "score": x["_score"],
            })
        return out or None
    except Exception as e:
        logger.warning("[vector_store lookup] 실패: %s", e)
        return None


def format_cached_results(results: List[Dict]) -> str:
    """캐시 결과를 search_web과 동일한 출력 포맷으로 직렬화."""
    if not results:
        return "[검색 결과 — vectorDB cache] 관련 결과를 찾을 수 없습니다."
    return "[검색 결과 — vectorDB cache]\n" + "\n".join(
        f"- {r['content'][:500]}" for r in results
    )
