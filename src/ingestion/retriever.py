"""
retriever.py — 가중치 기반 multi-domain retrieval

category 기반 "제한" 대신 가중치(weight) 기반으로 검색한다.
cross-domain 근거도 포함하여 토론의 다양성을 확보한다.

[점수 산정]
    최종 점수 = cosine_similarity × domain_weight × credibility_weight

[Multi-domain 슬롯 비율]
    primary (해당 카테고리):  60%
    cross-domain (관련 분야): 30%
    기타:                     10%
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from src.ingestion.loader import _get_collection
from src.ingestion.source_config import (
    CREDIBILITY_WEIGHTS,
    DOMAIN_WEIGHTS,
    MULTI_DOMAIN_RATIO,
)

logger = logging.getLogger(__name__)


def _compute_domain_weight(
    doc_meta: dict,
    query_category: str,
) -> float:
    """문서의 도메인과 쿼리 카테고리 간 가중치를 계산한다."""
    primary = doc_meta.get("primary_domain", "")
    secondary_str = doc_meta.get("secondary_domains", "")
    secondaries = [s.strip() for s in secondary_str.split(",") if s.strip()]

    if primary == query_category:
        return DOMAIN_WEIGHTS["primary"]
    elif query_category in secondaries:
        return DOMAIN_WEIGHTS["secondary"]
    else:
        return DOMAIN_WEIGHTS["cross"]


def _compute_credibility_weight(doc_meta: dict) -> float:
    """문서의 공신력 등급 가중치를 반환한다."""
    tier = int(doc_meta.get("credibility_tier", 3))
    return CREDIBILITY_WEIGHTS.get(tier, 0.7)


def weighted_search(
    query: str,
    category: str,
    stance: Optional[str] = None,
    n_results: int = 5,
    exclude_ids: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], List[Dict]]:
    """가중치 기반 multi-domain 검색을 수행한다.

    1. ChromaDB에서 stance 필터만 적용하여 넓게 검색 (category 제한 없음)
    2. 각 결과에 domain_weight × credibility_weight 적용
    3. Multi-domain 비율에 맞춰 슬롯 배분

    Args:
        query:       검색 쿼리
        category:    쿼리의 주 카테고리 (tech/econ/poli/env)
        stance:      필터링할 stance (PRO/CON/None=전체)
        n_results:   최종 반환 문서 수
        exclude_ids: 제외할 문서 ID 리스트

    Returns:
        (texts, doc_ids, metadatas)
    """
    collection = _get_collection()
    exclude_ids = exclude_ids or []

    # 넓게 검색 (category 필터 없음, stance만 필터)
    fetch_n = min(n_results * 5 + len(exclude_ids), collection.count())
    if fetch_n == 0:
        return [], [], []

    where_filter = {"stance": stance} if stance else None

    try:
        results = collection.query(
            query_texts=[query],
            n_results=max(1, fetch_n),
            where=where_filter,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        logger.warning("[retriever] 검색 실패: %s", e)
        return [], [], []

    all_docs = results.get("documents", [[]])[0]
    all_ids = results.get("ids", [[]])[0]
    all_metas = results.get("metadatas", [[]])[0]

    if not all_docs:
        return [], [], []

    # ── 가중치 적용 ──────────────────────────────────────────────────────────
    scored: List[Tuple[float, int]] = []  # (score, index)

    for idx, (doc, doc_id, meta) in enumerate(zip(all_docs, all_ids, all_metas)):
        if doc_id in exclude_ids:
            continue

        # cosine similarity는 ChromaDB가 이미 정렬해줌 (index가 낮을수록 유사도 높음)
        # 역순위를 유사도 proxy로 사용
        similarity_proxy = 1.0 / (1 + idx * 0.1)

        domain_w = _compute_domain_weight(meta, category)
        credibility_w = _compute_credibility_weight(meta)

        final_score = similarity_proxy * domain_w * credibility_w
        scored.append((final_score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)

    # ── Multi-domain 슬롯 배분 ───────────────────────────────────────────────
    n_primary = max(1, int(n_results * MULTI_DOMAIN_RATIO["primary"]))
    n_cross = max(1, int(n_results * MULTI_DOMAIN_RATIO["cross"]))
    n_other = n_results - n_primary - n_cross

    primary_results = []
    cross_results = []
    other_results = []

    for score, idx in scored:
        meta = all_metas[idx]
        primary_domain = meta.get("primary_domain", "")
        secondary_str = meta.get("secondary_domains", "")

        if primary_domain == category and len(primary_results) < n_primary:
            primary_results.append(idx)
        elif category in secondary_str and len(cross_results) < n_cross:
            cross_results.append(idx)
        elif len(other_results) < n_other:
            other_results.append(idx)

        if len(primary_results) + len(cross_results) + len(other_results) >= n_results:
            break

    # 슬롯이 안 차면 나머지에서 채움
    selected_indices = primary_results + cross_results + other_results
    if len(selected_indices) < n_results:
        for score, idx in scored:
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= n_results:
                break

    selected_indices = selected_indices[:n_results]

    # ── 결과 구성 ────────────────────────────────────────────────────────────
    texts = [all_docs[i] for i in selected_indices]
    ids = [all_ids[i] for i in selected_indices]
    metas = [all_metas[i] for i in selected_indices]

    logger.info(
        "[retriever] 검색 완료: %d개 반환 (primary %d / cross %d / other %d)",
        len(texts), len(primary_results), len(cross_results), len(other_results),
    )

    return texts, ids, metas
