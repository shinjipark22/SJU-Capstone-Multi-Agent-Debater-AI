"""
vector_db.py — 토론 VectorDB 검색 (v2)

ingestion 파이프라인이 적재한 debate_docs_v2 컬렉션을 사용한다.
가중치 기반 multi-domain retrieval로 검색 품질을 높인다.

[v1 → v2 변경점]
    - 하드코딩 샘플 8개 → ingestion 파이프라인이 적재한 600개 실데이터
    - topic 필터 → 의미 유사도 검색 (토픽 문자열 불일치 문제 해결)
    - 단순 cosine → cosine × credibility_weight 점수 산정
    - 에이전트 간 중복 인용 방지 (exclude_ids) 유지
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_CHROMA_PATH = str(Path(__file__).parent.parent.parent / "data" / "chroma_db")
_COLLECTION_NAME = "debate_docs_v2"

# 공신력 등급별 가중치
_CREDIBILITY_WEIGHTS = {1: 1.0, 2: 0.85, 3: 0.7}


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    """debate_docs_v2 컬렉션을 로드한다."""
    model = _get_model()

    class _EF(chromadb.EmbeddingFunction):
        def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
            return model.encode(list(input), normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_EF(),
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        logger.warning(
            "[vector_db] %s 컬렉션이 비어있습니다. "
            "python -m src.ingestion.run 으로 데이터를 적재하세요.",
            _COLLECTION_NAME,
        )

    return collection


def query_vector_db(
    query: str,
    topic: str,
    stance: str,
    n_results: int = 2,
    exclude_ids: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """의미 유사도 + 공신력 가중치 기반 검색.

    topic 문자열로 하드필터하지 않고, 의미 유사도로 관련 문서를 찾는다.
    공신력(credibility_tier)이 높은 문서에 가중치를 부여한다.

    Args:
        query:       검색 질의 (자연어)
        topic:       토론 주제 (검색 쿼리에 결합하여 관련성 향상)
        stance:      "PRO" 또는 "CON"
        n_results:   반환할 문서 수
        exclude_ids: 제외할 문서 ID 리스트

    Returns:
        (문서 텍스트 리스트, 문서 ID 리스트)
    """
    collection = _get_collection()
    exclude_ids = exclude_ids or []

    if collection.count() == 0:
        return [], []

    # 넓게 검색 (stance만 필터, topic은 쿼리에 결합)
    fetch_n = min(n_results * 4 + len(exclude_ids), collection.count())

    try:
        results = collection.query(
            query_texts=[f"{topic} {query}"],
            n_results=max(1, fetch_n),
            where={"stance": stance},
            include=["documents", "metadatas"],
        )
    except Exception:
        try:
            results = collection.query(
                query_texts=[f"{topic} {query}"],
                n_results=1,
                where={"stance": stance},
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning("[vector_db] query 실패 (stance=%s): %s", stance, e)
            return [], []

    all_docs: List[str] = results.get("documents", [[]])[0]
    all_ids: List[str] = results.get("ids", [[]])[0]
    all_metas: list = results.get("metadatas", [[]])[0]

    # ── 가중치 점수 산정 + exclude 필터 ──────────────────────────────────────
    exclude_set = set(exclude_ids)
    scored = []

    for idx, (doc, doc_id) in enumerate(zip(all_docs, all_ids)):
        if doc_id in exclude_set:
            continue

        # ChromaDB는 cosine 유사도 순으로 반환 → 역순위를 유사도 proxy로 사용
        similarity_proxy = 1.0 / (1 + idx * 0.1)

        # 공신력 가중치
        meta = all_metas[idx] if idx < len(all_metas) else {}
        tier = int(meta.get("credibility_tier", 3))
        cred_weight = _CREDIBILITY_WEIGHTS.get(tier, 0.7)

        final_score = similarity_proxy * cred_weight
        scored.append((final_score, idx))

    # 점수 내림차순 정렬
    scored.sort(key=lambda x: x[0], reverse=True)

    # 상위 n_results 반환
    selected = scored[:n_results]

    if not selected:
        return [], []

    docs = [all_docs[idx] for _, idx in selected]
    ids = [all_ids[idx] for _, idx in selected]

    return docs, ids
