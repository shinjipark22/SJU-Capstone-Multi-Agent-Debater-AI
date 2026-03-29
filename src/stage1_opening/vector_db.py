"""
vector_db.py — 토론 전문가 문서 VectorDB (ChromaDB 기반)

[구조]
    - ChromaDB in-memory 컬렉션에 주제별·진영별 샘플 문서를 적재
    - sentence-transformers(paraphrase-multilingual-MiniLM-L12-v2)로 임베딩
    - topic / stance 메타데이터 필터링 후 코사인 유사도 검색

[확장 포인트]
    - 실제 운용 시 in-memory → persistent 모드로 전환 (chromadb.PersistentClient)
    - 샘플 문서를 실제 논문·기사 데이터로 교체
    - 임베딩 모델을 도메인 특화 모델로 교체
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

# ── 샘플 문서 ─────────────────────────────────────────────────────────────────
# (topic, stance, text) 형식
# 실제 운용 시 이 목록을 DB에서 로드하거나 크롤링 데이터로 교체할 것

_SAMPLE_DOCS = [
    # ── AI & 일자리 ────────────────────────────────────────────────────────
    (
        "인공지능 발전과 일자리",
        "PRO",
        "세계경제포럼(WEF) 2023 보고서에 따르면 AI 도입으로 2027년까지 "
        "6,900만 개의 새로운 일자리가 창출될 전망이다. "
        "AI 전문가·데이터 엔지니어·자동화 설계자 등 새로운 직군이 빠르게 성장하고 있다.",
    ),
    (
        "인공지능 발전과 일자리",
        "PRO",
        "맥킨지 글로벌 인스티튜트는 AI가 반복 업무를 자동화하는 동시에 "
        "인간 고유의 창의성·공감·복잡한 의사결정 역할을 강화한다고 분석했다. "
        "생산성 향상이 경제 성장을 이끌어 장기적으로 고용을 늘린다.",
    ),
    (
        "인공지능 발전과 일자리",
        "CON",
        "옥스퍼드대 연구(Frey & Osborne, 2013)는 미국 직업의 47%가 "
        "향후 20년 내 자동화 위험에 처해 있다고 경고했다. "
        "특히 운송·물류·사무직 등 중산층 일자리가 가장 취약하다.",
    ),
    (
        "인공지능 발전과 일자리",
        "CON",
        "IMF 2024 보고서는 선진국 전체 일자리의 약 40%가 AI 영향권에 있으며 "
        "고소득 국가일수록 노출 비율이 높다고 밝혔다. "
        "일자리 창출보다 대체 속도가 빠를 경우 구조적 실업이 심화될 수 있다.",
    ),
    # ── 기본소득 ───────────────────────────────────────────────────────────
    (
        "기본소득",
        "PRO",
        "핀란드 기본소득 실험(2017~2018)에서 수급자들의 정신 건강·신뢰도·고용 의욕이 "
        "대조군 대비 유의미하게 향상되었다. 경제적 안정이 창업·재교육 동기를 높인다.",
    ),
    (
        "기본소득",
        "CON",
        "기본소득 재원 마련을 위한 증세는 중산층과 소기업에 부담을 집중시킨다. "
        "노동 공급 감소로 인한 생산성 하락 우려도 있으며, "
        "복지 효율성 측면에서 표적 지원보다 비효율적이라는 비판이 있다.",
    ),
    # ── 환경 규제 ──────────────────────────────────────────────────────────
    (
        "환경 규제",
        "PRO",
        "IPCC 6차 보고서는 2030년까지 탄소 배출을 45% 감축하지 않으면 "
        "1.5°C 목표 달성이 불가능하다고 경고한다. "
        "엄격한 환경 규제가 녹색 기술 혁신과 신산업 창출로 이어진 사례가 다수 존재한다.",
    ),
    (
        "환경 규제",
        "CON",
        "과도한 환경 규제는 개발도상국의 산업화를 가로막아 빈곤 해소를 지연시킨다. "
        "탄소세 도입 시 에너지 비용 상승이 저소득층에 역진적으로 작용할 수 있다.",
    ),
]


# ── ChromaDB 초기화 (프로세스당 1회) ──────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


_CHROMA_PATH = str(Path(__file__).parent.parent.parent / "data" / "chroma_db")


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    """ChromaDB 퍼시스턴트 컬렉션을 초기화하고 샘플 문서를 적재한다.

    data/chroma_db/ 디렉터리에 임베딩 결과를 저장하므로
    최초 실행 후에는 임베딩 재계산 없이 즉시 로드된다.
    """
    model = _get_model()

    # SentenceTransformer를 ChromaDB 커스텀 임베딩 함수로 래핑
    class _EF(chromadb.EmbeddingFunction):
        def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
            return model.encode(list(input), normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=_CHROMA_PATH)  # 디스크 저장
    collection = client.get_or_create_collection(
        name="debate_docs",
        embedding_function=_EF(),
        metadata={"hnsw:space": "cosine"},
    )

    # 이미 적재됐으면 skip
    if collection.count() > 0:
        return collection

    ids, documents, metadatas = [], [], []
    for topic, stance, text in _SAMPLE_DOCS:
        doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
        ids.append(doc_id)
        documents.append(text)
        metadatas.append({"topic": topic, "stance": stance})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


# ── 공개 검색 함수 ─────────────────────────────────────────────────────────────

def query_vector_db(
    query: str,
    topic: str,
    stance: str,
    n_results: int = 2,
    exclude_ids: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """topic·stance 메타데이터 필터 + 의미 유사도 검색으로 관련 문서를 반환한다.

    이미 다른 에이전트가 인용한 문서 ID를 exclude_ids로 전달하면
    해당 문서를 건너뛰고 다음 순위 문서를 반환하여 에이전트 간 중복 인용을 방지한다.

    Args:
        query:       검색 질의 (자연어)
        topic:       메타데이터 필터 — 토론 주제 키워드
        stance:      메타데이터 필터 — "PRO" 또는 "CON"
        n_results:   반환할 문서 수
        exclude_ids: 제외할 문서 ID 리스트 (이미 다른 에이전트가 사용한 문서)

    Returns:
        (문서 텍스트 리스트, 문서 ID 리스트) — 둘 다 없으면 ([], [])
    """
    collection = _get_collection()
    exclude_ids = exclude_ids or []

    # 제외 문서 수만큼 여유분을 더 fetch하여 필터링 후에도 n_results를 채울 수 있도록 함
    fetch_n = min(n_results + len(exclude_ids), collection.count())

    results = collection.query(
        query_texts=[f"{topic} {query}"],
        n_results=max(1, fetch_n),
        where={"stance": stance},
        include=["documents", "ids"],
    )

    all_docs: List[str] = results.get("documents", [[]])[0]
    all_ids: List[str] = results.get("ids", [[]])[0]

    # 이미 사용된 문서 ID 제외
    exclude_set = set(exclude_ids)
    filtered = [
        (doc, doc_id)
        for doc, doc_id in zip(all_docs, all_ids)
        if doc_id not in exclude_set
    ][:n_results]

    if not filtered:
        return [], []

    docs, ids = zip(*filtered)
    return list(docs), list(ids)
