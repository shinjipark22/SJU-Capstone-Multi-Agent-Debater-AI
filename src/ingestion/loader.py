"""
loader.py — ChromaDB 적재

새 스키마(v2)에 맞춰 라벨링된 청크를 ChromaDB에 적재한다.
기존 debate_docs 컬렉션과 별도로 debate_docs_v2 컬렉션을 사용한다.

[스키마]
    text, category, topic, topic_id, stance, stance_score,
    claim_type, source_type, source_name, source_url, language, ingested_at
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_CHROMA_PATH = str(Path(__file__).parent.parent.parent / "data" / "chroma_db")
_COLLECTION_NAME = "debate_docs_v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """임베딩 모델을 로드한다 (프로세스당 1회)."""
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")


@lru_cache(maxsize=1)
def _get_collection() -> chromadb.Collection:
    """ChromaDB 컬렉션을 초기화한다 (프로세스당 1회)."""
    model = _get_model()

    class _EF(chromadb.EmbeddingFunction):
        def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
            return model.encode(list(input), normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_EF(),
        metadata={"hnsw:space": "cosine"},
    )


def upsert_documents(documents: List[Dict]) -> int:
    """라벨링된 문서 리스트를 ChromaDB에 적재한다.

    Args:
        documents: 라벨링 완료된 문서 리스트. 필수 키: "text"

    Returns:
        적재된 문서 수
    """
    collection = _get_collection()

    ids, texts, metadatas = [], [], []

    for doc in documents:
        text = doc["text"]
        doc_id = hashlib.md5(text.encode()).hexdigest()[:16]

        metadata = {
            "category": doc.get("category", ""),
            "topic": doc.get("topic", ""),
            "topic_id": doc.get("topic_id", ""),
            "stance": doc.get("stance", "NEUTRAL"),
            "stance_score": float(doc.get("stance_score", 0.0)),
            "claim_type": doc.get("claim_type", "evidence"),
            "source_type": doc.get("source_type", "news"),
            "source_name": doc.get("source_name", ""),
            "source_url": doc.get("source_url", ""),
            "language": doc.get("language", "ko"),
            "ingested_at": datetime.now().isoformat(),
        }

        ids.append(doc_id)
        texts.append(text)
        metadatas.append(metadata)

    if ids:
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        logger.info("[loader] %d개 문서 적재 완료 → %s", len(ids), _COLLECTION_NAME)

    return len(ids)


def get_collection_stats() -> Dict:
    """현재 컬렉션 통계를 반환한다."""
    collection = _get_collection()
    return {
        "name": _COLLECTION_NAME,
        "count": collection.count(),
        "path": _CHROMA_PATH,
    }
