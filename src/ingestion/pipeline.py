"""
pipeline.py — 전체 ingestion 파이프라인 오케스트레이터

[파이프라인 단계 (v2)]
    1. collect  : 토픽별 웹 검색 + 기사 본문 추출
    2. clean    : HTML 잡음·깨진 문자열 제거
    3. extract  : LLM이 기사를 통째로 읽고 논증 추출 (기존 chunk+label 대체)
    4. select   : PRO/CON 균형 맞춰 목표 수만큼 선별
    5. load     : ChromaDB 적재

[v1 대비 변경점]
    - chunker.py + labeler.py → extractor.py (LLM 기반 논증 추출)
    - cleaner.py 추가 (텍스트 전처리)
    - GPT가 기사를 통째로 읽으므로 네비게이션 노이즈 자동 무시
    - 기사당 2~3개의 고품질 논증 생성

각 단계의 중간 결과를 data/ingestion_results/에 JSON으로 저장하여 검증 가능하게 한다.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from src.ingestion.collector import collect_for_topic
from src.ingestion.cleaner import clean_text
from src.ingestion.extractor import extract_arguments
from src.ingestion.loader import upsert_documents

logger = logging.getLogger(__name__)

_RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "ingestion_results"


def _save_json(data, path: Path) -> None:
    """JSON 파일을 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_topic(topic: dict, target_chunks: int = 20) -> List[Dict]:
    """단일 토픽의 전체 파이프라인을 실행한다.

    Args:
        topic:         topics.json 항목
        target_chunks: 최종 목표 청크 수

    Returns:
        ChromaDB에 적재된 문서 리스트
    """
    topic_id = topic.get("id", "unknown")
    category = topic_id.split("_")[0] if "_" in topic_id else ""
    title = topic["title"]

    print(f"\n{'='*60}")
    print(f"[{topic_id}] {title}")
    print(f"{'='*60}")

    # ── 1단계: 수집 ──────────────────────────────────────────────────────────
    raw_articles = collect_for_topic(topic, target_count=target_chunks)
    _save_json(raw_articles, _RESULTS_DIR / f"{topic_id}_1_raw.json")
    print(f"  [1/5] 수집: {len(raw_articles)}개 기사")

    if not raw_articles:
        print(f"  [SKIP] 수집된 기사가 없습니다.")
        return []

    # ── 2단계: 텍스트 전처리 ─────────────────────────────────────────────────
    cleaned_articles: List[Dict] = []
    for article in raw_articles:
        cleaned = clean_text(article["text"])
        if cleaned:
            cleaned_articles.append({**article, "text": cleaned})

    _save_json(cleaned_articles, _RESULTS_DIR / f"{topic_id}_2_cleaned.json")
    skipped = len(raw_articles) - len(cleaned_articles)
    print(f"  [2/5] 전처리: {len(cleaned_articles)}개 통과, {skipped}개 제거")

    if not cleaned_articles:
        print(f"  [SKIP] 전처리 후 남은 기사가 없습니다.")
        return []

    # ── 3단계: LLM 논증 추출 ─────────────────────────────────────────────────
    all_arguments: List[Dict] = []
    for i, article in enumerate(cleaned_articles):
        arguments = extract_arguments(article["text"], topic)
        for arg in arguments:
            all_arguments.append({
                **arg,
                "source_url": article.get("url", ""),
                "source_name": article.get("title", ""),
                "source_type": article.get("source_type", "news"),
                "category": category,
                "topic": title,
                "topic_id": topic_id,
                "language": "ko",
            })

        if (i + 1) % 5 == 0:
            print(f"  [3/5] 논증 추출 진행: {i + 1}/{len(cleaned_articles)} 기사, "
                  f"누적 {len(all_arguments)}개 논증")

    _save_json(all_arguments, _RESULTS_DIR / f"{topic_id}_3_arguments.json")
    print(f"  [3/5] 논증 추출: {len(all_arguments)}개 (기사 {len(cleaned_articles)}개에서)")

    if not all_arguments:
        print(f"  [SKIP] 추출된 논증이 없습니다.")
        return []

    # ── 4단계: 선별 (PRO/CON 균형 + relevance 우선) ──────────────────────────
    pro_docs = [d for d in all_arguments if d["stance"] == "PRO"]
    con_docs = [d for d in all_arguments if d["stance"] == "CON"]
    neutral_docs = [d for d in all_arguments if d["stance"] == "NEUTRAL"]

    # relevance × stance_score로 정렬 (관련성 높고 입장이 명확한 것 우선)
    pro_sorted = sorted(
        pro_docs,
        key=lambda d: d["relevance_score"] * abs(d["stance_score"]),
        reverse=True,
    )
    con_sorted = sorted(
        con_docs,
        key=lambda d: d["relevance_score"] * abs(d["stance_score"]),
        reverse=True,
    )

    half = target_chunks // 2
    selected: List[Dict] = []
    selected.extend(pro_sorted[:half])
    selected.extend(con_sorted[:half])

    remaining = target_chunks - len(selected)
    if remaining > 0:
        extras = neutral_docs + pro_sorted[half:] + con_sorted[half:]
        extras_sorted = sorted(extras, key=lambda d: d["relevance_score"], reverse=True)
        selected.extend(extras_sorted[:remaining])

    selected = selected[:target_chunks]

    n_pro = len([d for d in selected if d["stance"] == "PRO"])
    n_con = len([d for d in selected if d["stance"] == "CON"])
    n_neu = len([d for d in selected if d["stance"] == "NEUTRAL"])
    print(f"  [4/5] 선별: {len(selected)}개 (PRO {n_pro} / CON {n_con} / NEUTRAL {n_neu})")

    # ── 5단계: ChromaDB 적재 ─────────────────────────────────────────────────
    loaded = upsert_documents(selected)
    _save_json(selected, _RESULTS_DIR / f"{topic_id}_4_final.json")
    print(f"  [5/5] 적재: {loaded}개 → ChromaDB")

    return selected


def run_pipeline(topics: List[dict], target_per_topic: int = 20) -> Dict:
    """전체 토픽에 대해 ingestion 파이프라인을 실행한다.

    Args:
        topics:           토픽 리스트
        target_per_topic: 토픽당 목표 청크 수

    Returns:
        실행 요약 딕셔너리
    """
    all_results: Dict[str, List] = {}
    total_loaded = 0

    print(f"\n{'#'*60}")
    print(f"# Ingestion Pipeline v2 — {len(topics)}개 토픽, 토픽당 {target_per_topic}개")
    print(f"{'#'*60}")

    for topic in topics:
        topic_id = topic.get("id", "unknown")
        try:
            results = process_topic(topic, target_per_topic)
            all_results[topic_id] = results
            total_loaded += len(results)
        except Exception as e:
            logger.error("[pipeline] %s 처리 실패: %s", topic_id, e, exc_info=True)
            all_results[topic_id] = []
            print(f"  [ERROR] {topic_id}: {e}")

    summary = {
        "total_topics": len(topics),
        "total_documents": total_loaded,
        "target_per_topic": target_per_topic,
        "per_topic": {
            tid: {"count": len(docs), "status": "ok" if docs else "empty"}
            for tid, docs in all_results.items()
        },
    }
    _save_json(summary, _RESULTS_DIR / "summary.json")

    print(f"\n{'#'*60}")
    print(f"# 완료: {len(topics)}개 토픽, 총 {total_loaded}개 문서 적재")
    print(f"# 결과: {_RESULTS_DIR}")
    print(f"{'#'*60}")

    return summary
