"""
run.py — ingestion 파이프라인 CLI 실행 스크립트

Usage:
    # 전체 12개 토픽 (기본)
    python -m src.ingestion.run

    # 특정 토픽만
    python -m src.ingestion.run --topics tech_001 econ_002

    # 토픽당 30개
    python -m src.ingestion.run --count 30

    # vLLM 서버 주소 지정
    VLLM_BASE_URL=http://localhost:8001/v1 python -m src.ingestion.run

사전 조건:
    - vLLM 서버 가동 중 (라벨링 단계에서 사용)
    - 인터넷 연결 (웹 검색 + 기사 추출)
    - (권장) pip install trafilatura — 기사 본문 추출 품질 향상
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion.pipeline import run_pipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="토론 VectorDB 데이터 수집 파이프라인",
    )
    parser.add_argument(
        "--topics",
        nargs="*",
        default=None,
        help="처리할 토픽 ID 목록 (미지정 시 전체 12개)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="토픽당 목표 청크 수 (기본 20)",
    )
    args = parser.parse_args()

    # ── 토픽 로드 ─────────────────────────────────────────────────────────────
    topics_path = ROOT / "data" / "topics_20260323_processed.json"
    if not topics_path.exists():
        print(f"토픽 파일을 찾을 수 없습니다: {topics_path}")
        sys.exit(1)

    with open(topics_path, encoding="utf-8") as f:
        topics_data = json.load(f)

    # 전체 토픽 목록 (카테고리 평탄화)
    all_topics = []
    for category_topics in topics_data.get("categories", {}).values():
        all_topics.extend(category_topics)

    # 특정 토픽 필터링
    if args.topics:
        all_topics = [t for t in all_topics if t["id"] in args.topics]
        if not all_topics:
            print(f"지정한 토픽을 찾을 수 없습니다: {args.topics}")
            print(f"사용 가능한 토픽: {[t['id'] for t in all_topics]}")
            sys.exit(1)

    print(f"대상: {len(all_topics)}개 토픽, 토픽당 {args.count}개 목표")
    print(f"토픽: {[t['id'] for t in all_topics]}")

    # ── 파이프라인 실행 ───────────────────────────────────────────────────────
    summary = run_pipeline(all_topics, target_per_topic=args.count)

    # 실패한 토픽 안내
    failed = [
        tid for tid, info in summary.get("per_topic", {}).items()
        if info.get("status") != "ok" or info.get("count", 0) == 0
    ]
    if failed:
        print(f"\n⚠ 다음 토픽은 데이터 수집에 실패했습니다: {failed}")
        print("  → 네트워크 상태 또는 vLLM 서버를 확인하세요.")


if __name__ == "__main__":
    main()
