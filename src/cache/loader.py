"""
Speech cache loader — 사전 생성된 발언 캐시에서 variant 별 lookup.

지원 캐시 (5종, 모두 variant 1/2/3 슬롯):
  - AI 입론                       data/cache/openings/{topic}_{stance}_i{intensity}_f{focus}__v{N}.json
  - 어시스턴트 입론 가이드        data/cache/assistant_openings/{topic}_{stance}_f{focus}__v{N}.json
  - AI-AI 연쇄논박                data/cache/ai_rebuttals/{topic}_{att_stance}_af{af}_tf{tf}__v{N}.json
  - 역할반전 (AI)                 data/cache/role_reversals/{topic}_{reversed_stance}_i{intensity}__v{N}.json
  - 어시스턴트 역할반전 가이드    data/cache/assistant_role_reversals/{topic}_{reversed_stance}__v{N}.json

각 파일은 발언 1개 (`"speech": "..."`) 만 저장. variant 는 파일명 서픽스로 구분.

variant_idx 가 None 이면 cache miss 로 처리 — 호출자가 variant 를 정하지 못한 경로.

환경변수:
  SPEECH_CACHE_ENABLED=0  → 캐시 끄기 (벤치마크용)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent
_OPENING_CACHE = _ROOT / "data" / "cache" / "openings"
_ASSISTANT_CACHE = _ROOT / "data" / "cache" / "assistant_openings"
_RR_CACHE = _ROOT / "data" / "cache" / "role_reversals"
_AI_REBUTTAL_CACHE = _ROOT / "data" / "cache" / "ai_rebuttals"
_ASSISTANT_RR_CACHE = _ROOT / "data" / "cache" / "assistant_role_reversals"
_SEARCH_QUERIES_PATH = _ROOT / "data" / "search_queries.json"

# 프롬프트 버전 — 캐시 생성 스크립트와 일치해야 한다.
# prompt 가 의미 있게 바뀌면 양쪽 모두 올리고 캐시 재생성.
EXPECTED_PROMPT_VERSION = "v1"


def _cache_enabled() -> bool:
    return os.environ.get("SPEECH_CACHE_ENABLED", "1").strip() not in ("0", "false", "False", "")


# ── focus_idx 매핑 (focus_area 문자열 → search_queries 인덱스) ───────────
_SEARCH_QUERIES_CACHE: Optional[dict] = None


def _load_search_queries() -> dict:
    global _SEARCH_QUERIES_CACHE
    if _SEARCH_QUERIES_CACHE is not None:
        return _SEARCH_QUERIES_CACHE
    try:
        with _SEARCH_QUERIES_PATH.open(encoding="utf-8") as f:
            _SEARCH_QUERIES_CACHE = json.load(f)
    except Exception:
        _SEARCH_QUERIES_CACHE = {}
    return _SEARCH_QUERIES_CACHE


def focus_area_to_idx(topic_id: str, stance: str, focus_area: str) -> Optional[int]:
    """focus_area 문자열 → search_queries 내 인덱스. 못 찾으면 None."""
    if not (topic_id and stance and focus_area):
        return None
    queries = _load_search_queries().get(topic_id, {}).get(stance, [])
    try:
        return queries.index(focus_area)
    except ValueError:
        return None


# ── 캐시 로드 공통 ───────────────────────────────────────────────────────

def _read_speech(path: Path) -> Optional[str]:
    """캐시 파일 → 발언 문자열. version mismatch / 누락 시 None."""
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("[cache] %s 로드 실패: %s", path.name, e)
        return None
    if data.get("prompt_version") != EXPECTED_PROMPT_VERSION:
        logger.info(
            "[cache] %s prompt_version 불일치 (%s != %s) — skip",
            path.name, data.get("prompt_version"), EXPECTED_PROMPT_VERSION,
        )
        return None
    speech = data.get("speech")
    if not speech:
        return None
    return speech


def _valid_variant_idx(variant_idx: Optional[int]) -> bool:
    return isinstance(variant_idx, int) and variant_idx in (1, 2, 3)


# ── 공개 API ─────────────────────────────────────────────────────────────

def load_opening(
    topic_id: str, stance: str, intensity: int, focus_area: str,
    variant_idx: Optional[int],
) -> Optional[str]:
    """AI 입론 캐시 lookup."""
    if not _cache_enabled() or not _valid_variant_idx(variant_idx):
        return None
    focus_idx = focus_area_to_idx(topic_id, stance, focus_area)
    if focus_idx is None:
        return None
    path = _OPENING_CACHE / f"{topic_id}_{stance}_i{intensity}_f{focus_idx}__v{variant_idx}.json"
    speech = _read_speech(path)
    if speech is None:
        return None
    logger.info(
        "[cache] HIT opening %s/%s/i%d/f%d/v%d",
        topic_id, stance, intensity, focus_idx, variant_idx,
    )
    return speech


def load_assistant_opening(
    topic_id: str, stance: str, focus_area: str,
    variant_idx: Optional[int],
) -> Optional[str]:
    """어시스턴트 입론 가이드 캐시 lookup."""
    if not _cache_enabled() or not _valid_variant_idx(variant_idx):
        return None
    focus_idx = focus_area_to_idx(topic_id, stance, focus_area)
    if focus_idx is None:
        return None
    path = _ASSISTANT_CACHE / f"{topic_id}_{stance}_f{focus_idx}__v{variant_idx}.json"
    speech = _read_speech(path)
    if speech is None:
        return None
    logger.info(
        "[cache] HIT assistant_opening %s/%s/f%d/v%d",
        topic_id, stance, focus_idx, variant_idx,
    )
    return speech


def load_ai_rebuttal(
    topic_id: str, attacker_stance: str,
    attacker_focus_idx: int, target_focus_idx: int,
    variant_idx: Optional[int],
) -> Optional[str]:
    """AI-AI 연쇄논박 캐시 lookup. attacker_focus·target_focus 둘 다 인덱스로 매칭."""
    if not _cache_enabled() or not _valid_variant_idx(variant_idx):
        return None
    if not (topic_id and attacker_stance):
        return None
    path = _AI_REBUTTAL_CACHE / (
        f"{topic_id}_{attacker_stance}_af{attacker_focus_idx}_tf{target_focus_idx}__v{variant_idx}.json"
    )
    speech = _read_speech(path)
    if speech is None:
        return None
    logger.info(
        "[cache] HIT ai_rebuttal %s/%s/af%d/tf%d/v%d",
        topic_id, attacker_stance, attacker_focus_idx, target_focus_idx, variant_idx,
    )
    return speech


def load_role_reversal(
    topic_id: str, reversed_stance: str, intensity: int,
    variant_idx: Optional[int],
) -> Optional[str]:
    """역할반전 AI 발언 캐시 lookup."""
    if not _cache_enabled() or not _valid_variant_idx(variant_idx):
        return None
    if not (topic_id and reversed_stance):
        return None
    path = _RR_CACHE / f"{topic_id}_{reversed_stance}_i{intensity}__v{variant_idx}.json"
    speech = _read_speech(path)
    if speech is None:
        return None
    logger.info(
        "[cache] HIT role_reversal %s/%s/i%d/v%d",
        topic_id, reversed_stance, intensity, variant_idx,
    )
    return speech


def load_assistant_role_reversal(
    topic_id: str, reversed_stance: str,
    variant_idx: Optional[int],
) -> Optional[str]:
    """어시스턴트 역할반전 가이드 캐시 lookup.

    role_reversal 단계의 어시스턴트 가이드는 (topic, reversed_stance) 만으로 키 결정 —
    focus_area 적용 안 함 (text_guide 의 role_reversal phase 가 focus_area 안 씀).
    """
    if not _cache_enabled() or not _valid_variant_idx(variant_idx):
        return None
    if not (topic_id and reversed_stance):
        return None
    path = _ASSISTANT_RR_CACHE / f"{topic_id}_{reversed_stance}__v{variant_idx}.json"
    speech = _read_speech(path)
    if speech is None:
        return None
    logger.info(
        "[cache] HIT assistant_role_reversal %s/%s/v%d",
        topic_id, reversed_stance, variant_idx,
    )
    return speech
