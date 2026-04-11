"""
reviewer.py — 발언 품질 검증 모듈

역할:
    - 입장 혼동 체크 (check_stance)
    - 끊김/불완전 문장 감지 (check_truncation)
    - 영어/외국어 혼용 감지 (check_language)
    - 종합 품질 검증 (review_speech) — 위 3가지를 한 번에
"""

from __future__ import annotations

import logging
import re

from src.graph.searcher import check_stance as _qwen_check_stance

logger = logging.getLogger(__name__)


def check_stance(speech: str, expected_stance: str, topic: str) -> bool:
    """Qwen 7B로 발언이 기대 입장과 일치하는지 검증."""
    return _qwen_check_stance(speech, expected_stance, topic)


def check_truncation(speech: str) -> bool:
    """끊김/불완전 문장이 있으면 True (문제 있음)."""
    # 조사+동사 바로 붙은 패턴
    if re.search(r'[을를이가은는에]합니다(?!\s*[.?!])', speech):
        return True
    # 혼합어 (한글+영어 3자 이상)
    if re.search(r'[가-힣]+[a-zA-Z]{3,}', speech):
        return True
    if re.search(r'[a-zA-Z]{3,}[가-힣]+[a-zA-Z]', speech):
        return True
    # 마지막 문장이 끝맺음 없이 끊김
    lines = speech.rstrip().split('\n')
    if lines:
        last = lines[-1].rstrip()
        if last and not last.startswith('###') and not re.search(r'[.?!다까요\*]$', last):
            return True
    return False


def check_language(speech: str) -> bool:
    """영어/외국어가 과도하게 혼용되었으면 True (문제 있음)."""
    # 영어 비율 계산
    korean_chars = len(re.findall(r'[가-힣]', speech))
    english_chars = len(re.findall(r'[a-zA-Z]', speech))
    if korean_chars + english_chars > 0:
        ratio = english_chars / (korean_chars + english_chars)
        if ratio > 0.3:
            return True
    # 한국어 중간에 소문자 영어 5자+ 단어
    if re.findall(r'(?<=[가-힣\s])[a-z]{5,}(?=[가-힣\s.,])', speech, re.IGNORECASE):
        return True
    return False


def review_speech(
    speech: str,
    expected_stance: str,
    topic: str,
    check_stance_flag: bool = True,
) -> dict:
    """종합 품질 검증. 모든 체크를 한 번에 수행.

    Returns:
        {
            "passed": bool,          # 전체 통과 여부
            "stance_ok": bool,       # 입장 일치
            "truncation_ok": bool,   # 끊김 없음
            "language_ok": bool,     # 언어 혼용 없음
            "issues": list[str],     # 발견된 문제 목록
        }
    """
    issues = []

    stance_ok = True
    if check_stance_flag:
        stance_ok = check_stance(speech, expected_stance, topic)
        if not stance_ok:
            issues.append(f"입장 혼동: {expected_stance} 기대")

    truncation_ok = not check_truncation(speech)
    if not truncation_ok:
        issues.append("생성 끊김/불완전 문장")

    language_ok = not check_language(speech)
    if not language_ok:
        issues.append("영어/외국어 과다 혼용")

    passed = stance_ok and truncation_ok and language_ok

    if not passed:
        logger.warning("[reviewer] 품질 이슈: %s", ", ".join(issues))

    return {
        "passed": passed,
        "stance_ok": stance_ok,
        "truncation_ok": truncation_ok,
        "language_ok": language_ok,
        "issues": issues,
    }
