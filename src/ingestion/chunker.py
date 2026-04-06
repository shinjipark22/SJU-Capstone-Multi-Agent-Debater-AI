"""
chunker.py — 논증 단위 청킹

기사 본문을 토론 시스템에 적합한 150~400자 논증 단위로 분할한다.

[청킹 전략]
    - 문단(paragraph) 기반 분할을 기본으로 한다.
    - 짧은 문단은 다음 문단과 병합하여 최소 길이를 보장한다.
    - 긴 문단은 문장 경계에서 분할한다.
    - 목표: 하나의 청크 = 하나의 주장 + 그 근거가 포함된 독립적 단위
"""

from __future__ import annotations

import re
from typing import List

MIN_CHUNK_CHARS = 150
MAX_CHUNK_CHARS = 400


def _split_sentences(text: str) -> List[str]:
    """한국어/영어 문장 단위로 분할한다."""
    # 한국어 종결어미(다/요/임/음) 또는 영어 마침표/물음표/느낌표 뒤에서 분할
    sentences = re.split(r'(?<=[.!?다요임음])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
) -> List[str]:
    """텍스트를 논증 단위 청크로 분할한다.

    Args:
        text:      분할할 원본 텍스트
        min_chars: 청크 최소 길이 (기본 150자)
        max_chars: 청크 최대 길이 (기본 400자)

    Returns:
        분할된 청크 리스트
    """
    # 1. 문단 분할 (연속 개행 기준)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]

    chunks: List[str] = []
    buffer = ""

    for para in paragraphs:
        # 매우 짧은 단락 건너뛰기 (제목/캡션/광고 등)
        if len(para) < 30:
            continue

        # 버퍼 + 현재 문단이 최대 길이 이내면 병합
        if len(buffer) + len(para) + 1 <= max_chars:
            buffer = f"{buffer}\n{para}".strip() if buffer else para
            continue

        # 버퍼가 최소 길이 이상이면 확정
        if len(buffer) >= min_chars:
            chunks.append(buffer)
            buffer = ""
        elif buffer:
            # 버퍼가 너무 짧으면 현재 문단과 병합 시도
            merged = f"{buffer} {para}"
            if len(merged) <= max_chars:
                buffer = merged
                continue
            # 병합 불가 → 버퍼가 의미 있으면 그대로 저장
            if len(buffer) > 50:
                chunks.append(buffer)
            buffer = ""

        # 현재 문단 처리
        if len(para) <= max_chars:
            buffer = para
        else:
            # 긴 문단은 문장 경계에서 분할
            sentences = _split_sentences(para)
            for sent in sentences:
                if len(buffer) + len(sent) + 1 <= max_chars:
                    buffer = f"{buffer} {sent}".strip() if buffer else sent
                else:
                    if len(buffer) >= min_chars:
                        chunks.append(buffer)
                    buffer = sent

    # 마지막 버퍼 처리
    if buffer and len(buffer) >= min_chars:
        chunks.append(buffer)
    elif buffer and len(buffer) > 50:
        chunks.append(buffer)

    return chunks
