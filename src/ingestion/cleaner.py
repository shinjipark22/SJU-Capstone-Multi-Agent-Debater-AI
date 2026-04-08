"""
cleaner.py — 수집된 텍스트 전처리

HTML 잡음, 네비게이션 메뉴, 깨진 문자열 등을 제거하여
LLM 논증 추출의 입력 품질을 높인다.
"""

from __future__ import annotations

import re
from typing import Optional


def clean_text(text: str) -> Optional[str]:
    """수집된 기사 텍스트에서 노이즈를 제거한다.

    Returns:
        정제된 텍스트. 유의미한 내용이 없으면 None.
    """
    if not text:
        return None

    # 1. 바이너리/깨진 문자열 감지 — 비 ASCII 제어문자가 10% 이상이면 스킵
    control_chars = sum(1 for c in text if ord(c) < 32 and c not in '\n\r\t')
    if len(text) > 0 and control_chars / len(text) > 0.1:
        return None

    # 2. HTML 태그 제거
    text = re.sub(r'<[^>]+>', ' ', text)

    # 3. 마크다운 이미지/링크 잡음 제거
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)           # ![alt](url)
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)  # [text](url) → text

    # 4. URL 단독 줄 제거
    text = re.sub(r'^https?://\S+$', '', text, flags=re.MULTILINE)

    # 5. 네비게이션/메뉴 패턴 제거
    nav_patterns = [
        r'(?i)^(skip to|jump to|go to|back to|browse all|explore|sign up|log in|subscribe|cookie|privacy|terms of use|contact us).*$',
        r'(?i)^(share|facebook|twitter|linkedin|instagram|youtube|rss feed).*$',
        r'(?i)^(related content|related topics|more on|see all|read more|download pdf).*$',
        r'(?i)^(©|copyright).*$',
        r'(?i)^(search|menu|close|open)$',
    ]
    for pattern in nav_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)

    # 6. 반복되는 짧은 줄 제거 (메뉴 항목: 한 줄에 50자 미만이 5줄 이상 연속)
    lines = text.split('\n')
    cleaned_lines = []
    short_streak = 0
    short_buffer = []

    for line in lines:
        stripped = line.strip()
        if 0 < len(stripped) < 50:
            short_streak += 1
            short_buffer.append(stripped)
        else:
            if short_streak < 5:
                cleaned_lines.extend(short_buffer)
            # else: 5줄 이상 연속 짧은 줄 → 메뉴로 판단하여 제거
            short_streak = 0
            short_buffer = []
            cleaned_lines.append(stripped)

    if short_streak < 5:
        cleaned_lines.extend(short_buffer)

    text = '\n'.join(cleaned_lines)

    # 7. 과도한 공백 정리
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()

    # 8. 최소 길이 확인 — 정제 후 200자 미만이면 스킵
    if len(text) < 200:
        return None

    return text
