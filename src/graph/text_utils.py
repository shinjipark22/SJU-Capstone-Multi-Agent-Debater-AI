"""공유 텍스트 유틸 — 후처리·검증·delimiter 추출.

stage1_opening/nodes.py 에 흩어져 있던 공통 함수들을 한 곳에 모은 하부 모듈.
운영 코드 호환성을 위해 stage1_opening 의 원본 함수들을 그대로 재출력한다
(추후 stage1_opening 쪽 원본을 이쪽으로 이동시킬 수 있음).

어시스턴트(DebateAssistant)는 토론 발언과 톤이 달라(친근체 vs 격식체)
완전한 _postprocess_speech 대신 경량 sanitize 만 적용한다.
"""

from __future__ import annotations

import re

from src.phase1.stage1_opening.nodes import (  # noqa: F401
    _extract_delimited_text,
    _postprocess_speech,
    validate_quality,
)


__all__ = [
    "_extract_delimited_text",
    "_postprocess_speech",
    "validate_quality",
    "sanitize_assistant_text",
]


# 외국 문자 제거 (CJK·키릴·태국·아랍·라틴 확장·중국어 문장부호 등)
_FOREIGN_CHAR_RE = re.compile(
    r"[一-鿿㐀-䶿豈-﫿"
    r"぀-ゟ゠-ヿ"
    r"Ѐ-ӿ฀-๿؀-ۿ"
    r"Ā-ɏḀ-ỿÀ-ÿŐ-ſ"
    r"　-〿＀-｠]+"
)

# 도구 흔적·메타 표현
_TOOL_ARTIFACTS = [
    (re.compile(r"search_web"), ""),
    (re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL), ""),
    (re.compile(r"^.*(?:followed by|then the|end with).*$", re.MULTILINE), ""),
    (re.compile(r"검색 결과"), "관련 분석"),
    (re.compile(r"검색결과"), "관련 분석"),
]

# 영어 CoT 누설 패턴 (validate_quality 와 동일 — 사후 클린업도 시도)
_COT_HINT_RE = re.compile(
    r"\b(?:Okay|OK),?\s+so\b|"
    r"\bLet me\b|\bIn order to\b|"
    r"\bI (?:need|should|will|can|must)\b|"
    r"\b(?:First|Second|Third|Next|Then|Finally),?\s+I\b|"
    r"\bHmm\b",
    re.IGNORECASE,
)


def sanitize_assistant_text(text: str) -> str:
    """어시스턴트 응답용 경량 후처리.

    - 외국 문자(CJK·키릴 등) 제거
    - 영어 단독 줄 제거 (한글 1자 이상 있는 줄만 유지, 단 '###'·bullet 헤더는 예외)
    - 도구명/메타 표현 흔적 제거
    - 다중 공백·줄바꿈 정리

    `_postprocess_speech` 와 달리 격식체 변환·소제목 정규화·"검색 결과"→"관련 분석"
    같은 토론 발언 전용 규칙은 적용하지 않는다. 어시스턴트는 친근체를 유지한다.
    """
    if not text:
        return ""

    # 깨진 유니코드 → 제거
    text = text.replace("�", "")

    # 외국 문자 제거
    text = _FOREIGN_CHAR_RE.sub("", text)

    # 도구 흔적/메타 정리
    for pat, repl in _TOOL_ARTIFACTS:
        text = pat.sub(repl, text)

    # 영어 단독 줄 제거 (한글이 한 자도 없고 비어 있지 않은 줄)
    cleaned_lines = []
    for ln in text.split("\n"):
        stripped = ln.strip()
        if not stripped:
            cleaned_lines.append(ln)
            continue
        if stripped.startswith(("#", "-", "*", "•", "·")):
            cleaned_lines.append(ln)
            continue
        if re.search(r"[가-힣]", ln):
            cleaned_lines.append(ln)
            continue
        # 한글 없는 영어 전용 줄 → 드롭
    text = "\n".join(cleaned_lines)

    # CoT 누설 흔적 한 줄 통째로 제거 (validate 가 잡지만 LLM 재호출 비용 피하려고 클린업)
    text = _COT_HINT_RE.sub("", text)

    # 공백·줄바꿈 정리
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
