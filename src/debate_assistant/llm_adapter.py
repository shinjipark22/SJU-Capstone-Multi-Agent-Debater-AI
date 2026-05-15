"""DebateAssistant LLM 어댑터.

토론 측 자산(`src/graph/llm.py`의 Qwen2.5-32B-AWQ + search_web tool)을
text_guide.LLMCall (prompt → response 단순 시그니처)로 변환한다.

- tool calling 결합: 어시스턴트 tips 생성 시 LLM이 필요하면 search_web 호출
- 후처리 + 검증: 외국어 혼용·CoT 누설 차단
- 재시도: invoke_with_retry 통해 타임아웃·5xx 안전 처리
"""

from __future__ import annotations

import logging
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.llm import (
    invoke_with_retry,
    invoke_with_tools,
    llm_assistant,
    llm_assistant_with_tools,
)
from src.graph.text_utils import sanitize_assistant_text, validate_quality


logger = logging.getLogger(__name__)


# 어시스턴트 공용 시스템 프롬프트 (모든 tips 호출에 prepend)
ASSISTANT_SYSTEM = (
    "너는 토론하는 사용자의 친근한 어시스턴트다. 평가자도 코치도 아니다.\n"
    "\n"
    "[말투]\n"
    "- **편안한 반말체(해체)**로 작성해라. 어미는 \"~해\", \"~할 거야\", \"~지\", \"~겠지\", \"~봐\" 같이 자연스럽게.\n"
    "- 친한 동료가 옆에서 한마디 건네듯 편안하게. 단, 거칠거나 예의 없는 표현은 금지.\n"
    "- **이모티콘은 기본 0개. 정말 자연스러운 자리에서 1개만.** 거의 안 쓰는 게 낫다. 친근함은 어미·어휘로 충분.\n"
    "- 평가체(\"잘하셨습니다\", \"~하면 더 좋을 거야\") 금지. 친구 톤이지 코치 톤 아님.\n"
    "- 마크다운 강조(**굵게**)는 꼭 필요할 때만. 친근체에 과한 격식 무드 어울림 안 좋음.\n"
    "\n"
    "[언어·내용]\n"
    "- 한국어만. 영어/한자/일본어 단어 금지 (기관명·고유명사 제외).\n"
    "- 답변은 사용자에게 곧장 보여진다. 메타 멘트(\"제 안내는…\"), CoT 누설 금지.\n"
    "- 출력은 안내 본문만.\n"
    "\n"
    "[완결성 — 절대 어기지 마라]\n"
    "- 답변은 반드시 **완결된 문장**으로 끝낸다. 마지막 문장이 마침표/물음표/느낌표/명확한 종결어미(\"~지\", \"~어\", \"~야\" 등)로 끝나야 한다.\n"
    "- \"이렇게...\", \"...등\" 같이 줄임표나 미완 형태로 끊지 마라.\n"
    "- 분량 한계가 가까워지면 새 내용을 시작하지 말고 현재 문장을 깨끗이 마무리해라.\n"
    "- 응답 출력 전 마음속으로 한 번 점검: 마지막 문장이 완전한가?"
)


import re as _re

_SENTENCE_END_RE = _re.compile(r'[.?!다까요)\*"]$')

# 빈 불릿 패턴: 내용 없이 화살표/따옴표만 있는 미완 줄
#   예) `- "" → `, `- → `, `- ""`, `- → 에너지`
_EMPTY_BULLET_RE = _re.compile(r"^\s*[-*•·]\s*[\"'“”‘’]*\s*(?:→|->)?\s*[\"'“”‘’]*\s*$")


def _looks_unfinished(line: str) -> bool:
    """줄이 미완으로 보이는지 휴리스틱."""
    s = line.strip()
    if not s:
        return False
    if _EMPTY_BULLET_RE.match(s):
        return True
    # 불릿인데 화살표 뒤 내용이 거의 없음
    if s.startswith(("-", "*", "•", "·")):
        # 마지막 글자가 종결부호도 아니고, → 뒤 글자 수가 너무 짧으면 미완
        if not _SENTENCE_END_RE.search(s):
            # 화살표 뒤 콘텐츠 글자 수 측정
            m = _re.search(r"(?:→|->)\s*(.*)$", s)
            if m and len(m.group(1).strip()) < 4:
                return True
    return False


def _clean_truncated_lines(text: str) -> str:
    """본문 곳곳의 명백히 비어/미완인 불릿 라인 제거 + 꼬리 trim.

    `_trim_trailing_truncation` 의 강화판. validate_quality 통과 여부와
    무관하게 항상 실행해 시각적 깨짐을 막는다.
    """
    if not text:
        return text

    # 1) 본문 내 명백히 빈 불릿 제거
    kept = []
    for ln in text.split("\n"):
        if _EMPTY_BULLET_RE.match(ln.strip()):
            continue
        kept.append(ln)
    text = "\n".join(kept)

    # 2) 꼬리에서부터 미완 라인 제거 (sentence-end 만나면 stop)
    lines = text.rstrip().split("\n")
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if _SENTENCE_END_RE.search(last):
            break
        if last.startswith("###") or set(last) <= set("-—_=") or last in ("-", "*", "•"):
            lines.pop()
            continue
        if _looks_unfinished(last):
            lines.pop()
            continue
        # 길이 10 이상인 미완 일반 라인 → 잘라냄 후 break
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
        break
    # 연속 빈 줄 정리
    out = "\n".join(lines).rstrip()
    out = _re.sub(r"\n{3,}", "\n\n", out)
    return out


def _invoke(prompt: str, use_tools: bool, label: str) -> str:
    """내부 공통 호출 — tool 사용 여부에 따라 분기."""
    messages = [
        SystemMessage(content=ASSISTANT_SYSTEM),
        HumanMessage(content=prompt),
    ]
    if use_tools:
        raw, _ = invoke_with_tools(
            messages,
            label=label,
            tools_llm=llm_assistant_with_tools,
            final_llm=llm_assistant,
        )
    else:
        resp = invoke_with_retry(llm_assistant, messages, label=label)
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)

    cleaned = sanitize_assistant_text(raw)
    # 미완 꼬리·빈 불릿 제거는 validate 통과 여부와 무관하게 항상 적용
    cleaned = _clean_truncated_lines(cleaned)
    ok, reason = validate_quality(cleaned, min_chars=20)
    if not ok:
        logger.warning("[%s] 품질 검증 실패: %s (raw 80자: %r)", label, reason, raw[:80])
    return cleaned


def make_llm_call(use_tools: bool = True, label: str = "assistant_tips") -> Callable[[str], str]:
    """text_guide.LLMCall 어댑터 생성.

    Parameters
    ----------
    use_tools : True 이면 search_web tool calling 허용 (Qwen이 필요 시 호출).
                False 이면 순수 LLM (역할반전·종합처럼 history 기반 분석 단계).
    label : 로그 라벨.

    Returns
    -------
    Callable[[str], str] : prompt → 후처리된 응답
    """

    def _call(prompt: str) -> str:
        return _invoke(prompt, use_tools=use_tools, label=label)

    return _call
