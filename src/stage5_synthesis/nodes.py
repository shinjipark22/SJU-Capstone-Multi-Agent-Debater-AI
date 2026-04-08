"""
nodes.py — 5단계: 종합 및 재개념화(Synthesis & Reconceptualization) 노드

[동작 흐름]
    1. 전체 토론 히스토리(입론~역할반전)를 요약
    2. 모든 AI 에이전트가 교차 순서로 각자의 최적해를 도출
    3. 사용자도 최적해를 API로 제출 (노드 외부)

[설계 노트]
    - 입장 고수가 아닌, 토론 전체를 종합한 최적해 도출이 목표
    - 양측 주장에서 타당한 부분을 인정하고, 구체적 해결책 제시
    - DeepSeek-R1-Distill-Qwen-14B 최적화
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import src.stage1_opening.nodes as _opening_mod
from src.stage1_opening.nodes import (
    _invoke_with_retry,
    _postprocess_speech,
    _extract_delimited_text,
    _is_valid_speech,
    _truncate_tool_result,
    search_web,
    _LLM_KWARGS,
)
from src.stage2_rebuttal.nodes import build_agent_stance_nums
from src.state import DebateEntry, DebateState

# ── 종합 전용 LLM ──────────────────────────────────────────────────────────
_syn_llm = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 4096, "temperature": 0.7})


# ── 토론 히스토리 요약 ────────────────────────────────────────────────────

def _summarize_debate(
    history: List[DebateEntry],
    agents: List[Dict],
    speaking_order: List[str],
) -> str:
    """전체 토론 히스토리를 단계별로 요약한다."""
    stance_nums = build_agent_stance_nums(agents, speaking_order)

    def _speaker_display(entry: DebateEntry) -> str:
        if entry["speaker_id"] == "user":
            return "사용자"
        s = entry["stance"]
        label = "찬성" if s == "PRO" else "반대"
        num = stance_nums.get(entry["speaker_id"], 1)
        return f"{label} 에이전트{num}"

    sections = []

    # 입론
    openings = [e for e in history if e["phase"] == "opening"]
    if openings:
        lines = ["[1단계: 입론]"]
        for e in openings:
            stance_kr = "찬성" if e["stance"] == "PRO" else "반대"
            lines.append(f"- {_speaker_display(e)} ({stance_kr}): {e['content'][:200]}")
        sections.append("\n".join(lines))

    # 연쇄논박
    rebuttals = [e for e in history if e["phase"] == "chained_rebuttal"]
    if rebuttals:
        lines = ["[2단계: 연쇄논박]"]
        for e in rebuttals:
            stance_kr = "찬성" if e["stance"] == "PRO" else "반대"
            target = e.get("target_id", "")
            lines.append(f"- {_speaker_display(e)} ({stance_kr}) → {target}: {e['content'][:150]}")
        sections.append("\n".join(lines))

    # 자유논박
    free = [e for e in history if e["phase"] == "free_rebuttal"]
    if free:
        lines = ["[3단계: 자유논박]"]
        for e in free:
            stance_kr = "찬성" if e["stance"] == "PRO" else "반대"
            lines.append(f"- {_speaker_display(e)} ({stance_kr}): {e['content'][:150]}")
        sections.append("\n".join(lines))

    # 역할반전
    rr = [e for e in history if e["phase"] == "role_reversal"]
    if rr:
        lines = ["[4단계: 역할반전]"]
        for e in rr:
            stance_kr = "찬성" if e["stance"] == "PRO" else "반대"
            lines.append(f"- {_speaker_display(e)} → {stance_kr} 옹호: {e['content'][:200]}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


# ── 종합 프롬프트 ──────────────────────────────────────────────────────────

def _build_synthesis_prompt(
    topic: str,
    original_stance: str,
    agent_name: str,
    debate_summary: str,
) -> str:
    """종합 및 재개념화 프롬프트를 생성한다."""
    stance_kr = "찬성" if original_stance == "PRO" else "반대"
    opposite_kr = "반대" if original_stance == "PRO" else "찬성"

    return f"""[5단계: 종합 및 재개념화]
지금까지의 토론을 종합하여, '{topic}'에 대한 **최적의 해결책**을 도출하라.

너는 원래 {stance_kr} 입장이었지만, 이제는 자기 입장에 집착하지 말고
양측 주장에서 타당한 부분을 인정하고 통합적인 최적해를 제시하라.

[토론 요약]
{debate_summary}

[지시]
- 찬성측과 반대측의 주장에서 각각 타당한 점을 명시적으로 인정하라
- 양측 주장을 종합하여 이 논제의 현실적이고 구체적인 최적해를 도출하라
- "A의 장점을 살리되 B의 문제를 해결하기 위해 C를 해야 한다"처럼 구체적으로 제안하라
- 추상적 표현("균형이 필요하다", "조화를 이루어야 한다") 금지. 구체적 정책/방안/행동을 제시하라

[구조]
- 찬성측 타당한 점 인정 (1~2줄)
- 반대측 타당한 점 인정 (1~2줄)
- 최적해 제시 (3~5줄, 구체적 방안 포함)
- 핵심 문장에 **강조** 사용

[형식]
- 한국어로 작성. 고유명사(기관명, 인명, 기술명)만 영어 허용
- 합니다체(격식체)

반드시 아래 형식으로만 출력:

### 답변 시작
### 찬성측 타당한 점
(내용)
### 반대측 타당한 점
(내용)
### 최적해
(구체적 해결책)
### 답변 끝"""


# ── 종합 발언 생성 ──────────────────────────────────────────────────────────

def _generate_synthesis(agent: Dict, prompt: str) -> Tuple[str, str]:
    """종합 발언 생성. 입론과 동일한 패턴."""
    messages = [
        SystemMessage(content=agent["system_prompt"]),
        HumanMessage(content=prompt),
    ]

    response: AIMessage = _invoke_with_retry(_syn_llm, messages, label="synthesis")
    raw = response.content if isinstance(response.content, str) else str(response.content)
    speech = _postprocess_speech(_extract_delimited_text(raw))

    if not _is_valid_speech(speech):
        logger.warning("[synthesis] speech 무효, 재시도")
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content="한국어로만 종합 발언을 작성하세요.\n\n### 답변 시작\n(발언)\n### 답변 끝"))
        retry: AIMessage = _invoke_with_retry(_syn_llm, messages, label="synthesis_retry")
        raw = retry.content if isinstance(retry.content, str) else str(retry.content)
        speech = _postprocess_speech(_extract_delimited_text(raw))

    # 영어 잔재 감지 → 수정 요청
    eng_words = re.findall(r'(?<![a-zA-Z])[a-z]{4,}(?![a-zA-Z])', speech)
    if eng_words:
        logger.warning("[synthesis] 영어 감지: %s → 수정 요청", eng_words[:3])
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content=f'다음 영어 단어를 한국어로 바꿔서 다시 작성하라: {", ".join(eng_words[:5])}\n한국어만 사용. 같은 형식 유지.'))
        fix: AIMessage = _invoke_with_retry(_syn_llm, messages, label="synthesis_fix_eng")
        raw_fix = fix.content if isinstance(fix.content, str) else str(fix.content)
        fixed = _postprocess_speech(_extract_delimited_text(raw_fix))
        if _is_valid_speech(fixed):
            speech = fixed
            raw = raw_fix

    return speech, raw


# ── 메인 노드 ──────────────────────────────────────────────────────────────

def synthesis_node(state: DebateState) -> DebateState:
    """5단계 종합 및 재개념화 노드.

    모든 AI 에이전트가 교차 순서로 최적해를 도출한다.
    사용자는 별도 API로 최적해를 제출한다.
    """
    _opening_mod._used_doc_ids = set()

    topic = state["topic"]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    agent_map = {a["agent_id"]: a for a in state["agents"]}
    speaking_order = state["speaking_order"]
    stance_nums = build_agent_stance_nums(state["agents"], speaking_order)

    # ── 토론 히스토리 요약
    debate_summary = _summarize_debate(history, state["agents"], speaking_order)

    print(f"\n[5단계: 종합 및 재개념화] 발언 순서: {speaking_order}\n")

    for speaker_id in speaking_order:
        if speaker_id == "user":
            continue

        agent = agent_map[speaker_id]
        slabel = "찬성" if agent["stance"] == "PRO" else "반대"
        snum = stance_nums.get(speaker_id, 1)
        display = f"{slabel} 에이전트{snum}"

        print(f"  [{display}] 종합 발언 생성 중...")

        # 프롬프트 구성 + LLM 호출
        prompt = _build_synthesis_prompt(
            topic=topic,
            original_stance=agent["stance"],
            agent_name=display,
            debate_summary=debate_summary,
        )
        final_text, raw = _generate_synthesis(agent, prompt)

        # fallback
        if not _is_valid_speech(final_text):
            logger.warning("[synthesis] fallback 사용")
            final_text = (
                f"### 찬성측 타당한 점\n"
                f"찬성측의 주장에도 타당한 근거가 있습니다.\n\n"
                f"### 반대측 타당한 점\n"
                f"반대측의 우려 역시 간과할 수 없습니다.\n\n"
                f"### 최적해\n"
                f"양측의 장점을 살린 구체적인 정책적 방안이 필요합니다."
            )

        entry = DebateEntry(
            turn=current_turn,
            speaker_id=speaker_id,
            stance=agent["stance"],
            phase="synthesis",
            content=final_text,
            target_id=None,
            tool_calls_log=[],
            json_raw=raw,
        )
        history.append(entry)
        current_turn += 1

        print(f"  [{display}] 종합 발언 완료 (turn={entry['turn']})")
        print(f"  {final_text[:100]}...\n")

    print(f"[5단계: 종합 및 재개념화] AI 발언 완료 → 사용자 종합 발언 대기\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "phase": "synthesis",
        "is_finished": False,
    })
