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
지금까지의 토론을 종합하여, '{topic}'에 대한 **논쟁의 핵심 충돌을 해결하는 구조적 최적해**를 도출하라.

너는 원래 {stance_kr} 입장이었지만, 이제는 자기 입장에 집착하지 말고
양측 주장의 충돌 지점을 정확히 파악하고, 그 충돌을 해소하는 해결 구조를 설계하라.

[토론 요약]
{debate_summary}

[절대 금지]
- "재교육 강화", "정책적 지원 필요", "균형이 필요하다", "조화를 이루어야 한다" 같은 일반적·추상적 표현 금지
- 어떤 주제에든 그대로 붙여넣을 수 있는 범용 해결책 금지
- 단순 정책 나열 금지 (예: "첫째 ~, 둘째 ~, 셋째 ~" 식의 병렬 나열)

[지시]
1. 찬성측·반대측 타당한 점을 각각 1~2줄로 간결히 인정하라 (여기에 토큰을 낭비하지 마라)
2. "문제의 핵심 구조"를 반드시 분석하라:
   - 이 논쟁에서 찬성과 반대가 충돌하는 근본 원인이 무엇인지 밝혀라
   - 예: 속도 불균형, 비용 분배 문제, 정보 비대칭, 시간 지평 차이 등
3. 최적해는 다음을 반드시 포함하라:
   - **문제 재정의**: 이 논쟁을 어떤 문제로 다시 정의하는지
   - **작동 메커니즘**: 단순 정책이 아닌, "어떻게 작동하는지" 설명 (누가 → 무엇을 → 어떤 조건에서 → 어떤 결과를 만드는지)
   - **비용 부담 구조**: 이 해결책의 비용을 누가 부담하는지 (기업, 정부, 개인 등)
   - **왜 이것이 문제를 해결하는지**: 위에서 분석한 핵심 구조와 연결하여 설명
4. 이 주제에만 적용되는 해결책을 제시하라. '{topic}'의 고유한 맥락을 반영하라

[구조 — 대부분의 설명을 "최적해"에 집중하라]
- 찬성측 타당한 점 (1~2줄)
- 반대측 타당한 점 (1~2줄)
- 문제의 핵심 구조 (2~3줄: 충돌의 근본 원인)
- 최적해 (4~6줄: 문제 재정의 + 메커니즘 + 비용 부담 + 해결 근거)
- 핵심 문장에 **강조** 사용

[형식]
- 한국어로 작성. 고유명사(기관명, 인명, 기술명)만 영어 허용
- 합니다체(격식체)

반드시 아래 형식으로만 출력:

### 답변 시작
### 찬성측 타당한 점
(1~2줄)
### 반대측 타당한 점
(1~2줄)
### 문제의 핵심 구조
(충돌의 근본 원인 분석)
### 최적해
(문제 재정의 → 작동 메커니즘 → 비용 부담 → 해결 근거)
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
                f"찬성측의 핵심 근거에는 타당한 부분이 있습니다.\n\n"
                f"### 반대측 타당한 점\n"
                f"반대측이 지적한 문제 역시 현실적입니다.\n\n"
                f"### 문제의 핵심 구조\n"
                f"이 논쟁의 근본 원인은 양측이 서로 다른 시간 지평에서 문제를 바라보고 있기 때문입니다.\n\n"
                f"### 최적해\n"
                f"이 문제를 해결하려면 단기와 장기 관점을 분리하여 각각에 맞는 구체적 메커니즘을 설계해야 합니다."
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
