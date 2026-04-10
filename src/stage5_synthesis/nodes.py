"""
nodes.py — 5단계: 종합 및 재개념화(Synthesis & Reconceptualization) 노드

[동작 흐름]
    1. 전체 토론 히스토리(입론~역할반전)를 요약
    2. 모든 AI 에이전트가 교차 순서로 최적해 의견 제시 (회의 발언)
    3. 사용자와 AI가 최적해에 대해 회의 (자유논박과 유사한 토론)
    4. 사용자가 대표로 "우리의 최적해"를 최종 확정

[설계 노트]
    - 회의식 토론: AI가 먼저 의견을 제시하고, 사용자와 주고받으며 다듬음
    - 사용자가 최종 의사결정자 — "우리의 최적해: ~"로 확정
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
    _LLM_KWARGS,
)
from src.stage2_rebuttal.nodes import (
    _extract_rebuttal_text,
    _is_valid_rebuttal,
    build_agent_stance_nums,
)
from src.state import DebateEntry, DebateState

# ── 종합 전용 LLM ──────────────────────────────────────────────────────────
_syn_llm = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 1024, "temperature": 0.7})


# ── 토론 히스토리 요약 ────────────────────────────────────────────────────

def _summarize_debate(
    history: List[DebateEntry],
    agents: List[Dict],
    speaking_order: List[str],
) -> str:
    """전체 토론 히스토리를 단계별로 요약한다.

    참여자 수에 따라 절단 길이를 자동 조절하여 컨텍스트 초과를 방지한다.
    """
    stance_nums = build_agent_stance_nums(agents, speaking_order)
    num_speakers = len(speaking_order)

    # 참여자 수에 따라 절단 길이 조절 (3:3이면 짧게)
    if num_speakers <= 4:  # 2:2
        len_long, len_short = 200, 150
    elif num_speakers <= 6:  # 3:3
        len_long, len_short = 120, 80
    else:
        len_long, len_short = 80, 60

    def _speaker_display(entry: DebateEntry) -> str:
        if entry["speaker_id"] == "user":
            return "사용자"
        s = entry["stance"]
        label = "찬성" if s == "PRO" else "반대"
        num = stance_nums.get(entry["speaker_id"], 1)
        return f"{label}{num}"

    sections = []

    for phase_name, phase_label in [
        ("opening", "입론"),
        ("chained_rebuttal", "연쇄논박"),
        ("free_rebuttal", "자유논박"),
        ("role_reversal", "역할반전"),
    ]:
        entries = [e for e in history if e["phase"] == phase_name]
        if not entries:
            continue
        max_len = len_long if phase_name in ("opening", "role_reversal") else len_short
        lines = [f"[{phase_label}]"]
        for e in entries:
            stance_kr = "찬성" if e["stance"] == "PRO" else "반대"
            lines.append(f"- {_speaker_display(e)} ({stance_kr}): {e['content'][:max_len]}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


# ── 에이전트별 고유 관점 (에코 방지) ───────────────────────────────────────
_AGENT_PERSPECTIVES = [
    "실현 가능성 관점: 이 해결책이 현실에서 실행 가능한지, 비용과 시간은 어떤지 따져라",
    "피해자/수혜자 관점: 이 문제로 누가 가장 피해를 보고, 해결책은 누구에게 이득인지 따져라",
    "장기적 영향 관점: 이 해결책이 5~10년 후에도 유효한지, 부작용은 없는지 따져라",
    "국제 비교 관점: 다른 나라에서 비슷한 문제를 어떻게 해결했는지 사례를 들어라",
    "구조적 원인 관점: 표면적 증상이 아닌 근본 원인이 무엇인지 짚어라",
]


# ── 초기 의견 제시 프롬프트 (회의 오프너) ──────────────────────────────────

def _build_proposal_prompt(
    topic: str,
    original_stance: str,
    debate_summary: str,
    perspective: str = "",
) -> str:
    """회의 첫 발언: 최적해에 대한 의견 제시. 에이전트별 고유 관점 할당."""
    stance_kr = "찬성" if original_stance == "PRO" else "반대"

    perspective_block = f"\n[너의 고유 관점 — 반드시 이 관점에서만 발언하라]\n{perspective}\n" if perspective else ""

    return f"""[5단계: 최적해 회의]
'{topic}'에 대한 토론이 끝났다. 이제 모두가 입장을 내려놓고 최적해를 함께 찾아야 한다.

너는 원래 {stance_kr} 입장이었지만, 지금은 입장을 버려라.
{perspective_block}
[토론 요약]
{debate_summary}

[지시]
- 너의 고유 관점에서만 의견을 말하라. 다른 관점은 다른 사람이 말한다
- 1~2문장으로 짧게. 길게 쓰지 마라
- 다른 사람과 같은 말을 하지 마라

[절대 금지]
- 추상적 표현 ("균형 필요", "조화를 이루어야")
- 정책 나열 ("첫째, 둘째, 셋째")
- 소제목/번호/목록 사용
- 다른 에이전트의 발언을 복사하거나 비슷하게 반복

[형식]
- 한국어. 합니다체(격식체)
- 1~2문장

반드시 아래 형식으로만 출력:

### 반박 시작
### 반박 끝"""


# ── 멀티턴 메시지 체인 구축 (종합 회의) ───────────────────────────────────

def _build_synthesis_chain(
    agent: Dict,
    history: List,
    speaker_id: str,
) -> List:
    """종합 회의 히스토리에서 멀티턴 메시지 체인을 구축한다.

    해당 에이전트 발언 → AIMessage, 그 외(사용자+다른 에이전트) → HumanMessage.
    """
    system = (
        f"{agent['system_prompt']}\n\n"
        f"[최우선 규칙] 지금은 최적해 회의 중이다. "
        f"찬성/반대 입장을 완전히 버려라. 이전 단계에서 주장한 내용을 반복하지 마라. "
        f"중립적 관점에서 최선의 해결책을 함께 찾아라. "
        f"1~2문장으로만 답변하라. 반드시 한국어로만 답변하라."
    )
    messages = [SystemMessage(content=system)]

    syn_entries = [e for e in history if e["phase"] == "synthesis"]
    for entry in syn_entries:
        if entry["speaker_id"] == speaker_id:
            messages.append(AIMessage(content=entry["content"]))
        else:
            label = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]
            messages.append(HumanMessage(content=f"[{label}] {entry['content']}"))

    return messages


_FALLBACK_MARKER = "이 문제의 핵심을 다시 짚어볼 필요가 있습니다"
_FALLBACK_RESPONSES = [
    "이 해결책의 실현 가능성을 따져보면, 비용과 시간 측면에서 단계적 접근이 필요합니다.",
    "이 문제로 가장 피해를 보는 계층을 우선 고려한 방안이 되어야 합니다.",
    "장기적 관점에서 이 방안이 5년 후에도 유효한지 검토가 필요합니다.",
    "유사한 문제를 해결한 다른 국가의 사례를 참고하면 도움이 될 것입니다.",
    "표면적 증상이 아닌 구조적 원인에 집중한 해결책이 필요합니다.",
]
_fallback_idx = 0


def _generate_with_synthesis_chain(
    messages: List,
    prompt: str,
) -> Tuple[str, str]:
    """멀티턴 체인에 새 프롬프트를 추가하고 생성한다. 빈 응답 시 최대 2회 재시도."""
    messages.append(HumanMessage(content=prompt))

    for attempt in range(3):
        response: AIMessage = _invoke_with_retry(_syn_llm, messages, label=f"synthesis_attempt{attempt}")
        raw = response.content if isinstance(response.content, str) else str(response.content)
        speech = _postprocess_speech(_extract_rebuttal_text(raw))

        # 유효하고 fallback 문장이 아니면 사용
        if _is_valid_rebuttal(speech) and _FALLBACK_MARKER not in speech:
            return speech, raw

        logger.warning("[synthesis] speech 무효 또는 fallback, 재시도 %d/3", attempt + 1)
        if attempt < 2:
            messages.append(HumanMessage(content="이전 응답이 부적절합니다. 한국어로 1~2문장, 구체적인 의견을 말하세요."))

    if not _is_valid_rebuttal(speech):
        global _fallback_idx
        speech = _FALLBACK_RESPONSES[_fallback_idx % len(_FALLBACK_RESPONSES)]
        _fallback_idx += 1

    return speech, raw


# ── 메인 노드: 초기 의견 제시 ──────────────────────────────────────────────

def synthesis_node(state: DebateState) -> DebateState:
    """5단계 종합 — 모든 AI 에이전트가 최적해에 대한 초기 의견을 제시한다.

    회의의 시작: 각 에이전트가 2~3문장으로 의견을 던진다.
    이후 사용자와의 회의는 synthesis_discuss_node로 진행.
    """
    _opening_mod._used_doc_ids = set()

    topic = state["topic"]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    agent_map = {a["agent_id"]: a for a in state["agents"]}
    speaking_order = state["speaking_order"]
    stance_nums = build_agent_stance_nums(state["agents"], speaking_order)

    debate_summary = _summarize_debate(history, state["agents"], speaking_order)

    print(f"\n[5단계: 최적해 회의] 초기 의견 제시\n")

    agent_idx = 0
    for speaker_id in speaking_order:
        if speaker_id == "user":
            continue

        agent = agent_map[speaker_id]
        slabel = "찬성" if agent["stance"] == "PRO" else "반대"
        snum = stance_nums.get(speaker_id, 1)
        display = f"{slabel}{snum}"

        # 에이전트별 고유 관점 할당
        perspective = _AGENT_PERSPECTIVES[agent_idx % len(_AGENT_PERSPECTIVES)]
        agent_idx += 1

        print(f"  [{display}] 의견 제시 중... (관점: {perspective[:20]})")

        prompt = _build_proposal_prompt(
            topic=topic,
            original_stance=agent["stance"],
            debate_summary=debate_summary,
            perspective=perspective,
        )
        chain = _build_synthesis_chain(agent, history, speaker_id)
        speech, raw = _generate_with_synthesis_chain(chain, prompt)

        entry = DebateEntry(
            turn=current_turn,
            speaker_id=speaker_id,
            stance=agent["stance"],
            phase="synthesis",
            content=speech,
            target_id=None,
            tool_calls_log=[],
            json_raw=raw,
        )
        history.append(entry)
        current_turn += 1

        print(f"  [{display}] {speech[:80]}...\n")

    print(f"[5단계: 최적해 회의] 초기 의견 완료 → 사용자 발언 대기\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "phase": "synthesis",
        "is_finished": False,
    })


# ── 회의 응답 노드 (사용자 발언 후 AI 반응) ──────────────────────────────

def synthesis_discuss_node(state: DebateState) -> DebateState:
    """사용자의 최적해 의견에 대해 모든 AI 에이전트가 반응한다."""
    _opening_mod._used_doc_ids = set()

    topic = state["topic"]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    agent_map = {a["agent_id"]: a for a in state["agents"]}
    speaking_order = state["speaking_order"]
    stance_nums = build_agent_stance_nums(state["agents"], speaking_order)

    # 사용자 최근 발언
    user_messages = [e for e in history if e["speaker_id"] == "user" and e["phase"] == "synthesis"]
    user_latest = user_messages[-1]["content"] if user_messages else ""

    print(f"\n[5단계: 최적해 회의] AI 응답 생성 중...\n")

    agent_idx = 0
    for speaker_id in speaking_order:
        if speaker_id == "user":
            continue

        agent = agent_map[speaker_id]
        slabel = "찬성" if agent["stance"] == "PRO" else "반대"
        snum = stance_nums.get(speaker_id, 1)
        display = f"{slabel}{snum}"

        perspective = _AGENT_PERSPECTIVES[agent_idx % len(_AGENT_PERSPECTIVES)]
        agent_idx += 1

        print(f"  [{display}] 응답 중...")

        # 멀티턴 체인으로 이전 회의 맥락 유지
        chain = _build_synthesis_chain(agent, history, speaker_id)
        prompt = (
            f"[너의 고유 관점] {perspective}\n\n"
            f"[사용자 발언]\n{user_latest}\n\n"
            f"반드시 너의 고유 관점에서만 응답하라. "
            f"사용자 문장을 그대로 쓰지 마라. "
            f"다른 에이전트가 이미 말한 내용도 반복하지 마라. "
            f"1~2문장.\n\n"
            f"### 반박 시작\n### 반박 끝"
        )
        speech, raw = _generate_with_synthesis_chain(chain, prompt)

        entry = DebateEntry(
            turn=current_turn,
            speaker_id=speaker_id,
            stance=agent["stance"],
            phase="synthesis",
            content=speech,
            target_id="user",
            tool_calls_log=[],
            json_raw=raw,
        )
        history.append(entry)
        current_turn += 1

        print(f"  [{display}] {speech[:80]}...\n")

    print(f"[5단계: 최적해 회의] AI 응답 완료 → 사용자 발언 대기\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "phase": "synthesis",
        "is_finished": False,
    })


# ── 턴 라우터 ──────────────────────────────────────────────────────────────

def should_end_synthesis(state: DebateState) -> bool:
    """종합 회의 종료 조건: 사용자 2턴 완료."""
    return state.get("synthesis_user_turns", 0) >= 2
