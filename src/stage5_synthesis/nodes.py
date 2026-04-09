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
    """전체 토론 히스토리를 단계별로 요약한다."""
    stance_nums = build_agent_stance_nums(agents, speaking_order)

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
        max_len = 200 if phase_name in ("opening", "role_reversal") else 150
        lines = [f"[{phase_label}]"]
        for e in entries:
            stance_kr = "찬성" if e["stance"] == "PRO" else "반대"
            lines.append(f"- {_speaker_display(e)} ({stance_kr}): {e['content'][:max_len]}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


# ── 초기 의견 제시 프롬프트 (회의 오프너) ──────────────────────────────────

def _build_proposal_prompt(
    topic: str,
    original_stance: str,
    debate_summary: str,
) -> str:
    """회의 첫 발언: 최적해에 대한 의견 제시."""
    stance_kr = "찬성" if original_stance == "PRO" else "반대"

    return f"""[5단계: 최적해 회의]
'{topic}'에 대한 토론이 끝났다. 이제 모두가 입장을 내려놓고 최적해를 함께 찾아야 한다.

너는 원래 {stance_kr} 입장이었지만, 지금은 입장을 버려라.
토론 전체를 돌아보고, 이 문제의 핵심이 무엇이며 어떻게 해결할 수 있는지 의견을 제시하라.

[토론 요약]
{debate_summary}

[지시]
- 회의에서 의견을 말하듯이 자연스럽게 말하라
- 이 문제의 핵심이 뭔지 한마디로 짚고, 자기 생각하는 해결 방향을 제시하라
- 1~2문장으로 짧게. 길게 쓰지 마라
- 다른 사람이 이어서 말할 수 있도록 열린 표현으로 마무리하라

[절대 금지]
- 추상적 표현 ("균형 필요", "조화를 이루어야")
- 정책 나열 ("첫째, 둘째, 셋째")
- 소제목/번호/목록 사용

[형식]
- 한국어. 합니다체(격식체)
- 1~2문장

반드시 아래 형식으로만 출력:

### 반박 시작
(의견)
### 반박 끝"""


# ── 회의 응답 프롬프트 (사용자 발언에 대한 반응) ─────────────────────────

def _build_discuss_prompt(
    topic: str,
    original_stance: str,
    user_message: str,
    previous_discussion: str,
) -> str:
    """회의 중 사용자 발언에 대한 응답."""
    stance_kr = "찬성" if original_stance == "PRO" else "반대"

    return f"""[5단계: 최적해 회의 — 계속]
'{topic}'에 대한 최적해를 함께 찾고 있다.

[이전 회의 내용]
{previous_discussion}

[사용자 발언]
{user_message}

[지시]
- 사용자의 의견에 기본적으로 동조하라. 사용자가 제시한 방향을 기반으로 발전시켜라
- 동의하면서 빠진 부분을 보완하거나, 구체적인 수치/조건/사례를 덧붙여라
- 사용자 의견을 정면 반박하지 마라. 같은 방향에서 더 나은 안을 제안하라
- 1~2문장으로 짧게
- 소제목/번호/목록 금지

[형식]
- 한국어. 합니다체(격식체)
- 1~2문장

반드시 아래 형식으로만 출력:

### 반박 시작
(의견)
### 반박 끝"""


# ── 단발 생성 (자유논박과 동일 패턴) ──────────────────────────────────────

def _generate_single(agent: Dict, prompt: str) -> Tuple[str, str]:
    """회의 발언 단발 생성."""
    system = (
        f"{agent['system_prompt']}\n\n"
        f"[최우선 규칙] 지금은 최적해 회의 중이다. "
        f"입장을 버리고 최선의 해결책을 함께 찾아라. "
        f"1~2문장으로만 답변하라."
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ]

    response: AIMessage = _invoke_with_retry(_syn_llm, messages, label="synthesis")
    raw = response.content if isinstance(response.content, str) else str(response.content)
    speech = _postprocess_speech(_extract_rebuttal_text(raw))

    if not _is_valid_rebuttal(speech):
        logger.warning("[synthesis] speech 무효, 재시도")
        messages.append(HumanMessage(content="한국어로만 2~3문장으로 의견을 말하세요."))
        retry: AIMessage = _invoke_with_retry(_syn_llm, messages, label="synthesis_retry")
        raw = retry.content if isinstance(retry.content, str) else str(retry.content)
        speech = _postprocess_speech(_extract_rebuttal_text(raw))

    # 영어 잔재 감지
    eng_words = re.findall(r'(?<![a-zA-Z])[a-z]{4,}(?![a-zA-Z])', speech)
    if eng_words:
        logger.warning("[synthesis] 영어 감지: %s → 수정 요청", eng_words[:3])
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content=f'다음 영어 단어를 한국어로 바꿔서 다시 작성하라: {", ".join(eng_words[:5])}\n한국어만 사용. 같은 형식 유지.'))
        fix: AIMessage = _invoke_with_retry(_syn_llm, messages, label="synthesis_fix_eng")
        raw_fix = fix.content if isinstance(fix.content, str) else str(fix.content)
        fixed = _postprocess_speech(_extract_rebuttal_text(raw_fix))
        if _is_valid_rebuttal(fixed):
            speech = fixed
            raw = raw_fix

    if not _is_valid_rebuttal(speech):
        speech = "이 문제의 핵심을 다시 짚어볼 필요가 있습니다. 구체적인 해결 방안을 함께 논의해야 합니다."

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

    for speaker_id in speaking_order:
        if speaker_id == "user":
            continue

        agent = agent_map[speaker_id]
        slabel = "찬성" if agent["stance"] == "PRO" else "반대"
        snum = stance_nums.get(speaker_id, 1)
        display = f"{slabel}{snum}"

        print(f"  [{display}] 의견 제시 중...")

        prompt = _build_proposal_prompt(
            topic=topic,
            original_stance=agent["stance"],
            debate_summary=debate_summary,
        )
        speech, raw = _generate_single(agent, prompt)

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

    # 이전 회의 내용 요약
    syn_entries = [e for e in history if e["phase"] == "synthesis"]
    previous = "\n".join(
        f"- {'사용자' if e['speaker_id'] == 'user' else e['speaker_id']}: {e['content'][:150]}"
        for e in syn_entries
    )

    print(f"\n[5단계: 최적해 회의] AI 응답 생성 중...\n")

    for speaker_id in speaking_order:
        if speaker_id == "user":
            continue

        agent = agent_map[speaker_id]
        slabel = "찬성" if agent["stance"] == "PRO" else "반대"
        snum = stance_nums.get(speaker_id, 1)
        display = f"{slabel}{snum}"

        print(f"  [{display}] 응답 중...")

        prompt = _build_discuss_prompt(
            topic=topic,
            original_stance=agent["stance"],
            user_message=user_latest,
            previous_discussion=previous,
        )
        speech, raw = _generate_single(agent, prompt)

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
