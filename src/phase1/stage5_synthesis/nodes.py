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

import src.phase1.stage1_opening.nodes as _opening_mod
from src.phase1.stage1_opening.nodes import (
    _invoke_with_retry,
    _postprocess_speech,
    _extract_delimited_text,
    _is_valid_speech,
    _LLM_KWARGS,
)
from src.phase1.stage2_rebuttal.nodes import (
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

# ── 강경도별 타협 태도 (합의 속도 조절) ──────────────────────────────────────
_INTENSITY_NEGOTIATION = {
    1: "상대 의견에 열린 태도를 보이되, 자기 핵심 조건 1개는 반드시 지키면서 구체적 절충안을 제시하라",
    2: "양쪽 의견을 수용하되, 빠진 조건이나 실행 가능성에 대해 구체적으로 질문하라",
    3: "양쪽 장단점을 비교하며, 아직 논의되지 않은 새로운 쟁점을 제기하라",
    4: "자기 원래 입장의 핵심 조건이 빠지면 최적해가 실패한다고 경고하라. 절대 쉽게 동의하지 마라",
    5: "자기 원래 입장을 강하게 고수하며, 상대 제안의 약점을 구체적으로 지적하라. 동의하지 마라",
}


# ── 초기 의견 제시 프롬프트 (회의 오프너) ──────────────────────────────────

def _build_proposal_prompt(
    topic: str,
    original_stance: str,
    perspective: str = "",
    intensity: int = 3,
) -> str:
    """회의 첫 발언: 최적해에 대한 의견 제시. 에이전트별 고유 관점 + 강경도 반영.
    토론 히스토리는 메시지 체인으로 이미 포함되어 있으므로 요약 불필요.
    """
    stance_kr = "찬성" if original_stance == "PRO" else "반대"

    perspective_block = f"\n[너의 고유 관점 — 반드시 이 관점에서만 발언하라]\n{perspective}\n" if perspective else ""
    negotiation = _INTENSITY_NEGOTIATION.get(intensity, _INTENSITY_NEGOTIATION[3])

    return f"""[5단계: 최적해 회의 — 단톡방]
'{topic}'에 대한 토론이 끝났다. 이제 단톡방에서 최적해를 함께 찾는 대화를 하고 있다.

너는 원래 {stance_kr} 입장이었다.
{perspective_block}
[너의 협상 태도]
{negotiation}

[지시]
- 위 대화 내용에 이어서 자연스럽게 대화하라
- "동의합니다", "좋은 의견입니다", "맞습니다"로 시작하지 마라
- 앞 사람이 말한 내용을 그대로 반복하지 마라. 반드시 다른 각도의 의견을 제시하라
- 너의 고유 관점에서 아직 언급되지 않은 쟁점이나 조건을 제기하라
- 1~2문장으로 짧게. 대화체로
- 핵심 주장에 **강조** 표시
- 소제목/번호/목록 금지

[형식]
- 한국어. 합니다체
- 대화하듯이 자연스럽게

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
        f"[최우선 규칙] 최적해를 찾는 단톡방 대화 중이다. "
        f"자기 관점을 유지하면서 앞 사람 발언에 자연스럽게 반응하라. "
        f"이전 발언자와 같은 결론을 반복하지 마라. "
        f"1~2문장. 한국어. 합니다체."
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
    agent: Dict,
    prompt: str,
    chain: List = None,
    topic: str = "",
) -> Tuple[str, str, List]:
    """멀티턴 체인에 새 프롬프트를 추가하고 생성한다. 빈 응답 시 최대 2회 재시도."""
    messages = list(chain) if chain else []
    messages.append(HumanMessage(content=prompt))

    for attempt in range(3):
        response: AIMessage = _invoke_with_retry(_syn_llm, messages, label=f"synthesis_attempt{attempt}")
        raw = response.content if isinstance(response.content, str) else str(response.content)
        speech = _postprocess_speech(_extract_rebuttal_text(raw))

        # 유효하고 fallback 문장이 아니면 사용
        if _is_valid_rebuttal(speech) and _FALLBACK_MARKER not in speech:
            return speech, raw, []

        logger.warning("[synthesis] speech 무효 또는 fallback, 재시도 %d/3", attempt + 1)
        if attempt < 2:
            messages.append(HumanMessage(content="이전 응답이 부적절합니다. 한국어로 1~2문장, 구체적인 의견을 말하세요."))

    if not _is_valid_rebuttal(speech):
        global _fallback_idx
        speech = _FALLBACK_RESPONSES[_fallback_idx % len(_FALLBACK_RESPONSES)]
        _fallback_idx += 1

    return speech, raw, []


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

    print(f"\n[5단계: 최적해 회의] 초기 의견 제시\n")

    agent_idx = 0
    this_round_speeches: List[str] = []  # 이번 라운드에서 다른 에이전트가 한 말 수집
    for speaker_id in speaking_order:
        if speaker_id == "user":
            continue

        agent = agent_map[speaker_id]
        slabel = "찬성" if agent["stance"] == "PRO" else "반대"
        snum = stance_nums.get(speaker_id, 1)
        display = f"{slabel}{snum}"

        perspective = _AGENT_PERSPECTIVES[agent_idx % len(_AGENT_PERSPECTIVES)]
        agent_idx += 1
        intensity = agent.get("intensity", 3)

        print(f"  [{display}] 의견 제시 중... (관점: {perspective[:20]}, 강경도: {intensity})")

        prompt = _build_proposal_prompt(
            topic=topic,
            original_stance=agent["stance"],
            perspective=perspective,
            intensity=intensity,
        )
        # 이전 에이전트가 이미 한 말 추가
        if this_round_speeches:
            already_said = (
                "\n[다른 참여자가 이미 한 말 — 같은 내용 반복 금지]\n"
                + "\n".join(f"- {s}" for s in this_round_speeches)
                + "\n위와 완전히 다른 관점에서 발언하라.\n"
            )
            prompt += f"\n{already_said}"

        # 전체 토론 히스토리를 메시지 체인으로 전달 (_summarize_debate 대체)
        from src.graph.llm import build_debate_chain
        debate_chain = build_debate_chain(history, speaker_id)
        chain = _build_synthesis_chain(agent, history, speaker_id)
        # debate_chain(1~4단계) + synthesis_chain(5단계) 합치기
        full_chain = debate_chain + [m for m in chain if m not in debate_chain]
        speech, raw, _logs = _generate_with_synthesis_chain(agent, prompt, full_chain, topic)
        this_round_speeches.append(speech[:80])  # 다음 에이전트가 참고

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
    this_round_speeches: List[str] = []  # 이번 라운드에서 다른 에이전트가 한 말 수집
    for speaker_id in speaking_order:
        if speaker_id == "user":
            continue

        agent = agent_map[speaker_id]
        slabel = "찬성" if agent["stance"] == "PRO" else "반대"
        snum = stance_nums.get(speaker_id, 1)
        display = f"{slabel}{snum}"

        perspective = _AGENT_PERSPECTIVES[agent_idx % len(_AGENT_PERSPECTIVES)]
        agent_idx += 1
        intensity = agent.get("intensity", 3)
        negotiation = _INTENSITY_NEGOTIATION.get(intensity, _INTENSITY_NEGOTIATION[3])

        print(f"  [{display}] 응답 중... (강경도: {intensity})")

        # 이번 라운드에서 다른 에이전트가 이미 말한 내용을 프롬프트에 포함
        already_said = ""
        if this_round_speeches:
            already_said = (
                "\n[이번 라운드에서 다른 참여자가 이미 한 말 — 같은 내용 반복 금지]\n"
                + "\n".join(f"- {s}" for s in this_round_speeches)
                + "\n"
            )

        # 전체 토론 히스토리 + 종합 회의 체인
        from src.graph.llm import build_debate_chain
        debate_chain = build_debate_chain(history, speaker_id)
        prompt = (
            f"[너의 고유 관점 — 반드시 이 관점에서만 발언하라] {perspective}\n\n"
            f"[너의 협상 태도] {negotiation}\n"
            f"{already_said}\n"
            f"사용자가 방금 '{user_latest[:100]}...'라고 말했다.\n\n"
            f"위에서 이미 언급된 내용과 완전히 다른 관점에서 구체적 조건이나 미해결 쟁점을 제기하라. "
            f"'동의합니다'/'좋은 의견입니다'/'좋은 출발점'으로 시작하지 마라. "
            f"대화하듯이 자연스럽게. 1~2문장.\n\n"
            f"### 반박 시작\n### 반박 끝"
        )
        speech, raw, _logs = _generate_with_synthesis_chain(agent, prompt, debate_chain, topic)
        this_round_speeches.append(speech[:80])  # 다음 에이전트가 참고할 수 있도록 수집

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
