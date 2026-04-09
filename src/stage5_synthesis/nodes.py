"""
nodes.py — 5단계: 종합 및 재개념화(Synthesis & Reconceptualization) 노드

[동작 흐름]
    1. 전체 토론 히스토리(입론~역할반전)를 요약
    2. 모든 AI 에이전트가 교차 순서로 각자의 최적해를 도출 (내부 기록)
    3. AI 최적해들을 종합하여 합의 요약(공통 동의점 + 핵심 갈등 + 해결 방향) 생성
    4. 사용자에게 합의 요약만 제시 → 사용자가 최종 결정자로 "우리의 최적해" 작성

[설계 노트]
    - AI 개별 최적해는 사용자에게 직접 보여주지 않음
    - 합의 요약을 통해 사용자가 최종 의사결정자 역할
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
지금까지의 토론을 종합하여, '{topic}'에 대한 최적해를 도출하라.

너는 원래 {stance_kr} 입장이었지만, 이제는 입장을 버리고
양측 주장의 충돌 원인을 짚은 뒤 그것을 해소하는 해결책을 제시하라.

[토론 요약]
{debate_summary}

[절대 금지]
- "재교육 강화", "지원 확대", "균형 필요", "조화를 이루어야" 같은 추상적 표현
- 어떤 주제에든 붙여넣을 수 있는 범용 해결책
- "첫째 ~, 둘째 ~, 셋째 ~" 식의 정책 나열

[지시]
1. 찬성·반대 타당한 점을 각 1줄로 인정 (짧게)
2. 문제 요약: 이 논쟁에서 양측이 충돌하는 핵심 원인을 1~2줄로 짚어라
3. 최적해: 그 충돌을 해소하는 구체적 방안을 2~3줄로 제시하라
   - 누가 무엇을 하는지 명확히
   - 왜 이것이 충돌을 해소하는지 한 문장으로
   - '{topic}'에만 적용되는 해결책일 것
4. 핵심에 **강조** 사용

[형식]
- 한국어. 고유명사만 영어 허용
- 합니다체(격식체)
- 전체 10줄 이내로 간결하게

반드시 아래 형식으로만 출력:

### 답변 시작
### 찬성측 타당한 점
(1줄)
### 반대측 타당한 점
(1줄)
### 문제 요약
(1~2줄)
### 최적해
(2~3줄)
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


# ── 합의 요약 프롬프트 ────────────────────────────────────────────────────

def _build_consensus_prompt(
    topic: str,
    agent_solutions: List[str],
) -> str:
    """AI 에이전트들의 개별 최적해를 종합하여 합의 요약을 생성하는 프롬프트."""
    solutions_block = "\n---\n".join(
        f"[에이전트 {i+1}]\n{s}" for i, s in enumerate(agent_solutions)
    )

    return f"""다음은 '{topic}'에 대해 여러 토론자가 제시한 최적해입니다.

{solutions_block}

위 최적해들을 분석하여 다음 3가지를 각각 1~2줄로 추출하라:

1. 공통 동의점: 모든 토론자가 동의하는 사실 또는 전제
2. 핵심 갈등: 토론자들 간에 여전히 남아있는 핵심 쟁점
3. 공통 해결 방향: 토론자들의 해결책에서 공통으로 나타나는 방향성

[절대 금지]
- 새로운 해결책을 제시하지 마라. 위 내용을 요약만 하라
- 추상적 표현 ("균형 필요", "조화를 이루어야") 금지

[형식]
- 한국어. 합니다체(격식체)
- 간결하게. 각 항목 1~2줄

반드시 아래 형식으로만 출력:

### 답변 시작
### 공통 동의점
(1~2줄)
### 핵심 갈등
(1~2줄)
### 공통 해결 방향
(1~2줄)
### 답변 끝"""


def _generate_consensus(prompt: str) -> Tuple[str, str]:
    """합의 요약 생성. 시스템 프롬프트 없이 단일 호출."""
    messages = [HumanMessage(content=prompt)]

    response: AIMessage = _invoke_with_retry(_syn_llm, messages, label="consensus")
    raw = response.content if isinstance(response.content, str) else str(response.content)
    speech = _postprocess_speech(_extract_delimited_text(raw))

    if not _is_valid_speech(speech):
        logger.warning("[consensus] speech 무효, 재시도")
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content="한국어로만 합의 요약을 작성하세요.\n\n### 답변 시작\n(요약)\n### 답변 끝"))
        retry: AIMessage = _invoke_with_retry(_syn_llm, messages, label="consensus_retry")
        raw = retry.content if isinstance(retry.content, str) else str(retry.content)
        speech = _postprocess_speech(_extract_delimited_text(raw))

    return speech, raw


# ── 메인 노드 ──────────────────────────────────────────────────────────────

def synthesis_node(state: DebateState) -> DebateState:
    """5단계 종합 및 재개념화 노드.

    1. 모든 AI 에이전트가 각자 최적해를 도출 (내부 기록, 사용자에게 직접 노출 안 함)
    2. AI 최적해들을 종합하여 합의 요약 생성 (synthesis_draft에 저장)
    3. 사용자에게 합의 요약만 제시 → 사용자가 "우리의 최적해" 작성
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

    print(f"\n[5단계: 종합 및 재개념화] AI 에이전트 최적해 도출 중...\n")

    agent_solutions: List[str] = []

    for speaker_id in speaking_order:
        if speaker_id == "user":
            continue

        agent = agent_map[speaker_id]
        slabel = "찬성" if agent["stance"] == "PRO" else "반대"
        snum = stance_nums.get(speaker_id, 1)
        display = f"{slabel} 에이전트{snum}"

        print(f"  [{display}] 종합 발언 생성 중...")

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
                f"### 문제 요약\n"
                f"양측의 충돌 원인을 구체적으로 파악해야 합니다.\n\n"
                f"### 최적해\n"
                f"충돌의 원인을 해소하는 구체적 방안이 필요합니다."
            )

        # 내부 기록 (사용자에게 직접 노출하지 않음)
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
        agent_solutions.append(final_text)
        current_turn += 1

        print(f"  [{display}] 종합 발언 완료 (turn={entry['turn']})\n")

    # ── 합의 요약 생성
    print(f"  [합의 요약] 생성 중...")
    consensus_prompt = _build_consensus_prompt(topic, agent_solutions)
    consensus_text, consensus_raw = _generate_consensus(consensus_prompt)

    if not _is_valid_speech(consensus_text):
        consensus_text = (
            "### 공통 동의점\n이 주제에 대해 양측 모두 문제의 존재를 인정합니다.\n\n"
            "### 핵심 갈등\n해결 방식과 우선순위에서 의견이 갈립니다.\n\n"
            "### 공통 해결 방향\n구체적인 메커니즘을 통한 문제 해소가 필요하다는 점에서 방향은 일치합니다."
        )

    print(f"  [합의 요약] 완료\n")
    print(f"  {consensus_text[:100]}...\n")
    print(f"[5단계: 종합 및 재개념화] 합의 요약 완료 → 사용자 최종 결정 대기\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "phase": "synthesis",
        "synthesis_draft": consensus_text,
        "is_finished": False,
    })
