"""
nodes.py — 3단계: 자유 논박(Free Rebuttal) 노드

[동작 흐름]
    1. speaking_order(입론과 동일한 교차 순서)를 사이클 단위로 반복
    2. 각 사이클마다 모든 발언자가 1회씩 발언
    3. AI 에이전트는 debate_history에서 반박 대상을 자동 선정
       (직전 발언한 상대 진영 에이전트를 타겟으로 지정)
    4. 사용자(user) 차례는 건너뛰고 API에서 처리
    5. current_cycle >= max_cycle 이면 phase를 "role_reversal"로 전환

[설계 노트]
    - 입론 단계의 도구(search_web, search_vector_db)와 LLM 인스턴스를 재사용한다.
    - 연쇄 논박과 동일한 _run_rebuttal_loop를 재사용한다.
    - 발화 시 반박 대상(@에이전트명)을 명시한다.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage

import src.stage1_opening.nodes as _opening_mod
from src.stage1_opening.nodes import _postprocess_speech
from src.stage2_rebuttal.nodes import (
    _build_rebuttal_prompt,
    _run_rebuttal_loop,
    build_agent_stance_nums,
)
from src.state import DebateEntry, DebateState


# ── 타겟 자동 선정 ───────────────────────────────────────────────────────────

def _pick_target(
    history: List[DebateEntry],
    speaker_id: str,
    speaker_stance: str,
    agents: List[Dict],
) -> Optional[str]:
    """debate_history에서 반박 대상을 자동 선정한다.

    규칙: 가장 최근에 발언한 상대 진영 발언자를 타겟으로 선택한다.
    상대 진영 발언이 없으면 None을 반환한다.
    """
    opposite = "CON" if speaker_stance == "PRO" else "PRO"
    for entry in reversed(history):
        if entry["stance"] == opposite and entry["speaker_id"] != speaker_id:
            return entry["speaker_id"]
    return None


# ── AI 자유논박 발언 생성 ─────────────────────────────────────────────────────

def generate_ai_free_rebuttal(
    topic: str,
    history: List[DebateEntry],
    agent: Dict,
    target_id: str,
    stance_num: int,
    target_stance_num: int,
    current_turn: int,
) -> DebateEntry:
    """단일 AI 에이전트의 자유논박 발언을 생성한다.

    Args:
        topic:             토론 주제
        history:           현재까지의 debate_history
        agent:             발언할 AI 에이전트 스냅샷
        target_id:         반박 대상 speaker_id
        stance_num:        발언자의 진영 내 번호
        target_stance_num: 타겟의 진영 내 번호
        current_turn:      할당할 turn 번호

    Returns:
        생성된 DebateEntry
    """
    # 타겟의 가장 최근 발언 찾기
    target_speech = "(발언 기록 없음)"
    target_stance = "CON" if agent["stance"] == "PRO" else "PRO"
    for entry in reversed(history):
        if entry["speaker_id"] == target_id:
            target_speech = entry["content"]
            target_stance = entry["stance"]
            break

    messages = [
        SystemMessage(content=agent["system_prompt"]),
        HumanMessage(content=_build_rebuttal_prompt(
            topic, agent["stance"], target_id, target_speech,
            target_stance, stance_num, target_stance_num, is_response=False,
        )),
    ]

    final_text, json_raw, tool_calls_log = _run_rebuttal_loop(messages, target_id)

    return DebateEntry(
        turn=current_turn,
        speaker_id=agent["agent_id"],
        stance=agent["stance"],
        phase="free_rebuttal",
        content=final_text,
        target_id=target_id,
        tool_calls_log=tool_calls_log,
        json_raw=json_raw,
    )


# ── 메인 노드 ─────────────────────────────────────────────────────────────────

def free_rebuttal_node(state: DebateState) -> DebateState:
    """3단계 자유 논박 노드.

    speaking_order를 사이클 단위로 반복하며 AI 에이전트가 자유롭게 반박한다.
    사용자(user) 차례는 건너뛰고 API를 통해 별도 처리한다.
    current_cycle >= max_cycle이면 phase를 "role_reversal"로 전환한다.

    Args:
        state: 현재 DebateState (phase == "free_rebuttal" 전제)

    Returns:
        debate_history가 누적되고, 사이클이 진행된 DebateState
    """
    # 문서 중복 추적 초기화
    _opening_mod._used_doc_ids = set()

    topic: str = state["topic"]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    current_cycle: int = state["current_cycle"]
    max_cycle: int = state["max_cycle"]
    agent_map = {a["agent_id"]: a for a in state["agents"]}
    speaking_order = state["speaking_order"]

    # 에이전트별 진영 번호 매핑
    stance_nums = build_agent_stance_nums(state["agents"], speaking_order)

    print(f"\n[3단계: 자유 논박] 사이클 {current_cycle + 1}/{max_cycle}\n")

    # 한 사이클: speaking_order 전체를 1회 순회
    user_encountered = False
    for speaker_id in speaking_order:
        if speaker_id == "user":
            user_encountered = True
            print(f"  [{speaker_id}] 사용자 차례 → API 대기\n")
            continue

        agent = agent_map[speaker_id]
        stance_label = "찬성" if agent["stance"] == "PRO" else "반대"
        display_name = f"{stance_label} 에이전트{stance_nums[speaker_id]}"

        # 반박 대상 자동 선정
        target_id = _pick_target(history, speaker_id, agent["stance"], state["agents"])
        if target_id is None:
            logger.warning("[free_rebuttal] %s의 반박 대상을 찾을 수 없음, 건너뜀", speaker_id)
            continue

        target_snum = stance_nums.get(target_id, 0)
        target_display = "사용자" if target_id == "user" else (
            f"{'찬성' if agent_map[target_id]['stance'] == 'PRO' else '반대'} "
            f"에이전트{target_snum}"
        ) if target_id in agent_map else target_id

        print(f"  [{display_name}] → {target_display} 반박 중...")

        entry = generate_ai_free_rebuttal(
            topic=topic,
            history=history,
            agent=agent,
            target_id=target_id,
            stance_num=stance_nums[speaker_id],
            target_stance_num=target_snum,
            current_turn=current_turn,
        )
        history.append(entry)
        current_turn += 1

        print(f"  [{display_name}] 반박 완료 (turn={entry['turn']})\n")

    current_cycle += 1

    # 종료 판단
    if user_encountered:
        # 사용자 차례가 있으면 사용자 입력 대기 (사이클 미완료)
        next_phase = "free_rebuttal"
        print(f"[3단계: 자유 논박] AI 발언 완료 → 사용자 발언 대기 (사이클 {current_cycle}/{max_cycle})\n")
    elif current_cycle >= max_cycle:
        next_phase = "role_reversal"
        print(f"[3단계: 자유 논박] 최대 사이클 도달 → 4단계 역할 반전(role_reversal)으로 전환\n")
    else:
        next_phase = "free_rebuttal"
        print(f"[3단계: 자유 논박] 사이클 {current_cycle}/{max_cycle} 완료\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "current_cycle": current_cycle,
        "phase": next_phase,
    })
