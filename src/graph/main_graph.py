"""
main_graph.py — LangGraph 메인 토론 그래프

전체 토론 흐름(1~5단계)을 하나의 StateGraph로 구현.
사용자 입력은 interrupt로 대기, FastAPI에서 Command(resume=)로 재개.

[그래프 흐름]
    ai_opening → user_opening → ai_rebuttal → user_rebuttal
    → ai_free_rebuttal ↔ user_free_rebuttal (루프)
    → ai_free_final
    → ai_role_reversal → user_role_reversal
    → ai_synthesis ↔ user_synthesis (루프)
    → user_finalize → END
"""

from __future__ import annotations

import logging
from typing import Dict, List

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from src.state import DebateEntry, DebateState
from src.phase1.stage1_opening.nodes import opening_arguments_node
from src.phase1.stage2_rebuttal.nodes import chained_rebuttal_node, build_agent_stance_nums
from src.phase1.stage3_free_rebuttal.nodes import free_rebuttal_node
from src.phase1.stage4_role_reversal.nodes import role_reversal_node
from src.phase1.stage5_synthesis.nodes import synthesis_node, synthesis_discuss_node

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 노드 함수
# ═══════════════════════════════════════════════════════════════════════════

# ── 1단계: 입론 ────────────────────────────────────────────────────────────

def ai_opening_node(state: DebateState) -> dict:
    """AI 에이전트 입론 생성. speaking_order에서 사용자 제외하고 생성."""
    updated = opening_arguments_node(state)
    return {
        "debate_history": updated["debate_history"],
        "current_turn": updated["current_turn"],
        "current_speaker_index": updated["current_speaker_index"],
        "phase": "opening",
    }


def user_opening_node(state: DebateState) -> dict:
    """사용자 입론 — interrupt로 대기."""
    user_content = interrupt("사용자 입론을 입력하세요")

    history = list(state["debate_history"])
    user_turn = next(
        (i for i, sid in enumerate(state["speaking_order"]) if sid == "user"), 0
    )
    history.append(DebateEntry(
        turn=user_turn,
        speaker_id="user",
        stance=state["user_stance"],
        phase="opening",
        content=user_content,
        target_id=None,
    ))
    history.sort(key=lambda e: e["turn"])

    return {
        "debate_history": history,
        "phase": "chained_rebuttal",
    }


# ── 2단계: 연쇄논박 ───────────────────────────────────────────────────────

def ai_rebuttal_node(state: DebateState) -> dict:
    """AI 연쇄논박 생성."""
    updated = chained_rebuttal_node(state)
    return {
        "debate_history": updated["debate_history"],
        "current_turn": updated["current_turn"],
        "rebuttal_pairs": updated["rebuttal_pairs"],
        "phase": updated["phase"],
    }


def user_rebuttal_node(state: DebateState) -> dict:
    """사용자 연쇄논박 — interrupt로 대기."""
    # 사용자를 공격한 에이전트 찾기
    attacker_id = None
    for e in reversed(state["debate_history"]):
        if e["phase"] == "chained_rebuttal" and e.get("target_id") == "user":
            attacker_id = e["speaker_id"]
            break
    target_id = attacker_id or "agent_1"

    user_content = interrupt(f"{target_id}에 대한 반박을 입력하세요")

    history = list(state["debate_history"])
    history.append(DebateEntry(
        turn=state["current_turn"],
        speaker_id="user",
        stance=state["user_stance"],
        phase="chained_rebuttal",
        content=user_content,
        target_id=target_id,
    ))

    return {
        "debate_history": history,
        "current_turn": state["current_turn"] + 1,
        "phase": "free_rebuttal",
    }


# ── 3단계: 자유논박 ───────────────────────────────────────────────────────

def ai_free_rebuttal_node(state: DebateState) -> dict:
    """AI 자유논박 공격/답변+공격."""
    # 상대 에이전트 자동 선택 (아직 선택 안 됐으면)
    if not state.get("selected_opponent_id"):
        opposite = "CON" if state["user_stance"] == "PRO" else "PRO"
        opponent = next((a for a in state["agents"] if a["stance"] == opposite), None)
        if opponent:
            state = DebateState(**{**state, "selected_opponent_id": opponent["agent_id"]})

    updated = free_rebuttal_node(state)
    return {
        "debate_history": updated["debate_history"],
        "current_turn": updated["current_turn"],
        "selected_opponent_id": updated.get("selected_opponent_id", state.get("selected_opponent_id")),
        "phase": "free_rebuttal",
    }


def user_free_rebuttal_node(state: DebateState) -> dict:
    """사용자 자유논박 답변+공격 — interrupt 2회 (답변, 공격)."""
    selected_id = state.get("selected_opponent_id", "agent_1")

    user_defense = interrupt("상대 공격에 대한 답변을 입력하세요")
    user_attack = interrupt("상대 논거를 공격하세요")

    history = list(state["debate_history"])
    current_turn = state["current_turn"]

    # 답변 기록
    history.append(DebateEntry(
        turn=current_turn,
        speaker_id="user",
        stance=state["user_stance"],
        phase="free_rebuttal",
        content=user_defense,
        target_id=selected_id,
    ))
    current_turn += 1

    # 공격 기록
    history.append(DebateEntry(
        turn=current_turn,
        speaker_id="user",
        stance=state["user_stance"],
        phase="free_rebuttal",
        content=user_attack,
        target_id=selected_id,
    ))
    current_turn += 1

    new_user_turns = state.get("free_rebuttal_user_turns", 0) + 1

    return {
        "debate_history": history,
        "current_turn": current_turn,
        "free_rebuttal_user_turns": new_user_turns,
    }




# ── 4단계: 역할반전 ───────────────────────────────────────────────────────

def ai_role_reversal_node(state: DebateState) -> dict:
    """AI 역할반전 발언 생성."""
    updated = role_reversal_node(state)
    return {
        "debate_history": updated["debate_history"],
        "current_turn": updated["current_turn"],
        "role_reversed": updated["role_reversed"],
        "phase": "role_reversal",
    }


def user_role_reversal_node(state: DebateState) -> dict:
    """사용자 역할반전 — interrupt로 대기."""
    reversed_stance = "CON" if state["user_stance"] == "PRO" else "PRO"
    user_content = interrupt("상대 입장을 옹호하는 발언을 입력하세요")

    history = list(state["debate_history"])
    history.append(DebateEntry(
        turn=state["current_turn"],
        speaker_id="user",
        stance=reversed_stance,
        phase="role_reversal",
        content=user_content,
        target_id=None,
    ))

    return {
        "debate_history": history,
        "current_turn": state["current_turn"] + 1,
        "phase": "synthesis",
    }


# ── 5단계: 종합 ───────────────────────────────────────────────────────────

def ai_synthesis_node(state: DebateState) -> dict:
    """AI 종합 의견 제시 / 응답."""
    syn_entries = [e for e in state["debate_history"] if e["phase"] == "synthesis"]

    if not syn_entries:
        # 초기 의견
        updated = synthesis_node(state)
    else:
        # 사용자 발언에 대한 응답
        updated = synthesis_discuss_node(state)

    return {
        "debate_history": updated["debate_history"],
        "current_turn": updated["current_turn"],
        "phase": "synthesis",
    }


def user_synthesis_node(state: DebateState) -> dict:
    """사용자 종합 의견 — interrupt로 대기."""
    user_content = interrupt("최적해에 대한 의견을 입력하세요")

    history = list(state["debate_history"])
    history.append(DebateEntry(
        turn=state["current_turn"],
        speaker_id="user",
        stance=state["user_stance"],
        phase="synthesis",
        content=user_content,
        target_id=None,
    ))

    new_user_turns = state.get("synthesis_user_turns", 0) + 1

    return {
        "debate_history": history,
        "current_turn": state["current_turn"] + 1,
        "synthesis_user_turns": new_user_turns,
    }


def user_finalize_node(state: DebateState) -> dict:
    """사용자 최적해 확정 — interrupt로 대기."""
    user_content = interrupt("우리의 최적해를 작성하세요")

    history = list(state["debate_history"])
    history.append(DebateEntry(
        turn=state["current_turn"],
        speaker_id="user",
        stance=state["user_stance"],
        phase="synthesis",
        content=user_content,
        target_id=None,
    ))

    return {
        "debate_history": history,
        "current_turn": state["current_turn"] + 1,
        "synthesis_draft": user_content,
        "is_finished": True,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 라우터 (조건부 엣지)
# ═══════════════════════════════════════════════════════════════════════════

def route_free_rebuttal(state: DebateState) -> str:
    """자유논박 루프 라우터: 사용자 2턴 완료 → done, 아니면 continue."""
    if state.get("free_rebuttal_user_turns", 0) >= 2:
        return "done"
    return "continue"


def route_after_ai_free(state: DebateState) -> str:
    """AI 자유논박 후: 사용자 2턴 완료 상태면 역할반전으로, 아니면 사용자 턴."""
    if state.get("free_rebuttal_user_turns", 0) >= 2:
        return "end_free"
    return "to_user"


def route_synthesis(state: DebateState) -> str:
    """종합 회의 루프 라우터: 사용자 2턴 완료 → finalize, 아니면 continue."""
    if state.get("synthesis_user_turns", 0) >= 2:
        return "finalize"
    return "continue"


# ═══════════════════════════════════════════════════════════════════════════
# 그래프 빌더
# ═══════════════════════════════════════════════════════════════════════════

def build_debate_graph():
    """전체 토론 메인 그래프를 빌드하고 컴파일한다."""
    graph = StateGraph(DebateState)

    # 노드 등록
    graph.add_node("ai_opening", ai_opening_node)
    graph.add_node("user_opening", user_opening_node)
    graph.add_node("ai_rebuttal", ai_rebuttal_node)
    graph.add_node("user_rebuttal", user_rebuttal_node)
    graph.add_node("ai_free_rebuttal", ai_free_rebuttal_node)
    graph.add_node("user_free_rebuttal", user_free_rebuttal_node)
    graph.add_node("ai_role_reversal", ai_role_reversal_node)
    graph.add_node("user_role_reversal", user_role_reversal_node)
    graph.add_node("ai_synthesis", ai_synthesis_node)
    graph.add_node("user_synthesis", user_synthesis_node)
    graph.add_node("user_finalize", user_finalize_node)

    # 엣지: 순차 흐름
    graph.set_entry_point("ai_opening")
    graph.add_edge("ai_opening", "user_opening")
    graph.add_edge("user_opening", "ai_rebuttal")
    graph.add_edge("ai_rebuttal", "user_rebuttal")
    graph.add_edge("user_rebuttal", "ai_free_rebuttal")

    # 자유논박: AI 발언 후 → 사용자 턴 or 역할반전
    graph.add_conditional_edges("ai_free_rebuttal", route_after_ai_free, {
        "to_user": "user_free_rebuttal",
        "end_free": "ai_role_reversal",
    })

    # 사용자 자유논박 후 → AI 자유논박 (루프)
    graph.add_conditional_edges("user_free_rebuttal", route_free_rebuttal, {
        "continue": "ai_free_rebuttal",
        "done": "ai_free_rebuttal",  # 마지막 답변 생성 후 route_after_ai_free에서 역할반전으로
    })

    # 역할반전 → 종합
    graph.add_edge("ai_role_reversal", "user_role_reversal")
    graph.add_edge("user_role_reversal", "ai_synthesis")
    graph.add_edge("ai_synthesis", "user_synthesis")

    # 종합 루프
    graph.add_conditional_edges("user_synthesis", route_synthesis, {
        "continue": "ai_synthesis",
        "finalize": "user_finalize",
    })
    graph.add_edge("user_finalize", END)

    # 체크포인터 (세션 상태 저장)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
