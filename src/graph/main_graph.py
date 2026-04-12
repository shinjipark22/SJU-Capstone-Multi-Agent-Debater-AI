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

def _generate_openings_for(state: DebateState, speaker_ids: list) -> dict:
    """지정된 speaker_ids에 대해서만 입론을 생성한다."""
    from src.graph.subgraphs import write_review
    from src.phase1.stage1_opening.nodes import _pre_search, _build_opening_prompt, _is_valid_speech

    import src.phase1.stage1_opening.nodes as _opening_mod
    _opening_mod._used_doc_ids = set()

    topic = state["topic"]
    history = list(state["debate_history"])
    agent_map = {a["agent_id"]: a for a in state["agents"]}
    stance_counter = {"PRO": 0, "CON": 0}

    # 이미 입론한 에이전트의 stance 카운트
    for e in history:
        if e["phase"] == "opening" and e["speaker_id"] != "user":
            stance_counter[e["stance"]] += 1

    current_turn = state["current_turn"]

    for speaker_id in speaker_ids:
        if speaker_id == "user" or speaker_id not in agent_map:
            continue

        agent = agent_map[speaker_id]
        stance_counter[agent["stance"]] += 1
        snum = stance_counter[agent["stance"]]
        slabel = "찬성" if agent["stance"] == "PRO" else "반대"
        display = f"{slabel}{snum}"

        print(f"  [{display}] 입론 생성 중...")

        search_results, tool_calls_log = _pre_search(
            topic, agent["stance"], topic_id=state.get("topic_id", ""),
        )
        prompt = _build_opening_prompt(topic, agent["stance"], display, search_results)

        result = write_review.invoke({
            "topic": topic, "agent": agent,
            "expected_stance": agent["stance"],
            "target_argument": prompt,
            "my_opening": "", "opp_opening": "",
            "chain": [], "prev_weaknesses": "", "prev_attacks": "",
            "mode": "opening",
            "weakness": "", "search_results": "", "search_query": "",
            "speech": "", "raw": "",
            "review_result": {}, "retry_count": 0, "tool_calls_log": [],
        })

        final_text = result["speech"]
        raw = result["raw"]

        if not _is_valid_speech(final_text):
            final_text = (
                f"### 자기소개와 입장 표명\n"
                f"저는 {display}입니다. {topic}에 대해 {slabel} 입장입니다.\n\n"
                f"### 결론\n저는 {slabel} 입장을 유지합니다."
            )

        # 자기소개 소제목 보장
        import re
        if '### 자기소개' not in final_text and '### 입장 표명' not in final_text:
            first_h = re.search(r'^### ', final_text, re.MULTILINE)
            if first_h and first_h.start() > 0:
                intro = final_text[:first_h.start()].strip()
                rest = final_text[first_h.start():]
                if intro:
                    final_text = f"### 자기소개와 입장 표명\n{intro}\n\n{rest}"
            elif not final_text.startswith('###'):
                final_text = f"### 자기소개와 입장 표명\n{final_text}"

        idx = state["speaking_order"].index(speaker_id)
        history.append(DebateEntry(
            turn=idx, speaker_id=speaker_id, stance=agent["stance"],
            phase="opening", content=final_text, target_id=None,
            tool_calls_log=tool_calls_log, json_raw=raw,
        ))
        current_turn += 1
        print(f"  [{display}] 입론 완료\n")

    history.sort(key=lambda e: e["turn"])
    return {"debate_history": history, "current_turn": current_turn}


def ai_opening_node(state: DebateState) -> dict:
    """사용자 전의 AI 에이전트 입론만 생성."""
    speaking_order = state["speaking_order"]
    user_idx = speaking_order.index("user")
    before_user = speaking_order[:user_idx]

    print(f"\n[1단계: 입론] 사용자 전 AI: {before_user}\n")
    result = _generate_openings_for(state, before_user)
    result["phase"] = "opening"
    return result


def user_opening_node(state: DebateState) -> dict:
    """사용자 입론 interrupt → 이후 남은 AI 입론 생성."""
    user_content = interrupt("사용자 입론을 입력하세요")

    history = list(state["debate_history"])
    user_turn = state["speaking_order"].index("user")
    history.append(DebateEntry(
        turn=user_turn,
        speaker_id="user",
        stance=state["user_stance"],
        phase="opening",
        content=user_content,
        target_id=None,
    ))
    history.sort(key=lambda e: e["turn"])

    # 사용자 후 남은 AI 입론 생성
    speaking_order = state["speaking_order"]
    after_user = speaking_order[user_turn + 1:]
    after_ai = [s for s in after_user if s != "user"]

    if after_ai:
        print(f"\n[1단계: 입론] 사용자 후 AI: {after_ai}\n")
        temp_state = DebateState(**{
            **state,
            "debate_history": history,
            "current_turn": state["current_turn"],
        })
        result = _generate_openings_for(temp_state, after_ai)
        history = result["debate_history"]

    history.sort(key=lambda e: e["turn"])

    return {
        "debate_history": history,
        "current_turn": len(history),
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
