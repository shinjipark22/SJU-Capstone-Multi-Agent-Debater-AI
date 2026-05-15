"""
main_graph.py — LangGraph 메인 토론 그래프

전체 토론 흐름(1~5단계)을 하나의 StateGraph로 구현.
사용자 입력은 interrupt로 대기, FastAPI에서 Command(resume=)로 재개.

[그래프 흐름]
    ai_opening_pre (loop) → user_opening → ai_opening_post (loop)
    → ai_rebuttal_step (loop) ↔ user_rebuttal
    → ai_free_rebuttal_defense → ai_free_rebuttal_attack
    → user_free_rebuttal_defense → user_free_rebuttal_attack (루프)
    → ai_role_reversal → user_role_reversal
    → ai_synthesis ↔ user_synthesis (루프)
    → user_finalize → END

각 step 노드는 한 발화만 생성한 뒤 LangGraph 가 chunk 를 yield 하므로
SSE 가 발화 단위로 즉시 push 됨 (TTFT 개선).
"""

from __future__ import annotations

import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from src.state import DebateEntry, DebateState
from src.phase1.stage2_rebuttal.nodes import process_one_rebuttal_step
from src.phase1.stage3_free_rebuttal.nodes import (
    free_rebuttal_attack_node as _fr_attack_impl,
    free_rebuttal_defense_node as _fr_defense_impl,
)
from src.phase1.stage4_role_reversal.nodes import role_reversal_node
from src.phase1.stage5_synthesis.nodes import (
    synthesis_propose_one_node,
    synthesis_discuss_one_node,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 노드 함수
# ═══════════════════════════════════════════════════════════════════════════

# ── 1단계: 입론 ────────────────────────────────────────────────────────────

def _generate_openings_for(state: DebateState, speaker_ids: list) -> dict:
    """지정된 speaker_ids에 대해서만 입론을 생성한다."""
    import re
    from src.phase1.stage1_opening.nodes import (
        _generate_opening, _get_focus_area, _build_opening_prompt, _is_valid_speech,
        _focus_index_within_stance,
    )
    import src.phase1.stage1_opening.nodes as _opening_mod
    _opening_mod._used_doc_ids = set()

    topic = state["topic"]
    history = list(state["debate_history"])
    agent_map = {a["agent_id"]: a for a in state["agents"]}
    stance_counter = {"PRO": 0, "CON": 0}

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

        focus_area = agent.get("focus_area") or _get_focus_area(
            agent["stance"],
            topic_id=state.get("topic_id", ""),
            index=_focus_index_within_stance(agent, state["agents"]),
        )
        prompt = _build_opening_prompt(topic, agent["stance"], display, focus_area)
        final_text, raw, tool_calls_log = _generate_opening(agent, prompt)

        if not _is_valid_speech(final_text):
            final_text = (
                f"### 자기소개와 입장 표명\n"
                f"저는 {display}입니다. {topic}에 대해 {slabel} 입장입니다.\n\n"
                f"### 결론\n저는 {slabel} 입장을 유지합니다."
            )

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


def _opening_pre_speakers(state: DebateState) -> list:
    """사용자 전에 발언할 AI speaker_id 리스트."""
    speaking_order = state["speaking_order"]
    user_idx = speaking_order.index("user")
    return [s for s in speaking_order[:user_idx] if s != "user"]


def _opening_post_speakers(state: DebateState) -> list:
    """사용자 후에 발언할 AI speaker_id 리스트."""
    speaking_order = state["speaking_order"]
    user_idx = speaking_order.index("user")
    return [s for s in speaking_order[user_idx + 1:] if s != "user"]


def ai_opening_pre_step_node(state: DebateState) -> dict:
    """사용자 전 AI 입론 — 한 명씩 생성. opening_pre_idx 카운터로 다음 에이전트 결정.

    노드 분리 이유: 각 에이전트 발화가 LangGraph chunk 로 따로 yield 되어
    SSE 가 발화 단위로 즉시 push 됨 (TTFT 개선).
    """
    pre_speakers = _opening_pre_speakers(state)
    idx = state.get("opening_pre_idx", 0)
    if idx >= len(pre_speakers):
        return {"phase": "opening"}

    speaker_id = pre_speakers[idx]
    print(f"\n[1단계: 입론] 사용자 전 AI {idx+1}/{len(pre_speakers)}: {speaker_id}\n")
    result = _generate_openings_for(state, [speaker_id])
    return {
        "debate_history": result["debate_history"],
        "current_turn": result["current_turn"],
        "opening_pre_idx": idx + 1,
        "phase": "opening",
    }


def route_opening_pre(state: DebateState) -> str:
    """사용자 전 AI 입론이 다 끝났으면 user_opening 으로, 아니면 다시 step."""
    return "next" if state.get("opening_pre_idx", 0) < len(_opening_pre_speakers(state)) else "done"


def user_opening_node(state: DebateState) -> dict:
    """사용자 입론 interrupt — 입력만 받음 (post AI 는 별도 step 노드에서 처리)."""
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

    return {
        "debate_history": history,
        "current_turn": state["current_turn"] + 1,
        "phase": "opening",
    }


def ai_opening_post_step_node(state: DebateState) -> dict:
    """사용자 후 AI 입론 — 한 명씩 생성. opening_post_idx 카운터."""
    post_speakers = _opening_post_speakers(state)
    idx = state.get("opening_post_idx", 0)
    if idx >= len(post_speakers):
        return {"phase": "chained_rebuttal"}

    speaker_id = post_speakers[idx]
    print(f"\n[1단계: 입론] 사용자 후 AI {idx+1}/{len(post_speakers)}: {speaker_id}\n")
    result = _generate_openings_for(state, [speaker_id])
    return {
        "debate_history": result["debate_history"],
        "current_turn": result["current_turn"],
        "opening_post_idx": idx + 1,
        "phase": "opening",
    }


def route_opening_post(state: DebateState) -> str:
    """사용자 후 AI 입론이 다 끝났으면 chained_rebuttal 으로, 아니면 다시 step."""
    return "next" if state.get("opening_post_idx", 0) < len(_opening_post_speakers(state)) else "done"


# ── 2단계: 연쇄논박 ───────────────────────────────────────────────────────

def ai_rebuttal_step_node(state: DebateState) -> dict:
    """미완료 pair 중 다음 AI 공격 1개만 처리. 사용자 차례면 entry 추가 없이 phase 그대로 두고 종료.

    노드 분리 이유: 한 pair 처리할 때마다 LangGraph chunk yield → SSE 즉시 push.
    """
    updated = process_one_rebuttal_step(state)
    return {
        "debate_history": updated["debate_history"],
        "current_turn": updated["current_turn"],
        "rebuttal_pairs": updated["rebuttal_pairs"],
        "phase": updated["phase"],
    }


def route_rebuttal(state: DebateState) -> str:
    """미완료 pair 중 다음이 사용자면 'to_user', AI면 'next', 모두 완료면 'done'."""
    pairs = state.get("rebuttal_pairs") or []
    for p in pairs:
        if p["done"]:
            continue
        return "to_user" if p["attacker_id"] == "user" else "next"
    return "done"


def user_rebuttal_node(state: DebateState) -> dict:
    """사용자 연쇄논박 — interrupt로 대기.

    분리 후 동작: 사용자가 발언한 후 미완료 user pair 를 모두 done 마크 →
    ai_rebuttal_step 으로 돌아가도 user pair 무한 loop 안 걸림.
    """
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

    # 미완료 user pair 모두 done 마크 (사용자가 한 번 발언했으니 user 측 발언 라운드 완료)
    pairs = [dict(p) for p in (state.get("rebuttal_pairs") or [])]
    for p in pairs:
        if not p["done"] and p["attacker_id"] == "user":
            p["done"] = True

    return {
        "debate_history": history,
        "current_turn": state["current_turn"] + 1,
        "rebuttal_pairs": pairs,
        "phase": "chained_rebuttal",
    }


# ── 3단계: 자유논박 ───────────────────────────────────────────────────────

def _ensure_opponent_selected(state: DebateState) -> DebateState:
    """selected_opponent_id 가 없으면 사용자 진영 반대 쪽 첫 에이전트로 자동 선택."""
    if state.get("selected_opponent_id"):
        return state
    opposite = "CON" if state["user_stance"] == "PRO" else "PRO"
    opponent = next((a for a in state["agents"] if a["stance"] == opposite), None)
    if opponent:
        return DebateState(**{**state, "selected_opponent_id": opponent["agent_id"]})
    return state


def ai_free_rebuttal_defense_node(state: DebateState) -> dict:
    """AI 자유논박 — 방어 단계만. 첫 턴/사용자 공격 없으면 entry 없이 통과.

    노드 분리 이유: SSE 가 발화 단위로 즉시 push 되도록 (방어 끝나면 바로 frontend 로,
    공격은 다음 노드에서 이어서 push).
    """
    state = _ensure_opponent_selected(state)
    updated = _fr_defense_impl(state)
    return {
        "debate_history": updated["debate_history"],
        "current_turn": updated["current_turn"],
        "selected_opponent_id": updated.get("selected_opponent_id", state.get("selected_opponent_id")),
        "phase": "free_rebuttal",
    }


def ai_free_rebuttal_attack_node(state: DebateState) -> dict:
    """AI 자유논박 — 공격 단계만. 마지막 턴이면 entry 없이 통과."""
    updated = _fr_attack_impl(state)
    return {
        "debate_history": updated["debate_history"],
        "current_turn": updated["current_turn"],
        "selected_opponent_id": updated.get("selected_opponent_id", state.get("selected_opponent_id")),
        "phase": "free_rebuttal",
    }


def user_free_rebuttal_defense_node(state: DebateState) -> dict:
    """사용자 자유논박 — 답변 단계 (interrupt 1회).

    한 노드 안 interrupt 2회 패턴이 LangGraph 재실행 시 `graph_state.next`를
    빈 리스트로 보고하는 이슈가 있어 답변/공격 노드를 분리.
    카운터 증가는 attack 노드에서 일괄 수행한다.
    """
    selected_id = state.get("selected_opponent_id", "agent_1")

    user_defense = interrupt("상대 공격에 대한 답변을 입력하세요")

    history = list(state["debate_history"])
    current_turn = state["current_turn"]

    history.append(DebateEntry(
        turn=current_turn,
        speaker_id="user",
        stance=state["user_stance"],
        phase="free_rebuttal",
        content=user_defense,
        target_id=selected_id,
    ))
    current_turn += 1

    return {
        "debate_history": history,
        "current_turn": current_turn,
    }


def user_free_rebuttal_attack_node(state: DebateState) -> dict:
    """사용자 자유논박 — 공격 단계 (interrupt 1회, 최종 턴이면 스킵).

    1회차: 공격 입력 받음 + 카운터 +1 (0→1).
    2회차(최종, `free_rebuttal_user_turns >= 1`): 공격 스킵 + 카운터 +1 (1→2).
    """
    selected_id = state.get("selected_opponent_id", "agent_1")
    current_turns = state.get("free_rebuttal_user_turns", 0)
    is_final = (current_turns >= 1)

    history = list(state["debate_history"])
    current_turn = state["current_turn"]

    if not is_final:
        user_attack = interrupt("상대 논거를 공격하세요")
        history.append(DebateEntry(
            turn=current_turn,
            speaker_id="user",
            stance=state["user_stance"],
            phase="free_rebuttal",
            content=user_attack,
            target_id=selected_id,
        ))
        current_turn += 1

    return {
        "debate_history": history,
        "current_turn": current_turn,
        "free_rebuttal_user_turns": current_turns + 1,
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

def _ai_speakers_count(state: DebateState) -> int:
    return len([s for s in state["speaking_order"] if s != "user"])


def _is_first_synthesis_round(state: DebateState) -> bool:
    """현재 라운드가 초기 의견 제시 (사용자 발언 없음) 인지."""
    return not any(
        e["speaker_id"] == "user" and e["phase"] == "synthesis"
        for e in state.get("debate_history", [])
    )


def ai_synthesis_step_node(state: DebateState) -> dict:
    """종합 회의 — 한 AI 에이전트만 발언. 첫 라운드면 propose, 사용자 발언 후면 discuss.

    카운터: synthesis_propose_idx (첫 라운드), synthesis_discuss_idx (응답 라운드).
    한 AI 발언 = 한 LangGraph chunk → SSE 즉시 push.
    """
    if _is_first_synthesis_round(state):
        updated = synthesis_propose_one_node(state)
    else:
        updated = synthesis_discuss_one_node(state)
    return {
        "debate_history": updated["debate_history"],
        "current_turn": updated["current_turn"],
        "synthesis_propose_idx": updated.get("synthesis_propose_idx", state.get("synthesis_propose_idx", 0)),
        "synthesis_discuss_idx": updated.get("synthesis_discuss_idx", state.get("synthesis_discuss_idx", 0)),
        "phase": "synthesis",
        "is_finished": False,
    }


def route_synthesis_step(state: DebateState) -> str:
    """현재 라운드 AI 들이 다 말했는지 — 카운터 vs AI 수 비교."""
    n_ai = _ai_speakers_count(state)
    if _is_first_synthesis_round(state):
        idx = state.get("synthesis_propose_idx", 0)
    else:
        idx = state.get("synthesis_discuss_idx", 0)
    return "next" if idx < n_ai else "done"


def user_synthesis_node(state: DebateState) -> dict:
    """사용자 종합 의견 — interrupt로 대기. Round 3에는 개별 최적해 선언 유도.

    분리 후 동작: 사용자 발언 후 다음 discuss 라운드를 위해 synthesis_discuss_idx 를 0 으로 리셋.
    """
    is_final_round = state.get("synthesis_user_turns", 0) >= 2
    if is_final_round:
        user_content = interrupt(
            "Round 3: '제가 생각하는 최적해는 ~입니다' 형식으로 개별 최적해를 선언해주세요 (2~3문장)"
        )
    else:
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
        "synthesis_discuss_idx": 0,  # 다음 discuss 라운드를 위해 카운터 리셋
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
    """종합 회의 루프 라우터: 사용자 3턴 완료 → finalize, 아니면 continue.

    기대 흐름: (에이전트들 + 사용자) × 3라운드 → 마지막에 user_finalize.
    """
    if state.get("synthesis_user_turns", 0) >= 3:
        return "finalize"
    return "continue"


# ═══════════════════════════════════════════════════════════════════════════
# 그래프 빌더
# ═══════════════════════════════════════════════════════════════════════════

def build_debate_graph():
    """전체 토론 메인 그래프를 빌드하고 컴파일한다."""
    graph = StateGraph(DebateState)

    # 노드 등록
    # 1단계 입론: pre/post 모두 한 에이전트씩 step 노드로 — SSE 가 발화 단위로 즉시 push
    graph.add_node("ai_opening_pre", ai_opening_pre_step_node)
    graph.add_node("user_opening", user_opening_node)
    graph.add_node("ai_opening_post", ai_opening_post_step_node)
    # 2단계 연쇄논박: pair 한 개씩 step 노드 — 사용자 차례 만나면 user_rebuttal 로 분기
    graph.add_node("ai_rebuttal_step", ai_rebuttal_step_node)
    graph.add_node("user_rebuttal", user_rebuttal_node)
    # 3단계 자유논박: 방어/공격 노드 분리
    graph.add_node("ai_free_rebuttal_defense", ai_free_rebuttal_defense_node)
    graph.add_node("ai_free_rebuttal_attack", ai_free_rebuttal_attack_node)
    graph.add_node("user_free_rebuttal_defense", user_free_rebuttal_defense_node)
    graph.add_node("user_free_rebuttal_attack", user_free_rebuttal_attack_node)
    graph.add_node("ai_role_reversal", ai_role_reversal_node)
    graph.add_node("user_role_reversal", user_role_reversal_node)
    # 5단계 종합: AI 한 명씩 step 노드 — propose/discuss 모두
    graph.add_node("ai_synthesis_step", ai_synthesis_step_node)
    graph.add_node("user_synthesis", user_synthesis_node)
    graph.add_node("user_finalize", user_finalize_node)

    # 엣지: 순차 흐름
    # 1단계: pre step (loop) → user_opening → post step (loop) → 2단계
    graph.set_entry_point("ai_opening_pre")
    graph.add_conditional_edges("ai_opening_pre", route_opening_pre, {
        "next": "ai_opening_pre",
        "done": "user_opening",
    })
    graph.add_edge("user_opening", "ai_opening_post")
    graph.add_conditional_edges("ai_opening_post", route_opening_post, {
        "next": "ai_opening_post",
        "done": "ai_rebuttal_step",
    })
    # 2단계: AI step (loop) → 사용자 차례 만나면 user_rebuttal → 다시 ai_rebuttal_step
    # → done 이면 자유논박. user_rebuttal 노드가 user pair 들을 done 마크하므로 무한 loop 없음.
    graph.add_conditional_edges("ai_rebuttal_step", route_rebuttal, {
        "next": "ai_rebuttal_step",
        "to_user": "user_rebuttal",
        "done": "ai_free_rebuttal_defense",
    })
    graph.add_edge("user_rebuttal", "ai_rebuttal_step")

    # AI 자유논박: 방어 → 공격 직진. 라우팅(역할반전 분기)은 공격 노드 뒤에서 결정.
    graph.add_edge("ai_free_rebuttal_defense", "ai_free_rebuttal_attack")
    graph.add_conditional_edges("ai_free_rebuttal_attack", route_after_ai_free, {
        "to_user": "user_free_rebuttal_defense",
        "end_free": "ai_role_reversal",
    })

    # 답변 → 공격 (공격 노드가 최종 턴이면 interrupt 없이 바로 카운터만 증가)
    graph.add_edge("user_free_rebuttal_defense", "user_free_rebuttal_attack")

    # 사용자 자유논박 후 → continue면 AI 방어부터 다시, done(2라운드 완료)이면 바로 역할반전
    graph.add_conditional_edges("user_free_rebuttal_attack", route_free_rebuttal, {
        "continue": "ai_free_rebuttal_defense",
        "done": "ai_role_reversal",  # user의 최종 답변 후 AI 응답 없이 역할반전으로
    })

    # 역할반전 → 종합
    graph.add_edge("ai_role_reversal", "user_role_reversal")
    graph.add_edge("user_role_reversal", "ai_synthesis_step")

    # 5단계 종합: ai_synthesis_step (loop, AI 한 명씩) → 모두 끝나면 user_synthesis
    graph.add_conditional_edges("ai_synthesis_step", route_synthesis_step, {
        "next": "ai_synthesis_step",
        "done": "user_synthesis",
    })
    # 사용자 발언 후 → 다음 라운드 (continue) 또는 finalize
    graph.add_conditional_edges("user_synthesis", route_synthesis, {
        "continue": "ai_synthesis_step",
        "finalize": "user_finalize",
    })
    graph.add_edge("user_finalize", END)

    # 체크포인터 (세션 상태 저장)
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
