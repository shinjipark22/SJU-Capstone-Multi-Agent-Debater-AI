"""
nodes.py — 3단계: 자유 논박(Free Rebuttal) 노드

[동작 흐름]
    LangGraph 서브그래프 기반:
    1. 방어: Write → Review (서브그래프)
    2. 공격: Search → Write → Review (서브그래프) + 질문 추가
    3. 멀티턴 메시지 체인으로 대화 맥락 유지
"""

from __future__ import annotations

import logging
import random
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.graph.subgraphs import search_write_review, write_review
from src.graph.searcher import generate_attack_question
from src.phase1.stage2_rebuttal.nodes import build_agent_stance_nums
from src.state import DebateEntry, DebateState


# ── 입론에서 논거 추출 ────────────────────────────────────────────────────

def _pick_one_argument(speech: str) -> str:
    """입론에서 논거 1 또는 논거 2를 랜덤으로 하나만 추출한다."""
    parts = re.split(r'###\s*논거\s*\d+\s*[:：]?', speech)
    arguments = []
    for i, p in enumerate(parts):
        if i == 0:
            continue
        conclusion_idx = p.find('### 결론')
        if conclusion_idx != -1:
            p = p[:conclusion_idx]
        text = p.strip()
        if text and len(text) > 20:
            arguments.append(text)
    if arguments:
        return random.choice(arguments)
    return speech


# ── 멀티턴 메시지 체인 구축 ───────────────────────────────────────────────

def _build_message_chain(history: List, selected_id: str) -> List:
    """자유논박 히스토리에서 멀티턴 메시지 체인을 구축한다 (SystemMessage 없이)."""
    messages = []
    fr_entries = [e for e in history if e["phase"] == "free_rebuttal"]
    for entry in fr_entries:
        if entry["speaker_id"] == selected_id:
            messages.append(AIMessage(content=entry["content"]))
        elif entry["speaker_id"] == "user":
            messages.append(HumanMessage(content=entry["content"]))
    return messages


# ── 이전 약점/공격 수집 ───────────────────────────────────────────────────

def _collect_prev_context(agent_entries: List) -> tuple:
    """이전 공격 내용과 약점 분석을 수집한다."""
    prev_attacks = [e["content"][:100] for e in agent_entries]
    prev_attacks_text = "\n".join(f"- {a}" for a in prev_attacks[-3:]) if prev_attacks else ""

    prev_weaknesses = "\n".join(
        log.get("result", "")
        for e in agent_entries
        for log in e.get("tool_calls_log", [])
        if log.get("name") == "analyze_weakness" and log.get("result")
    )
    return prev_attacks_text, prev_weaknesses


# ── 서브그래프용 TurnState 초기값 ─────────────────────────────────────────

def _make_turn_state(
    topic: str, agent: Dict, expected_stance: str,
    target_argument: str, my_opening: str, opp_opening: str,
    chain: List, prev_weaknesses: str, prev_attacks: str, mode: str,
) -> dict:
    """TurnState 초기값을 생성한다."""
    return {
        "topic": topic,
        "agent": agent,
        "expected_stance": expected_stance,
        "target_argument": target_argument,
        "my_opening": my_opening,
        "opp_opening": opp_opening,
        "chain": chain,
        "prev_weaknesses": prev_weaknesses,
        "prev_attacks": prev_attacks,
        "mode": mode,
        "weakness": "",
        "search_results": "",
        "search_query": "",
        "speech": "",
        "raw": "",
        "review_result": {},
        "retry_count": 0,
        "tool_calls_log": [],
    }


# ── 메인 노드 ──────────────────────────────────────────────────────────────

def free_rebuttal_node(state: DebateState) -> DebateState:
    """3단계 자유 논박 노드 (LangGraph 서브그래프 기반).

    방어: write_review 서브그래프
    공격: search_write_review 서브그래프 + Qwen 질문 추가
    """
    # ── 상대 에이전트 확인
    selected_id = state.get("selected_opponent_id")
    if not selected_id:
        raise ValueError("[free_rebuttal] selected_opponent_id가 필요합니다.")

    agent_map = {a["agent_id"]: a for a in state["agents"]}
    if selected_id not in agent_map:
        raise ValueError(f"[free_rebuttal] 존재하지 않는 에이전트: {selected_id}")

    opponent = agent_map[selected_id]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    speaking_order = state["speaking_order"]
    stance_nums = build_agent_stance_nums(state["agents"], speaking_order)

    opponent_display = (
        f"{'찬성' if opponent['stance'] == 'PRO' else '반대'}"
        f"{stance_nums.get(selected_id, 0)}"
    )
    print(f"\n[3단계: 자유 논박] 사용자 ↔ {opponent_display}\n")

    # ── 입론 추출
    my_opening = ""
    opp_opening = ""
    for e in history:
        if e["phase"] == "opening" and e["speaker_id"] == selected_id:
            my_opening = e["content"]
        if e["phase"] == "opening" and e["speaker_id"] == "user":
            opp_opening = e["content"]

    # ── 멀티턴 체인 + 이전 맥락 수집
    chain = _build_message_chain(history, selected_id)
    user_entries = [e for e in history if e["speaker_id"] == "user" and e["phase"] == "free_rebuttal"]
    agent_entries = [e for e in history if e["speaker_id"] == selected_id and e["phase"] == "free_rebuttal"]
    is_first_turn = len(agent_entries) == 0

    user_latest_defense = user_entries[-2]["content"] if len(user_entries) >= 2 else (user_entries[-1]["content"] if user_entries else "")
    user_latest_attack = user_entries[-1]["content"] if user_entries else ""
    prev_attacks_text, prev_weaknesses = _collect_prev_context(agent_entries)

    speeches = []
    all_tool_calls = []

    # ── Step 1: 방어 (search_write_review 서브그래프 — 팩트체크 검색 포함)
    if not is_first_turn and user_latest_attack:
        print(f"  [Step 1 - 방어] search_write_review 서브그래프 실행\n")

        defense_result = search_write_review.invoke(_make_turn_state(
            topic=state["topic"], agent=opponent,
            expected_stance=opponent["stance"],
            target_argument=user_latest_attack,
            my_opening=my_opening, opp_opening=opp_opening,
            chain=list(chain), prev_weaknesses="", prev_attacks="",
            mode="defense",
        ))
        defense = defense_result["speech"]
        raw_def = defense_result["raw"]
        all_tool_calls.extend(defense_result.get("tool_calls_log", []))
        speeches.append(("답변", defense, raw_def))
        print(f"  [방어 완료] review: {defense_result.get('review_result', {})}\n")

    # ── Step 2: 공격 (search_write_review 서브그래프)
    if not is_first_turn and user_latest_defense:
        target_argument = user_latest_defense
    else:
        target_argument = _pick_one_argument(opp_opening)

    print(f"  [Step 2 - 공격] search_write_review 서브그래프 실행\n")

    attack_chain = list(chain)
    if speeches:
        attack_chain.append(AIMessage(content=speeches[0][1]))

    attack_result = search_write_review.invoke(_make_turn_state(
        topic=state["topic"], agent=opponent,
        expected_stance=opponent["stance"],
        target_argument=target_argument,
        my_opening=my_opening, opp_opening=opp_opening,
        chain=attack_chain, prev_weaknesses=prev_weaknesses,
        prev_attacks=prev_attacks_text, mode="attack",
    ))
    attack = attack_result["speech"]
    raw_atk = attack_result["raw"]
    all_tool_calls.extend(attack_result.get("tool_calls_log", []))
    print(f"  [공격 완료] review: {attack_result.get('review_result', {})}\n")

    # ── Qwen 질문 생성 + 후처리
    attack_question = generate_attack_question(attack, opponent["stance"], state["topic"])
    if attack_question:
        all_tool_calls.append({"name": "attack_question", "result": attack_question})
        print(f"  [공격 질문] {attack_question[:60]}\n")

    if "?" not in attack:
        q = attack_question if attack_question else "이에 대해 상대는 어떻게 설명하시겠습니까?"
        attack = attack.rstrip() + " " + q

    speeches.append(("공격", attack, raw_atk))

    # ── 발언 기록
    for label, speech, raw in speeches:
        history.append(DebateEntry(
            turn=current_turn,
            speaker_id=selected_id,
            stance=opponent["stance"],
            phase="free_rebuttal",
            content=speech,
            target_id="user",
            tool_calls_log=all_tool_calls,
            json_raw=raw,
        ))
        current_turn += 1
        print(f"  [{opponent_display} - {label}] (turn={current_turn - 1})")
        print(f"  {speech[:80]}...\n")

    print(f"[3단계: 자유 논박] 에이전트 발언 완료 → 사용자 발언 대기\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "phase": "free_rebuttal",
    })


# ── 턴 라우터 ──────────────────────────────────────────────────────────────

def should_end_free_rebuttal(state: DebateState) -> bool:
    """자유논박 종료 조건: 사용자 2턴 완료."""
    return state.get("free_rebuttal_user_turns", 0) >= 2


def is_final_agent_turn(state: DebateState) -> bool:
    """에이전트 최종 답변 차례인지."""
    return state.get("free_rebuttal_user_turns", 0) >= 2
