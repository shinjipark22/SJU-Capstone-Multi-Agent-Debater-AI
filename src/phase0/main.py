"""
main.py — FastAPI 애플리케이션 진입점 (Phase 0)

사용자 입력을 받아 LangGraph 초기 상태를 생성하고,
세션 정보를 반환한다. 실제 토론 실행은 Phase 1에서 구현한다.
"""

import json
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException

from src.models import (
    AgentInfo,
    DebateInitRequest,
    DebateInitResponse,
    FreeRebuttalRunResponse,
    OpeningRunResponse,
    RebuttalRunResponse,
    UserFreeRebuttalRequest,
    UserFreeRebuttalResponse,
    UserOpeningRequest,
    UserOpeningResponse,
    UserRebuttalRequest,
    UserRebuttalResponse,
)
from src.phase0.persona_factory import create_agents, AgentPersona
from src.stage1_opening.nodes import opening_arguments_node
import src.stage1_opening.nodes as _opening_mod
from src.stage2_rebuttal.nodes import (
    build_agent_stance_nums,
    chained_rebuttal_node,
    generate_ai_rebuttal,
)
from src.stage3_free_rebuttal.nodes import (
    free_rebuttal_node,
    generate_ai_free_rebuttal,
    _pick_target,
)
from src.state import (
    AgentSnapshot,
    DebateEntry,
    DebateState,
    build_initial_state,
    generate_session_id,
)

app = FastAPI(
    title="Multi-Agent Debater AI",
    description="LangGraph 기반 멀티 에이전트 토론 시스템 API",
    version="0.1.0",
)

# ── 세션 저장소 (인메모리) ──────────────────────────────────────────────────────
_sessions: Dict[str, DebateState] = {}


@app.post("/debate/init", response_model=DebateInitResponse)
def initialize_debate(request: DebateInitRequest) -> DebateInitResponse:
    """토론 세션을 초기화한다.

    1. persona_factory로 AI 에이전트 생성
    2. LangGraph 초기 State 구성
    3. 세션 ID 발급 후 응답 반환

    Phase 1에서는 이 엔드포인트 이후 /debate/run 등을 추가한다.
    """
    # ── 1. topics.json에서 topic ID로 dict 조회 ──────────────────────────────
    topics_path = Path(__file__).parent.parent / "data" / "topics_20260323_processed.json"
    if not topics_path.exists():
        raise HTTPException(status_code=404, detail="topics_20260323_processed.json 파일을 찾을 수 없습니다.")

    with topics_path.open(encoding="utf-8") as f:
        topics_data = json.load(f)

    topic_dict = None
    for category_topics in topics_data.get("categories", {}).values():
        for t in category_topics:
            if t["id"] == request.topic:
                topic_dict = t
                break
        if topic_dict:
            break

    if topic_dict is None:
        raise HTTPException(status_code=404, detail=f"topic ID '{request.topic}'를 찾을 수 없습니다.")

    # ── 2. AI 에이전트 생성 ──────────────────────────────────────────────────
    try:
        personas: list[AgentPersona] = create_agents(
            topic=topic_dict,
            debate_format=request.debate_format,
            user_stance=request.user_stance,
            agent_intensities=request.agent_intensities,
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ── 3. State용 AgentSnapshot 변환 ────────────────────────────────────────
    snapshots: list[AgentSnapshot] = [
        AgentSnapshot(
            agent_id=p.agent_id,
            stance=p.stance,
            intensity=p.intensity,
            role_description=p.role_description,
            system_prompt=p.system_prompt,
            focus_area=p.focus_area,
        )
        for p in personas
    ]

    # ── 4. LangGraph 초기 State 생성 ─────────────────────────────────────────
    initial_state = build_initial_state(
        topic=request.topic,
        user_stance=request.user_stance,
        user_intensity=request.user_intensity,
        agents=snapshots,
    )

    # ── 5. 응답 구성 ─────────────────────────────────────────────────────────
    session_id = generate_session_id()

    agent_info_list = [
        AgentInfo(
            agent_id=p.agent_id,
            stance=p.stance,
            intensity=p.intensity,
            role_description=p.role_description,
        )
        for p in personas
    ]

    _sessions[session_id] = initial_state

    return DebateInitResponse(
        session_id=session_id,
        topic=request.topic,
        agents=agent_info_list,
        initial_state=dict(initial_state),
        message="Debate workflow initialized successfully.",
    )


@app.get("/topics")
def get_topics():
    """사용 가능한 토론 주제 목록을 반환한다."""
    import json
    from pathlib import Path

    topics_path = Path(__file__).parent.parent / "data" / "topics_20260323_processed.json"
    if not topics_path.exists():
        raise HTTPException(status_code=404, detail="topics_20260323_processed.json 파일을 찾을 수 없습니다.")

    with topics_path.open(encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
def health_check():
    """서버 상태 확인."""
    return {"status": "ok", "phase": "0"}


# ── Stage 1: 입론 엔드포인트 ─────────────────────────────────────────────────────

@app.post("/debate/{session_id}/opening/run", response_model=OpeningRunResponse)
def run_opening_arguments(session_id: str) -> OpeningRunResponse:
    """AI 에이전트들의 입론을 생성한다.

    speaking_order에 따라 모든 AI 에이전트가 순차적으로 입론을 생성한다.
    사용자(user) 턴은 건너뛰며, POST /debate/{session_id}/opening/user로 별도 제출해야 한다.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    state = _sessions[session_id]

    if state["phase"] != "opening":
        raise HTTPException(
            status_code=400,
            detail=f"현재 phase가 '{state['phase']}'입니다. 'opening' 단계에서만 실행 가능합니다.",
        )

    # AI 입론이 이미 생성되었는지 확인
    ai_openings = [
        e for e in state["debate_history"]
        if e["phase"] == "opening" and e["speaker_id"] != "user"
    ]
    if ai_openings:
        raise HTTPException(status_code=400, detail="AI 입론이 이미 생성되었습니다.")

    updated_state = opening_arguments_node(state)
    _sessions[session_id] = updated_state

    # 사용자 턴 위치 확인
    user_turn = next(
        (i for i, sid in enumerate(updated_state["speaking_order"]) if sid == "user"),
        -1,
    )

    return OpeningRunResponse(
        session_id=session_id,
        phase=updated_state["phase"],
        user_turn=user_turn,
        debate_history=[dict(e) for e in updated_state["debate_history"]],
        message=f"AI 에이전트 입론 완료. 사용자 입론을 제출하세요. (turn={user_turn})",
    )


@app.post("/debate/{session_id}/opening/user", response_model=UserOpeningResponse)
def submit_user_opening(
    session_id: str,
    request: UserOpeningRequest,
) -> UserOpeningResponse:
    """사용자의 입론을 제출한다.

    사용자 입론은 speaking_order상의 위치에 해당하는 turn 번호로 기록된다.
    모든 발언자(AI + 사용자)의 입론이 완료되면 phase가 'chained_rebuttal'로 전환된다.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    state = _sessions[session_id]

    if state["phase"] != "opening":
        raise HTTPException(
            status_code=400,
            detail=f"현재 phase가 '{state['phase']}'입니다. 'opening' 단계에서만 제출 가능합니다.",
        )

    # 사용자 입론 중복 제출 방지
    if any(e["speaker_id"] == "user" and e["phase"] == "opening" for e in state["debate_history"]):
        raise HTTPException(status_code=400, detail="사용자 입론이 이미 제출되었습니다.")

    # 사용자의 speaking_order 내 턴 번호 확인
    user_turn = next(
        (i for i, sid in enumerate(state["speaking_order"]) if sid == "user"),
        None,
    )
    if user_turn is None:
        raise HTTPException(status_code=500, detail="speaking_order에서 사용자를 찾을 수 없습니다.")

    # 사용자 DebateEntry 생성
    entry = DebateEntry(
        turn=user_turn,
        speaker_id="user",
        stance=state["user_stance"],
        phase="opening",
        content=request.content,
        target_id=None,
    )

    history = list(state["debate_history"])
    history.append(entry)
    history.sort(key=lambda e: e["turn"])

    # 모든 발언자 입론 완료 여부 확인 → phase 전환
    opening_count = len([e for e in history if e["phase"] == "opening"])
    all_done = opening_count >= len(state["speaking_order"])
    next_phase: str = "chained_rebuttal" if all_done else "opening"

    updated_state = DebateState(**{
        **state,
        "debate_history": history,
        "phase": next_phase,
    })
    _sessions[session_id] = updated_state

    msg = "사용자 입론이 제출되었습니다."
    if all_done:
        msg += " 모든 입론이 완료되어 2단계 연쇄 논박(chained_rebuttal)으로 전환합니다."

    return UserOpeningResponse(
        session_id=session_id,
        phase=next_phase,
        debate_history=[dict(e) for e in history],
        message=msg,
    )


@app.get("/debate/{session_id}/state")
def get_debate_state(session_id: str):
    """현재 토론 세션 상태를 반환한다."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return dict(_sessions[session_id])


# ── Stage 2: 연쇄 논박 엔드포인트 ────────────────────────────────────────────────

def _find_pending_user_rounds(pairs: list) -> list:
    """사용자 참여가 필요한 미완료 라운드를 반환한다."""
    pending = []
    for pair in pairs:
        if pair["done"]:
            continue
        if not pair["awaiting_response"] and pair["attacker_id"] == "user":
            pending.append({**pair, "user_role": "attacker"})
        elif pair["awaiting_response"] and pair["target_id"] == "user":
            pending.append({**pair, "user_role": "responder"})
    return pending


@app.post("/debate/{session_id}/rebuttal/run", response_model=RebuttalRunResponse)
def run_chained_rebuttal(session_id: str) -> RebuttalRunResponse:
    """AI 에이전트들의 연쇄 논박을 생성한다.

    rebuttal_pairs를 기반으로 각 라운드의 공격·응답을 순서대로 처리한다.
    사용자(user) 차례는 건너뛰며, POST /debate/{session_id}/rebuttal/user로 별도 제출해야 한다.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    state = _sessions[session_id]

    if state["phase"] != "chained_rebuttal":
        raise HTTPException(
            status_code=400,
            detail=f"현재 phase가 '{state['phase']}'입니다. 'chained_rebuttal' 단계에서만 실행 가능합니다.",
        )

    # 이미 AI 논박이 진행된 경우 중복 실행 방지
    ai_rebuttals = [
        e for e in state["debate_history"]
        if e["phase"] == "chained_rebuttal" and e["speaker_id"] != "user"
    ]
    if ai_rebuttals:
        raise HTTPException(status_code=400, detail="AI 연쇄 논박이 이미 생성되었습니다.")

    updated_state = chained_rebuttal_node(state)
    _sessions[session_id] = updated_state

    pairs = updated_state["rebuttal_pairs"] or []
    pending = _find_pending_user_rounds(pairs)

    msg = "AI 에이전트 연쇄 논박 완료."
    if pending:
        msg += f" 사용자 참여 대기 중인 라운드가 {len(pending)}개 있습니다."
    else:
        msg += f" phase={updated_state['phase']}."

    return RebuttalRunResponse(
        session_id=session_id,
        phase=updated_state["phase"],
        rebuttal_pairs=[dict(p) for p in pairs],
        pending_user_rounds=pending,
        debate_history=[dict(e) for e in updated_state["debate_history"]],
        message=msg,
    )


@app.post("/debate/{session_id}/rebuttal/user", response_model=UserRebuttalResponse)
def submit_user_rebuttal(
    session_id: str,
    request: UserRebuttalRequest,
) -> UserRebuttalResponse:
    """사용자의 연쇄 논박을 제출한다.

    미완료 라운드 중 사용자 차례인 첫 번째 라운드를 자동으로 찾아 처리한다.
    사용자가 공격자인 경우, 제출 후 AI 응답자의 반박을 자동 생성한다.
    모든 라운드 완료 시 phase가 'free_rebuttal'로 전환된다.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    state = _sessions[session_id]

    if state["phase"] != "chained_rebuttal":
        raise HTTPException(
            status_code=400,
            detail=f"현재 phase가 '{state['phase']}'입니다. 'chained_rebuttal' 단계에서만 제출 가능합니다.",
        )

    pairs = [dict(p) for p in (state["rebuttal_pairs"] or [])]
    pending = _find_pending_user_rounds(pairs)

    if not pending:
        raise HTTPException(status_code=400, detail="사용자 연쇄 논박 차례가 없습니다.")

    # 첫 번째 대기 라운드 처리
    target_round = pending[0]
    pair_idx = next(i for i, p in enumerate(pairs) if p["round"] == target_round["round"])
    pair = pairs[pair_idx]
    user_role = target_round["user_role"]

    history = list(state["debate_history"])
    current_turn = state["current_turn"]
    agent_map = {a["agent_id"]: a for a in state["agents"]}

    opponent_id = pair["target_id"] if user_role == "attacker" else pair["attacker_id"]

    # 사용자 DebateEntry 생성
    user_entry = DebateEntry(
        turn=current_turn,
        speaker_id="user",
        stance=state["user_stance"],
        phase="chained_rebuttal",
        content=request.content,
        target_id=opponent_id,
    )
    history.append(user_entry)
    current_turn += 1

    if user_role == "attacker":
        # 사용자가 공격 → AI 응답자의 반박 자동 생성
        pairs[pair_idx]["awaiting_response"] = True

        responder_id = pair["target_id"]
        agent = agent_map[responder_id]
        stance_nums = build_agent_stance_nums(state["agents"], state["speaking_order"])

        _opening_mod._used_doc_ids = set()

        ai_entry = generate_ai_rebuttal(
            topic=state["topic"],
            history=history,
            agent=agent,
            target_id="user",
            stance_num=stance_nums[responder_id],
            target_stance_num=0,
            current_turn=current_turn,
            is_response=True,
        )
        history.append(ai_entry)
        current_turn += 1
        pairs[pair_idx]["done"] = True
    else:
        # 사용자가 응답 → 라운드 완료
        pairs[pair_idx]["done"] = True

    # 완료 판단
    all_done = all(p["done"] for p in pairs)
    next_phase = "free_rebuttal" if all_done else "chained_rebuttal"

    updated_state = DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "rebuttal_pairs": pairs,
        "phase": next_phase,
    })
    _sessions[session_id] = updated_state

    msg = "사용자 연쇄 논박이 제출되었습니다."
    if user_role == "attacker":
        msg += f" {opponent_id}의 AI 응답이 자동 생성되었습니다."
    if all_done:
        msg += " 모든 연쇄 논박이 완료되어 3단계 자유 논박(free_rebuttal)으로 전환합니다."

    return UserRebuttalResponse(
        session_id=session_id,
        phase=next_phase,
        debate_history=[dict(e) for e in history],
        message=msg,
    )


# ── Stage 3: 자유 논박 엔드포인트 ────────────────────────────────────────────────

@app.post("/debate/{session_id}/free-rebuttal/run", response_model=FreeRebuttalRunResponse)
def run_free_rebuttal(session_id: str) -> FreeRebuttalRunResponse:
    """AI 에이전트들의 자유 논박을 생성한다.

    speaking_order에 따라 한 사이클 동안 모든 AI 에이전트가 순서대로 발언한다.
    사용자(user) 차례는 건너뛰며, POST /debate/{session_id}/free-rebuttal/user로 별도 제출해야 한다.
    current_cycle >= max_cycle이면 phase가 'role_reversal'로 전환된다.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    state = _sessions[session_id]

    if state["phase"] != "free_rebuttal":
        raise HTTPException(
            status_code=400,
            detail=f"현재 phase가 '{state['phase']}'입니다. 'free_rebuttal' 단계에서만 실행 가능합니다.",
        )

    updated_state = free_rebuttal_node(state)
    _sessions[session_id] = updated_state

    has_user_turn = "user" in updated_state["speaking_order"]
    cycle = updated_state["current_cycle"]
    max_c = updated_state["max_cycle"]

    msg = f"AI 자유 논박 완료 (사이클 {cycle}/{max_c})."
    if has_user_turn:
        msg += " 사용자 발언을 제출하세요."
    if updated_state["phase"] == "role_reversal":
        msg += " 최대 사이클 도달 → 4단계 역할 반전으로 전환합니다."

    return FreeRebuttalRunResponse(
        session_id=session_id,
        phase=updated_state["phase"],
        current_cycle=updated_state["current_cycle"],
        max_cycle=updated_state["max_cycle"],
        debate_history=[dict(e) for e in updated_state["debate_history"]],
        message=msg,
    )


@app.post("/debate/{session_id}/free-rebuttal/user", response_model=UserFreeRebuttalResponse)
def submit_user_free_rebuttal(
    session_id: str,
    request: UserFreeRebuttalRequest,
) -> UserFreeRebuttalResponse:
    """사용자의 자유 논박을 제출한다.

    사용자 발언 후, 현재 사이클의 남은 AI 발언자가 있으면 자동 생성한다.
    사이클 완료 후 current_cycle >= max_cycle이면 phase를 'role_reversal'로 전환한다.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    state = _sessions[session_id]

    if state["phase"] != "free_rebuttal":
        raise HTTPException(
            status_code=400,
            detail=f"현재 phase가 '{state['phase']}'입니다. 'free_rebuttal' 단계에서만 제출 가능합니다.",
        )

    # target_id 유효성 검증
    agent_map = {a["agent_id"]: a for a in state["agents"]}
    if request.target_id not in agent_map:
        raise HTTPException(
            status_code=400,
            detail=f"target_id '{request.target_id}'는 유효한 에이전트가 아닙니다.",
        )

    history = list(state["debate_history"])
    current_turn = state["current_turn"]
    speaking_order = state["speaking_order"]
    stance_nums = build_agent_stance_nums(state["agents"], speaking_order)

    # 사용자 DebateEntry 생성
    user_entry = DebateEntry(
        turn=current_turn,
        speaker_id="user",
        stance=state["user_stance"],
        phase="free_rebuttal",
        content=request.content,
        target_id=request.target_id,
    )
    history.append(user_entry)
    current_turn += 1

    # 사용자 이후 남은 AI 발언자 자동 생성
    user_idx = speaking_order.index("user")
    remaining_speakers = speaking_order[user_idx + 1:]

    _opening_mod._used_doc_ids = set()

    for speaker_id in remaining_speakers:
        if speaker_id == "user":
            continue
        agent = agent_map[speaker_id]
        target_id = _pick_target(history, speaker_id, agent["stance"], state["agents"])
        if target_id is None:
            continue

        stance_label = "찬성" if agent["stance"] == "PRO" else "반대"
        display_name = f"{stance_label} 에이전트{stance_nums[speaker_id]}"
        target_snum = stance_nums.get(target_id, 0)
        print(f"  [{display_name}] 자유 논박 생성 중...")

        entry = generate_ai_free_rebuttal(
            topic=state["topic"],
            history=history,
            agent=agent,
            target_id=target_id,
            stance_num=stance_nums[speaker_id],
            target_stance_num=target_snum,
            current_turn=current_turn,
        )
        history.append(entry)
        current_turn += 1

    # 사이클 완료 처리
    current_cycle = state["current_cycle"] + 1
    max_cycle = state["max_cycle"]

    if current_cycle >= max_cycle:
        next_phase = "role_reversal"
    else:
        next_phase = "free_rebuttal"

    updated_state = DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "current_cycle": current_cycle,
        "phase": next_phase,
    })
    _sessions[session_id] = updated_state

    msg = f"사용자 자유 논박이 제출되었습니다 (사이클 {current_cycle}/{max_cycle})."
    if next_phase == "role_reversal":
        msg += " 최대 사이클 도달 → 4단계 역할 반전(role_reversal)으로 전환합니다."

    return UserFreeRebuttalResponse(
        session_id=session_id,
        phase=next_phase,
        current_cycle=current_cycle,
        max_cycle=max_cycle,
        debate_history=[dict(e) for e in history],
        message=msg,
    )
