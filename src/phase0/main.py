"""
main.py — FastAPI 애플리케이션 진입점 (Phase 0)

사용자 입력을 받아 LangGraph 초기 상태를 생성하고,
세션 정보를 반환한다. 실제 토론 실행은 Phase 1에서 구현한다.
"""

import json
from pathlib import Path
from typing import Dict, Generator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.models import (
    AgentInfo,
    DebateInitRequest,
    DebateInitResponse,
    UserOpeningRequest,
    UserOpeningResponse,
)
from src.phase0.persona_factory import create_agents, AgentPersona
from src.stage1_opening.nodes import opening_arguments_stream
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

@app.get("/debate/{session_id}/opening/run")
def run_opening_arguments(session_id: str) -> StreamingResponse:
    """AI 에이전트 입론을 SSE로 스트리밍한다.

    에이전트 하나 완료될 때마다 DebateEntry를 SSE 이벤트로 전송한다.
    모든 에이전트 완료 시 {"type": "done"} 이벤트를 전송한다.
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

    ai_openings = [
        e for e in state["debate_history"]
        if e["phase"] == "opening" and e["speaker_id"] != "user"
    ]
    if ai_openings:
        raise HTTPException(status_code=400, detail="AI 입론이 이미 생성되었습니다.")

    def event_stream() -> Generator[str, None, None]:
        history = list(state["debate_history"])
        for entry in opening_arguments_stream(state):
            history.append(entry)
            payload = {
                "turn": entry["turn"],
                "speaker_id": entry["speaker_id"],
                "stance": entry["stance"],
                "phase": entry["phase"],
                "content": entry["content"],
                "target_id": entry["target_id"],
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # 세션 상태 업데이트
        history.sort(key=lambda e: e["turn"])
        _sessions[session_id] = DebateState(**{
            **state,
            "debate_history": history,
            "current_turn": len(state["speaking_order"]),
            "current_speaker_index": len(state["speaking_order"]),
        })
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
