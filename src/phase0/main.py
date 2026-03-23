"""
main.py — FastAPI 애플리케이션 진입점 (Phase 0)

사용자 입력을 받아 LangGraph 초기 상태를 생성하고,
세션 정보를 반환한다. 실제 토론 실행은 Phase 1에서 구현한다.
"""

from fastapi import FastAPI, HTTPException

from src.models import DebateInitRequest, DebateInitResponse, AgentInfo
from src.phase0.persona_factory import create_agents, AgentPersona
from src.state import (
    AgentSnapshot,
    build_initial_state,
    generate_session_id,
)

app = FastAPI(
    title="Multi-Agent Debater AI",
    description="LangGraph 기반 멀티 에이전트 토론 시스템 API",
    version="0.1.0",
)


@app.post("/debate/init", response_model=DebateInitResponse)
def initialize_debate(request: DebateInitRequest) -> DebateInitResponse:
    """토론 세션을 초기화한다.

    1. persona_factory로 AI 에이전트 생성
    2. LangGraph 초기 State 구성
    3. 세션 ID 발급 후 응답 반환

    Phase 1에서는 이 엔드포인트 이후 /debate/run 등을 추가한다.
    """
    # ── 1. AI 에이전트 생성 ──────────────────────────────────────────────────
    try:
        personas: list[AgentPersona] = create_agents(
            topic=request.topic,
            debate_format=request.debate_format,
            user_stance=request.user_stance,
            agent_intensities=request.agent_intensities,
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ── 2. State용 AgentSnapshot 변환 ────────────────────────────────────────
    snapshots: list[AgentSnapshot] = [
        AgentSnapshot(
            agent_id=p.agent_id,
            stance=p.stance,
            intensity=p.intensity,
            role_description=p.role_description,
            system_prompt=p.system_prompt,
        )
        for p in personas
    ]

    # ── 3. LangGraph 초기 State 생성 ─────────────────────────────────────────
    initial_state = build_initial_state(
        topic=request.topic,
        user_stance=request.user_stance,
        user_intensity=request.user_intensity,
        agents=snapshots,
    )

    # ── 4. 응답 구성 ─────────────────────────────────────────────────────────
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

    topics_path = Path(__file__).parent.parent / "data" / "topics.json"
    if not topics_path.exists():
        raise HTTPException(status_code=404, detail="topics.json 파일을 찾을 수 없습니다.")

    with topics_path.open(encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
def health_check():
    """서버 상태 확인."""
    return {"status": "ok", "phase": "0"}
