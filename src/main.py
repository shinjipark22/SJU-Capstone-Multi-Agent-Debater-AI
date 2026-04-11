"""
main.py — FastAPI 애플리케이션 진입점

LangGraph 메인 토론 그래프 기반.
- POST /debate/init    → 세션 생성 + 그래프 시작 (AI 입론 후 사용자 대기)
- POST /debate/{id}/submit → interrupt에서 멈춘 그래프 재개 (사용자 입력)
- GET  /debate/{id}/state  → 현재 상태 조회
- GET  /topics             → 토픽 목록
- GET  /health             → 서버 상태
"""

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, field_validator

from src.graph.main_graph import build_debate_graph
from src.models import AgentInfo, DebateInitRequest, DebateInitResponse
from src.phase0.persona_factory import create_agents, AgentPersona
from src.state import AgentSnapshot, DebateState, build_initial_state, generate_session_id

app = FastAPI(
    title="Multi-Agent Debater AI",
    description="LangGraph 기반 멀티 에이전트 토론 시스템 API",
    version="2.0.0",
)

# ── LangGraph 메인 그래프 (싱글톤) ────────────────────────────────────────
debate_graph = build_debate_graph()


# ── 요청/응답 모델 ────────────────────────────────────────────────────────

class UserSubmitRequest(BaseModel):
    """사용자 입력 제출 (모든 단계 공용)."""
    content: str

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("입력 내용은 비어 있을 수 없습니다.")
        return v.strip()


class DebateStateResponse(BaseModel):
    """토론 상태 응답."""
    session_id: str
    phase: str
    current_turn: int
    is_finished: bool
    waiting_for: str  # 현재 interrupt에서 대기 중인 내용
    debate_history: list
    message: str


# ── 엔드포인트 ─────────────────────────────────────────────────────────────

@app.post("/debate/init", response_model=DebateStateResponse)
def initialize_debate(request: DebateInitRequest) -> DebateStateResponse:
    """토론 세션을 초기화하고 AI 입론을 생성한다.

    그래프가 ai_opening을 실행한 뒤 user_opening의 interrupt에서 멈춘다.
    """
    # 1. topics.json에서 토픽 조회
    topics_path = Path(__file__).resolve().parent.parent / "data" / "topics_20260323_processed.json"
    if not topics_path.exists():
        raise HTTPException(status_code=404, detail="topics 파일을 찾을 수 없습니다.")

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

    # 2. AI 에이전트 생성
    try:
        personas = create_agents(
            topic=topic_dict,
            debate_format=request.debate_format,
            user_stance=request.user_stance,
            agent_intensities=request.agent_intensities,
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 3. 초기 State 생성
    snapshots = [
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
    initial_state = build_initial_state(
        topic=topic_dict["title"],
        user_stance=request.user_stance,
        user_intensity=request.user_intensity,
        agents=snapshots,
        topic_id=request.topic,
    )

    # 4. 세션 ID 생성 + 그래프 실행 (ai_opening → user_opening interrupt에서 멈춤)
    session_id = generate_session_id()
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = debate_graph.invoke(dict(initial_state), config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"그래프 실행 오류: {e}")

    # 5. 현재 상태에서 interrupt 정보 추출
    graph_state = debate_graph.get_state(config)
    waiting_for = ""
    if graph_state.next:
        waiting_for = graph_state.next[0]  # 다음에 실행될 노드 = interrupt 대기 중인 노드

    phase = result.get("phase", "opening") if isinstance(result, dict) else "opening"
    history = result.get("debate_history", []) if isinstance(result, dict) else []

    return DebateStateResponse(
        session_id=session_id,
        phase=phase,
        current_turn=len(history),
        is_finished=False,
        waiting_for=waiting_for,
        debate_history=[dict(e) if isinstance(e, dict) else e for e in history],
        message=f"AI 입론 완료. '{waiting_for}' 단계에서 사용자 입력 대기 중.",
    )


@app.post("/debate/{session_id}/submit", response_model=DebateStateResponse)
def submit_user_input(session_id: str, request: UserSubmitRequest) -> DebateStateResponse:
    """interrupt에서 멈춘 그래프를 사용자 입력으로 재개한다.

    그래프가 다음 interrupt까지 자동으로 진행된 뒤 다시 멈춘다.
    """
    config = {"configurable": {"thread_id": session_id}}

    # 세션 존재 확인
    graph_state = debate_graph.get_state(config)
    if not graph_state or not graph_state.next:
        raise HTTPException(
            status_code=404,
            detail="세션을 찾을 수 없거나 이미 완료된 토론입니다.",
        )

    # resume
    try:
        result = debate_graph.invoke(
            Command(resume=request.content),
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"그래프 재개 오류: {e}")

    # 다음 interrupt 확인
    graph_state = debate_graph.get_state(config)
    waiting_for = ""
    is_finished = True
    if graph_state.next:
        waiting_for = graph_state.next[0]
        is_finished = False

    phase = result.get("phase", "") if isinstance(result, dict) else ""
    history = result.get("debate_history", []) if isinstance(result, dict) else []
    synthesis_draft = result.get("synthesis_draft", "") if isinstance(result, dict) else ""

    msg = "토론이 완료되었습니다." if is_finished else f"'{waiting_for}' 단계에서 사용자 입력 대기 중."

    return DebateStateResponse(
        session_id=session_id,
        phase=phase,
        current_turn=len(history),
        is_finished=is_finished,
        waiting_for=waiting_for,
        debate_history=[dict(e) if isinstance(e, dict) else e for e in history],
        message=msg,
    )


@app.get("/debate/{session_id}/state")
def get_debate_state(session_id: str):
    """현재 토론 세션 상태를 반환한다."""
    config = {"configurable": {"thread_id": session_id}}
    graph_state = debate_graph.get_state(config)

    if not graph_state or not graph_state.values:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    values = graph_state.values
    return {
        "session_id": session_id,
        "phase": values.get("phase", ""),
        "current_turn": values.get("current_turn", 0),
        "is_finished": values.get("is_finished", False),
        "waiting_for": graph_state.next[0] if graph_state.next else "",
        "debate_history": values.get("debate_history", []),
        "synthesis_draft": values.get("synthesis_draft", ""),
    }


@app.get("/topics")
def get_topics():
    """사용 가능한 토론 주제 목록을 반환한다."""
    topics_path = Path(__file__).resolve().parent.parent / "data" / "topics_20260323_processed.json"
    if not topics_path.exists():
        raise HTTPException(status_code=404, detail="topics 파일을 찾을 수 없습니다.")
    with topics_path.open(encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
def health_check():
    """서버 상태 확인."""
    return {"status": "ok", "version": "2.0.0", "graph": "langgraph"}
