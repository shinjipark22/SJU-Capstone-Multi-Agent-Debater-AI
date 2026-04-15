"""
main.py — FastAPI 애플리케이션 진입점

LangGraph 메인 토론 그래프 기반.
- POST /debate/init         → 세션 생성 + SSE 스트리밍 (AI 입론 실시간 전송)
- POST /debate/{id}/submit  → SSE 스트리밍으로 그래프 재개 (다음 interrupt까지)
- GET  /debate/{id}/state   → 현재 상태 조회
- GET  /topics              → 토픽 목록
- GET  /health              → 서버 상태
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, field_validator

from src.final_evaluator import build_final_report
from src.graph.main_graph import build_debate_graph
from src.live_analyzer import DebatrixJudge, project_frontend_event
from src.models import DebateInitRequest
from src.phase0.persona_factory import create_agents, AgentPersona
from src.state import AgentSnapshot, DebateState, build_initial_state, generate_session_id

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Agent Debater AI",
    description="LangGraph 기반 멀티 에이전트 토론 시스템 API (SSE 스트리밍)",
    version="2.0.0",
)

# ── LangGraph 메인 그래프 (싱글톤) ────────────────────────────────────────
debate_graph = build_debate_graph()

# ── 세션별 실시간 분석기 캐시 ──────────────────────────────────────────────
# 토론 세션 단위로 DebatrixJudge 인스턴스를 유지, 턴별 누적 분석 + 최종 리포트 생성을 위해.
# (is_finished=True 이후에도 유지 — 최종 리포트 생성 후 삭제)
_session_judges: Dict[str, DebatrixJudge] = {}
# 최종 리포트 캐시 — 같은 세션에서 반복 호출 시 재사용
_final_reports: Dict[str, dict] = {}


def _build_teams_from_state(initial_state: DebateState) -> dict:
    """initial_state에서 PRO/CON 팀 구성을 만든다 (judge init용)."""
    pro: list = []
    con: list = []
    user_stance = initial_state.get("user_stance", "PRO")
    for agent in initial_state.get("agents", []):
        aid = agent.get("agent_id") if isinstance(agent, dict) else agent.agent_id
        side = agent.get("stance") if isinstance(agent, dict) else agent.stance
        (pro if side == "PRO" else con).append(aid)
    # user 포함
    (pro if user_stance == "PRO" else con).append("user")
    return {"PRO": sorted(pro), "CON": sorted(con)}


async def _run_judge_turn(judge: DebatrixJudge, entry: dict) -> Optional[dict]:
    """judge.judge_turn을 async로 실행. 실패 시 None 반환 (debate는 계속)."""
    speaker = entry.get("speaker_id") or entry.get("speaker") or ""
    stance = entry.get("stance") or entry.get("side") or "PRO"
    phase = entry.get("phase") or ""
    content = entry.get("content") or entry.get("text") or ""
    target = entry.get("target_id")
    if target in (None, "None", ""):
        target = None

    def _call():
        try:
            analysis = judge.judge_turn(
                speaker_id=speaker,
                speaker_stance=stance,
                phase=phase,
                speech_content=content,
                target_id=target,
            )
            if analysis is None:
                return None  # 분석 대상 외 phase
            live = judge.memory.live_debate_snapshot()
            analysis_row = judge.memory.analysis_memory[-1]
            return project_frontend_event(analysis_row, live)
        except Exception as e:
            logger.warning("live_analyzer 실패: %s", e)
            return None

    return await asyncio.to_thread(_call)


# ── 요청 모델 ─────────────────────────────────────────────────────────────

class UserSubmitRequest(BaseModel):
    """사용자 입력 제출 (모든 단계 공용)."""
    content: str

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("입력 내용은 비어 있을 수 없습니다.")
        return v.strip()


# ── SSE 이벤트 헬퍼 ───────────────────────────────────────────────────────

def _sse_event(event_type: str, data: dict) -> str:
    """SSE 포맷 이벤트 문자열을 생성한다."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extract_new_entries(prev_history: list, cur_history: list) -> list:
    """이전 대비 새로 추가된 debate_history 엔트리를 추출한다."""
    prev_len = len(prev_history)
    return cur_history[prev_len:]


_WAITING_ALIAS = {
    # 내부 노드 분할이 API 계약에 노출되지 않도록 통합 이름으로 매핑
    "user_free_rebuttal_defense": "user_free_rebuttal",
    "user_free_rebuttal_attack": "user_free_rebuttal",
}


def _get_waiting_info(config: dict) -> tuple:
    """현재 그래프 상태에서 waiting_for, is_finished를 추출한다."""
    graph_state = debate_graph.get_state(config)
    if graph_state and graph_state.next:
        raw = graph_state.next[0]
        return _WAITING_ALIAS.get(raw, raw), False
    return "", True


# ── 초기화 + 토픽 조회 헬퍼 ───────────────────────────────────────────────

def _load_topic(topic_id: str) -> dict:
    """topics.json에서 토픽을 조회한다."""
    topics_path = Path(__file__).resolve().parent.parent / "data" / "topics_20260323_processed.json"
    if not topics_path.exists():
        raise HTTPException(status_code=404, detail="topics 파일을 찾을 수 없습니다.")
    with topics_path.open(encoding="utf-8") as f:
        topics_data = json.load(f)
    for category_topics in topics_data.get("categories", {}).values():
        for t in category_topics:
            if t["id"] == topic_id:
                return t
    raise HTTPException(status_code=404, detail=f"topic ID '{topic_id}'를 찾을 수 없습니다.")


def _create_initial_state(request: DebateInitRequest, topic_dict: dict) -> DebateState:
    """AI 에이전트 생성 + 초기 State 빌드."""
    try:
        personas = create_agents(
            topic=topic_dict,
            debate_format=request.debate_format,
            user_stance=request.user_stance,
            agent_intensities=request.agent_intensities,
        )
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    snapshots = [
        AgentSnapshot(
            agent_id=p.agent_id, stance=p.stance, intensity=p.intensity,
            role_description=p.role_description, system_prompt=p.system_prompt,
            focus_area=p.focus_area,
        )
        for p in personas
    ]
    return build_initial_state(
        topic=topic_dict["title"],
        user_stance=request.user_stance,
        user_intensity=request.user_intensity,
        agents=snapshots,
        topic_id=request.topic,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SSE 스트리밍 엔드포인트
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/debate/init")
async def initialize_debate(request: DebateInitRequest):
    """토론 세션 초기화 + AI 입론을 SSE로 실시간 스트리밍.

    각 AI 에이전트의 입론이 생성될 때마다 'entry' 이벤트로 전송.
    마지막에 'waiting' 이벤트로 사용자 입력 대기 알림.
    """
    topic_dict = _load_topic(request.topic)
    initial_state = _create_initial_state(request, topic_dict)
    session_id = generate_session_id()
    config = {"configurable": {"thread_id": session_id}}

    # 세션별 실시간 분석기 준비
    judge = DebatrixJudge(
        topic=topic_dict["title"],
        debate_format=request.debate_format,
        user_id="user",
        user_stance=request.user_stance,
        teams=_build_teams_from_state(initial_state),
    )
    _session_judges[session_id] = judge

    async def event_stream():
        # 세션 시작 이벤트
        yield _sse_event("session", {"session_id": session_id, "topic": topic_dict["title"]})

        prev_history = []

        # 그래프 스트리밍 실행
        for chunk in debate_graph.stream(dict(initial_state), config=config, stream_mode="values"):
            if not isinstance(chunk, dict):
                continue

            cur_history = chunk.get("debate_history", [])
            new_entries = _extract_new_entries(prev_history, cur_history)

            for entry in new_entries:
                entry_dict = dict(entry) if isinstance(entry, dict) else entry
                yield _sse_event("entry", entry_dict)
                # 실시간 분석 — 턴 발행 직후 점수 emit
                ev = await _run_judge_turn(judge, entry_dict)
                if ev is not None:
                    yield _sse_event("analysis", ev)

            prev_history = list(cur_history)

        # interrupt 대기 정보
        waiting_for, is_finished = _get_waiting_info(config)
        yield _sse_event("waiting", {
            "session_id": session_id,
            "waiting_for": waiting_for,
            "is_finished": is_finished,
            "phase": prev_history[-1].get("phase", "opening") if prev_history else "opening",
            "total_entries": len(prev_history),
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/debate/{session_id}/submit")
async def submit_user_input(session_id: str, request: UserSubmitRequest):
    """사용자 입력으로 그래프 재개 + 다음 interrupt까지 SSE 스트리밍.

    AI 에이전트의 발언이 생성될 때마다 'entry' 이벤트로 전송.
    """
    config = {"configurable": {"thread_id": session_id}}

    # 세션 확인
    graph_state = debate_graph.get_state(config)
    if not graph_state or not graph_state.next:
        raise HTTPException(
            status_code=404,
            detail="세션을 찾을 수 없거나 이미 완료된 토론입니다.",
        )

    judge = _session_judges.get(session_id)

    async def event_stream():
        # 현재 히스토리 길이 기록
        current_values = graph_state.values or {}
        prev_history = list(current_values.get("debate_history", []))

        # 그래프 resume 스트리밍
        for chunk in debate_graph.stream(
            Command(resume=request.content),
            config=config,
            stream_mode="values",
        ):
            if not isinstance(chunk, dict):
                continue

            cur_history = chunk.get("debate_history", [])
            new_entries = _extract_new_entries(prev_history, cur_history)

            for entry in new_entries:
                entry_dict = dict(entry) if isinstance(entry, dict) else entry
                yield _sse_event("entry", entry_dict)
                # 실시간 분석 — 턴 발행 직후 점수 emit
                if judge is not None:
                    ev = await _run_judge_turn(judge, entry_dict)
                    if ev is not None:
                        yield _sse_event("analysis", ev)

            prev_history = list(cur_history)

        # interrupt 대기 정보
        waiting_for, is_finished = _get_waiting_info(config)

        synthesis_draft = ""
        if is_finished:
            final_state = debate_graph.get_state(config)
            if final_state and final_state.values:
                synthesis_draft = final_state.values.get("synthesis_draft", "")
            # 완료된 세션의 judge는 최종 리포트 생성을 위해 유지
            # (별도 엔드포인트 GET /debate/{id}/final-report에서 소비 후 정리)

        yield _sse_event("waiting", {
            "session_id": session_id,
            "waiting_for": waiting_for,
            "is_finished": is_finished,
            "phase": prev_history[-1].get("phase", "") if prev_history else "",
            "total_entries": len(prev_history),
            "synthesis_draft": synthesis_draft,
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════════════════
# REST 엔드포인트 (상태 조회, 토픽, 헬스체크)
# ═══════════════════════════════════════════════════════════════════════════

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


@app.get("/debate/{session_id}/final-report")
def get_final_report(session_id: str, refresh: bool = False):
    """종료된 토론 세션의 최종 평가 리포트.

    - 토론이 `is_finished=True` 이후에 호출 권장.
    - `refresh=true` 쿼리 파라미터 시 캐시 무시하고 재생성.
    - judge 인스턴스(`_session_judges`)와 LangGraph state를 사용.
    """
    if not refresh and session_id in _final_reports:
        return _final_reports[session_id]

    judge = _session_judges.get(session_id)
    if judge is None:
        raise HTTPException(
            status_code=404,
            detail="세션의 분석 데이터를 찾을 수 없습니다 (judge 없음).",
        )

    config = {"configurable": {"thread_id": session_id}}
    graph_state = debate_graph.get_state(config)
    if not graph_state or not graph_state.values:
        raise HTTPException(status_code=404, detail="세션 상태를 찾을 수 없습니다.")

    values = graph_state.values
    analysis_memory = list(judge.memory.analysis_memory)
    speech_memory = list(judge.memory.speech_memory)
    live_debate = judge.memory.live_debate_snapshot()

    if not analysis_memory:
        raise HTTPException(
            status_code=400,
            detail="분석 메모리가 비어 있습니다. 토론이 진행되지 않았거나 평가 대상 phase가 없습니다.",
        )

    try:
        report = build_final_report(
            topic=values.get("topic", ""),
            debate_format=values.get("debate_format") or judge.memory.debate_format,
            user_stance=values.get("user_stance", "PRO"),
            analysis_memory=analysis_memory,
            speech_memory=speech_memory,
            live_debate=live_debate,
        )
    except Exception as e:
        logger.exception("[final-report] 빌드 실패: %s", e)
        raise HTTPException(status_code=500, detail=f"final report 생성 실패: {e}")

    payload = report.model_dump()
    _final_reports[session_id] = payload
    return payload


@app.delete("/debate/{session_id}")
def delete_session_cache(session_id: str):
    """세션 관련 메모리 캐시 정리 (judge, final report).

    토론 종료 + 리포트 조회 후 리소스 회수용.
    """
    removed = {
        "judge": _session_judges.pop(session_id, None) is not None,
        "final_report": _final_reports.pop(session_id, None) is not None,
    }
    return {"session_id": session_id, "removed": removed}


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
    return {"status": "ok", "version": "2.0.0", "graph": "langgraph", "streaming": "SSE"}
