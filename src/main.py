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
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, field_validator

from src.debate_assistant import (
    GuideContext,
    HistoryEntry,
    build_guide_message,
    get_user_slot_focus_area,
)
from src.final_evaluator import build_final_report
from src.graph.main_graph import build_debate_graph
from src.live_analyzer import DebatrixJudge, project_frontend_event
from src.models import DebateInitRequest
from src.phase0.persona_factory import create_agents, AgentPersona
from src.state import AgentSnapshot, DebateState, build_initial_state, generate_session_id
from src.storage import (
    create_session as store_create_session,
    get_session_by_graph_id as store_get_session_by_graph_id,
    load_turn_analyses as store_load_turn_analyses,
    save_turn_analysis as store_save_turn_analysis,
    export_csv as store_export_csv,
    export_survey_csv as store_export_survey_csv,
    get_session as store_get_session,
    get_survey as store_get_survey,
    list_sessions as store_list_sessions,
    save_evaluation as store_save_evaluation,
    save_survey as store_save_survey,
    save_synthesis as store_save_synthesis,
)
from src.survey import PHASES as SURVEY_PHASES, load_schema as load_survey_schema

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Agent Debater AI",
    description="LangGraph 기반 멀티 에이전트 토론 시스템 API (SSE 스트리밍)",
    version="2.0.0",
)

# 프론트엔드(web/, 로컬 dev 서버 및 배포된 정적 호스팅) CORS 허용.
# 운영 배포 시 CORS_ALLOW_ORIGINS 환경변수로 허용 도메인을 좁힐 것.
import os

_cors_origins = os.environ.get("CORS_ALLOW_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _cors_origins == "*" else _cors_origins.split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LangGraph 메인 그래프 (싱글톤) ────────────────────────────────────────
debate_graph = build_debate_graph()

# ── 세션별 실시간 분석기 캐시 ──────────────────────────────────────────────
# 토론 세션 단위로 DebatrixJudge 인스턴스를 유지, 턴별 누적 분석 + 최종 리포트 생성을 위해.
# (is_finished=True 이후에도 유지 — 최종 리포트 생성 후 삭제)
_session_judges: Dict[str, DebatrixJudge] = {}
# 최종 리포트 캐시 — 같은 세션에서 반복 호출 시 재사용
_final_reports: Dict[str, dict] = {}

_STREAM_SENTINEL = object()


async def _astream_in_thread(sync_stream_factory: Callable[[], Iterator[Any]]):
    """동기 debate_graph.stream()을 전용 스레드 하나에서 그대로 실행하고,
    결과를 큐를 통해 이벤트 루프를 막지 않으면서 비동기로 넘겨준다.

    astream()으로 바꾸면 LangGraph가 동기 노드를 요청마다 다른 스레드풀 스레드에서
    실행하게 되어, interrupt()가 의존하는 contextvar가 끊기고 깨진다
    (RuntimeError: Called get_config outside of a runnable context).
    이 방식은 한 세션의 그래프 실행 전체를 하나의 스레드에 그대로 묶어두므로
    (원래의 단일 스레드 동기 실행 모델과 동일) interrupt()는 안전하게 유지되면서,
    다른 세션의 요청은 이벤트 루프가 계속 처리할 수 있다.
    """
    q: "queue.Queue" = queue.Queue()

    def _worker() -> None:
        try:
            for item in sync_stream_factory():
                q.put((True, item))
        except Exception as e:  # noqa: BLE001 - 그대로 상위로 전달
            q.put((False, e))
        finally:
            q.put(_STREAM_SENTINEL)

    threading.Thread(target=_worker, daemon=True).start()

    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _STREAM_SENTINEL:
            break
        ok, payload = item
        if not ok:
            raise payload
        yield payload


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


async def _run_judge_turn(
    judge: DebatrixJudge,
    entry: dict,
    graph_session_id: Optional[str] = None,
) -> Optional[dict]:
    """judge.judge_turn을 async로 실행. 실패 시 None 반환 (debate는 계속).

    분석 결과는 곧바로 저장소에도 남긴다. judge 인스턴스는 프로세스 메모리에만
    있어서 서버가 재시작되면 최종 리포트를 만들 수 없기 때문.
    """
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

            if graph_session_id:
                speech_rows = judge.memory.speech_memory
                turn_index = analysis_row.get("turn_index")
                speech = next(
                    (s for s in reversed(speech_rows) if s.get("turn_index") == turn_index), None
                )
                try:
                    store_save_turn_analysis(
                        graph_session_id, turn_index, analysis_row, speech, live
                    )
                except Exception:
                    logger.exception("[storage] 턴 분석 저장 실패 (session=%s)", graph_session_id)

            return project_frontend_event(analysis_row, live)
        except Exception as e:
            logger.warning("live_analyzer 실패: %s", e)
            return None

    return await asyncio.to_thread(_call)


# ── 요청 모델 ─────────────────────────────────────────────────────────────

class UserSubmitRequest(BaseModel):
    """사용자 입력 제출 (모든 단계 공용)."""
    content: str
    target_id: Optional[str] = None  # 자유논박: 사용자가 선택한 상대 에이전트 ID

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
    "user_select_opponent_node": "user_select_opponent",
    "user_free_rebuttal_defense": "user_free_rebuttal",
    "user_free_rebuttal_attack": "user_free_rebuttal",
}


def _get_waiting_info(config: dict) -> tuple:
    """현재 그래프 상태에서 (waiting_for, is_finished, waiting_detail)을 추출한다.

    waiting_detail 은 alias 이전의 노드 이름 그대로다. 자유논박처럼 한 단계가
    방어(user_free_rebuttal_defense)·공격(user_free_rebuttal_attack) 차례로 나뉘는 경우
    프론트가 무엇을 써야 하는지 안내하는 데 쓴다.
    """
    graph_state = debate_graph.get_state(config)
    if graph_state and graph_state.next:
        raw = graph_state.next[0]
        return _WAITING_ALIAS.get(raw, raw), False, raw
    return "", True, ""


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


def _load_topic_for_evaluation(topic_id: str) -> dict:
    """평가 API는 과거 프론트 세션의 카테고리 ID도 방어적으로 처리한다."""
    try:
        return _load_topic(topic_id)
    except HTTPException as e:
        if e.status_code != 404:
            raise
        logger.warning("평가 topic_id를 topics 파일에서 찾지 못해 기본 라벨로 진행합니다: %s", topic_id)
        return {
            "id": topic_id,
            "title": topic_id,
            "pro": "찬성",
            "con": "반대",
        }


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
        mode=request.mode,
    )


# ═══════════════════════════════════════════════════════════════════════════
# SSE 스트리밍 엔드포인트
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/debate/init")
async def initialize_debate(request: DebateInitRequest):
    """토론 세션 초기화 + AI 입론을 SSE로 실시간 스트리밍.

    각 AI 에이전트의 입론이 생성될 때마다 'turn' 이벤트로 (발화 + 실시간 분석) 묶어서 전송.
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

    # 데이터 수집 — 저장 실패가 토론 진행을 막지 않도록 기록만 남기고 계속한다.
    try:
        record_id = store_create_session(
            topic=request.topic,
            user_stance=request.user_stance,
            user_intensity=request.user_intensity,
            debate_format=request.debate_format,
            nickname=request.nickname,
            email=request.email,
            graph_session_id=session_id,
            mode=request.mode,
        )
    except Exception:
        logger.exception("[storage] 세션 기록 실패 (session_id=%s)", session_id)
        record_id = None

    async def event_stream():
        # 세션 시작 이벤트
        yield _sse_event("session", {
            "session_id": session_id,
            "record_id": record_id,  # /evaluation?session_id= 에 그대로 넘기면 채점 결과가 이어 저장됨
            "topic": topic_dict["title"],
            "debate_format": request.debate_format,
        })

        prev_history = []

        # 그래프 스트리밍 실행 (전용 스레드에서, 이벤트 루프는 막지 않음)
        async for chunk in _astream_in_thread(
            lambda: debate_graph.stream(dict(initial_state), config=config, stream_mode="values")
        ):
            if not isinstance(chunk, dict):
                continue

            cur_history = chunk.get("debate_history", [])
            new_entries = _extract_new_entries(prev_history, cur_history)

            for entry in new_entries:
                entry_dict = dict(entry) if isinstance(entry, dict) else entry
                # 발화 + 실시간 분석 결과를 한 이벤트로 묶어서 전송 (frontend 가 매칭 부담 없도록)
                ev = await _run_judge_turn(judge, entry_dict, session_id)
                yield _sse_event("turn", {"entry": entry_dict, "analysis": ev})

            prev_history = list(cur_history)

        # interrupt 대기 정보
        waiting_for, is_finished, waiting_detail = _get_waiting_info(config)
        yield _sse_event("waiting", {
            "session_id": session_id,
            "waiting_for": waiting_for,
            "waiting_detail": waiting_detail,
            "is_finished": is_finished,
            "phase": prev_history[-1].get("phase", "opening") if prev_history else "opening",
            "total_entries": len(prev_history),
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/debate/{session_id}/submit")
async def submit_user_input(session_id: str, request: UserSubmitRequest):
    """사용자 입력으로 그래프 재개 + 다음 interrupt까지 SSE 스트리밍.

    AI 에이전트의 발언이 생성될 때마다 'turn' 이벤트로 (발화 + 실시간 분석) 묶어서 전송.
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

    # 자유논박에서 사용자가 상대를 선택한 경우 state에 반영 (resume 전에 수행)
    if request.target_id:
        debate_graph.update_state(config, {"selected_opponent_id": request.target_id})

    async def event_stream():
        # 현재 히스토리 길이 기록
        current_values = graph_state.values or {}
        prev_history = list(current_values.get("debate_history", []))

        # 그래프 resume 스트리밍 (전용 스레드에서, 이벤트 루프는 막지 않음)
        async for chunk in _astream_in_thread(
            lambda: debate_graph.stream(
                Command(resume=request.content),
                config=config,
                stream_mode="values",
            )
        ):
            if not isinstance(chunk, dict):
                continue

            cur_history = chunk.get("debate_history", [])
            new_entries = _extract_new_entries(prev_history, cur_history)

            for entry in new_entries:
                entry_dict = dict(entry) if isinstance(entry, dict) else entry
                # 발화 + 실시간 분석 결과를 한 이벤트로 묶어서 전송 (frontend 가 매칭 부담 없도록)
                ev = await _run_judge_turn(judge, entry_dict, session_id) if judge is not None else None
                yield _sse_event("turn", {"entry": entry_dict, "analysis": ev})

            prev_history = list(cur_history)

        # interrupt 대기 정보
        waiting_for, is_finished, waiting_detail = _get_waiting_info(config)

        synthesis_draft = ""
        if is_finished:
            final_state = debate_graph.get_state(config)
            if final_state and final_state.values:
                synthesis_draft = final_state.values.get("synthesis_draft", "")
            # 구성적 논쟁의 결과물이라 리포트·연구 데이터 양쪽에 필요하다.
            if synthesis_draft:
                try:
                    store_save_synthesis(session_id, synthesis_draft)
                except Exception:
                    logger.exception("[storage] 최적해 저장 실패 (session=%s)", session_id)
            # 완료된 세션의 judge는 최종 리포트 생성을 위해 유지
            # (별도 엔드포인트 GET /debate/{id}/final-report에서 소비 후 정리)

        yield _sse_event("waiting", {
            "session_id": session_id,
            "waiting_for": waiting_for,
            "waiting_detail": waiting_detail,
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
        "waiting_for": _WAITING_ALIAS.get(graph_state.next[0], graph_state.next[0]) if graph_state.next else "",
        "waiting_detail": graph_state.next[0] if graph_state.next else "",
        "debate_history": values.get("debate_history", []),
        "synthesis_draft": values.get("synthesis_draft", ""),
    }


@app.get("/debate/{session_id}/final-report")
def get_final_report(session_id: str, refresh: bool = False):
    """종료된 토론 세션의 최종 평가 리포트.

    - 토론이 `is_finished=True` 이후에 호출 권장.
    - `refresh=true` 쿼리 파라미터 시 캐시 무시하고 재생성.
    - judge 인스턴스(`_session_judges`)와 LangGraph state를 우선 사용하고,
      서버 재시작 등으로 메모리가 비었으면 저장소(turn_analyses)에서 복구한다.
    """
    if not refresh and session_id in _final_reports:
        return _final_reports[session_id]

    judge = _session_judges.get(session_id)
    graph_state = debate_graph.get_state({"configurable": {"thread_id": session_id}})
    values = graph_state.values if graph_state and graph_state.values else {}

    if judge is not None:
        analysis_memory = list(judge.memory.analysis_memory)
        speech_memory = list(judge.memory.speech_memory)
        live_debate = judge.memory.live_debate_snapshot()
        topic = values.get("topic", "")
        debate_format = values.get("debate_format") or judge.memory.debate_format
        user_stance = values.get("user_stance", "PRO")
    else:
        # 메모리 유실 복구 경로 — 턴 분석은 저장소에, 세션 메타는 debate_sessions 에 있다.
        stored = store_load_turn_analyses(session_id)
        analysis_memory = stored["analysis_memory"]
        speech_memory = stored["speech_memory"]
        live_debate = stored["live_debate"]

        record = store_get_session_by_graph_id(session_id) or {}
        topic = values.get("topic") or _load_topic_for_evaluation(
            record.get("topic", "")
        ).get("title", "")
        debate_format = values.get("debate_format") or record.get("debate_format") or ""
        user_stance = values.get("user_stance") or record.get("user_stance") or "PRO"

    if not analysis_memory:
        raise HTTPException(
            status_code=404,
            detail="세션의 분석 데이터를 찾을 수 없습니다. 토론이 진행되지 않았거나 기록이 남지 않은 세션입니다.",
        )

    try:
        report = build_final_report(
            topic=topic,
            debate_format=debate_format,
            user_stance=user_stance,
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


# ═══════════════════════════════════════════════════════════════════════════
# 어시스턴트 안내문 — 단계 시작 시 사용자에게 보여줄 친근체 가이드 생성
# ═══════════════════════════════════════════════════════════════════════════

_ASSISTANT_PHASES = {"opening", "chained_rebuttal", "free_rebuttal", "role_reversal", "synthesis"}


def _derive_opponent_speech(
    history: list,
    phase: str,
    user_stance: str,
    opponent_id: Optional[str] = None,
) -> str:
    """현재 단계에서 사용자가 받아쳐야 할 상대 발언을 history 에서 도출.

    우선순위:
      1) opponent_id 지정 시 → 해당 agent 의 가장 최근 발언
      2) 현재 phase 의 마지막 user 가 아닌 발언
      3) 그것도 없으면 history 전체에서 마지막 user 가 아닌 발언

    역할반전(role_reversal)에서는 사용자가 반대 진영을 옹호하는 발언을 준비하므로
    "받아칠 상대" 의 의미가 약하지만, 사전 검색 자료의 맥락 토대로 가장 최근
    AI 발언을 그대로 전달한다.
    """
    if opponent_id:
        for entry in reversed(history):
            if entry.get("speaker_id") == opponent_id:
                return entry.get("content", "")

    for entry in reversed(history):
        if entry.get("speaker_id") == "user":
            continue
        if entry.get("phase") == phase:
            return entry.get("content", "")

    for entry in reversed(history):
        if entry.get("speaker_id") != "user":
            return entry.get("content", "")
    return ""


def _build_assistant_ctx(
    state: dict,
    topic_dict: dict,
    phase: str,
    opponent_id: Optional[str],
) -> GuideContext:
    """LangGraph state 를 어시스턴트용 GuideContext 로 변환."""
    history = state.get("debate_history", [])
    agents = state.get("agents", [])
    user_stance = state.get("user_stance", "PRO")

    # 사용자 진영 AI 가 이미 쓰는 focus_area 는 제외해 사용자 슬롯에 다른 자료 노출
    excluded = [
        (a.get("focus_area") if isinstance(a, dict) else getattr(a, "focus_area", ""))
        for a in agents
        if (a.get("stance") if isinstance(a, dict) else getattr(a, "stance", "")) == user_stance
    ]
    excluded = [f for f in excluded if f]
    user_focus_area = get_user_slot_focus_area(
        state.get("topic_id", ""), user_stance, excluded_focuses=excluded
    )

    opponent_speech = _derive_opponent_speech(history, phase, user_stance, opponent_id)

    hist_entries = [
        HistoryEntry(
            speaker_id=e.get("speaker_id", ""),
            stance=e.get("stance", ""),
            phase=e.get("phase", ""),
            content=e.get("content", ""),
        )
        for e in history
    ]

    return GuideContext(
        topic=state.get("topic", ""),
        user_stance=user_stance,
        pro_claim=topic_dict.get("pro", "찬성"),
        con_claim=topic_dict.get("con", "반대"),
        topic_id=state.get("topic_id", ""),
        user_focus_area=user_focus_area,
        assistant_name="비비드",
        opponent_speech=opponent_speech,
        history=hist_entries,
        links=[],
        cache_variant_idx=state.get("cache_variant_idx"),
    )


@app.get("/debate/{session_id}/assistant/{phase}")
async def get_assistant_guide(
    session_id: str,
    phase: str,
    opponent_id: Optional[str] = None,
):
    """단계 시작 시 사용자에게 노출할 어시스턴트 안내문 생성.

    - `phase`: opening | chained_rebuttal | free_rebuttal | role_reversal | synthesis
    - `opponent_id` (query): 자유논박 등에서 사용자가 받아칠 상대 agent_id 지정 시
      해당 agent 의 가장 최근 발언을 opponent_speech 로 사용.
    """
    if phase not in _ASSISTANT_PHASES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 phase '{phase}'. 지원: {sorted(_ASSISTANT_PHASES)}",
        )

    config = {"configurable": {"thread_id": session_id}}
    graph_state = debate_graph.get_state(config)
    if not graph_state or not graph_state.values:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    state = graph_state.values

    # debate 모드 가드 — 자유논박까지만 진행하므로 역할반전·종합 단계 안내 거부
    if state.get("mode") == "debate" and phase in {"role_reversal", "synthesis"}:
        raise HTTPException(
            status_code=400,
            detail=f"phase '{phase}' 는 'debate' 모드에서 지원되지 않습니다. constructive 모드에서만 사용 가능합니다.",
        )
    topic_id = state.get("topic_id", "")
    try:
        topic_dict = _load_topic(topic_id)
    except HTTPException:
        topic_dict = {"id": topic_id, "title": state.get("topic", ""), "pro": "찬성", "con": "반대"}

    ctx = _build_assistant_ctx(state, topic_dict, phase, opponent_id)

    try:
        text = await asyncio.to_thread(build_guide_message, phase, ctx)
    except Exception as e:
        logger.exception("[assistant] %s 생성 실패", phase)
        raise HTTPException(status_code=500, detail=f"어시스턴트 안내문 생성 실패: {e}")

    return {"session_id": session_id, "phase": phase, "text": text}


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


# ── 토론 전후 사용자 평가 ──────────────────────────────────────────────────
# evaluation/ 패키지의 analyze_user_before_after() 를 vLLM 인스턴스로 호출.
# 5가지 지표(근거 확장성/지식 구체성/근거 타당성/논리 추론 밀도/관점 다각성)를
# 토론 전·후 × 찬·반 4쌍에 대해 채점하고, 100점 환산 + 변화량(delta) 까지 반환.

from src.evaluation import analyze_user_before_after


class EvaluateRequest(BaseModel):
    """토론 전·후 사용자 답변 (찬성/반대 측 모두 입력)."""
    pre_pro: str
    pre_con: str
    post_pro: str
    post_con: str


class MetricScore(BaseModel):
    label: str
    score: int
    reason: str


class PhaseResult(BaseModel):
    """Wachsmuth et al. (2017) taxonomy 의 5개 하위 차원 점수.

    - 논리(Cogency): local_acceptability / local_relevance / local_sufficiency
    - 수사(Effectiveness): clarity / appropriateness
    """
    local_acceptability: MetricScore
    local_relevance: MetricScore
    local_sufficiency: MetricScore
    clarity: MetricScore
    appropriateness: MetricScore
    average_100: float
    overall_summary: str


class SideResult(BaseModel):
    pre: PhaseResult
    post: PhaseResult
    delta_100: float  # post - pre (양수 = 토론 후 향상)


class EvaluateResponse(BaseModel):
    topic_id: str
    topic: str
    pro_label: str
    con_label: str
    pro: SideResult
    con: SideResult
    record_saved: bool = False  # session_id 로 지정한 수집 레코드에 저장됐는지


def _build_phase_result(phase: dict) -> PhaseResult:
    s = phase["scores"]
    return PhaseResult(
        local_acceptability=MetricScore(**{
            "label": s["local_acceptability"]["label"],
            "score": s["local_acceptability"]["score"],
            "reason": s["local_acceptability"]["reason"],
        }),
        local_relevance=MetricScore(**{
            "label": s["local_relevance"]["label"],
            "score": s["local_relevance"]["score"],
            "reason": s["local_relevance"]["reason"],
        }),
        local_sufficiency=MetricScore(**{
            "label": s["local_sufficiency"]["label"],
            "score": s["local_sufficiency"]["score"],
            "reason": s["local_sufficiency"]["reason"],
        }),
        clarity=MetricScore(**{
            "label": s["clarity"]["label"],
            "score": s["clarity"]["score"],
            "reason": s["clarity"]["reason"],
        }),
        appropriateness=MetricScore(**{
            "label": s["appropriateness"]["label"],
            "score": s["appropriateness"]["score"],
            "reason": s["appropriateness"]["reason"],
        }),
        average_100=phase["average_100"],
        overall_summary=phase.get("overall_summary", ""),
    )


@app.post("/evaluation", response_model=EvaluateResponse)
async def evaluate_user_before_after(
    req: EvaluateRequest,
    topic_id: str,
    session_id: Optional[int] = None,
):
    """
    토론 전·후 사용자 답변(찬·반 양쪽)을 5개 지표로 채점하고 변화량을 반환한다.

    - **topic_id**: query 파라미터 (예: `tech_001`). topics JSON 에서 title/pro/con 자동 조회.
    - **session_id**: query 파라미터 (선택). `/debate/init` 의 session 이벤트가 준 `record_id`.
      주면 답변 원문과 채점 결과가 수집 레코드에 저장되고 status 가 COMPLETED 로 바뀐다.
    - 요청 body: pre_pro / pre_con / post_pro / post_con 각각 문자열.
    - 응답: pro/con 각 진영의 pre/post 점수(5개 지표 + 100점 환산 + 요약) + delta_100.
    """
    topic_data = _load_topic_for_evaluation(topic_id)

    try:
        # vLLM 호출이 무거우므로 thread pool 에서 실행 (이벤트 루프 차단 방지)
        result = await asyncio.to_thread(
            analyze_user_before_after,
            req.pre_pro, req.pre_con, req.post_pro, req.post_con,
            topic=topic_data["title"],
        )
    except Exception as e:
        logger.exception("평가 실패")
        raise HTTPException(status_code=500, detail=f"평가 중 오류: {e}")

    record_saved = False
    if session_id is not None:
        try:
            record_saved = store_save_evaluation(
                session_id, req.pre_pro, req.pre_con, req.post_pro, req.post_con, result
            )
        except Exception:
            logger.exception("[storage] 평가 결과 저장 실패 (session_id=%s)", session_id)
        if not record_saved:
            logger.warning("[storage] session_id=%s 레코드를 찾지 못했습니다.", session_id)

    return EvaluateResponse(
        topic_id=topic_id,
        topic=topic_data["title"],
        pro_label=topic_data.get("pro", "찬성"),
        con_label=topic_data.get("con", "반대"),
        pro=SideResult(
            pre=_build_phase_result(result["pro"]["pre"]),
            post=_build_phase_result(result["pro"]["post"]),
            delta_100=result["pro"]["delta_100"],
        ),
        con=SideResult(
            pre=_build_phase_result(result["con"]["pre"]),
            post=_build_phase_result(result["con"]["post"]),
            delta_100=result["con"]["delta_100"],
        ),
        record_saved=record_saved,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 수집된 세션 데이터 조회 / 내보내기
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/sessions")
def list_collected_sessions(limit: int = 200, offset: int = 0):
    """수집된 토론 세션 목록 (최신순). CSV 와 동일한 필드 + 내부 식별자."""
    return {"sessions": store_list_sessions(limit=limit, offset=offset)}


@app.get("/sessions/export.csv")
def export_collected_sessions():
    """수집 데이터를 data-*.csv 와 동일한 컬럼·순서의 CSV 로 내보낸다."""
    filename = f"sessions-{int(time.time() * 1000)}.csv"
    return Response(
        content=store_export_csv(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/sessions/{session_id}")
def get_collected_session(session_id: int):
    record = store_get_session(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="세션 레코드를 찾을 수 없습니다.")
    return record


# ═══════════════════════════════════════════════════════════════════════════
# 연구 설문 (구글폼 4종을 웹에서 직접 수집)
# ═══════════════════════════════════════════════════════════════════════════

class SurveySubmitRequest(BaseModel):
    """설문 응답. answers 는 {문항 key: 응답} — 스키마의 key 를 그대로 쓴다."""
    phase: Literal["pre", "post"]
    answers: Dict[str, Any]


@app.get("/survey/schema/{phase}")
def get_survey_schema(
    phase: str,
    mode: Literal["debate", "constructive"] = "debate",
    topic_id: Optional[str] = None,
):
    """설문 문항 스키마. 주제·모드 의존 문구는 치환해서 내려준다.

    - **phase**: `pre` | `post`
    - **mode**: `debate`(토론) | `constructive`(구성적 논쟁) — 사후 섹션 제목에 반영
    - **topic_id**: 주면 해당 주제 문장을 입장 문항에 넣는다
    """
    if phase not in SURVEY_PHASES:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 phase '{phase}'. 지원: {list(SURVEY_PHASES)}")

    topic_title = ""
    if topic_id:
        topic_title = _load_topic_for_evaluation(topic_id).get("title", "")

    return {"phase": phase, "mode": mode, "topic": topic_title,
            "sections": load_survey_schema(phase, mode, topic_title)}


@app.post("/sessions/{session_id}/survey")
def submit_survey(session_id: int, request: SurveySubmitRequest):
    """토론 전·후 설문 응답 저장. 같은 단계를 다시 제출하면 덮어쓴다.

    세션 행이 없더라도(기록 유실 등) 응답 자체는 잃지 않도록 저장하고,
    `session_missing` 으로 알린다.
    """
    record = store_get_session(session_id) or {}
    session_exists = store_save_survey(
        session_id, request.phase, request.answers, mode=record.get("mode")
    )
    if not session_exists:
        logger.warning("[survey] session_id=%s 세션 행 없이 응답만 저장했습니다.", session_id)

    return {
        "session_id": session_id,
        "phase": request.phase,
        "saved": True,
        "session_missing": not session_exists,
    }


@app.get("/surveys/export.csv")
def export_surveys():
    """설문 응답을 세션당 한 행(pre_*, post_* 컬럼)으로 펼친 CSV."""
    filename = f"surveys-{int(time.time() * 1000)}.csv"
    return Response(
        content=store_export_survey_csv(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/sessions/{session_id}/survey/{phase}")
def get_submitted_survey(session_id: int, phase: str):
    record = store_get_survey(session_id, phase)
    if record is None:
        raise HTTPException(status_code=404, detail="설문 응답을 찾을 수 없습니다.")
    return record
