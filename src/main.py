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
                # 발화 + 실시간 분석 결과를 한 이벤트로 묶어서 전송 (frontend 가 매칭 부담 없도록)
                ev = await _run_judge_turn(judge, entry_dict)
                yield _sse_event("turn", {"entry": entry_dict, "analysis": ev})

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
                # 발화 + 실시간 분석 결과를 한 이벤트로 묶어서 전송 (frontend 가 매칭 부담 없도록)
                ev = await _run_judge_turn(judge, entry_dict) if judge is not None else None
                yield _sse_event("turn", {"entry": entry_dict, "analysis": ev})

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

from evaluation import analyze_user_before_after


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
    evidence_expansion: MetricScore
    knowledge_specificity: MetricScore
    evidence_validity: MetricScore
    reasoning_density: MetricScore
    perspective_diversity: MetricScore
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


def _build_phase_result(phase: dict) -> PhaseResult:
    s = phase["scores"]
    return PhaseResult(
        evidence_expansion=MetricScore(**{
            "label": s["evidence_expansion"]["label"],
            "score": s["evidence_expansion"]["score"],
            "reason": s["evidence_expansion"]["reason"],
        }),
        knowledge_specificity=MetricScore(**{
            "label": s["knowledge_specificity"]["label"],
            "score": s["knowledge_specificity"]["score"],
            "reason": s["knowledge_specificity"]["reason"],
        }),
        evidence_validity=MetricScore(**{
            "label": s["evidence_validity"]["label"],
            "score": s["evidence_validity"]["score"],
            "reason": s["evidence_validity"]["reason"],
        }),
        reasoning_density=MetricScore(**{
            "label": s["reasoning_density"]["label"],
            "score": s["reasoning_density"]["score"],
            "reason": s["reasoning_density"]["reason"],
        }),
        perspective_diversity=MetricScore(**{
            "label": s["perspective_diversity"]["label"],
            "score": s["perspective_diversity"]["score"],
            "reason": s["perspective_diversity"]["reason"],
        }),
        average_100=phase["average_100"],
        overall_summary=phase.get("overall_summary", ""),
    )


@app.post("/evaluation", response_model=EvaluateResponse)
async def evaluate_user_before_after(req: EvaluateRequest, topic_id: str):
    """
    토론 전·후 사용자 답변(찬·반 양쪽)을 5개 지표로 채점하고 변화량을 반환한다.

    - **topic_id**: query 파라미터 (예: `tech_001`). topics JSON 에서 title/pro/con 자동 조회.
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
    )
