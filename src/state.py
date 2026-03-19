"""
state.py — LangGraph State 설계 및 초기화 유틸리티 (Phase 0)

토론 워크플로우 전체에서 공유되는 상태(State)를 정의한다.
LangGraph의 StateGraph는 이 TypedDict를 노드 간 데이터 컨테이너로 사용한다.

Phase 1에서 노드(opening / rebuttal / synthesis)를 추가할 때
이 State를 그대로 확장하여 사용한다.
"""

import uuid
from typing import Annotated, List, Literal, Optional
from typing_extensions import TypedDict

# LangGraph reducer: 리스트 필드에 append-only 병합을 적용할 때 사용
from langgraph.graph import add_messages  # noqa: F401 — Phase 1에서 메시지 누적에 활용


# ── 토론 단계 타입 ────────────────────────────────────────────────────────────
DebatePhase = Literal["opening", "rebuttal", "synthesis"]


# ── 히스토리 엔트리 타입 ──────────────────────────────────────────────────────
class DebateEntry(TypedDict):
    """단일 발언 기록.

    Attributes:
        turn: 발언 순서 번호 (0부터 시작)
        speaker_id: 발언자 ID ("user" 또는 "agent_N")
        stance: 발언자 진영
        phase: 발언 시점의 토론 단계
        content: 발언 내용
    """

    turn: int
    speaker_id: str
    stance: Literal["PRO", "CON"]
    phase: DebatePhase
    content: str


# ── 에이전트 스냅샷 타입 (State 저장용) ──────────────────────────────────────
class AgentSnapshot(TypedDict):
    """State 내부에 저장되는 에이전트 요약.

    AgentPersona 전체를 넣지 않고 필요한 필드만 보존하여
    직렬화(JSON) 친화적 구조를 유지한다.
    """

    agent_id: str
    stance: Literal["PRO", "CON"]
    intensity: int
    role_description: str
    system_prompt: str


# ── 핵심 State TypedDict ──────────────────────────────────────────────────────
class DebateState(TypedDict):
    """LangGraph StateGraph에서 사용하는 토론 공유 상태.

    모든 노드는 이 State를 읽고 부분적으로 업데이트하여 반환한다.
    LangGraph는 반환된 딕셔너리를 기존 State에 병합(shallow merge)한다.

    Fields:
        topic           : 토론 주제
        user_stance     : 사용자 진영
        user_intensity  : 사용자 강경도 (1~5)
        agents          : 생성된 AI 에이전트 스냅샷 리스트
        debate_history  : 전체 발언 기록 (시간순 append)
        current_turn    : 현재 발언 차례 번호
        current_speaker : 현재 발언자 ID
        phase           : 현재 토론 단계
        synthesis_draft : synthesis 단계에서 누적되는 합의 초안
        is_finished     : 토론 종료 여부
    """

    topic: str
    user_stance: Literal["PRO", "CON"]
    user_intensity: int
    agents: List[AgentSnapshot]
    debate_history: List[DebateEntry]
    current_turn: int
    current_speaker: str
    phase: DebatePhase
    synthesis_draft: Optional[str]
    is_finished: bool


# ── 초기 State 생성 유틸리티 ──────────────────────────────────────────────────

def build_initial_state(
    topic: str,
    user_stance: Literal["PRO", "CON"],
    user_intensity: int,
    agents: List[AgentSnapshot],
) -> DebateState:
    """Phase 0 초기화 시 LangGraph에 주입할 기본 State를 생성한다.

    첫 발언자는 PRO 진영의 첫 번째 참가자(사용자 또는 agent)로 설정한다.
    사용자가 PRO이면 사용자가 먼저 발언하고, CON이면 PRO 측 agent_1이 먼저 발언한다.

    Args:
        topic: 토론 주제
        user_stance: 사용자 진영
        user_intensity: 사용자 강경도
        agents: persona_factory에서 생성된 AgentSnapshot 리스트

    Returns:
        초기화된 DebateState
    """
    # PRO 진영 첫 발언자 결정
    if user_stance == "PRO":
        first_speaker = "user"
    else:
        # PRO 에이전트 중 첫 번째를 탐색
        pro_agents = [a for a in agents if a["stance"] == "PRO"]
        first_speaker = pro_agents[0]["agent_id"] if pro_agents else "user"

    return DebateState(
        topic=topic,
        user_stance=user_stance,
        user_intensity=user_intensity,
        agents=agents,
        debate_history=[],       # Phase 1에서 발언이 누적됨
        current_turn=0,
        current_speaker=first_speaker,
        phase="opening",         # 항상 opening 단계에서 시작
        synthesis_draft=None,    # synthesis 단계 전까지 None
        is_finished=False,
    )


def generate_session_id() -> str:
    """고유 세션 ID를 생성한다. 형식: debate_<uuid4 앞 8자리>"""
    return f"debate_{uuid.uuid4().hex[:8]}"
