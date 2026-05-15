"""
state.py — LangGraph State 설계 및 초기화 유틸리티

구성적 논쟁(Constructive Controversy) 기반 토론 워크플로우의 공유 상태를 정의한다.

[발언 순서 원칙]
    - 사용자는 항상 모든 발언자 중 **가장 마지막**에 배치된다.
    - 입론·자유 논박 발언 순서: 상대 진영 → 같은 진영 → 상대 → 같은 → ... → 사용자 마지막
      · 사용자 PRO: CON → PRO → CON → PRO → ... → user(PRO)
      · 사용자 CON: PRO → CON → PRO → CON → ... → user(CON)
    - 연쇄 논박은 (PRO_i, CON_i) 쌍 기반으로 자동 생성되며, 사용자가 포함된 쌍이 마지막 라운드에 배치된다.

[Phase 1] 1~4단계
    1단계 입론        (opening)          : PRO1 → CON1 → PRO2 → CON2 → ... (교차)
    2단계 연쇄 논박   (chained_rebuttal) : (PRO_i, CON_i) 쌍마다 CON→PRO, PRO→CON 순으로 1:1 비판
                                          각 라운드는 공격 발언 → 응답 발언 2개 서브턴으로 구성
    3단계 자유 논박   (free_rebuttal)    : 입론과 동일한 고정 루프 + Max_Cycle 자동 종료
                                          발화 시 @에이전트명 반드시 포함 (사용자 포함)
    4단계 역할 반전   (role_reversal)    : 사회자가 사용자에게 역할 반전 강제

[Phase 2] 5단계
    5단계 종합 및 재개념 (synthesis)     : 병렬 판정단 승패 + 제3의 최선택 합의안
"""

import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple
from typing_extensions import NotRequired, TypedDict

from langgraph.graph import add_messages  # noqa: F401 — 메시지 누적에 활용


# ── 토론 단계 타입 ────────────────────────────────────────────────────────────
DebatePhase = Literal[
    "opening",           # 1단계: 입론
    "chained_rebuttal",  # 2단계: 연쇄 논박
    "free_rebuttal",     # 3단계: 자유 논박
    "role_reversal",     # 4단계: 역할 반전
    "synthesis",         # 5단계: 종합 및 재개념 (Phase 2)
]


# ── 히스토리 엔트리 타입 ──────────────────────────────────────────────────────
class DebateEntry(TypedDict):
    """단일 발언 기록.

    Attributes:
        turn        : 발언 순서 번호 (0부터 시작)
        speaker_id  : 발언자 ID ("user" 또는 "agent_N")
        stance      : 발언자 진영
        phase       : 발언 시점의 토론 단계
        content     : 발언 내용
        target_id   : 비판 대상 ID (chained_rebuttal / free_rebuttal 에서 사용, 그 외 None)
    """

    turn: int
    speaker_id: str
    stance: Literal["PRO", "CON"]
    phase: DebatePhase
    content: str
    target_id: Optional[str]
    tool_calls_log: NotRequired[List[Dict[str, Any]]]  # 사용된 도구 목록 (디버깅용)
    json_raw: NotRequired[Optional[str]]               # LLM JSON 원본 (reasoning 포함, 디버깅용)


# ── 연쇄 논박 페어 타입 ───────────────────────────────────────────────────────
class RebuttalPair(TypedDict):
    """2단계 연쇄 논박의 1:1 비판 쌍.

    (PRO_i, CON_i) 쌍마다 2개의 라운드가 생성된다:
        R(2i-1): CON_i 공격 → PRO_i 응답
        R(2i)  : PRO_i 공격 → CON_i 응답

    Attributes:
        round             : 라운드 번호 (1부터 시작)
        attacker_id       : 공격 발언자 ID
        target_id         : 응답 발언자 ID
        awaiting_response : False=공격 발언 차례, True=응답 발언 차례
        done              : 공격+응답 모두 완료 여부
    """

    round: int
    attacker_id: str
    target_id: str
    awaiting_response: bool
    done: bool


# ── 에이전트 스냅샷 타입 ──────────────────────────────────────────────────────
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
    focus_area: str  # 에이전트별 논증 전문 분야 (같은 진영 내 의견 다양성 확보)


# ── 핵심 State TypedDict ──────────────────────────────────────────────────────
class DebateState(TypedDict):
    """LangGraph StateGraph에서 사용하는 토론 공유 상태.

    모든 노드는 이 State를 읽고 부분적으로 업데이트하여 반환한다.
    LangGraph는 반환된 딕셔너리를 기존 State에 shallow merge한다.

    [기본 정보]
        topic                 : 토론 주제
        user_stance           : 사용자 진영
        user_intensity        : 사용자 강경도 (1~5)
        agents                : 생성된 AI 에이전트 스냅샷 리스트

    [발언 기록]
        debate_history        : 전체 발언 기록 (시간순 append)

    [발언 순서 제어]
        speaking_order        : 현재 단계의 발언 순서 (speaker_id 리스트)
                                1단계·3단계 공용 (상대 진영 → 같은 진영 교차, 사용자 항상 마지막)
        current_speaker_index : speaking_order 내 현재 위치
        current_turn          : 전체 누적 발언 번호

    [단계 제어]
        phase                 : 현재 토론 단계
        current_cycle         : 자유 논박(3단계) 현재 사이클 번호
        max_cycle             : 자유 논박 최대 사이클 수 (AI 폭주 방지)

    [연쇄 논박 - 2단계]
        rebuttal_pairs        : RebuttalPair 리스트 (2단계 진입 시 생성)
        current_rebuttal_round: 현재 진행 중인 라운드 인덱스 (0부터 시작)

    [역할 반전 - 4단계]
        role_reversed         : 역할 반전 완료 여부

    [종합 - 5단계]
        synthesis_draft       : 누적되는 합의 초안

    [종료]
        is_finished           : 토론 종료 여부
    """

    # 기본 정보
    topic: str
    topic_id: str  # 토픽 ID (검색 쿼리 매칭용)
    user_stance: Literal["PRO", "CON"]
    user_intensity: int
    agents: List[AgentSnapshot]

    # 발언 기록
    debate_history: List[DebateEntry]

    # 발언 순서 제어
    speaking_order: List[str]
    current_speaker_index: int
    current_turn: int

    # 단계 제어
    phase: DebatePhase
    current_cycle: int
    max_cycle: int

    # 입론 (1단계) — step 노드 카운터. TypedDict 에 정의돼야 LangGraph 체크포인트에 저장됨
    # (정의 안 하면 노드가 idx+1 반환해도 다음 호출 때 초기값 0 으로 돌아가 무한 loop).
    opening_pre_idx: int   # 사용자 전 AI 입론 step 인덱스 (0→len(pre_speakers))
    opening_post_idx: int  # 사용자 후 AI 입론 step 인덱스

    # 연쇄 논박 (2단계)
    rebuttal_pairs: Optional[List[RebuttalPair]]
    current_rebuttal_round: int

    # 자유 논박 (3단계) — 1:1 핑퐁
    selected_opponent_id: Optional[str]  # 사용자가 선택한 상대 에이전트 ID
    free_rebuttal_user_turns: int  # 사용자 자유논박 라운드 (0→1→2). 1=답변+공격, 2=최종 답변만. 2 도달 시 역할반전

    # 역할 반전 (4단계)
    role_reversed: bool

    # 종합 (5단계)
    synthesis_draft: Optional[str]
    synthesis_user_turns: int  # 사용자 의견 제출 수 (0→1→2, 2 도달 시 확정 화면)
    synthesis_propose_idx: int  # 초기 의견 제시 step 인덱스 (첫 라운드)
    synthesis_discuss_idx: int  # 사용자 발언 후 응답 step 인덱스 (라운드마다 user_synthesis 가 0 으로 리셋)

    # 종료
    is_finished: bool


# ── 내부 유틸리티 ─────────────────────────────────────────────────────────────

def _build_stance_lists(
    agents: List[AgentSnapshot],
    user_stance: Literal["PRO", "CON"],
) -> Tuple[List[str], List[str]]:
    """진영별 발언자 ID 리스트를 반환한다. 사용자는 자신의 진영 마지막에 배치.

    Args:
        agents      : AI 에이전트 스냅샷 리스트
        user_stance : 사용자 진영

    Returns:
        (pro_ids, con_ids) — 각 진영의 발언자 ID 리스트 (사용자 포함)
    """
    pro_ids = [a["agent_id"] for a in agents if a["stance"] == "PRO"]
    con_ids = [a["agent_id"] for a in agents if a["stance"] == "CON"]

    if user_stance == "PRO":
        pro_ids.append("user")
    else:
        con_ids.append("user")

    return pro_ids, con_ids


def _build_interleaved_order(
    agents: List[AgentSnapshot],
    user_stance: Literal["PRO", "CON"],
) -> List[str]:
    """발언 순서를 생성한다: 상대 진영 → 같은 진영 교차, 사용자는 맨 마지막.

    - 사용자 PRO:  CON1 → PRO1 → CON2 → PRO2 → ... → user
    - 사용자 CON:  PRO1 → CON1 → PRO2 → CON2 → ... → user

    진영 간 AI 인원수가 다를 경우 남는 쪽을 뒤에 이어 붙이고, 사용자는 항상 맨 끝.

    Args:
        agents      : AI 에이전트 스냅샷 리스트 (사용자 제외)
        user_stance : 사용자 진영

    Returns:
        speaker_id 리스트 (마지막 요소는 항상 "user")
    """
    pro_ids = [a["agent_id"] for a in agents if a["stance"] == "PRO"]
    con_ids = [a["agent_id"] for a in agents if a["stance"] == "CON"]

    if user_stance == "PRO":
        opposite, own = con_ids, pro_ids
    else:
        opposite, own = pro_ids, con_ids

    order: List[str] = []
    for opp, ow in zip(opposite, own):
        order.append(opp)
        order.append(ow)
    # 남는 쪽 뒤에 추가
    order.extend(opposite[len(own):])
    order.extend(own[len(opposite):])
    # 사용자는 항상 맨 마지막
    order.append("user")
    return order


# ── 공개 유틸리티 ─────────────────────────────────────────────────────────────

def build_chained_rebuttal_pairs(
    agents: List[AgentSnapshot],
    user_stance: Literal["PRO", "CON"],
) -> List[RebuttalPair]:
    """2단계 진입 시 연쇄 논박 페어 리스트를 생성한다.

    (PRO_i, CON_i) 쌍마다 2개의 라운드를 생성한다:
        R(2i-1): CON_i 공격 → PRO_i 응답
        R(2i)  : PRO_i 공격 → CON_i 응답

    총 라운드 수 = 2 * min(len(pro_ids), len(con_ids))
    인원이 늘어나도 자동으로 쌍이 추가된다.

    Args:
        agents      : AI 에이전트 스냅샷 리스트
        user_stance : 사용자 진영

    Returns:
        RebuttalPair 리스트
    """
    pro_ids, con_ids = _build_stance_lists(agents, user_stance)

    pairs: List[RebuttalPair] = []
    round_num = 1
    for pro_id, con_id in zip(pro_ids, con_ids):
        # 사용자 진영의 상대방이 먼저 공격
        if user_stance == "PRO":
            # 사용자가 찬성이면 반대(상대)가 먼저 공격
            first_attacker, first_target = con_id, pro_id
            second_attacker, second_target = pro_id, con_id
        else:
            # 사용자가 반대면 찬성(상대)이 먼저 공격
            first_attacker, first_target = pro_id, con_id
            second_attacker, second_target = con_id, pro_id

        pairs.append(RebuttalPair(
            round=round_num,
            attacker_id=first_attacker,
            target_id=first_target,
            awaiting_response=False,
            done=False,
        ))
        round_num += 1
        pairs.append(RebuttalPair(
            round=round_num,
            attacker_id=second_attacker,
            target_id=second_target,
            awaiting_response=False,
            done=False,
        ))
        round_num += 1

    return pairs


def build_initial_state(
    topic: str,
    user_stance: Literal["PRO", "CON"],
    user_intensity: int,
    agents: List[AgentSnapshot],
    max_cycle: int = 4,
    topic_id: str = "",
) -> DebateState:
    """Phase 0 초기화 시 LangGraph에 주입할 기본 State를 생성한다.

    사용자는 모든 발언자 중 맨 마지막에 배치되며, 입론 순서는 상대 진영 → 같은 진영 교차.
    포맷(1:1/2:2/3:3)과 사용자 진영(PRO/CON)에 관계없이 동일한 로직이 적용된다.

    Args:
        topic         : 토론 주제
        user_stance   : 사용자 진영
        user_intensity: 사용자 강경도
        agents        : persona_factory에서 생성된 AgentSnapshot 리스트
        max_cycle     : 자유 논박 최대 사이클 수 (기본값 4)

    Returns:
        초기화된 DebateState
    """
    speaking_order = _build_interleaved_order(agents, user_stance)

    return DebateState(
        topic=topic, # 입력받은 주제
        topic_id=topic_id, # 토픽 ID
        user_stance=user_stance, # 사용자 진영
        user_intensity=user_intensity, # 사용자 강경도
        agents=agents, # AI Agent 목록 저장
        debate_history=[], # 발언 기록(처음엔 발언 기록이 없으니까 빈 리스트)
        speaking_order=speaking_order, # 교차 발언 순서 
        current_speaker_index=0, # 첫 번째 화자부터 시작
        current_turn=0, # 아직 아무도 말 안 했으니 첫 발언 번호 0
        phase="opening", # 토론은 항상 입론에서 시작
        current_cycle=0, # 자유논박은 시작 안 했으니 0
        max_cycle=max_cycle, # 기본으로 4
        opening_pre_idx=0,         # 1단계 step 카운터 (사용자 전 AI)
        opening_post_idx=0,        # 1단계 step 카운터 (사용자 후 AI)
        rebuttal_pairs=None,       # 2단계 진입 시 build_chained_rebuttal_pairs()로 생성
        current_rebuttal_round=0, # 라운드 시작 전, 기본은 0
        selected_opponent_id=None, # 3단계 진입 시 사용자가 선택
        free_rebuttal_user_turns=0, # 자유논박 사용자 라운드 (최대 2, 2회차는 답변만)
        role_reversed=False, # 역할 반전 아직 시작 안 함
        synthesis_draft=None,      # 5단계 진입 전까지 None
        synthesis_user_turns=0, # 종합 회의 사용자 턴 수 (최대 2)
        synthesis_propose_idx=0,   # 5단계 step 카운터 (초기 의견 제시 라운드)
        synthesis_discuss_idx=0,   # 5단계 step 카운터 (사용자 발언 후 응답 라운드, 매 라운드 0 으로 리셋)
        is_finished=False, # 토론 시작 상태이므로 종료 아님
    )


def generate_session_id() -> str:
    """고유 세션 ID를 생성한다. 형식: debate_<uuid4 앞 8자리>"""
    return f"debate_{uuid.uuid4().hex[:8]}"
