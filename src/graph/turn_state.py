"""
turn_state.py — 서브그래프 내부 State (턴 단위)

메인 DebateState와 분리된, 1턴 생성 파이프라인 전용 상태.
Searcher → Writer → Reviewer 사이에서 전달됨.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class TurnState(TypedDict):
    """서브그래프 1턴 실행 State."""

    # 입력 (호출 시 설정)
    topic: str
    agent: Dict                    # AgentSnapshot dict
    expected_stance: str           # "PRO" or "CON"
    target_argument: str           # 공격 대상 텍스트
    my_opening: str                # 나의 입론 (방어용)
    opp_opening: str               # 상대 입론 (공격 모순 지적용)
    chain: List                    # 멀티턴 메시지 체인
    prev_weaknesses: str           # 이전 약점 분석 (중복 방지)
    prev_attacks: str              # 이전 공격 내용 (중복 방지)
    mode: str                      # "opening" | "rebuttal" | "defense" | "attack" | "role_reversal" | "synthesis"

    # Searcher 출력
    weakness: str
    search_results: str
    search_query: str

    # Writer 출력
    speech: str
    raw: str

    # Reviewer 출력
    review_result: Dict
    retry_count: int

    # 메타
    tool_calls_log: List[Dict]
