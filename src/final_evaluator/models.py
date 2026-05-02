"""final_evaluator 출력 JSON 스키마."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel


class SideStats(BaseModel):
    """진영별 3지표 평균."""
    argument: float
    evidence: float
    language: float
    weighted_mean: float
    turn_count: int


class WinnerBlock(BaseModel):
    side: Literal["PRO", "CON", "DRAW"]
    pro_percent: float
    con_percent: float
    margin: float
    summary: str                     # LLM 생성 1~2줄 평


class SwingTurn(BaseModel):
    turn_index: int
    speaker_id: str
    side: Literal["PRO", "CON"]
    phase: str
    type: Literal[
        "biggest_swing",
        "best_rebuttal",
        "worst_turn",
        "logical_error",
    ]
    weighted_score: float
    impact: float                    # 해당 턴이 진영에 기여한 강도
    speech_summary: str
    narrative: str                   # LLM 생성 줄글


class MVPBlock(BaseModel):
    speaker_id: str
    side: Literal["PRO", "CON"]
    total_impact: float
    avg_weighted: float
    avg_argument: float
    avg_evidence: float
    avg_language: float
    reason: str                      # 선정 근거 (규칙 기반 설명)


class DimensionFeedback(BaseModel):
    praise: str                      # 칭찬 (LLM)
    critique: str                    # 지적 (LLM)
    suggestion: str                  # 제안 (LLM)


class CoachFeedback(BaseModel):
    argument: DimensionFeedback
    evidence: DimensionFeedback
    language: DimensionFeedback


class FinalReport(BaseModel):
    """프론트 대시보드용 최종 평가 리포트."""
    topic: str
    debate_format: str
    user_stance: Literal["PRO", "CON"]
    total_turns: int

    winner: WinnerBlock
    stats: dict                      # {"PRO": SideStats, "CON": SideStats}
    swing_turns: List[SwingTurn]
    mvp: Optional[MVPBlock]
    coach_feedback: CoachFeedback

    generated_at: str                # ISO timestamp
