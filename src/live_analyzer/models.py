from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field
from .config import DIMENSION_WEIGHTS, TOTAL_WEIGHT


class DimensionResult(BaseModel):
    score: float = Field(..., ge=1.0, le=10.0)
    summary: str


class TurnAnalysis(BaseModel):
    turn_index: int
    speaker_id: str
    speaker_stance: Literal["PRO", "CON"]
    phase: str
    target_id: Optional[str] = None
    argument: DimensionResult
    evidence: DimensionResult
    language: DimensionResult
    overall_summary: str
    memory_summary: str
    weighted_score: float

    @classmethod
    def compute(
        cls,
        turn_index: int,
        speaker_id: str,
        speaker_stance: str,
        phase: str,
        target_id: Optional[str],
        argument: DimensionResult,
        source: DimensionResult,
        language: DimensionResult,
        overall_summary: str,
        memory_summary: str,
    ) -> "TurnAnalysis":
        ws = (
            argument.score * DIMENSION_WEIGHTS["argument"]
            + source.score * DIMENSION_WEIGHTS["evidence"]
            + language.score * DIMENSION_WEIGHTS["language"]
        ) / TOTAL_WEIGHT
        return cls(
            turn_index=turn_index,
            speaker_id=speaker_id,
            speaker_stance=speaker_stance,
            phase=phase,
            target_id=target_id,
            argument=argument,
            evidence=source,
            language=language,
            overall_summary=overall_summary,
            memory_summary=memory_summary,
            weighted_score=round(ws, 2),
        )
