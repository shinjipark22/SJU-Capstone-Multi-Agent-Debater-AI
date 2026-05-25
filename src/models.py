"""
models.py — FastAPI 입력 스키마

토론 초기화 요청을 검증한다. 응답은 SSE 이벤트로 스트리밍되므로
별도 응답 모델은 사용하지 않는다.
"""

from typing import Dict, List, Literal
from pydantic import BaseModel, field_validator, model_validator


# ── 포맷별 생성해야 할 AI 수 매핑 ─────────────────────────────────────────────
DEBATE_FORMAT_AI_COUNT: Dict[str, int] = {
    "1:1": 1,
    "2:2": 3,
    "3:3": 5,
}


class DebateInitRequest(BaseModel):
    """토론 초기화 요청 모델."""

    topic: str
    user_stance: Literal["PRO", "CON"]
    user_intensity: int
    agent_intensities: List[int]
    debate_format: Literal["1:1", "2:2", "3:3"]
    # 토론 모드:
    #   "debate"        — 입론·연쇄논박·자유논박까지 (3단계). 평가는 /evaluation 으로 별도.
    #   "constructive"  — 전체 5단계 (역할반전·종합 포함).
    # 미지정 시 기존 동작 (전체 5단계) 유지.
    mode: Literal["debate", "constructive"] = "constructive"

    @field_validator("user_intensity")
    @classmethod
    def validate_user_intensity(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError(f"user_intensity는 1~5 사이여야 합니다. 입력값: {v}")
        return v

    @field_validator("agent_intensities")
    @classmethod
    def validate_agent_intensities(cls, v: List[int]) -> List[int]:
        for i, intensity in enumerate(v):
            if not (1 <= intensity <= 5):
                raise ValueError(f"agent_intensities[{i}]는 1~5 사이여야 합니다. 입력값: {intensity}")
        return v

    @field_validator("topic")
    @classmethod
    def validate_topic_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic은 비어 있을 수 없습니다.")
        return v.strip()

    @model_validator(mode="after")
    def validate_agent_count_matches_format(self) -> "DebateInitRequest":
        expected = DEBATE_FORMAT_AI_COUNT[self.debate_format]
        actual = len(self.agent_intensities)
        if actual != expected:
            raise ValueError(
                f"debate_format='{self.debate_format}'에서 AI {expected}명이 필요하지만, "
                f"agent_intensities 길이가 {actual}입니다."
            )
        return self
