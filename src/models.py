"""
models.py — FastAPI 입력 스키마 및 응답 모델 (Phase 0)

사용자로부터 받는 토론 초기화 요청을 검증하고,
LangGraph 초기화에 필요한 구조로 정의한다.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, field_validator, model_validator


# ── 포맷별 생성해야 할 AI 수 매핑 ─────────────────────────────────────────────
DEBATE_FORMAT_AI_COUNT: Dict[str, int] = {
    "1:1": 1,
    "2:2": 3,
    "3:3": 5,
}


class DebateInitRequest(BaseModel):
    """토론 초기화 요청 모델.

    사용자가 API에 전달하는 모든 입력값을 검증한다.
    agent_intensities 길이는 토론 포맷에서 요구하는 AI 수와 일치해야 한다.
    """

    topic: str # 토론 주제 
    user_stance: Literal["PRO", "CON"] # 사용자의 찬반 입장 
    user_intensity: int  # 사용자의 강경도 (1~5)
    agent_intensities: List[int]  # 각 AI 에이전트 강경도 (1~5). 진영 할당은 응답의 agents 배열에서 확인
    debate_format: Literal["1:1", "2:2", "3:3"] # 토론 형식

    # ── 단일 필드 검증 ─────────────────────────────────────────────────────────

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
                raise ValueError(
                    f"agent_intensities[{i}]는 1~5 사이여야 합니다. 입력값: {intensity}"
                )
        return v

    @field_validator("topic")
    @classmethod
    def validate_topic_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic은 비어 있을 수 없습니다.")
        return v.strip()

    # ── 복합 필드 검증 (format ↔ agent_intensities 길이 일치) ──────────────────

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


# ── 응답 모델 ─────────────────────────────────────────────────────────────────

class AgentInfo(BaseModel):
    """생성된 AI 에이전트 요약 정보 (응답 전용)."""

    agent_id: str
    stance: Literal["PRO", "CON"]
    intensity: int
    role_description: str  # persona_factory에서 생성된 역할 설명


class DebateInitResponse(BaseModel):
    """토론 초기화 API 응답 모델."""

    session_id: str
    topic: str
    agents: List[AgentInfo]
    initial_state: Dict[str, Any]
    message: str


# ── Stage 1: 입론 API 모델 ──────────────────────────────────────────────────────

class UserOpeningRequest(BaseModel):
    """사용자 입론 제출 요청."""

    content: str

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("입론 내용은 비어 있을 수 없습니다.")
        return v.strip()


class OpeningRunResponse(BaseModel):
    """AI 입론 생성 응답."""

    session_id: str
    phase: str
    user_turn: int
    debate_history: List[Dict[str, Any]]
    message: str


class UserOpeningResponse(BaseModel):
    """사용자 입론 제출 응답."""

    session_id: str
    phase: str
    debate_history: List[Dict[str, Any]]
    message: str
