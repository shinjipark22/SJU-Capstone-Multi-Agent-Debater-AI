"""AnalysisMemory — DebateState(dict)를 백킹 스토어로 사용하는 얇은 어댑터.

원본 ymj-judge는 독립 객체였으나, 메모리·sync·persistence 효율을 위해
DebateState의 `judge_live` / `judge_analyses` 필드를 직접 읽고 쓴다.
speech_memory도 별도 유지하지 않고 debate_history(원본 발언)에서 온디맨드로 재구성.

하위 호환을 위해 기존 public API (add_speech / add_analysis / live_debate_snapshot /
to_snapshot / speech_memory 속성 등)은 그대로 유지.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .config import EMA_ALPHA, RECENT_WINDOW

if TYPE_CHECKING:
    from .models import TurnAnalysis


def _ensure_judge_fields(state: Dict[str, Any]) -> None:
    """DebateState에 judge 필드가 없으면 초기화 (in-place)."""
    if "judge_live" not in state:
        state["judge_live"] = {
            "pro_percent": 50.0,
            "con_percent": 50.0,
            "pro_ema": 5.0,
            "con_ema": 5.0,
        }
    if "judge_analyses" not in state:
        state["judge_analyses"] = []
    if "judge_speech_summaries" not in state:
        state["judge_speech_summaries"] = []  # 발언별 요약만 경량 저장


class AnalysisMemory:
    """DebateState를 백킹 스토어로 하는 adapter.

    호출자는 기존과 동일하게 `memory.add_speech(...)`, `memory.add_analysis(...)`를 쓰되,
    내부적으로 전달받은 state dict를 수정한다. state 미전달 시 fallback dict를 자체 생성.
    """

    def __init__(
        self,
        topic: str,
        debate_format: str,
        user_id: str,
        user_stance: str,
        teams: dict,
        state: Optional[Dict[str, Any]] = None,
    ):
        self.topic = topic
        self.debate_format = debate_format
        self.user_id = user_id
        self.user_stance = user_stance
        self.teams = teams
        # state가 주어지면 그걸 직접 사용 (LangGraph state 공유)
        # 아니면 독립 dict 생성 (단위 테스트/시뮬레이션용)
        self._state: Dict[str, Any] = state if state is not None else {}
        _ensure_judge_fields(self._state)

    # ── LangGraph state 접근 ──────────────────────────────────────────
    @property
    def state(self) -> Dict[str, Any]:
        return self._state

    @property
    def live_debate(self) -> Dict[str, float]:
        return self._state["judge_live"]

    @property
    def speech_memory(self) -> List[dict]:
        """발언 요약 히스토리 — LangGraph state에서 직접 참조."""
        return self._state["judge_speech_summaries"]

    @property
    def analysis_memory(self) -> List[dict]:
        return self._state["judge_analyses"]

    @property
    def _recent(self) -> List[str]:
        return self._state.setdefault("judge_recent", [])

    # ── 업데이트 로직 (원본과 동일) ──────────────────────────────────
    def _update_live_debate(self, score: float, stance: str) -> None:
        live = self._state["judge_live"]
        if stance == "PRO":
            live["pro_ema"] = round(
                EMA_ALPHA * score + (1 - EMA_ALPHA) * live["pro_ema"], 4
            )
        else:
            live["con_ema"] = round(
                EMA_ALPHA * score + (1 - EMA_ALPHA) * live["con_ema"], 4
            )
        diff = live["pro_ema"] - live["con_ema"]
        pro_percent = 50.0 + diff * 15
        pro_percent = max(5.0, min(95.0, pro_percent))
        live["pro_percent"] = round(pro_percent, 2)
        live["con_percent"] = round(100.0 - pro_percent, 2)

    def live_debate_snapshot(self) -> dict:
        return dict(self._state["judge_live"])

    def add_speech(
        self,
        turn_index: int,
        speaker_id: str,
        speaker_stance: str,
        phase: str,
        target_id: Optional[str],
        summary: str,
    ) -> None:
        self._state["judge_speech_summaries"].append({
            "turn_index": turn_index,
            "speaker_id": speaker_id,
            "speaker_stance": speaker_stance,
            "phase": phase,
            "target_id": target_id,
            "summary": summary,
        })

    def add_analysis(
        self,
        turn_index: int,
        speaker_stance: str,
        analysis: "TurnAnalysis",
    ) -> None:
        self._state["judge_analyses"].append({
            "turn_index": turn_index,
            "speaker_id": analysis.speaker_id,
            "speaker_stance": speaker_stance,
            "phase": analysis.phase,
            "target_id": analysis.target_id,
            "argument": analysis.argument.model_dump(),
            "evidence": analysis.evidence.model_dump(),
            "language": analysis.language.model_dump(),
            "weighted_score": analysis.weighted_score,
            "overall_summary": analysis.overall_summary,
        })
        self._update_live_debate(analysis.weighted_score, speaker_stance)
        line = (
            f"[T{turn_index}] {analysis.speaker_id}({speaker_stance}) {analysis.phase}: "
            f"종합 {analysis.weighted_score:.2f} | 현재 "
            f"{self._state['judge_live']['con_percent']:.1f}:"
            f"{self._state['judge_live']['pro_percent']:.1f}"
        )
        recent = self._state.setdefault("judge_recent", [])
        recent.append(line)
        if len(recent) > RECENT_WINDOW:
            recent.pop(0)

    def to_snapshot(self) -> dict:
        return {
            "topic": self.topic,
            "format": self.debate_format,
            "teams": self.teams,
            "user_id": self.user_id,
            "user_stance": self.user_stance,
            "live_debate": self.live_debate_snapshot(),
            "speech_memory": list(self._state["judge_speech_summaries"]),
            "analysis_memory": list(self._state["judge_analyses"]),
            "recent_turn_summaries": list(self._state.get("judge_recent", [])),
        }
