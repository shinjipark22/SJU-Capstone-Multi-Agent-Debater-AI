from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
from .config import EMA_ALPHA, RECENT_WINDOW

if TYPE_CHECKING:
    from .models import TurnAnalysis


class AnalysisMemory:
    def __init__(self, topic: str, debate_format: str, user_id: str,
                 user_stance: str, teams: dict):
        self.topic = topic
        self.debate_format = debate_format
        self.user_id = user_id
        self.user_stance = user_stance
        self.teams = teams
        self.live_debate = {
            "pro_percent": 50.0,
            "con_percent": 50.0,
            "pro_ema": 5.0,
            "con_ema": 5.0,
        }
        self.speech_memory: List[dict] = []
        self.analysis_memory: List[dict] = []
        self._recent: List[str] = []

    def _update_live_debate(self, score: float, stance: str):
        if stance == "PRO":
            self.live_debate["pro_ema"] = round(
                EMA_ALPHA * score + (1 - EMA_ALPHA) * self.live_debate["pro_ema"], 4
            )
        else:
            self.live_debate["con_ema"] = round(
                EMA_ALPHA * score + (1 - EMA_ALPHA) * self.live_debate["con_ema"], 4
            )
        # EMA 차이를 증폭해서 판세에 반영 (차이 * 15 배율)
        diff = self.live_debate["pro_ema"] - self.live_debate["con_ema"]
        pro_percent = 50.0 + diff * 15
        pro_percent = max(5.0, min(95.0, pro_percent))
        self.live_debate["pro_percent"] = round(pro_percent, 2)
        self.live_debate["con_percent"] = round(100.0 - pro_percent, 2)

    def live_debate_snapshot(self) -> dict:
        return dict(self.live_debate)

    def add_speech(self, turn_index: int, speaker_id: str, speaker_stance: str,
                   phase: str, target_id: Optional[str], summary: str):
        """발언 요약을 분석 전에 speech_memory에 저장."""
        self.speech_memory.append({
            "turn_index": turn_index,
            "speaker_id": speaker_id,
            "speaker_stance": speaker_stance,
            "phase": phase,
            "target_id": target_id,
            "summary": summary,
        })

    def add_analysis(self, turn_index: int, speaker_stance: str,
                     analysis: "TurnAnalysis"):
        """3차원 분석 결과를 analysis_memory에 저장."""
        self.analysis_memory.append({
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
            f"{self.live_debate['con_percent']:.1f}:{self.live_debate['pro_percent']:.1f}"
        )
        self._recent.append(line)
        if len(self._recent) > RECENT_WINDOW:
            self._recent.pop(0)

    def to_snapshot(self) -> dict:
        return {
            "topic": self.topic,
            "format": self.debate_format,
            "teams": self.teams,
            "user_id": self.user_id,
            "user_stance": self.user_stance,
            "live_debate": self.live_debate_snapshot(),
            "speech_memory": self.speech_memory,
            "analysis_memory": self.analysis_memory,
            "recent_turn_summaries": list(self._recent),
        }
