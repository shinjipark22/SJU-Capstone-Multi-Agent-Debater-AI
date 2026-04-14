"""최종 평가자 에이전트용 입력 빌더.

DebateState에 누적된 judge_* 필드로부터 최종 평가에 필요한
compact한 dict를 구성한다. 최종 평가자는 이 입력을 LLM에 넘겨 총평·승패 판정 생성.
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_final_judge_input(state: Dict[str, Any]) -> Dict[str, Any]:
    """state에서 최종 평가자용 요약 입력을 생성한다.

    포함:
      - topic, debate_format, user_stance
      - final_stats: {pro_percent, con_percent, pro_ema, con_ema}
      - turn_history: 턴별 {
            turn, speaker_id, stance, phase,
            speech_summary,              # 발언 2~3문장 요약
            score,                       # weighted 1~10
            dimension_scores: {argument, evidence, language},
            dimension_feedbacks: {argument, evidence, language},  # 각 차원 이유
            overall_feedback,            # "논증:... | 근거:... | 언어:..."
        }
    """
    analyses: List[dict] = state.get("judge_analyses", [])
    speech_sums: List[dict] = state.get("judge_speech_summaries", [])
    summary_by_turn = {s["turn_index"]: s["summary"] for s in speech_sums}

    turn_history = []
    for a in analyses:
        turn_idx = a["turn_index"]
        arg = a["argument"]
        ev = a["evidence"]
        lang = a["language"]
        turn_history.append({
            "turn": turn_idx,
            "speaker_id": a["speaker_id"],
            "stance": a["speaker_stance"],
            "phase": a["phase"],
            "target_id": a.get("target_id"),
            "speech_summary": summary_by_turn.get(turn_idx, ""),
            "score": a["weighted_score"],
            "dimension_scores": {
                "argument": arg["score"],
                "evidence": ev["score"],
                "language": lang["score"],
            },
            "dimension_feedbacks": {
                "argument": arg["summary"],
                "evidence": ev["summary"],
                "language": lang["summary"],
            },
            "overall_feedback": a.get("overall_summary", ""),
        })

    return {
        "topic": state.get("topic", ""),
        "debate_format": state.get("debate_format", ""),
        "user_stance": state.get("user_stance", ""),
        "final_stats": dict(state.get("judge_live", {})),
        "turn_history": turn_history,
    }


def project_frontend_event(analysis: Dict[str, Any], live_debate: Dict[str, float]) -> Dict[str, Any]:
    """프론트 SSE 이벤트용 최소 projection (score 3개 + percent 2개).

    Args:
        analysis: TurnAnalysis.model_dump() 또는 state["judge_analyses"]의 원소
        live_debate: state["judge_live"]
    """
    return {
        "turn_index": analysis.get("turn_index"),
        "speaker_id": analysis.get("speaker_id"),
        "argument_score": analysis["argument"]["score"] if "argument" in analysis else None,
        "evidence_score": analysis["evidence"]["score"] if "evidence" in analysis else None,
        "language_score": analysis["language"]["score"] if "language" in analysis else None,
        "pro_percent": live_debate.get("pro_percent"),
        "con_percent": live_debate.get("con_percent"),
    }
