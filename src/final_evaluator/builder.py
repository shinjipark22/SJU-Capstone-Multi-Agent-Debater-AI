"""builder — 집계 + LLM 호출 + 최종 JSON 조립."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

from src.live_analyzer.inference import parse_json, qwen_chat

from .aggregator import (
    compute_mvp,
    compute_side_stats,
    decide_winner,
    find_extreme_per_dimension,
    select_swing_turns,
)
from .models import (
    CoachFeedback,
    DimensionFeedback,
    FinalReport,
    MVPBlock,
    SideStats,
    SwingTurn,
    WinnerBlock,
)
from .prompts import (
    SYSTEM_COACH,
    SYSTEM_SUMMARY,
    SYSTEM_SWING_NARRATIVE,
    build_coach_user_msg,
    build_summary_user_msg,
    build_swing_user_msg,
)

logger = logging.getLogger(__name__)


# ── LLM 호출 래퍼 (장애 내성) ─────────────────────────────────────────────

def _safe_summary(topic: str, winner_side: str, pro: float, con: float, stats: dict) -> str:
    try:
        msg = build_summary_user_msg(topic, winner_side, pro, con, stats)
        return qwen_chat(SYSTEM_SUMMARY, msg, max_new_tokens=200).strip()
    except Exception as e:
        logger.warning("[final_evaluator] summary LLM 실패: %s", e)
        return f"{winner_side} 진영이 우세로 평가되었습니다 (PRO {pro:.1f}% vs CON {con:.1f}%)."


def _safe_swing_narrative(turn: dict) -> str:
    try:
        msg = build_swing_user_msg(turn)
        return qwen_chat(SYSTEM_SWING_NARRATIVE, msg, max_new_tokens=220).strip()
    except Exception as e:
        logger.warning("[final_evaluator] swing narrative 실패 (turn %s): %s", turn.get("turn_index"), e)
        return turn.get("overall_summary", "") or "해설 생성 실패"


def _safe_coach(dim_label: str, best_row: Optional[dict], worst_row: Optional[dict]) -> DimensionFeedback:
    fallback = DimensionFeedback(
        praise=f"{dim_label} 지표에서 최고 점수 턴을 참조 바랍니다.",
        critique=f"{dim_label} 지표에서 최저 점수 턴을 재검토하세요.",
        suggestion=f"{dim_label} 영역의 보완 액션을 별도로 수립하세요.",
    )
    if best_row is None and worst_row is None:
        return fallback
    try:
        msg = build_coach_user_msg(dim_label, best_row or {}, worst_row or {})
        raw = qwen_chat(SYSTEM_COACH, msg, max_new_tokens=400, assistant_prefix='{"praise":').strip()
        obj = parse_json(raw) or {}
        return DimensionFeedback(
            praise=str(obj.get("praise", fallback.praise)).strip() or fallback.praise,
            critique=str(obj.get("critique", fallback.critique)).strip() or fallback.critique,
            suggestion=str(obj.get("suggestion", fallback.suggestion)).strip() or fallback.suggestion,
        )
    except Exception as e:
        logger.warning("[final_evaluator] coach LLM 실패 (%s): %s", dim_label, e)
        return fallback


# ── 메인 오케스트레이터 ────────────────────────────────────────────────────

def build_final_report(
    *,
    topic: str,
    debate_format: str,
    user_stance: str,
    analysis_memory: List[dict],
    speech_memory: List[dict],
    live_debate: dict,
) -> FinalReport:
    """분석 메모리로부터 최종 리포트 빌드.

    모든 LLM 호출은 실패 시 안전한 폴백으로 대체되어 리포트 자체는 항상 반환된다.
    """
    # 1. 진영별 평균
    stats_dict = compute_side_stats(analysis_memory)

    # 2. 승자 판정
    winner_side, pro_pct, con_pct, margin = decide_winner(live_debate, stats_dict)

    # 3. MVP
    mvp_dict = compute_mvp(analysis_memory, winner_side if winner_side != "DRAW" else None)

    # 4. swing 턴 후보 선정 (LLM 없이 규칙 기반)
    swing_candidates = select_swing_turns(analysis_memory, speech_memory, top_k=5)

    # 5. 지표별 최고/최저 턴 선정 (코치 피드백용)
    extremes = find_extreme_per_dimension(analysis_memory, speech_memory)

    # 6. LLM 호출을 병렬화 (요약 1 + swing N + coach 3)
    with ThreadPoolExecutor(max_workers=6) as pool:
        f_summary = pool.submit(_safe_summary, topic, winner_side, pro_pct, con_pct, stats_dict)
        f_swings = [pool.submit(_safe_swing_narrative, t) for t in swing_candidates]
        f_arg = pool.submit(_safe_coach, "논증", extremes["argument"]["best"], extremes["argument"]["worst"])
        f_evi = pool.submit(_safe_coach, "근거", extremes["evidence"]["best"], extremes["evidence"]["worst"])
        f_lang = pool.submit(_safe_coach, "언어", extremes["language"]["best"], extremes["language"]["worst"])

        summary_text = f_summary.result()
        swing_narratives = [f.result() for f in f_swings]
        coach_arg = f_arg.result()
        coach_evi = f_evi.result()
        coach_lang = f_lang.result()

    # 7. Pydantic 모델로 조립
    swing_turns: List[SwingTurn] = []
    for t, narrative in zip(swing_candidates, swing_narratives):
        swing_turns.append(SwingTurn(
            turn_index=t["turn_index"],
            speaker_id=t["speaker_id"],
            side=t["side"],
            phase=t["phase"],
            type=t["type"],
            weighted_score=t["weighted_score"],
            impact=t["impact"],
            speech_summary=t.get("speech_summary", ""),
            narrative=narrative,
        ))

    stats_obj: Dict[str, SideStats] = {
        side: SideStats(**stats_dict[side]) for side in ("PRO", "CON")
    }

    mvp_block: Optional[MVPBlock] = None
    if mvp_dict:
        mvp_reason = (
            f"{mvp_dict['side']} 진영에서 총 {mvp_dict['total_impact']}점의 누적 영향(Impact)을 "
            f"기록하며 3지표 평균 "
            f"논증 {mvp_dict['avg_argument']} / 근거 {mvp_dict['avg_evidence']} / "
            f"언어 {mvp_dict['avg_language']}로 기여도가 가장 높은 발언자입니다."
        )
        mvp_block = MVPBlock(
            speaker_id=mvp_dict["speaker_id"],
            side=mvp_dict["side"],
            total_impact=mvp_dict["total_impact"],
            avg_weighted=mvp_dict["avg_weighted"],
            avg_argument=mvp_dict["avg_argument"],
            avg_evidence=mvp_dict["avg_evidence"],
            avg_language=mvp_dict["avg_language"],
            reason=mvp_reason,
        )

    report = FinalReport(
        topic=topic,
        debate_format=debate_format,
        user_stance=user_stance if user_stance in ("PRO", "CON") else "PRO",
        total_turns=len(analysis_memory),
        winner=WinnerBlock(
            side=winner_side,
            pro_percent=round(pro_pct, 2),
            con_percent=round(con_pct, 2),
            margin=round(margin, 2),
            summary=summary_text,
        ),
        stats={"PRO": stats_obj["PRO"].model_dump(), "CON": stats_obj["CON"].model_dump()},
        swing_turns=swing_turns,
        mvp=mvp_block,
        coach_feedback=CoachFeedback(
            argument=coach_arg,
            evidence=coach_evi,
            language=coach_lang,
        ),
        generated_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    return report
