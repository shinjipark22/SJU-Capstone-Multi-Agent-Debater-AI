"""builder — 집계 + LLM 호출 + 최종 JSON 조립."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.live_analyzer.inference import qwen_chat

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
    SYSTEM_COACH_CRITIQUE,
    SYSTEM_COACH_PRAISE,
    SYSTEM_COACH_SUGGESTION,
    SYSTEM_SUMMARY,
    SYSTEM_SWING_NARRATIVE,
    build_coach_critique_msg,
    build_coach_praise_msg,
    build_coach_suggestion_msg,
    build_summary_user_msg,
    build_swing_user_msg,
)

logger = logging.getLogger(__name__)


# ── LLM 호출 래퍼 (장애 내성) ─────────────────────────────────────────────
# 모든 final_report LLM 텍스트 필드 (summary, swing_narrative, coach_feedback) 가
# 동일한 layered 방어 (cleaning + 외국어 검증 + 재시도) 를 거치도록 통합.
# _safe_summary, _safe_swing_narrative, _coach_field 모두 _safe_llm_text 호출.


def _safe_llm_text(
    system: str, msg: str, fallback: str, label: str,
    max_new_tokens: int = 200, max_retries: int = 3,
) -> str:
    """LLM 평문 호출 — cleaning (외국어 제거) + 검증 (외국어 거부) + 재시도.

    final_report 의 모든 LLM 텍스트 필드 공통 래퍼.
    입론 단계의 _postprocess_speech + validate_quality + 재시도 패턴 그대로 이식.
    """
    last_cleaned = ""
    for attempt in range(max_retries):
        try:
            raw = qwen_chat(system, msg, max_new_tokens=max_new_tokens)
            cleaned = _cleanup_text(raw, "")
            if not cleaned:
                logger.warning(
                    "[final_evaluator] %s cleanup 실패, 재시도 %d/%d",
                    label, attempt + 1, max_retries,
                )
                continue
            ok, reason = _coach_response_valid(cleaned)
            if ok:
                return cleaned
            last_cleaned = cleaned
            logger.warning(
                "[final_evaluator] %s 품질 미달 (%s), 재시도 %d/%d",
                label, reason, attempt + 1, max_retries,
            )
        except Exception as e:
            logger.warning(
                "[final_evaluator] %s 호출 실패 (%s), 재시도 %d/%d",
                label, e, attempt + 1, max_retries,
            )
    # 3회 다 실패 — 마지막 정리본이 길이만 부족했으면 그대로, 외국어 케이스면 fallback
    if last_cleaned and len(last_cleaned) >= 15 and not re.search(
        r'[一-鿿　-〿＀-｠。，]', last_cleaned,
    ):
        return last_cleaned
    return fallback


def _safe_summary(topic: str, winner_side: str, pro: float, con: float, stats: dict) -> str:
    msg = build_summary_user_msg(topic, winner_side, pro, con, stats)
    fallback = f"{winner_side} 진영이 우세로 평가되었습니다 (PRO {pro:.1f}% vs CON {con:.1f}%)."
    return _safe_llm_text(SYSTEM_SUMMARY, msg, fallback, "summary", max_new_tokens=200)


def _safe_swing_narrative(turn: dict) -> str:
    msg = build_swing_user_msg(turn)
    fallback = turn.get("overall_summary", "") or "해설 생성 실패"
    return _safe_llm_text(
        SYSTEM_SWING_NARRATIVE, msg, fallback,
        f"swing_{turn.get('turn_index','?')}", max_new_tokens=220,
    )


# 외국어 (중국어·일본어·러시아어·아랍어·타이어 등) 제거용 정규식.
# stage1_opening/nodes.py 의 _postprocess_speech 와 동일 패턴 — 한글 + 영어 + 숫자 +
# 일반 부호는 유지, 그 외 비-한국어 문자만 제거. coach_feedback 의 중국어 섞임 방지.
_FOREIGN_CHARS_RE = re.compile(
    r'[一-鿿㐀-䶿豈-﫿'      # CJK Unified Ideographs (한자 포함 중국어)
    r'぀-ゟ゠-ヿ'                    # 일본어 히라가나·가타카나
    r'Ѐ-ӿ'                                  # 러시아어 (키릴)
    r'฀-๿؀-ۿ'                    # 타이·아랍
    r'Ā-ɏḀ-ỿÀ-ÿ'       # 라틴 확장 (베트남어 포함)
    r'Ő-ſ'
    r'　-〿＀-｠'                    # 중국어 문장부호·전각
    r']+',
)


def _cleanup_text(raw: str, fallback: str) -> str:
    """LLM 평문 응답 정리 — 입론 단계 _postprocess_speech 와 동일한 외국어 처리.

    입론 단계의 정리 단계 그대로 적용:
      1. 깨진 유니코드 제거
      2. 외국 문자 (CJK·일본어·러시아·아랍·태국·라틴 확장 등) 제거
      3. 한글 없는 영어 전용 줄 제거 (혼종·CoT 흔적)
      4. 도구명·메타 흔적 제거
      5. 빈 줄·중복 공백·잘린 부호 정리
    """
    if not raw:
        return fallback
    t = raw.strip()
    # 접두어/코드블록 제거
    t = t.lstrip("#*-•·").strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1]
            if t.lower().startswith(("json", "text")):
                t = "\n".join(t.splitlines()[1:])
            t = t.strip()

    # 1. 깨진 유니코드 제거
    t = t.replace('�', '')

    # 2. 외국 문자 + 중국어 문장부호 제거 (입론 단계와 동일 정규식)
    t = _FOREIGN_CHARS_RE.sub('', t)

    # 3. 한글 없는 영어 전용 줄 제거
    lines = t.split('\n')
    t = '\n'.join(l for l in lines if not l.strip() or re.search(r'[가-힣]', l))

    # 4. 도구명·메타 흔적 제거 (입론 단계 _postprocess_speech 패턴)
    t = re.sub(r'search_web', '', t)
    t = re.sub(r'를 통해 확인되는 자료에 따르면[,.]?\s*', '', t)
    t = re.sub(r'를 통해 (?:최근|확인)', '', t)

    # 5. 한 줄로 합치고 잘린 부호 정리 (외국어 절 사라진 자리 ",,:;" 같은 잔여)
    cleaned = " ".join(line.strip() for line in t.splitlines() if line.strip())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'\s*[,.:;]\s*[,.:;]+', ', ', cleaned)  # 연속 부호 통합
    cleaned = re.sub(r'\s+([,.:;])', r'\1', cleaned)
    cleaned = re.sub(r'([가-힣])\s*[,.:;]\s*$', r'\1', cleaned)  # 끝 절단 부호 제거

    if len(cleaned) < 10:
        return fallback
    return cleaned


def _coach_response_valid(text: str) -> Tuple[bool, str]:
    """coach 응답이 한국어 위주이고 외국어 깨짐 없는지 검증.

    입론 단계 validate_quality 와 동일 패턴 (간소 버전).
    Returns:
        (ok, reason) — ok=False 면 재시도 트리거.
    """
    if not text or len(text) < 15:
        return False, f"길이 부족 ({len(text or '') }자)"
    # 외국어 깨진 문자 (CJK·중국어 문장부호)
    if re.search(r'[一-鿿　-〿＀-｠。，]', text):
        return False, "외국어 (CJK) 포함"
    # 한국어 비율
    korean = len(re.findall(r'[가-힣]', text))
    english = len(re.findall(r'[a-zA-Z]', text))
    if korean < 10:
        return False, f"한국어 부족 ({korean}자)"
    if korean + english > 0 and english / (korean + english) > 0.5:
        return False, f"영어 비율 과다 ({english / (korean + english):.0%})"
    # CoT 유출 (영어 사고 과정)
    cot_patterns = [
        r'\b(?:First|Second|Third|Next|Then|Finally),?\s+I\b',
        r'\bI (?:need|should|will|can|must)\b',
        r'\bLet me\b', r'\bIn order to\b',
    ]
    for p in cot_patterns:
        if re.search(p, text, re.IGNORECASE):
            return False, "CoT 유출"
    return True, "OK"


def _coach_field(system: str, msg: str, fallback: str, label: str) -> str:
    """단일 코치 필드(칭찬/지적/제안) — _safe_llm_text 사용."""
    return _safe_llm_text(system, msg, fallback, f"coach_{label}", max_new_tokens=180)


def _safe_coach(dim_label: str, best_row: Optional[dict], worst_row: Optional[dict]) -> DimensionFeedback:
    """3회 개별 LLM 호출(칭찬/지적/제안) → DimensionFeedback 조립.

    JSON 파싱 불필요 — 각 필드는 자연어 1~2문장만 반환.
    """
    fb_praise = f"{dim_label} 지표에서 최고 점수 턴을 참조 바랍니다."
    fb_critique = f"{dim_label} 지표에서 최저 점수 턴을 재검토하세요."
    fb_suggestion = f"{dim_label} 영역의 보완 액션을 별도로 수립하세요."

    if best_row is None and worst_row is None:
        return DimensionFeedback(praise=fb_praise, critique=fb_critique, suggestion=fb_suggestion)

    # 칭찬·지적은 병렬, 제안은 지적 완료 후 (critique 맥락 필요)
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_praise = pool.submit(
            _coach_field,
            SYSTEM_COACH_PRAISE,
            build_coach_praise_msg(dim_label, best_row or {}),
            fb_praise,
            f"{dim_label}/칭찬",
        ) if best_row else None
        f_critique = pool.submit(
            _coach_field,
            SYSTEM_COACH_CRITIQUE,
            build_coach_critique_msg(dim_label, worst_row or {}),
            fb_critique,
            f"{dim_label}/지적",
        ) if worst_row else None

        praise_text = f_praise.result() if f_praise else fb_praise
        critique_text = f_critique.result() if f_critique else fb_critique

    # 제안은 critique 결과를 입력으로 받아 생성 (일관성 ↑)
    if worst_row:
        suggestion_text = _coach_field(
            SYSTEM_COACH_SUGGESTION,
            build_coach_suggestion_msg(dim_label, worst_row, critique_text),
            fb_suggestion,
            f"{dim_label}/제안",
        )
    else:
        suggestion_text = fb_suggestion

    return DimensionFeedback(
        praise=praise_text,
        critique=critique_text,
        suggestion=suggestion_text,
    )


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
