"""aggregator — LLM 없이 순수 계산으로 끝나는 집계 로직.

- 진영별 평균 점수
- 승자 판정
- MVP 선정
- 핵심 swing 턴 필터링
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional, Tuple


_DIMENSIONS = ("argument", "evidence", "language")


def _dim_score(row: dict, dim: str) -> float:
    """analysis_memory row에서 argument.score 같은 nested 값을 안전 추출."""
    d = row.get(dim) or {}
    if isinstance(d, dict):
        return float(d.get("score", 0.0))
    return 0.0


def compute_side_stats(analysis_memory: List[dict]) -> Dict[str, dict]:
    """PRO/CON 진영별 3지표 평균과 턴 수 계산."""
    buckets: Dict[str, Dict[str, List[float]]] = {
        "PRO": {d: [] for d in _DIMENSIONS},
        "CON": {d: [] for d in _DIMENSIONS},
    }
    weighted: Dict[str, List[float]] = {"PRO": [], "CON": []}

    for r in analysis_memory:
        side = r.get("speaker_stance")
        if side not in ("PRO", "CON"):
            continue
        for d in _DIMENSIONS:
            buckets[side][d].append(_dim_score(r, d))
        weighted[side].append(float(r.get("weighted_score", 0.0)))

    out: Dict[str, dict] = {}
    for side in ("PRO", "CON"):
        n = len(weighted[side])
        out[side] = {
            "argument": round(mean(buckets[side]["argument"]) if n else 0.0, 2),
            "evidence": round(mean(buckets[side]["evidence"]) if n else 0.0, 2),
            "language": round(mean(buckets[side]["language"]) if n else 0.0, 2),
            "weighted_mean": round(mean(weighted[side]) if n else 0.0, 2),
            "turn_count": n,
        }
    return out


def decide_winner(
    live_debate: dict,
    stats: Dict[str, dict],
    margin_draw: float = 2.0,
) -> Tuple[str, float, float, float]:
    """최종 우세도 기반 승자 결정.

    live_debate.{pro_percent, con_percent} 우선, 없으면 stats.weighted_mean으로 대체.

    Returns: (winner_side, pro_pct, con_pct, margin)
    """
    p = float(live_debate.get("pro_percent", 0.0) or 0.0)
    c = float(live_debate.get("con_percent", 0.0) or 0.0)
    if p == 0.0 and c == 0.0:
        # 폴백: weighted_mean 비교
        p = stats["PRO"]["weighted_mean"]
        c = stats["CON"]["weighted_mean"]
    margin = abs(p - c)
    if margin < margin_draw:
        return "DRAW", p, c, margin
    return ("PRO" if p > c else "CON"), p, c, margin


def compute_mvp(analysis_memory: List[dict], winner_side: Optional[str]) -> Optional[dict]:
    """MVP 선정.

    규칙:
      1. 각 speaker의 누적 weighted_score(=impact) 가장 큰 사람
      2. 동점 시 3지표 평균이 가장 높은 사람
      3. winner_side에 속한 speaker 우선 (DRAW/None이면 전체 대상)
    """
    per_speaker: Dict[str, dict] = defaultdict(lambda: {
        "speaker_id": "", "side": None,
        "turns": 0, "impact": 0.0,
        "arg_sum": 0.0, "evi_sum": 0.0, "lang_sum": 0.0,
    })
    for r in analysis_memory:
        sid = r.get("speaker_id")
        if not sid:
            continue
        b = per_speaker[sid]
        b["speaker_id"] = sid
        b["side"] = r.get("speaker_stance")
        b["turns"] += 1
        b["impact"] += float(r.get("weighted_score", 0.0))
        b["arg_sum"] += _dim_score(r, "argument")
        b["evi_sum"] += _dim_score(r, "evidence")
        b["lang_sum"] += _dim_score(r, "language")

    candidates = list(per_speaker.values())
    if winner_side in ("PRO", "CON"):
        same_side = [c for c in candidates if c["side"] == winner_side]
        if same_side:
            candidates = same_side
    if not candidates:
        return None

    # 1차: impact, 2차: 3지표 평균
    def key(c: dict) -> Tuple[float, float]:
        avg3 = (c["arg_sum"] + c["evi_sum"] + c["lang_sum"]) / (3 * max(c["turns"], 1))
        return (c["impact"], avg3)

    best = max(candidates, key=key)
    n = max(best["turns"], 1)
    return {
        "speaker_id": best["speaker_id"],
        "side": best["side"],
        "total_impact": round(best["impact"], 2),
        "avg_weighted": round(best["impact"] / n, 2),
        "avg_argument": round(best["arg_sum"] / n, 2),
        "avg_evidence": round(best["evi_sum"] / n, 2),
        "avg_language": round(best["lang_sum"] / n, 2),
        "turn_count": n,
    }


def _infer_logical_error(row: dict) -> bool:
    """overall_summary나 각 dim.summary에 논리적 오류 의심 키워드 있는지 검사."""
    keywords = ("논리적 오류", "오류", "모순", "성급한", "왜곡", "무관", "근거 없음", "근거 부족")
    txt = row.get("overall_summary", "") + " "
    for d in _DIMENSIONS:
        sub = row.get(d) or {}
        if isinstance(sub, dict):
            txt += sub.get("summary", "") + " "
    return any(k in txt for k in keywords)


def select_swing_turns(
    analysis_memory: List[dict],
    speech_memory: List[dict],
    top_k: int = 5,
) -> List[dict]:
    """핵심 턴 5~6개 필터링.

    카테고리:
      1. biggest_swing  — weighted_score 최상위
      2. best_rebuttal  — chained/free_rebuttal phase 중 최고 점수
      3. worst_turn     — weighted_score 최저
      4. logical_error  — 평가 요약에 오류 키워드가 있는 턴

    중복 turn_index는 더 상위 카테고리가 선점. 최대 top_k+1 반환.
    """
    if not analysis_memory:
        return []
    speech_by_turn = {s["turn_index"]: s.get("summary", "") for s in speech_memory}

    rebuttals = [r for r in analysis_memory if r.get("phase") in ("chained_rebuttal", "free_rebuttal")]

    picks: List[dict] = []
    used: set = set()

    def add(row: dict, ttype: str) -> None:
        ti = row.get("turn_index")
        if ti in used:
            return
        used.add(ti)
        picks.append({
            "turn_index": ti,
            "speaker_id": row.get("speaker_id", ""),
            "side": row.get("speaker_stance", "PRO"),
            "phase": row.get("phase", ""),
            "type": ttype,
            "weighted_score": round(float(row.get("weighted_score", 0.0)), 2),
            "impact": round(float(row.get("weighted_score", 0.0)), 2),
            "overall_summary": row.get("overall_summary", ""),
            "speech_summary": speech_by_turn.get(ti, ""),
        })

    # 1. biggest swing: top 2 by weighted_score
    for r in sorted(analysis_memory, key=lambda x: float(x.get("weighted_score", 0)), reverse=True)[:2]:
        add(r, "biggest_swing")

    # 2. best rebuttal: top 1 in rebuttal phases
    if rebuttals:
        best = max(rebuttals, key=lambda x: float(x.get("weighted_score", 0)))
        add(best, "best_rebuttal")

    # 3. worst turn: lowest weighted
    worst = min(analysis_memory, key=lambda x: float(x.get("weighted_score", 0)))
    add(worst, "worst_turn")

    # 4. logical_error
    for r in analysis_memory:
        if _infer_logical_error(r):
            add(r, "logical_error")
            if len(picks) >= top_k + 1:
                break

    # turn_index 오름차순 정렬 (타임라인 표시용)
    picks.sort(key=lambda x: x["turn_index"])
    return picks[: top_k + 1]


def find_extreme_per_dimension(analysis_memory: List[dict], speech_memory: List[dict]) -> Dict[str, dict]:
    """3지표 각각에 대해 최고/최저 점수 턴 선택.

    Returns: {"argument": {"best": row, "worst": row}, ...}
    """
    speech_by_turn = {s["turn_index"]: s.get("summary", "") for s in speech_memory}

    def enrich(row: dict) -> dict:
        return {**row, "_speech": speech_by_turn.get(row.get("turn_index"), "")}

    out = {}
    for dim in _DIMENSIONS:
        if not analysis_memory:
            out[dim] = {"best": None, "worst": None}
            continue
        best = max(analysis_memory, key=lambda x: _dim_score(x, dim))
        worst = min(analysis_memory, key=lambda x: _dim_score(x, dim))
        out[dim] = {"best": enrich(best), "worst": enrich(worst)}
    return out
